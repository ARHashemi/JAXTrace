#!/usr/bin/env python3
"""
Test block-local fallback with 1,000 particles.

Quick validation test to ensure:
1. Block element lists are built correctly
2. Block IDs are tracked through time marching
3. Block-local fallback integrates correctly with GPU-fused RK4
4. No GPU-CPU transfer issues
"""

import os
import sys
import time
import numpy as np
import jax
from pathlib import Path

# Force CPU-GPU memory management
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# JAXTrace imports
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.forest.block_grid import create_regular_grid
from jaxtrace.gpu.forest.block_mapper import assign_elements_to_blocks
from jaxtrace.gpu.forest.padded_arrays import build_padded_block_arrays
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search import classify_blocks
from jaxtrace.gpu.search.hash_bucket import build_hash_bucket_arrays
from jaxtrace.gpu.search.initial_assignment import initial_search_batch
from jaxtrace.gpu.search.block_local_search import build_block_element_lists
from jaxtrace.gpu.particles import ParticleData
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.tracking.rk4_gpu_fused import rk4_step_gpu_fused_for_production_with_block_fallback
from jaxtrace.tracking.seeding import uniform_grid_seeds

# ============================================================================
# Configuration
# ============================================================================
MESH_PATH = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_120.pvtu"
GRID_SIZE = (8, 8, 4)  # 256 blocks
N_PARTICLES = 1000
N_TIMESTEPS = 100
DT = 1e-5
RK4_L1_HOP_COUNT = 3
PARTICLE_GRID_RESOLUTION = (10, 10, 10)  # Generates ~1000 particles
PARTICLE_BOUNDS_FRACTION = 0.05

print("=" * 80)
print("BLOCK-LOCAL FALLBACK TEST (1k particles)")
print("=" * 80)
print(f"Particles: {N_PARTICLES}")
print(f"Timesteps: {N_TIMESTEPS}")
print(f"dt: {DT} s")
print(f"L1 hops: {RK4_L1_HOP_COUNT}")
print()

# ============================================================================
# Load Mesh
# ============================================================================
print("Loading mesh...")
t0 = time.perf_counter()
node_positions, connectivity, velocity_field = load_mesh_from_pvtu(
    Path(MESH_PATH),
    field_name='Displacement'
)
print(f"✓ Mesh loaded ({time.perf_counter() - t0:.2f} s)")
print(f"  Nodes: {len(node_positions):,}")
print(f"  Elements: {len(connectivity):,}")
print()

# Ensure velocity is 3D and float32
if velocity_field.shape[1] == 2:
    velocity_field = np.column_stack([velocity_field, np.zeros(velocity_field.shape[0])])
velocity_field = velocity_field.astype(np.float32)

# ============================================================================
# Create Forest Structure
# ============================================================================
print("Creating forest structure...")
t0 = time.perf_counter()

# Compute bounding box
bbox = np.array([
    node_positions[:, 0].min(), node_positions[:, 0].max(),
    node_positions[:, 1].min(), node_positions[:, 1].max(),
    node_positions[:, 2].min(), node_positions[:, 2].max(),
], dtype=np.float32)

# Create blocks
blocks = create_regular_grid(bbox, GRID_SIZE)

# Assign elements
element_to_block, stats = assign_elements_to_blocks(
    node_positions, connectivity, bbox, GRID_SIZE, verbose=False
)

# Build element neighbors
element_neighbors = build_element_neighbors_array(connectivity, verbose=False)

# Build padded arrays
padded_arrays = build_padded_block_arrays(
    element_to_block, stats,
    node_positions=node_positions,
    connectivity=connectivity,
    element_neighbors=element_neighbors,
    verbose=False
)

print(f"✓ Forest created ({time.perf_counter() - t0:.2f} s)")
print(f"  Blocks: {len(blocks)}")
print(f"  Elements per block: {stats.min_elements} - {stats.max_elements}")
print()

# ============================================================================
# Build Block Element Lists
# ============================================================================
print("Building block element lists...")
t0 = time.perf_counter()

# Populate block.elements arrays
for block_id in range(len(blocks)):
    block_count = int(padded_arrays.block_sizes[block_id])
    block_elems = padded_arrays.block_elements[block_id, :block_count]
    blocks[block_id].elements = block_elems[block_elems >= 0]

block_lists = build_block_element_lists(blocks, len(blocks))
t_block_lists = time.perf_counter() - t0

print(f"✓ Block element lists built ({t_block_lists:.2f} s)")
print(f"  Total elements: {len(block_lists.all_elements):,}")
print(f"  Max elements per block: {block_lists.max_elements_per_block:,}")
print(f"  Memory (flat): {len(block_lists.all_elements) * 4 / 1024**2:.1f} MB")
print(f"  Memory (padded): {padded_arrays.memory_mb:.1f} MB")
print(f"  Savings: {padded_arrays.memory_mb / (len(block_lists.all_elements) * 4 / 1024**2):.1f}×")
print()

# ============================================================================
# Classify Blocks and Build Hash Buckets
# ============================================================================
print("Classifying blocks...")
classification = classify_blocks(padded_arrays, threshold=10000, verbose=False)
print(f"  Light blocks: {len(classification.light_blocks)}")
print(f"  Heavy blocks: {len(classification.heavy_blocks)}")

# Build hash buckets
hash_bucket_data = {}
if classification.heavy_blocks:
    element_centroids = np.mean(node_positions[connectivity], axis=1).astype(np.float32)
    for block_id in classification.heavy_blocks:
        block_elems = padded_arrays.block_elements[block_id]
        block_count = int(padded_arrays.block_sizes[block_id])
        elem_ids = block_elems[:block_count]
        elem_ids = elem_ids[elem_ids >= 0]
        if len(elem_ids) == 0:
            continue
        centroids = element_centroids[elem_ids]
        block_bounds = blocks[block_id].bounds
        hash_arrays = build_hash_bucket_arrays(
            block_id, elem_ids, centroids, block_bounds,
            target_bucket_size=200, morton_bits=10
        )
        hash_bucket_data[block_id] = hash_arrays

block_neighbors_26 = np.array([b.neighbors_26 for b in blocks], dtype=np.int32)
print()

# ============================================================================
# Generate Particles
# ============================================================================
print("Generating particles...")

# Compute domain bounds
domain_min = np.array([bbox[0], bbox[2], bbox[4]], dtype=np.float32)
domain_max = np.array([bbox[1], bbox[3], bbox[5]], dtype=np.float32)
domain_size = domain_max - domain_min

# Compute particle bounds (use simple fraction)
frac = PARTICLE_BOUNDS_FRACTION
par_bounds_min = domain_min + frac * domain_size
par_bounds_max = domain_max - frac * domain_size
par_bounds = [par_bounds_min, par_bounds_max]

# Generate uniform grid
particle_positions = uniform_grid_seeds(
    resolution=PARTICLE_GRID_RESOLUTION,
    bounds=par_bounds
)

# Limit to N_PARTICLES
particle_positions = particle_positions[:N_PARTICLES]
print(f"✓ Generated {len(particle_positions)} particles")
print()

# ============================================================================
# Initial Assignment
# ============================================================================
print("Performing initial assignment...")
t0 = time.perf_counter()
element_ids, block_ids, search_stats = initial_search_batch(
    particle_positions, bbox, GRID_SIZE,
    classification, padded_arrays, block_neighbors_26,
    hash_bucket_data, node_positions, connectivity,
    verbose=False
)
t_init = time.perf_counter() - t0

found_mask = element_ids >= 0
n_found = found_mask.sum()
print(f"✓ Initial assignment ({t_init:.2f} s)")
print(f"  Found: {n_found}/{len(particle_positions)} ({100*n_found/len(particle_positions):.1f}%)")
print()

# Create particle data
particle_data = ParticleData(
    positions=particle_positions[found_mask],
    velocities=np.zeros((n_found, 3), dtype=np.float32),
    element_ids=element_ids[found_mask],
    block_ids=block_ids[found_mask],
    active_mask=np.ones(n_found, dtype=bool)
)

print(f"✓ Particle data created")
print(f"  Active particles: {particle_data.n_active}")
print()

# ============================================================================
# Upload Mesh to GPU
# ============================================================================
print("Uploading mesh to GPU...")
t0 = time.perf_counter()
mesh_gpu = upload_mesh_to_gpu(
    connectivity,
    node_positions,
    element_neighbors,
    verbose=False
)
velocity_field_gpu = jax.device_put(velocity_field.astype(np.float32))
print(f"✓ Mesh uploaded ({time.perf_counter() - t0:.2f} s)")
print(f"  GPU memory: {mesh_gpu.memory_mb:.2f} MB")
print()

# ============================================================================
# JIT Warm-up
# ============================================================================
print("Warming up JIT compilation...")
t0 = time.perf_counter()
_, _ = rk4_step_gpu_fused_for_production_with_block_fallback(
    particle_data, velocity_field_gpu, DT, mesh_gpu,
    block_lists=block_lists, current_time=0.0, n_hops=RK4_L1_HOP_COUNT
)
print(f"✓ JIT warm-up complete ({time.perf_counter() - t0:.2f} s)")
print()

# ============================================================================
# Time Marching
# ============================================================================
print("=" * 80)
print("TIME MARCHING")
print("=" * 80)
print()

tracking_start = time.perf_counter()
step_times = []

for step in range(N_TIMESTEPS):
    step_start = time.perf_counter()

    # Perform RK4 time step with block fallback
    particle_data, rk4_stats = rk4_step_gpu_fused_for_production_with_block_fallback(
        particle_data, velocity_field_gpu, DT, mesh_gpu,
        block_lists=block_lists, current_time=step * DT, n_hops=RK4_L1_HOP_COUNT
    )

    step_time = time.perf_counter() - step_start
    step_times.append(step_time)

    # Progress reporting
    if (step + 1) % 10 == 0:
        avg_throughput = particle_data.n_active / np.mean(step_times[-10:])
        print(f"Step {step+1:>3}/{N_TIMESTEPS} | "
              f"Active: {particle_data.n_active:>4} | "
              f"Throughput: {avg_throughput:>7.1f} p/s")

tracking_elapsed = time.perf_counter() - tracking_start

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print()
print(f"Total time: {tracking_elapsed:.2f} s")
print(f"Time per step: {np.mean(step_times):.4f} s ± {np.std(step_times):.4f} s")
print(f"Mean throughput: {particle_data.n_active / np.mean(step_times):.1f} p/s")
print(f"Final active particles: {particle_data.n_active}/{n_found} ({100*particle_data.n_active/n_found:.1f}%)")
print()

if particle_data.n_active == n_found:
    print("✅ SUCCESS: All particles retained!")
else:
    retention = 100 * particle_data.n_active / n_found
    print(f"⚠️  Particle retention: {retention:.1f}%")

print()
print("✓ Block-local fallback test complete")
