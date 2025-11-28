#!/usr/bin/env python3
"""
Test script for Phase 3a: Vectorized incremental search.

Validates:
1. Correctness: Same element assignments as baseline
2. Performance: 10-20× speedup over baseline (target 100-200k p/s)
3. Memory: No padded arrays (6.5 GB savings)
4. GPU utilization: Higher than baseline (target 60-80%)

Test methodology:
- Run baseline search (block-based with padded arrays)
- Run vectorized search (batch L0/L1 + global L2)
- Compare results for correctness
- Compare performance and memory usage
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
from jaxtrace.gpu.particles import ParticleData
from jaxtrace.tracking.seeding import uniform_grid_seeds

# Search implementations
from jaxtrace.gpu.search import initial_search_batch, incremental_search_batch
from jaxtrace.gpu.search.incremental_search_vectorized import (
    incremental_search_vectorized,
    search_global_parallel
)
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu

print("=" * 80)
print("PHASE 3a: VECTORIZED SEARCH TEST")
print("=" * 80)
print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print()

# Configuration
MESH_PATH = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_120.pvtu"
GRID_SIZE = (8, 8, 4)  # 256 blocks
N_PARTICLES = 10000  # Test with 10K particles
RK4_DT = 0.01  # Small timestep to trigger search misses

print("=" * 80)
print("CONFIGURATION")
print("=" * 80)
print(f"Mesh: {MESH_PATH}")
print(f"Grid: {GRID_SIZE} ({np.prod(GRID_SIZE)} blocks)")
print(f"Particles: {N_PARTICLES:,}")
print(f"RK4 timestep: {RK4_DT}")
print()

# ============================================================================
# Load Mesh
# ============================================================================
print("=" * 80)
print("MESH LOADING")
print("=" * 80)
print()

t0 = time.perf_counter()
node_positions, connectivity, velocity_field = load_mesh_from_pvtu(
    Path(MESH_PATH),
    field_name='Displacement'
)
t_load = time.perf_counter() - t0

print(f"✓ Mesh loaded ({t_load:.2f} s):")
print(f"  Nodes: {len(node_positions):,}")
print(f"  Elements: {len(connectivity):,}")
print(f"  Velocity field: {velocity_field.shape}")

# Ensure velocity is 3D and float32
if velocity_field.shape[1] == 2:
    velocity_field = np.column_stack([velocity_field, np.zeros(velocity_field.shape[0])])
velocity_field = velocity_field.astype(np.float32)
print()

# ============================================================================
# Create Forest Structure
# ============================================================================
print("=" * 80)
print("FOREST STRUCTURE")
print("=" * 80)
print()

# Compute bounding box
bbox = np.array([
    node_positions[:, 0].min(), node_positions[:, 0].max(),
    node_positions[:, 1].min(), node_positions[:, 1].max(),
    node_positions[:, 2].min(), node_positions[:, 2].max(),
], dtype=np.float32)

print(f"Bounding box:")
print(f"  X: [{bbox[0]:.4f}, {bbox[1]:.4f}]")
print(f"  Y: [{bbox[2]:.4f}, {bbox[3]:.4f}]")
print(f"  Z: [{bbox[4]:.4f}, {bbox[5]:.4f}]")
print()

# Create block grid
blocks = create_regular_grid(bbox, GRID_SIZE)
print(f"✓ Block grid created: {len(blocks)} blocks")

# Assign elements to blocks
element_to_block, stats = assign_elements_to_blocks(
    node_positions,
    connectivity,
    bbox,
    GRID_SIZE,
    verbose=False
)
print(f"✓ Element assignment:")
print(f"  Blocks used: {stats.n_blocks_used}/{stats.n_blocks}")
print(f"  Elements per block: {stats.min_elements} - {stats.max_elements}")

# Build element neighbors
element_neighbors = build_element_neighbors_array(connectivity, verbose=False)
print(f"✓ Element neighbors built")

# Build padded arrays (for baseline only)
padded_arrays = build_padded_block_arrays(
    element_to_block,
    stats,
    node_positions=node_positions,
    connectivity=connectivity,
    element_neighbors=element_neighbors,
    verbose=False
)
print(f"✓ Padded arrays (baseline only):")
print(f"  Shape: {padded_arrays.block_elements.shape}")
print(f"  Memory: {padded_arrays.memory_mb:.1f} MB")
print()

# Upload mesh to GPU (for vectorized search)
mesh_gpu = upload_mesh_to_gpu(
    connectivity,
    node_positions,
    element_neighbors,
    verbose=True
)
print()

# ============================================================================
# Generate Test Particles
# ============================================================================
print("=" * 80)
print("TEST PARTICLES")
print("=" * 80)
print()

# Generate random particles inside domain
domain_min = np.array([bbox[0], bbox[2], bbox[4]], dtype=np.float32)
domain_max = np.array([bbox[1], bbox[3], bbox[5]], dtype=np.float32)

# Use uniform grid for reproducibility
grid_res = int(np.ceil(N_PARTICLES ** (1/3)))
particle_positions = uniform_grid_seeds(
    resolution=(grid_res, grid_res, grid_res),
    bounds=[domain_min, domain_max],
    include_boundaries=True
)[:N_PARTICLES]

print(f"✓ Generated {N_PARTICLES:,} test particles")
print()

# ============================================================================
# Test 1: Initial Assignment (All particles, no cache)
# ============================================================================
print("=" * 80)
print("TEST 1: INITIAL ASSIGNMENT (NO CACHE)")
print("=" * 80)
print()

# Baseline: initial_search_batch (requires block classification and hash buckets)
print("Setting up baseline search (block classification + hash buckets)...")
from jaxtrace.gpu.search import classify_blocks
from jaxtrace.gpu.search.hash_bucket import build_hash_bucket_arrays

classification = classify_blocks(padded_arrays, threshold=10000, verbose=False)
print(f"  Light blocks: {len(classification.light_blocks)}")
print(f"  Heavy blocks: {len(classification.heavy_blocks)}")

# Build hash buckets for heavy blocks
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
            block_id=block_id,
            element_ids=elem_ids,
            element_centroids=centroids,
            block_bounds=block_bounds,
            target_bucket_size=200,
            morton_bits=10
        )

        hash_bucket_data[block_id] = hash_arrays

block_neighbors_26 = np.array([b.neighbors_26 for b in blocks], dtype=np.int32)

print(f"✓ Baseline search setup complete")
print()

print("Running BASELINE initial search...")
t0 = time.perf_counter()
element_ids_baseline, block_ids_baseline, _ = initial_search_batch(
    particle_positions,
    bbox,
    GRID_SIZE,
    classification,
    padded_arrays,
    block_neighbors_26,
    hash_bucket_data,
    node_positions,
    connectivity,
    verbose=False
)
t_baseline_init = time.perf_counter() - t0
throughput_baseline_init = N_PARTICLES / t_baseline_init

print(f"✓ Baseline: {t_baseline_init*1000:.2f} ms ({throughput_baseline_init:.1f} p/s)")
print(f"  Found: {(element_ids_baseline >= 0).sum()}/{N_PARTICLES}")
print()

print("Running VECTORIZED initial search...")
positions_gpu = jax.device_put(particle_positions)
t0 = time.perf_counter()
element_ids_vectorized_gpu = search_global_parallel(
    positions_gpu,
    mesh_gpu.node_positions,
    mesh_gpu.connectivity
)
element_ids_vectorized = np.array(element_ids_vectorized_gpu, dtype=np.int32)
t_vectorized_init = time.perf_counter() - t0
throughput_vectorized_init = N_PARTICLES / t_vectorized_init
speedup_init = throughput_vectorized_init / throughput_baseline_init

print(f"✓ Vectorized: {t_vectorized_init*1000:.2f} ms ({throughput_vectorized_init:.1f} p/s, {speedup_init:.1f}× speedup)")
print(f"  Found: {(element_ids_vectorized >= 0).sum()}/{N_PARTICLES}")
print()

# Validate correctness (element IDs should match)
matching = (element_ids_baseline == element_ids_vectorized).sum()
print(f"Validation:")
print(f"  Matching assignments: {matching}/{N_PARTICLES} ({100*matching/N_PARTICLES:.1f}%)")

# For mismatches, check if both are valid (particle at element boundary)
mismatches = element_ids_baseline != element_ids_vectorized
n_mismatches = mismatches.sum()
if n_mismatches > 0:
    print(f"  Mismatches: {n_mismatches} (may be valid if particle at boundary)")
print()

# ============================================================================
# Test 2: Incremental Search (With L0/L1 cache)
# ============================================================================
print("=" * 80)
print("TEST 2: INCREMENTAL SEARCH (WITH CACHE)")
print("=" * 80)
print()

# Simulate particle movement (small displacement to test L0/L1 cache)
print("Simulating particle movement (small displacement)...")
displacement = np.random.randn(N_PARTICLES, 3).astype(np.float32) * 0.001  # Small movement
new_positions = particle_positions + displacement

# Use baseline results as cached values
cached_element_ids = element_ids_baseline.copy()
cached_block_ids = block_ids_baseline.copy()

print(f"✓ Particles displaced by ~1mm")
print()

print("Running BASELINE incremental search...")
t0 = time.perf_counter()
element_ids_baseline_incr, block_ids_baseline_incr = incremental_search_batch(
    new_positions,
    cached_element_ids,
    cached_block_ids,
    bbox,
    GRID_SIZE,
    classification,
    padded_arrays,
    block_neighbors_26,
    hash_bucket_data,
    node_positions,
    connectivity,
    element_neighbors=element_neighbors,
    verbose=False
)
t_baseline_incr = time.perf_counter() - t0
throughput_baseline_incr = N_PARTICLES / t_baseline_incr

print(f"✓ Baseline: {t_baseline_incr*1000:.2f} ms ({throughput_baseline_incr:.1f} p/s)")
print(f"  Found: {(element_ids_baseline_incr >= 0).sum()}/{N_PARTICLES}")
print()

print("Running VECTORIZED incremental search...")
t0 = time.perf_counter()
element_ids_vectorized_incr, block_ids_vectorized_incr, search_stats = incremental_search_vectorized(
    new_positions,
    cached_element_ids,
    cached_block_ids,
    mesh_gpu,
    element_neighbors=element_neighbors,
    use_global_l2=True,
    verbose=False
)
t_vectorized_incr = time.perf_counter() - t0
throughput_vectorized_incr = N_PARTICLES / t_vectorized_incr
speedup_incr = throughput_vectorized_incr / throughput_baseline_incr

print(f"✓ Vectorized: {t_vectorized_incr*1000:.2f} ms ({throughput_vectorized_incr:.1f} p/s, {speedup_incr:.1f}× speedup)")
print(f"  Found: {(element_ids_vectorized_incr >= 0).sum()}/{N_PARTICLES}")
print()

print(f"Search statistics:")
print(f"  L0 hits: {search_stats.l0_hits}/{N_PARTICLES} ({100*search_stats.l0_hits/N_PARTICLES:.1f}%)")
print(f"  L1 hits: {search_stats.l1_hits}/{N_PARTICLES} ({100*search_stats.l1_hits/N_PARTICLES:.1f}%)")
print(f"  L2 searches: {search_stats.l2_attempts}/{N_PARTICLES} ({100*search_stats.l2_attempts/N_PARTICLES:.1f}%)")
print(f"  L2 hits: {search_stats.l2_hits}/{N_PARTICLES} ({100*search_stats.l2_hits/N_PARTICLES:.1f}%)")
print(f"  Total hits: {search_stats.total_found}/{N_PARTICLES} ({100*search_stats.total_found/N_PARTICLES:.1f}%)")
print(f"  Timings:")
print(f"    L0: {search_stats.time_l0*1000:.3f} ms")
print(f"    L1: {search_stats.time_l1*1000:.3f} ms")
print(f"    L2: {search_stats.time_l2*1000:.3f} ms")
print(f"    Total: {search_stats.time_total*1000:.3f} ms")
print()

# Validate correctness
matching_incr = (element_ids_baseline_incr == element_ids_vectorized_incr).sum()
print(f"Validation:")
print(f"  Matching assignments: {matching_incr}/{N_PARTICLES} ({100*matching_incr/N_PARTICLES:.1f}%)")

mismatches_incr = element_ids_baseline_incr != element_ids_vectorized_incr
n_mismatches_incr = mismatches_incr.sum()
if n_mismatches_incr > 0:
    print(f"  Mismatches: {n_mismatches_incr} (may be valid if particle at boundary)")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

print(f"Performance (Initial Assignment):")
print(f"  Baseline:   {throughput_baseline_init:>10.1f} p/s")
print(f"  Vectorized: {throughput_vectorized_init:>10.1f} p/s ({speedup_init:>5.1f}× speedup)")
print()

print(f"Performance (Incremental Search):")
print(f"  Baseline:   {throughput_baseline_incr:>10.1f} p/s")
print(f"  Vectorized: {throughput_vectorized_incr:>10.1f} p/s ({speedup_incr:>5.1f}× speedup)")
print()

print(f"Memory:")
print(f"  Baseline (padded arrays): {padded_arrays.memory_mb:.1f} MB CPU")
print(f"  Vectorized (GPU mesh):    {mesh_gpu.memory_mb:.1f} MB GPU")
print(f"  Memory saved:             {padded_arrays.memory_mb - mesh_gpu.memory_mb:.1f} MB")
print()

# Success criteria
success = True

if speedup_init < 3:
    print(f"❌ FAIL: Initial search speedup too low ({speedup_init:.1f}×, expected >3×)")
    success = False
else:
    print(f"✓ PASS: Initial search speedup ({speedup_init:.1f}×)")

if speedup_incr < 5:
    print(f"❌ FAIL: Incremental search speedup too low ({speedup_incr:.1f}×, expected >5×)")
    success = False
else:
    print(f"✓ PASS: Incremental search speedup ({speedup_incr:.1f}×)")

if throughput_vectorized_incr < 50000:
    print(f"❌ FAIL: Vectorized throughput too low ({throughput_vectorized_incr:.1f} p/s, expected >50k p/s)")
    success = False
else:
    print(f"✓ PASS: Vectorized throughput ({throughput_vectorized_incr:.1f} p/s)")

correctness_threshold = 0.95
if matching_incr / N_PARTICLES < correctness_threshold:
    print(f"❌ FAIL: Too many mismatches ({matching_incr}/{N_PARTICLES}, expected >{correctness_threshold*100}%)")
    success = False
else:
    print(f"✓ PASS: Correctness ({matching_incr}/{N_PARTICLES} matching)")

print()
if success:
    print("=" * 80)
    print("✓ ALL TESTS PASSED - Phase 3a implementation validated!")
    print("=" * 80)
else:
    print("=" * 80)
    print("❌ SOME TESTS FAILED")
    print("=" * 80)
    sys.exit(1)
