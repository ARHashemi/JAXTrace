#!/usr/bin/env python3
"""
Test script for global GPU interpolation architecture.

Tests both Phase 1 and Phase 2 implementations:
- Phase 1: Persistent mesh + block-by-block particles
- Phase 2: Persistent mesh + single batch

Validates:
1. Correctness: Same velocities as baseline
2. Performance: 20-60× speedup over baseline
3. Memory: GPU memory usage within expected bounds
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

# Interpolation implementations
from jaxtrace.gpu.tracking.velocity_interpolation_blockwise import create_blockwise_interpolator
from jaxtrace.gpu.tracking.velocity_interpolation_global import create_global_interpolator
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu, estimate_mesh_memory_mb

print("=" * 80)
print("GLOBAL GPU INTERPOLATION TEST")
print("=" * 80)
print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print()

# Configuration
MESH_PATH = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_120.pvtu"
GRID_SIZE = (8, 8, 4)  # 256 blocks
N_PARTICLES = 10000  # Test with 10K particles

print("=" * 80)
print("CONFIGURATION")
print("=" * 80)
print(f"Mesh: {MESH_PATH}")
print(f"Grid: {GRID_SIZE} ({np.prod(GRID_SIZE)} blocks)")
print(f"Particles: {N_PARTICLES:,}")
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
# Create Forest Structure (for baseline)
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

# Build padded arrays
padded_arrays = build_padded_block_arrays(
    element_to_block,
    stats,
    node_positions=node_positions,
    connectivity=connectivity,
    element_neighbors=element_neighbors,
    verbose=False
)
print(f"✓ Padded arrays:")
print(f"  Shape: {padded_arrays.block_elements.shape}")
print(f"  Memory: {padded_arrays.memory_mb:.1f} MB")
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

# Assign random element IDs and block IDs (for testing only)
element_ids = np.random.randint(0, len(connectivity), size=N_PARTICLES, dtype=np.int32)
block_ids = element_to_block[element_ids]

particle_data = ParticleData(
    positions=particle_positions,
    velocities=np.zeros((N_PARTICLES, 3), dtype=np.float32),
    element_ids=element_ids,
    block_ids=block_ids,
    active_mask=np.ones(N_PARTICLES, dtype=bool)
)

print(f"✓ Generated {N_PARTICLES:,} test particles")
print()

# ============================================================================
# Setup Baseline Interpolator
# ============================================================================
print("=" * 80)
print("BASELINE INTERPOLATOR SETUP")
print("=" * 80)
print()

# Prepare velocity field for block-wise interpolation
velocity_field_all_blocks = np.tile(velocity_field, (len(blocks), 1, 1)).astype(np.float32)
print(f"Velocity field shape: {velocity_field_all_blocks.shape}")

# Upload mesh to GPU
connectivity_gpu = jax.device_put(connectivity)
node_positions_gpu = jax.device_put(node_positions)

# Create baseline interpolator
interpolator_baseline = create_blockwise_interpolator(
    velocity_field_all_blocks,
    padded_arrays,
    connectivity_gpu,
    node_positions_gpu
)

print(f"✓ Baseline interpolator created")
print()

# ============================================================================
# Setup Global Interpolators
# ============================================================================
print("=" * 80)
print("GLOBAL INTERPOLATOR SETUP")
print("=" * 80)
print()

# Estimate mesh memory
mesh_memory_mb = estimate_mesh_memory_mb(len(connectivity), len(node_positions))
print(f"Estimated mesh memory: {mesh_memory_mb:.2f} MB")

# Upload mesh to GPU once
mesh_gpu = upload_mesh_to_gpu(
    connectivity,
    node_positions,
    element_neighbors,
    verbose=True
)

print()

# Create Phase 1 interpolator
interpolator_phase1 = create_global_interpolator(
    velocity_field,
    mesh_gpu,
    padded_arrays=padded_arrays,
    phase=1
)
print(f"✓ Phase 1 interpolator created (persistent mesh + block-by-block)")

# Create Phase 2 interpolator
interpolator_phase2 = create_global_interpolator(
    velocity_field,
    mesh_gpu,
    phase=2
)
print(f"✓ Phase 2 interpolator created (persistent mesh + single batch)")
print()

# ============================================================================
# Benchmark & Validation
# ============================================================================
print("=" * 80)
print("BENCHMARK & VALIDATION")
print("=" * 80)
print()

# Warm-up JIT compilation
print("Warming up JIT compilation...")
_ = interpolator_baseline(particle_data, 0.0)
_ = interpolator_phase1(particle_data, 0.0)
_ = interpolator_phase2(particle_data, 0.0)
print("✓ JIT warm-up complete")
print()

# Number of iterations for benchmark
N_ITERS = 10

# ===== Baseline =====
print(f"Benchmarking BASELINE (block-wise)...")
times_baseline = []
for i in range(N_ITERS):
    t0 = time.perf_counter()
    vels_baseline = interpolator_baseline(particle_data, 0.0)
    times_baseline.append(time.perf_counter() - t0)

mean_time_baseline = np.mean(times_baseline)
throughput_baseline = N_PARTICLES / mean_time_baseline

print(f"✓ Baseline: {mean_time_baseline*1000:.2f} ms/iter ({throughput_baseline:.1f} p/s)")
print()

# ===== Phase 1 =====
print(f"Benchmarking PHASE 1 (persistent mesh + block-by-block)...")
times_phase1 = []
for i in range(N_ITERS):
    t0 = time.perf_counter()
    vels_phase1 = interpolator_phase1(particle_data, 0.0)
    times_phase1.append(time.perf_counter() - t0)

mean_time_phase1 = np.mean(times_phase1)
throughput_phase1 = N_PARTICLES / mean_time_phase1
speedup_phase1 = throughput_phase1 / throughput_baseline

print(f"✓ Phase 1: {mean_time_phase1*1000:.2f} ms/iter ({throughput_phase1:.1f} p/s, {speedup_phase1:.1f}× speedup)")

# Validate correctness
diff_phase1 = np.linalg.norm(vels_phase1 - vels_baseline) / np.linalg.norm(vels_baseline)
print(f"  Relative error vs baseline: {diff_phase1:.2e}")
print()

# ===== Phase 2 =====
print(f"Benchmarking PHASE 2 (persistent mesh + single batch)...")
times_phase2 = []
for i in range(N_ITERS):
    t0 = time.perf_counter()
    vels_phase2 = interpolator_phase2(particle_data, 0.0)
    times_phase2.append(time.perf_counter() - t0)

mean_time_phase2 = np.mean(times_phase2)
throughput_phase2 = N_PARTICLES / mean_time_phase2
speedup_phase2 = throughput_phase2 / throughput_baseline

print(f"✓ Phase 2: {mean_time_phase2*1000:.2f} ms/iter ({throughput_phase2:.1f} p/s, {speedup_phase2:.1f}× speedup)")

# Validate correctness
diff_phase2 = np.linalg.norm(vels_phase2 - vels_baseline) / np.linalg.norm(vels_baseline)
print(f"  Relative error vs baseline: {diff_phase2:.2e}")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

print(f"Performance:")
print(f"  Baseline:  {throughput_baseline:>10.1f} p/s")
print(f"  Phase 1:   {throughput_phase1:>10.1f} p/s ({speedup_phase1:>5.1f}× speedup)")
print(f"  Phase 2:   {throughput_phase2:>10.1f} p/s ({speedup_phase2:>5.1f}× speedup)")
print()

print(f"Validation (relative error):")
print(f"  Phase 1 vs Baseline: {diff_phase1:.2e}")
print(f"  Phase 2 vs Baseline: {diff_phase2:.2e}")
print()

print(f"Memory:")
print(f"  Baseline (padded arrays):  {padded_arrays.memory_mb:.1f} MB CPU")
print(f"  Global mesh (GPU):         {mesh_gpu.memory_mb:.1f} MB GPU")
print(f"  Memory reduction:          {padded_arrays.memory_mb / mesh_gpu.memory_mb:.1f}×")
print()

# Success criteria
success = True
if speedup_phase1 < 5:
    print(f"❌ FAIL: Phase 1 speedup too low ({speedup_phase1:.1f}×, expected >5×)")
    success = False
else:
    print(f"✓ PASS: Phase 1 speedup ({speedup_phase1:.1f}×)")

if speedup_phase2 < 10:
    print(f"❌ FAIL: Phase 2 speedup too low ({speedup_phase2:.1f}×, expected >10×)")
    success = False
else:
    print(f"✓ PASS: Phase 2 speedup ({speedup_phase2:.1f}×)")

if diff_phase1 > 1e-5:
    print(f"❌ FAIL: Phase 1 correctness error too high ({diff_phase1:.2e})")
    success = False
else:
    print(f"✓ PASS: Phase 1 correctness")

if diff_phase2 > 1e-5:
    print(f"❌ FAIL: Phase 2 correctness error too high ({diff_phase2:.2e})")
    success = False
else:
    print(f"✓ PASS: Phase 2 correctness")

print()
if success:
    print("=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)
else:
    print("=" * 80)
    print("❌ SOME TESTS FAILED")
    print("=" * 80)
    sys.exit(1)
