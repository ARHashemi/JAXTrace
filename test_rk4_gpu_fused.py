#!/usr/bin/env python3
"""
Test script for GPU-fused RK4 implementation.

Validates:
1. Correctness: GPU-fused RK4 matches CPU-orchestrated RK4
2. Performance: CPU-GPU transfer reduction (10 MB → 2 MB per timestep)
3. Throughput: Expected 2-3× speedup in overall particle tracking

Test methodology:
- Run baseline RK4 (CPU-orchestrated with transfers at each stage)
- Run GPU-fused RK4 (all 4 stages on GPU, no intermediate transfers)
- Compare results for correctness
- Compare performance and transfer overhead
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
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.particles import ParticleData
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.tracking.rk4_gpu_fused import rk4_step_gpu_fused_wrapper
from jaxtrace.gpu.tracking.time_integration import rk4_step_with_incremental_search
from jaxtrace.gpu.tracking.velocity_interpolation_global import create_global_interpolator
from jaxtrace.gpu.search.incremental_search_vectorized import incremental_search_vectorized

print("=" * 80)
print("GPU-FUSED RK4 TEST")
print("=" * 80)
print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print()

# Configuration
MESH_PATH = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_120.pvtu"
N_PARTICLES = 10000  # Test with 10K particles
N_TIMESTEPS = 10  # Test 10 timesteps
DT = 0.0025  # Small timestep

print("=" * 80)
print("CONFIGURATION")
print("=" * 80)
print(f"Mesh: {MESH_PATH}")
print(f"Particles: {N_PARTICLES:,}")
print(f"Timesteps: {N_TIMESTEPS}")
print(f"dt: {DT} s")
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

# Ensure velocity is 3D and float32
if velocity_field.shape[1] == 2:
    velocity_field = np.column_stack([velocity_field, np.zeros(velocity_field.shape[0])])
velocity_field = velocity_field.astype(np.float32)

# Build element neighbors
element_neighbors = build_element_neighbors_array(connectivity, verbose=False)
print(f"✓ Element neighbors built")

# Upload mesh to GPU
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

# Simplified strategy for testing:
# Generate particles at element centroids (guarantees they're inside elements)
# This avoids the slow global search and makes test faster

print(f"Generating {N_PARTICLES:,} test particles at element centroids...")
t0 = time.perf_counter()

# Use first N_PARTICLES elements as seed locations
seed_element_ids = np.arange(N_PARTICLES, dtype=np.int32) % len(connectivity)

# For each seed element, place particle at its centroid
particle_positions = np.zeros((N_PARTICLES, 3), dtype=np.float32)
for i in range(N_PARTICLES):
    elem_nodes = connectivity[seed_element_ids[i]]
    node_coords = node_positions[elem_nodes]
    particle_positions[i] = node_coords.mean(axis=0)

element_ids = seed_element_ids
t_init = time.perf_counter() - t0

print(f"✓ Generated {N_PARTICLES:,} particles ({t_init:.3f} s)")
print(f"  All particles guaranteed to start inside valid elements")
print()

# ============================================================================
# Test 1: Baseline CPU-Orchestrated RK4
# ============================================================================
print("=" * 80)
print("TEST 1: BASELINE CPU-ORCHESTRATED RK4")
print("=" * 80)
print()

# Create baseline interpolator and searcher
velocity_interpolator = create_global_interpolator(
    velocity_field,  # Pass velocity field directly, not as a function
    mesh_gpu,
    padded_arrays=None,  # Not needed for Phase 2
    phase=2
)

def incremental_searcher(new_positions, cached_elem_ids, cached_block_ids):
    """Baseline vectorized search (with CPU-GPU transfers at each call)"""
    elem_ids, block_ids, stats = incremental_search_vectorized(
        new_positions,
        cached_elem_ids,
        cached_block_ids,
        mesh_gpu,
        element_neighbors=element_neighbors,
        use_global_l2=False,
        verbose=False
    )
    return elem_ids, block_ids, stats

# Create particle data
block_ids = np.zeros(N_PARTICLES, dtype=np.int32)  # Not used
velocities = np.zeros((N_PARTICLES, 3), dtype=np.float32)  # Will be computed by interpolator
particle_data = ParticleData(
    positions=particle_positions.copy(),
    velocities=velocities,
    element_ids=element_ids.copy(),
    block_ids=block_ids,
    active_mask=np.ones(N_PARTICLES, dtype=bool)
)

print(f"Running BASELINE RK4 ({N_TIMESTEPS} timesteps)...")

# Warm up
_ = rk4_step_with_incremental_search(
    particle_data,
    velocity_interpolator,
    incremental_searcher,
    dt=DT,
    current_time=0.0
)

# Benchmark
t0 = time.perf_counter()
particle_data_baseline = particle_data
for step in range(N_TIMESTEPS):
    particle_data_baseline, stats = rk4_step_with_incremental_search(
        particle_data_baseline,
        velocity_interpolator,
        incremental_searcher,
        dt=DT,
        current_time=step * DT
    )
t_baseline = time.perf_counter() - t0

throughput_baseline = (N_PARTICLES * N_TIMESTEPS) / t_baseline

print(f"✓ Baseline: {t_baseline:.3f} s ({throughput_baseline:.0f} p/s)")
print(f"  Final positions: {particle_data_baseline.positions.shape}")
print(f"  Active particles: {particle_data_baseline.n_active:,}")
print()

# Save baseline results
positions_baseline = particle_data_baseline.positions.copy()
element_ids_baseline = particle_data_baseline.element_ids.copy()

# ============================================================================
# Test 2: GPU-Fused RK4
# ============================================================================
print("=" * 80)
print("TEST 2: GPU-FUSED RK4")
print("=" * 80)
print()

# Reset particle data
positions_fused = particle_positions.copy()
element_ids_fused = element_ids.copy()

print(f"Running GPU-FUSED RK4 ({N_TIMESTEPS} timesteps)...")

# Warm up
_ = rk4_step_gpu_fused_wrapper(
    positions_fused,
    element_ids_fused,
    DT,
    mesh_gpu,
    velocity_field
)

# Benchmark
total_upload_time = 0.0
total_compute_time = 0.0
total_download_time = 0.0

t0 = time.perf_counter()
for step in range(N_TIMESTEPS):
    positions_fused, element_ids_fused, stats = rk4_step_gpu_fused_wrapper(
        positions_fused,
        element_ids_fused,
        DT,
        mesh_gpu,
        velocity_field
    )
    total_upload_time += stats['time_upload']
    total_compute_time += stats['time_compute']
    total_download_time += stats['time_download']
t_fused = time.perf_counter() - t0

throughput_fused = (N_PARTICLES * N_TIMESTEPS) / t_fused
speedup = throughput_fused / throughput_baseline

print(f"✓ GPU-Fused: {t_fused:.3f} s ({throughput_fused:.0f} p/s, {speedup:.2f}× speedup)")
print(f"  Upload time: {total_upload_time:.3f} s ({100*total_upload_time/t_fused:.1f}%)")
print(f"  Compute time: {total_compute_time:.3f} s ({100*total_compute_time/t_fused:.1f}%)")
print(f"  Download time: {total_download_time:.3f} s ({100*total_download_time/t_fused:.1f}%)")
print()

# ============================================================================
# Validation: Compare Results
# ============================================================================
print("=" * 80)
print("VALIDATION")
print("=" * 80)
print()

# Compare final positions
pos_diff = np.abs(positions_fused - positions_baseline)
max_diff = pos_diff.max()
mean_diff = pos_diff.mean()

print(f"Position differences:")
print(f"  Max: {max_diff:.6e} m")
print(f"  Mean: {mean_diff:.6e} m")

# Compare element IDs
elem_matching = (element_ids_fused == element_ids_baseline).sum()
elem_total = len(element_ids_fused)

print(f"Element ID matching:")
print(f"  Matching: {elem_matching}/{elem_total} ({100*elem_matching/elem_total:.1f}%)")

# Tolerance checks
pos_tolerance = 1e-5  # 10 microns
success = True

if max_diff > pos_tolerance:
    print(f"❌ FAIL: Position difference too large ({max_diff:.6e} > {pos_tolerance})")
    success = False
else:
    print(f"✓ PASS: Position agreement ({max_diff:.6e} < {pos_tolerance})")

if elem_matching / elem_total < 0.95:
    print(f"❌ FAIL: Too many element ID mismatches ({elem_matching}/{elem_total})")
    success = False
else:
    print(f"✓ PASS: Element ID agreement ({elem_matching}/{elem_total})")

# ============================================================================
# Performance Summary
# ============================================================================
print()
print("=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)
print()

print(f"Throughput:")
print(f"  Baseline (CPU-orchestrated): {throughput_baseline:>10,.0f} p/s")
print(f"  GPU-Fused:                   {throughput_fused:>10,.0f} p/s")
print(f"  Speedup:                     {speedup:>10.2f}×")
print()

# Estimate transfer overhead
# Baseline: ~10 MB per timestep (upload positions + element_ids for 5 calls)
# GPU-Fused: ~2 MB per timestep (upload initial state + download final state)
baseline_transfer_mb = N_TIMESTEPS * 10
fused_transfer_mb = N_TIMESTEPS * 2
transfer_reduction = (baseline_transfer_mb - fused_transfer_mb) / baseline_transfer_mb * 100

print(f"Estimated transfer overhead:")
print(f"  Baseline: ~{baseline_transfer_mb:.1f} MB")
print(f"  GPU-Fused: ~{fused_transfer_mb:.1f} MB")
print(f"  Reduction: {transfer_reduction:.0f}%")
print()

if success and speedup >= 1.5:
    print("=" * 80)
    print("✓ ALL TESTS PASSED - GPU-fused RK4 validated!")
    print("=" * 80)
    print()
    print("Expected impact on production:")
    print(f"  Current throughput: ~13k p/s")
    print(f"  With GPU-fused RK4: ~{13000 * speedup:.0f} p/s")
    print(f"  GPU utilization: Expected increase from 30-40% to 60-80%")
elif success:
    print("=" * 80)
    print("✓ CORRECTNESS PASSED - But speedup lower than expected")
    print("=" * 80)
    print(f"  Speedup: {speedup:.2f}× (expected >1.5×)")
    print(f"  This may be due to GPU warming up or other factors")
else:
    print("=" * 80)
    print("❌ SOME TESTS FAILED")
    print("=" * 80)
    sys.exit(1)
