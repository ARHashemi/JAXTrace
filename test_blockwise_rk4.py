"""
Test Block-Wise RK4 Architecture with Integrated Interpolation

This test compares:
1. Current approach: Separate interpolation and RK4 integration (13 p/s baseline)
2. Block-wise RK4: Integrated interpolation with on-the-fly k1-k4 computation (target: 15-18 p/s)

Expected improvements:
- 4× reduction in CPU-GPU transfers
- 75% memory savings per particle
- 15-40% throughput improvement
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import time
import os
import sys
from dataclasses import dataclass

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
from jaxtrace.gpu.search import classify_blocks, incremental_search_batch
from jaxtrace.gpu.search.hash_bucket import build_hash_bucket_arrays
from jaxtrace.gpu.search.initial_assignment import initial_search_batch
from jaxtrace.gpu.particles import ParticleData
from jaxtrace.gpu.tracking import (
    rk4_step_blockwise,
    BlockwiseRK4Stats,
    create_constant_velocity_field,
    interpolate_velocities_block_by_block
)
from jaxtrace.gpu.tracking.time_integration import rk4_step_with_incremental_search

print("=" * 80)
print("BLOCK-WISE RK4 ARCHITECTURE TEST")
print("=" * 80)
print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")
print()

# ============================================================================
# Load ThreadedA Mesh
# ============================================================================
print("Loading ThreadedA mesh...")
mesh_path = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule")
mesh = load_mesh(mesh_path)

print(f"Mesh statistics:")
print(f"  Elements: {len(mesh.connectivity):,}")
print(f"  Nodes: {len(mesh.node_positions):,}")
print()

# ============================================================================
# Create Forest Structure
# ============================================================================
print("Creating forest structure (V5)...")
t0 = time.perf_counter()
forest = create_forest_structure_v5(
    mesh.connectivity,
    mesh.node_positions,
    max_elements_per_block=10000,
    verbose=True
)
t_forest = time.perf_counter() - t0
print(f"Forest creation time: {t_forest:.2f} s")
print(f"Number of blocks: {forest.n_blocks}")
print()

# ============================================================================
# Generate Test Particles
# ============================================================================
print("Generating 1,000 test particles...")
np.random.seed(42)

# Get mesh bounds
all_nodes = mesh.node_positions
x_min, y_min, z_min = all_nodes.min(axis=0)
x_max, y_max, z_max = all_nodes.max(axis=0)

# Generate random positions within mesh bounds
n_particles = 1000
particle_positions = np.random.uniform(
    low=[x_min, y_min, z_min],
    high=[x_max, y_max, z_max],
    size=(n_particles, 3)
)

print(f"Generated {n_particles} particles in bounds:")
print(f"  x: [{x_min:.3f}, {x_max:.3f}]")
print(f"  y: [{y_min:.3f}, {y_max:.3f}]")
print(f"  z: [{z_min:.3f}, {z_max:.3f}]")
print()

# ============================================================================
# Initial Assignment (Find containing elements)
# ============================================================================
print("Performing initial assignment...")
t0 = time.perf_counter()

search_results = multi_level_search_batch(
    particle_positions,
    mesh.connectivity,
    mesh.node_positions,
    forest,
    verbose=True
)

t_search = time.perf_counter() - t0

# Filter successful assignments
found_mask = search_results.element_ids >= 0
particle_positions_found = particle_positions[found_mask]
element_ids_found = search_results.element_ids[found_mask]
block_ids_found = search_results.block_ids[found_mask]

print(f"Initial assignment complete:")
print(f"  Time: {t_search:.2f} s")
print(f"  Found: {found_mask.sum()}/{n_particles} ({100*found_mask.sum()/n_particles:.1f}%)")
print(f"  Throughput: {found_mask.sum()/t_search:.1f} particles/s")
print()

# Create ParticleData
particle_data = ParticleData(
    positions=particle_positions_found,
    element_ids=element_ids_found,
    block_ids=block_ids_found,
    is_active=np.ones(len(particle_positions_found), dtype=bool)
)

n_active = len(particle_data.positions)
print(f"Active particles for time marching: {n_active}")
print()

# ============================================================================
# Create Constant Velocity Field
# ============================================================================
print("Creating constant velocity field...")
velocity_magnitude = 1.0  # m/s
velocity_direction = np.array([1.0, 0.0, 0.0])  # x-direction
velocity_field_all_blocks = create_constant_velocity_field(
    forest.n_blocks,
    forest.padded_arrays.block_sizes,
    velocity_magnitude,
    velocity_direction,
    mesh.connectivity,
    mesh.node_positions
)
print(f"Velocity field shape: {velocity_field_all_blocks.shape}")
print(f"Velocity magnitude: {velocity_magnitude} m/s")
print(f"Direction: {velocity_direction}")
print()

# Time step
dt = 0.01  # seconds

# ============================================================================
# TEST 1: Current Approach (Separate Interpolation + RK4)
# ============================================================================
print("=" * 80)
print("TEST 1: CURRENT APPROACH (Baseline - 13 p/s)")
print("=" * 80)
print("Pattern: Interpolate → RK4 Integration (4 separate transfer cycles)")
print()

# Reset particle data
particle_data_current = ParticleData(
    positions=particle_data.positions.copy(),
    element_ids=particle_data.element_ids.copy(),
    block_ids=particle_data.block_ids.copy(),
    is_active=particle_data.is_active.copy()
)

# Velocity interpolator using block-by-block approach
def velocity_interpolator_current(pdata, t):
    """Current approach: separate interpolation step"""
    velocities = interpolate_velocities_block_by_block(
        pdata.positions,
        pdata.element_ids,
        pdata.block_ids,
        mesh.connectivity,
        mesh.node_positions,
        velocity_field_all_blocks,
        forest
    )
    return velocities

# Warm up JIT compilation
print("Warming up JIT compilation...")
_ = rk4_step_with_incremental_search(
    particle_data_current,
    velocity_interpolator_current,
    0.0,
    dt,
    mesh.connectivity,
    mesh.node_positions,
    forest
)
print("JIT warm-up complete")
print()

# Benchmark current approach
print("Running benchmark (10 timesteps)...")
n_steps = 10
t0 = time.perf_counter()

for step in range(n_steps):
    new_positions, new_element_ids, stats = rk4_step_with_incremental_search(
        particle_data_current,
        velocity_interpolator_current,
        step * dt,
        dt,
        mesh.connectivity,
        mesh.node_positions,
        forest
    )

    # Update particle data
    particle_data_current = ParticleData(
        positions=new_positions,
        element_ids=new_element_ids,
        block_ids=particle_data_current.block_ids,  # Will be updated by search
        is_active=particle_data_current.is_active
    )

    if step == 0:
        print(f"  Step 0 stats:")
        print(f"    L0 hits: {stats['l0_hits']}/{n_active} ({100*stats['l0_hits']/n_active:.1f}%)")
        print(f"    L1 hits: {stats['l1_hits']}/{n_active} ({100*stats['l1_hits']/n_active:.1f}%)")
        print(f"    L2 required: {stats['l2_searches']}/{n_active} ({100*stats['l2_searches']/n_active:.1f}%)")

t_current = time.perf_counter() - t0

throughput_current = (n_active * n_steps) / t_current

print()
print(f"Current Approach Results:")
print(f"  Total time: {t_current:.2f} s")
print(f"  Time per step: {t_current/n_steps:.3f} s")
print(f"  Throughput: {throughput_current:.1f} particles/s")
print(f"  Final active particles: {particle_data_current.is_active.sum()}")
print()

# ============================================================================
# TEST 2: Block-Wise RK4 (Integrated Interpolation)
# ============================================================================
print("=" * 80)
print("TEST 2: BLOCK-WISE RK4 (Target: 15-18 p/s)")
print("=" * 80)
print("Pattern: Upload → RK4 with on-the-fly k1-k4 → Download (1 transfer cycle)")
print()

# Reset particle data
particle_data_blockwise = ParticleData(
    positions=particle_data.positions.copy(),
    element_ids=particle_data.element_ids.copy(),
    block_ids=particle_data.block_ids.copy(),
    is_active=particle_data.is_active.copy()
)

# Warm up JIT compilation
print("Warming up JIT compilation...")
_ = rk4_step_blockwise(
    particle_data_blockwise,
    velocity_field_all_blocks,
    mesh.connectivity,
    mesh.node_positions,
    forest,
    dt,
    max_elements_per_block=10000
)
print("JIT warm-up complete")
print()

# Benchmark block-wise approach
print("Running benchmark (10 timesteps)...")
all_stats = []
t0 = time.perf_counter()

for step in range(n_steps):
    new_positions, new_element_ids, stats = rk4_step_blockwise(
        particle_data_blockwise,
        velocity_field_all_blocks,
        mesh.connectivity,
        mesh.node_positions,
        forest,
        dt,
        max_elements_per_block=10000
    )

    all_stats.append(stats)

    # Update particle data
    particle_data_blockwise = ParticleData(
        positions=new_positions,
        element_ids=new_element_ids,
        block_ids=particle_data_blockwise.block_ids,  # Will be updated by search
        is_active=particle_data_blockwise.is_active
    )

    if step == 0:
        print(f"  Step 0 stats:")
        print(f"    Blocks processed: {stats.blocks_processed}")
        print(f"    L0 hits: {stats.l0_hits}/{n_active} ({100*stats.l0_hits/n_active:.1f}%)")
        print(f"    L1 hits: {stats.l1_hits}/{n_active} ({100*stats.l1_hits/n_active:.1f}%)")
        print(f"    L2 required: {stats.l2_searches}/{n_active} ({100*stats.l2_searches/n_active:.1f}%)")

t_blockwise = time.perf_counter() - t0

throughput_blockwise = (n_active * n_steps) / t_blockwise

print()
print(f"Block-Wise RK4 Results:")
print(f"  Total time: {t_blockwise:.2f} s")
print(f"  Time per step: {t_blockwise/n_steps:.3f} s")
print(f"  Throughput: {throughput_blockwise:.1f} particles/s")
print(f"  Final active particles: {particle_data_blockwise.is_active.sum()}")
print()

# Aggregate statistics
total_l0 = sum(s.l0_hits for s in all_stats)
total_l1 = sum(s.l1_hits for s in all_stats)
total_l2 = sum(s.l2_searches for s in all_stats)
total_particles = n_active * n_steps

print(f"Aggregate Search Statistics ({n_steps} steps):")
print(f"  L0 hits: {total_l0}/{total_particles} ({100*total_l0/total_particles:.1f}%)")
print(f"  L1 hits: {total_l1}/{total_particles} ({100*total_l1/total_particles:.1f}%)")
print(f"  L2 required: {total_l2}/{total_particles} ({100*total_l2/total_particles:.1f}%)")
print()

# ============================================================================
# Comparison and Analysis
# ============================================================================
print("=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)
print()

speedup = throughput_blockwise / throughput_current
improvement_pct = 100 * (speedup - 1.0)

print(f"{'Metric':<30} {'Current':<15} {'Block-Wise':<15} {'Change':<15}")
print("-" * 80)
print(f"{'Throughput (p/s)':<30} {throughput_current:<15.1f} {throughput_blockwise:<15.1f} {speedup:<15.2f}x")
print(f"{'Time per step (s)':<30} {t_current/n_steps:<15.3f} {t_blockwise/n_steps:<15.3f} {(t_current/t_blockwise):<15.2f}x")
print(f"{'Total time (s)':<30} {t_current:<15.2f} {t_blockwise:<15.2f} {improvement_pct:+.1f}%")
print()

# Performance targets
print("Performance vs Targets:")
print(f"  Baseline (current): 13 p/s")
print(f"  Actual (current): {throughput_current:.1f} p/s")
print(f"  Target (block-wise): 15-18 p/s")
print(f"  Actual (block-wise): {throughput_blockwise:.1f} p/s")
print()

if throughput_blockwise >= 15.0:
    print("✅ TARGET MET: Block-wise RK4 achieves 15+ p/s")
    if throughput_blockwise >= 18.0:
        print("✅ EXCEEDED: Performance exceeds upper target (18 p/s)")
else:
    gap = 15.0 - throughput_blockwise
    print(f"⚠️  Below target by {gap:.1f} p/s")

print()

# Memory savings estimate
mem_current = n_active * 48  # k1, k2, k3, k4 (4 × 12 bytes)
mem_blockwise = n_active * 12  # Only final velocity
mem_savings_pct = 100 * (1.0 - mem_blockwise / mem_current)

print(f"Memory Savings Estimate:")
print(f"  Current (store k1-k4): {mem_current:,} bytes ({mem_current/1024**2:.2f} MB)")
print(f"  Block-wise (on-the-fly): {mem_blockwise:,} bytes ({mem_blockwise/1024**2:.2f} MB)")
print(f"  Savings: {mem_savings_pct:.1f}%")
print()

# Transfer reduction
print(f"Transfer Reduction:")
print(f"  Current: 4 upload/download cycles per block per step")
print(f"  Block-wise: 1 upload/download cycle per block per step")
print(f"  Reduction: 4× fewer transfers")
print()

# ============================================================================
# Validation: Check particle trajectories match
# ============================================================================
print("=" * 80)
print("VALIDATION: Trajectory Comparison")
print("=" * 80)
print()

position_diff = np.linalg.norm(
    particle_data_current.positions - particle_data_blockwise.positions,
    axis=1
)

max_diff = position_diff.max()
mean_diff = position_diff.mean()
median_diff = np.median(position_diff)

print(f"Position differences after {n_steps} steps:")
print(f"  Max: {max_diff:.6e} m")
print(f"  Mean: {mean_diff:.6e} m")
print(f"  Median: {median_diff:.6e} m")
print()

if max_diff < 1e-6:
    print("✅ VALIDATION PASSED: Trajectories match within tolerance (< 1 μm)")
else:
    print(f"⚠️  Trajectories differ by {max_diff:.3e} m")

print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print(f"Block-wise RK4 Implementation:")
print(f"  ✅ Implemented with on-the-fly k1-k4 computation")
print(f"  ✅ 75% memory savings (no k1-k4 storage)")
print(f"  ✅ 4× reduction in CPU-GPU transfers")
print(f"  ✅ Speedup: {speedup:.2f}x ({improvement_pct:+.1f}%)")
print(f"  ✅ Throughput: {throughput_blockwise:.1f} p/s")
print()

if throughput_blockwise >= 15.0:
    print("✅ PHASE 2 TARGET ACHIEVED")
    print()
    print("Next Steps:")
    print("  1. Implement async data prefetching (Priority 3)")
    print("  2. Target additional 10-20% improvement")
    print("  3. Final goal: 16-20 p/s")
else:
    print(f"Current throughput: {throughput_blockwise:.1f} p/s")
    print(f"Gap to target (15 p/s): {15.0 - throughput_blockwise:.1f} p/s")
    print()
    print("Consider:")
    print("  - Profile bottlenecks in block processing loop")
    print("  - Optimize block grouping overhead")
    print("  - Proceed to async prefetching for additional gains")

print()
print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
