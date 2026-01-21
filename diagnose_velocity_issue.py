#!/usr/bin/env python3
"""
Diagnostic Script: Verify RK4, Interpolation, and Time-Dependent Velocity
===========================================================================

This script systematically checks:
1. Velocity field magnitudes (are they non-zero?)
2. Time indexing (which velocity timestep is used at each tracking step?)
3. Barycentric interpolation (is it computing correctly?)
4. RK4 integration (is the formula correct?)
5. Element search impact (do failed searches cause zero velocity?)
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import sys

# Import necessary modules
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu
from jaxtrace.gpu.tracking.rk4_fully_fused_timedep import create_rk4_fully_fused_timedep
from jaxtrace.gpu.tracking.initial_assignment_extended import initial_assignment_extended_batch

# Configuration
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 159)  # 40 timesteps
VELOCITY_FIELD_NAME = 'Displacement'
VELOCITY_DT = 0.1
DT = 0.0025
N_TEST_STEPS = 10  # Just test first 10 steps
N_TEST_PARTICLES = 100  # Small number for detailed analysis

print("="*80)
print("VELOCITY ISSUE DIAGNOSTIC")
print("="*80)

# ============================================================================
# STEP 1: Load velocity sequence and check magnitudes
# ============================================================================
print("\n[1/6] Loading velocity sequence and checking magnitudes...")

node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    MESH_BASE_PATH,
    MESH_FILE_PATTERN,
    VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    verbose=True
)

n_timesteps = velocity_sequence.shape[0]
n_nodes = velocity_sequence.shape[1]

print(f"\n  Velocity sequence shape: {velocity_sequence.shape}")
print(f"  Number of timesteps: {n_timesteps}")
print(f"  Number of nodes: {n_nodes}")

# Check velocity magnitudes for each timestep
print(f"\n  Velocity magnitude statistics per timestep:")
print(f"  {'Timestep':>10} {'Min':>12} {'Mean':>12} {'Max':>12} {'Std':>12}")
print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

for i in range(n_timesteps):
    vel_mag = np.linalg.norm(velocity_sequence[i], axis=1)
    print(f"  {i:10d} {vel_mag.min():12.6e} {vel_mag.mean():12.6e} {vel_mag.max():12.6e} {vel_mag.std():12.6e}")

# Overall statistics
vel_mag_all = np.linalg.norm(velocity_sequence.reshape(-1, 3), axis=1)
print(f"\n  Overall velocity magnitude:")
print(f"    Min:  {vel_mag_all.min():.6e}")
print(f"    Mean: {vel_mag_all.mean():.6e}")
print(f"    Max:  {vel_mag_all.max():.6e}")
print(f"    Std:  {vel_mag_all.std():.6e}")

# Check if velocities are suspiciously small
if vel_mag_all.mean() < 1e-6:
    print(f"\n  ⚠️  WARNING: Velocity magnitudes are very small (mean < 1e-6)")
    print(f"      This might explain why particles don't move!")

# ============================================================================
# STEP 2: Set up mesh and Morton structure
# ============================================================================
print("\n[2/6] Setting up mesh and Morton structure...")

# Upload mesh to GPU
mesh_gpu_connectivity = jax.device_put(connectivity)
mesh_gpu_node_positions = jax.device_put(node_positions)

# Compute element neighbors
print("  Computing element neighbors...")
element_neighbors = build_element_neighbors_array(connectivity)
mesh_gpu_element_neighbors = jax.device_put(element_neighbors)
print(f"    Element neighbors computed: {element_neighbors.shape}")

# Build Morton structure (CPU)
print("  Building global Morton octree (CPU)...")
morton_cpu = build_global_morton_octree(
    node_positions,
    connectivity,
    leaf_capacity=256
)
print(f"    Built {morton_cpu.n_leaves} leaves")

# Upload Morton structure to GPU
print("  Uploading Morton structure to GPU...")
mesh_gpu_global_morton = upload_global_morton_to_gpu(morton_cpu, connectivity, node_positions)
print(f"    Morton leaves: {mesh_gpu_global_morton.n_leaves}")

# ============================================================================
# STEP 3: Initialize test particles and check element assignment
# ============================================================================
print("\n[3/6] Initializing test particles...")

# Create small grid of particles in center of domain
bounds = np.array([
    [node_positions[:, 0].min(), node_positions[:, 0].max()],
    [node_positions[:, 1].min(), node_positions[:, 1].max()],
    [node_positions[:, 2].min(), node_positions[:, 2].max()]
])
x_center = (bounds[0, 0] + bounds[0, 1]) / 2
y_center = (bounds[1, 0] + bounds[1, 1]) / 2
z_center = (bounds[2, 0] + bounds[2, 1]) / 2

x_range = 0.005
y_range = 0.005
z_range = 0.005

n_per_dim = int(np.round(N_TEST_PARTICLES**(1/3)))
x = np.linspace(x_center - x_range, x_center + x_range, n_per_dim)
y = np.linspace(y_center - y_range, y_center + y_range, n_per_dim)
z = np.linspace(z_center - z_range, z_center + z_range, n_per_dim)

xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
positions = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)
positions = positions[:N_TEST_PARTICLES]  # Truncate to exact count

print(f"  Created {len(positions)} test particles")
print(f"  Particle bounds:")
print(f"    X: [{positions[:, 0].min():.6f}, {positions[:, 0].max():.6f}]")
print(f"    Y: [{positions[:, 1].min():.6f}, {positions[:, 1].max():.6f}]")
print(f"    Z: [{positions[:, 2].min():.6f}, {positions[:, 2].max():.6f}]")

# Upload to GPU
positions_gpu = jax.device_put(positions)

# Initial element assignment
print("\n  Running initial element assignment (radius=50)...")
element_ids_gpu = initial_assignment_extended_batch(
    positions_gpu,
    mesh_gpu_global_morton,
    max_radius=50
)
element_ids = np.array(element_ids_gpu)

n_assigned = np.sum(element_ids >= 0)
print(f"    Assigned: {n_assigned}/{len(positions)} ({100*n_assigned/len(positions):.1f}%)")

if n_assigned < len(positions):
    print(f"    ⚠️  WARNING: Not all particles assigned!")
    print(f"        Unassigned particles will have zero velocity")

# ============================================================================
# STEP 4: Create RK4 integrator and upload velocity
# ============================================================================
print("\n[4/6] Creating RK4 integrator...")

rk4_step = create_rk4_fully_fused_timedep(
    mesh_gpu_connectivity,
    mesh_gpu_node_positions,
    mesh_gpu_element_neighbors,
    mesh_gpu_global_morton,
    n_hops=3,
    l2_search_radius=2
)

print("  Uploading velocity sequence to GPU...")
velocity_fields_gpu = jax.device_put(velocity_sequence)
print(f"    Velocity on GPU: {velocity_sequence.nbytes / (1024**2):.1f} MB")

# ============================================================================
# STEP 5: Test time indexing
# ============================================================================
print("\n[5/6] Testing time indexing...")

print(f"\n  Time indexing test (first {min(50, 2*n_timesteps)} steps):")
print(f"  {'Tracking Step':>15} {'Time Index':>12} {'Velocity Timestep':>18} {'Physical Time':>15}")
print(f"  {'-'*15} {'-'*12} {'-'*18} {'-'*15}")

for step in range(min(50, 2*n_timesteps)):
    time_idx = step
    vel_idx = time_idx % n_timesteps
    phys_time = step * DT
    print(f"  {step:15d} {time_idx:12d} {vel_idx:18d} {phys_time:15.6f}")

# Check cycling behavior
print(f"\n  Velocity cycling:")
print(f"    Total tracking steps: {N_TEST_STEPS}")
print(f"    Velocity timesteps: {n_timesteps}")
print(f"    Steps per velocity: {VELOCITY_DT / DT:.1f}")
print(f"    Cycles in test: {N_TEST_STEPS / n_timesteps:.2f}")

# ============================================================================
# STEP 6: Run test integration with detailed diagnostics
# ============================================================================
print("\n[6/6] Running test integration with diagnostics...")

# Compile first step
print("\n  Compiling RK4 (first step)...")
import time
t0 = time.time()
positions_new, element_ids_new = rk4_step(
    positions_gpu,
    element_ids_gpu,
    DT,
    velocity_fields_gpu,
    0  # time_idx = 0
)
positions_new.block_until_ready()
compile_time = time.time() - t0
print(f"    Compilation time: {compile_time:.2f}s")

# Check first step results
positions_step0 = np.array(positions_gpu)
positions_step1 = np.array(positions_new)
element_ids_step1 = np.array(element_ids_new)

displacement = positions_step1 - positions_step0
displacement_mag = np.linalg.norm(displacement, axis=1)

print(f"\n  First step results:")
print(f"    Displacement magnitude:")
print(f"      Min:  {displacement_mag.min():.6e}")
print(f"      Mean: {displacement_mag.mean():.6e}")
print(f"      Max:  {displacement_mag.max():.6e}")
print(f"      Std:  {displacement_mag.std():.6e}")

# Check if particles moved at all
n_moved = np.sum(displacement_mag > 1e-10)
print(f"    Particles that moved (|disp| > 1e-10): {n_moved}/{len(positions)}")

if n_moved == 0:
    print(f"\n  ❌ CRITICAL: No particles moved!")
    print(f"     Possible causes:")
    print(f"     1. Velocity field is all zeros")
    print(f"     2. Interpolation is broken")
    print(f"     3. Element search is failing")
    print(f"     4. RK4 formula is wrong")

# Check element retention
n_retained = np.sum(element_ids_step1 >= 0)
print(f"    Elements retained: {n_retained}/{len(positions)} ({100*n_retained/len(positions):.1f}%)")

# Run full test (10 steps) and track displacement
print(f"\n  Running {N_TEST_STEPS} timesteps...")
print(f"  {'Step':>6} {'Active':>8} {'Mean |disp|':>12} {'Max |disp|':>12} {'Vel Idx':>8}")
print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*12} {'-'*8}")

positions_current = positions_gpu
element_ids_current = element_ids_gpu
positions_prev = np.array(positions_current)

for step in range(1, N_TEST_STEPS + 1):
    time_idx = step
    vel_idx = time_idx % n_timesteps

    # Take step
    positions_current, element_ids_current = rk4_step(
        positions_current,
        element_ids_current,
        DT,
        velocity_fields_gpu,
        time_idx
    )

    # Analyze
    positions_now = np.array(positions_current)
    element_ids_now = np.array(element_ids_current)

    step_disp = positions_now - positions_prev
    step_disp_mag = np.linalg.norm(step_disp, axis=1)

    n_active = np.sum(element_ids_now >= 0)

    print(f"  {step:6d} {n_active:8d} {step_disp_mag.mean():12.6e} {step_disp_mag.max():12.6e} {vel_idx:8d}")

    positions_prev = positions_now

# Final summary
print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)

# Determine most likely issue
issues_found = []

if vel_mag_all.mean() < 1e-6:
    issues_found.append("⚠️  Velocity magnitudes are very small (< 1e-6)")

if n_assigned < 0.95 * len(positions):
    issues_found.append(f"⚠️  Low initial assignment rate: {100*n_assigned/len(positions):.1f}%")

if n_moved == 0:
    issues_found.append("❌ CRITICAL: Particles did not move in first step")
elif displacement_mag.mean() < 1e-8:
    issues_found.append("⚠️  Particles moved very little in first step (< 1e-8)")

if n_retained < 0.95 * n_assigned:
    issues_found.append(f"⚠️  Low element retention after first step: {100*n_retained/n_assigned:.1f}%")

if issues_found:
    print("\nIssues detected:")
    for issue in issues_found:
        print(f"  {issue}")
else:
    print("\n✅ No obvious issues detected in diagnostic tests")

print("\nNext steps:")
if vel_mag_all.mean() < 1e-6:
    print("  1. Verify 'Displacement' field is actually velocity (not just mesh displacement)")
    print("  2. Check if field needs to be scaled or converted")
    print("  3. Inspect PVTU files directly to confirm field values")
elif n_moved == 0:
    print("  1. Add detailed tracing inside RK4 to see intermediate velocities")
    print("  2. Test interpolation function with known element/position")
    print("  3. Check if element IDs are valid before interpolation")
else:
    print("  1. Run longer simulation to see if displacement accumulates")
    print("  2. Compare with expected trajectories from velocity field")
    print("  3. Visualize particle paths in ParaView")

print("="*80)
