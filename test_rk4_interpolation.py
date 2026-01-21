#!/usr/bin/env python3
"""
Test RK4 and Barycentric Interpolation Correctness
===================================================

This tests:
1. Barycentric interpolation for tetrahedra
2. RK4 integration with known velocity field
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

# Load velocity and mesh
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu
from jaxtrace.gpu.tracking.rk4_fully_fused_timedep import create_rk4_fully_fused_timedep
from jaxtrace.gpu.tracking.initial_assignment_extended import initial_assignment_extended_batch

print("="*80)
print("RK4 AND INTERPOLATION CORRECTNESS TEST")
print("="*80)

# Load minimal data
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 120)  # Just one timestep
VELOCITY_FIELD_NAME = 'Displacement'
DT = 0.0025

print("\n[1/3] Loading mesh and velocity...")
node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    MESH_BASE_PATH,
    MESH_FILE_PATTERN,
    VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    verbose=False
)

print(f"  Loaded {node_positions.shape[0]:,} nodes, {connectivity.shape[0]:,} elements")
velocity_field = velocity_sequence[0]  # (n_nodes, 3)

# Compute velocity statistics
vel_mag = np.linalg.norm(velocity_field, axis=1)
print(f"  Velocity magnitude: min={vel_mag.min():.6e}, mean={vel_mag.mean():.6e}, max={vel_mag.max():.6e}")

print("\n[2/3] Setting up mesh structures...")
mesh_gpu_connectivity = jax.device_put(connectivity)
mesh_gpu_node_positions = jax.device_put(node_positions)

element_neighbors = build_element_neighbors_array(connectivity)
mesh_gpu_element_neighbors = jax.device_put(element_neighbors)

morton_cpu = build_global_morton_octree(node_positions, connectivity, leaf_capacity=256)
mesh_gpu_global_morton = upload_global_morton_to_gpu(morton_cpu, connectivity, node_positions)

# Create RK4 integrator
rk4_step = create_rk4_fully_fused_timedep(
    mesh_gpu_connectivity,
    mesh_gpu_node_positions,
    mesh_gpu_element_neighbors,
    mesh_gpu_global_morton,
    n_hops=3,
    l2_search_radius=2
)

velocity_fields_gpu = jax.device_put(velocity_sequence)

print("\n[3/3] Testing RK4 integration...")

# Create test particle in center of domain
bounds = np.array([
    [node_positions[:, 0].min(), node_positions[:, 0].max()],
    [node_positions[:, 1].min(), node_positions[:, 1].max()],
    [node_positions[:, 2].min(), node_positions[:, 2].max()]
])
center = bounds.mean(axis=1)

# Single test particle
test_pos = np.array([center], dtype=np.float32)
print(f"  Test particle at: [{test_pos[0,0]:.6f}, {test_pos[0,1]:.6f}, {test_pos[0,2]:.6f}]")

# Find element
test_pos_gpu = jax.device_put(test_pos)
test_elem_gpu = initial_assignment_extended_batch(
    test_pos_gpu,
    mesh_gpu_global_morton,
    max_radius=50
)
test_elem = np.array(test_elem_gpu)[0]

print(f"  Assigned to element: {test_elem}")

if test_elem < 0:
    print("  ❌ FAIL: Could not assign particle to element!")
    exit(1)

# Get element nodes and their velocities
elem_nodes_idx = connectivity[test_elem]
elem_nodes = node_positions[elem_nodes_idx]
elem_vels = velocity_field[elem_nodes_idx]

print(f"\n  Element nodes:")
for i in range(4):
    print(f"    Node {i}: pos=[{elem_nodes[i,0]:.6f}, {elem_nodes[i,1]:.6f}, {elem_nodes[i,2]:.6f}], "
          f"vel=[{elem_vels[i,0]:.6e}, {elem_vels[i,1]:.6e}, {elem_vels[i,2]:.6e}], "
          f"|vel|={np.linalg.norm(elem_vels[i]):.6e}")

# Manual barycentric interpolation (using standard formula)
# For tetrahedral element with vertices v0, v1, v2, v3 and point p:
# Solve: p = b0*v0 + b1*v1 + b2*v2 + b3*v3 with b0+b1+b2+b3=1

def barycentric_coords_tet(p, v0, v1, v2, v3):
    """Compute barycentric coordinates using volume method"""
    # Matrix method: [v1-v0, v2-v0, v3-v0] * [b1, b2, b3]^T = p - v0
    mat = np.column_stack([v1-v0, v2-v0, v3-v0])
    rhs = p - v0
    try:
        coords_123 = np.linalg.solve(mat, rhs)
        b1, b2, b3 = coords_123
        b0 = 1.0 - b1 - b2 - b3
        return np.array([b0, b1, b2, b3])
    except np.linalg.LinAlgError:
        return None

bary = barycentric_coords_tet(test_pos[0], elem_nodes[0], elem_nodes[1], elem_nodes[2], elem_nodes[3])
print(f"\n  Barycentric coords (standard formula): [{bary[0]:.6f}, {bary[1]:.6f}, {bary[2]:.6f}, {bary[3]:.6f}]")
print(f"  Sum of coords: {bary.sum():.10f} (should be 1.0)")

# Manual velocity interpolation
vel_manual = (bary[0] * elem_vels[0] + bary[1] * elem_vels[1] +
              bary[2] * elem_vels[2] + bary[3] * elem_vels[3])
print(f"  Manual interpolated velocity: [{vel_manual[0]:.6e}, {vel_manual[1]:.6e}, {vel_manual[2]:.6e}]")
print(f"  Manual interpolated |vel|: {np.linalg.norm(vel_manual):.6e}")

# Expected displacement for one RK4 step
# For constant velocity: RK4 gives exact result = v * dt
expected_disp_manual = vel_manual * DT
print(f"  Expected displacement (manual): [{expected_disp_manual[0]:.6e}, {expected_disp_manual[1]:.6e}, {expected_disp_manual[2]:.6e}]")
print(f"  Expected |disp|: {np.linalg.norm(expected_disp_manual):.6e}")

# Run RK4 step
print(f"\n  Running RK4 step...")
pos_new, elem_new = rk4_step(
    test_pos_gpu,
    test_elem_gpu,
    DT,
    velocity_fields_gpu,
    0  # time_idx
)

pos_new_np = np.array(pos_new)[0]
actual_disp = pos_new_np - test_pos[0]

print(f"  Actual displacement: [{actual_disp[0]:.6e}, {actual_disp[1]:.6e}, {actual_disp[2]:.6e}]")
print(f"  Actual |disp|: {np.linalg.norm(actual_disp):.6e}")

# Compare
ratio = np.linalg.norm(actual_disp) / np.linalg.norm(expected_disp_manual)
print(f"\n  Displacement ratio (actual/expected): {ratio:.6f}")

if abs(ratio - 1.0) < 0.01:
    print(f"  ✅ PASS: RK4 displacement matches expected (within 1%)")
else:
    print(f"  ⚠️  WARNING: RK4 displacement differs from expected by {abs(ratio-1.0)*100:.1f}%")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

if abs(ratio - 1.0) < 0.01:
    print("\nRK4 and interpolation appear to be working correctly.")
    print("The issue with slow particle movement may be related to:")
    print("1. Velocity field being very small in magnitude")
    print("2. Time-dependent velocity indexing")
    print("3. Integration timestep being too small")
else:
    print(f"\n⚠️  RK4 or interpolation may have an issue!")
    print(f"   Expected displacement: {np.linalg.norm(expected_disp_manual):.6e}")
    print(f"   Actual displacement: {np.linalg.norm(actual_disp):.6e}")
    print(f"   Ratio: {ratio:.6f}")

print("="*80)
