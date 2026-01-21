#!/usr/bin/env python3
"""
Detailed RK4 Step Diagnosis
============================

Trace through a single RK4 step to see exactly what velocities are being computed.
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
print("DETAILED RK4 STEP DIAGNOSIS")
print("="*80)

# Load data
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 120)  # Just one timestep
VELOCITY_FIELD_NAME = 'Displacement'
DT = 0.0025

print("\n[1/4] Loading mesh and velocity...")
node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    MESH_BASE_PATH,
    MESH_FILE_PATTERN,
    VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    verbose=False
)

velocity_field = velocity_sequence[0]
vel_mag = np.linalg.norm(velocity_field, axis=1)
print(f"  Velocity field mean: {vel_mag.mean():.6e} m/s")

print("\n[2/4] Setting up mesh structures...")
mesh_gpu_connectivity = jax.device_put(connectivity)
mesh_gpu_node_positions = jax.device_put(node_positions)

element_neighbors = build_element_neighbors_array(connectivity)
mesh_gpu_element_neighbors = jax.device_put(element_neighbors)

morton_cpu = build_global_morton_octree(node_positions, connectivity, leaf_capacity=256)
mesh_gpu_global_morton = upload_global_morton_to_gpu(morton_cpu, connectivity, node_positions)

print("\n[3/4] Creating custom RK4 with debug output...")

# Create a MODIFIED version of RK4 that prints intermediate values
def create_rk4_debug(
    connectivity,
    node_positions,
    element_neighbors,
    global_morton,
    n_hops=3,
    l2_search_radius=2
):
    """Debug version of RK4 that exposes intermediate values."""

    # Import search function
    from jaxtrace.gpu.search.morton_global_search import create_search_l0_l1_l2_single

    search_l0_l1_l2_single = create_search_l0_l1_l2_single(
        connectivity,
        node_positions,
        element_neighbors,
        global_morton,
        n_hops=n_hops,
        l2_search_radius=l2_search_radius
    )

    def interpolate_velocity_single(pos, elem_id, velocity_field):
        """Barycentric velocity interpolation."""
        valid = (elem_id >= 0) & (elem_id < len(connectivity))

        nodes_idx = connectivity[elem_id]
        nodes = node_positions[nodes_idx]
        node_vels = velocity_field[nodes_idx]

        # Barycentric coordinates (same as production code)
        v0 = nodes[1] - nodes[0]
        v1 = nodes[2] - nodes[0]
        v2 = nodes[3] - nodes[0]
        vp = pos - nodes[0]

        d00 = jnp.dot(v0, v0)
        d01 = jnp.dot(v0, v1)
        d02 = jnp.dot(v0, v2)
        d11 = jnp.dot(v1, v1)
        d12 = jnp.dot(v1, v2)
        d22 = jnp.dot(v2, v2)

        dp0 = jnp.dot(vp, v0)
        dp1 = jnp.dot(vp, v1)
        dp2 = jnp.dot(vp, v2)

        det = d00 * (d11*d22 - d12*d12) - d01 * (d01*d22 - d02*d12) + d02 * (d01*d12 - d02*d11)
        det = jnp.where(jnp.abs(det) < 1e-12, 1e-12, det)

        b1 = (dp0 * (d11*d22 - d12*d12) - d01 * (dp1*d22 - dp2*d12) + d02 * (dp1*d12 - dp2*d11)) / det
        b2 = (d00 * (dp1*d22 - dp2*d12) - dp0 * (d01*d22 - d02*d12) + d02 * (d01*dp2 - d02*dp1)) / det
        b3 = (d00 * (d11*dp2 - d12*dp1) - d01 * (d01*dp2 - d02*dp1) + dp0 * (d01*d12 - d02*d11)) / det
        b0 = 1.0 - b1 - b2 - b3

        vel = b0 * node_vels[0] + b1 * node_vels[1] + b2 * node_vels[2] + b3 * node_vels[3]

        return jnp.where(valid, vel, jnp.zeros(3, dtype=jnp.float32)), node_vels, jnp.array([b0, b1, b2, b3])

    def rk4_debug_step(pos, elem_id, dt, velocity_field):
        """Debug version that returns all intermediate values."""

        # Stage 1
        elem_k1 = search_l0_l1_l2_single(pos, elem_id)
        vel_k1, node_vels_k1, bary_k1 = interpolate_velocity_single(pos, elem_k1, velocity_field)
        pos_k1 = pos + 0.5 * dt * vel_k1

        # Stage 2
        elem_k2 = search_l0_l1_l2_single(pos_k1, elem_k1)
        vel_k2, node_vels_k2, bary_k2 = interpolate_velocity_single(pos_k1, elem_k2, velocity_field)
        pos_k2 = pos + 0.5 * dt * vel_k2

        # Stage 3
        elem_k3 = search_l0_l1_l2_single(pos_k2, elem_k2)
        vel_k3, node_vels_k3, bary_k3 = interpolate_velocity_single(pos_k2, elem_k3, velocity_field)
        pos_k3 = pos + dt * vel_k3

        # Stage 4
        elem_k4 = search_l0_l1_l2_single(pos_k3, elem_k3)
        vel_k4, node_vels_k4, bary_k4 = interpolate_velocity_single(pos_k3, elem_k4, velocity_field)

        # Final position
        pos_final = pos + (dt / 6.0) * (vel_k1 + 2.0*vel_k2 + 2.0*vel_k3 + vel_k4)
        elem_final = search_l0_l1_l2_single(pos_final, elem_k4)

        return {
            'pos_final': pos_final,
            'elem_final': elem_final,
            'vel_k1': vel_k1,
            'vel_k2': vel_k2,
            'vel_k3': vel_k3,
            'vel_k4': vel_k4,
            'node_vels_k1': node_vels_k1,
            'bary_k1': bary_k1,
            'elem_k1': elem_k1,
        }

    return rk4_debug_step

rk4_debug = create_rk4_debug(
    mesh_gpu_connectivity,
    mesh_gpu_node_positions,
    mesh_gpu_element_neighbors,
    mesh_gpu_global_morton,
    n_hops=3,
    l2_search_radius=2
)

print("\n[4/4] Testing with a single particle...")

# Create test particle in center of domain
bounds = np.array([
    [node_positions[:, 0].min(), node_positions[:, 0].max()],
    [node_positions[:, 1].min(), node_positions[:, 1].max()],
    [node_positions[:, 2].min(), node_positions[:, 2].max()]
])
center = bounds.mean(axis=1)

test_pos = jnp.array(center, dtype=jnp.float32)
print(f"  Test particle at: [{test_pos[0]:.6f}, {test_pos[1]:.6f}, {test_pos[2]:.6f}]")

# Find element
test_elem = initial_assignment_extended_batch(
    jnp.array([test_pos]),
    mesh_gpu_global_morton,
    max_radius=50
)[0]

print(f"  Assigned to element: {test_elem}")

# Run debug RK4 step
velocity_field_gpu = jax.device_put(velocity_field)
result = rk4_debug(test_pos, test_elem, DT, velocity_field_gpu)

print(f"\n" + "="*80)
print("RK4 INTERMEDIATE VALUES")
print("="*80)

print(f"\nElement K1: {result['elem_k1']}")
print(f"Barycentric coords K1: [{result['bary_k1'][0]:.6f}, {result['bary_k1'][1]:.6f}, "
      f"{result['bary_k1'][2]:.6f}, {result['bary_k1'][3]:.6f}]")
print(f"Sum: {result['bary_k1'].sum():.10f}")

node_vels = np.array(result['node_vels_k1'])
print(f"\nNode velocities in element {result['elem_k1']}:")
for i in range(4):
    mag = np.linalg.norm(node_vels[i])
    print(f"  Node {i}: [{node_vels[i,0]:12.6e}, {node_vels[i,1]:12.6e}, {node_vels[i,2]:12.6e}]  |v|={mag:.6e}")

print(f"\nInterpolated velocities:")
for stage in ['k1', 'k2', 'k3', 'k4']:
    vel = np.array(result[f'vel_{stage}'])
    mag = np.linalg.norm(vel)
    print(f"  {stage}: [{vel[0]:12.6e}, {vel[1]:12.6e}, {vel[2]:12.6e}]  |v|={mag:.6e}")

print(f"\nFinal displacement:")
pos_initial = np.array(test_pos)
pos_final = np.array(result['pos_final'])
disp = pos_final - pos_initial
disp_mag = np.linalg.norm(disp)

print(f"  Initial pos: [{pos_initial[0]:.6f}, {pos_initial[1]:.6f}, {pos_initial[2]:.6f}]")
print(f"  Final pos:   [{pos_final[0]:.6f}, {pos_final[1]:.6f}, {pos_final[2]:.6f}]")
print(f"  Displacement: [{disp[0]:.6e}, {disp[1]:.6e}, {disp[2]:.6e}]")
print(f"  |disp|: {disp_mag:.6e} m = {disp_mag*1000:.6f} mm")

# Expected displacement using mean of k velocities
vel_mean = (np.array(result['vel_k1']) + 2*np.array(result['vel_k2']) +
            2*np.array(result['vel_k3']) + np.array(result['vel_k4'])) / 6.0
expected_disp = vel_mean * DT
expected_disp_mag = np.linalg.norm(expected_disp)

print(f"\nExpected displacement:")
print(f"  Mean vel: [{vel_mean[0]:.6e}, {vel_mean[1]:.6e}, {vel_mean[2]:.6e}]  |v|={np.linalg.norm(vel_mean):.6e}")
print(f"  Expected disp: {expected_disp_mag:.6e} m = {expected_disp_mag*1000:.6f} mm")

ratio = disp_mag / expected_disp_mag
print(f"\nRatio (actual/expected): {ratio:.10f}")

if abs(ratio - 1.0) < 0.001:
    print(f"  ✅ Displacement matches RK4 formula exactly")
else:
    print(f"  ⚠️  Displacement differs from RK4 formula!")

print("="*80)
