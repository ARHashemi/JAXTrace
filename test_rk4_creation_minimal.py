#!/usr/bin/env python3
"""
Minimal test to isolate the 41.70 TiB error.

This script creates the RK4 function with the same parameters as production,
but with minimal data sizes to see if the error still occurs.
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from jaxtrace.gpu.tracking.rk4_fully_fused_timedep import create_rk4_fully_fused_timedep
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import extract_octree_cells_vertex_multi
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu
from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu
from jaxtrace.gpu.search.morton_global_builder import build_global_morton_octree
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
from jaxtrace.gpu.search.point_in_tet_methods import set_inverse_matrices_gpu
from jaxtrace.gpu.mesh_upload import build_element_neighbors_array, upload_mesh_to_gpu
import jaxtrace.config as config

print("="*80)
print("Minimal RK4 Creation Test - Isolating 41.70 TiB Error")
print("="*80)

# Set config
config.POINT_IN_TET_METHOD = 'inverse'
config.L2_SEARCH_METHOD = 'mesh_aligned_octree'

# Load mesh (same as production)
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
print("\n[1] Loading mesh...")
node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    base_path=MESH_BASE_PATH,
    file_pattern="featurelessAvtk_{timestep}.pvtu",
    timestep_range=(158, 159),  # Just 2 timesteps
    field_name='Displacement',
    verbose=False
)

print(f"  Loaded: {connectivity.shape[0]:,} elements, {node_positions.shape[0]:,} nodes, {velocity_sequence.shape[0]} timesteps")

# Deduplicate
print("\n[2] Deduplicating...")
node_positions, connectivity, _, velocity_sequence = deduplicate_nodes(
    node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
)
print(f"  After dedup: {node_positions.shape[0]:,} nodes")

# Precompute inverse matrices
print("\n[3] Precomputing inverse matrices...")
M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)
M_inv_gpu = jax.device_put(M_inv_array)
p0_gpu = jax.device_put(p0_array)
set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)

# Build Morton octree
print("\n[4] Building Morton octree...")
octree_struct = build_global_morton_octree(
    node_positions=node_positions,
    connectivity=connectivity,
    leaf_capacity=256,
    max_depth=21,
    verbose=False
)
print(f"  Built {octree_struct.n_leaves:,} leaves")

# Build multi-cell mesh-aligned octree
print("\n[5] Building multi-cell mesh-aligned octree...")
mesh_octree_multi_cells = extract_octree_cells_vertex_multi(
    node_positions, connectivity, tolerance=1e-6, verbose=False
)
print(f"  Extracted {mesh_octree_multi_cells.n_cells:,} cells")

# Upload to GPU
print("\n[6] Uploading to GPU...")
element_neighbors = build_element_neighbors_array(connectivity, method='face', verbose=False)
mesh_gpu = upload_mesh_to_gpu(connectivity, node_positions, element_neighbors, verbose=False)
mesh_gpu_octree = upload_global_morton_to_gpu(octree_struct, connectivity, node_positions)
mesh_aligned_octree_gpu = upload_mesh_aligned_octree_to_gpu(
    node_positions, connectivity, mesh_octree_multi_cells, verbose=False
)

# Compute element volumes
print("\n[7] Computing element volumes...")
v0 = node_positions[connectivity[:, 0]]
v1 = node_positions[connectivity[:, 1]]
v2 = node_positions[connectivity[:, 2]]
v3 = node_positions[connectivity[:, 3]]
e1 = v1 - v0
e2 = v2 - v0
e3 = v3 - v0
cross_e2_e3 = np.cross(e2, e3)
det = np.sum(e1 * cross_e2_e3, axis=1)
element_volumes = np.abs(det) / 6.0
element_volumes_gpu = jax.device_put(element_volumes.astype(np.float32))

# Upload velocity
print("\n[8] Uploading velocity sequence...")
velocity_fields_gpu = jax.device_put(velocity_sequence)
print(f"  Shape: {velocity_fields_gpu.shape}")

print("\n" + "="*80)
print("ATTEMPTING TO CREATE RK4 FUNCTION")
print("="*80)
print("\nThis is where the error should occur if the problem is in function creation...")
print()

try:
    rk4_step = create_rk4_fully_fused_timedep(
        mesh_gpu_connectivity=mesh_gpu.connectivity,
        mesh_gpu_node_positions=mesh_gpu.node_positions,
        mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
        mesh_gpu_element_volumes=element_volumes_gpu,
        mesh_gpu_global_morton=mesh_gpu_octree,
        n_hops=5,
        enable_l1_search=True,
        l2_search_method='radius',
        mesh_aligned_octree=mesh_aligned_octree_gpu,
        mesh_aligned_octree_use_multi_local=True
    )
    print("✅ RK4 function created successfully!")
    print()

    # Now try to call it
    print("="*80)
    print("ATTEMPTING TO CALL RK4 FUNCTION (JIT COMPILATION)")
    print("="*80)
    print()

    # Create minimal particle data
    positions_gpu = jnp.array([[0.0, 0.0, -0.005]], dtype=jnp.float32)
    element_ids_gpu = jnp.array([0], dtype=jnp.int32)
    dt = 0.0025

    print("Calling rk4_step with:")
    print(f"  positions_gpu.shape: {positions_gpu.shape}")
    print(f"  element_ids_gpu.shape: {element_ids_gpu.shape}")
    print(f"  dt: {dt}")
    print(f"  velocity_fields_gpu.shape: {velocity_fields_gpu.shape}")
    print(f"  time_idx: 0")
    print()

    positions_out, element_ids_out = rk4_step(
        positions_gpu,
        element_ids_gpu,
        dt,
        velocity_fields_gpu,
        0
    )
    positions_out = jax.block_until_ready(positions_out)

    print("✅ RK4 step executed successfully!")
    print(f"  Output positions: {positions_out.shape}")
    print(f"  Output element_ids: {element_ids_out.shape}")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*80)
print("TEST COMPLETE")
print("="*80)
