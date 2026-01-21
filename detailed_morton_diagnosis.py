#!/usr/bin/env python3
"""
Detailed Morton Search Diagnosis
=================================

Debug exactly why centroid search fails.
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree, morton_encode_3d

print("="*80)
print("DETAILED MORTON DIAGNOSIS")
print("="*80)

# Load mesh
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 120)
VELOCITY_FIELD_NAME = 'Displacement'

print("\n[1/2] Loading mesh...")
node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    MESH_BASE_PATH,
    MESH_FILE_PATTERN,
    VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    verbose=False
)

# Find smallest element
element_sizes = np.zeros(len(connectivity))
element_centroids = np.zeros((len(connectivity), 3))

for i, elem_nodes_idx in enumerate(connectivity):
    elem_nodes = node_positions[elem_nodes_idx]
    centroid = elem_nodes.mean(axis=0)
    element_centroids[i] = centroid

    max_edge = 0.0
    for j in range(4):
        for k in range(j+1, 4):
            edge_len = np.linalg.norm(elem_nodes[j] - elem_nodes[k])
            max_edge = max(max_edge, edge_len)
    element_sizes[i] = max_edge

smallest_elem_idx = np.argmin(element_sizes)
centroid = element_centroids[smallest_elem_idx]

print(f"\nSmallest element: {smallest_elem_idx}")
print(f"  Centroid: [{centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f}]")

# Build Morton octree
print("\n[2/2] Building Morton octree...")
morton_cpu = build_global_morton_octree(node_positions, connectivity, leaf_capacity=256, verbose=False)

print(f"\nMorton structure:")
print(f"  Number of leaves: {morton_cpu.n_leaves}")
print(f"  Bounding box: min={morton_cpu.bbox_min}, max={morton_cpu.bbox_max}")

# Compute Morton code for centroid
centroid_morton = morton_encode_3d(centroid, morton_cpu.bbox_min, morton_cpu.bbox_max, max_depth=morton_cpu.max_depth)
print(f"\nCentroid Morton code: {centroid_morton}")

# Compute Morton code for element (from sorted list)
elem_position_in_sorted = np.where(morton_cpu.elem_ids_sorted == smallest_elem_idx)[0]
if len(elem_position_in_sorted) > 0:
    pos = elem_position_in_sorted[0]
    elem_morton = morton_cpu.morton_sorted[pos]
    print(f"Element Morton code (from sorted list): {elem_morton}")
    print(f"Element position in sorted list: {pos}")

    # Find which leaf contains this element
    for leaf_idx in range(morton_cpu.n_leaves):
        leaf_start = morton_cpu.leaf_start[leaf_idx]
        leaf_size = morton_cpu.leaf_length[leaf_idx]
        if leaf_start <= pos < leaf_start + leaf_size:
            print(f"\nElement is in leaf {leaf_idx}:")
            print(f"  Leaf start: {leaf_start}")
            print(f"  Leaf size: {leaf_size}")
            print(f"  Leaf Morton code: {morton_cpu.leaf_morton_codes[leaf_idx]}")

            # Check if centroid Morton matches leaf Morton
            if centroid_morton == elem_morton:
                print(f"  ✅ Centroid Morton MATCHES element Morton")
            else:
                print(f"  ❌ Centroid Morton DIFFERS from element Morton!")
                print(f"     Difference: {abs(int(centroid_morton) - int(elem_morton))}")

            break
else:
    print(f"❌ Element {smallest_elem_idx} NOT found in sorted list!")

# Now check point-in-tet manually
elem_nodes_idx = connectivity[smallest_elem_idx]
elem_nodes = node_positions[elem_nodes_idx]

print(f"\nElement nodes:")
for i in range(4):
    print(f"  Node {i}: [{elem_nodes[i,0]:.6f}, {elem_nodes[i,1]:.6f}, {elem_nodes[i,2]:.6f}]")

# Manual point-in-tet test
def point_in_tet_manual(p, v0, v1, v2, v3):
    """Manual point-in-tet test."""
    mat = np.column_stack([v1-v0, v2-v0, v3-v0])
    rhs = p - v0
    try:
        coords_123 = np.linalg.solve(mat, rhs)
        b1, b2, b3 = coords_123
        b0 = 1.0 - b1 - b2 - b3
        bary = np.array([b0, b1, b2, b3])

        tol = -1e-6
        inside = np.all(bary >= tol) and np.all(bary <= 1.0 + 1e-10)

        return inside, bary
    except np.linalg.LinAlgError:
        return False, None

inside, bary = point_in_tet_manual(
    centroid,
    elem_nodes[0], elem_nodes[1], elem_nodes[2], elem_nodes[3]
)

print(f"\nPoint-in-tet test:")
print(f"  Barycentric coords: [{bary[0]:.10f}, {bary[1]:.10f}, {bary[2]:.10f}, {bary[3]:.10f}]")
print(f"  Sum: {bary.sum():.15f}")
print(f"  Inside (tol=-1e-6): {inside}")

# Check with different tolerances
for tol in [0, -1e-7, -1e-6, -1e-5, -1e-4]:
    inside_tol = np.all(bary >= tol) and np.all(bary <= 1.0 - tol)
    print(f"  Inside (tol={tol:8.0e}): {inside_tol}")

print("="*80)
