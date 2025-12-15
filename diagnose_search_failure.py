#!/usr/bin/env python3
"""
Diagnose why search_L2_global_morton_single fails to find elements
even when position_to_leaf_id_octree finds the correct leaf.

Strategy:
1. Pick a single element
2. Compute its centroid
3. Find which leaf contains this element
4. Use position_to_leaf_id_octree to find leaf from centroid
5. Search within that leaf for the element
6. Test if centroid is actually inside the element (point-in-tet)

This will identify whether:
- A) The centroid is NOT inside its own element (geometry issue)
- B) The search_in_leaf_global is broken (implementation bug)
- C) The leaf lookup is wrong (already ruled out by test_prefix_table_fixed.py)
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import (
    upload_global_morton_to_gpu,
    position_to_leaf_id_octree,
    search_in_leaf_global,
    point_in_tet_gpu,
    search_L2_global_morton_single
)

print("=" * 80)
print("DIAGNOSE SEARCH FAILURE")
print("=" * 80)
print()

# Configuration
MESH_PATH = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_159.pvtu"

# Load mesh (same as test_global_morton_accuracy.py)
print("[1/4] Loading mesh...")
node_positions, connectivity, velocity_field = load_mesh_from_pvtu(
    Path(MESH_PATH),
    field_name='Displacement'
)
n_nodes = node_positions.shape[0]
n_elements = connectivity.shape[0]
print(f"  Mesh: {n_elements:,} elements, {n_nodes:,} nodes")
print()

# Build octree (same as test_global_morton_accuracy.py)
print("[2/4] Building global Morton structure (CPU)...")
morton_struct = build_global_morton_octree(
    node_positions=node_positions,
    connectivity=connectivity,
    leaf_capacity=256,
    max_depth=21,
    verbose=False
)
print(f"  Built {morton_struct.n_leaves:,} leaves")
print()

# Upload to GPU (same as test_global_morton_accuracy.py)
print("[3/4] Uploading mesh and Morton structure to GPU...")

# Compute element neighbors
element_neighbors = build_element_neighbors_array(connectivity)

# Upload standard mesh data
mesh_gpu = upload_mesh_to_gpu(
    connectivity=connectivity,
    node_positions=node_positions,
    element_neighbors=element_neighbors,
    verbose=False
)

# Upload global Morton structure
mesh_gpu_morton = upload_global_morton_to_gpu(
    morton_struct,
    connectivity,
    node_positions
)

# Force transfer
_ = jax.block_until_ready(mesh_gpu.connectivity)
_ = jax.block_until_ready(mesh_gpu_morton.elem_ids_sorted)

print(f"  Upload complete")
print()

# Test a few random elements
print("[4/4] Testing element search...")
print()

np.random.seed(42)
test_elements = np.random.choice(n_elements, 10, replace=False)

for test_elem_id in test_elements:
    print(f"Element {test_elem_id}:")

    # 1. Compute centroid
    nodes = connectivity[test_elem_id]
    centroid = node_positions[nodes].mean(axis=0)
    centroid_jax = jnp.array(centroid, dtype=jnp.float32)

    # 2. Find which leaf contains this element
    elem_sorted_idx = np.where(morton_struct.elem_ids_sorted == test_elem_id)[0][0]
    expected_leaf = -1
    for i in range(morton_struct.n_leaves):
        start = morton_struct.leaf_start[i]
        end = start + morton_struct.leaf_length[i]
        if start <= elem_sorted_idx < end:
            expected_leaf = i
            break

    print(f"  Expected leaf: {expected_leaf}")

    # 3. Use position_to_leaf_id_octree to find leaf from centroid
    found_leaf = int(position_to_leaf_id_octree(centroid_jax, mesh_gpu_morton))
    print(f"  Found leaf:    {found_leaf}")
    print(f"  Leaf match:    {'✅' if found_leaf == expected_leaf else '❌'}")

    # 4. Test if centroid is inside its own element
    is_inside = point_in_tet_gpu(
        centroid_jax,
        jnp.int32(test_elem_id),
        mesh_gpu_morton.connectivity,
        mesh_gpu_morton.node_positions
    )
    print(f"  Centroid inside element: {'✅ YES' if is_inside else '❌ NO'}")

    # 5. Search within the correct leaf
    found_elem_in_leaf = int(search_in_leaf_global(centroid_jax, jnp.int32(expected_leaf), mesh_gpu_morton))
    print(f"  Found in leaf search:    {found_elem_in_leaf} (expected {test_elem_id})")
    print(f"  Leaf search match:       {'✅' if found_elem_in_leaf == test_elem_id else '❌'}")

    # 6. Full L2 search with radius=4
    found_elem_l2 = int(search_L2_global_morton_single(centroid_jax, mesh_gpu_morton, jnp.int32(4)))
    print(f"  L2 search (radius=4):    {found_elem_l2} (expected {test_elem_id})")
    print(f"  L2 search match:         {'✅' if found_elem_l2 == test_elem_id else '❌'}")

    # Additional diagnostics if not found
    if found_elem_l2 == -1:
        # Check all elements in the expected leaf
        leaf_start = morton_struct.leaf_start[expected_leaf]
        leaf_length = morton_struct.leaf_length[expected_leaf]
        print(f"\n  Leaf {expected_leaf} details:")
        print(f"    Start: {leaf_start}, Length: {leaf_length}")
        print(f"    Elements in leaf: {morton_struct.elem_ids_sorted[leaf_start:leaf_start+leaf_length][:10]}...")

        # Check if ANY element in the leaf contains the centroid
        found_any = False
        for j in range(leaf_length):
            elem_id = morton_struct.elem_ids_sorted[leaf_start + j]
            is_in = point_in_tet_gpu(
                centroid_jax,
                jnp.int32(elem_id),
                mesh_gpu_morton.connectivity,
                mesh_gpu_morton.node_positions
            )
            if is_in:
                print(f"    ✅ Centroid IS inside element {elem_id} in this leaf")
                found_any = True
                break

        if not found_any:
            print(f"    ❌ Centroid is NOT inside ANY element in leaf {expected_leaf}")

    print()

print("=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
