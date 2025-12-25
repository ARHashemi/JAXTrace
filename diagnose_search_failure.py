#!/usr/bin/env python3
"""
Diagnose Why Search Fails in Refined Regions
=============================================

Investigate why initial_assignment_extended_batch returns -1 for particles
in refined element centroids.
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu
from jaxtrace.gpu.tracking.initial_assignment_extended import initial_assignment_extended_batch

print("="*80)
print("SEARCH FAILURE DIAGNOSIS")
print("="*80)

# Load mesh
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 120)
VELOCITY_FIELD_NAME = 'Displacement'

print("\n[1/3] Loading mesh...")
node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    MESH_BASE_PATH,
    MESH_FILE_PATTERN,
    VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    verbose=False
)

# Find one refined element
element_sizes = np.zeros(len(connectivity))
for i, elem_nodes_idx in enumerate(connectivity):
    elem_nodes = node_positions[elem_nodes_idx]
    max_edge = 0.0
    for j in range(4):
        for k in range(j+1, 4):
            edge_len = np.linalg.norm(elem_nodes[j] - elem_nodes[k])
            max_edge = max(max_edge, edge_len)
    element_sizes[i] = max_edge

# Find smallest element
smallest_elem_idx = np.argmin(element_sizes)
smallest_size = element_sizes[smallest_elem_idx]

print(f"\nSmallest element:")
print(f"  Element ID: {smallest_elem_idx}")
print(f"  Size: {smallest_size:.6e} m = {smallest_size*1000:.6f} mm")

# Get element info
elem_nodes_idx = connectivity[smallest_elem_idx]
elem_nodes = node_positions[elem_nodes_idx]
centroid = elem_nodes.mean(axis=0)

print(f"  Centroid: [{centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f}]")
print(f"  Node indices: {elem_nodes_idx}")

# Set up search
print("\n[2/3] Setting up search structures...")
mesh_gpu_connectivity = jax.device_put(connectivity)
mesh_gpu_node_positions = jax.device_put(node_positions)

element_neighbors = build_element_neighbors_array(connectivity)
mesh_gpu_element_neighbors = jax.device_put(element_neighbors)

morton_cpu = build_global_morton_octree(node_positions, connectivity, leaf_capacity=256)
mesh_gpu_global_morton = upload_global_morton_to_gpu(morton_cpu, connectivity, node_positions)

# Test with increasing search radii
print("\n[3/3] Testing search with increasing radii...")

test_pos = centroid.astype(np.float32)
test_pos_gpu = jax.device_put(np.array([test_pos]))

for radius in [1, 2, 5, 10, 20, 50, 100, 200]:
    found_elem = initial_assignment_extended_batch(
        test_pos_gpu,
        mesh_gpu_global_morton,
        max_radius=radius
    )[0]

    found_elem_cpu = int(found_elem)

    if found_elem_cpu >= 0:
        found_size = element_sizes[found_elem_cpu]
        correct = (found_elem_cpu == smallest_elem_idx)
        print(f"  radius={radius:3d}: Found elem {found_elem_cpu:7d} (size={found_size*1000:.4f}mm) "
              f"{'✅ CORRECT' if correct else '❌ WRONG'}")
    else:
        print(f"  radius={radius:3d}: NOT FOUND (elem_id = -1)")

print("="*80)
