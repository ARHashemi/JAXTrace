#!/usr/bin/env python3
"""
Minimal test - search ONE particle with and without neighbors
"""

import numpy as np
from pathlib import Path
import jax
import jax.numpy as jnp

from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import extract_octree_cells_single
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu
from jaxtrace.gpu.search.mesh_aligned_point_location import search_mesh_aligned_octree_single
from jaxtrace.gpu.search.mesh_aligned_octree_with_neighbor_table import (
    add_neighbor_table_to_octree,
    upload_octree_with_neighbors_to_gpu
)
from jaxtrace.gpu.search.mesh_aligned_search_with_neighbors import (
    search_with_precomputed_neighbors_single
)
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

# Load mesh (small portion)
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
node_positions, connectivity, _ = load_velocity_sequence_from_pvtu(
    base_path=MESH_BASE_PATH,
    file_pattern="featurelessAvtk_{timestep}.pvtu",
    timestep_range=(158, 159),
    field_name='Displacement',
    verbose=False
)
node_positions, connectivity, _, _ = deduplicate_nodes(
    node_positions, connectivity, velocity_sequence=None, verbose=False
)

# Extract octree
octree_cells = extract_octree_cells_single(node_positions, connectivity, verbose=False)

# Upload baseline
octree_baseline = upload_mesh_aligned_octree_to_gpu(connectivity, node_positions, octree_cells, verbose=False)

# Build neighbor table
octree_with_neighbors = add_neighbor_table_to_octree(octree_cells, verbose=False)
octree_neighbors = upload_octree_with_neighbors_to_gpu(connectivity, node_positions, octree_with_neighbors, verbose=False)

# Test ONE particle - use element centroid (guaranteed to be found)
elem_idx = 1000
elem_nodes = connectivity[elem_idx]
elem_positions = node_positions[elem_nodes]
test_pos = elem_positions.mean(axis=0).astype(np.float32)
test_pos_gpu = jnp.array(test_pos)

bbox_min = np.array(octree_baseline.bbox_min)
bbox_max = np.array(octree_baseline.bbox_max)

print(f"Test position: {test_pos}")
print(f"Bbox: [{bbox_min[0]:.6f}, {bbox_max[0]:.6f}] × [{bbox_min[1]:.6f}, {bbox_max[1]:.6f}] × [{bbox_min[2]:.6f}, {bbox_max[2]:.6f}]")
print()

# Test baseline
print("Baseline search...")
elem_baseline, tests_baseline = search_mesh_aligned_octree_single(test_pos_gpu, octree_baseline, max_tests=150)
elem_baseline = int(elem_baseline)
tests_baseline = int(tests_baseline)
print(f"  Found: elem {elem_baseline}, tests: {tests_baseline}")
print()

# Test with neighbors
print("Neighbor search...")
from jaxtrace.gpu.search.mesh_aligned_search_with_neighbors import search_multi_level_with_precomputed_neighbors
elem_neighbors, tests_neighbors = search_multi_level_with_precomputed_neighbors(
    test_pos_gpu, octree_neighbors, levels_to_try=(14, 13, 12, 11, 10, 9, 8, 7), max_tests_per_cell=20
)
elem_neighbors = int(elem_neighbors)
tests_neighbors = int(tests_neighbors)
print(f"  Found: elem {elem_neighbors}, tests: {tests_neighbors}")
print()

# Summary
print("=" * 80)
if elem_baseline >= 0 and elem_neighbors < 0:
    print("❌ ERROR: Baseline found it, neighbors didn't!")
elif elem_baseline < 0 and elem_neighbors >= 0:
    print("✅ Neighbors found it, baseline didn't (good!)")
elif elem_baseline >= 0 and elem_neighbors >= 0:
    if elem_baseline == elem_neighbors:
        print("✅ Both found same element")
    else:
        print("⚠️  Both found different elements (baseline={}, neighbors={})".format(elem_baseline, elem_neighbors))
else:
    print("Neither found it (particle in void)")
print("=" * 80)
