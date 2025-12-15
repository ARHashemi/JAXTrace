#!/usr/bin/env python3
"""
Small-scale bbox-based octree test.

Tests the bbox-based assignment logic with a small synthetic dataset.
"""

import numpy as np
from jaxtrace.gpu.search.octree_builder import build_octree_for_level, flatten_octree_to_arrays

# Create synthetic test data
print("Creating synthetic test data...")
print()

# 100 tetrahedral elements in a small cube
n_elements = 100
np.random.seed(42)

# Node positions (random in [0, 1]^3)
n_nodes = n_elements * 4  # 4 nodes per element (some shared, but doesn't matter for this test)
node_positions = np.random.rand(n_nodes, 3).astype(np.float32)

# Connectivity (each element has 4 unique nodes for simplicity)
connectivity = np.arange(n_nodes, dtype=np.int32).reshape(n_elements, 4)

# Element centroids
element_centroids = np.array([
    node_positions[connectivity[i]].mean(axis=0)
    for i in range(n_elements)
], dtype=np.float32)

# Element IDs
element_ids = np.arange(n_elements, dtype=np.int32)

# Dummy level field (all elements included)
level_field = np.ones(n_elements, dtype=np.float32) * 5.0

print(f"Test data:")
print(f"  Elements: {n_elements}")
print(f"  Nodes: {n_nodes}")
print()

# Build octree with bbox-based assignment
print("Building octree with bbox-based element assignment...")
print()

octree_nodes, octree_stats = build_octree_for_level(
    element_centroids=element_centroids,
    element_ids=element_ids,
    node_positions=node_positions,
    connectivity=connectivity,
    level_field=level_field,
    level_threshold=3.0,
    max_depth=5,
    max_leaf_size=10,
    use_levelset=False
)

print()
print("="*80)
print("OCTREE BUILD COMPLETE")
print("="*80)
from jaxtrace.gpu.search.octree_builder import print_octree_stats
print_octree_stats(octree_stats)
print()

# Flatten octree
octree_metadata, octree_elements = flatten_octree_to_arrays(octree_nodes)

print(f"Flattened arrays:")
print(f"  Metadata shape: {octree_metadata.shape}")
print(f"  Elements shape: {octree_elements.shape}")
print()

# Test: For each element, check if navigating its centroid leads to a leaf containing it
print("="*80)
print("BBOX-BASED ASSIGNMENT VERIFICATION")
print("="*80)
print()


def navigate_to_leaf(position: np.ndarray, octree_metadata: np.ndarray) -> int:
    """Navigate to leaf using octree search logic."""
    node_id = 0
    while True:
        is_leaf = octree_metadata[node_id, 0]
        if is_leaf == 1:
            return node_id

        bbox_min = octree_metadata[node_id, 1:4]
        bbox_max = octree_metadata[node_id, 4:7]
        bbox_mid = (bbox_min + bbox_max) / 2.0

        octant = (
            int(position[0] >= bbox_mid[0]) +
            (int(position[1] >= bbox_mid[1]) << 1) +
            (int(position[2] >= bbox_mid[2]) << 2)
        )

        first_child = int(octree_metadata[node_id, 7])
        node_id = first_child + octant


match_count = 0
mismatch_count = 0

for elem_id in range(n_elements):
    centroid = element_centroids[elem_id]

    # Navigate centroid to leaf
    navigated_leaf = navigate_to_leaf(centroid, octree_metadata)

    # Check if element is in that leaf
    leaf_elements = octree_elements[navigated_leaf]
    element_in_leaf = elem_id in leaf_elements

    if element_in_leaf:
        match_count += 1
    else:
        mismatch_count += 1
        if mismatch_count <= 3:
            # Get element bbox
            elem_nodes = node_positions[connectivity[elem_id]]
            elem_bbox_min = elem_nodes.min(axis=0)
            elem_bbox_max = elem_nodes.max(axis=0)

            # Get navigated leaf bbox
            leaf_bbox_min = octree_metadata[navigated_leaf, 1:4]
            leaf_bbox_max = octree_metadata[navigated_leaf, 4:7]

            # Check bbox intersection
            bbox_intersects = (
                (elem_bbox_min[0] <= leaf_bbox_max[0]) and (elem_bbox_max[0] >= leaf_bbox_min[0]) and
                (elem_bbox_min[1] <= leaf_bbox_max[1]) and (elem_bbox_max[1] >= leaf_bbox_min[1]) and
                (elem_bbox_min[2] <= leaf_bbox_max[2]) and (elem_bbox_max[2] >= leaf_bbox_min[2])
            )

            centroid_inside = (
                (centroid[0] >= leaf_bbox_min[0]) and (centroid[0] <= leaf_bbox_max[0]) and
                (centroid[1] >= leaf_bbox_min[1]) and (centroid[1] <= leaf_bbox_max[1]) and
                (centroid[2] >= leaf_bbox_min[2]) and (centroid[2] <= leaf_bbox_max[2])
            )

            print(f"MISMATCH {mismatch_count}: Element {elem_id}")
            print(f"  Centroid: {centroid}")
            print(f"  Element bbox: min={elem_bbox_min}, max={elem_bbox_max}")
            print(f"  Navigated to leaf: {navigated_leaf}")
            print(f"  Leaf bbox: min={leaf_bbox_min}, max={leaf_bbox_max}")
            print(f"  Centroid inside leaf bbox: {centroid_inside}")
            print(f"  Element bbox intersects leaf bbox: {bbox_intersects}")
            print(f"  Leaf contains: {leaf_elements[leaf_elements >= 0][:10]}...")
            print()

total = match_count + mismatch_count
print(f"Results:")
print(f"  Centroid navigates to correct leaf: {match_count}/{total} ({100*match_count/total:.1f}%)")
print(f"  Centroid navigates to wrong leaf: {mismatch_count}/{total} ({100*mismatch_count/total:.1f}%)")
print()

if match_count == total:
    print("✓ SUCCESS: Bbox-based assignment ensures all elements are findable by their centroids!")
else:
    print("✗ FAILURE: Some elements still not findable by their centroids")
