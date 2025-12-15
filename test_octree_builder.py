#!/usr/bin/env python3
"""
Unit test for octree builder.

Tests:
1. Basic octree construction
2. Level-based filtering
3. Fixed-size node arrays
4. Memory estimates
"""

import numpy as np
from jaxtrace.gpu.search.octree_builder import (
    build_octree_for_level,
    flatten_octree_to_arrays,
    print_octree_stats
)

print("=" * 80)
print("OCTREE BUILDER UNIT TEST")
print("=" * 80)
print()

# Test 1: Basic construction
print("Test 1: Basic octree construction")
print("-" * 80)

# Generate synthetic element data
N_ELEMENTS = 10_000
np.random.seed(42)

element_centroids = np.random.rand(N_ELEMENTS, 3).astype(np.float32)
element_ids = np.arange(N_ELEMENTS, dtype=np.int32)
level_field = np.random.randint(0, 10, N_ELEMENTS, dtype=np.int32)

print(f"Test data:")
print(f"  Elements: {N_ELEMENTS:,}")
print(f"  Level range: {level_field.min()} - {level_field.max()}")
print()

# Build octree (no filtering)
print("Building octree (no level filtering)...")
nodes, metadata = build_octree_for_level(
    element_centroids,
    element_ids,
    level_field=None,  # No filtering
    max_depth=8,
    max_leaf_size=100
)

print()
print_octree_stats(metadata)

# Verify
assert len(nodes) == metadata['n_nodes'], "Node count mismatch"
assert metadata['n_elements'] == N_ELEMENTS, "Element count mismatch (no filtering)"
assert metadata['max_depth'] <= 8, "Max depth exceeded"

print()
print("✅ Test 1 PASSED: Basic construction works")
print()

# Test 2: Level-based filtering
print("=" * 80)
print("Test 2: Level-based filtering")
print("-" * 80)

level_threshold = 7
print(f"Filtering elements with level >= {level_threshold}...")

nodes_filtered, metadata_filtered = build_octree_for_level(
    element_centroids,
    element_ids,
    level_field=level_field,
    level_threshold=level_threshold,
    max_depth=8,
    max_leaf_size=100
)

expected_count = (level_field >= level_threshold).sum()
print()
print(f"Expected filtered count: {expected_count:,}")
print(f"Actual filtered count: {metadata_filtered['n_elements']:,}")
print()
print_octree_stats(metadata_filtered)

# Verify
assert metadata_filtered['n_elements'] == expected_count, "Filtering failed"
assert metadata_filtered['n_elements'] < N_ELEMENTS, "No filtering applied"
assert len(nodes_filtered) < len(nodes), "Filtered tree should be smaller"

print()
print("✅ Test 2 PASSED: Level filtering works")
print()

# Test 3: Fixed-size arrays
print("=" * 80)
print("Test 3: Fixed-size node arrays")
print("-" * 80)

print("Flattening octree to GPU-compatible arrays...")

node_metadata, node_elements = flatten_octree_to_arrays(
    nodes_filtered,
    max_leaf_size=100
)

print()
print(f"Node metadata shape: {node_metadata.shape}")
print(f"Node elements shape: {node_elements.shape}")
print(f"Node metadata dtype: {node_metadata.dtype}")
print(f"Node elements dtype: {node_elements.dtype}")

# Verify shapes
assert node_metadata.shape == (len(nodes_filtered), 15), "Metadata shape wrong"
assert node_elements.shape == (len(nodes_filtered), 100), "Elements shape wrong"
assert node_metadata.dtype == np.float32, "Metadata dtype wrong"
assert node_elements.dtype == np.int32, "Elements dtype wrong"

# Verify contents
for i, node in enumerate(nodes_filtered):
    # Check is_leaf
    assert node_metadata[i, 0] == (1.0 if node.is_leaf else 0.0), f"is_leaf mismatch at {i}"

    # Check bbox
    assert np.allclose(node_metadata[i, 1:4], node.bbox_min), f"bbox_min mismatch at {i}"
    assert np.allclose(node_metadata[i, 4:7], node.bbox_max), f"bbox_max mismatch at {i}"

    # Check children
    assert np.allclose(node_metadata[i, 7:15], node.children.astype(np.float32)), f"children mismatch at {i}"

    # Check elements
    assert np.array_equal(node_elements[i], node.elements), f"elements mismatch at {i}"

print()
print("✅ Test 3 PASSED: Array flattening works")
print()

# Test 4: Memory estimates
print("=" * 80)
print("Test 4: Memory estimates")
print("-" * 80)

# Calculate actual memory
metadata_mem_mb = node_metadata.nbytes / (1024 ** 2)
elements_mem_mb = node_elements.nbytes / (1024 ** 2)
total_mem_mb = metadata_mem_mb + elements_mem_mb

print(f"Actual memory:")
print(f"  Metadata: {metadata_mem_mb:.2f} MB")
print(f"  Elements: {elements_mem_mb:.2f} MB")
print(f"  Total: {total_mem_mb:.2f} MB")
print()
print(f"Estimated memory: {metadata_filtered['memory_mb']:.2f} MB")

# Verify (estimate should be within 50% of actual)
ratio = metadata_filtered['memory_mb'] / total_mem_mb
assert 0.5 < ratio < 1.5, f"Memory estimate off by {ratio:.2f}×"

print()
print("✅ Test 4 PASSED: Memory estimates reasonable")
print()

# Test 5: Stress test (larger mesh)
print("=" * 80)
print("Test 5: Stress test (300k elements)")
print("-" * 80)

N_LARGE = 300_000
print(f"Generating {N_LARGE:,} elements...")

centroids_large = np.random.rand(N_LARGE, 3).astype(np.float32)
ids_large = np.arange(N_LARGE, dtype=np.int32)
levels_large = np.random.randint(0, 10, N_LARGE, dtype=np.int32)

print("Building octree...")
import time
t_start = time.time()

nodes_large, metadata_large = build_octree_for_level(
    centroids_large,
    ids_large,
    level_field=levels_large,
    level_threshold=7,
    max_depth=10,
    max_leaf_size=500
)

t_build = time.time() - t_start

print()
print(f"Build time: {t_build:.2f} s")
print()
print_octree_stats(metadata_large)

# Verify reasonable tree structure
assert metadata_large['n_nodes'] < N_LARGE, "Tree should have fewer nodes than elements"
assert metadata_large['n_leaves'] > 0, "Tree should have leaves"
assert metadata_large['max_depth'] <= 10, "Max depth exceeded"
assert metadata_large['memory_mb'] < 50, f"Memory too high: {metadata_large['memory_mb']:.1f} MB"

print()
print("✅ Test 5 PASSED: Stress test successful")
print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("✅ ALL TESTS PASSED")
print()
print("Octree builder verified:")
print("  - Basic construction: ✓")
print("  - Level filtering: ✓")
print("  - Fixed-size arrays: ✓")
print("  - Memory estimates: ✓")
print("  - Stress test (300k elements): ✓")
print()
print("Ready for GPU integration.")
print()
