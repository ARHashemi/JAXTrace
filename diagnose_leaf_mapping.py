#!/usr/bin/env python3
"""
Diagnose position→leaf mapping issue.

The linear approximation assumes Morton codes are uniformly distributed,
but real meshes have non-uniform distributions.
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.search.morton_global_builder import build_global_morton_structure
from jaxtrace.gpu.search.morton_global_search import (
    morton_encode_position_jax,
    position_to_leaf_id_linear,
    upload_global_morton_to_gpu,
    search_in_leaf_global,
)

MESH_PATH = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_159.pvtu"

# Load mesh
print("Loading mesh...")
node_positions, connectivity, _ = load_mesh_from_pvtu(Path(MESH_PATH), field_name='Displacement')

# Build Morton
print("Building Morton structure...")
morton_struct = build_global_morton_structure(node_positions, connectivity, leaf_capacity=256, max_depth=21, verbose=False)
mesh_gpu = upload_global_morton_to_gpu(morton_struct, connectivity, node_positions)

print(f"\nMorton structure:")
print(f"  n_leaves: {int(mesh_gpu.n_leaves)}")
print(f"  morton_min: {int(mesh_gpu.morton_min)}")
print(f"  morton_max: {int(mesh_gpu.morton_max)}")

# Pick a random element and its centroid
test_elem_id = 1000
nodes = connectivity[test_elem_id]
centroid_np = node_positions[nodes].mean(axis=0)
centroid = jax.device_put(centroid_np.astype(np.float32))

print(f"\nTest element {test_elem_id}:")
print(f"  Centroid: {centroid_np}")

# Compute Morton code for centroid
morton_code = morton_encode_position_jax(
    centroid,
    mesh_gpu.bbox_min,
    mesh_gpu.bbox_max,
    mesh_gpu.max_depth
)
print(f"  Morton code: {int(morton_code)}")

# Find which leaf contains this element
elem_id_in_sorted = np.where(morton_struct.elem_ids_sorted == test_elem_id)[0]
if len(elem_id_in_sorted) > 0:
    idx = elem_id_in_sorted[0]
    actual_leaf_id = idx // 256
    print(f"  Element index in sorted list: {idx}")
    print(f"  Actual leaf ID: {actual_leaf_id}")

    # What does linear mapping predict?
    predicted_leaf_id = position_to_leaf_id_linear(centroid, mesh_gpu)
    print(f"  Predicted leaf ID (linear): {int(predicted_leaf_id)}")
    print(f"  Difference: {int(predicted_leaf_id) - actual_leaf_id}")

    # Try searching in predicted leaf
    found_in_predicted = search_in_leaf_global(centroid, predicted_leaf_id, mesh_gpu)
    print(f"  Found in predicted leaf: {int(found_in_predicted)}")

    # Try searching in actual leaf
    found_in_actual = search_in_leaf_global(centroid, jnp.int32(actual_leaf_id), mesh_gpu)
    print(f"  Found in actual leaf: {int(found_in_actual)}")

    # Analyze Morton distribution
    print("\n" + "=" * 80)
    print("Morton Distribution Analysis")
    print("=" * 80)

    # Check if Morton codes are uniformly distributed
    morton_sorted_np = np.array(morton_struct.morton_sorted, dtype=np.uint64)
    morton_range = morton_struct.morton_max - morton_struct.morton_min

    # Expected: each leaf has ~256 elements spanning morton_range / n_leaves
    expected_span_per_leaf = morton_range / morton_struct.n_leaves

    # Actual: check spacing
    morton_diffs = np.diff(morton_sorted_np.astype(np.int64))

    print(f"\nMorton code spacing:")
    print(f"  Expected span per leaf: {expected_span_per_leaf:.2e}")
    print(f"  Mean Morton diff: {np.mean(morton_diffs):.2e}")
    print(f"  Median Morton diff: {np.median(morton_diffs):.2e}")
    print(f"  Std Morton diff: {np.std(morton_diffs):.2e}")
    print(f"  Max Morton diff: {np.max(morton_diffs):.2e}")
    print(f"  Min Morton diff: {np.min(morton_diffs):.2e}")

    # Check distribution: cumulative Morton vs element index
    # If uniform, should be linear
    sample_indices = np.linspace(0, len(morton_sorted_np)-1, 1000).astype(int)
    sample_morton = morton_sorted_np[sample_indices]
    sample_normalized = (sample_morton - morton_struct.morton_min) / morton_range
    expected_normalized = sample_indices / len(morton_sorted_np)

    deviation = np.abs(sample_normalized - expected_normalized)
    print(f"\nLinear approximation error:")
    print(f"  Mean deviation: {np.mean(deviation):.4f}")
    print(f"  Max deviation: {np.max(deviation):.4f}")

    # Conclusion
    print("\n" + "=" * 80)
    if np.mean(deviation) > 0.05:
        print("❌ Morton codes are NOT uniformly distributed!")
        print("   Linear approximation is not suitable.")
        print("   Need to implement binary search or prefix table lookup.")
    else:
        print("✅ Morton codes are reasonably uniform.")
        print("   Linear approximation should work with small tolerance.")
    print("=" * 80)
