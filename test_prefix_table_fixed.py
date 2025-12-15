#!/usr/bin/env python3
"""
Test the complete prefix table fix with GPU search integration.

Tests:
1. Build octree with new prefix_start/prefix_length tables
2. Upload to GPU
3. Test position_to_leaf_id_octree() with centroid lookups
4. Verify accuracy improvement from 10.8% → >95%
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import (
    upload_global_morton_to_gpu,
    position_to_leaf_id_octree
)

print("="*80)
print("PREFIX TABLE FIX - FULL INTEGRATION TEST")
print("="*80)
print()

# Load mesh
mesh_path = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_159.pvtu")
print(f"[1/4] Loading mesh...")
node_positions, connectivity, _ = load_mesh_from_pvtu(mesh_path)
n_elements = connectivity.shape[0]
print(f"  Elements: {n_elements:,}")
print()

# Build octree
print("[2/4] Building octree with fixed prefix table...")
morton_struct = build_global_morton_octree(
    node_positions,
    connectivity,
    leaf_capacity=256,
    max_depth=21,
    verbose=True
)
print()

# Upload to GPU
print("[3/4] Uploading to GPU...")
mesh_gpu = upload_global_morton_to_gpu(
    morton_struct,
    connectivity,
    node_positions
)
print(f"  Uploaded {mesh_gpu.n_leaves} leaves to GPU")
print(f"  Prefix table: prefix_start.shape={mesh_gpu.prefix_start.shape}, prefix_length.shape={mesh_gpu.prefix_length.shape}")
print()

# Test with random sample of elements
print("[4/4] Testing centroid-based leaf lookup...")
n_test = min(10000, n_elements)
test_indices = np.random.choice(n_elements, n_test, replace=False)

success_count = 0
for idx in test_indices:
    # Compute centroid
    nodes = connectivity[idx]
    centroid = node_positions[nodes].mean(axis=0)
    centroid_jax = jnp.array(centroid, dtype=jnp.float32)

    # Find expected leaf (which leaf contains this element in Morton order)
    elem_sorted_idx = np.where(morton_struct.elem_ids_sorted == idx)[0][0]
    expected_leaf = -1
    for i in range(morton_struct.n_leaves):
        start = morton_struct.leaf_start[i]
        end = start + morton_struct.leaf_length[i]
        if start <= elem_sorted_idx < end:
            expected_leaf = i
            break

    # Query GPU function
    leaf_id = int(position_to_leaf_id_octree(centroid_jax, mesh_gpu))

    # Check if correct
    if leaf_id == expected_leaf:
        success_count += 1

accuracy = (success_count / n_test) * 100

print(f"  Tested: {n_test:,} elements")
print(f"  Success: {success_count:,}")
print(f"  Accuracy: {accuracy:.1f}%")
print()

if accuracy > 95:
    print("✅ SUCCESS! Accuracy >95% achieved!")
elif accuracy > 50:
    print("⚠️  Partial success - accuracy improved but not to target")
else:
    print("❌ FAILED - accuracy still low")

print("="*80)
