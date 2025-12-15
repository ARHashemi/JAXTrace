#!/usr/bin/env python3
"""
Diagnose the Morton octree bug by tracing through a single element.

Strategy:
1. Pick a single element
2. Compute its Morton code
3. Build octree and see which leaf it goes to
4. Recompute Morton code for query position
5. Look up leaf via prefix table
6. Compare leaf IDs

This will reveal where the mismatch occurs.
"""

import numpy as np
import sys
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree

print("="*80)
print("MORTON OCTREE BUG DIAGNOSIS")
print("="*80)
print()

# Load mesh
from pathlib import Path
mesh_path = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_159.pvtu")
print(f"[1/4] Loading mesh: {mesh_path}")
node_positions, connectivity, _ = load_mesh_from_pvtu(mesh_path)
n_elements = connectivity.shape[0]
print(f"  Elements: {n_elements:,}")
print()

# Build octree
print("[2/4] Building octree structure...")
morton_struct = build_global_morton_octree(
    node_positions,
    connectivity,
    leaf_capacity=256,
    max_depth=21,
    verbose=True
)
print()

# Pick a test element (middle of sorted array for better coverage)
test_idx_sorted = len(morton_struct.elem_ids_sorted) // 2
test_elem_id = morton_struct.elem_ids_sorted[test_idx_sorted]
test_morton = morton_struct.morton_sorted[test_idx_sorted]

# Find which leaf this element belongs to
leaf_id_build = -1
for i in range(morton_struct.n_leaves):
    start = morton_struct.leaf_start[i]
    end = start + morton_struct.leaf_length[i]
    if start <= test_idx_sorted < end:
        leaf_id_build = i
        break

print(f"[3/4] Test element analysis:")
print(f"  Element ID: {test_elem_id}")
print(f"  Sorted index: {test_idx_sorted}")
print(f"  Morton code: 0x{test_morton:016X}")
print(f"  Leaf ID (build): {leaf_id_build}")
print()

# Compute centroid and re-encode Morton code
nodes = connectivity[test_elem_id]
centroid = node_positions[nodes].mean(axis=0)
print(f"  Centroid: ({centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f})")

# Manually encode Morton code for centroid (same as build)
normalized = (centroid - morton_struct.bbox_min) / (morton_struct.bbox_max - morton_struct.bbox_min)
normalized = np.clip(normalized, 0.0, 1.0)
grid_max = (2 ** 21) - 1
u = np.floor(normalized * grid_max).astype(np.uint32)

# FIX: Cast to uint64 BEFORE bit operations
u = u.astype(np.uint64)

morton_query = np.uint64(0)
for i in range(21):
    morton_query |= ((u[0] >> i) & 1) << (3*i + 0)
    morton_query |= ((u[1] >> i) & 1) << (3*i + 1)
    morton_query |= ((u[2] >> i) & 1) << (3*i + 2)

print(f"  Morton (query): 0x{morton_query:016X}")
print(f"  Morton match: {morton_query == test_morton}")
print()

# Lookup leaf via prefix table
table_depth = int(morton_struct.table_depth)
prefix_bits = table_depth * 3

# CURRENT METHOD (MSB extraction):
shift_msb = 63 - prefix_bits
prefix_msb = morton_query >> shift_msb
leaf_id_msb = morton_struct.prefix_table[prefix_msb] if prefix_msb < len(morton_struct.prefix_table) else -1

# ALTERNATIVE METHOD (LSB extraction):
prefix_mask = (1 << prefix_bits) - 1
prefix_lsb = morton_query & prefix_mask
leaf_id_lsb = morton_struct.prefix_table[prefix_lsb] if prefix_lsb < len(morton_struct.prefix_table) else -1

print(f"[4/4] Prefix table lookup:")
print(f"  Table depth: {table_depth}")
print(f"  Prefix bits: {prefix_bits}")
print()
print(f"  MSB extraction (current):")
print(f"    Shift: {shift_msb}")
print(f"    Prefix: 0x{prefix_msb:06X}")
print(f"    Leaf ID: {leaf_id_msb}")
print(f"    Match: {leaf_id_msb == leaf_id_build}")
print()
print(f"  LSB extraction (alternative):")
print(f"    Mask: 0x{prefix_mask:06X}")
print(f"    Prefix: 0x{prefix_lsb:06X}")
print(f"    Leaf ID: {leaf_id_lsb}")
print(f"    Match: {leaf_id_lsb == leaf_id_build}")
print()

print("="*80)
print("CONCLUSION")
print("="*80)

if leaf_id_msb == leaf_id_build:
    print("✅ MSB extraction is CORRECT (current implementation)")
    print("❌ LSB extraction is WRONG")
elif leaf_id_lsb == leaf_id_build:
    print("❌ MSB extraction is WRONG (current implementation)")
    print("✅ LSB extraction is CORRECT - should use LSB!")
else:
    print("❌ BOTH methods are WRONG - deeper issue in octree build/prefix table")
    print(f"   Expected leaf: {leaf_id_build}")
    print(f"   MSB lookup: {leaf_id_msb}")
    print(f"   LSB lookup: {leaf_id_lsb}")

print("="*80)
