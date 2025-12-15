#!/usr/bin/env python3
"""
Debug prefix table construction for specific leaves.

Strategy:
1. Build octree
2. Find leaf 16115 (the expected leaf)
3. Find leaf 16120 (the lookup result)
4. Trace how both leaves fill the prefix table
5. Identify which leaf overwrites which prefix
"""

import numpy as np
from pathlib import Path
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree

print("="*80)
print("PREFIX TABLE FILL DEBUG")
print("="*80)
print()

# Load mesh
mesh_path = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_159.pvtu")
print(f"[1/3] Loading mesh...")
node_positions, connectivity, _ = load_mesh_from_pvtu(mesh_path)
print()

# Build octree
print("[2/3] Building octree...")
morton_struct = build_global_morton_octree(
    node_positions,
    connectivity,
    leaf_capacity=256,
    max_depth=21,
    verbose=False
)
print(f"  Built {morton_struct.n_leaves:,} leaves")
print()

# Get target prefix
target_prefix = 0x03124C
print(f"[3/3] Analyzing prefix 0x{target_prefix:06X}:")
print()

# Manually rebuild prefix table with logging
table_depth = int(morton_struct.table_depth)
table_size = 8 ** table_depth
prefix_table_debug = np.full(table_size, -1, dtype=np.int32)

# Track which leaf writes to target prefix
writes_to_target = []

# Reconstruct leaves from arrays
from jaxtrace.gpu.search.morton_octree_builder import build_adaptive_octree_leaves

leaves, _ = build_adaptive_octree_leaves(
    morton_struct.morton_sorted,
    morton_struct.elem_ids_sorted,
    leaf_capacity=256,
    max_depth=21
)

print(f"Analyzing {len(leaves):,} leaves...")
print()

# Fill prefix table with logging
for leaf_id, leaf in enumerate(leaves):
    leaf_depth = leaf.prefix_bits // 3

    if leaf_depth >= table_depth:
        # Deep leaf: extract prefix
        shift = leaf.prefix_bits - (table_depth * 3)
        prefix = leaf.morton_prefix >> shift

        # Check if this affects target prefix
        if prefix == target_prefix:
            writes_to_target.append({
                'leaf_id': leaf_id,
                'leaf_depth': leaf_depth,
                'leaf_prefix': leaf.morton_prefix,
                'leaf_prefix_bits': leaf.prefix_bits,
                'table_prefix': prefix,
                'overwrites': prefix_table_debug[prefix]
            })
            prefix_table_debug[prefix] = leaf_id
    else:
        # Shallow leaf: fill descendants
        n_descendants = 8 ** (table_depth - leaf_depth)
        base_prefix = leaf.morton_prefix << ((table_depth - leaf_depth) * 3)

        for i in range(n_descendants):
            prefix = base_prefix + i

            # Check if this affects target prefix
            if prefix == target_prefix:
                writes_to_target.append({
                    'leaf_id': leaf_id,
                    'leaf_depth': leaf_depth,
                    'leaf_prefix': leaf.morton_prefix,
                    'leaf_prefix_bits': leaf.prefix_bits,
                    'table_prefix': prefix,
                    'descendant_idx': i,
                    'overwrites': prefix_table_debug[prefix]
                })
                prefix_table_debug[prefix] = leaf_id

# Print results
print(f"Prefix 0x{target_prefix:06X} writes:")
print()

if not writes_to_target:
    print("  ❌ NO WRITES - prefix not covered by any leaf!")
else:
    for write in writes_to_target:
        print(f"  Leaf {write['leaf_id']:,}:")
        print(f"    Depth: {write['leaf_depth']}")
        print(f"    Morton prefix: 0x{write['leaf_prefix']:X} ({write['leaf_prefix_bits']} bits)")
        if 'descendant_idx' in write:
            print(f"    Type: Shallow leaf (descendant {write['descendant_idx']})")
        else:
            print(f"    Type: Deep leaf")
        if write['overwrites'] != -1:
            print(f"    ⚠️  Overwrites leaf {write['overwrites']}")
        print()

print(f"Final value: prefix_table[0x{target_prefix:06X}] = {prefix_table_debug[target_prefix]}")
print(f"Expected: 16115")
print(f"Actual:   {prefix_table_debug[target_prefix]}")
print()

# Additional analysis: examine leaves 16115 and 16120
print("="*80)
print("LEAF COMPARISON")
print("="*80)
print()

for leaf_id in [16115, 16120]:
    if leaf_id < len(leaves):
        leaf = leaves[leaf_id]
        depth = leaf.prefix_bits // 3
        print(f"Leaf {leaf_id}:")
        print(f"  Start index: {leaf.start_idx:,}")
        print(f"  Length: {leaf.length}")
        print(f"  Depth: {depth}")
        print(f"  Morton prefix: 0x{leaf.morton_prefix:X} ({leaf.prefix_bits} bits)")

        # Show what prefixes this leaf should cover
        if depth >= table_depth:
            shift = leaf.prefix_bits - (table_depth * 3)
            prefix = leaf.morton_prefix >> shift
            print(f"  Maps to prefix: 0x{prefix:06X}")
        else:
            n_descendants = 8 ** (table_depth - depth)
            base_prefix = leaf.morton_prefix << ((table_depth - depth) * 3)
            print(f"  Maps to prefixes: 0x{base_prefix:06X} - 0x{base_prefix + n_descendants - 1:06X}")
        print()

print("="*80)
