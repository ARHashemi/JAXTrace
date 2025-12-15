#!/usr/bin/env python3
"""
Test the fixed prefix table with prefix_start and prefix_length.
"""

import numpy as np
from pathlib import Path
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree

print("="*80)
print("PREFIX TABLE RANGE FIX TEST")
print("="*80)
print()

# Load mesh
mesh_path = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_159.pvtu")
print(f"[1/3] Loading mesh...")
node_positions, connectivity, _ = load_mesh_from_pvtu(mesh_path)
print()

# Build octree with NEW prefix table structure
print("[2/3] Building octree with fixed prefix table...")
morton_struct = build_global_morton_octree(
    node_positions,
    connectivity,
    leaf_capacity=256,
    max_depth=21,
    verbose=True
)
print()

# Test with the problematic prefix
target_prefix = 0x03124C
print(f"[3/3] Testing prefix 0x{target_prefix:06X}:")
print()

# Get leaf range from prefix table
leaf_start = morton_struct.prefix_start[target_prefix]
leaf_count = morton_struct.prefix_length[target_prefix]

print(f"  Prefix start: {leaf_start}")
print(f"  Prefix length: {leaf_count}")
print(f"  Leaf range: [{leaf_start}, {leaf_start + leaf_count})")
print()

# Expected leaf is 16115
expected_leaf = 16115
if leaf_start <= expected_leaf < leaf_start + leaf_count:
    print(f"✅ Expected leaf {expected_leaf} IS in the range!")
    print(f"   Leaf {expected_leaf} is at offset {expected_leaf - leaf_start} in the range")
else:
    print(f"❌ Expected leaf {expected_leaf} is NOT in range [{leaf_start}, {leaf_start + leaf_count})")

print()
print("="*80)
