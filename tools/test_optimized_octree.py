#!/usr/bin/env python3
"""
Test optimized octree building with real mesh data.
"""

import sys
import time
sys.path.insert(0, '/home/arhashemi/Workspace/welding/JAXTrace')

from jaxtrace.fields.coarse_octree_builder import load_mesh_from_pvtu, build_coarse_octree

# Test with last refined mesh (780k cells)
filepath = "/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_2.pvtu"

print("Testing Optimized Octree Builder")
print("=" * 60)
print(f"Loading mesh: {filepath}")

start = time.time()
mesh = load_mesh_from_pvtu(filepath)
load_time = time.time() - start

print(f"  Loaded in {load_time:.2f}s")
print(f"  Points: {len(mesh.points):,}")
print(f"  Cells: {len(mesh.cells):,}")

print("\nBuilding coarse octree (6 levels)...")
start = time.time()
octree = build_coarse_octree(mesh, n_coarse_levels=6, max_cells_per_node=32)
build_time = time.time() - start

n_nodes = len(octree.node_centers)
memory_mb = octree.get_memory_size() / (1024**2)

print(f"  Built in {build_time:.2f}s")
print(f"  Nodes: {n_nodes:,}")
print(f"  Memory: {memory_mb:.2f} MB")

# Count nodes per level
import numpy as np
for level in range(6):
    count = np.sum(octree.node_levels == level)
    print(f"    Level {level}: {count:,} nodes")

print(f"\nTotal time: {load_time + build_time:.2f}s")

if build_time < 120:  # Should be under 2 minutes
    print("✓ Performance acceptable")
else:
    print("✗ Performance still slow")
