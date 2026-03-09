#!/usr/bin/env python3
"""
Test cell construction for a single element to verify algorithm correctness.

Uses Element 1550192 from diagnostic as test case.
"""

import numpy as np
from jaxtrace.gpu.search.mesh_aligned_octree_fast import (
    find_axis_aligned_edges_fast,
    compute_8cell_pattern,
)

# Load mesh
print("Loading mesh...")
from pathlib import Path
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME = 'Displacement'

node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    base_path=MESH_BASE_PATH,
    file_pattern=MESH_FILE_PATTERN,
    timestep_range=VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    verbose=False
)

node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
    node_positions,
    connectivity,
    velocity_sequence=velocity_sequence,
    verbose=False
)

print(f"  Loaded {connectivity.shape[0]:,} elements, {node_positions.shape[0]:,} nodes\n")

# Test Element 1550192
elem_id = 1550192
print(f"Testing Element {elem_id}:")
print("="*80)

node_ids = connectivity[elem_id]
vertices = node_positions[node_ids]

print("\nVertices:")
for i, v in enumerate(vertices):
    print(f"  v{i}: ({v[0]:12.8f}, {v[1]:12.8f}, {v[2]:12.8f})")

# Find axis-aligned edges
cell_size, level = find_axis_aligned_edges_fast(vertices)
print(f"\nAxis-aligned edges (cell size):")
print(f"  X: {cell_size[0]:.8f}")
print(f"  Y: {cell_size[1]:.8f}")
print(f"  Z: {cell_size[2]:.8f}")
print(f"  Level: {level}")

# Compute bbox
bbox_min = vertices.min(axis=0)
bbox_max = vertices.max(axis=0)

print(f"\nBounding box:")
print(f"  Min: ({bbox_min[0]:.8f}, {bbox_min[1]:.8f}, {bbox_min[2]:.8f})")
print(f"  Max: ({bbox_max[0]:.8f}, {bbox_max[1]:.8f}, {bbox_max[2]:.8f})")

# Compute grid indices using current implementation
cell_size_safe = np.where(cell_size > 1e-6, cell_size, 1.0)

i_min = int(np.floor(bbox_min[0] / cell_size_safe[0]))
i_max = int(np.floor(bbox_max[0] / cell_size_safe[0]))
j_min = int(np.floor(bbox_min[1] / cell_size_safe[1]))
j_max = int(np.floor(bbox_max[1] / cell_size_safe[1]))
k_min = int(np.floor(bbox_min[2] / cell_size_safe[2]))
k_max = int(np.floor(bbox_max[2] / cell_size_safe[2]))

print(f"\nGrid index ranges (floor division):")
print(f"  i: [{i_min}, {i_max}]  (span: {i_max - i_min + 1})")
print(f"  j: [{j_min}, {j_max}]  (span: {j_max - j_min + 1})")
print(f"  k: [{k_min}, {k_max}]  (span: {k_max - k_min + 1})")
print(f"  Total cells: {(i_max - i_min + 1) * (j_max - j_min + 1) * (k_max - k_min + 1)}")

# Current implementation
print(f"\nCurrent implementation (mesh_aligned_octree_fast.py):")
i_range = [i_min, i_max] if i_max > i_min else [i_min, i_min]
j_range = [j_min, j_max] if j_max > j_min else [j_min, j_min]
k_range = [k_min, k_max] if k_max > k_min else [k_min, k_min]

print(f"  i_range: {i_range}")
print(f"  j_range: {j_range}")
print(f"  k_range: {k_range}")

current_cells = [
    [i, j, k]
    for i in i_range
    for j in j_range
    for k in k_range
]

print(f"\n  Generated cells:")
for idx, cell in enumerate(current_cells):
    print(f"    Cell {idx}: grid=({cell[0]:3d}, {cell[1]:3d}, {cell[2]:3d})")

print(f"  Total: {len(current_cells)} cells")

# Correct implementation (diagnostic)
print(f"\nCorrect implementation (diagnose_mesh_octree_structure.py):")
correct_cells = [
    [i, j, k]
    for i in range(i_min, i_max + 1)
    for j in range(j_min, j_max + 1)
    for k in range(k_min, k_max + 1)
]

print(f"  Generated cells:")
for idx, cell in enumerate(correct_cells):
    print(f"    Cell {idx}: grid=({cell[0]:3d}, {cell[1]:3d}, {cell[2]:3d})")

print(f"  Total: {len(correct_cells)} cells")

# Check if they match
print(f"\n" + "="*80)
if len(current_cells) == len(correct_cells) and all(
    c1 == c2 for c1, c2 in zip(current_cells, correct_cells)
):
    print("✅ Implementations MATCH!")
else:
    print("❌ Implementations DIFFER!")
    print(f"   Current: {len(current_cells)} cells")
    print(f"   Correct: {len(correct_cells)} cells")

# Now test using compute_8cell_pattern function
print(f"\n" + "="*80)
print("Testing compute_8cell_pattern() function:")
grid_indices, morton_codes, level_out = compute_8cell_pattern(bbox_min, bbox_max, cell_size)

print(f"  Returned {len(grid_indices)} cells:")
for idx, grid_idx in enumerate(grid_indices):
    print(f"    Cell {idx}: grid=({grid_idx[0]:3d}, {grid_idx[1]:3d}, {grid_idx[2]:3d})")

if len(grid_indices) == len(correct_cells):
    print(f"\n✅ compute_8cell_pattern() returns correct count: {len(grid_indices)} cells")
else:
    print(f"\n❌ compute_8cell_pattern() returns WRONG count!")
    print(f"   Expected: {len(correct_cells)} cells")
    print(f"   Got:      {len(grid_indices)} cells")

print("="*80)
