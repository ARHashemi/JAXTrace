#!/usr/bin/env python3
"""
Detailed analysis of Element 1550192 to understand the tetrahedron-cell relationship.

Shows:
1. Tetrahedron vertices
2. All 8 octree cell cubes
3. Which cube corners coincide with tet vertices
4. Bounding box relationship
"""

import numpy as np
from pathlib import Path
from jaxtrace.gpu.search.mesh_aligned_octree_fast import (
    find_axis_aligned_edges_fast,
    compute_8cell_pattern,
)

# Load mesh
print("Loading mesh...")
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
print(f"Element {elem_id} Detailed Analysis")
print("="*80)

node_ids = connectivity[elem_id]
vertices = node_positions[node_ids]

print("\n1. TETRAHEDRON VERTICES:")
print("-"*80)
for i, v in enumerate(vertices):
    print(f"  v{i}: ({v[0]:12.8f}, {v[1]:12.8f}, {v[2]:12.8f})")

# Find axis-aligned edges
cell_size, level = find_axis_aligned_edges_fast(vertices)
print(f"\n2. CELL SIZE (from axis-aligned edges):")
print("-"*80)
print(f"  Cell dimensions: ({cell_size[0]:.8f}, {cell_size[1]:.8f}, {cell_size[2]:.8f})")
print(f"  Level: {level}")

# Compute bbox
bbox_min = vertices.min(axis=0)
bbox_max = vertices.max(axis=0)

print(f"\n3. BOUNDING BOX:")
print("-"*80)
print(f"  Min: ({bbox_min[0]:.8f}, {bbox_min[1]:.8f}, {bbox_min[2]:.8f})")
print(f"  Max: ({bbox_max[0]:.8f}, {bbox_max[1]:.8f}, {bbox_max[2]:.8f})")
print(f"  Size: ({bbox_max[0]-bbox_min[0]:.8f}, {bbox_max[1]-bbox_min[1]:.8f}, {bbox_max[2]-bbox_min[2]:.8f})")
print(f"\n  Bbox size relative to cell size:")
print(f"    X: {(bbox_max[0]-bbox_min[0])/cell_size[0]:.3f} cells")
print(f"    Y: {(bbox_max[1]-bbox_min[1])/cell_size[1]:.3f} cells")
print(f"    Z: {(bbox_max[2]-bbox_min[2])/cell_size[2]:.3f} cells")

# Compute 8-cell pattern
grid_indices, morton_codes, level_out = compute_8cell_pattern(bbox_min, bbox_max, cell_size)

print(f"\n4. OCTREE CELLS (8 cubes in 2×2×2 pattern):")
print("-"*80)
print(f"  Number of cells: {len(grid_indices)}")

for idx, grid_idx in enumerate(grid_indices):
    # Compute cube bounds
    x_min = grid_idx[0] * cell_size[0]
    y_min = grid_idx[1] * cell_size[1]
    z_min = grid_idx[2] * cell_size[2]

    x_max = x_min + cell_size[0]
    y_max = y_min + cell_size[1]
    z_max = z_min + cell_size[2]

    print(f"\n  Cell {idx}: Grid=({grid_idx[0]:3d}, {grid_idx[1]:3d}, {grid_idx[2]:3d})")
    print(f"    X: [{x_min:.8f}, {x_max:.8f}]")
    print(f"    Y: [{y_min:.8f}, {y_max:.8f}]")
    print(f"    Z: [{z_min:.8f}, {z_max:.8f}]")

    # Check which tet vertices are inside or on boundary of this cell
    corners = [
        (x_min, y_min, z_min), (x_max, y_min, z_min),
        (x_max, y_max, z_min), (x_min, y_max, z_min),
        (x_min, y_min, z_max), (x_max, y_min, z_max),
        (x_max, y_max, z_max), (x_min, y_max, z_max),
    ]

    # Check which tet vertices coincide with cell corners
    coincident = []
    for vi, v in enumerate(vertices):
        for ci, c in enumerate(corners):
            if np.allclose(v, c, atol=1e-10):
                coincident.append(f"v{vi}=corner{ci}")

    # Check which tet vertices are strictly inside
    inside = []
    on_boundary = []
    outside = []

    for vi, v in enumerate(vertices):
        eps = 1e-10
        # Strictly inside: all coordinates strictly between min and max
        if (x_min + eps < v[0] < x_max - eps and
            y_min + eps < v[1] < y_max - eps and
            z_min + eps < v[2] < z_max - eps):
            inside.append(vi)
        # On boundary: at least one coordinate equals min or max
        elif (x_min - eps <= v[0] <= x_max + eps and
              y_min - eps <= v[1] <= y_max + eps and
              z_min - eps <= v[2] <= z_max + eps):
            on_boundary.append(vi)
        else:
            outside.append(vi)

    if coincident:
        print(f"    Coincident vertices: {', '.join(coincident)}")
    if on_boundary:
        print(f"    Vertices on boundary: v{on_boundary}")
    if inside:
        print(f"    Vertices strictly inside: v{inside}")

print(f"\n5. UNDERSTANDING THE RELATIONSHIP:")
print("-"*80)
print("The tetrahedron has:")
print("  - 3 axis-aligned edges (parallel to X, Y, Z)")
print("  - 1 diagonal edge (crossing through space)")
print()
print("Each axis-aligned edge has length = 1 cell size")
print("But the tetrahedron's bbox spans 2 cells in each dimension because:")
print("  - The diagonal edge causes the bbox to extend across cell boundaries")
print()
print("Result: The tet overlaps with 2×2×2 = 8 octree cells")
print("="*80)
