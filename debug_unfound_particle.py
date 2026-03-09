#!/usr/bin/env python3
"""
Debug why specific particles are not found.

Test one unfound particle and trace through the search logic manually.
"""

import numpy as np
from pathlib import Path
import sys

# Import JAX first
import jax
import jax.numpy as jnp

from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import (
    extract_octree_cells_single,
    encode_morton_3d_single,
)
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

print(f"{'='*80}")
print("Debug Unfound Particle")
print(f"{'='*80}\n")

# Load mesh
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME = 'Displacement'

print("Loading mesh...")
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

# Extract octree cells
print("Extracting octree cells...")
cells = extract_octree_cells_single(
    node_positions,
    connectivity,
    tolerance=1e-6,
    verbose=False
)

print(f"  Unique cells: {cells.n_cells:,}\n")

# Test unfound particle from v4 log
test_pos = np.array([-0.00752759, 0.02073286, -0.00268006], dtype=np.float32)

print(f"Testing particle at position: {test_pos}")
print(f"  X={test_pos[0]:.8f}, Y={test_pos[1]:.8f}, Z={test_pos[2]:.8f}\n")

# Manually compute what cells should be searched
print(f"{'='*80}")
print("Manual Cell Lookup")
print(f"{'='*80}\n")

# Base sizes from v4 code
base_x = 1.28
base_y = 1.3084106445312499
base_z = 1.28

offset = (1 << 19)  # 2^19
max_coord = (1 << 20)  # 2^20

for level in [14, 13, 12, 11, 10, 9, 8, 7]:
    cell_size_x = base_x / (2.0 ** level)
    cell_size_y = base_y / (2.0 ** level)
    cell_size_z = base_z / (2.0 ** level)

    i = int(np.floor(test_pos[0] / cell_size_x))
    j = int(np.floor(test_pos[1] / cell_size_y))
    k = int(np.floor(test_pos[2] / cell_size_z))

    i_morton = np.clip(i + offset, 0, max_coord - 1)
    j_morton = np.clip(j + offset, 0, max_coord - 1)
    k_morton = np.clip(k + offset, 0, max_coord - 1)

    morton = encode_morton_3d_single(i_morton, j_morton, k_morton, max_depth=21)

    # Search for this cell
    cell_key = (morton, level)

    # Check if cell exists
    level_mask = cells.cell_levels == level
    morton_mask = cells.cell_morton_codes == morton
    cell_exists = np.any(level_mask & morton_mask)

    if cell_exists:
        cell_idx = np.where(level_mask & morton_mask)[0][0]
        n_elements = cells.cell_to_elements_offsets[cell_idx + 1] - cells.cell_to_elements_offsets[cell_idx]
        print(f"Level {level:2d}: Grid=({i:6d}, {j:6d}, {k:6d}), Morton={morton:20d}, ✅ FOUND ({n_elements} elements)")
    else:
        print(f"Level {level:2d}: Grid=({i:6d}, {j:6d}, {k:6d}), Morton={morton:20d}, ❌ NOT FOUND")

# Now let's find which cells actually contain this position
print(f"\n{'='*80}")
print("Brute Force Search: Which elements contain this particle?")
print(f"{'='*80}\n")

def point_in_tet(pos, v0, v1, v2, v3):
    """Simple point-in-tet test."""
    # Compute barycentric coordinates
    v0p = pos - v0
    v01 = v1 - v0
    v02 = v2 - v0
    v03 = v3 - v0

    # Build matrix
    mat = np.array([v01, v02, v03]).T  # 3x3
    try:
        bary = np.linalg.solve(mat, v0p)  # (b1, b2, b3)
    except np.linalg.LinAlgError:
        return False

    b0 = 1.0 - bary[0] - bary[1] - bary[2]

    # Check if all barycentric coords are in [0, 1]
    tol = 1e-6
    return (b0 >= -tol and b0 <= 1.0 + tol and
            bary[0] >= -tol and bary[0] <= 1.0 + tol and
            bary[1] >= -tol and bary[1] <= 1.0 + tol and
            bary[2] >= -tol and bary[2] <= 1.0 + tol)

containing_elements = []
for elem_id in range(min(connectivity.shape[0], 100000)):  # Test first 100k elements
    node_ids = connectivity[elem_id]
    vertices = node_positions[node_ids]

    if point_in_tet(test_pos, vertices[0], vertices[1], vertices[2], vertices[3]):
        containing_elements.append(elem_id)
        if len(containing_elements) >= 5:  # Find up to 5 containing elements
            break

if containing_elements:
    print(f"Found {len(containing_elements)} containing element(s) in first 100k:")
    for elem_id in containing_elements:
        print(f"\n  Element {elem_id}:")

        # Get element info from cells
        # Find which cell this element belongs to
        elem_cell_idx = cells.element_to_cells[elem_id]
        cell_morton = cells.cell_morton_codes[elem_cell_idx]
        cell_level = cells.cell_levels[elem_cell_idx]
        cell_size = cells.cell_sizes[elem_cell_idx]
        cell_grid_idx = cells.cell_grid_indices[elem_cell_idx]

        print(f"    Cell level: {cell_level}")
        print(f"    Cell grid indices: {cell_grid_idx}")
        print(f"    Cell size: {cell_size}")
        print(f"    Cell morton: {cell_morton}")

        # Compute what grid indices the query SHOULD compute
        for test_level in [14, 13, 12, 11, 10, 9, 8]:
            if test_level == cell_level:
                cell_size_x_test = base_x / (2.0 ** test_level)
                cell_size_y_test = base_y / (2.0 ** test_level)
                cell_size_z_test = base_z / (2.0 ** test_level)

                i_test = int(np.floor(test_pos[0] / cell_size_x_test))
                j_test = int(np.floor(test_pos[1] / cell_size_y_test))
                k_test = int(np.floor(test_pos[2] / cell_size_z_test))

                print(f"\n    Query at level {test_level} computes grid: ({i_test}, {j_test}, {k_test})")
                print(f"    Actual element grid:                      ({cell_grid_idx[0]}, {cell_grid_idx[1]}, {cell_grid_idx[2]})")

                if (i_test, j_test, k_test) == tuple(cell_grid_idx):
                    print(f"    ✅ MATCH! Query should find this element.")
                else:
                    print(f"    ❌ MISMATCH! Query will not find this element.")
                    print(f"\n    Difference:")
                    print(f"      ΔX grid: {i_test - cell_grid_idx[0]}")
                    print(f"      ΔY grid: {j_test - cell_grid_idx[1]}")
                    print(f"      ΔZ grid: {k_test - cell_grid_idx[2]}")

                    # Compute what base sizes would make them match
                    # i_test = floor(pos[0] / (base_x / 2^L))
                    # We want: i_test = cell_grid_idx[0]
                    # So: floor(pos[0] * 2^L / base_x) = cell_grid_idx[0]
                    # This means: pos[0] * 2^L / base_x should be in [cell_grid_idx[0], cell_grid_idx[0] + 1)
                    # So: base_x should be in (pos[0] * 2^L / (cell_grid_idx[0] + 1), pos[0] * 2^L / cell_grid_idx[0]]

                    if cell_grid_idx[0] != 0:
                        correct_base_x_max = test_pos[0] * (2 ** test_level) / cell_grid_idx[0]
                        correct_base_x_min = test_pos[0] * (2 ** test_level) / (cell_grid_idx[0] + 1)
                        print(f"\n    For X to match, base_x should be in ({correct_base_x_min:.10f}, {correct_base_x_max:.10f}]")
                        print(f"    Current base_x: {base_x:.10f}")
                        print(f"    Actual element cell_size[0]: {cell_size[0]:.10f}")
                        print(f"    Inferred base from element: {cell_size[0] * (2 ** test_level):.10f}")

else:
    print("❌ No containing element found in first 100k elements!")
    print("   This particle is likely outside the mesh or in the later 2.9M elements.")

print(f"\n{'='*80}\n")
