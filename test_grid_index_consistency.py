#!/usr/bin/env python3
"""
Test if grid index computation is consistent between assignment and query.

Pick random elements, compute their grid indices during assignment,
then verify that query positions within those elements compute the same grid indices.
"""

import numpy as np
from pathlib import Path

# Import JAX
import jax

from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import (
    extract_octree_cells_single,
    find_axis_aligned_edges_single,
)
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

print(f"{'='*80}")
print("Grid Index Consistency Test")
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

# Test consistency for random elements at each level
print(f"{'='*80}")
print("Consistency Test: Do query positions compute correct grid indices?")
print(f"{'='*80}\n")

# Base sizes from v4 code
base_x = 1.28
base_y = 1.3084106445312499
base_z = 1.28

offset = (1 << 19)
tolerance = 1e-6

unique_levels = np.unique(cells.cell_levels)
n_samples_per_level = 10

total_tested = 0
total_matches = 0
total_mismatches = 0

for level in sorted(unique_levels):
    # Find elements at this level
    level_elems = []
    for elem_id in range(connectivity.shape[0]):
        elem_cell_idx = cells.element_to_cells[elem_id]
        if cells.cell_levels[elem_cell_idx] == level:
            level_elems.append(elem_id)
            if len(level_elems) >= n_samples_per_level:
                break

    if len(level_elems) == 0:
        continue

    print(f"Level {level:2d} (testing {len(level_elems)} elements):")

    level_matches = 0
    level_mismatches = 0

    for elem_id in level_elems:
        # Get element info from cells
        elem_cell_idx = cells.element_to_cells[elem_id]
        actual_grid_idx = cells.cell_grid_indices[elem_cell_idx]
        actual_cell_size = cells.cell_sizes[elem_cell_idx]

        # Get element vertices
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]

        # Pick center of element as test position
        test_pos = np.mean(vertices, axis=0)

        # Compute what the query would compute
        cell_size_x_query = base_x / (2.0 ** level)
        cell_size_y_query = base_y / (2.0 ** level)
        cell_size_z_query = base_z / (2.0 ** level)

        i_query = int(np.floor(test_pos[0] / cell_size_x_query))
        j_query = int(np.floor(test_pos[1] / cell_size_y_query))
        k_query = int(np.floor(test_pos[2] / cell_size_z_query))

        # Compare
        match = (i_query == actual_grid_idx[0] and
                 j_query == actual_grid_idx[1] and
                 k_query == actual_grid_idx[2])

        if match:
            level_matches += 1
            total_matches += 1
        else:
            level_mismatches += 1
            total_mismatches += 1

            # Print first mismatch details
            if level_mismatches == 1:
                print(f"  ❌ First mismatch (element {elem_id}):")
                print(f"     Position: {test_pos}")
                print(f"     Actual grid: ({actual_grid_idx[0]}, {actual_grid_idx[1]}, {actual_grid_idx[2]})")
                print(f"     Query grid:  ({i_query}, {j_query}, {k_query})")
                print(f"     Actual cell size: {actual_cell_size}")
                print(f"     Query cell size:  ({cell_size_x_query:.10f}, {cell_size_y_query:.10f}, {cell_size_z_query:.10f})")

                # Compute what base sizes would make it match
                if actual_grid_idx[0] != 0:
                    correct_base_x = actual_cell_size[0] * (2 ** level)
                    print(f"     Correct BASE_X: {correct_base_x:.15f} (current: {base_x:.15f})")

                if actual_grid_idx[1] != 0:
                    correct_base_y = actual_cell_size[1] * (2 ** level)
                    print(f"     Correct BASE_Y: {correct_base_y:.15f} (current: {base_y:.15f})")

                if actual_grid_idx[2] != 0:
                    correct_base_z = actual_cell_size[2] * (2 ** level)
                    print(f"     Correct BASE_Z: {correct_base_z:.15f} (current: {base_z:.15f})")

        total_tested += 1

    match_rate = 100.0 * level_matches / len(level_elems)
    if level_mismatches == 0:
        print(f"  ✅ {level_matches}/{len(level_elems)} matches ({match_rate:.1f}%)")
    else:
        print(f"  ⚠  {level_matches}/{len(level_elems)} matches, {level_mismatches} mismatches ({match_rate:.1f}%)")

    print()

# Summary
print(f"{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}\n")

overall_match_rate = 100.0 * total_matches / total_tested if total_tested > 0 else 0
print(f"Total tested: {total_tested}")
print(f"Matches: {total_matches} ({overall_match_rate:.1f}%)")
print(f"Mismatches: {total_mismatches} ({100.0 - overall_match_rate:.1f}%)")
print()

if total_mismatches == 0:
    print("✅ Grid index computation is CONSISTENT!")
    print("   Query positions should find correct cells.")
else:
    print(f"❌ Grid index computation is INCONSISTENT!")
    print(f"   {total_mismatches} elements will not be found during search.")
    print(f"   This explains the low searchability!")

print(f"\n{'='*80}\n")
