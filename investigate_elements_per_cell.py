#!/usr/bin/env python3
"""
Investigate why elements per cell = 12.27 instead of expected 5-6.

Key observations to verify:
1. Are cells with 6 elements single cubes with standard Kuhn subdivision?
2. Are cells with 12 elements shared across refinement boundaries?
3. Do all elements in a cell truly share the same parent cube?
4. Does 1 cell per element guarantee 100% searchability?
"""

import numpy as np
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, '/home/arhashemi/Workspace/welding/JAXTrace')

# Import directly to avoid JAX dependency in __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "mesh_aligned_octree_single_cell",
    "/home/arhashemi/Workspace/welding/JAXTrace/jaxtrace/gpu/search/mesh_aligned_octree_single_cell.py"
)
octree_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(octree_module)

extract_octree_cells_single = octree_module.extract_octree_cells_single
find_axis_aligned_edges_single = octree_module.find_axis_aligned_edges_single
find_parent_cube = octree_module.find_parent_cube

# These don't depend on JAX
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

# Load mesh
print("Loading mesh...")
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

# Extract cells
print("Extracting cells with detailed tracking...")
cells = extract_octree_cells_single(
    node_positions,
    connectivity,
    tolerance=1e-6,
    verbose=False
)

print(f"\n{'='*80}")
print("INVESTIGATION: Why 12.27 elements per cell?")
print(f"{'='*80}\n")

# Find cells with different element counts
elements_per_cell = np.diff(cells.cell_to_elements_offsets)

# Get indices for cells with 6, 12, 24 elements
cells_with_6 = np.where(elements_per_cell == 6)[0]
cells_with_12 = np.where(elements_per_cell == 12)[0]
cells_with_24 = np.where(elements_per_cell == 24)[0]

print(f"Sample cells:")
print(f"  6 elements: {len(cells_with_6):,} cells")
print(f"  12 elements: {len(cells_with_12):,} cells")
print(f"  24 elements: {len(cells_with_24):,} cells")


def analyze_cell(cell_idx, label):
    """Analyze a specific cell in detail."""
    print(f"\n{'-'*80}")
    print(f"{label}: Cell {cell_idx}")
    print(f"{'-'*80}")

    # Get cell properties
    morton = cells.cell_morton_codes[cell_idx]
    level = cells.cell_levels[cell_idx]
    cell_size = cells.cell_sizes[cell_idx]
    grid_idx = cells.cell_grid_indices[cell_idx]

    print(f"  Morton: {morton}")
    print(f"  Level: {level}")
    print(f"  Cell size: ({cell_size[0]:.10f}, {cell_size[1]:.10f}, {cell_size[2]:.10f})")
    print(f"  Grid indices: ({grid_idx[0]}, {grid_idx[1]}, {grid_idx[2]})")

    # Get elements in this cell
    start = cells.cell_to_elements_offsets[cell_idx]
    end = cells.cell_to_elements_offsets[cell_idx + 1]
    elem_ids = cells.cell_to_elements_data[start:end]

    print(f"  Elements in cell: {len(elem_ids)}")

    # Analyze each element
    cell_corner = grid_idx * cell_size

    print(f"\n  Cell corner: ({cell_corner[0]:.10f}, {cell_corner[1]:.10f}, {cell_corner[2]:.10f})")

    # Check if all elements truly belong to this cell
    cell_sizes_found = []
    parent_cubes_found = []

    for i, elem_id in enumerate(elem_ids):
        vertices = node_positions[connectivity[elem_id]]

        # Find this element's cell size and parent cube
        elem_cell_size, elem_level = find_axis_aligned_edges_single(vertices, 1e-6)
        elem_cube_corner, elem_cube_center, i_idx, j_idx, k_idx = find_parent_cube(
            vertices, elem_cell_size, 1e-6
        )

        cell_sizes_found.append(elem_cell_size)
        parent_cubes_found.append((i_idx, j_idx, k_idx))

        if i < 3 or not np.allclose(elem_cell_size, cell_size, atol=1e-9):
            print(f"    Element {elem_id}:")
            print(f"      Cell size: ({elem_cell_size[0]:.10f}, {elem_cell_size[1]:.10f}, {elem_cell_size[2]:.10f})")
            print(f"      Grid indices: ({i_idx}, {j_idx}, {k_idx})")
            print(f"      Cube corner: ({elem_cube_corner[0]:.10f}, {elem_cube_corner[1]:.10f}, {elem_cube_corner[2]:.10f})")

            if not np.allclose(elem_cell_size, cell_size, atol=1e-9):
                print(f"      ⚠️  CELL SIZE MISMATCH!")

    # Check uniqueness of cell sizes
    unique_cell_sizes = []
    for cs in cell_sizes_found:
        is_unique = True
        for ucs in unique_cell_sizes:
            if np.allclose(cs, ucs, atol=1e-9):
                is_unique = False
                break
        if is_unique:
            unique_cell_sizes.append(cs)

    print(f"\n  Unique cell sizes found: {len(unique_cell_sizes)}")
    for i, ucs in enumerate(unique_cell_sizes):
        count = sum(1 for cs in cell_sizes_found if np.allclose(cs, ucs, atol=1e-9))
        print(f"    {i+1}. ({ucs[0]:.10f}, {ucs[1]:.10f}, {ucs[2]:.10f}) - {count} elements")

    # Check uniqueness of parent cube grid indices
    unique_parents = list(set(parent_cubes_found))
    print(f"\n  Unique parent cube grid indices: {len(unique_parents)}")
    for i, parent in enumerate(unique_parents[:10]):  # Show first 10
        count = parent_cubes_found.count(parent)
        print(f"    {i+1}. ({parent[0]:4d}, {parent[1]:4d}, {parent[2]:4d}) - {count} elements")
        if i >= 9 and len(unique_parents) > 10:
            print(f"    ... ({len(unique_parents) - 10} more)")
            break

    # Key insight: If len(unique_parents) > 1, elements from multiple cubes!
    if len(unique_parents) > 1:
        print(f"\n  ⚠️  MULTIPLE PARENT CUBES DETECTED!")
        print(f"     This cell contains elements from {len(unique_parents)} different parent cubes")
        print(f"     This explains why elements/cell > 6")


# Analyze sample cells
if len(cells_with_6) > 0:
    analyze_cell(cells_with_6[0], "Sample: 6 elements")

if len(cells_with_12) > 0:
    analyze_cell(cells_with_12[0], "Sample: 12 elements")

if len(cells_with_24) > 0:
    analyze_cell(cells_with_24[0], "Sample: 24 elements")


# CRITICAL INVESTIGATION: Verify parent cube identification
print(f"\n{'='*80}")
print("CRITICAL CHECK: Parent Cube Identification")
print(f"{'='*80}\n")

print("Checking if elements are correctly assigned to their parent cubes...")

# Build reverse mapping: elem_id -> cell_idx
elem_to_cell_idx = {}
for cell_idx in range(cells.n_cells):
    start = cells.cell_to_elements_offsets[cell_idx]
    end = cells.cell_to_elements_offsets[cell_idx + 1]
    for elem_id in cells.cell_to_elements_data[start:end]:
        elem_to_cell_idx[elem_id] = cell_idx

# Sample 1000 elements and verify they're in the correct cell
n_mismatches = 0
sample_size = min(1000, connectivity.shape[0])

for elem_id in range(sample_size):
    if elem_id not in elem_to_cell_idx:
        continue  # Skipped non-Kuhn element

    vertices = node_positions[connectivity[elem_id]]

    # Compute this element's parent cube
    cell_size, level = find_axis_aligned_edges_single(vertices, 1e-6)
    if np.any(cell_size == 0):
        continue

    cube_corner, cube_center, i, j, k = find_parent_cube(vertices, cell_size, 1e-6)

    # Get the cell this element is assigned to
    cell_idx = elem_to_cell_idx[elem_id]
    assigned_grid_idx = cells.cell_grid_indices[cell_idx]

    # Check if they match
    if not (i == assigned_grid_idx[0] and j == assigned_grid_idx[1] and k == assigned_grid_idx[2]):
        n_mismatches += 1
        if n_mismatches <= 3:
            print(f"  MISMATCH for element {elem_id}:")
            print(f"    Computed grid index: ({i}, {j}, {k})")
            print(f"    Assigned grid index: ({assigned_grid_idx[0]}, {assigned_grid_idx[1]}, {assigned_grid_idx[2]})")

if n_mismatches == 0:
    print(f"  ✅ All {sample_size} sampled elements correctly assigned to parent cubes")
else:
    print(f"  ❌ Found {n_mismatches}/{sample_size} mismatches")


# SEARCHABILITY ANALYSIS
print(f"\n{'='*80}")
print("SEARCHABILITY GUARANTEE")
print(f"{'='*80}\n")

print("Question: Does 1 cell per element guarantee 100% searchability?")
print()
print("Answer:")
print("  Given a query position P:")
print("    1. Compute P's grid indices: (i, j, k) = floor(P / cell_size)")
print("    2. Look up cell with grid indices (i, j, k)")
print("    3. If cell exists, test all elements in that cell")
print()
print("  CRITICAL ISSUE:")
print("    The cell_size used for query lookup must match the element's cell_size!")
print("    But elements have DIFFERENT cell sizes (refinement levels)!")
print()
print("  Current approach:")
print("    - Try multiple levels (14, 13, 12, 11, 10, 9)")
print("    - For each level, compute grid indices and look up cell")
print("    - This SHOULD provide 100% searchability IF:")
print("      ✅ We try all relevant levels")
print("      ✅ Morton encoding is consistent")
print("      ✅ Grid index computation is identical for element assignment and query")
print()

# Verify refinement level distribution
print("Refinement level distribution:")
level_counts = {}
for cell_idx in range(cells.n_cells):
    level = cells.cell_levels[cell_idx]
    level_counts[level] = level_counts.get(level, 0) + 1

for level in sorted(level_counts.keys()):
    count = level_counts[level]
    pct = 100.0 * count / cells.n_cells
    print(f"  Level {level:2d}: {count:8,} cells ({pct:5.2f}%)")

print()
print(f"Levels to search: {sorted(level_counts.keys())}")
print()
print("CONCLUSION:")
print("  IF we search ALL levels present in the mesh, we SHOULD achieve 100% searchability")
print("  because each element belongs to exactly ONE cell at its refinement level.")
print()

print(f"{'='*80}\n")
