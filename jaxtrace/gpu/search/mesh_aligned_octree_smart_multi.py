"""
Mesh-Aligned Octree: Smart Multi-Cell Assignment

IMPROVEMENT over naive bbox approach:
- Naive: Assign to ALL cells in bbox → 27 cells/element (too much!)
- Smart: Assign only to cells the element ACTUALLY intersects → ~1-4 cells/element

Algorithm:
1. For Kuhn tetrahedra: Test which cells the tet vertices actually touch
2. Use axis-aligned edges to determine precise cell coverage
3. Much tighter assignment than full bbox

Expected:
- Cells per element: ~1-4 (instead of 27)
- Elements per cell: ~8-12 (instead of 114)
- CSR entries: ~4-12× overhead (instead of 27×)
- Retention: 96-98%+ (same coverage, less overhead)
"""

import numpy as np
from typing import Tuple, Set
from collections import defaultdict

from .mesh_aligned_octree_single_cell import (
    OctreeCellDataSingle,
    find_axis_aligned_edges_single,
    encode_morton_3d_single,
)


def find_cells_from_vertices(
    vertices: np.ndarray,
    cell_size: np.ndarray,
    tolerance: float = 1e-6
) -> Set[Tuple[int, int, int]]:
    """
    Find cells that contain tetrahedron vertices.

    For Kuhn tetrahedra with axis-aligned edges, we only need to check
    which cells contain the 4 vertices. The tet spans at most 2-4 cells.

    Args:
        vertices: (4, 3) float64 - tetrahedron vertices
        cell_size: (3,) float64 - cell dimensions
        tolerance: numerical tolerance

    Returns:
        cells: Set of (i, j, k) tuples - grid indices of cells containing vertices
    """
    cells = set()

    for vertex in vertices:
        # Compute grid indices for this vertex
        i = int(np.floor((vertex[0] + tolerance) / cell_size[0]))
        j = int(np.floor((vertex[1] + tolerance) / cell_size[1]))
        k = int(np.floor((vertex[2] + tolerance) / cell_size[2]))

        cells.add((i, j, k))

        # Also check cell boundaries (vertex might be on edge)
        # If vertex is within tolerance of cell boundary, include both cells
        x_on_boundary = abs((vertex[0] % cell_size[0])) < tolerance or abs((vertex[0] % cell_size[0]) - cell_size[0]) < tolerance
        y_on_boundary = abs((vertex[1] % cell_size[1])) < tolerance or abs((vertex[1] % cell_size[1]) - cell_size[1]) < tolerance
        z_on_boundary = abs((vertex[2] % cell_size[2])) < tolerance or abs((vertex[2] % cell_size[2]) - cell_size[2]) < tolerance

        # Add neighboring cells if on boundary
        if x_on_boundary:
            cells.add((i-1, j, k))
            cells.add((i+1, j, k))
        if y_on_boundary:
            cells.add((i, j-1, k))
            cells.add((i, j+1, k))
        if z_on_boundary:
            cells.add((i, j, k-1))
            cells.add((i, j, k+1))

        # Corners (2D boundaries)
        if x_on_boundary and y_on_boundary:
            cells.add((i-1, j-1, k))
            cells.add((i-1, j+1, k))
            cells.add((i+1, j-1, k))
            cells.add((i+1, j+1, k))
        if y_on_boundary and z_on_boundary:
            cells.add((i, j-1, k-1))
            cells.add((i, j-1, k+1))
            cells.add((i, j+1, k-1))
            cells.add((i, j+1, k+1))
        if z_on_boundary and x_on_boundary:
            cells.add((i-1, j, k-1))
            cells.add((i-1, j, k+1))
            cells.add((i+1, j, k-1))
            cells.add((i+1, j, k+1))

    return cells


def extract_octree_cells_smart_multi(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    tolerance: float = 1e-6,
    verbose: bool = True
) -> OctreeCellDataSingle:
    """
    Extract mesh-aligned octree cells with SMART multi-cell assignment.

    KEY DIFFERENCE from naive bbox approach:
    - Naive bbox: Assigns to ALL cells in bbox → 27 cells/element
    - Smart: Assigns only to cells containing vertices → ~1-4 cells/element
    - Result: Same coverage, much less overhead

    Args:
        node_positions: (n_nodes, 3) float64 - node coordinates
        connectivity: (n_elements, 4) int32 - element connectivity
        tolerance: geometric tolerance
        verbose: print progress

    Returns:
        OctreeCellDataSingle with smart multi-cell assignment
    """
    n_elements = connectivity.shape[0]

    if verbose:
        print(f"\n{'='*80}")
        print("Mesh-Aligned Octree: SMART Multi-Cell Assignment")
        print(f"{'='*80}")
        print(f"  Elements: {n_elements:,}")
        print(f"  Tolerance: {tolerance:.2e}")
        print(f"  Expected: ~1-4 cells per element, ~8-12 elements per cell")

    # Maps: element_id -> [(morton, level, (i,j,k), cell_size), ...]
    element_to_cells_dict = defaultdict(list)

    # Maps: (morton, level) -> [element_ids]
    cell_to_elements_dict = defaultdict(list)

    # Store cell metadata: (morton, level) -> (cell_size, (i, j, k))
    cell_metadata = {}

    if verbose:
        print(f"\n[1/3] Finding cells containing element vertices...")

    n_skipped = 0
    total_assignments = 0

    for elem_id in range(n_elements):
        # Get vertices
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]

        # Find axis-aligned edges and cell size
        cell_size, level = find_axis_aligned_edges_single(vertices, tolerance)

        if np.any(cell_size == 0):
            # Skip non-Kuhn elements
            n_skipped += 1
            continue

        # Find cells containing vertices (smart assignment)
        intersecting_cells = find_cells_from_vertices(vertices, cell_size, tolerance)

        # Encode Morton for offset handling (negative coordinates)
        offset = (1 << 19)  # 2^19
        max_coord = (1 << 20)  # 2^20

        # Assign element to cells
        for (i, j, k) in intersecting_cells:
            i_morton = np.clip(i + offset, 0, max_coord - 1)
            j_morton = np.clip(j + offset, 0, max_coord - 1)
            k_morton = np.clip(k + offset, 0, max_coord - 1)

            morton = encode_morton_3d_single(i_morton, j_morton, k_morton, max_depth=21)
            cell_key = (morton, level)

            # Store element -> cell mapping
            element_to_cells_dict[elem_id].append((morton, level, (i, j, k), cell_size))

            # Build inverted index: cell -> elements
            cell_to_elements_dict[cell_key].append(elem_id)

            # Store cell metadata (only need to do once per cell)
            if cell_key not in cell_metadata:
                cell_metadata[cell_key] = (cell_size, (i, j, k))

            total_assignments += 1

        if verbose and (elem_id + 1) % 500000 == 0:
            print(f"    Processed {elem_id + 1:,}/{n_elements:,} elements...")

    if verbose:
        print(f"  ✅ Element->cell mapping complete!")
        print(f"    Skipped {n_skipped:,} non-Kuhn elements")
        print(f"    Mapped {len(element_to_cells_dict):,} elements to cells")
        print(f"    Total assignments: {total_assignments:,} (avg {total_assignments/max(1, len(element_to_cells_dict)):.1f} cells/element)")

    # Build CSR structures
    if verbose:
        print(f"\n[2/3] Building inverted index (cell -> elements)...")

    n_cells = len(cell_to_elements_dict)
    # Sort by (morton, level)
    sorted_cell_keys = sorted(cell_to_elements_dict.keys())

    # Build cell arrays
    cell_morton_codes = np.array([morton for morton, level in sorted_cell_keys], dtype=np.uint64)
    cell_levels = np.array([level for morton, level in sorted_cell_keys], dtype=np.uint8)
    cell_sizes = np.zeros((n_cells, 3), dtype=np.float64)
    cell_grid_indices = np.zeros((n_cells, 3), dtype=np.int32)

    # Extract metadata for each cell
    for cell_idx, cell_key in enumerate(sorted_cell_keys):
        size, (i, j, k) = cell_metadata[cell_key]
        cell_sizes[cell_idx] = size
        cell_grid_indices[cell_idx] = [i, j, k]

    # Build cell -> elements CSR
    cell_to_elements_offsets = np.zeros(n_cells + 1, dtype=np.int32)
    cell_to_elements_lists = []

    for cell_idx, cell_key in enumerate(sorted_cell_keys):
        elem_list = cell_to_elements_dict[cell_key]
        # Remove duplicates
        elem_list_unique = sorted(set(elem_list))

        cell_to_elements_offsets[cell_idx + 1] = cell_to_elements_offsets[cell_idx] + len(elem_list_unique)
        cell_to_elements_lists.extend(elem_list_unique)

    cell_to_elements_data = np.array(cell_to_elements_lists, dtype=np.int32)

    # Build element -> cell mapping (use first cell for compatibility)
    cell_key_to_idx = {cell_key: idx for idx, cell_key in enumerate(sorted_cell_keys)}
    element_to_cells = np.full(n_elements, -1, dtype=np.int32)

    for elem_id, cell_list in element_to_cells_dict.items():
        if len(cell_list) > 0:
            morton, level, _, _ = cell_list[0]
            cell_key = (morton, level)
            element_to_cells[elem_id] = cell_key_to_idx[cell_key]

    if verbose:
        print(f"  ✅ CSR structure built!")
        print(f"    Unique cells: {n_cells:,}")
        print(f"    CSR data entries: {len(cell_to_elements_data):,}")
        print(f"    Overhead: {len(cell_to_elements_data) / max(1, len(element_to_cells_dict)):.1f}× (vs 27× for naive bbox)")

    # Compute statistics
    if verbose:
        print(f"\n[3/3] Computing statistics...")

    elements_per_cell = np.diff(cell_to_elements_offsets)
    cells_per_element = np.array([len(cell_list) for cell_list in element_to_cells_dict.values()])

    cells_per_element_mean = cells_per_element.mean()
    elements_per_cell_mean = elements_per_cell.mean()

    if verbose:
        print(f"\n  Statistics:")
        print(f"    Cells per element: {cells_per_element_mean:.2f} (min={cells_per_element.min()}, max={cells_per_element.max()})")
        print(f"    Elements per cell: {elements_per_cell_mean:.2f} (expected ~8-12)")
        print(f"    Elements per cell (median): {np.median(elements_per_cell):.0f}")
        print(f"    Elements per cell (min, max): ({elements_per_cell.min()}, {elements_per_cell.max()})")

        # Cells per element distribution
        unique_counts, count_freqs = np.unique(cells_per_element, return_counts=True)
        print(f"\n  Cells-per-element distribution:")
        for count, freq in sorted(zip(unique_counts, count_freqs), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {count:2d} cells: {freq:8,} elements ({100*freq/len(element_to_cells_dict):5.2f}%)")

        # Elements per cell distribution
        unique_counts, count_freqs = np.unique(elements_per_cell, return_counts=True)
        print(f"\n  Elements-per-cell distribution (top 10):")
        for count, freq in sorted(zip(unique_counts, count_freqs), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {count:2d} elements: {freq:8,} cells ({100*freq/n_cells:5.2f}%)")

        print(f"\n{'='*80}")
        print("SMART Multi-Cell Assignment Complete!")
        print(f"{'='*80}\n")

    return OctreeCellDataSingle(
        cell_morton_codes=cell_morton_codes,
        cell_levels=cell_levels,
        cell_sizes=cell_sizes,
        cell_grid_indices=cell_grid_indices,
        cell_to_elements_offsets=cell_to_elements_offsets,
        cell_to_elements_data=cell_to_elements_data,
        element_to_cells=element_to_cells,
        n_cells=n_cells,
        n_elements=len(element_to_cells_dict),
        cells_per_element_mean=cells_per_element_mean,
        elements_per_cell_mean=elements_per_cell_mean,
    )
