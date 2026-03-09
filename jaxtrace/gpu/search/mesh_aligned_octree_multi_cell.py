"""
Mesh-Aligned Octree: Multiple Cells per Element (Bbox-Based Assignment)

CRITICAL FIX: Each element is assigned to ALL cells its bounding box intersects.

This is the CORRECT approach for mesh-aligned octree search:
- Elements can span multiple octree cells (especially at refinement boundaries)
- A particle inside an element might fall in a different cell than the element's centroid
- Solution: Assign element to ALL cells it intersects (bbox-based)

Architecture:
1. Extract mesh-aligned octree cells (same as single-cell version)
2. Compute element bounding boxes
3. For each element, find ALL cells its bbox intersects
4. Build CSR: cell → [list of elements]
5. Result: More CSR entries, but complete coverage

Expected:
- Elements per cell: ~5-6 (same as single-cell)
- Cells per element: ~1-8 (depending on how element spans cells)
- Retention: 96-98%+ (same as original element-based Morton)
"""

import numpy as np
from typing import Tuple
from collections import defaultdict

from .mesh_aligned_octree_single_cell import (
    OctreeCellDataSingle,
    find_axis_aligned_edges_single,
    encode_morton_3d_single,
)


def compute_element_bbox(
    vertices: np.ndarray,
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute axis-aligned bounding box for element.

    Args:
        vertices: (4, 3) float64 - tetrahedron vertices
        tolerance: numerical tolerance

    Returns:
        bbox_min: (3,) float64 - minimum corner
        bbox_max: (3,) float64 - maximum corner
    """
    bbox_min = vertices.min(axis=0) - tolerance
    bbox_max = vertices.max(axis=0) + tolerance
    return bbox_min, bbox_max


def find_intersecting_cells(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    cell_size: np.ndarray,
    tolerance: float = 1e-6
) -> list:
    """
    Find all octree cells that intersect with element's bounding box.

    Algorithm:
    1. Compute grid indices for bbox corners
    2. Generate all grid cells in the range [i_min, i_max] × [j_min, j_max] × [k_min, k_max]

    Args:
        bbox_min: (3,) float64 - element bbox minimum
        bbox_max: (3,) float64 - element bbox maximum
        cell_size: (3,) float64 - octree cell size [dx, dy, dz]
        tolerance: numerical tolerance

    Returns:
        cells: list of (i, j, k) tuples - grid indices of intersecting cells
    """
    # Compute grid indices for bbox corners
    i_min = int(np.floor(bbox_min[0] / cell_size[0]))
    j_min = int(np.floor(bbox_min[1] / cell_size[1]))
    k_min = int(np.floor(bbox_min[2] / cell_size[2]))

    i_max = int(np.floor(bbox_max[0] / cell_size[0]))
    j_max = int(np.floor(bbox_max[1] / cell_size[1]))
    k_max = int(np.floor(bbox_max[2] / cell_size[2]))

    # Generate all cells in range
    cells = []
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            for k in range(k_min, k_max + 1):
                cells.append((i, j, k))

    return cells


def extract_octree_cells_multi(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    tolerance: float = 1e-6,
    verbose: bool = True
) -> OctreeCellDataSingle:
    """
    Extract mesh-aligned octree cells with bbox-based multi-cell assignment.

    KEY DIFFERENCE from single-cell version:
    - Each element assigned to ALL cells its bbox intersects (not just parent cell)
    - CSR structure has MORE entries (multiple cells per element)
    - Ensures complete coverage: particle will find element regardless of which cell it lands in

    Args:
        node_positions: (n_nodes, 3) float64 - node coordinates
        connectivity: (n_elements, 4) int32 - element connectivity
        tolerance: geometric tolerance
        verbose: print progress

    Returns:
        OctreeCellDataSingle with multi-cell assignment
    """
    n_elements = connectivity.shape[0]

    if verbose:
        print(f"\n{'='*80}")
        print("Mesh-Aligned Octree: Multi-Cell Assignment (Bbox-Based)")
        print(f"{'='*80}")
        print(f"  Elements: {n_elements:,}")
        print(f"  Tolerance: {tolerance:.2e}")
        print(f"  Expected: ~1-8 cells per element, ~5-6 elements per cell")

    # Maps: element_id -> [(morton, level, (i,j,k), cell_size), ...]
    element_to_cells_dict = defaultdict(list)

    # Maps: (morton, level) -> [element_ids]
    cell_to_elements_dict = defaultdict(list)

    # Store cell metadata: (morton, level) -> (cell_size, (i, j, k))
    cell_metadata = {}

    if verbose:
        print(f"\n[1/3] Finding all intersecting cells for each element...")

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

        # Compute element bbox
        bbox_min, bbox_max = compute_element_bbox(vertices, tolerance)

        # Find ALL cells that intersect this bbox
        intersecting_cells = find_intersecting_cells(
            bbox_min, bbox_max, cell_size, tolerance
        )

        # Encode Morton for offset handling (negative coordinates)
        offset = (1 << 19)  # 2^19
        max_coord = (1 << 20)  # 2^20

        # Assign element to ALL intersecting cells
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
        # Remove duplicates (element might be added multiple times)
        elem_list_unique = sorted(set(elem_list))

        cell_to_elements_offsets[cell_idx + 1] = cell_to_elements_offsets[cell_idx] + len(elem_list_unique)
        cell_to_elements_lists.extend(elem_list_unique)

    cell_to_elements_data = np.array(cell_to_elements_lists, dtype=np.int32)

    # Build element -> cell mapping (just use first cell for compatibility)
    cell_key_to_idx = {cell_key: idx for idx, cell_key in enumerate(sorted_cell_keys)}
    element_to_cells = np.full(n_elements, -1, dtype=np.int32)

    for elem_id, cell_list in element_to_cells_dict.items():
        if len(cell_list) > 0:
            # Use first cell as primary (for compatibility with single-cell API)
            morton, level, _, _ = cell_list[0]
            cell_key = (morton, level)
            element_to_cells[elem_id] = cell_key_to_idx[cell_key]

    if verbose:
        print(f"  ✅ CSR structure built!")
        print(f"    Unique cells: {n_cells:,}")
        print(f"    CSR data entries: {len(cell_to_elements_data):,} (was {len(element_to_cells_dict):,} with single-cell)")
        print(f"    Increase: {len(cell_to_elements_data) / max(1, len(element_to_cells_dict)):.1f}× more CSR entries")

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
        print(f"    Elements per cell: {elements_per_cell_mean:.2f} (expected ~5-6)")
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
        print("Multi-Cell Assignment Complete!")
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
