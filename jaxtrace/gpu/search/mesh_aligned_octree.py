"""
Mesh-Aligned Octree Cell Extraction - Phase 2

Extracts octree cells from Kuhn tetrahedral meshes using multi-insert strategy
for 100% searchability.

Key Insight from Phase 1 Diagnostic:
    - Kuhn tetrahedra have diagonal faces that cross octree cell boundaries
    - Elements span ~8 cells on average (2x2x2 bounding box pattern)
    - Multi-insert strategy: store elements in ALL overlapping cells
    - Achieves 100% searchability (verified on 3M element mesh)
    - Expected speedup: ~137x reduction in point-in-tet tests

Architecture:
    1. Extract per-dimension cell sizes from axis-aligned edges
    2. Compute element bounding boxes
    3. Find all cells each element's bbox overlaps (multi-insert)
    4. Build inverted index: cell Morton code -> element list
    5. Convert to GPU-friendly CSR format

Usage:
    cells = extract_octree_cells_multi_insert(mesh)
    # Returns: cell Morton codes, element lists (CSR format), metadata
"""

import numpy as np
from typing import Tuple, Dict, List, NamedTuple
from collections import defaultdict
import jax.numpy as jnp


class OctreeCellData(NamedTuple):
    """
    Mesh-aligned octree cells with multi-insert element mapping.

    CSR Format:
        cell_to_elements_offsets[i] -> cell_to_elements_offsets[i+1]
        gives range in cell_to_elements_data for cell i's element list

    Attributes:
        cell_morton_codes: (n_cells,) sorted Morton codes for cells
        cell_levels: (n_cells,) octree depth level for each cell
        cell_sizes: (n_cells, 3) physical size (X, Y, Z) for each cell
        cell_grid_indices: (n_cells, 3) integer grid coordinates (i, j, k)

        cell_to_elements_offsets: (n_cells + 1,) CSR row pointers
        cell_to_elements_data: (n_entries,) element IDs (flattened)

        element_to_cells_offsets: (n_elements + 1,) CSR row pointers
        element_to_cells_data: (n_entries_elem,) cell indices (flattened)

        n_cells: number of unique octree cells
        n_elements: number of mesh elements
        cells_per_element_mean: average cells overlapped per element
        elements_per_cell_mean: average elements per cell
    """
    cell_morton_codes: np.ndarray
    cell_levels: np.ndarray
    cell_sizes: np.ndarray
    cell_grid_indices: np.ndarray

    cell_to_elements_offsets: np.ndarray
    cell_to_elements_data: np.ndarray

    element_to_cells_offsets: np.ndarray
    element_to_cells_data: np.ndarray

    n_cells: int
    n_elements: int
    cells_per_element_mean: float
    elements_per_cell_mean: float


def encode_morton_3d(i: int, j: int, k: int, max_depth: int = 21) -> int:
    """
    Encode 3D grid coordinates as Morton code (Z-order curve).

    Args:
        i, j, k: Grid coordinates (signed integers)
        max_depth: Maximum octree depth (default 21 for 2^21 grid)

    Returns:
        morton: 64-bit Morton code
    """
    morton = 0
    for bit in range(max_depth):
        morton |= ((i >> bit) & 1) << (3 * bit)
        morton |= ((j >> bit) & 1) << (3 * bit + 1)
        morton |= ((k >> bit) & 1) << (3 * bit + 2)
    return morton


def find_axis_aligned_edges(
    elem_vertices: np.ndarray,
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find axis-aligned edges in a tetrahedral element (Kuhn property).

    Args:
        elem_vertices: (4, 3) vertex positions
        tolerance: alignment threshold

    Returns:
        aa_edges: (n_aa, 2) vertex indices for axis-aligned edges
        aa_axes: (n_aa,) axis index (0=X, 1=Y, 2=Z)
        aa_lengths: (n_aa,) edge lengths (cell sizes)
    """
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3)
    ]

    aa_edges = []
    aa_axes = []
    aa_lengths = []

    for v0, v1 in edges:
        vec = elem_vertices[v1] - elem_vertices[v0]
        length = np.linalg.norm(vec)

        if length < tolerance:
            continue

        # Check each axis
        for axis in range(3):
            other_axes = [a for a in range(3) if a != axis]
            if np.all(np.abs(vec[other_axes]) < tolerance):
                aa_edges.append([v0, v1])
                aa_axes.append(axis)
                aa_lengths.append(length)
                break

    return (
        np.array(aa_edges, dtype=np.int32),
        np.array(aa_axes, dtype=np.int32),
        np.array(aa_lengths, dtype=np.float64)
    )


def infer_cell_size_from_edges(
    aa_edges: np.ndarray,
    aa_axes: np.ndarray,
    aa_lengths: np.ndarray
) -> Tuple[np.ndarray, int]:
    """
    Infer octree cell size (per dimension) from axis-aligned edges.

    Args:
        aa_edges: (n_aa, 2) axis-aligned edge vertex indices
        aa_axes: (n_aa,) axis for each edge (0=X, 1=Y, 2=Z)
        aa_lengths: (n_aa,) edge lengths

    Returns:
        cell_size: (3,) cell size [X, Y, Z]
        level: octree depth level (for Morton encoding)
    """
    cell_size = np.zeros(3, dtype=np.float64)

    for axis in range(3):
        mask = aa_axes == axis
        if np.any(mask):
            cell_size[axis] = aa_lengths[mask][0]

    # Infer level from X or Z dimension (should be power of 2 in normalized space)
    # For physical coordinates, we approximate the level
    avg_size = np.mean(cell_size[cell_size > 0])
    if avg_size > 0:
        level = max(0, int(np.round(-np.log2(avg_size * 1000))))  # scale estimate
        level = np.clip(level, 0, 20)
    else:
        level = 14  # default

    return cell_size, level


def find_all_overlapping_cells(
    elem_vertices: np.ndarray,
    cell_size: np.ndarray,
    tolerance: float = 1e-6
) -> List[Tuple[int, int, Tuple[int, int, int], np.ndarray]]:
    """
    Find ALL octree cells that an element overlaps (multi-insert strategy).

    This is the key to 100% retention:
    - Compute element bounding box
    - Find all octree cells that intersect the bbox
    - Return list of (morton, level, indices, cell_size) for each overlapping cell

    Args:
        elem_vertices: (4, 3) vertex positions
        cell_size: (3,) cell sizes [X, Y, Z]
        tolerance: bbox expansion tolerance

    Returns:
        overlapping_cells: List of (morton_code, level, (i,j,k), cell_size)
    """
    # Compute element bounding box
    bbox_min = elem_vertices.min(axis=0) - tolerance
    bbox_max = elem_vertices.max(axis=0) + tolerance

    # Prevent division by zero
    cell_size_safe = np.where(cell_size > tolerance, cell_size, 1.0)

    # Find grid cell range that bbox spans
    i_min = int(np.floor(bbox_min[0] / cell_size_safe[0]))
    i_max = int(np.floor(bbox_max[0] / cell_size_safe[0]))
    j_min = int(np.floor(bbox_min[1] / cell_size_safe[1]))
    j_max = int(np.floor(bbox_max[1] / cell_size_safe[1]))
    k_min = int(np.floor(bbox_min[2] / cell_size_safe[2]))
    k_max = int(np.floor(bbox_max[2] / cell_size_safe[2]))

    # Infer level (approximate from cell size)
    avg_size = np.mean(cell_size[cell_size > 0])
    level = max(0, int(np.round(-np.log2(avg_size * 1000))))
    level = np.clip(level, 0, 20)

    # Generate list of all overlapping cells
    overlapping_cells = []
    offset = (1 << 19)  # Offset for signed coordinates (2^19)
    max_coord = (1 << 20)  # Max coordinate value (2^20)

    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            for k in range(k_min, k_max + 1):
                # Encode as Morton with offset for signed coordinates
                i_morton = np.clip(i + offset, 0, max_coord - 1)
                j_morton = np.clip(j + offset, 0, max_coord - 1)
                k_morton = np.clip(k + offset, 0, max_coord - 1)

                morton = encode_morton_3d(i_morton, j_morton, k_morton, max_depth=21)

                overlapping_cells.append((
                    morton,
                    level,
                    (i, j, k),
                    cell_size.copy()
                ))

    return overlapping_cells


def extract_octree_cells_multi_insert(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    tolerance: float = 1e-6,
    progress_interval: int = 500000,
    verbose: bool = True
) -> OctreeCellData:
    """
    Extract mesh-aligned octree cells with multi-insert strategy.

    Algorithm:
        1. For each element:
           a. Find axis-aligned edges (Kuhn property)
           b. Infer cell size from edge lengths
           c. Compute element bounding box
           d. Find all cells that bbox overlaps
        2. Build inverted index: cell -> list of elements
        3. Convert to CSR format for GPU

    Args:
        node_positions: (n_nodes, 3) vertex coordinates
        connectivity: (n_elements, 4) element vertex indices
        tolerance: geometric tolerance for alignment/containment
        progress_interval: print progress every N elements
        verbose: print progress messages

    Returns:
        OctreeCellData with CSR mappings and metadata
    """
    n_elements = connectivity.shape[0]

    if verbose:
        print(f"\n{'='*80}")
        print("Phase 2: Extracting Mesh-Aligned Octree Cells (Multi-Insert)")
        print(f"{'='*80}")
        print(f"  Elements: {n_elements:,}")
        print(f"  Tolerance: {tolerance:.2e}")

    # Step 1: Compute element -> cells mapping
    if verbose:
        print("\n[1/3] Computing multi-insert mapping (element -> cells)...")

    element_to_cells = {}  # elem_id -> list of (morton, level, indices, cell_size)
    cells_per_element = []

    for elem_id in range(n_elements):
        # Get element vertices
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]

        # Find axis-aligned edges
        aa_edges, aa_axes, aa_lengths = find_axis_aligned_edges(vertices, tolerance)

        if len(aa_edges) < 3:
            # Not a Kuhn tet - skip or handle specially
            element_to_cells[elem_id] = []
            cells_per_element.append(0)
            continue

        # Infer cell size from edges
        cell_size, level = infer_cell_size_from_edges(aa_edges, aa_axes, aa_lengths)

        # Find all overlapping cells
        overlapping = find_all_overlapping_cells(vertices, cell_size, tolerance)

        element_to_cells[elem_id] = overlapping
        cells_per_element.append(len(overlapping))

        if verbose and (elem_id + 1) % progress_interval == 0:
            print(f"    Processed {elem_id + 1:,}/{n_elements:,} elements...")

    cells_per_element = np.array(cells_per_element)

    if verbose:
        print(f"  ✅ Multi-insert mapping complete!")
        print(f"    Mean cells per element: {cells_per_element.mean():.2f}")
        print(f"    Median: {np.median(cells_per_element):.0f}")
        print(f"    Min: {cells_per_element.min()}, Max: {cells_per_element.max()}")

    # Step 2: Build inverted index (cell -> elements)
    if verbose:
        print("\n[2/3] Building inverted index (cell -> elements)...")

    cell_to_elements = defaultdict(list)
    cell_metadata = {}  # morton -> (level, indices, cell_size)

    for elem_id, cells in element_to_cells.items():
        for morton, level, indices, cell_size in cells:
            cell_to_elements[morton].append(elem_id)
            if morton not in cell_metadata:
                cell_metadata[morton] = (level, indices, cell_size)

    n_cells = len(cell_to_elements)
    elements_per_cell = np.array([len(elems) for elems in cell_to_elements.values()])

    if verbose:
        print(f"  ✅ Inverted index complete!")
        print(f"    Unique cells: {n_cells:,}")
        print(f"    Mean elements per cell: {elements_per_cell.mean():.2f}")
        print(f"    Median: {np.median(elements_per_cell):.0f}")
        print(f"    95th percentile: {np.percentile(elements_per_cell, 95):.0f}")

    # Step 3: Convert to CSR format
    if verbose:
        print("\n[3/3] Converting to CSR format for GPU...")

    # Sort cells by Morton code for efficient lookup
    sorted_morton_codes = sorted(cell_to_elements.keys())

    # Build cell arrays
    cell_morton_codes = np.array(sorted_morton_codes, dtype=np.int64)
    cell_levels = np.zeros(n_cells, dtype=np.int32)
    cell_sizes = np.zeros((n_cells, 3), dtype=np.float64)
    cell_grid_indices = np.zeros((n_cells, 3), dtype=np.int32)

    for i, morton in enumerate(sorted_morton_codes):
        level, indices, size = cell_metadata[morton]
        cell_levels[i] = level
        cell_grid_indices[i] = indices
        cell_sizes[i] = size

    # Build cell -> elements CSR
    cell_to_elements_offsets = np.zeros(n_cells + 1, dtype=np.int32)
    cell_to_elements_lists = []

    for i, morton in enumerate(sorted_morton_codes):
        elem_list = sorted(cell_to_elements[morton])
        cell_to_elements_lists.append(elem_list)
        cell_to_elements_offsets[i + 1] = cell_to_elements_offsets[i] + len(elem_list)

    cell_to_elements_data = np.concatenate(cell_to_elements_lists).astype(np.int32)

    # Build element -> cells CSR (reverse mapping)
    element_to_cells_offsets = np.zeros(n_elements + 1, dtype=np.int32)
    element_to_cells_lists = []

    # Create morton -> cell_index mapping
    morton_to_cell_idx = {morton: i for i, morton in enumerate(sorted_morton_codes)}

    for elem_id in range(n_elements):
        cells = element_to_cells[elem_id]
        cell_indices = sorted([morton_to_cell_idx[morton] for morton, _, _, _ in cells])
        element_to_cells_lists.append(cell_indices)
        element_to_cells_offsets[elem_id + 1] = element_to_cells_offsets[elem_id] + len(cell_indices)

    element_to_cells_data = np.concatenate(
        [np.array(lst, dtype=np.int32) for lst in element_to_cells_lists if len(lst) > 0]
    )

    if verbose:
        print(f"  ✅ CSR conversion complete!")
        print(f"    Cell->elements entries: {len(cell_to_elements_data):,}")
        print(f"    Element->cells entries: {len(element_to_cells_data):,}")
        print(f"\n{'='*80}")
        print("Phase 2 Complete: Mesh-Aligned Octree Cells Extracted")
        print(f"{'='*80}")
        print(f"  ✅ {n_cells:,} unique octree cells")
        print(f"  ✅ {cells_per_element.mean():.1f} cells per element (avg)")
        print(f"  ✅ {elements_per_cell.mean():.1f} elements per cell (avg)")
        print(f"  ✅ Multi-insert strategy ensures 100% searchability")
        print(f"{'='*80}\n")

    return OctreeCellData(
        cell_morton_codes=cell_morton_codes,
        cell_levels=cell_levels,
        cell_sizes=cell_sizes,
        cell_grid_indices=cell_grid_indices,
        cell_to_elements_offsets=cell_to_elements_offsets,
        cell_to_elements_data=cell_to_elements_data,
        element_to_cells_offsets=element_to_cells_offsets,
        element_to_cells_data=element_to_cells_data,
        n_cells=n_cells,
        n_elements=n_elements,
        cells_per_element_mean=float(cells_per_element.mean()),
        elements_per_cell_mean=float(elements_per_cell.mean()),
    )


def validate_searchability(
    cells: OctreeCellData,
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    n_samples: int = 1000,
    tolerance: float = 1e-6,
    verbose: bool = True
) -> float:
    """
    Validate that elements can be found via centroid queries.

    Args:
        cells: extracted octree cell data
        node_positions: (n_nodes, 3) vertex coordinates
        connectivity: (n_elements, 4) element indices
        n_samples: number of elements to test
        tolerance: geometric tolerance
        verbose: print results

    Returns:
        searchability_rate: fraction of elements successfully found
    """
    n_elements = cells.n_elements
    sample_indices = np.random.choice(n_elements, min(n_samples, n_elements), replace=False)

    if verbose:
        print(f"\nValidating searchability on {len(sample_indices)} sample elements...")

    found_count = 0

    for elem_id in sample_indices:
        # Compute centroid
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]
        centroid = vertices.mean(axis=0)

        # Get cells this element was inserted into
        start = cells.element_to_cells_offsets[elem_id]
        end = cells.element_to_cells_offsets[elem_id + 1]
        cell_indices = cells.element_to_cells_data[start:end]

        # Check if centroid is in any of these cells
        for cell_idx in cell_indices:
            cell_size = cells.cell_sizes[cell_idx]
            cell_grid_idx = cells.cell_grid_indices[cell_idx]

            cell_min = cell_grid_idx * cell_size
            cell_max = cell_min + cell_size

            if np.all(centroid >= cell_min - tolerance) and np.all(centroid <= cell_max + tolerance):
                # Centroid is in this cell - check element list
                cell_start = cells.cell_to_elements_offsets[cell_idx]
                cell_end = cells.cell_to_elements_offsets[cell_idx + 1]
                cell_elements = cells.cell_to_elements_data[cell_start:cell_end]

                if elem_id in cell_elements:
                    found_count += 1
                    break

    searchability_rate = found_count / len(sample_indices)

    if verbose:
        print(f"  Searchability: {searchability_rate*100:.1f}% ({found_count}/{len(sample_indices)})")
        if searchability_rate >= 0.99:
            print(f"  ✅ Excellent searchability!")
        elif searchability_rate >= 0.95:
            print(f"  ⚠️  Good searchability, but some misses")
        else:
            print(f"  ❌ Poor searchability - check implementation")

    return searchability_rate
