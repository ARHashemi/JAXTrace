"""
Corrected Mesh-Aligned Octree: Single Parent Cube per Element

CRITICAL FIX v2: Each tetrahedron belongs to ONE parent octree cube at a specific refinement level.

Previous approach v1 (WRONG):
- Computed element bbox
- Found all grid cells overlapping bbox
- Result: 8 cells per element, wrong cell identification

Previous approach v2 (PARTIAL):
- Found parent cube using floor(min_vertex / cell_size)
- Used Morton code alone as cell key
- Result: 1 cell per element ✅, but 12.27 elements per cell ❌
- Problem: Different refinement levels at same grid position collided

Correct approach v3:
- Find parent cube: floor(min_vertex / cell_size) * cell_size
- Cell key: (morton_code, refinement_level) - BOTH components required
- Each tet belongs to exactly ONE (cube, level) pair
- Expected: ~5-6 elements per cell (Kuhn subdivision)

This matches the actual mesh generation process:
1. Start with octree cube at refinement level L
2. Subdivide cube into 5-6 Kuhn tetrahedra
3. Each tet has 3 axis-aligned edges with length = 2^(-L)
"""

import numpy as np
from typing import Tuple, NamedTuple
from collections import defaultdict


class OctreeCellDataSingle(NamedTuple):
    """
    Mesh-aligned octree with SINGLE parent cube per element.

    Key difference from previous version:
    - 1 cell per element (not 8)
    - ~6 elements per cell (not 37)
    - Cells are actual parent cubes from mesh generation
    """
    cell_morton_codes: np.ndarray      # (n_cells,) uint64
    cell_levels: np.ndarray            # (n_cells,) uint8
    cell_sizes: np.ndarray             # (n_cells, 3) float64
    cell_grid_indices: np.ndarray      # (n_cells, 3) int32

    cell_to_elements_offsets: np.ndarray  # (n_cells + 1,) int32
    cell_to_elements_data: np.ndarray     # (total_entries,) int32

    element_to_cells: np.ndarray          # (n_elements,) int32 - cell index per element (-1 if skipped)

    n_cells: int
    n_elements: int
    cells_per_element_mean: float         # Should be ~1.0
    elements_per_cell_mean: float         # Should be ~5-6


def encode_morton_3d_single(i: int, j: int, k: int, max_depth: int = 21) -> int:
    """Encode grid indices to Morton code."""
    morton = 0
    for bit in range(max_depth):
        morton |= ((i >> bit) & 1) << (3 * bit)
        morton |= ((j >> bit) & 1) << (3 * bit + 1)
        morton |= ((k >> bit) & 1) << (3 * bit + 2)
    return morton


def find_axis_aligned_edges_single(
    vertices: np.ndarray,
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, int]:
    """
    Find axis-aligned edges and infer cell size.

    Returns:
        cell_size: (3,) array [dx, dy, dz]
        level: octree refinement level
    """
    edges = np.array([
        vertices[1] - vertices[0],
        vertices[2] - vertices[0],
        vertices[3] - vertices[0],
        vertices[2] - vertices[1],
        vertices[3] - vertices[1],
        vertices[3] - vertices[2],
    ])

    cell_size = np.zeros(3, dtype=np.float64)

    for axis in range(3):
        other_axes = [a for a in range(3) if a != axis]

        # Find edges aligned with this axis
        for edge in edges:
            if abs(edge[other_axes[0]]) < tolerance and abs(edge[other_axes[1]]) < tolerance:
                if abs(edge[axis]) > tolerance:
                    cell_size[axis] = abs(edge[axis])
                    break

    # Infer level from average cell size
    if np.all(cell_size > 0):
        avg_size = np.mean(cell_size)
        level = max(0, int(round(-np.log2(avg_size))))
        level = np.clip(level, 0, 20)
    else:
        level = 14  # Default

    return cell_size, level


def find_parent_cube(
    vertices: np.ndarray,
    cell_size: np.ndarray,
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    """
    Find the parent octree cube for this tetrahedron.

    The parent cube is the octree cell that contains the tetrahedron's centroid.

    CRITICAL FIX: Use centroid instead of minimum vertex.
    Kuhn tetrahedra can span multiple grid cells. The minimum vertex approach
    resulted in 82.3% of centroids falling OUTSIDE their assigned cubes.

    Algorithm:
        1. Compute tetrahedron centroid
        2. Compute grid indices: floor(centroid / cell_size)
        3. Compute cube corner from grid indices

    Args:
        vertices: (4, 3) tet vertices
        cell_size: (3,) cell dimensions
        tolerance: numerical tolerance

    Returns:
        cube_corner: (3,) corner position
        cube_center: (3,) center position
        i, j, k: grid indices
    """
    # CRITICAL FIX: Use centroid instead of v_min
    # This ensures the centroid is inside the assigned cube
    centroid = vertices.mean(axis=0)

    # Compute grid indices using floor division
    # This gives us the cell that contains the centroid
    i = int(np.floor(centroid[0] / cell_size[0]))
    j = int(np.floor(centroid[1] / cell_size[1]))
    k = int(np.floor(centroid[2] / cell_size[2]))

    # Compute cube corner
    cube_corner = np.array([
        i * cell_size[0],
        j * cell_size[1],
        k * cell_size[2]
    ])

    # Compute cube center
    cube_center = cube_corner + cell_size / 2.0

    return cube_corner, cube_center, i, j, k


def extract_octree_cells_single(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    tolerance: float = 1e-6,
    verbose: bool = True
) -> OctreeCellDataSingle:
    """
    Extract octree cells using SINGLE parent cube per element.

    This is the corrected approach that matches actual mesh generation:
    - Each tet belongs to ONE parent cube
    - Multiple tets (5-6) share the same cube
    - Cubes are actual octree cells from subdivision

    Args:
        node_positions: (n_nodes, 3) node coordinates
        connectivity: (n_elements, 4) element connectivity
        tolerance: geometric tolerance
        verbose: print progress

    Returns:
        OctreeCellDataSingle with single-cube mapping
    """
    n_elements = connectivity.shape[0]

    if verbose:
        print(f"\n{'='*80}")
        print("Corrected Phase 2: Single Parent Cube per Element")
        print(f"{'='*80}")
        print(f"  Elements: {n_elements:,}")
        print(f"  Tolerance: {tolerance:.2e}")
        print(f"  Expected: ~1 cell per element, ~5-6 elements per cell")

    # Maps: element_id -> (morton, level, grid_indices, cell_size, cube_corner)
    element_to_cell_dict = {}

    # Maps: morton -> list of element_ids
    cell_to_elements_dict = defaultdict(list)

    if verbose:
        print(f"\n[1/3] Finding parent cube for each element...")

    n_skipped = 0

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

        # Find parent cube
        cube_corner, cube_center, i, j, k = find_parent_cube(
            vertices, cell_size, tolerance
        )

        # Encode to Morton (with offset for negative coordinates)
        offset = (1 << 19)  # 2^19
        max_coord = (1 << 20)  # 2^20

        i_morton = np.clip(i + offset, 0, max_coord - 1)
        j_morton = np.clip(j + offset, 0, max_coord - 1)
        k_morton = np.clip(k + offset, 0, max_coord - 1)

        morton = encode_morton_3d_single(i_morton, j_morton, k_morton, max_depth=21)

        # CRITICAL FIX: Use (morton, level) as key to avoid collisions
        # Different refinement levels at the same grid position must be separate cells
        cell_key = (morton, level)

        # Store element -> cell mapping
        element_to_cell_dict[elem_id] = (morton, level, (i, j, k), cell_size, cube_corner)

        # Build inverted index: cell -> elements
        cell_to_elements_dict[cell_key].append(elem_id)

        if verbose and (elem_id + 1) % 500000 == 0:
            print(f"    Processed {elem_id + 1:,}/{n_elements:,} elements...")

    if verbose:
        print(f"  ✅ Element->cell mapping complete!")
        print(f"    Skipped {n_skipped:,} non-Kuhn elements")
        print(f"    Mapped {len(element_to_cell_dict):,} elements to cells")

    # Build CSR structures
    if verbose:
        print(f"\n[2/3] Building inverted index (cell -> elements)...")

    n_cells = len(cell_to_elements_dict)
    # Sort by (morton, level) - this keeps cells organized spatially, then by refinement
    sorted_cell_keys = sorted(cell_to_elements_dict.keys())

    # Build cell arrays - extract morton and level from keys
    cell_morton_codes = np.array([morton for morton, level in sorted_cell_keys], dtype=np.uint64)
    cell_levels = np.array([level for morton, level in sorted_cell_keys], dtype=np.uint8)
    cell_sizes = np.zeros((n_cells, 3), dtype=np.float64)
    cell_grid_indices = np.zeros((n_cells, 3), dtype=np.int32)

    # Get metadata from first element in each cell
    for cell_idx, cell_key in enumerate(sorted_cell_keys):
        elem_list = cell_to_elements_dict[cell_key]
        first_elem = elem_list[0]
        _, level, (i, j, k), cell_size, _ = element_to_cell_dict[first_elem]

        # level already set above from cell_key
        cell_sizes[cell_idx] = cell_size
        cell_grid_indices[cell_idx] = [i, j, k]

    # Build cell -> elements CSR
    cell_to_elements_offsets = np.zeros(n_cells + 1, dtype=np.int32)
    cell_to_elements_lists = []

    for cell_idx, cell_key in enumerate(sorted_cell_keys):
        elem_list = cell_to_elements_dict[cell_key]
        cell_to_elements_offsets[cell_idx + 1] = cell_to_elements_offsets[cell_idx] + len(elem_list)
        cell_to_elements_lists.extend(elem_list)

    cell_to_elements_data = np.array(cell_to_elements_lists, dtype=np.int32)

    # Build element -> cell mapping (single cell index per element)
    # Build reverse lookup: cell_key -> cell_idx
    cell_key_to_idx = {cell_key: idx for idx, cell_key in enumerate(sorted_cell_keys)}

    element_to_cells = np.full(n_elements, -1, dtype=np.int32)  # -1 for skipped elements
    for elem_id, (morton, level, _, _, _) in element_to_cell_dict.items():
        cell_key = (morton, level)
        element_to_cells[elem_id] = cell_key_to_idx[cell_key]

    if verbose:
        print(f"  ✅ CSR structure built!")
        print(f"    Unique cells: {n_cells:,}")
        print(f"    CSR data entries: {len(cell_to_elements_data):,}")

    # Compute statistics
    if verbose:
        print(f"\n[3/3] Computing statistics...")

    elements_per_cell = np.diff(cell_to_elements_offsets)
    cells_per_element_mean = len(element_to_cell_dict) / n_elements
    elements_per_cell_mean = elements_per_cell.mean()

    if verbose:
        print(f"\n  Statistics:")
        print(f"    Cells per element: {cells_per_element_mean:.2f} (expected ~1.0)")
        print(f"    Elements per cell: {elements_per_cell_mean:.2f} (expected ~5-6)")
        print(f"    Elements per cell (median): {np.median(elements_per_cell):.0f}")
        print(f"    Elements per cell (min, max): ({elements_per_cell.min()}, {elements_per_cell.max()})")

        # Distribution
        unique_counts, count_freqs = np.unique(elements_per_cell, return_counts=True)
        print(f"\n  Elements-per-cell distribution (top 10):")
        for count, freq in sorted(zip(unique_counts, count_freqs), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {count:2d} elements: {freq:8,} cells ({100*freq/n_cells:5.2f}%)")

        print(f"\n{'='*80}")
        print("Corrected Phase 2 Complete!")
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
        n_elements=len(element_to_cell_dict),
        cells_per_element_mean=cells_per_element_mean,
        elements_per_cell_mean=elements_per_cell_mean,
    )
