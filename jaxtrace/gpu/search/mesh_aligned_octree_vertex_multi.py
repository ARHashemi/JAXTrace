"""
Mesh-Aligned Octree: Multi-Cell Vertex Registration

This module addresses the element retention problem by registering each element
in ALL cells that its vertices touch, not just the single cell containing its centroid.

Problem Analysis:
- Current single-cell registration: 88.59% retention over 100 RK4 steps
- Root cause: 100% of Kuhn elements span cell boundaries (vertices at cube corners)
- When particle crosses to adjacent cell, element not found → particle lost

Solution: Multi-Cell Vertex Registration
- Register each element in ALL cells its 4 vertices touch (~4 cells per element)
- Bidirectional mapping: element→cells and cell→elements
- Expected retention: ~95%+ (similar to traditional methods)
- Memory cost: ~97.5 MB increase (from 37.5 MB to 135 MB) - acceptable

Data Structure Comparison:
                            Single-Cell    Multi-Cell
    Cells                   517,069        517,069 (same)
    Element registrations   3,047,074      12,188,296 (4×)
    Elements per cell       5.89           23.57
    Cells per element       1.00           4.00
    GPU memory             37.5 MB        135.0 MB
    Tests per particle     ~35            ~141

Implementation:
- Reuse robust axis-aligned edge detection from single-cell version
- Find ALL cells touched by element's 4 vertices
- Build CSR structures in both directions
"""

import numpy as np
from typing import Tuple, NamedTuple
from collections import defaultdict

from .mesh_aligned_octree_single_cell import (
    encode_morton_3d_single,
    find_axis_aligned_edges_single
)


class OctreeCellDataVertexMulti(NamedTuple):
    """
    Mesh-aligned octree with MULTI-CELL vertex registration.

    Key difference from single-cell version:
    - Each element registered in ~4 cells (all cells touched by vertices)
    - Each cell contains ~24 elements (4× more than single-cell)
    - Bidirectional mapping for efficient queries
    """
    cell_morton_codes: np.ndarray      # (n_cells,) uint64
    cell_levels: np.ndarray            # (n_cells,) uint8
    cell_sizes: np.ndarray             # (n_cells, 3) float64
    cell_grid_indices: np.ndarray      # (n_cells, 3) int32

    cell_to_elements_offsets: np.ndarray  # (n_cells + 1,) int32 - CSR offsets
    cell_to_elements_data: np.ndarray     # (total_entries,) int32 - element IDs

    element_to_cells_offsets: np.ndarray  # (n_elements + 1,) int32 - CSR offsets (NEW)
    element_to_cells_data: np.ndarray     # (total_entries,) int32 - cell IDs (NEW)

    n_cells: int
    n_elements: int
    cells_per_element_mean: float         # Should be ~4.0
    elements_per_cell_mean: float         # Should be ~23-24


def build_node_to_elements(connectivity: np.ndarray) -> dict:
    """
    Build a mapping from node ID to set of element IDs sharing that node.

    This enables O(1) face-neighbor lookup for Non-Kuhn elements,
    independent of element processing order.
    """
    node_to_elements = defaultdict(set)
    for elem_id in range(connectivity.shape[0]):
        for node_id in connectivity[elem_id]:
            node_to_elements[int(node_id)].add(elem_id)
    return node_to_elements


def find_kuhn_face_neighbor(
    elem_id: int,
    connectivity: np.ndarray,
    node_to_elements: dict,
    kuhn_element_info: dict,
) -> tuple:
    """
    Find a Kuhn face-neighbor of a Non-Kuhn element using node-to-element index.

    Returns the first face neighbor (shares >= 3 nodes) that has known
    Kuhn cell_size/level, so the Non-Kuhn element can borrow its grid parameters.

    Args:
        elem_id: Non-Kuhn element ID
        connectivity: Mesh connectivity (n_elements, 4)
        node_to_elements: Mapping from node_id -> set of element IDs
        kuhn_element_info: Dict of {elem_id: (cell_size, level)} for Kuhn elements

    Returns:
        (neighbor_id, cell_size, level) or (-1, None, None) if not found
    """
    elem_nodes = connectivity[elem_id]
    elem_node_set = set(int(n) for n in elem_nodes)

    # Collect candidate neighbors: elements sharing at least one node
    candidates = set()
    for node_id in elem_nodes:
        candidates.update(node_to_elements[int(node_id)])
    candidates.discard(elem_id)

    # Find face neighbor (shares >= 3 nodes) that is a Kuhn element
    for neighbor_id in candidates:
        if neighbor_id not in kuhn_element_info:
            continue
        neighbor_node_set = set(int(n) for n in connectivity[neighbor_id])
        shared = elem_node_set & neighbor_node_set
        if len(shared) >= 3:
            cell_size, level = kuhn_element_info[neighbor_id]
            return neighbor_id, cell_size, level

    return -1, None, None


def extract_octree_cells_vertex_multi(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    tolerance: float = 1e-6,
    verbose: bool = True
) -> OctreeCellDataVertexMulti:
    """
    Extract octree cells using MULTI-CELL vertex registration.

    This approach registers each element in ALL cells its vertices touch,
    solving the retention problem caused by elements spanning cell boundaries.

    Algorithm:
        1. For each element:
            a. Find axis-aligned edges → cell_size and level
            b. For each vertex:
                - Compute grid cell: floor(vertex / cell_size)
                - Register element in that cell
        2. Build bidirectional CSR mappings:
            - cell → elements (for search)
            - element → cells (for tracking which cells contain element)

    Args:
        node_positions: (n_nodes, 3) node coordinates
        connectivity: (n_elements, 4) element connectivity
        tolerance: geometric tolerance
        verbose: print progress

    Returns:
        OctreeCellDataVertexMulti with multi-cell mapping
    """
    n_elements = connectivity.shape[0]

    if verbose:
        print(f"\n{'='*80}")
        print("Multi-Cell Vertex Registration")
        print(f"{'='*80}")
        print(f"  Elements: {n_elements:,}")
        print(f"  Tolerance: {tolerance:.2e}")
        print(f"  Expected: ~4 cells per element, ~24 elements per cell")

    # Bidirectional mappings
    element_to_cells_dict = defaultdict(set)  # elem_id -> set of cell_keys
    cell_to_elements_dict = defaultdict(set)  # cell_key -> set of elem_ids

    # Cell metadata: cell_key -> (morton, level, grid_indices, cell_size)
    cell_metadata = {}

    # --- Pass 1: Process all Kuhn elements (have axis-aligned edges) ---
    if verbose:
        print(f"\n[1/5] Pass 1: Processing Kuhn elements...")

    n_non_kuhn = 0
    non_kuhn_ids = []
    kuhn_element_info = {}  # elem_id -> (cell_size, level) for Kuhn elements
    total_vertex_cell_registrations = 0

    for elem_id in range(n_elements):
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]

        cell_size, level = find_axis_aligned_edges_single(vertices, tolerance)

        if np.any(cell_size == 0):
            # Non-Kuhn: defer to pass 2
            n_non_kuhn += 1
            non_kuhn_ids.append(elem_id)
            continue

        # Store Kuhn info for neighbor lookup in pass 2
        kuhn_element_info[elem_id] = (cell_size.copy(), level)

        # Register in ALL cells touched by vertices
        vertex_cells = set()
        for vertex in vertices:
            i = int(np.floor(vertex[0] / cell_size[0]))
            j = int(np.floor(vertex[1] / cell_size[1]))
            k = int(np.floor(vertex[2] / cell_size[2]))

            offset = (1 << 19)
            max_coord = (1 << 20)

            i_morton = np.clip(i + offset, 0, max_coord - 1)
            j_morton = np.clip(j + offset, 0, max_coord - 1)
            k_morton = np.clip(k + offset, 0, max_coord - 1)

            morton = encode_morton_3d_single(i_morton, j_morton, k_morton, max_depth=21)

            cell_key = (morton, level)
            vertex_cells.add(cell_key)

            if cell_key not in cell_metadata:
                cell_metadata[cell_key] = (morton, level, (i, j, k), cell_size.copy())

        for cell_key in vertex_cells:
            element_to_cells_dict[elem_id].add(cell_key)
            cell_to_elements_dict[cell_key].add(elem_id)
            total_vertex_cell_registrations += 1

        if verbose and (elem_id + 1) % 500000 == 0:
            print(f"    Processed {elem_id + 1:,}/{n_elements:,} elements...")

    if verbose:
        print(f"  Kuhn elements: {n_elements - n_non_kuhn:,}")
        print(f"  Non-Kuhn elements (deferred): {n_non_kuhn:,}")

    # --- Pass 2: Process Non-Kuhn elements using Kuhn neighbor's grid ---
    if verbose:
        print(f"\n[2/5] Pass 2: Processing {n_non_kuhn:,} Non-Kuhn elements...")
        print(f"  Building node-to-element index...")

    node_to_elements = build_node_to_elements(connectivity)

    n_neighbor_found = 0
    n_neighbor_not_found = 0

    for elem_id in non_kuhn_ids:
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]

        # Find a Kuhn face neighbor to borrow its cell_size and level
        neighbor_id, neighbor_cell_size, neighbor_level = find_kuhn_face_neighbor(
            elem_id, connectivity, node_to_elements, kuhn_element_info
        )

        if neighbor_id >= 0:
            n_neighbor_found += 1

            # Register using the Non-Kuhn element's OWN vertex positions
            # but with the neighbor's cell_size and level.
            # This ensures the element is in cells the search will actually look in.
            vertex_cells = set()
            for vertex in vertices:
                i = int(np.floor(vertex[0] / neighbor_cell_size[0]))
                j = int(np.floor(vertex[1] / neighbor_cell_size[1]))
                k = int(np.floor(vertex[2] / neighbor_cell_size[2]))

                offset = (1 << 19)
                max_coord = (1 << 20)

                i_morton = np.clip(i + offset, 0, max_coord - 1)
                j_morton = np.clip(j + offset, 0, max_coord - 1)
                k_morton = np.clip(k + offset, 0, max_coord - 1)

                morton = encode_morton_3d_single(i_morton, j_morton, k_morton, max_depth=21)

                cell_key = (morton, neighbor_level)
                vertex_cells.add(cell_key)

                if cell_key not in cell_metadata:
                    cell_metadata[cell_key] = (morton, neighbor_level, (i, j, k), neighbor_cell_size.copy())

            for cell_key in vertex_cells:
                element_to_cells_dict[elem_id].add(cell_key)
                cell_to_elements_dict[cell_key].add(elem_id)
                total_vertex_cell_registrations += 1
        else:
            n_neighbor_not_found += 1
            if verbose:
                print(f"    WARNING: Element {elem_id} has no Kuhn face neighbor, skipping")

    if verbose:
        print(f"  Non-Kuhn with Kuhn neighbor: {n_neighbor_found:,}")
        print(f"  Non-Kuhn without Kuhn neighbor: {n_neighbor_not_found:,}")
        print(f"  Total registrations: {total_vertex_cell_registrations:,}")
        print(f"  Cells per element (mean): {total_vertex_cell_registrations / n_elements:.2f}")

    # Build CSR structures
    if verbose:
        print(f"\n[3/5] Building cell→elements CSR structure...")

    unique_cells = sorted(cell_to_elements_dict.keys())
    n_cells = len(unique_cells)

    # Create mapping: cell_key -> cell_idx
    cell_key_to_idx = {cell_key: idx for idx, cell_key in enumerate(unique_cells)}

    # Allocate CSR arrays for cell→elements
    cell_to_elements_offsets = np.zeros(n_cells + 1, dtype=np.int32)
    cell_to_elements_data = np.zeros(total_vertex_cell_registrations, dtype=np.int32)

    # Allocate cell metadata arrays
    cell_morton_codes = np.zeros(n_cells, dtype=np.uint64)
    cell_levels = np.zeros(n_cells, dtype=np.uint8)
    cell_sizes = np.zeros((n_cells, 3), dtype=np.float64)
    cell_grid_indices = np.zeros((n_cells, 3), dtype=np.int32)

    # Fill cell→elements CSR
    write_idx = 0
    for cell_idx, cell_key in enumerate(unique_cells):
        elem_ids = sorted(cell_to_elements_dict[cell_key])
        n_elems = len(elem_ids)

        cell_to_elements_offsets[cell_idx] = write_idx
        cell_to_elements_data[write_idx:write_idx + n_elems] = elem_ids
        write_idx += n_elems

        # Fill cell metadata
        morton, level, (i, j, k), cell_size = cell_metadata[cell_key]
        cell_morton_codes[cell_idx] = morton
        cell_levels[cell_idx] = level
        cell_sizes[cell_idx] = cell_size
        cell_grid_indices[cell_idx] = [i, j, k]

    cell_to_elements_offsets[n_cells] = write_idx

    if verbose:
        print(f"  ✅ CSR structure built!")
        print(f"    Unique cells: {n_cells:,}")
        print(f"    CSR data entries: {write_idx:,}")

    # Build element→cells CSR structure
    if verbose:
        print(f"\n[4/5] Building element→cells CSR structure...")

    element_to_cells_offsets = np.zeros(n_elements + 1, dtype=np.int32)
    element_to_cells_data = np.zeros(total_vertex_cell_registrations, dtype=np.int32)

    write_idx = 0
    for elem_id in range(n_elements):
        if elem_id in element_to_cells_dict:
            # Convert cell_keys to cell_indices
            cell_keys = sorted(element_to_cells_dict[elem_id])
            cell_indices = [cell_key_to_idx[ck] for ck in cell_keys]
            n_cells_for_elem = len(cell_indices)

            element_to_cells_offsets[elem_id] = write_idx
            element_to_cells_data[write_idx:write_idx + n_cells_for_elem] = cell_indices
            write_idx += n_cells_for_elem
        else:
            # Skipped element
            element_to_cells_offsets[elem_id] = write_idx

    element_to_cells_offsets[n_elements] = write_idx

    if verbose:
        print(f"  ✅ Element→cells CSR structure built!")
        print(f"    CSR data entries: {write_idx:,}")

    # Compute statistics
    if verbose:
        print(f"\n[5/5] Computing statistics...")

    cells_per_element = np.diff(element_to_cells_offsets)
    cells_per_element = cells_per_element[cells_per_element > 0]  # Exclude skipped elements

    elements_per_cell = np.diff(cell_to_elements_offsets)

    cells_per_element_mean = np.mean(cells_per_element)
    elements_per_cell_mean = np.mean(elements_per_cell)
    elements_per_cell_median = np.median(elements_per_cell)

    if verbose:
        print(f"\n  Statistics:")
        print(f"    Cells per element: {cells_per_element_mean:.2f} (expected ~4.0)")
        print(f"    Elements per cell: {elements_per_cell_mean:.2f} (expected ~23-24)")
        print(f"    Elements per cell (median): {int(elements_per_cell_median)}")
        print(f"    Elements per cell (min, max): ({elements_per_cell.min()}, {elements_per_cell.max()})")

        # Show distribution
        unique_counts, count_freqs = np.unique(elements_per_cell, return_counts=True)
        print(f"\n  Elements-per-cell distribution (top 10):")
        sorted_indices = np.argsort(count_freqs)[::-1][:10]
        for idx in sorted_indices:
            count = unique_counts[idx]
            freq = count_freqs[idx]
            pct = 100.0 * freq / n_cells
            print(f"    {count:3d} elements: {freq:7,} cells ({pct:5.2f}%)")

        print(f"\n{'='*80}")
        print("Multi-Cell Vertex Registration Complete!")
        print(f"{'='*80}\n")

    return OctreeCellDataVertexMulti(
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
        cells_per_element_mean=float(cells_per_element_mean),
        elements_per_cell_mean=float(elements_per_cell_mean),
    )
