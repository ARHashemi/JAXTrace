"""
Mesh-Aligned Octree: Parent-Cube Registration

Each element is registered in its ONE parent cube — the Kuhn hexahedral cell
whose octree subdivision produced this tetrahedron.  This gives a tight,
predictable element-per-cell count (5–6 for Kuhn, max ~8) that enables
fully static inner search loops on GPU.

Non-Kuhn elements (those without 3 axis-aligned edges) are handled by
borrowing a face- or node-neighbour's cell_size/level and registering via
centroid into that grid, adding at most +1 per cell.

Combined with 3x3x3 neighbourhood search, this gives 100% found rate on
Kuhn meshes (Proposition 1 in the paper).

Data Structure (for the FLA/cylA benchmark mesh):
                            Parent-Cube     Vertex-Multi (for comparison)
    Cells                   ~517,069        ~665,824
    Element registrations   ~3,048,900      ~12,194,568
    Elements per cell       5.89 (max 8)    18.3 (max 129)
    Cells per element       1.00            4.00
    Inner loop bound        8 (static)      dynamic (CSR)
"""

import numpy as np
from typing import Tuple, NamedTuple
from collections import defaultdict

from .mesh_aligned_octree_single_cell import (
    encode_morton_3d_single,
    find_axis_aligned_edges_single,
    find_parent_cube,
)
from .mesh_aligned_octree_vertex_multi import (
    build_node_to_elements,
    find_kuhn_face_neighbor,
)


class OctreeCellDataParentCube(NamedTuple):
    """
    Mesh-aligned octree with single parent-cube registration.

    Each Kuhn element is in exactly 1 cell (its parent cube).
    Non-Kuhn elements are placed in their Kuhn neighbour's grid.

    Compatible with upload_mesh_aligned_octree_to_gpu() — it reads:
        cell_morton_codes, cell_levels, cell_sizes, cell_grid_indices,
        cell_to_elements_offsets, cell_to_elements_data,
        n_cells, n_elements, elements_per_cell_mean
    """
    cell_morton_codes: np.ndarray       # (n_cells,) uint64
    cell_levels: np.ndarray             # (n_cells,) uint8
    cell_sizes: np.ndarray              # (n_cells, 3) float64
    cell_grid_indices: np.ndarray       # (n_cells, 3) int32

    cell_to_elements_offsets: np.ndarray  # (n_cells + 1,) int32 — CSR offsets
    cell_to_elements_data: np.ndarray     # (total_entries,) int32 — element IDs

    n_cells: int
    n_elements: int
    cells_per_element_mean: float
    elements_per_cell_mean: float

    # Extra statistics
    max_elements_per_cell: int
    n_non_kuhn: int
    n_non_kuhn_registered: int


def _find_kuhn_node_neighbor(
    elem_id: int,
    connectivity: np.ndarray,
    node_to_elements: dict,
    kuhn_element_info: dict,
) -> tuple:
    """
    Find a Kuhn node-neighbor (shares >= 1 node) for a Non-Kuhn element.

    Fallback when no face-neighbor (>= 3 shared nodes) is Kuhn.
    """
    elem_nodes = connectivity[elem_id]

    for node_id in elem_nodes:
        for neighbor_id in node_to_elements[int(node_id)]:
            if neighbor_id == elem_id:
                continue
            if neighbor_id in kuhn_element_info:
                cell_size, level = kuhn_element_info[neighbor_id]
                return neighbor_id, cell_size, level

    return -1, None, None


def extract_octree_cells_parent_cube(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    tolerance: float = 1e-6,
    verbose: bool = True,
) -> OctreeCellDataParentCube:
    """
    Extract octree cells using SINGLE parent-cube registration.

    Each Kuhn element is registered in the ONE cell that contains its
    centroid (the parent cube from the Kuhn decomposition).  Non-Kuhn
    elements borrow a Kuhn neighbour's grid and are registered via
    centroid into that grid.

    Args:
        node_positions: (n_nodes, 3) float64 — node coordinates
        connectivity: (n_elements, 4) int32 — element connectivity
        tolerance: geometric tolerance for axis-aligned edge detection
        verbose: print progress

    Returns:
        OctreeCellDataParentCube with single-cube-per-element mapping
    """
    n_elements = connectivity.shape[0]

    if verbose:
        print(f"\n{'='*80}")
        print("Mesh-Aligned Octree: Parent-Cube Registration")
        print(f"{'='*80}")
        print(f"  Elements: {n_elements:,}")
        print(f"  Tolerance: {tolerance:.2e}")
        print(f"  Expected: 1 cell per element, ~5-6 elements per cell, max ~8")

    # cell_key = (morton, level)  →  list of element IDs
    cell_to_elements_dict = defaultdict(list)
    # cell_key  →  (cell_size, (i, j, k))
    cell_metadata = {}
    # For non-Kuhn pass 2
    kuhn_element_info = {}  # elem_id → (cell_size, level)
    non_kuhn_ids = []

    # ------------------------------------------------------------------
    # Pass 1: Kuhn elements — centroid → parent cube
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[1/3] Pass 1: Kuhn elements → parent cube...")

    for elem_id in range(n_elements):
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]

        cell_size, level = find_axis_aligned_edges_single(vertices, tolerance)

        if np.any(cell_size == 0):
            non_kuhn_ids.append(elem_id)
            continue

        kuhn_element_info[elem_id] = (cell_size.copy(), level)

        # Parent cube via centroid
        _, _, i, j, k = find_parent_cube(vertices, cell_size, tolerance)

        offset = (1 << 19)
        max_coord = (1 << 20)
        i_m = int(np.clip(i + offset, 0, max_coord - 1))
        j_m = int(np.clip(j + offset, 0, max_coord - 1))
        k_m = int(np.clip(k + offset, 0, max_coord - 1))

        morton = encode_morton_3d_single(i_m, j_m, k_m, max_depth=21)
        cell_key = (morton, level)

        cell_to_elements_dict[cell_key].append(elem_id)

        if cell_key not in cell_metadata:
            cell_metadata[cell_key] = (cell_size.copy(), (i, j, k))

        if verbose and (elem_id + 1) % 500000 == 0:
            print(f"    Processed {elem_id + 1:,}/{n_elements:,} elements...")

    n_kuhn = n_elements - len(non_kuhn_ids)
    if verbose:
        print(f"  Kuhn elements: {n_kuhn:,}")
        print(f"  Non-Kuhn elements (deferred): {len(non_kuhn_ids):,}")

    # ------------------------------------------------------------------
    # Pass 2: Non-Kuhn elements — borrow neighbour's grid, centroid assignment
    # ------------------------------------------------------------------
    n_non_kuhn_registered = 0

    if non_kuhn_ids:
        if verbose:
            print(f"\n[2/3] Pass 2: Non-Kuhn elements → neighbour's grid...")

        node_to_elements = build_node_to_elements(connectivity)

        for elem_id in non_kuhn_ids:
            vertices = node_positions[connectivity[elem_id]]

            # Try face-neighbour first (shares ≥ 3 nodes)
            nbr_id, nbr_size, nbr_level = find_kuhn_face_neighbor(
                elem_id, connectivity, node_to_elements, kuhn_element_info
            )

            # Fallback: any node-neighbour that is Kuhn
            if nbr_id < 0:
                nbr_id, nbr_size, nbr_level = _find_kuhn_node_neighbor(
                    elem_id, connectivity, node_to_elements, kuhn_element_info
                )

            if nbr_id < 0:
                if verbose:
                    print(f"    WARNING: Element {elem_id} has no Kuhn neighbour, skipped")
                continue

            # Centroid of the non-Kuhn element in the neighbour's grid
            centroid = vertices.mean(axis=0)
            i = int(np.floor(centroid[0] / nbr_size[0]))
            j = int(np.floor(centroid[1] / nbr_size[1]))
            k = int(np.floor(centroid[2] / nbr_size[2]))

            offset = (1 << 19)
            max_coord = (1 << 20)
            i_m = int(np.clip(i + offset, 0, max_coord - 1))
            j_m = int(np.clip(j + offset, 0, max_coord - 1))
            k_m = int(np.clip(k + offset, 0, max_coord - 1))

            morton = encode_morton_3d_single(i_m, j_m, k_m, max_depth=21)
            cell_key = (morton, nbr_level)

            cell_to_elements_dict[cell_key].append(elem_id)

            if cell_key not in cell_metadata:
                cell_metadata[cell_key] = (nbr_size.copy(), (i, j, k))

            n_non_kuhn_registered += 1

        if verbose:
            print(f"  Non-Kuhn registered: {n_non_kuhn_registered:,} / {len(non_kuhn_ids):,}")
    else:
        if verbose:
            print(f"\n[2/3] Pass 2: No non-Kuhn elements — skipped")

    # ------------------------------------------------------------------
    # Build CSR
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[3/3] Building CSR structure...")

    sorted_cell_keys = sorted(cell_to_elements_dict.keys())
    n_cells = len(sorted_cell_keys)

    cell_morton_codes = np.zeros(n_cells, dtype=np.uint64)
    cell_levels = np.zeros(n_cells, dtype=np.uint8)
    cell_sizes = np.zeros((n_cells, 3), dtype=np.float64)
    cell_grid_indices = np.zeros((n_cells, 3), dtype=np.int32)

    cell_to_elements_offsets = np.zeros(n_cells + 1, dtype=np.int32)
    total_entries = sum(len(cell_to_elements_dict[k]) for k in sorted_cell_keys)
    cell_to_elements_data = np.zeros(total_entries, dtype=np.int32)

    write_idx = 0
    max_elems = 0

    for cell_idx, cell_key in enumerate(sorted_cell_keys):
        morton, level = cell_key
        cell_morton_codes[cell_idx] = morton
        cell_levels[cell_idx] = level

        size, (i, j, k) = cell_metadata[cell_key]
        cell_sizes[cell_idx] = size
        cell_grid_indices[cell_idx] = [i, j, k]

        elem_list = sorted(set(cell_to_elements_dict[cell_key]))  # deduplicate
        n_elems = len(elem_list)

        cell_to_elements_offsets[cell_idx] = write_idx
        cell_to_elements_data[write_idx:write_idx + n_elems] = elem_list
        write_idx += n_elems

        if n_elems > max_elems:
            max_elems = n_elems

    cell_to_elements_offsets[n_cells] = write_idx
    # Trim data array (deduplication may have reduced total)
    cell_to_elements_data = cell_to_elements_data[:write_idx]

    n_registered = n_kuhn + n_non_kuhn_registered
    elements_per_cell = np.diff(cell_to_elements_offsets)

    if verbose:
        print(f"  Unique cells: {n_cells:,}")
        print(f"  Registered elements: {n_registered:,} / {n_elements:,}")
        print(f"  CSR data entries: {write_idx:,}")
        print(f"\n  Elements-per-cell statistics:")
        print(f"    Mean:   {elements_per_cell.mean():.2f}")
        print(f"    Median: {np.median(elements_per_cell):.0f}")
        print(f"    Min:    {elements_per_cell.min()}")
        print(f"    Max:    {elements_per_cell.max()}")

        unique_counts, count_freqs = np.unique(elements_per_cell, return_counts=True)
        print(f"\n  Elements-per-cell distribution:")
        for count, freq in sorted(zip(unique_counts, count_freqs),
                                  key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {count:3d} elements: {freq:>8,} cells "
                  f"({100 * freq / n_cells:5.2f}%)")

        # Level distribution
        unique_lvls, lvl_counts = np.unique(cell_levels, return_counts=True)
        print(f"\n  Level distribution:")
        for lvl, cnt in zip(unique_lvls, lvl_counts):
            print(f"    Level {lvl:2d}: {cnt:>8,} cells")

        print(f"\n{'='*80}")
        print(f"Parent-Cube Registration Complete!")
        print(f"{'='*80}\n")

    return OctreeCellDataParentCube(
        cell_morton_codes=cell_morton_codes,
        cell_levels=cell_levels,
        cell_sizes=cell_sizes,
        cell_grid_indices=cell_grid_indices,
        cell_to_elements_offsets=cell_to_elements_offsets,
        cell_to_elements_data=cell_to_elements_data,
        n_cells=n_cells,
        n_elements=n_registered,
        cells_per_element_mean=1.0,
        elements_per_cell_mean=float(elements_per_cell.mean()),
        max_elements_per_cell=int(max_elems),
        n_non_kuhn=len(non_kuhn_ids),
        n_non_kuhn_registered=n_non_kuhn_registered,
    )
