"""
Mesh-Aligned Octree: AABB-Overlap Registration

Each element is registered in every level cell that its axis-aligned bounding
box (AABB) overlaps.  This is a general-mesh registration strategy: no Kuhn
decomposition or axis-aligned-edge structure is assumed.

For Kuhn tetrahedra the AABB coincides with the parent hexahedral cell, so
this strategy produces the same single-cell registration as parent-cube
(1 cell per element, ~6 elements per cell).  The AABB is contracted inward
by a negligible epsilon to avoid registering in adjacent cells when tet
vertices sit exactly on grid boundaries.

For general (non-Kuhn) tetrahedra the element may span 1-8 cells at its
level, depending on cell-boundary crossings.

Data Structure (expected for FLA/cylA benchmark mesh, mostly Kuhn):
                            AABB-Overlap     Parent-Cube     Vertex-Multi
    Cells per element       ~1.0 (Kuhn)      1.00            4.00
    Elements per cell       ~6 (Kuhn)        5.89 (max 8)    18.3 (max 129)
"""

import numpy as np
from typing import NamedTuple
from collections import defaultdict

from .mesh_aligned_octree_single_cell import (
    encode_morton_3d_single,
    find_axis_aligned_edges_single,
)
from .mesh_aligned_octree_vertex_multi import (
    build_node_to_elements,
    find_kuhn_face_neighbor,
)
from .mesh_aligned_octree_parent_cube import _find_kuhn_node_neighbor


class OctreeCellDataAABB(NamedTuple):
    """
    Mesh-aligned octree with AABB-overlap registration.

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


def _register_aabb_cells(vertices, cell_size, level, cell_to_elements_dict,
                         cell_metadata, elem_id):
    """Register an element in all cells overlapped by its AABB.

    The AABB is contracted inward by a small epsilon to avoid registering
    in neighbouring cells when vertices sit exactly on a grid boundary.
    For Kuhn tetrahedra whose vertices coincide with grid corners, this
    reduces the registration from 2×2×2 = 8 cells to 1 (the parent cube),
    matching the parent-cube strategy.  For general tetrahedra the
    contraction is negligible relative to cell size.

    Returns the number of cells registered.
    """
    # Compute AABB, contracted inward to avoid grid-boundary straddling.
    # eps is relative to cell size so it adapts across refinement levels.
    eps = 1e-10 * cell_size
    v_min = vertices.min(axis=0) + eps
    v_max = vertices.max(axis=0) - eps

    # Grid range
    i_min = int(np.floor(v_min[0] / cell_size[0]))
    j_min = int(np.floor(v_min[1] / cell_size[1]))
    k_min = int(np.floor(v_min[2] / cell_size[2]))

    i_max = int(np.floor(v_max[0] / cell_size[0]))
    j_max = int(np.floor(v_max[1] / cell_size[1]))
    k_max = int(np.floor(v_max[2] / cell_size[2]))

    offset = (1 << 19)
    max_coord = (1 << 20)

    n_registered = 0
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            for k in range(k_min, k_max + 1):
                i_m = int(np.clip(i + offset, 0, max_coord - 1))
                j_m = int(np.clip(j + offset, 0, max_coord - 1))
                k_m = int(np.clip(k + offset, 0, max_coord - 1))

                morton = encode_morton_3d_single(i_m, j_m, k_m, max_depth=21)
                cell_key = (morton, level)

                cell_to_elements_dict[cell_key].append(elem_id)

                if cell_key not in cell_metadata:
                    cell_metadata[cell_key] = (cell_size.copy(), (i, j, k))

                n_registered += 1

    return n_registered


def extract_octree_cells_aabb(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    tolerance: float = 1e-6,
    verbose: bool = True,
    orphan_fallback: bool = True,
) -> OctreeCellDataAABB:
    """
    Extract octree cells using AABB-overlap registration.

    Each element is registered in every level cell that its axis-aligned
    bounding box overlaps.  For Kuhn elements the level is detected from
    axis-aligned edges; for non-Kuhn elements the level is borrowed from
    a Kuhn neighbour (same fallback as parent-cube and vertex-multi).

    Args:
        node_positions: (n_nodes, 3) float64 — node coordinates
        connectivity: (n_elements, 4) int32 — element connectivity
        tolerance: geometric tolerance for axis-aligned edge detection
        verbose: print progress
        orphan_fallback: when True (default), non-Kuhn elements that
            have no Kuhn face- or node-neighbour fall back to their own
            AABB to derive cell_size/level. When False, those elements
            are dropped from the octree (legacy behaviour) — spatial
            search will not find them.

    Returns:
        OctreeCellDataAABB with AABB-overlap registration
    """
    n_elements = connectivity.shape[0]

    if verbose:
        print(f"\n{'='*80}")
        print("Mesh-Aligned Octree: AABB-Overlap Registration")
        print(f"{'='*80}")
        print(f"  Elements: {n_elements:,}")
        print(f"  Tolerance: {tolerance:.2e}")

    cell_to_elements_dict = defaultdict(list)
    cell_metadata = {}
    kuhn_element_info = {}
    non_kuhn_ids = []
    total_cell_registrations = 0

    # ------------------------------------------------------------------
    # Pass 1: Kuhn elements — AABB overlap
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[1/3] Pass 1: Kuhn elements → AABB overlap...")

    for elem_id in range(n_elements):
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]

        cell_size, level = find_axis_aligned_edges_single(vertices, tolerance)

        if np.any(cell_size == 0):
            non_kuhn_ids.append(elem_id)
            continue

        kuhn_element_info[elem_id] = (cell_size.copy(), level)

        n_reg = _register_aabb_cells(
            vertices, cell_size, level,
            cell_to_elements_dict, cell_metadata, elem_id,
        )
        total_cell_registrations += n_reg

        if verbose and (elem_id + 1) % 500000 == 0:
            print(f"    Processed {elem_id + 1:,}/{n_elements:,} elements...")

    n_kuhn = n_elements - len(non_kuhn_ids)
    if verbose:
        print(f"  Kuhn elements: {n_kuhn:,}")
        print(f"  Non-Kuhn elements (deferred): {len(non_kuhn_ids):,}")
        if n_kuhn > 0:
            print(f"  Kuhn cells/element (mean): "
                  f"{total_cell_registrations / n_kuhn:.2f}")

    # ------------------------------------------------------------------
    # Pass 2: Non-Kuhn elements — borrow level, AABB overlap
    # ------------------------------------------------------------------
    n_non_kuhn_registered = 0
    n_non_kuhn_orphan_fallback = 0
    n_non_kuhn_dropped = 0

    # Precompute median Kuhn (cell_size, level) as the orphan fallback.
    # The kernel iterates a fixed range of levels (currently 7..14);
    # orphan cells MUST sit at one of those levels to be visited, so we
    # snap to the global median Kuhn level rather than computing one
    # from the orphan's own AABB.
    if kuhn_element_info:
        _kuhn_levels = np.array(
            [info[1] for info in kuhn_element_info.values()],
            dtype=np.int32,
        )
        _kuhn_sizes = np.array(
            [info[0] for info in kuhn_element_info.values()],
            dtype=np.float64,
        )
        median_level = int(np.median(_kuhn_levels))
        median_cell_size = np.median(_kuhn_sizes, axis=0)
    else:
        median_level = 14
        median_cell_size = np.array([
            max(tolerance, 1e-12),
            max(tolerance, 1e-12),
            max(tolerance, 1e-12),
        ], dtype=np.float64)

    if non_kuhn_ids:
        if verbose:
            print(f"\n[2/3] Pass 2: Non-Kuhn elements → neighbour level + AABB overlap...")

        node_to_elements = build_node_to_elements(connectivity)

        for elem_id in non_kuhn_ids:
            vertices = node_positions[connectivity[elem_id]]

            # Try face-neighbour first
            nbr_id, nbr_size, nbr_level = find_kuhn_face_neighbor(
                elem_id, connectivity, node_to_elements, kuhn_element_info
            )

            # Fallback: any node-neighbour that is Kuhn
            if nbr_id < 0:
                nbr_id, nbr_size, nbr_level = _find_kuhn_node_neighbor(
                    elem_id, connectivity, node_to_elements, kuhn_element_info
                )

            if nbr_id < 0:
                if not orphan_fallback:
                    if verbose:
                        print(f"    WARNING: Element {elem_id} has no "
                              f"Kuhn neighbour, skipped")
                    n_non_kuhn_dropped += 1
                    continue
                # Use the global median Kuhn (cell_size, level) so the
                # cell sits at a level the kernel actually walks.
                # Overlap with existing cells is harmless: the point-
                # in-tet test is the authoritative decider.
                nbr_size = median_cell_size
                nbr_level = median_level
                n_non_kuhn_orphan_fallback += 1

            n_reg = _register_aabb_cells(
                vertices, nbr_size, nbr_level,
                cell_to_elements_dict, cell_metadata, elem_id,
            )
            total_cell_registrations += n_reg
            n_non_kuhn_registered += 1

        if verbose:
            print(f"  Non-Kuhn registered: {n_non_kuhn_registered:,} / "
                  f"{len(non_kuhn_ids):,}")
            if n_non_kuhn_orphan_fallback:
                print(f"  Non-Kuhn AABB-fallback cells: "
                      f"{n_non_kuhn_orphan_fallback:,}")
            if n_non_kuhn_dropped:
                print(f"  Non-Kuhn dropped (no fallback): "
                      f"{n_non_kuhn_dropped:,}")
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
    cell_to_elements_data = cell_to_elements_data[:write_idx]

    n_registered = n_kuhn + n_non_kuhn_registered
    elements_per_cell = np.diff(cell_to_elements_offsets)
    cells_per_elem_mean = total_cell_registrations / max(n_registered, 1)

    if verbose:
        print(f"  Unique cells: {n_cells:,}")
        print(f"  Registered elements: {n_registered:,} / {n_elements:,}")
        print(f"  CSR data entries: {write_idx:,}")
        print(f"  Cells per element (mean): {cells_per_elem_mean:.2f}")
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

        unique_lvls, lvl_counts = np.unique(cell_levels, return_counts=True)
        print(f"\n  Level distribution:")
        for lvl, cnt in zip(unique_lvls, lvl_counts):
            print(f"    Level {lvl:2d}: {cnt:>8,} cells")

        print(f"\n{'='*80}")
        print(f"AABB-Overlap Registration Complete!")
        print(f"{'='*80}\n")

    return OctreeCellDataAABB(
        cell_morton_codes=cell_morton_codes,
        cell_levels=cell_levels,
        cell_sizes=cell_sizes,
        cell_grid_indices=cell_grid_indices,
        cell_to_elements_offsets=cell_to_elements_offsets,
        cell_to_elements_data=cell_to_elements_data,
        n_cells=n_cells,
        n_elements=n_registered,
        cells_per_element_mean=cells_per_elem_mean,
        elements_per_cell_mean=float(elements_per_cell.mean()),
        max_elements_per_cell=int(max_elems),
        n_non_kuhn=len(non_kuhn_ids),
        n_non_kuhn_registered=n_non_kuhn_registered,
    )
