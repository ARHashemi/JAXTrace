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

from ... import config

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


def tet_box_intersect(vertices: np.ndarray,
                      box_min: np.ndarray,
                      box_max: np.ndarray) -> bool:
    """Exact tetrahedron / axis-aligned-box overlap, by separating axes.

    Two convex bodies are disjoint iff some axis exists on which their
    projections do not overlap. For a tet against an axis-aligned box it is
    sufficient to test 13 axes: the 3 box face normals, the 4 tet face
    normals, and the 6 cross products of box edge directions with tet edge
    directions. Touching counts as intersecting, which is the conservative
    choice for a point-location structure.

    This is exact, not a tolerance: a cell is rejected only when the
    tetrahedron provably does not reach it. That matters because dropping a
    cell the tet does touch would make particles there unfindable -- the same
    failure mode as the cell-overflow defect (docs/dfamily/11_*).

    Used to prune AABB-overlap registrations: a tet fills at most ~1/6 of its
    bounding box, so most cells its AABB touches are never reached by the tet
    itself. See docs/dfamily/14_concepts_explained.md for why pruning cannot
    lose coverage (in short: a particle's own cell is the centre of the 3x3x3
    neighbourhood and is always intersected by its host, so it always
    survives pruning).
    """
    bc = 0.5 * (box_min + box_max)
    be = 0.5 * (box_max - box_min)

    def separated(axis: np.ndarray) -> bool:
        n = np.linalg.norm(axis)
        if n < 1e-300:                    # degenerate axis carries no info
            return False
        a = axis / n
        r = abs(a[0]) * be[0] + abs(a[1]) * be[1] + abs(a[2]) * be[2]
        pc = float(a @ bc)
        pv = vertices @ a
        return (pv.min() > pc + r) or (pv.max() < pc - r)

    for i in range(3):                    # box face normals
        e = np.zeros(3)
        e[i] = 1.0
        if separated(e):
            return False
    for a, b, c in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        if separated(np.cross(vertices[b] - vertices[a],
                              vertices[c] - vertices[a])):
            return False
    tet_edges = (vertices[1] - vertices[0],
                 vertices[2] - vertices[0],
                 vertices[3] - vertices[0])
    for i in range(3):
        e = np.zeros(3)
        e[i] = 1.0
        for te in tet_edges:
            if separated(np.cross(e, te)):
                return False
    return True


def extract_octree_cells_parent_cube(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    tolerance: float = 1e-6,
    verbose: bool = True,
    orphan_fallback: bool = True,
    hybrid_non_kuhn: bool = True,
    exact_tet_box: bool = None,
) -> OctreeCellDataParentCube:
    """
    Extract octree cells using SINGLE parent-cube registration for Kuhn
    elements and (by default) AABB-overlap registration for non-Kuhn
    elements.

    Each Kuhn element is registered in the ONE cell that contains its
    centroid (the parent cube from the Kuhn decomposition).  Non-Kuhn
    elements borrow a Kuhn neighbour's grid (face → node neighbour →
    global median Kuhn fallback) and are registered either:
      - in every cell their axis-aligned bounding box overlaps
        (hybrid_non_kuhn=True, default), or
      - in the single cell their centroid falls in
        (hybrid_non_kuhn=False, legacy behaviour).

    Background: on meshes with a high non-Kuhn fraction (>20 % of
    elements), centroid-only registration creates coverage holes —
    non-Kuhn tets whose vertices fall outside the 3x3x3 cell
    neighbourhood around the centroid cell. A particle landing on such
    a vertex is invisible to the kernel's L0/L1/L2 search and ends up
    with element_id=-1 inside the geometric domain. AABB-overlap
    registration on non-Kuhn elements closes those holes by listing
    each non-Kuhn tet in every cell its AABB touches. Kuhn elements are
    untouched — they keep the cheap single-cell registration.

    Args:
        node_positions: (n_nodes, 3) float64 — node coordinates
        connectivity: (n_elements, 4) int32 — element connectivity
        tolerance: geometric tolerance for axis-aligned edge detection
        verbose: print progress
        orphan_fallback: when True (default), non-Kuhn elements that
            have no Kuhn face- or node-neighbour fall back to the
            global median Kuhn (cell_size, level). When False, those
            elements are dropped from the octree.
        hybrid_non_kuhn: when True (default), non-Kuhn elements use
            AABB-overlap registration. When False, they use the
            centroid-only registration (legacy behaviour).
        exact_tet_box: prune AABB-overlap registrations to the cells the
            tetrahedron genuinely intersects, instead of every cell its
            bounding box touches. None (default) reads
            config.OCTREE_EXACT_TET_BOX. A tet fills at most ~1/6 of its
            AABB, so this removed ~29% of registrations on D2 and lowered
            max cell occupancy 79 -> 56, i.e. 1.41x fewer point-in-tet
            tests per query, with coverage preserved (0 losses over 13,000
            adversarial probes; see docs/dfamily/14_concepts_explained.md).
            Build-time only: the branch is evaluated once on the host, so
            neither setting costs anything at run time.

    Returns:
        OctreeCellDataParentCube with single-cube-per-Kuhn-element and
        (AABB-overlap | centroid)-per-non-Kuhn-element mapping.
    """
    n_elements = connectivity.shape[0]

    # Resolved once, on the host, before anything is built. A plain Python
    # branch: no kernel ever sees it, so neither setting has a runtime cost.
    if exact_tet_box is None:
        exact_tet_box = bool(getattr(config, "OCTREE_EXACT_TET_BOX", False))

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
    # Pass 0: classify every element and derive the per-level lookup
    # lattice BEFORE anything registers.
    #
    # Registration and lookup must divide by the same number. The search
    # computes its index as floor(pos / level_cell_sizes[level]), where
    # upload_mesh_aligned_octree_to_gpu sets level_cell_sizes[level] to
    # the per-axis MEDIAN of the sizes at that level. Any element filed
    # under its own cell_size instead is filed on a different grid than
    # the one it is later looked up on.
    #
    # On an isotropic mesh every size at a level is identical, so the
    # median IS each element's own size and this is a no-op (A/B/C).
    # On an anisotropic mesh the sizes form a continuum (D2 level 14
    # carries 3.5753e-05, 3.5839e-05, 3.6146e-05 ... against a median of
    # 5.6117e-05) and the two grids drift apart. At a particle position
    # the disagreement is an index error of 2 (median 2, p90 2) while the
    # 3x3x3 neighbourhood reaches only +-1, so the host cell is never
    # tested and the particle is reported lost with element_id=-1.
    #
    # Measured on D2: registering Pass 2 on the shared pitch recovered
    # 1,164 of 2,000 previously-lost particles, and every one of the 644
    # remaining reproducible failures sat on a mismatched PASS 1 (Kuhn)
    # cell — 0 on a non-Kuhn cell, 0 on a matching cell. Hence the pitch
    # is derived here, once, and both passes register on it.
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[1/4] Pass 0: classify elements, derive lookup lattice...")

    element_size_level = {}   # elem_id → (cell_size, level)  Kuhn only
    _level_sizes = {}

    for elem_id in range(n_elements):
        vertices = node_positions[connectivity[elem_id]]
        cell_size, level = find_axis_aligned_edges_single(vertices, tolerance)

        if np.any(cell_size == 0):
            non_kuhn_ids.append(elem_id)
            continue

        element_size_level[elem_id] = (cell_size, level)
        _level_sizes.setdefault(int(level), []).append(cell_size)

        if verbose and (elem_id + 1) % 500000 == 0:
            print(f"    Classified {elem_id + 1:,}/{n_elements:,} elements...")

    # The one lattice shared by Pass 1, Pass 2 and the GPU search.
    lookup_pitch = {lv: np.median(np.asarray(v), axis=0)
                    for lv, v in _level_sizes.items()}

    if verbose and lookup_pitch:
        print(f"  Lookup lattice: {len(lookup_pitch)} level(s)")
        for lv in sorted(lookup_pitch):
            _p = lookup_pitch[lv]
            _n = len(_level_sizes[lv])
            _own = np.asarray(_level_sizes[lv])
            _agree = int(np.sum(np.all(np.isclose(_own, _p, rtol=1e-6), axis=1)))
            print(f"    level {lv:2d}: pitch {_p[0]:.6e} {_p[1]:.6e} "
                  f"{_p[2]:.6e}  ({_agree:,}/{_n:,} elements match it "
                  f"exactly = {100.0 * _agree / max(_n, 1):.2f}%)")

    # ------------------------------------------------------------------
    # Pass 1: Kuhn elements — centroid → parent cube ON THE SHARED PITCH
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[2/4] Pass 1: Kuhn elements → parent cube...")

    for elem_id, (cell_size, level) in element_size_level.items():
        vertices = node_positions[connectivity[elem_id]]

        # Register on the level's lookup pitch, not on this element's own
        # size, so floor(pos/pitch) in the search reproduces (i, j, k).
        reg_size = lookup_pitch[int(level)]

        kuhn_element_info[elem_id] = (cell_size.copy(), level)

        # Parent cube via centroid, measured on the lookup lattice.
        _, _, i, j, k = find_parent_cube(vertices, reg_size, tolerance)

        offset = (1 << 19)
        max_coord = (1 << 20)
        i_m = int(np.clip(i + offset, 0, max_coord - 1))
        j_m = int(np.clip(j + offset, 0, max_coord - 1))
        k_m = int(np.clip(k + offset, 0, max_coord - 1))

        morton = encode_morton_3d_single(i_m, j_m, k_m, max_depth=21)
        cell_key = (morton, level)

        cell_to_elements_dict[cell_key].append(elem_id)

        if cell_key not in cell_metadata:
            cell_metadata[cell_key] = (np.asarray(reg_size).copy(), (i, j, k))

    del element_size_level

    n_kuhn = n_elements - len(non_kuhn_ids)
    if verbose:
        print(f"  Kuhn elements: {n_kuhn:,}")
        print(f"  Non-Kuhn elements (deferred): {len(non_kuhn_ids):,}")

    # ------------------------------------------------------------------
    # Pass 2: Non-Kuhn elements — borrow neighbour's grid, centroid assignment
    # ------------------------------------------------------------------
    n_non_kuhn_registered = 0
    n_non_kuhn_orphan_fallback = 0
    n_non_kuhn_dropped = 0
    n_non_kuhn_cell_entries = 0
    n_non_kuhn_cells_pruned = 0

    # Precompute a global fallback (cell_size, level) from the median
    # Kuhn element. We DO NOT compute orphan levels from the orphan's
    # own AABB — the kernel only searches a fixed range of levels (7..14
    # currently), so a level outside that range would never be visited
    # and the orphan would be invisible to the spatial search.
    # Snapping to the global median guarantees the orphan cell sits at
    # a level the kernel actually walks. Overlap with neighbouring
    # Kuhn cells at the same level is harmless: the search reads every
    # element listed in a visited cell and the point-in-tet test
    # (kernel side) decides — duplicate listings cannot produce a wrong
    # answer, only redundant tests.
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
        # Pathological all-non-Kuhn mesh (no axis-aligned edges anywhere,
        # e.g. Delaunay-like general unstructured meshes such as the
        # StanfordBunny benchmark).  We can't infer a level or cell size
        # from Kuhn structure, so derive them from mesh geometry:
        #   - cell_size := median tet AABB diagonal (isotropic).
        #   This gives ~ (n_elements)^(1/3) cells per axis for uniform
        #   meshes — bounded work, mesh-scale-appropriate — and avoids
        #   the tolerance-floor pathology that produced ~10^24-cell grids.
        _diag_lengths = np.empty(n_elements, dtype=np.float64)
        for _eid in range(n_elements):
            _v = node_positions[connectivity[_eid]]
            _diag_lengths[_eid] = np.linalg.norm(_v.max(axis=0) - _v.min(axis=0))
        _iso_size = float(np.median(_diag_lengths))
        _iso_size = max(_iso_size, 1e-12)  # guard against degenerate meshes
        median_cell_size = np.array(
            [_iso_size, _iso_size, _iso_size], dtype=np.float64,
        )
        # median_level is set so the resulting cells are visible to the
        # kernel's level range.  Since we have no Kuhn levels to snap
        # to, pick level 0 (finest granular cell) — the search's
        # min_level..max_level loop will find it as long as any cell
        # actually registers at level 0.  We DO use level 0 here rather
        # than a nominal "14" because the fallback path only fires when
        # kuhn_element_info is empty, so no other level is defined and
        # the emitted octree will contain only these fallback cells.
        median_level = 0
        if verbose:
            print(f"  [fallback] all-non-Kuhn mesh detected: using isotropic "
                  f"cell_size={_iso_size:.6g} at level 0")

    if non_kuhn_ids:
        if verbose:
            print(f"\n[3/4] Pass 2: Non-Kuhn elements → neighbour's grid...")

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
                # No Kuhn neighbour at all. Two options:
                #   orphan_fallback=True  (default): borrow the global
                #     median Kuhn (cell_size, level) so the orphan sits
                #     at a level the kernel walks, and let it be
                #     registered by centroid into that grid.
                #   orphan_fallback=False: drop the element (legacy).
                if not orphan_fallback:
                    if verbose:
                        print(f"    WARNING: Element {elem_id} has no "
                              f"Kuhn neighbour, skipped")
                    n_non_kuhn_dropped += 1
                    continue
                nbr_size = median_cell_size
                nbr_level = median_level
                n_non_kuhn_orphan_fallback += 1

            offset = (1 << 19)
            max_coord = (1 << 20)

            if hybrid_non_kuhn:
                # AABB-overlap registration: every cell the tet's AABB
                # touches at the chosen (cell_size, level). Covers the
                # vertices that centroid-only registration would miss.
                lo = vertices.min(axis=0)
                hi = vertices.max(axis=0)
                # Enumerate on the LOOKUP lattice for this level, not on the
                # borrowed neighbour's size, so registration and search agree.
                reg_size = lookup_pitch.get(int(nbr_level), nbr_size)
                i_lo = int(np.floor(lo[0] / reg_size[0]))
                j_lo = int(np.floor(lo[1] / reg_size[1]))
                k_lo = int(np.floor(lo[2] / reg_size[2]))
                i_hi = int(np.floor(hi[0] / reg_size[0]))
                j_hi = int(np.floor(hi[1] / reg_size[1]))
                k_hi = int(np.floor(hi[2] / reg_size[2]))
                _rs = np.asarray(reg_size, dtype=np.float64)
                for i in range(i_lo, i_hi + 1):
                    for j in range(j_lo, j_hi + 1):
                        for k in range(k_lo, k_hi + 1):
                            if exact_tet_box:
                                # Keep only cells the TET reaches, not every
                                # cell its bounding box touches. The particle's
                                # own cell is always intersected by its host,
                                # so coverage is preserved.
                                _bmin = np.array([i, j, k],
                                                 dtype=np.float64) * _rs
                                if not tet_box_intersect(vertices, _bmin,
                                                         _bmin + _rs):
                                    n_non_kuhn_cells_pruned += 1
                                    continue
                            i_m = int(np.clip(i + offset, 0, max_coord - 1))
                            j_m = int(np.clip(j + offset, 0, max_coord - 1))
                            k_m = int(np.clip(k + offset, 0, max_coord - 1))
                            morton = encode_morton_3d_single(
                                i_m, j_m, k_m, max_depth=21,
                            )
                            cell_key = (morton, nbr_level)
                            cell_to_elements_dict[cell_key].append(elem_id)
                            if cell_key not in cell_metadata:
                                cell_metadata[cell_key] = (
                                    np.asarray(reg_size).copy(), (i, j, k),
                                )
                            n_non_kuhn_cell_entries += 1
            else:
                # Centroid-only registration (legacy). Same lattice rule
                # as every other path: index on the level's lookup pitch.
                reg_size = lookup_pitch.get(int(nbr_level), nbr_size)
                centroid = vertices.mean(axis=0)
                i = int(np.floor(centroid[0] / reg_size[0]))
                j = int(np.floor(centroid[1] / reg_size[1]))
                k = int(np.floor(centroid[2] / reg_size[2]))
                i_m = int(np.clip(i + offset, 0, max_coord - 1))
                j_m = int(np.clip(j + offset, 0, max_coord - 1))
                k_m = int(np.clip(k + offset, 0, max_coord - 1))
                morton = encode_morton_3d_single(i_m, j_m, k_m, max_depth=21)
                cell_key = (morton, nbr_level)
                cell_to_elements_dict[cell_key].append(elem_id)
                if cell_key not in cell_metadata:
                    cell_metadata[cell_key] = (
                        np.asarray(reg_size).copy(), (i, j, k),
                    )
                n_non_kuhn_cell_entries += 1

            n_non_kuhn_registered += 1

        if verbose:
            print(f"  Non-Kuhn registered: {n_non_kuhn_registered:,} / "
                  f"{len(non_kuhn_ids):,}")
            if hybrid_non_kuhn and n_non_kuhn_registered > 0:
                avg = n_non_kuhn_cell_entries / n_non_kuhn_registered
                print(f"  Non-Kuhn registration mode: AABB-overlap "
                      f"({avg:.2f} cells/element on average)")
            elif n_non_kuhn_registered > 0:
                print(f"  Non-Kuhn registration mode: centroid (legacy)")
            if n_non_kuhn_orphan_fallback:
                print(f"  Non-Kuhn AABB-fallback (orphan): "
                      f"{n_non_kuhn_orphan_fallback:,}")
            if n_non_kuhn_dropped:
                print(f"  Non-Kuhn dropped (no fallback): "
                      f"{n_non_kuhn_dropped:,}")
            if exact_tet_box:
                _cand = n_non_kuhn_cell_entries + n_non_kuhn_cells_pruned
                print(f"  Exact tet-box pruning: kept "
                      f"{n_non_kuhn_cell_entries:,} of {_cand:,} candidate "
                      f"cells ({100.0 * n_non_kuhn_cell_entries / max(_cand, 1):.1f}%), "
                      f"pruned {n_non_kuhn_cells_pruned:,} the tet does not reach")
    else:
        if verbose:
            print(f"\n[3/4] Pass 2: No non-Kuhn elements — skipped")

    # ------------------------------------------------------------------
    # Build CSR
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[4/4] Building CSR structure...")

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
