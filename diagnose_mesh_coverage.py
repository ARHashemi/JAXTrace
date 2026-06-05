"""
diagnose_mesh_coverage.py
-------------------------

Audit an unstructured tetrahedral mesh for the issues that cause the
"particle stuck at element face, flagged as Escaped, while geometrically
inside the domain" symptom.

What it reports
===============
1. Element-count breakdown:
     total elements, Kuhn, non-Kuhn, non-Kuhn registered via:
       - face-neighbour Kuhn grid
       - node-neighbour Kuhn grid
       - AABB orphan fallback (median Kuhn cell)
       - dropped (orphan_fallback=False)
   So you can see how many tets are non-Kuhn-island.

2. Coverage check (the key diagnostic):
     for every element, walk the cell(s) it is registered in plus the
     surrounding 3x3x3 cell neighbourhood at every level that cell sits
     on. Check whether the element's *vertices* and *centroid* all fall
     inside this neighbourhood's spatial bounds.
     -> Elements whose vertices fall OUTSIDE their own neighbourhood
        are coverage holes: a particle inside that element will not be
        found by the L0/L1/L2 search because the search visits only
        the centroid-cell's 3x3x3 window.

3. Level-range check:
     report the (min, max) of cell_levels and warn if levels fall
     outside the kernel's hard-coded scan range [7, 14]. Cells at
     levels outside that range are never visited.

4. Optional sampling of points reported lost:
     give it a .npy of N x 3 lost-particle positions (e.g. exported
     from a stuck-particle run) and it will, for each, run the same
     element-id search the kernel would and print the result.

Usage
=====
  python diagnose_mesh_coverage.py \\
      --input <case>.gid/post \\
      --mesh-pattern '<stem>_{timestep}.pvtu' \\
      --vel-start 0 \\
      [--registration parent_cube] \\
      [--max-elements 0] \\
      [--lost-positions stuck.npy]

  --max-elements N : only run the coverage check on the first N elements
                     (default 0 = all). Use for sanity check on huge
                     meshes; default 0 still works but is O(n_elements).
"""

import argparse
import numpy as np
from pathlib import Path

# Reuse the same loading + registration paths as run_tracking.py so we
# audit the exact data structure that the kernel will see.
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import (
    extract_octree_cells_parent_cube,
)
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import (
    extract_octree_cells_vertex_multi,
)
from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import (
    encode_morton_3d_single,
    find_axis_aligned_edges_single,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _cell_aabb_from_key(morton, level, cell_size, grid_indices):
    """Return (lo, hi) of the cell's spatial AABB.

    We use the (cell_size, grid_indices) stored in the cell metadata —
    those are the same numbers the kernel will use when hashing a
    particle position to this cell at search time.
    """
    i, j, k = grid_indices
    lo = np.array([i * cell_size[0], j * cell_size[1], k * cell_size[2]],
                  dtype=np.float64)
    hi = lo + cell_size
    return lo, hi


def _neighborhood_aabb(cell_size, grid_indices, half_window=1):
    """3x3x3 (half_window=1) or 5x5x5 (half_window=2) neighbourhood AABB
    in cell-grid coordinates."""
    i, j, k = grid_indices
    n = half_window
    lo = np.array([
        (i - n) * cell_size[0],
        (j - n) * cell_size[1],
        (k - n) * cell_size[2],
    ], dtype=np.float64)
    hi = np.array([
        (i + n + 1) * cell_size[0],
        (j + n + 1) * cell_size[1],
        (k + n + 1) * cell_size[2],
    ], dtype=np.float64)
    return lo, hi


# ---------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------
def coverage_check(node_positions, connectivity, octree, max_elements=0,
                   half_window=1):
    """For each element, verify that the 3x3x3 cell neighbourhood
    around its registered cell actually covers every vertex of the
    element. Elements that have a vertex *outside* the neighbourhood
    are coverage holes — a particle landing on that vertex (or near it)
    will not find the element via L0/L1/L2 search.

    Returns a per-element dict of:
       n_total, n_covered_ok, n_coverage_hole, hole_element_ids[:50]
    """
    n_elements = connectivity.shape[0]
    if max_elements and max_elements < n_elements:
        n_check = max_elements
    else:
        n_check = n_elements

    # Build a map from element_id -> list of (cell_key, cell_idx).
    # The CSR storage is cell -> elements, so we invert it.
    elem_to_cells = [[] for _ in range(n_elements)]
    for cell_idx in range(octree.n_cells):
        lo = octree.cell_to_elements_offsets[cell_idx]
        hi = octree.cell_to_elements_offsets[cell_idx + 1]
        for elem_id in octree.cell_to_elements_data[lo:hi]:
            elem_to_cells[int(elem_id)].append(cell_idx)

    n_zero_cells = 0
    n_covered_ok = 0
    n_coverage_hole = 0
    hole_examples = []

    for elem_id in range(n_check):
        cells = elem_to_cells[elem_id]
        if not cells:
            n_zero_cells += 1
            if len(hole_examples) < 50:
                hole_examples.append((int(elem_id), 'no_cells'))
            continue

        verts = node_positions[connectivity[elem_id]]  # (4, 3)
        centroid = verts.mean(axis=0)
        all_points = np.vstack([verts, centroid[None, :]])  # (5, 3)

        # The element is "covered" if every (vertex, centroid) point
        # falls inside the 3x3x3 spatial AABB of AT LEAST ONE of its
        # registered cells at that cell's level.
        covered_mask = np.zeros(all_points.shape[0], dtype=bool)
        for cidx in cells:
            cell_size = octree.cell_sizes[cidx]
            grid_idx = octree.cell_grid_indices[cidx]
            lo, hi = _neighborhood_aabb(cell_size, grid_idx, half_window)
            in_box = ((all_points >= lo[None, :]).all(axis=1)
                      & (all_points <= hi[None, :]).all(axis=1))
            covered_mask |= in_box

        if covered_mask.all():
            n_covered_ok += 1
        else:
            n_coverage_hole += 1
            if len(hole_examples) < 50:
                hole_examples.append((int(elem_id), 'partial'))

    return {
        'n_total': int(n_check),
        'n_covered_ok': int(n_covered_ok),
        'n_coverage_hole': int(n_coverage_hole),
        'n_zero_cells': int(n_zero_cells),
        'hole_examples': hole_examples,
    }


# ---------------------------------------------------------------------
# Level-range check
# ---------------------------------------------------------------------
def level_range_check(octree, kernel_min_level=7, kernel_max_level=14):
    """Report cell-level distribution and warn about cells outside the
    kernel's hard-coded scan range."""
    levels, counts = np.unique(octree.cell_levels, return_counts=True)
    out_of_range = 0
    print(f"\n  Cell-level distribution:")
    for lv, ct in zip(levels, counts):
        flag = ""
        if lv < kernel_min_level or lv > kernel_max_level:
            flag = (f"  <-- OUT OF KERNEL RANGE "
                    f"[{kernel_min_level}..{kernel_max_level}], "
                    f"NEVER VISITED")
            out_of_range += int(ct)
        print(f"    Level {int(lv):2d}: {int(ct):>10,d} cells{flag}")
    if out_of_range:
        print(f"\n  WARNING: {out_of_range:,} cells are at levels the "
              f"kernel does not iterate. Particles that hash there will "
              f"miss every element registered in those cells.")
    return out_of_range


# ---------------------------------------------------------------------
# Lost-position replay
# ---------------------------------------------------------------------
def replay_lost_positions(positions, node_positions, connectivity, octree,
                          half_window=1, tol=1e-6, max_show=20):
    """For each of `positions[:, :3]`, find which cells the position
    hashes to at each level the octree uses, list candidate elements,
    and run a point-in-tet test on each. Report whether any element
    contains the point.
    """
    cell_keys = {}
    for cidx in range(octree.n_cells):
        key = (int(octree.cell_morton_codes[cidx]), int(octree.cell_levels[cidx]))
        cell_keys[key] = cidx

    n_found = 0
    n_miss = 0
    examples = []
    for pi, pos in enumerate(positions):
        found = False
        candidates = []
        for level in sorted(set(int(lv) for lv in octree.cell_levels)):
            # cell_size at this level: take any cell at this level
            # (they all share cell_size by construction).
            level_mask = octree.cell_levels == level
            if not level_mask.any():
                continue
            csz = octree.cell_sizes[level_mask][0]
            i = int(np.floor(pos[0] / csz[0]))
            j = int(np.floor(pos[1] / csz[1]))
            k = int(np.floor(pos[2] / csz[2]))
            offset = 1 << 19
            max_coord = 1 << 20
            for di in range(-half_window, half_window + 1):
                for dj in range(-half_window, half_window + 1):
                    for dk in range(-half_window, half_window + 1):
                        ii = int(np.clip(i + di + offset, 0, max_coord - 1))
                        jj = int(np.clip(j + dj + offset, 0, max_coord - 1))
                        kk = int(np.clip(k + dk + offset, 0, max_coord - 1))
                        m = encode_morton_3d_single(ii, jj, kk, max_depth=21)
                        cidx = cell_keys.get((m, level))
                        if cidx is None:
                            continue
                        lo = octree.cell_to_elements_offsets[cidx]
                        hi = octree.cell_to_elements_offsets[cidx + 1]
                        for elem_id in octree.cell_to_elements_data[lo:hi]:
                            candidates.append(int(elem_id))
        candidates = list(set(candidates))
        # Point-in-tet test on each candidate
        for elem_id in candidates:
            verts = node_positions[connectivity[elem_id]]
            # Barycentric via 4x4 solve
            T = np.zeros((4, 4), dtype=np.float64)
            T[:3, :] = verts.T
            T[3, :] = 1.0
            b = np.array([pos[0], pos[1], pos[2], 1.0], dtype=np.float64)
            try:
                bary = np.linalg.solve(T, b)
            except np.linalg.LinAlgError:
                continue
            if np.all(bary >= -tol) and np.all(bary <= 1.0 + tol):
                found = True
                if len(examples) < max_show:
                    examples.append((
                        int(pi), pos.tolist(), int(elem_id),
                        bary.tolist(), len(candidates),
                    ))
                break
        if found:
            n_found += 1
        else:
            n_miss += 1
            if len(examples) < max_show:
                examples.append((
                    int(pi), pos.tolist(), -1, None, len(candidates),
                ))
    return {
        'n_found': n_found, 'n_miss': n_miss, 'examples': examples,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True,
                    help='Path to <case>.gid/post (PVTU directory)')
    ap.add_argument('--mesh-pattern', required=True,
                    help='e.g. "B4_{timestep}.pvtu"')
    ap.add_argument('--vel-start', type=int, default=0,
                    help='Single timestep used to load mesh topology')
    ap.add_argument('--velocity-field', type=str, default='Displacement')
    ap.add_argument('--registration', choices=['parent_cube', 'vertex_multi'],
                    default='parent_cube')
    ap.add_argument('--max-elements', type=int, default=0,
                    help='Coverage check: limit to first N elements (0=all)')
    ap.add_argument('--half-window', type=int, default=1,
                    help='Coverage neighbourhood: 1 = 3x3x3, 2 = 5x5x5')
    ap.add_argument('--lost-positions', type=str, default=None,
                    help='Optional .npy of (N,3) positions to replay')
    args = ap.parse_args()

    print("=" * 80)
    print("Mesh coverage diagnostic")
    print("=" * 80)

    print(f"\n[1/4] Loading mesh ({args.input})...")
    node_positions, connectivity, _ = load_velocity_sequence_from_pvtu(
        base_path=args.input,
        file_pattern=args.mesh_pattern,
        timestep_range=(args.vel_start, args.vel_start),
        field_name=args.velocity_field,
        verbose=False,
    )
    print(f"  Elements: {connectivity.shape[0]:,}, "
          f"Nodes: {node_positions.shape[0]:,}")
    node_positions, connectivity, n_dup, _ = deduplicate_nodes(
        node_positions, connectivity, verbose=False,
    )
    connectivity = connectivity.astype(np.int32)
    print(f"  Deduplicated: removed {n_dup:,} duplicates -> "
          f"{node_positions.shape[0]:,} nodes")
    print(f"  Bbox: [{node_positions.min(axis=0)}] -> "
          f"[{node_positions.max(axis=0)}]")

    print(f"\n[2/4] Classifying Kuhn vs non-Kuhn elements...")
    n_elements = connectivity.shape[0]
    is_kuhn = np.zeros(n_elements, dtype=bool)
    for elem_id in range(n_elements):
        vs = node_positions[connectivity[elem_id]]
        cs, _ = find_axis_aligned_edges_single(vs, tolerance=1e-6)
        is_kuhn[elem_id] = bool(np.all(cs > 0))
    n_kuhn = int(is_kuhn.sum())
    n_non_kuhn = n_elements - n_kuhn
    print(f"  Kuhn:     {n_kuhn:,} ({100*n_kuhn/n_elements:.2f}%)")
    print(f"  Non-Kuhn: {n_non_kuhn:,} ({100*n_non_kuhn/n_elements:.2f}%)")

    print(f"\n[3/4] Building octree ({args.registration})...")
    if args.registration == 'parent_cube':
        octree = extract_octree_cells_parent_cube(
            node_positions, connectivity, tolerance=1e-6, verbose=True,
            orphan_fallback=True,
        )
    else:
        octree = extract_octree_cells_vertex_multi(
            node_positions, connectivity, tolerance=1e-6, verbose=True,
            orphan_fallback=True,
        )

    out_of_range = level_range_check(octree)

    print(f"\n[4/4] Coverage check (3x3x3 spatial neighbourhood)...")
    cov = coverage_check(
        node_positions, connectivity, octree,
        max_elements=args.max_elements, half_window=args.half_window,
    )
    print(f"\n  Total checked:       {cov['n_total']:,}")
    print(f"  Fully covered:       {cov['n_covered_ok']:,} "
          f"({100*cov['n_covered_ok']/cov['n_total']:.4f}%)")
    print(f"  Coverage holes:      {cov['n_coverage_hole']:,} "
          f"({100*cov['n_coverage_hole']/cov['n_total']:.4f}%)")
    print(f"  Elements w/o cells:  {cov['n_zero_cells']:,}")
    if cov['hole_examples']:
        print(f"  First few hole examples (elem_id, kind):")
        for eid, kind in cov['hole_examples'][:20]:
            print(f"    {eid:>10d}  {kind}")

    if args.lost_positions and Path(args.lost_positions).exists():
        print(f"\n[bonus] Replaying lost positions from "
              f"{args.lost_positions}...")
        positions = np.load(args.lost_positions)
        print(f"  {len(positions):,} positions to test")
        result = replay_lost_positions(
            positions, node_positions, connectivity, octree,
            half_window=args.half_window,
        )
        print(f"  Found a host:        {result['n_found']:,}")
        print(f"  Truly unreachable:   {result['n_miss']:,}")
        if result['examples']:
            print(f"  First examples (idx, pos, elem_id, bary, n_cands):")
            for ex in result['examples'][:20]:
                print(f"    {ex}")

    print("\nDone.")


if __name__ == '__main__':
    main()
