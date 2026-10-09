#!/usr/bin/env python3
"""
Coverage audit: for particles that lost their host, is the element that
REALLY contains them registered in a cell the 3x3x3 search would visit?

This tests the registration-coverage hypothesis directly, without rerunning
MALMO. For each lost particle it:

  1. brute-force finds the element that geometrically contains its final
     position (point-in-tet over all elements, via a bounding-box prefilter),
  2. computes the cells that element IS registered in,
  3. computes the 3x3x3 cell neighbourhood the search would visit from the
     particle's position, at each octree level present,
  4. reports whether those two sets intersect.

Outcomes
--------
  no containing element found
      The particle is genuinely outside the mesh (advected out, or the free
      surface moved). Not a search bug.
  containing element found, registered in a VISITED cell
      The search should have found it -> the defect is in the SEARCH
      (traversal, point-in-tet tolerance, or level selection).
  containing element found, registered ONLY in cells NOT visited
      The defect is REGISTRATION COVERAGE: the element exists in the octree
      but not in any cell the search looks at. This is the hypothesis
      "we are searching in cells the elements are not registered to".
  containing element found, registered in NO cell at all
      The element was dropped from the octree entirely (the JAXTrace_stable
      orphan-drop path).

Usage
-----
  audit_octree_coverage.py --run <results_dir> --case <case.gid> \
      [--registration parent_cube|aabb|vertex_multi] [--max-particles 400] \
      [--out report.md]

Run inside the tracking singularity image (needs VTK + numpy; no GPU).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# This audit is pure NumPy, but importing the octree extractors pulls in JAX
# via jaxtrace. On a login node there is no GPU, and JAX aborts with
# "No visible GPU devices" unless it is told to use the CPU backend. Set this
# BEFORE any jaxtrace import.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")


def _vtk():
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError:
        sys.exit("ERROR: needs VTK. Run inside the tracking singularity image.")
    return vtk, vtk_to_numpy


def list_steps(run_dir: Path):
    pat = re.compile(r"particles_step_(\d+)\.vtu$")

    def scan(d: Path):
        try:
            names = os.listdir(d)
        except OSError:
            return []
        return [(int(m.group(1)), d / n) for n in names if (m := pat.match(n))]

    out = scan(run_dir)
    if not out:
        for sub in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            out = scan(sub)
            if out:
                break
    return sorted(out)


def read_particles(path: Path):
    vtk, vtk_to_numpy = _vtk()
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    g = r.GetOutput()
    pts = vtk_to_numpy(g.GetPoints().GetData())           # metres, as stored
    a = g.GetPointData().GetArray("ElementID")
    eid = vtk_to_numpy(a).astype(np.int64) if a is not None else None
    return pts, eid


def point_in_tet(p, v0, v1, v2, v3, tol=1e-12):
    """Barycentric containment for one point against many tets (vectorised)."""
    d0 = v1 - v0
    d1 = v2 - v0
    d2 = v3 - v0
    det = np.einsum("ij,ij->i", d0, np.cross(d1, d2))
    safe = np.where(np.abs(det) < 1e-300, 1.0, det)
    r = p[None, :] - v0
    b1 = np.einsum("ij,ij->i", r, np.cross(d1, d2)) / safe
    b2 = np.einsum("ij,ij->i", d0, np.cross(r, d2)) / safe
    b3 = np.einsum("ij,ij->i", d0, np.cross(d1, r)) / safe
    b0 = 1.0 - b1 - b2 - b3
    ok = (b0 >= -tol) & (b1 >= -tol) & (b2 >= -tol) & (b3 >= -tol)
    return ok & (np.abs(det) > 1e-300)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="results dir")
    ap.add_argument("--case", type=Path, required=True, help="<case>.gid dir")
    ap.add_argument("--registration", default="parent_cube",
                    choices=["parent_cube", "aabb", "vertex_multi"])
    ap.add_argument("--max-particles", type=int, default=400,
                    help="how many lost particles to audit (default 400)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    L = []
    def say(line=""):
        print(line)
        L.append(line)

    steps = list_steps(args.run)
    if not steps:
        sys.exit(f"ERROR: no particle files under {args.run}")
    pF, eF = read_particles(steps[-1][1])
    if eF is None:
        sys.exit("ERROR: no ElementID array; re-run with --export-element-ids")
    lost_idx = np.flatnonzero(eF < 0)
    say(f"# Octree coverage audit — `{args.run.name}`")
    say()
    say(f"Final step {steps[-1][0]}; {len(lost_idx):,} particles with "
        f"`ElementID < 0` out of {len(pF):,}.")
    say()
    if not len(lost_idx):
        say("No lost particles — nothing to audit.")
        _write(args.out, L)
        return

    rng = np.random.default_rng(42)
    if len(lost_idx) > args.max_particles:
        sel = rng.choice(lost_idx, args.max_particles, replace=False)
        say(f"Auditing a random sample of {len(sel):,} (seed 42).")
    else:
        sel = lost_idx
        say(f"Auditing all {len(sel):,}.")
    say()

    # ---- load the mesh the run used ------------------------------------
    from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
    from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

    post = args.case / "post"
    pvtus = sorted(post.glob("*_34.pvtu")) or sorted(post.glob("*.pvtu"))
    if not pvtus:
        sys.exit(f"ERROR: no .pvtu under {post}")
    pattern = re.sub(r"_(\d+)\.pvtu$", "_{timestep}.pvtu", pvtus[0].name)
    ts = int(re.search(r"_(\d+)\.pvtu$", pvtus[0].name).group(1))
    say(f"Mesh: `{pvtus[0].name}` (pattern `{pattern}`, timestep {ts})")
    say()

    node_positions, connectivity, vel_seq = load_velocity_sequence_from_pvtu(
        base_path=post, file_pattern=pattern, timestep_range=(ts, ts),
        field_name="Displacement", verbose=False,
    )
    # deduplicate_nodes returns (positions, connectivity, n_dup, velocity_seq);
    # match run_tracking.py:1862 so the mesh is identical to the tracked one.
    node_positions, connectivity, n_dup, _ = deduplicate_nodes(
        node_positions, connectivity,
        velocity_sequence=vel_seq, verbose=False,
    )
    connectivity = connectivity.astype(np.int32)
    say(f"Mesh: {len(node_positions):,} nodes, {len(connectivity):,} elements "
        f"({n_dup:,} duplicate nodes merged)")
    say()

    # ---- build the SAME octree the run used ----------------------------
    if args.registration == "aabb":
        from jaxtrace.gpu.search.mesh_aligned_octree_aabb import (
            extract_octree_cells_aabb as extract)
        cells = extract(node_positions, connectivity, tolerance=1e-6, verbose=False)
    elif args.registration == "vertex_multi":
        from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import (
            extract_octree_cells_vertex_multi as extract)
        cells = extract(node_positions, connectivity, tolerance=1e-6, verbose=False)
    else:
        from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import (
            extract_octree_cells_parent_cube as extract)
        cells = extract(node_positions, connectivity, tolerance=1e-6, verbose=False)
    say(f"Octree (`{args.registration}`): {cells.n_cells:,} cells, "
        f"{cells.elements_per_cell_mean:.1f} elem/cell, "
        f"max {cells.max_elements_per_cell}")
    say()

    # element -> set of cell rows it is registered in
    offs = np.asarray(cells.cell_to_elements_offsets)
    data = np.asarray(cells.cell_to_elements_data)
    cell_of_entry = np.repeat(np.arange(len(offs) - 1), np.diff(offs))
    order = np.argsort(data, kind="stable")
    elem_sorted = data[order]
    cell_sorted = cell_of_entry[order]
    elem_start = np.searchsorted(elem_sorted, np.arange(len(connectivity)), "left")
    elem_end = np.searchsorted(elem_sorted, np.arange(len(connectivity)), "right")

    grid_idx = np.asarray(cells.cell_grid_indices)
    levels = np.asarray(cells.cell_levels)
    sizes = np.asarray(cells.cell_sizes)
    # (level, i, j, k) -> row, for the visited-set test
    key_to_row = {}
    for row in range(len(levels)):
        key_to_row[(int(levels[row]), *map(int, grid_idx[row]))] = row

    # The SEARCH does floor(pos / level_cell_sizes[level]), where
    # level_cell_sizes[level] is the size of the FIRST cell seen at that level
    # (mesh_aligned_octree_gpu.py:452-457, "all should be nearly identical per
    # level"). Replicate that exactly: using each cell's own stored size would
    # place the visited set differently wherever sizes vary within a level,
    # which is precisely what a non-Kuhn mesh produces.
    level_cell_size = {}
    for lvl in np.unique(levels):
        first = np.flatnonzero(levels == lvl)[0]
        level_cell_size[int(lvl)] = sizes[first]
    say(f"Levels present: {sorted(level_cell_size)} "
        f"(canonical cell size per level, as the search uses)")
    say()

    # element AABBs for the brute-force prefilter
    v = node_positions[connectivity]           # (n_elem, 4, 3)
    emin = v.min(axis=1)
    emax = v.max(axis=1)

    # ---- audit ---------------------------------------------------------
    counts = dict(outside=0, visited=0, not_visited=0, unregistered=0)
    examples = []

    for pi in sel:
        p = pF[pi]
        cand = np.flatnonzero(
            (emin[:, 0] <= p[0]) & (emax[:, 0] >= p[0]) &
            (emin[:, 1] <= p[1]) & (emax[:, 1] >= p[1]) &
            (emin[:, 2] <= p[2]) & (emax[:, 2] >= p[2])
        )
        host = -1
        if len(cand):
            inside = point_in_tet(p, v[cand, 0], v[cand, 1], v[cand, 2], v[cand, 3])
            if inside.any():
                host = int(cand[np.flatnonzero(inside)[0]])

        if host < 0:
            counts["outside"] += 1
            continue

        rows = cell_sorted[elem_start[host]:elem_end[host]]
        if not len(rows):
            counts["unregistered"] += 1
            if len(examples) < 5:
                examples.append((int(pi), host, "registered in NO cell"))
            continue

        # Which cells would the 3x3x3 search visit from p? For each level the
        # element's cells live at, take the 27 neighbours of p's own cell.
        visited = False
        row_set = set(rows.tolist())
        for lvl in sorted({int(levels[r]) for r in rows}):
            sz = level_cell_size[lvl]
            base = np.floor(p / sz).astype(np.int64)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for dk in (-1, 0, 1):
                        k = (lvl, int(base[0]) + di, int(base[1]) + dj,
                             int(base[2]) + dk)
                        r2 = key_to_row.get(k)
                        if r2 is not None and r2 in row_set:
                            visited = True
                            break
                    if visited: break
                if visited: break
            if visited: break

        if visited:
            counts["visited"] += 1
            if len(examples) < 5:
                examples.append((int(pi), host, "in a VISITED cell"))
        else:
            counts["not_visited"] += 1
            if len(examples) < 5:
                examples.append((int(pi), host, "only in NON-visited cells"))

    tot = sum(counts.values())
    say("## Result")
    say()
    say("| outcome | n | % | meaning |")
    say("|---|---:|---:|---|")
    say(f"| no containing element | {counts['outside']:,} | "
        f"{100*counts['outside']/tot:.1f} | genuinely outside the mesh — not a search bug |")
    say(f"| host in a VISITED cell | {counts['visited']:,} | "
        f"{100*counts['visited']/tot:.1f} | **SEARCH defect** — it should have been found |")
    say(f"| host only in NON-visited cells | {counts['not_visited']:,} | "
        f"{100*counts['not_visited']/tot:.1f} | **REGISTRATION COVERAGE defect** |")
    say(f"| host registered in NO cell | {counts['unregistered']:,} | "
        f"{100*counts['unregistered']/tot:.1f} | element dropped from the octree |")
    say()

    dom = max(counts, key=counts.get)
    verdict = {
        "outside": "Most lost particles are genuinely outside the mesh. The "
                   "search is behaving correctly; look at the free surface / "
                   "boundary handling instead.",
        "visited": "The containing element IS in a cell the search visits, so "
                   "the registration is adequate and the defect is in the "
                   "SEARCH itself — traversal, level selection, or the "
                   "point-in-tet tolerance.",
        "not_visited": "The containing element is registered, but only in cells "
                       "the 3x3x3 neighbourhood never visits. This is the "
                       "registration-coverage failure: widening the search "
                       "bands cannot fix it, but AABB-overlap registration "
                       "(--registration aabb) should.",
        "unregistered": "The containing element is in NO octree cell at all — "
                        "it was dropped at build time. Check for the "
                        "orphan-drop path (JAXTrace_stable) and rebuild with "
                        "orphan_fallback enabled.",
    }[dom]
    say(f"**Verdict:** {verdict}")
    say()
    if examples:
        say("Examples:")
        say()
        for pi, host, what in examples:
            say(f"- particle {pi}: true host element {host}, {what}")
        say()

    say("_Caveat: the audit rebuilds the octree from the mesh at a single "
        "timestep. If the mesh moves between timesteps, a particle may have "
        "been lost against a different mesh state than the one audited here._")
    _write(args.out, L)


def _write(out: Path | None, L: list[str]):
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"\n[report written to {out}]")


if __name__ == "__main__":
    main()
