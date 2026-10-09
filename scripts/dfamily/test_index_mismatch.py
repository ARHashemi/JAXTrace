#!/usr/bin/env python3
"""
DECISIVE TEST: does the search compute the WRONG cell index for the particles
it loses?

The hypothesis
--------------
`level` is derived from the MEAN of an anisotropic cell_size
(mesh_aligned_octree_single_cell.py:101-104):

    avg_size = np.mean(cell_size)        # cell_size is [dx, dy, dz]
    level    = round(-np.log2(avg_size))

so two cells with DIFFERENT per-axis sizes can share one level. The GPU upload
then collapses each level to a single size — the FIRST cell seen at that level
(mesh_aligned_octree_gpu.py:452-457, "all should be nearly identical per
level") — and the search indexes with that one pitch
(mesh_aligned_point_location.py:206-211):

    cell_size = octree_gpu.level_cell_sizes[level]
    i = floor(pos[0] / cell_size[0])

If a particle's true host cell has a different size from its level's first
cell, `i` is computed on the wrong grid pitch, the 3x3x3 neighbourhood centres
on the wrong cell, and the host can lie outside all 27 cells visited.

What this script measures, per particle
---------------------------------------
  1. brute-force the element that really contains the particle's final position
  2. find the octree cell(s) that element is registered in, and their (level,
     grid index) as stored at BUILD time
  3. recompute the index the SEARCH would use: floor(pos / level_cell_sizes[L])
     with level_cell_sizes[L] = the first cell's size at level L, exactly as
     upload_mesh_aligned_octree_to_gpu does
  4. report whether the search's index is within +-1 of the stored index
     (i.e. inside the 3x3x3 neighbourhood) or outside it

Read it like this
-----------------
  LOST particles mostly INDEX-MISMATCHED, survivors mostly matched
      -> hypothesis CONFIRMED. The wrong-pitch index is the mechanism.
  LOST and survivors mismatch at the SAME rate
      -> hypothesis REFUTED. Index mismatch is common and harmless; the loss
         has another cause. Report that plainly.
  LOST particles mostly index-MATCHED
      -> hypothesis REFUTED; look at the point-in-tet test or level ordering.

Usage
-----
  test_index_mismatch.py --run <results_dir> --case <case>.gid
      [--registration parent_cube|aabb|vertex_multi]
      [--max-particles 300] [--out report.md]

Needs ~240 GB for a 10.8M-element D mesh; run it as a batch job, not on a
login node. No GPU: pure NumPy, JAX pinned to CPU.
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

# Importing the octree extractors pulls in JAX; on a CPU node it aborts with
# "No visible GPU devices" unless told to use the CPU backend. Set before any
# jaxtrace import.
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
    pts = vtk_to_numpy(g.GetPoints().GetData())          # metres, as stored
    a = g.GetPointData().GetArray("ElementID")
    eid = vtk_to_numpy(a).astype(np.int64) if a is not None else None
    return pts, eid


def point_in_tet(p, v0, v1, v2, v3, tol=1e-12):
    """Barycentric containment of one point against many tets (vectorised)."""
    d0, d1, d2 = v1 - v0, v2 - v0, v3 - v0
    det = np.einsum("ij,ij->i", d0, np.cross(d1, d2))
    safe = np.where(np.abs(det) < 1e-300, 1.0, det)
    r = p[None, :] - v0
    b1 = np.einsum("ij,ij->i", r, np.cross(d1, d2)) / safe
    b2 = np.einsum("ij,ij->i", d0, np.cross(r, d2)) / safe
    b3 = np.einsum("ij,ij->i", d0, np.cross(d1, r)) / safe
    b0 = 1.0 - b1 - b2 - b3
    return ((b0 >= -tol) & (b1 >= -tol) & (b2 >= -tol) & (b3 >= -tol)
            & (np.abs(det) > 1e-300))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--registration", default="parent_cube",
                    choices=["parent_cube", "aabb", "vertex_multi"])
    ap.add_argument("--max-particles", type=int, default=300,
                    help="lost particles to test, and the same number of "
                         "survivors as a control (default 300)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    L = []
    def say(line=""):
        print(line, flush=True)
        L.append(line)

    steps = list_steps(args.run)
    if not steps:
        sys.exit(f"ERROR: no particle files under {args.run}")
    pF, eF = read_particles(steps[-1][1])
    if eF is None:
        sys.exit("ERROR: no ElementID array; re-run with --export-element-ids")

    say(f"# Index-mismatch test — `{args.run.name}` (`{args.registration}`)")
    say()
    say(f"Final step {steps[-1][0]}; {len(pF):,} particles, "
        f"{int((eF < 0).sum()):,} with `ElementID < 0`.")
    say()

    # ---- rebuild the mesh exactly as the run did -----------------------
    from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
    from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

    post = args.case / "post"
    pv = sorted(post.glob("*_34.pvtu")) or sorted(post.glob("*.pvtu"))
    if not pv:
        sys.exit(f"ERROR: no .pvtu under {post}")
    pattern = re.sub(r"_(\d+)\.pvtu$", "_{timestep}.pvtu", pv[0].name)
    ts = int(re.search(r"_(\d+)\.pvtu$", pv[0].name).group(1))
    npos, conn, vel = load_velocity_sequence_from_pvtu(
        base_path=post, file_pattern=pattern, timestep_range=(ts, ts),
        field_name="Displacement", verbose=False)
    npos, conn, ndup, _ = deduplicate_nodes(
        npos, conn, velocity_sequence=vel, verbose=False)
    conn = conn.astype(np.int32)
    say(f"Mesh `{pv[0].name}`: {len(npos):,} nodes, {len(conn):,} elements.")
    say()

    # ---- rebuild the octree exactly as the run did ----------------------
    if args.registration == "aabb":
        from jaxtrace.gpu.search.mesh_aligned_octree_aabb import (
            extract_octree_cells_aabb as extract)
    elif args.registration == "vertex_multi":
        from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import (
            extract_octree_cells_vertex_multi as extract)
    else:
        from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import (
            extract_octree_cells_parent_cube as extract)
    cells = extract(npos, conn, tolerance=1e-6, verbose=False)
    lev = np.asarray(cells.cell_levels)
    sz = np.asarray(cells.cell_sizes)
    gidx = np.asarray(cells.cell_grid_indices)
    say(f"Octree: {cells.n_cells:,} cells, levels "
        f"{sorted(int(x) for x in np.unique(lev))}.")
    say()

    # ---- replicate level_cell_sizes EXACTLY as the GPU upload does -----
    # mesh_aligned_octree_gpu.py:452-457 — "Use first cell size for this level
    # (all should be nearly identical per level)". That assumption is what we
    # are testing.
    level_first_size = {}
    level_spread = {}
    for Lv in np.unique(lev):
        m = lev == Lv
        s = sz[m]
        level_first_size[int(Lv)] = s[0]
        level_spread[int(Lv)] = float(np.max(s.max(axis=0) / np.maximum(s[0], 1e-300)))

    say("## 0. Is the one-size-per-level assumption violated?")
    say()
    say("| level | n cells | first cell size (dx) | max/first | verdict |")
    say("|---|---:|---:|---:|---|")
    for Lv in sorted(level_first_size):
        n = int((lev == Lv).sum())
        f = level_first_size[Lv]
        sp = level_spread[Lv]
        v = "**mixed sizes**" if sp > 1.5 else "round-off only"
        say(f"| {Lv} | {n:,} | {f[0]:.6e} | {sp:.6f} | {v} |")
    say()

    # element -> rows (cells) it is registered in
    offs = np.asarray(cells.cell_to_elements_offsets)
    data = np.asarray(cells.cell_to_elements_data)
    cell_of_entry = np.repeat(np.arange(len(offs) - 1), np.diff(offs))
    order = np.argsort(data, kind="stable")
    el_sorted = data[order]
    cell_sorted = cell_of_entry[order]
    n_elem = len(conn)
    e_start = np.searchsorted(el_sorted, np.arange(n_elem), "left")
    e_end = np.searchsorted(el_sorted, np.arange(n_elem), "right")

    v = npos[conn]
    emin = v.min(axis=1)
    emax = v.max(axis=1)

    def classify(idxs, label):
        """For each particle: is the search's index within +-1 of a cell the
        true host is registered in?"""
        res = dict(no_host=0, matched=0, mismatched=0)
        detail = []
        for pi in idxs:
            p = pF[pi]
            cand = np.flatnonzero(
                (emin[:, 0] <= p[0]) & (emax[:, 0] >= p[0]) &
                (emin[:, 1] <= p[1]) & (emax[:, 1] >= p[1]) &
                (emin[:, 2] <= p[2]) & (emax[:, 2] >= p[2]))
            host = -1
            if len(cand):
                ins = point_in_tet(p, v[cand, 0], v[cand, 1], v[cand, 2], v[cand, 3])
                if ins.any():
                    host = int(cand[np.flatnonzero(ins)[0]])
            if host < 0:
                res["no_host"] += 1
                continue
            rows = cell_sorted[e_start[host]:e_end[host]]
            if not len(rows):
                res["mismatched"] += 1
                continue
            ok = False
            worst = None
            for row in rows:
                Lv = int(lev[row])
                pitch = level_first_size[Lv]          # what the SEARCH uses
                search_idx = np.floor(p / pitch).astype(np.int64)
                stored_idx = gidx[row].astype(np.int64)
                d = np.abs(search_idx - stored_idx)
                if np.all(d <= 1):                    # inside the 3x3x3
                    ok = True
                    break
                if worst is None or d.max() > worst[0]:
                    worst = (int(d.max()), Lv, tuple(search_idx), tuple(stored_idx),
                             float(sz[row][0] / pitch[0]))
            if ok:
                res["matched"] += 1
            else:
                res["mismatched"] += 1
                if len(detail) < 6 and worst:
                    detail.append((int(pi), host) + worst)
        return res, detail

    rng = np.random.default_rng(42)
    lost_all = np.flatnonzero(eF < 0)
    surv_all = np.flatnonzero(eF >= 0)
    n = args.max_particles
    lost = rng.choice(lost_all, min(n, len(lost_all)), replace=False) if len(lost_all) else np.array([], int)
    surv = rng.choice(surv_all, min(n, len(surv_all)), replace=False) if len(surv_all) else np.array([], int)

    say("## 1. Index mismatch: lost particles vs surviving particles")
    say()
    rl, dl = classify(lost, "lost") if len(lost) else (dict(no_host=0, matched=0, mismatched=0), [])
    rs, _ = classify(surv, "survivors") if len(surv) else (dict(no_host=0, matched=0, mismatched=0), [])

    say("| group | n tested | no containing element | index MATCHED | index MISMATCHED |")
    say("|---|---:|---:|---:|---:|")
    for name, r in (("lost (`ElementID<0`)", rl), ("survivors (control)", rs)):
        t = max(sum(r.values()), 1)
        say(f"| {name} | {sum(r.values()):,} | {r['no_host']:,} | "
            f"{r['matched']:,} ({100*r['matched']/t:.1f}%) | "
            f"**{r['mismatched']:,} ({100*r['mismatched']/t:.1f}%)** |")
    say()

    # ---- verdict -------------------------------------------------------
    tl = max(rl["matched"] + rl["mismatched"], 1)
    tscv = max(rs["matched"] + rs["mismatched"], 1)
    fl = 100 * rl["mismatched"] / tl
    fs = 100 * rs["mismatched"] / tscv
    say("## 2. Verdict")
    say()
    say(f"Index mismatch rate — lost: **{fl:.1f}%**, survivors: **{fs:.1f}%**.")
    say()
    if fl > 70 and fs < 30:
        say("**CONFIRMED.** The search computes an out-of-neighbourhood cell")
        say("index for the particles it loses, and the correct index for the")
        say("ones it keeps. The wrong grid pitch is the mechanism: `level` is")
        say("derived from the MEAN of an anisotropic `cell_size`, so cells of")
        say("different size share a level, and the search indexes them all")
        say("with the level's first cell size.")
        say()
        say("This is a build-time defect, not a search-kernel defect: the fix")
        say("is to stop putting differently-sized cells on one level.")
    elif abs(fl - fs) < 15:
        say("**REFUTED.** Lost and surviving particles mismatch at a similar")
        say("rate, so index mismatch is common and largely harmless. The loss")
        say("has another cause — look at the point-in-tet tolerance, the level")
        say("iteration order, or the velocity/geometry handling at the tool")
        say("boundary.")
    elif fl < 30:
        say("**REFUTED.** The lost particles mostly have a CORRECT index, so")
        say("the search was looking in the right cell and still failed. Suspect")
        say("the point-in-tet test (tolerance at the tool boundary) or the")
        say("per-level iteration order.")
    else:
        say("**INCONCLUSIVE.** The signal is in the right direction but not")
        say("clean. Raise `--max-particles` and re-run before drawing a")
        say("conclusion.")
    say()

    if dl:
        say("### Example mismatches (lost particles)")
        say()
        say("| particle | true host | level | max |Δindex| | search index | stored index | cell size / level pitch |")
        say("|---|---:|---:|---:|---|---|---:|")
        for pi, host, dmax, Lv, sidx, stidx, ratio in dl:
            say(f"| {pi} | {host} | {Lv} | {dmax} | {sidx} | {stidx} | {ratio:.3f} |")
        say()
        say("A `cell size / level pitch` of ~2.0 is the smoking gun: that cell")
        say("is twice the size the search assumes for its level.")
        say()

    say("_Caveat: the octree is rebuilt from a single mesh timestep. If the "
        "mesh moves between timesteps, a particle may have been lost against "
        "a slightly different mesh state than the one tested here._")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"\n[report written to {args.out}]")


if __name__ == "__main__":
    main()
