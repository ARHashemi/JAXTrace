#!/usr/bin/env python3
"""Why does the search return -1 for a particle whose host element exists?

The index-mismatch test showed that a wrong cell index is NOT the dominant
cause: of 272 lost particles that had a true host, 175 (64.3%) had the CORRECT
index, and 16 SURVIVORS had a wrong index yet were found anyway. So this script
tests every mechanism that can make the search fail once the cell lookup is
already right.

For each lost particle it establishes the ground truth by brute force, then
attributes the failure to exactly one cause:

  OUTSIDE_MESH     no element contains the point. Not a search bug.
  NOT_REGISTERED   the true host is in no octree cell at all.
  CELL_NOT_VISITED the host's cells are outside the 3x3x3 neighbourhood at
                   every level -- a genuine index/registration-reach failure.
  PREDICATE_FAILS  the host IS in a visited cell, but the point-in-tet
                   predicate the search uses reports "outside". This is the
                   interesting case: a tolerance or degeneracy problem.
  LEVEL_NOT_WALKED the host's cells sit at a level outside [min_level,
                   max_level], so the per-level loop never reaches them.
  UNEXPLAINED      none of the above -- means the model of the search is wrong.

It also answers the Morton question directly: for the host's registered cells
it recomputes the Morton code from the stored grid index and compares it with
the stored code, so a mis-mapped or far-away registration shows up rather than
being assumed.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def skala_inside(p, v0, v1, v2, v3, tol):
    """Replicate point_in_tet_skala exactly, including its sign convention."""
    a = v1 - v0; b = v2 - v0; c = v3 - v0; r = p - v0
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    degenerate = np.abs(det) < 1e-20
    safe = np.where(degenerate, 1.0, det)
    inv = 1.0 / safe
    b1 = np.einsum("ij,ij->i", r, np.cross(b, c)) * inv
    b2 = np.einsum("ij,ij->i", a, np.cross(r, c)) * inv
    b3 = np.einsum("ij,ij->i", a, np.cross(b, r)) * inv
    b0 = 1.0 - b1 - b2 - b3
    t = -tol
    inside = (b0 >= t) & (b1 >= t) & (b2 >= t) & (b3 >= t) & (~degenerate)
    return inside, np.stack([b0, b1, b2, b3], axis=1), degenerate


def morton3(i, j, k, bits=21):
    """Interleave three 21-bit ints, as encode_morton_3d does."""
    def spread(x):
        x = int(x) & ((1 << bits) - 1)
        r = 0
        for b in range(bits):
            r |= ((x >> b) & 1) << (3 * b)
        return r
    return spread(i) | (spread(j) << 1) | (spread(k) << 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="final_state.npz")
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="POINT_IN_TET_TOLERANCE used by the run (default 1e-6)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    L = []
    def say(s=""):
        print(s, flush=True); L.append(s)

    d = np.load(args.run)
    p_all = d["positions"].astype(np.float64)
    e_all = d["element_ids"].astype(np.int64)
    lost_idx = np.flatnonzero(e_all < 0)
    say(f"# Why the search returns -1  (`{args.run.name}`)")
    say()
    say(f"{len(p_all):,} particles, {len(lost_idx):,} lost. "
        f"point-in-tet tolerance {args.tol:g}, method 'skala'.")
    say()

    # ---- mesh + octree, exactly as the run built them -------------------
    from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
    from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
    from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import (
        extract_octree_cells_parent_cube)
    import re
    post = args.case / "post"
    pv = sorted(post.glob("*_34.pvtu")) or sorted(post.glob("*.pvtu"))
    pattern = re.sub(r"_(\d+)\.pvtu$", "_{timestep}.pvtu", pv[0].name)
    ts = int(re.search(r"_(\d+)\.pvtu$", pv[0].name).group(1))
    npos, conn, vel = load_velocity_sequence_from_pvtu(
        base_path=post, file_pattern=pattern, timestep_range=(ts, ts),
        field_name="Displacement", verbose=False)
    npos, conn, _, _ = deduplicate_nodes(npos, conn, velocity_sequence=vel,
                                         verbose=False)
    conn = conn.astype(np.int32)
    cells = extract_octree_cells_parent_cube(npos, conn, tolerance=1e-6,
                                             verbose=False)
    lev = np.asarray(cells.cell_levels)
    sz = np.asarray(cells.cell_sizes)
    gidx = np.asarray(cells.cell_grid_indices).astype(np.int64)
    mort = np.asarray(cells.cell_morton_codes)
    say(f"octree {cells.n_cells:,} cells, levels "
        f"{sorted(int(x) for x in np.unique(lev))}")

    lo_lv, hi_lv = int(lev.min()), int(lev.max())
    # level pitch, exactly as upload_mesh_aligned_octree_to_gpu computes it
    pitch = {}
    for Lv in np.unique(lev):
        pitch[int(Lv)] = np.median(sz[lev == Lv], axis=0)

    # element -> registered cell rows
    offs = np.asarray(cells.cell_to_elements_offsets)
    data = np.asarray(cells.cell_to_elements_data)
    cell_of = np.repeat(np.arange(len(offs) - 1), np.diff(offs))
    order = np.argsort(data, kind="stable")
    es, cs = data[order], cell_of[order]
    st = np.searchsorted(es, np.arange(len(conn)), "left")
    en = np.searchsorted(es, np.arange(len(conn)), "right")

    v = npos[conn]
    emin, emax = v.min(axis=1), v.max(axis=1)

    rng = np.random.default_rng(42)
    sel = rng.choice(lost_idx, min(args.n, len(lost_idx)), replace=False)
    say(f"testing {len(sel):,} lost particles")
    say()

    counts = {k: 0 for k in ("OUTSIDE_MESH", "NOT_REGISTERED",
                             "CELL_NOT_VISITED", "PREDICATE_FAILS",
                             "LEVEL_NOT_WALKED", "UNEXPLAINED")}
    bary_fail, morton_bad, examples = [], 0, []

    for pi in sel:
        q = p_all[pi]
        cand = np.flatnonzero((emin[:, 0] <= q[0]) & (emax[:, 0] >= q[0]) &
                              (emin[:, 1] <= q[1]) & (emax[:, 1] >= q[1]) &
                              (emin[:, 2] <= q[2]) & (emax[:, 2] >= q[2]))
        host = -1
        if len(cand):
            ins, bar, _ = skala_inside(q, v[cand, 0], v[cand, 1], v[cand, 2],
                                       v[cand, 3], args.tol)
            if ins.any():
                host = int(cand[np.flatnonzero(ins)[0]])
            else:
                # brute force with a generous tolerance: is it inside at all?
                ins2, bar2, _ = skala_inside(q, v[cand, 0], v[cand, 1],
                                             v[cand, 2], v[cand, 3], 1e-3)
                if ins2.any():
                    host = int(cand[np.flatnonzero(ins2)[0]])
                    bary_fail.append(bar2[np.flatnonzero(ins2)[0]].min())
        if host < 0:
            counts["OUTSIDE_MESH"] += 1
            continue

        rows = cs[st[host]:en[host]]
        if len(rows) == 0:
            counts["NOT_REGISTERED"] += 1
            continue

        row_lv = {int(lev[r]) for r in rows}
        if not (row_lv & set(range(lo_lv, hi_lv + 1))):
            counts["LEVEL_NOT_WALKED"] += 1
            continue

        # Morton consistency: recompute the code from the stored grid index
        for r in rows[:4]:
            off = 1 << 19
            mm = morton3(int(gidx[r][0]) + off, int(gidx[r][1]) + off,
                         int(gidx[r][2]) + off)
            if mm != int(mort[r]):
                morton_bad += 1
                break

        # does the 3x3x3 at the host's level reach one of its cells?
        visited = False
        for Lv in sorted(row_lv):
            base = np.floor(q / pitch[Lv]).astype(np.int64)
            for r in rows:
                if int(lev[r]) != Lv:
                    continue
                if np.all(np.abs(base - gidx[r]) <= 1):
                    visited = True; break
            if visited:
                break
        if not visited:
            counts["CELL_NOT_VISITED"] += 1
            continue

        # the cell IS reachable and lists the host -> the predicate must have
        # rejected it. Re-run the exact predicate on the true host.
        ins, bar, degen = skala_inside(q, v[host:host+1, 0], v[host:host+1, 1],
                                       v[host:host+1, 2], v[host:host+1, 3],
                                       args.tol)
        if not ins[0]:
            counts["PREDICATE_FAILS"] += 1
            if len(examples) < 8:
                examples.append((int(pi), host, float(bar[0].min()),
                                 bool(degen[0])))
        else:
            counts["UNEXPLAINED"] += 1
            if len(examples) < 8:
                examples.append((int(pi), host, float(bar[0].min()), False))

    tot = max(sum(counts.values()), 1)
    say("## Attribution")
    say()
    say("| cause | n | % | meaning |")
    say("|---|---:|---:|---|")
    why = {
        "OUTSIDE_MESH":     "no element contains the point - not a search bug",
        "NOT_REGISTERED":   "true host in no octree cell",
        "CELL_NOT_VISITED": "host's cells outside the 3x3x3 at every level",
        "PREDICATE_FAILS":  "**host reachable, point-in-tet says outside**",
        "LEVEL_NOT_WALKED": "host's level outside [min_level, max_level]",
        "UNEXPLAINED":      "model of the search is incomplete",
    }
    for k, n in counts.items():
        say(f"| {k} | {n:,} | {100*n/tot:.1f} | {why[k]} |")
    say()
    say(f"Morton code inconsistent with stored grid index: {morton_bad:,} "
        f"of {len(sel):,}")
    if morton_bad == 0:
        say("-> Morton encoding is self-consistent; registrations are not "
            "being mapped to far-away cells.")
    say()
    if bary_fail:
        b = np.array(bary_fail)
        say(f"Of the points the strict predicate rejected but a 1e-3 tolerance "
            f"accepted ({len(b):,}):")
        say(f"  min barycentric: median {np.median(b):.3e}  "
            f"worst {b.min():.3e}")
        say(f"  the run used tol {args.tol:g}; these sit "
            f"{np.median(b)/args.tol:.1f}x beyond it")
    if examples:
        say()
        say("| particle | true host | min barycentric | degenerate |")
        say("|---|---:|---:|---|")
        for pi, h, mb, dg in examples:
            say(f"| {pi} | {h} | {mb:.3e} | {dg} |")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"\n[report -> {args.out}]")


if __name__ == "__main__":
    main()
