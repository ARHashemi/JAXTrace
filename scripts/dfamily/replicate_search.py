#!/usr/bin/env python3
"""Replicate the production search EXACTLY and say why it returns -1.

My earlier diagnostic approximated the search with a "is a host cell within
+-1 of floor(q/pitch)?" test. That left 57.1% of losses UNEXPLAINED, because
the real search does something stricter: for each of the 27 neighbourhood
offsets at each level it computes a Morton code and BINARY SEARCHES the sorted
cell array for an exact (morton, level) match. A cell is only testable if that
exact key exists -- and the octree stores only non-empty cells.

This script calls the real functions instead of modelling them:
    encode_morton_3d_jax        from mesh_aligned_octree_gpu
    find_cell_by_morton_and_level  from mesh_aligned_octree_gpu
    upload_mesh_aligned_octree_to_gpu
    search_mesh_aligned_octree_multi_local_where   <- what tracking calls
so the reproduction is the production code path, not a description of it.

For each lost particle it:
  1. runs the real search and confirms it returns -1,
  2. brute-forces the true host,
  3. walks the 27 keys at every level the host is registered at and reports
     the smallest |delta index| between the search's base cell and a host cell,
     i.e. HOW FAR out of reach the host is. That number chooses the fix:
     delta <= 2 -> a 5x5x5 neighbourhood suffices; large delta -> it does not.
"""
from __future__ import annotations
import argparse, os, re, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="final_state.npz")
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    L = []
    def say(s=""):
        print(s, flush=True); L.append(s)

    import jax, jax.numpy as jnp
    from jaxtrace import config
    from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
    from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
    from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import (
        extract_octree_cells_parent_cube)
    from jaxtrace.gpu.search.mesh_aligned_octree_gpu import (
        upload_mesh_aligned_octree_to_gpu, encode_morton_3d_jax,
        find_cell_by_morton_and_level)
    from jaxtrace.gpu.search.mesh_aligned_point_location import (
        search_mesh_aligned_octree_multi_local_where)

    d = np.load(args.run)
    pos_all = d["positions"].astype(np.float64)
    eid_all = d["element_ids"].astype(np.int64)
    lost_idx = np.flatnonzero(eid_all < 0)
    say(f"# Exact search replication — `{args.run.name}`")
    say()
    say(f"{len(pos_all):,} particles, {len(lost_idx):,} lost.")
    say()

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
    oct_gpu = upload_mesh_aligned_octree_to_gpu(conn, npos, cells, verbose=False)
    say(f"octree {cells.n_cells:,} cells, levels "
        f"[{int(oct_gpu.min_level)}..{int(oct_gpu.max_level)}]")
    say(f"point-in-tet method '{config.POINT_IN_TET_METHOD}', "
        f"tol {config.POINT_IN_TET_TOLERANCE:g}")
    say()

    lev = np.asarray(cells.cell_levels)
    gidx = np.asarray(cells.cell_grid_indices).astype(np.int64)
    lcs = np.asarray(oct_gpu.level_cell_sizes)
    offs = np.asarray(cells.cell_to_elements_offsets)
    data = np.asarray(cells.cell_to_elements_data)
    cell_of = np.repeat(np.arange(len(offs) - 1), np.diff(offs))
    order = np.argsort(data, kind="stable")
    es, cs = data[order], cell_of[order]
    st = np.searchsorted(es, np.arange(len(conn)), "left")
    en = np.searchsorted(es, np.arange(len(conn)), "right")

    v = npos[conn]
    emin, emax = v.min(axis=1), v.max(axis=1)
    TOL = float(config.POINT_IN_TET_TOLERANCE)

    def inside(p, idx):
        a = v[idx, 1] - v[idx, 0]; b = v[idx, 2] - v[idx, 0]
        c = v[idx, 3] - v[idx, 0]; r = p - v[idx, 0]
        det = np.einsum("ij,ij->i", a, np.cross(b, c))
        bad = np.abs(det) < 1e-20
        inv = 1.0 / np.where(bad, 1.0, det)
        b1 = np.einsum("ij,ij->i", r, np.cross(b, c)) * inv
        b2 = np.einsum("ij,ij->i", a, np.cross(r, c)) * inv
        b3 = np.einsum("ij,ij->i", a, np.cross(b, r)) * inv
        b0 = 1.0 - b1 - b2 - b3
        t = -TOL
        return (b0 >= t) & (b1 >= t) & (b2 >= t) & (b3 >= t) & (~bad)

    # the real search, batched
    rng = np.random.default_rng(42)
    sel = rng.choice(lost_idx, min(args.n, len(lost_idx)), replace=False)
    qs = jnp.asarray(pos_all[sel], dtype=config.FLOAT_DTYPE_JNP)
    search = jax.jit(jax.vmap(
        lambda q: search_mesh_aligned_octree_multi_local_where(q, oct_gpu)[0]))
    got = np.asarray(search(qs))
    say(f"real search re-run on {len(sel):,} lost particles: "
        f"{int((got < 0).sum()):,} still return -1 "
        f"({100*(got < 0).mean():.1f}%) -- confirms reproducibility")
    say()

    OFF = int(oct_gpu.morton_offset); MAXC = int(oct_gpu.morton_max_coord)
    lo_lv, hi_lv = int(oct_gpu.min_level), int(oct_gpu.max_level)

    cause = {k: 0 for k in ("OUTSIDE_MESH", "NOT_REGISTERED",
                            "OUT_OF_REACH", "PREDICATE", "FOUND_ANYWAY")}
    deltas, ex = [], []

    for n, pi in enumerate(sel):
        if got[n] >= 0:
            cause["FOUND_ANYWAY"] += 1
            continue
        q = pos_all[pi]
        cand = np.flatnonzero((emin[:, 0] <= q[0]) & (emax[:, 0] >= q[0]) &
                              (emin[:, 1] <= q[1]) & (emax[:, 1] >= q[1]) &
                              (emin[:, 2] <= q[2]) & (emax[:, 2] >= q[2]))
        host = -1
        if len(cand):
            ok = inside(q, cand)
            if ok.any():
                host = int(cand[np.flatnonzero(ok)[0]])
        if host < 0:
            cause["OUTSIDE_MESH"] += 1
            continue
        rows = cs[st[host]:en[host]]
        if len(rows) == 0:
            cause["NOT_REGISTERED"] += 1
            continue

        # EXACTLY what the search does: per level, 27 Morton keys, binary search
        reached = False
        best = None
        for lvl_idx in range(hi_lv - lo_lv + 1):
            level = hi_lv - lvl_idx
            cs_lv = lcs[level]
            if not np.all(cs_lv > 0):
                continue
            base = np.floor(q / cs_lv).astype(np.int64)
            hrows = [r for r in rows if int(lev[r]) == level]
            if hrows:
                dmin = min(int(np.abs(base - gidx[r]).max()) for r in hrows)
                best = dmin if best is None else min(best, dmin)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for dk in (-1, 0, 1):
                        ii = int(np.clip(base[0] + di + OFF, 0, MAXC - 1))
                        jj = int(np.clip(base[1] + dj + OFF, 0, MAXC - 1))
                        kk = int(np.clip(base[2] + dk + OFF, 0, MAXC - 1))
                        mc = encode_morton_3d_jax(jnp.int32(ii), jnp.int32(jj),
                                                  jnp.int32(kk))
                        ci = int(find_cell_by_morton_and_level(
                            mc, jnp.uint8(level), oct_gpu.cell_morton_codes,
                            oct_gpu.cell_levels))
                        if ci >= 0 and ci in set(int(x) for x in rows):
                            reached = True; break
                    if reached: break
                if reached: break
            if reached: break

        if reached:
            cause["PREDICATE"] += 1
            if len(ex) < 6:
                ex.append((int(pi), host, best))
        else:
            cause["OUT_OF_REACH"] += 1
            if best is not None:
                deltas.append(best)
            if len(ex) < 6:
                ex.append((int(pi), host, best))

    tot = max(sum(cause.values()), 1)
    say("## Attribution (exact replication)")
    say()
    say("| cause | n | % | meaning |")
    say("|---|---:|---:|---|")
    why = {
        "OUTSIDE_MESH": "no element contains the point - not a bug",
        "NOT_REGISTERED": "true host in no octree cell",
        "OUT_OF_REACH": "**the 27 keys never name a cell holding the host**",
        "PREDICATE": "cell reached, yet the search still missed it",
        "FOUND_ANYWAY": "search found a host now (mesh/timestep differs)",
    }
    for k, n in cause.items():
        say(f"| {k} | {n:,} | {100*n/tot:.1f} | {why[k]} |")

    if deltas:
        a = np.array(deltas)
        say()
        say(f"HOW FAR out of reach ({len(a):,} cases), max-axis |delta index| "
            f"between the search's base cell and the nearest host cell:")
        for q_ in (50, 75, 90, 95, 99, 100):
            say(f"  p{q_:<3d} {np.percentile(a, q_):6.1f}")
        say()
        for lim, nm in ((1, "3x3x3 (current)"), (2, "5x5x5"), (3, "7x7x7")):
            say(f"  delta <= {lim} ({nm:16}): {int((a <= lim).sum()):,} "
                f"= {100*(a <= lim).mean():.1f}% would be recovered")
    if ex:
        say()
        say("| particle | true host | min max-axis delta |")
        say("|---|---:|---:|")
        for pi, h, b in ex:
            say(f"| {pi} | {h} | {b} |")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"\n[report -> {args.out}]")


if __name__ == "__main__":
    main()
