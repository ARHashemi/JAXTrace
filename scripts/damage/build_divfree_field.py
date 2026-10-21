"""Build a divergence-free BDM1 field for one snapshot, by overlapping Schwarz.

    python3 build_divfree_field.py --case <post dir> --pattern 'cylA_{timestep}.pvtu' \
            --timestep 159 --out /scratch/.../divfree_159.npz

PIPELINE
  1. Load + deduplicate the global mesh (as tracking already does).
  2. Read the FEMUSS PVTU pieces as PARTITION SEEDS.
     ⚠️ The pieces share POINTS but no CELLS -- GhostLevel="1" is declared but
     there is no ghost layer and no vtkGhostType array. So they are a seed
     partition, not a usable overlap.
  3. Grow each seed by 3 face-adjacent element layers.
     ⚠️ 3 is MEASURED: preservation by layer depth from a pinned boundary is
     2.72x / 2.07x / 1.47x / 1.01x the interior asymptote at layers 0/1/2/3.
  4. Solve BDM1 on each grown patch (direct LU, ~25 s at 50 k elements).
  5. Keep only the SEED elements' coefficients; discard the overlap ring.
  6. Verify: divergence, preservation, and the SEAM jump across piece
     interfaces.

OUTPUT: (n_elements, 12) float32 -- the 9 entries of A then b, so the tracking
kernel evaluates u(x) = A x + b. ⚠️ Per element, so it is a PREBUILT field:
252 MB per snapshot on D2, streamed exactly like the velocity stack. It cannot
be computed in-step -- it is a global solve, and 48 RK4 steps share each
snapshot.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


_G = None


def _init_worker(shared):
    """Publish the shared mesh once per worker process.

    ⚠️ Called by ProcessPoolExecutor's initializer, so it runs ONCE per
    worker instead of once per task. With the fork context the arrays are
    copy-on-write, so this costs nothing beyond the fork itself.
    """
    global _G
    _G = shared


def _solve_one(sd):
    """Worker: grow one seed, solve, return only the seed's coefficients.

    ⚠️ Reads the shared mesh from the module global _G, which fork gives each
    worker copy-on-write. Passing the arrays through the pickle boundary
    instead would copy ~1 GB per patch.
    """
    from jaxtrace.damage.divfree import grow_patch, solve_patch
    import numpy as _np
    g = _G
    patch = grow_patch(sd, g["fidx_g"], g["elems_sorted"], g["starts"],
                       g["ends"], layers=g["layers"])
    co, _, _ = solve_patch(g["nodes"], g["conn"][patch], g["vel"],
                           reg=g["reg"])
    pos = {e: j for j, e in enumerate(patch)}
    return sd, co[_np.array([pos[e] for e in sd])]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--timestep", type=int, required=True)
    ap.add_argument("--field", default="Displacement")
    ap.add_argument("--layers", type=int, default=3,
                    help="overlap layers; 3 is measured, see the module docstring")
    ap.add_argument("--reg", type=float, default=1e-8)
    ap.add_argument("--max-pieces", type=int, default=0,
                    help="0 = all; otherwise process only the first N (testing)")
    ap.add_argument("--workers", type=int, default=8,
                    help="threads for the PVTU read")
    ap.add_argument("--jobs", type=int, default=16,
                    help="⚠️ PROCESSES for the patch solves. splu is 99.5 "
                         "percent of the cost and single-threaded, so this is "
                         "the main lever; 1 disables parallelism.")
    ap.add_argument("--seed-size", type=int, default=10000,
                    help="split seeds to about this many elements. 10,000 "
                         "is the MEASURED knee over 4 positions per size: "
                         "preservation p50 0.342/0.237/0.106/0.092 at "
                         "2k/5k/10k/20k for 59/80/144/352 s per cylA "
                         "snapshot on 16 threads. ⚠️ 10k->20k buys only 13 "
                         "percent preservation for 2.4x the cost, and 10k has the "
                         "tightest spread (std 0.038 vs 0.324 at 2k). 0 "
                         "keeps the FEMUSS pieces whole (~720 s).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from jaxtrace.gpu.mesh_loader_timedep import load_field_sequences_from_pvtu
    from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
    from jaxtrace.damage.divfree import (
        OPP, build_face_topology, element_adjacency, grow_patch, solve_patch)

    t0 = time.time()
    nodes, conn, seq = load_field_sequences_from_pvtu(
        Path(args.case), args.pattern, (args.timestep, args.timestep),
        [args.field], verbose=False, n_workers=args.workers)
    o = deduplicate_nodes(nodes, conn, seq[args.field])
    nodes, conn, vel = o[0], o[1].astype(np.int64), o[3][0].astype(np.float64)
    m_all = len(conn)
    print(f"  global mesh: {len(nodes):,} nodes, {m_all:,} elements "
          f"({time.time()-t0:.1f}s)")

    # ── seeds from the FEMUSS pieces ────────────────────────────────────
    # ⚠️ Map each piece's CELLS onto global element ids by their sorted node
    # coordinates. The pieces have their own local numbering, and the global
    # mesh has been deduplicated, so index-based matching would be wrong.
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    pvtu = Path(args.case) / args.pattern.format(timestep=args.timestep)
    txt = pvtu.read_text()
    srcs = [ln.split('Source="')[1].split('"')[0]
            for ln in txt.splitlines() if "<Piece" in ln]
    if args.max_pieces:
        srcs = srcs[:args.max_pieces]
    print(f"  {len(srcs)} FEMUSS pieces used as seeds")

    t1 = time.time()
    key = {}
    for e in range(m_all):
        key[tuple(np.sort(conn[e]))] = e
    print(f"  global element key map in {time.time()-t1:.1f}s")

    uf, fidx_g, cnt_g = build_face_topology(conn)
    elems_sorted, starts, ends = element_adjacency(fidx_g, cnt_g)
    print(f"  global topology: {len(uf):,} faces")

    # ⚠️ PARALLELISM. Profiling showed 99.5 % of the cost is a single
    # scipy splu call, and SuperLU is SINGLE-THREADED -- which is exactly why
    # htop showed one busy core. The patches are completely independent, so
    # cross-patch parallelism is free and is the single biggest fix.
    #
    # ✅ ProcessPool, not ThreadPool: splu releases the GIL only partially and
    # the work is pure CPU, so threads would contend. Each worker re-derives
    # nothing -- the mesh arrays are inherited by fork.
    from concurrent.futures import ProcessPoolExecutor
    seeds_record = []
    coeffs = np.zeros((m_all, 12), dtype=np.float64)
    filled = np.zeros(m_all, dtype=bool)
    # ⚠️ Build the node key map ONCE. It was inside the piece loop, i.e.
    # rebuilt 64 times over 571k nodes -- pure waste.
    nk = {tuple(np.round(nodes[i].astype(np.float64), 9)): i
          for i in range(len(nodes))}
    # ⚠️ Collect ALL seeds first, then solve them in parallel. The original
    # loop interleaved VTK reading with solving, so nothing could overlap.
    t_seed = time.time()
    all_seeds = []
    for pi, src in enumerate(srcs):
        r = vtk.vtkXMLUnstructuredGridReader()
        r.SetFileName(str(Path(args.case) / src)); r.Update()
        d = r.GetOutput()
        P = vtk_to_numpy(d.GetPoints().GetData()).astype(np.float64)
        C = vtk_to_numpy(d.GetCells().GetConnectivityArray()).reshape(-1, 4)
        gl = np.array([nk.get(tuple(np.round(pt, 9)), -1) for pt in P])
        ok = (gl[C] >= 0).all(axis=1)
        sd = np.array([key.get(tuple(np.sort(gl[t])), -1) for t in C[ok]])
        sd = sd[sd >= 0]
        if len(sd):
            all_seeds.append(sd)
    print(f"  mapped {len(all_seeds)} seeds in {time.time()-t_seed:.1f}s")

    # ⚠️ SPLIT oversized seeds. splu scales as n^3.10, so a 55k FEMUSS piece
    # costs 131.8 s while four 14k chunks of it cost ~4x1.3 s. The piece
    # boundaries are a partition HINT, not a constraint -- any partition
    # works, because each patch is grown and trimmed independently.
    if args.seed_size:
        split = []
        for sd in all_seeds:
            k = max(1, int(np.ceil(len(sd) / args.seed_size)))
            split.extend(np.array_split(sd, k))
        all_seeds = split
        print(f"  split to {len(all_seeds)} seeds of ~{args.seed_size:,} elements")

    t2 = time.time()
    if args.jobs > 1:
        # ⚠️ Python 3.14 defaults to SPAWN, not fork, so module globals are
        # NOT inherited -- the worker raised NameError: _G. And we must not
        # pickle the mesh per task: ~1 GB per patch.
        #
        # ✅ Use an explicit fork context plus an initializer. fork gives the
        # arrays copy-on-write; the initializer publishes them once per
        # worker rather than once per task.
        import multiprocessing as _mp
        _ctx = _mp.get_context("fork")
        _shared = dict(
            fidx_g=fidx_g, elems_sorted=elems_sorted, starts=starts,
            ends=ends, nodes=nodes, conn=conn, vel=vel,
            layers=args.layers, reg=args.reg)
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=_ctx,
                                 initializer=_init_worker,
                                 initargs=(_shared,)) as ex:
            for n_done, (sd, vals) in enumerate(
                    ex.map(_solve_one, all_seeds), 1):
                coeffs[sd] = vals; filled[sd] = True; seeds_record.append(sd)
                if n_done % 20 == 0 or n_done == len(all_seeds):
                    print(f"    {n_done}/{len(all_seeds)} patches "
                          f"({time.time()-t2:.0f}s)")
    else:
        for n_done, sd in enumerate(all_seeds, 1):
            patch = grow_patch(sd, fidx_g, elems_sorted, starts, ends,
                               layers=args.layers)
            co, _, _ = solve_patch(nodes, conn[patch], vel, reg=args.reg)
            pos = {e: j for j, e in enumerate(patch)}
            coeffs[sd] = co[np.array([pos[e] for e in sd])]
            filled[sd] = True; seeds_record.append(sd)
            if n_done % 20 == 0 or n_done == len(all_seeds):
                print(f"    {n_done}/{len(all_seeds)} patches "
                      f"({time.time()-t2:.0f}s)")
    print(f"  solved in {time.time()-t2:.1f}s; "
          f"filled {filled.sum():,}/{m_all:,} ({100*filled.mean():.1f} %)")

    # ── VERDICT: divergence, preservation, and the SEAM jump ────────────
    # ⚠️ The seam check is the one this decomposition exists to pass. If the
    # discard rule is right, elements on either side of a piece interface are
    # both solved as INTERIOR elements of their own patch, so their quality
    # should match the bulk. A visible seam means the 3-layer ring is too thin.
    if filled.any():
        sel = np.where(filled)[0]
        Amat = coeffs[sel, :9].reshape(-1, 3, 3)
        bvec = coeffs[sel, 9:]
        dv = np.abs(np.trace(Amat, axis1=1, axis2=2))
        nA = np.sqrt(np.sum(Amat*Amat, axis=(1, 2)))
        print()
        print("  ── DIVERGENCE (|div u| / |grad u|) ──")
        r1 = dv / np.maximum(nA, 1e-30)
        print(f"     median {np.median(r1):.4e}  p99 {np.percentile(r1,99):.4e}"
              f"  max {r1.max():.4e}")
        print(f"     {'✅ PASS' if np.median(r1) < 1e-10 else '⚠️  above 1e-10'}")

        p_ = nodes[conn[sel]]
        xc_ = p_.mean(axis=1)
        u_new = np.einsum('mij,mj->mi', Amat, xc_) + bvec
        u_p1 = vel[conn[sel]].mean(axis=1)
        rr = (np.linalg.norm(u_new - u_p1, axis=1) /
              np.maximum(np.linalg.norm(u_p1, axis=1), 1e-30))
        print()
        print("  ── PRESERVATION (||u_new - u_P1|| / ||u_P1||) ──")
        for qq in (50, 90, 99):
            print(f"     p{qq:<3} {np.percentile(rr, qq):.4e}")

        # ⚠️ SEAM: elements adjacent across a piece boundary. Compare their
        # preservation to the bulk -- a jump means the overlap is too thin.
        own = np.full(m_all, -1, dtype=np.int64)
        for pi2, sd in enumerate(seeds_record):
            own[sd] = pi2
        import collections
        face_owner = collections.defaultdict(list)
        for e in sel:
            for f in range(4):
                face_owner[fidx_g[e, f]].append(e)
        seam_el = set()
        for gf, els in face_owner.items():
            if len(els) == 2 and own[els[0]] != own[els[1]]                     and own[els[0]] >= 0 and own[els[1]] >= 0:
                seam_el.update(els)
        if seam_el:
            sm = np.isin(sel, list(seam_el))
            print()
            print("  ── SEAM CHECK (elements across a piece interface) ──")
            print(f"     seam elements {int(sm.sum()):,} of {len(sel):,}")
            print(f"     seam     p50 {np.percentile(rr[sm],50):.4e}")
            print(f"     non-seam p50 {np.percentile(rr[~sm],50):.4e}")
            ratio = (np.percentile(rr[sm], 50) /
                     max(np.percentile(rr[~sm], 50), 1e-300))
            print(f"     ratio {ratio:.2f}x  "
                  f"{'✅ no seam penalty' if ratio < 1.3 else '⚠️  SEAM VISIBLE — widen the overlap'}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out,
                            coeffs=coeffs.astype(np.float32),
                            filled=filled, timestep=args.timestep)
        print(f"  wrote {args.out} "
              f"({args.out.stat().st_size/1024**2:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
