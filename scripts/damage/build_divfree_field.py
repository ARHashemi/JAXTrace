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
    ap.add_argument("--workers", type=int, default=8)
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

    coeffs = np.zeros((m_all, 12), dtype=np.float64)
    filled = np.zeros(m_all, dtype=bool)
    t2 = time.time()
    for pi, src in enumerate(srcs):
        r = vtk.vtkXMLUnstructuredGridReader()
        r.SetFileName(str(Path(args.case) / src)); r.Update()
        d = r.GetOutput()
        P = vtk_to_numpy(d.GetPoints().GetData()).astype(np.float64)
        C = vtk_to_numpy(d.GetCells().GetConnectivityArray()).reshape(-1, 4)
        # piece-local -> global via coordinates
        gl = np.full(len(P), -1, dtype=np.int64)
        nk = {tuple(np.round(nodes[i], 10)): i for i in range(len(nodes))}
        for i, pt in enumerate(P):
            gl[i] = nk.get(tuple(np.round(pt, 10)), -1)
        ok = (gl[C] >= 0).all(axis=1)
        seed = np.array([key.get(tuple(np.sort(gl[t])), -1) for t in C[ok]])
        seed = seed[seed >= 0]
        if len(seed) == 0:
            print(f"    piece {pi}: ⚠️ no elements mapped, skipped")
            continue
        patch = grow_patch(seed, fidx_g, elems_sorted, starts, ends,
                           layers=args.layers)
        co, _, _ = solve_patch(nodes, conn[patch], vel, reg=args.reg)
        # ✅ keep ONLY the seed elements -- the grown ring is discarded
        pos = {e: j for j, e in enumerate(patch)}
        idx = np.array([pos[e] for e in seed])
        coeffs[seed] = co[idx]
        filled[seed] = True
        print(f"    piece {pi:3d}: seed {len(seed):,} -> patch {len(patch):,}"
              f"  ({time.time()-t2:.0f}s elapsed)")
    print(f"  solved in {time.time()-t2:.1f}s; "
          f"filled {filled.sum():,}/{m_all:,} ({100*filled.mean():.1f} %)")

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
