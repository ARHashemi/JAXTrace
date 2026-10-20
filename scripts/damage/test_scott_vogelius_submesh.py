"""Scott-Vogelius feasibility test on a SUB-MESH — before building the real thing.

    python3 test_scott_vogelius_submesh.py --case <dir> --pattern 'cylA_{timestep}.pvtu' \
            --timestep 159 --n-elements 100000

WHAT THIS DECIDES, AND WHY A SUB-MESH IS ENOUGH
-----------------------------------------------
docs/damage/NEXT_STEPS_detachment_criterion.md B.6p ranks Scott-Vogelius first
of the divergence-free options, because SV is POINTWISE divergence-free on a
barycentrically refined (Alfeld) mesh and HCT already builds that split.

⚠️ But SV is weeks of work: a global Stokes-like projection, an iterative solver
with a preconditioner, and boundary-condition extraction that does not exist in
this codebase. This script spends ONE DAY instead, to answer three questions:

  Q1 ✅ Does the projection actually reach ~1e-6 relative pointwise divergence
        on OUR data? If it cannot on 1e5 elements it will not on 10.8M.
  Q2 ✅ What does the solve COST per element? That extrapolates to the full mesh
        and to 166 snapshots, which decides Q3.
  Q3 ✅ PREBUILT-AND-STORED, or RECONSTRUCT-INSIDE-EACH-STEP? Measured here as
        bytes per element and seconds per snapshot, not guessed.

⚠️ WHAT THIS IS NOT. This is a LOCAL, per-element projection -- the cheapest
thing that can be called Scott-Vogelius-like -- not the global SV solve. A
global solve couples elements through the continuity of the velocity space and
can only do BETTER on divergence while costing MORE. So:
  * if the LOCAL projection already reaches the target, the global solve is
    unnecessary and we have saved weeks;
  * if the LOCAL projection cannot, that is NOT proof the global one fails --
    but it tells us the easy version is out, which is itself decisive for
    planning.
This asymmetry is the whole reason the cheap test is worth running first.
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
    ap.add_argument("--case", required=True,
                    help="directory holding the PVTU files")
    ap.add_argument("--pattern", required=True,
                    help="e.g. 'cylA_{timestep}.pvtu'")
    ap.add_argument("--timestep", type=int, required=True)
    ap.add_argument("--n-elements", type=int, default=100_000,
                    help="sub-mesh size; 0 = whole mesh")
    ap.add_argument("--field", default="Displacement")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from jaxtrace.gpu.mesh_loader_timedep import load_field_sequences_from_pvtu
    from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
    from jaxtrace.damage.fields import (
        tet_shape_gradients, velocity_gradient_per_element,
    )

    print("=" * 72)
    print(" Scott-Vogelius feasibility on a sub-mesh")
    print("=" * 72)
    t0 = time.time()
    nodes, conn, seq = load_field_sequences_from_pvtu(
        Path(args.case), args.pattern,
        (args.timestep, args.timestep), [args.field],
        verbose=False, n_workers=args.workers)
    out = deduplicate_nodes(nodes, conn, seq[args.field])
    nodes, conn, vel = out[0], out[1].astype(np.int32), out[3][0]
    print(f"  loaded + dedup in {time.time()-t0:.1f}s: "
          f"{nodes.shape[0]:,} nodes, {conn.shape[0]:,} elements")

    # ── sub-mesh ────────────────────────────────────────────────────────
    n_el_all = conn.shape[0]
    if args.n_elements and args.n_elements < n_el_all:
        rng = np.random.default_rng(args.seed)
        # ⚠️ A CONTIGUOUS block, not a random scatter. A random subset of
        # elements is topologically disconnected, and the whole point of SV is
        # that it couples neighbours through the velocity space. Taking a
        # contiguous slice keeps real adjacency.
        start = rng.integers(0, n_el_all - args.n_elements)
        sel = np.arange(start, start + args.n_elements)
        print(f"  sub-mesh: contiguous elements [{start:,}, "
              f"{start + args.n_elements:,})")
    else:
        sel = np.arange(n_el_all)
        print(f"  sub-mesh: WHOLE mesh ({n_el_all:,} elements)")
    c = conn[sel]

    gN_all, vol_all = tet_shape_gradients(nodes, conn)
    gN, vol = gN_all[sel], vol_all[sel]

    def div_of(gradN, velocity, cells):
        L = velocity_gradient_per_element(velocity, cells, gradN)
        D = 0.5 * (L + np.transpose(L, (0, 2, 1)))
        trD = np.trace(D, axis1=1, axis2=2)
        nD = np.sqrt(np.sum(D * D, axis=(1, 2)))
        return trD, np.abs(trD) / np.maximum(nD, 1e-30)

    trD0, rel0 = div_of(gN, vel, c)
    print()
    print("  ── BEFORE (raw P1) ──")
    print(f"     |div u|/|D| median {np.median(rel0):.4e}  "
          f"p90 {np.percentile(rel0,90):.4e}  max {rel0.max():.4e}")
    print(f"     raw |div u| median {np.median(np.abs(trD0)):.4e} 1/s")
    print(f"     frac rel>1e-2 {100*(rel0>1e-2).mean():.2f} %   "
          f"frac rel>1e-6 {100*(rel0>1e-6).mean():.2f} %")

    # ── The local SV-like projection ────────────────────────────────────
    # ⚠️ THE CONSTRUCTION, stated plainly so the result can be judged.
    # On the Alfeld split each parent tet becomes 4 sub-tets sharing the
    # parent centroid vc. We give the velocity ONE extra degree of freedom per
    # parent element -- the value at vc -- and choose it to minimise the
    # divergence over the 4 sub-tets.
    #
    # On a sub-tet with vertices (vc, and 3 parent vertices), P1 velocity gives
    # a CONSTANT gradient, hence a constant divergence:
    #     div_s = sum_{a in subtet} grad(N_a^s) . u_a
    # Only the vc term depends on the unknown u_vc, linearly:
    #     div_s = g_s . u_vc + r_s
    # So minimising sum_s w_s |div_s|^2 over u_vc is a 3x3 linear least-squares
    # problem per element -- solvable in closed form, fully vectorised, no
    # global solve and no boundary conditions.
    #
    # ✅ This is the CHEAPEST member of the SV family: it buys divergence
    # reduction with the centroid DOF that the Alfeld split already provides.
    print()
    print("  ── building the local Alfeld/SV projection ──")
    t1 = time.time()
    ALFELD = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int32)
    xn = nodes[c]                                     # (m, 4, 3)
    un = vel[c].astype(np.float64)                    # (m, 4, 3)
    vc = xn.mean(axis=1)                              # (m, 3)
    m = c.shape[0]

    A = np.zeros((m, 3, 3))
    b = np.zeros((m, 3))
    w_tot = np.zeros(m)
    for s in range(4):
        pv = ALFELD[s]
        # sub-tet vertices: [vc, x_pv0, x_pv1, x_pv2]
        p0 = vc
        p1, p2, p3 = xn[:, pv[0]], xn[:, pv[1]], xn[:, pv[2]]
        e1, e2, e3 = p1 - p0, p2 - p0, p3 - p0
        # 6V = det[e1 e2 e3]
        detJ = np.einsum('mi,mi->m', e1, np.cross(e2, e3))
        volS = detJ / 6.0
        # grad of the 3 non-centroid shape functions: rows of inv(J)
        Jm = np.stack([e1, e2, e3], axis=2)           # (m,3,3) columns
        # ⚠️ Guard degenerate sub-tets rather than letting inv() raise: a
        # near-zero volume means a flat sub-tet, which contributes no
        # meaningful constraint and must be weighted out, not crashed on.
        ok = np.abs(detJ) > 1e-30
        Jinv = np.zeros_like(Jm)
        Jinv[ok] = np.linalg.inv(Jm[ok])
        gpv = np.transpose(Jinv, (0, 2, 1))           # (m,3,3): g for p1,p2,p3
        gvc = -gpv.sum(axis=1)                        # (m,3): grad N_vc
        # residual from the three parent-vertex contributions
        r = np.einsum('mai,mai->m', gpv, un[:, pv, :])
        w = np.abs(volS) * ok
        A += w[:, None, None] * np.einsum('mi,mj->mij', gvc, gvc)
        b -= (w * r)[:, None] * gvc
        w_tot += w

    # ⚠️ Regularise toward the P1 centroid value (the mean of the 4 vertices).
    # Without it an element whose gvc vectors are nearly coplanar gives a
    # singular 3x3 and the "solution" is arbitrary -- it would reduce
    # divergence while moving the velocity anywhere. lam ties u_vc to the data.
    u_p1_vc = un.mean(axis=1)
    lam = 1.0e-6 * np.maximum(w_tot, 1e-30)
    A += lam[:, None, None] * np.eye(3)[None]
    b += lam[:, None] * u_p1_vc
    u_vc = np.linalg.solve(A, b)                      # (m, 3)
    t_solve = time.time() - t1
    print(f"     closed-form 3x3 per element: {t_solve:.2f}s for {m:,} elements")
    print(f"     = {1e6*t_solve/m:.3f} us/element")

    # ── divergence AFTER, on the 4 sub-tets ─────────────────────────────
    div_after = []
    vols_after = []
    for s in range(4):
        pv = ALFELD[s]
        p0 = vc
        p1, p2, p3 = xn[:, pv[0]], xn[:, pv[1]], xn[:, pv[2]]
        e1, e2, e3 = p1 - p0, p2 - p0, p3 - p0
        detJ = np.einsum('mi,mi->m', e1, np.cross(e2, e3))
        ok = np.abs(detJ) > 1e-30
        Jm = np.stack([e1, e2, e3], axis=2)
        Jinv = np.zeros_like(Jm)
        Jinv[ok] = np.linalg.inv(Jm[ok])
        gpv = np.transpose(Jinv, (0, 2, 1))
        gvc = -gpv.sum(axis=1)
        d = (np.einsum('mai,mai->m', gpv, un[:, pv, :])
             + np.einsum('mi,mi->m', gvc, u_vc))
        # |D| on the sub-tet, for the relative measure
        L = (np.einsum('mai,maj->mij', un[:, pv, :], gpv)
             + np.einsum('mi,mj->mij', u_vc, gvc))
        D = 0.5 * (L + np.transpose(L, (0, 2, 1)))
        nD = np.sqrt(np.sum(D * D, axis=(1, 2)))
        div_after.append(np.abs(d) / np.maximum(nD, 1e-30))
        vols_after.append(np.abs(detJ) / 6.0)
    rel1 = np.concatenate(div_after)
    print()
    print("  ── AFTER (local Alfeld/SV projection, per sub-tet) ──")
    print(f"     |div u|/|D| median {np.median(rel1):.4e}  "
          f"p90 {np.percentile(rel1,90):.4e}  max {rel1.max():.4e}")
    print(f"     frac rel>1e-2 {100*(rel1>1e-2).mean():.2f} %   "
          f"frac rel>1e-6 {100*(rel1>1e-6).mean():.2f} %")
    red = np.median(rel0) / max(np.median(rel1), 1e-30)
    print()
    print(f"     ✅ median reduction {red:.2f}x" if red > 1.5 else
          f"     ⚠️  median reduction only {red:.2f}x")

    # how far did the velocity move?
    dv = np.linalg.norm(u_vc - u_p1_vc, axis=1)
    sv = np.linalg.norm(u_p1_vc, axis=1)
    print(f"     velocity change at centroid: ||du||/||u|| median "
          f"{np.median(dv/np.maximum(sv,1e-30)):.3e}  "
          f"p99 {np.percentile(dv/np.maximum(sv,1e-30),99):.3e}")

    # ── Q1 verdict ──────────────────────────────────────────────────────
    print()
    print("  " + "=" * 68)
    med1 = float(np.median(rel1))
    if med1 < 1e-6:
        print("  ✅ Q1 PASSED: pointwise divergence reaches ~1e-6 relative.")
    elif med1 < float(np.median(rel0)) / 10.0:
        print(f"  ⚠️  Q1 PARTIAL: {red:.1f}x better but median {med1:.2e}, "
              f"not 1e-6. The LOCAL projection is insufficient; only a GLOBAL "
              f"SV solve could close the rest.")
    else:
        print(f"  ❌ Q1 FAILED: median {med1:.2e}, reduction {red:.2f}x. The "
              f"centroid DOF alone cannot fix this field.")

    # ── Q2/Q3: cost and the prebuild-vs-in-step decision ────────────────
    print()
    print("  ── Q2/Q3: cost, and PREBUILT vs IN-STEP ──")
    us_per_el = 1e6 * t_solve / m
    for name, n_el, n_snap in (("cylA", 3_050_196, 1),
                               ("D2", 10_807_677, 166)):
        t_one = us_per_el * n_el / 1e6
        # storage: the extra centroid velocity, 3 float32 per element
        mb_one = n_el * 3 * 4 / 1024**2
        print(f"     {name:5s} {n_el:>12,} el x {n_snap:>3} snap: "
              f"solve {t_one:7.1f}s/snap -> {t_one*n_snap/60:7.1f} min total; "
              f"store {mb_one:7.1f} MB/snap -> {mb_one*n_snap/1024:6.2f} GB all")
    print()
    print("     ✅ PREBUILT is the right answer when: the per-snapshot solve is")
    print("        cheap relative to the mesh load, AND the stored field fits.")
    print("     ⚠️  The stored object is only 3 floats/element -- the centroid")
    print("        velocity. It is NOT the HCT Bernstein array (240 B/el).")
    print("        So storage is ~24x cheaper than HCT for the same mesh.")
    print("  " + "=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
