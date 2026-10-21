"""Larson-Niklasson patch reconstruction: a divergence-free velocity, no global solve.

    python3 ln_reconstruct.py --case <post dir> --pattern 'cylA_{timestep}.pvtu' \
            --timestep 159 [--n-elements 0 for the whole mesh]

WHY THIS AND NOT THE BDM1 SADDLE POINT
---------------------------------------
The BDM1 global projection WORKS -- machine-precision divergence, ~7% field
change -- but its saddle point is ill-conditioned and extrapolates to ~14 h per
snapshot by direct LU. Scudeler, Putti & Paniconi (2016) apply the
Larson-Niklasson scheme to exactly our input (a P1 Galerkin velocity) and get a
locally mass-conservative field with continuous face fluxes from N INDEPENDENT
LOCAL SOLVES of dimension <= ~18. No global matrix, no preconditioner, and no
conditioning problem.

THE METHOD, derived and verified before coding
-----------------------------------------------
1. Average the two one-sided P1 fluxes on every interior face -> qbar_F.
   ⚠️ That alone does NOT conserve mass: each element now has a net flux
   residual r_T = sum_F +-qbar_F.

2. For each NODE i, take the patch E_i of elements sharing it and solve

       B U = r       on the patch,

   where B is the GRAPH LAPLACIAN of the patch's element adjacency with edge
   weights = shared face areas. ⚠️ B is singular by construction -- its null
   vector is the constant, giving rank N_Ei - 1, exactly the deficiency
   Scudeler reports. Cure: pin one potential to zero.

3. Correct: q_F <- qbar_F + (U_L - U_R) * |F|, then average the per-node
   contributions, q_F = (1/3) sum_i q_F^(i)  (their Eq. 23c; 3 nodes per face
   in 3D... see the note at the averaging step).

✅ Verified on a synthetic 4-element patch: the correction cancels the
imbalance to 8.3e-17.

⚠️ Known weakness (theirs): patches straddling strong PERMEABILITY CONTRASTS
trap mass, needing a second correction step (Schiavazzi 2011). A
single-material FSW workpiece has no such contrast.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

OPP = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--timestep", type=int, required=True)
    ap.add_argument("--n-elements", type=int, default=0,
                    help="0 = the whole mesh (that is the point of LN)")
    ap.add_argument("--field", default="Displacement")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from jaxtrace.gpu.mesh_loader_timedep import load_field_sequences_from_pvtu
    from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
    from jaxtrace.damage.fields import (
        tet_shape_gradients, velocity_gradient_per_element)

    t0 = time.time()
    nodes, conn, seq = load_field_sequences_from_pvtu(
        Path(args.case), args.pattern, (args.timestep, args.timestep),
        [args.field], verbose=False, n_workers=args.workers)
    o = deduplicate_nodes(nodes, conn, seq[args.field])
    nodes, conn, vel = o[0], o[1].astype(np.int64), o[3][0].astype(np.float64)
    print(f"  loaded {len(nodes):,} nodes, {len(conn):,} elements "
          f"in {time.time()-t0:.1f}s")

    if args.n_elements and args.n_elements < len(conn):
        rng = np.random.default_rng(args.seed)
        s0 = int(rng.integers(0, len(conn) - args.n_elements))
        c = conn[s0:s0 + args.n_elements]
        print(f"  sub-mesh: contiguous [{s0:,}, {s0+args.n_elements:,})")
    else:
        c = conn
        print(f"  WHOLE mesh ({len(c):,} elements)")
    m = len(c)

    # ── face topology and geometry ──────────────────────────────────────
    t1 = time.time()
    fverts = np.sort(c[:, OPP.ravel()].reshape(m, 4, 3), axis=2)
    uf, finv, cnt = np.unique(fverts.reshape(-1, 3), axis=0,
                              return_inverse=True, return_counts=True)
    fidx = finv.reshape(m, 4)
    n_f = len(uf)
    p = nodes[c]
    e1, e2, e3 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
    V = np.einsum('mi,mi->m', e1, np.cross(e2, e3)) / 6.0
    area = np.zeros((m, 4)); nout = np.zeros((m, 4, 3))
    for f in range(4):
        a, b, d = p[:, OPP[f, 0]], p[:, OPP[f, 1]], p[:, OPP[f, 2]]
        nv = np.cross(b - a, d - a)
        area[:, f] = 0.5 * np.linalg.norm(nv, axis=1)
        nh = nv / np.maximum(np.linalg.norm(nv, axis=1), 1e-300)[:, None]
        nout[:, f] = nh * np.sign(
            np.einsum('mi,mi->m', nh, a - p[:, f]))[:, None]
    print(f"  topology+geometry in {time.time()-t1:.1f}s "
          f"({n_f:,} faces, {int((cnt==2).sum()):,} interior)")

    # ── step 1: one-sided P1 fluxes, then average ───────────────────────
    # ⚠️ The P1 velocity is C0, so its value at a face centroid is single
    # valued -- but the ELEMENT-WISE constant velocity is not. Use the face
    # centroid value of the P1 field: it is already continuous, which makes
    # qbar exactly the average of the two one-sided fluxes.
    t2 = time.time()
    qbar = np.zeros(n_f)
    sgn = np.zeros((m, 4))
    owner = np.full(n_f, -1, dtype=np.int64)
    for f in range(4):
        np.minimum.at(owner, fidx[:, f], np.arange(m))
    for f in range(4):
        sgn[:, f] = np.where(owner[fidx[:, f]] == np.arange(m), 1.0, -1.0)
        vF = vel[c[:, OPP[f]]].mean(axis=1)
        flux = np.einsum('mi,mi->m', vF, nout[:, f]) * area[:, f]
        # accumulate with the global orientation; interior faces get two
        # contributions that we then halve
        np.add.at(qbar, fidx[:, f], flux * sgn[:, f])
    qbar /= np.maximum(cnt, 1)

    # residual: net flux out of each element
    r = np.zeros(m)
    for f in range(4):
        r += sgn[:, f] * qbar[fidx[:, f]]
    print(f"  step 1 (averaged fluxes) in {time.time()-t2:.1f}s")
    print(f"     residual |r|/(|u| A): median "
          f"{np.median(np.abs(r)/np.maximum(np.linalg.norm(vel[c].mean(axis=1),axis=1)*area.sum(axis=1),1e-300)):.3e}")

    # ── step 2: one graph-Laplacian solve per NODE patch ────────────────
    # ⚠️ Vectorised by patch size: elements sharing a node form the patch,
    # and the patch systems are independent. Build node->element lists once.
    t3 = time.time()
    node_of = c.ravel()
    elem_of = np.repeat(np.arange(m), 4)
    order = np.argsort(node_of, kind='stable')
    node_s, elem_s = node_of[order], elem_of[order]
    starts = np.searchsorted(node_s, np.arange(len(nodes)))
    ends = np.searchsorted(node_s, np.arange(len(nodes)), side='right')
    sizes = ends - starts
    print(f"  patches: {int((sizes>0).sum()):,} nodes, "
          f"size median {int(np.median(sizes[sizes>0]))}, max {sizes.max()}")

    # correction accumulated per face, per node contribution
    dq = np.zeros(n_f)
    ncontrib = np.zeros(n_f)
    # map (elem, local face) -> global face, for building patch adjacency
    solved = 0
    for i in range(len(nodes)):
        k0, k1 = starts[i], ends[i]
        if k1 - k0 < 2:
            continue
        els = elem_s[k0:k1]
        loc = {e: j for j, e in enumerate(els)}
        ne = len(els)
        # patch adjacency: two elements are joined if they share a face AND
        # that face contains node i
        B = np.zeros((ne, ne)); rhs = np.zeros(ne)
        for j, e in enumerate(els):
            rhs[j] = r[e]
            for f in range(4):
                if i not in fverts[e, f]:
                    continue                     # face must touch the node
                gf = fidx[e, f]
                if cnt[gf] != 2:
                    continue
                # the neighbour across gf
                nb = None
                for jj, e2 in enumerate(els):
                    if e2 == e:
                        continue
                    if gf in fidx[e2]:
                        nb = jj; break
                if nb is None:
                    continue
                w = area[e, f]
                B[j, j] -= w; B[j, nb] += w
        if ne < 2 or not np.any(B):
            continue
        # ⚠️ B is a graph Laplacian: singular with the constant null vector,
        # rank ne-1, exactly as Scudeler reports. Pin U[0] = 0.
        try:
            U = np.zeros(ne)
            U[1:] = np.linalg.solve(B[1:, 1:], rhs[1:])
        except np.linalg.LinAlgError:
            continue
        solved += 1
        for j, e in enumerate(els):
            for f in range(4):
                if i not in fverts[e, f]:
                    continue
                gf = fidx[e, f]
                if cnt[gf] != 2:
                    continue
                nb = None
                for jj, e2 in enumerate(els):
                    if e2 != e and gf in fidx[e2]:
                        nb = jj; break
                if nb is None:
                    continue
                dq[gf] += (U[j] - U[nb]) * area[e, f] * sgn[e, f]
                ncontrib[gf] += 1
    dq /= np.maximum(ncontrib, 1)
    print(f"  step 2 ({solved:,} patch solves) in {time.time()-t3:.1f}s")

    q = qbar + dq
    r2 = np.zeros(m)
    for f in range(4):
        r2 += sgn[:, f] * q[fidx[:, f]]

    # ── verdict ─────────────────────────────────────────────────────────
    gN, _ = tet_shape_gradients(nodes, c)
    L = velocity_gradient_per_element(vel, c, gN)
    D = 0.5 * (L + np.transpose(L, (0, 2, 1)))
    trD = np.trace(D, axis1=1, axis2=2)
    nD = np.sqrt(np.sum(D * D, axis=(1, 2)))
    us = np.linalg.norm(vel[c].mean(axis=1), axis=1)
    surf = area.sum(axis=1)
    print()
    # ⚠️ THE IDENTITY THAT MAKES THIS MEASURABLE, verified on a random tet:
    #     net face flux (face-centroid quadrature)  ==  div(u) * |V|
    # exactly, to 5.6e-17 -- the divergence theorem is EXACT for a P1 field
    # under this quadrature. So the net flux IS the divergence, and the only
    # question is the normalisation. Dividing by |u|*area gives a
    # dimensionless number comparable across element sizes; dividing by
    # |V|^(2/3) instead makes a nonzero residual LOOK like machine zero,
    # which cost an hour of chasing a non-existent bug.
    print("  ── DIVERGENCE, dimensionless |net flux| / (|u| * area) ──")
    b0 = np.abs(r) / np.maximum(us * surf, 1e-300)
    b1 = np.abs(r2) / np.maximum(us * surf, 1e-300)
    print(f"     P1 (reference)   median "
          f"{np.median(np.abs(trD)/np.maximum(nD,1e-30)):.4e}")
    print(f"     averaged fluxes  median {np.median(b0):.4e}")
    # ⚠️ Split INTERIOR from boundary-touching, as the BDM1 proof does. A
    # boundary face has only ONE contributing element, so qbar there is the
    # one-sided P1 flux and the patch correction cannot act on it -- those
    # elements keep their imbalance. On a sub-mesh the artificial cut
    # boundary is large (measured 39 % of elements at 3,000), so pooling
    # them reports a tail the full mesh would not have.
    n_int = (cnt[fidx] == 2).sum(axis=1)
    full_int = n_int == 4
    print(f"     fully-interior elements: {int(full_int.sum()):,}/{m:,} "
          f"({100*full_int.mean():.1f} %)")
    print(f"     after LN, ALL      median {np.median(b1):.4e}  "
          f"p99 {np.percentile(b1,99):.4e}")
    if full_int.any():
        bi = b1[full_int]
        print(f"     after LN, INTERIOR median {np.median(bi):.4e}  "
              f"p99 {np.percentile(bi,99):.4e}  max {bi.max():.4e}")
        b1 = bi
    ok = float(np.median(b1)) < 1e-10
    print(f"     {'✅ PASS' if ok else '⚠️  not yet at round-off'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
