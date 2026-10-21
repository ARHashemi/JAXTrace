"""RTN0 projection: the sub-mesh proof. Step 1 of DIVERGENCE_FREE_PLAN.md.

    python3 rtn0_submesh_proof.py --case <post dir> --pattern 'cylA_{timestep}.pvtu' \
            --timestep 159 --n-elements 100000

WHAT THIS PROVES, AND THE PASS/FAIL
------------------------------------
Selzer & Cirpka (2020) project a non-conforming P1 velocity onto lowest-order
Raviart-Thomas-Nedelec space subject to element-wise mass conservation. The
constraint is enforced EXACTLY by a Lagrange multiplier, so after the solve the
element divergence must be at ROUND-OFF -- target |div u| / |u| * h < 1e-10.

⚠️ That is a far sharper test than "is it smaller than before". A previous
attempt (one extra centroid DOF) returned exactly 1.00x and would have been
caught here in minutes; it was not run, and weeks of reasoning followed. This
script exists so that cannot happen again.

THE SYSTEM
----------
Unknowns q = face normal fluxes. Objective (their Eq. 22): match the P1
velocity at element centroids. Constraint (their Eq. 24): div u = 0 per
element, since the FSW workpiece has no volumetric source.

    [ N^T N   M^T ] [ q      ]   [ N^T g ]
    [ M       0   ] [ lambda ] = [ 0     ]

with N_im = psi_{F_m}(x_c,i) the RTN0 basis at the centroid of E_i, and
M_im = sign * |F_m| the signed face area.

⚠️ K = I, unlike the paper. Their N is weighted by the inverse hydraulic
conductivity because in groundwater the natural metric is the hydraulic
gradient. We have no K -- our velocity is a computed flow field, not a Darcy
flux -- so the objective reduces to plain L2 closeness of the centroid
velocity. See DIVERGENCE_FREE_PLAN.md section 3.4.

RTN0 BASIS on a tet with vertices x_0..x_3, face F_m opposite vertex n_m:

    psi_{F_m}(x) = sigma_m * |F_m| / (3 |E|) * (x - x_{n_m})

sigma_m = +-1 orients the face globally so the normal component is single
valued across it -- the property that makes the field conforming. We fix it by
the global vertex ordering of the face, which both neighbours agree on.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--timestep", type=int, required=True)
    ap.add_argument("--n-elements", type=int, default=100_000)
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

    # ── sub-mesh: a CONTIGUOUS block, so adjacency is real ──────────────
    # ⚠️ A random subset of elements is topologically disconnected, and RTN0
    # couples neighbours through shared faces -- a scattered sample would
    # have almost no interior faces and would test nothing.
    n_all = len(conn)
    if args.n_elements and args.n_elements < n_all:
        rng = np.random.default_rng(args.seed)
        s0 = int(rng.integers(0, n_all - args.n_elements))
        sel = np.arange(s0, s0 + args.n_elements)
        print(f"  sub-mesh: contiguous [{s0:,}, {s0+args.n_elements:,})")
    else:
        sel = np.arange(n_all)
        print(f"  sub-mesh: WHOLE mesh")
    c = conn[sel]
    m = len(c)

    # ── face topology ───────────────────────────────────────────────────
    # Face f of a tet is the one OPPOSITE vertex f.
    OPP = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    faces = np.sort(c[:, OPP.ravel()].reshape(m, 4, 3), axis=2).reshape(-1, 3)
    uf, inv, cnt = np.unique(faces, axis=0, return_inverse=True,
                             return_counts=True)
    n_f = len(uf)
    interior = cnt[inv].reshape(m, 4) == 2
    print(f"  faces {n_f:,}  interior {int((cnt==2).sum()):,}  "
          f"boundary {int((cnt==1).sum()):,}")

    # ── geometry ────────────────────────────────────────────────────────
    p = nodes[c]
    e1, e2, e3 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
    V = np.einsum('mi,mi->m', e1, np.cross(e2, e3)) / 6.0
    xc = p.mean(axis=1)
    gN, _ = tet_shape_gradients(nodes, c)

    # Face areas and outward normals, per (element, local face).
    area = np.zeros((m, 4)); nout = np.zeros((m, 4, 3))
    for f in range(4):
        a, b, d = (p[:, OPP[f, 0]], p[:, OPP[f, 1]], p[:, OPP[f, 2]])
        nv = np.cross(b - a, d - a)
        A = 0.5 * np.linalg.norm(nv, axis=1)
        nh = nv / np.maximum(np.linalg.norm(nv, axis=1), 1e-300)[:, None]
        # orient outward: away from the opposite vertex
        s = np.sign(np.einsum('mi,mi->m', nh, a - p[:, f]))
        area[:, f] = A
        nout[:, f] = nh * s[:, None]

    # ── assemble N (objective) and M (constraint) ───────────────────────
    # ⚠️ GLOBAL face orientation. Both neighbours must agree on the sign of
    # q_F, or the normal component is double valued and the field is not
    # conforming -- the whole point of RTN0. Fix it by the element index:
    # the LOWER-indexed element of the pair defines the positive direction.
    owner = np.full(n_f, -1, dtype=np.int64)
    idx = inv.reshape(m, 4)
    for f in range(4):
        np.minimum.at(owner, idx[:, f], np.arange(m))
    owner_ok = owner[idx]                       # (m,4) lowest elem per face
    sgn = np.where(owner_ok == np.arange(m)[:, None], 1.0, -1.0)

    rows_N, cols_N, vals_N = [], [], []
    rows_M, cols_M, vals_M = [], [], []
    for f in range(4):
        # psi_F(x_c) = |F| / (3 V) * (x_c - x_opp)
        d = xc - p[:, f]                        # vertex f is opposite face f
        coef = (area[:, f] / (3.0 * np.abs(V)))[:, None] * d   # (m,3)
        coef = coef * sgn[:, f][:, None]
        for comp in range(3):
            rows_N.append(3 * np.arange(m) + comp)
            cols_N.append(idx[:, f])
            vals_N.append(coef[:, comp])
        # mass: flux out of E through F is q_F * sgn, times |F| folded in
        rows_M.append(np.arange(m))
        cols_M.append(idx[:, f])
        vals_M.append(sgn[:, f] * area[:, f])

    N = sp.csr_matrix(
        (np.concatenate(vals_N),
         (np.concatenate(rows_N), np.concatenate(cols_N))), shape=(3 * m, n_f))
    M = sp.csr_matrix(
        (np.concatenate(vals_M),
         (np.concatenate(rows_M), np.concatenate(cols_M))), shape=(m, n_f))

    # target: the P1 velocity at element centroids
    g = vel[c].mean(axis=1).reshape(-1)

    # ⚠️ BOUNDARY FACES are not free: without pinning them the constraint
    # set is rank deficient (a face touched by one element can absorb any
    # residual). Pin them to the P1 normal flux, which is also what step 2
    # of the plan must do properly at full scale.
    bfac = np.where(cnt == 1)[0]
    qb = np.zeros(n_f)
    for f in range(4):
        isb = cnt[idx[:, f]] == 1
        if isb.any():
            vF = vel[c[:, OPP[f]]].mean(axis=1)              # face-centre vel
            qb[idx[isb, f]] = (np.einsum('mi,mi->m', vF[isb], nout[isb, f])
                               * area[isb, f] * sgn[isb, f])
    free = np.setdiff1d(np.arange(n_f), bfac)
    print(f"  unknowns: {len(free):,} free + {len(bfac):,} pinned boundary")

    Nf, Mf = N[:, free], M[:, free]
    rhs_g = g - N[:, bfac] @ qb[bfac]
    rhs_f = -(M[:, bfac] @ qb[bfac])

    A = sp.bmat([[Nf.T @ Nf, Mf.T], [Mf, None]], format='csc')
    b = np.concatenate([Nf.T @ rhs_g, rhs_f])
    print(f"  saddle-point system {A.shape[0]:,} x {A.shape[1]:,}, "
          f"nnz {A.nnz:,}")

    t1 = time.time()
    # ⚠️ The block is symmetric INDEFINITE, so a Cholesky-style solver is
    # wrong; splu handles it. A tiny regularisation on the multiplier block
    # guards the constant-pressure null space of a closed domain.
    A = A + sp.diags(np.concatenate([
        np.zeros(len(free)), np.full(m, -1e-12)]), format='csc')
    sol = spla.splu(A).solve(b)
    print(f"  solved in {time.time()-t1:.1f}s")

    q = np.zeros(n_f); q[free] = sol[:len(free)]; q[bfac] = qb[bfac]

    # ── VERDICT 1: divergence ───────────────────────────────────────────
    div = (M @ q) / np.abs(V)            # 1/s, per element
    L = velocity_gradient_per_element(vel, c, gN)
    D = 0.5 * (L + np.transpose(L, (0, 2, 1)))
    trD = np.trace(D, axis1=1, axis2=2)
    nD = np.sqrt(np.sum(D * D, axis=(1, 2)))
    rel0 = np.abs(trD) / np.maximum(nD, 1e-30)
    rel1 = np.abs(div) / np.maximum(nD, 1e-30)
    print()
    print("  ── DIVERGENCE ──")
    print(f"     BEFORE (P1)   median {np.median(rel0):.4e}  "
          f"max {rel0.max():.4e}")
    print(f"     AFTER  (RTN0) median {np.median(rel1):.4e}  "
          f"max {rel1.max():.4e}")
    ok = float(np.median(rel1)) < 1e-10
    print(f"     {'✅ PASS' if ok else '❌ FAIL'}: target median < 1e-10 "
          f"(the constraint is EXACT, so anything larger is an assembly bug)")

    # ── VERDICT 2: preservation ─────────────────────────────────────────
    # ⚠️ A projection that conserves mass by destroying the field is not a
    # success. This is the number that answers "while preserving it".
    u_rtn = np.zeros((m, 3))
    for f in range(4):
        d = xc - p[:, f]
        u_rtn += ((area[:, f] / (3.0 * np.abs(V)))[:, None] * d
                  * (sgn[:, f] * q[idx[:, f]])[:, None])
    u_p1 = vel[c].mean(axis=1)
    du = np.linalg.norm(u_rtn - u_p1, axis=1)
    su = np.linalg.norm(u_p1, axis=1)
    r = du / np.maximum(su, 1e-30)
    print()
    print("  ── PRESERVATION (||u_RTN0 - u_P1|| / ||u_P1||) ──")
    for qq in (50, 75, 90, 95, 99):
        print(f"     p{qq:<3} {np.percentile(r, qq):.4e}")
    print(f"     mean {r.mean():.4e}")
    print(f"     {'✅' if np.median(r) < 0.1 else '⚠️ '} median "
          f"{np.median(r):.4e}" +
          ("" if np.median(r) < 0.1 else "  -- the field moved a lot; "
           "report this, do not bury it"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
