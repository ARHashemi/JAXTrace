"""BDM1 divergence-free projection: the sub-mesh proof.

    python3 bdm1_submesh_proof.py --case <post dir> --pattern 'cylA_{timestep}.pvtu' \
            --timestep 159 --n-elements 20000

WHY BDM1 AND NOT RTN0
----------------------
RTN0 was built, solved to machine precision, and then measured to DISCARD
ROTATION: on a tet it spans only {constants} + {x}, so the rotational motion
that dominates FSW flow near the tool is not in the space. Measured
preservation loss ~85%. See DIVERGENCE_FREE_PLAN.md section 4b.

BDM1 has 12 DOF per tet -- exactly the 12 coefficients of a P1 VECTOR field --
so it contains ALL linear fields including both rotations (verified to 1e-15 in
test_fe_space_content.py). And div u = tr(A) is CONSTANT per element, so
div u = 0 is one scalar equation and removes ONLY the radial mode:
12 - 1 = 11 free parameters against RTN0's 4 - 1 = 3.

FORMULATION
-----------
Per element the unknown is u(x) = A x + b, 12 coefficients, but the DOFs are
shared: BDM1's degrees of freedom are the NORMAL COMPONENT sampled at the 3
vertices of each face, and they are continuous across faces. We therefore
solve for the global vector of face-vertex normal components.

  minimise   || u_h(x_c) - u_P1(x_c) ||^2        (match the P1 centroid velocity)
  subject to tr(A_E) = 0  for every element E     (exact mass conservation)

⚠️ Unlike RTN0 this is NOT a one-DOF-per-face problem, so the element-to-DOF
map is built explicitly: each (face, face-vertex) pair is one global unknown,
shared by the two elements touching that face.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

OPP = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--timestep", type=int, required=True)
    ap.add_argument("--n-elements", type=int, default=20_000)
    ap.add_argument("--field", default="Displacement")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solver", choices=["lu", "minres"], default="lu",
                    help="Direct LU (exact, but does not scale past ~1e5 "
                         "elements in 3D) or preconditioned MINRES. ⚠️ MINRES "
                         "not CG: the saddle point is symmetric INDEFINITE "
                         "and CG returns garbage on such a system.")
    ap.add_argument("--rtol", type=float, default=1e-10)
    ap.add_argument("--maxiter", type=int, default=20000)
    ap.add_argument("--ilu-drop", type=float, default=1e-5,
                    help="ILU drop tolerance. Smaller = stronger "
                         "preconditioner, more memory.")
    ap.add_argument("--ilu-fill", type=float, default=20.0)
    ap.add_argument("--precond", choices=["pazner", "ilu"], default="pazner",
                    help="'pazner' (default): block-diagonal with the "
                         "M-matrix graph-Laplacian Schur complement of "
                         "Pazner et al. 2023, Props. 4-5. 'ilu': the earlier "
                         "attempt, which fails 'exactly singular'.")
    ap.add_argument("--schur-shift", type=float, default=1e-10,
                    help="Pin for the graph-Laplacian constant null vector "
                         "in the PRECONDITIONER only.")
    ap.add_argument("--ilu-shift", type=float, default=1e-8,
                    help="Negative shift on the multiplier block, in the "
                         "PRECONDITIONER ONLY. ⚠️ Needed because ILU of a "
                         "saddle point with an exactly-zero (2,2) block "
                         "fails as 'exactly singular'. A is untouched, so "
                         "the constraint stays exact.")
    ap.add_argument("--reg", type=float, default=1e-6,
                    help="Tikhonov weight toward the P1 normal trace, as a "
                         "fraction of |N|^2. ⚠️ Needed because N^T N is "
                         "rank deficient: 9 of each element's 12 DOF are "
                         "invisible to a centroid-velocity objective. 0 "
                         "reproduces the unregularised (ill-posed) system.")
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

    n_all = len(conn)
    if args.n_elements and args.n_elements < n_all:
        rng = np.random.default_rng(args.seed)
        s0 = int(rng.integers(0, n_all - args.n_elements))
        sel = np.arange(s0, s0 + args.n_elements)
        print(f"  sub-mesh: contiguous [{s0:,}, {s0+args.n_elements:,})")
    else:
        sel = np.arange(n_all)
    c = conn[sel]
    m = len(c)

    # ── global DOF map: (face, face-vertex) ─────────────────────────────
    # ⚠️ The DOF is attached to a (face, NODE) pair, not to a local index,
    # because the two elements sharing a face list its vertices in different
    # local orders. Keying on the global node id is what makes the normal
    # component single-valued across the face.
    fverts = np.sort(c[:, OPP.ravel()].reshape(m, 4, 3), axis=2)
    flat = fverts.reshape(-1, 3)
    uf, finv, cnt = np.unique(flat, axis=0, return_inverse=True,
                              return_counts=True)
    n_f = len(uf)
    fidx = finv.reshape(m, 4)
    # DOF id = 3*face + k, where k indexes the face's SORTED vertices
    n_dof = 3 * n_f
    print(f"  faces {n_f:,}  -> {n_dof:,} BDM1 DOF "
          f"(interior {int((cnt==2).sum()):,}, boundary {int((cnt==1).sum()):,})")

    # ── geometry ────────────────────────────────────────────────────────
    p = nodes[c]
    e1, e2, e3 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
    V = np.einsum('mi,mi->m', e1, np.cross(e2, e3)) / 6.0
    xc = p.mean(axis=1)
    nout = np.zeros((m, 4, 3))
    for f in range(4):
        a, b, d = p[:, OPP[f, 0]], p[:, OPP[f, 1]], p[:, OPP[f, 2]]
        nv = np.cross(b - a, d - a)
        nh = nv / np.maximum(np.linalg.norm(nv, axis=1), 1e-300)[:, None]
        s = np.sign(np.einsum('mi,mi->m', nh, a - p[:, f]))
        nout[:, f] = nh * s[:, None]

    # ── per-element: DOFs -> (A, b) ─────────────────────────────────────
    # For each element solve the 12x12 system mapping the 12 face-vertex
    # normal components to the 12 coefficients of u = A x + b.
    # ⚠️ Built once per element and inverted locally; this is the piece that
    # makes BDM1 more work than RTN0, and it is embarrassingly parallel.
    print("  building per-element BDM1 trace maps...")
    t1 = time.time()
    T = np.zeros((m, 12, 12))            # trace map: coeffs -> DOF values
    dof_of = np.zeros((m, 12), dtype=np.int64)
    for f in range(4):
        fv = fverts[:, f, :]             # (m,3) SORTED global node ids
        for k in range(3):
            vtx = nodes[fv[:, k]]        # (m,3) that vertex's coordinates
            row = 3 * f + k
            for i in range(3):
                T[:, row, i*3:(i+1)*3] = nout[:, f, i][:, None] * vtx
                T[:, row, 9 + i] = nout[:, f, i]
            dof_of[:, row] = 3 * fidx[:, f] + k
    Tinv = np.linalg.inv(T)              # (m,12,12) DOF values -> coeffs
    print(f"     done in {time.time()-t1:.1f}s")

    # ── objective: u(x_c) = A x_c + b, linear in the DOFs ───────────────
    # u_i(x_c) = sum_j Tinv[(i*3+j), r] * xc_j * dof_r  + Tinv[9+i, r]*dof_r
    rows_N, cols_N, vals_N = [], [], []
    for i in range(3):
        coef = (np.einsum('mrj,mj->mr', Tinv[:, i*3:(i+1)*3, :].transpose(0,2,1)
                          .reshape(m, 12, 3), xc)
                + Tinv[:, 9 + i, :])
        rows_N.append(np.repeat(3*np.arange(m) + i, 12))
        cols_N.append(dof_of.ravel())
        vals_N.append(coef.ravel())
    N = sp.csr_matrix((np.concatenate(vals_N),
                       (np.concatenate(rows_N), np.concatenate(cols_N))),
                      shape=(3*m, n_dof))

    # ── constraint: tr(A) = Tinv[0,:] + Tinv[4,:] + Tinv[8,:] ───────────
    trc = Tinv[:, 0, :] + Tinv[:, 4, :] + Tinv[:, 8, :]      # (m,12)
    M = sp.csr_matrix((trc.ravel(),
                       (np.repeat(np.arange(m), 12), dof_of.ravel())),
                      shape=(m, n_dof))

    g = vel[c].mean(axis=1).reshape(-1)

    # ── boundary DOFs pinned to the P1 normal trace ─────────────────────
    isb_face = cnt == 1
    bdof = np.where(np.repeat(isb_face, 3))[0]
    qb = np.zeros(n_dof)
    for f in range(4):
        sb = isb_face[fidx[:, f]]
        if not sb.any():
            continue
        for k in range(3):
            vid = fverts[sb, f, k]
            qb[3*fidx[sb, f] + k] = np.einsum(
                'mi,mi->m', vel[vid], nout[sb, f])
    free = np.setdiff1d(np.arange(n_dof), bdof)
    print(f"  unknowns: {len(free):,} free + {len(bdof):,} pinned boundary")

    # ⚠️ DROP EMPTY CONSTRAINT ROWS -- the exact cure for the rank deficiency.
    # An element whose four faces all lie on the (sub-mesh cut) boundary has
    # EVERY DOF pinned, so its div = 0 row acts on no free unknown and is
    # identically zero. Measured on a 1,500-element block: rank(M) = 1,498 of
    # 1,500, deficiency exactly 2, and exactly 2 such elements -- a perfect
    # match. The two left null vectors were NOT the constant vector, which
    # ruled out the classic enclosed-domain pressure mode.
    #
    # ✅ Dropping them is exact, not a regularisation: the rows carry no
    # information. And on the FULL mesh they cannot occur at all -- an element
    # with four true-boundary faces would be a detached tetrahedron -- so this
    # is a sub-mesh artefact that was blocking every preconditioner.
    Nf, Mf = N[:, free], M[:, free]
    rhs_g = g - N[:, bdof] @ qb[bdof]
    rhs_f = -(M[:, bdof] @ qb[bdof])
    _rownnz = np.diff(Mf.tocsr().indptr)
    _keep = _rownnz > 0
    if not _keep.all():
        print(f"  ⚠️ dropping {int((~_keep).sum())} empty constraint rows "
              f"(elements with all 4 faces pinned) -- these were the exact "
              f"rank deficiency")
        Mf = Mf[_keep]
        rhs_f = rhs_f[_keep]

    # ⚠️ Row-scale the constraint block: the objective entries carry the
    # geometry while tr(A) carries 1/length, so the two blocks differ by
    # orders of magnitude on a 1e-4 m mesh. Scaling changes the multiplier,
    # never the solution.
    cs = float(np.sqrt((Nf.multiply(Nf)).sum() / max(Nf.nnz, 1)) /
               max(np.sqrt((Mf.multiply(Mf)).sum() / max(Mf.nnz, 1)), 1e-300))
    Mf = Mf * cs
    rhs_f = rhs_f * cs
    print(f"  constraint rows scaled by {cs:.3e}")

    # ⚠️ N^T N IS SINGULAR BY CONSTRUCTION, and this is the fix.
    # The objective sees only the 3-component centroid velocity per element,
    # while the element carries 12 DOF. So 9 directions per element are
    # invisible to it and are pinned only through face sharing -- which is
    # why preservation degraded 0.010 -> 1.62 as the sub-mesh grew while the
    # divergence stayed at 1e-16: the constraint was satisfied and the
    # objective was free to wander in the null space.
    #
    # ✅ Tikhonov term pulling every DOF toward its P1 normal trace. That is
    # the natural prior -- "stay as close to the input field as the
    # divergence constraint allows" -- and it makes the objective block
    # positive definite, so the saddle point is well posed.
    # ⚠️ Selzer & Cirpka never meet this because RTN0 has ONE dof per face
    # (4 per element), so their objective is not rank deficient.
    qp1 = np.zeros(n_dof)
    for f in range(4):
        for k in range(3):
            vid = fverts[:, f, k]
            np.add.at(qp1, 3*fidx[:, f] + k,
                      np.einsum('mi,mi->m', vel[vid], nout[:, f]))
    # average the (up to 2) contributions per DOF
    ndup = np.zeros(n_dof)
    for f in range(4):
        for k in range(3):
            np.add.at(ndup, 3*fidx[:, f] + k, 1.0)
    qp1 /= np.maximum(ndup, 1.0)

    nscale = float(np.sqrt((Nf.multiply(Nf)).sum() / max(Nf.nnz, 1)))
    eps = args.reg * nscale ** 2
    R = sp.diags(np.full(len(free), eps), format='csr')
    print(f"  Tikhonov toward the P1 trace: eps = {eps:.3e} "
          f"(--reg {args.reg:g} x |N|^2)")

    A = sp.bmat([[Nf.T @ Nf + R, Mf.T], [Mf, None]], format='csc')
    b = np.concatenate([Nf.T @ rhs_g + eps * qp1[free], rhs_f])
    print(f"  saddle-point {A.shape[0]:,} x {A.shape[1]:,}, nnz {A.nnz:,}")

    t2 = time.time()
    if args.solver == "lu":
        sol = spla.splu(A).solve(b)
        print(f"  solved (direct LU) in {time.time()-t2:.1f}s")
    else:
        # ⚠️ MINRES, not CG: the saddle point is symmetric INDEFINITE (the
        # zero block gives negative eigenvalues), and CG silently returns
        # garbage on an indefinite system rather than failing.
        #
        # Preconditioner: block-diagonal with the objective block approximated
        # by its diagonal and the Schur complement by its own diagonal. ✅ That
        # is the cheapest thing that is spectrally sensible here, and it costs
        # one vector divide per iteration.
        # ⚠️ A DIAGONAL preconditioner is too weak here: measured, MINRES hit
        # 20,000 iterations with a relative residual of 1.3e-02 and the
        # divergence degraded from 8.5e-17 to 5.1e-04. The objective block is
        # not diagonally dominant -- BDM1 DOFs couple through shared faces.
        #
        # ✅ Use an INCOMPLETE LU of the whole saddle point instead. spilu
        # handles indefinite systems with a pivot threshold, and one
        # factorisation is reused across every iteration and every snapshot
        # of the same mesh.
        n1 = len(free)
        # ── PAZNER BLOCK PRECONDITIONER (Paz23, Props. 4-5) ──────────────
        # ⚠️ My earlier diagonal preconditioner built the right Schur matrix
        # and then kept only its DIAGONAL -- which is why MINRES stalled at
        # 20,000 iterations. Pazner's result is that
        #     S~ = M diag(A)^-1 M^T
        # is an M-MATRIX GRAPH LAPLACIAN on the face-neighbour connectivity
        # of the elements, so it is well preconditioned by AMG *uniformly in
        # problem size*. Keeping S~ as a SPARSE MATRIX rather than a diagonal
        # is the whole fix.
        #
        # ✅ With <= 5 nonzeros per row (diagonal + at most 4 face
        # neighbours) S~ has 3D-Poisson sparsity and is ~7x smaller than the
        # full saddle point, so a direct factorisation of S~ alone is cheap
        # even where one of the full system is not. pyamg would be better at
        # full scale but does not build in this venv.
        if args.precond == "pazner":
            d1 = np.asarray((Nf.T @ Nf).diagonal()) + eps
            Dinv = sp.diags(1.0 / np.maximum(d1, 1e-300))
            St = (Mf @ Dinv @ Mf.T).tocsc()
            # ⚠️ S~ inherits the constant null vector of a graph Laplacian on
            # a connected patch, so pin one row to make it invertible. This
            # is a PRECONDITIONER, so the pin cannot bias the solution.
            St = St + sp.eye(St.shape[0], format='csc') * (
                args.schur_shift * float(np.abs(St.diagonal()).max()))
            t_s = time.time()
            lu_S = spla.splu(St)
            print(f"  Pazner preconditioner: S~ {St.shape[0]:,}^2, "
                  f"nnz {St.nnz:,} ({St.nnz/St.shape[0]:.1f}/row), "
                  f"factorised in {time.time()-t_s:.1f}s")

            def _apply(v):
                # block-diagonal: A-block by its diagonal, Schur by its LU
                return np.concatenate([v[:n1] / np.maximum(d1, 1e-300),
                                       lu_S.solve(v[n1:])])

            Pre = spla.LinearOperator(A.shape, matvec=_apply)
            it = {"n": 0}

            def _cb2(_):
                it["n"] += 1

            sol, info = spla.minres(A, b, M=Pre, rtol=args.rtol,
                                    maxiter=args.maxiter, callback=_cb2)
            tag = "converged" if info == 0 else f"⚠️ info={info}"
            print(f"  solved (MINRES+Pazner, {it['n']} it, {tag}) in "
                  f"{time.time()-t2:.1f}s")
            rr = float(np.linalg.norm(A @ sol - b) /
                       max(np.linalg.norm(b), 1e-300))
            print(f"     relative residual {rr:.3e}")
            # ⚠️ q is assembled once, after the branch -- do NOT set it
            # here as well, or the two paths can silently diverge.
            _skip_rest = True
        else:
            _skip_rest = False
        try:
            if _skip_rest:
                raise StopIteration
            # ⚠️ ILU on the RAW saddle point fails with "Factor is exactly
            # singular": the (2,2) block is exactly zero, so the first
            # elimination touching a multiplier row divides by zero. Measured.
            #
            # ✅ Factorise a SHIFTED matrix instead -- a small negative shift
            # on the multiplier block makes it non-singular while leaving it
            # spectrally close to the original. The shift is ONLY in the
            # preconditioner; A itself is untouched, so the solution still
            # satisfies the exact constraint.
            # ⚠️ Scale the shift by the OBJECTIVE block's magnitude, not by
            # max|diag(A)| -- the multiplier rows have a ZERO diagonal, so a
            # shift relative to the max diagonal can still be far below the
            # entries in M that surround it, leaving the pivot effectively
            # zero. Measured: three different --reg values gave byte-identical
            # output and the ILU never succeeded, which is the signature of a
            # shift too small to matter.
            _mscale = float(np.abs(Mf.data).max()) if Mf.nnz else 1.0
            shift = max(args.ilu_shift * float(np.abs(A.diagonal()).max()),
                        args.ilu_shift * _mscale)
            print(f"  ILU shift {shift:.3e} "
                  f"(diag max {float(np.abs(A.diagonal()).max()):.3e}, "
                  f"|M| max {_mscale:.3e})")
            Ap = (A + sp.diags(np.concatenate(
                [np.zeros(n1), np.full(Mf.shape[0], -shift)]),
                format='csc')).tocsc()
            ilu = spla.spilu(Ap, drop_tol=args.ilu_drop,
                             fill_factor=args.ilu_fill)
            Pre = spla.LinearOperator(A.shape, matvec=ilu.solve)
            print(f"  ILU preconditioner: drop_tol {args.ilu_drop:g}, "
                  f"fill {args.ilu_fill:g}, nnz {ilu.nnz:,} "
                  f"({ilu.nnz/A.nnz:.1f}x the matrix)")
        except StopIteration:
            pass
        except Exception as _e:        # pragma: no cover
            print(f"  ⚠️ ILU failed ({_e}); falling back to diagonal")
            d1 = np.asarray((Nf.T @ Nf).diagonal()) + eps
            S = (Mf @ sp.diags(1.0/np.maximum(d1, 1e-300)) @ Mf.T)
            d2 = np.asarray(S.diagonal())
            dinv = np.concatenate([1.0/np.maximum(d1, 1e-300),
                                   1.0/np.maximum(np.abs(d2), 1e-300)])
            Pre = spla.LinearOperator(A.shape, matvec=lambda v: dinv * v)
        it = {"n": 0}

        def _cb(_):
            it["n"] += 1

        sol, info = spla.minres(A, b, M=Pre, rtol=args.rtol,
                                maxiter=args.maxiter, callback=_cb)
        tag = "converged" if info == 0 else f"⚠️ info={info}"
        print(f"  solved (MINRES, {it['n']} it, {tag}) in "
              f"{time.time()-t2:.1f}s")
        # ⚠️ An iterative solve does NOT satisfy the constraint exactly, so
        # report the residual rather than assuming it.
        rr = float(np.linalg.norm(A @ sol - b) / max(np.linalg.norm(b), 1e-300))
        print(f"     relative residual {rr:.3e}")
    q = np.zeros(n_dof); q[free] = sol[:len(free)]; q[bdof] = qb[bdof]

    # ── verdicts ────────────────────────────────────────────────────────
    coeffs = np.einsum('mij,mj->mi', Tinv, q[dof_of])        # (m,12)
    Amat = coeffs[:, :9].reshape(m, 3, 3)
    bvec = coeffs[:, 9:]
    divu = np.trace(Amat, axis1=1, axis2=2)

    L = velocity_gradient_per_element(vel, c, gN := tet_shape_gradients(nodes, c)[0])
    D = 0.5 * (L + np.transpose(L, (0, 2, 1)))
    trD = np.trace(D, axis1=1, axis2=2)
    nD = np.sqrt(np.sum(D * D, axis=(1, 2)))
    nA = np.sqrt(np.sum(Amat * Amat, axis=(1, 2)))
    print()
    print("  ── DIVERGENCE (dimensionless: |div u| / |grad u|) ──")
    print(f"     BEFORE (P1)   median {np.median(np.abs(trD)/np.maximum(nD,1e-30)):.4e}")
    r1 = np.abs(divu) / np.maximum(nA, 1e-30)
    print(f"     AFTER  (BDM1) median {np.median(r1):.4e}  "
          f"p99 {np.percentile(r1,99):.4e}  max {r1.max():.4e}")
    ok = float(np.median(r1)) < 1e-10
    print(f"     {'✅ PASS' if ok else '❌ FAIL'}: target median < 1e-10")

    u_bdm = np.einsum('mij,mj->mi', Amat, xc) + bvec
    u_p1 = vel[c].mean(axis=1)
    r = np.linalg.norm(u_bdm - u_p1, axis=1) / np.maximum(
        np.linalg.norm(u_p1, axis=1), 1e-30)
    print()
    print("  ── PRESERVATION (||u_BDM1 - u_P1|| / ||u_P1||) ──")
    # ⚠️ Separate INTERIOR from boundary-touching elements. A boundary element
    # has some of its 12 DOFs PINNED to the P1 normal trace, which over-
    # constrains it: with fewer free parameters it cannot both satisfy
    # div = 0 and match the centroid velocity. On a sub-mesh the artificial
    # cut boundary is large (here ~40 % of faces), so pooling them would
    # report a tail that the FULL mesh would not have.
    touches_b = isb_face[fidx].any(axis=1)
    print(f"     elements touching the (sub-mesh cut) boundary: "
          f"{int(touches_b.sum()):,}/{m:,} ({100*touches_b.mean():.1f} %)")
    for lab, mask in (("INTERIOR", ~touches_b), ("boundary", touches_b),
                      ("all", np.ones(m, bool))):
        if not mask.any():
            continue
        rr = r[mask]
        print(f"     {lab:9s} n={int(mask.sum()):>7,}  "
              + "  ".join(f"p{qq}={np.percentile(rr,qq):.3e}"
                          for qq in (50, 90, 99)))
    # ✅ The interior figure is the one that predicts full-mesh behaviour.
    r = r[~touches_b] if (~touches_b).any() else r
    for qq in (50, 75, 90, 95, 99):
        print(f"     p{qq:<3} {np.percentile(r, qq):.4e}   (interior)")
    good = np.median(r) < 0.1
    print(f"     {'✅' if good else '⚠️ '} median {np.median(r):.4e}"
          + ("" if good else "   -- report this, do not bury it"))
    return 0 if (ok and good) else 1


if __name__ == "__main__":
    raise SystemExit(main())
