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

    Nf, Mf = N[:, free], M[:, free]
    rhs_g = g - N[:, bdof] @ qb[bdof]
    rhs_f = -(M[:, bdof] @ qb[bdof])

    # ⚠️ Row-scale the constraint block: the objective entries carry the
    # geometry while tr(A) carries 1/length, so the two blocks differ by
    # orders of magnitude on a 1e-4 m mesh. Scaling changes the multiplier,
    # never the solution.
    cs = float(np.sqrt((Nf.multiply(Nf)).sum() / max(Nf.nnz, 1)) /
               max(np.sqrt((Mf.multiply(Mf)).sum() / max(Mf.nnz, 1)), 1e-300))
    Mf = Mf * cs
    rhs_f = rhs_f * cs
    print(f"  constraint rows scaled by {cs:.3e}")

    A = sp.bmat([[Nf.T @ Nf, Mf.T], [Mf, None]], format='csc')
    b = np.concatenate([Nf.T @ rhs_g, rhs_f])
    print(f"  saddle-point {A.shape[0]:,} x {A.shape[1]:,}, nnz {A.nnz:,}")

    t2 = time.time()
    sol = spla.splu(A).solve(b)
    print(f"  solved in {time.time()-t2:.1f}s")
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
    for qq in (50, 75, 90, 95, 99):
        print(f"     p{qq:<3} {np.percentile(r, qq):.4e}")
    good = np.median(r) < 0.1
    print(f"     {'✅' if good else '⚠️ '} median {np.median(r):.4e}"
          + ("" if good else "   -- report this, do not bury it"))
    return 0 if (ok and good) else 1


if __name__ == "__main__":
    raise SystemExit(main())
