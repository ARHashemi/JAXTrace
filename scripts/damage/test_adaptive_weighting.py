"""Does NON-UNIFORM weighting change the BDM1 outcome? (§4v next test)

§4v confirmed that the repair the divergence constraint demands varies ~43x
across the mesh while solve_patch meets it with THREE GLOBAL SCALARS. This
asks whether that uniformity is actually causing harm -- the premise first,
before any adaptive machinery is built (§4-standing-rule).

Three variants on the SAME patch, identical in every other respect:
  uniform    production: eps = reg * nscale^2 on every DOF        (baseline)
  by_repair  data rows weighted by the local repair cost |div|/(sqrt3 |A|):
             elements that need a big correction get a SMALLER data weight,
             so the fit yields where the constraint is expensive
  by_normA   data rows weighted by 1/|A|_F: equalises the GRADIENT scale,
             since |A| spans orders of magnitude and the unweighted normal
             equations are therefore dominated by the stiffest elements

⚠️ This does NOT touch jaxtrace/damage/divfree.py. The assembly is duplicated
here deliberately so the production path stays byte-identical while the
experiment varies one thing.

If all three give the same preservation, uniformity is NOT the cause and the
remaining suspect is the pinned boundary trace alone.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jaxtrace.damage.divfree import (                       # noqa: E402
    OPP, build_face_topology, element_adjacency, grow_patch)


def assemble_and_solve(nodes, conn_patch, vel, *, reg=1e-8, mode="uniform"):
    """solve_patch's assembly, with the row weighting as a free parameter.

    Returns (coeffs, info). mode selects the data-row weights w (one per
    element, broadcast to its 3 centroid equations).
    """
    m = len(conn_patch)
    uf, fidx, cnt = build_face_topology(conn_patch)
    n_f = len(uf)
    n_dof = 3 * n_f

    p = nodes[conn_patch]
    xc = p.mean(axis=1)
    nout = np.zeros((m, 4, 3))
    for f in range(4):
        a, b_, d = p[:, OPP[f, 0]], p[:, OPP[f, 1]], p[:, OPP[f, 2]]
        nv = np.cross(b_ - a, d - a)
        nh = nv / np.maximum(np.linalg.norm(nv, axis=1), 1e-300)[:, None]
        nout[:, f] = nh * np.sign(
            np.einsum('mi,mi->m', nh, a - p[:, f]))[:, None]

    fverts = np.sort(conn_patch[:, OPP.ravel()].reshape(m, 4, 3), axis=2)
    T = np.zeros((m, 12, 12))
    dof_of = np.zeros((m, 12), dtype=np.int64)
    for f in range(4):
        for k in range(3):
            vtx = nodes[fverts[:, f, k]]
            row = 3 * f + k
            for i in range(3):
                T[:, row, i*3:(i+1)*3] = nout[:, f, i][:, None] * vtx
                T[:, row, 9 + i] = nout[:, f, i]
            dof_of[:, row] = 3 * fidx[:, f] + k
    Tinv = np.linalg.inv(T)

    # ── the per-element weight ──────────────────────────────────────────────
    # Computed from the P1 DATA (exact affine fit per tet), so it is available
    # before any solve -- which is what makes an adaptive scheme feasible.
    X4 = np.concatenate([p, np.ones((m, 4, 1))], axis=2)
    Cfit = np.linalg.solve(X4, vel[conn_patch])
    Afit = Cfit[:, :3, :].transpose(0, 2, 1)
    nA = np.sqrt(np.sum(Afit * Afit, axis=(1, 2)))
    divf = np.abs(np.trace(Afit, axis1=1, axis2=2))
    repair = divf / np.maximum(np.sqrt(3.0) * nA, 1e-300)

    if mode == "uniform":
        w = np.ones(m)
    elif mode == "by_repair":
        # yield where the constraint is expensive
        w = 1.0 / np.maximum(repair / max(np.median(repair), 1e-300), 1e-3)
        w = np.clip(w, 1e-3, 1e3)
    elif mode == "by_normA":
        w = np.median(nA) / np.maximum(nA, 1e-300)
        w = np.clip(w, 1e-3, 1e3)
    else:
        raise SystemExit(f"unknown mode {mode}")
    w = w / np.sqrt(np.mean(w * w))          # unit RMS: keeps nscale comparable

    rows, cols, vals = [], [], []
    for i in range(3):
        coef = (np.einsum('mrj,mj->mr',
                          Tinv[:, i*3:(i+1)*3, :].transpose(0, 2, 1)
                          .reshape(m, 12, 3), xc) + Tinv[:, 9 + i, :])
        rows.append(np.repeat(3*np.arange(m) + i, 12))
        cols.append(dof_of.ravel())
        vals.append((coef * w[:, None]).ravel())
    N = sp.csr_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(3*m, n_dof))
    trc = Tinv[:, 0, :] + Tinv[:, 4, :] + Tinv[:, 8, :]
    M = sp.csr_matrix((trc.ravel(),
                       (np.repeat(np.arange(m), 12), dof_of.ravel())),
                      shape=(m, n_dof))
    g = (vel[conn_patch].mean(axis=1) * w[:, None]).reshape(-1)

    isb_face = cnt == 1
    bdof = np.where(np.repeat(isb_face, 3))[0]
    qb = np.zeros(n_dof)
    qp1 = np.zeros(n_dof); ndup = np.zeros(n_dof)
    for f in range(4):
        for k in range(3):
            vid = fverts[:, f, k]
            tr = np.einsum('mi,mi->m', vel[vid], nout[:, f])
            np.add.at(qp1, 3*fidx[:, f] + k, tr)
            np.add.at(ndup, 3*fidx[:, f] + k, 1.0)
            sb = isb_face[fidx[:, f]]
            if sb.any():
                qb[3*fidx[sb, f] + k] = tr[sb]
    qp1 /= np.maximum(ndup, 1.0)

    free = np.setdiff1d(np.arange(n_dof), bdof)
    Nf, Mf = N[:, free], M[:, free]
    rhs_g = g - N[:, bdof] @ qb[bdof]
    rhs_f = -(M[:, bdof] @ qb[bdof])
    keep = np.diff(Mf.tocsr().indptr) > 0
    Mf, rhs_f = Mf[keep], rhs_f[keep]

    nscale = float(np.sqrt((Nf.multiply(Nf)).sum() / max(Nf.nnz, 1)))
    eps = reg * nscale ** 2
    cs = nscale / max(
        float(np.sqrt((Mf.multiply(Mf)).sum() / max(Mf.nnz, 1))), 1e-300)
    Mf, rhs_f = Mf * cs, rhs_f * cs

    K = sp.bmat([[Nf.T @ Nf + sp.diags(np.full(len(free), eps)), Mf.T],
                 [Mf, None]], format='csc')
    rhs = np.concatenate([Nf.T @ rhs_g + eps * qp1[free], rhs_f])
    t0 = time.time()
    sol = spla.splu(K).solve(rhs)
    q = np.zeros(n_dof); q[free] = sol[:len(free)]; q[bdof] = qb[bdof]
    coeffs = np.einsum('mij,mj->mi', Tinv, q[dof_of])
    return coeffs, dict(lu=time.time()-t0, repair=repair, nA=nA, w=w)


def report(tag, nodes, conn_patch, vel, coeffs, interior):
    Amat = coeffs[:, :9].reshape(-1, 3, 3)
    bvec = coeffs[:, 9:]
    dv = np.abs(np.trace(Amat, axis1=1, axis2=2))
    nA = np.sqrt(np.sum(Amat*Amat, axis=(1, 2)))
    div_rel = dv / np.maximum(nA, 1e-30)
    xc = nodes[conn_patch].mean(axis=1)
    u_new = np.einsum('mij,mj->mi', Amat, xc) + bvec
    u_p1 = vel[conn_patch].mean(axis=1)
    pres = np.linalg.norm(u_new - u_p1, axis=1) / np.maximum(
        np.linalg.norm(u_p1, axis=1), 1e-30)
    pi = pres[interior]
    print(f"  {tag:<10}  div p50 {np.median(div_rel):.3e}"
          f"   preservation p50 {np.median(pi):.4e}"
          f"  p90 {np.percentile(pi,90):.4e}"
          f"  p99 {np.percentile(pi,99):.4e}")
    return np.median(pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--pattern', required=True)
    ap.add_argument('--timestep', type=int, required=True)
    ap.add_argument('--field', default='Displacement')
    ap.add_argument('--seed-size', type=int, default=4000)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--reg', type=float, default=1e-8)
    ap.add_argument('--n-patches', type=int, default=4)
    a = ap.parse_args()

    from jaxtrace.gpu.mesh_loader_timedep import load_field_sequences_from_pvtu
    from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
    nodes, conn, seq = load_field_sequences_from_pvtu(
        Path(a.case), a.pattern, (a.timestep, a.timestep), [a.field],
        verbose=False, n_workers=8)
    o = deduplicate_nodes(nodes, conn, seq[a.field], verbose=False)
    nodes, conn, vel = o[0], o[1].astype(np.int64), o[3][0].astype(np.float64)
    print(f"mesh {len(nodes):,} nodes  {len(conn):,} elements", flush=True)

    uf_g, fidx_g, cnt_g = build_face_topology(conn)
    # ⚠️ takes the face-COUNT array, not the face count. Passing len(uf_g)
    # yields an int and searchsorted then builds the wrong ranges.
    es, st, en = element_adjacency(fidx_g, cnt_g)
    rng = np.random.default_rng(0)

    for ip in range(a.n_patches):
        # contiguous seed: grow from one element until seed-size is reached
        start = int(rng.integers(len(conn)))
        seed = np.array([start])
        while len(seed) < a.seed_size:
            bigger = grow_patch(seed, fidx_g, es, st, en, layers=1)
            if len(bigger) == len(seed):
                break
            seed = bigger
        seed = seed[:a.seed_size]
        patch = grow_patch(seed, fidx_g, es, st, en, layers=a.layers)
        # interior = the seed elements, which is what production keeps
        inseed = np.isin(patch, seed)
        print(f"\npatch {ip}: seed {len(seed):,} -> grown {len(patch):,} el"
              f"  (interior {inseed.sum():,})", flush=True)
        cp = conn[patch]
        for mode in ("uniform", "by_repair", "by_normA"):
            co, info = assemble_and_solve(nodes, cp, vel, reg=a.reg, mode=mode)
            report(mode, nodes, cp, vel, co, inseed)


if __name__ == '__main__':
    main()
