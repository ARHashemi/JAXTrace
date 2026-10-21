"""BDM1 divergence-free projection by overlapping additive Schwarz.

WHAT THIS IS, AND WHY IT IS BUILT THIS WAY
-------------------------------------------
The FSW FOM is a P1/P1 velocity-pressure discretisation. It is not inf-sup
stable, so it is stabilised, and its velocity is therefore NEITHER pointwise
NOR discretely divergence-free. Measured on cylA: |div u|/|D| median 2.5e-02,
and the WEAK divergence against the P1 pressure space has median 1.06e-01.

⚠️ That rules out every LOCAL method we tested:
  * Larson-Niklasson (Scudeler 2016) needs DISCONTINUOUS face fluxes -- ours
    are exactly continuous, so there is nothing to equilibrate.
  * Lederer-Schoberl (2016) needs a DISCRETELY divergence-free input -- ours
    is not, by the measurement above.
  * Pazner (2023) block preconditioning needs the (1,1) block to be spectrally
    equivalent to its diagonal -- ours has eig(D^-1 A) spanning -2.07..+8.11.

✅ So a GLOBAL constrained projection is required, and BDM1 is the space: it
contains ALL linear fields (hence rotation, which dominates FSW flow and which
lowest-order RT discards), while div u = tr(A) is CONSTANT per element, making
div u = 0 one scalar equation per element and the result POINTWISE
divergence-free inside every element.

THE DECOMPOSITION
-----------------
The global saddle point is ill-conditioned -- intrinsically, because the
objective matches only the 3-component centroid velocity against 12 DOF per
element, and that under-determinacy is exactly what lets the Tikhonov term keep
the field close to P1. Direct LU is therefore used on SUBDOMAINS.

⚠️ Each patch is a FEMUSS piece GROWN BY 3 ELEMENT LAYERS, and only the seed
elements' DOFs are kept. 3 is measured, not guessed: preservation by layer
depth from a pinned boundary is 2.72x / 2.07x / 1.47x / 1.01x the interior
asymptote at layers 0/1/2/3.

⚠️ The layer count is in ELEMENTS, not millimetres, because the contamination
propagates through FACE ADJACENCY and decays per element hop. A refined region
has more elements per mm but also needs more hops to carry the perturbation the
same distance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

OPP = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int64)

# ⚠️ Measured on cylA: preservation reaches its interior asymptote at layer 3.
# 1-2 layers retains elements 1.5-2.1x worse than interior.
DEFAULT_OVERLAP_LAYERS = 3


@dataclass
class PatchResult:
    """DOF values on a patch, and which of them are trustworthy."""
    dof_ids: np.ndarray      # global BDM1 DOF indices solved on this patch
    dof_vals: np.ndarray     # their values
    keep: np.ndarray         # bool: True where the DOF belongs to a seed element


def build_face_topology(conn: np.ndarray):
    """Global face numbering and per-element face indices.

    Returns (face_verts, elem_face_idx, face_count). A face is keyed by its
    SORTED global node ids, which is what makes it single valued across the
    two elements that share it -- they list its vertices in different local
    orders.
    """
    m = len(conn)
    fverts = np.sort(conn[:, OPP.ravel()].reshape(m, 4, 3), axis=2)
    uf, inv, cnt = np.unique(fverts.reshape(-1, 3), axis=0,
                             return_inverse=True, return_counts=True)
    return uf, inv.reshape(m, 4), cnt


def element_adjacency(elem_face_idx: np.ndarray, face_count: np.ndarray):
    """Face-neighbour lists. ⚠️ Built once for the whole mesh and reused by
    every patch, since growing a patch is a pure neighbour query."""
    m = len(elem_face_idx)
    order = np.argsort(elem_face_idx.ravel(), kind='stable')
    faces_sorted = elem_face_idx.ravel()[order]
    elems_sorted = (order // 4).astype(np.int64)
    starts = np.searchsorted(faces_sorted, np.arange(len(face_count)))
    ends = np.searchsorted(faces_sorted, np.arange(len(face_count)), side='right')
    return elems_sorted, starts, ends


def grow_patch(seed: np.ndarray, elem_face_idx: np.ndarray,
               elems_sorted: np.ndarray, starts: np.ndarray, ends: np.ndarray,
               layers: int = DEFAULT_OVERLAP_LAYERS) -> np.ndarray:
    """Grow a seed element set by `layers` face-adjacent layers.

    ✅ Returns the ENLARGED element list. The caller keeps only the seed's
    DOFs; the grown ring absorbs the boundary over-constraint.
    """
    cur = np.unique(seed)
    allp = cur
    for _ in range(layers):
        fs = np.unique(elem_face_idx[cur].ravel())
        nb = []
        for f in fs:
            nb.append(elems_sorted[starts[f]:ends[f]])
        if not nb:
            break
        cand = np.unique(np.concatenate(nb))
        new = np.setdiff1d(cand, allp, assume_unique=False)
        if len(new) == 0:
            break
        allp = np.union1d(allp, new)
        cur = new
    return allp


def solve_patch(nodes: np.ndarray, conn_patch: np.ndarray,
                vel: np.ndarray, *, reg: float = 1e-8,
                verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BDM1 divergence-free projection on one patch.

    Returns (coeffs, dof_global_within_patch, patch_face_count) where coeffs
    is (n_elem, 12): the 9 entries of A followed by b, so the reconstructed
    velocity is u(x) = A x + b -- which is what the tracking kernel evaluates.

    ⚠️ The objective matches ONLY the element-centroid velocity: 3 equations
    on 12 DOF. That is deliberate. The 'faces' alternative (12 equations) makes
    N^T N full rank but OVER-determines the element once the divergence
    constraint is added, and was measured far worse: divergence 5.5e-09 and
    preservation 9.6e+03, against 8.5e-17 and 0.097 here.
    """
    m = len(conn_patch)
    uf, fidx, cnt = build_face_topology(conn_patch)
    n_f = len(uf)
    n_dof = 3 * n_f

    p = nodes[conn_patch]
    e1, e2, e3 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
    V = np.einsum('mi,mi->m', e1, np.cross(e2, e3)) / 6.0
    xc = p.mean(axis=1)
    nout = np.zeros((m, 4, 3))
    for f in range(4):
        a, b, d = p[:, OPP[f, 0]], p[:, OPP[f, 1]], p[:, OPP[f, 2]]
        nv = np.cross(b - a, d - a)
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

    rows, cols, vals = [], [], []
    for i in range(3):
        coef = (np.einsum('mrj,mj->mr',
                          Tinv[:, i*3:(i+1)*3, :].transpose(0, 2, 1)
                          .reshape(m, 12, 3), xc) + Tinv[:, 9 + i, :])
        rows.append(np.repeat(3*np.arange(m) + i, 12))
        cols.append(dof_of.ravel())
        vals.append(coef.ravel())
    N = sp.csr_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(3*m, n_dof))
    trc = Tinv[:, 0, :] + Tinv[:, 4, :] + Tinv[:, 8, :]
    M = sp.csr_matrix((trc.ravel(),
                       (np.repeat(np.arange(m), 12), dof_of.ravel())),
                      shape=(m, n_dof))
    g = vel[conn_patch].mean(axis=1).reshape(-1)

    # Boundary DOFs pinned to the P1 normal trace.
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

    # ⚠️ Drop constraint rows with no free unknown: an element whose faces are
    # ALL pinned contributes an identically-zero row, and that is the exact
    # rank deficiency (measured: rank(M) = 1,498 of 1,500 with exactly 2 such
    # elements). Dropping is exact, not a regularisation.
    keep_rows = np.diff(Mf.tocsr().indptr) > 0
    Mf, rhs_f = Mf[keep_rows], rhs_f[keep_rows]

    nscale = float(np.sqrt((Nf.multiply(Nf)).sum() / max(Nf.nnz, 1)))
    eps = reg * nscale ** 2
    cs = nscale / max(
        float(np.sqrt((Mf.multiply(Mf)).sum() / max(Mf.nnz, 1))), 1e-300)
    Mf = Mf * cs
    rhs_f = rhs_f * cs

    A = sp.bmat([[Nf.T @ Nf + sp.diags(np.full(len(free), eps)), Mf.T],
                 [Mf, None]], format='csc')
    b = np.concatenate([Nf.T @ rhs_g + eps * qp1[free], rhs_f])
    sol = spla.splu(A).solve(b)
    q = np.zeros(n_dof); q[free] = sol[:len(free)]; q[bdof] = qb[bdof]

    coeffs = np.einsum('mij,mj->mi', Tinv, q[dof_of])
    if verbose:
        Amat = coeffs[:, :9].reshape(m, 3, 3)
        dv = np.abs(np.trace(Amat, axis1=1, axis2=2))
        nA = np.sqrt(np.sum(Amat*Amat, axis=(1, 2)))
        print(f"      patch {m:,} el: div median "
              f"{np.median(dv/np.maximum(nA,1e-30)):.3e}")
    return coeffs, fidx, cnt
