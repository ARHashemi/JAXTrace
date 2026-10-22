"""THE DECISIVE TEST (§4y next step): does ln J vanish ALONG A PATHLINE?

§4y: the closed-form correction A' = A - (tr A/3) I is divergence-free to
~4e-18 PER ELEMENT, with the gradient preserved to 0.99998. But the field is
discontinuous across faces, so a particle crossing one sees a jump in A.

⚠️ Per-element divergence of 4e-18 is NECESSARY, NOT SUFFICIENT. What tracking
actually integrates is

    ln J = int div(u) dt   along the pathline,  det F = exp(ln J)

accumulated at the particle position each RK4 substep (see
benchmark_femuss_comparison.py:1855). If the integrator's velocity is the
corrected field, every sample lands inside some element where div = 0 to
round-off, so ln J should stay at round-off NO MATTER how the particle crosses
faces -- the jump in A changes WHICH zero-divergence map is sampled, never
that it is zero.

That is the prediction. It could still fail two ways, which is why this is run
rather than argued:
  (a) the particle leaves the mesh / lands in no element, and the sampler
      returns 0 or garbage (the tau_min trap of the detachment work);
  (b) the corrected field's DISPLACEMENT differs enough from the raw P1 field
      that particles visit materially different elements, so det F is fine but
      the TRAJECTORY has moved -- which matters for the damage map even if
      ln J is clean.

So this marches the SAME seeds through BOTH fields and reports ln J and the
trajectory divergence side by side.

✅ Pure numpy, single process, no JAX and no vmap -- this is a correctness
probe, not the production integrator, and it is deliberately independent of it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def affine_maps(nodes, conn, vel):
    """Per-element affine map of the RAW P1 field: A (m,3,3), b (m,3)."""
    p = nodes[conn]
    X4 = np.concatenate([p, np.ones((len(conn), 4, 1))], axis=2)
    C = np.linalg.solve(X4, vel[conn])
    return C[:, :3, :].transpose(0, 2, 1), C[:, 3, :]


def corrected(A, b, nodes, conn):
    """§4y closed form: minimum-norm divergence-free correction.

    ⚠️ This is the SAME field the kernel builds in-kernel as
        u'(x) = u(x) - (tr A/3)(x - xc)
    (§4ab) -- verified identical to 5.4e-15 by
    scripts/damage/test_divfree_kernel_equiv.py. Expressed here as (A', b')
    because this integrator stores per-element affine maps.
    """
    xc = nodes[conn].mean(axis=1)
    tr = np.trace(A, axis1=1, axis2=2)
    Ap = A - (tr / 3.0)[:, None, None] * np.eye(3)
    bp = b + np.einsum('mij,mj->mi', A - Ap, xc)
    return Ap, bp


class Locator:
    """Barycentric point location by brute force over a LOCAL candidate set.

    ⚠️ Deliberately simple: a uniform background grid of element bounding
    boxes. The production octree is not reused here because the point of this
    test is to be INDEPENDENT of the tracking machinery it is validating.
    """

    def __init__(self, nodes, conn, target_per_cell=12.0):
        self.nodes, self.conn = nodes, conn
        p = nodes[conn]
        self.lo = p.min(axis=1)
        self.hi = p.max(axis=1)
        self.gmin = nodes.min(axis=0)
        self.gmax = nodes.max(axis=0)
        ext = self.gmax - self.gmin
        n = max(int(round((len(conn) / target_per_cell) ** (1.0 / 3.0))), 1)
        self.n = n
        self.h = ext / n
        self.h[self.h <= 0] = 1.0
        ilo = np.floor((self.lo - self.gmin) / self.h).astype(np.int64)
        ihi = np.floor((self.hi - self.gmin) / self.h).astype(np.int64)
        np.clip(ilo, 0, n - 1, out=ilo)
        np.clip(ihi, 0, n - 1, out=ihi)
        # ⚠️ An element spanning many cells is registered in all of them, as
        # the production multi-cell vertex registration does.
        cells, elems = [], []
        span = (ihi - ilo + 1)
        if span.prod(axis=1).sum() > 60_000_000:
            raise SystemExit("locator grid too fine; raise target_per_cell")
        for e in range(len(conn)):
            xs = np.arange(ilo[e, 0], ihi[e, 0] + 1)
            ys = np.arange(ilo[e, 1], ihi[e, 1] + 1)
            zs = np.arange(ilo[e, 2], ihi[e, 2] + 1)
            gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
            c = (gx.ravel() * n + gy.ravel()) * n + gz.ravel()
            cells.append(c)
            elems.append(np.full(len(c), e, dtype=np.int64))
        cells = np.concatenate(cells)
        elems = np.concatenate(elems)
        order = np.argsort(cells, kind='stable')
        self.cell_of = cells[order]
        self.elem_of = elems[order]
        self.start = np.searchsorted(self.cell_of, np.arange(n**3))
        self.end = np.searchsorted(self.cell_of, np.arange(n**3), side='right')
        # precompute the 4x4 inverse for barycentric tests
        pp = nodes[conn]
        M = np.concatenate([pp.transpose(0, 2, 1),
                            np.ones((len(conn), 1, 4))], axis=1)
        self.Minv = np.linalg.inv(M)

    def find(self, x, hint=-1, tol=-1e-9):
        """Return element id containing x, or -1. Tries `hint` first."""
        if hint >= 0:
            lam = self.Minv[hint] @ np.append(x, 1.0)
            if lam.min() >= tol:
                return hint
        i = np.floor((x - self.gmin) / self.h).astype(np.int64)
        if np.any(i < 0) or np.any(i >= self.n):
            return -1
        c = (i[0] * self.n + i[1]) * self.n + i[2]
        cand = self.elem_of[self.start[c]:self.end[c]]
        if len(cand) == 0:
            return -1
        lam = np.einsum('kij,j->ki', self.Minv[cand], np.append(x, 1.0))
        ok = np.where(lam.min(axis=1) >= tol)[0]
        return int(cand[ok[0]]) if len(ok) else -1


def march(seeds, loc, A, b, dt, nsteps):
    """RK4 with ln J accumulated per substep, exactly as the kernel does.

    Returns (pos, lnJ, alive, n_cross) -- n_cross counts element changes, so
    the face-jump exposure is measured rather than assumed.
    """
    n = len(seeds)
    pos = seeds.copy()
    lnJ = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    hint = np.full(n, -1, dtype=np.int64)
    ncross = np.zeros(n, dtype=np.int64)
    for _ in range(nsteps):
        for p_i in range(n):
            if not alive[p_i]:
                continue
            x = pos[p_i]
            ks = []
            acc = 0.0
            for stage, (cx, w) in enumerate(
                    ((x, 0.5), (None, 1.0), (None, 1.0), (None, 0.5))):
                if stage == 1:
                    cx = x + 0.5 * dt * ks[0]
                elif stage == 2:
                    cx = x + 0.5 * dt * ks[1]
                elif stage == 3:
                    cx = x + dt * ks[2]
                e = loc.find(cx, hint[p_i])
                if e < 0:
                    alive[p_i] = False
                    break
                if hint[p_i] >= 0 and e != hint[p_i]:
                    ncross[p_i] += 1
                hint[p_i] = e
                ks.append(A[e] @ cx + b[e])
                # ⚠️ div is sampled at EVERY substep, as the kernel does, with
                # the RK4 weights -- not once per step.
                acc += (1.0 / 6.0 if stage in (0, 3) else 2.0 / 6.0) * \
                    np.trace(A[e])
            if not alive[p_i]:
                continue
            pos[p_i] = x + dt / 6.0 * (ks[0] + 2*ks[1] + 2*ks[2] + ks[3])
            lnJ[p_i] += dt * acc
    return pos, lnJ, alive, ncross


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--pattern', required=True)
    ap.add_argument('--timestep', type=int, required=True)
    ap.add_argument('--field', default='Displacement')
    ap.add_argument('--n-seeds', type=int, default=300)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--dt', type=float, default=0.0)
    ap.add_argument('--sub-elems', type=int, default=1200000,
                    help="restrict to a spatial sub-box of ~this many "
                         "elements so the locator stays cheap")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from jaxtrace.gpu.mesh_loader_timedep import load_field_sequences_from_pvtu
    from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

    t0 = time.time()
    nodes, conn, seq = load_field_sequences_from_pvtu(
        Path(a.case), a.pattern, (a.timestep, a.timestep), [a.field],
        verbose=False, n_workers=8)
    o = deduplicate_nodes(nodes, conn, seq[a.field], verbose=False)
    nodes, conn, vel = o[0], o[1].astype(np.int64), o[3][0].astype(np.float64)
    print(f"mesh {len(nodes):,} nodes {len(conn):,} el ({time.time()-t0:.1f}s)",
          flush=True)

    # ⚠️ Restrict to a sub-box around the mesh centre: the locator is O(N) to
    # build and this test needs locality, not coverage. Seeds are placed well
    # inside it so particles do not leave through the artificial boundary.
    xc_all = nodes[conn].mean(axis=1)
    c = 0.5 * (nodes.min(axis=0) + nodes.max(axis=0))
    frac = min((a.sub_elems / len(conn)) ** (1.0 / 3.0), 1.0)
    half = 0.5 * frac * (nodes.max(axis=0) - nodes.min(axis=0))
    m = np.all(np.abs(xc_all - c) <= half, axis=1)
    sub = np.where(m)[0]
    print(f"sub-box: {len(sub):,} elements ({100*len(sub)/len(conn):.1f} %)",
          flush=True)
    conn_s = conn[sub]

    A_raw, b_raw = affine_maps(nodes, conn_s, vel)
    A_cor, b_cor = corrected(A_raw, b_raw, nodes, conn_s)
    print(f"raw  div |tr A| p50 {np.median(np.abs(np.trace(A_raw,axis1=1,axis2=2))):.4e}")
    print(f"corr div |tr A| p50 {np.median(np.abs(np.trace(A_cor,axis1=1,axis2=2))):.4e}",
          flush=True)

    t1 = time.time()
    loc = Locator(nodes, conn_s)
    print(f"locator built {time.time()-t1:.1f}s  grid {loc.n}^3", flush=True)

    # seeds: element centroids well inside the sub-box
    rng = np.random.default_rng(0)
    xc_s = nodes[conn_s].mean(axis=1)
    inner = np.all(np.abs(xc_s - c) <= 0.6 * half, axis=1)
    cand = np.where(inner)[0]
    pick = rng.choice(cand, min(a.n_seeds, len(cand)), replace=False)
    seeds = xc_s[pick]

    # dt from a CFL-like estimate on the sub-box
    h = np.cbrt(np.abs(np.einsum(
        'mi,mi->m', nodes[conn_s][:, 1]-nodes[conn_s][:, 0],
        np.cross(nodes[conn_s][:, 2]-nodes[conn_s][:, 0],
                 nodes[conn_s][:, 3]-nodes[conn_s][:, 0]))/6.0).mean())
    umax = np.linalg.norm(vel, axis=1).max()
    dt = a.dt if a.dt > 0 else 0.25 * h / max(umax, 1e-30)
    print(f"seeds {len(seeds)}  steps {a.steps}  dt {dt:.4e} "
          f"(h~{h:.4e}, |u|max {umax:.4e})", flush=True)

    out = {}
    for tag, (A_, b_) in (("raw P1", (A_raw, b_raw)),
                          ("corrected", (A_cor, b_cor))):
        t2 = time.time()
        pos, lnJ, alive, ncr = march(seeds, loc, A_, b_, dt, a.steps)
        out[tag] = (pos, lnJ, alive, ncr)
        al = alive.sum()
        print(f"\n=== {tag} === ({time.time()-t2:.1f}s)  alive {al}/{len(seeds)}",
              flush=True)
        if al:
            lj = np.abs(lnJ[alive])
            detF = np.exp(lnJ[alive])
            print(f"  |ln J|      p50 {np.median(lj):.4e}  p99 "
                  f"{np.percentile(lj,99):.4e}  max {lj.max():.4e}")
            print(f"  |det F - 1| p50 {np.median(np.abs(detF-1)):.4e}  max "
                  f"{np.abs(detF-1).max():.4e}")
            print(f"  element crossings per particle: p50 "
                  f"{np.median(ncr[alive]):.0f}  max {ncr[alive].max()}")

    ra, ca = out["raw P1"], out["corrected"]
    both = ra[2] & ca[2]
    print(f"\n=== COMPARISON (both alive: {both.sum()}/{len(seeds)}) ===")
    if both.any():
        d = np.linalg.norm(ra[0][both] - ca[0][both], axis=1)
        trav = np.linalg.norm(ca[0][both] - seeds[both], axis=1)
        print(f"  |ln J| raw  p50 {np.median(np.abs(ra[1][both])):.4e}")
        print(f"  |ln J| corr p50 {np.median(np.abs(ca[1][both])):.4e}")
        gain = np.median(np.abs(ra[1][both]))/max(np.median(np.abs(ca[1][both])),1e-300)
        print(f"  -> ln J improved {gain:.3e} x")
        print(f"  trajectory shift |x_raw - x_corr| p50 {np.median(d):.4e}"
              f"  p99 {np.percentile(d,99):.4e}")
        print(f"  distance travelled               p50 {np.median(trav):.4e}")
        print(f"  relative trajectory shift        p50 "
              f"{np.median(d/np.maximum(trav,1e-300)):.4e}")
        ljc = np.abs(ca[1][both])
        print(f"\n  {'✅ PASS' if np.percentile(ljc,99) < 1e-10 else '❌ FAIL'}"
              f" -- ln J along the pathline (p99 {np.percentile(ljc,99):.3e},"
              f" want < 1e-10)")


if __name__ == '__main__':
    main()
