"""Is the divergence-free fit limited by the MODEL ORDER or by the CONSTRAINT?

The user's question, made precise. Two distinct ways a fit can go wrong:

  (a) MODEL TOO LOW ORDER. The target field needs more degrees of freedom
      than u = A x + b has. ⚠️ On a single tet this CANNOT happen: the FOM
      velocity is P1, so 4 nodal values define an exactly affine field and
      BDM1's 12 DOF contain it. Measured below as check [1] -> expect ~1e-16.

  (b) CONSTRAINT INCOMPATIBLE WITH THE DATA. div u = 0 removes one DOF per
      element, and the patch boundary is PINNED to the P1 normal trace. If
      the data's own divergence is large, satisfying the constraint forces a
      change in u -- and that change is what 'preservation' measures. This
      is the real mechanism, and it is LOCAL: it depends on how much
      divergence sits in the patch relative to how much the interior DOFs
      can absorb.

Check [2] isolates (b) with NO solver: for each element, the minimum change
to its affine map needed to make it divergence-free is exactly

    min ||A' - A||_F  s.t. tr(A') = 0   ->   A' = A - (tr A / 3) I
    ||A' - A||_F = |tr A| / sqrt(3)

so the IRREDUCIBLE relative change is  |div u| / (sqrt(3) * ||A||_F).
That is a LOWER BOUND on preservation error, computed from the data alone.
If it already differs by orders of magnitude between cylA and D2, the gap is
a property of the FIELD, not of our solver or its regularisation.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '.')
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes


def load(case, pattern, ts, field='Displacement'):
    """⚠️ Loads EXACTLY as build_divfree_field.py does, so this diagnostic
    sees the same field the solver sees. Two traps, both already paid for
    elsewhere in this project:
      - timestep_range is an INCLUSIVE (start, end) PAIR, not a list.
      - deduplicate_nodes REMAPS the velocity itself when you pass it in, and
        returns a 4-tuple (nodes, conn, n_dup, velocity). Remapping by hand
        from its index map silently permutes the field.
    """
    from jaxtrace.gpu.mesh_loader_timedep import load_field_sequences_from_pvtu
    nodes, conn, seq = load_field_sequences_from_pvtu(
        Path(case), pattern, (ts, ts), [field], verbose=False, n_workers=8)
    o = deduplicate_nodes(nodes, conn, seq[field], verbose=False)
    return o[0], o[1].astype(np.int64), o[3][0].astype(np.float64)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--pattern', required=True)
    ap.add_argument('--timestep', type=int, required=True)
    ap.add_argument('--sample', type=int, default=200000)
    a = ap.parse_args()

    nodes, conn, vel = load(a.case, a.pattern, a.timestep)
    print(f"mesh {len(nodes):,} nodes  {len(conn):,} elements", flush=True)

    rng = np.random.default_rng(0)
    n = min(a.sample, len(conn))
    sel = rng.choice(len(conn), n, replace=False)
    p = nodes[conn[sel]]
    uv = vel[conn[sel]]

    # Exact affine map per tet: solve [x y z 1] c = u  (4x4).
    X = np.concatenate([p, np.ones((n, 4, 1))], axis=2)
    detX = np.linalg.det(X)
    ok = np.abs(detX) > 0
    X, uv, sel = X[ok], uv[ok], sel[ok]
    C = np.linalg.solve(X, uv)                 # (m,4,3): rows = dN coeffs
    A = C[:, :3, :].transpose(0, 2, 1)         # du_i/dx_j
    res = np.einsum('mij,mjk->mik', X, C) - uv

    nA = np.sqrt(np.sum(A * A, axis=(1, 2)))
    uvs = np.maximum(np.abs(uv).max(axis=(1, 2)), 1e-300)
    print(f"\n[1] MODEL ORDER -- affine fit residual on each tet, /|u|max")
    print(f"    max {np.max(np.abs(res).max(axis=(1,2))/uvs):.3e}"
          f"   median {np.median(np.abs(res).max(axis=(1,2))/uvs):.3e}")
    print("    -> the P1 data IS exactly affine per element; u = A x + b is")
    print("       NOT too low order. Hypothesis (a) is refuted by construction.")

    div = np.trace(A, axis1=1, axis2=2)
    rel = np.abs(div) / np.maximum(np.sqrt(3.0) * nA, 1e-300)
    q = [50, 75, 90, 95, 99, 99.9]
    print(f"\n[2] CONSTRAINT COST -- irreducible |div|/(sqrt3 |A|) per element")
    print("    (a LOWER BOUND on any divergence-free projection's change)")
    for qq in q:
        print(f"      p{qq:<5} {np.percentile(rel, qq):.4e}")
    print(f"      mean  {rel.mean():.4e}   max {rel.max():.4e}")
    for t in (0.01, 0.1, 0.5, 1.0):
        print(f"    fraction needing >{t*100:5.1f}% change: "
              f"{100.0*np.mean(rel > t):6.3f}%")
    print(f"\n    |div| p50 {np.percentile(np.abs(div),50):.4e}"
          f"  p99 {np.percentile(np.abs(div),99):.4e}")
    print(f"    |A|_F p50 {np.percentile(nA,50):.4e}"
          f"  p99 {np.percentile(nA,99):.4e}")

    # ── [3] The quantity the solver actually REPORTS ────────────────────────
    # ⚠️ Checks [2] and the solver's 'preservation' are DIFFERENT quantities:
    #   [2]          ||A' - A||_F / ||A||_F   -- change in the GRADIENT
    #   preservation ||u_new - u_P1|| / ||u_P1|| at the CENTROID -- the VALUE
    # The centroid value can be preserved exactly while the gradient changes a
    # lot, because b is free to absorb it: u(xc) = A xc + b, so setting
    # b' = b + (A - A') xc leaves u(xc) untouched. Hence the solver can and
    # does beat check [2]'s number -- it is not a bound on preservation.
    #
    # So the true lower bound on PRESERVATION from the pointwise constraint
    # alone is ZERO: per element, divergence-free + exact centroid value is
    # always satisfiable (11 free DOF, 1 + 3 equations). Any preservation
    # error we measure therefore comes from COUPLING -- face continuity across
    # elements and the pinned patch boundary -- never from the element model.
    uc = np.einsum('mij,mj->mi', A, p.mean(axis=1)) + C[:, 3, :]
    u_p1c = uv.mean(axis=1)
    chk = np.linalg.norm(uc - u_p1c, axis=1) / np.maximum(
        np.linalg.norm(u_p1c, axis=1), 1e-300)
    print(f"\n[3] sanity: affine map reproduces the centroid value to "
          f"{np.median(chk):.3e} (median)")
    print("    -> per element, div-free AND exact centroid is ALWAYS reachable")
    print("       (11 free DOF vs 4 equations). Preservation error comes from")
    print("       INTER-ELEMENT COUPLING + the pinned boundary, not the model.")

    # ── [4] Is the required correction UNIFORM or does it need adaptivity? ──
    # The user's second question. If the per-element repair cost spans orders
    # of magnitude, one global Tikhonov weight and one global row scaling are
    # the wrong instrument -- a uniform knob on a non-uniform problem.
    lo, hi = np.percentile(rel, [50, 99])
    print(f"\n[4] NON-UNIFORMITY of the required repair: p99/p50 = "
          f"{hi/max(lo,1e-300):.1f}x")
    print(f"    max/p50 = {rel.max()/max(lo,1e-300):.1f}x")
    print("    -> a SINGLE global reg weight is applied across this spread.")


if __name__ == '__main__':
    main()
