"""Do the pathlines move when the field is made divergence-free?

Compares two full tracking runs that differ in ONE flag (--divergence-free),
matching particles by particle_id rather than by array position -- the arrays
can be ordered differently and can lose different particles.

⚠️ THE EXPECTATION, stated before measuring: the correction removes each
element's own tr A and leaves u(x_c) EXACTLY unchanged, so it constrains
VOLUME, not streamline spacing. The per-element measurement was a relative
trajectory shift of 8.2e-06 on D2 (DIVERGENCE_FREE_METHOD.md 6.3). So the
pathlines should be nearly identical.

⚠️ WHAT WOULD FALSIFY THAT: a shift that grows with distance travelled, or a
systematic drift in one direction (e.g. particles behind the tool spreading
more evenly). Both are reported below, so the data can contradict the
expectation rather than merely agreeing with it.

Usage:
    python3 compare_divfree_trajectories.py BASELINE_DIR DIVFREE_DIR
where each DIR is a run output directory containing
femuss_*/final_state.npz (and optionally damage_models.npz).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def load_run(d: Path):
    """positions keyed by particle_id, plus the damage table if present."""
    fs = sorted(d.glob("*/final_state.npz"))
    if not fs:
        raise SystemExit(f"no */final_state.npz under {d}")
    z = np.load(fs[0])
    pos = np.asarray(z["positions"], dtype=np.float64)
    pid = np.asarray(z["particle_ids"])
    eid = np.asarray(z["element_ids"])
    dmg = None
    for cand in (d / "damage_models.npz", *d.glob("*/damage_models.npz")):
        if cand.exists():
            dz = np.load(cand)
            key = next((k for k in dz.keys() if dz[k].ndim == 2), None)
            if key is not None:
                dmg = np.asarray(dz[key], dtype=np.float64)
            break
    return dict(pos=pos, pid=pid, eid=eid, dmg=dmg, src=fs[0])


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    A = load_run(Path(sys.argv[1]))
    B = load_run(Path(sys.argv[2]))
    print(f"baseline     {A['src']}")
    print(f"divfree      {B['src']}")
    print(f"particles    {len(A['pid']):,} vs {len(B['pid']):,}")

    # ⚠️ Match by particle_id. Comparing arrays positionally would silently
    # pair different particles if either run lost a different subset.
    ia = np.argsort(A["pid"]); ib = np.argsort(B["pid"])
    pa, pb = A["pid"][ia], B["pid"][ib]
    common, ca, cb = np.intersect1d(pa, pb, return_indices=True)
    xa = A["pos"][ia][ca]
    xb = B["pos"][ib][cb]
    ea = A["eid"][ia][ca]
    eb = B["eid"][ib][cb]
    print(f"matched      {len(common):,} by particle_id")

    alive = (ea >= 0) & (eb >= 0)
    print(f"alive in both {int(alive.sum()):,} "
          f"({100.0*alive.mean():.2f} %)")
    xa, xb = xa[alive], xb[alive]

    d = np.linalg.norm(xa - xb, axis=1)
    # distance travelled is unknown without the seed positions, so use the
    # spread of the cloud as the length scale -- a conservative denominator.
    scale = float(np.linalg.norm(xa.max(axis=0) - xa.min(axis=0)))
    print(f"\n── FINAL-POSITION DIFFERENCE ──")
    for q in (50, 90, 99, 100):
        v = d.max() if q == 100 else np.percentile(d, q)
        print(f"   p{q:<4} {v:.4e} m   ({v/scale*100:.3e} % of cloud extent)")
    print(f"   cloud extent {scale:.4e} m")

    # ⚠️ Does the shift GROW with how far the particle got? If the correction
    # merely perturbs locally, there is no trend; if it redistributes the flow,
    # downstream particles should differ more.
    r = np.linalg.norm(xa - xa.mean(axis=0), axis=1)
    if r.std() > 0 and d.std() > 0:
        print(f"\n   corr(|shift|, distance from cloud centre) = "
              f"{np.corrcoef(d, r)[0,1]:+.4f}")
    # and is there a systematic DIRECTION to the shift?
    mean_vec = (xb - xa).mean(axis=0)
    print(f"   mean shift vector {mean_vec} m "
          f"(|.| = {np.linalg.norm(mean_vec):.3e})")
    print(f"   rms shift         {float(np.sqrt((d**2).mean())):.3e} m")
    print(f"   -> a mean vector much smaller than the rms means the shift is")
    print(f"      SCATTER, not a bulk drift.")

    # element occupancy: did particles end in the same elements?
    same = int((ea[alive] == eb[alive]).sum())
    print(f"\n   ended in the SAME element: {same:,}/{int(alive.sum()):,} "
          f"({100.0*same/max(int(alive.sum()),1):.2f} %)")

    if A["dmg"] is not None and B["dmg"] is not None:
        print(f"\n── DAMAGE ACCUMULATORS ──")
        print(f"   shapes {A['dmg'].shape} vs {B['dmg'].shape}")
        n = min(A["dmg"].shape[0], B["dmg"].shape[0])
        m = min(A["dmg"].shape[1], B["dmg"].shape[1])
        names = ["ln_phi", "C", "t_res", "w_bond", "A_edot", "ln_J",
                 "tau_min", "pi1", "bu_min", "bu"][:m]
        for j, nm in enumerate(names):
            va, vb = A["dmg"][:n, j], B["dmg"][:n, j]
            fa = np.isfinite(va) & np.isfinite(vb)
            if not fa.any():
                continue
            print(f"   {nm:8s} baseline p50 {np.median(va[fa]):+.4e}   "
                  f"divfree p50 {np.median(vb[fa]):+.4e}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
