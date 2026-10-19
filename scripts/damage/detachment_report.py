"""Recompute the detachment statistics from a damage npz, ALIVE particles only.

    python3 detachment_report.py <damage_models.npz> [--thresholds 0.1,0.2,...]

WHY THIS EXISTS
---------------
⚠️ The in-run reporting for the detachment columns initially pooled ALL
particles, including the ~1.1 % that leave the domain before the last step. An
escapee's accumulators are frozen mid-trajectory, so its det F measures how far
it got, not the scheme's incompressibility. Measured on synthetic data matched
to the D2 loss profile, pooling inflates max|det F - 1| by ~135x (6.6e-05 ->
9.0e-03) -- the difference between passing the step-1 gate with four orders of
headroom and appearing to fail it.

run_tracking.py now filters by `alive` itself, but a run already in flight does
not have that fix. The npz carries every column plus element_ids, so the correct
numbers can always be recovered after the fact. This script is that recovery,
and it is also the right place to do the threshold sweep.

✅ ALIVE is element_ids >= 0, exactly as run_tracking.py defines it.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    cols = [str(c) for c in d["columns"]]
    dmg = d["damage"]
    if "ln_J" not in cols or "tau_min" not in cols:
        print(f"  {args.npz}: no detachment columns (has {cols})")
        return 1
    i_lnJ, i_tau = cols.index("ln_J"), cols.index("tau_min")

    # ⚠️ Same definition as run_tracking.py: a negative element id means the
    # particle is no longer hosted, i.e. it escaped.
    if "element_ids" in d:
        alive = np.asarray(d["element_ids"]) >= 0
    else:
        print("  ⚠️  no element_ids in the archive — cannot identify escapees, "
              "reporting POOLED statistics which overstate |det F - 1|")
        alive = np.ones(len(dmg), dtype=bool)

    n_all, n_al = len(dmg), int(alive.sum())
    print(f"  {args.npz}")
    print(f"  columns : {cols}")
    print(f"  alive   : {n_al:,}/{n_all:,} ({100.0*n_al/n_all:.2f} %)   "
          f"escaped {n_all-n_al:,}")
    print()

    for label, sel in (("ALIVE only (correct)", alive),
                       ("pooled (for contrast)", np.ones(n_all, bool))):
        lnJ = dmg[sel, i_lnJ].astype(np.float64)
        tau = dmg[sel, i_tau].astype(np.float64)
        detF = np.exp(lnJ)
        dev = np.abs(detF - 1.0)
        print(f"  ── {label}  (n={int(sel.sum()):,}) ──")
        print(f"     det F     median {np.median(detF):.8f}  "
              f"min {detF.min():.8f}  max {detF.max():.8f}")
        print(f"     |detF-1|  median {np.median(dev):.3e}  "
              f"p99 {np.percentile(dev,99):.3e}  max {dev.max():.3e}")
        fin = np.isfinite(tau)
        if fin.any():
            t = tau[fin]
            print(f"     Pi_1 min  median {np.median(t):.6f}  "
                  f"min {t.min():.6f}  max {t.max():.6f}  "
                  f"p01 {np.percentile(t,1):.6f}  "
                  f"({int(fin.sum()):,} finite)")
            # ⚠️ The SPREAD is the question step 3 depends on. A criterion whose
            # group varies by a few percent has no discriminating power.
            span = (t.max() - t.min()) / max(abs(np.median(t)), 1e-30)
            print(f"     spread    {100.0*span:.2f} % of the median"
                  f"   {'✅ usable' if span > 0.15 else '⚠️  NARROW — may not discriminate'}")
            # Did the clip at 10.0 bind anywhere?
            n_clip = int((t >= 9.999).sum())
            print(f"     at clip   {n_clip:,} particles at tau_ratio >= 9.999"
                  f"   {'⚠️  CLIP BINDS' if n_clip else '✅ clip never binds'}")
        print()

    lnJ = dmg[alive, i_lnJ].astype(np.float64)
    dev = np.abs(np.exp(lnJ) - 1.0)
    med = float(np.median(dev))
    print("  ── step-1 gate verdict (alive only) ──")
    if med > 1.0e-2:
        print(f"     ⚠️  median |det F - 1| = {med:.3e} > 1e-2: the discrete "
              f"field is NOT incompressible enough to attribute a density "
              f"deficit to detachment.")
    else:
        print(f"     ✅ median |det F - 1| = {med:.3e}. A void fraction of "
              f"interest is O(1e-2..1e-1), so the floor is "
              f"{1.0e-2/max(med,1e-30):.0f}x below it — any detachment signal "
              f"above ~{10*med:.1e} is resolvable, not numerical noise.")
    print()

    tau = dmg[alive, i_tau].astype(np.float64)
    tau = tau[np.isfinite(tau)]
    print("  ── threshold sweep (alive only) — free, from the recorded minimum ──")
    print(f"     {'Pi_1_crit':>10s}  {'detached':>10s}  {'count':>10s}")
    for thr in [float(x) for x in args.thresholds.split(",")]:
        f = float((tau < thr).mean())
        print(f"     {thr:>10.2f}  {100.0*f:>9.3f} %  {int(f*len(tau)):>10,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
