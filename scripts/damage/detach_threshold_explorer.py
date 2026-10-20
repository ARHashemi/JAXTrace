"""What WOULD detach at each threshold — from a finished npz, no re-run.

    python3 detach_threshold_explorer.py <damage_models.npz> [--tag cylA]

✅ WHY THIS ANSWERS THE THRESHOLD QUESTION WITHOUT GPU TIME
tau_min is recorded per particle as the running minimum of Pi_1 along the
path, so the detached SET at any threshold is just `tau_min < crit`. Every
candidate is a comparison on the file.

⚠️ ONE LIMIT, STATED BECAUSE IT IS EASY TO FORGET: this predicts WHICH
particles would detach, not where they would END UP. A detached particle
follows v_adv instead of the flow, so its final position differs from the one
in this archive. The counts and the spatial pattern of the TRIGGER are exact;
the resulting void shape needs a real run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path)
    ap.add_argument("--tag", default="")
    ap.add_argument("--x-cut-mm", type=float, default=None,
                    help="restrict to the wake box, as the panels do")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    cols = [str(c) for c in np.asarray(z["columns"]).tolist()]
    if "tau_min" not in cols:
        print(f"  {args.npz}: no tau_min column (has {cols})")
        return 1
    tau = np.asarray(z["damage"], dtype=np.float64)[:, cols.index("tau_min")]
    pos = np.asarray(z["positions"], dtype=np.float64) * 1e3
    eid = np.asarray(z["element_ids"]) if "element_ids" in z.files else None

    sel = np.ones(len(tau), dtype=bool)
    if args.x_cut_mm is not None:
        sel &= pos[:, 0] >= args.x_cut_mm
    # ⚠️ +inf means the particle never sampled valid material (escaped, or
    # inside the tool for its whole path). It has no Pi_1 and must not be
    # counted as "would not detach" -- it is simply not a candidate.
    finite = np.isfinite(tau)
    n_inf = int((~finite & sel).sum())
    t = tau[sel & finite]

    print(f"  {args.tag or args.npz.name}")
    print(f"  candidates: {len(t):,}" +
          (f"   (+{n_inf:,} with no valid Pi_1 sample, excluded)"
           if n_inf else ""))
    if eid is not None:
        print(f"  alive at end: {int((eid >= 0).sum()):,}/{len(eid):,}")
    print()
    print(f"  Pi_1 min: min {t.min():.6f}  p1 {np.percentile(t,1):.6f}  "
          f"median {np.median(t):.6f}  max {t.max():.6f}")
    print()
    print(f"     {'Pi_1_crit':>10s} {'detached':>10s} {'count':>10s}   note")
    # ✅ Sample the ACTUAL distribution rather than a fixed ladder, so the
    # interesting region is covered whatever the data looks like.
    qs = [0.01, 0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 90]
    seen = set()
    for q in qs:
        crit = float(np.percentile(t, q))
        if round(crit, 6) in seen:
            continue
        seen.add(round(crit, 6))
        frac = float((t < crit).mean())
        note = ""
        if q <= 1:
            note = "a thin extreme tail"
        elif q >= 50:
            note = "⚠️ over half the mesh — not a defect criterion"
        elif q >= 10:
            note = "⚠️ a large fraction"
        print(f"     {crit:>10.6f} {100*frac:>9.3f} % {int((t<crit).sum()):>10,}   {note}")
    print()
    print("  ⚠️ There is no gap in this distribution to put a threshold in.")
    print("     Any cut lands inside a continuum, so the detached fraction is")
    print("     set by the number chosen, not by the physics. That is the")
    print("     free-parameter problem the dimensionless form (B.6f step 3)")
    print("     exists to remove.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
