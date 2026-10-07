"""Compare the Euler (Phase 3) and RK4 (Phase 4) damage accumulators.

    python3 compare_euler_rk4.py --euler <phase3_dir> --rk4 <phase4_dir>

⚠️ The two runs differ in TWO ways at once -- quadrature order AND travel distance
(10 mm -> 30 mm) -- so a raw difference in lnPhi conflates them. What IS comparable:

  * rank correlation between the two models WITHIN each run (an implementation
    self-check that should hold regardless of order or distance);
  * the BYPASS fraction, which should drop sharply with 3x travel if the Phase 3
    diagnosis (O34) was right;
  * the regional contrast (stir zone vs bypass), which is the quantity the longer
    run was meant to recover.

To isolate the quadrature effect alone you need two runs at the SAME travel
distance differing only in --damage-order; that is a separate, cheaper experiment.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

LN_FAIL = float(np.log(1.0 / 1.0e-4))        # 9.21


def load(run_dir: str) -> dict:
    out = {}
    for c in sorted(glob.glob(os.path.join(run_dir, "*", "damage_models.npz"))):
        tag = os.path.basename(os.path.dirname(c))
        z = np.load(c, allow_pickle=True)
        dm = z["damage"]
        out[tag] = {
            "pos": z["positions"],
            "ln": dm[:, 0] if dm.ndim == 2 else dm.reshape(-1),
            "C": dm[:, 1] if (dm.ndim == 2 and dm.shape[1] > 1) else None,
            "n_steps": int(z["n_steps"]),
            "dt": float(z["dt"]),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--euler", required=True, help="Phase 3 run directory")
    ap.add_argument("--rk4", required=True, help="Phase 4 run directory")
    args = ap.parse_args()

    E, R = load(args.euler), load(args.rk4)
    common = sorted(set(E) & set(R))
    if not common:
        print(f"  no cases in common\n    euler: {len(E)}\n    rk4:   {len(R)}")
        return 1

    print(f"  {len(common)} cases in common "
          f"(euler {len(E)}, rk4 {len(R)})\n")
    print("%-30s %13s %13s | %9s %9s | %8s" % (
        "case", "steps E / R", "bypass E / R", "stir_E", "stir_R", "rho"))
    print("-" * 96)

    drops = []
    for t in common:
        e, r = E[t], R[t]
        be = float((e["ln"] < 1.0).mean())
        br = float((r["ln"] < 1.0).mean())
        drops.append(be - br)

        def stir(d):
            p = d["pos"]
            rad = np.hypot(p[:, 0], p[:, 1]) * 1e3
            z = p[:, 2] * 1e3
            thick = z.max() - z.min()
            m = (rad < 7.0) & (z < z.max() - 0.20 * thick)
            return float(np.median(d["ln"][m])) if m.any() else np.nan

        rho = np.nan
        if r["C"] is not None:
            from scipy.stats import spearmanr
            rho = float(spearmanr(r["ln"], r["C"]).statistic)

        print("%-30s %6d/%-6d %6.1f%%/%-6.1f%% | %9.3f %9.3f | %8.4f" % (
            t, e["n_steps"], r["n_steps"], be * 100, br * 100,
            stir(e), stir(r), rho))

    # ---- controlled pair: same travel & particles, order 1 vs order 4 ----
    # When the two runs DO have the same n_steps, the quadrature effect is
    # isolated and a direct per-particle comparison is meaningful.
    same = [t for t in common if E[t]["n_steps"] == R[t]["n_steps"]]
    if same:
        print()
        print("  ✅ %d case(s) with IDENTICAL n_steps -> quadrature effect isolated:"
              % len(same))
        for t in same:
            a, b = E[t]["ln"], R[t]["ln"]
            n = min(a.size, b.size)
            a, b = a[:n], b[:n]
            rel = np.abs(b - a) / np.maximum(np.abs(a), 1e-12)
            print("     %-28s med %.6f -> %.6f  (%+.2f%%)  p99 rel.diff %.2f%%"
                  % (t, np.median(a), np.median(b),
                     100 * (np.median(b) / max(np.median(a), 1e-12) - 1),
                     100 * np.percentile(rel, 99)))
        print("     ⚠️ Order 4 samples the step MIDPOINT; order 1 samples its START.")
        print("        A systematic shift here is the k1 bias, not noise.")

    print()
    print("  mean bypass reduction: %+.1f percentage points" % (np.mean(drops) * 100))
    print("  ⚠️ A LARGE reduction confirms O34 (particles were simply not travelling")
    print("     far enough). A SMALL one means the bypass population is structural")
    print("     and the seed box, not the duration, is what needs changing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
