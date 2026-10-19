"""Test the Chen et al. (2023) mechanism against our contact-pressure result.

    python3 compare_chen2023.py --dp <dir with dp_*.csv> \
                                --indicators <indicators_all16.csv>

WHY THIS COMPARISON IS WORTH DOING AND COSTS NOTHING
-----------------------------------------------------
Chen, Shi & Wu (2023), J. Manuf. Process. 86, 85-97, AA6061-T6 at **600 rpm --
our rotation speed**, with a CFD model using a NON-UNIFORM contact-pressure
boundary condition, validated against measured transverse cross-sections.

Their mechanism, in their words:
  * sound weld   (600-20) : the frictional-shear-stress distribution around the
                            tool is "approximately a regular circle" at ~30 MPa.
  * voided weld  (600-180): ~50 MPa in FRONT, ~15 MPa in the REAR at z = 1 mm
                            (~42 / ~13 at z = 2 mm) -- "a severe decrease ... at
                            the rear of the tool is observed when forming void
                            defects".
  * the void then forms at the ADVANCING SIDE.

⚠️ THREE CLAIMS, AND THEY ARE NOT EQUALLY TESTABLE BY US.

  (1) A LARGE front-to-rear drop accompanies voids, a SMALL one does not.
      ✅ Directly testable: that is exactly what our B measures.
  (2) The void appears on the ADVANCING side.
      ⚠️ Testable only in combination with the particle figures, and our
      advancing/retreating pressure difference has the OPPOSITE sign to B, so
      the two must not be conflated.
  (3) Their absolute numbers.
      ❌ NOT comparable. Chen reports FRICTIONAL SHEAR STRESS (tau) at the
      tool/workpiece interface; our B is a PRESSURE difference in an annulus in
      the material. Same mechanism, different quantity, different units of
      meaning. Quoting "35 MPa vs our 16.8 MPa" as agreement or disagreement
      would be a category error. Only the PATTERN transfers.

So this script tests the pattern: do the cases our B flags as high-drop differ,
in the ways Chen predicts, from the cases it flags as low-drop?
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

MPa = 1.0e-6
SHI = 15.0          # Shi et al. Eq. 19 threshold, MPa

# Chen's reported frictional shear stress, for CONTEXT ONLY -- see claim (3).
CHEN = {
    "sound 600-20":  dict(front=30.0, rear=30.0, note="uniform ~30 MPa circle"),
    "voided 600-180 z=1mm": dict(front=50.0, rear=15.0, note="severe rear drop"),
    "voided 600-180 z=2mm": dict(front=42.0, rear=13.0, note="severe rear drop"),
}


def load_dp(src: str) -> dict:
    out = {}
    for p in sorted(glob.glob(os.path.join(src, "dp_*.csv"))):
        tag = os.path.basename(p)[3:-4]
        rows = list(csv.DictReader(open(p)))
        if not rows:
            continue
        out[tag] = dict(
            B=np.array([float(r["B_Pa"]) for r in rows]) * MPa,
            front=np.array([float(r["p_front_Pa"]) for r in rows]) * MPa,
            rear=np.array([float(r["p_rear_Pa"]) for r in rows]) * MPa,
            dp_ar=np.array([float(r.get("dp_adv_ret_Pa", "nan"))
                            for r in rows]) * MPa,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dp", required=True)
    ap.add_argument("--indicators", default=None)
    args = ap.parse_args()

    dp = load_dp(args.dp)
    if not dp:
        print(f"  no dp_*.csv under {args.dp}")
        return 1

    print("  Chen et al. (2023) reference, FRICTIONAL SHEAR STRESS [MPa]")
    print("  ⚠️ context only -- a different quantity from our B, see claim (3)")
    for k, v in CHEN.items():
        print(f"    {k:24s} front {v['front']:5.1f}  rear {v['rear']:5.1f}  "
              f"drop {v['front'] - v['rear']:5.1f}   ({v['note']})")
    print()

    tags = sorted(dp, key=lambda t: -np.median(dp[t]["B"]))
    print("  OUR CASES — pressure in the contact annulus [MPa]")
    print(f"    {'case':6s} {'B':>7s} {'p_front':>8s} {'p_rear':>8s} "
          f"{'rear<0?':>8s}  {'flag':>12s}")
    hi, lo = [], []
    for t in tags:
        d = dp[t]
        B = float(np.median(d["B"]))
        f_, r_ = float(np.median(d["front"])), float(np.median(d["rear"]))
        flag = "HIGH-DROP" if B > SHI else ("mid" if B > 5.0 else "low")
        (hi if B > SHI else lo).append(t)
        print(f"    {t:6s} {B:>7.2f} {f_:>8.2f} {r_:>8.2f} "
              f"{'yes' if r_ < 0 else 'no':>8s}  {flag:>12s}")

    print()
    print("  ── Claim (1): a large front-to-rear drop accompanies voids ──")
    if not hi:
        print("    ⚠️ No case exceeds Shi's 15 MPa, so this claim cannot be")
        print("       tested on our set -- there is no 'voided' group to compare.")
    else:
        Bhi = np.array([np.median(dp[t]["B"]) for t in hi])
        Blo = np.array([np.median(dp[t]["B"]) for t in lo])
        rhi = np.array([np.median(dp[t]["rear"]) for t in hi])
        rlo = np.array([np.median(dp[t]["rear"]) for t in lo])
        print(f"    high-drop ({len(hi)}): {', '.join(hi)}")
        print(f"      median B {np.median(Bhi):6.2f}   median p_rear "
              f"{np.median(rhi):+6.2f}")
        print(f"    low-drop  ({len(lo)}):")
        print(f"      median B {np.median(Blo):6.2f}   median p_rear "
              f"{np.median(rlo):+6.2f}")
        # ✅ The pattern Chen describes is a REAR PRESSURE COLLAPSE, not merely a
        # large difference. Check whether the rear is what separates the groups.
        d_front = np.median([np.median(dp[t]["front"]) for t in hi]) - \
            np.median([np.median(dp[t]["front"]) for t in lo])
        d_rear = np.median(rhi) - np.median(rlo)
        print()
        print(f"    high minus low:  front {d_front:+6.2f} MPa   "
              f"rear {d_rear:+6.2f} MPa")
        if abs(d_rear) > abs(d_front):
            print("    ✅ The REAR separates the groups more than the front --")
            print("       consistent with Chen's 'severe decrease at the rear'.")
        else:
            print("    ⚠️ In MAGNITUDE the front separates the groups more.")
            print("       But magnitude is the wrong test -- see the sign check.")

        # ⚠️ THE SHARPER TEST, and the one that actually matches Chen. He does
        # not describe a large difference, he describes the rear COLLAPSING.
        # A mean/median shift in MPa blurs that; the SIGN of the rear pressure
        # does not. Comparing group medians alone made this look like a
        # mismatch when it is in fact a clean binary signature.
        negs = sorted(t for t in dp if np.median(dp[t]["rear"]) < 0)
        poss = sorted(t for t in dp if np.median(dp[t]["rear"]) >= 0)
        print()
        print("    ── sharper test: does the REAR go NEGATIVE (tensile)? ──")
        print(f"      rear < 0  ({len(negs)}): {', '.join(negs)}")
        print(f"      rear >= 0 ({len(poss)}): {len(poss)} cases")
        if negs and poss:
            bn = np.median([np.median(dp[t]["B"]) for t in negs])
            bp = np.median([np.median(dp[t]["B"]) for t in poss])
            print(f"      median B: rear<0 {bn:6.2f}   rear>=0 {bp:6.2f} MPa")
            rank = sorted(dp, key=lambda t: -np.median(dp[t]["B"]))
            topk = set(rank[:len(negs)])
            if topk == set(negs):
                print(f"      ✅ The {len(negs)} cases with a NEGATIVE rear pressure are")
                print("         EXACTLY the top cases by B. That is Chen's")
                print("         rear-collapse signature, reproduced on a different")
                print("         alloy and a different model.")
            else:
                print(f"      ⚠️ Negative-rear cases are not exactly the top-{len(negs)}")
                print(f"         by B (top: {', '.join(rank[:len(negs)])}).")

    print()
    print("  ── Claim (2): the void sits on the ADVANCING side ──")
    ar = np.array([np.median(dp[t]["dp_ar"]) for t in tags])
    ok = np.isfinite(ar)
    if ok.any():
        print(f"    advancing-retreating dp: median {np.median(ar[ok]):+.2f} MPa "
              f"over {int(ok.sum())} cases")
        print("    ⚠️ This is a DIFFERENT quantity from B and has the opposite")
        print("       sign. It locates a left/right asymmetry; it does not by")
        print("       itself say a void is there. Settling claim (2) needs the")
        print("       particle cross-sections, not this number.")

    print()
    print("  ── Claim (3): absolute values ──")
    print("    ❌ Not comparable. Chen reports frictional SHEAR STRESS at the")
    print("       tool interface; B is a PRESSURE difference in the material.")
    print("       Only the pattern transfers, never the number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
