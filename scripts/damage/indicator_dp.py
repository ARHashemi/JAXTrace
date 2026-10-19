"""A.4 — contact-pressure differential Delta p across the tool.

    python3 indicator_dp.py --case <case>.gid [--vel-range 34 199] [--out dp.csv]

WHY THIS IS A SEPARATE SCRIPT AND NOT A KERNEL FLAG
---------------------------------------------------
Delta p is a PRESSURE-FIELD diagnostic, not a pathline integral. It compares the
mean pressure on the advancing side of the pin against the retreating side at one
instant; no particle is integrated and no element search is involved. Bolting it
into the RK4 kernel would add GPU state and a per-step branch to compute something
that depends only on the field, so it belongs here, run after the fact on data the
tracking run already needs.

See docs/damage/NEXT_STEPS_detachment_criterion.md Part A.4.

WHAT IT MEASURES
----------------
    B = P_max(theta=0 deg, FRONT)  -  P_min(theta=180 deg, REAR)

evaluated in an annulus around the pin axis, within the pin's own z-extent.

✅ VERIFIED against the full text 2026-10-19: Shi et al., Int. J. Mech. Sci. 241
(2023) 107969, Eq. (19) and Fig. 21 -- "void defects formed in the joint when the
value of B is greater than 15 MPa". A sound joint forms when B is LOWER than the
critical value. Shi states no second condition; B alone is the criterion.

⚠️ TWO THINGS AN EARLIER VERSION OF THIS SCRIPT GOT WRONG:
  * The inequality was inverted ("< 15 MPa for void formation").
  * The quantity was an ADVANCING-TO-RETREATING difference, splitting on dy.
    Eq. (19) is FRONT-TO-REAR along the travel direction, and uses the MAXIMUM
    in front and the MINIMUM in the rear rather than sector means. The two are
    perpendicular. Both are now reported, but only B may be compared to 15 MPa.

⚠️ The 15 MPa value is from THEIR alloy and tool. This script reports the MEASURED
value and the sign pattern across cases; it deliberately does not print a
pass/fail verdict against a number we cannot yet stand behind. Use it to RANK
cases first.

⚠️ SIGN AND MAGNITUDE SANITY. The 45-case survey measured roughly -6 .. -7.5 MPa
ahead of the tool and +0.4 .. +2.2 MPa behind it, with the pattern holding in 9 of
10 cases (DAMAGE_REPORT.md). An earlier set of figures quoting +1.70/-1.35 MPa was
NOT reproducible and was retracted -- do not use those numbers as a reference.

ADVANCING VS RETREATING
-----------------------
For a tool rotating with angular velocity omega about +z and travelling along +x,
the advancing side is the one where the surface velocity adds to the travel
direction. With omega_z < 0 (clockwise seen from +z, as in the PinShapes cases,
PIN_RPM is negative) that is the +y side. This script derives the side from the
SIGN OF PIN_RPM rather than assuming, and prints which side it chose.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
import sys

import numpy as np


def load_sequence(case: str, vel_range, field: str, n_workers: int):
    """Positions + the pressure sequence, via the project's THREADED loader.

    ⚠️ Uses jaxtrace.gpu.mesh_loader_timedep.load_field_sequences_from_pvtu --
    the same loader the tracking and damage runs use -- rather than opening each
    PVTU with a fresh vtkXMLPUnstructuredGridReader. An earlier version of this
    script did the latter and measured ~205 s PER SNAPSHOT on LUMI's Lustre at
    D2 scale (2.8M nodes), i.e. 9.5 h per case. The threaded loader parses the
    whole sequence concurrently (VTK releases the GIL), which is why the damage
    runs load 166 snapshots in ~24 min rather than ~9 h.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    from jaxtrace.gpu.mesh_loader_timedep import load_field_sequences_from_pvtu

    post = case if os.path.basename(case) == "post" else os.path.join(case, "post")
    if not os.path.isdir(post):
        post = case
    # Derive the stem from the files themselves: the folder name and the file
    # stem differ in these cases (D2.gid holds A2_*.pvtu).
    cands = sorted(glob.glob(os.path.join(post, "*.pvtu")))
    stems = {re.sub(r"_\d+\.pvtu$", "", os.path.basename(f)) for f in cands
             if re.search(r"_\d+\.pvtu$", os.path.basename(f))}
    if not stems:
        raise SystemExit(f"  no <stem>_<step>.pvtu under {post}")
    stem = sorted(stems)[0]
    steps = sorted(int(re.search(r"_(\d+)\.pvtu$", os.path.basename(f)).group(1))
                   for f in cands
                   if re.search(r"_(\d+)\.pvtu$", os.path.basename(f))
                   and os.path.basename(f).startswith(stem + "_"))
    lo, hi = (vel_range if vel_range else (steps[0], steps[-1]))
    pos, _conn, fields = load_field_sequences_from_pvtu(
        base_path=Path(post), file_pattern=stem + "_{timestep}.pvtu",
        timestep_range=(lo, hi), field_names=[field],
        verbose=True, n_workers=n_workers,
    )
    return np.asarray(pos, dtype=np.float64), np.asarray(fields[field]), (lo, hi)


def delta_p(pos, p, centre, r_pin, z_lo, z_hi, band, advancing_sign,
            sector_deg=30.0):
    """Shi et al. (2022) contact-pressure difference B = P_max - P_min.

    ⚠️ CORRECTED 2026-10-19 against the full text (Int. J. Mech. Sci. 241
    (2023) 107969, Eq. 19). The paper defines

        B = P_max - P_min,
        P_max = maximum pressure IN FRONT of the tool  (theta = 0 deg)
        P_min = minimum pressure IN THE REAR of the tool (theta = 180 deg)

    and reports that voids form when B > 15 MPa.

    ⚠️ That is a FRONT-TO-REAR difference along the travel direction. An earlier
    version of this function computed an ADVANCING-TO-RETREATING difference
    (splitting on dy, with points restricted to |dx| <= r_pin/2) -- a different
    quantity, perpendicular to Shi's, to which the 15 MPa threshold does NOT
    apply. Both are physically interesting, so both are returned, but only
    `B` may be compared against the published criterion.

    Returns (B, p_front, p_rear, n_used, dp_adv_ret).
    """
    dx = pos[:, 0] - centre[0]
    dy = pos[:, 1] - centre[1]
    r = np.hypot(dx, dy)
    # Annulus around the pin, within its z-extent: the contact region.
    sel = (r >= r_pin) & (r <= r_pin * (1.0 + band)) \
        & (pos[:, 2] >= z_lo) & (pos[:, 2] <= z_hi)
    if sel.sum() < 20:
        return np.nan, np.nan, np.nan, int(sel.sum()), np.nan

    # theta measured from +x (the travel direction): 0 deg is in FRONT.
    theta = np.degrees(np.arctan2(dy, dx))
    half = sector_deg * 0.5
    front = sel & (np.abs(theta) <= half)
    rear = sel & (np.abs(np.abs(theta) - 180.0) <= half)
    if front.sum() < 10 or rear.sum() < 10:
        return np.nan, np.nan, np.nan, int(sel.sum()), np.nan

    # ⚠️ MAX in front and MIN in the rear, per Eq. (19) -- not sector means. A
    # mean would understate B and make every case look safer than it is.
    p_front = float(p[front].max())
    p_rear = float(p[rear].min())

    # The advancing/retreating contrast, kept as a SEPARATE diagnostic.
    near = sel & (np.abs(dx) <= r_pin * 0.5)
    adv = near & (advancing_sign * dy > 0)
    ret = near & (advancing_sign * dy < 0)
    dp_ar = (float(p[adv].mean()) - float(p[ret].mean())
             if adv.sum() >= 10 and ret.sum() >= 10 else np.nan)

    return p_front - p_rear, p_front, p_rear, int(sel.sum()), dp_ar


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True,
                    help="<case>.gid folder (or its post/ subfolder)")
    ap.add_argument("--vel-range", nargs=2, type=int, default=None,
                    metavar=("START", "END"))
    ap.add_argument("--pin-center", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "Z"),
                    help="pin axis point [m]; default = mesh centre in x,y")
    ap.add_argument("--r-pin", type=float, default=None,
                    help="pin radius [m]; default = 3e-3")
    ap.add_argument("--pin-rpm", type=float, default=None,
                    help="signed rotation rate; its SIGN picks the advancing "
                         "side. Default: read from the case data, else -600.")
    ap.add_argument("--band", type=float, default=0.6,
                    help="annulus thickness as a fraction of r_pin (0.6)")
    ap.add_argument("--pressure-field", default="Pressure")
    ap.add_argument("--out", default=None, help="write a per-step CSV here")
    ap.add_argument("--workers", type=int, default=16,
                    help="threads for the PVTU loader (default 16)")
    ap.add_argument("--n-snapshots", type=int, default=0,
                    help="Evaluate only this many snapshots, evenly spaced "
                         "(0 = all, the default). ⚠️ This does NOT save IO: the "
                         "threaded loader reads the whole range in one pass. It "
                         "only thins the reported series.")
    args = ap.parse_args()

    pos, pseq, (lo, hi) = load_sequence(
        args.case, tuple(args.vel_range) if args.vel_range else None,
        args.pressure_field, args.workers)
    n_all = pseq.shape[0]
    # ⚠️ The threaded loader reads the WHOLE range in one pass, so subsampling
    # no longer saves IO -- it is kept only to thin the reported series.
    sel_idx = (np.unique(np.linspace(0, n_all - 1, args.n_snapshots).astype(int))
               if args.n_snapshots and 0 < args.n_snapshots < n_all
               else np.arange(n_all))
    print(f"  {n_all} snapshots loaded (steps {lo}..{hi}), "
          f"evaluating {len(sel_idx)}")

    rpm = args.pin_rpm
    if rpm is None:
        rpm = -600.0
        print(f"  [warn] --pin-rpm not given; assuming {rpm:g}")
    # ⚠️ CORRECTED 2026-10-19. The advancing side is where the tool's SURFACE
    # motion ADDS to the travel direction. With omega = (0,0,w) and travel along
    # +x, the surface velocity at (0,y) has v_x = -w*y. So for w < 0 (clockwise
    # seen from +z, our case) v_x > 0 on the +y side: ADVANCING IS +y.
    # The previous line had this inverted.
    # ⚠️ This affects ONLY the secondary advancing-retreating diagnostic. B
    # itself is a front/rear quantity and does not use advancing_sign at all,
    # so every B result already reported stands unchanged.
    advancing_sign = +1.0 if rpm < 0 else -1.0
    print(f"  pin_rpm {rpm:+g} -> advancing side is "
          f"{'+y' if advancing_sign > 0 else '-y'}")

    centre = (np.array(args.pin_center, dtype=np.float64)
              if args.pin_center is not None
              else np.array([pos[:, 0].mean(), pos[:, 1].mean(), 0.0]))
    r_pin = args.r_pin if args.r_pin is not None else 3.0e-3
    z_lo, z_hi = pos[:, 2].min(), pos[:, 2].max()
    print(f"  centre ({centre[0]:.4g}, {centre[1]:.4g}) m   r_pin {r_pin:.4g} m")

    rows = []
    for j, i in enumerate(sel_idx, 1):
        p_i = np.asarray(pseq[i], dtype=np.float64).reshape(-1)
        d, pa, pr, n, dar = delta_p(pos, p_i, centre, r_pin, z_lo, z_hi,
                                    args.band, advancing_sign)
        rows.append((f"step_{lo + int(i)}", d, pa, pr, n, dar))
        if j % 5 == 0 or j == len(sel_idx):
            print(f"    [{j}/{len(sel_idx)}] B = {d * 1e-6:+.2f} MPa", flush=True)

    vals = np.array([r[1] for r in rows], dtype=np.float64)
    vals_ar = np.array([r[5] for r in rows], dtype=np.float64)
    ok = np.isfinite(vals)
    if not ok.any():
        print("  no snapshot had enough points in the annulus — "
              "check --pin-center / --r-pin / --band")
        return 1

    MPa = 1.0e-6
    print()
    print(f"  Delta p over {int(ok.sum())} snapshots [MPa]:")
    print(f"    median {np.median(vals[ok]) * MPa:+.3f}   "
          f"mean {vals[ok].mean() * MPa:+.3f}   "
          f"min {vals[ok].min() * MPa:+.3f}   max {vals[ok].max() * MPa:+.3f}")
    pa = np.array([r[2] for r in rows])[ok].mean() * MPa
    pr = np.array([r[3] for r in rows])[ok].mean() * MPa
    print(f"    mean p advancing {pa:+.3f}   retreating {pr:+.3f}")
    frac_pos = float((vals[ok] > 0).mean())
    print(f"    sign consistency: {100.0 * frac_pos:.0f} % of snapshots positive")
    _ok = np.isfinite(vals_ar)
    if _ok.any():
        print(f"    (separate diagnostic) advancing-retreating dp: median "
              f"{np.median(vals_ar[_ok]) * MPa:+.3f} MPa")
    print()
    # ✅ Threshold VERIFIED against the full text 2026-10-19: Shi et al.,
    # Int. J. Mech. Sci. 241 (2023) 107969, Eq. (19) and Fig. 21 -- "void
    # defects formed in the joint when the value of B is greater than 15 MPa".
    _med = float(np.median(vals[ok])) * MPa
    print(f"  Shi et al. (2022) criterion: voids form when B > 15 MPa "
          f"(Eq. 19, Fig. 21).")
    print(f"    median B = {_med:+.2f} MPa  ->  "
          f"{'EXCEEDS' if _med > 15.0 else 'below'} the threshold")
    print("  ⚠️  The threshold was established for THEIR alloy and tool; treat "
          "the\n      ranking across our cases as the primary result and the "
          "absolute\n      pass/fail as indicative.")

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("file,B_Pa,p_front_Pa,p_rear_Pa,n_points,"
                     "dp_adv_ret_Pa\n")
            for nm, d, a, r, n, dar in rows:
                fh.write(f"{nm},{d:.6e},{a:.6e},{r:.6e},{n},{dar:.6e}\n")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
