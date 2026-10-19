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
    Delta p = < p >_advancing  -  < p >_retreating

evaluated in an annulus around the pin axis, within the pin's own z-extent. Shi et
al. (2022) report Delta p < 15 MPa as a void-formation threshold.

⚠️ THE THRESHOLD IS NOT VERIFIED HERE. We have not read Shi et al. (2022), and the
15 MPa value is alloy- and geometry-specific. This script reports the MEASURED
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
import sys

import numpy as np


def find_pvtu(case: str, vel_range: tuple[int, int] | None) -> list[str]:
    """PVTU files for the case, restricted to vel_range when given."""
    post = case if os.path.basename(case) == "post" else os.path.join(case, "post")
    if not os.path.isdir(post):
        post = case
    files = sorted(glob.glob(os.path.join(post, "*.pvtu")))
    # Keep only the mesh series (names ending _<step>.pvtu), not particle files.
    stepped = []
    for f in files:
        m = re.search(r"_(\d+)\.pvtu$", os.path.basename(f))
        if m:
            stepped.append((int(m.group(1)), f))
    if not stepped:
        return files
    stepped.sort()
    if vel_range is not None:
        lo, hi = vel_range
        stepped = [(n, f) for n, f in stepped if lo <= n <= hi]
    return [f for _, f in stepped]


def read_fields(path: str, pressure_field: str = "Pressure"):
    """Node positions and pressure from one PVTU."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    rdr = vtk.vtkXMLPUnstructuredGridReader()
    rdr.SetFileName(path)
    rdr.Update()
    out = rdr.GetOutput()
    pos = vtk_to_numpy(out.GetPoints().GetData()).astype(np.float64)
    arr = out.GetPointData().GetArray(pressure_field)
    if arr is None:
        names = [out.GetPointData().GetArrayName(i)
                 for i in range(out.GetPointData().GetNumberOfArrays())]
        raise SystemExit(f"  '{pressure_field}' not in {path}\n  have: {names}")
    return pos, vtk_to_numpy(arr).astype(np.float64)


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
    ap.add_argument("--n-snapshots", type=int, default=20,
                    help="⚠️ Subsample this many snapshots EVENLY from the "
                         "range (default 20; 0 = all). Delta p is an "
                         "INSTANTANEOUS field diagnostic, not a time integral, "
                         "so it needs only enough snapshots to average over the "
                         "tool rotation and show the spread. One PVTU read is "
                         "~205 s on LUMI at D2 scale, so all 166 would be 9.5 h "
                         "per case (151 h for 16); 20 is 1.1 h per case.")
    args = ap.parse_args()

    files = find_pvtu(args.case, tuple(args.vel_range) if args.vel_range else None)
    if not files:
        print(f"  no PVTU found under {args.case}")
        return 1
    n_all = len(files)
    if args.n_snapshots and 0 < args.n_snapshots < n_all:
        # Evenly spaced INCLUDING both ends, so the sample spans the range
        # rather than clustering at the start.
        idx = np.unique(np.linspace(0, n_all - 1, args.n_snapshots).astype(int))
        files = [files[i] for i in idx]
        print(f"  {n_all} snapshots available, using {len(files)} evenly spaced "
              f"(~{len(files) * 205 / 60:.0f} min at ~205 s/read)")
    else:
        print(f"  {len(files)} snapshots (all)")

    rpm = args.pin_rpm
    if rpm is None:
        rpm = -600.0
        print(f"  [warn] --pin-rpm not given; assuming {rpm:g} "
              f"(advancing side follows its sign)")
    # omega_z < 0 => advancing side is +y (see module docstring).
    advancing_sign = -1.0 if rpm < 0 else +1.0
    side = "+y" if advancing_sign > 0 else "-y"
    print(f"  pin_rpm {rpm:+g} -> advancing side is {side}")

    pos0, _ = read_fields(files[0], args.pressure_field)
    centre = (np.array(args.pin_center, dtype=np.float64)
              if args.pin_center is not None
              else np.array([pos0[:, 0].mean(), pos0[:, 1].mean(), 0.0]))
    r_pin = args.r_pin if args.r_pin is not None else 3.0e-3
    z_lo, z_hi = pos0[:, 2].min(), pos0[:, 2].max()
    print(f"  centre ({centre[0]:.4g}, {centre[1]:.4g}) m   r_pin {r_pin:.4g} m   "
          f"z [{z_lo:.4g}, {z_hi:.4g}] m")

    rows = []
    for i, f in enumerate(files, 1):
        pos, p = read_fields(f, args.pressure_field)
        d, pa, pr, n, dar = delta_p(pos, p, centre, r_pin, z_lo, z_hi,
                                    args.band, advancing_sign)
        rows.append((os.path.basename(f), d, pa, pr, n, dar))
        # ⚠️ Progress line per snapshot. One PVTU read is ~205 s on LUMI's
        # Lustre at D2 scale (2.8M nodes), so a silent loop looks hung for
        # hours; this also makes a killed run show how far it got.
        print(f"    [{i}/{len(files)}] {os.path.basename(f)}  "
              f"B = {d * 1e-6:+.2f} MPa", flush=True)

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
