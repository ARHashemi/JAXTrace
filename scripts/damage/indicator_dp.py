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


def delta_p(pos, p, centre, r_pin, z_lo, z_hi, band, advancing_sign):
    """Mean pressure difference between the two sides of the pin."""
    dx = pos[:, 0] - centre[0]
    dy = pos[:, 1] - centre[1]
    r = np.hypot(dx, dy)
    # Annulus around the pin, within its z-extent: the contact region.
    sel = (r >= r_pin) & (r <= r_pin * (1.0 + band)) \
        & (pos[:, 2] >= z_lo) & (pos[:, 2] <= z_hi)
    if sel.sum() < 20:
        return np.nan, np.nan, np.nan, int(sel.sum())
    # ⚠️ Compare only points NEAR THE y AXIS (|dx| small), so the difference is
    # across the tool rather than along the weld. Mixing in points ahead of and
    # behind the pin would average the travel-direction gradient into Delta p.
    near = sel & (np.abs(dx) <= r_pin * 0.5)
    adv = near & (advancing_sign * dy > 0)
    ret = near & (advancing_sign * dy < 0)
    if adv.sum() < 10 or ret.sum() < 10:
        return np.nan, np.nan, np.nan, int(near.sum())
    pa, pr = float(p[adv].mean()), float(p[ret].mean())
    return pa - pr, pa, pr, int(near.sum())


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
    args = ap.parse_args()

    files = find_pvtu(args.case, tuple(args.vel_range) if args.vel_range else None)
    if not files:
        print(f"  no PVTU found under {args.case}")
        return 1
    print(f"  {len(files)} snapshots")

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
    for f in files:
        pos, p = read_fields(f, args.pressure_field)
        d, pa, pr, n = delta_p(pos, p, centre, r_pin, z_lo, z_hi,
                               args.band, advancing_sign)
        rows.append((os.path.basename(f), d, pa, pr, n))

    vals = np.array([r[1] for r in rows], dtype=np.float64)
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
    print()
    print("  ⚠️  Reported as a MEASURED value and a ranking quantity. The "
          "15 MPa\n      threshold of Shi et al. (2022) is unverified here and "
          "alloy-specific.")

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("file,delta_p_Pa,p_advancing_Pa,p_retreating_Pa,n_points\n")
            for nm, d, a, r, n in rows:
                fh.write(f"{nm},{d:.6e},{a:.6e},{r:.6e},{n}\n")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
