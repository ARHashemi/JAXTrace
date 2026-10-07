"""M1 + M2 -- shear-layer width delta per case, and a delta-RELATIVE survey window.

    python3 measure_delta_window.py --cases-root <PinShapes> --out <dir>

WHY
---
The published PinShapes result uses a window fixed in tool geometry:
``r/R in [0.40, 0.70]`` of the tool radius. Across the 20 cases that captures
between 2,375 and 114,260 nodes -- a **48x range at identical r_tool = 7 mm** --
and ``rho(n_selected, growth_ratio) = +0.571`` (reproduced 2026-09-28). So the
sampled volume predicts the answer, and the family separation is provisional
(``BENCHMARK_READINESS.md`` section 2; open items O10, O16).

The physics lives in the **shear layer**, not at a fixed fraction of the tool
radius. Different pin shapes present very different amounts of material inside
the same annulus. This measures the shear layer per case and re-surveys relative
to it.

WHAT THIS DOES NOT DO
---------------------
It does **not** replace or modify the static-window results. Both are reported
side by side, per case, so the two can be compared and discussed:

  * ``ratio_static`` -- r/R in [0.40, 0.70]; reproduces the published number
  * ``ratio_delta``  -- r in [0.5*delta, 1.2*delta]; the delta-relative window

M1 = the ``delta_*`` columns.  M2 = ``ratio_delta`` vs ``ratio_static``, and
whether rho(n_selected, ratio) survives.

METHOD (M1)
-----------
Same construction as ``results/measure_shear_layer.py``, generalised from its 8
hardcoded cylindrical cases to any case, and with the depth band taken from each
case's OWN plate thickness.

  1. tool axis from LEVEL < 0; r_tool = max radius of that set
  2. median |u_theta| profiled against r/r_tool, 120 bins over [0.30, 2.0]
  3. peak, then the outward 50 % crossing by linear interpolation
     -> ``r50_over_R``;  ``delta = r50_over_R * r_tool`` in metres

⚠️ 120 bins with interpolation, not 27. A 27-bin profile (0.063 wide) once
returned exactly 0.49 for five different cases -- a quantisation artefact that
looked like a perfect null result. A null from a coarse measurement is not a
null.

⚠️ The original script hardcoded the depth band as z in [-0.0045, -0.0013],
which assumes a 10 mm plate. LUMI PinShapes plates are 6 mm, so the band is
taken as fractions of the measured thickness instead.

⚠️ SNAPSHOT vs PHASE AVERAGE
The published Stage-1 numbers are **full-revolution phase averages** (166 phases,
steps 34..199; the survey runs on time-averaged eta and edot). This script reads a
**single last timestep**, which is much cheaper and is what M1/M2 need in order to
compare window definitions. For A1 the per-phase spread is tiny
(1.447..1.461, std 0.0035), so a snapshot is representative -- but the two are
different quantities and ``ratio_static`` here is reported so the offset is
visible rather than hidden.

READ-ONLY on the case folders.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

# Static window -- kept identical to jaxtrace/damage/run_stage1.py so the
# reproduced number is comparable with the published one.
R_LO_FRAC, R_HI_FRAC = 0.40, 0.70
Z_LO_FRAC, Z_HI_FRAC = 0.00, 0.70
EDOT_MIN = 1.0

# ⚠️ O23 -- the window is anchored on the PIN SURFACE, not scaled from the axis.
#
# First attempt used [0.5*delta, 1.2*delta] with delta measured from the tool
# axis. Because the shear layer hugs the pin, r50/R = 0.361 sits only 0.025
# outside the profile peak, so that window became r/R in [0.18, 0.433] -- i.e.
# INSIDE the pin, where there is no material. Its quadrant counts came out
# near-uniform (64k-70k each), the signature of sampling a disc rather than an
# annulus, and the resulting ratio (4.25) was meaningless.
#
# Measured on A1: r_tool = 7.01 mm is the SHOULDER; there is zero material inside
# r/R = 0.357, so the pin surface is at ~2.50 mm. The window therefore runs
# OUTWARD from r_pin through the velocity decay:
#
#     r in [r_pin, r_pin + PIN_OUT_FRAC * (r50 - r_pin)]
#
# which tracks each pin's own surface instead of a fixed fraction of the shoulder.
PIN_OUT_FRAC = 1.5

# Pin surface detection: the innermost radius holding real material at depth.
# A bin must hold at least this fraction of the busiest bin's count to count as
# "material", so a handful of stray nodes inside the pin cannot define r_pin.
PIN_MIN_COUNT_FRAC = 0.02

N_BINS = 120
# ⚠️ Must start near the axis: clipping at 0.30 made the peak-finder latch
# the first populated bin and hid the fact that r_pin ~ 0.36.
R_PROFILE_LO, R_PROFILE_HI = 0.02, 2.2
MIN_PER_BIN = 30
# How many consecutive bins must stay below half-peak for a crossing to count.
# 3 bins at 0.018 width = ~0.05 in r/R, wide enough to reject single-bin noise
# and narrow enough not to skip a genuinely sharp layer.
CROSS_CONFIRM = 3
# Depth band for the profile, as fractions of the case's own plate thickness.
Z_PROF_LO, Z_PROF_HI = 0.25, 0.75


def read_case(path: Path) -> dict:
    """Points, connectivity and the fields the survey needs, from one PVTU."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    d = r.GetOutput()
    if d.GetNumberOfPoints() == 0:
        raise RuntimeError("no points read")

    pd = d.GetPointData()

    def arr(name):
        a = pd.GetArray(name)
        return None if a is None else vtk_to_numpy(a)

    # ⚠️ Stress/Strain are CELL data, not point data (6 components each).
    # Declared under <PCellData> in the PVTU. Querying GetPointData() returns
    # None for them, which is how they were first missed.
    cd = d.GetCellData()

    def carr(name):
        a = cd.GetArray(name)
        return None if a is None else vtk_to_numpy(a)

    conn = vtk_to_numpy(d.GetCells().GetConnectivityArray())
    return {
        "points": vtk_to_numpy(d.GetPoints().GetData()).astype(np.float64),
        "connectivity": conn.reshape(-1, 4).astype(np.int64),
        "velocity": arr("Displacement"),
        "levelset": arr("LEVEL"),
        "pressure": arr("Pressure"),
        "temperature": arr("Temperature"),
        # Present on the cylindrical FOM cohorts, absent on PinShapes.
        "stress_dev": carr("Stress"),
        "strain_rate": carr("Strain"),
    }


def tool_axis(points, levelset):
    inside = levelset < 0
    if not inside.any():
        raise RuntimeError("no nodes with LEVEL < 0; cannot locate the tool axis")
    tool = points[inside]
    cx, cy = float(tool[:, 0].mean()), float(tool[:, 1].mean())
    r_tool = float(np.hypot(tool[:, 0] - cx, tool[:, 1] - cy).max())
    return cx, cy, r_tool


def measure_delta(points, levelset, velocity, cx, cy, r_tool) -> dict:
    """M1 -- shear-layer width from the tangential-velocity profile."""
    z_min, z_max = float(points[:, 2].min()), float(points[:, 2].max())
    thickness = z_max - z_min
    z_lo = z_min + Z_PROF_LO * thickness
    z_hi = z_min + Z_PROF_HI * thickness

    r = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    m = (levelset >= 0) & (points[:, 2] > z_lo) & (points[:, 2] < z_hi)
    if int(m.sum()) < 500:
        return {"error": "too few nodes in the depth band (%d)" % int(m.sum())}

    ang = np.arctan2(points[m, 1] - cy, points[m, 0] - cx)
    u_t = np.abs(-np.sin(ang) * velocity[m, 0] + np.cos(ang) * velocity[m, 1])
    rr = r[m] / r_tool

    edges = np.linspace(R_PROFILE_LO, R_PROFILE_HI, N_BINS)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    prof = np.full(len(ctr), np.nan)
    cnt = np.zeros(len(ctr), dtype=np.int64)
    for i in range(len(ctr)):
        sel = (rr >= edges[i]) & (rr < edges[i + 1])
        cnt[i] = int(sel.sum())
        if cnt[i] > MIN_PER_BIN:
            prof[i] = float(np.median(u_t[sel]))

    ok = ~np.isnan(prof)
    if int(ok.sum()) < 10:
        return {"error": "profile too sparse (%d usable bins)" % int(ok.sum())}
    c2, p2 = ctr[ok], prof[ok]

    # ---- O23: locate the PIN SURFACE from where material actually starts ----
    # r_tool is the SHOULDER radius (max radius of LEVEL<0). At depth the pin
    # occupies the inner region and there is simply no material there, so the
    # innermost radius carrying a real share of nodes IS the pin surface.
    cnt_ok = cnt[ok]
    thresh = PIN_MIN_COUNT_FRAC * float(cnt_ok.max())
    idx_mat = np.nonzero(cnt_ok >= thresh)[0]
    r_pin_over_R = float(c2[idx_mat[0]]) if len(idx_mat) else float(c2[0])

    ip = int(np.argmax(p2))
    peak = float(p2[ip])
    half = 0.5 * peak

    # ⚠️ Require a SUSTAINED crossing, not the first dip.
    #
    # With 120 bins the width is ~0.018 in r/R, and a single noisy bin just
    # outside the peak can fall below half-peak while the profile is still high.
    # Measured on A1: the naive first-dip search returned r50/R = 0.368, right
    # next to the peak at 0.341, giving a 0.19 mm "shear layer". A coarse 0.04
    # diagnostic of the same data puts the real crossing at r/R = 0.460.
    #
    # So the crossing only counts if the profile STAYS below half-peak for the
    # next CROSS_CONFIRM bins -- the decay is monotone in this flow, so a real
    # crossing never comes back up.
    r50 = float("nan")
    for i in range(ip, len(p2) - 1):
        if p2[i] >= half > p2[i + 1]:
            tail = p2[i + 1:i + 1 + CROSS_CONFIRM]
            if len(tail) and float(np.max(tail)) < half:
                f = (p2[i] - half) / (p2[i] - p2[i + 1])
                r50 = float(c2[i] + f * (c2[i + 1] - c2[i]))
                break

    good = r50 == r50          # NaN check
    # Window runs OUTWARD from the pin surface through the velocity decay.
    if good and r50 > r_pin_over_R:
        w_lo = r_pin_over_R
        w_hi = r_pin_over_R + PIN_OUT_FRAC * (r50 - r_pin_over_R)
    else:
        w_lo = w_hi = float("nan")
    return {
        "r_pin_over_R": r_pin_over_R,
        "r_pin_mm": float(r_pin_over_R * r_tool * 1e3),
        "r_peak_over_R": float(c2[ip]),
        "u_peak": peak,
        "r50_over_R": r50,
        # Shear-layer THICKNESS measured from the pin surface, not the axis.
        "delta_over_R": float(r50 - r_pin_over_R) if good else float("nan"),
        "delta_mm": float((r50 - r_pin_over_R) * r_tool * 1e3) if good else float("nan"),
        "win_lo_over_R": w_lo,
        "win_hi_over_R": w_hi,
        "n_bins_used": int(ok.sum()),
        "thickness_mm": thickness * 1e3,
    }


def sigma_eq_from_solver(stress_dev):
    """sigma_eq = sqrt(3 J2) from the solver's OWN deviatoric stress.

    ⚠️ Verified on cylindrical_000: trace(Stress) has max|.| = 3e-8 Pa, i.e.
    machine zero, so this array IS the pure deviator and sqrt(3*J2) is a valid
    von Mises equivalent stress with NO rheology assumption. That makes it an
    independent check on the Norton route, which is otherwise unvalidated.

    GiD component order is (xx, yy, zz, xy, yz, xz).
    """
    if stress_dev is None or stress_dev.ndim != 2 or stress_dev.shape[1] != 6:
        return None
    sd = stress_dev
    J2 = (0.5 * (sd[:, 0] ** 2 + sd[:, 1] ** 2 + sd[:, 2] ** 2)
          + sd[:, 3] ** 2 + sd[:, 4] ** 2 + sd[:, 5] ** 2)
    return np.sqrt(3.0 * J2)


def detect_advancing(points, levelset, velocity, cx, cy, r_tool) -> dict:
    """Advancing flank sign, measured not assumed.

    The definition is relative to TOOL TRAVEL. Material flows +x past a
    stationary tool => the tool travels -x in the workpiece frame => the
    advancing flank is the one whose tool-surface u_x is more negative.
    """
    r = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    shell = (levelset >= 0) & (r > 0.95 * r_tool) & (r < 1.15 * r_tool)
    if int(shell.sum()) < 100:
        return {"advancing_y_sign": -1, "n_shell": int(shell.sum()),
                "note": "too few shell nodes; defaulted to -y"}
    ys = points[shell, 1] - cy
    ux = velocity[shell, 0]
    ux_p = float(np.median(ux[ys > 0])) if int((ys > 0).sum()) > 10 else float("nan")
    ux_m = float(np.median(ux[ys < 0])) if int((ys < 0).sum()) > 10 else float("nan")
    adv = 1 if ux_p < ux_m else -1
    return {"advancing_y_sign": adv, "ux_plus_y": ux_p, "ux_minus_y": ux_m,
            "n_shell": int(shell.sum())}


def survey(points, levelset, edot, eta, cx, cy, r_tool, adv,
           r_lo_m, r_hi_m, label) -> dict:
    """Triaxiality by quadrant inside an explicit radial window (metres).

    Mirrors jaxtrace.damage.run_stage1.quadrant_survey, but takes the radial
    window in metres so the same code serves both the static and the
    delta-relative definition.
    """
    z_min, z_max = float(points[:, 2].min()), float(points[:, 2].max())
    thickness = z_max - z_min
    z_lo = z_min + Z_LO_FRAC * thickness
    z_hi = z_min + Z_HI_FRAC * thickness

    r = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    sel = ((levelset >= 0) & (r > r_lo_m) & (r < r_hi_m)
           & (points[:, 2] >= z_lo) & (points[:, 2] < z_hi)
           & (edot > EDOT_MIN))

    x = points[sel, 0] - cx
    y = points[sel, 1] - cy
    e = eta[sel]

    quads = {
        "adv_wake":  (y * adv > 0) & (x > 0),
        "adv_ahead": (y * adv > 0) & (x < 0),
        "ret_wake":  (y * adv < 0) & (x > 0),
        "ret_ahead": (y * adv < 0) & (x < 0),
    }

    out = {"window": label,
           "r_window_m": [float(r_lo_m), float(r_hi_m)],
           "r_window_over_R": [float(r_lo_m / r_tool), float(r_hi_m / r_tool)],
           "n_selected": int(sel.sum()), "quadrants": {}}

    for name, mask in quads.items():
        n = int(mask.sum())
        if n < 30:
            out["quadrants"][name] = {"n": n, "insufficient": True}
            continue
        ee = e[mask]
        out["quadrants"][name] = {
            "n": n,
            "eta_median": float(np.median(ee)),
            "frac_positive": float((ee > 0).mean()),
            # ⚠️ MEDIAN, not mean — must match run_stage1.quadrant_survey:205.
            # exp(1.5*eta) is right-skewed, so the mean sits well above the
            # median and inflates the adv/ret ratio. Using the mean gave A1
            # ratio 1.947 against the published 1.454 on identical eta values
            # (agreeing to 4 decimals), which is how this was caught.
            "growth_median": float(np.median(np.exp(1.5 * ee))),
        }

    aw = out["quadrants"].get("adv_wake", {})
    rw = out["quadrants"].get("ret_wake", {})
    if "growth_median" in aw and "growth_median" in rw and rw["growth_median"] > 0:
        out["growth_ratio_adv_over_ret"] = aw["growth_median"] / rw["growth_median"]
    else:
        out["growth_ratio_adv_over_ret"] = None
    return out


def spearman(x, y):
    """Spearman rho via ranks. No scipy on LUMI's python3.6."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def find_cases(root: Path, max_depth: int = 3):
    found, stack = [], [(root, 0)]
    while stack:
        d, depth = stack.pop(0)
        try:
            entries = sorted(d.iterdir())
        except OSError:
            continue
        for e in entries:
            if not e.is_dir():
                continue
            if e.name.endswith(".gid"):
                found.append(e)
            elif depth < max_depth:
                stack.append((e, depth + 1))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases-root", type=Path, nargs="+", required=True,
                    help="one or more roots to sweep (READ-ONLY). "
                         "e.g. .../Cases/PinShapes .../Cases/ROM/FOM_cases")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", type=Path,
                    default=Path("/projappl/project_465002752/hashemia/JAXTrace"))
    ap.add_argument("--model", default="norton", choices=["norton", "constant"])
    ap.add_argument("--mu-eff", type=float, default=1.0e6)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated case names, e.g. A1,C1")
    args = ap.parse_args()

    sys.path.insert(0, str(args.repo))
    from jaxtrace.damage.fields import build_damage_fields
    from jaxtrace.damage.rheology import build_sigma_eq

    args.out.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None

    cases = []
    for root in args.cases_root:
        found = find_cases(root)
        print("found %3d cases under %s" % (len(found), root))
        cases += found
    print("total %d cases" % len(cases))
    print()
    rows = []
    for cdir in cases:
        name = cdir.name[:-4]
        if only and name not in only:
            continue
        post = cdir / "post"
        if not post.is_dir():
            print("  SKIP %-8s no post/" % name)
            continue
        # Highest-numbered PVTU whatever its prefix: copied cases legitimately
        # keep the SOURCE case's naming (e.g. D4.gid holding C2_*.pvtu).
        try:
            pv = sorted(post.glob("*_[0-9]*.pvtu"),
                        key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        except (ValueError, IndexError):
            pv = []
        if not pv:
            print("  SKIP %-8s no PVTU" % name)
            continue
        # ⚠️ The highest-numbered PVTU is not always USABLE. On D4 the last two
        # files (C2_200, C2_201) declare only "Points" -- truncated final writes
        # -- while C2_199 is complete. Taking the highest number made D4 fail
        # here AND in the M5 sweep, and the case looked broken when only its last
        # two files were. Walk DOWN to the newest file that has the fields.
        last = None
        for cand in reversed(pv):
            try:
                head = cand.read_text(errors="replace")[:4000]
            except OSError:
                continue
            if 'Name="Displacement"' in head and 'Name="LEVEL"' in head:
                last = cand
                break
        if last is None:
            print("  SKIP %-8s no PVTU declaring Displacement+LEVEL" % name)
            continue
        if last is not pv[-1]:
            print("  NOTE %-8s newest usable PVTU is %s (skipped %d truncated)"
                  % (name, last.name, len(pv) - 1 - pv.index(last)))

        try:
            d = read_case(last)
            if d["velocity"] is None or d["levelset"] is None:
                print("  SKIP %-8s missing Displacement or LEVEL" % name)
                continue

            cx, cy, r_tool = tool_axis(d["points"], d["levelset"])
            dl = measure_delta(d["points"], d["levelset"], d["velocity"],
                               cx, cy, r_tool)
            if "error" in dl:
                print("  SKIP %-8s delta: %s" % (name, dl["error"]))
                continue

            # Two passes, as run_stage1 does: edot first, then sigma_eq, then
            # rebuild with sigma_flow so eta uses the real rheology.
            edot0 = build_damage_fields(
                d["points"], d["connectivity"], d["velocity"], d["pressure"],
                mu_eff=np.full(len(d["points"]), args.mu_eff), verbose=False,
            )["edot"].astype(np.float64)

            mats = sorted(cdir.glob("*.mat"))
            sigma_eq, _ = build_sigma_eq(
                args.model, edot0, d["temperature"],
                mu_eff=args.mu_eff,
                mat_path=(mats[0] if mats else None),
            )
            f = build_damage_fields(
                d["points"], d["connectivity"], d["velocity"], d["pressure"],
                sigma_flow=sigma_eq, verbose=False,
            )
            edot, eta = f["edot"], f["eta"]

            # Independent sigma_eq from the solver's own deviatoric stress,
            # where available. No rheology assumption, so it validates the
            # Norton route rather than restating it.
            seq_solver = sigma_eq_from_solver(d.get("stress_dev"))
            if seq_solver is not None:
                # sigma_eq is per-ELEMENT here; compare against the per-element
                # Norton value by averaging the nodal Norton field onto cells.
                seq_norton_cell = sigma_eq[d["connectivity"]].mean(axis=1)
                act = seq_solver > 1.0e5        # skip dead material
                if int(act.sum()) > 1000:
                    ratio = float(np.median(seq_norton_cell[act])
                                  / np.median(seq_solver[act]))
                    corr = float(np.corrcoef(seq_norton_cell[act],
                                             seq_solver[act])[0, 1])
                else:
                    ratio, corr = float("nan"), float("nan")
                solver_chk = {
                    "seq_solver_med_MPa": float(np.median(seq_solver[act])) / 1e6,
                    "seq_norton_med_MPa": float(np.median(seq_norton_cell[act])) / 1e6,
                    "norton_over_solver": ratio,
                    "corr": corr,
                    "trace_max_Pa": float(np.abs(
                        d["stress_dev"][:, 0] + d["stress_dev"][:, 1]
                        + d["stress_dev"][:, 2]).max()),
                    "n_active_cells": int(act.sum()),
                }
            else:
                solver_chk = None

            ainfo = detect_advancing(d["points"], d["levelset"], d["velocity"],
                                     cx, cy, r_tool)
            adv = ainfo["advancing_y_sign"]

            s_static = survey(d["points"], d["levelset"], edot, eta, cx, cy,
                              r_tool, adv, R_LO_FRAC * r_tool, R_HI_FRAC * r_tool,
                              "static_rR")
            if dl["win_hi_over_R"] != dl["win_hi_over_R"]:     # NaN
                print("  WARN %-8s no usable delta window; delta survey skipped"
                      % name)
                s_delta = {"n_selected": 0, "growth_ratio_adv_over_ret": None,
                           "r_window_over_R": [float("nan")] * 2,
                           "window": "delta_pin_anchored", "quadrants": {}}
            else:
                s_delta = survey(d["points"], d["levelset"], edot, eta, cx, cy,
                                 r_tool, adv,
                                 dl["win_lo_over_R"] * r_tool,
                                 dl["win_hi_over_R"] * r_tool,
                                 "delta_pin_anchored")

            row = {
                "family": cdir.parent.name, "case": name, "pvtu": last.name,
                "root": str(cdir.parent.parent.name),
                "r_tool_mm": r_tool * 1e3, "n_nodes": len(d["points"]),
                "advancing_y_sign": adv,
                "thickness_mm": dl["thickness_mm"],
                "r_pin_over_R": dl["r_pin_over_R"],
                "r_pin_mm": dl["r_pin_mm"],
                "r_peak_over_R": dl["r_peak_over_R"],
                "r50_over_R": dl["r50_over_R"],
                "delta_over_R": dl["delta_over_R"],
                "delta_mm": dl["delta_mm"],
                "win_lo_over_R": dl["win_lo_over_R"],
                "win_hi_over_R": dl["win_hi_over_R"],
                "seq_solver_med_MPa": (solver_chk or {}).get("seq_solver_med_MPa"),
                "seq_norton_med_MPa": (solver_chk or {}).get("seq_norton_med_MPa"),
                "norton_over_solver": (solver_chk or {}).get("norton_over_solver"),
                "seq_corr": (solver_chk or {}).get("corr"),
                "n_sel_static": s_static["n_selected"],
                "n_sel_delta": s_delta["n_selected"],
                "ratio_static": s_static["growth_ratio_adv_over_ret"],
                "ratio_delta": s_delta["growth_ratio_adv_over_ret"],
                "delta_window_over_R": s_delta["r_window_over_R"],
            }
            rows.append(row)
            json.dump({"row": row, "static": s_static, "delta": s_delta,
                       "advancing": ainfo, "delta_fit": dl,
                       "solver_stress_check": solver_chk},
                      open(args.out / ("delta_%s.json" % name), "w"), indent=1)

            def fmt(v):
                return ("%.3f" % v) if v is not None else "  n/a"
            print("  %-8s R=%.2f r_pin=%.2f d=%.2fmm win=[%.2f,%.2f] | "
                  "static n=%6d ratio=%s | delta n=%6d ratio=%s"
                  % (name, r_tool * 1e3, dl["r_pin_mm"], dl["delta_mm"],
                     dl["win_lo_over_R"], dl["win_hi_over_R"],
                     s_static["n_selected"], fmt(s_static["growth_ratio_adv_over_ret"]),
                     s_delta["n_selected"], fmt(s_delta["growth_ratio_adv_over_ret"])),
                  flush=True)
        except Exception as exc:
            print("  FAIL %-8s %s: %s" % (name, type(exc).__name__, exc), flush=True)
            continue

    if not rows:
        print("no cases measured")
        return 1

    keys = ["family", "case"] + sorted({k for r in rows for k in r}
                                       - {"family", "case"})
    with open(args.out / "delta_window.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json.dump(rows, open(args.out / "delta_window.json", "w"), indent=1)

    # ---- M2: did the confound survive the window change? ----
    print()
    print("=" * 74)
    print(" M2 -- does the n_selected confound survive a delta-relative window?")
    print("=" * 74)
    for tag, nkey, rkey in (("static", "n_sel_static", "ratio_static"),
                            ("delta ", "n_sel_delta", "ratio_delta")):
        ok = [r for r in rows if r[rkey] is not None]
        if len(ok) < 5:
            print("  %s: too few usable cases (%d)" % (tag, len(ok)))
            continue
        ns = [r[nkey] for r in ok]
        rs = [r[rkey] for r in ok]
        rho = spearman(ns, rs)
        print("  %s window: n=%2d  n_selected %6d..%6d (%4.0fx)  "
              "rho(n_sel, ratio) = %+.3f"
              % (tag, len(ok), min(ns), max(ns), max(ns) / max(1, min(ns)), rho))

    print()
    print("  delta spread across cases:")
    ds = [r["delta_mm"] for r in rows if r["delta_mm"] == r["delta_mm"]]
    rp = [r["r_pin_mm"] for r in rows if r["r_pin_mm"] == r["r_pin_mm"]]
    if ds:
        print("    delta (from pin) : %.2f .. %.2f mm  (mean %.2f, spread %.0f %%)"
              % (min(ds), max(ds), sum(ds) / len(ds),
                 100.0 * (max(ds) - min(ds)) / (sum(ds) / len(ds))))
    if rp:
        print("    r_pin            : %.2f .. %.2f mm  (mean %.2f)"
              % (min(rp), max(rp), sum(rp) / len(rp)))

    # ---- independent sigma_eq cross-check, where the solver exported stress ----
    chk = [r for r in rows if r.get("norton_over_solver") is not None
           and r["norton_over_solver"] == r["norton_over_solver"]]
    if chk:
        rs = [r["norton_over_solver"] for r in chk]
        cs = [r["seq_corr"] for r in chk]
        print()
        print("=" * 74)
        print(" sigma_eq cross-check vs the SOLVER'S OWN deviatoric stress")
        print("=" * 74)
        print("  cases with CellData Stress: %d / %d" % (len(chk), len(rows)))
        print("  Norton / solver  : %.3f .. %.3f  (median %.3f)"
              % (min(rs), max(rs), sorted(rs)[len(rs) // 2]))
        print("  correlation      : %.3f .. %.3f  (median %.3f)"
              % (min(cs), max(cs), sorted(cs)[len(cs) // 2]))
        print("  ratio == 1 and corr -> 1 would mean the Norton route reproduces")
        print("  the solver's stress. Departures bound the rheology error in eta.")
    else:
        print()
        print("  (no case exported CellData Stress -- cross-check unavailable)")

    print()
    print("wrote %s" % (args.out / "delta_window.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
