"""Stage 1 driver — damage fields on the original FOM mesh, one case.

Invoked by ``1phase-2phase/run_damage_stage1.sh``; also runnable directly:

    python -m jaxtrace.damage.run_stage1 \
        --pvtu /scratch/shared/ROM/FOM/cylindrical_000.gid/post/cylindrical_119.pvtu \
        --case 000 --model norton --mat /flash/users/ali/data/cylA.gid/cylA.mat \
        --outdir /scratch/shared/ROM/damage/stage1

Writes ``dmg_<case>_<model>.npz`` (nodal fields), a ``.json`` of diagnostics
and the quadrant survey, and appends one row to the summary CSV.

No pass/fail verdict is issued: the whole cohort is surveyed first and the
acceptance criterion chosen afterwards from the observed distribution.

The survey evaluates triaxiality in the PIN REGION AT DEPTH, not
depth-averaged: on cylA the pattern inverts in the upper layers where the
shoulder forges material downward, so a depth-average mixes two opposite
signatures.  See ``results/RESULTS_LOG.md`` item O2.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .fields import build_damage_fields
from .rheology import build_sigma_eq

# Acceptance-test sampling window, as fractions of the tool radius and of the
# plate thickness.  Expressed relatively so it transfers between cylA (9 mm
# tool, 10 mm plate) and the cohort (smaller tool, 4.5 mm plate).
R_LO_FRAC, R_HI_FRAC = 0.40, 0.70     # of tool radius: the pin flank
Z_LO_FRAC, Z_HI_FRAC = 0.00, 0.70     # of thickness from the bottom: below shoulder
EDOT_MIN = 1.0                        # [1/s] only actively deforming material


def read_pvtu(path: Path) -> dict:
    """Read points, tetra connectivity and the point fields we need."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()

    if grid.GetNumberOfPoints() == 0:
        raise RuntimeError(f"no points read from {path}")
    if grid.GetCellType(0) != 10:
        raise RuntimeError(f"expected tetrahedra (VTK type 10), got {grid.GetCellType(0)}")

    pd = grid.GetPointData()
    names = {pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())}
    for required in ("Displacement", "Pressure", "Temperature", "LEVEL"):
        if required not in names:
            raise RuntimeError(f"{path} lacks {required!r}; has {sorted(names)}")

    conn = vtk_to_numpy(grid.GetCells().GetConnectivityArray()).reshape(-1, 4)

    return {
        "points": vtk_to_numpy(grid.GetPoints().GetData()).astype(np.float64),
        "connectivity": conn,
        # 'Displacement' stores VELOCITY in m/s in these files — confirmed,
        # no division by dt.
        "velocity": vtk_to_numpy(pd.GetArray("Displacement")).astype(np.float64),
        "pressure": vtk_to_numpy(pd.GetArray("Pressure")).astype(np.float64),
        "temperature": vtk_to_numpy(pd.GetArray("Temperature")).astype(np.float64),
        "levelset": vtk_to_numpy(pd.GetArray("LEVEL")).astype(np.float64),
    }


def detect_orientation(points, levelset, velocity, cx, cy, r_tool) -> dict:
    """Determine rotation sense and hence which flank is advancing.

    ⚠️ This MUST be measured per case, not assumed.  cylA and the 20-case
    cohort rotate clockwise (signed RPM negative) so their advancing side is
    -y, but A2 rotates COUNTER-clockwise (signed RPM +800) and its advancing
    side is +y.  Hardcoding one convention silently mislabels the other.

    Method: material flows +x past a stationary tool, so in the workpiece
    frame the tool travels -x.  The advancing flank is the one where the tool
    surface velocity is parallel to travel, i.e. u_x < 0.
    """
    r = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    z_mid_lo = z_min + 0.15 * (z_max - z_min)
    z_mid_hi = z_min + 0.85 * (z_max - z_min)

    # A thin shell just outside the tool, at mid-depth, where the shear layer
    # carries the tool's own motion.
    shell = (
        (levelset >= 0)
        & (r > 0.30 * r_tool) & (r < 0.50 * r_tool)
        & (points[:, 2] > z_mid_lo) & (points[:, 2] < z_mid_hi)
    )
    if shell.sum() < 100:
        shell = (levelset >= 0) & (r > 0.30 * r_tool) & (r < 0.60 * r_tool)

    xs = points[shell, 0] - cx
    ys = points[shell, 1] - cy
    ang = np.arctan2(ys, xs)
    u_theta = (-np.sin(ang) * velocity[shell, 0]
               + np.cos(ang) * velocity[shell, 1])
    omega_sign = float(np.sign(np.median(u_theta)))

    # Cross-check directly from the flank u_x, which is what actually defines
    # the advancing side.
    near_axis = np.abs(xs) < 0.15 * r_tool
    ux = velocity[shell, 0]
    ux_plus_y = float(np.median(ux[near_axis & (ys > 0)])) if (near_axis & (ys > 0)).sum() > 20 else np.nan
    ux_minus_y = float(np.median(ux[near_axis & (ys < 0)])) if (near_axis & (ys < 0)).sum() > 20 else np.nan

    if np.isfinite(ux_plus_y) and np.isfinite(ux_minus_y):
        advancing_y_sign = +1.0 if ux_plus_y < ux_minus_y else -1.0
    else:
        # Fall back on the rotation sense: CW (omega<0) -> advancing -y.
        advancing_y_sign = -omega_sign

    return {
        "omega_sign": omega_sign,
        "rotation": "CCW" if omega_sign > 0 else "CW",
        "u_theta_median": float(np.median(u_theta)),
        "ux_plus_y": ux_plus_y,
        "ux_minus_y": ux_minus_y,
        "advancing_y_sign": advancing_y_sign,
        "advancing_side": "+y" if advancing_y_sign > 0 else "-y",
        "n_shell": int(shell.sum()),
    }


def quadrant_survey(points, levelset, edot, eta, velocity, verbose=True) -> dict:
    """Triaxiality by quadrant in the pin region at depth — an OBSERVATION.

    No pass/fail judgement is made here.  The full cohort is being surveyed
    first; any acceptance criterion gets chosen afterwards, from the observed
    distribution.

    Geometry conventions (``STAGED_VALIDATION_PLAN.md`` 0.3b):
        material flows +x  ->  tool travels -x  ->  wake is x > 0
    The advancing flank is DETECTED per case (see ``detect_orientation``)
    rather than assumed, because rotation sense differs between cases.
    """
    inside = levelset < 0
    if not inside.any():
        raise RuntimeError("no nodes inside the tool (LEVEL<0); cannot locate axis")

    tool = points[inside]
    cx, cy = tool[:, 0].mean(), tool[:, 1].mean()
    r_tool = float(np.hypot(tool[:, 0] - cx, tool[:, 1] - cy).max())

    orient = detect_orientation(points, levelset, velocity, cx, cy, r_tool)
    adv = orient["advancing_y_sign"]

    z_min, z_max = float(points[:, 2].min()), float(points[:, 2].max())
    thickness = z_max - z_min

    r = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    z_lo = z_min + Z_LO_FRAC * thickness
    z_hi = z_min + Z_HI_FRAC * thickness

    sel = (
        (levelset >= 0)
        & (r > R_LO_FRAC * r_tool) & (r < R_HI_FRAC * r_tool)
        & (points[:, 2] >= z_lo) & (points[:, 2] < z_hi)
        & (edot > EDOT_MIN)
    )

    x = points[sel, 0] - cx
    y = points[sel, 1] - cy
    e = eta[sel]

    # adv > 0 means the advancing flank is +y; adv < 0 means -y.
    quads = {
        "adv_wake":  (y * adv > 0) & (x > 0),
        "adv_ahead": (y * adv > 0) & (x < 0),
        "ret_wake":  (y * adv < 0) & (x > 0),
        "ret_ahead": (y * adv < 0) & (x < 0),
    }

    out = {
        "tool_axis": [float(cx), float(cy)],
        "tool_radius_m": r_tool,
        "z_window_m": [float(z_lo), float(z_hi)],
        "n_selected": int(sel.sum()),
        "orientation": orient,
        "quadrants": {},
    }

    for name, mask in quads.items():
        if mask.sum() < 30:
            out["quadrants"][name] = {"n": int(mask.sum()), "insufficient": True}
            continue
        ee = e[mask]
        out["quadrants"][name] = {
            "n": int(mask.sum()),
            "eta_median": float(np.median(ee)),
            "frac_positive": float(np.mean(ee > 0)),
            "growth_median": float(np.median(np.exp(1.5 * ee))),
        }

    aw = out["quadrants"].get("adv_wake", {})
    rw = out["quadrants"].get("ret_wake", {})
    if "growth_median" in aw and "growth_median" in rw and rw["growth_median"] > 0:
        ratio = aw["growth_median"] / rw["growth_median"]
    else:
        ratio = float("nan")
    out["growth_ratio_adv_over_ret"] = float(ratio)

    # Which flank carries the tensile (void-opening) wake.  Recorded as an
    # OBSERVATION, not a pass/fail: case 003 inverts relative to case 000 at
    # identical weld pitch, so whether "advancing" is the right expectation is
    # exactly what the full cohort sweep is meant to establish.  Any acceptance
    # criterion should be chosen after seeing the distribution, not before.
    out["tensile_flank"] = (
        "advancing"
        if aw.get("frac_positive", 0.0) > rw.get("frac_positive", 0.0)
        else "retreating"
    )
    out["matches_literature_expectation"] = bool(
        out["tensile_flank"] == "advancing"
    )

    if verbose:
        print(f"  quadrant survey (pin flank, r/R in [{R_LO_FRAC},{R_HI_FRAC}], "
              f"lower {Z_HI_FRAC:.0%} of thickness, edot>{EDOT_MIN})")
        print(f"    tool radius {r_tool*1e3:.2f} mm, "
              f"z window [{z_lo*1e3:+.1f},{z_hi*1e3:+.1f}] mm, "
              f"n={out['n_selected']:,}")
        print(f"    rotation {orient['rotation']} "
              f"(u_theta med {orient['u_theta_median']:+.4f}) "
              f"-> advancing side {orient['advancing_side']}")
        for name in ("adv_wake", "adv_ahead", "ret_wake", "ret_ahead"):
            q = out["quadrants"][name]
            if q.get("insufficient"):
                print(f"    {name:10s} n={q['n']:6d}  (too few)")
            else:
                print(f"    {name:10s} n={q['n']:6d}  eta_med={q['eta_median']:+.4f}  "
                      f"frac+={q['frac_positive']:.3f}  growth={q['growth_median']:.4f}")
        print(f"    growth ratio adv/ret = {ratio:.4f}   "
              f"tensile flank: {out['tensile_flank'].upper()}"
              f"{'' if out['matches_literature_expectation'] else '  (inverted vs literature)'}")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pvtu", required=True, type=Path)
    ap.add_argument("--case", required=True)
    ap.add_argument("--model", default="norton",
                    choices=["constant", "norton", "sellars_tegart"])
    ap.add_argument("--mat", type=Path, default=None,
                    help="GiD .mat file supplying the Norton tables")
    ap.add_argument("--mu-eff", type=float, default=1.0e6,
                    help="uniform viscosity for --model constant [Pa s]")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--summary", type=Path, default=None)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.case}_{args.model}"
    t0 = time.time()

    print(f"=== stage 1 : case {args.case}, model {args.model} ===")
    print(f"  {args.pvtu}")

    data = read_pvtu(args.pvtu)
    n_nodes = len(data["points"])
    n_elems = len(data["connectivity"])
    print(f"  loaded {n_nodes:,} nodes / {n_elems:,} tets in {time.time()-t0:.1f}s")

    # Pass 1: kinematics only, to get edot for the rheology.
    base = build_damage_fields(
        data["points"], data["connectivity"],
        data["velocity"], data["pressure"],
        mu_eff=np.full(n_nodes, args.mu_eff),
        verbose=False,
    )
    edot = base["edot"].astype(np.float64)

    # Pass 2: the chosen sigma_eq, then recompute eta with it.
    sigma_eq, rheo_info = build_sigma_eq(
        args.model, edot, data["temperature"],
        mu_eff=args.mu_eff, mat_path=args.mat,
    )
    fields = build_damage_fields(
        data["points"], data["connectivity"],
        data["velocity"], data["pressure"],
        sigma_flow=sigma_eq,
        verbose=True,
    )

    print(f"  rheology: {rheo_info['model']} — "
          f"sigma_eq median {np.median(sigma_eq)/1e6:.2f} MPa")

    acc = quadrant_survey(
        data["points"], data["levelset"],
        fields["edot"].astype(np.float64), fields["eta"].astype(np.float64),
        data["velocity"],
    )

    runtime = time.time() - t0

    npz = args.outdir / f"dmg_{tag}.npz"
    np.savez_compressed(
        npz,
        **{k: v for k, v in fields.items() if k != "diagnostics"},
        points=data["points"].astype(np.float32),
        levelset=data["levelset"].astype(np.float32),
        temperature=data["temperature"].astype(np.float32),
    )

    meta = {
        "case": args.case,
        "model": args.model,
        "pvtu": str(args.pvtu),
        "n_nodes": n_nodes,
        "n_elems": n_elems,
        "rheology": rheo_info,
        "diagnostics": fields["diagnostics"],
        "sigma_eq_MPa": {
            "median": float(np.median(sigma_eq) / 1e6),
            "p10": float(np.percentile(sigma_eq, 10) / 1e6),
            "p90": float(np.percentile(sigma_eq, 90) / 1e6),
        },
        "survey": acc,
        "runtime_s": runtime,
    }
    (args.outdir / f"dmg_{tag}.json").write_text(json.dumps(meta, indent=2))

    if args.summary is not None:
        aw = acc["quadrants"].get("adv_wake", {})
        rw = acc["quadrants"].get("ret_wake", {})
        row = ",".join(str(v) for v in [
            args.case, args.model, n_nodes, n_elems,
            f"{fields['diagnostics']['incompressibility']:.5f}",
            f"{np.median(fields['edot']):.3f}",
            f"{fields['edot'].max():.1f}",
            f"{np.median(sigma_eq)/1e6:.2f}",
            f"{np.median(fields['eta']):.5f}",
            f"{aw.get('eta_median', float('nan')):.5f}",
            f"{aw.get('frac_positive', float('nan')):.4f}",
            f"{rw.get('eta_median', float('nan')):.5f}",
            f"{rw.get('frac_positive', float('nan')):.4f}",
            f"{acc['growth_ratio_adv_over_ret']:.4f}",
            acc["tensile_flank"],
            int(acc["matches_literature_expectation"]),
            acc["orientation"]["rotation"],
            acc["orientation"]["advancing_side"],
            f"{runtime:.1f}",
        ])
        with open(args.summary, "a") as fh:
            fh.write(row + "\n")

    print(f"  wrote {npz.name} + .json   ({runtime:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
