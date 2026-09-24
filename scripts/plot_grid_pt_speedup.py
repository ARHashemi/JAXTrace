"""Compare grid PT to mesh PT on the analytic reference cases.

Presents the story on two axes rather than one scatter:

  (a) EULERIAN error: grid-projected velocity (mesh -> grid, raw P1) vs
      the mesh's own P1 velocity.  Both compared to the analytic
      velocity.  Number source: rom_out/analytic_grid_projection/
      projection_summary_<case>.csv (uniform_base row) for grid; from
      the same tool but sampled at the mesh nodes for mesh -- but the
      mesh-vs-analytic velocity error is zero by construction here
      (we sampled the analytic at the mesh nodes to define the mesh
      velocity in the first place), so we report:
        - grid-vs-analytic (from the projection CSV)
        - as a REFERENCE, we also show the mesh's *nodal* velocity
          error against analytic, which is by construction zero.
      This makes the point that "grid at mesh's natural resolution
      preserves the nodal velocity essentially perfectly".

  (b) LAGRANGIAN error and wall time: grid PT (out_grid_uniform) vs
      mesh + HCT-3D PT (best mesh variant, from summary.json), both
      compared to the analytic reference run.  Bar chart with
      annotations for wall time.

Output: paper_figs/rom_pt_analytic/grid_vs_mesh_pt.{png,svg}
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASE_ROOTS = {
    "recirc_2026":  Path("/home/arhashemi/fsw-gpu/scratch/shared/ROM/FOM_analytic/recirc_2026"),
    "rot_cyl_2026": Path("/home/arhashemi/fsw-gpu/scratch/shared/ROM/FOM_analytic/rot_cyl_2026"),
}
DIAG_DIR = Path("/home/arhashemi/Workspace/welding/JAXTrace/rom_out/analytic_grid_projection")
OUT_DIR  = Path("/home/arhashemi/Workspace/welding/JAXTrace/paper_figs/rom_pt_analytic")

_WALL_RE = re.compile(r"Wall time:\s+([\d.]+)s")


def _wall_from_log(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    m = _WALL_RE.findall(log_path.read_text())
    return float(m[-1]) if m else None


def _walltime_for(case_root: Path, variant: str) -> float | None:
    if variant.startswith("grid_"):
        d = case_root / f"out_grid_{variant[len('grid_'):]}"
        candidates = list(d.glob("run_grid.log")) + list(d.glob("*.log"))
    else:
        d = case_root / f"out_mesh_{variant}"
        candidates = list(d.glob("run_mesh_*.log"))
    for c in candidates:
        w = _wall_from_log(c)
        if w is not None:
            return w
    return None


def _grid_projection_rel_rms(case: str) -> float:
    csv_path = DIAG_DIR / f"projection_summary_{case}.csv"
    if not csv_path.exists():
        return float("nan")
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if r["grid"] == "uniform_base" and r["method"] == "p1_raw":
                return float(r["err_rel_rms_pct"])
    return float("nan")


def _best_mesh_variant(variants: dict) -> tuple[str, float]:
    """Return (variant_name, rms_final) for the mesh variant with the
    smallest final-step RMS PT error.  Excludes grid_* variants."""
    best = min(
        ((v["rms_err"][-1] if isinstance(v["rms_err"], list)
          else v["rms_err"], k) for k, v in variants.items()
         if not k.startswith("grid_") and "rms_err" in v),
        default=(float("nan"), None),
    )
    return (best[1], best[0])


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(CASE_ROOTS), figsize=(12, 8),
                             sharey=False, gridspec_kw={"hspace": 0.35})

    for col, (case, root) in enumerate(CASE_ROOTS.items()):
        # ---- panel (a) EULERIAN --------------------------------------
        ax_e = axes[0, col]
        # grid rel_rms from projection tool
        grid_rel = _grid_projection_rel_rms(case)
        # mesh nodal error is 0 by construction (we sampled analytic
        # at mesh nodes to define mesh velocity in the projection
        # experiment).  Display as a floor at 10^-4.
        floor = 1e-4
        mesh_disp = floor
        grid_disp = max(grid_rel, floor)
        colours = ["#5289c7", "#3f8f3f"]
        labels  = ["mesh nodal\n(= analytic)",
                   "grid_uniform\nprojected"]
        bars = ax_e.bar(np.arange(2), [mesh_disp, grid_disp], 0.6,
                        color=colours, edgecolor="black", linewidth=0.4)
        # Annotate above each bar; use log-space multiplier so the label
        # sits just above.
        for bar, actual_val in zip(bars, [0.0, grid_rel]):
            text = "≈ 0 %" if actual_val < 1e-4 else f"{actual_val:.3f} %"
            ax_e.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() * 1.4, text,
                      ha="center", va="bottom", fontsize=9)
        ax_e.set_xticks(np.arange(2))
        ax_e.set_xticklabels(labels, fontsize=8)
        ax_e.set_yscale("log")
        ax_e.set_ylim(floor, 1.0)
        ax_e.grid(axis="y", which="both", alpha=0.3)
        if col == 0:
            ax_e.set_ylabel("Eulerian rel_rms (% of |v|_max), log",
                            fontsize=9)
        ax_e.set_title(f"{case} · (a) Velocity — mesh → grid projection",
                       fontsize=10)

        # ---- panel (b) LAGRANGIAN + wall time ------------------------
        ax_l = axes[1, col]
        summ = root / "summary.json"
        if not summ.exists():
            ax_l.text(0.5, 0.5, f"missing summary.json",
                      ha="center", va="center", transform=ax_l.transAxes)
            continue
        data = json.load(summ.open())
        variants = data.get("variants", {})
        mesh_name, mesh_rms = _best_mesh_variant(variants)
        grid_rms = variants.get("grid_uniform", {}).get("rms_err")
        if isinstance(grid_rms, list): grid_rms = grid_rms[-1]
        mesh_wall = _walltime_for(root, mesh_name) if mesh_name else None
        grid_wall = _walltime_for(root, "grid_uniform")

        pt_labels = [f"mesh · {mesh_name}\n(best HCT variant)",
                     "grid_uniform"]
        pt_vals   = [mesh_rms, grid_rms]
        pt_walls  = [mesh_wall, grid_wall]
        pt_colours = ["#5289c7", "#3f8f3f"]

        # Give the y-axis 25% headroom so bar-top annotations don't
        # collide with the panel title.
        y_max = max(v for v in pt_vals if v is not None) * 1.35
        ax_l.set_ylim(0, y_max)
        bars = ax_l.bar(np.arange(2), pt_vals, 0.6, color=pt_colours,
                        edgecolor="black", linewidth=0.4)
        for bar, val, w in zip(bars, pt_vals, pt_walls):
            head = f"{val:.3e} m"
            tail = f"\nwall: {w:.0f} s" if w is not None else ""
            ax_l.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + y_max * 0.02,
                      head + tail,
                      ha="center", va="bottom", fontsize=9)
        ax_l.set_xticks(np.arange(2))
        ax_l.set_xticklabels(pt_labels, fontsize=8)
        if col == 0:
            ax_l.set_ylabel("Lagrangian PT rms error (m)", fontsize=9)
        ax_l.set_title("(b) Trajectory — PT vs analytic reference",
                       fontsize=10)
        ax_l.grid(axis="y", alpha=0.3)
        # Ratios as a callout inside the panel, bottom-right so it
        # doesn't fight with the bar tops.
        if mesh_rms and grid_rms and mesh_wall and grid_wall:
            acc = mesh_rms / grid_rms
            spd = mesh_wall / grid_wall
            ax_l.text(0.98, 0.05,
                      f"grid_uniform is\n{acc:.1f}× more accurate\n"
                      f"and {spd:.0f}× faster",
                      transform=ax_l.transAxes, va="bottom", ha="right",
                      fontsize=9, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.4",
                                facecolor="#eef5ea",
                                edgecolor="#3f8f3f"))

    fig.suptitle("Grid-based PT vs mesh-based PT — "
                 "velocity projection + trajectory error, side by side",
                 y=1.005, fontsize=11)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"grid_vs_mesh_pt.{ext}",
                    dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'grid_vs_mesh_pt.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
