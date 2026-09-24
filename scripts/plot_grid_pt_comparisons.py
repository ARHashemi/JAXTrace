"""Three focused comparisons of grid PT vs mesh PT.

Reads:
  /scratch/shared/ROM/FOM_analytic/<case>/summary.json      (per-variant + per-step PT error)
  /scratch/shared/ROM/FOM_analytic/<case>/out_*/run_*.log   (per-variant wall time)
  rom_out/analytic_grid_projection/projection_summary_<case>.csv  (Eulerian projection error)

Writes:
  paper_figs/rom_pt_analytic/grid_vs_mesh_time.{png,svg}
  paper_figs/rom_pt_analytic/grid_vs_mesh_velocity.{png,svg}
  paper_figs/rom_pt_analytic/grid_vs_mesh_pt_step2500.{png,svg}
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


CASES = ["recirc_2026", "rot_cyl_2026"]
CASE_ROOTS = {c: Path(f"/home/arhashemi/fsw-gpu/scratch/shared/ROM/FOM_analytic/{c}")
              for c in CASES}
DIAG_DIR = Path("/home/arhashemi/Workspace/welding/JAXTrace/rom_out/analytic_grid_projection")
OUT_DIR = Path("/home/arhashemi/Workspace/welding/JAXTrace/paper_figs/rom_pt_analytic")

GRID_TYPES = ["uniform", "2lvl", "4lvl"]

# Available mesh recovery methods vary by case (rot_cyl only has HCT).
# The plot handles missing gracefully.
MESH_METHODS = ["raw_P1", "vertex_taylor", "hct_cubic"]

# Mesh method name resolver: summary.json uses labels like
# "uniform" (raw P1), "uniform_vertex_taylor", "uniform_hct_cubic".
def _mesh_variant_name(grid_type: str, method: str) -> str:
    if method == "raw_P1":
        return grid_type
    return f"{grid_type}_{method}"

_WALL_RE = re.compile(r"Wall time:\s+([\d.]+)s")


def _wall_from_log(p: Path) -> float | None:
    if not p.exists():
        return None
    m = _WALL_RE.findall(p.read_text())
    return float(m[-1]) if m else None


def _walltime_for(case_root: Path, variant: str) -> float | None:
    """variant is either 'grid_<type>' or a mesh label like 'uniform_hct_cubic'."""
    if variant.startswith("grid_"):
        d = case_root / f"out_grid_{variant[len('grid_'):]}"
        for c in d.glob("run_grid.log"): return _wall_from_log(c)
        for c in d.glob("*.log"):
            w = _wall_from_log(c)
            if w is not None: return w
        return None
    d = case_root / f"out_mesh_{variant}"
    for c in d.glob("run_mesh_*.log"): return _wall_from_log(c)
    return None


def _load_summary(case: str) -> dict:
    with (CASE_ROOTS[case] / "summary.json").open() as f:
        return json.load(f)


def _load_projection_csv(case: str) -> list[dict]:
    p = DIAG_DIR / f"projection_summary_{case}.csv"
    if not p.exists():
        return []
    with p.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("err_rel_rms_pct", "err_rms", "v_scale"):
            r[k] = float(r[k])
    return rows


def _step_index(summary: dict, target_step: float) -> int | None:
    markers = summary.get("step_markers", [])
    if target_step in markers:
        return markers.index(target_step)
    if not markers:
        return None
    diffs = [abs(m - target_step) for m in markers]
    i = min(range(len(diffs)), key=lambda k: diffs[k])
    return i if diffs[i] < 1 else None


# ---------------------------------------------------------------------------
# Figure 1: Wall time — grid vs mesh, per grid type per case
# ---------------------------------------------------------------------------

def plot_time_comparison() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    for ax, case in zip(axes, CASES):
        root = CASE_ROOTS[case]
        # Mesh wall times (HCT-cubic variant since that's the one on
        # every case, and it's the one Slide 12 uses as best mesh).
        mesh_walls = [_walltime_for(root, f"{g}_hct_cubic") for g in GRID_TYPES]
        grid_walls = [_walltime_for(root, f"grid_{g}") for g in GRID_TYPES]

        x = np.arange(len(GRID_TYPES))
        width = 0.38
        ax.bar(x - width/2, mesh_walls, width, color="#5289c7",
               edgecolor="black", label="mesh + HCT-3D")
        ax.bar(x + width/2, grid_walls, width, color="#3f8f3f",
               edgecolor="black", label="grid PT")
        # labels on the top of each bar (mesh + grid)
        for xi, w in zip(x - width/2, mesh_walls):
            if w:
                ax.text(xi, w + max(mesh_walls) * 0.02, f"{w:.0f} s",
                        ha="center", va="bottom", fontsize=9)
        for xi, w in zip(x + width/2, grid_walls):
            if w:
                ax.text(xi, w + max(mesh_walls) * 0.02, f"{w:.0f} s",
                        ha="center", va="bottom", fontsize=9)
        # ratio callout INSIDE the mesh bar (bottom), so it doesn't
        # collide with the panel title or the wall-time labels
        for xi, m, g in zip(x, mesh_walls, grid_walls):
            if m and g:
                ratio = m / g
                ax.text(xi - width/2, m * 0.05, f"{ratio:.0f}×\nfaster",
                        ha="center", va="bottom", fontsize=9,
                        fontweight="bold", color="white")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}" for g in GRID_TYPES], fontsize=10)
        ax.set_ylabel("wall time (s)")
        ax.set_title(case)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
        ax.set_ylim(0, max(mesh_walls) * 1.20)
    fig.suptitle("PT wall time — grid vs corresponding mesh (HCT-3D) "
                 "at three refinement levels", y=1.02)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"grid_vs_mesh_time.{ext}", dpi=180,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'grid_vs_mesh_time.png'}")


# ---------------------------------------------------------------------------
# Figure 2: Eulerian velocity — projected grid vs analytic (raw P1 only,
# same as Slide 10 but simplified to the row we actually care about)
# ---------------------------------------------------------------------------

def plot_velocity_comparison() -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x_labels = []
    values = {"recirc_2026": [], "rot_cyl_2026": []}
    for case in CASES:
        rows = _load_projection_csv(case)
        d = {(r["grid"], r["method"]): r for r in rows}
        for g in GRID_TYPES:
            key = f"blockref_{g}" if g != "uniform" else "uniform_base"
            r = d.get((key, "p1_raw"), {})
            values[case].append(r.get("err_rel_rms_pct", np.nan))

    x = np.arange(len(GRID_TYPES))
    width = 0.38
    ax.bar(x - width/2, values["recirc_2026"], width,
           color="#c14040", edgecolor="black", label="recirc_2026")
    ax.bar(x + width/2, values["rot_cyl_2026"], width,
           color="#5289c7", edgecolor="black", label="rot_cyl_2026")
    for xi, v in zip(x - width/2, values["recirc_2026"]):
        if np.isfinite(v):
            ax.text(xi, max(v, 1e-4) * 1.35,
                    "≈ 0 %" if v < 1e-4 else f"{v:.3f} %",
                    ha="center", fontsize=9)
    for xi, v in zip(x + width/2, values["rot_cyl_2026"]):
        if np.isfinite(v):
            ax.text(xi, max(v, 1e-4) * 1.35,
                    "≈ 0 %" if v < 1e-4 else f"{v:.3f} %",
                    ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}" for g in GRID_TYPES])
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 5.0)
    ax.set_ylabel("Eulerian projection rel_rms (% of |v|_max), log")
    ax.set_title("Velocity projection accuracy — mesh → grid (raw P1) "
                 "vs analytic reference")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"grid_vs_mesh_velocity.{ext}", dpi=180,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'grid_vs_mesh_velocity.png'}")


# ---------------------------------------------------------------------------
# Figure 3: PT accuracy at step 2500 — grid + mesh(HCT) per grid type
# ---------------------------------------------------------------------------

def plot_pt_step_comparison(target_step: float = 2500.0) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    for ax, case in zip(axes, CASES):
        summ = _load_summary(case)
        step_idx = _step_index(summ, target_step)
        if step_idx is None:
            ax.text(0.5, 0.5, f"step {target_step} not in summary",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        variants = summ["variants"]
        mesh_rms = [variants.get(f"{g}_hct_cubic", {}).get("rms_err", [np.nan])[step_idx]
                    if isinstance(variants.get(f"{g}_hct_cubic", {}).get("rms_err"), list)
                    else np.nan
                    for g in GRID_TYPES]
        grid_rms = [variants.get(f"grid_{g}", {}).get("rms_err", [np.nan])[step_idx]
                    if isinstance(variants.get(f"grid_{g}", {}).get("rms_err"), list)
                    else np.nan
                    for g in GRID_TYPES]
        x = np.arange(len(GRID_TYPES))
        width = 0.38
        ax.bar(x - width/2, mesh_rms, width, color="#5289c7",
               edgecolor="black", label="mesh + HCT-3D")
        ax.bar(x + width/2, grid_rms, width, color="#3f8f3f",
               edgecolor="black", label="grid PT")
        # numeric labels on top of each bar
        for xi, v in zip(x - width/2, mesh_rms):
            if np.isfinite(v):
                ax.text(xi, v * 1.15, f"{v:.2e}", ha="center",
                        va="bottom", fontsize=8)
        for xi, v in zip(x + width/2, grid_rms):
            if np.isfinite(v):
                ax.text(xi, v * 1.15, f"{v:.2e}", ha="center",
                        va="bottom", fontsize=8)
        # Ratio annotations placed just below the panel title as an
        # x-tick suffix (grid_type + ratio on one line under each x tick).
        # This keeps the callout inside the panel's horizontal extent.
        xtick_lines = []
        for gt, m, g in zip(GRID_TYPES, mesh_rms, grid_rms):
            if np.isfinite(m) and np.isfinite(g) and g > 0:
                r = m / g
                if r >= 1:
                    tag = f"grid {r:.1f}× better"
                    colour = "#2c7a2c"
                else:
                    tag = f"grid {1/r:.2f}× worse"
                    colour = "#a03030"
            else:
                tag = "—"; colour = "black"
            xtick_lines.append((gt, tag, colour))
        ax.set_xticks(x)
        ax.set_xticklabels([f"{gt}\n" for gt, _, _ in xtick_lines],
                           fontsize=10)
        # add the ratio tag as a second-line xtick via text annotations
        # (mpl xticklabel newlines get one colour only; use text
        # annotations to keep the green/red distinction).
        ymin, ymax = ax.get_ylim()  # log scale, defer until after ylim set
        # (do coloured tags after ylim is finalised, below)
        ax.set_yscale("log")
        ax.set_ylabel("PT trajectory rms error (m), log")
        ax.set_title(case)
        ax.grid(axis="y", which="both", alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        # give the y-axis a bit of headroom on the log scale so labels fit
        y_max = max(v for v in list(mesh_rms) + list(grid_rms)
                    if np.isfinite(v))
        y_min = min(v for v in list(mesh_rms) + list(grid_rms)
                    if np.isfinite(v) and v > 0)
        ax.set_ylim(y_min * 0.4, y_max * 3)
        # Colour-coded ratio tag under each x tick (below the group label)
        for xi, (_gt, tag, colour) in zip(x, xtick_lines):
            ax.annotate(tag,
                        xy=(xi, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -32), textcoords="offset points",
                        ha="center", va="top",
                        fontsize=9, fontweight="bold", color=colour)
    fig.suptitle(f"PT trajectory accuracy at step {int(target_step)} · "
                 f"grid vs mesh (HCT-3D) per grid type",
                 y=1.02)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"grid_vs_mesh_pt_step{int(target_step)}.{ext}",
                    dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / f'grid_vs_mesh_pt_step{int(target_step)}.png'}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_time_comparison()
    plot_velocity_comparison()
    plot_pt_step_comparison(2500.0)
    plot_pt_step_comparison(6000.0)   # also emit final step for reference
    return 0


if __name__ == "__main__":
    sys.exit(main())
