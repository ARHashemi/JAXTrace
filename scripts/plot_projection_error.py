"""Plot mesh->grid projection-error results.

Reads:
    rom_out/analytic_grid_projection/projection_summary_<case>.csv
    rom_out/analytic_grid_projection/projection_error_<case>_<grid>_<method>.vtu

Writes:
    paper_figs/rom_pt_analytic/projection_error_bars.{png,svg}
    paper_figs/rom_pt_analytic/projection_error_regions.{png,svg}

Design goals (after 2026-07-28 feedback):
- All figures render at <=2000x1200 to fit slide layouts and avoid
  Pillow's decompression-bomb threshold.
- Log-scale y-axis so raw P1 (0.06%) and HCT-cubic (1.4%) can share a
  panel without either bar disappearing.
- Region-stratified bars use grouped bars (near vs outer side-by-side)
  so the reader can compare them at a glance.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy


DIAG_DIR = Path("/home/arhashemi/Workspace/welding/JAXTrace/rom_out/analytic_grid_projection")
OUT_DIR = Path("/home/arhashemi/Workspace/welding/JAXTrace/paper_figs/rom_pt_analytic")

CASES = ["recirc_2026", "rot_cyl_2026"]
GRIDS = ["uniform_base", "blockref_2lvl", "blockref_4lvl"]

METHODS = [
    ("p1_raw",        "#c14040", "raw P1"),
    ("vertex_taylor", "#f0a020", "vertex_taylor (SPR)"),
    ("hct_cubic",     "#3f8f3f", "HCT-3D cubic"),
]

R_FEATURE_M = {
    "recirc_2026":  0.010,
    "rot_cyl_2026": 0.010,
}


def _load(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("n_cells", "n_valid"):
            r[k] = int(r[k])
        for k in ("v_scale", "err_rms", "err_mean", "err_max",
                  "err_rel_rms_pct", "wall_time_s"):
            r[k] = float(r[k])
    return rows


# ---------------------------------------------------------------------------
# Aggregate bar chart (log scale)
# ---------------------------------------------------------------------------

def plot_aggregate_bars() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(CASES), figsize=(11, 4.6),
                             sharey=False)
    x = np.arange(len(GRIDS))
    n_m = len(METHODS)
    width = 0.72 / n_m

    for ax, case in zip(axes, CASES):
        p = DIAG_DIR / f"projection_summary_{case}.csv"
        if not p.exists():
            ax.text(0.5, 0.5, f"missing {p.name}", ha="center",
                    va="center", transform=ax.transAxes)
            continue
        rows = _load(p)
        d = {(r["grid"], r["method"]): r for r in rows}
        for m_i, (method, colour, label) in enumerate(METHODS):
            vals = [d.get((g, method), {}).get("err_rel_rms_pct", np.nan)
                    for g in GRIDS]
            # Floor NaN + 0 to a tiny value so log-scale still renders
            # (0% projections from over-resolved meshes are meaningful,
            # not missing data — annotate them explicitly).
            plot_vals = np.array([max(v, 1e-4) if np.isfinite(v)
                                  else np.nan for v in vals])
            offset = (m_i - (n_m - 1) / 2) * width
            bars = ax.bar(x + offset, plot_vals, width, color=colour,
                          edgecolor="black", linewidth=0.4, label=label)
            for bar, val in zip(bars, vals):
                if not np.isfinite(val):
                    continue
                y = bar.get_height()
                text = "≈0" if val < 1e-4 else f"{val:.3f}%"
                ax.text(bar.get_x() + bar.get_width() / 2, y * 1.15,
                        text, ha="center", va="bottom", fontsize=7,
                        rotation=0)

        cell_counts = [d.get((g, "p1_raw"), {}).get("n_cells", 0)
                       for g in GRIDS]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\n({c:,} cells)"
                            for g, c in zip(GRIDS, cell_counts)],
                           fontsize=8)
        ax.set_title(case)
        ax.set_ylabel("Projection error rel_rms (% of |v|_max), log")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 5.0)
        ax.grid(axis="y", which="both", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    fig.suptitle("Mesh → grid projection error · analytic sampled at "
                 "4-lvl mesh nodes, projected to 3 target grids, "
                 "compared to analytic at the same grid centres",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"projection_error_bars.{ext}",
                    dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'projection_error_bars.png'}")


# ---------------------------------------------------------------------------
# Region-stratified bar chart
# ---------------------------------------------------------------------------

def _region_split(vtu_path: Path, r_feature: float
                  ) -> tuple[float, float, int, int]:
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(vtu_path))
    r.Update()
    ug = r.GetOutput()
    err = vtk_to_numpy(ug.GetCellData().GetArray("proj_error")).astype(np.float64)
    n_cells = ug.GetNumberOfCells()
    centroids = np.zeros((n_cells, 3), dtype=np.float64)
    cell = vtk.vtkGenericCell()
    for i in range(n_cells):
        ug.GetCell(i, cell)
        pts = cell.GetPoints()
        c = np.zeros(3)
        for j in range(8):
            p = np.zeros(3); pts.GetPoint(j, p); c += p
        centroids[i] = c / 8.0
    r_c = np.sqrt(centroids[:, 0] ** 2 + centroids[:, 1] ** 2)
    good = np.isfinite(err)
    near = good & (r_c <= r_feature)
    outer = good & (r_c > r_feature)

    def _rms(mask):
        if not mask.any():
            return float("nan")
        return float(np.sqrt((err[mask] ** 2).mean()))
    return _rms(near), _rms(outer), int(near.sum()), int(outer.sum())


def plot_region_stratified() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # cases × grids layout, but constrain figure size so pixel count
    # stays under Pillow's default 178M threshold.
    fig, axes = plt.subplots(len(CASES), len(GRIDS),
                             figsize=(10, 5.6), sharey="row")
    for row, case in enumerate(CASES):
        r_feat = R_FEATURE_M[case]
        for col, grid in enumerate(GRIDS):
            ax = axes[row, col]
            near_by = {}; outer_by = {}
            for method, colour, label in METHODS:
                vtu = (DIAG_DIR /
                       f"projection_error_{case}_{grid}_{method}.vtu")
                if not vtu.exists():
                    continue
                n_rms, o_rms, _, _ = _region_split(vtu, r_feat)
                near_by[method] = n_rms
                outer_by[method] = o_rms
            if not near_by:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            m_keys = list(near_by.keys())
            x = np.arange(len(m_keys))
            width = 0.4
            labels = [dict((m, l) for m, _, l in METHODS)[m] for m in m_keys]
            colours = [dict((m, c) for m, c, _ in METHODS)[m] for m in m_keys]
            near_vals = [max(near_by[m], 1e-6) for m in m_keys]
            outer_vals = [max(outer_by[m], 1e-6) for m in m_keys]
            ax.bar(x - width / 2, near_vals, width, color=colours,
                   edgecolor="black", hatch="//",  linewidth=0.4,
                   label=f"near-feature (r ≤ {r_feat*1e3:.0f} mm)")
            ax.bar(x + width / 2, outer_vals, width, color=colours,
                   edgecolor="black", linewidth=0.4, label="outer")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
            ax.set_yscale("log")
            ax.grid(axis="y", which="both", alpha=0.3)
            if col == 0:
                ax.set_ylabel(f"{case}\nRMS |error| (m/s), log", fontsize=9)
            if row == 0:
                ax.set_title(grid, fontsize=10)
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=7, framealpha=0.95)
    fig.suptitle("Region-stratified projection error · near-feature "
                 "(hatched) vs outer.  HCT helps only where near-feature "
                 "bar is lower.",
                 y=1.005, fontsize=10)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"projection_error_regions.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'projection_error_regions.png'}")


def main() -> int:
    plot_aggregate_bars()
    plot_region_stratified()
    return 0


if __name__ == "__main__":
    sys.exit(main())
