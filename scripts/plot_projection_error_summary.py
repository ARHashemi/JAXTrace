"""Plot projection-error summary CSVs as bar charts.

Reads:
    rom_out/analytic_grid_projection/projection_summary_<case>.csv

Writes:
    paper_figs/rom_pt_analytic/projection_error_bars.{png,svg}

Also produces a region-stratified breakdown (near-feature vs outer)
per case × grid × method by reading the per-cell error VTUs.
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

# Method column order + colour map.  hct_cubic added.
METHODS = [
    ("p1_raw",        "#c14040", "raw P1"),
    ("vertex_taylor", "#f0a020", "vertex_taylor (SPR)"),
    ("hct_cubic",     "#3f8f3f", "HCT-3D cubic"),
]

# Feature-region radius per case (m) used for the region-stratified
# breakdown.  Cells with cx² + cy² ≤ r_feature² are "near feature";
# the rest are "outer".
R_FEATURE_M = {
    "recirc_2026":  0.010,   # matches recirc's Gaussian half-width
    "rot_cyl_2026": 0.010,   # a few cylinder radii out (a' = 0.00391)
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


def _region_stratified_from_vtu(vtu_path: Path, r_feature: float
                                ) -> tuple[float, float, float, float]:
    """Load per-cell |error| from the projection VTU + cell centroids,
    return (rms_near, rms_outer, n_near, n_outer).  NaN entries in the
    error field (mask-by-finer regions) are dropped.
    """
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(vtu_path))
    r.Update()
    ug = r.GetOutput()
    err = vtk_to_numpy(ug.GetCellData().GetArray("proj_error")).astype(np.float64)
    # centroids: for a hex cell, mean of 8 corner points
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
    outer = good & (r_c >  r_feature)
    def _rms(mask):
        if not mask.any():
            return float("nan")
        return float(np.sqrt((err[mask] ** 2).mean()))
    return _rms(near), _rms(outer), int(near.sum()), int(outer.sum())


def plot_aggregate_bars() -> None:
    """The 2-panel bar chart of rel_rms per (case × grid × method)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(CASES), figsize=(13, 5), sharey=False)
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
            offset = (m_i - (n_m - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, color=colour,
                          edgecolor="black", linewidth=0.4, label=label)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height(),
                            f"{val:.3f}%", ha="center", va="bottom",
                            fontsize=7)
        cell_counts = [d.get((g, "p1_raw"), {}).get("n_cells", 0)
                       for g in GRIDS]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\n({c:,} cells)"
                            for g, c in zip(GRIDS, cell_counts)],
                           fontsize=8)
        ax.set_title(case)
        ax.set_ylabel("Projection error rel_rms (% of |v|_max)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Mesh → grid projection error · "
                 "4-lvl FEM mesh sampled at grid cell centres, "
                 "compared to analytic ground truth", y=1.02)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"projection_error_bars.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"wrote {OUT_DIR / 'projection_error_bars.png'}")


def plot_region_stratified() -> None:
    """Per-region rms bar chart: near-feature vs outer for each
    (case × grid × method).  Reveals where HCT actually matters.
    """
    fig, axes = plt.subplots(len(CASES), len(GRIDS),
                             figsize=(11, 3.2 * len(CASES)),
                             sharey="row")
    for row, case in enumerate(CASES):
        r_feat = R_FEATURE_M[case]
        for col, grid in enumerate(GRIDS):
            ax = axes[row, col]
            method_stats: dict[str, tuple[float, float]] = {}
            for method, colour, label in METHODS:
                vtu = (DIAG_DIR /
                       f"projection_error_{case}_{grid}_{method}.vtu")
                if not vtu.exists():
                    continue
                rms_near, rms_outer, n_near, n_outer = \
                    _region_stratified_from_vtu(vtu, r_feat)
                method_stats[method] = (rms_near, rms_outer)
            if not method_stats:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            x = np.arange(len(method_stats))
            width = 0.4
            labels = [dict((m, l) for m, _, l in METHODS)[m]
                      for m in method_stats.keys()]
            colours = [dict((m, c) for m, c, _ in METHODS)[m]
                       for m in method_stats.keys()]
            near_vals = [s[0] for s in method_stats.values()]
            outer_vals = [s[1] for s in method_stats.values()]
            b1 = ax.bar(x - width / 2, near_vals, width,
                        color=colours, edgecolor="#c14040", hatch="//",
                        linewidth=0.4,
                        label=f"near-feature (r ≤ {r_feat*1e3:.0f} mm)")
            b2 = ax.bar(x + width / 2, outer_vals, width,
                        color=colours, edgecolor="black", linewidth=0.4,
                        label="outer")
            for bar, val in zip(b1, near_vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, val,
                            f"{val:.2e}", ha="center", va="bottom",
                            fontsize=7, rotation=90)
            for bar, val in zip(b2, outer_vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, val,
                            f"{val:.2e}", ha="center", va="bottom",
                            fontsize=7, rotation=90)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8, rotation=15,
                               ha="right")
            ax.set_yscale("log")
            ax.grid(axis="y", which="both", alpha=0.3)
            if col == 0:
                ax.set_ylabel(f"{case}\nRMS |error| (m/s)")
            if row == 0:
                ax.set_title(grid)
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.suptitle("Region-stratified projection error · "
                 "near-feature (hatched) vs outer, log scale.  "
                 "HCT matters most where near-feature is much larger "
                 "than outer.", y=1.005)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"projection_error_regions.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"wrote {OUT_DIR / 'projection_error_regions.png'}")


def main() -> int:
    plot_aggregate_bars()
    plot_region_stratified()
    return 0


if __name__ == "__main__":
    sys.exit(main())
