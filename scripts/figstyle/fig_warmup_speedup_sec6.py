#!/usr/bin/env python3
"""
Fig C — Warmed-vs-cold GPU speedup heatmap for sec6.

Renders the R2-7 T1–T5 warmed re-sweep result as a 10×3 heatmap
(rows = meshes, columns = MALMO variants), each cell coloured by the
cold-to-warm speedup ratio.  The cell text shows the ratio in
Nx form.  Meshes where the cold time was already <300 ms cluster
around 1× and are drawn in a pale colour; the meshes where the
paper's headline claims live (Tet1820, Tet2856, Poly940, Poly1560,
Bunny.vertex_multi, FSW) show the 5-9× speedups clearly.

Reads:
    <cold_root>/<mesh>__<variant>/result.json
    <warm_root>/<mesh>__<variant>/result.json

Writes:
    fig_warmup_speedup_sec6.{pdf,png}
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import setup, palette, save, MESH_SHORT


COHORT_ORDER = [
    "FinalFuelPinTet63", "FinalFuelPinTet137", "FinalFuelPinTet298",
    "FinalFuelPinTet1820", "FinalFuelPinTet2856",
    "FinalFuelPinPoly264", "FinalFuelPinPoly940", "FinalFuelPinPoly1560",
    "StanfordBunny_LowPoly",
    "FSW_paper",
]
VARIANT_ORDER = ["aabb", "centroid", "vertex_multi"]
VARIANT_LABEL = {
    "aabb":         r"MALMO$^{\rm A}$",
    "centroid":     r"MALMO$^{\rm C}$",
    "vertex_multi": r"MALMO$^{\rm V}$",
}


def load(cold_root: Path, warm_root: Path):
    """Return {(mesh, variant): (cold_s, warm_s, speedup)}."""
    out = {}
    for m in COHORT_ORDER:
        for v in VARIANT_ORDER:
            cold_f = cold_root / f"{m}__{v}" / "result.json"
            warm_f = warm_root / f"{m}__{v}" / "result.json"
            if not (cold_f.exists() and warm_f.exists()):
                continue
            cold = json.load(open(cold_f))["wall_seconds"]["query_min_of_3"]
            warm = json.load(open(warm_f))["wall_seconds"]["query_min_of_3"]
            out[(m, v)] = (cold, warm, cold / warm)
    return out


def emit(data, out_base: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.colors as mcolors

    setup()

    # Build the 10x3 matrix
    grid = np.full((len(COHORT_ORDER), len(VARIANT_ORDER)), np.nan)
    for i, m in enumerate(COHORT_ORDER):
        for j, v in enumerate(VARIANT_ORDER):
            if (m, v) in data:
                grid[i, j] = data[(m, v)][2]  # speedup

    fig, ax = plt.subplots(figsize=(4.4, 4.6))

    # Custom colormap: pale grey around 1×, deepening to blue as speedup grows.
    # Log-normalise so 1× is neutral, 10× is dark blue.
    from matplotlib.colors import LogNorm
    cmap = LinearSegmentedColormap.from_list(
        "warmup",
        [(1.0, 1.0, 1.0), palette["sky"], palette["blue"], "#003b6f"],
        N=256,
    )
    norm = LogNorm(vmin=1.0, vmax=10.0, clip=True)

    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")

    # Cell text
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isnan(v):
                txt = "—"
                tc = palette["grey"]
            else:
                txt = f"{v:.1f}×"
                # White text on dark cells, black on light
                tc = "white" if v > 3.5 else palette["black"]
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9.0, color=tc)

    # Ticks
    ax.set_xticks(range(len(VARIANT_ORDER)))
    ax.set_xticklabels([VARIANT_LABEL[v] for v in VARIANT_ORDER])
    ax.set_yticks(range(len(COHORT_ORDER)))
    ax.set_yticklabels([MESH_SHORT[m] for m in COHORT_ORDER])
    ax.tick_params(axis="x", top=False, bottom=False, direction="out", length=0)
    ax.tick_params(axis="y", left=False, direction="out", length=0)
    ax.grid(False)

    # Faint grid lines between cells
    for x in np.arange(-0.5, len(VARIANT_ORDER), 1):
        ax.axvline(x, color=palette["grey"], lw=0.3)
    for y in np.arange(-0.5, len(COHORT_ORDER), 1):
        ax.axhline(y, color=palette["grey"], lw=0.3)

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, orientation="vertical",
                        fraction=0.05, pad=0.02, aspect=25)
    cbar.set_label("Cold-to-warm speedup", fontsize=9.0)
    cbar.ax.tick_params(labelsize=8.0)

    fig.tight_layout()
    save(fig, out_base)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold-root",
                    default="/home/arhashemi/fsw-gpu/flash/shared/jax/JAXTrace/malmo_runs_in_mesh")
    ap.add_argument("--warm-root",
                    default="/home/arhashemi/fsw-gpu/flash/shared/jax/JAXTrace/malmo_runs_in_mesh_warmed")
    ap.add_argument("--out",
                    default="/home/arhashemi/Workspace/welding/JAXTrace/fig_warmup_speedup_sec6")
    a = ap.parse_args()

    data = load(Path(a.cold_root), Path(a.warm_root))
    if not data:
        print("no runs found; nothing to plot"); sys.exit(2)
    emit(data, Path(a.out))
