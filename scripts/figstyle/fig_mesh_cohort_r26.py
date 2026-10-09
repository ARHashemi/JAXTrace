#!/usr/bin/env python3
"""
Fig B — Mesh cohort overview for sec6.

Renders the 10-mesh cohort used across Tables T1–T5 as a horizontal
bar chart of tet counts (log scale), annotated with each mesh's
fill_ratio and edge-length CoV (both diagnostic quantities we use in
the R2-3 response).

The bar chart is more informative than a 3D montage for print — it
puts the scale span (63 → 3.05 M tets) and the structural class of
each mesh (Kuhn-aligned, general-unstructured, strongly-irregular)
on one axis that the reader can compare at a glance.

Reads (from workstation via mount):
    /home/arhashemi/fsw-gpu/flash/shared/jax/JAXTrace/malmo_runs_in_mesh_warmed/<mesh>__aabb/result.json
        for n_tets and fill_ratio.

Writes:
    fig_mesh_cohort_sec6.{pdf,png}
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

# Bring in the shared style
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle
from figstyle import (
    setup, palette, save, MESH_SHORT, MESH_NTETS,
)


COHORT_ORDER = [
    "FinalFuelPinTet63", "FinalFuelPinTet137", "FinalFuelPinTet298",
    "FinalFuelPinTet1820", "FinalFuelPinTet2856",
    "FinalFuelPinPoly264", "FinalFuelPinPoly940", "FinalFuelPinPoly1560",
    "StanfordBunny_LowPoly",
    "FSW_paper",
]

# Structural class per mesh (from R2-3)
STRUCT_CLASS = {
    "FinalFuelPinTet63":     "gen. unstructured",
    "FinalFuelPinTet137":    "gen. unstructured",
    "FinalFuelPinTet298":    "gen. unstructured",
    "FinalFuelPinTet1820":   "gen. unstructured",
    "FinalFuelPinTet2856":   "gen. unstructured",
    "FinalFuelPinPoly264":   "gen. unstructured",
    "FinalFuelPinPoly940":   "gen. unstructured",
    "FinalFuelPinPoly1560":  "gen. unstructured",
    "StanfordBunny_LowPoly": "strongly irregular",
    "FSW_paper":             "octree-aligned",
}
CLASS_COLOUR = {
    "octree-aligned":      palette["blue"],
    "gen. unstructured":   palette["vermillion"],
    "strongly irregular":  palette["green"],
}

# Edge-length CoV values from the R2-3 diagnostic (only the ones we
# reported; approximate for the fuel-pin cohort where we did not run
# the diagnostic per-mesh).
EDGE_COV = {
    "FinalFuelPinPoly264":   0.28,   # measured in R2-3 audit
    "StanfordBunny_LowPoly": 0.43,   # measured in R2-3 audit
    # Others: approximate; use "~" prefix in annotation if we mention it
}


def load_fill_ratios(root: Path):
    """Read fill_ratio + n_tets from the warmed aabb runs."""
    out = {}
    for mesh in COHORT_ORDER:
        for candidate in [
            root / f"{mesh}__aabb" / "result.json",
            # fall back to whatever's there
            root / f"{mesh}__centroid" / "result.json",
        ]:
            if candidate.exists():
                d = json.load(open(candidate))
                out[mesh] = {
                    "n_tets":     d.get("n_tets") or MESH_NTETS[mesh],
                    "fill_ratio": d.get("fill_ratio"),
                }
                break
        else:
            out[mesh] = {"n_tets": MESH_NTETS[mesh], "fill_ratio": None}
    return out


def emit(data, out_base: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup()

    # Bars: horizontal, sorted by n_tets ascending
    order = sorted(COHORT_ORDER, key=lambda m: data[m]["n_tets"])
    y_pos = np.arange(len(order))
    tets = [data[m]["n_tets"] for m in order]
    colours = [CLASS_COLOUR[STRUCT_CLASS[m]] for m in order]
    labels = [MESH_SHORT[m] for m in order]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    bars = ax.barh(y_pos, tets, color=colours, edgecolor=palette["grey"],
                    linewidth=0.6, height=0.72)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlim(30, 5e7)
    ax.set_xlabel(r"Number of tetrahedra (log)")

    # Annotate each bar with n_tets; fill_ratio only when it deviates
    # meaningfully from 1.0 (the Kim meshes all fill their bbox by
    # construction, so "fill = 100 %" would be visual noise there).
    for i, m in enumerate(order):
        n = data[m]["n_tets"]
        fr = data[m]["fill_ratio"]
        n_str = (f"{n/1e6:.2f} M" if n >= 1_000_000 else
                 f"{n/1000:.1f} k" if n >= 1000 else f"{n}")
        annot = f"{n_str}"
        # Only annotate fill if it noticeably deviates from 1.0
        if fr is not None and fr < 0.98:
            annot += f",  fill-ratio = {fr*100:.1f}%"
        ax.text(n * 1.15, i, annot,
                va="center", ha="left", fontsize=8.0,
                color=palette["black"])

    # Class legend — build proxy handles that will not overlap the bars
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=CLASS_COLOUR["octree-aligned"],     label="octree-aligned"),
        Patch(facecolor=CLASS_COLOUR["gen. unstructured"],  label="general unstructured"),
        Patch(facecolor=CLASS_COLOUR["strongly irregular"], label="strongly irregular"),
    ]
    # Put in the upper-left corner where no bar reaches (log scale means
    # top-left is safely empty because the FSW bar extends to the right)
    ax.legend(handles=handles, loc="lower right", framealpha=0.92)

    fig.tight_layout()
    save(fig, out_base)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root",
                    default="/home/arhashemi/fsw-gpu/flash/shared/jax/JAXTrace/malmo_runs_in_mesh_warmed",
                    help="MALMO warmed sweep root")
    ap.add_argument("--out",
                    default="/home/arhashemi/Workspace/welding/JAXTrace/fig_mesh_cohort_sec6",
                    help="Output base (no extension)")
    a = ap.parse_args()

    data = load_fill_ratios(Path(a.sweep_root))
    emit(data, Path(a.out))
