#!/usr/bin/env python3
"""
Fig B -- Mesh cohort overview for sec6.

Renders the FIVE-mesh benchmark cohort of Table~\ref{tab:mesh_cohort}
as a horizontal bar chart of tet counts (log scale), annotated with
fill ratio and non-Kuhn fraction and coloured by structural class.

The bar chart puts the scale span (379 -> 3.05 M tets) and the
structural class of each mesh on one axis the reader can compare at a
glance.

Values are those of tab:mesh_cohort in sec6_validation.tex; they are
kept here explicitly so the figure cannot drift from the table.

Writes:  fig_mesh_cohort_sec6.{pdf,png}
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import setup, palette, save

# label : (n_tets, fill_ratio, non_kuhn_frac, class)
COHORT = [
    ("Bunny",         379,        0.272, 1.000, "strongly irregular"),
    ("FP-8.7k",       8_712,      1.000, 0.938, "general unstructured"),
    ("Porous media",  819_726,    0.659, 1.000, "pore-scale"),
    ("Microfluidics", 1_659_240,  0.664, 1.000, "extruded slab"),
    ("FSW",           3_050_196,  1.000, 0.0006, "octree-aligned"),
]

CLASS_COLOUR = {
    "octree-aligned":       palette["blue"],
    "general unstructured": palette["vermillion"],
    "strongly irregular":   palette["green"],
    "pore-scale":           palette["orange"],
    "extruded slab":        palette["purple"],
}


def emit(out_base: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    setup()
    order = sorted(COHORT, key=lambda r: r[1])
    y = np.arange(len(order))
    tets = [r[1] for r in order]
    colours = [CLASS_COLOUR[r[4]] for r in order]
    labels = [r[0] for r in order]

    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ax.barh(y, tets, color=colours, edgecolor=palette["grey"],
            linewidth=0.6, height=0.68)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlim(200, 2e8)
    ax.set_xlabel("Number of tetrahedra (log scale)")

    for i, (name, n, fill, nonkuhn, cls) in enumerate(order):
        n_str = (f"{n/1e6:.2f} M" if n >= 1_000_000 else
                 f"{n/1000:.1f} k" if n >= 1000 else f"{n}")
        parts = [n_str]
        if fill < 0.98:
            parts.append(f"fill {fill*100:.1f}%")
        # flag the one mesh that is essentially all-Kuhn
        if nonkuhn < 0.01:
            parts.append("Kuhn")
        ax.text(n * 1.25, i, ",  ".join(parts),
                va="center", ha="left", fontsize=8.0,
                color=palette["black"])

    seen, handles = set(), []
    for _, _, _, _, cls in order:
        if cls not in seen:
            seen.add(cls)
            handles.append(Patch(facecolor=CLASS_COLOUR[cls], label=cls))
    ax.legend(handles=handles, loc="lower right", frameon=False,
              fontsize=7.5, ncol=2, handlelength=1.2)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", which="major", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, out_base)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fig_mesh_cohort_sec6")
    emit(out)
    print("wrote", out.with_suffix(".pdf"))
