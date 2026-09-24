"""
Shared style for CMAME paper figures.

Design principles (baked in per user instruction, 2026-09-04):
    - No figure titles (the LaTeX caption is the title).
    - Font sizes readable in print without being ugly: 9-10 pt axis
      labels, 8-9 pt tick labels, 9 pt legend, 300+ dpi PDF output.
    - Legends must NOT overlap any data; we use `_place_legend()`
      which picks the emptiest corner automatically.
    - Colour-blind-safe palette (Wong 2011): distinguishable by both
      hue and value for the common deuteranope / protanope cases.
    - Vector output (PDF) preferred over raster.

Usage:
    from figstyle import setup, palette, place_legend, save

    setup()                     # applies rcParams globally
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(x, y, color=palette["blue"], label=...)
    place_legend(ax)            # picks the emptiest corner
    save(fig, "fig_out")        # writes .pdf and .png at 300 dpi
"""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path


# ---- Wong 2011 colour-blind-safe palette (used across all figures) --------
palette = {
    "blue":       "#0072B2",   # MALMO variant A
    "vermillion": "#D55E00",   # MALMO variant C
    "green":      "#009E73",   # MALMO variant V
    "orange":     "#E69F00",   # SCONE
    "purple":     "#CC79A7",   # RTXAdvect
    "yellow":     "#F0E442",   # accent
    "sky":        "#56B4E9",   # accent
    "black":      "#000000",   # reference / grid lines
    "grey":       "#666666",   # secondary text / dashed lines
}

# ---- Framework-consistent colour map used across every paper figure ------
framework_colour = {
    "MALMO_A":     palette["blue"],
    "MALMO_C":     palette["vermillion"],
    "MALMO_V":     palette["green"],
    "aabb":        palette["blue"],
    "centroid":    palette["vermillion"],
    "vertex_multi":palette["green"],
    "SCONE":       palette["orange"],
    "RTXAdvect":   palette["purple"],
}

# ---- Mesh short labels + tet counts (mirrors build_paper_tables.py) -------
MESH_SHORT = {
    "FinalFuelPinTet63":     "Tet63",
    "FinalFuelPinTet137":    "Tet137",
    "FinalFuelPinTet298":    "Tet298",
    "FinalFuelPinTet1820":   "Tet1820",
    "FinalFuelPinTet2856":   "Tet2856",
    "FinalFuelPinPoly264":   "Poly264",
    "FinalFuelPinPoly940":   "Poly940",
    "FinalFuelPinPoly1560":  "Poly1560",
    "StanfordBunny_LowPoly": "Bunny",
    "FSW_paper":             "FSW",
}
MESH_NTETS = {
    "FinalFuelPinTet63":         63,
    "FinalFuelPinTet137":       137,
    "FinalFuelPinTet298":       298,
    "FinalFuelPinTet1820":     1820,
    "FinalFuelPinTet2856":     2856,
    "FinalFuelPinPoly264":     1404,
    "FinalFuelPinPoly940":     5220,
    "FinalFuelPinPoly1560":    8712,
    "StanfordBunny_LowPoly":    379,
    "FSW_paper":            3050196,
}


def setup():
    """Apply the shared rcParams.  Call once at the top of any fig script."""
    mpl.rcParams.update({
        # Fonts (sans-serif everywhere; Helvetica-lookalike stack)
        "font.family":         "sans-serif",
        "font.sans-serif":     ["Helvetica", "Nimbus Sans", "Arial",
                                "DejaVu Sans", "Liberation Sans"],
        "font.size":            9.0,
        "axes.labelsize":      10.0,
        "axes.titlesize":       0,    # titles disabled -> forced empty
        "xtick.labelsize":      8.5,
        "ytick.labelsize":      8.5,
        "legend.fontsize":      9.0,

        # No axis titles; captions do that in LaTeX
        "axes.titlepad":        0,

        # Frame
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.linewidth":       0.9,
        "axes.edgecolor":       "#333333",

        # Grid (light, always dotted, low z)
        "axes.grid":           True,
        "grid.linestyle":      ":",
        "grid.alpha":           0.45,
        "grid.linewidth":       0.6,

        # Ticks
        "xtick.direction":     "in",
        "ytick.direction":     "in",
        "xtick.major.size":     3.0,
        "ytick.major.size":     3.0,
        "xtick.minor.size":     1.5,
        "ytick.minor.size":     1.5,
        "xtick.major.width":    0.7,
        "ytick.major.width":    0.7,

        # Lines
        "lines.linewidth":      1.6,
        "lines.markersize":     5.0,
        "lines.markeredgewidth": 0.9,

        # Legend (frame, but subtle — placement decided by place_legend)
        "legend.frameon":       True,
        "legend.framealpha":    0.92,
        "legend.edgecolor":     "#CCCCCC",
        "legend.borderaxespad": 0.4,
        "legend.borderpad":     0.4,
        "legend.handletextpad": 0.5,
        "legend.handlelength":  1.8,
        "legend.labelspacing":  0.35,

        # Output
        "figure.dpi":           120,   # screen preview
        "savefig.dpi":          320,   # print (CMAME requires ≥300)
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.02,
        "pdf.fonttype":         42,    # TrueType embed
        "ps.fonttype":          42,
    })


def _no_title(ax):
    """Enforce no title even if the caller sets one."""
    ax.set_title("")


def place_legend(ax, prefer=("upper right", "upper left",
                              "lower right", "lower left"),
                  **kwargs):
    """Pick the emptiest corner of the axes for the legend.

    Heuristic: for each candidate corner, compute the density of
    plotted data in the 30% x 30% axes-fraction rectangle rooted at
    that corner; place the legend in the corner with the least data.

    Falls back to `prefer[0]` if all corners are similarly dense.
    """
    import numpy as np
    corners = {
        "upper right": (0.70, 0.70),
        "upper left":  (0.00, 0.70),
        "lower right": (0.70, 0.00),
        "lower left":  (0.00, 0.00),
    }
    # Collect all plotted (x, y) pairs in axes-fraction coordinates.
    xy_ax = []
    for ln in ax.get_lines():
        try:
            xs, ys = ln.get_data()
        except Exception:
            continue
        for xd, yd in zip(xs, ys):
            xa, ya = ax.transLimits.transform((xd, yd))
            xy_ax.append((xa, ya))

    if not xy_ax:
        loc = prefer[0]
    else:
        arr = np.array(xy_ax)
        best_loc, best_density = prefer[0], float("inf")
        for name in prefer:
            x0, y0 = corners[name]
            in_box = ((arr[:, 0] >= x0) & (arr[:, 0] <= x0 + 0.30) &
                      (arr[:, 1] >= y0) & (arr[:, 1] <= y0 + 0.30))
            density = int(in_box.sum())
            if density < best_density:
                best_density, best_loc = density, name
        loc = best_loc

    return ax.legend(loc=loc, **kwargs)


def save(fig, out_base, formats=("pdf", "png"), **kwargs):
    """Save the figure to <out_base>.<fmt> for each requested format.

    out_base may be a Path or string.  Kwargs forwarded to `savefig`.
    """
    out = Path(out_base)
    out.parent.mkdir(parents=True, exist_ok=True)
    for f in fig.axes:
        _no_title(f)
    for fmt in formats:
        p = out.with_suffix(f".{fmt}")
        fig.savefig(p, **kwargs)
        print(f"  wrote {p}")
