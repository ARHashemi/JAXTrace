"""Schematic figures — what the quantities MEAN, not what they measured.

    python3 make_concept_figures.py [--out figs_concept]

Companion to `CARDS_one_concept_each.md`. Each figure explains one concept that is
hard to hold in words alone. Drawn, not measured -- except fig C4, which uses the
real A1 tool STL so the geometry is honest.

  figC1_geometry     the tool, the plate, and WHERE advancing/retreating are
  figC2_stress_split stress = volume part + shape part, and why that matters
  figC3_triaxiality  how the sign of eta decides open vs close
  figC4_pin_shapes   the four pin families, from the actual STL files
  figC5_pathline     how damage accumulates ALONG a route, not at a point
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Wedge

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"      # compression / closing / safe
RED = "#e34948"       # tension / opening / risk
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
TOOL = "#9a9a95"

STL_ROOT = Path("/home/arhashemi/lumi/lumi_scratch/lorenzgl/Cases/PinShapes")


def save(fig, out: Path, name: str):
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=170, bbox_inches="tight",
                    facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {name}.png / .svg")


def bare(ax):
    ax.set_facecolor(SURFACE)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_aspect("equal")


# ── C1 — the geometry, and which side is which ─────────────────────────────
def figC1_geometry(out):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 5.0))

    # --- plan view ---
    bare(ax)
    ax.set_title("Plan view (looking down)", color=INK, fontsize=11.5, loc="left")
    ax.add_patch(Rectangle((-20, -13), 40, 26, facecolor="#f0efe9",
                           edgecolor=GRID, lw=1.2))
    ax.add_patch(Circle((0, 0), 7.0, facecolor=TOOL, edgecolor=INK2, lw=1.2,
                        alpha=0.55))
    ax.add_patch(Circle((0, 0), 2.4, facecolor=TOOL, edgecolor=INK2, lw=1.4))
    ax.text(0, 0, "pin", ha="center", va="center", fontsize=8, color=INK)
    ax.text(5.0, 5.0, "shoulder", fontsize=8, color=INK2)

    # tool travel and rotation
    ax.add_patch(FancyArrowPatch((6, 11), (-6, 11), arrowstyle="-|>",
                                 mutation_scale=15, color=INK, lw=2))
    ax.text(0, 12.0, "tool travels", ha="center", fontsize=9, color=INK)
    ax.add_patch(Wedge((0, 0), 8.6, 35, 145, width=0.25, facecolor=INK2))
    ax.add_patch(FancyArrowPatch((-6.1, 6.1), (-7.0, 4.9), arrowstyle="-|>",
                                 mutation_scale=13, color=INK2, lw=1.6))
    ax.text(-8.6, 7.2, "rotation", fontsize=9, color=INK2)

    # the two flanks. With the tool travelling -x and rotating CCW, the surface
    # velocity ADDS to travel on +y: that is the advancing side.
    ax.annotate("ADVANCING\nrotation + travel add\n→ defects form here",
                (0, 7.0), xytext=(9.5, 7.4), fontsize=9.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.annotate("RETREATING\nthey oppose",
                (0, -7.0), xytext=(9.5, -9.0), fontsize=9.5, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    ax.plot([-20, 20], [0, 0], color=GRID, lw=1.0, zorder=0)
    ax.text(-18.5, 1.0, "wake  →", fontsize=8.5, color=INK2)
    ax.text(14.5, 1.0, "← ahead", fontsize=8.5, color=INK2)
    ax.set_xlim(-21, 21); ax.set_ylim(-14, 14)

    # --- section view ---
    bare(ax2)
    ax2.set_title("Section through the tool", color=INK, fontsize=11.5,
                  loc="left")
    ax2.add_patch(Rectangle((-14, -6), 28, 6, facecolor="#f0efe9",
                            edgecolor=GRID, lw=1.2))
    ax2.text(-13.0, -5.4, "plate, 6 mm", fontsize=8.5, color=INK2)
    ax2.add_patch(Rectangle((-7, 0), 14, 1.6, facecolor=TOOL,
                            edgecolor=INK2, lw=1.2))
    ax2.add_patch(Rectangle((-2.4, -5.4), 4.8, 5.4, facecolor=TOOL,
                            edgecolor=INK2, lw=1.4))
    ax2.text(0, 0.8, "shoulder  r = 7 mm", ha="center", fontsize=8, color=INK)
    ax2.text(0, -2.8, "pin\nr ≈ 2.4 mm", ha="center", va="center",
             fontsize=8, color=INK)

    # the shear layer, which is where the physics lives
    for sgn in (-1, 1):
        ax2.add_patch(Rectangle((sgn * 2.4, -5.4), sgn * 0.9, 5.4,
                                facecolor=ORANGE, alpha=0.5, edgecolor="none"))
    ax2.annotate("shear layer\nδ ≈ 0.5–2.8 mm\n(where the physics happens)",
                 (3.1, -3.0), xytext=(6.0, -4.6), fontsize=9, color=ORANGE,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))
    # ⚠️ The shoulder/pin radius distinction is labelled on the drawing itself
    # (both are dimensioned), so no warning annotation is needed. An earlier
    # version carried an internal note about the level-set radius here; that is a
    # tooling detail, not process geometry, and does not belong in a figure used
    # to introduce the process.
    # Plain caption, no leader line: an arrow into the section drawing crosses the
    # pin and shoulder labels that are already dimensioned on it.
    ax2.text(-14.0, 4.6,
             "shoulder contacts the top surface; the pin stirs the full depth",
             fontsize=9, color=INK2, va="top", ha="left")
    ax2.set_xlim(-15, 15); ax2.set_ylim(-7, 6.2)
    save(fig, out, "figC1_geometry")


# ── C2 — stress splits into two parts ──────────────────────────────────────
def figC2_stress_split(out):
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.0))
    titles = ["total stress  σ", "volume part  σₘ δ", "shape part  s"]
    notes = ["what the material feels",
             "squeeze / pull\n→ opens or closes VOIDS",
             "distortion\n→ makes metal FLOW"]
    for ax, t, n in zip(axes, titles, notes):
        bare(ax)
        ax.set_title(t, color=INK, fontsize=11.5)
        ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.4, 2.2)
        ax.text(0, -2.2, n, ha="center", fontsize=9.5, color=INK2)

    # total: a square with mixed arrows
    ax = axes[0]
    ax.add_patch(Rectangle((-1, -1), 2, 2, facecolor="#f0efe9",
                           edgecolor=INK2, lw=1.4))
    for (x, y, dx, dy) in [(0, 1.05, 0, .55), (0, -1.05, 0, -.55),
                           (1.05, 0, .55, 0), (-1.05, 0, -.55, 0)]:
        ax.add_patch(FancyArrowPatch((x, y), (x + dx, y + dy),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color=INK, lw=1.8))
    for s in (-1, 1):
        ax.add_patch(FancyArrowPatch((-0.75 * s, 1.05 * s), (0.75 * s, 1.05 * s),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color=ORANGE, lw=1.6))

    # volume part: all arrows the same, pointing out (tension) or in
    ax = axes[1]
    ax.add_patch(Rectangle((-1, -1), 2, 2, facecolor="#fde8e8",
                           edgecolor=RED, lw=1.4))
    for (x, y, dx, dy) in [(0, 1.05, 0, .6), (0, -1.05, 0, -.6),
                           (1.05, 0, .6, 0), (-1.05, 0, -.6, 0)]:
        ax.add_patch(FancyArrowPatch((x, y), (x + dx, y + dy),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color=RED, lw=2.0))
    ax.text(0, 0, "σₘ > 0\ntension", ha="center", va="center", fontsize=9,
            color=RED)

    # shape part: pure shear, no volume change
    ax = axes[2]
    ax.add_patch(Rectangle((-1, -1), 2, 2, facecolor="#e8f3ff",
                           edgecolor=BLUE, lw=1.4, ls=(0, (4, 3))))
    ax.add_patch(plt.Polygon([(-0.65, -1), (1.35, -1), (0.65, 1), (-1.35, 1)],
                             closed=True, facecolor="#cde2fb", alpha=0.6,
                             edgecolor=BLUE, lw=1.6))
    for s, yy in ((1, 1.08), (-1, -1.08)):
        ax.add_patch(FancyArrowPatch((-0.8 * s, yy), (0.8 * s, yy),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color=BLUE, lw=2.0))
    ax.text(0, 0, "same volume\ndifferent shape", ha="center", va="center",
            fontsize=9, color=BLUE)

    fig.text(0.5, -0.06,
             "Metal FLOWS because of the shape part — but voids OPEN or CLOSE "
             "because of the volume part.\n"
             "That is why the ratio between them, η = σₘ / σ_eq, is the variable "
             "the whole method tracks.",
             ha="center", fontsize=10, color=INK)
    save(fig, out, "figC2_stress_split")


# ── C3 — triaxiality decides open vs close ─────────────────────────────────
def figC3_triaxiality(out):
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_yticks([])
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.05, 1.5)
    ax.set_xlabel("stress triaxiality   η = σₘ / σ_eq", color=INK, fontsize=10.5)

    # the growth factor exp(1.5 eta) -- the actual weighting in Rice-Tracey
    xs = np.linspace(-1.5, 1.5, 300)
    ax.plot(xs, np.exp(1.5 * xs) / np.exp(1.5 * 1.5), color=INK2, lw=2.2,
            zorder=3)
    ax.text(1.02, 0.80, "void growth rate\n∝ exp(1.5 η)", fontsize=9.5,
            color=INK2, ha="right")

    ax.axvspan(-1.5, 0, color="#e8f3ff", alpha=0.75, zorder=0)
    ax.axvspan(0, 1.5, color="#fde8e8", alpha=0.75, zorder=0)
    ax.axvline(0, color=INK2, lw=1.2, zorder=2)

    for x, col, lab, note in ((-0.95, BLUE, "η < 0", "COMPRESSION\nvoids CLOSE"),
                              (0.95, RED, "η > 0", "TENSION\nvoids OPEN")):
        ax.add_patch(Circle((x, -0.55), 0.17 if col == BLUE else 0.31,
                            facecolor="none", edgecolor=col, lw=2.4, zorder=4))
        ax.text(x, -0.95, note, ha="center", fontsize=10, color=col)
        ax.text(x, 1.30, lab, ha="center", fontsize=12, color=col)

    ax.annotate("", (-0.55, -0.55), xytext=(-1.35, -0.55),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
    ax.annotate("", (1.45, -0.55), xytext=(1.30, -0.55),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax.text(0, 1.30, "pure shear", ha="center", fontsize=9.5, color=INK2)

    fig.text(0.5, -0.07,
             "Measured in the FSW stir zone: η ranges about −1.2 to +1.4.  "
             "The dependence is EXPONENTIAL,\nso a modest change in η is a large "
             "change in void growth — which is what makes it discriminating.",
             ha="center", fontsize=10, color=INK)
    save(fig, out, "figC3_triaxiality")


# ── C4 — the four pin families, from the real STLs ─────────────────────────
def read_stl_vertices(path: Path, max_tri: int = 60000):
    raw = path.read_bytes()
    n = struct.unpack("<I", raw[80:84])[0]
    if 84 + 50 * n != len(raw):
        return None
    step = max(1, n // max_tri)
    out = []
    for i in range(0, n, step):
        o = 84 + 50 * i + 12
        out.append(np.frombuffer(raw[o:o + 36], dtype="<f4").reshape(3, 3))
    return np.concatenate(out, axis=0).astype(np.float64)


def figC4_pin_shapes(out):
    """The actual pin geometries -- the independent variable of the study."""
    cases = [("A-FlatsVariations", "A1", "A-Flats"),
             ("B-FluteVariations", "B1", "B-Flutes"),
             ("C-ThreadsVariations", "C1", "C-Threads"),
             ("D-ConcavityTilt", "D3", "D-Concavity")]
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.0))

    for k, (fam, case, label) in enumerate(cases):
        stl = STL_ROOT / fam / f"{case}.gid" / f"{case}.stl"
        v = read_stl_vertices(stl) if stl.exists() else None
        axT, axS = axes[0][k], axes[1][k]
        for a in (axT, axS):
            bare(a)
        if v is None:
            axT.text(0.5, 0.5, f"{label}\n(STL unavailable)", ha="center",
                     transform=axT.transAxes, fontsize=10, color=INK2)
            continue
        # pin only: the STL carries the shoulder too (r = 7 mm)
        r = np.hypot(v[:, 0], v[:, 1])
        pin = (v[:, 2] < -0.3) & (r < 5.0)
        vp = v[pin]

        axT.set_title(f"{label}  ({case})", color=INK, fontsize=10.5)
        # ⚠️ ONE colour for both rows. These are the SAME STL vertices in two
        # projections, so two colours would imply two different quantities --
        # which is exactly how the first version was misread.
        axT.scatter(vp[:, 0], vp[:, 1], s=0.35, color=BLUE, alpha=0.35,
                    linewidths=0)
        axT.set_xlim(-4, 4); axT.set_ylim(-4, 4)
        axT.text(0, -3.65, "looking DOWN the axis   (x, y)", ha="center",
                 fontsize=8.5, color=INK2)

        axS.scatter(vp[:, 0], vp[:, 2], s=0.35, color=BLUE, alpha=0.35,
                    linewidths=0)
        axS.set_xlim(-4, 4); axS.set_ylim(-6.2, 0.4)
        axS.text(0, -6.0, "from the SIDE   (x, z)", ha="center",
                 fontsize=8.5, color=INK2)

        # ⚠️ Say so when the STL cannot show the pin, rather than letting an
        # almost-empty panel read as a rendering failure. A1 has 7,560 vertices
        # ALL at the tip (z -6..-5) and NOTHING between -5 and -1: the pin's side
        # surface is simply not tessellated. This is the same coarse-STL limit
        # that made M4's envelope descriptors fail on 9 of 20 cases.
        zspan = vp[:, 2].max() - vp[:, 2].min() if len(vp) else 0.0
        if len(vp) < 1000 or zspan < 2.0:
            for a in (axT, axS):
                a.text(0.5, 0.5,
                       "⚠ STL too coarse\nto show the pin surface\n"
                       "(tip + shoulder only)",
                       transform=a.transAxes, ha="center", va="center",
                       fontsize=9, color=RED)

    fig.text(0.5, 0.02,
             "Each dot is one VERTEX of the tool's STL surface mesh — the pin's "
             "actual geometry, not a computed field.\n"
             "Top row and bottom row are the SAME points in two projections.  "
             "The four families are the only real\nindependent variable in the "
             "PinShapes study.\n"
             "Measured descriptors (M4): B-Flutes 3–8 lobes at 20–32 % depth · "
             "C-Threads 2 lobes at 3–8 % · A-Flats below the detection floor.",
             ha="center", fontsize=10, color=INK)
    save(fig, out, "figC4_pin_shapes")


# ── C5 — damage accumulates ALONG a route ──────────────────────────────────
def figC5_pathline(out):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6),
                                  gridspec_kw={"width_ratios": [1.25, 1]})

    bare(ax)
    ax.set_title("A particle's route past the tool", color=INK, fontsize=11.5,
                 loc="left")
    ax.add_patch(Rectangle((-20, -11), 40, 22, facecolor="#f0efe9",
                           edgecolor=GRID, lw=1.2))
    ax.add_patch(Circle((0, 0), 7.0, facecolor=TOOL, alpha=0.4,
                        edgecolor=INK2, lw=1.0))
    ax.add_patch(Circle((0, 0), 2.4, facecolor=TOOL, edgecolor=INK2, lw=1.3))

    # a plausible route: in from upstream, swept around the pin, out into the wake
    # ⚠️ The route must go AROUND the pin, not through it -- material cannot
    # occupy the tool. Build it in polar form so the radius never drops below the
    # pin surface: approach, sweep around at r ~ 3.2 mm, then exit into the wake.
    t = np.linspace(0, 1, 300)
    sweep = np.sin(np.pi * np.clip((t - 0.30) / 0.40, 0, 1))      # 0 -> 1 -> 0
    ang = np.pi * 1.00 - np.pi * 1.25 * np.clip((t - 0.30) / 0.40, 0, 1)
    r_far = 17.0 * np.abs(1 - 2 * t) + 3.2
    rad = r_far * (1 - sweep) + 3.2 * sweep
    rad = np.maximum(rad, 3.0)                                     # never inside
    theta = np.where(t < 0.5, np.pi, 0.0) * (1 - sweep) + ang * sweep
    x = rad * np.cos(theta)
    y = rad * np.sin(theta) - 0.8
    rr = np.hypot(x, y)
    # damage rate: high only inside the shear layer
    rate = np.exp(-((rr - 3.0) ** 2) / 4.0)
    rate[rr < 2.4] = 0.0
    sc = ax.scatter(x, y, c=rate, cmap="YlOrRd", s=15, zorder=4, vmin=0, vmax=1,
                    linewidths=0)
    ax.plot(x, y, color=INK2, lw=0.7, alpha=0.5, zorder=3)
    ax.add_patch(FancyArrowPatch((x[-14], y[-14]), (x[-1], y[-1]),
                                 arrowstyle="-|>", mutation_scale=15,
                                 color=INK, lw=1.6, zorder=5))
    ax.text(-18.5, 8.6, "seeded upstream", fontsize=9, color=INK2)
    ax.text(10.5, -9.4, "ends in the wake", fontsize=9, color=INK2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02,
                      orientation="horizontal", location="bottom")
    cb.set_label("instantaneous damage RATE along the route", fontsize=9,
                 color=INK2)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlim(-21, 21); ax.set_ylim(-12, 12)

    # the accumulation
    ax2.set_facecolor(SURFACE)
    ax2.grid(True, color=GRID, lw=0.7); ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax2.spines[sp].set_color(GRID)
    ax2.tick_params(colors=INK2, labelsize=9)
    ax2.set_title("…and what it accumulates", color=INK, fontsize=11.5,
                  loc="left")
    acc = np.cumsum(rate) / len(rate) * 3.0
    ax2.plot(t, rate, color=ORANGE, lw=2.0, label="rate  (what it feels NOW)")
    ax2.plot(t, acc, color=BLUE, lw=2.4, label="ln Φ  (the INTEGRAL — what we store)")
    ax2.set_xlabel("fraction of the journey", color=INK2, fontsize=9.5)
    ax2.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    ax2.annotate("rises only while\ninside the shear layer",
                 (0.5, acc[len(acc) // 2]), xytext=(0.52, 0.42),
                 textcoords="axes fraction", fontsize=9, color=INK2,
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=1.1))
    ax2.text(0.02, 0.02,
             "⚠ the accumulated value NEVER decreases —\n"
             "damage is a history, not a state",
             transform=ax2.transAxes, fontsize=9, color=INK)

    fig.text(0.5, -0.05,
             "This is why the method tracks PARTICLES and not a grid: a grid cell "
             "has no history, only\nwhatever material happens to be in it now.",
             ha="center", fontsize=10, color=INK)
    save(fig, out, "figC5_pathline")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("figs_concept"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    figC1_geometry(args.out)
    figC2_stress_split(args.out)
    figC3_triaxiality(args.out)
    figC4_pin_shapes(args.out)
    figC5_pathline(args.out)
    print(f"\n  figures in {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
