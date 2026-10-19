"""Schematic of the contact-pressure sampling geometry for B (A.4).

    python3 make_dp_schematic.py [--out figs_dp/fig_dp_schematic.png]

⚠️ DRAWN TO THE ACTUAL PARAMETERS the measurement uses, not illustratively:

    r_pin       3.0 mm          indicator_dp.py --r-pin default
    band        0.6             annulus r in [r_pin, 1.6 r_pin] = 3.0 .. 4.8 mm
    sector      30 deg total    +-15 deg about theta = 0 and theta = 180
    theta       from +x         the travel direction
    rpm         -600            so the ADVANCING side is +y

⚠️ ADVANCING SIDE, derived rather than assumed: the advancing side is where the
tool's SURFACE motion ADDS to the travel direction. With omega = (0,0,w) and
travel along +x, the surface velocity at (0,y) has v_x = -w*y, so for w < 0
(clockwise seen from +z, our case) v_x > 0 on the +y side. ADVANCING IS +y.
indicator_dp.py had this inverted; corrected 2026-10-19. It affected only the
secondary advancing-retreating diagnostic -- B is a front/rear quantity and does
not use it -- so every B result stands unchanged.

A schematic that quietly used rounder numbers would be worse than none: the
whole point of this figure is to show a reader where the two numbers in
B = P_max(front) - P_min(rear) actually come from.

⚠️ THE PLANE IS xy, NOT yz. B is a front-to-rear difference along the travel
direction, and yz is a plane of constant x, so it cannot contain both ends of
the quantity. This figure is therefore a TOP VIEW, unlike every cross-section
figure in the rest of this work -- which is exactly the point it has to make.
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figs_dp/fig_dp_schematic.png")
    ap.add_argument("--r-pin", type=float, default=3.0)       # mm
    ap.add_argument("--band", type=float, default=0.6)
    ap.add_argument("--sector-deg", type=float, default=30.0)
    ap.add_argument("--r-shoulder", type=float, default=9.0)  # mm, for context
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge, Circle, FancyArrowPatch

    rp, rb = args.r_pin, args.r_pin * (1.0 + args.band)
    half = args.sector_deg / 2.0
    INK = "#1a3a5c"
    FRONT, REAR = "#1a3a5c", "#c0392b"

    fig, ax = plt.subplots(figsize=(9.6, 6.2))

    # ── tool ────────────────────────────────────────────────────────────
    ax.add_patch(Circle((0, 0), args.r_shoulder, fc="#f0f2f5", ec="#aab",
                        lw=1.2, zorder=1))
    ax.add_patch(Circle((0, 0), rp, fc="#dfe3e8", ec="#889", lw=1.4, zorder=2))
    ax.text(0, 0, "pin", ha="center", va="center", fontsize=9, color="#556")
    ax.text(0, args.r_shoulder - 0.75, "shoulder", ha="center", va="center",
            fontsize=8.5, color="#889")

    # ── the annulus the measurement samples ─────────────────────────────
    ax.add_patch(Wedge((0, 0), rb, 0, 360, width=rb - rp, fc="#ffd98a",
                       alpha=0.30, ec="none", zorder=3))

    # ── the two sampling sectors ────────────────────────────────────────
    ax.add_patch(Wedge((0, 0), rb, -half, half, width=rb - rp, fc=FRONT,
                       alpha=0.55, ec=FRONT, lw=1.6, zorder=4))
    ax.add_patch(Wedge((0, 0), rb, 180 - half, 180 + half, width=rb - rp,
                       fc=REAR, alpha=0.55, ec=REAR, lw=1.6, zorder=4))

    rm = 0.5 * (rp + rb)
    ax.annotate(r"FRONT  $\theta=0^\circ$" "\n" r"$\to\ P_{\max}$",
                (rm, 0), (rb + 2.6, 1.4), fontsize=9.5, color="white",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc=FRONT, ec="none"),
                arrowprops=dict(arrowstyle="->", color=FRONT, lw=1.5))
    ax.annotate(r"REAR  $\theta=180^\circ$" "\n" r"$\to\ P_{\min}$",
                (-rm, 0), (-rb - 2.6, 1.4), fontsize=9.5, color="white",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc=REAR, ec="none"),
                arrowprops=dict(arrowstyle="->", color=REAR, lw=1.5))

    # ── travel direction, and the rotation sense that fixes AS/RS ───────
    # ⚠️ Axis is standard: +x to the RIGHT. The arrow must point right, towards
    # the FRONT sector at theta = 0. An earlier version drew it pointing left,
    # which contradicted the front/rear labels on the same figure.
    ax.add_patch(FancyArrowPatch((args.r_shoulder + 0.9, args.r_shoulder + 1.6),
                                 (args.r_shoulder + 4.2, args.r_shoulder + 1.6),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=INK, lw=2.0))
    ax.text(args.r_shoulder + 2.5, args.r_shoulder + 2.3,
            r"travel  $+x$", fontsize=10, color=INK, ha="center")

    th = np.linspace(np.radians(55), np.radians(125), 60)
    rr = args.r_shoulder + 1.1
    ax.plot(rr * np.cos(th), rr * np.sin(th), color="#777", lw=1.4)
    ax.add_patch(FancyArrowPatch(
        (rr * np.cos(th[1]), rr * np.sin(th[1])),
        (rr * np.cos(th[0]), rr * np.sin(th[0])),
        arrowstyle="-|>", mutation_scale=13, color="#777", lw=1.4))
    ax.text(0, rr + 1.0, r"$\omega = -600$ rpm (clockwise)",
            fontsize=8.5, color="#777", ha="center")

    # ⚠️ With omega < 0 the advancing side is -y. Labelling it matters: this is
    # the axis the ADVANCING/RETREATING difference uses, and that is a DIFFERENT
    # quantity from B -- a different magnitude on D2 (+12.47 vs +16.8 MPa),
    # and it splits on a different axis entirely.
    ax.text(0, -args.r_shoulder - 1.9, "RETREATING side  ($-y$)",
            fontsize=9, color="#555", ha="center")
    ax.text(0, args.r_shoulder + 3.4, "ADVANCING side  ($+y$)",
            fontsize=9, color="#555", ha="center")

    # ── radii ───────────────────────────────────────────────────────────
    for r, lab, dy in ((rp, f"$r_{{pin}}$ = {rp:.1f} mm", -0.55),
                       (rb, f"$1.6\\,r_{{pin}}$ = {rb:.1f} mm", -0.55)):
        ax.plot([0, r * np.cos(np.radians(-58))],
                [0, r * np.sin(np.radians(-58))], color="#889", lw=0.9, ls=":")
        ax.text(r * np.cos(np.radians(-58)) + 0.3,
                r * np.sin(np.radians(-58)) + dy, lab, fontsize=8, color="#667")

    ax.text(rb + 0.4, -rb - 1.1,
            f"sampling annulus,\n±{half:.0f}° sectors",
            fontsize=8.5, color="#a07000", ha="left")

    # ⚠️ Asymmetric limits: the drawing needs head-room above for the rotation
    # arc and travel arrow, but little below the 'RETREATING side' label.
    # Symmetric limits wasted ~25 % of the frame and overflowed the slide.
    lim = args.r_shoulder + 6.6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-args.r_shoulder - 3.4, args.r_shoulder + 4.8)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(
        "Where $B$ comes from — TOP VIEW ($xy$ plane, pin mid-depth)\n"
        r"$B=P_{\max}(\theta=0^\circ)-P_{\min}(\theta=180^\circ)$"
        "    ·    Shi et al. (2022): voids form when $B>15$ MPa",
        fontsize=11, color=INK, loc="left", pad=14)

    fig.text(0.01, 0.015,
             "⚠️ This is the ONLY figure in this work drawn in the xy plane. B is "
             "a FRONT-TO-REAR difference along the travel direction, and a yz "
             "cross-section is a plane of constant x — it cannot contain both "
             "ends of the quantity.\n"
             "⚠️ Not to be confused with the ADVANCING−RETREATING difference, "
             "which splits on ±y instead: a different quantity, measured at "
             "+12.47 MPa on D2 where B is +16.8 — a different axis and a\n"
             "different magnitude.",
             fontsize=8, color="#555")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
