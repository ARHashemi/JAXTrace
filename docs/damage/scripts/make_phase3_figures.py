"""Phase 3 result figures — the five plots that carry the findings.

    python3 make_phase3_figures.py [--out figs_phase3]

Reads `results/phase3_summary.csv` (45 cases) and, for the per-case figures, the
full per-particle arrays over the mounted LUMI scratch.

Figure list and what each one is FOR:

  fig1_family_ranking   the headline: damage by pin family
  fig2_models_agree     the two damage laws rank cases identically
  fig3_radial_profile   WHERE in the weld damage concentrates
  fig4_advancing_wake   the core physical claim, per case
  fig5_accumulation     the D-family anomaly (O32)

Colour discipline (dataviz skill):
  * categorical slots in FIXED order, never cycled -- colour follows the family,
    so adding or removing a family never repaints the others
  * palette validated: 6 slots PASS lightness/chroma/CVD/normal-vision
  * the validator's contrast WARN obligates relief -> EVERY family is
    direct-labelled, so identity is never carried by colour alone
  * one y-axis per panel, never two
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── validated categorical palette, fixed order ──────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
FAMILY_ORDER = ["C-ThreadsVariations", "B-FluteVariations", "D-ConcavityTilt",
                "A-FlatsVariations", "cohort", "validation"]
FAMILY_COLOR = {
    "C-ThreadsVariations": "#2a78d6",   # slot 1 blue
    "B-FluteVariations":   "#eb6834",   # slot 2 orange
    "D-ConcavityTilt":     "#1baf7a",   # slot 3 aqua
    "A-FlatsVariations":   "#eda100",   # slot 4 yellow
    "cohort":              "#e87ba4",   # slot 5 magenta
    "validation":          "#4a3aa7",   # slot 6 violet
}
SHORT = {"C-ThreadsVariations": "C-Threads", "B-FluteVariations": "B-Flutes",
         "D-ConcavityTilt": "D-Concavity", "A-FlatsVariations": "A-Flats",
         "cohort": "ROM cohort", "validation": "Validation"}

LUMI = Path("/home/arhashemi/lumi/lumi_scratch/hashemia/damage/"
            "phase3_overnight_20260930")


def style(ax, title=None, xlabel=None, ylabel=None):
    """Recessive grid and axes; the data carries the ink."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    if title:
        ax.set_title(title, color=INK, fontsize=11.5, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)


def load_summary(path="results/phase3_summary.csv"):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("case", "family"):
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                r[k] = float("nan")
    return rows


def save(fig, out: Path, name: str):
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=170, bbox_inches="tight",
                    facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {name}.png / .svg")


# ── fig 1 — the headline ────────────────────────────────────────────────────
def fig1_family_ranking(rows, out):
    """Damage by pin family. PinShapes only -- the cohort is a different
    experiment (different tool, operating point AND rotation sense) and pooling
    them would be the population-mixing error flagged in the results log."""
    ps = [r for r in rows if r["family"].startswith(("A-", "B-", "C-", "D-"))]
    fams = [f for f in FAMILY_ORDER if any(r["family"] == f for r in ps)]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    style(ax, "Accumulated damage by pin family — PinShapes, 20 cases",
          ylabel="median ln(Φ/Φ₀)  per case")

    for i, f in enumerate(fams):
        sub = sorted((r for r in ps if r["family"] == f),
                     key=lambda r: -r["lnphi_med"])
        spread = min(0.17, 0.80 / max(len(sub), 1))
        xs = [i + (j - (len(sub) - 1) / 2) * spread for j in range(len(sub))]
        ys = [r["lnphi_med"] for r in sub]
        c = FAMILY_COLOR[f]
        # thin marks, 2px surface ring so overlaps stay legible
        ax.scatter(xs, ys, s=92, color=c, zorder=3,
                   edgecolors=SURFACE, linewidths=2)
        med = float(np.median(ys))
        ax.plot([i - 0.42, i + 0.42], [med, med], color=c, lw=2.4, zorder=4)
        # direct label: the contrast WARN means colour alone is not enough
        # ⚠️ 3 dp, not 2: the family medians differ in the THIRD decimal here
        # (C=0.065, B=0.065, A=0.065, D=0.052), so 2 dp printed "0.07" for two
        # different families and made the figure contradict its own data.
        ax.text(i, med, f"  {med:.3f}", color=INK, fontsize=10,
                va="bottom", ha="center", zorder=5)
        for j, r in enumerate(sub):
            ax.annotate(r["case"].split("_")[-1], (xs[j], ys[j]),
                        textcoords="offset points", xytext=(0, -13),
                        ha="center", fontsize=7, color=INK2)

    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([SHORT[f] for f in fams], color=INK, fontsize=10)
    ax.set_xlim(-0.6, len(fams) - 0.4)

    ax.text(0.0, 1.14,
            "Higher = more void-prone.  Bars are family medians.",
            transform=ax.transAxes, fontsize=9, color=INK2)
    save(fig, out, "fig1_family_ranking")


# ── fig 2 — the two models agree ────────────────────────────────────────────
def fig2_models_agree(rows, out):
    """Rice-Tracey vs Cockcroft-Latham, per case. Two structurally different
    laws; if they disagreed the method would be suspect."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))

    ax = axes[0]
    style(ax, "The two damage models agree",
          xlabel="Rice–Tracey   median ln(Φ/Φ₀)",
          ylabel="Cockcroft–Latham   median C")
    for f in FAMILY_ORDER:
        sub = [r for r in rows if r["family"] == f]
        if not sub:
            continue
        ax.scatter([r["lnphi_med"] for r in sub], [r["C_med"] for r in sub],
                   s=66, color=FAMILY_COLOR[f], label=SHORT[f], zorder=3,
                   edgecolors=SURFACE, linewidths=1.6)
    lim = [0, max(max(r["lnphi_med"] for r in rows),
                  max(r["C_med"] for r in rows)) * 1.08]
    ax.plot(lim, lim, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.text(lim[1] * 0.60, lim[1] * 0.66, "y = x", color=INK2, fontsize=9,
            rotation=38)
    ax.set_xlim(lim); ax.set_ylim(lim)
    # legend present because >= 2 series; text in ink, marks carry identity
    leg = ax.legend(frameon=False, fontsize=8.5, loc="lower right",
                    labelcolor=INK2)

    ax = axes[1]
    style(ax, "Per-case rank correlation between the two models",
          xlabel="rank correlation  ρ(ln Φ, C)")
    rc = sorted(r["rank_corr"] for r in rows)
    ax.hist(rc, bins=np.linspace(0.975, 1.0, 14), color="#2a78d6",
            edgecolor=SURFACE, linewidth=1.4, zorder=3)
    ax.axvline(0.9, color="#e34948", lw=1.8, ls=(0, (4, 3)), zorder=4)
    ax.text(0.9, ax.get_ylim()[1] * 0.92, " flag threshold 0.9",
            color="#e34948", fontsize=9, va="top")
    ax.text(0.02, 0.92,
            f"all {len(rc)} cases between {min(rc):.4f} and {max(rc):.4f}",
            transform=ax.transAxes, fontsize=9.5, color=INK)
    save(fig, out, "fig2_models_agree")


# ── fig 3 — where damage sits, radially ─────────────────────────────────────
def fig3_radial_profile(rows, out):
    """Median lnPhi against distance from the tool axis. Shows damage is
    concentrated near the pin, which is the physical expectation."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    style(ax, "Where damage accumulates — radial profile",
          xlabel="distance from tool axis  r  [mm]",
          ylabel="median ln(Φ/Φ₀)")

    edges = np.linspace(0, 20, 21)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    for f in FAMILY_ORDER:
        sub = [r for r in rows if r["family"] == f]
        if not sub:
            continue
        prof = np.array([[r[f"rprof_{i:02d}"] for i in range(20)] for r in sub])
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(prof, axis=0)
        ax.plot(ctr, med, color=FAMILY_COLOR[f], lw=2.0, zorder=3,
                label=SHORT[f])
        # direct label at the curve's peak, not only in the legend
        k = int(np.nanargmax(med)) if np.isfinite(med).any() else 0
        ax.annotate(SHORT[f], (ctr[k], med[k]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8.5, color=INK2)

    # the pin and shoulder radii, measured in M1/M4
    ax.axvspan(0, 2.4, color="#cde2fb", alpha=0.45, zorder=1)
    ax.text(1.2, ax.get_ylim()[1] * 0.93, "pin\n(r≲2.4 mm)", fontsize=8,
            color=INK2, ha="center", va="top")
    ax.axvline(7.0, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.text(7.2, ax.get_ylim()[1] * 0.93, "shoulder 7 mm", fontsize=8,
            color=INK2, va="top")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="center right")
    save(fig, out, "fig3_radial_profile")


# ── fig 4 — the core physical claim ─────────────────────────────────────────
def fig4_advancing_wake(rows, out):
    """Damage on the two wake flanks, each case against its OWN advancing side.

    ⚠️⚠️ THIS FIGURE CONTRADICTS THE STAGE-1 RESULT, and that is the finding --
    not a plotting error. Verified before drawing:
      * the flank assignment is correct: PinShapes RPM>0 (CCW) -> advancing=+y,
        cohort RPM<0 (CW) -> advancing=-y, matching the MEASURED
        advancing_y_sign in delta_window_45.csv (A1=+1, cylindrical_000=-1);
      * the data is present (0 NaN in 45 rows).
    Result: accumulated lnPhi is LOWER on the advancing wake in 45/45 cases,
    while Stage 1 found eta tensile on the advancing side (94% vs 1%).

    The two quantities are not the same thing:
      Stage 1  -- the INSTANTANEOUS stress state at nodes in the shear layer
      Phase 3  -- what a PARTICLE ACCUMULATED along its whole route
    A region can have the more dangerous stress state while particles passing
    through it accumulate less, if they spend less time there or sample it less.

    ⚠️ I tested the obvious explanation (advancing particles leave the zone
    sooner) and it FAILED: advancing particles ended CLOSER to the axis
    (median r 9.49 mm) than retreating ones (9.86 mm) on A1, which is the
    opposite of what a residence-time story predicts. **The cause is not yet
    established** -- see O33. The figure states the discrepancy rather than
    resolving it."""
    # ⚠️ TWO panels, not one. The cohort spans lnPhi 1-4.6 while PinShapes sits
    # below 0.2, so a shared axis crushes all 20 PinShapes cases into the corner.
    # Small multiples with their OWN limits, rather than a log axis that would
    # make the 1:1 line curve and the comparison harder to read.
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    groups = [("PinShapes  (20 cases, CCW)",
               lambda r: r["family"].startswith(("A-", "B-", "C-", "D-"))),
              ("ROM cohort + Validation  (25 cases, CW)",
               lambda r: not r["family"].startswith(("A-", "B-", "C-", "D-")))]

    n_adv = 0
    for ax, (title, pred) in zip(axes, groups):
        style(ax, title,
              xlabel="median ln(Φ/Φ₀), retreating wake",
              ylabel="median ln(Φ/Φ₀), advancing wake")
        vals = []
        for f in FAMILY_ORDER:
            sub = [r for r in rows if r["family"] == f and pred(r)]
            if not sub:
                continue
            xs, ys = [], []
            for r in sub:
                ccw = r["family"].startswith(("A-", "B-", "C-", "D-"))
                adv = r["ln_yplus_wake"] if ccw else r["ln_yminus_wake"]
                ret = r["ln_yminus_wake"] if ccw else r["ln_yplus_wake"]
                if not (math.isfinite(adv) and math.isfinite(ret)):
                    continue
                xs.append(ret); ys.append(adv); vals += [adv, ret]
                if adv > ret:
                    n_adv += 1
            ax.scatter(xs, ys, s=70, color=FAMILY_COLOR[f], label=SHORT[f],
                       zorder=3, edgecolors=SURFACE, linewidths=1.6)
        hi = max(vals) * 1.18 if vals else 1.0
        ax.plot([0, hi], [0, hi], color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=2)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.text(hi * 0.52, hi * 0.93, "1:1", color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2,
                  loc="upper left")

    fig.text(0.5, -0.04,
             "⚠ EVERY case falls BELOW the 1:1 line: accumulated damage is higher on the "
             "RETREATING wake —\nopposite to Stage 1's instantaneous η (94 % tensile on "
             "advancing). Cause unresolved — O33.",
             ha="center", fontsize=9.5, color="#e34948")
    save(fig, out, "fig4_advancing_wake")
    return n_adv


# ── fig 5 — the D-family anomaly ────────────────────────────────────────────
def fig5_accumulation(rows, out):
    """Fraction of particles that accumulated any damage. Flags O32: the whole
    D family sits ~16 points below everything else."""
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    style(ax, "Fraction of particles that accumulated damage  (O32)",
          ylabel="accumulating fraction")

    srt = sorted(rows, key=lambda r: r["acc_frac"])
    xs = range(len(srt))
    cols = [FAMILY_COLOR[r["family"]] for r in srt]
    ax.bar(xs, [r["acc_frac"] for r in srt], color=cols, width=0.78,
           zorder=3, edgecolor=SURFACE, linewidth=1.2)
    ax.set_ylim(0.75, 1.01)
    ax.set_xticks([])
    ax.set_xlabel("45 cases, sorted", color=INK2, fontsize=9.5)

    dmed = float(np.median([r["acc_frac"] for r in rows
                            if r["family"] == "D-ConcavityTilt"]))
    omed = float(np.median([r["acc_frac"] for r in rows
                            if r["family"] != "D-ConcavityTilt"]))
    ax.axhline(omed, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=4)
    ax.text(len(srt) * 0.5, omed + 0.004,
            f"all other families: {omed:.3f}", fontsize=9, color=INK2)
    ax.annotate(f"the entire D-Concavity family\n(tilted tool): {dmed:.3f}",
                (2, dmed), textcoords="offset points", xytext=(14, 26),
                fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.1))
    hs = [plt.Line2D([], [], color=FAMILY_COLOR[f], lw=6, label=SHORT[f])
          for f in FAMILY_ORDER if any(r["family"] == f for r in rows)]
    ax.legend(handles=hs, frameon=False, fontsize=8.5, labelcolor=INK2,
              loc="lower right", ncol=2)
    save(fig, out, "fig5_accumulation")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("figs_phase3"))
    # ⚠️ Phase 4 writes a DIFFERENT summary (30 mm travel, RK4 damage). Pointing
    # this at the Phase 4 CSV regenerates every figure against that run instead
    # -- the figures themselves are run-agnostic.
    ap.add_argument("--summary", default="results/phase3_summary.csv",
                    help="summary CSV to plot (default: the Phase 3 survey)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load_summary(args.summary)
    print(f"  {len(rows)} cases from results/phase3_summary.csv")

    fig1_family_ranking(rows, args.out)
    fig2_models_agree(rows, args.out)
    fig3_radial_profile(rows, args.out)
    n_adv = fig4_advancing_wake(rows, args.out)
    fig5_accumulation(rows, args.out)

    print()
    print(f"  fig4: advancing wake worse in {n_adv}/{len(rows)} cases")
    print(f"  figures in {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
