"""Separate the two defect indicators: particle DENSITY and accumulated DAMAGE (O45).

    python3 make_density_vs_damage.py --src <run_dir> [--cases A1 D1 ...]

WHY THIS EXISTS
---------------
The wake cross-section figures (`make_wake_metrics.py`) colour a kNN median of
ln(Phi/Phi0). That estimator is **not independent of particle density**:

  * the k nearest neighbours of a sparse point are drawn from a WIDER region, so the
    estimate is smoothed over more space exactly where particles are scarce;
  * the blanking threshold is a pure density test -- a blank cell means "no particles
    nearby", which says nothing about damage.

Measured density-vs-damage rank correlation inside the wake box:

    A-Flats A1    rho = -0.063   (density  230..1125 /mm2)
    B-Flutes B2   rho = -0.073   (density  202..1129 /mm2)
    D-Concavity   rho = -0.291   (density   32.. 879 /mm2)   <-- 27x density range

So for A and B the confound is weak, but for D the figure partly renders its density
deficit as a damage pattern -- and D is the case whose void is in dispute (O43).

WHAT THIS PRODUCES
------------------
Three panels per case, on the SAME grid, so they can be read against each other:

  1. DENSITY      particles per mm2 -- the particle-tracking indicator on its own.
                  A void is a density hole. Damage plays no part in this panel.
  2. DAMAGE       median ln(Phi/Phi0) per cell, computed ONLY where the density is
                  high enough to support an estimate (>= MIN_N particles in a FIXED
                  area, not a k-nearest-neighbour radius). This removes the
                  variable-support bias: every reported cell is averaged over the
                  same physical area.
  3. AGREEMENT    where the two indicators point to the same place and where they
                  disagree -- which is the actual question.

⚠️ THE KEY DESIGN CHOICE. Panel 2 uses a FIXED-AREA estimator (a bin), not kNN.
A fixed bin has constant support everywhere, so a cell's value cannot be influenced
by how far away its neighbours are. The cost is that sparse cells report nothing
rather than something smoothed -- which is correct here, because "we cannot measure
damage here" and "damage is low here" are different statements and the whole point
is to keep them apart.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

X_AFTER_TOOL_MM = 7.4
DZ_FRAC = 0.08
MIN_N = 8                    # particles needed in a cell before damage is reported


def wake(pos, ln):
    x, y, z = pos[:, 0] * 1e3, pos[:, 1] * 1e3, pos[:, 2] * 1e3
    m = (x >= X_AFTER_TOOL_MM) & (z <= z.max() - DZ_FRAC * (z.max() - z.min()))
    return y[m], z[m], ln[m]


def fields(Y, Z, L, nby, nbz, extent):
    """Density and fixed-area median damage on one shared grid."""
    from scipy.stats import binned_statistic_2d

    y0, y1, z0, z1 = extent
    ye = np.linspace(y0, y1, nby + 1)
    ze = np.linspace(z0, z1, nbz + 1)
    cell = (ye[1] - ye[0]) * (ze[1] - ze[0])

    cnt, _, _, _ = binned_statistic_2d(Y, Z, L, statistic="count", bins=[ye, ze])
    med, _, _, _ = binned_statistic_2d(Y, Z, np.log10(np.maximum(L, 1e-4)),
                                       statistic="median", bins=[ye, ze])
    dens = cnt / cell
    # ⚠️ Damage is reported ONLY where the cell holds enough particles. Elsewhere it
    # is NaN -- "not measurable", which is deliberately distinct from "low".
    dmg = np.where(cnt >= MIN_N, np.power(10.0, med), np.nan)
    return ye, ze, dens, dmg, cnt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--out", default="figs_compare")
    ap.add_argument("--bins-y", type=int, default=150)
    ap.add_argument("--bins-z", type=int, default=60)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.src, "*", "damage_models.npz")))
    if args.cases:
        keep = set(args.cases)
        paths = [p for p in paths
                 if os.path.basename(os.path.dirname(p)) in keep
                 or os.path.basename(os.path.dirname(p)).split("_")[-1] in keep]
    if not paths:
        print("  no cases matched")
        return 1

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for p in paths:
        tag = os.path.basename(os.path.dirname(p))
        z = np.load(p, allow_pickle=True)
        Y, Z, L = wake(z["positions"], z["damage"][:, 0])
        extent = (Y.min(), Y.max(), Z.min(), Z.max())
        ye, ze, dens, dmg, cnt = fields(Y, Z, L, args.bins_y, args.bins_z, extent)
        rows.append((tag, ye, ze, dens, dmg, cnt))

        seed_density = len(Y) / ((Y.max() - Y.min()) * (Z.max() - Z.min()))
        empty = float((cnt == 0).mean())
        low = float((cnt < MIN_N).mean())
        # Correlation between the two indicators, over cells where BOTH exist.
        ok = np.isfinite(dmg) & (dens > 0)
        from scipy.stats import spearmanr
        rho = float(spearmanr(dens[ok], dmg[ok]).statistic) if ok.sum() > 100 else np.nan
        print("  %-30s mean dens %6.0f /mm2 | empty cells %5.1f%% | "
              "unmeasurable %5.1f%% | rho(dens,dmg) %+.3f"
              % (tag, seed_density, 100 * empty, 100 * low, rho))

    make_fig(rows, args.out)
    return 0


def make_fig(rows, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # Shared scales across every case, or the panels cannot be compared.
    alld = np.concatenate([r[3].ravel() for r in rows])
    allm = np.concatenate([r[4].ravel() for r in rows])
    dlo = max(float(np.percentile(alld[alld > 0], 2)), 1.0)
    dhi = float(np.percentile(alld[alld > 0], 99.5))
    mlo = max(float(np.nanpercentile(allm, 2)), 1e-3)
    mhi = float(np.nanpercentile(allm, 99.5))

    INK2 = "#555555"
    for tag, ye, ze, dens, dmg, cnt in rows:
        fig, axes = plt.subplots(3, 1, figsize=(12.0, 8.2))

        # --- 1. density: the particle-tracking indicator, alone ---
        a = axes[0]
        im = a.pcolormesh(ye, ze, np.ma.masked_less_equal(dens, 0).T,
                          cmap="viridis", norm=LogNorm(vmin=dlo, vmax=dhi),
                          shading="flat", rasterized=True)
        a.set_facecolor("#ffffff")
        fig.colorbar(im, ax=a, fraction=0.016, pad=0.012).set_label(
            "particles / mm²  (log)", fontsize=8, color=INK2)
        a.set_title("1 — PARTICLE DENSITY: where material ended up. "
                    "White = no particles at all.",
                    fontsize=9.5, color="#1a3a5c", loc="left")

        # --- 2. damage on fixed-area support, only where measurable ---
        a = axes[1]
        im = a.pcolormesh(ye, ze, np.ma.masked_invalid(dmg).T, cmap="inferno",
                          norm=LogNorm(vmin=mlo, vmax=mhi), shading="flat",
                          rasterized=True)
        a.set_facecolor("#d9d9d9")
        fig.colorbar(im, ax=a, fraction=0.016, pad=0.012).set_label(
            "median ln(Φ/Φ₀)  (log)", fontsize=8, color=INK2)
        a.set_title(f"2 — ACCUMULATED DAMAGE, fixed-area median "
                    f"(cells with ≥ {MIN_N} particles). Grey = not measurable, "
                    f"NOT low.",
                    fontsize=9.5, color="#1a3a5c", loc="left")

        # --- 3. where do the two indicators agree? ---
        # ⚠️ Both are converted to WITHIN-CASE percentile ranks before comparing.
        # The raw units are unrelated (particles/mm2 vs a log damage ratio), so any
        # direct difference would be meaningless; ranks make "high for this weld"
        # comparable between the two.
        a = axes[2]
        dr = np.full(dens.shape, np.nan)
        ok = dens > 0
        dr[ok] = _rank(dens[ok])
        mr = np.full(dmg.shape, np.nan)
        okm = np.isfinite(dmg)
        mr[okm] = _rank(dmg[okm])
        # low density AND high damage  ->  +1  (both indicate a defect)
        # low density AND low damage   ->  -1  (density only)
        diff = np.where(np.isfinite(dr) & np.isfinite(mr), mr - (1.0 - dr), np.nan)
        im = a.pcolormesh(ye, ze, np.ma.masked_invalid(diff).T, cmap="coolwarm",
                          vmin=-1, vmax=1, shading="flat", rasterized=True)
        a.set_facecolor("#d9d9d9")
        fig.colorbar(im, ax=a, fraction=0.016, pad=0.012).set_label(
            "damage rank − (1 − density rank)", fontsize=8, color=INK2)
        a.set_title("3 — AGREEMENT: red = damage high where material is sparse "
                    "(both indicators agree) · blue = they disagree",
                    fontsize=9.5, color="#1a3a5c", loc="left")

        for a in axes:
            a.set_aspect("equal", adjustable="box")   # 1 mm in y == 1 mm in z
            a.set_ylabel("z [mm]", fontsize=8.5, color=INK2)
            a.tick_params(labelsize=7.5, colors=INK2)
        axes[-1].set_xlabel("y [mm]      (advancing / retreating — per-case sense)",
                            fontsize=8.5, color=INK2)
        fig.suptitle(f"Two independent defect indicators — {tag}",
                     fontsize=11, color="#1a3a5c", x=0.01, ha="left")
        fig.text(0.01, 0.005,
                 "Panels 1 and 2 share no estimator: density is a count, damage is a "
                 "fixed-area median computed only where particles support it. A "
                 "feature appearing in BOTH is not an artefact of either.",
                 fontsize=7.5, color=INK2)
        out = os.path.join(outdir, f"figX_{tag}_density_vs_damage.png")
        fig.savefig(out, dpi=165, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  wrote {out}")


def _rank(v):
    """Percentile rank in [0,1], ties averaged."""
    from scipy.stats import rankdata
    return (rankdata(v) - 0.5) / v.size


if __name__ == "__main__":
    raise SystemExit(main())
