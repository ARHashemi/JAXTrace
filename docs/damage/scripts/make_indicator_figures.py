"""yz cross-sections for ALL FOUR pathline indicators (Part A).

    python3 make_indicator_figures.py --npz <damage_models.npz> [--out figs_ind]
    python3 make_indicator_figures.py --src <run_root> [--cases D2 B2]

Produces, for each indicator, the same two-stage presentation already used for
ln(Phi/Phi0) in make_wake_metrics.py:

  stage 1  raw particles in the wake box, projected on yz, coloured by the value
  stage 2  the same field smoothed by a k-nearest-neighbour median, so
           resolution is a display choice rather than a particle-count artefact

⚠️ THE FOUR INDICATORS ARE NOT INTERCHANGEABLE, so they do not share a colour
treatment. Measured on the D2 wake box (268,764 particles):

    indicator   min        median     max        negative   exactly zero
    ln_phi      +0.0725    +0.256     +366.9       0.0 %       0.0 %
    C           +0.0634    +0.240     +342.0       0.0 %       0.0 %
    t_res        0.0        0.0       +2.169       0.0 %      65.6 %
    w_bond      -0.0378    +0.0457    +0.410      27.1 %       0.0 %

  * ln_phi, C  -- strictly positive over ~5 decades and heavily right-skewed.
                  LOG colour scale, sequential map, median taken in log10 space.
                  This is the existing protocol and is unchanged.
  * t_res      -- a TIME, and 65.6 % of particles are EXACTLY ZERO because they
                  never entered the shear layer. ⚠️ A log scale is impossible
                  here, and a sequential map would render "never sheared" and
                  "barely sheared" as the same dark colour. Plotted LINEAR, with
                  the exact zeros drawn as a separate grey category so the
                  never-entrained population is visible as a population rather
                  than hidden in the bottom of the ramp.
  * w_bond     -- SIGNED: 27.1 % negative. The sign is the physics (w > 0 means
                  compression, which accumulates bonding; w < 0 means tension,
                  which does not). ⚠️ A sequential map would destroy that, so
                  this uses a DIVERGING map centred exactly on zero, with the
                  limits symmetric so the midpoint colour always means w = 0.

⚠️ For t_res the kNN MEDIAN is reported for consistency with the other panels,
but note that where more than half the neighbours are zero the median is zero --
which is correct ("most material here never entered the layer") but means the
smoothed t_res panel reads as coverage rather than intensity. The raw panel is
the more informative one for this indicator; both are produced so the difference
is visible rather than hidden.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

X_AFTER_TOOL_MM = 7.4      # wake box start, matching make_wake_metrics.py
DZ_FRAC = 0.08             # drop the top 8 % of thickness (BC-dominated band)

# How each indicator is displayed. 'scale' drives both stages.
SPEC = {
    "ln_phi": dict(label=r"$\ln(\Phi/\Phi_0)$  Rice–Tracey",
                   scale="log", cmap="inferno"),
    "C":      dict(label=r"$C$  Cockcroft–Latham",
                   scale="log", cmap="inferno"),
    "t_res":  dict(label=r"$t_\mathrm{res}$  residence in shear layer [s]",
                   scale="linear_zero", cmap="viridis"),
    "w_bond": dict(label=r"$w=\int p/\bar\sigma\,dt$  Piwnik–Plata",
                   scale="diverging", cmap="RdBu_r"),
}
ORDER = ["ln_phi", "C", "t_res", "w_bond"]


def load(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    d = np.asarray(z["damage"], dtype=np.float64)
    pos = np.asarray(z["positions"], dtype=np.float64)
    cols = ([str(c) for c in np.asarray(z["columns"]).tolist()]
            if "columns" in z.files else
            (["ln_phi", "C", "t_res", "w_bond"][:d.shape[1]]))
    thr = float(z["residence_edot_threshold"]) if \
        "residence_edot_threshold" in z.files else float("nan")
    return d, pos, cols, thr


def wake_mask(pos):
    x, z = pos[:, 0] * 1e3, pos[:, 2] * 1e3
    return (x >= X_AFTER_TOOL_MM) & (z <= z.max() - DZ_FRAC * (z.max() - z.min()))


def knn_median(y, zz, v, gy, gz, k=16):
    """k-nearest-neighbour median onto a display grid.

    Resolution is decoupled from particle count: every output point is the
    median of its k nearest particles wherever they are, so the estimator stays
    supported even where a histogram bin would be empty. Same choice, and the
    same k, as make_wake_metrics.py.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([y, zz]))
    GY, GZ = np.meshgrid(gy, gz, indexing="ij")
    q = np.column_stack([GY.ravel(), GZ.ravel()])
    _, idx = tree.query(q, k=min(k, len(y)), workers=-1)
    if idx.ndim == 1:
        idx = idx[:, None]
    return np.median(v[idx], axis=1).reshape(GY.shape)


def _limits(v, scale):
    """Colour limits appropriate to the indicator's character."""
    if scale == "log":
        pos = v[v > 0]
        lo = max(float(np.percentile(pos, 5)), 1e-3) if pos.size else 1e-3
        return lo, float(np.percentile(v, 99.0))
    if scale == "diverging":
        # ⚠️ Centred on zero so the midpoint colour ALWAYS means w = 0, but the
        # limits are NOT forced symmetric: w is 27 % negative and strongly
        # right-skewed (D2: -0.038 .. +0.410), so a symmetric span wastes most
        # of the blue half and compresses the positive signal that carries the
        # physics. Each side is scaled to its own 99th percentile instead.
        neg, pos = v[v < 0], v[v > 0]
        lo = -float(np.percentile(np.abs(neg), 99.0)) if neg.size else -1e-12
        hi = float(np.percentile(pos, 99.0)) if pos.size else 1e-12
        return lo, hi
    return 0.0, float(np.percentile(v, 99.0))      # linear_zero


def make_panels(tag, pos, d, cols, thr, outdir, nby, nbz, knn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm, Normalize

    os.makedirs(outdir, exist_ok=True)
    m = wake_mask(pos)
    if m.sum() < 100:
        print(f"  {tag}: only {int(m.sum())} particles in the wake box — skipped")
        return
    y, zz = pos[m, 1] * 1e3, pos[m, 2] * 1e3
    gy = np.linspace(y.min(), y.max(), nby)
    gz = np.linspace(zz.min(), zz.max(), nbz)
    INK2 = "#555555"

    for name in ORDER:
        if name not in cols:
            continue
        spec = SPEC[name]
        v = d[m, cols.index(name)]
        lo, hi = _limits(v, spec["scale"])

        fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.4))

        # ── stage 1: raw particles ──────────────────────────────────────
        a = axes[0]
        if spec["scale"] == "log":
            norm = LogNorm(vmin=lo, vmax=hi)
            o = np.argsort(v)                     # draw high values last
            sc = a.scatter(y[o], zz[o], c=np.clip(v[o], lo, hi), s=0.7,
                           cmap=spec["cmap"], norm=norm, linewidths=0,
                           rasterized=True)
        elif spec["scale"] == "diverging":
            norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
            o = np.argsort(np.abs(v))             # draw strong signs last
            sc = a.scatter(y[o], zz[o], c=np.clip(v[o], lo, hi), s=0.7,
                           cmap=spec["cmap"], norm=norm, linewidths=0,
                           rasterized=True)
        else:
            # ⚠️ Exact zeros drawn FIRST, in grey, as their own category: they
            # are "never entered the shear layer", not "a small amount of
            # time", and a sequential ramp would hide that distinction.
            z0 = v <= 0
            a.scatter(y[z0], zz[z0], s=0.7, c="#c8c8c8", linewidths=0,
                      rasterized=True,
                      label=f"never entrained ({100*z0.mean():.0f} %)")
            norm = Normalize(vmin=max(lo, 1e-12), vmax=hi)
            o = np.argsort(v[~z0])
            sc = a.scatter(y[~z0][o], zz[~z0][o], c=np.clip(v[~z0][o], lo, hi),
                           s=0.7, cmap=spec["cmap"], norm=norm, linewidths=0,
                           rasterized=True)
            a.legend(loc="upper right", fontsize=7, framealpha=0.9,
                     markerscale=6)
        fig.colorbar(sc, ax=a, fraction=0.016, pad=0.012).set_label(
            spec["label"], fontsize=8, color=INK2)
        a.set_title(f"1 — raw particles, wake box (x ≥ {X_AFTER_TOOL_MM} mm), "
                    f"projected on yz", fontsize=9.5, color="#1a3a5c", loc="left")

        # ── stage 2: kNN-median smoothed field ──────────────────────────
        a = axes[1]
        fld = knn_median(y, zz, v, gy, gz, k=knn)
        if spec["scale"] == "log":
            im = a.pcolormesh(gy, gz, np.clip(fld.T, lo, hi), cmap=spec["cmap"],
                              norm=LogNorm(vmin=lo, vmax=hi), shading="auto",
                              rasterized=True)
        elif spec["scale"] == "diverging":
            im = a.pcolormesh(gy, gz, np.clip(fld.T, lo, hi), cmap=spec["cmap"],
                              norm=TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi),
                              shading="auto", rasterized=True)
        else:
            im = a.pcolormesh(gy, gz, np.clip(fld.T, lo, hi), cmap=spec["cmap"],
                              vmin=max(lo, 0.0), vmax=hi, shading="auto",
                              rasterized=True)
        fig.colorbar(im, ax=a, fraction=0.016, pad=0.012).set_label(
            spec["label"], fontsize=8, color=INK2)
        a.set_title(f"2 — k-NN median field (k = {knn}, {nby}×{nbz})",
                    fontsize=9.5, color="#1a3a5c", loc="left")
        if spec["scale"] == "linear_zero":
            # ⚠️ As a figure note, not in the title: appended to the title it
            # overran the panel width and was clipped by the colourbar.
            a.text(0.0, -0.22,
                   "⚠️ where most neighbours are zero the median is zero, so "
                   "this panel reads as COVERAGE of the shear layer, not "
                   "intensity — panel 1 is the more informative one here",
                   transform=a.transAxes, fontsize=7.5, color="#8a4a00")

        for ax in axes:
            ax.set_aspect("equal", adjustable="box")
            ax.set_ylabel("z [mm]", fontsize=8.5, color=INK2)
            ax.tick_params(labelsize=7.5, colors=INK2)
        axes[-1].set_xlabel("y [mm]", fontsize=8.5, color=INK2)

        sub = (f"{int(m.sum()):,} particles"
               + (f"   threshold ε̇ > {thr:g} 1/s" if name == "t_res"
                  and np.isfinite(thr) else ""))
        fig.suptitle(f"{tag} — {spec['label']}        {sub}",
                     fontsize=11, color="#1a3a5c", x=0.01, ha="left")
        out = os.path.join(outdir, f"fig_{tag}_{name}.png")
        fig.savefig(out, dpi=165, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=None, help="a single damage_models.npz")
    ap.add_argument("--src", default=None, help="root holding <case>/**/damage_*.npz")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--out", default="figs_indicators")
    ap.add_argument("--bins-y", type=int, default=600)
    ap.add_argument("--bins-z", type=int, default=240)
    ap.add_argument("--knn", type=int, default=16)
    args = ap.parse_args()

    items = []
    if args.npz:
        items.append((os.path.basename(os.path.dirname(args.npz)) or "case",
                      args.npz))
    elif args.src:
        hits = (glob.glob(os.path.join(args.src, "*", "damage_*.npz"))
                + glob.glob(os.path.join(args.src, "*", "*", "damage_*.npz")))
        seen = {}
        for p in sorted(hits, key=os.path.getmtime, reverse=True):
            tag = os.path.relpath(p, args.src).split(os.sep)[0]
            seen.setdefault(tag, p)
        items = sorted(seen.items())
        if args.cases:
            keep = set(args.cases)
            items = [(t, p) for t, p in items if t in keep]
    else:
        print("  need --npz or --src")
        return 1
    if not items:
        print("  nothing to plot")
        return 1

    for tag, path in items:
        d, pos, cols, thr = load(path)
        print(f"  {tag}: {d.shape[0]:,} particles, columns {cols}")
        make_panels(tag, pos, d, cols, thr, args.out,
                    args.bins_y, args.bins_z, args.knn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
