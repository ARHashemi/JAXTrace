"""Wake-box damage metrics and yz projections (O38).

    python3 make_wake_metrics.py --src <phase4_run_dir> [--out figs_wake]

THE WAKE BOX, and why each wall sits where it does
--------------------------------------------------
The whole point is to score only material that has *been through the process and
come out the other side*, with the two known contaminants removed:

  x_min  = +X_AFTER_TOOL mm      AFTER the tool -> excludes the tool footprint and
                                 the recirculating cap still orbiting the pin
  x_max  = max x over particles  the downstream front at the final step
  y      = full domain span      kept: the advancing/retreating contrast lives here
  z_min  = domain z_min          kept: the weld root is physically interesting
  z_max  = domain z_max - DZ     ⚠️ drops the top band, which sits against the
                                 prescribed-velocity surface and whose p99 is
                                 5-16x the bulk (O35) -- a BC response, not damage

⚠️ DZ cannot be "2-3 particle layers": seeding is RANDOM, so z is continuous and
there are no layers (6001 unique z values in 100k particles). DZ is therefore a
fraction of PLATE THICKNESS, which is mesh- and case-independent. The default 8%
is 0.36 mm (cohort) / 0.48 mm (PinShapes) -- comparable to the ~0.3-0.6 mm a
2-3 layer band would be, but reproducible.

WHAT IS REPORTED
----------------
Per case: particle count in the box, and median / p90 / p99 / mean of lnPhi and C,
plus frac_failed. Averages over the box are the headline the user asked for.

Figures: one panel per case, particles in the box projected onto the yz plane and
coloured by lnPhi. The projection collapses x, so each pixel is the whole wake
depth at that (y,z) -- which is exactly the cross-section a macrograph would cut.
"""
from __future__ import annotations

import argparse
import glob
import os
import csv

import numpy as np

LN_FAIL = float(np.log(1.0 / 1.0e-4))        # 9.21 -> Phi = 1 for Phi0 = 1e-4
# ⚠️ Panels per figure. A-Flats has 7 cases; stacked at equal y/z aspect that is
# 1.31x taller than wide and unreadable on a 16:9 slide. 3 keeps every figure wider
# than tall.
PER_FIG = 3
# ⚠️ The upstream wall of the wake box. The shoulder radius is 7.0 mm, but a cut at
# exactly 7.0 leaves a thin rind of particles still hugging the tool: at x = 7.0-7.2
# the median lnPhi is 7.48 against 0.16 at x = 9-10, a 46x difference. They are ~100
# of ~88,000 particles, so they shift the global median by only 0.02-0.19 % -- but in
# a kNN-median FIELD they sit at specific (y,z) cells and dominate them locally.
# 7.4 mm clears the rind while discarding ~0.1 % of the population.
X_AFTER_TOOL_MM = 7.4
DZ_FRAC = 0.08                               # top band dropped, as a fraction of t


def wake_mask(pos: np.ndarray, x_after: float = X_AFTER_TOOL_MM,
              dz_frac: float = DZ_FRAC) -> tuple[np.ndarray, dict]:
    x, y, z = pos[:, 0] * 1e3, pos[:, 1] * 1e3, pos[:, 2] * 1e3
    zmin, zmax = z.min(), z.max()
    thick = zmax - zmin
    dz = dz_frac * thick
    m = (x >= x_after) & (z <= zmax - dz)
    box = dict(x_min=x_after, x_max=float(x.max()),
               y_min=float(y.min()), y_max=float(y.max()),
               z_min=float(zmin), z_max=float(zmax - dz),
               dz_dropped=float(dz), thickness=float(thick))
    return m, box


def stats(v: np.ndarray) -> dict:
    if v.size == 0:
        return dict(n=0, mean=np.nan, med=np.nan, p90=np.nan, p99=np.nan, mx=np.nan)
    return dict(n=int(v.size), mean=float(v.mean()), med=float(np.median(v)),
                p90=float(np.percentile(v, 90)), p99=float(np.percentile(v, 99)),
                mx=float(v.max()))


def famof(tag: str) -> str:
    if tag.startswith("rom_"):
        return "cohort"
    if tag.startswith("val_"):
        return "validation"
    return tag.split("_")[1] if tag.startswith("ps_") else "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="figs_wake")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--x-after", type=float, default=X_AFTER_TOOL_MM)
    ap.add_argument("--dz-frac", type=float, default=DZ_FRAC)
    ap.add_argument("--no-figs", action="store_true")
    # ⚠️ The scatter panels show Poisson speckle from RANDOM seeding (O39). Binning
    # removes it without changing the physics: each cell reports the MEDIAN lnPhi of
    # the particles in it, so the picture is a field estimate rather than a sample.
    ap.add_argument("--binned", action="store_true",
                    help="also write binned (2D median) yz panels, free of "
                         "random-seeding speckle")
    # ⚠️ Bin resolution is matched to the SEED GRID used by the per-case
    # run_jaxtrace.sh (SEED_GRID="60 120 50" -> 120 in y, 50 in z; older cases
    # used 40 in z). Binning finer than the particles were seeded invents
    # resolution that is not in the data: at 240x80 the median cell holds 4
    # particles and 89 % of cells fall below MIN_N, so the picture becomes
    # sampling noise. At 120x50 the median cell holds 15-19 and ~100 % of cells
    # are occupied.
    ap.add_argument("--bins-y", type=int, default=600,
                    help="y resolution of the smoothed field (default 600)")
    ap.add_argument("--bins-z", type=int, default=240,
                    help="z resolution of the smoothed field (default 240)")
    ap.add_argument("--knn", type=int, default=16,
                    help="neighbours per output point for the kNN median "
                         "(default 16). Larger = smoother, smaller = sharper "
                         "but noisier.")
    ap.add_argument("--knn-maxr", type=float, default=0.45,
                    help="blank an output point whose kth neighbour is farther "
                         "than this [mm] -- marks genuinely empty regions")
    args = ap.parse_args()

    cases = sorted(glob.glob(os.path.join(args.src, "*", "damage_models.npz")))
    if not cases:
        print(f"  no cases under {args.src}")
        return 1

    rows, panels = [], []
    for c in cases:
        tag = os.path.basename(os.path.dirname(c))
        z = np.load(c, allow_pickle=True)
        pos, dm = z["positions"], z["damage"]
        ln = dm[:, 0] if dm.ndim == 2 else dm.reshape(-1)
        C = dm[:, 1] if (dm.ndim == 2 and dm.shape[1] > 1) else np.full_like(ln, np.nan)

        m, box = wake_mask(pos, args.x_after, args.dz_frac)
        sl, sc = stats(ln[m]), stats(C[m])
        rows.append(dict(
            case=tag, family=famof(tag),
            n_total=int(ln.size), n_wake=sl["n"],
            frac_in_wake=float(m.mean()),
            **{f"ln_{k}": v for k, v in sl.items() if k != "n"},
            **{f"C_{k}": v for k, v in sc.items() if k != "n"},
            ln_frac_failed=float((ln[m] > LN_FAIL).mean()) if sl["n"] else np.nan,
            **{f"box_{k}": v for k, v in box.items()},
        ))
        panels.append((tag, pos[m], ln[m], box))
        print("  %-30s wake n=%6d (%4.1f%%)  lnPhi mean %8.3f  med %7.4f  ffail %5.1f%%"
              % (tag, sl["n"], 100 * m.mean(), sl["mean"], sl["med"],
                 100 * rows[-1]["ln_frac_failed"]))

    out_csv = args.csv or os.path.join(
        "results", os.path.basename(args.src.rstrip("/")) + "_wake.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {out_csv}  ({len(rows)} cases)")

    # ---- family averages over the box: the headline the user asked for ----
    print("\n  === averaged over the wake box, by family ===")
    print("  %-22s %4s %9s %9s %9s %8s" % (
        "family", "n", "ln_mean", "ln_med", "ln_p99", "ffail"))
    fam: dict[str, list] = {}
    for r in rows:
        fam.setdefault(r["family"], []).append(r)
    for k in sorted(fam):
        v = fam[k]
        print("  %-22s %4d %9.3f %9.4f %9.1f %7.1f%%" % (
            k, len(v),
            float(np.mean([r["ln_mean"] for r in v])),
            float(np.mean([r["ln_med"] for r in v])),
            float(np.mean([r["ln_p99"] for r in v])),
            100 * float(np.mean([r["ln_frac_failed"] for r in v]))))

    # --no-figs suppresses the SCATTER panels; --binned is independent, so the
    # smoothed field can be produced without the (slow, speckled) scatter.
    if not args.no_figs:
        make_figs(panels, args.out)
    if args.binned:
        make_figs_binned(panels, args.out, args.bins_y, args.bins_z,
                         args.knn, args.knn_maxr)
    return 0


def make_figs(panels, outdir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    os.makedirs(outdir, exist_ok=True)
    INK, INK2 = "#1a1a1a", "#555555"

    # ⚠️ ONE shared colour scale across every panel, or the panels cannot be
    # compared -- which is the whole point of a family figure. lnPhi spans ~5
    # decades and is heavily right-skewed, so the scale is LOG; values at or
    # below 0 are clipped to the floor rather than dropped, so no particle is
    # silently invisible.
    allv = np.concatenate([p[2] for p in panels if p[2].size])
    lo = max(float(np.percentile(allv[allv > 0], 1)), 1e-4) if (allv > 0).any() else 1e-4
    hi = float(np.percentile(allv, 99.5))

    groups: dict[str, list] = {}
    for tag, pos, ln, box in panels:
        groups.setdefault(famof(tag), []).append((tag, pos, ln, box))

    for gname, all_items in groups.items():
      for _k in range(0, len(all_items), PER_FIG):
        items = all_items[_k:_k + PER_FIG]
        _part = f"_{_k // PER_FIG + 1}" if len(all_items) > PER_FIG else ""
        # ⚠️ ONE COLUMN. With equal y/z scales a wake panel is ~30 mm x 5 mm, i.e.
        # 6:1; side-by-side columns would shrink each one to an unreadable strip.
        # Stacking keeps every panel full width and puts the cases directly above
        # one another, which is also the easiest arrangement for comparing them.
        n = len(items)
        ncol, nrow = 1, n
        fig, axes = plt.subplots(nrow, ncol, figsize=(12.5, 2.45 * nrow),
                                 squeeze=False)
        for ax, (tag, pos, ln, box) in zip(axes.ravel(), items):
            y, zz = pos[:, 1] * 1e3, pos[:, 2] * 1e3
            v = np.clip(ln, lo, hi)
            # Draw the most damaged last so the tail is not hidden under the bulk.
            o = np.argsort(v)
            sc = ax.scatter(y[o], zz[o], c=v[o], s=0.7, cmap="inferno",
                            norm=LogNorm(vmin=lo, vmax=hi), linewidths=0,
                            rasterized=True)
            # ⚠️ Title INSIDE the axes: stacked panels sit flush, so an external
            # title lands on the tick labels of the panel above.
            ax.text(0.006, 0.955, tag.replace("ps_", "").replace("_", " "),
                    transform=ax.transAxes, fontsize=9, color="white",
                    va="top", ha="left", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#00000088", ec="none"))
            ax.set_ylabel("z  [mm]", fontsize=8, color=INK2)
            ax.tick_params(labelsize=7, colors=INK2)
            ax.set_facecolor("#f7f7f7")
            # ⚠️ EQUAL SCALES on y and z. The wake is 30 mm wide and only 4.5-6 mm
            # deep; stretching z to fill the panel distorts the geometry and makes
            # the stir zone look narrower and the defects taller than they are.
            # The cross-section must be read like a macrograph, so 1 mm in y is
            # 1 mm in z.
            ax.set_aspect("equal", adjustable="box")
            for s in ax.spines.values():
                s.set_color("#cccccc")
        axes.ravel()[n - 1].set_xlabel(
            "y  [mm]      (advancing / retreating — see per-case sense)",
            fontsize=8, color=INK2)
        cb = fig.colorbar(sc, ax=axes, fraction=0.016, pad=0.02)
        cb.set_label("ln(Φ/Φ₀)   (log scale, shared across panels)",
                     fontsize=8, color=INK2)
        cb.ax.tick_params(labelsize=7, colors=INK2)
        fig.suptitle(f"Wake box, yz projection — {gname}", fontsize=11,
                     color="#1a3a5c", x=0.01, ha="left")
        p = os.path.join(outdir, f"figW_{gname}{_part}_yz.png")
        fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  wrote {p}")


def make_figs_binned(panels, outdir: str, nby: int, nbz: int,
                     knn: int, maxr: float) -> None:
    """Smoothed yz damage field, at display resolution rather than bin resolution.

    ⚠️ WHY NOT HISTOGRAM BINS. Binning ties the picture's resolution to how many
    particles land in a cell. With ~88,600 wake particles over ~165 mm2 the mean
    spacing is 0.18 mm, so a 200x80 grid already has a median of 5 particles per
    cell and 21 % of cells below 4 -- and 600x240 leaves 54 % of cells EMPTY. The
    limit is the particle count, not the choice of bin size, so no bin grid can be
    both fine and populated.

    A k-NEAREST-NEIGHBOUR median breaks that coupling: every output point is the
    median of its k nearest particles, wherever they are. Resolution is then a
    display choice and the estimator stays well-supported everywhere. At 600x240
    (dy = 0.05 mm, 12x finer in y than the 120-wide seed grid) the 16th neighbour
    is a median 0.097 mm away and nothing is blanked.

    Median, not mean: lnPhi spans five decades and is heavily right-skewed, so a
    mean maps outliers instead of where the bulk of the material is damaged.
    The median is taken in LOG10 space, which is also what the colour scale shows.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from scipy.spatial import cKDTree

    os.makedirs(outdir, exist_ok=True)
    INK2 = "#555555"

    allv = np.concatenate([p[2] for p in panels if p[2].size])
    lo = max(float(np.percentile(allv[allv > 0], 5)), 1e-3) if (allv > 0).any() else 1e-3
    hi = float(np.percentile(allv, 99.0))

    groups: dict[str, list] = {}
    for tag, pos, ln, box in panels:
        groups.setdefault(famof(tag), []).append((tag, pos, ln, box))

    for gname, all_items in groups.items():
      for _k in range(0, len(all_items), PER_FIG):
        items = all_items[_k:_k + PER_FIG]
        _part = f"_{_k // PER_FIG + 1}" if len(all_items) > PER_FIG else ""
        n = len(items)
        # One column: with equal y/z scales a wake panel is ~30 x 5.5 mm.
        fig, axes = plt.subplots(n, 1, figsize=(12.5, 2.45 * n), squeeze=False)
        im = None
        for ax, (tag, pos, ln, box) in zip(axes.ravel(), items):
            y, zz = pos[:, 1] * 1e3, pos[:, 2] * 1e3
            L = np.log10(np.maximum(ln, 1e-4))
            tree = cKDTree(np.column_stack([y, zz]))
            gy = np.linspace(y.min(), y.max(), nby)
            gz = np.linspace(zz.min(), zz.max(), nbz)
            GY, GZ = np.meshgrid(gy, gz, indexing="ij")
            d, idx = tree.query(np.column_stack([GY.ravel(), GZ.ravel()]),
                                k=knn, workers=-1)
            fld = np.median(L[idx], axis=1)
            fld[d[:, -1] > maxr] = np.nan        # genuinely empty -> blank
            fld = np.power(10.0, fld).reshape(nby, nbz)

            im = ax.pcolormesh(gy, gz, np.clip(fld.T, lo, hi), cmap="inferno",
                               norm=LogNorm(vmin=lo, vmax=hi), shading="auto",
                               rasterized=True)
            ax.text(0.006, 0.955, tag.replace("ps_", "").replace("_", " "),
                    transform=ax.transAxes, fontsize=9, color="white",
                    va="top", ha="left", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#00000088", ec="none"))
            ax.set_ylabel("z  [mm]", fontsize=8, color=INK2)
            ax.tick_params(labelsize=7, colors=INK2)
            ax.set_facecolor("#eeeeee")
            # 1 mm in y is 1 mm in z: the wake is 30 mm wide and 5.5 mm deep, so
            # stretching z would make the weld look like a tall cone.
            ax.set_aspect("equal", adjustable="box")
            for sp in ax.spines.values():
                sp.set_color("#cccccc")
        axes.ravel()[n - 1].set_xlabel(
            "y  [mm]      (advancing / retreating — see per-case sense)",
            fontsize=8, color=INK2)
        cb = fig.colorbar(im, ax=axes, fraction=0.016, pad=0.02)
        cb.set_label(f"ln(Φ/Φ₀), {knn}-NN median   (log, shared)",
                     fontsize=8, color=INK2)
        cb.ax.tick_params(labelsize=7, colors=INK2)
        fig.suptitle(f"Wake box, yz damage field — {gname}", fontsize=11,
                     color="#1a3a5c", x=0.01, ha="left")
        out = os.path.join(outdir, f"figWB_{gname}{_part}_yz_field.png")
        fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
