#!/usr/bin/env python3
"""
Manifold-insight diagnostics for the ROM case manifold.

Each FOM case is one high-dim snapshot; the cases trace a manifold
parametrised by (v_adv, omega_pin). This driver characterises that
manifold with LINEAR / metric methods first (trustworthy), before any
nonlinear embedding:

  * PCA explained-variance spectrum   -> linear dimensionality
  * linearity residual vs #PCs        -> curvature signal
  * intrinsic-dimension estimates     -> TwoNN + correlation dimension
  * pairwise distance matrix          -> case clustering
  * classical MDS (2D)                -> distance-faithful layout
  * parameter overlay                 -> is it smoothly (v_adv,omega)-organised?

A high linear dimension together with a LOW intrinsic dimension is the
signature of a curved (nonlinear) manifold — the case where an
autoencoder/kernel method could eventually help.

Target: particle final-step clouds (default) or density fields.

Usage
-----
    python run_rom_manifold.py --target particles --exclude 000 001
    python run_rom_manifold.py --target density
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    from jaxtrace.rom.dataset import DEFAULT_FOM_ROOT
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fom-root", type=Path, default=DEFAULT_FOM_ROOT)
    p.add_argument("--out-dir", type=Path, default=Path("rom_out"))
    p.add_argument("--density-filename", default="particles_union_density.vtkhdf",
                   help="Density product: particles_union_density.vtkhdf (union/time-avg, default) or finalstep_union_density.vtkhdf (final step).")
    p.add_argument("--project-2d", default=None, choices=["sum","slice"],
                   help="Reduce 3D density to a 2D (y,z) cross-section: sum over an x-window (slab) or single x-slice.")
    p.add_argument("--x-window", type=float, nargs=2, default=None,
                   metavar=("XLO","XHI"), help="x-window [m] for --project-2d sum (default: data band).")
    p.add_argument("--x-slice", type=float, default=None,
                   help="x position [m] for --project-2d slice (default: peak-mass x).")
    p.add_argument("--target", choices=["particles", "density"], default="particles")
    p.add_argument("--mode", default="final",
                   help="Particle snapshot mode (final/raw/comoving). Ignored for density.")
    p.add_argument("--components", nargs="+", default=["x", "y", "z"],
                   choices=["x", "y", "z"],
                   help="Particle components (use 'y z' for the 2D y,z study).")
    p.add_argument("--exclude", nargs="*", default=None,
                   help="Cases to drop. Default: particles 000 001 (runaways); "
                        "density 000 001 002.")
    p.add_argument("--clip-percentile", type=float, default=99.9,
                   help="Per-feature clip to tame residual outliers (0 disables).")
    p.add_argument("--tag", default="")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def make_plots(rep, target, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # (1) explained-variance spectrum
    ax = axes[0, 0]
    evr = rep.spectrum.explained_variance_ratio
    k = np.arange(1, len(evr) + 1)
    ax.bar(k, evr * 100, color="C0", alpha=0.7, label="per-PC %")
    ax2 = ax.twinx()
    ax2.plot(k, rep.spectrum.cumulative * 100, "o-", color="C3", label="cumulative %")
    ax2.axhline(90, color="grey", ls=":", lw=1)
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance [%]")
    ax2.set_ylabel("cumulative [%]", color="C3")
    ax.set_title(f"Linear spectrum (90%: {rep.spectrum.n_components_for(0.9)} PCs)")

    # (2) linearity residual vs #PCs
    ax = axes[0, 1]
    kk = np.arange(1, len(rep.residual) + 1)
    ax.plot(kk, rep.residual * 100, "o-", color="C2")
    ax.set_xlabel("number of PCs")
    ax.set_ylabel("relative L2 residual [%]")
    ax.set_title("Linearity: reconstruction residual\n"
                 f"(intrinsic dim: TwoNN={rep.id_twonn:.1f}, "
                 f"corr={rep.id_correlation:.1f})")
    ax.grid(True, alpha=0.3)

    # (3) pairwise distance matrix
    ax = axes[1, 0]
    im = ax.imshow(rep.distances, cmap="viridis")
    ax.set_xticks(range(len(rep.case_numbers)))
    ax.set_yticks(range(len(rep.case_numbers)))
    ax.set_xticklabels(rep.case_numbers, rotation=90, fontsize=6)
    ax.set_yticklabels(rep.case_numbers, fontsize=6)
    ax.set_title("Case-to-case distance matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (4) MDS layout coloured by v_adv (size ~ |omega|)
    ax = axes[1, 1]
    c = rep.mds_coords
    v = rep.params[:, 0]
    w = np.abs(rep.params[:, 1])
    sizes = 40 + 200 * (w - w.min()) / (np.ptp(w) + 1e-30)
    sc = ax.scatter(c[:, 0], c[:, 1], c=v, s=sizes, cmap="plasma",
                    edgecolors="k", zorder=3)
    for (x, y), cn in zip(c, rep.case_numbers):
        ax.annotate(cn, (x, y), fontsize=6, ha="center", va="center")
    a = rep.alignment
    ax.set_xlabel(f"MDS axis 1  (corr v={a['axis1'][0]:+.2f}, ω={a['axis1'][1]:+.2f})")
    ax.set_ylabel(f"MDS axis 2  (corr v={a['axis2'][0]:+.2f}, ω={a['axis2'][1]:+.2f})")
    ax.set_title("Classical MDS (colour=v_adv, size=|ω|)")
    fig.colorbar(sc, ax=ax, fraction=0.046, label="v_adv")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"ROM case-manifold diagnostics — {target} "
                 f"({len(rep.case_numbers)} cases)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[mfld] wrote {out_path}")


def main():
    args = parse_args()
    from jaxtrace.rom import (
        load_particle_dataset, load_dataset, analyze_manifold,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    if args.target == "particles":
        excl = tuple(args.exclude) if args.exclude is not None else ("000", "001")
        ds = load_particle_dataset(mode=args.mode, components=args.components,
                                   exclude=excl, verbose=False)
    else:
        excl = tuple(args.exclude) if args.exclude is not None else ("000", "001", "002")
        ds = load_dataset(exclude=excl, verbose=False,
        density_filename=args.density_filename,
        project_2d=args.project_2d, x_window=(tuple(args.x_window) if args.x_window else None), x_slice=args.x_slice,
    )

    print(f"[mfld] target={args.target}, {ds.matrix.shape[0]} cases, "
          f"{ds.matrix.shape[1]} dims, excluded {list(excl)}")

    clip = None if args.clip_percentile <= 0 else args.clip_percentile
    rep = analyze_manifold(ds.matrix, ds.params, ds.case_numbers, clip_percentile=clip)

    print(f"\n[mfld] LINEAR spectrum: PCs for 90/95/99% = "
          f"{[rep.spectrum.n_components_for(f) for f in (.9,.95,.99)]}")
    print(f"[mfld]   PC1 explains {rep.spectrum.explained_variance_ratio[0]*100:.1f}%")
    print(f"[mfld] INTRINSIC dim: TwoNN={rep.id_twonn:.2f}  "
          f"correlation={rep.id_correlation:.2f}")
    nneg = int((rep.mds_eigenvalues < 0).sum())
    print(f"[mfld] MDS: {nneg}/{len(rep.mds_eigenvalues)} negative eigenvalues "
          f"(non-Euclidean/curvature signal)")
    a = rep.alignment
    print(f"[mfld] MDS axis1 corr: v_adv={a['axis1'][0]:+.3f} ω={a['axis1'][1]:+.3f}")
    print(f"[mfld] MDS axis2 corr: v_adv={a['axis2'][0]:+.3f} ω={a['axis2'][1]:+.3f}")

    # interpretation hint
    lin90 = rep.spectrum.n_components_for(0.9)
    idim = rep.id_correlation
    if lin90 >= 5 and np.isfinite(idim) and idim < lin90 / 2:
        print(f"\n[mfld] => HIGH linear dim ({lin90} PCs for 90%) but LOW intrinsic "
              f"dim (~{idim:.1f}): the manifold is CURVED/nonlinear.")
    elif lin90 <= 3:
        print(f"\n[mfld] => Low linear dim ({lin90} PCs for 90%): manifold is "
              f"essentially linear; nonlinear embedding adds little.")

    np.savez_compressed(
        args.out_dir / f"rom_manifold_{args.target}{tag}.npz",
        target=args.target, case_numbers=np.array(rep.case_numbers),
        params=rep.params, evr=rep.spectrum.explained_variance_ratio,
        distances=rep.distances, mds_coords=rep.mds_coords,
        mds_eigenvalues=rep.mds_eigenvalues, residual=rep.residual,
        id_twonn=rep.id_twonn, id_correlation=rep.id_correlation,
    )
    if not args.no_plot:
        make_plots(rep, args.target,
                   args.out_dir / f"rom_manifold_{args.target}{tag}.png")
    print("[mfld] done.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
