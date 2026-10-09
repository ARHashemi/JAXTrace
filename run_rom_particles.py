#!/usr/bin/env python3
"""
SVD/POD feasibility on particle-cloud snapshots (final step per case).

Every case seeds the same 360k particles on the same t=0 uniform grid
with identical ParticleID ordering, so a per-particle displacement vector
is directly stackable across cases. This driver:

  1. reports the co-moving displacement |xi| statistics (to choose, or
     reject, an 'affected-particle' threshold),
  2. runs PCA on several snapshot representations and overlays their
     elbow / coverage curves (final / raw / comoving / seedrel), and
  3. runs LOOCV of the PCA+regression surrogate for the chosen mode.

Drift for comoving/seedrel is v_adv*t_max with v_adv, DT, N_STEPS read
from each case's run_jaxtrace.sh (authoritative; the CSV dt disagrees).

Outputs go to --out-dir with a 'particles' prefix so they never collide
with the density-field outputs.

Usage
-----
    python run_rom_particles.py --out-dir rom_out
    python run_rom_particles.py --mode comoving --exclude 001 --tag no001
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    from jaxtrace.rom.dataset import DEFAULT_FOM_ROOT
    from jaxtrace.rom import PARTICLE_MODES, NORMALIZE_CHOICES, REGRESSORS

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fom-root", type=Path, default=DEFAULT_FOM_ROOT)
    p.add_argument("--out-dir", type=Path, default=Path("rom_out"))
    p.add_argument("--mode", default="comoving", choices=list(PARTICLE_MODES),
                   help="Snapshot representation for the saved PCA + LOOCV.")
    p.add_argument("--compare-modes", nargs="*",
                   default=["final", "raw", "comoving"],
                   help="Representations to overlay on the elbow/coverage plot.")
    p.add_argument("--drift-source", default="formula",
                   choices=["formula", "empirical"])
    p.add_argument("--components", nargs="+", default=["x", "y", "z"],
                   choices=["x", "y", "z"])
    p.add_argument("--exclude", nargs="*", default=[],
                   help="Case numbers to drop (e.g. 001, the runaway outlier).")
    p.add_argument("--normalize", default="none", choices=list(NORMALIZE_CHOICES))
    p.add_argument("--regressors", nargs="+", default=list(REGRESSORS),
                   choices=list(REGRESSORS))
    p.add_argument("--tag", default="")
    p.add_argument("--no-loocv", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def plot_stats(stats, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    n = len(stats.case_numbers)
    x = np.arange(n)

    # |xi|/dp percentiles per case.
    for j, lvl in enumerate(stats.pct_levels):
        ax0.plot(x, stats.percentiles[:, j], "o-", label=f"p{lvl:g}")
    ax0.set_xticks(x)
    ax0.set_xticklabels(stats.case_numbers, rotation=90, fontsize=7)
    ax0.set_ylabel("|xi| / mean seed spacing")
    ax0.set_title("Co-moving displacement magnitude percentiles")
    ax0.set_yscale("log")
    ax0.grid(True, which="both", alpha=0.3)
    ax0.legend(fontsize=8)

    # fraction affected vs threshold.
    for c in sorted(stats.frac_affected):
        ax1.plot(x, stats.frac_affected[c] * 100.0, "o-",
                 label=f"|xi| > {c:g}·Δp")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats.case_numbers, rotation=90, fontsize=7)
    ax1.set_ylabel("% particles above threshold")
    ax1.set_title("'Affected' fraction vs threshold "
                  f"(Δp={stats.delta_p_mean:.2e})")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[pcv] wrote {out_path}")


def plot_elbow_compare(pca_by_mode, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    for idx, (name, pca) in enumerate(pca_by_mode.items()):
        S = pca.singular_values
        k = np.arange(1, len(S) + 1)
        ax0.semilogy(k, S / S[0], "o-", color=f"C{idx}", label=name)
        ax1.plot(k, pca.cumulative_energy * 100.0, "o-", color=f"C{idx}",
                 label=f"{name} (90%: {pca.n_modes_for(0.9)} modes)")
    ax0.set_xlabel("mode index k")
    ax0.set_ylabel("normalised singular value σ_k/σ_1 (log)")
    ax0.set_title("Particle-cloud spectrum by representation")
    ax0.grid(True, which="both", alpha=0.3)
    ax0.legend(fontsize=8)
    ax1.set_xlabel("number of modes retained")
    ax1.set_ylabel("cumulative coverage [%]")
    ax1.set_title("Coverage by representation")
    ax1.set_ylim(0, 101)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    fig.suptitle("Particle-cloud POD feasibility", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[pcv] wrote {out_path}")


def main() -> int:
    args = parse_args()
    from jaxtrace.rom import (
        load_particle_dataset, compute_displacement_stats, fit_pca, loocv,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    pre = "rom_particles"

    # --- primary dataset (for stats, saved PCA, LOOCV) ---
    ds = load_particle_dataset(
        fom_root=args.fom_root, mode=args.mode,
        components=args.components, exclude=tuple(args.exclude),
        drift_source=args.drift_source,
    )
    print(f"[pcv] matrix {ds.matrix.shape} "
          f"({ds.matrix.nbytes/1024**2:.1f} MiB)")

    # --- displacement statistics ---
    stats = compute_displacement_stats(ds)
    print("\n[pcv] |xi|/Δp percentiles (median, p99, max) per case:")
    for i, c in enumerate(ds.case_numbers):
        print(f"   {c}: p50={stats.percentiles[i,0]:6.1f}  "
              f"p99={stats.percentiles[i,3]:7.1f}  "
              f"max={stats.percentiles[i,-1]:8.1f}")
    print("\n[pcv] fraction 'affected' (|xi|>c·Δp), mean over cases:")
    for c in sorted(stats.frac_affected):
        print(f"   c={c:>4g}: {stats.frac_affected[c].mean()*100:5.1f}%")

    # --- elbow/coverage comparison across representations ---
    pca_by_mode = {}
    for m in args.compare_modes:
        src = args.drift_source
        d = (ds if m == args.mode and args.drift_source == src
             else load_particle_dataset(
                 fom_root=args.fom_root, mode=m, components=args.components,
                 exclude=tuple(args.exclude), drift_source=src, verbose=False))
        pca_by_mode[m] = fit_pca(d.matrix, normalize=args.normalize)
        p = pca_by_mode[m]
        print(f"[pcv] {m:9s} coverage 90/95/99 = "
              f"{[p.n_modes_for(c) for c in (.9,.95,.99)]}  "
              f"mode1 energy%={p.energy_fraction[0]*100:.1f}")

    # --- save primary PCA ---
    pca = pca_by_mode.get(args.mode) or fit_pca(ds.matrix, normalize=args.normalize)
    npz = args.out_dir / f"{pre}_pca_{args.mode}{tag}.npz"
    np.savez_compressed(
        npz, mode=args.mode, drift_source=args.drift_source,
        singular_values=pca.singular_values, coeffs=pca.coeffs,
        params=ds.params, case_numbers=np.array(ds.case_numbers),
        seed_positions=ds.seed_positions, delta_p=ds.delta_p,
        drift_used=ds.drift_used,
    )
    print(f"[pcv] wrote {npz}")

    if not args.no_plot:
        plot_stats(stats, args.out_dir / f"{pre}_displacement_stats{tag}.png")
        plot_elbow_compare(
            pca_by_mode, args.out_dir / f"{pre}_elbow_compare{tag}.png")

    # --- LOOCV on the primary mode ---
    if not args.no_loocv:
        print(f"\n[pcv] LOOCV ({args.mode}, normalize={args.normalize}):")
        results = []
        for reg in args.regressors:
            res = loocv(ds.matrix, ds.params, ds.case_numbers,
                        regressor=reg, normalize=args.normalize, verbose=False)
            print(f"   {reg:5s} best K={res.best_k}, "
                  f"mean error={res.mean_error.min()*100:.2f}%")
            results.append(res)
        np.savez_compressed(
            args.out_dir / f"{pre}_loocv_{args.mode}{tag}.npz",
            mode=args.mode, params=ds.params,
            case_numbers=np.array(ds.case_numbers),
            k_values=results[0].k_values,
            **{f"rel_error_{r.regressor}": r.rel_error for r in results},
        )
        if not args.no_plot:
            # reuse the density LOOCV plotters.
            from run_rom_loocv import plot_error_vs_modes, plot_error_maps
            plot_error_vs_modes(
                results, args.out_dir / f"{pre}_loocv_error_vs_modes_{args.mode}{tag}.png")
            plot_error_maps(
                results, args.out_dir / f"{pre}_loocv_error_maps_{args.mode}{tag}.png")

    print("[pcv] done.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
