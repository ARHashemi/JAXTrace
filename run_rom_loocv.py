#!/usr/bin/env python3
"""
Leave-one-out cross-validation of the PCA + regression density surrogate.

For each of the 17 cases: fit PCA on the other 16, fit a regressor
(inputs -> mode coefficients) on them, predict the held-out coefficients
from that case's (v_adv, omega_pin), reconstruct its density field, and
measure the relative L2 error. Sweeps the number of retained modes and
compares the rbf / gp / poly regressors.

Outputs (in --out-dir):
  * rom_loocv_error_vs_modes.png   mean held-out error vs #modes, per regressor
  * rom_loocv_error_maps.png       2D (v_adv, omega) error colormap per regressor
  * rom_loocv.npz                  raw per-case errors

Usage
-----
    python run_rom_loocv.py
    python run_rom_loocv.py --normalize log --regressors rbf gp
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    from jaxtrace.rom.dataset import (
        DEFAULT_EXCLUDE, DEFAULT_FOM_ROOT, DEFAULT_TRIM_LO, DEFAULT_TRIM_HI,
    )
    from jaxtrace.rom import NORMALIZE_CHOICES, REGRESSORS

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fom-root", type=Path, default=DEFAULT_FOM_ROOT)
    p.add_argument("--out-dir", type=Path, default=Path("rom_out"))
    p.add_argument("--resolution", type=int, nargs=3, default=None,
                   metavar=("NX", "NY", "NZ"))
    p.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE))
    p.add_argument("--trim-lo", type=int, nargs=3, default=list(DEFAULT_TRIM_LO),
                   metavar=("X", "Y", "Z"))
    p.add_argument("--trim-hi", type=int, nargs=3, default=list(DEFAULT_TRIM_HI),
                   metavar=("X", "Y", "Z"))
    p.add_argument("--no-trim", action="store_true")
    p.add_argument("--normalize", default="none", choices=list(NORMALIZE_CHOICES))
    p.add_argument("--regressors", nargs="+", default=list(REGRESSORS),
                   choices=list(REGRESSORS),
                   help="Regressors to compare (default: rbf gp poly).")
    p.add_argument("--max-modes", type=int, default=None,
                   help="Cap the mode sweep (default: n-2).")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def plot_error_vs_modes(results, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, res in enumerate(results):
        c = f"C{idx}"
        ax.plot(res.k_values, res.mean_error * 100.0, "o-", color=c,
                label=f"{res.regressor}  (best K={res.best_k}, "
                      f"{res.mean_error.min()*100:.1f}%)")
        kb = res.best_k
        ax.axvline(kb, color=c, ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("number of PCA modes retained")
    ax.set_ylabel("mean held-out relative L2 error [%]")
    ax.set_title("LOOCV error vs #modes "
                 f"(normalize={results[0].normalize})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[cv] wrote {out_path}")


def plot_error_maps(results, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), squeeze=False)
    axes = axes[0]

    # Shared color scale across regressors for fair comparison.
    all_err = np.concatenate([r.error_at_best_k() * 100.0 for r in results])
    vmin, vmax = float(all_err.min()), float(all_err.max())

    for ax, res in zip(axes, results):
        v = res.params[:, 0]
        w = res.params[:, 1]
        err = res.error_at_best_k() * 100.0
        # Filled tricontour over the scattered points + the points on top.
        tri = Triangulation(v, w)
        tcf = ax.tricontourf(tri, err, levels=14, cmap="viridis",
                             vmin=vmin, vmax=vmax)
        sc = ax.scatter(v, w, c=err, cmap="viridis", vmin=vmin, vmax=vmax,
                        edgecolors="k", s=60, zorder=3)
        for vi, wi, ci in zip(v, w, res.case_numbers):
            ax.annotate(ci, (vi, wi), fontsize=6, ha="center", va="center",
                        color="white", zorder=4)
        ax.set_xlabel("v_adv  (INLET_VELOCITY)")
        ax.set_ylabel("omega_pin  (PIN_RPM)")
        ax.set_title(f"{res.regressor}  (K={res.best_k})\n"
                     f"mean {err.mean():.1f}%  max {err.max():.1f}%")
        fig.colorbar(tcf, ax=ax, label="held-out rel. L2 error [%]")

    fig.suptitle("LOOCV reconstruction error over the (v_adv, omega_pin) plane "
                 f"(normalize={results[0].normalize})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[cv] wrote {out_path}")


def main() -> int:
    args = parse_args()
    from jaxtrace.rom import load_dataset, loocv

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trim_lo = None if args.no_trim else tuple(args.trim_lo)
    trim_hi = None if args.no_trim else tuple(args.trim_hi)

    ds = load_dataset(
        fom_root=args.fom_root,
        exclude=tuple(args.exclude),
        resolution=tuple(args.resolution) if args.resolution else None,
        trim_lo=trim_lo,
        trim_hi=trim_hi,
    )
    print(f"[cv] snapshot matrix: {ds.matrix.shape}")

    k_values = None
    if args.max_modes is not None:
        k_values = np.arange(1, int(args.max_modes) + 1)

    results = []
    for reg in args.regressors:
        res = loocv(
            ds.matrix, ds.params, ds.case_numbers,
            regressor=reg, normalize=args.normalize, k_values=k_values,
        )
        print(f"[cv] {reg}: best K={res.best_k}, "
              f"mean held-out error={res.mean_error.min()*100:.2f}%  "
              f"(per-case at best K: min={res.error_at_best_k().min()*100:.1f}% "
              f"max={res.error_at_best_k().max()*100:.1f}%)")
        results.append(res)

    # Persist raw errors.
    npz = args.out_dir / "rom_loocv.npz"
    np.savez_compressed(
        npz,
        normalize=args.normalize,
        regressors=np.array(args.regressors),
        params=ds.params,
        case_numbers=np.array(ds.case_numbers),
        k_values=results[0].k_values,
        **{f"rel_error_{r.regressor}": r.rel_error for r in results},
    )
    print(f"[cv] wrote {npz}")

    if not args.no_plot:
        plot_error_vs_modes(results, args.out_dir / "rom_loocv_error_vs_modes.png")
        plot_error_maps(results, args.out_dir / "rom_loocv_error_maps.png")

    print("[cv] done.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
