#!/usr/bin/env python3
"""
PCA feasibility study for the data-driven density surrogate.

Loads the 17 common-grid FOM density snapshots, runs a mean-centred SVD
(POD), and writes the diagnostic the study hinges on: an elbow plot of
the singular-value spectrum alongside the cumulative energy-coverage
curve. This tells us how many modes are needed to describe the density
response to ``(v_adv, omega_pin)``.

Usage
-----
    python run_rom_pca.py                       # defaults, writes PNG + npz
    python run_rom_pca.py --resolution 160 64 64
    python run_rom_pca.py --fom-root /path/to/FOM --out-dir rom_out
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    from jaxtrace.rom.dataset import DEFAULT_EXCLUDE, DEFAULT_FOM_ROOT

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fom-root", type=Path, default=DEFAULT_FOM_ROOT,
                   help="Root of the FOM cases.")
    p.add_argument("--out-dir", type=Path, default=Path("rom_out"),
                   help="Where to write the plot and the PCA .npz.")
    p.add_argument("--resolution", type=int, nargs=3, default=None,
                   metavar=("NX", "NY", "NZ"),
                   help="Common-grid resolution. Default: from median native spacing.")
    p.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE),
                   help="Case numbers to drop (default: 000 001 002).")
    p.add_argument("--coverage", type=float, nargs="+",
                   default=[0.90, 0.95, 0.99],
                   help="Coverage thresholds to annotate / report.")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip matplotlib; just compute and save the .npz.")
    return p.parse_args()


def make_plot(pca, out_path: Path, coverage_levels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = pca.singular_values
    k = np.arange(1, len(S) + 1)
    cum = pca.cumulative_energy * 100.0
    frac = pca.energy_fraction * 100.0

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    # --- left: singular-value / energy elbow (log scale) ---
    ax0.semilogy(k, S, "o-", color="C0", label="singular values $\\sigma_k$")
    ax0.set_xlabel("mode index $k$")
    ax0.set_ylabel("singular value $\\sigma_k$ (log)")
    ax0.set_title("Elbow: singular-value spectrum")
    ax0.grid(True, which="both", alpha=0.3)
    ax0b = ax0.twinx()
    ax0b.semilogy(k, frac, "s--", color="C3", alpha=0.7,
                  label="per-mode energy %")
    ax0b.set_ylabel("per-mode energy fraction [%] (log)", color="C3")
    ax0b.tick_params(axis="y", labelcolor="C3")
    ax0.set_xticks(k)

    # --- right: cumulative coverage ---
    ax1.plot(k, cum, "o-", color="C2")
    ax1.set_xlabel("number of modes retained")
    ax1.set_ylabel("cumulative energy coverage [%]")
    ax1.set_title("Cumulative variance coverage")
    ax1.set_ylim(min(0, cum[0] - 5), 101)
    ax1.set_xticks(k)
    ax1.grid(True, alpha=0.3)
    for cl in coverage_levels:
        nm = pca.n_modes_for(cl)
        ax1.axhline(cl * 100, color="grey", ls=":", lw=1)
        ax1.axvline(nm, color="grey", ls=":", lw=1)
        ax1.annotate(f"{int(cl*100)}%: {nm} modes",
                     xy=(nm, cl * 100), xytext=(5, -12),
                     textcoords="offset points", fontsize=9)

    fig.suptitle("PCA feasibility — density surrogate (v_adv, omega_pin) "
                 f"→ ρ̄, {pca.n_cases} snapshots", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[rom] wrote plot: {out_path}")


def main() -> int:
    args = parse_args()
    from jaxtrace.rom import fit_pca, load_dataset

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        fom_root=args.fom_root,
        exclude=tuple(args.exclude),
        resolution=tuple(args.resolution) if args.resolution else None,
    )
    print(f"[rom] snapshot matrix: {ds.matrix.shape} "
          f"({ds.matrix.nbytes / 1024**2:.1f} MiB)")

    pca = fit_pca(ds.matrix)

    # Report.
    print("\n[rom] singular values:")
    for i, s in enumerate(pca.singular_values):
        print(f"   mode {i:2d}: sigma={s:.6e}  "
              f"energy%={pca.energy_fraction[i]*100:7.3f}  "
              f"cum%={pca.cumulative_energy[i]*100:7.3f}")
    print("\n[rom] modes needed for coverage:")
    for cl in args.coverage:
        print(f"   {int(cl*100)}% : {pca.n_modes_for(cl)} modes")

    # Persist for downstream regression (coeffs vs params).
    npz_path = args.out_dir / "rom_pca.npz"
    np.savez_compressed(
        npz_path,
        mean=pca.mean,
        modes=pca.modes,
        singular_values=pca.singular_values,
        coeffs=pca.coeffs,
        params=ds.params,
        case_numbers=np.array(ds.case_numbers),
        grid_origin=ds.grid.origin,
        grid_spacing=ds.grid.spacing,
        grid_shape=np.array(ds.grid.shape),
    )
    print(f"\n[rom] wrote PCA data: {npz_path} "
          f"({npz_path.stat().st_size / 1024**2:.1f} MiB)")

    if not args.no_plot:
        make_plot(pca, args.out_dir / "rom_pca_elbow_coverage.png", args.coverage)

    print("[rom] done.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
