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
    from jaxtrace.rom import NORMALIZE_CHOICES

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fom-root", type=Path, default=DEFAULT_FOM_ROOT,
                   help="Root of the FOM cases.")
    p.add_argument("--out-dir", type=Path, default=Path("rom_out"),
                   help="Where to write the plot and the PCA .npz.")
    p.add_argument("--density-filename", default="particles_union_density.vtkhdf",
                   help="Density product: particles_union_density.vtkhdf (union/time-avg, default) or finalstep_union_density.vtkhdf (final step).")
    p.add_argument("--resolution", type=int, nargs=3, default=None,
                   metavar=("NX", "NY", "NZ"),
                   help="Common-grid resolution. Default: from median native spacing.")
    p.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE),
                   help="Case numbers to drop (default: 000 001 002).")
    p.add_argument("--coverage", type=float, nargs="+",
                   default=[0.90, 0.95, 0.99],
                   help="Coverage thresholds to annotate / report.")
    # Trim margins (x, y, z) voxels off the low / high walls.
    from jaxtrace.rom.dataset import DEFAULT_TRIM_LO, DEFAULT_TRIM_HI
    p.add_argument("--trim-lo", type=int, nargs=3, default=list(DEFAULT_TRIM_LO),
                   metavar=("X", "Y", "Z"),
                   help="Voxels trimmed off each low wall (default: 1 1 1).")
    p.add_argument("--trim-hi", type=int, nargs=3, default=list(DEFAULT_TRIM_HI),
                   metavar=("X", "Y", "Z"),
                   help="Voxels trimmed off each high wall (default: 1 1 6, "
                        "the extra z_max margin removes the top-layer artefact).")
    p.add_argument("--no-trim", action="store_true",
                   help="Disable trimming entirely.")
    p.add_argument("--x-keep-fraction", type=float, default=None,
                   help="Keep only the first fraction of the x-extent (the "
                        "near-pin region), e.g. 0.2. Applied after the voxel "
                        "trims. Default: keep all of x.")
    p.add_argument("--tag", default="",
                   help="Suffix appended to output filenames so multiple runs "
                        "coexist in one out-dir (e.g. --tag x20).")
    # Normalization: the primary one saved to .npz, plus extras to overlay.
    p.add_argument("--normalize", default="none", choices=list(NORMALIZE_CHOICES),
                   help="Primary normalization for the saved PCA (default: none).")
    p.add_argument("--compare-normalize", nargs="*", default=None,
                   metavar="NAME",
                   help="Extra normalizations to overlay on the coverage plot "
                        f"for comparison. Choices: {list(NORMALIZE_CHOICES)}. "
                        "Default overlay: none per_case_l2 per_case_mass log.")
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


def make_compare_plot(pca_by_name: dict, out_path: Path, coverage_levels):
    """Overlay singular-value spectra and coverage curves across
    normalization variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.5))
    for idx, (name, pca) in enumerate(pca_by_name.items()):
        S = pca.singular_values
        k = np.arange(1, len(S) + 1)
        c = f"C{idx}"
        # Normalised singular values (÷σ_1) so different scalings overlay.
        ax0.semilogy(k, S / S[0], "o-", color=c, label=name, alpha=0.85)
        ax1.plot(k, pca.cumulative_energy * 100.0, "o-", color=c,
                 label=name, alpha=0.85)

    ax0.set_xlabel("mode index $k$")
    ax0.set_ylabel("normalised singular value $\\sigma_k/\\sigma_1$ (log)")
    ax0.set_title("Spectrum (scaled to σ₁) by normalization")
    ax0.grid(True, which="both", alpha=0.3)
    ax0.legend(fontsize=8)

    ax1.set_xlabel("number of modes retained")
    ax1.set_ylabel("cumulative energy coverage [%]")
    ax1.set_title("Coverage by normalization")
    ax1.set_ylim(0, 101)
    ax1.grid(True, alpha=0.3)
    for cl in coverage_levels:
        ax1.axhline(cl * 100, color="grey", ls=":", lw=1)
    ax1.legend(fontsize=8, loc="lower right")

    fig.suptitle("Normalization comparison — density-surrogate PCA",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[rom] wrote comparison plot: {out_path}")


def main() -> int:
    args = parse_args()
    from jaxtrace.rom import fit_pca, load_dataset

    args.out_dir.mkdir(parents=True, exist_ok=True)

    trim_lo = None if args.no_trim else tuple(args.trim_lo)
    trim_hi = None if args.no_trim else tuple(args.trim_hi)

    tag = f"_{args.tag}" if args.tag else ""

    ds = load_dataset(
        fom_root=args.fom_root,
        exclude=tuple(args.exclude),
        resolution=tuple(args.resolution) if args.resolution else None,
        trim_lo=trim_lo,
        trim_hi=trim_hi,
        x_keep_fraction=args.x_keep_fraction,
        density_filename=args.density_filename,
    )
    print(f"[rom] snapshot matrix: {ds.matrix.shape} "
          f"({ds.matrix.nbytes / 1024**2:.1f} MiB)")

    # Primary PCA (the one saved to .npz).
    pca = fit_pca(ds.matrix, normalize=args.normalize)

    # Report.
    print(f"\n[rom] singular values (normalize={args.normalize}):")
    for i, s in enumerate(pca.singular_values):
        print(f"   mode {i:2d}: sigma={s:.6e}  "
              f"energy%={pca.energy_fraction[i]*100:7.3f}  "
              f"cum%={pca.cumulative_energy[i]*100:7.3f}")
    print("\n[rom] modes needed for coverage:")
    for cl in args.coverage:
        print(f"   {int(cl*100)}% : {pca.n_modes_for(cl)} modes")

    # Persist for downstream regression (coeffs vs params).
    npz_path = args.out_dir / f"rom_pca{tag}.npz"
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
        normalize=args.normalize,
        global_scale=pca.global_scale,
        per_case_scale=(pca.per_case_scale if pca.per_case_scale is not None
                        else np.ones(ds.n_cases)),
    )
    print(f"\n[rom] wrote PCA data: {npz_path} "
          f"({npz_path.stat().st_size / 1024**2:.1f} MiB)")

    if not args.no_plot:
        make_plot(pca, args.out_dir / f"rom_pca_elbow_coverage{tag}.png",
                  args.coverage)

        # Comparison overlay across normalizations. Note: global_* options
        # are mathematically identical in *coverage* to none (they only
        # rescale sigma), so the meaningful comparison is none vs per_case_*
        # vs log.
        compare = args.compare_normalize
        if compare is None:
            compare = ["none", "per_case_l2", "per_case_mass", "log"]
        if compare:
            print(f"\n[rom] comparison normalizations: {compare}")
            pca_by_name = {}
            for nm in compare:
                pca_by_name[nm] = (pca if nm == args.normalize
                                   else fit_pca(ds.matrix, normalize=nm))
                p = pca_by_name[nm]
                cov = {int(cl*100): p.n_modes_for(cl) for cl in args.coverage}
                print(f"   {nm:16s} modes for coverage {cov}")
            make_compare_plot(
                pca_by_name,
                args.out_dir / f"rom_pca_normalize_compare{tag}.png",
                args.coverage,
            )

    print("[rom] done.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
