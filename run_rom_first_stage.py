#!/usr/bin/env python3
"""
Second-stage ROM with first-stage (field-POD) coefficients as inputs.

Tests whether replacing or augmenting the two raw scalars (v_adv,
omega_pin) with the colleague's first-stage ROM coefficients
(Displacement/velocity, Pressure, Temperature reduced coordinates)
improves the density / particle coefficient regression (LOOCV).

Outputs a bar-chart of LOOCV error per input configuration and a
.npz with the raw numbers. Files use a 'rom_firststage' prefix so they
do not collide with earlier outputs.

Usage
-----
    python run_rom_first_stage.py --target density
    python run_rom_first_stage.py --target particles --exclude 001
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
    p.add_argument("--target", choices=["density", "particles"], default="density")
    p.add_argument("--regressor", default="rbf")
    p.add_argument("--exclude", nargs="*", default=None,
                   help="Cases to drop. Default: density 000/001/002, "
                        "particles 001.")
    p.add_argument("--x-keep-fraction", type=float, default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


# input configurations: (label, fields, modes-cap, use_base_features)
CONFIGS = [
    ("(v,ω) baseline",        None,                                      None, "base_only"),
    ("velocity ALONE",        ["Displacement"],                          None, "extra_only"),
    ("(v,ω)+velocity",        ["Displacement"],                          None, "both"),
    ("temp ALONE",            ["Temperature"],                           None, "extra_only"),
    ("(v,ω)+temp",            ["Temperature"],                           None, "both"),
    ("all fields ALONE",      ["Displacement", "Temperature", "Pressure"], None, "extra_only"),
    ("(v,ω)+all fields",      ["Displacement", "Temperature", "Pressure"], None, "both"),
]


def main():
    args = parse_args()
    from jaxtrace.rom import load_dataset, load_particle_dataset, loocv
    from jaxtrace.rom.first_stage import load_first_stage_coeffs

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    if args.target == "density":
        excl = tuple(args.exclude) if args.exclude is not None else ("000", "001", "002")
        ds = load_dataset(exclude=excl, x_keep_fraction=args.x_keep_fraction,
                          verbose=False)
    else:
        excl = tuple(args.exclude) if args.exclude is not None else ("001",)
        ds = load_particle_dataset(mode="comoving", exclude=excl, verbose=False)

    fs = load_first_stage_coeffs(fom_root=args.fom_root)
    print(f"[fs] target={args.target}, {ds.matrix.shape[0]} cases, "
          f"first-stage modes: "
          f"{ {f: fs.coeffs[f].shape[1] for f in fs.coeffs} }")

    labels, errors, ks, nfeat = [], [], [], []
    for label, fields, modes, mode_flag in CONFIGS:
        if fields is None:
            ef, useb = None, True
        else:
            ef = fs.select(fields=fields, modes=modes, case_numbers=ds.case_numbers)
            useb = (mode_flag == "both")
        res = loocv(ds.matrix, ds.params, ds.case_numbers,
                    regressor=args.regressor, extra_features=ef,
                    use_base_features=(useb if fields is not None else True),
                    verbose=False)
        nf = (2 if (fields is None or useb) else 0) + (0 if ef is None else ef.shape[1])
        labels.append(label)
        errors.append(res.mean_error.min() * 100)
        ks.append(res.best_k)
        nfeat.append(nf)
        print(f"  {label:22s} err={errors[-1]:5.1f}%  K={ks[-1]}  n_feat={nf}")

    npz = args.out_dir / f"rom_firststage_{args.target}{tag}.npz"
    np.savez_compressed(npz, labels=np.array(labels), errors=np.array(errors),
                        best_k=np.array(ks), n_features=np.array(nfeat),
                        target=args.target, regressor=args.regressor)
    print(f"[fs] wrote {npz}")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["C0" if "baseline" in l else
                  ("C1" if "ALONE" in l else "C2") for l in labels]
        bars = ax.bar(range(len(labels)), errors, color=colors)
        ax.axhline(errors[0], color="C0", ls="--", lw=1, alpha=0.7,
                   label="(v,ω) baseline")
        for b, k, nf in zip(bars, ks, nfeat):
            ax.annotate(f"K={k}\n{nf}f", (b.get_x() + b.get_width() / 2,
                        b.get_height()), ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("LOOCV mean held-out error [%]")
        ax.set_title(f"First-stage ROM coeffs as 2nd-stage inputs — "
                     f"{args.target} ({args.regressor})")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = args.out_dir / f"rom_firststage_{args.target}{tag}.png"
        fig.savefig(out, dpi=140)
        print(f"[fs] wrote {out}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
