#!/usr/bin/env python3
"""
Discrete-curvature diagnostics for the ROM case manifold.

Measures HOW curved / bumpy the parameter-to-snapshot map is, using
secant-vector geometry in the high-dimensional snapshot space:
  * turning-angle curvature along 1D parameter paths,
  * Menger curvature (dimensionless, scale-normalised) along the same paths,
  * local-PCA curvature per case (k-NN; fragile at n~20 — comparative only).

Complements run_rom_manifold.py (intrinsic dim / MDS). Produces a per-case
curvature map over the (v_adv, omega) plane and prints a cross-manifold
summary. See docs/manifold_curvature.md.

Usage
-----
  python run_rom_curvature.py --target density
  python run_rom_curvature.py --target particles --components y z
  python run_rom_curvature.py --target density --project-2d sum --tag 2d
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
    p.add_argument("--density-filename", default="particles_union_density.vtkhdf")
    p.add_argument("--project-2d", default=None, choices=["sum", "slice"])
    p.add_argument("--x-window", type=float, nargs=2, default=None)
    p.add_argument("--mode", default="comoving",
                   help="Particle snapshot mode (ignored for density).")
    p.add_argument("--components", nargs="+", default=["x", "y", "z"],
                   choices=["x", "y", "z"])
    p.add_argument("--exclude", nargs="*", default=None)
    p.add_argument("--intrinsic-dim", type=int, default=None,
                   help="Local-PCA tangent dim (default: 2 density, 3 particles).")
    p.add_argument("--local-pca-k", type=int, default=5)
    p.add_argument("--tag", default="")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def make_plot(rep, target, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    cn = rep.case_numbers
    v = rep.params[:, 0]
    w = rep.params[:, 1]
    # per-case vectors aligned to case order (nan if a case has no interior value)
    turn = np.array([rep.turning_angle_deg.get(c, np.nan) for c in cn])
    meng = np.array([rep.menger.get(c, np.nan) for c in cn])
    lpca = np.array([rep.local_pca.get(c, np.nan) for c in cn])

    panels = [("turning angle [deg]", turn),
              ("Menger curvature (dimensionless)", meng),
              (f"local-PCA curvature (k={rep.local_pca_k}, d={rep.local_pca_dim})", lpca)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (ttl, val) in zip(axes, panels):
        m = np.isfinite(val)
        if m.sum() >= 3:
            try:
                tcf = ax.tricontourf(Triangulation(v[m], w[m]), val[m],
                                     levels=12, cmap="magma")
                fig.colorbar(tcf, ax=ax, fraction=0.046)
            except Exception:
                pass
        sc = ax.scatter(v, w, c=val, cmap="magma", edgecolors="k", s=70, zorder=3)
        for vi, wi, ci in zip(v, w, cn):
            ax.annotate(ci, (vi, wi), fontsize=6, ha="center", va="center",
                        color="cyan", zorder=4)
        ax.set_xlabel("v_adv"); ax.set_ylabel("omega_pin")
        ax.set_title(ttl)
    fig.suptitle(f"Manifold curvature over the (v_adv, omega) plane — {target}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[curv] wrote {out_path}")


def main():
    a = parse_args()
    from jaxtrace.rom import (load_dataset, load_particle_dataset,
                              analyze_curvature)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{a.tag}" if a.tag else ""

    if a.target == "density":
        excl = tuple(a.exclude) if a.exclude is not None else ()
        ds = load_dataset(exclude=excl, density_filename=a.density_filename,
                          project_2d=a.project_2d,
                          x_window=(tuple(a.x_window) if a.x_window else None),
                          verbose=False)
        idim = a.intrinsic_dim if a.intrinsic_dim is not None else 2
    else:
        excl = tuple(a.exclude) if a.exclude is not None else ()
        ds = load_particle_dataset(mode=a.mode, components=a.components,
                                   exclude=excl, verbose=False)
        idim = a.intrinsic_dim if a.intrinsic_dim is not None else 3

    print(f"[curv] target={a.target}, {ds.matrix.shape[0]} cases, "
          f"{ds.matrix.shape[1]} dims, local-PCA d={idim} k={a.local_pca_k}")
    rep = analyze_curvature(ds.matrix, ds.params, ds.case_numbers,
                            local_pca_k=a.local_pca_k, intrinsic_dim=idim)

    s = rep.summary
    print("\n[curv] summary:")
    print(f"  turning angle:  mean {s['turning_angle_mean_deg']:.1f} deg, "
          f"max {s['turning_angle_max_deg']:.1f} deg")
    print(f"  Menger (dimensionless): mean {s['menger_mean']:.4f}, "
          f"max {s['menger_max']:.4f}  (length scale {s['length_scale']:.4g})")
    print(f"  local-PCA curvature: mean {s['local_pca_mean']:.4f}, "
          f"max {s['local_pca_max']:.4f}")

    # rank the most-curved cases (by local-PCA and by Menger)
    def _top(d, n=5):
        return sorted(d.items(), key=lambda kv: -kv[1])[:n]
    print("\n[curv] most-curved cases (local-PCA):",
          [(c, round(x, 3)) for c, x in _top(rep.local_pca)])
    print("[curv] most-curved cases (Menger):    ",
          [(c, round(x, 3)) for c, x in _top(rep.menger)])

    np.savez_compressed(
        a.out_dir / f"rom_curvature_{a.target}{tag}.npz",
        target=a.target, case_numbers=np.array(rep.case_numbers),
        params=rep.params,
        turning=np.array([rep.turning_angle_deg.get(c, np.nan)
                          for c in rep.case_numbers]),
        menger=np.array([rep.menger.get(c, np.nan) for c in rep.case_numbers]),
        local_pca=np.array([rep.local_pca.get(c, np.nan)
                            for c in rep.case_numbers]),
        **{f"summary_{k}": v for k, v in s.items()},
    )
    if not a.no_plot:
        make_plot(rep, a.target,
                  a.out_dir / f"rom_curvature_{a.target}{tag}.png")
    print("[curv] done.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
