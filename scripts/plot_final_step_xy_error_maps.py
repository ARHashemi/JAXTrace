"""plot_final_step_xy_error_maps.py -- per-case xy scatter of final-step
mesh particle positions coloured by grid-vs-mesh relative error.

For every case in the 20-case cohort, plot every particle's mesh-side
final position projected to the (x, y) plane, coloured by

    relative_error = |x_grid - x_mesh| / max(|x_mesh(step) - x_seed|, eps)

which is "how much did the grid tracker's position disagree with the
mesh tracker's, as a fraction of the distance the mesh tracker actually
moved the particle from its seed".  Values pinned to [0, 100 %] for the
colour scale; a "%travel" annotation is written next to each subplot
title so cases with mostly-stationary particles can be spotted.

Also emits a raw-|err| version (mm scale) alongside so the reader can
compare absolute vs relative framings.

Outputs (one figure per variant per metric):
    <out>/xy_error_<VARIANT>_step<S>_pct.png    -- colour = %travel
    <out>/xy_error_<VARIANT>_step<S>_mm.png     -- colour = raw mm

Usage:
    python3 scripts/plot_final_step_xy_error_maps.py \\
        --vtu-dir /scratch/shared/ROM/rom_out/fom_grid_pt_all20 \\
        --fom-root /scratch/shared/ROM/FOM \\
        --variants 4lvl_hct malmo6_hct \\
        --step 2000 \\
        --out paper_figs/rom_pt_step5_step6_all20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import h5py
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  9,
    "figure.titlesize": 15,
})

CASES = [f"{i:03d}" for i in range(20)]
OUTLET_X = 0.030


def _find_mesh_ref(fom_root: Path, case: str) -> Path | None:
    d = fom_root / f"cylindrical_{case}.gid"
    for sub in ("post_pt/fom_hct_on", "post_pt/run_grid-frac_n360000_s2000"):
        for hdf in (d / sub).glob("**/particles.vtkhdf"):
            return hdf
    p = d / "post_pt" / "run_grid-frac_n360000_s2000" / "particles.vtkhdf"
    return p if p.exists() else None


def _load_seed(hdf: Path) -> np.ndarray:
    with h5py.File(hdf, "r") as f:
        offsets = f["VTKHDF/Steps/PointOffsets"][:]
        n = int(offsets[1] - offsets[0]) if len(offsets) > 1 \
            else f["VTKHDF/Points"].shape[0]
        return f["VTKHDF/Points"][:n].astype(np.float64)


def _read_vtu(path: Path):
    """Return (mesh_pos, err_magnitude) from the compare tool's per-particle VTU."""
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    ug = r.GetOutput()
    pos = vtk_to_numpy(ug.GetPoints().GetData()).astype(np.float64)
    err_arr = ug.GetPointData().GetArray("error_magnitude")
    if err_arr is None:
        return None, None
    err = vtk_to_numpy(err_arr).astype(np.float64)
    return pos, err


def make_figure(variant: str, step: int, vtu_dir: Path, fom_root: Path,
                out_dir: Path, metric: str = "pct"):
    """metric in {'pct', 'mm'}."""
    # Layout: 5 columns wide, 4 rows tall.  Each subplot spans
    # x=[-15, 80] mm (95 mm) and y=[-15, 15] mm (30 mm); equal aspect
    # gives each panel a ~3.2:1 aspect ratio, so total figure should be
    # ~(5 * 3.2) : (4 * 1) with some padding.
    fig, axs = plt.subplots(4, 5, figsize=(26, 10), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.12, hspace=0.4, top=0.93)
    axs = axs.flatten()

    if metric == "pct":
        norm = Normalize(vmin=0, vmax=100)
        cbar_label = "|err| as % of |mesh − seed|  (clipped to 100 %)"
        title_tail = "coloured by relative error (% of travelled distance)"
    else:
        # mm colour scale — pick a shared upper cap from 95th pct across all cases
        cap = 15.0
        norm = Normalize(vmin=0, vmax=cap)
        cbar_label = f"|err| (mm)  (capped at {cap:.0f} mm)"
        title_tail = "coloured by absolute error (mm)"

    scatter_handle = None
    for ci, case in enumerate(CASES):
        ax = axs[ci]
        vtu = vtu_dir / f"pt_error_case{case}_grid_{variant}_step{step}.vtu"
        if not vtu.exists():
            ax.text(0.5, 0.5, f"c{case}\nno VTU", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="grey")
            ax.set_xticks([]); ax.set_yticks([]); continue
        mesh_pos, err = _read_vtu(vtu)
        if mesh_pos is None:
            ax.text(0.5, 0.5, f"c{case}\nno err field", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="grey")
            continue

        ref_hdf = _find_mesh_ref(fom_root, case)
        if ref_hdf is None:
            ax.text(0.5, 0.5, f"c{case}\nno ref", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="grey")
            continue
        seed = _load_seed(ref_hdf)
        n = min(mesh_pos.shape[0], seed.shape[0], err.shape[0])
        mp = mesh_pos[:n]; sd = seed[:n]; er = err[:n]

        # Plot every mesh-tracked particle at the reporting step (some
        # sit past the nominal outlet at x = 30 mm because the tracker
        # applies ballistic extension after escape -- those positions
        # are still meaningful and their err vs the grid tracker is a
        # real signal).
        travel = np.linalg.norm(mp - sd, axis=1)   # mesh-side travel

        if metric == "pct":
            eps = 1e-3         # 1 mm floor to avoid /0 on stationary particles
            rel = np.where(travel > eps, 100 * er / travel, np.nan)
            rel = np.clip(rel, 0, 100)
            col = rel
        else:
            col = er * 1000  # mm

        # Random shuffle so hot colours don't hide behind cold pointspile-up
        rng = np.random.default_rng(seed=0)
        order = rng.permutation(len(mp))
        mp = mp[order]; col = col[order]

        sc = ax.scatter(mp[:, 0] * 1000, mp[:, 1] * 1000,
                         c=col, s=1.2, cmap="turbo", norm=norm,
                         linewidths=0, rasterized=True)
        scatter_handle = sc

        # Sub-title: case + % inside domain + median travel  (kept short
        # to avoid overlap between adjacent subplots)
        pct_inside = 100 * float((mp[:, 0] < OUTLET_X).mean())
        med_travel_mm = float(np.median(travel)) * 1000
        ax.set_title(f"c{case}  ·  {pct_inside:.0f}% inside  ·  travel {med_travel_mm:.0f} mm",
                     fontsize=10, pad=6, loc="left")

        # Pin marker at origin
        ax.plot(0, 0, marker="+", markersize=9, color="black", markeredgewidth=1.5)
        # Nominal outlet plane (particles past this line have been
        # ballistically extended by the tracker).
        ax.axvline(OUTLET_X * 1000, color="red", linestyle=":",
                   linewidth=1.0, alpha=0.6)
        # Shared axis range across all cases -- x extended to +80 mm
        # to fit the ballistic tail of the fastest cases (case 001/003
        # reach ~76 mm at step 2000).
        ax.set_xlim(-15, 80); ax.set_ylim(-15, 15)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25, linewidth=0.4)

    for i in range(len(CASES), len(axs)):
        axs[i].axis("off")

    for i in (15, 16, 17, 18, 19):
        if i < len(axs): axs[i].set_xlabel("x (mm)")
    for i in (0, 5, 10, 15):
        if i < len(axs): axs[i].set_ylabel("y (mm)")

    if scatter_handle is not None:
        cbar = fig.colorbar(scatter_handle, ax=axs,
                            shrink=0.85, pad=0.02, aspect=35)
        cbar.set_label(cbar_label, fontsize=11)

    fig.suptitle(f"Final-step (step {step}, t = 7.5 s) mesh particle positions, xy projection\n"
                 f"variant = {variant}  ·  {title_tail}",
                 y=0.995, fontsize=15)
    out = out_dir / f"xy_error_{variant}_step{step}_{metric}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vtu-dir",  type=Path,
                    default=Path("/scratch/shared/ROM/rom_out/fom_grid_pt_all20"))
    ap.add_argument("--fom-root", type=Path,
                    default=Path("/scratch/shared/ROM/FOM"))
    ap.add_argument("--variants", type=str, nargs="+",
                    default=["4lvl_hct", "malmo6_hct"])
    ap.add_argument("--step", type=int, default=2000)
    ap.add_argument("--out", type=Path,
                    default=Path("paper_figs/rom_pt_step5_step6_all20"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for v in args.variants:
        make_figure(v, args.step, args.vtu_dir, args.fom_root, args.out, "pct")
        make_figure(v, args.step, args.vtu_dir, args.fom_root, args.out, "mm")


if __name__ == "__main__":
    main()
