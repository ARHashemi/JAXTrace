"""
compare_grid_vs_mesh_pt_fom.py -- Compare grid PT to existing mesh PT
at step 950 for the FSW FOM cases (000, 001, 003, 004).

For each case:

  reference (mesh + HCT-3D):
    <case>.gid/post_pt/fom_hct_on/run_*/particles.vtkhdf

  grid PT variants:
    <case>.gid/out_grid_{uniform,2lvl,4lvl}/run_*/particles.vtkhdf

Because the grid PT was seeded from the mesh PT's step-0 slice
(see run_grid_all.sh), particle_id 0 in each file corresponds to the
same physical particle -- no ID matching required.

For each (case, grid_type) pair, reports at step 950:
  - n_particles
  - per-particle position error rms, mean, p95, max
  - per-particle error mag as a per-cell VTU (for ParaView)
  - CSV table

Also emits a summary bar chart:
  paper_figs/rom_pt_analytic/fom_grid_vs_mesh_pt_step950.png

Usage:
  python3 scripts/compare_grid_vs_mesh_pt_fom.py \\
      --cases 000 001 003 004 \\
      --step 950 \\
      --out rom_out/fom_grid_pt/

  # single case, single grid, verbose:
  python3 scripts/compare_grid_vs_mesh_pt_fom.py \\
      --cases 000 --grid-types 2lvl --step 950 --verbose
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import h5py


# FOM root: try workstation path first, then the local NFS mount used
# during development.  Override via --fom-root on the CLI.
_FOM_ROOT_CANDIDATES = [
    Path("/scratch/shared/ROM/FOM"),                                       # workstation
    Path("/home/arhashemi/fsw-gpu/scratch/shared/ROM/FOM"),                # local dev mount
]
FOM_ROOT_DEFAULT = next((p for p in _FOM_ROOT_CANDIDATES if p.exists()),
                        _FOM_ROOT_CANDIDATES[0])
CASES_DEFAULT = ["000", "001", "003", "004"]
GRIDS_DEFAULT = ["uniform", "2lvl", "4lvl", "malmo", "malmo3", "uniform_half",
                 "uniform_tricubic", "2lvl_tricubic",
                 "4lvl_tricubic", "malmo_tricubic", "malmo3_tricubic",
                 "uniform_half_tricubic",
                 "uniform_hct", "2lvl_hct", "4lvl_hct", "malmo_hct",
                 "malmo3_hct", "uniform_half_hct",
                 "uniform_hct_tricubic", "2lvl_hct_tricubic",
                 "4lvl_hct_tricubic", "malmo_hct_tricubic",
                 "malmo3_hct_tricubic", "uniform_half_hct_tricubic"]


def _load_step(vtkhdf_path: Path, target_step: float) -> tuple[np.ndarray, int, float]:
    """Return (positions (n, 3), n_particles, matched_step)."""
    with h5py.File(vtkhdf_path, "r") as f:
        offsets = f["VTKHDF/Steps/PointOffsets"][:]
        values = f["VTKHDF/Steps/Values"][:]
        idx = int(np.argmin(np.abs(values - target_step)))
        matched = float(values[idx])
        n = int(offsets[1] - offsets[0]) if len(offsets) > 1 \
            else f["VTKHDF/Points"].shape[0]
        start = int(offsets[idx])
        pos = f["VTKHDF/Points"][start:start + n].astype(np.float64)
    return pos, n, matched


def _load_step_zero(vtkhdf_path: Path) -> np.ndarray:
    """Return the (n, 3) position array at simulation step 0.

    Used to compute each particle's initial radial position for
    the near-pin vs outer-bulk r-binning (matches the convention
    in `rom_spatial_residual.py`).
    """
    with h5py.File(vtkhdf_path, "r") as f:
        offsets = f["VTKHDF/Steps/PointOffsets"][:]
        n = int(offsets[1] - offsets[0]) if len(offsets) > 1 \
            else f["VTKHDF/Points"].shape[0]
        pos = f["VTKHDF/Points"][:n].astype(np.float64)
    return pos


def _find_hdf(case_dir: Path, subpath: str) -> Path | None:
    """Find the first particles.vtkhdf under case_dir/subpath/run_*/."""
    for hdf in (case_dir / subpath).glob("run_*/particles.vtkhdf"):
        return hdf
    return None


def _find_mesh_reference(case_dir: Path,
                         preferred_variant: str = "fom_hct_on") -> Path | None:
    """Locate the mesh PT reference particles.vtkhdf for a case.

    Preference order:
      1. `post_pt/<preferred_variant>/run_*/particles.vtkhdf`
         (curated 4-case layout with fom_hct_on / fom_hct_off).
      2. `post_pt/run_grid-frac_n360000_s2000/particles.vtkhdf`
         (auto-generated 2000-step run from run_jaxtrace.sh, present
         on all 20 cases).
    Returns None if no suitable reference is found.
    """
    p = _find_hdf(case_dir, f"post_pt/{preferred_variant}")
    if p is not None:
        return p
    # Cohort default: 360k particles x 2000 steps
    p = case_dir / "post_pt" / "run_grid-frac_n360000_s2000" / "particles.vtkhdf"
    if p.exists():
        return p
    return None


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt((x ** 2).mean())) if x.size else float("nan")


def _detailed_stats(mesh_pos: np.ndarray, grid_pos: np.ndarray,
                    step_zero_pos: np.ndarray,
                    r_bin: float = 0.010,
                    outlet_x: float = 0.030) -> dict:
    """Rich per-variant statistics.

    All arrays are (n, 3).  Binning uses the **mesh position at the
    reporting step** as the truth radial position (particles are
    seeded upstream so step-0 radial position is a poor bin — most
    of the interesting physics lives in the flow *toward* the pin).
    `step_zero_pos` is kept in the signature for future use.
    `outlet_x` is the boundary at which particles are considered
    "escaped" (matches the `--boundary-walls x_max=outlet` setting).
    """
    n = min(mesh_pos.shape[0], grid_pos.shape[0], step_zero_pos.shape[0])
    mp = mesh_pos[:n]; gp = grid_pos[:n]

    err_vec = gp - mp                                # (n, 3)
    err = np.linalg.norm(err_vec, axis=1)            # (n,)

    # Bin by mesh position at reporting step — this is the "did the
    # particle end up near the pin" question that FSW mixing cares
    # about.  See docstring above for why step-0 r0 is inadequate.
    r_grid  = np.sqrt(gp[:, 0] ** 2 + gp[:, 1] ** 2)
    r_mesh  = np.sqrt(mp[:, 0] ** 2 + mp[:, 1] ** 2)
    near = r_mesh <= r_bin
    outer = ~near

    trapped_grid = r_grid <= r_bin
    trapped_mesh = near                              # mesh is truth
    escaped_grid = gp[:, 0] >= outlet_x
    escaped_mesh = mp[:, 0] >= outlet_x

    # "Both alive" mask -- both trackers agree the particle is still
    # inside the domain at the reporting step.  RMS restricted to this
    # subset is the "clean comparison" that isn't polluted by
    # boundary/outlet dynamics: we compare *trajectories inside the
    # domain* only.
    alive_grid = ~escaped_grid
    alive_mesh = ~escaped_mesh
    both_alive = alive_grid & alive_mesh
    err_alive       = err[both_alive]
    err_vec_alive   = err_vec[both_alive]
    r_mesh_alive    = r_mesh[both_alive]
    near_alive      = r_mesh_alive <= r_bin
    outer_alive     = ~near_alive

    return {
        # aggregate
        "n":        int(n),
        "rms":      _rms(err),
        "mean":     float(err.mean()),
        "p50":      float(np.percentile(err, 50)),
        "p95":      float(np.percentile(err, 95)),
        "p99":      float(np.percentile(err, 99)),
        "max":      float(err.max()),
        # per-component
        "rms_x":    _rms(err_vec[:, 0]),
        "rms_y":    _rms(err_vec[:, 1]),
        "rms_z":    _rms(err_vec[:, 2]),
        # per-r-bin (initial radial position)
        "n_near":   int(near.sum()),
        "n_outer":  int(outer.sum()),
        "rms_near": _rms(err[near]),
        "rms_outer": _rms(err[outer]),
        "mean_near": float(err[near].mean()) if near.any() else float("nan"),
        "mean_outer": float(err[outer].mean()) if outer.any() else float("nan"),
        "p95_near":  float(np.percentile(err[near], 95)) if near.any() else float("nan"),
        "p95_outer": float(np.percentile(err[outer], 95)) if outer.any() else float("nan"),
        # trapping (fraction of particles inside r <= r_bin at report step)
        "frac_trapped_grid":  float(trapped_grid.mean()),
        "frac_trapped_mesh":  float(trapped_mesh.mean()),
        "delta_trapped":      float(trapped_grid.mean() - trapped_mesh.mean()),
        # escape / alive agreement (mesh side is truth)
        "n_both_alive":         int(both_alive.sum()),
        "n_both_escaped":       int((escaped_grid & escaped_mesh).sum()),
        "n_only_grid_escaped":  int((escaped_grid & ~escaped_mesh).sum()),
        "n_only_mesh_escaped":  int((~escaped_grid & escaped_mesh).sum()),
        # alive-only metrics -- restrict to particles both trackers say
        # are still inside the domain at the reporting step.  This is
        # the clean cross-tracker comparison uncontaminated by outlet
        # dynamics; the "aggregate rms" above includes escapees.
        "rms_alive":       _rms(err_alive),
        "mean_alive":      float(err_alive.mean()) if err_alive.size else float("nan"),
        "p95_alive":       float(np.percentile(err_alive, 95)) if err_alive.size else float("nan"),
        "rms_x_alive":     _rms(err_vec_alive[:, 0]) if err_alive.size else float("nan"),
        "rms_y_alive":     _rms(err_vec_alive[:, 1]) if err_alive.size else float("nan"),
        "rms_z_alive":     _rms(err_vec_alive[:, 2]) if err_alive.size else float("nan"),
        "rms_near_alive":  _rms(err_alive[near_alive]) if near_alive.any() else float("nan"),
        "rms_outer_alive": _rms(err_alive[outer_alive]) if outer_alive.any() else float("nan"),
        "n_near_alive":    int(near_alive.sum()),
        "n_outer_alive":   int(outer_alive.sum()),
    }


def _write_error_vtu(out_path: Path, mesh_pos: np.ndarray,
                     grid_pos: np.ndarray, errs: np.ndarray) -> None:
    """Emit a POLYDATA VTU with mesh positions as points, per-particle
    displacement vector to grid position, and the |err| scalar."""
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk
    except ImportError:
        return
    n = mesh_pos.shape[0]
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(mesh_pos.astype(np.float64), deep=True))
    ug = vtk.vtkUnstructuredGrid()
    ug.SetPoints(pts)
    # vertex cell per point
    verts = np.empty(2 * n, dtype=np.int64)
    verts[0::2] = 1
    verts[1::2] = np.arange(n, dtype=np.int64)
    from vtk.util.numpy_support import numpy_to_vtkIdTypeArray
    id_arr = numpy_to_vtkIdTypeArray(verts, deep=True)
    cells = vtk.vtkCellArray()
    cells.SetCells(n, id_arr)
    ug.SetCells(vtk.VTK_VERTEX, cells)

    disp = (grid_pos - mesh_pos).astype(np.float32)
    d_arr = numpy_to_vtk(np.ascontiguousarray(disp), deep=True)
    d_arr.SetName("displacement_grid_minus_mesh")
    d_arr.SetNumberOfComponents(3)
    ug.GetPointData().AddArray(d_arr)
    e_arr = numpy_to_vtk(errs.astype(np.float32), deep=True)
    e_arr.SetName("error_magnitude")
    ug.GetPointData().AddArray(e_arr)

    w = vtk.vtkXMLUnstructuredGridWriter()
    w.SetFileName(str(out_path))
    w.SetInputData(ug)
    w.SetDataModeToBinary()
    w.SetCompressorTypeToZLib()
    w.Write()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=str, nargs="+", default=CASES_DEFAULT)
    ap.add_argument("--grid-types", type=str, nargs="+", default=GRIDS_DEFAULT)
    ap.add_argument("--step", type=float, default=950.0)
    ap.add_argument("--reference-variant", type=str, default="fom_hct_on",
                    help="Mesh variant under post_pt/ used as reference. "
                         "Grid PT was seeded from this variant's step-0.")
    ap.add_argument("--fom-root", type=Path, default=FOM_ROOT_DEFAULT,
                    help="Root under which cylindrical_<case>.gid live. "
                         "Default auto-picks the first path that exists "
                         "from '/scratch/shared/ROM/FOM' (workstation) "
                         "or '/home/arhashemi/fsw-gpu/scratch/shared/ROM/FOM' "
                         "(local mount).")
    ap.add_argument("--out", type=Path, default=Path("rom_out/fom_grid_pt"))
    ap.add_argument("--r-bin", type=float, default=0.010,
                    help="Near-pin/outer split at this initial radial "
                         "position (m). Matches roadmap §7 conventions.")
    ap.add_argument("--outlet-x", type=float, default=0.030,
                    help="X-boundary at which a particle is considered "
                         "escaped. Matches --boundary-walls x_max=outlet.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.verbose:
        print(f"  FOM_ROOT = {args.fom_root}")

    rows: list[dict] = []
    # (case, gt) -> {"r": (m,), "z": (m,), "err": (m,)} for both-alive
    # subset, used by the spatial 2D-heatmap figure.
    spatial_records: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for case in args.cases:
        case_dir = args.fom_root / f"cylindrical_{case}.gid"
        if not case_dir.exists():
            print(f"  [skip] case {case}: dir not found: {case_dir}",
                  file=sys.stderr)
            continue
        ref_hdf = _find_mesh_reference(case_dir, args.reference_variant)
        if ref_hdf is None:
            print(f"  [skip] case {case}: no mesh reference under "
                  f"post_pt/{args.reference_variant}/ nor "
                  f"post_pt/run_grid-frac_n360000_s2000/", file=sys.stderr)
            continue
        mesh_pos, n_mesh, matched_step = _load_step(ref_hdf, args.step)
        # Step-0 positions kept for reference / future diagnostics
        # (particles are seeded upstream at x ~ -0.02 m, so their
        # initial radial position has no near-pin population).  Binning
        # in _detailed_stats uses the mesh position at the reporting
        # step, which is where the FSW mixing question actually lives.
        step_zero = _load_step_zero(ref_hdf)
        r_mesh_now = np.sqrt(mesh_pos[:, 0] ** 2 + mesh_pos[:, 1] ** 2)
        n_near_now = int((r_mesh_now <= args.r_bin).sum())
        if args.verbose:
            print(f"case {case}  reference = {ref_hdf.name}, "
                  f"n={n_mesh:,}, step_match={matched_step}, "
                  f"n_near(r_mesh(step {int(matched_step)}) <= "
                  f"{args.r_bin*1000:.0f}mm)={n_near_now:,}")

        for gt in args.grid_types:
            grid_hdf = _find_hdf(case_dir, f"out_grid_{gt}")
            if grid_hdf is None:
                print(f"  [skip] case {case} grid_{gt}: no PT output",
                      file=sys.stderr)
                continue
            grid_pos, n_grid, _ = _load_step(grid_hdf, args.step)
            if n_grid != n_mesh:
                print(f"  [warn] case {case} grid_{gt}: n particles differ "
                      f"({n_grid} vs {n_mesh}); truncating to min",
                      file=sys.stderr)
            n = min(n_mesh, n_grid)
            s = _detailed_stats(mesh_pos[:n], grid_pos[:n], step_zero[:n],
                                r_bin=args.r_bin, outlet_x=args.outlet_x)
            row = {"case": case, "grid_type": gt,
                   "step_matched": matched_step, **s}
            rows.append(row)
            if args.verbose:
                print(f"  case {case} grid_{gt}: "
                      f"rms_alive={s['rms_alive']:.3e} "
                      f"[near {s['rms_near_alive']:.3e} / outer {s['rms_outer_alive']:.3e}] "
                      f"(n_alive={s['n_both_alive']:,} / {s['n']:,})")

            # Per-particle VTU + collect for spatial 2D histogram
            errs = np.linalg.norm(grid_pos[:n] - mesh_pos[:n], axis=1)
            vtu = args.out / f"pt_error_case{case}_grid_{gt}_step{int(matched_step)}.vtu"
            _write_error_vtu(vtu, mesh_pos[:n], grid_pos[:n], errs)

            # Cache mesh xyz + err for the "where does deviation come from"
            # 2D heatmap (r_mesh vs z_mesh).  Restrict to both-alive so we
            # only visualise particles that are still inside the domain.
            r_mesh = np.sqrt(mesh_pos[:n,0]**2 + mesh_pos[:n,1]**2)
            z_mesh = mesh_pos[:n,2]
            escaped_grid = grid_pos[:n,0] >= args.outlet_x
            escaped_mesh = mesh_pos[:n,0] >= args.outlet_x
            alive_mask = ~escaped_grid & ~escaped_mesh
            spatial_records[(case, gt)] = {
                "r": r_mesh[alive_mask],
                "z": z_mesh[alive_mask],
                "err": errs[alive_mask],
            }

    # CSV — legacy summary (kept for backwards compat with downstream
    # tools) + rich detail table.
    summary_fields = ["case", "grid_type", "step_matched",
                      "n", "rms", "mean", "p95", "max"]
    summary_path = args.out / f"grid_vs_mesh_pt_summary_step{int(args.step)}.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {summary_path}  ({len(rows)} rows)")

    detail_fields = ["case", "grid_type", "step_matched",
                     "n", "rms", "mean", "p50", "p95", "p99", "max",
                     "rms_x", "rms_y", "rms_z",
                     "n_near", "n_outer",
                     "rms_near", "rms_outer",
                     "mean_near", "mean_outer",
                     "p95_near", "p95_outer",
                     "frac_trapped_grid", "frac_trapped_mesh",
                     "delta_trapped",
                     "n_both_alive", "n_both_escaped",
                     "n_only_grid_escaped", "n_only_mesh_escaped",
                     "rms_alive", "mean_alive", "p95_alive",
                     "rms_x_alive", "rms_y_alive", "rms_z_alive",
                     "rms_near_alive", "rms_outer_alive",
                     "n_near_alive", "n_outer_alive"]
    detail_path = args.out / f"grid_vs_mesh_pt_detail_step{int(args.step)}.csv"
    with detail_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {detail_path}  ({len(rows)} rows)")

    # ------------------------------------------------------------------
    # Figures.  All rms-based figures use the both-alive subset so that
    # cross-tracker outlet dynamics do not pollute the metric.  The
    # `rms` column in the CSV keeps the full-particle rms for
    # backwards compat.
    # ------------------------------------------------------------------
    if rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cases = sorted({r["case"] for r in rows})
        family_colour = {
            "uniform":      "#c14040",
            "2lvl":         "#f0a020",
            "4lvl":         "#3f8f3f",
            "malmo":        "#3060c0",
            "malmo3":       "#8020a0",
            "uniform_half": "#606060",
        }

        def _decode(gt):
            """Split a variant name into (family, is_hct, is_tricubic)."""
            g = gt
            is_tri = g.endswith("_tricubic")
            if is_tri:
                g = g[:-len("_tricubic")]
            is_hct = g.endswith("_hct")
            if is_hct:
                g = g[:-len("_hct")]
            return g, is_hct, is_tri

        # Split variants into two panels: raw-P1 stage 1 vs HCT stage 1
        p1_grids  = [gt for gt in args.grid_types if not _decode(gt)[1]]
        hct_grids = [gt for gt in args.grid_types if _decode(gt)[1]]

        # Compute a shared y-range so P1 vs HCT are directly comparable
        # by bar height (log-scale y-axis defeats "same colours = same
        # story" illusion).  Metric: rms restricted to both-alive
        # particles (see docstring in _detailed_stats).
        all_vals = [r["rms_alive"] for r in rows
                    if np.isfinite(r.get("rms_alive", np.nan)) and r["rms_alive"] > 0]
        y_lo = 10 ** np.floor(np.log10(min(all_vals) * 0.9))
        y_hi = 10 ** np.ceil(np.log10(max(all_vals) * 1.15))

        def _plot(panel_grids, tag, title):
            if not panel_grids:
                return
            fig_path = args.out / f"fom_grid_vs_mesh_pt_step{int(args.step)}_{tag}.png"
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(cases) * len(panel_grids) / 4), 5.0))
            x = np.arange(len(cases))
            width = 0.9 / max(1, len(panel_grids))
            for gi, gt in enumerate(panel_grids):
                fam, _, is_tri = _decode(gt)
                hatch = "///" if is_tri else ""
                vals = []
                for c in cases:
                    match = [r for r in rows if r["case"] == c and r["grid_type"] == gt]
                    vals.append(match[0]["rms_alive"] if match else np.nan)
                offset = (gi - (len(panel_grids) - 1) / 2) * width
                colour = family_colour.get(fam, "grey")
                ax.bar(x + offset, vals, width, color=colour,
                       edgecolor="black", linewidth=0.4, hatch=hatch,
                       label=gt.replace("_", " "))
                for xi, v in zip(x + offset, vals):
                    if np.isfinite(v):
                        ax.text(xi, v * 1.05, f"{v:.1e}",
                                ha="center", va="bottom", fontsize=7, rotation=90)
            ax.set_xticks(x)
            ax.set_xticklabels([f"case {c}" for c in cases])
            ax.set_ylabel(f"rms position error [both-alive] at step {int(args.step)} (m), log")
            ax.set_yscale("log")
            ax.set_ylim(y_lo, y_hi)
            ax.set_title(title)
            ax.grid(axis="y", which="both", alpha=0.3)
            ax.legend(loc="upper right", fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(fig_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {fig_path}")

        _plot(p1_grids,  "p1",
              f"Grid PT · Stage-1 = RAW P1 · both-alive rms at step {int(args.step)} · ref = mesh")
        _plot(hct_grids, "hct",
              f"Grid PT · Stage-1 = HCT-3D · both-alive rms at step {int(args.step)} · ref = mesh")

        # Also emit a delta panel: (P1 rms) - (HCT rms) per family x case
        # to make the size of the HCT gain unmissable.
        delta_fig_path = args.out / f"fom_grid_vs_mesh_pt_step{int(args.step)}_p1_minus_hct.png"
        fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(cases) * len(p1_grids) / 4), 5.0))
        x = np.arange(len(cases))
        width = 0.9 / max(1, len(p1_grids))
        for gi, gt in enumerate(p1_grids):
            fam, _, is_tri = _decode(gt)
            hatch = "///" if is_tri else ""
            hct_gt = (fam + "_hct" + ("_tricubic" if is_tri else ""))
            deltas = []
            for c in cases:
                p1_row = [r for r in rows if r["case"] == c and r["grid_type"] == gt]
                hct_row = [r for r in rows if r["case"] == c and r["grid_type"] == hct_gt]
                if p1_row and hct_row:
                    deltas.append(p1_row[0]["rms_alive"] - hct_row[0]["rms_alive"])
                else:
                    deltas.append(np.nan)
            offset = (gi - (len(p1_grids) - 1) / 2) * width
            colour = family_colour.get(fam, "grey")
            ax.bar(x + offset, deltas, width, color=colour,
                   edgecolor="black", linewidth=0.4, hatch=hatch,
                   label=gt.replace("_", " "))
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"case {c}" for c in cases])
        ax.set_ylabel("rms(P1) - rms(HCT) at step 950 (m); >0 means HCT is better")
        ax.set_title("Stage-1 upgrade P1 -> HCT: rms reduction per (case x family x Stage 2 interp)")
        ax.grid(axis="y", which="both", alpha=0.3)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(delta_fig_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {delta_fig_path}")

        # ------------------------------------------------------------------
        # Per-r-bin RMS panels (near-pin r <= r_bin  vs  outer r > r_bin)
        # ------------------------------------------------------------------
        # Same shared y-range so near vs outer are directly comparable.
        near_vals = [r["rms_near_alive"] for r in rows
                     if np.isfinite(r.get("rms_near_alive", np.nan)) and r["rms_near_alive"] > 0]
        outer_vals = [r["rms_outer_alive"] for r in rows
                      if np.isfinite(r.get("rms_outer_alive", np.nan)) and r["rms_outer_alive"] > 0]
        rbin_all = near_vals + outer_vals
        r_lo = 10 ** np.floor(np.log10(min(rbin_all) * 0.9)) if rbin_all else 1e-4
        r_hi = 10 ** np.ceil(np.log10(max(rbin_all) * 1.15)) if rbin_all else 1e-1

        def _plot_rbin(panel_grids, tag, stage_label):
            if not panel_grids:
                return
            fig_path = args.out / f"fom_grid_vs_mesh_pt_step{int(args.step)}_{tag}_rbin.png"
            fig, axs = plt.subplots(2, 1, figsize=(max(10, 1.6 * len(cases) * len(panel_grids) / 4), 9.0),
                                    sharex=True, sharey=True)
            for ax, key, region in ((axs[0], "rms_near_alive", f"near-pin, both-alive (r_mesh(step={int(args.step)}) <= {args.r_bin*1000:.0f} mm)"),
                                    (axs[1], "rms_outer_alive", f"outer, both-alive (r_mesh(step={int(args.step)}) > {args.r_bin*1000:.0f} mm)")):
                x = np.arange(len(cases)); width = 0.9 / max(1, len(panel_grids))
                for gi, gt in enumerate(panel_grids):
                    fam, _, is_tri = _decode(gt)
                    hatch = "///" if is_tri else ""
                    vals = []
                    for c in cases:
                        m = [r for r in rows if r["case"] == c and r["grid_type"] == gt]
                        vals.append(m[0][key] if m else np.nan)
                    offset = (gi - (len(panel_grids) - 1) / 2) * width
                    ax.bar(x + offset, vals, width,
                           color=family_colour.get(fam, "grey"),
                           edgecolor="black", linewidth=0.4, hatch=hatch,
                           label=gt.replace("_", " "))
                    for xi, v in zip(x + offset, vals):
                        if np.isfinite(v):
                            ax.text(xi, v * 1.05, f"{v:.1e}",
                                    ha="center", va="bottom", fontsize=6.5, rotation=90)
                ax.set_xticks(x); ax.set_xticklabels([f"case {c}" for c in cases])
                ax.set_ylabel(f"rms position error, {region} (m), log")
                ax.set_yscale("log"); ax.set_ylim(r_lo, r_hi)
                ax.grid(axis="y", which="both", alpha=0.3)
            axs[0].legend(loc="upper right", fontsize=6.5, ncol=2)
            axs[0].set_title(f"Grid PT · Stage-1 = {stage_label} · per-r-bin rms at step {int(args.step)}")
            fig.tight_layout()
            fig.savefig(fig_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {fig_path}")

        _plot_rbin(p1_grids,  "p1",  "RAW P1")
        _plot_rbin(hct_grids, "hct", "HCT-3D")

        # ------------------------------------------------------------------
        # Trapping delta bars — |Δfrac_trapped| per (case, variant).
        # A negative bar means the grid loses more particles from the
        # r<=r_bin ring than the mesh does; positive means grid retains
        # more.  Small |Δ| ≡ grid matches mesh trapping behaviour.
        # ------------------------------------------------------------------
        def _plot_trapping(panel_grids, tag, stage_label):
            if not panel_grids:
                return
            fig_path = args.out / f"fom_grid_vs_mesh_pt_step{int(args.step)}_{tag}_trapping.png"
            fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(cases) * len(panel_grids) / 4), 5.0))
            x = np.arange(len(cases)); width = 0.9 / max(1, len(panel_grids))
            for gi, gt in enumerate(panel_grids):
                fam, _, is_tri = _decode(gt)
                hatch = "///" if is_tri else ""
                vals = []
                for c in cases:
                    m = [r for r in rows if r["case"] == c and r["grid_type"] == gt]
                    vals.append(m[0]["delta_trapped"] * 100 if m else np.nan)
                offset = (gi - (len(panel_grids) - 1) / 2) * width
                ax.bar(x + offset, vals, width,
                       color=family_colour.get(fam, "grey"),
                       edgecolor="black", linewidth=0.4, hatch=hatch,
                       label=gt.replace("_", " "))
            ax.axhline(0.0, color="black", linewidth=0.6)
            ax.set_xticks(x); ax.set_xticklabels([f"case {c}" for c in cases])
            ax.set_ylabel(f"trapped-fraction difference (grid - mesh, "
                          f"r<={args.r_bin*1000:.0f}mm at step {int(args.step)}) [percentage points]")
            ax.set_title(f"Grid PT · Stage-1 = {stage_label} · trapped-fraction agreement")
            ax.grid(axis="y", which="both", alpha=0.3)
            ax.legend(loc="best", fontsize=6.5, ncol=2)
            fig.tight_layout()
            fig.savefig(fig_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {fig_path}")

        _plot_trapping(p1_grids,  "p1",  "RAW P1")
        _plot_trapping(hct_grids, "hct", "HCT-3D")

        # ------------------------------------------------------------------
        # Spatial 2D heatmap of per-particle |err| in the (r_mesh, z_mesh)
        # plane, restricted to both-alive particles.  Answers *where in
        # the domain does the deviation come from* directly: rows are
        # variants, columns are cases; each cell is a pcolormesh of the
        # median |err| in each (r, z) bin.  Median (not mean) so a few
        # long-tail particles don't dominate the colour scale.
        # ------------------------------------------------------------------
        def _plot_spatial(panel_grids, tag, stage_label):
            if not panel_grids:
                return
            fig_path = args.out / f"fom_grid_vs_mesh_pt_step{int(args.step)}_{tag}_spatial.png"
            n_case = len(cases)
            n_var = len(panel_grids)
            if n_case == 0 or n_var == 0:
                return
            fig, axs = plt.subplots(n_var, n_case,
                                    figsize=(max(3.2 * n_case, 12),
                                             max(2.6 * n_var, 4)),
                                    sharex=True, sharey=True, squeeze=False)
            # Bin edges: r in [0, 0.030], z in [-0.0045, 0]; 30 r bins x 15 z bins
            r_edges = np.linspace(0.0,     0.030,  31)
            z_edges = np.linspace(-0.0045, 0.0,    16)
            # Shared colour scale across the whole panel-grid so cells
            # are comparable across cases + variants.
            all_err = np.concatenate([
                spatial_records[(c, gt)]["err"]
                for c in cases for gt in panel_grids
                if (c, gt) in spatial_records and spatial_records[(c, gt)]["err"].size
            ]) if any((c, gt) in spatial_records for c in cases for gt in panel_grids) else np.array([])
            if all_err.size == 0:
                plt.close(fig); return
            vmax = np.percentile(all_err, 95)
            vmin = np.percentile(all_err[all_err > 0], 5) if (all_err > 0).any() else 1e-4
            from matplotlib.colors import LogNorm
            norm = LogNorm(vmin=max(vmin, 1e-5), vmax=vmax)

            mesh_im = None
            for vi, gt in enumerate(panel_grids):
                for ci, c in enumerate(cases):
                    ax = axs[vi][ci]
                    rec = spatial_records.get((c, gt))
                    if rec is None or rec["err"].size == 0:
                        ax.set_facecolor("#f0f0f0")
                        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                                transform=ax.transAxes, fontsize=8, color="grey")
                    else:
                        # 2D median error per bin
                        from scipy.stats import binned_statistic_2d
                        stat, _, _, _ = binned_statistic_2d(
                            rec["r"], rec["z"], rec["err"],
                            statistic="median", bins=[r_edges, z_edges])
                        R, Z = np.meshgrid(r_edges, z_edges, indexing="ij")
                        mesh_im = ax.pcolormesh(R * 1000, Z * 1000,
                                                np.where(np.isnan(stat), 0, stat),
                                                cmap="viridis", norm=norm, shading="auto")
                        ax.axvline(args.r_bin * 1000, color="red", ls="--", lw=0.5, alpha=0.7)
                    if vi == 0:
                        ax.set_title(f"case {c}", fontsize=9)
                    if ci == 0:
                        ax.set_ylabel(f"{gt}\nz (mm)", fontsize=7)
                    if vi == n_var - 1:
                        ax.set_xlabel("r_mesh at step (mm)", fontsize=7)
                    ax.tick_params(labelsize=6)
            if mesh_im is not None:
                cbar = fig.colorbar(mesh_im, ax=axs.ravel().tolist(),
                                    shrink=0.9, pad=0.02)
                cbar.set_label("median |err| per (r, z) bin (m), log", fontsize=8)
            fig.suptitle(f"Grid PT · Stage-1 = {stage_label} · spatial breakdown of |err| "
                         f"(both-alive) at step {int(args.step)}",
                         fontsize=10)
            fig.savefig(fig_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {fig_path}")

        _plot_spatial(p1_grids,  "p1",  "RAW P1")
        _plot_spatial(hct_grids, "hct", "HCT-3D")

        # ------------------------------------------------------------------
        # Per-component RMS bars — rms_x, rms_y, rms_z per (case, variant)
        # Reveals directional bias in the grid representation.
        # ------------------------------------------------------------------
        def _plot_percomp(panel_grids, tag, stage_label):
            if not panel_grids:
                return
            fig_path = args.out / f"fom_grid_vs_mesh_pt_step{int(args.step)}_{tag}_percomp.png"
            n_case = len(cases)
            fig, axs = plt.subplots(1, n_case,
                                    figsize=(max(4 * n_case, 12), 4.5),
                                    sharey=True)
            if n_case == 1: axs = [axs]
            comp_colours = {"rms_x": "#c14040", "rms_y": "#3f8f3f", "rms_z": "#3060c0"}
            for ci, c in enumerate(cases):
                ax = axs[ci]
                variants = []
                xs = []; ys = []; zs = []
                for gt in panel_grids:
                    m = [r for r in rows if r["case"] == c and r["grid_type"] == gt]
                    if not m: continue
                    r = m[0]
                    variants.append(gt.replace("_", " "))
                    xs.append(r["rms_x_alive"]); ys.append(r["rms_y_alive"]); zs.append(r["rms_z_alive"])
                x = np.arange(len(variants)); w = 0.28
                ax.bar(x - w, xs, w, color=comp_colours["rms_x"],
                       edgecolor="black", linewidth=0.3, label="rms_x")
                ax.bar(x,     ys, w, color=comp_colours["rms_y"],
                       edgecolor="black", linewidth=0.3, label="rms_y")
                ax.bar(x + w, zs, w, color=comp_colours["rms_z"],
                       edgecolor="black", linewidth=0.3, label="rms_z")
                ax.set_xticks(x)
                ax.set_xticklabels(variants, rotation=90, fontsize=6)
                ax.set_title(f"case {c}", fontsize=9)
                ax.set_yscale("log")
                ax.grid(axis="y", which="both", alpha=0.3)
                if ci == 0:
                    ax.set_ylabel("per-component rms (m), log")
                    ax.legend(loc="lower left", fontsize=6.5)
            fig.suptitle(f"Grid PT · Stage-1 = {stage_label} · per-component rms at step {int(args.step)}",
                         fontsize=10)
            fig.tight_layout()
            fig.savefig(fig_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {fig_path}")

        _plot_percomp(p1_grids,  "p1",  "RAW P1")
        _plot_percomp(hct_grids, "hct", "HCT-3D")

    return 0


if __name__ == "__main__":
    sys.exit(main())
