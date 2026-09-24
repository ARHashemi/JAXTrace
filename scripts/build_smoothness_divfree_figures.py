"""
build_smoothness_divfree_figures.py -- Eulerian smoothness and
divergence-free diagnostics for the ROM PT roadmap §4.

Reads:
  * rom_out/velocity_diagnostics/radial_samples_cylindrical_<case>.csv
  * rom_out/velocity_diagnostics/ring_samples_cylindrical_<case>.csv
  * rom_out/velocity_diagnostics/divergence_summary_cylindrical_<case>.csv
  * rom_out/velocity_diagnostics/divergence_<case>.vtu    (per-cell |∇·v|)
  * ROM_recon_centered/<case>.gid/post_pt_mixing/hct_{on,off}/
       {residence_time.csv, pairwise_separation.csv}

Writes to <out-dir>/:
  figA_radial_velocity_profiles.{png,svg}
  figB_azimuthal_velocity_profiles.{png,svg}
  figC_azimuthal_fft_spectra.{png,svg}
  figD_divergence_ccdf.{png,svg}
  figE_divergence_vs_pt.{png,svg}
  figF_mixing_residence_time.{png,svg}
  figG_mixing_pairwise_separation.{png,svg}
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy


CASES = ["000", "001", "003", "004"]

CASE_COLOURS = {
    "000": "#d62728",
    "001": "#1f77b4",
    "003": "#2ca02c",
    "004": "#ff7f0e",
}

# From docs/rom_reconstruction_findings.md (centered formula, per-case).
EULERIAN_REL_RMS_PCT = {
    "000": 6.40, "001": 3.75, "003": 2.62, "004": 3.61,
}


def _load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _to_float(rows: list[dict], keys: tuple[str, ...]) -> None:
    for r in rows:
        for k in keys:
            if k in r and r[k] not in ("", None):
                try:
                    r[k] = float(r[k])
                except ValueError:
                    pass


# ---------------------------------------------------------------------------
# Fig A — radial |v| profiles
# ---------------------------------------------------------------------------

def figA_radial_profiles(diag_dir: Path, out_dir: Path) -> None:
    """Radial |v| per case: 4 rows (cases) × 3 cols (FOM, ROM, residual).

    Uses probe-interpolated line samples (P1 barycentric — same
    interpolation the tracker uses).  Shaded band marks the
    inside-pin region (LEVEL < 0) so it is clear when the samples
    are inside the tool body vs the workpiece.
    """
    fig, axes = plt.subplots(len(CASES), 3, figsize=(15, 3.2 * len(CASES)),
                             sharex=True)
    line_colours = {"east":  "#1f77b4", "north": "#2ca02c",
                    "west":  "#ff7f0e", "south": "#d62728"}
    for row, case in enumerate(CASES):
        rows = _load_csv(diag_dir / f"radial_samples_cylindrical_{case}.csv")
        _to_float(rows, ("s", "v_fom_mag", "v_rom_mag", "residual_mag",
                          "level"))
        # inside-pin band: derived from east-line samples where level<0
        east = [r for r in rows if r["line"] == "east"]
        east_s = np.array([r["s"] for r in east])
        east_lvl = np.array([r["level"] for r in east])
        inside_mask = east_lvl < 0
        if inside_mask.any():
            # extent of the inside-pin region along the east radial
            r_pin_out = float(east_s[inside_mask].max()) * 1e3
        else:
            r_pin_out = None

        for line in ("east", "north", "west", "south"):
            sub = [r for r in rows if r["line"] == line]
            s = np.array([r["s"] for r in sub])
            axes[row, 0].plot(s * 1e3, [r["v_fom_mag"] for r in sub],
                              color=line_colours[line], linewidth=1.3,
                              label=line if row == 0 else None)
            axes[row, 1].plot(s * 1e3, [r["v_rom_mag"] for r in sub],
                              color=line_colours[line], linewidth=1.3)
            axes[row, 2].plot(s * 1e3, [r["residual_mag"] for r in sub],
                              color=line_colours[line], linewidth=1.3)
        for col, title in enumerate(("FOM |v|", "ROM |v|", "|v_FOM − v_ROM|")):
            if r_pin_out is not None:
                axes[row, col].axvspan(0, r_pin_out, color="grey", alpha=0.13)
            axes[row, col].axvline(10, color="grey", linestyle=":",
                                   linewidth=0.7)
            axes[row, col].grid(True, alpha=0.3)
            if row == 0:
                axes[row, col].set_title(title)
        axes[row, 0].set_ylabel(f"case {case}\n|v| (m/s)")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    for col in range(3):
        axes[-1, col].set_xlabel("r (mm)")
    fig.suptitle("Fig A · Radial |v| profiles at ts = 119 "
                 "(4 radial lines from origin, grey band = inside pin body, "
                 "dotted = r = 10 mm)",
                 y=1.001)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"figA_radial_velocity_profiles.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig B — azimuthal |v| profiles
# ---------------------------------------------------------------------------

def figB_azimuthal_profiles(diag_dir: Path, out_dir: Path) -> None:
    """Azimuthal |v| per case × per ring.  Column count = number of
    rings in the CSV.  FOM (blue solid) + ROM (red dashed) overlaid.
    Ring title notes whether that radius sits inside or outside the
    pin body (from the LEVEL scalar carried by the sample rows).
    """
    fig = None
    for row, case in enumerate(CASES):
        rows = _load_csv(diag_dir / f"ring_samples_cylindrical_{case}.csv")
        _to_float(rows, ("r", "theta", "v_fom_mag", "v_rom_mag",
                         "residual_mag", "level"))
        radii = sorted({round(r["r"], 6) for r in rows})
        if fig is None:
            n_rings = len(radii)
            fig, axes = plt.subplots(len(CASES), n_rings,
                                     figsize=(4.5 * n_rings,
                                              3.2 * len(CASES)),
                                     sharex=True)
        for col, r_val in enumerate(radii):
            sub = [r for r in rows if abs(r["r"] - r_val) < 1e-8]
            sub.sort(key=lambda r: r["theta"])
            theta_deg = np.rad2deg([r["theta"] for r in sub])
            inside_frac = float(
                np.mean([r["level"] < 0 for r in sub]))
            axes[row, col].plot(theta_deg, [r["v_fom_mag"] for r in sub],
                                color="#1f77b4", linewidth=1.5,
                                label="FOM" if row == 0 else None)
            axes[row, col].plot(theta_deg, [r["v_rom_mag"] for r in sub],
                                color="#d62728", linewidth=1.5,
                                linestyle="--",
                                label="ROM" if row == 0 else None)
            axes[row, col].grid(True, alpha=0.3)
            if row == 0:
                inside_note = ("inside pin" if inside_frac > 0.99
                               else ("outside pin" if inside_frac < 0.01
                                     else f"{100*inside_frac:.0f}% inside pin"))
                axes[row, col].set_title(
                    f"r = {r_val*1e3:.1f} mm ({inside_note})")
            axes[row, col].set_xlim(0, 360)
        axes[row, 0].set_ylabel(f"case {case}\n|v| (m/s)")
    axes[0, 0].legend(loc="upper right", fontsize=9)
    for col in range(axes.shape[1]):
        axes[-1, col].set_xlabel("θ (deg)")
    fig.suptitle("Fig B · Azimuthal |v| profiles at ts = 119 "
                 "(probe-interpolated: same P1 barycentric interpolation "
                 "the tracker uses)", y=1.001)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"figB_azimuthal_velocity_profiles.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig C — azimuthal FFT spectra
# ---------------------------------------------------------------------------

def figC_azimuthal_fft(diag_dir: Path, out_dir: Path) -> None:
    """FFT of azimuthal |v| for FOM, ROM, residual, per case × per ring.

    Because the samples are uniformly spaced in θ and the field is
    2π-periodic, the FFT gives the true azimuthal spectrum with no
    windowing bias.  Modes k = 1..N/2 (DC omitted since we're
    comparing structure, not mean level).
    """
    fig = None
    for row, case in enumerate(CASES):
        rows = _load_csv(diag_dir / f"ring_samples_cylindrical_{case}.csv")
        _to_float(rows, ("r", "theta", "v_fom_mag", "v_rom_mag",
                         "residual_mag", "level"))
        radii = sorted({round(r["r"], 6) for r in rows})
        if fig is None:
            n_rings = len(radii)
            fig, axes = plt.subplots(len(CASES), n_rings,
                                     figsize=(4.5 * n_rings,
                                              3.2 * len(CASES)),
                                     sharex=True, sharey=True)
        for col, r_val in enumerate(radii):
            sub = [r for r in rows if abs(r["r"] - r_val) < 1e-8]
            sub.sort(key=lambda r: r["theta"])
            inside_frac = float(np.mean([r["level"] < 0 for r in sub]))
            fom_prof = np.array([r["v_fom_mag"] for r in sub])
            rom_prof = np.array([r["v_rom_mag"] for r in sub])
            res_prof = np.array([r["residual_mag"] for r in sub])
            n = fom_prof.size
            k = np.arange(1, n // 2 + 1)
            fom_fft = np.abs(np.fft.rfft(fom_prof))[1:len(k) + 1]
            rom_fft = np.abs(np.fft.rfft(rom_prof))[1:len(k) + 1]
            res_fft = np.abs(np.fft.rfft(res_prof))[1:len(k) + 1]
            axes[row, col].loglog(k, fom_fft, color="#1f77b4", linewidth=1.5,
                                  label="FOM" if row == 0 else None)
            axes[row, col].loglog(k, rom_fft, color="#d62728", linewidth=1.5,
                                  linestyle="--",
                                  label="ROM" if row == 0 else None)
            axes[row, col].loglog(k, res_fft, color="#2ca02c", linewidth=1.5,
                                  linestyle=":",
                                  label="residual" if row == 0 else None)
            axes[row, col].grid(True, alpha=0.3, which="both")
            if row == 0:
                inside_note = ("inside pin" if inside_frac > 0.99
                               else ("outside pin" if inside_frac < 0.01
                                     else f"{100*inside_frac:.0f}% inside pin"))
                axes[row, col].set_title(
                    f"r = {r_val*1e3:.1f} mm ({inside_note})")
        axes[row, 0].set_ylabel(f"case {case}\n|FFT(|v|)|")
    axes[0, 0].legend(loc="lower left", fontsize=9)
    for col in range(axes.shape[1]):
        axes[-1, col].set_xlabel("wavenumber k")
    fig.suptitle("Fig C · Azimuthal FFT spectra (probe-interpolated)\n"
                 "residual above FOM/ROM at any k → ROM misses structure "
                 "at that scale", y=1.001)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"figC_azimuthal_fft_spectra.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig D — divergence complementary CDF
# ---------------------------------------------------------------------------

def _load_divergence_cells(diag_dir: Path, case: str):
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(diag_dir / f"divergence_cylindrical_{case}.vtu"))
    r.Update()
    ug = r.GetOutput()

    # cell centroid r_xy for near-pin selection
    n_cells = ug.GetNumberOfCells()
    centroids = np.empty((n_cells, 3))
    cell = vtk.vtkGenericCell()
    for i in range(n_cells):
        ug.GetCell(i, cell)
        pts = cell.GetPoints()
        c = np.zeros(3)
        for j in range(4):
            p = np.zeros(3)
            pts.GetPoint(j, p)
            c += p
        centroids[i] = c / 4.0
    r_cell = np.sqrt(centroids[:, 0] ** 2 + centroids[:, 1] ** 2)
    div_fom = vtk_to_numpy(ug.GetCellData().GetArray("div_fom"))
    div_rom = vtk_to_numpy(ug.GetCellData().GetArray("div_rom"))
    return r_cell, div_fom, div_rom


def figD_divergence_ccdf(diag_dir: Path, out_dir: Path) -> None:
    """Near-pin (r<=10mm) |∇·v| complementary-CDF per case; FOM vs ROM."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    for col, case in enumerate(CASES):
        r_cell, div_fom, div_rom = _load_divergence_cells(diag_dir, case)
        mask = r_cell <= 0.010
        for label, arr, colour, ls in (
            ("FOM", np.abs(div_fom[mask]), "#1f77b4", "-"),
            ("ROM", np.abs(div_rom[mask]), "#d62728", "--"),
        ):
            arr = np.sort(arr)
            ccdf = 1.0 - np.arange(1, arr.size + 1) / arr.size
            axes[col].loglog(arr, ccdf, color=colour, linestyle=ls,
                             linewidth=1.6, label=label)
        axes[col].set_title(f"case {case}   Eulerian rel_rms "
                            f"{EULERIAN_REL_RMS_PCT[case]:.2f}%")
        axes[col].set_xlabel("|∇·v| (1/s)")
        axes[col].grid(True, which="both", alpha=0.3)
        if col == 0:
            axes[col].set_ylabel("P(|∇·v| > x)")
            axes[col].legend(loc="lower left", fontsize=10)
    fig.suptitle("Fig D · Near-pin (r ≤ 10 mm) |∇·v| complementary CDF "
                 "\nlog-log: FOM vs ROM per case at ts = 119. Truly div-free "
                 "would be a spike at x = 0.", y=1.02)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"figD_divergence_ccdf.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig E — divergence rms vs PT rel_rms
# ---------------------------------------------------------------------------

def figE_divergence_vs_pt(diag_dir: Path, out_dir: Path,
                          pt_rel_rms_step950: dict[str, float]) -> None:
    """Scatter: near-pin |∇·v|_rms of FOM vs Lagrangian PT rel_rms at step 950."""
    fig, ax = plt.subplots(figsize=(7, 6))
    xs, ys = [], []
    for case in CASES:
        rows = _load_csv(diag_dir / f"divergence_summary_cylindrical_{case}.csv")
        _to_float(rows, ("abs_div_rms",))
        # near-pin FOM
        for r in rows:
            if r["field"] == "fom" and float(r["r_lo"]) == 0.0:
                x = float(r["abs_div_rms"])
                break
        y = pt_rel_rms_step950.get(case)
        if y is None:
            continue
        xs.append(x); ys.append(y)
        ax.scatter(x, y, s=180, color=CASE_COLOURS[case], edgecolor="black",
                   linewidth=0.7, zorder=3)
        ax.annotate(f"case {case}", (x, y), xytext=(8, 6),
                    textcoords="offset points", fontsize=11, fontweight="bold",
                    color=CASE_COLOURS[case])
    ax.set_xlabel("FOM near-pin |∇·v| rms (1/s) at ts = 119")
    ax.set_ylabel("ROM–FOM PT rel_rms at step 950 (%)")
    ax.set_title("Fig E · Eulerian divergence vs Lagrangian PT gap\n"
                 "does the least divergence-free case give the worst PT?")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"figE_divergence_vs_pt.{ext}", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig F / G — mixing diagnostics from the overnight sweep
# ---------------------------------------------------------------------------

def _load_mixing_csvs(rom_recon_root: Path, case: str, hct: str,
                      filename: str) -> list[dict]:
    p = (rom_recon_root / f"cylindrical_{case}.gid" / "post_pt_mixing"
         / hct / filename)
    if not p.exists():
        return []
    return _load_csv(p)


def figF_residence_time(rom_recon_root: Path, out_dir: Path) -> None:
    """Residence-time curves per case × HCT setting.

    Two rows per case: linear-scale (top) shows the peak, log-scale
    (bottom) shows the tail — the number of particles that remain
    inside the annular probe at late steps.  The tail is buried at
    plot resolution on a linear axis but carries the trapping story
    (e.g. FOM case 000 retains 14 k particles at step 2000 while ROM
    retains only 55).  Only HCT-on is shown to keep the figure
    readable; HCT-off is essentially identical (see Fig 4 in the §2/§3
    report).
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
    for col, case in enumerate(CASES):
        rows = _load_mixing_csvs(rom_recon_root, case, "hct_on",
                                 "residence_time.csv")
        if not rows:
            for row in range(2):
                axes[row, col].text(0.5, 0.5, "missing", ha="center",
                                    va="center",
                                    transform=axes[row, col].transAxes)
            continue
        _to_float(rows, ("step", "fom_n_alive", "fom_n_inside_annulus",
                         "rom_n_alive", "rom_n_inside_annulus"))
        step = np.array([r["step"] for r in rows])
        fom = np.array([r["fom_n_inside_annulus"] for r in rows])
        rom = np.array([r["rom_n_inside_annulus"] for r in rows])
        # Linear top
        axes[0, col].plot(step, fom, color="#1f77b4", linewidth=1.7,
                          label="FOM")
        axes[0, col].plot(step, rom, color="#d62728", linewidth=1.7,
                          linestyle="--", label="ROM")
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].set_title(f"case {case}")
        if col == 0:
            axes[0, col].set_ylabel("linear\n# in annulus (r ∈ [5, 10] mm)")
        # Log bottom — replace non-positive with a small floor for log
        # display without hiding the trapping story.
        fom_p = np.where(fom > 0, fom, 0.5)
        rom_p = np.where(rom > 0, rom, 0.5)
        axes[1, col].semilogy(step, fom_p, color="#1f77b4", linewidth=1.7,
                              label="FOM")
        axes[1, col].semilogy(step, rom_p, color="#d62728", linewidth=1.7,
                              linestyle="--", label="ROM")
        axes[1, col].grid(True, which="both", alpha=0.3)
        # annotate final tail values
        final_fom = int(fom[-1]); final_rom = int(rom[-1])
        axes[1, col].annotate(
            f"final:\n  FOM {final_fom:,}\n  ROM {final_rom:,}",
            xy=(step[-1], max(final_fom, 1)), xytext=(-6, 0),
            textcoords="offset points", ha="right", fontsize=9,
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.9))
        if col == 0:
            axes[1, col].set_ylabel("log\n# in annulus")
    axes[0, 0].legend(loc="upper right", fontsize=10)
    for col in range(4):
        axes[-1, col].set_xlabel("step")
    fig.suptitle("Fig F · Particles inside annular probe vs step "
                 "(HCT-on)\nlog panel exposes the tail: FOM traps 10 – 1000× "
                 "more particles than ROM at step 2000", y=1.001)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"figF_mixing_residence_time.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


def figG_pairwise_separation(rom_recon_root: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    for col, case in enumerate(CASES):
        for row, hct in enumerate(("hct_on", "hct_off")):
            rows = _load_mixing_csvs(rom_recon_root, case, hct,
                                     "pairwise_separation.csv")
            if not rows:
                axes[row, col].text(0.5, 0.5, "missing", ha="center",
                                    va="center", transform=axes[row, col].transAxes)
                continue
            _to_float(rows, ("step", "fom_gmean_sep", "rom_gmean_sep"))
            step = np.array([r["step"] for r in rows])
            axes[row, col].semilogy(step,
                                    [r["fom_gmean_sep"] for r in rows],
                                    color="#1f77b4", linewidth=1.6,
                                    label="FOM")
            axes[row, col].semilogy(step,
                                    [r["rom_gmean_sep"] for r in rows],
                                    color="#d62728", linewidth=1.6,
                                    linestyle="--", label="ROM")
            axes[row, col].grid(True, alpha=0.3, which="both")
            if row == 0:
                axes[row, col].set_title(f"case {case}")
            if col == 0:
                axes[row, col].set_ylabel(
                    f"{'HCT-on' if hct == 'hct_on' else 'HCT-off'}\n"
                    f"gmean pair separation (m)")
    axes[0, 0].legend(loc="upper left", fontsize=10)
    for col in range(4):
        axes[-1, col].set_xlabel("step")
    fig.suptitle("Fig G · Pairwise separation vs step (top-FTLE proxy)\n"
                 "slope difference → different chaotic-advection rate; "
                 "flatter ROM → damped chaotic stretching", y=1.001)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"figG_mixing_pairwise_separation.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diag-dir", type=Path, required=True,
                    help="Directory containing "
                         "radial_samples_cylindrical_<case>.csv, "
                         "ring_samples_cylindrical_<case>.csv, "
                         "divergence_summary_cylindrical_<case>.csv, "
                         "divergence_cylindrical_<case>.vtu")
    ap.add_argument("--rom-recon-root", type=Path, required=True,
                    help="For the mixing CSVs.")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # PT rel_rms at step 950 for Fig E — derived from the same csv already
    # used by the step-950 report (near-pin + outer pooled).  Values
    # inlined here so this script is standalone; adjust if step-950
    # numbers change.
    pt_rel_rms_step950 = {
        "000": 12.17,
        "001": 24.36,     # near-pin 27.08, outer 10.30 — pooled ~24.4
        "003": 42.83,
        "004": 18.10,
    }

    print("Fig A: radial profiles"); figA_radial_profiles(args.diag_dir, args.out_dir)
    print("Fig B: azimuthal profiles"); figB_azimuthal_profiles(args.diag_dir, args.out_dir)
    print("Fig C: azimuthal FFT"); figC_azimuthal_fft(args.diag_dir, args.out_dir)
    print("Fig D: divergence CCDF"); figD_divergence_ccdf(args.diag_dir, args.out_dir)
    print("Fig E: divergence vs PT"); figE_divergence_vs_pt(
        args.diag_dir, args.out_dir, pt_rel_rms_step950)
    print("Fig F: residence-time mixing"); figF_residence_time(args.rom_recon_root, args.out_dir)
    print("Fig G: pairwise separation mixing"); figG_pairwise_separation(args.rom_recon_root, args.out_dir)

    print(f"\nAll figures written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
