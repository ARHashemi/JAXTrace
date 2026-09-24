"""
build_step2_step3_figures.py -- comprehensive visualisations for the
ROM PT roadmap sections 2 (FOM-vs-ROM PT comparison) and 3 (HCT-3D
ablation).

Reads:
  * pt_r_binning_step500.csv                 (rom-vs-fom per-r-bin stats)
  * pt_r_binning_hct_ablation_step500.csv    (fom_on-vs-off + rom_on-vs-off)
  * compare.log files under each case's post_pt_compare/ directories
    (for the aggregate both-alive PT rel_rms).
  * Case-level Eulerian rel_rms from rom_reconstruction_findings.md
    (hardcoded here so the figures survive a doc edit).

Writes to <outdir>/:
  fig1_r_bin_bars.{png,svg}      -- rel_rms near-pin vs outer per case
  fig2_eulerian_vs_lagrangian.{png,svg}  -- rel_rms scatter + y=x
  fig3_hct_ablation_bars.{png,svg}   -- HCT tracking effect
  fig4_hct_gap_closure.{png,svg} -- rom-vs-fom rel_rms with HCT on vs off
  fig5_spatial_residual.{png,svg}    -- 4-panel (x,y) scatter coloured
                                        by displacement magnitude
  fig6_displacement_cdf.{png,svg}    -- per-case CDF split near/outer

All figures at 300 dpi, both PNG and SVG for slide/paper reuse.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy


# Eulerian rel_rms values from rom_reconstruction_findings.md (per-case
# table, "centered" formula column).  Kept as a small hardcoded dict so
# the figures do not silently drift if the findings doc is edited.
EULERIAN_REL_RMS_PCT = {
    "000": 6.40,
    "001": 3.75,
    "003": 2.62,
    "004": 3.61,
}

CASES = ["000", "001", "003", "004"]
HCT_MODES = ["hct_on", "hct_off"]

# Consistent case colours across figures
CASE_COLOURS = {
    "000": "#d62728",  # red
    "001": "#1f77b4",  # blue
    "003": "#2ca02c",  # green
    "004": "#ff7f0e",  # orange
}

# Consistent HCT hatches
HCT_HATCH = {"hct_on": "", "hct_off": "//"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_binning(path: Path) -> list[dict]:
    """Return the pt_r_binning CSV as a list of dicts with numeric fields."""
    out: list[dict] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            for k in ("r_lo", "r_hi", "n_alive", "n_bin", "mean", "median",
                     "rms", "p95", "p99", "max", "rms_x", "rms_y", "rms_z",
                     "rms_rel_fom_diag"):
                row[k] = float(row[k]) if row[k] != "" else float("nan")
            out.append(row)
    return out


def _bin_lookup(rows: list[dict], case: str, label: str) -> dict[str, float]:
    """Extract near-pin + outer rel_rms% (both-alive) for a given case, label.

    Returns dict {'near': %, 'outer': %}.
    """
    hits = [r for r in rows if f"cylindrical_{case}.gid" in r["vtu"]
            and f"/{label}/" in r["vtu"]]
    near = next(r["rms_rel_fom_diag"] for r in hits if r["r_lo"] == 0.0)
    outer = next(r["rms_rel_fom_diag"] for r in hits if r["r_lo"] == 0.010)
    return {"near": near, "outer": outer}


def _parse_compare_log(log_path: Path) -> dict[str, float]:
    """Extract the both-alive rel_rms% from a compare.log."""
    text = log_path.read_text()
    # The both-alive block always comes after the "all particles" block.
    ba_pos = text.find("both-alive subset")
    if ba_pos < 0:
        raise KeyError(f"{log_path}: no both-alive block")
    tail = text[ba_pos:]
    m = re.search(r"rms\s*/\s*FOM diagonal\s*:\s*([\d.]+)\s*%", tail)
    if not m:
        raise KeyError(f"{log_path}: no rel_rms line under both-alive")
    return {"rel_rms_pct": float(m.group(1))}


def _load_aggregate_pt(rom_root: Path) -> dict[tuple[str, str], float]:
    """{(case, hct_mode): rel_rms_pct} from compare logs."""
    out = {}
    for case in CASES:
        for hct in HCT_MODES:
            log = (rom_root / f"cylindrical_{case}.gid"
                   / "post_pt_compare" / f"rom_vs_fom_{hct}" / "compare.log")
            if log.exists():
                out[(case, hct)] = _parse_compare_log(log)["rel_rms_pct"]
    return out


def _load_vtu_for_spatial(vtu_path: Path):
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(vtu_path))
    r.Update()
    ug = r.GetOutput()
    pts = vtk_to_numpy(ug.GetPoints().GetData()).astype(np.float64)
    pd = ug.GetPointData()
    disp_mag = vtk_to_numpy(pd.GetArray("displacement_mag")).astype(np.float64)
    fom_esc = vtk_to_numpy(pd.GetArray("fom_escaped")).astype(np.float32)
    rom_esc = vtk_to_numpy(pd.GetArray("rom_escaped")).astype(np.float32)
    both_alive = (fom_esc == 0) & (rom_esc == 0)
    return pts[both_alive], disp_mag[both_alive]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig1_r_bin_bars(rows: list[dict], out_dir: Path) -> None:
    """§2 — near-pin vs outer per case and HCT setting.

    Grouped bars: x = case, hue = (HCT × bin).  Four bars per case.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CASES))
    width = 0.20
    offsets = {
        ("hct_on", "near"):  -1.5 * width,
        ("hct_on", "outer"): -0.5 * width,
        ("hct_off", "near"):  0.5 * width,
        ("hct_off", "outer"): 1.5 * width,
    }
    colours = {"near": "#c14040", "outer": "#5289c7"}
    hatches = {"hct_on": "", "hct_off": "//"}
    labels_used = set()

    for hct in HCT_MODES:
        for bin_kind in ("near", "outer"):
            heights = []
            for case in CASES:
                stats = _bin_lookup(rows, case, f"rom_vs_fom_{hct}")
                heights.append(stats[bin_kind])
            label = f"{bin_kind} r bin, HCT-{'on' if hct == 'hct_on' else 'off'}"
            if label in labels_used:
                label = None
            else:
                labels_used.add(label)
            ax.bar(x + offsets[(hct, bin_kind)], heights, width,
                   color=colours[bin_kind], hatch=hatches[hct],
                   edgecolor="black", linewidth=0.5, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([f"case {c}" for c in CASES])
    ax.set_ylabel("ROM–FOM displacement rel_rms (% of FOM diag)")
    ax.set_title("§2 · ROM–FOM PT displacement, by radial bin and HCT setting"
                 "\n(both-alive particles, step 500)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig1_r_bin_bars.{ext}", dpi=300)
    plt.close(fig)


def fig2_eulerian_vs_lagrangian(agg_pt: dict[tuple[str, str], float],
                                out_dir: Path) -> None:
    """§2 — cohort cross-plot: Eulerian rel_rms vs Lagrangian rel_rms."""
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    lim_max = max(EULERIAN_REL_RMS_PCT.values()) * 1.15
    lim_max = max(lim_max, max(agg_pt.values()) * 1.15)
    ax.plot([0, lim_max], [0, lim_max], color="grey", linestyle=":",
            linewidth=1.0, label="y = x")

    # Manual label offsets per case to prevent overlap.  Since HCT-on and
    # HCT-off values differ by <0.3 %pt, the two markers per case stack —
    # we annotate only once per case (using HCT-on as the anchor) and
    # place the label where it won't collide.
    label_offsets = {
        "000": (10, -4),
        "001": (10, 0),
        "003": (10, 4),
        "004": (10, -12),
    }
    for case in CASES:
        eu = EULERIAN_REL_RMS_PCT[case]
        for hct in HCT_MODES:
            if (case, hct) not in agg_pt:
                continue
            la = agg_pt[(case, hct)]
            marker = "o" if hct == "hct_on" else "s"
            ax.scatter(eu, la, s=140, marker=marker,
                       facecolor=CASE_COLOURS[case], edgecolor="black",
                       linewidth=0.7, zorder=3)
        # One annotation per case, anchored on the HCT-on point.
        if (case, "hct_on") in agg_pt:
            la_anchor = agg_pt[(case, "hct_on")]
            eu = EULERIAN_REL_RMS_PCT[case]
            dx, dy = label_offsets[case]
            # Amplification factor, shown right next to the case label.
            amp = la_anchor / eu
            txt = f"case {case}  ({amp:.2f}×)"
            ax.annotate(txt, (eu, la_anchor), xytext=(dx, dy),
                        textcoords="offset points", fontsize=10,
                        fontweight="bold", color=CASE_COLOURS[case])

    # legend proxies for HCT
    from matplotlib.lines import Line2D
    proxies = [
        Line2D([0], [0], marker="o", linestyle="", markersize=10,
               markerfacecolor="lightgrey", markeredgecolor="black",
               label="HCT on"),
        Line2D([0], [0], marker="s", linestyle="", markersize=10,
               markerfacecolor="lightgrey", markeredgecolor="black",
               label="HCT off"),
        Line2D([0], [0], color="grey", linestyle=":", label="y = x"),
    ]
    ax.legend(handles=proxies, loc="upper left", framealpha=0.95)

    ax.set_xlabel("Eulerian rel_rms (% of FOM velocity max) — "
                  "reconstruction error")
    ax.set_ylabel("Lagrangian PT rel_rms (% of FOM diag) — "
                  "trajectory error")
    ax.set_title("§2 · Eulerian-to-Lagrangian error amplification\n"
                 "point above y=x → PT error worse than Eulerian error")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig2_eulerian_vs_lagrangian.{ext}", dpi=300)
    plt.close(fig)


def fig3_hct_ablation_bars(rows: list[dict], out_dir: Path) -> None:
    """§3 — HCT on-vs-off tracking effect, per source (FOM, ROM) and per r-bin."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    x = np.arange(len(CASES))
    width = 0.35
    for ax, source in zip(axes, ["fom", "rom"]):
        label = f"{source}_hct_on_vs_off"
        near = [_bin_lookup(rows, c, label)["near"] for c in CASES]
        outer = [_bin_lookup(rows, c, label)["outer"] for c in CASES]
        ax.bar(x - width / 2, near, width, color="#c14040",
               edgecolor="black", linewidth=0.5, label="near-pin (r ≤ 10 mm)")
        ax.bar(x + width / 2, outer, width, color="#5289c7",
               edgecolor="black", linewidth=0.5, label="outer (r > 10 mm)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"case {c}" for c in CASES])
        ax.set_title(f"HCT tracking effect on {source.upper()} field\n"
                     f"(HCT-on vs HCT-off, same velocity source)")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("HCT-induced trajectory shift rel_rms (% of FOM diag)")

    axes[0].legend(loc="upper left", framealpha=0.95, fontsize=9)
    fig.suptitle("§3 · HCT-3D recovery: tracking perturbation "
                 "(both-alive, step 500)", y=1.02)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig3_hct_ablation_bars.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def fig4_hct_gap_closure(rows: list[dict], out_dir: Path) -> None:
    """§3 — does HCT close the ROM/FOM gap? Compare rom-vs-fom rel_rms
    near-pin with HCT-on vs HCT-off, side by side.  If HCT compensates,
    the HCT-on bar should be lower than HCT-off.  If HCT masks, they
    match.  Here they match → neither.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CASES))
    width = 0.35
    on_vals  = [_bin_lookup(rows, c, "rom_vs_fom_hct_on")["near"] for c in CASES]
    off_vals = [_bin_lookup(rows, c, "rom_vs_fom_hct_off")["near"] for c in CASES]
    ax.bar(x - width / 2, on_vals,  width, color="#3f8f3f",
           edgecolor="black", linewidth=0.5, label="HCT on")
    ax.bar(x + width / 2, off_vals, width, color="#c14040",
           edgecolor="black", linewidth=0.5, label="HCT off")
    for i, (a, b) in enumerate(zip(on_vals, off_vals)):
        ax.text(i, max(a, b) + 0.15, f"Δ = {a - b:+.2f}%pt", ha="center",
                fontsize=9, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"case {c}" for c in CASES])
    ax.set_ylabel("ROM–FOM near-pin rel_rms (% of FOM diag)")
    ax.set_title("§3 · Does HCT-3D close the ROM–FOM gap? (near-pin subset)\n"
                 "Δ ≈ 0 → HCT is neither compensating nor masking the ROM error")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig4_hct_gap_closure.{ext}", dpi=300)
    plt.close(fig)


def fig5_spatial_residual(rom_root: Path, out_dir: Path) -> None:
    """§2 — where does the residual live?  Scatter (x,y) coloured by
    displacement magnitude, one panel per case, all rom_vs_fom_hct_on."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)

    # Compute a global vmax so the colour scales are comparable across cases.
    all_max = 0.0
    data = {}
    for case in CASES:
        vtu = (rom_root / f"cylindrical_{case}.gid" / "post_pt_compare"
               / "rom_vs_fom_hct_on" / "rom_vs_fom_hct_on_step500.vtu")
        pts, disp_mag = _load_vtu_for_spatial(vtu)
        data[case] = (pts, disp_mag)
        all_max = max(all_max, float(np.percentile(disp_mag, 99)))

    vmax = all_max
    scatter_ret = None
    for ax, case in zip(axes.flat, CASES):
        pts, disp_mag = data[case]
        # Subsample for plotting speed — 15k points is enough visually.
        n = pts.shape[0]
        idx = np.random.default_rng(seed=int(case)).choice(
            n, size=min(15000, n), replace=False)
        sc = ax.scatter(pts[idx, 0], pts[idx, 1], c=disp_mag[idx],
                        s=3.0, cmap="viridis", vmin=0.0, vmax=vmax,
                        rasterized=True)
        # Sketch the near-pin r=0.010 circle for reference.
        theta = np.linspace(0, 2 * np.pi, 128)
        ax.plot(0.010 * np.cos(theta), 0.010 * np.sin(theta),
                color="white", linestyle="--", linewidth=1.0)
        ax.set_aspect("equal", "box")
        ax.set_title(f"case {case} "
                     f"(Eulerian {EULERIAN_REL_RMS_PCT[case]:.2f}%)")
        ax.grid(True, alpha=0.2)
        scatter_ret = sc

    for ax in axes[-1, :]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")

    cbar = fig.colorbar(scatter_ret, ax=axes.ravel().tolist(),
                        fraction=0.03, pad=0.03)
    cbar.set_label("‖ROM − FOM‖ displacement (m), clipped at 99th pctile")
    fig.suptitle("§2 · Spatial map of ROM–FOM displacement, rom_vs_fom_hct_on"
                 " (step 500)\ndashed circle = r = 10 mm near-pin boundary",
                 y=0.995)

    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig5_spatial_residual.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def fig6_displacement_cdf(rom_root: Path, out_dir: Path) -> None:
    """§2 — CDF of displacement magnitude per case, split near-pin vs outer."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, case in zip(axes, CASES):
        vtu = (rom_root / f"cylindrical_{case}.gid" / "post_pt_compare"
               / "rom_vs_fom_hct_on" / "rom_vs_fom_hct_on_step500.vtu")
        pts, disp_mag = _load_vtu_for_spatial(vtu)
        r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        for kind, mask, colour in (
            ("near-pin (r ≤ 10 mm)", r <= 0.010, "#c14040"),
            ("outer (r > 10 mm)",    r >  0.010, "#5289c7"),
        ):
            vals = np.sort(disp_mag[mask])
            if vals.size == 0:
                continue
            cdf = np.arange(1, vals.size + 1) / vals.size
            ax.plot(vals, cdf, color=colour, linewidth=1.8, label=kind)
        ax.set_xscale("log")
        ax.set_xlabel("‖ROM − FOM‖ (m)")
        ax.set_title(f"case {case}")
        ax.grid(True, which="both", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("cumulative fraction of particles")
            ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.suptitle("§2 · CDF of ROM–FOM displacement per case (rom_vs_fom_hct_on)"
                 "\nnear-pin curve shifted right → heavier residual near the pin",
                 y=1.02)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig6_displacement_cdf.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rom-vs-fom-csv", type=Path, required=True)
    ap.add_argument("--hct-ablation-csv", type=Path, required=True)
    ap.add_argument("--rom-recon-root", type=Path, required=True,
                    help="Root of the ROM reconstruction tree, e.g. "
                         "/scratch/shared/ROM/ROM_recon_centered/.  Used to "
                         "reach compare.log files and the rom_vs_fom VTUs "
                         "for spatial + CDF figures.")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rvf_rows = _load_binning(args.rom_vs_fom_csv)
    hct_rows = _load_binning(args.hct_ablation_csv)
    agg_pt = _load_aggregate_pt(args.rom_recon_root)

    print(f"[figures] loaded {len(rvf_rows)} rom-vs-fom rows, "
          f"{len(hct_rows)} HCT-ablation rows, "
          f"{len(agg_pt)} aggregate PT points")

    fig1_r_bin_bars(rvf_rows, args.out_dir)
    print("  fig1 (r-bin bars) done")
    fig2_eulerian_vs_lagrangian(agg_pt, args.out_dir)
    print("  fig2 (eulerian vs lagrangian scatter) done")
    fig3_hct_ablation_bars(hct_rows, args.out_dir)
    print("  fig3 (HCT ablation bars) done")
    fig4_hct_gap_closure(rvf_rows, args.out_dir)
    print("  fig4 (HCT gap closure) done")
    fig5_spatial_residual(args.rom_recon_root, args.out_dir)
    print("  fig5 (spatial residual) done")
    fig6_displacement_cdf(args.rom_recon_root, args.out_dir)
    print("  fig6 (CDF) done")

    print(f"\n[figures] all outputs under {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
