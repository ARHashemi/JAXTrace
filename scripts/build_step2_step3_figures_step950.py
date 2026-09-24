"""
build_step2_step3_figures_step950.py -- late-time (step 950) figures
for the ROM PT roadmap §§ 2 and 3, including a new
divergence-free-proxy figure (particles trapped near the tool).

Design differences vs the step-500 companion script
(build_step2_step3_figures.py):

  1. Statistics use the FULL particle set (both alive and escaped).
     Rationale: the tracker extends escaped particles ballistically
     from the last valid velocity + inlet extension, so an escaped
     particle continues on a physically defined trajectory.  Dropping
     it hides ROM-vs-FOM residual that accumulates on the extension
     and biases the near-pin bin (particles that reach the outlet
     get pruned from the outer-domain sample).  The step-500 report
     used the both-alive subset for compatibility with compare.log's
     "fair comparison" block; here we prefer the full set.

  2. Reporting step is 950 (t = 3.56 s given DT = 3.75e-3 s), late
     enough for particles seeded upstream of the tool to have
     traversed the tool region.  Trapping is defined at this step:
     a particle whose final r = sqrt(x**2 + y**2) is still <=
     r_near_pin (default 0.010 m) is counted as trapped.  For a
     divergence-free advective field with inlet/outlet handling
     that respects incompressibility, trapping should be zero.
     A non-zero trapping count is a Lagrangian proxy for how far
     off div-free the numerical velocity field behaves.

  3. Fig 5 is expanded to four separate PNGs, one per comparison
     type, each with four case panels: rom_vs_fom_hct_on/off and
     fom/rom_hct_on_vs_off.  The step-500 script only had a
     rom_vs_fom_hct_on panel set.

  4. New Fig 7: bar chart of trapped-particle counts per (case,
     comparison, tracker source).  This is the divergence-free
     figure.

Writes to <outdir>/step950/:
  fig1_step950_r_bin_bars.{png,svg}
  fig2_step950_eulerian_vs_lagrangian.{png,svg}
  fig3_step950_hct_ablation_bars.{png,svg}
  fig4_step950_hct_gap_closure.{png,svg}
  fig5a_step950_spatial_rom_vs_fom_hct_on.{png,svg}
  fig5b_step950_spatial_rom_vs_fom_hct_off.{png,svg}
  fig5c_step950_spatial_fom_hct_on_vs_off.{png,svg}
  fig5d_step950_spatial_rom_hct_on_vs_off.{png,svg}
  fig6_step950_displacement_cdf.{png,svg}
  fig7_step950_trapped_particles.{png,svg}
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
from matplotlib.lines import Line2D
import vtk
from vtk.util.numpy_support import vtk_to_numpy


EULERIAN_REL_RMS_PCT = {
    "000": 6.40,
    "001": 3.75,
    "003": 2.62,
    "004": 3.61,
}

CASES = ["000", "001", "003", "004"]
HCT_MODES = ["hct_on", "hct_off"]
COMPARISONS = [
    ("rom_vs_fom_hct_on",  "ROM (HCT-on) vs FOM (HCT-on)"),
    ("rom_vs_fom_hct_off", "ROM (HCT-off) vs FOM (HCT-off)"),
    ("fom_hct_on_vs_off",  "FOM HCT-on vs FOM HCT-off"),
    ("rom_hct_on_vs_off",  "ROM HCT-on vs ROM HCT-off"),
]

CASE_COLOURS = {
    "000": "#d62728",
    "001": "#1f77b4",
    "003": "#2ca02c",
    "004": "#ff7f0e",
}

STEP = 950
R_NEAR_PIN = 0.010     # metres


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_binning(path: Path) -> list[dict]:
    """Return the pt_r_binning CSV as a list of dicts with numeric fields."""
    out: list[dict] = []
    numeric = ("r_lo", "r_hi", "n_total", "n_used", "n_bin",
               "mean", "median", "rms", "p95", "p99", "max",
               "rms_x", "rms_y", "rms_z", "rms_rel_fom_diag",
               "n_trapped_fom", "n_trapped_rom", "n_trapped_both")
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            for k in numeric:
                if k not in row or row[k] == "":
                    row[k] = float("nan")
                else:
                    row[k] = float(row[k])
            out.append(row)
    return out


def _bin_lookup(rows: list[dict], case: str, label: str) -> dict:
    """Extract per-r-bin stats for a given case+label."""
    hits = [r for r in rows if f"cylindrical_{case}.gid" in r["vtu"]
            and f"/{label}/" in r["vtu"]]
    if not hits:
        return {}
    near = next((r for r in hits if r["r_lo"] == 0.0), None)
    outer = next((r for r in hits if r["r_lo"] == 0.010), None)
    return {"near": near, "outer": outer}


def _load_aggregate_pt_from_csv(rows: list[dict]) -> dict[tuple[str, str], float]:
    """{(case, hct_mode): rel_rms_pct} aggregated from the pt_r_binning CSV.

    We pool the near-pin and outer bins for each case+hct comparison by
    computing rms = sqrt(sum_i (rms_i^2 * n_i) / sum_i n_i), then divide
    by the FOM diagonal (both bins share the same diagonal for a given
    comparison, so we can recover the diagonal from rms and rel_rms of
    either bin).  This is the exact all-particles aggregate rel_rms at
    the same step the CSV was generated for, without depending on
    compare.log parsing (compare.log always reports the step at which
    the tool was last invoked -- may be different from our target step).
    """
    out: dict[tuple[str, str], float] = {}
    for case in CASES:
        for hct in HCT_MODES:
            label = f"rom_vs_fom_{hct}"
            near_outer = _bin_lookup(rows, case, label)
            if not near_outer:
                continue
            near = near_outer.get("near")
            outer = near_outer.get("outer")
            if not near or not outer:
                continue

            # Recover FOM diagonal from either row: rms = rel% * diag / 100.
            # Both rows share the same diagonal so pick near.
            rel = near["rms_rel_fom_diag"]
            rms = near["rms"]
            if not (rel > 0 and rms > 0):
                continue
            fom_diag = 100.0 * rms / rel

            # Pool the two bins by squared-rms with counts as weights.
            n_near = near["n_bin"]
            n_outer = outer["n_bin"]
            rms_near = near["rms"]
            rms_outer = outer["rms"]
            n_tot = n_near + n_outer
            if n_tot <= 0:
                continue
            rms_pool = np.sqrt(
                (n_near * rms_near ** 2 + n_outer * rms_outer ** 2) / n_tot)
            out[(case, hct)] = 100.0 * rms_pool / fom_diag
    return out


def _resolve_vtu_path(fom_root: Path, rom_root: Path, case: str,
                      comparison: str) -> Path:
    """Which case tree owns which comparison at step 950."""
    root = fom_root if comparison == "fom_hct_on_vs_off" else rom_root
    return (root / f"cylindrical_{case}.gid" / "post_pt_compare"
            / comparison / f"{comparison}_step{STEP}.vtu")


def _load_vtu_full(vtu_path: Path):
    """Load FOM positions + ROM positions + displacement + escape flags.
    Returns the FULL particle set (no both-alive filter)."""
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(vtu_path))
    r.Update()
    ug = r.GetOutput()
    fom_pos = vtk_to_numpy(ug.GetPoints().GetData()).astype(np.float64)
    pd = ug.GetPointData()
    disp_vec = vtk_to_numpy(pd.GetArray("displacement_vec")).astype(np.float64)
    disp_mag = vtk_to_numpy(pd.GetArray("displacement_mag")).astype(np.float64)
    fom_esc = vtk_to_numpy(pd.GetArray("fom_escaped")).astype(np.float32)
    rom_esc = vtk_to_numpy(pd.GetArray("rom_escaped")).astype(np.float32)
    return {
        "fom_pos":  fom_pos,
        "rom_pos":  fom_pos + disp_vec,
        "disp_mag": disp_mag,
        "fom_esc":  fom_esc,
        "rom_esc":  rom_esc,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig1_r_bin_bars(rows: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
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
                heights.append(stats[bin_kind]["rms_rel_fom_diag"]
                               if stats.get(bin_kind) else 0.0)
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
    ax.set_title("§2 · ROM–FOM PT displacement, radial bin × HCT setting"
                 "\n(all particles, step 950 ≈ t = 3.56 s)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig1_step950_r_bin_bars.{ext}", dpi=300)
    plt.close(fig)


def fig2_eulerian_vs_lagrangian(agg_pt: dict[tuple[str, str], float],
                                out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    if not agg_pt:
        print("WARN: no aggregate PT data — Fig 2 skipped")
        plt.close(fig)
        return
    # Add extra top-and-right headroom so the case 003 annotation (which
    # sits well above y=x) is not clipped.
    lim_max = max(EULERIAN_REL_RMS_PCT.values())
    lim_max = max(lim_max, max(agg_pt.values())) * 1.25
    ax.plot([0, lim_max], [0, lim_max], color="grey", linestyle=":",
            linewidth=1.0, label="y = x")

    # Fine-tuned label offsets to avoid overlap.  At step 950 case 000
    # and case 001 come close in y so their labels need vertical
    # separation.
    label_offsets = {
        "000": (10, -18),   # push below the marker
        "001": (10,  8),    # push above
        "003": (10, -22),   # push below marker (avoids top clipping)
        "004": (10,  8),    # push above
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
        if (case, "hct_on") in agg_pt:
            la_anchor = agg_pt[(case, "hct_on")]
            eu = EULERIAN_REL_RMS_PCT[case]
            dx, dy = label_offsets[case]
            amp = la_anchor / eu
            ax.annotate(f"case {case}  ({amp:.2f}×)",
                        (eu, la_anchor), xytext=(dx, dy),
                        textcoords="offset points", fontsize=10,
                        fontweight="bold", color=CASE_COLOURS[case])

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
                  "trajectory error, step 950")
    ax.set_title("§2 · Eulerian → Lagrangian amplification at step 950\n"
                 "(all particles; longer integration than step 500)")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig2_step950_eulerian_vs_lagrangian.{ext}",
                    dpi=300)
    plt.close(fig)


def fig3_hct_ablation_bars(rows: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    x = np.arange(len(CASES))
    width = 0.35
    for ax, source in zip(axes, ["fom", "rom"]):
        label = f"{source}_hct_on_vs_off"
        near = [_bin_lookup(rows, c, label)["near"]["rms_rel_fom_diag"]
                for c in CASES]
        outer = [_bin_lookup(rows, c, label)["outer"]["rms_rel_fom_diag"]
                 for c in CASES]
        ax.bar(x - width / 2, near, width, color="#c14040",
               edgecolor="black", linewidth=0.5,
               label="near-pin (r ≤ 10 mm)")
        ax.bar(x + width / 2, outer, width, color="#5289c7",
               edgecolor="black", linewidth=0.5,
               label="outer (r > 10 mm)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"case {c}" for c in CASES])
        ax.set_title(f"HCT tracking effect on {source.upper()} field\n"
                     f"(HCT-on vs HCT-off, same velocity source)")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("HCT-induced trajectory shift rel_rms (% of FOM diag)")
    axes[0].legend(loc="upper left", framealpha=0.95, fontsize=9)
    fig.suptitle("§3 · HCT-3D recovery: tracking perturbation "
                 "(all particles, step 950)", y=1.02)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig3_step950_hct_ablation_bars.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig4_hct_gap_closure(rows: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(CASES))
    width = 0.35
    on_vals = [_bin_lookup(rows, c, "rom_vs_fom_hct_on")["near"]["rms_rel_fom_diag"]
               for c in CASES]
    off_vals = [_bin_lookup(rows, c, "rom_vs_fom_hct_off")["near"]["rms_rel_fom_diag"]
                for c in CASES]
    ax.bar(x - width / 2, on_vals,  width, color="#3f8f3f",
           edgecolor="black", linewidth=0.5, label="HCT on")
    ax.bar(x + width / 2, off_vals, width, color="#c14040",
           edgecolor="black", linewidth=0.5, label="HCT off")
    for i, (a, b) in enumerate(zip(on_vals, off_vals)):
        ax.text(i, max(a, b) + 0.6, f"Δ = {a - b:+.2f}%pt", ha="center",
                fontsize=9, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"case {c}" for c in CASES])
    ax.set_ylabel("ROM–FOM near-pin rel_rms (% of FOM diag)")
    ax.set_title("§3 · Does HCT-3D close the ROM–FOM gap? (near-pin, "
                 "step 950)\nΔ ≈ 0 → HCT is neither compensating nor "
                 "masking the ROM error")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig4_step950_hct_gap_closure.{ext}", dpi=300)
    plt.close(fig)


def fig5_spatial_per_comparison(fom_root: Path, rom_root: Path,
                                out_dir: Path) -> None:
    """One 4-panel figure per comparison type."""
    for comparison, title_body in COMPARISONS:
        fig, axes = plt.subplots(2, 2, figsize=(11, 10),
                                 sharex=True, sharey=True)
        data = {}
        all_p99 = 0.0
        for case in CASES:
            vtu = _resolve_vtu_path(fom_root, rom_root, case, comparison)
            if not vtu.exists():
                continue
            d = _load_vtu_full(vtu)
            data[case] = d
            if d["disp_mag"].size:
                all_p99 = max(all_p99, float(np.percentile(d["disp_mag"], 99)))
        vmax = all_p99
        scatter_ret = None

        for ax, case in zip(axes.flat, CASES):
            if case not in data:
                ax.axis("off")
                continue
            d = data[case]
            n = d["fom_pos"].shape[0]
            idx = np.random.default_rng(seed=int(case)).choice(
                n, size=min(15000, n), replace=False)
            sc = ax.scatter(d["fom_pos"][idx, 0], d["fom_pos"][idx, 1],
                            c=d["disp_mag"][idx], s=3.0, cmap="viridis",
                            vmin=0.0, vmax=vmax, rasterized=True)
            theta = np.linspace(0, 2 * np.pi, 128)
            ax.plot(R_NEAR_PIN * np.cos(theta), R_NEAR_PIN * np.sin(theta),
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
        if scatter_ret is not None:
            cbar = fig.colorbar(scatter_ret, ax=axes.ravel().tolist(),
                                fraction=0.03, pad=0.03)
            cbar.set_label("|displacement| (m), colour clipped at 99th pctile")
        fig.suptitle(f"§2 · Spatial map of displacement — {title_body} "
                     f"(step 950)\ndashed circle = r = 10 mm near-pin "
                     f"boundary; all particles",
                     y=0.995)

        # Panel letter: a=rom_vs_fom_hct_on, b=rom_vs_fom_hct_off, ...
        panel_letter = {c[0]: chr(ord("a") + i)
                        for i, c in enumerate(COMPARISONS)}[comparison]
        stem = f"fig5{panel_letter}_step950_spatial_{comparison}"
        for ext in ("png", "svg"):
            fig.savefig(out_dir / f"{stem}.{ext}", dpi=300,
                        bbox_inches="tight")
        plt.close(fig)


def fig6_displacement_cdf(fom_root: Path, rom_root: Path,
                          out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, case in zip(axes, CASES):
        vtu = _resolve_vtu_path(fom_root, rom_root, case,
                                "rom_vs_fom_hct_on")
        if not vtu.exists():
            ax.axis("off"); continue
        d = _load_vtu_full(vtu)
        r_fom = np.sqrt(d["fom_pos"][:, 0] ** 2 + d["fom_pos"][:, 1] ** 2)
        for kind, mask, colour in (
            ("near-pin (r ≤ 10 mm)", r_fom <= R_NEAR_PIN, "#c14040"),
            ("outer (r > 10 mm)",    r_fom >  R_NEAR_PIN, "#5289c7"),
        ):
            vals = np.sort(d["disp_mag"][mask])
            if vals.size == 0:
                continue
            cdf = np.arange(1, vals.size + 1) / vals.size
            ax.plot(vals, cdf, color=colour, linewidth=1.8, label=kind)
        ax.set_xscale("log")
        ax.set_xlabel("|ROM − FOM| (m)")
        ax.set_title(f"case {case}")
        ax.grid(True, which="both", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("cumulative fraction of particles")
            ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    fig.suptitle("§2 · CDF of ROM–FOM displacement per case (rom_vs_fom_hct_on"
                 ", step 950, all particles)"
                 "\nnear-pin curve shifted right → heavier residual near pin",
                 y=1.02)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig6_step950_displacement_cdf.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig7_trapped_particles(rows_rvf: list[dict],
                           rows_hct: list[dict],
                           out_dir: Path) -> None:
    """Divergence-free proxy: bar chart of trapped-particle counts per
    case, per tracker (FOM vs ROM), from the rom_vs_fom_hct_on
    comparison.  For a truly divergence-free field with proper inlet/
    outlet handling all bars should be near zero (all seeded streamlines
    should pass the tool).
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = np.arange(len(CASES))
    width = 0.35
    fom_traps = []
    rom_traps = []
    n_totals = []
    for case in CASES:
        near = _bin_lookup(rows_rvf, case, "rom_vs_fom_hct_on")["near"]
        fom_traps.append(near["n_trapped_fom"] if near else 0)
        rom_traps.append(near["n_trapped_rom"] if near else 0)
        n_totals.append(near["n_total"] if near else 360_000)

    fom_pct = [100.0 * a / b for a, b in zip(fom_traps, n_totals)]
    rom_pct = [100.0 * a / b for a, b in zip(rom_traps, n_totals)]

    ax.bar(x - width / 2, fom_pct, width, color="#c14040",
           edgecolor="black", linewidth=0.5, label="FOM tracker")
    ax.bar(x + width / 2, rom_pct, width, color="#5289c7",
           edgecolor="black", linewidth=0.5, label="ROM tracker")

    for i, (a, b, ac, bc) in enumerate(
            zip(fom_pct, rom_pct, fom_traps, rom_traps)):
        # Show absolute count above the bar, followed by percentage.
        ax.text(i - width / 2, a + 0.7,
                f"{int(ac):,}\n({a:.1f}%)", ha="center", fontsize=8)
        ax.text(i + width / 2, b + 0.7,
                f"{int(bc):,}\n({b:.1f}%)", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"case {c}" for c in CASES])
    ax.set_ylabel(f"particles with r ≤ {R_NEAR_PIN * 1e3:.0f} mm "
                  f"at step 950 (% of seeded)")
    ax.set_title("§2/§3 · Trapped-particle count — divergence-free proxy "
                 "(HCT-on)\nfor an incompressible field with proper "
                 "outlet handling all bars should be ≈ 0")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig7_step950_trapped_particles.{ext}",
                    dpi=300)
    plt.close(fig)


def fig7b_trapping_hct_effect(rows_rvf: list[dict],
                              out_dir: Path) -> None:
    """Companion to Fig 7: does HCT change the trapping count?
    Side-by-side FOM/ROM trapping under HCT-on vs HCT-off."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    x = np.arange(len(CASES))
    width = 0.35
    for ax, tracker in zip(axes, ["fom", "rom"]):
        on_vals = []
        off_vals = []
        n_tot = []
        for case in CASES:
            on = _bin_lookup(rows_rvf, case, "rom_vs_fom_hct_on")["near"]
            off = _bin_lookup(rows_rvf, case, "rom_vs_fom_hct_off")["near"]
            key = f"n_trapped_{tracker}"
            on_vals.append(on[key] if on else 0)
            off_vals.append(off[key] if off else 0)
            n_tot.append(on["n_total"] if on else 360_000)
        on_pct = [100 * a / b for a, b in zip(on_vals, n_tot)]
        off_pct = [100 * a / b for a, b in zip(off_vals, n_tot)]
        ax.bar(x - width / 2, on_pct, width, color="#3f8f3f",
               edgecolor="black", linewidth=0.5, label="HCT on")
        ax.bar(x + width / 2, off_pct, width, color="#c14040",
               edgecolor="black", linewidth=0.5, label="HCT off")
        for i, (a, b) in enumerate(zip(on_pct, off_pct)):
            delta = a - b
            ax.text(i, max(a, b) + 1.0, f"Δ = {delta:+.2f}%pt",
                    ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"case {c}" for c in CASES])
        ax.set_title(f"{tracker.upper()} tracker trapping")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel(f"trapped (% of seeded)")
    axes[0].legend(loc="upper right", framealpha=0.95)
    fig.suptitle("§3 · Does HCT-3D improve trapping? "
                 "(step 950, r ≤ 10 mm)", y=1.02)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig7b_step950_trapping_hct_effect.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rom-vs-fom-csv", type=Path, required=True)
    ap.add_argument("--hct-ablation-csv", type=Path, required=True)
    ap.add_argument("--rom-recon-root", type=Path, required=True)
    ap.add_argument("--fom-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rvf_rows = _load_binning(args.rom_vs_fom_csv)
    hct_rows = _load_binning(args.hct_ablation_csv)
    agg_pt = _load_aggregate_pt_from_csv(rvf_rows)
    print(f"[figures] {len(rvf_rows)} rom-vs-fom rows, "
          f"{len(hct_rows)} hct-ablation rows, "
          f"{len(agg_pt)} aggregate PT points")

    fig1_r_bin_bars(rvf_rows, args.out_dir); print("  fig1 done")
    fig2_eulerian_vs_lagrangian(agg_pt, args.out_dir); print("  fig2 done")
    fig3_hct_ablation_bars(hct_rows, args.out_dir); print("  fig3 done")
    fig4_hct_gap_closure(rvf_rows, args.out_dir); print("  fig4 done")
    fig5_spatial_per_comparison(args.fom_root, args.rom_recon_root,
                                args.out_dir); print("  fig5 done (4 files)")
    fig6_displacement_cdf(args.fom_root, args.rom_recon_root,
                          args.out_dir); print("  fig6 done")
    fig7_trapped_particles(rvf_rows, hct_rows, args.out_dir)
    print("  fig7 done")
    fig7b_trapping_hct_effect(rvf_rows, args.out_dir)
    print("  fig7b done")
    print(f"\n[figures] all outputs under {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
