"""plot_grid_pt_presentation.py -- presentation-quality figures for the
all-20 cohort grid-PT vs mesh-PT comparison (roadmap §5 / §6).

Input: the two detail CSVs produced by compare_grid_vs_mesh_pt_fom.py at
step 950 and step 2000.  Plus the per-particle displacement VTUs for
the spatial heat-map.

Outputs (all under --out):

  fig1_cohort_rms_step<S>.png       -- one bar per case per variant,
                                        cohort mean line + annotation
  fig2_cohort_distribution.png      -- box plot per variant per step
                                        (spread across 20 cases)
  fig3_growth_scatter.png           -- rms(950) vs rms(2000), one dot
                                        per case per variant + y=x line
  fig4_survivorship.png             -- both-alive fraction per case at
                                        both steps, sorted
  fig5_near_pin_vs_outer.png        -- cohort mean + IQR for near-pin /
                                        outer per variant per step
  fig6_spatial_pattern.png          -- (r, z) heatmap of median |err|,
                                        4 representative cases, winner
                                        variant only

All figures use serif fonts, thick strokes, and a shared 'winner-focus'
palette that keeps 4lvl_hct as the reference and de-emphasises the
other two variants slightly.

Usage:
    python3 scripts/plot_grid_pt_presentation.py \\
        --in-dir /scratch/shared/ROM/rom_out/fom_grid_pt_all20 \\
        --out    paper_figs/rom_pt_step5_step6_all20
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        13,
    "axes.titlesize":   15,
    "axes.labelsize":   13,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  11,
    "figure.titlesize": 17,
    "axes.linewidth":   1.1,
    "grid.linewidth":   0.7,
    "grid.alpha":       0.35,
    "lines.linewidth":  1.8,
})

VARIANTS = ["4lvl_hct", "4lvl_hct_tricubic", "2lvl_hct_tricubic",
            "4lvl_r22_hct", "5lvl_hct", "malmo6_hct"]
VARIANT_LABEL = {
    "4lvl_hct":          "4lvl · HCT · trilin",
    "4lvl_hct_tricubic": "4lvl · HCT · tricubic",
    "2lvl_hct_tricubic": "2lvl · HCT · tricubic",
    "4lvl_r22_hct":      "4lvl_r22 · HCT · trilin",
    "5lvl_hct":          "5lvl · HCT · trilin",
    "malmo6_hct":        "malmo6 · HCT · trilin",
}
VARIANT_COLOUR = {
    "4lvl_hct":          "#1f6f3d",   # deep green -- winner from Part 4
    "4lvl_hct_tricubic": "#4fa26e",   # mid green
    "2lvl_hct_tricubic": "#e0a336",   # amber
    "4lvl_r22_hct":      "#8f4dbf",   # purple  -- shear-band hypothesis
    "5lvl_hct":          "#c14a4a",   # brick   -- extra refinement
    "malmo6_hct":        "#3060c0",   # blue    -- MALMO close-out
}
VARIANT_HATCH = {
    "4lvl_hct":          "",
    "4lvl_hct_tricubic": "///",
    "2lvl_hct_tricubic": "\\\\\\",
    "4lvl_r22_hct":      "xx",
    "5lvl_hct":          "..",
    "malmo6_hct":        "++",
}


def _load(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        for k in ("rms", "rms_alive", "rms_near_alive", "rms_outer_alive",
                  "rms_x_alive", "rms_y_alive", "rms_z_alive",
                  "delta_trapped"):
            r[k] = float(r[k]) if r.get(k, "") not in ("", "nan") else float("nan")
        for k in ("n", "n_both_alive", "n_near_alive", "n_outer_alive",
                  "n_only_grid_escaped", "n_only_mesh_escaped",
                  "n_both_escaped"):
            r[k] = int(r[k])
    return rows


# --------------------------------------------------------------------------
# Fig 1: cohort strip — one bar per case × variant.
# --------------------------------------------------------------------------
def fig1_cohort_rms(rows, step, out_path):
    cases  = sorted({r["case"] for r in rows})
    # Sort by mean rms_alive across variants -- helps the reader see
    # the case-difficulty spectrum left-to-right.
    def _case_score(c):
        vals = [r["rms_alive"] for r in rows if r["case"] == c]
        return np.mean(vals) if vals else np.inf
    cases = sorted(cases, key=_case_score)

    n_case = len(cases); n_var = len(VARIANTS)
    fig, ax = plt.subplots(figsize=(16, 6.5))
    x = np.arange(n_case)
    width = 0.8 / n_var

    for vi, v in enumerate(VARIANTS):
        vals = []
        for c in cases:
            m = [r for r in rows if r["case"] == c and r["grid_type"] == v]
            vals.append(m[0]["rms_alive"] if m else np.nan)
        offset = (vi - (n_var - 1) / 2) * width
        ax.bar(x + offset, np.array(vals) * 1000, width,
               color=VARIANT_COLOUR[v], edgecolor="black",
               linewidth=0.6, hatch=VARIANT_HATCH[v],
               label=VARIANT_LABEL[v])

    # Cohort mean line for the winner
    winner_vals = np.array([
        [r["rms_alive"] for r in rows if r["case"] == c and r["grid_type"] == "4lvl_hct"][0]
        for c in cases])
    cohort_mean = winner_vals.mean() * 1000
    ax.axhline(cohort_mean, color="#1f6f3d", linestyle=":", linewidth=1.8,
               alpha=0.7,
               label=f"4lvl_hct cohort mean = {cohort_mean:.2f} mm")

    ax.set_xticks(x)
    ax.set_xticklabels([f"c{c}" for c in cases], rotation=0)
    ax.set_ylabel("rms position error [both-alive] (mm)")
    ax.set_xlabel("case (sorted by cohort-mean error, left = easiest)")
    subtitle = (f"20-case cohort · step {step} "
                f"({'mid-run t=3.6s' if step == 950 else 'final t=7.5s'}) "
                f"· reference: mesh + HCT-3D")
    ax.set_title(f"Grid PT accuracy across the FSW 20-case cohort\n{subtitle}",
                 pad=12)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y")
    ax.legend(loc="upper left", framealpha=0.95, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------
# Fig 2: cohort distribution -- box plots
# --------------------------------------------------------------------------
def fig2_cohort_distribution(r950, r2000, out_path):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, rows, step in ((axs[0], r950, 950), (axs[1], r2000, 2000)):
        data = []
        for v in VARIANTS:
            vals = np.array([r["rms_alive"] for r in rows if r["grid_type"] == v]) * 1000
            data.append(vals)
        bp = ax.boxplot(data, patch_artist=True,
                        tick_labels=[VARIANT_LABEL[v] for v in VARIANTS],
                        widths=0.55, showmeans=True,
                        meanprops={"marker": "D", "markerfacecolor": "white",
                                   "markeredgecolor": "black", "markersize": 8},
                        medianprops={"color": "black", "linewidth": 2})
        for patch, v in zip(bp["boxes"], VARIANTS):
            patch.set_facecolor(VARIANT_COLOUR[v])
            patch.set_alpha(0.85)
            patch.set_edgecolor("black")
        # Overlay individual points
        for xi, v in enumerate(VARIANTS, start=1):
            vals = np.array([r["rms_alive"] for r in rows if r["grid_type"] == v]) * 1000
            jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full_like(vals, xi) + jitter, vals,
                       s=22, color="black", alpha=0.55, zorder=3)
        title_step = f"step {step} ({'mid-run t=3.6s' if step == 950 else 'final t=7.5s'})"
        ax.set_title(title_step)
        ax.set_ylabel("rms position error [both-alive] (mm)")
        ax.grid(axis="y")
        for label in ax.get_xticklabels():
            label.set_rotation(15)
    axs[0].set_ylabel("rms position error [both-alive] (mm)")
    fig.suptitle("Cohort robustness of the top-3 grid variants  ·  20-case spread",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------
# Fig 3: rms(950) vs rms(2000) scatter -- shows compounding + survivorship
# --------------------------------------------------------------------------
def fig3_growth_scatter(r950, r2000, out_path):
    fig, ax = plt.subplots(figsize=(9, 8))

    # Reference y = x line -- auto-fit range from data
    all_950  = [r["rms_alive"]*1000 for r in r950]
    all_2000 = [r["rms_alive"]*1000 for r in r2000]
    lo = max(0, min(min(all_950), min(all_2000)) - 0.5)
    hi = max(max(all_950), max(all_2000)) + 0.5
    ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--",
            linewidth=1.2, label="y = x (no change)")

    for v in VARIANTS:
        xs, ys = [], []
        for r_a in r950:
            if r_a["grid_type"] != v:
                continue
            r_b = next((r for r in r2000
                        if r["case"] == r_a["case"] and r["grid_type"] == v),
                       None)
            if r_b is None: continue
            xs.append(r_a["rms_alive"] * 1000)
            ys.append(r_b["rms_alive"] * 1000)
        ax.scatter(xs, ys, s=110, color=VARIANT_COLOUR[v], edgecolor="black",
                   linewidth=0.9, alpha=0.85, label=VARIANT_LABEL[v])

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("rms at step 950 (mm)")
    ax.set_ylabel("rms at step 2000 (mm)")
    ax.grid()
    ax.set_title("Error evolution 950 → 2000 · one point per case per variant\n"
                 "(points below y=x = survivors are 'easier'; above = "
                 "difficulty grows with time)", pad=10)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------
# Fig 4: survivorship
# --------------------------------------------------------------------------
def fig4_survivorship(r950, r2000, out_path):
    # Use 4lvl_hct as representative -- survivorship depends only on
    # mesh + grid geometry, not on the interp choice.
    cases = sorted({r["case"] for r in r950})
    def _frac(rows, c):
        r = next((r for r in rows if r["case"] == c and r["grid_type"] == "4lvl_hct"), None)
        return 100 * r["n_both_alive"] / r["n"] if r else np.nan
    frac_950  = np.array([_frac(r950,  c) for c in cases])
    frac_2000 = np.array([_frac(r2000, c) for c in cases])
    order = np.argsort(-frac_2000)   # descending: keepers first
    cases_ord = [cases[i] for i in order]
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(cases_ord))
    width = 0.4
    ax.bar(x - width/2, frac_950[order], width,
           color="#8fb8d4", edgecolor="black", linewidth=0.6,
           label="step 950 (t=3.6s)")
    ax.bar(x + width/2, frac_2000[order], width,
           color="#1f4d6b", edgecolor="black", linewidth=0.6,
           label="step 2000 (t=7.5s)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{c}" for c in cases_ord])
    ax.set_ylabel("both-alive particle fraction (%)")
    ax.set_xlabel("case (sorted by final-step survivorship, left = keeper cases)")
    ax.set_title("Survivorship across the cohort · "
                 f"median at final step = {np.median(frac_2000):.1f}% "
                 f"(range {frac_2000.min():.1f}% – {frac_2000.max():.1f}%)")
    ax.axhline(np.median(frac_2000), color="#1f4d6b", linestyle=":",
               linewidth=1.6, alpha=0.7,
               label=f"final-step median = {np.median(frac_2000):.1f}%")
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------
# Fig 5: near-pin vs outer at both time steps
# --------------------------------------------------------------------------
def fig5_near_pin_vs_outer(r950, r2000, out_path):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    for ax, rows, step in ((axs[0], r950, 950), (axs[1], r2000, 2000)):
        x = np.arange(len(VARIANTS))
        width = 0.35
        for offset, key, label, hatch in ((-width/2, "rms_near_alive",  "near-pin (r ≤ 10 mm)", ""),
                                          ( width/2, "rms_outer_alive", "outer (r > 10 mm)",   "///")):
            vals = []
            errs = []
            for v in VARIANTS:
                arr = np.array([r[key] for r in rows if r["grid_type"] == v
                                and np.isfinite(r[key])]) * 1000
                vals.append(np.mean(arr))
                errs.append(np.std(arr))
            colours = [VARIANT_COLOUR[v] for v in VARIANTS]
            ax.bar(x + offset, vals, width, yerr=errs, capsize=5,
                   color=colours, edgecolor="black", linewidth=0.6,
                   hatch=hatch, label=label,
                   error_kw={"linewidth": 1.2, "ecolor": "black"})
        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_LABEL[v] for v in VARIANTS])
        ax.set_title(f"step {step} ({'mid-run t=3.6s' if step == 950 else 'final t=7.5s'})")
        ax.grid(axis="y")
        ax.legend(loc="upper left")
        for label in ax.get_xticklabels():
            label.set_rotation(12)
    axs[0].set_ylabel("cohort-mean rms position error (mm)  ·  bars = ±1σ")
    fig.suptitle("Near-pin vs outer error breakdown  ·  cohort mean ± 1σ across 20 cases",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------
# Fig 6: spatial (r, z) heat-map for representative cases
# --------------------------------------------------------------------------
def _read_error_vtu(path):
    """Load the mesh_pos + error_magnitude arrays from the per-particle VTU."""
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError:
        return None, None
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    ug = reader.GetOutput()
    pos = vtk_to_numpy(ug.GetPoints().GetData()).astype(np.float64)
    err_arr = ug.GetPointData().GetArray("error_magnitude")
    err = vtk_to_numpy(err_arr).astype(np.float64) if err_arr else None
    return pos, err


def fig6_spatial_pattern(vtu_dir, out_path, step=2000,
                         variant="4lvl_hct",
                         cases=("008", "018", "011", "013")):
    """4 representative cases: 008/018 = keepers, 011 = mid, 013 = flusher."""
    from scipy.stats import binned_statistic_2d
    r_edges = np.linspace(0.0,     0.030,  31)
    z_edges = np.linspace(-0.0045, 0.0,    16)
    fig, axs = plt.subplots(1, len(cases), figsize=(16, 4.5),
                            sharex=True, sharey=True)
    # First pass: collect all errs to fix a common colour scale
    stats = []
    for c in cases:
        vtu = vtu_dir / f"pt_error_case{c}_grid_{variant}_step{step}.vtu"
        pos, err = _read_error_vtu(vtu)
        if pos is None or err is None:
            stats.append(None); continue
        r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        z = pos[:, 2]
        # Restrict to inside the domain: x_mesh < outlet 0.03 (rough proxy)
        alive = pos[:, 0] < 0.030
        s, _, _, _ = binned_statistic_2d(r[alive], z[alive], err[alive],
                                          statistic="median",
                                          bins=[r_edges, z_edges])
        stats.append(s)
    all_finite = np.concatenate([s[np.isfinite(s)] for s in stats if s is not None])
    if all_finite.size == 0:
        print(f"[skip] fig6: no data in any of {cases}")
        plt.close(fig); return
    from matplotlib.colors import LogNorm
    vmax = np.percentile(all_finite, 95)
    vmin = max(np.percentile(all_finite[all_finite > 0], 5)
               if (all_finite > 0).any() else 1e-4, 1e-5)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    im = None
    for ax, c, stat in zip(axs, cases, stats):
        if stat is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        R, Z = np.meshgrid(r_edges * 1000, z_edges * 1000, indexing="ij")
        im = ax.pcolormesh(R, Z, np.where(np.isnan(stat), 0, stat) * 1000,
                            cmap="viridis",
                            norm=LogNorm(vmin=vmin*1000, vmax=vmax*1000),
                            shading="auto")
        ax.axvline(10, color="red", linestyle="--", linewidth=1.2,
                   alpha=0.7, label="r = 10 mm (near-pin boundary)")
        ax.set_title(f"case {c}")
        ax.set_xlabel("r_mesh at step 2000 (mm)")
    axs[0].set_ylabel("z_mesh (mm)")
    axs[0].legend(loc="lower left", fontsize=10)
    if im is not None:
        cbar = fig.colorbar(im, ax=axs, shrink=0.9, pad=0.02)
        cbar.set_label("median |err| per (r, z) bin (mm), log", fontsize=11)
    fig.suptitle(f"Spatial breakdown of grid PT error  ·  variant = {variant}  ·  step {step}\n"
                 "Where in the domain does the deviation come from?", y=1.02)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir",  type=Path,
                    default=Path("/scratch/shared/ROM/rom_out/fom_grid_pt_all20"),
                    help="Directory containing grid_vs_mesh_pt_detail_step<S>.csv + "
                         "pt_error_case<c>_grid_<VAR>_step<S>.vtu files.")
    ap.add_argument("--out", type=Path,
                    default=Path("paper_figs/rom_pt_step5_step6_all20"),
                    help="Output directory for the presentation PNGs.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    r950  = _load(args.in_dir / "grid_vs_mesh_pt_detail_step950.csv")
    r2000 = _load(args.in_dir / "grid_vs_mesh_pt_detail_step2000.csv")

    fig1_cohort_rms(r950,  950,  args.out / "fig1_cohort_rms_step950.png")
    fig1_cohort_rms(r2000, 2000, args.out / "fig1_cohort_rms_step2000.png")
    fig2_cohort_distribution(r950, r2000, args.out / "fig2_cohort_distribution.png")
    fig3_growth_scatter(r950, r2000, args.out / "fig3_growth_scatter.png")
    fig4_survivorship(r950, r2000, args.out / "fig4_survivorship.png")
    fig5_near_pin_vs_outer(r950, r2000, args.out / "fig5_near_pin_vs_outer.png")
    fig6_spatial_pattern(args.in_dir, args.out / "fig6_spatial_pattern.png")


if __name__ == "__main__":
    main()
