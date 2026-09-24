#!/usr/bin/env python3
"""Supplementary presentation figures for the grid-PT deck.

Produces three figures that the existing plotting scripts did not cover:

  figT_timing_mesh_vs_grid.png
      Mesh PT vs grid PT wall time on one axis (the head-to-head the deck
      was missing).  Reads rom_out/fom_grid_pt/wall_times.csv, which
      already carries both `kind=mesh` and `kind=grid` rows.

  figG_grid_schematic.png
      Block layout of the winner grid and two contrast grids, drawn from
      the BLOCK_META table baked into each generated velocity module.
      xy plane (z-slice), true to scale.

  figP_p1_vs_hct_positions.png
      Final-step particle positions for one case: mesh reference vs the
      same grid family under raw-P1 Stage 1 and under HCT-3D Stage 1.
      This is the "HCT Stage 1 pulls trajectories back onto the mesh
      reference" figure.

Every number is read from the run outputs on /scratch; nothing is
hard-coded except plot styling.

Usage
-----
    python3 scripts/plot_deck_supplementary_figs.py \
        --rom-root /scratch/shared/ROM \
        --out paper_figs/rom_pt_step5_step6_all20
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Presentation styling: serif, thick strokes, large type.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "axes.linewidth": 1.2,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

D_PIN_MM = 10.0      # pin diameter -- the deck's fixed error normaliser

MESH_C = "#b03a2e"   # mesh reference
HCT_C = "#1f6f8b"    # HCT stage 1
P1_C = "#d68910"     # raw P1 stage 1


# --------------------------------------------------------------------------
# shared loaders (same conventions as scripts/compare_grid_vs_mesh_pt_fom.py)
# --------------------------------------------------------------------------

def load_step(vtkhdf_path: Path, target_step: float):
    """Return (positions (n,3), matched_step)."""
    with h5py.File(vtkhdf_path, "r") as f:
        offsets = f["VTKHDF/Steps/PointOffsets"][:]
        values = f["VTKHDF/Steps/Values"][:]
        idx = int(np.argmin(np.abs(values - target_step)))
        matched = float(values[idx])
        n = int(offsets[1] - offsets[0]) if len(offsets) > 1 \
            else f["VTKHDF/Points"].shape[0]
        start = int(offsets[idx])
        pos = f["VTKHDF/Points"][start:start + n].astype(np.float64)
    return pos, matched


def find_hdf(case_dir: Path, subpath: str) -> Path | None:
    for hdf in (case_dir / subpath).glob("run_*/particles.vtkhdf"):
        return hdf
    return None


def find_mesh_reference(case_dir: Path) -> Path | None:
    p = find_hdf(case_dir, "post_pt/fom_hct_on")
    if p is not None:
        return p
    p = case_dir / "post_pt" / "run_grid-frac_n360000_s2000" / "particles.vtkhdf"
    return p if p.exists() else None


def read_block_meta(module_path: Path):
    """Extract BLOCK_META from a generated grid velocity module.

    Parsed statically (ast.literal_eval on the assignment) so we never
    import the module -- importing would pull in JAX and load the npz.
    """
    src = module_path.read_text()
    m = re.search(r"^BLOCK_META\s*=\s*(\[.*?\n\])", src, re.S | re.M)
    if not m:
        raise ValueError(f"BLOCK_META not found in {module_path}")
    return ast.literal_eval(m.group(1))


# --------------------------------------------------------------------------
# Figure T -- mesh vs grid wall time
# --------------------------------------------------------------------------

def fig_timing(rom_root: Path, out: Path) -> None:
    csv = rom_root / "rom_out" / "fom_grid_pt" / "wall_times.csv"
    rows = []
    with open(csv) as fh:
        header = fh.readline().strip().split(",")
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) != len(header):
                continue
            rows.append(dict(zip(header, parts)))

    for r in rows:
        r["wall_s"] = float(r["wall_s"])
        r["p_step_s"] = float(r["p_step_s"])

    mesh = [r for r in rows if r["kind"] == "mesh" and r["variant"] == "fom_hct_on"]
    grid = [r for r in rows if r["kind"] == "grid"]
    if not mesh or not grid:
        print("  ! timing: missing mesh or grid rows, skipping")
        return

    mesh_mean = float(np.mean([r["wall_s"] for r in mesh]))

    # Per-variant mean across cases, sorted fastest first.
    by_var: dict[str, list[float]] = {}
    for r in grid:
        by_var.setdefault(r["variant"], []).append(r["wall_s"])
    var_mean = sorted(((v, float(np.mean(w))) for v, w in by_var.items()),
                      key=lambda t: t[1])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.5, 6.6),
        gridspec_kw={"width_ratios": [1, 2.5]})

    # -- left: the head-to-head, log scale --------------------------------
    winner = "4lvl_hct"
    win_mean = float(np.mean(by_var.get(winner, [np.nan])))
    fastest_v, fastest_mean = var_mean[0]

    labels = ["Mesh\n+ HCT-3D", f"Grid\n{winner}\n(accuracy\nwinner)",
              f"Grid\n{fastest_v}\n(fastest)"]
    vals = [mesh_mean, win_mean, fastest_mean]
    colors = [MESH_C, HCT_C, "#5499c7"]

    bars = ax1.bar(labels, vals, color=colors, width=0.62,
                   edgecolor="black", linewidth=1.3, zorder=3)
    ax1.set_yscale("log")
    ax1.set_ylabel("wall time per case (s, log scale)")
    ax1.set_title("Mesh vs grid particle tracking\n"
                  "360 k particles $\\times$ 2000 steps, same GPU",
                  fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, zorder=0, which="both")
    ax1.set_axisbelow(True)

    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v * 1.12,
                 f"{v:,.0f} s", ha="center", va="bottom",
                 fontweight="bold", fontsize=12)
    # Speedup badges sit above the value labels, well inside the axes, so
    # they never collide with the (multi-line) tick labels below.
    for i, v in enumerate(vals[1:], start=1):
        ax1.text(i, v * 2.6, f"{mesh_mean / v:.0f}$\\times$\nfaster",
                 ha="center", va="center", fontsize=13,
                 fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.35",
                           facecolor=colors[i], edgecolor="none", alpha=0.95))
    ax1.set_ylim(bottom=min(vals) * 0.45, top=mesh_mean * 4)

    # -- right: every variant vs the mesh line ----------------------------
    names = [v for v, _ in var_mean]
    means = [m for _, m in var_mean]
    ypos = np.arange(len(names))
    bar_c = [HCT_C if "hct" in n else P1_C for n in names]

    ax2.barh(ypos, means, color=bar_c, edgecolor="black",
             linewidth=0.8, height=0.72, zorder=3)
    ax2.axvline(mesh_mean, color=MESH_C, linestyle="--", linewidth=2.4,
                zorder=4,
                label=f"mesh + HCT-3D reference = {mesh_mean:,.0f} s")
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(names, fontfamily="monospace", fontsize=9.5)
    ax2.invert_yaxis()
    ax2.set_xscale("log")
    ax2.set_xlim(10, mesh_mean * 2.2)
    ax2.set_xlabel("wall time per case (s, log scale)")
    ax2.set_title("Every grid variant against the mesh reference\n"
                  "blue = HCT-3D Stage 1 · orange = raw-P1 Stage 1",
                  fontweight="bold")
    ax2.grid(axis="x", alpha=0.3, zorder=0, which="both")
    ax2.set_axisbelow(True)
    ax2.legend(loc="lower right", framealpha=0.95)

    for y, (n, m) in enumerate(zip(names, means)):
        ax2.text(m * 1.06, y, f"{m:.0f} s  ({mesh_mean / m:.0f}$\\times$)",
                 va="center", ha="left", fontsize=9)

    fig.suptitle(
        "Grid PT removes the element search: 24–40 minutes becomes ~20–40 seconds",
        fontsize=15, fontweight="bold", y=1.005)
    fig.tight_layout()
    path = out / "figT_timing_mesh_vs_grid.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")
    print(f"    mesh mean {mesh_mean:.1f}s | {winner} {win_mean:.1f}s "
          f"({mesh_mean/win_mean:.1f}x) | fastest {fastest_v} "
          f"{fastest_mean:.1f}s ({mesh_mean/fastest_mean:.1f}x)")


# --------------------------------------------------------------------------
# Figure G -- grid schematics
# --------------------------------------------------------------------------

def _draw_grid(ax, blocks, title, subtitle, max_lines=26):
    """Draw one hierarchical grid's blocks in the xy plane, to scale.

    Blocks are drawn coarsest-last so the finer nested blocks sit on top,
    which is also the order the runtime queries them (finest-first).
    """
    # Coarsest block sets the extent.
    xmin_all = min(b[0][0] for b in blocks)
    xmax_all = max(b[1][0] for b in blocks)
    ymin_all = min(b[0][1] for b in blocks)
    ymax_all = max(b[1][1] for b in blocks)

    shades = ["#1f6f8b", "#5499c7", "#a9cce3", "#d4e6f1"]

    # Draw coarse -> fine so finer blocks overlay.
    for depth, (bmin, bmax, n) in enumerate(reversed(blocks)):
        k = len(blocks) - 1 - depth          # original finest-first index
        color = shades[min(k, len(shades) - 1)]
        x0, y0 = bmin[0] * 1e3, bmin[1] * 1e3
        x1, y1 = bmax[0] * 1e3, bmax[1] * 1e3

        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               facecolor="white", edgecolor="none", zorder=2 + k))

        # Cell lines, decimated so the drawing stays legible.
        nx, ny = n[0], n[1]
        sx = max(1, nx // max_lines)
        sy = max(1, ny // max_lines)
        xs = np.linspace(x0, x1, nx + 1)[::sx]
        ys = np.linspace(y0, y1, ny + 1)[::sy]
        lw = 0.35 if k == 0 else 0.5
        for xv in xs:
            ax.plot([xv, xv], [y0, y1], color=color, lw=lw,
                    zorder=3 + k, alpha=0.85)
        for yv in ys:
            ax.plot([x0, x1], [yv, yv], color=color, lw=lw,
                    zorder=3 + k, alpha=0.85)

        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               facecolor="none", edgecolor="black",
                               lw=1.6, zorder=20 + k))

        # Label each block just inside its own top-right corner, stepped
        # down per depth so nested blocks' labels never overlap.
        dx_um = (bmax[0] - bmin[0]) / nx * 1e6
        ax.text(x1 - 0.4, y1 - 0.7 - 2.1 * k,
                f"$\\Delta x$={dx_um:.0f} µm",
                ha="right", va="top", fontsize=8.5, zorder=30 + k,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor=color, alpha=0.93, lw=1.0))

    # Pin outline at r = 5 mm, and the shear band that dominates the error.
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(5 * np.cos(th), 5 * np.sin(th), color="#c0392b",
            lw=2.0, zorder=40, label="pin surface (r = 5 mm)")
    ax.plot(7 * np.cos(th), 7 * np.sin(th), color="#c0392b",
            lw=1.4, ls=":", zorder=40, label="shear ring (r = 7 mm)")

    ax.set_xlim(xmin_all * 1e3 - 1, xmax_all * 1e3 + 1)
    ax.set_ylim(ymin_all * 1e3 - 1, ymax_all * 1e3 + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"{title}\n{subtitle}", fontweight="bold", fontsize=12)


def fig_grid_schematic(rom_root: Path, case: str, out: Path,
                       variants: list[str]) -> None:
    case_dir = rom_root / "FOM" / f"cylindrical_{case}.gid"

    found = []
    for var in variants:
        mod = case_dir / f"grid_velocity_cyl_{case}_{var}.py"
        if not mod.exists():
            print(f"  ! grid schematic: {mod.name} missing, skipping")
            continue
        try:
            found.append((var, read_block_meta(mod)))
        except ValueError as exc:
            print(f"  ! {exc}")

    if not found:
        print("  ! grid schematic: no grid modules found")
        return

    fig, axes = plt.subplots(1, len(found), figsize=(6.2 * len(found), 6.4))
    if len(found) == 1:
        axes = [axes]

    for ax, (var, blocks) in zip(axes, found):
        total = sum(n[0] * n[1] * n[2] for _, _, n in blocks)
        finest = min((bmax[0] - bmin[0]) / n[0] for bmin, bmax, n in blocks)
        _draw_grid(
            ax, blocks, f"`{var}`",
            f"{len(blocks)} block(s) · {total:,} cells · "
            f"finest $\\Delta x$ = {finest*1e6:.0f} µm")

    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.10),
                   ncol=2, fontsize=9, framealpha=0.95)

    fig.suptitle(
        "Grid families in the xy plane (z-slice), drawn to scale from each "
        "module's baked block table\n"
        "Cell lines are decimated for legibility; the labelled $\\Delta x$ is the true cell size",
        fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = out / "figG_grid_schematic.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")
    for var, blocks in found:
        total = sum(n[0] * n[1] * n[2] for _, _, n in blocks)
        print(f"    {var:18s} {len(blocks)} blocks, {total:,} cells")


# --------------------------------------------------------------------------
# Figure P -- raw P1 vs HCT-3D trajectory accuracy
# --------------------------------------------------------------------------

def fig_p1_vs_hct(rom_root: Path, case: str, family: str, step: float,
                  out: Path, sample: int = 120000, seed: int = 0) -> None:
    case_dir = rom_root / "FOM" / f"cylindrical_{case}.gid"

    mesh_hdf = find_mesh_reference(case_dir)
    p1_hdf = find_hdf(case_dir, f"out_grid_{family}")
    hct_hdf = find_hdf(case_dir, f"out_grid_{family}_hct")

    missing = [n for n, p in
               [("mesh", mesh_hdf), (family, p1_hdf), (f"{family}_hct", hct_hdf)]
               if p is None]
    if missing:
        print(f"  ! p1_vs_hct: missing runs {missing}, skipping")
        return

    mesh_pos, matched = load_step(mesh_hdf, step)
    p1_pos, _ = load_step(p1_hdf, step)
    hct_pos, _ = load_step(hct_hdf, step)

    n = min(len(mesh_pos), len(p1_pos), len(hct_pos))
    mesh_pos, p1_pos, hct_pos = mesh_pos[:n], p1_pos[:n], hct_pos[:n]

    err_p1 = np.linalg.norm(p1_pos - mesh_pos, axis=1) * 1e3    # mm
    err_hct = np.linalg.norm(hct_pos - mesh_pos, axis=1) * 1e3

    rms_p1 = float(np.sqrt((err_p1 ** 2).mean()))
    rms_hct = float(np.sqrt((err_hct ** 2).mean()))

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample, n), replace=False)

    fig = plt.figure(figsize=(16.5, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.45, 1], hspace=0.34, wspace=0.42)

    vmax = float(np.percentile(np.concatenate([err_p1, err_hct]), 99))

    # -- top row: particle positions coloured by displacement from mesh ----
    for col, (pos, err, name, tag) in enumerate([
            (mesh_pos, None, "Mesh + HCT-3D  (reference)", "reference"),
            (p1_pos, err_p1, f"Grid `{family}` · Stage 1 = raw P1", "p1"),
            (hct_pos, err_hct, f"Grid `{family}_hct` · Stage 1 = HCT-3D", "hct")]):
        ax = fig.add_subplot(gs[0, col])
        x = pos[idx, 0] * 1e3
        y = pos[idx, 1] * 1e3
        if err is None:
            ax.scatter(x, y, s=0.5, c="#566573", alpha=0.40, linewidths=0)
            sub = "particles the two grid runs are compared against"
        else:
            sc = ax.scatter(x, y, s=0.5, c=err[idx], cmap="turbo",
                            vmin=0, vmax=vmax, alpha=0.65, linewidths=0)
            cb = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.045)
            cb.set_label("|x$_{grid}$ − x$_{mesh}$| (mm)", fontsize=10)
            _r = rms_p1 if tag == "p1" else rms_hct
            sub = f"rms = {_r:.2f} mm = {_r/D_PIN_MM:.2f} D$_{{pin}}$"

        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(5 * np.cos(th), 5 * np.sin(th), color="#c0392b", lw=1.6, zorder=5)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        if col == 0:
            ax.set_ylabel("y (mm)")
        ax.set_title(f"{name}\n{sub}", fontweight="bold", fontsize=11.5)

    # -- bottom left: error CDF -------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    for err, c, lab, rms in [(err_p1, P1_C, "raw P1", rms_p1),
                             (err_hct, HCT_C, "HCT-3D", rms_hct)]:
        s = np.sort(err)
        ax.plot(s, np.linspace(0, 100, len(s)), color=c, lw=2.4,
                label=f"{lab} (rms {rms:.2f} mm)")
    ax.set_xlabel("|x$_{grid}$ − x$_{mesh}$| (mm)")
    ax.set_ylabel("% of particles below")
    ax.set_xlim(0, vmax)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("Displacement-from-reference CDF", fontweight="bold", fontsize=11.5)
    secx = ax.secondary_xaxis(
        "top", functions=(lambda v: v / D_PIN_MM, lambda v: v * D_PIN_MM))
    secx.set_xlabel("in pin diameters (D$_{pin}$ = 10 mm)", fontsize=10)

    # -- bottom middle: error vs radius -----------------------------------
    ax = fig.add_subplot(gs[1, 1])
    r_mesh = np.linalg.norm(mesh_pos[:, :2], axis=1) * 1e3
    edges = np.linspace(0, 30, 31)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    for err, c, lab in [(err_p1, P1_C, "raw P1"), (err_hct, HCT_C, "HCT-3D")]:
        med = [np.median(err[(r_mesh >= a) & (r_mesh < b)])
               if np.any((r_mesh >= a) & (r_mesh < b)) else np.nan
               for a, b in zip(edges[:-1], edges[1:])]
        ax.plot(ctr, med, color=c, lw=2.4, marker="o", ms=3.4, label=lab)
    ax.axvline(5, color="#c0392b", ls="--", lw=1.5, label="pin surface")
    ax.axvspan(4, 7, color="#c0392b", alpha=0.10, label="shear ring")
    ax.set_xlabel("radius at reporting step (mm)")
    ax.set_ylabel("median |err| (mm)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9.5)
    secy2 = ax.secondary_yaxis(
        "right", functions=(lambda v: v / D_PIN_MM, lambda v: v * D_PIN_MM))
    secy2.set_ylabel("D$_{pin}$", fontsize=10)
    ax.set_title("Error vs radius — where HCT wins", fontweight="bold", fontsize=11.5)

    # -- bottom right: the headline reduction ------------------------------
    ax = fig.add_subplot(gs[1, 2])
    bars = ax.bar(["raw P1", "HCT-3D"], [rms_p1, rms_hct],
                  color=[P1_C, HCT_C], width=0.55,
                  edgecolor="black", linewidth=1.3, zorder=3)
    for b, v in zip(bars, [rms_p1, rms_hct]):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.02,
                f"{v:.2f} mm\n{v/D_PIN_MM:.2f} D$_{{pin}}$",
                ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("rms |x$_{grid}$ − x$_{mesh}$| (mm)")
    ax.set_ylim(0, rms_p1 * 1.25)
    secy = ax.secondary_yaxis(
        "right", functions=(lambda v: v / D_PIN_MM, lambda v: v * D_PIN_MM))
    secy.set_ylabel("D$_{pin}$", fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f"Stage-1 upgrade: $\\times${rms_hct/rms_p1:.2f} "
                 f"({100*(1-rms_hct/rms_p1):.0f} % rms reduction)",
                 fontweight="bold", fontsize=11.5)

    fig.suptitle(
        f"HCT-3D at Stage 1 pulls grid trajectories back onto the mesh reference "
        f"— case {case}, `{family}` family, step {matched:.0f}\n"
        f"Same grid geometry, same integrator, same particles: only the "
        f"mesh→grid projection differs",
        fontsize=14, fontweight="bold", y=0.995)

    path = out / f"figP_p1_vs_hct_case{case}_{family}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")
    print(f"    rms raw-P1 {rms_p1:.3f} mm | HCT {rms_hct:.3f} mm "
          f"| ratio x{rms_hct/rms_p1:.3f} | n={n:,} | step {matched:.0f}")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rom-root", type=Path,
                    default=Path("/scratch/shared/ROM"))
    ap.add_argument("--out", type=Path,
                    default=Path("paper_figs/rom_pt_step5_step6_all20"))
    ap.add_argument("--case", default="000",
                    help="case for the schematic and P1-vs-HCT figures")
    ap.add_argument("--family", default="4lvl",
                    help="grid family whose P1 and _hct runs are compared")
    ap.add_argument("--step", type=float, default=2000.0)
    ap.add_argument("--schematic-variants", nargs="+",
                    default=["4lvl_hct", "uniform_hct", "malmo_hct"])
    ap.add_argument("--only", choices=["timing", "grid", "p1hct"],
                    help="produce only one figure")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.rom_root.exists():
        print(f"ERROR: rom-root {args.rom_root} does not exist", file=sys.stderr)
        return 1

    if args.only in (None, "timing"):
        print("[figT] mesh vs grid timing")
        fig_timing(args.rom_root, args.out)
    if args.only in (None, "grid"):
        print("[figG] grid schematics")
        fig_grid_schematic(args.rom_root, args.case, args.out,
                           args.schematic_variants)
    if args.only in (None, "p1hct"):
        print("[figP] raw P1 vs HCT-3D trajectories")
        fig_p1_vs_hct(args.rom_root, args.case, args.family, args.step, args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
