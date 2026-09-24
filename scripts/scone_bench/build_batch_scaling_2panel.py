#!/usr/bin/env python3
"""
Two-panel batch-scaling figure for the R2-7 response (Fig A revised).

Emits ONE .pdf per panel, matching the paper's Figs 1 and 2 convention
(sec4 composes the (a)/(b) letters via LaTeX \subcaption{}, not inside
the plot).  The two panels are:

    Panel (a) -> fig_batch_scaling_panelA_FSW.pdf
                 FSW paper mesh (3.05 M tets, octree-aligned)
    Panel (b) -> fig_batch_scaling_panelB_Kim.pdf
                 FinalFuelPinPoly1560 (8 712 tets, general unstructured)

Both panels use the shared figstyle, log-log axes with explicit
"10 k / 50 k / ..." x-tick labels, and kq/s throughput units (positive
powers of 10 on the y-axis instead of a 10^-1 messy range).  A
per-panel legend sits in the middle-left region where no data line
crosses.

The LaTeX side (sec6) should compose the two panels with:

    \begin{figure}[!ht]
      \centering
      \begin{subfigure}[b]{0.47\textwidth}
        \subcaption{}
        \label{fig:batch_scaling_fsw}
        \includegraphics[width=\textwidth]{fig_batch_scaling_panelA_FSW.pdf}
      \end{subfigure}\hfill
      \begin{subfigure}[b]{0.47\textwidth}
        \subcaption{}
        \label{fig:batch_scaling_kim}
        \includegraphics[width=\textwidth]{fig_batch_scaling_panelB_Kim.pdf}
      \end{subfigure}
      \caption{...  (a) On the FSW mesh ... (b) On the Kim mesh ...}
      \label{fig:batch_scaling}
    \end{figure}

Reads:
    malmo_runs_batch_sweep/<mesh>__<variant>__np<N>/result.json

Writes:
    fig_batch_scaling_panelA_FSW.{pdf,png}
    fig_batch_scaling_panelB_Kim.{pdf,png}
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


VARIANT_ORDER = ["aabb", "centroid", "vertex_multi"]


def load(root: Path, mesh: str):
    """Return {variant: [(n_p, tput_qps), ...]}"""
    out = defaultdict(list)
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if not name.startswith(f"{mesh}__"):
            continue
        rest = name[len(mesh) + 2:]
        if "__np" not in rest:
            continue
        variant, np_str = rest.split("__np")
        try:
            n_p = int(np_str)
        except ValueError:
            continue
        rj = d / "result.json"
        if not rj.exists():
            continue
        d_json = json.load(open(rj))
        tput = d_json.get("throughput_queries_per_second")
        if tput is None:
            continue
        out[variant].append((n_p, tput))
    for v in out:
        out[v].sort()
    return out


def emit_panel(data, out_base: Path, show_legend: bool = True,
               legend_loc: str = "center left"):
    """Render one mesh's throughput-vs-Np curve as its own PDF/PNG.

    No panel-letter annotation is drawn inside the plot; the LaTeX
    subfigure environment adds "(a)" / "(b)" via \subcaption{}.
    Throughput is reported in kq/s so the y-axis tick labels are
    plain positive powers of 10 (e.g. 10^2 = 100 kq/s), rather than
    the messy 10^-1 range that Mq/s produces on our data.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "figstyle"))
    from figstyle import setup, framework_colour, save

    setup()

    variant_style = {
        "aabb":         dict(color=framework_colour["aabb"],         marker="o", label=r"MALMO$^{\rm A}$ (aabb)"),
        "centroid":     dict(color=framework_colour["centroid"],     marker="s", label=r"MALMO$^{\rm C}$ (centroid)"),
        "vertex_multi": dict(color=framework_colour["vertex_multi"], marker="^", label=r"MALMO$^{\rm V}$ (vertex-multi)"),
    }

    fig, ax = plt.subplots(figsize=(4.2, 3.2))

    for variant in VARIANT_ORDER:
        rows = data.get(variant, [])
        if not rows:
            continue
        xs = [r[0] for r in rows]
        ys = [r[1] / 1e3 for r in rows]     # convert qps -> kq/s
        st = variant_style[variant]
        ax.plot(xs, ys, **st)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Batch size $N_p$")
    ax.set_ylabel(r"Kernel-only throughput (kq/s)")

    xt = [10_000, 50_000, 100_000, 200_000, 500_000]
    xt_lab = ["10 k", "50 k", "100 k", "200 k", "500 k"]
    ax.set_xticks(xt)
    ax.set_xticklabels(xt_lab)
    ax.set_xticks([], minor=True)

    if show_legend:
        ax.legend(loc=legend_loc, framealpha=0.94, borderaxespad=0.7)

    fig.tight_layout()
    save(fig, out_base)
    plt.close(fig)



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root",
                    default="/home/arhashemi/fsw-gpu/flash/shared/jax/JAXTrace/malmo_runs_batch_sweep")
    ap.add_argument("--mesh-fsw", default="FSW_paper")
    ap.add_argument("--mesh-kim", default="FinalFuelPinPoly1560")
    ap.add_argument("--out-dir",
                    default="/home/arhashemi/Workspace/welding/JAXTrace",
                    help="Directory to write the two panel files")
    a = ap.parse_args()

    root = Path(a.sweep_root)
    data_fsw = load(root, a.mesh_fsw)
    data_kim = load(root, a.mesh_kim)
    if not data_fsw or not data_kim:
        print(f"ERROR: missing data (fsw: {len(data_fsw)}, kim: {len(data_kim)})")
        return 2

    out_dir = Path(a.out_dir)
    # Panel A: FSW.  Put the legend here (this panel has the widest y-range
    # spread between variants, so an inside legend is safe).
    emit_panel(data_fsw,
               out_dir / "fig_batch_scaling_panelA_FSW",
               show_legend=True, legend_loc="center left")
    # Panel B: Kim.  No legend (shared with panel A via the LaTeX caption).
    emit_panel(data_kim,
               out_dir / "fig_batch_scaling_panelB_Kim",
               show_legend=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
