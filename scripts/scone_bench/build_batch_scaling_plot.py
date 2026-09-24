#!/usr/bin/env python3
"""
Build the R2-7 figure + extended Table 11 from the batch-size sweep.

Reads:
    malmo_runs_batch_sweep/<mesh>__<variant>__np<N>/result.json
        (produced by scripts/scone_bench/sweep_batch_size.sh)

Emits:
    fig_batch_scaling_<mesh>.pdf   log-log throughput vs N_p, one
                                    curve per variant, with a
                                    launch-latency / compute-bound
                                    transition annotation.
    paper_table_batch_<mesh>.tex   extended Table 11 (methods x N_p),
                                    LaTeX table environment with
                                    caption + label.
    paper_table_batch_<mesh>.md    same, Markdown.

Usage:
    python3 scripts/scone_bench/build_batch_scaling_plot.py --mesh FSW_paper
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


VARIANT_ORDER = ["aabb", "centroid", "vertex_multi"]
VARIANT_LABEL = {
    "aabb":         r"MALMO$^{\rm A}$ (aabb)",
    "centroid":     r"MALMO$^{\rm C}$ (centroid)",
    "vertex_multi": r"MALMO$^{\rm V}$ (vertex-multi)",
}
VARIANT_MD = {
    "aabb":         "MALMO$^\\mathrm{A}$",
    "centroid":     "MALMO$^\\mathrm{C}$",
    "vertex_multi": "MALMO$^\\mathrm{V}$",
}


def load(root: Path, mesh: str):
    """Return {variant: [(n_p, query_seconds, throughput_qps, rss_mb), ...]}"""
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
        ws = d_json.get("wall_seconds") or {}
        q_s = ws.get("query_min_of_3")
        tput = d_json.get("throughput_queries_per_second")
        rss = (d_json.get("process") or {}).get("max_rss_kb")
        rss_mb = (rss / 1024.0) if rss else None
        if q_s is None or tput is None:
            continue
        out[variant].append((n_p, q_s, tput, rss_mb))
    for v in out:
        out[v].sort()
    return out


def emit_plot(data, mesh: str, out_pdf: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "figstyle"))
    from figstyle import setup, framework_colour, place_legend, save

    setup()

    fig, ax = plt.subplots(figsize=(5.4, 3.4))

    variant_style = {
        "aabb":         dict(color=framework_colour["aabb"],         marker="o", label=r"MALMO$^{\rm A}$ (aabb)"),
        "centroid":     dict(color=framework_colour["centroid"],     marker="s", label=r"MALMO$^{\rm C}$ (centroid)"),
        "vertex_multi": dict(color=framework_colour["vertex_multi"], marker="^", label=r"MALMO$^{\rm V}$ (vertex-multi)"),
    }

    for variant in VARIANT_ORDER:
        rows = data.get(variant, [])
        if not rows:
            continue
        xs = [r[0] for r in rows]
        ys = [r[2] / 1e6 for r in rows]
        st = variant_style.get(variant, {})
        ax.plot(xs, ys, **st)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Batch size $N_p$")
    ax.set_ylabel(r"Kernel-only throughput (Mq/s)")

    # No title (LaTeX caption does that).
    # Auto-place legend in the emptiest corner so it never covers data.
    place_legend(ax)

    fig.tight_layout()
    save(fig, str(out_pdf).replace(".pdf", ""))
    plt.close(fig)


def emit_table_md(data, mesh: str, out_md: Path):
    # Column set = union of all N_p seen
    all_np = sorted({r[0] for rows in data.values() for r in rows})
    lines = [f"# Table T6 · Batch-size scaling (kernel-only), {mesh}", ""]
    lines.append(f"Per-variant query wall time (best of 3 trials, "
                 f"`jax.block_until_ready()` synchronised) on the "
                 f"{mesh} mesh.  Below $N_p \\approx 50\\,000$ the "
                 f"kernel is launch-latency dominated; above, the GPU "
                 f"transitions into the compute-bound regime where "
                 f"cross-variant comparisons are meaningful.")
    lines.append("")

    header = ["variant"] + [f"$N_p={n:,}$" for n in all_np]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * len(all_np)) + "|")

    for variant in VARIANT_ORDER:
        rows = data.get(variant, [])
        if not rows:
            continue
        by_np = {n: (q_s, tput, rss) for n, q_s, tput, rss in rows}
        cells = [VARIANT_MD[variant].replace("$", "$")]
        for n in all_np:
            if n in by_np:
                q_s, tput, rss = by_np[n]
                cells.append(f"{q_s*1000:.1f} ms")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("")
    lines.append("_Throughput (Mq/s) equivalent:_")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * len(all_np)) + "|")
    for variant in VARIANT_ORDER:
        rows = data.get(variant, [])
        if not rows:
            continue
        by_np = {n: (q_s, tput, rss) for n, q_s, tput, rss in rows}
        cells = [VARIANT_MD[variant].replace("$", "$")]
        for n in all_np:
            if n in by_np:
                _q, tput, _r = by_np[n]
                cells.append(f"{tput / 1e6:.2f} Mq/s")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_md}")


def emit_table_tex(data, mesh: str, out_tex: Path):
    all_np = sorted({r[0] for rows in data.values() for r in rows})

    L = []
    L.append(f"% Table T6 (batch-size scaling, {mesh})")
    L.append(r"\begin{table}[t]")
    L.append(r"\centering\small")
    L.append(rf"\caption{{Batch-size scaling on the {mesh.replace('_', r' ')} "
             rf"mesh, kernel-only-equivalent query wall time (best of 3 "
             rf"trials, \texttt{{jax.block\_until\_ready()}} synchronised) "
             rf"per MALMO variant.  Upper block: query wall time in "
             rf"milliseconds.  Lower block: throughput in "
             rf"$\text{{Mq}}/\text{{s}}$ (= $n_\text{{particles}}/t_\text{{query}}$).  "
             rf"Below $N_p \\approx 50\\,000$ the kernel is launch-latency "
             rf"dominated, so cross-variant comparisons in that regime are "
             rf"not meaningful; the reader should draw variant-ordering "
             rf"conclusions from the $N_p \\ge 100\\,000$ columns.}}")
    L.append(rf"\label{{tab:r27-batch-{mesh.replace('_', '-')}}}")
    col_spec = "l" + "r" * len(all_np)
    L.append(rf"\begin{{tabular}}{{{col_spec}}}")
    L.append(r"\toprule")
    hdr = ["variant"] + [rf"$N_p{{=}}{n//1000}{{,}}000$"
                          if n >= 1000 else rf"$N_p{{=}}{n}$"
                          for n in all_np]
    L.append(" & ".join(hdr) + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{" + str(len(all_np) + 1) +
             r"}{l}{\emph{query wall time (ms)}} \\")
    for variant in VARIANT_ORDER:
        rows = data.get(variant, [])
        if not rows:
            continue
        by_np = {n: (q_s, tput, rss) for n, q_s, tput, rss in rows}
        label = {
            "aabb":         r"MALMO$^{\mathrm{A}}$",
            "centroid":     r"MALMO$^{\mathrm{C}}$",
            "vertex_multi": r"MALMO$^{\mathrm{V}}$",
        }[variant]
        cells = [label]
        for n in all_np:
            if n in by_np:
                q_s, tput, _r = by_np[n]
                cells.append(rf"{q_s*1000:.1f}")
            else:
                cells.append("---")
        L.append(" & ".join(cells) + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{" + str(len(all_np) + 1) +
             r"}{l}{\emph{throughput (Mq/s)}} \\")
    for variant in VARIANT_ORDER:
        rows = data.get(variant, [])
        if not rows:
            continue
        by_np = {n: (q_s, tput, rss) for n, q_s, tput, rss in rows}
        label = {
            "aabb":         r"MALMO$^{\mathrm{A}}$",
            "centroid":     r"MALMO$^{\mathrm{C}}$",
            "vertex_multi": r"MALMO$^{\mathrm{V}}$",
        }[variant]
        cells = [label]
        for n in all_np:
            if n in by_np:
                _q, tput, _r = by_np[n]
                cells.append(rf"{tput/1e6:.2f}")
            else:
                cells.append("---")
        L.append(" & ".join(cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    out_tex.write_text("\n".join(L) + "\n")
    print(f"wrote {out_tex}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True,
                    help="Mesh tag, e.g. FSW_paper")
    ap.add_argument("--sweep-root",
                    default="malmo_runs_batch_sweep",
                    help="Directory containing <mesh>__<variant>__np<N>/ dirs")
    ap.add_argument("--out-dir", default=".",
                    help="Directory to write outputs")
    a = ap.parse_args()

    root = Path(a.sweep_root)
    if not root.is_dir():
        print(f"ERROR: sweep root not found: {root}")
        return 2
    data = load(root, a.mesh)
    if not data:
        print(f"ERROR: no runs found for mesh '{a.mesh}' under {root}")
        return 2

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emit_plot(data, a.mesh, out_dir / f"fig_batch_scaling_{a.mesh}.pdf")
    emit_table_md(data, a.mesh, out_dir / f"paper_table_batch_{a.mesh}.md")
    emit_table_tex(data, a.mesh, out_dir / f"paper_table_batch_{a.mesh}.tex")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
