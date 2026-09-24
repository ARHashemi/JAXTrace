#!/usr/bin/env python3
"""
Build the R1-g float-precision ablation table (methods x precision x tolerance).

Reads:
    malmo_runs_float_ablation/<mesh>__<variant>__{fp32,fp64}__tol{4,6,8}/result.json
        (produced by scripts/scone_bench/sweep_float_ablation.sh)

Emits (LaTeX + Markdown, one pair per mesh):
    paper_table_float_<mesh>.md
    paper_table_float_<mesh>.tex

Columns pivot: (variant, precision) rows x (tolerance) cols, cells report
`correct_rate_strict` (per-tet correct%) as the primary metric.  A
compact secondary block reports `found_rate` for completeness.

Usage:
    python3 scripts/scone_bench/build_float_ablation_table.py --mesh FSW_paper
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

VARIANT_ORDER = ["aabb", "centroid", "vertex_multi"]
PREC_ORDER = ["fp32", "fp64"]
TOL_ORDER = [4, 6, 8]  # exponents; label as 1e-4, 1e-6, 1e-8

VARIANT_LATEX = {
    "aabb":         r"MALMO$^{\mathrm{A}}$",
    "centroid":     r"MALMO$^{\mathrm{C}}$",
    "vertex_multi": r"MALMO$^{\mathrm{V}}$",
}
VARIANT_MD = {
    "aabb":         "MALMO$^\\mathrm{A}$",
    "centroid":     "MALMO$^\\mathrm{C}$",
    "vertex_multi": "MALMO$^\\mathrm{V}$",
}


def load(root: Path, mesh: str):
    """Return {(variant, prec, tol_exp): result_dict}."""
    out = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if not name.startswith(f"{mesh}__"):
            continue
        rest = name[len(mesh) + 2:]
        parts = rest.split("__")
        if len(parts) != 3:
            continue
        variant, prec, tol_str = parts
        if not tol_str.startswith("tol"):
            continue
        try:
            tol_exp = int(tol_str[3:])
        except ValueError:
            continue
        rj = d / "result.json"
        if not rj.exists():
            continue
        d_json = json.load(open(rj))
        out[(variant, prec, tol_exp)] = d_json
    return out


def _fmt_pct(v):
    if v is None:
        return "—"
    p = v * 100.0
    if p >= 99.99:
        return "100.00 %"
    return f"{p:.2f} %"


def emit_md(data, mesh: str, out_md: Path):
    lines = [f"# Table T7 · Float-precision ablation — {mesh}", ""]
    lines.append(f"Per-tet correctness (`correct_rate_strict`) of each MALMO "
                 f"variant on the {mesh} mesh across the "
                 f"{{float32, float64}} × {{1e-4, 1e-6, 1e-8}} PIT-tolerance "
                 f"design.  Sampling: in-mesh barycentric, "
                 f"$N_p = 100\\,000$ queries.  Rates ≥ 99.99 % (fewer than "
                 f"1 miss in 10 000) are rendered as 100.00 %.")
    lines.append("")

    header = ["variant", "precision"] + [f"$\\varepsilon = 10^{{-{e}}}$" for e in TOL_ORDER]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---", "---"] + ["---:"] * len(TOL_ORDER)) + "|")

    for variant in VARIANT_ORDER:
        for prec in PREC_ORDER:
            cells = [VARIANT_MD[variant], prec]
            for e in TOL_ORDER:
                d = data.get((variant, prec, e))
                if d is None:
                    cells.append("—")
                else:
                    cells.append(_fmt_pct(d.get("correct_rate_strict")))
            lines.append("| " + " | ".join(cells) + " |")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_md}")


def emit_tex(data, mesh: str, out_tex: Path):
    L = []
    L.append(f"% Table T7 (float-precision ablation, {mesh})")
    L.append(r"\begin{table}[t]")
    L.append(r"\centering\small")
    L.append(rf"\caption{{Float-precision ablation on the "
             rf"{mesh.replace('_', r' ')} mesh: per-tet correctness "
             rf"(\texttt{{correct\_rate\_strict}}) of each MALMO variant "
             rf"across the $\{{\text{{float32}}, \text{{float64}}\}}$ "
             rf"$\times$ $\{{10^{{-4}}, 10^{{-6}}, 10^{{-8}}\}}$ "
             rf"PIT-tolerance grid.  In-mesh barycentric sampling with "
             rf"$N_p = 100\,000$ queries.  Rates at or above $99.99\%$ "
             rf"(fewer than one miss in $10\,000$) are shown as "
             rf"$100.00\%$; smaller values are reported as measured, "
             rf"including the two-vertex floating-point misassignments "
             rf"already flagged by the daggers in the original Table~9.}}")
    L.append(rf"\label{{tab:r1g-float-{mesh.replace('_', '-')}}}")
    col_spec = "ll" + "r" * len(TOL_ORDER)
    L.append(rf"\begin{{tabular}}{{{col_spec}}}")
    L.append(r"\toprule")
    hdr = ["variant", "precision"] + [rf"$\varepsilon{{=}}10^{{-{e}}}$" for e in TOL_ORDER]
    L.append(" & ".join(hdr) + r" \\")
    L.append(r"\midrule")

    prev_variant = None
    for variant in VARIANT_ORDER:
        for prec in PREC_ORDER:
            if prev_variant and prev_variant != variant:
                L.append(r"\midrule")
            prev_variant = variant
            cells = [VARIANT_LATEX[variant], prec]
            for e in TOL_ORDER:
                d = data.get((variant, prec, e))
                if d is None:
                    cells.append("---")
                else:
                    v = d.get("correct_rate_strict")
                    if v is None:
                        cells.append("---")
                    elif v * 100 >= 99.99:
                        cells.append(r"$100.00\%$")
                    else:
                        cells.append(rf"${v*100:.2f}\%$")
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
    ap.add_argument("--sweep-root", default="malmo_runs_float_ablation")
    ap.add_argument("--out-dir", default=".")
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

    emit_md(data, a.mesh, out_dir / f"paper_table_float_{a.mesh}.md")
    emit_tex(data, a.mesh, out_dir / f"paper_table_float_{a.mesh}.tex")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
