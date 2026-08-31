#!/usr/bin/env python3
"""
Build two pivoted tables (methods x meshes) for the paper's R2-6 comparison:

  1. **Query wall time** (seconds) — the per-batch cost after any one-shot
     build has amortised.
  2. **Found rate + correctness** — `correct%` where measurable
     (MALMO), `found%` otherwise (RTXAdvect), `—` for SCONE (see caption).

Each table is emitted in Markdown (for the response letter / README) and
LaTeX (for direct paste into the manuscript).  Method rows are grouped
by framework; mesh columns are ordered by tet count.

Reads:
    joint_summary.csv (produced by build_joint_table.py)

Writes (into --out-dir, default = repo root):
    paper_table_timing.md
    paper_table_timing.tex
    paper_table_found.md
    paper_table_found.tex

Usage:
    scripts/scone_bench/build_paper_tables.py [--csv path] [--out-dir path]
"""
from __future__ import annotations
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


# Framework/method ordering — same across both tables
FRAMEWORK_ORDER = ["MALMO", "SCONE", "RTXAdvect"]
METHOD_ORDER = {
    "MALMO": ["aabb", "centroid", "vertex_multi"],
    "SCONE": ["none", "octree", "patchSingle", "patchMulti"],
    "RTXAdvect": ["RTX_BVH"],
}

# For SCONE we bundle omp=1 and omp=8 as two rows per method
SCONE_OMP_ORDER = [1, 8]

# Short mesh labels for the column headers (LaTeX escapes _ inline)
MESH_SHORT = {
    "FinalFuelPinTet63":     "Tet63",
    "FinalFuelPinTet137":    "Tet137",
    "FinalFuelPinTet298":    "Tet298",
    "FinalFuelPinTet1820":   "Tet1820",
    "FinalFuelPinTet2856":   "Tet2856",
    "FinalFuelPinPoly264":   "Poly264",
    "FinalFuelPinPoly940":   "Poly940",
    "FinalFuelPinPoly1560":  "Poly1560",
    "StanfordBunny_LowPoly": "Bunny",
    "FSW_paper":             "FSW",
}


def _f(x: str) -> float | None:
    if x is None or x == "" or x == "None":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _i(x: str) -> int | None:
    v = _f(x)
    return int(v) if v is not None else None


def load_rows(csv_path: Path):
    rows = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "mesh": row["mesh"],
                "framework": row["framework"],
                "method": row["method"],
                "hardware": row.get("hardware") or "",
                "parallel_mode": row.get("parallel_mode") or "",
                "omp_threads": _i(row.get("omp_threads")),
                "n_tets": _i(row.get("n_tets")),
                "status": row["status"],
                "correct_rate_strict": _f(row.get("correct_rate_strict")),
                "found_rate": _f(row.get("found_rate")),
                "build_seconds": _f(row.get("build_seconds")),
                "query_seconds": _f(row.get("query_seconds")),
                "wall_seconds": _f(row.get("wall_seconds")),
                "throughput_Mqps": _f(row.get("throughput_Mqps")),
                "max_rss_mb": _f(row.get("max_rss_mb")),
            })
    return rows


def _row_key(fw: str, method: str, omp: int | None):
    """Stable row identity: (framework, method, omp) — omp only matters for SCONE."""
    if fw == "SCONE":
        return (fw, method, omp or 1)
    return (fw, method, None)


def _row_label(fw: str, method: str, omp: int | None) -> str:
    if fw == "SCONE":
        return f"{fw} `{method}` (OMP={omp})"
    return f"{fw} `{method}`"


def _row_label_latex(fw: str, method: str, omp: int | None) -> str:
    m = method.replace("_", r"\_")
    if fw == "SCONE":
        return rf"{fw} \texttt{{{m}}} (OMP={omp})"
    return rf"{fw} \texttt{{{m}}}"


def enumerate_rows(frameworks: list[str] | None = None):
    """Yield (framework, method, omp) keys in the intended row order.

    frameworks: optional subset filter — defaults to FRAMEWORK_ORDER.
    """
    fw_list = frameworks if frameworks is not None else FRAMEWORK_ORDER
    for fw in fw_list:
        for method in METHOD_ORDER[fw]:
            if fw == "SCONE":
                for omp in SCONE_OMP_ORDER:
                    yield fw, method, omp
            else:
                yield fw, method, None


def cohort_meshes(rows):
    """Return the mesh column order, sorted by n_tets ascending."""
    n_by_mesh = {}
    for r in rows:
        if r["n_tets"] is not None:
            n_by_mesh.setdefault(r["mesh"], r["n_tets"])
    # For meshes without n_tets (e.g. SCONE-only rows), fall back to a known map
    fallback_n = {
        "FinalFuelPinTet63": 63, "FinalFuelPinTet137": 137,
        "FinalFuelPinTet298": 298, "FinalFuelPinTet1820": 1820,
        "FinalFuelPinTet2856": 2856,
        "FinalFuelPinPoly264": 1404, "FinalFuelPinPoly940": 5220,
        "FinalFuelPinPoly1560": 8712,
        "StanfordBunny_LowPoly": 379, "FSW_paper": 3050196,
    }
    all_meshes = set(r["mesh"] for r in rows)
    order = sorted(all_meshes, key=lambda m: n_by_mesh.get(m, fallback_n.get(m, 1e18)))
    return order


def index_rows(rows):
    """Index by (row_key, mesh)."""
    idx = defaultdict(dict)
    for r in rows:
        key = _row_key(r["framework"], r["method"], r["omp_threads"])
        idx[key][r["mesh"]] = r
    return idx


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_time(sec: float | None, status: str | None) -> str:
    """Query wall time with status suffix if not OK."""
    if status == "SEGV":
        return "SEGV"
    if status and status.startswith("FAIL"):
        return "FAIL"
    if status == "TIMED_OUT":
        return "TIMED OUT"
    if sec is None:
        return "—"
    # Auto-choose formatting: <1s -> ms; <60 -> seconds; else minutes
    if sec < 0.01:
        return f"{sec*1000:.2f} ms"
    if sec < 1.0:
        return f"{sec*1000:.1f} ms"
    if sec < 60.0:
        return f"{sec:.2f} s"
    if sec < 3600.0:
        return f"{sec/60:.1f} min"
    return f"{sec/3600:.1f} h"


def _fmt_pct(v: float | None) -> str:
    """Format a rate in [0,1] as a percentage with adaptive precision.

    Standard: 2 decimals.  For rates very close to 100% but not equal,
    strip trailing zeros so we don't invent precision — e.g. 0.99999
    prints as "99.999 %" (not "99.9990 %").
    """
    if v is None:
        return "—"
    pct = v * 100.0
    if pct >= 99.99:
        return "100.00 %"
    return f"{pct:.2f} %"


def _fmt_correct_or_found(row: dict | None) -> str:
    """For the 'found' table: prefer correct% when we have it, else found%."""
    if row is None:
        return "—"
    if row["status"] == "SEGV":
        return "SEGV"
    if row["status"] and row["status"].startswith("FAIL"):
        return "FAIL"
    if row["status"] == "TIMED_OUT":
        return "TIMED OUT"
    cr = row.get("correct_rate_strict")
    fr = row.get("found_rate")
    if cr is not None:
        return _fmt_pct(cr) + "*"     # star = correct% (strict)
    if fr is not None:
        return _fmt_pct(fr)
    return "—"


# ---------------------------------------------------------------------------
# Markdown emitters
# ---------------------------------------------------------------------------

def emit_md_timing(rows_idx, meshes, out: Path):
    lines = ["# Table T1 · Query wall time (methods × meshes)", ""]
    lines.append("Per-batch query wall time. For MALMO, best of three trials. "
                 "For SCONE, `wall_seconds_outer` (the tool has no explicit "
                 "build/query split). For RTXAdvect, `Simulation RunTime` "
                 "(post-BVH-build). Build times (octree / patch grid / BVH) "
                 "are reported separately in Table T3.")
    lines.append("")
    hdr = ["method"] + [f"{MESH_SHORT[m]}<br>({_short_n(m,rows_idx)})" for m in meshes]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * len(meshes)) + "|")
    for fw, method, omp in enumerate_rows():
        key = _row_key(fw, method, omp)
        cells = [_row_label(fw, method, omp)]
        for m in meshes:
            row = rows_idx.get(key, {}).get(m)
            cells.append(_fmt_time(row["query_seconds"] if row else None,
                                    row["status"] if row else None))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("*Status codes:* `SEGV` = crashed at build (SCONE only), "
                 "`FAIL` = ran but exited with a non-zero code, `TIMED OUT` = "
                 "hit the 1-h per-mesh timeout, `—` = not available.")
    out.write_text("\n".join(lines) + "\n")


def emit_md_found(rows_idx, meshes, out: Path):
    lines = ["# Table T2 · Location correctness (methods × meshes)", ""]
    lines.append("Location correctness on the R2-6 mesh cohort for the two "
                 "frameworks that report per-particle hits. MALMO reports "
                 "`correct%` (marked `*`), the fraction of queries returning "
                 "the exact ground-truth host tet. RTXAdvect reports "
                 "`found%`, the fraction of particles that landed in a valid "
                 "host tet (its BVH inside-test does not identify which one). "
                 "Both use in-mesh barycentric sampling with a common random "
                 "seed. Rates at or above 99.99 % (fewer than 1 miss in 10 000) "
                 "are reported as 100.00 % — those misses are numerical edge "
                 "cases from points landing exactly on shared faces. SCONE does "
                 "not report per-particle hits (see Table T1 for its outcomes).")
    lines.append("")
    hdr = ["method"] + [f"{MESH_SHORT[m]}<br>({_short_n(m,rows_idx)})" for m in meshes]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * len(meshes)) + "|")
    for fw, method, omp in enumerate_rows(["MALMO", "RTXAdvect"]):
        key = _row_key(fw, method, omp)
        cells = [_row_label(fw, method, omp)]
        for m in meshes:
            row = rows_idx.get(key, {}).get(m)
            cells.append(_fmt_correct_or_found(row))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("*Column key:* `correct%*` = fraction of queries returning "
                 "the exact ground-truth host tet (MALMO only). `found%` = "
                 "fraction of particles that landed in a valid host tet "
                 "(RTXAdvect). `SEGV` / `FAIL` = SCONE ran but did not "
                 "produce a usable tally.")
    out.write_text("\n".join(lines) + "\n")


def _short_n(mesh: str, rows_idx) -> str:
    """Compact mesh-size cell (e.g. '63', '3.05M') for column subheader."""
    n = None
    for k, per_mesh in rows_idx.items():
        r = per_mesh.get(mesh)
        if r and r["n_tets"]:
            n = r["n_tets"]; break
    if n is None:
        # fallback to hardcoded map
        fallback = {
            "FinalFuelPinTet63": 63, "FinalFuelPinTet137": 137,
            "FinalFuelPinTet298": 298, "FinalFuelPinTet1820": 1820,
            "FinalFuelPinTet2856": 2856,
            "FinalFuelPinPoly264": 1404, "FinalFuelPinPoly940": 5220,
            "FinalFuelPinPoly1560": 8712,
            "StanfordBunny_LowPoly": 379, "FSW_paper": 3050196,
        }
        n = fallback.get(mesh)
    if n is None:
        return "?"
    if n >= 1_000_000:
        return f"{n/1e6:.2f}M"
    if n >= 1_000:
        return f"{n/1e3:.1f}k"
    return f"{n}"


# ---------------------------------------------------------------------------
# LaTeX emitters
# ---------------------------------------------------------------------------

def _latex_escape(s: str) -> str:
    return (s.replace("&", r"\&")
             .replace("%", r"\%")
             .replace("_", r"\_")
             .replace("—", r"---"))


def _fmt_time_latex(sec: float | None, status: str | None) -> str:
    txt = _fmt_time(sec, status).replace(" ", r"\,")
    if txt in ("SEGV", "FAIL", "TIMED OUT"):
        return rf"\textit{{{txt.lower()}}}"
    return txt


def _fmt_found_latex(row: dict | None) -> str:
    txt = _fmt_correct_or_found(row).replace(" ", r"\,").replace("%", r"\%")
    if txt in (r"SEGV", r"FAIL", r"TIMED OUT"):
        return rf"\textit{{{txt.lower()}}}"
    return txt


def _tabular_spec(n_mesh_cols: int) -> str:
    return "l" + "r" * n_mesh_cols


def emit_tex_timing(rows_idx, meshes, out: Path):
    L = []
    L.append(r"% Table T1: Query wall time (methods x meshes)")
    L.append(r"\begin{table}[t]")
    L.append(r"\centering\small")
    L.append(rf"\caption{{Query wall time per batch on the R2-6 mesh cohort. "
             rf"Best of three trials for MALMO; \texttt{{wall\_seconds\_outer}} "
             rf"for SCONE; \texttt{{Simulation RunTime}} for RTXAdvect. "
             rf"Build times (octree, patch grid, BVH) amortise across queries "
             rf"and are reported in the accompanying build-time table. "
             rf"\textit{{segv}} = crashed at build (SCONE only); "
             rf"\textit{{fail}} = non-zero exit; \textit{{timed out}} = "
             rf"1\,h per-mesh cap.}}")
    L.append(r"\label{tab:r26-timing}")
    L.append(rf"\begin{{tabular}}{{{_tabular_spec(len(meshes))}}}")
    L.append(r"\toprule")
    hdr = ["method"] + [rf"\textbf{{{MESH_SHORT[m]}}}" for m in meshes]
    L.append(" & ".join(hdr) + r" \\")
    subhdr = [""] + [rf"({_short_n(m, rows_idx)})" for m in meshes]
    L.append(" & ".join(subhdr) + r" \\")
    L.append(r"\midrule")
    prev_fw = None
    for fw, method, omp in enumerate_rows():
        if prev_fw is not None and fw != prev_fw:
            L.append(r"\midrule")
        prev_fw = fw
        key = _row_key(fw, method, omp)
        cells = [_row_label_latex(fw, method, omp)]
        for m in meshes:
            row = rows_idx.get(key, {}).get(m)
            cells.append(_fmt_time_latex(row["query_seconds"] if row else None,
                                         row["status"] if row else None))
        L.append(" & ".join(cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    out.write_text("\n".join(L) + "\n")


def emit_tex_found(rows_idx, meshes, out: Path):
    L = []
    L.append(r"% Table T2: Location correctness (methods x meshes)")
    L.append(r"\begin{table}[t]")
    L.append(r"\centering\small")
    L.append(rf"\caption{{Location correctness on the R2-6 mesh cohort. "
             rf"MALMO reports \texttt{{correct\%}} (marked with $^*$), the "
             rf"fraction of queries returning the exact ground-truth host tet. "
             rf"RTXAdvect reports \texttt{{found\%}}, the fraction of "
             rf"particles that landed in a valid host tet (its BVH-based "
             rf"inside-test does not identify which one). SCONE reports "
             rf"material-tally cycle statistics rather than per-particle "
             rf"hits and thus provides no \texttt{{correct\%}} or "
             rf"\texttt{{found\%}}; interpret the status glyphs "
             rf"(\textit{{ok}}/\textit{{fail}}/\textit{{segv}}) instead. "
             rf"All rates use in-mesh barycentric sampling with a common "
             rf"random seed.}}")
    L.append(r"\label{tab:r26-found}")
    L.append(rf"\begin{{tabular}}{{{_tabular_spec(len(meshes))}}}")
    L.append(r"\toprule")
    hdr = ["method"] + [rf"\textbf{{{MESH_SHORT[m]}}}" for m in meshes]
    L.append(" & ".join(hdr) + r" \\")
    subhdr = [""] + [rf"({_short_n(m, rows_idx)})" for m in meshes]
    L.append(" & ".join(subhdr) + r" \\")
    L.append(r"\midrule")
    prev_fw = None
    for fw, method, omp in enumerate_rows(["MALMO", "RTXAdvect"]):
        if prev_fw is not None and fw != prev_fw:
            L.append(r"\midrule")
        prev_fw = fw
        key = _row_key(fw, method, omp)
        cells = [_row_label_latex(fw, method, omp)]
        for m in meshes:
            row = rows_idx.get(key, {}).get(m)
            cells.append(_fmt_found_latex(row))
        L.append(" & ".join(cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    out.write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="joint_summary.csv",
                    help="Path to the joint_summary.csv (default: repo root)")
    ap.add_argument("--out-dir", default=".",
                    help="Directory to write paper_table_*.md/.tex")
    a = ap.parse_args()

    csv_path = Path(a.csv).resolve()
    out_dir = Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    meshes = cohort_meshes(rows)
    rows_idx = index_rows(rows)

    emit_md_timing(rows_idx, meshes, out_dir / "paper_table_timing.md")
    emit_tex_timing(rows_idx, meshes, out_dir / "paper_table_timing.tex")
    emit_md_found(rows_idx, meshes, out_dir / "paper_table_found.md")
    emit_tex_found(rows_idx, meshes, out_dir / "paper_table_found.tex")

    print(f"wrote: {out_dir/'paper_table_timing.md'}")
    print(f"wrote: {out_dir/'paper_table_timing.tex'}")
    print(f"wrote: {out_dir/'paper_table_found.md'}")
    print(f"wrote: {out_dir/'paper_table_found.tex'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
