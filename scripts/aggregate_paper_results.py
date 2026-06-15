#!/usr/bin/env python3
"""
aggregate_paper_results.py

Read the log files produced by run_paper_benchmarks.sh and emit a
markdown document shaped like the paper's Section 6 / Section 7
tables. Goal: drop-in source the user can pull table values from
when revising jaxtrace_restructured.tex.

The aggregator is intentionally tolerant: missing log files or
missing sections result in placeholder text in the markdown
("(section not in log)") rather than crashes, so a partial run
still yields a useful report.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------

# Each section is delimited by:
#   ============================
#   SECTION HEADER
#   ============================
# followed by a column header line and a delimiter, then data rows.
#
# We parse by chunking the log into blocks bounded by lines of "=" * N
# and then matching the header within each block.
EQ_LINE = re.compile(r"^=+\s*$")
DASH_LINE = re.compile(r"^-{20,}\s*$")


def _read_log(path: Path) -> List[str]:
    if not path.is_file():
        return []
    return path.read_text(errors="replace").splitlines()


def _split_blocks(lines: List[str]) -> List[Tuple[str, List[str]]]:
    """Split a log into (header, body_lines) blocks delimited by ==== lines.

    A block's body starts on the line AFTER the second ==== (i.e. after
    the section title), and ends at the next ==== line. The returned
    header is the line between two adjacent ==== lines (just below the
    opening delimiter).
    """
    blocks: List[Tuple[str, List[str]]] = []
    i = 0
    n = len(lines)
    while i < n:
        if EQ_LINE.match(lines[i]):
            # Look for a header line followed by another ==== line.
            j = i + 1
            while j < n and not EQ_LINE.match(lines[j]):
                j += 1
            if j >= n:
                break
            header = "\n".join(lines[i + 1 : j]).strip()
            # Body runs from j+1 until the next ==== line.
            k = j + 1
            while k < n and not EQ_LINE.match(lines[k]):
                k += 1
            body = lines[j + 1 : k]
            blocks.append((header, body))
            i = k
        else:
            i += 1
    return blocks


def _find_block(blocks: List[Tuple[str, List[str]]], needle: str) -> Optional[List[str]]:
    """First block whose header contains `needle` (case-insensitive)."""
    needle_low = needle.lower()
    for header, body in blocks:
        if needle_low in header.lower():
            return body
    return None


def _parse_table_rows(body: List[str], methods: List[str]) -> Dict[str, List[str]]:
    """Pick out rows whose first token matches one of the known method
    names. Returns method -> list of cell tokens (excluding the method
    name itself). Tokens are returned as raw strings so '%' or 'x'
    suffixes survive.
    """
    out: Dict[str, List[str]] = {}
    for raw in body:
        line = raw.rstrip()
        if not line.strip():
            continue
        # Match the longest known method name as a prefix (so e.g.
        # '3x3x3^PC' is preferred over '3x3x3').
        match: Optional[str] = None
        for m in sorted(methods, key=len, reverse=True):
            if line.lstrip().startswith(m):
                match = m
                break
        if match is None:
            continue
        # Tokenise the rest, splitting on whitespace.
        rest = line.lstrip()[len(match) :].split()
        out[match] = rest
    return out


# ---------------------------------------------------------------------------
# Method canonical names — match what benchmark_l2_accuracy.py prints
# and what the paper calls them.
# ---------------------------------------------------------------------------

ALL_METHODS = [
    # Paper name, log-script name, latex label
    ("Morton-linear w=5",   "radius r=2",   "Morton-linear $w{=}5$"),
    ("Morton-linear w=21",  "radius r=10",  "Morton-linear $w{=}21$"),
    ("MALMO_V 1x1x1",       "1x1x1",        "MALMO\\textsuperscript{V} $1{\\times}1{\\times}1$"),
    ("MALMO_V 3x3x3",       "3x3x3",        "MALMO\\textsuperscript{V} $3{\\times}3{\\times}3$"),
    ("MALMO_V 5x5x5",       "5x5x5",        "MALMO\\textsuperscript{V} $5{\\times}5{\\times}5$"),
    ("MALMO_C 3x3x3",       "3x3x3^PC",     "MALMO\\textsuperscript{C} $3{\\times}3{\\times}3$"),
    ("MALMO_AABB 3x3x3",    "3x3x3^AABB",   "MALMO\\textsuperscript{AABB} $3{\\times}3{\\times}3$"),
]

LOG_NAMES = [m[1] for m in ALL_METHODS]
PAPER_NAMES = {m[1]: m[0] for m in ALL_METHODS}


# ---------------------------------------------------------------------------
# Markdown emission helpers
# ---------------------------------------------------------------------------

def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    """Produce a GitHub-flavoured markdown table."""
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([head, sep, body])


def _section_or_placeholder(
    blocks, needle: str, label: str
) -> Tuple[Optional[List[str]], str]:
    body = _find_block(blocks, needle)
    if body is None:
        return None, f"_({label} not in log — block matching '{needle}' missing)_"
    return body, ""


# ---------------------------------------------------------------------------
# Per-section formatters
# ---------------------------------------------------------------------------

def _format_perturbation_table(
    blocks, needle: str, value_format: str = "{}"
) -> str:
    """Generic formatter for a section whose body is a per-method
    table with columns indexed by perturbation factors. We don't know
    the perturbation header values up front; we infer them from the
    column header line that benchmark_l2_accuracy.py emits.
    """
    body, placeholder = _section_or_placeholder(blocks, needle, needle)
    if body is None:
        return placeholder

    # Find the column header row (starts with 'Method')
    header_tokens: Optional[List[str]] = None
    for raw in body:
        if raw.lstrip().startswith("Method"):
            header_tokens = raw.split()
            break
    if header_tokens is None:
        return "_(could not find 'Method' header in section)_"
    col_labels = header_tokens[1:]  # skip 'Method'

    rows_by_method = _parse_table_rows(body, LOG_NAMES)
    out_rows: List[List[str]] = []
    for paper_name, log_name, _ in ALL_METHODS:
        if log_name not in rows_by_method:
            continue
        cells = rows_by_method[log_name]
        # The TIMING section uses a "min--max" composite per column, so
        # tokens-per-column may not be 1. To keep things robust, we just
        # rejoin all cells with " | " and let the user inspect.
        if len(cells) == len(col_labels):
            row = [paper_name] + [value_format.format(c) for c in cells]
        else:
            # Token count mismatched the header — likely a multi-token
            # value per column. Pack everything into a single cell.
            row = [paper_name, " ".join(cells)]
        out_rows.append(row)

    if not out_rows:
        return "_(no matching method rows found in section)_"
    headers = ["Method"] + col_labels
    # When the row width doesn't match headers (composite cells), shrink
    # the header to fit.
    if any(len(r) != len(headers) for r in out_rows):
        max_w = max(len(r) for r in out_rows)
        headers = headers[:max_w] + [""] * (max_w - len(headers))
    return _md_table(headers, out_rows)


def _format_performance_summary(blocks) -> str:
    body, placeholder = _section_or_placeholder(
        blocks, "PERFORMANCE:", "Performance summary"
    )
    if body is None:
        return placeholder

    rows_by_method = _parse_table_rows(body, LOG_NAMES)
    out_rows: List[List[str]] = []
    for paper_name, log_name, _ in ALL_METHODS:
        if log_name not in rows_by_method:
            continue
        cells = rows_by_method[log_name]
        # Expected: Queries/s, PIT tests/s, Mean time (s), Relative
        if len(cells) >= 4:
            out_rows.append([paper_name] + cells[:4])
        else:
            out_rows.append([paper_name] + cells + [""] * (4 - len(cells)))
    if not out_rows:
        return "_(no matching method rows found in performance section)_"
    return _md_table(
        ["Method", "Queries/s", "PIT tests/s", "Mean time (s)", "Relative"],
        out_rows,
    )


def _format_build_stats(blocks) -> str:
    body, placeholder = _section_or_placeholder(
        blocks, "PREPROCESSING / BUILD", "Build stats"
    )
    if body is None:
        return placeholder
    # Just dump the body verbatim — it's a multi-block summary that
    # doesn't fit a uniform table.
    return "```\n" + "\n".join(body).rstrip() + "\n```"


def _format_intra_summary(blocks) -> str:
    """Find the intra-element FOUND-rate block."""
    body = _find_block(blocks, "INTRA-ELEMENT")
    if body is None:
        return "_(INTRA-ELEMENT block not in log)_"
    # Inside the intra block, the script prints another "SUMMARY: Percentage
    # found by position type" sub-section. Try to find a per-method table.
    # The intra block emits one method-row per method with per-position columns.
    header_tokens: Optional[List[str]] = None
    for raw in body:
        if raw.lstrip().startswith("Method") or raw.lstrip().startswith("Position"):
            header_tokens = raw.split()
            break
    if header_tokens is None:
        return "```\n" + "\n".join(body).rstrip() + "\n```"
    col_labels = header_tokens[1:]
    rows_by_method = _parse_table_rows(body, LOG_NAMES)
    out_rows = []
    for paper_name, log_name, _ in ALL_METHODS:
        if log_name not in rows_by_method:
            continue
        cells = rows_by_method[log_name]
        if len(cells) == len(col_labels):
            row = [paper_name] + cells
        else:
            row = [paper_name, " ".join(cells)]
        out_rows.append(row)
    if not out_rows:
        return "```\n" + "\n".join(body).rstrip() + "\n```"
    return _md_table(["Method"] + col_labels, out_rows)


def _format_scalability(blocks) -> str:
    body = _find_block(blocks, "SCALABILITY")
    if body is None:
        return "_(SCALABILITY block not in log)_"
    # Pull every data-looking line and emit a markdown code block — the
    # exact column set varies a bit between runs.
    return "```\n" + "\n".join(body).rstrip() + "\n```"


def _format_level_distribution(blocks) -> str:
    body = _find_block(blocks, "LEVEL DISTRIBUTION")
    if body is None:
        return "_(LEVEL DISTRIBUTION block not in log)_"
    return "```\n" + "\n".join(body).rstrip() + "\n```"


def _format_data_structure_stats(blocks, log_lines: List[str]) -> str:
    """Try to recover the octree size/elems-per-cell numbers that the
    builder prints during stage [3/5]. These aren't inside a ==== block
    so we scan the log lines directly for the printed octree summaries.
    """
    pat = re.compile(
        r"^\s*(Vertex-multi octree|Parent-cube octree|AABB-overlap octree):\s*"
        r"(\d[\d,]*) cells,\s*([\d.]+) elem/cell"
        r"(?:\s*\(max\s*(\d+)\))?"
        r"(?:,\s*([\d.]+) cells/elem)?"
    )
    rows: List[List[str]] = []
    seen = set()
    for line in log_lines:
        m = pat.search(line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            name, n_cells, ec, max_e, cpe = m.groups()
            rows.append(
                [
                    name,
                    n_cells,
                    ec,
                    cpe or "1.0",
                    max_e or "—",
                ]
            )
    if not rows:
        return "_(data-structure stats not found in log)_"
    return _md_table(
        ["Registration", "Active cells", "elems/cell (mean)", "cells/elem (mean)", "max elems/cell"],
        rows,
    )


def _format_sec7_tracking(sec7_log: List[str]) -> str:
    """Extract the headline numbers from benchmark_femuss_comparison.py
    output: wall time, throughput, GPU memory, deviation stats. The
    paper currently quotes 'Tracking wall time', 'Throughput',
    'GPU memory'. We grep for those keys leniently.
    """
    if not sec7_log:
        return "_(sec7 log missing)_"
    text = "\n".join(sec7_log)

    def grab(pattern: str, default: str = "—") -> str:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        return m.group(1).strip() if m else default

    rows = [
        ["Tracking wall time (s)",
         grab(r"wall.?time[^\d]+([\d.,]+)\s*s")],
        ["Throughput (p·step/s)",
         grab(r"throughput[^\d]+([\d.,]+)")],
        ["GPU memory",
         grab(r"GPU memory[^\d]+([\d.,]+\s*[KMG]?B)")],
        ["Mean deviation (m)",
         grab(r"mean (?:deviation|error)[^\d]+([\d.\-eE]+)")],
        ["Max deviation (m)",
         grab(r"max (?:deviation|error)[^\d]+([\d.\-eE]+)")],
    ]
    return _md_table(["Metric", "Value"], rows)


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------

PAPER_DOC_TEMPLATE = """\
# RTX 5090 Benchmark Report

Generated: {now}

This document organizes new benchmark runs from the RTX 5090
workstation in the same structure as the paper's Section 6
(Numerical Validation) and Section 7 (Application). Use these
tables to update `sec6_validation.tex` and `sec7_application.tex`.

## Manifest

```json
{manifest}
```

---

## Section 6 — Numerical Validation

### 6.1  Data-structure statistics per registration strategy

Replaces the "Data-structure statistics" sub-block of Table 1 in
`sec6_validation.tex`. Morton-linear is omitted because it is a
flat element array (no cells); see `VALIDATION_UPDATE.md §1`.

{sec6_data_structure}

### 6.2.1  Found rate F(%) under perturbation

Replaces `tab:found_rate`. The MALMO\\textsuperscript{{AABB}} row,
previously "---" at σ ≥ 0.5, is now populated with actual numbers.

{sec6_found_rate}

### 6.2.2  Domain-interior search failures N_fail

Replaces `tab:search_failures`. Row order matches the paper.

{sec6_search_failures}

### 6.3  Intra-element found rate

Replaces `tab:intra_found`.

{sec6_intra}

### 6.4.1  Computational performance — N_p = 10,000

Replaces `tab:timing`. Wall-time min–max over 7 timed runs after
3 warm-ups.

{sec6_timing}

### 6.4.1bis  Performance summary (Queries/s, PIT/s, relative)

{sec6_performance}

### 6.4.1ter  Mean PIT tests per query

{sec6_mean_pit}

### 6.4.3  Throughput scaling

Replaces `tab:scalability`. Reported for MALMO\\textsuperscript{{C}}
$3\\times3\\times3$.

{sec6_scalability}

### 6.4.4  Resolving level distribution

Replaces `tab:level_dist`.

{sec6_level_dist}

### 6.4.5  Build cost and amortisation

Replaces the build-cost paragraph and Appendix `sec:app_build_cost`.
Per `VALIDATION_UPDATE.md §3`, verify whether JAX JIT warm-up is
included; cold vs. amortised should be separated.

{sec6_build_stats}

---

## Section 7 — Application

### 7.x  End-to-end tracking on cylA mesh

Replaces the headline-numbers table in `sec7_application.tex`. The
full per-step CSV and deviation maps are under
`{sec7_outdir}/` for plotting.

{sec7_tracking}

---

## Notes on the previous version

- VALIDATION_UPDATE.md §1: Morton-linear "cells" entries removed.
- §2: "(max 8)" annotation on MALMO\\textsuperscript{{C}} dropped — it
  was a stray copy of the elems-per-cell maximum.
- §3: Build times verified (see §6.4.5 above). Watch for JIT
  warm-up inclusion in those numbers.
- §3: MALMO\\textsuperscript{{AABB}} now evaluated at all σ levels;
  the table no longer has "---" entries at σ ≥ 0.5.
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate run_paper_benchmarks.sh output into a markdown report."
    )
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        print(f"ERROR: results dir not found: {results_dir}", file=sys.stderr)
        return 2

    sec6_log_path = results_dir / "sec6_l2_accuracy.log"
    sec7_log_path = results_dir / "sec7_femuss_comparison.log"
    sec7_outdir = results_dir / "sec7_femuss_comparison"
    manifest_path = results_dir / "manifest.json"

    sec6_lines = _read_log(sec6_log_path)
    sec7_lines = _read_log(sec7_log_path)
    sec6_blocks = _split_blocks(sec6_lines)

    if manifest_path.is_file():
        manifest_str = manifest_path.read_text()
    else:
        manifest_str = "(manifest.json not found)"

    doc = PAPER_DOC_TEMPLATE.format(
        now=datetime.now().isoformat(timespec="seconds"),
        manifest=manifest_str,
        sec6_data_structure=_format_data_structure_stats(sec6_blocks, sec6_lines),
        sec6_found_rate=_format_perturbation_table(
            sec6_blocks, "Percentage of particles FOUND"
        ),
        sec6_search_failures=_format_perturbation_table(
            sec6_blocks, "UNFOUND ANALYSIS"
        ),
        sec6_intra=_format_intra_summary(sec6_blocks),
        sec6_timing=_format_perturbation_table(sec6_blocks, "TIMING:"),
        sec6_performance=_format_performance_summary(sec6_blocks),
        sec6_mean_pit=_format_perturbation_table(sec6_blocks, "MEAN PIT TESTS PER QUERY"),
        sec6_scalability=_format_scalability(sec6_blocks),
        sec6_level_dist=_format_level_distribution(sec6_blocks),
        sec6_build_stats=_format_build_stats(sec6_blocks),
        sec7_tracking=_format_sec7_tracking(sec7_lines),
        sec7_outdir=sec7_outdir.relative_to(results_dir),
    )

    args.output.write_text(doc)
    print(f"[aggregate] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
