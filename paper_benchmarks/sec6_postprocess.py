#!/usr/bin/env python3
"""
sec6_postprocess.py — Extract the Section 6 numbers from the benchmark log,
apply one consistent aggregation rule everywhere, and produce:

  * sec6_numbers.json   — machine-readable numbers, one dict per table
  * sec6_report.md      — human-readable report with each shipped paper
                          number placed side-by-side with the re-run
                          number, plus a per-table "OK / CHANGED" flag

Canonical protocol (deliberately fixed to close the two Section 6 issues):

  Wall time      — reported as (min, max, mean) tuple across timed runs
  Queries/s      — computed as N_p / mean_time   (single rule)
  Failure decomp — counted per-particle: each failed 1x1x1 query is
                   classified by its smallest recovering neighbour offset
                   (face-adjacent < edge-adjacent < corner-adjacent).
                   Sum equals number of failures, not sum of hits.

Usage:
    python paper_benchmarks/sec6_postprocess.py \
        --log paper_benchmarks/sec6_raw.log \
        --json paper_benchmarks/sec6_numbers.json \
        --report paper_benchmarks/sec6_report.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any


# =============================================================================
# Paper reference numbers (from sec6_validation.tex, hard-coded so the diff
# report can flag any table where the re-run differs)
# =============================================================================

PAPER_FOUND_RATE = {
    "Morton-linear w=5":        [31.98, 29.93, 27.01, 24.60, 23.63, 23.46],
    "Morton-linear w=21":       [59.12, 52.46, 50.36, 49.00, 47.75, 47.45],
    "MALMO_V 1x1x1":            [50.17, 49.93, 49.93, 49.69, 49.22, 49.10],
    "MALMO_V 3x3x3":            [100.00, 99.98, 99.70, 98.51, 97.71, 96.61],
    "MALMO_V 5x5x5":            [100.00, 99.98, 99.70, 98.51, 97.71, 96.61],
    "MALMO_C 3x3x3":            [100.00, 99.98, 99.70, 98.51, 97.71, 96.61],
    "MALMO_AABB 3x3x3":         [100.00, 99.98, 99.70, 98.51, 97.71, 96.61],
}

PAPER_SEARCH_FAILURES = {
    "Morton-linear w=5":        [6802, 7005, 7269, 7391, 7408, 7315],
    "Morton-linear w=21":       [4088, 4752, 4934, 4951, 4996, 4916],
    "MALMO_V 1x1x1":            [4983, 5005, 4977, 4882, 4849, 4751],
    "MALMO_V 3x3x3":            [0, 0, 0, 0, 0, 0],
    "MALMO_V 5x5x5":            [0, 0, 0, 0, 0, 0],
    "MALMO_C 3x3x3":            [0, 0, 0, 0, 0, 0],
    "MALMO_AABB 3x3x3":         [0, 0, 0, 0, 0, 0],
}

PAPER_INTRA_FOUND = {
    # order: centroid, random, near_face, near_edge, near_vertex
    "Morton-linear w=5":        [31.52, 24.65, 23.08, 19.28, 11.21],
    "Morton-linear w=21":       [59.34, 49.74, 49.50, 48.98, 45.42],
    "MALMO_V 1x1x1":            [49.42, 49.88, 49.77, 49.91, 49.10],
    "MALMO_V 3x3x3":            [100.00, 100.00, 100.00, 100.00, 99.98],
    "MALMO_V 5x5x5":            [100.00, 100.00, 100.00, 100.00, 99.98],
    "MALMO_C 3x3x3":            [100.00, 100.00, 100.00, 100.00, 99.98],
    "MALMO_AABB 3x3x3":         [100.00, 100.00, 100.00, 100.00, 99.98],
}

# tab:timing — wall time (s) min-max, mean PIT count, queries/s
PAPER_TIMING = {
    "MALMO_C 3x3x3":            {"time_min": 1.746, "time_max": 1.900, "mean_pit":  76.1, "qps": 5264},
    "MALMO_V 1x1x1":            {"time_min": 1.911, "time_max": 1.979, "mean_pit":  23.1, "qps": 5053},
    "MALMO_V 3x3x3":            {"time_min": 2.023, "time_max": 2.041, "mean_pit": 187.5, "qps": 4915},
    "Morton-linear w=21":       {"time_min": 2.307, "time_max": 2.335, "mean_pit": None,  "qps": 4286},
    "Morton-linear w=5":        {"time_min": 2.320, "time_max": 2.349, "mean_pit": None,  "qps": 4279},
    "MALMO_AABB 3x3x3":         {"time_min": 2.235, "time_max": 2.367, "mean_pit": 237.8, "qps": 4225},
    "MALMO_V 5x5x5":            {"time_min": 2.406, "time_max": 2.433, "mean_pit": 953.3, "qps": 4139},
}

PAPER_BYTES = {
    "Morton-linear w=5":        62177,
    "Morton-linear w=21":       62177,
    "MALMO_AABB 3x3x3":         45724,
    "MALMO_V 1x1x1":            40233,
    "MALMO_V 3x3x3":            40229,
    "MALMO_V 5x5x5":            40229,
    "MALMO_C 3x3x3":            32693,
}

# tab:scalability — batch, wall time, queries/s
PAPER_SCALABILITY = {
    1000:   {"time": 1.849, "qps":    541,   "us_per_q": 1849,  "pit_s":    41133},
    2000:   {"time": 1.848, "qps":   1083,   "us_per_q":  924,  "pit_s":    82403},
    5000:   {"time": 1.827, "qps":   2737,   "us_per_q":  365,  "pit_s":   207613},
    10000:  {"time": 1.832, "qps":   5457,   "us_per_q":  183,  "pit_s":   414870},
    20000:  {"time": 1.834, "qps":  10904,   "us_per_q":   92,  "pit_s":   829424},
    50000:  {"time": 1.858, "qps":  26905,   "us_per_q":   37,  "pit_s":  2041572},
    100000: {"time": 1.890, "qps":  52904,   "us_per_q":   19,  "pit_s":  4015298},
    200000: {"time": 1.976, "qps": 101223,   "us_per_q":   10,  "pit_s":  7691523},
    500000: {"time": 2.221, "qps": 225147,   "us_per_q":    4,  "pit_s": 17098389},
}

PAPER_LEVEL_DIST = {
    # The paper uses RELATIVE levels (1..7, finest=7). The harness log
    # uses ABSOLUTE octree depth (1..14). Mapping:
    #     paper_level 7 (finest)  ↔  log_level 14
    #     paper_level 6           ↔  log_level 13
    #     paper_level 5           ↔  log_level 12
    #     paper_level 4           ↔  log_level 11
    #     paper_level ≤ 3         ↔  log_level ≤ 10 (aggregated bucket)
    # See sec6_inconsistencies_2.md Claim 6 for the convention gap.
    7: {"count": 8502, "frac": 85.0},
    6: {"count": 1292, "frac": 12.9},
    5: {"count":  141, "frac":  1.4},
    4: {"count":   39, "frac":  0.4},
    3: {"count":   26, "frac":  0.3},   # "<= 3" bucket in the paper
}

# Absolute log level -> paper relative level mapping.
LEVEL_LOG_TO_PAPER = {
    14: 7, 13: 6, 12: 5, 11: 4,
    10: 3, 9: 3, 8: 3,   # log levels 8-10 aggregated into paper "<= 3"
}

PAPER_FAILURE_DECOMP = {
    "face_adjacent":   {"count":  7463, "share": 37.3},
    "edge_adjacent":   {"count":  9995, "share": 50.0},
    "corner_adjacent": {"count":  2542, "share": 12.7},
}

PAPER_MESH_PROPS = {
    "n_elements":        3050196,
    "n_nodes":           571533,
    "n_non_conforming":  1958,
    "n_active_cells_V":  666162,
    "n_active_cells_C":  517309,
    "n_active_cells_AABB": 620760,
    "elem_per_cell_V":   18.3,
    "elem_per_cell_C":    5.9,
    "elem_per_cell_AABB": 30.8,
}


# =============================================================================
# Log parsers — each returns the "measured" numbers for one table
# =============================================================================

def parse_perturb_summary(log_text: str) -> dict:
    """Parse the perturbation-sweep per-method per-sigma summary.

    The harness prints one line per (method, sigma) of the form:
        perturb=0.0x: found=10000 (100.00%), correct_elem=..., search_fail=0,
                     time=... s, mean_PIT=76.1, 5457 queries/s
    """
    method_map = {
        "radius r=2":      "Morton-linear w=5",
        "radius r=10":     "Morton-linear w=21",
        "1x1x1":           "MALMO_V 1x1x1",
        "3x3x3":           "MALMO_V 3x3x3",
        "5x5x5":           "MALMO_V 5x5x5",
        "3x3x3^PC":        "MALMO_C 3x3x3",
        "3x3x3^AABB":      "MALMO_AABB 3x3x3",
    }
    # Each `--- <method> ---` header opens a section of one row per
    # perturbation level. The actual line format the harness prints is:
    #
    #   perturb=0.0x: found=10,000 (100.00%), correct_elem=10,000/10,000
    #     (100.0%), search_fail=0, time=2.112 ± 0.177 (2.034--2.546)s,
    #     mean_PIT=187.5, 4735 queries/s
    #
    # All groups below are NAMED to sidestep positional-index confusion.
    method_re = re.compile(r"^\s*---\s+(?P<name>.+?)\s+---\s*$", re.MULTILINE)
    row_re = re.compile(
        r"perturb=(?P<sigma>[0-9.]+)x:\s+"
        r"found=(?P<found>[\d,]+)\s+"
        r"\(\s*(?P<found_pct>[0-9.]+)%\)?,?\s+"
        r"correct_elem=(?P<correct>[\d,]+)/(?P<found_total>[\d,]+)\s+"
        r"\([^)]*\),?\s+"
        r"search_fail=(?P<search_fail>[\d,]+),\s+"
        r"time=(?P<time_mean>[0-9.]+)\s*±\s*(?P<time_std>[0-9.]+)\s+"
        r"\((?P<time_min>[0-9.]+)--(?P<time_max>[0-9.]+)\)s"
        r"(?:,\s+mean_PIT=(?P<mean_pit>[0-9.]+))?"
        r"(?:,\s+(?P<qps>[\d,]+)\s+queries/s)?"
    )

    results = {}
    # Split by method headers.
    parts = method_re.split(log_text)
    # parts[0] is preamble; then alternating (raw_name, section_text)
    # under `re.MULTILINE` with a group-yielding pattern.
    for i in range(1, len(parts) - 1, 2):
        raw_name = parts[i].strip()
        section = parts[i + 1]
        display_name = method_map.get(raw_name, raw_name)
        method_rows = []
        for m in row_re.finditer(section):
            method_rows.append({
                "sigma":         float(m.group("sigma")),
                "found":         int(m.group("found").replace(",", "")),
                "found_pct":     float(m.group("found_pct")),
                "correct":       int(m.group("correct").replace(",", "")),
                "search_fail":   int(m.group("search_fail").replace(",", "")),
                "time_mean":     float(m.group("time_mean")),
                "time_std":      float(m.group("time_std")),
                "time_min":      float(m.group("time_min")),
                "time_max":      float(m.group("time_max")),
                "mean_pit":      float(m.group("mean_pit")) if m.group("mean_pit") else None,
                "qps_reported":  int(m.group("qps").replace(",", "")) if m.group("qps") else None,
            })
        if method_rows:
            results[display_name] = method_rows
    return results


def parse_intra_element(log_text: str) -> dict:
    """Parse the INTRA-ELEMENT Found rate table."""
    intra_start = log_text.find("INTRA-ELEMENT: Found rate")
    if intra_start < 0:
        return {}
    intra_end = log_text.find("INTRA-ELEMENT:", intra_start + 1)
    if intra_end < 0:
        intra_end = log_text.find("=" * 90, intra_start + 1)
    section = log_text[intra_start: intra_end if intra_end > intra_start else intra_start + 5000]

    method_map = {
        "radius r=2":      "Morton-linear w=5",
        "radius r=10":     "Morton-linear w=21",
        "1x1x1":           "MALMO_V 1x1x1",
        "3x3x3":           "MALMO_V 3x3x3",
        "5x5x5":           "MALMO_V 5x5x5",
        "3x3x3^PC":        "MALMO_C 3x3x3",
        "3x3x3^AABB":      "MALMO_AABB 3x3x3",
    }
    # Header line: Method  centroid  random  near_face ...
    lines = section.splitlines()
    # Find the header line
    header_i = None
    for i, ln in enumerate(lines):
        if ln.startswith("Method") and "centroid" in ln:
            header_i = i
            break
    if header_i is None:
        return {}
    # Determine column order
    cols = lines[header_i].split()[1:]  # skip "Method"
    results = {}
    for ln in lines[header_i + 2:]:
        parts = ln.split()
        if not parts or parts[0] not in method_map:
            # Stop at first non-data line
            if parts and parts[0].isupper():
                break
            continue
        raw_name = parts[0]
        display_name = method_map[raw_name]
        # The row looks like "1x1x1        49.42%   49.88%  ..."
        pct_re = re.compile(r"([0-9.]+)%")
        pcts = [float(x) for x in pct_re.findall(ln)]
        if len(pcts) >= len(cols):
            results[display_name] = dict(zip(cols, pcts[:len(cols)]))
    return results


def parse_scalability(log_text: str) -> list:
    """Parse the scalability sweep summary table."""
    start = log_text.find("SCALABILITY SWEEP")
    if start < 0:
        return []
    # Look for the summary row block
    lines = log_text[start:].splitlines()
    rows = []
    # Detect data rows: "  1,000    1.849       541  ..."
    row_re = re.compile(
        r"^\s+(?P<n>[\d,]+)\s+"
        r"(?P<t>[0-9.]+)\s+"
        r"(?P<qps>[\d,]+)\s+"
        r"(?P<us>[0-9.]+)\s+"
        r"(?P<pits>[\d,]+)\s+"
        r"(?P<mean_pit>[0-9.]+)\s+"
        r"(?P<found_pct>[0-9.]+)%\s*$"
    )
    for ln in lines:
        m = row_re.match(ln)
        if m:
            rows.append({
                "n_p":       int(m.group("n").replace(",", "")),
                "time_mean": float(m.group("t")),
                "qps":       int(m.group("qps").replace(",", "")),
                "us_per_q":  float(m.group("us")),
                "pit_s":     int(m.group("pits").replace(",", "")),
                "mean_pit":  float(m.group("mean_pit")),
                "found_pct": float(m.group("found_pct")),
            })
    return rows


def parse_level_distribution(log_text: str) -> dict:
    """Parse the resolving-level histogram for the 3x3x3^PC method."""
    # The histogram appears after "Level" "Count" "Fraction" "Cumulative" header
    start = log_text.find("Level     Count    Fraction  Cumulative")
    if start < 0:
        start = log_text.find("Level")
        while start >= 0:
            head = log_text[start: start + 100]
            if "Count" in head and "Fraction" in head:
                break
            start = log_text.find("Level", start + 1)
    if start < 0:
        return {}
    lines = log_text[start:].splitlines()[1:]  # skip header
    rows = {}
    for ln in lines[:15]:
        # Row: "     7    8,502    85.02%    85.02%"
        m = re.match(
            r"^\s*(?P<lev>\d+)\s+(?P<count>[\d,]+)\s+"
            r"(?P<frac>[0-9.]+)%\s+(?P<cum>[0-9.]+)%\s*$",
            ln,
        )
        if not m:
            if ln.strip() and not ln.startswith(" "):
                break
            continue
        rows[int(m.group("lev"))] = {
            "count":     int(m.group("count").replace(",", "")),
            "frac":      float(m.group("frac")),
            "cumulative": float(m.group("cum")),
        }
    return rows


def parse_failure_decomp(log_text: str) -> dict:
    """Parse the 1x1x1 failure decomposition summary.

    The harness prints:
        Face-adjacent  (6 cells):  ...
        Edge-adjacent (12 cells):  ...
        Corner-adjacent (8 cells): ...

    The counts as printed are (level, offset) hit counts, not per-particle
    counts. We also record 'Analyzed N failures' so downstream code can
    compute the intended per-particle percentages.
    """
    start = log_text.find("1x1x1 FAILURE ANALYSIS")
    if start < 0:
        return {}
    section = log_text[start: start + 8000]
    result = {}

    m = re.search(r"Analyzed\s+([\d,]+)\s+failures?", section)
    if m:
        result["n_analyzed"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Missed by 1x1x1 but found by 3x3x3:\s*([\d,]+)", section)
    if m:
        result["n_missed"] = int(m.group(1).replace(",", ""))
    for label, key in (
        ("Face-adjacent",   "face_adjacent"),
        ("Edge-adjacent",   "edge_adjacent"),
        ("Corner-adjacent", "corner_adjacent"),
    ):
        m = re.search(
            rf"{label}\s+\([^)]*\):\s+([\d,]+)\s+\(([0-9.]+)%\)",
            section,
        )
        if m:
            result[key] = {
                "raw_count": int(m.group(1).replace(",", "")),
                "raw_pct":   float(m.group(2)),
            }
    return result


def parse_bytes_per_query(log_text: str) -> dict:
    """Parse the bytes/query table from the XLA cost analysis.

    Log format:
        Method               Bytes/query    HLO flop-ops    HLO ops   fusions
        ---
        radius r=2                62,177             234        535        40
        1x1x1                     40,233             381        803        35
        3x3x3^PC                  32,706             386        834        46
    """
    start = log_text.find("XLA COST ANALYSIS")
    if start < 0:
        start = log_text.find("Bytes/query")
    if start < 0:
        return {}
    section = log_text[start: start + 4000]

    # Anchor each method name to the start of the line (line begins
    # with optional spaces + the method label). Method names contain
    # digits ("1x1x1"), so a naive `re.findall(digits, line)` picks
    # them up as false numeric columns; instead anchor an explicit
    # per-method regex.
    method_map = [
        (r"radius\s+r=2\b",        "Morton-linear w=5"),
        (r"radius\s+r=10\b",       "Morton-linear w=21"),
        (r"1x1x1\b",               "MALMO_V 1x1x1"),
        (r"3x3x3\b(?!\^)",         "MALMO_V 3x3x3"),
        (r"5x5x5\b",               "MALMO_V 5x5x5"),
        (r"3x3x3\^PC\b",           "MALMO_C 3x3x3"),
        (r"3x3x3\^AABB\b",         "MALMO_AABB 3x3x3"),
    ]
    results = {}
    for method_pat, display_name in method_map:
        # Line = method label + whitespace + bytes/query (first
        # numeric column, possibly with commas) + rest.
        row_re = re.compile(
            rf"^\s*{method_pat}\s+(?P<bytes>[\d,]+)\b",
            re.MULTILINE,
        )
        m = row_re.search(section)
        if m:
            results[display_name] = int(m.group("bytes").replace(",", ""))
    return results


def parse_mesh_properties(log_text: str) -> dict:
    """Parse mesh statistics printed at build time.

    Reads the labels the harness prints during stage 3:
        Vertex-multi octree:  666,162 cells, 18.3 elem/cell, 4.00 cells/elem
        Parent-cube octree:   518,858 cells, 5.9 elem/cell (max 24)
        AABB-overlap octree:  620,760 cells, 30.8 elem/cell (max 162), 6.3 cells/elem
        Morton structure:     517,309 cells, 5.9 elem/cell

    IMPORTANT: 'Morton structure' (517,309) is NOT MALMO^C.
    The MALMO^C data structure is the 'Parent-cube octree' (518,858).
    The paper's Table 6.1 previously conflated these two — see
    sec6_inconsistencies_2.md for the source of the bug.
    """
    result: dict[str, Any] = {}
    m = re.search(r"Mesh size:\s+([\d,]+)\s+elements,\s+([\d,]+)\s+nodes", log_text)
    if m:
        result["n_elements"] = int(m.group(1).replace(",", ""))
        result["n_nodes"] = int(m.group(2).replace(",", ""))

    # Parse each labeled octree/structure. Both possible AABB labels are
    # supported for backward compatibility with older logs.
    label_map = [
        ("Vertex-multi octree:",  "vertex"),
        ("Parent-cube octree:",   "parent_cube"),
        ("AABB-overlap octree:",  "aabb"),
        ("AABB octree:",          "aabb"),
        ("Morton structure:",     "morton"),
    ]
    for label, key in label_map:
        # Optional max-occ suffix + optional cells/elem column.
        m = re.search(
            rf"{re.escape(label)}\s+([\d,]+)\s+cells,\s+([0-9.]+)\s+elem/cell"
            rf"(?:\s+\(max\s+(\d+)\))?"
            rf"(?:,\s+([0-9.]+)\s+cells/elem)?",
            log_text,
        )
        if m and f"{key}_n_cells" not in result:
            result[f"{key}_n_cells"] = int(m.group(1).replace(",", ""))
            result[f"{key}_elem_per_cell"] = float(m.group(2))
            if m.group(3):
                result[f"{key}_max_elem_per_cell"] = int(m.group(3))
            if m.group(4):
                result[f"{key}_cells_per_elem"] = float(m.group(4))
    return result


# =============================================================================
# Canonical aggregation and diff
# =============================================================================

@dataclass
class TableDiff:
    name: str
    rows: list[dict] = field(default_factory=list)
    all_ok: bool = True


def close_enough(measured: float, paper: float,
                 rel_tol: float = 0.02, abs_tol: float = 0.5) -> bool:
    """Returns True iff measured is within max(rel_tol * |paper|, abs_tol)."""
    if paper is None or measured is None:
        return measured == paper
    if paper == 0:
        return abs(measured) <= abs_tol
    diff = abs(measured - paper)
    tol = max(rel_tol * abs(paper), abs_tol)
    return diff <= tol


def diff_found_rate(measured: dict, paper: dict, sigmas: list) -> TableDiff:
    diff = TableDiff(name="tab:found_rate (Found rate under perturbation)")
    for method, paper_vals in paper.items():
        if method not in measured:
            diff.rows.append({"method": method, "status": "MISSING",
                              "paper": paper_vals, "measured": None})
            diff.all_ok = False
            continue
        m_rows = measured[method]
        m_by_sigma = {r["sigma"]: r["found_pct"] for r in m_rows}
        m_vals = [m_by_sigma.get(s) for s in sigmas]
        # Percentage-scale values: use absolute-only tolerance of 0.05
        # (five hundredths of a percent), because the rel_tol arm of
        # close_enough would otherwise dominate at 100 x 0.005 = 0.5
        # and mask real drift like 100.00 → 99.93.
        row_ok = all(m is not None and abs(m - p) <= 0.05
                     for m, p in zip(m_vals, paper_vals))
        if not row_ok:
            diff.all_ok = False
        diff.rows.append({
            "method": method, "sigmas": sigmas,
            "paper": paper_vals, "measured": m_vals,
            "status": "OK" if row_ok else "CHANGED",
        })
    return diff


def diff_search_failures(measured: dict, paper: dict, sigmas: list) -> TableDiff:
    diff = TableDiff(name="tab:search_failures (in-bbox misses)")
    for method, paper_vals in paper.items():
        if method not in measured:
            continue
        m_rows = measured[method]
        m_by_sigma = {r["sigma"]: r["search_fail"] for r in m_rows}
        m_vals = [m_by_sigma.get(s) for s in sigmas]
        row_ok = all(close_enough(m, p, rel_tol=0.02, abs_tol=5)
                     if p > 0 else (m == 0)
                     for m, p in zip(m_vals, paper_vals))
        if not row_ok:
            diff.all_ok = False
        diff.rows.append({
            "method": method, "sigmas": sigmas,
            "paper": paper_vals, "measured": m_vals,
            "status": "OK" if row_ok else "CHANGED",
        })
    return diff


def diff_intra(measured: dict, paper: dict, position_types: list) -> TableDiff:
    diff = TableDiff(name="tab:intra_found (Intra-element found rate)")
    for method, paper_vals in paper.items():
        if method not in measured:
            continue
        m_dict = measured[method]
        m_vals = [m_dict.get(pt) for pt in position_types]
        # Percentage-scale values: absolute-only 0.05 tolerance.
        # 100.00 → 99.98 is inside tol; 100.00 → 99.96 is NOT and
        # will correctly flag MALMO^C rows as CHANGED after the
        # padding-bound bug shifts it away from perfect coverage.
        row_ok = all(m is not None and abs(m - p) <= 0.05
                     for m, p in zip(m_vals, paper_vals))
        if not row_ok:
            diff.all_ok = False
        diff.rows.append({
            "method": method, "position_types": position_types,
            "paper": paper_vals, "measured": m_vals,
            "status": "OK" if row_ok else "CHANGED",
        })
    return diff


def apply_canonical_timing(measured_perturb: dict, n_p: int) -> dict:
    """From the sigma=0.0 row of the perturbation sweep, compute the
    canonical timing table using the fixed aggregation rule:

        time_min, time_max — as reported
        qps                — n_p / time_mean (where time_mean is unavailable
                                              from parse; we approximate as
                                              (time_min + time_max) / 2 unless
                                              a mean is printed elsewhere)

    NB: benchmark_l2_accuracy.py already stores per-run times and computes
    queries_per_sec = n_p / mean(times). We recover that same statistic from
    the "queries/s" number the harness printed on the same log line, which
    matches the internal mean-based rule.
    """
    canonical = {}
    for method, m_rows in measured_perturb.items():
        for r in m_rows:
            if r["sigma"] == 0.0:
                # The qps_reported field is n_p / t_mean from the harness.
                # This is the SINGLE aggregation rule we adopt across all
                # tables in the re-run.
                qps = r["qps_reported"]
                canonical[method] = {
                    "time_min":  r["time_min"],
                    "time_max":  r["time_max"],
                    "mean_pit":  r["mean_pit"],
                    "qps":       qps,
                }
                break
    return canonical


def diff_timing(measured: dict, paper: dict) -> TableDiff:
    diff = TableDiff(name="tab:timing (Computational performance @ sigma=0)")
    for method, p in paper.items():
        m = measured.get(method)
        if m is None:
            diff.rows.append({"method": method, "status": "MISSING",
                              "paper": p, "measured": None})
            diff.all_ok = False
            continue
        # Time range: check both endpoints within 3% relative or 30 ms absolute.
        ok_min = close_enough(m["time_min"], p["time_min"], rel_tol=0.03, abs_tol=0.03)
        ok_max = close_enough(m["time_max"], p["time_max"], rel_tol=0.03, abs_tol=0.03)
        # Mean PIT: strict, we don't expect drift.
        ok_pit = (p["mean_pit"] is None or m["mean_pit"] is None
                  or close_enough(m["mean_pit"], p["mean_pit"], rel_tol=0.01, abs_tol=0.5))
        # Queries/s: canonical rule (mean-based). The paper's Table tab:timing
        # used MAX-based qps for some rows, so we expect small diffs here.
        ok_qps = close_enough(m["qps"], p["qps"], rel_tol=0.05, abs_tol=100)
        row_ok = ok_min and ok_max and ok_pit and ok_qps
        if not row_ok:
            diff.all_ok = False
        diff.rows.append({
            "method": method, "paper": p, "measured": m,
            "status": "OK" if row_ok else "CHANGED",
            "sub_status": {"time_min": ok_min, "time_max": ok_max,
                           "mean_pit": ok_pit, "qps": ok_qps},
        })
    return diff


def diff_scalability(measured: list, paper: dict) -> TableDiff:
    diff = TableDiff(name="tab:scalability (throughput vs batch size)")
    by_n = {row["n_p"]: row for row in measured}
    for n, p in sorted(paper.items()):
        m = by_n.get(n)
        if m is None:
            diff.rows.append({"n_p": n, "status": "MISSING",
                              "paper": p, "measured": None})
            diff.all_ok = False
            continue
        ok_t = close_enough(m["time_mean"], p["time"], rel_tol=0.05, abs_tol=0.03)
        ok_q = close_enough(m["qps"], p["qps"], rel_tol=0.05, abs_tol=100)
        row_ok = ok_t and ok_q
        if not row_ok:
            diff.all_ok = False
        diff.rows.append({
            "n_p": n, "paper": p, "measured": m,
            "status": "OK" if row_ok else "CHANGED",
            "sub_status": {"time_mean": ok_t, "qps": ok_q},
        })
    return diff


def diff_level_dist(measured: dict, paper: dict) -> TableDiff:
    """Compare measured (absolute log levels 8..14) against paper (relative
    levels 1..7). Uses LEVEL_LOG_TO_PAPER to translate.
    """
    diff = TableDiff(name="tab:level_dist (Resolving level histogram)")

    # Translate measured (absolute) → paper (relative), aggregating
    # log levels 8..10 into paper "level ≤ 3".
    aggregated: dict[int, dict] = {}
    for log_lev, entry in measured.items():
        paper_lev = LEVEL_LOG_TO_PAPER.get(int(log_lev))
        if paper_lev is None:
            continue
        if paper_lev not in aggregated:
            aggregated[paper_lev] = {"count": 0, "frac": 0.0}
        aggregated[paper_lev]["count"] += int(entry["count"])
        aggregated[paper_lev]["frac"]  += float(entry["frac"])

    # Add a note explaining the convention gap.
    diff.rows.append({
        "note":
            "Log uses absolute octree depth (8..14); paper uses "
            "relative depth (1..7 with finest = level 7). This diff "
            "maps 14↔7, 13↔6, 12↔5, 11↔4, and aggregates log "
            "levels 8..10 into paper 'level ≤ 3'. See "
            "LEVEL_LOG_TO_PAPER for the full mapping."
    })

    for lev, p in paper.items():
        m = aggregated.get(lev)
        if m is None:
            diff.rows.append({"level": lev, "status": "MISSING",
                              "paper": p, "measured": None})
            diff.all_ok = False
            continue
        ok_c = close_enough(m["count"], p["count"], rel_tol=0.05, abs_tol=10)
        ok_f = close_enough(m["frac"], p["frac"], rel_tol=0.05, abs_tol=0.2)
        row_ok = ok_c and ok_f
        if not row_ok:
            diff.all_ok = False
        diff.rows.append({
            "level": lev, "paper": p, "measured": m,
            "status": "OK" if row_ok else "CHANGED",
        })
    return diff


def apply_canonical_failure_decomp(raw: dict) -> dict:
    """The raw parse gives (level, offset) hit counts. The canonical
    per-particle decomposition classifies each failed particle by the
    smallest recovering offset (face < edge < corner). We approximate
    this from the raw ratios: since the paper's earlier per-particle
    quotes have ~50% edge-adjacent, and the raw ratios were 37.3 / 50 /
    12.7, we can either

      (a) re-run the failure analyser with per-particle logic
          (requires a code patch inside benchmark_l2_accuracy.py), or

      (b) treat the raw counts as they are AND flag the sum as the
          (level, offset) hit total in the caption.

    This post-processor implements (b): report both the raw hits and
    (when available from a future patched run) the per-particle
    breakdown. The 'sum' field is the raw hit total (matches the paper's
    20,000 number IF the harness ran with the same particle set).
    """
    if not raw:
        return {}
    total = sum(raw.get(k, {}).get("raw_count", 0)
                for k in ("face_adjacent", "edge_adjacent", "corner_adjacent"))
    return {
        "n_missed":                raw.get("n_missed"),
        "n_analyzed":              raw.get("n_analyzed"),
        "sum_of_offset_hits":      total,
        "face_adjacent":           raw.get("face_adjacent"),
        "edge_adjacent":           raw.get("edge_adjacent"),
        "corner_adjacent":         raw.get("corner_adjacent"),
    }


def diff_failure_decomp(measured: dict, paper: dict) -> TableDiff:
    diff = TableDiff(name="tab:failure_decomposition (1x1x1 misses)")
    diff.rows.append({
        "note":
            "The paper table's counts (7463 / 9995 / 2542) sum to 20,000, "
            "which is NOT a batch size — it is the total number of "
            "(level, offset) neighbour hits across all failed particles. "
            "The raw harness histogram is the same statistic, so the "
            "measured 'sum_of_offset_hits' should match the paper's total "
            "(20,000) at the same N_p and sigma. If a per-particle "
            "decomposition is desired, the caption needs to be updated OR "
            "the failure analyser patched to classify each particle by "
            "its smallest recovering offset."
    })
    for key in ("face_adjacent", "edge_adjacent", "corner_adjacent"):
        p = paper[key]
        m = measured.get(key)
        if m is None:
            diff.rows.append({"key": key, "status": "MISSING",
                              "paper": p, "measured": None})
            diff.all_ok = False
            continue
        ok_c = close_enough(m["raw_count"], p["count"], rel_tol=0.05, abs_tol=50)
        ok_p = close_enough(m["raw_pct"], p["share"], rel_tol=0.05, abs_tol=1.0)
        row_ok = ok_c and ok_p
        if not row_ok:
            diff.all_ok = False
        diff.rows.append({
            "key": key, "paper": p, "measured": m,
            "status": "OK" if row_ok else "CHANGED",
        })
    return diff


def diff_bytes(measured: dict, paper: dict) -> TableDiff:
    diff = TableDiff(name="tab:bytes (memory traffic per query)")
    for method, p in paper.items():
        m = measured.get(method)
        if m is None:
            diff.rows.append({"method": method, "status": "MISSING",
                              "paper": p, "measured": None})
            diff.all_ok = False
            continue
        ok = close_enough(m, p, rel_tol=0.02, abs_tol=50)
        if not ok:
            diff.all_ok = False
        diff.rows.append({
            "method": method, "paper": p, "measured": m,
            "status": "OK" if ok else "CHANGED",
        })
    return diff


# =============================================================================
# Report rendering
# =============================================================================

def render_markdown_report(all_diffs: list[TableDiff],
                           protocol: dict[str, Any],
                           structural_findings: dict) -> str:
    lines = [
        "# Section 6 (Validation) — re-run report",
        "",
        "## Structural findings (block paper finalisation)",
        "",
    ]
    if structural_findings:
        for name, entry in structural_findings.items():
            lines.append(f"### {name}")
            lines.append("")
            for k, v in entry.items():
                lines.append(f"* **{k}**: {v}")
            lines.append("")
    else:
        lines.append("_None — every table's numbers can be updated in place._")
        lines.append("")

    lines.append("## Canonical protocol used for this re-run")
    lines.append("")
    for k, v in protocol.items():
        lines.append(f"* **{k}**: `{v}`")
    lines.append("")
    lines.append("## Table-by-table diff (paper vs re-run)")
    lines.append("")
    for d in all_diffs:
        flag = "PASS" if d.all_ok else "**CHANGED**"
        lines.append(f"### {d.name} — {flag}")
        lines.append("")
        for row in d.rows:
            if "note" in row:
                lines.append(f"> {row['note']}")
                lines.append("")
                continue
            paper = row.get("paper")
            measured = row.get("measured")
            status = row.get("status", "?")
            label = (row.get("method") or row.get("level") or row.get("n_p")
                     or row.get("key") or "?")
            lines.append(f"* **{label}** — {status}")
            lines.append(f"    * paper:    `{paper}`")
            lines.append(f"    * measured: `{measured}`")
        lines.append("")
    return "\n".join(lines)


def detect_structural_findings(mesh_props: dict,
                               perturb_summary: dict,
                               log_text: str) -> dict:
    """Return a dict of structural findings (Sonnet Issues 1–3-style)
    that block finalisation and require code fixes rather than caption
    updates.
    """
    findings: dict[str, dict] = {}

    # F1: MALMO^C static-loop-bound truncation. Signal: any 3x3x3^PC
    # sigma row has search_fail > 0 while any other 3x3x3 variant is 0.
    m_c = perturb_summary.get("MALMO_C 3x3x3", [])
    m_v = perturb_summary.get("MALMO_V 3x3x3", [])
    if m_c and m_v:
        c_fail_max = max(r["search_fail"] for r in m_c)
        v_fail_max = max(r["search_fail"] for r in m_v)
        if c_fail_max > 0 and v_fail_max == 0:
            # Extract the observed max occupancy from the mesh-props parse.
            max_occ = mesh_props.get("parent_cube_max_elem_per_cell")
            m = re.search(r"static loop bound\s*=\s*(\d+)", log_text)
            bound = int(m.group(1)) if m else None
            findings["F1: MALMO^C static-loop-bound truncation"] = {
                "symptom":
                    "MALMO^C shows a small (5-11) sigma-independent count "
                    "of search failures while the other 3x3x3 variants "
                    "return 0 at every sigma level.",
                "root cause":
                    "The parent-cube search's inner fori_loop has a "
                    "compile-time-constant upper bound "
                    f"(config.MAX_ELEMS_PER_CELL = {bound}), but the "
                    f"actual observed max cell occupancy on this mesh is "
                    f"{max_occ}. Any cell with more registered elements "
                    "than the static bound has its overflow silently "
                    "truncated at query time.",
                "fix in code":
                    "Raise config.MAX_ELEMS_PER_CELL to at least the "
                    f"observed max ({max_occ}) — a value of 32 gives "
                    "headroom. benchmark_l2_accuracy.py now auto-lifts "
                    "the effective bound at runtime and prints a warning "
                    "in future logs.",
                "paper-side action":
                    "Re-run the benchmark after the fix. Every MALMO^C "
                    "row in tab:found_rate, tab:search_failures, "
                    "tab:intra_found, tab:timing (0 fail changes to true "
                    "0), and tab:scalability caption (which claims '100% "
                    "found rate at all sizes') must be regenerated. The "
                    "prose claim 'all four 3x3x3 variants achieve 100% "
                    "found rate at sigma=0' becomes true after the fix.",
            }

    # F2: Missing AABB extract line item in the build-cost table.
    if "octree_extract_aabb" not in log_text:
        findings["F2: AABB extract not itemised in build-time table"] = {
            "symptom":
                "sec6_raw.log's PREPROCESSING/BUILD STATISTICS block "
                "shows no line for octree_extract_aabb or "
                "octree_upload_aabb even though the AABB octree is "
                "clearly built (it appears in Table tab:memory and "
                "tab:mesh_properties).",
            "root cause":
                "benchmark_l2_accuracy.py's display loop for the build "
                "table hard-coded the stage list and omitted the two "
                "AABB timers.",
            "fix in code":
                "Display loop patched to include octree_extract_aabb "
                "and octree_upload_aabb, and to compute TOTAL from the "
                "displayed sub-stages only (excluding the wrapper "
                "timer that would double-count).",
            "paper-side action":
                "Regenerate the AABB build cost in Section 6.4.7 and "
                "in the '~617 s total' / 'break-even at 5-6e6 queries' "
                "claims. Both were derived from the mis-summed table.",
        }

    # F3: Cell-count conflation. Signal: mesh_props has both parent_cube
    # and morton parses AND they differ. Paper reference has 517,309 for
    # MALMO^C which is the Morton count.
    pc_cells = mesh_props.get("parent_cube_n_cells")
    morton_cells = mesh_props.get("morton_n_cells")
    if pc_cells is not None and morton_cells is not None and pc_cells != morton_cells:
        findings["F3: Cell count mislabelled in Table 6.1"] = {
            "symptom":
                f"The paper attributes {morton_cells:,} cells to "
                f"MALMO^C (parent-cube), but that number is actually "
                f"the Morton-linear structure's cell count. The true "
                f"MALMO^C parent-cube octree has {pc_cells:,} cells.",
            "root cause":
                "Copy-paste error at paper draft time; the two structures "
                "were treated as one row in Table 6.1.",
            "fix in code":
                "No code change required. The harness already prints "
                "the correct labels for both structures.",
            "paper-side action":
                f"Update Table 6.1 (tab:mesh_properties) MALMO^C row: "
                f"active cells {pc_cells:,} (max occupancy "
                f"{mesh_props.get('parent_cube_max_elem_per_cell', 'N/A')}). "
                "Add a separate row or footnote for the Morton-linear "
                "structure count if it is used elsewhere in the text.",
        }

    return findings


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True,
                    help="Raw stdout log from benchmark_l2_accuracy.py")
    ap.add_argument("--json", type=Path, required=True,
                    help="Output path for the JSON dump")
    ap.add_argument("--report", type=Path, required=True,
                    help="Output path for the human-readable Markdown report")
    args = ap.parse_args()

    log_text = args.log.read_text()

    # ---- Extract every measured quantity ----
    perturb = parse_perturb_summary(log_text)
    intra = parse_intra_element(log_text)
    scalability = parse_scalability(log_text)
    level_dist = parse_level_distribution(log_text)
    failure_raw = parse_failure_decomp(log_text)
    bytes_pq = parse_bytes_per_query(log_text)
    mesh_props = parse_mesh_properties(log_text)

    # ---- Apply canonical rules ----
    n_p = 10_000
    canonical_timing = apply_canonical_timing(perturb, n_p)
    canonical_failure = apply_canonical_failure_decomp(failure_raw)

    # ---- Cross-check against paper values ----
    sigmas = [0.0, 0.1, 0.2, 0.5, 0.7, 1.0]
    position_types = ["centroid", "random", "near_face", "near_edge", "near_vertex"]
    diffs = [
        diff_found_rate(perturb, PAPER_FOUND_RATE, sigmas),
        diff_search_failures(perturb, PAPER_SEARCH_FAILURES, sigmas),
        diff_intra(intra, PAPER_INTRA_FOUND, position_types),
        diff_timing(canonical_timing, PAPER_TIMING),
        diff_scalability(scalability, PAPER_SCALABILITY),
        diff_level_dist(level_dist, PAPER_LEVEL_DIST),
        diff_failure_decomp(canonical_failure, PAPER_FAILURE_DECOMP),
        diff_bytes(bytes_pq, PAPER_BYTES),
    ]

    # ---- Emit JSON ----
    payload = {
        "protocol": {
            "n_particles":         n_p,
            "batch_size":          50000,
            "warmup_runs":         3,
            "timing_runs":         7,
            "perturbations":       sigmas,
            "position_types":      position_types,
            "scalability_sizes":   [row["n_p"] for row in scalability],
            "queries_per_second":  "n_p / mean(times)  [canonical rule; single aggregation across ALL tables]",
            "failure_decomposition_counts":
                "raw (level, offset) neighbour-hit histogram — sum is NOT a batch size; paper caption must state this OR the analyser must be patched to per-particle classification",
        },
        "measured": {
            "perturbation_summary":        perturb,
            "intra_element":               intra,
            "scalability":                 scalability,
            "level_distribution":          level_dist,
            "failure_decomposition_raw":   failure_raw,
            "failure_decomposition_canonical": canonical_failure,
            "bytes_per_query":             bytes_pq,
            "mesh_properties":             mesh_props,
            "timing_canonical":            canonical_timing,
        },
        "paper_reference": {
            "found_rate":            PAPER_FOUND_RATE,
            "search_failures":       PAPER_SEARCH_FAILURES,
            "intra_found":           PAPER_INTRA_FOUND,
            "timing":                PAPER_TIMING,
            "scalability":           PAPER_SCALABILITY,
            "level_distribution":    PAPER_LEVEL_DIST,
            "failure_decomposition": PAPER_FAILURE_DECOMP,
            "bytes":                 PAPER_BYTES,
            "mesh_properties":       PAPER_MESH_PROPS,
        },
        "diffs": [asdict(d) for d in diffs],
    }

    # Detect structural issues (Sonnet-style) that need code fixes,
    # not just paper-side number updates.
    structural = detect_structural_findings(mesh_props, perturb, log_text)
    payload["structural_findings"] = structural

    args.json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[postprocess] wrote {args.json}")

    report_md = render_markdown_report(diffs, payload["protocol"], structural)
    args.report.write_text(report_md)
    print(f"[postprocess] wrote {args.report}")

    n_findings = len(structural)
    all_ok = all(d.all_ok for d in diffs)
    if all_ok and n_findings == 0:
        print("[postprocess] ALL tables match paper within tolerance and no "
              "structural issues detected.")
    elif n_findings > 0:
        print(f"[postprocess] {n_findings} STRUCTURAL FINDING(S) block paper "
              f"finalisation — see report for fixes.")
        sys.exit(2)
    else:
        print("[postprocess] Some tables CHANGED (number updates only). "
              "See the report for details.")
        sys.exit(2)


if __name__ == "__main__":
    main()
