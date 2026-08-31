#!/usr/bin/env python3
"""
Build the combined MALMO + SCONE + RTXAdvect comparison table across
the Kim cohort plus the FSW paper mesh.

Reads from (workstation paths):
  - malmo_runs_in_mesh/*/result.json          (MALMO on Kim cohort, in-mesh)
  - malmo_runs_fsw_paper/*/result.json        (MALMO on 3M-tet FSW mesh)
  - scone_runs_padfix/scone_runs/*/result.json (SCONE pad-fix sweep, Kim)
  - scone_runs/FSW_paper*/result.json         (SCONE on FSW mesh, overnight)
  - rtxadvect_runs/*/result.json              (RTXAdvect scale sweep, 1M x 100)

Emits:
  - joint_summary.csv  (one row per (mesh, framework, method, omp))
  - joint_summary.md   (Markdown table grouped by mesh)
  - stdout: pretty ASCII table
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

MESH_ORDER = [
    # Kim et al. 2026 fuel-pin cohort (SCONE reference meshes)
    "FinalFuelPinTet63", "FinalFuelPinTet137", "FinalFuelPinTet298",
    "FinalFuelPinTet1820", "FinalFuelPinTet2856",
    "FinalFuelPinPoly264", "FinalFuelPinPoly940", "FinalFuelPinPoly1560",
    # Cross-methodology sanity-check (low fill_ratio)
    "StanfordBunny_LowPoly",
    # Paper's target application mesh
    "FSW_paper",
]

# Group labels for the MD output
MESH_GROUPS = {
    "Kim cohort (fuel pins, SCONE-native)": [
        "FinalFuelPinTet63", "FinalFuelPinTet137", "FinalFuelPinTet298",
        "FinalFuelPinTet1820", "FinalFuelPinTet2856",
        "FinalFuelPinPoly264", "FinalFuelPinPoly940", "FinalFuelPinPoly1560",
    ],
    "Cross-methodology check": ["StanfordBunny_LowPoly"],
    "Paper's target application mesh": ["FSW_paper"],
}

# Different frameworks label the FSW paper mesh differently — MALMO uses
# the source file stem ("cylA_119"), SCONE+RTXAdvect use the tag
# ("FSW_paper"). Normalize on the tag.
MESH_ALIASES = {
    "cylA_119": "FSW_paper",
    "cylA_119.pvtu": "FSW_paper",
}

def _rss_mb(d):
    v = d.get("process", {}).get("max_rss_kb")
    return (v / 1024.0) if v else None

def _malmo_row(mesh, variant, d):
    ws = d.get("wall_seconds", {}) or {}
    return {
        "mesh": mesh, "framework": "MALMO", "method": variant,
        "hardware": "GPU", "parallel_mode": "JAX",
        "omp_threads": 1,
        "n_tets": d.get("n_tets"), "n_vertices": d.get("n_nodes"),
        "sampling": d.get("sampling", "in_mesh_barycentric"),
        "correct_rate_strict": d.get("correct_rate_strict"),
        "found_rate": d.get("found_rate"),
        "build_seconds": ws.get("build_octree"),
        "query_seconds": ws.get("query_min_of_3"),
        "wall_seconds": ws.get("query_min_of_3"),  # kept for back-compat
        "throughput_Mqps": (d.get("throughput_queries_per_second") or 0) / 1e6 if d.get("throughput_queries_per_second") else None,
        "max_rss_mb": _rss_mb(d),
        "mean_pit_tests": d.get("mean_pit_tests"),
        "fill_ratio": d.get("fill_ratio"),
        "n_octree_cells": d.get("n_octree_cells"),
        "status": "TIMED_OUT" if d.get("timed_out") else ("FAILED" if d.get("found_rate") is None else "OK"),
    }

def _scone_row(mesh, method, omp, d):
    seg = d.get("crashed_sigsegv")
    ec = d.get("shell_exit_code")
    if seg: status = "SEGV"
    elif ec == 0: status = "OK"
    else: status = f"FAIL(ec={ec})"
    return {
        "mesh": mesh, "framework": "SCONE", "method": method,
        "hardware": "CPU", "parallel_mode": f"OMP_{omp}" if omp else "serial",
        "omp_threads": omp,
        "n_tets": None, "n_vertices": None,
        "sampling": "random_in_bbox",  # SCONE ray-traces into the box
        "correct_rate_strict": None,
        "found_rate": None,
        "build_seconds": None,          # SCONE reports no build/query split
        "query_seconds": d.get("wall_seconds_outer"),
        "wall_seconds": d.get("wall_seconds_outer"),
        "throughput_Mqps": None,
        "max_rss_mb": _rss_mb(d),
        "mean_pit_tests": None,
        "fill_ratio": None,
        "n_octree_cells": None,
        "status": status,
        "box_pad": d.get("box_pad"),
    }

def _rtx_row(mesh, d):
    ec = d.get("shell_exit_code")
    status = "OK" if ec == 0 else f"FAIL(ec={ec})"
    ws = d.get("wall_seconds") or {}
    bvh_s = (ws.get("bvh_ms") / 1000.0) if ws.get("bvh_ms") is not None else None
    sim_s = (ws.get("simulation_ms") / 1000.0) if ws.get("simulation_ms") is not None else None
    return {
        "mesh": mesh, "framework": "RTXAdvect", "method": "RTX_BVH",
        "hardware": "GPU", "parallel_mode": "OptiX_RT",
        "omp_threads": 1,
        "n_tets": d.get("n_tets"), "n_vertices": d.get("n_vertices"),
        "sampling": d.get("sampling", "random_in_bbox"),
        "correct_rate_strict": None,   # RTXAdvect doesn't dump host-tet IDs
        "found_rate": d.get("found_rate"),
        "build_seconds": bvh_s,
        "query_seconds": sim_s,
        "wall_seconds": ws.get("outer"),
        "throughput_Mqps": d.get("throughput_Mqps"),
        "max_rss_mb": (d.get("process", {}).get("max_rss_kb") / 1024.0)
                       if d.get("process", {}).get("max_rss_kb") else None,
        "mean_pit_tests": None,
        "fill_ratio": None,
        "n_octree_cells": None,
        "status": status,
        "bvh_ms": ws.get("bvh_ms"),
        "sim_ms": ws.get("simulation_ms"),
        "n_out_of_domain": d.get("n_out_of_domain"),
    }

def _norm_mesh(name):
    return MESH_ALIASES.get(name, name)

def collect(malmo_root: Path, malmo_fsw_root: Path, scone_root: Path,
            scone_fsw_root: Path, rtx_root: Path):
    rows = []
    # MALMO Kim cohort
    for f in sorted(malmo_root.glob("*/result.json")):
        d = json.load(open(f))
        mesh = _norm_mesh(d.get("mesh"))
        if mesh not in MESH_ORDER:
            continue
        variant = (d.get("method") or "").replace("MALMO_", "")
        rows.append(_malmo_row(mesh, variant, d))
    # MALMO FSW paper mesh (three variants) — but skip variants already
    # covered by malmo_runs_in_mesh (which has correct_rate_strict);
    # only add rows where the in-mesh cohort didn't produce one.
    covered = {(r["mesh"], r["method"]) for r in rows if r["framework"] == "MALMO"}
    if malmo_fsw_root and malmo_fsw_root.is_dir():
        for f in sorted(malmo_fsw_root.glob("*/result.json")):
            d = json.load(open(f))
            mesh = _norm_mesh(d.get("mesh"))
            if mesh not in MESH_ORDER:
                continue
            variant = (d.get("method") or "").replace("MALMO_", "")
            if (mesh, variant) in covered:
                continue
            rows.append(_malmo_row(mesh, variant, d))
    # SCONE Kim (walk recursively — padfix nests one level deeper)
    for f in sorted(scone_root.rglob("result.json")):
        d = json.load(open(f))
        mesh = _norm_mesh(d.get("mesh"))
        if mesh not in MESH_ORDER:
            continue
        rows.append(_scone_row(mesh, d.get("accelerationMethod"), d.get("omp_threads"), d))
    # SCONE FSW paper mesh (top-level scone_runs/FSW_paper__*)
    if scone_fsw_root and scone_fsw_root.is_dir():
        for f in sorted(scone_fsw_root.glob("FSW_paper__*/result.json")):
            d = json.load(open(f))
            mesh = _norm_mesh(d.get("mesh"))
            if mesh not in MESH_ORDER:
                continue
            rows.append(_scone_row(mesh, d.get("accelerationMethod"),
                                    d.get("omp_threads"), d))
    # RTXAdvect (all meshes) — prefer in-mesh sampling result if present
    rtx_inmesh_root = (rtx_root.parent / "rtxadvect_runs_inmesh") if rtx_root else None
    inmesh_meshes = set()
    if rtx_inmesh_root and rtx_inmesh_root.is_dir():
        for f in sorted(rtx_inmesh_root.glob("*/result.json")):
            d = json.load(open(f))
            mesh = _norm_mesh(d.get("mesh"))
            if mesh not in MESH_ORDER:
                continue
            rows.append(_rtx_row(mesh, d))
            inmesh_meshes.add(mesh)
    if rtx_root and rtx_root.is_dir():
        for f in sorted(rtx_root.glob("*/result.json")):
            d = json.load(open(f))
            mesh = _norm_mesh(d.get("mesh"))
            if mesh not in MESH_ORDER or mesh in inmesh_meshes:
                continue
            rows.append(_rtx_row(mesh, d))
    return rows

def _fmt(v, spec="", nafill="--"):
    if v is None: return nafill
    try: return format(v, spec)
    except Exception: return str(v)

def print_ascii(rows):
    by_mesh = defaultdict(list)
    for r in rows:
        by_mesh[r["mesh"]].append(r)
    hdr = f"{'framework':<7} {'method':<14} {'omp':>3} {'status':<10} {'correct%':>8} {'wall_s':>8} {'RSS_MB':>7} {'PIT':>7} {'tput_Mqps':>10}"
    print()
    print("=" * 96)
    for mesh in MESH_ORDER:
        entries = by_mesh.get(mesh, [])
        if not entries: continue
        print(f"\n--- {mesh}")
        print(hdr)
        # sort: MALMO first (by variant name), then SCONE (by method, then omp)
        def sk(r):
            fw = {"MALMO": 0, "SCONE": 1, "RTXAdvect": 2}.get(r["framework"], 9)
            return (fw, r["method"], r["omp_threads"] or 0)
        for r in sorted(entries, key=sk):
            cr = _fmt((r["correct_rate_strict"] or 0)*100 if r["correct_rate_strict"] is not None else None, ".2f", "--")
            print(f"{r['framework']:<7} {r['method']:<14} {r['omp_threads']:>3} {r['status']:<10} "
                  f"{cr:>7}% {_fmt(r['wall_seconds'], '.3f'):>8} "
                  f"{_fmt(r['max_rss_mb'], '.0f'):>7} {_fmt(r['mean_pit_tests'], '.1f'):>7} "
                  f"{_fmt(r['throughput_Mqps'], '.3f'):>10}")

def write_csv(rows, path: Path):
    cols = [
        # Identity
        "mesh", "framework", "method", "hardware", "parallel_mode", "omp_threads",
        # Mesh scale
        "n_tets", "n_vertices",
        # Query methodology
        "sampling", "status",
        # Correctness
        "correct_rate_strict", "found_rate",
        # Timing split
        "build_seconds", "query_seconds", "wall_seconds",
        # Throughput
        "throughput_Mqps",
        # Memory
        "max_rss_mb",
        # Framework diagnostics
        "mean_pit_tests", "fill_ratio", "n_octree_cells", "box_pad",
        "bvh_ms", "sim_ms", "n_out_of_domain",
    ]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    print(f"\nwrote CSV: {path}")

def write_md(rows, path: Path):
    by_mesh = defaultdict(list)
    for r in rows:
        by_mesh[r["mesh"]].append(r)
    out = ["# Joint MALMO + SCONE + RTXAdvect benchmark",
           "",
           "**Mesh cohort**: 10 meshes across three groups (Kim fuel-pin cohort of 8, "
           "cross-methodology sanity check on the Stanford Bunny, and the paper's target "
           "application mesh — a 3.05 M-tet friction-stir-welding domain). Scale spans "
           "60 tets to 3 050 196 tets. There is no established community-standard tet-mesh "
           "cohort for point-location benchmarks; the meshes cited by Wang 2022, Morrical "
           "2020 and Wald 2019 are gated interactive-rendering datasets 10-2000× larger "
           "than the FSW target and would not run to completion for SCONE.",
           "",
           "**Frameworks**:",
           "- **MALMO** — this paper's mesh-aligned multi-level octree (GPU / JAX). "
           "In-mesh barycentric sampling with ground-truth host IDs. `correct%` = fraction "
           "of queries where MALMO returned the SAME tet the point was drawn from.",
           "- **SCONE** — Kim et al. 2026 (Cambridge Nuclear) reference implementation of "
           "AMLG-Patch and its predecessors, four acceleration methods "
           "(`none`/`octree`/`patchSingle`/`patchMulti`), CPU + OpenMP (1 and 8 threads). "
           "Runs on the pad-fix sweep with `BOX_PAD=5`. SCONE's `rayVolPhysicsPackage` is a "
           "material-tally Monte-Carlo package, not a point-location tool: it reports "
           "cycle-level statistics (ray speed, elapsed time, per-material relative volume) "
           "but never per-particle host-cell hits, so both `correct%` and `found%` show `—`. "
           "Interpret the `status` column instead: `OK` means every ray contributed to the "
           "tally (implicitly all found valid material); `FAIL(ec=1)` means SCONE's own "
           "\"Ray has lost correct material\" error terminated tracking mid-cycle; `SEGV` "
           "means the acceleration structure could not be built.",
           "- **RTXAdvect** — Wang et al. 2022 (CPC), ported to CUDA 13.3 + OptiX 9.1 on "
           "RTX 5090 / Blackwell (compute capability 12.0). BVH-based host-tet locator using "
           "RT-cores. Runs 1M particles × 100 steps. `found%` = 1 - (out-of-domain / N_seeded). "
           "Sampling is **in-mesh barycentric** (matching MALMO) when a rerun is available "
           "under `rtxadvect_runs_inmesh/`; otherwise axis-aligned bounding-box seeding, "
           "which biases `found%` downward on meshes with low fill-ratio. Correct-tet "
           "ground truth is not compared because RTXAdvect does not report per-particle "
           "host-tet IDs back to the CPU.",
           "",
           "**Column key**",
           "- `hw` — CPU or GPU",
           "- `par` — parallel mode (JAX / OptiX_RT / OMP_N / serial)",
           "- `n_tets` — mesh size for reference",
           "- `correct%` / `found%` — see framework notes above; `—` = not measurable",
           "- `build (s)` — one-shot preprocessing (octree / BVH / patch grid) that "
           "amortises across queries",
           "- `query (s)` — per-batch query wall time (best of 3 for MALMO)",
           "- `RSS (MB)` — peak resident set size of the process",
           "- `tput (Mq/s)` — throughput in million-queries-per-second "
           "(`n_points_queried / query_seconds`)",
           ""]
    for group_name, group_meshes in MESH_GROUPS.items():
        # Only emit a group header if at least one mesh in it has data
        if not any(by_mesh.get(m) for m in group_meshes):
            continue
        out.append(f"# {group_name}")
        out.append("")
        for mesh in group_meshes:
            entries = by_mesh.get(mesh, [])
            if not entries: continue
            # Show n_tets in the mesh section header for reference
            n_tets_vals = [r.get("n_tets") for r in entries if r.get("n_tets")]
            n_tets_str = f" ({n_tets_vals[0]:,} tets)" if n_tets_vals else ""
            out.append(f"## {mesh}{n_tets_str}")
            out.append("")
            out.append("| framework | method | hw | par | status | correct% | found% | build (s) | query (s) | RSS (MB) | PIT | tput (Mq/s) |")
            out.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
            def sk(r):
                fw = {"MALMO": 0, "SCONE": 1, "RTXAdvect": 2}.get(r["framework"], 9)
                return (fw, r["method"], r["omp_threads"] or 0)
            for r in sorted(entries, key=sk):
                cr = ((r["correct_rate_strict"] or 0)*100) if r["correct_rate_strict"] is not None else None
                fr = ((r["found_rate"] or 0)*100) if r["found_rate"] is not None else None
                fr_fmt = ".4f" if (fr is not None and 99.99 < fr < 100.0) else ".2f"
                out.append(f"| {r['framework']} | `{r['method']}` | {r.get('hardware','?')} | "
                           f"{r.get('parallel_mode','?')} | {r['status']} | "
                           f"{_fmt(cr,'.2f','—')} | {_fmt(fr, fr_fmt, '—')} | "
                           f"{_fmt(r.get('build_seconds'),'.3f','—')} | "
                           f"{_fmt(r.get('query_seconds'),'.3f','—')} | "
                           f"{_fmt(r['max_rss_mb'],'.0f','—')} | "
                           f"{_fmt(r['mean_pit_tests'],'.1f','—')} | "
                           f"{_fmt(r['throughput_Mqps'],'.3f','—')} |")
            out.append("")
    Path(path).write_text("\n".join(out))
    print(f"wrote MD: {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--malmo-root", default="/flash/shared/jax/JAXTrace/malmo_runs_in_mesh")
    ap.add_argument("--malmo-fsw-root", default="/flash/shared/jax/JAXTrace/malmo_runs_fsw_paper")
    ap.add_argument("--scone-root", default="/flash/shared/jax/JAXTrace/scone_runs_padfix")
    ap.add_argument("--scone-fsw-root", default="/flash/shared/jax/JAXTrace/scone_runs")
    ap.add_argument("--rtx-root", default="/flash/shared/jax/JAXTrace/rtxadvect_runs")
    ap.add_argument("--csv", default="/flash/shared/jax/JAXTrace/joint_summary.csv")
    ap.add_argument("--md", default="/flash/shared/jax/JAXTrace/joint_summary.md")
    a = ap.parse_args()
    rows = collect(Path(a.malmo_root), Path(a.malmo_fsw_root),
                   Path(a.scone_root), Path(a.scone_fsw_root),
                   Path(a.rtx_root))
    print_ascii(rows)
    write_csv(rows, Path(a.csv))
    write_md(rows, Path(a.md))

if __name__ == "__main__":
    sys.exit(main())
