#!/usr/bin/env python3
"""
Aggregate scone_runs/*/result.json into a single CSV for cross-method
+ cross-mesh comparison against MALMO.

Usage:
    scripts/scone_bench/collect_results.py \\
        --root ./scone_runs \\
        --csv  ./scone_runs/summary.csv

Output columns (one row per SCONE run):
  mesh, accelerationMethod, n_zones, pop, cycles, omp_threads,
  main_timer_seconds, wall_seconds_outer, user_cpu_seconds,
  sys_cpu_seconds, percent_cpu, max_rss_kb,
  grid_min_angle, grid_min_edge_length, grid_n_layers,
  grid_spacing_coarsest, grid_spacing_finest, grid_size_x, grid_size_y,
  grid_size_z, n_cycles_measured, cycle_mean_wall, cycle_min_wall,
  cycle_max_wall
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


COLS = [
    "mesh",
    "accelerationMethod",
    "n_zones",
    "pop",
    "cycles",
    "omp_threads",
    "shell_exit_code",
    "crashed_sigsegv",
    "main_timer_seconds",
    "wall_seconds_outer",
    "user_cpu_seconds",
    "sys_cpu_seconds",
    "percent_cpu",
    "max_rss_kb",
    "grid_min_angle",
    "grid_min_edge_length",
    "grid_n_layers",
    "grid_spacing_coarsest",
    "grid_spacing_finest",
    "grid_size_x",
    "grid_size_y",
    "grid_size_z",
    "n_cells_coarsest",
    "n_cells_finest",
    "n_cycles_measured",
    "cycle_mean_wall",
    "cycle_min_wall",
    "cycle_max_wall",
    "result_json_path",
]


def flatten(obj: dict) -> dict:
    grid = obj.get("grid") or {}
    proc = obj.get("process") or {}
    cw = obj.get("cycle_wall_seconds") or []

    row = {
        "mesh": obj.get("mesh"),
        "accelerationMethod": obj.get("accelerationMethod"),
        "n_zones": obj.get("n_zones"),
        "pop": obj.get("pop"),
        "cycles": obj.get("cycles"),
        "omp_threads": obj.get("omp_threads"),
        "shell_exit_code": obj.get("shell_exit_code"),
        "crashed_sigsegv": obj.get("crashed_sigsegv"),
        "main_timer_seconds": obj.get("main_timer_seconds"),
        "wall_seconds_outer": obj.get("wall_seconds_outer"),
        "user_cpu_seconds": proc.get("user_cpu_seconds"),
        "sys_cpu_seconds": proc.get("sys_cpu_seconds"),
        "percent_cpu": proc.get("percent_cpu"),
        "max_rss_kb": proc.get("max_rss_kb"),
        "grid_min_angle": grid.get("min_angle"),
        "grid_min_edge_length": grid.get("min_edge_length"),
        "grid_n_layers": grid.get("n_layers"),
        "grid_spacing_coarsest": grid.get("grid_spacing_coarsest") or grid.get("grid_spacing_single"),
        "grid_spacing_finest": grid.get("grid_spacing_finest"),
        "grid_size_x": grid.get("grid_size_x"),
        "grid_size_y": grid.get("grid_size_y"),
        "grid_size_z": grid.get("grid_size_z"),
        "n_cells_coarsest": grid.get("n_cells_coarsest"),
        "n_cells_finest": grid.get("n_cells_finest"),
        "n_cycles_measured": len(cw),
        "cycle_mean_wall": statistics.fmean(cw) if cw else None,
        "cycle_min_wall": min(cw) if cw else None,
        "cycle_max_wall": max(cw) if cw else None,
    }
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate SCONE point-location benchmark results")
    ap.add_argument("--root", default="./scone_runs",
                    help="Directory containing per-run subdirectories with result.json")
    ap.add_argument("--csv", default=None,
                    help="Output CSV path (default: <root>/summary.csv)")
    args = ap.parse_args()

    root = Path(args.root)
    csv_path = Path(args.csv) if args.csv else root / "summary.csv"

    if not root.is_dir():
        print(f"ERROR: --root {root} is not a directory", file=sys.stderr)
        return 2

    result_jsons = sorted(root.glob("*/result.json"))
    if not result_jsons:
        # Try one level deeper (common when run wrapper changes cwd)
        result_jsons = sorted(root.glob("*/*/result.json"))
        if not result_jsons:
            result_jsons = sorted(root.rglob("result.json"))
    if not result_jsons:
        print(f"ERROR: no result.json files under {root}", file=sys.stderr)
        return 1
    print(f"found {len(result_jsons)} result.json files under {root}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for jp in result_jsons:
            try:
                obj = json.loads(jp.read_text())
            except Exception as e:
                print(f"WARN: could not parse {jp}: {e}", file=sys.stderr)
                continue
            row = flatten(obj)
            row["result_json_path"] = str(jp)
            w.writerow(row)

    print(f"wrote {csv_path} with {len(result_jsons)} rows")
    # Quick pretty print of the key columns
    print()
    def fmt(v, spec):
        if v is None:
            return "-"
        try:
            return format(v, spec)
        except Exception:
            return str(v)

    print(f"{'mesh':<28} {'method':<12} {'omp':>4} {'status':<8} {'wall_s':>10} {'user_s':>10} {'RSS_MB':>10} {'cells_finest':>14}")
    for jp in result_jsons:
        try:
            obj = json.loads(jp.read_text())
        except Exception:
            continue
        row = flatten(obj)
        rss_mb = row["max_rss_kb"] / 1024.0 if row.get("max_rss_kb") is not None else None
        # crashed_sigsegv is the authoritative signal (time.log's exit_status
        # reflects the timer's own view, not SCONE's actual exit — a SIGSEGV
        # of the timed process still gives time.log Exit status: 0).
        if row.get("crashed_sigsegv"):
            status = "SEGV"
        elif row.get("shell_exit_code") not in (0, None):
            status = "FAIL"
        else:
            status = "OK"
        print(
            f"{str(row['mesh'])[:28]:<28} "
            f"{str(row['accelerationMethod']):<12} "
            f"{fmt(row['omp_threads'], 'd'):>4} "
            f"{status:<8} "
            f"{fmt(row['wall_seconds_outer'], '.3f'):>10} "
            f"{fmt(row['user_cpu_seconds'], '.3f'):>10} "
            f"{fmt(rss_mb, '.1f'):>10} "
            f"{fmt(row['n_cells_finest'], ',d'):>14}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
