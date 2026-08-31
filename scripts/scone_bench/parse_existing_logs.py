#!/usr/bin/env python3
"""
Post-hoc parser: walk every run dir under scone_runs/ that has stdout.log +
input.txt but is missing result.json, and reconstruct result.json from what
already lies on disk. Useful when the driver crashed after SCONE but before
JSON emission (e.g. the 'set -e' bug we hit on 2026-08-25).

No SCONE reruns. Just parsing + writing.

Usage:
    scripts/scone_bench/parse_existing_logs.py --root ./scone_runs
    scripts/scone_bench/parse_existing_logs.py --root ./scone_runs --force

--force overwrites already-present result.json files.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_int_list_line(line: str):
    m = re.match(r"\s*Grid size in xyz for (?:coarsest|finest)\s*:\s*([-0-9. \t]+)", line)
    if not m:
        return None
    try:
        return [int(x) for x in m.group(1).split()]
    except Exception:
        return None


def find(pat, txt, cast=str, default=None, flags=0):
    m = re.search(pat, txt, flags)
    if not m:
        return default
    try:
        return cast(m.group(1))
    except Exception:
        return default


def parse_elapsed(s):
    if not s:
        return None
    try:
        parts = [float(p) for p in s.split(":")]
    except Exception:
        return None
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 1:
        return parts[0]
    return None


def tfind(tlog, key, cast=float, default=None):
    m = re.search(rf"^\s*{re.escape(key)}:\s*(.+)$", tlog, re.M)
    if not m:
        return default
    v = m.group(1).strip()
    try:
        return cast(v)
    except Exception:
        return v


def parse_input_for_meta(inp_text: str):
    """Extract accelMethod, box halfwidth, and fills count from the SCONE input."""
    meta = {}
    m = re.search(r"accelerationMethod\s+(\w+)", inp_text)
    if m:
        meta["accelerationMethod"] = m.group(1)
    m = re.search(r"fills\s*\(\s*([^)]+)\)", inp_text)
    if m:
        meta["n_zones"] = len(m.group(1).split())
    m = re.search(r"pop\s*[=\s]\s*(\d+)", inp_text)
    if m:
        meta["pop"] = int(m.group(1))
    m = re.search(r"cycles\s*[=\s]\s*(\d+)", inp_text)
    if m:
        meta["cycles"] = int(m.group(1))
    return meta


def polymesh_dir_from_symlink(rundir: Path) -> str:
    m = rundir / "m"
    if m.is_symlink():
        return str(m.resolve())
    return ""


def mesh_and_method_from_dirname(name: str):
    # <mesh>__<method>__omp<N>
    if "__" not in name:
        return name, None, None
    parts = name.split("__")
    if len(parts) < 3:
        return parts[0], parts[1] if len(parts) > 1 else None, None
    mesh = parts[0]
    method = parts[1]
    omp = None
    m = re.match(r"omp(\d+)$", parts[2])
    if m:
        omp = int(m.group(1))
    return mesh, method, omp


def reconstruct(rundir: Path) -> dict:
    log_path = rundir / "stdout.log"
    tlog_path = rundir / "time.log"
    inp_path = rundir / "input.txt"

    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    tlog = tlog_path.read_text(errors="replace") if tlog_path.exists() else ""
    inp = inp_path.read_text(errors="replace") if inp_path.exists() else ""

    mesh, method, omp = mesh_and_method_from_dirname(rundir.name)
    input_meta = parse_input_for_meta(inp)

    # Grid diagnostics (Cartesian banner)
    grid = {
        "min_angle":        find(r"Minimum angle\s*:\s*([-0-9.Ee+]+)", log, float),
        "min_edge_length":  find(r"Minimum edge length\s*:\s*([-0-9.Ee+]+)", log, float),
        "n_vertices":       find(r"No\. of vertices\s*:\s*([0-9]+)", log, int),
        "n_edges":          find(r"No\. of edges\s*:\s*([0-9]+)", log, int),
        "n_faces":          find(r"No\. of faces\s*:\s*([0-9]+)", log, int),
        "n_elements":       find(r"No\. of elements\s*:\s*([0-9]+)", log, int),
        "n_layers":         find(r"No\. of layers\s*:\s*([0-9]+)", log, int),
        "grid_spacing_coarsest": find(r"Grid spacing for coarsest\s*:\s*([-0-9.Ee+]+)", log, float),
        "grid_spacing_finest":   find(r"Grid spacing for finest\s*:\s*([-0-9.Ee+]+)", log, float),
        "grid_spacing_single":   find(r"^ Grid spacing\s*:\s*([-0-9.Ee+]+)", log, float, flags=re.M),
        "grid_size_x":           find(r"Grid size in x\s*:\s*([0-9]+)", log, int),
        "grid_size_y":           find(r"Grid size in y\s*:\s*([0-9]+)", log, int),
        "grid_size_z":           find(r"Grid size in z\s*:\s*([0-9]+)", log, int),
    }
    xyz_coarsest = None
    xyz_finest = None
    for line in log.splitlines():
        m = re.match(r"\s*Grid size in xyz for coarsest\s*:\s*([-0-9. \t]+)", line)
        if m:
            try: xyz_coarsest = [int(x) for x in m.group(1).split()]
            except Exception: pass
        m = re.match(r"\s*Grid size in xyz for finest\s*:\s*([-0-9. \t]+)", line)
        if m:
            try: xyz_finest = [int(x) for x in m.group(1).split()]
            except Exception: pass
    grid["grid_xyz_coarsest"] = xyz_coarsest
    grid["grid_xyz_finest"] = xyz_finest
    grid["n_cells_coarsest"] = (xyz_coarsest[0] * xyz_coarsest[1] * xyz_coarsest[2]) if xyz_coarsest and len(xyz_coarsest) == 3 else None
    grid["n_cells_finest"] = (xyz_finest[0] * xyz_finest[1] * xyz_finest[2]) if xyz_finest and len(xyz_finest) == 3 else None

    main_timer_s = find(r"Main Timer[^0-9]*([0-9.Ee+]+)\s*s", log, float)
    cycle_wall = [float(x) for x in re.findall(r"Cycle\s+\d+.*?wall\s*=\s*([0-9.Ee+]+)", log)]

    elapsed_raw = tfind(tlog, "Elapsed (wall clock) time (h:mm:ss or m:ss)", str, None)
    time_metrics = {
        "user_cpu_seconds": tfind(tlog, "User time (seconds)", float),
        "sys_cpu_seconds":  tfind(tlog, "System time (seconds)", float),
        "elapsed_seconds":  parse_elapsed(elapsed_raw),
        "percent_cpu":      tfind(tlog, "Percent of CPU this job got", str),
        "max_rss_kb":       tfind(tlog, "Maximum resident set size (kbytes)", int),
        "voluntary_ctx_switches":   tfind(tlog, "Voluntary context switches", int),
        "involuntary_ctx_switches": tfind(tlog, "Involuntary context switches", int),
        "exit_status":              tfind(tlog, "Exit status", int),
    }

    # Was there a segfault?
    crashed = bool(re.search(r"SIGSEGV|Segmentation fault", log))
    # Was there a SCONE fatal error?
    scone_fatal = None
    m = re.search(r"Fatal has occurred in:\s*\n\s*(.+?)\n\s*Because:\s*\n\s*(.+?)\n", log)
    if m:
        scone_fatal = {"location": m.group(1).strip(), "reason": m.group(2).strip()}
    # exit code inferred from time.log or from presence of Fatal
    if time_metrics["exit_status"] is not None:
        shell_exit_code = int(time_metrics["exit_status"])
    elif crashed:
        shell_exit_code = 139
    elif scone_fatal is not None:
        shell_exit_code = 1
    else:
        shell_exit_code = 0

    polymesh = polymesh_dir_from_symlink(rundir)

    out = {
        "mesh": mesh,
        "polymesh_dir": polymesh,
        "accelerationMethod": input_meta.get("accelerationMethod") or method,
        "n_zones": input_meta.get("n_zones"),
        "pop": input_meta.get("pop"),
        "cycles": input_meta.get("cycles"),
        "omp_threads": omp,
        "shell_exit_code": shell_exit_code,
        "crashed_sigsegv": bool(crashed),
        "scone_fatal": scone_fatal,
        "wall_seconds_outer": time_metrics.get("elapsed_seconds"),
        "main_timer_seconds": main_timer_s,
        "cycle_wall_seconds": cycle_wall,
        "grid": grid,
        "process": time_metrics,
        "reconstructed_from_logs": True,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Reconstruct missing result.json from existing logs")
    ap.add_argument("--root", default="./scone_runs")
    ap.add_argument("--force", action="store_true", help="Overwrite existing result.json files")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 2

    n_new, n_kept, n_skip = 0, 0, 0
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        result_json = d / "result.json"
        if result_json.exists() and not args.force:
            n_kept += 1
            continue
        if not (d / "stdout.log").exists():
            n_skip += 1
            continue
        try:
            out = reconstruct(d)
            result_json.write_text(json.dumps(out, indent=2, default=str))
            n_new += 1
            print(f"wrote {result_json}  (exit={out['shell_exit_code']}, "
                  f"crashed={out['crashed_sigsegv']}, "
                  f"cells_finest={out['grid'].get('n_cells_finest')})")
        except Exception as e:
            print(f"WARN: {d}: {e}", file=sys.stderr)
            n_skip += 1

    print(f"\nsummary: wrote {n_new} new result.json, kept {n_kept} existing, skipped {n_skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
