#!/usr/bin/env python3
"""
Parse a RTXAdvect run.log into a result.json schema that merges cleanly
with malmo_runs_in_mesh/*/result.json for joint comparison tables.

Usage:
    parse_rtxadvect_log.py --log <run.log> --meta <mesh.meta.json> \
                            --out <result.json> \
                            [--n-particles 100000] [--wall-seconds 3.4]
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path


def _find(pat, text, cast=str, default=None, flags=0):
    m = re.search(pat, text, flags)
    if not m:
        return default
    try:
        return cast(m.group(1))
    except Exception:
        return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--meta", required=True, help="mesh .meta.json produced by vtu_to_rtxadvect.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-particles", type=int, required=True)
    ap.add_argument("--n-steps", type=int, required=True)
    ap.add_argument("--wall-seconds", type=float, required=True, help="Outer wall time (from `time` or driver)")
    ap.add_argument("--max-rss-kb", type=int, default=None)
    a = ap.parse_args()

    log = Path(a.log).read_text(errors="replace")
    meta = json.load(open(a.meta))

    # --- Mesh loading ---
    n_verts_reported  = _find(r"NumTetVerts\s+(\d+)", log, int)
    n_cells_reported  = _find(r"NumTetCells\s+(\d+)", log, int)

    # --- BVH init + Init RunTime (both are "init" — one for BVH-only,
    #     other includes CPU-side particle allocation) ---
    bvh_ms       = _find(r"BVH Construction Time=([0-9.eE+]+)\s*ms", log, float)
    init_ms      = _find(r"Init RunTime=([0-9.eE+]+)\s*ms", log, float)

    # --- Simulation stats ---
    sim_ms       = _find(r"Simulation RunTime=([0-9.eE+]+)\s*ms", log, float)
    steps_per_s  = _find(r"Simulation Performance=([0-9.eE+]+)\s*steps/secs", log, float)
    n_out        = _find(r"Out-of-domain particles\(-tetID\)\s*=\s*(\d+)", log, int)

    # RTXAdvect's own item-wise breakdown (last one printed)
    breakdown = {}
    for label, key in [("BVH init", "bvh_init_s"), ("Adv", "advection_s"),
                       ("Dfs", "diffusion_s"), ("Qry", "query_s"),
                       ("Rft", "reflection_s"), ("Mov", "move_s"),
                       ("IO", "io_s")]:
        v = _find(rf"^\s*{re.escape(label)}\s+([0-9.eE+]+)", log, float, flags=re.M)
        if v is not None:
            breakdown[key] = v

    total_ms = _find(r"Total Time = ([0-9.eE+]+)\s*ms", log, float)
    perf     = _find(r"Performance = ([0-9.eE+]+)\s*steps/secs", log, float)

    # Compute a query-throughput-equivalent number: (n_particles) / (Qry time)
    # If Qry time is 0 (below timer resolution) we fall back to total/n_steps.
    tput_mqps = None
    if breakdown.get("query_s") and breakdown["query_s"] > 0:
        tput_mqps = (a.n_particles / breakdown["query_s"]) / 1e6
    elif steps_per_s and steps_per_s > 0:
        # Approximation: (particles/step) * (steps/s) = particles/s
        tput_mqps = (a.n_particles * steps_per_s) / 1e6

    # correct_rate_strict — RTXAdvect doesn't compare to a ground-truth ID
    # for random-in-box seeding, but if 100% of particles are in-domain the
    # host-cell locator succeeded for all of them.
    found_rate = None
    if n_out is not None:
        found_rate = float(1.0 - (n_out / a.n_particles))

    out = {
        "framework": "RTXAdvect",
        "method": "RTX_BVH",  # RT-cores + BVH host-cell locator
        "mesh": meta.get("vtu", "").split("/")[-1].replace(".vtu", "") or Path(a.meta).stem,
        "vtu_path": meta.get("vtu"),
        "n_tets": meta.get("n_tets"),
        "n_vertices": meta.get("n_vertices"),
        "n_verts_reported": n_verts_reported,
        "n_cells_reported": n_cells_reported,
        "n_points_queried": a.n_particles,
        "n_steps": a.n_steps,
        "seeding_box": meta.get("seeding_box_flag"),
        "found_rate": found_rate,
        "n_out_of_domain": n_out,
        "correct_rate_strict": None,  # not comparable — see notes
        "wall_seconds": {
            "outer": a.wall_seconds,
            "bvh_ms": bvh_ms,
            "init_ms": init_ms,
            "simulation_ms": sim_ms,
            "total_ms": total_ms,
        },
        "wall_seconds_breakdown_s": breakdown,
        "steps_per_second": steps_per_s or perf,
        "throughput_Mqps": tput_mqps,
        "process": {
            "max_rss_kb": a.max_rss_kb,
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
