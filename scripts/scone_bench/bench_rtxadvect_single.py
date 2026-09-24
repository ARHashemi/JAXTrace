#!/usr/bin/env python3
"""
Run RTXAdvect on ONE mesh, time the outer wall, capture max RSS,
and emit a result.json compatible with the joint table builder.

Usage:
    bench_rtxadvect_single.py --exe <cudaParticleAdvection> \
                              --meta <mesh.meta.json> \
                              --out-dir <run/> \
                              --n-particles 100000 --n-steps 1 \
                              [--dt 1e-3]
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-particles", type=int, default=100_000)
    ap.add_argument("--n-steps", type=int, default=1)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--parser", required=True,
                    help="Path to parse_rtxadvect_log.py")
    ap.add_argument("--particles", default=None,
                    help="Optional particle-seed file (RTXAdvect --input-particles). "
                         "When given, replaces --seeding-box so the query "
                         "distribution matches MALMO's in-mesh barycentric "
                         "sampler. Produced by make_inmesh_particles.py.")
    a = ap.parse_args()

    meta = json.load(open(a.meta))
    sbox = meta["seeding_box_flag"].split()
    verts = meta["verts_dat"]
    cells = meta["cells_dat"]
    zvel  = meta["zero_velocity_dat"]

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"
    result = out / "result.json"

    cmd = [
        "/usr/bin/time", "-v",
        a.exe,
        "--num-particles", str(a.n_particles),
        "--num-steps",     str(a.n_steps),
        "--input_mesh", verts, cells,
        "--input_tet_velocity_field", zvel,
        "-dt", str(a.dt),
    ]
    # Seeding: in-mesh particle file when supplied, else bbox-uniform.
    if a.particles:
        cmd += ["--input-particles", a.particles]
    else:
        cmd += ["--seeding-box", *sbox]
    cmd += ["--save-streamline-to-vtk", str(out / "streamline.vtk")]
    t0 = time.time()
    with open(log, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=out)
    t1 = time.time()
    wall = t1 - t0

    # Parse `time -v` max RSS
    max_rss = None
    for line in log.read_text().splitlines():
        line = line.strip()
        if line.startswith("Maximum resident set size"):
            try:
                max_rss = int(line.split(":")[1].strip())
            except Exception:
                pass

    parse_cmd = [
        sys.executable, a.parser,
        "--log", str(log),
        "--meta", a.meta,
        "--out", str(result),
        "--n-particles", str(a.n_particles),
        "--n-steps", str(a.n_steps),
        "--wall-seconds", f"{wall:.6f}",
    ]
    if max_rss is not None:
        parse_cmd += ["--max-rss-kb", str(max_rss)]
    subprocess.run(parse_cmd, check=True)

    # Also stamp exit code
    d = json.load(open(result))
    d["shell_exit_code"] = proc.returncode
    Path(result).write_text(json.dumps(d, indent=2, default=str))

    print(f"[{meta.get('vtu', a.meta)}] "
          f"ec={proc.returncode} wall={wall:.3f}s found={d.get('found_rate')} "
          f"tput={d.get('throughput_Mqps')}Mqps")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
