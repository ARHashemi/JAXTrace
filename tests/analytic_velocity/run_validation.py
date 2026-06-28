"""
run_validation.py
=================

Three-way Phase 3 validation:

  1. Analytic path: invokes run_tracking.py --velocity-source analytic.
     Per-step .npz output.

  2. Mesh path: invokes run_tracking.py --velocity-source mesh on a
     mesh produced by generate_test_mesh.py. Per-step VTKHDF output.

  3. Reference: scipy.integrate.solve_ivp at rtol=atol=1e-12.
     Computed in-process.

All three trajectories share the same seed positions and DT / N_STEPS.
The three errors are:

    eps_analytic = |traj_analytic - traj_reference|
        — pure RK4 truncation error.

    eps_mesh = |traj_mesh - traj_reference|
        — RK4 truncation + mesh interpolation error.

A convergence sweep over mesh resolution shows eps_mesh shrinking as
N grows, while eps_analytic is mesh-independent.

Both run_tracking.py invocations happen in fresh subprocesses to avoid
JAX OOM in this driver process. Each subprocess pays a ~10s JIT-compile
cost.

Output
------
  <output>/analytic/run_analytic_*/step_*.npz       — analytic trajectory
  <output>/mesh_<N>x<N>x<N>/                        — mesh files
  <output>/mesh_<N>x<N>x<N>/tracking/run_grid_*/particles.vtkhdf
                                                    — mesh trajectory
  <output>/summary.json                             — per-resolution errors
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Path setup
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np


# =============================================================================
# Helpers
# =============================================================================

def run_subprocess(cmd, log_file=None, env=None):
    """Run a subprocess, tee'ing to a log file. Returns rc."""
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        return proc.returncode
    return subprocess.run(cmd, env=env).returncode


def load_npz_trajectory(run_dir):
    """Load all step_*.npz files; return (steps, positions(T,N,3))."""
    files = sorted(run_dir.glob("step_*.npz"))
    if not files:
        raise FileNotFoundError(f"no step_*.npz under {run_dir}")
    arrs = []
    steps = []
    for f in files:
        d = np.load(f)
        arrs.append(d["positions"])
        steps.append(int(d["step"]))
    return np.asarray(steps), np.stack(arrs, axis=0)


def load_vtkhdf_trajectory(particles_vtkhdf):
    """Load positions from a transient VTKHDF particle file produced by
    run_tracking.py's mesh path. Returns (steps, positions(T,N,3))."""
    import h5py
    with h5py.File(particles_vtkhdf, 'r') as f:
        # See jaxtrace/io/transient_vtkhdf_writer.py for the layout.
        # Points is concatenated across timesteps in a flat (T*N, 3) array.
        all_points = f['VTKHDF/Points'][:]
        offsets = f['VTKHDF/Steps/PointOffsets'][:]
        values = f['VTKHDF/Steps/Values'][:]  # time values per step
    n_steps = len(offsets)
    # Reshape: positions[t] = all_points[offsets[t]:offsets[t]+n_points_t]
    # For a fixed particle count, the slices are equal-sized.
    # Compute n_points = total / n_steps (or use the first slice size).
    if n_steps == 1:
        n_points = all_points.shape[0]
    else:
        n_points = int(offsets[1] - offsets[0])
    traj = np.empty((n_steps, n_points, 3), dtype=np.float64)
    for t in range(n_steps):
        traj[t] = all_points[offsets[t]:offsets[t] + n_points].astype(np.float64)
    return np.asarray(values, dtype=np.float64), traj


# =============================================================================
# Workflow
# =============================================================================

def run_analytic_path(args, work_dir):
    """Subprocess: run_tracking.py --velocity-source analytic."""
    run_dir = work_dir / "analytic"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(_REPO_ROOT / "run_tracking.py"),
        "--velocity-source", "analytic",
        "--velocity-module", str(args.velocity_module),
        "--seed-source", "grid",
        "--seed-box", *map(str, args.seed_box),
        "--seed-grid", *map(str, args.seed_grid),
        "--n-steps", str(args.n_steps),
        "--dt", str(args.dt),
        "--log-interval", str(max(args.n_steps // 5, 1)),
        "--export-freq", str(args.export_freq),
        "--output", str(run_dir),
    ]
    print(f"  cmd: run_tracking.py --velocity-source analytic ...")
    rc = run_subprocess(cmd, log_file=run_dir / "log.txt")
    if rc != 0:
        raise RuntimeError(f"analytic path failed rc={rc}; see {run_dir}/log.txt")

    runs = list(run_dir.glob("run_analytic_*"))
    if not runs:
        raise FileNotFoundError(f"no run_analytic_* under {run_dir}")
    return load_npz_trajectory(runs[0])


def generate_mesh_and_run_mesh_path(args, work_dir, n_cells):
    """Subprocess 1: generate_test_mesh.py.
    Subprocess 2: run_tracking.py --velocity-source mesh.
    Loads the resulting VTKHDF, returns trajectory."""
    mesh_dir = work_dir / f"mesh_{'x'.join(map(str, n_cells))}"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate mesh.
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "tests/analytic_velocity/generate_test_mesh.py"),
        "--velocity-module", str(args.velocity_module),
        "--bbox", *map(str, args.bbox),
        "--n-cells", *map(str, n_cells),
        "--output", str(mesh_dir),
        "--stem", "mesh_0",
    ]
    print(f"  cmd: generate_test_mesh.py --n-cells {n_cells}")
    rc = run_subprocess(cmd, log_file=mesh_dir / "mesh_gen_log.txt")
    if rc != 0:
        raise RuntimeError(
            f"mesh generation rc={rc}; see {mesh_dir}/mesh_gen_log.txt"
        )

    # 2. Run mesh-path tracking.
    track_dir = mesh_dir / "tracking"
    track_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(_REPO_ROOT / "run_tracking.py"),
        "--velocity-source", "mesh",
        "--input", str(mesh_dir),
        "--mesh-subdir", "",
        "--mesh-pattern", "mesh_0.pvtu",
        "--vel-range", "0", "0",
        "--velocity-field", "Displacement",
        "--no-levelset",
        "--no-pin-velocity",
        "--seed-source", "grid",
        "--seed-box", *map(str, args.seed_box),
        "--seed-grid", *map(str, args.seed_grid),
        "--n-steps", str(args.n_steps),
        "--dt", str(args.dt),
        "--log-interval", str(max(args.n_steps // 5, 1)),
        "--export-freq", str(args.export_freq),
        "--output", str(track_dir),
    ]
    print(f"  cmd: run_tracking.py --velocity-source mesh ...")
    rc = run_subprocess(cmd, log_file=mesh_dir / "tracking_log.txt")
    if rc != 0:
        raise RuntimeError(
            f"mesh tracking rc={rc}; see {mesh_dir}/tracking_log.txt"
        )

    # Find the VTKHDF.
    candidates = sorted(track_dir.glob("run_*/particles.vtkhdf"))
    if not candidates:
        raise FileNotFoundError(
            f"no particles.vtkhdf under {track_dir}/run_*"
        )
    return load_vtkhdf_trajectory(candidates[0])


def scipy_reference(args, seed_positions):
    """scipy DOP853 reference at each seed position."""
    from jaxtrace.gpu.tracking.velocity_provider import load_analytic_provider
    provider = load_analytic_provider(
        module_path=str(args.velocity_module), domain_bbox=None, dt=args.dt,
    )
    velocity_fn = provider.velocity_fn

    import jax.numpy as jnp

    def f(t, p):
        return np.asarray(velocity_fn(jnp.asarray(p, dtype=jnp.float64)))

    from scipy.integrate import solve_ivp

    t_final = args.dt * args.n_steps
    t_eval = np.arange(0, args.n_steps + 1, args.export_freq) * args.dt

    n_seeds = len(seed_positions)
    refs = np.empty((len(t_eval), n_seeds, 3), dtype=np.float64)
    for i, p0 in enumerate(seed_positions):
        sol = solve_ivp(
            f, (0.0, t_final), np.asarray(p0, dtype=np.float64),
            method='DOP853', rtol=1e-12, atol=1e-12, t_eval=t_eval,
        )
        if not sol.success:
            raise RuntimeError(f"scipy solve_ivp failed seed {i}: {sol.message}")
        refs[:, i, :] = sol.y.T
    return t_eval / args.dt, refs


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Phase 3 mesh-vs-analytic validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--velocity-module", type=Path,
        default=_REPO_ROOT / "tests/analytic_velocity/recirculation_scaled.py",
    )
    ap.add_argument(
        "--bbox", type=float, nargs=6,
        default=[-0.0625, 0.0625, -0.03125, 0.03125, -0.00390625, 0.00390625],
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
    )
    ap.add_argument(
        "--seed-box", type=float, nargs=6,
        default=[-0.04, 0.04, -0.02, 0.02, -0.001, 0.001],
    )
    ap.add_argument(
        "--seed-grid", type=int, nargs=3, default=[4, 3, 2],
    )
    ap.add_argument(
        "--n-steps", type=int, default=20,
    )
    ap.add_argument(
        "--dt", type=float, default=0.0001,
    )
    ap.add_argument(
        "--export-freq", type=int, default=5,
    )
    ap.add_argument(
        "--mesh-resolutions", type=int, nargs="+", default=[64],
        help="Per-axis cell count for each mesh-resolution to sweep.",
    )
    ap.add_argument(
        "--output", type=Path, required=True,
        help="Working dir for all sub-runs.",
    )
    args = ap.parse_args()

    print("=" * 80)
    print("Phase 3 — analytic-vs-mesh validation")
    print("=" * 80)
    print(f"  velocity-module:   {args.velocity_module}")
    print(f"  bbox:              {args.bbox}")
    print(f"  seed-box:          {args.seed_box}")
    print(f"  seed-grid:         {args.seed_grid} = {np.prod(args.seed_grid)} particles")
    print(f"  n_steps × dt:      {args.n_steps} × {args.dt} = {args.n_steps * args.dt:g}")
    print(f"  mesh resolutions:  {args.mesh_resolutions}")
    print(f"  output:            {args.output}")
    print()

    work_dir = Path(args.output)
    work_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "velocity_module": str(args.velocity_module),
        "n_steps": args.n_steps,
        "dt": args.dt,
        "seed_grid": list(args.seed_grid),
        "n_seeds": int(np.prod(args.seed_grid)),
    }

    print("[1/3] Running analytic path...")
    t0 = time.time()
    a_steps, a_traj = run_analytic_path(args, work_dir)
    summary["analytic_wall_time"] = time.time() - t0
    print(f"  trajectory shape: {a_traj.shape}  ({summary['analytic_wall_time']:.1f}s)")
    print()

    seed_pos = a_traj[0]  # (N, 3) — same seeds for all paths

    print("[2/3] Computing scipy DOP853 reference...")
    t0 = time.time()
    s_steps, s_traj = scipy_reference(args, seed_pos)
    summary["scipy_wall_time"] = time.time() - t0
    print(f"  trajectory shape: {s_traj.shape}  ({summary['scipy_wall_time']:.1f}s)")
    print()

    # Analytic vs scipy
    a_err = np.linalg.norm(a_traj - s_traj, axis=-1)  # (T, N)
    summary["analytic_max_err_vs_scipy"] = float(a_err.max())
    summary["analytic_rms_err_vs_scipy"] = float(np.sqrt((a_err**2).mean()))
    print(f"  analytic vs scipy: max={summary['analytic_max_err_vs_scipy']:.3e}, "
          f"rms={summary['analytic_rms_err_vs_scipy']:.3e}")
    print()

    print("[3/3] Mesh paths at each resolution...")
    mesh_results = []
    for n_axis in args.mesh_resolutions:
        n_cells = (n_axis, n_axis // 2, max(n_axis // 8, 2))
        print(f"  -> n_cells = {n_cells}")
        t0 = time.time()
        m_steps, m_traj = generate_mesh_and_run_mesh_path(args, work_dir, n_cells)
        wt = time.time() - t0

        # Sanity: same number of timesteps and particles?
        print(f"     mesh trajectory shape: {m_traj.shape}  ({wt:.1f}s)")

        m_err = np.linalg.norm(m_traj - s_traj, axis=-1)
        diff = float(m_err.max() - a_err.max())
        print(f"     mesh vs scipy:    max={m_err.max():.3e}, rms={np.sqrt((m_err**2).mean()):.3e}")
        print(f"     mesh - analytic:  {diff:.3e}  <- interpolation-error contribution")
        mesh_results.append({
            "n_cells": list(n_cells),
            "wall_time": wt,
            "max_err_vs_scipy": float(m_err.max()),
            "rms_err_vs_scipy": float(np.sqrt((m_err**2).mean())),
            "max_minus_analytic": diff,
        })
    summary["mesh_results"] = mesh_results

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  analytic:        max={summary['analytic_max_err_vs_scipy']:.3e},  "
          f"rms={summary['analytic_rms_err_vs_scipy']:.3e}")
    for r in mesh_results:
        print(f"  mesh {r['n_cells']}: max={r['max_err_vs_scipy']:.3e}, "
              f"rms={r['rms_err_vs_scipy']:.3e}  (Δmax={r['max_minus_analytic']:+.3e})")

    summary_path = work_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
