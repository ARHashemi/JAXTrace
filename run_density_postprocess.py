#!/usr/bin/env python3
"""
Offline density-field post-processor for a JAXTrace particles.vtkhdf trajectory.

Examples
--------

Minimal — uniform voxel grid sized by particle-bbox union from a 2-pass read::

    python run_density_postprocess.py \\
        --particles /path/to/particles.vtkhdf \\
        --output-dir density_out \\
        --resolution 128 \\
        --kernel wendland_c2

With explicit bounds and adaptive bandwidth::

    python run_density_postprocess.py \\
        --particles /path/to/particles.vtkhdf \\
        --output-dir density_out \\
        --bounds -0.05 0.05 -0.05 0.05 0.0 0.2 \\
        --voxel-size 0.001 \\
        --kernel wendland_c2 \\
        --bandwidth-mode knn_adaptive --knn-k 32

With inside-mesh masking from a velocity .pvtu / .pvd::

    python run_density_postprocess.py \\
        --particles /path/to/particles.vtkhdf \\
        --velocity-mesh /path/to/velocity.pvd \\
        --output-dir density_out

This script is self-contained: it does not modify the JAXTrace tracking
pipeline and does not require a GPU mesh upload unless --velocity-mesh is
given for inside-mesh masking.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--particles", required=True, type=Path,
                   help="Path to particles.vtkhdf written by jaxtrace.io.vtkhdf_writer.")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--output-format", choices=["vtkhdf", "vti"], default="vtkhdf")
    p.add_argument("--filename-stem", default="density")

    # Bounds
    g = p.add_argument_group("Voxel grid bounds")
    g.add_argument("--bounds", type=float, nargs=6, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                   default=None, help="Explicit voxel grid bounds. Default: union over the trajectory (2-pass).")
    g.add_argument("--bounds-from", choices=["explicit", "prepass", "mesh"], default=None,
                   help="Override the default bounds source.")
    g.add_argument("--pad-fraction", type=float, default=0.0)

    # Resolution
    p.add_argument("--resolution", type=int, default=128,
                   help="Cubic voxel-grid resolution; ignored if --voxel-size is set.")
    p.add_argument("--voxel-size", type=float, default=None,
                   help="Target physical voxel edge length. Overrides --resolution.")

    # Kernel / bandwidth
    p.add_argument("--kernel", default="wendland_c2",
                   choices=["wendland_c2", "wendland_c4", "cubic_spline",
                            "gaussian", "epanechnikov", "quintic_spline"])
    p.add_argument("--bandwidth-mode", default="fixed",
                   choices=["fixed", "scott", "silverman", "knn_adaptive"])
    p.add_argument("--bandwidth", type=float, default=None,
                   help="Fixed bandwidth (only if --bandwidth-mode fixed). "
                        "Default = bandwidth-factor * voxel_size.")
    p.add_argument("--bandwidth-factor", type=float, default=2.0)
    p.add_argument("--bandwidth-refresh-every", type=int, default=0,
                   help="Recompute bandwidth every N steps. 0 = once (default).")
    p.add_argument("--knn-k", type=int, default=32)
    p.add_argument("--knn-safety", type=float, default=1.2)

    # Engine
    p.add_argument("--engine", choices=["auto", "brute", "octree"], default="auto")
    p.add_argument("--auto-threshold", type=float, default=1e10)
    p.add_argument("--brute-query-chunk", type=int, default=8192)
    p.add_argument("--octree-target-n-per-cell", type=int, default=9,
                   help="Backend P (particle-hash octree) target average "
                        "particles per cell. Lower => more, smaller cells + "
                        "larger stencils; higher => fewer, bigger cells + "
                        "smaller stencils. 9 is a balanced default.")
    # Deprecated knobs accepted for back-compat; ignored at runtime.
    p.add_argument("--octree-cells-per-dim", type=int, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--octree-max-neighbors", type=int, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--particle-bucket", type=int, default=4096)

    # Output toggles
    p.add_argument("--no-per-step", action="store_true",
                   help="Skip per-step grid output; only write time-average.")
    p.add_argument("--no-time-average", action="store_true",
                   help="Skip the time-average output.")
    p.add_argument("--no-particle-density", action="store_true",
                   help="Skip per-particle density samples.")
    p.add_argument("--normalization", choices=["pdf", "mass", "unnormalized"], default="pdf")

    # Mesh (inside-mesh masking)
    p.add_argument("--velocity-mesh", type=Path, default=None,
                   help="Optional .pvd/.pvtu/.vtu/.vtk to compute inside-mesh mask.")
    p.add_argument("--no-mask-inside-mesh", action="store_true")

    # Step subsampling
    p.add_argument("--step-stride", type=int, default=1,
                   help="Process every Nth step from the trajectory (default 1 = all).")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Process at most this many steps (after stride).")
    p.add_argument("--step-range", type=int, nargs=2, default=None,
                   metavar=("START", "END"),
                   help="Process steps in [START, END) only.")
    p.add_argument("--step-tail", type=int, default=None,
                   help="Process the LAST N steps. Overrides --step-range/--max-steps when set.")

    # Compression
    p.add_argument("--compression", default="gzip",
                   choices=["gzip", "lzf", "blosc", "none"],
                   help="HDF5 compression filter (default gzip — only filter "
                        "that ParaView's bundled vtkhdf5 can decompress).")
    p.add_argument("--compression-opts", type=int, default=1)
    p.add_argument("--blosc-threads", type=int, default=4)

    # Prefetch
    p.add_argument("--read-prefetch", type=int, default=4,
                   help="Background trajectory reader queue depth. 0 disables prefetch.")

    args = p.parse_args()
    if args.octree_cells_per_dim is not None or args.octree_max_neighbors is not None:
        print("[density-postprocess] note: --octree-cells-per-dim and "
              "--octree-max-neighbors are deprecated and ignored; the "
              "particle-hash backend now auto-sizes cells from "
              "--octree-target-n-per-cell.")
    return args


def _resolve_bounds(args) -> Tuple[Optional[Tuple[Tuple[float, float], ...]], str]:
    """Return (bounds_tuple_or_None, bounds_mode)."""
    if args.bounds is not None:
        bb = args.bounds
        return ((bb[0], bb[1]), (bb[2], bb[3]), (bb[4], bb[5])), "explicit"
    if args.bounds_from == "mesh":
        return None, "mesh"
    return None, "prepass"  # need pre-pass


def _derive_pvtu_pattern(mesh_pvtu: Path) -> tuple[Path, str, int]:
    """
    From a single .pvtu path like ``/.../post/cylindrical_119.pvtu``, derive
    ``(base_path, file_pattern, timestep)`` arguments suitable for
    :func:`jaxtrace.gpu.mesh_loader_timedep.load_velocity_sequence_from_pvtu`.

    The trailing integer in the file stem is treated as the timestep index;
    the rest of the stem becomes the case stem and the pattern is reconstructed
    as ``"<stem>_{timestep}.pvtu"``.
    """
    import re

    mesh_pvtu = Path(mesh_pvtu)
    if not mesh_pvtu.is_file():
        raise FileNotFoundError(f"velocity mesh file not found: {mesh_pvtu}")

    stem = mesh_pvtu.stem  # "cylindrical_119"
    m = re.match(r"^(?P<stem>.+?)_(?P<ts>\d+)$", stem)
    if not m:
        raise ValueError(
            f"Cannot parse '<stem>_<timestep>.pvtu' from {mesh_pvtu.name}. "
            f"Expected something like 'cylindrical_119.pvtu'."
        )
    case_stem = m.group("stem")
    timestep = int(m.group("ts"))
    pattern = f"{case_stem}_{{timestep}}.pvtu"
    return mesh_pvtu.parent, pattern, timestep


def _load_velocity_mesh_octree(mesh_path: Path):
    """Load mesh topology from a single PVTU file and build the GPU
    mesh-aligned octree. Only one timestep is read (the one named in the
    file), which is all we need for inside-mesh masking.

    Mirrors the post-load steps in run_tracking.py.
    """
    from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
    from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
    from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import (
        extract_octree_cells_parent_cube,
    )
    from jaxtrace.gpu.search.mesh_aligned_octree_gpu import (
        upload_mesh_aligned_octree_to_gpu,
    )

    base_path, pattern, ts = _derive_pvtu_pattern(mesh_path)
    print(f"[density-postprocess] loading velocity mesh:")
    print(f"  base_path={base_path}  pattern='{pattern}'  timestep={ts}")
    node_positions, connectivity, _velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=base_path,
        file_pattern=pattern,
        timestep_range=(ts, ts),
        verbose=False,
    )
    node_positions, connectivity, n_dup, _ = deduplicate_nodes(
        node_positions, connectivity, None,
    )
    connectivity = connectivity.astype(np.int32)
    print(f"  nodes: {node_positions.shape[0]:,}  elements: {connectivity.shape[0]:,}  dedup: {n_dup:,}")

    cells = extract_octree_cells_parent_cube(
        node_positions, connectivity, tolerance=1e-6, verbose=False,
    )
    octree_gpu = upload_mesh_aligned_octree_to_gpu(
        connectivity, node_positions, cells, verbose=False,
    )
    mesh_bbox_min = node_positions.min(axis=0).astype(np.float32)
    mesh_bbox_max = node_positions.max(axis=0).astype(np.float32)
    return octree_gpu, mesh_bbox_min, mesh_bbox_max


def main() -> int:
    args = parse_args()

    from jaxtrace.density import (
        DensityRunner, DensityRunnerConfig,
        trajectory_bbox_union_from_vtkhdf,
        iterate_vtkhdf_steps, prefetch_vtkhdf_steps,
    )

    # Resolve bounds source
    bounds_tuple, bounds_mode = _resolve_bounds(args)

    # Optionally load velocity mesh
    mesh_octree_gpu = None
    mesh_bbox_min = None
    mesh_bbox_max = None
    if args.velocity_mesh is not None and not args.no_mask_inside_mesh:
        mesh_octree_gpu, mesh_bbox_min, mesh_bbox_max = _load_velocity_mesh_octree(args.velocity_mesh)
        if bounds_mode == "mesh":
            bounds_tuple = ((float(mesh_bbox_min[0]), float(mesh_bbox_max[0])),
                            (float(mesh_bbox_min[1]), float(mesh_bbox_max[1])),
                            (float(mesh_bbox_min[2]), float(mesh_bbox_max[2])))
            bounds_mode = "explicit"

    if bounds_mode == "mesh" and mesh_octree_gpu is None:
        raise SystemExit("--bounds-from mesh requires --velocity-mesh")

    # Pre-pass if needed
    if bounds_mode == "prepass":
        t0 = time.time()
        print(f"[density-postprocess] computing particle-bbox union from {args.particles}")
        lo, hi = trajectory_bbox_union_from_vtkhdf(str(args.particles))
        bounds_tuple = ((float(lo[0]), float(hi[0])),
                        (float(lo[1]), float(hi[1])),
                        (float(lo[2]), float(hi[2])))
        print(f"  bbox = {bounds_tuple}   ({time.time() - t0:.2f}s)")

    # Build runner config
    cfg = DensityRunnerConfig(
        bounds_mode="explicit",
        bounds=bounds_tuple,
        resolution=None if args.voxel_size is not None else args.resolution,
        voxel_size=args.voxel_size,
        pad_fraction=args.pad_fraction,
        mask_inside_mesh=(mesh_octree_gpu is not None) and (not args.no_mask_inside_mesh),
        kernel=args.kernel,
        bandwidth_mode=args.bandwidth_mode,
        bandwidth=args.bandwidth,
        bandwidth_factor=args.bandwidth_factor,
        bandwidth_refresh_every=args.bandwidth_refresh_every,
        knn_k=args.knn_k,
        knn_safety=args.knn_safety,
        normalization=args.normalization,
        engine=args.engine,
        auto_threshold=args.auto_threshold,
        brute_query_chunk=args.brute_query_chunk,
        octree_target_n_per_cell=args.octree_target_n_per_cell,
        particle_bucket=args.particle_bucket,
        eval_on_grid=True,
        eval_at_particles=(not args.no_particle_density),
        write_per_step=(not args.no_per_step),
        write_time_average=(not args.no_time_average),
        output_format=args.output_format,
        output_dir=str(args.output_dir),
        filename_stem=args.filename_stem,
        compression=args.compression,
        compression_opts=args.compression_opts,
        blosc_threads=args.blosc_threads,
    )

    runner = DensityRunner(
        cfg=cfg,
        mesh_octree_gpu=mesh_octree_gpu,
        mesh_bbox_min=mesh_bbox_min,
        mesh_bbox_max=mesh_bbox_max,
    )

    # Resolve which step indices to actually process.
    import h5py
    with h5py.File(str(args.particles), "r") as _f:
        n_steps_total = int(_f["/VTKHDF/Steps"].attrs["NSteps"])

    if args.step_tail is not None:
        n = int(args.step_tail)
        step_indices = list(range(max(0, n_steps_total - n), n_steps_total))
    elif args.step_range is not None:
        s, e = args.step_range
        step_indices = list(range(max(0, int(s)), min(n_steps_total, int(e))))
    else:
        step_indices = list(range(n_steps_total))
    # Apply stride
    if args.step_stride > 1:
        step_indices = step_indices[::args.step_stride]
    # Cap by max_steps
    if args.max_steps is not None:
        step_indices = step_indices[: int(args.max_steps)]
    print(f"[density-postprocess] will process {len(step_indices)} of {n_steps_total} steps "
          f"(first={step_indices[0] if step_indices else None}, last={step_indices[-1] if step_indices else None})")

    # Iterate trajectory (with optional reader prefetch).
    import jax.numpy as jnp
    if args.read_prefetch and args.read_prefetch > 0:
        step_iter = prefetch_vtkhdf_steps(
            str(args.particles),
            step_indices=step_indices,
            prefetch=int(args.read_prefetch),
        )
    else:
        step_iter = iterate_vtkhdf_steps(str(args.particles), step_indices=step_indices)

    prev_t: Optional[float] = None
    processed = 0
    total_t0 = time.time()
    for step, t, positions_np in step_iter:
        # dt = current - previous, or fall back to a unit step for the very first
        if prev_t is None:
            dt = 0.0  # first step contributes only via peak/coverage with dt==0; skip accum
        else:
            dt = float(t - prev_t)
        prev_t = float(t)

        positions = jnp.asarray(positions_np, dtype=jnp.float32)
        runner.step(positions, dt=dt, time_value=float(t), step_index=step)

        processed += 1
        if processed % 25 == 0 or processed == 1:
            print(f"  step={step:6d}  t={t:.4g}  N={positions_np.shape[0]:,}  "
                  f"elapsed={time.time() - total_t0:.1f}s")

    runner.close()
    print(f"[density-postprocess] done. processed {processed} steps in {time.time() - total_t0:.1f}s")
    print(f"  output dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
