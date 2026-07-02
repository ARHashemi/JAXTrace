#!/usr/bin/env python3
"""
JAXTrace production particle tracking driver.

Thin runner built on top of ``benchmark_femuss_comparison.py``: it reuses the
exact mesh-loading, deduplication, octree, RK4, and VTU-export helpers, but
exposes them as a general-purpose production script that runs on LUMI or on a
local workstation without requiring a FEMUSS reference dataset.

Key differences vs. ``benchmark_femuss_comparison.py``:
  * FEMUSS comparison is **optional** — enabled only via ``--femuss-compare``.
    If the FEMUSS reference file is missing, the run proceeds (warning printed)
    instead of aborting.
  * Three particle seeding modes:
      - ``--seed-source femuss``  (default when ``--femuss-compare`` is set)
        Load initial particle positions from a FEMUSS PVTU at ``--femuss-start``.
      - ``--seed-source box``
        Uniform random seeding inside an axis-aligned box.
      - ``--seed-source file``
        Load positions from a user-supplied ``.npy`` / ``.npz`` file.
  * All CLI options from the benchmark are preserved; defaults are unchanged.

Usage
-----
# Basic run without FEMUSS (random seeding in an absolute box)
python run_tracking.py \\
    --input /path/to/data \\
    --output /path/to/output \\
    --seed-source box --seed-box -0.01 0.01 -0.005 0.005 0.0 0.002 \\
    --n-particles 100000 --n-steps 2684

# Gridded seeding in an absolute box (count = 50*70*30 = 105000)
python run_tracking.py \\
    --input /path/to/data --output /path/to/output \\
    --seed-source grid --seed-box -0.01 0.01 -0.005 0.005 0.0 0.002 \\
    --seed-grid 50 70 30 --n-steps 2684

# Gridded seeding in the first 20% of X (full Y/Z) of the mesh bbox
python run_tracking.py \\
    --input /path/to/data --output /path/to/output \\
    --seed-source grid-frac --seed-fraction 0.0 0.2 0.0 1.0 0.0 1.0 \\
    --seed-grid 50 70 30 --n-steps 2684

# Same fractional region but random sampling
python run_tracking.py \\
    --input /path/to/data --output /path/to/output \\
    --seed-source box-frac --seed-fraction 0.0 0.2 0.0 1.0 0.0 1.0 \\
    --n-particles 100000 --n-steps 2684

# FEMUSS-style run (same as benchmark_femuss_comparison.py)
python run_tracking.py \\
    --input /path/to/data --output /path/to/output \\
    --seed-source femuss --femuss-start 0 \\
    --femuss-compare --n-steps 2684
"""
from __future__ import annotations

import os
# Use setdefault so the caller (e.g. run_lumi.sh BENCHMARK_MODE) can override.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('JAX_PLATFORMS', 'cuda,rocm,cpu')

import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Pre-parse --precision before any JAX import.
_precision_arg = 'float32'
for i, arg in enumerate(sys.argv):
    if arg == '--precision' and i + 1 < len(sys.argv):
        _precision_arg = sys.argv[i + 1]
        break
    elif arg.startswith('--precision='):
        _precision_arg = arg.split('=', 1)[1]
        break

import jaxtrace.config as config
config.set_precision(_precision_arg == 'float64')

import jax
import jax.numpy as jnp

# Reuse helpers from benchmark_femuss_comparison (kept as the source of truth
# until this logic is merged into jaxtrace core).
from jaxtrace.io.vtkhdf_writer import VTKHDFExportThread

from benchmark_femuss_comparison import (
    load_femuss_particles,
    reconstruct_pin_velocity,
    create_rk4_comparison,
    compare_with_femuss,
    VTKExportThread,
    VELOCITY_FIELD_NAME,
    LEVELSET_FIELD_NAME,
    ENABLE_L0_SEARCH,
    ENABLE_L1_SEARCH,
    N_HOPS,
    L2_METHOD,
    POINT_IN_TET_METHOD,
)

from jaxtrace.gpu.mesh_loader_timedep import (
    load_velocity_sequence_from_pvtu,
    load_scalar_sequence_from_pvtu,
)
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import extract_octree_cells_vertex_multi
from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import extract_octree_cells_parent_cube
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu
from jaxtrace.gpu.tracking.initial_assignment_cascading import (
    initial_assignment_mesh_aligned_multi_local,
)
from jaxtrace.gpu.search.aa_detection import (
    precompute_aa_metadata,
    precompute_element_vertices,
    AxisAlignedMetadata,
)
from jaxtrace.gpu.search.point_in_tet_methods import (
    set_corrected_metadata,
    set_inverse_matrices_gpu,
)
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices

import vtk
from vtk.util.numpy_support import vtk_to_numpy


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="JAXTrace production particle tracking driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- I/O ---
    # Convention: FEMUSS cases live in <case>.gid/post with mesh PVTU in
    # post/0eule/<case>_*.pvtu and particle PVTU in post/1part/<case>_pt_*.pvtu.
    # Pass --input as either the <case>.gid folder OR the post folder;
    # patterns auto-derive from the case stem unless explicitly overridden.
    parser.add_argument("--input", type=Path, default=Path("data/FLA/post"),
                        help="FEMUSS case folder. Either '.../<case>.gid' or "
                             "'.../<case>.gid/post'. Patterns auto-derive from "
                             "the case stem (e.g. 'cylA.gid' -> 'cylA_*.pvtu').")
    parser.add_argument("--output", type=Path, default=Path("output/tracking"),
                        help="Output directory for VTU files and artifacts")
    parser.add_argument("--mesh-subdir", type=str, default="0eule",
                        help="Subdirectory under --input containing mesh PVTU files")
    parser.add_argument("--mesh-pattern", type=str, default=None,
                        help="Override mesh PVTU pattern with {timestep} placeholder "
                             "(default: auto '<case>_{timestep}.pvtu').")
    parser.add_argument("--femuss-subdir", type=str, default="1part",
                        help="Subdirectory under --input containing FEMUSS particle PVTU files")
    parser.add_argument("--mesh-dir", type=Path, default=None,
                        help="Direct path to mesh PVTU folder, bypassing --input/--mesh-subdir")
    parser.add_argument("--femuss-dir", type=Path, default=None,
                        help="Direct path to FEMUSS particle PVTU folder, bypassing --input/--femuss-subdir")
    parser.add_argument("--femuss-pattern", type=str, default=None,
                        help="Override FEMUSS particle PVTU pattern with {timestep} placeholder "
                             "(default: auto '<case>_pt_{timestep}.pvtu').")
    parser.add_argument("--case-stem", type=str, default=None,
                        help="Override auto-detected case stem (derived from "
                             "'<case>.gid' folder name). Used to build default patterns.")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Optional subfolder name under --output (defaults to auto-generated)")

    # --- Precision ---
    parser.add_argument("--precision", type=str, default="float32",
                        choices=["float32", "float64"],
                        help="Floating-point precision (parsed early from sys.argv)")

    # --- Velocity source: mesh or analytic ---
    # Two paths through the kernel:
    #   mesh      — load velocity from mesh PVTU files; per-stage interpolate
    #               via L0/L1/L2 octree search (default; this is what every
    #               cohort case uses today).
    #   analytic  — load a user-supplied JAX-pure velocity function
    #               (velocity_module). No mesh, no search, no interpolation;
    #               the same fully fused RK4 step calls velocity_fn(pos) or
    #               velocity_fn(pos, t) at each sub-stage. Particles are
    #               seeded inside --domain-bbox; wall behaviour (clamp /
    #               outlet) is controlled by --boundary-walls as on the
    #               mesh path. Inlet drift / FEMUSS / level-set masking
    #               are NOT used here — fold any upstream-stream behaviour
    #               into velocity_fn itself.
    parser.add_argument("--velocity-source", type=str, default="mesh",
                        choices=["mesh", "analytic", "rom"],
                        help="Where velocity values come from per sub-stage")
    parser.add_argument("--velocity-module", type=str, default=None,
                        help="Path to a user .py exporting build_provider(...). "
                             "Required when --velocity-source=analytic. See "
                             "jaxtrace/analytic_fields/uniform.py for a template "
                             "and jaxtrace/analytic_fields/divergence_free_recirculation.py "
                             "for the streamfunction-derived field from the FSW "
                             "Internal Summary appendix §A.")

    # --- ROM velocity reconstruction (--velocity-source rom) ---
    parser.add_argument("--rom-basis", type=str, default=None,
                        help="Path to the FSW-ROM basis file (*.fswrom.basis). "
                             "Required when --velocity-source=rom.")
    parser.add_argument("--rom-romdata", type=str, default=None,
                        help="Path to the FSW-ROM per-case coefficient file "
                             "(*.fswrom.romdata). Required when "
                             "--velocity-source=rom.")
    parser.add_argument("--rom-case-idx", type=int, default=1,
                        help="0-based case index into the ROM coefficient "
                             "table (0..19 for the cohort cylindrical dataset). "
                             "Used only when --velocity-source=rom.")
    parser.add_argument("--rom-formula", type=str, default="centered",
                        choices=["centered", "sigma_c", "c_over_sig",
                                 "no_mean", "no_mean_sig"],
                        help="ROM reconstruction formula. 'centered' = "
                             "SnapshotsMean + sum_k c_k * mode_k. See "
                             "jaxtrace/rom/velocity_recon.py for the others. "
                             "If unsure which formula the ROM was built with, "
                             "run tests/rom/compare_rom_recon.py against a "
                             "known FOM snapshot first.")
    parser.add_argument("--rom-field-group", type=str, default="Displacement",
                        help="HDF5 group name inside the basis/romdata files. "
                             "Default 'Displacement' matches the cohort "
                             "cylindrical case.")
    parser.add_argument("--domain-bbox", type=float, nargs=6, default=None,
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                        help="Six floats defining the logical domain bounding box "
                             "for the analytic path (xmin xmax ymin ymax zmin zmax). "
                             "When omitted, the velocity module's default bbox is "
                             "used; if both are absent, falls back to a generous "
                             "10m cube around the origin.")

    # --- Velocity field (mesh path) ---
    parser.add_argument("--vel-range", type=int, nargs=2, default=[159, 159],
                        metavar=("START", "END"),
                        help="Velocity timestep range (inclusive) for cyclic loading "
                             "(mesh path only)")
    parser.add_argument("--velocity-field", type=str, default=VELOCITY_FIELD_NAME,
                        help="Velocity field name in mesh PVTU (mesh path only)")
    parser.add_argument("--levelset-field", type=str, default=LEVELSET_FIELD_NAME,
                        help="Level-set field name in mesh PVTU (mesh path only)")

    # --- Simulation parameters ---
    parser.add_argument("--n-steps", type=int, default=2684,
                        help="Number of RK4 tracking steps")
    parser.add_argument("--dt", type=float, default=0.0025,
                        help="Timestep size for RK4 integration")
    parser.add_argument("--export-freq", type=int, default=1,
                        help="VTU export frequency (every N steps)")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Print progress every N steps")

    # --- Seeding ---
    parser.add_argument("--seed-source", type=str, default="femuss",
                        choices=["femuss", "box", "grid", "box-frac", "grid-frac", "file"],
                        help="Particle initialization source. "
                             "'femuss' loads from a FEMUSS particle PVTU. "
                             "'box' samples uniformly at random inside --seed-box. "
                             "'grid' places --seed-grid Nx Ny Nz particles on a uniform "
                             "grid inside --seed-box. "
                             "'box-frac' / 'grid-frac' behave like 'box' / 'grid' but the "
                             "bounds come from --seed-fraction (per-axis fractions of the "
                             "mesh bounding box). "
                             "'file' loads positions from --seed-file.")
    parser.add_argument("--femuss-start", type=int, default=0,
                        help="FEMUSS start step when --seed-source=femuss")
    parser.add_argument("--seed-box", type=float, nargs=6, default=None,
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                        help="Absolute seed box bounds (required for --seed-source=box or grid)")
    parser.add_argument("--seed-fraction", type=float, nargs=6, default=None,
                        metavar=("XLO", "XHI", "YLO", "YHI", "ZLO", "ZHI"),
                        help="Per-axis fractions of mesh bbox "
                             "(required for --seed-source=box-frac or grid-frac). "
                             "Each pair must satisfy lo < hi and 0 <= lo,hi <= 1 "
                             "on every face EXCEPT the inlet face named by "
                             "--inlet-wall: that face may extend outside [0, 1] "
                             "for continuous particle entry. "
                             "Example without inlet: 0.0 0.2 0.0 1.0 0.0 1.0 = "
                             "first 20%% of X, full Y/Z. "
                             "Example with --inlet-wall=x_min: -1.5 0.1 0 1 0 1 = "
                             "seed box extends 1.5 mesh-widths outside the mesh on "
                             "the x_min face, ends at 10%% of mesh extent on x_max.")
    parser.add_argument("--seed-grid", type=int, nargs=3, default=None,
                        metavar=("NX", "NY", "NZ"),
                        help="Grid resolution along each axis "
                             "(required for --seed-source=grid or grid-frac). "
                             "Total particle count = NX*NY*NZ; --n-particles is ignored.")
    parser.add_argument("--n-particles", type=int, default=100000,
                        help="Number of particles (used by --seed-source=box and box-frac; "
                             "ignored by grid modes which derive count from --seed-grid).")
    parser.add_argument("--seed-file", type=Path, default=None,
                        help="Path to .npy / .npz file containing (N,3) positions "
                             "(required if --seed-source=file). If .npz, the "
                             "'positions' key is used.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for random box modes")

    # --- FEMUSS comparison (OPTIONAL) ---
    parser.add_argument("--femuss-compare", action="store_true", default=False,
                        help="Compare final positions against FEMUSS at femuss_start + n_steps. "
                             "Requires --seed-source=femuss. If the reference file is missing, "
                             "a warning is printed and the run completes without comparison.")

    # --- Particle export options ---
    parser.add_argument("--export-format", type=str, default="vtkhdf",
                        choices=["vtkhdf", "vtu"],
                        help="Particle export format. "
                             "'vtkhdf' (default): one single .vtkhdf archive for the whole "
                             "run (transient PolyData; ParaView >= 6.0). "
                             "'vtu': one .vtu per step (legacy path, opened directly in "
                             "ParaView as a numbered series). "
                             "VTKHDF is ~5-10x faster to write and ~5-10x faster to transfer "
                             "to a workstation; switch to 'vtu' for older ParaView installs.")
    parser.add_argument("--export-element-ids", action="store_true", default=False,
                        help="Include element IDs in particle export")
    parser.add_argument("--n-groups", type=int, default=5,
                        help="Number of particle groups by initial X position (0 to disable)")
    parser.add_argument("--no-groups", action="store_true", default=False,
                        help="Disable particle group export")
    parser.add_argument("--no-export", action="store_true", default=False,
                        help="Disable all particle export (timing/statistics only)")
    parser.add_argument("--export-escaped-flag", action="store_true", default=False,
                        help="Add a per-particle 'Escaped' UInt8 field (0/1) to VTU output: "
                             "1 if the particle has ever had element_id<0 (escaped the domain "
                             "at any point during its trajectory). Use this in Paraview's "
                             "Threshold filter to remove escaped particles.")
    parser.add_argument("--track-max-temperature", action="store_true", default=False,
                        help="Track the maximum value of the temperature field experienced "
                             "by each particle over its trajectory and export as 'MaxTemperature'. "
                             "Default off. Requires the temperature field to be present in the "
                             "PVTU files (name set by --temperature-field). Loads an extra "
                             "(n_timesteps, n_nodes) scalar stack on GPU; per-step overhead is "
                             "one scalar P1 interpolation per particle (~1-3%% of RK4 step).")
    parser.add_argument("--export-temperature", action="store_true", default=False,
                        help="Export the *instantaneous* temperature interpolated at each "
                             "particle's current position as the 'Temperature' point-data "
                             "field on every exported step. Independent of "
                             "--track-max-temperature; the two share the same on-GPU "
                             "temperature stack and the same per-step P1 interpolation, so "
                             "enabling both adds no measurable cost over enabling either one. "
                             "Default off.")
    parser.add_argument("--temperature-field", type=str, default="Temperature",
                        help="Name of the per-node temperature field in PVTU files "
                             "(used when --track-max-temperature or --export-temperature is set).")

    # --- Tolerances ---
    parser.add_argument("--point-in-tet-tol", type=float, default=1e-6,
                        help="Point-in-tet containment tolerance")
    parser.add_argument("--interpolation-det-min", type=float, default=None,
                        help="Minimum determinant for barycentric interpolation")
    parser.add_argument("--interpolation-method", type=str, default="direct_inverse",
                        choices=["direct_inverse", "gram_matrix"],
                        help="Velocity interpolation method")

    # --- Gradient recovery / higher-order velocity reconstruction ---
    parser.add_argument("--gradient-recovery", type=int, default=1,
                        choices=[0, 1],
                        help="Enable (1) or disable (0) SPR-based gradient "
                             "recovery + higher-order velocity "
                             "reconstruction before the RK4 loop. When "
                             "enabled, the raw nodal velocities feed a "
                             "Step 1-5 pipeline (see "
                             "docs/gradient_recovery_pipeline.md) and the "
                             "kernel samples the reconstruction instead of "
                             "raw P1 barycentric interpolation. Default 1.")
    parser.add_argument("--recovery-method", type=str, default="taylor",
                        choices=["taylor"],
                        help="Which reconstruction to build in Step 5. "
                             "'taylor' = per-element (v_c, G_c, x_c) with "
                             "v(p) = v_c + G_c @ (p - x_c). More methods "
                             "(hct_cubic, ...) will be added; the CLI "
                             "'choices' list will grow with them.")

    parser.add_argument("--boundary-proj-tol", type=float, default=1e-6,
                        help="Inward tolerance for boundary projection clamping")

    # --- RK4 substep policies ---
    parser.add_argument("--bbox-clamp", action="store_true", default=False,
                        help="Enable substep bbox clamping")
    parser.add_argument("--no-bbox-clamp", action="store_true",
                        help="Disable substep bbox clamping (default)")
    parser.add_argument("--failed-substage", type=str, default="zero_vel",
                        choices=["zero_vel", "last_valid_vel", "skip_step"],
                        help="Policy for failed RK4 substages")

    # --- Boundary projection ---
    parser.add_argument("--boundary-proj", action="store_true", default=True,
                        help="Enable boundary projection recovery (default: on)")
    parser.add_argument("--no-boundary-proj", action="store_true",
                        help="Disable boundary projection recovery")
    parser.add_argument("--boundary-walls", type=str, default=None,
                        help="Per-wall boundary projection: 'wall=mode,...'")

    # --- Inlet (continuous influx of particles outside the inlet wall) ---
    parser.add_argument("--inlet-wall", type=str, default="none",
                        choices=["none", "x_min", "x_max", "y_min", "y_max",
                                 "z_min", "z_max"],
                        help="Name of the wall through which the seed box is "
                             "allowed to protrude outside the mesh. Particles "
                             "seeded outside the mesh on this wall are kept "
                             "alive with element_id=-1 and drift at "
                             "--inlet-velocity along the inward normal until "
                             "they cross into the mesh and a real host element "
                             "is found. Default 'none' disables the inlet.")
    parser.add_argument("--inlet-velocity", type=float, default=0.0,
                        help="Signed scalar drift speed for pending-entry "
                             "particles, in m/s along the inward normal of "
                             "--inlet-wall. Positive = toward the mesh, "
                             "negative = away. Required when --inlet-wall is "
                             "set and not 'none'.")

    # --- Level-set ---
    parser.add_argument("--no-levelset", action="store_true",
                        help="Disable level-set velocity masking")
    parser.add_argument("--levelset-mode", type=str, default="zero_vel",
                        choices=["zero_vel", "skip_step"],
                        help="Level-set masking mode")
    parser.add_argument("--l0-skip-band", type=float, default=0.0,
                        help="Level-set band width for L0 cache skip")
    parser.add_argument("--no-l0-skip-boundary", action="store_true", default=False,
                        help="Disable L0 skip entirely")
    parser.add_argument("--enhanced-search-band", type=float, default=0.0,
                        help="Level-set band for enhanced search (node L1 + 5x5x5 L2)")

    # --- L1/L2 ---
    parser.add_argument("--l1-method", type=str, default="face", choices=["face", "node"],
                        help="L1 neighbor method")
    parser.add_argument("--l2-neighborhood", type=int, default=3, choices=[3, 5],
                        help="L2 neighborhood size")

    # --- Pin velocity (FEMUSS FSW) ---
    parser.add_argument("--pin-velocity", action="store_true", default=True,
                        help="Reconstruct pin rotation velocity (default: on)")
    parser.add_argument("--no-pin-velocity", action="store_true", default=False,
                        help="Disable pin velocity reconstruction")
    parser.add_argument("--pin-rpm", type=float, default=-600.0,
                        help="Pin rotation speed in RPM")
    parser.add_argument("--pin-center", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        metavar=("X", "Y", "Z"), help="Pin rotation axis center")
    parser.add_argument("--pin-axis", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                        metavar=("AX", "AY", "AZ"), help="Pin rotation axis direction")
    parser.add_argument("--pin-tilt", type=float, default=0.0,
                        help="Pin tilting angle in degrees")

    # --- RK4 mode / experimental ---
    parser.add_argument("--rk4-mode", type=str, default="fused", choices=["fused", "split"],
                        help="RK4 kernel mode")
    parser.add_argument("--l2-vectorized", action="store_true", default=False,
                        help="Use vectorized L2 (experimental)")
    parser.add_argument("--registration", type=str, default=None,
                        choices=["vertex_multi", "parent_cube"],
                        help="Override octree registration method")
    parser.add_argument("--no-orphan-fallback", action="store_true", default=False,
                        help="Disable the AABB fallback for non-Kuhn elements "
                             "that have no Kuhn neighbour. By default such "
                             "elements get a private cell derived from their "
                             "own AABB; with this flag they are dropped from "
                             "the octree (legacy behaviour).")
    parser.add_argument("--no-hybrid-non-kuhn", action="store_true", default=False,
                        help="Disable AABB-overlap registration for non-Kuhn "
                             "elements (revert to centroid-only). Default is "
                             "hybrid: Kuhn elements use cheap parent-cube "
                             "registration, non-Kuhn elements use AABB-overlap "
                             "so vertices fall within the search neighbourhood. "
                             "Use this flag only to reproduce legacy behaviour "
                             "or to benchmark the difference.")

    # --- density-field online estimation (opt-in) ---
    # All --density-* args are inert unless --density-enable is passed. When
    # disabled, the density runner is never constructed and adds zero overhead
    # to tracking.
    parser.add_argument("--density-enable", action="store_true", default=False,
                        help="Enable online density-field estimation alongside particle tracking.")
    parser.add_argument("--density-output-dir", type=str, default=None,
                        help="Output directory for density files (default: <output_subdir>/density).")
    parser.add_argument("--density-output-format", choices=["vtkhdf", "vti"], default="vtkhdf")
    parser.add_argument("--density-filename-stem", default="density")
    parser.add_argument("--density-kernel", default="wendland_c2",
                        choices=["wendland_c2", "wendland_c4", "cubic_spline",
                                 "gaussian", "epanechnikov", "quintic_spline"])
    parser.add_argument("--density-bandwidth-mode", default="fixed",
                        choices=["fixed", "scott", "silverman", "knn_adaptive",
                                 "initial_spacing"],
                        help="initial_spacing: h_xyz = bandwidth_factor * Δp_axis "
                             "from the seeded (step-0) particle positions.")
    parser.add_argument("--density-bandwidth", type=float, default=None,
                        help="Fixed bandwidth (fixed mode only). Default: bandwidth-factor*voxel_size.")
    parser.add_argument("--density-bandwidth-xyz", type=float, nargs=3, default=None,
                        metavar=("HX", "HY", "HZ"),
                        help="Per-axis fixed bandwidth. Overrides --density-bandwidth. "
                             "Kernel is normalised by 1/(hx hy hz).")
    parser.add_argument("--density-bandwidth-factor", type=float, default=2.0)
    parser.add_argument("--density-bandwidth-refresh-every", type=int, default=0)
    parser.add_argument("--density-knn-k", type=int, default=32)
    parser.add_argument("--density-knn-safety", type=float, default=1.2)
    parser.add_argument("--density-bounds", type=float, nargs=6, default=None,
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                        help="Explicit voxel grid bounds. Default: mesh bbox.")
    parser.add_argument("--density-bounds-from", choices=["mesh", "particles"], default="mesh")
    parser.add_argument("--density-resolution", type=int, default=128,
                        help="Cubic voxel-grid resolution; ignored if "
                             "--density-resolution-xyz / --density-voxel-size(-xyz) is set.")
    parser.add_argument("--density-resolution-xyz", type=int, nargs=3, default=None,
                        metavar=("NX", "NY", "NZ"),
                        help="Per-axis voxel-grid resolution. "
                             "Overrides --density-resolution.")
    parser.add_argument("--density-voxel-size", type=float, default=None)
    parser.add_argument("--density-voxel-size-xyz", type=float, nargs=3, default=None,
                        metavar=("HX", "HY", "HZ"),
                        help="Per-axis voxel edge length. Overrides --density-voxel-size "
                             "and --density-resolution.")
    parser.add_argument("--density-voxel-size-from-particles", action="store_true",
                        help="Size the voxel grid to the per-axis initial "
                             "inter-particle spacing Δp_axis (from seeded "
                             "positions). Overrides all other voxel-size / "
                             "resolution flags.")
    parser.add_argument("--density-pad-fraction", type=float, default=0.0)
    parser.add_argument("--density-no-mask-inside-mesh", action="store_true")
    parser.add_argument("--density-engine", choices=["auto", "brute", "octree"], default="auto")
    parser.add_argument("--density-auto-threshold", type=float, default=1e10)
    parser.add_argument("--density-brute-query-chunk", type=int, default=8192)
    parser.add_argument("--density-octree-target-n-per-cell", type=int, default=9,
                        help="Particle-hash target average particles per cell.")
    # Deprecated CLI knobs accepted for back-compat; ignored at runtime.
    parser.add_argument("--density-octree-cells-per-dim", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--density-octree-max-neighbors", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--density-particle-bucket", type=int, default=4096)
    parser.add_argument("--density-no-per-step", action="store_true")
    parser.add_argument("--density-no-time-average", action="store_true")
    parser.add_argument("--density-no-particle-density", action="store_true",
                        help="Skip per-particle density samples in extra_scalars.")
    parser.add_argument("--density-export-freq", type=int, default=None,
                        help="Compute density every Nth step. Default: same as --export-freq.")
    parser.add_argument("--density-normalization", choices=["pdf", "mass", "unnormalized"], default="pdf")
    parser.add_argument("--density-compression", choices=["gzip", "lzf", "blosc", "none"], default="gzip",
                        help="HDF5 compression filter for density files. Default gzip — "
                             "only filter ParaView's bundled vtkhdf5 can decompress.")
    parser.add_argument("--density-compression-opts", type=int, default=1,
                        help="Compression level for gzip (1-9) or blosc clevel.")
    parser.add_argument("--density-blosc-threads", type=int, default=4)

    return parser.parse_args()


# =============================================================================
# PARTICLE SEEDING
# =============================================================================

def seed_from_femuss(args):
    """Load particle positions from a FEMUSS PVTU file."""
    femuss_dir = args._femuss_dir
    start_file = femuss_dir / args.femuss_pattern.format(timestep=args.femuss_start)
    if not start_file.exists():
        raise FileNotFoundError(f"FEMUSS seed file not found: {start_file}")
    print(f"  Seed source: FEMUSS particles at step {args.femuss_start}")
    print(f"    {start_file}")
    femuss = load_femuss_particles(start_file)
    active = ~femuss['has_left_domain']
    idx = np.where(active)[0]
    positions = femuss['current_positions'][idx].astype(config.FLOAT_DTYPE_NP)
    particle_ids = idx.astype(np.int32)
    print(f"    Active at start: {len(idx):,}/{femuss['n_particles']:,}")
    return positions, particle_ids, femuss


_WALL_TO_AXIS = {
    "x_min": (0, "min"), "x_max": (0, "max"),
    "y_min": (1, "min"), "y_max": (1, "max"),
    "z_min": (2, "min"), "z_max": (2, "max"),
}


def _inlet_inward_normal(inlet_wall: str) -> np.ndarray:
    """Return the inward-pointing unit normal for the named wall."""
    axis, side = _WALL_TO_AXIS[inlet_wall]
    n = np.zeros(3, dtype=np.float32)
    n[axis] = 1.0 if side == "min" else -1.0
    return n


def _crop_seed_bounds_to_mesh(bounds, mesh_bbox_min, mesh_bbox_max,
                              inlet_wall="none"):
    """Crop a (2, 3) seed box to the mesh bounding box on every face except
    the inlet face.

    Returns
    -------
    cropped : np.ndarray, shape (2, 3)
    changed : bool — True if any face was actually cropped
    """
    cropped = np.array(bounds, dtype=np.float32).copy()
    mesh_bbox_min = np.asarray(mesh_bbox_min, dtype=np.float32)
    mesh_bbox_max = np.asarray(mesh_bbox_max, dtype=np.float32)
    inlet_face = _WALL_TO_AXIS.get(inlet_wall) if inlet_wall != "none" else None

    changed = False
    axis_names = ("x", "y", "z")
    for axis in range(3):
        # Lower face of this axis
        if (inlet_face is None) or (inlet_face != (axis, "min")):
            if cropped[0, axis] < mesh_bbox_min[axis]:
                print(f"  [seed-crop] {axis_names[axis]}_min face "
                      f"{cropped[0, axis]:.6g} -> {mesh_bbox_min[axis]:.6g}")
                cropped[0, axis] = mesh_bbox_min[axis]
                changed = True
        # Upper face of this axis
        if (inlet_face is None) or (inlet_face != (axis, "max")):
            if cropped[1, axis] > mesh_bbox_max[axis]:
                print(f"  [seed-crop] {axis_names[axis]}_max face "
                      f"{cropped[1, axis]:.6g} -> {mesh_bbox_max[axis]:.6g}")
                cropped[1, axis] = mesh_bbox_max[axis]
                changed = True

    if cropped[0].max() >= cropped[1].max() or np.any(cropped[0] >= cropped[1]):
        raise ValueError(
            f"Cropped seed bounds are degenerate: min={cropped[0]} max={cropped[1]}. "
            f"Original bounds={bounds}, mesh bbox=[{mesh_bbox_min}, {mesh_bbox_max}]"
        )
    return cropped, changed


def _resolve_seed_bounds(args, mesh_bbox_min=None, mesh_bbox_max=None):
    """Resolve absolute (2, 3) bounds for box-based seeding.

    Uses --seed-box for absolute bounds, or --seed-fraction (and the mesh
    bbox passed in) for fraction-based bounds. When --inlet-wall is not
    'none', the seed box is allowed to extend outside the mesh on that
    one face; every other face is cropped to the mesh bounding box (with
    a warning). Fraction-based bounds are always inside the mesh by
    construction so no cropping is performed.
    """
    from jaxtrace.tracking.seeding import bounds_from_fractions

    is_frac = args.seed_source.endswith("-frac")
    if is_frac:
        if args.seed_fraction is None:
            raise ValueError(
                f"--seed-source={args.seed_source} requires "
                "--seed-fraction XLO XHI YLO YHI ZLO ZHI"
            )
        if mesh_bbox_min is None or mesh_bbox_max is None:
            raise RuntimeError("Mesh bbox required for fraction-based seeding")
        xl, xh, yl, yh, zl, zh = args.seed_fraction
        inlet_wall = getattr(args, "inlet_wall", "none")
        # Fraction extension outside [0, 1] is permitted ONLY on the inlet
        # face. bounds_from_fractions enforces this and raises a clear
        # error if the user sets fractions outside [0, 1] without a
        # matching inlet wall.
        bounds = bounds_from_fractions(
            mesh_bbox_min, mesh_bbox_max,
            x_frac=(xl, xh), y_frac=(yl, yh), z_frac=(zl, zh),
            inlet_wall=inlet_wall,
        )
        return bounds, False

    if args.seed_box is None:
        raise ValueError(
            f"--seed-source={args.seed_source} requires "
            "--seed-box XMIN XMAX YMIN YMAX ZMIN ZMAX"
        )
    x0, x1, y0, y1, z0, z1 = args.seed_box
    bounds = np.array([[x0, y0, z0], [x1, y1, z1]], dtype=np.float32)

    inlet_wall = getattr(args, "inlet_wall", "none")
    if mesh_bbox_min is not None and mesh_bbox_max is not None:
        bounds, cropped = _crop_seed_bounds_to_mesh(
            bounds, mesh_bbox_min, mesh_bbox_max, inlet_wall=inlet_wall,
        )
        return bounds, cropped
    return bounds, False


def seed_from_box(args, mesh_bbox_min=None, mesh_bbox_max=None):
    """Uniform random seeding inside an axis-aligned box.

    Box bounds come from --seed-box (absolute) when seed-source is 'box',
    or from --seed-fraction + mesh bbox when seed-source is 'box-frac'.
    The box is cropped to the mesh bbox on every face except the inlet
    face when --inlet-wall is set; particles falling outside the mesh on
    the inlet face are kept and initialised with element_id=-1.
    """
    bounds, _ = _resolve_seed_bounds(args, mesh_bbox_min, mesh_bbox_max)
    rng = np.random.default_rng(args.seed)
    n = args.n_particles
    positions = np.stack([
        rng.uniform(bounds[0, 0], bounds[1, 0], n),
        rng.uniform(bounds[0, 1], bounds[1, 1], n),
        rng.uniform(bounds[0, 2], bounds[1, 2], n),
    ], axis=1).astype(config.FLOAT_DTYPE_NP)
    particle_ids = np.arange(n, dtype=np.int32)
    print(f"  Seed source: uniform random in box ({args.seed_source})")
    print(f"    X=[{bounds[0,0]:.6g},{bounds[1,0]:.6g}]  "
          f"Y=[{bounds[0,1]:.6g},{bounds[1,1]:.6g}]  "
          f"Z=[{bounds[0,2]:.6g},{bounds[1,2]:.6g}]")
    print(f"    n_particles={n:,}  seed={args.seed}")
    return positions, particle_ids, None


def seed_from_grid(args, mesh_bbox_min=None, mesh_bbox_max=None):
    """Uniform-grid seeding inside an axis-aligned box.

    Particle count = Nx*Ny*Nz (--seed-grid) when no cropping is applied.
    When the seed box is cropped against the mesh bounding box (any face
    except --inlet-wall), the grid SPACING along each axis is preserved
    and grid rows/columns falling outside the cropped bounds are dropped;
    the final particle count is therefore lower than Nx*Ny*Nz and is
    printed alongside the requested resolution.
    """
    from jaxtrace.tracking.seeding import uniform_grid_seeds

    if args.seed_grid is None:
        raise ValueError(
            f"--seed-source={args.seed_source} requires --seed-grid NX NY NZ"
        )
    nx, ny, nz = (int(v) for v in args.seed_grid)
    if min(nx, ny, nz) < 1:
        raise ValueError(f"--seed-grid values must be >= 1; got {args.seed_grid}")

    # Original bounds (pre-crop) used as the reference for grid spacing
    # when the absolute-box path is cropped against the mesh bbox. For
    # the fraction-based path no cropping is applied (the user's fraction
    # directly defines the seeded volume, including any inlet-axis
    # extension outside [0, 1]), so bounds_orig is irrelevant.
    bounds, cropped = _resolve_seed_bounds(args, mesh_bbox_min, mesh_bbox_max)
    if cropped and args.seed_box is not None:
        x0, x1, y0, y1, z0, z1 = args.seed_box
        bounds_orig = np.array([[x0, y0, z0], [x1, y1, z1]], dtype=np.float32)
    else:
        bounds_orig = None

    if cropped and bounds_orig is not None:
        # Preserve spacing from the user-requested grid; emit only the
        # grid nodes that fall inside the cropped bounds.
        spacing = (bounds_orig[1] - bounds_orig[0]) / np.maximum(
            np.array([nx - 1, ny - 1, nz - 1]), 1
        )
        # Generate the full uncropped axis coordinates, then mask each axis.
        axes_full = [
            np.linspace(bounds_orig[0, a], bounds_orig[1, a],
                        max(int([nx, ny, nz][a]), 1), dtype=np.float32)
            for a in range(3)
        ]
        axes_kept = [
            a[(a >= bounds[0, i] - 0.5 * spacing[i]) &
              (a <= bounds[1, i] + 0.5 * spacing[i])]
            for i, a in enumerate(axes_full)
        ]
        X, Y, Z = np.meshgrid(*axes_kept, indexing='ij')
        positions = np.column_stack(
            [X.ravel(), Y.ravel(), Z.ravel()]
        ).astype(config.FLOAT_DTYPE_NP)
        kept_counts = tuple(a.size for a in axes_kept)
        print(f"  Seed source: uniform grid in cropped box ({args.seed_source})")
        print(f"    Requested grid: {nx} x {ny} x {nz} = {nx*ny*nz:,} particles")
        print(f"    After crop:     {kept_counts[0]} x {kept_counts[1]} x "
              f"{kept_counts[2]} = {positions.shape[0]:,} particles")
    else:
        positions = uniform_grid_seeds(
            resolution=(nx, ny, nz), bounds=bounds, include_boundaries=True,
        ).astype(config.FLOAT_DTYPE_NP)
        print(f"  Seed source: uniform grid in box ({args.seed_source})")
        print(f"    Grid: {nx} x {ny} x {nz} = {positions.shape[0]:,} particles")

    n = positions.shape[0]
    if args.n_particles and args.n_particles != n:
        print(f"    Note: --n-particles={args.n_particles:,} ignored; "
              f"actual grid count = {n:,}")
    particle_ids = np.arange(n, dtype=np.int32)
    print(f"    X=[{bounds[0,0]:.6g},{bounds[1,0]:.6g}]  "
          f"Y=[{bounds[0,1]:.6g},{bounds[1,1]:.6g}]  "
          f"Z=[{bounds[0,2]:.6g},{bounds[1,2]:.6g}]")
    return positions, particle_ids, None


def seed_from_file(args):
    """Load positions from .npy or .npz."""
    if args.seed_file is None:
        raise ValueError("--seed-source=file requires --seed-file PATH")
    path = Path(args.seed_file)
    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")
    if path.suffix == '.npz':
        data = np.load(path)
        if 'positions' not in data.files:
            raise KeyError(f"{path} has no 'positions' array; available: {data.files}")
        positions = data['positions']
    else:
        positions = np.load(path)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Seed positions must be (N,3); got {positions.shape}")
    positions = positions.astype(config.FLOAT_DTYPE_NP)
    particle_ids = np.arange(positions.shape[0], dtype=np.int32)
    print(f"  Seed source: {path}")
    print(f"    n_particles={positions.shape[0]:,}")
    return positions, particle_ids, None


# =============================================================================
# MAIN
# =============================================================================

def _resolve_case_paths(args):
    """Auto-derive case stem and PVTU file patterns from --input.

    Supports --input given as either '<case>.gid' or '<case>.gid/post'.
    Normalises args.input to point at the 'post' directory and fills in
    args.mesh_pattern / args.femuss_pattern when the user did not override.
    """
    in_path = Path(args.input).resolve()

    # Normalise: if user passed '<case>.gid', descend into its 'post/' child.
    if in_path.name.endswith('.gid') and (in_path / 'post').is_dir():
        case_dir = in_path
        post_dir = in_path / 'post'
    elif in_path.name == 'post' and in_path.parent.name.endswith('.gid'):
        case_dir = in_path.parent
        post_dir = in_path
    else:
        # Fallback: treat --input as the 'post' directory itself.
        case_dir = in_path.parent if in_path.parent.name.endswith('.gid') else in_path
        post_dir = in_path

    # Derive case stem: 'cylA.gid' -> 'cylA'
    if args.case_stem is not None:
        stem = args.case_stem
    elif case_dir.name.endswith('.gid'):
        stem = case_dir.name[:-4]
    else:
        stem = case_dir.name

    # Fill in patterns only if the user did not override
    if args.mesh_pattern is None:
        args.mesh_pattern = f"{stem}_{{timestep}}.pvtu"
    if args.femuss_pattern is None:
        args.femuss_pattern = f"{stem}_pt_{{timestep}}.pvtu"

    args.input = post_dir

    # Resolve mesh / FEMUSS directories: direct override or standard layout
    args._mesh_dir = (Path(args.mesh_dir).resolve() if args.mesh_dir
                      else post_dir / args.mesh_subdir)
    args._femuss_dir = (Path(args.femuss_dir).resolve() if args.femuss_dir
                        else post_dir / args.femuss_subdir)

    return stem


# =============================================================================
# Analytic-source path
# =============================================================================

def _run_analytic_tracking(args):
    """Drive RK4 tracking against a user-supplied analytic velocity field.

    No mesh load, no octree build, no FEMUSS comparison. Particles are
    seeded inside `--domain-bbox` (or the velocity module's default
    bbox) according to `--seed-source`. The RK4 step is built via
    create_rk4_analytic(); per-step plumbing (export, monitor, manifest)
    is shared with the mesh path where it doesn't depend on the mesh.

    This function is intentionally simpler than the mesh main(). Several
    features are deliberately not supported on the analytic path:

      * --seed-source femuss / file (mesh-tied)
      * --inlet-wall / --inlet-velocity (mesh-tied; embed any upstream
        stream behaviour in velocity_fn directly)
      * --pin-velocity / --no-levelset (mesh-tied)
      * --femuss-compare (mesh-tied)
      * --export-temperature / --track-max-temperature (mesh-tied)
      * --boundary-walls ballistic / freeze (would require last_vel
        tracking and pre/post-step snapshots; out of scope here)

    Supported:
      * --seed-source grid / box / grid-frac / box-frac
      * --seed-grid / --seed-box / --seed-fraction
      * --domain-bbox (or auto from the velocity module)
      * --boundary-walls clamp / outlet (per-axis via the bbox clamp)
      * --n-steps / --dt
      * --output / --export-freq / --log-interval
      * --no-export (skip particle export entirely)
      * --bbox-clamp / --no-bbox-clamp (substep clamp)
    """
    import datetime
    import time as _time

    from jaxtrace.gpu.tracking.velocity_provider import (
        load_analytic_provider,
    )
    from benchmark_femuss_comparison import create_rk4_analytic

    # ----- Validate args for the analytic path -----
    if not args.velocity_module:
        print("ERROR: --velocity-source=analytic requires --velocity-module PATH",
              file=sys.stderr)
        return 2
    if not Path(args.velocity_module).is_file():
        print(f"ERROR: --velocity-module file not found: {args.velocity_module}",
              file=sys.stderr)
        return 2

    # Reject features that are genuinely incompatible with the analytic
    # path (so the user fails fast). Silently downgrade pin / level-set /
    # temperature: those default to "on" but only do anything when a mesh
    # is loaded, so leaving them in their CLI defaults must not block an
    # analytic run.
    rejected = []
    if args.seed_source not in ("grid", "box", "grid-frac", "box-frac"):
        rejected.append(
            f"--seed-source={args.seed_source} (analytic path: use grid/box/grid-frac/box-frac)"
        )
    if getattr(args, "femuss_compare", False):
        rejected.append("--femuss-compare (mesh-tied)")
    if getattr(args, "inlet_wall", "none") != "none":
        rejected.append("--inlet-wall (mesh-tied; embed upstream stream in velocity_fn)")
    if rejected:
        print("ERROR: the following options are not supported on --velocity-source=analytic:",
              file=sys.stderr)
        for r in rejected:
            print(f"  {r}", file=sys.stderr)
        return 2

    # Silently disable mesh-tied features that the user may have left at
    # their CLI defaults. The analytic path doesn't load a mesh, so
    # nothing reads these anyway — printing a notice keeps the run banner
    # honest without forcing the user to add `--no-pin-velocity` etc.
    silently_disabled = []
    if getattr(args, "pin_velocity", False) and not getattr(args, "no_pin_velocity", False):
        silently_disabled.append("pin-velocity")
    if not getattr(args, "no_levelset", False):
        silently_disabled.append("level-set masking")
    if getattr(args, "track_max_temperature", False):
        silently_disabled.append("max-temperature tracking")
    if getattr(args, "export_temperature", False):
        silently_disabled.append("temperature export")
    if silently_disabled:
        print(f"  [info] mesh-only features ignored on analytic path: "
              f"{', '.join(silently_disabled)}")

    # ----- Apply CLI flags to config -----
    config.RK4_SUBSTEP_BBOX_CLAMP = args.bbox_clamp and not args.no_bbox_clamp
    config.RK4_BOUNDARY_PROJECTION = not args.no_boundary_proj
    config.RK4_BOUNDARY_PROJECTION_TOL = args.boundary_proj_tol

    DT = args.dt
    N_STEPS = args.n_steps
    LOG_INTERVAL = args.log_interval
    EXPORT_FREQUENCY = args.export_freq

    # ----- Banner -----
    print("=" * 80)
    print("JAXTrace Production Tracking — ANALYTIC velocity source")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print()

    # ----- Load the user analytic-velocity module -----
    print("[1/4] Loading analytic velocity module...")
    t_load = _time.time()

    # Resolve the domain bbox precedence: explicit --domain-bbox wins;
    # otherwise the velocity module's default; otherwise a generous cube.
    cli_bbox = None
    if args.domain_bbox is not None:
        cli_bbox = (
            (args.domain_bbox[0], args.domain_bbox[1]),
            (args.domain_bbox[2], args.domain_bbox[3]),
            (args.domain_bbox[4], args.domain_bbox[5]),
        )

    provider = load_analytic_provider(
        module_path=args.velocity_module,
        domain_bbox=cli_bbox,
        dt=DT,
    )

    if provider.domain_bbox is None:
        print("  WARNING: no domain bbox supplied via --domain-bbox or velocity "
              "module; falling back to ((-10,10),(-10,10),(-10,10))")
        provider = provider.__class__(
            velocity_fn=provider.velocity_fn,
            is_time_dependent=provider.is_time_dependent,
            level_set_fn=provider.level_set_fn,
            domain_bbox=((-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)),
            meta=provider.meta,
        )

    domain_bbox = provider.domain_bbox
    bbox_min_np = np.asarray(
        [domain_bbox[0][0], domain_bbox[1][0], domain_bbox[2][0]],
        dtype=config.FLOAT_DTYPE_NP,
    )
    bbox_max_np = np.asarray(
        [domain_bbox[0][1], domain_bbox[1][1], domain_bbox[2][1]],
        dtype=config.FLOAT_DTYPE_NP,
    )

    print(f"  Module:        {args.velocity_module}")
    if provider.meta:
        print(f"  Field name:    {provider.meta.get('name', '<unnamed>')}")
        if 'source' in provider.meta:
            print(f"  Field source:  {provider.meta['source']}")
        if 'params' in provider.meta:
            print(f"  Parameters:    {provider.meta['params']}")
    print(f"  Time-dependent: {provider.is_time_dependent}")
    print(f"  Domain bbox:   x=[{bbox_min_np[0]:g}, {bbox_max_np[0]:g}]  "
          f"y=[{bbox_min_np[1]:g}, {bbox_max_np[1]:g}]  "
          f"z=[{bbox_min_np[2]:g}, {bbox_max_np[2]:g}]")
    print(f"  Loaded in {_time.time() - t_load:.2f}s")
    print()

    # ----- Seed particles -----
    print("[2/4] Seeding particles...")
    t_seed = _time.time()
    rng = np.random.default_rng(args.seed if hasattr(args, 'seed') else 42)

    if args.seed_source in ("grid", "grid-frac"):
        # Build a uniform grid inside the bbox.
        if args.seed_source == "grid-frac":
            if args.seed_fraction is None or len(args.seed_fraction) != 6:
                print("ERROR: --seed-source=grid-frac requires "
                      "--seed-fraction 'XLO XHI YLO YHI ZLO ZHI'", file=sys.stderr)
                return 2
            fx0, fx1, fy0, fy1, fz0, fz1 = args.seed_fraction
            seed_lo = bbox_min_np + np.array([fx0, fy0, fz0]) * (bbox_max_np - bbox_min_np)
            seed_hi = bbox_min_np + np.array([fx1, fy1, fz1]) * (bbox_max_np - bbox_min_np)
        else:
            # absolute box from --seed-box
            if args.seed_box is None or len(args.seed_box) != 6:
                print("ERROR: --seed-source=grid requires --seed-box "
                      "'XMIN XMAX YMIN YMAX ZMIN ZMAX'", file=sys.stderr)
                return 2
            seed_lo = np.array(args.seed_box[::2], dtype=config.FLOAT_DTYPE_NP)
            seed_hi = np.array(args.seed_box[1::2], dtype=config.FLOAT_DTYPE_NP)

        if args.seed_grid is None or len(args.seed_grid) != 3:
            print("ERROR: --seed-grid 'NX NY NZ' required for grid seeding",
                  file=sys.stderr)
            return 2
        nx, ny, nz = args.seed_grid
        xs = np.linspace(seed_lo[0], seed_hi[0], nx, endpoint=True)
        ys = np.linspace(seed_lo[1], seed_hi[1], ny, endpoint=True)
        zs = np.linspace(seed_lo[2], seed_hi[2], nz, endpoint=True)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        positions = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(
            config.FLOAT_DTYPE_NP
        )
        n_particles = positions.shape[0]
        print(f"  Grid:          {nx} x {ny} x {nz} = {n_particles:,} particles")
        print(f"  Seed box:      x=[{seed_lo[0]:g}, {seed_hi[0]:g}]  "
              f"y=[{seed_lo[1]:g}, {seed_hi[1]:g}]  "
              f"z=[{seed_lo[2]:g}, {seed_hi[2]:g}]")

    else:  # box / box-frac
        if args.seed_source == "box-frac":
            fx0, fx1, fy0, fy1, fz0, fz1 = args.seed_fraction
            seed_lo = bbox_min_np + np.array([fx0, fy0, fz0]) * (bbox_max_np - bbox_min_np)
            seed_hi = bbox_min_np + np.array([fx1, fy1, fz1]) * (bbox_max_np - bbox_min_np)
        else:
            seed_lo = np.array(args.seed_box[::2], dtype=config.FLOAT_DTYPE_NP)
            seed_hi = np.array(args.seed_box[1::2], dtype=config.FLOAT_DTYPE_NP)

        n_particles = args.n_particles
        positions = rng.uniform(
            low=seed_lo, high=seed_hi, size=(n_particles, 3)
        ).astype(config.FLOAT_DTYPE_NP)
        print(f"  Random uniform: {n_particles:,} particles")
        print(f"  Seed box:      x=[{seed_lo[0]:g}, {seed_hi[0]:g}]  "
              f"y=[{seed_lo[1]:g}, {seed_hi[1]:g}]  "
              f"z=[{seed_lo[2]:g}, {seed_hi[2]:g}]")

    print(f"  Seeded in {_time.time() - t_seed:.2f}s")
    print()

    # ----- Build kernel -----
    print("[3/4] Building analytic RK4 kernel...")
    t_build = _time.time()

    # Per-wall clamp masks. We accept --boundary-walls "x_max=outlet,..." with
    # the same syntax as the mesh path, but on the analytic side only "clamp"
    # and "outlet" make sense (no mesh, no ballistic last-vel tracking).
    clamp_min_mask = jnp.array([True, True, True])
    clamp_max_mask = jnp.array([True, True, True])
    if args.boundary_walls:
        wall_dict = {}
        for pair in args.boundary_walls.split(','):
            if '=' not in pair:
                continue
            w, m = pair.split('=', 1)
            wall_dict[w.strip()] = m.strip()
        if any(m in ("ballistic", "freeze") for m in wall_dict.values()):
            print("  WARNING: --boundary-walls ballistic/freeze is mesh-only; "
                  "on analytic path those walls behave as 'outlet' (no clamp).")
        clamp_min_mask = jnp.array([
            wall_dict.get('x_min', 'clamp') == 'clamp',
            wall_dict.get('y_min', 'clamp') == 'clamp',
            wall_dict.get('z_min', 'clamp') == 'clamp',
        ])
        clamp_max_mask = jnp.array([
            wall_dict.get('x_max', 'clamp') == 'clamp',
            wall_dict.get('y_max', 'clamp') == 'clamp',
            wall_dict.get('z_max', 'clamp') == 'clamp',
        ])

    bbox_min_gpu = jnp.asarray(bbox_min_np)
    bbox_max_gpu = jnp.asarray(bbox_max_np)
    rk4_step = create_rk4_analytic(
        provider,
        domain_bbox_min=bbox_min_gpu,
        domain_bbox_max=bbox_max_gpu,
        clamp_min_mask=clamp_min_mask,
        clamp_max_mask=clamp_max_mask,
    )

    positions_gpu = jax.device_put(positions)
    # element_ids: always 0 on the analytic path so "active" mask is uniformly
    # True throughout. The provider doesn't track host elements; -1 would make
    # every particle "lost" by the mesh-path convention.
    element_ids_gpu = jnp.zeros(n_particles, dtype=jnp.int32)

    # Dummy velocity-fields array (analytic kernel ignores it; just a
    # placeholder that satisfies the signature).
    velocity_sequence_gpu = jnp.zeros((1, 1, 3), dtype=config.FLOAT_DTYPE_JNP)

    # Warmup compile.
    positions_gpu, element_ids_gpu = rk4_step(
        positions_gpu, element_ids_gpu, DT, velocity_sequence_gpu, 0
    )
    jax.block_until_ready(positions_gpu)
    print(f"  Compiled + warmup in {_time.time() - t_build:.2f}s")
    print()

    # ----- Output directory + exporter -----
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    field_name = (provider.meta or {}).get('name', 'analytic')
    output_subdir = output_dir / f"run_analytic_{field_name}_n{n_particles}_s{N_STEPS}"
    output_subdir.mkdir(parents=True, exist_ok=True)
    print(f"  Output subdir: {output_subdir}")

    # Particle exporter — mirrors the mesh path's VTKHDF / VTU choice via
    # --export-format. Provider-level mesh-tied extras (Temperature,
    # MaxTemperature, Density) are not available on the analytic path
    # and are simply omitted; Group (initial-X bucketing) and Escaped
    # (bbox-clamp triggered) are mesh-independent and supported here.
    EXPORT = not args.no_export
    EXPORT_ELEMENT_IDS = bool(getattr(args, "export_element_ids", False))
    N_GROUPS = args.n_groups if not args.no_groups else 0
    particle_ids = np.arange(n_particles, dtype=np.int32)

    # Particle groups by initial X (same convention as mesh path).
    particle_groups = None
    if N_GROUPS > 0:
        initial_x = np.asarray(positions_gpu, dtype=config.FLOAT_DTYPE_NP)[:, 0]
        x_min, x_max = float(initial_x.min()), float(initial_x.max())
        x_range = x_max - x_min
        if x_range > 0:
            g = (initial_x - x_min) / x_range * N_GROUPS
            particle_groups = np.clip(g.astype(np.int32), 0, N_GROUPS - 1).astype(np.uint8)
        else:
            particle_groups = np.zeros(n_particles, dtype=np.uint8)
    extra_scalars_base = (
        {'Group': particle_groups} if particle_groups is not None else None
    )

    # Escaped tracking: a particle is marked Escaped on the first step its
    # position is exactly on a bbox face (the substep / boundary-projection
    # machinery just clamped it). bbox_min_np / bbox_max_np are already in
    # scope from the kernel-build stage above.
    track_escaped = bool(args.export_escaped_flag)
    escaped_ever = (
        np.zeros(n_particles, dtype=np.uint8) if track_escaped else None
    )

    exporter = None
    if EXPORT:
        if args.export_format == "vtkhdf":
            exporter = VTKHDFExportThread(
                output_subdir,
                n_particles_hint=int(n_particles),
            )
            print(f"  Particle export: VTKHDF -> {exporter.output_path}")
        else:
            exporter = VTKExportThread(output_subdir)
            print(f"  Particle export: VTU+.pvd -> {output_subdir}")
        exporter.start()

    # Graceful shutdown so a SIGTERM/SIGINT during the loop still leaves
    # a recoverable archive (same pattern as the mesh path).
    import signal as _signal
    _shutdown_called = [False]

    def _graceful_shutdown(signum, frame):
        if _shutdown_called[0]:
            return
        _shutdown_called[0] = True
        print(f"\n[!] Received signal {signum}; flushing exporter and exiting.")
        try:
            if exporter is not None:
                exporter.stop()
        except Exception as e:
            print(f"  Exporter shutdown error: {e}")
        _signal.signal(signum, _signal.SIG_DFL)
        import os as _os
        _os.kill(_os.getpid(), signum)

    _signal.signal(_signal.SIGTERM, _graceful_shutdown)
    _signal.signal(_signal.SIGINT, _graceful_shutdown)

    # Export initial state.
    if exporter is not None:
        pos_cpu = np.asarray(positions_gpu, dtype=config.FLOAT_DTYPE_NP)
        eid_cpu = (
            np.asarray(element_ids_gpu, dtype=np.int32)
            if EXPORT_ELEMENT_IDS else None
        )
        init_extras = dict(extra_scalars_base) if extra_scalars_base else {}
        if track_escaped:
            init_extras['Escaped'] = escaped_ever.copy()
        exporter.enqueue_export(
            0, pos_cpu,
            particle_ids=particle_ids,
            element_ids=eid_cpu,
            extra_scalars=init_extras or None,
        )
        print(f"  Exported initial state (step 0)")

    # ----- Run the loop -----
    print("[4/4] Running {} RK4 steps...".format(N_STEPS))
    print("=" * 80)
    t_start = _time.time()
    DT_jnp = jnp.asarray(DT, dtype=config.FLOAT_DTYPE_JNP)

    for step in range(1, N_STEPS + 1):
        positions_gpu, element_ids_gpu = rk4_step(
            positions_gpu, element_ids_gpu, DT_jnp, velocity_sequence_gpu, step - 1
        )

        do_log = (step % LOG_INTERVAL == 0) or (step == N_STEPS)
        do_export = EXPORT and ((step % EXPORT_FREQUENCY == 0) or (step == N_STEPS))

        if do_log:
            elapsed = _time.time() - t_start
            sps = step / elapsed if elapsed > 0 else 0
            eta = (N_STEPS - step) / sps if sps > 0 else 0
            print(f"  Step {step:5d}/{N_STEPS}: active={n_particles:,} lost=0  "
                  f"[{elapsed:.0f}s, {sps:.1f} step/s, ETA {eta:.0f}s]")

        if do_export and exporter is not None:
            pos_cpu = np.asarray(positions_gpu, dtype=config.FLOAT_DTYPE_NP)
            if track_escaped:
                # Particle sits at a bbox face (within boundary_projection_tol
                # of an inset face) => the clamp has fired. Use 2x the tol
                # so the inset clamp position (= bbox ± tol) is reliably
                # caught even in float32.
                _esc_tol = max(2.0 * config.RK4_BOUNDARY_PROJECTION_TOL, 1e-6)
                on_face = (
                    np.any(np.isclose(pos_cpu, bbox_min_np, atol=_esc_tol, rtol=0), axis=1)
                    | np.any(np.isclose(pos_cpu, bbox_max_np, atol=_esc_tol, rtol=0), axis=1)
                )
                escaped_ever |= on_face.astype(np.uint8)
            step_extras = dict(extra_scalars_base) if extra_scalars_base else {}
            if track_escaped:
                step_extras['Escaped'] = escaped_ever.copy()
            eid_cpu = (
                np.asarray(element_ids_gpu, dtype=np.int32)
                if EXPORT_ELEMENT_IDS else None
            )
            exporter.enqueue_export(
                step, pos_cpu,
                particle_ids=particle_ids,
                element_ids=eid_cpu,
                extra_scalars=step_extras or None,
            )

    elapsed = _time.time() - t_start
    if exporter is not None:
        exporter.stop()
        if args.export_format == "vtkhdf":
            print(f"  Exported {exporter.n_exported} steps to "
                  f"{exporter.output_path}")
        else:
            print(f"  Exported {exporter.n_exported} VTU files")

    print("=" * 80)
    print(f"\nTRACKING SUMMARY")
    print("=" * 80)
    print(f"  Particles:   {n_particles:,}")
    print(f"  Steps:       {N_STEPS}")
    print(f"  Wall time:   {elapsed:.1f}s ({n_particles*N_STEPS/elapsed:,.0f} p·step/s)")
    print(f"  Output:      {output_subdir}")

    # Manifest
    manifest = {
        "velocity_source": "analytic",
        "velocity_module": str(Path(args.velocity_module).resolve()),
        "field_meta": provider.meta or {},
        "domain_bbox": list(map(list, provider.domain_bbox)),
        "n_particles": int(n_particles),
        "n_steps": int(N_STEPS),
        "dt": float(DT),
        "seed_source": args.seed_source,
        "started": datetime.datetime.now().isoformat(timespec='seconds'),
        "wall_time_s": float(elapsed),
        "throughput_p_step_per_s": float(n_particles * N_STEPS / elapsed),
    }
    with open(output_subdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Manifest:    {output_subdir}/manifest.json")
    return 0


def main():
    args = parse_args()

    # Fork on velocity source. The analytic path lives in
    # _run_analytic_tracking() below and shares all the post-RK4
    # plumbing (export, monitor, manifest) but skips the mesh-loading
    # stages and uses create_rk4_analytic() instead of
    # create_rk4_comparison(). Argument validation for the analytic
    # path is centralised at the top of that function so any user
    # configuration error fails before we touch the GPU.
    if args.velocity_source == "analytic":
        return _run_analytic_tracking(args)

    # ROM path validation: required inputs must exist before we touch
    # the mesh or the GPU. Fails fast with a clear message.
    if args.velocity_source == "rom":
        missing = []
        if not args.rom_basis:
            missing.append("--rom-basis")
        if not args.rom_romdata:
            missing.append("--rom-romdata")
        if missing:
            print(
                f"ERROR: --velocity-source=rom requires {', '.join(missing)}",
                file=sys.stderr,
            )
            return 2
        for label, path in (
            ("--rom-basis", args.rom_basis),
            ("--rom-romdata", args.rom_romdata),
        ):
            if not Path(path).is_file():
                print(f"ERROR: {label}: file not found: {path}",
                      file=sys.stderr)
                return 2

    case_stem = _resolve_case_paths(args)
    print(f"[case] stem='{case_stem}'  post_dir={args.input}")
    print(f"[case] mesh_dir='{args._mesh_dir}'  femuss_dir='{args._femuss_dir}'")
    print(f"[case] mesh_pattern='{args.mesh_pattern}'  "
          f"femuss_pattern='{args.femuss_pattern}'")

    # Paths / params
    MESH_BASE_PATH = args._mesh_dir
    VELOCITY_TIMESTEP_RANGE = tuple(args.vel_range)
    N_STEPS = args.n_steps
    DT = args.dt
    EXPORT_FREQUENCY = args.export_freq
    LOG_INTERVAL = args.log_interval

    # Apply CLI flags to config
    config.RK4_SUBSTEP_BBOX_CLAMP = args.bbox_clamp and not args.no_bbox_clamp
    config.RK4_BOUNDARY_PROJECTION = not args.no_boundary_proj
    config.RK4_BOUNDARY_PROJECTION_TOL = args.boundary_proj_tol
    config.RK4_LEVELSET_MASK = not args.no_levelset
    config.RK4_LEVELSET_MODE = args.levelset_mode
    config.RK4_FAILED_SUBSTAGE_POLICY = args.failed_substage
    config.RK4_SUBSTEP_LAST_VALID_VEL = False
    config.RK4_L0_SKIP_BOUNDARY_ELEMENTS = not args.no_l0_skip_boundary
    if args.point_in_tet_tol is not None:
        config.POINT_IN_TET_TOLERANCE = args.point_in_tet_tol
    if args.interpolation_det_min is not None:
        config.INTERPOLATION_DET_MIN = args.interpolation_det_min
    if args.registration is not None:
        config.OCTREE_REGISTRATION_METHOD = args.registration
    config.POINT_IN_TET_METHOD = POINT_IN_TET_METHOD

    INTERPOLATION_METHOD = args.interpolation_method
    L1_METHOD = args.l1_method
    L2_NEIGHBORHOOD = args.l2_neighborhood
    L0_SKIP_BAND = args.l0_skip_band
    ENHANCED_SEARCH_BAND = args.enhanced_search_band
    RK4_MODE = args.rk4_mode
    L2_VECTORIZED = args.l2_vectorized
    LEVELSET_FIELD = args.levelset_field
    VELOCITY_FIELD = args.velocity_field

    # Parse --boundary-walls (comma-separated 'wall=mode' pairs).
    #
    # Modes:
    #   clamp     — kernel snaps lost particles to bbox+tol and runs a
    #               recovery search (today's default). Use for solid
    #               walls / numerical jitter.
    #   outlet    — kernel does NOT snap on this axis; particles cross
    #               freely and stay element_id<0. No driver-side action.
    #   ballistic — kernel treats it as outlet; driver detects escape,
    #               snapshots the particle's last-step velocity, and
    #               advances x += dt * last_vel every step from then on.
    #   freeze    — kernel treats it as outlet; driver detects escape
    #               and freezes the position at the escape point.
    #
    # The kernel's clamp_to_bbox honors the clamp masks per-axis, so
    # only walls with mode='clamp' invoke the recovery snap. ballistic
    # and freeze are handled entirely driver-side; the kernel does not
    # see them. When no wall is ballistic or freeze, the driver attaches
    # no helpers and per-step cost is byte-identical to today.
    _valid_walls = set(_WALL_TO_AXIS.keys())
    _valid_modes = {"clamp", "outlet", "ballistic", "freeze"}
    if args.boundary_walls is not None:
        wall_dict = {}
        for pair in args.boundary_walls.split(','):
            if '=' not in pair:
                continue
            w, m = pair.split('=', 1)
            w, m = w.strip(), m.strip()
            if w not in _valid_walls:
                raise ValueError(
                    f"--boundary-walls: unknown wall '{w}'. "
                    f"Valid walls: {sorted(_valid_walls)}"
                )
            if m not in _valid_modes:
                raise ValueError(
                    f"--boundary-walls: unknown mode '{m}' for wall '{w}'. "
                    f"Valid modes: {sorted(_valid_modes)}"
                )
            wall_dict[w] = m
        config.RK4_BOUNDARY_WALLS = wall_dict or None
    else:
        config.RK4_BOUNDARY_WALLS = None

    # Inlet flag validation (incompatible with femuss / file seed sources).
    if getattr(args, "inlet_wall", "none") != "none":
        if args.seed_source in ("femuss", "file"):
            raise ValueError(
                f"--inlet-wall is not supported with "
                f"--seed-source={args.seed_source}"
            )

    use_pin_velocity = args.pin_velocity and not args.no_pin_velocity

    # Validate FEMUSS comparison pre-conditions
    if args.femuss_compare and args.seed_source != 'femuss':
        print("WARNING: --femuss-compare requires --seed-source=femuss; disabling comparison.")
        args.femuss_compare = False

    t_total_start = time.time()
    stage_times = {}

    print("=" * 80)
    print("JAXTrace Production Tracking")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"\nConfiguration:")
    print(f"  Input base:           {args.input}")
    print(f"  Mesh dir:             {MESH_BASE_PATH}")
    print(f"  Output dir:           {args.output}")
    print(f"  Seed source:          {args.seed_source}")
    print(f"  N_STEPS / DT:         {N_STEPS} / {DT}")
    print(f"  Velocity range:       {VELOCITY_TIMESTEP_RANGE[0]}-{VELOCITY_TIMESTEP_RANGE[1]}  field='{VELOCITY_FIELD}'")
    print(f"  L0/L1/L2:             L0={'on' if ENABLE_L0_SEARCH else 'off'}  "
          f"L1={L1_METHOD}  L2={L2_NEIGHBORHOOD}x{L2_NEIGHBORHOOD}x{L2_NEIGHBORHOOD}"
          + (" [vectorized]" if L2_VECTORIZED else ""))
    print(f"  RK4 mode:             {RK4_MODE}")
    print(f"  Precision:            {'float64' if config.USE_FLOAT64 else 'float32'}")
    print(f"  Point-in-tet tol:     {config.POINT_IN_TET_TOLERANCE:.0e}")
    print(f"  Interpolation method: {INTERPOLATION_METHOD}")
    print(f"  Bbox clamp:           {'on' if config.RK4_SUBSTEP_BBOX_CLAMP else 'off'}")
    print(f"  Failed substage:      {config.RK4_FAILED_SUBSTAGE_POLICY}")
    print(f"  Boundary projection:  {'on' if config.RK4_BOUNDARY_PROJECTION else 'off'} "
          f"(tol={config.RK4_BOUNDARY_PROJECTION_TOL:.0e})")
    print(f"  Boundary walls:       {config.RK4_BOUNDARY_WALLS or 'all clamp (default)'}")
    if getattr(args, "inlet_wall", "none") != "none":
        print(f"  Inlet:                wall={args.inlet_wall}  "
              f"v={args.inlet_velocity} m/s")
    else:
        print(f"  Inlet:                off")
    print(f"  Level-set mask:       {'on' if config.RK4_LEVELSET_MASK else 'off'} "
          f"(mode={config.RK4_LEVELSET_MODE}, field='{LEVELSET_FIELD}')")
    if use_pin_velocity:
        axis_str = (f"tilt={args.pin_tilt}°" if abs(args.pin_tilt) > 1e-12
                    else f"axis={args.pin_axis}")
        print(f"  Pin velocity:         on (RPM={args.pin_rpm}, center={args.pin_center}, {axis_str})")
    else:
        print(f"  Pin velocity:         off")
    print(f"  FEMUSS comparison:    {'on' if args.femuss_compare else 'off'}")
    print("=" * 80)

    # Build per-wall clamp masks for the kernel. Only walls with
    # mode='clamp' invoke the kernel's snap-to-bbox recovery. Every
    # other mode (outlet, ballistic, freeze) is non-clamp from the
    # kernel's point of view — it leaves the particle lost. Ballistic
    # and freeze are then picked up by driver-side helpers below.
    wall_config = config.RK4_BOUNDARY_WALLS
    if wall_config is None or not isinstance(wall_config, dict):
        clamp_min_mask = jnp.array([True, True, True])
        clamp_max_mask = jnp.array([True, True, True])
        _wall_modes = {w: 'clamp' for w in _WALL_TO_AXIS}
    else:
        _wall_modes = {w: wall_config.get(w, 'clamp') for w in _WALL_TO_AXIS}
        clamp_min_mask = jnp.array([
            _wall_modes['x_min'] == 'clamp',
            _wall_modes['y_min'] == 'clamp',
            _wall_modes['z_min'] == 'clamp',
        ])
        clamp_max_mask = jnp.array([
            _wall_modes['x_max'] == 'clamp',
            _wall_modes['y_max'] == 'clamp',
            _wall_modes['z_max'] == 'clamp',
        ])
    # Per-wall mode arrays for driver-side ballistic/freeze logic.
    # Encoding: 0=clamp, 1=outlet, 2=ballistic, 3=freeze. Six walls
    # laid out as [x_min, y_min, z_min, x_max, y_max, z_max].
    _MODE_CODE = {'clamp': 0, 'outlet': 1, 'ballistic': 2, 'freeze': 3}
    wall_mode_codes = np.array([
        _MODE_CODE[_wall_modes['x_min']],
        _MODE_CODE[_wall_modes['y_min']],
        _MODE_CODE[_wall_modes['z_min']],
        _MODE_CODE[_wall_modes['x_max']],
        _MODE_CODE[_wall_modes['y_max']],
        _MODE_CODE[_wall_modes['z_max']],
    ], dtype=np.int32)
    has_ballistic_or_freeze = bool(
        ((wall_mode_codes == 2) | (wall_mode_codes == 3)).any()
    )

    # ==================================================================
    # [1/7] Load mesh
    # ==================================================================
    t_stage = time.time()
    print(f"\n[1/7] Loading mesh...")
    # Load all requested fields in a single PVTU traversal. PVTU parse +
    # decompression is the dominant per-file cost (~50-200 MB per file on
    # large meshes); reading additional fields from the already-open VTK
    # output is essentially free. With temperature off this loads only
    # Displacement, identical wall time to the legacy single-field path.
    fields_to_load = [VELOCITY_FIELD]
    needs_temperature = args.track_max_temperature or args.export_temperature
    if needs_temperature:
        fields_to_load.append(args.temperature_field)

    if len(fields_to_load) == 1:
        # Use the legacy single-field loader so existing runs are unchanged.
        node_positions, connectivity, velocity_sequence = (
            load_velocity_sequence_from_pvtu(
                base_path=MESH_BASE_PATH,
                file_pattern=args.mesh_pattern,
                timestep_range=VELOCITY_TIMESTEP_RANGE,
                field_name=VELOCITY_FIELD,
                verbose=False,
            )
        )
        temperature_sequence = None
    else:
        # Multi-field path: one pass over the PVTUs for every needed field.
        from jaxtrace.gpu.mesh_loader_timedep import load_field_sequences_from_pvtu
        node_positions, connectivity, _fields = load_field_sequences_from_pvtu(
            base_path=MESH_BASE_PATH,
            file_pattern=args.mesh_pattern,
            timestep_range=VELOCITY_TIMESTEP_RANGE,
            field_names=fields_to_load,
            verbose=False,
        )
        velocity_sequence = _fields[VELOCITY_FIELD]
        temperature_sequence = _fields[args.temperature_field]
        print(f"    Loaded fields in single pass: {fields_to_load}")
        print(f"    Temperature: {temperature_sequence.shape}  "
              f"({temperature_sequence.nbytes / (1024**2):.1f} MB)")

    # ROM reconstruction path. When --velocity-source=rom the mesh path
    # loads the connectivity + node positions from the reference case's
    # PVTU (above) BUT DISCARDS the per-node velocity field it loaded
    # and replaces it with a POD-reconstructed field for the selected
    # case index. The reconstruction is a one-shot per-node compute
    # before the RK4 loop; downstream (dedup, GPU upload, tracking) is
    # completely unchanged.
    if args.velocity_source == "rom":
        from jaxtrace.rom.velocity_recon import precompute_case_velocity
        print("[rom] Reconstructing per-node velocity from POD basis + coeffs")
        rec = precompute_case_velocity(
            basis_path=args.rom_basis,
            romdata_path=args.rom_romdata,
            case_idx=args.rom_case_idx,
            formula=args.rom_formula,
            field_group=args.rom_field_group,
            verbose=True,
        )
        if rec.velocity.shape[0] != node_positions.shape[0]:
            raise RuntimeError(
                f"ROM node count ({rec.velocity.shape[0]}) does not match "
                f"mesh node count ({node_positions.shape[0]}). The ROM was "
                f"built on a different mesh than the --input case."
            )
        # Replace the mesh-loaded velocity with the ROM reconstruction.
        # The velocity_sequence shape is (n_timesteps, n_nodes, 3); we
        # keep n_timesteps=1 (steady field) and let the loader path's
        # cyclic-indexing return the same field for every RK4 step.
        velocity_sequence = rec.velocity.astype(config.FLOAT_DTYPE_NP)[None, :, :]
        print(f"[rom] velocity_sequence replaced: shape={velocity_sequence.shape}, "
              f"dtype={velocity_sequence.dtype}")

    print(f"  Elements: {connectivity.shape[0]:,}, Nodes: {node_positions.shape[0]:,}")

    print("  Deduplicating...")
    scalar_seqs = {'Temperature': temperature_sequence} if temperature_sequence is not None else None
    node_positions, connectivity, n_dup, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity,
        velocity_sequence=velocity_sequence,
        scalar_sequences=scalar_seqs,
        verbose=False,
    )
    if scalar_seqs is not None:
        temperature_sequence = scalar_seqs['Temperature']
    connectivity = connectivity.astype(np.int32)
    print(f"  Removed {n_dup:,} duplicates -> {node_positions.shape[0]:,} nodes")

    mesh_bbox_min_cpu = node_positions.min(axis=0).astype(config.FLOAT_DTYPE_NP)
    mesh_bbox_max_cpu = node_positions.max(axis=0).astype(config.FLOAT_DTYPE_NP)
    print(f"  Mesh bbox: [{mesh_bbox_min_cpu}] → [{mesh_bbox_max_cpu}]")

    # Level-set field
    levelset_cpu = None
    if config.RK4_LEVELSET_MASK:
        start_ts, end_ts = VELOCITY_TIMESTEP_RANGE
        ls_raw = None
        reader = None
        for ts in range(start_ts, end_ts + 1):
            ls_file = MESH_BASE_PATH / args.mesh_pattern.format(timestep=ts)
            reader = vtk.vtkXMLPUnstructuredGridReader()
            reader.SetFileName(str(ls_file))
            reader.Update()
            pd = reader.GetOutput().GetPointData()
            if pd.HasArray(LEVELSET_FIELD):
                ls_raw = vtk_to_numpy(pd.GetArray(LEVELSET_FIELD)).astype(np.float64)
                break
        if ls_raw is None:
            print(f"  WARNING: Level-set field '{LEVELSET_FIELD}' not found, disabling mask")
            config.RK4_LEVELSET_MASK = False
        else:
            n_raw = ls_raw.shape[0]
            ls_scalar = ls_raw.ravel()
            raw_pos = vtk_to_numpy(reader.GetOutput().GetPoints().GetData()).astype(np.float64)
            seen = {}
            node_map = np.zeros(n_raw, dtype=np.int32)
            new_id = 0
            for old_id in range(n_raw):
                key = tuple(raw_pos[old_id])
                if key not in seen:
                    seen[key] = new_id
                    new_id += 1
                node_map[old_id] = seen[key]
            n_dedup = node_positions.shape[0]
            levelset_cpu = np.zeros(n_dedup, dtype=config.FLOAT_DTYPE_NP)
            for old_id in range(n_raw):
                levelset_cpu[node_map[old_id]] = ls_scalar[old_id]
            n_neg = int(np.sum(levelset_cpu < 0))
            print(f"  Level-set loaded: {n_neg:,}/{n_dedup:,} inside tool "
                  f"({100*n_neg/n_dedup:.1f}%)")

    if use_pin_velocity:
        if levelset_cpu is None:
            print("  WARNING: --pin-velocity requires level-set field; skipping reconstruction")
        else:
            print(f"  Reconstructing pin velocity (RPM={args.pin_rpm}, "
                  f"center={args.pin_center}, tilt={args.pin_tilt}°)...")
            velocity_sequence, n_overwritten = reconstruct_pin_velocity(
                node_positions=node_positions,
                velocity_sequence=velocity_sequence,
                levelset=levelset_cpu,
                rpm=args.pin_rpm,
                center=np.array(args.pin_center),
                axis=np.array(args.pin_axis),
                tilt_deg=args.pin_tilt,
            )
            n_nodes = node_positions.shape[0]
            print(f"  Pin velocity applied to {n_overwritten:,}/{n_nodes:,} nodes "
                  f"({100*n_overwritten/n_nodes:.1f}%)")

    stage_times['1_load_mesh'] = time.time() - t_stage
    print(f"  Stage 1 time: {stage_times['1_load_mesh']:.1f}s")

    # ==================================================================
    # [2/7] Precompute metadata
    # ==================================================================
    t_stage = time.time()
    print(f"\n[2/7] Precomputing metadata...")
    aa_metadata = precompute_aa_metadata(connectivity, node_positions, verbose=False)
    element_vertices = precompute_element_vertices(connectivity, node_positions, verbose=False)
    M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)

    v0 = node_positions[connectivity[:, 0]]
    v1 = node_positions[connectivity[:, 1]]
    v2 = node_positions[connectivity[:, 2]]
    v3 = node_positions[connectivity[:, 3]]
    e1 = v1 - v0; e2 = v2 - v0; e3 = v3 - v0
    det_values = np.sum(e1 * np.cross(e2, e3), axis=1)
    element_volumes = np.abs(det_values) / 6.0
    stage_times['2_precompute'] = time.time() - t_stage
    print(f"  Stage 2 time: {stage_times['2_precompute']:.1f}s")

    # ==================================================================
    # [3/7] Build structures and upload
    # ==================================================================
    t_stage = time.time()
    print(f"\n[3/7] Building structures and uploading to GPU...")
    octree_struct = build_global_morton_octree(
        node_positions=node_positions, connectivity=connectivity,
        leaf_capacity=256, max_depth=21, verbose=False,
    )
    if config.OCTREE_REGISTRATION_METHOD == "parent_cube":
        mesh_octree_cells = extract_octree_cells_parent_cube(
            node_positions, connectivity, tolerance=1e-6, verbose=True,
            orphan_fallback=not args.no_orphan_fallback,
            hybrid_non_kuhn=not args.no_hybrid_non_kuhn,
        )
        print(f"  Parent-cube octree: {mesh_octree_cells.n_cells:,} cells, "
              f"{mesh_octree_cells.elements_per_cell_mean:.1f} elem/cell "
              f"(max {mesh_octree_cells.max_elements_per_cell})")
    else:
        mesh_octree_cells = extract_octree_cells_vertex_multi(
            node_positions, connectivity, tolerance=1e-6, verbose=False,
        )
        print(f"  Vertex-multi octree: {mesh_octree_cells.n_cells:,} cells, "
              f"{mesh_octree_cells.elements_per_cell_mean:.1f} elem/cell, "
              f"{mesh_octree_cells.cells_per_element_mean:.1f} cells/elem")

    element_neighbors_face = build_element_neighbors_array(
        connectivity, method='face', verbose=False)
    use_enhanced_band = (ENHANCED_SEARCH_BAND > 0 and levelset_cpu is not None)
    need_node_neighbors = (L1_METHOD == 'node') or use_enhanced_band
    element_neighbors_node = None
    if need_node_neighbors:
        element_neighbors_node = build_element_neighbors_array(
            connectivity, method='node', verbose=True)

    if L1_METHOD == 'node' and element_neighbors_node is not None:
        element_neighbors = element_neighbors_node
    else:
        element_neighbors = element_neighbors_face
    n_neighbors_per_element = element_neighbors.shape[1]

    mesh_gpu = upload_mesh_to_gpu(connectivity, node_positions, element_neighbors, verbose=False)
    morton_gpu = upload_global_morton_to_gpu(octree_struct, connectivity, node_positions)
    mesh_aligned_octree_multi_gpu = upload_mesh_aligned_octree_to_gpu(
        connectivity, node_positions, mesh_octree_cells, verbose=False,
    )

    aa_metadata_gpu = AxisAlignedMetadata(
        base_vertex_indices=jax.device_put(aa_metadata.base_vertex_indices),
        base_vertices=jax.device_put(aa_metadata.base_vertices),
        inv_edge_lengths=jax.device_put(aa_metadata.inv_edge_lengths),
        axis_indices=jax.device_put(aa_metadata.axis_indices),
        is_axis_aligned=jax.device_put(aa_metadata.is_axis_aligned),
    )
    element_vertices_gpu = jax.device_put(element_vertices)
    M_inv_gpu = jax.device_put(M_inv_array)
    p0_gpu = jax.device_put(p0_array)
    element_volumes_gpu = jax.device_put(element_volumes.astype(config.FLOAT_DTYPE_NP))
    set_corrected_metadata(aa_metadata_gpu, element_vertices_gpu)
    set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)

    velocity_sequence_gpu = jax.device_put(velocity_sequence)

    # ------------------------------------------------------------------
    # Gradient recovery + higher-order velocity reconstruction.
    #
    # See docs/gradient_recovery_pipeline.md for the full derivation. This
    # runs ONCE after velocity_sequence is finalised (post-dedup, post-
    # pin-velocity, post-ROM/analytic-substitution) and produces three
    # per-element arrays that the RK4 kernel samples in place of raw P1
    # barycentric interpolation.
    #
    # Only supports single-timestep velocity fields for now. For
    # time-dependent PVTU sequences (vel-range with multiple frames)
    # we skip the recovery and fall back to raw P1 (which is what the
    # kernel does when the taylor arrays are None).
    # ------------------------------------------------------------------
    element_centroid_gpu = None
    element_v_centroid_gpu = None
    element_gradient_gpu = None
    if args.gradient_recovery == 1:
        n_frames = velocity_sequence.shape[0]
        if n_frames == 1:
            from jaxtrace.gpu.recovery.gradient_recovery import build_recovery
            print(f"\n[gradient-recovery] Building higher-order "
                  f"velocity reconstruction (method='{args.recovery_method}')")
            recovery = build_recovery(
                node_positions=node_positions,
                connectivity=connectivity,
                node_velocities=velocity_sequence[0],
                method=args.recovery_method,
                verbose=True,
            )
            element_centroid_gpu = jax.device_put(recovery.element_centroid)
            element_v_centroid_gpu = jax.device_put(recovery.element_v_centroid)
            element_gradient_gpu = jax.device_put(recovery.element_gradient)
            for k, v in recovery.stage_times.items():
                stage_times[f'gr_{k}'] = v
        else:
            print(f"\n[gradient-recovery] SKIPPED: velocity_sequence has "
                  f"{n_frames} frames; only single-frame (steady) fields "
                  f"are supported today. Falling back to raw P1 "
                  f"barycentric interpolation.")
    else:
        print(f"\n[gradient-recovery] disabled by --gradient-recovery=0")

    # Optional temperature stack (one extra (n_timesteps, n_nodes) scalar field).
    temperature_sequence_gpu = None
    if temperature_sequence is not None:
        temperature_sequence_gpu = jax.device_put(temperature_sequence)
        print(f"  Temperature stack on GPU: {temperature_sequence.shape}  "
              f"({temperature_sequence.nbytes / (1024**2):.1f} MB)")

    mesh_bbox_min_gpu = jax.device_put(mesh_bbox_min_cpu)
    mesh_bbox_max_gpu = jax.device_put(mesh_bbox_max_cpu)
    levelset_gpu = jax.device_put(levelset_cpu) if levelset_cpu is not None else None

    boundary_elements_gpu = None
    enhanced_elements_gpu = None
    element_neighbors_node_gpu = None
    if levelset_cpu is not None:
        node_ls = levelset_cpu[connectivity]
        has_positive = np.any(node_ls >= 0, axis=1)
        has_negative = np.any(node_ls < 0, axis=1)
        min_abs_ls = np.min(np.abs(node_ls), axis=1)
        if config.RK4_L0_SKIP_BOUNDARY_ELEMENTS:
            is_l0_skip = has_positive & has_negative
            if L0_SKIP_BAND > 0:
                is_l0_skip = is_l0_skip | (min_abs_ls < L0_SKIP_BAND)
            boundary_elements_gpu = jax.device_put(is_l0_skip)
        if use_enhanced_band:
            is_enhanced = (min_abs_ls < ENHANCED_SEARCH_BAND)
            enhanced_elements_gpu = jax.device_put(is_enhanced)
            element_neighbors_node_gpu = jax.device_put(element_neighbors_node)
    stage_times['3_build_upload'] = time.time() - t_stage
    print(f"  Stage 3 time: {stage_times['3_build_upload']:.1f}s")

    # ==================================================================
    # [4/7] Seed particles
    # ==================================================================
    t_stage = time.time()
    print(f"\n[4/7] Seeding particles...")
    femuss_start_data = None
    if args.seed_source == 'femuss':
        particle_positions, particle_ids, femuss_start_data = seed_from_femuss(args)
    elif args.seed_source in ('box', 'box-frac'):
        particle_positions, particle_ids, _ = seed_from_box(
            args, mesh_bbox_min_cpu, mesh_bbox_max_cpu,
        )
    elif args.seed_source in ('grid', 'grid-frac'):
        particle_positions, particle_ids, _ = seed_from_grid(
            args, mesh_bbox_min_cpu, mesh_bbox_max_cpu,
        )
    else:  # file
        particle_positions, particle_ids, _ = seed_from_file(args)

    n_particles = particle_positions.shape[0]
    stage_times['4_seed'] = time.time() - t_stage
    print(f"  Stage 4 time: {stage_times['4_seed']:.1f}s")

    # ==================================================================
    # [5/7] Initial assignment
    # ==================================================================
    t_stage = time.time()
    print(f"\n[5/7] Initial assignment...")
    positions_gpu = jax.device_put(particle_positions)
    element_ids_initial = initial_assignment_mesh_aligned_multi_local(
        positions_gpu, mesh_aligned_octree_multi_gpu,
        batch_size=50000, max_tests=600, verbose=True,
    )
    element_ids_initial = jax.block_until_ready(element_ids_initial)
    n_assigned = int(jnp.sum(element_ids_initial >= 0))
    print(f"  Assigned: {n_assigned:,}/{n_particles:,} ({100*n_assigned/n_particles:.2f}%)")
    if n_particles - n_assigned > 0:
        print(f"  WARNING: {n_particles - n_assigned:,} particles not assigned to any element")

    # ------------------------------------------------------------------
    # Inlet drift setup. Particles seeded outside the mesh on the inlet
    # face are kept alive with element_id=-1 and given a drift velocity
    # equal to INLET_VELOCITY * inward_normal. Pre-step driver hook
    # advances those particles by dt*drift each step until the kernel's
    # internal search finds a host element (element_id becomes >= 0).
    # Particles whose initial assignment failed on a non-inlet face are
    # left with element_id=-1 and zero drift (frozen). All active
    # particles have zero drift.
    # ------------------------------------------------------------------
    # Pending-entry state is tracked by an explicit boolean array so it
    # is independent of drift_vel (which a future ballistic/freeze mode
    # may want to set non-zero on escaped particles too) and independent
    # of element_id (which the kernel's boundary-projection recovery may
    # flip back to >= 0 by snapping the particle to bbox+tol).
    inlet_wall = getattr(args, "inlet_wall", "none")
    inlet_velocity = float(getattr(args, "inlet_velocity", 0.0))
    drift_vel = np.zeros((n_particles, 3), dtype=config.FLOAT_DTYPE_NP)
    pending_entry = np.zeros(n_particles, dtype=bool)
    n_pending_inlet = 0
    if inlet_wall != "none":
        if inlet_velocity == 0.0:
            print(f"  [inlet] WARNING: --inlet-wall={inlet_wall} set but "
                  f"--inlet-velocity=0; pending particles will never enter.")
        axis, side = _WALL_TO_AXIS[inlet_wall]
        inward = _inlet_inward_normal(inlet_wall)
        eid_cpu_initial = np.asarray(element_ids_initial, dtype=np.int32)
        positions_cpu_initial = np.asarray(particle_positions,
                                           dtype=config.FLOAT_DTYPE_NP)
        # A pending-entry particle is any failed-assignment particle that
        # sits on the outside of the inlet face.
        if side == "min":
            is_outside = positions_cpu_initial[:, axis] < mesh_bbox_min_cpu[axis]
        else:
            is_outside = positions_cpu_initial[:, axis] > mesh_bbox_max_cpu[axis]
        is_pending = (eid_cpu_initial < 0) & is_outside
        n_pending_inlet = int(is_pending.sum())
        if n_pending_inlet:
            drift_vec = (inward * inlet_velocity).astype(config.FLOAT_DTYPE_NP)
            drift_vel[is_pending] = drift_vec
            pending_entry[is_pending] = True
            print(f"  [inlet] {n_pending_inlet:,} pending-entry particles on "
                  f"{inlet_wall} face")
            print(f"  [inlet] drift = {tuple(float(v) for v in drift_vec)} m/s")
    n_frozen = int(((element_ids_initial < 0) & ~pending_entry).sum()) \
        if np.asarray(element_ids_initial < 0).any() else 0
    if n_frozen and inlet_wall != "none":
        # Frozen = failed assignment AND not pending-entry.
        print(f"  [inlet] {n_frozen:,} particles failed assignment off the "
              f"inlet face; kept with element_id=-1 and zero drift.")

    drift_vel_gpu = jax.device_put(drift_vel)
    pending_entry_gpu = jax.device_put(pending_entry)

    # Per-particle escape-mode state for ballistic/freeze walls.
    # 0=normal (in domain or just lost waiting for kernel recovery),
    # 1=ballistic (advance by stored escape_vel each step),
    # 2=frozen (position never updates again).
    # Latched on first escape; never reset.
    if has_ballistic_or_freeze:
        escape_mode_gpu = jnp.zeros(n_particles, dtype=jnp.uint8)
        escape_vel_gpu = jnp.zeros((n_particles, 3),
                                   dtype=config.FLOAT_DTYPE_NP)
        last_vel_gpu = jnp.zeros((n_particles, 3),
                                 dtype=config.FLOAT_DTYPE_NP)
    else:
        escape_mode_gpu = None
        escape_vel_gpu = None
        last_vel_gpu = None

    stage_times['5_initial_assign'] = time.time() - t_stage
    print(f"  Stage 5 time: {stage_times['5_initial_assign']:.1f}s")

    # ==================================================================
    # [6/7] Build RK4 and warmup
    # ==================================================================
    t_stage = time.time()
    print(f"\n[6/7] Building RK4...")
    rk4_step = create_rk4_comparison(
        mesh_gpu_connectivity=mesh_gpu.connectivity,
        mesh_gpu_node_positions=mesh_gpu.node_positions,
        mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
        mesh_gpu_element_volumes=element_volumes_gpu,
        mesh_gpu_global_morton=morton_gpu,
        mesh_aligned_octree=mesh_aligned_octree_multi_gpu,
        enable_l0=ENABLE_L0_SEARCH,
        enable_l1=ENABLE_L1_SEARCH,
        n_hops=N_HOPS,
        n_neighbors_per_element=n_neighbors_per_element,
        mesh_bbox_min=mesh_bbox_min_gpu,
        mesh_bbox_max=mesh_bbox_max_gpu,
        clamp_min_mask=clamp_min_mask,
        clamp_max_mask=clamp_max_mask,
        levelset_gpu=levelset_gpu,
        boundary_elements_gpu=boundary_elements_gpu,
        l2_neighborhood=L2_NEIGHBORHOOD,
        enhanced_elements_gpu=enhanced_elements_gpu,
        element_neighbors_node_gpu=element_neighbors_node_gpu,
        interpolation_method=INTERPOLATION_METHOD,
        M_inv_gpu=M_inv_gpu,
        p0_gpu=p0_gpu,
        rk4_mode=RK4_MODE,
        use_l2_vectorized=L2_VECTORIZED,
        element_centroid_gpu=element_centroid_gpu,
        element_v_centroid_gpu=element_v_centroid_gpu,
        element_gradient_gpu=element_gradient_gpu,
    )

    # Per-step inlet drift helper. Drifts only particles whose
    # pending_entry bit is set. The bit is one-shot: cleared by
    # _suppress_pending_recovery the moment a particle's drifted
    # position crosses the mesh bbox.
    @jax.jit
    def _apply_inlet_drift(positions, pending, dt, drift):
        new_positions = positions + dt * drift
        return jnp.where(pending[:, None], new_positions, positions)

    # Post-step recovery for pending-entry particles.
    #
    # The RK4 kernel's boundary-projection logic (--boundary-proj, on by
    # default) snaps any particle with element_id<0 to bbox+tol and runs
    # a recovery search there that almost always succeeds — so the
    # kernel returns element_id >= 0 even though geometrically the
    # particle is still outside the mesh. We therefore cannot use
    # element_id<0 as the "still pending" signal.
    #
    # The criterion used here is geometric: a pending particle remains
    # pending while its *pre-kernel* drifted position lies outside the
    # mesh bbox. The moment the drifted position crosses the bbox the
    # pending bit is cleared and its drift_vel is zeroed; from then on
    # the particle is handled by the normal active/lost path. This is a
    # one-shot transition — if the particle later gets lost again, that
    # is a separate concern (future ballistic/freeze mode), not a
    # re-entry into inlet drift.
    bbox_min_gpu = jnp.asarray(mesh_bbox_min_cpu, dtype=config.FLOAT_DTYPE_NP)
    bbox_max_gpu = jnp.asarray(mesh_bbox_max_cpu, dtype=config.FLOAT_DTYPE_NP)

    @jax.jit
    def _suppress_pending_recovery(
        positions_post, element_ids_post, positions_pre_kernel,
        pending, drift,
    ):
        inside_bbox = (
            (positions_pre_kernel >= bbox_min_gpu).all(axis=-1)
            & (positions_pre_kernel <= bbox_max_gpu).all(axis=-1)
        )
        # Still pending: was pending coming in AND drifted position is
        # still outside the bbox. Restore the kernel-overwritten
        # position and force element_id back to -1.
        still_pending = pending & (~inside_bbox)
        positions_out = jnp.where(
            still_pending[:, None], positions_pre_kernel, positions_post,
        )
        element_ids_out = jnp.where(
            still_pending, jnp.int32(-1), element_ids_post,
        )
        # Just-entered: was pending, now inside bbox. Zero its drift
        # and clear the pending bit. From this step on it behaves like
        # a normal particle.
        entered = pending & inside_bbox
        new_drift = jnp.where(entered[:, None], jnp.zeros_like(drift), drift)
        new_pending = pending & (~inside_bbox)
        return positions_out, element_ids_out, new_drift, new_pending

    apply_inlet_drift = _apply_inlet_drift if n_pending_inlet else None
    suppress_pending_recovery = (
        _suppress_pending_recovery if n_pending_inlet else None
    )

    # ------------------------------------------------------------------
    # Ballistic / freeze driver-side helpers. These are built only when
    # at least one wall is ballistic or freeze, so all-outlet / all-clamp
    # runs pay zero per-step cost from this block.
    # ------------------------------------------------------------------
    if has_ballistic_or_freeze:
        # Per-wall mode codes as a JAX array, ordered as
        # [x_min, y_min, z_min, x_max, y_max, z_max].
        # 0=clamp, 1=outlet, 2=ballistic, 3=freeze.
        wall_modes_gpu = jnp.asarray(wall_mode_codes, dtype=jnp.int32)

        @jax.jit
        def _classify_escape_mode(positions_post):
            """For each particle return the wall mode it would inherit
            if it escaped at this position. Uses axis priority: a
            particle outside on multiple faces takes the first match in
            order [x_min, y_min, z_min, x_max, y_max, z_max]. Returns
            the per-particle uint8 mode code (0..3); for particles
            inside the bbox the returned value is meaningless and is
            masked away by the caller.
            """
            below = positions_post < bbox_min_gpu
            above = positions_post > bbox_max_gpu
            # 6-bit one-hot over [x_min,y_min,z_min,x_max,y_max,z_max].
            # Priority via first-true selection on each face in order.
            # Default mode for "no face matches" is 1 (outlet) — never
            # actually used (gated by element_id<0 caller), but a sane
            # neutral value if it leaks through.
            mode = jnp.full(positions_post.shape[:-1], jnp.int32(1))
            # Apply in reverse priority so earlier faces overwrite later.
            faces = [
                (above[..., 2], wall_modes_gpu[5]),  # z_max
                (above[..., 1], wall_modes_gpu[4]),  # y_max
                (above[..., 0], wall_modes_gpu[3]),  # x_max
                (below[..., 2], wall_modes_gpu[2]),  # z_min
                (below[..., 1], wall_modes_gpu[1]),  # y_min
                (below[..., 0], wall_modes_gpu[0]),  # x_min
            ]
            for hit, code in faces:
                mode = jnp.where(hit, code, mode)
            return mode.astype(jnp.uint8)

        @jax.jit
        def _latch_and_advance_escape(
            positions_post, positions_pre, element_ids_post,
            escape_mode, escape_vel, last_vel, dt,
        ):
            """One JIT for the full ballistic/freeze step.

            Inputs:
              positions_post, element_ids_post: kernel output this step.
              positions_pre: particle positions before this step's
                rk4_step (and after apply_inlet_drift, if any).
              escape_mode: per-particle latched mode (0=normal,
                2=ballistic, 3=freeze).
              escape_vel: per-particle stored ballistic velocity.
              last_vel: per-particle running estimate of the last
                in-domain velocity (updated for active particles).
              dt: time step.

            Returns updated (positions, element_ids, escape_mode,
            escape_vel, last_vel).
            """
            # Update last_vel for particles that were active going into
            # this step (element_ids_post >= 0 captures particles that
            # the kernel kept; element_ids_post may be <0 for newly-
            # escaped particles, in which case last_vel from a prior
            # step is what we'll latch). Use the post-kernel position
            # delta as the effective velocity.
            active = element_ids_post >= 0
            v_est = (positions_post - positions_pre) / dt
            new_last_vel = jnp.where(active[:, None], v_est, last_vel)

            # Detect freshly-escaped particles: lost this step and not
            # already latched. Classify their face mode.
            lost = element_ids_post < 0
            not_latched = escape_mode == 0
            newly_escaped = lost & not_latched
            classified = _classify_escape_mode(positions_post)
            # Latch only ballistic/freeze; outlet/clamp stay mode=0.
            mode_is_active = (classified == 2) | (classified == 3)
            latch_now = newly_escaped & mode_is_active
            new_escape_mode = jnp.where(latch_now, classified, escape_mode)
            # Snapshot the escape velocity for ballistic particles at
            # the moment of latch. Use last_vel from PRIOR step (i.e.,
            # before this step's update), since v_est for a newly-lost
            # particle reflects its (probably erratic) motion outside.
            latch_ballistic = latch_now & (classified == 2)
            new_escape_vel = jnp.where(
                latch_ballistic[:, None], last_vel, escape_vel,
            )

            # Advance already-latched particles. For ballistic, the
            # kernel may have moved them — we override with the
            # ballistic trajectory based on pre-step position. For
            # freeze, restore the pre-step position.
            is_ballistic = new_escape_mode == 2
            is_freeze = new_escape_mode == 3
            ballistic_pos = positions_pre + dt * new_escape_vel
            positions_out = jnp.where(
                is_ballistic[:, None], ballistic_pos, positions_post,
            )
            positions_out = jnp.where(
                is_freeze[:, None], positions_pre, positions_out,
            )
            # Latched particles keep element_id = -1.
            latched_now = is_ballistic | is_freeze
            element_ids_out = jnp.where(
                latched_now, jnp.int32(-1), element_ids_post,
            )
            return (positions_out, element_ids_out,
                    new_escape_mode, new_escape_vel, new_last_vel)

        latch_and_advance_escape = _latch_and_advance_escape
    else:
        latch_and_advance_escape = None

    print(f"  Compiling...")
    t_compile = time.time()
    if apply_inlet_drift is not None:
        positions_gpu = apply_inlet_drift(
            positions_gpu, pending_entry_gpu, DT, drift_vel_gpu,
        )
    positions_gpu, element_ids_gpu = rk4_step(
        positions_gpu, element_ids_initial, DT, velocity_sequence_gpu, 0
    )
    jax.block_until_ready(positions_gpu)
    stage_times['6_compile'] = time.time() - t_compile
    print(f"  Compilation: {stage_times['6_compile']:.1f}s")
    stage_times['6_build_rk4'] = time.time() - t_stage

    # Build the per-step temperature interpolation kernel. Returns a tuple
    # (T_at_pos, new_max) so a single P1 evaluation covers both
    # --export-temperature (instantaneous field) and --track-max-temperature
    # (running maximum). Enabling both costs the same as enabling either one.
    # The host element id and the new position from rk4_step are already
    # known, so this is just a P1 scalar interpolation per particle — no
    # fresh spatial search.
    interpolate_temperature_step = None
    if needs_temperature:
        conn_gpu = jax.device_put(connectivity)
        node_pos_gpu = jax.device_put(node_positions.astype(config.FLOAT_DTYPE_NP))

        @jax.jit
        def interpolate_temperature_step(positions, element_ids, time_idx, max_so_far):
            """For each particle, P1-interpolate temperature at the current
            position in the cyclically-indexed temperature field. Returns the
            instantaneous values and the elementwise max against ``max_so_far``.
            Particles with element_id<0 keep their previous max and report 0.0
            for the instantaneous value (caller can ignore via the Escaped
            flag)."""
            n_t = temperature_sequence_gpu.shape[0]
            T_field = temperature_sequence_gpu[time_idx % n_t]    # (n_nodes,)

            def per_particle(pos, eid, prev_max):
                valid = (eid >= 0) & (eid < conn_gpu.shape[0])
                safe_eid = jnp.where(valid, eid, 0)
                nodes_idx = conn_gpu[safe_eid]                    # (4,)
                nodes = node_pos_gpu[nodes_idx]                   # (4, 3)
                node_T = T_field[nodes_idx]                       # (4,)

                # Barycentric weights from the tet's local frame.
                v0 = nodes[1] - nodes[0]
                v1 = nodes[2] - nodes[0]
                v2 = nodes[3] - nodes[0]
                vp = pos - nodes[0]
                d00 = jnp.dot(v0, v0); d01 = jnp.dot(v0, v1); d02 = jnp.dot(v0, v2)
                d11 = jnp.dot(v1, v1); d12 = jnp.dot(v1, v2); d22 = jnp.dot(v2, v2)
                dp0 = jnp.dot(vp, v0); dp1 = jnp.dot(vp, v1); dp2 = jnp.dot(vp, v2)
                det = (d00 * (d11*d22 - d12*d12)
                       - d01 * (d01*d22 - d02*d12)
                       + d02 * (d01*d12 - d02*d11))
                det = jnp.where(jnp.abs(det) < config.INTERPOLATION_DET_MIN,
                                config.INTERPOLATION_DET_MIN, det)
                b1 = (dp0*(d11*d22 - d12*d12) - d01*(dp1*d22 - dp2*d12)
                      + d02*(dp1*d12 - dp2*d11)) / det
                b2 = (d00*(dp1*d22 - dp2*d12) - dp0*(d01*d22 - d02*d12)
                      + d02*(d01*dp2 - d02*dp1)) / det
                b3 = (d00*(d11*dp2 - d12*dp1) - d01*(d01*dp2 - d02*dp1)
                      + dp0*(d01*d12 - d02*d11)) / det
                b0 = 1.0 - b1 - b2 - b3

                T_at_pos = b0 * node_T[0] + b1 * node_T[1] + b2 * node_T[2] + b3 * node_T[3]
                T_instant = jnp.where(valid, T_at_pos, jnp.float32(0.0))
                new_max = jnp.where(valid, jnp.maximum(prev_max, T_at_pos), prev_max)
                return T_instant, new_max

            return jax.vmap(per_particle)(positions, element_ids, max_so_far)

    # Output directory
    if args.run_tag:
        output_subdir = args.output / args.run_tag
    elif args.seed_source == 'femuss':
        output_subdir = args.output / f"femuss_{args.femuss_start}_to_{args.femuss_start + N_STEPS}"
    else:
        output_subdir = args.output / f"run_{args.seed_source}_n{n_particles}_s{N_STEPS}"
    output_subdir.mkdir(parents=True, exist_ok=True)
    print(f"  Output subdir: {output_subdir}")

    # Stats CSV
    stats_csv_path = output_subdir / "search_stats.csv"
    stats_csv = open(stats_csv_path, 'w')
    stats_csv.write("step,n_active,n_lost,new_lost\n")

    # Particle exporter — VTKHDF (default; single transient archive) or
    # legacy VTU+.pvd path (one file per step).
    EXPORT_ELEMENT_IDS = args.export_element_ids
    N_GROUPS = args.n_groups if not args.no_groups else 0
    exporter = None
    if not args.no_export:
        if args.export_format == "vtkhdf":
            exporter = VTKHDFExportThread(
                output_subdir,
                n_particles_hint=int(particle_positions.shape[0]),
            )
            print(f"  Particle export: VTKHDF -> {exporter.output_path}")
        else:
            exporter = VTKExportThread(output_subdir)
            print(f"  Particle export: VTU+.pvd -> {output_subdir}")
        exporter.start()

    # Signal handler for graceful shutdown on SIGTERM (SLURM timeout) and
    # SIGINT (Ctrl-C). Flushes pending writes and closes the file so the
    # archive is recoverable. Re-raises the default action so Python exits
    # cleanly after the writer has drained.
    import signal as _signal
    _shutdown_called = [False]
    # Predeclare density_runner so the signal handler's closure can see it
    # even if the runner is never assigned (--density-enable not set).
    density_runner = None

    def _graceful_shutdown(signum, frame):
        if _shutdown_called[0]:
            return
        _shutdown_called[0] = True
        print(f"\n[!] Received signal {signum}; flushing exporter and exiting.")
        try:
            if exporter is not None:
                exporter.stop()
        except Exception as e:
            print(f"  Exporter shutdown error: {e}")
        try:
            if density_runner is not None:
                density_runner.close()
        except Exception as e:
            print(f"  Density runner shutdown error: {e}")
        # Re-raise so the interpreter exits with the conventional 128+signum.
        _signal.signal(signum, _signal.SIG_DFL)
        import os as _os
        _os.kill(_os.getpid(), signum)

    _signal.signal(_signal.SIGTERM, _graceful_shutdown)
    _signal.signal(_signal.SIGINT, _graceful_shutdown)

    # Re-initialize for actual run
    positions_gpu = jax.device_put(particle_positions)
    element_ids_gpu = element_ids_initial

    # Particle groups by initial X
    particle_groups = None
    if N_GROUPS > 0:
        initial_x = particle_positions[:, 0]
        x_min, x_max = float(initial_x.min()), float(initial_x.max())
        x_range = x_max - x_min
        if x_range > 0:
            g = (initial_x - x_min) / x_range * N_GROUPS
            particle_groups = np.clip(g.astype(np.int32), 0, N_GROUPS - 1).astype(np.uint8)
        else:
            particle_groups = np.zeros(n_particles, dtype=np.uint8)

    extra_scalars = {'Group': particle_groups} if particle_groups is not None else None

    # Per-particle trajectory state (GPU-resident, updated every RK4 step).
    #   escaped_ever_gpu     -- True if element_id has ever been <0
    #   temperature_gpu      -- instantaneous P1-interpolated temperature at the
    #                           current position (only populated when
    #                           --export-temperature is set)
    #   max_temperature_gpu  -- running maximum of the interpolated temperature
    #                           (only populated when --track-max-temperature)
    escaped_ever_gpu = None
    temperature_gpu = None
    max_temperature_gpu = None
    if args.export_escaped_flag:
        escaped_ever_gpu = jnp.zeros(n_particles, dtype=jnp.bool_)

        @jax.jit
        def _update_escaped(prev_flag, element_ids, pending):
            # Pending-entry particles legitimately have element_id<0 —
            # they have not "escaped", they have not yet arrived. Mask
            # them out so the flag flips only when a previously-active
            # particle leaves the domain.
            return prev_flag | ((element_ids < 0) & (~pending))

    if args.export_temperature:
        temperature_gpu = jnp.zeros((n_particles,), dtype=config.FLOAT_DTYPE_JNP)
    if args.track_max_temperature:
        max_temperature_gpu = jnp.full(
            (n_particles,), -jnp.inf, dtype=config.FLOAT_DTYPE_JNP,
        )

    def _run_temperature_step(positions, element_ids, time_idx):
        """Wrapper that calls the joint T_instant/T_max kernel and updates
        whichever globals are active. Returns nothing; mutates the GPU
        state arrays declared above via assignment in the enclosing scope.

        Centralises the per-step temperature work so it only happens once
        per step regardless of whether one or both flags are on.
        """
        nonlocal temperature_gpu, max_temperature_gpu
        if interpolate_temperature_step is None:
            return
        # When --track-max-temperature is off, the kernel still wants a valid
        # max_so_far argument; pass a -inf placeholder we then discard.
        prev_max = (max_temperature_gpu if max_temperature_gpu is not None
                    else jnp.full((n_particles,), -jnp.inf,
                                  dtype=config.FLOAT_DTYPE_JNP))
        T_inst, new_max = interpolate_temperature_step(
            positions, element_ids, time_idx, prev_max,
        )
        if args.export_temperature:
            temperature_gpu = T_inst
        if args.track_max_temperature:
            max_temperature_gpu = new_max

    def _collect_temperature_extras(extras: dict) -> None:
        """Pull whichever temperature fields are enabled into ``extras``."""
        if args.export_temperature and temperature_gpu is not None:
            extras['Temperature'] = np.asarray(temperature_gpu, dtype=np.float32)
        if args.track_max_temperature and max_temperature_gpu is not None:
            extras['MaxTemperature'] = np.asarray(
                max_temperature_gpu, dtype=np.float32,
            )

    # --- Density runner (opt-in; zero overhead when --density-enable is unset) -
    density_runner = None
    density_export_freq = args.density_export_freq or args.export_freq
    if args.density_enable:
        try:
            from jaxtrace.density import DensityRunner, DensityRunnerConfig

            density_output_dir = (
                args.density_output_dir if args.density_output_dir is not None
                else str(output_subdir / "density")
            )
            density_bounds = None
            if args.density_bounds is not None:
                bb = args.density_bounds
                density_bounds = ((bb[0], bb[1]), (bb[2], bb[3]), (bb[4], bb[5]))

            # Resolve per-axis variants — when set they override the scalar form.
            _density_resolution_cfg = (
                args.density_resolution_xyz
                if args.density_resolution_xyz is not None
                else args.density_resolution
            )
            _density_voxel_size_cfg = (
                args.density_voxel_size_xyz
                if args.density_voxel_size_xyz is not None
                else args.density_voxel_size
            )
            _density_bandwidth_cfg = (
                args.density_bandwidth_xyz
                if args.density_bandwidth_xyz is not None
                else args.density_bandwidth
            )
            density_cfg = DensityRunnerConfig(
                bounds_mode="explicit" if density_bounds is not None else args.density_bounds_from,
                bounds=density_bounds,
                resolution=None if _density_voxel_size_cfg is not None else _density_resolution_cfg,
                voxel_size=_density_voxel_size_cfg,
                voxel_size_from_particles=args.density_voxel_size_from_particles,
                pad_fraction=args.density_pad_fraction,
                mask_inside_mesh=not args.density_no_mask_inside_mesh,
                kernel=args.density_kernel,
                bandwidth_mode=args.density_bandwidth_mode,
                bandwidth=_density_bandwidth_cfg,
                bandwidth_factor=args.density_bandwidth_factor,
                bandwidth_refresh_every=args.density_bandwidth_refresh_every,
                knn_k=args.density_knn_k,
                knn_safety=args.density_knn_safety,
                normalization=args.density_normalization,
                engine=args.density_engine,
                auto_threshold=args.density_auto_threshold,
                brute_query_chunk=args.density_brute_query_chunk,
                octree_target_n_per_cell=args.density_octree_target_n_per_cell,
                particle_bucket=args.density_particle_bucket,
                eval_on_grid=True,
                eval_at_particles=not args.density_no_particle_density,
                write_per_step=not args.density_no_per_step,
                write_time_average=not args.density_no_time_average,
                output_format=args.density_output_format,
                output_dir=density_output_dir,
                filename_stem=args.density_filename_stem,
                compression=args.density_compression,
                compression_opts=args.density_compression_opts,
                blosc_threads=args.density_blosc_threads,
            )

            density_runner = DensityRunner(
                cfg=density_cfg,
                mesh_octree_gpu=mesh_aligned_octree_multi_gpu,
                mesh_bbox_min=mesh_bbox_min_cpu,
                mesh_bbox_max=mesh_bbox_max_cpu,
                initial_positions=particle_positions,
            )
            print(f"  Density estimation ENABLED: kernel={args.density_kernel}, "
                  f"bandwidth_mode={args.density_bandwidth_mode}, "
                  f"engine={args.density_engine}, grid={density_runner.voxel_grid.resolution}, "
                  f"output={density_output_dir}")
        except Exception as e:
            print(f"  [!] Density runner construction failed: {e}")
            print(f"  [!] Continuing tracking without density output.")
            density_runner = None

    # Export initial state
    if exporter is not None:
        pos_cpu = np.array(positions_gpu, dtype=config.FLOAT_DTYPE_NP)
        eid_cpu = np.array(element_ids_initial, dtype=np.int32) if EXPORT_ELEMENT_IDS else None
        init_extras = dict(extra_scalars) if extra_scalars else {}
        if args.export_escaped_flag:
            init_extras['Escaped'] = np.zeros(n_particles, dtype=np.uint8)
        if needs_temperature:
            # One temperature evaluation at t=0 so the initial export is
            # non-empty for the user (both Temperature and MaxTemperature if
            # those fields are enabled).
            _run_temperature_step(positions_gpu, element_ids_initial, 0)
            _collect_temperature_extras(init_extras)
        exporter.enqueue_export(0, pos_cpu, particle_ids=particle_ids,
                                element_ids=eid_cpu,
                                extra_scalars=init_extras or extra_scalars)
        print(f"  Exported initial state (step 0)")

    print(f"\n[7/7] Running {N_STEPS} RK4 steps...")
    print("=" * 80)
    t_start = time.time()
    prev_lost = 0
    for step in range(1, N_STEPS + 1):
        if apply_inlet_drift is not None:
            # Drift pending-entry particles by dt * drift_vel. The
            # pending_entry bit is the authoritative gate; particles
            # with it clear are unaffected. The kernel's internal search
            # will assign a real element to any drifted particle that
            # has now crossed into the mesh.
            positions_gpu = apply_inlet_drift(
                positions_gpu, pending_entry_gpu, DT, drift_vel_gpu,
            )
            # Snapshot pre-kernel positions so we can restore them for
            # any pending-entry particle that the kernel's boundary
            # projection would otherwise snap to the mesh bbox.
            positions_pre_kernel = positions_gpu
        # Snapshot pre-step position for ballistic/freeze. For ballistic
        # this is the launch point; for freeze this is the value we
        # restore. Reference assignment — free when unused.
        if latch_and_advance_escape is not None:
            positions_pre_step = positions_gpu
        positions_gpu, element_ids_gpu = rk4_step(
            positions_gpu, element_ids_gpu, DT, velocity_sequence_gpu, step - 1
        )
        if suppress_pending_recovery is not None:
            positions_gpu, element_ids_gpu, drift_vel_gpu, pending_entry_gpu = \
                suppress_pending_recovery(
                    positions_gpu, element_ids_gpu,
                    positions_pre_kernel,
                    pending_entry_gpu, drift_vel_gpu,
                )
        if latch_and_advance_escape is not None:
            (positions_gpu, element_ids_gpu,
             escape_mode_gpu, escape_vel_gpu, last_vel_gpu) = \
                latch_and_advance_escape(
                    positions_gpu, positions_pre_step, element_ids_gpu,
                    escape_mode_gpu, escape_vel_gpu, last_vel_gpu, DT,
                )
        # Update per-step trajectory flags on GPU (cheap, fully fused).
        if args.export_escaped_flag:
            escaped_ever_gpu = _update_escaped(
                escaped_ever_gpu, element_ids_gpu, pending_entry_gpu,
            )
        if needs_temperature:
            _run_temperature_step(positions_gpu, element_ids_gpu, step - 1)

        do_log = (step % LOG_INTERVAL == 0) or (step == N_STEPS)
        do_export = ((exporter is not None)
                     and ((step % EXPORT_FREQUENCY == 0) or (step == N_STEPS)))
        do_density = (density_runner is not None
                      and ((step % density_export_freq == 0) or (step == N_STEPS)))
        if do_log or do_export or do_density:
            pos_cpu = np.array(positions_gpu, dtype=config.FLOAT_DTYPE_NP)
            eid_cpu_raw = np.array(element_ids_gpu, dtype=np.int32)
            if do_log:
                n_active = int(np.sum(eid_cpu_raw >= 0))
                n_lost = n_particles - n_active
                new_lost = n_lost - prev_lost
                stats_csv.write(f"{step},{n_active},{n_lost},{new_lost}\n")
                elapsed = time.time() - t_start
                sps = step / elapsed if elapsed > 0 else 0
                eta = (N_STEPS - step) / sps if sps > 0 else 0
                print(f"  Step {step:5d}/{N_STEPS}: active={n_active:,} lost={n_lost:,} "
                      f"(+{new_lost})  [{elapsed:.0f}s, {sps:.1f} step/s, ETA {eta:.0f}s]")
                prev_lost = n_lost
                if new_lost > 0 and exporter is not None:
                    do_export = True
            # Density evaluation. Use only active particles (element_id >= 0)
            # so escaped / pending particles don't contribute. Failures are
            # logged but do not affect tracking throughput.
            rho_part_np = None
            if do_density:
                try:
                    active_mask = eid_cpu_raw >= 0
                    if active_mask.any():
                        active_pos = positions_gpu[active_mask]
                    else:
                        active_pos = positions_gpu[:0]
                    _, rho_part_active = density_runner.step(
                        active_pos, dt=float(DT),
                        time_value=float(step * DT),
                        step_index=int(step),
                    )
                    if rho_part_active is not None:
                        rho_part_np = np.zeros((n_particles,), dtype=np.float32)
                        rho_part_np[active_mask] = rho_part_active
                except Exception as e:
                    print(f"  [!] Density step failed at step {step}: {e} -- disabling density.")
                    try:
                        density_runner.close()
                    except Exception:
                        pass
                    density_runner = None
            if do_export and exporter is not None:
                eid_export = eid_cpu_raw if EXPORT_ELEMENT_IDS else None
                step_extras = dict(extra_scalars) if extra_scalars else {}
                if args.export_escaped_flag:
                    step_extras['Escaped'] = np.asarray(
                        escaped_ever_gpu, dtype=np.uint8,
                    )
                _collect_temperature_extras(step_extras)
                if rho_part_np is not None and not args.density_no_particle_density:
                    step_extras['Density'] = rho_part_np
                exporter.enqueue_export(step, pos_cpu, particle_ids=particle_ids,
                                        element_ids=eid_export,
                                        extra_scalars=step_extras or extra_scalars)

    t_elapsed = time.time() - t_start
    stage_times['7_tracking'] = t_elapsed
    stats_csv.close()
    if exporter is not None:
        exporter.stop()
        if args.export_format == "vtkhdf":
            print(f"  Exported {exporter.n_exported} steps to "
                  f"{exporter.output_path}")
        else:
            print(f"  Exported {exporter.n_exported} VTU files")
    if density_runner is not None:
        try:
            density_runner.close()
            print(f"  Density: wrote {density_runner.writer.n_written if density_runner.writer else 0} "
                  f"step files + time-average to {density_runner.cfg.output_dir}")
        except Exception as e:
            print(f"  [!] Density runner close failed: {e}")
    print(f"  Stage 7 time: {t_elapsed:.1f}s")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*80}")
    print(f"TRACKING SUMMARY")
    print(f"{'='*80}")
    n_active_final = int(jnp.sum(element_ids_gpu >= 0))
    n_lost_final = n_particles - n_active_final
    print(f"  Particles: {n_particles:,}")
    print(f"  Final active: {n_active_final:,} ({100*n_active_final/n_particles:.2f}%)")
    print(f"  Final lost:   {n_lost_final:,} ({100*n_lost_final/n_particles:.2f}%)")
    print(f"  Steps:        {N_STEPS}")
    print(f"  Wall time:    {t_elapsed:.1f}s ({n_particles*N_STEPS/t_elapsed:,.0f} p·step/s)")
    print(f"  Output:       {output_subdir}")

    jt_positions_final = np.array(positions_gpu, dtype=config.FLOAT_DTYPE_NP)
    jt_elem_ids_final = np.array(element_ids_gpu, dtype=np.int32)

    # ==================================================================
    # Optional: FEMUSS comparison
    # ==================================================================
    if args.femuss_compare and femuss_start_data is not None:
        femuss_end_step = args.femuss_start + N_STEPS
        femuss_dir = args._femuss_dir
        end_file = femuss_dir / args.femuss_pattern.format(timestep=femuss_end_step)
        if end_file.exists():
            print(f"\n  Loading FEMUSS end-state: {end_file}")
            femuss_end = load_femuss_particles(end_file)
            if not np.allclose(femuss_start_data['initial_positions'],
                               femuss_end['initial_positions']):
                print("  WARNING: FEMUSS start/end Lagrangian positions differ; "
                      "skipping comparison.")
            else:
                comparison = compare_with_femuss(
                    jt_positions_final, jt_elem_ids_final,
                    femuss_end, particle_ids, verbose=True,
                )
                np.savez(
                    output_subdir / "comparison_data.npz",
                    jaxtrace_positions=jt_positions_final,
                    jaxtrace_elem_ids=jt_elem_ids_final,
                    femuss_current_positions=femuss_end['current_positions'][particle_ids],
                    femuss_has_left=femuss_end['has_left_domain'][particle_ids],
                    particle_ids=particle_ids,
                    error_magnitudes=comparison['error_mag'],
                    config={'femuss_start_step': args.femuss_start,
                            'femuss_end_step': femuss_end_step,
                            'n_steps': N_STEPS, 'dt': DT},
                )
                print(f"  Comparison saved to {output_subdir / 'comparison_data.npz'}")
        else:
            print(f"\n  FEMUSS comparison requested but reference file not found:")
            print(f"    {end_file}")
            print(f"  Skipping comparison.")

    # Always save a lightweight final-state npz (plus optional trajectory flags)
    final_state = {
        'positions': jt_positions_final,
        'element_ids': jt_elem_ids_final,
        'particle_ids': particle_ids,
    }
    if args.export_escaped_flag and escaped_ever_gpu is not None:
        final_state['escaped_ever'] = np.asarray(escaped_ever_gpu, dtype=np.uint8)
    if args.export_temperature and temperature_gpu is not None:
        final_state['temperature'] = np.asarray(temperature_gpu, dtype=np.float32)
    if args.track_max_temperature and max_temperature_gpu is not None:
        final_state['max_temperature'] = np.asarray(max_temperature_gpu, dtype=np.float32)
    np.savez(output_subdir / "final_state.npz", **final_state)

    # ==================================================================
    # Timing
    # ==================================================================
    t_total = time.time() - t_total_start
    print(f"\n{'='*80}")
    print(f"TIMING SUMMARY")
    print(f"{'='*80}")
    for name, t in stage_times.items():
        pct = 100 * t / t_total if t_total > 0 else 0
        print(f"  {name:<25s} {t:8.1f}s  ({pct:5.1f}%)")
    print(f"  {'-'*45}")
    print(f"  {'TOTAL':<25s} {t_total:8.1f}s")
    print(f"{'='*80}\n")
    print("Done.")


if __name__ == '__main__':
    main()
