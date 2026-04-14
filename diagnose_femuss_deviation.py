#!/usr/bin/env python3
"""
FEMUSS–JAXTrace Trajectory Deviation Diagnostic

Loads a spatial subset of FEMUSS particles (near tool region), tracks them
with JAXTrace, and compares at user-specified frequencies to identify when,
where, and how trajectories diverge.

Outputs:
  - deviation_timeline.csv: per-comparison-step summary statistics
  - particle_history.npz: full per-particle position/error history
  - worst_particles.csv: top worst-error particles with mechanism classification
  - deviation_step_XXXX.vtu + deviation_trajectory.pvd: Paraview time series
    with both JAXTrace and FEMUSS particles labeled for trajectory visualization

Usage:
    python diagnose_femuss_deviation.py \
      --input /path/to/FLA/post --n-steps 50 --compare-freq 5 \
      --y-range -0.01 0.01 --z-range -0.005 0.0 \
      --boundary-proj --pin-velocity --point-in-tet-tol 1e-6
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ.setdefault('JAX_PLATFORMS', 'cuda,rocm,cpu')

import sys
import time
import csv
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Parse --precision BEFORE importing config (which sets jax_enable_x64 at import time)
_precision_arg = 'float32'  # default
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

from benchmark_femuss_comparison import (
    load_femuss_particles,
    reconstruct_pin_velocity,
    create_rk4_comparison,
    write_vtu_simple,
)

import vtk
from vtk.util.numpy_support import vtk_to_numpy

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
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
from jaxtrace.gpu.search.aa_detection import precompute_aa_metadata, precompute_element_vertices
from jaxtrace.gpu.search.point_in_tet_methods import set_corrected_metadata, set_inverse_matrices_gpu
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="FEMUSS–JAXTrace Trajectory Deviation Diagnostic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Precision ---
    parser.add_argument(
        "--precision", type=str, default="float32", choices=["float32", "float64"],
        help="Floating-point precision (parsed early from sys.argv before config import).",
    )
    # --- I/O ---
    parser.add_argument(
        "--input", type=Path, default=Path("data/FLA/post"),
        help="Base input directory containing 0eule/ (mesh) and 1part/ (FEMUSS particles)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("deviation_output"),
        help="Output directory for all artifacts",
    )
    parser.add_argument(
        "--mesh-pattern", type=str, default="cylA_{timestep}.pvtu",
        help="Mesh PVTU file pattern with {timestep} placeholder",
    )
    parser.add_argument(
        "--femuss-pattern", type=str, default="cylA_pt_{timestep}.pvtu",
        help="FEMUSS particle PVTU file pattern with {timestep} placeholder",
    )
    # --- Simulation parameters ---
    parser.add_argument(
        "--vel-range", type=int, nargs=2, default=[159, 159], metavar=("START", "END"),
        help="Velocity timestep range (inclusive) for cyclic loading",
    )
    parser.add_argument(
        "--n-steps", type=int, default=100,
        help="Number of RK4 tracking steps",
    )
    parser.add_argument(
        "--dt", type=float, default=0.0025,
        help="Timestep size for RK4 integration",
    )
    parser.add_argument(
        "--femuss-start", type=int, default=0,
        help="FEMUSS start step (load particles from this timestep)",
    )
    # --- Diagnostic parameters ---
    parser.add_argument(
        "--compare-freq", type=int, default=10,
        help="Compare with FEMUSS every N steps",
    )
    parser.add_argument(
        "--y-range", type=float, nargs=2, default=[-0.01, 0.01], metavar=("Y0", "Y1"),
        help="Spatial filter Y range",
    )
    parser.add_argument(
        "--z-range", type=float, nargs=2, default=[-0.005, 0.0], metavar=("Z0", "Z1"),
        help="Spatial filter Z range",
    )
    parser.add_argument(
        "--x-range", type=float, nargs=2, default=None, metavar=("X0", "X1"),
        help="Spatial filter X range (default: all)",
    )
    parser.add_argument(
        "--error-threshold", type=float, default=1e-5,
        help="Flag particles exceeding this position error",
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Top N worst particles per comparison",
    )
    parser.add_argument(
        "--export-vtu", action="store_true", default=True,
        help="Export VTU files for deviated particles",
    )
    parser.add_argument(
        "--no-export-vtu", action="store_true", default=False,
        help="Disable VTU export",
    )
    # --- Tolerances (same as benchmark) ---
    parser.add_argument("--point-in-tet-tol", type=float, default=1e-6)
    parser.add_argument("--interpolation-det-min", type=float, default=None)
    parser.add_argument(
        "--interpolation-method", type=str, default="direct_inverse",
        choices=["direct_inverse", "gram_matrix"],
        help="Velocity interpolation method. "
             "'direct_inverse' uses precomputed M_inv (FEMUSS-equivalent). "
             "'gram_matrix' uses Gram matrix normal equations (legacy).",
    )
    parser.add_argument("--boundary-proj-tol", type=float, default=1e-6)
    # --- RK4 substep policies ---
    parser.add_argument("--bbox-clamp", action="store_true", default=False)
    parser.add_argument("--no-bbox-clamp", action="store_true")
    parser.add_argument(
        "--failed-substage", type=str, default="zero_vel",
        choices=["zero_vel", "last_valid_vel", "skip_step"],
    )
    # --- Boundary projection ---
    parser.add_argument("--boundary-proj", action="store_true", default=True)
    parser.add_argument("--no-boundary-proj", action="store_true")
    # --- Level-set ---
    parser.add_argument("--no-levelset", action="store_true")
    parser.add_argument(
        "--levelset-mode", type=str, default="zero_vel",
        choices=["zero_vel", "skip_step"],
    )
    # --- Level-set bands ---
    parser.add_argument("--l0-skip-band", type=float, default=0.0)
    parser.add_argument("--no-l0-skip-boundary", action="store_true", default=False)
    parser.add_argument("--enhanced-search-band", type=float, default=0.0)
    # --- L1/L2 ---
    parser.add_argument("--l1-method", type=str, default="face", choices=["face", "node"])
    parser.add_argument("--l2-neighborhood", type=int, default=3, choices=[3, 5])
    # --- Registration / RK4 mode ---
    parser.add_argument(
        "--registration", type=str, default=None,
        choices=["vertex_multi", "parent_cube"],
        help="Override octree registration method (default: use config value)",
    )
    parser.add_argument("--rk4-mode", type=str, default="fused", choices=["fused", "split"])
    parser.add_argument("--l2-vectorized", action="store_true", default=False)
    # --- Pin velocity ---
    parser.add_argument("--pin-velocity", action="store_true", default=True)
    parser.add_argument("--no-pin-velocity", action="store_true", default=False)
    parser.add_argument("--pin-rpm", type=float, default=-600.0)
    parser.add_argument("--pin-center", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--pin-axis", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    parser.add_argument("--pin-tilt", type=float, default=0.0)
    return parser.parse_args()


# =============================================================================
# VTU/PVD EXPORT FOR DEVIATION VISUALIZATION
# =============================================================================

def write_deviation_vtu(filepath, jt_positions, femuss_positions, particle_ids,
                        error_magnitudes, status_array):
    """
    Write a VTU with both JAXTrace and FEMUSS particles concatenated.

    Points: [JT_0, JT_1, ..., JT_N, FEM_0, FEM_1, ..., FEM_N]
    Scalar fields:
      - source: 0=JAXTrace, 1=FEMUSS
      - particle_id: FEMUSS global ID (same for both copies)
      - error_magnitude: L2 error (same value on both copies)
      - status: 0=both active, 1=lost by JT, 2=lost by FEMUSS
    """
    n = len(particle_ids)
    positions = np.vstack([jt_positions, femuss_positions])  # (2N, 3)
    pids = np.concatenate([particle_ids, particle_ids])
    errors = np.concatenate([error_magnitudes, error_magnitudes])
    source = np.concatenate([np.zeros(n, dtype=np.int32), np.ones(n, dtype=np.int32)])
    statuses = np.concatenate([status_array, status_array])

    write_vtu_simple(
        str(filepath),
        positions=positions,
        particle_ids=pids.astype(np.int32),
        extra_scalars={
            'source': source,
            'error_magnitude': errors.astype(np.float32),
            'status': statuses.astype(np.int32),
        },
    )


def write_pvd_file(pvd_path, vtu_entries):
    """Write a PVD collection file for Paraview time-series loading.

    vtu_entries: list of (timestep_float, vtu_filename_relative)
    """
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="1.0" byte_order="LittleEndian">',
        '  <Collection>',
    ]
    for t, fname in vtu_entries:
        lines.append(f'    <DataSet timestep="{t}" file="{fname}"/>')
    lines.extend([
        '  </Collection>',
        '</VTKFile>',
    ])
    with open(pvd_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# =============================================================================
# MECHANISM CLASSIFICATION
# =============================================================================

def classify_mechanisms(error_history, jt_active_history, femuss_left_history,
                        threshold, comparison_steps):
    """
    Classify deviation mechanism for each particle.

    Returns: (N_particles,) array of strings
    """
    n_comparisons, n_particles = error_history.shape
    mechanisms = np.full(n_particles, 'ok', dtype='U20')

    for i in range(n_particles):
        # Find first deviation step
        ever_jt_lost = np.any(~jt_active_history[:, i])
        ever_femuss_left = np.any(femuss_left_history[:, i])

        if ever_jt_lost and not ever_femuss_left:
            mechanisms[i] = 'lost_by_jt'
            continue
        if ever_femuss_left and not ever_jt_lost:
            mechanisms[i] = 'lost_by_femuss'
            continue
        if ever_jt_lost and ever_femuss_left:
            mechanisms[i] = 'both_lost'
            continue

        # Both stayed active — check error pattern
        above = error_history[:, i] > threshold
        if not np.any(above):
            mechanisms[i] = 'ok'
            continue

        first_above = np.argmax(above)
        if first_above == 0:
            mechanisms[i] = 'sudden_jump'
            continue

        prev_error = error_history[first_above - 1, i]
        jump_error = error_history[first_above, i]
        if prev_error < threshold * 0.1 and jump_error > threshold * 10:
            mechanisms[i] = 'sudden_jump'
        else:
            mechanisms[i] = 'gradual_drift'

    return mechanisms


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    export_vtu = args.export_vtu and not args.no_export_vtu

    # Paths
    MESH_BASE_PATH = args.input / "0eule"
    FEMUSS_PARTICLE_PATH = args.input / "1part"
    FEMUSS_FILE_PATTERN = args.femuss_pattern
    VELOCITY_FIELD_NAME = 'Displacement'
    LEVELSET_FIELD_NAME = 'LEVEL'
    POINT_IN_TET_METHOD = 'inverse'

    N_STEPS = args.n_steps
    DT = args.dt
    FEMUSS_START_STEP = args.femuss_start
    COMPARE_FREQ = args.compare_freq

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

    L1_METHOD = args.l1_method
    L2_NEIGHBORHOOD = args.l2_neighborhood
    L0_SKIP_BAND = args.l0_skip_band
    ENHANCED_SEARCH_BAND = args.enhanced_search_band

    if args.registration is not None:
        config.OCTREE_REGISTRATION_METHOD = args.registration

    use_pin_velocity = args.pin_velocity and not args.no_pin_velocity

    # Output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    t_total_start = time.time()

    print("=" * 80)
    print("FEMUSS–JAXTrace Trajectory Deviation Diagnostic")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"\nSpatial filter:")
    if args.x_range:
        print(f"  X: [{args.x_range[0]}, {args.x_range[1]}]")
    else:
        print(f"  X: all")
    print(f"  Y: [{args.y_range[0]}, {args.y_range[1]}]")
    print(f"  Z: [{args.z_range[0]}, {args.z_range[1]}]")
    print(f"Compare freq: every {COMPARE_FREQ} steps")
    print(f"Error threshold: {args.error_threshold:.0e}")
    print(f"N steps: {N_STEPS}")
    print(f"VTU export: {'ON' if export_vtu else 'OFF'}")
    print("=" * 80)

    # ==================================================================
    # [1/7] Load mesh
    # ==================================================================
    t_stage = time.time()
    print(f"\n[1/7] Loading mesh...")
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=args.mesh_pattern,
        timestep_range=tuple(args.vel_range),
        field_name=VELOCITY_FIELD_NAME,
        verbose=False
    )

    print("  Deduplicating...")
    node_positions, connectivity, n_dup, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    connectivity = connectivity.astype(np.int32)
    print(f"  Elements: {connectivity.shape[0]:,}, Nodes: {node_positions.shape[0]:,}")

    mesh_bbox_min_cpu = node_positions.min(axis=0).astype(config.FLOAT_DTYPE_NP)
    mesh_bbox_max_cpu = node_positions.max(axis=0).astype(config.FLOAT_DTYPE_NP)

    # Load level-set
    levelset_cpu = None
    if config.RK4_LEVELSET_MASK:
        start_ts, end_ts = args.vel_range
        ls_raw = None
        for ts in range(start_ts, end_ts + 1):
            ls_file = MESH_BASE_PATH / args.mesh_pattern.format(timestep=ts)
            reader = vtk.vtkXMLPUnstructuredGridReader()
            reader.SetFileName(str(ls_file))
            reader.Update()
            pd = reader.GetOutput().GetPointData()
            if pd.HasArray(LEVELSET_FIELD_NAME):
                ls_raw = vtk_to_numpy(pd.GetArray(LEVELSET_FIELD_NAME)).astype(np.float64)
                break
        if ls_raw is None:
            print(f"  WARNING: Level-set field not found, disabling mask")
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
            n_neg = np.sum(levelset_cpu < 0)
            print(f"  Level-set: {n_neg:,}/{n_dedup:,} inside tool")

    # Pin velocity reconstruction
    if use_pin_velocity and levelset_cpu is not None:
        print(f"  Reconstructing pin velocity...")
        velocity_sequence, n_overwritten = reconstruct_pin_velocity(
            node_positions=node_positions,
            velocity_sequence=velocity_sequence,
            levelset=levelset_cpu,
            rpm=args.pin_rpm,
            center=np.array(args.pin_center),
            axis=np.array(args.pin_axis),
            tilt_deg=args.pin_tilt,
        )
        print(f"  Pin velocity applied to {n_overwritten:,} nodes")

    print(f"  Stage 1: {time.time() - t_stage:.1f}s")

    # ==================================================================
    # [2/7] Precompute metadata
    # ==================================================================
    t_stage = time.time()
    print(f"\n[2/7] Precomputing metadata...")
    config.POINT_IN_TET_METHOD = POINT_IN_TET_METHOD

    aa_metadata = precompute_aa_metadata(connectivity, node_positions, verbose=False)
    element_vertices = precompute_element_vertices(connectivity, node_positions, verbose=False)
    M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)

    v0 = node_positions[connectivity[:, 0]]
    v1 = node_positions[connectivity[:, 1]]
    v2 = node_positions[connectivity[:, 2]]
    v3 = node_positions[connectivity[:, 3]]
    e1 = v1 - v0; e2 = v2 - v0; e3 = v3 - v0
    cross_e2_e3 = np.cross(e2, e3)
    det_values = np.sum(e1 * cross_e2_e3, axis=1)
    element_volumes = np.abs(det_values) / 6.0
    print(f"  Stage 2: {time.time() - t_stage:.1f}s")

    # ==================================================================
    # [3/7] Build structures and upload to GPU
    # ==================================================================
    t_stage = time.time()
    print(f"\n[3/7] Building structures and uploading to GPU...")

    octree_struct = build_global_morton_octree(
        node_positions=node_positions,
        connectivity=connectivity,
        leaf_capacity=256, max_depth=21, verbose=False
    )
    if config.OCTREE_REGISTRATION_METHOD == "parent_cube":
        mesh_octree_cells_multi = extract_octree_cells_parent_cube(
            node_positions, connectivity, tolerance=1e-6, verbose=True
        )
        print(f"  Parent-cube octree: {mesh_octree_cells_multi.n_cells:,} cells, "
              f"{mesh_octree_cells_multi.elements_per_cell_mean:.1f} elem/cell")
    else:
        mesh_octree_cells_multi = extract_octree_cells_vertex_multi(
            node_positions, connectivity, tolerance=1e-6, verbose=False
        )
        print(f"  Vertex-multi octree: {mesh_octree_cells_multi.n_cells:,} cells")

    element_neighbors_face = build_element_neighbors_array(connectivity, method='face', verbose=False)
    use_enhanced_band = (ENHANCED_SEARCH_BAND > 0 and levelset_cpu is not None)
    need_node_neighbors = (L1_METHOD == 'node') or use_enhanced_band
    element_neighbors_node = None
    if need_node_neighbors:
        element_neighbors_node = build_element_neighbors_array(connectivity, method='node', verbose=False)

    if L1_METHOD == 'node' and element_neighbors_node is not None:
        element_neighbors = element_neighbors_node
    else:
        element_neighbors = element_neighbors_face
    n_neighbors_per_element = element_neighbors.shape[1]

    mesh_gpu = upload_mesh_to_gpu(connectivity, node_positions, element_neighbors, verbose=False)
    morton_gpu = upload_global_morton_to_gpu(octree_struct, connectivity, node_positions)
    mesh_aligned_octree_multi_gpu = upload_mesh_aligned_octree_to_gpu(
        connectivity, node_positions, mesh_octree_cells_multi, verbose=False
    )

    from jaxtrace.gpu.search.aa_detection import AxisAlignedMetadata
    aa_metadata_gpu = AxisAlignedMetadata(
        base_vertex_indices=jax.device_put(aa_metadata.base_vertex_indices),
        base_vertices=jax.device_put(aa_metadata.base_vertices),
        inv_edge_lengths=jax.device_put(aa_metadata.inv_edge_lengths),
        axis_indices=jax.device_put(aa_metadata.axis_indices),
        is_axis_aligned=jax.device_put(aa_metadata.is_axis_aligned)
    )
    element_vertices_gpu = jax.device_put(element_vertices)
    M_inv_gpu = jax.device_put(M_inv_array)
    p0_gpu = jax.device_put(p0_array)
    element_volumes_gpu = jax.device_put(element_volumes.astype(config.FLOAT_DTYPE_NP))

    set_corrected_metadata(aa_metadata_gpu, element_vertices_gpu)
    set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)

    velocity_sequence_gpu = jax.device_put(velocity_sequence)
    mesh_bbox_min_gpu = jax.device_put(mesh_bbox_min_cpu)
    mesh_bbox_max_gpu = jax.device_put(mesh_bbox_max_cpu)
    levelset_gpu = jax.device_put(levelset_cpu) if levelset_cpu is not None else None

    # Per-wall clamp masks
    wall_config = getattr(config, 'RK4_BOUNDARY_WALLS', None)
    if wall_config is not None and not isinstance(wall_config, dict):
        wall_config = None
    if wall_config is None:
        clamp_min_mask = jnp.array([True, True, True])
        clamp_max_mask = jnp.array([True, True, True])
    else:
        clamp_min_mask = jnp.array([
            wall_config.get('x_min', 'clamp') == 'clamp',
            wall_config.get('y_min', 'clamp') == 'clamp',
            wall_config.get('z_min', 'clamp') == 'clamp',
        ])
        clamp_max_mask = jnp.array([
            wall_config.get('x_max', 'clamp') == 'clamp',
            wall_config.get('y_max', 'clamp') == 'clamp',
            wall_config.get('z_max', 'clamp') == 'clamp',
        ])

    # Boundary/enhanced elements
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

    print(f"  Stage 3: {time.time() - t_stage:.1f}s")

    # ==================================================================
    # [4/7] Load FEMUSS particles + spatial filter
    # ==================================================================
    t_stage = time.time()
    print(f"\n[4/7] Loading FEMUSS particles with spatial filter...")

    start_file = FEMUSS_PARTICLE_PATH / FEMUSS_FILE_PATTERN.format(timestep=FEMUSS_START_STEP)
    femuss_start = load_femuss_particles(start_file)
    print(f"  Total particles: {femuss_start['n_particles']:,}")
    print(f"  Left domain at start: {femuss_start['has_left_domain'].sum():,}")

    # Filter: active AND in spatial region
    active_mask = ~femuss_start['has_left_domain']
    pos = femuss_start['current_positions']

    spatial_mask = active_mask.copy()
    spatial_mask &= (pos[:, 1] >= args.y_range[0]) & (pos[:, 1] <= args.y_range[1])
    spatial_mask &= (pos[:, 2] >= args.z_range[0]) & (pos[:, 2] <= args.z_range[1])
    if args.x_range is not None:
        spatial_mask &= (pos[:, 0] >= args.x_range[0]) & (pos[:, 0] <= args.x_range[1])

    particle_ids = np.where(spatial_mask)[0].astype(np.int32)
    n_filtered = len(particle_ids)
    print(f"  Filtered: {n_filtered:,} particles in spatial region")

    particle_positions = femuss_start['current_positions'][particle_ids].astype(config.FLOAT_DTYPE_NP)

    print(f"  Stage 4: {time.time() - t_stage:.1f}s")

    # ==================================================================
    # [5/7] Initial assignment
    # ==================================================================
    t_stage = time.time()
    print(f"\n[5/7] Initial assignment...")
    positions_gpu = jax.device_put(particle_positions)

    element_ids_initial = initial_assignment_mesh_aligned_multi_local(
        positions_gpu, mesh_aligned_octree_multi_gpu,
        batch_size=50000, max_tests=600, verbose=True
    )
    element_ids_initial = jax.block_until_ready(element_ids_initial)
    n_assigned = int(jnp.sum(element_ids_initial >= 0))
    print(f"  Assigned: {n_assigned:,}/{n_filtered:,}")

    print(f"  Stage 5: {time.time() - t_stage:.1f}s")

    # ==================================================================
    # [6/7] Build RK4 + compile
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
        enable_l0=True,
        enable_l1=True,
        n_hops=5,
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
        interpolation_method=getattr(args, 'interpolation_method', 'direct_inverse'),
        M_inv_gpu=M_inv_gpu,
        p0_gpu=p0_gpu,
        rk4_mode=args.rk4_mode,
        use_l2_vectorized=args.l2_vectorized,
    )

    # Warmup/compile
    print(f"  Compiling...")
    t_compile = time.time()
    positions_gpu, element_ids_gpu = rk4_step(
        positions_gpu, element_ids_initial, DT, velocity_sequence_gpu, 0
    )
    jax.block_until_ready(positions_gpu)
    print(f"  Compilation: {time.time() - t_compile:.1f}s")

    # Re-initialize for actual run
    positions_gpu = jax.device_put(particle_positions)
    element_ids_gpu = element_ids_initial

    print(f"  Stage 6: {time.time() - t_stage:.1f}s")

    # ==================================================================
    # [7/7] Tracking loop with periodic comparison
    # ==================================================================
    print(f"\n[7/7] Running {N_STEPS} steps, comparing every {COMPARE_FREQ}...")
    print("=" * 80)

    # Determine comparison steps
    comparison_steps = list(range(COMPARE_FREQ, N_STEPS + 1, COMPARE_FREQ))
    if N_STEPS not in comparison_steps:
        comparison_steps.append(N_STEPS)
    n_comparisons = len(comparison_steps)

    # Allocate history arrays
    jt_positions_history = np.zeros((n_comparisons, n_filtered, 3), dtype=np.float64)
    femuss_positions_history = np.zeros((n_comparisons, n_filtered, 3), dtype=np.float64)
    error_history = np.zeros((n_comparisons, n_filtered), dtype=np.float64)
    jt_active_history = np.zeros((n_comparisons, n_filtered), dtype=bool)
    femuss_left_history = np.zeros((n_comparisons, n_filtered), dtype=bool)

    first_deviation_step = np.full(n_filtered, -1, dtype=np.int32)

    # CSV timeline
    timeline_path = output_dir / "deviation_timeline.csv"
    timeline_csv = open(timeline_path, 'w', newline='')
    timeline_writer = csv.writer(timeline_csv)
    timeline_writer.writerow([
        'step', 'femuss_step', 'n_both_active', 'n_jt_lost', 'n_femuss_left',
        'error_mean', 'error_median', 'error_max', 'error_p95', 'error_p99',
        'n_above_threshold', 'n_newly_deviated', 'n_newly_lost_jt', 'n_newly_lost_femuss',
    ])

    # VTU tracking
    vtu_entries = []
    prev_jt_lost_mask = np.zeros(n_filtered, dtype=bool)
    prev_femuss_left_mask = np.zeros(n_filtered, dtype=bool)

    t_start = time.time()
    comp_idx = 0

    for step in range(1, N_STEPS + 1):
        positions_gpu, element_ids_gpu = rk4_step(
            positions_gpu, element_ids_gpu, DT, velocity_sequence_gpu, step - 1
        )

        if step not in comparison_steps:
            continue

        # --- Comparison point ---
        femuss_step = FEMUSS_START_STEP + step
        femuss_file = FEMUSS_PARTICLE_PATH / FEMUSS_FILE_PATTERN.format(timestep=femuss_step)

        if not femuss_file.exists():
            print(f"  Step {step}: FEMUSS file not found ({femuss_file.name}), skipping")
            comp_idx += 1
            continue

        femuss_data = load_femuss_particles(str(femuss_file))

        # Extract positions
        jt_pos = np.array(positions_gpu, dtype=np.float64)
        jt_eids = np.array(element_ids_gpu, dtype=np.int32)
        femuss_pos = femuss_data['current_positions'][particle_ids]
        femuss_left = femuss_data['has_left_domain'][particle_ids]
        jt_active = jt_eids >= 0
        jt_lost = ~jt_active

        # Store history
        jt_positions_history[comp_idx] = jt_pos
        femuss_positions_history[comp_idx] = femuss_pos
        jt_active_history[comp_idx] = jt_active
        femuss_left_history[comp_idx] = femuss_left

        # Compute errors (for both-active particles; NaN otherwise)
        both_active = jt_active & (~femuss_left)
        errors = np.full(n_filtered, np.nan)
        if np.any(both_active):
            diff = jt_pos[both_active] - femuss_pos[both_active]
            errors[both_active] = np.linalg.norm(diff, axis=1)
        error_history[comp_idx] = errors

        # Track first deviation step
        above_threshold = both_active & (errors > args.error_threshold)
        newly_deviated = above_threshold & (first_deviation_step < 0)
        first_deviation_step[newly_deviated] = step
        n_newly_deviated = int(np.sum(newly_deviated))

        # Status transitions
        newly_lost_jt = jt_lost & (~prev_jt_lost_mask)
        newly_lost_femuss = femuss_left & (~prev_femuss_left_mask)
        n_newly_lost_jt = int(np.sum(newly_lost_jt))
        n_newly_lost_femuss = int(np.sum(newly_lost_femuss))

        prev_jt_lost_mask = jt_lost.copy()
        prev_femuss_left_mask = femuss_left.copy()

        # Statistics
        n_both_active = int(np.sum(both_active))
        n_jt_lost = int(np.sum(jt_lost))
        n_femuss_left_count = int(np.sum(femuss_left))

        valid_errors = errors[both_active] if n_both_active > 0 else np.array([0.0])
        n_above = int(np.sum(above_threshold))

        timeline_writer.writerow([
            step, femuss_step, n_both_active, n_jt_lost, n_femuss_left_count,
            f'{np.nanmean(valid_errors):.6e}', f'{np.nanmedian(valid_errors):.6e}',
            f'{np.nanmax(valid_errors):.6e}',
            f'{np.nanpercentile(valid_errors, 95):.6e}',
            f'{np.nanpercentile(valid_errors, 99):.6e}',
            n_above, n_newly_deviated, n_newly_lost_jt, n_newly_lost_femuss,
        ])

        elapsed = time.time() - t_start
        print(f"  Step {step:5d}/{N_STEPS} (FEMUSS={femuss_step}): "
              f"active={n_both_active}/{n_filtered}, "
              f"jt_lost={n_jt_lost}, fem_left={n_femuss_left_count}")
        print(f"    Error: mean={np.nanmean(valid_errors):.2e} "
              f"max={np.nanmax(valid_errors):.2e} "
              f"P95={np.nanpercentile(valid_errors, 95):.2e} "
              f">thr={n_above}")
        print(f"    New deviations: {n_newly_deviated}  "
              f"Lost(JT): {n_newly_lost_jt}  Lost(FEM): {n_newly_lost_femuss}  "
              f"[{elapsed:.0f}s elapsed]")

        # VTU export for deviated particles
        if export_vtu:
            # Select particles to export: above threshold OR lost by either
            export_mask = above_threshold | jt_lost | femuss_left
            if np.any(export_mask):
                exp_idx = np.where(export_mask)[0]
                exp_jt_pos = jt_pos[exp_idx]
                exp_femuss_pos = femuss_pos[exp_idx]
                exp_pids = particle_ids[exp_idx]
                exp_errors = np.where(np.isnan(errors[exp_idx]), -1.0, errors[exp_idx])

                # Status: 0=both active, 1=lost by JT, 2=lost by FEMUSS
                exp_status = np.zeros(len(exp_idx), dtype=np.int32)
                exp_status[jt_lost[exp_idx]] = 1
                exp_status[femuss_left[exp_idx]] = 2

                vtu_name = f"deviation_step_{femuss_step:04d}.vtu"
                vtu_path = output_dir / vtu_name
                write_deviation_vtu(
                    vtu_path, exp_jt_pos, exp_femuss_pos, exp_pids,
                    exp_errors, exp_status
                )
                vtu_entries.append((float(femuss_step), vtu_name))

        comp_idx += 1

    timeline_csv.close()
    t_tracking = time.time() - t_start
    print(f"\n  Tracking: {t_tracking:.1f}s")

    # ==================================================================
    # Post-processing: mechanism classification
    # ==================================================================
    print(f"\nClassifying deviation mechanisms...")

    # Replace NaN with 0 for classification
    error_for_class = np.nan_to_num(error_history, nan=0.0)
    mechanisms = classify_mechanisms(
        error_for_class, jt_active_history, femuss_left_history,
        args.error_threshold, comparison_steps
    )

    # Count mechanisms
    unique, counts = np.unique(mechanisms, return_counts=True)
    print(f"  Mechanism breakdown:")
    for m, c in zip(unique, counts):
        print(f"    {m:<20s}: {c:,}")

    # ==================================================================
    # Save outputs
    # ==================================================================
    print(f"\nSaving outputs to {output_dir}/...")

    # particle_history.npz
    np.savez(
        output_dir / "particle_history.npz",
        particle_ids=particle_ids,
        initial_positions=particle_positions,
        first_deviation_step=first_deviation_step,
        comparison_steps=np.array(comparison_steps[:comp_idx]),
        jt_positions=jt_positions_history[:comp_idx],
        femuss_positions=femuss_positions_history[:comp_idx],
        error_magnitude=error_history[:comp_idx],
        jt_active=jt_active_history[:comp_idx],
        femuss_left=femuss_left_history[:comp_idx],
        mechanisms=mechanisms,
    )
    print(f"  particle_history.npz")

    # worst_particles.csv
    worst_path = output_dir / "worst_particles.csv"
    with open(worst_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'particle_id', 'first_deviation_step', 'final_error', 'mechanism',
            'initial_x', 'initial_y', 'initial_z',
        ])

        # Get final error for ranking
        final_errors = error_for_class[-1] if comp_idx > 0 else np.zeros(n_filtered)
        worst_indices = np.argsort(final_errors)[-args.top_n:][::-1]

        for idx in worst_indices:
            if final_errors[idx] <= 0:
                continue
            writer.writerow([
                particle_ids[idx],
                first_deviation_step[idx],
                f'{final_errors[idx]:.6e}',
                mechanisms[idx],
                f'{particle_positions[idx, 0]:.8e}',
                f'{particle_positions[idx, 1]:.8e}',
                f'{particle_positions[idx, 2]:.8e}',
            ])
    print(f"  worst_particles.csv")

    # PVD file
    if export_vtu and vtu_entries:
        pvd_path = output_dir / "deviation_trajectory.pvd"
        write_pvd_file(pvd_path, vtu_entries)
        print(f"  deviation_trajectory.pvd ({len(vtu_entries)} timesteps)")

    print(f"  deviation_timeline.csv")

    # ==================================================================
    # Summary
    # ==================================================================
    t_total = time.time() - t_total_start
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"  Particles tracked: {n_filtered:,}")
    print(f"  Steps: {N_STEPS}")
    print(f"  Comparisons: {comp_idx}")
    n_ever_deviated = int(np.sum(first_deviation_step >= 0))
    print(f"  Ever deviated (>{args.error_threshold:.0e}): {n_ever_deviated:,}")
    print(f"  Total time: {t_total:.1f}s")
    print(f"  Output: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
