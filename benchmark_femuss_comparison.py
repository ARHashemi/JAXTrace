#!/usr/bin/env python3
"""
FEMUSS Particle Tracking Comparison

Loads particle positions from FEMUSS Sparticle output at a given timestep,
tracks them with JAXTrace RK4 for N steps, and compares final positions
against FEMUSS results at the target timestep.

FEMUSS particle data:
  - Positions are LAGRANGIAN (initial/reference) — identical across all timesteps
  - 'Displacements' are CUMULATIVE — current_pos = initial_pos + displacement
  - 'Has Once Left Domain' flag marks particles that left the mesh

Usage:
    python benchmark_femuss_comparison.py 2>&1 | tee logs/femuss_comparison.log
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'

import sys
import time
import csv
import queue
import threading
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import jaxtrace.config as config  # Must be imported before JAX array creation (sets jax_enable_x64)
import jax
import jax.numpy as jnp

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu
from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import extract_octree_cells_single
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import extract_octree_cells_vertex_multi
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu
from jaxtrace.gpu.tracking.initial_assignment_cascading import (
    initial_assignment_cascading_fallback,
    initial_assignment_mesh_aligned_multi_local,
)
from jaxtrace.gpu.search.aa_detection import precompute_aa_metadata, precompute_element_vertices
from jaxtrace.gpu.search.point_in_tet_methods import set_corrected_metadata, set_inverse_matrices_gpu
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
from jaxtrace.gpu.search.mesh_aligned_point_location import (
    search_mesh_aligned_octree_multi_local_where,
)
from jaxtrace.gpu.search.point_in_tet_methods import (
    point_in_tet_gpu as point_in_tet_dispatcher,
)

import vtk
from vtk.util.numpy_support import vtk_to_numpy

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# --- Mesh ---
MESH_BASE_PATH = Path("data/FLA/post/0eule")
MESH_FILE_PATTERN = "cylA_{timestep}.pvtu"
VELOCITY_FIELD_NAME = 'Displacement'

# --- Mesh velocity timestep range ---
# Range of mesh files to load. The 'Displacement' field is actual velocity (m/s),
# not cumulative displacement — used directly without differencing.
# FEMUSS recycles 40 velocity snapshots (1 revolution). Load the full range
# to match: e.g. (121, 160) for 40 snapshots, cycled via vel_idx % n_timesteps.
VELOCITY_TIMESTEP_RANGE = (121, 160)

# --- FEMUSS particle data ---
FEMUSS_PARTICLE_PATH = Path("data/FLA/post/1part")
FEMUSS_FILE_PATTERN = "cylA_pt_{timestep}.pvtu"

# --- Comparison timesteps ---
# Load particles from FEMUSS at START_STEP, track for N_STEPS,
# compare with FEMUSS at END_STEP.
# END_STEP = START_STEP + N_STEPS
FEMUSS_START_STEP = 0
N_STEPS = 2684                   # <-- small for quick testing, set to 50/250 for real runs
DT = 0.0025

# --- Search configuration ---
ENABLE_L0_SEARCH = True
ENABLE_L1_SEARCH = True
N_HOPS = 5

# --- L2 method ---
L2_METHOD = 'mesh_aligned_octree_multi_local_where'

# --- Point-in-tet ---
POINT_IN_TET_METHOD = 'inverse'

# --- RK4 sub-step recovery ---
RK4_SUBSTEP_BBOX_CLAMP = True       # Clamp sub-step positions to mesh bbox
RK4_SUBSTEP_LAST_VALID_VEL = True   # Fall back to last valid velocity if sub-step search fails
RK4_BOUNDARY_PROJECTION = True      # Clamp final pos to bbox (inset by tol) and re-search if lost

# --- Boundary projection tolerance (FEMUSS uses ±1e-6) ---
config.RK4_BOUNDARY_PROJECTION_TOL = 1e-6

# --- Per-wall boundary control ---
# Set to None for default (all walls clamped), or a dict:
#   'clamp'   = project back onto this wall (default)
#   'outlet'  = no treatment (particle exits freely)
#   'extended' = continue with last velocity outside mesh (not wired in this benchmark)
# Example: allow particles to exit through x_max:
#   config.RK4_BOUNDARY_WALLS = {'x_max': 'outlet'}
config.RK4_BOUNDARY_WALLS = 'clamp'

# --- Level-set velocity masking ---
# Zero out velocity where level-set < 0 (inside tool region).
# Matches FEMUSS: if (levelSetValue < 0) AddFluidInteraction = .false.
config.RK4_LEVELSET_MASK = True
LEVELSET_FIELD_NAME = 'LEVEL'

# --- Export ---
EXPORT_FREQUENCY = 1
LOG_INTERVAL = 10
OUTPUT_DIR = Path("output/femuss_comparison")

SEED = 42


# =============================================================================
# FEMUSS Particle Loading
# =============================================================================

def load_femuss_particles(filepath):
    """
    Load particle data from FEMUSS Sparticle PVTU output.

    Returns
    -------
    dict with keys:
        'initial_positions': (N, 3) float64 — Lagrangian reference positions
        'displacements': (N, 3) float64 — cumulative displacements
        'current_positions': (N, 3) float64 — initial + displacement
        'has_left_domain': (N,) bool — True if particle has once left domain
        'n_particles': int
    """
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(str(filepath))
    reader.Update()
    data = reader.GetOutput()

    n = data.GetNumberOfPoints()
    if n == 0:
        raise RuntimeError(f"No particles found in {filepath}")

    # Positions (Lagrangian reference — same across all timesteps)
    pts = vtk_to_numpy(data.GetPoints().GetData()).astype(np.float64)

    # Displacements (cumulative)
    disp_arr = data.GetPointData().GetArray('Displacements')
    if disp_arr is None:
        raise RuntimeError(f"No 'Displacements' field in {filepath}")
    disp = vtk_to_numpy(disp_arr).astype(np.float64)

    # Has Once Left Domain
    left_arr = data.GetPointData().GetArray('Has Once Left Domain')
    has_left = np.zeros(n, dtype=bool)
    if left_arr is not None:
        has_left = vtk_to_numpy(left_arr).astype(bool)

    return {
        'initial_positions': pts,
        'displacements': disp,
        'current_positions': pts + disp,
        'has_left_domain': has_left,
        'n_particles': n,
    }


# =============================================================================
# VTU Export (background thread, same pattern as benchmark_rk4_diagnostic.py)
# =============================================================================

def write_vtu_simple(filename, positions, particle_ids=None, element_ids=None,
                     extra_scalars=None):
    """Write VTU file (simple ASCII format, no VTK dependency)."""
    n_points = len(positions)

    xml_lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">',
        '  <UnstructuredGrid>',
        f'    <Piece NumberOfPoints="{n_points}" NumberOfCells="{n_points}">',
        '      <Points>',
        '        <DataArray type="Float32" NumberOfComponents="3" format="ascii">',
    ]
    for pos in positions:
        xml_lines.append(f'          {pos[0]:.6e} {pos[1]:.6e} {pos[2]:.6e}')
    xml_lines.extend([
        '        </DataArray>',
        '      </Points>',
        '      <Cells>',
        '        <DataArray type="Int32" Name="connectivity" format="ascii">',
        '          ' + ' '.join(str(i) for i in range(n_points)),
        '        </DataArray>',
        '        <DataArray type="Int32" Name="offsets" format="ascii">',
        '          ' + ' '.join(str(i + 1) for i in range(n_points)),
        '        </DataArray>',
        '        <DataArray type="UInt8" Name="types" format="ascii">',
        '          ' + ' '.join('1' for _ in range(n_points)),
        '        </DataArray>',
        '      </Cells>',
    ])

    has_pd = (particle_ids is not None or element_ids is not None or extra_scalars)
    if has_pd:
        xml_lines.append('      <PointData>')
        if particle_ids is not None:
            xml_lines.append('        <DataArray type="Int32" Name="ParticleID" format="ascii">')
            xml_lines.append('          ' + ' '.join(str(int(p)) for p in particle_ids))
            xml_lines.append('        </DataArray>')
        if element_ids is not None:
            xml_lines.append('        <DataArray type="Int32" Name="ElementID" format="ascii">')
            xml_lines.append('          ' + ' '.join(str(int(e)) for e in element_ids))
            xml_lines.append('        </DataArray>')
        if extra_scalars:
            for name, arr in extra_scalars.items():
                dtype_str = 'Int32' if arr.dtype in (np.int32, np.int64) else 'Float32'
                xml_lines.append(f'        <DataArray type="{dtype_str}" Name="{name}" format="ascii">')
                xml_lines.append('          ' + ' '.join(str(v) for v in arr))
                xml_lines.append('        </DataArray>')
        xml_lines.append('      </PointData>')

    xml_lines.extend([
        '    </Piece>',
        '  </UnstructuredGrid>',
        '</VTKFile>',
    ])

    with open(filename, 'w') as f:
        f.write('\n'.join(xml_lines) + '\n')


class VTKExportThread:
    """Background thread for VTK export."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.export_queue = queue.Queue(maxsize=5)
        self.worker_thread = threading.Thread(target=self._export_worker, daemon=True)
        self.stop_event = threading.Event()
        self.n_exported = 0

    def start(self):
        self.worker_thread.start()

    def enqueue_export(self, step, positions, element_ids, particle_ids=None,
                       extra_scalars=None):
        try:
            self.export_queue.put(
                (step, positions, element_ids, particle_ids, extra_scalars),
                timeout=10.0
            )
        except queue.Full:
            print(f"Warning: Export queue full at step {step}, skipping")

    def _export_worker(self):
        while not self.stop_event.is_set():
            try:
                data = self.export_queue.get(timeout=1.0)
                if data is None:
                    break
                step, positions, element_ids, particle_ids, extra_scalars = data

                output_file = self.output_dir / f"particles_step_{step:06d}.vtu"
                write_vtu_simple(
                    str(output_file),
                    positions=positions,
                    particle_ids=particle_ids,
                    element_ids=element_ids,
                    extra_scalars=extra_scalars,
                )
                self.n_exported += 1
                self.export_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Export error: {e}")

    def stop(self):
        self.export_queue.put(None)
        self.stop_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=30.0)


# =============================================================================
# RK4 Construction (inline, same as diagnostic)
# =============================================================================

def create_rk4_comparison(
    mesh_gpu_connectivity,
    mesh_gpu_node_positions,
    mesh_gpu_element_neighbors,
    mesh_gpu_element_volumes,
    mesh_gpu_global_morton,
    mesh_aligned_octree,
    enable_l0=True,
    enable_l1=True,
    n_hops=5,
    mesh_bbox_min=None,
    mesh_bbox_max=None,
    use_bbox_clamp=False,
    use_last_valid_vel=False,
    use_boundary_projection=False,
    boundary_projection_tol=1e-6,
    clamp_min_mask=None,
    clamp_max_mask=None,
    levelset_gpu=None,
):
    """Create RK4 step function (same inline pattern as diagnostic)."""
    use_levelset_mask = config.RK4_LEVELSET_MASK and levelset_gpu is not None

    # Per-wall masks: default to all-clamp if not provided
    if clamp_min_mask is None:
        clamp_min_mask = jnp.array([True, True, True])
    if clamp_max_mask is None:
        clamp_max_mask = jnp.array([True, True, True])

    def clamp_to_bbox(pos):
        """Clamp to bbox, applying +tol inward only on the clamped component."""
        tol = boundary_projection_tol
        hit_min = clamp_min_mask & (pos < mesh_bbox_min)
        clamped = jnp.where(hit_min, mesh_bbox_min + tol, pos)
        hit_max = clamp_max_mask & (clamped > mesh_bbox_max)
        clamped = jnp.where(hit_max, mesh_bbox_max - tol, clamped)
        return clamped

    connectivity = mesh_gpu_connectivity
    node_positions = mesh_gpu_node_positions
    element_neighbors = mesh_gpu_element_neighbors

    # ---- L0 ----
    def search_l0_single(pos, cached_elem_id):
        if not enable_l0:
            return jnp.int32(-1)
        is_valid = (cached_elem_id >= 0) & (cached_elem_id < len(connectivity))
        inside = jnp.where(
            is_valid,
            point_in_tet_dispatcher(pos, cached_elem_id, connectivity, node_positions,
                                    config.POINT_IN_TET_METHOD),
            False
        )
        return jnp.where(inside, cached_elem_id, jnp.int32(-1))

    # ---- L1 ----
    def search_l1_single(pos, start_elem_id):
        if not enable_l1:
            return jnp.int32(-1)

        current_elem = start_elem_id
        found = False

        start_elem_valid = start_elem_id >= 0
        start_volume = jnp.where(
            start_elem_valid,
            mesh_gpu_element_volumes[start_elem_id],
            config.FLOAT_DTYPE_JNP(1.0)
        )
        neighbors_of_start = element_neighbors[jnp.where(start_elem_valid, start_elem_id, 0)]
        valid_neighbor_mask = neighbors_of_start >= 0
        neighbor_volumes = jnp.where(
            valid_neighbor_mask,
            mesh_gpu_element_volumes[jnp.where(valid_neighbor_mask, neighbors_of_start, 0)],
            start_volume
        )
        median_neighbor_volume = jnp.median(neighbor_volumes)
        size_ratio = start_volume / (median_neighbor_volume + 1e-10)
        n_hops_adaptive = jnp.where(size_ratio < 0.1, jnp.int32(6), jnp.int32(n_hops))

        for hop_idx in range(6):
            hop_enabled = hop_idx < n_hops_adaptive
            should_search = (~found) & (current_elem >= 0) & hop_enabled

            neighbors = element_neighbors[jnp.where(should_search, current_elem, 0)]
            found_containing = jnp.int32(-1)

            for neighbor_idx in range(4):
                elem_id = neighbors[neighbor_idx]
                valid = elem_id >= 0
                check_this = (found_containing < 0) & valid
                inside = jnp.where(
                    check_this,
                    point_in_tet_dispatcher(pos, elem_id, connectivity, node_positions,
                                            config.POINT_IN_TET_METHOD),
                    False
                )
                found_containing = jnp.where(inside & check_this, elem_id, found_containing)

            first_valid_neighbor = jnp.where(
                jnp.any(neighbors >= 0),
                neighbors[jnp.argmax(neighbors >= 0)],
                current_elem
            )
            current_elem = jnp.where(
                should_search,
                jnp.where(found_containing >= 0, found_containing, first_valid_neighbor),
                current_elem
            )
            found = found | (found_containing >= 0)

        return jnp.where(found, current_elem, jnp.int32(-1))

    # ---- L2 ----
    def search_l2_single(pos):
        elem_id, _ = search_mesh_aligned_octree_multi_local_where(
            pos, mesh_aligned_octree, max_tests=jnp.int32(600)
        )
        return elem_id

    # ---- Combined search ----
    def search_l0_l1_l2(pos, cached_elem_id):
        elem_l0 = search_l0_single(pos, cached_elem_id)
        found_l0 = elem_l0 >= 0

        if enable_l1:
            elem_l1 = jnp.where(found_l0, elem_l0, search_l1_single(pos, cached_elem_id))
            found_l1 = elem_l1 >= 0
            elem_final = jnp.where(found_l1, elem_l1, search_l2_single(pos))
        else:
            elem_final = jnp.where(found_l0, elem_l0, search_l2_single(pos))

        return elem_final

    # ---- Velocity interpolation ----
    def interpolate_velocity_single(pos, elem_id, velocity_field):
        valid = (elem_id >= 0) & (elem_id < len(connectivity))
        nodes_idx = connectivity[elem_id]
        nodes = node_positions[nodes_idx]
        node_vels = velocity_field[nodes_idx]

        v0 = nodes[1] - nodes[0]
        v1 = nodes[2] - nodes[0]
        v2 = nodes[3] - nodes[0]
        vp = pos - nodes[0]

        d00, d01, d02 = jnp.dot(v0, v0), jnp.dot(v0, v1), jnp.dot(v0, v2)
        d11, d12 = jnp.dot(v1, v1), jnp.dot(v1, v2)
        d22 = jnp.dot(v2, v2)
        dp0, dp1, dp2 = jnp.dot(vp, v0), jnp.dot(vp, v1), jnp.dot(vp, v2)

        det = d00 * (d11*d22 - d12*d12) - d01 * (d01*d22 - d02*d12) + d02 * (d01*d12 - d02*d11)
        det = jnp.where(jnp.abs(det) < config.INTERPOLATION_DET_MIN, config.INTERPOLATION_DET_MIN, det)

        b1 = (dp0*(d11*d22-d12*d12) - d01*(dp1*d22-dp2*d12) + d02*(dp1*d12-dp2*d11)) / det
        b2 = (d00*(dp1*d22-dp2*d12) - dp0*(d01*d22-d02*d12) + d02*(d01*dp2-d02*dp1)) / det
        b3 = (d00*(d11*dp2-d12*dp1) - d01*(d01*dp2-d02*dp1) + dp0*(d01*d12-d02*d11)) / det
        b0 = 1.0 - b1 - b2 - b3

        vel = b0*node_vels[0] + b1*node_vels[1] + b2*node_vels[2] + b3*node_vels[3]

        # Level-set masking: zero velocity inside tool (level-set < 0)
        if use_levelset_mask:
            node_ls = levelset_gpu[nodes_idx]  # (4,)
            ls_val = b0 * node_ls[0] + b1 * node_ls[1] + b2 * node_ls[2] + b3 * node_ls[3]
            vel = jnp.where(ls_val >= 0.0, vel, jnp.zeros(3, dtype=config.FLOAT_DTYPE_JNP))

        return jnp.where(valid, vel, jnp.zeros(3, dtype=config.FLOAT_DTYPE_JNP))

    # ---- RK4 step ----
    @jax.jit
    def rk4_step(positions_gpu, element_ids_gpu, dt, velocity_fields_gpu, time_idx):
        n_timesteps = velocity_fields_gpu.shape[0]
        vel_idx = time_idx % n_timesteps
        velocity_field = velocity_fields_gpu[vel_idx]

        def rk4_single(pos, elem_id):
            # Stage 1
            elem_k1 = search_l0_l1_l2(pos, elem_id)
            vel_k1 = interpolate_velocity_single(pos, elem_k1, velocity_field)
            pos_k1 = pos + 0.5 * dt * vel_k1

            # Stage 2
            if use_bbox_clamp:
                pos_k1 = clamp_to_bbox(pos_k1)
            elem_k2 = search_l0_l1_l2(pos_k1, elem_k1)
            vel_k2 = interpolate_velocity_single(pos_k1, elem_k2, velocity_field)
            if use_last_valid_vel:
                vel_k2 = jnp.where(elem_k2 >= 0, vel_k2, vel_k1)
            pos_k2 = pos + 0.5 * dt * vel_k2

            # Stage 3
            if use_bbox_clamp:
                pos_k2 = clamp_to_bbox(pos_k2)
            elem_k3 = search_l0_l1_l2(pos_k2, elem_k2)
            vel_k3 = interpolate_velocity_single(pos_k2, elem_k3, velocity_field)
            if use_last_valid_vel:
                vel_k3 = jnp.where(elem_k3 >= 0, vel_k3, vel_k2)
            pos_k3 = pos + dt * vel_k3

            # Stage 4
            if use_bbox_clamp:
                pos_k3 = clamp_to_bbox(pos_k3)
            elem_k4 = search_l0_l1_l2(pos_k3, elem_k3)
            vel_k4 = interpolate_velocity_single(pos_k3, elem_k4, velocity_field)
            if use_last_valid_vel:
                vel_k4 = jnp.where(elem_k4 >= 0, vel_k4, vel_k3)

            pos_final = pos + (dt / 6.0) * (vel_k1 + 2.0*vel_k2 + 2.0*vel_k3 + vel_k4)
            elem_final = search_l0_l1_l2(pos_final, elem_k4)

            # Boundary projection: clamp to bbox and re-search if lost
            if use_boundary_projection:
                pos_clamped = clamp_to_bbox(pos_final)
                elem_clamped = search_l0_l1_l2(pos_clamped, elem_k4)
                lost = elem_final < 0
                pos_final = jnp.where(lost, pos_clamped, pos_final)
                elem_final = jnp.where(lost, elem_clamped, elem_final)

            return pos_final, elem_final

        positions_final, element_ids_final = jax.vmap(rk4_single)(
            positions_gpu, element_ids_gpu
        )
        return positions_final, element_ids_final

    return rk4_step


# =============================================================================
# Comparison Analysis
# =============================================================================

def compare_with_femuss(jaxtrace_positions, jaxtrace_elem_ids,
                        femuss_data, particle_indices, verbose=True):
    """
    Compare JAXTrace final positions with FEMUSS reference positions.

    Parameters
    ----------
    jaxtrace_positions : (N, 3) float32 — JAXTrace final positions
    jaxtrace_elem_ids : (N,) int32 — JAXTrace final element IDs
    femuss_data : dict — FEMUSS data at target timestep
    particle_indices : (N,) int — indices into FEMUSS arrays
    """
    femuss_current = femuss_data['current_positions'][particle_indices]
    femuss_left = femuss_data['has_left_domain'][particle_indices]

    jt_pos = np.array(jaxtrace_positions, dtype=np.float64)
    jt_eids = np.array(jaxtrace_elem_ids, dtype=np.int32)

    # Active vs lost
    jt_active = jt_eids >= 0
    jt_lost = ~jt_active
    n_total = len(jt_pos)
    n_active = int(jt_active.sum())
    n_lost = int(jt_lost.sum())

    # FEMUSS left domain count
    n_femuss_left = int(femuss_left.sum())
    n_femuss_active = n_total - n_femuss_left

    if verbose:
        print(f"\n{'='*80}")
        print(f"JAXTRACE vs FEMUSS COMPARISON")
        print(f"{'='*80}")
        print(f"  Particles: {n_total:,}")
        print(f"")
        print(f"  JAXTrace:  active={n_active:,} ({100*n_active/n_total:.2f}%), "
              f"lost={n_lost:,} ({100*n_lost/n_total:.2f}%)")
        print(f"  FEMUSS:    active={n_femuss_active:,} ({100*n_femuss_active/n_total:.2f}%), "
              f"left_domain={n_femuss_left:,} ({100*n_femuss_left/n_total:.2f}%)")

    # Position errors for ACTIVE particles (both JT active and FEMUSS active)
    both_active = jt_active & (~femuss_left)
    n_both_active = int(both_active.sum())

    if n_both_active > 0:
        errors = jt_pos[both_active] - femuss_current[both_active]
        error_mag = np.linalg.norm(errors, axis=1)

        # Displacement magnitude (to compute relative error)
        femuss_disp = femuss_data['displacements'][particle_indices[both_active]]
        disp_mag = np.linalg.norm(femuss_disp, axis=1)
        rel_errors = error_mag / (disp_mag + 1e-15)

        if verbose:
            print(f"\n  Position errors (both active, N={n_both_active:,}):")
            print(f"    Absolute error:")
            print(f"      Mean:   {error_mag.mean():.6e}")
            print(f"      Median: {np.median(error_mag):.6e}")
            print(f"      Max:    {error_mag.max():.6e}")
            print(f"      Std:    {error_mag.std():.6e}")
            for pct in [90, 95, 99, 99.9]:
                print(f"      P{pct:4.1f}:  {np.percentile(error_mag, pct):.6e}")

            print(f"    Relative error (|err| / |displacement|):")
            print(f"      Mean:   {rel_errors.mean():.6e}")
            print(f"      Median: {np.median(rel_errors):.6e}")
            print(f"      Max:    {rel_errors.max():.6e}")

            # Per-axis errors
            print(f"    Per-axis mean absolute error:")
            for ax, name in enumerate(['X', 'Y', 'Z']):
                ax_err = np.abs(errors[:, ax])
                print(f"      {name}: mean={ax_err.mean():.6e}, max={ax_err.max():.6e}")

            # Error distribution
            print(f"\n    Error magnitude distribution:")
            thresholds = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
            for thr in thresholds:
                count = int((error_mag < thr).sum())
                print(f"      < {thr:.0e}: {count:>8,} ({100*count/n_both_active:6.2f}%)")

    # Particles that FEMUSS lost but JAXTrace kept (and vice versa)
    jt_active_femuss_left = jt_active & femuss_left
    jt_lost_femuss_active = jt_lost & (~femuss_left)
    both_lost = jt_lost & femuss_left
    n_jt_active_femuss_left = int(jt_active_femuss_left.sum())
    n_jt_lost_femuss_active = int(jt_lost_femuss_active.sum())
    n_both_lost = int(both_lost.sum())

    if verbose:
        print(f"\n  Retention agreement:")
        print(f"    Both active:            {n_both_active:,}")
        print(f"    Both lost/left:         {n_both_lost:,}")
        print(f"    JAXTrace active, FEMUSS left: {n_jt_active_femuss_left:,}")
        print(f"    JAXTrace lost, FEMUSS active: {n_jt_lost_femuss_active:,}")

    # Worst-error particles detail
    if n_both_active > 0 and verbose:
        worst_idx = np.argsort(error_mag)[-5:][::-1]
        print(f"\n  Top 5 worst-error particles:")
        both_active_indices = np.where(both_active)[0]
        for rank, bi in enumerate(worst_idx):
            gi = both_active_indices[bi]  # global index
            pi = particle_indices[gi]     # FEMUSS particle index
            print(f"    #{rank+1}: particle={pi}, error={error_mag[bi]:.6e}, "
                  f"rel={rel_errors[bi]:.4e}")
            print(f"         JT:    [{jt_pos[gi,0]:.8e}, {jt_pos[gi,1]:.8e}, {jt_pos[gi,2]:.8e}]")
            print(f"         FEMUSS:[{femuss_current[gi,0]:.8e}, {femuss_current[gi,1]:.8e}, {femuss_current[gi,2]:.8e}]")

    print(f"{'='*80}")

    return {
        'n_total': n_total,
        'n_both_active': n_both_active,
        'n_jt_lost': n_lost,
        'n_femuss_left': n_femuss_left,
        'error_mag': error_mag if n_both_active > 0 else np.array([]),
        'rel_errors': rel_errors if n_both_active > 0 else np.array([]),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("FEMUSS Particle Tracking Comparison")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")

    femuss_end_step = FEMUSS_START_STEP + N_STEPS

    print(f"\nConfiguration:")
    print(f"  FEMUSS start step:    {FEMUSS_START_STEP}")
    print(f"  FEMUSS end step:      {femuss_end_step}")
    print(f"  N_STEPS:              {N_STEPS}")
    print(f"  DT:                   {DT}")
    print(f"  L0: {'ON' if ENABLE_L0_SEARCH else 'OFF'}")
    print(f"  L1: {'ON' if ENABLE_L1_SEARCH else 'OFF'}")
    print(f"  L2 method:            {L2_METHOD}")
    print(f"  Bbox clamp:           {'ON' if RK4_SUBSTEP_BBOX_CLAMP else 'OFF'}")
    print(f"  Last-valid vel:       {'ON' if RK4_SUBSTEP_LAST_VALID_VEL else 'OFF'}")
    print(f"  Boundary projection:  {'ON' if RK4_BOUNDARY_PROJECTION else 'OFF'} (tol={config.RK4_BOUNDARY_PROJECTION_TOL:.0e})")
    print(f"  Boundary walls:       {config.RK4_BOUNDARY_WALLS or 'all walls clamp (default)'}")
    print(f"  Precision:            {'float64' if config.USE_FLOAT64 else 'float32'}")
    print(f"  Collect stats:        {config.RK4_COLLECT_STATS} (library RK4 only, not this benchmark)")
    print(f"  Level-set mask:       {'ON' if config.RK4_LEVELSET_MASK else 'OFF'} (field: '{LEVELSET_FIELD_NAME}')")
    print("=" * 80)

    # Build per-wall clamp masks from config
    wall_config = config.RK4_BOUNDARY_WALLS
    if wall_config is not None and not isinstance(wall_config, dict):
        # Shorthand: 'clamp' or 'outlet' applied to all walls → treat as None (all clamp)
        # For per-wall control, use a dict: {'x_max': 'outlet', ...}
        print(f"  WARNING: RK4_BOUNDARY_WALLS='{wall_config}' is not a dict, treating as None (all clamp)")
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

    # ==================================================================
    # [1/7] Load mesh (identical to benchmark_rk4_diagnostic.py)
    # ==================================================================
    print(f"\n[1/7] Loading mesh...")
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=False
    )
    print(f"  Elements: {connectivity.shape[0]:,}, Nodes: {node_positions.shape[0]:,}")

    print("  Deduplicating...")
    node_positions, connectivity, n_dup, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    connectivity = connectivity.astype(np.int32)
    print(f"  Removed {n_dup:,} duplicates -> {node_positions.shape[0]:,} nodes")

    # Compute mesh bounding box for sub-step clamping
    mesh_bbox_min_cpu = node_positions.min(axis=0).astype(config.FLOAT_DTYPE_NP)
    mesh_bbox_max_cpu = node_positions.max(axis=0).astype(config.FLOAT_DTYPE_NP)
    print(f"  Mesh bbox: [{mesh_bbox_min_cpu}] → [{mesh_bbox_max_cpu}]")

    # Load level-set field for tool masking (if enabled)
    levelset_cpu = None
    if config.RK4_LEVELSET_MASK:
        # Load level-set from the first valid mesh file
        # Use the same file used for velocity to ensure same node ordering
        start_ts, end_ts = VELOCITY_TIMESTEP_RANGE
        ls_raw = None
        for ts in range(start_ts, end_ts + 1):
            ls_file = MESH_BASE_PATH / MESH_FILE_PATTERN.format(timestep=ts)
            reader = vtk.vtkXMLPUnstructuredGridReader()
            reader.SetFileName(str(ls_file))
            reader.Update()
            pd = reader.GetOutput().GetPointData()
            if pd.HasArray(LEVELSET_FIELD_NAME):
                ls_raw = vtk_to_numpy(pd.GetArray(LEVELSET_FIELD_NAME)).astype(np.float64)
                break
        if ls_raw is None:
            print(f"  WARNING: Level-set field '{LEVELSET_FIELD_NAME}' not found, disabling mask")
            config.RK4_LEVELSET_MASK = False
        else:
            # Deduplicate: expand to (1, n_nodes_raw, 3) to pass through deduplicate_nodes
            n_raw = ls_raw.shape[0]
            ls_scalar = ls_raw.ravel()  # (n_nodes_raw,)
            ls_as_vel = np.zeros((1, n_raw, 3), dtype=np.float64)
            ls_as_vel[0, :, 0] = ls_scalar
            # Re-load raw positions for dedup (same file)
            raw_pos = vtk_to_numpy(reader.GetOutput().GetPoints().GetData()).astype(np.float64)
            raw_conn = connectivity  # already remapped, but we need raw for dedup
            # Build node_map directly from raw positions (same logic as deduplicate_nodes)
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
            print(f"  Level-set loaded: {n_neg:,}/{n_dedup:,} nodes inside tool "
                  f"({100*n_neg/n_dedup:.1f}%)")

    # ==================================================================
    # [2/7] Precompute metadata (identical to benchmark_rk4_diagnostic.py)
    # ==================================================================
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
    print("  Done")

    # ==================================================================
    # [3/7] Build structures and upload to GPU (identical to benchmark_rk4_diagnostic.py)
    # ==================================================================
    print(f"\n[3/7] Building structures and uploading to GPU...")

    octree_struct = build_global_morton_octree(
        node_positions=node_positions,
        connectivity=connectivity,
        leaf_capacity=256, max_depth=21, verbose=False
    )

    mesh_octree_cells_multi = extract_octree_cells_vertex_multi(
        node_positions, connectivity, tolerance=1e-6, verbose=False
    )
    print(f"  Multi-cell octree: {mesh_octree_cells_multi.n_cells:,} cells, "
          f"{mesh_octree_cells_multi.elements_per_cell_mean:.1f} elem/cell, "
          f"{mesh_octree_cells_multi.cells_per_element_mean:.1f} cells/elem")

    element_neighbors = build_element_neighbors_array(connectivity, method='face', verbose=False)

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
    print("  Uploaded to GPU")

    # ==================================================================
    # [4/7] Load FEMUSS particles at start and end timesteps
    # ==================================================================
    print(f"\n[4/7] Loading FEMUSS particles...")

    start_file = FEMUSS_PARTICLE_PATH / FEMUSS_FILE_PATTERN.format(timestep=FEMUSS_START_STEP)
    end_file = FEMUSS_PARTICLE_PATH / FEMUSS_FILE_PATTERN.format(timestep=femuss_end_step)

    print(f"  Start: {start_file}")
    femuss_start = load_femuss_particles(start_file)
    print(f"    Particles: {femuss_start['n_particles']:,}")
    print(f"    Left domain: {femuss_start['has_left_domain'].sum():,}")

    print(f"  End:   {end_file}")
    femuss_end = load_femuss_particles(end_file)
    print(f"    Particles: {femuss_end['n_particles']:,}")
    print(f"    Left domain: {femuss_end['has_left_domain'].sum():,}")

    # Verify same particle count and ordering
    assert femuss_start['n_particles'] == femuss_end['n_particles'], \
        "Particle count mismatch between start and end FEMUSS files"
    assert np.allclose(femuss_start['initial_positions'], femuss_end['initial_positions']), \
        "Lagrangian positions differ — files are not from the same simulation"

    # Filter: only track particles that are active at start step
    active_at_start = ~femuss_start['has_left_domain']
    active_indices = np.where(active_at_start)[0]
    n_active_start = len(active_indices)
    print(f"\n  Active at start: {n_active_start:,} / {femuss_start['n_particles']:,}")

    # Use FEMUSS current positions (initial + displacement) as JAXTrace starting positions
    particle_positions = femuss_start['current_positions'][active_indices].astype(config.FLOAT_DTYPE_NP)

    # Particle IDs = FEMUSS array indices (for matching at comparison)
    particle_ids = active_indices.astype(np.int32)

    # ==================================================================
    # [5/7] Initial assignment (mesh-aligned multi-local, same as benchmark)
    # ==================================================================
    print(f"\n[5/7] Initial assignment...")
    positions_gpu = jax.device_put(particle_positions)
    n_particles = len(particle_positions)

    element_ids_initial = initial_assignment_mesh_aligned_multi_local(
        positions_gpu,
        mesh_aligned_octree_multi_gpu,
        batch_size=50000,
        max_tests=600,
        verbose=True
    )
    element_ids_initial = jax.block_until_ready(element_ids_initial)
    n_assigned = int(jnp.sum(element_ids_initial >= 0))
    print(f"  Assigned: {n_assigned:,}/{n_particles:,} ({100*n_assigned/n_particles:.2f}%)")

    n_unassigned = n_particles - n_assigned
    if n_unassigned > 0:
        print(f"  WARNING: {n_unassigned:,} particles not assigned to any element")

    # ==================================================================
    # [6/7] Build RK4 and run
    # ==================================================================
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
        mesh_bbox_min=mesh_bbox_min_gpu,
        mesh_bbox_max=mesh_bbox_max_gpu,
        use_bbox_clamp=RK4_SUBSTEP_BBOX_CLAMP,
        use_last_valid_vel=RK4_SUBSTEP_LAST_VALID_VEL,
        use_boundary_projection=RK4_BOUNDARY_PROJECTION,
        boundary_projection_tol=config.RK4_BOUNDARY_PROJECTION_TOL,
        clamp_min_mask=clamp_min_mask,
        clamp_max_mask=clamp_max_mask,
        levelset_gpu=levelset_gpu,
    )

    # Warmup
    print(f"  Compiling...")
    t_compile = time.time()
    positions_gpu, element_ids_gpu = rk4_step(
        positions_gpu, element_ids_initial, DT, velocity_sequence_gpu, 0
    )
    jax.block_until_ready(positions_gpu)
    print(f"  Compilation: {time.time() - t_compile:.1f}s")

    # Output directory
    output_subdir = OUTPUT_DIR / f"femuss_{FEMUSS_START_STEP}_to_{femuss_end_step}"
    output_subdir.mkdir(parents=True, exist_ok=True)

    # CSV stats
    stats_csv_path = output_subdir / "search_stats.csv"
    stats_csv = open(stats_csv_path, 'w')
    stats_csv.write("step,n_active,n_lost,new_lost\n")

    # Setup VTU export
    exporter = VTKExportThread(output_subdir)
    exporter.start()

    # Re-initialize for actual run
    positions_gpu = jax.device_put(particle_positions)
    element_ids_gpu = element_ids_initial

    # Export initial state
    pos_cpu = np.array(positions_gpu, dtype=config.FLOAT_DTYPE_NP)
    eid_cpu = np.array(element_ids_initial, dtype=np.int32)
    exporter.enqueue_export(0, pos_cpu, eid_cpu, particle_ids)
    print(f"  Exported initial state (step 0)")

    print(f"\n[7/7] Running {N_STEPS} RK4 steps...")
    print("=" * 80)

    t_start = time.time()
    prev_lost = 0
    new_lost = 0

    for step in range(1, N_STEPS + 1):
        positions_gpu, element_ids_gpu = rk4_step(
            positions_gpu, element_ids_gpu, DT, velocity_sequence_gpu, step - 1
        )

        if step % LOG_INTERVAL == 0 or step == N_STEPS:
            eid_cpu = np.array(element_ids_gpu, dtype=np.int32)
            n_active = int(np.sum(eid_cpu >= 0))
            n_lost = n_particles - n_active
            new_lost = n_lost - prev_lost

            stats_csv.write(f"{step},{n_active},{n_lost},{new_lost}\n")

            print(f"  Step {step:5d}: active={n_active:,} lost={n_lost:,} (+{new_lost})")

            prev_lost = n_lost

        # Export VTU
        if step % EXPORT_FREQUENCY == 0 or step == N_STEPS or new_lost > 0:
            pos_cpu = np.array(positions_gpu, dtype=config.FLOAT_DTYPE_NP)
            eid_cpu = np.array(element_ids_gpu, dtype=np.int32)
            exporter.enqueue_export(step, pos_cpu, eid_cpu, particle_ids)

    t_elapsed = time.time() - t_start
    stats_csv.close()
    exporter.stop()
    print(f"  Exported {exporter.n_exported} VTU files")

    # ==================================================================
    # Summary & Comparison
    # ==================================================================
    print(f"\n{'='*80}")
    print(f"TRACKING SUMMARY")
    print(f"{'='*80}")
    n_active_final = int(jnp.sum(element_ids_gpu >= 0))
    n_lost_final = n_particles - n_active_final
    print(f"  Particles: {n_particles:,}")
    print(f"  Final active: {n_active_final:,} ({100*n_active_final/n_particles:.2f}%)")
    print(f"  Final lost: {n_lost_final:,} ({100*n_lost_final/n_particles:.2f}%)")
    print(f"  Steps: {N_STEPS}")
    print(f"  Time: {t_elapsed:.1f}s ({n_particles * N_STEPS / t_elapsed:,.0f} p*step/s)")
    print(f"  Output: {output_subdir}")

    # Compare with FEMUSS end state
    jt_positions_final = np.array(positions_gpu, dtype=config.FLOAT_DTYPE_NP)
    jt_elem_ids_final = np.array(element_ids_gpu, dtype=np.int32)

    comparison = compare_with_femuss(
        jt_positions_final, jt_elem_ids_final,
        femuss_end, particle_ids,
        verbose=True
    )

    # Save comparison data
    np.savez(
        output_subdir / "comparison_data.npz",
        jaxtrace_positions=jt_positions_final,
        jaxtrace_elem_ids=jt_elem_ids_final,
        femuss_current_positions=femuss_end['current_positions'][particle_ids],
        femuss_has_left=femuss_end['has_left_domain'][particle_ids],
        particle_ids=particle_ids,
        error_magnitudes=comparison['error_mag'],
        config={
            'femuss_start_step': FEMUSS_START_STEP,
            'femuss_end_step': femuss_end_step,
            'n_steps': N_STEPS,
            'dt': DT,
        },
    )
    print(f"\n  Comparison data saved to {output_subdir / 'comparison_data.npz'}")

    print(f"\nDone.")


if __name__ == '__main__':
    main()
