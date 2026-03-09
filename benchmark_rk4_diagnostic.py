#!/usr/bin/env python3
"""
RK4 Diagnostic Benchmark — Focused Investigation of Particle Loss

This is a diagnostic variant of benchmark_l2_search_methods_with-export.py
designed to isolate WHY particles are lost inside the vmapped RK4 pipeline
while L2 search works 100% in standalone tests.

Key diagnostic features:
1. Load initial positions from previous simulation VTU output at user-specified step
2. Positional filter to focus on regions where loss occurs
3. Toggle L0, L1 on/off to isolate search level contributions
4. Per-sub-step logging inside RK4 (which sub-step fails?)
5. Standalone L2 verification of intermediate positions
6. Comparison with previous run's search_stats.csv

Usage:
    python benchmark_rk4_diagnostic.py 2>&1 | tee logs/benchmark_rk4_diagnostic.log
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'

import sys
import time
import queue
import threading
import csv
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

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
)
from jaxtrace.tracking.seeding import uniform_grid_seeds
from jaxtrace.gpu.search.aa_detection import precompute_aa_metadata, precompute_element_vertices
from jaxtrace.gpu.search.point_in_tet_methods import set_corrected_metadata, set_inverse_matrices_gpu
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
from jaxtrace.gpu.search.mesh_aligned_point_location import (
    search_mesh_aligned_octree_multi_local_where,
)
from jaxtrace.gpu.search.point_in_tet_methods import (
    point_in_tet_gpu as point_in_tet_dispatcher,
)
import jaxtrace.config as config

# Import VTK for loading previous results
try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    HAS_VTK = True
except ImportError:
    HAS_VTK = False

# =============================================================================
# USER CONFIGURATION — Edit these to focus the diagnostic
# =============================================================================

# --- Mesh ---
MESH_BASE_PATH = Path("data/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_FIELD_NAME = 'Displacement'

# --- Starting timestep ---
# Mesh velocity timestep range: (start, end)
# The mesh .pvtu numbering, not the RK4 step number.
VELOCITY_TIMESTEP_RANGE = (158, 159)

# --- Load initial positions from previous output? ---
# Set 'enabled' to True to load positions from a previous VTU file.
# Set 'enabled' to False to generate fresh uniform grid particles.
LOAD_FROM_PREVIOUS = {
    'enabled': True,
    # Directory containing particles_step_XXXXXX.vtu files
    'output_dir': Path("output/benchmark_with_export/"
                       "Mesh-Aligned_Multi-Cell_+_3×3×3_Local_(jnp.where)"),
    # Which step to load (0 = initial, 100 = step 100, etc.)
    'step': 1125,
}

# --- Positional filter ---
# Filter loaded/generated particles to a spatial region of interest.
#
# 'mode' selects how the bounds are interpreted:
#   'fraction'  — values are fractions (0-1) of the mesh bounding box
#                 e.g. x=(0.20, 0.35) means 20%-35% of domain X extent
#   'absolute'  — values are raw world coordinates (metres, same units as mesh)
#                 e.g. x=(-0.022, -0.016) means exactly those X coordinates
#
# Use None for any axis to skip filtering along that axis.
POSITION_FILTER = {
    'enabled': False,
    'mode': 'fraction',   # 'fraction' or 'absolute'
    'x': (0.20, 0.35),   # fraction: 20-35% of domain X  |  absolute: world coords
    'y': (0.2, 0.8),
    'z': (0.01, 1.0),
}

# --- Particle seeding (used only if LOAD_FROM_PREVIOUS is disabled) ---
PARTICLE_SEEDING = 'uniform_grid'
PARTICLE_GRID_RESOLUTION = (60, 90, 60)
PARTICLE_BOUNDS_FRACTION = {
    'x': (0.12, 0.22),
    'y': (0.2, 0.8),
    'z': (0.01, 1.0),
}

# --- RK4 ---
DT = 0.0025
N_STEPS = 500
POINT_IN_TET_METHOD = 'inverse'

# --- Search configuration ---
# Toggle L0, L1 on/off to isolate their effect on particle loss
ENABLE_L0_SEARCH = True   # Set False to skip cached element check
ENABLE_L1_SEARCH = True   # Set False to skip neighbor hops
N_HOPS = 5

# --- RK4 sub-step recovery ---
RK4_SUBSTEP_BBOX_CLAMP = True
RK4_SUBSTEP_LAST_VALID_VEL = True

# --- L2 method ---
# 'mesh_aligned_octree_multi_local_where' is the default
L2_METHOD = 'mesh_aligned_octree_multi_local_where'

# --- Export & logging ---
EXPORT_FREQUENCY = 1   # Export EVERY step for detailed analysis
LOG_INTERVAL = 1        # Log stats every step
OUTPUT_DIR = Path("output/rk4_diagnostic")

# --- Comparison with previous run ---
COMPARE_STATS_CSV = Path(
    "output/benchmark_with_export/"
    "Mesh-Aligned Multi-Cell + 3×3×3 Local (jnp.where)/search_stats.csv"
)

# --- Standalone L2 verification ---
# After each RK4 step, take the newly-lost particles and verify them with
# standalone L2 search (outside the fused RK4 graph).
VERIFY_LOST_WITH_STANDALONE_L2 = True
STANDALONE_L2_BATCH_SIZE = 50000

# --- Brute-force verification ---
# For newly-lost particles, also do CPU brute-force point-in-tet against ALL
# mesh elements. This is slow but definitively answers whether the particle is
# inside the mesh.  Only the first BRUTEFORCE_MAX_PER_STEP newly-lost particles
# are tested per step (0 = disabled).
BRUTEFORCE_MAX_PER_STEP = 5

SEED = 42

# =============================================================================
# VTU I/O
# =============================================================================

def load_vtu(filepath):
    """Load positions, particle_ids, element_ids from a VTU file."""
    if not HAS_VTK:
        raise RuntimeError("VTK is required to load VTU files. Install with: pip install vtk")

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(filepath))
    reader.Update()
    output = reader.GetOutput()

    pts = vtk_to_numpy(output.GetPoints().GetData()).astype(np.float32)

    pd = output.GetPointData()
    pid_arr = pd.GetArray('ParticleID')
    eid_arr = pd.GetArray('ElementID')

    return {
        'positions':    pts,
        'particle_ids': vtk_to_numpy(pid_arr).astype(np.int32) if pid_arr else np.arange(len(pts), dtype=np.int32),
        'element_ids':  vtk_to_numpy(eid_arr).astype(np.int32) if eid_arr else np.full(len(pts), -1, dtype=np.int32),
    }


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


# =============================================================================
# VTK Export Thread (same as benchmark)
# =============================================================================

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

                # Write ALL particles (active + lost) with element_ids to track loss
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
# RK4 with per-sub-step diagnostics
# =============================================================================

def create_rk4_diagnostic(
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
):
    """
    Create a diagnostic RK4 step function that reports per-sub-step hit levels.

    Returns:
        rk4_step: (positions, element_ids, dt, vel_fields, time_idx) ->
                  (positions_final, element_ids_final)
        rk4_step_with_substep_stats: same inputs ->
                  (positions_final, element_ids_final, substep_hit_levels)
                  where substep_hit_levels is (N, 5) int8
                    0=L0, 1=L1, 2=L2, -1=miss
    """

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
            jnp.float32(1.0)
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

    # ---- Combined search with hit level ----
    def search_l0_l1_l2_with_level(pos, cached_elem_id):
        elem_l0 = search_l0_single(pos, cached_elem_id)
        found_l0 = elem_l0 >= 0

        if enable_l1:
            elem_l1_raw = search_l1_single(pos, cached_elem_id)
            elem_l1 = jnp.where(found_l0, elem_l0, elem_l1_raw)
            found_l1 = elem_l1 >= 0

            elem_l2 = search_l2_single(pos)
            elem_final = jnp.where(found_l1, elem_l1, elem_l2)
            found_l2 = elem_l2 >= 0

            hit_level = jnp.where(
                found_l0, jnp.int8(0),
                jnp.where(found_l1, jnp.int8(1),
                          jnp.where(found_l2, jnp.int8(2), jnp.int8(-1)))
            )
        else:
            elem_l2 = search_l2_single(pos)
            elem_final = jnp.where(found_l0, elem_l0, elem_l2)
            found_l2 = elem_l2 >= 0

            hit_level = jnp.where(
                found_l0, jnp.int8(0),
                jnp.where(found_l2, jnp.int8(2), jnp.int8(-1))
            )
        return elem_final, hit_level

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
        det = jnp.where(jnp.abs(det) < 1e-12, 1e-12, det)

        b1 = (dp0*(d11*d22-d12*d12) - d01*(dp1*d22-dp2*d12) + d02*(dp1*d12-dp2*d11)) / det
        b2 = (d00*(dp1*d22-dp2*d12) - dp0*(d01*d22-d02*d12) + d02*(d01*dp2-d02*dp1)) / det
        b3 = (d00*(d11*dp2-d12*dp1) - d01*(d01*dp2-d02*dp1) + dp0*(d01*d12-d02*d11)) / det
        b0 = 1.0 - b1 - b2 - b3

        vel = b0*node_vels[0] + b1*node_vels[1] + b2*node_vels[2] + b3*node_vels[3]
        return jnp.where(valid, vel, jnp.zeros(3, dtype=jnp.float32))

    # ---- RK4 with sub-step stats ----
    @jax.jit
    def rk4_step_with_substep_stats(positions_gpu, element_ids_gpu, dt,
                                     velocity_fields_gpu, time_idx):
        n_timesteps = velocity_fields_gpu.shape[0]
        vel_idx = time_idx % n_timesteps
        velocity_field = velocity_fields_gpu[vel_idx]

        def rk4_single(pos, elem_id):
            # k1
            elem_k1, lvl_k1 = search_l0_l1_l2_with_level(pos, elem_id)
            vel_k1 = interpolate_velocity_single(pos, elem_k1, velocity_field)
            pos_k1 = pos + 0.5 * dt * vel_k1

            # k2
            if use_bbox_clamp:
                pos_k1 = jnp.clip(pos_k1, mesh_bbox_min, mesh_bbox_max)
            elem_k2, lvl_k2 = search_l0_l1_l2_with_level(pos_k1, elem_k1)
            vel_k2 = interpolate_velocity_single(pos_k1, elem_k2, velocity_field)
            if use_last_valid_vel:
                vel_k2 = jnp.where(elem_k2 >= 0, vel_k2, vel_k1)
            pos_k2 = pos + 0.5 * dt * vel_k2

            # k3
            if use_bbox_clamp:
                pos_k2 = jnp.clip(pos_k2, mesh_bbox_min, mesh_bbox_max)
            elem_k3, lvl_k3 = search_l0_l1_l2_with_level(pos_k2, elem_k2)
            vel_k3 = interpolate_velocity_single(pos_k2, elem_k3, velocity_field)
            if use_last_valid_vel:
                vel_k3 = jnp.where(elem_k3 >= 0, vel_k3, vel_k2)
            pos_k3 = pos + dt * vel_k3

            # k4
            if use_bbox_clamp:
                pos_k3 = jnp.clip(pos_k3, mesh_bbox_min, mesh_bbox_max)
            elem_k4, lvl_k4 = search_l0_l1_l2_with_level(pos_k3, elem_k3)
            vel_k4 = interpolate_velocity_single(pos_k3, elem_k4, velocity_field)
            if use_last_valid_vel:
                vel_k4 = jnp.where(elem_k4 >= 0, vel_k4, vel_k3)

            # Final
            pos_final = pos + (dt / 6.0) * (vel_k1 + 2.0*vel_k2 + 2.0*vel_k3 + vel_k4)
            elem_final, lvl_final = search_l0_l1_l2_with_level(pos_final, elem_k4)

            hit_levels = jnp.array([lvl_k1, lvl_k2, lvl_k3, lvl_k4, lvl_final], dtype=jnp.int8)
            return pos_final, elem_final, hit_levels

        positions_final, element_ids_final, all_hit_levels = jax.vmap(rk4_single)(
            positions_gpu, element_ids_gpu
        )

        return positions_final, element_ids_final, all_hit_levels

    # Non-stats version for warmup compilation
    @jax.jit
    def rk4_step(positions_gpu, element_ids_gpu, dt, velocity_fields_gpu, time_idx):
        pos_final, elem_final, _ = rk4_step_with_substep_stats(
            positions_gpu, element_ids_gpu, dt, velocity_fields_gpu, time_idx
        )
        return pos_final, elem_final

    return rk4_step, rk4_step_with_substep_stats


# =============================================================================
# Standalone L2 verification (outside RK4 graph)
# =============================================================================

def standalone_l2_search_batch(positions_gpu, octree_gpu, batch_size=50000):
    """Run L2 search on positions OUTSIDE the RK4 graph, in batches."""
    max_tests_jax = jnp.int32(600)

    @jax.jit
    def _search_batch(positions_batch):
        def single(pos):
            elem_id, n_tests = search_mesh_aligned_octree_multi_local_where(
                pos, octree_gpu, max_tests=max_tests_jax
            )
            return elem_id, n_tests
        return jax.vmap(single)(positions_batch)

    n = positions_gpu.shape[0]
    all_elem_ids = np.full(n, -1, dtype=np.int32)
    all_n_tests = np.full(n, 0, dtype=np.int32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = positions_gpu[start:end]
        elem_ids_batch, n_tests_batch = _search_batch(batch)
        all_elem_ids[start:end] = np.array(elem_ids_batch, dtype=np.int32)
        all_n_tests[start:end] = np.array(n_tests_batch, dtype=np.int32)

    return all_elem_ids, all_n_tests


# =============================================================================
# Mesh Quality Analysis
# =============================================================================

def analyze_mesh_quality(node_positions, connectivity, element_volumes, det_values,
                         element_neighbors, verbose=True):
    """
    Comprehensive mesh quality analysis to detect potential artifact elements.

    Checks:
    1. Zero/near-zero volume elements (degenerate tets from merging/dedup)
    2. Negative-determinant elements (inverted tets)
    3. Volume distribution and outliers
    4. Face neighbor consistency (1:2 / 2:1 refinement boundaries)
    5. Elements with fewer than 4 face neighbors (boundary detection)
    6. Duplicate elements (same 4 nodes after dedup)
    """
    n_elements = len(connectivity)
    print("\n" + "=" * 80)
    print("MESH QUALITY ANALYSIS")
    print("=" * 80)

    # --- 1. Volume analysis ---
    vol_zero = np.sum(element_volumes < 1e-20)
    vol_tiny = np.sum(element_volumes < 1e-15)
    vol_small = np.sum((element_volumes >= 1e-15) & (element_volumes < 1e-12))
    vol_min = np.min(element_volumes)
    vol_max = np.max(element_volumes)
    vol_median = np.median(element_volumes)
    vol_mean = np.mean(element_volumes)

    print(f"\n  1. Element volumes ({n_elements:,} elements):")
    print(f"     Min: {vol_min:.4e}  Max: {vol_max:.4e}")
    print(f"     Mean: {vol_mean:.4e}  Median: {vol_median:.4e}")
    print(f"     Zero volume (< 1e-20): {vol_zero:,}")
    print(f"     Tiny volume (< 1e-15): {vol_tiny:,}")
    print(f"     Small volume (< 1e-12): {vol_small:,}")
    if vol_tiny > 0:
        tiny_ids = np.where(element_volumes < 1e-15)[0]
        print(f"     *** WARNING: {vol_tiny} potentially degenerate elements! ***")
        if len(tiny_ids) <= 20:
            for eid in tiny_ids:
                nodes = connectivity[eid]
                verts = node_positions[nodes]
                print(f"       elem {eid}: vol={element_volumes[eid]:.4e} "
                      f"nodes={nodes} "
                      f"pos={verts.mean(axis=0)}")

    # Volume ratio (max/min for non-degenerate)
    non_degen_vols = element_volumes[element_volumes > 1e-20]
    if len(non_degen_vols) > 0:
        vol_ratio = np.max(non_degen_vols) / np.min(non_degen_vols)
        print(f"     Volume ratio (max/min, non-degenerate): {vol_ratio:.1f}x")

    # --- 2. Determinant analysis ---
    n_negative_det = np.sum(det_values < 0)
    n_zero_det = np.sum(np.abs(det_values) < 1e-15)
    print(f"\n  2. Determinant analysis:")
    print(f"     Negative det (inverted tets): {n_negative_det:,}")
    print(f"     Zero det (degenerate): {n_zero_det:,}")
    if n_negative_det > 0:
        neg_ids = np.where(det_values < 0)[0]
        print(f"     *** WARNING: {n_negative_det} inverted elements! ***")
        if len(neg_ids) <= 10:
            for eid in neg_ids[:10]:
                print(f"       elem {eid}: det={det_values[eid]:.4e} vol={element_volumes[eid]:.4e}")

    # --- 3. Duplicate element check ---
    print(f"\n  3. Duplicate element check:")
    sorted_conn = np.sort(connectivity, axis=1)
    _, unique_indices, unique_counts = np.unique(
        sorted_conn, axis=0, return_index=True, return_counts=True
    )
    n_duplicates = np.sum(unique_counts > 1)
    if n_duplicates > 0:
        print(f"     *** WARNING: {n_duplicates} duplicate element sets! ***")
        dup_mask = unique_counts > 1
        for idx in unique_indices[dup_mask][:5]:
            nodes = sorted_conn[idx]
            count = unique_counts[np.where(unique_indices == idx)[0][0]]
            print(f"       nodes={nodes} appears {count} times")
    else:
        print(f"     No duplicate elements found")

    # --- 4. Face neighbor analysis ---
    print(f"\n  4. Face neighbor analysis:")
    # Count valid neighbors per element
    # element_neighbors is (n_elements, 4) with -1 for no neighbor
    n_neighbors = np.sum(element_neighbors >= 0, axis=1)
    n_boundary_elems = np.sum(n_neighbors < 4)
    neighbor_hist = np.bincount(n_neighbors, minlength=5)

    print(f"     Neighbors distribution:")
    for k in range(5):
        if neighbor_hist[k] > 0:
            print(f"       {k} neighbors: {neighbor_hist[k]:,} elements "
                  f"({100*neighbor_hist[k]/n_elements:.2f}%)")
    print(f"     Boundary elements (< 4 neighbors): {n_boundary_elems:,} "
          f"({100*n_boundary_elems/n_elements:.2f}%)")

    # Check for asymmetric neighbors (A neighbors B but B doesn't neighbor A)
    n_asymmetric = 0
    for eid in range(min(n_elements, 100000)):  # Sample first 100K
        for face_idx in range(4):
            nid = element_neighbors[eid, face_idx]
            if nid >= 0:
                # Check if nid has eid as neighbor
                if eid not in element_neighbors[nid]:
                    n_asymmetric += 1
    sample_size = min(n_elements, 100000)
    if n_asymmetric > 0:
        print(f"     *** WARNING: {n_asymmetric} asymmetric neighbor pairs "
              f"(in first {sample_size:,} elements) ***")
        print(f"     This may indicate 1:2 / 2:1 face refinement boundaries!")
    else:
        print(f"     Neighbor symmetry OK (checked {sample_size:,} elements)")

    # --- 5. Hanging nodes / 1:2 face refinement ---
    print(f"\n  5. Refinement boundary analysis (face size mismatch):")
    # For each face, compute face area. If two elements share a face but
    # the face areas differ significantly, this indicates a refinement boundary.
    n_mismatched_faces = 0
    face_area_ratios = []
    for eid in range(min(n_elements, 50000)):  # Sample
        nodes_eid = connectivity[eid]
        verts_eid = node_positions[nodes_eid]
        face_node_combos = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
        for face_idx, (a, b, c) in enumerate(face_node_combos):
            nid = element_neighbors[eid, face_idx]
            if nid < 0 or nid <= eid:  # Skip boundary + avoid double-counting
                continue
            # Compute face area for this element's face
            e1 = verts_eid[b] - verts_eid[a]
            e2 = verts_eid[c] - verts_eid[a]
            area_eid = 0.5 * np.linalg.norm(np.cross(e1, e2))

            # Compute smallest face area of neighbor
            nodes_nid = connectivity[nid]
            verts_nid = node_positions[nodes_nid]
            min_area_nid = float('inf')
            for na, nb, nc in face_node_combos:
                e1n = verts_nid[nb] - verts_nid[na]
                e2n = verts_nid[nc] - verts_nid[na]
                area_n = 0.5 * np.linalg.norm(np.cross(e1n, e2n))
                min_area_nid = min(min_area_nid, area_n)

            if area_eid > 0 and min_area_nid > 0:
                ratio = max(area_eid, min_area_nid) / min(area_eid, min_area_nid)
                if ratio > 3.5:  # Face is ~4x larger → 1:2 refinement
                    n_mismatched_faces += 1
                    face_area_ratios.append(ratio)

    if n_mismatched_faces > 0:
        print(f"     Found {n_mismatched_faces} face pairs with area ratio > 3.5x "
              f"(in first {min(n_elements, 50000):,} elements)")
        print(f"     Ratio range: {min(face_area_ratios):.1f}x - {max(face_area_ratios):.1f}x")
        print(f"     These are likely 1:2 refinement boundaries where particles "
              f"can slip through!")
    else:
        print(f"     No significant face area mismatches found")

    # --- 6. Summary ---
    issues = []
    if vol_tiny > 0:
        issues.append(f"{vol_tiny} degenerate (tiny volume) elements")
    if n_negative_det > 0:
        issues.append(f"{n_negative_det} inverted elements")
    if n_duplicates > 0:
        issues.append(f"{n_duplicates} duplicate element sets")
    if n_asymmetric > 0:
        issues.append(f"{n_asymmetric} asymmetric neighbor pairs")
    if n_mismatched_faces > 0:
        issues.append(f"{n_mismatched_faces} refinement boundary faces")

    print(f"\n  SUMMARY:")
    if issues:
        print(f"     *** ISSUES FOUND: ***")
        for issue in issues:
            print(f"       - {issue}")
    else:
        print(f"     No mesh quality issues detected")

    print("=" * 80)

    return {
        'vol_tiny': vol_tiny,
        'vol_zero': vol_zero,
        'n_negative_det': n_negative_det,
        'n_duplicates': n_duplicates,
        'n_asymmetric': n_asymmetric,
        'n_mismatched_faces': n_mismatched_faces,
        'n_boundary_elems': n_boundary_elems,
        'vol_min': vol_min,
        'vol_max': vol_max,
    }


# =============================================================================
# Lost Particle Deep Analysis
# =============================================================================

def analyze_lost_particles(lost_positions, node_positions, connectivity,
                           element_volumes, det_values, element_neighbors,
                           mesh_aligned_octree_multi_gpu, M_inv_array, p0_array,
                           max_analyze=500, verbose=True):
    """
    Deep analysis of lost particle positions.

    For each lost particle:
    1. Check if position is inside mesh bounding box
    2. Brute-force point-in-tet against ALL elements (CPU, sampled)
    3. Check which octree cells the position maps to at each level
    4. If brute-force finds an element, check if that element is in the octree
    5. Analyze the neighborhood: what elements are nearby?
    """
    n_lost = len(lost_positions)
    if n_lost == 0:
        print("  No lost particles to analyze")
        return {}

    domain_min = node_positions.min(axis=0)
    domain_max = node_positions.max(axis=0)

    print(f"\n  Analyzing {min(n_lost, max_analyze)} of {n_lost:,} lost particles...")

    # Sample if too many
    if n_lost > max_analyze:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n_lost, max_analyze, replace=False)
        sample_positions = lost_positions[sample_idx]
    else:
        sample_positions = lost_positions
        sample_idx = np.arange(n_lost)

    n_sample = len(sample_positions)

    # --- 1. Bounding box check ---
    inside_bbox = np.all(
        (sample_positions >= domain_min) & (sample_positions <= domain_max),
        axis=1
    )
    n_inside_bbox = int(np.sum(inside_bbox))
    n_outside_bbox = n_sample - n_inside_bbox
    print(f"\n  1. Bounding box check ({n_sample} sampled):")
    print(f"     Inside mesh bbox:  {n_inside_bbox:,}")
    print(f"     Outside mesh bbox: {n_outside_bbox:,}")
    if n_outside_bbox > 0:
        outside_pos = sample_positions[~inside_bbox]
        for i in range(min(5, n_outside_bbox)):
            pos = outside_pos[i]
            dist_to_bbox = np.maximum(domain_min - pos, pos - domain_max)
            dist_to_bbox = np.maximum(dist_to_bbox, 0)
            print(f"       pos={pos}, distance outside bbox={dist_to_bbox}")

    # --- 2. Brute-force point-in-tet (CPU, on particles inside bbox) ---
    inside_positions = sample_positions[inside_bbox]
    n_to_bruteforce = len(inside_positions)
    print(f"\n  2. Brute-force point-in-tet ({n_to_bruteforce} inside bbox):")

    if n_to_bruteforce > 0:
        # Use precomputed inverse matrices for fast CPU point-in-tet
        bf_found = 0
        bf_not_found = 0
        bf_found_elements = []
        bf_not_found_positions = []
        tolerance = 1e-6

        # Vectorized brute force: for each particle, test all elements
        # Use batches to avoid memory issues
        bf_batch_size = min(50, n_to_bruteforce)
        for pi in range(min(bf_batch_size, n_to_bruteforce)):
            pos = inside_positions[pi]
            found_elem = -1

            # Vectorized: compute barycentric coords for ALL elements at once
            local = pos - p0_array  # (n_elements, 3)
            # bary = M_inv @ local for each element
            bary = np.einsum('eij,ej->ei', M_inv_array, local)  # (n_elements, 3)
            bary4 = 1.0 - bary.sum(axis=1)  # fourth barycentric coordinate

            # Check containment with tolerance
            inside_mask = (
                (bary[:, 0] >= -tolerance) &
                (bary[:, 1] >= -tolerance) &
                (bary[:, 2] >= -tolerance) &
                (bary4 >= -tolerance)
            )
            containing_elems = np.where(inside_mask)[0]

            if len(containing_elems) > 0:
                bf_found += 1
                bf_found_elements.append({
                    'position': pos.copy(),
                    'elements': containing_elems.copy(),
                    'volumes': element_volumes[containing_elems].copy(),
                    'det_values': det_values[containing_elems].copy(),
                })
            else:
                bf_not_found += 1
                bf_not_found_positions.append(pos.copy())

        print(f"     Brute-force tested: {min(bf_batch_size, n_to_bruteforce)} particles")
        print(f"     Found by brute-force: {bf_found}")
        print(f"     NOT found by brute-force: {bf_not_found}")

        if bf_found > 0:
            print(f"\n     *** CRITICAL: {bf_found} particles ARE inside mesh elements ***")
            print(f"     *** but octree search cannot find them! ***")
            print(f"\n     Details of particles found by brute-force but missed by octree:")
            for i, info in enumerate(bf_found_elements[:10]):
                pos = info['position']
                elems = info['elements']
                vols = info['volumes']
                dets = info['det_values']
                # Check element properties
                print(f"\n       Particle {i}: pos=({pos[0]:.8f}, {pos[1]:.8f}, {pos[2]:.8f})")
                print(f"         Found in {len(elems)} element(s):")
                for j, (eid, vol, det) in enumerate(zip(elems[:5], vols[:5], dets[:5])):
                    nodes = connectivity[eid]
                    verts = node_positions[nodes]
                    centroid = verts.mean(axis=0)
                    n_nbrs = np.sum(element_neighbors[eid] >= 0)
                    edge_lens = []
                    for a in range(4):
                        for b in range(a+1, 4):
                            edge_lens.append(np.linalg.norm(verts[a] - verts[b]))
                    edge_lens = np.array(edge_lens)
                    print(f"           elem {eid}: vol={vol:.4e} det={det:.4e} "
                          f"neighbors={n_nbrs}/4 "
                          f"edge_range=[{edge_lens.min():.4e}, {edge_lens.max():.4e}]")

                    # Check octree registration for this element
                    _check_element_in_octree(eid, node_positions, connectivity,
                                            mesh_aligned_octree_multi_gpu)

        if bf_not_found > 0:
            print(f"\n     Particles NOT in any element (true mesh gaps):")
            for i in range(min(5, bf_not_found)):
                pos = bf_not_found_positions[i]
                # Find nearest element centroid
                centroids = node_positions[connectivity].mean(axis=1)  # (n_elem, 3)
                dists = np.linalg.norm(centroids - pos, axis=1)
                nearest_idx = np.argpartition(dists, 3)[:3]
                nearest_idx = nearest_idx[np.argsort(dists[nearest_idx])]
                print(f"       pos=({pos[0]:.8f}, {pos[1]:.8f}, {pos[2]:.8f})")
                for nid in nearest_idx:
                    verts = node_positions[connectivity[nid]]
                    vol = element_volumes[nid]
                    n_nbrs = np.sum(element_neighbors[nid] >= 0)
                    print(f"         nearest elem {nid}: dist={dists[nid]:.6e} "
                          f"vol={vol:.4e} neighbors={n_nbrs}/4")

    # --- 3. Spatial distribution of lost particles ---
    print(f"\n  3. Spatial distribution of ALL {n_lost:,} lost particles:")
    for axis, name in enumerate(['X', 'Y', 'Z']):
        coords = lost_positions[:, axis]
        print(f"     {name}: [{coords.min():.8f}, {coords.max():.8f}] "
              f"mean={coords.mean():.8f} std={coords.std():.8f}")
        # Domain fraction
        frac_min = (coords.min() - domain_min[axis]) / (domain_max[axis] - domain_min[axis])
        frac_max = (coords.max() - domain_min[axis]) / (domain_max[axis] - domain_min[axis])
        print(f"          domain fraction: [{frac_min:.4f}, {frac_max:.4f}]")

    # Histogram in X (the primary flow direction)
    n_bins = 20
    x_coords = lost_positions[:, 0]
    hist, bin_edges = np.histogram(x_coords, bins=n_bins)
    print(f"\n     X-distribution histogram ({n_bins} bins):")
    max_count = max(hist) if max(hist) > 0 else 1
    for i in range(n_bins):
        bar_len = int(40 * hist[i] / max_count)
        x_frac = (bin_edges[i] - domain_min[0]) / (domain_max[0] - domain_min[0])
        print(f"       [{x_frac:.2f}] {hist[i]:5d} {'#' * bar_len}")

    # --- 4. Check all lost particles against bbox (full set, not sampled) ---
    all_inside_bbox = np.all(
        (lost_positions >= domain_min) & (lost_positions <= domain_max),
        axis=1
    )
    n_all_inside = int(np.sum(all_inside_bbox))
    n_all_outside = n_lost - n_all_inside
    print(f"\n  4. Full bounding box check (all {n_lost:,} lost particles):")
    print(f"     Inside bbox:  {n_all_inside:,} ({100*n_all_inside/n_lost:.1f}%)")
    print(f"     Outside bbox: {n_all_outside:,} ({100*n_all_outside/n_lost:.1f}%)")

    return {
        'n_inside_bbox': n_all_inside,
        'n_outside_bbox': n_all_outside,
        'bf_found': bf_found if n_to_bruteforce > 0 else 0,
        'bf_not_found': bf_not_found if n_to_bruteforce > 0 else 0,
    }


def _check_element_in_octree(elem_id, node_positions, connectivity,
                             octree_gpu):
    """Check if element is registered in octree cells and at which levels."""
    from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import (
        find_axis_aligned_edges_single, encode_morton_3d_single
    )

    nodes = connectivity[elem_id]
    verts = node_positions[nodes]

    # Find cell size from axis-aligned edges
    cell_size, level = find_axis_aligned_edges_single(verts, tolerance=1e-6)
    if np.any(cell_size == 0):
        print(f"             -> NOT a Kuhn tet (no axis-aligned edges)")
        return

    # Compute vertex cells
    vertex_cells = set()
    for v in verts:
        i = int(np.floor(v[0] / cell_size[0]))
        j = int(np.floor(v[1] / cell_size[1]))
        k = int(np.floor(v[2] / cell_size[2]))
        vertex_cells.add((i, j, k))

    print(f"             -> Kuhn level={level} cell_size={cell_size} "
          f"spans {len(vertex_cells)} cells: {vertex_cells}")

    # Check octree level_cell_sizes
    level_cell_sizes = np.array(octree_gpu.level_cell_sizes)
    octree_cell_size = level_cell_sizes[level]
    cell_size_match = np.allclose(cell_size, octree_cell_size, rtol=1e-4)
    print(f"             -> Octree cell_size at level {level}: {octree_cell_size} "
          f"{'(MATCH)' if cell_size_match else '*** MISMATCH ***'}")


# =============================================================================
# Load comparison stats
# =============================================================================

def load_comparison_stats(csv_path, max_steps=None):
    """Load search_stats.csv from a previous run for comparison."""
    if not csv_path.exists():
        return None
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            row = {}
            for k, v in r.items():
                try:
                    row[k] = int(v) if '.' not in v else float(v)
                except (ValueError, TypeError):
                    row[k] = v
            rows.append(row)
            if max_steps and len(rows) >= max_steps:
                break
    return rows


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("RK4 Diagnostic Benchmark")
    print("Isolating particle loss root cause in vmapped RK4 pipeline")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print()
    print("Configuration:")
    print(f"  L0 (cached element):  {'ENABLED' if ENABLE_L0_SEARCH else 'DISABLED'}")
    print(f"  L1 (neighbor hops):   {'ENABLED' if ENABLE_L1_SEARCH else 'DISABLED'}")
    print(f"  L2 method:            {L2_METHOD}")
    print(f"  N_STEPS:              {N_STEPS}")
    print(f"  DT:                   {DT}")
    print(f"  Export frequency:     every {EXPORT_FREQUENCY} steps")
    if LOAD_FROM_PREVIOUS['enabled']:
        print(f"  Loading from:         {LOAD_FROM_PREVIOUS['output_dir']}")
        print(f"  Starting step:        {LOAD_FROM_PREVIOUS['step']}")
    if POSITION_FILTER['enabled']:
        mode = POSITION_FILTER.get('mode', 'fraction')
        print(f"  Position filter ({mode}): "
              f"X={POSITION_FILTER.get('x')}, "
              f"Y={POSITION_FILTER.get('y')}, "
              f"Z={POSITION_FILTER.get('z')}")
    print(f"  Standalone L2 verify: {VERIFY_LOST_WITH_STANDALONE_L2}")
    print("=" * 80)

    # ==================================================================
    # 1. Load mesh
    # ==================================================================
    print("\n[1/7] Loading mesh...")
    config.POINT_IN_TET_METHOD = POINT_IN_TET_METHOD

    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=False
    )

    n_nodes_orig = node_positions.shape[0]
    n_elements = connectivity.shape[0]
    print(f"  Elements: {n_elements:,}, Nodes: {n_nodes_orig:,}")

    print("  Deduplicating...")
    node_positions, connectivity, n_dup, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    connectivity = connectivity.astype(np.int32)
    n_nodes = node_positions.shape[0]
    print(f"  Removed {n_dup:,} duplicates -> {n_nodes:,} nodes")

    # Compute mesh bounding box for sub-step clamping
    mesh_bbox_min_cpu = node_positions.min(axis=0).astype(np.float32)
    mesh_bbox_max_cpu = node_positions.max(axis=0).astype(np.float32)
    print(f"  Mesh bbox: [{mesh_bbox_min_cpu}] → [{mesh_bbox_max_cpu}]")

    # ==================================================================
    # 2. Precompute metadata
    # ==================================================================
    print("\n[2/7] Precomputing metadata...")
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
    # 3. Build octree and upload to GPU
    # ==================================================================
    print("\n[3/7] Building structures and uploading to GPU...")
    octree_struct = build_global_morton_octree(
        node_positions=node_positions,
        connectivity=connectivity,
        leaf_capacity=256, max_depth=21, verbose=False
    )

    # Multi-cell vertex registration octree
    mesh_octree_cells_multi = extract_octree_cells_vertex_multi(
        node_positions, connectivity, tolerance=1e-6, verbose=False
    )
    print(f"  Multi-cell octree: {mesh_octree_cells_multi.n_cells:,} cells, "
          f"{mesh_octree_cells_multi.elements_per_cell_mean:.1f} elem/cell, "
          f"{mesh_octree_cells_multi.cells_per_element_mean:.1f} cells/elem")

    element_neighbors = build_element_neighbors_array(connectivity, method='face', verbose=False)

    # --- Mesh quality analysis ---
    mesh_quality = analyze_mesh_quality(
        node_positions, connectivity, element_volumes, det_values,
        element_neighbors, verbose=True
    )

    mesh_gpu = upload_mesh_to_gpu(connectivity, node_positions, element_neighbors, verbose=False)
    mesh_gpu_octree = upload_global_morton_to_gpu(octree_struct, connectivity, node_positions)

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
    element_volumes_gpu = jax.device_put(element_volumes.astype(np.float32))

    set_corrected_metadata(aa_metadata_gpu, element_vertices_gpu)
    set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)

    velocity_sequence_gpu = jax.device_put(velocity_sequence)

    mesh_bbox_min_gpu = jax.device_put(mesh_bbox_min_cpu)
    mesh_bbox_max_gpu = jax.device_put(mesh_bbox_max_cpu)
    print("  Uploaded to GPU")

    # ==================================================================
    # 4. Generate or load particles
    # ==================================================================
    domain_min = node_positions.min(axis=0)
    domain_max = node_positions.max(axis=0)
    domain_size = domain_max - domain_min

    if LOAD_FROM_PREVIOUS['enabled']:
        print(f"\n[4/7] Loading particles from previous output...")
        step_num = LOAD_FROM_PREVIOUS['step']
        vtu_path = LOAD_FROM_PREVIOUS['output_dir'] / f"particles_step_{step_num:06d}.vtu"
        print(f"  File: {vtu_path}")

        vtu_data = load_vtu(str(vtu_path))
        particle_positions = vtu_data['positions']
        particle_ids = vtu_data['particle_ids']
        loaded_element_ids = vtu_data['element_ids']

        # NOTE: VTU export only contains ACTIVE particles, so all loaded
        # particles should have valid element_ids (>= 0).
        n_loaded = len(particle_positions)
        n_active_loaded = int(np.sum(loaded_element_ids >= 0))
        print(f"  Loaded {n_loaded:,} particles ({n_active_loaded:,} active)")

    else:
        print(f"\n[4/7] Generating particles...")
        nx, ny, nz = PARTICLE_GRID_RESOLUTION

        par_bounds_min = np.zeros(3, dtype=np.float32)
        par_bounds_max = np.zeros(3, dtype=np.float32)
        for i, axis in enumerate(['x', 'y', 'z']):
            mn, mx = PARTICLE_BOUNDS_FRACTION[axis]
            par_bounds_min[i] = domain_min[i] + mn * domain_size[i]
            par_bounds_max[i] = domain_min[i] + mx * domain_size[i]

        particle_positions = uniform_grid_seeds(
            resolution=(nx, ny, nz),
            bounds=[par_bounds_min, par_bounds_max],
            include_boundaries=True
        )

        # Clip to mesh bounds
        margin = 0.01
        bbox_min_safe = domain_min + margin * domain_size
        bbox_max_safe = domain_max - margin * domain_size
        particle_positions = np.clip(particle_positions, bbox_min_safe, bbox_max_safe)

        particle_ids = np.arange(len(particle_positions), dtype=np.int32)
        loaded_element_ids = None

        print(f"  Generated {len(particle_positions):,} particles ({nx}x{ny}x{nz})")
        print(f"    X: [{par_bounds_min[0]:.6f}, {par_bounds_max[0]:.6f}]")
        print(f"    Y: [{par_bounds_min[1]:.6f}, {par_bounds_max[1]:.6f}]")
        print(f"    Z: [{par_bounds_min[2]:.6f}, {par_bounds_max[2]:.6f}]")

    # --- Apply positional filter ---
    if POSITION_FILTER['enabled']:
        print(f"\n  Applying positional filter (mode='{POSITION_FILTER['mode']}')...")
        filter_mode = POSITION_FILTER.get('mode', 'fraction')
        fmin = np.full(3, -np.inf, dtype=np.float32)
        fmax = np.full(3,  np.inf, dtype=np.float32)

        for i, axis in enumerate(['x', 'y', 'z']):
            bounds = POSITION_FILTER.get(axis)
            if bounds is None:
                continue  # No filter on this axis
            mn, mx = bounds
            if filter_mode == 'absolute':
                fmin[i] = float(mn)
                fmax[i] = float(mx)
            else:  # 'fraction'
                fmin[i] = domain_min[i] + float(mn) * domain_size[i]
                fmax[i] = domain_min[i] + float(mx) * domain_size[i]

        mask = (
            (particle_positions[:, 0] >= fmin[0]) & (particle_positions[:, 0] <= fmax[0]) &
            (particle_positions[:, 1] >= fmin[1]) & (particle_positions[:, 1] <= fmax[1]) &
            (particle_positions[:, 2] >= fmin[2]) & (particle_positions[:, 2] <= fmax[2])
        )
        n_before = len(particle_positions)
        particle_positions = particle_positions[mask]
        particle_ids = particle_ids[mask]
        if loaded_element_ids is not None:
            loaded_element_ids = loaded_element_ids[mask]

        # Report resolved bounds
        eff_fmin = np.where(np.isfinite(fmin), fmin, particle_positions.min(axis=0))
        eff_fmax = np.where(np.isfinite(fmax), fmax, particle_positions.max(axis=0))
        print(f"    X=[{eff_fmin[0]:.6f}, {eff_fmax[0]:.6f}], "
              f"Y=[{eff_fmin[1]:.6f}, {eff_fmax[1]:.6f}], "
              f"Z=[{eff_fmin[2]:.6f}, {eff_fmax[2]:.6f}]")
        print(f"    {n_before:,} -> {len(particle_positions):,} particles "
              f"({100*len(particle_positions)/n_before:.1f}%)")

    n_particles = len(particle_positions)
    positions_gpu = jax.device_put(particle_positions)

    # ==================================================================
    # 5. Initial assignment
    # ==================================================================
    print(f"\n[5/7] Initial assignment...")

    if loaded_element_ids is not None and np.all(loaded_element_ids >= 0):
        # Use element IDs from loaded VTU (they were active at the loaded step)
        print(f"  Using element IDs from loaded VTU file")
        # Verify with standalone L2
        print(f"  Verifying with standalone L2...")
        sa_elem_ids, sa_n_tests = standalone_l2_search_batch(
            positions_gpu, mesh_aligned_octree_multi_gpu, STANDALONE_L2_BATCH_SIZE
        )
        sa_found = int(np.sum(sa_elem_ids >= 0))
        print(f"    Standalone L2: {sa_found:,}/{n_particles:,} ({100*sa_found/n_particles:.2f}%)")
        # Use loaded element IDs (from cascade_radius, which may be more accurate)
        element_ids_initial = jax.device_put(loaded_element_ids)
    else:
        # Run cascade_radius initial assignment
        print(f"  Running cascade_radius initial assignment...")
        element_ids_initial = initial_assignment_cascading_fallback(
            positions_gpu, mesh_gpu_octree,
            initial_radius=500, fallback_radii=[1000, 2000, 5000, 10000, 100000],
            verbose=True
        )
        element_ids_initial = jax.block_until_ready(element_ids_initial)

    n_assigned = int(jnp.sum(element_ids_initial >= 0))
    print(f"  Assigned: {n_assigned:,}/{n_particles:,} ({100*n_assigned/n_particles:.2f}%)")

    # ==================================================================
    # 6. Build RK4 functions and run diagnostic tracking
    # ==================================================================
    print(f"\n[6/7] Building diagnostic RK4 (L0={'ON' if ENABLE_L0_SEARCH else 'OFF'}, "
          f"L1={'ON' if ENABLE_L1_SEARCH else 'OFF'})...")

    rk4_step, rk4_step_with_substep_stats = create_rk4_diagnostic(
        mesh_gpu_connectivity=mesh_gpu.connectivity,
        mesh_gpu_node_positions=mesh_gpu.node_positions,
        mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
        mesh_gpu_element_volumes=element_volumes_gpu,
        mesh_gpu_global_morton=mesh_gpu_octree,
        mesh_aligned_octree=mesh_aligned_octree_multi_gpu,
        enable_l0=ENABLE_L0_SEARCH,
        enable_l1=ENABLE_L1_SEARCH,
        n_hops=N_HOPS,
        mesh_bbox_min=mesh_bbox_min_gpu,
        mesh_bbox_max=mesh_bbox_max_gpu,
        use_bbox_clamp=RK4_SUBSTEP_BBOX_CLAMP,
        use_last_valid_vel=RK4_SUBSTEP_LAST_VALID_VEL,
    )

    # Setup export
    run_label = f"L0={'ON' if ENABLE_L0_SEARCH else 'OFF'}_L1={'ON' if ENABLE_L1_SEARCH else 'OFF'}"
    output_subdir = OUTPUT_DIR / run_label
    exporter = VTKExportThread(output_subdir)
    exporter.start()

    # Export initial state
    pos_cpu = np.array(positions_gpu, dtype=np.float32)
    eid_cpu = np.array(element_ids_initial, dtype=np.int32)
    exporter.enqueue_export(0, pos_cpu, eid_cpu, particle_ids)
    print(f"  Exported initial state (step 0): {n_assigned:,} active")

    # Setup stats CSV
    stats_csv_path = output_subdir / "search_stats.csv"
    stats_csv_path.parent.mkdir(parents=True, exist_ok=True)
    stats_csv = open(stats_csv_path, 'w')
    stats_csv.write(
        "step,n_active,n_lost,new_lost,"
        "l0_hits,l1_hits,l2_hits,misses,"
        "l0_pct,l1_pct,l2_pct,miss_pct,"
        "k1_miss,k2_miss,k3_miss,k4_miss,final_miss,"
        "sa_found,sa_missed,"
        "bf_found,bf_not_found\n"
    )

    # Load comparison stats
    comparison_stats = None
    comparison_offset = LOAD_FROM_PREVIOUS['step'] if LOAD_FROM_PREVIOUS['enabled'] else 0
    if COMPARE_STATS_CSV.exists():
        comparison_stats = load_comparison_stats(COMPARE_STATS_CSV,
                                                 max_steps=comparison_offset + N_STEPS)
        print(f"  Loaded comparison stats: {len(comparison_stats)} steps from {COMPARE_STATS_CSV}")
        if comparison_offset > 0:
            print(f"  Comparison offset: {comparison_offset} (comparing absolute steps {comparison_offset+1}-{comparison_offset+N_STEPS})")

    # Warmup / compile
    print(f"  Compiling...")
    t_compile = time.time()
    positions_gpu, element_ids_gpu = rk4_step(
        positions_gpu, element_ids_initial, DT, velocity_sequence_gpu, 0
    )
    positions_gpu = jax.block_until_ready(positions_gpu)
    element_ids_gpu = jax.block_until_ready(element_ids_gpu)
    t_compile = time.time() - t_compile
    print(f"  Compilation: {t_compile:.1f}s")

    # Warmup stats version too
    _warmup = rk4_step_with_substep_stats(
        positions_gpu, element_ids_gpu, DT, velocity_sequence_gpu, 0
    )
    jax.block_until_ready(_warmup[0])

    # ==================================================================
    # 7. Run tracking loop with diagnostics
    # ==================================================================
    print(f"\n[7/7] Running {N_STEPS} RK4 steps with diagnostics...")
    print("=" * 80)

    prev_lost = n_particles - int(jnp.sum(element_ids_gpu >= 0))
    prev_eid_cpu = np.array(element_ids_gpu, dtype=np.int32)
    t_start = time.time()

    for step in range(1, N_STEPS + 1):
        time_idx = step - 1

        # Run RK4 with sub-step stats
        positions_gpu, element_ids_gpu, all_hit_levels = rk4_step_with_substep_stats(
            positions_gpu, element_ids_gpu, DT, velocity_sequence_gpu, time_idx
        )
        positions_gpu = jax.block_until_ready(positions_gpu)
        element_ids_gpu = jax.block_until_ready(element_ids_gpu)

        # all_hit_levels: (N, 5) int8 — hit levels for k1, k2, k3, k4, final
        hit_levels_np = np.array(all_hit_levels, dtype=np.int8)
        eid_cpu_now = np.array(element_ids_gpu, dtype=np.int32)

        # Aggregate stats
        l0_hits = int(np.sum(hit_levels_np == 0))
        l1_hits = int(np.sum(hit_levels_np == 1))
        l2_hits = int(np.sum(hit_levels_np == 2))
        misses  = int(np.sum(hit_levels_np == -1))
        total = l0_hits + l1_hits + l2_hits + misses

        n_active = int(np.sum(eid_cpu_now >= 0))
        n_lost = n_particles - n_active
        new_lost = n_lost - prev_lost

        # Per-sub-step miss counts
        k1_miss = int(np.sum(hit_levels_np[:, 0] == -1))
        k2_miss = int(np.sum(hit_levels_np[:, 1] == -1))
        k3_miss = int(np.sum(hit_levels_np[:, 2] == -1))
        k4_miss = int(np.sum(hit_levels_np[:, 3] == -1))
        final_miss = int(np.sum(hit_levels_np[:, 4] == -1))

        # Percentages
        l0p = 100 * l0_hits / total if total > 0 else 0
        l1p = 100 * l1_hits / total if total > 0 else 0
        l2p = 100 * l2_hits / total if total > 0 else 0
        mp  = 100 * misses  / total if total > 0 else 0

        # --- Standalone L2 verification of NEWLY lost particles ---
        # Identify particles that were active last step but lost this step
        newly_lost_mask = (prev_eid_cpu >= 0) & (eid_cpu_now < 0)
        n_newly_lost = int(np.sum(newly_lost_mask))
        pos_cpu_now = None  # Will be fetched if needed

        sa_found, sa_missed = 0, 0
        if VERIFY_LOST_WITH_STANDALONE_L2 and n_newly_lost > 0:
            # Get final positions of newly-lost particles
            pos_cpu_now = np.array(positions_gpu, dtype=np.float32)
            newly_lost_positions = pos_cpu_now[newly_lost_mask]
            newly_lost_ids = particle_ids[newly_lost_mask] if particle_ids is not None else None

            if len(newly_lost_positions) > 0:
                newly_lost_gpu = jax.device_put(newly_lost_positions)
                sa_eids, sa_ntests = standalone_l2_search_batch(
                    newly_lost_gpu, mesh_aligned_octree_multi_gpu,
                    STANDALONE_L2_BATCH_SIZE
                )
                sa_found = int(np.sum(sa_eids >= 0))
                sa_missed = int(np.sum(sa_eids < 0))

        # --- Brute-force CPU point-in-tet on newly-lost particles ---
        bf_found_step, bf_not_found_step = 0, 0
        if BRUTEFORCE_MAX_PER_STEP > 0 and n_newly_lost > 0:
            if pos_cpu_now is None:
                pos_cpu_now = np.array(positions_gpu, dtype=np.float32)
            newly_lost_pos = pos_cpu_now[newly_lost_mask]
            n_bf = min(BRUTEFORCE_MAX_PER_STEP, len(newly_lost_pos))
            tolerance_bf = 1e-6

            # Use float64 for precision
            M_inv_f64 = M_inv_array.astype(np.float64)
            p0_f64 = p0_array.astype(np.float64)

            for pi in range(n_bf):
                pos = newly_lost_pos[pi].astype(np.float64)
                local = pos - p0_f64
                bary = np.einsum('eij,ej->ei', M_inv_f64, local)
                bary4 = 1.0 - bary.sum(axis=1)
                inside_mask_bf = (
                    (bary[:, 0] >= -tolerance_bf) &
                    (bary[:, 1] >= -tolerance_bf) &
                    (bary[:, 2] >= -tolerance_bf) &
                    (bary4 >= -tolerance_bf)
                )
                containing = np.where(inside_mask_bf)[0]
                if len(containing) > 0:
                    bf_found_step += 1
                    # Report immediately for first few occurrences
                    if bf_found_step <= 3:
                        prev_elem = prev_eid_cpu[newly_lost_mask][pi]
                        print(f"    *** BF-FOUND step {step}: particle at "
                              f"({pos[0]:.8f},{pos[1]:.8f},{pos[2]:.8f}) "
                              f"is in elem(s) {containing[:5]} "
                              f"(prev_elem={prev_elem}) ***")
                        for cid in containing[:3]:
                            vol = element_volumes[cid]
                            n_nbrs = np.sum(element_neighbors[cid] >= 0)
                            print(f"       elem {cid}: vol={vol:.4e} "
                                  f"nbrs={n_nbrs}/4 "
                                  f"det={det_values[cid]:.4e}")
                else:
                    bf_not_found_step += 1

        # Write stats
        stats_csv.write(
            f"{step},{n_active},{n_lost},{new_lost},"
            f"{l0_hits},{l1_hits},{l2_hits},{misses},"
            f"{l0p:.2f},{l1p:.2f},{l2p:.2f},{mp:.2f},"
            f"{k1_miss},{k2_miss},{k3_miss},{k4_miss},{final_miss},"
            f"{sa_found},{sa_missed},"
            f"{bf_found_step},{bf_not_found_step}\n"
        )
        stats_csv.flush()

        # Log
        if step % LOG_INTERVAL == 0 or step <= 5 or new_lost > 0:
            log_parts = [
                f"Step {step:4d}: active={n_active:,} lost={n_lost:,} (+{new_lost})",
                f"L0={l0p:.1f}% L1={l1p:.1f}% L2={l2p:.1f}% miss={mp:.1f}%",
            ]
            if new_lost > 0:
                log_parts.append(
                    f"sub-step misses: k1={k1_miss} k2={k2_miss} k3={k3_miss} "
                    f"k4={k4_miss} final={final_miss}"
                )
            if sa_found > 0 or (sa_missed > 0 and n_newly_lost > 0):
                log_parts.append(
                    f"SA-L2: {sa_found}/{n_newly_lost} newly-lost found by standalone"
                )
            if bf_found_step > 0:
                log_parts.append(
                    f"BF: {bf_found_step}/{bf_found_step+bf_not_found_step} "
                    f"found by brute-force!"
                )
            # Comparison with previous run (offset by starting step)
            comp_idx = comparison_offset + step - 1
            if comparison_stats and comp_idx < len(comparison_stats):
                comp = comparison_stats[comp_idx]
                comp_lost = comp['n_lost']
                diff = n_lost - comp_lost
                log_parts.append(f"(prev: lost={comp_lost:,}, diff={diff:+d})")

            print("  " + " | ".join(log_parts))

        # Export
        if step % EXPORT_FREQUENCY == 0 or step == N_STEPS or new_lost > 0:
            pos_cpu = np.array(positions_gpu, dtype=np.float32)
            eid_cpu = np.array(element_ids_gpu, dtype=np.int32)
            exporter.enqueue_export(step, pos_cpu, eid_cpu, particle_ids)

        prev_lost = n_lost
        prev_eid_cpu = eid_cpu_now.copy()

    t_elapsed = time.time() - t_start
    stats_csv.close()

    # Wait for exports
    exporter.stop()

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    n_active_final = int(jnp.sum(element_ids_gpu >= 0))
    n_lost_final = n_particles - n_active_final
    print(f"  Particles: {n_particles:,}")
    print(f"  Final active: {n_active_final:,} ({100*n_active_final/n_particles:.2f}%)")
    print(f"  Final lost: {n_lost_final:,} ({100*n_lost_final/n_particles:.2f}%)")
    print(f"  Steps: {N_STEPS}")
    print(f"  Time: {t_elapsed:.1f}s ({n_particles * N_STEPS / t_elapsed:,.0f} p*step/s)")
    print(f"  L0: {'ON' if ENABLE_L0_SEARCH else 'OFF'}")
    print(f"  L1: {'ON' if ENABLE_L1_SEARCH else 'OFF'}")
    print(f"  Output: {output_subdir}")
    print(f"  Stats: {stats_csv_path}")

    if comparison_stats:
        comp_idx = comparison_offset + N_STEPS - 1
        if comp_idx < len(comparison_stats):
            comp_lost = comparison_stats[comp_idx]['n_lost']
            abs_step = comparison_offset + N_STEPS
            print(f"\n  Comparison with previous run (absolute step {abs_step}):")
            print(f"    Previous lost at step {abs_step}: {comp_lost:,}")
            print(f"    This run lost at step {N_STEPS}: {n_lost_final:,}")
            print(f"    Difference: {n_lost_final - comp_lost:+d}")

    # Final standalone L2 verification on ALL lost particles
    if n_lost_final > 0:
        print(f"\n  Final standalone L2 on all {n_lost_final:,} lost particles...")
        eid_final_cpu = np.array(element_ids_gpu, dtype=np.int32)
        lost_mask = eid_final_cpu < 0
        pos_final_cpu = np.array(positions_gpu, dtype=np.float32)
        lost_positions = pos_final_cpu[lost_mask]
        lost_positions_gpu = jax.device_put(lost_positions)

        sa_eids, sa_n_tests = standalone_l2_search_batch(
            lost_positions_gpu, mesh_aligned_octree_multi_gpu, STANDALONE_L2_BATCH_SIZE
        )
        sa_found = int(np.sum(sa_eids >= 0))
        sa_missed = int(np.sum(sa_eids < 0))
        print(f"    Standalone L2 found: {sa_found:,}/{n_lost_final:,} "
              f"({100*sa_found/n_lost_final:.2f}%)")
        print(f"    Standalone L2 missed: {sa_missed:,}/{n_lost_final:,} "
              f"({100*sa_missed/n_lost_final:.2f}%)")
        if sa_found > 0:
            print(f"    *** {sa_found:,} particles are findable by standalone L2 "
                  f"but lost in RK4! ***")
            print(f"    This confirms the loss is due to the vmapped RK4 graph, "
                  f"not L2 search itself.")
        if sa_missed > 0:
            print(f"    {sa_missed:,} particles are genuinely outside the mesh "
                  f"at their final positions.")

    # ==================================================================
    # Deep analysis of lost particles
    # ==================================================================
    if n_lost_final > 0:
        print("\n" + "=" * 80)
        print("LOST PARTICLE DEEP ANALYSIS")
        print("=" * 80)

        lost_analysis = analyze_lost_particles(
            lost_positions=lost_positions,
            node_positions=node_positions,
            connectivity=connectivity,
            element_volumes=element_volumes,
            det_values=det_values,
            element_neighbors=element_neighbors,
            mesh_aligned_octree_multi_gpu=mesh_aligned_octree_multi_gpu,
            M_inv_array=M_inv_array.astype(np.float64),
            p0_array=p0_array.astype(np.float64),
            max_analyze=500,
            verbose=True,
        )

        # --- Also analyze EARLY lost particles (from first loss event) ---
        # Check the exported VTU for the first step with losses to analyze
        # particles at their moment of loss (not after 2500 steps of drift)
        print("\n  Note: The above analysis is on FINAL positions of lost particles.")
        print("  Particles may have drifted significantly after being marked lost.")
        print("  For root cause analysis, examine particles at their MOMENT of loss")
        print("  using the per-step SA-L2 data in the search_stats.csv file.")

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
