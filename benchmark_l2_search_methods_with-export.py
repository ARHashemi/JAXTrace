#!/usr/bin/env python3
"""
Comprehensive L2 Search Methods Benchmark

Compares different L2 search strategies with FAIR comparison metrics:
- Fixed computation budget (same search radius or equivalent work)
- Accuracy (retention/success rate)
- Performance (throughput)

L2 Search Methods:
1. 'radius' (radius=10): Fixed radius search (baseline)
2. 'radius' (radius=30): Fixed large radius (max coverage)
3. 'incremental' (2,4,8,15,30): 5-tier cascading (PRODUCTION CONFIG)
4. 'incremental' (2,5,10): 3-tier cascading (simpler alternative)
5. 'neighbors': Morton neighbor arithmetic
6. 'hierarchical': Multi-depth conditional search
7. 'mesh_aligned_octree': Mesh-aligned octree search (NEW - Kuhn meshes only)

Fair Comparison Approaches:
A) Equal Maximum Coverage: All methods search up to radius=30
B) Equal Average Work: Tune radii to match ~20 leaves average
C) Production Realistic: Use actual production configuration

Metrics:
- Initial assignment success rate
- RK4 retention at step 100
- Throughput (particles/second)
- Average leaves searched (efficiency metric)
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF/XLA warnings
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # Explicitly set platform order

import warnings
# warnings.filterwarnings('ignore', category=DeprecationWarning)
# warnings.filterwarnings('ignore', message='.*PJRT_Api.*')
# warnings.filterwarnings('ignore', message='.*cudart_stub.*')

import sys
import time
import queue
import threading
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
from jaxtrace.gpu.search.mesh_aligned_morton_builder import build_mesh_aligned_morton_structure
from jaxtrace.gpu.search.mesh_aligned_morton_search import upload_mesh_aligned_morton_to_gpu
from jaxtrace.gpu.tracking.initial_assignment_cascading import (
    initial_assignment_cascading_fallback,
    initial_assignment_mesh_aligned_multi_local,
)
from jaxtrace.tracking.seeding import uniform_grid_seeds
from jaxtrace.gpu.search.aa_detection import precompute_aa_metadata, precompute_element_vertices
from jaxtrace.gpu.search.point_in_tet_methods import set_corrected_metadata, set_inverse_matrices_gpu
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
from jaxtrace.gpu.tracking.rk4_fully_fused_timedep import (
    create_rk4_fully_fused_timedep,
    create_rk4_fully_fused_timedep_with_stats,
)
import jaxtrace.config as config

# Import VTK export (same as production)
try:
    from jaxtrace.io import VTKTrajectoryWriter
    import vtk
    HAS_VTK = True
except ImportError:
    HAS_VTK = False


def write_vtu_simple(filename: str, positions: np.ndarray, velocities: np.ndarray = None,
                     particle_ids: np.ndarray = None, element_ids: np.ndarray = None):
    """
    Write VTU file without requiring vtk package.

    Simple XML-based VTK Unstructured Grid writer for particle data.
    Compatible with ParaView and VisIt.
    """
    import base64
    import struct

    n_points = len(positions)

    # Build VTU XML
    xml_lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
        '  <UnstructuredGrid>',
        f'    <Piece NumberOfPoints="{n_points}" NumberOfCells="{n_points}">',
        '      <Points>',
        '        <DataArray type="Float32" NumberOfComponents="3" format="ascii">',
    ]

    # Write positions (ASCII for simplicity)
    for pos in positions:
        xml_lines.append(f'          {pos[0]:.6e} {pos[1]:.6e} {pos[2]:.6e}')

    xml_lines.extend([
        '        </DataArray>',
        '      </Points>',
        '      <Cells>',
        '        <DataArray type="Int32" Name="connectivity" format="ascii">',
    ])

    # Connectivity (each particle is a vertex cell)
    xml_lines.append('          ' + ' '.join(str(i) for i in range(n_points)))

    xml_lines.extend([
        '        </DataArray>',
        '        <DataArray type="Int32" Name="offsets" format="ascii">',
    ])

    # Offsets (each vertex cell has 1 point)
    xml_lines.append('          ' + ' '.join(str(i + 1) for i in range(n_points)))

    xml_lines.extend([
        '        </DataArray>',
        '        <DataArray type="UInt8" Name="types" format="ascii">',
    ])

    # Cell types (1 = VTK_VERTEX)
    xml_lines.append('          ' + ' '.join('1' for _ in range(n_points)))

    xml_lines.extend([
        '        </DataArray>',
        '      </Cells>',
    ])

    # Add PointData (particle IDs, element IDs, velocities)
    has_pointdata = (velocities is not None or particle_ids is not None or element_ids is not None)
    if has_pointdata:
        # Build PointData attributes
        pd_attrs = []
        if velocities is not None:
            pd_attrs.append('Vectors="Velocity"')
        if particle_ids is not None:
            pd_attrs.append('Scalars="ParticleID"')
        xml_lines.append(f'      <PointData {" ".join(pd_attrs)}>')

        if particle_ids is not None:
            xml_lines.append('        <DataArray type="Int32" Name="ParticleID" format="ascii">')
            xml_lines.append('          ' + ' '.join(str(int(pid)) for pid in particle_ids))
            xml_lines.append('        </DataArray>')

        if element_ids is not None:
            xml_lines.append('        <DataArray type="Int32" Name="ElementID" format="ascii">')
            xml_lines.append('          ' + ' '.join(str(int(eid)) for eid in element_ids))
            xml_lines.append('        </DataArray>')

        if velocities is not None:
            xml_lines.append('        <DataArray type="Float32" Name="Velocity" NumberOfComponents="3" format="ascii">')
            for vel in velocities:
                xml_lines.append(f'          {vel[0]:.6e} {vel[1]:.6e} {vel[2]:.6e}')
            xml_lines.append('        </DataArray>')

        xml_lines.append('      </PointData>')

    xml_lines.extend([
        '    </Piece>',
        '  </UnstructuredGrid>',
        '</VTKFile>',
    ])

    # Write to file
    with open(filename, 'w') as f:
        f.write('\n'.join(xml_lines))
        f.write('\n')


# ============================================================================
# Configuration
# ============================================================================

MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)  # 20 timesteps for RK4 testing
VELOCITY_FIELD_NAME = 'Displacement'

# ============================================================================
# Particle Seeding Configuration
# ============================================================================
# Choose particle seeding strategy:
#   'uniform_grid': Uniform grid seeding (same as production)
#                   - Regular grid with configurable resolution and bounds
#                   - Particles may start outside elements (requires robust initial assignment)
#                   - Realistic for production scenarios (inflow, injection, etc.)
#
#   'centroids':    Perturbed element centroids
#                   - Particles start inside elements (guaranteed initial assignment)
#                   - Better for testing search methods (no initial assignment failures)
#                   - Uses same grid resolution to determine particle count
#
PARTICLE_SEEDING = 'uniform_grid'  # 'uniform_grid' or 'centroids'

# Particle grid resolution (used for both seeding strategies)
PARTICLE_GRID_RESOLUTION = (60, 90, 60)  # 324,000 particles
PARTICLE_BOUNDS_FRACTION = {
    'x': (0.12, 0.22),  # Use middle 40% of domain in X
    'y': (0.2, 0.8),  # Use middle 60% of domain in Y
    'z': (0.01, 1.0),  # Use upper 70% of domain in Z
}
# ============================================================================

# RK4 integration
DT = 0.0025 # Timestep
N_STEPS = 2500  # Number of RK4 steps (reduced for faster benchmark)

# L1 configuration (consistent across all tests)
ENABLE_L1_SEARCH = True
N_HOPS = 5

# Point-in-tet method (use INVERSE for fair comparison - fastest validated)
POINT_IN_TET_METHOD = 'inverse'

# VTK export configuration
EXPORT_FREQUENCY = 10  # Export every 10 steps
OUTPUT_DIR = Path("output/benchmark_with_export")
LOG_INTERVAL = 10

SEED = 42

# Initial assignment method ('cascade_radius' or 'mesh_aligned_octree_multi_local')
config.INITIAL_ASSIGNMENT_METHOD = 'cascade_radius'
config.INITIAL_ASSIGNMENT_BATCH_SIZE = 50000

# ============================================================================
# VTK Export Thread Class
# ============================================================================

class VTKExportThread:
    """
    Background thread for VTK export (EXACT same format as production).

    Uses VTKTrajectoryWriter.write_particles_at_time() to write unstructured
    grid files with positions and velocities.
    """

    def __init__(self, output_dir: Path, method_name: str):
        self.output_dir = output_dir / method_name.replace(" ", "_").replace("/", "_")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.export_queue = queue.Queue(maxsize=5)  # Limit queue size to prevent memory issues
        self.worker_thread = threading.Thread(target=self._export_worker, daemon=True)
        self.stop_event = threading.Event()
        self.n_exported = 0
        self.export_times = []

    def start(self):
        """Start background export worker"""
        self.worker_thread.start()

    def enqueue_export(self, step: int, positions: np.ndarray, element_ids: np.ndarray,
                       particle_ids: np.ndarray = None):
        """
        Add export task to queue (non-blocking).

        Args:
            step: Timestep number
            positions: ALL particle positions (N, 3)
            element_ids: Element IDs (N,) - used to create active_mask
            particle_ids: Original particle indices (N,) - for tracking across timesteps
        """
        try:
            # Create active_mask (same as production)
            active_mask = np.array(element_ids >= 0, dtype=bool)

            # Put ALL data in queue (will filter inside worker)
            self.export_queue.put(
                (step, positions, None, element_ids, active_mask, particle_ids),
                timeout=10.0
            )
        except queue.Full:
            print(f"Warning: Export queue full at step {step}, skipping export")

    def _export_worker(self):
        """Background thread that processes export queue (EXACT production pattern)"""
        while not self.stop_event.is_set():
            try:
                # Wait for data with timeout to allow checking stop_event
                export_data = self.export_queue.get(timeout=1.0)

                if export_data is None:  # Sentinel value
                    break

                step, positions, velocities, element_ids, active_mask, particle_ids = export_data

                # Write VTK file
                t0 = time.perf_counter()
                output_file = self.output_dir / f"particles_step_{step:06d}.vtu"

                # Filter to active particles only
                active_positions = positions[active_mask]
                active_velocities = None  # No velocities in benchmark
                active_particle_ids = particle_ids[active_mask] if particle_ids is not None else None
                active_element_ids = element_ids[active_mask]

                # Use VTK writer directly (EXACT production code)
                if HAS_VTK:
                    from jaxtrace.io import VTKTrajectoryWriter
                    writer = VTKTrajectoryWriter()
                    writer.write_particles_at_time(
                        positions=active_positions,
                        velocities=active_velocities,
                        time=step,
                        filename=str(output_file),
                        format='xml',
                        particle_ids=active_particle_ids,
                        element_ids=active_element_ids
                    )
                else:
                    # Fallback: write VTU file using simple writer (no vtk package needed)
                    write_vtu_simple(
                        filename=str(output_file),
                        positions=active_positions,
                        velocities=active_velocities,
                        particle_ids=active_particle_ids,
                        element_ids=active_element_ids
                    )

                export_time = time.perf_counter() - t0
                self.export_times.append(export_time)
                self.n_exported += 1
                self.export_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Export error: {e}")

    def stop(self):
        """Stop background worker and wait for queue to finish"""
        # Signal worker to stop
        self.export_queue.put(None)
        self.stop_event.set()

        # Wait for worker to finish
        if self.worker_thread:
            self.worker_thread.join(timeout=30.0)

    def get_stats(self):
        """Get export statistics"""
        if not self.export_times:
            return {'n_exported': 0, 'mean_time': 0, 'total_time': 0, 'queue_size': 0}

        return {
            'n_exported': self.n_exported,
            'mean_time': np.mean(self.export_times),
            'total_time': np.sum(self.export_times),
            'queue_size': self.export_queue.qsize(),
        }


def run_initial_assignment(positions_gpu, mesh_gpu_octree, l2_method, l2_radius=None, incremental_radii=None,
                           mesh_aligned_octree_neighbors_gpu=None, mesh_aligned_octree_multi_gpu=None):
    """Run initial assignment with specified L2 method."""

    # Set configuration
    config.POINT_IN_TET_METHOD = POINT_IN_TET_METHOD

    # Check if user has requested mesh_aligned_octree_multi_local for initial assignment
    use_multi_local_ia = (
        config.INITIAL_ASSIGNMENT_METHOD == 'mesh_aligned_octree_multi_local'
        and mesh_aligned_octree_multi_gpu is not None
    )

    if use_multi_local_ia:
        # Use 3×3×3 multi-local search for all methods that involve the multi-cell octree
        t_start = time.time()
        element_ids = initial_assignment_mesh_aligned_multi_local(
            positions_gpu,
            mesh_aligned_octree_multi_gpu,
            batch_size=config.INITIAL_ASSIGNMENT_BATCH_SIZE,
            max_tests=600,
            verbose=False
        )
        element_ids = jax.block_until_ready(element_ids)
        t_elapsed = time.time() - t_start

    elif l2_method == 'mesh_aligned_neighbors':
        # Use mesh-aligned neighbor search for initial assignment
        from jaxtrace.gpu.search.mesh_aligned_search_with_neighbors import search_batch_with_precomputed_neighbors

        octree_to_use = mesh_aligned_octree_neighbors_gpu
        max_tests = 20

        t_start = time.time()
        element_ids, n_tests = search_batch_with_precomputed_neighbors(
            positions_gpu,
            octree_to_use,
            levels_to_try=(14, 13, 12),
            max_tests_per_cell=max_tests
        )
        element_ids = jax.block_until_ready(element_ids)
        t_elapsed = time.time() - t_start

    elif l2_method in ['radius', 'incremental', 'neighbors', 'hierarchical',
                       'mesh_aligned_octree', 'mesh_aligned_octree_multi',
                       'mesh_aligned_octree_multi_local', 'mesh_aligned_octree_multi_local_where',
                       'mesh_aligned_morton']:
        # Default: cascade radius fallback (works for all mesh types)
        initial_radius = 500
        fallback_radii = [1000, 2000, 5000, 10000, 100000]

        t_start = time.time()
        element_ids = initial_assignment_cascading_fallback(
            positions_gpu,
            mesh_gpu_octree,
            initial_radius=initial_radius,
            fallback_radii=fallback_radii,
            verbose=False
        )
        element_ids = jax.block_until_ready(element_ids)
        t_elapsed = time.time() - t_start

    else:
        raise ValueError(f"Unknown L2 method: {l2_method}")

    n_assigned = int(jnp.sum(element_ids >= 0))

    return element_ids, n_assigned, t_elapsed


def _build_rk4_functions(l2_method, mesh_gpu, mesh_gpu_octree, element_volumes_gpu,
                         l2_radius, incremental_radii,
                         mesh_aligned_octree_gpu, mesh_aligned_morton_gpu,
                         mesh_aligned_octree_neighbors_gpu, mesh_aligned_octree_multi_gpu):
    """Build (rk4_step, rk4_step_with_stats) for the given L2 method."""
    common = dict(
        mesh_gpu_connectivity=mesh_gpu.connectivity,
        mesh_gpu_node_positions=mesh_gpu.node_positions,
        mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
        mesh_gpu_element_volumes=element_volumes_gpu,
        mesh_gpu_global_morton=mesh_gpu_octree,
        n_hops=N_HOPS,
        enable_l1_search=ENABLE_L1_SEARCH,
    )

    if l2_method == 'radius':
        return create_rk4_fully_fused_timedep_with_stats(
            **common,
            l2_search_radius=l2_radius if l2_radius is not None else 10,
            l2_search_method='radius',
        )

    elif l2_method == 'incremental':
        return create_rk4_fully_fused_timedep_with_stats(
            **common,
            l2_search_method='incremental',
            l2_incremental_radii=incremental_radii if incremental_radii is not None else (2, 4, 8, 15, 30),
        )

    elif l2_method == 'neighbors':
        return create_rk4_fully_fused_timedep_with_stats(**common, l2_search_method='neighbors')

    elif l2_method == 'hierarchical':
        return create_rk4_fully_fused_timedep_with_stats(**common, l2_search_method='hierarchical')

    elif l2_method == 'mesh_aligned_octree':
        original = config.L2_SEARCH_METHOD
        config.L2_SEARCH_METHOD = 'mesh_aligned_octree'
        fns = create_rk4_fully_fused_timedep_with_stats(
            **common, l2_search_method='radius', mesh_aligned_octree=mesh_aligned_octree_gpu
        )
        config.L2_SEARCH_METHOD = original
        return fns

    elif l2_method == 'mesh_aligned_morton':
        original = config.L2_SEARCH_METHOD
        config.L2_SEARCH_METHOD = 'mesh_aligned_morton'
        fns = create_rk4_fully_fused_timedep_with_stats(
            **common,
            l2_search_radius=l2_radius if l2_radius is not None else 2,
            l2_incremental_radii=incremental_radii if incremental_radii is not None else (2, 5, 10),
            l2_search_method='incremental' if incremental_radii is not None else 'radius',
            mesh_aligned_morton=mesh_aligned_morton_gpu,
        )
        config.L2_SEARCH_METHOD = original
        return fns

    elif l2_method == 'mesh_aligned_octree_multi':
        original = config.L2_SEARCH_METHOD
        config.L2_SEARCH_METHOD = 'mesh_aligned_octree'
        fns = create_rk4_fully_fused_timedep_with_stats(
            **common, l2_search_method='radius',
            mesh_aligned_octree=mesh_aligned_octree_multi_gpu,
            mesh_aligned_octree_use_multi_local=False,
        )
        config.L2_SEARCH_METHOD = original
        return fns

    elif l2_method == 'mesh_aligned_octree_multi_local':
        original = config.L2_SEARCH_METHOD
        config.L2_SEARCH_METHOD = 'mesh_aligned_octree'
        fns = create_rk4_fully_fused_timedep_with_stats(
            **common,
            l2_search_method='radius',
            l2_search_radius=l2_radius if l2_radius is not None else 10,
            mesh_aligned_octree=mesh_aligned_octree_multi_gpu,
            mesh_aligned_octree_use_multi_local=True,
        )
        config.L2_SEARCH_METHOD = original
        return fns

    elif l2_method == 'mesh_aligned_octree_multi_local_where':
        original = config.L2_SEARCH_METHOD
        config.L2_SEARCH_METHOD = 'mesh_aligned_octree'
        fns = create_rk4_fully_fused_timedep_with_stats(
            **common,
            l2_search_method='radius',
            l2_search_radius=l2_radius if l2_radius is not None else 10,
            mesh_aligned_octree=mesh_aligned_octree_multi_gpu,
            mesh_aligned_octree_use_multi_local=True,
            mesh_aligned_octree_use_where=True,
        )
        config.L2_SEARCH_METHOD = original
        return fns

    elif l2_method == 'mesh_aligned_neighbors':
        original = config.L2_SEARCH_METHOD
        config.L2_SEARCH_METHOD = 'mesh_aligned_neighbors'
        fns = create_rk4_fully_fused_timedep_with_stats(
            **common, l2_search_method='radius',
            mesh_aligned_octree_neighbors=mesh_aligned_octree_neighbors_gpu,
        )
        config.L2_SEARCH_METHOD = original
        return fns

    else:
        raise ValueError(f"Unknown L2 method: {l2_method}")


def run_rk4_tracking(positions_gpu, element_ids_gpu, mesh_gpu, mesh_gpu_octree,
                     element_volumes_gpu, velocity_sequence_gpu,
                     l2_method, l2_radius=None, incremental_radii=None, n_steps=100,
                     mesh_aligned_octree_gpu=None, mesh_aligned_morton_gpu=None,
                     mesh_aligned_octree_neighbors_gpu=None, mesh_aligned_octree_multi_gpu=None,
                     method_name="unknown", particle_ids=None):
    """Run RK4 tracking with specified L2 method, logging L0/L1/L2/miss stats per step."""

    n_particles = positions_gpu.shape[0]

    # Set configuration
    config.POINT_IN_TET_METHOD = POINT_IN_TET_METHOD

    # Setup VTK export
    exporter = VTKExportThread(OUTPUT_DIR, method_name)
    exporter.start()

    # Snapshot initial state BEFORE warmup/compile (true post-assignment state)
    positions_cpu_init = np.array(positions_gpu, dtype=config.FLOAT_DTYPE_NP)
    element_ids_cpu_init = np.array(element_ids_gpu, dtype=np.int32)
    n_active_init = int(np.sum(element_ids_cpu_init >= 0))
    exporter.enqueue_export(0, positions_cpu_init, element_ids_cpu_init, particle_ids)
    print(f"      Initial state exported (step 0): {n_active_init:,} active particles")

    # Build both step functions (shared closure, compiled separately on first call)
    rk4_step, rk4_step_with_stats = _build_rk4_functions(
        l2_method, mesh_gpu, mesh_gpu_octree, element_volumes_gpu,
        l2_radius, incremental_radii,
        mesh_aligned_octree_gpu, mesh_aligned_morton_gpu,
        mesh_aligned_octree_neighbors_gpu, mesh_aligned_octree_multi_gpu,
    )

    # Warmup: compile production step (no stats overhead during compile timing)
    print(f"      Compiling...")
    t_compile = time.time()
    positions_gpu, element_ids_gpu = rk4_step(
        positions_gpu, element_ids_gpu, DT, velocity_sequence_gpu, 0
    )
    positions_gpu = jax.block_until_ready(positions_gpu)
    element_ids_gpu = jax.block_until_ready(element_ids_gpu)
    t_compile = time.time() - t_compile
    print(f"      Compilation time: {t_compile:.2f}s")

    # Also trigger compilation of stats step (warm up before timed loop)
    _warmup = rk4_step_with_stats(
        positions_gpu, element_ids_gpu, DT, velocity_sequence_gpu, 0
    )
    jax.block_until_ready(_warmup[0])

    # Prepare stats CSV
    stats_csv_path = OUTPUT_DIR / method_name / "search_stats.csv"
    stats_csv_path.parent.mkdir(parents=True, exist_ok=True)
    stats_csv = open(stats_csv_path, 'w')
    # Header: counts are totals over N_particles × 5 sub-step searches per step
    stats_csv.write(
        "step,n_active,n_lost,"
        "l0_hits,l1_hits,l2_hits,misses,"
        "l0_pct,l1_pct,l2_pct,miss_pct\n"
    )

    # Run tracking with stats.
    # Step numbering: step 0 = initial state (already exported above).
    # RK4 loop produces states at steps 1..n_steps; the velocity-field index
    # passed to rk4_step_with_stats is the 0-based time index (step-1).
    print(f"      Running {n_steps} RK4 steps...")
    t_start = time.time()

    for step in range(1, n_steps + 1):
        time_idx = step - 1  # 0-based index into velocity sequence
        positions_gpu, element_ids_gpu, (l0, l1, l2, miss) = rk4_step_with_stats(
            positions_gpu, element_ids_gpu, DT, velocity_sequence_gpu, time_idx
        )

        # Block every step to collect stats (stats scalars are tiny, no overhead)
        positions_gpu = jax.block_until_ready(positions_gpu)
        element_ids_gpu = jax.block_until_ready(element_ids_gpu)

        # Write stats every step
        l0_i, l1_i, l2_i, miss_i = int(l0), int(l1), int(l2), int(miss)
        total = l0_i + l1_i + l2_i + miss_i
        n_active = int(jnp.sum(element_ids_gpu >= 0))
        n_lost = n_particles - n_active
        if total > 0:
            l0_pct = 100.0 * l0_i / total
            l1_pct = 100.0 * l1_i / total
            l2_pct = 100.0 * l2_i / total
            miss_pct = 100.0 * miss_i / total
        else:
            l0_pct = l1_pct = l2_pct = miss_pct = 0.0
        stats_csv.write(
            f"{step},{n_active},{n_lost},"
            f"{l0_i},{l1_i},{l2_i},{miss_i},"
            f"{l0_pct:.2f},{l1_pct:.2f},{l2_pct:.2f},{miss_pct:.2f}\n"
        )

        # Export VTK at intervals (step 0 was already exported as initial state)
        if step % EXPORT_FREQUENCY == 0 or step == n_steps:
            stats_csv.flush()
            positions_cpu = np.array(positions_gpu, dtype=config.FLOAT_DTYPE_NP)
            element_ids_cpu = np.array(element_ids_gpu, dtype=np.int32)
            exporter.enqueue_export(step, positions_cpu, element_ids_cpu, particle_ids)

        # Log to console at intervals
        if step % LOG_INTERVAL == 0 or step == n_steps:
            print(f"        step {step:4d}: active={n_active:,}  "
                  f"L0={l0_pct:.1f}% L1={l1_pct:.1f}% L2={l2_pct:.1f}% miss={miss_pct:.2f}%")

    stats_csv.close()

    # Final sync
    positions_gpu = jax.block_until_ready(positions_gpu)
    element_ids_gpu = jax.block_until_ready(element_ids_gpu)
    t_elapsed = time.time() - t_start

    print(f"      Stats CSV: {stats_csv_path}")

    # Wait for exports to complete
    print(f"      Waiting for exports...")
    exporter.stop()
    export_stats = exporter.get_stats()
    print(f"      Exported {export_stats['n_exported']} files (mean time: {export_stats['mean_time']:.3f}s)")

    # Final metrics
    n_active_final = int(jnp.sum(element_ids_gpu >= 0))
    retention = (n_active_final / n_particles) * 100
    throughput = (n_particles * n_steps) / t_elapsed

    return positions_gpu, element_ids_gpu, n_active_final, retention, t_elapsed, throughput


def main():
    print("=" * 80)
    print("Comprehensive L2 Search Methods Benchmark")
    print("Fair comparison of all L2 search strategies")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print("=" * 80)

    # ========================================================================
    # 1-7. Load Mesh, Deduplicate, Build Octree, Upload (same as before)
    # ========================================================================

    print("\n[1/10] Loading mesh...")
    t_load = time.time()
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=False
    )
    t_load = time.time() - t_load

    n_nodes_orig = node_positions.shape[0]
    n_elements = connectivity.shape[0]
    n_timesteps = len(velocity_sequence)

    print(f"  Loaded in {t_load:.2f}s")
    print(f"  Elements: {n_elements:,}, Nodes: {n_nodes_orig:,}, Timesteps: {n_timesteps}")

    print("\n[2/10] Deduplicating...")
    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    n_nodes = node_positions.shape[0]
    print(f"  Removed {n_duplicates_removed:,} duplicates")

    print("\n[3/10] Precomputing metadata...")
    aa_metadata = precompute_aa_metadata(connectivity, node_positions, verbose=False)
    element_vertices = precompute_element_vertices(connectivity, node_positions, verbose=False)
    M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)

    # Compute element volumes (needed for adaptive L1 hop count)
    v0 = node_positions[connectivity[:, 0]]
    v1 = node_positions[connectivity[:, 1]]
    v2 = node_positions[connectivity[:, 2]]
    v3 = node_positions[connectivity[:, 3]]
    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0
    cross_e2_e3 = np.cross(e2, e3)
    det = np.sum(e1 * cross_e2_e3, axis=1)
    element_volumes = np.abs(det) / 6.0

    print(f"  Metadata ready")

    print("\n[4/10] Building octree...")
    octree_struct = build_global_morton_octree(
        node_positions=node_positions,
        connectivity=connectivity,
        leaf_capacity=256,
        max_depth=21,
        verbose=False
    )
    print(f"  Built {octree_struct.n_leaves:,} leaves")

    # Build mesh-aligned octree (single-cell registration)
    print(f"\n  Building mesh-aligned octree (single-cell)...")
    t_mesh_octree = time.time()
    mesh_octree_cells = extract_octree_cells_single(
        node_positions, connectivity, tolerance=1e-6, verbose=False
    )
    t_mesh_octree = time.time() - t_mesh_octree
    print(f"    Extracted {mesh_octree_cells.n_cells:,} cells in {t_mesh_octree:.2f}s")
    print(f"    Elements per cell: {mesh_octree_cells.elements_per_cell_mean:.2f}")
    print(f"    Cells per element: {mesh_octree_cells.cells_per_element_mean:.2f}")

    # Build mesh-aligned octree (multi-cell vertex registration)
    print(f"\n  Building mesh-aligned octree (multi-cell vertex registration)...")
    t_mesh_octree_multi = time.time()
    mesh_octree_cells_multi = extract_octree_cells_vertex_multi(
        node_positions, connectivity, tolerance=1e-6, verbose=False
    )
    t_mesh_octree_multi = time.time() - t_mesh_octree_multi
    print(f"    Extracted {mesh_octree_cells_multi.n_cells:,} cells in {t_mesh_octree_multi:.2f}s")
    print(f"    Elements per cell: {mesh_octree_cells_multi.elements_per_cell_mean:.2f}")
    print(f"    Cells per element: {mesh_octree_cells_multi.cells_per_element_mean:.2f}")

    # Build mesh-aligned octree with neighbor table (Option B)
    print(f"\n  Building mesh-aligned octree with neighbor table (Option B)...")
    from jaxtrace.gpu.search.mesh_aligned_octree_with_neighbor_table import (
        add_neighbor_table_to_octree,
        upload_octree_with_neighbors_to_gpu
    )
    t_neighbor_build = time.time()
    octree_with_neighbors = add_neighbor_table_to_octree(mesh_octree_cells, verbose=False)
    t_neighbor_build = time.time() - t_neighbor_build
    print(f"    Neighbor table built in {t_neighbor_build:.2f}s")
    # Compute mean neighbors from cell_neighbors array (count non-negative entries)
    mean_neighbors = (octree_with_neighbors.cell_neighbors >= 0).sum(axis=1).mean()
    print(f"    Mean neighbors per cell: {mean_neighbors:.1f}")

    print("\n[5/10] Uploading to GPU...")
    element_neighbors = build_element_neighbors_array(connectivity, method='face', verbose=False)
    mesh_gpu = upload_mesh_to_gpu(connectivity, node_positions, element_neighbors, verbose=False)
    mesh_gpu_octree = upload_global_morton_to_gpu(octree_struct, connectivity, node_positions)

    # Upload mesh-aligned structures
    print(f"  Uploading mesh-aligned octree (single-cell)...")
    mesh_aligned_octree_gpu = upload_mesh_aligned_octree_to_gpu(
        connectivity, node_positions, mesh_octree_cells, verbose=False
    )

    print(f"  Uploading mesh-aligned octree (multi-cell)...")
    mesh_aligned_octree_multi_gpu = upload_mesh_aligned_octree_to_gpu(
        connectivity, node_positions, mesh_octree_cells_multi, verbose=False
    )

    print(f"  Uploading mesh-aligned octree with neighbors (single-cell)...")
    mesh_aligned_octree_neighbors_gpu = upload_octree_with_neighbors_to_gpu(
        connectivity, node_positions, octree_with_neighbors, verbose=False
    )

    # For multi-cell octree, we don't need a neighbor table since each element
    # is already registered in ~4 cells (vertices at cube corners).
    # The neighbor table is designed for single-cell registration where elements
    # only appear in one cell. For multi-cell, searching neighbors would mean
    # searching ~4 cells × 26 neighbors = ~104 cells, which is excessive.
    #
    # Instead, we'll use the multi-cell octree directly without neighbors.
    # This will search ~4 cells per particle (much better than single-cell's 1 cell).
    print(f"\n  NOTE: Multi-cell octree doesn't use neighbor table")
    print(f"    Multi-cell registration already covers ~4 cells per element")
    print(f"    Neighbor table would search ~104 cells (excessive)")
    mesh_aligned_octree_neighbors_multi_gpu = None  # Not used

    print(f"  Building mesh-aligned Morton (hybrid)...")
    mesh_aligned_morton_struct = build_mesh_aligned_morton_structure(
        node_positions, connectivity, mesh_octree_cells=mesh_octree_cells, verbose=False
    )
    print(f"    Elements per cell: mean={mesh_aligned_morton_struct.elements_per_cell_mean:.1f}, "
          f"max={mesh_aligned_morton_struct.elements_per_cell_max}")
    mesh_aligned_morton_gpu = upload_mesh_aligned_morton_to_gpu(
        node_positions, connectivity, mesh_aligned_morton_struct, verbose=False
    )

    # Upload metadata
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

    # Upload velocity sequence
    velocity_sequence_gpu = jax.device_put(velocity_sequence)

    print(f"  Uploaded to GPU")

    # ========================================================================
    # 6. Generate Particles
    # ========================================================================

    # Compute mesh bounds
    domain_min = node_positions.min(axis=0)
    domain_max = node_positions.max(axis=0)
    domain_size = domain_max - domain_min

    # Calculate particle count from grid resolution
    nx, ny, nz = PARTICLE_GRID_RESOLUTION
    n_particles = nx * ny * nz

    if PARTICLE_SEEDING == 'uniform_grid':
        print(f"\n[6/10] Generating particles (uniform grid: {nx}×{ny}×{nz} = {n_particles:,})...")

        # Calculate particle bounds from fractions (same as production)
        par_bounds_min = np.zeros(3, dtype=config.FLOAT_DTYPE_NP)
        par_bounds_max = np.zeros(3, dtype=config.FLOAT_DTYPE_NP)
        for i, axis in enumerate(['x', 'y', 'z']):
            min_frac, max_frac = PARTICLE_BOUNDS_FRACTION[axis]
            par_bounds_min[i] = domain_min[i] + min_frac * domain_size[i]
            par_bounds_max[i] = domain_min[i] + max_frac * domain_size[i]
        par_bounds = [par_bounds_min, par_bounds_max]

        print(f"  Particle bounds:")
        print(f"    X: [{par_bounds_min[0]:.6f}, {par_bounds_max[0]:.6f}] (domain fraction: {PARTICLE_BOUNDS_FRACTION['x']})")
        print(f"    Y: [{par_bounds_min[1]:.6f}, {par_bounds_max[1]:.6f}] (domain fraction: {PARTICLE_BOUNDS_FRACTION['y']})")
        print(f"    Z: [{par_bounds_min[2]:.6f}, {par_bounds_max[2]:.6f}] (domain fraction: {PARTICLE_BOUNDS_FRACTION['z']})")

        # Generate uniform grid (same as production)
        particle_positions = uniform_grid_seeds(
            resolution=(nx, ny, nz),
            bounds=par_bounds,
            include_boundaries=True
        )

        # Clip particles to mesh bounds with 1% safety margin (Phase 1.1 fix from production)
        print(f"  Clipping particles to mesh bounds (Phase 1.1 fix)...")
        mesh_bbox_min = domain_min
        mesh_bbox_max = domain_max
        margin = 0.01
        bbox_min_safe = mesh_bbox_min + margin * (mesh_bbox_max - mesh_bbox_min)
        bbox_max_safe = mesh_bbox_max - margin * (mesh_bbox_max - mesh_bbox_min)

        particle_positions = np.clip(particle_positions, bbox_min_safe, bbox_max_safe)

        # No ground truth for uniform grid (particles may start outside elements)
        ground_truth_element_ids = np.full(n_particles, -1, dtype=np.int32)

        positions_gpu = jax.device_put(particle_positions)
        ground_truth_element_ids_gpu = jax.device_put(ground_truth_element_ids)

        print(f"  Generated {n_particles:,} particles on uniform grid")
        print(f"    Clipped to safe mesh bounds (1% margin)")
        print(f"    No ground truth element IDs (particles may be outside mesh)")

    elif PARTICLE_SEEDING == 'centroids':
        print(f"\n[6/10] Generating particles (perturbed element centroids: {n_particles:,})...")

        # Compute element centroids
        n_elements = connectivity.shape[0]
        element_centroids = np.zeros((n_elements, 3), dtype=config.FLOAT_DTYPE_NP)
        for elem_idx in range(n_elements):
            elem_nodes = connectivity[elem_idx]
            elem_positions = node_positions[elem_nodes]
            element_centroids[elem_idx] = elem_positions.mean(axis=0)

        # Filter elements using PARTICLE_BOUNDS_FRACTION (same bounds as uniform grid)
        par_bounds_min = np.zeros(3, dtype=config.FLOAT_DTYPE_NP)
        par_bounds_max = np.zeros(3, dtype=config.FLOAT_DTYPE_NP)
        for i, axis in enumerate(['x', 'y', 'z']):
            min_frac, max_frac = PARTICLE_BOUNDS_FRACTION[axis]
            par_bounds_min[i] = domain_min[i] + min_frac * domain_size[i]
            par_bounds_max[i] = domain_min[i] + max_frac * domain_size[i]

        valid_elements_mask = (
            (element_centroids[:, 0] >= par_bounds_min[0]) & (element_centroids[:, 0] <= par_bounds_max[0]) &
            (element_centroids[:, 1] >= par_bounds_min[1]) & (element_centroids[:, 1] <= par_bounds_max[1]) &
            (element_centroids[:, 2] >= par_bounds_min[2]) & (element_centroids[:, 2] <= par_bounds_max[2])
        )
        valid_element_ids = np.where(valid_elements_mask)[0]

        print(f"  Filtering elements using PARTICLE_BOUNDS_FRACTION")
        print(f"    X: [{par_bounds_min[0]:.6f}, {par_bounds_max[0]:.6f}] (fraction: {PARTICLE_BOUNDS_FRACTION['x']})")
        print(f"    Y: [{par_bounds_min[1]:.6f}, {par_bounds_max[1]:.6f}] (fraction: {PARTICLE_BOUNDS_FRACTION['y']})")
        print(f"    Z: [{par_bounds_min[2]:.6f}, {par_bounds_max[2]:.6f}] (fraction: {PARTICLE_BOUNDS_FRACTION['z']})")
        print(f"    Valid elements: {len(valid_element_ids):,} / {n_elements:,} ({100*len(valid_element_ids)/n_elements:.1f}%)")

        # Select random elements from valid set
        np.random.seed(SEED)
        selected_elements = np.random.choice(valid_element_ids, n_particles, replace=True)

        # Compute element centroids
        particle_positions = np.zeros((n_particles, 3), dtype=config.FLOAT_DTYPE_NP)
        for i, elem_idx in enumerate(selected_elements):
            elem_nodes = connectivity[elem_idx]
            elem_positions = node_positions[elem_nodes]
            particle_positions[i] = elem_positions.mean(axis=0)

        # Add small perturbations (10% of smallest element size)
        sample_size = min(100000, len(valid_element_ids))
        element_sizes = np.zeros(sample_size, dtype=config.FLOAT_DTYPE_NP)
        for i in range(sample_size):
            elem_idx = valid_element_ids[i % len(valid_element_ids)]
            elem_nodes = connectivity[elem_idx]
            elem_positions = node_positions[elem_nodes]
            edges = []
            for j in range(4):
                for k in range(j+1, 4):
                    edge_len = np.linalg.norm(elem_positions[j] - elem_positions[k])
                    edges.append(edge_len)
            element_sizes[i] = min(edges)

        min_element_size = np.percentile(element_sizes[element_sizes > 0], 5)
        perturbation_scale = min_element_size * 0.1

        perturbations = np.random.randn(n_particles, 3).astype(config.FLOAT_DTYPE_NP) * perturbation_scale
        particle_positions += perturbations

        # Store ground truth element IDs for RK4 tracking
        ground_truth_element_ids = selected_elements.astype(np.int32)

        positions_gpu = jax.device_put(particle_positions)
        ground_truth_element_ids_gpu = jax.device_put(ground_truth_element_ids)

        print(f"  Generated {n_particles:,} particles from perturbed centroids")
        print(f"    Perturbation scale: {perturbation_scale:.6e} ({perturbation_scale/min_element_size:.1%} of min element)")
        print(f"    Mean perturbation: {np.linalg.norm(perturbations, axis=1).mean():.6e}")
        print(f"    Ground truth element IDs stored for RK4 tracking")

    else:
        raise ValueError(f"Invalid PARTICLE_SEEDING: {PARTICLE_SEEDING}. Choose 'uniform_grid' or 'centroids'.")

    # Generate unique particle IDs for tracking across timesteps
    particle_ids = np.arange(n_particles, dtype=np.int32)
    print(f"  Particle IDs: 0..{n_particles - 1} (for cross-timestep tracking)")

    # ========================================================================
    # 7. Define Test Configurations
    # ========================================================================

    print("\n[7/10] Defining test configurations...")
    print("=" * 80)

    test_configs = [
        # # Baseline: Fixed radius=10
        # {
        #     'name': 'Fixed radius=10 (baseline)',
        #     'l2_method': 'radius',
        #     'l2_radius': 10,
        #     'incremental_radii': None,
        #     'description': 'Fixed radius search (21 leaves)',
        #     'expected_leaves': 21
        # },

        # # Fixed radius=30 (max coverage)
        # {
        #     'name': 'Fixed radius=30 (max coverage)',
        #     'l2_method': 'radius',
        #     'l2_radius': 30,
        #     'incremental_radii': None,
        #     'description': 'Large radius for maximum retention (61 leaves)',
        #     'expected_leaves': 61
        # },

        # # Incremental 5-tier (PRODUCTION)
        # {
        #     'name': 'Incremental (2,4,8,15,30) - PRODUCTION',
        #     'l2_method': 'incremental',
        #     'l2_radius': None,
        #     'incremental_radii': (2, 4, 8, 15, 30),
        #     'description': '5-tier cascading (production config)',
        #     'expected_leaves': '22.5 avg (conservative)'
        # },

        # # Incremental 3-tier (simpler)
        # {
        #     'name': 'Incremental (2,5,10) - 3-tier',
        #     'l2_method': 'incremental',
        #     'l2_radius': None,
        #     'incremental_radii': (2, 5, 10),
        #     'description': '3-tier cascading (simpler alternative)',
        #     'expected_leaves': '11.5 avg (60/30/10)'
        # },

        # # Neighbors
        # {
        #     'name': 'Neighbors (Morton arithmetic)',
        #     'l2_method': 'neighbors',
        #     'l2_radius': None,
        #     'incremental_radii': None,
        #     'description': 'Morton neighbor arithmetic',
        #     'expected_leaves': 'Variable'
        # },

        # # Hierarchical
        # {
        #     'name': 'Hierarchical (multi-depth)',
        #     'l2_method': 'hierarchical',
        #     'l2_radius': None,
        #     'incremental_radii': None,
        #     'description': 'Multi-depth conditional search',
        #     'expected_leaves': 'Variable'
        # },

        # # Mesh-aligned octree (DIRECT)
        # {
        #     'name': 'Mesh-Aligned Octree (direct)',
        #     'l2_method': 'mesh_aligned_octree',
        #     'l2_radius': None,
        #     'incremental_radii': None,
        #     'description': 'Direct cell lookup (center cell only)',
        #     'expected_leaves': '~5.9 elements/cell'
        # },

        # # Mesh-aligned Morton (HYBRID - NEW)
        # {
        #     'name': 'Mesh-Aligned Morton r=2 (HYBRID - NEW)',
        #     'l2_method': 'mesh_aligned_morton',
        #     'l2_radius': 2,
        #     'incremental_radii': None,
        #     'description': 'Morton radius over cell centers (5 cells)',
        #     'expected_leaves': '~30 tests (5 cells × 5.9 elem/cell)'
        # },

        # # Mesh-aligned Morton incremental
        # {
        #     'name': 'Mesh-Aligned Morton (2,5,10) (HYBRID - NEW)',
        #     'l2_method': 'mesh_aligned_morton',
        #     'l2_radius': None,
        #     'incremental_radii': (2, 5, 10),
        #     'description': 'Incremental radius over cell centers',
        #     'expected_leaves': '~68 tests avg (11.5 cells × 5.9 elem/cell)'
        # },

        # # Mesh-aligned neighbors (Option B - NEW)
        # {
        #     'name': 'Mesh-Aligned Neighbors (Option B - NEW)',
        #     'l2_method': 'mesh_aligned_neighbors',
        #     'l2_radius': None,
        #     'incremental_radii': None,
        #     'description': 'Pre-computed neighbor table (27 cells @ 3 levels)',
        #     'expected_leaves': '~13.9 tests/particle, 99.95% for centroids'
        # },

        # # Mesh-aligned octree MULTI-CELL vertex registration (NEW - Phase 2)
        # {
        #     'name': 'Mesh-Aligned Octree Multi-Cell (Phase 2 - NEW)',
        #     'l2_method': 'mesh_aligned_octree_multi',
        #     'l2_radius': None,
        #     'incremental_radii': None,
        #     'description': 'Multi-cell vertex registration (~4 cells per element)',
        #     'expected_leaves': '~94 tests/particle (~4 cells × ~23.6 elem/cell)'
        # },

        # Mesh-aligned octree MULTI-CELL with 3×3×3 local search (Option A) - lax.cond version
        {
            'name': 'Mesh-Aligned Multi-Cell + 3×3×3 Local (lax.cond)',
            'l2_method': 'mesh_aligned_octree_multi_local',
            'l2_radius': None,
            'incremental_radii': None,
            'description': 'Multi-cell + 3×3×3 local search with lax.cond (original)',
            'expected_leaves': '~494 tests/particle (27 cells × 18.31 elem/cell)'
        },

        # Mesh-aligned octree MULTI-CELL with 3×3×3 local search - jnp.where version
        {
            'name': 'Mesh-Aligned Multi-Cell + 3×3×3 Local (jnp.where)',
            'l2_method': 'mesh_aligned_octree_multi_local_where',
            'l2_radius': None,
            'incremental_radii': None,
            'description': 'Multi-cell + 3×3×3 local search with jnp.where (vmap fix)',
            'expected_leaves': '~494 tests/particle (27 cells × 18.31 elem/cell)'
        },
    ]

    for i, cfg in enumerate(test_configs, 1):
        print(f"{i}. {cfg['name']}")
        print(f"   {cfg['description']}")
        print(f"   Expected work: {cfg['expected_leaves']}")
        print()

    # ========================================================================
    # 8. Run Initial Assignment for All Configurations
    # ========================================================================

    print("\n[8/10] Running initial assignment for all configurations...")
    print("=" * 80)

    initial_results = {}

    for cfg in test_configs:
        name = cfg['name']
        print(f"\n  Config: {name}")

        element_ids, n_assigned, t_elapsed = run_initial_assignment(
            positions_gpu,
            mesh_gpu_octree,
            l2_method=cfg['l2_method'],
            l2_radius=cfg['l2_radius'],
            incremental_radii=cfg['incremental_radii'],
            mesh_aligned_octree_neighbors_gpu=mesh_aligned_octree_neighbors_gpu,
            mesh_aligned_octree_multi_gpu=mesh_aligned_octree_multi_gpu
        )

        success_rate = (n_assigned / n_particles) * 100
        throughput = n_particles / t_elapsed

        initial_results[name] = {
            'element_ids': element_ids,
            'n_assigned': n_assigned,
            'success_rate': success_rate,
            'time': t_elapsed,
            'throughput': throughput
        }

        print(f"    Time: {t_elapsed:.3f}s")
        print(f"    Assigned: {n_assigned:,}/{n_particles:,} ({success_rate:.2f}%)")
        print(f"    Throughput: {throughput:,.0f} p/s")

    # ========================================================================
    # 9. Run RK4 Tracking for All Configurations
    # ========================================================================

    print("\n[9/10] Running RK4 tracking for all configurations...")
    print("=" * 80)
    print(f"Configuration: {N_STEPS} steps, dt={DT}, point-in-tet={POINT_IN_TET_METHOD}")
    print()

    tracking_results = {}

    for cfg in test_configs:
        name = cfg['name']
        print(f"\n  Config: {name}")

        # Use initial assignment results for uniform_grid seeding (particles start
        # outside elements, so we need L2 to find them first).
        # For centroid seeding, ground truth element IDs are available directly.
        if PARTICLE_SEEDING == 'uniform_grid':
            element_ids_initial = initial_results[name]['element_ids']
        else:
            element_ids_initial = ground_truth_element_ids_gpu

        positions_final, element_ids_final, n_active_final, retention, t_elapsed, throughput = run_rk4_tracking(
            positions_gpu,
            element_ids_initial,
            mesh_gpu,
            mesh_gpu_octree,
            element_volumes_gpu,
            velocity_sequence_gpu,
            l2_method=cfg['l2_method'],
            l2_radius=cfg['l2_radius'],
            incremental_radii=cfg['incremental_radii'],
            n_steps=N_STEPS,
            mesh_aligned_octree_gpu=mesh_aligned_octree_gpu,
            mesh_aligned_morton_gpu=mesh_aligned_morton_gpu,
            mesh_aligned_octree_neighbors_gpu=mesh_aligned_octree_neighbors_gpu,
            mesh_aligned_octree_multi_gpu=mesh_aligned_octree_multi_gpu,
            method_name=name,
            particle_ids=particle_ids
        )

        tracking_results[name] = {
            'positions': positions_final,
            'element_ids': element_ids_final,
            'n_active_final': n_active_final,
            'retention': retention,
            'time': t_elapsed,
            'throughput': throughput
        }

        print(f"    Time: {t_elapsed:.3f}s")
        print(f"    Final active: {n_active_final:,}/{n_particles:,} ({retention:.2f}%)")
        print(f"    Throughput: {throughput:,.0f} p/s")

    # ========================================================================
    # 10. Results Analysis
    # ========================================================================

    print("\n[10/10] Results Analysis")
    print("=" * 80)

    # Initial Assignment Summary
    print("\nINITIAL ASSIGNMENT RESULTS")
    print("=" * 80)
    print(f"{'Configuration':<40s}  {'Success Rate':>12s}  {'Throughput':>14s}  {'Time':>8s}")
    print("-" * 80)

    for cfg in test_configs:
        name = cfg['name']
        r = initial_results[name]
        print(f"{name:<40s}  {r['success_rate']:11.2f}%  {r['throughput']:13,.0f} p/s  {r['time']:7.3f}s")

    # RK4 Tracking Summary
    print("\n\nRK4 TRACKING RESULTS ({} steps)".format(N_STEPS))
    print("=" * 80)
    print("Note: All methods start with initial assignment element IDs (per-method)")
    print(f"{'Configuration':<40s}  {'Retention':>10s}  {'Throughput':>14s}  {'Speedup':>8s}")
    print("-" * 80)

    baseline_name = test_configs[0]['name']  # Use first config as baseline
    baseline_time = tracking_results[baseline_name]['time']

    best_throughput = 0
    best_config = None

    for cfg in test_configs:
        name = cfg['name']
        r = tracking_results[name]
        speedup = baseline_time / r['time']

        marker = ""
        if r['throughput'] > best_throughput:
            best_throughput = r['throughput']
            best_config = name
            marker = " ★"

        print(f"{name:<40s}  {r['retention']:9.2f}%  {r['throughput']:13,.0f} p/s  {speedup:7.2f}×{marker}")

    # Accuracy vs Performance Trade-off
    print("\n\nACCURACY vs PERFORMANCE TRADE-OFF")
    print("=" * 80)
    print(f"{'Configuration':<40s}  {'Retention':>10s}  {'Speedup':>8s}  {'Rating':>10s}")
    print("-" * 80)

    for cfg in test_configs:
        name = cfg['name']
        r = tracking_results[name]
        retention = r['retention']
        speedup = baseline_time / r['time']

        # Rating based on retention + speedup
        if retention >= 93.0 and speedup >= 1.8:
            rating = "EXCELLENT"
        elif retention >= 90.0 and speedup >= 1.5:
            rating = "GOOD"
        elif retention >= 85.0 and speedup >= 1.2:
            rating = "ACCEPTABLE"
        else:
            rating = "POOR"

        print(f"{name:<40s}  {retention:9.2f}%  {speedup:7.2f}×  {rating:>10s}")

    # Recommendations
    print("\n\nRECOMMENDATIONS")
    print("=" * 80)

    print(f"\nBest Throughput: {best_config}")
    best_retention = tracking_results[best_config]['retention']
    best_speedup = baseline_time / tracking_results[best_config]['time']

    print(f"  Retention: {best_retention:.2f}%")
    print(f"  Speedup: {best_speedup:.2f}×")
    print(f"  Throughput: {best_throughput:,.0f} p/s")

    # Find best accuracy
    best_retention_val = 0
    best_retention_config = None
    for cfg in test_configs:
        name = cfg['name']
        r = tracking_results[name]
        if r['retention'] > best_retention_val:
            best_retention_val = r['retention']
            best_retention_config = name

    print(f"\nBest Retention: {best_retention_config}")
    print(f"  Retention: {best_retention_val:.2f}%")

    retention_speedup = baseline_time / tracking_results[best_retention_config]['time']
    print(f"  Speedup: {retention_speedup:.2f}×")

    # Production recommendation
    print("\n\nPRODUCTION RECOMMENDATION")
    print("=" * 80)

    production_config = 'Incremental (2,4,8,15,30) - PRODUCTION'
    if production_config in tracking_results:
        prod_r = tracking_results[production_config]
        prod_speedup = baseline_time / prod_r['time']

        print(f"\nCurrent Production Config: {production_config}")
        print(f"  Retention: {prod_r['retention']:.2f}%")
        print(f"  Speedup: {prod_speedup:.2f}×")
        print(f"  Throughput: {prod_r['throughput']:,.0f} p/s")

        if prod_speedup >= 1.8:
            print(f"\n✅ Production config achieves {prod_speedup:.2f}× speedup - EXCELLENT")
            print(f"   Recommendation: Continue using current configuration")
        else:
            print(f"\n⚠️  Production config achieves {prod_speedup:.2f}× speedup")
            print(f"   Consider alternative: {best_config} ({best_speedup:.2f}×)")

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
