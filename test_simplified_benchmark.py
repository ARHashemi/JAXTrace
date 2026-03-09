#!/usr/bin/env python3
"""
Simplified benchmark with only baseline and mesh_aligned_octree_multi_local.
Extended to 2500 steps with VTK export every 100 steps.
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from queue import Queue
from threading import Thread

sys.path.insert(0, str(Path(__file__).parent))

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import extract_octree_cells_vertex_multi
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu
from jaxtrace.gpu.tracking.initial_assignment_cascading import initial_assignment_cascading_fallback
from jaxtrace.gpu.search.point_in_tet_methods import set_inverse_matrices_gpu
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
from jaxtrace.gpu.tracking.rk4_fully_fused_timedep import create_rk4_fully_fused_timedep
import jaxtrace.config as config

# Import VTK export functionality
try:
    from jaxtrace.io.vtk_io_enhanced import VTKDataCLI, write_vtk_cli
    HAS_VTK = True
except ImportError:
    HAS_VTK = False
    print("⚠️  Warning: VTK export not available")


# ============================================================================
# Configuration
# ============================================================================

MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)  # 2 timesteps
VELOCITY_FIELD_NAME = 'Displacement'

# Particle seeding (same as production)
PARTICLE_GRID_RESOLUTION = (60, 90, 60)  # 324,000 particles
PARTICLE_BOUNDS_FRACTION = {
    'x': (0.3, 0.7),
    'y': (0.2, 0.8),
    'z': (0.3, 1.0),
}

# RK4 integration
DT = 0.0025
N_STEPS = 2500  # INCREASED from 100 to 2500

# L1 configuration
ENABLE_L1_SEARCH = True
N_HOPS = 5

# Point-in-tet method
POINT_IN_TET_METHOD = 'inverse'

# VTK export configuration
EXPORT_FREQUENCY = 10  # Export every 10 steps
OUTPUT_DIR = Path("output/simplified_benchmark")
LOG_INTERVAL = 100

SEED = 42


# ============================================================================
# VTK Export Thread Class
# ============================================================================

class VTKExportThread:
    """Background thread for VTK export."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.queue = Queue()
        self.thread = Thread(target=self._export_worker, daemon=True)
        self.running = True
        self.stats = {
            'n_exported': 0,
            'total_time': 0.0,
        }

    def start(self):
        """Start the export thread."""
        self.thread.start()

    def enqueue_export(self, step: int, positions: np.ndarray, element_ids: np.ndarray):
        """Add export task to queue."""
        self.queue.put((step, positions, element_ids))

    def _export_worker(self):
        """Worker thread for VTK export."""
        while self.running:
            try:
                task = self.queue.get(timeout=0.1)
                if task is None:
                    break

                step, positions, element_ids = task
                t_start = time.time()

                # Write VTK file
                vtk_file = self.output_dir / f"particles_step_{step:06d}.vtu"

                if HAS_VTK:
                    n_particles = positions.shape[0]
                    vtk_data = VTKDataCLI(
                        positions=positions,
                        point_data={
                            'element_id': element_ids,
                            'active': (element_ids >= 0).astype(np.int32)
                        },
                        cell_data=None
                    )
                    write_vtk_cli(vtk_file, vtk_data)
                else:
                    # Fallback: write simple text file
                    np.savetxt(
                        vtk_file.with_suffix('.txt'),
                        positions,
                        header=f"Step {step} - {positions.shape[0]} particles"
                    )

                t_elapsed = time.time() - t_start
                self.stats['n_exported'] += 1
                self.stats['total_time'] += t_elapsed

                self.queue.task_done()

            except Exception:
                continue

    def stop(self):
        """Stop the export thread."""
        self.queue.put(None)
        self.running = False
        self.thread.join()

    def get_stats(self):
        """Get export statistics."""
        return {
            'n_exported': self.stats['n_exported'],
            'total_time': self.stats['total_time'],
            'mean_time': self.stats['total_time'] / max(1, self.stats['n_exported'])
        }


# ============================================================================
# Main Function
# ============================================================================

def main():
    print("=" * 80)
    print("Simplified Benchmark: Baseline vs Mesh-Aligned Multi-Cell + 2×2×2 Local")
    print("Extended to 2500 steps with VTK export every 100 steps")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print("=" * 80)
    print("\nTest Configuration:")
    print(f"  Particles: {PARTICLE_GRID_RESOLUTION[0]} × {PARTICLE_GRID_RESOLUTION[1]} × {PARTICLE_GRID_RESOLUTION[2]} = {PARTICLE_GRID_RESOLUTION[0] * PARTICLE_GRID_RESOLUTION[1] * PARTICLE_GRID_RESOLUTION[2]:,}")
    print(f"  RK4 steps: {N_STEPS}")
    print(f"  Timestep dt: {DT}")
    print(f"  Velocity timesteps: {VELOCITY_TIMESTEP_RANGE[0]} to {VELOCITY_TIMESTEP_RANGE[1]} ({VELOCITY_TIMESTEP_RANGE[1] - VELOCITY_TIMESTEP_RANGE[0]} timesteps)")
    print(f"  VTK export: every {EXPORT_FREQUENCY} steps")
    print(f"  Point-in-tet method: {POINT_IN_TET_METHOD}")
    print(f"  L1 search: {ENABLE_L1_SEARCH} (n_hops={N_HOPS})")
    print("=" * 80)

    # ========================================================================
    # Load and Prepare Mesh
    # ========================================================================

    print("\n[1/9] Loading mesh...")
    t_load = time.time()
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=False
    )
    t_load = time.time() - t_load

    n_elements = connectivity.shape[0]
    n_timesteps = len(velocity_sequence)

    print(f"  Loaded in {t_load:.2f}s")
    print(f"  Elements: {n_elements:,}, Nodes: {node_positions.shape[0]:,}, Timesteps: {n_timesteps}")

    print("\n[2/9] Deduplicating...")
    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    n_nodes = node_positions.shape[0]
    print(f"  Removed {n_duplicates_removed:,} duplicates, final nodes: {n_nodes:,}")

    print("\n[3/9] Precomputing metadata...")
    M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)

    # Compute element volumes
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

    print("\n[4/9] Building octrees...")
    octree_struct = build_global_morton_octree(
        node_positions=node_positions,
        connectivity=connectivity,
        leaf_capacity=256,
        max_depth=21,
        verbose=False
    )
    print(f"  Global Morton octree: {octree_struct.n_leaves:,} leaves")

    # Build multi-cell octree
    print(f"  Building mesh-aligned octree (multi-cell vertex registration)...")
    t_mesh_octree_multi = time.time()
    mesh_octree_cells_multi = extract_octree_cells_vertex_multi(
        node_positions, connectivity, tolerance=1e-6, verbose=False
    )
    t_mesh_octree_multi = time.time() - t_mesh_octree_multi
    print(f"    Extracted {mesh_octree_cells_multi.n_cells:,} cells in {t_mesh_octree_multi:.2f}s")
    print(f"    Elements per cell: {mesh_octree_cells_multi.elements_per_cell_mean:.2f}")
    print(f"    Cells per element: {mesh_octree_cells_multi.cells_per_element_mean:.2f}")

    print("\n[5/9] Uploading to GPU...")
    element_neighbors = build_element_neighbors_array(connectivity, method='face', verbose=False)
    mesh_gpu = upload_mesh_to_gpu(connectivity, node_positions, element_neighbors, verbose=False)
    mesh_gpu_octree = upload_global_morton_to_gpu(octree_struct, connectivity, node_positions)

    mesh_aligned_octree_multi_gpu = upload_mesh_aligned_octree_to_gpu(
        node_positions, connectivity, mesh_octree_cells_multi, verbose=False
    )

    M_inv_gpu = jax.device_put(M_inv_array)
    p0_gpu = jax.device_put(p0_array)
    element_volumes_gpu = jax.device_put(element_volumes.astype(np.float32))
    velocity_sequence_gpu = jax.device_put(velocity_sequence)

    set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)

    print(f"  Uploaded to GPU")

    print("\n[6/9] Generating particles...")
    # Compute mesh bounds
    domain_min = node_positions.min(axis=0)
    domain_max = node_positions.max(axis=0)

    # Filter elements in middle region
    element_centroids = np.zeros((n_elements, 3), dtype=np.float32)
    for elem_idx in range(n_elements):
        elem_nodes = connectivity[elem_idx]
        elem_positions = node_positions[elem_nodes]
        element_centroids[elem_idx] = elem_positions.mean(axis=0)

    x_min_filter = domain_min[0] + 0.3 * (domain_max[0] - domain_min[0])
    x_max_filter = domain_min[0] + 0.7 * (domain_max[0] - domain_min[0])

    valid_elements_mask = (element_centroids[:, 0] >= x_min_filter) & (element_centroids[:, 0] <= x_max_filter)
    valid_element_ids = np.where(valid_elements_mask)[0]

    print(f"  Valid elements: {len(valid_element_ids):,} / {n_elements:,}")

    # Generate particles
    np.random.seed(SEED)
    nx, ny, nz = PARTICLE_GRID_RESOLUTION
    n_particles = nx * ny * nz

    selected_elements = np.random.choice(valid_element_ids, n_particles, replace=True)

    particle_positions = np.zeros((n_particles, 3), dtype=np.float32)
    for i, elem_idx in enumerate(selected_elements):
        elem_nodes = connectivity[elem_idx]
        elem_positions = node_positions[elem_nodes]
        particle_positions[i] = elem_positions.mean(axis=0)

    # Add perturbations
    sample_size = min(100000, len(valid_element_ids))
    element_sizes = np.zeros(sample_size, dtype=np.float32)
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

    perturbations = np.random.randn(n_particles, 3).astype(np.float32) * perturbation_scale
    particle_positions += perturbations

    ground_truth_element_ids = selected_elements.copy()

    positions_gpu = jax.device_put(particle_positions)
    ground_truth_element_ids_gpu = jax.device_put(ground_truth_element_ids)

    print(f"  Generated {n_particles:,} particles")
    print(f"    Perturbation scale: {perturbation_scale:.6e}")

    # ========================================================================
    # Run Baseline Method
    # ========================================================================

    print("\n[7/9] Testing Baseline Method (radius=10)...")
    print("=" * 80)

    config.POINT_IN_TET_METHOD = POINT_IN_TET_METHOD

    print("\n  DEBUG: Array shapes before baseline RK4 creation:")
    print(f"    positions_gpu: {positions_gpu.shape}")
    print(f"    ground_truth_element_ids_gpu: {ground_truth_element_ids_gpu.shape}")
    print(f"    velocity_sequence_gpu: {velocity_sequence_gpu.shape}")
    print(f"    DT: {DT}")
    print(f"    mesh_gpu.connectivity: {mesh_gpu.connectivity.shape}")
    print(f"    mesh_gpu.node_positions: {mesh_gpu.node_positions.shape}")
    print(f"    mesh_gpu.element_neighbors: {mesh_gpu.element_neighbors.shape}")
    print(f"    element_volumes_gpu: {element_volumes_gpu.shape}")
    print(f"    mesh_gpu_octree.elem_ids_sorted: {mesh_gpu_octree.elem_ids_sorted.shape}")
    print(f"    mesh_gpu_octree.morton_sorted: {mesh_gpu_octree.morton_sorted.shape}")
    print(f"    mesh_gpu_octree.leaf_start: {mesh_gpu_octree.leaf_start.shape}")
    print(f"    mesh_gpu_octree.leaf_length: {mesh_gpu_octree.leaf_length.shape}")
    print()

    print("  Creating RK4 function...")
    rk4_step_baseline = create_rk4_fully_fused_timedep(
        mesh_gpu_connectivity=mesh_gpu.connectivity,
        mesh_gpu_node_positions=mesh_gpu.node_positions,
        mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
        mesh_gpu_element_volumes=element_volumes_gpu,
        mesh_gpu_global_morton=mesh_gpu_octree,
        n_hops=N_HOPS,
        l2_search_radius=10,
        enable_l1_search=ENABLE_L1_SEARCH,
        l2_search_method='radius'
    )

    print("  Compiling...")
    positions_baseline = positions_gpu
    element_ids_baseline = ground_truth_element_ids_gpu

    print("\n  DEBUG: Arguments for baseline compilation:")
    print(f"    positions_baseline: {positions_baseline.shape}, dtype={positions_baseline.dtype}")
    print(f"    element_ids_baseline: {element_ids_baseline.shape}, dtype={element_ids_baseline.dtype}")
    print(f"    DT: {DT}")
    print(f"    velocity_sequence_gpu: {velocity_sequence_gpu.shape}, dtype={velocity_sequence_gpu.dtype}")
    print(f"    time_idx: 0")
    print()

    t_compile = time.time()
    positions_baseline, element_ids_baseline = rk4_step_baseline(
        positions_baseline,
        element_ids_baseline,
        DT,
        velocity_sequence_gpu,
        0
    )
    positions_baseline = jax.block_until_ready(positions_baseline)
    element_ids_baseline = jax.block_until_ready(element_ids_baseline)
    t_compile = time.time() - t_compile
    print(f"  Compilation time: {t_compile:.2f}s")

    # Setup VTK export for baseline
    baseline_exporter = VTKExportThread(OUTPUT_DIR / "baseline")
    baseline_exporter.start()

    print(f"\n  Running {N_STEPS} RK4 steps...")
    print(f"  {'Step':>6s} {'Active':>10s} {'Retention':>10s} {'Time':>10s} {'Throughput':>12s} | Exported")
    print("  " + "-" * 70)

    step_times_baseline = []
    retention_history_baseline = []
    t_integration_start = time.time()

    for step in range(1, N_STEPS + 1):
        t_step = time.time()

        positions_baseline, element_ids_baseline = rk4_step_baseline(
            positions_baseline,
            element_ids_baseline,
            DT,
            velocity_sequence_gpu,
            step
        )

        positions_baseline = jax.block_until_ready(positions_baseline)
        element_ids_baseline = jax.block_until_ready(element_ids_baseline)

        t_step = time.time() - t_step
        step_times_baseline.append(t_step)

        n_active = int(jnp.sum(element_ids_baseline >= 0))
        retention = (n_active / n_particles) * 100
        retention_history_baseline.append(retention)

        throughput = n_particles / t_step

        # Export at intervals
        if step % EXPORT_FREQUENCY == 0:
            positions_cpu = np.array(positions_baseline, dtype=np.float32)
            element_ids_cpu = np.array(element_ids_baseline, dtype=np.int32)
            baseline_exporter.enqueue_export(step, positions_cpu, element_ids_cpu)

        # Log at intervals
        if step % LOG_INTERVAL == 0 or step == N_STEPS:
            export_stats = baseline_exporter.get_stats()
            print(f"  {step:6d} {n_active:10,} {retention:9.2f}% {t_step*1000:10.2f} ms {throughput:12.0f} p/s | {export_stats['n_exported']:>4}")

    t_baseline = time.time() - t_integration_start

    print("\n  Waiting for exports to complete...")
    baseline_exporter.stop()
    export_stats = baseline_exporter.get_stats()
    print(f"  ✅ Baseline exports complete: {export_stats['n_exported']} files")

    final_active_baseline = int(jnp.sum(element_ids_baseline >= 0))
    final_retention_baseline = (final_active_baseline / n_particles) * 100

    print(f"\n  Baseline Results:")
    print(f"    Total time: {t_baseline:.2f}s")
    print(f"    Final retention: {final_retention_baseline:.2f}%")
    print(f"    Mean throughput: {n_particles * N_STEPS / t_baseline:,.0f} p/s")

    # ========================================================================
    # Run Mesh-Aligned Multi-Cell + 2×2×2 Local Method
    # ========================================================================

    print("\n[8/9] Testing Mesh-Aligned Multi-Cell + 2×2×2 Local...")
    print("=" * 80)

    config.L2_SEARCH_METHOD = 'mesh_aligned_octree'

    print("\n  DEBUG: Array shapes before multi-cell RK4 creation:")
    print(f"    positions_gpu: {positions_gpu.shape}")
    print(f"    ground_truth_element_ids_gpu: {ground_truth_element_ids_gpu.shape}")
    print(f"    velocity_sequence_gpu: {velocity_sequence_gpu.shape}")
    print(f"    DT: {DT}")
    print(f"    mesh_gpu.connectivity: {mesh_gpu.connectivity.shape}")
    print(f"    mesh_gpu.node_positions: {mesh_gpu.node_positions.shape}")
    print(f"    mesh_gpu.element_neighbors: {mesh_gpu.element_neighbors.shape}")
    print(f"    element_volumes_gpu: {element_volumes_gpu.shape}")
    print(f"    mesh_gpu_octree.elem_ids_sorted: {mesh_gpu_octree.elem_ids_sorted.shape}")
    print(f"    mesh_gpu_octree.morton_sorted: {mesh_gpu_octree.morton_sorted.shape}")
    print(f"    mesh_gpu_octree.leaf_start: {mesh_gpu_octree.leaf_start.shape}")
    print(f"    mesh_gpu_octree.leaf_length: {mesh_gpu_octree.leaf_length.shape}")
    print(f"    mesh_aligned_octree_multi_gpu.cell_to_elements_offsets: {mesh_aligned_octree_multi_gpu.cell_to_elements_offsets.shape}")
    print(f"    mesh_aligned_octree_multi_gpu.cell_to_elements_data: {mesh_aligned_octree_multi_gpu.cell_to_elements_data.shape}")
    print(f"    mesh_aligned_octree_multi_gpu.cell_morton_codes: {mesh_aligned_octree_multi_gpu.cell_morton_codes.shape}")
    print(f"    mesh_aligned_octree_multi_gpu.cell_levels: {mesh_aligned_octree_multi_gpu.cell_levels.shape}")
    print(f"    mesh_aligned_octree_multi_gpu.cell_sizes: {mesh_aligned_octree_multi_gpu.cell_sizes.shape}")
    print(f"    mesh_aligned_octree_multi_gpu.cell_grid_indices: {mesh_aligned_octree_multi_gpu.cell_grid_indices.shape}")
    print(f"    mesh_aligned_octree_multi_gpu.n_cells: {mesh_aligned_octree_multi_gpu.n_cells}")
    print()

    print("  Creating RK4 function...")
    rk4_step_multi = create_rk4_fully_fused_timedep(
        mesh_gpu_connectivity=mesh_gpu.connectivity,
        mesh_gpu_node_positions=mesh_gpu.node_positions,
        mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
        mesh_gpu_element_volumes=element_volumes_gpu,
        mesh_gpu_global_morton=mesh_gpu_octree,
        n_hops=N_HOPS,
        enable_l1_search=ENABLE_L1_SEARCH,
        l2_search_method='radius',
        mesh_aligned_octree=mesh_aligned_octree_multi_gpu,
        mesh_aligned_octree_use_multi_local=True
    )

    print("  Compiling...")
    positions_multi = positions_gpu
    element_ids_multi = ground_truth_element_ids_gpu

    print("\n  DEBUG: Arguments for multi-cell compilation:")
    print(f"    positions_multi: {positions_multi.shape}, dtype={positions_multi.dtype}")
    print(f"    element_ids_multi: {element_ids_multi.shape}, dtype={element_ids_multi.dtype}")
    print(f"    DT: {DT}")
    print(f"    velocity_sequence_gpu: {velocity_sequence_gpu.shape}, dtype={velocity_sequence_gpu.dtype}")
    print(f"    time_idx: 0")
    print()

    t_compile = time.time()
    positions_multi, element_ids_multi = rk4_step_multi(
        positions_multi,
        element_ids_multi,
        DT,
        velocity_sequence_gpu,
        0
    )
    positions_multi = jax.block_until_ready(positions_multi)
    element_ids_multi = jax.block_until_ready(element_ids_multi)
    t_compile = time.time() - t_compile
    print(f"  Compilation time: {t_compile:.2f}s")

    # Setup VTK export for multi-cell
    multi_exporter = VTKExportThread(OUTPUT_DIR / "multi_cell_local")
    multi_exporter.start()

    print(f"\n  Running {N_STEPS} RK4 steps...")
    print(f"  {'Step':>6s} {'Active':>10s} {'Retention':>10s} {'Time':>10s} {'Throughput':>12s} | Exported")
    print("  " + "-" * 70)

    step_times_multi = []
    retention_history_multi = []
    t_integration_start = time.time()

    for step in range(1, N_STEPS + 1):
        t_step = time.time()

        positions_multi, element_ids_multi = rk4_step_multi(
            positions_multi,
            element_ids_multi,
            DT,
            velocity_sequence_gpu,
            step
        )

        positions_multi = jax.block_until_ready(positions_multi)
        element_ids_multi = jax.block_until_ready(element_ids_multi)

        t_step = time.time() - t_step
        step_times_multi.append(t_step)

        n_active = int(jnp.sum(element_ids_multi >= 0))
        retention = (n_active / n_particles) * 100
        retention_history_multi.append(retention)

        throughput = n_particles / t_step

        # Export at intervals
        if step % EXPORT_FREQUENCY == 0:
            positions_cpu = np.array(positions_multi, dtype=np.float32)
            element_ids_cpu = np.array(element_ids_multi, dtype=np.int32)
            multi_exporter.enqueue_export(step, positions_cpu, element_ids_cpu)

        # Log at intervals
        if step % LOG_INTERVAL == 0 or step == N_STEPS:
            export_stats = multi_exporter.get_stats()
            print(f"  {step:6d} {n_active:10,} {retention:9.2f}% {t_step*1000:10.2f} ms {throughput:12.0f} p/s | {export_stats['n_exported']:>4}")

    t_multi = time.time() - t_integration_start

    print("\n  Waiting for exports to complete...")
    multi_exporter.stop()
    export_stats = multi_exporter.get_stats()
    print(f"  ✅ Multi-cell exports complete: {export_stats['n_exported']} files")

    final_active_multi = int(jnp.sum(element_ids_multi >= 0))
    final_retention_multi = (final_active_multi / n_particles) * 100

    print(f"\n  Multi-Cell Results:")
    print(f"    Total time: {t_multi:.2f}s")
    print(f"    Final retention: {final_retention_multi:.2f}%")
    print(f"    Mean throughput: {n_particles * N_STEPS / t_multi:,.0f} p/s")

    # ========================================================================
    # Final Comparison
    # ========================================================================

    print("\n[9/9] Final Comparison")
    print("=" * 80)
    print(f"\n{'Method':<40s}  {'Retention':>10s}  {'Throughput':>14s}  {'Speedup':>8s}")
    print("-" * 80)

    baseline_throughput = n_particles * N_STEPS / t_baseline
    multi_throughput = n_particles * N_STEPS / t_multi
    speedup = t_baseline / t_multi

    print(f"{'Baseline (radius=10)':<40s}  {final_retention_baseline:9.2f}%  {baseline_throughput:13,.0f} p/s  {'1.00×':>8s}")
    print(f"{'Multi-Cell + 2×2×2 Local':<40s}  {final_retention_multi:9.2f}%  {multi_throughput:13,.0f} p/s  {speedup:7.2f}×")

    retention_improvement = final_retention_multi - final_retention_baseline

    print(f"\n  Retention improvement: {retention_improvement:+.2f}%")
    print(f"  Speedup: {speedup:.2f}×")

    print("\n" + "=" * 80)
    print("Test complete!")
    print(f"VTK files exported to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
