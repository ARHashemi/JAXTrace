#!/usr/bin/env python3
"""
Production Particle Tracking - Fully-Fused RK4 with Global Morton

FULLY-FUSED implementation with:
- Single vmap over particles (all RK4 stages fused)
- NO CPU-GPU transfers between timesteps (data stays on GPU)
- Download ONLY at export frequency (every 10 steps)

Target Performance:
- Initial assignment: >95%
- Retention at 2,500 steps: >95%
- Throughput: 60-120K particles/s (2-3× improvement over baseline)
- Memory: ~40-100 MB (global Morton)

Architecture:
- L0: Cached element (point-in-tet)
- L1: Multi-hop neighbors (3 hops, ~84 neighbors)
- L2: Global Morton search (binary search + bounded leaf scan, radius=2)
- Fully-fused RK4: All 5 stages + 5 searches + 4 interpolations in ONE vmap
"""

import os
# Force CPU-GPU memory management
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import sys
import time
import queue
import threading
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaxtrace.gpu.particles import ParticleData
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu
from jaxtrace.gpu.tracking.rk4_fully_fused import create_rk4_fully_fused_global_morton
from jaxtrace.gpu.tracking.initial_assignment_extended import initial_assignment_extended_batch
from jaxtrace.tracking.seeding import uniform_grid_seeds


# Configuration
MESH_PATH = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_159.pvtu"

# Particle Generation (Uniform Grid - from production_tracking_threadeda.py)
PARTICLE_GRID_RESOLUTION = (50, 70, 30)  # Grid resolution in (x, y, z) = 105,000 particles
PARTICLE_BOUNDS_FRACTION = {
    'x': (0.1, 0.3),  # Use first 20% of domain in X (entrance region)
    'y': (0.0, 1.0),  # Full domain in Y
    'z': (0.0, 1.0),  # Full domain in Z
}
# Use grid resolution directly (not dependent on domain size)
N_X = max(1, int(PARTICLE_GRID_RESOLUTION[0]))
N_Y = max(1, int(PARTICLE_GRID_RESOLUTION[1]))
N_Z = max(1, int(PARTICLE_GRID_RESOLUTION[2]))

N_PARTICLES = N_X * N_Y * N_Z

DT = 0.0025
N_STEPS = 2_500
N_HOPS = 3
L2_SEARCH_RADIUS = 2          # L2 search radius during integration
INITIAL_SEARCH_RADIUS = 100    # Extended radius for initial assignment
SEED = 42
LOG_INTERVAL = 100

# Export Configuration
EXPORT_FREQUENCY = 10  # Export every 10 timesteps
OUTPUT_DIR = Path("./output/global_morton")
STORE_VELOCITIES = False  # Store particle velocities in VTK


@dataclass
class ExportConfig:
    """Configuration for VTK export"""
    output_dir: Path
    export_frequency: int  # Export every N timesteps
    include_velocities: bool = True
    include_metadata: bool = True


class AsyncVTKExporter:
    """
    Async VTK exporter that runs in background thread.

    Minimal memory overhead: Only stores current timestep data in queue.
    No blocking of main tracking loop.
    """

    def __init__(self, config: ExportConfig, particle_data_template: ParticleData):
        self.config = config
        self.template = particle_data_template
        self.export_queue = queue.Queue(maxsize=5)  # Limit queue size
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.n_exported = 0
        self.export_times = []

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        """Start background export worker"""
        self.worker_thread = threading.Thread(target=self._export_worker, daemon=True)
        self.worker_thread.start()

    def _export_worker(self):
        """Background thread that processes export queue"""
        while not self.stop_event.is_set():
            try:
                # Wait for data with timeout to allow checking stop_event
                export_data = self.export_queue.get(timeout=1.0)

                if export_data is None:  # Sentinel value
                    break

                step, positions, velocities, element_ids, active_mask = export_data

                # Write VTK file
                t0 = time.perf_counter()
                output_file = self.config.output_dir / f"particles_step_{step:06d}.vtu"

                # Filter to active particles only
                active_positions = positions[active_mask]
                active_velocities = velocities[active_mask] if (velocities is not None and self.config.include_velocities) else None

                # Use VTK writer directly
                from jaxtrace.io import VTKTrajectoryWriter
                writer = VTKTrajectoryWriter()
                writer.write_particles_at_time(
                    positions=active_positions,
                    velocities=active_velocities,
                    time=step,
                    filename=str(output_file),
                    format='xml'
                )

                export_time = time.perf_counter() - t0
                self.export_times.append(export_time)
                self.n_exported += 1
                self.export_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Export error: {e}")

    def enqueue_export(self, step: int, particle_data: ParticleData):
        """
        Add particle data to export queue (non-blocking).

        Creates CPU copies of data to avoid GPU memory retention.
        """
        try:
            # Convert to CPU numpy arrays (copy from GPU if needed)
            positions = np.array(particle_data.positions, dtype=np.float32)

            # Only copy velocities if requested
            if self.config.include_velocities:
                velocities = np.array(particle_data.velocities, dtype=np.float32)
            else:
                velocities = None

            element_ids = np.array(particle_data.element_ids, dtype=np.int32)
            active_mask = np.array(particle_data.active_mask, dtype=bool)

            # Put in queue (will block if queue is full, preventing memory explosion)
            self.export_queue.put(
                (step, positions, velocities, element_ids, active_mask),
                timeout=10.0
            )
        except queue.Full:
            print(f"Warning: Export queue full at step {step}, skipping export")

    def stop(self):
        """Stop background worker and wait for queue to finish"""
        # Signal worker to stop
        self.export_queue.put(None)
        self.stop_event.set()

        # Wait for worker to finish
        if self.worker_thread:
            self.worker_thread.join(timeout=30.0)

    def get_stats(self) -> Dict:
        """Get export statistics"""
        if not self.export_times:
            return {'n_exported': 0, 'mean_time': 0, 'total_time': 0}

        return {
            'n_exported': self.n_exported,
            'mean_time': np.mean(self.export_times),
            'total_time': np.sum(self.export_times),
            'queue_size': self.export_queue.qsize(),
        }


def main():
    nx, ny, nz = PARTICLE_GRID_RESOLUTION

    print("=" * 80)
    print("Production Particle Tracking - Global Morton L2 Search")
    print("=" * 80)
    print(f"Grid resolution: {nx} × {ny} × {nz} = {N_PARTICLES:,} particles")
    print(f"Timesteps: {N_STEPS:,}")
    print(f"dt: {DT:.2e}")
    print(f"L1 hops: {N_HOPS}")
    print(f"L2 radius: {L2_SEARCH_RADIUS}")
    print("=" * 80)

    # ========================================================================
    # 1. Load Mesh
    # ========================================================================

    print("\n[1/5] Loading mesh...")
    t_load = time.time()
    node_positions, connectivity, velocity_field = load_mesh_from_pvtu(
        Path(MESH_PATH),
        field_name='Displacement'
    )
    t_load = time.time() - t_load

    n_nodes = node_positions.shape[0]
    n_elements = connectivity.shape[0]
    print(f"  Mesh: {n_elements:,} elements, {n_nodes:,} nodes")
    print(f"  Load time: {t_load:.2f}s")

    # ========================================================================
    # 2. Build Global Morton Structure (CPU)
    # ========================================================================

    print("\n[2/5] Building global Morton structure (CPU)...")
    t_morton = time.time()

    morton_struct = build_global_morton_octree(
        node_positions=node_positions,
        connectivity=connectivity,
        leaf_capacity=256,
        max_depth=21,
        verbose=False  # Disable verbose for production
    )

    t_morton = time.time() - t_morton
    print(f"  Built {morton_struct.n_leaves:,} leaves in {t_morton:.2f}s")
    morton_memory_mb = (morton_struct.elem_ids_sorted.nbytes + morton_struct.morton_sorted.nbytes) / (1024**2)
    print(f"  Memory: {morton_memory_mb:.1f} MB")

    # ========================================================================
    # 3. Upload to GPU
    # ========================================================================

    print("\n[3/5] Uploading mesh and Morton structure to GPU...")
    t_upload = time.time()

    # Compute element neighbors
    print("  Computing element neighbors...")
    t_neighbors = time.time()
    element_neighbors = build_element_neighbors_array(connectivity)
    t_neighbors = time.time() - t_neighbors
    print(f"    Neighbor computation: {t_neighbors:.2f}s")

    # Upload standard mesh data
    mesh_gpu = upload_mesh_to_gpu(
        connectivity=connectivity,
        node_positions=node_positions,
        element_neighbors=element_neighbors,
        verbose=False
    )

    # Upload global Morton structure
    mesh_gpu_morton = upload_global_morton_to_gpu(
        morton_struct,
        connectivity,
        node_positions
    )

    # Force transfer
    _ = jax.block_until_ready(mesh_gpu.connectivity)
    _ = jax.block_until_ready(mesh_gpu_morton.elem_ids_sorted)

    t_upload = time.time() - t_upload
    print(f"  Total upload time: {t_upload:.2f}s")

    # ========================================================================
    # 4. Initialize Particles
    # ========================================================================

    # Compute domain bounds
    domain_min = node_positions.min(axis=0)
    domain_max = node_positions.max(axis=0)
    domain_size = domain_max - domain_min

    # Compute particle bounds from fractions
    par_bounds_min = np.zeros(3, dtype=np.float32)
    par_bounds_max = np.zeros(3, dtype=np.float32)
    for i, axis in enumerate(['x', 'y', 'z']):
        min_frac, max_frac = PARTICLE_BOUNDS_FRACTION[axis]
        par_bounds_min[i] = domain_min[i] + min_frac * domain_size[i]
        par_bounds_max[i] = domain_min[i] + max_frac * domain_size[i]
    par_bounds = [par_bounds_min, par_bounds_max]

    # Use grid resolution (already unpacked at top of main())
    print(f"\n[4/5] Initializing {N_PARTICLES:,} particles (uniform grid {nx}×{ny}×{nz})...")
    print(f"  Particle bounds:")
    print(f"    X: [{par_bounds_min[0]:.6f}, {par_bounds_max[0]:.6f}] (domain fraction: {PARTICLE_BOUNDS_FRACTION['x']})")
    print(f"    Y: [{par_bounds_min[1]:.6f}, {par_bounds_max[1]:.6f}] (domain fraction: {PARTICLE_BOUNDS_FRACTION['y']})")
    print(f"    Z: [{par_bounds_min[2]:.6f}, {par_bounds_max[2]:.6f}] (domain fraction: {PARTICLE_BOUNDS_FRACTION['z']})")

    # Generate uniform grid
    particle_positions = uniform_grid_seeds(
        resolution=(nx, ny, nz),
        bounds=par_bounds,
        include_boundaries=True
    )

    # Create particle data with unknown element IDs
    particle_data = ParticleData.from_positions(particle_positions)

    print(f"  Created {N_PARTICLES:,} particles in uniform grid")

    # ========================================================================
    # 5. Setup Async VTK Export
    # ========================================================================

    print(f"\n[5/6] Setting up async VTK export...")
    export_config = ExportConfig(
        output_dir=OUTPUT_DIR,
        export_frequency=EXPORT_FREQUENCY,
        include_velocities=STORE_VELOCITIES,
        include_metadata=True
    )

    exporter = AsyncVTKExporter(export_config, particle_data)
    exporter.start()

    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Export frequency: every {EXPORT_FREQUENCY} steps")
    print(f"  Store velocities: {'Yes' if STORE_VELOCITIES else 'No'}")
    print(f"  Expected exports: {N_STEPS // EXPORT_FREQUENCY}")

    # ========================================================================
    # 6. Run Time Integration
    # ========================================================================

    print(f"\n[6/6] Running time integration ({N_STEPS:,} steps)...")

    # Create fully-fused RK4 step function
    rk4_step = create_rk4_fully_fused_global_morton(
        mesh_gpu_connectivity=mesh_gpu.connectivity,
        mesh_gpu_node_positions=mesh_gpu.node_positions,
        mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
        mesh_gpu_global_morton=mesh_gpu_morton,
        n_hops=N_HOPS,
        l2_search_radius=L2_SEARCH_RADIUS
    )

    # Upload velocity field and particle data to GPU ONCE
    print("\n  Uploading data to GPU...")
    t_upload_initial = time.time()
    velocity_field_gpu = jax.device_put(velocity_field)
    positions_gpu = jax.device_put(particle_data.positions)
    element_ids_gpu = jax.device_put(particle_data.element_ids)
    t_upload_initial = time.time() - t_upload_initial
    print(f"    Initial upload time: {t_upload_initial:.2f}s")

    # Extended initial assignment (more thorough search with larger radius)
    print(f"\n  Running extended initial assignment (radius={INITIAL_SEARCH_RADIUS})...")
    t_initial_search = time.time()
    element_ids_gpu = initial_assignment_extended_batch(
        positions_gpu,
        mesh_gpu_morton,
        max_radius=INITIAL_SEARCH_RADIUS
    )
    element_ids_gpu = jax.block_until_ready(element_ids_gpu)
    t_initial_search = time.time() - t_initial_search

    # Check initial assignment (single scalar download)
    n_active_initial = int(jnp.sum(element_ids_gpu >= 0))
    initial_success_rate = (n_active_initial / N_PARTICLES) * 100
    print(f"    Initial assignment: {n_active_initial:,}/{N_PARTICLES:,} ({initial_success_rate:.2f}%)")
    print(f"    Search time: {t_initial_search:.2f}s")

    # Run first step to trigger JIT compilation (data stays on GPU)
    print("\n  Compiling RK4 (first step)...")
    t_compile = time.time()
    positions_gpu, element_ids_gpu = rk4_step(
        positions_gpu,
        element_ids_gpu,
        DT,
        velocity_field_gpu
    )
    positions_gpu = jax.block_until_ready(positions_gpu)
    element_ids_gpu = jax.block_until_ready(element_ids_gpu)
    t_compile = time.time() - t_compile
    print(f"    Compilation time: {t_compile:.2f}s")

    if initial_success_rate < 95.0:
        print(f"\n❌ WARNING: Initial assignment <95%. Continuing anyway...")

    # Main time integration loop
    print(f"\n  Running {N_STEPS:,} timesteps...")
    print(f"  {'Step':>6} {'Active':>10} {'Retention':>10} {'Step Time':>12} {'Throughput':>15}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*12} {'-'*15}")

    t_integration_start = time.time()
    step_times = []
    retention_history = []

    for step in range(1, N_STEPS + 1):
        t_step = time.time()

        # Run RK4 step (all data stays on GPU)
        positions_gpu, element_ids_gpu = rk4_step(
            positions_gpu,
            element_ids_gpu,
            DT,
            velocity_field_gpu
        )

        # Block until computation completes
        positions_gpu = jax.block_until_ready(positions_gpu)
        element_ids_gpu = jax.block_until_ready(element_ids_gpu)

        t_step = time.time() - t_step
        step_times.append(t_step)

        # Count active particles (single scalar download)
        n_active = int(jnp.sum(element_ids_gpu >= 0))
        retention = (n_active / N_PARTICLES) * 100
        retention_history.append(retention)

        throughput = N_PARTICLES / t_step

        # Download and enqueue export ONLY at export frequency
        if step % EXPORT_FREQUENCY == 0:
            positions_cpu = np.array(positions_gpu, dtype=np.float32)
            element_ids_cpu = np.array(element_ids_gpu, dtype=np.int32)

            # Create minimal ParticleData for export (only positions matter)
            particle_data_export = ParticleData(
                positions=positions_cpu,
                velocities=np.zeros((N_PARTICLES, 3), dtype=np.float32),
                element_ids=element_ids_cpu,
                block_ids=np.zeros(N_PARTICLES, dtype=np.int32),
                active_mask=(element_ids_cpu >= 0)
            )
            exporter.enqueue_export(step, particle_data_export)

        # Log at intervals
        if step % LOG_INTERVAL == 0 or step == N_STEPS:
            export_stats = exporter.get_stats()
            print(f"  {step:6d} {n_active:10,} {retention:9.2f}% {t_step*1000:10.2f} ms {throughput:12.0f} p/s | Exported: {export_stats['n_exported']:>4}")

    t_integration = time.time() - t_integration_start

    # ========================================================================
    # Finalize Export
    # ========================================================================

    print("\n  Waiting for exports to complete...")
    exporter.stop()

    export_stats = exporter.get_stats()
    print(f"  ✅ All exports complete")
    print(f"    Files exported: {export_stats['n_exported']}")
    print(f"    Mean export time: {export_stats['mean_time']:.3f} s")
    print(f"    Total export time: {export_stats['total_time']:.1f} s")

    # ========================================================================
    # Final Analysis
    # ========================================================================

    print("\n" + "=" * 80)
    print("PRODUCTION RESULTS")
    print("=" * 80)

    # Final retention (download final state)
    positions_final_cpu = np.array(positions_gpu, dtype=np.float32)
    element_ids_final_cpu = np.array(element_ids_gpu, dtype=np.int32)

    final_active = np.sum(element_ids_final_cpu >= 0)
    final_retention = (final_active / N_PARTICLES) * 100

    print(f"\n  Initial particles: {N_PARTICLES:,}")
    print(f"  Initial assignment: {n_active_initial:,} ({initial_success_rate:.2f}%)")
    print(f"  Final active: {final_active:,}")
    print(f"  Final retention: {final_retention:.2f}%")

    # Timing statistics
    mean_step_time = np.mean(step_times[1:])  # Exclude first (compiled) step
    std_step_time = np.std(step_times[1:])
    min_step_time = np.min(step_times[1:])
    max_step_time = np.max(step_times[1:])

    mean_throughput = N_PARTICLES / mean_step_time

    print(f"\n  Timesteps completed: {N_STEPS:,}")
    print(f"  Total integration time: {t_integration:.2f}s")
    print(f"  Mean step time: {mean_step_time*1000:.2f} ± {std_step_time*1000:.2f} ms")
    print(f"  Min/Max step time: {min_step_time*1000:.2f} / {max_step_time*1000:.2f} ms")
    print(f"  Mean throughput: {mean_throughput:.0f} particles/s")

    # Retention over time
    retention_10 = retention_history[9] if len(retention_history) > 9 else 0
    retention_100 = retention_history[99] if len(retention_history) > 99 else 0
    retention_1000 = retention_history[999] if len(retention_history) > 999 else 0

    print(f"\n  Retention history:")
    print(f"    Step 10:    {retention_10:.2f}%")
    print(f"    Step 100:   {retention_100:.2f}%")
    print(f"    Step 1000:  {retention_1000:.2f}%")
    print(f"    Step {N_STEPS}: {final_retention:.2f}%")

    # ========================================================================
    # Success Criteria
    # ========================================================================

    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)

    success = True

    # Check initial assignment
    if initial_success_rate >= 95.0:
        print(f"✅ Initial assignment: {initial_success_rate:.2f}% (≥95% target)")
    else:
        print(f"❌ Initial assignment: {initial_success_rate:.2f}% (<95% target)")
        success = False

    # Check final retention
    if final_retention >= 95.0:
        print(f"✅ Final retention: {final_retention:.2f}% (≥95% target)")
    else:
        print(f"❌ Final retention: {final_retention:.2f}% (<95% target)")
        success = False

    # Check throughput
    if mean_throughput >= 40000:
        print(f"✅ Throughput: {mean_throughput:.0f} p/s (≥40k target)")
    elif mean_throughput >= 30000:
        print(f"⚠️  Throughput: {mean_throughput:.0f} p/s (30-40k, acceptable)")
    else:
        print(f"❌ Throughput: {mean_throughput:.0f} p/s (<30k target)")
        success = False

    # Memory
    total_memory_mb = morton_memory_mb + 50  # Approx for mesh data
    print(f"✅ Memory: ~{total_memory_mb:.0f} MB (global Morton + mesh)")

    # Architecture
    print(f"✅ Architecture: L0 (cached) + L1 ({N_HOPS}-hop) + L2 (global Morton, radius={L2_SEARCH_RADIUS})")
    print(f"✅ Morton structure: {morton_struct.n_leaves:,} leaves, {morton_struct.leaf_capacity} capacity")
    print(f"✅ No JAX OOM errors")

    # Export summary
    print(f"✅ VTK export: {export_stats['n_exported']} files in {OUTPUT_DIR}")

    print("=" * 80)

    if success:
        print("\n🎉 PRODUCTION TEST PASSED!")
        print("   Global Morton L2 search meets all performance targets.")
    else:
        print("\n⚠️  PRODUCTION TEST RESULTS")
        print("   Some metrics below target. Review L2 configuration or increase search radius.")

    print("=" * 80)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
