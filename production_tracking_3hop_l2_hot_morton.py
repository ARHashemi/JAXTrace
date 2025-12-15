#!/usr/bin/env python3
"""
Production Particle Tracking - 3-Hop L1 + HOT Morton L2 Fallback

Production test for HOT Morton architecture solving JAX OOM issue from Phase 2:
- 105,000 particles (uniform grid: 50x70x30)
- 2,500 timesteps
- Three-tier search: L0 + L1 (3-hop) + L2 (HOT Morton with local connectivity)
- Async VTK export (no tracking impact)
- Search accuracy validation

HOT Morton Architecture (Phase 2 Improved):
- L0: Check cached elements (85-95% hit rate)
- L1: Hierarchical 3-hop neighbor search (99.9% cumulative)
- L2: HOT Morton with LOCAL connectivity (OOM-safe, 99.99% cumulative)
  * Cube-aligned block partitioning (8×8×4 grid)
  * Global Morton sorting per block
  * Octree leaf structure with bounded capacity (256 elements/leaf)
  * LOCAL connectivity per leaf (CRITICAL innovation):
    - Pre-compute unique nodes per leaf during CPU preprocessing
    - Build local connectivity: element → local node indices
    - GPU search accesses ONLY local arrays (no global mesh indexing)
  * Memory: ~100-800 MB (vs 8 MB Phase 2, but OOM-safe vs 4.88 TiB crash)

Key Difference from Phase 2:
- Phase 2: connectivity[elem_id] inside vmap → 4.88 TiB OOM
- HOT Morton: leaf_local_connectivity[leaf_id] → Fixed-size local array → OOM-safe

Expected Performance:
- Hit rate: >99.95% (L0+L1+L2)
- Retention: >95% at 2,500 steps
- Throughput: 40-50k p/s
- Memory overhead: ~100-800 MB (acceptable trade-off for OOM safety)
"""

import os
import sys
import time
import queue
import threading
import psutil
import numpy as np
import jax
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple

# Force CPU-GPU memory management
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production_3hop_hot_morton.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# JAXTrace imports
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.forest.block_grid import create_regular_grid
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.particles import ParticleData
from jaxtrace.tracking.seeding import uniform_grid_seeds

# HOT Morton imports
from jaxtrace.gpu.search.hot_morton_builder import build_hot_morton_structures
from jaxtrace.gpu.search.hot_morton_search import (
    upload_hot_morton_structures_to_gpu,
    create_level2_hot_morton_search_unconditional,
    compute_block_id_from_position_hot
)
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.tracking.rk4_gpu_fused import create_rk4_step_gpu_fused_for_production_with_hot_morton


@dataclass
class ExportConfig:
    """Configuration for VTK export"""
    output_dir: Path
    export_frequency: int
    include_velocities: bool = False


class AsyncVTKExporter:
    """Async VTK exporter that runs in background thread."""

    def __init__(self, config: ExportConfig, particle_data_template: ParticleData):
        self.config = config
        self.template = particle_data_template
        self.export_queue = queue.Queue(maxsize=5)
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.n_exported = 0
        self.export_times = []
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        """Start background export worker"""
        self.worker_thread = threading.Thread(target=self._export_worker, daemon=True)
        self.worker_thread.start()

    def _export_worker(self):
        """Background thread that processes export queue"""
        from jaxtrace.gpu.particles import export_particles_vtk

        while not self.stop_event.is_set():
            try:
                export_data = self.export_queue.get(timeout=1.0)
                if export_data is None:
                    break

                step, positions, velocities, element_ids, particle_ids, active_mask = export_data

                # Create temporary particle data for export
                particle_data = ParticleData(
                    positions=positions,
                    velocities=velocities,
                    element_ids=element_ids,
                    particle_ids=particle_ids,
                    active_mask=active_mask,
                    block_ids=np.zeros(len(positions), dtype=np.int32)  # Not used for export
                )

                t_export = time.time()
                vtk_file = self.config.output_dir / f"particles_{step:06d}.vtu"
                export_particles_vtk(particle_data, str(vtk_file), include_velocities=self.config.include_velocities)
                t_export = time.time() - t_export

                self.n_exported += 1
                self.export_times.append(t_export)

                if self.n_exported % 100 == 0:
                    avg_time = np.mean(self.export_times[-100:])
                    logger.info(f"[Export] Exported {self.n_exported} files, avg={avg_time:.3f}s")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[Export] Error: {e}")

    def queue_export(self, step: int, particle_data: ParticleData):
        """Queue export data (non-blocking)"""
        try:
            export_data = (
                step,
                particle_data.positions.copy(),
                particle_data.velocities.copy(),
                particle_data.element_ids.copy(),
                particle_data.particle_ids.copy(),
                particle_data.active_mask.copy()
            )
            self.export_queue.put(export_data, block=False)
        except queue.Full:
            logger.warning(f"[Export] Queue full, skipping step {step}")

    def stop(self):
        """Stop background export worker"""
        self.export_queue.put(None)
        self.stop_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=30.0)


# ============================================================================
# Configuration
# ============================================================================

MESH_PATH = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_159.pvtu"
N_TIMESTEPS = 2500
DT = 1e-4
GRID_SIZE = (8, 8, 4)  # Coarse block grid (256 blocks)
RK4_L1_HOP_COUNT = 3  # L1 neighbor hops

# HOT Morton parameters
MAX_ELEMENTS_PER_BLOCK = 50000  # Padding size for block arrays
MAX_LEAF_CAPACITY = 256         # Max elements per octree leaf (JAX bounded loop)
MAX_LOCAL_NODES = 1024          # Max unique nodes per leaf

# Particle seeding
SEED_GRID = (50, 70, 30)  # 105,000 particles

# Export configuration
EXPORT_CONFIG = ExportConfig(
    output_dir=Path("output_hot_morton"),
    export_frequency=100,
    include_velocities=False
)

# Progress reporting
PROGRESS_FREQUENCY = 25  # Print progress every N steps


# ============================================================================
# Main Production Script
# ============================================================================

def main():
    logger.info("=" * 80)
    logger.info("PRODUCTION PARTICLE TRACKING - HOT Morton L2 (OOM-Safe)")
    logger.info("=" * 80)

    # Memory tracking
    process = psutil.Process()
    mem_start = process.memory_info().rss / (1024**3)
    logger.info(f"Initial memory: {mem_start:.2f} GB")

    # ========================================================================
    # 1. Load Mesh
    # ========================================================================

    logger.info("\n[1/8] Loading mesh...")
    t_load = time.time()
    node_positions, connectivity, velocity_field = load_mesh_from_pvtu(
        Path(MESH_PATH),
        field_name='Displacement'
    )
    t_load = time.time() - t_load

    n_nodes = node_positions.shape[0]
    n_elements = connectivity.shape[0]
    logger.info(f"  Mesh: {n_elements:,} elements, {n_nodes:,} nodes")
    logger.info(f"  Load time: {t_load:.2f}s")

    # Ensure velocity is 3D and float32
    if velocity_field.shape[1] == 2:
        velocity_field = np.column_stack([velocity_field, np.zeros(velocity_field.shape[0])])
    velocity_field = velocity_field.astype(np.float32)
    logger.info(f"  Velocity field: {velocity_field.shape}")
    logger.info(f"  Velocity magnitude range: [{velocity_field.flatten().min():.6f}, {velocity_field.flatten().max():.6f}] m/s")

    mem_after_mesh = process.memory_info().rss / (1024**3)
    logger.info(f"  Memory after mesh: {mem_after_mesh:.2f} GB (+{mem_after_mesh - mem_start:.2f} GB)")

    # Domain bounds
    domain_min = node_positions.min(axis=0)
    domain_max = node_positions.max(axis=0)
    domain_bounds = np.array([
        domain_min[0], domain_max[0],
        domain_min[1], domain_max[1],
        domain_min[2], domain_max[2]
    ], dtype=np.float32)

    logger.info(f"  Domain: X=[{domain_min[0]:.3f}, {domain_max[0]:.3f}], "
                f"Y=[{domain_min[1]:.3f}, {domain_max[1]:.3f}], "
                f"Z=[{domain_min[2]:.3f}, {domain_max[2]:.3f}]")

    # ========================================================================
    # 2. Build Element Neighbors
    # ========================================================================

    logger.info("\n[2/8] Building element neighbors...")
    t_neighbors = time.time()
    element_neighbors = build_element_neighbors_array(connectivity)
    t_neighbors = time.time() - t_neighbors
    logger.info(f"  Element neighbors: {element_neighbors.shape}")
    logger.info(f"  Build time: {t_neighbors:.2f}s")

    # ========================================================================
    # 3. Build HOT Morton Structures (CPU Preprocessing)
    # ========================================================================

    logger.info("\n[3/8] Building HOT Morton structures (CPU preprocessing)...")
    t_hot = time.time()

    hot_structures = build_hot_morton_structures(
        node_positions=node_positions,
        connectivity=connectivity,
        grid_size=GRID_SIZE,
        max_elements_per_block=MAX_ELEMENTS_PER_BLOCK,
        max_leaf_capacity=MAX_LEAF_CAPACITY,
        max_local_nodes=MAX_LOCAL_NODES,
        verbose=True
    )

    t_hot = time.time() - t_hot
    logger.info(f"  HOT Morton preprocessing: {t_hot:.2f}s")

    mem_after_hot = process.memory_info().rss / (1024**3)
    logger.info(f"  Memory after HOT Morton: {mem_after_hot:.2f} GB (+{mem_after_hot - mem_after_mesh:.2f} GB)")

    # ========================================================================
    # 4. Upload Mesh and HOT Structures to GPU
    # ========================================================================

    logger.info("\n[4/8] Uploading mesh and HOT structures to GPU...")
    t_upload = time.time()

    # Upload standard mesh
    mesh_gpu = upload_mesh_to_gpu(
        node_positions=node_positions,
        connectivity=connectivity,
        element_neighbors=element_neighbors
    )

    # Upload HOT Morton structures
    mesh_gpu_hot = upload_hot_morton_structures_to_gpu(
        hot_structures,
        domain_bounds=domain_bounds,
        grid_size=GRID_SIZE
    )

    # Upload velocity field
    velocity_field_gpu = jax.device_put(velocity_field.astype(np.float32))

    t_upload = time.time() - t_upload
    logger.info(f"  GPU upload: {t_upload:.2f}s")

    # ========================================================================
    # 5. Seed Particles
    # ========================================================================

    logger.info("\n[5/8] Seeding particles...")
    particle_positions = uniform_grid_seeds(
        domain_min, domain_max, SEED_GRID
    )
    n_particles = particle_positions.shape[0]
    logger.info(f"  Seeded {n_particles:,} particles on {SEED_GRID} grid")

    # ========================================================================
    # 6. Initial Element Assignment (HOT Morton L2 Unconditional)
    # ========================================================================

    logger.info("\n[6/8] Initial element assignment (HOT Morton L2)...")
    t_init = time.time()

    # Create unconditional L2 HOT Morton search
    search_hot_unconditional = create_level2_hot_morton_search_unconditional(mesh_gpu_hot)

    # Upload positions to GPU
    positions_gpu = jax.device_put(particle_positions.astype(np.float32))

    # Search for containing elements
    element_ids_gpu = search_hot_unconditional(positions_gpu)

    # Download results
    element_ids = np.array(element_ids_gpu, dtype=np.int32)

    # Compute block IDs (batch)
    from jaxtrace.gpu.tracking.rk4_gpu_fused import compute_block_ids_batch

    block_ids_gpu = compute_block_ids_batch(
        positions_gpu,
        mesh_gpu_hot.domain_bounds,
        mesh_gpu_hot.grid_size
    )
    block_ids = np.array(block_ids_gpu, dtype=np.int32)

    t_init = time.time() - t_init

    # Validate initial assignment
    n_found = np.sum(element_ids >= 0)
    success_rate = (n_found / n_particles) * 100
    logger.info(f"  Initial assignment: {n_found:,}/{n_particles:,} particles ({success_rate:.2f}%)")
    logger.info(f"  Assignment time: {t_init:.2f}s")

    if success_rate < 95.0:
        logger.warning(f"  WARNING: Initial assignment success rate < 95%!")

    # Create particle data
    particle_data = ParticleData(
        positions=particle_positions,
        velocities=np.zeros_like(particle_positions),
        element_ids=element_ids,
        particle_ids=np.arange(n_particles, dtype=np.int32),
        active_mask=(element_ids >= 0),
        block_ids=block_ids
    )

    logger.info(f"  Active particles: {np.sum(particle_data.active_mask):,}")

    # ========================================================================
    # 7. Create RK4 Wrapper with HOT Morton L2
    # ========================================================================

    logger.info("\n[7/8] Creating RK4 wrapper with HOT Morton L2...")
    rk4_step_func = create_rk4_step_gpu_fused_for_production_with_hot_morton(
        mesh_gpu_hot=mesh_gpu_hot,
        n_hops=RK4_L1_HOP_COUNT
    )
    logger.info(f"  RK4 wrapper created (L1 hops: {RK4_L1_HOP_COUNT})")

    # ========================================================================
    # 8. Time Integration Loop
    # ========================================================================

    logger.info("\n[8/8] Starting time integration loop...")
    logger.info(f"  Timesteps: {N_TIMESTEPS}")
    logger.info(f"  dt: {DT}")
    logger.info(f"  Export frequency: {EXPORT_CONFIG.export_frequency}")

    # Start async VTK exporter
    exporter = AsyncVTKExporter(EXPORT_CONFIG, particle_data)
    exporter.start()

    # Export initial state
    exporter.queue_export(0, particle_data)

    # Time integration statistics
    step_times = []
    retention_history = []
    lost_particles_cumulative = 0

    # Progress tracking
    t_integration_start = time.time()
    current_time = 0.0

    for step in range(1, N_TIMESTEPS + 1):
        t_step = time.time()

        # RK4 step with HOT Morton L2 fallback
        particle_data, rk4_stats = rk4_step_func(
            particle_data,
            velocity_field,
            DT,
            mesh_gpu,
            current_time
        )

        current_time += DT
        t_step = time.time() - t_step
        step_times.append(t_step)

        # Track retention
        n_active = np.sum(particle_data.active_mask)
        retention_pct = (n_active / n_particles) * 100
        retention_history.append(retention_pct)

        # Track lost particles
        lost_this_step = n_particles - n_active - lost_particles_cumulative
        if lost_this_step > 0:
            lost_particles_cumulative += lost_this_step

        # Export if needed
        if step % EXPORT_CONFIG.export_frequency == 0:
            exporter.queue_export(step, particle_data)

        # Progress reporting
        if step % PROGRESS_FREQUENCY == 0 or step == N_TIMESTEPS:
            avg_time = np.mean(step_times[-PROGRESS_FREQUENCY:])
            throughput = n_particles / avg_time if avg_time > 0 else 0

            logger.info(f"  Step {step}/{N_TIMESTEPS}: "
                        f"{n_active:,} active ({retention_pct:.2f}%), "
                        f"{throughput:.1f} p/s, "
                        f"{avg_time*1000:.1f}ms/step")

    t_integration = time.time() - t_integration_start

    # ========================================================================
    # Final Statistics
    # ========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("FINAL STATISTICS")
    logger.info("=" * 80)

    # Retention
    final_active = np.sum(particle_data.active_mask)
    final_retention = (final_active / n_particles) * 100
    logger.info(f"\nRetention:")
    logger.info(f"  Final: {final_active:,}/{n_particles:,} ({final_retention:.2f}%)")
    logger.info(f"  Lost: {lost_particles_cumulative:,}")

    # Throughput
    total_particle_steps = n_particles * N_TIMESTEPS
    avg_step_time = np.mean(step_times)
    throughput = n_particles / avg_step_time
    logger.info(f"\nThroughput:")
    logger.info(f"  Average: {throughput:.1f} particles/second")
    logger.info(f"  Total particle-steps: {total_particle_steps:,}")
    logger.info(f"  Total integration time: {t_integration:.1f}s")
    logger.info(f"  Average step time: {avg_step_time*1000:.1f}ms")

    # Memory
    mem_final = process.memory_info().rss / (1024**3)
    logger.info(f"\nMemory:")
    logger.info(f"  Initial: {mem_start:.2f} GB")
    logger.info(f"  Final: {mem_final:.2f} GB")
    logger.info(f"  Peak increase: {mem_final - mem_start:.2f} GB")

    # Export statistics
    exporter.stop()
    logger.info(f"\nExport:")
    logger.info(f"  Total exports: {exporter.n_exported}")
    if exporter.export_times:
        logger.info(f"  Average export time: {np.mean(exporter.export_times):.3f}s")

    logger.info("\n" + "=" * 80)
    logger.info("HOT Morton Production Run Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
