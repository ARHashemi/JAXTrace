#!/usr/bin/env python3
"""
Production Particle Tracking Workflow with Async VTK Export

Implements full-scale particle tracking with:
- 100,000 particles
- 2,500 timesteps
- Async VTK export (no impact on tracking performance)
- Complete resource monitoring
- Based on baseline RK4 (5,855 p/s throughput)

Architecture:
- Main thread: GPU tracking computation
- Background thread: Async VTK file writing
- Queue-based data transfer (minimal memory footprint)
- Export frequency configurable (default: every 10 steps)
"""

import os
import sys
import time
import queue
import threading
import psutil
import numpy as np
import jax
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Force CPU-GPU memory management
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# JAXTrace imports
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.forest.block_grid import create_regular_grid
from jaxtrace.gpu.forest.block_mapper import assign_elements_to_blocks
from jaxtrace.gpu.forest.padded_arrays import build_padded_block_arrays
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search import classify_blocks, incremental_search_batch
from jaxtrace.gpu.search.hash_bucket import build_hash_bucket_arrays
from jaxtrace.gpu.search.initial_assignment import initial_search_batch
from jaxtrace.gpu.particles import ParticleData
from jaxtrace.gpu.tracking.time_integration import rk4_step_with_incremental_search
from jaxtrace.gpu.tracking import batch_interpolate_velocities
from jaxtrace.gpu.batching.block_grouping import group_particles_by_block
from jaxtrace.tracking.seeding import uniform_grid_seeds


@dataclass
class ExportConfig:
    """Configuration for VTK export"""
    output_dir: Path
    export_frequency: int  # Export every N timesteps
    include_velocities: bool = True
    include_metadata: bool = True


@dataclass
class TrackingStats:
    """Per-timestep tracking statistics"""
    step: int
    time: float
    n_active: int
    throughput: float  # particles/second
    search_stats: Dict
    gpu_mem_mb: float
    ram_mb: float


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

                step, positions, velocities, _, _, active_mask = export_data

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
            block_ids = np.array(particle_data.block_ids, dtype=np.int32)
            active_mask = np.array(particle_data.active_mask, dtype=bool)

            # Put in queue (will block if queue is full, preventing memory explosion)
            self.export_queue.put(
                (step, positions, velocities, element_ids, block_ids, active_mask),
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


def get_system_stats() -> Tuple[float, float]:
    """Get current GPU memory and RAM usage"""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=1.0
        )
        if result.returncode == 0:
            gpu_mem_mb = float(result.stdout.strip().split('\n')[0])
        else:
            gpu_mem_mb = 0.0
    except Exception:
        gpu_mem_mb = 0.0

    process = psutil.Process()
    ram_mb = process.memory_info().rss / (1024 ** 2)

    return gpu_mem_mb, ram_mb


print("=" * 80)
print("PRODUCTION PARTICLE TRACKING WITH ASYNC VTK EXPORT")
print("=" * 80)
print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")
print()

# ============================================================================
# Configuration
# ============================================================================
print("=" * 80)
print("CONFIGURATION")
print("=" * 80)
print()

MESH_PATH = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_120.pvtu"

# Particle Generation (Uniform Grid)
PARTICLE_GRID_RESOLUTION = (50, 50, 25)  # Grid resolution in (x, y, z)
PARTICLE_BOUNDS_FRACTION = {
    'x': (0.1, 0.3),  # Use first 30% of domain in X
    'y': (0.0, 1.0),  # Full domain in Y
    'z': (0.0, 1.0),  # Full domain in Z
}

# Time Integration
N_TIMESTEPS = 2500
DT = 0.0025  # seconds

# Mesh Processing
GRID_SIZE = (8, 8, 4)  # 256 blocks for forest-of-octrees

# Export Configuration
EXPORT_FREQUENCY = 10  # Export every 10 timesteps
OUTPUT_DIR = Path("./output/threadeda_global_optimized")  # Separate folder for optimized run
STORE_VELOCITIES = False  # Store particle velocities in VTK (default: off)

# Boundary Conditions
ENABLE_BOUNDARY_DEACTIVATION = True  # Deactivate particles that leave domain

# Performance Mode
# False = Baseline block-wise (5-7k p/s, 17 GB RAM, backwards compatible)
# True  = Global GPU mesh (200-300k p/s, 2 GB RAM, optimized)
USE_GLOBAL_GPU_INTERPOLATION = True

# Global interpolation phase (only used if USE_GLOBAL_GPU_INTERPOLATION=True)
# 1 = Phase 1: Persistent mesh + block-by-block particles (100-150k p/s, safer)
# 2 = Phase 2: Persistent mesh + single batch (200-300k p/s, maximum performance)
GLOBAL_INTERPOLATION_PHASE = 2

# Search optimization (only used if USE_GLOBAL_GPU_INTERPOLATION=True)
# False = Use baseline block-based search (incremental_search_batch)
# True  = Use vectorized search (batch L0/L1, global L2) - Phase 3a
USE_VECTORIZED_SEARCH = True

# GPU-fused RK4 (Phase 3a Part 2)
# False = CPU-orchestrated RK4 with 8 CPU-GPU round trips per timestep
# True  = GPU-fused RK4 with 2 transfers per timestep (4× faster)
USE_GPU_FUSED_RK4 = True

# L1 Neighbor Search Hop Count (only used if USE_GPU_FUSED_RK4=True)
# Number of hops for extended neighbor search (pure GPU, no CPU fallback)
# - 2: ~20 neighbors (95-98% hit rate, ~200k p/s, fastest)
# - 3: ~84 neighbors (98-99.5% hit rate, ~120k p/s, good balance)
# - 4: ~340 neighbors (99.5-99.9% hit rate, ~80k p/s, most thorough)
# Higher hop counts = more particles retained, but use more GPU memory
# Default: 2 (original working value, ~20 neighbors, 95-98% hit rate)
RK4_L1_HOP_COUNT = 2  # Original working value with good performance

print(f"Mesh: {MESH_PATH}")
print(f"Particle grid resolution: {PARTICLE_GRID_RESOLUTION}")
print(f"Particle bounds (fraction): {PARTICLE_BOUNDS_FRACTION}")
print(f"Timesteps: {N_TIMESTEPS:,}")
print(f"dt: {DT} s")
print(f"Total time: {N_TIMESTEPS * DT:.3f} s")
print(f"Grid: {GRID_SIZE} ({np.prod(GRID_SIZE)} blocks)")
print(f"Export frequency: every {EXPORT_FREQUENCY} steps")
print(f"Store velocities: {'Yes' if STORE_VELOCITIES else 'No'}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Boundary deactivation: {'Enabled' if ENABLE_BOUNDARY_DEACTIVATION else 'Disabled'}")
print()
print(f"Performance Mode:")
if USE_GLOBAL_GPU_INTERPOLATION:
    print(f"  Mode: GLOBAL GPU MESH (optimized) - Phase {GLOBAL_INTERPOLATION_PHASE}")
    if GLOBAL_INTERPOLATION_PHASE == 1:
        print(f"  Expected throughput: 100,000-150,000 p/s")
        print(f"  Memory: ~14 GB CPU RAM, ~500 MB GPU")
    else:
        print(f"  Expected throughput: 200,000-300,000 p/s")
        print(f"  Memory: ~2 GB CPU RAM, ~500 MB GPU")
else:
    print(f"  Mode: BLOCKWISE (baseline)")
    print(f"  Expected throughput: 5,000-7,000 p/s")
    print(f"  Memory: ~17 GB CPU RAM, ~2.3 GB GPU")
print()

# ============================================================================
# Load Mesh
# ============================================================================
print("=" * 80)
print("MESH LOADING")
print("=" * 80)
print()

t0 = time.perf_counter()
node_positions, connectivity, velocity_field = load_mesh_from_pvtu(
    Path(MESH_PATH),
    field_name='Displacement'
)
t_load = time.perf_counter() - t0

print(f"✓ Mesh loaded ({t_load:.2f} s):")
print(f"  Nodes: {len(node_positions):,}")
print(f"  Elements: {len(connectivity):,}")
print(f"  Velocity field: {velocity_field.shape}")

# Ensure velocity is 3D and float32
if velocity_field.shape[1] == 2:
    velocity_field = np.column_stack([velocity_field, np.zeros(velocity_field.shape[0])])
velocity_field = velocity_field.astype(np.float32)

print(f"  Velocity magnitude range: [{velocity_field.flatten().min():.6f}, {velocity_field.flatten().max():.6f}] m/s")
print()

# ============================================================================
# Create Forest Structure
# ============================================================================
print("=" * 80)
print("FOREST STRUCTURE")
print("=" * 80)
print()

# Compute bounding box
bbox = np.array([
    node_positions[:, 0].min(), node_positions[:, 0].max(),
    node_positions[:, 1].min(), node_positions[:, 1].max(),
    node_positions[:, 2].min(), node_positions[:, 2].max(),
], dtype=np.float32)

print(f"Bounding box:")
print(f"  X: [{bbox[0]:.4f}, {bbox[1]:.4f}]")
print(f"  Y: [{bbox[2]:.4f}, {bbox[3]:.4f}]")
print(f"  Z: [{bbox[4]:.4f}, {bbox[5]:.4f}]")
print()

# Create block grid
t0 = time.perf_counter()
blocks = create_regular_grid(bbox, GRID_SIZE)
t_grid = time.perf_counter() - t0
print(f"✓ Block grid created ({t_grid:.2f} s): {len(blocks)} blocks")

# Assign elements to blocks
t0 = time.perf_counter()
element_to_block, stats = assign_elements_to_blocks(
    node_positions,
    connectivity,
    bbox,
    GRID_SIZE,
    verbose=False
)
t_assign = time.perf_counter() - t0
print(f"✓ Element assignment ({t_assign:.2f} s)")
print(f"  Blocks used: {stats.n_blocks_used}/{stats.n_blocks}")
print(f"  Elements per block: {stats.min_elements} - {stats.max_elements}")

# Build element neighbors
t0 = time.perf_counter()
element_neighbors = build_element_neighbors_array(connectivity, verbose=False)
t_neighbors = time.perf_counter() - t0
print(f"✓ Element neighbors built ({t_neighbors:.2f} s)")

# Build padded arrays (needed for initial assignment)
t0 = time.perf_counter()
padded_arrays = build_padded_block_arrays(
    element_to_block,
    stats,
    node_positions=node_positions,
    connectivity=connectivity,
    element_neighbors=element_neighbors,
    verbose=False
)
t_padded = time.perf_counter() - t0
print(f"✓ Padded arrays ({t_padded:.2f} s)")
print(f"  Shape: {padded_arrays.block_elements.shape}")
print(f"  Memory: {padded_arrays.memory_mb:.1f} MB")
if USE_VECTORIZED_SEARCH:
    print(f"  Note: Used for initial assignment only, not for incremental search")
print()

# Classify blocks and build hash buckets (needed for initial assignment)
classification = classify_blocks(padded_arrays, threshold=10000, verbose=False)
print(f"✓ Block classification:")
print(f"  Light blocks: {len(classification.light_blocks)}")
print(f"  Heavy blocks: {len(classification.heavy_blocks)}")

# Build hash buckets for heavy blocks
hash_bucket_data = {}
if classification.heavy_blocks:
    print(f"\nBuilding hash buckets for {len(classification.heavy_blocks)} heavy blocks...")
    element_centroids = np.mean(node_positions[connectivity], axis=1).astype(np.float32)

    for block_id in classification.heavy_blocks:
        block_elems = padded_arrays.block_elements[block_id]
        block_count = int(padded_arrays.block_sizes[block_id])
        elem_ids = block_elems[:block_count]
        elem_ids = elem_ids[elem_ids >= 0]

        if len(elem_ids) == 0:
            continue

        centroids = element_centroids[elem_ids]
        block_bounds = blocks[block_id].bounds

        hash_arrays = build_hash_bucket_arrays(
            block_id=block_id,
            element_ids=elem_ids,
            element_centroids=centroids,
            block_bounds=block_bounds,
            target_bucket_size=200,
            morton_bits=10
        )

        hash_bucket_data[block_id] = hash_arrays

# Compute block neighbors
block_neighbors_26 = np.array([b.neighbors_26 for b in blocks], dtype=np.int32)

print()

# ============================================================================
# Generate Particles
# ============================================================================
print("=" * 80)
print("PARTICLE GENERATION (UNIFORM GRID)")
print("=" * 80)
print()

# Compute domain bounds
domain_min = np.array([bbox[0], bbox[2], bbox[4]], dtype=np.float32)
domain_max = np.array([bbox[1], bbox[3], bbox[5]], dtype=np.float32)
domain_size = domain_max - domain_min

print(f"Domain bounds:")
print(f"  X: [{domain_min[0]:.4f}, {domain_max[0]:.4f}] (size: {domain_size[0]:.4f})")
print(f"  Y: [{domain_min[1]:.4f}, {domain_max[1]:.4f}] (size: {domain_size[1]:.4f})")
print(f"  Z: [{domain_min[2]:.4f}, {domain_max[2]:.4f}] (size: {domain_size[2]:.4f})")
print()

# Compute particle bounds from fractions
par_bounds_min = np.zeros(3, dtype=np.float32)
par_bounds_max = np.zeros(3, dtype=np.float32)

for i, axis in enumerate(['x', 'y', 'z']):
    min_frac, max_frac = PARTICLE_BOUNDS_FRACTION[axis]
    par_bounds_min[i] = domain_min[i] + min_frac * domain_size[i]
    par_bounds_max[i] = domain_min[i] + max_frac * domain_size[i]

par_bounds = [par_bounds_min, par_bounds_max]
par_size = par_bounds_max - par_bounds_min

print(f"Particle region (fractional):")
print(f"  X: [{par_bounds_min[0]:.4f}, {par_bounds_max[0]:.4f}] (fraction: {PARTICLE_BOUNDS_FRACTION['x']})")
print(f"  Y: [{par_bounds_min[1]:.4f}, {par_bounds_max[1]:.4f}] (fraction: {PARTICLE_BOUNDS_FRACTION['y']})")
print(f"  Z: [{par_bounds_min[2]:.4f}, {par_bounds_max[2]:.4f}] (fraction: {PARTICLE_BOUNDS_FRACTION['z']})")
print(f"  Particle region size: {par_size}")
print()

# Use grid resolution directly (not dependent on domain size)
nx = max(1, int(PARTICLE_GRID_RESOLUTION[0]))
ny = max(1, int(PARTICLE_GRID_RESOLUTION[1]))
nz = max(1, int(PARTICLE_GRID_RESOLUTION[2]))

N_PARTICLES = nx * ny * nz

print(f"Grid resolution: {nx} × {ny} × {nz} = {N_PARTICLES:,} particles")
print()

# Generate uniform grid
print(f"Generating uniform grid particles...")
particle_positions = uniform_grid_seeds(
    resolution=(nx, ny, nz),
    bounds=par_bounds,
    include_boundaries=True
)

print(f"✓ Generated {len(particle_positions):,} particles on uniform grid")
print()

# ============================================================================
# Upload Mesh to GPU (if using global interpolation)
# ============================================================================
if USE_GLOBAL_GPU_INTERPOLATION:
    from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu, estimate_mesh_memory_mb

    print("=" * 80)
    print("GPU MESH UPLOAD")
    print("=" * 80)
    print()

    mesh_memory_mb = estimate_mesh_memory_mb(len(connectivity), len(node_positions))
    print(f"Estimated mesh memory: {mesh_memory_mb:.2f} MB")

    mesh_gpu = upload_mesh_to_gpu(
        connectivity,
        node_positions,
        element_neighbors,
        verbose=True
    )

    # Upload velocity field to GPU ONCE (avoid repeated uploads per timestep)
    if USE_GPU_FUSED_RK4:
        print(f"Uploading velocity field to GPU...")
        velocity_field_gpu = jax.device_put(velocity_field.astype(np.float32))
        print(f"✓ Velocity field uploaded to GPU: {velocity_field.shape}")
    print()

# ============================================================================
# Initial Assignment
# ============================================================================
print("=" * 80)
print("INITIAL ASSIGNMENT")
print("=" * 80)
print()

print(f"Finding containing elements...")
t0 = time.perf_counter()

# For initial assignment, always use baseline search (global search too slow for large meshes)
# Vectorized search will be used for incremental search during time-stepping
if padded_arrays is not None and classification is not None and block_neighbors_26 is not None:
    # Use baseline search
    element_ids, block_ids, search_stats = initial_search_batch(
        particle_positions,
        bbox,
        GRID_SIZE,
        classification,
        padded_arrays,
        block_neighbors_26,
        hash_bucket_data,
        node_positions,
        connectivity,
        verbose=False
    )
else:
    raise RuntimeError("Cannot perform initial search: neither vectorized nor baseline search is configured")

t_search = time.perf_counter() - t0

found_mask = element_ids >= 0
n_found = found_mask.sum()

print(f"✓ Initial assignment ({t_search:.2f} s):")
print(f"  Found: {n_found}/{N_PARTICLES} ({100*n_found/N_PARTICLES:.1f}%)")
print(f"  Throughput: {n_found/t_search:.1f} p/s")
print()

# Create ParticleData
particle_velocities = np.zeros((n_found, 3), dtype=np.float32)
particle_data = ParticleData(
    positions=particle_positions[found_mask],
    velocities=particle_velocities,
    element_ids=element_ids[found_mask],
    block_ids=block_ids[found_mask],
    active_mask=np.ones(n_found, dtype=bool)
)

print(f"Active particles: {particle_data.n_active:,}")
print()

# ============================================================================
# Prepare for Tracking
# ============================================================================
print("=" * 80)
print("TRACKING SETUP")
print("=" * 80)
print()

# Prepare velocity field for block-wise interpolation (n_blocks × n_nodes × 3)
n_blocks = len(blocks)
velocity_field_all_blocks = np.tile(velocity_field, (n_blocks, 1, 1)).astype(np.float32)

print(f"✓ Velocity field prepared for block-wise tracking")
print(f"  Source: Loaded from mesh file ('Displacement')")
print(f"  Shape: {velocity_field_all_blocks.shape} (n_blocks × n_nodes × 3)")
print(f"  Velocity magnitude range: [{np.linalg.norm(velocity_field, axis=1).min():.6f}, {np.linalg.norm(velocity_field, axis=1).max():.6f}] m/s")
print()

# Create velocity interpolator based on config
if USE_GLOBAL_GPU_INTERPOLATION:
    # Global mesh mode (optimized)
    from jaxtrace.gpu.tracking.velocity_interpolation_global import create_global_interpolator

    # mesh_gpu was already uploaded earlier for initial assignment
    # Create global interpolator
    velocity_interpolator = create_global_interpolator(
        velocity_field,  # Single copy, not per-block
        mesh_gpu,
        padded_arrays=(padded_arrays if GLOBAL_INTERPOLATION_PHASE == 1 else None),
        phase=GLOBAL_INTERPOLATION_PHASE
    )

    print(f"✓ Using GLOBAL MESH interpolator (Phase {GLOBAL_INTERPOLATION_PHASE})")
    if GLOBAL_INTERPOLATION_PHASE == 1:
        print(f"  Architecture: Persistent GPU mesh + block-by-block particles")
        print(f"  Expected speedup: 20-30× over baseline")
    else:
        print(f"  Architecture: Persistent GPU mesh + single batch")
        print(f"  Expected speedup: 40-60× over baseline")
    print()

if not USE_GLOBAL_GPU_INTERPOLATION:
    # Baseline block-wise mode (preserved for compatibility)
    from jaxtrace.gpu.tracking.velocity_interpolation_blockwise import create_blockwise_interpolator

    # Upload mesh to GPU (for baseline interpolator)
    connectivity_gpu = jax.device_put(connectivity)
    node_positions_gpu = jax.device_put(node_positions)

    velocity_interpolator = create_blockwise_interpolator(
        velocity_field_all_blocks,
        padded_arrays,
        connectivity_gpu,
        node_positions_gpu
    )
    print(f"✓ Using BLOCKWISE interpolator (baseline)")
    print(f"  Note: This mode has known bottlenecks (4.9 GB CPU-GPU transfers per RK4)")
    print(f"  For better performance, set USE_GLOBAL_GPU_INTERPOLATION = True")
    print()

# Create incremental search function (conditional on configuration)
# Note: When USE_GPU_FUSED_RK4=True, these functions are not used (GPU-fused RK4 handles its own search)
if not USE_GPU_FUSED_RK4 and USE_GLOBAL_GPU_INTERPOLATION and USE_VECTORIZED_SEARCH:
    # Phase 3a: Hybrid vectorized L0/L1 + block-based L2 fallback
    from jaxtrace.gpu.search.incremental_search_vectorized import incremental_search_vectorized

    def incremental_searcher(new_positions, cached_elem_ids, cached_block_ids):
        """
        Hybrid incremental search (Phase 3a - Option A):
        1. Vectorized L0/L1 for all particles (fast, handles 80-90%)
        2. Block-based L2 fallback for L0/L1 misses (handles remaining 10-20%)
        """
        # Step 1: Vectorized L0/L1 for all particles
        element_ids, block_ids, search_stats_vec = incremental_search_vectorized(
            new_positions,
            cached_elem_ids,
            cached_block_ids,
            mesh_gpu,
            element_neighbors=element_neighbors,
            use_global_l2=False,  # Don't use slow global L2 in vectorized search
            verbose=False
        )

        # Step 2: Find particles that need L2 (missed L0 and L1)
        unmapped_mask = element_ids < 0
        n_unmapped = unmapped_mask.sum()

        if n_unmapped > 0:
            # Fall back to L2/L3 block search for unmapped particles (skip redundant L0/L1)
            from jaxtrace.gpu.search import initial_search_batch

            elem_ids_fallback, block_ids_fallback, search_stats_fallback = initial_search_batch(
                new_positions[unmapped_mask],
                bbox,
                GRID_SIZE,
                classification,
                padded_arrays,
                block_neighbors_26,
                hash_bucket_data,
                node_positions,
                connectivity,
                verbose=False
            )

            # Update unmapped particles with fallback results
            element_ids[unmapped_mask] = elem_ids_fallback
            block_ids[unmapped_mask] = block_ids_fallback

            # Merge statistics
            from jaxtrace.gpu.search.incremental_search_vectorized import VectorizedSearchStats
            search_stats = VectorizedSearchStats(
                n_particles=len(new_positions),
                n_found=(element_ids >= 0).sum(),
                l0_hits=search_stats_vec.l0_hits,
                l1_hits=search_stats_vec.l1_hits,
                l2_hits=search_stats_fallback.l2_hits + search_stats_fallback.l3_hits,
                l0_time=search_stats_vec.l0_time,
                l1_time=search_stats_vec.l1_time,
                l2_time=search_stats_fallback.total_search_time,
                total_time=search_stats_vec.total_time + search_stats_fallback.total_search_time
            )
        else:
            # All particles found via L0/L1, no fallback needed
            search_stats = search_stats_vec

        return element_ids, block_ids, search_stats

    print("✓ Using HYBRID incremental search (Phase 3a - Option A+D optimized)")
    print("  Architecture: Vectorized L0 + Extended L1 (2-hop, ~20 neighbors)")
    print("  Expected: 95%+ via vectorized path (L0+L1 extended), <5% L2/L3 fallback")
    print("  Optimizations: Extended neighborhood + skip redundant L0/L1 in fallback")
    print()

elif not USE_GPU_FUSED_RK4 and padded_arrays is not None and classification is not None:
    # Baseline: Block-based search with padded arrays
    def incremental_searcher(new_positions, cached_elem_ids, cached_block_ids):
        """L0+L1+L2 incremental search (baseline)"""
        return incremental_search_batch(
            new_positions,
            cached_elem_ids,
            cached_block_ids,
            bbox,
            GRID_SIZE,
            classification,
            padded_arrays,
            block_neighbors_26,
            hash_bucket_data,
            node_positions,
            connectivity,
            element_neighbors=element_neighbors,
            verbose=False
        )

    print("✓ Using BASELINE incremental search (block-based)")
    print("  Note: This mode uses padded arrays (6.5 GB CPU + 2 GB GPU transfers)")
    print("  For better performance, set USE_VECTORIZED_SEARCH = True")
    print()

elif USE_GPU_FUSED_RK4:
    # GPU-fused RK4: Doesn't use separate interpolator/searcher functions
    # Create dummy functions to satisfy type checking
    def velocity_interpolator(particle_data, time):
        raise RuntimeError("velocity_interpolator should not be called when USE_GPU_FUSED_RK4=True")

    def incremental_searcher(new_positions, cached_elem_ids, cached_block_ids):
        raise RuntimeError("incremental_searcher should not be called when USE_GPU_FUSED_RK4=True")

    # No message here - will be printed after JIT warm-up in "Display RK4 mode" section

else:
    raise RuntimeError("Cannot create search/interpolation functions: invalid configuration")

if not USE_GPU_FUSED_RK4:
    print("✓ Interpolator and searcher functions created")
    print()

# Display RK4 mode
if USE_GPU_FUSED_RK4:
    # Print search architecture (GPU-fused RK4 uses vectorized L0+L1 internally)
    print("✓ Using HYBRID incremental search (Phase 3a - Option A+D optimized)")
    print(f"  Architecture: Vectorized L0 + Extended L1 ({RK4_L1_HOP_COUNT}-hop, ~20 neighbors)")
    print("  Expected: 95%+ via vectorized path (L0+L1 extended), <5% L2/L3 fallback")
    print("  Optimizations: Extended neighborhood + skip redundant L0/L1 in fallback")
    print()

    print("✓ Interpolator and searcher functions created")
    print()

    print("✓ Using GPU-FUSED RK4 (Phase 3a Part 2)")
    print("  Architecture: All 4 RK4 stages execute on GPU")
    print("  Transfer reduction: 8 round trips → 2 transfers per timestep")
    print("  Expected throughput: 50-100k p/s (4-8× improvement)")
    print()
else:
    print("✓ Using CPU-ORCHESTRATED RK4 (baseline)")
    print("  Note: 8 CPU-GPU round trips per timestep")
    print("  For better performance, set USE_GPU_FUSED_RK4 = True")
    print()

# JIT compile RK4 step
print("Warming up JIT compilation...")
t0 = time.perf_counter()

if USE_GPU_FUSED_RK4:
    # Warm up GPU-fused RK4
    from jaxtrace.gpu.tracking.rk4_gpu_fused import rk4_step_gpu_fused_for_production

    _, _ = rk4_step_gpu_fused_for_production(
        particle_data,
        velocity_field_gpu,  # Use GPU-resident velocity field (uploaded once)
        DT,
        mesh_gpu,
        current_time=0.0,
        n_hops=RK4_L1_HOP_COUNT
    )
else:
    # Warm up CPU-orchestrated RK4
    _, _ = rk4_step_with_incremental_search(
        particle_data,
        velocity_interpolator,
        incremental_searcher,
        dt=DT,
        current_time=0.0
    )

t_jit = time.perf_counter() - t0
print(f"✓ JIT warm-up complete ({t_jit:.2f} s)")
print()

# ============================================================================
# Setup Async VTK Export
# ============================================================================
print("=" * 80)
print("ASYNC VTK EXPORT SETUP")
print("=" * 80)
print()

export_config = ExportConfig(
    output_dir=OUTPUT_DIR,
    export_frequency=EXPORT_FREQUENCY,
    include_velocities=STORE_VELOCITIES,
    include_metadata=True
)

exporter = AsyncVTKExporter(export_config, particle_data)
exporter.start()

print(f"✓ Async VTK exporter initialized")
print(f"  Output directory: {OUTPUT_DIR}")
print(f"  Export frequency: every {EXPORT_FREQUENCY} steps")
print(f"  Store velocities: {'Yes' if STORE_VELOCITIES else 'No'}")
print(f"  Expected exports: {N_TIMESTEPS // EXPORT_FREQUENCY}")
print()

# ============================================================================
# Time Marching Loop
# ============================================================================
print("=" * 80)
print("TIME MARCHING")
print("=" * 80)
print()

print(f"Running {N_TIMESTEPS:,} timesteps with dt={DT} s...")
print(f"(Export happens in background thread, no blocking)")
print()

tracking_start = time.perf_counter()
step_times = []
throughputs = []

for step in range(N_TIMESTEPS):
    step_start = time.perf_counter()

    # Perform RK4 time step
    if USE_GPU_FUSED_RK4:
        # GPU-fused RK4: Everything stays on GPU (2 transfers per timestep)
        from jaxtrace.gpu.tracking.rk4_gpu_fused import rk4_step_gpu_fused_for_production

        particle_data, rk4_stats = rk4_step_gpu_fused_for_production(
            particle_data,
            velocity_field_gpu,  # Use GPU-resident velocity field (no repeated uploads)
            DT,
            mesh_gpu,
            current_time=step * DT,
            n_hops=RK4_L1_HOP_COUNT
        )
    else:
        # Baseline: CPU-orchestrated RK4 (8 round trips per timestep)
        particle_data, rk4_stats = rk4_step_with_incremental_search(
            particle_data,
            velocity_interpolator,
            incremental_searcher,
            dt=DT,
            current_time=step * DT
        )

    # Boundary deactivation: Deactivate particles that left domain
    if ENABLE_BOUNDARY_DEACTIVATION:
        positions = particle_data.positions
        out_of_bounds = (
            (positions[:, 0] < bbox[0]) | (positions[:, 0] > bbox[1]) |
            (positions[:, 1] < bbox[2]) | (positions[:, 1] > bbox[3]) |
            (positions[:, 2] < bbox[4]) | (positions[:, 2] > bbox[5])
        )
        particle_data.active_mask &= ~out_of_bounds

    step_time = time.perf_counter() - step_start
    step_times.append(step_time)
    throughput = particle_data.n_active / step_time
    throughputs.append(throughput)

    # Enqueue export (non-blocking if queue has space)
    if (step + 1) % EXPORT_FREQUENCY == 0:
        exporter.enqueue_export(step + 1, particle_data)

    # Progress reporting
    if (step + 1) % 100 == 0:
        elapsed = time.perf_counter() - tracking_start
        avg_throughput = np.mean(throughputs[-100:])
        eta = (N_TIMESTEPS - step - 1) * np.mean(step_times[-100:])
        gpu_mem, ram_mb = get_system_stats()
        export_stats = exporter.get_stats()

        print(f"Step {step+1:>5}/{N_TIMESTEPS} | "
              f"Active: {particle_data.n_active:>6,} | "
              f"Throughput: {avg_throughput:>7.1f} p/s | "
              f"GPU: {gpu_mem:>5.0f} MB | "
              f"RAM: {ram_mb:>6.0f} MB | "
              f"Exported: {export_stats['n_exported']:>4} | "
              f"ETA: {eta/60:.1f} min")

tracking_elapsed = time.perf_counter() - tracking_start

print()
print("=" * 80)
print("TRACKING COMPLETE")
print("=" * 80)
print()

# ============================================================================
# Finalize Export
# ============================================================================
print("Waiting for exports to complete...")
exporter.stop()

export_stats = exporter.get_stats()
print(f"✓ All exports complete")
print(f"  Files exported: {export_stats['n_exported']}")
print(f"  Mean export time: {export_stats['mean_time']:.3f} s")
print(f"  Total export time: {export_stats['total_time']:.1f} s")
print()

# ============================================================================
# Final Statistics
# ============================================================================
print("=" * 80)
print("FINAL STATISTICS")
print("=" * 80)
print()

print(f"Tracking Performance:")
print(f"  Total time: {tracking_elapsed:.1f} s ({tracking_elapsed/60:.1f} min)")
print(f"  Time per step: {np.mean(step_times):.4f} s ± {np.std(step_times):.4f} s")
print(f"  Mean throughput: {np.mean(throughputs):.1f} p/s")
print(f"  Final active particles: {particle_data.n_active:,}")
print()

print(f"Export Performance:")
print(f"  Files written: {export_stats['n_exported']}")
print(f"  Mean write time: {export_stats['mean_time']:.3f} s")
print(f"  Total write time: {export_stats['total_time']:.1f} s")
print(f"  Tracking overhead: {export_stats['total_time']/tracking_elapsed*100:.2f}%")
print()

final_gpu_mem, final_ram = get_system_stats()
print(f"Final Resource Usage:")
print(f"  GPU Memory: {final_gpu_mem:.0f} MB")
print(f"  RAM: {final_ram:.0f} MB")
print()

print("=" * 80)
print("SUCCESS - Production tracking complete!")
print("=" * 80)
print(f"Output files: {OUTPUT_DIR}")
print()
