#!/usr/bin/env python3
"""
Production Particle Tracking - Octree Search Only (No L0/L1)

Tests pure octree search performance by bypassing L0 cache and L1 neighbor search.

Search Architecture:
- NO L0: Skip cached element check
- NO L1: Skip hierarchical neighbor search
- L2 ONLY: Direct octree search for all particles

Configuration:
- 105,000 particles (uniform grid: 50x70x30)
- 2,500 timesteps
- GPU-fused RK4 with octree-only search
- Async VTK export (every 10 steps)

Purpose:
- Measure pure octree search performance
- Understand overhead of multilevel search vs direct octree
- Baseline for comparing with L0+L1+L2 architecture
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
from jaxtrace.gpu.search import classify_blocks
from jaxtrace.gpu.search.hash_bucket import build_hash_bucket_arrays
from jaxtrace.gpu.search.initial_assignment import initial_search_batch
from jaxtrace.gpu.search.octree_builder import build_octree_for_level, flatten_octree_to_arrays
from jaxtrace.gpu.particles import ParticleData
from jaxtrace.tracking.seeding import uniform_grid_seeds


@dataclass
class ExportConfig:
    """Configuration for VTK export"""
    output_dir: Path
    export_frequency: int
    include_velocities: bool = True
    include_metadata: bool = True


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
                export_data = self.export_queue.get(timeout=1.0)

                if export_data is None:
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
        """Add particle data to export queue (non-blocking)."""
        try:
            positions = np.array(particle_data.positions, dtype=np.float32)

            if self.config.include_velocities:
                velocities = np.array(particle_data.velocities, dtype=np.float32)
            else:
                velocities = None

            element_ids = np.array(particle_data.element_ids, dtype=np.int32)
            block_ids = np.array(particle_data.block_ids, dtype=np.int32)
            active_mask = np.array(particle_data.active_mask, dtype=bool)

            self.export_queue.put(
                (step, positions, velocities, element_ids, block_ids, active_mask),
                timeout=10.0
            )
        except queue.Full:
            print(f"Warning: Export queue full at step {step}, skipping export")

    def stop(self):
        """Stop background worker and wait for queue to finish"""
        self.export_queue.put(None)
        self.stop_event.set()

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
print("PRODUCTION PARTICLE TRACKING - OCTREE SEARCH ONLY")
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
PARTICLE_GRID_RESOLUTION = (50, 70, 30)
PARTICLE_BOUNDS_FRACTION = {
    'x': (0.1, 0.3),
    'y': (0.0, 1.0),
    'z': (0.0, 1.0),
}

# Time Integration
N_TIMESTEPS = 2500
DT = 0.0025  # seconds

# Mesh Processing
GRID_SIZE = (8, 8, 4)

# Export Configuration
EXPORT_FREQUENCY = 10
OUTPUT_DIR = Path("./output/threadeda_octree_only")
STORE_VELOCITIES = False

# Boundary Conditions
ENABLE_BOUNDARY_DEACTIVATION = True

# Octree Configuration
OCTREE_LEVELSET_THRESHOLD = 1.1
OCTREE_MAX_DEPTH = 15  # Increased from 10 to ensure deep traversal
OCTREE_MAX_LEAF_SIZE = 50

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
print(f"Search Architecture:")
print(f"  L0 (cached): DISABLED")
print(f"  L1 (neighbor): DISABLED")
print(f"  L2 (octree): ENABLED (direct octree search for all particles)")
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
# Create Forest Structure (for initial assignment only)
# ============================================================================
print("=" * 80)
print("FOREST STRUCTURE (Initial Assignment Only)")
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

# ============================================================================
# Build Octree
# ============================================================================
print()
print("=" * 80)
print("OCTREE CONSTRUCTION")
print("=" * 80)
print()

# Load LEVEL field from mesh using VTK
print(f"Loading LEVEL field from mesh...")
import vtk
from vtk.util import numpy_support

reader = vtk.vtkXMLPUnstructuredGridReader()
reader.SetFileName(str(MESH_PATH))
reader.Update()
vtk_mesh = reader.GetOutput()

# Check both cell data and point data for LEVEL
cell_data = vtk_mesh.GetCellData()
point_data = vtk_mesh.GetPointData()

level_field = None

if cell_data.HasArray('LEVEL'):
    level_field = numpy_support.vtk_to_numpy(cell_data.GetArray('LEVEL')).astype(np.float32)
    print(f"✓ Found LEVEL in cell data: {len(level_field):,} elements")
    print(f"  Levelset range: [{level_field.min():.6f}, {level_field.max():.6f}]")
elif point_data.HasArray('LEVEL'):
    print(f"✓ Found LEVEL in point data: {vtk_mesh.GetNumberOfPoints():,} nodes")
    node_level = numpy_support.vtk_to_numpy(point_data.GetArray('LEVEL')).astype(np.float32)
    print(f"  Node levelset range: [{node_level.min():.6f}, {node_level.max():.6f}]")

    print(f"  Computing per-element levelset (max of element's nodes)...")
    level_field = np.array([
        node_level[connectivity[i]].max()
        for i in range(len(connectivity))
    ], dtype=np.float32)
    print(f"✓ Computed element levelset: {len(level_field):,} elements")
    print(f"  Element levelset range: [{level_field.min():.6f}, {level_field.max():.6f}]")

if level_field is None:
    raise RuntimeError("No LEVEL field found in mesh - cannot build octree")

# Compute element centroids
element_centroids = np.array([
    node_positions[connectivity[i]].mean(axis=0)
    for i in range(len(connectivity))
], dtype=np.float32)

element_ids = np.arange(len(connectivity), dtype=np.int32)

# Build octree for refined regions (levelset < threshold)
print()
print(f"Building octree (levelset < {OCTREE_LEVELSET_THRESHOLD})...")
t0 = time.perf_counter()

nodes, metadata = build_octree_for_level(
    element_centroids,
    element_ids,
    level_field=level_field,
    level_threshold=OCTREE_LEVELSET_THRESHOLD,
    max_depth=OCTREE_MAX_DEPTH,
    max_leaf_size=OCTREE_MAX_LEAF_SIZE,
    use_levelset=True
)

t_octree_build = time.perf_counter() - t0

print(f"✓ Octree built ({t_octree_build:.2f} s)")
print(f"  Filtered elements: {metadata['n_elements']:,}/{len(connectivity):,} ({metadata['n_elements']/len(connectivity)*100:.1f}%)")
print(f"  Total nodes: {metadata['n_nodes']:,}")
print(f"  Leaf nodes: {metadata['n_leaves']:,}")
print(f"  Max depth: {metadata['max_depth']}")

# Flatten to GPU-compatible arrays
print()
print("Flattening octree to fixed-size arrays...")
node_metadata_np, node_elements_np = flatten_octree_to_arrays(nodes, max_leaf_size=OCTREE_MAX_LEAF_SIZE)

print(f"  Metadata array: {node_metadata_np.shape} ({node_metadata_np.nbytes / 1024:.1f} KB)")
print(f"  Elements array: {node_elements_np.shape} ({node_elements_np.nbytes / 1024:.1f} KB)")

# Upload to GPU
print()
print("Uploading octree to GPU...")
octree_metadata_gpu = jax.device_put(node_metadata_np)
octree_elements_gpu = jax.device_put(node_elements_np)

print(f"✓ Octree uploaded to GPU")
print(f"  Total octree memory: {(node_metadata_np.nbytes + node_elements_np.nbytes) / (1024**2):.2f} MB")
print()

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
print(f"  Note: Used for initial assignment only")
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
    element_centroids_full = np.mean(node_positions[connectivity], axis=1).astype(np.float32)

    for block_id in classification.heavy_blocks:
        block_elems = padded_arrays.block_elements[block_id]
        block_count = int(padded_arrays.block_sizes[block_id])
        elem_ids = block_elems[:block_count]
        elem_ids = elem_ids[elem_ids >= 0]

        if len(elem_ids) == 0:
            continue

        centroids = element_centroids_full[elem_ids]
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

# Use grid resolution directly
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
# Upload Mesh to GPU
# ============================================================================
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

# Upload velocity field to GPU ONCE
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
# Create Octree-Only RK4 Function
# ============================================================================
print("=" * 80)
print("OCTREE-ONLY RK4 SETUP")
print("=" * 80)
print()

print("Creating octree-only RK4 function...")
print("  This function bypasses L0 (cache) and L1 (neighbor) search")
print("  All particles use direct octree search")
print()

# Import the octree-only RK4 function
from jaxtrace.gpu.tracking.rk4_gpu_fused import create_rk4_step_octree_only

rk4_step_func = create_rk4_step_octree_only(
    octree_metadata=octree_metadata_gpu,
    octree_elements=octree_elements_gpu,
    max_octree_depth=OCTREE_MAX_DEPTH
)

print("✓ Octree-only RK4 function created")
print()

# JIT warm-up
print("Warming up JIT compilation...")
warmup_batch_size = 80000
warmup_data = ParticleData(
    positions=particle_data.positions[:warmup_batch_size],
    velocities=particle_data.velocities[:warmup_batch_size],
    element_ids=particle_data.element_ids[:warmup_batch_size],
    block_ids=particle_data.block_ids[:warmup_batch_size],
    active_mask=particle_data.active_mask[:warmup_batch_size]
)

t0 = time.perf_counter()
_, _ = rk4_step_func(
    warmup_data,
    velocity_field_gpu,
    DT,
    mesh_gpu,
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

# Store initial particle count for consistent throughput calculation
n_total_particles = particle_data.n_active

for step in range(N_TIMESTEPS):
    step_start = time.perf_counter()

    # Perform RK4 time step with octree-only search
    particle_data, rk4_stats = rk4_step_func(
        particle_data,
        velocity_field_gpu,
        DT,
        mesh_gpu,
        current_time=step * DT
    )

    # Boundary deactivation
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
    # Use total particle count for consistent throughput metric
    throughput = particle_data.n_active / step_time
    throughputs.append(throughput)

    # Enqueue export
    if (step + 1) % EXPORT_FREQUENCY == 0:
        exporter.enqueue_export(step + 1, particle_data)

    # Progress reporting
    if (step + 1) % 100 == 0:
        elapsed = time.perf_counter() - tracking_start
        avg_step_time = np.mean(step_times[-100:])
        avg_throughput = np.mean(throughputs[-100:])
        retention_pct = 100 * particle_data.n_active / n_total_particles
        eta = (N_TIMESTEPS - step - 1) * avg_step_time
        gpu_mem, ram_mb = get_system_stats()
        export_stats = exporter.get_stats()

        print(f"Step {step+1:>5}/{N_TIMESTEPS} | "
              f"Active: {particle_data.n_active:>6,} ({retention_pct:>5.1f}%) | "
              f"Time/step: {avg_step_time:>6.3f}s | "
              f"Throughput: {avg_throughput:>7.0f} p/s | "
              f"GPU: {gpu_mem:>5.0f} MB | "
              f"Exported: {export_stats['n_exported']:>3} | "
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
print("SUCCESS - Octree-only tracking complete!")
print("=" * 80)
print(f"Output files: {OUTPUT_DIR}")
print()
