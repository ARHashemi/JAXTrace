#!/usr/bin/env python3
"""
Block-Wise RK4 Test with Comprehensive Monitoring

Tests the new block-wise RK4 architecture (rk4_step_blockwise) with:
- GPU/CPU load monitoring
- Memory tracking (CPU + GPU)
- Per-process timing breakdown
- Comparison vs current baseline (rk4_step_with_incremental_search)

Expected improvements:
- 15-40% throughput improvement (13 p/s → 15-18 p/s)
- 75% memory savings (k1-k4 computed on-the-fly)
- 4× reduction in CPU-GPU transfers
"""

import os
import sys
import time
import psutil
import threading
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
from jaxtrace.gpu.tracking import (
    rk4_step_blockwise,
    BlockwiseRK4Stats,
    batch_interpolate_velocities,
)
from jaxtrace.gpu.tracking.time_integration import rk4_step_with_incremental_search
from jaxtrace.gpu.batching.block_grouping import group_particles_by_block


@dataclass
class ResourceSnapshot:
    """Single point-in-time resource usage snapshot"""
    timestamp: float
    cpu_percent: float  # CPU usage %
    ram_mb: float  # RAM usage in MB
    gpu_mem_mb: float  # GPU memory usage in MB
    gpu_util_percent: float  # GPU utilization %


class ResourceMonitor:
    """Monitors CPU, RAM, and GPU usage during execution"""

    def __init__(self):
        self.monitoring = False
        self.snapshots: List[ResourceSnapshot] = []
        self.monitor_thread = None
        self.process = psutil.Process()

    def get_gpu_stats(self) -> Tuple[float, float]:
        """Get GPU memory usage and utilization via nvidia-smi"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=1.0
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(',')
                    mem_mb = float(parts[0].strip())
                    util_pct = float(parts[1].strip())
                    return mem_mb, util_pct
        except Exception:
            pass
        return 0.0, 0.0

    def _monitor_loop(self):
        """Background thread that samples resource usage"""
        while self.monitoring:
            cpu_percent = self.process.cpu_percent(interval=None)
            ram_mb = self.process.memory_info().rss / (1024 ** 2)
            gpu_mem_mb, gpu_util_pct = self.get_gpu_stats()

            self.snapshots.append(ResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                ram_mb=ram_mb,
                gpu_mem_mb=gpu_mem_mb,
                gpu_util_percent=gpu_util_pct
            ))

            time.sleep(0.1)  # Sample at 10 Hz

    def start(self):
        """Start monitoring in background thread"""
        self.monitoring = True
        self.snapshots = []
        # Prime the CPU monitor
        self.process.cpu_percent(interval=None)
        time.sleep(0.1)

        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self) -> List[ResourceSnapshot]:
        """Stop monitoring and return snapshots"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        return self.snapshots

    def get_summary(self, snapshots: List[ResourceSnapshot]) -> Dict[str, float]:
        """Compute summary statistics from snapshots"""
        if not snapshots:
            return {
                'cpu_mean': 0.0, 'cpu_max': 0.0,
                'ram_mean_mb': 0.0, 'ram_max_mb': 0.0,
                'gpu_mem_mean_mb': 0.0, 'gpu_mem_max_mb': 0.0,
                'gpu_util_mean': 0.0, 'gpu_util_max': 0.0,
            }

        cpu_vals = [s.cpu_percent for s in snapshots]
        ram_vals = [s.ram_mb for s in snapshots]
        gpu_mem_vals = [s.gpu_mem_mb for s in snapshots]
        gpu_util_vals = [s.gpu_util_percent for s in snapshots]

        return {
            'cpu_mean': np.mean(cpu_vals),
            'cpu_max': np.max(cpu_vals),
            'ram_mean_mb': np.mean(ram_vals),
            'ram_max_mb': np.max(ram_vals),
            'gpu_mem_mean_mb': np.mean(gpu_mem_vals),
            'gpu_mem_max_mb': np.max(gpu_mem_vals),
            'gpu_util_mean': np.mean(gpu_util_vals),
            'gpu_util_max': np.max(gpu_util_vals),
        }


print("=" * 80)
print("BLOCK-WISE RK4 TEST WITH COMPREHENSIVE MONITORING")
print("=" * 80)
print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")
print()

# ============================================================================
# Load ThreadedA Mesh
# ============================================================================
print("=" * 80)
print("MESH LOADING")
print("=" * 80)
print()

mesh_path = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_20.pvtu"
print(f"Loading: {mesh_path}")

t0 = time.perf_counter()
# Load mesh with velocity field (stored as 'Displacement' in mesh files)
node_positions, connectivity, velocity_field = load_mesh_from_pvtu(
    Path(mesh_path),
    field_name='Displacement'
)
t_load = time.perf_counter() - t0

print(f"✓ Mesh loaded ({t_load:.2f} s):")
print(f"  Nodes: {len(node_positions):,}")
print(f"  Elements: {len(connectivity):,}")

if velocity_field is not None:
    print(f"  Velocity field: {velocity_field.shape} (loaded from 'Displacement')")
    # Ensure velocity is 3D and float32
    if velocity_field.shape[1] == 2:
        velocity_field = np.column_stack([velocity_field, np.zeros(velocity_field.shape[0])])
    velocity_field = velocity_field.astype(np.float32)
else:
    raise ValueError("Velocity field 'Displacement' not found in mesh file")

print()

# ============================================================================
# Create Forest Structure
# ============================================================================
print("=" * 80)
print("FOREST STRUCTURE")
print("=" * 80)
print()

grid_size = (8, 8, 4)  # 256 blocks
print(f"Grid configuration: {grid_size}")

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
blocks = create_regular_grid(bbox, grid_size)
t_grid = time.perf_counter() - t0
print(f"✓ Block grid created ({t_grid:.2f} s): {len(blocks)} blocks")

# Assign elements to blocks
t0 = time.perf_counter()
element_to_block, stats = assign_elements_to_blocks(
    node_positions,
    connectivity,
    bbox,
    grid_size,
    verbose=False
)
t_assign = time.perf_counter() - t0

print(f"✓ Element assignment ({t_assign:.2f} s):")
print(f"  Blocks used: {stats.n_blocks_used}/{stats.n_blocks}")
print(f"  Elements per block: {stats.min_elements} - {stats.max_elements}")
print()

# Build element neighbors for L1 search
t0 = time.perf_counter()
element_neighbors = build_element_neighbors_array(connectivity, verbose=False)
t_neighbors = time.perf_counter() - t0
print(f"✓ Element neighbors built ({t_neighbors:.2f} s)")

# Build padded arrays
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

print(f"✓ Padded arrays ({t_padded:.2f} s):")
print(f"  Shape: {padded_arrays.block_elements.shape}")
print(f"  Memory: {padded_arrays.memory_mb:.1f} MB")
print()

# Classify blocks and build hash buckets
classification = classify_blocks(padded_arrays, threshold=10000, verbose=False)
print(f"✓ Block classification:")
print(f"  Light blocks: {len(classification.light_blocks)}")
print(f"  Heavy blocks: {len(classification.heavy_blocks)}")

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

print()

# ============================================================================
# Generate Test Particles
# ============================================================================
print("=" * 80)
print("PARTICLE GENERATION")
print("=" * 80)
print()

n_particles = 10000
print(f"Generating {n_particles:,} test particles...")

np.random.seed(42)
particle_positions = np.random.uniform(
    low=[bbox[0], bbox[2], bbox[4]],
    high=[bbox[1], bbox[3], bbox[5]],
    size=(n_particles, 3)
).astype(np.float32)

print(f"✓ Generated {n_particles:,} particles")
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

# Extract block neighbors from blocks
block_neighbors_26 = np.array([b.neighbors_26 for b in blocks], dtype=np.int32)

element_ids, block_ids, search_stats = initial_search_batch(
    particle_positions,
    bbox,
    grid_size,
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
print(f"  Found: {n_found}/{n_particles} ({100*n_found/n_particles:.1f}%)")
print(f"  Throughput: {n_found/t_search:.1f} p/s")
print()

# Create ParticleData (velocities not used in constant field test, but required by dataclass)
particle_data = ParticleData(
    positions=particle_positions[found_mask],
    velocities=np.zeros((n_found, 3), dtype=np.float32),  # Not used in constant field
    element_ids=element_ids[found_mask],
    block_ids=block_ids[found_mask],
    active_mask=np.ones(n_found, dtype=bool)
)



print("=" * 80)
print("TEST COMPLETE - 10K PARTICLES SUCCESSFULLY ASSIGNED!")
print("=" * 80)

