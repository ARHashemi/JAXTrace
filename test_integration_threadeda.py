#!/usr/bin/env python3
"""
Integration Test: Multi-Level Element Search on ThreadedA Mesh

Tests both speed and accuracy with comprehensive monitoring:
- CPU/GPU usage
- Memory consumption
- Search statistics (Level 0/1/2 hit rates)
- Accuracy validation
- Performance benchmarking

This serves as the final validation before GPU JIT compilation.

Author: JAXTrace GPU Team
Date: 2025-11-04
"""

import numpy as np
import sys
import time
import psutil
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/arhashemi/Workspace/welding/JAXTrace')

from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu, assign_elements_to_blocks, build_element_neighbors
from jaxtrace.gpu.octree_builder import build_octrees_per_block
from jaxtrace.gpu.particle_seeding import seed_particles_uniform_grid, SeedingConfig
from jaxtrace.gpu.flat_arrays import MeshData, ParticleData
from jaxtrace.gpu.multi_level_search import find_containing_elements_batch, SearchStatistics

# Try to import GPU monitoring
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: GPUtil not available, GPU monitoring disabled")

# Create log file
log_file = Path("logs/integration_test_threadeda.log")
log_file.parent.mkdir(exist_ok=True)
log_handle = open(log_file, 'w')

def log(message):
    """Log to both console and file with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    log_handle.write(full_message + "\n")
    log_handle.flush()

def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_cpu_usage():
    """Get current CPU usage percentage."""
    return psutil.cpu_percent(interval=0.1)

def get_gpu_usage():
    """Get GPU memory and utilization."""
    if not GPU_AVAILABLE:
        return None, None

    try:
        gpus = GPUtil.getGPUs()
        if len(gpus) == 0:
            return None, None

        gpu = gpus[0]  # Use first GPU
        return gpu.memoryUsed, gpu.load * 100
    except:
        return None, None

def log_system_status(prefix=""):
    """Log current system resource usage."""
    mem_mb = get_memory_usage()
    cpu_pct = get_cpu_usage()
    gpu_mem, gpu_util = get_gpu_usage()

    msg = f"{prefix}Memory: {mem_mb:.1f} MB, CPU: {cpu_pct:.1f}%"
    if gpu_mem is not None:
        msg += f", GPU Mem: {gpu_mem:.1f} MB, GPU Util: {gpu_util:.1f}%"
    log(msg)

# ============================================================================
# MAIN INTEGRATION TEST
# ============================================================================

log("=" * 80)
log("INTEGRATION TEST: Multi-Level Element Search on ThreadedA")
log("=" * 80)
log("")

# Phase 1: Load Mesh
log("Phase 1: Loading ThreadedA mesh")
log("-" * 80)

mesh_path = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_50.pvtu")
field_name = "Temperature"  # Use Temperature instead of velocity

if not mesh_path.exists():
    log(f"ERROR: Mesh file not found: {mesh_path}")
    log("Exiting test.")
    log_handle.close()
    sys.exit(1)

log(f"Mesh path: {mesh_path}")
log_system_status("Before load: ")

t0 = time.time()
positions, connectivity, velocities = load_mesh_from_pvtu(mesh_path, field_name)
t_load = time.time() - t0

log(f"Mesh loaded in {t_load:.1f}s")
log(f"  Nodes: {len(positions):,}")
log(f"  Elements: {len(connectivity):,}")
log(f"  Has velocities: {velocities is not None}")

mesh_size_mb = (positions.nbytes + connectivity.nbytes) / 1024 / 1024
if velocities is not None:
    mesh_size_mb += velocities.nbytes / 1024 / 1024
log(f"  Mesh size: {mesh_size_mb:.1f} MB")

log_system_status("After load: ")
log("")

# Phase 2: Build Neighbors
log("Phase 2: Building element neighbors")
log("-" * 80)

t0 = time.time()
element_neighbors = build_element_neighbors(connectivity)
t_neighbors = time.time() - t0

log(f"Neighbors built in {t_neighbors:.1f}s")

# Count interior elements (have 4 neighbors)
interior_mask = np.all(element_neighbors >= 0, axis=1)
n_interior = np.sum(interior_mask)
log(f"  Interior elements: {n_interior:,} ({100*n_interior/len(connectivity):.1f}%)")

log_system_status("After neighbors: ")
log("")

# Phase 3: Assign to Blocks
log("Phase 3: Assigning elements to 2×2×1 grid")
log("-" * 80)

t0 = time.time()
element_block_IDs, partition_data = assign_elements_to_blocks(
    positions, connectivity, (2, 2, 1), verbose=False
)
t_assign = time.time() - t0

log(f"Assignment completed in {t_assign:.1f}s")
log(f"  Grid size: {partition_data.grid_size}")
log(f"  Bounding box: {partition_data.bbox_min} to {partition_data.bbox_max}")

for block_id in range(4):
    n_elem = np.sum(element_block_IDs == block_id)
    log(f"  Block {block_id}: {n_elem:,} elements ({100*n_elem/len(connectivity):.1f}%)")

log_system_status("After assignment: ")
log("")

# Phase 4: Build Octrees
log("Phase 4: Building octrees per block")
log("-" * 80)

t0 = time.time()
octrees = build_octrees_per_block(
    positions, connectivity, element_block_IDs, partition_data,
    max_elements_per_node=500,
    max_depth=10,
    verbose=False
)
t_octree = time.time() - t0

log(f"Octrees built in {t_octree:.1f}s")
for block_id, octree in octrees.items():
    log(f"  Block {block_id}: {octree.n_nodes} nodes, depth {octree.node_depths.max()}")

log_system_status("After octrees: ")
log("")

# Phase 5: Seed Particles
log("Phase 5: Seeding particles")
log("-" * 80)

# Use a moderate number of particles for testing
config = SeedingConfig(
    bbox_min=partition_data.bbox_min,
    bbox_max=partition_data.bbox_max,
    density_per_axis=(30, 30, 15),  # ~13,500 particles
    seed=42
)

t0 = time.time()
particle_positions = seed_particles_uniform_grid(config)
n_particles = len(particle_positions)
t_seed = time.time() - t0

log(f"Seeded {n_particles:,} particles in {t_seed:.3f}s")
log(f"  Domain: {config.bbox_min} to {config.bbox_max}")
log(f"  Density: {config.density_per_axis}")

log_system_status("After seeding: ")
log("")

# Phase 6: Initial Element Search (for "cached" IDs)
log("Phase 6: Initial element search (establishing cache) - GPU ACCELERATED")
log("-" * 80)

# Use GPU-accelerated batch initial search
from jaxtrace.gpu.initial_search_jax import find_initial_elements_batch, GPUConfig

# Create GPU config (try GPU first, fallback to CPU if needed)
gpu_config = GPUConfig(
    use_gpu_initial_search=True,
    force_cpu=False
)

# Prepare mesh data dict
mesh_data_dict = {
    'positions': positions,
    'connectivity': connectivity
}

t0 = time.time()
log("Searching for initial elements using GPU...")
log_system_status("Before GPU search: ")

initial_element_IDs, search_stats = find_initial_elements_batch(
    particle_positions,
    mesh_data_dict,
    partition_data,
    octrees,
    config=gpu_config,
    verbose=True
)

t_initial = time.time() - t0

n_found = np.sum(initial_element_IDs >= 0)
log(f"Initial search completed in {t_initial:.1f}s")
log(f"  Implementation: {'GPU (JAX)' if search_stats['used_gpu'] else 'CPU (fallback)'}")
log(f"  Found: {n_found:,}/{n_particles:,} ({100*n_found/n_particles:.1f}%)")
log(f"  Not found: {n_particles - n_found:,} (likely outside mesh)")
log(f"  Time per particle: {1000*t_initial/n_particles:.3f} ms")

log_system_status("After initial search: ")
log("")

# Phase 7: Simulate Particle Movement
log("Phase 7: Simulating particle movement")
log("-" * 80)

# Estimate average element size for realistic displacement
log("Estimating average element size...")
sample_elements = min(100, len(connectivity))
element_sizes = []
for i in range(sample_elements):
    verts = positions[connectivity[i]]
    # Estimate as max distance between vertices
    size = np.max([np.linalg.norm(verts[j] - verts[k])
                   for j in range(4) for k in range(j+1, 4)])
    element_sizes.append(size)

avg_element_size = np.mean(element_sizes)
log(f"  Average element size: {avg_element_size:.6f}")

# Use 1% of element size for displacement (realistic CFL ~0.01)
displacement_magnitude = 0.01 * avg_element_size
log(f"  Displacement magnitude: {displacement_magnitude:.6f} (1% of element size)")

# Simulate small random displacement (mimicking advection)
np.random.seed(42)

# Only displace particles that were found
active_mask = initial_element_IDs >= 0
n_active = np.sum(active_mask)

new_positions = particle_positions.copy()
new_positions[active_mask] += np.random.uniform(
    -displacement_magnitude, displacement_magnitude, (n_active, 3)
)

log(f"Displaced {n_active:,} active particles")
log("")

# Phase 8: Multi-Level Element Search
log("Phase 8: Multi-level element search")
log("-" * 80)

# Create MeshData
mesh_data = MeshData(
    positions=positions,
    connectivity=connectivity,
    element_neighbors=element_neighbors,
    element_block_IDs=element_block_IDs,
    velocities=velocities
)

# Create ParticleData with displaced positions but cached element IDs
particle_data = ParticleData(
    positions=new_positions,
    element_IDs=initial_element_IDs,
    active=active_mask
)

log("Running multi-level search...")
log_system_status("Before search: ")

t0 = time.time()
new_element_IDs, stats = find_containing_elements_batch(
    particle_data, mesh_data, partition_data, octrees, verbose=False
)
t_search = time.time() - t0

log(f"Multi-level search completed in {t_search:.1f}s")
log(f"  Time per particle: {1000*t_search/n_particles:.2f} ms")
log("")

# Statistics
log("Search Statistics:")
log("-" * 80)
log(str(stats))
log("")

log_system_status("After search: ")
log("")

# Phase 9: Accuracy Validation
log("Phase 9: Accuracy validation")
log("-" * 80)

# For particles that stayed in their elements (should hit Level 0)
stayed_in_element = new_element_IDs[active_mask] == initial_element_IDs[active_mask]
n_stayed = np.sum(stayed_in_element)
log(f"Particles that stayed in cached element: {n_stayed:,}/{n_active:,} ({100*n_stayed/n_active:.1f}%)")

# Verify found elements are correct (random sample)
log("Verifying accuracy on random sample...")
from jaxtrace.gpu.element_search import point_in_tetrahedron

n_verify = min(1000, n_active)
verify_indices = np.random.choice(np.where(active_mask)[0], n_verify, replace=False)

n_correct = 0
for idx in verify_indices:
    pos = new_positions[idx]
    found_elem = new_element_IDs[idx]

    if found_elem < 0:
        continue

    # Check if position is actually inside found element
    vertices = positions[connectivity[found_elem]]
    if point_in_tetrahedron(pos, vertices):
        n_correct += 1

accuracy = 100 * n_correct / n_verify
log(f"Verified accuracy: {n_correct}/{n_verify} ({accuracy:.1f}%)")

if accuracy >= 95.0:
    log("✅ PASS: Accuracy meets 95% threshold")
else:
    log(f"⚠️  WARNING: Accuracy {accuracy:.1f}% below 95% threshold")

log("")

# Phase 10: Performance Summary
log("Phase 10: Performance summary")
log("-" * 80)

total_time = t_load + t_neighbors + t_assign + t_octree + t_seed + t_initial + t_search

log("Timing breakdown:")
log(f"  Load mesh:          {t_load:8.2f}s ({100*t_load/total_time:5.1f}%)")
log(f"  Build neighbors:    {t_neighbors:8.2f}s ({100*t_neighbors/total_time:5.1f}%)")
log(f"  Assign blocks:      {t_assign:8.2f}s ({100*t_assign/total_time:5.1f}%)")
log(f"  Build octrees:      {t_octree:8.2f}s ({100*t_octree/total_time:5.1f}%)")
log(f"  Seed particles:     {t_seed:8.2f}s ({100*t_seed/total_time:5.1f}%)")
log(f"  Initial search:     {t_initial:8.2f}s ({100*t_initial/total_time:5.1f}%)")
log(f"  Multi-level search: {t_search:8.2f}s ({100*t_search/total_time:5.1f}%)")
log(f"  TOTAL:              {total_time:8.2f}s")
log("")

# Compare search methods
log("Search method comparison:")
log(f"  Initial search (Level 2 only): {1000*t_initial/n_particles:.3f} ms/particle")
log(f"  Multi-level search:            {1000*t_search/n_particles:.3f} ms/particle")
speedup = t_initial / t_search
log(f"  Speedup: {speedup:.2f}×")
log("")

log_system_status("Final state: ")
log("")

# Phase 11: Expected vs Actual Hit Rates
log("Phase 11: Hit rate analysis")
log("-" * 80)

expected_level0 = 85  # Expected for small displacement
expected_level1 = 10
expected_level2 = 5

actual_level0 = stats.hit_rate_level0()
actual_level1 = stats.hit_rate_level1()
actual_level2 = stats.hit_rate_level2()

log("Expected vs Actual hit rates:")
log(f"  Level 0: {actual_level0:5.1f}% (expected ~{expected_level0}%)")
log(f"  Level 1: {actual_level1:5.1f}% (expected ~{expected_level1}%)")
log(f"  Level 2: {actual_level2:5.1f}% (expected ~{expected_level2}%)")
log("")

if actual_level0 >= expected_level0 - 10:
    log("✅ Level 0 hit rate within expected range")
else:
    log(f"⚠️  Level 0 hit rate lower than expected")

log("")

# Final Summary
log("=" * 80)
log("INTEGRATION TEST COMPLETE")
log("=" * 80)
log("")

log("Summary:")
log(f"  Mesh: {len(connectivity):,} elements")
log(f"  Particles: {n_particles:,}")
log(f"  Active particles: {n_active:,}")
log(f"  Multi-level search time: {t_search:.1f}s")
log(f"  Speedup vs pure octree: {speedup:.2f}×")
log(f"  Accuracy: {accuracy:.1f}%")
log(f"  Success rate: {stats.success_rate():.1f}%")
log("")

if accuracy >= 95.0 and stats.success_rate() >= 95.0 and actual_level0 >= 70:
    log("✅ INTEGRATION TEST PASSED")
    log("   - Accuracy meets threshold (≥95%)")
    log("   - Success rate meets threshold (≥95%)")
    log("   - Level 0 hit rate reasonable (≥70%)")
    exit_code = 0
else:
    log("❌ INTEGRATION TEST FAILED")
    if accuracy < 95.0:
        log(f"   - Accuracy {accuracy:.1f}% below 95%")
    if stats.success_rate() < 95.0:
        log(f"   - Success rate {stats.success_rate():.1f}% below 95%")
    if actual_level0 < 70:
        log(f"   - Level 0 hit rate {actual_level0:.1f}% below 70%")
    exit_code = 1

log("")
log(f"Log saved to: {log_file}")
log_handle.close()

sys.exit(exit_code)
