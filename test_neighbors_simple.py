#!/usr/bin/env python3
"""
Test simplified neighbor search (memory-safe version)
"""

import time
import numpy as np
from pathlib import Path

import jax
import jax.numpy as jnp

from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import extract_octree_cells_single
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu
from jaxtrace.gpu.search.mesh_aligned_neighbors_simple import search_mesh_aligned_with_neighbors_batch
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

print("="*80)
print("Mesh-Aligned Octree with Neighbors (Simple, Memory-Safe Version)")
print("="*80 + "\n")

# Load mesh
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")

print("Loading mesh...")
node_positions, connectivity, _ = load_velocity_sequence_from_pvtu(
    base_path=MESH_BASE_PATH,
    file_pattern="featurelessAvtk_{timestep}.pvtu",
    timestep_range=(158, 159),
    field_name='Displacement',
    verbose=False
)

node_positions, connectivity, _, _ = deduplicate_nodes(
    node_positions, connectivity, velocity_sequence=None, verbose=False
)
print(f"  {connectivity.shape[0]:,} elements, {node_positions.shape[0]:,} nodes\n")

# Extract octree
print("Extracting octree...")
t0 = time.time()
cells = extract_octree_cells_single(node_positions, connectivity, verbose=False)
print(f"  {cells.n_cells:,} cells, {cells.elements_per_cell_mean:.1f} elem/cell, {time.time()-t0:.1f}s\n")

# Upload to GPU
print("Uploading to GPU...")
t0 = time.time()
octree_gpu = upload_mesh_aligned_octree_to_gpu(node_positions, connectivity, cells, verbose=False)
print(f"  Done in {time.time()-t0:.2f}s\n")

# Generate test positions
n_particles = 10000
np.random.seed(42)
bbox_min = np.array(octree_gpu.bbox_min)
bbox_max = np.array(octree_gpu.bbox_max)

# Generate random positions for each dimension separately
test_positions = np.column_stack([
    np.random.uniform(bbox_min[i], bbox_max[i], n_particles) for i in range(3)
]).astype(np.float32)
test_positions_gpu = jnp.array(test_positions)

print(f"Testing {n_particles:,} random particles...\n")

# Warmup
print("Warming up...")
_ = search_mesh_aligned_with_neighbors_batch(
    test_positions_gpu[:10], octree_gpu, levels_to_try=(14,), max_elements_per_cell=20
)
jax.block_until_ready(_)
print("  JIT compiled\n")

# Test at level 14 only (finest) with 27 cells
print("Test 1: Level 14 only + 27 cells (1 primary + 26 neighbors)")
t0 = time.time()
found_1 = search_mesh_aligned_with_neighbors_batch(
    test_positions_gpu, octree_gpu, levels_to_try=(14,), max_elements_per_cell=20
)
jax.block_until_ready(found_1)
t1 = time.time() - t0

found_1_cpu = np.array(found_1)
n_found_1 = np.sum(found_1_cpu >= 0)
print(f"  Found: {n_found_1:,}/{n_particles:,} ({100.0*n_found_1/n_particles:.1f}%)")
print(f"  Time: {t1:.3f}s, Throughput: {n_particles/t1:,.0f} p/s\n")

# Test at levels 14 + 13
print("Test 2: Levels 14 + 13 + 27 cells each")
t0 = time.time()
found_2 = search_mesh_aligned_with_neighbors_batch(
    test_positions_gpu, octree_gpu, levels_to_try=(14, 13), max_elements_per_cell=20
)
jax.block_until_ready(found_2)
t2 = time.time() - t0

found_2_cpu = np.array(found_2)
n_found_2 = np.sum(found_2_cpu >= 0)
print(f"  Found: {n_found_2:,}/{n_particles:,} ({100.0*n_found_2/n_particles:.1f}%)")
print(f"  Time: {t2:.3f}s, Throughput: {n_particles/t2:,.0f} p/s\n")

# Test at levels 14 + 13 + 12
print("Test 3: Levels 14 + 13 + 12 + 27 cells each")
t0 = time.time()
found_3 = search_mesh_aligned_with_neighbors_batch(
    test_positions_gpu, octree_gpu, levels_to_try=(14, 13, 12), max_elements_per_cell=20
)
jax.block_until_ready(found_3)
t3 = time.time() - t0

found_3_cpu = np.array(found_3)
n_found_3 = np.sum(found_3_cpu >= 0)
print(f"  Found: {n_found_3:,}/{n_particles:,} ({100.0*n_found_3/n_particles:.1f}%)")
print(f"  Time: {t3:.3f}s, Throughput: {n_particles/t3:,.0f} p/s\n")

print("="*80)
print("Summary")
print("="*80)
print(f"Baseline (primary cell only): 74.6% @ 12,106 p/s")
print(f"Level 14 + neighbors: {100.0*n_found_1/n_particles:.1f}% @ {n_particles/t1:,.0f} p/s")
print(f"Levels 14+13 + neighbors: {100.0*n_found_2/n_particles:.1f}% @ {n_particles/t2:,.0f} p/s")
print(f"Levels 14+13+12 + neighbors: {100.0*n_found_3/n_particles:.1f}% @ {n_particles/t3:,.0f} p/s")

if n_found_3 / n_particles >= 0.95:
    print(f"\n✅ SUCCESS: {100.0*n_found_3/n_particles:.1f}% searchability >= 95%")
else:
    print(f"\n⚠️  Searchability: {100.0*n_found_3/n_particles:.1f}% < 95%")

print("="*80 + "\n")
