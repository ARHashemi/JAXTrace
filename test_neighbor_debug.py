#!/usr/bin/env python3
"""
Debug neighbor search - compare with baseline
"""

import time
import numpy as np
from pathlib import Path

import jax
import jax.numpy as jnp

from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import extract_octree_cells_single
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu
from jaxtrace.gpu.search.mesh_aligned_point_location import search_mesh_aligned_octree_batch
from jaxtrace.gpu.search.mesh_aligned_octree_with_neighbor_table import (
    add_neighbor_table_to_octree,
    upload_octree_with_neighbors_to_gpu
)
from jaxtrace.gpu.search.mesh_aligned_search_with_neighbors import (
    search_batch_with_precomputed_neighbors
)
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

print("="*80)
print("Debug: Compare Baseline vs Neighbor Search")
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
print(f"  Mesh: {connectivity.shape[0]:,} elements, {node_positions.shape[0]:,} nodes\n")

# Extract octree
print("Extracting octree...")
t0 = time.time()
octree_cells = extract_octree_cells_single(node_positions, connectivity, verbose=False)
print(f"  ✅ {octree_cells.n_cells:,} cells in {time.time()-t0:.1f}s\n")

# Upload baseline (without neighbors)
print("Uploading baseline octree to GPU...")
octree_baseline_gpu = upload_mesh_aligned_octree_to_gpu(connectivity, node_positions, octree_cells, verbose=False)
print(f"  ✅ Baseline uploaded\n")

# Build neighbor table
print("Building neighbor table...")
t0 = time.time()
octree_with_neighbors = add_neighbor_table_to_octree(octree_cells, verbose=False)
print(f"  ✅ Neighbor table built in {time.time()-t0:.1f}s\n")

# Upload with neighbors
print("Uploading octree with neighbors to GPU...")
octree_neighbors_gpu = upload_octree_with_neighbors_to_gpu(
    connectivity, node_positions, octree_with_neighbors, verbose=False
)
print(f"  ✅ Octree with neighbors uploaded\n")

# Generate test positions
n_particles = 100  # Small test set
np.random.seed(42)
bbox_min = np.array(octree_baseline_gpu.bbox_min)
bbox_max = np.array(octree_baseline_gpu.bbox_max)

test_positions = np.column_stack([
    np.random.uniform(bbox_min[i], bbox_max[i], n_particles) for i in range(3)
]).astype(np.float32)
test_positions_gpu = jnp.array(test_positions)

print(f"Testing {n_particles} random particles...\n")

# Test baseline
print("Test 1: Baseline (primary cell only)")
found_baseline, tests_baseline = search_mesh_aligned_octree_batch(
    test_positions_gpu, octree_baseline_gpu, max_tests=150
)
jax.block_until_ready(found_baseline)

found_baseline_cpu = np.array(found_baseline)
tests_baseline_cpu = np.array(tests_baseline)
n_found_baseline = np.sum(found_baseline_cpu >= 0)
print(f"  Found: {n_found_baseline}/{n_particles} ({100.0*n_found_baseline/n_particles:.1f}%)")
print(f"  Mean tests: {tests_baseline_cpu[found_baseline_cpu >= 0].mean():.1f}\n")

# Test with neighbors
print("Test 2: With pre-computed neighbors")
found_neighbors, tests_neighbors = search_batch_with_precomputed_neighbors(
    test_positions_gpu, octree_neighbors_gpu, levels_to_try=(14,), max_tests_per_cell=20
)
jax.block_until_ready(found_neighbors)

found_neighbors_cpu = np.array(found_neighbors)
tests_neighbors_cpu = np.array(tests_neighbors)
n_found_neighbors = np.sum(found_neighbors_cpu >= 0)
print(f"  Found: {n_found_neighbors}/{n_particles} ({100.0*n_found_neighbors/n_particles:.1f}%)")
print(f"  Mean tests: {tests_neighbors_cpu[found_neighbors_cpu >= 0].mean():.1f}\n")

# Compare
print("="*80)
print("COMPARISON")
print("="*80)
print(f"Baseline:         {n_found_baseline}/{n_particles} ({100.0*n_found_baseline/n_particles:.1f}%)")
print(f"With neighbors:   {n_found_neighbors}/{n_particles} ({100.0*n_found_neighbors/n_particles:.1f}%)")

if n_found_neighbors < n_found_baseline:
    print(f"\n❌ ERROR: Neighbor search found FEWER particles than baseline!")
    print(f"   Expected: neighbor >= baseline")
    print(f"   Got: {n_found_neighbors} < {n_found_baseline}")

    # Show first few mismatches
    print(f"\nFirst 10 mismatches:")
    for i in range(min(10, n_particles)):
        if found_baseline_cpu[i] >= 0 and found_neighbors_cpu[i] < 0:
            print(f"  Particle {i}: baseline found {found_baseline_cpu[i]}, neighbors found {found_neighbors_cpu[i]}")
elif n_found_neighbors == n_found_baseline:
    print(f"\n⚠️  Neighbor search found SAME as baseline (neighbors not helping)")
else:
    print(f"\n✅ Neighbor search found MORE particles (+{n_found_neighbors - n_found_baseline})")

print("="*80)
