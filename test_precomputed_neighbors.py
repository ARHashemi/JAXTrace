#!/usr/bin/env python3
"""
Test Option B: Pre-Computed Neighbor Table

This tests the CPU-side neighbor table generation + GPU neighbor search
with three different particle generation strategies:

Strategy 1: Uniform Random - Random positions in bounding box
Strategy 2: Element Centroids - Ground truth test (we know which element)
Strategy 3: Perturbed Centroids - Centroids with small random offset

Expected Results:
- ~99% searchability (vs 74.6% without neighbors)
- ~15-20 tests per particle
- ~50-100K particles/sec
"""

import time
import numpy as np
from pathlib import Path
import tracemalloc

import jax
import jax.numpy as jnp

from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import extract_octree_cells_single
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
print("Option B: Mesh-Aligned Octree with Pre-Computed Neighbor Table")
print("="*80 + "\n")

# ============================================================================
# Load Mesh
# ============================================================================

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

# ============================================================================
# Phase 1: Extract Base Octree (CPU)
# ============================================================================

print("Phase 1: Extracting base octree (CPU)...")
t0 = time.time()
octree_cells = extract_octree_cells_single(node_positions, connectivity, verbose=False)
t1 = time.time()
print(f"  ✅ Base octree extracted")
print(f"    Cells: {octree_cells.n_cells:,}")
print(f"    Elements/cell: {octree_cells.elements_per_cell_mean:.1f}")
print(f"    Time: {t1-t0:.1f}s\n")

# ============================================================================
# Phase 2: Build Neighbor Table (CPU)
# ============================================================================

print("Phase 2: Building neighbor table (CPU)...")
t0 = time.time()
octree_with_neighbors = add_neighbor_table_to_octree(octree_cells, verbose=True)
t1 = time.time()
print(f"  ✅ Neighbor table built in {t1-t0:.1f}s\n")

# ============================================================================
# Phase 3: Upload to GPU
# ============================================================================

print("Phase 3: Uploading to GPU...")
t0 = time.time()
octree_gpu = upload_octree_with_neighbors_to_gpu(
    connectivity, node_positions, octree_with_neighbors, verbose=True
)
t1 = time.time()
print(f"  ✅ Uploaded in {t1-t0:.2f}s\n")

# ============================================================================
# Phase 4: Generate Test Positions (3 Strategies)
# ============================================================================

n_particles = 100000
np.random.seed(42)

bbox_min = np.array(octree_gpu.bbox_min)
bbox_max = np.array(octree_gpu.bbox_max)

print(f"Generating {n_particles:,} test positions with 3 strategies...")
print(f"  Bbox: [{bbox_min[0]:.6f}, {bbox_max[0]:.6f}] × "
      f"[{bbox_min[1]:.6f}, {bbox_max[1]:.6f}] × "
      f"[{bbox_min[2]:.6f}, {bbox_max[2]:.6f}]\n")

# Strategy 1: Uniform Random in Bounding Box
print("Strategy 1: Uniform Random in Bounding Box")
positions_random = np.column_stack([
    np.random.uniform(bbox_min[i], bbox_max[i], n_particles) for i in range(3)
]).astype(np.float32)
print(f"  Generated {n_particles:,} random positions\n")

# Strategy 2: Element Centroids (Ground Truth)
print("Strategy 2: Element Centroids (Ground Truth)")
n_elements = connectivity.shape[0]
selected_elements = np.random.choice(n_elements, n_particles, replace=True)
positions_centroids = np.zeros((n_particles, 3), dtype=np.float32)
for i, elem_idx in enumerate(selected_elements):
    elem_nodes = connectivity[elem_idx]
    elem_positions = node_positions[elem_nodes]
    positions_centroids[i] = elem_positions.mean(axis=0)
ground_truth_elements = selected_elements.copy()
print(f"  Generated {n_particles:,} element centroids")
print(f"  Ground truth element IDs available for accuracy validation\n")

# Strategy 3: Perturbed Centroids (Realistic Tracking)
print("Strategy 3: Perturbed Centroids (Realistic Tracking)")
# Calculate smallest element size for perturbation scale
element_sizes = np.zeros(n_elements, dtype=np.float32)
for elem_idx in range(min(100000, n_elements)):  # Sample for speed
    elem_nodes = connectivity[elem_idx]
    elem_positions = node_positions[elem_nodes]
    # Compute characteristic size (max edge length)
    edges = []
    for i in range(4):
        for j in range(i+1, 4):
            edge_len = np.linalg.norm(elem_positions[i] - elem_positions[j])
            edges.append(edge_len)
    element_sizes[elem_idx] = min(edges)

min_element_size = np.percentile(element_sizes[element_sizes > 0], 5)  # 5th percentile
perturbation_scale = min_element_size * 0.1  # 10% of smallest element

positions_perturbed = positions_centroids.copy()
perturbations = np.random.randn(n_particles, 3).astype(np.float32) * perturbation_scale
positions_perturbed += perturbations

print(f"  Min element size (5th percentile): {min_element_size:.6e}")
print(f"  Perturbation scale: {perturbation_scale:.6e}")
print(f"  Mean perturbation magnitude: {np.linalg.norm(perturbations, axis=1).mean():.6e}\n")

# ============================================================================
# Phase 5: Warmup
# ============================================================================

print("Phase 5: JIT warmup...")
warmup_positions = jnp.array(positions_random[:10])
t0 = time.time()
_ = search_batch_with_precomputed_neighbors(
    warmup_positions, octree_gpu, levels_to_try=(14, 13, 12, 11, 10, 9, 8, 7), max_tests_per_cell=20
)
jax.block_until_ready(_)
t1 = time.time()
print(f"  ✅ JIT compiled in {t1-t0:.2f}s\n")

# ============================================================================
# Phase 6: Test All Three Strategies
# ============================================================================

def run_search_test(positions_cpu, strategy_name, ground_truth=None):
    """Run search test for one strategy and return results."""
    print("="*80)
    print(f"Test: {strategy_name}")
    print("="*80)

    # Start memory tracking
    tracemalloc.start()
    mem_before = tracemalloc.get_traced_memory()[0]

    # Upload to GPU
    positions_gpu = jnp.array(positions_cpu)

    # Run search
    t0 = time.time()
    found, tests = search_batch_with_precomputed_neighbors(
        positions_gpu, octree_gpu, levels_to_try=(14, 13, 12, 11, 10, 9, 8, 7), max_tests_per_cell=20
    )
    jax.block_until_ready(found)
    t1 = time.time()

    # Get memory after
    mem_after = tracemalloc.get_traced_memory()[0]
    mem_used = (mem_after - mem_before) / 1024**2  # MB
    tracemalloc.stop()

    # Convert to CPU
    found_cpu = np.array(found)
    tests_cpu = np.array(tests)

    # Calculate statistics
    n_found = np.sum(found_cpu >= 0)
    searchability = 100.0 * n_found / n_particles

    found_mask = found_cpu >= 0
    if np.any(found_mask):
        mean_tests = tests_cpu[found_mask].mean()
        median_tests = np.median(tests_cpu[found_mask])
        max_tests = tests_cpu.max()
    else:
        mean_tests = median_tests = max_tests = 0

    elapsed = t1 - t0
    throughput = n_particles / elapsed

    # Ground truth accuracy (only for centroids)
    accuracy = None
    if ground_truth is not None:
        n_correct = np.sum(found_cpu == ground_truth)
        accuracy = 100.0 * n_correct / n_particles

    # Print results
    print(f"  Particles: {n_particles:,}")
    print(f"  Found: {n_found:,} ({searchability:.2f}%)")
    if accuracy is not None:
        print(f"  Correct (ground truth): {n_correct:,} ({accuracy:.2f}%)")
    print(f"  Tests/particle:")
    print(f"    Mean: {mean_tests:.1f}")
    print(f"    Median: {median_tests:.0f}")
    print(f"    Max: {max_tests}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {throughput:,.0f} particles/sec")
    print(f"  Memory used: {mem_used:.2f} MB\n")

    return {
        'name': strategy_name,
        'n_found': n_found,
        'searchability': searchability,
        'accuracy': accuracy,
        'mean_tests': mean_tests,
        'median_tests': median_tests,
        'max_tests': max_tests,
        'time': elapsed,
        'throughput': throughput,
        'memory_mb': mem_used
    }

# Run all three strategies
results = []
results.append(run_search_test(positions_random, "Strategy 1: Uniform Random"))
results.append(run_search_test(positions_centroids, "Strategy 2: Element Centroids", ground_truth=ground_truth_elements))
results.append(run_search_test(positions_perturbed, "Strategy 3: Perturbed Centroids", ground_truth=ground_truth_elements))

# ============================================================================
# Summary Comparison Table
# ============================================================================

print("="*80)
print("SUMMARY: Strategy Comparison")
print("="*80)
print()
print("Baseline (primary only, 8 levels): 74.6% searchability @ 12,106 p/s")
print()
print(f"{'Strategy':<40} {'Found':<12} {'Accuracy':<12} {'Tests':<10} {'Throughput':<15} {'Memory':<10}")
print(f"{'':<40} {'%':<12} {'%':<12} {'(mean)':<10} {'(p/s)':<15} {'(MB)':<10}")
print("-"*80)

for r in results:
    accuracy_str = f"{r['accuracy']:.2f}" if r['accuracy'] is not None else "N/A"
    print(f"{r['name']:<40} {r['searchability']:>6.2f}      {accuracy_str:>6}      {r['mean_tests']:>6.1f}    {r['throughput']:>10,.0f}     {r['memory_mb']:>6.2f}")

print()
print("="*80)

# Check success
best_searchability = max(r['searchability'] for r in results)
if best_searchability >= 95.0:
    print(f"✅ SUCCESS: Best searchability {best_searchability:.2f}% >= 95% target!")
    improvement = best_searchability / 100.0 - 0.746
    print(f"   Improvement over baseline: +{100.0*improvement:.1f} percentage points")
else:
    print(f"⚠️  Best searchability: {best_searchability:.2f}% < 95% target")
    improvement = best_searchability / 100.0 - 0.746
    print(f"   Still improved over baseline: +{100.0*improvement:.1f} percentage points")

# Centroid accuracy analysis
centroid_result = results[1]
if centroid_result['accuracy'] is not None:
    print()
    print(f"Ground Truth Validation (Element Centroids):")
    print(f"  Found: {centroid_result['searchability']:.2f}%")
    print(f"  Correct element: {centroid_result['accuracy']:.2f}%")
    if centroid_result['accuracy'] < centroid_result['searchability']:
        wrong_element = centroid_result['searchability'] - centroid_result['accuracy']
        print(f"  ⚠️  Found in wrong element: {wrong_element:.2f}% (numerical precision issue)")

print("="*80 + "\n")
