#!/usr/bin/env python3
"""
Test Option A: Multi-Cell Vertex Registration + 2×2×2 Local Search

This tests the multi-cell vertex registration with 2×2×2 local neighborhood search
with multiple particle generation strategies at varying perturbation levels:

Strategy 1: Uniform Random - Random positions in bounding box
Strategy 2: Element Centroids - Ground truth test (we know which element)
Strategy 3: Small Perturbation - 0.1× minimum element size
Strategy 4: Medium Perturbation - 1.0× minimum element size
Strategy 5: Large Perturbation - 2.0× minimum element size
Strategy 6: Very Large Perturbation - 3.0× minimum element size

Expected Results:
- ~95-98% searchability (vs 80.23% without 2×2×2 local search)
- ~146 tests per particle (8 cells × 18.31 elem/cell)
- Comparable throughput to single-cell methods
"""

import time
import numpy as np
from pathlib import Path
import tracemalloc

import jax
import jax.numpy as jnp

from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import extract_octree_cells_vertex_multi
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu
from jaxtrace.gpu.search.mesh_aligned_point_location import search_mesh_aligned_octree_multi_local_batch
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
from jaxtrace.gpu.search.point_in_tet_methods import set_inverse_matrices_gpu
import jaxtrace.config as config

# Initialize config for mesh-aligned octree
config.POINT_IN_TET_METHOD = 'inverse'
config.L2_SEARCH_METHOD = 'mesh_aligned_octree'

print("="*80)
print("Option A: Multi-Cell Vertex Registration + 2×2×2 Local Search")
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

# Precompute inverse matrices for point-in-tet tests (required for config-based dispatcher)
print("Precomputing inverse matrices for point-in-tet tests...")
M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)
M_inv_gpu = jax.device_put(M_inv_array)
p0_gpu = jax.device_put(p0_array)
set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)
print(f"  Point-in-tet method: {config.POINT_IN_TET_METHOD}\n")

# ============================================================================
# Phase 1: Extract Multi-Cell Octree (CPU)
# ============================================================================

print("Phase 1: Extracting multi-cell octree (CPU)...")
t0 = time.time()
octree_cells = extract_octree_cells_vertex_multi(node_positions, connectivity, verbose=False)
t1 = time.time()
print(f"  ✅ Multi-cell octree extracted")
print(f"    Cells: {octree_cells.n_cells:,}")
print(f"    Elements/cell (avg): {octree_cells.elements_per_cell_mean:.1f}")
print(f"    Cells/element (avg): {octree_cells.cells_per_element_mean:.1f}")
print(f"    Time: {t1-t0:.1f}s\n")

# ============================================================================
# Phase 2: Upload to GPU
# ============================================================================

print("Phase 2: Uploading to GPU...")
t0 = time.time()
octree_gpu = upload_mesh_aligned_octree_to_gpu(
    node_positions, connectivity, octree_cells, verbose=False
)
t1 = time.time()
print(f"  ✅ Uploaded in {t1-t0:.2f}s\n")

# ============================================================================
# Phase 3: Calculate Element Size Distribution
# ============================================================================

print("Phase 3: Calculating element size distribution...")
n_elements = connectivity.shape[0]
element_sizes = np.zeros(n_elements, dtype=np.float32)

# Sample elements for speed
n_sample = min(100000, n_elements)
sample_indices = np.random.choice(n_elements, n_sample, replace=False)

for elem_idx in sample_indices:
    elem_nodes = connectivity[elem_idx]
    elem_positions = node_positions[elem_nodes]
    # Compute characteristic size (min edge length)
    edges = []
    for i in range(4):
        for j in range(i+1, 4):
            edge_len = np.linalg.norm(elem_positions[i] - elem_positions[j])
            edges.append(edge_len)
    element_sizes[elem_idx] = min(edges)

valid_sizes = element_sizes[element_sizes > 0]
min_element_size = np.percentile(valid_sizes, 5)  # 5th percentile
mean_element_size = np.mean(valid_sizes)

print(f"  Min element size (5th percentile): {min_element_size:.6e}")
print(f"  Mean element size: {mean_element_size:.6e}\n")

# ============================================================================
# Phase 4: Generate Test Positions (6 Strategies with Varying Perturbation)
# ============================================================================

n_particles = 100000
np.random.seed(42)

bbox_min = np.array(octree_gpu.bbox_min)
bbox_max = np.array(octree_gpu.bbox_max)

print(f"Generating {n_particles:,} test positions with 6 strategies...")
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
selected_elements = np.random.choice(n_elements, n_particles, replace=True)
positions_centroids = np.zeros((n_particles, 3), dtype=np.float32)
for i, elem_idx in enumerate(selected_elements):
    elem_nodes = connectivity[elem_idx]
    elem_positions = node_positions[elem_nodes]
    positions_centroids[i] = elem_positions.mean(axis=0)
ground_truth_elements = selected_elements.copy()
print(f"  Generated {n_particles:,} element centroids")
print(f"  Ground truth element IDs available for accuracy validation\n")

# Strategy 3-6: Perturbed Centroids with Increasing Perturbation
perturbation_configs = [
    (0.1, "Small Perturbation (0.1× min element)"),
    (1.0, "Medium Perturbation (1.0× min element)"),
    (2.0, "Large Perturbation (2.0× min element)"),
    (3.0, "Very Large Perturbation (3.0× min element)")
]

perturbed_positions = {}
for scale_factor, desc in perturbation_configs:
    print(f"Strategy {3 + len(perturbed_positions)}: {desc}")
    perturbation_scale = min_element_size * scale_factor

    positions = positions_centroids.copy()
    perturbations = np.random.randn(n_particles, 3).astype(np.float32) * perturbation_scale
    positions += perturbations

    perturbed_positions[scale_factor] = positions

    print(f"  Perturbation scale: {perturbation_scale:.6e} ({scale_factor:.1f}× min element)")
    print(f"  Mean perturbation magnitude: {np.linalg.norm(perturbations, axis=1).mean():.6e}\n")

# ============================================================================
# Phase 5: Warmup
# ============================================================================

print("Phase 5: JIT warmup...")
warmup_positions = jnp.array(positions_random[:10])
t0 = time.time()
_ = search_mesh_aligned_octree_multi_local_batch(
    warmup_positions, octree_gpu, max_tests=200
)
jax.block_until_ready(_)
t1 = time.time()
print(f"  ✅ JIT compiled in {t1-t0:.2f}s\n")

# ============================================================================
# Phase 6: Test All Strategies
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
    found, tests = search_mesh_aligned_octree_multi_local_batch(
        positions_gpu, octree_gpu, max_tests=200
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

    # Print results
    print(f"\n  Results:")
    print(f"    Found: {n_found:,} / {n_particles:,} ({searchability:.2f}%)")
    print(f"    Not found: {n_particles - n_found:,}")
    print(f"\n  Tests per particle (found only):")
    print(f"    Mean: {mean_tests:.1f}")
    print(f"    Median: {median_tests:.0f}")
    print(f"    Max: {max_tests}")
    print(f"\n  Performance:")
    print(f"    Time: {elapsed:.3f}s")
    print(f"    Throughput: {throughput:,.0f} particles/sec")
    print(f"    Memory used: {mem_used:.2f} MB")

    # Ground truth validation
    if ground_truth is not None:
        correct_mask = found_cpu == ground_truth
        n_correct = np.sum(correct_mask)
        accuracy = 100.0 * n_correct / n_found if n_found > 0 else 0
        print(f"\n  Ground Truth Validation:")
        print(f"    Correct elements: {n_correct:,} / {n_found:,} ({accuracy:.2f}%)")

        # Sample mismatches
        if n_found > n_correct:
            mismatch_indices = np.where(~correct_mask & (found_cpu >= 0))[0][:5]
            if len(mismatch_indices) > 0:
                print(f"    Sample mismatches (first 5):")
                for idx in mismatch_indices:
                    print(f"      Particle {idx}: expected {ground_truth[idx]}, got {found_cpu[idx]}")

    print("="*80 + "\n")

    return {
        'strategy': strategy_name,
        'searchability': searchability,
        'n_found': n_found,
        'mean_tests': mean_tests,
        'median_tests': median_tests,
        'max_tests': max_tests,
        'throughput': throughput,
        'time': elapsed,
        'memory_mb': mem_used
    }

# Run all tests
results = []

results.append(run_search_test(positions_random, "Strategy 1: Uniform Random"))
results.append(run_search_test(positions_centroids, "Strategy 2: Element Centroids", ground_truth_elements))

for i, (scale_factor, desc) in enumerate(perturbation_configs):
    strategy_name = f"Strategy {3+i}: {desc}"
    results.append(run_search_test(perturbed_positions[scale_factor], strategy_name, ground_truth_elements))

# ============================================================================
# Summary Table
# ============================================================================

print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print()
print(f"{'Strategy':<50} {'Searchability':<15} {'Tests/Particle':<18} {'Throughput':<18}")
print("-"*80)

for r in results:
    print(f"{r['strategy']:<50} {r['searchability']:>6.2f}%         "
          f"{r['mean_tests']:>6.1f} (median {r['median_tests']:.0f})    "
          f"{r['throughput']:>10,.0f} p/s")

print("="*80)
print()

# Key findings
best_searchability = max(results, key=lambda x: x['searchability'])
print(f"✅ Best searchability: {best_searchability['strategy']} - {best_searchability['searchability']:.2f}%")

# Compare to baseline (expected 80.23% without 2×2×2 local search)
baseline_searchability = 80.23
improvement = best_searchability['searchability'] - baseline_searchability
print(f"✅ Improvement over baseline (single-cell): +{improvement:.2f}% (from {baseline_searchability:.2f}% → {best_searchability['searchability']:.2f}%)")

print()
print("="*80)
print("TEST COMPLETE")
print("="*80)
