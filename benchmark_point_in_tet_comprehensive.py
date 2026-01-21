#!/usr/bin/env python3
"""
Comprehensive Point-in-Tetrahedron Benchmark

Tests all point-in-tet methods with REALISTIC particle distributions:
1. Random uniform distribution over domain
2. Element centroids + small perturbations (near-element tests)

Includes INVERSE method (precomputed inverse matrices).

Methods tested:
- current: Original barycentric method
- skala: Skála's optimized Cramer's rule
- axis_aligned: OLD AA detection (may be broken)
- pure_aa: NEW AA-only method (corrected)
- skala_memory_opt: NEW Skála with precomputed vertices (corrected)
- branchless_hybrid: NEW hybrid AA+Skála (corrected)
- inverse: NEW precomputed inverse matrix method (4.36× expected)

Metrics:
- Assignment success rate (% particles found)
- Throughput (particles/second)
- Time per million point-in-tet calls
- FLOPs estimate
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu
from jaxtrace.gpu.tracking.initial_assignment_cascading import initial_assignment_cascading_fallback
from jaxtrace.gpu.search.aa_detection import (
    precompute_aa_metadata,
    precompute_element_vertices,
)
from jaxtrace.gpu.search.point_in_tet_methods import set_corrected_metadata
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
import jaxtrace.config as config


# ============================================================================
# Configuration
# ============================================================================

MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 121)  # Single timestep
VELOCITY_FIELD_NAME = 'Displacement'

# Search radii for initial assignment
# Using production values for realistic comparison
INITIAL_SEARCH_RADIUS = 500
INITIAL_SEARCH_FALLBACK_RADII = [1000, 2000, 5000, 10000, 100000]

SEED = 42


def generate_random_particles(node_positions, n_particles, seed=42):
    """Generate random particles uniformly distributed over domain."""
    np.random.seed(seed)

    domain_min = node_positions.min(axis=0)
    domain_max = node_positions.max(axis=0)

    # Add small margin to stay inside mesh
    margin = 0.01
    domain_size = domain_max - domain_min
    domain_min_safe = domain_min + margin * domain_size
    domain_max_safe = domain_max - margin * domain_size

    # Generate random positions
    positions = np.random.uniform(
        low=domain_min_safe,
        high=domain_max_safe,
        size=(n_particles, 3)
    ).astype(np.float32)

    return positions


def generate_perturbed_centroids(connectivity, node_positions, perturbation_scale=0.1, seed=42):
    """Generate particles at element centroids + small random perturbations.

    Args:
        connectivity: (n_elements, 4) element connectivity
        node_positions: (n_nodes, 3) node coordinates
        perturbation_scale: Scale of perturbation relative to smallest element size
        seed: Random seed

    Returns:
        positions: (n_elements, 3) perturbed centroid positions
    """
    np.random.seed(seed)

    n_elements = connectivity.shape[0]

    # Compute centroids
    centroids = np.zeros((n_elements, 3), dtype=np.float32)
    for i in range(n_elements):
        elem_nodes = connectivity[i]
        elem_vertices = node_positions[elem_nodes]
        centroids[i] = elem_vertices.mean(axis=0)

    # Compute smallest element size for perturbation scale
    min_edge_length = np.inf
    for i in range(min(1000, n_elements)):  # Sample first 1000 elements
        elem_nodes = connectivity[i]
        elem_vertices = node_positions[elem_nodes]
        # Compute edge lengths
        for j in range(4):
            for k in range(j+1, 4):
                edge_length = np.linalg.norm(elem_vertices[j] - elem_vertices[k])
                min_edge_length = min(min_edge_length, edge_length)

    perturbation_magnitude = perturbation_scale * min_edge_length

    # Add random perturbations
    perturbations = np.random.uniform(
        low=-perturbation_magnitude,
        high=perturbation_magnitude,
        size=(n_elements, 3)
    ).astype(np.float32)

    positions = centroids + perturbations

    print(f"  Perturbation magnitude: {perturbation_magnitude:.6f}")
    print(f"  Min element size (sampled): {min_edge_length:.6f}")

    return positions


def main():
    print("=" * 80)
    print("Comprehensive Point-in-Tetrahedron Benchmark")
    print("Testing all methods with realistic particle distributions")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print("=" * 80)

    # ========================================================================
    # 1. Load Mesh
    # ========================================================================

    print("\n[1/9] Loading mesh from PVTU...")
    t_load = time.time()
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=False
    )
    t_load = time.time() - t_load

    n_nodes_orig = node_positions.shape[0]
    n_elements = connectivity.shape[0]

    print(f"  Mesh loaded in {t_load:.2f}s")
    print(f"  Elements: {n_elements:,}, Nodes: {n_nodes_orig:,}")

    # ========================================================================
    # 2. Deduplicate Nodes
    # ========================================================================

    print(f"\n[2/9] Deduplicating nodes...")
    t_dedup = time.time()
    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    t_dedup = time.time() - t_dedup

    n_nodes = node_positions.shape[0]
    print(f"  Removed {n_duplicates_removed:,} duplicates in {t_dedup:.2f}s")
    print(f"  Nodes: {n_nodes:,}")

    # ========================================================================
    # 3. Precompute AA Metadata
    # ========================================================================

    print(f"\n[3/9] Precomputing AA metadata...")
    t_aa_start = time.time()
    aa_metadata = precompute_aa_metadata(connectivity, node_positions, verbose=True)
    t_aa_elapsed = time.time() - t_aa_start
    print(f"  AA metadata precomputed in {t_aa_elapsed:.2f}s")

    n_aa_elements = int(np.sum(aa_metadata.is_axis_aligned))
    aa_percentage = (n_aa_elements / n_elements) * 100
    print(f"  Axis-aligned elements: {n_aa_elements:,}/{n_elements:,} ({aa_percentage:.2f}%)")

    # ========================================================================
    # 4. Precompute Element Vertices
    # ========================================================================

    print(f"\n[4/9] Precomputing element vertices...")
    t_elem_start = time.time()
    element_vertices = precompute_element_vertices(connectivity, node_positions, verbose=True)
    t_elem_elapsed = time.time() - t_elem_start
    print(f"  Element vertices precomputed in {t_elem_elapsed:.2f}s")

    # ========================================================================
    # 5. Precompute Inverse Matrices (NEW)
    # ========================================================================

    print(f"\n[5/9] Precomputing inverse matrices (for 'inverse' method)...")
    t_inv_start = time.time()
    M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)
    t_inv_elapsed = time.time() - t_inv_start
    print(f"  Inverse matrices precomputed in {t_inv_elapsed:.2f}s")

    # Estimate memory overhead
    memory_mb = (M_inv_array.nbytes + p0_array.nbytes) / (1024**2)
    print(f"  Memory: {memory_mb:.1f} MB ({n_elements:,} elements × 3×3 + p0)")

    # ========================================================================
    # 6. Build Morton Octree
    # ========================================================================

    print(f"\n[6/9] Building Morton octree...")
    t_octree = time.time()
    octree_struct = build_global_morton_octree(
        node_positions=node_positions,
        connectivity=connectivity,
        leaf_capacity=256,
        max_depth=21,
        verbose=False
    )
    t_octree = time.time() - t_octree
    print(f"  Built {octree_struct.n_leaves:,} leaves in {t_octree:.2f}s")

    # ========================================================================
    # 7. Upload to GPU
    # ========================================================================

    print("\n[7/9] Uploading to GPU...")

    # Build element neighbors
    element_neighbors = build_element_neighbors_array(connectivity, method='face', verbose=False)

    # Upload mesh
    mesh_gpu = upload_mesh_to_gpu(
        connectivity=connectivity,
        node_positions=node_positions,
        element_neighbors=element_neighbors,
        verbose=False
    )

    # Upload Morton structure
    mesh_gpu_octree = upload_global_morton_to_gpu(
        octree_struct,
        connectivity,
        node_positions
    )

    # Upload AA metadata
    from jaxtrace.gpu.search.aa_detection import AxisAlignedMetadata
    aa_metadata_gpu = AxisAlignedMetadata(
        base_vertex_indices=jax.device_put(aa_metadata.base_vertex_indices),
        base_vertices=jax.device_put(aa_metadata.base_vertices),
        inv_edge_lengths=jax.device_put(aa_metadata.inv_edge_lengths),
        axis_indices=jax.device_put(aa_metadata.axis_indices),
        is_axis_aligned=jax.device_put(aa_metadata.is_axis_aligned)
    )
    element_vertices_gpu = jax.device_put(element_vertices)
    M_inv_gpu = jax.device_put(M_inv_array)
    p0_gpu = jax.device_put(p0_array)

    # Set corrected metadata
    set_corrected_metadata(aa_metadata_gpu, element_vertices_gpu)

    # Set inverse matrices globally (use correct function name)
    from jaxtrace.gpu.search.point_in_tet_methods import set_inverse_matrices_gpu
    set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)

    print(f"  Uploaded mesh, Morton octree, and all metadata to GPU")

    # ========================================================================
    # 8. Generate Particle Distributions
    # ========================================================================

    print(f"\n[8/9] Generating particle distributions...")

    # Distribution 1: Random uniform (225,000 particles - same as production)
    n_random = 225_000
    print(f"\n  Distribution 1: {n_random:,} random uniform particles")
    particles_random = generate_random_particles(node_positions, n_random, seed=SEED)
    particles_random_gpu = jax.device_put(particles_random)

    # Distribution 2: Perturbed centroids (all elements, ~3.3M particles)
    print(f"\n  Distribution 2: Element centroids + perturbations")
    particles_centroids = generate_perturbed_centroids(
        connectivity, node_positions, perturbation_scale=0.1, seed=SEED
    )
    n_centroids = particles_centroids.shape[0]
    print(f"  Generated {n_centroids:,} perturbed centroid particles")
    particles_centroids_gpu = jax.device_put(particles_centroids)

    # ========================================================================
    # 9. Benchmark: Test All Methods with Both Distributions
    # ========================================================================

    print("\n[9/9] Benchmarking all methods...")
    print("=" * 80)

    methods = [
        'current',
        'skala',
        'axis_aligned',
        'pure_aa',
        'skala_memory_opt',
        'branchless_hybrid',
        'inverse'  # NEW
    ]

    distributions = [
        ('Random Uniform', particles_random_gpu, n_random),
        ('Perturbed Centroids', particles_centroids_gpu, n_centroids)
    ]

    all_results = {}

    for dist_name, positions_gpu, n_particles in distributions:
        print(f"\n{'='*80}")
        print(f"Testing Distribution: {dist_name} ({n_particles:,} particles)")
        print(f"{'='*80}")

        results = {}

        for method_name in methods:
            print(f"\n  Method: {method_name}")

            # Set config
            config.POINT_IN_TET_METHOD = method_name

            # Warmup (compile)
            print(f"    Compiling...")
            _ = initial_assignment_cascading_fallback(
                positions_gpu[:100],
                mesh_gpu_octree,
                initial_radius=INITIAL_SEARCH_RADIUS,
                fallback_radii=[INITIAL_SEARCH_FALLBACK_RADII[0]],
                verbose=False
            )

            # Benchmark
            print(f"    Running initial assignment...")
            t_start = time.time()
            element_ids_gpu = initial_assignment_cascading_fallback(
                positions_gpu,
                mesh_gpu_octree,
                initial_radius=INITIAL_SEARCH_RADIUS,
                fallback_radii=INITIAL_SEARCH_FALLBACK_RADII,
                verbose=True
            )
            element_ids_gpu = jax.block_until_ready(element_ids_gpu)
            t_elapsed = time.time() - t_start

            # Metrics
            n_assigned = int(jnp.sum(element_ids_gpu >= 0))
            success_rate = (n_assigned / n_particles) * 100
            throughput = n_particles / t_elapsed

            results[method_name] = {
                'time': t_elapsed,
                'n_assigned': n_assigned,
                'success_rate': success_rate,
                'throughput': throughput,
                'element_ids': element_ids_gpu
            }

            print(f"    Time: {t_elapsed:.3f}s")
            print(f"    Assigned: {n_assigned:,}/{n_particles:,} ({success_rate:.2f}%)")
            print(f"    Throughput: {throughput:,.0f} particles/s")

        all_results[dist_name] = results

    # ========================================================================
    # 10. Results Analysis
    # ========================================================================

    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 80)

    for dist_name in ['Random Uniform', 'Perturbed Centroids']:
        results = all_results[dist_name]

        print(f"\n{dist_name} Distribution:")
        print("=" * 80)

        # Find baseline and best
        baseline_method = 'current'
        baseline_time = results[baseline_method]['time']
        baseline_throughput = results[baseline_method]['throughput']

        best_method = max(methods, key=lambda m: results[m]['throughput'])
        best_time = results[best_method]['time']
        best_throughput = results[best_method]['throughput']
        best_speedup = baseline_time / best_time

        print(f"\nMethod                  Time (s)    Throughput (p/s)  Speedup  Success Rate")
        print("-" * 80)

        for method_name in methods:
            r = results[method_name]
            speedup = baseline_time / r['time']
            marker = " ★" if method_name == best_method else "  "

            print(f"{method_name:20s}  {r['time']:8.3f}  {r['throughput']:14,.0f}  "
                  f"{speedup:6.2f}×  {r['success_rate']:6.2f}%{marker}")

        print(f"\nBest Method: {best_method}")
        print(f"  Speedup: {best_speedup:.2f}×")
        print(f"  Throughput: {best_throughput:,.0f} p/s")

        # Check assignment agreement
        print(f"\nAssignment Agreement (vs {baseline_method}):")
        print("-" * 80)

        baseline_ids = results[baseline_method]['element_ids']
        for method_name in methods:
            if method_name == baseline_method:
                continue

            test_ids = results[method_name]['element_ids']

            # Check agreement
            agree_assignment = jnp.all((baseline_ids >= 0) == (test_ids >= 0))

            assigned_both = (baseline_ids >= 0) & (test_ids >= 0)
            if jnp.sum(assigned_both) > 0:
                same_element = jnp.all(baseline_ids[assigned_both] == test_ids[assigned_both])
            else:
                same_element = True

            passed = agree_assignment and same_element

            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {method_name:20s}  {status}")

    # ========================================================================
    # 11. FLOPs Analysis
    # ========================================================================

    print("\n" + "=" * 80)
    print("ESTIMATED FLOPs ANALYSIS")
    print("=" * 80)

    # FLOPs estimates per point-in-tet call (from literature and analysis)
    flops_estimates = {
        'current': 145,  # Original barycentric (lots of redundant computation)
        'skala': 87,  # Skála's optimized Cramer's rule
        'axis_aligned': 50,  # AA fast path (when applicable)
        'pure_aa': 25,  # Pure AA (corrected, optimized)
        'skala_memory_opt': 87,  # Same as Skála (memory vs computation trade-off)
        'branchless_hybrid': 60,  # Hybrid (weighted average)
        'inverse': 22  # Precomputed inverse (minimal FLOPs)
    }

    print("\nMethod                  Est. FLOPs/call  Theoretical Speedup")
    print("-" * 80)

    baseline_flops = flops_estimates['current']
    for method_name in methods:
        flops = flops_estimates.get(method_name, 0)
        theoretical_speedup = baseline_flops / flops if flops > 0 else 0

        print(f"{method_name:20s}  {flops:15d}  {theoretical_speedup:18.2f}×")

    print("\nNote: Actual speedup depends on memory bandwidth, GPU utilization, and control flow")

    # ========================================================================
    # 12. Recommendations
    # ========================================================================

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR PRODUCTION")
    print("=" * 80)

    # Find best method across both distributions
    avg_speedups = {}
    for method_name in methods:
        speedups = []
        for dist_name in ['Random Uniform', 'Perturbed Centroids']:
            baseline_time = all_results[dist_name]['current']['time']
            method_time = all_results[dist_name][method_name]['time']
            speedups.append(baseline_time / method_time)
        avg_speedups[method_name] = np.mean(speedups)

    best_overall = max(avg_speedups.keys(), key=lambda m: avg_speedups[m])
    best_avg_speedup = avg_speedups[best_overall]

    print(f"\nBest Overall Method: {best_overall}")
    print(f"  Average Speedup: {best_avg_speedup:.2f}×")
    print(f"  Recommendation: Use POINT_IN_TET_METHOD='{best_overall}' in production")

    if best_avg_speedup >= 4.0:
        print(f"\n✅ EXCELLENT: Achieves {best_avg_speedup:.2f}× speedup (target: 3-4×)")
    elif best_avg_speedup >= 3.0:
        print(f"\n✅ GOOD: Achieves {best_avg_speedup:.2f}× speedup")
    elif best_avg_speedup >= 2.0:
        print(f"\n⚠️  MODEST: Achieves {best_avg_speedup:.2f}× speedup")
    else:
        print(f"\n❌ POOR: Only achieves {best_avg_speedup:.2f}× speedup")

    # Check if inverse is best
    if best_overall == 'inverse':
        inverse_speedup = avg_speedups['inverse']
        print(f"\n★ INVERSE METHOD IS BEST:")
        print(f"  - Speedup: {inverse_speedup:.2f}×")
        print(f"  - Memory overhead: {memory_mb:.1f} MB (acceptable for modern GPUs)")
        print(f"  - FLOPs: 22 (vs 145 baseline) - 6.6× theoretical reduction")
        print(f"  - Recommendation: ✅ USE IN PRODUCTION")

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
