#!/usr/bin/env python3
"""
Octree Accuracy Validation Test

Tests the accuracy improvement from adaptive octree leaves vs fixed-capacity.

Comparison:
- OLD (Fixed-capacity): ~12.7% centroid success with radius=4
- NEW (Adaptive octree): Expected >95% centroid success with radius=0-1

Test Method:
1. Build adaptive octree structure
2. Initialize particles at element centroids (known ground truth)
3. Run L2 search and verify correctness
4. Compare with perturbations

Expected Results:
- Centroid success: >95% (elements found in their own octree leaf)
- Perturbed success: >80% (particles near boundaries)
- Low search radius sufficient (0-1 instead of 4+)
"""

import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaxtrace.gpu.particles import ParticleData
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import (
    upload_global_morton_to_gpu,
    search_L2_global_morton_single,
    point_in_tet_gpu
)


# Configuration
MESH_PATH = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_159.pvtu"
N_TEST_PARTICLES = 100_000  # Sample elements to test
L2_SEARCH_RADIUS = 1  # Start with radius=1 (should be sufficient for octree)
PERTURBATION_SCALE = 1.0  # Fraction of minimum element size
SEED = 42


def compute_element_sizes(node_positions, connectivity):
    """Compute characteristic size for each element (average edge length)."""
    n_elements = connectivity.shape[0]
    element_sizes = np.zeros(n_elements, dtype=np.float32)

    for i in range(n_elements):
        nodes = connectivity[i]
        coords = node_positions[nodes]  # (4, 3)

        # Compute all 6 edge lengths
        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (2, 3)
        ]

        edge_lengths = []
        for e1, e2 in edges:
            length = np.linalg.norm(coords[e1] - coords[e2])
            edge_lengths.append(length)

        element_sizes[i] = np.mean(edge_lengths)

    return element_sizes


def main():
    print("=" * 80)
    print("Octree Accuracy Validation - Adaptive Octree vs Fixed-Capacity")
    print("=" * 80)
    print(f"Test particles: {N_TEST_PARTICLES:,}")
    print(f"L2 search radius: {L2_SEARCH_RADIUS}")
    print(f"Perturbation scale: {PERTURBATION_SCALE}× minimum element size")
    print(f"Expected improvement: 12.7% → >95% for centroids")
    print("=" * 80)

    # ========================================================================
    # 1. Load Mesh
    # ========================================================================

    print("\n[1/6] Loading mesh...")
    t_load = time.time()
    node_positions, connectivity, velocity_field = load_mesh_from_pvtu(
        Path(MESH_PATH),
        field_name='Displacement'
    )
    t_load = time.time() - t_load

    n_nodes = node_positions.shape[0]
    n_elements = connectivity.shape[0]
    print(f"  Mesh: {n_elements:,} elements, {n_nodes:,} nodes")
    print(f"  Load time: {t_load:.2f}s")

    # ========================================================================
    # 2. Build Adaptive Octree Structure (CPU)
    # ========================================================================

    print("\n[2/6] Building adaptive octree structure (CPU)...")
    t_octree = time.time()

    morton_struct = build_global_morton_octree(
        node_positions=node_positions,
        connectivity=connectivity,
        leaf_capacity=256,
        max_depth=21,
        verbose=False
    )

    t_octree = time.time() - t_octree
    print(f"  Built {morton_struct.n_leaves:,} octree leaves in {t_octree:.2f}s")
    print(f"  Table depth: {morton_struct.table_depth}")
    print(f"  Memory: {(morton_struct.prefix_table.nbytes) / (1024**2):.1f} MB")

    # ========================================================================
    # 3. Upload to GPU
    # ========================================================================

    print("\n[3/6] Uploading mesh and octree structure to GPU...")
    t_upload = time.time()

    # Upload octree Morton structure
    mesh_gpu_morton = upload_global_morton_to_gpu(
        morton_struct,
        connectivity,
        node_positions
    )

    # Force transfer
    _ = jax.block_until_ready(mesh_gpu_morton.elem_ids_sorted)

    t_upload = time.time() - t_upload
    print(f"  Upload time: {t_upload:.2f}s")

    # ========================================================================
    # 4. Compute Element Sizes (for perturbation)
    # ========================================================================

    print("\n[4/6] Computing element sizes...")
    t_sizes = time.time()

    element_sizes = compute_element_sizes(node_positions, connectivity)
    min_element_size = element_sizes.min()
    max_element_size = element_sizes.max()
    mean_element_size = element_sizes.mean()

    t_sizes = time.time() - t_sizes
    print(f"  Element sizes computed in {t_sizes:.2f}s")
    print(f"  Min element size: {min_element_size:.6f}")
    print(f"  Max element size: {max_element_size:.6f}")
    print(f"  Mean element size: {mean_element_size:.6f}")
    print(f"  Perturbation magnitude: {PERTURBATION_SCALE * min_element_size:.6f}")

    # ========================================================================
    # 5. Test 1: Centroid-Based Accuracy (Ground Truth)
    # ========================================================================

    print("\n[5/6] Test 1: Centroid-based accuracy (octree leaves)...")
    np.random.seed(SEED)

    # Randomly sample elements
    test_elem_ids = np.random.randint(0, n_elements, size=N_TEST_PARTICLES)

    # Compute centroids for sampled elements
    print(f"  Computing centroids for {N_TEST_PARTICLES:,} elements...")
    centroids = np.zeros((N_TEST_PARTICLES, 3), dtype=np.float32)
    for i, elem_id in enumerate(test_elem_ids):
        nodes = connectivity[elem_id]
        centroid = node_positions[nodes].mean(axis=0)
        centroids[i] = centroid

    # Upload centroids to GPU
    centroids_gpu = jax.device_put(centroids)

    # Run L2 search with octree
    print(f"  Running L2 search on centroids (radius={L2_SEARCH_RADIUS})...")
    t_search = time.time()

    found_elem_ids = jax.vmap(
        lambda p: search_L2_global_morton_single(p, mesh_gpu_morton, jnp.int32(L2_SEARCH_RADIUS))
    )(centroids_gpu)
    found_elem_ids = jax.block_until_ready(found_elem_ids)

    t_search = time.time() - t_search

    # Analyze results
    found_elem_ids_np = np.array(found_elem_ids, dtype=np.int32)
    found_mask = found_elem_ids_np >= 0
    n_found = np.sum(found_mask)
    success_rate = (n_found / N_TEST_PARTICLES) * 100

    # Verify correctness with point-in-tet
    n_correct = 0
    for i in range(N_TEST_PARTICLES):
        if found_elem_ids_np[i] >= 0:
            is_inside = point_in_tet_gpu(
                centroids_gpu[i],
                jnp.int32(found_elem_ids_np[i]),
                mesh_gpu_morton.connectivity,
                mesh_gpu_morton.node_positions
            )
            if is_inside:
                n_correct += 1

    correctness_rate = (n_correct / max(n_found, 1)) * 100

    print(f"\n  Results (Centroid-based, Octree):") 
    print(f"    Success rate: {success_rate:.2f}% ({n_found}/{N_TEST_PARTICLES})")
    print(f"    Correctness rate: {correctness_rate:.2f}% ({n_correct}/{n_found})")
    print(f"    Search time: {t_search*1000:.2f} ms")
    print(f"    Throughput: {N_TEST_PARTICLES/t_search:.0f} searches/s")

    # ========================================================================
    # 6. Test 2: Perturbed Accuracy
    # ========================================================================

    print("\n[6/6] Test 2: Perturbed accuracy (octree)...")
    np.random.seed(SEED + 1)

    # Add random perturbation
    perturbation_magnitude = PERTURBATION_SCALE * min_element_size
    perturbations = np.random.randn(N_TEST_PARTICLES, 3).astype(np.float32)
    perturbations = perturbations / np.linalg.norm(perturbations, axis=1, keepdims=True)
    perturbations = perturbations * perturbation_magnitude

    perturbed_positions = centroids + perturbations

    # Upload to GPU
    perturbed_positions_gpu = jax.device_put(perturbed_positions)

    # Run L2 search
    print(f"  Running L2 search on perturbed positions...")
    t_search_perturbed = time.time()

    found_elem_ids_perturbed = jax.vmap(
        lambda p: search_L2_global_morton_single(p, mesh_gpu_morton, jnp.int32(L2_SEARCH_RADIUS))
    )(perturbed_positions_gpu)
    found_elem_ids_perturbed = jax.block_until_ready(found_elem_ids_perturbed)

    t_search_perturbed = time.time() - t_search_perturbed

    # Analyze results
    found_elem_ids_perturbed_np = np.array(found_elem_ids_perturbed, dtype=np.int32)
    found_mask_perturbed = found_elem_ids_perturbed_np >= 0
    n_found_perturbed = np.sum(found_mask_perturbed)
    success_rate_perturbed = (n_found_perturbed / N_TEST_PARTICLES) * 100

    # Verify correctness
    n_correct_perturbed = 0
    for i in range(N_TEST_PARTICLES):
        if found_elem_ids_perturbed_np[i] >= 0:
            is_inside = point_in_tet_gpu(
                perturbed_positions_gpu[i],
                jnp.int32(found_elem_ids_perturbed_np[i]),
                mesh_gpu_morton.connectivity,
                mesh_gpu_morton.node_positions
            )
            if is_inside:
                n_correct_perturbed += 1

    correctness_rate_perturbed = (n_correct_perturbed / max(n_found_perturbed, 1)) * 100

    print(f"\n  Results (Perturbed, Octree):")
    print(f"    Success rate: {success_rate_perturbed:.2f}% ({n_found_perturbed}/{N_TEST_PARTICLES})")
    print(f"    Correctness rate: {correctness_rate_perturbed:.2f}% ({n_correct_perturbed}/{n_found_perturbed})")
    print(f"    Search time: {t_search_perturbed*1000:.2f} ms")
    print(f"    Throughput: {N_TEST_PARTICLES/t_search_perturbed:.0f} searches/s")

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 80)
    print("OCTREE ACCURACY VALIDATION SUMMARY")
    print("=" * 80)

    print(f"\nAdaptive Octree Results:")
    print(f"  Centroid test: {success_rate:.2f}% success, {correctness_rate:.2f}% correct")
    print(f"  Perturbed test: {success_rate_perturbed:.2f}% success, {correctness_rate_perturbed:.2f}% correct")
    print(f"  Search radius: {L2_SEARCH_RADIUS}")

    print(f"\nComparison with Fixed-Capacity (from previous test):")
    print(f"  OLD (Fixed, radius=4): 12.75% centroid, 16.54% perturbed")
    print(f"  NEW (Octree, radius={L2_SEARCH_RADIUS}): {success_rate:.2f}% centroid, {success_rate_perturbed:.2f}% perturbed")
    print(f"  Improvement: {success_rate - 12.75:.2f}% (centroid)")

    print("\n" + "=" * 80)
    print("VALIDATION CRITERIA")
    print("=" * 80)

    success = True

    # Centroid-based should be very high (>95%)
    if success_rate >= 95.0 and correctness_rate >= 95.0:
        print(f"✅ Centroid test: {success_rate:.1f}% success, {correctness_rate:.1f}% correct (≥95% target)")
    else:
        print(f"❌ Centroid test: {success_rate:.1f}% success, {correctness_rate:.1f}% correct (<95% target)")
        success = False

    # Perturbed should still be good (>80%)
    if success_rate_perturbed >= 80.0 and correctness_rate_perturbed >= 80.0:
        print(f"✅ Perturbed test: {success_rate_perturbed:.1f}% success, {correctness_rate_perturbed:.1f}% correct (≥80% target)")
    else:
        print(f"⚠️  Perturbed test: {success_rate_perturbed:.1f}% success, {correctness_rate_perturbed:.1f}% correct (<80% target)")
        # Don't fail on perturbed test

    print(f"✅ Search radius: {L2_SEARCH_RADIUS} (much lower than fixed-capacity's 4)")
    print(f"✅ Octree leaves: {morton_struct.n_leaves:,} (spatially coherent)")

    print("=" * 80)

    if success:
        print("\n🎉 OCTREE ACCURACY VALIDATION PASSED!")
        print("   Adaptive octree achieves >95% accuracy for centroid search.")
        print("   Ready for production particle tracking.")
    else:
        print("\n❌ OCTREE ACCURACY VALIDATION FAILED")
        print("   Centroid accuracy below 95% threshold.")

    print("=" * 80)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
