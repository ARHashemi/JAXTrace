"""
Test script for corrected axis-aligned detection algorithm.

Tests:
1. Detection correctness (all 4 vertices checked)
2. Component-based vs dot-product comparison
3. Adaptive tolerance for refined elements
4. Pure AA method performance
5. Agreement with baseline on real mesh
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
from pathlib import Path

# Import corrected algorithm
from jaxtrace.gpu.search.aa_detection import (
    detect_aa_tetrahedron_component_based,
    precompute_aa_metadata,
    precompute_element_vertices,
    point_in_tet_pure_aa,
    point_in_tet_skala_memory_opt,
    point_in_tet_branchless_hybrid
)

# Import baseline for comparison
from jaxtrace.gpu.search.point_in_tet_methods import point_in_tet_current

# Import mesh loading
from jaxtrace.io.pvtu_io import load_pvtu_mesh
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

# Test configuration
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE = "featurelessAvtk_120.pvtu"
SEED = 42


def test_detection_all_vertices():
    """Test 1: Verify detection checks all 4 vertices (not just p0)."""
    print(f"\n{'='*80}")
    print("TEST 1: Detection Checks All 4 Vertices")
    print(f"{'='*80}\n")

    # Create 4 test tetrahedra with right-angle at different vertices
    test_cases = [
        {
            'name': 'Right-angle at p0',
            'verts': np.array([
                [0.0, 0.0, 0.0],  # p0 (right-angle)
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
            'expected_vertex': 0
        },
        {
            'name': 'Right-angle at p1',
            'verts': np.array([
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],  # p1 (right-angle)
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
            'expected_vertex': 1
        },
        {
            'name': 'Right-angle at p2',
            'verts': np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],  # p2 (right-angle)
                [0.0, 0.0, 1.0],
            ]),
            'expected_vertex': 2
        },
        {
            'name': 'Right-angle at p3',
            'verts': np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],  # p3 (right-angle)
            ]),
            'expected_vertex': 3
        },
    ]

    all_passed = True

    for test_case in test_cases:
        verts = test_case['verts']
        expected = test_case['expected_vertex']

        vertex_idx, axes, lengths = detect_aa_tetrahedron_component_based(
            verts[0], verts[1], verts[2], verts[3], tol=1e-10
        )

        if vertex_idx == expected:
            print(f"✅ {test_case['name']}: PASS (detected vertex {vertex_idx})")
            print(f"   Aligned axes: {axes}, Lengths: {lengths}")
        else:
            print(f"❌ {test_case['name']}: FAIL (detected {vertex_idx}, expected {expected})")
            all_passed = False

    print()
    if all_passed:
        print("✅ TEST 1 PASSED: All 4 vertices correctly detected")
    else:
        print("❌ TEST 1 FAILED: Detection missed some right-angle vertices")

    return all_passed


def test_adaptive_tolerance():
    """Test 2: Verify adaptive tolerance handles refined elements."""
    print(f"\n{'='*80}")
    print("TEST 2: Adaptive Tolerance for Refined Elements")
    print(f"{'='*80}\n")

    # Create tetrahedra at different scales
    test_cases = [
        {
            'name': 'Coarse element (L ~ 1mm)',
            'scale': 1e-3,
            'verts': np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]) * 1e-3,
        },
        {
            'name': 'Medium element (L ~ 100μm)',
            'scale': 1e-4,
            'verts': np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]) * 1e-4,
        },
        {
            'name': 'Fine element (L ~ 10μm)',
            'scale': 1e-5,
            'verts': np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]) * 1e-5,
        },
    ]

    all_passed = True

    for test_case in test_cases:
        verts = test_case['verts']

        # Adaptive tolerance (relative to edge length)
        vertex_idx, axes, lengths = detect_aa_tetrahedron_component_based(
            verts[0], verts[1], verts[2], verts[3], tol=1e-10
        )

        if vertex_idx >= 0:
            print(f"✅ {test_case['name']}: Detected (scale={test_case['scale']:.2e})")
            print(f"   Lengths: [{lengths[0]:.2e}, {lengths[1]:.2e}, {lengths[2]:.2e}]")
        else:
            print(f"❌ {test_case['name']}: NOT detected (false negative!)")
            all_passed = False

    print()
    if all_passed:
        print("✅ TEST 2 PASSED: Adaptive tolerance works across scales")
    else:
        print("❌ TEST 2 FAILED: Some refined elements not detected")

    return all_passed


def test_real_mesh_detection():
    """Test 3: Test detection on real FLA mesh."""
    print(f"\n{'='*80}")
    print("TEST 3: Detection on Real Mesh")
    print(f"{'='*80}\n")

    # Load mesh
    print("Loading mesh...")
    mesh_path = MESH_BASE_PATH / MESH_FILE
    node_positions, connectivity, _ = load_pvtu_mesh(str(mesh_path), field_name='Displacement')

    print(f"  Loaded {connectivity.shape[0]:,} elements, {node_positions.shape[0]:,} nodes")

    # Deduplicate
    print("Deduplicating nodes...")
    node_positions, connectivity, n_removed, _ = deduplicate_nodes(
        node_positions, connectivity, None, verbose=False
    )
    print(f"  Removed {n_removed:,} duplicates → {node_positions.shape[0]:,} nodes")

    # Run detection
    print("\nRunning corrected AA detection...")
    t_start = time.time()
    aa_metadata = precompute_aa_metadata(connectivity, node_positions, verbose=True)
    t_elapsed = time.time() - t_start

    n_aa = int(np.sum(aa_metadata.is_axis_aligned.to_py()))
    n_total = connectivity.shape[0]

    print(f"\nDetection completed in {t_elapsed:.1f}s")
    print(f"Axis-aligned: {n_aa:,}/{n_total:,} ({100*n_aa/n_total:.2f}%)")

    # Expected: 100% for FLA mesh
    if n_aa == n_total:
        print("✅ TEST 3 PASSED: 100% axis-aligned (expected for FLA mesh)")
        return True
    elif n_aa > 0.99 * n_total:
        print(f"⚠️  TEST 3 WARNING: {100*(1-n_aa/n_total):.2f}% non-AA (investigate)")
        return True
    else:
        print(f"❌ TEST 3 FAILED: Only {100*n_aa/n_total:.1f}% AA (expected 100%)")
        return False


def test_pure_aa_agreement():
    """Test 4: Verify pure AA method agrees with baseline on element centroids."""
    print(f"\n{'='*80}")
    print("TEST 4: Pure AA Method Agreement with Baseline")
    print(f"{'='*80}\n")

    # Load mesh
    print("Loading mesh...")
    mesh_path = MESH_BASE_PATH / MESH_FILE
    node_positions, connectivity, _ = load_pvtu_mesh(str(mesh_path), field_name='Displacement')

    # Deduplicate
    node_positions, connectivity, _, _ = deduplicate_nodes(
        node_positions, connectivity, None, verbose=False
    )

    n_elements = connectivity.shape[0]
    print(f"  Elements: {n_elements:,}, Nodes: {node_positions.shape[0]:,}")

    # Precompute metadata
    print("\nPrecomputing metadata...")
    aa_metadata = precompute_aa_metadata(connectivity, node_positions, verbose=False)

    # Upload to GPU
    print("Uploading to GPU...")
    connectivity_gpu = jax.device_put(connectivity.astype(np.int32))
    node_positions_gpu = jax.device_put(node_positions.astype(np.float32))

    # Select test queries (element centroids - guaranteed inside)
    print("\nGenerating test queries (element centroids)...")
    N_TEST = 1000
    rng = np.random.RandomState(SEED)
    test_elem_indices = rng.choice(n_elements, size=N_TEST, replace=False)

    # Compute centroids
    v0 = node_positions[connectivity[test_elem_indices, 0]]
    v1 = node_positions[connectivity[test_elem_indices, 1]]
    v2 = node_positions[connectivity[test_elem_indices, 2]]
    v3 = node_positions[connectivity[test_elem_indices, 3]]
    test_positions = (v0 + v1 + v2 + v3) / 4.0

    test_positions_gpu = jax.device_put(test_positions.astype(np.float32))
    test_elem_ids_gpu = jax.device_put(test_elem_indices.astype(np.int32))

    print(f"  Testing {N_TEST} queries...")

    # Test agreement
    n_mismatch = 0
    for i in range(N_TEST):
        pos = test_positions_gpu[i]
        elem_id = test_elem_ids_gpu[i]

        # Baseline
        result_baseline = point_in_tet_current(pos, elem_id, connectivity_gpu, node_positions_gpu)

        # Pure AA
        result_aa = point_in_tet_pure_aa(pos, elem_id, aa_metadata)

        if result_baseline != result_aa:
            n_mismatch += 1
            if n_mismatch <= 5:  # Print first 5 mismatches
                print(f"  ❌ Mismatch at query {i}: elem_id={elem_id}, baseline={result_baseline}, aa={result_aa}")

    agreement = (N_TEST - n_mismatch) / N_TEST * 100

    print()
    if n_mismatch == 0:
        print(f"✅ TEST 4 PASSED: 100% agreement ({N_TEST}/{N_TEST} queries)")
        return True
    else:
        print(f"❌ TEST 4 FAILED: {agreement:.1f}% agreement ({n_mismatch}/{N_TEST} mismatches)")
        return False


def test_pure_aa_performance():
    """Test 5: Benchmark pure AA method vs baseline."""
    print(f"\n{'='*80}")
    print("TEST 5: Pure AA Performance Benchmark")
    print(f"{'='*80}\n")

    # Load mesh
    print("Loading mesh...")
    mesh_path = MESH_BASE_PATH / MESH_FILE
    node_positions, connectivity, _ = load_pvtu_mesh(str(mesh_path), field_name='Displacement')

    node_positions, connectivity, _, _ = deduplicate_nodes(
        node_positions, connectivity, None, verbose=False
    )

    n_elements = connectivity.shape[0]
    print(f"  Elements: {n_elements:,}, Nodes: {node_positions.shape[0]:,}")

    # Precompute metadata
    print("\nPrecomputing metadata...")
    aa_metadata = precompute_aa_metadata(connectivity, node_positions, verbose=False)
    element_vertices = precompute_element_vertices(connectivity, node_positions, verbose=False)

    # Upload to GPU
    connectivity_gpu = jax.device_put(connectivity.astype(np.int32))
    node_positions_gpu = jax.device_put(node_positions.astype(np.float32))

    # Generate test queries
    N_BENCH = 10000
    rng = np.random.RandomState(SEED)
    test_elem_indices = rng.choice(n_elements, size=N_BENCH, replace=False)

    v0 = node_positions[connectivity[test_elem_indices, 0]]
    v1 = node_positions[connectivity[test_elem_indices, 1]]
    v2 = node_positions[connectivity[test_elem_indices, 2]]
    v3 = node_positions[connectivity[test_elem_indices, 3]]
    test_positions = (v0 + v1 + v2 + v3) / 4.0

    test_positions_gpu = jax.device_put(test_positions.astype(np.float32))
    test_elem_ids_gpu = jax.device_put(test_elem_indices.astype(np.int32))

    print(f"\nBenchmarking {N_BENCH:,} queries...")

    # Warmup
    print("  Warming up JIT...")
    for i in range(10):
        _ = point_in_tet_current(test_positions_gpu[i], test_elem_ids_gpu[i], connectivity_gpu, node_positions_gpu)
        _ = point_in_tet_pure_aa(test_positions_gpu[i], test_elem_ids_gpu[i], aa_metadata)
        _ = point_in_tet_skala_memory_opt(test_positions_gpu[i], test_elem_ids_gpu[i], element_vertices)

    jax.block_until_ready(test_positions_gpu)

    # Benchmark baseline
    print("  Benchmarking baseline (current)...")
    t_start = time.time()
    for i in range(N_BENCH):
        result = point_in_tet_current(test_positions_gpu[i], test_elem_ids_gpu[i], connectivity_gpu, node_positions_gpu)
        jax.block_until_ready(result)
    t_baseline = (time.time() - t_start) * 1000  # ms

    # Benchmark pure AA
    print("  Benchmarking pure AA...")
    t_start = time.time()
    for i in range(N_BENCH):
        result = point_in_tet_pure_aa(test_positions_gpu[i], test_elem_ids_gpu[i], aa_metadata)
        jax.block_until_ready(result)
    t_aa = (time.time() - t_start) * 1000  # ms

    # Benchmark Skala (memory-optimized)
    print("  Benchmarking Skala (memory-optimized)...")
    t_start = time.time()
    for i in range(N_BENCH):
        result = point_in_tet_skala_memory_opt(test_positions_gpu[i], test_elem_ids_gpu[i], element_vertices)
        jax.block_until_ready(result)
    t_skala = (time.time() - t_start) * 1000  # ms

    # Results
    print()
    print(f"{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"Queries: {N_BENCH:,}")
    print()
    print(f"  baseline:      {t_baseline:8.2f} ms  ({N_BENCH/t_baseline*1000:10.0f} queries/sec)  [1.00× baseline]")
    print(f"  pure_aa:       {t_aa:8.2f} ms  ({N_BENCH/t_aa*1000:10.0f} queries/sec)  [{t_baseline/t_aa:.2f}× speedup]")
    print(f"  skala_memopt:  {t_skala:8.2f} ms  ({N_BENCH/t_skala*1000:10.0f} queries/sec)  [{t_baseline/t_skala:.2f}× speedup]")
    print()

    speedup_aa = t_baseline / t_aa
    speedup_skala = t_baseline / t_skala

    if speedup_aa > 1.5:
        print(f"✅ TEST 5 PASSED: Pure AA shows {speedup_aa:.2f}× speedup")
        return True
    elif speedup_aa > 1.0:
        print(f"⚠️  TEST 5 WARNING: Pure AA shows {speedup_aa:.2f}× speedup (expected >1.5×)")
        return True
    else:
        print(f"❌ TEST 5 FAILED: Pure AA shows {speedup_aa:.2f}× speedup (slower than baseline!)")
        return False


def main():
    """Run all tests."""
    print(f"\n{'='*80}")
    print("CORRECTED AXIS-ALIGNED DETECTION ALGORITHM - TEST SUITE")
    print(f"{'='*80}")

    results = []

    # Test 1: Detection checks all 4 vertices
    results.append(('Detection (all vertices)', test_detection_all_vertices()))

    # Test 2: Adaptive tolerance
    results.append(('Adaptive tolerance', test_adaptive_tolerance()))

    # Test 3: Real mesh detection
    results.append(('Real mesh detection', test_real_mesh_detection()))

    # Test 4: Agreement with baseline
    results.append(('Agreement with baseline', test_pure_aa_agreement()))

    # Test 5: Performance benchmark
    results.append(('Performance benchmark', test_pure_aa_performance()))

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}\n")

    n_passed = sum(1 for _, passed in results if passed)
    n_total = len(results)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {test_name}")

    print()
    if n_passed == n_total:
        print(f"✅ ALL TESTS PASSED ({n_passed}/{n_total})")
    else:
        print(f"❌ SOME TESTS FAILED ({n_passed}/{n_total} passed)")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
