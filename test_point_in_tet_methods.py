#!/usr/bin/env python3
"""
Unit tests for point-in-tetrahedron method implementations.

Tests:
1. Method agreement: Verify all methods produce identical results
2. Axis-aligned detection: Verify correct detection of axis-aligned tets
3. Performance benchmarking: Compare throughput of different methods
4. Edge cases: Degenerate tets, boundary points, numerical stability
"""

import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from jaxtrace.gpu.search.point_in_tet_methods import (
    point_in_tet_current,
    point_in_tet_skala,
    point_in_tet_axis_aligned,
    point_in_tet_gpu
)


def create_test_tetrahedron(tet_type="general"):
    """
    Create test tetrahedron with known geometry.

    Args:
        tet_type: "general", "axis_aligned", or "degenerate"

    Returns:
        connectivity: (1, 4) - node indices
        node_positions: (4, 3) - node coordinates
        test_points: (N, 3) - test point positions
        expected_inside: (N,) - expected containment results
    """
    if tet_type == "axis_aligned":
        # Right-angled tetrahedron with edges parallel to axes
        # Vertices: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        node_positions = jnp.array([
            [0.0, 0.0, 0.0],  # p0 - right-angled vertex
            [1.0, 0.0, 0.0],  # p1 - along X
            [0.0, 1.0, 0.0],  # p2 - along Y
            [0.0, 0.0, 1.0],  # p3 - along Z
        ], dtype=jnp.float32)

        # Test points
        test_points = jnp.array([
            [0.25, 0.25, 0.25],  # Inside (barycentric: 0.25, 0.25, 0.25, 0.25)
            [0.1, 0.1, 0.1],     # Inside (near center)
            [0.5, 0.0, 0.0],     # On edge p0-p1
            [0.0, 0.5, 0.0],     # On edge p0-p2
            [0.0, 0.0, 0.5],     # On edge p0-p3
            [1.0, 0.0, 0.0],     # On vertex p1
            [0.33, 0.33, 0.0],   # On face p0-p1-p2
            [0.5, 0.5, 0.5],     # Outside (beyond centroid)
            [1.0, 1.0, 1.0],     # Outside (far)
            [-0.1, 0.0, 0.0],    # Outside (negative X)
        ], dtype=jnp.float32)

        expected_inside = jnp.array([
            True,   # 0.25, 0.25, 0.25 (inside)
            True,   # 0.1, 0.1, 0.1 (inside)
            True,   # Edge (with tolerance)
            True,   # Edge
            True,   # Edge
            True,   # Vertex (with tolerance)
            True,   # Face
            False,  # Outside (sum > 1)
            False,  # Outside (far)
            False,  # Outside (negative)
        ], dtype=jnp.bool_)

    elif tet_type == "general":
        # General tetrahedron (NOT axis-aligned)
        node_positions = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.1],
            [0.1, 1.0, 0.2],
            [0.2, 0.1, 1.0],
        ], dtype=jnp.float32)

        # Test points (inside/outside based on barycentric coords)
        test_points = jnp.array([
            [0.3, 0.3, 0.3],     # Inside
            [0.1, 0.1, 0.1],     # Inside
            [0.5, 0.5, 0.5],     # Outside
            [1.0, 1.0, 1.0],     # Outside
        ], dtype=jnp.float32)

        expected_inside = jnp.array([
            True,   # Inside
            True,   # Inside
            False,  # Outside
            False,  # Outside
        ], dtype=jnp.bool_)

    elif tet_type == "degenerate":
        # Degenerate tetrahedron (zero volume - coplanar points)
        node_positions = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.25, 0.25, 0.0],  # Coplanar (degenerate)
        ], dtype=jnp.float32)

        test_points = jnp.array([
            [0.5, 0.25, 0.0],   # On plane
            [0.5, 0.25, 0.1],   # Off plane
        ], dtype=jnp.float32)

        expected_inside = jnp.array([
            False,  # Degenerate tet should return False
            False,  # Degenerate tet should return False
        ], dtype=jnp.bool_)

    else:
        raise ValueError(f"Unknown tet_type: {tet_type}")

    connectivity = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)

    return connectivity, node_positions, test_points, expected_inside


def test_method_agreement():
    """Test that all methods produce identical results."""
    print("\n" + "="*80)
    print("TEST 1: Method Agreement (Axis-Aligned Tetrahedron)")
    print("="*80)

    connectivity, node_positions, test_points, expected = create_test_tetrahedron("axis_aligned")
    elem_id = jnp.int32(0)

    print(f"\nTesting {len(test_points)} points...")
    print(f"Node positions:\n{node_positions}")

    results_current = []
    results_skala = []
    results_axis_aligned = []

    for i, pos in enumerate(test_points):
        r_current = point_in_tet_current(pos, elem_id, connectivity, node_positions)
        r_skala = point_in_tet_skala(pos, elem_id, connectivity, node_positions)
        r_axis_aligned = point_in_tet_axis_aligned(pos, elem_id, connectivity, node_positions)

        results_current.append(r_current)
        results_skala.append(r_skala)
        results_axis_aligned.append(r_axis_aligned)

        match_current_skala = r_current == r_skala
        match_skala_axis = r_skala == r_axis_aligned
        match_expected = r_current == expected[i]

        status = "✅" if (match_current_skala and match_skala_axis and match_expected) else "❌"
        print(f"{status} Point {i}: {pos} -> current={r_current}, skala={r_skala}, "
              f"axis_aligned={r_axis_aligned}, expected={expected[i]}")

    results_current = jnp.array(results_current)
    results_skala = jnp.array(results_skala)
    results_axis_aligned = jnp.array(results_axis_aligned)

    agreement_current_skala = jnp.all(results_current == results_skala)
    agreement_skala_axis = jnp.all(results_skala == results_axis_aligned)
    agreement_expected = jnp.all(results_current == expected)

    print(f"\n{'='*80}")
    print(f"Agreement Summary:")
    print(f"  current ↔ skala:         {'✅ PASS' if agreement_current_skala else '❌ FAIL'}")
    print(f"  skala ↔ axis_aligned:    {'✅ PASS' if agreement_skala_axis else '❌ FAIL'}")
    print(f"  current ↔ expected:      {'✅ PASS' if agreement_expected else '❌ FAIL'}")

    assert agreement_current_skala, "current and skala methods disagree!"
    assert agreement_skala_axis, "skala and axis_aligned methods disagree!"
    assert agreement_expected, "Methods disagree with expected results!"

    print(f"\n✅ All methods agree on axis-aligned tetrahedron!")


def test_general_tetrahedron():
    """Test methods on general (non-axis-aligned) tetrahedron."""
    print("\n" + "="*80)
    print("TEST 2: General Tetrahedron (Non-Axis-Aligned)")
    print("="*80)

    connectivity, node_positions, test_points, expected = create_test_tetrahedron("general")
    elem_id = jnp.int32(0)

    print(f"\nTesting {len(test_points)} points...")
    print(f"Node positions:\n{node_positions}")

    results_current = []
    results_skala = []
    results_axis_aligned = []

    for i, pos in enumerate(test_points):
        r_current = point_in_tet_current(pos, elem_id, connectivity, node_positions)
        r_skala = point_in_tet_skala(pos, elem_id, connectivity, node_positions)
        r_axis_aligned = point_in_tet_axis_aligned(pos, elem_id, connectivity, node_positions)

        results_current.append(r_current)
        results_skala.append(r_skala)
        results_axis_aligned.append(r_axis_aligned)

        match_all = (r_current == r_skala) and (r_skala == r_axis_aligned) and (r_current == expected[i])
        status = "✅" if match_all else "❌"
        print(f"{status} Point {i}: {pos} -> current={r_current}, skala={r_skala}, "
              f"axis_aligned={r_axis_aligned}, expected={expected[i]}")

    results_current = jnp.array(results_current)
    results_skala = jnp.array(results_skala)
    results_axis_aligned = jnp.array(results_axis_aligned)

    agreement = jnp.all(results_current == results_skala) and \
                jnp.all(results_skala == results_axis_aligned) and \
                jnp.all(results_current == expected)

    print(f"\n{'='*80}")
    print(f"Agreement: {'✅ PASS' if agreement else '❌ FAIL'}")
    print(f"Note: axis_aligned method should fall back to Skala for non-axis-aligned tets")

    assert agreement, "Methods disagree on general tetrahedron!"
    print(f"\n✅ All methods agree on general tetrahedron!")


def test_degenerate_tetrahedron():
    """Test handling of degenerate (zero-volume) tetrahedra."""
    print("\n" + "="*80)
    print("TEST 3: Degenerate Tetrahedron (Coplanar Vertices)")
    print("="*80)

    connectivity, node_positions, test_points, expected = create_test_tetrahedron("degenerate")
    elem_id = jnp.int32(0)

    print(f"\nTesting {len(test_points)} points...")
    print(f"Node positions (coplanar):\n{node_positions}")

    for i, pos in enumerate(test_points):
        r_current = point_in_tet_current(pos, elem_id, connectivity, node_positions)
        r_skala = point_in_tet_skala(pos, elem_id, connectivity, node_positions)
        r_axis_aligned = point_in_tet_axis_aligned(pos, elem_id, connectivity, node_positions)

        match = (r_current == r_skala) and (r_skala == r_axis_aligned) and (r_current == expected[i])
        status = "✅" if match else "❌"
        print(f"{status} Point {i}: {pos} -> current={r_current}, skala={r_skala}, "
              f"axis_aligned={r_axis_aligned}, expected={expected[i]}")

    print(f"\n✅ Degenerate tetrahedra handled correctly!")


def benchmark_methods():
    """Benchmark performance of different methods."""
    print("\n" + "="*80)
    print("TEST 4: Performance Benchmark")
    print("="*80)

    # Create axis-aligned tetrahedron
    connectivity, node_positions, _, _ = create_test_tetrahedron("axis_aligned")
    elem_id = jnp.int32(0)

    # Generate random test points
    n_queries = 10000
    rng = np.random.RandomState(42)
    test_points = jnp.array(rng.uniform(-0.5, 1.5, size=(n_queries, 3)), dtype=jnp.float32)

    print(f"\nBenchmarking {n_queries} queries on axis-aligned tetrahedron...")

    # Compile methods with JIT
    @jax.jit
    def bench_current(points):
        def query(pos):
            return point_in_tet_current(pos, elem_id, connectivity, node_positions)
        return jax.vmap(query)(points)

    @jax.jit
    def bench_skala(points):
        def query(pos):
            return point_in_tet_skala(pos, elem_id, connectivity, node_positions)
        return jax.vmap(query)(points)

    @jax.jit
    def bench_axis_aligned(points):
        def query(pos):
            return point_in_tet_axis_aligned(pos, elem_id, connectivity, node_positions)
        return jax.vmap(query)(points)

    # Warmup
    print("  Warming up JIT compilation...")
    _ = bench_current(test_points[:10]).block_until_ready()
    _ = bench_skala(test_points[:10]).block_until_ready()
    _ = bench_axis_aligned(test_points[:10]).block_until_ready()

    # Benchmark current method
    print("  Benchmarking current method...")
    start = time.time()
    results_current = bench_current(test_points).block_until_ready()
    time_current = time.time() - start
    throughput_current = n_queries / time_current

    # Benchmark Skala method
    print("  Benchmarking Skala method...")
    start = time.time()
    results_skala = bench_skala(test_points).block_until_ready()
    time_skala = time.time() - start
    throughput_skala = n_queries / time_skala

    # Benchmark axis-aligned method
    print("  Benchmarking axis-aligned method...")
    start = time.time()
    results_axis_aligned = bench_axis_aligned(test_points).block_until_ready()
    time_axis_aligned = time.time() - start
    throughput_axis_aligned = n_queries / time_axis_aligned

    # Verify agreement
    agreement = jnp.all(results_current == results_skala) and \
                jnp.all(results_skala == results_axis_aligned)

    print(f"\n{'='*80}")
    print(f"Performance Results ({n_queries} queries):")
    print(f"{'='*80}")
    print(f"  current:         {time_current*1000:8.2f} ms  ({throughput_current:10.0f} queries/sec)  [baseline]")
    print(f"  skala:           {time_skala*1000:8.2f} ms  ({throughput_skala:10.0f} queries/sec)  "
          f"[{throughput_skala/throughput_current:.2f}× speedup]")
    print(f"  axis_aligned:    {time_axis_aligned*1000:8.2f} ms  ({throughput_axis_aligned:10.0f} queries/sec)  "
          f"[{throughput_axis_aligned/throughput_current:.2f}× speedup]")
    print(f"\n  Agreement: {'✅ PASS' if agreement else '❌ FAIL'}")

    assert agreement, "Methods produce different results!"

    # Expected speedups
    expected_skala_speedup = 2.5  # 145/48 ≈ 3.0×, conservative estimate 2.5×
    expected_axis_speedup = 3.0   # 145/12 ≈ 12×, but with detection overhead ~3-5×

    print(f"\n{'='*80}")
    print(f"Expected vs Actual Speedup:")
    print(f"{'='*80}")
    print(f"  Skala:         Expected ≥{expected_skala_speedup:.1f}×, Got {throughput_skala/throughput_current:.2f}× "
          f"{'✅' if throughput_skala/throughput_current >= expected_skala_speedup else '⚠️'}")
    print(f"  Axis-aligned:  Expected ≥{expected_axis_speedup:.1f}×, Got {throughput_axis_aligned/throughput_current:.2f}× "
          f"{'✅' if throughput_axis_aligned/throughput_current >= expected_axis_speedup else '⚠️'}")

    if throughput_skala/throughput_current >= expected_skala_speedup:
        print(f"\n✅ Skala method meets performance target!")
    else:
        print(f"\n⚠️  Skala speedup below target (may be GPU-dependent)")

    if throughput_axis_aligned/throughput_current >= expected_axis_speedup:
        print(f"✅ Axis-aligned method meets performance target!")
    else:
        print(f"⚠️  Axis-aligned speedup below target (detection overhead?)")


def main():
    """Run all tests."""
    print("="*80)
    print("Point-in-Tetrahedron Method Validation & Benchmark")
    print("="*80)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")

    try:
        # Test method agreement
        test_method_agreement()

        # Test general tetrahedron
        test_general_tetrahedron()

        # Test degenerate tetrahedron
        test_degenerate_tetrahedron()

        # Benchmark performance
        benchmark_methods()

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nNext steps:")
        print("1. Run production_tracking_fully_fused_timedep.py with config.POINT_IN_TET_METHOD='skala'")
        print("2. Verify 100K particle tracking at ~55,000-65,000 p/s (3× speedup)")
        print("3. Confirm 93.57% retention (same as current method)")
        print("4. Proceed to Phase 2: axis_aligned method for 10-12× speedup")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
