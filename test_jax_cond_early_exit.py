"""
Test whether jax.lax.cond provides early exit for expensive operations.

This benchmark tests if JAX's lax.cond can skip expensive operations when the
condition evaluates to true, which is critical for the proposed single-particle
architecture where we want to skip L2 octree search for particles already found
by L0+L1.

Expected outcomes:
1. If lax.cond provides early exit: time_cond << time_where (significant speedup)
2. If lax.cond compiles both branches: time_cond ≈ time_where (no speedup)
"""

import jax
import jax.numpy as jnp
import time
import numpy as np


def expensive_operation(x):
    """
    Simulate expensive operation similar to octree scan.

    This mimics the computational cost of traversing an octree with lax.scan.
    Each iteration performs trigonometric operations similar to point-in-tet checks.
    """
    result = x
    for _ in range(100):  # Simulate 100 iterations of expensive computation
        result = jnp.sin(result) + jnp.cos(result) + jnp.sqrt(jnp.abs(result) + 1e-6)
    return result


@jax.jit
def search_with_cond_single(pos, element_id_cached):
    """
    Single-particle search using lax.cond for early exit.

    This is the proposed architecture:
    - If cached element still contains particle: return immediately
    - Else: perform expensive L2 search

    Parameters
    ----------
    pos : jax.Array, shape (3,)
        Particle position
    element_id_cached : jax.Array, scalar int32
        Cached element ID (-1 if not found)

    Returns
    -------
    element_id : jax.Array, scalar int32
        Found element ID
    """
    # Simulate L0 check (cheap)
    found_in_l0 = element_id_cached >= 0

    def return_cached(_):
        """Particle still in cached element - return immediately."""
        return element_id_cached

    def do_expensive_search(_):
        """Particle not in cached element - do expensive L2 search."""
        # Simulate expensive octree search
        result = expensive_operation(pos[0])
        return jnp.where(result > 0, jnp.int32(42), jnp.int32(-1))

    return jax.lax.cond(
        found_in_l0,
        return_cached,
        do_expensive_search,
        None
    )


@jax.jit
def search_with_where_single(pos, element_id_cached):
    """
    Single-particle search using jnp.where (current approach).

    This is the current architecture:
    - Always perform expensive L2 search
    - Use jnp.where to select output based on condition

    Parameters
    ----------
    pos : jax.Array, shape (3,)
        Particle position
    element_id_cached : jax.Array, scalar int32
        Cached element ID (-1 if not found)

    Returns
    -------
    element_id : jax.Array, scalar int32
        Found element ID
    """
    # Simulate L0 check (cheap)
    found_in_l0 = element_id_cached >= 0

    # Always do expensive search (cannot skip)
    result = expensive_operation(pos[0])
    element_id_l2 = jnp.where(result > 0, jnp.int32(42), jnp.int32(-1))

    # Merge using jnp.where
    return jnp.where(found_in_l0, element_id_cached, element_id_l2)


def benchmark_early_exit():
    """
    Benchmark lax.cond vs jnp.where for early exit capability.

    Scenario: 45,000 particles, 99.5% already found by L0+L1 (44,775 found, 225 unfound)
    - Found particles should skip expensive L2 octree search
    - Unfound particles must perform expensive L2 search

    Test:
    - lax.cond approach: Should only execute expensive_operation for 225 particles
    - jnp.where approach: Must execute expensive_operation for all 45,000 particles
    """
    print("=" * 80)
    print("JAX lax.cond Early Exit Benchmark")
    print("=" * 80)
    print()
    print("Scenario: 45,000 particles, 99.5% found by L0+L1, 0.5% need L2 octree")
    print()

    # Setup: 45,000 particles
    N = 45000
    positions = jnp.ones((N, 3))

    # 99.5% found (element_id >= 0), 0.5% unfound (element_id = -1)
    n_found = int(N * 0.995)
    n_unfound = N - n_found

    element_ids_cached = jnp.concatenate([
        jnp.ones(n_found, dtype=jnp.int32) * 100,  # Found: element_id = 100
        jnp.ones(n_unfound, dtype=jnp.int32) * -1   # Unfound: element_id = -1
    ])

    print(f"Total particles: {N:,}")
    print(f"Found by L0+L1: {n_found:,} ({100*n_found/N:.1f}%)")
    print(f"Need L2 search: {n_unfound:,} ({100*n_unfound/N:.1f}%)")
    print()

    # Warm-up JIT compilation
    print("Warming up JIT compilation...")
    _ = jax.vmap(search_with_cond_single)(positions[:100], element_ids_cached[:100])
    _ = jax.vmap(search_with_where_single)(positions[:100], element_ids_cached[:100])
    print("✓ JIT compilation complete")
    print()

    # Benchmark lax.cond approach
    print("Benchmarking lax.cond approach (proposed single-particle architecture)...")
    start = time.time()
    result_cond = jax.vmap(search_with_cond_single)(positions, element_ids_cached)
    result_cond.block_until_ready()  # Wait for GPU to finish
    time_cond = time.time() - start
    print(f"  Time: {time_cond:.4f} s")
    print(f"  Throughput: {N/time_cond:,.0f} particles/s")
    print()

    # Benchmark jnp.where approach
    print("Benchmarking jnp.where approach (current batch-level architecture)...")
    start = time.time()
    result_where = jax.vmap(search_with_where_single)(positions, element_ids_cached)
    result_where.block_until_ready()  # Wait for GPU to finish
    time_where = time.time() - start
    print(f"  Time: {time_where:.4f} s")
    print(f"  Throughput: {N/time_where:,.0f} particles/s")
    print()

    # Analysis
    speedup = time_where / time_cond
    print("=" * 80)
    print("Results")
    print("=" * 80)
    print(f"lax.cond time:  {time_cond:.4f} s")
    print(f"jnp.where time: {time_where:.4f} s")
    print(f"Speedup:        {speedup:.2f}×")
    print()

    # Verify correctness
    correct = jnp.allclose(result_cond, result_where)
    print(f"Results match:  {correct}")
    print()

    # Interpretation
    print("=" * 80)
    print("Interpretation")
    print("=" * 80)

    if speedup > 5.0:
        print("✓ SIGNIFICANT SPEEDUP DETECTED")
        print()
        print("lax.cond provides early exit! JAX skips expensive operations when")
        print("condition is true. This means the proposed single-particle architecture")
        print("with lax.cond can skip L2 octree search for particles already found by L0+L1.")
        print()
        print("RECOMMENDATION: Implement single-particle architecture.")
        print(f"Expected speedup: {speedup:.0f}× faster for L2 search")
        print(f"Expected overall throughput: {N/time_cond:,.0f} p/s (from current 3,109 p/s)")

    elif speedup > 1.2:
        print("⚠ MODERATE SPEEDUP DETECTED")
        print()
        print("lax.cond provides some benefit, but not as much as expected.")
        print("XLA may be partially optimizing early exit, but expensive operations")
        print("are still being executed for some found particles.")
        print()
        print("RECOMMENDATION: Run additional benchmarks with real octree scan operations.")
        print(f"Speedup: {speedup:.2f}× (expected ~200× if full early exit)")

    else:
        print("✗ NO SPEEDUP DETECTED")
        print()
        print("lax.cond does NOT provide early exit. JAX compiles both branches")
        print("and executes expensive operations regardless of condition value.")
        print("The proposed single-particle architecture will NOT improve performance.")
        print()
        print("RECOMMENDATION: Abandon octree, use block-based L2 fallback instead.")
        print("Expected throughput: 40-48k p/s (from hierarchical 4-hop baseline)")

    print("=" * 80)

    return {
        'time_cond': time_cond,
        'time_where': time_where,
        'speedup': speedup,
        'n_particles': N,
        'n_found': n_found,
        'n_unfound': n_unfound
    }


def benchmark_with_real_scan():
    """
    Benchmark with real lax.scan operation (closer to actual octree traversal).

    This test uses actual lax.scan loops to simulate octree traversal,
    providing a more realistic measure of early exit capability.
    """
    print()
    print("=" * 80)
    print("Benchmark with Real lax.scan (Octree-like)")
    print("=" * 80)
    print()

    def octree_scan_expensive(pos):
        """Simulate octree traversal with lax.scan."""
        def step(carry, _):
            node_id, found_id = carry
            # Simulate expensive point-in-tet checks
            check_result = jnp.sin(pos[0] + node_id) + jnp.cos(pos[1]) + jnp.sqrt(jnp.abs(pos[2]) + 1e-6)
            new_found_id = jnp.where(check_result > 0.5, jnp.int32(42), found_id)
            new_node_id = node_id + 1
            return (new_node_id, new_found_id), None

        (_, element_id), _ = jax.lax.scan(
            step,
            (jnp.int32(0), jnp.int32(-1)),
            None,
            length=10  # 10 iterations like real octree
        )
        return element_id

    @jax.jit
    def search_with_cond_scan(pos, element_id_cached):
        """Search with lax.cond wrapping lax.scan."""
        found_in_l0 = element_id_cached >= 0

        def return_cached(_):
            return element_id_cached

        def do_scan(_):
            return octree_scan_expensive(pos)

        return jax.lax.cond(found_in_l0, return_cached, do_scan, None)

    @jax.jit
    def search_with_where_scan(pos, element_id_cached):
        """Search with jnp.where and lax.scan."""
        found_in_l0 = element_id_cached >= 0
        element_id_l2 = octree_scan_expensive(pos)
        return jnp.where(found_in_l0, element_id_cached, element_id_l2)

    # Setup: 45,000 particles, 99.5% found
    N = 45000
    positions = jnp.ones((N, 3))
    n_found = int(N * 0.995)
    element_ids_cached = jnp.concatenate([
        jnp.ones(n_found, dtype=jnp.int32) * 100,
        jnp.ones(N - n_found, dtype=jnp.int32) * -1
    ])

    print(f"Particles: {N:,} (99.5% found, 0.5% unfound)")
    print("Using real lax.scan with 10 iterations per particle")
    print()

    # Warm-up
    print("Warming up...")
    _ = jax.vmap(search_with_cond_scan)(positions[:100], element_ids_cached[:100])
    _ = jax.vmap(search_with_where_scan)(positions[:100], element_ids_cached[:100])
    print("✓ Ready")
    print()

    # Benchmark
    print("Benchmarking lax.cond + lax.scan...")
    start = time.time()
    result_cond = jax.vmap(search_with_cond_scan)(positions, element_ids_cached)
    result_cond.block_until_ready()
    time_cond = time.time() - start
    print(f"  Time: {time_cond:.4f} s")

    print("Benchmarking jnp.where + lax.scan...")
    start = time.time()
    result_where = jax.vmap(search_with_where_scan)(positions, element_ids_cached)
    result_where.block_until_ready()
    time_where = time.time() - start
    print(f"  Time: {time_where:.4f} s")
    print()

    speedup = time_where / time_cond
    print(f"Speedup with real scan: {speedup:.2f}×")
    print()

    if speedup > 5.0:
        print("✓ lax.cond skips lax.scan when condition is true!")
    else:
        print("✗ lax.cond does NOT skip lax.scan (both branches execute)")

    print("=" * 80)

    return {
        'time_cond_scan': time_cond,
        'time_where_scan': time_where,
        'speedup_scan': speedup
    }


if __name__ == "__main__":
    # Run simple benchmark
    results1 = benchmark_early_exit()

    # Run benchmark with real lax.scan
    results2 = benchmark_with_real_scan()

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Simple operations speedup:     {results1['speedup']:.2f}×")
    print(f"With lax.scan speedup:         {results2['speedup_scan']:.2f}×")
    print()

    if results2['speedup_scan'] > 5.0:
        print("CONCLUSION: Single-particle architecture with lax.cond WILL improve performance.")
        print("            Implement the proposed architecture redesign.")
    else:
        print("CONCLUSION: Single-particle architecture with lax.cond will NOT improve performance.")
        print("            Abandon octree, use block-based L2 fallback instead.")
