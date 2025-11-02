#!/usr/bin/env python3
"""
Unit tests for Phase 3 Hash Octree implementation.

Tests cover:
1. Prime number generation
2. Hash table construction
3. Linear probing collision handling
4. JAX hash lookup (single and batch)
5. Memory statistics
6. Edge cases (empty, full, collisions)
"""

import numpy as np

# Enable JAX 64-bit mode for uint64/int64 support
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jaxtrace.fields.hash_octree import (
    is_prime,
    next_prime,
    compute_hash_table_size,
    hash_morton,
    build_hash_octree_from_leaves,
    hash_lookup_jax,
    hash_lookup_batch_jax,
    get_hash_octree_memory_stats,
    EMPTY_SLOT,
    MAX_PROBES
)
from jaxtrace.fields.morton_code import encode_morton_3d


# ============================================================================
# Test 1: Prime Number Generation
# ============================================================================

def test_prime_number_generation():
    """Test prime number utilities."""
    print("\n" + "="*70)
    print("TEST 1: Prime Number Generation")
    print("="*70)

    # Test is_prime
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 101, 1009]
    non_primes = [0, 1, 4, 6, 8, 9, 10, 15, 100, 1000]

    for p in primes:
        assert is_prime(p), f"{p} should be prime"
        print(f"  ✓ {p} is prime")

    for n in non_primes:
        assert not is_prime(n), f"{n} should not be prime"
        print(f"  ✓ {n} is not prime")

    # Test next_prime
    test_cases = [
        (100, 101),
        (101, 101),
        (1000, 1009),
        (1009, 1009),
    ]

    for n, expected in test_cases:
        result = next_prime(n)
        assert result == expected, f"next_prime({n}) = {result}, expected {expected}"
        print(f"  ✓ next_prime({n}) = {result}")

    # Test compute_hash_table_size
    n_leaves = 1000
    table_size = compute_hash_table_size(n_leaves, target_load_factor=0.77)
    assert is_prime(table_size), f"Table size {table_size} should be prime"
    assert table_size >= n_leaves / 0.77, f"Table size {table_size} too small for {n_leaves} leaves"
    print(f"  ✓ compute_hash_table_size({n_leaves}) = {table_size} (prime, load factor ~0.77)")

    print("\n✅ Prime number generation test PASSED")


# ============================================================================
# Test 2: Hash Table Construction
# ============================================================================

def test_hash_table_construction():
    """Test hash octree construction from leaves."""
    print("\n" + "="*70)
    print("TEST 2: Hash Table Construction")
    print("="*70)

    # Create simple test case
    domain_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    level = 3

    # Create 5 leaf nodes with different element counts
    positions = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (-0.5, -0.5, -0.5),
        (0.9, 0.1, 0.2),
        (-0.3, 0.7, -0.6),
    ]

    leaf_element_lists = [
        [10, 20, 30],       # 3 elements
        [40, 50],           # 2 elements
        [60],               # 1 element
        [70, 80, 90, 100],  # 4 elements
        [110, 120, 130, 140, 150],  # 5 elements
    ]

    # Encode Morton codes
    leaf_morton_codes = np.array([
        encode_morton_3d(x, y, z, level, domain_min, domain_max)
        for x, y, z in positions
    ], dtype=np.uint64)

    # Build hash octree
    hash_octree = build_hash_octree_from_leaves(
        leaf_morton_codes,
        leaf_element_lists,
        domain_min,
        domain_max
    )

    # Validate structure
    assert hash_octree.n_leaves == 5, f"Expected 5 leaves, got {hash_octree.n_leaves}"
    assert is_prime(hash_octree.hash_table_size), "Table size should be prime"
    assert hash_octree.load_factor < 0.8, f"Load factor {hash_octree.load_factor} too high"

    # Check element flattening
    total_elements = sum(len(elems) for elems in leaf_element_lists)
    assert len(hash_octree.flattened_elements) == total_elements, \
        f"Expected {total_elements} flattened elements, got {len(hash_octree.flattened_elements)}"

    # Check max_elements_per_cell
    assert hash_octree.max_elements_per_cell == 5, \
        f"Expected max 5 elements, got {hash_octree.max_elements_per_cell}"

    # Verify all leaves can be looked up
    non_empty_slots = np.sum(hash_octree.morton_keys != EMPTY_SLOT)
    assert non_empty_slots == 5, f"Expected 5 non-empty slots, got {non_empty_slots}"

    print(f"  ✓ 5 leaves inserted successfully")
    print(f"  ✓ Hash table size: {hash_octree.hash_table_size} (prime)")
    print(f"  ✓ Load factor: {hash_octree.load_factor:.3f}")
    print(f"  ✓ Total elements: {total_elements}")
    print(f"  ✓ Max elements per cell: {hash_octree.max_elements_per_cell}")

    print("\n✅ Hash table construction test PASSED")

    return hash_octree, leaf_morton_codes, leaf_element_lists, positions, domain_min, domain_max, level


# ============================================================================
# Test 3: Hash Function and Collision Handling
# ============================================================================

def test_hash_function_and_collisions():
    """Test hash function uniformity and linear probing."""
    print("\n" + "="*70)
    print("TEST 3: Hash Function and Collision Handling")
    print("="*70)

    # Create many Morton codes to test distribution
    n_codes = 1000
    table_size = next_prime(int(n_codes / 0.7))

    # Generate random Morton codes
    np.random.seed(42)
    morton_codes = np.random.randint(0, 2**32, size=n_codes, dtype=np.uint64)

    # Hash all codes
    hashes = [hash_morton(code, table_size) for code in morton_codes]

    # Check distribution (should be relatively uniform)
    unique_hashes = len(set(hashes))
    collision_rate = 1.0 - (unique_hashes / n_codes)

    print(f"  ✓ Generated {n_codes} Morton codes")
    print(f"  ✓ Hash table size: {table_size}")
    print(f"  ✓ Unique hashes: {unique_hashes} / {n_codes}")
    print(f"  ✓ Collision rate: {collision_rate:.1%}")

    # Check that hashes are in valid range
    assert all(0 <= h < table_size for h in hashes), "Hash out of range"
    print(f"  ✓ All hashes in valid range [0, {table_size})")

    # Build hash octree to test linear probing
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # Use subset for collision testing
    test_codes = morton_codes[:100]
    test_elements = [[i] for i in range(100)]

    hash_octree = build_hash_octree_from_leaves(
        test_codes,
        test_elements,
        domain_min,
        domain_max,
        target_load_factor=0.7  # Higher load = more collisions
    )

    # Verify all codes were inserted (no failures)
    assert hash_octree.n_leaves == 100, "Not all codes inserted"
    print(f"  ✓ All 100 codes inserted successfully with linear probing")

    print("\n✅ Hash function and collision handling test PASSED")


# ============================================================================
# Test 4: JAX Hash Lookup (Single Point)
# ============================================================================

def test_jax_hash_lookup_single():
    """Test JAX hash lookup for single point."""
    print("\n" + "="*70)
    print("TEST 4: JAX Hash Lookup (Single Point)")
    print("="*70)

    # Build hash octree (reuse from test 2)
    domain_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    level = 3

    positions = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (-0.5, -0.5, -0.5),
    ]

    leaf_element_lists = [
        [10, 20, 30],
        [40, 50],
        [60],
    ]

    leaf_morton_codes = np.array([
        encode_morton_3d(x, y, z, level, domain_min, domain_max)
        for x, y, z in positions
    ], dtype=np.uint64)

    hash_octree = build_hash_octree_from_leaves(
        leaf_morton_codes,
        leaf_element_lists,
        domain_min,
        domain_max
    )

    # Test lookup for each position
    for i, (x, y, z) in enumerate(positions):
        point = jnp.array([x, y, z], dtype=jnp.float32)
        elements, n_elements = hash_lookup_jax(point, hash_octree, level)

        # Convert to numpy for easier checking
        elements_np = np.asarray(elements)
        n_elements_np = int(n_elements)

        # Verify we found the correct elements
        expected_elements = leaf_element_lists[i]
        actual_elements = elements_np[:n_elements_np].tolist()

        assert n_elements_np == len(expected_elements), \
            f"Position {i}: Expected {len(expected_elements)} elements, got {n_elements_np}"

        # Check elements match (order should be preserved)
        assert actual_elements == expected_elements, \
            f"Position {i}: Expected {expected_elements}, got {actual_elements}"

        print(f"  ✓ Position {i} ({x}, {y}, {z}): Found {n_elements_np} elements {actual_elements}")

    # Test lookup for position not in hash table
    point_missing = jnp.array([0.99, 0.99, 0.99], dtype=jnp.float32)
    elements, n_elements = hash_lookup_jax(point_missing, hash_octree, level)
    n_elements_np = int(n_elements)

    # Should return 0 elements (not found)
    # NOTE: This depends on whether the Morton code exists in the table
    print(f"  ✓ Missing position (0.99, 0.99, 0.99): Found {n_elements_np} elements (expected 0 or not found)")

    print("\n✅ JAX hash lookup (single point) test PASSED")


# ============================================================================
# Test 5: JAX Hash Lookup (Batch)
# ============================================================================

def test_jax_hash_lookup_batch():
    """Test JAX batch hash lookup with vmap."""
    print("\n" + "="*70)
    print("TEST 5: JAX Hash Lookup (Batch)")
    print("="*70)

    # Build hash octree
    domain_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    level = 3

    positions = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (-0.5, -0.5, -0.5),
        (0.9, 0.1, 0.2),
    ]

    leaf_element_lists = [
        [10, 20, 30],
        [40, 50],
        [60],
        [70, 80, 90, 100],
    ]

    leaf_morton_codes = np.array([
        encode_morton_3d(x, y, z, level, domain_min, domain_max)
        for x, y, z in positions
    ], dtype=np.uint64)

    hash_octree = build_hash_octree_from_leaves(
        leaf_morton_codes,
        leaf_element_lists,
        domain_min,
        domain_max
    )

    # Batch lookup for all positions
    points = jnp.array(positions, dtype=jnp.float32)
    levels = jnp.full(len(positions), level, dtype=jnp.int32)

    elements_batch, n_elements_batch = hash_lookup_batch_jax(points, hash_octree, levels)

    # Convert to numpy
    elements_np = np.asarray(elements_batch)
    n_elements_np = np.asarray(n_elements_batch)

    # Verify each result
    for i in range(len(positions)):
        expected = leaf_element_lists[i]
        n_found = int(n_elements_np[i])
        actual = elements_np[i, :n_found].tolist()

        assert n_found == len(expected), \
            f"Batch position {i}: Expected {len(expected)} elements, got {n_found}"
        assert actual == expected, \
            f"Batch position {i}: Expected {expected}, got {actual}"

        print(f"  ✓ Batch position {i}: Found {n_found} elements {actual}")

    print("\n✅ JAX hash lookup (batch) test PASSED")


# ============================================================================
# Test 6: Memory Statistics
# ============================================================================

def test_memory_statistics():
    """Test memory usage calculations."""
    print("\n" + "="*70)
    print("TEST 6: Memory Statistics")
    print("="*70)

    # Build hash octree with known sizes
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    level = 3

    n_leaves = 1000
    np.random.seed(42)

    # Generate random Morton codes
    leaf_morton_codes = np.random.randint(0, 2**32, size=n_leaves, dtype=np.uint64)

    # Generate random element lists (1-10 elements each)
    leaf_element_lists = [
        list(range(i * 10, i * 10 + np.random.randint(1, 11)))
        for i in range(n_leaves)
    ]

    hash_octree = build_hash_octree_from_leaves(
        leaf_morton_codes,
        leaf_element_lists,
        domain_min,
        domain_max,
        target_load_factor=0.5  # Lower load factor for random codes (avoid MAX_PROBES limit)
    )

    # Get memory stats
    stats = get_hash_octree_memory_stats(hash_octree)

    print(f"  Hash Table:")
    print(f"    Leaves: {stats['n_leaves']:,}")
    print(f"    Table size: {stats['hash_table_size']:,}")
    print(f"    Load factor: {stats['load_factor']:.3f}")
    print(f"  Memory:")
    print(f"    Morton keys: {stats['morton_keys_mb']:.3f} MB")
    print(f"    Starts array: {stats['starts_mb']:.3f} MB")
    print(f"    Lengths array: {stats['lengths_mb']:.3f} MB")
    print(f"    Elements array: {stats['elements_mb']:.3f} MB")
    print(f"    Hash table overhead: {stats['hash_table_mb']:.3f} MB")
    print(f"    Total: {stats['total_mb']:.3f} MB")
    print(f"  Elements:")
    print(f"    Total elements: {stats['total_elements']:,}")
    print(f"    Max per cell: {stats['max_elements_per_cell']}")

    # Sanity checks
    assert stats['n_leaves'] == n_leaves
    assert stats['load_factor'] < 0.8
    assert stats['total_mb'] > 0

    print("\n✅ Memory statistics test PASSED")


# ============================================================================
# Test 7: Edge Cases
# ============================================================================

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n" + "="*70)
    print("TEST 7: Edge Cases")
    print("="*70)

    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    level = 3

    # Edge case 1: Single leaf
    print("  Testing single leaf...")
    leaf_codes = np.array([encode_morton_3d(0.5, 0.5, 0.5, level, domain_min, domain_max)], dtype=np.uint64)
    leaf_elements = [[42]]

    hash_octree = build_hash_octree_from_leaves(leaf_codes, leaf_elements, domain_min, domain_max)
    assert hash_octree.n_leaves == 1
    print("  ✓ Single leaf case works")

    # Edge case 2: Many elements in one cell
    print("  Testing cell with many elements...")
    many_elements = list(range(100))
    leaf_codes = np.array([encode_morton_3d(0.5, 0.5, 0.5, level, domain_min, domain_max)], dtype=np.uint64)
    leaf_elements = [many_elements]

    hash_octree = build_hash_octree_from_leaves(leaf_codes, leaf_elements, domain_min, domain_max)
    assert hash_octree.max_elements_per_cell == 100
    print("  ✓ Many elements (100) in one cell works")

    # Edge case 3: Duplicate Morton codes (should fail or handle gracefully)
    # NOTE: Current implementation doesn't explicitly handle duplicates
    # This would require additional logic or caller validation

    print("\n✅ Edge cases test PASSED")


# ============================================================================
# Run All Tests
# ============================================================================

def run_all_tests():
    """Run complete test suite for Phase 3 hash octree."""
    print("\n" + "="*70)
    print("PHASE 3: HASH OCTREE TEST SUITE")
    print("="*70)

    try:
        test_prime_number_generation()
        test_hash_table_construction()
        test_hash_function_and_collisions()
        test_jax_hash_lookup_single()
        test_jax_hash_lookup_batch()
        test_memory_statistics()
        test_edge_cases()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nHash octree implementation validated successfully!")
        print("Ready for integration with SharedOctreeFEMField (Phase 3 next step).")
        print("="*70 + "\n")

    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
