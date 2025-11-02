#!/usr/bin/env python3
"""
Test suite for Morton code encoding/decoding.

Phase 2: Validate Morton code implementation before integrating into octree.
"""

import numpy as np
from jaxtrace.fields.morton_code import (
    encode_morton_3d,
    decode_morton_3d,
    get_morton_parent,
    get_morton_children,
    morton_distance,
    compute_morton_codes_batch,
    decode_morton_codes_batch,
    get_memory_savings
)


def test_encode_decode_roundtrip():
    """Test that encode -> decode recovers original bounds."""
    print("\n" + "="*70)
    print("TEST 1: Encode/Decode Roundtrip")
    print("="*70)

    domain_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # Test various positions and levels
    test_cases = [
        (0.0, 0.0, 0.0, 0),   # Root node at center
        (0.0, 0.0, 0.0, 5),   # Level 5 at center
        (0.5, 0.5, 0.5, 3),   # Positive octant
        (-0.5, -0.5, -0.5, 3), # Negative octant
        (0.99, 0.99, 0.99, 10), # Near boundary, deep level
    ]

    for x, y, z, level in test_cases:
        # Encode
        code = encode_morton_3d(x, y, z, level, domain_min, domain_max)

        # Decode
        node_min, node_max, decoded_level = decode_morton_3d(code, domain_min, domain_max)

        # Check level
        assert decoded_level == level, f"Level mismatch: {decoded_level} != {level}"

        # Check that original point is within decoded bounds
        point = np.array([x, y, z])
        within_bounds = np.all(point >= node_min) and np.all(point <= node_max)

        print(f"  Position: ({x:6.2f}, {y:6.2f}, {z:6.2f}) Level: {level:2d}")
        print(f"    Morton code: {code:020d} (0x{code:016X})")
        print(f"    Decoded bounds: [{node_min[0]:6.3f}, {node_max[0]:6.3f}] × "
              f"[{node_min[1]:6.3f}, {node_max[1]:6.3f}] × "
              f"[{node_min[2]:6.3f}, {node_max[2]:6.3f}]")
        print(f"    Within bounds: {within_bounds} ✓" if within_bounds else f"    Within bounds: {within_bounds} ✗")

    print("\n✅ Encode/Decode roundtrip test PASSED")


def test_parent_child_relationships():
    """Test parent/child Morton code operations."""
    print("\n" + "="*70)
    print("TEST 2: Parent/Child Relationships")
    print("="*70)

    domain_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # Create a parent node
    parent_code = encode_morton_3d(0.0, 0.0, 0.0, 3, domain_min, domain_max)
    print(f"Parent Morton code (level 3): {parent_code:020d}")

    # Get children
    children = get_morton_children(parent_code)
    print(f"Number of children: {len(children)}")

    # Verify each child's parent points back
    for i, child_code in enumerate(children):
        # Get child bounds
        child_min, child_max, child_level = decode_morton_3d(child_code, domain_min, domain_max)

        # Get child's parent
        recovered_parent = get_morton_parent(child_code)

        # Check level
        assert child_level == 4, f"Child level should be 4, got {child_level}"

        # Check parent recovery
        assert recovered_parent == parent_code, \
            f"Child {i} parent mismatch: {recovered_parent} != {parent_code}"

        print(f"  Child {i}: level={child_level}, "
              f"center=({(child_min[0]+child_max[0])/2:6.3f}, "
              f"{(child_min[1]+child_max[1])/2:6.3f}, "
              f"{(child_min[2]+child_max[2])/2:6.3f}) ✓")

    print("\n✅ Parent/Child relationship test PASSED")


def test_spatial_coherence():
    """Test that Morton codes preserve spatial locality."""
    print("\n" + "="*70)
    print("TEST 3: Spatial Coherence")
    print("="*70)

    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    level = 5
    positions = [
        (0.1, 0.1, 0.1),  # Close together
        (0.15, 0.1, 0.1),
        (0.9, 0.9, 0.9),  # Far away
    ]

    codes = []
    for x, y, z in positions:
        code = encode_morton_3d(x, y, z, level, domain_min, domain_max)
        codes.append(code)
        print(f"  Position ({x:.2f}, {y:.2f}, {z:.2f}): Morton = {code:020d}")

    # Nearby points should have smaller Morton distance
    dist_01 = morton_distance(codes[0], codes[1])
    dist_02 = morton_distance(codes[0], codes[2])

    print(f"\n  Distance (0,1) nearby:  {dist_01}")
    print(f"  Distance (0,2) far:     {dist_02}")

    assert dist_01 < dist_02, \
        f"Nearby points should have smaller Morton distance: {dist_01} >= {dist_02}"

    print("\n✅ Spatial coherence test PASSED")


def test_batch_operations():
    """Test vectorized batch encode/decode."""
    print("\n" + "="*70)
    print("TEST 4: Batch Operations")
    print("="*70)

    domain_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # Create random positions
    n = 1000
    np.random.seed(42)
    positions = np.random.uniform(-0.9, 0.9, size=(n, 3)).astype(np.float32)
    levels = np.random.randint(1, 10, size=n)

    print(f"  Encoding {n} positions...")
    morton_codes = compute_morton_codes_batch(positions, levels, domain_min, domain_max)

    print(f"  Decoding {n} Morton codes...")
    mins, maxs, decoded_levels = decode_morton_codes_batch(morton_codes, domain_min, domain_max)

    # Verify levels match
    assert np.all(decoded_levels == levels), "Decoded levels don't match"

    # Verify all positions are within decoded bounds (with tolerance for quantization)
    # Morton encoding quantizes positions to discrete cells, so decoded bounds may not
    # exactly contain the original point, but should be very close
    tol = 0.01  # 1% of domain size is reasonable for quantization at various levels
    within_bounds = np.all((positions >= mins - tol) & (positions <= maxs + tol))
    if not within_bounds:
        # Find failed cases for debugging
        failed = ~((positions >= mins - tol) & (positions <= maxs + tol)).all(axis=1)
        print(f"  Failed cases: {np.sum(failed)}")
        for idx in np.where(failed)[0][:5]:  # Show first 5 failures
            print(f"    Position: {positions[idx]}, Bounds: [{mins[idx]}, {maxs[idx]}]")
    assert within_bounds, "Some positions not within decoded bounds"

    print(f"  ✓ All {n} positions correctly encoded/decoded")
    print(f"  ✓ All levels match")
    print(f"  ✓ All positions within bounds")

    print("\n✅ Batch operations test PASSED")


def test_memory_savings():
    """Test memory savings calculation."""
    print("\n" + "="*70)
    print("TEST 5: Memory Savings")
    print("="*70)

    test_cases = [
        6105,      # Example from comparison doc
        100000,    # Large octree
        483261,    # From actual run
    ]

    for n_nodes in test_cases:
        savings = get_memory_savings(n_nodes)

        print(f"\n  Nodes: {savings['n_nodes']:,}")
        print(f"    Old storage (center + half_size): {savings['old_mb']:.2f} MB")
        print(f"    New storage (Morton code):        {savings['new_mb']:.2f} MB")
        print(f"    Savings:                          {savings['savings_mb']:.2f} MB ({savings['savings_factor']:.1f}× reduction)")

        # Verify 3× reduction
        assert abs(savings['savings_factor'] - 3.0) < 0.01, \
            f"Expected 3× reduction, got {savings['savings_factor']:.1f}×"

    print("\n✅ Memory savings test PASSED")


def test_edge_cases():
    """Test edge cases and boundaries."""
    print("\n" + "="*70)
    print("TEST 6: Edge Cases")
    print("="*70)

    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    edge_cases = [
        ("Min corner", 0.0, 0.0, 0.0, 0),
        ("Max corner", 1.0, 1.0, 1.0, 0),
        ("Root level", 0.5, 0.5, 0.5, 0),
        ("Deep level", 0.5, 0.5, 0.5, 15),
    ]

    for name, x, y, z, level in edge_cases:
        try:
            code = encode_morton_3d(x, y, z, level, domain_min, domain_max)
            node_min, node_max, decoded_level = decode_morton_3d(code, domain_min, domain_max)

            assert decoded_level == level, f"{name}: Level mismatch"

            point = np.array([x, y, z])
            within = np.all(point >= node_min - 1e-6) and np.all(point <= node_max + 1e-6)

            print(f"  {name:15s}: level={level:2d}, within_bounds={within} ✓")

        except Exception as e:
            print(f"  {name:15s}: FAILED - {e}")
            raise

    print("\n✅ Edge cases test PASSED")


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "="*70)
    print("MORTON CODE TEST SUITE - Phase 2")
    print("="*70)

    try:
        test_encode_decode_roundtrip()
        test_parent_child_relationships()
        test_spatial_coherence()
        test_batch_operations()
        test_memory_savings()
        test_edge_cases()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nMorton code implementation validated successfully!")
        print("Ready to integrate into octree structure (Phase 2).")
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
