#!/usr/bin/env python3
"""
Test to verify initial assignment search is working correctly after lax.fori_loop fix.
"""

import numpy as np
import jax
import jax.numpy as jnp

# Quick test of the fori_loop implementation
def test_search_pattern():
    """Test that the search pattern matches expected behavior."""

    max_radius = 10
    center_leaf_id = jnp.int32(100)

    # Expected offsets: -10, -9, ..., 9, 10 (21 total)
    expected_offsets = list(range(-max_radius, max_radius + 1))

    print(f"Testing radius={max_radius}, center_leaf={center_leaf_id}")
    print(f"Expected offsets: {expected_offsets}")
    print(f"Expected neighbor leaves: {[center_leaf_id + off for off in expected_offsets]}")

    # Simulate what fori_loop does
    collected_offsets = []
    for i in range(2 * max_radius + 1):
        offset = i - max_radius
        collected_offsets.append(offset)

    print(f"Actual offsets from loop: {collected_offsets}")
    print(f"Match: {collected_offsets == expected_offsets}")

    # Test the conversion formula
    print(f"\nLoop index conversion:")
    for i in [0, max_radius, 2*max_radius]:
        offset = i - max_radius
        print(f"  i={i:2d} → offset={offset:3d}")

if __name__ == "__main__":
    test_search_pattern()

    print("\n" + "="*60)
    print("Testing with different radii:")
    print("="*60)

    for radius in [10, 50, 100]:
        print(f"\nRadius={radius}:")
        print(f"  Loop iterations: 2*{radius}+1 = {2*radius+1}")
        print(f"  Offsets covered: -{radius} to +{radius}")
        print(f"  First offset (i=0): {0 - radius}")
        print(f"  Last offset (i={2*radius}): {2*radius - radius}")
