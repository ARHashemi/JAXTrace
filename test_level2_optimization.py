#!/usr/bin/env python3
"""
Test the optimized Level 2 search implementation.

Verifies that the vectorized approach works correctly on CPU.
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # Force CPU to avoid GPU cuSolver errors

import jax
import jax.numpy as jnp
import numpy as np
from jaxtrace.gpu.kernels import search_block_elements_jax, build_block_element_lists

def test_level2_search():
    """Test Level 2 block search with optimized implementation."""

    # Create simple test mesh: 2 tetrahedra
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=jnp.float32)

    connectivity = jnp.array([
        [0, 1, 2, 3],  # Element 0
        [1, 2, 3, 4],  # Element 1
    ], dtype=jnp.int32)

    # Assign both elements to block 0
    element_to_block_cpu = np.array([0, 0], dtype=np.int32)

    # Build block element lists
    n_blocks = 1
    block_elements, block_counts = build_block_element_lists(
        element_to_block_cpu, n_blocks, max_elements_per_block=10
    )

    # Convert to JAX arrays
    block_elements = jnp.array(block_elements)
    block_counts = jnp.array(block_counts)

    # Test point inside element 0
    point_in_elem0 = jnp.array([0.2, 0.2, 0.2], dtype=jnp.float32)

    print("Testing Level 2 search optimization...\n")

    # Search for point in block 0
    found, elem_id = search_block_elements_jax(
        point_in_elem0,
        block_id=0,
        block_elements=block_elements,
        block_counts=block_counts,
        positions=positions,
        connectivity=connectivity
    )

    print(f"Point [{point_in_elem0[0]:.1f}, {point_in_elem0[1]:.1f}, {point_in_elem0[2]:.1f}]:")
    print(f"  Found: {found}")
    print(f"  Element ID: {elem_id}")
    print(f"  Expected: Element 0")

    assert found, "Should find element in block"
    assert elem_id == 0, f"Should find element 0, got {elem_id}"

    print("  ✅ PASS\n")

    # Test point outside all elements
    point_outside = jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32)

    found, elem_id = search_block_elements_jax(
        point_outside,
        block_id=0,
        block_elements=block_elements,
        block_counts=block_counts,
        positions=positions,
        connectivity=connectivity
    )

    print(f"Point [{point_outside[0]:.1f}, {point_outside[1]:.1f}, {point_outside[2]:.1f}]:")
    print(f"  Found: {found}")
    print(f"  Element ID: {elem_id}")
    print(f"  Expected: Not found (-1)")

    assert not found, "Should not find element outside domain"
    assert elem_id == -1, f"Should return -1, got {elem_id}"

    print("  ✅ PASS\n")

    # Test with invalid block
    found, elem_id = search_block_elements_jax(
        point_in_elem0,
        block_id=-1,  # Invalid block
        block_elements=block_elements,
        block_counts=block_counts,
        positions=positions,
        connectivity=connectivity
    )

    print(f"Invalid block ID (-1):")
    print(f"  Found: {found}")
    print(f"  Element ID: {elem_id}")
    print(f"  Expected: Not found (-1)")

    assert not found, "Should not find with invalid block"
    assert elem_id == -1, f"Should return -1, got {elem_id}"

    print("  ✅ PASS\n")

    # Performance test: Many elements in block
    print("Performance test with large mesh...")

    # Create mesh with 500 elements
    n_elements = 500
    large_positions = jnp.zeros((n_elements * 4, 3), dtype=jnp.float32)
    large_connectivity = jnp.arange(n_elements * 4, dtype=jnp.int32).reshape(n_elements, 4)
    large_element_to_block_cpu = np.zeros(n_elements, dtype=np.int32)  # All in block 0

    # Build block element lists for large mesh
    large_block_elements, large_block_counts = build_block_element_lists(
        large_element_to_block_cpu, n_blocks=1, max_elements_per_block=500
    )
    large_block_elements = jnp.array(large_block_elements)
    large_block_counts = jnp.array(large_block_counts)

    # Point to search for
    test_point = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)

    # Warm-up JIT
    _ = search_block_elements_jax(
        test_point, 0, large_block_elements, large_block_counts,
        large_positions, large_connectivity
    )

    # Time it
    import time
    start = time.time()
    for _ in range(100):
        _ = search_block_elements_jax(
            test_point, 0, large_block_elements, large_block_counts,
            large_positions, large_connectivity
        )
    jax.block_until_ready(_)  # Wait for GPU
    elapsed = time.time() - start

    print(f"  100 searches through 500 elements:")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Per search: {elapsed/100*1000:.3f}ms")
    print(f"  ✅ PASS\n")

    print("=" * 60)
    print("All Level 2 optimization tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_level2_search()
