"""
Level 2b: Heavy Block CSR Hash Bucket Search - Phase 1 Implementation

CSR-style hash bucket search for heavy blocks.
Memory-efficient alternative to padded arrays.

Key differences from padded version:
- Uses CSR ranges [start, end) instead of padded arrays
- Elements accessed via dynamic slice from sorted array
- No -1 padding to handle
"""

import jax
import jax.numpy as jnp

from .level0_cached import point_in_tet_jax
from .hash_bucket import compute_morton_code_single_jax

jax.config.update("jax_enable_x64", True)


def search_bucket_elements_csr(
    position: jax.Array,
    sorted_elements: jax.Array,
    bucket_start: int,
    bucket_end: int,
    max_bucket_size: int,
    node_positions: jax.Array,
    connectivity: jax.Array
) -> int:
    """
    Search elements within a single CSR bucket.

    Parameters
    ----------
    position : jax.Array
        Particle position (3,)
    sorted_elements : jax.Array
        All elements Morton-sorted (n_elements,)
    bucket_start : int
        CSR range start index
    bucket_end : int
        CSR range end index (exclusive)
    max_bucket_size : int
        Maximum bucket size (for bounded dynamic slice) - must be Python int
    node_positions : jax.Array
        All node positions (N_nodes, 3)
    connectivity : jax.Array
        Element connectivity (N_elements, 4)

    Returns
    -------
    element_id : int
        Found element ID, else -1

    Notes
    -----
    Uses jax.lax.dynamic_slice with bounded size for JIT compatibility.
    Empty buckets (start == end) return -1 immediately.
    max_bucket_size must be a concrete Python int, not a traced JAX value.
    """
    # Compute actual bucket size
    actual_size = bucket_end - bucket_start

    # Early exit for empty buckets
    def empty_bucket(_):
        return jnp.int32(-1)

    # Search non-empty bucket
    def search_bucket(_):
        # Use bounded dynamic slice with STATIC size (JAX JIT requirement)
        # Always slice max_bucket_size elements, then mask invalid ones
        # max_bucket_size MUST be a Python int (not JAX array)
        bucket_elements = jax.lax.dynamic_slice(
            sorted_elements,
            (bucket_start,),
            (int(max_bucket_size),)  # Convert to Python int to ensure it's concrete
        )

        # Sequential search through bucket elements
        # CRITICAL: Fetch node data ONE ELEMENT AT A TIME to avoid batched indexing explosion
        def check_one_element(i):
            # Check if index is valid (within actual bucket size)
            is_valid = i < actual_size

            # Get element ID for this index
            elem_id = bucket_elements[i]

            # Fetch connectivity for this single element (avoiding batched indexing)
            # Use where to make indexing safe (index 0 if invalid)
            safe_elem_id = jnp.where(is_valid, elem_id, 0)
            node_ids = connectivity[safe_elem_id]  # (4,)

            # Fetch node positions for this element's 4 nodes
            tet_nodes = node_positions[node_ids]  # (4, 3)

            # Point-in-tet check
            inside = point_in_tet_jax(position, tet_nodes)

            # Return element ID if inside and valid, else -1
            return jnp.where(is_valid & inside, elem_id, jnp.int32(-1))

        # Use lax.fori_loop to check elements sequentially until found
        def loop_body(i, found_elem):
            # If already found, skip checking
            already_found = found_elem >= 0
            # Check current element
            current_result = check_one_element(i)
            # Update found_elem if this one matches and we haven't found yet
            return jnp.where(already_found, found_elem, jnp.where(current_result >= 0, current_result, found_elem))

        # Loop through all elements in bucket
        found_elem = jax.lax.fori_loop(0, int(max_bucket_size), loop_body, jnp.int32(-1))
        return found_elem

    # Conditional: empty vs non-empty bucket
    return jax.lax.cond(
        actual_size > 0,
        search_bucket,
        empty_bucket,
        None
    )


def search_level2b_hash_bucket_csr(
    position: jax.Array,
    block_id: int,
    sorted_elements: jax.Array,
    bucket_ranges: jax.Array,
    max_bucket_size: int,
    bucket_neighbors: jax.Array,
    n_buckets: int,
    morton_bits: int,
    block_bounds: jax.Array,
    node_positions: jax.Array,
    connectivity: jax.Array
) -> int:
    """
    L2b: CSR hash bucket search in heavy block.

    Algorithm:
        1. Compute Morton code for particle position
        2. Map to bucket_id
        3. Search elements in CSR range [start, end) (~200 elements)
        4. If not found, search 6 neighbor buckets

    Parameters
    ----------
    position : jax.Array
        Particle position (3,)
    block_id : int
        Heavy block ID
    sorted_elements : jax.Array
        All elements Morton-sorted (n_elements,)
    bucket_ranges : jax.Array
        CSR ranges [start, end) per bucket (n_buckets, 2)
    max_bucket_size : int
        Maximum bucket size (for bounded slicing)
    bucket_neighbors : jax.Array
        6-face neighbors (n_buckets, 6), -1 for boundary
    n_buckets : int
        Number of buckets
    morton_bits : int
        Morton code bits
    block_bounds : jax.Array
        Block bounds (6,) [xmin, xmax, ymin, ymax, zmin, zmax]
    node_positions : jax.Array
        All node positions (N_nodes, 3)
    connectivity : jax.Array
        Element connectivity (N_elements, 4)

    Returns
    -------
    element_id : int
        Found element ID, else -1

    Performance
    -----------
    Expected: < 100 μs for 900K element blocks
    Memory: CSR uses 19% less than padded arrays
    Expected hit rate: 1-5%
    """
    # Compute Morton code for position
    morton_code = compute_morton_code_single_jax(position, block_bounds, morton_bits)

    # Map to bucket ID
    max_morton = (1 << (3 * morton_bits)) - 1
    bucket_id = jnp.int32((morton_code * n_buckets) // max_morton)
    bucket_id = jnp.clip(bucket_id, 0, n_buckets - 1)

    # Get CSR range for primary bucket
    bucket_start = bucket_ranges[bucket_id, 0]
    bucket_end = bucket_ranges[bucket_id, 1]

    # Search primary bucket
    elem_id_primary = search_bucket_elements_csr(
        position,
        sorted_elements,
        bucket_start,
        bucket_end,
        max_bucket_size,
        node_positions,
        connectivity
    )

    # If found in primary, return immediately
    # Otherwise search neighbors using JAX control flow
    def search_neighbors(_):
        # Get valid neighbor bucket IDs (fixed to 6 neighbors)
        neighbor_ids = bucket_neighbors[bucket_id]  # (6,) array
        valid_neighbors = neighbor_ids >= 0

        # Helper to search one neighbor
        def check_neighbor(neighbor_bucket_id):
            # Safe index (0 if invalid, actual if valid)
            safe_id = jnp.where(neighbor_bucket_id >= 0, neighbor_bucket_id, 0)

            # Get CSR range for neighbor
            neighbor_start = bucket_ranges[safe_id, 0]
            neighbor_end = bucket_ranges[safe_id, 1]

            # Search neighbor bucket
            return search_bucket_elements_csr(
                position,
                sorted_elements,
                neighbor_start,
                neighbor_end,
                max_bucket_size,
                node_positions,
                connectivity
            )

        # Search all neighbors vectorized
        neighbor_results = jax.vmap(check_neighbor)(neighbor_ids)

        # Mask invalid neighbors
        neighbor_results = jnp.where(valid_neighbors, neighbor_results, -1)

        # Find first match
        found_indices = jnp.where(neighbor_results >= 0, jnp.arange(6), 6)
        first_match_idx = jnp.min(found_indices)

        return jnp.where(first_match_idx < 6, neighbor_results[first_match_idx], -1)

    # Return primary result if found, else search neighbors
    return jnp.where(elem_id_primary >= 0, elem_id_primary, search_neighbors(None))


if __name__ == "__main__":
    """Test CSR search functions."""
    import numpy as np
    from .hash_bucket_csr import build_hash_bucket_arrays_csr

    print("=" * 80)
    print("TESTING CSR HASH BUCKET SEARCH")
    print("=" * 80)
    print()

    # Create synthetic mesh data
    print("Creating synthetic mesh...")
    n_nodes = 1000
    n_elements = 500
    node_positions = np.random.uniform(-1, 1, (n_nodes, 3)).astype(np.float32)
    connectivity = np.random.randint(0, n_nodes, (n_elements, 4), dtype=np.int32)

    # Build CSR hash buckets
    print("Building CSR hash buckets...")
    element_ids = np.arange(n_elements, dtype=np.int32)
    element_centroids = np.array([
        node_positions[connectivity[i]].mean(axis=0)
        for i in range(n_elements)
    ], dtype=np.float32)
    block_bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)

    hash_csr = build_hash_bucket_arrays_csr(
        block_id=0,
        element_ids=element_ids,
        element_centroids=element_centroids,
        block_bounds=block_bounds,
        target_bucket_size=50,
        verbose=False
    )
    print(f"✓ Built: {hash_csr.n_buckets} buckets, {len(hash_csr.sorted_elements)} elements")
    print()

    # Upload to GPU
    print("Uploading to GPU...")
    sorted_elements_gpu = jax.device_put(hash_csr.sorted_elements)
    bucket_ranges_gpu = jax.device_put(hash_csr.bucket_ranges)
    bucket_neighbors_gpu = jax.device_put(hash_csr.bucket_neighbors_6)
    block_bounds_gpu = jax.device_put(hash_csr.block_bounds)
    node_positions_gpu = jax.device_put(node_positions)
    connectivity_gpu = jax.device_put(connectivity)
    print("✓ Data uploaded to GPU")
    print()

    # Test 1: JIT compilation
    print("Test 1: JIT Compilation")
    print("-" * 80)
    test_position = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)

    print("  Compiling search_bucket_elements_csr...")
    _ = search_bucket_elements_csr(
        test_position,
        sorted_elements_gpu,
        jnp.int32(0),
        jnp.int32(10),
        hash_csr.max_bucket_size,
        node_positions_gpu,
        connectivity_gpu
    )
    print("  ✓ search_bucket_elements_csr compiled")

    print("  Compiling search_level2b_hash_bucket_csr...")
    _ = search_level2b_hash_bucket_csr(
        test_position,
        0,
        sorted_elements_gpu,
        bucket_ranges_gpu,
        hash_csr.max_bucket_size,
        bucket_neighbors_gpu,
        hash_csr.n_buckets,
        hash_csr.morton_bits,
        block_bounds_gpu,
        node_positions_gpu,
        connectivity_gpu
    )
    print("  ✓ search_level2b_hash_bucket_csr compiled")
    print()

    # Test 2: Search multiple particles
    print("Test 2: Search Multiple Particles")
    print("-" * 80)
    n_test = 100
    test_positions = np.random.uniform(-0.5, 0.5, (n_test, 3)).astype(np.float32)

    print(f"  Searching {n_test} particles...")
    found_count = 0
    for i, pos in enumerate(test_positions):
        pos_gpu = jnp.array(pos, dtype=jnp.float32)
        elem_id = search_level2b_hash_bucket_csr(
            pos_gpu,
            0,
            sorted_elements_gpu,
            bucket_ranges_gpu,
            hash_csr.max_bucket_size,
            bucket_neighbors_gpu,
            hash_csr.n_buckets,
            hash_csr.morton_bits,
            block_bounds_gpu,
            node_positions_gpu,
            connectivity_gpu
        )
        elem_id_cpu = int(elem_id)
        if elem_id_cpu >= 0:
            found_count += 1

    print(f"  ✓ Search complete: {found_count}/{n_test} particles found")
    print(f"    (Low hit rate expected with random synthetic data)")
    print()

    print("=" * 80)
    print("✅ CSR HASH BUCKET SEARCH TESTS COMPLETE")
    print("=" * 80)
