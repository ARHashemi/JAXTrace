"""
Level 2b: Heavy Block Hash Bucket Search - Phase 4, Task 4.6

Hash bucket search for heavy blocks (>10K elements).
This is the KEY INNOVATION: reduces O(900K) to O(200).

Performance: < 100 μs per particle for 900K element blocks
"""

import jax
import jax.numpy as jnp

from .level0_cached import point_in_tet_jax
from .hash_bucket import compute_morton_code_single_jax

jax.config.update("jax_enable_x64", True)


@jax.jit
def search_bucket_elements(
    position: jax.Array,
    bucket_elements: jax.Array,
    bucket_elem_count: int,
    node_positions: jax.Array,
    connectivity: jax.Array
) -> int:
    """
    Search elements within a single bucket.

    Parameters
    ----------
    position : jax.Array
        Particle position (3,)
    bucket_elements : jax.Array
        Element IDs in bucket (max_elem_per_bucket,)
    bucket_elem_count : int
        Actual element count in bucket
    node_positions : jax.Array
        All node positions (N_nodes, 3)
    connectivity : jax.Array
        Element connectivity (N_elements, 4)

    Returns
    -------
    element_id : int
        Found element ID, else -1
    """
    # Create mask for valid elements (not -1 padding)
    valid_mask = bucket_elements >= 0

    # Helper function to check if point is in a single element
    def check_element(elem_id):
        node_ids = connectivity[elem_id]
        tet_nodes = node_positions[node_ids]
        return point_in_tet_jax(position, tet_nodes)

    # Vectorized check over all elements in bucket
    safe_elements = jnp.where(valid_mask, bucket_elements, 0)
    inside_flags = jax.vmap(check_element)(safe_elements)

    # Mask out invalid elements
    inside_flags = inside_flags & valid_mask

    # Find first matching element, or return -1
    found_indices = jnp.where(inside_flags, jnp.arange(len(bucket_elements)), len(bucket_elements))
    first_match_idx = jnp.min(found_indices)

    return jnp.where(first_match_idx < len(bucket_elements), bucket_elements[first_match_idx], -1)


@jax.jit
def search_level2b_hash_bucket(
    position: jax.Array,
    block_id: int,
    hash_bucket_elements: jax.Array,
    hash_bucket_counts: jax.Array,
    hash_bucket_neighbors: jax.Array,
    hash_n_buckets: int,
    hash_morton_bits: int,
    hash_block_bounds: jax.Array,
    node_positions: jax.Array,
    connectivity: jax.Array
) -> int:
    """
    L2b: Hash bucket search in heavy block.

    Algorithm:
        1. Compute Morton code for particle position
        2. Map to bucket_id
        3. Search elements in that bucket (~200 elements)
        4. If not found, search 6 neighbor buckets

    Parameters
    ----------
    position : jax.Array
        Particle position (3,)
    block_id : int
        Heavy block ID
    hash_bucket_elements : jax.Array
        Bucket element IDs (n_buckets, max_elem_per_bucket)
    hash_bucket_counts : jax.Array
        Element counts per bucket (n_buckets,)
    hash_bucket_neighbors : jax.Array
        6-face neighbors (n_buckets, 6), -1 for boundary
    hash_n_buckets : int
        Number of buckets
    hash_morton_bits : int
        Morton code bits
    hash_block_bounds : jax.Array
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
    Speedup: 4,500× vs direct search
    Expected hit rate: 1-5%
    """
    # Compute Morton code for position
    morton_code = compute_morton_code_single_jax(position, hash_block_bounds, hash_morton_bits)

    # Map to bucket ID
    max_morton = (1 << (3 * hash_morton_bits)) - 1
    bucket_id = jnp.int32((morton_code * hash_n_buckets) // max_morton)
    bucket_id = jnp.clip(bucket_id, 0, hash_n_buckets - 1)

    # Search primary bucket
    elem_id_primary = search_bucket_elements(
        position,
        hash_bucket_elements[bucket_id],
        hash_bucket_counts[bucket_id],
        node_positions,
        connectivity
    )

    # If found in primary, return immediately
    # Otherwise search neighbors using JAX control flow
    def search_neighbors():
        # Get valid neighbor bucket IDs (fixed to 6 neighbors)
        neighbor_ids = hash_bucket_neighbors[bucket_id]  # (6,) array
        valid_neighbors = neighbor_ids >= 0

        # Helper to search one neighbor
        def check_neighbor(neighbor_bucket_id):
            safe_id = jnp.where(neighbor_bucket_id >= 0, neighbor_bucket_id, 0)
            return search_bucket_elements(
                position,
                hash_bucket_elements[safe_id],
                hash_bucket_counts[safe_id],
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
    return jnp.where(elem_id_primary >= 0, elem_id_primary, search_neighbors())
