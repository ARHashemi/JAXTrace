"""
Level 2a: Light Block Direct Search - Phase 4, Task 4.5

Direct search in light blocks (<10K elements) using Phase 2 padded arrays.

Performance: < 10 μs per particle for 1K-10K element blocks
"""

import jax
import jax.numpy as jnp

from .level0_cached import point_in_tet_jax

jax.config.update("jax_enable_x64", True)


@jax.jit
def search_level2a_light_block(
    position: jax.Array,
    block_id: int,
    block_elements: jax.Array,
    block_elem_count: int,
    node_positions: jax.Array,
    connectivity: jax.Array
) -> int:
    """
    L2a: Direct search in light block.

    For blocks with <10K elements, brute-force search is efficient.
    Uses Phase 2 padded arrays for vectorized search.

    Parameters
    ----------
    position : jax.Array
        Particle position (3,)
    block_id : int
        Block ID to search
    block_elements : jax.Array
        Padded element IDs for this block (max_elem_per_block,)
    block_elem_count : int
        Actual element count in block
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
    Expected: < 10 μs for 1K-10K elements
    Expected hit rate: 1-5%
    """
    # Create mask for valid elements (not -1 padding)
    valid_mask = block_elements >= 0

    # Helper function to check if point is in a single element
    def check_element(elem_id):
        # Get tet nodes
        node_ids = connectivity[elem_id]
        tet_nodes = node_positions[node_ids]
        # Test if inside
        return point_in_tet_jax(position, tet_nodes)

    # Vectorized check over all elements in block
    safe_elements = jnp.where(valid_mask, block_elements, 0)  # Replace -1 with 0 for indexing
    inside_flags = jax.vmap(check_element)(safe_elements)

    # Mask out invalid elements
    inside_flags = inside_flags & valid_mask

    # Find first matching element, or return -1
    found_indices = jnp.where(inside_flags, jnp.arange(len(block_elements)), len(block_elements))
    first_match_idx = jnp.min(found_indices)

    return jnp.where(first_match_idx < len(block_elements), block_elements[first_match_idx], -1)
