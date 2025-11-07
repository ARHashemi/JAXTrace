"""
Level 1: Neighbor Element Search - Phase 4, Task 4.4

Checks 3-4 face-adjacent neighbor elements.
Expected 3-10% hit rate for particles crossing element boundaries.

Performance: < 5 μs per particle
"""

import jax
import jax.numpy as jnp

from .level0_cached import point_in_tet_jax

jax.config.update("jax_enable_x64", True)


@jax.jit
def search_level1_neighbors(
    position: jax.Array,
    cached_element_id: int,
    element_neighbors: jax.Array,
    node_positions: jax.Array,
    connectivity: jax.Array
) -> int:
    """
    L1: Check face-adjacent neighbor elements.

    Uses Phase 2 neighbor arrays to check 3-4 adjacent elements.
    Expected to catch particles that crossed an element face.

    Parameters
    ----------
    position : jax.Array
        Particle position (3,)
    cached_element_id : int
        Last known element ID
    element_neighbors : jax.Array
        Neighbor IDs for cached element (max_neighbors,)
        From Phase 2, -1 indicates no neighbor
    node_positions : jax.Array
        All node positions (N_nodes, 3)
    connectivity : jax.Array
        Element connectivity (N_elements, 4)

    Returns
    -------
    element_id : int
        Neighbor element ID if found, else -1

    Performance
    -----------
    Expected: < 5 μs per particle
    Expected hit rate: 3-10%
    """
    # Check each neighbor
    for i in range(len(element_neighbors)):
        neighbor_id = element_neighbors[i]

        # Skip invalid neighbors
        if neighbor_id < 0:
            continue

        # Get tet nodes
        node_ids = connectivity[neighbor_id]
        tet_nodes = node_positions[node_ids]

        # Test if inside
        inside = point_in_tet_jax(position, tet_nodes)

        if inside:
            return neighbor_id

    return -1
