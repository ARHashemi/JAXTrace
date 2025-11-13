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


# @jax.jit
# def search_level1_neighbors(
#     position: jax.Array,
#     cached_element_id: int,
#     element_neighbors: jax.Array,
#     node_positions: jax.Array,
#     connectivity: jax.Array
# ) -> int:
#     """
#     L1: Check face-adjacent neighbor elements.

#     Uses Phase 2 neighbor arrays to check 3-4 adjacent elements.
#     Expected to catch particles that crossed an element face.

#     Parameters
#     ----------
#     position : jax.Array
#         Particle position (3,)
#     cached_element_id : int
#         Last known element ID
#     element_neighbors : jax.Array
#         Neighbor IDs for cached element (max_neighbors,)
#         From Phase 2, -1 indicates no neighbor
#     node_positions : jax.Array
#         All node positions (N_nodes, 3)
#     connectivity : jax.Array
#         Element connectivity (N_elements, 4)

#     Returns
#     -------
#     element_id : int
#         Neighbor element ID if found, else -1

#     Performance
#     -----------
#     Expected: < 5 μs per particle
#     Expected hit rate: 3-10%
#     """
#     # Check each neighbor
#     for i in range(len(element_neighbors)):
#         neighbor_id = element_neighbors[i]

#         # Skip invalid neighbors
#         if neighbor_id < 0:
#             continue

#         # Get tet nodes
#         node_ids = connectivity[neighbor_id]
#         tet_nodes = node_positions[node_ids]

#         # Test if inside
#         inside = point_in_tet_jax(position, tet_nodes)

#         if inside:
#             return neighbor_id

#     return -1

@jax.jit
def search_level1_neighbors(
    position: jax.Array,
    cached_element_id: int,
    element_neighbors: jax.Array,    # (max_neighbors,)
    node_positions: jax.Array,
    connectivity: jax.Array,
    tolerance: float = 1e-10
) -> int:
    """
    L1: Check face-adjacent neighbor elements.
    Uses vectorized pattern for JAX.
    Returns first matching neighbor element_id, or -1 if not found.
    """

    # Define helper for one neighbor
    def check_neighbor(neighbor_id):
        valid = neighbor_id >= 0
        # Ensure valid index even if not used
        safe_id = jnp.where(valid, neighbor_id, 0)
        node_ids = connectivity[safe_id]
        tet_nodes = node_positions[node_ids]
        inside = point_in_tet_jax(position, tet_nodes, tolerance)
        # Only true if valid and inside
        return jnp.where(valid & inside, safe_id, -1)

    # Vectorize over max_neighbors axis
    neighbor_ids = element_neighbors  # shape: (max_neighbors,)
    found_ids = jax.vmap(check_neighbor)(neighbor_ids)  # (max_neighbors,)

    # Find first match (index)
    found_indices = jnp.where(found_ids >= 0, jnp.arange(found_ids.shape[0]), found_ids.shape[0])
    first_idx = jnp.min(found_indices)

    return jnp.where(first_idx < found_ids.shape[0], found_ids[first_idx], -1)