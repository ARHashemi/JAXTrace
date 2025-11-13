"""
GPU Kernels V2 - Flat Array Design for JAX Compatibility.

This module implements particle tracking using flat, fixed-size arrays that are
compatible with JAX's vmap and GPU execution model.

Design principles:
1. All arrays are flat with fixed dimensions
2. Use indexing instead of dynamic lists
3. Pad with -1 for missing neighbors/children
4. Enable full vectorization with vmap
"""

import jax
import jax.numpy as jnp
from typing import Tuple


@jax.jit
def point_in_tetrahedron_batch(
    points: jnp.ndarray,
    vertices: jnp.ndarray
) -> jnp.ndarray:
    """
    Test if points are inside tetrahedra using barycentric coordinates.

    Vectorized version that processes multiple point-tet pairs in parallel.

    Args:
        points: Points to test [N, 3]
        vertices: Tetrahedron vertices [N, 4, 3]

    Returns:
        inside: Boolean array [N] indicating if point is inside tetrahedron

    Algorithm:
        For each point-tet pair:
        1. Set up linear system: A @ lambda = (p - v0)
           where A = [v1-v0, v2-v0, v3-v0]
        2. Solve for barycentric coordinates λ₁, λ₂, λ₃
        3. Compute λ₀ = 1 - λ₁ - λ₂ - λ₃
        4. Check: all λᵢ ≥ -ε (with tolerance)
    """
    # Extract vertices
    v0 = vertices[:, 0, :]  # [N, 3]
    v1 = vertices[:, 1, :]
    v2 = vertices[:, 2, :]
    v3 = vertices[:, 3, :]

    # Build matrices A = [v1-v0, v2-v0, v3-v0] for each tetrahedron
    # A shape: [N, 3, 3]
    A = jnp.stack([v1 - v0, v2 - v0, v3 - v0], axis=-1)

    # RHS: point - v0
    b = points - v0  # [N, 3]

    # Solve A @ lambda = b
    # Check condition number first for numerical stability
    cond = jnp.linalg.cond(A)  # [N]
    is_well_conditioned = cond < 1e6

    # Solve (returns garbage for ill-conditioned, but we mask it)
    lambdas = jnp.linalg.solve(A, b)  # [N, 3]

    # Compute lambda0
    lambda0 = 1.0 - jnp.sum(lambdas, axis=1)  # [N]

    # Check if all barycentric coordinates are non-negative (with tolerance)
    epsilon = 1e-6
    lambda_valid = jnp.all(lambdas >= -epsilon, axis=1) & (lambda0 >= -epsilon)
    lambda_sum = jnp.sum(lambdas, axis=1)
    sum_valid = lambda_sum <= 1.0 + epsilon

    # Combine conditions
    inside = lambda_valid & sum_valid & is_well_conditioned

    return inside


@jax.jit
def search_level0_cached(
    particle_positions: jnp.ndarray,
    particle_element_ids: jnp.ndarray,
    element_nodes: jnp.ndarray,
    node_positions: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Level 0: Check if particles are still in their cached elements.

    Args:
        particle_positions: Particle positions [N_particles, 3]
        particle_element_ids: Cached element IDs [N_particles]
        element_nodes: Element node IDs [N_elements, 4]
        node_positions: Node positions [N_nodes, 3]

    Returns:
        found: Boolean array [N_particles] indicating if still in cached element
        result_ids: Element IDs [N_particles] (same as input if found, -1 if not)

    Note:
        This is fully vectorized - checks all particles in parallel.
    """
    n_particles = particle_positions.shape[0]

    # Handle invalid cached IDs
    valid_cache = particle_element_ids >= 0

    # Safe indexing: use 0 for invalid IDs (result will be masked anyway)
    safe_ids = jnp.where(valid_cache, particle_element_ids, 0)

    # Get element nodes for all particles [N_particles, 4]
    elem_node_ids = element_nodes[safe_ids]

    # Get vertices [N_particles, 4, 3]
    vertices = node_positions[elem_node_ids]

    # Check if points are inside
    inside = point_in_tetrahedron_batch(particle_positions, vertices)

    # Mask out invalid cache
    found = inside & valid_cache

    # Return original ID if found, -1 if not
    result_ids = jnp.where(found, particle_element_ids, -1)

    return found, result_ids


@jax.jit
def search_level1_neighbors(
    particle_positions: jnp.ndarray,
    particle_element_ids: jnp.ndarray,
    level0_found: jnp.ndarray,
    element_neighbors: jnp.ndarray,
    element_nodes: jnp.ndarray,
    node_positions: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Level 1: Check neighbor elements for particles not found in Level 0.

    Args:
        particle_positions: Particle positions [N_particles, 3]
        particle_element_ids: Cached element IDs [N_particles]
        level0_found: Boolean mask from Level 0 [N_particles]
        element_neighbors: Neighbor element IDs [N_elements, max_neighbors]
        element_nodes: Element node IDs [N_elements, 4]
        node_positions: Node positions [N_nodes, 3]

    Returns:
        found: Boolean array [N_particles] indicating if found in neighbors
        result_ids: Element IDs [N_particles] (neighbor ID if found, -1 if not)

    Algorithm:
        For each particle not found in Level 0:
        1. Get its cached element's neighbors
        2. Check each neighbor (up to max_neighbors)
        3. Return first match

    Note:
        Uses vmap to check all neighbors in parallel for each particle.
    """
    n_particles = particle_positions.shape[0]
    max_neighbors = element_neighbors.shape[1]

    # Only search for particles not found in Level 0
    needs_search = ~level0_found & (particle_element_ids >= 0)

    # Safe indexing
    safe_ids = jnp.where(particle_element_ids >= 0, particle_element_ids, 0)

    # Get neighbors for all particles [N_particles, max_neighbors]
    neighbor_ids = element_neighbors[safe_ids]

    def check_single_particle_neighbors(pos, neighbors, search_needed):
        """Check all neighbors for a single particle."""

        def check_single_neighbor(neighbor_id):
            """Check if point is in this neighbor element."""
            is_valid = neighbor_id >= 0
            safe_id = jnp.where(is_valid, neighbor_id, 0)

            # Get vertices
            elem_node_ids = element_nodes[safe_id]
            vertices = node_positions[elem_node_ids]

            # Check containment
            inside = point_in_tetrahedron_batch(
                pos.reshape(1, 3),
                vertices.reshape(1, 4, 3)
            )[0]

            return is_valid & inside, jnp.where(is_valid & inside, neighbor_id, -1)

        # Check all neighbors using vmap
        found_array, id_array = jax.vmap(check_single_neighbor)(neighbors)

        # Find first match
        found_any = jnp.any(found_array)
        first_match_idx = jnp.argmax(found_array)  # Returns 0 if all False
        result_id = jnp.where(found_any, id_array[first_match_idx], -1)

        # Only return result if search was needed
        final_found = found_any & search_needed
        final_id = jnp.where(search_needed, result_id, -1)

        return final_found, final_id

    # Vectorize over all particles
    found, result_ids = jax.vmap(check_single_particle_neighbors)(
        particle_positions,
        neighbor_ids,
        needs_search
    )

    return found, result_ids


@jax.jit
def search_level2_block_elements(
    particle_positions: jnp.ndarray,
    particle_element_ids: jnp.ndarray,
    level0_found: jnp.ndarray,
    level1_found: jnp.ndarray,
    element_block_ids: jnp.ndarray,
    element_nodes: jnp.ndarray,
    node_positions: jnp.ndarray,
    max_elements_to_check: int = 1000
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Level 2: Search all elements in the same block as cached element.

    Args:
        particle_positions: Particle positions [N_particles, 3]
        particle_element_ids: Cached element IDs [N_particles]
        level0_found: Boolean mask from Level 0 [N_particles]
        level1_found: Boolean mask from Level 1 [N_particles]
        element_block_ids: Block ID for each element [N_elements]
        element_nodes: Element node IDs [N_elements, 4]
        node_positions: Node positions [N_nodes, 3]
        max_elements_to_check: Maximum elements to check per particle

    Returns:
        found: Boolean array [N_particles]
        result_ids: Element IDs [N_particles]

    Algorithm:
        For each particle not found in Level 0 or 1:
        1. Get block ID from cached element: block_id = element_block_ids[cached_elem]
        2. Check all elements where element_block_ids == block_id
        3. Return first match

    Note:
        This uses the flat array design - instead of pre-computing block element lists,
        we check ALL elements but only process those in the target block.

        This is JAX-compatible because:
        - All arrays are fixed size
        - We use masking (element_block_ids == block_id) instead of dynamic lists
        - vmap handles parallelization
    """
    n_particles = particle_positions.shape[0]
    n_elements = element_block_ids.shape[0]

    # Only search for particles not found in Level 0 or 1
    needs_search = ~level0_found & ~level1_found & (particle_element_ids >= 0)

    # Get block ID for each particle (from its cached element)
    safe_cached_ids = jnp.where(particle_element_ids >= 0, particle_element_ids, 0)
    particle_block_ids = element_block_ids[safe_cached_ids]

    def check_single_particle_block(pos, block_id, search_needed):
        """
        Check all elements in this particle's block.

        Key insight: Instead of creating a list of elements in the block,
        we check ALL elements but only count matches where element is in block.

        This is vectorizable because we're checking a fixed set (all elements).
        """
        if not search_needed:
            return False, -1

        # Create mask: which elements are in this block?
        # This is the KEY to JAX compatibility - masking instead of dynamic lists!
        in_block = element_block_ids == block_id  # [N_elements] boolean

        # Limit search scope for memory (check at most max_elements_to_check)
        # Use jnp.where to get indices, padded to fixed size
        element_indices = jnp.arange(n_elements)

        def check_single_element(elem_idx):
            """Check if this element contains the point."""
            # Only check if element is in block
            should_check = in_block[elem_idx]

            if not should_check:
                return False, -1

            # Get vertices
            elem_node_ids = element_nodes[elem_idx]
            vertices = node_positions[elem_node_ids]

            # Check containment
            inside = point_in_tetrahedron_batch(
                pos.reshape(1, 3),
                vertices.reshape(1, 4, 3)
            )[0]

            return inside, jnp.where(inside, elem_idx, -1)

        # Check all elements using vmap (but most will be masked out)
        # Limit to max_elements_to_check for memory
        elements_to_check = element_indices[:max_elements_to_check]
        found_array, id_array = jax.vmap(check_single_element)(elements_to_check)

        # Find first match
        found_any = jnp.any(found_array)
        first_match_idx = jnp.argmax(found_array)
        result_id = jnp.where(found_any, id_array[first_match_idx], -1)

        return found_any, result_id

    # Vectorize over all particles
    found, result_ids = jax.vmap(check_single_particle_block)(
        particle_positions,
        particle_block_ids,
        needs_search
    )

    return found, result_ids


@jax.jit
def multi_level_search(
    particle_positions: jnp.ndarray,
    particle_element_ids: jnp.ndarray,
    element_nodes: jnp.ndarray,
    element_neighbors: jnp.ndarray,
    element_block_ids: jnp.ndarray,
    node_positions: jnp.ndarray
) -> jnp.ndarray:
    """
    Complete multi-level element search for all particles.

    Args:
        particle_positions: Particle positions [N_particles, 3]
        particle_element_ids: Cached element IDs [N_particles]
        element_nodes: Element node IDs [N_elements, 4]
        element_neighbors: Neighbor element IDs [N_elements, max_neighbors]
        element_block_ids: Block ID for each element [N_elements]
        node_positions: Node positions [N_nodes, 3]

    Returns:
        new_element_ids: Updated element IDs [N_particles]

    Algorithm:
        For each particle:
        1. Level 0: Check cached element (85-95% hit rate)
        2. Level 1: Check neighbor elements (3-10% hit rate)
        3. Level 2: Check all elements in same block (1-5% hit rate)
        4. Return -1 if not found (particle left domain)

    Note:
        This is fully vectorized - all levels run in parallel for all particles.
        The masking ensures we only update particles not found in earlier levels.
    """
    # Level 0: Check cached elements
    level0_found, level0_ids = search_level0_cached(
        particle_positions,
        particle_element_ids,
        element_nodes,
        node_positions
    )

    # Level 1: Check neighbors (only for particles not found in Level 0)
    level1_found, level1_ids = search_level1_neighbors(
        particle_positions,
        particle_element_ids,
        level0_found,
        element_neighbors,
        element_nodes,
        node_positions
    )

    # Level 2: Search block elements (only for particles not found in Level 0 or 1)
    level2_found, level2_ids = search_level2_block_elements(
        particle_positions,
        particle_element_ids,
        level0_found,
        level1_found,
        element_block_ids,
        element_nodes,
        node_positions
    )

    # Combine results: use first level that found the particle
    new_element_ids = jnp.where(
        level0_found,
        level0_ids,
        jnp.where(
            level1_found,
            level1_ids,
            jnp.where(
                level2_found,
                level2_ids,
                -1  # Not found
            )
        )
    )

    return new_element_ids


# Export key functions
__all__ = [
    'point_in_tetrahedron_batch',
    'search_level0_cached',
    'search_level1_neighbors',
    'search_level2_block_elements',
    'multi_level_search',
]
