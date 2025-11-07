"""
Level 0: Cached Element Search - Phase 4, Task 4.3

Checks if particle is still in its last known (cached) element.
This is the fastest search level with expected 85-95% hit rate for small time steps.

Performance: < 1 μs per particle
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@jax.jit
def point_in_tet_jax(
    point: jax.Array,
    tet_nodes: jax.Array,
    tolerance: float = 1e-10
) -> bool:
    """
    Test if point is inside tetrahedron using barycentric coordinates.

    Parameters
    ----------
    point : jax.Array
        Point position (3,)
    tet_nodes : jax.Array
        Tetrahedron node positions (4, 3)
    tolerance : float
        Numerical tolerance for boundary cases

    Returns
    -------
    inside : bool
        True if point is inside tetrahedron
    """
    v0, v1, v2, v3 = tet_nodes[0], tet_nodes[1], tet_nodes[2], tet_nodes[3]

    # Build matrix for barycentric coordinates
    mat = jnp.column_stack([v1 - v0, v2 - v0, v3 - v0])

    # Solve for barycentric coordinates
    det = jnp.linalg.det(mat)

    # Handle degenerate case
    is_degenerate = jnp.abs(det) < tolerance

    # Compute barycentric coordinates
    rhs = point - v0
    lambdas_123 = jnp.linalg.solve(mat, rhs)
    lambda_0 = 1.0 - jnp.sum(lambdas_123)

    all_lambdas = jnp.concatenate([jnp.array([lambda_0]), lambdas_123])

    # Check if all in [0, 1] with tolerance
    inside = jnp.all(all_lambdas >= -tolerance) & jnp.all(all_lambdas <= 1.0 + tolerance)

    # Return false for degenerate tets
    return jnp.where(is_degenerate, False, inside)


@jax.jit
def search_level0_cached(
    position: jax.Array,
    cached_element_id: int,
    node_positions: jax.Array,
    connectivity: jax.Array
) -> int:
    """
    L0: Check if particle still in cached element.

    This is the fastest search level and should have 85-95% hit rate
    for small time steps (particles don't move far).

    Parameters
    ----------
    position : jax.Array
        Particle position (3,)
    cached_element_id : int
        Last known element ID for this particle
    node_positions : jax.Array
        All node positions (N_nodes, 3)
    connectivity : jax.Array
        Element connectivity (N_elements, 4)

    Returns
    -------
    element_id : int
        cached_element_id if still inside, else -1

    Performance
    -----------
    Expected: < 1 μs per particle
    Expected hit rate: 85-95%
    """
    # Check if cached element is valid
    is_valid = (cached_element_id >= 0) & (cached_element_id < len(connectivity))

    # Get tet nodes (use jnp.where to handle invalid indices safely)
    safe_idx = jnp.where(is_valid, cached_element_id, 0)
    node_ids = connectivity[safe_idx]
    tet_nodes = node_positions[node_ids]

    # Test if still inside
    inside = point_in_tet_jax(position, tet_nodes)

    # Return cached_element_id only if valid AND inside
    return jnp.where(is_valid & inside, cached_element_id, -1)
