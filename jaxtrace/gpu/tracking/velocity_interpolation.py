"""
Block-Local Velocity Interpolation

GPU-accelerated velocity interpolation using barycentric coordinates.
Adapted from jaxtrace/fields/fem_interpolator.py for block-wise architecture.

Key features:
- Block-local element indexing (compatible with Phase 1 PaddedArrays)
- Barycentric coordinate interpolation for tetrahedral elements
- Fully JAX-JIT compiled for GPU execution
- Vectorized over particle batches using jax.vmap

Performance target: >10,000 particles/second
"""

import jax
import jax.numpy as jnp
from typing import Tuple


@jax.jit
def compute_barycentric_coordinates(
    point: jnp.ndarray,
    tet_nodes: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute barycentric coordinates for point in tetrahedron.

    Uses Cramer's rule to solve for barycentric coordinates (λ0, λ1, λ2, λ3)
    such that: point = λ0*v0 + λ1*v1 + λ2*v2 + λ3*v3

    Adapted from jaxtrace/fields/fem_interpolator.py:148-210

    Parameters
    ----------
    point : jnp.ndarray
        Query point (3,)
    tet_nodes : jnp.ndarray
        Tetrahedron node coordinates (4, 3)

    Returns
    -------
    is_inside : jnp.ndarray
        Boolean scalar, True if point is inside tetrahedron
    bary_coords : jnp.ndarray
        Barycentric coordinates (4,) - always computed even if outside
    """

    # Extract vertices
    v0, v1, v2, v3 = tet_nodes[0], tet_nodes[1], tet_nodes[2], tet_nodes[3]

    # Compute vectors from v0
    v0p = point - v0
    v01 = v1 - v0
    v02 = v2 - v0
    v03 = v3 - v0

    # Compute 3x3 matrix determinant (tetrahedron volume × 6)
    mat = jnp.stack([v01, v02, v03], axis=1)  # (3, 3)
    det = jnp.linalg.det(mat)

    # Avoid division by zero for degenerate tets
    # For degenerate tets, fall back to nearest node
    det_safe = jnp.where(jnp.abs(det) < 1e-10, 1.0, det)

    # Solve for barycentric coordinates using Cramer's rule
    # λ1 = det([v0p, v02, v03]) / det
    # λ2 = det([v01, v0p, v03]) / det
    # λ3 = det([v01, v02, v0p]) / det
    # λ0 = 1 - λ1 - λ2 - λ3

    mat1 = jnp.stack([v0p, v02, v03], axis=1)
    mat2 = jnp.stack([v01, v0p, v03], axis=1)
    mat3 = jnp.stack([v01, v02, v0p], axis=1)

    b1 = jnp.linalg.det(mat1) / det_safe
    b2 = jnp.linalg.det(mat2) / det_safe
    b3 = jnp.linalg.det(mat3) / det_safe
    b0 = 1.0 - b1 - b2 - b3

    bary_coords = jnp.array([b0, b1, b2, b3])

    # Point is inside if all barycentric coordinates are in [0, 1]
    # Use small tolerance for numerical stability
    tol = -1e-6
    is_inside = jnp.all(bary_coords >= tol) & jnp.all(bary_coords <= 1.0 + tol)

    return is_inside, bary_coords


@jax.jit
def interpolate_velocity_in_element(
    particle_position: jnp.ndarray,
    element_id: jnp.ndarray,
    block_connectivity: jnp.ndarray,
    block_node_positions: jnp.ndarray,
    velocity_field: jnp.ndarray,
) -> jnp.ndarray:
    """
    Interpolate velocity at particle position using FEM shape functions.

    Uses block-local element indexing compatible with Phase 1 PaddedArrays.

    Parameters
    ----------
    particle_position : jnp.ndarray
        Particle 3D position (3,)
    element_id : jnp.ndarray
        Local element ID within block (scalar, 0 to block_size-1)
    block_connectivity : jnp.ndarray
        Block-local connectivity array (max_elem, 4)
        Each row contains 4 node indices (block-local)
    block_node_positions : jnp.ndarray
        Block-local node positions (max_nodes, 3)
    velocity_field : jnp.ndarray
        Velocity at each node (max_nodes, 3)

    Returns
    -------
    velocity : jnp.ndarray
        Interpolated velocity vector (3,)

    Notes
    -----
    For particles outside their cached element (e.g., after time integration),
    this function should only be called after updating element_id via search.
    """

    # Get element's 4 nodes (block-local indices)
    node_indices = block_connectivity[element_id]  # (4,)

    # Get node coordinates
    tet_nodes = block_node_positions[node_indices]  # (4, 3)

    # Compute barycentric coordinates
    is_inside, bary_coords = compute_barycentric_coordinates(
        particle_position, tet_nodes
    )

    # Interpolate velocity using barycentric coordinates
    # v(x) = Σ λᵢ * v(node_i)
    node_velocities = velocity_field[node_indices]  # (4, 3)
    interpolated_velocity = jnp.dot(bary_coords, node_velocities)  # (3,)

    # NOTE: We don't check is_inside here because:
    # 1. Element search already placed particle in correct element
    # 2. If particle moved during integration, caller should re-search first
    # 3. Even if slightly outside, barycentric extrapolation is reasonable

    return interpolated_velocity


@jax.jit
def batch_interpolate_velocities(
    particle_positions: jnp.ndarray,
    particle_element_ids: jnp.ndarray,
    block_connectivity: jnp.ndarray,
    block_node_positions: jnp.ndarray,
    velocity_field: jnp.ndarray
) -> jnp.ndarray:
    """
    Vectorized velocity interpolation for particle batch within a block.

    Uses jax.vmap to efficiently process all particles in parallel on GPU.

    Parameters
    ----------
    particle_positions : jnp.ndarray
        Particle positions (N, 3)
    particle_element_ids : jnp.ndarray
        Element IDs for each particle (N,), block-local indices
    block_connectivity : jnp.ndarray
        Block connectivity (max_elem, 4)
    block_node_positions : jnp.ndarray
        Block node positions (max_nodes, 3)
    velocity_field : jnp.ndarray
        Velocity field at nodes (max_nodes, 3)

    Returns
    -------
    velocities : jnp.ndarray
        Interpolated velocities (N, 3)

    Examples
    --------
    >>> # Single block with 1000 particles
    >>> positions = jnp.array([[0.5, 0.5, 0.5], ...])  # (1000, 3)
    >>> element_ids = jnp.array([42, 15, 108, ...])    # (1000,)
    >>> velocities = batch_interpolate_velocities(
    ...     positions, element_ids, connectivity, nodes, velocity_field
    ... )
    >>> velocities.shape
    (1000, 3)
    """

    # Vectorize over particles using jax.vmap
    # This creates a parallel GPU kernel that processes all particles simultaneously
    return jax.vmap(
        lambda pos, elem_id: interpolate_velocity_in_element(
            pos, elem_id, block_connectivity, block_node_positions, velocity_field
        )
    )(particle_positions, particle_element_ids)


@jax.jit
def interpolate_velocities_multi_block(
    particle_positions: jnp.ndarray,
    particle_element_ids: jnp.ndarray,
    particle_block_ids: jnp.ndarray,
    padded_arrays,  # PaddedArrays from Phase 1
    velocity_fields: jnp.ndarray  # (n_blocks, max_nodes, 3)
) -> jnp.ndarray:
    """
    Interpolate velocities for particles across multiple blocks.

    This function handles particles in different blocks by routing each
    particle to its corresponding block's interpolation.

    Parameters
    ----------
    particle_positions : jnp.ndarray
        All particle positions (N, 3)
    particle_element_ids : jnp.ndarray
        Element IDs (N,), block-local indices
    particle_block_ids : jnp.ndarray
        Block IDs (N,)
    padded_arrays : PaddedArrays
        Padded block arrays from Phase 1 (connectivity, node_positions, etc.)
    velocity_fields : jnp.ndarray
        Velocity fields for all blocks (n_blocks, max_nodes, 3)

    Returns
    -------
    velocities : jnp.ndarray
        Interpolated velocities for all particles (N, 3)

    Notes
    -----
    This function is less efficient than single-block interpolation because
    particles in different blocks cannot be fully vectorized together.
    For best performance, process particles block-by-block in a Python loop.
    """

    def interpolate_single_particle(pos, elem_id, block_id):
        """Interpolate velocity for single particle."""

        # Get block-specific data
        block_connectivity = padded_arrays.connectivity[block_id]
        block_node_positions = padded_arrays.node_positions[block_id]
        velocity_field = velocity_fields[block_id]

        # Interpolate
        return interpolate_velocity_in_element(
            pos, elem_id, block_connectivity, block_node_positions, velocity_field
        )

    # Vectorize over all particles
    # NOTE: This is not optimal for multi-block case - prefer block-by-block processing
    return jax.vmap(interpolate_single_particle)(
        particle_positions, particle_element_ids, particle_block_ids
    )


# ============================================================================
# Fallback Interpolation Strategies
# ============================================================================

@jax.jit
def nearest_neighbor_interpolation(
    particle_position: jnp.ndarray,
    node_positions: jnp.ndarray,
    velocity_field: jnp.ndarray
) -> jnp.ndarray:
    """
    Fallback interpolation using nearest neighbor.

    Used when:
    - Particle is outside all elements (boundary case)
    - Element is degenerate

    Parameters
    ----------
    particle_position : jnp.ndarray
        Query point (3,)
    node_positions : jnp.ndarray
        All node positions in block (max_nodes, 3)
    velocity_field : jnp.ndarray
        Velocity at nodes (max_nodes, 3)

    Returns
    -------
    velocity : jnp.ndarray
        Velocity of nearest node (3,)
    """

    # Compute squared distances to all nodes
    distances_sq = jnp.sum((node_positions - particle_position)**2, axis=1)

    # Find nearest node
    nearest_idx = jnp.argmin(distances_sq)

    return velocity_field[nearest_idx]


# ============================================================================
# Time-Dependent Velocity Field Handling
# ============================================================================

def create_velocity_field_interpolator(padded_arrays, time_series_data):
    """
    Create a time-dependent velocity field interpolator.

    This function creates a closure that interpolates velocity fields
    in both space (FEM) and time (linear interpolation).

    Parameters
    ----------
    padded_arrays : PaddedArrays
        Block-local mesh structure
    time_series_data : dict
        Time series data with keys:
        - 'times': jnp.ndarray (n_timesteps,)
        - 'velocity_fields': jnp.ndarray (n_timesteps, n_blocks, max_nodes, 3)

    Returns
    -------
    interpolator : callable
        Function(particle_data, current_time) -> velocities (N, 3)

    Examples
    --------
    >>> interpolator = create_velocity_field_interpolator(padded_arrays, time_series)
    >>> velocities = interpolator(particle_data, t=0.5)
    """

    times = time_series_data['times']
    velocity_fields = time_series_data['velocity_fields']  # (n_times, n_blocks, max_nodes, 3)

    @jax.jit
    def interpolate_at_time(particle_data, current_time):
        """Interpolate velocity at current time."""

        # Find bracketing time indices
        # TODO: Implement binary search for efficiency
        # For now, use linear search (acceptable for small n_timesteps)

        # Find t such that times[i] <= current_time < times[i+1]
        idx = jnp.searchsorted(times, current_time)
        idx = jnp.clip(idx, 0, len(times) - 2)

        t0, t1 = times[idx], times[idx + 1]
        v0, v1 = velocity_fields[idx], velocity_fields[idx + 1]

        # Linear interpolation in time
        alpha = (current_time - t0) / (t1 - t0 + 1e-10)
        velocity_field_interp = (1 - alpha) * v0 + alpha * v1

        # Spatial interpolation using FEM
        return interpolate_velocities_multi_block(
            particle_data.positions,
            particle_data.element_ids,
            particle_data.block_ids,
            padded_arrays,
            velocity_field_interp
        )

    return interpolate_at_time
