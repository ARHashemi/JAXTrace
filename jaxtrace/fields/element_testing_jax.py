#!/usr/bin/env python3
"""
Phase 3C: Pure JAX Element Testing (Zero Numba).

Replaces Numba CPU element testing with JAX GPU-compilable functions.
This enables element testing to run on GPU alongside hash lookup.

Key Features:
- Pure JAX (no Numba, no NumPy)
- GPU-compilable with @jax.jit
- Vectorized with vmap for batch processing
- Bounded loops (fori_loop) to avoid memory explosion
- Safe for JAX compilation (no dynamic slicing)

References:
- Critical_JAX_Memory_Issues_Phase3_Hash.md
- GPU_OCTREE_IMPLEMENTATION_ROADMAP.md Phase 3
"""

import jax
import jax.numpy as jnp
from typing import Tuple


@jax.jit
def compute_barycentric_coords_jax(point: jnp.ndarray, vertices: jnp.ndarray) -> jnp.ndarray:
    """
    Compute barycentric coordinates for tetrahedral element (Pure JAX).

    Solves: point = bary[0]*v0 + bary[1]*v1 + bary[2]*v2 + bary[3]*v3
    where sum(bary) = 1.0

    Uses lstsq for numerical stability with near-degenerate elements.

    Args:
        point: [3] query point coordinates
        vertices: [4, 3] element vertices (4 nodes, 3D coordinates)

    Returns:
        bary: [4] barycentric coordinates (sum = 1.0 if valid)
    """
    v0 = vertices[0]
    v1 = vertices[1]
    v2 = vertices[2]
    v3 = vertices[3]

    # Build matrix [v1-v0, v2-v0, v3-v0]
    # This gives the 3 edge vectors from v0
    mat = jnp.stack([v1 - v0, v2 - v0, v3 - v0], axis=-1)

    # Right-hand side: point - v0
    rhs = point - v0

    # Solve for bary[1:4] using least squares
    # lstsq is more stable than direct solve for near-degenerate elements
    bary123, residuals, rank, s = jnp.linalg.lstsq(mat, rhs, rcond=None)

    # Compute bary[0] from constraint: sum(bary) = 1
    bary0 = 1.0 - jnp.sum(bary123)

    # Pack into [4] array
    return jnp.array([bary0, bary123[0], bary123[1], bary123[2]], dtype=jnp.float32)


@jax.jit
def is_inside_tetrahedron_jax(bary: jnp.ndarray, tolerance: float = 1e-6) -> jnp.bool_:
    """
    Check if barycentric coordinates indicate point inside tetrahedron.

    A point is inside if all barycentric coordinates are non-negative
    and sum to approximately 1.0 (within tolerance).

    Args:
        bary: [4] barycentric coordinates
        tolerance: Numerical tolerance for boundary checks

    Returns:
        is_inside: Boolean indicating if point is inside element
    """
    # Check all coordinates >= -tolerance (allowing small numerical errors)
    all_positive = jnp.all(bary >= -tolerance)

    # Check sum approximately equals 1.0
    sum_valid = jnp.sum(bary) <= 1.0 + tolerance

    return all_positive & sum_valid


@jax.jit
def test_single_particle_jax(
    point: jnp.ndarray,
    candidate_elements: jnp.ndarray,
    n_candidates: jnp.ndarray,
    mesh_positions: jnp.ndarray,
    connectivity: jnp.ndarray,
    max_candidates: int
) -> jnp.int32:
    """
    Test candidate elements for a single particle (Pure JAX).

    Uses bounded fori_loop to test up to max_candidates elements.
    Stops at first element that contains the point.

    Args:
        point: [3] query point
        candidate_elements: [max_candidates] candidate element IDs (-1 padded)
        n_candidates: scalar number of valid candidates
        mesh_positions: [N_vertices, 3] mesh vertex positions
        connectivity: [N_elements, 4] element connectivity (vertex indices)
        max_candidates: Maximum candidates to test (compile-time constant)

    Returns:
        element_id: ID of containing element, or -1 if not found
    """

    def test_one_candidate(i, best_elem):
        """Test candidate i, update best_elem if inside."""
        # Check if this iteration is valid
        is_valid_iteration = i < n_candidates
        elem_id = candidate_elements[i]
        is_valid_elem = elem_id >= 0
        already_found = best_elem >= 0

        # Only test if valid and not already found
        should_test = is_valid_iteration & is_valid_elem & (~already_found)

        # Get vertices (safe even if elem_id invalid - clamp to bounds)
        # We won't use the result if should_test is False
        safe_elem_id = jnp.clip(elem_id, 0, connectivity.shape[0] - 1)
        elem_nodes = connectivity[safe_elem_id]

        # Get vertex positions [4, 3]
        vertices = mesh_positions[elem_nodes]

        # Compute barycentric coordinates
        bary = compute_barycentric_coords_jax(point, vertices)

        # Check if inside
        is_inside = is_inside_tetrahedron_jax(bary)

        # Update best_elem if inside and should_test
        return jnp.where(should_test & is_inside, elem_id, best_elem)

    # Bounded loop over candidates (max_candidates is compile-time constant)
    initial_elem = jnp.int32(-1)
    return jax.lax.fori_loop(0, max_candidates, test_one_candidate, initial_elem)


# Vectorized version for batch processing
test_candidates_batch_jax = jax.vmap(
    test_single_particle_jax,
    in_axes=(0, 0, 0, None, None, None)
    # vmap over: points, candidate_elements, n_candidates
    # broadcast: mesh_positions, connectivity, max_candidates
)


@jax.jit
def test_candidates_batch_jax_compiled(
    query_positions: jnp.ndarray,
    candidate_elements: jnp.ndarray,
    n_candidates: jnp.ndarray,
    mesh_positions: jnp.ndarray,
    connectivity: jnp.ndarray,
    max_candidates: int
) -> jnp.ndarray:
    """
    Wrapper for test_candidates_batch_jax with explicit JIT compilation.

    This is the main entry point for batch element testing on GPU.

    Args:
        query_positions: [N_particles, 3] query points
        candidate_elements: [N_particles, max_candidates] candidate element IDs
        n_candidates: [N_particles] number of valid candidates per particle
        mesh_positions: [N_vertices, 3] mesh vertex positions
        connectivity: [N_elements, 4] element connectivity
        max_candidates: Maximum candidates per particle (compile-time constant)

    Returns:
        element_ids: [N_particles] containing element IDs (-1 if not found)
    """
    return test_candidates_batch_jax(
        query_positions,
        candidate_elements,
        n_candidates,
        mesh_positions,
        connectivity,
        max_candidates
    )


# ============================================================================
# Utility Functions
# ============================================================================

def validate_element_testing(
    query_positions: jnp.ndarray,
    candidate_elements: jnp.ndarray,
    n_candidates: jnp.ndarray,
    mesh_positions: jnp.ndarray,
    connectivity: jnp.ndarray,
    max_candidates: int
) -> dict:
    """
    Validate element testing correctness and performance.

    Args:
        Same as test_candidates_batch_jax_compiled

    Returns:
        stats: Dictionary with validation statistics
    """
    # Run element testing
    element_ids = test_candidates_batch_jax_compiled(
        query_positions,
        candidate_elements,
        n_candidates,
        mesh_positions,
        connectivity,
        max_candidates
    )

    # Compute statistics
    n_particles = len(query_positions)
    n_found = jnp.sum(element_ids >= 0)
    n_not_found = jnp.sum(element_ids < 0)
    success_rate = float(n_found) / float(n_particles) if n_particles > 0 else 0.0

    return {
        'n_particles': n_particles,
        'n_found': int(n_found),
        'n_not_found': int(n_not_found),
        'success_rate': success_rate,
        'max_candidates': max_candidates
    }
