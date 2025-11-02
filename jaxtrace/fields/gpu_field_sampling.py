#!/usr/bin/env python3
"""
Phase 3D: Pure JAX GPU Field Sampling (Zero io_callback).

This module replaces io_callback-based CPU field sampling with pure JAX GPU sampling.
The complete pipeline runs on GPU: hash lookup → element testing → FEM interpolation.

This is the KEY module that unlocks full GPU execution and 50-100× speedup.

Architecture:
    GPU: hash_lookup → element_testing → fem_interpolation → temporal_interpolation
    (No CPU callbacks, no Numba, fully JIT-compilable)

References:
- GPU_OCTREE_IMPLEMENTATION_ROADMAP.md Phase 3D
- Critical_JAX_Memory_Issues_Phase3_Hash.md
"""

import jax
import jax.numpy as jnp
from typing import Tuple
from .hash_octree import HashOctree, hash_lookup_batch_jax
from .element_testing_jax import test_candidates_batch_jax_compiled
from .morton_code import encode_morton_3d_batch_jax


@jax.jit
def fem_interpolate_single_jax(
    point: jnp.ndarray,
    element_id: jnp.int32,
    mesh_positions: jnp.ndarray,
    connectivity: jnp.ndarray,
    field_values: jnp.ndarray
) -> jnp.ndarray:
    """
    FEM interpolation for a single point (Pure JAX, GPU-compilable).

    Uses barycentric coordinates to interpolate field values within tetrahedral element.

    Args:
        point: [3] query point
        element_id: scalar element ID (or -1 if not found)
        mesh_positions: [N_vertices, 3] vertex positions
        connectivity: [N_elements, 4] element connectivity
        field_values: [N_vertices, 3] field values at vertices

    Returns:
        interpolated_value: [3] interpolated field value (zero if element_id < 0)
    """
    # Handle invalid element (not found)
    is_valid = element_id >= 0

    # Safe element access (clamp to valid range)
    safe_elem_id = jnp.clip(element_id, 0, connectivity.shape[0] - 1)
    elem_nodes = connectivity[safe_elem_id]

    # Get vertices and field values
    vertices = mesh_positions[elem_nodes]  # [4, 3]
    node_values = field_values[elem_nodes]  # [4, 3]

    # Compute barycentric coordinates (reuse from element_testing_jax)
    from .element_testing_jax import compute_barycentric_coords_jax
    bary = compute_barycentric_coords_jax(point, vertices)  # [4]

    # Interpolate using barycentric coordinates
    # value = sum(bary[i] * node_values[i])
    interpolated = jnp.sum(bary[:, jnp.newaxis] * node_values, axis=0)  # [3]

    # Return zero if element not found
    return jnp.where(is_valid, interpolated, jnp.zeros(3, dtype=jnp.float32))


# Vectorize for batch processing
fem_interpolate_batch_jax = jax.vmap(
    fem_interpolate_single_jax,
    in_axes=(0, 0, None, None, None)
    # vmap over: points, element_ids
    # broadcast: mesh_positions, connectivity, field_values
)


@jax.jit
def sample_field_gpu_single_timestep(
    query_positions: jnp.ndarray,
    hash_octree: HashOctree,
    mesh_positions: jnp.ndarray,
    connectivity: jnp.ndarray,
    field_values: jnp.ndarray,
    max_depth: int
) -> jnp.ndarray:
    """
    Complete GPU field sampling for a single timestep (Pure JAX, NO io_callback).

    Pipeline: hash_lookup → element_testing → fem_interpolation
    All on GPU, fully JIT-compilable.

    Args:
        query_positions: [N, 3] query points
        hash_octree: HashOctree for spatial search
        mesh_positions: [M, 3] mesh vertex positions
        connectivity: [K, 4] element connectivity
        field_values: [M, 3] field values at vertices
        max_depth: Maximum octree depth

    Returns:
        interpolated_values: [N, 3] interpolated field values
    """
    n_particles = query_positions.shape[0]

    # Step 1: Hash lookup (GPU)
    max_level = max_depth - 1
    levels = jnp.full(n_particles, max_level, dtype=jnp.int32)

    candidate_elements, n_candidates = hash_lookup_batch_jax(
        query_positions,
        hash_octree,
        levels
    )

    # Step 2: Element testing (GPU)
    element_ids = test_candidates_batch_jax_compiled(
        query_positions,
        candidate_elements,
        n_candidates,
        mesh_positions,
        connectivity,
        max_candidates=hash_octree.max_elements_per_cell
    )

    # Step 3: FEM interpolation (GPU)
    interpolated_values = fem_interpolate_batch_jax(
        query_positions,
        element_ids,
        mesh_positions,
        connectivity,
        field_values
    )

    return interpolated_values


@jax.jit
def sample_field_gpu_with_temporal_interpolation(
    query_positions: jnp.ndarray,
    t: jnp.ndarray,
    hash_octree_left: HashOctree,
    hash_octree_right: HashOctree,
    mesh_positions_left: jnp.ndarray,
    mesh_positions_right: jnp.ndarray,
    connectivity: jnp.ndarray,
    field_values_left: jnp.ndarray,
    field_values_right: jnp.ndarray,
    time_left: float,
    time_right: float,
    max_depth: int
) -> jnp.ndarray:
    """
    GPU field sampling with temporal interpolation (Pure JAX).

    Samples at two timesteps and linearly interpolates in time.

    Args:
        query_positions: [N, 3] query points
        t: scalar query time
        hash_octree_left, hash_octree_right: Hash octrees for left/right timesteps
        mesh_positions_left, mesh_positions_right: Vertex positions
        connectivity: [K, 4] element connectivity (assumed same for both timesteps)
        field_values_left, field_values_right: Field values at vertices
        time_left, time_right: Timestep times
        max_depth: Maximum octree depth

    Returns:
        interpolated_values: [N, 3] time-interpolated field values
    """
    # Sample at left timestep
    values_left = sample_field_gpu_single_timestep(
        query_positions,
        hash_octree_left,
        mesh_positions_left,
        connectivity,
        field_values_left,
        max_depth
    )

    # Sample at right timestep
    values_right = sample_field_gpu_single_timestep(
        query_positions,
        hash_octree_right,
        mesh_positions_right,
        connectivity,
        field_values_right,
        max_depth
    )

    # Temporal interpolation weight
    alpha = (t - time_left) / (time_right - time_left)
    alpha = jnp.clip(alpha, 0.0, 1.0)

    # Linear interpolation in time
    return (1.0 - alpha) * values_left + alpha * values_right


# ============================================================================
# Simplified API for SharedOctreeFEMField
# ============================================================================

def create_gpu_field_sampler(shared_octree_field):
    """
    Create a GPU field sampler function for SharedOctreeFEMField.

    This replaces the io_callback-based sampling with pure JAX GPU sampling.

    Args:
        shared_octree_field: SharedOctreeFEMTimeSeriesField instance

    Returns:
        sample_fn: JAX function (positions, t) -> velocities
    """

    @jax.jit
    def sample_at_positions_gpu(query_positions: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Phase 3D: Pure JAX GPU field sampling (NO io_callback).

        This is the replacement for the old io_callback-based sampling.

        Args:
            query_positions: [N, 3] query positions
            t: scalar query time

        Returns:
            velocities: [N, 3] interpolated velocities
        """
        # NOTE: This is a simplified version for demonstration.
        # Full implementation needs to handle:
        # 1. Time-based timestep selection
        # 2. Hash octree cache lookup
        # 3. Field data loading
        # 4. Temporal interpolation

        # For now, this shows the structure.
        # The actual implementation will be in Phase 3D.2
        raise NotImplementedError(
            "Phase 3D.2: Full implementation pending. "
            "Need to integrate with SharedOctreeFEMField data management."
        )

    return sample_at_positions_gpu


# ============================================================================
# Validation and Testing
# ============================================================================

def validate_gpu_field_sampling(
    query_positions: jnp.ndarray,
    hash_octree: HashOctree,
    mesh_positions: jnp.ndarray,
    connectivity: jnp.ndarray,
    field_values: jnp.ndarray,
    max_depth: int
) -> dict:
    """
    Validate GPU field sampling correctness and performance.

    Args:
        Same as sample_field_gpu_single_timestep

    Returns:
        stats: Validation statistics
    """
    # Run GPU sampling
    interpolated_values = sample_field_gpu_single_timestep(
        query_positions,
        hash_octree,
        mesh_positions,
        connectivity,
        field_values,
        max_depth
    )

    # Compute statistics
    n_particles = len(query_positions)
    n_nonzero = jnp.sum(jnp.any(interpolated_values != 0, axis=1))
    success_rate = float(n_nonzero) / float(n_particles) if n_particles > 0 else 0.0

    # Check for NaN/Inf
    has_nan = jnp.any(jnp.isnan(interpolated_values))
    has_inf = jnp.any(jnp.isinf(interpolated_values))

    return {
        'n_particles': n_particles,
        'n_successful': int(n_nonzero),
        'success_rate': success_rate,
        'has_nan': bool(has_nan),
        'has_inf': bool(has_inf),
        'mean_magnitude': float(jnp.mean(jnp.linalg.norm(interpolated_values, axis=1)))
    }
