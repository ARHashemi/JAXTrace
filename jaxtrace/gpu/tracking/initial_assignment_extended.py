#!/usr/bin/env python3
"""
Extended Initial Assignment using L2 Global Morton Search

This module provides a more thorough initial element assignment for particles
using an extended L2 Morton search with larger radius.

The extended search is only used for initial assignment. The regular RK4
integration uses the standard L2 search with smaller radius for efficiency.
"""

import jax
import jax.numpy as jnp
from jaxtrace.gpu.search.morton_global_search import (
    MeshGPUGlobalMorton,
    search_in_leaf_global,
    position_to_leaf_id_octree,
    position_to_leaf_id_linear
)


def search_L2_extended_single(
    pos: jax.Array,
    mesh_gpu: MeshGPUGlobalMorton,
    max_radius: int = 10
) -> jnp.int32:
    """
    Extended L2 Morton search for initial assignment (single particle).

    Searches center leaf and all neighbors within ±max_radius along Morton curve.
    More thorough than standard L2 search used during integration.

    OPTIMIZED: Uses lax.fori_loop to avoid massive XLA graph compilation.
    The unrolled Python loop caused RAM explosion during JIT compilation.

    Args:
        pos: (3,) float32 - particle position
        mesh_gpu: GPU-resident Morton structure
        max_radius: Maximum search radius (default 10, supports up to 300)

    Returns:
        elem_id: int32 - found element, or -1 if not found
    """
    from jax import lax

    # Map position to leaf
    center_leaf_id = jnp.where(
        mesh_gpu.table_depth > 0,
        position_to_leaf_id_octree(pos, mesh_gpu),
        position_to_leaf_id_linear(pos, mesh_gpu)
    )

    # Use lax.fori_loop to avoid massive XLA graph from unrolled Python loop
    # This approach uses bounded iteration without nested vmap

    def search_offset_body(i, elem_id):
        """Search a single offset iteration."""
        offset = i - max_radius
        active = elem_id < 0
        neighbor_leaf = center_leaf_id + offset
        valid = active & (neighbor_leaf >= 0) & (neighbor_leaf < mesh_gpu.n_leaves)
        result = jnp.where(
            valid,
            search_in_leaf_global(pos, neighbor_leaf, mesh_gpu),
            jnp.int32(-1)
        )
        elem_id = jnp.where((result >= 0) & valid, result, elem_id)
        return elem_id

    # Search from 0 to 2*max_radius (representing offsets -max_radius to +max_radius)
    elem_id = lax.fori_loop(
        0,
        2 * max_radius + 1,
        search_offset_body,
        jnp.int32(-1)
    )

    return elem_id


def initial_assignment_extended_batch(
    positions_gpu: jax.Array,
    mesh_gpu_global_morton: MeshGPUGlobalMorton,
    max_radius: int = 10
) -> jax.Array:
    """
    Extended initial assignment for batch of particles.

    Uses larger search radius for more thorough initial element assignment.

    Args:
        positions_gpu: (N, 3) float32 - particle positions
        mesh_gpu_global_morton: GPU-resident Morton structure
        max_radius: Maximum search radius (default 10)

    Returns:
        element_ids: (N,) int32 - found elements, -1 if not found
    """
    # Vmap over all particles and JIT compile
    @jax.jit
    def search_batch(positions):
        return jax.vmap(
            lambda pos: search_L2_extended_single(pos, mesh_gpu_global_morton, max_radius)
        )(positions)

    return search_batch(positions_gpu)
