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

    Args:
        pos: (3,) float32 - particle position
        mesh_gpu: GPU-resident Morton structure
        max_radius: Maximum search radius (default 10)

    Returns:
        elem_id: int32 - found element, or -1 if not found
    """
    # Map position to leaf
    center_leaf_id = jnp.where(
        mesh_gpu.table_depth > 0,
        position_to_leaf_id_octree(pos, mesh_gpu),
        position_to_leaf_id_linear(pos, mesh_gpu)
    )

    # Search all leaves within ±max_radius
    def search_neighbor_leaf(offset):
        neighbor_leaf = center_leaf_id + offset
        valid = (neighbor_leaf >= 0) & (neighbor_leaf < mesh_gpu.n_leaves)
        result = jnp.where(
            valid,
            search_in_leaf_global(pos, neighbor_leaf, mesh_gpu),
            jnp.int32(-1)
        )
        return result

    # Create all offsets from -max_radius to +max_radius
    offsets = jnp.arange(-max_radius, max_radius + 1, dtype=jnp.int32)

    # Search all neighbors (vectorized)
    neighbor_results = jax.vmap(search_neighbor_leaf)(offsets)

    # Find first valid result
    neighbor_mask = neighbor_results >= 0
    elem_id = jnp.where(
        jnp.any(neighbor_mask),
        neighbor_results[jnp.argmax(neighbor_mask)],
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
