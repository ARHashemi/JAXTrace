"""
Phase 2: Mesh-Aligned Octree WITH Neighbor Search (Memory-Safe Version)

Simplified version that searches ONLY at finest level (14) with 26 neighbors.
NO lax.cond, only jnp.where - completely safe for vmap.

Expected performance:
- ~99% searchability for particles inside mesh
- ~15-20 tests per particle
- ~50-100K particles/sec
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple

# Import point-in-tet methods
from jaxtrace.gpu.search.point_in_tet_methods import point_in_tet_gpu as point_in_tet_dispatcher
import jaxtrace.config as config

# Import GPU structure
from .mesh_aligned_octree_gpu import (
    MeshAlignedOctreeGPU,
    encode_morton_3d_jax,
    find_cell_by_morton_and_level,
    get_cell_elements,
)


# ============================================================================
# Helper: Search One Cell (No Conditionals)
# ============================================================================

def search_one_cell_unconditional(
    pos: jax.Array,
    grid_i: jnp.int32,
    grid_j: jnp.int32,
    grid_k: jnp.int32,
    level: jnp.uint8,
    octree_gpu: MeshAlignedOctreeGPU,
    max_elements: jnp.int32 = 50
) -> jnp.int32:
    """
    Search one cell UNCONDITIONALLY (always executes, uses jnp.where for result).

    This is safe for vmap - no conditional execution.

    Args:
        pos: (3,) query position
        grid_i/j/k: Grid indices
        level: Refinement level
        octree_gpu: GPU octree
        max_elements: Max elements to test in this cell

    Returns:
        Element ID (-1 if not found)
    """
    # Find cell at grid indices
    i_offset = jnp.clip(grid_i + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)
    j_offset = jnp.clip(grid_j + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)
    k_offset = jnp.clip(grid_k + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)

    morton = encode_morton_3d_jax(i_offset, j_offset, k_offset)
    cell_idx = find_cell_by_morton_and_level(morton, level, octree_gpu.cell_morton_codes, octree_gpu.cell_levels)

    # Get elements in cell
    start_idx, n_elements_in_cell = get_cell_elements(
        cell_idx, octree_gpu.cell_to_elements_offsets, octree_gpu.cell_to_elements_data
    )

    # Limit number of tests
    n_to_test = jnp.minimum(n_elements_in_cell, max_elements)

    # Test elements (always runs, but returns -1 if cell doesn't exist)
    def test_element(i, found_elem):
        elem_idx = start_idx + i
        # Clamp to valid range to prevent out-of-bounds access
        elem_idx_safe = jnp.clip(elem_idx, 0, octree_gpu.cell_to_elements_data.shape[0] - 1)
        elem_id = octree_gpu.cell_to_elements_data[elem_idx_safe]
        # Also clamp elem_id to valid range
        elem_id_safe = jnp.clip(elem_id, 0, octree_gpu.connectivity.shape[0] - 1)

        is_inside = point_in_tet_dispatcher(
            pos, elem_id_safe, octree_gpu.connectivity, octree_gpu.node_positions, config.POINT_IN_TET_METHOD
        )

        # Update if found (and not already found) - use original elem_id for result
        new_found = jnp.where(
            jnp.logical_and(is_inside, found_elem < 0),
            elem_id,  # Use original, not clamped
            found_elem
        )
        return new_found

    # Test elements
    found_elem = lax.fori_loop(0, n_to_test, test_element, jnp.int32(-1))

    # Return -1 if cell didn't exist
    result = jnp.where(cell_idx >= 0, found_elem, jnp.int32(-1))

    return result


# ============================================================================
# Single Particle Search with Neighbors (Unrolled, No Conditionals)
# ============================================================================

def search_with_neighbors_single(
    pos: jax.Array,
    octree_gpu: MeshAlignedOctreeGPU,
    level: jnp.int32 = 14,
    max_elements_per_cell: jnp.int32 = 20
) -> jnp.int32:
    """
    Search at one level with 27 cells (center + 26 neighbors).

    FULLY UNROLLED - no conditionals, just pure functional operations.
    Safe for vmap - no memory explosion.

    Args:
        pos: (3,) query position
        octree_gpu: GPU octree
        level: Refinement level to search (default 14 - finest)
        max_elements_per_cell: Max elements to test per cell

    Returns:
        Element ID (-1 if not found)
    """
    # Compute primary grid indices
    cell_size = octree_gpu.level_cell_sizes[level]
    grid_i = jnp.floor(pos[0] / cell_size[0]).astype(jnp.int32)
    grid_j = jnp.floor(pos[1] / cell_size[1]).astype(jnp.int32)
    grid_k = jnp.floor(pos[2] / cell_size[2]).astype(jnp.int32)

    # Search 27 cells: center + 26 neighbors
    # FULLY UNROLLED - JAX will compile this to efficient code
    found = jnp.int32(-1)
    level_u8 = jnp.uint8(level)

    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                # Compute neighbor grid
                ni = grid_i + di
                nj = grid_j + dj
                nk = grid_k + dk

                # Search this cell (ALWAYS executes)
                elem = search_one_cell_unconditional(
                    pos, ni, nj, nk, level_u8, octree_gpu, max_elements_per_cell
                )

                # Update found (only if not already found)
                found = jnp.where(
                    jnp.logical_and(elem >= 0, found < 0),
                    elem,
                    found
                )

    return found


# ============================================================================
# Multi-Level Search with Neighbors
# ============================================================================

def search_multi_level_with_neighbors(
    pos: jax.Array,
    octree_gpu: MeshAlignedOctreeGPU,
    levels_to_try: Tuple[int, ...] = (14, 13, 12),
    max_elements_per_cell: jnp.int32 = 20
) -> jnp.int32:
    """
    Search multiple levels sequentially with neighbor fallback.

    Args:
        pos: (3,) query position
        octree_gpu: GPU octree
        levels_to_try: Tuple of levels to search (default: 14, 13, 12)
        max_elements_per_cell: Max elements to test per cell

    Returns:
        Element ID (-1 if not found)
    """
    found = jnp.int32(-1)

    for level in levels_to_try:
        # Search this level with neighbors
        elem = search_with_neighbors_single(pos, octree_gpu, level, max_elements_per_cell)

        # Update if found
        found = jnp.where(
            jnp.logical_and(elem >= 0, found < 0),
            elem,
            found
        )

    return found


# ============================================================================
# Batch Search
# ============================================================================

def search_mesh_aligned_with_neighbors_batch(
    positions: jax.Array,
    octree_gpu: MeshAlignedOctreeGPU,
    levels_to_try: Tuple[int, ...] = (14, 13, 12),
    max_elements_per_cell: jnp.int32 = 20
) -> jax.Array:
    """
    Batch search with neighbors (vmap-safe).

    Args:
        positions: (n_particles, 3) query positions
        octree_gpu: GPU octree
        levels_to_try: Levels to search
        max_elements_per_cell: Max elements per cell

    Returns:
        elem_ids: (n_particles,) element IDs (-1 if not found)
    """
    elem_ids = jax.vmap(
        lambda pos: search_multi_level_with_neighbors(
            pos, octree_gpu, levels_to_try, max_elements_per_cell
        )
    )(positions)

    return elem_ids
