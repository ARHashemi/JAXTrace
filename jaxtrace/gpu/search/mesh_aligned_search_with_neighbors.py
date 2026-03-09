"""
Mesh-Aligned Octree Point Location with Pre-Computed Neighbor Table

GPU search kernel that uses pre-computed neighbor indices for neighbor search.
This avoids all JAX tracing issues by performing O(1) neighbor lookups.

Expected performance:
- ~99% searchability
- ~15-20 tests per particle
- ~50-100K particles/sec
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple

# Import point-in-tet dispatcher
from jaxtrace.gpu.search.point_in_tet_methods import point_in_tet_gpu as point_in_tet_dispatcher
import jaxtrace.config as config

# Import GPU structures
from .mesh_aligned_octree_with_neighbor_table import MeshAlignedOctreeGPUWithNeighbors
from .mesh_aligned_octree_gpu import (
    encode_morton_3d_jax,
    position_to_grid_indices,
    find_cell_by_morton_and_level,
    get_cell_elements,
)


# ============================================================================
# Helper: Search Elements in One Cell
# ============================================================================

def search_elements_in_cell(
    pos: jax.Array,
    cell_idx: jnp.int32,
    octree_gpu: MeshAlignedOctreeGPUWithNeighbors,
    max_tests: jnp.int32 = 30
) -> Tuple[jnp.int32, jnp.int32]:
    """
    Search elements in one cell.

    Args:
        pos: (3,) query position
        cell_idx: Cell index to search (-1 if invalid)
        octree_gpu: GPU octree with neighbors
        max_tests: Max elements to test in this cell

    Returns:
        (found_elem_id, n_tests): Element ID (-1 if not found) and test count
    """
    # Get elements in cell
    start_idx, n_elements_in_cell = get_cell_elements(
        cell_idx, octree_gpu.cell_to_elements_offsets, octree_gpu.cell_to_elements_data
    )

    # Limit tests
    n_to_test = jnp.minimum(n_elements_in_cell, max_tests)

    # Test elements
    def test_element(i, carry):
        found_elem, n_tests = carry

        # Only test if not already found
        should_test = jnp.logical_and(found_elem < 0, i < n_to_test)

        elem_idx = start_idx + i
        # Clamp to valid range
        elem_idx_safe = jnp.clip(elem_idx, 0, octree_gpu.cell_to_elements_data.shape[0] - 1)
        elem_id = octree_gpu.cell_to_elements_data[elem_idx_safe]
        # Clamp element ID
        elem_id_safe = jnp.clip(elem_id, 0, octree_gpu.connectivity.shape[0] - 1)

        # Test point-in-tet
        is_inside = jnp.where(
            should_test,
            point_in_tet_dispatcher(
                pos, elem_id_safe, octree_gpu.connectivity, octree_gpu.node_positions,
                config.POINT_IN_TET_METHOD
            ),
            False
        )

        # Update result
        new_found = jnp.where(is_inside, elem_id, found_elem)
        new_n_tests = jnp.where(should_test, n_tests + 1, n_tests)

        return (new_found, new_n_tests)

    init_state = (jnp.int32(-1), jnp.int32(0))
    found_elem, n_tests = lax.fori_loop(0, max_tests, test_element, init_state)

    # Return -1 tests if cell was invalid
    n_tests_final = jnp.where(cell_idx >= 0, n_tests, jnp.int32(0))

    return found_elem, n_tests_final


# ============================================================================
# Single Particle Search with Pre-Computed Neighbors
# ============================================================================

def search_with_precomputed_neighbors_single(
    pos: jax.Array,
    octree_gpu: MeshAlignedOctreeGPUWithNeighbors,
    level: jnp.int32 = 14,
    max_tests_per_cell: jnp.int32 = 30
) -> Tuple[jnp.int32, jnp.int32]:
    """
    Search one particle with pre-computed neighbor table.

    Algorithm:
    1. Find primary cell at specified level
    2. Search primary cell elements
    3. If not found, search 26 pre-computed neighbors
    4. Return element ID and total test count

    Args:
        pos: (3,) query position
        octree_gpu: GPU octree with neighbor table
        level: Refinement level to search (default 14 - finest)
        max_tests_per_cell: Max elements to test per cell

    Returns:
        (elem_id, n_tests): Element ID (-1 if not found) and total tests
    """
    # Find primary cell
    cell_size = octree_gpu.level_cell_sizes[level]
    grid_i, grid_j, grid_k = position_to_grid_indices(pos, cell_size)

    # Encode to Morton
    i_offset = jnp.clip(grid_i + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)
    j_offset = jnp.clip(grid_j + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)
    k_offset = jnp.clip(grid_k + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)

    morton = encode_morton_3d_jax(i_offset, j_offset, k_offset)
    primary_cell_idx = find_cell_by_morton_and_level(
        morton, jnp.uint8(level), octree_gpu.cell_morton_codes, octree_gpu.cell_levels
    )

    # Search primary cell
    found_elem, n_tests = search_elements_in_cell(
        pos, primary_cell_idx, octree_gpu, max_tests_per_cell
    )

    # Search 26 neighbors using pre-computed table (always execute)
    def search_neighbor(i, carry_inner):
        found_inner, n_tests_inner = carry_inner

        # Get neighbor cell index from pre-computed table
        neighbor_idx = jnp.where(
            primary_cell_idx >= 0,
            octree_gpu.cell_neighbors[primary_cell_idx, i],
            jnp.int32(-1)
        )

        # ALWAYS search (unconditional execution)
        # search_elements_in_cell handles invalid cells safely
        elem_neighbor, tests_neighbor = search_elements_in_cell(
            pos, neighbor_idx, octree_gpu, max_tests_per_cell
        )

        # Only update if: not already found AND neighbor exists AND actually found in neighbor
        should_update_found = jnp.logical_and(
            jnp.logical_and(found_inner < 0, neighbor_idx >= 0),
            elem_neighbor >= 0
        )
        should_count_tests = jnp.logical_and(found_inner < 0, neighbor_idx >= 0)

        new_found = jnp.where(should_update_found, elem_neighbor, found_inner)
        new_n_tests = jnp.where(should_count_tests, n_tests_inner + tests_neighbor, n_tests_inner)

        return (new_found, new_n_tests)

    # Loop over 26 neighbors (even if found in primary - will just not update)
    final_elem, final_n_tests = lax.fori_loop(0, 26, search_neighbor, (found_elem, n_tests))

    return final_elem, final_n_tests


# ============================================================================
# Multi-Level Search with Pre-Computed Neighbors
# ============================================================================

def search_multi_level_with_precomputed_neighbors(
    pos: jax.Array,
    octree_gpu: MeshAlignedOctreeGPUWithNeighbors,
    levels_to_try: Tuple[int, ...] = (14, 13, 12),
    max_tests_per_cell: jnp.int32 = 30
) -> Tuple[jnp.int32, jnp.int32]:
    """
    Search multiple levels sequentially with neighbor fallback.

    Args:
        pos: (3,) query position
        octree_gpu: GPU octree with neighbors
        levels_to_try: Tuple of levels to search (default: 14, 13, 12)
        max_tests_per_cell: Max elements to test per cell

    Returns:
        (elem_id, n_tests): Element ID (-1 if not found) and total tests
    """
    found_elem = jnp.int32(-1)
    total_tests = jnp.int32(0)

    for level in levels_to_try:
        # Search this level with neighbors
        elem, tests = search_with_precomputed_neighbors_single(
            pos, octree_gpu, level, max_tests_per_cell
        )

        # Update if found
        found_elem = jnp.where(
            jnp.logical_and(elem >= 0, found_elem < 0),
            elem,
            found_elem
        )
        total_tests = total_tests + tests

        # Early exit if found (note: this won't actually exit in JIT, but saves computation)
        # The jnp.where above already handles not overwriting found results

    return found_elem, total_tests


# ============================================================================
# Batch Search (vmap-safe)
# ============================================================================

def search_batch_with_precomputed_neighbors(
    positions: jax.Array,
    octree_gpu: MeshAlignedOctreeGPUWithNeighbors,
    levels_to_try: Tuple[int, ...] = (14,),
    max_tests_per_cell: jnp.int32 = 30
) -> Tuple[jax.Array, jax.Array]:
    """
    Batch search with pre-computed neighbors (vmap-safe).

    Args:
        positions: (n_particles, 3) query positions
        octree_gpu: GPU octree with neighbor table
        levels_to_try: Levels to search
        max_tests_per_cell: Max elements per cell

    Returns:
        (elem_ids, n_tests): (n_particles,) element IDs and test counts
    """
    elem_ids, n_tests = jax.vmap(
        lambda pos: search_multi_level_with_precomputed_neighbors(
            pos, octree_gpu, levels_to_try, max_tests_per_cell
        )
    )(positions)

    return elem_ids, n_tests
