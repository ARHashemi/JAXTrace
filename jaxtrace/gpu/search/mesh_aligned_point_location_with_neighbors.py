"""
Phase 2: Mesh-Aligned Octree Point Location WITH 26-NEIGHBOR SEARCH

JAX GPU kernels for fast point-in-element search using mesh-aligned octree
with 26-neighbor cell fallback.

Architecture:
    1. Position → Find primary cell at finest level
    2. Search elements in primary cell
    3. If not found, search 26 spatial neighbors (unrolled static loop)
    4. If still not found, try parent level
    5. Return first containing element

Expected Performance:
    - ~15-20 point-in-tet tests per query (vs ~5 without neighbors)
    - ~99% searchability (vs ~75% without neighbors)
    - ~50-100K particles/sec throughput
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
# Helper: Find Neighbor Cell
# ============================================================================

def find_neighbor_cell_at_grid(
    grid_i: jnp.int32,
    grid_j: jnp.int32,
    grid_k: jnp.int32,
    level: jnp.uint8,
    octree_gpu: MeshAlignedOctreeGPU
) -> jnp.int32:
    """
    Find cell at given grid indices and level.

    Args:
        grid_i, grid_j, grid_k: Grid indices
        level: Refinement level
        octree_gpu: GPU octree structure

    Returns:
        Cell index (-1 if not found)
    """
    # Apply offset for negative coordinates
    i_offset = jnp.clip(grid_i + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)
    j_offset = jnp.clip(grid_j + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)
    k_offset = jnp.clip(grid_k + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)

    # Encode to Morton
    morton_code = encode_morton_3d_jax(i_offset, j_offset, k_offset)

    # Binary search for (morton, level)
    cell_idx = find_cell_by_morton_and_level(
        morton_code,
        level,
        octree_gpu.cell_morton_codes,
        octree_gpu.cell_levels
    )

    return cell_idx


# ============================================================================
# Helper: Search Elements in Cell
# ============================================================================

def search_elements_in_cell(
    pos: jax.Array,
    cell_idx: jnp.int32,
    octree_gpu: MeshAlignedOctreeGPU,
    max_tests: jnp.int32,
    current_tests: jnp.int32
) -> Tuple[jnp.int32, jnp.int32]:
    """
    Search all elements in a given cell.

    Args:
        pos: (3,) query position
        cell_idx: Cell index to search
        octree_gpu: GPU octree structure
        max_tests: Maximum total tests allowed
        current_tests: Tests already performed

    Returns:
        (elem_id, n_tests): Found element ID (-1 if not found), total tests performed
    """
    # Get elements in this cell
    start_idx, n_elements_in_cell = get_cell_elements(
        cell_idx,
        octree_gpu.cell_to_elements_offsets,
        octree_gpu.cell_to_elements_data
    )

    # Only search if cell exists and we have tests remaining
    cell_exists = cell_idx >= 0
    tests_remaining = max_tests - current_tests
    n_to_test = jnp.where(
        cell_exists,
        jnp.minimum(n_elements_in_cell, tests_remaining),
        0
    )

    # Test elements in this cell
    def test_element(i, inner_carry):
        inner_found, inner_tests = inner_carry

        elem_idx = start_idx + i
        elem_id = octree_gpu.cell_to_elements_data[elem_idx]

        # Point-in-tet test
        is_inside = point_in_tet_dispatcher(
            pos,
            elem_id,
            octree_gpu.connectivity,
            octree_gpu.node_positions,
            method=config.POINT_IN_TET_METHOD
        )

        # Update if found
        new_found = jnp.where(
            jnp.logical_or(inner_found >= 0, jnp.logical_not(is_inside)),
            inner_found,
            elem_id
        )

        return (new_found, inner_tests + 1)

    # Run tests
    found_elem_id, total_tests = lax.fori_loop(
        0, n_to_test, test_element, (jnp.int32(-1), current_tests)
    )

    return found_elem_id, total_tests


# ============================================================================
# 26-Neighbor Search (Unrolled Static Loop)
# ============================================================================

def search_26_neighbors(
    pos: jax.Array,
    primary_grid_i: jnp.int32,
    primary_grid_j: jnp.int32,
    primary_grid_k: jnp.int32,
    level: jnp.uint8,
    octree_gpu: MeshAlignedOctreeGPU,
    max_tests: jnp.int32,
    current_tests: jnp.int32
) -> Tuple[jnp.int32, jnp.int32]:
    """
    Search 26 spatial neighbors of the primary cell.

    Uses UNROLLED STATIC LOOP to avoid memory explosion in vmap.

    Args:
        pos: (3,) query position
        primary_grid_i/j/k: Primary cell grid indices
        level: Refinement level
        octree_gpu: GPU octree structure
        max_tests: Maximum total tests
        current_tests: Tests already performed

    Returns:
        (elem_id, n_tests): Found element ID (-1 if not found), total tests
    """
    found_elem = jnp.int32(-1)
    total_tests = current_tests

    # Unroll all 27 cells (26 neighbors + center, but we skip center since it's already searched)
    # This creates a STATIC computation graph (no dynamic branching)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                # Skip center cell (already searched in primary)
                if di == 0 and dj == 0 and dk == 0:
                    continue

                # Compute neighbor grid indices
                neighbor_i = primary_grid_i + di
                neighbor_j = primary_grid_j + dj
                neighbor_k = primary_grid_k + dk

                # Find neighbor cell
                neighbor_idx = find_neighbor_cell_at_grid(
                    neighbor_i, neighbor_j, neighbor_k, level, octree_gpu
                )

                # Search elements in neighbor (only if not already found)
                should_search = jnp.logical_and(
                    neighbor_idx >= 0,
                    found_elem < 0  # Not yet found
                )

                # Conditional search (lax.cond is safe here since it's not nested in vmap)
                elem_result, tests_result = lax.cond(
                    should_search,
                    lambda: search_elements_in_cell(pos, neighbor_idx, octree_gpu, max_tests, total_tests),
                    lambda: (found_elem, total_tests)
                )

                # Update state
                found_elem = jnp.where(elem_result >= 0, elem_result, found_elem)
                total_tests = tests_result

    return found_elem, total_tests


# ============================================================================
# Single-Particle Point Location WITH NEIGHBORS
# ============================================================================

def search_mesh_aligned_octree_with_neighbors_single(
    pos: jax.Array,
    octree_gpu: MeshAlignedOctreeGPU,
    max_tests: jnp.int32 = 200
) -> Tuple[jnp.int32, jnp.int32]:
    """
    Find containing element using mesh-aligned octree WITH 26-neighbor search.

    Algorithm:
        1. Find primary cell at finest level (14)
        2. Search primary cell
        3. If not found, search 26 neighbors
        4. If still not found, try coarser levels (13, 12, ...)
        5. Return first containing element

    Args:
        pos: (3,) float32 - query position
        octree_gpu: GPU octree structure
        max_tests: Maximum elements to test (increased from 150 to 200 for neighbors)

    Returns:
        (elem_id, n_tests):
            elem_id: Element ID (-1 if not found)
            n_tests: Number of point-in-tet tests performed
    """

    def try_level_with_neighbors(level_idx, carry):
        """Try searching at one refinement level with neighbor fallback."""
        found_elem_id, total_tests = carry

        # Skip if already found
        already_found = found_elem_id >= 0

        # Level to try (14, 13, 12, 11, 10, 9, 8, 7)
        level = 14 - level_idx

        # Use EXACT cell sizes from mesh for this level
        cell_size = octree_gpu.level_cell_sizes[level]

        # Compute grid indices
        grid_i = jnp.floor(pos[0] / cell_size[0]).astype(jnp.int32)
        grid_j = jnp.floor(pos[1] / cell_size[1]).astype(jnp.int32)
        grid_k = jnp.floor(pos[2] / cell_size[2]).astype(jnp.int32)

        # Find primary cell
        primary_cell_idx = find_neighbor_cell_at_grid(
            grid_i, grid_j, grid_k, jnp.uint8(level), octree_gpu
        )

        # Search primary cell (only if not already found)
        elem_primary, tests_primary = lax.cond(
            jnp.logical_and(jnp.logical_not(already_found), primary_cell_idx >= 0),
            lambda: search_elements_in_cell(pos, primary_cell_idx, octree_gpu, max_tests, total_tests),
            lambda: (found_elem_id, total_tests)
        )

        # Search 26 neighbors (only if not found in primary and not already found)
        need_neighbor_search = jnp.logical_and(
            jnp.logical_not(already_found),
            elem_primary < 0  # Not found in primary
        )

        elem_neighbors, tests_neighbors = lax.cond(
            need_neighbor_search,
            lambda: search_26_neighbors(
                pos, grid_i, grid_j, grid_k, jnp.uint8(level),
                octree_gpu, max_tests, tests_primary
            ),
            lambda: (elem_primary, tests_primary)
        )

        # Return updated state
        return (elem_neighbors, tests_neighbors)

    # Try levels 14, 13, 12, 11, 10, 9, 8, 7 in sequence
    # Stop early if found
    n_levels_to_try = 8
    init_state = (jnp.int32(-1), jnp.int32(0))

    final_elem_id, final_n_tests = lax.fori_loop(
        0, n_levels_to_try, try_level_with_neighbors, init_state
    )

    return final_elem_id, final_n_tests


# ============================================================================
# Batch Point Location WITH NEIGHBORS
# ============================================================================

def search_mesh_aligned_octree_with_neighbors_batch(
    positions: jax.Array,
    octree_gpu: MeshAlignedOctreeGPU,
    max_tests: jnp.int32 = 200
) -> Tuple[jax.Array, jax.Array]:
    """
    Find containing elements for a batch of particles WITH neighbor search.

    Args:
        positions: (n_particles, 3) float32 - query positions
        octree_gpu: GPU octree structure
        max_tests: Maximum elements to test per particle

    Returns:
        (elem_ids, n_tests):
            elem_ids: (n_particles,) int32 - element IDs (-1 if not found)
            n_tests: (n_particles,) int32 - tests performed per particle
    """
    # Vmap over particles
    elem_ids, n_tests = jax.vmap(
        lambda pos: search_mesh_aligned_octree_with_neighbors_single(pos, octree_gpu, max_tests),
        in_axes=0
    )(positions)

    return elem_ids, n_tests
