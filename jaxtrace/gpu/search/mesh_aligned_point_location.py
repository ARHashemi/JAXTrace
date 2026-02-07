"""
Phase 4: Mesh-Aligned Octree Point Location

JAX GPU kernels for fast point-in-element search using mesh-aligned octree.

Architecture:
    1. Position → Try multiple refinement levels
    2. For each level: Grid indices → Morton code → Binary search
    3. Test all elements in found cell
    4. Return first containing element

Expected Performance:
    - ~37 point-in-tet tests per query (vs ~536 in Morton blocks)
    - ~144× reduction in tests
    - Near-perfect searchability (100% from Phase 1 validation)
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
    find_cell_by_morton,
    find_cell_by_morton_and_level,
    get_cell_elements,
)


# ============================================================================
# Helper: Compute cell size for a given octree level
# ============================================================================

def level_to_cell_size(level: jnp.int32, base_size: jnp.float32 = 1.0) -> jnp.float32:
    """
    Convert octree level to cell size.

    Args:
        level: Octree refinement level (higher = finer)
        base_size: Size at level 0

    Returns:
        Cell size at this level
    """
    # cell_size = base_size / 2^level
    return base_size / (2.0 ** level)


# ============================================================================
# Single-Particle Point Location
# ============================================================================

def search_mesh_aligned_octree_single(
    pos: jax.Array,
    octree_gpu: MeshAlignedOctreeGPU,
    max_tests: jnp.int32 = 150
) -> Tuple[jnp.int32, jnp.int32]:
    """
    Find containing element using mesh-aligned octree (single particle).

    Simple multi-level search (center cell only):
    - For RK4 integration, assumes particles move incrementally between steps
    - L1 neighbor search should handle most adjacent-cell cases
    - This L2 search only needs center cell lookup (no neighbor search)
    - Neighbor search caused 631 GB memory allocation (nested lax.cond in vmap)

    Args:
        pos: (3,) float32 - query position
        octree_gpu: GPU octree structure
        max_tests: Maximum elements to test (safety bound)

    Returns:
        (elem_id, n_tests):
            elem_id: Element ID (-1 if not found)
            n_tests: Number of point-in-tet tests performed
    """

    def try_level(level_idx, carry):
        """Try searching center cell at one refinement level."""
        found_elem_id, total_tests = carry

        # Skip if already found
        already_found = found_elem_id >= 0

        # Level to try (14, 13, 12, 11, 10, 9, 8, 7)
        level = 14 - level_idx

        # CRITICAL: Use EXACT cell sizes from mesh for each level
        cell_size = octree_gpu.level_cell_sizes[level]

        # Grid indices (floor division)
        i = jnp.floor(pos[0] / cell_size[0]).astype(jnp.int32)
        j = jnp.floor(pos[1] / cell_size[1]).astype(jnp.int32)
        k = jnp.floor(pos[2] / cell_size[2]).astype(jnp.int32)

        # Apply offset for negative coordinates
        i_offset = jnp.clip(i + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)
        j_offset = jnp.clip(j + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)
        k_offset = jnp.clip(k + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)

        # Encode to Morton code
        morton_code = encode_morton_3d_jax(i_offset, j_offset, k_offset)

        # Binary search for cell with BOTH morton and level
        cell_idx = find_cell_by_morton_and_level(
            morton_code,
            jnp.uint8(level),
            octree_gpu.cell_morton_codes,
            octree_gpu.cell_levels
        )

        # Get elements in this cell
        start_idx, n_elements_in_cell = get_cell_elements(
            cell_idx,
            octree_gpu.cell_to_elements_offsets,
            octree_gpu.cell_to_elements_data
        )

        # Only proceed if cell exists and we haven't found yet
        should_search = jnp.logical_and(cell_idx >= 0, jnp.logical_not(already_found))
        tests_remaining = max_tests - total_tests
        n_to_test = jnp.where(
            should_search,
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
        level_found, level_tests = lax.fori_loop(
            0, n_to_test, test_element, (found_elem_id, total_tests)
        )

        return (level_found, level_tests)

    # Try levels 14, 13, 12, 11, 10, 9, 8, 7 in sequence
    # Stop early if found
    n_levels_to_try = 8
    init_state = (jnp.int32(-1), jnp.int32(0))

    final_elem_id, final_n_tests = lax.fori_loop(
        0, n_levels_to_try, try_level, init_state
    )

    return final_elem_id, final_n_tests


def search_mesh_aligned_octree_multi_local(
    pos: jax.Array,
    octree_gpu: MeshAlignedOctreeGPU,
    max_tests: jnp.int32 = 200
) -> Tuple[jnp.int32, jnp.int32]:
    """
    Find containing element using 2×2×2 local neighborhood search.

    For multi-cell vertex registration where each element is registered in ~4 cells
    (where its vertices are located), we need to search a local neighborhood of cells
    to find the element.

    This function searches 8 cells (2×2×2 cube) centered around the particle position.
    This covers all possible cells where the element's vertices could be registered.

    Algorithm:
        1. For each refinement level:
            a. Compute base cell indices (i, j, k) for particle position
            b. Search 8 cells: (i+di, j+dj, k+dk) for di,dj,dk in [0,1]
            c. Test elements in each cell
            d. Return first containing element found
        2. Try levels from finest to coarsest

    Args:
        pos: (3,) float32 - query position
        octree_gpu: GPU octree structure (multi-cell vertex registration)
        max_tests: Maximum elements to test across all cells

    Returns:
        (elem_id, n_tests):
            elem_id: Element ID (-1 if not found)
            n_tests: Total number of point-in-tet tests
    """

    def try_level(level_idx, carry):
        """Try searching 2×2×2 neighborhood at one refinement level."""
        found_elem, total_tests = carry

        # Skip if already found
        def skip_level():
            return found_elem, total_tests

        def search_level():
            # Level to try (14, 13, 12, 11, 10, 9, 8, 7)
            level = 14 - level_idx

            # CRITICAL: Use EXACT cell sizes from mesh for each level
            cell_size = octree_gpu.level_cell_sizes[level]

            # Compute base cell indices (floor division)
            i_base = jnp.floor(pos[0] / cell_size[0]).astype(jnp.int32)
            j_base = jnp.floor(pos[1] / cell_size[1]).astype(jnp.int32)
            k_base = jnp.floor(pos[2] / cell_size[2]).astype(jnp.int32)

            # Search 2×2×2 = 8 cells
            def try_cell(cell_offset, inner_carry):
                """Try searching one cell in the 2×2×2 neighborhood."""
                inner_found_elem, inner_tests = inner_carry
                di, dj, dk = cell_offset[0], cell_offset[1], cell_offset[2]

                # Skip if already found
                def skip_cell():
                    return inner_found_elem, inner_tests

                def search_cell():
                    # Compute cell indices
                    i = i_base + di
                    j = j_base + dj
                    k = k_base + dk

                    # Apply offset for negative coordinates
                    i_offset = jnp.clip(i + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)
                    j_offset = jnp.clip(j + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)
                    k_offset = jnp.clip(k + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)

                    # Encode to Morton code
                    morton_code = encode_morton_3d_jax(i_offset, j_offset, k_offset)

                    # Binary search for cell with BOTH morton and level
                    cell_idx = find_cell_by_morton_and_level(
                        morton_code,
                        jnp.uint8(level),
                        octree_gpu.cell_morton_codes,
                        octree_gpu.cell_levels
                    )

                    # If cell doesn't exist, return not found
                    def cell_not_found():
                        return inner_found_elem, inner_tests

                    def cell_found():
                        # Get elements in this cell
                        elem_start_idx = octree_gpu.cell_to_elements_offsets[cell_idx]
                        elem_end_idx = octree_gpu.cell_to_elements_offsets[cell_idx + 1]
                        n_elems_in_cell = elem_end_idx - elem_start_idx

                        # Test each element
                        def test_element(elem_offset, test_carry):
                            test_found_elem, test_n_tests = test_carry

                            # Skip if already found
                            def skip_elem():
                                return test_found_elem, test_n_tests

                            def test_elem():
                                elem_idx_in_data = elem_start_idx + elem_offset
                                elem_id = octree_gpu.cell_to_elements_data[elem_idx_in_data]

                                # Point-in-tet test (uses config-based dispatcher)
                                is_inside = point_in_tet_dispatcher(
                                    pos,
                                    elem_id,
                                    octree_gpu.connectivity,
                                    octree_gpu.node_positions,
                                    config.POINT_IN_TET_METHOD
                                )

                                new_found = jnp.where(is_inside, elem_id, test_found_elem)
                                new_tests = test_n_tests + 1

                                return new_found, new_tests

                            return jax.lax.cond(
                                test_found_elem >= 0,  # Already found?
                                skip_elem,
                                test_elem
                            )

                        # Scan over elements in cell
                        cell_found_elem, cell_tests = jax.lax.fori_loop(
                            0, n_elems_in_cell,
                            test_element,
                            (inner_found_elem, inner_tests)
                        )

                        return cell_found_elem, cell_tests

                    return jax.lax.cond(
                        cell_idx < 0,  # Cell not found?
                        cell_not_found,
                        cell_found
                    )

                return jax.lax.cond(
                    inner_found_elem >= 0,  # Already found?
                    skip_cell,
                    search_cell
                )

            # Define 8 cell offsets for 2×2×2 neighborhood CENTERED on particle
            # Since Kuhn element vertices are at cube corners, and particle can be
            # anywhere in the containing element, we search cells in both negative
            # and positive directions. The particle's base cell (i,j,k) is computed
            # as floor(pos/cell_size), so we search the 2×2×2 cube centered at
            # (i-0.5, j-0.5, k-0.5), which covers cells from (i-1,j-1,k-1) to (i,j,k).
            cell_offsets = jnp.array([
                [-1, -1, -1],
                [-1, -1,  0],
                [-1,  0, -1],
                [-1,  0,  0],
                [ 0, -1, -1],
                [ 0, -1,  0],
                [ 0,  0, -1],
                [ 0,  0,  0],
            ], dtype=jnp.int32)

            # Scan over 8 cells
            level_found_elem, level_tests = jax.lax.fori_loop(
                0, 8,
                lambda i, c: try_cell(cell_offsets[i], c),
                (found_elem, total_tests)
            )

            return level_found_elem, level_tests

        return jax.lax.cond(
            found_elem >= 0,  # Already found?
            skip_level,
            search_level
        )

    # Try refinement levels 14, 13, 12, 11, 10, 9, 8, 7 (8 levels total)
    n_levels = 8
    final_elem_id, final_n_tests = jax.lax.fori_loop(
        0, n_levels,
        try_level,
        (jnp.int32(-1), jnp.int32(0))
    )

    return final_elem_id, final_n_tests


# ============================================================================
# Batch Point Location
# ============================================================================

def search_mesh_aligned_octree_batch(
    positions: jax.Array,
    octree_gpu: MeshAlignedOctreeGPU,
    max_tests: jnp.int32 = 150
) -> Tuple[jax.Array, jax.Array]:
    """
    Find containing elements for a batch of particles.

    This vmaps the single-particle kernel over the batch.

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
        lambda pos: search_mesh_aligned_octree_single(pos, octree_gpu, max_tests),
        in_axes=0
    )(positions)

    return elem_ids, n_tests


def search_mesh_aligned_octree_multi_local_batch(
    positions: jax.Array,
    octree_gpu: MeshAlignedOctreeGPU,
    max_tests: jnp.int32 = 200
) -> Tuple[jax.Array, jax.Array]:
    """
    Find containing elements for a batch of particles using 2×2×2 local search.

    This vmaps the multi-local single-particle kernel over the batch.

    For multi-cell vertex registration where each element is in ~4 cells.

    Args:
        positions: (n_particles, 3) float32 - query positions
        octree_gpu: GPU octree structure (multi-cell vertex registration)
        max_tests: Maximum elements to test per particle

    Returns:
        (elem_ids, n_tests):
            elem_ids: (n_particles,) int32 - element IDs (-1 if not found)
            n_tests: (n_particles,) int32 - tests performed per particle
    """
    # Vmap over particles
    elem_ids, n_tests = jax.vmap(
        lambda pos: search_mesh_aligned_octree_multi_local(pos, octree_gpu, max_tests),
        in_axes=0
    )(positions)

    return elem_ids, n_tests


# ============================================================================
# Multi-Level Cell Search (for handling multiple refinement levels)
# ============================================================================

def search_mesh_aligned_multi_level(
    pos: jax.Array,
    octree_gpu: MeshAlignedOctreeGPU,
    max_levels_to_try: jnp.int32 = 6,
    max_tests: jnp.int32 = 150
) -> Tuple[jnp.int32, jnp.int32]:
    """
    Search with fallback to different refinement levels.

    This is an alias for search_mesh_aligned_octree_single, which already
    implements multi-level search.

    Args:
        pos: (3,) float32 - query position
        octree_gpu: GPU octree structure
        max_levels_to_try: Number of levels to try (up to 6)
        max_tests: Maximum tests total

    Returns:
        (elem_id, n_tests):
            elem_id: Element ID (-1 if not found)
            n_tests: Total number of point-in-tet tests
    """
    return search_mesh_aligned_octree_single(pos, octree_gpu, max_tests)


# ============================================================================
# Statistics and Monitoring
# ============================================================================

def compute_search_statistics(
    elem_ids: jax.Array,
    n_tests: jax.Array
) -> dict:
    """
    Compute search performance statistics.

    Args:
        elem_ids: (n_particles,) int32 - element IDs
        n_tests: (n_particles,) int32 - tests per particle

    Returns:
        stats: Dictionary with performance metrics
    """
    n_particles = elem_ids.shape[0]
    n_found = jnp.sum(elem_ids >= 0)
    success_rate = n_found / n_particles

    mean_tests = jnp.mean(n_tests)
    median_tests = jnp.median(n_tests)
    max_tests = jnp.max(n_tests)

    stats = {
        'n_particles': int(n_particles),
        'n_found': int(n_found),
        'success_rate': float(success_rate),
        'mean_tests': float(mean_tests),
        'median_tests': float(median_tests),
        'max_tests': int(max_tests),
    }

    return stats


def print_search_statistics(stats: dict, label: str = "Mesh-Aligned Octree"):
    """Print search statistics in human-readable format."""
    print(f"\n{label} Search Statistics:")
    print(f"{'='*60}")
    print(f"  Particles searched: {stats['n_particles']:,}")
    print(f"  Found: {stats['n_found']:,} ({stats['success_rate']*100:.2f}%)")
    print(f"  Point-in-tet tests:")
    print(f"    Mean: {stats['mean_tests']:.1f}")
    print(f"    Median: {stats['median_tests']:.0f}")
    print(f"    Max: {stats['max_tests']:,}")
    print(f"{'='*60}")


# ============================================================================
# JIT-Compiled Versions
# ============================================================================

# Pre-compile for common use cases
search_mesh_aligned_octree_single_jit = jax.jit(
    search_mesh_aligned_octree_single,
    static_argnames=('max_tests',)
)

search_mesh_aligned_octree_batch_jit = jax.jit(
    search_mesh_aligned_octree_batch,
    static_argnames=('max_tests',)
)

search_mesh_aligned_octree_multi_local_jit = jax.jit(
    search_mesh_aligned_octree_multi_local,
    static_argnames=('max_tests',)
)

search_mesh_aligned_octree_multi_local_batch_jit = jax.jit(
    search_mesh_aligned_octree_multi_local_batch,
    static_argnames=('max_tests',)
)
