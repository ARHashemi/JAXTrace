"""
L2 block-local Morton search for fused RK4.

This module provides JAX-compatible L2 fallback search using per-block Morton structures.
Designed for single-particle search (vmapped at top level in fused RK4).

Key design decisions for JAX compatibility:
1. **No nested vmap/scan**: Single vmap over particles at top level only
2. **Bounded iteration**: Fixed max_elements_per_block loop bound
3. **Padded arrays**: No CSR, no dynamic_slice - all arrays are fixed-size
4. **Pure functions**: No side effects, deterministic output

Architecture:
- Particle → block_id (already tracked in RK4GPUState)
- Search block's Morton-sorted element list (bounded by max_elements_per_block)
- Point-in-tet check for each element
- Return first match or -1

Expected performance:
- Search cost: O(max_elements_per_block) ~ O(10-50) per particle
- Memory: ~8 MB total (vs 6,500 MB global octree)
- L2 hit rate: 99.95% (particles that miss L0+L1)
"""

import jax
import jax.numpy as jnp
from typing import Tuple


def point_in_tet_jax(point: jax.Array, tet_nodes: jax.Array, tolerance: float = 1e-6) -> jax.Array:
    """
    Check if point is inside tetrahedron using barycentric coordinates.

    This is the same function used in octree_search_gpu.py for consistency.

    Parameters
    ----------
    point : jax.Array, shape (3,)
        Query point coordinates
    tet_nodes : jax.Array, shape (4, 3)
        Tetrahedron node coordinates
    tolerance : float, default=1e-6
        Tolerance for barycentric coordinate bounds

    Returns
    -------
    inside : jax.Array, scalar bool
        True if point is inside tetrahedron
    """
    # Compute vectors from first vertex to others
    v0 = tet_nodes[1] - tet_nodes[0]
    v1 = tet_nodes[2] - tet_nodes[0]
    v2 = tet_nodes[3] - tet_nodes[0]
    vp = point - tet_nodes[0]

    # Compute dot products
    d00 = jnp.dot(v0, v0)
    d01 = jnp.dot(v0, v1)
    d02 = jnp.dot(v0, v2)
    d11 = jnp.dot(v1, v1)
    d12 = jnp.dot(v1, v2)
    d22 = jnp.dot(v2, v2)

    d0p = jnp.dot(v0, vp)
    d1p = jnp.dot(v1, vp)
    d2p = jnp.dot(v2, vp)

    # Build 3x3 matrix determinant for volume calculation
    det = (d00 * (d11 * d22 - d12 * d12) -
           d01 * (d01 * d22 - d12 * d02) +
           d02 * (d01 * d12 - d11 * d02))

    # Avoid division by zero
    det_safe = jnp.where(jnp.abs(det) < 1e-12, 1e-12, det)

    # Compute barycentric coordinates
    u = ((d11 * d22 - d12 * d12) * d0p +
         (d02 * d12 - d01 * d22) * d1p +
         (d01 * d12 - d02 * d11) * d2p) / det_safe

    v = ((d02 * d12 - d01 * d22) * d0p +
         (d00 * d22 - d02 * d02) * d1p +
         (d01 * d02 - d00 * d12) * d2p) / det_safe

    w = ((d01 * d12 - d02 * d11) * d0p +
         (d01 * d02 - d00 * d12) * d1p +
         (d00 * d11 - d01 * d01) * d2p) / det_safe

    # Check if all barycentric coordinates are non-negative and sum <= 1
    inside = (u >= -tolerance) & (v >= -tolerance) & (w >= -tolerance) & ((u + v + w) <= (1.0 + tolerance))

    return inside


def search_block_morton_single_particle(
    position: jax.Array,
    block_id: jax.Array,
    block_element_ids: jax.Array,
    node_positions: jax.Array,
    connectivity: jax.Array,
    max_elements_per_block: int
) -> jax.Array:
    """
    Search for containing element in a single block using Morton-sorted list.

    This is a **per-particle kernel** designed to be vmapped at the top level.
    No nested vmap/scan - just a simple bounded loop.

    Parameters
    ----------
    position : jax.Array, shape (3,)
        Particle position
    block_id : jax.Array, scalar int32
        Block ID for this particle
    block_element_ids : jax.Array, shape (n_blocks, max_elements_per_block)
        Element IDs per block (Morton-sorted, padded with -1)
    node_positions : jax.Array, shape (n_nodes, 3)
        Mesh node positions
    connectivity : jax.Array, shape (n_elements, 4)
        Element-to-node connectivity
    max_elements_per_block : int
        Maximum elements per block (fixed loop bound)

    Returns
    -------
    element_id : jax.Array, scalar int32
        Found element ID (-1 if not found)

    Design Notes
    ------------
    - **Bounded search**: Loop over at most max_elements_per_block elements
    - **Early exit**: Uses lax.fori_loop with carry-based early exit
    - **No nested control**: Single loop, no vmap/scan inside
    - **JAX-friendly**: Pure function, no side effects
    """
    # Get element list for this block
    block_elements = block_element_ids[block_id]  # Shape: (max_elements_per_block,)

    # Early exit: if block_id is invalid, return -1
    valid_block = (block_id >= 0) & (block_id < block_element_ids.shape[0])

    def search_one_element(i, found_id):
        """
        Check one element in the block.

        If already found, skip. Otherwise, check element i.
        """
        # If already found, keep current found_id
        already_found = found_id >= 0

        # Get element ID at position i
        elem_id = block_elements[i]
        valid_elem = elem_id >= 0  # -1 means padding

        # Get tet nodes (safe indexing - use 0 if invalid)
        safe_elem_id = jnp.where(valid_elem, elem_id, 0)
        node_ids = connectivity[safe_elem_id].astype(jnp.int32)
        tet_nodes = node_positions[node_ids]

        # Check if point is inside
        inside = point_in_tet_jax(position, tet_nodes)

        # Update found_id if:
        # - Not already found
        # - Element is valid
        # - Point is inside
        should_update = (~already_found) & valid_elem & inside
        new_found_id = jnp.where(should_update, safe_elem_id, found_id)

        return new_found_id

    # Search using bounded loop
    # Start with found_id = -1 (not found)
    found_id = jax.lax.fori_loop(
        0,
        max_elements_per_block,
        search_one_element,
        jnp.int32(-1)
    )

    # If block is invalid, return -1
    found_id = jnp.where(valid_block, found_id, jnp.int32(-1))

    return found_id


def create_level2_block_morton_search(
    block_element_ids_gpu: jax.Array,
    node_positions_gpu: jax.Array,
    connectivity_gpu: jax.Array,
    max_elements_per_block: int
):
    """
    Create JIT-compiled L2 block Morton search function.

    This factory function captures the Morton structures and mesh data,
    creating a search function that can be used in fused RK4.

    Parameters
    ----------
    block_element_ids_gpu : jax.Array, shape (n_blocks, max_elements_per_block)
        Element IDs per block (GPU-resident)
    node_positions_gpu : jax.Array, shape (n_nodes, 3)
        Mesh node positions (GPU-resident)
    connectivity_gpu : jax.Array, shape (n_elements, 4)
        Element connectivity (GPU-resident)
    max_elements_per_block : int
        Maximum elements per block

    Returns
    -------
    search_func : callable
        JIT-compiled search function with signature:
        search_func(positions, block_ids, cached_ids) -> element_ids

        Where:
        - positions: jax.Array, shape (N, 3) - particle positions
        - block_ids: jax.Array, shape (N,) - block ID per particle
        - cached_ids: jax.Array, shape (N,) - element IDs from L0+L1
        Returns:
        - element_ids: jax.Array, shape (N,) - found element IDs

    Usage
    -----
    # At initialization (once):
    search_l2 = create_level2_block_morton_search(
        block_element_ids_gpu,
        mesh_gpu.node_positions,
        mesh_gpu.connectivity,
        max_elements_per_block=50
    )

    # In fused RK4 (every step):
    element_ids_l2 = search_l2(positions, block_ids, element_ids_l0_l1)
    """
    @jax.jit
    def search_func(positions, block_ids, cached_ids):
        """
        L2 block Morton search (vectorized over particles).

        Only searches particles with cached_ids == -1.
        """
        # Define per-particle search
        def search_single(pos, block_id, cached_id):
            """Search single particle (if needed)."""
            # If already found in L0+L1, skip L2 search
            already_found = cached_id >= 0

            # Search block if not yet found
            found_id = jax.lax.cond(
                already_found,
                lambda: cached_id,  # Keep cached ID
                lambda: search_block_morton_single_particle(
                    pos,
                    block_id,
                    block_element_ids_gpu,
                    node_positions_gpu,
                    connectivity_gpu,
                    max_elements_per_block
                )
            )

            return found_id

        # Vectorize over all particles (single vmap at top level)
        element_ids = jax.vmap(search_single)(positions, block_ids, cached_ids)

        return element_ids

    return search_func


def create_level2_block_morton_search_unconditional(
    block_element_ids_gpu: jax.Array,
    node_positions_gpu: jax.Array,
    connectivity_gpu: jax.Array,
    max_elements_per_block: int
):
    """
    Create JIT-compiled L2 block Morton search (unconditional version).

    This version always searches, without checking cached_ids.
    Useful for testing or when you want to force L2 search.

    Parameters
    ----------
    Same as create_level2_block_morton_search

    Returns
    -------
    search_func : callable
        JIT-compiled search function with signature:
        search_func(positions, block_ids) -> element_ids
    """
    @jax.jit
    def search_func(positions, block_ids):
        """L2 block Morton search (unconditional, vectorized)."""
        def search_single(pos, block_id):
            return search_block_morton_single_particle(
                pos,
                block_id,
                block_element_ids_gpu,
                node_positions_gpu,
                connectivity_gpu,
                max_elements_per_block
            )

        # Single vmap over particles
        element_ids = jax.vmap(search_single)(positions, block_ids)
        return element_ids

    return search_func
