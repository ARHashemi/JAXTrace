"""
Block-Local GPU Search for Particle Tracking

This module implements block-local fallback search that searches only within
a particle's assigned block, avoiding the need to search the entire mesh.

Key features:
- Pure GPU implementation (no CPU-GPU transfers)
- Searches only elements in particle's block (1-450k elements vs 3.5M global)
- Used as fallback when L1 multi-hop search fails
- Memory-efficient: uses flat element lists instead of padded arrays

Performance:
- Light blocks (240): 2-10k elements → 0.1-0.5 ms per particle
- Heavy blocks (16): 50-450k elements → 5-50 ms per particle
- Average: ~2-5 ms per failed particle (vs 350 ms for global search)
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from dataclasses import dataclass

from jaxtrace.gpu.search.level0_cached import point_in_tet_jax


@dataclass
class BlockElementLists:
    """
    Variable-length element lists per block (GPU-resident).

    Instead of padded arrays (256 × 450k = 461 MB), uses flat list
    with offsets (14 MB = 33× less memory).
    """
    all_elements: jax.Array      # (3.5M,) int32 - all block elements concatenated
    block_offsets: jax.Array     # (n_blocks,) int32 - start index per block
    block_lengths: jax.Array     # (n_blocks,) int32 - element count per block
    max_elements_per_block: int  # Maximum block size (for dynamic slicing)


def build_block_element_lists(blocks: list, n_blocks: int) -> BlockElementLists:
    """
    Build flat element lists from block data.

    Parameters
    ----------
    blocks : list
        List of blocks, each with 'elements' attribute
    n_blocks : int
        Number of blocks

    Returns
    -------
    block_lists : BlockElementLists
        GPU-resident block element lists
    """
    # Build flat list
    all_elements = []
    block_offsets = []
    block_lengths = []

    max_len = 0
    for block in blocks:
        block_offsets.append(len(all_elements))
        block_elems = block.elements  # Assume this is a numpy array
        block_lengths.append(len(block_elems))
        all_elements.extend(block_elems)
        max_len = max(max_len, len(block_elems))

    # Convert to numpy arrays
    all_elements_np = np.array(all_elements, dtype=np.int32)
    block_offsets_np = np.array(block_offsets, dtype=np.int32)
    block_lengths_np = np.array(block_lengths, dtype=np.int32)

    # Upload to GPU
    all_elements_gpu = jax.device_put(all_elements_np)
    block_offsets_gpu = jax.device_put(block_offsets_np)
    block_lengths_gpu = jax.device_put(block_lengths_np)

    return BlockElementLists(
        all_elements=all_elements_gpu,
        block_offsets=block_offsets_gpu,
        block_lengths=block_lengths_gpu,
        max_elements_per_block=max_len
    )


@jax.jit
def search_single_particle_in_block(
    position: jax.Array,           # (3,)
    block_id: jax.Array,           # scalar int32
    all_elements: jax.Array,       # (total_elements,)
    block_offsets: jax.Array,      # (n_blocks,)
    block_lengths: jax.Array,      # (n_blocks,)
    max_block_size: int,           # Maximum elements per block
    node_positions: jax.Array,     # (n_nodes, 3)
    connectivity: jax.Array        # (n_elements, 4)
) -> jax.Array:
    """
    Search for containing element within a single block (GPU-only).

    Parameters
    ----------
    position : jax.Array, shape (3,)
        Particle position
    block_id : jax.Array, scalar
        Block ID to search in
    all_elements : jax.Array
        Flat list of all block elements
    block_offsets : jax.Array
        Start index for each block
    block_lengths : jax.Array
        Element count for each block
    max_block_size : int
        Maximum elements per block (for dynamic slicing)
    node_positions : jax.Array
        Node positions (n_nodes, 3)
    connectivity : jax.Array
        Element connectivity (n_elements, 4)

    Returns
    -------
    element_id : jax.Array, scalar
        Element ID containing particle (-1 if not found)
    """
    # Validate block_id
    n_blocks = len(block_offsets)
    valid_block = (block_id >= 0) & (block_id < n_blocks)
    safe_block_id = jnp.where(valid_block, block_id, 0)

    # Get block info
    start_idx = block_offsets[safe_block_id]
    block_len = block_lengths[safe_block_id]

    # Dynamic slice to get block elements (JAX supports this in JIT)
    # Note: We slice up to max_block_size, then mask based on actual length
    block_elements = jax.lax.dynamic_slice(
        all_elements,
        (start_idx,),
        (max_block_size,)
    )

    # Create mask for valid elements
    valid_mask = jnp.arange(max_block_size) < block_len

    # Check each element in block
    def check_element(elem_idx):
        """Check if particle is in this element."""
        is_valid = valid_mask[elem_idx]
        elem_id = block_elements[elem_idx]

        # Bounds check
        elem_valid = (elem_id >= 0) & (elem_id < len(connectivity))
        safe_elem_id = jnp.where(elem_valid, elem_id, 0)

        # Get element nodes
        node_ids = connectivity[safe_elem_id]
        tet_nodes = node_positions[node_ids]

        # Check if point is inside
        inside = point_in_tet_jax(position, tet_nodes)

        # Return element ID if valid and inside
        return jnp.where(is_valid & elem_valid & inside, elem_id, -1)

    # Vectorize over all elements in block
    found_ids = jax.vmap(check_element)(jnp.arange(max_block_size))

    # Find first match
    found_indices = jnp.where(found_ids >= 0, jnp.arange(max_block_size), max_block_size)
    first_idx = jnp.min(found_indices)
    result = jnp.where(first_idx < max_block_size, found_ids[first_idx], -1)

    # Return -1 if block_id was invalid
    return jnp.where(valid_block, result, -1)


def create_block_local_search_func(block_lists: BlockElementLists):
    """
    Create a JIT-compiled block-local search function.

    Parameters
    ----------
    block_lists : BlockElementLists
        GPU-resident block element lists

    Returns
    -------
    search_func : callable
        JIT-compiled function: (positions, block_ids, node_positions, connectivity) -> element_ids
    """
    # Extract data (these will be captured in closure)
    all_elements = block_lists.all_elements
    block_offsets = block_lists.block_offsets
    block_lengths = block_lists.block_lengths
    max_block_size = int(block_lists.max_elements_per_block)  # Convert to Python int for JIT

    # Create a version of search_single_particle_in_block with max_block_size baked in
    def search_single_particle_in_block_closure(
        position, block_id, all_elements, block_offsets, block_lengths,
        node_positions, connectivity
    ):
        """Closure that captures max_block_size as a compile-time constant."""
        # Validate block_id
        n_blocks = len(block_offsets)
        valid_block = (block_id >= 0) & (block_id < n_blocks)
        safe_block_id = jnp.where(valid_block, block_id, 0)

        # Get block info
        start_idx = block_offsets[safe_block_id]
        block_len = block_lengths[safe_block_id]

        # Dynamic slice - max_block_size is now a Python int (compile-time constant)
        block_elements = jax.lax.dynamic_slice(
            all_elements,
            (start_idx,),
            (max_block_size,)  # This is now a concrete Python int
        )

        # Create mask for valid elements
        valid_mask = jnp.arange(max_block_size) < block_len

        # Check elements sequentially using scan (memory-efficient)
        # This avoids creating massive intermediate arrays
        def scan_elements(carry, elem_idx):
            """Sequential scan over block elements."""
            found_id, found = carry

            # Only check if we haven't found yet
            is_valid = valid_mask[elem_idx]
            elem_id = block_elements[elem_idx]

            # Bounds check
            elem_valid = (elem_id >= 0) & (elem_id < len(connectivity))
            safe_elem_id = jnp.where(elem_valid, elem_id, 0)

            # Get element nodes
            node_ids = connectivity[safe_elem_id]
            tet_nodes = node_positions[node_ids]

            # Check if point is inside
            inside = point_in_tet_jax(position, tet_nodes)

            # Update if found and valid
            should_update = is_valid & elem_valid & inside & ~found
            new_found_id = jnp.where(should_update, elem_id, found_id)
            new_found = found | should_update

            return (new_found_id, new_found), None

        # Sequential scan (memory-efficient, no huge intermediate arrays)
        (result, _), _ = jax.lax.scan(
            scan_elements,
            (-1, False),  # Initial: (found_id=-1, found=False)
            jnp.arange(max_block_size)
        )

        # Return -1 if block_id was invalid
        return jnp.where(valid_block, result, -1)

    @jax.jit
    def search_batch_in_blocks(
        positions: jax.Array,      # (N, 3)
        block_ids: jax.Array,      # (N,)
        node_positions: jax.Array, # (n_nodes, 3)
        connectivity: jax.Array    # (n_elements, 4)
    ) -> jax.Array:
        """
        Search for containing elements in each particle's block.

        Pure GPU implementation, no CPU-GPU transfers.

        Parameters
        ----------
        positions : jax.Array, shape (N, 3)
            Particle positions
        block_ids : jax.Array, shape (N,)
            Block ID for each particle
        node_positions : jax.Array
            Node positions
        connectivity : jax.Array
            Element connectivity

        Returns
        -------
        element_ids : jax.Array, shape (N,)
            Element IDs (-1 if not found)
        """
        # Vectorize search over all particles
        def search_one(pos, block_id):
            return search_single_particle_in_block_closure(
                pos, block_id,
                all_elements, block_offsets, block_lengths,
                node_positions, connectivity
            )

        return jax.vmap(search_one)(positions, block_ids)

    return search_batch_in_blocks


@jax.jit
def search_global_gpu_native_scan(
    positions: jax.Array,      # (N, 3)
    search_mask: jax.Array,    # (N,) bool - which particles to search
    node_positions: jax.Array, # (n_nodes, 3)
    connectivity: jax.Array    # (n_elements, 4)
) -> jax.Array:
    """
    GPU-native global search using scan over particles.

    Avoids CPU loop while preventing memory explosion from nested vmap.
    Uses scan to iterate over particles sequentially, with each particle
    checking all elements in parallel. Uses masking to skip particles
    that don't need searching.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Particle positions to search
    search_mask : jax.Array, shape (N,)
        Boolean mask indicating which particles to search (True = search)
    node_positions : jax.Array, shape (n_nodes, 3)
        Node coordinates (GPU-resident)
    connectivity : jax.Array, shape (n_elements, 4)
        Element connectivity (GPU-resident)

    Returns
    -------
    element_ids : jax.Array, shape (N,)
        Element IDs for each particle (-1 if not found or not searched)

    Notes
    -----
    Memory efficiency:
    - Scan over N particles (no nested vmap)
    - Each particle vmaps over 3.5M elements (3.5 MB per particle)
    - Skips particles where search_mask is False (early exit)
    - Total: N × 3.5 MB (sequential, not materialized at once)
    - For 100 failed particles: searches all but only ~100 need results
    """
    n_elements = len(connectivity)

    def check_element(position, elem_id):
        """Check if particle is in this element."""
        node_ids = connectivity[elem_id]
        tet_nodes = node_positions[node_ids]
        return point_in_tet_jax(position, tet_nodes)

    def search_one_particle(carry, position_and_mask):
        """Search for containing element for one particle."""
        position, should_search = position_and_mask

        # Skip search if mask is False (particle already found)
        def do_search(_):
            # Vmap over all elements for THIS particle
            inside_mask = jax.vmap(lambda e: check_element(position, e))(
                jnp.arange(n_elements)
            )
            # Find first containing element
            first_hit = jnp.argmax(inside_mask)
            return jnp.where(inside_mask[first_hit], first_hit, -1)

        def skip_search(_):
            return -1

        # Use lax.cond to avoid expensive search for particles that hit L1
        elem_id = jax.lax.cond(
            should_search,
            do_search,
            skip_search,
            None
        )

        return carry, elem_id

    # Scan over particles (sequential, memory-efficient)
    _, element_ids = jax.lax.scan(
        search_one_particle,
        None,  # No carry state needed
        (positions, search_mask)
    )

    return element_ids


def create_search_with_block_fallback(n_hops: int = 3, block_lists: BlockElementLists = None):
    """
    Create search function with L1 multi-hop + global GPU fallback.

    NOTE: Block-local fallback causes OOM (218 GB) for 100k particles due to
    nested vmap/scan. Replaced with GPU-native global search using scan over
    particles to keep GPU busy while avoiding memory explosion.

    Parameters
    ----------
    n_hops : int, default=3
        Number of hops for L1 search
    block_lists : BlockElementLists, optional
        Block element lists for fallback. CURRENTLY UNUSED (reserved for future).
        If None, no fallback is used.

    Returns
    -------
    search_func : callable
        JIT-compiled search function with GPU-native global fallback
    """
    from jaxtrace.gpu.search.incremental_search_vectorized import (
        search_level1_multihop_vectorized
    )

    # DISABLED: Block-local search causes OOM for large particle counts
    # Preserved for future use when memory issue is resolved
    # if block_lists is not None:
    #     block_search_func = create_block_local_search_func(block_lists)
    # else:
    #     block_search_func = None
    block_search_func = None  # Force disabled

    @jax.jit
    def search_with_fallback(
        positions_gpu: jax.Array,         # (N, 3)
        cached_element_ids_gpu: jax.Array, # (N,)
        block_ids_gpu: jax.Array,         # (N,) - NOT USED (reserved for block search)
        node_positions_gpu: jax.Array,
        connectivity_gpu: jax.Array,
        element_neighbors_gpu: jax.Array
    ) -> jax.Array:
        """
        Two-tier search: L1 multi-hop → GPU-native global fallback.

        Pure GPU, no CPU-GPU transfers, no CPU loops.

        Parameters
        ----------
        positions_gpu : jax.Array, shape (N, 3)
            Particle positions
        cached_element_ids_gpu : jax.Array, shape (N,)
            Cached element IDs from previous timestep
        block_ids_gpu : jax.Array, shape (N,)
            Block ID for each particle (NOT USED, reserved for future)
        node_positions_gpu : jax.Array
            Node positions
        connectivity_gpu : jax.Array
            Element connectivity
        element_neighbors_gpu : jax.Array
            Element neighbor connectivity

        Returns
        -------
        element_ids : jax.Array, shape (N,)
            Updated element IDs
        """
        # Tier 1: L1 multi-hop search (fast, 99.9% success)
        element_ids = search_level1_multihop_vectorized(
            positions_gpu,
            cached_element_ids_gpu,
            element_neighbors_gpu,
            node_positions_gpu,
            connectivity_gpu,
            n_hops=n_hops
        )

        # Tier 2: GPU-native global fallback (only for L1 failures)
        # Uses scan over particles to keep GPU busy without memory explosion
        failed_mask = element_ids < 0

        # Run global search with masking (only searches failed particles)
        # Uses lax.cond inside scan to skip particles where L1 succeeded
        # This is JIT-compatible and avoids dynamic shape issues
        global_results = search_global_gpu_native_scan(
            positions_gpu,
            failed_mask,  # Only search where L1 failed
            node_positions_gpu,
            connectivity_gpu
        )

        # Update element IDs only where L1 failed and global search succeeded
        element_ids = jnp.where(failed_mask & (global_results >= 0), global_results, element_ids)

        return element_ids

    return search_with_fallback
