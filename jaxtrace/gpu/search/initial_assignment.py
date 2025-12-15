"""
Initial Particle Assignment Using GPU Multi-Level Search

Uses Phase 4's L2 (block) and L3 (neighbor blocks) search levels for fast
initial particle-to-element assignment, replacing the slow CPU baseline search.

Search Strategy:
    1. Find containing block (O(1) arithmetic)
    2. L2: Search within block
       - L2a: Light blocks (<10K) - direct search
       - L2b: Heavy blocks (≥10K) - hash bucket search
    3. L3: Fallback to 26-neighbor blocks if not found

Performance: Expected 1,000-5,000 particles/s (5-25× faster than CPU baseline)

Note: Skips L0 (cached) and L1 (neighbors) as particles have no cache yet.
"""

import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import time

from .level2a_light import search_level2a_light_block
from .level2b_heavy import search_level2b_hash_bucket
from .level3_neighbor_blocks import search_level3_neighbor_blocks
from .block_classifier import BlockClassification
from .hash_bucket import HashBucketArrays
from ..forest.padded_arrays import PaddedArrays

jax.config.update("jax_enable_x64", True)


@dataclass
class InitialSearchStats:
    """Statistics from initial GPU search."""
    n_particles: int
    n_found: int
    n_not_found: int
    n_found_in_primary_block: int
    n_found_in_neighbor_blocks: int
    l2_hits: int  # Found in L2 (block search)
    l3_hits: int  # Found in L3 (neighbor blocks)
    total_search_time: float
    particles_per_second: float

    def __repr__(self) -> str:
        return (
            f"InitialSearchStats(\n"
            f"  Particles: {self.n_particles:,}\n"
            f"  Found: {self.n_found:,} ({100*self.n_found/self.n_particles:.1f}%)\n"
            f"  - In primary block: {self.n_found_in_primary_block:,} ({100*self.n_found_in_primary_block/self.n_particles:.1f}%)\n"
            f"  - In neighbor blocks: {self.n_found_in_neighbor_blocks:,} ({100*self.n_found_in_neighbor_blocks/self.n_particles:.1f}%)\n"
            f"  Not found: {self.n_not_found:,} ({100*self.n_not_found/self.n_particles:.1f}%)\n"
            f"  Time: {self.total_search_time:.2f} s\n"
            f"  Rate: {self.particles_per_second:.0f} particles/s\n"
            f")"
        )


@jax.jit
def find_containing_block_jax(
    position: jax.Array,
    domain_bounds: jax.Array,
    grid_size: Tuple[int, int, int]
) -> int:
    """
    Find which block contains a particle position (JAX version).

    Fast O(1) mapping from position to block ID using arithmetic.

    Parameters
    ----------
    position : jax.Array
        Particle position [x, y, z], shape (3,)
    domain_bounds : jax.Array
        Domain bounds [xmin, xmax, ymin, ymax, zmin, zmax], shape (6,)
    grid_size : Tuple[int, int, int]
        Grid dimensions (nx, ny, nz)

    Returns
    -------
    block_id : int
        Block ID (0 to n_blocks-1) or -1 if outside domain

    Performance
    -----------
    O(1) constant time arithmetic
    """
    xmin, xmax, ymin, ymax, zmin, zmax = domain_bounds
    nx, ny, nz = grid_size

    x, y, z = position[0], position[1], position[2]

    # Check if outside domain
    outside = (x < xmin) | (x > xmax) | (y < ymin) | (y > ymax) | (z < zmin) | (z > zmax)

    # Compute block size
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    dz = (zmax - zmin) / nz

    # Compute grid indices
    i = jnp.floor((x - xmin) / dx).astype(jnp.int32)
    j = jnp.floor((y - ymin) / dy).astype(jnp.int32)
    k = jnp.floor((z - zmin) / dz).astype(jnp.int32)

    # Clamp to valid range [0, n-1]
    i = jnp.clip(i, 0, nx - 1)
    j = jnp.clip(j, 0, ny - 1)
    k = jnp.clip(k, 0, nz - 1)

    # Convert to block ID: block_id = i + j*nx + k*nx*ny
    block_id = i + j * nx + k * nx * ny

    # Return -1 if outside domain, else block_id
    return jnp.where(outside, -1, block_id)


def initial_search_single(
    position: np.ndarray,
    domain_bounds: np.ndarray,
    grid_size: Tuple[int, int, int],
    block_classification: BlockClassification,
    padded_arrays: PaddedArrays,
    block_neighbors_26: np.ndarray,
    hash_bucket_data: Dict[int, HashBucketArrays],
    node_positions: np.ndarray,
    connectivity: np.ndarray
) -> Tuple[int, int]:
    """
    Find containing element for a single particle (initial assignment).

    Uses L2 (block search) + L3 (neighbor blocks) from Phase 4.
    Skips L0 (cached) and L1 (neighbors) as particle has no cache yet.

    Parameters
    ----------
    position : np.ndarray
        Particle position [x, y, z], shape (3,)
    domain_bounds : np.ndarray
        Domain bounds [xmin, xmax, ymin, ymax, zmin, zmax]
    grid_size : Tuple[int, int, int]
        Grid dimensions (nx, ny, nz)
    block_classification : BlockClassification
        Light/heavy block categorization
    padded_arrays : PaddedArrays
        Block-local element storage from Phase 2
    block_neighbors_26 : np.ndarray
        26-neighbor connectivity, shape (n_blocks, 26)
    hash_bucket_data : Dict[int, HashBucketArrays]
        Hash bucket data for heavy blocks
    node_positions : np.ndarray
        Node coordinates, shape (n_nodes, 3)
    connectivity : np.ndarray
        Element connectivity, shape (n_elements, 4)

    Returns
    -------
    element_id : int
        Found element ID or -1 if not found
    block_id : int
        Block ID where found or -1 if not found

    Performance
    -----------
    Expected: 0.1-1 ms per particle (depends on block size)
    """
    # Step 1: Find containing block (O(1))
    pos_jax = jnp.array(position, dtype=jnp.float32)
    bounds_jax = jnp.array(domain_bounds, dtype=jnp.float32)

    block_id = int(find_containing_block_jax(pos_jax, bounds_jax, grid_size))

    if block_id < 0:
        return -1, -1  # Outside domain

    # Convert to JAX arrays for search functions
    node_pos_jax = jnp.array(node_positions, dtype=jnp.float32)
    connectivity_jax = jnp.array(connectivity, dtype=jnp.int32)

    # Step 2: Search within primary block (L2)
    is_light = block_id in block_classification.light_blocks

    if is_light:
        # L2a: Direct search in light block
        block_elements = jnp.array(padded_arrays.block_elements[block_id], dtype=jnp.int32)
        block_count = int(padded_arrays.block_sizes[block_id])

        elem_id = search_level2a_light_block(
            pos_jax,
            block_id,
            block_elements,
            block_count,
            node_pos_jax,
            connectivity_jax
        )
    else:
        # L2b: Hash bucket search in heavy block
        if block_id not in hash_bucket_data:
            # Heavy block but no hash buckets (shouldn't happen)
            return -1, -1

        hash_arrays = hash_bucket_data[block_id]

        elem_id = search_level2b_hash_bucket(
            pos_jax,
            block_id,
            jnp.array(hash_arrays.bucket_elements, dtype=jnp.int32),
            jnp.array(hash_arrays.bucket_elem_counts, dtype=jnp.int32),
            jnp.array(hash_arrays.bucket_neighbors_6, dtype=jnp.int32),
            hash_arrays.n_buckets,
            hash_arrays.morton_bits,
            jnp.array(hash_arrays.block_bounds, dtype=jnp.float32),
            node_pos_jax,
            connectivity_jax
        )

    elem_id = int(elem_id)

    if elem_id >= 0:
        return elem_id, block_id  # Found in primary block

    # Step 3: Fallback to neighbor blocks (L3)
    neighbors_26 = jnp.array(block_neighbors_26[block_id], dtype=jnp.int32)
    block_elements_all = jnp.array(padded_arrays.block_elements, dtype=jnp.int32)
    block_counts_all = jnp.array(padded_arrays.block_sizes, dtype=jnp.int32)

    # Create heavy block flags array
    n_blocks = len(padded_arrays.block_sizes)
    heavy_block_flags = jnp.zeros(n_blocks, dtype=jnp.bool_)
    for hb_id in block_classification.heavy_blocks:
        heavy_block_flags = heavy_block_flags.at[hb_id].set(True)

    elem_id = search_level3_neighbor_blocks(
        pos_jax,
        block_id,
        neighbors_26,
        heavy_block_flags,
        block_elements_all,
        block_counts_all,
        node_pos_jax,
        connectivity_jax
    )

    elem_id = int(elem_id)

    if elem_id >= 0:
        # Find which neighbor block contains this element
        for neighbor_id in block_neighbors_26[block_id]:
            if neighbor_id < 0:
                continue
            block_elems = padded_arrays.block_elements[neighbor_id]
            if elem_id in block_elems:
                return elem_id, int(neighbor_id)

        # Shouldn't reach here, but return primary block as fallback
        return elem_id, block_id

    return -1, -1  # Not found


def initial_search_batch(
    particle_positions: np.ndarray,
    domain_bounds: np.ndarray,
    grid_size: Tuple[int, int, int],
    block_classification: BlockClassification,
    padded_arrays: PaddedArrays,
    block_neighbors_26: np.ndarray,
    hash_bucket_data: Dict[int, HashBucketArrays],
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, InitialSearchStats]:
    """
    Find containing elements for a batch of particles (initial assignment).

    Uses GPU multi-level search (L2 + L3) for fast initial assignment.

    Parameters
    ----------
    particle_positions : np.ndarray
        Particle positions, shape (n_particles, 3)
    domain_bounds : np.ndarray
        Domain bounds [xmin, xmax, ymin, ymax, zmin, zmax]
    grid_size : Tuple[int, int, int]
        Grid dimensions (nx, ny, nz)
    block_classification : BlockClassification
        Light/heavy block categorization
    padded_arrays : PaddedArrays
        Block-local element storage
    block_neighbors_26 : np.ndarray
        26-neighbor connectivity
    hash_bucket_data : Dict[int, HashBucketArrays]
        Hash bucket data for heavy blocks
    node_positions : np.ndarray
        Node coordinates
    connectivity : np.ndarray
        Element connectivity
    verbose : bool, optional
        Print progress (default: True)

    Returns
    -------
    element_ids : np.ndarray
        Found element IDs, shape (n_particles,), -1 if not found
    block_ids : np.ndarray
        Block IDs where found, shape (n_particles,), -1 if not found
    stats : InitialSearchStats
        Search statistics

    Performance
    -----------
    Expected: 1,000-5,000 particles/s (5-25× faster than CPU baseline)
    With full GPU JIT (Phase 5): >10,000 particles/s
    """
    n_particles = len(particle_positions)
    element_ids = np.full(n_particles, -1, dtype=np.int32)
    block_ids = np.full(n_particles, -1, dtype=np.int32)

    if verbose:
        print(f"GPU Initial Search (VECTORIZED): {n_particles:,} particles...")

    start_time = time.time()

    # STEP 1: Vectorized block finding for ALL particles at once
    if verbose:
        print(f"  Finding containing blocks...")

    positions_jax = jnp.array(particle_positions, dtype=jnp.float32)
    bounds_jax = jnp.array(domain_bounds, dtype=jnp.float32)

    # Vectorize over all particles using vmap
    find_blocks_vmap = jax.vmap(
        lambda pos: find_containing_block_jax(pos, bounds_jax, grid_size)
    )
    particle_block_ids = np.array(find_blocks_vmap(positions_jax), dtype=np.int32)

    n_outside = np.sum(particle_block_ids < 0)
    if verbose:
        print(f"  ✓ Blocks found: {n_particles - n_outside:,}/{n_particles:,}")
        if n_outside > 0:
            print(f"    ⚠️  {n_outside:,} particles outside domain")

    # STEP 2: Group particles by block for batched processing
    if verbose:
        print(f"  Grouping particles by block...")

    # Create dictionary: block_id -> list of particle indices
    particles_per_block = {}
    for i, block_id in enumerate(particle_block_ids):
        if block_id >= 0:
            if block_id not in particles_per_block:
                particles_per_block[block_id] = []
            particles_per_block[block_id].append(i)

    n_blocks_with_particles = len(particles_per_block)
    if verbose:
        print(f"  ✓ Particles distributed across {n_blocks_with_particles} blocks")

    # Convert mesh arrays ONCE to JAX (shared across all searches)
    node_pos_jax = jnp.array(node_positions, dtype=jnp.float32)
    connectivity_jax = jnp.array(connectivity, dtype=jnp.int32)

    # STEP 3: Process each block with batched L2 search
    if verbose:
        print(f"  Running L2 block search (batched)...")

    l2_hits = 0
    processed = 0

    # Batch size to avoid OOM (tune based on GPU memory)
    # Reduced from 1000 to 250 for 4GB GPU with octree loaded
    BATCH_SIZE = 250  # Process 250 particles at a time

    for block_id, particle_indices in particles_per_block.items():
        n_in_block = len(particle_indices)
        is_heavy = block_classification.is_heavy(block_id)

        # Get block data
        if is_heavy and hash_bucket_data and block_id in hash_bucket_data:
            # L2b: Heavy block with hash buckets (CSR)
            hash_arrays = hash_bucket_data[block_id]
            hash_bucket_elements_jax = jnp.array(hash_arrays.bucket_elements, dtype=jnp.int32)
            hash_bucket_counts_jax = jnp.array(hash_arrays.bucket_elem_counts, dtype=jnp.int32)
            bucket_neighbors_jax = jnp.array(hash_arrays.bucket_neighbors_6, dtype=jnp.int32)
            block_bounds_jax = jnp.array(hash_arrays.block_bounds, dtype=jnp.float32)
        else:
            # L2a: Light block - get block elements
            block_elems_cpu = padded_arrays.block_elements[block_id]
            block_count = int(padded_arrays.block_sizes[block_id])
            block_elems_jax = jnp.array(block_elems_cpu[:block_count], dtype=jnp.int32)

        # Process particles in batches to avoid OOM
        for batch_start in range(0, n_in_block, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, n_in_block)
            batch_indices = particle_indices[batch_start:batch_end]
            particle_batch = particle_positions[batch_indices]
            particle_batch_jax = jnp.array(particle_batch, dtype=jnp.float32)

            # L2: Search within primary block (vectorized over batch)
            if is_heavy and hash_bucket_data and block_id in hash_bucket_data:
                # L2b: Heavy block with hash bucket search
                search_hash_vmap = jax.vmap(
                    lambda pos: search_level2b_hash_bucket(
                        pos,
                        block_id,
                        hash_bucket_elements_jax,
                        hash_bucket_counts_jax,
                        bucket_neighbors_jax,
                        hash_arrays.n_buckets,
                        hash_arrays.morton_bits,
                        block_bounds_jax,
                        node_pos_jax,
                        connectivity_jax
                    )
                )
                found_elem_ids = np.array(search_hash_vmap(particle_batch_jax), dtype=np.int32)
            else:
                # L2a: Light block search
                search_light_vmap = jax.vmap(
                    lambda pos: search_level2a_light_block(
                        pos,
                        block_id,
                        block_elems_jax,
                        block_count,
                        node_pos_jax,
                        connectivity_jax
                    )
                )
                found_elem_ids = np.array(search_light_vmap(particle_batch_jax), dtype=np.int32)

            # Update results for particles found in L2 (batch)
            for i, elem_id in enumerate(found_elem_ids):
                particle_idx = batch_indices[i]
                if elem_id >= 0:
                    element_ids[particle_idx] = elem_id
                    block_ids[particle_idx] = block_id
                    l2_hits += 1

        processed += n_in_block

        if verbose and processed % 10000 < n_in_block:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"    Progress: {processed:,}/{n_particles:,} ({100*processed/n_particles:.1f}%) - {rate:.0f} p/s", end='\r')

    if verbose:
        print(f"\n  ✓ L2 search complete: {l2_hits:,}/{n_particles:,} found ({100*l2_hits/n_particles:.1f}%)")

    # STEP 4: L3 fallback for particles not found in primary block
    not_found_indices = np.where(element_ids < 0)[0]
    n_not_found_l2 = len(not_found_indices)

    l3_hits = 0

    if n_not_found_l2 > 0:
        if verbose:
            print(f"  Running L3 fallback for {n_not_found_l2:,} particles...")

        # Create heavy block flags for L3 search
        n_blocks = len(padded_arrays.block_sizes)
        heavy_flags = jnp.zeros(n_blocks, dtype=jnp.bool_)
        for hb_id in block_classification.heavy_blocks:
            heavy_flags = heavy_flags.at[hb_id].set(True)

        # L3: Sequential search (cannot vectorize due to memory constraints)
        # L3 searches 26 neighbor blocks with full padded arrays - vmap would cause OOM
        # Since L3 is only needed for <5% of particles, sequential is acceptable
        #
        # IMPORTANT: Don't upload full padded_arrays to GPU! Process one particle at a time
        # to avoid memory explosion (OOM with 10k particles)
        for idx in not_found_indices:
            block_id = particle_block_ids[idx]
            if block_id < 0:
                continue  # Outside domain

            pos = particle_positions[idx]  # Use CPU array, not JAX
            neighbors_26 = block_neighbors_26[block_id]

            # Search in neighbor blocks one at a time (avoids OOM)
            for neighbor_id in neighbors_26:
                if neighbor_id < 0:
                    continue

                # Get neighbor block data
                neighbor_block_elements = padded_arrays.block_elements[neighbor_id]
                neighbor_block_size = int(padded_arrays.block_sizes[neighbor_id])

                # Only search valid elements in this neighbor block
                if neighbor_block_size == 0:
                    continue

                # Upload only this block's data to GPU
                pos_jax = jnp.array(pos, dtype=jnp.float32)
                block_elem_jax = jnp.array(neighbor_block_elements[:neighbor_block_size], dtype=jnp.int32)

                # Check if heavy block - use hash bucket search
                is_heavy = neighbor_id in block_classification.heavy_blocks

                if is_heavy and hash_bucket_data and neighbor_id in hash_bucket_data:
                    # L2b: Heavy block hash bucket search
                    hash_arrays = hash_bucket_data[neighbor_id]
                    hash_bucket_elements_jax = jnp.array(hash_arrays.bucket_elements, dtype=jnp.int32)
                    hash_bucket_counts_jax = jnp.array(hash_arrays.bucket_elem_counts, dtype=jnp.int32)
                    bucket_neighbors_jax = jnp.array(hash_arrays.bucket_neighbors_6, dtype=jnp.int32)
                    block_bounds_jax = jnp.array(hash_arrays.block_bounds, dtype=jnp.float32)

                    elem_id = search_level2b_hash_bucket(
                        pos_jax,
                        neighbor_id,
                        hash_bucket_elements_jax,
                        hash_bucket_counts_jax,
                        bucket_neighbors_jax,
                        hash_arrays.n_buckets,
                        hash_arrays.morton_bits,
                        block_bounds_jax,
                        node_pos_jax,
                        connectivity_jax
                    )
                else:
                    # L2a: Light block search
                    elem_id = search_level2a_light_block(
                        pos_jax,
                        neighbor_id,
                        block_elem_jax,
                        neighbor_block_size,
                        node_pos_jax,
                        connectivity_jax
                    )

                if int(elem_id) >= 0:
                    element_ids[idx] = int(elem_id)
                    block_ids[idx] = int(neighbor_id)
                    l3_hits += 1
                    break

        if verbose:
            print(f"  ✓ L3 search complete: {l3_hits:,} additional particles found")

    total_time = time.time() - start_time

    # Compute statistics
    n_found = np.sum(element_ids >= 0)
    n_not_found = n_particles - n_found

    n_found_in_primary = l2_hits
    n_found_in_neighbor = l3_hits

    stats = InitialSearchStats(
        n_particles=n_particles,
        n_found=n_found,
        n_not_found=n_not_found,
        n_found_in_primary_block=n_found_in_primary,
        n_found_in_neighbor_blocks=n_found_in_neighbor,
        l2_hits=l2_hits,
        l3_hits=l3_hits,
        total_search_time=total_time,
        particles_per_second=n_particles / total_time if total_time > 0 else 0
    )

    if verbose:
        print(f"\n{stats}")

    return element_ids, block_ids, stats
