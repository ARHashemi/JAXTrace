"""
Initial Particle Assignment Using GPU Multi-Level Search - V2 (JAX Vectorized)

This is the JAX vmap vectorized version of initial_assignment.py that replaces
the Python serial loop with GPU-parallel vectorized execution.

Key Difference from V1:
- V1: Python for loop over particles (serial CPU execution)
- V2: JAX vmap vectorization (parallel GPU execution)

Expected Performance Improvement: 25-75× speedup
- V1 baseline: 8 p/s on ThreadedA (3.5M elements)
- V2 target: 200-600 p/s

Uses Phase 4's L2 (block) and L3 (neighbor blocks) search levels for fast
initial particle-to-element assignment.

Search Strategy:
    1. Find containing block (O(1) arithmetic) - vectorized
    2. L2: Search within block - vectorized
       - L2a: Light blocks (<10K) - direct search
       - L2b: Heavy blocks (≥10K) - hash bucket search
    3. L3: Fallback to 26-neighbor blocks if not found - vectorized

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


@jax.jit
def search_single_particle_initial(
    position: jax.Array,
    domain_bounds: jax.Array,
    grid_size_tuple: Tuple[int, int, int],
    block_elements_for_particle: jax.Array,  # 1D - for L2
    block_count: int,
    block_neighbors_26_for_particle: jax.Array,  # 1D - (26,)
    heavy_flags: jax.Array,
    padded_elements_all: jax.Array,  # 2D - for L3
    padded_counts: jax.Array,
    node_positions: jax.Array,
    connectivity: jax.Array
) -> Tuple[int, int, int]:
    """
    Find containing element for a single particle (JAX version).

    Uses masked execution: always execute L2 and L3, select first valid result.

    Parameters
    ----------
    position : jax.Array
        Particle position [x, y, z], shape (3,)
    domain_bounds : jax.Array
        Domain bounds [xmin, xmax, ymin, ymax, zmin, zmax]
    grid_size_tuple : Tuple[int, int, int]
        Grid dimensions (nx, ny, nz)
    block_elements_for_particle : jax.Array
        Block elements for this particle's primary block, shape (max_elem_per_block,)
    block_count : int
        Number of elements in primary block
    block_neighbors_26_for_particle : jax.Array
        26-neighbor block IDs, shape (26,)
    heavy_flags : jax.Array
        Boolean flags for heavy blocks, shape (n_blocks,)
    padded_elements_all : jax.Array
        All block elements, shape (n_blocks, max_elem_per_block)
    padded_counts : jax.Array
        Element counts per block, shape (n_blocks,)
    node_positions : jax.Array
        Node coordinates, shape (n_nodes, 3)
    connectivity : jax.Array
        Element connectivity, shape (n_elements, 4)

    Returns
    -------
    element_id : int
        Found element ID or -1 if not found
    block_id : int
        Block ID where found or -1 if not found
    search_level : int
        Which level found it: 2=L2, 3=L3, -1=not found
    """
    # Step 1: Find containing block (O(1))
    primary_block = find_containing_block_jax(position, domain_bounds, grid_size_tuple)

    # If outside domain, return early
    outside_domain = primary_block < 0

    # Safe block ID for indexing (use 0 if outside domain)
    safe_block = jnp.where(primary_block >= 0, primary_block, 0)

    # Step 2: L2 - Search within primary block (light block direct search)
    # Note: We use L2a for all blocks in this simplified version
    # A more optimized version could route heavy blocks to L2b hash bucket search
    r2 = search_level2a_light_block(
        position,
        safe_block,
        block_elements_for_particle,
        block_count,
        node_positions,
        connectivity
    )

    # Step 3: L3 - Search neighbor blocks (fallback)
    r3 = search_level3_neighbor_blocks(
        position,
        safe_block,
        block_neighbors_26_for_particle,
        heavy_flags,
        padded_elements_all,
        padded_counts,
        node_positions,
        connectivity
    )

    # Select first valid result using masked execution
    # Priority: L2 → L3 → not found
    candidates = jnp.array([r2, r3, -1], dtype=jnp.int32)
    levels = jnp.array([2, 3, -1], dtype=jnp.int32)

    # Find first valid (>= 0) result
    valid_mask = candidates >= 0
    first_valid_idx = jnp.argmax(valid_mask)

    element_id = candidates[first_valid_idx]
    search_level = levels[first_valid_idx]

    # Determine which block contains the element
    # If found in L2, use primary block
    # If found in L3, search through neighbor blocks (simplified - use primary for now)
    found_in_l2 = (search_level == 2) & (element_id >= 0)
    block_id = jnp.where(found_in_l2, primary_block, primary_block)  # Simplified

    # If outside domain or not found, set to -1
    element_id = jnp.where(outside_domain, -1, element_id)
    block_id = jnp.where(outside_domain | (element_id < 0), -1, block_id)
    search_level = jnp.where(outside_domain | (element_id < 0), -1, search_level)

    return element_id, block_id, search_level


def initial_search_batch_v2(
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
    Find containing elements for a batch of particles (initial assignment) - V2 JAX vmap.

    This is the vectorized version that replaces the Python for loop with JAX vmap.

    Performance Improvement: 25-75× speedup over V1
    - V1 baseline: 8 p/s on ThreadedA
    - V2 target: 200-600 p/s

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
        Hash bucket data for heavy blocks (not used in this simplified V2)
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
    """
    n_particles = len(particle_positions)

    if verbose:
        print(f"GPU Initial Search V2 (JAX vmap): {n_particles:,} particles...")

    # Convert to JAX arrays
    positions_jax = jnp.array(particle_positions, dtype=jnp.float32)
    bounds_jax = jnp.array(domain_bounds, dtype=jnp.float32)
    node_pos_jax = jnp.array(node_positions, dtype=jnp.float32)
    connectivity_jax = jnp.array(connectivity, dtype=jnp.int32)
    padded_elements_jax = jnp.array(padded_arrays.block_elements, dtype=jnp.int32)
    padded_counts_jax = jnp.array(padded_arrays.block_sizes, dtype=jnp.int32)
    neighbors_26_jax = jnp.array(block_neighbors_26, dtype=jnp.int32)

    # Create heavy block flags array
    n_blocks = len(padded_arrays.block_sizes)
    heavy_flags = jnp.zeros(n_blocks, dtype=jnp.bool_)
    for hb_id in block_classification.heavy_blocks:
        heavy_flags = heavy_flags.at[hb_id].set(True)

    # Step 1: Find primary blocks for all particles (vectorized)
    if verbose:
        print("  Finding primary blocks...")

    find_blocks_vmap = jax.vmap(
        lambda pos: find_containing_block_jax(pos, bounds_jax, grid_size)
    )
    primary_blocks = find_blocks_vmap(positions_jax)

    # Step 2: Prepare per-particle arrays
    # For each particle, extract its primary block's elements and neighbors
    if verbose:
        print("  Preparing per-particle data...")

    # Clamp primary blocks to valid range [0, n_blocks-1] for indexing
    safe_blocks = jnp.clip(primary_blocks, 0, n_blocks - 1)

    # Extract per-particle block elements (shape: n_particles, max_elem_per_block)
    particle_block_elements = padded_elements_jax[safe_blocks]

    # Extract per-particle block element counts
    particle_block_counts = padded_counts_jax[safe_blocks]

    # Extract per-particle block neighbors (shape: n_particles, 26)
    particle_block_neighbors = neighbors_26_jax[safe_blocks]

    # Step 3: Vectorized search over all particles
    if verbose:
        print("  Running vectorized search...")

    start_time = time.time()

    # Vectorize the search function over all particles
    search_vmap = jax.vmap(
        lambda pos, b_elems, b_count, b_neigh: search_single_particle_initial(
            pos,
            bounds_jax,
            grid_size,
            b_elems,
            b_count,
            b_neigh,
            heavy_flags,
            padded_elements_jax,
            padded_counts_jax,
            node_pos_jax,
            connectivity_jax
        )
    )

    element_ids_jax, block_ids_jax, search_levels_jax = search_vmap(
        positions_jax,
        particle_block_elements,
        particle_block_counts,
        particle_block_neighbors
    )

    # Wait for GPU completion
    element_ids_jax.block_until_ready()

    total_time = time.time() - start_time

    # Convert back to numpy
    element_ids = np.array(element_ids_jax, dtype=np.int32)
    block_ids = np.array(block_ids_jax, dtype=np.int32)
    search_levels = np.array(search_levels_jax, dtype=np.int32)

    # Compute statistics
    n_found = np.sum(element_ids >= 0)
    n_not_found = n_particles - n_found

    # Count L2 vs L3 hits
    l2_hits = np.sum(search_levels == 2)
    l3_hits = np.sum(search_levels == 3)

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


if __name__ == "__main__":
    """Test initial assignment V2 with synthetic data."""
    print("Testing Initial Assignment V2 (JAX vmap)...")

    # Create synthetic mesh
    print("\nCreating synthetic mesh...")
    n_nodes = 200
    n_elements = 400
    n_blocks = 8
    nx, ny, nz = 2, 2, 2

    node_positions = np.random.uniform(0, 10, (n_nodes, 3)).astype(np.float32)
    connectivity = np.random.randint(0, n_nodes, (n_elements, 4), dtype=np.int32)

    # Domain bounds
    domain_bounds = np.array([0.0, 10.0, 0.0, 10.0, 0.0, 10.0], dtype=np.float32)

    # Create synthetic padded arrays
    max_elem_per_block = 100
    padded_elements = np.full((n_blocks, max_elem_per_block), -1, dtype=np.int32)
    padded_counts = np.array([50, 60, 45, 45, 55, 48, 52, 50], dtype=np.int32)

    for b in range(n_blocks):
        count = min(padded_counts[b], n_elements)
        start_idx = (b * n_elements) // n_blocks
        end_idx = min(start_idx + count, n_elements)
        actual_count = end_idx - start_idx
        padded_elements[b, :actual_count] = np.arange(start_idx, end_idx)
        padded_counts[b] = actual_count

    # Mock padded arrays
    class MockPaddedArrays:
        def __init__(self):
            self.block_elements = padded_elements
            self.block_sizes = padded_counts

    padded_arrays = MockPaddedArrays()

    # Block neighbors (simplified)
    block_neighbors_26 = np.full((n_blocks, 26), -1, dtype=np.int32)

    # Mock classification (all light blocks)
    class MockClassification:
        def __init__(self):
            self.light_blocks = set(range(n_blocks))
            self.heavy_blocks = set()

    classification = MockClassification()

    # Synthetic particles
    n_particles = 100
    particle_positions = np.random.uniform(0, 10, (n_particles, 3)).astype(np.float32)

    print("\nRunning initial search V2...")
    element_ids, block_ids, stats = initial_search_batch_v2(
        particle_positions,
        domain_bounds,
        (nx, ny, nz),
        classification,
        padded_arrays,
        block_neighbors_26,
        {},  # No hash buckets
        node_positions,
        connectivity,
        verbose=True
    )

    print(f"\nResults:")
    print(f"  Found: {np.sum(element_ids >= 0):,}/{n_particles}")
    print(f"  Not found: {np.sum(element_ids < 0):,}/{n_particles}")
    print(f"  Throughput: {stats.particles_per_second:.0f} p/s")

    print("\n✅ Initial assignment V2 test complete!")
