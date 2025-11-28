"""
Incremental Search for RK4 Time Integration

Optimized search for particles with small displacements (RK4 intermediate stages).
Uses L0 (cached element) + L1 (face neighbors) before falling back to full search.

Expected performance:
- L0 hit rate: 60-80% for dt/2 displacement (< 1 μs/particle)
- L1 hit rate: 15-25% for boundary crossings (< 5 μs/particle)
- Full search fallback: ~5% (10-80 s / 1000 particles)

Total expected speedup: 10-50× vs always doing full search
"""

import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import time

from .level0_cached import search_level0_cached
from .level1_neighbors import search_level1_neighbors
from .level2a_light import search_level2a_light_block
from .level2b_heavy import search_level2b_hash_bucket
from .level3_neighbor_blocks import search_level3_neighbor_blocks
from .block_classifier import BlockClassification
from .hash_bucket import HashBucketArrays
from ..forest.padded_arrays import PaddedArrays

jax.config.update("jax_enable_x64", True)


@dataclass
class IncrementalSearchStats:
    """Statistics from incremental search."""
    n_particles: int
    n_found: int
    l0_hits: int  # Found in cached element
    l1_hits: int  # Found in face neighbors
    l2_hits: int  # Found in block search
    l3_hits: int  # Found in neighbor blocks
    n_not_found: int
    total_search_time: float
    particles_per_second: float

    def __repr__(self) -> str:
        return (
            f"IncrementalSearchStats(\n"
            f"  Particles: {self.n_particles:,}\n"
            f"  Found: {self.n_found:,} ({100*self.n_found/self.n_particles:.1f}%)\n"
            f"  L0 hits (cached): {self.l0_hits:,} ({100*self.l0_hits/self.n_particles:.1f}%)\n"
            f"  L1 hits (neighbors): {self.l1_hits:,} ({100*self.l1_hits/self.n_particles:.1f}%)\n"
            f"  L2 hits (block): {self.l2_hits:,} ({100*self.l2_hits/self.n_particles:.1f}%)\n"
            f"  L3 hits (neighbor blocks): {self.l3_hits:,} ({100*self.l3_hits/self.n_particles:.1f}%)\n"
            f"  Not found: {self.n_not_found:,} ({100*self.n_not_found/self.n_particles:.1f}%)\n"
            f"  Time: {self.total_search_time:.2f} s\n"
            f"  Rate: {self.particles_per_second:.0f} particles/s\n"
            f")"
        )


def incremental_search_batch(
    particle_positions: np.ndarray,
    cached_element_ids: np.ndarray,
    cached_block_ids: np.ndarray,
    domain_bounds: np.ndarray,
    grid_size: Tuple[int, int, int],
    block_classification: BlockClassification,
    padded_arrays: PaddedArrays,
    block_neighbors_26: np.ndarray,
    hash_bucket_data: Dict[int, HashBucketArrays],
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    element_neighbors: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, IncrementalSearchStats]:
    """
    Incremental search optimized for small particle displacements.

    Search hierarchy:
    1. L0: Check if still in cached element (fastest, 60-80% hit rate)
    2. L1: Check 4 face-adjacent neighbors (fast, 15-25% hit rate)
    3. L2: Search block (medium, ~3-5% hit rate)
    4. L3: Search 26-neighbor blocks (slow, ~1-2% hit rate)

    Parameters
    ----------
    particle_positions : np.ndarray
        Current particle positions (N, 3)
    cached_element_ids : np.ndarray
        Last known element IDs (N,), from previous timestep or RK4 stage
    cached_block_ids : np.ndarray
        Last known block IDs (N,)
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
    element_neighbors : np.ndarray, optional
        Element face neighbors (n_elements, 4), -1 for no neighbor
        If None, L1 search is skipped
    verbose : bool
        Print progress

    Returns
    -------
    element_ids : np.ndarray
        Found element IDs (N,), -1 if not found
    block_ids : np.ndarray
        Block IDs where found (N,), -1 if not found
    stats : IncrementalSearchStats
        Search statistics with hit rates per level

    Performance
    -----------
    Expected: 10-50× faster than full search for RK4 intermediate stages
    With 70% L0 + 20% L1 hit rate:
    - 0.7 * 1 μs + 0.2 * 5 μs + 0.1 * 10 ms = ~1.7 μs avg per particle
    - vs full search: ~10 ms per particle
    - Speedup: ~6000×
    """

    n_particles = len(particle_positions)
    element_ids = np.full(n_particles, -1, dtype=np.int32)
    block_ids = np.full(n_particles, -1, dtype=np.int32)

    # Track hit rates per level
    l0_hits = 0
    l1_hits = 0
    l2_hits = 0
    l3_hits = 0

    if verbose:
        print(f"GPU Incremental Search (L0+L1 optimized): {n_particles:,} particles...")

    start_time = time.time()

    # Convert to JAX arrays once
    positions_jax = jnp.array(particle_positions, dtype=jnp.float32)
    node_pos_jax = jnp.array(node_positions, dtype=jnp.float32)
    connectivity_jax = jnp.array(connectivity, dtype=jnp.int32)
    cached_elem_ids_jax = jnp.array(cached_element_ids, dtype=jnp.int32)

    # ========================================================================
    # LEVEL 0: Check cached elements (vectorized)
    # ========================================================================
    if verbose:
        print(f"  L0: Checking cached elements...")

    # Vectorize L0 search over all particles
    search_l0_vmap = jax.vmap(
        lambda pos, elem_id: search_level0_cached(
            pos, elem_id, node_pos_jax, connectivity_jax
        )
    )
    l0_results = np.array(search_l0_vmap(positions_jax, cached_elem_ids_jax), dtype=np.int32)

    # Update results for L0 hits
    l0_mask = l0_results >= 0
    element_ids[l0_mask] = l0_results[l0_mask]
    block_ids[l0_mask] = cached_block_ids[l0_mask]
    l0_hits = np.sum(l0_mask)

    if verbose:
        print(f"  ✓ L0 hits: {l0_hits:,}/{n_particles:,} ({100*l0_hits/n_particles:.1f}%)")

    # ========================================================================
    # LEVEL 1: Check face neighbors (vectorized, if data available)
    # ========================================================================
    if element_neighbors is not None:
        not_found_l0 = np.where(~l0_mask)[0]

        if len(not_found_l0) > 0 and verbose:
            print(f"  L1: Checking face neighbors for {len(not_found_l0):,} particles...")

        element_neighbors_jax = jnp.array(element_neighbors, dtype=jnp.int32)

        for idx in not_found_l0:
            pos = positions_jax[idx]
            cached_elem = cached_element_ids[idx]

            # Skip invalid cached elements
            if cached_elem < 0 or cached_elem >= len(element_neighbors):
                continue

            # Get face neighbors for cached element
            neighbors = element_neighbors_jax[cached_elem]

            # Search neighbors
            elem_id = int(search_level1_neighbors(
                pos, cached_elem, neighbors,
                node_pos_jax, connectivity_jax
            ))

            if elem_id >= 0:
                element_ids[idx] = elem_id
                # Find block containing this element
                # For now, keep cached block (may be incorrect if crossed block boundary)
                block_ids[idx] = cached_block_ids[idx]
                l1_hits += 1

        if verbose and element_neighbors is not None:
            print(f"  ✓ L1 hits: {l1_hits:,}/{n_particles:,} ({100*l1_hits/n_particles:.1f}%)")
    else:
        if verbose:
            print(f"  ⚠️  L1 skipped: element_neighbors not available")

    # ========================================================================
    # LEVEL 2+3: Fall back to block search for remaining particles
    # ========================================================================
    not_found_mask = element_ids < 0
    not_found_indices = np.where(not_found_mask)[0]

    if len(not_found_indices) > 0:
        if verbose:
            print(f"  L2/L3: Full search for remaining {len(not_found_indices):,} particles...")

        # Use existing initial_search_batch for L2+L3
        # NOTE: This imports from initial_assignment to avoid code duplication
        from .initial_assignment import initial_search_batch

        remaining_positions = particle_positions[not_found_indices]
        elem_found, block_found, search_stats = initial_search_batch(
            remaining_positions,
            domain_bounds,
            grid_size,
            block_classification,
            padded_arrays,
            block_neighbors_26,
            hash_bucket_data,
            node_positions,
            connectivity,
            verbose=False
        )

        # Update results
        element_ids[not_found_indices] = elem_found
        block_ids[not_found_indices] = block_found

        # Track L2/L3 hits
        l2_hits = search_stats.l2_hits
        l3_hits = search_stats.l3_hits

        if verbose:
            print(f"  ✓ L2 hits: {l2_hits:,}/{n_particles:,} ({100*l2_hits/n_particles:.1f}%)")
            print(f"  ✓ L3 hits: {l3_hits:,}/{n_particles:,} ({100*l3_hits/n_particles:.1f}%)")

    total_time = time.time() - start_time

    # Compute final statistics
    n_found = np.sum(element_ids >= 0)
    n_not_found = n_particles - n_found

    stats = IncrementalSearchStats(
        n_particles=n_particles,
        n_found=n_found,
        l0_hits=l0_hits,
        l1_hits=l1_hits,
        l2_hits=l2_hits,
        l3_hits=l3_hits,
        n_not_found=n_not_found,
        total_search_time=total_time,
        particles_per_second=n_particles / total_time if total_time > 0 else 0
    )

    if verbose:
        print(f"\n{stats}")

    return element_ids, block_ids, stats
