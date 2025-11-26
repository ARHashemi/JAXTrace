"""
Multi-Level Search Orchestrator - Phase 4, Task 4.8

Integrates all search levels (L0-L3) into a complete hierarchical search pipeline.

Search Order:
    L0: Cached element (85-95% hit rate, < 1 μs)
    L1: Neighbor elements (3-10% hit rate, < 5 μs)
    L2: Current block (1-5% hit rate, < 100 μs)
        - L2a: Light blocks (<10K) - direct search
        - L2b: Heavy blocks (≥10K) - hash bucket search
    L3: Neighbor blocks (0.1-1% hit rate, < 1000 μs)

Performance: Expected > 10,000 particles/second on ThreadedA
"""

import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import time

from .level0_cached import search_level0_cached
from .level1_neighbors import search_level1_neighbors
from .level2a_light import search_level2a_light_block
from .level2b_heavy import search_level2b_hash_bucket
from .level3_neighbor_blocks import search_level3_neighbor_blocks
from .block_classifier import BlockClassification
from .hash_bucket import HashBucketArrays

jax.config.update("jax_enable_x64", True)


@dataclass
class SearchStats:
    """
    Per-level search statistics.

    Tracks hit rates and performance for each search level.
    """
    n_particles: int
    l0_hits: int
    l1_hits: int
    l2_hits: int
    l3_hits: int
    not_found: int
    l0_time: float
    l1_time: float
    l2_time: float
    l3_time: float
    total_time: float

    def __repr__(self) -> str:
        """Human-readable summary."""
        total_found = self.l0_hits + self.l1_hits + self.l2_hits + self.l3_hits
        success_rate = 100 * total_found / self.n_particles if self.n_particles > 0 else 0

        return (
            f"SearchStats(\n"
            f"  Particles: {self.n_particles:,}\n"
            f"  Found: {total_found:,} ({success_rate:.1f}%)\n"
            f"  L0 hits: {self.l0_hits:,} ({100*self.l0_hits/self.n_particles:.1f}%)\n"
            f"  L1 hits: {self.l1_hits:,} ({100*self.l1_hits/self.n_particles:.1f}%)\n"
            f"  L2 hits: {self.l2_hits:,} ({100*self.l2_hits/self.n_particles:.1f}%)\n"
            f"  L3 hits: {self.l3_hits:,} ({100*self.l3_hits/self.n_particles:.1f}%)\n"
            f"  Not found: {self.not_found:,} ({100*self.not_found/self.n_particles:.1f}%)\n"
            f"  Total time: {self.total_time:.2f} s\n"
            f"  Throughput: {self.n_particles/self.total_time:.0f} particles/s\n"
            f")"
        )

    def print_detailed(self):
        """Print detailed per-level statistics."""
        print("\n" + "=" * 80)
        print("MULTI-LEVEL SEARCH STATISTICS")
        print("=" * 80)
        print(f"\nParticles processed: {self.n_particles:,}")
        print()

        # Hit rates
        print("Hit Rates:")
        print(f"  L0 (Cached):       {self.l0_hits:8,} ({100*self.l0_hits/self.n_particles:5.1f}%)")
        print(f"  L1 (Neighbors):    {self.l1_hits:8,} ({100*self.l1_hits/self.n_particles:5.1f}%)")
        print(f"  L2 (Block):        {self.l2_hits:8,} ({100*self.l2_hits/self.n_particles:5.1f}%)")
        print(f"  L3 (Neighbor blk): {self.l3_hits:8,} ({100*self.l3_hits/self.n_particles:5.1f}%)")
        print(f"  Not found:         {self.not_found:8,} ({100*self.not_found/self.n_particles:5.1f}%)")
        print()

        # Timing
        print("Timing:")
        print(f"  L0 time: {self.l0_time:6.2f} s ({100*self.l0_time/self.total_time:5.1f}%)")
        print(f"  L1 time: {self.l1_time:6.2f} s ({100*self.l1_time/self.total_time:5.1f}%)")
        print(f"  L2 time: {self.l2_time:6.2f} s ({100*self.l2_time/self.total_time:5.1f}%)")
        print(f"  L3 time: {self.l3_time:6.2f} s ({100*self.l3_time/self.total_time:5.1f}%)")
        print(f"  Total:   {self.total_time:6.2f} s")
        print()

        # Performance
        throughput = self.n_particles / self.total_time if self.total_time > 0 else 0
        print(f"Throughput: {throughput:,.0f} particles/second")
        print("=" * 80)


def multi_level_search_batch(
    particle_positions: np.ndarray,
    cached_element_ids: np.ndarray,
    cached_block_ids: np.ndarray,
    block_classification: BlockClassification,
    padded_block_elements: np.ndarray,
    padded_block_counts: np.ndarray,
    element_neighbors: np.ndarray,
    block_neighbors_26: np.ndarray,
    hash_bucket_data: Optional[Dict[int, HashBucketArrays]],
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, SearchStats]:
    """
    Multi-level search for batch of particles.

    Search order: L0 → L1 → L2 (L2a/L2b) → L3

    Parameters
    ----------
    particle_positions : np.ndarray
        Particle positions (n_particles, 3)
    cached_element_ids : np.ndarray
        Last known element IDs (n_particles,)
    cached_block_ids : np.ndarray
        Last known block IDs (n_particles,)
    block_classification : BlockClassification
        Light vs heavy block classification
    padded_block_elements : np.ndarray
        Padded element arrays (n_blocks, max_elem_per_block)
    padded_block_counts : np.ndarray
        Element counts (n_blocks,)
    element_neighbors : np.ndarray
        Neighbor arrays (n_elements, max_neighbors)
    block_neighbors_26 : np.ndarray
        26-neighbor topology (n_blocks, 26)
    hash_bucket_data : Dict[int, HashBucketArrays] or None
        Hash bucket arrays for heavy blocks
    node_positions : np.ndarray
        Node positions (n_nodes, 3)
    connectivity : np.ndarray
        Element connectivity (n_elements, 4)
    verbose : bool
        Print progress

    Returns
    -------
    element_ids : np.ndarray
        Found element IDs (n_particles,), -1 if not found
    block_ids : np.ndarray
        Found block IDs (n_particles,), -1 if not found
    stats : SearchStats
        Search statistics

    Performance
    -----------
    Expected: > 10,000 particles/second on ThreadedA
    """
    n_particles = len(particle_positions)

    if verbose:
        print(f"\nMulti-level search: {n_particles:,} particles")

    # Convert to JAX arrays
    positions_jax = jnp.array(particle_positions, dtype=jnp.float32)
    node_pos_jax = jnp.array(node_positions, dtype=jnp.float32)
    connectivity_jax = jnp.array(connectivity, dtype=jnp.int32)
    elem_neighbors_jax = jnp.array(element_neighbors, dtype=jnp.int32)
    padded_elements_jax = jnp.array(padded_block_elements, dtype=jnp.int32)
    padded_counts_jax = jnp.array(padded_block_counts, dtype=jnp.int32)

    # Initialize results
    element_ids = np.full(n_particles, -1, dtype=np.int32)
    block_ids = np.full(n_particles, -1, dtype=np.int32)
    search_levels = np.full(n_particles, -1, dtype=np.int32)  # Track which level found it

    # Statistics
    l0_hits = l1_hits = l2_hits = l3_hits = 0
    t0_total = time.time()
    l0_time = l1_time = l2_time = l3_time = 0.0

    # Process each particle
    for i in range(n_particles):
        pos = positions_jax[i]
        cached_elem = int(cached_element_ids[i])
        cached_block = int(cached_block_ids[i])

        # L0: Check cached element
        t0 = time.time()
        elem_id = search_level0_cached(pos, cached_elem, node_pos_jax, connectivity_jax)
        l0_time += time.time() - t0

        if int(elem_id) >= 0:
            element_ids[i] = int(elem_id)
            block_ids[i] = cached_block
            search_levels[i] = 0
            l0_hits += 1
            continue

        # L1: Check neighbor elements
        if cached_elem >= 0:
            t1 = time.time()
            neighbors = elem_neighbors_jax[cached_elem]
            elem_id = search_level1_neighbors(pos, cached_elem, neighbors, node_pos_jax, connectivity_jax)
            l1_time += time.time() - t1

            if int(elem_id) >= 0:
                element_ids[i] = int(elem_id)
                # Find block for this element (simplified - use cached for now)
                block_ids[i] = cached_block
                search_levels[i] = 1
                l1_hits += 1
                continue

        # L2: Search current block
        if cached_block >= 0:
            t2 = time.time()

            is_heavy = block_classification.is_heavy(cached_block)

            if is_heavy and hash_bucket_data and cached_block in hash_bucket_data:
                # L2b: Heavy block hash bucket search
                hash_arrays = hash_bucket_data[cached_block]
                elem_id = search_level2b_hash_bucket(
                    pos,
                    cached_block,
                    jnp.array(hash_arrays.bucket_elements),
                    jnp.array(hash_arrays.bucket_elem_counts),
                    jnp.array(hash_arrays.bucket_neighbors_6),
                    hash_arrays.n_buckets,
                    hash_arrays.morton_bits,
                    jnp.array(hash_arrays.block_bounds),
                    node_pos_jax,
                    connectivity_jax
                )
            else:
                # L2a: Light block direct search
                elem_id = search_level2a_light_block(
                    pos,
                    cached_block,
                    padded_elements_jax[cached_block],
                    int(padded_counts_jax[cached_block]),
                    node_pos_jax,
                    connectivity_jax
                )

            l2_time += time.time() - t2

            if int(elem_id) >= 0:
                element_ids[i] = int(elem_id)
                block_ids[i] = cached_block
                search_levels[i] = 2
                l2_hits += 1
                continue

        # L3: Search neighbor blocks
        if cached_block >= 0 and block_neighbors_26 is not None:
            t3 = time.time()

            neighbors_26 = jnp.array(block_neighbors_26[cached_block], dtype=jnp.int32)

            # Create heavy block flags array
            n_blocks = len(padded_block_counts)
            heavy_flags = jnp.zeros(n_blocks, dtype=jnp.bool_)
            for hb_id in block_classification.heavy_blocks:
                heavy_flags = heavy_flags.at[hb_id].set(True)

            elem_id = search_level3_neighbor_blocks(
                pos,
                cached_block,
                neighbors_26,
                heavy_flags,
                padded_elements_jax,
                padded_counts_jax,
                node_pos_jax,
                connectivity_jax
            )

            l3_time += time.time() - t3

            if int(elem_id) >= 0:
                element_ids[i] = int(elem_id)
                # Find which neighbor block contains this element
                for neighbor_id in block_neighbors_26[cached_block]:
                    if neighbor_id < 0:
                        continue
                    block_ids[i] = int(neighbor_id)  # Simplified - use first valid neighbor
                    break
                search_levels[i] = 3
                l3_hits += 1
                continue

        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i+1:,}/{n_particles:,}")

    total_time = time.time() - t0_total
    not_found = n_particles - (l0_hits + l1_hits + l2_hits + l3_hits)

    stats = SearchStats(
        n_particles=n_particles,
        l0_hits=l0_hits,
        l1_hits=l1_hits,
        l2_hits=l2_hits,
        l3_hits=l3_hits,
        not_found=not_found,
        l0_time=l0_time,
        l1_time=l1_time,
        l2_time=l2_time,
        l3_time=l3_time,
        total_time=total_time
    )

    if verbose:
        print(f"\n{stats}")

    return element_ids, block_ids, stats


def multi_level_search_batch_vectorized(
    particle_positions: np.ndarray,
    cached_element_ids: np.ndarray,
    cached_block_ids: np.ndarray,
    block_classification: BlockClassification,
    padded_block_elements: np.ndarray,
    padded_block_counts: np.ndarray,
    element_neighbors: np.ndarray,
    block_neighbors_26: np.ndarray,
    hash_bucket_data: Optional[Dict[int, HashBucketArrays]],
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, SearchStats]:
    """
    Vectorized multi-level search for batch of particles.

    Vectorization Strategy:
        L0 (Cached):     Full vmap over ALL particles (85-95% hit rate)
        L1 (Neighbors):  Full vmap over L0-miss particles (3-10% hit rate)
        L2 (Block):      Block-grouped vmap over L1-miss (1-5% hit rate)
        L3 (26-neighbors): Sequential over L2-miss (<1% hit rate, avoid OOM)

    Expected Performance: 5,000-15,000 p/s (15-40× speedup over sequential)

    Parameters
    ----------
    Same as multi_level_search_batch()

    Returns
    -------
    element_ids : np.ndarray
        Found element IDs (n_particles,), -1 if not found
    block_ids : np.ndarray
        Found block IDs (n_particles,), -1 if not found
    stats : SearchStats
        Search statistics
    """
    n_particles = len(particle_positions)

    if verbose:
        print(f"\n{'='*80}")
        print(f"VECTORIZED MULTI-LEVEL SEARCH: {n_particles:,} particles")
        print(f"{'='*80}")

    # Convert to JAX arrays
    positions_jax = jnp.array(particle_positions, dtype=jnp.float32)
    node_pos_jax = jnp.array(node_positions, dtype=jnp.float32)
    connectivity_jax = jnp.array(connectivity, dtype=jnp.int32)
    elem_neighbors_jax = jnp.array(element_neighbors, dtype=jnp.int32)
    padded_elements_jax = jnp.array(padded_block_elements, dtype=jnp.int32)
    padded_counts_jax = jnp.array(padded_block_counts, dtype=jnp.int32)
    cached_elem_jax = jnp.array(cached_element_ids, dtype=jnp.int32)
    cached_block_jax = jnp.array(cached_block_ids, dtype=jnp.int32)

    # Initialize results
    element_ids = np.full(n_particles, -1, dtype=np.int32)
    block_ids = np.full(n_particles, -1, dtype=np.int32)

    # Statistics
    l0_hits = l1_hits = l2_hits = l3_hits = 0
    t0_total = time.time()

    # ========================================================================
    # LEVEL 0: Vectorized cached element check (ALL particles)
    # ========================================================================
    if verbose:
        print(f"\n🔍 L0: Checking cached elements (vectorized over {n_particles:,} particles)...")

    t0 = time.time()

    # Vectorize over ALL particles
    search_l0_vmap = jax.vmap(
        lambda pos, cached_elem: search_level0_cached(pos, cached_elem, node_pos_jax, connectivity_jax)
    )

    l0_results = np.array(search_l0_vmap(positions_jax, cached_elem_jax), dtype=np.int32)
    l0_time = time.time() - t0

    # Extract L0 hits
    l0_mask = l0_results >= 0
    l0_indices = np.where(l0_mask)[0]
    l0_hits = len(l0_indices)

    element_ids[l0_indices] = l0_results[l0_indices]
    block_ids[l0_indices] = cached_block_ids[l0_indices]

    if verbose:
        print(f"   ✓ L0 hits: {l0_hits:,}/{n_particles:,} ({100*l0_hits/n_particles:.1f}%) in {l0_time:.2f}s")

    # L0 misses proceed to L1
    l0_miss_indices = np.where(~l0_mask)[0]
    n_l0_miss = len(l0_miss_indices)

    if n_l0_miss == 0:
        # All particles found in L0
        stats = SearchStats(
            n_particles=n_particles, l0_hits=l0_hits, l1_hits=0, l2_hits=0, l3_hits=0,
            not_found=0, l0_time=l0_time, l1_time=0.0, l2_time=0.0, l3_time=0.0,
            total_time=time.time() - t0_total
        )
        if verbose:
            print(f"\n✅ All particles found in L0!")
        return element_ids, block_ids, stats

    # ========================================================================
    # LEVEL 1: Vectorized neighbor element search (L0-miss particles)
    # ========================================================================
    if verbose:
        print(f"\n🔍 L1: Checking neighbor elements (vectorized over {n_l0_miss:,} particles)...")

    t1 = time.time()

    # Filter particles with valid cached elements
    l0_miss_cached_elem = cached_elem_jax[l0_miss_indices]
    valid_cache_mask = l0_miss_cached_elem >= 0
    valid_cache_indices = l0_miss_indices[np.array(valid_cache_mask)]

    l1_results = np.full(len(valid_cache_indices), -1, dtype=np.int32)

    if len(valid_cache_indices) > 0:
        # Vectorize over particles with valid cache
        valid_positions = positions_jax[valid_cache_indices]
        valid_cached_elems = cached_elem_jax[valid_cache_indices]

        search_l1_vmap = jax.vmap(
            lambda pos, cached_elem: search_level1_neighbors(
                pos, cached_elem, elem_neighbors_jax[cached_elem],
                node_pos_jax, connectivity_jax
            )
        )

        l1_results = np.array(search_l1_vmap(valid_positions, valid_cached_elems), dtype=np.int32)

    l1_time = time.time() - t1

    # Extract L1 hits
    l1_mask = l1_results >= 0
    l1_hit_local_indices = np.where(l1_mask)[0]
    l1_hit_global_indices = valid_cache_indices[l1_hit_local_indices]
    l1_hits = len(l1_hit_global_indices)

    element_ids[l1_hit_global_indices] = l1_results[l1_hit_local_indices]
    block_ids[l1_hit_global_indices] = cached_block_ids[l1_hit_global_indices]

    if verbose:
        print(f"   ✓ L1 hits: {l1_hits:,}/{n_l0_miss:,} ({100*l1_hits/n_l0_miss:.1f}%) in {l1_time:.2f}s")

    # L1 misses proceed to L2
    found_mask = element_ids >= 0
    l1_miss_indices = np.where(~found_mask)[0]
    n_l1_miss = len(l1_miss_indices)

    if n_l1_miss == 0:
        # All remaining found in L1
        stats = SearchStats(
            n_particles=n_particles, l0_hits=l0_hits, l1_hits=l1_hits, l2_hits=0, l3_hits=0,
            not_found=0, l0_time=l0_time, l1_time=l1_time, l2_time=0.0, l3_time=0.0,
            total_time=time.time() - t0_total
        )
        if verbose:
            print(f"\n✅ All particles found in L0+L1!")
        return element_ids, block_ids, stats

    # ========================================================================
    # LEVEL 2: Vectorized block search (L1-miss particles, grouped by block)
    # ========================================================================
    if verbose:
        print(f"\n🔍 L2: Searching blocks (vectorized over {n_l1_miss:,} particles, block-grouped)...")

    t2 = time.time()

    # Group L1-miss particles by their cached block
    particles_per_block = {}
    for idx in l1_miss_indices:
        block_id = int(cached_block_ids[idx])
        if block_id >= 0:
            particles_per_block.setdefault(block_id, []).append(idx)

    # Create heavy block flags
    n_blocks = len(padded_block_counts)
    is_heavy = np.zeros(n_blocks, dtype=bool)
    for hb_id in block_classification.heavy_blocks:
        is_heavy[hb_id] = True

    # Process each block
    for block_id, particle_indices in particles_per_block.items():
        particle_batch = particle_positions[particle_indices]
        particle_batch_jax = jnp.array(particle_batch, dtype=jnp.float32)

        if is_heavy[block_id] and hash_bucket_data and block_id in hash_bucket_data:
            # L2b: Heavy block hash bucket search (vectorized)
            hash_arrays = hash_bucket_data[block_id]

            bucket_elements_jax = jnp.array(hash_arrays.bucket_elements, dtype=jnp.int32)
            bucket_counts_jax = jnp.array(hash_arrays.bucket_elem_counts, dtype=jnp.int32)
            bucket_neighbors_jax = jnp.array(hash_arrays.bucket_neighbors_6, dtype=jnp.int32)
            block_bounds_jax = jnp.array(hash_arrays.block_bounds, dtype=jnp.float32)

            search_hash_vmap = jax.vmap(
                lambda pos: search_level2b_hash_bucket(
                    pos, block_id, bucket_elements_jax, bucket_counts_jax,
                    bucket_neighbors_jax, hash_arrays.n_buckets,
                    hash_arrays.morton_bits, block_bounds_jax,
                    node_pos_jax, connectivity_jax
                )
            )

            found_elem_ids = np.array(search_hash_vmap(particle_batch_jax), dtype=np.int32)
        else:
            # L2a: Light block direct search (vectorized)
            block_elems = padded_elements_jax[block_id]
            block_count = int(padded_counts_jax[block_id])

            search_light_vmap = jax.vmap(
                lambda pos: search_level2a_light_block(
                    pos, block_id, block_elems, block_count,
                    node_pos_jax, connectivity_jax
                )
            )

            found_elem_ids = np.array(search_light_vmap(particle_batch_jax), dtype=np.int32)

        # Store results
        for local_idx, global_idx in enumerate(particle_indices):
            elem_id = found_elem_ids[local_idx]
            if elem_id >= 0:
                element_ids[global_idx] = elem_id
                block_ids[global_idx] = block_id

    l2_time = time.time() - t2

    # Count L2 hits
    l2_mask = (element_ids >= 0) & ~found_mask  # New hits since L1
    l2_indices = np.where(l2_mask)[0]
    l2_hits = len(l2_indices)

    if verbose:
        print(f"   ✓ L2 hits: {l2_hits:,}/{n_l1_miss:,} ({100*l2_hits/n_l1_miss:.1f}% of L1-miss) in {l2_time:.2f}s")

    # L2 misses proceed to L3
    found_mask = element_ids >= 0
    l2_miss_indices = np.where(~found_mask)[0]
    n_l2_miss = len(l2_miss_indices)

    if n_l2_miss == 0:
        # All remaining found in L2
        stats = SearchStats(
            n_particles=n_particles, l0_hits=l0_hits, l1_hits=l1_hits, l2_hits=l2_hits,
            l3_hits=0, not_found=0, l0_time=l0_time, l1_time=l1_time, l2_time=l2_time,
            l3_time=0.0, total_time=time.time() - t0_total
        )
        if verbose:
            print(f"\n✅ All particles found in L0+L1+L2!")
        return element_ids, block_ids, stats

    # ========================================================================
    # LEVEL 3: Sequential neighbor block search (L2-miss particles)
    # ========================================================================
    # L3 searches 26 neighbor blocks with full padded arrays
    # Vectorizing would cause OOM (1.91 GiB allocation on 4GB GPU)
    # Since L3 is <1% of particles, sequential is acceptable

    if verbose:
        print(f"\n🔍 L3: Searching 26-neighbor blocks (sequential over {n_l2_miss:,} particles)...")
        print(f"   ⚠️  L3 cannot be vectorized (would cause OOM on 4GB GPU)")

    t3 = time.time()

    # Create heavy block flags array
    heavy_flags = jnp.zeros(n_blocks, dtype=jnp.bool_)
    for hb_id in block_classification.heavy_blocks:
        heavy_flags = heavy_flags.at[hb_id].set(True)

    # Sequential search for L2-miss particles
    for idx in l2_miss_indices:
        block_id = int(cached_block_ids[idx])
        if block_id < 0:
            continue

        pos = positions_jax[idx]
        neighbors_26 = jnp.array(block_neighbors_26[block_id], dtype=jnp.int32)

        elem_id = search_level3_neighbor_blocks(
            pos, block_id, neighbors_26, heavy_flags,
            padded_elements_jax, padded_counts_jax,
            node_pos_jax, connectivity_jax
        )

        if int(elem_id) >= 0:
            element_ids[idx] = int(elem_id)
            # Find which neighbor block contains this element
            for neighbor_id in block_neighbors_26[block_id]:
                if neighbor_id < 0:
                    continue
                block_ids[idx] = int(neighbor_id)
                break

    l3_time = time.time() - t3

    # Count L3 hits
    l3_mask = (element_ids >= 0) & ~found_mask  # New hits since L2
    l3_indices = np.where(l3_mask)[0]
    l3_hits = len(l3_indices)

    if verbose:
        print(f"   ✓ L3 hits: {l3_hits:,}/{n_l2_miss:,} ({100*l3_hits/n_l2_miss:.1f}% of L2-miss) in {l3_time:.2f}s")

    # ========================================================================
    # FINAL STATISTICS
    # ========================================================================
    total_time = time.time() - t0_total
    not_found = np.sum(element_ids < 0)

    stats = SearchStats(
        n_particles=n_particles,
        l0_hits=l0_hits,
        l1_hits=l1_hits,
        l2_hits=l2_hits,
        l3_hits=l3_hits,
        not_found=not_found,
        l0_time=l0_time,
        l1_time=l1_time,
        l2_time=l2_time,
        l3_time=l3_time,
        total_time=total_time
    )

    if verbose:
        print(f"\n{'='*80}")
        print(f"VECTORIZED SEARCH RESULTS")
        print(f"{'='*80}")
        print(f"\n📊 Hit Rates:")
        print(f"   L0 (cached):       {l0_hits:8,} ({100*l0_hits/n_particles:5.1f}%)")
        print(f"   L1 (neighbors):    {l1_hits:8,} ({100*l1_hits/n_particles:5.1f}%)")
        print(f"   L2 (block):        {l2_hits:8,} ({100*l2_hits/n_particles:5.1f}%)")
        print(f"   L3 (26-neighbors): {l3_hits:8,} ({100*l3_hits/n_particles:5.1f}%)")
        print(f"   Not found:         {not_found:8,} ({100*not_found/n_particles:5.1f}%)")

        total_found = l0_hits + l1_hits + l2_hits + l3_hits
        print(f"\n   Total found:       {total_found:8,} ({100*total_found/n_particles:5.1f}%)")

        print(f"\n⏱️  Timing:")
        print(f"   L0: {l0_time:6.2f}s ({100*l0_time/total_time:5.1f}%)")
        print(f"   L1: {l1_time:6.2f}s ({100*l1_time/total_time:5.1f}%)")
        print(f"   L2: {l2_time:6.2f}s ({100*l2_time/total_time:5.1f}%)")
        print(f"   L3: {l3_time:6.2f}s ({100*l3_time/total_time:5.1f}%)")
        print(f"   Total: {total_time:.2f}s")

        throughput = n_particles / total_time
        print(f"\n⚡ Throughput: {throughput:,.0f} particles/second")

        if throughput >= 5000:
            print(f"   ✅ EXCELLENT (>5,000 p/s target)")
        elif throughput >= 1000:
            print(f"   ✅ GOOD (>1,000 p/s minimum)")
        else:
            print(f"   ⚠️  Below 1,000 p/s target")

        print(f"{'='*80}\n")

    return element_ids, block_ids, stats


if __name__ == "__main__":
    """Test multi-level search with synthetic data."""
    print("Testing Multi-Level Search Orchestrator...")

    # Create synthetic mesh (small tetrahedral mesh)
    print("\nCreating synthetic mesh...")
    n_nodes = 100
    n_elements = 200
    n_blocks = 4

    node_positions = np.random.uniform(0, 1, (n_nodes, 3)).astype(np.float32)
    connectivity = np.random.randint(0, n_nodes, (n_elements, 4), dtype=np.int32)

    # Create synthetic padded arrays
    max_elem_per_block = 100
    padded_elements = np.full((n_blocks, max_elem_per_block), -1, dtype=np.int32)
    padded_counts = np.array([50, 60, 45, 45], dtype=np.int32)

    for b in range(n_blocks):
        padded_elements[b, :padded_counts[b]] = np.arange(padded_counts[b])

    # Create synthetic element neighbors
    max_neighbors = 4
    element_neighbors = np.random.randint(-1, n_elements, (n_elements, max_neighbors), dtype=np.int32)

    # Block neighbors (simplified)
    block_neighbors_26 = np.full((n_blocks, 26), -1, dtype=np.int32)

    # Synthetic particles
    n_particles = 100
    particle_positions = np.random.uniform(0, 1, (n_particles, 3)).astype(np.float32)
    cached_element_ids = np.random.randint(0, n_elements, n_particles, dtype=np.int32)
    cached_block_ids = np.random.randint(0, n_blocks, n_particles, dtype=np.int32)

    # Mock classification (all light blocks)
    class MockClassification:
        def is_heavy(self, block_id):
            return False

    classification = MockClassification()

    print("\nRunning multi-level search...")
    element_ids, block_ids, stats = multi_level_search_batch(
        particle_positions,
        cached_element_ids,
        cached_block_ids,
        classification,
        padded_elements,
        padded_counts,
        element_neighbors,
        block_neighbors_26,
        None,  # No hash buckets for light blocks
        node_positions,
        connectivity,
        verbose=True
    )

    print(f"\nResults:")
    print(f"  Found: {np.sum(element_ids >= 0):,}/{n_particles}")
    print(f"  Not found: {np.sum(element_ids < 0):,}/{n_particles}")

    stats.print_detailed()

    print("\n✅ Multi-level search orchestrator test complete!")
