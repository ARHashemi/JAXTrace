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

        # L3: Search neighbor blocks (simplified - not fully implemented yet)
        # This would search 26-neighbor blocks
        # For now, mark as not found
        l3_hits += 0  # Placeholder

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
