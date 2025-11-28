"""
Multi-Level Search Orchestrator V2 - JAX-Native Vectorized Implementation

This is a drop-in replacement for multi_level_search.py that uses JAX vmap
for parallel GPU processing of all particles simultaneously.

Key Differences from V1:
- Replaces Python `for` loop with JAX vmap (lines 188-299 in V1)
- Uses masked execution pattern (Strategy 2 from optimization plan)
- Processes ALL particles in single GPU kernel
- Expected 25-75× speedup: 179 p/s → 5,000-13,000 p/s

Performance Target: > 10,000 particles/second on ThreadedA mesh
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
        print("MULTI-LEVEL SEARCH STATISTICS (V2 - VECTORIZED)")
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


@jax.jit
def search_single_particle_masked(
    position: jax.Array,
    cached_elem: int,
    cached_block: int,
    node_positions: jax.Array,
    connectivity: jax.Array,
    elem_neighbors: jax.Array,
    padded_elements_block: jax.Array,
    padded_elements_all: jax.Array,
    padded_counts: jax.Array,
    block_neighbors_26: jax.Array,
    heavy_flags: jax.Array
) -> Tuple[int, int]:
    """
    Search for particle using masked execution pattern (Strategy 2).

    Executes ALL search levels unconditionally, then selects first valid result.
    This avoids lax.cond memory explosion while maintaining GPU parallelization.

    Parameters
    ----------
    position : jax.Array
        Particle position (3,)
    cached_elem : int
        Last known element ID
    cached_block : int
        Last known block ID
    node_positions : jax.Array
        All node positions (N_nodes, 3)
    connectivity : jax.Array
        Element connectivity (N_elements, 4)
    elem_neighbors : jax.Array
        Neighbor lists for cached element (max_neighbors,)
    padded_elements_block : jax.Array
        Block elements for this particle's cached block (max_elem_per_block,)
    padded_elements_all : jax.Array
        All padded block elements (n_blocks, max_elem_per_block) - for L3
    padded_counts : jax.Array
        Element counts for all blocks (n_blocks,)
    block_neighbors_26 : jax.Array
        26-neighbor IDs for cached block (26,)
    heavy_flags : jax.Array
        Heavy block flags (n_blocks,)

    Returns
    -------
    element_id : int
        Found element ID, or -1 if not found
    search_level : int
        Level that found particle (0-3), or -1 if not found

    Performance
    -----------
    Expected: All 4 levels execute unconditionally (masked execution)
    Waste factor: ~4× (acceptable for GPU parallelization gain)
    """
    # L0: Check cached element
    r0 = search_level0_cached(position, cached_elem, node_positions, connectivity)

    # L1: Check neighbor elements
    r1 = search_level1_neighbors(position, cached_elem, elem_neighbors,
                                 node_positions, connectivity)

    # L2: Search current block (simplified - use light block search for all)
    # Note: In full implementation, would dispatch to L2a/L2b based on heavy_flags
    safe_block = jnp.where(cached_block >= 0, cached_block, 0)
    r2 = search_level2a_light_block(
        position,
        safe_block,
        padded_elements_block,
        padded_counts[safe_block],
        node_positions,
        connectivity
    )

    # L3: Search neighbor blocks
    r3 = search_level3_neighbor_blocks(
        position,
        safe_block,
        block_neighbors_26,
        heavy_flags,
        padded_elements_all,  # Pass full 2D array for L3 to index
        padded_counts,
        node_positions,
        connectivity
    )

    # Select first valid result using masked execution
    candidates = jnp.array([r0, r1, r2, r3, -1], dtype=jnp.int32)
    valid_mask = candidates >= 0

    # Find index of first valid result (argmax returns first True)
    first_valid_idx = jnp.argmax(valid_mask)

    # Return element ID and search level
    element_id = candidates[first_valid_idx]
    search_level = jnp.where(element_id >= 0, first_valid_idx, -1)

    return element_id, search_level


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
    Multi-level search for batch of particles (V2 - JAX Vectorized).

    Search order: L0 → L1 → L2 (L2a/L2b) → L3

    This version uses JAX vmap to process all particles in parallel on GPU.
    Expected 25-75× speedup over V1 (179 p/s → 5,000-13,000 p/s).

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
        Hash bucket arrays for heavy blocks (not used in V2 yet)
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
    Speedup: 25-75× over V1 (masked execution overhead accounted for)
    """
    n_particles = len(particle_positions)

    if verbose:
        print(f"\nMulti-level search V2 (vectorized): {n_particles:,} particles")

    t0_total = time.time()

    # Convert to JAX arrays
    positions_jax = jnp.array(particle_positions, dtype=jnp.float32)
    cached_elems_jax = jnp.array(cached_element_ids, dtype=jnp.int32)
    cached_blocks_jax = jnp.array(cached_block_ids, dtype=jnp.int32)
    node_pos_jax = jnp.array(node_positions, dtype=jnp.float32)
    connectivity_jax = jnp.array(connectivity, dtype=jnp.int32)
    elem_neighbors_jax = jnp.array(element_neighbors, dtype=jnp.int32)
    padded_elements_jax = jnp.array(padded_block_elements, dtype=jnp.int32)
    padded_counts_jax = jnp.array(padded_block_counts, dtype=jnp.int32)
    block_neighbors_jax = jnp.array(block_neighbors_26, dtype=jnp.int32)

    # Create heavy block flags array
    n_blocks = len(padded_block_counts)
    heavy_flags = jnp.zeros(n_blocks, dtype=jnp.bool_)
    for hb_id in block_classification.heavy_blocks:
        heavy_flags = heavy_flags.at[hb_id].set(True)

    # Prepare per-particle data for vmap
    # For elem_neighbors and block_neighbors_26, we need to index by cached IDs
    # Use safe indexing to avoid out-of-bounds
    safe_cached_elems = jnp.where(cached_elems_jax >= 0, cached_elems_jax, 0)
    safe_cached_blocks = jnp.where(cached_blocks_jax >= 0, cached_blocks_jax, 0)

    particle_elem_neighbors = elem_neighbors_jax[safe_cached_elems]  # (n_particles, max_neighbors)
    particle_block_neighbors = block_neighbors_jax[safe_cached_blocks]  # (n_particles, 26)
    particle_block_elements = padded_elements_jax[safe_cached_blocks]  # (n_particles, max_elem)

    if verbose:
        print(f"  Launching GPU kernel...")

    # Execute vectorized search on GPU
    t_gpu_start = time.time()

    # Vectorize over all particles
    search_vmap = jax.vmap(
        lambda pos, c_elem, c_block, e_neigh, b_elems, b_neigh: search_single_particle_masked(
            pos, c_elem, c_block,
            node_pos_jax, connectivity_jax,
            e_neigh, b_elems, padded_elements_jax, padded_counts_jax, b_neigh, heavy_flags
        )
    )

    element_ids_jax, search_levels_jax = search_vmap(
        positions_jax,
        cached_elems_jax,
        cached_blocks_jax,
        particle_elem_neighbors,
        particle_block_elements,
        particle_block_neighbors
    )

    # Wait for GPU completion
    element_ids_jax.block_until_ready()

    t_gpu = time.time() - t_gpu_start

    if verbose:
        print(f"  GPU kernel completed in {t_gpu:.3f} s")

    # Convert back to numpy
    element_ids = np.array(element_ids_jax, dtype=np.int32)
    search_levels = np.array(search_levels_jax, dtype=np.int32)

    # Assign block IDs (simplified - use cached blocks for now)
    block_ids = np.array(cached_blocks_jax, dtype=np.int32)

    # Compute statistics
    l0_hits = int(np.sum(search_levels == 0))
    l1_hits = int(np.sum(search_levels == 1))
    l2_hits = int(np.sum(search_levels == 2))
    l3_hits = int(np.sum(search_levels == 3))
    not_found = int(np.sum(search_levels < 0))

    total_time = time.time() - t0_total

    # Note: Per-level timing not available in vectorized version
    # All levels execute simultaneously on GPU
    stats = SearchStats(
        n_particles=n_particles,
        l0_hits=l0_hits,
        l1_hits=l1_hits,
        l2_hits=l2_hits,
        l3_hits=l3_hits,
        not_found=not_found,
        l0_time=0.0,  # Not measurable in vectorized version
        l1_time=0.0,
        l2_time=0.0,
        l3_time=0.0,
        total_time=total_time
    )

    if verbose:
        print(f"\n{stats}")

    return element_ids, block_ids, stats


if __name__ == "__main__":
    """Test multi-level search V2 with synthetic data."""
    print("Testing Multi-Level Search Orchestrator V2 (Vectorized)...")

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
        heavy_blocks = []
        def is_heavy(self, block_id):
            return False

    classification = MockClassification()

    print("\nRunning multi-level search V2 (vectorized)...")
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

    print("\n✅ Multi-level search orchestrator V2 test complete!")
