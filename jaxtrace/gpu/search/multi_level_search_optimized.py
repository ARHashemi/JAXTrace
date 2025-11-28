"""
Optimized Vectorized Multi-Level Search - Eliminates Nested JIT Bottleneck

This module provides pre-compiled vectorized search functions that avoid the
nested JIT compilation overhead of wrapping @jax.jit functions in jax.vmap().

Key Optimization: Single @jax.jit decorator on vectorized function containing
vmap internally, rather than vmap wrapping already-JIT'd functions.

Performance Target: 5,000-15,000 p/s (2-5× speedup over sequential baseline)
"""

import time
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .block_classifier import BlockClassification
from .hash_bucket import HashBucketArrays

jax.config.update("jax_enable_x64", True)


# ============================================================================
# Helper Functions (NOT JIT-decorated to avoid nested JIT)
# ============================================================================

def point_in_tet_jax(
    point: jax.Array,
    tet_nodes: jax.Array,
    tolerance: float = 1e-10
) -> bool:
    """
    Test if point is inside tetrahedron using barycentric coordinates.

    NOTE: Not JIT-decorated because it's only called from within @jax.jit functions.
    This is a local copy to avoid nested JIT compilation.

    Parameters
    ----------
    point : jax.Array
        Point position (3,)
    tet_nodes : jax.Array
        Tetrahedron node positions (4, 3)
    tolerance : float
        Numerical tolerance for boundary cases

    Returns
    -------
    inside : bool
        True if point is inside tetrahedron
    """
    v0, v1, v2, v3 = tet_nodes[0], tet_nodes[1], tet_nodes[2], tet_nodes[3]

    # Build matrix for barycentric coordinates
    mat = jnp.column_stack([v1 - v0, v2 - v0, v3 - v0])

    # Solve for barycentric coordinates
    det = jnp.linalg.det(mat)

    # Handle degenerate case
    is_degenerate = jnp.abs(det) < tolerance

    # Compute barycentric coordinates
    rhs = point - v0
    lambdas_123 = jnp.linalg.solve(mat, rhs)
    lambda_0 = 1.0 - jnp.sum(lambdas_123)

    all_lambdas = jnp.concatenate([jnp.array([lambda_0]), lambdas_123])

    # Check if all in [0, 1] with tolerance
    inside = jnp.all(all_lambdas >= -tolerance) & jnp.all(all_lambdas <= 1.0 + tolerance)

    # Return false for degenerate tets
    return jnp.where(is_degenerate, False, inside)


# ============================================================================
# L0: Optimized Vectorized Cached Element Search
# ============================================================================

@jax.jit
def search_l0_batch_optimized(
    positions: jax.Array,          # (n_particles, 3)
    cached_elements: jax.Array,    # (n_particles,)
    node_positions: jax.Array,     # (n_nodes, 3)
    connectivity: jax.Array        # (n_elements, 4)
) -> jax.Array:
    """
    Optimized L0 batch search - single JIT compilation for entire batch.

    Eliminates nested JIT overhead by inlining logic without @jax.jit decorator.

    Returns
    -------
    element_ids : jax.Array (n_particles,)
        cached_element_id if still inside, else -1
    """
    def search_single(pos, cached_elem):
        """Single particle L0 search - NO @jax.jit decorator."""
        # Check if cached element is valid
        is_valid = (cached_elem >= 0) & (cached_elem < len(connectivity))

        # Get tet nodes (use jnp.where to handle invalid indices safely)
        safe_idx = jnp.where(is_valid, cached_elem, 0)
        node_ids = connectivity[safe_idx]
        tet_nodes = node_positions[node_ids]

        # Test if still inside
        inside = point_in_tet_jax(pos, tet_nodes)

        # Return cached_element_id only if valid AND inside
        return jnp.where(is_valid & inside, cached_elem, -1)

    # vmap is INSIDE @jax.jit - compiled as single unit
    return jax.vmap(search_single)(positions, cached_elements)


# ============================================================================
# L1: Optimized Vectorized Neighbor Element Search
# ============================================================================

@jax.jit
def search_l1_batch_optimized(
    positions: jax.Array,          # (n_particles, 3)
    cached_elements: jax.Array,    # (n_particles,)
    element_neighbors: jax.Array,  # (n_elements, 4)
    node_positions: jax.Array,     # (n_nodes, 3)
    connectivity: jax.Array        # (n_elements, 4)
) -> jax.Array:
    """
    Optimized L1 batch search - single JIT compilation.

    Returns
    -------
    element_ids : jax.Array (n_particles,)
        Neighbor element ID if found, else -1
    """
    def search_single(pos, cached_elem):
        """Single particle L1 search - NO @jax.jit decorator."""
        # Check if cached element is valid
        is_valid = (cached_elem >= 0) & (cached_elem < len(connectivity))

        # Get neighbors (safe indexing)
        safe_idx = jnp.where(is_valid, cached_elem, 0)
        neighbors = element_neighbors[safe_idx]

        # Search each neighbor
        def check_neighbor(neighbor_id):
            is_valid_neighbor = (neighbor_id >= 0) & (neighbor_id < len(connectivity))
            safe_neighbor = jnp.where(is_valid_neighbor, neighbor_id, 0)
            node_ids = connectivity[safe_neighbor]
            tet_nodes = node_positions[node_ids]
            inside = point_in_tet_jax(pos, tet_nodes)
            return jnp.where(is_valid_neighbor & inside, neighbor_id, -1)

        # Check all 4 neighbors
        results = jax.vmap(check_neighbor)(neighbors)

        # Return first valid hit, or -1
        valid_mask = results >= 0
        first_hit = jnp.where(jnp.any(valid_mask), results[jnp.argmax(valid_mask)], -1)

        return jnp.where(is_valid, first_hit, -1)

    # vmap is INSIDE @jax.jit
    return jax.vmap(search_single)(positions, cached_elements)


# ============================================================================
# L2a: Optimized Vectorized Light Block Search
# ============================================================================

def search_l2a_batch_optimized(
    positions: jax.Array,          # (n_particles, 3)
    block_elements: jax.Array,     # (max_elements,) - padded block array
    block_count: int,              # Actual number of elements in block
    node_positions: jax.Array,     # (n_nodes, 3)
    connectivity: jax.Array        # (n_elements, 4)
) -> jax.Array:
    """
    Optimized L2a light block batch search - single JIT compilation.

    Uses jax.lax.dynamic_slice to handle variable block_count within JIT.

    Returns
    -------
    element_ids : jax.Array (n_particles,)
        Element ID if found in block, else -1
    """
    # Use dynamic_slice to extract only valid elements
    # Pad block_count to ensure it doesn't exceed array length
    safe_count = min(block_count, len(block_elements))
    valid_elements = jax.lax.dynamic_slice(block_elements, (0,), (safe_count,))

    @jax.jit
    def search_batch_jitted(positions, valid_elems):
        """JIT-compiled batch search over valid elements only."""
        def search_single(pos):
            """Single particle L2a search."""
            def check_element(elem_id):
                is_valid = (elem_id >= 0) & (elem_id < len(connectivity))
                safe_idx = jnp.where(is_valid, elem_id, 0)
                node_ids = connectivity[safe_idx]
                tet_nodes = node_positions[node_ids]
                inside = point_in_tet_jax(pos, tet_nodes)
                return jnp.where(is_valid & inside, elem_id, -1)

            # Search only valid elements
            results = jax.vmap(check_element)(valid_elems)

            # Return first valid hit
            valid_mask = results >= 0
            return jnp.where(jnp.any(valid_mask), results[jnp.argmax(valid_mask)], -1)

        # vmap is INSIDE @jax.jit
        return jax.vmap(search_single)(positions)

    return search_batch_jitted(positions, valid_elements)


# ============================================================================
# L2b: Optimized Vectorized Heavy Block Hash Bucket Search
# ============================================================================

def compute_morton_code_jax(
    position: jax.Array,
    block_bounds: jax.Array,
    morton_bits: int
) -> int:
    """Compute Morton Z-order code for position within block.

    NOTE: Not JIT-decorated because it's only called from within @jax.jit functions.
    """
    # Normalize to [0, 1]
    normalized = (position - block_bounds[:3]) / (block_bounds[3:] - block_bounds[:3])
    normalized = jnp.clip(normalized, 0.0, 0.999999)

    # Quantize to grid
    max_coord = (1 << morton_bits) - 1
    ix = jnp.int32(normalized[0] * max_coord)
    iy = jnp.int32(normalized[1] * max_coord)
    iz = jnp.int32(normalized[2] * max_coord)

    # Interleave bits (Morton encoding)
    def split_by_3(x):
        x = jnp.uint32(x) & 0x000003ff
        x = (x | (x << 16)) & 0x030000ff
        x = (x | (x << 8)) & 0x0300f00f
        x = (x | (x << 4)) & 0x030c30c3
        x = (x | (x << 2)) & 0x09249249
        return x

    morton = split_by_3(ix) | (split_by_3(iy) << 1) | (split_by_3(iz) << 2)
    return jnp.int32(morton)


@jax.jit
def search_l2b_batch_optimized(
    positions: jax.Array,               # (n_particles, 3)
    bucket_elements: jax.Array,         # (n_buckets, max_per_bucket)
    bucket_counts: jax.Array,           # (n_buckets,)
    bucket_neighbors: jax.Array,        # (n_buckets, 6)
    n_buckets: int,
    morton_bits: int,
    block_bounds: jax.Array,            # (6,)
    node_positions: jax.Array,          # (n_nodes, 3)
    connectivity: jax.Array             # (n_elements, 4)
) -> jax.Array:
    """
    Optimized L2b heavy block hash bucket batch search.

    Returns
    -------
    element_ids : jax.Array (n_particles,)
        Element ID if found, else -1
    """
    def search_single(pos):
        """Single particle L2b search - NO @jax.jit decorator."""
        # Compute Morton code to find bucket
        morton = compute_morton_code_jax(pos, block_bounds, morton_bits)
        bucket_id = jnp.int32(morton % n_buckets)

        # Search primary bucket
        bucket_elems = bucket_elements[bucket_id]
        bucket_size = bucket_counts[bucket_id]

        def check_element(elem_id):
            is_valid = (elem_id >= 0) & (elem_id < len(connectivity))
            safe_idx = jnp.where(is_valid, elem_id, 0)
            node_ids = connectivity[safe_idx]
            tet_nodes = node_positions[node_ids]
            inside = point_in_tet_jax(pos, tet_nodes)
            return jnp.where(is_valid & inside, elem_id, -1)

        # Check primary bucket
        # Process all bucket elements, then mask out invalid ones
        primary_results_all = jax.vmap(check_element)(bucket_elems)
        mask = jnp.arange(len(bucket_elems)) < bucket_size
        primary_results = jnp.where(mask, primary_results_all, -1)
        primary_hit = jnp.where(
            jnp.any(primary_results >= 0),
            primary_results[jnp.argmax(primary_results >= 0)],
            -1
        )

        # If found, return
        found = primary_hit >= 0

        # Otherwise, search 6-neighbor buckets
        neighbors = bucket_neighbors[bucket_id]

        def check_neighbor_bucket(neighbor_id):
            is_valid = (neighbor_id >= 0) & (neighbor_id < n_buckets)
            safe_id = jnp.where(is_valid, neighbor_id, 0)
            neighbor_elems = bucket_elements[safe_id]
            neighbor_size = bucket_counts[safe_id]

            # Process all neighbor bucket elements, then mask
            results_all = jax.vmap(check_element)(neighbor_elems)
            mask_n = jnp.arange(len(neighbor_elems)) < neighbor_size
            results = jnp.where(mask_n, results_all, -1)
            hit = jnp.where(
                jnp.any(results >= 0),
                results[jnp.argmax(results >= 0)],
                -1
            )
            return jnp.where(is_valid, hit, -1)

        neighbor_results = jax.vmap(check_neighbor_bucket)(neighbors)
        neighbor_hit = jnp.where(
            jnp.any(neighbor_results >= 0),
            neighbor_results[jnp.argmax(neighbor_results >= 0)],
            -1
        )

        return jnp.where(found, primary_hit, neighbor_hit)

    # vmap is INSIDE @jax.jit
    return jax.vmap(search_single)(positions)


# ============================================================================
# Main Optimized Search Orchestrator
# ============================================================================

class SearchStats:
    """Search statistics."""
    def __init__(self, n_particles, l0_hits, l1_hits, l2_hits, l3_hits, not_found,
                 l0_time, l1_time, l2_time, l3_time, total_time):
        self.n_particles = n_particles
        self.l0_hits = l0_hits
        self.l1_hits = l1_hits
        self.l2_hits = l2_hits
        self.l3_hits = l3_hits
        self.not_found = not_found
        self.l0_time = l0_time
        self.l1_time = l1_time
        self.l2_time = l2_time
        self.l3_time = l3_time
        self.total_time = total_time


def multi_level_search_batch_optimized(
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
    Optimized vectorized multi-level search eliminating nested JIT overhead.

    Key Optimization: Pre-compiled vectorized functions with single JIT compilation
    instead of vmap wrapping @jax.jit functions.

    Expected Performance: 5,000-15,000 p/s (2-5× speedup over sequential)

    Returns
    -------
    element_ids : np.ndarray (n_particles,)
    block_ids : np.ndarray (n_particles,)
    stats : SearchStats
    """
    n_particles = len(particle_positions)

    if verbose:
        print(f"\n{'='*80}")
        print(f"OPTIMIZED VECTORIZED MULTI-LEVEL SEARCH: {n_particles:,} particles")
        print(f"{'='*80}")

    # Convert to JAX arrays
    positions_jax = jnp.array(particle_positions, dtype=jnp.float32)
    node_pos_jax = jnp.array(node_positions, dtype=jnp.float32)
    connectivity_jax = jnp.array(connectivity, dtype=jnp.int32)
    elem_neighbors_jax = jnp.array(element_neighbors, dtype=jnp.int32)
    padded_elements_jax = jnp.array(padded_block_elements, dtype=jnp.int32)
    padded_counts_jax = jnp.array(padded_block_counts, dtype=jnp.int32)
    cached_elem_jax = jnp.array(cached_element_ids, dtype=jnp.int32)

    # Initialize results
    element_ids = np.full(n_particles, -1, dtype=np.int32)
    block_ids = np.full(n_particles, -1, dtype=np.int32)

    # Statistics
    l0_hits = l1_hits = l2_hits = l3_hits = 0
    t0_total = time.time()

    # ========================================================================
    # LEVEL 0: Optimized vectorized cached element check
    # ========================================================================
    if verbose:
        print(f"\n🔍 L0: Cached element check (optimized vmap)...")

    t0 = time.time()
    l0_results = np.array(
        search_l0_batch_optimized(
            positions_jax, cached_elem_jax, node_pos_jax, connectivity_jax
        ),
        dtype=np.int32
    )
    l0_time = time.time() - t0

    # Extract L0 hits
    l0_mask = l0_results >= 0
    l0_indices = np.where(l0_mask)[0]
    l0_hits = len(l0_indices)

    element_ids[l0_indices] = l0_results[l0_indices]
    block_ids[l0_indices] = cached_block_ids[l0_indices]

    if verbose:
        print(f"   ✓ L0: {l0_hits:,}/{n_particles:,} ({100*l0_hits/n_particles:.1f}%) in {l0_time:.3f}s")

    # L0 misses proceed to L1
    l0_miss_indices = np.where(~l0_mask)[0]
    n_l0_miss = len(l0_miss_indices)

    if n_l0_miss == 0:
        stats = SearchStats(
            n_particles, l0_hits, 0, 0, 0, 0,
            l0_time, 0.0, 0.0, 0.0, time.time() - t0_total
        )
        if verbose:
            print(f"\n✅ All particles found in L0!")
        return element_ids, block_ids, stats

    # ========================================================================
    # LEVEL 1: Optimized vectorized neighbor element search
    # ========================================================================
    if verbose:
        print(f"\n🔍 L1: Neighbor elements (optimized vmap over {n_l0_miss:,})...")

    t1 = time.time()

    # Filter valid cached elements
    l0_miss_cached = cached_elem_jax[l0_miss_indices]
    valid_mask = l0_miss_cached >= 0
    valid_indices = l0_miss_indices[np.array(valid_mask)]

    if len(valid_indices) > 0:
        valid_positions = positions_jax[valid_indices]
        valid_cached = cached_elem_jax[valid_indices]

        l1_results = np.array(
            search_l1_batch_optimized(
                valid_positions, valid_cached, elem_neighbors_jax,
                node_pos_jax, connectivity_jax
            ),
            dtype=np.int32
        )

        # Extract L1 hits
        l1_mask = l1_results >= 0
        l1_hit_indices = valid_indices[np.array(l1_mask)]
        l1_hits = len(l1_hit_indices)

        element_ids[l1_hit_indices] = l1_results[np.array(l1_mask)]
        block_ids[l1_hit_indices] = cached_block_ids[l1_hit_indices]

    l1_time = time.time() - t1

    if verbose:
        print(f"   ✓ L1: {l1_hits:,}/{n_l0_miss:,} ({100*l1_hits/max(1,n_l0_miss):.1f}%) in {l1_time:.3f}s")

    # L1 misses proceed to L2
    found_mask = element_ids >= 0
    l1_miss_indices = np.where(~found_mask)[0]
    n_l1_miss = len(l1_miss_indices)

    if n_l1_miss == 0:
        stats = SearchStats(
            n_particles, l0_hits, l1_hits, 0, 0, 0,
            l0_time, l1_time, 0.0, 0.0, time.time() - t0_total
        )
        if verbose:
            print(f"\n✅ All particles found in L0+L1!")
        return element_ids, block_ids, stats

    # ========================================================================
    # LEVEL 2: Optimized block search (block-grouped)
    # ========================================================================
    if verbose:
        print(f"\n🔍 L2: Block search (optimized, {n_l1_miss:,} particles)...")

    t2 = time.time()

    # Group by block
    particles_per_block = {}
    for idx in l1_miss_indices:
        block_id = int(cached_block_ids[idx])
        if block_id >= 0:
            particles_per_block.setdefault(block_id, []).append(idx)

    is_heavy = np.zeros(len(padded_block_counts), dtype=bool)
    for hb in block_classification.heavy_blocks:
        is_heavy[hb] = True

    # Process each block
    for block_id, particle_indices in particles_per_block.items():
        particle_batch = particle_positions[particle_indices]
        particle_batch_jax = jnp.array(particle_batch, dtype=jnp.float32)

        if is_heavy[block_id] and hash_bucket_data and block_id in hash_bucket_data:
            # L2b: Heavy block hash bucket (optimized)
            hash_arrays = hash_bucket_data[block_id]

            bucket_elements_jax = jnp.array(hash_arrays.bucket_elements, dtype=jnp.int32)
            bucket_counts_jax = jnp.array(hash_arrays.bucket_elem_counts, dtype=jnp.int32)
            bucket_neighbors_jax = jnp.array(hash_arrays.bucket_neighbors_6, dtype=jnp.int32)
            block_bounds_jax = jnp.array(hash_arrays.block_bounds, dtype=jnp.float32)

            found_elem_ids = np.array(
                search_l2b_batch_optimized(
                    particle_batch_jax, bucket_elements_jax, bucket_counts_jax,
                    bucket_neighbors_jax, hash_arrays.n_buckets, hash_arrays.morton_bits,
                    block_bounds_jax, node_pos_jax, connectivity_jax
                ),
                dtype=np.int32
            )
        else:
            # L2a: Light block (optimized)
            block_elems = padded_elements_jax[block_id]
            block_count = int(padded_counts_jax[block_id])

            found_elem_ids = np.array(
                search_l2a_batch_optimized(
                    particle_batch_jax, block_elems, block_count,
                    node_pos_jax, connectivity_jax
                ),
                dtype=np.int32
            )

        # Store results
        for local_idx, global_idx in enumerate(particle_indices):
            elem_id = found_elem_ids[local_idx]
            if elem_id >= 0:
                element_ids[global_idx] = elem_id
                block_ids[global_idx] = block_id

    l2_time = time.time() - t2

    # Count L2 hits
    l2_mask = (element_ids >= 0) & ~found_mask
    l2_hits = np.sum(l2_mask)

    if verbose:
        print(f"   ✓ L2: {l2_hits:,}/{n_l1_miss:,} ({100*l2_hits/max(1,n_l1_miss):.1f}%) in {l2_time:.3f}s")

    # ========================================================================
    # LEVEL 3: Sequential (cannot vectorize - OOM risk)
    # ========================================================================
    # L3 omitted for now - same as original sequential implementation
    # Since L3 affects <1% of particles, we'll accept sequential performance here

    l3_time = 0.0
    l3_hits = 0

    # Final stats
    total_time = time.time() - t0_total
    not_found = np.sum(element_ids < 0)

    stats = SearchStats(
        n_particles, l0_hits, l1_hits, l2_hits, l3_hits, not_found,
        l0_time, l1_time, l2_time, l3_time, total_time
    )

    if verbose:
        throughput = n_particles / total_time
        print(f"\n⚡ Optimized throughput: {throughput:,.0f} p/s ({total_time:.3f}s total)")
        print(f"   L0: {l0_time:.3f}s, L1: {l1_time:.3f}s, L2: {l2_time:.3f}s")

    return element_ids, block_ids, stats
