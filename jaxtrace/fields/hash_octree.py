#!/usr/bin/env python3
"""
GPU-Native Hash Octree for JAX Compilation (Phase 3).

This module implements a non-hierarchical hash-based octree that avoids
JAX memory explosion issues caused by tree traversal and dynamic slicing.

Key Features:
- O(1) hash table lookup instead of O(log n) tree traversal
- Flattened element lists with static array shapes (JAX-compilable)
- Morton codes as hash keys (from Phase 2)
- Bounded linear probing (max 20 probes) for collision resolution
- No io_callback, no lax.scan, no dynamic slicing

Architecture:
- Hash Table: Morton code → (element_list_start, element_list_length)
- Flattened Elements: Single static array of all element IDs
- Prime-sized table: 1.3× load factor for good performance

References:
- GPU_OCTREE_IMPLEMENTATION_ROADMAP.md (Phase 3)
- Critical_JAX_Memory_Issues_Phase3_Hash.md
- Details_of_hash_octree_without_hierarchi.md
"""

import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, Optional
import numba

from .morton_code import encode_morton_3d, decode_morton_3d, encode_morton_3d_numpy  # Phase 3B


# Constants
EMPTY_SLOT = np.uint64(0xFFFFFFFFFFFFFFFF)  # Marker for empty hash table slots
MAX_PROBES = 200  # Maximum linear probing attempts (high for worst-case with large meshes)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class HashOctree:
    """
    Non-hierarchical hash-based octree for GPU-native JAX compilation.

    This structure replaces hierarchical tree traversal with O(1) hash lookup,
    avoiding JAX memory explosion issues while enabling full GPU acceleration.

    Attributes:
        bbox_min: Domain minimum bounds [3]
        bbox_max: Domain maximum bounds [3]

        hash_table_size: Prime number size of hash table
        morton_keys: Morton codes in hash table [hash_table_size]
                     EMPTY_SLOT (0xFF...) indicates empty slot

        element_list_starts: Starting index in flattened_elements [hash_table_size]
        element_list_lengths: Number of elements for this key [hash_table_size]

        flattened_elements: All element IDs concatenated [total_elements]
        max_elements_per_cell: Maximum elements in any cell (for bounds checking)

        n_leaves: Number of actual leaf nodes (non-empty slots)
        load_factor: Actual load factor (n_leaves / hash_table_size)

    Memory Layout Example:
        Leaf 1: Morton=100, Elements=[5, 12, 18]
        Leaf 2: Morton=250, Elements=[3, 7]

        morton_keys = [100, 250, EMPTY, EMPTY, ...]
        element_list_starts = [0, 3, -1, -1, ...]
        element_list_lengths = [3, 2, 0, 0, ...]
        flattened_elements = [5, 12, 18, 3, 7, ...]
    """
    bbox_min: jnp.ndarray  # [3] float32
    bbox_max: jnp.ndarray  # [3] float32

    hash_table_size: int  # Prime number
    morton_keys: jnp.ndarray  # [hash_table_size] uint64

    element_list_starts: jnp.ndarray  # [hash_table_size] int32
    element_list_lengths: jnp.ndarray  # [hash_table_size] int32

    flattened_elements: jnp.ndarray  # [total_elements] int32
    max_elements_per_cell: int  # For bounds checking

    n_leaves: int  # Number of non-empty slots
    load_factor: float  # n_leaves / hash_table_size


# ============================================================================
# Prime Number Utilities
# ============================================================================

@numba.njit
def is_prime(n: int) -> bool:
    """
    Check if a number is prime using trial division.

    Args:
        n: Number to test

    Returns:
        True if n is prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    # Check odd divisors up to sqrt(n)
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2

    return True


@numba.njit
def next_prime(n: int) -> int:
    """
    Find the smallest prime number >= n.

    Args:
        n: Starting number

    Returns:
        Next prime number >= n

    Example:
        >>> next_prime(100)
        101
        >>> next_prime(101)
        101
    """
    if n < 2:
        return 2

    # Start with n if odd, otherwise n+1
    candidate = n if n % 2 == 1 else n + 1

    while not is_prime(candidate):
        candidate += 2

    return candidate


def compute_hash_table_size(n_leaves: int, target_load_factor: float = 0.77) -> int:
    """
    Compute prime-sized hash table for given number of leaves.

    Uses a target load factor (default 0.77 ≈ 1/1.3) for good performance.
    Lower load factor = fewer collisions but more memory.

    Args:
        n_leaves: Number of leaf nodes to store
        target_load_factor: Desired load factor (default 0.77)

    Returns:
        Prime number >= n_leaves / target_load_factor

    Example:
        >>> compute_hash_table_size(1000)  # 1000 / 0.77 ≈ 1299 → next prime
        1301
    """
    min_size = int(np.ceil(n_leaves / target_load_factor))
    return next_prime(min_size)


# ============================================================================
# Hash Function
# ============================================================================

@numba.njit
def hash_morton_scrambled(morton_code: np.uint64, table_size: int) -> int:
    """
    Scrambled hash function for Morton codes using MurmurHash3 finalizer.

    This breaks the spatial locality of Morton codes to prevent clustering.
    Morton codes have high spatial locality (nearby 3D positions → nearby codes),
    which causes massive clustering with simple modulo hashing.

    MurmurHash3 finalizer provides excellent avalanche properties:
    - Single bit change → ~50% of output bits change
    - Uniform distribution across hash table
    - Prevents primary clustering

    Args:
        morton_code: 64-bit Morton code from Phase 2
        table_size: Hash table size (must be prime)

    Returns:
        Hash bucket index in [0, table_size)

    References:
        - MurmurHash3: https://github.com/aappleby/smhasher
        - docs/HASH_TABLE_COLLISION_ANALYSIS.md
    """
    # MurmurHash3 finalizer (64-bit)
    h = np.uint64(morton_code)

    # First mix
    h ^= h >> np.uint64(33)
    h = (h * np.uint64(0xff51afd7ed558ccd)) & np.uint64(0xFFFFFFFFFFFFFFFF)

    # Second mix
    h ^= h >> np.uint64(33)
    h = (h * np.uint64(0xc4ceb9fe1a85ec53)) & np.uint64(0xFFFFFFFFFFFFFFFF)

    # Third mix
    h ^= h >> np.uint64(33)

    return int(h % np.uint64(table_size))


@numba.njit
def hash_morton(morton_code: np.uint64, table_size: int) -> int:
    """
    Legacy simple hash function (DEPRECATED - causes clustering).

    Kept for reference only. Use hash_morton_scrambled() instead.

    Args:
        morton_code: 64-bit Morton code from Phase 2
        table_size: Hash table size (must be prime)

    Returns:
        Hash bucket index in [0, table_size)
    """
    return int(morton_code % np.uint64(table_size))


# ============================================================================
# Hash Table Construction (CPU/Numba)
# ============================================================================

@numba.njit
def insert_with_linear_probing(
    morton_code: np.uint64,
    element_start: int,
    element_length: int,
    morton_keys: np.ndarray,
    element_list_starts: np.ndarray,
    element_list_lengths: np.ndarray,
    table_size: int
) -> bool:
    """
    Insert entry into hash table using linear probing with scrambled hashing.

    Uses hash_morton_scrambled() to break spatial locality and prevent clustering.

    Args:
        morton_code: Morton code key
        element_start: Starting index in flattened array
        element_length: Number of elements
        morton_keys: Hash table keys (modified in-place)
        element_list_starts: Element starts array (modified in-place)
        element_list_lengths: Element lengths array (modified in-place)
        table_size: Hash table size

    Returns:
        True if inserted successfully, False if table is full (>MAX_PROBES)
    """
    # Use scrambled hash to prevent clustering
    slot = hash_morton_scrambled(morton_code, table_size)

    for probe in range(MAX_PROBES):
        current_slot = (slot + probe) % table_size

        if morton_keys[current_slot] == EMPTY_SLOT:
            # Found empty slot
            morton_keys[current_slot] = morton_code
            element_list_starts[current_slot] = element_start
            element_list_lengths[current_slot] = element_length
            return True

    # Failed to insert after MAX_PROBES attempts
    return False


def build_hash_octree_from_leaves(
    leaf_morton_codes: np.ndarray,
    leaf_element_lists: list,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    target_load_factor: float = 0.77
) -> HashOctree:
    """
    Build hash octree from leaf nodes (CPU construction).

    This function is called during initialization to convert hierarchical
    octree leaf nodes into a flat hash table structure.

    Args:
        leaf_morton_codes: Morton codes for each leaf [n_leaves] uint64
        leaf_element_lists: List of element ID arrays (variable length)
        bbox_min: Domain minimum bounds [3]
        bbox_max: Domain maximum bounds [3]
        target_load_factor: Hash table load factor (default 0.77)

    Returns:
        HashOctree: Complete hash octree structure

    Example:
        >>> leaf_codes = np.array([100, 250, 500], dtype=np.uint64)
        >>> leaf_elements = [[5, 12, 18], [3, 7], [9]]
        >>> hash_octree = build_hash_octree_from_leaves(
        ...     leaf_codes, leaf_elements, bbox_min, bbox_max
        ... )
    """
    n_leaves = len(leaf_morton_codes)

    if n_leaves == 0:
        raise ValueError("Cannot build hash octree with zero leaves")

    # Compute hash table size (prime number)
    table_size = compute_hash_table_size(n_leaves, target_load_factor)

    # Initialize hash table arrays
    morton_keys = np.full(table_size, EMPTY_SLOT, dtype=np.uint64)
    element_list_starts = np.full(table_size, -1, dtype=np.int32)
    element_list_lengths = np.zeros(table_size, dtype=np.int32)

    # Flatten element lists
    total_elements = sum(len(elems) for elems in leaf_element_lists)
    flattened_elements = np.zeros(total_elements, dtype=np.int32)
    max_elements_per_cell = max(len(elems) for elems in leaf_element_lists)

    current_offset = 0
    for i, (morton_code, element_list) in enumerate(zip(leaf_morton_codes, leaf_element_lists)):
        n_elements = len(element_list)

        # Copy elements to flattened array
        flattened_elements[current_offset:current_offset + n_elements] = element_list

        # Insert into hash table
        success = insert_with_linear_probing(
            morton_code,
            current_offset,
            n_elements,
            morton_keys,
            element_list_starts,
            element_list_lengths,
            table_size
        )

        if not success:
            raise RuntimeError(
                f"Hash table insertion failed for leaf {i}/{n_leaves}. "
                f"Try increasing target_load_factor (current: {target_load_factor})"
            )

        current_offset += n_elements

    actual_load_factor = n_leaves / table_size

    # Convert to JAX arrays
    return HashOctree(
        bbox_min=jnp.array(bbox_min, dtype=jnp.float32),
        bbox_max=jnp.array(bbox_max, dtype=jnp.float32),
        hash_table_size=table_size,
        morton_keys=jnp.array(morton_keys),
        element_list_starts=jnp.array(element_list_starts),
        element_list_lengths=jnp.array(element_list_lengths),
        flattened_elements=jnp.array(flattened_elements),
        max_elements_per_cell=max_elements_per_cell,
        n_leaves=n_leaves,
        load_factor=actual_load_factor
    )


# ============================================================================
# Memory Statistics
# ============================================================================

def build_hash_octree_from_fine_octree(fine_octree, bbox_min, bbox_max, target_load_factor: float = 0.77) -> HashOctree:
    """
    Build hash octree from Phase 2 fine octree structure.

    Extracts leaf nodes from the hierarchical octree (OctreeFineLevel) and
    converts to flat hash table representation for GPU-native lookup.

    Args:
        fine_octree: OctreeFineLevel from Phase 2 (has node_morton_codes, node_children, etc.)
        bbox_min: Domain minimum bounds [3] (from coarse octree)
        bbox_max: Domain maximum bounds [3] (from coarse octree)
        target_load_factor: Hash table load factor (default 0.77)

    Returns:
        HashOctree: Flat hash table structure ready for JAX compilation

    Example:
        >>> fine = shared_octree.get_fine_level_for_timestep(0)
        >>> coarse = shared_octree.coarse_levels
        >>> hash_octree = build_hash_octree_from_fine_octree(
        ...     fine, coarse.bbox_min, coarse.bbox_max
        ... )
    """
    # Extract leaf nodes (nodes with no children, i.e., all children == -1)
    node_children_np = np.asarray(fine_octree.node_children, dtype=np.int32)
    is_leaf = np.all(node_children_np == -1, axis=1)
    leaf_indices = np.where(is_leaf)[0]

    if len(leaf_indices) == 0:
        raise ValueError("Fine octree has no leaf nodes")

    # Extract Morton codes for leaf nodes
    morton_codes_np = np.asarray(fine_octree.node_morton_codes, dtype=np.uint64)
    leaf_morton_codes = morton_codes_np[leaf_indices]

    # Extract element lists for leaf nodes
    element_lists_np = np.asarray(fine_octree.node_element_lists, dtype=np.int32)
    element_counts_np = np.asarray(fine_octree.node_element_counts, dtype=np.int32)

    leaf_element_lists = []
    total_elements = 0
    for idx in leaf_indices:
        count = int(element_counts_np[idx])
        elements = element_lists_np[idx, :count].tolist()
        leaf_element_lists.append(elements)
        total_elements += count

    if len(leaf_indices) > 0:
        print(f"      DEBUG build_hash_octree_from_fine_octree:")
        print(f"        Leaf nodes: {len(leaf_indices)}")
        print(f"        Element counts: {element_counts_np[leaf_indices]}")
        print(f"        Total elements: {total_elements}")

    # Use provided domain bounds
    bbox_min_np = np.asarray(bbox_min, dtype=np.float32)
    bbox_max_np = np.asarray(bbox_max, dtype=np.float32)

    # Build hash octree
    return build_hash_octree_from_leaves(
        leaf_morton_codes,
        leaf_element_lists,
        bbox_min_np,
        bbox_max_np,
        target_load_factor
    )


def get_hash_octree_memory_stats(hash_octree: HashOctree) -> dict:
    """
    Compute memory usage statistics for hash octree.

    Args:
        hash_octree: HashOctree structure

    Returns:
        Dictionary with memory statistics
    """
    morton_keys_bytes = hash_octree.morton_keys.nbytes
    starts_bytes = hash_octree.element_list_starts.nbytes
    lengths_bytes = hash_octree.element_list_lengths.nbytes
    elements_bytes = hash_octree.flattened_elements.nbytes

    hash_table_bytes = morton_keys_bytes + starts_bytes + lengths_bytes
    total_bytes = hash_table_bytes + elements_bytes

    return {
        'n_leaves': hash_octree.n_leaves,
        'hash_table_size': hash_octree.hash_table_size,
        'load_factor': hash_octree.load_factor,
        'total_elements': len(hash_octree.flattened_elements),
        'max_elements_per_cell': hash_octree.max_elements_per_cell,
        'morton_keys_mb': morton_keys_bytes / (1024 ** 2),
        'starts_mb': starts_bytes / (1024 ** 2),
        'lengths_mb': lengths_bytes / (1024 ** 2),
        'elements_mb': elements_bytes / (1024 ** 2),
        'hash_table_mb': hash_table_bytes / (1024 ** 2),
        'total_mb': total_bytes / (1024 ** 2),
    }


# ============================================================================
# JAX Hash Lookup (GPU-Compilable)
# ============================================================================

@jax.jit
def hash_morton_scrambled_jax(morton_code: jnp.ndarray, table_size: int) -> jnp.ndarray:
    """
    JAX version of scrambled hash function using MurmurHash3 finalizer.

    This is identical to hash_morton_scrambled() but uses JAX operations
    for GPU compilation.

    Args:
        morton_code: Morton code (int64)
        table_size: Hash table size (must be prime)

    Returns:
        Hash bucket index in [0, table_size)
    """
    # MurmurHash3 finalizer (64-bit)
    # Use numpy uint64 to avoid overflow with large constants
    h = jnp.uint64(morton_code)

    # MurmurHash3 constants (as uint64)
    C1 = jnp.uint64(0xff51afd7ed558ccd)
    C2 = jnp.uint64(0xc4ceb9fe1a85ec53)
    MASK = jnp.uint64(0xFFFFFFFFFFFFFFFF)

    # First mix
    h = h ^ (h >> jnp.uint64(33))
    h = (h * C1) & MASK

    # Second mix
    h = h ^ (h >> jnp.uint64(33))
    h = (h * C2) & MASK

    # Third mix
    h = h ^ (h >> jnp.uint64(33))

    return jnp.int32(h % jnp.uint64(table_size))


def hash_lookup_jax_from_morton(
    morton_code: int,
    hash_octree: HashOctree
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    GPU-compilable hash table lookup from Morton code (pure JAX).

    This is the core lookup function that works with pre-computed Morton codes,
    making it fully JIT-compilable and vmap-compatible.

    Args:
        morton_code: Pre-computed Morton code (int64 or uint64)
        hash_octree: HashOctree structure

    Returns:
        elements: Element IDs for this cell [max_elements_per_cell]
        n_elements: Number of valid elements (0 if not found)
    """
    morton_code_jax = jnp.uint64(morton_code)  # Use uint64 for consistency

    # Initial hash using scrambled hash
    table_size = hash_octree.hash_table_size
    initial_slot = hash_morton_scrambled_jax(morton_code_jax, table_size)

    # Linear probing using fori_loop (bounded iteration)
    def probe_step(probe, carry):
        """Single probe step in linear probing."""
        found_slot, found = carry

        # Compute current slot
        current_slot = (initial_slot + probe) % table_size

        # Check if this slot matches our key
        key_at_slot = hash_octree.morton_keys[current_slot]
        is_match = key_at_slot == morton_code_jax
        is_empty = key_at_slot == jnp.uint64(EMPTY_SLOT)

        # Update found_slot if we found a match (and haven't found one yet)
        found_slot = jnp.where(is_match & ~found, current_slot, found_slot)

        # Update found flag if we found a match
        found = found | is_match

        # Early exit if found or empty (but fori_loop doesn't support early exit,
        # so we just track the flag and ignore subsequent iterations)
        return (found_slot, found)

    # Initial state: slot=-1 (not found), found=False
    initial_carry = (jnp.int32(-1), jnp.bool_(False))

    # Linear probing loop (max MAX_PROBES iterations)
    found_slot, found = jax.lax.fori_loop(
        0, MAX_PROBES,
        probe_step,
        initial_carry
    )

    # Extract element list if found
    element_start = jnp.where(
        found,
        hash_octree.element_list_starts[found_slot],
        jnp.int32(0)
    )
    element_length = jnp.where(
        found,
        hash_octree.element_list_lengths[found_slot],
        jnp.int32(0)
    )

    # Clamp length to max_elements_per_cell (safety bound)
    element_length = jnp.minimum(element_length, hash_octree.max_elements_per_cell)

    # Extract elements (pad with -1 for unfound or short lists)
    # We use static indexing with padding to avoid dynamic slicing
    max_elements = hash_octree.max_elements_per_cell
    elements = jnp.full(max_elements, -1, dtype=jnp.int32)

    # Use lax.fori_loop to copy elements (bounded, compile-time known bounds)
    flattened_size = hash_octree.flattened_elements.shape[0]

    def copy_element(i, elements_array):
        """Copy single element if within bounds."""
        src_idx = element_start + i
        is_valid = (i < element_length) & (src_idx < flattened_size) & (flattened_size > 0)

        # Safe indexing: only access array if non-empty
        # When flattened_size == 0, safe_idx will be 0 but we won't use it (is_valid = False)
        safe_idx = jnp.where(
            flattened_size > 0,
            jnp.clip(src_idx, 0, flattened_size - 1),
            jnp.int32(0)
        )

        # Only read from flattened_elements if valid
        elem_value = jnp.where(
            is_valid,
            jnp.where(
                flattened_size > 0,
                hash_octree.flattened_elements[safe_idx],
                jnp.int32(-1)
            ),
            jnp.int32(-1)
        )

        # Update elements array at position i
        return elements_array.at[i].set(elem_value)

    elements = jax.lax.fori_loop(0, max_elements, copy_element, elements)

    return elements, element_length


def hash_lookup_jax(
    point: jnp.ndarray,
    hash_octree: HashOctree,
    level: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Hash table lookup for single point (with Morton encoding).

    This is a convenience wrapper that encodes the point to Morton code
    then calls hash_lookup_jax_from_morton().

    Args:
        point: 3D query position [3]
        hash_octree: HashOctree structure
        level: Octree level to query (for Morton encoding)

    Returns:
        elements: Element IDs for this cell [max_elements_per_cell]
        n_elements: Number of valid elements (0 if not found)
    """
    # Convert point to Morton code (CPU operation)
    point_np = np.asarray(point, dtype=np.float32)
    domain_min_np = np.asarray(hash_octree.bbox_min, dtype=np.float32)
    domain_max_np = np.asarray(hash_octree.bbox_max, dtype=np.float32)

    morton_code = encode_morton_3d(
        float(point_np[0]), float(point_np[1]), float(point_np[2]),
        level,
        domain_min_np, domain_max_np
    )

    return hash_lookup_jax_from_morton(morton_code, hash_octree)


def hash_lookup_batch_jax(
    points: jnp.ndarray,
    hash_octree: HashOctree,
    levels: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Batch hash lookup for multiple points (vectorized via vmap).

    This function uses jax.vmap to parallelize hash lookups across many points,
    enabling efficient GPU acceleration for particle tracking.

    Args:
        points: Query positions [n_points, 3]
        hash_octree: HashOctree structure
        levels: Octree levels for each point [n_points]

    Returns:
        elements: Element IDs for each point [n_points, max_elements_per_cell]
        n_elements: Number of valid elements per point [n_points]

    Example:
        >>> points = jnp.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]])
        >>> levels = jnp.array([5, 5])
        >>> elements, n_elements = hash_lookup_batch_jax(points, hash_octree, levels)
    """
    # Pre-compute Morton codes on CPU (since encode_morton_3d uses numba)
    points_np = np.asarray(points, dtype=np.float32)
    levels_np = np.asarray(levels, dtype=np.int32)
    domain_min_np = np.asarray(hash_octree.bbox_min, dtype=np.float32)
    domain_max_np = np.asarray(hash_octree.bbox_max, dtype=np.float32)

    n_points = len(points_np)
    morton_codes = np.empty(n_points, dtype=np.uint64)

    for i in range(n_points):
        morton_codes[i] = encode_morton_3d(
            float(points_np[i, 0]), float(points_np[i, 1]), float(points_np[i, 2]),
            int(levels_np[i]),
            domain_min_np, domain_max_np
        )

    # Convert to JAX array
    morton_codes_jax = jnp.array(morton_codes, dtype=jnp.int64)

    # Vectorize hash_lookup_jax_from_morton over Morton codes
    return jax.vmap(
        lambda code: hash_lookup_jax_from_morton(code, hash_octree),
        in_axes=0  # Map over Morton codes (axis 0)
    )(morton_codes_jax)

def build_hash_octree_from_mesh_data(
    positions: np.ndarray,
    connectivity: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    max_depth: int = 12,
    max_elements_per_leaf: int = 32,
    target_load_factor: float = 0.77
) -> HashOctree:
    """
    Build hash octree directly from mesh data.
    
    This builds a complete octree from scratch, then converts it to a hash octree.
    Used for lazy loading of hash octrees per timestep.
    
    Args:
        positions: Vertex positions (N, 3)
        connectivity: Element connectivity (M, 4)  
        bbox_min: Domain minimum bounds [3]
        bbox_max: Domain maximum bounds [3]
        max_depth: Maximum octree depth
        max_elements_per_leaf: Max elements before subdivision
        target_load_factor: Hash table load factor
        
    Returns:
        Hash octree structure
    """
    from .morton_code import encode_morton_3d
    
    # Compute element centers
    n_elements = len(connectivity)
    element_centers = np.zeros((n_elements, 3), dtype=np.float32)
    
    for elem_idx in range(n_elements):
        elem_nodes = connectivity[elem_idx]
        center = np.mean(positions[elem_nodes], axis=0)
        element_centers[elem_idx] = center
    
    # Build octree by subdividing recursively
    # Start with root node containing all elements
    
    def subdivide_node(center, half_size, elements, depth):
        """
        Recursively subdivide octree node.

        Args:
            center: Node center position
            half_size: Half the node size
            elements: List of element indices
            depth: Current depth level

        Returns list of (morton_code, element_list) for leaf nodes.
        """
        if depth >= max_depth or len(elements) <= max_elements_per_leaf:
            # Leaf node - encode Morton code from spatial position
            # Calculate grid coordinates from normalized position at this depth
            # Normalize center to [0, 1] relative to domain bounds
            normalized = (center - bbox_min) / (bbox_max - bbox_min)

            # Convert to integer grid coordinates at this depth
            # At depth D, grid is 2^D × 2^D × 2^D
            grid_size = 1 << depth  # 2^depth
            grid_i = int(np.clip(normalized[0] * grid_size, 0, grid_size - 1))
            grid_j = int(np.clip(normalized[1] * grid_size, 0, grid_size - 1))
            grid_k = int(np.clip(normalized[2] * grid_size, 0, grid_size - 1))

            from .morton_code import morton_encode_3d
            morton_code = morton_encode_3d(grid_i, grid_j, grid_k, depth)
            return [(morton_code, elements)]
        
        # Subdivide into 8 children
        child_half_size = half_size / 2.0
        leaves = []

        for child_idx in range(8):
            # Child offset
            offset_x = -child_half_size if (child_idx & 1) == 0 else child_half_size
            offset_y = -child_half_size if (child_idx & 2) == 0 else child_half_size
            offset_z = -child_half_size if (child_idx & 4) == 0 else child_half_size

            child_center = center + np.array([offset_x, offset_y, offset_z], dtype=np.float32)

            # Find elements in this child
            child_min = child_center - child_half_size
            child_max = child_center + child_half_size

            child_elements = []
            for elem_idx in elements:
                elem_center = element_centers[elem_idx]
                if (elem_center[0] >= child_min[0] and elem_center[0] <= child_max[0] and
                    elem_center[1] >= child_min[1] and elem_center[1] <= child_max[1] and
                    elem_center[2] >= child_min[2] and elem_center[2] <= child_max[2]):
                    child_elements.append(elem_idx)

            if len(child_elements) > 0:
                leaves.extend(subdivide_node(child_center, child_half_size, child_elements, depth + 1))
        
        return leaves
    
    # Start subdivision from root
    root_center = (bbox_min + bbox_max) / 2.0
    root_half_size = np.max(bbox_max - bbox_min) / 2.0
    all_elements = list(range(n_elements))
    
    leaf_nodes = subdivide_node(root_center, root_half_size, all_elements, 0)
    
    # Extract Morton codes and element lists
    morton_codes = [node[0] for node in leaf_nodes]
    element_lists = [node[1] for node in leaf_nodes]

    # Debug: Check for duplicate Morton codes
    morton_codes_np = np.array(morton_codes, dtype=np.uint64)
    unique_codes = np.unique(morton_codes_np)
    n_duplicates = len(morton_codes_np) - len(unique_codes)

    if n_duplicates > 0:
        print(f"\n⚠️  WARNING: Found {n_duplicates} duplicate Morton codes out of {len(morton_codes_np)} leaves!")
        print(f"   This will cause hash table insertion failures.")
        print(f"   Unique codes: {len(unique_codes)}")

        # Find and report duplicates
        from collections import Counter
        code_counts = Counter(morton_codes)
        duplicates = [(code, count) for code, count in code_counts.items() if count > 1]
        print(f"   Example duplicates (showing first 5):")
        for code, count in duplicates[:5]:
            print(f"      Morton code {code}: appears {count} times")
    else:
        print(f"\n✅ All {len(morton_codes_np)} Morton codes are unique")

    # Build hash octree
    return build_hash_octree_from_leaves(
        morton_codes,
        element_lists,
        bbox_min,
        bbox_max,
        target_load_factor
    )
