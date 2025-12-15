"""
Adaptive Octree Builder for Global Morton Structure

This module implements the correct HOT-style octree subdivision where leaves
align with spatial octants (defined by Morton code prefixes) rather than
arbitrary fixed-capacity segments.

Key Differences from Fixed-Capacity Approach:
- OLD: Leaf i = elements [i*256, (i+1)*256] in Morton order (arbitrary spatial mix)
- NEW: Leaf i = elements with Morton prefix P (spatial octant), ≤256 elements

Architecture:
- CPU: Build octree with capacity-constrained recursive subdivision
- CPU: Create prefix→leaf_id lookup table (all possible prefixes at final depth)
- GPU: O(1) position→leaf mapping via prefix table

Expected Performance:
- Success rate: 12.7% → >95% (correct leaf for element centroids)
- Leaf coherence: Elements in same leaf are spatially close
- Search radius: 0-1 sufficient (neighbors in adjacent octants)
"""

import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class OctreeLeaf:
    """Single leaf in the adaptive octree."""
    start_idx: int        # Index in morton_sorted where leaf starts
    length: int           # Number of elements in this leaf
    morton_prefix: int    # Morton code prefix defining this octant
    prefix_bits: int      # Number of bits in the prefix (depth * 3)

    def __repr__(self):
        return f"Leaf(start={self.start_idx}, len={self.length}, prefix=0x{self.morton_prefix:X}, bits={self.prefix_bits})"


def compute_octant_ranges(
    morton_sorted: np.ndarray,
    start_idx: int,
    end_idx: int,
    morton_prefix: int,
    prefix_bits: int
) -> List[Tuple[int, int, int]]:
    """
    Partition Morton range [start_idx, end_idx) into 8 octant subranges.

    Each octant is defined by appending 3 bits (0-7) to the current prefix.
    Uses binary search to find octant boundaries in sorted Morton array.

    Parameters
    ----------
    morton_sorted : np.ndarray
        Sorted Morton codes (uint64)
    start_idx, end_idx : int
        Range in morton_sorted to partition
    morton_prefix : int
        Current Morton prefix (defines parent octant)
    prefix_bits : int
        Number of bits in morton_prefix (must be multiple of 3)

    Returns
    -------
    octant_ranges : List[Tuple[int, int, int]]
        List of (octant_id, octant_start_idx, octant_end_idx) for non-empty octants
    """
    octant_ranges = []

    # Compute shift: how many bits to shift to align with next 3-bit level
    # Morton codes are 63 bits total, prefixes grow from MSB
    shift = 63 - (prefix_bits + 3)

    for octant in range(8):
        # Compute Morton prefix for this octant: parent_prefix || octant_bits
        octant_prefix = (morton_prefix << 3) | octant

        # Find range of elements with this prefix using binary search
        # Elements match if: (morton >> shift) == octant_prefix

        # Lower bound: first morton where (morton >> shift) >= octant_prefix
        left = start_idx
        right = end_idx
        while left < right:
            mid = (left + right) // 2
            morton_mid = morton_sorted[mid] >> shift
            if morton_mid < octant_prefix:
                left = mid + 1
            else:
                right = mid
        octant_start = left

        # Upper bound: first morton where (morton >> shift) > octant_prefix
        left = octant_start
        right = end_idx
        while left < right:
            mid = (left + right) // 2
            morton_mid = morton_sorted[mid] >> shift
            if morton_mid <= octant_prefix:
                left = mid + 1
            else:
                right = mid
        octant_end = left

        # Store non-empty octants
        if octant_end > octant_start:
            octant_ranges.append((octant, octant_start, octant_end))

    return octant_ranges


def build_adaptive_octree_leaves(
    morton_sorted: np.ndarray,
    elem_ids_sorted: np.ndarray,
    leaf_capacity: int = 256,
    max_depth: int = 21
) -> Tuple[List[OctreeLeaf], np.ndarray]:
    """
    Build adaptive octree with capacity-constrained leaves.

    Each leaf:
    - Aligns with a spatial octant (defined by Morton prefix)
    - Contains ≤ leaf_capacity elements
    - Covers a contiguous range in morton_sorted

    Algorithm:
    - Start with root node (entire mesh)
    - Recursively subdivide into 8 octants if > leaf_capacity elements
    - Stop at max_depth or when octant small enough
    - Return list of leaves and prefix→leaf_id mapping

    Parameters
    ----------
    morton_sorted : np.ndarray
        Sorted Morton codes, shape (n_elements,), dtype uint64
    elem_ids_sorted : np.ndarray
        Element IDs in Morton order, shape (n_elements,), dtype int32
    leaf_capacity : int, default=256
        Maximum elements per leaf
    max_depth : int, default=21
        Maximum octree depth (21 → 63-bit Morton codes)

    Returns
    -------
    leaves : List[OctreeLeaf]
        List of octree leaves (spatial octants with ≤256 elements)
    prefix_to_leaf : np.ndarray
        Lookup table: prefix→leaf_id for all prefixes at final depth
        Shape: (8**max_depth,) but sparse representation (only valid prefixes)
        For now, return None (will implement efficient sparse mapping separately)
    """
    n_elements = len(morton_sorted)
    leaves = []

    def subdivide_node(
        start_idx: int,
        end_idx: int,
        morton_prefix: int,
        prefix_bits: int,
        depth: int
    ):
        """Recursively subdivide octree node."""
        n_elements_node = end_idx - start_idx

        # Base case 1: small enough to be a leaf
        if n_elements_node <= leaf_capacity:
            leaf = OctreeLeaf(
                start_idx=start_idx,
                length=n_elements_node,
                morton_prefix=morton_prefix,
                prefix_bits=prefix_bits
            )
            leaves.append(leaf)
            return

        # Base case 2: reached max depth (force leaf)
        if depth >= max_depth:
            leaf = OctreeLeaf(
                start_idx=start_idx,
                length=n_elements_node,
                morton_prefix=morton_prefix,
                prefix_bits=prefix_bits
            )
            leaves.append(leaf)
            return

        # Recursive case: subdivide into 8 octants
        octant_ranges = compute_octant_ranges(
            morton_sorted,
            start_idx,
            end_idx,
            morton_prefix,
            prefix_bits
        )

        # Handle degenerate case: all elements in single octant (shouldn't happen with proper Morton codes)
        if len(octant_ranges) == 1 and octant_ranges[0][1] == start_idx and octant_ranges[0][2] == end_idx:
            # Force leaf to avoid infinite recursion
            leaf = OctreeLeaf(
                start_idx=start_idx,
                length=n_elements_node,
                morton_prefix=morton_prefix,
                prefix_bits=prefix_bits
            )
            leaves.append(leaf)
            return

        # Subdivide non-empty octants
        for octant, octant_start, octant_end in octant_ranges:
            octant_prefix = (morton_prefix << 3) | octant
            subdivide_node(
                octant_start,
                octant_end,
                octant_prefix,
                prefix_bits + 3,
                depth + 1
            )

    # Build octree starting from root
    subdivide_node(
        start_idx=0,
        end_idx=n_elements,
        morton_prefix=0,
        prefix_bits=0,
        depth=0
    )

    # Prefix table will be implemented separately (requires efficient sparse mapping)
    prefix_to_leaf = None

    return leaves, prefix_to_leaf


def build_prefix_table(
    leaves: List[OctreeLeaf],
    max_depth: int = 21
) -> np.ndarray:
    """
    Build prefix→leaf_id lookup table for O(1) position→leaf mapping.

    Strategy:
    Since full table of 8^21 entries is too large, we use a two-level approach:
    1. Determine minimum common depth D where all leaves have unique D-bit prefixes
    2. Create table of size 8^D mapping D-bit prefixes to leaf_id
    3. For GPU lookup: extract D-bit prefix from Morton code → table[prefix]

    Alternative (if 8^D still too large):
    - Use hash table (prefix → leaf_id)
    - Or binary search in sorted prefix list

    For now, implement direct table at depth where table size is reasonable (≤1M entries).

    Parameters
    ----------
    leaves : List[OctreeLeaf]
        Octree leaves from build_adaptive_octree_leaves
    max_depth : int
        Maximum Morton depth

    Returns
    -------
    prefix_table : np.ndarray or Dict
        Lookup structure for prefix→leaf_id
        If direct table: shape (8^D,), dtype int32
        If hash table: dict mapping prefix→leaf_id
    table_depth : int
        Number of bits used in prefix table
    """
    # Find maximum prefix_bits among all leaves (deepest level)
    max_prefix_bits = max(leaf.prefix_bits for leaf in leaves)

    # Determine table depth: use minimum depth where all leaves distinguishable
    # Start from max_prefix_bits and reduce until table size reasonable
    for table_depth_bits in range(max_prefix_bits, 2, -3):  # Step by 3 (one octree level)
        table_size = 8 ** (table_depth_bits // 3)
        if table_size <= 1_000_000:  # 1M entries ≈ 4 MB for int32
            break

    table_depth = table_depth_bits // 3
    table_size = 8 ** table_depth

    # Create prefix table (initialize to -1 = invalid)
    prefix_table = np.full(table_size, -1, dtype=np.int32)

    # Fill table: for each leaf, set all prefixes that map to this leaf
    for leaf_id, leaf in enumerate(leaves):
        leaf_depth = leaf.prefix_bits // 3

        if leaf_depth >= table_depth:
            # Leaf is at or deeper than table depth: extract table_depth-bit prefix
            shift = leaf.prefix_bits - (table_depth * 3)
            prefix = leaf.morton_prefix >> shift
            prefix_table[prefix] = leaf_id
        else:
            # Leaf is shallower than table depth: fill all descendant prefixes
            # Example: leaf at depth 2 (prefix=0b101) with table_depth=4
            # Fill prefixes: 0b101000, 0b101001, ..., 0b101111 (all 64 descendants)
            n_descendants = 8 ** (table_depth - leaf_depth)
            base_prefix = leaf.morton_prefix << ((table_depth - leaf_depth) * 3)
            for i in range(n_descendants):
                prefix = base_prefix + i
                prefix_table[prefix] = leaf_id

    return prefix_table, table_depth


def convert_leaves_to_arrays(
    leaves: List[OctreeLeaf]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert list of OctreeLeaf objects to arrays for GPU upload.

    Parameters
    ----------
    leaves : List[OctreeLeaf]
        Octree leaves

    Returns
    -------
    leaf_start : np.ndarray
        Start index in morton_sorted for each leaf, shape (n_leaves,), dtype int32
    leaf_length : np.ndarray
        Number of elements in each leaf, shape (n_leaves,), dtype int32
    """
    n_leaves = len(leaves)
    leaf_start = np.zeros(n_leaves, dtype=np.int32)
    leaf_length = np.zeros(n_leaves, dtype=np.int32)

    for i, leaf in enumerate(leaves):
        leaf_start[i] = leaf.start_idx
        leaf_length[i] = leaf.length

    return leaf_start, leaf_length


def build_global_morton_octree(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    leaf_capacity: int = 256,
    max_depth: int = 21,
    verbose: bool = True
):
    """
    Build global Morton structure with adaptive octree leaves.

    This is the main entry point that replaces build_global_morton_structure
    in morton_global_builder.py.

    Parameters
    ----------
    node_positions : np.ndarray
        Node coordinates, shape (n_nodes, 3), dtype float32
    connectivity : np.ndarray
        Element connectivity, shape (n_elements, 4), dtype int32
    leaf_capacity : int, default=256
        Maximum elements per leaf
    max_depth : int, default=21
        Maximum octree depth (63-bit Morton codes)
    verbose : bool
        Print build statistics

    Returns
    -------
    morton_struct : MortonStructure
        Named tuple with:
        - elem_ids_sorted: (n_elements,) int32
        - morton_sorted: (n_elements,) uint64
        - leaf_start: (n_leaves,) int32
        - leaf_length: (n_leaves,) int32
        - prefix_table: (8^D,) int32 (prefix→leaf_id)
        - table_depth: int (number of octree levels in prefix table)
        - n_leaves: int
        - bbox_min, bbox_max: (3,) float32
        - max_depth: int
        - leaf_capacity: int
    """
    from collections import namedtuple

    n_elements = connectivity.shape[0]

    # 1. Compute element centroids
    if verbose:
        print(f"[1/5] Computing element centroids...")

    centroids = np.zeros((n_elements, 3), dtype=np.float32)
    for i in range(n_elements):
        nodes = connectivity[i]
        centroid = node_positions[nodes].mean(axis=0)
        centroids[i] = centroid

    # 2. Compute bounding box
    bbox_min = centroids.min(axis=0).astype(np.float32)
    bbox_max = centroids.max(axis=0).astype(np.float32)

    # Add small epsilon to avoid boundary issues
    epsilon = 1e-6 * (bbox_max - bbox_min)
    bbox_min -= epsilon
    bbox_max += epsilon

    if verbose:
        print(f"  Bounding box: min={bbox_min}, max={bbox_max}")

    # 3. Compute Morton codes for all elements
    if verbose:
        print(f"[2/5] Computing Morton codes for {n_elements:,} elements...")

    morton_codes = np.zeros(n_elements, dtype=np.uint64)

    # Vectorized Morton encoding
    normalized = (centroids - bbox_min) / (bbox_max - bbox_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    grid_max = (2 ** max_depth) - 1
    u = np.floor(normalized * grid_max).astype(np.uint32)

    # FIX: Cast to uint64 BEFORE bit operations to avoid overflow
    u = u.astype(np.uint64)

    # Interleave bits (vectorized)
    for i in range(21):
        morton_codes |= ((u[:, 0] >> i) & 1) << (3*i + 0)
        morton_codes |= ((u[:, 1] >> i) & 1) << (3*i + 1)
        morton_codes |= ((u[:, 2] >> i) & 1) << (3*i + 2)

    # 4. Sort elements by Morton code
    if verbose:
        print(f"[3/5] Sorting elements by Morton code...")

    sort_indices = np.argsort(morton_codes)
    morton_sorted = morton_codes[sort_indices]
    elem_ids_sorted = np.arange(n_elements, dtype=np.int32)[sort_indices]

    # 5. Build adaptive octree leaves
    if verbose:
        print(f"[4/5] Building adaptive octree (capacity={leaf_capacity})...")

    leaves, _ = build_adaptive_octree_leaves(
        morton_sorted,
        elem_ids_sorted,
        leaf_capacity=leaf_capacity,
        max_depth=max_depth
    )

    n_leaves = len(leaves)

    if verbose:
        print(f"  Built {n_leaves:,} octree leaves")
        print(f"  Depth distribution:")
        depth_counts = {}
        for leaf in leaves:
            depth = leaf.prefix_bits // 3
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        for depth in sorted(depth_counts.keys()):
            print(f"    Depth {depth}: {depth_counts[depth]:,} leaves")

    # 6. Build prefix table for O(1) lookup
    if verbose:
        print(f"[5/5] Building prefix table...")

    prefix_table, table_depth = build_prefix_table(leaves, max_depth)

    if verbose:
        print(f"  Prefix table: {len(prefix_table):,} entries (depth={table_depth})")
        print(f"  Memory: prefix_table={prefix_table.nbytes / (1024**2):.1f} MB")

    # Convert leaves to arrays
    leaf_start, leaf_length = convert_leaves_to_arrays(leaves)

    # Create output structure
    MortonStructure = namedtuple('MortonStructure', [
        'elem_ids_sorted',
        'morton_sorted',
        'leaf_start',
        'leaf_length',
        'prefix_table',
        'table_depth',
        'n_leaves',
        'bbox_min',
        'bbox_max',
        'max_depth',
        'leaf_capacity'
    ])

    morton_struct = MortonStructure(
        elem_ids_sorted=elem_ids_sorted,
        morton_sorted=morton_sorted,
        leaf_start=leaf_start,
        leaf_length=leaf_length,
        prefix_table=prefix_table,
        table_depth=table_depth,
        n_leaves=n_leaves,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        max_depth=max_depth,
        leaf_capacity=leaf_capacity
    )

    return morton_struct
