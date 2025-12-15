"""
Per-block Morton hash bucket builder for L2 search.

This module builds spatial hash structures PER COARSE BLOCK instead of globally,
enabling bounded O(bucket_size) L2 search that's JAX-friendly (no CSR, no nested vmap).

Architecture:
- Each coarse block gets its own Morton-sorted element list
- Elements stored in padded arrays (JAX-friendly, no dynamic indexing)
- Bounded search: max_L2_elements_per_block (~10-50 elements checked per particle)

Key advantages over global octree:
1. Memory efficient: ~8 MB total vs 6,500 MB global octree
2. Bounded search: O(block_elements) vs O(depth + leaf_size)
3. JAX-compatible: Padded arrays, no CSR, no nested control flow
4. Architecture-aligned: Fits naturally with coarse block structure
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class BlockMortonStructure:
    """
    Per-block Morton spatial indexing structure.

    Attributes
    ----------
    block_id : int
        Block ID
    n_elements : int
        Number of elements in this block
    element_ids : np.ndarray, shape (max_elements_per_block,)
        Element IDs in Morton order (padded with -1)
    morton_codes : np.ndarray, shape (max_elements_per_block,)
        Morton codes for elements (Z-order curve encoding)
    block_bbox_min : np.ndarray, shape (3,)
        Block bounding box minimum
    block_bbox_max : np.ndarray, shape (3,)
        Block bounding box maximum
    """
    block_id: int
    n_elements: int
    element_ids: np.ndarray
    morton_codes: np.ndarray
    block_bbox_min: np.ndarray
    block_bbox_max: np.ndarray


def morton_encode_3d(x: int, y: int, z: int) -> int:
    """
    Encode 3D grid coordinates to Morton code (Z-order curve).

    Morton code interleaves bits: ...z2y2x2z1y1x1z0y0x0
    This preserves spatial locality: nearby points have similar codes.

    Parameters
    ----------
    x, y, z : int
        Grid coordinates (must be < 1024 for 30-bit code)

    Returns
    -------
    morton_code : int
        64-bit Morton code
    """
    # Spread bits by inserting 2 zeros between each bit
    def part_by_2(n: int) -> int:
        n = (n | (n << 16)) & 0x030000FF
        n = (n | (n << 8))  & 0x0300F00F
        n = (n | (n << 4))  & 0x030C30C3
        n = (n | (n << 2))  & 0x09249249
        return n

    return (part_by_2(z) << 2) | (part_by_2(y) << 1) | part_by_2(x)


def compute_element_morton_codes(
    element_centroids: np.ndarray,
    block_bbox_min: np.ndarray,
    block_bbox_max: np.ndarray,
    grid_resolution: int = 1024
) -> np.ndarray:
    """
    Compute Morton codes for elements within a block.

    Maps element centroids to discrete grid, then computes Morton code.

    Parameters
    ----------
    element_centroids : np.ndarray, shape (n_elements, 3)
        Element centroid positions
    block_bbox_min : np.ndarray, shape (3,)
        Block bounding box minimum
    block_bbox_max : np.ndarray, shape (3,)
        Block bounding box maximum
    grid_resolution : int, default=1024
        Grid resolution (max coordinate value = grid_resolution - 1)

    Returns
    -------
    morton_codes : np.ndarray, shape (n_elements,), dtype=int64
        Morton codes for elements
    """
    n_elements = len(element_centroids)
    morton_codes = np.zeros(n_elements, dtype=np.int64)

    # Normalize centroids to [0, 1]^3 within block bbox
    bbox_size = block_bbox_max - block_bbox_min
    normalized = (element_centroids - block_bbox_min) / bbox_size

    # Clamp to [0, 1] (handle boundary cases)
    normalized = np.clip(normalized, 0.0, 1.0)

    # Map to grid coordinates [0, grid_resolution-1]
    grid_coords = (normalized * (grid_resolution - 1)).astype(np.int32)

    # Compute Morton code for each element
    for i in range(n_elements):
        x, y, z = grid_coords[i]
        morton_codes[i] = morton_encode_3d(x, y, z)

    return morton_codes


def build_block_morton_structure(
    block_id: int,
    element_ids: np.ndarray,
    element_centroids: np.ndarray,
    block_bbox_min: np.ndarray,
    block_bbox_max: np.ndarray,
    max_elements_per_block: int = 50
) -> BlockMortonStructure:
    """
    Build Morton hash structure for a single block.

    Parameters
    ----------
    block_id : int
        Block ID
    element_ids : np.ndarray, shape (n_elements,)
        Element IDs in this block
    element_centroids : np.ndarray, shape (n_elements, 3)
        Element centroids in this block
    block_bbox_min : np.ndarray, shape (3,)
        Block bounding box minimum
    block_bbox_max : np.ndarray, shape (3,)
        Block bounding box maximum
    max_elements_per_block : int, default=50
        Maximum elements per block (for padding)

    Returns
    -------
    structure : BlockMortonStructure
        Per-block Morton structure
    """
    n_elements = len(element_ids)

    if n_elements == 0:
        # Empty block
        return BlockMortonStructure(
            block_id=block_id,
            n_elements=0,
            element_ids=np.full(max_elements_per_block, -1, dtype=np.int32),
            morton_codes=np.zeros(max_elements_per_block, dtype=np.int64),
            block_bbox_min=block_bbox_min,
            block_bbox_max=block_bbox_max
        )

    # Compute Morton codes
    morton_codes = compute_element_morton_codes(
        element_centroids,
        block_bbox_min,
        block_bbox_max
    )

    # Sort elements by Morton code
    sort_indices = np.argsort(morton_codes)
    sorted_element_ids = element_ids[sort_indices]
    sorted_morton_codes = morton_codes[sort_indices]

    # Pad to fixed size
    padded_element_ids = np.full(max_elements_per_block, -1, dtype=np.int32)
    padded_morton_codes = np.zeros(max_elements_per_block, dtype=np.int64)

    n_to_copy = min(n_elements, max_elements_per_block)
    padded_element_ids[:n_to_copy] = sorted_element_ids[:n_to_copy]
    padded_morton_codes[:n_to_copy] = sorted_morton_codes[:n_to_copy]

    if n_elements > max_elements_per_block:
        print(f"  WARNING: Block {block_id} has {n_elements} elements, truncated to {max_elements_per_block}")

    return BlockMortonStructure(
        block_id=block_id,
        n_elements=min(n_elements, max_elements_per_block),
        element_ids=padded_element_ids,
        morton_codes=padded_morton_codes,
        block_bbox_min=block_bbox_min,
        block_bbox_max=block_bbox_max
    )


def build_all_block_morton_structures(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    block_ids_per_element: np.ndarray,
    n_blocks: int,
    max_elements_per_block: int = 50,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build Morton structures for all coarse blocks.

    Parameters
    ----------
    node_positions : np.ndarray, shape (n_nodes, 3)
        Node positions
    connectivity : np.ndarray, shape (n_elements, 4)
        Element-to-node connectivity
    block_ids_per_element : np.ndarray, shape (n_elements,)
        Block ID for each element
    n_blocks : int
        Total number of blocks
    max_elements_per_block : int, default=50
        Maximum elements per block (for padding)
    verbose : bool, default=True
        Print progress

    Returns
    -------
    block_element_ids : np.ndarray, shape (n_blocks, max_elements_per_block)
        Element IDs per block (Morton-sorted, padded with -1)
    block_morton_codes : np.ndarray, shape (n_blocks, max_elements_per_block)
        Morton codes per block
    block_bbox_min : np.ndarray, shape (n_blocks, 3)
        Block bounding box minimums
    block_bbox_max : np.ndarray, shape (n_blocks, 3)
        Block bounding box maximums
    """
    n_elements = len(connectivity)

    if verbose:
        print(f"Building per-block Morton structures...")
        print(f"  Elements: {n_elements:,}")
        print(f"  Blocks: {n_blocks:,}")
        print(f"  Max elements per block: {max_elements_per_block}")
        print()

    # Compute element centroids
    if verbose:
        print("  Computing element centroids...")
    element_centroids = np.zeros((n_elements, 3), dtype=np.float32)
    for i in range(n_elements):
        node_ids = connectivity[i]
        element_centroids[i] = node_positions[node_ids].mean(axis=0)

    # Initialize output arrays
    block_element_ids = np.full((n_blocks, max_elements_per_block), -1, dtype=np.int32)
    block_morton_codes = np.zeros((n_blocks, max_elements_per_block), dtype=np.int64)
    block_bbox_min = np.zeros((n_blocks, 3), dtype=np.float32)
    block_bbox_max = np.zeros((n_blocks, 3), dtype=np.float32)

    # Group elements by block
    if verbose:
        print("  Grouping elements by block...")

    block_element_lists = [[] for _ in range(n_blocks)]
    for elem_id in range(n_elements):
        block_id = block_ids_per_element[elem_id]
        if 0 <= block_id < n_blocks:
            block_element_lists[block_id].append(elem_id)

    # Build Morton structure for each block
    if verbose:
        print("  Building Morton structures per block...")

    n_elements_per_block = [len(elems) for elems in block_element_lists]
    max_elems_in_any_block = max(n_elements_per_block) if n_elements_per_block else 0

    for block_id in range(n_blocks):
        block_elem_ids = np.array(block_element_lists[block_id], dtype=np.int32)

        if len(block_elem_ids) == 0:
            # Empty block - use default bbox
            block_bbox_min[block_id] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            block_bbox_max[block_id] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            continue

        block_centroids = element_centroids[block_elem_ids]

        # Compute block bbox
        bbox_min = block_centroids.min(axis=0).astype(np.float32)
        bbox_max = block_centroids.max(axis=0).astype(np.float32)

        # Expand slightly to avoid boundary issues
        bbox_size = bbox_max - bbox_min
        bbox_min -= 0.01 * bbox_size
        bbox_max += 0.01 * bbox_size

        # Build Morton structure
        structure = build_block_morton_structure(
            block_id=block_id,
            element_ids=block_elem_ids,
            element_centroids=block_centroids,
            block_bbox_min=bbox_min,
            block_bbox_max=bbox_max,
            max_elements_per_block=max_elements_per_block
        )

        # Store in arrays
        block_element_ids[block_id] = structure.element_ids
        block_morton_codes[block_id] = structure.morton_codes
        block_bbox_min[block_id] = structure.block_bbox_min
        block_bbox_max[block_id] = structure.block_bbox_max

    if verbose:
        print(f"  ✓ Built {n_blocks:,} block Morton structures")
        print(f"  Elements per block: min={min(n_elements_per_block)}, max={max_elems_in_any_block}, avg={np.mean(n_elements_per_block):.1f}")

        # Memory estimate
        memory_bytes = (
            block_element_ids.nbytes +
            block_morton_codes.nbytes +
            block_bbox_min.nbytes +
            block_bbox_max.nbytes
        )
        memory_mb = memory_bytes / (1024 ** 2)
        print(f"  Memory: {memory_mb:.2f} MB")
        print()

    return block_element_ids, block_morton_codes, block_bbox_min, block_bbox_max


def print_morton_structure_stats(
    block_element_ids: np.ndarray,
    block_bbox_min: np.ndarray,
    block_bbox_max: np.ndarray
):
    """
    Print statistics about Morton structures.

    Parameters
    ----------
    block_element_ids : np.ndarray, shape (n_blocks, max_elements_per_block)
        Element IDs per block
    block_bbox_min : np.ndarray, shape (n_blocks, 3)
        Block bbox minimums
    block_bbox_max : np.ndarray, shape (n_blocks, 3)
        Block bbox maximums
    """
    n_blocks, max_elements_per_block = block_element_ids.shape

    # Count elements per block
    elements_per_block = np.sum(block_element_ids >= 0, axis=1)

    print("Per-Block Morton Structure Statistics:")
    print(f"  Blocks: {n_blocks:,}")
    print(f"  Max elements per block (padded size): {max_elements_per_block}")
    print(f"  Elements per block:")
    print(f"    Min: {elements_per_block.min()}")
    print(f"    Max: {elements_per_block.max()}")
    print(f"    Mean: {elements_per_block.mean():.1f}")
    print(f"    Median: {np.median(elements_per_block):.0f}")
    print(f"  Total elements: {elements_per_block.sum():,}")
    print(f"  Empty blocks: {np.sum(elements_per_block == 0):,}")

    # Memory
    memory_mb = (
        block_element_ids.nbytes +
        block_bbox_min.nbytes +
        block_bbox_max.nbytes
    ) / (1024 ** 2)
    print(f"  Memory: {memory_mb:.2f} MB")
