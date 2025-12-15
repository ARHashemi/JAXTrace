"""
HOT Morton Builder - CPU Preprocessing for JAX-Compatible Morton Search

This module implements the preprocessing pipeline for HOT-like (Hashed Oct-Tree)
Morton-based element search with LOCAL connectivity per leaf to avoid JAX OOM issues.

Key Innovation: Pre-compute local connectivity arrays per octree leaf during CPU
preprocessing, so GPU search accesses only fixed-size local arrays instead of
dynamic global mesh indexing.

Architecture:
1. Cube-aligned block partitioning (no shared elements between blocks)
2. Global Morton sorting of all elements
3. Per-block octree leaf construction with bounded capacity
4. Local connectivity extraction (global → local node ID mapping)

Memory Trade-off:
- Phase 2 Morton: 8 MB (but OOM during execution)
- HOT Morton: ~100-800 MB (OOM-safe, JAX-compatible)
"""

import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class HOTMortonStructures:
    """Complete HOT Morton preprocessing result."""

    # Block-level structures
    n_blocks: int
    block_bbox_min: np.ndarray  # (n_blocks, 3) float32
    block_bbox_max: np.ndarray  # (n_blocks, 3) float32

    # Octree leaf structures (per block)
    n_leaves_per_block: np.ndarray  # (n_blocks,) int32
    max_leaves_per_block: int
    max_leaf_capacity: int  # Max elements per leaf

    # Leaf metadata (n_blocks, max_leaves_per_block)
    leaf_morton_start: np.ndarray  # (n_blocks, max_leaves, 2) int64 - [low, high] Morton range
    leaf_elem_start: np.ndarray    # (n_blocks, max_leaves) int32 - start index in sorted array
    leaf_elem_count: np.ndarray    # (n_blocks, max_leaves) int32 - number of elements

    # Global Morton-sorted element IDs (per block)
    block_morton_sorted_elem_ids: np.ndarray  # (n_blocks, max_elems_per_block) int32
    block_n_elements: np.ndarray              # (n_blocks,) int32

    # Local connectivity per leaf (n_blocks, max_leaves_per_block, max_leaf_capacity)
    leaf_local_connectivity: np.ndarray  # (n_blocks, max_leaves, max_capacity, 4) int32
    leaf_node_coords: np.ndarray         # (n_blocks, max_leaves, max_local_nodes, 3) float32
    leaf_n_local_nodes: np.ndarray       # (n_blocks, max_leaves) int32
    leaf_global_elem_ids: np.ndarray     # (n_blocks, max_leaves, max_capacity) int32

    max_local_nodes: int  # Max unique nodes per leaf


# ============================================================================
# Morton Code Utilities
# ============================================================================

def interleave_bits_3d(x: int, y: int, z: int) -> int:
    """
    Interleave 3 x 21-bit integers into a 63-bit Morton code.

    Morton Z-order curve encoding: xxxyyyzzzxxxyyyzzzxxx...
    Each coordinate is limited to 21 bits (max value ~2M).
    """
    def expand_bits(v: int) -> int:
        """Expand bits: abc → a00b00c00"""
        v &= 0x1fffff  # 21 bits
        v = (v | v << 32) & 0x1f00000000ffff
        v = (v | v << 16) & 0x1f0000ff0000ff
        v = (v | v << 8) & 0x100f00f00f00f00f
        v = (v | v << 4) & 0x10c30c30c30c30c3
        v = (v | v << 2) & 0x1249249249249249
        return v

    return expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2)


def compute_element_morton_code(
    centroid: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    morton_resolution: int = 2097151  # 2^21 - 1
) -> int:
    """
    Compute Morton code for element centroid.

    Args:
        centroid: (3,) element centroid in world space
        bbox_min: (3,) domain bounding box min
        bbox_max: (3,) domain bounding box max
        morton_resolution: quantization resolution (default 2^21 - 1)

    Returns:
        64-bit Morton code (63 bits used)
    """
    # Normalize to [0, 1]
    normalized = (centroid - bbox_min) / (bbox_max - bbox_min)
    normalized = np.clip(normalized, 0.0, 1.0)

    # Quantize to integer grid
    x = int(normalized[0] * morton_resolution)
    y = int(normalized[1] * morton_resolution)
    z = int(normalized[2] * morton_resolution)

    return interleave_bits_3d(x, y, z)


def compute_all_morton_codes(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray
) -> np.ndarray:
    """
    Compute Morton codes for all elements based on their centroids.

    Args:
        node_positions: (n_nodes, 3) float32
        connectivity: (n_elements, 4) int32
        bbox_min: (3,) domain bounds min
        bbox_max: (3,) domain bounds max

    Returns:
        morton_codes: (n_elements,) int64
    """
    n_elements = connectivity.shape[0]
    morton_codes = np.zeros(n_elements, dtype=np.int64)

    for elem_id in range(n_elements):
        node_ids = connectivity[elem_id]
        centroid = node_positions[node_ids].mean(axis=0)
        morton_codes[elem_id] = compute_element_morton_code(centroid, bbox_min, bbox_max)

    return morton_codes


# ============================================================================
# Cube-Aligned Block Construction
# ============================================================================

def build_cube_aligned_blocks(
    connectivity: np.ndarray,
    node_positions: np.ndarray,
    grid_size: Tuple[int, int, int] = (8, 8, 4),
    max_elements_per_block: int = 50000,
    verbose: bool = True
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Assign each element to a cube-aligned block based on parent cube overlap.

    Strategy: An element belongs to a block if its bounding cube overlaps the block.
    Elements may appear in multiple blocks (duplicated at boundaries).

    Args:
        connectivity: (n_elements, 4) int32 - tetrahedral connectivity
        node_positions: (n_nodes, 3) float32 - node coordinates
        grid_size: (nx, ny, nz) - coarse block grid dimensions
        max_elements_per_block: upper bound for padding
        verbose: print statistics

    Returns:
        element_to_blocks: (n_elements,) int32 - primary block assignment
        blocks: List of dicts with 'bbox_min', 'bbox_max', 'element_ids'
    """
    n_elements = connectivity.shape[0]
    nx, ny, nz = grid_size
    n_blocks = nx * ny * nz

    # Compute domain bounds
    domain_min = node_positions.min(axis=0)
    domain_max = node_positions.max(axis=0)

    # Compute block dimensions
    block_dx = (domain_max[0] - domain_min[0]) / nx
    block_dy = (domain_max[1] - domain_min[1]) / ny
    block_dz = (domain_max[2] - domain_min[2]) / nz

    if verbose:
        logger.info(f"Building {n_blocks} cube-aligned blocks ({nx}×{ny}×{nz})")
        logger.info(f"Domain: {domain_min} → {domain_max}")
        logger.info(f"Block size: [{block_dx:.4f}, {block_dy:.4f}, {block_dz:.4f}]")

    # Initialize blocks
    blocks = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                block_id = i + j * nx + k * nx * ny
                bbox_min = domain_min + np.array([i * block_dx, j * block_dy, k * block_dz])
                bbox_max = bbox_min + np.array([block_dx, block_dy, block_dz])
                blocks.append({
                    'block_id': block_id,
                    'bbox_min': bbox_min,
                    'bbox_max': bbox_max,
                    'element_ids': []
                })

    # Assign elements to blocks based on bounding box overlap
    element_to_blocks = np.full(n_elements, -1, dtype=np.int32)

    for elem_id in range(n_elements):
        node_ids = connectivity[elem_id]
        elem_nodes = node_positions[node_ids]

        # Element bounding box
        elem_min = elem_nodes.min(axis=0)
        elem_max = elem_nodes.max(axis=0)

        # Find overlapping blocks
        overlapping_blocks = []
        for block in blocks:
            # Check AABB overlap
            overlap = (
                elem_max[0] >= block['bbox_min'][0] and elem_min[0] <= block['bbox_max'][0] and
                elem_max[1] >= block['bbox_min'][1] and elem_min[1] <= block['bbox_max'][1] and
                elem_max[2] >= block['bbox_min'][2] and elem_min[2] <= block['bbox_max'][2]
            )
            if overlap:
                overlapping_blocks.append(block['block_id'])

        # Assign to first overlapping block (primary)
        if overlapping_blocks:
            primary_block = overlapping_blocks[0]
            element_to_blocks[elem_id] = primary_block

            # Add to all overlapping blocks (may be duplicated)
            for block_id in overlapping_blocks:
                blocks[block_id]['element_ids'].append(elem_id)

    # Validate and report statistics
    max_elems = max(len(block['element_ids']) for block in blocks)
    min_elems = min(len(block['element_ids']) for block in blocks)
    avg_elems = np.mean([len(block['element_ids']) for block in blocks])

    if verbose:
        logger.info(f"Block element counts: min={min_elems}, max={max_elems}, avg={avg_elems:.1f}")
        if max_elems > max_elements_per_block:
            logger.warning(f"Max elements ({max_elems}) exceeds limit ({max_elements_per_block})")

    # Convert element lists to arrays
    for block in blocks:
        block['element_ids'] = np.array(block['element_ids'], dtype=np.int32)

    return element_to_blocks, blocks


# ============================================================================
# Octree Leaf Construction (Per Block)
# ============================================================================

def build_octree_leaves_for_block(
    block_elem_ids: np.ndarray,
    morton_codes: np.ndarray,
    max_leaf_capacity: int = 256,
    max_depth: int = 10,
    verbose: bool = False
) -> List[Dict]:
    """
    Build octree leaves for a single block using Morton code ranges.

    Strategy: Recursively split Morton ranges until each leaf contains ≤ max_leaf_capacity elements.

    Args:
        block_elem_ids: (n_block_elems,) int32 - global element IDs in this block
        morton_codes: (n_total_elements,) int64 - global Morton codes for ALL elements
        max_leaf_capacity: maximum elements per leaf (JAX padding size)
        max_depth: maximum octree depth (prevents infinite recursion)
        verbose: print leaf statistics

    Returns:
        leaves: List of dicts with 'morton_range', 'elem_start', 'elem_count', 'global_elem_ids'
    """
    if len(block_elem_ids) == 0:
        return []

    # Get Morton codes for this block's elements
    block_morton_codes = morton_codes[block_elem_ids]

    # Sort elements by Morton code
    sort_idx = np.argsort(block_morton_codes)
    sorted_elem_ids = block_elem_ids[sort_idx]
    sorted_morton_codes = block_morton_codes[sort_idx]

    # Recursive leaf building
    leaves = []

    def split_recursive(start: int, end: int, depth: int, morton_low: int, morton_high: int):
        """Recursively split Morton range into leaves."""
        count = end - start

        # Base case: small enough or max depth reached
        if count <= max_leaf_capacity or depth >= max_depth:
            leaves.append({
                'morton_range': (morton_low, morton_high),
                'elem_start': start,
                'elem_count': count,
                'global_elem_ids': sorted_elem_ids[start:end]
            })
            return

        # Split at midpoint Morton code
        mid_morton = (morton_low + morton_high) // 2

        # Find split point in sorted array
        split_idx = np.searchsorted(sorted_morton_codes[start:end], mid_morton, side='right') + start

        # Ensure split is valid (at least 1 element on each side)
        if split_idx <= start or split_idx >= end:
            # Cannot split evenly, create leaf
            leaves.append({
                'morton_range': (morton_low, morton_high),
                'elem_start': start,
                'elem_count': count,
                'global_elem_ids': sorted_elem_ids[start:end]
            })
            return

        # Recurse on left and right
        split_recursive(start, split_idx, depth + 1, morton_low, mid_morton)
        split_recursive(split_idx, end, depth + 1, mid_morton, morton_high)

    # Start recursion with full Morton range
    morton_min = sorted_morton_codes[0]
    morton_max = sorted_morton_codes[-1]
    split_recursive(0, len(sorted_elem_ids), 0, morton_min, morton_max)

    if verbose:
        logger.info(f"  Created {len(leaves)} leaves for {len(block_elem_ids)} elements")
        leaf_sizes = [leaf['elem_count'] for leaf in leaves]
        logger.info(f"  Leaf sizes: min={min(leaf_sizes)}, max={max(leaf_sizes)}, avg={np.mean(leaf_sizes):.1f}")

    return leaves


# ============================================================================
# Local Connectivity Extraction (Per Leaf)
# ============================================================================

def build_local_connectivity_for_leaf(
    leaf: Dict,
    connectivity: np.ndarray,
    node_positions: np.ndarray,
    max_leaf_capacity: int = 256,
    max_local_nodes: int = 1024
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Build LOCAL connectivity for a single leaf to avoid global mesh indexing during GPU search.

    This is the CRITICAL innovation that solves JAX OOM issue:
    - Extract unique nodes used by elements in this leaf
    - Build global → local node ID mapping
    - Create local connectivity array (element → local node indices)
    - Extract local node coordinate array

    Args:
        leaf: dict with 'global_elem_ids', 'elem_count'
        connectivity: (n_elements, 4) int32 - GLOBAL connectivity
        node_positions: (n_nodes, 3) float32 - GLOBAL node positions
        max_leaf_capacity: padding size for local connectivity (must match GPU kernel)
        max_local_nodes: padding size for local node coords

    Returns:
        local_connectivity: (max_leaf_capacity, 4) int32 - LOCAL node indices (-1 for padding)
        local_node_coords: (max_local_nodes, 3) float32 - unique node positions
        n_local_nodes: int - actual number of unique nodes
        global_elem_ids_padded: (max_leaf_capacity,) int32 - global element IDs with padding
    """
    global_elem_ids = leaf['global_elem_ids']
    elem_count = leaf['elem_count']

    # CRITICAL: Truncate to max_leaf_capacity if needed
    if elem_count > max_leaf_capacity:
        logger.warning(f"Leaf has {elem_count} elements, exceeds max_leaf_capacity={max_leaf_capacity}, truncating")
        global_elem_ids = global_elem_ids[:max_leaf_capacity]
        elem_count = max_leaf_capacity

    # Get global connectivity for these elements
    global_conn = connectivity[global_elem_ids]  # (elem_count, 4)

    # Find unique nodes
    unique_global_nodes = np.unique(global_conn.flatten())
    n_local_nodes = len(unique_global_nodes)

    if n_local_nodes > max_local_nodes:
        logger.warning(f"Leaf has {n_local_nodes} nodes, exceeds max_local_nodes={max_local_nodes}")
        # Truncate (should not happen with proper max_leaf_capacity tuning)
        unique_global_nodes = unique_global_nodes[:max_local_nodes]
        n_local_nodes = max_local_nodes

    # Build global → local node mapping
    global_to_local = {int(g): l for l, g in enumerate(unique_global_nodes)}

    # Build local connectivity (ELEMENT → LOCAL NODE INDICES)
    local_connectivity = np.full((max_leaf_capacity, 4), -1, dtype=np.int32)
    for i in range(elem_count):
        global_nodes = global_conn[i]
        local_nodes = [global_to_local.get(int(gn), -1) for gn in global_nodes]
        local_connectivity[i] = local_nodes

    # Extract local node coordinates
    local_node_coords = np.zeros((max_local_nodes, 3), dtype=np.float32)
    local_node_coords[:n_local_nodes] = node_positions[unique_global_nodes]

    # Pad global element IDs
    global_elem_ids_padded = np.full(max_leaf_capacity, -1, dtype=np.int32)
    global_elem_ids_padded[:elem_count] = global_elem_ids

    return local_connectivity, local_node_coords, n_local_nodes, global_elem_ids_padded


# ============================================================================
# Complete HOT Morton Preprocessing Pipeline
# ============================================================================

def build_hot_morton_structures(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    grid_size: Tuple[int, int, int] = (8, 8, 4),
    max_elements_per_block: int = 50000,
    max_leaf_capacity: int = 256,
    max_local_nodes: int = 1024,
    verbose: bool = True
) -> HOTMortonStructures:
    """
    Complete CPU preprocessing pipeline for HOT Morton search.

    Pipeline:
    1. Build cube-aligned blocks
    2. Compute global Morton codes
    3. Build octree leaves per block
    4. Extract local connectivity per leaf
    5. Pad and organize into GPU-ready arrays

    Args:
        node_positions: (n_nodes, 3) float32
        connectivity: (n_elements, 4) int32
        grid_size: (nx, ny, nz) coarse block grid
        max_elements_per_block: padding size for block arrays
        max_leaf_capacity: max elements per leaf (JAX bounded loop size)
        max_local_nodes: max unique nodes per leaf
        verbose: print progress

    Returns:
        HOTMortonStructures with all preprocessed GPU-ready arrays
    """
    if verbose:
        logger.info("=" * 80)
        logger.info("HOT Morton Preprocessing Pipeline")
        logger.info("=" * 80)

    n_elements = connectivity.shape[0]
    n_nodes = node_positions.shape[0]
    nx, ny, nz = grid_size
    n_blocks = nx * ny * nz

    if verbose:
        logger.info(f"Mesh: {n_elements} elements, {n_nodes} nodes")
        logger.info(f"Grid: {n_blocks} blocks ({nx}×{ny}×{nz})")

    # Step 1: Build cube-aligned blocks
    if verbose:
        logger.info("\n[1/5] Building cube-aligned blocks...")
    element_to_blocks, blocks = build_cube_aligned_blocks(
        connectivity, node_positions, grid_size, max_elements_per_block, verbose
    )

    # Step 2: Compute global Morton codes
    if verbose:
        logger.info("\n[2/5] Computing global Morton codes...")
    domain_min = node_positions.min(axis=0)
    domain_max = node_positions.max(axis=0)
    morton_codes = compute_all_morton_codes(node_positions, connectivity, domain_min, domain_max)

    # Step 3: Build octree leaves per block
    if verbose:
        logger.info("\n[3/5] Building octree leaves per block...")
    all_leaves = []
    max_leaves_per_block = 0

    for block_id, block in enumerate(blocks):
        block_elem_ids = block['element_ids']
        leaves = build_octree_leaves_for_block(
            block_elem_ids, morton_codes, max_leaf_capacity, verbose=verbose
        )
        all_leaves.append(leaves)
        max_leaves_per_block = max(max_leaves_per_block, len(leaves))

        if verbose and block_id % 50 == 0:
            logger.info(f"  Block {block_id}/{n_blocks}: {len(leaves)} leaves, {len(block_elem_ids)} elements")

    if verbose:
        logger.info(f"Max leaves per block: {max_leaves_per_block}")

    # Step 4: Extract local connectivity per leaf
    if verbose:
        logger.info("\n[4/5] Extracting local connectivity per leaf...")

    # Initialize padded arrays
    leaf_local_connectivity = np.full(
        (n_blocks, max_leaves_per_block, max_leaf_capacity, 4), -1, dtype=np.int32
    )
    leaf_node_coords = np.zeros(
        (n_blocks, max_leaves_per_block, max_local_nodes, 3), dtype=np.float32
    )
    leaf_n_local_nodes = np.zeros((n_blocks, max_leaves_per_block), dtype=np.int32)
    leaf_global_elem_ids = np.full(
        (n_blocks, max_leaves_per_block, max_leaf_capacity), -1, dtype=np.int32
    )
    leaf_elem_count = np.zeros((n_blocks, max_leaves_per_block), dtype=np.int32)
    leaf_morton_start = np.zeros((n_blocks, max_leaves_per_block, 2), dtype=np.int64)

    total_leaves_processed = 0
    total_leaves_skipped = 0
    for block_id, leaves in enumerate(all_leaves):
        for leaf_idx, leaf in enumerate(leaves):
            # Check if we exceed max_leaves_per_block
            if leaf_idx >= max_leaves_per_block:
                if verbose and total_leaves_skipped == 0:
                    logger.warning(f"  Block {block_id} has {len(leaves)} leaves, exceeds max_leaves_per_block={max_leaves_per_block}")
                    logger.warning(f"  Skipping excess leaves (this may reduce search coverage)")
                total_leaves_skipped += 1
                continue

            local_conn, local_coords, n_local, global_elem_ids_padded = \
                build_local_connectivity_for_leaf(
                    leaf, connectivity, node_positions, max_leaf_capacity, max_local_nodes
                )

            leaf_local_connectivity[block_id, leaf_idx] = local_conn
            leaf_node_coords[block_id, leaf_idx] = local_coords
            leaf_n_local_nodes[block_id, leaf_idx] = n_local
            leaf_global_elem_ids[block_id, leaf_idx] = global_elem_ids_padded
            leaf_elem_count[block_id, leaf_idx] = leaf['elem_count']
            leaf_morton_start[block_id, leaf_idx] = leaf['morton_range']

            total_leaves_processed += 1

        if verbose and block_id % 50 == 0:
            logger.info(f"  Processed {total_leaves_processed} leaves ({block_id}/{n_blocks} blocks)")

    if verbose:
        logger.info(f"Total leaves processed: {total_leaves_processed}")
        if total_leaves_skipped > 0:
            logger.warning(f"Total leaves SKIPPED: {total_leaves_skipped} (exceeded max_leaves_per_block)")
            logger.warning(f"Consider increasing max_leaves_per_block or max_leaf_capacity")

    # Step 5: Organize block metadata
    if verbose:
        logger.info("\n[5/5] Organizing block metadata...")

    block_bbox_min = np.array([block['bbox_min'] for block in blocks], dtype=np.float32)
    block_bbox_max = np.array([block['bbox_max'] for block in blocks], dtype=np.float32)
    n_leaves_per_block = np.array([min(len(leaves), max_leaves_per_block) for leaves in all_leaves], dtype=np.int32)

    # Morton-sorted element IDs per block (for reference)
    block_morton_sorted_elem_ids = np.full((n_blocks, max_elements_per_block), -1, dtype=np.int32)
    block_n_elements = np.zeros(n_blocks, dtype=np.int32)

    for block_id, block in enumerate(blocks):
        block_elem_ids = block['element_ids']
        block_morton = morton_codes[block_elem_ids]
        sort_idx = np.argsort(block_morton)
        sorted_elem_ids = block_elem_ids[sort_idx]

        n_elems = len(sorted_elem_ids)
        block_morton_sorted_elem_ids[block_id, :n_elems] = sorted_elem_ids
        block_n_elements[block_id] = n_elems

    # Memory analysis
    if verbose:
        logger.info("\n" + "=" * 80)
        logger.info("Memory Analysis")
        logger.info("=" * 80)

        mem_local_conn = leaf_local_connectivity.nbytes / (1024**2)
        mem_node_coords = leaf_node_coords.nbytes / (1024**2)
        mem_elem_ids = leaf_global_elem_ids.nbytes / (1024**2)
        mem_metadata = (leaf_morton_start.nbytes + leaf_elem_count.nbytes +
                       leaf_n_local_nodes.nbytes) / (1024**2)
        mem_total = mem_local_conn + mem_node_coords + mem_elem_ids + mem_metadata

        logger.info(f"Local connectivity: {mem_local_conn:.2f} MB")
        logger.info(f"Node coordinates: {mem_node_coords:.2f} MB")
        logger.info(f"Global element IDs: {mem_elem_ids:.2f} MB")
        logger.info(f"Metadata: {mem_metadata:.2f} MB")
        logger.info(f"TOTAL: {mem_total:.2f} MB")
        logger.info("=" * 80)

    return HOTMortonStructures(
        n_blocks=n_blocks,
        block_bbox_min=block_bbox_min,
        block_bbox_max=block_bbox_max,
        n_leaves_per_block=n_leaves_per_block,
        max_leaves_per_block=max_leaves_per_block,
        max_leaf_capacity=max_leaf_capacity,
        leaf_morton_start=leaf_morton_start,
        leaf_elem_start=np.zeros((n_blocks, max_leaves_per_block), dtype=np.int32),  # Not used with local conn
        leaf_elem_count=leaf_elem_count,
        block_morton_sorted_elem_ids=block_morton_sorted_elem_ids,
        block_n_elements=block_n_elements,
        leaf_local_connectivity=leaf_local_connectivity,
        leaf_node_coords=leaf_node_coords,
        leaf_n_local_nodes=leaf_n_local_nodes,
        leaf_global_elem_ids=leaf_global_elem_ids,
        max_local_nodes=max_local_nodes
    )
