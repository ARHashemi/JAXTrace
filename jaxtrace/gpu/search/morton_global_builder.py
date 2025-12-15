"""
Global Morton Structure Builder (CPU Preprocessing)

This module builds a global HOT-like Morton structure for GPU element search.
Unlike block-based approaches, this uses a SINGLE global Morton-sorted element list
divided into fixed-capacity leaf segments.

Architecture:
1. Compute Morton codes for all element centroids
2. Sort elements globally by Morton code
3. Divide sorted list into fixed-capacity leaves (128-256 elements)
4. Upload to GPU for bounded search

Key Innovation: No blocks, no dynamic slicing - just offset-based leaf lookup
and bounded loops over small candidate sets.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger(__name__)


@dataclass
class GlobalMortonStructure:
    """
    Global HOT Morton structure - NO blocks.

    Attributes
    ----------
    elem_ids_sorted : np.ndarray, shape (n_elements,), dtype int32
        Element IDs in Morton order (global sort)
    morton_sorted : np.ndarray, shape (n_elements,), dtype uint64
        Sorted Morton codes (for debugging/validation)
    leaf_start : np.ndarray, shape (n_leaves,), dtype int32
        Start index of each leaf in elem_ids_sorted
    leaf_length : np.ndarray, shape (n_leaves,), dtype int32
        Number of elements in each leaf (≤ leaf_capacity)
    n_leaves : int
        Total number of leaves
    morton_min : np.uint64
        Minimum Morton code (for linear mapping)
    morton_max : np.uint64
        Maximum Morton code (for linear mapping)
    bbox_min : np.ndarray, shape (3,), dtype float32
        Domain minimum [xmin, ymin, zmin]
    bbox_max : np.ndarray, shape (3,), dtype float32
        Domain maximum [xmax, ymax, zmax]
    max_depth : int
        Morton encoding depth (bits per dimension, typically 21)
    leaf_capacity : int
        Maximum elements per leaf (e.g., 256)
    """
    elem_ids_sorted: np.ndarray
    morton_sorted: np.ndarray
    leaf_start: np.ndarray
    leaf_length: np.ndarray
    n_leaves: int
    morton_min: np.uint64
    morton_max: np.uint64
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    max_depth: int
    leaf_capacity: int


def interleave_bits_3d(x: np.uint32, y: np.uint32, z: np.uint32) -> np.uint64:
    """
    Interleave bits of 3D coordinates to compute Morton (Z-order) code.

    Morton code is a space-filling curve that preserves spatial locality:
    nearby points in 3D space have nearby Morton codes.

    Example:
        x = 0b101 (5)
        y = 0b110 (6)
        z = 0b011 (3)
        → morton = 0b011110011 (bit interleaving: zyxzyxzyx)

    Parameters
    ----------
    x, y, z : np.uint32
        Normalized integer coordinates (0 to 2^21 - 1)

    Returns
    -------
    morton : np.uint64
        Morton code (up to 63 bits for 21 bits per dimension)

    Notes
    -----
    This is a CPU implementation. GPU version uses JAX-compatible operations.
    """
    morton = np.uint64(0)

    # Interleave bits: bit i of x goes to position 3i+0,
    #                  bit i of y goes to position 3i+1,
    #                  bit i of z goes to position 3i+2
    for i in range(21):  # Up to 21 bits per dimension (63 total)
        morton |= ((x >> i) & np.uint64(1)) << (3*i + 0)
        morton |= ((y >> i) & np.uint64(1)) << (3*i + 1)
        morton |= ((z >> i) & np.uint64(1)) << (3*i + 2)

    return morton


def compute_morton_codes_for_elements(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    max_depth: int = 21
) -> np.ndarray:
    """
    Compute Morton code for each element based on centroid position.

    Steps:
    1. Compute element centroids (mean of 4 tet vertices)
    2. Normalize to integer grid [0, 2^max_depth - 1]
    3. Interleave bits to create Morton code

    Parameters
    ----------
    node_positions : np.ndarray, shape (n_nodes, 3), dtype float32 or float64
        Node coordinates
    connectivity : np.ndarray, shape (n_elements, 4), dtype int32
        Element-to-node connectivity (tetrahedral mesh)
    bbox_min : np.ndarray, shape (3,), dtype float32
        Domain minimum [xmin, ymin, zmin]
    bbox_max : np.ndarray, shape (3,), dtype float32
        Domain maximum [xmax, ymax, zmax]
    max_depth : int, default=21
        Bits per dimension (21 → 63 bits total)

    Returns
    -------
    morton_codes : np.ndarray, shape (n_elements,), dtype uint64
        Morton code for each element

    Notes
    -----
    - max_depth=21 provides 2^21 = 2M resolution per dimension
    - Total Morton space: 2^63 ≈ 9×10^18 unique codes
    """
    n_elements = connectivity.shape[0]
    morton_codes = np.empty(n_elements, dtype=np.uint64)

    # Scaling factor: map [bbox_min, bbox_max] → [0, 2^max_depth - 1]
    scale = (2**max_depth - 1) / (bbox_max - bbox_min)

    logger.debug(f"Computing Morton codes for {n_elements:,} elements")
    logger.debug(f"  Depth: {max_depth} bits/dim → {3*max_depth} bits total")
    logger.debug(f"  Resolution: {2**max_depth:,} per dimension")

    for e in range(n_elements):
        # Get node indices for this element
        nodes = connectivity[e]

        # Compute centroid
        centroid = node_positions[nodes].mean(axis=0)

        # Normalize to integer grid
        normalized = (centroid - bbox_min) * scale
        ux = np.uint32(np.floor(normalized[0]))
        uy = np.uint32(np.floor(normalized[1]))
        uz = np.uint32(np.floor(normalized[2]))

        # Clamp to valid range (handle floating-point edge cases)
        max_val = np.uint32(2**max_depth - 1)
        ux = min(ux, max_val)
        uy = min(uy, max_val)
        uz = min(uz, max_val)

        # Interleave bits
        morton_codes[e] = interleave_bits_3d(ux, uy, uz)

    return morton_codes


def build_global_morton_sorted_list(
    morton_codes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort elements globally by Morton code.

    This is the core of HOT Morton: a single global sort that preserves
    spatial locality via the Z-order curve.

    Parameters
    ----------
    morton_codes : np.ndarray, shape (n_elements,), dtype uint64
        Morton codes for all elements

    Returns
    -------
    elem_ids_sorted : np.ndarray, shape (n_elements,), dtype int32
        Element IDs in Morton order (sorted indices)
    morton_sorted : np.ndarray, shape (n_elements,), dtype uint64
        Morton codes in sorted order

    Notes
    -----
    - numpy.argsort provides stable sort (preserves order of equal keys)
    - Complexity: O(N log N) where N is number of elements
    """
    n_elements = len(morton_codes)

    logger.debug(f"Sorting {n_elements:,} elements by Morton code")

    # Get sorted indices
    sorted_indices = np.argsort(morton_codes)

    # Apply sorting
    elem_ids_sorted = np.arange(n_elements, dtype=np.int32)[sorted_indices]
    morton_sorted = morton_codes[sorted_indices]

    # Validation: check for duplicates (unlikely but possible)
    unique_codes = np.unique(morton_sorted)
    if len(unique_codes) < n_elements:
        logger.warning(f"  Duplicate Morton codes detected: {n_elements - len(unique_codes):,} duplicates")
        logger.warning(f"  This is expected for highly refined meshes (multiple elements per Morton cell)")

    return elem_ids_sorted, morton_sorted


def build_fixed_capacity_leaves(
    elem_ids_sorted: np.ndarray,
    leaf_capacity: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Divide sorted element list into fixed-capacity leaf segments.

    Phase 1 implementation: Simple uniform segmentation
    - Leaf 0: elements [0, C)
    - Leaf 1: elements [C, 2C)
    - ...
    - Leaf L: elements [L*C, min(L*C+C, N))

    Later phases can use geometric octree-aligned leaves based on Morton prefixes.

    Parameters
    ----------
    elem_ids_sorted : np.ndarray, shape (n_elements,), dtype int32
        Element IDs in Morton order
    leaf_capacity : int, default=256
        Maximum elements per leaf (fixed bound for JAX loops)

    Returns
    -------
    leaf_start : np.ndarray, shape (n_leaves,), dtype int32
        Start index of each leaf in elem_ids_sorted
    leaf_length : np.ndarray, shape (n_leaves,), dtype int32
        Number of elements in each leaf (≤ leaf_capacity)

    Notes
    -----
    - Last leaf may have fewer than leaf_capacity elements
    - All leaves except last have exactly leaf_capacity elements
    - GPU search uses bounded loop [0, leaf_capacity) with masks
    """
    n_elements = len(elem_ids_sorted)
    n_leaves = (n_elements + leaf_capacity - 1) // leaf_capacity

    logger.debug(f"Building {n_leaves:,} leaves with capacity {leaf_capacity}")

    # Compute leaf start indices
    leaf_start = np.arange(n_leaves, dtype=np.int32) * leaf_capacity

    # Compute leaf lengths (last leaf may be shorter)
    leaf_length = np.minimum(
        np.full(n_leaves, leaf_capacity, dtype=np.int32),
        n_elements - leaf_start
    )

    # Validation
    assert np.all(leaf_length > 0), "All leaves must have at least one element"
    assert np.all(leaf_length <= leaf_capacity), "No leaf can exceed capacity"
    assert np.sum(leaf_length) == n_elements, "All elements must be covered"

    # Statistics
    avg_length = leaf_length.mean()
    logger.debug(f"  Leaves: {n_leaves:,}")
    logger.debug(f"  Avg elements per leaf: {avg_length:.1f}")
    logger.debug(f"  Last leaf: {leaf_length[-1]} elements")

    return leaf_start, leaf_length


def build_morton_leaf_mapping(
    morton_sorted: np.ndarray,
    n_leaves: int
) -> Tuple[np.uint64, np.uint64]:
    """
    Compute Morton range for linear leaf mapping.

    Phase 1: Linear approximation
        leaf_id ≈ (morton - morton_min) / (morton_max - morton_min) * n_leaves

    Later phases can use prefix tables for exact geometric mapping.

    Parameters
    ----------
    morton_sorted : np.ndarray, shape (n_elements,), dtype uint64
        Sorted Morton codes
    n_leaves : int
        Number of leaves

    Returns
    -------
    morton_min : np.uint64
        Minimum Morton code
    morton_max : np.uint64
        Maximum Morton code

    Notes
    -----
    - Linear mapping assumes uniform distribution along Morton curve
    - Works well for uniformly refined meshes
    - For highly adaptive meshes, consider prefix table (Phase 2)
    """
    morton_min = np.uint64(morton_sorted[0])
    morton_max = np.uint64(morton_sorted[-1])

    morton_range = int(morton_max - morton_min)
    logger.debug(f"Morton range: [{morton_min}, {morton_max}] (span: {morton_range:,})")

    return morton_min, morton_max


def build_global_morton_structure(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    leaf_capacity: int = 256,
    max_depth: int = 21,
    verbose: bool = True
) -> GlobalMortonStructure:
    """
    Build complete global HOT Morton structure (CPU preprocessing).

    This is the main entry point for Phase 1. Builds all structures needed
    for GPU-resident L2 search.

    Steps:
    1. Compute Morton codes for element centroids
    2. Sort elements globally by Morton code
    3. Divide into fixed-capacity leaves
    4. Compute Morton range for mapping

    Parameters
    ----------
    node_positions : np.ndarray, shape (n_nodes, 3), dtype float32 or float64
        Node coordinates
    connectivity : np.ndarray, shape (n_elements, 4), dtype int32
        Element-to-node connectivity
    leaf_capacity : int, default=256
        Maximum elements per leaf (JAX bounded loop size)
    max_depth : int, default=21
        Morton encoding depth (bits per dimension)
    verbose : bool, default=True
        Print progress and statistics

    Returns
    -------
    GlobalMortonStructure
        Complete structure ready for GPU upload

    Examples
    --------
    >>> node_positions, connectivity, _ = load_mesh_from_pvtu(mesh_path)
    >>> morton_struct = build_global_morton_structure(
    ...     node_positions, connectivity, leaf_capacity=256
    ... )
    >>> # Upload to GPU
    >>> mesh_morton_gpu = upload_global_morton_to_gpu(morton_struct)

    Notes
    -----
    - Memory: ~4 bytes × n_elements for sorted list + ~8 bytes × n_leaves for metadata
    - Time: O(N log N) for sorting, ~30-60s for 3.5M elements
    - No OOM risk: all arrays are linear in mesh size
    """
    if verbose:
        logger.info("=" * 80)
        logger.info("Building Global HOT Morton Structure")
        logger.info("=" * 80)
        logger.info(f"Elements: {connectivity.shape[0]:,}")
        logger.info(f"Nodes: {node_positions.shape[0]:,}")
        logger.info(f"Leaf capacity: {leaf_capacity}")
        logger.info(f"Max depth: {max_depth} bits/dim")

    # Step 1: Compute bounding box
    if verbose:
        logger.info("\n[1/5] Computing domain bounds...")

    bbox_min = node_positions.min(axis=0).astype(np.float32)
    bbox_max = node_positions.max(axis=0).astype(np.float32)

    if verbose:
        logger.info(f"  Domain: X=[{bbox_min[0]:.3f}, {bbox_max[0]:.3f}]")
        logger.info(f"          Y=[{bbox_min[1]:.3f}, {bbox_max[1]:.3f}]")
        logger.info(f"          Z=[{bbox_min[2]:.3f}, {bbox_max[2]:.3f}]")

    # Step 2: Compute Morton codes
    if verbose:
        logger.info("\n[2/5] Computing Morton codes for element centroids...")

    morton_codes = compute_morton_codes_for_elements(
        node_positions, connectivity, bbox_min, bbox_max, max_depth
    )

    if verbose:
        logger.info(f"  Computed {len(morton_codes):,} Morton codes")
        logger.info(f"  Min code: {morton_codes.min()}")
        logger.info(f"  Max code: {morton_codes.max()}")

    # Step 3: Global sort
    if verbose:
        logger.info("\n[3/5] Sorting elements by Morton code (global sort)...")

    elem_ids_sorted, morton_sorted = build_global_morton_sorted_list(morton_codes)

    if verbose:
        logger.info(f"  Sorted {len(elem_ids_sorted):,} elements")

    # Step 4: Leaf segmentation
    if verbose:
        logger.info("\n[4/5] Dividing into fixed-capacity leaves...")

    leaf_start, leaf_length = build_fixed_capacity_leaves(elem_ids_sorted, leaf_capacity)
    n_leaves = len(leaf_start)

    if verbose:
        logger.info(f"  Created {n_leaves:,} leaves")
        logger.info(f"  Avg elements per leaf: {leaf_length.mean():.1f}")
        logger.info(f"  Full leaves: {np.sum(leaf_length == leaf_capacity):,}")
        logger.info(f"  Partial leaves: {np.sum(leaf_length < leaf_capacity):,}")

    # Step 5: Morton mapping
    if verbose:
        logger.info("\n[5/5] Computing Morton range for linear mapping...")

    morton_min, morton_max = build_morton_leaf_mapping(morton_sorted, n_leaves)

    if verbose:
        logger.info(f"  Morton min: {morton_min}")
        logger.info(f"  Morton max: {morton_max}")
        logger.info(f"  Range: {int(morton_max - morton_min):,}")

    # Create structure
    structure = GlobalMortonStructure(
        elem_ids_sorted=elem_ids_sorted,
        morton_sorted=morton_sorted,
        leaf_start=leaf_start,
        leaf_length=leaf_length,
        n_leaves=n_leaves,
        morton_min=morton_min,
        morton_max=morton_max,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        max_depth=max_depth,
        leaf_capacity=leaf_capacity
    )

    if verbose:
        logger.info("\n" + "=" * 80)
        logger.info("✅ Global HOT Morton Structure Complete")
        logger.info("=" * 80)
        logger.info(f"Memory footprint:")
        elem_sorted_mb = elem_ids_sorted.nbytes / (1024**2)
        morton_sorted_mb = morton_sorted.nbytes / (1024**2)
        leaf_start_mb = leaf_start.nbytes / (1024**2)
        leaf_length_mb = leaf_length.nbytes / (1024**2)
        total_mb = elem_sorted_mb + morton_sorted_mb + leaf_start_mb + leaf_length_mb
        logger.info(f"  elem_ids_sorted: {elem_sorted_mb:.2f} MB")
        logger.info(f"  morton_sorted: {morton_sorted_mb:.2f} MB")
        logger.info(f"  leaf metadata: {(leaf_start_mb + leaf_length_mb):.2f} MB")
        logger.info(f"  Total: {total_mb:.2f} MB")
        logger.info("=" * 80)

    return structure
