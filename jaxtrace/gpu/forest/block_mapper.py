"""
Element-to-block mapper for forest-of-octrees architecture.

Assigns mesh elements to spatial blocks based on element centroids.
Part of Phase 1: Forest Structure & Block Partitioning
"""

import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass

from .block_grid import Block, position_to_block_id


@dataclass
class BlockAssignmentStats:
    """Statistics about element-to-block assignment."""
    n_elements: int
    n_blocks: int
    n_blocks_used: int
    n_blocks_empty: int
    min_elements: int
    max_elements: int
    mean_elements: float
    median_elements: float
    std_elements: float
    imbalance_ratio: float  # max / mean
    elements_per_block: Dict[int, int]  # block_id -> count
    heavy_blocks: List[int]  # block_ids with > threshold elements

    def __repr__(self) -> str:
        """Human-readable summary."""
        return (
            f"BlockAssignmentStats(\n"
            f"  Total elements: {self.n_elements:,}\n"
            f"  Blocks: {self.n_blocks_used}/{self.n_blocks} used, {self.n_blocks_empty} empty\n"
            f"  Elements per block: min={self.min_elements:,}, "
            f"max={self.max_elements:,}, mean={self.mean_elements:,.1f}, "
            f"median={self.median_elements:,.1f}\n"
            f"  Std dev: {self.std_elements:,.1f}\n"
            f"  Imbalance ratio: {self.imbalance_ratio:.2f}x\n"
            f"  Heavy blocks (>{10000}): {len(self.heavy_blocks)}\n"
            f")"
        )


def compute_element_centroids(
    positions: np.ndarray,
    connectivity: np.ndarray
) -> np.ndarray:
    """
    Compute centroids for all mesh elements.

    Parameters
    ----------
    positions : np.ndarray
        Node positions, shape (N_nodes, 3), float32
    connectivity : np.ndarray
        Element connectivity, shape (N_elements, nodes_per_elem), int32

    Returns
    -------
    centroids : np.ndarray
        Element centroids, shape (N_elements, 3), float32

    Notes
    -----
    For tetrahedral elements: centroid = mean of 4 node positions
    For hexahedral elements: centroid = mean of 8 node positions
    """
    # Get node positions for all elements
    # connectivity: (N_elements, nodes_per_elem)
    # positions[connectivity]: (N_elements, nodes_per_elem, 3)
    element_nodes = positions[connectivity]

    # Compute mean along node axis
    centroids = np.mean(element_nodes, axis=1).astype(np.float32)

    return centroids


def assign_elements_to_blocks(
    positions: np.ndarray,
    connectivity: np.ndarray,
    domain_bounds: np.ndarray,
    grid_size: Tuple[int, int, int],
    heavy_threshold: int = 10000,
    verbose: bool = False
) -> Tuple[np.ndarray, BlockAssignmentStats]:
    """
    Assign each element to a block based on its centroid.

    Parameters
    ----------
    positions : np.ndarray
        Node positions, shape (N_nodes, 3), float32
    connectivity : np.ndarray
        Element connectivity, shape (N_elements, nodes_per_elem), int32
    domain_bounds : np.ndarray
        Domain bounding box [xmin, xmax, ymin, ymax, zmin, zmax], float32
    grid_size : Tuple[int, int, int]
        Number of blocks in (x, y, z) directions
    heavy_threshold : int, optional
        Threshold for identifying heavy blocks (default: 10000)
    verbose : bool, optional
        Print progress messages (default: False)

    Returns
    -------
    element_to_block : np.ndarray
        Block ID for each element, shape (N_elements,), int32
        Elements outside domain have block_id = -1
    stats : BlockAssignmentStats
        Assignment statistics

    Notes
    -----
    - Uses element centroids for block assignment
    - O(N_elements) complexity
    - Elements spanning multiple blocks assigned to block containing centroid
    """
    n_elements = connectivity.shape[0]
    n_blocks = grid_size[0] * grid_size[1] * grid_size[2]

    if verbose:
        print(f"\nAssigning {n_elements:,} elements to {n_blocks} blocks...")
        print(f"Grid size: {grid_size}")
        print(f"Domain: [{domain_bounds[0]:.4f}, {domain_bounds[1]:.4f}] × "
              f"[{domain_bounds[2]:.4f}, {domain_bounds[3]:.4f}] × "
              f"[{domain_bounds[4]:.4f}, {domain_bounds[5]:.4f}]")

    # Compute element centroids
    if verbose:
        print("Computing element centroids...")
    centroids = compute_element_centroids(positions, connectivity)

    # Assign elements to blocks
    if verbose:
        print("Assigning elements to blocks...")
    element_to_block = np.empty(n_elements, dtype=np.int32)

    for i in range(n_elements):
        block_id = position_to_block_id(centroids[i], domain_bounds, grid_size)
        element_to_block[i] = block_id

        if verbose and (i + 1) % 500000 == 0:
            print(f"  Processed {i + 1:,}/{n_elements:,} elements...")

    # Compute statistics
    if verbose:
        print("Computing statistics...")

    # Count elements per block
    elements_per_block = {}
    for block_id in range(n_blocks):
        count = np.sum(element_to_block == block_id)
        elements_per_block[block_id] = int(count)

    # Filter out empty blocks for statistics
    non_empty_counts = [c for c in elements_per_block.values() if c > 0]
    n_blocks_used = len(non_empty_counts)
    n_blocks_empty = n_blocks - n_blocks_used

    if n_blocks_used > 0:
        min_elements = min(non_empty_counts)
        max_elements = max(non_empty_counts)
        mean_elements = np.mean(non_empty_counts)
        median_elements = np.median(non_empty_counts)
        std_elements = np.std(non_empty_counts)
        imbalance_ratio = max_elements / mean_elements if mean_elements > 0 else 0.0
    else:
        min_elements = 0
        max_elements = 0
        mean_elements = 0.0
        median_elements = 0.0
        std_elements = 0.0
        imbalance_ratio = 0.0

    # Identify heavy blocks
    heavy_blocks = [bid for bid, count in elements_per_block.items()
                    if count > heavy_threshold]

    # Count elements outside domain (block_id = -1)
    n_outside = np.sum(element_to_block == -1)
    if n_outside > 0 and verbose:
        print(f"WARNING: {n_outside} elements outside domain bounds")

    stats = BlockAssignmentStats(
        n_elements=n_elements,
        n_blocks=n_blocks,
        n_blocks_used=n_blocks_used,
        n_blocks_empty=n_blocks_empty,
        min_elements=min_elements,
        max_elements=max_elements,
        mean_elements=mean_elements,
        median_elements=median_elements,
        std_elements=std_elements,
        imbalance_ratio=imbalance_ratio,
        elements_per_block=elements_per_block,
        heavy_blocks=heavy_blocks
    )

    if verbose:
        print(f"\n{stats}")
        if heavy_blocks:
            print(f"\nHeavy blocks (>{heavy_threshold:,} elements): {heavy_blocks}")
            for bid in heavy_blocks:
                print(f"  Block {bid}: {elements_per_block[bid]:,} elements")

    return element_to_block, stats


def assign_elements_to_block_list(
    positions: np.ndarray,
    connectivity: np.ndarray,
    blocks: List[Block],
    heavy_threshold: int = 10000,
    verbose: bool = False
) -> Tuple[np.ndarray, BlockAssignmentStats]:
    """
    Assign elements to blocks using a list of Block objects.

    This is a convenience wrapper around assign_elements_to_blocks()
    that extracts domain_bounds and grid_size from the block list.

    Parameters
    ----------
    positions : np.ndarray
        Node positions, shape (N_nodes, 3), float32
    connectivity : np.ndarray
        Element connectivity, shape (N_elements, nodes_per_elem), int32
    blocks : List[Block]
        List of Block objects
    heavy_threshold : int, optional
        Threshold for identifying heavy blocks (default: 10000)
    verbose : bool, optional
        Print progress messages (default: False)

    Returns
    -------
    element_to_block : np.ndarray
        Block ID for each element, shape (N_elements,), int32
    stats : BlockAssignmentStats
        Assignment statistics
    """
    # Infer domain bounds from blocks
    all_bounds = np.array([b.bounds for b in blocks])
    domain_bounds = np.array([
        all_bounds[:, 0].min(),  # xmin
        all_bounds[:, 1].max(),  # xmax
        all_bounds[:, 2].min(),  # ymin
        all_bounds[:, 3].max(),  # ymax
        all_bounds[:, 4].min(),  # zmin
        all_bounds[:, 5].max(),  # zmax
    ], dtype=np.float32)

    # Infer grid size from blocks
    grid_indices = [b.grid_index for b in blocks]
    grid_size = (
        max(idx[0] for idx in grid_indices) + 1,
        max(idx[1] for idx in grid_indices) + 1,
        max(idx[2] for idx in grid_indices) + 1,
    )

    return assign_elements_to_blocks(
        positions, connectivity, domain_bounds, grid_size,
        heavy_threshold=heavy_threshold, verbose=verbose
    )


def validate_assignment(
    element_to_block: np.ndarray,
    positions: np.ndarray,
    connectivity: np.ndarray,
    blocks: List[Block],
    n_samples: int = 1000
) -> bool:
    """
    Validate element-to-block assignment by checking centroid containment.

    Parameters
    ----------
    element_to_block : np.ndarray
        Block ID for each element, shape (N_elements,), int32
    positions : np.ndarray
        Node positions, shape (N_nodes, 3), float32
    connectivity : np.ndarray
        Element connectivity, shape (N_elements, nodes_per_elem), int32
    blocks : List[Block]
        List of Block objects
    n_samples : int, optional
        Number of random elements to check (default: 1000)

    Returns
    -------
    valid : bool
        True if all sampled elements are correctly assigned

    Notes
    -----
    Checks that element centroids are contained in their assigned blocks.
    Uses random sampling for large meshes to keep validation fast.
    """
    n_elements = connectivity.shape[0]
    n_samples = min(n_samples, n_elements)

    # Random sample of elements
    np.random.seed(42)
    sample_indices = np.random.choice(n_elements, size=n_samples, replace=False)

    centroids = compute_element_centroids(positions, connectivity)

    n_errors = 0
    for idx in sample_indices:
        block_id = element_to_block[idx]
        centroid = centroids[idx]

        if block_id == -1:
            # Element outside domain - should not contain centroid in any block
            continue

        block = blocks[block_id]
        if not block.contains_point(centroid):
            n_errors += 1
            print(f"ERROR: Element {idx} assigned to block {block_id}, "
                  f"but centroid {centroid} not in block bounds {block.bounds}")

    if n_errors > 0:
        print(f"\nValidation FAILED: {n_errors}/{n_samples} elements incorrectly assigned")
        return False
    else:
        print(f"\nValidation PASSED: All {n_samples} sampled elements correctly assigned")
        return True
