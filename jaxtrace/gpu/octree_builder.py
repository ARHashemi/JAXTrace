#!/usr/bin/env python3
"""
Octree Builder using Morton Code Sorting

Builds octrees within each block for efficient Level 2 element search.
Uses Morton code (Z-curve) ordering for cache-friendly spatial traversal.

Phase 2.2 of V3 Plan

The octree is built in a flat array format optimized for JAX:
- No pointers or dynamic structures
- Fixed-size padded arrays
- All data contiguous in memory
"""

from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
import jax

try:
    from .morton_code import compute_morton_codes, sort_by_morton_code
except ImportError:
    from morton_code import compute_morton_codes, sort_by_morton_code

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


@dataclass
class OctreeData:
    """
    Flat array representation of octree for JAX/GPU.

    The octree is stored as:
    - Sorted element IDs (Z-curve order)
    - Node ranges (start/end indices into sorted elements)
    - Node bounding boxes
    """

    # Element data (sorted by Morton code)
    sorted_element_IDs: np.ndarray      # (N_elements,) int32 - Z-curve order
    element_morton_codes: np.ndarray    # (N_elements,) uint64 - sorted Morton codes

    # Octree nodes (flat array)
    node_ranges: np.ndarray             # (N_nodes, 2) int32 - [start, end) indices
    node_depths: np.ndarray             # (N_nodes,) int32 - depth in tree (0 = root)
    node_bbox_min: np.ndarray           # (N_nodes, 3) float64 - node bounding box min
    node_bbox_max: np.ndarray           # (N_nodes, 3) float64 - node bounding box max

    # Metadata
    n_elements: int = 0
    n_nodes: int = 0
    max_depth: int = 0
    max_elements_per_node: int = 0

    def __post_init__(self):
        """Compute metadata."""
        if self.sorted_element_IDs is not None:
            self.n_elements = len(self.sorted_element_IDs)
        if self.node_ranges is not None:
            self.n_nodes = len(self.node_ranges)
        if self.node_depths is not None and len(self.node_depths) > 0:
            self.max_depth = int(np.max(self.node_depths))

    def memory_usage_mb(self) -> Dict[str, float]:
        """Compute memory usage."""
        usage = {}

        if self.sorted_element_IDs is not None:
            usage['sorted_element_IDs'] = self.sorted_element_IDs.nbytes / (1024**2)

        if self.element_morton_codes is not None:
            usage['element_morton_codes'] = self.element_morton_codes.nbytes / (1024**2)

        if self.node_ranges is not None:
            usage['node_ranges'] = self.node_ranges.nbytes / (1024**2)

        if self.node_depths is not None:
            usage['node_depths'] = self.node_depths.nbytes / (1024**2)

        if self.node_bbox_min is not None:
            usage['node_bbox_min'] = self.node_bbox_min.nbytes / (1024**2)

        if self.node_bbox_max is not None:
            usage['node_bbox_max'] = self.node_bbox_max.nbytes / (1024**2)

        usage['total'] = sum(usage.values())

        return usage

    def __str__(self) -> str:
        """Human-readable summary."""
        mem = self.memory_usage_mb()

        lines = [
            "=" * 80,
            "OCTREE DATA (Flat Arrays)",
            "=" * 80,
            "",
            f"Elements: {self.n_elements:,} (sorted by Morton code)",
            f"Octree nodes: {self.n_nodes:,}",
            f"Max depth: {self.max_depth}",
            f"Max elements/node: {self.max_elements_per_node}",
            "",
            "Memory Usage:",
        ]

        for key, value in mem.items():
            if key != 'total':
                lines.append(f"  {key}: {value:.2f} MB")
        lines.append(f"  {'=' * 20}")
        lines.append(f"  TOTAL: {mem['total']:.2f} MB")
        lines.append("=" * 80)

        return "\n".join(lines)


def build_octree(
    element_centroids: np.ndarray,
    element_IDs: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    max_elements_per_node: int = 500,
    max_depth: int = 10,
    verbose: bool = True,
    element_vertices: np.ndarray = None
) -> OctreeData:
    """
    Build octree using Morton code sorting.

    Uses top-down recursive subdivision:
    1. Sort elements by Morton code (Z-curve)
    2. Recursively split nodes that exceed max_elements_per_node
    3. Store in flat arrays

    Args:
        element_centroids: (N, 3) float64 - element center points (for sorting)
        element_IDs: (N,) int32 - element IDs to sort
        bbox_min: (3,) float64 - bounding box minimum
        bbox_max: (3,) float64 - bounding box maximum
        max_elements_per_node: Maximum elements before subdivision
        max_depth: Maximum tree depth
        verbose: Print progress
        element_vertices: (N, 4, 3) float64 - element vertices for bbox computation

    Returns:
        OctreeData object
    """
    n_elements = len(element_IDs)

    if verbose:
        print(f"\nBuilding octree for {n_elements:,} elements...")
        print(f"  Max elements/node: {max_elements_per_node}")
        print(f"  Max depth: {max_depth}")

    # Step 1: Compute Morton codes
    if verbose:
        print("  Computing Morton codes...")

    morton_codes = compute_morton_codes(element_centroids, bbox_min, bbox_max)

    # Step 2: Sort by Morton code
    if verbose:
        print("  Sorting elements by Z-curve...")

    sorted_morton, sorted_element_IDs = sort_by_morton_code(morton_codes, element_IDs)

    # Also create mapping from original indices to sorted indices
    # This is needed because sorted_element_IDs contains global element IDs
    local_to_global = element_IDs
    global_to_local = {gid: lid for lid, gid in enumerate(element_IDs)}
    sorted_centroids = element_centroids[[global_to_local[gid] for gid in sorted_element_IDs]]

    # Also sort element_vertices if provided (for proper bbox computation)
    if element_vertices is not None:
        sorted_element_vertices = element_vertices[[global_to_local[gid] for gid in sorted_element_IDs]]
    else:
        sorted_element_vertices = None

    # Step 3: Build octree nodes recursively
    if verbose:
        print("  Building octree nodes...")

    nodes = []

    def subdivide_node(start: int, end: int, depth: int,
                      node_min: np.ndarray, node_max: np.ndarray):
        """Recursively subdivide octree node."""
        n_elem = end - start

        # Create node
        node_id = len(nodes)
        nodes.append({
            'range': (start, end),
            'depth': depth,
            'bbox_min': node_min.copy(),
            'bbox_max': node_max.copy(),
        })

        # Check termination conditions
        if n_elem <= max_elements_per_node or depth >= max_depth:
            return node_id

        # Split into 8 children (octants)
        node_center = (node_min + node_max) / 2

        # Find split points in Morton code ordering
        # Elements are already sorted, so we use binary search on spatial location
        child_ranges = []

        for octant in range(8):
            # Octant bounding box
            child_min = np.array([
                node_min[0] if (octant & 1) == 0 else node_center[0],
                node_min[1] if (octant & 2) == 0 else node_center[1],
                node_min[2] if (octant & 4) == 0 else node_center[2],
            ])
            child_max = np.array([
                node_center[0] if (octant & 1) == 0 else node_max[0],
                node_center[1] if (octant & 2) == 0 else node_max[1],
                node_center[2] if (octant & 4) == 0 else node_max[2],
            ])

            # Find elements in this octant (using sorted centroids)
            in_octant = np.logical_and(
                np.all(sorted_centroids[start:end] >= child_min, axis=1),
                np.all(sorted_centroids[start:end] < child_max, axis=1)
            )

            indices = np.where(in_octant)[0] + start

            if len(indices) > 0:
                child_start = int(indices[0])
                child_end = int(indices[-1] + 1)

                # Compute actual bbox from element vertices in this child range
                # (if element_vertices provided, otherwise use spatial subdivision bbox)
                if sorted_element_vertices is not None:
                    # Get sorted element vertices for this range
                    child_vertices = sorted_element_vertices[child_start:child_end]  # (N_child, 4, 3)
                    child_min_actual = child_vertices.reshape(-1, 3).min(axis=0)
                    child_max_actual = child_vertices.reshape(-1, 3).max(axis=0)
                    child_ranges.append((child_start, child_end, child_min_actual, child_max_actual))
                else:
                    child_ranges.append((child_start, child_end, child_min, child_max))

        # Recursively subdivide non-empty children
        for child_start, child_end, child_min, child_max in child_ranges:
            if child_end - child_start > 0:
                subdivide_node(child_start, child_end, depth + 1, child_min, child_max)

        return node_id

    # Build from root
    subdivide_node(0, n_elements, 0, bbox_min, bbox_max)

    if verbose:
        print(f"  Created {len(nodes):,} octree nodes")
        depths = [node['depth'] for node in nodes]
        print(f"  Max depth: {max(depths)}")
        print(f"  Nodes per depth: {np.bincount([node['depth'] for node in nodes])}")

    # Step 4: Convert to flat arrays
    if verbose:
        print("  Converting to flat arrays...")

    n_nodes = len(nodes)

    node_ranges = np.zeros((n_nodes, 2), dtype=np.int32)
    node_depths = np.zeros(n_nodes, dtype=np.int32)
    node_bbox_min = np.zeros((n_nodes, 3), dtype=np.float64)
    node_bbox_max = np.zeros((n_nodes, 3), dtype=np.float64)

    for i, node in enumerate(nodes):
        node_ranges[i] = node['range']
        node_depths[i] = node['depth']
        node_bbox_min[i] = node['bbox_min']
        node_bbox_max[i] = node['bbox_max']

    octree_data = OctreeData(
        sorted_element_IDs=sorted_element_IDs,
        element_morton_codes=sorted_morton,
        node_ranges=node_ranges,
        node_depths=node_depths,
        node_bbox_min=node_bbox_min,
        node_bbox_max=node_bbox_max,
        max_elements_per_node=max_elements_per_node,
    )

    if verbose:
        print(str(octree_data))

    return octree_data


def build_octrees_per_block(
    positions: np.ndarray,
    connectivity: np.ndarray,
    element_block_IDs: np.ndarray,
    partition_data,
    max_elements_per_node: int = 500,
    max_depth: int = 10,
    verbose: bool = True
) -> Dict[int, OctreeData]:
    """
    Build octrees for each block.

    Args:
        positions: (N_nodes, 3) float64
        connectivity: (N_elements, 4) int32
        element_block_IDs: (N_elements,) int32
        partition_data: BlockPartitionData
        max_elements_per_node: Max elements before subdivision
        max_depth: Max octree depth
        verbose: Print progress

    Returns:
        octrees: Dict[block_id -> OctreeData]
    """
    n_blocks = partition_data.n_blocks

    if verbose:
        print("=" * 80)
        print(f"BUILDING OCTREES FOR {n_blocks} BLOCKS")
        print("=" * 80)

    octrees = {}

    for block_id in range(n_blocks):
        # Find elements in this block
        block_mask = element_block_IDs == block_id
        block_element_IDs = np.where(block_mask)[0].astype(np.int32)

        if len(block_element_IDs) == 0:
            if verbose:
                print(f"\nBlock {block_id}: Empty, skipping")
            continue

        if verbose:
            print(f"\nBlock {block_id}: {len(block_element_IDs):,} elements")

        # Compute element centroids for this block
        block_centroids = positions[connectivity[block_element_IDs]].mean(axis=1)

        # Compute block bounding box from actual element vertices (not centroids!)
        # This ensures the bbox contains all element vertices, not just centroids
        block_element_vertices = positions[connectivity[block_element_IDs]]  # (N_elem, 4, 3)
        block_bbox_min = block_element_vertices.reshape(-1, 3).min(axis=0)
        block_bbox_max = block_element_vertices.reshape(-1, 3).max(axis=0)

        # Build octree
        octree = build_octree(
            block_centroids,
            block_element_IDs,
            block_bbox_min,
            block_bbox_max,
            max_elements_per_node=max_elements_per_node,
            max_depth=max_depth,
            verbose=verbose,
            element_vertices=block_element_vertices
        )

        octrees[block_id] = octree

    if verbose:
        print("\n" + "=" * 80)
        print("OCTREE BUILD COMPLETE")
        print("=" * 80)
        total_mem = sum(oct.memory_usage_mb()['total'] for oct in octrees.values())
        total_nodes = sum(oct.n_nodes for oct in octrees.values())
        print(f"Total octree nodes: {total_nodes:,}")
        print(f"Total memory: {total_mem:.2f} MB")
        print("=" * 80)

    return octrees


if __name__ == "__main__":
    print("Testing octree builder...")

    # Test with synthetic mesh
    try:
        from .test_meshes import generate_test_mesh, SMALL_BALANCED_MESH
        from .mesh_loader import assign_elements_to_blocks
    except ImportError:
        from test_meshes import generate_test_mesh, SMALL_BALANCED_MESH
        from mesh_loader import assign_elements_to_blocks

    # Generate test mesh
    positions, connectivity = generate_test_mesh(SMALL_BALANCED_MESH)

    # Assign to blocks
    element_block_IDs, partition_data = assign_elements_to_blocks(
        positions, connectivity, (2, 2, 2), verbose=False
    )

    # Build octrees
    octrees = build_octrees_per_block(
        positions,
        connectivity,
        element_block_IDs,
        partition_data,
        max_elements_per_node=100,
        max_depth=5,
        verbose=True
    )

    print("\n✅ Octree builder test complete!")
