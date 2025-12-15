"""
Octree builder for GPU-native L2 fallback search.

This module provides CPU-side octree construction for level-filtered elements,
which can then be uploaded to GPU for scan-based traversal.

Key features:
- Level-based filtering: Build octree only for user-specified refinement level
- Fixed-size nodes: All nodes padded to max_leaf_size for GPU compatibility
- Recursive subdivision: Standard octree construction with max_depth limit
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np


@dataclass
class OctreeNode:
    """
    Single octree node (branch or leaf).

    Attributes
    ----------
    is_leaf : bool
        True if leaf node, False if branch node
    bbox_min : np.ndarray, shape (3,)
        Minimum corner of bounding box
    bbox_max : np.ndarray, shape (3,)
        Maximum corner of bounding box
    children : np.ndarray, shape (8,), dtype=int32
        Child node indices (-1 if empty child)
    elements : np.ndarray, shape (max_leaf_size,), dtype=int32
        Element IDs in leaf (-1 padding if fewer elements)
    depth : int
        Depth of node in tree (root = 0)
    """
    is_leaf: bool
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    children: np.ndarray  # (8,) node IDs
    elements: np.ndarray  # (max_leaf_size,) element IDs
    depth: int


def compute_element_bboxes(
    element_ids: np.ndarray,
    node_positions: np.ndarray,
    connectivity: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute bounding boxes for elements (vectorized).

    Parameters
    ----------
    element_ids : np.ndarray, shape (n_elements,)
        Element IDs to compute bboxes for
    node_positions : np.ndarray, shape (n_nodes, 3)
        Mesh node positions
    connectivity : np.ndarray, shape (n_elements_total, 4)
        Element-to-node connectivity

    Returns
    -------
    bbox_mins : np.ndarray, shape (n_elements, 3)
        Minimum corners of element bboxes
    bbox_maxs : np.ndarray, shape (n_elements, 3)
        Maximum corners of element bboxes
    """
    # Vectorized bbox computation
    # Get node IDs for all elements at once
    elem_node_ids = connectivity[element_ids]  # Shape: (n_elems, 4)

    # Get coordinates for all nodes of all elements
    # Shape: (n_elems, 4, 3)
    elem_nodes = node_positions[elem_node_ids]

    # Compute min/max across the 4 nodes
    bbox_mins = elem_nodes.min(axis=1).astype(np.float32)  # Shape: (n_elems, 3)
    bbox_maxs = elem_nodes.max(axis=1).astype(np.float32)  # Shape: (n_elems, 3)

    return bbox_mins, bbox_maxs


def bbox_intersects(
    bbox1_min: np.ndarray,
    bbox1_max: np.ndarray,
    bbox2_min: np.ndarray,
    bbox2_max: np.ndarray
) -> bool:
    """
    Check if two axis-aligned bounding boxes intersect.

    Parameters
    ----------
    bbox1_min : np.ndarray, shape (3,)
        Minimum corner of first bbox
    bbox1_max : np.ndarray, shape (3,)
        Maximum corner of first bbox
    bbox2_min : np.ndarray, shape (3,)
        Minimum corner of second bbox
    bbox2_max : np.ndarray, shape (3,)
        Maximum corner of second bbox

    Returns
    -------
    intersects : bool
        True if bboxes intersect (including touching)
    """
    # Two boxes intersect if they overlap on all 3 axes
    return (
        (bbox1_min[0] <= bbox2_max[0]) and (bbox1_max[0] >= bbox2_min[0]) and
        (bbox1_min[1] <= bbox2_max[1]) and (bbox1_max[1] >= bbox2_min[1]) and
        (bbox1_min[2] <= bbox2_max[2]) and (bbox1_max[2] >= bbox2_min[2])
    )


def build_octree_for_level(
    element_centroids: np.ndarray,
    element_ids: np.ndarray,
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    level_field: Optional[np.ndarray] = None,
    level_threshold: float = 7.0,
    max_depth: int = 10,
    max_leaf_size: int = 500,
    bbox_min: Optional[np.ndarray] = None,
    bbox_max: Optional[np.ndarray] = None,
    use_levelset: bool = False
) -> Tuple[List[OctreeNode], dict]:
    """
    Build octree for filtered elements using bbox-based assignment.

    **CRITICAL FIX**: Elements are now assigned to ALL leaves their bounding boxes
    intersect, not just the leaf containing their centroid. This ensures particles
    inside an element will find that element during octree search, even if they
    navigate to a different leaf than the element's centroid.

    Parameters
    ----------
    element_centroids : np.ndarray, shape (n_elements, 3)
        Element centroid positions (used for initial filtering only)
    element_ids : np.ndarray, shape (n_elements,)
        Element IDs (global indices)
    node_positions : np.ndarray, shape (n_nodes, 3)
        Mesh node positions (for computing element bboxes)
    connectivity : np.ndarray, shape (n_elements, 4)
        Element-to-node connectivity (for computing element bboxes)
    level_field : np.ndarray, shape (n_elements,), optional
        Refinement level or levelset per element. If None, use all elements.
    level_threshold : float, default=7.0
        Threshold for filtering:
        - If use_levelset=False: Include elements with level >= threshold (refinement)
        - If use_levelset=True: Include elements with level < threshold (levelset interface)
    max_depth : int, default=10
        Maximum tree depth
    max_leaf_size : int, default=500
        Maximum elements per leaf node (NOTE: with bbox-based assignment,
        leaves may contain more elements due to duplication)
    bbox_min : np.ndarray, shape (3,), optional
        Minimum corner of domain. If None, computed from data.
    bbox_max : np.ndarray, shape (3,), optional
        Maximum corner of domain. If None, computed from data.
    use_levelset : bool, default=False
        If True, use levelset filtering (level < threshold).
        If False, use refinement level filtering (level >= threshold).

    Returns
    -------
    nodes : List[OctreeNode]
        Flattened octree nodes (depth-first order)
    metadata : dict
        Octree statistics:
        - n_elements: Number of filtered elements (unique)
        - n_elements_duplicated: Total element slots (with duplication)
        - duplication_factor: Ratio of duplicated to unique elements
        - n_nodes: Total nodes (branches + leaves)
        - n_leaves: Number of leaf nodes
        - max_depth: Actual maximum depth reached
        - memory_mb: Estimated memory (MB)
    """
    # Filter elements by level
    if level_field is not None:
        if use_levelset:
            # Levelset mode: include elements where levelset < threshold (near interface)
            mask = level_field < level_threshold
        else:
            # Refinement level mode: include elements where level >= threshold
            mask = level_field >= level_threshold

        filtered_centroids = element_centroids[mask]
        filtered_ids = element_ids[mask]
    else:
        filtered_centroids = element_centroids
        filtered_ids = element_ids

    n_elements_filtered = len(filtered_ids)

    if n_elements_filtered == 0:
        if use_levelset:
            raise ValueError(f"No elements with levelset < {level_threshold}")
        else:
            raise ValueError(f"No elements with level >= {level_threshold}")

    # Compute element bboxes for all filtered elements
    print(f"Computing bboxes for {n_elements_filtered:,} elements...")
    elem_bbox_mins, elem_bbox_maxs = compute_element_bboxes(
        filtered_ids, node_positions, connectivity
    )

    # Compute domain bounding box
    if bbox_min is None:
        bbox_min = elem_bbox_mins.min(axis=0)
    if bbox_max is None:
        bbox_max = elem_bbox_maxs.max(axis=0)

    # Expand bbox slightly to avoid boundary issues
    bbox_size = bbox_max - bbox_min
    bbox_min -= 0.01 * bbox_size
    bbox_max += 0.01 * bbox_size

    # Build tree structure using centroid-based subdivision (PASS 1)
    # This creates the octree hierarchy
    nodes = []
    stats = {'n_leaves': 0, 'max_depth_reached': 0, 'total_element_slots': 0}

    def build_recursive(
        centroids: np.ndarray,
        elem_ids: np.ndarray,
        bbox_min_local: np.ndarray,
        bbox_max_local: np.ndarray,
        depth: int
    ) -> int:
        """
        Recursively build octree structure (PASS 1: create tree hierarchy).

        Uses centroid-based subdivision to determine tree structure.
        Element assignment will be done in PASS 2 using bbox intersection.

        Returns
        -------
        node_id : int
            Index of created node in nodes list
        """
        n_elems = len(elem_ids)

        # Update stats
        stats['max_depth_reached'] = max(stats['max_depth_reached'], depth)

        # Leaf condition: too few elements or max depth
        if n_elems <= max_leaf_size or depth >= max_depth:
            # Create leaf node (elements will be filled in PASS 2)
            node = OctreeNode(
                is_leaf=True,
                bbox_min=bbox_min_local.copy(),
                bbox_max=bbox_max_local.copy(),
                children=np.full(8, -1, dtype=np.int32),
                elements=np.full(max_leaf_size, -1, dtype=np.int32),  # Filled in PASS 2
                depth=depth
            )

            node_id = len(nodes)
            nodes.append(node)
            stats['n_leaves'] += 1

            return node_id

        # Branch node: Subdivide into 8 octants
        bbox_mid = (bbox_min_local + bbox_max_local) / 2.0

        # Vectorized octant computation for all centroids
        # Binary encoding: ix + 2*iy + 4*iz
        ix = (centroids[:, 0] >= bbox_mid[0]).astype(np.int32)
        iy = (centroids[:, 1] >= bbox_mid[1]).astype(np.int32)
        iz = (centroids[:, 2] >= bbox_mid[2]).astype(np.int32)
        octant_assignments = ix + (iy << 1) + (iz << 2)

        # Group elements by octant
        octant_masks = []
        for target_octant in range(8):
            mask = (octant_assignments == target_octant)

            # Compute octant bounding box
            ix = target_octant & 1
            iy = (target_octant >> 1) & 1
            iz = (target_octant >> 2) & 1

            x_min = bbox_min_local[0] if ix == 0 else bbox_mid[0]
            x_max = bbox_mid[0] if ix == 0 else bbox_max_local[0]
            y_min = bbox_min_local[1] if iy == 0 else bbox_mid[1]
            y_max = bbox_mid[1] if iy == 0 else bbox_max_local[1]
            z_min = bbox_min_local[2] if iz == 0 else bbox_mid[2]
            z_max = bbox_mid[2] if iz == 0 else bbox_max_local[2]

            octant_masks.append((mask, np.array([x_min, y_min, z_min]), np.array([x_max, y_max, z_max])))

        # Create branch node
        node = OctreeNode(
            is_leaf=False,
            bbox_min=bbox_min_local.copy(),
            bbox_max=bbox_max_local.copy(),
            children=np.full(8, -1, dtype=np.int32),
            elements=np.full(max_leaf_size, -1, dtype=np.int32),  # Not used for branches
            depth=depth
        )

        node_id = len(nodes)
        nodes.append(node)

        # Recursively build children
        for octant_idx, (mask, oct_min, oct_max) in enumerate(octant_masks):
            if mask.sum() > 0:
                child_id = build_recursive(
                    centroids[mask],
                    elem_ids[mask],
                    oct_min,
                    oct_max,
                    depth + 1
                )
                nodes[node_id].children[octant_idx] = child_id

        return node_id

    # PASS 1: Build tree from root (using centroid-based subdivision)
    print(f"PASS 1: Building octree structure...")
    root_id = build_recursive(
        filtered_centroids,
        filtered_ids,
        bbox_min,
        bbox_max,
        depth=0
    )

    assert root_id == 0, "Root should be first node"
    print(f"  Created {len(nodes):,} nodes ({stats['n_leaves']:,} leaves, depth {stats['max_depth_reached']})")

    # PASS 2: Assign elements to leaves based on bbox intersection
    print(f"PASS 2: Assigning elements to leaves using bbox intersection...")

    # Collect all leaf nodes
    leaf_nodes = [(i, node) for i, node in enumerate(nodes) if node.is_leaf]
    print(f"  Checking {n_elements_filtered:,} elements against {len(leaf_nodes):,} leaves...")

    # For each element, navigate its centroid to find the leaf, then check that leaf
    # and its neighbors (within the octree). This is much faster than checking ALL leaves.
    # However, for simplicity and correctness, we'll use a spatial recursion approach:
    # traverse the octree and for each leaf, find all elements that intersect it.

    def find_intersecting_elements_recursive(
        node_id: int,
        elem_bbox_mins: np.ndarray,
        elem_bbox_maxs: np.ndarray
    ) -> None:
        """
        Recursively traverse octree and assign elements to leaves they intersect.

        Parameters
        ----------
        node_id : int
            Current node ID
        elem_bbox_mins : np.ndarray, shape (n_elements, 3)
            Element bbox minimums
        elem_bbox_maxs : np.ndarray, shape (n_elements, 3)
            Element bbox maximums
        """
        node = nodes[node_id]

        if node.is_leaf:
            # Leaf node: check which elements intersect this leaf's bbox
            # Vectorized bbox intersection check
            node_min = node.bbox_min
            node_max = node.bbox_max

            # Check if element bbox intersects leaf bbox
            # Two bboxes intersect if they overlap on all 3 axes
            intersects = (
                (elem_bbox_mins[:, 0] <= node_max[0]) & (elem_bbox_maxs[:, 0] >= node_min[0]) &
                (elem_bbox_mins[:, 1] <= node_max[1]) & (elem_bbox_maxs[:, 1] >= node_min[1]) &
                (elem_bbox_mins[:, 2] <= node_max[2]) & (elem_bbox_maxs[:, 2] >= node_min[2])
            )

            # Get indices of intersecting elements
            intersecting_elem_indices = np.where(intersects)[0]

            # Add these elements to this leaf
            for elem_idx in intersecting_elem_indices:
                elem_id = filtered_ids[elem_idx]

                # Find first empty slot (-1)
                empty_idx = np.where(node.elements == -1)[0]

                if len(empty_idx) == 0:
                    # Leaf is full - need to expand
                    old_size = len(node.elements)
                    new_elements = np.full(old_size * 2, -1, dtype=np.int32)
                    new_elements[:old_size] = node.elements
                    new_elements[old_size] = elem_id
                    node.elements = new_elements
                else:
                    # Add to first empty slot
                    node.elements[empty_idx[0]] = elem_id
        else:
            # Branch node: recurse into children
            for child_id in node.children:
                if child_id >= 0:  # Valid child
                    find_intersecting_elements_recursive(child_id, elem_bbox_mins, elem_bbox_maxs)

    # Start recursive assignment from root
    find_intersecting_elements_recursive(0, elem_bbox_mins, elem_bbox_maxs)

    # Compute stats from assigned elements
    total_assignments = 0
    max_elements_in_leaf = 0

    for node in nodes:
        if node.is_leaf:
            n_elems_in_leaf = np.sum(node.elements >= 0)
            total_assignments += n_elems_in_leaf
            max_elements_in_leaf = max(max_elements_in_leaf, n_elems_in_leaf)

    duplication_factor = total_assignments / n_elements_filtered
    print(f"  Total assignments: {total_assignments:,} (duplication factor: {duplication_factor:.2f}×)")
    print(f"  Max elements in any leaf: {max_elements_in_leaf:,}")

    # Resize all leaf element arrays to actual max size (for memory efficiency)
    actual_max_leaf_size = max(max_elements_in_leaf, max_leaf_size)
    for node in nodes:
        if node.is_leaf:
            if len(node.elements) != actual_max_leaf_size:
                new_elements = np.full(actual_max_leaf_size, -1, dtype=np.int32)
                n_to_copy = min(len(node.elements), actual_max_leaf_size)
                new_elements[:n_to_copy] = node.elements[:n_to_copy]
                node.elements = new_elements

    stats['total_element_slots'] = total_assignments

    # Compute metadata
    n_nodes = len(nodes)
    n_branches = n_nodes - stats['n_leaves']

    # Memory estimate (using actual_max_leaf_size)
    node_size = 1 + 6 * 4 + 8 * 4 + actual_max_leaf_size * 4  # bytes
    memory_bytes = n_nodes * node_size
    memory_mb = memory_bytes / (1024 ** 2)

    metadata = {
        'n_elements': n_elements_filtered,
        'n_elements_duplicated': total_assignments,
        'duplication_factor': duplication_factor,
        'n_nodes': n_nodes,
        'n_leaves': stats['n_leaves'],
        'n_branches': n_branches,
        'max_depth': stats['max_depth_reached'],
        'max_leaf_size': actual_max_leaf_size,
        'memory_mb': memory_mb,
        'bbox_min': bbox_min,
        'bbox_max': bbox_max
    }

    return nodes, metadata


def flatten_octree_to_arrays(
    nodes: List[OctreeNode],
    max_leaf_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten octree nodes to fixed-size arrays for GPU upload.

    Parameters
    ----------
    nodes : List[OctreeNode]
        Octree nodes (from build_octree_for_level)
    max_leaf_size : int, optional
        Maximum elements per leaf. If None, inferred from nodes.

    Returns
    -------
    node_metadata : np.ndarray, shape (n_nodes, 15), dtype=float32
        Node metadata array:
        - [0]: is_leaf (0.0 or 1.0)
        - [1:4]: bbox_min (x, y, z)
        - [4:7]: bbox_max (x, y, z)
        - [7:15]: children node IDs (8 values, -1 if empty)
    node_elements : np.ndarray, shape (n_nodes, max_leaf_size), dtype=int32
        Element IDs per node (only used for leaves, -1 padding)
    """
    n_nodes = len(nodes)

    # Infer max_leaf_size from nodes if not provided
    if max_leaf_size is None:
        max_leaf_size = max(len(node.elements) for node in nodes)

    # Metadata: is_leaf + bbox + children
    node_metadata = np.zeros((n_nodes, 15), dtype=np.float32)

    # Elements
    node_elements = np.full((n_nodes, max_leaf_size), -1, dtype=np.int32)

    for i, node in enumerate(nodes):
        # is_leaf
        node_metadata[i, 0] = 1.0 if node.is_leaf else 0.0

        # bbox_min
        node_metadata[i, 1:4] = node.bbox_min

        # bbox_max
        node_metadata[i, 4:7] = node.bbox_max

        # children (cast to float32 for metadata array)
        node_metadata[i, 7:15] = node.children.astype(np.float32)

        # elements (copy what fits)
        n_to_copy = min(len(node.elements), max_leaf_size)
        node_elements[i, :n_to_copy] = node.elements[:n_to_copy]

    return node_metadata, node_elements


def print_octree_stats(metadata: dict, verbose: bool = True):
    """
    Print octree statistics.

    Parameters
    ----------
    metadata : dict
        Metadata from build_octree_for_level
    verbose : bool, default=True
        Print detailed statistics
    """
    if not verbose:
        return

    print("Octree Statistics:")
    print(f"  Elements (unique): {metadata['n_elements']:,}")
    if 'n_elements_duplicated' in metadata:
        print(f"  Element assignments (with duplication): {metadata['n_elements_duplicated']:,}")
        print(f"  Duplication factor: {metadata['duplication_factor']:.2f}×")
    print(f"  Total nodes: {metadata['n_nodes']:,}")
    print(f"  Branch nodes: {metadata['n_branches']:,}")
    print(f"  Leaf nodes: {metadata['n_leaves']:,}")
    print(f"  Max depth: {metadata['max_depth']}")
    print(f"  Max leaf size: {metadata['max_leaf_size']}")
    print(f"  Memory estimate: {metadata['memory_mb']:.2f} MB")
    print(f"  Bounding box:")
    print(f"    Min: [{metadata['bbox_min'][0]:.6f}, {metadata['bbox_min'][1]:.6f}, {metadata['bbox_min'][2]:.6f}]")
    print(f"    Max: [{metadata['bbox_max'][0]:.6f}, {metadata['bbox_max'][1]:.6f}, {metadata['bbox_max'][2]:.6f}]")
