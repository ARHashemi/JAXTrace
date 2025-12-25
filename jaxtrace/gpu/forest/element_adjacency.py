"""
Element face-adjacency extraction for tetrahedral meshes.

Part of Phase 2: Element Neighbors & Padded Block Arrays

Builds element neighbor relationships based on shared faces (3 nodes).
Critical for L1 neighbor cache optimization (85-95% hit rate).
"""

import numpy as np
from typing import Tuple, Dict, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class AdjacencyStats:
    """Statistics about element adjacency."""
    n_elements: int
    n_faces: int
    n_boundary_faces: int
    n_internal_faces: int
    avg_neighbors_per_element: float
    max_neighbors_per_element: int
    min_neighbors_per_element: int
    
    def __repr__(self) -> str:
        return (
            f"AdjacencyStats(\n"
            f"  Elements: {self.n_elements:,}\n"
            f"  Total faces: {self.n_faces:,}\n"
            f"  Internal faces: {self.n_internal_faces:,}\n"
            f"  Boundary faces: {self.n_boundary_faces:,}\n"
            f"  Neighbors per element: min={self.min_neighbors_per_element}, "
            f"max={self.max_neighbors_per_element}, avg={self.avg_neighbors_per_element:.2f}\n"
            f")"
        )


def get_tet_faces(element_id: int, nodes: np.ndarray) -> np.ndarray:
    """
    Get the 4 faces of a tetrahedral element.
    
    Each face is defined by 3 nodes in sorted order for consistent hashing.
    
    Parameters
    ----------
    element_id : int
        Element ID (for reference)
    nodes : np.ndarray
        Node IDs for the tet, shape (4,), int32
        
    Returns
    -------
    faces : np.ndarray
        Four faces, shape (4, 3), int32, each row sorted
        
    Notes
    -----
    Tet node ordering (standard VTK):
        3
       /|\\
      / | \\
     /  |  \\
    0---+---1
     \\  |  /
      \\ | /
       \\|/
        2
    
    Faces (opposite to node):
    - Face 0 (opposite node 0): [1, 2, 3]
    - Face 1 (opposite node 1): [0, 2, 3]
    - Face 2 (opposite node 2): [0, 1, 3]
    - Face 3 (opposite node 3): [0, 1, 2]
    """
    faces = np.array([
        [nodes[1], nodes[2], nodes[3]],  # Face opposite to node 0
        [nodes[0], nodes[2], nodes[3]],  # Face opposite to node 1
        [nodes[0], nodes[1], nodes[3]],  # Face opposite to node 2
        [nodes[0], nodes[1], nodes[2]],  # Face opposite to node 3
    ], dtype=np.int32)
    
    # Sort each face for consistent hashing
    faces.sort(axis=1)
    
    return faces


def build_face_to_element_map(
    connectivity: np.ndarray,
    verbose: bool = False
) -> Dict[Tuple[int, int, int], Set[int]]:
    """
    Build mapping from faces to elements that contain them.
    
    Parameters
    ----------
    connectivity : np.ndarray
        Element connectivity, shape (N_elements, 4), int32
    verbose : bool, optional
        Print progress messages (default: False)
        
    Returns
    -------
    face_to_elements : Dict[Tuple[int, int, int], Set[int]]
        Maps face (as sorted tuple of 3 nodes) to set of element IDs
        
    Notes
    -----
    - Internal faces appear in exactly 2 elements
    - Boundary faces appear in exactly 1 element
    - O(N_elements) complexity
    """
    n_elements = connectivity.shape[0]
    face_to_elements = defaultdict(set)
    
    if verbose:
        print(f"Building face-to-element map for {n_elements:,} elements...")
    
    for elem_id in range(n_elements):
        nodes = connectivity[elem_id]
        faces = get_tet_faces(elem_id, nodes)
        
        for face in faces:
            face_tuple = tuple(face)
            face_to_elements[face_tuple].add(elem_id)
        
        if verbose and (elem_id + 1) % 500000 == 0:
            print(f"  Processed {elem_id + 1:,}/{n_elements:,} elements...")
    
    if verbose:
        n_faces = len(face_to_elements)
        print(f"  Total unique faces: {n_faces:,}")
    
    return face_to_elements


def extract_element_neighbors(
    connectivity: np.ndarray,
    verbose: bool = False
) -> Tuple[Dict[int, np.ndarray], AdjacencyStats]:
    """
    Extract element face-adjacency neighbors.
    
    Two elements are neighbors if they share a face (3 nodes).
    
    Parameters
    ----------
    connectivity : np.ndarray
        Element connectivity, shape (N_elements, 4), int32
    verbose : bool, optional
        Print progress messages (default: False)
        
    Returns
    -------
    neighbors : Dict[int, np.ndarray]
        Maps element_id -> array of neighbor element IDs (variable length)
    stats : AdjacencyStats
        Adjacency statistics
        
    Notes
    -----
    - Tetrahedral elements have 0-4 neighbors (boundary vs interior)
    - Interior elements typically have 4 neighbors
    - Boundary elements have fewer neighbors
    - O(N_elements) complexity
    """
    n_elements = connectivity.shape[0]
    
    if verbose:
        print(f"\nExtracting element neighbors for {n_elements:,} elements...")
    
    # Build face-to-element map
    face_to_elements = build_face_to_element_map(connectivity, verbose=verbose)
    
    # Extract neighbors
    if verbose:
        print("Building neighbor lists...")
    
    neighbors = {}
    neighbor_counts = []
    
    for elem_id in range(n_elements):
        nodes = connectivity[elem_id]
        faces = get_tet_faces(elem_id, nodes)
        
        neighbor_set = set()
        for face in faces:
            face_tuple = tuple(face)
            elements_on_face = face_to_elements[face_tuple]
            
            # Add all elements sharing this face except self
            for other_elem in elements_on_face:
                if other_elem != elem_id:
                    neighbor_set.add(other_elem)
        
        neighbor_array = np.array(sorted(neighbor_set), dtype=np.int32)
        neighbors[elem_id] = neighbor_array
        neighbor_counts.append(len(neighbor_array))
        
        if verbose and (elem_id + 1) % 500000 == 0:
            print(f"  Processed {elem_id + 1:,}/{n_elements:,} elements...")
    
    # Compute statistics
    n_faces = len(face_to_elements)
    n_boundary_faces = sum(1 for elems in face_to_elements.values() if len(elems) == 1)
    n_internal_faces = sum(1 for elems in face_to_elements.values() if len(elems) == 2)
    
    avg_neighbors = np.mean(neighbor_counts)
    max_neighbors = max(neighbor_counts)
    min_neighbors = min(neighbor_counts)
    
    stats = AdjacencyStats(
        n_elements=n_elements,
        n_faces=n_faces,
        n_boundary_faces=n_boundary_faces,
        n_internal_faces=n_internal_faces,
        avg_neighbors_per_element=avg_neighbors,
        max_neighbors_per_element=max_neighbors,
        min_neighbors_per_element=min_neighbors,
    )
    
    if verbose:
        print(f"\n{stats}")
    
    return neighbors, stats


def validate_neighbor_symmetry(
    neighbors: Dict[int, np.ndarray],
    n_samples: int = 1000
) -> bool:
    """
    Validate that neighbor relationships are symmetric.

    If element A is a neighbor of element B, then B must be a neighbor of A.

    Parameters
    ----------
    neighbors : Dict[int, np.ndarray]
        Neighbor dictionary
    n_samples : int, optional
        Number of random elements to check (default: 1000)

    Returns
    -------
    valid : bool
        True if all sampled relationships are symmetric
    """
    element_ids = list(neighbors.keys())
    n_samples = min(n_samples, len(element_ids))

    np.random.seed(42)
    sample_ids = np.random.choice(element_ids, size=n_samples, replace=False)

    n_errors = 0
    for elem_id in sample_ids:
        for neighbor_id in neighbors[elem_id]:
            # Check if elem_id is in neighbor's neighbor list
            if elem_id not in neighbors[neighbor_id]:
                n_errors += 1
                print(f"ERROR: Element {elem_id} has neighbor {neighbor_id}, "
                      f"but {neighbor_id} does not have {elem_id} as neighbor")

    if n_errors > 0:
        print(f"\nValidation FAILED: {n_errors} asymmetric relationships found")
        return False
    else:
        print(f"\nValidation PASSED: All {n_samples} sampled elements have symmetric neighbors")
        return True


def build_node_to_elements_map(
    connectivity: np.ndarray,
    verbose: bool = False
) -> Dict[int, Set[int]]:
    """
    Build mapping from nodes to elements that contain them.

    Parameters
    ----------
    connectivity : np.ndarray
        Element connectivity, shape (N_elements, 4), int32
    verbose : bool, optional
        Print progress messages (default: False)

    Returns
    -------
    node_to_elements : Dict[int, Set[int]]
        Maps node_id -> set of element IDs containing that node

    Notes
    -----
    - Used for node-based neighbor extraction (shared node = neighbor)
    - More inclusive than face-based neighbors (captures edge/vertex neighbors)
    - O(N_elements) complexity
    """
    n_elements = connectivity.shape[0]
    node_to_elements = defaultdict(set)

    if verbose:
        print(f"Building node-to-element map for {n_elements:,} elements...")

    for elem_id in range(n_elements):
        nodes = connectivity[elem_id]
        for node_id in nodes:
            node_to_elements[node_id].add(elem_id)

        if verbose and (elem_id + 1) % 500000 == 0:
            print(f"  Processed {elem_id + 1:,}/{n_elements:,} elements...")

    if verbose:
        n_nodes = len(node_to_elements)
        avg_elems_per_node = np.mean([len(elems) for elems in node_to_elements.values()])
        print(f"  Total nodes: {n_nodes:,}")
        print(f"  Avg elements per node: {avg_elems_per_node:.2f}")

    return node_to_elements


def extract_element_neighbors_node_based(
    connectivity: np.ndarray,
    verbose: bool = False
) -> Tuple[Dict[int, np.ndarray], AdjacencyStats]:
    """
    Extract element node-adjacency neighbors.

    Two elements are neighbors if they share ANY node (vertex, edge, or face).
    This is more inclusive than face-based neighbors and captures refined mesh
    boundaries where coarse and fine elements share edges but not faces.

    Parameters
    ----------
    connectivity : np.ndarray
        Element connectivity, shape (N_elements, 4), int32
    verbose : bool, optional
        Print progress messages (default: False)

    Returns
    -------
    neighbors : Dict[int, np.ndarray]
        Maps element_id -> array of neighbor element IDs (variable length)
    stats : AdjacencyStats
        Adjacency statistics

    Notes
    -----
    - Node-based neighbors include ALL elements sharing at least 1 node
    - This captures edge-sharing and vertex-sharing neighbors (not just face-sharing)
    - Critical for adaptively refined meshes where coarse/fine elements share edges
    - Results in MORE neighbors per element than face-based (can be 20-100+ neighbors)
    - O(N_elements) complexity

    Examples
    --------
    In octree-refined mesh:
    - Coarse cube (8 tets) refined to 8 sub-cubes (64 tets total)
    - Coarse element at boundary shares EDGE with 2 fine elements
    - Face-based: 0 fine neighbors (no shared face)
    - Node-based: 2 fine neighbors (shared edge = shared 2 nodes)
    """
    n_elements = connectivity.shape[0]

    if verbose:
        print(f"\nExtracting NODE-BASED element neighbors for {n_elements:,} elements...")
        print("  (Elements sharing ANY node are considered neighbors)")

    # Build node-to-element map
    node_to_elements = build_node_to_elements_map(connectivity, verbose=verbose)

    # Extract neighbors
    if verbose:
        print("Building neighbor lists from node connectivity...")

    neighbors = {}
    neighbor_counts = []

    for elem_id in range(n_elements):
        nodes = connectivity[elem_id]

        # Find all elements sharing any node with this element
        neighbor_set = set()
        for node_id in nodes:
            elements_on_node = node_to_elements[node_id]
            neighbor_set.update(elements_on_node)

        # Remove self from neighbor set
        neighbor_set.discard(elem_id)

        neighbor_array = np.array(sorted(neighbor_set), dtype=np.int32)
        neighbors[elem_id] = neighbor_array
        neighbor_counts.append(len(neighbor_array))

        if verbose and (elem_id + 1) % 500000 == 0:
            print(f"  Processed {elem_id + 1:,}/{n_elements:,} elements...")

    # Compute statistics
    avg_neighbors = np.mean(neighbor_counts)
    max_neighbors = max(neighbor_counts)
    min_neighbors = min(neighbor_counts)

    stats = AdjacencyStats(
        n_elements=n_elements,
        n_faces=0,  # Not applicable for node-based
        n_boundary_faces=0,
        n_internal_faces=0,
        avg_neighbors_per_element=avg_neighbors,
        max_neighbors_per_element=max_neighbors,
        min_neighbors_per_element=min_neighbors,
    )

    if verbose:
        print(f"\nNode-based neighbor statistics:")
        print(f"  Elements: {n_elements:,}")
        print(f"  Neighbors per element: min={min_neighbors}, max={max_neighbors}, avg={avg_neighbors:.2f}")
        print(f"  NOTE: Node-based typically has {avg_neighbors/4:.1f}x more neighbors than face-based")

    return neighbors, stats


def build_element_neighbors_array(
    connectivity: np.ndarray,
    verbose: bool = False,
    method: str = 'face'
) -> np.ndarray:
    """
    Build element neighbors array in fixed-size padded format.

    This is a convenience wrapper around extract_element_neighbors() that
    returns a padded array suitable for GPU processing and incremental search.

    Parameters
    ----------
    connectivity : np.ndarray
        Element connectivity, shape (N_elements, 4), int32
    verbose : bool, optional
        Print progress messages (default: False)
    method : str, optional
        Neighbor definition method (default: 'face')
        - 'face': Elements sharing a face (3 nodes) are neighbors
                  Returns shape (N_elements, 4) - at most 4 face neighbors
                  Memory efficient, but misses edge/vertex neighbors
        - 'node': Elements sharing ANY node are neighbors
                  Returns shape (N_elements, MAX_NEIGHBORS) - typically 20-100+ neighbors
                  Captures refined mesh boundaries (coarse/fine edge-sharing)
                  Higher memory usage but ensures all geometric neighbors found

    Returns
    -------
    element_neighbors : np.ndarray
        Element neighbors, shape depends on method:
        - 'face': (N_elements, 4), int32 - 4 face neighbors max
        - 'node': (N_elements, MAX_NEIGHBORS), int32 - variable per mesh
        Each row contains neighbor element IDs, -1 for missing neighbors

    Notes
    -----
    **Face-based (method='face'):**
    - Tetrahedral elements have at most 4 face neighbors (one per face)
    - Boundary elements have fewer neighbors (padded with -1)
    - Interior elements typically have 4 neighbors
    - Misses edge-sharing neighbors in adaptively refined meshes
    - Memory: ~48 MB for 3M elements

    **Node-based (method='node'):**
    - Elements sharing ANY node are neighbors (vertex, edge, or face)
    - Captures ALL geometric neighbors including refined mesh boundaries
    - Critical for octree-refined meshes where coarse/fine share edges
    - More neighbors per element (20-100+ typical)
    - Memory: ~600-1200 MB for 3M elements (depends on max_neighbors)

    **When to use node-based:**
    - Adaptively refined meshes with multi-level refinement
    - Friction stir welding (rotating tool with refined region)
    - Particles crossing coarse/fine boundaries
    - L1 neighbor search must find fine elements from coarse elements

    Examples
    --------
    >>> # Face-based (default): Element 42 has 3 face neighbors
    >>> neighbors = build_element_neighbors_array(connectivity, method='face')
    >>> neighbors[42]  # [15, 108, 201, -1]

    >>> # Node-based: Element 42 has 47 node neighbors (includes edge/vertex)
    >>> neighbors = build_element_neighbors_array(connectivity, method='node')
    >>> neighbors[42]  # [12, 15, 18, 21, ..., 201, 208, -1, -1, ...]
    """
    n_elements = connectivity.shape[0]

    if verbose:
        print(f"\nBuilding element neighbors array for {n_elements:,} elements...")
        print(f"  Method: {method}")

    # Extract neighbors as dictionary using chosen method
    if method == 'face':
        neighbors_dict, stats = extract_element_neighbors(connectivity, verbose=verbose)
        max_neighbors_padded = 4  # Tets have at most 4 face neighbors
    elif method == 'node':
        neighbors_dict, stats = extract_element_neighbors_node_based(connectivity, verbose=verbose)
        max_neighbors_padded = stats.max_neighbors_per_element
        if verbose:
            print(f"  Max neighbors found: {max_neighbors_padded}")
            memory_mb = (n_elements * max_neighbors_padded * 4) / (1024**2)
            print(f"  Estimated memory: {memory_mb:.1f} MB")
    else:
        raise ValueError(f"Invalid method '{method}'. Must be 'face' or 'node'.")

    # Create padded array
    padded_neighbors = np.full((n_elements, max_neighbors_padded), -1, dtype=np.int32)

    # Fill array from dictionary
    if verbose:
        print(f"  Converting to padded array ({n_elements}, {max_neighbors_padded})...")

    for elem_id in range(n_elements):
        neighbors_list = neighbors_dict[elem_id]
        n_neighbors = len(neighbors_list)

        if n_neighbors > max_neighbors_padded:
            # Truncate to max (shouldn't happen with correct max_neighbors_padded)
            if verbose:
                print(f"  WARNING: Element {elem_id} has {n_neighbors} neighbors, truncating to {max_neighbors_padded}")
            neighbors_list = neighbors_list[:max_neighbors_padded]
            n_neighbors = max_neighbors_padded

        padded_neighbors[elem_id, :n_neighbors] = neighbors_list

    if verbose:
        n_boundary = np.sum(np.any(padded_neighbors == -1, axis=1))
        n_interior = n_elements - n_boundary
        avg_actual = np.mean([len(neighbors_dict[i]) for i in range(n_elements)])
        print(f"  Elements with max neighbors: {n_interior:,} ({100*n_interior/n_elements:.1f}%)")
        print(f"  Elements with < max neighbors: {n_boundary:,} ({100*n_boundary/n_elements:.1f}%)")
        print(f"  Avg neighbors per element: {avg_actual:.2f}")
        print(f"  Element neighbors array built successfully!")

    return padded_neighbors
