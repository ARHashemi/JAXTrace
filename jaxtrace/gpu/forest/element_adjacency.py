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
