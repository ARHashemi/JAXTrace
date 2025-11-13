"""
Element Neighbor Precomputation.

Extracts face-adjacency relationships from tetrahedral mesh connectivity.
Used for Level 1 search (neighbor element checking) in three-tier search.
"""

from typing import Dict, List, Tuple
import numpy as np


def build_element_adjacency(
    connectivity: np.ndarray,
    max_neighbors: int = 4
) -> np.ndarray:
    """
    Build face-adjacency graph for tetrahedral mesh.

    Two tetrahedral elements are neighbors if they share a triangular face
    (3 common nodes). Each tetrahedron has 4 faces, so up to 4 neighbors.

    Args:
        connectivity: Element connectivity [N_elements, 4] (tetrahedral)
        max_neighbors: Maximum neighbors per element (4 for tetrahedra)

    Returns:
        neighbors: Neighbor element IDs [N_elements, max_neighbors]
                  -1 indicates no neighbor (boundary face)

    Example:
        >>> connectivity = mesh.get_connectivity()  # [3.5M, 4]
        >>> neighbors = build_element_adjacency(connectivity)
        >>> # Result: [3.5M, 4] array of neighbor IDs
        >>> neighbors[100]  # Neighbors of element 100
        array([101, 98, -1, 105])  # 3 neighbors, 1 boundary face
    """
    n_elements = connectivity.shape[0]

    print(f"Building element adjacency for {n_elements:,} elements...")

    # Step 1: Build face-to-elements mapping
    print("  Step 1/2: Extracting faces...")
    face_to_elements = build_face_to_elements_map(connectivity)

    # Step 2: Build neighbor array from face mapping
    print("  Step 2/2: Building neighbor lists...")
    neighbors = np.full((n_elements, max_neighbors), -1, dtype=np.int32)

    for elem_id in range(n_elements):
        neighbor_set = set()

        # Get all faces of this element
        faces = get_tetrahedral_faces(connectivity[elem_id])

        # For each face, find adjacent element
        for face in faces:
            face_key = tuple(sorted(face))
            if face_key in face_to_elements:
                # Get elements sharing this face
                adjacent_elems = face_to_elements[face_key]

                # Add neighbors (excluding self)
                for adj_elem in adjacent_elems:
                    if adj_elem != elem_id:
                        neighbor_set.add(adj_elem)

        # Store up to max_neighbors neighbors
        neighbor_list = list(neighbor_set)[:max_neighbors]
        neighbors[elem_id, :len(neighbor_list)] = neighbor_list

        # Progress reporting
        if (elem_id + 1) % 500000 == 0:
            print(f"    Processed {elem_id + 1:,} / {n_elements:,} elements "
                  f"({100 * (elem_id + 1) / n_elements:.1f}%)")

    # Statistics
    print_neighbor_statistics(neighbors)

    return neighbors


def build_face_to_elements_map(
    connectivity: np.ndarray
) -> Dict[Tuple[int, int, int], List[int]]:
    """
    Build mapping from faces (3-tuples of node IDs) to element IDs.

    For tetrahedral mesh, each face is a triangle defined by 3 nodes.
    Faces are stored in sorted order to enable matching.

    Args:
        connectivity: Element connectivity [N_elements, 4]

    Returns:
        face_to_elements: Dict mapping face (sorted node tuple) → [element IDs]
    """
    n_elements = connectivity.shape[0]
    face_to_elements = {}

    for elem_id in range(n_elements):
        faces = get_tetrahedral_faces(connectivity[elem_id])

        for face in faces:
            # Use sorted tuple as key for consistent matching
            face_key = tuple(sorted(face))

            if face_key not in face_to_elements:
                face_to_elements[face_key] = []

            face_to_elements[face_key].append(elem_id)

    return face_to_elements


def get_tetrahedral_faces(nodes: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Get 4 triangular faces of a tetrahedral element.

    A tetrahedron with nodes [0, 1, 2, 3] has faces:
    - (0, 1, 2)
    - (0, 1, 3)
    - (0, 2, 3)
    - (1, 2, 3)

    Args:
        nodes: Node IDs [4]

    Returns:
        faces: List of 4 faces, each face is (node0, node1, node2)
    """
    return [
        (nodes[0], nodes[1], nodes[2]),
        (nodes[0], nodes[1], nodes[3]),
        (nodes[0], nodes[2], nodes[3]),
        (nodes[1], nodes[2], nodes[3])
    ]


def print_neighbor_statistics(neighbors: np.ndarray):
    """
    Print neighbor distribution statistics.

    Args:
        neighbors: Neighbor array [N_elements, max_neighbors]
    """
    n_elements = neighbors.shape[0]
    max_neighbors = neighbors.shape[1]

    # Count neighbors per element
    n_neighbors_per_elem = np.sum(neighbors != -1, axis=1)

    mean_neighbors = np.mean(n_neighbors_per_elem)
    min_neighbors = np.min(n_neighbors_per_elem)
    max_neighbors_found = np.max(n_neighbors_per_elem)

    # Count boundary elements (elements with < max_neighbors neighbors)
    boundary_elements = np.sum(n_neighbors_per_elem < max_neighbors)

    print(f"\n📊 Element Neighbor Statistics:")
    print(f"  Total elements: {n_elements:,}")
    print(f"  Neighbors per element:")
    print(f"    Min: {min_neighbors}")
    print(f"    Max: {max_neighbors_found}")
    print(f"    Mean: {mean_neighbors:.2f}")
    print(f"  Boundary elements: {boundary_elements:,} "
          f"({100 * boundary_elements / n_elements:.1f}%)")
    print(f"  Interior elements: {n_elements - boundary_elements:,} "
          f"({100 * (n_elements - boundary_elements) / n_elements:.1f}%)")


def validate_neighbor_symmetry(
    neighbors: np.ndarray,
    connectivity: np.ndarray,
    sample_size: int = 1000
) -> bool:
    """
    Validate that neighbor relationships are symmetric.

    If element A is a neighbor of element B, then B should be a neighbor of A.
    Uses sampling for large meshes.

    Args:
        neighbors: Neighbor array [N_elements, max_neighbors]
        connectivity: Element connectivity [N_elements, 4]
        sample_size: Number of elements to check

    Returns:
        True if validation passes, False otherwise
    """
    n_elements = neighbors.shape[0]
    sample_ids = np.random.choice(n_elements, min(sample_size, n_elements), replace=False)

    errors = 0

    for elem_id in sample_ids:
        elem_neighbors = neighbors[elem_id]

        for neighbor_id in elem_neighbors:
            if neighbor_id == -1:
                continue

            # Check if elem_id appears in neighbor_id's neighbor list
            reverse_neighbors = neighbors[neighbor_id]

            if elem_id not in reverse_neighbors:
                errors += 1
                if errors <= 5:  # Print first 5 errors
                    print(f"  ⚠️  Asymmetry: {elem_id} → {neighbor_id}, "
                          f"but {neighbor_id} ↛ {elem_id}")

    if errors > 0:
        print(f"⚠️  Warning: {errors} asymmetric neighbor relationships found "
              f"(out of {sample_size} sampled)")
        return False
    else:
        print(f"✅ Neighbor symmetry validated (sampled {sample_size} elements)")
        return True
