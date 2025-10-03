"""
Adaptive Octree-based FEM Interpolator for JAXTrace

Optimized for meshes with adaptive refinement (multiple resolution levels).
Uses octree spatial partitioning to achieve O(log N) element lookup.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class OctreeNode:
    """Single node in the octree structure."""
    # Bounding box
    min_corner: np.ndarray  # (3,)
    max_corner: np.ndarray  # (3,)

    # Element storage
    element_indices: np.ndarray  # Elements in this node

    # Children (8 octants, -1 if leaf)
    children: np.ndarray  # (8,) indices into node array

    # Metadata
    is_leaf: bool
    depth: int


@dataclass
class OctreeMesh:
    """Octree-based tetrahedral mesh for efficient spatial queries."""

    # Mesh geometry
    points: jnp.ndarray          # (N, 3) node coordinates
    connectivity: jnp.ndarray     # (M, 4) tetrahedral connectivity
    element_bounds: jnp.ndarray   # (M, 2, 3) element bounding boxes

    # Octree structure (flattened for JAX compatibility)
    nodes_min: jnp.ndarray        # (num_nodes, 3) node min corners
    nodes_max: jnp.ndarray        # (num_nodes, 3) node max corners
    nodes_elements: jnp.ndarray   # (num_nodes, max_elems) element indices per node
    nodes_elem_counts: jnp.ndarray # (num_nodes,) number of elements per node
    nodes_children: jnp.ndarray   # (num_nodes, 8) children indices
    nodes_is_leaf: jnp.ndarray    # (num_nodes,) leaf flags

    # Configuration
    max_elements_per_leaf: int
    max_depth: int


def build_octree_mesh(points: np.ndarray,
                      connectivity: np.ndarray,
                      max_elements_per_leaf: int = 32,
                      max_depth: int = 12) -> OctreeMesh:
    """
    Build adaptive octree mesh structure.

    Parameters
    ----------
    points : np.ndarray
        Node coordinates (N, 3)
    connectivity : np.ndarray
        Tetrahedral connectivity (M, 4)
    max_elements_per_leaf : int
        Maximum elements before subdivision (default: 32)
    max_depth : int
        Maximum tree depth (default: 12)

    Returns
    -------
    OctreeMesh
        Octree-structured mesh for efficient queries
    """

    points = np.asarray(points, dtype=np.float32)
    connectivity = np.asarray(connectivity, dtype=np.int32)

    n_elements = connectivity.shape[0]

    print(f"🌲 Building adaptive octree:")
    print(f"   Mesh: {points.shape[0]} points, {n_elements} elements")
    print(f"   Max elements/leaf: {max_elements_per_leaf}")
    print(f"   Max depth: {max_depth}")

    # Compute element bounds and centroids
    print(f"   Computing element bounds...")
    element_bounds = np.zeros((n_elements, 2, 3), dtype=np.float32)
    element_centroids = np.zeros((n_elements, 3), dtype=np.float32)

    for elem_idx in range(n_elements):
        node_indices = connectivity[elem_idx]
        elem_points = points[node_indices]

        elem_min = elem_points.min(axis=0)
        elem_max = elem_points.max(axis=0)
        element_bounds[elem_idx, 0] = elem_min
        element_bounds[elem_idx, 1] = elem_max
        element_centroids[elem_idx] = elem_points.mean(axis=0)

        if (elem_idx + 1) % 200000 == 0:
            print(f"      {elem_idx + 1}/{n_elements} elements...")

    # Compute global bounding box
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)

    print(f"   Domain: {bbox_min} to {bbox_max}")

    # Build octree recursively
    print(f"   Building octree structure...")

    nodes = []  # List of OctreeNode objects

    def subdivide_node(min_corner, max_corner, elem_indices, depth):
        """Recursively subdivide space into octree."""

        node_idx = len(nodes)
        n_elems = len(elem_indices)

        # Check termination criteria
        is_leaf = (n_elems <= max_elements_per_leaf) or (depth >= max_depth)

        if is_leaf:
            # Create leaf node
            node = OctreeNode(
                min_corner=min_corner,
                max_corner=max_corner,
                element_indices=np.array(elem_indices, dtype=np.int32),
                children=np.full(8, -1, dtype=np.int32),
                is_leaf=True,
                depth=depth
            )
            nodes.append(node)
            return node_idx

        # Create internal node and subdivide
        center = (min_corner + max_corner) / 2.0

        # Initialize children list
        children = np.full(8, -1, dtype=np.int32)

        # Subdivide into 8 octants
        octant_bounds = [
            (min_corner, center),  # 0: ---
            (np.array([center[0], min_corner[1], min_corner[2]]), np.array([max_corner[0], center[1], center[2]])),  # 1: +--
            (np.array([min_corner[0], center[1], min_corner[2]]), np.array([center[0], max_corner[1], center[2]])),  # 2: -+-
            (np.array([center[0], center[1], min_corner[2]]), np.array([max_corner[0], max_corner[1], center[2]])),  # 3: ++-
            (np.array([min_corner[0], min_corner[1], center[2]]), np.array([center[0], center[1], max_corner[2]])),  # 4: --+
            (np.array([center[0], min_corner[1], center[2]]), np.array([max_corner[0], center[1], max_corner[2]])),  # 5: +-+
            (np.array([min_corner[0], center[1], center[2]]), np.array([center[0], max_corner[1], max_corner[2]])),  # 6: -++
            (center, max_corner),  # 7: +++
        ]

        # Distribute elements to octants
        octant_elements = [[] for _ in range(8)]

        for elem_idx in elem_indices:
            elem_centroid = element_centroids[elem_idx]

            # Determine octant based on centroid
            octant = 0
            if elem_centroid[0] >= center[0]: octant += 1
            if elem_centroid[1] >= center[1]: octant += 2
            if elem_centroid[2] >= center[2]: octant += 4

            octant_elements[octant].append(elem_idx)

        # Create node (will be updated with children)
        node = OctreeNode(
            min_corner=min_corner,
            max_corner=max_corner,
            element_indices=np.array([], dtype=np.int32),  # Internal nodes store no elements
            children=children,
            is_leaf=False,
            depth=depth
        )
        nodes.append(node)

        # Recursively create children
        for octant_idx in range(8):
            if len(octant_elements[octant_idx]) > 0:
                child_min, child_max = octant_bounds[octant_idx]
                child_node_idx = subdivide_node(
                    child_min, child_max,
                    octant_elements[octant_idx],
                    depth + 1
                )
                nodes[node_idx].children[octant_idx] = child_node_idx

        return node_idx

    # Build octree starting from root
    all_elements = list(range(n_elements))
    root_idx = subdivide_node(bbox_min, bbox_max, all_elements, depth=0)

    print(f"   ✅ Octree built: {len(nodes)} nodes")

    # Analyze octree structure
    leaf_nodes = [n for n in nodes if n.is_leaf]
    max_leaf_elems = max(len(n.element_indices) for n in leaf_nodes) if leaf_nodes else 0
    avg_leaf_elems = np.mean([len(n.element_indices) for n in leaf_nodes]) if leaf_nodes else 0
    max_tree_depth = max(n.depth for n in nodes)

    print(f"   Leaf nodes: {len(leaf_nodes)}")
    print(f"   Max depth reached: {max_tree_depth}")
    print(f"   Elements per leaf: avg={avg_leaf_elems:.1f}, max={max_leaf_elems}")

    # Convert to flattened arrays for JAX
    print(f"   Converting to JAX-compatible format...")

    num_nodes = len(nodes)
    max_elems = max(len(n.element_indices) for n in nodes)

    nodes_min = np.zeros((num_nodes, 3), dtype=np.float32)
    nodes_max = np.zeros((num_nodes, 3), dtype=np.float32)
    nodes_elements = np.full((num_nodes, max_elems), -1, dtype=np.int32)
    nodes_elem_counts = np.zeros(num_nodes, dtype=np.int32)
    nodes_children = np.full((num_nodes, 8), -1, dtype=np.int32)
    nodes_is_leaf = np.zeros(num_nodes, dtype=bool)

    for i, node in enumerate(nodes):
        nodes_min[i] = node.min_corner
        nodes_max[i] = node.max_corner
        n_elems = len(node.element_indices)
        if n_elems > 0:
            nodes_elements[i, :n_elems] = node.element_indices
        nodes_elem_counts[i] = n_elems
        nodes_children[i] = node.children
        nodes_is_leaf[i] = node.is_leaf

    # Create octree mesh
    octree_mesh = OctreeMesh(
        points=jnp.array(points),
        connectivity=jnp.array(connectivity),
        element_bounds=jnp.array(element_bounds),
        nodes_min=jnp.array(nodes_min),
        nodes_max=jnp.array(nodes_max),
        nodes_elements=jnp.array(nodes_elements),
        nodes_elem_counts=jnp.array(nodes_elem_counts),
        nodes_children=jnp.array(nodes_children),
        nodes_is_leaf=jnp.array(nodes_is_leaf),
        max_elements_per_leaf=max_elements_per_leaf,
        max_depth=max_depth
    )

    print(f"   ✅ Octree ready for GPU!")

    return octree_mesh


@jax.jit
def point_in_tetrahedron(point: jnp.ndarray,
                         tet_nodes: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Check if point is inside tetrahedron using barycentric coordinates.

    Same implementation as before - tested and working.
    """

    # Extract vertices
    v0, v1, v2, v3 = tet_nodes[0], tet_nodes[1], tet_nodes[2], tet_nodes[3]

    # Compute vectors
    v0p = point - v0
    v01 = v1 - v0
    v02 = v2 - v0
    v03 = v3 - v0

    # Compute 3x3 matrix determinant
    mat = jnp.stack([v01, v02, v03], axis=1)
    det = jnp.linalg.det(mat)

    # Avoid division by zero
    det = jnp.where(jnp.abs(det) < 1e-10, 1.0, det)

    # Barycentric coordinates using Cramer's rule
    mat1 = jnp.stack([v0p, v02, v03], axis=1)
    mat2 = jnp.stack([v01, v0p, v03], axis=1)
    mat3 = jnp.stack([v01, v02, v0p], axis=1)

    b1 = jnp.linalg.det(mat1) / det
    b2 = jnp.linalg.det(mat2) / det
    b3 = jnp.linalg.det(mat3) / det
    b0 = 1.0 - b1 - b2 - b3

    bary_coords = jnp.array([b0, b1, b2, b3])

    # Point is inside if all barycentric coordinates in [0, 1]
    tol = -1e-6
    is_inside = jnp.all(bary_coords >= tol) & jnp.all(bary_coords <= 1.0 + tol)

    return is_inside, bary_coords


@jax.jit
def interpolate_octree(query_point: jnp.ndarray,
                       mesh_points: jnp.ndarray,
                       connectivity: jnp.ndarray,
                       field_values: jnp.ndarray,
                       element_bounds: jnp.ndarray,
                       nodes_min: jnp.ndarray,
                       nodes_max: jnp.ndarray,
                       nodes_elements: jnp.ndarray,
                       nodes_elem_counts: jnp.ndarray,
                       nodes_children: jnp.ndarray,
                       nodes_is_leaf: jnp.ndarray) -> jnp.ndarray:
    """
    Interpolate using octree traversal.

    Traverses octree to find leaf containing query point, then checks
    only elements in that leaf.
    """

    # Traverse octree to find containing leaf
    def traverse_tree(node_idx):
        """Find leaf node containing query point."""

        # Check if we're in a leaf
        is_leaf = nodes_is_leaf[node_idx]

        def leaf_case():
            return node_idx

        def internal_case():
            # Determine which child to visit
            node_min = nodes_min[node_idx]
            node_max = nodes_max[node_idx]
            center = (node_min + node_max) / 2.0

            # Compute octant
            octant = 0
            octant = jnp.where(query_point[0] >= center[0], octant + 1, octant)
            octant = jnp.where(query_point[1] >= center[1], octant + 2, octant)
            octant = jnp.where(query_point[2] >= center[2], octant + 4, octant)

            # Get child index
            child_idx = nodes_children[node_idx, octant]

            # If child exists, recurse; otherwise stay at current node
            return jnp.where(child_idx >= 0, child_idx, node_idx)

        return jax.lax.cond(is_leaf, leaf_case, internal_case)

    # Traverse from root (node 0) to find leaf
    # Use lax.while_loop for iterative traversal
    def cond_fn(state):
        node_idx, _ = state
        return ~nodes_is_leaf[node_idx]

    def body_fn(state):
        node_idx, _ = state

        node_min = nodes_min[node_idx]
        node_max = nodes_max[node_idx]
        center = (node_min + node_max) / 2.0

        # Compute octant
        octant = 0
        octant = jnp.where(query_point[0] >= center[0], octant + 1, octant)
        octant = jnp.where(query_point[1] >= center[1], octant + 2, octant)
        octant = jnp.where(query_point[2] >= center[2], octant + 4, octant)

        child_idx = nodes_children[node_idx, octant]

        # If child exists, go to it; otherwise stop
        next_idx = jnp.where(child_idx >= 0, child_idx, node_idx)

        return next_idx, None

    leaf_idx, _ = jax.lax.while_loop(cond_fn, body_fn, (jnp.int32(0), None))

    # Get candidate elements in leaf
    candidate_elements = nodes_elements[leaf_idx]
    n_candidates = nodes_elem_counts[leaf_idx]

    # Check elements in leaf
    def check_element(elem_idx):
        """Check if point is in element."""
        is_valid = elem_idx >= 0

        # Quick bounds check
        elem_min = element_bounds[elem_idx, 0]
        elem_max = element_bounds[elem_idx, 1]
        in_bounds = jnp.all(query_point >= elem_min) & jnp.all(query_point <= elem_max)

        # Get element nodes
        node_indices = connectivity[elem_idx]
        tet_nodes = mesh_points[node_indices]

        # Check containment
        is_inside, bary_coords = point_in_tetrahedron(query_point, tet_nodes)

        # Interpolate
        node_values = field_values[node_indices]
        interpolated = jnp.dot(bary_coords, node_values)

        found = is_valid & in_bounds & is_inside
        return found, interpolated

    # Scan through candidates
    def scan_fn(carry, elem_idx):
        found_prev, value_prev = carry
        found_curr, value_curr = check_element(elem_idx)

        found = found_prev | found_curr
        value = jnp.where(found_prev, value_prev, value_curr)

        return (found, value), None

    init_carry = (jnp.array(False), jnp.zeros(3, dtype=jnp.float32))

    (found, interpolated_value), _ = jax.lax.scan(
        scan_fn,
        init_carry,
        candidate_elements  # Will scan all (including -1 padding)
    )

    # Fallback to nearest neighbor if not found
    def nearest_neighbor_fallback():
        distances = jnp.sum((mesh_points - query_point)**2, axis=1)
        nearest_idx = jnp.argmin(distances)
        return field_values[nearest_idx]

    result = jnp.where(found, interpolated_value, nearest_neighbor_fallback())

    return result


def create_octree_fem_interpolator(mesh: OctreeMesh):
    """
    Create JIT-compiled octree FEM interpolator.

    Parameters
    ----------
    mesh : OctreeMesh
        Pre-built octree mesh structure

    Returns
    -------
    callable
        JIT-compiled interpolator function
    """

    @jax.jit
    def octree_fem_interpolate(query_points: jnp.ndarray,
                               field_values: jnp.ndarray) -> jnp.ndarray:
        """
        Interpolate field values at query points using octree traversal.

        Parameters
        ----------
        query_points : jnp.ndarray
            Query positions (M, 3)
        field_values : jnp.ndarray
            Field values at mesh nodes (N, 3)

        Returns
        -------
        jnp.ndarray
            Interpolated values (M, 3)
        """

        interpolate_single = lambda qp: interpolate_octree(
            qp,
            mesh.points,
            mesh.connectivity,
            field_values,
            mesh.element_bounds,
            mesh.nodes_min,
            mesh.nodes_max,
            mesh.nodes_elements,
            mesh.nodes_elem_counts,
            mesh.nodes_children,
            mesh.nodes_is_leaf
        )

        return jax.vmap(interpolate_single)(query_points)

    return octree_fem_interpolate


# Test
if __name__ == "__main__":
    print("Testing octree FEM interpolator...")

    # Create simple mesh
    points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1]
    ], dtype=np.float32)

    connectivity = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7]
    ], dtype=np.int32)

    # Build octree
    mesh = build_octree_mesh(points, connectivity, max_elements_per_leaf=1, max_depth=5)

    # Create interpolator
    interpolator = create_octree_fem_interpolator(mesh)

    # Test
    field_values = jnp.array(points, dtype=jnp.float32)
    query_points = jnp.array([[0.5, 0.5, 0.5]], dtype=jnp.float32)

    result = interpolator(query_points, field_values)
    print(f"Query: {query_points[0]}")
    print(f"Result: {result[0]}")
    print(f"✅ Octree FEM interpolator test complete!")
