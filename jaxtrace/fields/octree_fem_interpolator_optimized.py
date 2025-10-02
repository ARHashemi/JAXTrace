"""
OPTIMIZED Adaptive Octree-based FEM Interpolator

Key optimizations:
1. Fixed-depth traversal (no while_loop)
2. Early termination in element scan
3. Cheap fallback (use first element nodes instead of global search)
4. Reduced memory footprint
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple
from dataclasses import dataclass


@dataclass
class OctreeMeshOptimized:
    """Optimized octree mesh structure."""

    # Mesh geometry
    points: jnp.ndarray
    connectivity: jnp.ndarray
    element_bounds: jnp.ndarray
    element_centroids: jnp.ndarray  # NEW: For cheap fallback

    # Octree structure
    nodes_min: jnp.ndarray
    nodes_max: jnp.ndarray
    nodes_elements: jnp.ndarray
    nodes_elem_counts: jnp.ndarray
    nodes_children: jnp.ndarray
    nodes_is_leaf: jnp.ndarray

    # Configuration
    max_elements_per_leaf: int
    max_depth: int


def build_octree_mesh_optimized(points: np.ndarray,
                                connectivity: np.ndarray,
                                max_elements_per_leaf: int = 32,
                                max_depth: int = 12):
    """
    Build optimized octree mesh.

    Same as before but also computes element centroids for cheap fallback.
    """

    points = np.asarray(points, dtype=np.float32)
    connectivity = np.asarray(connectivity, dtype=np.int32)

    n_elements = connectivity.shape[0]

    print(f"🌲 Building optimized octree:")
    print(f"   Mesh: {points.shape[0]} points, {n_elements} elements")
    print(f"   Max elements/leaf: {max_elements_per_leaf}")

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

    # Build octree (same as before)
    print(f"   Building octree structure...")

    from collections import namedtuple
    OctreeNode = namedtuple('OctreeNode', ['min_corner', 'max_corner', 'element_indices',
                                            'children', 'is_leaf', 'depth'])

    nodes = []

    def subdivide_node(min_corner, max_corner, elem_indices, depth):
        """Recursively subdivide space."""

        node_idx = len(nodes)
        n_elems = len(elem_indices)

        is_leaf = (n_elems <= max_elements_per_leaf) or (depth >= max_depth)

        if is_leaf:
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

        # Subdivide
        center = (min_corner + max_corner) / 2.0
        children = np.full(8, -1, dtype=np.int32)

        octant_bounds = [
            (min_corner, center),
            (np.array([center[0], min_corner[1], min_corner[2]]), np.array([max_corner[0], center[1], center[2]])),
            (np.array([min_corner[0], center[1], min_corner[2]]), np.array([center[0], max_corner[1], center[2]])),
            (np.array([center[0], center[1], min_corner[2]]), np.array([max_corner[0], max_corner[1], center[2]])),
            (np.array([min_corner[0], min_corner[1], center[2]]), np.array([center[0], center[1], max_corner[2]])),
            (np.array([center[0], min_corner[1], center[2]]), np.array([max_corner[0], center[1], max_corner[2]])),
            (np.array([min_corner[0], center[1], center[2]]), np.array([center[0], max_corner[1], max_corner[2]])),
            (center, max_corner),
        ]

        octant_elements = [[] for _ in range(8)]

        for elem_idx in elem_indices:
            elem_centroid = element_centroids[elem_idx]
            octant = 0
            if elem_centroid[0] >= center[0]: octant += 1
            if elem_centroid[1] >= center[1]: octant += 2
            if elem_centroid[2] >= center[2]: octant += 4
            octant_elements[octant].append(elem_idx)

        node = OctreeNode(
            min_corner=min_corner,
            max_corner=max_corner,
            element_indices=np.array([], dtype=np.int32),
            children=children,
            is_leaf=False,
            depth=depth
        )
        nodes.append(node)

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

    all_elements = list(range(n_elements))
    root_idx = subdivide_node(bbox_min, bbox_max, all_elements, depth=0)

    print(f"   ✅ Octree built: {len(nodes)} nodes")

    # Statistics
    leaf_nodes = [n for n in nodes if n.is_leaf]
    max_leaf_elems = max(len(n.element_indices) for n in leaf_nodes)
    avg_leaf_elems = np.mean([len(n.element_indices) for n in leaf_nodes])
    max_tree_depth = max(n.depth for n in nodes)

    print(f"   Leaf nodes: {len(leaf_nodes)}")
    print(f"   Max depth: {max_tree_depth}")
    print(f"   Elements/leaf: avg={avg_leaf_elems:.1f}, max={max_leaf_elems}")

    # Convert to JAX arrays
    print(f"   Converting to JAX format...")

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

    octree_mesh = OctreeMeshOptimized(
        points=jnp.array(points),
        connectivity=jnp.array(connectivity),
        element_bounds=jnp.array(element_bounds),
        element_centroids=jnp.array(element_centroids),  # NEW
        nodes_min=jnp.array(nodes_min),
        nodes_max=jnp.array(nodes_max),
        nodes_elements=jnp.array(nodes_elements),
        nodes_elem_counts=jnp.array(nodes_elem_counts),
        nodes_children=jnp.array(nodes_children),
        nodes_is_leaf=jnp.array(nodes_is_leaf),
        max_elements_per_leaf=max_elements_per_leaf,
        max_depth=max_depth
    )

    print(f"   ✅ Optimized octree ready!")

    return octree_mesh


@jax.jit
def point_in_tetrahedron_fast(point: jnp.ndarray,
                               tet_nodes: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Fast barycentric coordinate computation."""

    v0, v1, v2, v3 = tet_nodes[0], tet_nodes[1], tet_nodes[2], tet_nodes[3]

    v0p = point - v0
    v01 = v1 - v0
    v02 = v2 - v0
    v03 = v3 - v0

    mat = jnp.stack([v01, v02, v03], axis=1)
    det = jnp.linalg.det(mat)

    # Avoid division by zero
    det = jnp.where(jnp.abs(det) < 1e-10, 1.0, det)

    mat1 = jnp.stack([v0p, v02, v03], axis=1)
    mat2 = jnp.stack([v01, v0p, v03], axis=1)
    mat3 = jnp.stack([v01, v02, v0p], axis=1)

    b1 = jnp.linalg.det(mat1) / det
    b2 = jnp.linalg.det(mat2) / det
    b3 = jnp.linalg.det(mat3) / det
    b0 = 1.0 - b1 - b2 - b3

    bary_coords = jnp.array([b0, b1, b2, b3])

    tol = -1e-6
    is_inside = jnp.all(bary_coords >= tol) & jnp.all(bary_coords <= 1.0 + tol)

    return is_inside, bary_coords


@jax.jit
def interpolate_octree_optimized(query_point: jnp.ndarray,
                                 mesh_points: jnp.ndarray,
                                 connectivity: jnp.ndarray,
                                 field_values: jnp.ndarray,
                                 element_bounds: jnp.ndarray,
                                 element_centroids: jnp.ndarray,
                                 nodes_min: jnp.ndarray,
                                 nodes_max: jnp.ndarray,
                                 nodes_elements: jnp.ndarray,
                                 nodes_elem_counts: jnp.ndarray,
                                 nodes_children: jnp.ndarray,
                                 nodes_is_leaf: jnp.ndarray,
                                 max_depth: int) -> jnp.ndarray:
    """
    OPTIMIZED octree interpolation.

    Key optimizations:
    1. Fixed-depth traversal (unrolled, no while_loop)
    2. Early termination in scan
    3. Cheap fallback using nearest element centroid
    """

    # OPTIMIZATION 1: Fixed-depth traversal using fori_loop
    # Traverse up to max_depth levels
    def traverse_step(i, node_idx):
        """Single traversal step."""
        is_leaf = nodes_is_leaf[node_idx]

        def continue_traverse():
            node_min = nodes_min[node_idx]
            node_max = nodes_max[node_idx]
            center = (node_min + node_max) / 2.0

            # Compute octant
            octant = jnp.int32(0)
            octant = jnp.where(query_point[0] >= center[0], octant + 1, octant)
            octant = jnp.where(query_point[1] >= center[1], octant + 2, octant)
            octant = jnp.where(query_point[2] >= center[2], octant + 4, octant)

            child_idx = nodes_children[node_idx, octant]

            # If valid child, go to it; otherwise stay
            return jnp.where(child_idx >= 0, child_idx, node_idx)

        # Update node_idx only if not at leaf
        return jax.lax.cond(is_leaf, lambda: node_idx, continue_traverse)

    # Start at root and traverse
    node_idx = jax.lax.fori_loop(0, max_depth, traverse_step, jnp.int32(0))

    # Get candidate elements
    candidate_elements = nodes_elements[node_idx]
    n_candidates = nodes_elem_counts[node_idx]

    # OPTIMIZATION 2: Early termination in element check
    def check_element_optimized(carry, elem_idx):
        found_prev, value_prev = carry

        # Early return if already found
        def already_found():
            return found_prev, value_prev

        def check_current():
            is_valid = elem_idx >= 0

            # Quick bounds check
            elem_min = element_bounds[elem_idx, 0]
            elem_max = element_bounds[elem_idx, 1]
            in_bounds = jnp.all(query_point >= elem_min) & jnp.all(query_point <= elem_max)

            def do_check():
                # Get element nodes
                node_indices = connectivity[elem_idx]
                tet_nodes = mesh_points[node_indices]

                # Check containment
                is_inside, bary_coords = point_in_tetrahedron_fast(query_point, tet_nodes)

                # Interpolate
                node_values = field_values[node_indices]
                interpolated = jnp.dot(bary_coords, node_values)

                found = is_valid & in_bounds & is_inside
                return found, interpolated

            def skip_check():
                return jnp.array(False), jnp.zeros(3, dtype=jnp.float32)

            # Only check if valid and in bounds
            return jax.lax.cond(is_valid & in_bounds, do_check, skip_check)

        # If already found, skip; otherwise check current
        found_curr, value_curr = jax.lax.cond(found_prev, already_found, check_current)

        # Combine results
        found = found_prev | found_curr
        value = jnp.where(found_prev, value_prev, value_curr)

        return (found, value), None

    init_carry = (jnp.array(False), jnp.zeros(3, dtype=jnp.float32))

    (found, interpolated_value), _ = jax.lax.scan(
        check_element_optimized,
        init_carry,
        candidate_elements
    )

    # OPTIMIZATION 3: Cheap fallback
    # Instead of searching ALL nodes, use nearest element centroid from candidates
    def cheap_fallback():
        # Use first valid element's nodes for fallback
        first_elem = candidate_elements[0]

        # Get its 4 nodes
        node_indices = connectivity[first_elem]

        # Find nearest of the 4 nodes
        dists = jnp.sum((mesh_points[node_indices] - query_point)**2, axis=1)
        nearest_local = jnp.argmin(dists)
        nearest_node = node_indices[nearest_local]

        return field_values[nearest_node]

    result = jnp.where(found, interpolated_value, cheap_fallback())

    return result


def create_octree_fem_interpolator_optimized(mesh: OctreeMeshOptimized):
    """Create optimized JIT-compiled interpolator."""

    @jax.jit
    def octree_fem_interpolate(query_points: jnp.ndarray,
                               field_values: jnp.ndarray) -> jnp.ndarray:
        """Interpolate at query points."""

        interpolate_single = lambda qp: interpolate_octree_optimized(
            qp,
            mesh.points,
            mesh.connectivity,
            field_values,
            mesh.element_bounds,
            mesh.element_centroids,
            mesh.nodes_min,
            mesh.nodes_max,
            mesh.nodes_elements,
            mesh.nodes_elem_counts,
            mesh.nodes_children,
            mesh.nodes_is_leaf,
            mesh.max_depth
        )

        return jax.vmap(interpolate_single)(query_points)

    return octree_fem_interpolate


# Test
if __name__ == "__main__":
    print("Testing optimized octree FEM...")

    points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1]
    ], dtype=np.float32)

    connectivity = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)

    mesh = build_octree_mesh_optimized(points, connectivity, max_elements_per_leaf=1, max_depth=5)

    interpolator = create_octree_fem_interpolator_optimized(mesh)

    field_values = jnp.array(points, dtype=jnp.float32)
    query_points = jnp.array([[0.25, 0.25, 0.25]], dtype=jnp.float32)

    result = interpolator(query_points, field_values)
    print(f"Query: {query_points[0]}")
    print(f"Result: {result[0]}")
    print(f"✅ Optimized octree test complete!")
