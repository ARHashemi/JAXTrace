"""
CPU-based octree search for particle element location.

This module provides fast, memory-efficient octree traversal using Numba JIT.
By doing the search on CPU, we avoid JAX compilation memory issues and can
then pass the found element IDs to JAX for GPU-accelerated interpolation.

Key Benefits:
- No JAX compilation overhead
- Minimal memory usage (~1 MB)
- Fast with Numba JIT (~1-5 ms for 500 particles)
- Can use actual Python loops (no XLA graph materialization)
"""

import numpy as np
from numba import njit, prange


@njit
def compute_barycentric_coords_cpu(point, vertices):
    """
    Compute barycentric coordinates for point in tetrahedron.

    Args:
        point: (3,) point coordinates
        vertices: (4, 3) tetrahedron vertices

    Returns:
        bary: (4,) barycentric coordinates
    """
    v0 = vertices[0]
    v1 = vertices[1]
    v2 = vertices[2]
    v3 = vertices[3]

    # Build matrix for barycentric system
    # Avoid column_stack - build manually for Numba compatibility
    mat = np.empty((3, 3), dtype=np.float32)
    mat[:, 0] = v1 - v0
    mat[:, 1] = v2 - v0
    mat[:, 2] = v3 - v0

    rhs = point - v0

    # Solve linear system
    try:
        bary123 = np.linalg.solve(mat, rhs)
        bary0 = 1.0 - (bary123[0] + bary123[1] + bary123[2])
        # Build result array manually to avoid tuple unpacking
        bary = np.empty(4, dtype=np.float32)
        bary[0] = bary0
        bary[1] = bary123[0]
        bary[2] = bary123[1]
        bary[3] = bary123[2]
        return bary
    except:
        # Singular matrix (degenerate element)
        bary = np.empty(4, dtype=np.float32)
        bary[0] = -1.0
        bary[1] = -1.0
        bary[2] = -1.0
        bary[3] = -1.0
        return bary


@njit
def is_point_in_tetrahedron_cpu(bary_coords, tolerance=1e-6):
    """
    Check if barycentric coordinates indicate point is inside tetrahedron.

    Args:
        bary_coords: (4,) barycentric coordinates
        tolerance: Tolerance for boundary (default: 1e-6)

    Returns:
        inside: bool, True if inside
    """
    return (bary_coords[0] >= -tolerance and
            bary_coords[1] >= -tolerance and
            bary_coords[2] >= -tolerance and
            bary_coords[3] >= -tolerance and
            bary_coords.sum() <= 1.0 + tolerance)


@njit
def find_octant(point, center):
    """
    Find which octant of a cube contains the point.

    Args:
        point: (3,) point coordinates
        center: (3,) octant center

    Returns:
        octant: int 0-7
    """
    octant = 0
    if point[0] > center[0]:
        octant += 1
    if point[1] > center[1]:
        octant += 2
    if point[2] > center[2]:
        octant += 4
    return octant


@njit
def traverse_octree_and_find_element(
    point,
    coarse_centers,
    coarse_children,
    coarse_elem_lists,
    coarse_elem_counts,
    fine_centers,
    fine_children,
    fine_elem_lists,
    fine_elem_counts,
    fine_parents,
    positions,
    connectivity,
    n_coarse_levels,
    max_depth
):
    """
    Traverse coarse+fine octree to find which element contains point.

    This is the CPU equivalent of interpolate_single_point from
    direct_octree_interpolator_jax.py, but WITHOUT the memory issues!

    Args:
        point: (3,) query point
        coarse_*: Coarse octree arrays
        fine_*: Fine octree arrays
        positions: (N, 3) mesh node positions
        connectivity: (M, 4) element connectivity
        n_coarse_levels: Number of coarse levels
        max_depth: Maximum octree depth

    Returns:
        element_id: int, index of containing element (-1 if not found)
    """
    # Step 1: Traverse coarse octree to find leaf
    coarse_node_idx = 0  # Start at root

    for level in range(n_coarse_levels):
        if coarse_node_idx < 0 or coarse_node_idx >= len(coarse_centers):
            break

        center = coarse_centers[coarse_node_idx]
        children = coarse_children[coarse_node_idx]

        # Check if leaf
        if children[0] == -1:
            break

        # Find child octant
        octant = find_octant(point, center)
        child_idx = children[octant]

        if child_idx == -1:
            break

        coarse_node_idx = child_idx

    # Step 2: Check elements in coarse leaf
    coarse_elements = coarse_elem_lists[coarse_node_idx]
    coarse_count = coarse_elem_counts[coarse_node_idx]

    for i in range(min(coarse_count, len(coarse_elements))):
        elem_idx = coarse_elements[i]

        # Validate element index
        if elem_idx < 0 or elem_idx >= len(connectivity):
            continue

        # Get element vertices
        node_indices = connectivity[elem_idx]
        vertices = positions[node_indices]

        # Check if inside
        bary_coords = compute_barycentric_coords_cpu(point, vertices)
        if is_point_in_tetrahedron_cpu(bary_coords):
            return elem_idx  # Found!

    # Step 3: If not found in coarse, check fine octree
    # Find fine root node (has parent = coarse_node_idx)
    fine_root_idx = -1
    for i in range(len(fine_parents)):
        if fine_parents[i] == coarse_node_idx:
            fine_root_idx = i
            break

    if fine_root_idx == -1:
        return -1  # No fine octree for this coarse leaf

    # Traverse fine octree
    fine_node_idx = fine_root_idx
    for level in range(n_coarse_levels, max_depth):
        if fine_node_idx < 0 or fine_node_idx >= len(fine_centers):
            break

        center = fine_centers[fine_node_idx]
        children = fine_children[fine_node_idx]

        # Check if leaf
        if children[0] == -1:
            break

        # Find child octant
        octant = find_octant(point, center)
        child_idx = children[octant]

        if child_idx == -1:
            break

        fine_node_idx = child_idx

    # Check elements in fine leaf
    fine_elements = fine_elem_lists[fine_node_idx]
    fine_count = fine_elem_counts[fine_node_idx]

    for i in range(min(fine_count, len(fine_elements))):
        elem_idx = fine_elements[i]

        # Validate element index
        if elem_idx < 0 or elem_idx >= len(connectivity):
            continue

        # Get element vertices
        node_indices = connectivity[elem_idx]
        vertices = positions[node_indices]

        # Check if inside
        bary_coords = compute_barycentric_coords_cpu(point, vertices)
        if is_point_in_tetrahedron_cpu(bary_coords):
            return elem_idx  # Found!

    return -1  # Not found


@njit(parallel=True)
def find_elements_for_particles(
    particles,
    coarse_centers,
    coarse_children,
    coarse_elem_lists,
    coarse_elem_counts,
    fine_centers,
    fine_children,
    fine_elem_lists,
    fine_elem_counts,
    fine_parents,
    positions,
    connectivity,
    n_coarse_levels,
    max_depth
):
    """
    Find containing elements for all particles (parallelized with Numba).

    Args:
        particles: (N, 3) particle positions
        ... (same octree/mesh arrays as single-particle version)

    Returns:
        element_ids: (N,) array of element indices (-1 if not found)
    """
    n_particles = len(particles)
    results = np.empty(n_particles, dtype=np.int32)

    for i in prange(n_particles):
        results[i] = traverse_octree_and_find_element(
            particles[i],
            coarse_centers,
            coarse_children,
            coarse_elem_lists,
            coarse_elem_counts,
            fine_centers,
            fine_children,
            fine_elem_lists,
            fine_elem_counts,
            fine_parents,
            positions,
            connectivity,
            n_coarse_levels,
            max_depth
        )

    return results


def find_elements_for_particles_interface(particles, shared_octree, positions, connectivity, timestep_idx):
    """
    High-level interface for finding elements (extracts arrays from shared_octree).

    Args:
        particles: (N, 3) particle positions
        shared_octree: SharedCoarseOctree instance
        positions: (M, 3) mesh node positions
        connectivity: (K, 4) element connectivity
        timestep_idx: Timestep index in revolution cycle

    Returns:
        element_ids: (N,) array of element indices
    """
    # Extract coarse octree data
    coarse = shared_octree.coarse_levels
    coarse_centers = np.asarray(coarse.node_centers, dtype=np.float32)  # Uses property
    coarse_children = np.asarray(coarse.node_children, dtype=np.int32)
    coarse_elem_lists = np.asarray(coarse.node_element_lists, dtype=np.int32)
    coarse_elem_counts = np.asarray(coarse.node_element_counts, dtype=np.int32)

    # Extract fine octree data for this timestep
    # Phase 2: Fine octree needs domain bounds to decode Morton codes
    fine = shared_octree.get_fine_level_for_timestep(timestep_idx)
    domain_min = np.asarray(coarse.bbox_min, dtype=np.float32)
    domain_max = np.asarray(coarse.bbox_max, dtype=np.float32)
    fine_centers = np.asarray(fine.decode_node_centers(domain_min, domain_max), dtype=np.float32)
    fine_children = np.asarray(fine.node_children, dtype=np.int32)
    fine_elem_lists = np.asarray(fine.node_element_lists, dtype=np.int32)
    fine_elem_counts = np.asarray(fine.node_element_counts, dtype=np.int32)
    fine_parents = np.asarray(fine.node_parents, dtype=np.int32)

    # Ensure arrays are contiguous for Numba
    particles = np.ascontiguousarray(particles, dtype=np.float32)
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    connectivity = np.ascontiguousarray(connectivity, dtype=np.int32)

    # Call Numba-accelerated search
    element_ids = find_elements_for_particles(
        particles,
        coarse_centers,
        coarse_children,
        coarse_elem_lists,
        coarse_elem_counts,
        fine_centers,
        fine_children,
        fine_elem_lists,
        fine_elem_counts,
        fine_parents,
        positions,
        connectivity,
        shared_octree.n_coarse_levels,
        shared_octree.max_octree_depth
    )

    return element_ids
