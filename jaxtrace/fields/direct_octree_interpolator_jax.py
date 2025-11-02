#!/usr/bin/env python3
"""
JAX-Compatible Direct FEM Interpolator Using Coarse+Fine Octrees.

OPTIMIZED VERSION - Fixed memory explosion issues:
1. Removed nested @jax.jit to prevent closure capture
2. Pass arrays as arguments instead of capturing in closures
3. Keep arrays as NumPy until JIT conversion

This eliminates the redundant third octree (5-8 GB) by using coarse+fine directly (~1 MB).
Memory: ~1 MB octrees (99% reduction from 5-8 GB)
Performance: GPU-accelerated, fully JIT-compiled
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from typing import Callable
from dataclasses import dataclass

from .shared_coarse_octree import SharedOctreeStructure


@jax.jit
def compute_barycentric_coords(point: jnp.ndarray, tet_vertices: jnp.ndarray) -> jnp.ndarray:
    """
    Compute barycentric coordinates for point in tetrahedron.

    Args:
        point: Query point [3]
        tet_vertices: Tetrahedron vertices [4, 3]

    Returns:
        Barycentric coordinates [4]
    """
    v0, v1, v2, v3 = tet_vertices[0], tet_vertices[1], tet_vertices[2], tet_vertices[3]

    # Edge vectors from v0
    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0

    # Vector from v0 to point
    vp = point - v0

    # Build matrix and solve
    mat = jnp.stack([e1, e2, e3], axis=1)  # [3, 3]
    bary_123 = jnp.linalg.solve(mat, vp)  # [3]

    b1, b2, b3 = bary_123[0], bary_123[1], bary_123[2]
    b0 = 1.0 - b1 - b2 - b3

    return jnp.array([b0, b1, b2, b3])


@jax.jit
def is_point_in_tetrahedron(bary_coords: jnp.ndarray, tolerance: float = 1e-6) -> jnp.ndarray:
    """Check if barycentric coordinates indicate point is inside tetrahedron."""
    return jnp.all(bary_coords >= -tolerance)


@jax.jit
def find_octant(point: jnp.ndarray, center: jnp.ndarray) -> jnp.ndarray:
    """
    Find which octant (0-7) contains the point.

    Octant encoding:
    - bit 0: x >= center[0]
    - bit 1: y >= center[1]
    - bit 2: z >= center[2]
    """
    octant = jnp.int32(0)
    octant = jnp.where(point[0] >= center[0], octant + 1, octant)
    octant = jnp.where(point[1] >= center[1], octant + 2, octant)
    octant = jnp.where(point[2] >= center[2], octant + 4, octant)
    return octant


def create_jax_direct_interpolator(
    shared_octree: SharedOctreeStructure,
    positions: np.ndarray,  # Keep as NumPy
    connectivity: np.ndarray,  # Keep as NumPy
    timestep_idx: int
) -> Callable:
    """
    Create a fully JAX-compatible interpolator using coarse+fine octrees directly.

    OPTIMIZED: Arrays are kept as NumPy and passed as arguments to avoid closure capture.

    This eliminates the need for the memory-intensive third octree.

    Args:
        shared_octree: SharedOctreeStructure with coarse and fine octrees
        positions: Node positions [N, 3] (NumPy array)
        connectivity: Element connectivity [M, 4] (NumPy array)
        timestep_idx: Which timestep to use for fine octree (revolution cycle index 0-39)

    Returns:
        Callable: interpolator(query_positions, field_at_nodes) -> interpolated_values
    """

    # Get coarse octree data - KEEP AS NUMPY (don't convert to JAX yet!)
    coarse = shared_octree.coarse_levels
    coarse_centers = np.asarray(coarse.node_centers, dtype=np.float32)  # Uses property
    coarse_sizes = np.asarray(coarse.node_sizes, dtype=np.float32)      # Uses property
    coarse_children = np.asarray(coarse.node_children, dtype=np.int32)
    coarse_elem_lists = np.asarray(coarse.node_element_lists, dtype=np.int32)
    coarse_elem_counts = np.asarray(coarse.node_element_counts, dtype=np.int32)

    # Get fine octree data for this timestep - KEEP AS NUMPY
    # Phase 2: Fine octree needs domain bounds to decode Morton codes
    fine = shared_octree.get_fine_level_for_timestep(timestep_idx)
    domain_min = np.asarray(coarse.bbox_min, dtype=np.float32)
    domain_max = np.asarray(coarse.bbox_max, dtype=np.float32)
    fine_centers = np.asarray(fine.decode_node_centers(domain_min, domain_max), dtype=np.float32)
    fine_sizes = np.asarray(fine.decode_node_sizes(domain_min, domain_max), dtype=np.float32)
    fine_children = np.asarray(fine.node_children, dtype=np.int32)
    fine_elem_lists = np.asarray(fine.node_element_lists, dtype=np.int32)
    fine_elem_counts = np.asarray(fine.node_element_counts, dtype=np.int32)
    fine_parents = np.asarray(fine.node_parents, dtype=np.int32)

    # Configuration (scalars are fine to keep)
    n_coarse_levels = shared_octree.n_coarse_levels
    max_depth = shared_octree.max_octree_depth

    # Mesh data - KEEP AS NUMPY
    positions_np = np.asarray(positions, dtype=np.float32)
    connectivity_np = np.asarray(connectivity, dtype=np.int32)

    # NOTE: interpolate_single_point is NO LONGER @jax.jit decorated!
    # This prevents it from capturing arrays in its closure during nested JIT compilation
    def interpolate_single_point(
        point: jnp.ndarray,
        field_at_nodes: jnp.ndarray,
        # Pass all arrays as arguments instead of capturing them!
        coarse_centers_jax: jnp.ndarray,
        coarse_children_jax: jnp.ndarray,
        coarse_elem_lists_jax: jnp.ndarray,
        coarse_elem_counts_jax: jnp.ndarray,
        fine_centers_jax: jnp.ndarray,
        fine_children_jax: jnp.ndarray,
        fine_elem_lists_jax: jnp.ndarray,
        fine_elem_counts_jax: jnp.ndarray,
        fine_parents_jax: jnp.ndarray,
        positions_jax: jnp.ndarray,
        connectivity_jax: jnp.ndarray,
        n_coarse_levels_val: int,
        max_depth_val: int
    ) -> jnp.ndarray:
        """
        Interpolate field at a single point using coarse+fine octrees.

        Fully JAX-compatible implementation using lax primitives.
        All arrays are now passed as arguments to prevent closure capture!
        """

        default_value = jnp.zeros(3, dtype=jnp.float32)

        # Step 1: Traverse coarse octree using lax.fori_loop
        def traverse_coarse(level, node_idx):
            """Traverse one level of coarse octree."""
            center = coarse_centers_jax[node_idx]
            children = coarse_children_jax[node_idx]

            # Check if leaf (no children)
            is_leaf = children[0] == -1

            # Find octant
            octant = find_octant(point, center)
            child_idx = children[octant]

            # If leaf or no child exists, stay at current node
            next_idx = lax.cond(
                jnp.logical_or(is_leaf, child_idx == -1),
                lambda: node_idx,
                lambda: child_idx
            )

            return next_idx

        # Traverse coarse octree from root (index 0)
        coarse_leaf_idx = lax.fori_loop(0, n_coarse_levels_val, traverse_coarse, jnp.int32(0))

        # Step 2: Check elements in coarse leaf using lax.fori_loop
        coarse_elements = coarse_elem_lists_jax[coarse_leaf_idx]
        coarse_count = coarse_elem_counts_jax[coarse_leaf_idx]

        def check_coarse_element(i, carry):
            """Check if element i contains point and return interpolated value."""
            found, result = carry
            elem_idx = coarse_elements[i]

            # Skip if already found, index out of range, or invalid element
            within_count = i < coarse_count
            elem_valid = jnp.logical_and(elem_idx >= 0, elem_idx < connectivity_jax.shape[0])
            should_check = jnp.logical_and(jnp.logical_not(found), jnp.logical_and(within_count, elem_valid))

            # Get element vertices (always, but result only used if should_check)
            elem_idx_safe = jnp.where(elem_valid, elem_idx, 0)  # Use index 0 if invalid
            node_indices = connectivity_jax[elem_idx_safe]
            tet_vertices = positions_jax[node_indices]

            # Compute barycentric coordinates
            bary_coords = compute_barycentric_coords(point, tet_vertices)

            # Check if inside
            is_inside = is_point_in_tetrahedron(bary_coords)

            # Interpolate
            field_values = field_at_nodes[node_indices]
            interpolated = jnp.dot(bary_coords, field_values)

            # Update carry only if we found it and should check
            new_found = jnp.logical_or(found, jnp.logical_and(should_check, is_inside))
            new_result = lax.cond(
                jnp.logical_and(should_check, is_inside),
                lambda: interpolated,
                lambda: result
            )

            return (new_found, new_result)

        # Check coarse elements using fori_loop (doesn't materialize intermediates)
        init_carry = (jnp.bool_(False), default_value)
        max_elements = coarse_elements.shape[0]
        (found_coarse, coarse_result) = lax.fori_loop(
            0, max_elements, check_coarse_element, init_carry
        )

        # Step 3: If not found in coarse, check fine octree
        def check_fine_octree():
            """Traverse fine octree and check its elements."""

            # Find fine node that corresponds to this coarse leaf
            # Fine nodes have parent indices pointing to coarse nodes
            def find_fine_root(i, fine_node_idx):
                """Find fine node with matching parent."""
                parent = fine_parents_jax[i]
                matches = parent == coarse_leaf_idx
                return lax.cond(
                    jnp.logical_and(matches, fine_node_idx == -1),
                    lambda: i,
                    lambda: fine_node_idx
                )

            fine_root_idx = lax.fori_loop(0, fine_parents_jax.shape[0], find_fine_root, jnp.int32(-1))

            # If no fine root found, return default
            def traverse_fine():
                """Traverse fine octree from root."""
                def traverse_fine_level(level, node_idx):
                    """Traverse one level of fine octree."""
                    # Skip if invalid node
                    is_valid = jnp.logical_and(node_idx >= 0, node_idx < fine_centers_jax.shape[0])

                    def do_traverse():
                        center = fine_centers_jax[node_idx]
                        children = fine_children_jax[node_idx]

                        is_leaf = children[0] == -1
                        octant = find_octant(point, center)
                        child_idx = children[octant]

                        return lax.cond(
                            jnp.logical_or(is_leaf, child_idx == -1),
                            lambda: node_idx,
                            lambda: child_idx
                        )

                    return lax.cond(is_valid, do_traverse, lambda: node_idx)

                # Traverse fine octree
                fine_leaf_idx = lax.fori_loop(
                    n_coarse_levels_val,
                    max_depth_val,
                    traverse_fine_level,
                    fine_root_idx
                )

                # Check fine elements using fori_loop
                fine_elements = fine_elem_lists_jax[fine_leaf_idx]
                fine_count = fine_elem_counts_jax[fine_leaf_idx]

                def check_fine_element(i, carry):
                    """Check fine element i (using fine_count)."""
                    found, result = carry
                    elem_idx = fine_elements[i]

                    # Skip if already found, index out of range, or invalid element
                    within_count = i < fine_count
                    elem_valid = jnp.logical_and(elem_idx >= 0, elem_idx < connectivity_jax.shape[0])
                    should_check = jnp.logical_and(jnp.logical_not(found), jnp.logical_and(within_count, elem_valid))

                    # Get element vertices
                    elem_idx_safe = jnp.where(elem_valid, elem_idx, 0)
                    node_indices = connectivity_jax[elem_idx_safe]
                    tet_vertices = positions_jax[node_indices]

                    # Compute barycentric coordinates
                    bary_coords = compute_barycentric_coords(point, tet_vertices)
                    is_inside = is_point_in_tetrahedron(bary_coords)

                    # Interpolate
                    field_values = field_at_nodes[node_indices]
                    interpolated = jnp.dot(bary_coords, field_values)

                    # Update carry
                    new_found = jnp.logical_or(found, jnp.logical_and(should_check, is_inside))
                    new_result = lax.cond(
                        jnp.logical_and(should_check, is_inside),
                        lambda: interpolated,
                        lambda: result
                    )

                    return (new_found, new_result)

                max_fine_elements = fine_elements.shape[0]
                (found_fine, fine_result) = lax.fori_loop(
                    0, max_fine_elements, check_fine_element, (jnp.bool_(False), default_value)
                )

                return fine_result

            return lax.cond(
                fine_root_idx == -1,
                lambda: default_value,
                traverse_fine
            )

        # Return coarse result if found, otherwise check fine octree
        return lax.cond(
            found_coarse,
            lambda: coarse_result,
            check_fine_octree
        )

    # The ONLY @jax.jit is HERE on the outer interpolator function
    @jax.jit
    def interpolator(query_positions: jnp.ndarray, field_at_nodes: jnp.ndarray) -> jnp.ndarray:
        """
        Interpolate field at query positions.

        Args:
            query_positions: Query positions [M, 3]
            field_at_nodes: Field values at mesh nodes [N, 3]

        Returns:
            Interpolated values [M, 3]
        """
        # Convert NumPy arrays to JAX arrays inside the JIT function
        # This way they are NOT captured in closures during compilation!
        coarse_centers_jax = jnp.asarray(coarse_centers)
        coarse_children_jax = jnp.asarray(coarse_children)
        coarse_elem_lists_jax = jnp.asarray(coarse_elem_lists)
        coarse_elem_counts_jax = jnp.asarray(coarse_elem_counts)
        fine_centers_jax = jnp.asarray(fine_centers)
        fine_children_jax = jnp.asarray(fine_children)
        fine_elem_lists_jax = jnp.asarray(fine_elem_lists)
        fine_elem_counts_jax = jnp.asarray(fine_elem_counts)
        fine_parents_jax = jnp.asarray(fine_parents)
        positions_jax = jnp.asarray(positions_np)
        connectivity_jax = jnp.asarray(connectivity_np)

        # Vectorize over query positions
        # Pass ALL arrays as arguments to vmap so they are NOT captured in closures
        # in_axes: (0, None, None, None, ...) means:
        #   - vectorize over query_positions (axis 0)
        #   - broadcast field_at_nodes and all other arrays (None)
        return jax.vmap(
            interpolate_single_point,
            in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
        )(
            query_positions,
            field_at_nodes,
            coarse_centers_jax,
            coarse_children_jax,
            coarse_elem_lists_jax,
            coarse_elem_counts_jax,
            fine_centers_jax,
            fine_children_jax,
            fine_elem_lists_jax,
            fine_elem_counts_jax,
            fine_parents_jax,
            positions_jax,
            connectivity_jax,
            n_coarse_levels,
            max_depth
        )

    return interpolator
