#!/usr/bin/env python3
"""
Element Search using Octrees

Implements efficient element search for particle initialization using the
octree structures built in Phase 2.

Phase 3.2 of V3 Plan

Search Strategy:
1. Find which block contains particle (spatial lookup)
2. Traverse octree within block to find leaf node
3. Test all elements in leaf node (typically 50-500 elements)
4. Return first containing element

This replaces brute-force search of all 870K elements per block.
"""

from typing import Tuple, Optional, Dict
import numpy as np
import jax.numpy as jnp
import jax

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


def point_in_tetrahedron(
    point: np.ndarray,
    vertices: np.ndarray
) -> bool:
    """
    Test if point is inside tetrahedron.

    Uses barycentric coordinates method.

    Args:
        point: (3,) position
        vertices: (4, 3) tetrahedron vertices

    Returns:
        inside: bool - True if point inside tetrahedron
    """
    # Compute barycentric coordinates
    # Point p = λ0*v0 + λ1*v1 + λ2*v2 + λ3*v3
    # where λ0 + λ1 + λ2 + λ3 = 1

    v0, v1, v2, v3 = vertices

    # Build matrix [v1-v0, v2-v0, v3-v0]
    mat = np.column_stack([v1 - v0, v2 - v0, v3 - v0])

    # Solve: mat @ [λ1, λ2, λ3] = point - v0
    try:
        lambdas_123 = np.linalg.solve(mat, point - v0)
    except np.linalg.LinAlgError:
        # Degenerate tetrahedron - use fallback
        # Check if point is very close to any vertex
        for v in vertices:
            if np.linalg.norm(point - v) < 1e-8:
                return True
        return False

    # Compute λ0
    lambda_0 = 1.0 - np.sum(lambdas_123)

    # Check if all lambdas in [0, 1] with relaxed tolerance for numerical precision
    # Using 1e-8 instead of 1e-10 to handle numerical errors better
    all_lambdas = np.concatenate([[lambda_0], lambdas_123])

    return np.all(all_lambdas >= -1e-8) and np.all(all_lambdas <= 1.0 + 1e-8)


def find_containing_block(
    position: np.ndarray,
    partition_data
) -> int:
    """
    Find which block contains a position.

    Args:
        position: (3,) particle position
        partition_data: BlockPartitionData from Phase 1 (can be dict or object)

    Returns:
        block_id: int - block ID, or -1 if outside mesh
    """
    # Handle both dict and object access patterns
    if isinstance(partition_data, dict):
        bbox_min = partition_data['bbox_min']
        bbox_max = partition_data['bbox_max']
        block_size = partition_data['block_size']
        grid_size = partition_data['grid_size']
    else:
        bbox_min = partition_data.bbox_min
        bbox_max = partition_data.bbox_max
        block_size = partition_data.block_size
        grid_size = partition_data.grid_size

    # Check if inside mesh bounding box
    if not (np.all(position >= bbox_min) and
            np.all(position <= bbox_max)):
        return -1

    # Compute block indices
    block_idx = np.floor(
        (position - bbox_min) / block_size
    ).astype(np.int32)

    # Clip to valid range
    block_idx = np.clip(block_idx, 0, np.array(grid_size) - 1)

    # Convert to flat block ID
    block_id = (
        block_idx[0] * grid_size[1] * grid_size[2] +
        block_idx[1] * grid_size[2] +
        block_idx[2]
    )

    return int(block_id)


def find_octree_leaf_node(
    position: np.ndarray,
    octree_data
) -> int:
    """
    Find octree leaf node containing a position.

    Since we use simple recursive subdivision without explicitly storing
    parent-child relationships, we find the leaf by checking all nodes
    at increasing depths.

    Simplified approach: Find the node at maximum depth that contains the position.

    Args:
        position: (3,) particle position
        octree_data: OctreeData from Phase 2 (or SimpleOctree for compatibility)

    Returns:
        node_id: int - leaf node ID, or -1 if not found
    """
    # Handle SimpleOctree (simplified structure with just element IDs)
    if not hasattr(octree_data, 'n_nodes'):
        # For SimpleOctree, return 0 as the single "node"
        # The caller will use sorted_element_IDs directly
        return 0

    # Find all nodes that contain this position
    contains_position = []

    for node_id in range(octree_data.n_nodes):
        bbox_min = octree_data.node_bbox_min[node_id]
        bbox_max = octree_data.node_bbox_max[node_id]

        if (np.all(position >= bbox_min - 1e-10) and
            np.all(position <= bbox_max + 1e-10)):
            contains_position.append(node_id)

    if not contains_position:
        return -1

    # Return the deepest node (highest depth value)
    depths = [octree_data.node_depths[nid] for nid in contains_position]
    deepest_idx = np.argmax(depths)

    return contains_position[deepest_idx]


def find_containing_element_in_node(
    position: np.ndarray,
    node_id: int,
    octree_data,
    positions: np.ndarray,
    connectivity: np.ndarray
) -> int:
    """
    Find containing element within an octree node.

    Tests all elements in the node (typically 50-500).

    Args:
        position: (3,) particle position
        node_id: Octree node ID
        octree_data: OctreeData or SimpleOctree
        positions: (N_nodes, 3) mesh node positions
        connectivity: (N_elements, 4) mesh connectivity

    Returns:
        element_id: int - containing element, or -1 if not found
    """
    # Handle SimpleOctree (no node_ranges, just sorted_element_IDs)
    if not hasattr(octree_data, 'node_ranges'):
        # For SimpleOctree, use all elements directly
        element_ids = octree_data.sorted_element_IDs
    else:
        # Get elements in this node for full OctreeData
        start, end = octree_data.node_ranges[node_id]
        element_ids = octree_data.sorted_element_IDs[start:end]

    # Test each element
    for elem_id in element_ids:
        # Get element vertices
        node_ids = connectivity[elem_id]
        vertices = positions[node_ids]

        # Test if point inside
        if point_in_tetrahedron(position, vertices):
            return int(elem_id)

    return -1


def find_containing_element(
    position: np.ndarray,
    partition_data,
    octrees: Dict,
    positions: np.ndarray,
    connectivity: np.ndarray
) -> int:
    """
    Find element containing a particle position.

    Complete search pipeline:
    1. Find block (try main block first, then neighbors if needed)
    2. Find octree leaf node
    3. Test elements in node

    Note: Elements can span block boundaries, so we may need to check
    neighboring blocks if not found in the primary block.

    Args:
        position: (3,) particle position
        partition_data: BlockPartitionData
        octrees: Dict[block_id -> OctreeData]
        positions: (N_nodes, 3) mesh positions
        connectivity: (N_elements, 4) mesh connectivity

    Returns:
        element_id: int - containing element, or -1 if not found
    """
    # Step 1: Find primary block
    block_id = find_containing_block(position, partition_data)

    if block_id < 0:
        return -1

    # Try primary block first
    if block_id in octrees:
        octree_data = octrees[block_id]

        # Step 2: Find octree leaf node
        node_id = find_octree_leaf_node(position, octree_data)

        if node_id >= 0:
            # Step 3: Find element in node
            element_id = find_containing_element_in_node(
                position, node_id, octree_data, positions, connectivity
            )

            if element_id >= 0:
                return element_id

    # Not found in primary block - try neighboring blocks
    # Elements can span block boundaries, especially near boundaries
    if isinstance(partition_data, dict):
        grid_size = np.array(partition_data['grid_size'])
        bbox_min = partition_data['bbox_min']
        block_size = partition_data['block_size']
    else:
        grid_size = np.array(partition_data.grid_size)
        bbox_min = partition_data.bbox_min
        block_size = partition_data.block_size

    block_idx = np.floor(
        (position - bbox_min) / block_size
    ).astype(np.int32)
    block_idx = np.clip(block_idx, 0, grid_size - 1)

    # Check all 26 neighboring blocks (3x3x3 minus center)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue  # Skip primary block (already checked)

                neighbor_idx = block_idx + np.array([dx, dy, dz])

                # Check bounds
                if np.any(neighbor_idx < 0) or np.any(neighbor_idx >= grid_size):
                    continue

                # Convert to block ID
                neighbor_block_id = (
                    neighbor_idx[0] * grid_size[1] * grid_size[2] +
                    neighbor_idx[1] * grid_size[2] +
                    neighbor_idx[2]
                )

                if neighbor_block_id not in octrees:
                    continue

                # Try this neighbor block
                octree_data = octrees[neighbor_block_id]
                node_id = find_octree_leaf_node(position, octree_data)

                if node_id >= 0:
                    element_id = find_containing_element_in_node(
                        position, node_id, octree_data, positions, connectivity
                    )

                    if element_id >= 0:
                        return element_id

    return -1


def find_containing_elements_batch(
    positions: np.ndarray,
    partition_data,
    octrees: Dict,
    mesh_positions: np.ndarray,
    connectivity: np.ndarray,
    verbose: bool = True
) -> np.ndarray:
    """
    Find containing elements for batch of particles.

    Args:
        positions: (N_particles, 3) particle positions
        partition_data: BlockPartitionData
        octrees: Dict[block_id -> OctreeData]
        mesh_positions: (N_nodes, 3) mesh positions
        connectivity: (N_elements, 4) mesh connectivity
        verbose: Print progress

    Returns:
        element_ids: (N_particles,) int32 - containing elements (-1 if not found)
    """
    n_particles = len(positions)
    element_ids = np.full(n_particles, -1, dtype=np.int32)

    if verbose:
        print(f"\nFinding containing elements for {n_particles:,} particles...")

    for i, position in enumerate(positions):
        if verbose and i > 0 and i % 10000 == 0:
            found_so_far = np.sum(element_ids[:i] >= 0)
            print(f"  Processed {i:,} / {n_particles:,} ({100*i/n_particles:.1f}%) - "
                  f"Found: {found_so_far:,} ({100*found_so_far/i:.1f}%)")

        element_ids[i] = find_containing_element(
            position, partition_data, octrees,
            mesh_positions, connectivity
        )

    n_found = np.sum(element_ids >= 0)
    success_rate = 100 * n_found / n_particles if n_particles > 0 else 0

    if verbose:
        print(f"  Found: {n_found:,} / {n_particles:,} ({success_rate:.1f}%)")

    return element_ids


if __name__ == "__main__":
    print("Testing element search...")

    # Test with synthetic mesh
    try:
        from .test_meshes import generate_test_mesh, SMALL_BALANCED_MESH
        from .mesh_loader import assign_elements_to_blocks
        from .octree_builder import build_octrees_per_block
        from .particle_seeding import seed_particles_uniform_grid, SeedingConfig
    except ImportError:
        from test_meshes import generate_test_mesh, SMALL_BALANCED_MESH
        from mesh_loader import assign_elements_to_blocks
        from octree_builder import build_octrees_per_block
        from particle_seeding import seed_particles_uniform_grid, SeedingConfig

    # Generate mesh
    print("\nGenerating test mesh...")
    positions, connectivity = generate_test_mesh(SMALL_BALANCED_MESH)

    # Assign to blocks
    element_block_IDs, partition_data = assign_elements_to_blocks(
        positions, connectivity, (2, 2, 2), verbose=False
    )

    # Build octrees
    print("Building octrees...")
    octrees = build_octrees_per_block(
        positions, connectivity, element_block_IDs, partition_data,
        max_elements_per_node=100,
        verbose=False
    )

    # Seed particles
    print("\nSeeding particles...")
    config = SeedingConfig(
        bbox_min=positions.min(axis=0),
        bbox_max=positions.max(axis=0),
        density_per_axis=(5, 5, 5),
    )
    particle_positions = seed_particles_uniform_grid(config)
    print(f"  Seeded {len(particle_positions):,} particles")

    # Find containing elements
    element_ids = find_containing_elements_batch(
        particle_positions, partition_data, octrees,
        positions, connectivity,
        verbose=True
    )

    # Summary
    n_found = np.sum(element_ids >= 0)
    print(f"\n✅ Element search test complete!")
    print(f"   Success rate: {100*n_found/len(particle_positions):.1f}%")
