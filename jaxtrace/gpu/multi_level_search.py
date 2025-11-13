#!/usr/bin/env python3
"""
Phase 4: Multi-level element search for GPU particle tracking.

Combines three search levels with early termination:
- Level 0: Cached element (85-95% hit rate expected)
- Level 1: Neighbor elements (3-10% hit rate expected)
- Level 2: Octree search (1-5% hit rate expected)

This module provides the core search algorithm that will be JIT-compiled
for GPU execution.

Author: JAXTrace GPU Team
Date: 2025-11-04
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

# Try relative imports first, fall back to absolute
try:
    from .flat_arrays import MeshData, ParticleData
    from .element_search import (
        point_in_tetrahedron,
        find_containing_block,
        find_octree_leaf_node,
        find_containing_element_in_node
    )
except ImportError:
    from flat_arrays import MeshData, ParticleData
    from element_search import (
        point_in_tetrahedron,
        find_containing_block,
        find_octree_leaf_node,
        find_containing_element_in_node
    )


@dataclass
class SearchStatistics:
    """Statistics for multi-level search performance."""
    n_particles: int = 0
    n_level0_hits: int = 0  # Cached element
    n_level1_hits: int = 0  # Neighbor elements
    n_level2_hits: int = 0  # Octree search
    n_not_found: int = 0    # Outside mesh

    def hit_rate_level0(self) -> float:
        """Percentage found in cached element."""
        if self.n_particles == 0:
            return 0.0
        return 100.0 * self.n_level0_hits / self.n_particles

    def hit_rate_level1(self) -> float:
        """Percentage found in neighbor elements."""
        if self.n_particles == 0:
            return 0.0
        return 100.0 * self.n_level1_hits / self.n_particles

    def hit_rate_level2(self) -> float:
        """Percentage found via octree search."""
        if self.n_particles == 0:
            return 0.0
        return 100.0 * self.n_level2_hits / self.n_particles

    def success_rate(self) -> float:
        """Percentage found (any level)."""
        if self.n_particles == 0:
            return 0.0
        n_found = self.n_level0_hits + self.n_level1_hits + self.n_level2_hits
        return 100.0 * n_found / self.n_particles

    def __str__(self) -> str:
        """Human-readable statistics."""
        lines = [
            f"Multi-Level Search Statistics:",
            f"  Total particles: {self.n_particles:,}",
            f"",
            f"  Level 0 (cached): {self.n_level0_hits:,} ({self.hit_rate_level0():.1f}%)",
            f"  Level 1 (neighbors): {self.n_level1_hits:,} ({self.hit_rate_level1():.1f}%)",
            f"  Level 2 (octree): {self.n_level2_hits:,} ({self.hit_rate_level2():.1f}%)",
            f"  Not found: {self.n_not_found:,} ({100.0 * self.n_not_found / max(1, self.n_particles):.1f}%)",
            f"",
            f"  Success rate: {self.success_rate():.1f}%"
        ]
        return "\n".join(lines)


def search_level0_cached(
    position: np.ndarray,
    cached_element_id: int,
    positions: np.ndarray,
    connectivity: np.ndarray
) -> int:
    """
    Level 0: Check if particle is still in cached element.

    This is the fastest check - just test if the particle is still
    inside the element it was in at the last timestep.

    Args:
        position: (3,) particle position
        cached_element_id: Last known element ID (from previous timestep)
        positions: (N_nodes, 3) mesh node positions
        connectivity: (N_elements, 4) mesh connectivity

    Returns:
        element_id: cached_element_id if particle still inside, else -1
    """
    if cached_element_id < 0 or cached_element_id >= len(connectivity):
        return -1

    # Get element vertices
    vertices = positions[connectivity[cached_element_id]]

    # Check if particle still inside
    if point_in_tetrahedron(position, vertices):
        return cached_element_id
    else:
        return -1


def search_level1_neighbors(
    position: np.ndarray,
    cached_element_id: int,
    positions: np.ndarray,
    connectivity: np.ndarray,
    element_neighbors: np.ndarray
) -> int:
    """
    Level 1: Check face neighbors of cached element.

    If the particle left the cached element, it likely went to a
    face neighbor. Check all 4 face neighbors before falling back
    to expensive octree search.

    Args:
        position: (3,) particle position
        cached_element_id: Last known element ID
        positions: (N_nodes, 3) mesh node positions
        connectivity: (N_elements, 4) mesh connectivity
        element_neighbors: (N_elements, 4) neighbor element IDs

    Returns:
        element_id: Neighbor element ID if found, else -1
    """
    if cached_element_id < 0 or cached_element_id >= len(connectivity):
        return -1

    # Check all 4 face neighbors
    neighbors = element_neighbors[cached_element_id]

    for neighbor_id in neighbors:
        if neighbor_id < 0:  # -1 indicates boundary (no neighbor)
            continue

        # Get neighbor vertices
        vertices = positions[connectivity[neighbor_id]]

        # Check if particle is inside this neighbor
        if point_in_tetrahedron(position, vertices):
            return neighbor_id

    return -1


def search_level2_octree(
    position: np.ndarray,
    partition_data,
    octrees: Dict,
    positions: np.ndarray,
    connectivity: np.ndarray
) -> int:
    """
    Level 2: Full octree search with neighbor block fallback.

    Use the complete element search from Phase 3, which includes
    neighbor block checking for elements spanning boundaries.

    Args:
        position: (3,) particle position
        partition_data: BlockPartitionData
        octrees: Dict[block_id -> OctreeData]
        positions: (N_nodes, 3) mesh node positions
        connectivity: (N_elements, 4) mesh connectivity

    Returns:
        element_id: Element ID if found, else -1
    """
    # Find primary block
    block_id = find_containing_block(position, partition_data)

    if block_id < 0:
        return -1

    # Try primary block
    if block_id in octrees:
        octree_data = octrees[block_id]
        node_id = find_octree_leaf_node(position, octree_data)

        if node_id >= 0:
            element_id = find_containing_element_in_node(
                position, node_id, octree_data, positions, connectivity
            )
            if element_id >= 0:
                return element_id

    # Try neighbor blocks (for elements spanning boundaries)
    grid_size = np.array(partition_data.grid_size)
    block_idx = np.floor(
        (position - partition_data.bbox_min) / partition_data.block_size
    ).astype(np.int32)
    block_idx = np.clip(block_idx, 0, grid_size - 1)

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue

                neighbor_idx = block_idx + np.array([dx, dy, dz])

                if np.any(neighbor_idx < 0) or np.any(neighbor_idx >= grid_size):
                    continue

                neighbor_block_id = (
                    neighbor_idx[0] * grid_size[1] * grid_size[2] +
                    neighbor_idx[1] * grid_size[2] +
                    neighbor_idx[2]
                )

                if neighbor_block_id not in octrees:
                    continue

                octree_data = octrees[neighbor_block_id]
                node_id = find_octree_leaf_node(position, octree_data)

                if node_id >= 0:
                    element_id = find_containing_element_in_node(
                        position, node_id, octree_data, positions, connectivity
                    )
                    if element_id >= 0:
                        return element_id

    return -1


def find_containing_element_multi_level(
    position: np.ndarray,
    cached_element_id: int,
    mesh_data: MeshData,
    partition_data,
    octrees: Dict
) -> Tuple[int, int]:
    """
    Find containing element using multi-level search.

    Tries levels in order with early termination:
    1. Cached element (fastest)
    2. Neighbor elements (fast local search)
    3. Octree search (global search with blocks)

    Args:
        position: (3,) particle position
        cached_element_id: Last known element ID
        mesh_data: MeshData with positions, connectivity, neighbors
        partition_data: BlockPartitionData
        octrees: Dict[block_id -> OctreeData]

    Returns:
        (element_id, search_level):
            element_id: Found element ID or -1
            search_level: 0=cached, 1=neighbor, 2=octree, -1=not_found
    """
    # Level 0: Check cached element
    element_id = search_level0_cached(
        position, cached_element_id,
        mesh_data.positions, mesh_data.connectivity
    )
    if element_id >= 0:
        return element_id, 0

    # Level 1: Check neighbors
    element_id = search_level1_neighbors(
        position, cached_element_id,
        mesh_data.positions, mesh_data.connectivity,
        mesh_data.element_neighbors
    )
    if element_id >= 0:
        return element_id, 1

    # Level 2: Full octree search
    element_id = search_level2_octree(
        position, partition_data, octrees,
        mesh_data.positions, mesh_data.connectivity
    )
    if element_id >= 0:
        return element_id, 2

    # Not found
    return -1, -1


def find_containing_elements_batch(
    particle_data: ParticleData,
    mesh_data: MeshData,
    partition_data,
    octrees: Dict,
    verbose: bool = False
) -> Tuple[np.ndarray, SearchStatistics]:
    """
    Find containing elements for a batch of particles.

    Uses multi-level search with statistics tracking.

    Args:
        particle_data: ParticleData with positions, element_IDs, active mask
        mesh_data: MeshData with positions, connectivity, neighbors
        partition_data: BlockPartitionData
        octrees: Dict[block_id -> OctreeData]
        verbose: Print progress

    Returns:
        (new_element_IDs, stats):
            new_element_IDs: (N_particles,) updated element IDs
            stats: SearchStatistics with hit rates
    """
    n_particles = len(particle_data.positions)
    new_element_IDs = np.copy(particle_data.element_IDs)

    stats = SearchStatistics(n_particles=n_particles)

    for i in range(n_particles):
        if not particle_data.active[i]:
            continue

        position = particle_data.positions[i]
        cached_id = particle_data.element_IDs[i]

        element_id, level = find_containing_element_multi_level(
            position, cached_id, mesh_data, partition_data, octrees
        )

        new_element_IDs[i] = element_id

        if level == 0:
            stats.n_level0_hits += 1
        elif level == 1:
            stats.n_level1_hits += 1
        elif level == 2:
            stats.n_level2_hits += 1
        else:
            stats.n_not_found += 1

        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i+1:,}/{n_particles:,} particles...")

    return new_element_IDs, stats


# Module self-test
if __name__ == "__main__":
    print("Multi-level search module")
    print("Run tests with: pytest tests/gpu/test_multi_level_search.py")
