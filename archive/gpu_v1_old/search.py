"""
Three-Tier Element Search Strategy.

Implements hierarchical element location with three levels of increasing cost:
- Level 0: Check cached element (O(1), 85-95% hit rate)
- Level 1: Check neighbor elements (O(1), 3-10% hit rate)
- Level 2: Block-local octree search (O(log n), 1-5% hit rate)

This strategy minimizes expensive searches by exploiting spatial coherence.
"""

from typing import Tuple, Optional
import numpy as np
from .particles import ParticleData
from .forest.block_builder import BlockMetadata


class SearchStatistics:
    """
    Track element search performance statistics.

    Monitors hit rates for each search level to validate the three-tier strategy.
    """

    def __init__(self):
        """Initialize search counters."""
        self.level0_hits = 0  # Cached element
        self.level1_hits = 0  # Neighbor elements
        self.level2_hits = 0  # Block search
        self.failures = 0      # Outside domain

    @property
    def total_searches(self) -> int:
        """Total number of searches."""
        return self.level0_hits + self.level1_hits + self.level2_hits + self.failures

    def print_statistics(self):
        """Print search statistics."""
        total = self.total_searches
        if total == 0:
            print("⚠️  No searches performed yet")
            return

        print(f"\n📊 Element Search Statistics:")
        print(f"  Total searches: {total:,}")
        print(f"\n  Level 0 (cached element):")
        print(f"    Hits: {self.level0_hits:,} ({100 * self.level0_hits / total:.1f}%)")
        print(f"\n  Level 1 (neighbor elements):")
        print(f"    Hits: {self.level1_hits:,} ({100 * self.level1_hits / total:.1f}%)")
        print(f"\n  Level 2 (block search):")
        print(f"    Hits: {self.level2_hits:,} ({100 * self.level2_hits / total:.1f}%)")
        print(f"\n  Search failures (outside domain):")
        print(f"    Count: {self.failures:,} ({100 * self.failures / total:.1f}%)")

        # Expected hit rates
        print(f"\n  Expected hit rates (from literature):")
        print(f"    Level 0: 85-95% (actual: {100 * self.level0_hits / total:.1f}%)")
        print(f"    Level 1: 3-10% (actual: {100 * self.level1_hits / total:.1f}%)")
        print(f"    Level 2: 1-5% (actual: {100 * self.level2_hits / total:.1f}%)")

    def reset(self):
        """Reset all counters."""
        self.level0_hits = 0
        self.level1_hits = 0
        self.level2_hits = 0
        self.failures = 0


def point_in_element(
    point: np.ndarray,
    element_nodes: np.ndarray
) -> bool:
    """
    Test if point is inside tetrahedral element using barycentric coordinates.

    A point is inside a tetrahedron if all four barycentric coordinates are
    in [0, 1] and sum to 1.

    Args:
        point: 3D point [x, y, z]
        element_nodes: Element node positions [4, 3]

    Returns:
        True if point is inside element

    Note:
        This is a CPU implementation. GPU version will use same algorithm
        but vectorized across particles and elements.
    """
    # Compute barycentric coordinates
    # [p0, p1, p2, p3] = tetrahedron vertices
    # point = λ0*p0 + λ1*p1 + λ2*p2 + λ3*p3, λ0+λ1+λ2+λ3=1, λi≥0

    p0, p1, p2, p3 = element_nodes

    # Build matrix for barycentric coordinate system
    # [p1-p0, p2-p0, p3-p0] @ [λ1, λ2, λ3] = point - p0
    mat = np.column_stack([p1 - p0, p2 - p0, p3 - p0])
    rhs = point - p0

    try:
        # Solve for [λ1, λ2, λ3]
        lambdas = np.linalg.solve(mat, rhs)
        lambda1, lambda2, lambda3 = lambdas
        lambda0 = 1.0 - lambda1 - lambda2 - lambda3

        # Check if all barycentric coordinates are in [0, 1]
        tolerance = 1e-6  # Small tolerance for numerical errors
        return (
            lambda0 >= -tolerance and lambda0 <= 1.0 + tolerance and
            lambda1 >= -tolerance and lambda1 <= 1.0 + tolerance and
            lambda2 >= -tolerance and lambda2 <= 1.0 + tolerance and
            lambda3 >= -tolerance and lambda3 <= 1.0 + tolerance
        )
    except np.linalg.LinAlgError:
        # Singular matrix - degenerate element
        return False


def search_level0_cached_element(
    point: np.ndarray,
    cached_element_id: int,
    positions: np.ndarray,
    connectivity: np.ndarray
) -> Tuple[bool, int]:
    """
    Level 0: Check if point is still in cached element.

    This is the fastest search level (O(1)). Expected hit rate: 85-95%.

    Args:
        point: 3D point [x, y, z]
        cached_element_id: Previously cached element ID
        positions: Node positions [N_nodes, 3]
        connectivity: Element connectivity [N_elements, 4]

    Returns:
        (found, element_id): found=True if point in cached element
    """
    if cached_element_id < 0:
        return False, -1

    # Get cached element nodes
    element_node_ids = connectivity[cached_element_id]
    element_nodes = positions[element_node_ids]

    # Check if point is inside
    if point_in_element(point, element_nodes):
        return True, cached_element_id

    return False, -1


def search_level1_neighbors(
    point: np.ndarray,
    cached_element_id: int,
    element_neighbors: np.ndarray,
    positions: np.ndarray,
    connectivity: np.ndarray
) -> Tuple[bool, int]:
    """
    Level 1: Check neighbor elements.

    If particle moved to adjacent element, check its neighbors first.
    Expected hit rate: 3-10%.

    Args:
        point: 3D point [x, y, z]
        cached_element_id: Previously cached element ID
        element_neighbors: Neighbor array [N_elements, max_neighbors]
        positions: Node positions [N_nodes, 3]
        connectivity: Element connectivity [N_elements, 4]

    Returns:
        (found, element_id): found=True if point in neighbor element
    """
    if cached_element_id < 0:
        return False, -1

    # Get neighbors of cached element
    neighbor_ids = element_neighbors[cached_element_id]

    # Check each neighbor
    for neighbor_id in neighbor_ids:
        if neighbor_id < 0:  # -1 indicates no neighbor (boundary)
            continue

        # Get neighbor element nodes
        element_node_ids = connectivity[neighbor_id]
        element_nodes = positions[element_node_ids]

        # Check if point is inside
        if point_in_element(point, element_nodes):
            return True, neighbor_id

    return False, -1


def search_level2_block_brute_force(
    point: np.ndarray,
    block_id: int,
    element_to_block: np.ndarray,
    positions: np.ndarray,
    connectivity: np.ndarray
) -> Tuple[bool, int]:
    """
    Level 2: Block-local brute-force search.

    Search all elements in the block. This is a temporary implementation
    until octree is integrated in Phase 2. Expected hit rate: 1-5%.

    Args:
        point: 3D point [x, y, z]
        block_id: Block containing point
        element_to_block: Element-to-block mapping [N_elements]
        positions: Node positions [N_nodes, 3]
        connectivity: Element connectivity [N_elements, 4]

    Returns:
        (found, element_id): found=True if element found in block

    Note:
        This is O(n) where n = elements in block (~110K for 32 blocks).
        Phase 2 will replace this with O(log n) octree search.
    """
    if block_id < 0:
        return False, -1

    # Get all elements in this block
    block_element_ids = np.where(element_to_block == block_id)[0]

    # Brute-force search through block elements
    for element_id in block_element_ids:
        # Get element nodes
        element_node_ids = connectivity[element_id]
        element_nodes = positions[element_node_ids]

        # Check if point is inside
        if point_in_element(point, element_nodes):
            return True, int(element_id)

    return False, -1


def find_containing_element(
    point: np.ndarray,
    cached_element_id: int,
    block_id: int,
    element_neighbors: np.ndarray,
    element_to_block: np.ndarray,
    positions: np.ndarray,
    connectivity: np.ndarray,
    stats: Optional[SearchStatistics] = None
) -> int:
    """
    Three-tier element search.

    Searches for element containing point using hierarchical strategy:
    1. Check cached element (O(1))
    2. Check neighbor elements (O(1))
    3. Block-local search (O(n) brute-force, O(log n) with octree in Phase 2)

    Args:
        point: 3D point [x, y, z]
        cached_element_id: Previously cached element ID (-1 if unknown)
        block_id: Block containing point (-1 if unknown/outside)
        element_neighbors: Neighbor array [N_elements, max_neighbors]
        element_to_block: Element-to-block mapping [N_elements]
        positions: Node positions [N_nodes, 3]
        connectivity: Element connectivity [N_elements, 4]
        stats: Optional statistics tracker

    Returns:
        element_id: ID of containing element (-1 if not found)

    Example:
        >>> stats = SearchStatistics()
        >>> elem_id = find_containing_element(
        ...     point, cached_id, block_id,
        ...     neighbors, elem_to_block, positions, connectivity,
        ...     stats
        ... )
        >>> elem_id
        12345
        >>> stats.level0_hits
        1
    """
    # Level 0: Check cached element
    found, element_id = search_level0_cached_element(
        point, cached_element_id, positions, connectivity
    )
    if found:
        if stats is not None:
            stats.level0_hits += 1
        return element_id

    # Level 1: Check neighbors
    found, element_id = search_level1_neighbors(
        point, cached_element_id, element_neighbors, positions, connectivity
    )
    if found:
        if stats is not None:
            stats.level1_hits += 1
        return element_id

    # Level 2: Block search
    found, element_id = search_level2_block_brute_force(
        point, block_id, element_to_block, positions, connectivity
    )
    if found:
        if stats is not None:
            stats.level2_hits += 1
        return element_id

    # Not found
    if stats is not None:
        stats.failures += 1
    return -1


def update_particle_element_ids(
    particles: ParticleData,
    element_neighbors: np.ndarray,
    element_to_block: np.ndarray,
    positions: np.ndarray,
    connectivity: np.ndarray,
    stats: Optional[SearchStatistics] = None
) -> ParticleData:
    """
    Update element IDs for all active particles using three-tier search.

    Args:
        particles: Particle data with cached element/block IDs
        element_neighbors: Neighbor array [N_elements, max_neighbors]
        element_to_block: Element-to-block mapping [N_elements]
        positions: Node positions [N_nodes, 3]
        connectivity: Element connectivity [N_elements, 4]
        stats: Optional statistics tracker

    Returns:
        Updated ParticleData with new element_ids

    Example:
        >>> stats = SearchStatistics()
        >>> particles_updated = update_particle_element_ids(
        ...     particles, neighbors, elem_to_block, positions, connectivity, stats
        ... )
        >>> stats.print_statistics()
    """
    # Create copy to avoid modifying input
    particles_updated = particles.copy()

    # Update element IDs for active particles
    for i in range(particles.n_particles):
        if not particles.active_mask[i]:
            continue

        point = particles.positions[i]
        cached_element_id = particles.element_ids[i]
        block_id = particles.block_ids[i]

        # Three-tier search
        element_id = find_containing_element(
            point, cached_element_id, block_id,
            element_neighbors, element_to_block,
            positions, connectivity, stats
        )

        # Update element ID
        particles_updated.element_ids[i] = element_id

        # Deactivate if element not found (particle left domain)
        if element_id < 0:
            particles_updated.active_mask[i] = False

    return particles_updated
