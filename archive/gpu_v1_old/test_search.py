"""
Unit tests for three-tier element search.

Tests element location using cached element, neighbors, and block search.
"""

import pytest
import numpy as np
from jaxtrace.gpu.search import (
    SearchStatistics,
    point_in_element,
    search_level0_cached_element,
    search_level1_neighbors,
    search_level2_block_brute_force,
    find_containing_element,
    update_particle_element_ids,
)
from jaxtrace.gpu.particles import ParticleData


@pytest.fixture
def simple_mesh():
    """
    Create a simple 2-tetrahedron mesh for testing.

    Elements:
        0: [0, 1, 2, 3] - unit tetrahedron
        1: [1, 2, 3, 4] - adjacent tetrahedron
    """
    positions = np.array([
        [0.0, 0.0, 0.0],  # node 0
        [1.0, 0.0, 0.0],  # node 1
        [0.0, 1.0, 0.0],  # node 2
        [0.0, 0.0, 1.0],  # node 3
        [1.0, 1.0, 1.0],  # node 4
    ], dtype=np.float32)

    connectivity = np.array([
        [0, 1, 2, 3],  # element 0
        [1, 2, 3, 4],  # element 1
    ], dtype=np.int32)

    # Element neighbors (0 and 1 are neighbors)
    neighbors = np.array([
        [1, -1, -1, -1],  # element 0 neighbors
        [0, -1, -1, -1],  # element 1 neighbors
    ], dtype=np.int32)

    # Both elements in block 0
    element_to_block = np.array([0, 0], dtype=np.int32)

    return positions, connectivity, neighbors, element_to_block


class TestSearchStatistics:
    """Test search statistics tracking."""

    def test_initialization(self):
        """Test statistics initialization."""
        stats = SearchStatistics()
        assert stats.level0_hits == 0
        assert stats.level1_hits == 0
        assert stats.level2_hits == 0
        assert stats.failures == 0
        assert stats.total_searches == 0

    def test_counting(self):
        """Test search counting."""
        stats = SearchStatistics()
        stats.level0_hits = 85
        stats.level1_hits = 10
        stats.level2_hits = 3
        stats.failures = 2

        assert stats.total_searches == 100

    def test_reset(self):
        """Test statistics reset."""
        stats = SearchStatistics()
        stats.level0_hits = 10
        stats.level1_hits = 5

        stats.reset()

        assert stats.level0_hits == 0
        assert stats.level1_hits == 0
        assert stats.total_searches == 0

    def test_print_statistics(self, capsys):
        """Test statistics printing."""
        stats = SearchStatistics()
        stats.level0_hits = 85
        stats.level1_hits = 10
        stats.level2_hits = 3
        stats.failures = 2

        stats.print_statistics()

        captured = capsys.readouterr()
        assert "Total searches: 100" in captured.out
        assert "Level 0" in captured.out
        assert "Level 1" in captured.out
        assert "Level 2" in captured.out


class TestPointInElement:
    """Test point-in-element queries."""

    def test_point_inside_unit_tetrahedron(self):
        """Test point inside unit tetrahedron."""
        element_nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        # Centroid is inside
        centroid = np.mean(element_nodes, axis=0)
        assert point_in_element(centroid, element_nodes)

        # Point near centroid
        point = np.array([0.2, 0.2, 0.2])
        assert point_in_element(point, element_nodes)

    def test_point_outside_element(self):
        """Test point outside element."""
        element_nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        # Point far outside
        point = np.array([10.0, 10.0, 10.0])
        assert not point_in_element(point, element_nodes)

        # Point just outside
        point = np.array([1.1, 0.0, 0.0])
        assert not point_in_element(point, element_nodes)

    def test_point_on_boundary(self):
        """Test point on element boundary."""
        element_nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        # Point on face (should be inside due to tolerance)
        point = np.array([0.5, 0.5, 0.0])
        assert point_in_element(point, element_nodes)

    def test_degenerate_element(self):
        """Test degenerate element (returns False)."""
        # Degenerate tetrahedron (all nodes coplanar)
        element_nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.0],
        ], dtype=np.float32)

        point = np.array([0.5, 0.25, 0.0])
        assert not point_in_element(point, element_nodes)


class TestLevel0Search:
    """Test Level 0: cached element search."""

    def test_level0_hit(self, simple_mesh):
        """Test Level 0 hit (point in cached element)."""
        positions, connectivity, _, _ = simple_mesh

        # Point in element 0
        point = np.array([0.2, 0.2, 0.2])
        cached_element_id = 0

        found, element_id = search_level0_cached_element(
            point, cached_element_id, positions, connectivity
        )

        assert found
        assert element_id == 0

    def test_level0_miss(self, simple_mesh):
        """Test Level 0 miss (point not in cached element)."""
        positions, connectivity, _, _ = simple_mesh

        # Point in element 1, but cached element is 0
        point = np.array([0.7, 0.7, 0.7])
        cached_element_id = 0

        found, element_id = search_level0_cached_element(
            point, cached_element_id, positions, connectivity
        )

        assert not found

    def test_level0_no_cache(self, simple_mesh):
        """Test Level 0 with no cached element."""
        positions, connectivity, _, _ = simple_mesh

        point = np.array([0.2, 0.2, 0.2])
        cached_element_id = -1  # No cache

        found, element_id = search_level0_cached_element(
            point, cached_element_id, positions, connectivity
        )

        assert not found


class TestLevel1Search:
    """Test Level 1: neighbor search."""

    def test_level1_hit(self, simple_mesh):
        """Test Level 1 hit (point in neighbor element)."""
        positions, connectivity, neighbors, _ = simple_mesh

        # Point in element 1, cached element is 0 (they are neighbors)
        point = np.array([0.7, 0.7, 0.7])
        cached_element_id = 0

        found, element_id = search_level1_neighbors(
            point, cached_element_id, neighbors, positions, connectivity
        )

        assert found
        assert element_id == 1

    def test_level1_miss(self, simple_mesh):
        """Test Level 1 miss (point not in neighbors)."""
        positions, connectivity, neighbors, _ = simple_mesh

        # Point outside all elements
        point = np.array([10.0, 10.0, 10.0])
        cached_element_id = 0

        found, element_id = search_level1_neighbors(
            point, cached_element_id, neighbors, positions, connectivity
        )

        assert not found

    def test_level1_no_cache(self, simple_mesh):
        """Test Level 1 with no cached element."""
        positions, connectivity, neighbors, _ = simple_mesh

        point = np.array([0.7, 0.7, 0.7])
        cached_element_id = -1  # No cache

        found, element_id = search_level1_neighbors(
            point, cached_element_id, neighbors, positions, connectivity
        )

        assert not found


class TestLevel2Search:
    """Test Level 2: block search."""

    def test_level2_hit(self, simple_mesh):
        """Test Level 2 hit (point found in block)."""
        positions, connectivity, _, element_to_block = simple_mesh

        # Point in element 0
        point = np.array([0.2, 0.2, 0.2])
        block_id = 0

        found, element_id = search_level2_block_brute_force(
            point, block_id, element_to_block, positions, connectivity
        )

        assert found
        assert element_id == 0

    def test_level2_miss(self, simple_mesh):
        """Test Level 2 miss (point not in block)."""
        positions, connectivity, _, element_to_block = simple_mesh

        # Point outside all elements
        point = np.array([10.0, 10.0, 10.0])
        block_id = 0

        found, element_id = search_level2_block_brute_force(
            point, block_id, element_to_block, positions, connectivity
        )

        assert not found

    def test_level2_invalid_block(self, simple_mesh):
        """Test Level 2 with invalid block ID."""
        positions, connectivity, _, element_to_block = simple_mesh

        point = np.array([0.2, 0.2, 0.2])
        block_id = -1  # Invalid

        found, element_id = search_level2_block_brute_force(
            point, block_id, element_to_block, positions, connectivity
        )

        assert not found


class TestThreeTierSearch:
    """Test complete three-tier search."""

    def test_find_element_level0_hit(self, simple_mesh):
        """Test three-tier search with Level 0 hit."""
        positions, connectivity, neighbors, element_to_block = simple_mesh
        stats = SearchStatistics()

        # Point in cached element 0
        point = np.array([0.2, 0.2, 0.2])
        cached_element_id = 0
        block_id = 0

        element_id = find_containing_element(
            point, cached_element_id, block_id,
            neighbors, element_to_block, positions, connectivity, stats
        )

        assert element_id == 0
        assert stats.level0_hits == 1
        assert stats.level1_hits == 0
        assert stats.level2_hits == 0

    def test_find_element_level1_hit(self, simple_mesh):
        """Test three-tier search with Level 1 hit."""
        positions, connectivity, neighbors, element_to_block = simple_mesh
        stats = SearchStatistics()

        # Point in element 1, cached element is 0 (neighbor)
        point = np.array([0.7, 0.7, 0.7])
        cached_element_id = 0
        block_id = 0

        element_id = find_containing_element(
            point, cached_element_id, block_id,
            neighbors, element_to_block, positions, connectivity, stats
        )

        assert element_id == 1
        assert stats.level0_hits == 0
        assert stats.level1_hits == 1
        assert stats.level2_hits == 0

    def test_find_element_level2_hit(self, simple_mesh):
        """Test three-tier search with Level 2 hit."""
        positions, connectivity, neighbors, element_to_block = simple_mesh
        stats = SearchStatistics()

        # Point in element 0, but no cache
        point = np.array([0.2, 0.2, 0.2])
        cached_element_id = -1  # No cache
        block_id = 0

        element_id = find_containing_element(
            point, cached_element_id, block_id,
            neighbors, element_to_block, positions, connectivity, stats
        )

        assert element_id == 0
        assert stats.level0_hits == 0
        assert stats.level1_hits == 0
        assert stats.level2_hits == 1

    def test_find_element_failure(self, simple_mesh):
        """Test three-tier search failure (point outside domain)."""
        positions, connectivity, neighbors, element_to_block = simple_mesh
        stats = SearchStatistics()

        # Point outside all elements
        point = np.array([10.0, 10.0, 10.0])
        cached_element_id = 0
        block_id = 0

        element_id = find_containing_element(
            point, cached_element_id, block_id,
            neighbors, element_to_block, positions, connectivity, stats
        )

        assert element_id == -1
        assert stats.failures == 1


class TestUpdateParticleElements:
    """Test batch element ID updates."""

    def test_update_all_particles(self, simple_mesh):
        """Test updating element IDs for all particles."""
        positions_mesh, connectivity, neighbors, element_to_block = simple_mesh
        stats = SearchStatistics()

        # Create particles in both elements
        particle_positions = np.array([
            [0.2, 0.2, 0.2],  # In element 0
            [0.7, 0.7, 0.7],  # In element 1
        ])
        particles = ParticleData.from_positions(particle_positions)
        particles.block_ids[:] = 0

        particles_updated = update_particle_element_ids(
            particles, neighbors, element_to_block,
            positions_mesh, connectivity, stats
        )

        assert particles_updated.element_ids[0] == 0
        assert particles_updated.element_ids[1] == 1
        assert stats.total_searches == 2

    def test_update_with_cache_hits(self, simple_mesh):
        """Test updating with cached element hits."""
        positions_mesh, connectivity, neighbors, element_to_block = simple_mesh
        stats = SearchStatistics()

        # Create particles with correct cached elements
        particle_positions = np.array([
            [0.2, 0.2, 0.2],  # In element 0
            [0.7, 0.7, 0.7],  # In element 1
        ])
        particles = ParticleData.from_positions(particle_positions)
        particles.element_ids[:] = [0, 1]  # Correct cache
        particles.block_ids[:] = 0

        particles_updated = update_particle_element_ids(
            particles, neighbors, element_to_block,
            positions_mesh, connectivity, stats
        )

        # Both should be Level 0 hits
        assert stats.level0_hits == 2
        assert stats.level1_hits == 0

    def test_update_deactivates_outside_particles(self, simple_mesh):
        """Test that particles outside domain are deactivated."""
        positions_mesh, connectivity, neighbors, element_to_block = simple_mesh

        # Create particle outside domain
        particle_positions = np.array([[10.0, 10.0, 10.0]])
        particles = ParticleData.from_positions(particle_positions)
        particles.block_ids[:] = 0

        particles_updated = update_particle_element_ids(
            particles, neighbors, element_to_block,
            positions_mesh, connectivity
        )

        assert particles_updated.n_active == 0
        assert particles_updated.element_ids[0] == -1

    def test_update_skips_inactive_particles(self, simple_mesh):
        """Test that inactive particles are skipped."""
        positions_mesh, connectivity, neighbors, element_to_block = simple_mesh
        stats = SearchStatistics()

        # Create particles
        particle_positions = np.array([
            [0.2, 0.2, 0.2],  # Active, in element 0
            [0.7, 0.7, 0.7],  # Inactive
        ])
        particles = ParticleData.from_positions(particle_positions)
        particles.block_ids[:] = 0
        particles.active_mask[1] = False  # Deactivate second

        particles_updated = update_particle_element_ids(
            particles, neighbors, element_to_block,
            positions_mesh, connectivity, stats
        )

        assert stats.total_searches == 1  # Only first particle
        assert particles_updated.element_ids[0] == 0
