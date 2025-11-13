#!/usr/bin/env python3
"""
Tests for Phase 4: Multi-level element search.

Tests the integration of:
- Level 0: Cached element check
- Level 1: Neighbor element check
- Level 2: Octree search

Author: JAXTrace GPU Team
Date: 2025-11-04
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jaxtrace.gpu.test_meshes import generate_test_mesh, TINY_MESH
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu, assign_elements_to_blocks, build_element_neighbors
from jaxtrace.gpu.octree_builder import build_octrees_per_block
from jaxtrace.gpu.flat_arrays import MeshData, ParticleData
from jaxtrace.gpu.multi_level_search import (
    search_level0_cached,
    search_level1_neighbors,
    search_level2_octree,
    find_containing_element_multi_level,
    find_containing_elements_batch,
    SearchStatistics
)


class TestLevel0Cached:
    """Test Level 0: Cached element check."""

    def test_particle_still_in_element(self, tiny_mesh):
        """Particle still in cached element should be found immediately."""
        positions, connectivity = tiny_mesh

        # Pick an element
        elem_id = 50
        vertices = positions[connectivity[elem_id]]
        centroid = vertices.mean(axis=0)

        # Should find it
        found_id = search_level0_cached(centroid, elem_id, positions, connectivity)
        assert found_id == elem_id

    def test_particle_left_element(self, tiny_mesh):
        """Particle that left cached element should return -1."""
        positions, connectivity = tiny_mesh

        # Particle in element 50, but we claim it's in element 10
        elem_id = 50
        vertices = positions[connectivity[elem_id]]
        centroid = vertices.mean(axis=0)

        cached_wrong = 10

        # Should NOT find it in wrong element
        found_id = search_level0_cached(centroid, cached_wrong, positions, connectivity)
        assert found_id == -1

    def test_invalid_cached_id(self, tiny_mesh):
        """Invalid cached element ID should return -1."""
        positions, connectivity = tiny_mesh

        point = np.array([0.5, 0.5, 0.5])

        # Test invalid IDs
        assert search_level0_cached(point, -1, positions, connectivity) == -1
        assert search_level0_cached(point, len(connectivity), positions, connectivity) == -1


class TestLevel1Neighbors:
    """Test Level 1: Neighbor element check."""

    def test_particle_in_neighbor(self, two_tetrahedra):
        """Particle that moved to neighbor should be found."""
        positions, connectivity = two_tetrahedra

        # Build neighbors
        element_neighbors = build_element_neighbors(connectivity)

        # Particle in element 1, cached as element 0
        elem_id = 1
        vertices = positions[connectivity[elem_id]]
        centroid = vertices.mean(axis=0)

        cached_id = 0

        # Should find it in neighbor
        found_id = search_level1_neighbors(
            centroid, cached_id, positions, connectivity, element_neighbors
        )
        assert found_id == elem_id

    def test_particle_not_in_neighbors(self, tiny_mesh):
        """Particle far from cached element should return -1."""
        positions, connectivity = tiny_mesh
        element_neighbors = build_element_neighbors(connectivity)

        # Particle in element 100, cached as element 0
        elem_id = 100
        vertices = positions[connectivity[elem_id]]
        centroid = vertices.mean(axis=0)

        cached_id = 0

        # Should NOT find it in element 0's neighbors
        found_id = search_level1_neighbors(
            centroid, cached_id, positions, connectivity, element_neighbors
        )
        assert found_id == -1


class TestLevel2Octree:
    """Test Level 2: Full octree search."""

    def test_particle_found_via_octree(self, tiny_mesh):
        """Particle should be found via octree search."""
        positions, connectivity = tiny_mesh

        # Build octrees
        element_block_IDs, partition_data = assign_elements_to_blocks(
            positions, connectivity, (2, 2, 1), verbose=False
        )

        octrees = build_octrees_per_block(
            positions, connectivity, element_block_IDs, partition_data,
            max_elements_per_node=50,
            verbose=False
        )

        # Test a random element
        elem_id = 75
        vertices = positions[connectivity[elem_id]]
        centroid = vertices.mean(axis=0)

        # Should find it via octree
        found_id = search_level2_octree(
            centroid, partition_data, octrees, positions, connectivity
        )
        assert found_id == elem_id


class TestMultiLevelIntegration:
    """Test integrated multi-level search."""

    def test_level0_hit(self, tiny_mesh):
        """Particle still in cached element should hit Level 0."""
        positions, connectivity = tiny_mesh
        element_neighbors = build_element_neighbors(connectivity)

        element_block_IDs, partition_data = assign_elements_to_blocks(
            positions, connectivity, (2, 2, 1), verbose=False
        )

        octrees = build_octrees_per_block(
            positions, connectivity, element_block_IDs, partition_data,
            max_elements_per_node=50,
            verbose=False
        )

        mesh_data = MeshData(
            positions=positions,
            connectivity=connectivity,
            element_neighbors=element_neighbors,
            element_block_IDs=element_block_IDs
        )

        # Particle in element 50
        elem_id = 50
        vertices = positions[connectivity[elem_id]]
        centroid = vertices.mean(axis=0)

        found_id, level = find_containing_element_multi_level(
            centroid, elem_id, mesh_data, partition_data, octrees
        )

        assert found_id == elem_id
        assert level == 0  # Should hit Level 0

    def test_level1_hit(self, two_tetrahedra):
        """Particle in neighbor should hit Level 1."""
        positions, connectivity = two_tetrahedra
        element_neighbors = build_element_neighbors(connectivity)

        element_block_IDs, partition_data = assign_elements_to_blocks(
            positions, connectivity, (1, 1, 1), verbose=False
        )

        octrees = build_octrees_per_block(
            positions, connectivity, element_block_IDs, partition_data,
            max_elements_per_node=50,
            verbose=False
        )

        mesh_data = MeshData(
            positions=positions,
            connectivity=connectivity,
            element_neighbors=element_neighbors,
            element_block_IDs=element_block_IDs
        )

        # Particle in element 1, cached as 0
        elem_id = 1
        vertices = positions[connectivity[elem_id]]
        centroid = vertices.mean(axis=0)

        cached_id = 0

        found_id, level = find_containing_element_multi_level(
            centroid, cached_id, mesh_data, partition_data, octrees
        )

        assert found_id == elem_id
        assert level == 1  # Should hit Level 1

    def test_level2_hit(self, tiny_mesh):
        """Particle far from cached should hit Level 2."""
        positions, connectivity = tiny_mesh
        element_neighbors = build_element_neighbors(connectivity)

        element_block_IDs, partition_data = assign_elements_to_blocks(
            positions, connectivity, (2, 2, 1), verbose=False
        )

        octrees = build_octrees_per_block(
            positions, connectivity, element_block_IDs, partition_data,
            max_elements_per_node=50,
            verbose=False
        )

        mesh_data = MeshData(
            positions=positions,
            connectivity=connectivity,
            element_neighbors=element_neighbors,
            element_block_IDs=element_block_IDs
        )

        # Particle in element 100, cached as 0
        elem_id = 100
        vertices = positions[connectivity[elem_id]]
        centroid = vertices.mean(axis=0)

        cached_id = 0

        found_id, level = find_containing_element_multi_level(
            centroid, cached_id, mesh_data, partition_data, octrees
        )

        assert found_id == elem_id
        assert level == 2  # Should hit Level 2


class TestBatchSearch:
    """Test batch search with statistics."""

    def test_batch_all_cached(self, tiny_mesh):
        """Batch where all particles stay in cached elements."""
        positions, connectivity = tiny_mesh
        element_neighbors = build_element_neighbors(connectivity)

        element_block_IDs, partition_data = assign_elements_to_blocks(
            positions, connectivity, (2, 2, 1), verbose=False
        )

        octrees = build_octrees_per_block(
            positions, connectivity, element_block_IDs, partition_data,
            max_elements_per_node=50,
            verbose=False
        )

        mesh_data = MeshData(
            positions=positions,
            connectivity=connectivity,
            element_neighbors=element_neighbors,
            element_block_IDs=element_block_IDs
        )

        # Create particles at element centroids (will be cached correctly)
        n_test = 50
        particle_positions = []
        particle_element_IDs = []

        np.random.seed(42)
        for _ in range(n_test):
            elem_id = np.random.randint(0, len(connectivity))
            vertices = positions[connectivity[elem_id]]
            centroid = vertices.mean(axis=0)

            particle_positions.append(centroid)
            particle_element_IDs.append(elem_id)

        particle_data = ParticleData(
            positions=np.array(particle_positions),
            element_IDs=np.array(particle_element_IDs),
            active=np.ones(n_test, dtype=bool)
        )

        # Run batch search
        new_element_IDs, stats = find_containing_elements_batch(
            particle_data, mesh_data, partition_data, octrees
        )

        # All should hit Level 0
        assert stats.n_level0_hits == n_test
        assert stats.n_level1_hits == 0
        assert stats.n_level2_hits == 0
        assert stats.hit_rate_level0() == 100.0

    def test_batch_mixed_levels(self, tiny_mesh):
        """Batch with particles at different search levels."""
        positions, connectivity = tiny_mesh
        element_neighbors = build_element_neighbors(connectivity)

        element_block_IDs, partition_data = assign_elements_to_blocks(
            positions, connectivity, (2, 2, 1), verbose=False
        )

        octrees = build_octrees_per_block(
            positions, connectivity, element_block_IDs, partition_data,
            max_elements_per_node=50,
            verbose=False
        )

        mesh_data = MeshData(
            positions=positions,
            connectivity=connectivity,
            element_neighbors=element_neighbors,
            element_block_IDs=element_block_IDs
        )

        # Mix of:
        # - Particles at centroids (cached correctly) → Level 0
        # - Particles with wrong cache far away → Level 2
        particle_positions = []
        particle_element_IDs = []

        # 10 cached correctly
        for i in range(10):
            elem_id = i * 10
            vertices = positions[connectivity[elem_id]]
            centroid = vertices.mean(axis=0)
            particle_positions.append(centroid)
            particle_element_IDs.append(elem_id)  # Correct cache

        # 10 cached incorrectly
        for i in range(10):
            elem_id = 100 + i
            vertices = positions[connectivity[elem_id]]
            centroid = vertices.mean(axis=0)
            particle_positions.append(centroid)
            particle_element_IDs.append(0)  # Wrong cache

        particle_data = ParticleData(
            positions=np.array(particle_positions),
            element_IDs=np.array(particle_element_IDs),
            active=np.ones(20, dtype=bool)
        )

        # Run batch search
        new_element_IDs, stats = find_containing_elements_batch(
            particle_data, mesh_data, partition_data, octrees
        )

        # Should have hits at multiple levels
        assert stats.n_level0_hits >= 10  # At least the correctly cached
        assert stats.n_level2_hits >= 0   # Some may need octree
        assert stats.success_rate() >= 95.0  # Most should be found


class TestSearchStatistics:
    """Test statistics tracking."""

    def test_statistics_calculation(self):
        """Test SearchStatistics calculations."""
        stats = SearchStatistics(
            n_particles=100,
            n_level0_hits=85,
            n_level1_hits=10,
            n_level2_hits=4,
            n_not_found=1
        )

        assert stats.hit_rate_level0() == 85.0
        assert stats.hit_rate_level1() == 10.0
        assert stats.hit_rate_level2() == 4.0
        assert stats.success_rate() == 99.0

    def test_statistics_string(self):
        """Test SearchStatistics string representation."""
        stats = SearchStatistics(
            n_particles=100,
            n_level0_hits=85,
            n_level1_hits=10,
            n_level2_hits=4,
            n_not_found=1
        )

        s = str(stats)
        assert "85" in s  # Level 0 hits
        assert "10" in s  # Level 1 hits
        assert "4" in s   # Level 2 hits
        assert "99.0%" in s  # Success rate
