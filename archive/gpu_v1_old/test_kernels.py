"""
Unit tests for GPU kernels.

Tests JAX-based point-in-element and search kernels.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jaxtrace.gpu.kernels import (
    point_in_tetrahedron_jax,
    point_in_tetrahedron_safe,
    search_cached_element_jax,
    search_neighbors_jax,
    find_containing_element_gpu,
    find_containing_elements_batch,
    position_to_block_id_jax,
    positions_to_block_ids_batch,
)


@pytest.fixture
def simple_mesh_jax():
    """Create a simple 2-tetrahedron mesh for testing (JAX arrays)."""
    positions = jnp.array([
        [0.0, 0.0, 0.0],  # node 0
        [1.0, 0.0, 0.0],  # node 1
        [0.0, 1.0, 0.0],  # node 2
        [0.0, 0.0, 1.0],  # node 3
        [1.0, 1.0, 1.0],  # node 4
    ], dtype=jnp.float32)

    connectivity = jnp.array([
        [0, 1, 2, 3],  # element 0
        [1, 2, 3, 4],  # element 1
    ], dtype=jnp.int32)

    neighbors = jnp.array([
        [1, -1, -1, -1],  # element 0 neighbors
        [0, -1, -1, -1],  # element 1 neighbors
    ], dtype=jnp.int32)

    element_to_block = jnp.array([0, 0], dtype=jnp.int32)

    return positions, connectivity, neighbors, element_to_block


class TestPointInTetrahedron:
    """Test point-in-tetrahedron kernel."""

    def test_point_inside_unit_tet(self):
        """Test point inside unit tetrahedron."""
        vertices = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=jnp.float32)

        # Centroid is inside
        centroid = jnp.mean(vertices, axis=0)
        assert point_in_tetrahedron_jax(centroid, vertices)

        # Point near centroid
        point = jnp.array([0.2, 0.2, 0.2])
        assert point_in_tetrahedron_jax(point, vertices)

    def test_point_outside_tet(self):
        """Test point outside tetrahedron."""
        vertices = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=jnp.float32)

        # Point far outside
        point = jnp.array([10.0, 10.0, 10.0])
        assert not point_in_tetrahedron_jax(point, vertices)

    def test_point_on_boundary(self):
        """Test point on tetrahedron boundary."""
        vertices = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=jnp.float32)

        # Point on face (should be inside due to tolerance)
        point = jnp.array([0.5, 0.5, 0.0])
        assert point_in_tetrahedron_jax(point, vertices)

    def test_safe_version_handles_degenerate(self):
        """Test safe version handles degenerate elements without crashing."""
        # Degenerate tetrahedron (all nodes coplanar)
        vertices = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.0],
        ], dtype=jnp.float32)

        point = jnp.array([0.5, 0.25, 0.0])
        # Should not crash (pseudoinverse handles degenerate case)
        result = point_in_tetrahedron_safe(point, vertices)
        # Result may be True or False for degenerate elements
        # Important thing is it doesn't crash
        assert isinstance(result, (bool, jnp.ndarray))


class TestSearchCachedElement:
    """Test Level 0: cached element search."""

    def test_level0_hit(self, simple_mesh_jax):
        """Test Level 0 hit."""
        positions, connectivity, _, _ = simple_mesh_jax

        point = jnp.array([0.2, 0.2, 0.2])
        cached_element_id = 0

        found, element_id = search_cached_element_jax(
            point, cached_element_id, positions, connectivity
        )

        assert found
        assert element_id == 0

    def test_level0_miss(self, simple_mesh_jax):
        """Test Level 0 miss."""
        positions, connectivity, _, _ = simple_mesh_jax

        point = jnp.array([0.7, 0.7, 0.7])
        cached_element_id = 0

        found, element_id = search_cached_element_jax(
            point, cached_element_id, positions, connectivity
        )

        assert not found

    def test_level0_no_cache(self, simple_mesh_jax):
        """Test Level 0 with no cache."""
        positions, connectivity, _, _ = simple_mesh_jax

        point = jnp.array([0.2, 0.2, 0.2])
        cached_element_id = -1

        found, element_id = search_cached_element_jax(
            point, cached_element_id, positions, connectivity
        )

        assert not found


class TestSearchNeighbors:
    """Test Level 1: neighbor search."""

    def test_level1_hit(self, simple_mesh_jax):
        """Test Level 1 hit."""
        positions, connectivity, neighbors, _ = simple_mesh_jax

        point = jnp.array([0.7, 0.7, 0.7])
        cached_element_id = 0

        found, element_id = search_neighbors_jax(
            point, cached_element_id, neighbors, positions, connectivity
        )

        assert found
        assert element_id == 1

    def test_level1_miss(self, simple_mesh_jax):
        """Test Level 1 miss."""
        positions, connectivity, neighbors, _ = simple_mesh_jax

        point = jnp.array([10.0, 10.0, 10.0])
        cached_element_id = 0

        found, element_id = search_neighbors_jax(
            point, cached_element_id, neighbors, positions, connectivity
        )

        assert not found


class TestThreeTierSearchGPU:
    """Test complete three-tier search on GPU."""

    def test_find_element_level0_hit(self, simple_mesh_jax):
        """Test GPU search with Level 0 hit."""
        positions, connectivity, neighbors, element_to_block = simple_mesh_jax

        point = jnp.array([0.2, 0.2, 0.2])
        cached_element_id = 0
        block_id = 0

        element_id = find_containing_element_gpu(
            point, cached_element_id, block_id,
            neighbors, element_to_block, positions, connectivity
        )

        assert element_id == 0

    def test_find_element_level1_hit(self, simple_mesh_jax):
        """Test GPU search with Level 1 hit."""
        positions, connectivity, neighbors, element_to_block = simple_mesh_jax

        point = jnp.array([0.7, 0.7, 0.7])
        cached_element_id = 0
        block_id = 0

        element_id = find_containing_element_gpu(
            point, cached_element_id, block_id,
            neighbors, element_to_block, positions, connectivity
        )

        assert element_id == 1

    def test_find_element_failure(self, simple_mesh_jax):
        """Test GPU search failure."""
        positions, connectivity, neighbors, element_to_block = simple_mesh_jax

        point = jnp.array([10.0, 10.0, 10.0])
        cached_element_id = 0
        block_id = 0

        element_id = find_containing_element_gpu(
            point, cached_element_id, block_id,
            neighbors, element_to_block, positions, connectivity
        )

        assert element_id == -1


class TestBatchSearch:
    """Test batched search over multiple particles."""

    def test_batch_search_simple(self, simple_mesh_jax):
        """Test batch search with 2 particles."""
        positions, connectivity, neighbors, element_to_block = simple_mesh_jax

        # Two particles: one in element 0, one in element 1
        points = jnp.array([
            [0.2, 0.2, 0.2],  # In element 0
            [0.7, 0.7, 0.7],  # In element 1
        ], dtype=jnp.float32)

        cached_ids = jnp.array([-1, -1], dtype=jnp.int32)  # No cache
        block_ids = jnp.array([0, 0], dtype=jnp.int32)

        element_ids = find_containing_elements_batch(
            points, cached_ids, block_ids,
            neighbors, element_to_block, positions, connectivity
        )

        assert element_ids[0] == 0
        assert element_ids[1] == 1

    def test_batch_search_with_cache(self, simple_mesh_jax):
        """Test batch search with cached elements."""
        positions, connectivity, neighbors, element_to_block = simple_mesh_jax

        points = jnp.array([
            [0.2, 0.2, 0.2],  # In element 0
            [0.7, 0.7, 0.7],  # In element 1
        ], dtype=jnp.float32)

        # Correct cache
        cached_ids = jnp.array([0, 1], dtype=jnp.int32)
        block_ids = jnp.array([0, 0], dtype=jnp.int32)

        element_ids = find_containing_elements_batch(
            points, cached_ids, block_ids,
            neighbors, element_to_block, positions, connectivity
        )

        # Should hit cache
        assert element_ids[0] == 0
        assert element_ids[1] == 1

    def test_batch_search_outside_particles(self, simple_mesh_jax):
        """Test batch search with particles outside domain."""
        positions, connectivity, neighbors, element_to_block = simple_mesh_jax

        points = jnp.array([
            [0.2, 0.2, 0.2],    # Inside
            [10.0, 10.0, 10.0],  # Outside
        ], dtype=jnp.float32)

        cached_ids = jnp.array([-1, -1], dtype=jnp.int32)
        block_ids = jnp.array([0, 0], dtype=jnp.int32)

        element_ids = find_containing_elements_batch(
            points, cached_ids, block_ids,
            neighbors, element_to_block, positions, connectivity
        )

        assert element_ids[0] == 0
        assert element_ids[1] == -1  # Outside


class TestPositionToBlockID:
    """Test fast position → block_id mapping."""

    def test_position_to_block_simple(self):
        """Test position to block ID."""
        domain_bounds = jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        grid_size = (2, 2, 2)

        # Point in block (0, 0, 0)
        position = jnp.array([0.25, 0.25, 0.25])
        block_id = position_to_block_id_jax(position, domain_bounds, grid_size)
        assert block_id == 0

        # Point in block (1, 0, 0)
        position = jnp.array([0.75, 0.25, 0.25])
        block_id = position_to_block_id_jax(position, domain_bounds, grid_size)
        assert block_id == 1

        # Point in block (0, 1, 0)
        position = jnp.array([0.25, 0.75, 0.25])
        block_id = position_to_block_id_jax(position, domain_bounds, grid_size)
        assert block_id == 2

    def test_position_outside_domain(self):
        """Test position outside domain."""
        domain_bounds = jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        grid_size = (2, 2, 2)

        position = jnp.array([10.0, 10.0, 10.0])
        block_id = position_to_block_id_jax(position, domain_bounds, grid_size)
        assert block_id == -1

    def test_batch_position_to_block(self):
        """Test batch position to block ID."""
        domain_bounds = jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        grid_size = (2, 2, 2)

        positions = jnp.array([
            [0.25, 0.25, 0.25],  # Block 0
            [0.75, 0.25, 0.25],  # Block 1
            [0.25, 0.75, 0.25],  # Block 2
            [10.0, 10.0, 10.0],  # Outside (-1)
        ])

        block_ids = positions_to_block_ids_batch(positions, domain_bounds, grid_size)

        assert block_ids[0] == 0
        assert block_ids[1] == 1
        assert block_ids[2] == 2
        assert block_ids[3] == -1
