"""
Tests for forest block grid generator.

Part of Phase 1: Forest Structure & Block Partitioning
"""

import pytest
import numpy as np
from jaxtrace.gpu.forest.block_grid import (
    create_regular_grid, compute_6_neighbors, compute_26_neighbors,
    position_to_block_id, find_block_containing_point, infer_grid_size, Block
)


class TestBlockCreation:
    """Test block creation and properties."""

    def test_grid_2x2x2(self):
        """Test 2×2×2 grid creation."""
        domain_bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)
        blocks = create_regular_grid(domain_bounds, (2, 2, 2))

        assert len(blocks) == 8
        assert all(b.block_id == i for i, b in enumerate(blocks))

        # Check bounds for first block (0,0,0)
        assert np.allclose(blocks[0].bounds, [-1, 0, -1, 0, -1, 0])
        # Check bounds for last block (1,1,1)
        assert np.allclose(blocks[7].bounds, [0, 1, 0, 1, 0, 1])

        # Check grid indices
        assert blocks[0].grid_index == (0, 0, 0)
        assert blocks[7].grid_index == (1, 1, 1)

    def test_grid_4x4x2_threadeda(self):
        """Test 4×4×2 grid creation (ThreadedA configuration)."""
        domain_bounds = np.array([-0.03, 0.03, -0.023, 0.023, -0.01, 0.0], dtype=np.float32)
        blocks = create_regular_grid(domain_bounds, (4, 4, 2))

        assert len(blocks) == 32

        # Check block sizes are uniform
        sizes = [b.size for b in blocks]
        assert np.allclose(sizes[0], sizes[-1])

        # Expected block size
        expected_size = np.array([0.015, 0.0115, 0.005], dtype=np.float32)  # 15mm, 11.5mm, 5mm
        assert np.allclose(sizes[0], expected_size, atol=1e-5)

        # Check total volume
        total_volume = sum(b.volume for b in blocks)
        domain_volume = 0.06 * 0.046 * 0.01
        assert np.isclose(total_volume, domain_volume, rtol=1e-6)

    def test_block_properties(self):
        """Test Block dataclass properties."""
        bounds = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32)
        center = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        neighbors_6 = np.array([1, -1, 2, -1, 3, -1], dtype=np.int32)
        neighbors_26 = np.full(26, -1, dtype=np.int32)

        block = Block(
            block_id=0,
            bounds=bounds,
            center=center,
            grid_index=(0, 0, 0),
            neighbors_6=neighbors_6,
            neighbors_26=neighbors_26
        )

        assert block.volume == pytest.approx(1.0)
        assert np.allclose(block.size, [1, 1, 1])
        assert block.contains_point(np.array([0.5, 0.5, 0.5]))
        assert not block.contains_point(np.array([1.5, 0.5, 0.5]))

    def test_block_contains_boundary(self):
        """Test boundary containment with tolerance."""
        block = Block(
            block_id=0,
            bounds=np.array([0, 1, 0, 1, 0, 1], dtype=np.float32),
            center=np.array([0.5, 0.5, 0.5], dtype=np.float32),
            grid_index=(0, 0, 0),
            neighbors_6=np.zeros(6, dtype=np.int32),
            neighbors_26=np.zeros(26, dtype=np.int32)
        )

        # Point exactly on boundary should be contained
        assert block.contains_point(np.array([1.0, 0.5, 0.5]))
        assert block.contains_point(np.array([0.0, 0.5, 0.5]))

        # Point just outside
        assert not block.contains_point(np.array([1.001, 0.5, 0.5]))


class TestNeighborTopology:
    """Test 6-face and 26-neighbor topology."""

    def test_6_neighbors_corner(self):
        """Corner block (0,0,0) has 3 face neighbors."""
        neighbors = compute_6_neighbors(0, 0, 0, 2, 2, 2)

        # +x, +y, +z should be valid
        assert neighbors[0] >= 0  # +x
        assert neighbors[2] >= 0  # +y
        assert neighbors[4] >= 0  # +z

        # -x, -y, -z should be -1 (boundary)
        assert neighbors[1] == -1  # -x
        assert neighbors[3] == -1  # -y
        assert neighbors[5] == -1  # -z

    def test_6_neighbors_interior(self):
        """Interior block has 6 face neighbors."""
        neighbors = compute_6_neighbors(1, 1, 1, 3, 3, 3)

        # All 6 should be valid
        assert all(n >= 0 for n in neighbors)
        assert len(set(neighbors)) == 6  # All unique

    def test_6_neighbors_values(self):
        """Check actual neighbor IDs are correct."""
        # Block (1, 1, 0) in 3×3×2 grid (block_id = 1 + 1*3 + 0*9 = 4)
        neighbors = compute_6_neighbors(1, 1, 0, 3, 3, 2)

        # Expected neighbors:
        # +x: (2,1,0) = 2 + 1*3 + 0*9 = 5
        # -x: (0,1,0) = 0 + 1*3 + 0*9 = 3
        # +y: (1,2,0) = 1 + 2*3 + 0*9 = 7
        # -y: (1,0,0) = 1 + 0*3 + 0*9 = 1
        # +z: (1,1,1) = 1 + 1*3 + 1*9 = 13
        # -z: -1 (boundary)

        assert neighbors[0] == 5   # +x
        assert neighbors[1] == 3   # -x
        assert neighbors[2] == 7   # +y
        assert neighbors[3] == 1   # -y
        assert neighbors[4] == 13  # +z
        assert neighbors[5] == -1  # -z (boundary)

    def test_26_neighbors_corner(self):
        """Corner block (0,0,0) has 7 neighbors (3 faces + 3 edges + 1 corner)."""
        neighbors = compute_26_neighbors(0, 0, 0, 2, 2, 2)

        valid_neighbors = neighbors[neighbors >= 0]
        assert len(valid_neighbors) == 7

    def test_26_neighbors_interior(self):
        """Interior block has 26 neighbors."""
        neighbors = compute_26_neighbors(1, 1, 1, 3, 3, 3)

        valid_neighbors = neighbors[neighbors >= 0]
        assert len(valid_neighbors) == 26
        assert len(set(valid_neighbors)) == 26  # All unique

    def test_neighbor_symmetry(self):
        """If A is neighbor of B, then B is neighbor of A."""
        blocks = create_regular_grid(np.array([0, 1, 0, 1, 0, 1], dtype=np.float32), (2, 2, 2))

        for block in blocks:
            for neighbor_id in block.neighbors_26:
                if neighbor_id >= 0:
                    neighbor_block = blocks[neighbor_id]
                    # block.block_id should be in neighbor's neighbor list
                    assert block.block_id in neighbor_block.neighbors_26, \
                        f"Asymmetric neighbors: {block.block_id} → {neighbor_id}"


class TestPositionMapping:
    """Test position to block ID mapping."""

    def test_position_to_block_id_center(self):
        """Position at block center maps correctly."""
        domain_bounds = np.array([0.0, 2.0, 0.0, 2.0, 0.0, 2.0], dtype=np.float32)
        grid_size = (2, 2, 2)

        # Center of block (0,0,0) = [0.5, 0.5, 0.5]
        pos = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        block_id = position_to_block_id(pos, domain_bounds, grid_size)
        assert block_id == 0

        # Center of block (1,1,1) = [1.5, 1.5, 1.5]
        pos = np.array([1.5, 1.5, 1.5], dtype=np.float32)
        block_id = position_to_block_id(pos, domain_bounds, grid_size)
        assert block_id == 7  # 1 + 1*2 + 1*4 = 7

    def test_position_to_block_id_boundary(self):
        """Position on block boundary handled consistently."""
        domain_bounds = np.array([0.0, 2.0, 0.0, 2.0, 0.0, 2.0], dtype=np.float32)
        grid_size = (2, 2, 2)

        # On boundary between blocks (0,0,0) and (1,0,0)
        pos = np.array([1.0, 0.5, 0.5], dtype=np.float32)
        block_id = position_to_block_id(pos, domain_bounds, grid_size)
        # Should assign to one (implementation: lower index due to floor)
        assert block_id in [0, 1]

    def test_position_outside_domain(self):
        """Position outside domain returns -1."""
        domain_bounds = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        grid_size = (2, 2, 2)

        pos = np.array([1.5, 0.5, 0.5], dtype=np.float32)  # Outside
        block_id = position_to_block_id(pos, domain_bounds, grid_size)
        assert block_id == -1

    def test_position_threadeda(self):
        """Test with ThreadedA domain."""
        domain_bounds = np.array([-0.03, 0.03, -0.023, 0.023, -0.01, 0.0], dtype=np.float32)
        grid_size = (4, 4, 2)

        # Origin (center of domain)
        pos = np.array([0.0, 0.0, -0.005], dtype=np.float32)
        block_id = position_to_block_id(pos, domain_bounds, grid_size)
        assert 0 <= block_id < 32

    def test_linear_search_matches_fast_lookup(self):
        """Linear search and O(1) lookup should agree."""
        domain_bounds = np.array([0.0, 2.0, 0.0, 2.0, 0.0, 2.0], dtype=np.float32)
        grid_size = (2, 2, 2)
        blocks = create_regular_grid(domain_bounds, grid_size)

        # Test 100 random points
        np.random.seed(42)
        for _ in range(100):
            pos = np.random.uniform(0, 2, 3).astype(np.float32)

            block_id_fast = position_to_block_id(pos, domain_bounds, grid_size)
            block_id_linear = find_block_containing_point(pos, blocks)

            assert block_id_fast == block_id_linear, \
                f"Mismatch for position {pos}: fast={block_id_fast}, linear={block_id_linear}"


class TestUtilities:
    """Test utility functions."""

    def test_infer_grid_size_2x2x2(self):
        """Infer grid size from blocks."""
        blocks = create_regular_grid(np.array([0, 1, 0, 1, 0, 1], dtype=np.float32), (2, 2, 2))
        grid_size = infer_grid_size(blocks)
        assert grid_size == (2, 2, 2)

    def test_infer_grid_size_4x4x2(self):
        """Infer ThreadedA grid size."""
        bounds = np.array([-0.03, 0.03, -0.023, 0.023, -0.01, 0.0], dtype=np.float32)
        blocks = create_regular_grid(bounds, (4, 4, 2))
        grid_size = infer_grid_size(blocks)
        assert grid_size == (4, 4, 2)


class TestDomainCoverage:
    """Test that grid covers domain completely."""

    def test_no_gaps(self):
        """Every point in domain belongs to exactly one block."""
        domain_bounds = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        blocks = create_regular_grid(domain_bounds, (2, 2, 2))

        # Test 1000 random points
        np.random.seed(42)
        for _ in range(1000):
            pos = np.random.uniform(0, 1, 3).astype(np.float32)

            # Find all blocks containing this point
            containing_blocks = [b.block_id for b in blocks if b.contains_point(pos)]

            # Should be in exactly one block (or two if on boundary)
            assert len(containing_blocks) >= 1, f"Point {pos} not in any block"

    def test_volume_conservation(self):
        """Sum of block volumes equals domain volume."""
        domain_bounds = np.array([0.0, 2.0, 0.0, 3.0, 0.0, 4.0], dtype=np.float32)
        blocks = create_regular_grid(domain_bounds, (2, 3, 4))

        total_block_volume = sum(b.volume for b in blocks)
        domain_volume = 2.0 * 3.0 * 4.0

        assert np.isclose(total_block_volume, domain_volume, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
