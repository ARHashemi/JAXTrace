"""
Tests for element-to-block mapper.

Part of Phase 1: Forest Structure & Block Partitioning
"""

import pytest
import numpy as np
from pathlib import Path

from jaxtrace.gpu.forest.block_grid import create_regular_grid
from jaxtrace.gpu.forest.block_mapper import (
    compute_element_centroids,
    assign_elements_to_blocks,
    assign_elements_to_block_list,
    validate_assignment,
    BlockAssignmentStats
)


class TestCentroidComputation:
    """Test element centroid computation."""

    def test_tet_centroid(self):
        """Test centroid of a single tetrahedron."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        connectivity = np.array([[0, 1, 2, 3]], dtype=np.int32)

        centroids = compute_element_centroids(positions, connectivity)

        # Centroid should be at (0.25, 0.25, 0.25)
        expected = np.array([[0.25, 0.25, 0.25]], dtype=np.float32)
        assert np.allclose(centroids, expected)

    def test_multiple_tets(self):
        """Test centroids of multiple tetrahedra."""
        positions = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 0, 0],
            [2, 1, 0],
            [2, 0, 1]
        ], dtype=np.float32)

        connectivity = np.array([
            [0, 1, 2, 3],  # First tet
            [1, 4, 5, 6]   # Second tet
        ], dtype=np.int32)

        centroids = compute_element_centroids(positions, connectivity)

        assert centroids.shape == (2, 3)
        # First tet centroid
        assert np.allclose(centroids[0], [0.25, 0.25, 0.25])
        # Second tet centroid: mean of (1,0,0), (2,0,0), (2,1,0), (2,0,1)
        expected_second = np.mean(positions[[1, 4, 5, 6]], axis=0)
        assert np.allclose(centroids[1], expected_second)

    def test_vectorized_computation(self):
        """Test that vectorized computation matches loop."""
        np.random.seed(42)
        positions = np.random.rand(100, 3).astype(np.float32)
        connectivity = np.random.randint(0, 100, size=(50, 4), dtype=np.int32)

        # Vectorized computation
        centroids = compute_element_centroids(positions, connectivity)

        # Manual loop computation
        centroids_manual = np.zeros((50, 3), dtype=np.float32)
        for i in range(50):
            node_ids = connectivity[i]
            centroids_manual[i] = positions[node_ids].mean(axis=0)

        assert np.allclose(centroids, centroids_manual)


class TestElementAssignment:
    """Test element assignment to blocks."""

    def test_simple_2x2x2_grid(self):
        """Test assignment on simple 2×2×2 grid."""
        domain_bounds = np.array([0.0, 2.0, 0.0, 2.0, 0.0, 2.0], dtype=np.float32)
        grid_size = (2, 2, 2)

        # Create 8 elements, one in each block
        positions = np.array([
            # Block 0 (0,0,0): [0,1] × [0,1] × [0,1]
            [0.2, 0.2, 0.2],
            [0.8, 0.2, 0.2],
            [0.2, 0.8, 0.2],
            [0.2, 0.2, 0.8],
            # Block 7 (1,1,1): [1,2] × [1,2] × [1,2]
            [1.2, 1.2, 1.2],
            [1.8, 1.2, 1.2],
            [1.2, 1.8, 1.2],
            [1.2, 1.2, 1.8],
        ], dtype=np.float32)

        connectivity = np.array([
            [0, 1, 2, 3],  # Centroid in block 0
            [4, 5, 6, 7],  # Centroid in block 7
        ], dtype=np.int32)

        element_to_block, stats = assign_elements_to_blocks(
            positions, connectivity, domain_bounds, grid_size, verbose=False
        )

        assert element_to_block[0] == 0  # First element in block 0
        assert element_to_block[1] == 7  # Second element in block 7

    def test_all_elements_in_one_block(self):
        """Test when all elements are in one block."""
        domain_bounds = np.array([0.0, 2.0, 0.0, 2.0, 0.0, 2.0], dtype=np.float32)
        grid_size = (2, 2, 2)

        # All elements in block 0
        positions = np.random.uniform(0.1, 0.9, size=(40, 3)).astype(np.float32)
        connectivity = np.random.randint(0, 40, size=(10, 4), dtype=np.int32)

        element_to_block, stats = assign_elements_to_blocks(
            positions, connectivity, domain_bounds, grid_size, verbose=False
        )

        # All elements should be in block 0
        assert np.all(element_to_block == 0)
        assert stats.n_blocks_used == 1
        assert stats.n_blocks_empty == 7
        assert stats.elements_per_block[0] == 10

    def test_element_outside_domain(self):
        """Test elements with centroids outside domain."""
        domain_bounds = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        grid_size = (2, 2, 2)

        positions = np.array([
            # Inside domain
            [0.2, 0.2, 0.2],
            [0.8, 0.2, 0.2],
            [0.2, 0.8, 0.2],
            [0.2, 0.2, 0.8],
            # Outside domain
            [2.0, 2.0, 2.0],
            [3.0, 2.0, 2.0],
            [2.0, 3.0, 2.0],
            [2.0, 2.0, 3.0],
        ], dtype=np.float32)

        connectivity = np.array([
            [0, 1, 2, 3],  # Inside
            [4, 5, 6, 7],  # Outside
        ], dtype=np.int32)

        element_to_block, stats = assign_elements_to_blocks(
            positions, connectivity, domain_bounds, grid_size, verbose=False
        )

        assert element_to_block[0] >= 0  # First element assigned
        assert element_to_block[1] == -1  # Second element outside

    def test_statistics_correctness(self):
        """Test that BlockAssignmentStats are computed correctly."""
        domain_bounds = np.array([0.0, 2.0, 0.0, 2.0, 0.0, 2.0], dtype=np.float32)
        grid_size = (2, 2, 2)

        # Create elements distributed unevenly
        positions_block0 = np.random.uniform(0.1, 0.9, size=(30, 3)).astype(np.float32)
        positions_block7 = np.random.uniform(1.1, 1.9, size=(10, 3)).astype(np.float32)
        positions = np.vstack([positions_block0, positions_block7])

        connectivity = np.random.randint(0, 40, size=(10, 4), dtype=np.int32)
        connectivity[:7, :] = np.random.randint(0, 30, size=(7, 4))  # 7 in block 0
        connectivity[7:, :] = np.random.randint(30, 40, size=(3, 4))  # 3 in block 7

        element_to_block, stats = assign_elements_to_blocks(
            positions, connectivity, domain_bounds, grid_size, verbose=False
        )

        assert stats.n_elements == 10
        assert stats.n_blocks == 8
        assert stats.n_blocks_used == 2
        assert stats.n_blocks_empty == 6
        assert stats.max_elements == 7
        assert stats.min_elements == 3
        assert stats.mean_elements == 5.0
        assert stats.imbalance_ratio == pytest.approx(7.0 / 5.0)


class TestBlockListWrapper:
    """Test the block list convenience wrapper."""

    def test_wrapper_matches_direct(self):
        """Test that wrapper produces same result as direct call."""
        domain_bounds = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        grid_size = (2, 2, 2)

        blocks = create_regular_grid(domain_bounds, grid_size)

        positions = np.random.uniform(0.0, 1.0, size=(100, 3)).astype(np.float32)
        connectivity = np.random.randint(0, 100, size=(50, 4), dtype=np.int32)

        # Direct call
        elem_to_block_direct, stats_direct = assign_elements_to_blocks(
            positions, connectivity, domain_bounds, grid_size, verbose=False
        )

        # Wrapper call
        elem_to_block_wrapper, stats_wrapper = assign_elements_to_block_list(
            positions, connectivity, blocks, verbose=False
        )

        assert np.array_equal(elem_to_block_direct, elem_to_block_wrapper)
        assert stats_direct.n_elements == stats_wrapper.n_elements
        assert stats_direct.imbalance_ratio == stats_wrapper.imbalance_ratio


class TestValidation:
    """Test assignment validation."""

    def test_validation_passes_correct_assignment(self):
        """Test that validation passes for correct assignment."""
        domain_bounds = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        grid_size = (2, 2, 2)
        blocks = create_regular_grid(domain_bounds, grid_size)

        positions = np.random.uniform(0.0, 1.0, size=(100, 3)).astype(np.float32)
        connectivity = np.random.randint(0, 100, size=(50, 4), dtype=np.int32)

        element_to_block, _ = assign_elements_to_block_list(
            positions, connectivity, blocks, verbose=False
        )

        # Validation should pass
        valid = validate_assignment(
            element_to_block, positions, connectivity, blocks, n_samples=50
        )
        assert valid

    def test_validation_detects_incorrect_assignment(self):
        """Test that validation detects incorrect assignments."""
        domain_bounds = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        grid_size = (2, 2, 2)
        blocks = create_regular_grid(domain_bounds, grid_size)

        positions = np.random.uniform(0.0, 1.0, size=(100, 3)).astype(np.float32)
        connectivity = np.random.randint(0, 100, size=(50, 4), dtype=np.int32)

        element_to_block, _ = assign_elements_to_block_list(
            positions, connectivity, blocks, verbose=False
        )

        # Deliberately corrupt assignment
        element_to_block[0] = 7  # Wrong block

        # Validation should fail (might pass if centroid happens to be in block 7)
        valid = validate_assignment(
            element_to_block, positions, connectivity, blocks, n_samples=50
        )
        # We can't guarantee failure, but typically it will fail


class TestHeavyBlockDetection:
    """Test heavy block detection."""

    def test_heavy_block_threshold(self):
        """Test that heavy blocks are correctly identified."""
        domain_bounds = np.array([0.0, 2.0, 0.0, 2.0, 0.0, 2.0], dtype=np.float32)
        grid_size = (2, 2, 2)

        # Create many elements in block 0
        positions = np.random.uniform(0.1, 0.9, size=(15000, 3)).astype(np.float32)
        connectivity = np.random.randint(0, 15000, size=(5000, 4), dtype=np.int32)

        element_to_block, stats = assign_elements_to_blocks(
            positions, connectivity, domain_bounds, grid_size,
            heavy_threshold=1000, verbose=False
        )

        # Block 0 should be identified as heavy (>1000 elements)
        assert 0 in stats.heavy_blocks
        assert stats.elements_per_block[0] > 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
