"""
Unit tests for Morton code (Z-order curve) implementation.

Tests encoding, decoding, and spatial ordering properties.
"""

import pytest
import numpy as np
from jaxtrace.gpu.forest.morton_code import (
    expand_bits,
    compact_bits,
    encode_morton_3d,
    decode_morton_3d,
    normalize_positions,
    positions_to_morton,
    sort_by_morton,
    compute_block_morton_range,
)


class TestBitOperations:
    """Test bit expansion and compaction."""

    def test_expand_bits_simple(self):
        """Test bit expansion for simple values."""
        assert expand_bits(np.array([0])) == 0
        assert expand_bits(np.array([1])) == 1
        assert expand_bits(np.array([2])) == 8

    def test_expand_bits_array(self):
        """Test bit expansion for arrays."""
        values = np.array([0, 1, 2, 3])
        expanded = expand_bits(values)
        assert expanded.shape == (4,)
        assert expanded[0] == 0
        assert expanded[1] == 1

    def test_compact_bits_inverse(self):
        """Test that compact_bits is inverse of expand_bits."""
        values = np.array([0, 1, 5, 10, 100, 1023])
        expanded = expand_bits(values)
        compacted = compact_bits(expanded)
        assert np.array_equal(compacted, values)


class TestMortonEncoding:
    """Test Morton code encoding."""

    def test_encode_origin(self):
        """Test encoding origin (0, 0, 0)."""
        x, y, z = np.array([0]), np.array([0]), np.array([0])
        morton = encode_morton_3d(x, y, z)
        assert morton[0] == 0

    def test_encode_unit_axes(self):
        """Test encoding unit vectors."""
        # (1, 0, 0) → 0b001 = 1
        morton_x = encode_morton_3d(np.array([1]), np.array([0]), np.array([0]))
        assert morton_x[0] == 1

        # (0, 1, 0) → 0b010 = 2
        morton_y = encode_morton_3d(np.array([0]), np.array([1]), np.array([0]))
        assert morton_y[0] == 2

        # (0, 0, 1) → 0b100 = 4
        morton_z = encode_morton_3d(np.array([0]), np.array([0]), np.array([1]))
        assert morton_z[0] == 4

    def test_encode_diagonal(self):
        """Test encoding diagonal point."""
        # (1, 1, 1) → 0b111 = 7
        morton = encode_morton_3d(np.array([1]), np.array([1]), np.array([1]))
        assert morton[0] == 7

    def test_encode_array(self):
        """Test encoding array of points."""
        x = np.array([0, 1, 2, 3])
        y = np.array([0, 0, 0, 0])
        z = np.array([0, 0, 0, 0])
        morton = encode_morton_3d(x, y, z)
        assert morton.shape == (4,)
        assert morton[0] == 0
        assert morton[1] == 1


class TestMortonDecoding:
    """Test Morton code decoding."""

    def test_decode_origin(self):
        """Test decoding origin."""
        morton = np.array([0], dtype=np.uint32)
        x, y, z = decode_morton_3d(morton)
        assert x[0] == 0
        assert y[0] == 0
        assert z[0] == 0

    def test_decode_unit_axes(self):
        """Test decoding unit vectors."""
        x, y, z = decode_morton_3d(np.array([1], dtype=np.uint32))
        assert x[0] == 1 and y[0] == 0 and z[0] == 0

        x, y, z = decode_morton_3d(np.array([2], dtype=np.uint32))
        assert x[0] == 0 and y[0] == 1 and z[0] == 0

        x, y, z = decode_morton_3d(np.array([4], dtype=np.uint32))
        assert x[0] == 0 and y[0] == 0 and z[0] == 1

    def test_decode_diagonal(self):
        """Test decoding diagonal."""
        x, y, z = decode_morton_3d(np.array([7], dtype=np.uint32))
        assert x[0] == 1
        assert y[0] == 1
        assert z[0] == 1

    def test_encode_decode_inverse(self):
        """Test that decode is inverse of encode."""
        x_orig = np.array([0, 1, 5, 10, 100])
        y_orig = np.array([0, 2, 6, 11, 101])
        z_orig = np.array([0, 3, 7, 12, 102])

        morton = encode_morton_3d(x_orig, y_orig, z_orig)
        x_dec, y_dec, z_dec = decode_morton_3d(morton)

        assert np.array_equal(x_dec, x_orig)
        assert np.array_equal(y_dec, y_orig)
        assert np.array_equal(z_dec, z_orig)


class TestNormalization:
    """Test position normalization."""

    def test_normalize_unit_cube(self):
        """Test normalization in unit cube."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
        ], dtype=np.float32)
        bounds = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

        normalized = normalize_positions(positions, bounds, max_value=1023)

        assert normalized[0, 0] == 0
        assert normalized[1, 0] == 1023
        assert np.allclose(normalized[2, 0], 511, atol=1)

    def test_normalize_arbitrary_bounds(self):
        """Test normalization with arbitrary bounds."""
        positions = np.array([
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
        ], dtype=np.float32)
        bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

        normalized = normalize_positions(positions, bounds, max_value=1023)

        assert normalized[0, 0] == 0
        assert normalized[1, 0] == 1023

    def test_normalize_clamps_outside(self):
        """Test that normalization clamps points outside bounds."""
        positions = np.array([
            [-10.0, -10.0, -10.0],  # Far outside
            [10.0, 10.0, 10.0],      # Far outside
        ], dtype=np.float32)
        bounds = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

        normalized = normalize_positions(positions, bounds, max_value=1023)

        # Should be clamped to [0, 1023]
        assert normalized[0, 0] == 0
        assert normalized[1, 0] == 1023


class TestPositionsToMorton:
    """Test convenience function for positions → Morton."""

    def test_positions_to_morton_simple(self):
        """Test positions to Morton codes."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ], dtype=np.float32)
        bounds = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

        morton = positions_to_morton(positions, bounds)

        assert morton.shape == (2,)
        assert morton[0] == 0  # Origin
        # morton[1] should be large (1023, 1023, 1023)

    def test_positions_to_morton_preserves_order(self):
        """Test that nearby positions have nearby Morton codes."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.1, 0.1, 0.0],
        ], dtype=np.float32)
        bounds = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

        morton = positions_to_morton(positions, bounds)

        # Morton codes should be in ascending order (Z-curve property)
        assert morton[0] < morton[3]  # (0,0,0) < (0.1, 0.1, 0)


class TestSortByMorton:
    """Test Morton order sorting."""

    def test_sort_by_morton_simple(self):
        """Test sorting positions by Morton code."""
        # Create positions in random order
        positions = np.array([
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ], dtype=np.float32)
        bounds = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

        sorted_pos, _, morton, indices = sort_by_morton(positions, bounds)

        # Positions should be sorted by Morton code
        assert np.allclose(sorted_pos[0], [0.0, 0.0, 0.0])
        assert np.allclose(sorted_pos[2], [1.0, 1.0, 1.0])

        # Morton codes should be sorted
        assert np.all(morton[:-1] <= morton[1:])

    def test_sort_by_morton_with_data(self):
        """Test sorting with associated data."""
        positions = np.array([
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float32)
        data = np.array([100, 200])
        bounds = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

        sorted_pos, sorted_data, _, _ = sort_by_morton(positions, bounds, data)

        # Data should follow position sorting
        assert sorted_data[0] == 200  # Was at position [0, 0, 0]
        assert sorted_data[1] == 100  # Was at position [1, 1, 1]

    def test_sort_large_array(self):
        """Test sorting large array for spatial coherence."""
        np.random.seed(42)
        positions = np.random.uniform(0, 1, (1000, 3)).astype(np.float32)
        bounds = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

        sorted_pos, _, morton, _ = sort_by_morton(positions, bounds)

        # Morton codes should be sorted
        assert np.all(morton[:-1] <= morton[1:])

        # Spatial coherence: consecutive points should be nearby
        # (not a strict test, but should hold statistically)
        distances = np.linalg.norm(sorted_pos[1:] - sorted_pos[:-1], axis=1)
        mean_distance = np.mean(distances)
        assert mean_distance < 0.5  # Consecutive points relatively close


class TestBlockMortonRange:
    """Test Morton range computation for blocks."""

    def test_block_morton_range_unit_cube(self):
        """Test Morton range for block in unit cube."""
        # Block covering lower octant
        block_bounds = np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])
        domain_bounds = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

        min_code, max_code = compute_block_morton_range(block_bounds, domain_bounds)

        # Min should be 0 (origin), max should be at (0.5, 0.5, 0.5)
        assert min_code == 0
        assert max_code > 0

    def test_block_morton_range_ordering(self):
        """Test that block ranges are ordered."""
        domain_bounds = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

        # Two non-overlapping blocks
        block1_bounds = np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])
        block2_bounds = np.array([0.5, 1.0, 0.5, 1.0, 0.5, 1.0])

        min1, max1 = compute_block_morton_range(block1_bounds, domain_bounds)
        min2, max2 = compute_block_morton_range(block2_bounds, domain_bounds)

        # Block 1 range should be less than block 2 range (generally)
        # This is not always true for Z-curve, but holds for these corners
        assert min1 < max2
