"""
Unit tests for forest block builder.
"""

import pytest
import numpy as np
from jaxtrace.gpu.forest import (
    BlockMetadata,
    create_regular_forest_grid,
    find_block_containing_point,
)
from jaxtrace.gpu.forest.block_builder import (
    compute_block_neighbors,
    position_to_block_id,
)


def test_block_metadata_creation():
    """Test BlockMetadata creation."""
    bounds = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    center = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    neighbors = np.array([1, -1, 2, -1, 3, -1], dtype=np.int32)

    block = BlockMetadata(
        block_id=0,
        bounds=bounds,
        center=center,
        grid_index=(0, 0, 0),
        neighbors=neighbors
    )

    assert block.block_id == 0
    assert block.grid_index == (0, 0, 0)
    assert np.allclose(block.center, [0.5, 0.5, 0.5])


def test_block_volume():
    """Test block volume calculation."""
    bounds = np.array([0.0, 2.0, 0.0, 3.0, 0.0, 4.0], dtype=np.float32)
    center = np.array([1.0, 1.5, 2.0], dtype=np.float32)
    neighbors = np.zeros(6, dtype=np.int32)

    block = BlockMetadata(
        block_id=0,
        bounds=bounds,
        center=center,
        grid_index=(0, 0, 0),
        neighbors=neighbors
    )

    assert block.volume == 2.0 * 3.0 * 4.0  # 24.0


def test_block_size():
    """Test block size calculation."""
    bounds = np.array([0.0, 2.0, 1.0, 4.0, 2.0, 7.0], dtype=np.float32)
    center = np.array([1.0, 2.5, 4.5], dtype=np.float32)
    neighbors = np.zeros(6, dtype=np.int32)

    block = BlockMetadata(
        block_id=0,
        bounds=bounds,
        center=center,
        grid_index=(0, 0, 0),
        neighbors=neighbors
    )

    size = block.size
    assert np.allclose(size, [2.0, 3.0, 5.0])


def test_block_contains_point():
    """Test point-in-block test."""
    bounds = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    center = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    neighbors = np.zeros(6, dtype=np.int32)

    block = BlockMetadata(
        block_id=0,
        bounds=bounds,
        center=center,
        grid_index=(0, 0, 0),
        neighbors=neighbors
    )

    # Inside
    assert block.contains_point(np.array([0.5, 0.5, 0.5]))
    assert block.contains_point(np.array([0.0, 0.0, 0.0]))  # corner
    assert block.contains_point(np.array([1.0, 1.0, 1.0]))  # corner

    # Outside
    assert not block.contains_point(np.array([1.5, 0.5, 0.5]))
    assert not block.contains_point(np.array([-0.1, 0.5, 0.5]))


def test_create_regular_forest_grid_2x2x2():
    """Test 2×2×2 grid creation."""
    bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    blocks = create_regular_forest_grid(bounds, (2, 2, 2))

    assert len(blocks) == 8

    # Check block 0 (corner)
    block0 = blocks[0]
    assert block0.block_id == 0
    assert block0.grid_index == (0, 0, 0)
    assert np.allclose(block0.bounds, [-1.0, 0.0, -1.0, 0.0, -1.0, 0.0])
    assert np.allclose(block0.center, [-0.5, -0.5, -0.5])

    # Check neighbors of block 0
    # [+x, -x, +y, -y, +z, -z]
    assert block0.neighbors[0] == 1   # +x
    assert block0.neighbors[1] == -1  # -x (boundary)
    assert block0.neighbors[2] == 2   # +y
    assert block0.neighbors[3] == -1  # -y (boundary)
    assert block0.neighbors[4] == 4   # +z
    assert block0.neighbors[5] == -1  # -z (boundary)


def test_create_regular_forest_grid_4x4x2():
    """Test 4×4×2 grid creation (production default)."""
    bounds = np.array([-0.03, 0.03, -0.023, 0.023, -0.01, 0.0], dtype=np.float32)
    blocks = create_regular_forest_grid(bounds, (4, 4, 2))

    assert len(blocks) == 32

    # Block size
    dx = 0.06 / 4  # 0.015
    dy = 0.046 / 4  # 0.0115
    dz = 0.01 / 2   # 0.005

    # Check first block
    block0 = blocks[0]
    assert block0.grid_index == (0, 0, 0)
    assert np.allclose(block0.size, [dx, dy, dz], atol=1e-6)

    # Check last block
    block31 = blocks[31]
    assert block31.grid_index == (3, 3, 1)
    assert block31.block_id == 31


def test_compute_block_neighbors_corner():
    """Test neighbor computation for corner block."""
    # Block at (0, 0, 0) in 3×3×3 grid
    neighbors = compute_block_neighbors(0, 0, 0, 3, 3, 3)

    # [+x, -x, +y, -y, +z, -z]
    assert neighbors[0] == 1   # +x exists
    assert neighbors[1] == -1  # -x boundary
    assert neighbors[2] == 3   # +y exists
    assert neighbors[3] == -1  # -y boundary
    assert neighbors[4] == 9   # +z exists
    assert neighbors[5] == -1  # -z boundary


def test_compute_block_neighbors_interior():
    """Test neighbor computation for interior block."""
    # Block at (1, 1, 1) in 3×3×3 grid (center)
    neighbors = compute_block_neighbors(1, 1, 1, 3, 3, 3)

    # All neighbors should exist (no -1)
    assert neighbors[0] == 2 + 1*3 + 1*9  # +x: (2, 1, 1)
    assert neighbors[1] == 0 + 1*3 + 1*9  # -x: (0, 1, 1)
    assert neighbors[2] == 1 + 2*3 + 1*9  # +y: (1, 2, 1)
    assert neighbors[3] == 1 + 0*3 + 1*9  # -y: (1, 0, 1)
    assert neighbors[4] == 1 + 1*3 + 2*9  # +z: (1, 1, 2)
    assert neighbors[5] == 1 + 1*3 + 0*9  # -z: (1, 1, 0)

    # No boundary neighbors
    assert all(n >= 0 for n in neighbors)


def test_find_block_containing_point():
    """Test point location."""
    bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    blocks = create_regular_forest_grid(bounds, (2, 2, 2))

    # Point in block 0 (negative octant)
    point0 = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
    block_id = find_block_containing_point(point0, blocks)
    assert block_id == 0

    # Point in block 7 (positive octant)
    point7 = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    block_id = find_block_containing_point(point7, blocks)
    assert block_id == 7

    # Point outside domain
    point_out = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    block_id = find_block_containing_point(point_out, blocks)
    assert block_id == -1


def test_position_to_block_id_fast():
    """Test fast position → block_id mapping."""
    bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    grid_size = (2, 2, 2)

    # Point in block 0
    pos0 = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
    block_id = position_to_block_id(pos0, bounds, grid_size)
    assert block_id == 0

    # Point in block 7
    pos7 = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    block_id = position_to_block_id(pos7, bounds, grid_size)
    assert block_id == 7

    # Point outside
    pos_out = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    block_id = position_to_block_id(pos_out, bounds, grid_size)
    assert block_id == -1


def test_position_to_block_id_matches_find_block():
    """Test that fast mapping matches linear search."""
    bounds = np.array([-0.03, 0.03, -0.023, 0.023, -0.01, 0.0], dtype=np.float32)
    grid_size = (4, 4, 2)
    blocks = create_regular_forest_grid(bounds, grid_size)

    # Test 100 random points
    np.random.seed(42)
    for _ in range(100):
        pos = np.random.uniform(
            [bounds[0], bounds[2], bounds[4]],
            [bounds[1], bounds[3], bounds[5]]
        ).astype(np.float32)

        block_id_fast = position_to_block_id(pos, bounds, grid_size)
        block_id_search = find_block_containing_point(pos, blocks)

        assert block_id_fast == block_id_search, \
            f"Mismatch at {pos}: fast={block_id_fast}, search={block_id_search}"


def test_all_blocks_cover_domain():
    """Test that blocks cover entire domain without gaps."""
    bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    blocks = create_regular_forest_grid(bounds, (4, 4, 2))

    # Test grid of points
    for x in np.linspace(bounds[0], bounds[1], 20):
        for y in np.linspace(bounds[2], bounds[3], 20):
            for z in np.linspace(bounds[4], bounds[5], 20):
                point = np.array([x, y, z], dtype=np.float32)
                block_id = find_block_containing_point(point, blocks)

                assert block_id >= 0, f"Point {point} not in any block"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
