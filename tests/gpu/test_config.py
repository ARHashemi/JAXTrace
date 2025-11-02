"""
Unit tests for GPUForestConfig.
"""

import pytest
import tempfile
import os
from jaxtrace.gpu.config import GPUForestConfig


def test_default_config():
    """Test default configuration values."""
    config = GPUForestConfig()

    assert config.block_grid == (4, 4, 2)
    assert config.n_blocks == 32
    assert config.max_octree_depth == 12
    assert config.field_name == "Displacement"
    assert config.auto_detect_field is True
    assert config.max_particles_per_block == 10000
    assert config.ghost_layer_thickness == 1


def test_custom_config():
    """Test custom configuration."""
    config = GPUForestConfig(
        block_grid=(2, 2, 2),
        max_octree_depth=10,
        field_name="velocity"
    )

    assert config.block_grid == (2, 2, 2)
    assert config.n_blocks == 8
    assert config.max_octree_depth == 10
    assert config.field_name == "velocity"


def test_n_blocks_property():
    """Test n_blocks property calculation."""
    config1 = GPUForestConfig(block_grid=(2, 2, 2))
    assert config1.n_blocks == 8

    config2 = GPUForestConfig(block_grid=(4, 4, 2))
    assert config2.n_blocks == 32

    config3 = GPUForestConfig(block_grid=(8, 8, 4))
    assert config3.n_blocks == 256


def test_validation_block_grid():
    """Test block_grid validation."""
    # Invalid length
    with pytest.raises(ValueError, match="must be \\(nx, ny, nz\\)"):
        GPUForestConfig(block_grid=(4, 4))

    # Zero dimension
    with pytest.raises(ValueError, match="must be > 0"):
        GPUForestConfig(block_grid=(4, 0, 2))

    # Negative dimension
    with pytest.raises(ValueError, match="must be > 0"):
        GPUForestConfig(block_grid=(4, 4, -1))


def test_validation_octree_depth():
    """Test max_octree_depth validation."""
    # Too small
    with pytest.raises(ValueError, match="must be in \\[1, 20\\]"):
        GPUForestConfig(max_octree_depth=0)

    # Too large
    with pytest.raises(ValueError, match="must be in \\[1, 20\\]"):
        GPUForestConfig(max_octree_depth=25)

    # Valid range
    GPUForestConfig(max_octree_depth=1)
    GPUForestConfig(max_octree_depth=20)


def test_validation_max_particles():
    """Test max_particles_per_block validation."""
    with pytest.raises(ValueError, match="must be > 0"):
        GPUForestConfig(max_particles_per_block=0)

    with pytest.raises(ValueError, match="must be > 0"):
        GPUForestConfig(max_particles_per_block=-100)


def test_validation_ghost_thickness():
    """Test ghost_layer_thickness validation."""
    with pytest.raises(ValueError, match="must be >= 0"):
        GPUForestConfig(ghost_layer_thickness=-1)

    # Valid
    GPUForestConfig(ghost_layer_thickness=0)
    GPUForestConfig(ghost_layer_thickness=2)


def test_validation_trajectory_stride():
    """Test trajectory_stride validation."""
    with pytest.raises(ValueError, match="must be >= 1"):
        GPUForestConfig(trajectory_stride=0)

    # Valid
    GPUForestConfig(trajectory_stride=1)
    GPUForestConfig(trajectory_stride=10)


def test_yaml_roundtrip():
    """Test saving and loading from YAML."""
    config_original = GPUForestConfig(
        block_grid=(8, 8, 4),
        max_octree_depth=10,
        field_name="velocity",
        max_particles_per_block=5000,
        revolution_cycle=(120, 159)
    )

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name

    try:
        config_original.to_yaml(temp_path)

        # Load back
        config_loaded = GPUForestConfig.from_yaml(temp_path)

        # Compare
        assert config_loaded.block_grid == config_original.block_grid
        assert config_loaded.max_octree_depth == config_original.max_octree_depth
        assert config_loaded.field_name == config_original.field_name
        assert config_loaded.max_particles_per_block == config_original.max_particles_per_block
        assert config_loaded.revolution_cycle == config_original.revolution_cycle

    finally:
        os.unlink(temp_path)


def test_str_representation():
    """Test string representation."""
    config = GPUForestConfig(block_grid=(4, 4, 2))
    s = str(config)

    assert "4×4×2 = 32 blocks" in s
    assert "Displacement" in s
    assert "10,000" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
