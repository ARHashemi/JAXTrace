"""
Unit tests for particle data structure.

Tests particle creation, manipulation, and partitioning for GPU tracking.
"""

import pytest
import numpy as np
from jaxtrace.gpu.particles import (
    ParticleData,
    partition_particles_by_block,
    print_partition_statistics,
)


class TestParticleData:
    """Test ParticleData dataclass."""

    def test_from_positions_default(self):
        """Test particle creation from positions with default velocities."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ])

        particles = ParticleData.from_positions(positions)

        assert particles.n_particles == 3
        assert particles.n_active == 3
        assert np.allclose(particles.positions, positions)
        assert np.allclose(particles.velocities, 0.0)
        assert np.all(particles.element_ids == -1)
        assert np.all(particles.block_ids == -1)
        assert np.all(particles.active_mask)

    def test_from_positions_with_velocities(self):
        """Test particle creation with provided velocities."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        velocities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        particles = ParticleData.from_positions(positions, velocities)

        assert particles.n_particles == 2
        assert np.allclose(particles.velocities, velocities)

    def test_from_positions_dtype_conversion(self):
        """Test that positions/velocities are converted to float32."""
        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

        particles = ParticleData.from_positions(positions)

        assert particles.positions.dtype == np.float32
        assert particles.velocities.dtype == np.float32

    def test_from_positions_invalid_shape(self):
        """Test error on invalid position shape."""
        positions = np.array([[0.0, 0.0]])  # Missing z

        with pytest.raises(ValueError, match="positions must be"):
            ParticleData.from_positions(positions)

    def test_copy(self):
        """Test deep copy of particle data."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        particles = ParticleData.from_positions(positions)
        particles.element_ids[0] = 10

        copy = particles.copy()

        # Verify copy
        assert np.allclose(copy.positions, particles.positions)
        assert copy.element_ids[0] == 10

        # Verify independence
        copy.element_ids[0] = 20
        assert particles.element_ids[0] == 10

    def test_get_active_particles(self):
        """Test extraction of active particles."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ])
        particles = ParticleData.from_positions(positions)

        # Deactivate second particle
        particles.active_mask[1] = False

        active = particles.get_active_particles()

        assert active.n_particles == 2
        assert active.n_active == 2
        assert np.allclose(active.positions[0], [0.0, 0.0, 0.0])
        assert np.allclose(active.positions[1], [2.0, 2.0, 2.0])

    def test_deactivate_particles(self):
        """Test particle deactivation."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ])
        particles = ParticleData.from_positions(positions)

        # Assign to valid blocks first
        particles.block_ids[:] = [0, 1, 2]

        # Mark second particle as outside domain
        particles.block_ids[1] = -1

        # Deactivate particles outside domain
        particles.deactivate_particles(particles.block_ids == -1)

        assert particles.n_active == 2
        assert not particles.active_mask[1]

    def test_validation_shapes(self):
        """Test validation of array shapes."""
        positions = np.zeros((10, 3), dtype=np.float32)
        velocities = np.zeros((10, 3), dtype=np.float32)
        element_ids = np.full(10, -1, dtype=np.int32)
        block_ids = np.full(10, -1, dtype=np.int32)
        active_mask = np.ones(10, dtype=bool)

        # Valid
        particles = ParticleData(
            positions, velocities, element_ids, block_ids, active_mask
        )
        assert particles.n_particles == 10

        # Invalid velocity shape
        bad_velocities = np.zeros((9, 3), dtype=np.float32)
        with pytest.raises(AssertionError):
            ParticleData(
                positions, bad_velocities, element_ids, block_ids, active_mask
            )

    def test_validation_dtypes(self):
        """Test validation of array dtypes."""
        positions = np.zeros((10, 3), dtype=np.float32)
        velocities = np.zeros((10, 3), dtype=np.float32)
        element_ids = np.full(10, -1, dtype=np.int32)
        block_ids = np.full(10, -1, dtype=np.int32)
        active_mask = np.ones(10, dtype=bool)

        # Invalid element_ids dtype (float instead of int32)
        bad_element_ids = np.full(10, -1.0, dtype=np.float32)
        with pytest.raises(AssertionError):
            ParticleData(
                positions, velocities, bad_element_ids, block_ids, active_mask
            )

    def test_print_statistics(self, capsys):
        """Test statistics printing."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ])
        particles = ParticleData.from_positions(positions)
        particles.element_ids[0] = 10
        particles.block_ids[:2] = [0, 1]

        particles.print_statistics()

        captured = capsys.readouterr()
        assert "Total particles: 3" in captured.out
        assert "Active particles: 3" in captured.out
        assert "Element ID cache" in captured.out
        assert "Block ID cache" in captured.out


class TestParticlePartitioning:
    """Test particle partitioning by block."""

    def test_partition_by_block_simple(self):
        """Test partitioning with simple block assignment."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ])
        particles = ParticleData.from_positions(positions)

        # Assign to blocks
        particles.block_ids[:] = [0, 0, 1, 2]

        partition = partition_particles_by_block(particles, n_blocks=3)

        assert len(partition[0]) == 2
        assert len(partition[1]) == 1
        assert len(partition[2]) == 1
        assert np.array_equal(partition[0], [0, 1])
        assert np.array_equal(partition[1], [2])
        assert np.array_equal(partition[2], [3])

    def test_partition_ignores_inactive(self):
        """Test that partitioning ignores inactive particles."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ])
        particles = ParticleData.from_positions(positions)
        particles.block_ids[:] = [0, 0, 1]
        particles.active_mask[1] = False  # Deactivate second particle

        partition = partition_particles_by_block(particles, n_blocks=2)

        assert len(partition[0]) == 1  # Only first particle
        assert len(partition[1]) == 1
        assert partition[0][0] == 0

    def test_partition_ignores_outside_domain(self):
        """Test that partitioning ignores particles outside domain."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ])
        particles = ParticleData.from_positions(positions)
        particles.block_ids[:] = [0, -1, 1]  # Second particle outside

        partition = partition_particles_by_block(particles, n_blocks=2)

        assert len(partition[0]) == 1
        assert len(partition[1]) == 1

    def test_partition_empty_blocks(self):
        """Test partitioning with empty blocks."""
        positions = np.array([[0.0, 0.0, 0.0]])
        particles = ParticleData.from_positions(positions)
        particles.block_ids[:] = [0]

        partition = partition_particles_by_block(particles, n_blocks=4)

        assert len(partition[0]) == 1
        assert len(partition[1]) == 0
        assert len(partition[2]) == 0
        assert len(partition[3]) == 0

    def test_print_partition_statistics(self, capsys):
        """Test partition statistics printing."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ])
        particles = ParticleData.from_positions(positions)
        particles.block_ids[:] = [0, 0, 1]

        partition = partition_particles_by_block(particles, n_blocks=2)
        print_partition_statistics(partition, n_blocks=2)

        captured = capsys.readouterr()
        assert "Particle Partitioning Statistics" in captured.out
        assert "Total blocks: 2" in captured.out
        assert "Non-empty blocks: 2" in captured.out
        assert "Total particles: 3" in captured.out
