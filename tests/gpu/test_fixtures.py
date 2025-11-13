"""
Test that fixtures are working correctly.

Phase 0.3 - Infrastructure validation
"""

import numpy as np
import pytest


def test_single_tetrahedron_fixture(single_tetrahedron):
    """Verify single tetrahedron fixture."""
    positions, connectivity = single_tetrahedron

    assert positions.shape == (4, 3)
    assert connectivity.shape == (1, 4)
    assert connectivity.dtype == np.int32
    assert positions.dtype == np.float64


def test_two_tetrahedra_fixture(two_tetrahedra):
    """Verify two tetrahedra fixture."""
    positions, connectivity = two_tetrahedra

    assert positions.shape == (5, 3)
    assert connectivity.shape == (2, 4)

    # Verify they share face (0,1,2)
    face1 = set(connectivity[0, :3])
    face2 = set(connectivity[1, :3])
    assert len(face1.intersection(face2)) == 3


def test_tiny_mesh_fixture(tiny_mesh):
    """Verify tiny mesh fixture."""
    positions, connectivity = tiny_mesh

    assert len(positions) > 0
    assert len(connectivity) > 0
    assert connectivity.max() < len(positions)


def test_small_mesh_fixture(small_mesh):
    """Verify small mesh fixture."""
    positions, connectivity = small_mesh

    assert len(connectivity) > 1000  # Should be ~6K
    assert len(connectivity) < 10000


def test_rotation_field_fixture(rotation_field, tiny_mesh):
    """Verify rotation field fixture."""
    positions, _ = tiny_mesh

    assert rotation_field.shape == positions.shape
    assert rotation_field.dtype == np.float64


def test_particles_grid_fixture(particles_grid):
    """Verify grid particles fixture."""
    assert particles_grid.shape == (1000, 3)
    assert np.all(particles_grid >= 0.0)
    assert np.all(particles_grid <= 1.0)


def test_particles_random_fixture(particles_random):
    """Verify random particles fixture."""
    assert particles_random.shape == (1000, 3)
    assert np.all(particles_random >= 0.0)
    assert np.all(particles_random <= 1.0)


def test_single_particle_inside(single_particle_inside):
    """Verify inside particle fixture."""
    assert single_particle_inside.shape == (1, 3)


def test_single_particle_outside(single_particle_outside):
    """Verify outside particle fixture."""
    assert single_particle_outside.shape == (1, 3)


def test_default_gpu_config(default_gpu_config):
    """Verify GPU config fixture."""
    assert default_gpu_config.field_storage == "nodes"
    assert default_gpu_config.max_neighbors == 4


def test_jax_imports():
    """Verify JAX is available."""
    import jax
    import jax.numpy as jnp

    # Simple JAX operation
    x = jnp.array([1, 2, 3])
    y = jnp.sum(x)

    assert y == 6
