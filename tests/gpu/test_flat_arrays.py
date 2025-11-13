"""
Tests for flat array data structures and mesh loading.

Phase 1.3 - Validation tests
"""

import numpy as np
import jax.numpy as jnp
import pytest

from jaxtrace.gpu.flat_arrays import (
    MeshData,
    ParticleData,
    BlockPartitionData,
    create_mesh_data,
    create_particle_data,
)
from jaxtrace.gpu.mesh_loader import (
    build_element_neighbors,
    assign_elements_to_blocks,
    load_mesh_complete,
)


class TestMeshData:
    """Test MeshData structure."""

    def test_create_single_tet(self, single_tetrahedron):
        """Test creating MeshData from single tetrahedron."""
        positions, connectivity = single_tetrahedron

        mesh_data = create_mesh_data(positions, connectivity, device="cpu")

        assert mesh_data.n_nodes == 4
        assert mesh_data.n_elements == 1
        assert mesh_data.positions.shape == (4, 3)
        assert mesh_data.connectivity.shape == (1, 4)

    def test_memory_usage(self, small_mesh):
        """Test memory usage computation."""
        positions, connectivity = small_mesh

        mesh_data = create_mesh_data(positions, connectivity, device="cpu")
        mem = mesh_data.memory_usage_mb()

        assert 'positions' in mem
        assert 'connectivity' in mem
        assert 'total' in mem
        assert mem['total'] > 0

    def test_with_neighbors(self, two_tetrahedra):
        """Test with element neighbors."""
        positions, connectivity = two_tetrahedra

        # Build neighbors
        element_neighbors = build_element_neighbors(connectivity, verbose=False)

        mesh_data = create_mesh_data(
            positions, connectivity, element_neighbors=element_neighbors, device="cpu"
        )

        assert mesh_data.element_neighbors is not None
        assert mesh_data.element_neighbors.shape == (2, 4)

        # Elements 0 and 1 should be neighbors (share face 0,1,2)
        assert 1 in mesh_data.element_neighbors[0]
        assert 0 in mesh_data.element_neighbors[1]

    def test_with_block_ids(self, small_mesh):
        """Test with block IDs."""
        positions, connectivity = small_mesh

        element_block_IDs, _ = assign_elements_to_blocks(
            positions, connectivity, (2, 2, 2), verbose=False
        )

        mesh_data = create_mesh_data(
            positions, connectivity, element_block_IDs=element_block_IDs, device="cpu"
        )

        assert mesh_data.element_block_IDs is not None
        assert mesh_data.n_blocks > 0
        assert jnp.all(mesh_data.element_block_IDs >= 0)
        assert jnp.all(mesh_data.element_block_IDs < mesh_data.n_blocks)


class TestParticleData:
    """Test ParticleData structure."""

    def test_create_single_particle(self, single_particle_inside):
        """Test creating ParticleData from single particle."""
        particle_data = create_particle_data(single_particle_inside, device="cpu")

        assert particle_data.n_particles == 1
        assert particle_data.positions.shape == (1, 3)
        assert particle_data.n_active() == 1

    def test_create_grid_particles(self, particles_grid):
        """Test with grid of particles."""
        particle_data = create_particle_data(particles_grid, device="cpu")

        assert particle_data.n_particles == 1000
        assert particle_data.n_active() == 1000

    def test_inactive_particles(self, particles_random):
        """Test with some inactive particles."""
        n = len(particles_random)
        active = np.ones(n, dtype=bool)
        active[:100] = False  # First 100 inactive

        particle_data = create_particle_data(
            particles_random, active=active, device="cpu"
        )

        assert particle_data.n_particles == n
        assert particle_data.n_active() == n - 100

    def test_with_element_ids(self, particles_grid):
        """Test with element IDs."""
        n = len(particles_grid)
        element_IDs = np.random.randint(0, 100, size=n, dtype=np.int32)

        particle_data = create_particle_data(
            particles_grid, element_IDs=element_IDs, device="cpu"
        )

        assert particle_data.element_IDs is not None
        assert jnp.all(particle_data.element_IDs >= 0)

    def test_memory_usage(self, particles_grid):
        """Test memory usage for particles."""
        particle_data = create_particle_data(particles_grid, device="cpu")
        mem = particle_data.memory_usage_mb()

        assert mem['total'] > 0

        # Should be approximately 28 bytes per particle
        # positions: 24 bytes, element_IDs: 4 bytes, active: 1 byte
        expected_kb_per_particle = 29 / 1024  # 29 bytes in KB
        actual_kb_per_particle = mem['total'] * 1024 / 1000

        # Allow 50% tolerance due to array overhead
        assert abs(actual_kb_per_particle - expected_kb_per_particle) < 0.02


class TestElementNeighbors:
    """Test element neighbor computation."""

    def test_single_tet_no_neighbors(self, single_tetrahedron):
        """Single tetrahedron should have no neighbors."""
        _, connectivity = single_tetrahedron

        neighbors = build_element_neighbors(connectivity, verbose=False)

        assert neighbors.shape == (1, 4)
        assert np.all(neighbors == -1)  # All boundary faces

    def test_two_tets_shared_face(self, two_tetrahedra):
        """Two tetrahedra sharing a face."""
        _, connectivity = two_tetrahedra

        neighbors = build_element_neighbors(connectivity, verbose=False)

        assert neighbors.shape == (2, 4)

        # Each should have the other as neighbor (one shared face)
        assert np.sum(neighbors[0] == 1) == 1
        assert np.sum(neighbors[1] == 0) == 1

        # Other faces are boundaries
        assert np.sum(neighbors[0] == -1) == 3
        assert np.sum(neighbors[1] == -1) == 3

    def test_tiny_mesh_connectivity(self, tiny_mesh):
        """Test on tiny mesh (162 elements)."""
        _, connectivity = tiny_mesh

        neighbors = build_element_neighbors(connectivity, verbose=False)

        assert neighbors.shape == (162, 4)

        # Should have mix of interior and boundary
        n_interior_faces = np.sum(neighbors >= 0)
        n_boundary_faces = np.sum(neighbors == -1)

        assert n_interior_faces > 0
        assert n_boundary_faces > 0

        # Boundary elements should have at least one -1
        n_boundary_elements = np.sum(np.any(neighbors == -1, axis=1))
        assert n_boundary_elements > 0

    def test_neighbor_symmetry(self, small_mesh):
        """Test neighbor relationship is symmetric."""
        _, connectivity = small_mesh

        neighbors = build_element_neighbors(connectivity, verbose=False)

        # For each element-neighbor pair, the reverse should exist
        for elem_id in range(len(connectivity)):
            for face_idx in range(4):
                neighbor_id = neighbors[elem_id, face_idx]
                if neighbor_id >= 0:
                    # neighbor_id should list elem_id as a neighbor
                    assert elem_id in neighbors[neighbor_id]


class TestBlockAssignment:
    """Test block assignment."""

    def test_single_tet_one_block(self, single_tetrahedron):
        """Single tet in single block."""
        positions, connectivity = single_tetrahedron

        block_IDs, partition_data = assign_elements_to_blocks(
            positions, connectivity, (1, 1, 1), verbose=False
        )

        assert block_IDs.shape == (1,)
        assert block_IDs[0] == 0
        assert partition_data.n_blocks == 1
        assert partition_data.elements_per_block[0] == 1

    def test_tiny_mesh_four_blocks(self, tiny_mesh):
        """Tiny mesh split into 4 blocks."""
        positions, connectivity = tiny_mesh

        block_IDs, partition_data = assign_elements_to_blocks(
            positions, connectivity, (2, 2, 1), verbose=False
        )

        assert block_IDs.shape == (162,)
        assert partition_data.n_blocks == 4

        # All blocks should be used
        assert len(np.unique(block_IDs)) == 4

        # Check counts
        for block_id in range(4):
            count = np.sum(block_IDs == block_id)
            assert count == partition_data.elements_per_block[block_id]

    def test_load_imbalance(self, small_mesh):
        """Test load imbalance computation."""
        positions, connectivity = small_mesh

        _, partition_data = assign_elements_to_blocks(
            positions, connectivity, (2, 2, 2), verbose=False
        )

        imbalance = partition_data.load_imbalance_factor()

        # For balanced mesh, should be close to 1
        assert imbalance >= 1.0
        assert imbalance < 2.0  # Should be reasonably balanced

    def test_bounding_box(self, small_mesh):
        """Test bounding box computation."""
        positions, connectivity = small_mesh

        _, partition_data = assign_elements_to_blocks(
            positions, connectivity, (2, 2, 2), verbose=False
        )

        bbox_min = partition_data.bbox_min
        bbox_max = partition_data.bbox_max

        # Should match mesh bounds
        assert np.allclose(bbox_min, positions.min(axis=0))
        assert np.allclose(bbox_max, positions.max(axis=0))


class TestMeshLoader:
    """Test complete mesh loading pipeline."""

    def test_load_tiny_mesh_complete(self, tiny_mesh):
        """Test loading tiny mesh through test_meshes."""
        positions, connectivity = tiny_mesh

        # Build manually
        element_neighbors = build_element_neighbors(connectivity, verbose=False)
        element_block_IDs, partition_data = assign_elements_to_blocks(
            positions, connectivity, (2, 2, 1), verbose=False
        )

        mesh_data = create_mesh_data(
            positions,
            connectivity,
            element_neighbors,
            element_block_IDs,
            device="cpu",
        )

        # Validate
        assert mesh_data.n_nodes == 64
        assert mesh_data.n_elements == 162
        assert mesh_data.n_blocks == 4
        assert partition_data.load_imbalance_factor() < 2.0

    def test_load_threadeda_if_available(self, threadeda_mesh_path):
        """Test loading ThreadedA mesh (if available)."""
        if threadeda_mesh_path is None:
            pytest.skip("ThreadedA mesh not available")

        mesh_data, partition_data = load_mesh_complete(
            threadeda_mesh_path,
            grid_size=(2, 2, 1),
            device="cpu",
            verbose=False,
        )

        # Should match known values
        assert mesh_data.n_nodes == 898_502
        assert mesh_data.n_elements == 3_494_800
        assert mesh_data.n_blocks == 4

        # Memory should be ~140 MB
        mem = mesh_data.memory_usage_mb()
        assert 130 < mem['total'] < 150

        # Load balance should be good for 2×2×1
        assert partition_data.load_imbalance_factor() < 1.5


class TestDevicePlacement:
    """Test CPU/GPU device placement."""

    def test_cpu_device(self, single_tetrahedron):
        """Test placing on CPU."""
        positions, connectivity = single_tetrahedron

        mesh_data = create_mesh_data(positions, connectivity, device="cpu")

        # Should be JAX arrays
        assert isinstance(mesh_data.positions, jnp.ndarray)
        assert isinstance(mesh_data.connectivity, jnp.ndarray)

    def test_gpu_device_if_available(self, single_tetrahedron, jax_gpu_available):
        """Test placing on GPU (if available)."""
        if not jax_gpu_available:
            pytest.skip("GPU not available")

        positions, connectivity = single_tetrahedron

        mesh_data = create_mesh_data(positions, connectivity, device="gpu")

        # Should be JAX arrays on GPU
        assert isinstance(mesh_data.positions, jnp.ndarray)
        assert isinstance(mesh_data.connectivity, jnp.ndarray)

        # Check device (JAX arrays track their device)
        import jax
        # Device should be GPU
        assert "CudaDevice" in str(mesh_data.positions.devices())


class TestMemoryEstimates:
    """Test memory usage matches estimates."""

    def test_threadeda_memory(self, threadeda_mesh_path):
        """Verify ThreadedA memory matches Phase 0 estimate."""
        if threadeda_mesh_path is None:
            pytest.skip("ThreadedA mesh not available")

        mesh_data, _ = load_mesh_complete(
            threadeda_mesh_path,
            grid_size=(2, 2, 1),
            device="cpu",
            verbose=False,
        )

        mem = mesh_data.memory_usage_mb()

        # Phase 0 estimated 140.5 MB
        expected = 140.5
        tolerance = 5.0  # ±5 MB

        assert abs(mem['total'] - expected) < tolerance

    def test_particle_memory_1m(self):
        """Test 1M particles memory usage."""
        n_particles = 1_000_000
        positions = np.random.uniform(0, 1, (n_particles, 3))

        particle_data = create_particle_data(positions, device="cpu")
        mem = particle_data.memory_usage_mb()

        # Phase 0 estimated 27.7 MB for 1M particles
        expected = 27.7
        tolerance = 2.0  # ±2 MB

        assert abs(mem['total'] - expected) < tolerance
