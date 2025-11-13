"""
Pytest fixtures for GPU tests

Provides common test fixtures for GPU implementation testing.
Phase 0.3 of V3 Plan
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Tuple

# Import test mesh generators
from jaxtrace.gpu.test_meshes import (
    generate_test_mesh,
    generate_test_field,
    TINY_MESH,
    SMALL_BALANCED_MESH,
    MEDIUM_MESH,
    TestMeshConfig,
)


# ============================================================================
# Test Mesh Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def tiny_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """
    Tiny test mesh (162 elements) for rapid testing.

    Returns:
        positions: (N_nodes, 3)
        connectivity: (N_elements, 4)
    """
    positions, connectivity = generate_test_mesh(TINY_MESH)
    return positions, connectivity


@pytest.fixture(scope="session")
def small_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """
    Small balanced mesh (~6K elements) for basic tests.

    Returns:
        positions: (N_nodes, 3)
        connectivity: (N_elements, 4)
    """
    positions, connectivity = generate_test_mesh(SMALL_BALANCED_MESH)
    return positions, connectivity


@pytest.fixture(scope="session")
def medium_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """
    Medium mesh (~48K elements) for performance tests.

    Returns:
        positions: (N_nodes, 3)
        connectivity: (N_elements, 4)
    """
    positions, connectivity = generate_test_mesh(MEDIUM_MESH)
    return positions, connectivity


@pytest.fixture(scope="session")
def single_tetrahedron() -> Tuple[np.ndarray, np.ndarray]:
    """
    Single tetrahedron for unit tests.

    Returns:
        positions: (4, 3) - unit tetrahedron
        connectivity: (1, 4)
    """
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    connectivity = np.array([[0, 1, 2, 3]], dtype=np.int32)

    return positions, connectivity


@pytest.fixture(scope="session")
def two_tetrahedra() -> Tuple[np.ndarray, np.ndarray]:
    """
    Two neighboring tetrahedra for connectivity tests.

    Returns:
        positions: (5, 3)
        connectivity: (2, 4)
    """
    positions = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [0.0, 0.0, 1.0],  # 3
        [0.0, 0.0, -1.0],  # 4 (forms second tet with 0,1,2)
    ], dtype=np.float64)

    # Two tetrahedra sharing face (0,1,2)
    connectivity = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 4],
    ], dtype=np.int32)

    return positions, connectivity


# ============================================================================
# Field Fixtures
# ============================================================================


@pytest.fixture
def rotation_field(tiny_mesh) -> np.ndarray:
    """
    Rotation velocity field on tiny mesh.

    Returns:
        velocities: (N_nodes, 3)
    """
    positions, _ = tiny_mesh
    return generate_test_field(positions, "rotation")


@pytest.fixture
def vortex_field(small_mesh) -> np.ndarray:
    """
    Vortex velocity field on small mesh.

    Returns:
        velocities: (N_nodes, 3)
    """
    positions, _ = small_mesh
    return generate_test_field(positions, "vortex")


# ============================================================================
# Particle Fixtures
# ============================================================================


@pytest.fixture
def single_particle_inside() -> np.ndarray:
    """
    Single particle inside unit tetrahedron.

    Returns:
        positions: (1, 3)
    """
    # Centroid of unit tetrahedron
    return np.array([[0.25, 0.25, 0.25]], dtype=np.float64)


@pytest.fixture
def single_particle_outside() -> np.ndarray:
    """
    Single particle outside unit tetrahedron.

    Returns:
        positions: (1, 3)
    """
    return np.array([[2.0, 2.0, 2.0]], dtype=np.float64)


@pytest.fixture
def particles_grid() -> np.ndarray:
    """
    Grid of particles in unit cube.

    Returns:
        positions: (1000, 3) - 10×10×10 grid
    """
    x = np.linspace(0.1, 0.9, 10)
    y = np.linspace(0.1, 0.9, 10)
    z = np.linspace(0.1, 0.9, 10)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    positions = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    return positions.astype(np.float64)


@pytest.fixture
def particles_random(seed=42) -> np.ndarray:
    """
    Random particles in unit cube.

    Returns:
        positions: (1000, 3)
    """
    rng = np.random.RandomState(seed)
    return rng.uniform(0.0, 1.0, (1000, 3)).astype(np.float64)


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def default_gpu_config():
    """
    Default GPU configuration for testing.

    Returns:
        GPUConfig object (will be defined in Phase 1)
    """
    # Placeholder - will be implemented in Phase 1
    from dataclasses import dataclass

    @dataclass
    class GPUConfig:
        field_storage: str = "nodes"
        octree_storage: str = "padded"
        block_storage: str = "padded"
        max_neighbors: int = 4
        max_elements_per_block: int = 1000
        max_elements_per_octree_node: int = 500

    return GPUConfig()


# ============================================================================
# JAX Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def jax_cpu_device():
    """
    Force JAX to use CPU device.

    Useful for testing JAX code without GPU.
    """
    import jax
    # Force CPU
    jax.config.update('jax_platform_name', 'cpu')
    return jax.devices('cpu')[0]


@pytest.fixture(scope="session")
def jax_gpu_available():
    """
    Check if GPU is available for JAX.

    Returns:
        bool: True if GPU available
    """
    import jax
    try:
        gpu_devices = jax.devices('gpu')
        return len(gpu_devices) > 0
    except RuntimeError:
        return False


@pytest.fixture
def skip_if_no_gpu(jax_gpu_available):
    """
    Skip test if no GPU is available.
    """
    if not jax_gpu_available:
        pytest.skip("GPU not available")


# ============================================================================
# ThreadedA Mesh Fixture (for integration tests)
# ============================================================================


@pytest.fixture(scope="session")
def threadeda_mesh_path():
    """
    Path to ThreadedA mesh (if available).

    Returns:
        Path or None
    """
    path = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule")
    if path.exists():
        return path
    return None


@pytest.fixture(scope="session")
def threadeda_mesh(threadeda_mesh_path):
    """
    Load ThreadedA mesh (if available).

    Returns:
        (positions, connectivity) or None
    """
    if threadeda_mesh_path is None:
        pytest.skip("ThreadedA mesh not available")

    from jaxtrace.gpu.mesh_analysis import load_mesh
    return load_mesh(threadeda_mesh_path)


# ============================================================================
# Comparison Fixtures (CPU vs GPU)
# ============================================================================


@pytest.fixture
def cpu_search_function():
    """
    CPU element search function for comparison.

    Returns:
        Function that searches for containing element
    """
    # Will be implemented when CPU search is available
    def search_cpu(point, positions, connectivity):
        """
        Find containing element for a point (CPU implementation).

        Args:
            point: (3,)
            positions: (N_nodes, 3)
            connectivity: (N_elements, 4)

        Returns:
            element_id: int or -1 if not found
        """
        # Placeholder - use brute force for now
        for elem_id, elem in enumerate(connectivity):
            vertices = positions[elem]
            # Simple bounding box test
            if (point >= vertices.min(axis=0)).all() and (point <= vertices.max(axis=0)).all():
                # More accurate test would use barycentric coords
                return elem_id
        return -1

    return search_cpu


# ============================================================================
# Performance Benchmarking Fixtures
# ============================================================================


@pytest.fixture
def benchmark_config():
    """
    Configuration for performance benchmarks.

    Returns:
        dict with benchmark parameters
    """
    return {
        "n_warmup": 3,
        "n_iterations": 10,
        "particle_counts": [100, 1000, 10000],
    }


# ============================================================================
# Utility Fixtures
# ============================================================================


@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Temporary directory for test outputs.

    Returns:
        Path to temporary directory
    """
    output_dir = tmp_path / "gpu_test_outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture(autouse=True)
def reset_jax_cache():
    """
    Reset JAX JIT cache between tests to avoid pollution.
    """
    import jax
    # Clear compilation cache
    jax.clear_caches()
    yield
    # Cleanup after test
    jax.clear_caches()
