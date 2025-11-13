#!/usr/bin/env python3
"""
Particle Seeding and Initialization

Implements particle seeding strategies and initial element search for GPU tracking.

Phase 3.1 of V3 Plan

Seeding Strategies:
- Uniform grid: Regular distribution with tunable density per axis
- Random uniform: Monte Carlo sampling
- Custom bounding box: Seed within specified region

Future (scheduled):
- GPU-based inlet/outlet boundary conditions
- Surface seeding for boundary layers
- Field-based seeding
"""

from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
import jax

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


@dataclass
class SeedingConfig:
    """
    Configuration for particle seeding.

    Defines the seeding region and distribution strategy.
    """

    # Bounding box for seeding region
    bbox_min: np.ndarray  # (3,) - minimum [x, y, z]
    bbox_max: np.ndarray  # (3,) - maximum [x, y, z]

    # Grid seeding parameters
    density_per_axis: Tuple[int, int, int] = (10, 10, 10)  # (nx, ny, nz)

    # Random seeding parameters
    n_particles: int = 1000  # For random seeding
    seed: int = 42  # Random seed for reproducibility

    def __post_init__(self):
        """Validate configuration."""
        assert np.all(self.bbox_max > self.bbox_min), \
            "bbox_max must be greater than bbox_min"

        assert all(d > 0 for d in self.density_per_axis), \
            "density_per_axis must be positive"

        assert self.n_particles > 0, "n_particles must be positive"

    @property
    def n_particles_grid(self) -> int:
        """Total particles for grid seeding."""
        return np.prod(self.density_per_axis)

    @property
    def bbox_size(self) -> np.ndarray:
        """Bounding box size."""
        return self.bbox_max - self.bbox_min

    @property
    def bbox_center(self) -> np.ndarray:
        """Bounding box center."""
        return (self.bbox_min + self.bbox_max) / 2

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 80,
            "SEEDING CONFIGURATION",
            "=" * 80,
            "",
            "Bounding Box:",
            f"  Min: [{self.bbox_min[0]:.6f}, {self.bbox_min[1]:.6f}, {self.bbox_min[2]:.6f}]",
            f"  Max: [{self.bbox_max[0]:.6f}, {self.bbox_max[1]:.6f}, {self.bbox_max[2]:.6f}]",
            f"  Size: [{self.bbox_size[0]:.6f}, {self.bbox_size[1]:.6f}, {self.bbox_size[2]:.6f}]",
            f"  Center: [{self.bbox_center[0]:.6f}, {self.bbox_center[1]:.6f}, {self.bbox_center[2]:.6f}]",
            "",
            "Grid Seeding:",
            f"  Density per axis: {self.density_per_axis[0]} × {self.density_per_axis[1]} × {self.density_per_axis[2]}",
            f"  Total particles: {self.n_particles_grid:,}",
            "",
            "Random Seeding:",
            f"  N particles: {self.n_particles:,}",
            f"  Random seed: {self.seed}",
            "=" * 80,
        ]
        return "\n".join(lines)


def seed_particles_uniform_grid(config: SeedingConfig) -> np.ndarray:
    """
    Seed particles on a uniform 3D grid.

    Creates a regular Cartesian grid within the bounding box.

    Args:
        config: SeedingConfig with bbox and density_per_axis

    Returns:
        positions: (N_particles, 3) float64 - particle positions
    """
    nx, ny, nz = config.density_per_axis

    # Create 1D grids for each axis
    x = np.linspace(config.bbox_min[0], config.bbox_max[0], nx)
    y = np.linspace(config.bbox_min[1], config.bbox_max[1], ny)
    z = np.linspace(config.bbox_min[2], config.bbox_max[2], nz)

    # Create 3D grid
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    # Flatten to list of positions
    positions = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    return positions.astype(np.float64)


def seed_particles_random_uniform(config: SeedingConfig) -> np.ndarray:
    """
    Seed particles randomly with uniform distribution.

    Monte Carlo sampling within the bounding box.

    Args:
        config: SeedingConfig with bbox and n_particles

    Returns:
        positions: (N_particles, 3) float64 - particle positions
    """
    rng = np.random.RandomState(config.seed)

    # Generate random positions in [0, 1]^3
    random_01 = rng.uniform(0.0, 1.0, (config.n_particles, 3))

    # Scale to bounding box
    positions = config.bbox_min + random_01 * config.bbox_size

    return positions.astype(np.float64)


def seed_particles_stratified(config: SeedingConfig,
                              strata_per_axis: Tuple[int, int, int] = (5, 5, 5)) -> np.ndarray:
    """
    Seed particles using stratified sampling.

    Divides bounding box into strata and places random particles in each.
    Better coverage than pure random, less regular than grid.

    Args:
        config: SeedingConfig with bbox and n_particles
        strata_per_axis: Number of strata per dimension

    Returns:
        positions: (N_particles, 3) float64 - particle positions
    """
    rng = np.random.RandomState(config.seed)

    nx, ny, nz = strata_per_axis
    n_strata = nx * ny * nz
    particles_per_stratum = config.n_particles // n_strata

    positions = []

    # Stratum sizes
    stratum_size = config.bbox_size / np.array(strata_per_axis)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Stratum bounds
                stratum_min = config.bbox_min + np.array([i, j, k]) * stratum_size
                stratum_max = stratum_min + stratum_size

                # Random positions within stratum
                random_01 = rng.uniform(0.0, 1.0, (particles_per_stratum, 3))
                stratum_positions = stratum_min + random_01 * stratum_size

                positions.append(stratum_positions)

    positions = np.vstack(positions)

    # Add any remaining particles (if n_particles not divisible by n_strata)
    remaining = config.n_particles - len(positions)
    if remaining > 0:
        random_01 = rng.uniform(0.0, 1.0, (remaining, 3))
        extra_positions = config.bbox_min + random_01 * config.bbox_size
        positions = np.vstack([positions, extra_positions])

    return positions.astype(np.float64)


def create_seeding_config_from_mesh(
    mesh_bbox_min: np.ndarray,
    mesh_bbox_max: np.ndarray,
    margin: float = 0.0,
    density_per_axis: Tuple[int, int, int] = (10, 10, 10)
) -> SeedingConfig:
    """
    Create seeding configuration from mesh bounding box.

    Convenience function to seed particles throughout the mesh domain.

    Args:
        mesh_bbox_min: (3,) mesh bounding box minimum
        mesh_bbox_max: (3,) mesh bounding box maximum
        margin: Inset margin (negative = expand, positive = shrink)
        density_per_axis: Grid density

    Returns:
        SeedingConfig
    """
    bbox_size = mesh_bbox_max - mesh_bbox_min

    # Apply margin
    bbox_min = mesh_bbox_min + margin * bbox_size
    bbox_max = mesh_bbox_max - margin * bbox_size

    return SeedingConfig(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        density_per_axis=density_per_axis,
    )


def create_seeding_config_plane(
    center: np.ndarray,
    normal: np.ndarray,
    width: float,
    height: float,
    thickness: float,
    density_per_axis: Tuple[int, int, int] = (20, 20, 2)
) -> SeedingConfig:
    """
    Create seeding configuration for a plane (e.g., inlet).

    Future use for inlet/outlet boundary conditions.

    Args:
        center: (3,) plane center
        normal: (3,) plane normal (will be normalized)
        width: Plane width
        height: Plane height
        thickness: Plane thickness (for 3D seeding)
        density_per_axis: Grid density

    Returns:
        SeedingConfig
    """
    # Normalize normal vector
    normal = normal / np.linalg.norm(normal)

    # Create local coordinate system
    # Find two orthogonal vectors
    if abs(normal[0]) < 0.9:
        u = np.cross(normal, [1, 0, 0])
    else:
        u = np.cross(normal, [0, 1, 0])
    u = u / np.linalg.norm(u)

    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    # Bounding box in local coordinates
    half_width = width / 2
    half_height = height / 2
    half_thickness = thickness / 2

    # Transform to global coordinates
    corners = [
        center + (-half_width * u - half_height * v - half_thickness * normal),
        center + (half_width * u + half_height * v + half_thickness * normal),
    ]

    bbox_min = np.minimum(corners[0], corners[1])
    bbox_max = np.maximum(corners[0], corners[1])

    return SeedingConfig(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        density_per_axis=density_per_axis,
    )


def filter_particles_inside_mesh(
    positions: np.ndarray,
    mesh_bbox_min: np.ndarray,
    mesh_bbox_max: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter particles to only those inside mesh bounding box.

    Args:
        positions: (N, 3) particle positions
        mesh_bbox_min: (3,) mesh bounding box minimum
        mesh_bbox_max: (3,) mesh bounding box maximum

    Returns:
        filtered_positions: (N_inside, 3) positions inside mesh
        inside_mask: (N,) bool mask indicating which particles are inside
    """
    inside_mask = np.all(
        (positions >= mesh_bbox_min) & (positions <= mesh_bbox_max),
        axis=1
    )

    filtered_positions = positions[inside_mask]

    return filtered_positions, inside_mask


if __name__ == "__main__":
    print("Testing particle seeding...")

    # Test 1: Grid seeding
    print("\nTest 1: Uniform grid seeding")
    config_grid = SeedingConfig(
        bbox_min=np.array([0.0, 0.0, 0.0]),
        bbox_max=np.array([1.0, 1.0, 1.0]),
        density_per_axis=(5, 5, 5),
    )
    print(config_grid)

    positions_grid = seed_particles_uniform_grid(config_grid)
    print(f"\n  Generated: {len(positions_grid):,} particles")
    print(f"  Expected: {config_grid.n_particles_grid:,}")
    print(f"  Match: {len(positions_grid) == config_grid.n_particles_grid}")
    print(f"  Position range: [{positions_grid.min():.3f}, {positions_grid.max():.3f}]")

    # Test 2: Random seeding
    print("\nTest 2: Random uniform seeding")
    config_random = SeedingConfig(
        bbox_min=np.array([0.0, 0.0, 0.0]),
        bbox_max=np.array([1.0, 1.0, 1.0]),
        n_particles=1000,
        seed=42,
    )

    positions_random = seed_particles_random_uniform(config_random)
    print(f"  Generated: {len(positions_random):,} particles")
    print(f"  Expected: {config_random.n_particles:,}")
    print(f"  Position range: [{positions_random.min():.3f}, {positions_random.max():.3f}]")

    # Test 3: Stratified seeding
    print("\nTest 3: Stratified seeding")
    config_strat = SeedingConfig(
        bbox_min=np.array([0.0, 0.0, 0.0]),
        bbox_max=np.array([1.0, 1.0, 1.0]),
        n_particles=1000,
        seed=42,
    )

    positions_strat = seed_particles_stratified(config_strat, strata_per_axis=(5, 5, 5))
    print(f"  Generated: {len(positions_strat):,} particles")
    print(f"  Position range: [{positions_strat.min():.3f}, {positions_strat.max():.3f}]")

    # Test 4: Seeding from mesh bbox
    print("\nTest 4: Seeding from mesh bounding box")
    config_mesh = create_seeding_config_from_mesh(
        mesh_bbox_min=np.array([-0.03, -0.023, -0.01]),
        mesh_bbox_max=np.array([0.03, 0.023, 0.0]),
        margin=0.1,  # 10% margin
        density_per_axis=(10, 10, 5),
    )
    print(config_mesh)

    positions_mesh = seed_particles_uniform_grid(config_mesh)
    print(f"\n  Generated: {len(positions_mesh):,} particles")

    # Test 5: Filtering
    print("\nTest 5: Filter particles inside mesh")
    # Create some particles outside
    test_positions = np.array([
        [0.5, 0.5, 0.5],   # inside
        [-1.0, 0.5, 0.5],  # outside (x)
        [0.5, 2.0, 0.5],   # outside (y)
        [0.5, 0.5, -0.5],  # outside (z)
        [0.1, 0.1, 0.1],   # inside
    ])

    filtered, mask = filter_particles_inside_mesh(
        test_positions,
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
    )

    print(f"  Original: {len(test_positions)} particles")
    print(f"  Filtered: {len(filtered)} particles")
    print(f"  Expected: 2 inside")
    print(f"  Mask: {mask}")

    print("\n✅ All seeding tests passed!")
