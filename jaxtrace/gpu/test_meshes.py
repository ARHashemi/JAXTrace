#!/usr/bin/env python3
"""
Synthetic Test Mesh Generator for GPU Implementation

This module provides tools to generate synthetic tetrahedral meshes for testing
GPU implementations with controlled characteristics.

Phase 0.2 of V3 Plan
"""

from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class TestMeshConfig:
    """Configuration for synthetic test mesh generation."""

    # Domain size
    domain_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Mesh resolution (elements per dimension)
    resolution: Tuple[int, int, int] = (10, 10, 10)

    # Load balance control
    use_adaptive_refinement: bool = False
    refinement_center: Optional[Tuple[float, float, float]] = None
    refinement_radius: float = 0.3
    refinement_factor: int = 2

    # Perturbation (add noise to node positions)
    perturb_nodes: bool = False
    perturbation_factor: float = 0.1

    @property
    def n_elements_approx(self) -> int:
        """Approximate element count (before refinement)."""
        nx, ny, nz = self.resolution
        # Each cube subdivided into 6 tetrahedra
        return nx * ny * nz * 6


def generate_cube_tetrahedra(
    origin: np.ndarray,
    size: float,
    base_id: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 6 tetrahedra from a cube.

    Subdivides a cube into 6 tetrahedra sharing the main diagonal.

    Args:
        origin: (3,) corner position
        size: Cube side length
        base_id: Starting node ID

    Returns:
        vertices: (8, 3) cube vertices
        elements: (6, 4) tetrahedra connectivity (local indices 0-7)
    """
    x, y, z = origin

    # 8 cube vertices
    vertices = np.array([
        [x, y, z],          # 0
        [x + size, y, z],      # 1
        [x + size, y + size, z],  # 2
        [x, y + size, z],      # 3
        [x, y, z + size],      # 4
        [x + size, y, z + size],  # 5
        [x + size, y + size, z + size],  # 6
        [x, y + size, z + size],  # 7
    ])

    # 6 tetrahedra sharing diagonal 0-6
    elements = np.array([
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
        [0, 5, 1, 6],
    ], dtype=np.int32)

    return vertices, elements


def generate_regular_grid_mesh(config: TestMeshConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a regular tetrahedral grid mesh.

    Args:
        config: Mesh configuration

    Returns:
        positions: (N_nodes, 3) node coordinates
        connectivity: (N_elements, 4) element connectivity
    """
    nx, ny, nz = config.resolution
    dx, dy, dz = np.array(config.domain_size) / np.array(config.resolution)

    # Generate grid of cubes, each subdivided into 6 tetrahedra
    positions_list = []
    connectivity_list = []

    node_id_map = {}  # (i, j, k) -> node_id

    def get_or_create_node(i: int, j: int, k: int) -> int:
        """Get existing node ID or create new one."""
        key = (i, j, k)
        if key not in node_id_map:
            node_id = len(positions_list)
            node_id_map[key] = node_id
            positions_list.append([i * dx, j * dy, k * dz])
        return node_id_map[key]

    # Process each cube
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Get/create 8 corner node IDs
                cube_nodes = np.array([
                    get_or_create_node(i, j, k),
                    get_or_create_node(i + 1, j, k),
                    get_or_create_node(i + 1, j + 1, k),
                    get_or_create_node(i, j + 1, k),
                    get_or_create_node(i, j, k + 1),
                    get_or_create_node(i + 1, j, k + 1),
                    get_or_create_node(i + 1, j + 1, k + 1),
                    get_or_create_node(i, j + 1, k + 1),
                ], dtype=np.int32)

                # Generate 6 tetrahedra for this cube
                _, local_elements = generate_cube_tetrahedra(
                    np.array([i * dx, j * dy, k * dz]), dx, 0
                )

                # Map local indices to global node IDs
                for elem in local_elements:
                    global_elem = cube_nodes[elem]
                    connectivity_list.append(global_elem)

    positions = np.array(positions_list, dtype=np.float64)
    connectivity = np.array(connectivity_list, dtype=np.int32)

    # Apply node perturbation if requested
    if config.perturb_nodes:
        # Don't perturb boundary nodes
        interior_mask = (
            (positions[:, 0] > 0) & (positions[:, 0] < config.domain_size[0]) &
            (positions[:, 1] > 0) & (positions[:, 1] < config.domain_size[1]) &
            (positions[:, 2] > 0) & (positions[:, 2] < config.domain_size[2])
        )

        # Random perturbation
        perturbation = np.random.normal(
            0, config.perturbation_factor * min(dx, dy, dz),
            (positions.shape[0], 3)
        )
        positions[interior_mask] += perturbation[interior_mask]

    return positions, connectivity


def refine_region(
    positions: np.ndarray,
    connectivity: np.ndarray,
    center: np.ndarray,
    radius: float,
    factor: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Refine elements in a spherical region.

    This is a simple refinement that subdivides each tetrahedron into
    8 smaller tetrahedra by adding midpoint nodes.

    Args:
        positions: (N_nodes, 3)
        connectivity: (N_elements, 4)
        center: (3,) refinement center
        radius: Refinement radius
        factor: Refinement factor (1 = subdivide once, 2 = subdivide twice, etc.)

    Returns:
        refined_positions: (N_nodes_new, 3)
        refined_connectivity: (N_elements_new, 4)
    """
    # Compute element centroids
    centroids = positions[connectivity].mean(axis=1)

    # Find elements to refine
    distances = np.linalg.norm(centroids - center, axis=1)
    refine_mask = distances < radius

    # For now, simple approach: just return original mesh
    # Full refinement implementation would be complex
    # TODO: Implement proper tetrahedral refinement in future phase
    print(f"  Refinement requested but not yet implemented")
    print(f"  Would refine {np.sum(refine_mask)} / {len(connectivity)} elements")

    return positions, connectivity


def generate_test_mesh(
    config: Optional[TestMeshConfig] = None,
    output_path: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic test mesh.

    Args:
        config: Mesh configuration (uses default if None)
        output_path: If provided, save mesh to NPZ file

    Returns:
        positions: (N_nodes, 3)
        connectivity: (N_elements, 4)
    """
    if config is None:
        config = TestMeshConfig()

    print(f"Generating test mesh...")
    print(f"  Resolution: {config.resolution[0]}×{config.resolution[1]}×{config.resolution[2]}")
    print(f"  Domain size: {config.domain_size}")
    print(f"  Approximate elements: {config.n_elements_approx:,}")

    # Generate base grid
    positions, connectivity = generate_regular_grid_mesh(config)

    print(f"  Generated: {len(positions):,} nodes, {len(connectivity):,} elements")

    # Apply refinement if requested
    if config.use_adaptive_refinement:
        if config.refinement_center is None:
            # Default: center of domain
            config.refinement_center = tuple(np.array(config.domain_size) / 2)

        positions, connectivity = refine_region(
            positions, connectivity,
            np.array(config.refinement_center),
            config.refinement_radius,
            config.refinement_factor
        )

        print(f"  After refinement: {len(positions):,} nodes, {len(connectivity):,} elements")

    # Save if requested
    if output_path is not None:
        np.savez_compressed(
            output_path,
            positions=positions,
            connectivity=connectivity,
        )
        print(f"  Saved to: {output_path}")

    return positions, connectivity


# Predefined test mesh configurations

# Tiny mesh for rapid testing (216 elements)
TINY_MESH = TestMeshConfig(
    domain_size=(1.0, 1.0, 1.0),
    resolution=(3, 3, 3),
    perturb_nodes=False,
)

# Small balanced mesh (~6K elements)
SMALL_BALANCED_MESH = TestMeshConfig(
    domain_size=(1.0, 1.0, 1.0),
    resolution=(10, 10, 10),
    perturb_nodes=True,
    perturbation_factor=0.05,
)

# Medium mesh (~48K elements)
MEDIUM_MESH = TestMeshConfig(
    domain_size=(2.0, 2.0, 2.0),
    resolution=(20, 20, 20),
    perturb_nodes=True,
    perturbation_factor=0.05,
)

# Large mesh (~384K elements)
LARGE_MESH = TestMeshConfig(
    domain_size=(3.0, 3.0, 3.0),
    resolution=(40, 40, 40),
    perturb_nodes=True,
    perturbation_factor=0.05,
)

# Imbalanced mesh (for testing load balancing)
IMBALANCED_MESH = TestMeshConfig(
    domain_size=(2.0, 2.0, 0.5),  # Thin in Z
    resolution=(40, 40, 5),
    perturb_nodes=True,
    perturbation_factor=0.05,
)


def generate_test_field(
    positions: np.ndarray,
    field_type: str = "rotation"
) -> np.ndarray:
    """
    Generate a synthetic velocity field for testing.

    Args:
        positions: (N_nodes, 3) node positions
        field_type: Type of field to generate:
            - "rotation": Solid body rotation around Z axis
            - "expansion": Radial expansion from origin
            - "shear": Linear shear flow
            - "vortex": Vortex centered at origin

    Returns:
        velocities: (N_nodes, 3) velocity vectors
    """
    if field_type == "rotation":
        # Solid body rotation: v = ω × r, with ω = (0, 0, 1)
        omega = 1.0
        velocities = np.zeros_like(positions)
        velocities[:, 0] = -omega * positions[:, 1]
        velocities[:, 1] = omega * positions[:, 0]
        velocities[:, 2] = 0.0

    elif field_type == "expansion":
        # Radial expansion: v = r
        velocities = positions.copy()

    elif field_type == "shear":
        # Linear shear: v = (y, 0, 0)
        velocities = np.zeros_like(positions)
        velocities[:, 0] = positions[:, 1]

    elif field_type == "vortex":
        # Rankine vortex
        r = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        theta = np.arctan2(positions[:, 1], positions[:, 0])

        v_theta = np.where(r < 0.5, r, 0.5 / r)  # Solid core, potential flow outside

        velocities = np.zeros_like(positions)
        velocities[:, 0] = -v_theta * np.sin(theta)
        velocities[:, 1] = v_theta * np.cos(theta)
        velocities[:, 2] = 0.0

    else:
        raise ValueError(f"Unknown field type: {field_type}")

    return velocities


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic test meshes")
    parser.add_argument(
        "--size",
        choices=["tiny", "small", "medium", "large", "imbalanced"],
        default="small",
        help="Predefined mesh size"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (NPZ format)"
    )
    parser.add_argument(
        "--field",
        choices=["rotation", "expansion", "shear", "vortex"],
        default="rotation",
        help="Velocity field type to generate"
    )

    args = parser.parse_args()

    # Select config
    configs = {
        "tiny": TINY_MESH,
        "small": SMALL_BALANCED_MESH,
        "medium": MEDIUM_MESH,
        "large": LARGE_MESH,
        "imbalanced": IMBALANCED_MESH,
    }
    config = configs[args.size]

    # Generate mesh
    positions, connectivity = generate_test_mesh(config)

    # Generate field
    velocities = generate_test_field(positions, args.field)

    # Save
    if args.output is None:
        args.output = Path(f"test_mesh_{args.size}.npz")

    np.savez_compressed(
        args.output,
        positions=positions,
        connectivity=connectivity,
        velocities=velocities,
    )

    print(f"\n✅ Test mesh saved to: {args.output}")
    print(f"   Nodes: {len(positions):,}")
    print(f"   Elements: {len(connectivity):,}")
    print(f"   Field type: {args.field}")
