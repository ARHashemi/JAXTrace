#!/usr/bin/env python3
"""
Flat Array Data Structures for GPU Implementation

This module defines the flat array data structures optimized for JAX/GPU
and provides loaders to convert mesh data into these structures.

Phase 1 of V3 Plan

Key Design Principles:
- All arrays are static-size (no dynamic allocation)
- Use indexing instead of filtering
- Pad with -1 for missing/invalid entries
- Node-based field storage (not element-based)
- Minimal memory footprint
"""

from pathlib import Path
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np

# Enable 64-bit precision in JAX (required for accurate particle tracking)
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


@dataclass
class MeshData:
    """
    Flat array representation of tetrahedral mesh.

    All arrays are designed for JAX/GPU with static shapes.
    """

    # Basic mesh topology
    positions: jnp.ndarray          # (N_nodes, 3) float64 - node coordinates
    connectivity: jnp.ndarray       # (N_elements, 4) int32 - element node indices

    # Element adjacency (for Level 1 search)
    element_neighbors: jnp.ndarray  # (N_elements, 4) int32 - neighbor elements
                                     # Padded with -1 for boundary faces

    # Spatial partitioning (for Level 2 search)
    element_block_IDs: jnp.ndarray  # (N_elements,) int32 - which block contains element

    # Optional: Velocity field (node-based storage)
    velocities: Optional[jnp.ndarray] = None  # (N_nodes, 3) float64

    # Metadata
    n_nodes: int = 0
    n_elements: int = 0
    n_blocks: int = 0

    def __post_init__(self):
        """Validate shapes and compute metadata."""
        if self.positions is not None:
            self.n_nodes = self.positions.shape[0]
        if self.connectivity is not None:
            self.n_elements = self.connectivity.shape[0]
        if self.element_block_IDs is not None:
            self.n_blocks = int(jnp.max(self.element_block_IDs) + 1)

        # Validate shapes
        if self.positions is not None:
            assert self.positions.shape == (self.n_nodes, 3), \
                f"positions shape mismatch: {self.positions.shape} vs ({self.n_nodes}, 3)"

        if self.connectivity is not None:
            assert self.connectivity.shape == (self.n_elements, 4), \
                f"connectivity shape mismatch: {self.connectivity.shape} vs ({self.n_elements}, 4)"

        if self.element_neighbors is not None:
            assert self.element_neighbors.shape == (self.n_elements, 4), \
                f"element_neighbors shape mismatch: {self.element_neighbors.shape} vs ({self.n_elements}, 4)"

        if self.element_block_IDs is not None:
            assert self.element_block_IDs.shape == (self.n_elements,), \
                f"element_block_IDs shape mismatch: {self.element_block_IDs.shape} vs ({self.n_elements},)"

        if self.velocities is not None:
            assert self.velocities.shape == (self.n_nodes, 3), \
                f"velocities shape mismatch: {self.velocities.shape} vs ({self.n_nodes}, 3)"

    def memory_usage_mb(self) -> Dict[str, float]:
        """
        Compute memory usage for each array.

        Returns:
            Dictionary with memory usage in MB
        """
        usage = {}

        if self.positions is not None:
            usage['positions'] = self.positions.nbytes / (1024**2)

        if self.connectivity is not None:
            usage['connectivity'] = self.connectivity.nbytes / (1024**2)

        if self.element_neighbors is not None:
            usage['element_neighbors'] = self.element_neighbors.nbytes / (1024**2)

        if self.element_block_IDs is not None:
            usage['element_block_IDs'] = self.element_block_IDs.nbytes / (1024**2)

        if self.velocities is not None:
            usage['velocities'] = self.velocities.nbytes / (1024**2)

        usage['total'] = sum(usage.values())

        return usage

    def __str__(self) -> str:
        """Human-readable summary."""
        mem = self.memory_usage_mb()

        lines = [
            "=" * 80,
            "MESH DATA (Flat Arrays)",
            "=" * 80,
            "",
            f"Nodes: {self.n_nodes:,}",
            f"Elements: {self.n_elements:,}",
            f"Blocks: {self.n_blocks}",
            "",
            "Array Shapes:",
            f"  positions: {self.positions.shape if self.positions is not None else 'None'}",
            f"  connectivity: {self.connectivity.shape if self.connectivity is not None else 'None'}",
            f"  element_neighbors: {self.element_neighbors.shape if self.element_neighbors is not None else 'None'}",
            f"  element_block_IDs: {self.element_block_IDs.shape if self.element_block_IDs is not None else 'None'}",
            f"  velocities: {self.velocities.shape if self.velocities is not None else 'None'}",
            "",
            "Memory Usage:",
        ]

        for key, value in mem.items():
            if key != 'total':
                lines.append(f"  {key}: {value:.2f} MB")
        lines.append(f"  {'=' * 20}")
        lines.append(f"  TOTAL: {mem['total']:.2f} MB")
        lines.append("=" * 80)

        return "\n".join(lines)


@dataclass
class ParticleData:
    """
    Flat array representation of particles.

    Minimal scan carry - only dynamic data stored.
    """

    # Particle state (scan carry)
    positions: jnp.ndarray          # (N_particles, 3) float64
    element_IDs: jnp.ndarray        # (N_particles,) int32 - current element
    active: jnp.ndarray             # (N_particles,) bool - active flag

    # Metadata
    n_particles: int = 0

    def __post_init__(self):
        """Validate shapes and compute metadata."""
        if self.positions is not None:
            self.n_particles = self.positions.shape[0]

        # Validate shapes
        if self.positions is not None:
            assert self.positions.shape == (self.n_particles, 3), \
                f"positions shape mismatch: {self.positions.shape} vs ({self.n_particles}, 3)"

        if self.element_IDs is not None:
            assert self.element_IDs.shape == (self.n_particles,), \
                f"element_IDs shape mismatch: {self.element_IDs.shape} vs ({self.n_particles},)"

        if self.active is not None:
            assert self.active.shape == (self.n_particles,), \
                f"active shape mismatch: {self.active.shape} vs ({self.n_particles},)"

    def memory_usage_mb(self) -> Dict[str, float]:
        """Compute memory usage."""
        usage = {}

        if self.positions is not None:
            usage['positions'] = self.positions.nbytes / (1024**2)

        if self.element_IDs is not None:
            usage['element_IDs'] = self.element_IDs.nbytes / (1024**2)

        if self.active is not None:
            usage['active'] = self.active.nbytes / (1024**2)

        usage['total'] = sum(usage.values())

        return usage

    def n_active(self) -> int:
        """Count active particles."""
        if self.active is None:
            return 0
        return int(jnp.sum(self.active))

    def __str__(self) -> str:
        """Human-readable summary."""
        mem = self.memory_usage_mb()

        lines = [
            "=" * 80,
            "PARTICLE DATA (Flat Arrays)",
            "=" * 80,
            "",
            f"Particles: {self.n_particles:,}",
            f"Active: {self.n_active():,} ({100*self.n_active()/max(self.n_particles,1):.1f}%)",
            "",
            "Array Shapes:",
            f"  positions: {self.positions.shape if self.positions is not None else 'None'}",
            f"  element_IDs: {self.element_IDs.shape if self.element_IDs is not None else 'None'}",
            f"  active: {self.active.shape if self.active is not None else 'None'}",
            "",
            "Memory Usage:",
        ]

        for key, value in mem.items():
            if key != 'total':
                lines.append(f"  {key}: {value:.2f} MB")
        lines.append(f"  {'=' * 20}")
        lines.append(f"  TOTAL: {mem['total']:.2f} MB")
        lines.append("")
        lines.append(f"Per-particle: {mem['total']*1024/max(self.n_particles,1):.2f} KB")
        lines.append("=" * 80)

        return "\n".join(lines)


@dataclass
class BlockPartitionData:
    """
    Block partitioning metadata.

    Used for Level 2 search - finding all elements in a block.
    """

    # Block grid configuration
    grid_size: Tuple[int, int, int]  # (nx, ny, nz)
    n_blocks: int

    # Spatial bounds (for computing block IDs from positions)
    bbox_min: jnp.ndarray            # (3,) - domain minimum
    bbox_max: jnp.ndarray            # (3,) - domain maximum
    block_size: jnp.ndarray          # (3,) - size of each block

    # Block statistics (for load balancing)
    elements_per_block: jnp.ndarray  # (n_blocks,) int32 - element count per block

    def __post_init__(self):
        """Validate and compute derived quantities."""
        if self.grid_size is not None:
            expected_n_blocks = np.prod(self.grid_size)
            assert self.n_blocks == expected_n_blocks, \
                f"n_blocks mismatch: {self.n_blocks} vs {expected_n_blocks}"

        if self.bbox_min is not None and self.bbox_max is not None:
            self.block_size = (self.bbox_max - self.bbox_min) / np.array(self.grid_size)

    def load_imbalance_factor(self) -> float:
        """Compute load imbalance (max/mean)."""
        if self.elements_per_block is None:
            return 0.0

        non_empty = self.elements_per_block > 0
        if not jnp.any(non_empty):
            return 0.0

        mean_size = jnp.mean(self.elements_per_block[non_empty])
        max_size = jnp.max(self.elements_per_block)

        return float(max_size / mean_size)

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 80,
            "BLOCK PARTITION DATA",
            "=" * 80,
            "",
            f"Grid size: {self.grid_size[0]}×{self.grid_size[1]}×{self.grid_size[2]}",
            f"Total blocks: {self.n_blocks}",
            "",
            f"Bounding box:",
            f"  Min: [{self.bbox_min[0]:.6f}, {self.bbox_min[1]:.6f}, {self.bbox_min[2]:.6f}]",
            f"  Max: [{self.bbox_max[0]:.6f}, {self.bbox_max[1]:.6f}, {self.bbox_max[2]:.6f}]",
            f"  Block size: [{self.block_size[0]:.6f}, {self.block_size[1]:.6f}, {self.block_size[2]:.6f}]",
            "",
            f"Elements per block:",
            f"  Min: {jnp.min(self.elements_per_block):,}",
            f"  Max: {jnp.max(self.elements_per_block):,}",
            f"  Mean: {jnp.mean(self.elements_per_block):.0f}",
            f"  Load imbalance: {self.load_imbalance_factor():.2f}×",
            "=" * 80,
        ]

        return "\n".join(lines)


def create_mesh_data(
    positions: np.ndarray,
    connectivity: np.ndarray,
    element_neighbors: Optional[np.ndarray] = None,
    element_block_IDs: Optional[np.ndarray] = None,
    velocities: Optional[np.ndarray] = None,
    device: str = "cpu"
) -> MeshData:
    """
    Create MeshData from NumPy arrays.

    Args:
        positions: (N_nodes, 3) float64
        connectivity: (N_elements, 4) int32
        element_neighbors: (N_elements, 4) int32 (optional)
        element_block_IDs: (N_elements,) int32 (optional)
        velocities: (N_nodes, 3) float64 (optional)
        device: "cpu" or "gpu" - where to place arrays

    Returns:
        MeshData object with JAX arrays
    """
    # Convert to JAX arrays
    if device == "gpu":
        # Transfer to GPU
        positions_jax = jnp.array(positions, dtype=jnp.float64)
        connectivity_jax = jnp.array(connectivity, dtype=jnp.int32)

        element_neighbors_jax = jnp.array(element_neighbors, dtype=jnp.int32) \
            if element_neighbors is not None else None

        element_block_IDs_jax = jnp.array(element_block_IDs, dtype=jnp.int32) \
            if element_block_IDs is not None else None

        velocities_jax = jnp.array(velocities, dtype=jnp.float64) \
            if velocities is not None else None
    else:
        # Keep on CPU (still wrapped in JAX arrays)
        import jax
        with jax.default_device(jax.devices('cpu')[0]):
            positions_jax = jnp.array(positions, dtype=jnp.float64)
            connectivity_jax = jnp.array(connectivity, dtype=jnp.int32)

            element_neighbors_jax = jnp.array(element_neighbors, dtype=jnp.int32) \
                if element_neighbors is not None else None

            element_block_IDs_jax = jnp.array(element_block_IDs, dtype=jnp.int32) \
                if element_block_IDs is not None else None

            velocities_jax = jnp.array(velocities, dtype=jnp.float64) \
                if velocities is not None else None

    return MeshData(
        positions=positions_jax,
        connectivity=connectivity_jax,
        element_neighbors=element_neighbors_jax,
        element_block_IDs=element_block_IDs_jax,
        velocities=velocities_jax,
    )


def create_particle_data(
    positions: np.ndarray,
    element_IDs: Optional[np.ndarray] = None,
    active: Optional[np.ndarray] = None,
    device: str = "cpu"
) -> ParticleData:
    """
    Create ParticleData from NumPy arrays.

    Args:
        positions: (N_particles, 3) float64
        element_IDs: (N_particles,) int32 (optional, defaults to -1)
        active: (N_particles,) bool (optional, defaults to True)
        device: "cpu" or "gpu"

    Returns:
        ParticleData object with JAX arrays
    """
    n_particles = positions.shape[0]

    # Default values
    if element_IDs is None:
        element_IDs = np.full(n_particles, -1, dtype=np.int32)

    if active is None:
        active = np.ones(n_particles, dtype=bool)

    # Convert to JAX
    if device == "gpu":
        positions_jax = jnp.array(positions, dtype=jnp.float64)
        element_IDs_jax = jnp.array(element_IDs, dtype=jnp.int32)
        active_jax = jnp.array(active, dtype=bool)
    else:
        import jax
        with jax.default_device(jax.devices('cpu')[0]):
            positions_jax = jnp.array(positions, dtype=jnp.float64)
            element_IDs_jax = jnp.array(element_IDs, dtype=jnp.int32)
            active_jax = jnp.array(active, dtype=bool)

    return ParticleData(
        positions=positions_jax,
        element_IDs=element_IDs_jax,
        active=active_jax,
    )


if __name__ == "__main__":
    # Example usage
    print("Testing flat array data structures...")

    # Create simple test mesh (single tetrahedron)
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    connectivity = np.array([[0, 1, 2, 3]], dtype=np.int32)

    element_neighbors = np.array([[-1, -1, -1, -1]], dtype=np.int32)

    element_block_IDs = np.array([0], dtype=np.int32)

    # Create mesh data
    mesh_data = create_mesh_data(
        positions, connectivity, element_neighbors, element_block_IDs
    )

    print(mesh_data)
    print()

    # Create particle data
    particle_positions = np.array([[0.25, 0.25, 0.25]], dtype=np.float64)
    particle_data = create_particle_data(particle_positions)

    print(particle_data)
