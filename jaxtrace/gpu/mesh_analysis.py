#!/usr/bin/env python3
"""
Mesh Analysis Tools for GPU Implementation Planning

This module provides tools to analyze mesh characteristics and generate
recommendations for GPU configuration parameters.

Phase 0.1 of V3 Plan
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import vtk
from vtk.util import numpy_support


@dataclass
class MeshStatistics:
    """Statistics about mesh topology and spatial distribution."""

    # Basic counts
    n_nodes: int
    n_elements: int

    # Spatial extent
    bbox_min: np.ndarray  # (3,)
    bbox_max: np.ndarray  # (3,)
    bbox_size: np.ndarray  # (3,)

    # Element distribution
    element_volume_min: float
    element_volume_max: float
    element_volume_mean: float
    element_volume_std: float

    # Node degree (elements sharing each node)
    node_degree_min: int
    node_degree_max: int
    node_degree_mean: float

    # Connectivity quality
    n_boundary_elements: int
    n_interior_elements: int
    neighbor_count_distribution: Dict[int, int]  # {n_neighbors: count}

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 80,
            "MESH STATISTICS",
            "=" * 80,
            "",
            "Basic Counts:",
            f"  Nodes: {self.n_nodes:,}",
            f"  Elements: {self.n_elements:,}",
            "",
            "Spatial Extent:",
            f"  X: [{self.bbox_min[0]:.6f}, {self.bbox_max[0]:.6f}]  (size: {self.bbox_size[0]:.6f})",
            f"  Y: [{self.bbox_min[1]:.6f}, {self.bbox_max[1]:.6f}]  (size: {self.bbox_size[1]:.6f})",
            f"  Z: [{self.bbox_min[2]:.6f}, {self.bbox_max[2]:.6f}]  (size: {self.bbox_size[2]:.6f})",
            "",
            "Element Volumes:",
            f"  Min: {self.element_volume_min:.3e}",
            f"  Max: {self.element_volume_max:.3e}",
            f"  Mean: {self.element_volume_mean:.3e}",
            f"  Std: {self.element_volume_std:.3e}",
            f"  Range ratio: {self.element_volume_max/self.element_volume_min:.2f}×",
            "",
            "Node Degree (elements per node):",
            f"  Min: {self.node_degree_min}",
            f"  Max: {self.node_degree_max}",
            f"  Mean: {self.node_degree_mean:.2f}",
            "",
            "Connectivity:",
            f"  Boundary elements: {self.n_boundary_elements:,} ({100*self.n_boundary_elements/self.n_elements:.1f}%)",
            f"  Interior elements: {self.n_interior_elements:,} ({100*self.n_interior_elements/self.n_elements:.1f}%)",
            "",
            "Neighbor Distribution:",
        ]
        for n_neighbors in sorted(self.neighbor_count_distribution.keys()):
            count = self.neighbor_count_distribution[n_neighbors]
            pct = 100 * count / self.n_elements
            lines.append(f"  {n_neighbors} neighbors: {count:,} ({pct:.1f}%)")

        lines.append("=" * 80)
        return "\n".join(lines)


@dataclass
class BlockPartitionAnalysis:
    """Analysis of spatial partitioning into blocks."""

    # Grid configuration
    grid_size: Tuple[int, int, int]
    n_blocks: int

    # Block element distribution
    elements_per_block: np.ndarray  # (n_blocks,)
    block_sizes_min: int
    block_sizes_max: int
    block_sizes_mean: float
    block_sizes_std: float
    block_sizes_median: int
    block_sizes_p95: int

    # Load imbalance
    load_imbalance_factor: float  # max/mean

    # Empty blocks
    n_empty_blocks: int

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 80,
            f"BLOCK PARTITION ANALYSIS ({self.grid_size[0]}×{self.grid_size[1]}×{self.grid_size[2]})",
            "=" * 80,
            "",
            f"Total blocks: {self.n_blocks}",
            f"Non-empty blocks: {self.n_blocks - self.n_empty_blocks} / {self.n_blocks}",
            "",
            "Elements per block:",
            f"  Min: {self.block_sizes_min:,}",
            f"  Max: {self.block_sizes_max:,}",
            f"  Mean: {self.block_sizes_mean:,.0f}",
            f"  Median: {self.block_sizes_median:,}",
            f"  Std: {self.block_sizes_std:,.0f}",
            f"  95th percentile: {self.block_sizes_p95:,}",
            "",
            f"Load imbalance factor: {self.load_imbalance_factor:.2f}× (max/mean)",
        ]

        if self.load_imbalance_factor > 2.0:
            lines.append("  ⚠️  HIGH load imbalance! Consider adaptive grid (Phase 8)")
        elif self.load_imbalance_factor > 1.5:
            lines.append("  ⚠️  Moderate load imbalance")
        else:
            lines.append("  ✅ Good load balance")

        lines.append("=" * 80)
        return "\n".join(lines)


@dataclass
class GPUConfigRecommendations:
    """Recommended GPUConfig parameters based on mesh analysis."""

    # Block partitioning
    recommended_grid_size: Tuple[int, int, int]
    recommended_n_blocks: int
    recommended_max_elements_per_block: int

    # Octree
    recommended_max_elements_per_octree_node: int

    # Memory estimates
    estimated_mesh_memory_mb: float
    estimated_particle_memory_mb: float  # Per 1M particles

    # Warnings
    warnings: list[str]

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 80,
            "GPU CONFIG RECOMMENDATIONS",
            "=" * 80,
            "",
            "Block Partitioning:",
            f"  grid_size = {self.recommended_grid_size}  # {self.recommended_n_blocks} blocks",
            f"  max_elements_per_block = {self.recommended_max_elements_per_block:,}",
            "",
            "Octree:",
            f"  max_elements_per_octree_node = {self.recommended_max_elements_per_octree_node:,}",
            "",
            "Memory Estimates:",
            f"  Mesh data (static): {self.estimated_mesh_memory_mb:.1f} MB",
            f"  Particle data (per 1M): {self.estimated_particle_memory_mb:.1f} MB",
            f"  Total for 1M particles: {self.estimated_mesh_memory_mb + self.estimated_particle_memory_mb:.1f} MB",
            "",
        ]

        if self.warnings:
            lines.append("⚠️  WARNINGS:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)


def load_mesh(mesh_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mesh from PVTU file.

    Args:
        mesh_path: Path to PVTU file or directory containing PVTU

    Returns:
        positions: (N_nodes, 3) node coordinates
        connectivity: (N_elements, 4) element node indices
    """
    # Find PVTU file
    if mesh_path.is_dir():
        pvtu_files = list(mesh_path.glob("*.pvtu"))
        if not pvtu_files:
            raise FileNotFoundError(f"No PVTU files in {mesh_path}")
        mesh_file = pvtu_files[0]
    else:
        mesh_file = mesh_path

    print(f"Loading mesh: {mesh_file}")

    # Load with VTK
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(str(mesh_file))
    reader.Update()
    output = reader.GetOutput()

    # Extract positions
    positions = numpy_support.vtk_to_numpy(output.GetPoints().GetData())
    print(f"  Nodes: {positions.shape[0]:,}")

    # Extract connectivity (assume all tetrahedral)
    n_cells = output.GetNumberOfCells()
    connectivity_data = numpy_support.vtk_to_numpy(output.GetCells().GetData())

    connectivity = np.zeros((n_cells, 4), dtype=np.int32)
    for i in range(n_cells):
        connectivity[i] = connectivity_data[i * 5 + 1 : i * 5 + 5]

    print(f"  Elements: {n_cells:,}")

    return positions, connectivity


def compute_tetrahedron_volume(vertices: np.ndarray) -> float:
    """
    Compute volume of tetrahedron.

    Args:
        vertices: (4, 3) vertex coordinates

    Returns:
        volume: Scalar volume
    """
    # V = |det([v1-v0, v2-v0, v3-v0])| / 6
    v0, v1, v2, v3 = vertices
    mat = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
    return abs(np.linalg.det(mat)) / 6.0


def analyze_mesh_statistics(
    positions: np.ndarray,
    connectivity: np.ndarray
) -> MeshStatistics:
    """
    Compute comprehensive mesh statistics.

    Args:
        positions: (N_nodes, 3)
        connectivity: (N_elements, 4)

    Returns:
        MeshStatistics object
    """
    n_nodes = positions.shape[0]
    n_elements = connectivity.shape[0]

    print("\n📊 Analyzing mesh statistics...")

    # Spatial extent
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    bbox_size = bbox_max - bbox_min

    # Element volumes
    print("  Computing element volumes...")
    volumes = np.zeros(n_elements)
    for i, elem in enumerate(connectivity):
        if i % 500000 == 0 and i > 0:
            print(f"    Processed {i:,} / {n_elements:,} ({100*i/n_elements:.1f}%)")
        vertices = positions[elem]
        volumes[i] = compute_tetrahedron_volume(vertices)

    volume_min = volumes.min()
    volume_max = volumes.max()
    volume_mean = volumes.mean()
    volume_std = volumes.std()

    # Node degree (count elements per node)
    print("  Computing node degrees...")
    node_degree = np.zeros(n_nodes, dtype=np.int32)
    for elem in connectivity:
        node_degree[elem] += 1

    node_degree_min = int(node_degree.min())
    node_degree_max = int(node_degree.max())
    node_degree_mean = float(node_degree.mean())

    # Neighbor analysis (faces shared between elements)
    print("  Analyzing element connectivity...")
    # Count neighbors per element (max 4 for tetrahedra)
    neighbor_counts = np.zeros(n_elements, dtype=np.int32)

    # Build face map
    face_to_elements = {}
    for elem_id, elem in enumerate(connectivity):
        if elem_id % 500000 == 0 and elem_id > 0:
            print(f"    Processed {elem_id:,} / {n_elements:,} ({100*elem_id/n_elements:.1f}%)")

        # 4 faces per tetrahedron
        faces = [
            tuple(sorted([elem[0], elem[1], elem[2]])),
            tuple(sorted([elem[0], elem[1], elem[3]])),
            tuple(sorted([elem[0], elem[2], elem[3]])),
            tuple(sorted([elem[1], elem[2], elem[3]])),
        ]

        for face in faces:
            if face not in face_to_elements:
                face_to_elements[face] = []
            face_to_elements[face].append(elem_id)

    # Count neighbors
    for face, elements in face_to_elements.items():
        if len(elements) == 2:
            # Shared face = neighbor
            neighbor_counts[elements[0]] += 1
            neighbor_counts[elements[1]] += 1

    neighbor_distribution = {}
    for n_neighbors in range(5):  # 0-4
        neighbor_distribution[n_neighbors] = int(np.sum(neighbor_counts == n_neighbors))

    n_boundary = int(np.sum(neighbor_counts < 4))
    n_interior = int(np.sum(neighbor_counts == 4))

    return MeshStatistics(
        n_nodes=n_nodes,
        n_elements=n_elements,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        bbox_size=bbox_size,
        element_volume_min=volume_min,
        element_volume_max=volume_max,
        element_volume_mean=volume_mean,
        element_volume_std=volume_std,
        node_degree_min=node_degree_min,
        node_degree_max=node_degree_max,
        node_degree_mean=node_degree_mean,
        n_boundary_elements=n_boundary,
        n_interior_elements=n_interior,
        neighbor_count_distribution=neighbor_distribution,
    )


def analyze_block_partition(
    positions: np.ndarray,
    connectivity: np.ndarray,
    grid_size: Tuple[int, int, int]
) -> BlockPartitionAnalysis:
    """
    Analyze spatial partitioning into blocks.

    Args:
        positions: (N_nodes, 3)
        connectivity: (N_elements, 4)
        grid_size: (nx, ny, nz) block grid

    Returns:
        BlockPartitionAnalysis object
    """
    n_elements = connectivity.shape[0]
    n_blocks = np.prod(grid_size)

    print(f"\n📊 Analyzing block partition ({grid_size[0]}×{grid_size[1]}×{grid_size[2]})...")

    # Compute bounding box
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)

    # Compute element centroids
    print("  Computing element centroids...")
    centroids = np.zeros((n_elements, 3))
    for i, elem in enumerate(connectivity):
        if i % 500000 == 0 and i > 0:
            print(f"    Processed {i:,} / {n_elements:,} ({100*i/n_elements:.1f}%)")
        centroids[i] = positions[elem].mean(axis=0)

    # Assign to blocks
    print("  Assigning elements to blocks...")
    block_counts = np.zeros(n_blocks, dtype=np.int32)

    # Compute block size
    block_size = (bbox_max - bbox_min) / np.array(grid_size)

    for i, centroid in enumerate(centroids):
        if i % 500000 == 0 and i > 0:
            print(f"    Processed {i:,} / {n_elements:,} ({100*i/n_elements:.1f}%)")

        # Compute block indices
        block_idx = np.floor((centroid - bbox_min) / block_size).astype(np.int32)
        block_idx = np.clip(block_idx, 0, np.array(grid_size) - 1)

        # Convert to flat block ID
        block_id = (block_idx[0] * grid_size[1] * grid_size[2] +
                   block_idx[1] * grid_size[2] +
                   block_idx[2])

        block_counts[block_id] += 1

    # Compute statistics
    non_empty = block_counts > 0
    n_empty = int(np.sum(~non_empty))

    block_min = int(block_counts[non_empty].min()) if np.any(non_empty) else 0
    block_max = int(block_counts.max())
    block_mean = float(block_counts[non_empty].mean()) if np.any(non_empty) else 0
    block_std = float(block_counts[non_empty].std()) if np.any(non_empty) else 0
    block_median = int(np.median(block_counts[non_empty])) if np.any(non_empty) else 0
    block_p95 = int(np.percentile(block_counts[non_empty], 95)) if np.any(non_empty) else 0

    load_imbalance = block_max / block_mean if block_mean > 0 else 0

    return BlockPartitionAnalysis(
        grid_size=grid_size,
        n_blocks=n_blocks,
        elements_per_block=block_counts,
        block_sizes_min=block_min,
        block_sizes_max=block_max,
        block_sizes_mean=block_mean,
        block_sizes_std=block_std,
        block_sizes_median=block_median,
        block_sizes_p95=block_p95,
        load_imbalance_factor=load_imbalance,
        n_empty_blocks=n_empty,
    )


def recommend_gpu_config(
    mesh_stats: MeshStatistics,
    block_analyses: list[BlockPartitionAnalysis],
    target_gpu_memory_gb: float = 8.0,
) -> GPUConfigRecommendations:
    """
    Generate GPU configuration recommendations.

    Args:
        mesh_stats: Mesh statistics
        block_analyses: List of block partition analyses for different grid sizes
        target_gpu_memory_gb: Available GPU memory

    Returns:
        GPUConfigRecommendations object
    """
    print("\n🔍 Generating GPU config recommendations...")

    warnings = []

    # Select best block partition (lowest load imbalance with reasonable block size)
    best_partition = None
    best_score = float('inf')

    for partition in block_analyses:
        # Score = load_imbalance + penalty for very small blocks
        penalty = 1.0 if partition.block_sizes_mean > 1000 else 2.0
        score = partition.load_imbalance_factor * penalty

        if score < best_score:
            best_score = score
            best_partition = partition

    if best_partition is None:
        raise ValueError("No valid block partitions")

    # Recommended max_elements_per_block: 95th percentile + 20% buffer
    recommended_max = int(best_partition.block_sizes_p95 * 1.2)

    # Cap at reasonable limit (10K for now)
    if recommended_max > 10000:
        warnings.append(f"95th percentile block size is {best_partition.block_sizes_p95:,}, "
                       f"capping max_elements_per_block at 10,000")
        recommended_max = 10000

    # Recommended octree max
    # For balanced meshes, use 1000; for imbalanced, use higher
    if best_partition.load_imbalance_factor > 2.0:
        recommended_octree_max = 1000
        warnings.append("High load imbalance detected - recommend Phase 8 adaptive grid")
    else:
        recommended_octree_max = 500

    # Memory estimates
    n_nodes = mesh_stats.n_nodes
    n_elements = mesh_stats.n_elements

    # Mesh memory (static):
    # - positions: N_nodes × 3 × 8 bytes
    # - connectivity: N_elements × 4 × 4 bytes
    # - element_block_IDs: N_elements × 4 bytes
    # - element_neighbors: N_elements × 4 × 4 bytes
    mesh_memory_mb = (
        n_nodes * 3 * 8 +           # positions (float64)
        n_elements * 4 * 4 +         # connectivity (int32)
        n_elements * 4 +             # element_block_IDs (int32)
        n_elements * 4 * 4          # element_neighbors (int32)
    ) / (1024**2)

    # Particle memory (per 1M particles):
    # - positions: 1M × 3 × 8 = 24 MB
    # - element_IDs: 1M × 4 = 4 MB
    # - active: 1M × 1 = 1 MB
    particle_memory_mb = (1_000_000 * 3 * 8 + 1_000_000 * 4 + 1_000_000 * 1) / (1024**2)

    # Check if fits in GPU memory
    total_for_1m = mesh_memory_mb + particle_memory_mb
    if total_for_1m > target_gpu_memory_gb * 1024:
        warnings.append(f"Mesh + 1M particles requires {total_for_1m:.1f} MB, "
                       f"exceeds target {target_gpu_memory_gb:.1f} GB")

    return GPUConfigRecommendations(
        recommended_grid_size=best_partition.grid_size,
        recommended_n_blocks=best_partition.n_blocks,
        recommended_max_elements_per_block=recommended_max,
        recommended_max_elements_per_octree_node=recommended_octree_max,
        estimated_mesh_memory_mb=mesh_memory_mb,
        estimated_particle_memory_mb=particle_memory_mb,
        warnings=warnings,
    )


def analyze_mesh_for_gpu(
    mesh_path: Path,
    grid_sizes: Optional[list[Tuple[int, int, int]]] = None,
    target_gpu_memory_gb: float = 8.0,
) -> Tuple[MeshStatistics, list[BlockPartitionAnalysis], GPUConfigRecommendations]:
    """
    Complete mesh analysis for GPU implementation planning.

    Args:
        mesh_path: Path to PVTU file or directory
        grid_sizes: List of (nx, ny, nz) grid configurations to test
                   If None, will test: (2,2,1), (4,4,2), (8,8,4), (16,16,8)
        target_gpu_memory_gb: Available GPU memory

    Returns:
        mesh_stats: Mesh statistics
        block_analyses: List of block partition analyses
        recommendations: GPU config recommendations
    """
    if grid_sizes is None:
        grid_sizes = [(2, 2, 1), (4, 4, 2), (8, 8, 4), (16, 16, 8)]

    # Load mesh
    positions, connectivity = load_mesh(mesh_path)

    # Analyze mesh
    mesh_stats = analyze_mesh_statistics(positions, connectivity)

    # Analyze different block partitions
    block_analyses = []
    for grid_size in grid_sizes:
        analysis = analyze_block_partition(positions, connectivity, grid_size)
        block_analyses.append(analysis)

    # Generate recommendations
    recommendations = recommend_gpu_config(
        mesh_stats, block_analyses, target_gpu_memory_gb
    )

    return mesh_stats, block_analyses, recommendations


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mesh_analysis.py <mesh_path>")
        sys.exit(1)

    mesh_path = Path(sys.argv[1])

    # Run analysis
    mesh_stats, block_analyses, recommendations = analyze_mesh_for_gpu(mesh_path)

    # Print results
    print("\n" + str(mesh_stats))
    print()
    for analysis in block_analyses:
        print(str(analysis))
        print()
    print(str(recommendations))
