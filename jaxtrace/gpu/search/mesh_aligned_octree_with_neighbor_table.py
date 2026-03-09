"""
Mesh-Aligned Octree with Pre-Computed Neighbor Table

This module extends the mesh-aligned octree structure with a pre-computed
neighbor lookup table. This avoids all JAX tracing issues by computing
neighbor relationships on the CPU and uploading them to GPU as a simple array.

Architecture:
    CPU: Build neighbor table (517k cells × 26 neighbors)
    GPU: Direct lookup - no dynamic neighbor computation

Expected Performance:
    - ~99% searchability (vs 74.6% without neighbors)
    - ~15-20 tests per particle
    - ~50-100K particles/sec
    - Extra memory: ~52 MB for neighbor table
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass

from .mesh_aligned_octree_single_cell import (
    OctreeCellDataSingle,
    encode_morton_3d_single,
)


@dataclass
class OctreeCellDataWithNeighbors:
    """
    Mesh-aligned octree with pre-computed neighbor lookup table.

    Extends OctreeCellDataSingle with:
        cell_neighbors: (n_cells, 26) int32 - neighbor cell indices
            For each cell, stores indices of its 26 spatial neighbors.
            -1 indicates neighbor doesn't exist (boundary condition).
    """
    # Original octree data
    cell_morton_codes: np.ndarray      # (n_cells,) uint64
    cell_levels: np.ndarray            # (n_cells,) uint8
    cell_sizes: np.ndarray             # (n_cells, 3) float64
    cell_grid_indices: np.ndarray      # (n_cells, 3) int32

    cell_to_elements_offsets: np.ndarray  # (n_cells + 1,) int32
    cell_to_elements_data: np.ndarray     # (total_entries,) int32

    element_to_cells: np.ndarray          # (n_elements,) int32

    # NEW: Pre-computed neighbor table
    cell_neighbors: np.ndarray            # (n_cells, 26) int32

    n_cells: int
    n_elements: int
    cells_per_element_mean: float
    elements_per_cell_mean: float


def build_neighbor_lookup_table(
    octree_cells: OctreeCellDataSingle,
    verbose: bool = True
) -> np.ndarray:
    """
    Build pre-computed neighbor lookup table on CPU.

    For each cell, finds its 26 spatial neighbors (3×3×3 - center) at the same
    refinement level using grid indices + Morton code lookup.

    Args:
        octree_cells: Octree structure from extract_octree_cells_single()
        verbose: Print progress

    Returns:
        cell_neighbors: (n_cells, 26) int32 array
            cell_neighbors[i, j] = index of j-th neighbor of cell i
            -1 if neighbor doesn't exist
    """
    n_cells = octree_cells.n_cells

    if verbose:
        print(f"\n{'='*80}")
        print("Building Neighbor Lookup Table")
        print(f"{'='*80}")
        print(f"  Cells: {n_cells:,}")
        print(f"  Finding 26 neighbors per cell...")

    # Build reverse lookup: (morton, level) -> cell_idx
    cell_key_to_idx = {}
    for cell_idx in range(n_cells):
        morton = octree_cells.cell_morton_codes[cell_idx]
        level = octree_cells.cell_levels[cell_idx]
        cell_key = (morton, level)
        cell_key_to_idx[cell_key] = cell_idx

    # Allocate neighbor table
    cell_neighbors = np.full((n_cells, 26), -1, dtype=np.int32)

    # Morton encoding parameters (must match extraction)
    morton_offset = (1 << 19)  # 2^19
    morton_max_coord = (1 << 20)  # 2^20

    # For each cell, find its 26 neighbors
    neighbor_idx_map = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                if di == 0 and dj == 0 and dk == 0:
                    continue  # Skip center
                neighbor_idx_map.append((di, dj, dk))

    n_found_total = 0
    n_missing_total = 0

    for cell_idx in range(n_cells):
        grid = octree_cells.cell_grid_indices[cell_idx]
        level = octree_cells.cell_levels[cell_idx]

        for neighbor_offset_idx, (di, dj, dk) in enumerate(neighbor_idx_map):
            # Compute neighbor grid indices
            neighbor_grid_i = grid[0] + di
            neighbor_grid_j = grid[1] + dj
            neighbor_grid_k = grid[2] + dk

            # Encode to Morton
            i_offset = np.clip(neighbor_grid_i + morton_offset, 0, morton_max_coord - 1)
            j_offset = np.clip(neighbor_grid_j + morton_offset, 0, morton_max_coord - 1)
            k_offset = np.clip(neighbor_grid_k + morton_offset, 0, morton_max_coord - 1)

            neighbor_morton = encode_morton_3d_single(
                int(i_offset), int(j_offset), int(k_offset), max_depth=21
            )

            # Look up neighbor cell
            neighbor_key = (neighbor_morton, level)
            neighbor_cell_idx = cell_key_to_idx.get(neighbor_key, -1)

            cell_neighbors[cell_idx, neighbor_offset_idx] = neighbor_cell_idx

            if neighbor_cell_idx >= 0:
                n_found_total += 1
            else:
                n_missing_total += 1

        if verbose and (cell_idx + 1) % 100000 == 0:
            pct = 100.0 * (cell_idx + 1) / n_cells
            print(f"    Processed {cell_idx + 1:,}/{n_cells:,} cells ({pct:.1f}%)")

    # Statistics
    neighbors_per_cell = (cell_neighbors >= 0).sum(axis=1)
    mean_neighbors = neighbors_per_cell.mean()

    if verbose:
        print(f"\n  ✅ Neighbor table built!")
        print(f"    Total neighbor lookups: {n_cells * 26:,}")
        print(f"    Found: {n_found_total:,} ({100.0 * n_found_total / (n_cells * 26):.1f}%)")
        print(f"    Missing (boundary): {n_missing_total:,} ({100.0 * n_missing_total / (n_cells * 26):.1f}%)")
        print(f"    Mean neighbors per cell: {mean_neighbors:.1f}")
        print(f"    Memory: {cell_neighbors.nbytes / (1024**2):.1f} MB")
        print(f"{'='*80}\n")

    return cell_neighbors


def add_neighbor_table_to_octree(
    octree_cells: OctreeCellDataSingle,
    verbose: bool = True
) -> OctreeCellDataWithNeighbors:
    """
    Add pre-computed neighbor table to existing octree structure.

    Args:
        octree_cells: Base octree from extract_octree_cells_single()
        verbose: Print progress

    Returns:
        Extended octree with neighbor table
    """
    # Build neighbor table
    cell_neighbors = build_neighbor_lookup_table(octree_cells, verbose=verbose)

    # Create extended structure
    octree_with_neighbors = OctreeCellDataWithNeighbors(
        cell_morton_codes=octree_cells.cell_morton_codes,
        cell_levels=octree_cells.cell_levels,
        cell_sizes=octree_cells.cell_sizes,
        cell_grid_indices=octree_cells.cell_grid_indices,
        cell_to_elements_offsets=octree_cells.cell_to_elements_offsets,
        cell_to_elements_data=octree_cells.cell_to_elements_data,
        element_to_cells=octree_cells.element_to_cells,
        cell_neighbors=cell_neighbors,
        n_cells=octree_cells.n_cells,
        n_elements=octree_cells.n_elements,
        cells_per_element_mean=octree_cells.cells_per_element_mean,
        elements_per_cell_mean=octree_cells.elements_per_cell_mean,
    )

    return octree_with_neighbors


# ============================================================================
# GPU Structure with Neighbor Table
# ============================================================================

import jax
import jax.numpy as jnp
from jax import lax


@dataclass
class MeshAlignedOctreeGPUWithNeighbors:
    """
    GPU-resident mesh-aligned octree with pre-computed neighbor table.

    Extends MeshAlignedOctreeGPU with:
        cell_neighbors: (n_cells, 26) int32 - neighbor cell indices
    """
    # Core mesh data
    connectivity: jax.Array  # (n_elements, 4) int32
    node_positions: jax.Array  # (n_nodes, 3) float32

    # Cell structure (sorted by Morton code)
    cell_morton_codes: jax.Array  # (n_cells,) uint64
    cell_levels: jax.Array  # (n_cells,) uint8
    cell_sizes: jax.Array  # (n_cells, 3) float32
    cell_grid_indices: jax.Array  # (n_cells, 3) int32

    # Cell → elements mapping (CSR)
    cell_to_elements_offsets: jax.Array  # (n_cells + 1,) int32
    cell_to_elements_data: jax.Array  # (total_entries,) int32

    # NEW: Pre-computed neighbor table
    cell_neighbors: jax.Array  # (n_cells, 26) int32

    # Morton parameters
    morton_offset: jnp.int32
    morton_max_coord: jnp.int32
    max_depth: jnp.int32

    # Bounding box
    bbox_min: jax.Array  # (3,) float32
    bbox_max: jax.Array  # (3,) float32

    # Level-specific cell sizes
    level_cell_sizes: jax.Array  # (max_level + 1, 3) float32

    # Statistics
    n_cells: jnp.int32
    n_elements: jnp.int32
    mean_elements_per_cell: jnp.float32


def upload_octree_with_neighbors_to_gpu(
    connectivity: np.ndarray,
    node_positions: np.ndarray,
    octree_with_neighbors: OctreeCellDataWithNeighbors,
    verbose: bool = True
) -> MeshAlignedOctreeGPUWithNeighbors:
    """
    Upload mesh-aligned octree with neighbor table to GPU.

    Args:
        connectivity: (n_elements, 4) int32 - element connectivity
        node_positions: (n_nodes, 3) float32 - node coordinates
        octree_with_neighbors: Extended octree from add_neighbor_table_to_octree()
        verbose: Print upload statistics

    Returns:
        GPU-resident octree with neighbor table
    """
    if verbose:
        print(f"\n{'='*80}")
        print("Uploading Octree with Neighbor Table to GPU")
        print(f"{'='*80}")

    # Upload mesh data
    connectivity_gpu = jnp.array(connectivity, dtype=jnp.int32)
    node_positions_gpu = jnp.array(node_positions, dtype=jnp.float32)

    # Upload cell structure
    cell_morton_codes_gpu = jnp.array(octree_with_neighbors.cell_morton_codes, dtype=jnp.uint64)
    cell_levels_gpu = jnp.array(octree_with_neighbors.cell_levels, dtype=jnp.uint8)
    cell_sizes_gpu = jnp.array(octree_with_neighbors.cell_sizes, dtype=jnp.float32)
    cell_grid_indices_gpu = jnp.array(octree_with_neighbors.cell_grid_indices, dtype=jnp.int32)

    # Upload CSR structure
    cell_to_elements_offsets_gpu = jnp.array(
        octree_with_neighbors.cell_to_elements_offsets, dtype=jnp.int32
    )
    cell_to_elements_data_gpu = jnp.array(
        octree_with_neighbors.cell_to_elements_data, dtype=jnp.int32
    )

    # Upload neighbor table
    cell_neighbors_gpu = jnp.array(octree_with_neighbors.cell_neighbors, dtype=jnp.int32)

    # Compute bounding box
    bbox_min = node_positions.min(axis=0).astype(np.float32)
    bbox_max = node_positions.max(axis=0).astype(np.float32)
    bbox_min_gpu = jnp.array(bbox_min)
    bbox_max_gpu = jnp.array(bbox_max)

    # Morton parameters
    morton_offset = jnp.int32(1 << 19)
    morton_max_coord = jnp.int32(1 << 20)
    max_depth = jnp.int32(21)

    # Compute level-specific cell sizes
    unique_levels = np.unique(octree_with_neighbors.cell_levels)
    max_level = int(np.max(unique_levels))
    level_cell_sizes_cpu = np.zeros((max_level + 1, 3), dtype=np.float32)

    for level in unique_levels:
        level_mask = octree_with_neighbors.cell_levels == level
        level_sizes = octree_with_neighbors.cell_sizes[level_mask]
        level_cell_sizes_cpu[level] = level_sizes[0]

    level_cell_sizes_gpu = jnp.array(level_cell_sizes_cpu, dtype=jnp.float32)

    # Statistics
    n_cells = jnp.int32(octree_with_neighbors.n_cells)
    n_elements = jnp.int32(octree_with_neighbors.n_elements)
    mean_elements_per_cell = jnp.float32(octree_with_neighbors.elements_per_cell_mean)

    # Create GPU structure
    octree_gpu = MeshAlignedOctreeGPUWithNeighbors(
        connectivity=connectivity_gpu,
        node_positions=node_positions_gpu,
        cell_morton_codes=cell_morton_codes_gpu,
        cell_levels=cell_levels_gpu,
        cell_sizes=cell_sizes_gpu,
        cell_grid_indices=cell_grid_indices_gpu,
        cell_to_elements_offsets=cell_to_elements_offsets_gpu,
        cell_to_elements_data=cell_to_elements_data_gpu,
        cell_neighbors=cell_neighbors_gpu,
        morton_offset=morton_offset,
        morton_max_coord=morton_max_coord,
        max_depth=max_depth,
        bbox_min=bbox_min_gpu,
        bbox_max=bbox_max_gpu,
        level_cell_sizes=level_cell_sizes_gpu,
        n_cells=n_cells,
        n_elements=n_elements,
        mean_elements_per_cell=mean_elements_per_cell,
    )

    if verbose:
        # Compute memory usage
        memory_mb = (
            connectivity_gpu.nbytes +
            node_positions_gpu.nbytes +
            cell_morton_codes_gpu.nbytes +
            cell_levels_gpu.nbytes +
            cell_sizes_gpu.nbytes +
            cell_grid_indices_gpu.nbytes +
            cell_to_elements_offsets_gpu.nbytes +
            cell_to_elements_data_gpu.nbytes +
            cell_neighbors_gpu.nbytes
        ) / (1024 ** 2)

        print(f"  GPU memory: {memory_mb:.1f} MB")
        print(f"    Base octree: {memory_mb - cell_neighbors_gpu.nbytes / (1024**2):.1f} MB")
        print(f"    Neighbor table: {cell_neighbors_gpu.nbytes / (1024**2):.1f} MB")
        print(f"  Cells: {octree_with_neighbors.n_cells:,}")
        print(f"  Elements: {octree_with_neighbors.n_elements:,}")
        print(f"  Mean elements/cell: {octree_with_neighbors.elements_per_cell_mean:.1f}")
        print(f"{'='*80}\n")

    return octree_gpu
