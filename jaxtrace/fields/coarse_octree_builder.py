#!/usr/bin/env python3
"""
Coarse Octree Builder for AMR Data.

Builds the static coarse octree structure (levels 0-6) from the first
few refinement timesteps. This structure is shared across all timesteps
during the revolution cycle.

Key insight: The coarse structure represents the basic tetrahedral mesh
before elements start splitting. It remains stable during tool rotation.
"""

import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import vtk

from .shared_coarse_octree import OctreeCoarseLevels


@dataclass
class MeshData:
    """Simplified mesh data for octree building."""
    points: np.ndarray  # [n_points, 3]
    cells: np.ndarray   # [n_cells, 4] - tetrahedral connectivity
    bbox_min: np.ndarray  # [3]
    bbox_max: np.ndarray  # [3]


def load_mesh_from_pvtu(filepath: str) -> MeshData:
    """
    Load tetrahedral mesh from PVTU file.

    Args:
        filepath: Path to .pvtu file

    Returns:
        MeshData with points, cells, and bounding box
    """
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(filepath)
    reader.Update()
    mesh = reader.GetOutput()

    # Extract points
    n_points = mesh.GetNumberOfPoints()
    points = np.zeros((n_points, 3), dtype=np.float32)
    for i in range(n_points):
        points[i] = mesh.GetPoint(i)

    # Extract cells (assuming tetrahedra)
    n_cells = mesh.GetNumberOfCells()
    cells = np.zeros((n_cells, 4), dtype=np.int32)
    for i in range(n_cells):
        cell = mesh.GetCell(i)
        if cell.GetNumberOfPoints() == 4:  # Tetrahedral
            for j in range(4):
                cells[i, j] = cell.GetPointId(j)

    # Compute bounding box
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)

    return MeshData(
        points=points,
        cells=cells,
        bbox_min=bbox_min,
        bbox_max=bbox_max
    )


def compute_cell_centers(mesh: MeshData) -> np.ndarray:
    """
    Compute center points of all tetrahedral cells (VECTORIZED).

    Args:
        mesh: Mesh data

    Returns:
        cell_centers: [n_cells, 3]
    """
    # Vectorized: Cell center is average of 4 vertices
    # mesh.points[mesh.cells] creates array of shape [n_cells, 4, 3]
    # .mean(axis=1) averages over the 4 vertices -> [n_cells, 3]
    centers = mesh.points[mesh.cells].mean(axis=1).astype(np.float32)
    return centers


def build_octree_node(
    cell_centers: np.ndarray,
    cell_indices: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    current_level: int,
    max_level: int,
    max_cells_per_node: int,
    nodes: List[dict]
) -> int:
    """
    Recursively build octree nodes.

    Args:
        cell_centers: Centers of all cells [n_cells, 3]
        cell_indices: Indices of cells in this node [n_node_cells]
        bbox_min: Node bounding box minimum [3]
        bbox_max: Node bounding box maximum [3]
        current_level: Current depth in tree
        max_level: Maximum depth to build
        max_cells_per_node: Max cells before subdivision
        nodes: List to accumulate node data

    Returns:
        node_index: Index of this node in nodes list
    """
    node_idx = len(nodes)
    center = (bbox_min + bbox_max) / 2
    size = np.max(bbox_max - bbox_min)

    # Create node
    node = {
        'center': center,
        'size': size,
        'level': current_level,
        'cells': cell_indices,
        'children': [-1] * 8  # 8 children for octree
    }

    # Check termination conditions
    if current_level >= max_level or len(cell_indices) <= max_cells_per_node:
        nodes.append(node)
        return node_idx

    # Subdivide into 8 octants (VECTORIZED)
    # Get centers for cells in this node
    node_cell_centers = cell_centers[cell_indices]

    # Compute octant for each cell using vectorized operations
    octant_bits = (
        ((node_cell_centers[:, 0] > center[0]).astype(np.int32) << 2) +
        ((node_cell_centers[:, 1] > center[1]).astype(np.int32) << 1) +
        ((node_cell_centers[:, 2] > center[2]).astype(np.int32))
    )

    # Group cells by octant
    children_indices = [cell_indices[octant_bits == i] for i in range(8)]

    # Recursively build children
    for octant in range(8):
        if len(children_indices[octant]) == 0:
            continue

        # Compute child bounding box
        child_min = bbox_min.copy()
        child_max = bbox_max.copy()

        if octant & 4:  # x-positive
            child_min[0] = center[0]
        else:
            child_max[0] = center[0]

        if octant & 2:  # y-positive
            child_min[1] = center[1]
        else:
            child_max[1] = center[1]

        if octant & 1:  # z-positive
            child_min[2] = center[2]
        else:
            child_max[2] = center[2]

        # Build child (children_indices[octant] is already numpy array)
        child_idx = build_octree_node(
            cell_centers,
            children_indices[octant],
            child_min,
            child_max,
            current_level + 1,
            max_level,
            max_cells_per_node,
            nodes
        )
        node['children'][octant] = child_idx

    nodes.append(node)
    return node_idx


def build_coarse_octree(
    mesh: MeshData,
    n_coarse_levels: int = 6,
    max_cells_per_node: int = 32
) -> OctreeCoarseLevels:
    """
    Build static coarse octree structure from mesh.

    This creates the shared octree structure (levels 0 to n_coarse_levels-1)
    that will be reused across all timesteps.

    Args:
        mesh: Mesh data from refinement timesteps
        n_coarse_levels: Number of coarse levels to build
        max_cells_per_node: Maximum cells per node before subdivision

    Returns:
        OctreeCoarseLevels: Static coarse octree structure
    """
    # Compute cell centers
    cell_centers = compute_cell_centers(mesh)

    # Build octree recursively
    nodes = []
    all_cell_indices = np.arange(len(mesh.cells), dtype=np.int32)

    build_octree_node(
        cell_centers,
        all_cell_indices,
        mesh.bbox_min,
        mesh.bbox_max,
        current_level=0,
        max_level=n_coarse_levels - 1,  # Build up to coarse depth
        max_cells_per_node=max_cells_per_node,
        nodes=nodes
    )

    # Convert to JAX arrays
    n_nodes = len(nodes)
    max_elements = max_cells_per_node

    node_centers = np.zeros((n_nodes, 3), dtype=np.float32)
    node_sizes = np.zeros(n_nodes, dtype=np.float32)
    node_levels = np.zeros(n_nodes, dtype=np.int32)
    node_children = np.full((n_nodes, 8), -1, dtype=np.int32)
    node_element_lists = np.full((n_nodes, max_elements), -1, dtype=np.int32)
    node_element_counts = np.zeros(n_nodes, dtype=np.int32)

    for i, node in enumerate(nodes):
        node_centers[i] = node['center']
        node_sizes[i] = node['size']
        node_levels[i] = node['level']
        node_children[i] = node['children']

        # Store cell indices
        cells = node['cells']
        n_cells = min(len(cells), max_elements)
        node_element_lists[i, :n_cells] = cells[:n_cells]
        node_element_counts[i] = n_cells

    return OctreeCoarseLevels(
        bbox_min=jnp.array(mesh.bbox_min),
        bbox_max=jnp.array(mesh.bbox_max),
        node_centers=jnp.array(node_centers),
        node_sizes=jnp.array(node_sizes),
        node_levels=jnp.array(node_levels),
        node_children=jnp.array(node_children),
        node_element_lists=jnp.array(node_element_lists),
        node_element_counts=jnp.array(node_element_counts),
        n_coarse_levels=n_coarse_levels,
        max_elements_per_node=max_elements
    )


def build_coarse_octree_from_refinement_steps(
    mesh_files: List[str],
    n_coarse_levels: int = 6,
    max_cells_per_node: int = 32
) -> OctreeCoarseLevels:
    """
    Build coarse octree from multiple refinement timesteps.

    Uses the LAST refinement timestep (most refined) to build the coarse
    structure, as it contains all the elements that will be present.

    Args:
        mesh_files: List of mesh files from refinement phase
        n_coarse_levels: Number of coarse levels to build
        max_cells_per_node: Maximum cells per node

    Returns:
        OctreeCoarseLevels: Static coarse octree structure
    """
    # Load the last (most refined) mesh
    print(f"Building coarse octree from {len(mesh_files)} refinement steps...")
    print(f"Loading most refined mesh: {mesh_files[-1]}")

    mesh = load_mesh_from_pvtu(mesh_files[-1])

    print(f"Mesh: {len(mesh.points)} points, {len(mesh.cells)} cells")
    print(f"Building coarse octree (levels 0-{n_coarse_levels-1})...")

    coarse_octree = build_coarse_octree(
        mesh,
        n_coarse_levels=n_coarse_levels,
        max_cells_per_node=max_cells_per_node
    )

    n_nodes = len(coarse_octree.node_centers)
    memory_mb = coarse_octree.get_memory_size() / (1024 ** 2)

    print(f"Coarse octree built: {n_nodes} nodes, {memory_mb:.2f} MB")

    return coarse_octree


def find_refinement_files(
    all_files: List[str],
    n_refinement_steps: Optional[int] = None
) -> List[str]:
    """
    Find refinement phase files from all timestep files.

    If n_refinement_steps is None, auto-detect by looking for mesh size changes.

    Args:
        all_files: All mesh files sorted by timestep
        n_refinement_steps: Number of refinement steps (or None for auto-detect)

    Returns:
        refinement_files: List of files from refinement phase
    """
    if n_refinement_steps is not None:
        return all_files[:n_refinement_steps]

    # Auto-detect: find where mesh size stabilizes
    print("Auto-detecting refinement steps...")

    prev_n_points = None
    stable_count = 0
    refinement_end = len(all_files)

    for i, filepath in enumerate(all_files[:20]):  # Check first 20
        reader = vtk.vtkXMLPUnstructuredGridReader()
        reader.SetFileName(filepath)
        reader.UpdateInformation()
        reader.Update()
        mesh = reader.GetOutput()
        n_points = mesh.GetNumberOfPoints()

        if prev_n_points is not None:
            change_pct = abs(n_points - prev_n_points) / prev_n_points * 100

            if change_pct < 0.5:  # Less than 0.5% change
                stable_count += 1
                if stable_count >= 3:  # 3 consecutive stable steps
                    refinement_end = i - 2
                    break
            else:
                stable_count = 0

        prev_n_points = n_points

    print(f"Detected {refinement_end} refinement steps")
    return all_files[:refinement_end]
