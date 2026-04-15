#!/usr/bin/env python3
"""
Mesh Loader for Flat Array Structures

Loads tetrahedral meshes from PVTU files and converts to flat array format
optimized for JAX/GPU.

Phase 1.2 of V3 Plan

Key Functions:
- load_mesh_from_pvtu: Load positions and connectivity
- build_element_neighbors: Compute face adjacency (Level 1 search)
- assign_elements_to_blocks: Spatial partitioning (Level 2 search)
"""

from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
import vtk
from vtk.util import numpy_support

try:
    from .flat_arrays import MeshData, BlockPartitionData, create_mesh_data
except ImportError:
    # Standalone execution
    from flat_arrays import MeshData, BlockPartitionData, create_mesh_data


def load_mesh_from_pvtu(
    mesh_path: Path,
    field_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load mesh from PVTU file.

    Args:
        mesh_path: Path to PVTU file or directory containing PVTU
        field_name: Optional velocity field name to load

    Returns:
        positions: (N_nodes, 3) float64
        connectivity: (N_elements, 4) int32
        velocities: (N_nodes, 3) float64 or None
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

    # Fail early and clearly if the file is missing rather than letting VTK
    # return a null output that crashes later with 'NoneType has no GetData'.
    if not Path(mesh_file).exists():
        raise FileNotFoundError(f"PVTU file not found: {mesh_file}")

    # Load with VTK
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(str(mesh_file))
    reader.Update()
    output = reader.GetOutput()
    if output is None or output.GetPoints() is None:
        raise RuntimeError(
            f"VTK failed to read {mesh_file}: got empty output. "
            f"Check that the file and its linked .vtu pieces are present and readable."
        )

    # Extract positions
    positions = numpy_support.vtk_to_numpy(output.GetPoints().GetData())
    positions = positions.astype(np.float64)
    print(f"  Nodes: {positions.shape[0]:,}")

    # Extract connectivity (assume all tetrahedral)
    n_cells = output.GetNumberOfCells()
    connectivity_data = numpy_support.vtk_to_numpy(output.GetCells().GetData())

    # Parse connectivity (VTK format: [4, node0, node1, node2, node3, 4, ...])
    connectivity = np.zeros((n_cells, 4), dtype=np.int32)
    for i in range(n_cells):
        connectivity[i] = connectivity_data[i * 5 + 1 : i * 5 + 5]

    print(f"  Elements: {n_cells:,}")

    # Load velocity field if requested
    velocities = None
    if field_name is not None:
        point_data = output.GetPointData()
        if point_data.HasArray(field_name):
            velocities = numpy_support.vtk_to_numpy(point_data.GetArray(field_name))
            velocities = velocities.astype(np.float64)
            print(f"  Loaded field '{field_name}': {velocities.shape}")
        else:
            print(f"  Warning: Field '{field_name}' not found")
            available = [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]
            print(f"  Available fields: {available}")

    return positions, connectivity, velocities


def build_element_neighbors(
    connectivity: np.ndarray,
    verbose: bool = True
) -> np.ndarray:
    """
    Build element neighbor adjacency list.

    Two elements are neighbors if they share a triangular face (3 nodes).
    For tetrahedral meshes, each element has up to 4 neighbors.

    Args:
        connectivity: (N_elements, 4) int32
        verbose: Print progress

    Returns:
        element_neighbors: (N_elements, 4) int32
            Each row contains indices of 4 neighboring elements.
            Padded with -1 for boundary faces (no neighbor).
    """
    n_elements = connectivity.shape[0]

    if verbose:
        print(f"\nBuilding element neighbors for {n_elements:,} elements...")

    # Initialize with -1 (no neighbor)
    element_neighbors = np.full((n_elements, 4), -1, dtype=np.int32)

    # Build face -> element mapping
    # Each tetrahedron has 4 faces
    face_to_elements = {}

    if verbose:
        print("  Step 1/2: Extracting faces...")

    for elem_id in range(n_elements):
        if verbose and elem_id % 500000 == 0 and elem_id > 0:
            print(f"    Processed {elem_id:,} / {n_elements:,} ({100*elem_id/n_elements:.1f}%)")

        nodes = connectivity[elem_id]

        # 4 faces of tetrahedron (opposite to each vertex)
        # Face opposite to vertex i contains the other 3 vertices
        faces = [
            tuple(sorted([nodes[1], nodes[2], nodes[3]])),  # Face 0: opposite vertex 0
            tuple(sorted([nodes[0], nodes[2], nodes[3]])),  # Face 1: opposite vertex 1
            tuple(sorted([nodes[0], nodes[1], nodes[3]])),  # Face 2: opposite vertex 2
            tuple(sorted([nodes[0], nodes[1], nodes[2]])),  # Face 3: opposite vertex 3
        ]

        for face_idx, face in enumerate(faces):
            if face not in face_to_elements:
                face_to_elements[face] = []
            face_to_elements[face].append((elem_id, face_idx))

    if verbose:
        print("  Step 2/2: Building neighbor lists...")

    # Process faces to find neighbors
    for face_id, (face, elements) in enumerate(face_to_elements.items()):
        if verbose and face_id % 1000000 == 0 and face_id > 0:
            print(f"    Processed {face_id:,} / {len(face_to_elements):,} faces")

        if len(elements) == 2:
            # Interior face: two elements share it
            (elem0, face_idx0), (elem1, face_idx1) = elements

            # Set neighbors
            element_neighbors[elem0, face_idx0] = elem1
            element_neighbors[elem1, face_idx1] = elem0

        elif len(elements) == 1:
            # Boundary face: only one element (already -1)
            pass
        else:
            # Non-manifold: more than 2 elements share face (should not happen)
            print(f"  Warning: Non-manifold face {face} with {len(elements)} elements")

    if verbose:
        # Statistics
        n_boundary_faces = np.sum(element_neighbors == -1)
        n_interior_faces = np.sum(element_neighbors >= 0)

        print(f"\n  Neighbor statistics:")
        print(f"    Interior faces: {n_interior_faces:,}")
        print(f"    Boundary faces: {n_boundary_faces:,}")
        print(f"    Elements with 0 neighbors: {np.sum(np.all(element_neighbors == -1, axis=1)):,}")
        print(f"    Elements with 4 neighbors: {np.sum(np.all(element_neighbors >= 0, axis=1)):,}")

    return element_neighbors


def assign_elements_to_blocks(
    positions: np.ndarray,
    connectivity: np.ndarray,
    grid_size: Tuple[int, int, int],
    verbose: bool = True
) -> Tuple[np.ndarray, BlockPartitionData]:
    """
    Assign elements to spatial blocks.

    Uses element centroids and a regular grid partitioning.

    Args:
        positions: (N_nodes, 3) float64
        connectivity: (N_elements, 4) int32
        grid_size: (nx, ny, nz) number of blocks per dimension
        verbose: Print progress

    Returns:
        element_block_IDs: (N_elements,) int32 - block ID for each element
        partition_data: BlockPartitionData object
    """
    n_elements = connectivity.shape[0]
    n_blocks = np.prod(grid_size)

    if verbose:
        print(f"\nAssigning {n_elements:,} elements to {grid_size[0]}×{grid_size[1]}×{grid_size[2]} = {n_blocks} blocks...")

    # Compute bounding box
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    block_size = (bbox_max - bbox_min) / np.array(grid_size)

    if verbose:
        print(f"  Bounding box: [{bbox_min[0]:.6f}, {bbox_min[1]:.6f}, {bbox_min[2]:.6f}] to "
              f"[{bbox_max[0]:.6f}, {bbox_max[1]:.6f}, {bbox_max[2]:.6f}]")
        print(f"  Block size: [{block_size[0]:.6f}, {block_size[1]:.6f}, {block_size[2]:.6f}]")

    # Compute element centroids
    if verbose:
        print("  Computing element centroids...")

    centroids = np.zeros((n_elements, 3), dtype=np.float64)
    for i in range(n_elements):
        if verbose and i % 500000 == 0 and i > 0:
            print(f"    Processed {i:,} / {n_elements:,} ({100*i/n_elements:.1f}%)")
        centroids[i] = positions[connectivity[i]].mean(axis=0)

    # Assign to blocks
    if verbose:
        print("  Assigning elements to blocks...")

    element_block_IDs = np.zeros(n_elements, dtype=np.int32)
    elements_per_block = np.zeros(n_blocks, dtype=np.int32)

    for i in range(n_elements):
        if verbose and i % 500000 == 0 and i > 0:
            print(f"    Processed {i:,} / {n_elements:,} ({100*i/n_elements:.1f}%)")

        centroid = centroids[i]

        # Compute block indices
        block_idx = np.floor((centroid - bbox_min) / block_size).astype(np.int32)
        block_idx = np.clip(block_idx, 0, np.array(grid_size) - 1)

        # Convert to flat block ID
        block_id = (
            block_idx[0] * grid_size[1] * grid_size[2] +
            block_idx[1] * grid_size[2] +
            block_idx[2]
        )

        element_block_IDs[i] = block_id
        elements_per_block[block_id] += 1

    # Create partition data
    partition_data = BlockPartitionData(
        grid_size=grid_size,
        n_blocks=n_blocks,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        block_size=block_size,
        elements_per_block=elements_per_block,
    )

    if verbose:
        print(f"\n  Block statistics:")
        print(f"    Non-empty blocks: {np.sum(elements_per_block > 0)} / {n_blocks}")
        print(f"    Elements per block:")
        print(f"      Min: {elements_per_block[elements_per_block > 0].min():,}")
        print(f"      Max: {elements_per_block.max():,}")
        print(f"      Mean: {elements_per_block[elements_per_block > 0].mean():.0f}")
        print(f"    Load imbalance: {partition_data.load_imbalance_factor():.2f}×")

    return element_block_IDs, partition_data


def load_mesh_complete(
    mesh_path: Path,
    grid_size: Tuple[int, int, int] = (2, 2, 1),
    field_name: Optional[str] = None,
    device: str = "cpu",
    verbose: bool = True
) -> Tuple[MeshData, BlockPartitionData]:
    """
    Complete mesh loading pipeline.

    Loads mesh, builds neighbors, assigns blocks, and creates MeshData.

    Args:
        mesh_path: Path to PVTU file or directory
        grid_size: (nx, ny, nz) block grid
        field_name: Optional velocity field to load
        device: "cpu" or "gpu"
        verbose: Print progress

    Returns:
        mesh_data: MeshData object (on specified device)
        partition_data: BlockPartitionData object
    """
    if verbose:
        print("=" * 80)
        print("LOADING MESH TO FLAT ARRAYS")
        print("=" * 80)

    # Step 1: Load from file
    positions, connectivity, velocities = load_mesh_from_pvtu(mesh_path, field_name)

    # Step 2: Build element neighbors
    element_neighbors = build_element_neighbors(connectivity, verbose=verbose)

    # Step 3: Assign to blocks
    element_block_IDs, partition_data = assign_elements_to_blocks(
        positions, connectivity, grid_size, verbose=verbose
    )

    # Step 4: Create MeshData
    if verbose:
        print(f"\nCreating MeshData on device: {device}")

    mesh_data = create_mesh_data(
        positions=positions,
        connectivity=connectivity,
        element_neighbors=element_neighbors,
        element_block_IDs=element_block_IDs,
        velocities=velocities,
        device=device,
    )

    if verbose:
        print("\n" + str(mesh_data))
        print("\n" + str(partition_data))
        print("=" * 80)

    return mesh_data, partition_data


if __name__ == "__main__":
    import sys

    # Test with ThreadedA mesh if available
    threadeda_path = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule")

    if threadeda_path.exists():
        print("Testing with ThreadedA mesh...")
        mesh_data, partition_data = load_mesh_complete(
            threadeda_path,
            grid_size=(2, 2, 1),
            field_name=None,
            device="cpu",
            verbose=True,
        )
        print("\n✅ Mesh loading successful!")
    else:
        print("ThreadedA mesh not found, testing with synthetic mesh...")
        try:
            from .test_meshes import generate_test_mesh, SMALL_BALANCED_MESH
        except ImportError:
            from test_meshes import generate_test_mesh, SMALL_BALANCED_MESH

        positions, connectivity = generate_test_mesh(SMALL_BALANCED_MESH)

        # Build neighbors and blocks
        element_neighbors = build_element_neighbors(connectivity)
        element_block_IDs, partition_data = assign_elements_to_blocks(
            positions, connectivity, (2, 2, 1)
        )

        # Create mesh data
        mesh_data = create_mesh_data(
            positions, connectivity, element_neighbors, element_block_IDs
        )

        print("\n" + str(mesh_data))
        print("\n" + str(partition_data))
        print("\n✅ Mesh loading successful!")
