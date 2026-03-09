#!/usr/bin/env python3
"""
Visualize Mesh-Octree Alignment for ParaView

Extracts sample tetrahedral elements of different sizes (from refined regions)
and their corresponding octree cells, exports to VTK for visualization.

Output files:
- mesh_tets_sample.vtu: Tetrahedral elements (colored by size/level)
- octree_cells_sample.vtu: Cube cells (colored by level)
- visualization.pvd: ParaView collection file

Usage:
    python3 visualize_mesh_octree_alignment.py
    # Open visualization.pvd in ParaView
"""

import numpy as np
from pathlib import Path
from jaxtrace.gpu.search.mesh_aligned_octree_fast import (
    extract_octree_cells_fast,
    find_axis_aligned_edges_fast,
)
import vtk
from vtk.util import numpy_support


# ============================================================================
# Configuration
# ============================================================================

MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME = 'Displacement'

OUTPUT_DIR = Path("./visualization_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_mesh():
    """Load and deduplicate mesh."""
    print("Loading mesh...")
    from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
    from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=True
    )

    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions,
        connectivity,
        velocity_sequence=velocity_sequence,
        verbose=True
    )

    return node_positions, connectivity


def select_diverse_elements(node_positions, connectivity, n_samples=50):
    """
    Select diverse elements from different refinement levels.

    Returns:
        elem_ids: array of selected element IDs
        elem_levels: array of octree levels for each element
        elem_sizes: array of average edge lengths for each element
    """
    print(f"\nSelecting {n_samples} diverse elements from different refinement levels...")

    n_elements = connectivity.shape[0]

    # Sample elements and compute their properties
    sample_size = min(10000, n_elements)
    sample_indices = np.random.choice(n_elements, sample_size, replace=False)

    elem_properties = []

    for elem_id in sample_indices:
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]

        # Find axis-aligned edges to determine level
        cell_size, level = find_axis_aligned_edges_fast(vertices)
        avg_size = np.mean(cell_size[cell_size > 0])

        if avg_size > 0:
            elem_properties.append((elem_id, level, avg_size))

    # Sort by size to get diversity
    elem_properties.sort(key=lambda x: x[2])

    # Select elements from different size bins
    n_bins = min(5, len(elem_properties) // 10)
    bin_size = len(elem_properties) // n_bins

    selected = []
    for bin_idx in range(n_bins):
        start = bin_idx * bin_size
        end = start + bin_size
        # Take n_samples // n_bins elements from each bin
        n_from_bin = n_samples // n_bins
        bin_elements = elem_properties[start:end]
        selected.extend(bin_elements[:n_from_bin])

    # Sort by element ID for consistent output
    selected.sort(key=lambda x: x[0])

    elem_ids = np.array([e[0] for e in selected], dtype=np.int32)
    elem_levels = np.array([e[1] for e in selected], dtype=np.int32)
    elem_sizes = np.array([e[2] for e in selected], dtype=np.float64)

    print(f"  Selected {len(elem_ids)} elements:")
    print(f"    Level range: {elem_levels.min()} - {elem_levels.max()}")
    print(f"    Size range: {elem_sizes.min():.2e} - {elem_sizes.max():.2e}")

    return elem_ids, elem_levels, elem_sizes


def extract_octree_cells_for_elements(node_positions, connectivity, elem_ids, cells):
    """
    Extract octree cells that the selected elements overlap.

    Returns:
        cell_indices: array of unique cell indices
        cell_morton_codes: array of Morton codes
        cell_levels: array of levels
        cell_grid_indices: array of (i,j,k) grid positions
        cell_sizes: array of (dx,dy,dz) sizes
    """
    print(f"\nExtracting octree cells for {len(elem_ids)} elements...")

    # Get all cells that these elements overlap
    unique_cell_indices = set()

    for elem_id in elem_ids:
        start = cells.element_to_cells_offsets[elem_id]
        end = cells.element_to_cells_offsets[elem_id + 1]
        cell_idx_list = cells.element_to_cells_data[start:end]
        unique_cell_indices.update(cell_idx_list)

    cell_indices = np.array(sorted(unique_cell_indices), dtype=np.int32)

    # Extract cell metadata
    cell_morton_codes = cells.cell_morton_codes[cell_indices]
    cell_levels = cells.cell_levels[cell_indices]
    cell_grid_indices = cells.cell_grid_indices[cell_indices]
    cell_sizes = cells.cell_sizes[cell_indices]

    print(f"  Extracted {len(cell_indices)} unique octree cells")
    print(f"    Level range: {cell_levels.min()} - {cell_levels.max()}")

    return cell_indices, cell_morton_codes, cell_levels, cell_grid_indices, cell_sizes


def create_tetrahedral_mesh_vtk(node_positions, connectivity, elem_ids, elem_levels, elem_sizes):
    """
    Create VTK unstructured grid for tetrahedral elements.

    Returns:
        vtkUnstructuredGrid with tetrahedral elements
    """
    print(f"\nCreating VTK tetrahedral mesh...")

    # Create VTK unstructured grid
    ugrid = vtk.vtkUnstructuredGrid()

    # Add points (only nodes used by selected elements)
    points = vtk.vtkPoints()
    node_map = {}  # old_node_id -> new_node_id
    new_node_id = 0

    for elem_id in elem_ids:
        node_ids = connectivity[elem_id]
        for old_node_id in node_ids:
            if old_node_id not in node_map:
                node_map[old_node_id] = new_node_id
                pos = node_positions[old_node_id]
                points.InsertNextPoint(pos[0], pos[1], pos[2])
                new_node_id += 1

    ugrid.SetPoints(points)

    # Add tetrahedra
    for elem_id in elem_ids:
        node_ids = connectivity[elem_id]
        tet = vtk.vtkTetra()
        for i, old_node_id in enumerate(node_ids):
            tet.GetPointIds().SetId(i, node_map[old_node_id])
        ugrid.InsertNextCell(tet.GetCellType(), tet.GetPointIds())

    # Add element level as cell data
    level_array = numpy_support.numpy_to_vtk(elem_levels, deep=True)
    level_array.SetName("OctreeLevel")
    ugrid.GetCellData().AddArray(level_array)

    # Add element size as cell data
    size_array = numpy_support.numpy_to_vtk(elem_sizes, deep=True)
    size_array.SetName("ElementSize")
    ugrid.GetCellData().AddArray(size_array)

    # Add element ID as cell data
    id_array = numpy_support.numpy_to_vtk(elem_ids, deep=True)
    id_array.SetName("ElementID")
    ugrid.GetCellData().AddArray(id_array)

    print(f"  Created {ugrid.GetNumberOfCells()} tetrahedra")
    print(f"  Using {ugrid.GetNumberOfPoints()} nodes")

    return ugrid


def create_octree_cells_vtk(cell_grid_indices, cell_sizes, cell_levels, cell_morton_codes):
    """
    Create VTK unstructured grid for octree cells (cubes).

    Returns:
        vtkUnstructuredGrid with hexahedral cells
    """
    print(f"\nCreating VTK octree cell mesh...")

    # Create VTK unstructured grid
    ugrid = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()

    # For each cell, create 8 corner points and a hexahedron
    point_id = 0

    for cell_idx in range(len(cell_grid_indices)):
        grid_idx = cell_grid_indices[cell_idx]
        cell_size = cell_sizes[cell_idx]

        # Compute cell bounds
        x_min = grid_idx[0] * cell_size[0]
        y_min = grid_idx[1] * cell_size[1]
        z_min = grid_idx[2] * cell_size[2]

        x_max = x_min + cell_size[0]
        y_max = y_min + cell_size[1]
        z_max = z_min + cell_size[2]

        # Create 8 corner points (VTK hexahedron ordering)
        corners = [
            (x_min, y_min, z_min),  # 0
            (x_max, y_min, z_min),  # 1
            (x_max, y_max, z_min),  # 2
            (x_min, y_max, z_min),  # 3
            (x_min, y_min, z_max),  # 4
            (x_max, y_min, z_max),  # 5
            (x_max, y_max, z_max),  # 6
            (x_min, y_max, z_max),  # 7
        ]

        # Add points
        corner_ids = []
        for corner in corners:
            points.InsertNextPoint(corner[0], corner[1], corner[2])
            corner_ids.append(point_id)
            point_id += 1

        # Create hexahedron
        hex_cell = vtk.vtkHexahedron()
        for i, cid in enumerate(corner_ids):
            hex_cell.GetPointIds().SetId(i, cid)

        ugrid.InsertNextCell(hex_cell.GetCellType(), hex_cell.GetPointIds())

    ugrid.SetPoints(points)

    # Add cell level as cell data
    level_array = numpy_support.numpy_to_vtk(cell_levels, deep=True)
    level_array.SetName("OctreeLevel")
    ugrid.GetCellData().AddArray(level_array)

    # Add Morton codes as cell data
    morton_array = numpy_support.numpy_to_vtk(cell_morton_codes, deep=True)
    morton_array.SetName("MortonCode")
    ugrid.GetCellData().AddArray(morton_array)

    # Add cell sizes as cell data
    size_x_array = numpy_support.numpy_to_vtk(cell_sizes[:, 0], deep=True)
    size_x_array.SetName("CellSizeX")
    ugrid.GetCellData().AddArray(size_x_array)

    print(f"  Created {ugrid.GetNumberOfCells()} hexahedral cells")
    print(f"  Using {ugrid.GetNumberOfPoints()} corner points")

    return ugrid


def write_vtk_file(ugrid, filename):
    """Write VTK unstructured grid to file."""
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(filename))
    writer.SetInputData(ugrid)
    writer.Write()
    print(f"  Wrote: {filename}")


def create_pvd_collection(files_dict, output_path):
    """
    Create ParaView collection file (.pvd) for easy loading.

    Args:
        files_dict: dict of {name: vtu_filename}
        output_path: path to output .pvd file
    """
    pvd_content = '<?xml version="1.0"?>\n'
    pvd_content += '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
    pvd_content += '  <Collection>\n'

    for name, vtu_file in files_dict.items():
        pvd_content += f'    <DataSet name="{name}" file="{vtu_file.name}"/>\n'

    pvd_content += '  </Collection>\n'
    pvd_content += '</VTKFile>\n'

    output_path.write_text(pvd_content)
    print(f"\n  Wrote ParaView collection: {output_path}")


def main():
    print("="*80)
    print("Mesh-Octree Alignment Visualization")
    print("="*80)

    # Load mesh
    node_positions, connectivity = load_mesh()

    # Extract octree cells
    print("\nExtracting octree cells from mesh...")
    cells = extract_octree_cells_fast(
        node_positions,
        connectivity,
        tolerance=1e-6,
        batch_size=100000,
        verbose=False
    )
    print(f"  Extracted {cells.n_cells:,} octree cells")

    # Select diverse sample elements
    elem_ids, elem_levels, elem_sizes = select_diverse_elements(
        node_positions, connectivity, n_samples=50
    )

    # Extract octree cells for these elements
    cell_indices, cell_morton_codes, cell_levels, cell_grid_indices, cell_sizes = \
        extract_octree_cells_for_elements(node_positions, connectivity, elem_ids, cells)

    # Create VTK meshes
    tet_mesh = create_tetrahedral_mesh_vtk(
        node_positions, connectivity, elem_ids, elem_levels, elem_sizes
    )

    octree_mesh = create_octree_cells_vtk(
        cell_grid_indices, cell_sizes, cell_levels, cell_morton_codes
    )

    # Write VTK files
    print(f"\nWriting VTK files to {OUTPUT_DIR}...")
    tet_file = OUTPUT_DIR / "mesh_tets_sample.vtu"
    octree_file = OUTPUT_DIR / "octree_cells_sample.vtu"

    write_vtk_file(tet_mesh, tet_file)
    write_vtk_file(octree_mesh, octree_file)

    # Create ParaView collection file
    pvd_file = OUTPUT_DIR / "visualization.pvd"
    create_pvd_collection({
        "Tetrahedral Elements": tet_file,
        "Octree Cells": octree_file,
    }, pvd_file)

    # Print summary
    print(f"\n{'='*80}")
    print("Visualization Export Complete!")
    print(f"{'='*80}")
    print(f"\nOutput files in: {OUTPUT_DIR}/")
    print(f"  - mesh_tets_sample.vtu: {len(elem_ids)} tetrahedral elements")
    print(f"  - octree_cells_sample.vtu: {len(cell_indices)} octree cubes")
    print(f"  - visualization.pvd: ParaView collection file")

    print(f"\nTo visualize in ParaView:")
    print(f"  1. Open: {pvd_file}")
    print(f"  2. Apply filters:")
    print(f"     - Tetrahedral Elements:")
    print(f"       • Color by 'OctreeLevel' (categorical)")
    print(f"       • Representation: Surface with edges")
    print(f"       • Opacity: 1.0")
    print(f"     - Octree Cells:")
    print(f"       • Color by 'OctreeLevel' (same colormap)")
    print(f"       • Representation: Wireframe or Surface")
    print(f"       • Opacity: 0.3 (semi-transparent)")
    print(f"  3. You'll see tetrahedra spanning across multiple cube boundaries!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
