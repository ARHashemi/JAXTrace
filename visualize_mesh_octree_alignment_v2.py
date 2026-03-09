#!/usr/bin/env python3
"""
Improved Visualization: Show exact 2×2×2 cell pattern for individual tetrahedra

Creates separate visualizations:
1. Single element with its exact 8 cells (2×2×2 pattern)
2. Multiple elements showing different refinement levels
3. Cross-section view showing how tets cross boundaries
"""

import numpy as np
from pathlib import Path
from jaxtrace.gpu.search.mesh_aligned_octree_fast import (
    extract_octree_cells_fast,
    find_axis_aligned_edges_fast,
    compute_8cell_pattern,
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
        verbose=False
    )

    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions,
        connectivity,
        velocity_sequence=velocity_sequence,
        verbose=False
    )

    return node_positions, connectivity


def create_single_tet_with_8cells_vtk(elem_id, node_positions, connectivity):
    """
    Create visualization for ONE tetrahedron with its exact 8 overlapping cells.

    This clearly shows the 2×2×2 pattern.
    """
    print(f"\nCreating single-element visualization (elem {elem_id})...")

    # Get element vertices
    node_ids = connectivity[elem_id]
    vertices = node_positions[node_ids]

    # Find cell size
    cell_size, level = find_axis_aligned_edges_fast(vertices)

    # Compute bbox and 8-cell pattern
    bbox_min = vertices.min(axis=0) - 1e-6
    bbox_max = vertices.max(axis=0) + 1e-6

    grid_indices, morton_codes, level = compute_8cell_pattern(bbox_min, bbox_max, cell_size)

    print(f"  Element spans {len(morton_codes)} cells (2×2×2 pattern)")
    print(f"  Cell size: {cell_size}")
    print(f"  Octree level: {level}")

    # Create tetrahedral mesh
    tet_ugrid = vtk.vtkUnstructuredGrid()
    tet_points = vtk.vtkPoints()

    for i, vtx in enumerate(vertices):
        tet_points.InsertNextPoint(vtx[0], vtx[1], vtx[2])

    tet_ugrid.SetPoints(tet_points)

    tet = vtk.vtkTetra()
    for i in range(4):
        tet.GetPointIds().SetId(i, i)
    tet_ugrid.InsertNextCell(tet.GetCellType(), tet.GetPointIds())

    # Add element ID
    id_array = vtk.vtkIntArray()
    id_array.SetName("ElementID")
    id_array.InsertNextValue(elem_id)
    tet_ugrid.GetCellData().AddArray(id_array)

    # Create octree cell mesh (8 cubes)
    cell_ugrid = vtk.vtkUnstructuredGrid()
    cell_points = vtk.vtkPoints()

    cell_ids = []
    point_id = 0

    for idx, grid_idx in enumerate(grid_indices):
        # Compute cube bounds
        x_min = grid_idx[0] * cell_size[0]
        y_min = grid_idx[1] * cell_size[1]
        z_min = grid_idx[2] * cell_size[2]

        x_max = x_min + cell_size[0]
        y_max = y_min + cell_size[1]
        z_max = z_min + cell_size[2]

        # 8 corners of cube
        corners = [
            (x_min, y_min, z_min), (x_max, y_min, z_min),
            (x_max, y_max, z_min), (x_min, y_max, z_min),
            (x_min, y_min, z_max), (x_max, y_min, z_max),
            (x_max, y_max, z_max), (x_min, y_max, z_max),
        ]

        corner_ids = []
        for corner in corners:
            cell_points.InsertNextPoint(corner[0], corner[1], corner[2])
            corner_ids.append(point_id)
            point_id += 1

        # Create hexahedron
        hex_cell = vtk.vtkHexahedron()
        for i, cid in enumerate(corner_ids):
            hex_cell.GetPointIds().SetId(i, cid)
        cell_ugrid.InsertNextCell(hex_cell.GetCellType(), hex_cell.GetPointIds())

        cell_ids.append(idx)

    cell_ugrid.SetPoints(cell_points)

    # Add cell index (0-7 for 2×2×2 pattern)
    cell_id_array = numpy_support.numpy_to_vtk(np.array(cell_ids, dtype=np.int32))
    cell_id_array.SetName("CellIndex")
    cell_ugrid.GetCellData().AddArray(cell_id_array)

    # Add grid position as label
    grid_x = numpy_support.numpy_to_vtk(grid_indices[:, 0])
    grid_x.SetName("GridX")
    cell_ugrid.GetCellData().AddArray(grid_x)

    grid_y = numpy_support.numpy_to_vtk(grid_indices[:, 1])
    grid_y.SetName("GridY")
    cell_ugrid.GetCellData().AddArray(grid_y)

    grid_z = numpy_support.numpy_to_vtk(grid_indices[:, 2])
    grid_z.SetName("GridZ")
    cell_ugrid.GetCellData().AddArray(grid_z)

    return tet_ugrid, cell_ugrid, level


def create_element_bbox_mesh(vertices):
    """Create axis-aligned bounding box mesh for an element."""
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)

    ugrid = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()

    # 8 corners of bbox
    corners = [
        (bbox_min[0], bbox_min[1], bbox_min[2]),
        (bbox_max[0], bbox_min[1], bbox_min[2]),
        (bbox_max[0], bbox_max[1], bbox_min[2]),
        (bbox_min[0], bbox_max[1], bbox_min[2]),
        (bbox_min[0], bbox_min[1], bbox_max[2]),
        (bbox_max[0], bbox_min[1], bbox_max[2]),
        (bbox_max[0], bbox_max[1], bbox_max[2]),
        (bbox_min[0], bbox_max[1], bbox_max[2]),
    ]

    for corner in corners:
        points.InsertNextPoint(corner[0], corner[1], corner[2])

    ugrid.SetPoints(points)

    # Create hex cell
    hex_cell = vtk.vtkHexahedron()
    for i in range(8):
        hex_cell.GetPointIds().SetId(i, i)
    ugrid.InsertNextCell(hex_cell.GetCellType(), hex_cell.GetPointIds())

    return ugrid


def find_good_example_elements(node_positions, connectivity, n_examples=3):
    """
    Find good example elements that clearly show boundary crossing.

    Criteria:
    - Different refinement levels
    - Well-formed (not degenerate)
    - Bbox clearly spans 2×2×2 cells
    """
    print("\nSearching for good example elements...")

    n_elements = connectivity.shape[0]
    sample_size = min(5000, n_elements)
    sample_indices = np.random.choice(n_elements, sample_size, replace=False)

    candidates = []

    for elem_id in sample_indices:
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]

        cell_size, level = find_axis_aligned_edges_fast(vertices)

        if np.any(cell_size == 0):
            continue

        # Check that bbox spans exactly 2 cells in each dimension
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)

        cell_size_safe = np.where(cell_size > 1e-6, cell_size, 1.0)

        i_min = int(np.floor(bbox_min[0] / cell_size_safe[0]))
        i_max = int(np.floor(bbox_max[0] / cell_size_safe[0]))
        j_min = int(np.floor(bbox_min[1] / cell_size_safe[1]))
        j_max = int(np.floor(bbox_max[1] / cell_size_safe[1]))
        k_min = int(np.floor(bbox_min[2] / cell_size_safe[2]))
        k_max = int(np.floor(bbox_max[2] / cell_size_safe[2]))

        # Count cells spanned
        n_cells_x = i_max - i_min + 1
        n_cells_y = j_max - j_min + 1
        n_cells_z = k_max - k_min + 1

        total_cells = n_cells_x * n_cells_y * n_cells_z

        # Prefer elements that span exactly 8 cells (2×2×2)
        if total_cells == 8:
            avg_size = np.mean(cell_size)
            candidates.append((elem_id, level, avg_size, total_cells))

    if not candidates:
        print("  Warning: No ideal candidates found, using fallback")
        return sample_indices[:n_examples]

    # Sort by size to get diversity
    candidates.sort(key=lambda x: x[2])

    # Select from different size ranges
    step = max(1, len(candidates) // n_examples)
    selected = [candidates[i * step][0] for i in range(min(n_examples, len(candidates)))]

    print(f"  Found {len(candidates)} good candidates, selected {len(selected)}")

    return np.array(selected)


def write_vtk_file(ugrid, filename):
    """Write VTK unstructured grid to file."""
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(filename))
    writer.SetInputData(ugrid)
    writer.Write()


def create_pvd_collection(files_dict, output_path):
    """Create ParaView collection file (.pvd)."""
    pvd_content = '<?xml version="1.0"?>\n'
    pvd_content += '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
    pvd_content += '  <Collection>\n'

    for name, vtu_file in files_dict.items():
        pvd_content += f'    <DataSet name="{name}" file="{vtu_file.name}"/>\n'

    pvd_content += '  </Collection>\n'
    pvd_content += '</VTKFile>\n'

    output_path.write_text(pvd_content)


def main():
    print("="*80)
    print("Improved Mesh-Octree Alignment Visualization (2×2×2 Pattern)")
    print("="*80)

    # Load mesh
    node_positions, connectivity = load_mesh()
    print(f"  Loaded {connectivity.shape[0]:,} elements")

    # Find good example elements
    example_elem_ids = find_good_example_elements(node_positions, connectivity, n_examples=3)

    # Create visualizations for each example
    files_dict = {}

    for idx, elem_id in enumerate(example_elem_ids):
        print(f"\n--- Example {idx + 1} ---")

        # Get element info
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]
        cell_size, level = find_axis_aligned_edges_fast(vertices)

        # Create visualization
        tet_mesh, cell_mesh, level = create_single_tet_with_8cells_vtk(
            elem_id, node_positions, connectivity
        )

        # Create bbox mesh
        bbox_mesh = create_element_bbox_mesh(vertices)

        # Write files
        tet_file = OUTPUT_DIR / f"example{idx+1}_tet_level{level}.vtu"
        cell_file = OUTPUT_DIR / f"example{idx+1}_cells_level{level}.vtu"
        bbox_file = OUTPUT_DIR / f"example{idx+1}_bbox_level{level}.vtu"

        write_vtk_file(tet_mesh, tet_file)
        write_vtk_file(cell_mesh, cell_file)
        write_vtk_file(bbox_mesh, bbox_file)

        files_dict[f"Example{idx+1}_Tetrahedron"] = tet_file
        files_dict[f"Example{idx+1}_8Cells"] = cell_file
        files_dict[f"Example{idx+1}_BBox"] = bbox_file

        print(f"  Wrote example {idx+1} (level {level})")

    # Create collection file
    pvd_file = OUTPUT_DIR / "visualization_2x2x2_pattern.pvd"
    create_pvd_collection(files_dict, pvd_file)

    print(f"\n{'='*80}")
    print("Visualization Complete!")
    print(f"{'='*80}")
    print(f"\nOutput: {OUTPUT_DIR}/")
    print(f"  visualization_2x2x2_pattern.pvd - Open this in ParaView")
    print(f"\nParaView Settings:")
    print(f"  For each example:")
    print(f"    1. Tetrahedron:")
    print(f"       - Representation: Surface With Edges")
    print(f"       - Color: Red or Yellow (solid)")
    print(f"       - Opacity: 1.0")
    print(f"    2. 8 Cells (octree cubes):")
    print(f"       - Representation: Wireframe")
    print(f"       - Color by: CellIndex (0-7)")
    print(f"       - Line Width: 2")
    print(f"    3. BBox (optional):")
    print(f"       - Representation: Wireframe")
    print(f"       - Color: Green")
    print(f"       - Shows element bounding box")
    print(f"\n  You should see:")
    print(f"    - One tetrahedron (solid colored)")
    print(f"    - 8 surrounding cube wireframes (2×2×2 arrangement)")
    print(f"    - Tet vertices/edges crossing cube boundaries")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
