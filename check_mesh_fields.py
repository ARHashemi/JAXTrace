#!/usr/bin/env python3
"""
Quick diagnostic to check what fields are available in ThreadedA mesh.
"""

import vtk
from vtk.util import numpy_support

mesh_path = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_120.pvtu"

print("=" * 80)
print("MESH FIELD DIAGNOSTICS")
print("=" * 80)
print()

print(f"Loading mesh: {mesh_path}")
reader = vtk.vtkXMLPUnstructuredGridReader()
reader.SetFileName(mesh_path)
reader.Update()
vtk_mesh = reader.GetOutput()

print(f"✓ Mesh loaded")
print(f"  Number of points: {vtk_mesh.GetNumberOfPoints():,}")
print(f"  Number of cells: {vtk_mesh.GetNumberOfCells():,}")
print()

# Check point data (node-based fields)
print("POINT DATA (node-based fields):")
point_data = vtk_mesh.GetPointData()
n_point_arrays = point_data.GetNumberOfArrays()
print(f"  Number of arrays: {n_point_arrays}")

if n_point_arrays > 0:
    for i in range(n_point_arrays):
        array_name = point_data.GetArrayName(i)
        array = point_data.GetArray(i)
        n_tuples = array.GetNumberOfTuples()
        n_components = array.GetNumberOfComponents()
        data_type = array.GetDataType()

        # Convert VTK type to string
        type_map = {
            vtk.VTK_FLOAT: "float32",
            vtk.VTK_DOUBLE: "float64",
            vtk.VTK_INT: "int32",
            vtk.VTK_UNSIGNED_INT: "uint32",
            vtk.VTK_CHAR: "int8",
            vtk.VTK_UNSIGNED_CHAR: "uint8"
        }
        type_str = type_map.get(data_type, f"unknown({data_type})")

        print(f"  [{i}] {array_name}: {n_tuples:,} tuples × {n_components} components ({type_str})")
else:
    print("  (No point data arrays)")

print()

# Check cell data (element-based fields)
print("CELL DATA (element-based fields):")
cell_data = vtk_mesh.GetCellData()
n_cell_arrays = cell_data.GetNumberOfArrays()
print(f"  Number of arrays: {n_cell_arrays}")

if n_cell_arrays > 0:
    for i in range(n_cell_arrays):
        array_name = cell_data.GetArrayName(i)
        array = cell_data.GetArray(i)
        n_tuples = array.GetNumberOfTuples()
        n_components = array.GetNumberOfComponents()
        data_type = array.GetDataType()

        # Convert VTK type to string
        type_map = {
            vtk.VTK_FLOAT: "float32",
            vtk.VTK_DOUBLE: "float64",
            vtk.VTK_INT: "int32",
            vtk.VTK_UNSIGNED_INT: "uint32",
            vtk.VTK_CHAR: "int8",
            vtk.VTK_UNSIGNED_CHAR: "uint8"
        }
        type_str = type_map.get(data_type, f"unknown({data_type})")

        print(f"  [{i}] {array_name}: {n_tuples:,} tuples × {n_components} components ({type_str})")

        # If this is LEVEL, show some statistics
        if array_name == 'LEVEL':
            level_data = numpy_support.vtk_to_numpy(array)
            print(f"      → Range: [{level_data.min()}, {level_data.max()}]")
            print(f"      → Unique values: {sorted(set(level_data.tolist()))}")
else:
    print("  (No cell data arrays)")

print()

# Direct check for LEVEL
print("DIRECT LEVEL CHECK:")
if cell_data.HasArray('LEVEL'):
    print("  ✓ cell_data.HasArray('LEVEL') = True")
    level_array = cell_data.GetArray('LEVEL')
    print(f"  ✓ Successfully retrieved LEVEL array")
    print(f"    Type: {type(level_array)}")
    print(f"    Tuples: {level_array.GetNumberOfTuples():,}")

    level_np = numpy_support.vtk_to_numpy(level_array)
    print(f"  ✓ Converted to numpy: shape={level_np.shape}, dtype={level_np.dtype}")
    print(f"    Range: [{level_np.min()}, {level_np.max()}]")
else:
    print("  ✗ cell_data.HasArray('LEVEL') = False")

print()
print("=" * 80)
