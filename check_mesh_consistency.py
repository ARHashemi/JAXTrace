#!/usr/bin/env python3
"""Check mesh consistency across revolution cycle."""
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import glob

files = sorted(glob.glob("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_*.pvtu"))

print(f"Total files: {len(files)}")
print("\nChecking revolution cycle (last 40 files):")
print("=" * 70)

revolution_files = files[-40:]
mesh_sizes = []

for i, file_path in enumerate(revolution_files):
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    mesh = reader.GetOutput()

    n_points = mesh.GetNumberOfPoints()
    n_cells = mesh.GetNumberOfCells()

    # Extract timestep from filename
    import re
    match = re.search(r'_(\d+)\.pvtu$', file_path)
    timestep = int(match.group(1)) if match else i

    mesh_sizes.append((timestep, n_points, n_cells))
    print(f"Timestep {timestep:3d}: {n_points:7d} points, {n_cells:7d} cells")

print("\n" + "=" * 70)
print("SUMMARY:")
unique_sizes = set([(p, c) for _, p, c in mesh_sizes])
print(f"Unique mesh configurations: {len(unique_sizes)}")

if len(unique_sizes) == 1:
    print("✅ ALL revolution cycle meshes are IDENTICAL!")
    print(f"   Points: {mesh_sizes[0][1]:,}")
    print(f"   Cells:  {mesh_sizes[0][2]:,}")
else:
    print("❌ PROBLEM: Revolution cycle meshes are NOT identical!")
    for size in unique_sizes:
        count = sum(1 for _, p, c in mesh_sizes if (p, c) == size)
        print(f"   {size[0]:7d} points, {size[1]:7d} cells: {count} timesteps")
