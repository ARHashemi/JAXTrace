import numpy as np
import vtk

# Load a single mesh file
reader = vtk.vtkXMLPUnstructuredGridReader()
reader.SetFileName("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_120.pvtu")
reader.Update()
mesh = reader.GetOutput()

# Get positions
points_vtk = mesh.GetPoints()
n_points = points_vtk.GetNumberOfPoints()
positions = np.zeros((n_points, 3), dtype=np.float32)
for i in range(n_points):
    positions[i] = points_vtk.GetPoint(i)

# Get connectivity
cells = mesh.GetCells()
connectivity_list = []
cells.InitTraversal()
id_list = vtk.vtkIdList()
while cells.GetNextCell(id_list):
    if id_list.GetNumberOfIds() == 4:  # Tetrahedra only
        connectivity_list.append([id_list.GetId(j) for j in range(4)])

connectivity = np.array(connectivity_list, dtype=np.int32)
print(f"Mesh: {n_points} points, {len(connectivity)} elements")

# Compute element bounds (sample 10000 elements)
sample_size = min(10000, len(connectivity))
sample_indices = np.random.choice(len(connectivity), sample_size, replace=False)

overlap_counts = []
for idx in sample_indices:
    node_indices = connectivity[idx]
    elem_points = positions[node_indices]
    elem_min = elem_points.min(axis=0)
    elem_max = elem_points.max(axis=0)
    elem_size = elem_max - elem_min
    
    # Simple domain division into 8 octants
    domain_min = positions.min(axis=0)
    domain_max = positions.max(axis=0)
    center = (domain_min + domain_max) / 2.0
    
    # Count how many top-level octants this element overlaps
    octant_bounds = [
        (domain_min, center),
        (np.array([center[0], domain_min[1], domain_min[2]]), np.array([domain_max[0], center[1], center[2]])),
        (np.array([domain_min[0], center[1], domain_min[2]]), np.array([center[0], domain_max[1], center[2]])),
        (np.array([center[0], center[1], domain_min[2]]), np.array([domain_max[0], domain_max[1], center[2]])),
        (np.array([domain_min[0], domain_min[1], center[2]]), np.array([center[0], center[1], domain_max[2]])),
        (np.array([center[0], domain_min[1], center[2]]), np.array([domain_max[0], center[1], domain_max[2]])),
        (np.array([domain_min[0], center[1], center[2]]), np.array([center[0], domain_max[1], domain_max[2]])),
        (center, domain_max),
    ]
    
    count = 0
    for octant_min, octant_max in octant_bounds:
        overlaps = (elem_min[0] <= octant_max[0] and elem_max[0] >= octant_min[0] and
                   elem_min[1] <= octant_max[1] and elem_max[1] >= octant_min[1] and
                   elem_min[2] <= octant_max[2] and elem_max[2] >= octant_min[2])
        if overlaps:
            count += 1
    overlap_counts.append(count)

overlap_counts = np.array(overlap_counts)
print(f"\nElement overlap statistics (sample of {sample_size} elements):")
print(f"  Mean octants per element: {overlap_counts.mean():.2f}")
print(f"  Median: {np.median(overlap_counts):.0f}")
print(f"  Max: {overlap_counts.max()}")
print(f"  Distribution:")
for i in range(1, 9):
    pct = (overlap_counts == i).sum() / len(overlap_counts) * 100
    print(f"    {i} octants: {pct:5.1f}%")
    
print(f"\nMemory multiplication factor estimate: {overlap_counts.mean():.2f}x")
