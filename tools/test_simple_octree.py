#!/usr/bin/env python3
"""
Simple test of octree building without full factory.
"""

import sys
import glob
sys.path.insert(0, '/home/arhashemi/Workspace/welding/JAXTrace')

import vtk
import numpy as np

def test_vtk_load():
    """Test basic VTK file loading."""
    print("Test 1: VTK File Loading")
    print("=" * 50)

    pattern = "/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_*.pvtu"
    files = sorted(glob.glob(pattern))

    print(f"Found {len(files)} files")

    if len(files) == 0:
        print("ERROR: No files found")
        return False

    # Test loading first file
    print(f"\nLoading first file: {files[0]}")

    try:
        reader = vtk.vtkXMLPUnstructuredGridReader()
        reader.SetFileName(files[0])
        reader.Update()
        mesh = reader.GetOutput()

        n_points = mesh.GetNumberOfPoints()
        n_cells = mesh.GetNumberOfCells()

        print(f"  Points: {n_points:,}")
        print(f"  Cells: {n_cells:,}")

        # Get first few points
        print("\n  First 3 points:")
        for i in range(min(3, n_points)):
            pt = mesh.GetPoint(i)
            print(f"    {i}: {pt}")

        # Get first cell
        if n_cells > 0:
            cell = mesh.GetCell(0)
            print(f"\n  First cell has {cell.GetNumberOfPoints()} points")

        print("\n✓ VTK loading successful")
        return True

    except Exception as e:
        print(f"\n✗ VTK loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_octree_building():
    """Test basic octree node creation."""
    print("\n\nTest 2: Basic Octree Building")
    print("=" * 50)

    try:
        # Create simple test data
        np.random.seed(42)
        n_cells = 1000
        cell_centers = np.random.rand(n_cells, 3).astype(np.float32)

        print(f"Created {n_cells} test cells")

        bbox_min = cell_centers.min(axis=0)
        bbox_max = cell_centers.max(axis=0)

        print(f"Bounding box: {bbox_min} to {bbox_max}")

        # Build simple octree
        all_indices = np.arange(n_cells)

        nodes = []

        def build_node(cell_indices, bbox_min, bbox_max, level, max_level=3, max_cells=32):
            """Simple recursive octree builder."""
            node_idx = len(nodes)
            center = (bbox_min + bbox_max) / 2

            node = {
                'level': level,
                'n_cells': len(cell_indices),
                'children': []
            }

            if level >= max_level or len(cell_indices) <= max_cells:
                nodes.append(node)
                return node_idx

            # Subdivide
            for octant in range(8):
                # Select cells in this octant
                mask_x = (cell_centers[cell_indices, 0] > center[0]) if (octant & 4) else (cell_centers[cell_indices, 0] <= center[0])
                mask_y = (cell_centers[cell_indices, 1] > center[1]) if (octant & 2) else (cell_centers[cell_indices, 1] <= center[1])
                mask_z = (cell_centers[cell_indices, 2] > center[2]) if (octant & 1) else (cell_centers[cell_indices, 2] <= center[2])

                mask = mask_x & mask_y & mask_z
                octant_indices = cell_indices[mask]

                if len(octant_indices) > 0:
                    # Compute child bbox
                    child_min = bbox_min.copy()
                    child_max = bbox_max.copy()

                    if octant & 4:
                        child_min[0] = center[0]
                    else:
                        child_max[0] = center[0]

                    if octant & 2:
                        child_min[1] = center[1]
                    else:
                        child_max[1] = center[1]

                    if octant & 1:
                        child_min[2] = center[2]
                    else:
                        child_max[2] = center[2]

                    child_idx = build_node(octant_indices, child_min, child_max, level + 1, max_level, max_cells)
                    node['children'].append(child_idx)

            nodes.append(node)
            return node_idx

        build_node(all_indices, bbox_min, bbox_max, 0)

        print(f"\n✓ Built octree with {len(nodes)} nodes")

        # Print distribution
        levels = [n['level'] for n in nodes]
        for level in range(max(levels) + 1):
            count = sum(1 for l in levels if l == level)
            print(f"  Level {level}: {count} nodes")

        return True

    except Exception as e:
        print(f"\n✗ Octree building failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("SIMPLE OCTREE TEST")
    print("=" * 50)
    print()

    test1 = test_vtk_load()
    test2 = test_octree_building()

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"VTK Loading: {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"Octree Building: {'✓ PASS' if test2 else '✗ FAIL'}")

    sys.exit(0 if (test1 and test2) else 1)
