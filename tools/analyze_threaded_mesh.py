#!/usr/bin/env python3
"""
Analyze ThreadedA mesh for more complex AMR patterns.
"""

import vtk
import glob
import sys
import numpy as np

def read_mesh_size_fast(filename):
    """Read mesh size without loading full data."""
    try:
        reader = vtk.vtkXMLPUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.UpdateInformation()
        reader.Update()
        mesh = reader.GetOutput()
        n_points = mesh.GetNumberOfPoints()
        n_cells = mesh.GetNumberOfCells()
        return n_points, n_cells
    except Exception as e:
        return None, None

def main():
    pattern = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/*.pvtu"
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        print(f"No files found matching: {pattern}")
        return

    print(f"ThreadedA case: {len(files)} files found")
    print(f"\nAnalyzing mesh sizes...\n")

    # Check first 10 and last 40
    n_total = len(files)
    last_40_start = max(0, n_total - 40)

    indices = list(range(min(10, n_total))) + list(range(last_40_start, n_total))

    print(f"{'Step':<6} {'Points':<12} {'Cells':<12} {'Change %':<12}")
    print("=" * 50)

    mesh_data = []
    prev_points = None

    for idx in indices:
        n_points, n_cells = read_mesh_size_fast(files[idx])

        if n_points is None:
            print(f"{idx:<6} ERROR")
            continue

        change_str = ""
        if prev_points is not None:
            change_pct = ((n_points - prev_points) / prev_points) * 100
            change_str = f"{change_pct:+.3f}%"

        print(f"{idx:<6} {n_points:<12,} {n_cells:<12,} {change_str:<12}")

        mesh_data.append((idx, n_points, n_cells))
        prev_points = n_points

        if idx == 9 and n_total > 40:
            print("-" * 50)

    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS:")
    print(f"{'='*60}\n")

    if len(mesh_data) == 0:
        print("No data collected")
        return

    # Initial refinement
    print("1. Initial Refinement Phase (first 10 steps):")
    initial_data = [m for m in mesh_data if m[0] < 10]
    if len(initial_data) > 1:
        print(f"   Start (step 0): {initial_data[0][1]:,} points")
        print(f"   End (step {initial_data[-1][0]}): {initial_data[-1][1]:,} points")
        growth = ((initial_data[-1][1] - initial_data[0][1]) / initial_data[0][1]) * 100
        print(f"   Growth: {growth:+.2f}%")

        # Track changes
        print(f"\n   Step-by-step changes:")
        for i in range(1, len(initial_data)):
            change = ((initial_data[i][1] - initial_data[i-1][1]) / initial_data[i-1][1]) * 100
            status = "stable" if abs(change) < 0.5 else "refinement"
            print(f"   Step {initial_data[i-1][0]} → {initial_data[i][0]}: {change:+.3f}% ({status})")

    # Revolution cycle
    print(f"\n2. Revolution Cycle (last {min(40, n_total)} steps):")
    revolution_data = [m for m in mesh_data if m[0] >= last_40_start]
    if len(revolution_data) > 1:
        points_list = [m[1] for m in revolution_data]
        min_pts = min(points_list)
        max_pts = max(points_list)
        avg_pts = np.mean(points_list)

        print(f"   Steps: {revolution_data[0][0]} to {revolution_data[-1][0]}")
        print(f"   Min points: {min_pts:,}")
        print(f"   Max points: {max_pts:,}")
        print(f"   Avg points: {avg_pts:,.0f}")
        print(f"   Range: {max_pts - min_pts:,} points ({((max_pts-min_pts)/avg_pts*100):.2f}% variation)")

        # Per-step changes
        changes = []
        large_changes = []
        for i in range(1, len(revolution_data)):
            change = abs((revolution_data[i][1] - revolution_data[i-1][1]) / revolution_data[i-1][1]) * 100
            changes.append(change)
            if change > 0.1:  # Track significant changes
                large_changes.append((revolution_data[i][0], change))

        if changes:
            print(f"\n   Change statistics:")
            print(f"   Avg change per step: {np.mean(changes):.3f}%")
            print(f"   Max change per step: {np.max(changes):.3f}%")
            print(f"   Median change: {np.median(changes):.3f}%")
            print(f"   Steps with >0.1% change: {len(large_changes)}")

        if large_changes:
            print(f"\n   Significant variations (>0.1%):")
            for step, change in large_changes[:10]:  # Show first 10
                print(f"   Step {step}: {change:.3f}%")

    # Comparison with FLA
    print(f"\n3. Comparison with FLA case:")
    print(f"   ThreadedA mesh variation: TBD based on above")
    print(f"   FLA mesh variation: 0.065% average, 1.2% max")
    print(f"   ThreadedA appears to have {'more' if len(large_changes) > 3 else 'similar'} variation")

if __name__ == "__main__":
    main()
