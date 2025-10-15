#!/usr/bin/env python3
"""
Quick mesh size analyzer for AMR data.
Reads only metadata (fast) from VTK files.
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
    pattern = "/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_*.pvtu"
    files = sorted(glob.glob(pattern))

    print(f"Total files found: {len(files)}")
    print(f"\nAnalyzing mesh sizes (this may take a few minutes)...\n")

    # Check specific indices: first 15, then 120-159
    indices = list(range(15)) + list(range(120, min(160, len(files))))

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

        if idx == 14:
            print("-" * 50)

    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS:")
    print(f"{'='*60}\n")

    if len(mesh_data) == 0:
        print("No data collected")
        return

    # Initial refinement phase
    print("1. Initial Refinement Phase (steps 0-14):")
    initial_data = [m for m in mesh_data if m[0] < 15]
    if len(initial_data) > 1:
        print(f"   Start (step 0): {initial_data[0][1]:,} points")
        print(f"   End (step 14): {initial_data[-1][1]:,} points")
        growth = ((initial_data[-1][1] - initial_data[0][1]) / initial_data[0][1]) * 100
        print(f"   Growth: {growth:+.2f}%")

        # Find stabilization point
        for i in range(1, len(initial_data)):
            change = abs((initial_data[i][1] - initial_data[i-1][1]) / initial_data[i-1][1]) * 100
            if change < 0.5:  # <0.5% change
                print(f"   Stabilizes around step: {initial_data[i][0]} ({initial_data[i][1]:,} points)")
                break

    # Revolution cycle phase
    print("\n2. Revolution Cycle (steps 120-159):")
    revolution_data = [m for m in mesh_data if m[0] >= 120]
    if len(revolution_data) > 1:
        points_list = [m[1] for m in revolution_data]
        min_pts = min(points_list)
        max_pts = max(points_list)
        avg_pts = np.mean(points_list)

        print(f"   Min points: {min_pts:,}")
        print(f"   Max points: {max_pts:,}")
        print(f"   Avg points: {avg_pts:,.0f}")
        print(f"   Range: {max_pts - min_pts:,} points ({((max_pts-min_pts)/avg_pts*100):.2f}% variation)")

        # Per-step changes
        changes = []
        for i in range(1, len(revolution_data)):
            change = abs((revolution_data[i][1] - revolution_data[i-1][1]) / revolution_data[i-1][1]) * 100
            changes.append(change)

        if changes:
            print(f"   Avg change per step: {np.mean(changes):.3f}%")
            print(f"   Max change per step: {np.max(changes):.3f}%")

    # Memory estimate
    print("\n3. Memory Estimates (for 40 revolution timesteps):")
    if revolution_data:
        avg_points = np.mean([m[1] for m in revolution_data])
        avg_cells = np.mean([m[2] for m in revolution_data])

        # Memory per timestep
        points_mem = avg_points * 3 * 4 / (1024**2)  # MB
        velocity_mem = avg_points * 3 * 4 / (1024**2)  # MB
        connectivity_mem = avg_cells * 4 * 4 / (1024**2)  # MB (shared)

        # Total for 40 timesteps
        total_points = points_mem * 40
        total_velocity = velocity_mem * 40
        total_connectivity = connectivity_mem  # Shared
        total_all = total_points + total_velocity + total_connectivity

        print(f"   Points (40 timesteps): {total_points:.1f} MB")
        print(f"   Velocity (40 timesteps): {total_velocity:.1f} MB")
        print(f"   Connectivity (shared): {total_connectivity:.1f} MB")
        print(f"   Subtotal (mesh data): {total_all:.1f} MB")
        print(f"   Octrees (estimated): {40 * 50:.0f} MB")
        print(f"   TOTAL: {total_all + 2000:.0f} MB (~{(total_all + 2000)/1024:.2f} GB)")

if __name__ == "__main__":
    main()
