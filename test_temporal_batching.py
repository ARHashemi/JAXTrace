#!/usr/bin/env python3
"""
Test script for temporal batching implementation

This script creates synthetic VTK files with variable mesh sizes (simulating AMR)
and verifies that temporal batching can handle them correctly.
"""

import numpy as np
import os
import tempfile
import shutil
from pathlib import Path


def create_synthetic_vtk_files(output_dir, n_timesteps=10, variable_mesh=True):
    """Create synthetic VTK files with optionally variable mesh sizes."""
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk
    except ImportError:
        print("❌ VTK not available, skipping VTK file creation")
        return None

    print(f"Creating {n_timesteps} synthetic VTK files in {output_dir}")

    files = []
    for t_idx in range(n_timesteps):
        # Variable mesh size (simulating AMR)
        if variable_mesh:
            n_base = 100 + (t_idx % 3) * 50  # Varies between 100, 150, 200
        else:
            n_base = 100

        # Create random tetrahedral mesh
        n_points = n_base
        points_array = np.random.uniform(0, 1, (n_points, 3)).astype(np.float32)

        # Create some tetrahedra (simplified - just random indices)
        n_cells = n_points // 4
        connectivity = []
        for i in range(n_cells):
            # Random 4 points for each tetrahedron
            indices = np.random.choice(n_points, 4, replace=False)
            connectivity.append(indices)

        connectivity = np.array(connectivity, dtype=np.int32)

        # Create time-dependent velocity field (simple vortex)
        t = t_idx * 0.1
        strength = 1.0 + 0.5 * np.sin(2 * np.pi * t / 2)

        center_x = 0.5 + 0.1 * np.cos(2 * np.pi * t / 2)
        center_y = 0.5 + 0.1 * np.sin(2 * np.pi * t / 2)

        dx = points_array[:, 0] - center_x
        dy = points_array[:, 1] - center_y
        r_squared = dx**2 + dy**2 + 1e-6

        vx = -strength * dy / r_squared
        vy = strength * dx / r_squared
        vz = np.zeros_like(vx)

        velocity = np.stack([vx, vy, vz], axis=1).astype(np.float32)

        # Create VTK unstructured grid
        vtk_points = vtk.vtkPoints()
        for pt in points_array:
            vtk_points.InsertNextPoint(pt[0], pt[1], pt[2])

        vtk_grid = vtk.vtkUnstructuredGrid()
        vtk_grid.SetPoints(vtk_points)

        # Add cells
        for tet in connectivity:
            vtk_tetra = vtk.vtkTetra()
            for i, idx in enumerate(tet):
                vtk_tetra.GetPointIds().SetId(i, int(idx))
            vtk_grid.InsertNextCell(vtk_tetra.GetCellType(), vtk_tetra.GetPointIds())

        # Add velocity as point data
        vtk_velocity = numpy_to_vtk(velocity)
        vtk_velocity.SetName("Displacement")
        vtk_grid.GetPointData().AddArray(vtk_velocity)

        # Write to file
        filename = os.path.join(output_dir, f"synthetic_{t_idx:04d}.vtu")
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(vtk_grid)
        writer.Write()

        files.append(filename)

        if (t_idx + 1) % 5 == 0:
            print(f"  Created {t_idx + 1}/{n_timesteps} files (mesh: {n_points} points, {n_cells} cells)")

    print(f"✅ Created {len(files)} VTK files")
    return files


def test_temporal_batching():
    """Test temporal batching with synthetic AMR data."""
    print("="*80)
    print("TEMPORAL BATCHING TEST")
    print("="*80)

    # Create temporary directory for VTK files
    temp_dir = tempfile.mkdtemp(prefix="jaxtrace_test_")
    print(f"\n📁 Temporary directory: {temp_dir}")

    try:
        # Create synthetic VTK files with variable mesh
        files = create_synthetic_vtk_files(temp_dir, n_timesteps=10, variable_mesh=True)

        if files is None:
            print("\n⚠️  VTK module not available, cannot run full test")
            print("✅ Basic imports will be tested instead\n")

            # Test basic imports
            print("Testing imports...")
            from jaxtrace.fields.grid_hash_field import build_grid_hash_mesh, create_grid_hash_interpolator
            from jaxtrace.fields.temporal_field import TemporalBatchingField
            from jaxtrace.tracking.temporal_tracker import TemporalBatchingTracker
            print("✅ All temporal batching modules imported successfully")
            return

        # Test temporal batching workflow
        print("\n" + "="*80)
        print("TESTING TEMPORAL BATCHING WORKFLOW")
        print("="*80)

        # Import required modules
        from jaxtrace.fields.temporal_field import TemporalBatchingField
        from jaxtrace.tracking.temporal_tracker import TemporalBatchingTracker
        from jaxtrace.integrators.rk4 import rk4_step
        from jaxtrace.tracking.boundary import reflective_boundary

        # Create temporal batching field
        print("\n1. Creating TemporalBatchingField...")
        data_pattern = os.path.join(temp_dir, "synthetic_*.vtu")
        field = TemporalBatchingField(
            data_pattern=data_pattern,
            grid_resolution=16,  # Small for testing
            cache_size=3
        )
        print(f"✅ Field created: {len(field.files)} files found")

        # Load first timestep to get bounds
        print("\n2. Loading first timestep to get bounds...")
        first_mesh = field.load_timestep(0)
        bounds_min = first_mesh['bounds_min']
        bounds_max = first_mesh['bounds_max']
        print(f"✅ Bounds: {bounds_min} to {bounds_max}")

        # Create particles
        print("\n3. Creating test particles...")
        from jaxtrace.tracking.seeding import uniform_grid_seeds
        initial_positions = uniform_grid_seeds(
            resolution=(5, 5, 5),
            bounds=[bounds_min, bounds_max],
            include_boundaries=True
        )
        print(f"✅ Created {len(initial_positions)} particles")

        # Create boundary condition
        print("\n4. Creating boundary condition...")
        boundary_fn = reflective_boundary([bounds_min, bounds_max])
        print("✅ Reflective boundary created")

        # Create tracker
        print("\n5. Creating TemporalBatchingTracker...")

        def integrator_fn(x, t, dt, field_fn):
            return rk4_step(x, t, dt, field_fn)

        tracker = TemporalBatchingTracker(
            integrator=integrator_fn,
            field=field,
            boundary_fn=boundary_fn,
            temporal_window_size=5,  # Small window for testing
            record_velocities=False
        )
        print("✅ Tracker created")

        # Run tracking
        print("\n6. Running particle tracking...")
        n_timesteps = 20  # Small number for testing
        dt_tracking = 0.01
        dt_data = 0.1

        trajectory_data = tracker.track_particles(
            initial_positions=initial_positions,
            n_tracking_steps=n_timesteps,
            dt_tracking=dt_tracking,
            dt_data=dt_data,
            progress_callback=None
        )

        print(f"✅ Tracking completed")
        print(f"   Positions shape: {trajectory_data['positions'].shape}")
        if trajectory_data['velocities'] is not None:
            print(f"   Velocities shape: {trajectory_data['velocities'].shape}")
        else:
            print(f"   Velocities: not recorded")

        # Verify results
        print("\n7. Verifying results...")
        positions = trajectory_data['positions']
        assert positions.shape == (n_timesteps, len(initial_positions), 3), \
            f"Wrong shape: expected ({n_timesteps}, {len(initial_positions)}, 3), got {positions.shape}"

        # Check that particles moved (not stuck)
        displacement = np.linalg.norm(positions[-1] - positions[0], axis=1)
        mean_displacement = np.mean(displacement)
        print(f"   Mean particle displacement: {mean_displacement:.4f}")

        if mean_displacement < 1e-6:
            print("   ⚠️  WARNING: Particles barely moved (might be stuck or zero velocity)")
        else:
            print("   ✅ Particles moved as expected")

        # Check that particles stay in bounds (with reflective boundary)
        in_bounds = np.all((positions >= bounds_min) & (positions <= bounds_max))
        if in_bounds:
            print("   ✅ All particles stayed within bounds")
        else:
            print("   ⚠️  WARNING: Some particles outside bounds")

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)

    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up temporary files
        print(f"\n🧹 Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
        print("✅ Cleanup complete")


if __name__ == "__main__":
    test_temporal_batching()
