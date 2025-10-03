#!/usr/bin/env python3
"""
JAXTrace GPU-Optimized Workflow with Octree FEM Interpolation

For meshes with adaptive refinement (6 levels in your case).
Uses octree spatial partitioning for O(log N) element lookup.
"""

import os
# GPU optimization
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

import time
from pathlib import Path
import numpy as np

import jax
import jax.numpy as jnp
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import jaxtrace as jt
from jaxtrace.fields.octree_fem_time_series import OctreeFEMTimeSeriesField
from jaxtrace.tracking import create_tracker, uniform_grid_seeds
from jaxtrace.tracking.boundary import reflective_boundary
from jaxtrace.io import export_trajectory_to_vtk

print("="*80)
print(f"JAXTrace v{jt.__version__} - Octree FEM Interpolation Workflow")
print("="*80)


def main():
    """Workflow with octree FEM interpolation."""

    # 1. Configure
    print("\\n1. GPU CONFIGURATION")
    print("="*80)

    jt.configure(dtype="float32", device="gpu", memory_limit_gb=3.0)
    print(f"✅ Device: {jax.devices()}")

    # 2. Load with octree FEM
    print("\\n2. LOAD WITH OCTREE FEM INTERPOLATION")
    print("="*80)

    field = load_vtk_with_octree_fem()

    # 3. Track
    print("\\n3. GPU-ACCELERATED TRACKING (OCTREE FEM)")
    print("="*80)

    trajectory = fast_octree_fem_tracking(field)

    # 4. Export
    print("\\n4. EXPORT")
    print("="*80)

    export_results(trajectory)

    print("\\n" + "="*80)
    print("✅ OCTREE FEM WORKFLOW COMPLETE!")
    print("="*80)


def load_vtk_with_octree_fem():
    """Load VTK data with octree FEM interpolation."""

    pattern = "/home/arhashemi/Workspace/welding/Cases/004_caseCoarse.gid/post/0eule/004_caseCoarse_*.pvtu"

    print(f"📁 Loading VTK with connectivity...")

    # Find files
    from glob import glob
    files = sorted(glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files found: {pattern}")

    print(f"   Found {len(files)} files")

    # Load subset of timesteps
    max_timesteps = 40
    stride = max(1, len(files) // max_timesteps)
    files_to_load = files[::stride][:max_timesteps]

    print(f"   Loading {len(files_to_load)} timesteps...")

    # Load first file to get mesh
    print(f"   Reading mesh from: {files_to_load[0]}")
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(files_to_load[0])
    reader.Update()
    mesh = reader.GetOutput()

    # Extract mesh data
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float32)
    n_points = points.shape[0]

    print(f"   Mesh: {n_points} points")

    # Extract connectivity (tetrahedral mesh)
    connectivity = []
    for i in range(mesh.GetNumberOfCells()):
        cell = mesh.GetCell(i)
        if cell.GetCellType() == vtk.VTK_TETRA:  # Type 10
            point_ids = cell.GetPointIds()
            tet = [point_ids.GetId(j) for j in range(4)]
            connectivity.append(tet)

    connectivity = np.array(connectivity, dtype=np.int32)
    print(f"   Elements: {connectivity.shape[0]} tetrahedra")

    # Load velocity data for all timesteps
    velocity_data = []
    times = []

    for idx, filename in enumerate(files_to_load):
        reader = vtk.vtkXMLPUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        mesh = reader.GetOutput()

        # Get velocity field (stored as 'Displacement')
        point_data = mesh.GetPointData()
        vel_array = None

        for name in ['Displacement', 'displacement', 'Velocity', 'velocity']:
            if point_data.HasArray(name):
                vel_array = point_data.GetArray(name)
                break

        if vel_array is None:
            raise ValueError(f"No velocity field found in {filename}")

        velocity = vtk_to_numpy(vel_array).astype(np.float32)

        # Ensure 3D
        if velocity.shape[1] == 2:
            velocity = np.column_stack([velocity, np.zeros(velocity.shape[0])])

        velocity_data.append(velocity)

        # Extract time from filename (e.g., 004_caseCoarse_100.pvtu -> 100)
        import re
        match = re.search(r'_(\\d+)\\.pvtu$', filename)
        if match:
            times.append(float(match.group(1)))
        else:
            times.append(float(idx))

        if (idx + 1) % 10 == 0:
            print(f"   Loaded {idx + 1}/{len(files_to_load)} timesteps...")

    velocity_data = np.array(velocity_data, dtype=np.float32)  # (T, N, 3)
    times = np.array(times, dtype=np.float32)

    print(f"✅ Loaded velocity data: {velocity_data.shape}")

    # Create octree FEM field
    print(f"🌲 Creating octree FEM interpolation field...")

    field = OctreeFEMTimeSeriesField(
        data=velocity_data,
        times=times,
        positions=points,
        connectivity=connectivity,
        interpolation="linear",
        extrapolation="constant",
        max_elements_per_leaf=32,  # Optimal for this mesh
        max_depth=12
    )

    # Convert to JAX arrays on GPU
    print(f"🔄 Converting to GPU...")
    field.data = jnp.array(field.data)
    field.positions = jnp.array(field.positions)
    field.times = jnp.array(field.times)
    field._data_dev = jax.device_put(field.data)
    field._times_dev = jax.device_put(field.times)
    field._pos_dev = jax.device_put(field.positions)

    data_mb = field.data.nbytes / 1024 / 1024
    print(f"✅ Field on GPU: {data_mb:.1f} MB")

    return field


def fast_octree_fem_tracking(field):
    """Fast tracking with octree FEM interpolation."""

    # Get bounds
    bounds_min, bounds_max = field.get_spatial_bounds()
    print(f"📏 Bounds: {bounds_min} to {bounds_max}")

    # Seed particles
    print(f"🎯 Seeding particles...")
    seeds = uniform_grid_seeds(
        resolution=(30, 25, 10),
        bounds=[bounds_min, bounds_max],
        include_boundaries=True
    )
    print(f"✅ {len(seeds)} particles")

    # Boundary
    boundary = reflective_boundary([bounds_min, bounds_max])

    # Config
    n_timesteps = 1500
    batch_size = len(seeds)
    dt = 0.005

    print(f"🎯 Config:")
    print(f"   Timesteps: {n_timesteps}")
    print(f"   Batch: {batch_size} (single batch)")
    print(f"   Interpolation: Octree FEM (adaptive)")

    # Create tracker
    print(f"🚀 Creating tracker...")
    tracker = create_tracker(
        integrator_name='rk4',
        field=field,
        boundary_condition=boundary,
        batch_size=batch_size,
        record_velocities=False,
        use_jax_jit=True,
        static_compilation=True,
        use_scan_jit=True,
        device_put_inputs=True,
        progress_callback=None
    )

    # Verify JIT
    print(f"   JIT step: {'✅ COMPILED' if tracker._compiled_step else '❌ FAILED'}")
    print(f"   JIT scan: {'✅ COMPILED' if tracker._compiled_simulate else '❌ FAILED'}")

    # Track
    print(f"\\n🏃 Running octree FEM tracking...")
    print(f"   Expect 30-60s for first JIT compilation")

    start = time.time()

    # Get time bounds from field
    t_min, t_max = field.get_time_bounds()

    trajectory = tracker.track_particles(
        initial_positions=seeds,
        time_span=(t_min, t_max),
        n_timesteps=n_timesteps,
        dt=dt
    )

    elapsed = time.time() - start

    print(f"\\n✅ Octree FEM tracking complete!")
    print(f"   Time: {elapsed:.2f} seconds")
    print(f"   Rate: {len(seeds) * n_timesteps / elapsed:.0f} particle-steps/sec")

    return trajectory


def export_results(trajectory):
    """Export trajectory."""

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    try:
        print(f"💾 Exporting...")
        export_trajectory_to_vtk(
            trajectory=trajectory,
            filename=str(output_dir / "trajectory_octree_fem.vtp"),
            include_velocities=False,
            time_series=False
        )
        print(f"   ✅ Saved: output/trajectory_octree_fem.vtp")
    except Exception as e:
        print(f"   ⚠️  Export failed: {e}")


if __name__ == "__main__":
    main()
