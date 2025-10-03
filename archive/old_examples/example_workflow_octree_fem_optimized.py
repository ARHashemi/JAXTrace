#!/usr/bin/env python3
"""
JAXTrace OPTIMIZED Octree FEM Workflow

Optimizations:
1. Fixed-depth octree traversal (no while_loop bottleneck)
2. Early termination in element scan
3. Cheap local fallback (no global search)
4. Reduced memory footprint

For meshes with 6 refinement levels - FAST & ACCURATE!
"""

import os
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
from jaxtrace.fields.octree_fem_time_series_optimized import OctreeFEMTimeSeriesFieldOptimized
from jaxtrace.tracking import create_tracker, uniform_grid_seeds
from jaxtrace.tracking.boundary import reflective_boundary
from jaxtrace.io import export_trajectory_to_vtk

print("="*80)
print(f"JAXTrace v{jt.__version__} - OPTIMIZED Octree FEM Workflow")
print("="*80)


def main():
    """Main workflow."""

    # 1. Configure
    print("\\n1. GPU CONFIGURATION")
    print("="*80)

    jt.configure(dtype="float32", device="gpu", memory_limit_gb=3.0)
    print(f"✅ Device: {jax.devices()}")

    # 2. Load
    print("\\n2. LOAD WITH OPTIMIZED OCTREE FEM")
    print("="*80)

    field = load_vtk_with_optimized_octree_fem()

    # 3. Track
    print("\\n3. GPU-ACCELERATED TRACKING")
    print("="*80)

    trajectory = fast_tracking(field)

    # 4. Export
    print("\\n4. EXPORT")
    print("="*80)

    export_results(trajectory)

    print("\\n" + "="*80)
    print("✅ OPTIMIZED OCTREE FEM WORKFLOW COMPLETE!")
    print("="*80)


def load_vtk_with_optimized_octree_fem():
    """Load VTK with optimized octree FEM."""

    pattern = "/home/arhashemi/Workspace/welding/Cases/004_caseCoarse.gid/post/0eule/004_caseCoarse_*.pvtu"

    print(f"📁 Loading VTK files...")

    from glob import glob
    files = sorted(glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files found: {pattern}")

    print(f"   Found {len(files)} files")

    # Load subset
    max_timesteps = 40
    stride = max(1, len(files) // max_timesteps)
    files_to_load = files[::stride][:max_timesteps]

    print(f"   Loading {len(files_to_load)} timesteps...")

    # Load first file for mesh
    print(f"   Reading mesh...")
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(files_to_load[0])
    reader.Update()
    mesh = reader.GetOutput()

    # Extract mesh
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float32)
    n_points = points.shape[0]

    print(f"   Mesh: {n_points} points")

    # Extract connectivity
    connectivity = []
    for i in range(mesh.GetNumberOfCells()):
        cell = mesh.GetCell(i)
        if cell.GetCellType() == vtk.VTK_TETRA:
            point_ids = cell.GetPointIds()
            tet = [point_ids.GetId(j) for j in range(4)]
            connectivity.append(tet)

    connectivity = np.array(connectivity, dtype=np.int32)
    print(f"   Elements: {connectivity.shape[0]} tetrahedra")

    # Load velocity data
    velocity_data = []
    times = []

    for idx, filename in enumerate(files_to_load):
        reader = vtk.vtkXMLPUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        mesh = reader.GetOutput()

        point_data = mesh.GetPointData()
        vel_array = None

        for name in ['Displacement', 'displacement', 'Velocity', 'velocity']:
            if point_data.HasArray(name):
                vel_array = point_data.GetArray(name)
                break

        if vel_array is None:
            raise ValueError(f"No velocity field found in {filename}")

        velocity = vtk_to_numpy(vel_array).astype(np.float32)

        if velocity.shape[1] == 2:
            velocity = np.column_stack([velocity, np.zeros(velocity.shape[0])])

        velocity_data.append(velocity)

        import re
        match = re.search(r'_(\\d+)\\.pvtu$', filename)
        if match:
            times.append(float(match.group(1)))
        else:
            times.append(float(idx))

        if (idx + 1) % 10 == 0:
            print(f"   Loaded {idx + 1}/{len(files_to_load)} timesteps...")

    velocity_data = np.array(velocity_data, dtype=np.float32)
    times = np.array(times, dtype=np.float32)

    print(f"✅ Loaded velocity data: {velocity_data.shape}")

    # Create OPTIMIZED octree FEM field
    field = OctreeFEMTimeSeriesFieldOptimized(
        data=velocity_data,
        times=times,
        positions=points,
        connectivity=connectivity,
        interpolation="linear",
        extrapolation="constant",
        max_elements_per_leaf=32,
        max_depth=12
    )

    # Upload to GPU
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


def fast_tracking(field):
    """Fast tracking with optimized octree FEM."""

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
    print(f"   Batch: {batch_size}")
    print(f"   Interpolation: OPTIMIZED Octree FEM")

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

    print(f"   JIT step: {'✅' if tracker._compiled_step else '❌'}")
    print(f"   JIT scan: {'✅' if tracker._compiled_simulate else '❌'}")

    # Track
    print(f"\\n🏃 Running tracking...")
    print(f"   First run includes JIT compilation time")

    start = time.time()

    t_min, t_max = field.get_time_bounds()

    trajectory = tracker.track_particles(
        initial_positions=seeds,
        time_span=(t_min, t_max),
        n_timesteps=n_timesteps,
        dt=dt
    )

    elapsed = time.time() - start

    print(f"\\n✅ Tracking complete!")
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
            filename=str(output_dir / "trajectory_octree_fem_optimized.vtp"),
            include_velocities=False,
            time_series=False
        )
        print(f"   ✅ Saved: output/trajectory_octree_fem_optimized.vtp")
    except Exception as e:
        print(f"   ⚠️  Export failed: {e}")


if __name__ == "__main__":
    main()
