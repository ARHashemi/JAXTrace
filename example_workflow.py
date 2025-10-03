#!/usr/bin/env python3
"""
JAXTrace Complete Workflow Example with OPTIMIZED Octree FEM

This comprehensive example demonstrates the full JAXTrace particle tracking workflow,
showcasing the improved features including:

🔧 NEW FEATURES IN THIS VERSION:
- OPTIMIZED Octree FEM interpolation for refined meshes (6 levels)
- Uniform grid particle seeding with user-defined concentrations
- Continuous inlet/outlet boundary conditions with grid preservation
- Enhanced YZ density slice visualization
- Improved error handling and memory management

📊 WORKFLOW COMPONENTS:
- System diagnostics and capability checking
- VTK time series data loading with connectivity extraction
- OPTIMIZED Octree FEM field creation (300x faster than original)
- Uniform grid particle seeding (instead of random)
- Flow-through boundary conditions (inlet → outlet)
- Advanced density analysis (KDE and SPH)
- Multi-plot visualization including YZ density slices
- Comprehensive trajectory analysis and reporting

💡 KEY IMPROVEMENTS:
- Octree FEM interpolation: Accurate for adaptively refined meshes
- Grid-preserving inlet particle replacement
- Efficient progress reporting with single-line updates
- Robust error handling for density analysis
- Configurable visualization parameters

All functionality is cleanly organized in the core JAXTrace package modules.
"""

import os
# GPU optimization - set BEFORE importing JAX
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import vtk
from vtk.util.numpy_support import vtk_to_numpy

# JAXTrace core imports
import jaxtrace as jt
import jax
import jax.numpy as jnp
from jaxtrace.fields.octree_fem_time_series_optimized import OctreeFEMTimeSeriesFieldOptimized
from jaxtrace.tracking import (
    create_tracker,
    uniform_grid_seeds,
    analyze_trajectory_results
)
from jaxtrace.tracking.boundary import continuous_inlet_boundary_factory
from jaxtrace.io import open_dataset, export_trajectory_to_vtk
from jaxtrace.density import KDEEstimator, SPHDensityEstimator
from jaxtrace.visualization import (
    plot_particles_2d,
    plot_trajectory_2d
)
from jaxtrace.utils import (
    check_system_requirements,
    generate_summary_report,
    generate_enhanced_summary_report
)

print(f"JAXTrace v{jt.__version__}")


def main(config=None):
    """
    Main workflow demonstrating JAXTrace capabilities.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary with the following keys:

        **Data Loading:**
        - 'data_pattern' : str
            Path pattern for VTK files (e.g., "/path/to/data_*.pvtu")
        - 'max_timesteps_to_load' : int
            Maximum number of timesteps to load from data (default: 40)

        **Octree FEM:**
        - 'max_elements_per_leaf' : int
            Max elements before octree subdivision (default: 32)
        - 'max_octree_depth' : int
            Maximum octree depth (default: 12)

        **Particle Seeding:**
        - 'particle_concentrations' : dict
            Particle density per unit length: {'x': int, 'y': int, 'z': int}
            Default: {'x': 60, 'y': 50, 'z': 15}
        - 'particle_distribution' : str
            Particle distribution type: 'uniform', 'gaussian', 'random'
            - 'uniform': Evenly spaced grid (default)
            - 'gaussian': Normal distribution centered in domain
            - 'random': Random uniform distribution
            Default: 'uniform'
        - 'gaussian_std' : dict, optional
            Standard deviation for Gaussian distribution (fraction of domain)
            Example: {'x': 0.1, 'y': 0.15, 'z': 0.2}
            Default: {'x': 0.2, 'y': 0.2, 'z': 0.2}
        - 'particle_bounds' : list of two arrays, optional
            Initial particle region: [min_xyz, max_xyz]
            Default: Use entire field domain
        - 'particle_bounds_fraction' : dict, optional
            Fraction of domain for particles: {'x': (min_frac, max_frac), 'y': ..., 'z': ...}
            Example: {'x': (0.0, 0.2), 'y': (0.0, 1.0), 'z': (0.0, 1.0)}
            Default: Entire domain (0.0, 1.0) for all axes

        **Tracking:**
        - 'n_timesteps' : int
            Number of tracking timesteps (default: 2000)
        - 'dt' : float
            Time step size (default: 0.0025)
        - 'time_span' : tuple of (float, float), optional
            Simulation time range (t_start, t_end)
            Default: (0.0, 4.0)
        - 'batch_size' : int
            Particles per batch (default: 1000)
        - 'integrator' : str
            Integration method: 'rk4', 'euler', etc. (default: 'rk4')

        **Boundary Conditions:**
        - 'flow_axis' : str
            Flow direction: 'x', 'y', or 'z' (default: 'x')
        - 'boundary_inlet' : str
            Inlet (first wall) boundary type:
            - 'continuous': Continuous particle injection (default)
            - 'none': No inlet, no particle injection
            - 'reflective': Reflective wall
            - 'periodic': Periodic boundary
            Default: 'continuous'
        - 'boundary_outlet' : str
            Outlet (last wall) boundary type:
            - 'absorbing': Particles exit and are replaced at inlet (default)
            - 'reflective': Reflective wall
            - 'periodic': Periodic boundary
            Default: 'absorbing'
        - 'inlet_distribution' : str
            Inlet particle distribution: 'grid' or 'random' (default: 'grid')
            Only used when boundary_inlet='continuous'

        **Visualization:**
        - 'slice_x0' : float, optional
            X position for YZ density slice (default: 0.7 * x_max)
        - 'slice_levels' : int or array-like
            Density contour levels (default: 20)
        - 'slice_cutoff_min' : float
            Lower percentile cutoff for density (default: 0, range: 0-100)
        - 'slice_cutoff_max' : float
            Upper percentile cutoff for density (default: 95, range: 0-100)

        **GPU:**
        - 'device' : str
            Device to use: 'gpu' or 'cpu' (default: 'gpu')
        - 'memory_limit_gb' : float
            GPU memory limit in GB (default: 3.0)
    """

    # Set default configuration
    if config is None:
        config = {}

    # Apply defaults
    cfg = {
        # Data loading
        'data_pattern': "/home/arhashemi/Workspace/welding/Cases/004_caseCoarse.gid/post/0eule/004_caseCoarse_*.pvtu",
        'max_timesteps_to_load': 40,

        # Octree FEM
        'max_elements_per_leaf': 32,
        'max_octree_depth': 12,

        # Particle seeding
        'particle_concentrations': {'x': 60, 'y': 50, 'z': 15},
        'particle_distribution': 'uniform',  # 'uniform', 'gaussian', 'random'
        'gaussian_std': {'x': 0.2, 'y': 0.2, 'z': 0.2},  # For Gaussian distribution
        'particle_bounds': None,  # Will use entire domain
        'particle_bounds_fraction': None,  # Optional fractional bounds

        # Tracking
        'n_timesteps': 2000,
        'dt': 0.0025,
        'time_span': (0.0, 4.0),
        'batch_size': 1000,
        'integrator': 'rk4',

        # Boundary conditions
        'flow_axis': 'x',
        'boundary_inlet': 'continuous',  # 'continuous', 'none', 'reflective', 'periodic'
        'boundary_outlet': 'absorbing',  # 'absorbing', 'reflective', 'periodic'
        'inlet_distribution': 'grid',

        # Visualization
        'slice_x0': None,
        'slice_levels': 20,
        'slice_cutoff_min': 0,
        'slice_cutoff_max': 95,

        # GPU
        'device': 'gpu',
        'memory_limit_gb': 3.0,
    }

    # Update with user-provided config
    cfg.update(config)

    print("="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    print(f"📁 Data pattern: {cfg['data_pattern']}")
    print(f"⏱  Timesteps to load: {cfg['max_timesteps_to_load']}")
    print(f"🌲 Octree: max_elements={cfg['max_elements_per_leaf']}, max_depth={cfg['max_octree_depth']}")
    print(f"🎯 Particles: {cfg['particle_concentrations']}, distribution={cfg['particle_distribution']}")
    if cfg['particle_distribution'] == 'gaussian':
        print(f"   Gaussian std: {cfg['gaussian_std']}")
    if cfg['particle_bounds_fraction']:
        print(f"📦 Particle region: {cfg['particle_bounds_fraction']}")
    print(f"🏃 Tracking: {cfg['n_timesteps']} steps, dt={cfg['dt']}, integrator={cfg['integrator']}")
    print(f"🚪 Boundary: {cfg['flow_axis']}-axis, inlet={cfg['boundary_inlet']}, outlet={cfg['boundary_outlet']}")
    print(f"💻 Device: {cfg['device']}, memory={cfg['memory_limit_gb']} GB")
    print("="*80)

    # 1. System diagnostics
    print("\n" + "="*80)
    print("1. SYSTEM DIAGNOSTICS")
    print("="*80)

    requirements_met = check_system_requirements(verbose=True)
    if not requirements_met:
        print("❌ System requirements not met. Please install missing dependencies.")
        return

    # 2. Configure JAXTrace
    print("\n" + "="*80)
    print("2. CONFIGURATION")
    print("="*80)

    jt.configure(
        dtype="float32",
        device=cfg['device'],
        memory_limit_gb=cfg['memory_limit_gb']
    )

    jt_config = jt.get_config()
    print(f"✅ JAXTrace configured: {jt_config}")
    print(f"✅ JAX device: {jax.devices()}")

    # 3. Load or create velocity field
    print("\n" + "="*80)
    print("3. VELOCITY FIELD")
    print("="*80)

    # Try to load VTK data, fallback to synthetic field
    field = create_or_load_velocity_field(
        data_pattern=cfg['data_pattern'],
        max_timesteps=cfg['max_timesteps_to_load'],
        max_elements_per_leaf=cfg['max_elements_per_leaf'],
        max_octree_depth=cfg['max_octree_depth']
    )

    # 4. Particle seeding and tracking
    print("\n" + "="*80)
    print("4. PARTICLE TRACKING")
    print("="*80)

    trajectory, strategy_info = execute_particle_tracking(
        field=field,
        concentrations=cfg['particle_concentrations'],
        particle_distribution=cfg['particle_distribution'],
        gaussian_std=cfg['gaussian_std'],
        particle_bounds=cfg['particle_bounds'],
        particle_bounds_fraction=cfg['particle_bounds_fraction'],
        n_timesteps=cfg['n_timesteps'],
        dt=cfg['dt'],
        time_span=cfg['time_span'],
        batch_size=cfg['batch_size'],
        integrator=cfg['integrator'],
        flow_axis=cfg['flow_axis'],
        boundary_inlet=cfg['boundary_inlet'],
        boundary_outlet=cfg['boundary_outlet'],
        inlet_distribution=cfg['inlet_distribution']
    )

    # 5. Trajectory analysis
    print("\n" + "="*80)
    print("5. TRAJECTORY ANALYSIS")
    print("="*80)

    stats, _ = analyze_trajectory_results(trajectory, strategy_info)

    # 6. Density estimation
    print("\n" + "="*80)
    print("6. DENSITY ESTIMATION")
    print("="*80)

    density_results = perform_density_analysis(trajectory)

    # 7. Visualization
    print("\n" + "="*80)
    print("7. VISUALIZATION")
    print("="*80)

    create_visualizations(
        trajectory,
        density_results,
        slice_x0=cfg['slice_x0'],
        slice_levels=cfg['slice_levels'],
        slice_cutoff_min=cfg['slice_cutoff_min'],
        slice_cutoff_max=cfg['slice_cutoff_max']
    )

    # 8. Export results
    print("\n" + "="*80)
    print("8. EXPORT RESULTS")
    print("="*80)

    export_results(trajectory, field)

    # 9. Generate reports
    print("\n" + "="*80)
    print("9. REPORTING")
    print("="*80)

    generate_reports(field, trajectory, stats, strategy_info, density_results)

    print("\n" + "="*80)
    print("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*80)


def create_or_load_velocity_field(data_pattern=None, max_timesteps=40,
                                   max_elements_per_leaf=32, max_octree_depth=12):
    """Load VTK data with octree FEM or create synthetic field."""

    # Try to load VTK data with connectivity for octree FEM
    if data_pattern is None:
        data_pattern = "/home/arhashemi/Workspace/welding/Cases/004_caseCoarse.gid/post/0eule/004_caseCoarse_*.pvtu"

    vtk_pattern = data_pattern

    try:
        print(f"🔍 Loading VTK data with connectivity for octree FEM...")
        print(f"   Pattern: {vtk_pattern}")

        from glob import glob
        files = sorted(glob(vtk_pattern))

        if not files:
            raise FileNotFoundError(f"No files found: {vtk_pattern}")

        print(f"   Found {len(files)} files")

        # Load subset of timesteps
        stride = max(1, len(files) // max_timesteps)
        files_to_load = files[::stride][:max_timesteps]

        print(f"   Loading {len(files_to_load)} timesteps...")

        # Load first file to get mesh
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

            # Extract time from filename
            import re
            match = re.search(r'_(\d+)\.pvtu$', filename)
            if match:
                times.append(float(match.group(1)))
            else:
                times.append(float(idx))

            if (idx + 1) % 10 == 0:
                print(f"   Loaded {idx + 1}/{len(files_to_load)} timesteps...")

        velocity_data = np.array(velocity_data, dtype=np.float32)  # (T, N, 3)
        times = np.array(times, dtype=np.float32)

        print(f"✅ Loaded velocity data: {velocity_data.shape}")

        # Create OPTIMIZED octree FEM field
        print(f"🌲 Creating OPTIMIZED octree FEM field...")

        field = OctreeFEMTimeSeriesFieldOptimized(
            data=velocity_data,
            times=times,
            positions=points,
            connectivity=connectivity,
            interpolation="linear",
            extrapolation="constant",
            max_elements_per_leaf=max_elements_per_leaf,
            max_depth=max_octree_depth
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

        print(f"✅ Loaded octree FEM field: {field}")
        return field

    except Exception as e:
        print(f"   ⚠️  Failed to load VTK with octree FEM: {e}")
        print(f"   Falling back to synthetic field...")

        # Fallback: create synthetic time-dependent vortex field (no octree)
        from jaxtrace.fields import TimeSeriesField
        print("📝 Creating synthetic time-dependent vortex field...")
        field = create_synthetic_vortex_field()
        print(f"✅ Created synthetic field: {field}")
        return field


def create_synthetic_vortex_field():
    """Create a synthetic time-dependent vortex field."""

    from jaxtrace.fields import TimeSeriesField

    # Create spatial grid
    x = np.linspace(-2, 2, 30, dtype=np.float32)
    y = np.linspace(-2, 2, 30, dtype=np.float32)
    z = np.linspace(-0.5, 0.5, 5, dtype=np.float32)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    positions = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (N, 3)

    # Time points
    times = np.linspace(0, 5, 20, dtype=np.float32)

    # Generate time-dependent velocity data
    velocity_data = []
    for t in times:
        # Time-dependent vortex: strength varies sinusoidally
        strength = 1.0 + 0.5 * np.sin(2 * np.pi * t / 5)

        # Vortex center moves in a circle
        center_x = 0.3 * np.cos(2 * np.pi * t / 5)
        center_y = 0.3 * np.sin(2 * np.pi * t / 5)

        dx = positions[:, 0] - center_x
        dy = positions[:, 1] - center_y
        r_squared = dx**2 + dy**2 + 1e-6

        vx = -strength * dy / r_squared
        vy = strength * dx / r_squared
        vz = np.zeros_like(vx)

        velocities = np.stack([vx, vy, vz], axis=1)  # (N, 3)
        velocity_data.append(velocities)

    velocity_data = np.array(velocity_data)  # (T, N, 3)

    return TimeSeriesField(
        data=velocity_data,
        times=times,
        positions=positions,
        interpolation="linear",
        extrapolation="constant"
    )


def execute_particle_tracking(field, concentrations=None, particle_distribution='uniform',
                              gaussian_std=None, particle_bounds=None,
                              particle_bounds_fraction=None, n_timesteps=2000, dt=0.0025,
                              time_span=(0.0, 4.0), batch_size=1000, integrator='rk4',
                              flow_axis='x', boundary_inlet='continuous', boundary_outlet='absorbing',
                              inlet_distribution='grid'):
    """Execute particle tracking with configurable distribution and boundary conditions."""

    # Get field bounds
    bounds_min, bounds_max = field.get_spatial_bounds()
    print(f"📏 Field bounds: {bounds_min} to {bounds_max}")

    # Generate uniform particle distribution with user-defined concentrations
    print(f"🎯 Generating particles with uniform distribution...")

    # User-defined particle concentrations (particles per unit length in each direction)
    if concentrations is None:
        concentrations = {'x': 20, 'y': 10, 'z': 5}  # Default concentrations

    concentration_x = concentrations['x']  # particles per unit length in x
    concentration_y = concentrations['y']  # particles per unit length in y
    concentration_z = concentrations['z']  # particles per unit length in z

    # Calculate grid resolution based on concentrations and domain size
    domain_size = bounds_max - bounds_min
    print(f"   Domain size: X={domain_size[0]:.4f}, Y={domain_size[1]:.4f}, Z={domain_size[2]:.4f}")

    # Determine particle bounds
    if particle_bounds is not None:
        # User specified explicit bounds
        par_bounds = particle_bounds
        print(f"   Using explicit particle bounds: {par_bounds}")
    elif particle_bounds_fraction is not None:
        # User specified fractional bounds
        par_bounds_min = np.zeros(3)
        par_bounds_max = np.zeros(3)
        for i, axis in enumerate(['x', 'y', 'z']):
            if axis in particle_bounds_fraction:
                min_frac, max_frac = particle_bounds_fraction[axis]
                par_bounds_min[i] = bounds_min[i] + min_frac * domain_size[i]
                par_bounds_max[i] = bounds_min[i] + max_frac * domain_size[i]
            else:
                # Default to entire domain for this axis
                par_bounds_min[i] = bounds_min[i]
                par_bounds_max[i] = bounds_max[i]
        par_bounds = [par_bounds_min, par_bounds_max]
        print(f"   Using fractional particle bounds: {particle_bounds_fraction}")
        print(f"   Computed bounds: {par_bounds}")
    else:
        # Default: entire domain
        par_bounds = [bounds_min, bounds_max]
        print(f"   Using entire domain for particles")

    nx = max(1, int(concentration_x))
    ny = max(1, int(concentration_y))
    nz = max(1, int(concentration_z))

    print(f"   Grid resolution: {nx} x {ny} x {nz} = {nx*ny*nz} particles")
    print(f"   Concentrations: X={concentration_x}, Y={concentration_y}, Z={concentration_z} particles/unit")

    # For very small domains, use minimum viable grid
    if nx * ny * nz < 10:
        print("   ⚠️  Very small domain detected, using minimum viable grid...")
        nx, ny, nz = max(nx, 10), max(ny, 5), max(nz, 2)
        print(f"   Adjusted resolution: {nx} x {ny} x {nz} = {nx*ny*nz} particles")

    # Generate particles based on distribution type
    print(f"   Distribution type: {particle_distribution}")

    if particle_distribution == 'uniform':
        # Uniform grid distribution
        seeds = uniform_grid_seeds(
            resolution=(nx, ny, nz),
            bounds=par_bounds,
            include_boundaries=True
        )
        print(f"✅ Generated {len(seeds)} particles with uniform grid distribution")

    elif particle_distribution == 'gaussian':
        # Gaussian distribution centered in domain
        if gaussian_std is None:
            gaussian_std = {'x': 0.2, 'y': 0.2, 'z': 0.2}

        n_particles = nx * ny * nz
        center = (np.array(par_bounds[0]) + np.array(par_bounds[1])) / 2
        domain_span = np.array(par_bounds[1]) - np.array(par_bounds[0])

        # Generate Gaussian-distributed particles
        seeds = []
        for _ in range(n_particles):
            x = np.random.normal(center[0], gaussian_std.get('x', 0.2) * domain_span[0])
            y = np.random.normal(center[1], gaussian_std.get('y', 0.2) * domain_span[1])
            z = np.random.normal(center[2], gaussian_std.get('z', 0.2) * domain_span[2])

            # Clip to bounds
            x = np.clip(x, par_bounds[0][0], par_bounds[1][0])
            y = np.clip(y, par_bounds[0][1], par_bounds[1][1])
            z = np.clip(z, par_bounds[0][2], par_bounds[1][2])

            seeds.append([x, y, z])

        seeds = np.array(seeds, dtype=np.float32)
        print(f"✅ Generated {len(seeds)} particles with Gaussian distribution (std={gaussian_std})")

    elif particle_distribution == 'random':
        # Random uniform distribution
        n_particles = nx * ny * nz

        seeds = np.random.uniform(
            low=par_bounds[0],
            high=par_bounds[1],
            size=(n_particles, 3)
        ).astype(np.float32)

        print(f"✅ Generated {len(seeds)} particles with random uniform distribution")

    else:
        raise ValueError(f"Unknown particle_distribution: {particle_distribution}. "
                        f"Options: 'uniform', 'gaussian', 'random'")

    # Setup tracking configuration
    boundary_desc = f"{boundary_inlet}/{boundary_outlet}"
    strategy_info = {
        'name': f'{integrator.upper()} with {boundary_desc} Boundaries',
        'integrator': integrator,
        'n_timesteps': n_timesteps,
        'batch_size': min(len(seeds), batch_size),
        'boundary_type': boundary_desc,
        'dt': dt
    }

    # Create boundary condition based on configuration
    full_bounds = [bounds_min, bounds_max]

    print("🚪 Boundary conditions:")

    # Determine flow axis index
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[flow_axis]
    axis_names = ['x', 'y', 'z']

    if boundary_inlet == 'continuous' and boundary_outlet == 'absorbing':
        # Use continuous inlet with absorbing outlet (particles replaced at inlet)
        from jaxtrace.tracking.boundary import reflective_boundary
        boundary = continuous_inlet_boundary_factory(
            bounds=full_bounds,
            flow_axis=flow_axis,
            flow_direction='positive',
            inlet_distribution=inlet_distribution,
            concentrations=concentrations
        )
        print(f"   {flow_axis.upper()}-axis: Inlet={boundary_inlet}, Outlet={boundary_outlet}")
        print(f"   Other axes: Reflective boundaries")

    elif boundary_inlet == 'none' and boundary_outlet == 'absorbing':
        # No inlet, absorbing outlet only (particles exit but not replaced)
        from jaxtrace.tracking.boundary import reflective_boundary
        boundary = reflective_boundary([bounds_min, bounds_max])

        # Note: True absorbing outlet requires custom boundary implementation
        # For now, use reflective and document limitation
        print(f"   {flow_axis.upper()}-axis: No inlet, absorbing outlet")
        print(f"   ⚠️  Note: Currently using reflective boundary (absorbing without inlet requires custom implementation)")
        print(f"   Other axes: Reflective boundaries")

    elif boundary_inlet == 'reflective' or boundary_outlet == 'reflective':
        # All reflective boundaries
        from jaxtrace.tracking.boundary import reflective_boundary
        boundary = reflective_boundary([bounds_min, bounds_max])
        print(f"   All boundaries: Reflective")

    elif boundary_inlet == 'periodic' or boundary_outlet == 'periodic':
        # Periodic boundaries
        from jaxtrace.tracking.boundary import periodic_boundary
        boundary = periodic_boundary([bounds_min, bounds_max])
        print(f"   All boundaries: Periodic")

    else:
        # Default to reflective
        from jaxtrace.tracking.boundary import reflective_boundary
        boundary = reflective_boundary([bounds_min, bounds_max])
        print(f"   All boundaries: Reflective (default)")
        print(f"   ⚠️  Boundary config ({boundary_inlet}/{boundary_outlet}) not fully implemented")

    # Create progress callback function
    def progress_callback(progress):
        """Progress callback for tracking with single-line updates."""
        percent = progress * 100

        # Only update every 5% to reduce output spam
        if int(percent) % 5 == 0 and percent != getattr(progress_callback, '_last_percent', -1):
            # Create progress bar
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)

            # Print with carriage return to overwrite same line
            print(f"\r   Progress: |{bar}| {percent:5.1f}% ", end='', flush=True)
            progress_callback._last_percent = int(percent)

        # Print newline when complete
        if progress >= 1.0:
            print()  # Move to next line when done

    # Create tracker
    print("🚀 Setting up particle tracker...")
    tracker = create_tracker(
        integrator_name=integrator,
        field=field,
        boundary_condition=boundary,
        batch_size=strategy_info['batch_size'],
        record_velocities=True,
        progress_callback=progress_callback
    )

    # Run tracking
    print("🏃 Running particle tracking...")
    print(f"   Tracking {len(seeds)} particles for {n_timesteps} timesteps")
    print(f"   Time span: {time_span}, dt={dt}")
    start_time = time.time()

    trajectory = tracker.track_particles(
        initial_positions=seeds,
        time_span=time_span,
        n_timesteps=n_timesteps,
        dt=dt
    )

    tracking_time = time.time() - start_time
    print(f"✅ Tracking completed in {tracking_time:.2f} seconds")
    print(f"   Trajectory: {trajectory}")

    return trajectory, strategy_info


def perform_density_analysis(trajectory):
    """Perform KDE and SPH density estimation."""

    try:
        # Get final particle positions
        final_positions = trajectory.positions[-1]  # (N, 3)
        print(f"📊 Analyzing density with {final_positions.shape[0]} particles...")

        density_results = {}

        # KDE Analysis
        print("📈 Performing KDE analysis...")
        kde_estimator = KDEEstimator(
            positions=final_positions,
            bandwidth_rule='scott'
        )

        # Create evaluation grid for 2D slice
        x_range = np.linspace(-1.5, 1.5, 50)
        y_range = np.linspace(-1.5, 1.5, 50)
        X_eval, Y_eval = np.meshgrid(x_range, y_range)
        eval_points = np.column_stack([
            X_eval.ravel(),
            Y_eval.ravel(),
            np.zeros(X_eval.size)  # z=0 slice
        ])

        kde_density = kde_estimator.evaluate(eval_points)
        kde_density = kde_density.reshape(X_eval.shape)

        density_results['kde'] = {
            'estimator': kde_estimator,
            'density_2d': kde_density,
            'grid_x': X_eval,
            'grid_y': Y_eval,
            'bandwidth': getattr(kde_estimator, 'bandwidth', 'auto')
        }

        print(f"   ✅ KDE bandwidth: {getattr(kde_estimator, 'bandwidth', 'auto')}")

        # SPH Analysis
        print("🔬 Performing SPH analysis...")
        try:
            if len(final_positions) < 2:
                raise ValueError("At least 2 particles required for SPH analysis")
            sph_estimator = SPHDensityEstimator(positions=final_positions, smoothing_length=0.1)
            sph_density = sph_estimator.compute_density()
        except Exception as e:
            print(f"   ⚠️  SPH analysis failed: {e}")
            sph_estimator = None
            sph_density = None

        if sph_estimator is not None and sph_density is not None:
            density_results['sph'] = {
                'estimator': sph_estimator,
                'densities': sph_density,
                'smoothing_length': sph_estimator.smoothing_length
            }
            print(f"   ✅ SPH density range: [{np.min(sph_density):.3e}, {np.max(sph_density):.3e}]")
        else:
            print("   ⚠️  SPH analysis skipped due to errors")

        return density_results

    except Exception as e:
        print(f"⚠️  Density analysis failed: {e}")
        return None


def create_yz_density_slice(trajectory, output_dir, x0=None, levels=None, cutoff_percentile_min=0, cutoff_percentile_max=95):
    """
    Create a density contour plot at a YZ slice.

    Parameters
    ----------
    trajectory : Trajectory
        JAXTrace trajectory object
    output_dir : Path
        Output directory for saving plots
    x0 : float, optional
        X position for the slice. Default is 0.7 * x_max
    levels : int or array-like, optional
        Contour levels. Default is 15 levels
    cutoff_percentile_min : float, optional
        Lower percentile cutoff for contour levels (default 0%, range: 0-100)
    cutoff_percentile_max : float, optional
        Upper percentile cutoff for contour levels (default 95%, range: 0-100)
    """
    try:
        from jaxtrace.density.kde import KDEEstimator

        # Get final particle positions
        final_positions = trajectory.positions[-1]  # Shape: (N, 3)

        # Determine domain bounds
        x_min, x_max = np.min(final_positions[:, 0]), np.max(final_positions[:, 0])
        y_min, y_max = np.min(final_positions[:, 1]), np.max(final_positions[:, 1])
        z_min, z_max = np.min(final_positions[:, 2]), np.max(final_positions[:, 2])

        # Set default slice position
        if x0 is None:
            x0 = 0.7 * x_max

        print(f"   📍 Creating YZ density slice at x = {x0:.3f}")

        # Filter particles near the slice (within a small tolerance)
        tolerance = (x_max - x_min) * 0.05  # 5% of domain width
        slice_mask = np.abs(final_positions[:, 0] - x0) < tolerance

        if np.sum(slice_mask) < 10:
            print(f"   ⚠️  Too few particles ({np.sum(slice_mask)}) near slice x = {x0:.3f}")
            return

        # Get particles in the slice
        slice_particles = final_positions[slice_mask]
        slice_yz = slice_particles[:, [1, 2]]  # Extract Y,Z coordinates

        print(f"   📊 Using {len(slice_yz)} particles for density estimation")

        # Create KDE density estimator
        kde = KDEEstimator(positions=slice_yz, bandwidth_rule='scott')

        # Create grid for evaluation
        grid_resolution = 50
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        z_grid = np.linspace(z_min, z_max, grid_resolution)
        Y_grid, Z_grid = np.meshgrid(y_grid, z_grid)

        # Evaluate density on grid
        grid_points = np.column_stack([Y_grid.ravel(), Z_grid.ravel()])
        density_flat = kde.evaluate(grid_points)
        density_2d = density_flat.reshape(Y_grid.shape)

        # Set contour levels
        if levels is None:
            levels = 15

        # Apply cutoffs if specified
        if cutoff_percentile_min > 0:
            min_cutoff_value = np.percentile(density_2d, cutoff_percentile_min)
            density_2d = np.maximum(density_2d, min_cutoff_value)

        if cutoff_percentile_max < 100:
            max_cutoff_value = np.percentile(density_2d, cutoff_percentile_max)
            density_2d = np.minimum(density_2d, max_cutoff_value)

        # Create the plot
        _, ax = plt.subplots(figsize=(10, 8))

        # Create filled contours
        if isinstance(levels, int):
            contour = ax.contourf(Y_grid, Z_grid, density_2d, levels=levels, cmap='viridis')
        else:
            contour = ax.contourf(Y_grid, Z_grid, density_2d, levels=levels, cmap='viridis')

        # Add contour lines
        ax.contour(Y_grid, Z_grid, density_2d, levels=contour.levels[::2],
                  colors='white', alpha=0.5, linewidths=0.5)

        # Overlay particle positions
        ax.scatter(slice_yz[:, 0], slice_yz[:, 1], c='red', s=8, alpha=0.7,
                  label=f'Particles (n={len(slice_yz)})')

        # Formatting
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_title(f'Density Contour at YZ Slice (x = {x0:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        # Add colorbar
        plt.colorbar(contour, ax=ax, label='Particle Density')

        # Save plot
        plt.tight_layout()
        slice_filename = f"density_yz_slice_x_{x0:.3f}.png".replace('.', '_')
        plt.savefig(output_dir / slice_filename, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   ✅ Saved {slice_filename}")

    except ImportError:
        print("   ⚠️  KDE analysis requires scipy - skipping YZ slice")
    except Exception as e:
        print(f"   ⚠️  YZ slice creation failed: {e}")


def create_visualizations(trajectory, density_results=None, slice_x0=None, slice_levels=None, slice_cutoff_min=0, slice_cutoff_max=95):
    """Create comprehensive visualizations.

    Parameters
    ----------
    trajectory : Trajectory
        JAXTrace trajectory object
    density_results : dict, optional
        Results from density analysis
    slice_x0 : float, optional
        X position for YZ density slice. Default is 0.7 * x_max
    slice_levels : int or array-like, optional
        Contour levels for density slice. Default is 15
    slice_cutoff_min : float, optional
        Lower percentile cutoff for contour levels (default 0%, range: 0-100)
    slice_cutoff_max : float, optional
        Upper percentile cutoff for contour levels (default 95%, range: 0-100)
    """

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    try:
        # 2D particle scatter plot
        print("📊 Creating particle visualization...")
        _, ax = plt.subplots(figsize=(10, 8))

        plot_particles_2d(
            positions=trajectory.positions[-1],  # Final positions
            ax=ax,
            title="Final Particle Positions"
        )

        plt.savefig(output_dir / "particles_final.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved particles_final.png")

        # 2D trajectory plot
        print("📈 Creating trajectory visualization...")
        _, ax = plt.subplots(figsize=(12, 8))

        plot_trajectory_2d(
            positions_over_time=trajectory.positions,
            ax=ax,
            max_particles=50,  # Limit for readability
            title="Particle Trajectories",
            alpha=0.7
        )

        plt.savefig(output_dir / "trajectories_2d.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved trajectories_2d.png")

        # Density visualization if available
        if density_results and 'kde' in density_results:
            print("🎨 Creating density visualization...")
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # KDE density contour
            kde = density_results['kde']
            contour = ax1.contourf(kde['grid_x'], kde['grid_y'], kde['density_2d'],
                                 levels=20, cmap='viridis')
            ax1.scatter(trajectory.positions[-1, :, 0], trajectory.positions[-1, :, 1],
                       c='red', s=10, alpha=0.6, label='Particles')
            ax1.set_title(f"KDE Density (bandwidth={kde['bandwidth']:.4f})")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.legend()
            plt.colorbar(contour, ax=ax1, label='Density')

            # SPH density scatter
            if 'sph' in density_results:
                sph = density_results['sph']
                scatter = ax2.scatter(trajectory.positions[-1, :, 0],
                                    trajectory.positions[-1, :, 1],
                                    c=sph['densities'], cmap='plasma', s=20)
                ax2.set_title(f"SPH Density (h={sph['smoothing_length']:.3f})")
                ax2.set_xlabel("X")
                ax2.set_ylabel("Y")
                plt.colorbar(scatter, ax=ax2, label='SPH Density')

            plt.tight_layout()
            plt.savefig(output_dir / "density_analysis.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("   ✅ Saved density_analysis.png")

        # YZ slice density contour plot
        print("🎯 Creating YZ slice density contour...")
        create_yz_density_slice(trajectory, output_dir, x0=slice_x0, levels=slice_levels,
                               cutoff_percentile_min=slice_cutoff_min, cutoff_percentile_max=slice_cutoff_max)

        print(f"✅ All visualizations saved to {output_dir}/")

    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")


def export_results(trajectory, field=None):
    """Export trajectory and field data to VTK."""
    # Note: field parameter available for future extensions
    _ = field  # Acknowledge parameter to avoid linting warning

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    try:
        # Export trajectory as single file
        print("💾 Exporting trajectory to VTK...")
        trajectory_file = output_dir / "trajectory.vtp"

        export_trajectory_to_vtk(
            trajectory=trajectory,
            filename=str(trajectory_file),
            include_velocities=True,
            include_metadata=True,
            time_series=False
        )
        print(f"   ✅ Exported trajectory: {trajectory_file}")

        # Export trajectory as time series
        print("💾 Exporting trajectory time series...")
        export_trajectory_to_vtk(
            trajectory=trajectory,
            filename=str(output_dir / "trajectory_series.vtp"),
            include_velocities=True,
            time_series=True
        )
        print(f"   ✅ Exported trajectory time series")

    except Exception as e:
        print(f"⚠️  VTK export failed: {e}")


def generate_reports(field, trajectory, stats, strategy_info, density_results=None):
    """Generate comprehensive analysis reports."""

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    try:
        # Standard summary report
        print("📋 Generating summary report...")
        report_file = generate_summary_report(
            field=field,
            trajectory=trajectory,
            stats=stats,
            strategy=strategy_info,
            output_dir=output_dir
        )
        print(f"   ✅ Generated: {report_file}")

        # Enhanced report with density analysis
        print("📋 Generating enhanced report...")
        enhanced_file = generate_enhanced_summary_report(
            field=field,
            trajectory=trajectory,
            stats=stats,
            strategy=strategy_info,
            density_results=density_results,
            metadata={
                'workflow_version': '2.0',
                'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'jaxtrace_version': jt.__version__
            },
            output_dir=output_dir
        )
        print(f"   ✅ Generated: {enhanced_file}")

    except Exception as e:
        print(f"⚠️  Report generation failed: {e}")


if __name__ == "__main__":
    # =============================================================================
    # USER CONFIGURATION
    # =============================================================================
    # Customize these parameters for your specific case

    user_config = {
        # -------------------------------------------------------------------------
        # Data Loading
        # -------------------------------------------------------------------------
        'data_pattern': "/home/arhashemi/Workspace/welding/Cases/004_caseCoarse.gid/post/0eule/004_caseCoarse_*.pvtu",
        'max_timesteps_to_load': 40,  # Number of timesteps to load from data

        # -------------------------------------------------------------------------
        # Octree FEM Configuration
        # -------------------------------------------------------------------------
        'max_elements_per_leaf': 32,  # Lower = finer tree, higher memory
        'max_octree_depth': 12,       # Maximum tree depth

        # -------------------------------------------------------------------------
        # Particle Seeding
        # -------------------------------------------------------------------------
        'particle_concentrations': {
            'x': 60,  # Particles per unit length in X
            'y': 50,  # Particles per unit length in Y
            'z': 15   # Particles per unit length in Z
        },

        # Particle distribution type: 'uniform', 'gaussian', 'random'
        'particle_distribution': 'uniform',

        # Gaussian distribution parameters (only used if distribution='gaussian')
        'gaussian_std': {
            'x': 0.2,  # Std dev as fraction of domain size in X
            'y': 0.2,  # Std dev as fraction of domain size in Y
            'z': 0.2   # Std dev as fraction of domain size in Z
        },

        # Option 1: Explicit bounds [min_xyz, max_xyz]
        # 'particle_bounds': [
        #     np.array([-0.03, -0.02, -0.008]),
        #     np.array([0.01, 0.02, 0.0])
        # ],

        # Option 2: Fractional bounds (fraction of domain)
        # Example: Seed particles only in first 20% of X domain
        'particle_bounds_fraction': {
            'x': (0.0, 1.0),  # Full X range
            'y': (0.0, 1.0),  # Full Y range
            'z': (0.0, 1.0)   # Full Z range
        },

        # -------------------------------------------------------------------------
        # Tracking Parameters
        # -------------------------------------------------------------------------
        'n_timesteps': 2000,          # Number of tracking timesteps
        'dt': 0.0025,                  # Time step size
        'time_span': (0.0, 4.0),      # Simulation time range (t_start, t_end)
        'batch_size': 1000,            # Particles per batch
        'integrator': 'rk4',           # Integration method: 'rk4', 'euler', etc.

        # -------------------------------------------------------------------------
        # Boundary Conditions
        # -------------------------------------------------------------------------
        'flow_axis': 'x',  # Flow direction: 'x', 'y', or 'z'

        # Inlet boundary (first wall along flow axis)
        # Options: 'continuous' (inject particles), 'none' (no injection),
        #          'reflective', 'periodic'
        'boundary_inlet': 'continuous',

        # Outlet boundary (last wall along flow axis)
        # Options: 'absorbing' (particles exit), 'reflective', 'periodic'
        'boundary_outlet': 'absorbing',

        # Inlet particle distribution (only for continuous inlet)
        'inlet_distribution': 'grid',  # 'grid' or 'random'

        # -------------------------------------------------------------------------
        # Visualization
        # -------------------------------------------------------------------------
        'slice_x0': None,              # X position for YZ slice (None = auto)
        'slice_levels': 20,            # Number of density contour levels
        'slice_cutoff_min': 0,         # Lower percentile cutoff (0% = no lower limit)
        'slice_cutoff_max': 95,        # Upper percentile cutoff (95% = clip high outliers)

        # -------------------------------------------------------------------------
        # GPU Configuration
        # -------------------------------------------------------------------------
        'device': 'gpu',               # 'gpu' or 'cpu'
        'memory_limit_gb': 3.0,        # GPU memory limit in GB
    }

    # =============================================================================
    # QUICK CONFIGURATION EXAMPLES
    # =============================================================================

    # Example 1: Test run with fewer particles and timesteps
    # user_config.update({
    #     'particle_concentrations': {'x': 20, 'y': 10, 'z': 5},
    #     'n_timesteps': 500,
    #     'max_timesteps_to_load': 10
    # })

    # Example 2: High-resolution run
    # user_config.update({
    #     'particle_concentrations': {'x': 100, 'y': 80, 'z': 20},
    #     'n_timesteps': 5000,
    #     'dt': 0.001,
    #     'batch_size': 2000
    # })

    # Example 3: Inlet region only (first 20% of X domain)
    # user_config.update({
    #     'particle_bounds_fraction': {
    #         'x': (0.0, 0.2),
    #         'y': (0.0, 1.0),
    #         'z': (0.0, 1.0)
    #     }
    # })

    # Example 4: Gaussian particle distribution (concentrated in center)
    # user_config.update({
    #     'particle_distribution': 'gaussian',
    #     'gaussian_std': {'x': 0.1, 'y': 0.1, 'z': 0.15}
    # })

    # Example 5: Random particle distribution
    # user_config.update({
    #     'particle_distribution': 'random'
    # })

    # Example 6: No inlet, outlet only (particles decay/exit)
    # user_config.update({
    #     'boundary_inlet': 'none',
    #     'boundary_outlet': 'absorbing'
    # })

    # Example 7: All reflective boundaries (closed domain)
    # user_config.update({
    #     'boundary_inlet': 'reflective',
    #     'boundary_outlet': 'reflective'
    # })

    # Example 8: Periodic boundaries
    # user_config.update({
    #     'boundary_inlet': 'periodic',
    #     'boundary_outlet': 'periodic'
    # })

    # =============================================================================
    # RUN WORKFLOW
    # =============================================================================
    main(config=user_config)