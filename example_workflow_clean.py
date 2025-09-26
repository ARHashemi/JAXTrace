#!/usr/bin/env python3
"""
JAXTrace Complete Workflow Example

This comprehensive example demonstrates the full JAXTrace particle tracking workflow,
showcasing the improved features including:

🔧 NEW FEATURES IN THIS VERSION:
- Uniform grid particle seeding with user-defined concentrations
- Continuous inlet/outlet boundary conditions with grid preservation
- Enhanced YZ density slice visualization
- Improved error handling and memory management

📊 WORKFLOW COMPONENTS:
- System diagnostics and capability checking
- VTK time series data loading with memory optimization
- Uniform grid particle seeding (instead of random)
- Flow-through boundary conditions (inlet → outlet)
- Advanced density analysis (KDE and SPH)
- Multi-plot visualization including YZ density slices
- Comprehensive trajectory analysis and reporting

💡 KEY IMPROVEMENTS:
- Grid-preserving inlet particle replacement
- Efficient progress reporting with single-line updates
- Robust error handling for density analysis
- Configurable visualization parameters

All functionality is cleanly organized in the core JAXTrace package modules.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# JAXTrace core imports
import jaxtrace as jt
from jaxtrace.fields import TimeSeriesField
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


def main():
    """Main workflow demonstrating JAXTrace capabilities."""

    # 1. System diagnostics
    print("="*80)
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
        device="cpu",
        memory_limit_gb=8.0
    )

    config = jt.get_config()
    print(f"✅ JAXTrace configured: {config}")

    # 3. Load or create velocity field
    print("\n" + "="*80)
    print("3. VELOCITY FIELD")
    print("="*80)

    # Try to load VTK data, fallback to synthetic field
    field = create_or_load_velocity_field()

    # 4. Particle seeding and tracking
    print("\n" + "="*80)
    print("4. PARTICLE TRACKING")
    print("="*80)

    # Custom particle concentrations (particles per unit length)
    custom_concentrations = {
        'x': 60,  # High concentration in flow direction
        'y': 50,  # Medium concentration across width
        'z': 15    # Lower concentration in height
    }

    trajectory, strategy_info = execute_particle_tracking(field, custom_concentrations)

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

    # YZ slice density parameters (user configurable)
    # These parameters control the new YZ density slice visualization
    slice_x0 = None       # X position for slice plane (default: 0.7 * x_max)
                         # Examples: 0.5, 1.2, bounds_max[0]*0.8
    slice_levels = 20     # Number of contour levels or array of specific levels
                         # Examples: 15, 25, [0.1, 0.5, 1.0, 2.0, 5.0]
    slice_cutoff = 95     # Percentile cutoff for intensity (removes extreme outliers)
                         # Examples: 90, 99, 100 (no cutoff)

    create_visualizations(trajectory, density_results,
                         slice_x0=slice_x0, slice_levels=slice_levels, slice_cutoff=slice_cutoff)

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


def create_or_load_velocity_field():
    """Load VTK data or create synthetic field."""

    # Try to load VTK data
    data_patterns = [
        "/home/arhashemi/Workspace/welding/Cases/004_caseCoarse.gid/post/0eule/",
        "data/**/caseCoarse_*.pvtu",
        "example_data/**/case*_*.vtu",
        "../example_data/**/case*_*.vtu"
    ]

    for pattern in data_patterns:
        try:
            print(f"🔍 Trying pattern: {pattern}")
            # Load only the last 40 time steps
            dataset = open_dataset(pattern, max_time_steps=40)

            # Convert to TimeSeriesField based on dataset type
            if hasattr(dataset, 'load_time_series'):
                # It's a VTK reader - load the time series data
                print("📦 Loading time series data from VTK reader...")
                time_series_data = dataset.load_time_series()

                field = TimeSeriesField(
                    data=time_series_data['velocity_data'],
                    times=time_series_data['times'],
                    positions=time_series_data['positions'],
                    interpolation="linear",
                    extrapolation="constant",
                    _source_info=time_series_data
                )
            elif isinstance(dataset, dict):
                # It's already loaded time series data
                print("📦 Converting dataset dict to TimeSeriesField...")
                field = TimeSeriesField(
                    data=dataset['velocity_data'],
                    times=dataset['times'],
                    positions=dataset['positions'],
                    interpolation="linear",
                    extrapolation="constant",
                    _source_info=dataset
                )
            else:
                # Assume it's already a field object
                field = dataset

            print(f"✅ Loaded VTK data: {field}")
            return field
        except Exception as e:
            print(f"   ⚠️  Failed: {e}")

    # Fallback: create synthetic time-dependent vortex field
    print("📝 Creating synthetic time-dependent vortex field...")
    field = create_synthetic_vortex_field()
    print(f"✅ Created synthetic field: {field}")
    return field


def create_synthetic_vortex_field():
    """Create a synthetic time-dependent vortex field."""

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


def execute_particle_tracking(field, concentrations=None):
    """Execute particle tracking with uniform distribution and flow-through boundaries."""

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
    par_bounds = [bounds_min,
                  [bounds_max[0],#bounds_min[0] + 0.2 * domain_size[0],
                   bounds_max[1],
                   bounds_max[2]]]

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

    # Generate uniform grid of particles across the entire domain
    seeds = uniform_grid_seeds(
        resolution=(nx, ny, nz),
        bounds= par_bounds,
        include_boundaries=True
    )

    print(f"✅ Generated {len(seeds)} uniform particles across domain")

    # Setup tracking configuration
    strategy_info = {
        'name': 'RK4 with Flow-Through Boundaries (Inlet/Outlet)',
        'integrator': 'rk4',
        'n_timesteps': 2000,
        'batch_size': min(len(seeds), 1000),
        'boundary_type': 'flow_through',
        'dt': 0.0025
    }

    # Create continuous inlet boundary condition:
    # - Particles continuously enter from x_min (yz plane inlet)
    # - Particles exit at x_max (yz plane outlet) and are replaced with new inlet particles
    # - Reflective boundaries on y and z directions
    full_bounds = [bounds_min, bounds_max]
    boundary = continuous_inlet_boundary_factory(
        bounds=full_bounds,
        flow_axis='x',              # Flow in x direction
        flow_direction='positive',  # From x_min to x_max
        inlet_distribution='grid',  # Grid distribution matching initial particle pattern
        concentrations=concentrations  # Pass user-defined concentrations
    )

    print("🚪 Boundary conditions:")
    print(f"   Inlet: YZ plane at x = {bounds_min[0]:.4f} (continuous injection)")
    print(f"   Outlet: YZ plane at x = {bounds_max[0]:.4f} (particles absorbed & replaced)")
    print(f"   Y boundaries: Reflective at y = [{bounds_min[1]:.4f}, {bounds_max[1]:.4f}]")
    print(f"   Z boundaries: Reflective at z = [{bounds_min[2]:.4f}, {bounds_max[2]:.4f}]")
    print(f"   Particle replacement: Exited particles replaced with new inlet particles")

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
    # Note: JAX compilation warnings are expected for complex field interpolation
    # The tracker will fall back to step-by-step execution, which works correctly
    tracker = create_tracker(
        integrator_name='rk4',
        field=field,
        boundary_condition=boundary,
        batch_size=strategy_info['batch_size'],
        record_velocities=True,
        progress_callback=progress_callback
    )

    # Run tracking
    print("🏃 Running particle tracking...")
    print(f"   Tracking {len(seeds)} particles for {strategy_info['n_timesteps']} timesteps")
    start_time = time.time()

    time_span = (0.0, 4.0)
    trajectory = tracker.track_particles(
        initial_positions=seeds,
        time_span=time_span,
        n_timesteps=strategy_info['n_timesteps'],
        dt=strategy_info['dt']
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


def create_yz_density_slice(trajectory, output_dir, x0=None, levels=None, cutoff_percentile=95):
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
    cutoff_percentile : float, optional
        Percentile cutoff for contour levels (default 95%)
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

        # Apply cutoff if specified
        if cutoff_percentile < 100:
            cutoff_value = np.percentile(density_2d, cutoff_percentile)
            density_2d = np.minimum(density_2d, cutoff_value)

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


def create_visualizations(trajectory, density_results=None, slice_x0=None, slice_levels=None, slice_cutoff=95):
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
    slice_cutoff : float, optional
        Percentile cutoff for contour levels (default 95%)
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
        create_yz_density_slice(trajectory, output_dir, x0=slice_x0, levels=slice_levels, cutoff_percentile=slice_cutoff)

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
    main()