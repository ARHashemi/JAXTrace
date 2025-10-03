#!/usr/bin/env python3
"""
JAXTrace Memory-Optimized Workflow Example

This example demonstrates memory optimization techniques for large datasets,
specifically designed for tracking ~500MB per timestep datasets with detailed
GPU memory monitoring and logging.

🔧 MEMORY OPTIMIZATIONS:
- Disabled JAX memory preallocation
- Essential fields only (velocity + mesh coordinates)
- Chunked data loading (10 timesteps at a time)
- float32 precision (instead of float64)
- Real-time GPU memory tracking and logging
- Variable-level memory allocation monitoring

📊 MONITORING FEATURES:
- Detailed GPU memory usage logging to file
- Memory usage tracking for each variable and process
- Memory leak detection
- Performance bottleneck identification
- Memory optimization recommendations

💡 DESIGNED FOR:
- Large VTK datasets (500MB+ per timestep)
- GPU-accelerated particle tracking
- Memory-constrained environments
- Production workflows requiring optimization

All memory tracking data is logged to timestamped JSON files for analysis.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# GPU memory optimization - MUST be set BEFORE importing JAX/JAXTrace
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Don't preallocate all GPU memory
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"   # Use up to 75% of GPU (3GB of 4GB)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # Use platform allocator for better control

# Performance optimization
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"  # Adjust path if needed

# JAXTrace core imports
import jaxtrace as jt
from jaxtrace.fields import TimeSeriesField
from jaxtrace.tracking import (
    create_tracker,
    uniform_grid_seeds,
    analyze_trajectory_results
)
from jaxtrace.tracking.boundary import continuous_inlet_boundary_factory
from jaxtrace.io import export_trajectory_to_vtk
from jaxtrace.io.memory_optimized_loader import (
    load_optimized_dataset,
    estimate_memory_usage,
    MemoryOptimizedLoader,
    MemoryOptimizedConfig
)
from jaxtrace.density import KDEEstimator, SPHDensityEstimator
from jaxtrace.visualization import (
    plot_particles_2d,
    plot_trajectory_2d
)
from jaxtrace.utils import (
    check_system_requirements,
    generate_summary_report,
    initialize_memory_tracking,
    get_memory_tracker,
    track_memory,
    track_variable_memory,
    track_operation_memory,
    configure_memory_optimization,
    memory_tracked
)

print(f"JAXTrace v{jt.__version__} - Memory Optimized Workflow")
print("="*80)


def main():
    """Main memory-optimized workflow."""

    # 1. Initialize memory tracking FIRST
    print("1. MEMORY TRACKING INITIALIZATION")
    print("="*80)

    memory_tracker = initialize_memory_tracking(
        log_file="memory_tracking_detailed.json",
        enable_detailed_tracking=True,
        track_stack_traces=False  # Disable for performance
    )

    # Configure memory optimization
    configure_memory_optimization()

    # Set baseline memory usage
    memory_tracker.set_baseline()
    track_memory("workflow_start")

    # 2. System diagnostics with memory tracking
    print("\n2. SYSTEM DIAGNOSTICS")
    print("="*80)

    with track_operation_memory("system_diagnostics"):
        requirements_met = check_system_requirements(verbose=True)
        if not requirements_met:
            print("❌ System requirements not met. Please install missing dependencies.")
            return

    # 3. Configure JAXTrace with memory constraints
    print("\n3. MEMORY-OPTIMIZED CONFIGURATION")
    print("="*80)

    with track_operation_memory("jaxtrace_configuration"):
        jt.configure(
            dtype="float32",      # Use float32 for memory efficiency
            device="gpu",         # Use GPU for acceleration (you have 4GB available)
            memory_limit_gb=3.0   # Use 3GB of GPU memory (75% of 4GB)
        )

        config = jt.get_config()
        print(f"✅ JAXTrace configured: {config}")
        print(f"   💡 GPU memory optimization: Using 3GB of 4GB available")
        print(f"   💡 This keeps data on GPU, reducing transfer delays")
        track_variable_memory("jaxtrace_config", config)

    # 4. Memory-optimized velocity field loading
    print("\n4. MEMORY-OPTIMIZED DATA LOADING")
    print("="*80)

    field = load_memory_optimized_field()

    # 5. Memory-efficient particle tracking
    print("\n5. MEMORY-EFFICIENT PARTICLE TRACKING")
    print("="*80)

    # Optimized particle concentrations for large datasets
    memory_optimized_concentrations = {
        'x': 30,  # Reduced from 60 to save memory
        'y': 25,  # Reduced from 50 to save memory
        'z': 10   # Reduced from 15 to save memory
    }

    trajectory, strategy_info = execute_memory_optimized_tracking(
        field, memory_optimized_concentrations
    )

    # 6. Memory-aware trajectory analysis
    print("\n6. MEMORY-AWARE ANALYSIS")
    print("="*80)

    with track_operation_memory("trajectory_analysis"):
        stats, _ = analyze_trajectory_results(trajectory, strategy_info)
        track_variable_memory("trajectory_stats", stats)

    # 7. Memory-efficient density estimation
    print("\n7. MEMORY-EFFICIENT DENSITY ESTIMATION")
    print("="*80)

    density_results = perform_memory_efficient_density_analysis(trajectory)

    # 8. Memory-aware visualization
    print("\n8. MEMORY-AWARE VISUALIZATION")
    print("="*80)

    create_memory_optimized_visualizations(trajectory, density_results)

    # 9. Export results with memory monitoring
    print("\n9. MEMORY-MONITORED EXPORT")
    print("="*80)

    export_results_with_monitoring(trajectory, field)

    # 10. Generate comprehensive memory report
    print("\n10. MEMORY ANALYSIS REPORT")
    print("="*80)

    generate_memory_reports(field, trajectory, stats, strategy_info, density_results)

    # Final memory summary
    memory_tracker.print_memory_report()

    print("\n" + "="*80)
    print("🎉 MEMORY-OPTIMIZED WORKFLOW COMPLETED!")
    print("="*80)

    # Save detailed memory report
    memory_tracker.save_detailed_report()


@memory_tracked
def load_memory_optimized_field():
    """Load velocity field with memory optimization."""

    # Data patterns to try (prioritize local data with correct naming)
    data_patterns = [
        "/home/arhashemi/Workspace/welding/Cases/004_caseCoarse.gid/post/0eule/004_caseCoarse_*.pvtu",
        "/home/arhashemi/Workspace/welding/Cases/004_caseCoarse.gid/post/0eule/",
        "data/**/004_caseCoarse_*.pvtu",
        "data/**/caseCoarse_*.pvtu",
        "example_data/**/case*_*.vtu",
        "../example_data/**/case*_*.vtu"
    ]

    for pattern in data_patterns:
        try:
            print(f"🔍 Trying pattern: {pattern}")

            # First, estimate memory usage
            with track_operation_memory("memory_estimation"):
                estimate = estimate_memory_usage(pattern, num_timesteps=40)
                track_variable_memory("memory_estimate", estimate)

                if "error" not in estimate:
                    print(f"📊 Memory estimate for 40 timesteps:")
                    print(f"   Total memory: {estimate['total_memory_mb']:.1f} MB")
                    print(f"   Memory per timestep: {estimate['velocity_memory_per_timestep_mb']:.1f} MB")
                    print(f"   Recommended chunk size: {estimate['recommended_chunk_size']}")

            # Configure memory-optimized loading
            config = MemoryOptimizedConfig(
                essential_fields_only=True,
                max_memory_per_timestep_mb=100.0,  # 100MB limit per timestep
                enable_compression=True,
                chunk_size=10,  # Process 10 timesteps at a time
                dtype="float32",
                discard_boundary_layers=True,
                subsample_factor=None  # No spatial subsampling for now
            )

            # Load with memory optimization
            with track_operation_memory("optimized_data_loading"):
                field = load_optimized_dataset(
                    data_pattern=pattern,
                    max_memory_per_timestep_mb=100.0,
                    max_time_steps=40,  # Limit to 40 timesteps
                    dtype="float32"
                )

            print(f"✅ Loaded optimized VTK data: {field}")

            # Convert data to JAX arrays for GPU acceleration
            print("🔄 Converting field data to JAX arrays on GPU...")
            try:
                with track_operation_memory("convert_to_jax"):
                    import jax.numpy as jnp
                    import jax
                    field.data = jnp.array(field.data)  # Move to GPU
                    field.positions = jnp.array(field.positions)  # Move to GPU
                    field.times = jnp.array(field.times)  # Move to GPU

                    # Update the internal JAX device arrays
                    field._data_dev = jax.device_put(field.data)
                    field._times_dev = jax.device_put(field.times)
                    field._pos_dev = jax.device_put(field.positions)

                    # Check device
                    try:
                        device_set = field.data.devices()
                        device_info = str(device_set)
                    except:
                        device_info = "GPU"

                    print(f"   ✅ Field data now on GPU: {device_info}")
                    print(f"   💾 GPU memory estimate: {field.data.nbytes / 1024 / 1024:.1f} MB")
            except Exception as e:
                print(f"   ⚠️  JAX conversion failed: {e}")
                import traceback
                traceback.print_exc()
                raise  # Re-raise to trigger fallback

            track_variable_memory("velocity_field", field)
            return field

        except Exception as e:
            print(f"   ⚠️  Failed: {e}")
            track_memory(f"loading_failed: {pattern}")

    # Fallback: create memory-efficient synthetic field
    print("📝 Creating memory-efficient synthetic field...")
    with track_operation_memory("synthetic_field_creation"):
        field = create_memory_efficient_synthetic_field()
        track_variable_memory("synthetic_field", field)

    print(f"✅ Created synthetic field: {field}")
    return field


@memory_tracked
def create_memory_efficient_synthetic_field():
    """Create a memory-efficient synthetic time-dependent vortex field."""

    # Reduced grid size for memory efficiency
    x = np.linspace(-2, 2, 20, dtype=np.float32)  # Reduced from 30
    y = np.linspace(-2, 2, 20, dtype=np.float32)  # Reduced from 30
    z = np.linspace(-0.5, 0.5, 3, dtype=np.float32)  # Reduced from 5

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    positions = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    track_variable_memory("mesh_positions", positions)

    # Reduced time steps for memory efficiency
    times = np.linspace(0, 5, 15, dtype=np.float32)  # Reduced from 20
    track_variable_memory("time_array", times)

    # Generate time-dependent velocity data
    velocity_data = []

    with track_operation_memory("velocity_field_generation"):
        for i, t in enumerate(times):
            track_memory(f"generating_timestep_{i}")

            # Time-dependent vortex
            strength = 1.0 + 0.5 * np.sin(2 * np.pi * t / 5)
            center_x = 0.3 * np.cos(2 * np.pi * t / 5)
            center_y = 0.3 * np.sin(2 * np.pi * t / 5)

            dx = positions[:, 0] - center_x
            dy = positions[:, 1] - center_y
            r_squared = dx**2 + dy**2 + 1e-6

            vx = -strength * dy / r_squared
            vy = strength * dx / r_squared
            vz = np.zeros_like(vx)

            velocities = np.stack([vx, vy, vz], axis=1).astype(np.float32)
            velocity_data.append(velocities)

            track_variable_memory(f"velocity_timestep_{i}", velocities)

    velocity_data = np.array(velocity_data)
    track_variable_memory("complete_velocity_data", velocity_data)

    # Convert to JAX arrays for GPU acceleration
    import jax.numpy as jnp
    velocity_data = jnp.array(velocity_data)
    positions = jnp.array(positions)
    times = jnp.array(times)

    return TimeSeriesField(
        data=velocity_data,
        times=times,
        positions=positions,
        interpolation="linear",
        extrapolation="constant"
    )


@memory_tracked
def execute_memory_optimized_tracking(field, concentrations=None):
    """Execute particle tracking with memory optimization."""

    with track_operation_memory("particle_tracking_setup"):
        # Get field bounds
        bounds_min, bounds_max = field.get_spatial_bounds()
        track_variable_memory("field_bounds", [bounds_min, bounds_max])
        print(f"📏 Field bounds: {bounds_min} to {bounds_max}")

        # Memory-optimized particle generation
        print(f"🎯 Generating memory-optimized particle distribution...")

        if concentrations is None:
            concentrations = {'x': 15, 'y': 8, 'z': 3}  # Reduced for memory

        concentration_x = concentrations['x']
        concentration_y = concentrations['y']
        concentration_z = concentrations['z']

        domain_size = bounds_max - bounds_min
        print(f"   Domain size: X={domain_size[0]:.4f}, Y={domain_size[1]:.4f}, Z={domain_size[2]:.4f}")

        par_bounds = [bounds_min, bounds_max]

        nx = max(1, int(concentration_x))
        ny = max(1, int(concentration_y))
        nz = max(1, int(concentration_z))

        print(f"   Memory-optimized grid: {nx} x {ny} x {nz} = {nx*ny*nz} particles")
        print(f"   Memory saving: ~{(60*50*15)/(nx*ny*nz):.1f}x fewer particles than default")

        # Generate uniform grid
        seeds = uniform_grid_seeds(
            resolution=(nx, ny, nz),
            bounds=par_bounds,
            include_boundaries=True
        )
        track_variable_memory("particle_seeds", seeds)

        print(f"✅ Generated {len(seeds)} memory-optimized particles")

    # GPU-optimized tracking configuration
    # With 3GB GPU memory available, we can use larger batches to reduce overhead
    strategy_info = {
        'name': 'GPU-Optimized RK4 with Reflective Boundaries',
        'integrator': 'rk4',
        'n_timesteps': 1500,
        'batch_size': min(len(seeds), 10000),  # Larger batch size for GPU (was 500)
        'boundary_type': 'reflective',  # Changed from continuous_inlet for JIT compatibility
        'dt': 0.005
    }

    print(f"🎯 GPU Optimization Strategy:")
    print(f"   Batch size: {strategy_info['batch_size']} particles")
    print(f"   Timesteps: {strategy_info['n_timesteps']}")
    print(f"   Expected memory per batch: ~{strategy_info['batch_size'] * 3 * 4 / 1024 / 1024:.1f} MB")
    print(f"   💡 Larger batches reduce CPU↔GPU transfer overhead")

    track_variable_memory("strategy_info", strategy_info)

    # Create boundary condition
    # NOTE: Using simple reflective boundary for GPU/JIT compatibility
    # Complex boundaries like continuous_inlet use NumPy and can't be JIT compiled
    with track_operation_memory("boundary_setup"):
        from jaxtrace.tracking.boundary import reflective_boundary
        full_bounds = [bounds_min, bounds_max]
        boundary = reflective_boundary(full_bounds)

        print("🚪 Using reflective boundaries for GPU acceleration:")
        print(f"   ⚠️  Note: Switched from continuous_inlet to reflective for JIT compatibility")
        print(f"   💡 Reflective boundaries are JAX-compatible and enable GPU JIT compilation")

        # Old boundary (doesn't work with JIT):
        # boundary = continuous_inlet_boundary_factory(
        #     bounds=full_bounds,
        #     flow_axis='x',
        #     flow_direction='positive',
        #     inlet_distribution='grid',
        #     concentrations=concentrations
        # )
        track_variable_memory("boundary_condition", boundary)

    print("🚪 Memory-optimized boundary conditions configured")

    # Progress callback with memory monitoring
    def memory_aware_progress_callback(progress):
        """Progress callback with memory monitoring."""
        percent = progress * 100

        # Update every 10% and track memory
        if int(percent) % 10 == 0 and percent != getattr(memory_aware_progress_callback, '_last_percent', -1):
            # Track memory at progress checkpoints
            track_memory(f"tracking_progress_{int(percent)}%")

            # Get current memory usage
            if get_memory_tracker():
                summary = get_memory_tracker().get_memory_summary()
                gpu_mb = summary["current_usage"]["gpu_mb"]
                cpu_mb = summary["current_usage"]["cpu_mb"]

                # Create progress bar
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)

                print(f"\r   Progress: |{bar}| {percent:5.1f}% [GPU:{gpu_mb:.0f}MB CPU:{cpu_mb:.0f}MB]",
                      end='', flush=True)
                memory_aware_progress_callback._last_percent = int(percent)

        if progress >= 1.0:
            print()

    # Create memory-optimized tracker with explicit JIT options
    print("🚀 Setting up GPU-accelerated tracker...")
    print("   Explicitly enabling JAX JIT compilation...")
    with track_operation_memory("tracker_creation"):
        tracker = create_tracker(
            integrator_name='rk4',
            field=field,
            boundary_condition=boundary,
            batch_size=strategy_info['batch_size'],
            record_velocities=False,  # Disable velocity recording to save memory
            progress_callback=memory_aware_progress_callback,
            # Explicit JIT options
            use_jax_jit=True,         # Enable JAX JIT
            static_compilation=True,   # Compile step function
            use_scan_jit=True,         # Use lax.scan for full trajectory
            device_put_inputs=True     # Keep data on GPU
        )
        track_variable_memory("particle_tracker", tracker)

        # Check if JIT compilation succeeded
        if tracker._compiled_step is not None:
            print("   ✅ JIT compilation: Single step COMPILED")
        else:
            print("   ❌ JIT compilation: Single step FAILED (will be slow!)")

        if tracker._compiled_simulate is not None:
            print("   ✅ JIT compilation: Full scan COMPILED")
        else:
            print("   ❌ JIT compilation: Full scan FAILED (will use fallback)")

    # Run tracking with memory monitoring
    print("🏃 Running GPU-accelerated particle tracking...")
    print(f"   Tracking {len(seeds)} particles for {strategy_info['n_timesteps']} timesteps")
    print(f"\n   ℹ️  EXPECTED BEHAVIOR:")
    print(f"   • First batch: ~10-30 sec delay for JIT compilation (one-time cost)")
    print(f"   • Subsequent batches: Much faster, smooth progress")
    print(f"   • Progress updates every 10%")
    print(f"   • GPU keeps data in VRAM, minimizing CPU↔GPU transfers\n")

    start_time = time.time()

    with track_operation_memory("particle_tracking_execution"):
        time_span = (0.0, 3.0)  # Reduced simulation time
        trajectory = tracker.track_particles(
            initial_positions=seeds,
            time_span=time_span,
            n_timesteps=strategy_info['n_timesteps'],
            dt=strategy_info['dt']
        )
        track_variable_memory("trajectory_result", trajectory)

    tracking_time = time.time() - start_time
    print(f"✅ Memory-optimized tracking completed in {tracking_time:.2f} seconds")
    print(f"   Trajectory: {trajectory}")

    return trajectory, strategy_info


@memory_tracked
def perform_memory_efficient_density_analysis(trajectory):
    """Perform memory-efficient density estimation."""

    try:
        with track_operation_memory("density_analysis"):
            # Use only a subset of particles for density analysis
            final_positions = trajectory.positions[-1]

            # Subsample if too many particles
            max_particles_for_density = 5000
            if len(final_positions) > max_particles_for_density:
                indices = np.random.choice(len(final_positions), max_particles_for_density, replace=False)
                final_positions = final_positions[indices]
                print(f"📊 Subsampled to {len(final_positions)} particles for density analysis")

            track_variable_memory("density_analysis_positions", final_positions)
            print(f"📊 Analyzing density with {final_positions.shape[0]} particles...")

            density_results = {}

            # Memory-efficient KDE Analysis
            print("📈 Performing memory-efficient KDE analysis...")
            with track_operation_memory("kde_analysis"):
                kde_estimator = KDEEstimator(
                    positions=final_positions,
                    bandwidth_rule='scott'
                )
                track_variable_memory("kde_estimator", kde_estimator)

                # Reduced evaluation grid for memory efficiency
                x_range = np.linspace(-1.5, 1.5, 30)  # Reduced from 50
                y_range = np.linspace(-1.5, 1.5, 30)  # Reduced from 50
                X_eval, Y_eval = np.meshgrid(x_range, y_range)
                eval_points = np.column_stack([
                    X_eval.ravel(),
                    Y_eval.ravel(),
                    np.zeros(X_eval.size)
                ])
                track_variable_memory("kde_eval_points", eval_points)

                kde_density = kde_estimator.evaluate(eval_points)
                kde_density = kde_density.reshape(X_eval.shape)
                track_variable_memory("kde_density", kde_density)

                density_results['kde'] = {
                    'estimator': kde_estimator,
                    'density_2d': kde_density,
                    'grid_x': X_eval,
                    'grid_y': Y_eval,
                    'bandwidth': getattr(kde_estimator, 'bandwidth', 'auto')
                }

                print(f"   ✅ Memory-efficient KDE completed")

            # Memory-efficient SPH Analysis
            print("🔬 Performing memory-efficient SPH analysis...")
            try:
                with track_operation_memory("sph_analysis"):
                    if len(final_positions) < 2:
                        raise ValueError("At least 2 particles required for SPH analysis")

                    sph_estimator = SPHDensityEstimator(
                        positions=final_positions,
                        smoothing_length=0.15  # Slightly larger for memory efficiency
                    )
                    sph_density = sph_estimator.compute_density()

                    track_variable_memory("sph_estimator", sph_estimator)
                    track_variable_memory("sph_density", sph_density)

                    density_results['sph'] = {
                        'estimator': sph_estimator,
                        'densities': sph_density,
                        'smoothing_length': sph_estimator.smoothing_length
                    }
                    print(f"   ✅ Memory-efficient SPH completed")

            except Exception as e:
                print(f"   ⚠️  SPH analysis failed: {e}")

            return density_results

    except Exception as e:
        print(f"⚠️  Density analysis failed: {e}")
        return None


@memory_tracked
def create_memory_optimized_visualizations(trajectory, density_results=None):
    """Create memory-optimized visualizations."""

    output_dir = Path("output_memory_optimized")
    output_dir.mkdir(exist_ok=True)

    try:
        with track_operation_memory("visualization_creation"):
            # Subsample trajectory for visualization if too large
            max_particles_viz = 1000
            positions = trajectory.positions[-1]

            if len(positions) > max_particles_viz:
                indices = np.random.choice(len(positions), max_particles_viz, replace=False)
                viz_positions = positions[indices]
                print(f"📊 Subsampled to {len(viz_positions)} particles for visualization")
            else:
                viz_positions = positions

            track_variable_memory("visualization_positions", viz_positions)

            # Memory-efficient particle plot
            print("📊 Creating memory-optimized particle visualization...")
            with track_operation_memory("particle_plot"):
                _, ax = plt.subplots(figsize=(8, 6))  # Smaller figure

                plot_particles_2d(
                    positions=viz_positions,
                    ax=ax,
                    title="Final Particle Positions (Memory Optimized)"
                )

                plt.savefig(output_dir / "particles_final_optimized.png",
                           dpi=100, bbox_inches='tight')  # Lower DPI
                plt.close()
                print("   ✅ Saved particles_final_optimized.png")

            # Memory-efficient trajectory plot (limited particles)
            print("📈 Creating memory-optimized trajectory visualization...")
            with track_operation_memory("trajectory_plot"):
                _, ax = plt.subplots(figsize=(10, 6))

                # Use only every 5th timestep and limited particles
                traj_positions = trajectory.positions[::5, :min(20, len(positions))]
                track_variable_memory("trajectory_viz_data", traj_positions)

                plot_trajectory_2d(
                    positions_over_time=traj_positions,
                    ax=ax,
                    max_particles=20,
                    title="Particle Trajectories (Memory Optimized)",
                    alpha=0.7
                )

                plt.savefig(output_dir / "trajectories_2d_optimized.png",
                           dpi=100, bbox_inches='tight')
                plt.close()
                print("   ✅ Saved trajectories_2d_optimized.png")

            print(f"✅ Memory-optimized visualizations saved to {output_dir}/")

    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")


@memory_tracked
def export_results_with_monitoring(trajectory, field=None):
    """Export results with memory monitoring."""

    output_dir = Path("output_memory_optimized")
    output_dir.mkdir(exist_ok=True)

    try:
        with track_operation_memory("vtk_export"):
            # Export only final trajectory state to save memory
            print("💾 Exporting memory-optimized trajectory to VTK...")
            trajectory_file = output_dir / "trajectory_optimized.vtp"

            export_trajectory_to_vtk(
                trajectory=trajectory,
                filename=str(trajectory_file),
                include_velocities=False,  # Skip velocities to save memory
                include_metadata=True,
                time_series=False  # Single file only
            )
            print(f"   ✅ Exported: {trajectory_file}")

    except Exception as e:
        print(f"⚠️  VTK export failed: {e}")


@memory_tracked
def generate_memory_reports(field, trajectory, stats, strategy_info, density_results=None):
    """Generate reports with memory analysis."""

    output_dir = Path("output_memory_optimized")
    output_dir.mkdir(exist_ok=True)

    try:
        with track_operation_memory("report_generation"):
            # Generate standard summary report
            print("📋 Generating memory-optimized summary report...")
            report_file = generate_summary_report(
                field=field,
                trajectory=trajectory,
                stats=stats,
                strategy=strategy_info,
                output_dir=output_dir
            )
            print(f"   ✅ Generated: {report_file}")

            # Generate memory analysis report
            print("📋 Generating memory analysis report...")
            memory_tracker = get_memory_tracker()
            if memory_tracker:
                memory_report_file = output_dir / "memory_analysis_report.json"
                memory_tracker.save_detailed_report(str(memory_report_file))
                print(f"   ✅ Generated: {memory_report_file}")

    except Exception as e:
        print(f"⚠️  Report generation failed: {e}")


if __name__ == "__main__":
    main()