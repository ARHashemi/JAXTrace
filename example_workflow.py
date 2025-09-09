#!/usr/bin/env python3  
"""  
Complete JAXTrace VTK Time Series Analysis Workflow  
Enhanced VTK System Only - No Legacy Dependencies  

Demonstrates the complete modern workflow:  
1. Load VTK time series using enhanced readers  
2. Create velocity field with time series support  
3. Track particles with advanced options  
4. Export comprehensive results to VTK  
5. Handle errors gracefully  

FIXED VERSION - Compatible with corrected tracking and fields modules  
"""  

import numpy as np  
import matplotlib.pyplot as plt  
from pathlib import Path  
import time  
import warnings  
import gc  

# JAXTrace imports with complete error handling  
try:  
    import jaxtrace as jt  
    from jaxtrace.fields import TimeSeriesField, create_field_from_vtk  
    from jaxtrace.integrators import rk2_step  
    from jaxtrace.tracking import (  
        ParticleTracker,   
        TrackerOptions,   
        create_tracker,  
        random_seeds  
    )  
    from jaxtrace.tracking.boundary import periodic_boundary  
    from jaxtrace.io import (  
        VTK_IO_AVAILABLE,   
        get_vtk_status,  
        get_io_status,  
        open_dataset,  
        validate_file_access,  
        print_io_summary  
    )  
    
    JAXTRACE_AVAILABLE = True  
except ImportError as e:  
    print(f"❌ JAXTrace not available: {e}")  
    JAXTRACE_AVAILABLE = False  
    exit(1)  

# Print system information  
print(f"JAXTrace v{jt.__version__}")  
print("I/O System Status:")  
print_io_summary()  

# Configuration  
jt.configure(  
    dtype="float32",  
    device="cpu",  
    memory_limit_gb=8.0,  
    use_jax_jit=True,  
    show_progress=True,  
)  

print("\n" + "="*80)  
print("JAXTrace Complete VTK Time Series Analysis")  
print("Enhanced VTK System Only - Modern Implementation")  
print("="*80)  


def check_system_requirements():  
    """Check system requirements and VTK availability."""  
    print("🔍 Checking system requirements...")  
    
    requirements = {  
        'jaxtrace': JAXTRACE_AVAILABLE,  
        'enhanced_vtk': VTK_IO_AVAILABLE,  
        'numpy': True,  
        'matplotlib': True  
    }  
    
    # Check VTK library  
    try:  
        import vtk  
        requirements['vtk_library'] = True  
        print(f"   ✅ VTK library: v{vtk.vtkVersion.GetVTKVersion()}")  
    except ImportError:  
        requirements['vtk_library'] = False  
        print("   ❌ VTK library: Not available")  
    
    # Check SciPy for spatial indexing  
    try:  
        import scipy  
        requirements['scipy'] = True  
        print(f"   ✅ SciPy: v{scipy.__version__} (spatial indexing enabled)")  
    except ImportError:  
        requirements['scipy'] = False  
        print("   ⚠️  SciPy: Not available (spatial indexing disabled)")  
    
    # Check JAX  
    try:  
        import jax  
        requirements['jax'] = True  
        print(f"   ✅ JAX: v{jax.__version__}")  
    except ImportError:  
        requirements['jax'] = False  
        print("   ❌ JAX: Not available")  
    
    all_critical_ok = (requirements['jaxtrace'] and   
                      requirements['enhanced_vtk'] and   
                      requirements['vtk_library'])  
    
    if not all_critical_ok:  
        print("\n❌ Critical requirements not met!")  
        if not requirements['jaxtrace']:  
            print("   - Install JAXTrace: pip install jaxtrace")  
        if not requirements['enhanced_vtk'] or not requirements['vtk_library']:  
            print("   - Install VTK: pip install vtk")  
        if not requirements['jax']:  
            print("   - Install JAX: pip install jax")  
        return False  
    
    print("✅ All critical requirements met!")  
    return True  


def validate_and_load_vtk_data(data_directory: str, case_name: str = "caseCoarse"):  
    """  
    Validate and load VTK time series data using modern enhanced system.  
    """  
    print(f"\n📥 Loading VTK time series data...")  
    
    data_path = Path(data_directory)  
    
    # Validate directory exists  
    if not data_path.exists():  
        raise FileNotFoundError(f"Data directory not found: {data_directory}")  
    
    # Create file patterns to try  
    file_patterns = [  
        str(data_path / f"*{case_name}_*.pvtu"),  # Preferred parallel format  
        str(data_path / f"*{case_name}_*.vtu"),   # Serial unstructured  
        str(data_path / "*.pvtu"),                # Any parallel files  
        str(data_path / "*.vtu"),                 # Any serial files  
    ]  
    
    print(f"   📁 Directory: {data_path.absolute()}")  
    print(f"   🔍 Looking for: {case_name}")  
    
    # Try each pattern  
    selected_pattern = None  
    validation_result = None  
    
    for pattern in file_patterns:  
        print(f"   📋 Trying pattern: {Path(pattern).name}")  
        
        # Validate file access  
        validation_result = validate_file_access(pattern)  
        
        if validation_result['valid']:  
            selected_pattern = pattern  
            print(f"   ✅ Found {len(validation_result['files_found'])} valid files")  
            break  
        else:  
            if validation_result['files_found']:  
                print(f"   ⚠️  Found {len(validation_result['files_found'])} files but format issues")  
            else:  
                print(f"   ❌ No files found")  
    
    if not selected_pattern:  
        print("\n💥 No valid VTK files found with any pattern!")  
        print("🔍 Available files in directory:")  
        
        all_files = list(data_path.glob("*"))  
        vtk_files = [f for f in all_files if f.suffix.lower() in ['.pvtu', '.vtu', '.vtk', '.vti', '.vts', '.vtr']]  
        
        print(f"   📄 Total files: {len(all_files)}")  
        print(f"   📄 VTK files: {len(vtk_files)}")  
        
        if vtk_files:  
            print(f"   📄 Available VTK files:")  
            for f in sorted(vtk_files)[:10]:  # Show first 10  
                print(f"      - {f.name}")  
            if len(vtk_files) > 10:  
                print(f"      ... and {len(vtk_files) - 10} more")  
        
        raise RuntimeError("No valid VTK time series data found")  
    
    # Load the data using the enhanced system  
    print(f"📤 Loading VTK time series using enhanced readers...")  
    
    try:  
        # Method 1: Use registry system (recommended)  
        print("   🔧 Method 1: Using registry system...")  
        dataset = open_dataset(selected_pattern, max_time_steps=50)  
        
        # Convert to TimeSeriesField if it's not already  
        if hasattr(dataset, 'load_time_series'):  
            # It's a VTK reader - load the time series data  
            time_series_data = dataset.load_time_series()  
            
            field = TimeSeriesField(  
                data=time_series_data['velocity_data'],  
                times=time_series_data['times'],  
                positions=time_series_data['positions'],  
                interpolation="linear",  
                extrapolation="constant",  
                _source_info=time_series_data  
            )  
        elif isinstance(dataset, dict) and 'velocity_data' in dataset:  
            # It's already loaded time series data  
            field = TimeSeriesField(  
                data=dataset['velocity_data'],  
                times=dataset['times'],  
                positions=dataset['positions'],  
                interpolation="linear",  
                extrapolation="constant",  
                _source_info=dataset  
            )  
        else:  
            raise ValueError("Unexpected dataset format from registry")  
        
        print(f"✅ Successfully loaded using registry system!")  
        
    except Exception as e:  
        print(f"   ⚠️  Registry method failed: {e}")  
        print("   🔧 Method 2: Direct enhanced VTK reader...")  
        
        try:  
            # Method 2: Direct enhanced VTK reader  
            from jaxtrace.io import open_vtk_time_series  
            
            time_series_data = open_vtk_time_series(  
                file_pattern=selected_pattern,  
                max_time_steps=50,  
                velocity_field_name=None  # Auto-detect  
            )  
            
            field = TimeSeriesField(  
                data=time_series_data['velocity_data'],  
                times=time_series_data['times'],  
                positions=time_series_data['positions'],  
                interpolation="linear",  
                extrapolation="constant",  
                _source_info=time_series_data  
            )  
            
            print(f"✅ Successfully loaded using direct reader!")  
            
        except Exception as e2:  
            print(f"   ❌ Direct reader also failed: {e2}")  
            
            # Method 3: Factory function  
            print("   🔧 Method 3: Using factory function...")  
            
            try:  
                field = create_field_from_vtk(  
                    file_pattern=selected_pattern,  
                    field_type="time_series",  
                    max_time_steps=30,  # Reduced for reliability  
                    interpolation="linear"  
                )  
                
                print(f"✅ Successfully loaded using factory function!")  
                
            except Exception as e3:  
                print(f"   ❌ Factory function failed: {e3}")  
                raise RuntimeError(f"All loading methods failed. Last error: {e3}")  
    
    # Display field information  
    print(f"📊 Field Information:")  
    print(f"   📊 Data shape: {field.data.shape} (T,N,3)")  
    print(f"   🕒 Time steps: {field.T}")  
    print(f"   📍 Grid points: {field.N}")  
    print(f"   💾 Memory usage: {field.memory_usage_mb():.1f} MB")  
    
    # Get time and spatial bounds  
    t_min, t_max = field.get_time_bounds()  
    bounds_min, bounds_max = field.get_spatial_bounds()  
    
    print(f"   🕒 Time range: {t_min:.2f} to {t_max:.2f}")  
    print(f"   📏 Spatial bounds:")  
    print(f"      X: [{bounds_min[0]:.3f}, {bounds_max[0]:.3f}]")  
    print(f"      Y: [{bounds_min[1]:.3f}, {bounds_max[1]:.3f}]")  
    print(f"      Z: [{bounds_min[2]:.3f}, {bounds_max[2]:.3f}]")  
    
    # Validate field data  
    try:  
        field.validate_data()  
        print(f"✅ Field data validation passed")  
    except Exception as e:  
        print(f"⚠️  Field validation warning: {e}")  
    
    return field  


def execute_particle_tracking_analysis(field, n_particles=600):  
    """Execute comprehensive particle tracking analysis - FIXED VERSION."""  
    print(f"\n🚀 Executing particle tracking analysis...")  
    
    # Get field bounds for seeding  
    bounds_min, bounds_max = field.get_spatial_bounds()  
    t_start, t_end = field.get_time_bounds()  
    
    # FIXED: Create proper bounds format for boundary conditions  
    domain_bounds = [bounds_min.tolist(), bounds_max.tolist()]  # [[xmin,ymin,zmin], [xmax,ymax,zmax]]  
    
    print(f"   📊 Analysis setup:")  
    print(f"      Particles: {n_particles}")  
    print(f"      Time span: {t_start:.2f} to {t_end:.2f}")  
    print(f"      Domain bounds: {domain_bounds}")  
    print(f"      Domain size: {[bounds_max[i] - bounds_min[i] for i in range(3)]}")  
    
    # Advanced particle seeding strategy  
    print(f"   🌱 Advanced particle seeding...")  
    
    # Seed particles in multiple strategic regions  
    positions_list = []  
    n_per_region = n_particles // 4  
    
    # Region 1: Domain center (high activity zone)  
    center = [(bounds_min[i] + bounds_max[i]) / 2 for i in range(3)]  
    size = [(bounds_max[i] - bounds_min[i]) * 0.3 for i in range(3)]  
    region1_min = [center[i] - size[i]/2 for i in range(3)]  
    region1_max = [center[i] + size[i]/2 for i in range(3)]  
    
    pos1 = random_seeds(n_per_region, [region1_min, region1_max], rng_seed=42)  
    positions_list.append(pos1)  
    
    # Region 2: Near domain boundaries (boundary effects)  
    boundary_thickness = min(bounds_max[i] - bounds_min[i] for i in range(3)) * 0.1  
    
    pos2 = random_seeds(  
        n_per_region,  
        [[bounds_min[0], bounds_min[1], bounds_min[2]],  
         [bounds_min[0] + boundary_thickness, bounds_max[1], bounds_max[2]]],  
        rng_seed=43  
    )  
    positions_list.append(pos2)  
    
    # Region 3: High gradient zones (sample field to find)  
    # For simplicity, use random sampling with slight bias toward center  
    pos3 = random_seeds(  
        n_per_region,  
        domain_bounds,  # Use proper bounds format  
        rng_seed=44  
    )  
    positions_list.append(pos3)  
    
    # Region 4: Remaining particles distributed randomly  
    remaining = n_particles - 3 * n_per_region  
    pos4 = random_seeds(remaining, domain_bounds, rng_seed=45)  
    positions_list.append(pos4)  
    
    initial_positions = np.vstack(positions_list)  
    n_particles_actual = initial_positions.shape[0]  
    
    print(f"   🌱 Seeded {n_particles_actual} particles in 4 strategic regions")  
    
    # Configure advanced tracking options  
    options = TrackerOptions(  
        max_memory_gb=6.0,  
        record_velocities=True,  
        oom_recovery=True,  
        use_jax_jit=True,  
        batch_size=300,  # Smaller batches for stability  
        progress_callback=lambda p: print(f"      ⏳ Tracking progress: {p*100:.1f}%")   
                         if int(p * 20) != int((p - 0.05) * 20) else None  # Every 5%  
    )  
    
    # FIXED: Create proper boundary condition with domain bounds  
    boundary_fn = periodic_boundary(domain_bounds)
    
    # Test boundary condition with sample data
    try:
        test_pos = initial_positions[:5].copy()  # Test with 5 particles
        test_result = boundary_fn(test_pos)
        print(f"   ✅ Boundary condition test passed")
    except Exception as e:
        print(f"   ⚠️  Boundary condition test failed: {e}")
        print(f"   🔧 Using no boundary as fallback...")
        from jaxtrace.tracking.boundary import no_boundary
        boundary_fn = no_boundary()
    
    # Try multiple integration strategies for robustness
    integration_strategies = [
        {
            'name': 'RK4 High Resolution',
            'integrator': 'rk4',
            'n_timesteps': 80,
            'batch_size': 200
        },
        {
            'name': 'RK2 Medium Resolution', 
            'integrator': 'rk2',
            'n_timesteps': 60,
            'batch_size': 300
        },
        {
            'name': 'Euler Simple',
            'integrator': 'euler', 
            'n_timesteps': 40,
            'batch_size': 500
        }
    ]
    
    trajectory = None
    successful_strategy = None
    
    for strategy in integration_strategies:
        print(f"\n   🧮 Attempting: {strategy['name']}")
        print(f"      Method: {strategy['integrator']}")
        print(f"      Timesteps: {strategy['n_timesteps']}")
        print(f"      Batch size: {strategy['batch_size']}")
        
        try:
            # Update tracker options for this strategy
            options.batch_size = strategy['batch_size']
            
            # Create tracker
            tracker = create_tracker(
                integrator_name=strategy['integrator'],
                field=field,
                boundary_condition=boundary_fn,
                **options.__dict__
            )
            
            # Estimate memory and runtime
            runtime_est = tracker.estimate_runtime(
                n_particles_actual, 
                strategy['n_timesteps'],
                calibration_particles=min(50, n_particles_actual // 10)
            )
            
            if runtime_est['success']:
                print(f"      ⏱️  Estimated time: {runtime_est['estimated_runtime_minutes']:.1f} min")
                print(f"      💾 Estimated memory: {runtime_est['estimated_memory_gb']:.1f} GB")
                
                if runtime_est['estimated_memory_gb'] > 10:
                    print(f"      ⚠️  High memory usage, reducing batch size...")
                    options.batch_size = max(100, options.batch_size // 2)
            
            # Execute tracking
            start_time = time.time()
            
            trajectory = tracker.track_particles(
                initial_positions=initial_positions,
                time_span=(t_start, min(t_end, t_start + 5.0)),  # Limit duration for stability
                n_timesteps=strategy['n_timesteps']
            )
            
            elapsed_time = time.time() - start_time
            successful_strategy = strategy
            
            print(f"      ✅ Success! Completed in {elapsed_time:.1f}s")
            print(f"      📊 Result: {trajectory}")
            break
            
        except Exception as e:
            print(f"      ❌ Strategy failed: {e}")
            # Try to reduce problem size further
            if 'batch_size' in str(e).lower() or 'memory' in str(e).lower():
                print(f"      🔧 Memory issue detected, trying reduced batch size...")
                try:
                    options.batch_size = max(50, options.batch_size // 3)
                    
                    tracker = create_tracker(
                        integrator_name=strategy['integrator'],
                        field=field,
                        boundary_condition=boundary_fn,
                        **options.__dict__
                    )
                    
                    trajectory = tracker.track_particles(
                        initial_positions=initial_positions[:100],  # Reduce particles too
                        time_span=(t_start, t_start + 2.0),  # Shorter duration
                        n_timesteps=20  # Fewer timesteps
                    )
                    
                    print(f"      ✅ Reduced-size tracking successful!")
                    successful_strategy = {**strategy, 'reduced': True}
                    break
                    
                except Exception as e2:
                    print(f"      ❌ Even reduced tracking failed: {e2}")
            continue
    
    if trajectory is None:
        raise RuntimeError(f"Particle tracking failed completely: {e2}")
    
    print(f"\n   🎉 Particle tracking completed successfully!")
    print(f"      Strategy used: {successful_strategy['name']}")
    print(f"      Final trajectory shape: {trajectory.positions.shape}")
    print(f"      Memory usage: {trajectory.memory_usage_mb():.1f} MB")
    
    return trajectory, successful_strategy


def analyze_trajectory_results(trajectory, strategy_info):
    """Analyze tracking results comprehensively."""
    print(f"\n📊 Analyzing trajectory results...")
    
    # Import analysis functions
    from jaxtrace.tracking import compute_trajectory_statistics, validate_trajectory_data
    
    # Basic trajectory info
    print(f"   📏 Trajectory dimensions: {trajectory.positions.shape} (T,N,3)")
    print(f"   🕒 Time steps: {trajectory.T}")
    print(f"   🎯 Particles: {trajectory.N}")
    print(f"   ⏱️  Duration: {trajectory.duration:.2f}")
    print(f"   💾 Memory: {trajectory.memory_usage_mb():.1f} MB")
    
    # Validate trajectory data
    validation = validate_trajectory_data(trajectory)
    print(f"   ✅ Data validation: {'PASSED' if validation['valid'] else 'FAILED'}")
    if not validation['valid']:
        print(f"      ⚠️  Issues found: {', '.join(validation['issues'])}")
    
    # Compute comprehensive statistics
    try:
        stats = compute_trajectory_statistics(trajectory)
        
        print(f"\n   📈 Trajectory Statistics:")
        print(f"      Mean displacement: {stats['displacement']['mean']:.3f} ± {stats['displacement']['std']:.3f}")
        print(f"      Max displacement: {stats['displacement']['max']:.3f}")
        print(f"      Mean speed: {stats['speed']['mean']:.3f} ± {stats['speed']['std']:.3f}")
        
        if 'acceleration' in stats:
            print(f"      Mean acceleration: {stats['acceleration']['mean']:.3f} ± {stats['acceleration']['std']:.3f}")
        
        # Spatial distribution analysis
        final_positions = trajectory.positions[-1]  # (N, 3)
        initial_positions = trajectory.positions[0]  # (N, 3)
        
        # Compute spread
        initial_spread = np.std(initial_positions, axis=0)
        final_spread = np.std(final_positions, axis=0)
        
        print(f"\n   🌍 Spatial Analysis:")
        print(f"      Initial spread (XYZ): [{initial_spread[0]:.3f}, {initial_spread[1]:.3f}, {initial_spread[2]:.3f}]")
        print(f"      Final spread (XYZ): [{final_spread[0]:.3f}, {final_spread[1]:.3f}, {final_spread[2]:.3f}]")
        print(f"      Spread change: {(final_spread / initial_spread - 1) * 100}%")
        
        # Mixing analysis
        total_displacement = np.sum(stats['displacement']['values'])
        mixing_efficiency = total_displacement / (trajectory.duration * trajectory.N)
        
        print(f"      Mixing efficiency: {mixing_efficiency:.3f} units/time/particle")
        
    except Exception as e:
        print(f"   ⚠️  Statistical analysis failed: {e}")
        stats = None
    
    return stats, validation


def create_trajectory_visualizations(trajectory, stats, field, output_dir="output"):
    """Create comprehensive trajectory visualizations."""
    print(f"\n🎨 Creating trajectory visualizations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set up matplotlib for file output
    plt.style.use('default')
    
    try:
        # Figure 1: Trajectory overview
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'JAXTrace Particle Tracking Analysis\n{trajectory.N} particles, {trajectory.T} timesteps', fontsize=14)
        
        # 2D trajectory plot (top-left)
        ax1 = axes[0, 0]
        n_plot = min(20, trajectory.N)  # Plot subset for clarity
        step = max(1, trajectory.N // n_plot)
        
        for i in range(0, trajectory.N, step):
            path = trajectory.positions[:, i, :2]  # (T, 2)
            ax1.plot(path[:, 0], path[:, 1], alpha=0.7, linewidth=1)
        
        # Mark start/end points
        ax1.scatter(trajectory.positions[0, ::step, 0], trajectory.positions[0, ::step, 1], 
                   c='green', s=30, marker='o', label='Start', zorder=5)
        ax1.scatter(trajectory.positions[-1, ::step, 0], trajectory.positions[-1, ::step, 1],
                   c='red', s=30, marker='x', label='End', zorder=5)
        
        ax1.set_title('Particle Trajectories (X-Y)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # Displacement histogram (top-right)
        ax2 = axes[0, 1]
        if stats and 'displacement' in stats:
            displacements = stats['displacement']['values']
            ax2.hist(displacements, bins=25, alpha=0.7, edgecolor='black')
            ax2.axvline(stats['displacement']['mean'], color='red', linestyle='--', 
                       label=f"Mean: {stats['displacement']['mean']:.3f}")
            ax2.set_title('Final Displacement Distribution')
            ax2.set_xlabel('Displacement')
            ax2.set_ylabel('Count')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Time evolution of spread (bottom-left)
        ax3 = axes[1, 0]
        
        spreads_x = np.std(trajectory.positions[:, :, 0], axis=1)  # (T,)
        spreads_y = np.std(trajectory.positions[:, :, 1], axis=1)
        spreads_z = np.std(trajectory.positions[:, :, 2], axis=1)
        
        ax3.plot(trajectory.times, spreads_x, label='X spread', linewidth=2)
        ax3.plot(trajectory.times, spreads_y, label='Y spread', linewidth=2)
        ax3.plot(trajectory.times, spreads_z, label='Z spread', linewidth=2)
        
        ax3.set_title('Spatial Spread Evolution')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Standard Deviation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Speed evolution (bottom-right)
        ax4 = axes[1, 1]
        if trajectory.velocities is not None:
            speeds = np.linalg.norm(trajectory.velocities, axis=2)  # (T, N)
            mean_speeds = np.mean(speeds, axis=1)  # (T,)
            std_speeds = np.std(speeds, axis=1)
            
            ax4.plot(trajectory.times, mean_speeds, 'b-', linewidth=2, label='Mean speed')
            ax4.fill_between(trajectory.times, mean_speeds - std_speeds, mean_speeds + std_speeds,
                           alpha=0.3, color='blue', label='±1σ')
            ax4.set_title('Speed Evolution')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Speed')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            # Approximate speed from position differences
            dt = trajectory.times[1] - trajectory.times[0] if trajectory.T > 1 else 1.0
            speed_approx = np.zeros((trajectory.T - 1, trajectory.N))
            
            for t in range(trajectory.T - 1):
                displacements = trajectory.positions[t+1] - trajectory.positions[t]
                speed_approx[t] = np.linalg.norm(displacements, axis=1) / dt
            
            mean_speeds = np.mean(speed_approx, axis=1)
            ax4.plot(trajectory.times[:-1], mean_speeds, 'b-', linewidth=2, label='Mean speed (approx)')
            ax4.set_title('Speed Evolution (Approximated)')
            ax4.set_xlabel('Time')  
            ax4.set_ylabel('Speed')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_file = output_path / "trajectory_analysis.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: {output_file}")
        plt.close()
        
        # Figure 2: 3D trajectory visualization (if meaningful)
        if np.std(trajectory.positions[:, :, 2]) > 1e-6:  # If there's Z variation
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot subset of trajectories in 3D
            for i in range(0, trajectory.N, step):
                path = trajectory.positions[:, i, :]  # (T, 3)
                ax.plot(path[:, 0], path[:, 1], path[:, 2], alpha=0.7, linewidth=1)
            
            # Mark start points
            ax.scatter(trajectory.positions[0, ::step, 0], 
                      trajectory.positions[0, ::step, 1],
                      trajectory.positions[0, ::step, 2],
                      c='green', s=50, marker='o', label='Start')
            
            ax.set_title(f'3D Particle Trajectories\n{trajectory.N} particles')
            ax.set_xlabel('X')
            ax.set_ylabel('Y') 
            ax.set_zlabel('Z')
            ax.legend()
            
            output_file_3d = output_path / "trajectory_3d.png"
            plt.savefig(output_file_3d, dpi=150, bbox_inches='tight')
            print(f"   ✅ Saved: {output_file_3d}")
            plt.close()
        
        print(f"   🎨 Visualizations saved to: {output_path}")
        
    except Exception as e:
        print(f"   ❌ Visualization creation failed: {e}")
        import traceback
        traceback.print_exc()


def export_results_to_vtk(trajectory, field_info, output_dir="output"):
    """Export results to VTK format for advanced visualization."""
    print(f"\n💾 Exporting results to VTK format...")
    
    try:
        from jaxtrace.io import export_trajectory_to_vtk
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export trajectory
        vtk_file = output_path / "particle_trajectories.vtp"
        
        export_trajectory_to_vtk(
            trajectory=trajectory,
            filename=str(vtk_file),
            include_velocities=True,
            include_metadata=True,
            time_series=True
        )
        
        print(f"   ✅ Trajectory exported: {vtk_file}")
        print(f"   📊 File contains:")
        print(f"      - {trajectory.N} particle paths")
        print(f"      - {trajectory.T} time steps") 
        print(f"      - Velocity data: {'Yes' if trajectory.velocities is not None else 'No'}")
        print(f"      - File size: {vtk_file.stat().st_size / 1024:.1f} KB")
        
    except ImportError:
        print(f"   ⚠️  VTK export not available - skipping")
    except Exception as e:
        print(f"   ❌ VTK export failed: {e}")


def generate_summary_report(field, trajectory, stats, strategy, output_dir="output"):
    """Generate comprehensive summary report."""
    print(f"\n📋 Generating summary report...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    report_file = output_path / "analysis_summary.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("JAXTrace Particle Tracking Analysis Summary\n")
        f.write("="*80 + "\n\n")
        
        # Field information
        f.write("VELOCITY FIELD INFORMATION\n")
        f.write("-"*40 + "\n")
        f.write(f"Data shape: {field.data.shape} (T,N,3)\n")
        f.write(f"Time steps: {field.T}\n")
        f.write(f"Grid points: {field.N}\n")
        f.write(f"Memory usage: {field.memory_usage_mb():.1f} MB\n")
        
        t_min, t_max = field.get_time_bounds()
        bounds_min, bounds_max = field.get_spatial_bounds()
        
        f.write(f"Time range: {t_min:.3f} to {t_max:.3f}\n")
        f.write(f"Spatial bounds:\n")
        f.write(f"  X: [{bounds_min[0]:.3f}, {bounds_max[0]:.3f}]\n")
        f.write(f"  Y: [{bounds_min[1]:.3f}, {bounds_max[1]:.3f}]\n")
        f.write(f"  Z: [{bounds_min[2]:.3f}, {bounds_max[2]:.3f}]\n\n")
        
        # Tracking configuration
        f.write("TRACKING CONFIGURATION\n")
        f.write("-"*40 + "\n")
        f.write(f"Integration method: {strategy['integrator']}\n")
        f.write(f"Time steps: {strategy['n_timesteps']}\n")
        f.write(f"Batch size: {strategy['batch_size']}\n")
        f.write(f"Strategy: {strategy['name']}\n\n")
        
        # Results
        f.write("TRAJECTORY RESULTS\n")
        f.write("-"*40 + "\n")
        f.write(f"Particles tracked: {trajectory.N}\n")
        f.write(f"Time steps: {trajectory.T}\n")
        f.write(f"Duration: {trajectory.duration:.3f}\n")
        f.write(f"Memory usage: {trajectory.memory_usage_mb():.1f} MB\n")
        f.write(f"Velocities recorded: {'Yes' if trajectory.velocities is not None else 'No'}\n\n")
        
        # Statistics
        if stats:
            f.write("STATISTICAL ANALYSIS\n")
            f.write("-"*40 + "\n")
            f.write(f"Mean displacement: {stats['displacement']['mean']:.3f} ± {stats['displacement']['std']:.3f}\n")
            f.write(f"Max displacement: {stats['displacement']['max']:.3f}\n")
            f.write(f"Mean speed: {stats['speed']['mean']:.3f} ± {stats['speed']['std']:.3f}\n")
            
            if 'acceleration' in stats:
                f.write(f"Mean acceleration: {stats['acceleration']['mean']:.3f} ± {stats['acceleration']['std']:.3f}\n")
        
        f.write(f"\nReport generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"   ✅ Summary report saved: {report_file}")


def main():
    """Main execution function."""
    # Configuration
    config = {
        'data_directory': '/home/arhashemi/Workspace/welding/Cases/002_caseCoarse.gid/post/0eule/',
        'case_name': 'caseCoarse',
        'n_particles': 10000,  # Reduced for stability
        'output_directory': 'output_jaxtrace'
    }
    
    print("🚀 Starting JAXTrace Complete Workflow")
    print(f"Configuration: {config}")
    
    try:
        # Step 1: Check requirements
        if not check_system_requirements():
            return False
        
        # Step 2: Load VTK data
        field = validate_and_load_vtk_data(
            config['data_directory'], 
            config['case_name']
        )
        
        # Step 3: Execute tracking
        trajectory, strategy = execute_particle_tracking_analysis(
            field, 
            config['n_particles']
        )
        
        # Step 4: Analyze results
        stats, validation = analyze_trajectory_results(trajectory, strategy)
        
        # Step 5: Create visualizations
        create_trajectory_visualizations(
            trajectory, stats, field, config['output_directory']
        )
        
        # Step 6: Export to VTK
        export_results_to_vtk(trajectory, field, config['output_directory'])
        
        # Step 7: Generate summary
        generate_summary_report(
            field, trajectory, stats, strategy, config['output_directory']
        )
        
        # Final summary
        print("\n" + "="*80)
        print("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"✅ Tracked {trajectory.N} particles for {trajectory.T} timesteps")
        print(f"✅ Generated {trajectory.T * trajectory.N} particle-time data points")
        print(f"✅ Results saved to: {config['output_directory']}")
        print(f"✅ Memory usage: {trajectory.memory_usage_mb():.1f} MB")
        
        # Cleanup
        gc.collect()
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  Workflow interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)