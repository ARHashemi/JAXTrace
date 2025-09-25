#!/usr/bin/env python3  
"""  
Enhanced JAXTrace VTK Time Series Analysis Workflow  
Complete workflow with KDE/SPH density estimation, advanced visualization, and comprehensive I/O  

ENHANCED VERSION - Integrates density, visualization, and I/O modules  
Based on the original workflow with all functions preserved  
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
        random_seeds,  
        compute_trajectory_statistics,  
        validate_trajectory_data  
    )  
    from jaxtrace.tracking.boundary import periodic_boundary  
    
    # Enhanced I/O with comprehensive features  
    from jaxtrace.io import (  
        VTK_IO_AVAILABLE,  
        get_vtk_status,  
        get_io_status,  
        open_dataset,  
        validate_file_access,  
        print_io_summary,  
        export_trajectory_to_vtk  
    )  
    
    # Enhanced I/O writers (if available)  
    try:  
        from jaxtrace.io import VTKTrajectoryWriter, VTKFieldWriter  
        ENHANCED_WRITERS_AVAILABLE = True  
    except ImportError:  
        ENHANCED_WRITERS_AVAILABLE = False  
    
    # Density estimation modules  
    try:  
        from jaxtrace.density import (  
            KDEEstimator,  
            SPHDensityEstimator,  
            scott_bandwidth,  
            silverman_bandwidth  
        )  
        DENSITY_AVAILABLE = True  
    except ImportError:  
        DENSITY_AVAILABLE = False  
    
    # Enhanced visualization  
    try:  
        from jaxtrace.visualization import (  
            plot_particles_2d,  
            plot_particles_3d,   
            plot_trajectory_2d,  
            plot_trajectory_3d,  
            render_frames_2d,  
            render_frames_3d,  
            save_gif_from_frames  
        )  
        VISUALIZATION_AVAILABLE = True  
    except ImportError:  
        VISUALIZATION_AVAILABLE = False  
    
    # Try to import dynamic visualization (optional)  
    try:  
        from jaxtrace.visualization import (  
            animate_trajectory_plotly_2d,  
            animate_trajectory_plotly_3d  
        )  
        PLOTLY_AVAILABLE = True  
    except ImportError:  
        PLOTLY_AVAILABLE = False  
    
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
print("JAXTrace Enhanced Analysis with Density Estimation")  
print("Advanced I/O, Visualization, and Density Features")  
print("="*80)  


def check_system_requirements():  
    """Check system requirements including enhanced modules."""  
    print("🔍 Checking enhanced system requirements...")  
    
    requirements = {  
        'jaxtrace': JAXTRACE_AVAILABLE,  
        'enhanced_vtk': VTK_IO_AVAILABLE,  
        'numpy': True,  
        'matplotlib': True,  
        'density_estimation': DENSITY_AVAILABLE,  
        'visualization': VISUALIZATION_AVAILABLE,  
        'enhanced_writers': ENHANCED_WRITERS_AVAILABLE,  
        'plotly_dynamic': PLOTLY_AVAILABLE  
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
        print("   ⚠️  SciPy: Not available (may affect density estimation)")  
    
    # Check JAX  
    try:  
        import jax  
        requirements['jax'] = True  
        print(f"   ✅ JAX: v{jax.__version__}")  
    except ImportError:  
        requirements['jax'] = False  
        print("   ❌ JAX: Not available")  
    
    # Report enhanced features  
    print(f"   {'✅' if DENSITY_AVAILABLE else '❌'} Density estimation: {'Available' if DENSITY_AVAILABLE else 'Not available'}")  
    print(f"   {'✅' if VISUALIZATION_AVAILABLE else '❌'} Enhanced visualization: {'Available' if VISUALIZATION_AVAILABLE else 'Not available'}")  
    print(f"   {'✅' if ENHANCED_WRITERS_AVAILABLE else '❌'} Enhanced I/O writers: {'Available' if ENHANCED_WRITERS_AVAILABLE else 'Not available'}")  
    print(f"   {'✅' if PLOTLY_AVAILABLE else '⚠️'} Interactive plots: {'Available' if PLOTLY_AVAILABLE else 'Not available'}")  
    
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
        dataset = open_dataset(selected_pattern, max_time_steps=40)  
        
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
                max_time_steps=40,  
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


# def execute_particle_tracking_analysis(field, n_particles=600):  
#     """Execute comprehensive particle tracking analysis - FIXED VERSION."""  
#     print(f"\n🚀 Executing particle tracking analysis...")  
    
#     # Get field bounds for seeding  
#     bounds_min, bounds_max = field.get_spatial_bounds()  
#     t_start, t_end = field.get_time_bounds()  
    
#     # FIXED: Create proper bounds format for boundary conditions  
#     domain_bounds = [bounds_min.tolist(), bounds_max.tolist()]  # [[xmin,ymin,zmin], [xmax,ymax,zmax]]  
    
#     print(f"   📊 Analysis setup:")  
#     print(f"      Particles: {n_particles}")  
#     print(f"      Time span: {t_start:.2f} to {t_end:.2f}")  
#     print(f"      Domain bounds: {domain_bounds}")  
#     print(f"      Domain size: {[bounds_max[i] - bounds_min[i] for i in range(3)]}")  
    
#     # Advanced particle seeding strategy  
#     print(f"   🌱 Advanced particle seeding...")  
    
#     # Seed particles in multiple strategic regions  
#     positions_list = []  
#     n_per_region = n_particles // 4  
    
#     # Region 1: Domain center (high activity zone)  
#     center = [(bounds_min[i] + bounds_max[i]) / 2 for i in range(3)]  
#     size = [(bounds_max[i] - bounds_min[i]) * 0.3 for i in range(3)]  
#     region1_min = [center[i] - size[i]/2 for i in range(3)]  
#     region1_max = [center[i] + size[i]/2 for i in range(3)]
    
#     pos1 = random_seeds(n_per_region, [region1_min, region1_max], rng_seed=42)
#     positions_list.append(pos1)
    
#     # Region 2: Near domain boundaries (boundary effects)
#     boundary_thickness = min(bounds_max[i] - bounds_min[i] for i in range(3)) * 0.1
    
#     pos2 = random_seeds(
#         n_per_region,
#         [[bounds_min[0], bounds_min[1], bounds_min[2]],
#          [bounds_min[0] + boundary_thickness, bounds_max[1], bounds_max[2]]],
#         rng_seed=43
#     )
#     positions_list.append(pos2)
    
#     # Region 3: High gradient zones (sample field to find)
#     # For simplicity, use random sampling with slight bias toward center
#     pos3 = random_seeds(
#         n_per_region,
#         domain_bounds,  # Use proper bounds format
#         rng_seed=44
#     )
#     positions_list.append(pos3)
    
#     # Region 4: Remaining particles distributed randomly
#     remaining = n_particles - 3 * n_per_region
#     pos4 = random_seeds(remaining, domain_bounds, rng_seed=45)
#     positions_list.append(pos4)
    
#     initial_positions = np.vstack(positions_list)
#     n_particles_actual = initial_positions.shape[0]
    
#     print(f"   🌱 Seeded {n_particles_actual} particles in 4 strategic regions")
    
#     # Configure advanced tracking options
#     options = TrackerOptions(
#         max_memory_gb=6.0,
#         record_velocities=True,
#         oom_recovery=True,
#         use_jax_jit=True,
#         batch_size=300,  # Smaller batches for stability
#         progress_callback=lambda p: print(f"      ⏳ Tracking progress: {p*100:.1f}%") 
#                          if int(p * 20) != int((p - 0.05) * 20) else None  # Every 5%
#     )
    
#     # FIXED: Create proper boundary condition with domain bounds
#     boundary_fn = periodic_boundary(domain_bounds)
    
#     # Test boundary condition with sample data
#     try:
#         test_pos = initial_positions[:5].copy()  # Test with 5 particles
#         test_result = boundary_fn(test_pos)
#         print(f"   ✅ Boundary condition test passed")
#     except Exception as e:
#         print(f"   ⚠️  Boundary condition test failed: {e}")
#         print(f"   🔧 Using no boundary as fallback...")
#         from jaxtrace.tracking.boundary import no_boundary
#         boundary_fn = no_boundary()
    
#     # Try multiple integration strategies for robustness
#     integration_strategies = [
#         {
#             'name': 'RK4 High Resolution',
#             'integrator': 'rk4',
#             'n_timesteps': 80,
#             'batch_size': 200
#         },
#         {
#             'name': 'RK2 Medium Resolution', 
#             'integrator': 'rk2',
#             'n_timesteps': 60,
#             'batch_size': 300
#         },
#         {
#             'name': 'Euler Simple',
#             'integrator': 'euler', 
#             'n_timesteps': 40,
#             'batch_size': 500
#         }
#     ]
    
#     trajectory = None
#     successful_strategy = None
    
#     for strategy in integration_strategies:
#         print(f"\n   🧮 Attempting: {strategy['name']}")
#         print(f"      Method: {strategy['integrator']}")
#         print(f"      Timesteps: {strategy['n_timesteps']}")
#         print(f"      Batch size: {strategy['batch_size']}")
        
#         try:
#             # Update tracker options for this strategy
#             options.batch_size = strategy['batch_size']
            
#             # Create tracker
#             tracker = create_tracker(
#                 integrator_name=strategy['integrator'],
#                 field=field,
#                 boundary_condition=boundary_fn,
#                 **options.__dict__
#             )
            
#             # Estimate memory and runtime
#             runtime_est = tracker.estimate_runtime(
#                 n_particles_actual, 
#                 strategy['n_timesteps'],
#                 calibration_particles=min(50, n_particles_actual // 10)
#             )
            
#             if runtime_est['success']:
#                 print(f"      ⏱️  Estimated time: {runtime_est['estimated_runtime_minutes']:.1f} min")
#                 print(f"      💾 Estimated memory: {runtime_est['estimated_memory_gb']:.1f} GB")
                
#                 if runtime_est['estimated_memory_gb'] > 10:
#                     print(f"      ⚠️  High memory usage, reducing batch size...")
#                     options.batch_size = max(100, options.batch_size // 2)
            
#             # Execute tracking
#             start_time = time.time()
            
#             trajectory = tracker.track_particles(
#                 initial_positions=initial_positions,
#                 time_span=(t_start, min(t_end, t_start + 5.0)),  # Limit duration for stability
#                 n_timesteps=strategy['n_timesteps']
#             )
            
#             elapsed_time = time.time() - start_time
#             successful_strategy = strategy
            
#             print(f"      ✅ Success! Completed in {elapsed_time:.1f}s")
#             print(f"      📊 Result: {trajectory}")
#             break
            
#         except Exception as e:
#             print(f"      ❌ Strategy failed: {e}")
#             # Try to reduce problem size further
#             if 'batch_size' in str(e).lower() or 'memory' in str(e).lower():
#                 print(f"      🔧 Memory issue detected, trying reduced batch size...")
#                 try:
#                     options.batch_size = max(50, options.batch_size // 3)
                    
#                     tracker = create_tracker(
#                         integrator_name=strategy['integrator'],
#                         field=field,
#                         boundary_condition=boundary_fn,
#                         **options.__dict__
#                     )
                    
#                     trajectory = tracker.track_particles(
#                         initial_positions=initial_positions[:100],  # Reduce particles too
#                         time_span=(t_start, t_start + 2.0),  # Shorter duration
#                         n_timesteps=20  # Fewer timesteps
#                     )
                    
#                     print(f"      ✅ Reduced-size tracking successful!")
#                     successful_strategy = {**strategy, 'reduced': True}
#                     break
                    
#                 except Exception as e2:
#                     print(f"      ❌ Even reduced tracking failed: {e2}")
#             continue
    
#     if trajectory is None:
#         raise RuntimeError(f"Particle tracking failed completely: {e2}")
    
#     print(f"\n   🎉 Particle tracking completed successfully!")
#     print(f"      Strategy used: {successful_strategy['name']}")
#     print(f"      Final trajectory shape: {trajectory.positions.shape}")
#     print(f"      Memory usage: {trajectory.memory_usage_mb():.1f} MB")
    
#     return trajectory, successful_strategy

def execute_particle_tracking_analysis(field, n_particles=600, dt = 0.01, use_time_periodicity=False, 
                                     periodic_config=None):
    """
    Execute comprehensive particle tracking analysis with optional time-periodic field.
    
    Args:
        field: Time series velocity field
        n_particles: Number of particles to track
        use_time_periodicity: If True, create time-periodic field for longer tracking
        periodic_config: Dictionary with periodic field configuration:
            - 'time_slice': (start_time, end_time) or None for full range
            - 'n_periods': Number of periods to simulate (default: 5)
            - 'target_duration': Target simulation time (overrides n_periods if set)
            - 'transition_smoothing': Smoothing parameter for period transitions (0-1, default: 0.1)
    """
    print(f"\n🚀 Executing particle tracking analysis...")
    
    # Get field bounds for seeding
    bounds_min, bounds_max = field.get_spatial_bounds()
    t_start, t_end = field.get_time_bounds()
    
    # FIXED: Create proper bounds format for boundary conditions
    domain_bounds = [bounds_min.tolist(), bounds_max.tolist()]  # [[xmin,ymin,zmin], [xmax,ymax,zmax]]
    
    # Handle time periodicity setup
    working_field = field
    original_duration = t_end - t_start
    simulation_t_start = t_start
    simulation_t_end = t_end
    
    if use_time_periodicity:
        print(f"\n   🔄 Setting up time-periodic field...")
        
        # Default periodic configuration
        default_config = {
            'time_slice': None,  # Use full time range
            'n_periods': 5,      # Simulate 5 periods
            'target_duration': None,  # Use n_periods instead
            'transition_smoothing': 0.1  # 10% overlap for smooth transitions
        }
        
        if periodic_config is None:
            periodic_config = {}
        
        # Merge with defaults
        config = {**default_config, **periodic_config}
        
        # Create time-periodic field wrapper
        working_field, periodic_info = create_time_periodic_field(field, config)
        
        # Update simulation time bounds
        simulation_t_start = periodic_info['simulation_t_start']
        simulation_t_end = periodic_info['simulation_t_end']
        simulation_duration = simulation_t_end - simulation_t_start
        
        print(f"      ✅ Periodic field created:")
        print(f"         Original duration: {original_duration:.3f}")
        print(f"         Base period: {periodic_info['period_duration']:.3f}")
        print(f"         Number of periods: {periodic_info['n_periods']}")
        print(f"         Total simulation time: {simulation_duration:.3f}")
        print(f"         Time slice used: {periodic_info['time_slice_used']}")
    
    print(f"   📊 Analysis setup:")
    print(f"      Particles: {n_particles}")
    print(f"      Time span: {simulation_t_start:.2f} to {simulation_t_end:.2f}")
    print(f"      Simulation duration: {simulation_t_end - simulation_t_start:.2f}")
    print(f"      Domain bounds: {domain_bounds}")
    print(f"      Domain size: {[bounds_max[i] - bounds_min[i] for i in range(3)]}")
    print(f"      Time periodicity: {'✅ Enabled' if use_time_periodicity else '❌ Disabled'}")
    
    # Advanced particle seeding strategy
    print(f"   🌱 Advanced particle seeding...")
    
    # Seed particles in multiple strategic regions
    positions_list = []
    n_regions = 3
    n_per_region = n_particles // n_regions
    
    # Region 1: Domain center (high activity zone)
    # center = [(bounds_min[i] + bounds_max[i]) / 2 for i in range(3)]
    
    # size = [(bounds_max[i] - bounds_min[i]) * 0.99 for i in range(3)]
    size = [(bounds_max[0] - bounds_min[0]) * 0.35,
            (bounds_max[1] - bounds_min[1]) * 0.5, 
            (bounds_max[2] - bounds_min[2]) * 0.9]
    center = [(bounds_min[0] + size[0]/2),
              (bounds_min[1] + bounds_max[1]) / 2,
              (bounds_min[2] + bounds_max[2]) / 2]
    region1_min = [center[i] - size[i]/2 for i in range(3)]
    region1_max = [center[i] + size[i]/2 for i in range(3)]
    n_region_1 = int(n_particles*0.75)#int(n_per_region*1.5)
    pos1 = random_seeds(n_region_1, [region1_min, region1_max], rng_seed=42)
    positions_list.append(pos1)
    
    # Region 2: Near domain boundaries (boundary effects)
    # boundary_thickness = min(bounds_max[i] - bounds_min[i] for i in range(3)) * 0.1
    n_region_2 = 0# int(n_per_region*0.7)
    # pos2 = random_seeds(
    #     n_region_2,
    #     [[bounds_min[0], bounds_min[1], bounds_min[2]],
    #      [bounds_min[0] + boundary_thickness, bounds_max[1], bounds_max[2]]],
    #     rng_seed=43
    # )
    # positions_list.append(pos2)
    
    # Region 3: High gradient zones (sample field to find)
    # For simplicity, use random sampling with slight bias toward center
    # pos3 = random_seeds(
    #     n_per_region,
    #     domain_bounds,  # Use proper bounds format
    #     rng_seed=44
    # )
    # positions_list.append(pos3)
    
    # Region 4: Remaining particles distributed randomly
    remaining = n_particles - n_region_1 - n_region_2 #n_particles - (n_regions-1) * n_per_region
    pos4 = random_seeds(remaining, domain_bounds, rng_seed=45)
    positions_list.append(pos4)
    
    initial_positions = np.vstack(positions_list)
    # print(f"   🌱 Initial positions shape: {initial_positions.shape}")
    n_particles_actual = initial_positions.shape[0]
    
    print(f"   🌱 Seeded {n_particles_actual} particles in 4 strategic regions")
    
    # Configure advanced tracking options
    options = TrackerOptions(
        max_memory_gb=6.0,
        record_velocities=True,
        oom_recovery=True,
        use_jax_jit=True,
        batch_size=1000,  # Smaller batches for stability
        progress_callback=None#lambda p: print(f"      ⏳ Tracking progress: {p*100:.1f}%") 
                        #  if int(p * 20) != int((p - 0.05) * 20) else None  # Every 5%
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
    
    # Adjust integration strategies for periodic fields
    base_timesteps_factor = 2.0 if use_time_periodicity else 1.0  # More timesteps for longer sims
    
    integration_strategies = [
        {
            'name': 'RK4 High Resolution' + (' (Periodic)' if use_time_periodicity else ''),
            'integrator': 'rk4',
            'n_timesteps': 4000,#int(80 * base_timesteps_factor),
            'batch_size': 1000
        }# ,
        # {
        #     'name': 'RK2 Medium Resolution' + (' (Periodic)' if use_time_periodicity else ''), 
        #     'integrator': 'rk2',
        #     'n_timesteps': int(60 * base_timesteps_factor),
        #     'batch_size': 300
        # },
        # {
        #     'name': 'Euler Simple' + (' (Periodic)' if use_time_periodicity else ''),
        #     'integrator': 'euler', 
        #     'n_timesteps': int(40 * base_timesteps_factor),
        #     'batch_size': 500
        # }
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
            
            # Create tracker with the working field (periodic or original)
            tracker = create_tracker(
                integrator_name=strategy['integrator'],
                field=working_field,
                boundary_condition=boundary_fn,
                **options.__dict__
            )
            print(f"      ✅ Tracker created successfully")
            
            # Estimate memory and runtime
            runtime_est = tracker.estimate_runtime(
                n_particles_actual, 
                strategy['n_timesteps'],
                calibration_particles=min(50, n_particles_actual // 10)
            )
            print(f"      ✅ Runtime estimation completed")
            
            if runtime_est['success']:
                print(f"      ⏱️  Estimated time: {runtime_est['estimated_runtime_minutes']:.1f} min")
                print(f"      💾 Estimated memory: {runtime_est['estimated_memory_gb']:.1f} GB")
                
                if runtime_est['estimated_memory_gb'] > 10:
                    print(f"      ⚠️  High memory usage, reducing batch size...")
                    options.batch_size = max(100, options.batch_size // 2)
            else:
                print(f"      ⚠️  Runtime estimation failed, proceeding with caution...")
            
            # Execute tracking
            start_time = time.time()
            
            # Use full simulation time span for periodic fields, limited span for original fields
            if use_time_periodicity:
                time_span = (simulation_t_start, simulation_t_end)
            else:
                # Limit duration for stability in non-periodic mode
                time_span = (simulation_t_start, min(simulation_t_end, simulation_t_start + 5.0))
            
            trajectory = tracker.track_particles(
                initial_positions=initial_positions,
                time_span=time_span,
                dt = dt,
                n_timesteps=strategy['n_timesteps']
            )
            
            elapsed_time = time.time() - start_time
            successful_strategy = strategy
            
            print(f"      ✅ Success! Completed in {elapsed_time:.1f}s")
            print(f"      📊 Result: {trajectory}")
            
            # Add periodic field info to trajectory if applicable
            if use_time_periodicity:
                trajectory._periodic_field_info = periodic_info
            
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
                        field=working_field,
                        boundary_condition=boundary_fn,
                        **options.__dict__
                    )
                    
                    # Further reduced simulation for fallback
                    fallback_positions = initial_positions[:100]  # Reduce particles
                    fallback_duration = 2.0 if not use_time_periodicity else min(4.0, simulation_t_end - simulation_t_start)
                    fallback_timesteps = 20
                    
                    trajectory = tracker.track_particles(
                        initial_positions=fallback_positions,
                        time_span=(simulation_t_start, simulation_t_start + fallback_duration),
                        n_timesteps=fallback_timesteps,
                        dt = dt
                    )
                    
                    print(f"      ✅ Reduced-size tracking successful!")
                    successful_strategy = {**strategy, 'reduced': True}
                    
                    if use_time_periodicity:
                        trajectory._periodic_field_info = periodic_info
                    
                    break
                    
                except Exception as e2:
                    print(f"      ❌ Even reduced tracking failed: {e2}")
            continue
    
    if trajectory is None:
        raise RuntimeError(f"Particle tracking failed completely")
    
    print(f"\n   🎉 Particle tracking completed successfully!")
    print(f"      Strategy used: {successful_strategy['name']}")
    print(f"      Final trajectory shape: {trajectory.positions.shape}")
    print(f"      Memory usage: {trajectory.memory_usage_mb():.1f} MB")
    
    if use_time_periodicity:
        print(f"      🔄 Periodic field statistics:")
        print(f"         Periods completed: {(trajectory.duration / periodic_info['period_duration']):.2f}")
        print(f"         Effective simulation multiplier: {(trajectory.duration / original_duration):.1f}x")
    
    return trajectory, successful_strategy


# def create_time_periodic_field(original_field, config):
#     """
#     Create a time-periodic field wrapper that repeats a time slice cyclically.
    
#     Args:
#         original_field: TimeSeriesField to make periodic
#         config: Configuration dictionary with periodic parameters
        
#     Returns:
#         (periodic_field, info_dict): Periodic field wrapper and metadata
#     """
#     print(f"      🔧 Creating time-periodic field wrapper...")
    
#     # Get original time bounds
#     t_orig_start, t_orig_end = original_field.get_time_bounds()
#     original_duration = t_orig_end - t_orig_start
#     print(f"         Original time range: [{t_orig_start:.3f}, {t_orig_end:.3f}] (duration: {original_duration:.3f})")
    
#     # Determine time slice to use
#     if config['time_slice'] is None:
#         # Use full time range
#         slice_start, slice_end = t_orig_start, t_orig_end
#         slice_info = "full range"
#     else:
#         slice_start, slice_end = config['time_slice']
#         slice_start = max(slice_start, t_orig_start)
#         slice_end = min(slice_end, t_orig_end)
#         slice_info = f"[{slice_start:.3f}, {slice_end:.3f}]"
#         print(f"         Using time slice: {slice_info}")
    
#     period_duration = slice_end - slice_start
    
#     if period_duration <= 0:
#         raise ValueError(f"Invalid time slice: period duration = {period_duration}")
    
#     # Determine total simulation duration
#     if config['target_duration'] is not None:
#         total_duration = config['target_duration']
#         n_periods = max(1, int(np.ceil(total_duration / period_duration)))
#     else:
#         n_periods = config['n_periods']
#         total_duration = n_periods * period_duration
    
#     # Create the periodic field wrapper
#     periodic_field = TimePeriodicFieldWrapper(
#         original_field=original_field,
#         period_start=slice_start,
#         period_end=slice_end,
#         n_periods=n_periods,
#         transition_smoothing=config['transition_smoothing']
#     )
    
#     # Information dictionary
#     info = {
#         'period_duration': period_duration,
#         'n_periods': n_periods,
#         'total_duration': total_duration,
#         'simulation_t_start': 0.0,  # Periodic field starts at t=0
#         'simulation_t_end': total_duration,
#         'time_slice_used': slice_info,
#         'original_duration': original_duration,
#         'transition_smoothing': config['transition_smoothing']
#     }
    
#     print(f"         ✅ Period duration: {period_duration:.3f}")
#     print(f"         ✅ Number of periods: {n_periods}")
#     print(f"         ✅ Total simulation time: {total_duration:.3f}")
    
#     return periodic_field, info


# class TimePeriodicFieldWrapper:
#     """
#     Wrapper that makes a time series field periodic by repeating a time slice.
    
#     Implements the same interface as TimeSeriesField but with periodic time behavior.
#     """
    
#     def __init__(self, original_field, period_start, period_end, n_periods, transition_smoothing=0.1):
#         self.original_field = original_field
#         self.period_start = period_start
#         self.period_end = period_end
#         self.period_duration = period_end - period_start
#         self.n_periods = n_periods
#         self.total_duration = n_periods * self.period_duration
#         self.transition_smoothing = transition_smoothing
        
#         # Cache original field properties
#         self._spatial_bounds = original_field.get_spatial_bounds()
        
#         # Validate period bounds
#         orig_t_start, orig_t_end = original_field.get_time_bounds()
#         if period_start < orig_t_start or period_end > orig_t_end:
#             raise ValueError(f"Period [{period_start}, {period_end}] outside original time range [{orig_t_start}, {orig_t_end}]")
    
#     def __call__(self, t, x):
#         """Evaluate velocity field at time t and positions x with periodic time."""
#         # Map simulation time to periodic time within original field
#         t_periodic = self._map_to_periodic_time(t)
        
#         # Handle smoothing at period boundaries if requested
#         if self.transition_smoothing > 0 and isinstance(t, (int, float)):
#             return self._evaluate_with_smoothing(t, x, t_periodic)
#         else:
#             return self.original_field(t_periodic, x)
    
#     def _map_to_periodic_time(self, t):
#         """Map simulation time to time within the original period."""
#         if isinstance(t, (int, float)):
#             # Scalar time
#             t_mod = t % self.period_duration
#             return self.period_start + t_mod
#         else:
#             # Array of times
#             t_array = np.asarray(t)
#             t_mod = t_array % self.period_duration
#             return self.period_start + t_mod
    
#     def _evaluate_with_smoothing(self, t, x, t_periodic):
#         """Evaluate with smooth transitions at period boundaries."""
#         smoothing_width = self.transition_smoothing * self.period_duration
        
#         # Check if we're near a period boundary
#         t_in_period = t % self.period_duration
        
#         # Near end of period - blend with beginning
#         if t_in_period > (self.period_duration - smoothing_width):
#             alpha = (self.period_duration - t_in_period) / smoothing_width
            
#             # Current time velocity
#             v1 = self.original_field(t_periodic, x)
            
#             # Beginning of period velocity
#             t_begin = self.period_start + (t_in_period - self.period_duration + smoothing_width)
#             v2 = self.original_field(t_begin, x)
            
#             return alpha * v1 + (1 - alpha) * v2
        
#         # Near beginning of period - blend with end
#         elif t_in_period < smoothing_width:
#             alpha = t_in_period / smoothing_width
            
#             # Current time velocity
#             v1 = self.original_field(t_periodic, x)
            
#             # End of period velocity  
#             t_end = self.period_end - smoothing_width + t_in_period
#             v2 = self.original_field(t_end, x)
            
#             return alpha * v1 + (1 - alpha) * v2
        
#         else:
#             # No smoothing needed
#             return self.original_field(t_periodic, x)
    
#     def get_time_bounds(self):
#         """Get time bounds of the periodic simulation."""
#         return (0.0, self.total_duration)
    
#     def get_spatial_bounds(self):
#         """Get spatial bounds (same as original field)."""
#         return self._spatial_bounds
    
#     def validate_data(self):
#         """Validate the periodic field setup."""
#         self.original_field.validate_data()
        
#         if self.period_duration <= 0:
#             raise ValueError(f"Invalid period duration: {self.period_duration}")
        
#         if self.n_periods < 1:
#             raise ValueError(f"Invalid number of periods: {self.n_periods}")
    
#     @property
#     def data(self):
#         """Access to original field data (for compatibility)."""
#         return self.original_field.data
    
#     @property
#     def T(self):
#         """Effective number of time steps in periodic simulation."""
#         return self.original_field.T * self.n_periods  # Approximate
    
#     @property
#     def N(self):
#         """Number of spatial grid points."""
#         return self.original_field.N
    
#     def memory_usage_mb(self):
#         """Estimate memory usage (same as original since we don't duplicate data)."""
#         return self.original_field.memory_usage_mb()

def create_time_periodic_field(original_field, config):
    """
    Create a time-periodic field wrapper that repeats a time slice cyclically.
    
    Args:
        original_field: TimeSeriesField to make periodic
        config: Configuration dictionary with periodic parameters
        
    Returns:
        (periodic_field, info_dict): Periodic field wrapper and metadata
    """
    print(f"      🔧 Creating time-periodic field wrapper...")
    
    # Get original time bounds
    t_orig_start, t_orig_end = original_field.get_time_bounds()
    original_duration = t_orig_end - t_orig_start
    
    # Determine time slice to use
    if config['time_slice'] is None:
        # Use full time range
        slice_start, slice_end = t_orig_start, t_orig_end
        slice_info = "full range"
    else:
        slice_start, slice_end = config['time_slice']
        slice_start = max(slice_start, t_orig_start)
        slice_end = min(slice_end, t_orig_end)
        slice_info = f"[{slice_start:.3f}, {slice_end:.3f}]"
    
    period_duration = slice_end - slice_start
    
    if period_duration <= 0:
        raise ValueError(f"Invalid time slice: period duration = {period_duration}")
    
    # Determine total simulation duration
    if config['target_duration'] is not None:
        total_duration = config['target_duration']
        n_periods = max(1, int(np.ceil(total_duration / period_duration)))
    else:
        n_periods = config['n_periods']
        total_duration = n_periods * period_duration
    
    # Create the periodic field wrapper
    periodic_field = TimePeriodicFieldWrapper(
        original_field=original_field,
        period_start=slice_start,
        period_end=slice_end,
        n_periods=n_periods,
        transition_smoothing=config['transition_smoothing']
    )
    
    # Information dictionary
    info = {
        'period_duration': period_duration,
        'n_periods': n_periods,
        'total_duration': total_duration,
        'simulation_t_start': 0.0,  # Periodic field starts at t=0
        'simulation_t_end': total_duration,
        'time_slice_used': slice_info,
        'original_duration': original_duration,
        'transition_smoothing': config['transition_smoothing']
    }
    
    print(f"         ✅ Period duration: {period_duration:.3f}")
    print(f"         ✅ Number of periods: {n_periods}")
    print(f"         ✅ Total simulation time: {total_duration:.3f}")
    
    return periodic_field, info


# class TimePeriodicFieldWrapper:
#     """
#     Wrapper that makes a time series field periodic by repeating a time slice.
    
#     Implements the same interface as TimeSeriesField but with periodic time behavior.
#     """
    
#     def __init__(self, original_field, period_start, period_end, n_periods, transition_smoothing=0.1):
#         self.original_field = original_field
#         self.period_start = period_start
#         self.period_end = period_end
#         self.period_duration = period_end - period_start
#         self.n_periods = n_periods
#         self.total_duration = n_periods * self.period_duration
#         self.transition_smoothing = transition_smoothing
        
#         # Cache original field properties
#         self._spatial_bounds = original_field.get_spatial_bounds()
        
#         # Validate period bounds
#         orig_t_start, orig_t_end = original_field.get_time_bounds()
#         if period_start < orig_t_start or period_end > orig_t_end:
#             raise ValueError(f"Period [{period_start}, {period_end}] outside original time range [{orig_t_start}, {orig_t_end}]")
        
#         # DEBUG: Check what methods the original field has
#         print(f"         🔍 Original field type: {type(original_field).__name__}")
#         field_methods = [method for method in dir(original_field) if not method.startswith('_')]
#         print(f"         🔍 Available methods: {field_methods}")
        
#         # Determine how to call the original field
#         self._field_evaluator = self._setup_field_evaluator()
    
#     def _setup_field_evaluator(self):
#         """Determine the correct way to evaluate the original field."""
#         original = self.original_field
        
#         # Try different common field interfaces
#         if hasattr(original, '__call__'):
#             print(f"         ✅ Using __call__ interface")
#             return lambda t, x: original(t, x)
#         elif hasattr(original, 'evaluate'):
#             print(f"         ✅ Using evaluate() interface")  
#             return lambda t, x: original.evaluate(t, x)
#         elif hasattr(original, 'get_velocity'):
#             print(f"         ✅ Using get_velocity() interface")
#             return lambda t, x: original.get_velocity(t, x)
#         elif hasattr(original, 'interpolate'):
#             print(f"         ✅ Using interpolate() interface")
#             return lambda t, x: original.interpolate(t, x)
#         else:
#             # Try to find any method that might work
#             callable_methods = []
#             for attr_name in dir(original):
#                 if not attr_name.startswith('_'):
#                     attr = getattr(original, attr_name)
#                     if callable(attr):
#                         callable_methods.append(attr_name)
            
#             raise AttributeError(f"Cannot find suitable evaluation method for field. "
#                                f"Available callable methods: {callable_methods}")
    
#     def __call__(self, t, x):
#         """Evaluate velocity field at time t and positions x with periodic time."""
#         # Map simulation time to periodic time within original field
#         t_periodic = self._map_to_periodic_time(t)
        
#         # Handle smoothing at period boundaries if requested
#         if self.transition_smoothing > 0 and isinstance(t, (int, float)):
#             return self._evaluate_with_smoothing(t, x, t_periodic)
#         else:
#             return self._field_evaluator(t_periodic, x)
    
#     def evaluate(self, t, x):
#         """Alternative interface - same as __call__."""
#         return self.__call__(t, x)
    
#     def get_velocity(self, t, x):
#         """Alternative interface - same as __call__."""
#         return self.__call__(t, x)
    
#     def interpolate(self, t, x):
#         """Alternative interface - same as __call__."""
#         return self.__call__(t, x)
    
#     def _map_to_periodic_time(self, t):
#         """Map simulation time to time within the original period."""
#         if isinstance(t, (int, float)):
#             # Scalar time
#             if t < 0:
#                 # Handle negative times
#                 t_mod = ((-t) % self.period_duration) 
#                 t_mod = self.period_duration - t_mod if t_mod > 0 else 0
#             else:
#                 t_mod = t % self.period_duration
#             return self.period_start + t_mod
#         else:
#             # Array of times
#             t_array = np.asarray(t)
#             t_mod = np.mod(t_array, self.period_duration)
#             return self.period_start + t_mod
    
#     def _evaluate_with_smoothing(self, t, x, t_periodic):
#         """Evaluate with smooth transitions at period boundaries."""
#         smoothing_width = self.transition_smoothing * self.period_duration
        
#         # Check if we're near a period boundary
#         t_in_period = t % self.period_duration
        
#         try:
#             # Near end of period - blend with beginning
#             if t_in_period > (self.period_duration - smoothing_width):
#                 alpha = (self.period_duration - t_in_period) / smoothing_width
                
#                 # Current time velocity
#                 v1 = self._field_evaluator(t_periodic, x)
                
#                 # Beginning of period velocity
#                 t_begin = self.period_start + (t_in_period - self.period_duration + smoothing_width)
#                 v2 = self._field_evaluator(t_begin, x)
                
#                 return alpha * v1 + (1 - alpha) * v2
            
#             # Near beginning of period - blend with end
#             elif t_in_period < smoothing_width:
#                 alpha = t_in_period / smoothing_width
                
#                 # Current time velocity
#                 v1 = self._field_evaluator(t_periodic, x)
                
#                 # End of period velocity  
#                 t_end = self.period_end - smoothing_width + t_in_period
#                 v2 = self._field_evaluator(t_end, x)
                
#                 return alpha * v1 + (1 - alpha) * v2
            
#             else:
#                 # No smoothing needed
#                 return self._field_evaluator(t_periodic, x)
        
#         except Exception as e:
#             print(f"         ⚠️  Smoothing failed, using direct evaluation: {e}")
#             return self._field_evaluator(t_periodic, x)
    
#     def get_time_bounds(self):
#         """Get time bounds of the periodic simulation."""
#         return (0.0, self.total_duration)
    
#     def get_spatial_bounds(self):
#         """Get spatial bounds (same as original field)."""
#         return self._spatial_bounds
    
#     def validate_data(self):
#         """Validate the periodic field setup."""
#         if hasattr(self.original_field, 'validate_data'):
#             self.original_field.validate_data()
        
#         if self.period_duration <= 0:
#             raise ValueError(f"Invalid period duration: {self.period_duration}")
        
#         if self.n_periods < 1:
#             raise ValueError(f"Invalid number of periods: {self.n_periods}")
    
#     @property
#     def data(self):
#         """Access to original field data (for compatibility)."""
#         return self.original_field.data
    
#     @property
#     def T(self):
#         """Effective number of time steps in periodic simulation."""
#         return getattr(self.original_field, 'T', 100) * self.n_periods  # Approximate
    
#     @property
#     def N(self):
#         """Number of spatial grid points."""
#         return getattr(self.original_field, 'N', 1000)
    
#     def memory_usage_mb(self):
#         """Estimate memory usage (same as original since we don't duplicate data)."""
#         if hasattr(self.original_field, 'memory_usage_mb'):
#             return self.original_field.memory_usage_mb()
#         else:
#             return 100.0  # Default estimate
    
#     # Forward other methods to original field
#     def __getattr__(self, name):
#         """Forward any missing methods to the original field."""
#         if hasattr(self.original_field, name):
#             attr = getattr(self.original_field, name)
#             if callable(attr):
#                 return attr
#             else:
#                 return attr
#         else:
#             raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

# class TimePeriodicFieldWrapper:
#     """
#     Wrapper that makes a time series field periodic by repeating a time slice.
#     Implements the same interface as the base field (evaluate/__call__), with periodic time behavior.
#     """

#     def __init__(self, original_field, period_start, period_end, n_periods, transition_smoothing=0.1):
#         self.original_field = original_field
#         self.period_start = period_start
#         self.period_end = period_end
#         self.period_duration = period_end - period_start
#         self.n_periods = n_periods
#         self.total_duration = n_periods * self.period_duration
#         self.transition_smoothing = transition_smoothing

#         # Cache bounds
#         self._spatial_bounds = original_field.get_spatial_bounds()

#         # Validate period
#         orig_t_start, orig_t_end = original_field.get_time_bounds()
#         if period_start < orig_t_start or period_end > orig_t_end:
#             raise ValueError(f"Period [{period_start}, {period_end}] outside original time range [{orig_t_start}, {orig_t_end}]")

#     # --- Robust evaluation dispatcher ---
#     def _eval_base(self, t, x):
#         """
#         Dispatch evaluation to the underlying field using the interface it provides.
#         Supports: __call__(t, x), evaluate(t, x), interp(t, x), interpolate(t, x)
#         """
#         f = self.original_field
#         if hasattr(f, "__call__"):
#             return f(t, x)
#         if hasattr(f, "evaluate"):
#             return f.evaluate(t, x)
#         if hasattr(f, "interp"):
#             return f.interp(t, x)
#         if hasattr(f, "interpolate"):
#             return f.interpolate(t, x)
#         raise TypeError("Underlying field is not callable and has no evaluate/interp/interpolate method")

#     # --- Public API used by tracker(s) ---
#     def __call__(self, t, x):
#         """Evaluate velocity field at time t and positions x with periodic time."""
#         t_periodic = self._map_to_periodic_time(t)
#         if self.transition_smoothing > 0 and np.isscalar(t):
#             return self._evaluate_with_smoothing(t, x, t_periodic)
#         return self._eval_base(t_periodic, x)

#     # Some trackers call .evaluate instead of __call__
#     def evaluate(self, t, x):
#         t_periodic = self._map_to_periodic_time(t)
#         if self.transition_smoothing > 0 and np.isscalar(t):
#             return self._evaluate_with_smoothing(t, x, t_periodic)
#         return self._eval_base(t_periodic, x)

#     # --- Internals ---
#     def _map_to_periodic_time(self, t):
#         """Map simulation time to time within the original period."""
#         if np.isscalar(t):
#             return self.period_start + (t % self.period_duration)
#         t_arr = np.asarray(t)
#         return self.period_start + (t_arr % self.period_duration)

#     def _evaluate_with_smoothing(self, t, x, t_periodic):
#         """Evaluate with smooth transitions near period boundaries."""
#         smoothing_width = self.transition_smoothing * self.period_duration
#         t_in_period = t % self.period_duration

#         # Near end of period: blend end -> start
#         if t_in_period > (self.period_duration - smoothing_width):
#             alpha = (self.period_duration - t_in_period) / smoothing_width
#             v_now = self._eval_base(t_periodic, x)

#             # Equivalent time near the beginning of the next period
#             t_begin = self.period_start + (t_in_period - self.period_duration + smoothing_width)
#             v_begin = self._eval_base(t_begin, x)
#             return alpha * v_now + (1 - alpha) * v_begin

#         # Near beginning of period: blend start -> end
#         if t_in_period < smoothing_width:
#             alpha = t_in_period / smoothing_width
#             v_now = self._eval_base(t_periodic, x)

#             # Equivalent time near the end of the previous period
#             t_end = self.period_end - smoothing_width + t_in_period
#             v_end = self._eval_base(t_end, x)
#             return alpha * v_now + (1 - alpha) * v_end

#         # Away from boundaries
#         return self._eval_base(t_periodic, x)

#     # --- Compatibility methods/properties often used by code ---
#     def get_time_bounds(self):
#         """Time bounds of the periodic simulation."""
#         return (0.0, self.total_duration)

#     def get_spatial_bounds(self):
#         """Same spatial bounds as the original field."""
#         return self._spatial_bounds

#     def validate_data(self):
#         self.original_field.validate_data()
#         if self.period_duration <= 0:
#             raise ValueError(f"Invalid period duration: {self.period_duration}")
#         if self.n_periods < 1:
#             raise ValueError(f"Invalid number of periods: {self.n_periods}")

#     @property
#     def data(self):
#         return self.original_field.data

#     @property
#     def T(self):
#         # If the tracker uses this, it's just an approximate "time steps" count
#         return int(self.original_field.T) * int(self.n_periods)

#     @property
#     def N(self):
#         return self.original_field.N

#     def memory_usage_mb(self):
#         return self.original_field.memory_usage_mb()

class TimePeriodicFieldWrapper:
    """
    Wrapper that makes a time series field periodic by repeating a time slice.
    Implements the same interface as the base field (evaluate/__call__), with periodic time behavior.
    """

    def __init__(self, original_field, period_start, period_end, n_periods, transition_smoothing=0.1):
        self.original_field = original_field
        self.period_start = period_start
        self.period_end = period_end
        self.period_duration = period_end - period_start
        self.n_periods = n_periods
        self.total_duration = n_periods * self.period_duration
        self.transition_smoothing = transition_smoothing

        # Cache bounds and field info
        self._spatial_bounds = original_field.get_spatial_bounds()
        self._base_T = int(getattr(original_field, "T", 0))
        if self._base_T == 0 and hasattr(original_field, "times"):
            try:
                self._base_T = int(len(original_field.times))
            except Exception:
                self._base_T = 0

        # Expose time-like attributes so trackers that rely on attributes (not methods)
        # will see the extended periodic time range.
        self.t_min = 0.0
        self.t_max = float(self.total_duration)

        # Optional: synthetic times cache
        self._times = None

        # Validate period
        orig_t_start, orig_t_end = original_field.get_time_bounds()
        if period_start < orig_t_start or period_end > orig_t_end:
            raise ValueError(
                f"Period [{period_start}, {period_end}] outside original time range "
                f"[{orig_t_start}, {orig_t_end}]"
            )

    # ---------- Robust argument order detection ----------
    @staticmethod
    def _is_positions_like(arr) -> bool:
        import numpy as np
        try:
            a = np.asarray(arr)
        except Exception:
            return False
        if a.ndim == 2 and a.shape[1] in (2, 3):
            return True
        if a.ndim == 1 and a.size in (2, 3):
            return True
        return False

    @staticmethod
    def _is_scalar_like(x) -> bool:
        import numpy as np
        try:
            a = np.asarray(x)
        except Exception:
            return np.isscalar(x)
        return a.ndim == 0

    def _parse_time_and_positions(self, a, b):
        """
        Accept both call conventions:
        - (t, positions) or (positions, t)
        Returns (t: float or array, positions: array)
        """
        import numpy as np
        if self._is_positions_like(a) and self._is_scalar_like(b):
            return float(np.asarray(b)), np.asarray(a)
        if self._is_positions_like(b) and self._is_scalar_like(a):
            return float(np.asarray(a)), np.asarray(b)

        # Fallbacks
        if self._is_scalar_like(a) and not self._is_scalar_like(b):
            return float(np.asarray(a)), np.asarray(b)
        if self._is_scalar_like(b) and not self._is_scalar_like(a):
            return float(np.asarray(b)), np.asarray(a)

        raise TypeError(
            f"Cannot determine (t, positions) from arguments: "
            f"a.shape={getattr(np.asarray(a), 'shape', None)}, "
            f"b.shape={getattr(np.asarray(b), 'shape', None)}"
        )
        # --- Robust evaluation dispatcher to underlying field ---
    def _eval_base(self, t, x):
        """
        Dispatch evaluation to the underlying field using the interface it provides.
        Supports: __call__(t, x), evaluate(t, x), interp(t, x), interpolate(t, x)
        """
        f = self.original_field
        if hasattr(f, "__call__"):
            return f(t, x)
        if hasattr(f, "evaluate"):
            return f.evaluate(t, x)
        if hasattr(f, "interp"):
            return f.interp(t, x)
        if hasattr(f, "interpolate"):
            return f.interpolate(t, x)
        raise TypeError(
            "Underlying field is not callable and has no evaluate/interp/interpolate method"
        )
    
    def _evaluate_with_smoothing(self, t, x, t_periodic):
        """Evaluate with smooth transitions near period boundaries."""
        smoothing_width = self.transition_smoothing * self.period_duration
        t_in_period = t % self.period_duration

        # Near end of period: blend end -> start
        if t_in_period > (self.period_duration - smoothing_width):
            alpha = (self.period_duration - t_in_period) / smoothing_width
            v_now = self._eval_base(t_periodic, x)

            # Equivalent time near the beginning of the next period
            t_begin = self.period_start + (t_in_period - self.period_duration + smoothing_width)
            v_begin = self._eval_base(t_begin, x)
            return alpha * v_now + (1 - alpha) * v_begin

        # Near beginning of period: blend start -> end
        if t_in_period < smoothing_width:
            alpha = t_in_period / smoothing_width
            v_now = self._eval_base(t_periodic, x)

            # Equivalent time near the end of the previous period
            t_end = self.period_end - smoothing_width + t_in_period
            v_end = self._eval_base(t_end, x)
            return alpha * v_now + (1 - alpha) * v_end

        # Away from boundaries
        return self._eval_base(t_periodic, x)

    # --- Public API used by tracker(s) ---
    def __call__(self, a, b):
        """Evaluate velocity field; accepts (t, positions) or (positions, t)."""
        t, x = self._parse_time_and_positions(a, b)
        t_periodic = self._map_to_periodic_time(t)
        if self.transition_smoothing > 0 and np.isscalar(t):
            return self._evaluate_with_smoothing(t, x, t_periodic)
        return self._eval_base(t_periodic, x)

    def evaluate(self, a, b):
        """Alternative evaluation API; accepts (t, positions) or (positions, t)."""
        t, x = self._parse_time_and_positions(a, b)
        t_periodic = self._map_to_periodic_time(t)
        if self.transition_smoothing > 0 and np.isscalar(t):
            return self._evaluate_with_smoothing(t, x, t_periodic)
        return self._eval_base(t_periodic, x)
    
        # --- Time mapping and bounds ---
    def _map_to_periodic_time(self, t):
        """Map simulation time to time within the original period: [start, end)."""
        import numpy as np
        if np.isscalar(t):
            return self.period_start + (t % self.period_duration)
        t_arr = np.asarray(t)
        return self.period_start + (t_arr % self.period_duration)

    def get_time_bounds(self):
        """Time bounds of the periodic simulation."""
        return (self.t_min, self.t_max)

    @property
    def times(self):
        """
        Synthetic time array across the periodic duration.
        Useful for components that inspect field.times.
        """
        import numpy as np
        if self._times is not None:
            return self._times
        if self._base_T and self._base_T > 1:
            # Build times per period with same sampling count as the base field
            t0 = 0.0
            times_period = np.linspace(
                t0, self.period_duration, self._base_T, endpoint=False, dtype=np.float32
            )
            self._times = np.concatenate(
                [times_period + p * self.period_duration for p in range(self.n_periods)]
            ).astype(np.float32)
        else:
            # Fallback: just two points
            self._times = np.array([0.0, self.total_duration], dtype=np.float32)
        return self._times

    @property
    def T(self):
        """Effective number of time samples across the periodic duration."""
        if self._base_T and self._base_T > 0:
            return self._base_T * int(self.n_periods)
        # Fallback to length of synthetic times
        return int(len(self.times))

    @property
    def N(self):
        return int(getattr(self.original_field, "N", 0))

    def memory_usage_mb(self):
        return self.original_field.memory_usage_mb()

def analyze_trajectory_results(trajectory, strategy_info):
    """Analyze tracking results comprehensively."""
    print(f"\n📊 Analyzing trajectory results...")
    
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


# def perform_density_estimation_analysis(trajectory, output_dir="output"):
#     """NEW: Perform comprehensive density estimation using KDE and SPH."""
#     print(f"\n🔬 Performing density estimation analysis...")
    
#     if not DENSITY_AVAILABLE:
#         print(f"   ⚠️  Density estimation modules not available - skipping")
#         return None
    
#     output_path = Path(output_dir)
#     output_path.mkdir(exist_ok=True)
    
#     # Get final particle positions for density analysis
#     final_positions = trajectory.positions[-1]  # (N, 3)
#     print(f"   📊 Analyzing density at final timestep with {final_positions.shape[0]} particles")
    
#     density_results = {}
    
#     # 1. KDE Analysis
#     print(f"   📈 Performing KDE analysis...")
    
#     try:
#         # Create KDE estimator with automatic bandwidth selection
#         kde_scott = KDEEstimator(
#             bandwidth_rule='scott'#,
#             # kernel='gaussian'
#         )
        
#         kde_silverman = KDEEstimator(
#             bandwidth_rule='silverman'#, 
#             # kernel='gaussian'
#         )
        
#         # Fit KDE models
#         kde_scott.fit(final_positions)
#         kde_silverman.fit(final_positions)
        
#         print(f"      ✅ Scott bandwidth: {kde_scott.bandwidth}")
#         print(f"      ✅ Silverman bandwidth: {kde_silverman.bandwidth}")
        
#         # Create evaluation grid
#         bounds_min = np.min(final_positions, axis=0)
#         bounds_max = np.max(final_positions, axis=0)
        
#         # Expand bounds slightly
#         margin = 0.1 * (bounds_max - bounds_min)
#         bounds_min_expanded = bounds_min - margin
#         bounds_max_expanded = bounds_max + margin
        
#         # Create 2D evaluation grid for visualization
#         n_grid = 50
#         x_eval = np.linspace(bounds_min_expanded[0], bounds_max_expanded[0], n_grid)
#         y_eval = np.linspace(bounds_min_expanded[1], bounds_max_expanded[1], n_grid)
#         X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
        
#         # For 2D density, use Z = mean(final_positions[:, 2])
#         z_mean = np.mean(final_positions[:, 2])
#         eval_points_2d = np.column_stack([
#             X_eval.ravel(), 
#             Y_eval.ravel(), 
#             np.full(X_eval.size, z_mean)
#         ])
        
#         # Evaluate densities
#         density_scott = kde_scott.evaluate_2d(eval_points_2d).reshape(X_eval.shape)
#         density_silverman = kde_silverman.evaluate_2d(eval_points_2d).reshape(X_eval.shape)
        
#         density_results['kde'] = {
#             'scott': {
#                 'estimator': kde_scott,
#                 'density': density_scott,
#                 'bandwidth': kde_scott.bandwidth
#             },
#             'silverman': {
#                 'estimator': kde_silverman,
#                 'density': density_silverman,
#                 'bandwidth': kde_silverman.bandwidth
#             },
#             'eval_grid': (X_eval, Y_eval),
#             'eval_points': eval_points_2d
#         }
        
#         print(f"      ✅ KDE analysis completed")
        
#     except Exception as e:
#         print(f"      ❌ KDE analysis failed: {e}")
#         density_results['kde'] = None
    
#     # 2. SPH Analysis
#     print(f"   🌊 Performing SPH density analysis...")
    
#     try:
#         # Create SPH estimator with different smoothing lengths
#         smoothing_lengths = [0.1, 0.2, 0.3]  # Relative to domain size
#         domain_size = np.mean(bounds_max - bounds_min)
        
#         sph_results = {}
        
#         for h_rel in smoothing_lengths:
#             h_abs = h_rel * domain_size
            
#             sph_estimator = SPHDensityEstimator(
#                 smoothing_length=h_abs,
#                 kernel_type='cubic_spline'
#             )
            
#             sph_estimator.fit(final_positions)
            
#             # Evaluate on same grid as KDE
#             density_sph = sph_estimator.evaluate(eval_points_2d).reshape(X_eval.shape)
            
#             sph_results[f'h_{h_rel:.1f}'] = {
#                 'estimator': sph_estimator,
#                 'density': density_sph,
#                 'smoothing_length': h_abs
#             }
            
#             print(f"      ✅ SPH h={h_abs:.3f} completed")
        
#         density_results['sph'] = sph_results
#         density_results['sph']['eval_grid'] = (X_eval, Y_eval)
        
#         print(f"      ✅ SPH analysis completed")
        
#     except Exception as e:
#         print(f"      ❌ SPH analysis failed: {e}")
#         density_results['sph'] = None
    
#     # 3. Create density visualizations
#     print(f"   🎨 Creating density visualizations...")
    
#     try:
#         fig, axes = plt.subplots(2, 3, figsize=(18, 12))
#         fig.suptitle('Density Estimation Analysis', fontsize=16)
        
#         # Plot particle scatter
#         ax = axes[0, 0]
#         scatter = ax.scatter(final_positions[:, 0], final_positions[:, 1], 
#                            alpha=0.6, s=20, c='blue')
#         ax.set_title('Final Particle Positions')
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.grid(True, alpha=0.3)
#         ax.set_aspect('equal', adjustable='box')
        
#         # KDE Scott
#         if density_results['kde']:
#             ax = axes[0, 1]
#             im = ax.contourf(X_eval, Y_eval, density_results['kde']['scott']['density'], 
#                            levels=20, cmap='viridis')
#             ax.scatter(final_positions[:, 0], final_positions[:, 1], 
#                       alpha=0.5, s=10, c='white', edgecolors='black', linewidths=0.5)
#             ax.set_title(f"KDE (Scott) - h={density_results['kde']['scott']['bandwidth']:.3f}")
#             ax.set_xlabel('X')
#             ax.set_ylabel('Y')
#             plt.colorbar(im, ax=ax)
#             ax.set_aspect('equal', adjustable='box')
        
#         # KDE Silverman
#         if density_results['kde']:
#             ax = axes[0, 2]
#             im = ax.contourf(X_eval, Y_eval, density_results['kde']['silverman']['density'], 
#                            levels=20, cmap='viridis')
#             ax.scatter(final_positions[:, 0], final_positions[:, 1], 
#                       alpha=0.5, s=10, c='white', edgecolors='black', linewidths=0.5)
#             ax.set_title(f"KDE (Silverman) - h={density_results['kde']['silverman']['bandwidth']:.3f}")
#             ax.set_xlabel('X')
#             ax.set_ylabel('Y')
#             plt.colorbar(im, ax=ax)
#             ax.set_aspect('equal', adjustable='box')
        
#         # SPH plots
#         if density_results['sph']:
#             sph_keys = list(density_results['sph'].keys())
#             sph_keys = [k for k in sph_keys if k != 'eval_grid'][:3]  # First 3
            
#             for idx, key in enumerate(sph_keys):
#                 if idx >= 3:
#                     break
#                 ax = axes[1, idx]
                
#                 sph_data = density_results['sph'][key]
#                 im = ax.contourf(X_eval, Y_eval, sph_data['density'], 
#                                levels=20, cmap='plasma')
#                 ax.scatter(final_positions[:, 0], final_positions[:, 1], 
#                           alpha=0.5, s=10, c='white', edgecolors='black', linewidths=0.5)
#                 ax.set_title(f"SPH {key} - h={sph_data['smoothing_length']:.3f}")
#                 ax.set_xlabel('X')
#                 ax.set_ylabel('Y')
#                 plt.colorbar(im, ax=ax)
#                 ax.set_aspect('equal', adjustable='box')
        
#         plt.tight_layout()
        
#         density_plot_file = output_path / "density_analysis.png"
#         plt.savefig(density_plot_file, dpi=150, bbox_inches='tight')
#         print(f"      ✅ Density plots saved: {density_plot_file}")
#         plt.close()
        
#     except Exception as e:
#         print(f"      ❌ Density visualization failed: {e}")
    
#     print(f"   🔬 Density estimation analysis completed!")
#     return density_results

# def perform_density_estimation_analysis(trajectory, output_dir="output"):
#     """Perform comprehensive density estimation using KDE and SPH - FIXED VERSION."""
#     print(f"\n🔬 Performing density estimation analysis...")
    
#     if not DENSITY_AVAILABLE:
#         print(f"   ⚠️  Density estimation modules not available - skipping")
#         return None
    
#     output_path = Path(output_dir)
#     output_path.mkdir(exist_ok=True)
    
#     # Get final particle positions for density analysis
#     final_positions = trajectory.positions[-1]  # (N, 3)
#     print(f"   📊 Analyzing density at final timestep with {final_positions.shape[0]} particles")
    
#     # FIXED: Calculate bounds once at the beginning (outside try blocks)
#     bounds_min = np.min(final_positions, axis=0)
#     bounds_max = np.max(final_positions, axis=0)
    
#     # Expand bounds slightly for evaluation grid
#     margin = 0.1 * (bounds_max - bounds_min)
#     bounds_min_expanded = bounds_min - margin
#     bounds_max_expanded = bounds_max + margin
    
#     # Create evaluation grid (common for both KDE and SPH)
#     n_grid = 50
#     x_eval = np.linspace(bounds_min_expanded[0], bounds_max_expanded[0], n_grid)
#     y_eval = np.linspace(bounds_min_expanded[1], bounds_max_expanded[1], n_grid)
#     X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
    
#     # For 2D density, use Z = mean(final_positions[:, 2])
#     z_mean = np.mean(final_positions[:, 2])
#     eval_points_2d = np.column_stack([
#         X_eval.ravel(), 
#         Y_eval.ravel(), 
#         np.full(X_eval.size, z_mean)
#     ])
    
#     density_results = {}
    
#     # 1. KDE Analysis - FIXED
#     print(f"   📈 Performing KDE analysis...")
    
#     try:
#         # FIXED: Create KDE estimator with positions in constructor
#         kde_scott = KDEEstimator(
#             positions=final_positions,
#             bandwidth_rule='scott'#,
#             # kernel='gaussian'
#         )
        
#         kde_silverman = KDEEstimator(
#             positions=final_positions,
#             bandwidth_rule='silverman'#, 
#             # kernel='gaussian'
#         )
        
#         print(f"      ✅ Scott bandwidth: {kde_scott.bandwidth}")
#         print(f"      ✅ Silverman bandwidth: {kde_silverman.bandwidth}")
        
#         # Evaluate densities
#         density_scott = kde_scott.evaluate(eval_points_2d).reshape(X_eval.shape)
#         density_silverman = kde_silverman.evaluate(eval_points_2d).reshape(X_eval.shape)
        
#         density_results['kde'] = {
#             'scott': {
#                 'estimator': kde_scott,
#                 'density': density_scott,
#                 'bandwidth': kde_scott.bandwidth
#             },
#             'silverman': {
#                 'estimator': kde_silverman,
#                 'density': density_silverman,
#                 'bandwidth': kde_silverman.bandwidth
#             },
#             'eval_grid': (X_eval, Y_eval),
#             'eval_points': eval_points_2d
#         }
        
#         print(f"      ✅ KDE analysis completed")
        
#     except Exception as e:
#         print(f"      ❌ KDE analysis failed: {e}")
#         print(f"      🔧 Trying alternative KDE approach...")
        
#         # Alternative KDE approach if the main one fails
#         try:
#             # Try without specifying kernel
#             kde_scott_alt = KDEEstimator(
#                 positions=final_positions,
#                 bandwidth_method='scott'
#             )
            
#             density_scott_alt = kde_scott_alt.evaluate(eval_points_2d).reshape(X_eval.shape)
            
#             density_results['kde'] = {
#                 'scott': {
#                     'estimator': kde_scott_alt,
#                     'density': density_scott_alt,
#                     'bandwidth': kde_scott_alt.bandwidth
#                 },
#                 'eval_grid': (X_eval, Y_eval),
#                 'eval_points': eval_points_2d
#             }
            
#             print(f"      ✅ Alternative KDE approach successful")
            
#         except Exception as e2:
#             print(f"      ❌ Alternative KDE approach also failed: {e2}")
#             density_results['kde'] = None
    
#     # 2. SPH Analysis - FIXED (bounds_max/bounds_min now available)
#     print(f"   🌊 Performing SPH density analysis...")
    
#     try:
#         # Create SPH estimator with different smoothing lengths
#         smoothing_lengths = [0.1, 0.2, 0.3]  # Relative to domain size
#         domain_size = np.mean(bounds_max - bounds_min)
        
#         sph_results = {}
        
#         for h_rel in smoothing_lengths:
#             h_abs = h_rel * domain_size
            
#             try:
#                 # FIXED: Try different SPH constructor approaches
#                 sph_estimator = SPHDensityEstimator(
#                     positions=final_positions,
#                     smoothing_length=h_abs,
#                     kernel_type='cubic_spline'
#                 )
                
#                 # Evaluate on same grid as KDE
#                 density_sph = sph_estimator.evaluate(eval_points_2d).reshape(X_eval.shape)
                
#                 sph_results[f'h_{h_rel:.1f}'] = {
#                     'estimator': sph_estimator,
#                     'density': density_sph,
#                     'smoothing_length': h_abs
#                 }
                
#                 print(f"      ✅ SPH h={h_abs:.3f} completed")
                
#             except Exception as sph_e:
#                 print(f"      ⚠️  SPH h={h_abs:.3f} failed: {sph_e}")
                
#                 # Try alternative SPH constructor
#                 try:
#                     sph_estimator_alt = SPHDensityEstimator(
#                         positions=final_positions,
#                         smoothing_length=h_abs,
#                         kernel_type='cubic_spline'
#                     )
                    
#                     # Fit with positions if fit method exists
#                     if hasattr(sph_estimator_alt, 'fit'):
#                         sph_estimator_alt.fit(final_positions)
                    
#                     density_sph_alt = sph_estimator_alt.evaluate(eval_points_2d).reshape(X_eval.shape)
                    
#                     sph_results[f'h_{h_rel:.1f}'] = {
#                         'estimator': sph_estimator_alt,
#                         'density': density_sph_alt,
#                         'smoothing_length': h_abs
#                     }
                    
#                     print(f"      ✅ SPH h={h_abs:.3f} completed (alternative method)")
                    
#                 except Exception as sph_e2:
#                     print(f"      ❌ SPH h={h_abs:.3f} failed completely: {sph_e2}")
#                     continue
        
#         if sph_results:
#             density_results['sph'] = sph_results
#             density_results['sph']['eval_grid'] = (X_eval, Y_eval)
#             print(f"      ✅ SPH analysis completed with {len(sph_results)} successful smoothing lengths")
#         else:
#             density_results['sph'] = None
#             print(f"      ❌ SPH analysis failed for all smoothing lengths")
        
#     except Exception as e:
#         print(f"      ❌ SPH analysis failed completely: {e}")
#         density_results['sph'] = None
    
#     # 3. Create density visualizations - ENHANCED ERROR HANDLING
#     print(f"   🎨 Creating density visualizations...")
    
#     try:
#         # Determine number of plots needed
#         n_kde_plots = 0
#         n_sph_plots = 0
        
#         if density_results.get('kde'):
#             n_kde_plots = len([k for k in density_results['kde'].keys() if k not in ['eval_grid', 'eval_points']])
        
#         if density_results.get('sph'):
#             n_sph_plots = len([k for k in density_results['sph'].keys() if k != 'eval_grid'])
        
#         total_plots = 1 + n_kde_plots + n_sph_plots  # 1 for particle scatter
        
#         # Create subplot grid
#         if total_plots <= 3:
#             fig, axes = plt.subplots(1, total_plots, figsize=(6*total_plots, 6))
#         else:
#             fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
#         # Ensure axes is always a list
#         if total_plots == 1:
#             axes = [axes]
#         elif total_plots <= 3:
#             axes = list(axes) if hasattr(axes, '__iter__') else [axes]
#         else:
#             axes = axes.flatten()
        
#         fig.suptitle('Density Estimation Analysis', fontsize=16)
        
#         plot_idx = 0
        
#         # Plot particle scatter
#         ax = axes[plot_idx]
#         scatter = ax.scatter(final_positions[:, 0], final_positions[:, 1], 
#                            alpha=0.6, s=20, c='blue')
#         ax.set_title('Final Particle Positions')
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.grid(True, alpha=0.3)
#         ax.set_aspect('equal', adjustable='box')
#         plot_idx += 1
        
#         # KDE plots
#         if density_results.get('kde'):
#             for method_name, kde_data in density_results['kde'].items():
#                 if method_name not in ['eval_grid', 'eval_points'] and plot_idx < len(axes):
#                     ax = axes[plot_idx]
#                     im = ax.contourf(X_eval, Y_eval, kde_data['density'], 
#                                    levels=20, cmap='viridis')
#                     ax.scatter(final_positions[:, 0], final_positions[:, 1], 
#                               alpha=0.5, s=10, c='white', edgecolors='black', linewidths=0.5)
#                     ax.set_title(f"KDE ({method_name}) - h={kde_data['bandwidth']:.3f}")
#                     ax.set_xlabel('X')
#                     ax.set_ylabel('Y')
#                     plt.colorbar(im, ax=ax)
#                     ax.set_aspect('equal', adjustable='box')
#                     plot_idx += 1
        
#         # SPH plots
#         if density_results.get('sph'):
#             for key, sph_data in density_results['sph'].items():
#                 if key != 'eval_grid' and plot_idx < len(axes):
#                     ax = axes[plot_idx]
#                     im = ax.contourf(X_eval, Y_eval, sph_data['density'], 
#                                    levels=20, cmap='plasma')
#                     ax.scatter(final_positions[:, 0], final_positions[:, 1], 
#                               alpha=0.5, s=10, c='white', edgecolors='black', linewidths=0.5)
#                     ax.set_title(f"SPH {key} - h={sph_data['smoothing_length']:.3f}")
#                     ax.set_xlabel('X')
#                     ax.set_ylabel('Y')
#                     plt.colorbar(im, ax=ax)
#                     ax.set_aspect('equal', adjustable='box')
#                     plot_idx += 1
        
#         # Hide unused subplots
#         for i in range(plot_idx, len(axes)):
#             axes[i].set_visible(False)
        
#         plt.tight_layout()
        
#         density_plot_file = output_path / "density_analysis.png"
#         plt.savefig(density_plot_file, dpi=150, bbox_inches='tight')
#         print(f"      ✅ Density plots saved: {density_plot_file}")
#         plt.close()
        
#     except Exception as e:
#         print(f"      ❌ Density visualization failed: {e}")
#         import traceback
#         traceback.print_exc()
    
#     # 4. Quantitative density comparison - ENHANCED
#     print(f"   📊 Quantitative density analysis...")
    
#     try:
#         # Compute density statistics at particle positions
#         particle_densities = {}
        
#         if density_results.get('kde'):
#             for method_name, kde_data in density_results['kde'].items():
#                 if method_name not in ['eval_grid', 'eval_points']:
#                     try:
#                         densities_at_particles = kde_data['estimator'].evaluate(final_positions)
#                         particle_densities[f'kde_{method_name}'] = densities_at_particles
#                     except Exception as e:
#                         print(f"      ⚠️  Failed to evaluate KDE {method_name} at particles: {e}")
        
#         if density_results.get('sph'):
#             for key, sph_data in density_results['sph'].items():
#                 if key != 'eval_grid':
#                     try:
#                         densities_at_particles = sph_data['estimator'].evaluate(final_positions)
#                         particle_densities[f'sph_{key}'] = densities_at_particles
#                     except Exception as e:
#                         print(f"      ⚠️  Failed to evaluate SPH {key} at particles: {e}")
        
#         # Create density comparison plots if we have data
#         if particle_densities:
#             fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
#             # Histogram comparison
#             ax = axes[0]
#             for name, densities in particle_densities.items():
#                 ax.hist(densities, bins=30, alpha=0.6, label=name, density=True)
            
#             ax.set_title('Density Distribution at Particle Positions')
#             ax.set_xlabel('Density')
#             ax.set_ylabel('Probability Density')
#             ax.legend()
#             ax.grid(True, alpha=0.3)
            
#             # Statistical comparison
#             ax = axes[1]
#             methods = list(particle_densities.keys())
#             means = [np.mean(particle_densities[m]) for m in methods]
#             stds = [np.std(particle_densities[m]) for m in methods]
            
#             x_pos = range(len(methods))
#             ax.bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5)
#             ax.set_xticks(x_pos)
#             ax.set_xticklabels([m.replace('_', '\n') for m in methods], rotation=45)
#             ax.set_title('Mean Density ± Std Dev')
#             ax.set_ylabel('Density')
#             ax.grid(True, alpha=0.3)
            
#             plt.tight_layout()
            
#             density_stats_file = output_path / "density_statistics.png"
#             plt.savefig(density_stats_file, dpi=150, bbox_inches='tight')
#             print(f"      ✅ Density statistics saved: {density_stats_file}")
#             plt.close()
            
#             # Print quantitative results
#             print(f"   📈 Density Statistics Summary:")
#             for name, densities in particle_densities.items():
#                 print(f"      {name}: mean={np.mean(densities):.4f}, std={np.std(densities):.4f}")
#         else:
#             print(f"      ⚠️  No density data available for quantitative analysis")
        
#     except Exception as e:
#         print(f"      ❌ Quantitative analysis failed: {e}")
#         import traceback
#         traceback.print_exc()
    
#     print(f"   🔬 Density estimation analysis completed!")
#     return density_results

def perform_density_estimation_analysis(trajectory, output_dir="output"):
    """
    Perform density estimation using KDE and SPH, robust to different estimator APIs.

    - Supports KDEEstimator constructed either as KDEEstimator(positions=..., ...) or
      instantiated then .fit(positions).
    - Same for SPHDensityEstimator.
    - Precomputes bounds and evaluation grid before running estimators.
    """
    print("\n🔬 Performing density estimation analysis...")

    try:
        from jaxtrace.density import KDEEstimator, SPHDensityEstimator
        DENSITY_AVAILABLE = True
    except Exception as e:
        print(f"   ⚠️  Density modules not available: {e} — skipping.")
        return None

    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Final particle positions
    final_positions = trajectory.positions[-1]  # (N, 3)
    if final_positions.ndim != 2 or final_positions.shape[1] != 3:
        raise ValueError(f"Expected final_positions to be (N,3), got {final_positions.shape}")

    print(f"   📊 Analyzing density at final timestep with {final_positions.shape[0]} particles")

    # Helpers to tolerate multiple estimator APIs
    def make_kde(positions, bandwidth_method, kernel):
        # Try constructor with positions=
        try:
            est = KDEEstimator(positions=positions, bandwidth_rule=bandwidth_method)#, kernel=kernel
            return est
        except TypeError:
            # Try constructor without positions then fit()
            est = KDEEstimator(bandwidth_rule=bandwidth_method)#, kernel=kernel
            if hasattr(est, "fit"):
                est.fit(positions)
                return est
            raise

    def make_sph(positions, smoothing_length, kernel):
        # Try constructor with positions=
        try:
            est = SPHDensityEstimator(positions=positions, smoothing_length=smoothing_length, kernel_type=kernel)
            return est
        except TypeError:
            # Try constructor without positions then fit()
            est = SPHDensityEstimator(smoothing_length=smoothing_length, kernel_type=kernel)
            if hasattr(est, "fit"):
                est.fit(positions)
                return est
            raise

    # def eval_density(estimator, pts):
    #     # Support both .evaluate() and .pdf()
    #     if hasattr(estimator, "evaluate_3d"):
    #         return estimator.evaluate_3d()#(pts)
    #     if hasattr(estimator, "evaluate"):
    #         return estimator.evaluate(pts)
    #     raise AttributeError("Estimator has neither .evaluate() nor .pdf()")
    def eval_density(estimator, pts):
        """
        Return densities at the given points (M,d). Prefer pointwise API.

        - Use estimator.evaluate(pts) if available.
        - Use estimator.pdf(pts) if available.
        - As a last resort, if a grid-evaluator exists (evaluate_2d/evaluate_3d), return
        the grid densities flattened (caller must NOT reshape to a different grid).
        """
        if hasattr(estimator, "evaluate"):
            return estimator.evaluate(pts)
        if hasattr(estimator, "pdf"):
            return estimator.pdf(pts)
        # Fallbacks (not preferred; shape may differ from the caller's grid)
        if hasattr(estimator, "evaluate_2d") and pts.shape[1] == 2:
            X, Y, Z = estimator.evaluate_2d()
            return Z.ravel()  # NOTE: caller should not reshape unless matching X,Y
        if hasattr(estimator, "evaluate_3d") and pts.shape[1] == 3:
            X, Y, Z, D = estimator.evaluate_3d()
            return D.ravel()
        raise AttributeError("Estimator has neither .evaluate/.pdf nor grid evaluators")
    # Precompute bounds and evaluation grid (so SPH doesn't depend on KDE success)
    bounds_min = np.min(final_positions, axis=0)
    bounds_max = np.max(final_positions, axis=0)

    # Slightly expand bounds
    margin = 0.1 * np.maximum(bounds_max - bounds_min, 1e-12)
    bmin = bounds_min - margin
    bmax = bounds_max + margin

    # 2D evaluation grid in the Z = mean(Z) plane
    n_grid = 80  # a bit denser for smoother contours
    x_eval = np.linspace(bmin[0], bmax[0], n_grid)
    y_eval = np.linspace(bmin[1], bmax[1], n_grid)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval)  # (ny, nx)
    z_mean = float(np.mean(final_positions[:, 2]))
    eval_points_2d = np.column_stack([X_eval.ravel(), Y_eval.ravel(), np.full(X_eval.size, z_mean)])

    density_results = {}

    # 1) KDE
    print("   📈 Performing KDE analysis...")
    try:
        kde_scott = make_kde(final_positions, bandwidth_method="scott", kernel="gaussian")
        kde_silver = make_kde(final_positions, bandwidth_method="silverman", kernel="gaussian")

        dens_scott = eval_density(kde_scott, eval_points_2d).reshape(X_eval.shape)
        dens_silver = eval_density(kde_silver, eval_points_2d).reshape(X_eval.shape)

        # Try to fetch bandwidth attribute (varies by implementation)
        def get_bw(est):
            for attr in ("bandwidth", "h", "bw"):
                if hasattr(est, attr):
                    val = getattr(est, attr)
                    try:
                        return float(np.asarray(val).mean())
                    except Exception:
                        return val
            return None

        bw_scott = get_bw(kde_scott)
        bw_silver = get_bw(kde_silver)

        density_results["kde"] = {
            "scott": {"estimator": kde_scott, "density": dens_scott, "bandwidth": bw_scott},
            "silverman": {"estimator": kde_silver, "density": dens_silver, "bandwidth": bw_silver},
            "eval_grid": (X_eval, Y_eval),
            "eval_points": eval_points_2d,
        }
        print(f"      ✅ KDE completed (Scott bw={bw_scott}, Silverman bw={bw_silver})")
    except Exception as e:
        print(f"      ❌ KDE analysis failed: {e}")
        density_results["kde"] = None

    # 2) SPH
    print("   🌊 Performing SPH density analysis...")
    try:
        # Relative smoothing lengths based on average span
        span = np.maximum(bounds_max - bounds_min, 1e-12)
        domain_size = float(np.mean(span))
        smoothing_rels = [0.10, 0.20, 0.30]
        sph_results = {}

        for h_rel in smoothing_rels:
            h_abs = max(h_rel * domain_size, 1e-9)
            sph_est = make_sph(final_positions, smoothing_length=h_abs, kernel="cubic_spline")
            # dens_sph = eval_density(sph_est, eval_points_2d).reshape(X_eval.shape)
            # SPH in 2D mode expects (M,2) evaluation points
            eval_points_xy = np.column_stack([X_eval.ravel(), Y_eval.ravel()])
            dens_sph = eval_density(sph_est, eval_points_xy).reshape(X_eval.shape)

            key = f"h_{h_rel:.2f}"
            sph_results[key] = {
                "estimator": sph_est,
                "density": dens_sph,
                "smoothing_length": h_abs,
            }
            print(f"      ✅ SPH h={h_abs:.4g} completed")

        sph_results["eval_grid"] = (X_eval, Y_eval)
        density_results["sph"] = sph_results
        print("      ✅ SPH analysis completed")
    except Exception as e:
        print(f"      ❌ SPH analysis failed: {e}")
        density_results["sph"] = None

    # 3) Visualization (robust to partial results)
    print("   🎨 Creating density visualizations...")
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Density Estimation Analysis", fontsize=16)

        # Base scatter
        ax = axes[0, 0]
        ax.scatter(final_positions[:, 0], final_positions[:, 1], s=14, alpha=0.6, c="tab:blue")
        ax.set_title("Final Particle Positions")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        # KDE plots if available
        if density_results.get("kde"):
            Xp, Yp = density_results["kde"]["eval_grid"]
            # Scott
            ax = axes[0, 1]
            im = ax.contourf(Xp, Yp, density_results["kde"]["scott"]["density"], levels=20, cmap="viridis")
            # ax.scatter(final_positions[:, 0], final_positions[:, 1], c="w", s=8, alpha=0.6, edgecolors="k", linewidths=0.3)
            bw_s = density_results["kde"]["scott"]["bandwidth"]
            ax.set_title(f"KDE (Scott){'' if bw_s is None else f' - bw={bw_s:.3g}'}")
            ax.set_aspect("equal", adjustable="box")
            plt.colorbar(im, ax=ax)

            # Silverman
            ax = axes[0, 2]
            im = ax.contourf(Xp, Yp, density_results["kde"]["silverman"]["density"], levels=20, cmap="viridis")
            # ax.scatter(final_positions[:, 0], final_positions[:, 1], c="w", s=8, alpha=0.6, edgecolors="k", linewidths=0.3)
            bw_si = density_results["kde"]["silverman"]["bandwidth"]
            ax.set_title(f"KDE (Silverman){'' if bw_si is None else f' - bw={bw_si:.3g}'}")
            ax.set_aspect("equal", adjustable="box")
            plt.colorbar(im, ax=ax)
        else:
            axes[0, 1].axis("off")
            axes[0, 2].axis("off")

        # SPH plots (first up to 3)
        if density_results.get("sph"):
            Xp, Yp = density_results["sph"]["eval_grid"]
            sph_keys = [k for k in density_results["sph"].keys() if k != "eval_grid"][:3]
            for idx, key in enumerate(sph_keys):
                ax = axes[1, idx]
                im = ax.contourf(Xp, Yp, density_results["sph"][key]["density"], levels=20, cmap="plasma")
                # ax.scatter(final_positions[:, 0], final_positions[:, 1], c="w", s=8, alpha=0.6, edgecolors="k", linewidths=0.3)
                ax.set_title(f"SPH {key} - h={density_results['sph'][key]['smoothing_length']:.3g}")
                ax.set_aspect("equal", adjustable="box")
                plt.colorbar(im, ax=ax)
            # Hide unused axes if less than 3 SPH maps
            for j in range(len(sph_keys), 3):
                axes[1, j].axis("off")
        else:
            for j in range(3):
                axes[1, j].axis("off")

        plt.tight_layout()
        density_plot_file = output_path / "density_analysis.png"
        plt.savefig(density_plot_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"      ✅ Density plots saved: {density_plot_file}")
    except Exception as e:
        print(f"      ❌ Density visualization failed: {e}")

    # 4) Quantitative comparison at particle locations (histograms + mean/std bars)
    print("   📊 Quantitative density comparison...")
    try:
        particle_densities = {}
        if density_results.get("kde"):
            est_s = density_results["kde"]["scott"]["estimator"]
            est_si = density_results["kde"]["silverman"]["estimator"]
            particle_densities["kde_scott"] = eval_density(est_s, final_positions)
            particle_densities["kde_silverman"] = eval_density(est_si, final_positions)
        if density_results.get("sph"):
            for k, v in density_results["sph"].items():
                if k == "eval_grid":
                    continue
                particle_densities[f"sph_{k}"] = eval_density(v["estimator"], final_positions)

        if particle_densities:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Histogram comparison
            ax = axes[0]
            for name, vals in particle_densities.items():
                ax.hist(vals, bins=30, alpha=0.60, density=True, label=name)
            ax.set_title("Density Distribution at Particle Positions")
            ax.set_xlabel("Density")
            ax.set_ylabel("Probability Density")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Mean ± std bar
            ax = axes[1]
            methods = list(particle_densities.keys())
            means = [float(np.mean(particle_densities[m])) for m in methods]
            stds = [float(np.std(particle_densities[m])) for m in methods]
            x = np.arange(len(methods))
            ax.bar(x, means, yerr=stds, alpha=0.75, capsize=6)
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace("_", "\n") for m in methods], rotation=0)
            ax.set_title("Mean Density ± Std Dev")
            ax.set_ylabel("Density")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            stats_file = output_path / "density_statistics.png"
            plt.savefig(stats_file, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"      ✅ Density statistics saved: {stats_file}")

            # Print statistics to console
            print("   📈 Density Statistics Summary:")
            for name, vals in particle_densities.items():
                print(f"      {name}: mean={np.mean(vals):.4g}, std={np.std(vals):.4g}, "
                      f"min={np.min(vals):.4g}, max={np.max(vals):.4g}")
        else:
            print("      ⚠️ No densities computed at particle positions.")

    except Exception as e:
        print(f"      ❌ Quantitative comparison failed: {e}")

    print("   🔬 Density estimation analysis completed!")
    return density_results

def create_enhanced_visualizations(trajectory, density_results=None, output_dir="output"):
    """NEW: Create enhanced visualizations using the visualization module."""
    print(f"\n🎨 Creating enhanced visualizations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if not VISUALIZATION_AVAILABLE:
        print(f"   ⚠️  Enhanced visualization modules not available - using basic plots")
        create_trajectory_visualizations(trajectory, None, None, output_dir)
        return
    
    try:
        # 1. Static 2D trajectory plot using enhanced visualization
        print(f"   📊 Creating enhanced 2D trajectory visualization...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        plot_trajectory_2d(
            trajectory,
            ax=ax,
            color_by='time',
            show_start_end=True,
            alpha=0.7,
            linewidth=1.0
        )
        
        ax.set_title(f'Enhanced 2D Trajectory Visualization\n{trajectory.N} particles, {trajectory.T} timesteps')
        
        trajectory_2d_file = output_path / "enhanced_trajectory_2d.png"
        plt.savefig(trajectory_2d_file, dpi=150, bbox_inches='tight')
        print(f"      ✅ Saved: {trajectory_2d_file}")
        plt.close()
        
        # 2. 3D trajectory plot (if meaningful Z variation)
        z_variation = np.std(trajectory.positions[:, :, 2])
        if z_variation > 1e-6:
            print(f"   📊 Creating 3D trajectory visualization...")
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            plot_trajectory_3d(
                trajectory,
                ax=ax,
                color_by='speed',
                show_start_end=True,
                alpha=0.6,
                linewidth=0.8
            )
            
            ax.set_title(f'Enhanced 3D Trajectory Visualization')
            
            trajectory_3d_file = output_path / "enhanced_trajectory_3d.png"
            plt.savefig(trajectory_3d_file, dpi=150, bbox_inches='tight')
            print(f"      ✅ Saved: {trajectory_3d_file}")
            plt.close()
        
        # 3. Final particle distribution with density coloring
        print(f"   📊 Creating enhanced final particle distribution...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        final_positions = trajectory.positions[-1]
        
        if density_results and density_results.get('kde'):
            # Use density-based coloring
            plot_particles_2d(
                final_positions,
                ax=ax,
                color_by_density=True,
                density_estimator=density_results['kde']['scott']['estimator'],
                alpha=0.8,
                marker_size=50
            )
        else:
            # Basic scatter plot
            plot_particles_2d(
                final_positions,
                ax=ax,
                color_by_density=False,
                alpha=0.8,
                marker_size=50
            )
        
        ax.set_title('Enhanced Final Particle Distribution')
        
        final_dist_file = output_path / "enhanced_final_distribution.png"
        plt.savefig(final_dist_file, dpi=150, bbox_inches='tight')
        print(f"      ✅ Saved: {final_dist_file}")
        plt.close()
        
        # 4. Animation frames for 2D trajectory evolution
        print(f"   🎬 Creating enhanced animation frames...")
        
        animation_dir = output_path / "enhanced_animation_frames"
        animation_dir.mkdir(exist_ok=True)
        
        # Create frames at selected timesteps
        n_frames = min(20, trajectory.T)  # Max 20 frames
        frame_indices = np.linspace(0, trajectory.T - 1, n_frames, dtype=int)
        
        frames = render_frames_2d(
            trajectory,
            frame_indices=frame_indices,
            figsize=(10, 8),
            show_trails=True,
            trail_length=10,
            color_by='time',
            alpha=0.7
        )
        
        # Save individual frames
        frame_files = []
        for i, (frame_idx, frame_fig) in enumerate(zip(frame_indices, frames)):
            frame_file = animation_dir / f"frame_{i:03d}_t{frame_idx:03d}.png"
            frame_fig.savefig(frame_file, dpi=100, bbox_inches='tight')
            frame_files.append(frame_file)
            plt.close(frame_fig)
        
        print(f"      ✅ Created {len(frames)} enhanced animation frames")
        
        # 5. Create GIF animation
        try:
            gif_file = output_path / "enhanced_trajectory_animation.gif"
            save_gif_from_frames(
                frame_files,
                output_path=gif_file,
                duration=200,  # ms per frame
                loop=0
            )
            print(f"      ✅ Created enhanced GIF animation: {gif_file}")
            
        except Exception as e:
            print(f"      ⚠️  Enhanced GIF creation failed: {e}")
        
        # 6. Interactive Plotly visualization (if available)
        if PLOTLY_AVAILABLE:
            print(f"   🌐 Creating interactive Plotly visualizations...")
            
            try:
                # Create interactive 2D plot
                fig_plotly = animate_trajectory_plotly_2d(
                    trajectory,
                    color_by='time',
                    show_density_background=True if density_results and density_results.get('kde') else False,
                    density_data=(
                        density_results['kde']['eval_grid'],
                        density_results['kde']['scott']['density']
                    ) if density_results and density_results.get('kde') else None
                )
                
                plotly_file = output_path / "interactive_trajectory.html"
                fig_plotly.write_html(str(plotly_file))
                print(f"      ✅ Interactive plot saved: {plotly_file}")
                
                # Create 3D interactive if meaningful
                if z_variation > 1e-6:
                    fig_plotly_3d = animate_trajectory_plotly_3d(
                        trajectory,
                        color_by='speed'
                    )
                    
                    plotly_3d_file = output_path / "interactive_trajectory_3d.html"
                    fig_plotly_3d.write_html(str(plotly_3d_file))
                    print(f"      ✅ Interactive 3D plot saved: {plotly_3d_file}")
                
            except Exception as e:
                print(f"      ⚠️  Interactive Plotly visualization failed: {e}")
        
        print(f"   🎨 Enhanced visualizations completed!")
        
    except Exception as e:
        print(f"   ❌ Enhanced visualization creation failed: {e}")
        import traceback
        traceback.print_exc()


def create_trajectory_visualizations(trajectory, stats, field, output_dir="output"):
    """Create comprehensive trajectory visualizations (ORIGINAL + ENHANCED)."""
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
        
        times = np.arange(trajectory.T)  # Use timestep indices if no times available
        if hasattr(trajectory, 'times') and trajectory.times is not None:
            times = trajectory.times
        
        ax3.plot(times, spreads_x, label='X spread', linewidth=2)
        ax3.plot(times, spreads_y, label='Y spread', linewidth=2)
        ax3.plot(times, spreads_z, label='Z spread', linewidth=2)
        
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
            
            ax4.plot(times, mean_speeds, 'b-', linewidth=2, label='Mean speed')
            ax4.fill_between(times, mean_speeds - std_speeds, mean_speeds + std_speeds,
                           alpha=0.3, color='blue', label='±1σ')
            ax4.set_title('Speed Evolution')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Speed')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            # Approximate speed from position differences
            dt = 1.0  # Default time step
            if hasattr(trajectory, 'times') and trajectory.times is not None and len(trajectory.times) > 1:
                dt = trajectory.times[1] - trajectory.times[0]
            
            speed_approx = np.zeros((trajectory.T - 1, trajectory.N))
            
            for t in range(trajectory.T - 1):
                displacements = trajectory.positions[t+1] - trajectory.positions[t]  # (N, 3)
                speed_approx[t] = np.linalg.norm(displacements, axis=1) / dt
            
            mean_speeds = np.mean(speed_approx, axis=1)
            times_speed = times[:-1]
            
            ax4.plot(times_speed, mean_speeds, 'b-', linewidth=2, label='Mean speed (approx)')
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
        z_variation = np.std(trajectory.positions[:, :, 2])
        if z_variation > 1e-6:  # If there's Z variation
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


def export_enhanced_results_to_vtk(trajectory, density_results=None, field_info=None, output_dir="output"):
    """NEW: Export results using enhanced I/O capabilities."""
    print(f"\n💾 Exporting results using enhanced I/O...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        # 1. Standard VTK export (always available)
        # print(f"   📤 Exporting trajectory to standard VTK...")
        
        # trajectory_file = output_path / "enhanced_trajectory.vtp"
        
        # export_trajectory_to_vtk(
        #     trajectory=trajectory,
        #     filename=str(trajectory_file),
        #     include_velocities=True,
        #     include_metadata=True,
        #     time_series=True
        # )
        
        # print(f"      ✅ Standard trajectory exported: {trajectory_file}")
        # print(f"         File size: {trajectory_file.stat().st_size / 1024:.1f} KB")
        # 1. Standard VTK export (time-series)
        print(f"   📤 Exporting trajectory to standard VTK...")

        trajectory_base = output_path / "enhanced_trajectory.vtp"
        result = export_trajectory_to_vtk(
            trajectory=trajectory,
            filename=str(trajectory_base),
            include_velocities=True,
            include_metadata=True,
            time_series=True
        )        
        if result['mode'] == 'series':
            print(f"      ✅ Standard trajectory exported as time series: {result['directory']}")
            print(f"         Files written: {result['count']}")

        # In time_series=True mode, files live in <stem>_series/
        series_dir = output_path / f"{trajectory_base.stem}_series"
        print(f"      ✅ Standard trajectory exported as time series: {series_dir}")

        # Optional: count files and show a quick summary
        try:
            n_series_files = len(list(series_dir.glob("*.vt*")))
            print(f"         Files written: {n_series_files}")
        except Exception as _:
            pass
        
        # 2. Enhanced VTK export (if available)
        if ENHANCED_WRITERS_AVAILABLE:
            print(f"   📤 Using enhanced VTK writers...")
            
            trajectory_writer = VTKTrajectoryWriter()
            field_writer = VTKFieldWriter()
            
            # Enhanced trajectory export
            enhanced_trajectory_file = output_path / "enhanced_trajectory_full.vtu"
            trajectory_writer.write_trajectory(
                trajectory,
                filename=str(enhanced_trajectory_file),
                include_velocities=True,
                include_particle_ids=True,
                include_timestep_data=True,
                format='xml'
            )
            
            print(f"      ✅ Enhanced trajectory: {enhanced_trajectory_file}")
            
            # Time series trajectory (multiple files)
            trajectory_series_dir = output_path / "trajectory_series"
            trajectory_series_dir.mkdir(exist_ok=True)
            
            trajectory_writer.write_time_series(
                trajectory,
                output_directory=str(trajectory_series_dir),
                filename_pattern="trajectory_{:04d}.vtu",
                include_velocities=True,
                format='xml'
            )
            
            series_files = list(trajectory_series_dir.glob("*.vtu"))
            print(f"      ✅ Time series: {len(series_files)} files in {trajectory_series_dir}")
            
            # 3. Export density fields (if available)
            if density_results and (density_results.get('kde') or density_results.get('sph')):
                print(f"   📤 Exporting density fields to VTK...")
                
                # KDE densities
                if density_results.get('kde'):
                    X_eval, Y_eval = density_results['kde']['eval_grid']
                    
                    # KDE Scott
                    density_scott_file = output_path / "density_kde_scott.vti"
                    field_writer.write_structured_2d(
                        X_eval, Y_eval,
                        density_results['kde']['scott']['density'],
                        filename=str(density_scott_file),
                        field_name="density",
                        format='xml'
                    )
                    print(f"      ✅ KDE Scott density: {density_scott_file}")
                    
                    # KDE Silverman
                    density_silverman_file = output_path / "density_kde_silverman.vti"
                    field_writer.write_structured_2d(
                        X_eval, Y_eval,
                        density_results['kde']['silverman']['density'],
                        filename=str(density_silverman_file),
                        field_name="density",
                        format='xml'
                    )
                    print(f"      ✅ KDE Silverman density: {density_silverman_file}")
                
                # SPH densities
                if density_results.get('sph'):
                    X_eval, Y_eval = density_results['sph']['eval_grid']
                    
                    for key, sph_data in density_results['sph'].items():
                        if key != 'eval_grid':
                            sph_file = output_path / f"density_{key}.vti"
                            field_writer.write_structured_2d(
                                X_eval, Y_eval,
                                sph_data['density'],
                                filename=str(sph_file),
                                field_name="density",
                                format='xml'
                            )
                            print(f"      ✅ SPH density {key}: {sph_file}")
        
        # 4. Export metadata
        metadata_file = output_path / "enhanced_analysis_metadata.json"
        
        metadata = {
            'trajectory_info': {
                'n_particles': trajectory.N,
                'n_timesteps': trajectory.T,
                'duration': trajectory.duration if hasattr(trajectory, 'duration') else 'N/A',
                'memory_mb': trajectory.memory_usage_mb()
            },
            'field_info': {
                'data_shape': field_info.data.shape if field_info else 'N/A',
                'time_steps': field_info.T if field_info else 'N/A',
                'grid_points': field_info.N if field_info else 'N/A',
                'memory_mb': field_info.memory_usage_mb() if field_info else 'N/A'
            },
            'density_info': {
                'kde_available': density_results is not None and density_results.get('kde') is not None,
                'sph_available': density_results is not None and density_results.get('sph') is not None,
                'enhanced_writers_used': ENHANCED_WRITERS_AVAILABLE
            },
            'export_files': {
                # 'standard_trajectory': str(trajectory_file.name),
                'enhanced_features_used': ENHANCED_WRITERS_AVAILABLE,
                'density_fields': len([f for f in output_path.glob("density_*.vti")])
            },
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"      ✅ Metadata exported: {metadata_file}")
        print(f"   💾 Enhanced export completed!")
        return metadata
        
    except Exception as e:
        print(f"   ❌ Enhanced export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_results_to_vtk(trajectory, field_info, output_dir="output"):
    """Export results to VTK format for advanced visualization (ORIGINAL)."""
    # print(f"\n💾 Exporting results to VTK format...")
    
    # try:
    #     output_path = Path(output_dir)
    #     output_path.mkdir(exist_ok=True)
        
    #     # Export trajectory
    #     vtk_file = output_path / "particle_trajectories.vtp"
        
    #     export_trajectory_to_vtk(
    #         trajectory=trajectory,
    #         filename=str(vtk_file),
    #         include_velocities=True,
    #         include_metadata=True,
    #         time_series=True
    #     )
        
    #     print(f"   ✅ Trajectory exported: {vtk_file}")
    #     print(f"   📊 File contains:")
    #     print(f"      - {trajectory.N} particle paths")
    #     print(f"      - {trajectory.T} time steps") 
    #     print(f"      - Velocity data: {'Yes' if trajectory.velocities is not None else 'No'}")
    #     print(f"      - File size: {vtk_file.stat().st_size / 1024:.1f} KB")
        
    # except ImportError:
    #     print(f"   ⚠️  VTK export not available - skipping")
    # except Exception as e:
    #     print(f"   ❌ VTK export failed: {e}")
    print(f"\n💾 Exporting results to VTK format...")

    try:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Export trajectory (time series)
        vtk_base = output_path / "particle_trajectories.vtp"
        export_trajectory_to_vtk(
            trajectory=trajectory,
            filename=str(vtk_base),
            include_velocities=True,
            include_metadata=True,
            time_series=True
        )

        series_dir = output_path / f"{vtk_base.stem}_series"
        print(f"   ✅ Trajectory exported as time series: {series_dir}")
        n_series_files = len(list(series_dir.glob("*.vt*")))
        print(f"   📊 Files written: {n_series_files}")
        print(f"      - {trajectory.N} particle paths")
        print(f"      - {trajectory.T} time steps")
        print(f"      - Velocity data: {'Yes' if trajectory.velocities is not None else 'No'}")

    except ImportError:
        print(f"   ⚠️  VTK export not available - skipping")
    except Exception as e:
        print(f"   ❌ VTK export failed: {e}")

def generate_summary_report(field, trajectory, stats, strategy, output_dir="output"):
    """Generate comprehensive summary report (ORIGINAL)."""
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
        f.write(f"Duration: {getattr(trajectory, 'duration', 'N/A')}\n")
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


def generate_enhanced_summary_report(field, trajectory, stats, strategy, density_results=None, metadata=None, output_dir="output"):
    """NEW: Generate comprehensive summary report with density analysis."""
    print(f"\n📋 Generating enhanced summary report...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    report_file = output_path / "enhanced_analysis_summary.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("JAXTrace Enhanced Particle Tracking Analysis Summary\n")
        f.write("With Density Estimation and Advanced Visualization\n")
        f.write("="*80 + "\n\n")
        
        # System capabilities
        f.write("SYSTEM CAPABILITIES\n")
        f.write("-"*40 + "\n")
        f.write(f"Density estimation: {'Available' if DENSITY_AVAILABLE else 'Not available'}\n")
        f.write(f"Enhanced visualization: {'Available' if VISUALIZATION_AVAILABLE else 'Not available'}\n")
        f.write(f"Enhanced I/O writers: {'Available' if ENHANCED_WRITERS_AVAILABLE else 'Not available'}\n")
        f.write(f"Interactive plots: {'Available' if PLOTLY_AVAILABLE else 'Not available'}\n\n")
        
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
        f.write(f"Duration: {getattr(trajectory, 'duration', 'N/A')}\n")
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
            f.write("\n")
        
        # Density analysis
        f.write("DENSITY ESTIMATION ANALYSIS\n")
        f.write("-"*40 + "\n")
        
        if density_results and density_results.get('kde'):
            f.write("KDE Analysis:\n")
            kde_scott_bw = density_results['kde']['scott']['bandwidth']
            kde_silverman_bw = density_results['kde']['silverman']['bandwidth']
            f.write(f"  Scott bandwidth: {kde_scott_bw:.6f}\n")
            f.write(f"  Silverman bandwidth: {kde_silverman_bw:.6f}\n")
        else:
            f.write("KDE Analysis: Not performed\n")
        
        if density_results and density_results.get('sph'):
            f.write("SPH Analysis:\n")
            for key, sph_data in density_results['sph'].items():
                if key != 'eval_grid':
                    f.write(f"  {key}: smoothing length = {sph_data['smoothing_length']:.6f}\n")
        else:
            f.write("SPH Analysis: Not performed\n")
        
        f.write("\n")
        
        # Export summary
        if metadata:
            f.write("EXPORT SUMMARY\n")
            f.write("-"*40 + "\n")
            export_info = metadata.get('export_files', {})
            f.write(f"Standard trajectory file: {export_info.get('standard_trajectory', 'N/A')}\n")
            f.write(f"Enhanced features used: {export_info.get('enhanced_features_used', False)}\n")
            f.write(f"Density field files: {export_info.get('density_fields', 0)}\n\n")
        
        f.write(f"Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")
    
    print(f"   ✅ Enhanced summary report saved: {report_file}")


def main():
    """Main execution function with enhanced features."""
    # Configuration
    # config = {
    #     'data_directory': '/home/arhashemi/Workspace/welding/Cases/002_caseCoarse.gid/post/0eule/',
    #     'case_name': 'caseCoarse',
    #     'n_particles': 10000,  # Reduced for performance with enhanced features
    #     'output_directory': 'jaxtrace_enhanced_output'
    # }
    # Configuration with periodic field options
    config = {
        'data_directory': '/home/arhashemi/Workspace/welding/Cases/002_caseCoarse.gid/post/0eule/',
        'case_name': 'caseCoarse',
        'n_particles': 10000,
        'output_directory': 'output_jaxtrace_enhanced',
        'dt': 0.0025,  # Time step for integration
        
        # NEW: Time periodicity options
        'use_time_periodicity': True,  # Enable periodic field
        'periodic_config': {
            'time_slice': None, #[118, 159],           # Use full time range (or specify (start, end))
            'n_periods': None,               # Simulate 8 periods
            'target_duration': 4000,      # Or specify target duration instead
            'transition_smoothing': 0.15  # 15% smoothing at period boundaries
        }
    }
    
    print("🚀 Starting JAXTrace Enhanced Workflow")
    print("With Density Estimation, Advanced Visualization, and Enhanced I/O")
    print(f"Configuration: {config}")
    
    try:
        # Step 1: Check enhanced requirements
        if not check_system_requirements():
            print("⚠️  Some enhanced features may not work, but continuing with basic functionality...")
        
        # Step 2: Load VTK data
        field = validate_and_load_vtk_data(
            config['data_directory'], 
            config['case_name']
        )
        
        # Step 3: Execute tracking
        # trajectory, strategy = execute_particle_tracking_analysis(
        #     field, 
        #     config['n_particles']
        # )
        trajectory, strategy = execute_particle_tracking_analysis(
            field, 
            config['n_particles'],
            use_time_periodicity=config['use_time_periodicity'],
            periodic_config=config.get('periodic_config'),
            dt=config.get('dt', 0.1)
        )
        
        # Step 4: Analyze results
        stats, validation = analyze_trajectory_results(trajectory, strategy)
        
        # Step 5: ENHANCED - Density estimation
        density_results = perform_density_estimation_analysis(
            trajectory, 
            config['output_directory']
        )
        
        # # Step 6: ENHANCED - Advanced visualizations
        # create_enhanced_visualizations(
        #     trajectory, 
        #     density_results, 
        #     config['output_directory']
        # )
        
        # # Step 7: Original visualizations (as backup/comparison)
        # create_trajectory_visualizations(
        #     trajectory, stats, field, config['output_directory']
        # )
        
        # Step 8: ENHANCED - Export with advanced I/O
        metadata = export_enhanced_results_to_vtk(
            trajectory, 
            density_results, 
            field, 
            config['output_directory']
        )
        
        # Step 9: Original VTK export (as backup)
        export_results_to_vtk(trajectory, field, config['output_directory'])
        
        # Step 10: Enhanced summary report
        generate_enhanced_summary_report(
            field, trajectory, stats, strategy, 
            density_results, metadata, 
            config['output_directory']
        )
        
        # Step 11: Original summary report (as backup)
        generate_summary_report(
            field, trajectory, stats, strategy, 
            config['output_directory']
        )
        
        # Final summary
        print("\n" + "="*80)
        print("🎉 ENHANCED WORKFLOW COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"✅ Tracked {trajectory.N} particles for {trajectory.T} timesteps")
        
        if density_results:
            print(f"✅ Performed density estimation:")
            if density_results.get('kde'):
                print(f"   📈 KDE analysis with Scott and Silverman bandwidths")
            if density_results.get('sph'):
                sph_count = len([k for k in density_results['sph'].keys() if k != 'eval_grid'])
                print(f"   🌊 SPH analysis with {sph_count} smoothing lengths")
        
        print(f"✅ Created enhanced visualizations and animations")
        print(f"✅ Exported comprehensive VTK datasets")
        print(f"✅ Results saved to: {config['output_directory']}")
        print(f"✅ Memory usage: {trajectory.memory_usage_mb():.1f} MB")
        
        # File count summary
        output_path = Path(config['output_directory'])
        if output_path.exists():
            png_files = len(list(output_path.rglob("*.png")))
            vtk_files = len(list(output_path.rglob("*.vt*")))
            html_files = len(list(output_path.rglob("*.html")))
            gif_files = len(list(output_path.rglob("*.gif")))
            
            print(f"📊 Generated files:")
            print(f"   📈 Plots and visualizations: {png_files}")
            print(f"   📤 VTK files: {vtk_files}")
            print(f"   🌐 Interactive files: {html_files}")
            print(f"   🎬 Animations: {gif_files}")
        
        print(f"\n🎊 Enhanced features used:")
        print(f"   📊 Density estimation: {'✅' if DENSITY_AVAILABLE and density_results else '❌'}")
        print(f"   🎨 Enhanced visualization: {'✅' if VISUALIZATION_AVAILABLE else '❌'}")
        print(f"   💾 Enhanced I/O: {'✅' if ENHANCED_WRITERS_AVAILABLE else '❌'}")
        print(f"   🌐 Interactive plots: {'✅' if PLOTLY_AVAILABLE else '❌'}")
        
        # Cleanup
        gc.collect()
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  Enhanced workflow interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Enhanced workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)