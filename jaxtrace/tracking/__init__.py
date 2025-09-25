# jaxtrace/tracking/__init__.py  
"""  
Particle tracking module with optimized trajectory computation.  

High-performance particle tracking with JAX acceleration, automatic memory  
management, and consistent float32 data types throughout.  

Main Components:  
- Trajectory: (T,N,3) format trajectory storage  
- ParticleTracker: High-performance tracking with batch processing  
- Seeding: Various seed position generation strategies  
- BoundaryConditions: Efficient vectorized boundary handling  
"""  

from .particles import (  
    Trajectory,  
    create_trajectory_from_positions,  
    create_single_particle_trajectory,  
    create_empty_trajectory,  
    merge_trajectories,  
    compute_trajectory_statistics,  
    validate_trajectory_data  
)  

from .seeding import (  
    random_seeds,  
    uniform_grid_seeds,  
    line_seeds,  
    circle_seeds,  
    sphere_seeds,  
    rectangular_seeds,  
    gaussian_cluster_seeds,  
    stratified_seeds,  
    custom_distribution_seeds,  
    field_informed_seeds,  
    boundary_seeds,  
    combine_seed_strategies,  
    validate_seeds,  
    seed_density_analysis  
)  

from .boundary import (
    periodic_boundary,
    reflective_boundary,
    clamping_boundary,
    absorbing_boundary_factory,
    mixed_boundary,
    spherical_boundary,
    cylindrical_boundary,
    distance_based_boundary,
    inlet_outlet_boundary_factory,
    flow_through_boundary_factory,
    continuous_inlet_boundary_factory,
    no_boundary,
    unit_box_periodic,
    unit_box_reflective,
    centered_box_periodic,
    unit_sphere_reflective,
    compose_boundary_conditions,
    check_boundary_violations,
    test_boundary_condition,
    create_boundary_from_config,
    visualize_boundary_effect,
    BoundaryCondition,
    CompositeBoundaryCondition
)  

from .tracker import (
    ParticleTracker,
    TrackerOptions,
    create_tracker,
    track_particles_simple,
    compare_integrators
)

from .analysis import (
    analyze_trajectory_results,
    compute_mixing_metrics,
    analyze_particle_clustering,
)  

# Import field module components (compatible with actual fields module)  
try:  
    from ..fields import (  
        BaseField,  
        StructuredGridSampler,  
        UnstructuredField,  
        TimeSeriesField,  
        create_field_from_vtk,  
        create_field_from_data  
    )  
    FIELDS_AVAILABLE = True  
except ImportError:  
    FIELDS_AVAILABLE = False  

# Import integration functions  
try:  
    from ..integrators import (  
        euler_step,  
        rk2_step,  
        rk4_step  
    )  
    INTEGRATION_AVAILABLE = True  
except ImportError:  
    INTEGRATION_AVAILABLE = False  

# Version and metadata  
__version__ = "1.0.0"  
__author__ = "JAXTrace Development Team"  

# Export all main functions and classes  
__all__ = [  
    # Trajectory handling  
    'Trajectory',  
    'create_trajectory_from_positions',   
    'create_single_particle_trajectory',  
    'create_empty_trajectory',  
    'merge_trajectories',  
    'compute_trajectory_statistics',  
    'validate_trajectory_data',  
    
    # Seeding functions  
    'random_seeds',  
    'uniform_grid_seeds',   
    'line_seeds',  
    'circle_seeds',  
    'sphere_seeds',  
    'rectangular_seeds',  
    'gaussian_cluster_seeds',  
    'stratified_seeds',  
    'custom_distribution_seeds',  
    'field_informed_seeds',  
    'boundary_seeds',  
    'combine_seed_strategies',  
    'validate_seeds',  
    'seed_density_analysis',  
    
    # Boundary conditions  
    'periodic_boundary',
    'reflective_boundary',
    'clamping_boundary',
    'absorbing_boundary_factory',
    'mixed_boundary',
    'spherical_boundary',
    'cylindrical_boundary',
    'distance_based_boundary',
    'inlet_outlet_boundary_factory',
    'flow_through_boundary_factory',
    'continuous_inlet_boundary_factory',
    'no_boundary',
    'unit_box_periodic',
    'unit_box_reflective',
    'centered_box_periodic',
    'unit_sphere_reflective',
    'compose_boundary_conditions',
    'check_boundary_violations',
    'test_boundary_condition',  
    'create_boundary_from_config',  
    'visualize_boundary_effect',  
    'BoundaryCondition',  
    'CompositeBoundaryCondition',  
    
    # Tracking
    'ParticleTracker',
    'TrackerOptions',
    'create_tracker',
    'track_particles_simple',
    'compare_integrators',

    # Analysis
    'analyze_trajectory_results',
    'compute_mixing_metrics',
    'analyze_particle_clustering'  
]  

# Conditionally add field and integration exports if available  
if FIELDS_AVAILABLE:  
    __all__.extend([  
        'BaseField',  
        'StructuredGridSampler',  
        'UnstructuredField',  
        'TimeSeriesField',  
        'create_field_from_vtk',  
        'create_field_from_data'  
    ])  

if INTEGRATION_AVAILABLE:  
    __all__.extend([  
        'euler_step',  
        'rk2_step',  
        'rk4_step'  
    ])  


# Convenience field creation functions compatible with fields module  

def create_uniform_field(velocity=(1.0, 0.0, 0.0), bounds=None):  
    """  
    Create a uniform velocity field.  
    
    Parameters  
    ----------  
    velocity : tuple  
        Constant velocity vector (vx, vy, vz)  
    bounds : array-like, optional  
        Field domain bounds  
        
    Returns  
    -------  
    BaseField  
        Uniform velocity field  
    """  
    import numpy as np  
    
    if not FIELDS_AVAILABLE:  
        raise ImportError("Fields module not available")  
    
    # Create simple uniform field function  
    velocity = np.array(velocity, dtype=np.float32)  
    if len(velocity) == 2:  
        velocity = np.append(velocity, 0.0)  
    
    def uniform_field_func(positions, t=0.0):  
        """Uniform velocity field function."""  
        positions = np.asarray(positions, dtype=np.float32)  
        if positions.ndim == 1:  
            positions = positions.reshape(1, -1)  
        
        n_points = positions.shape[0]  
        return np.tile(velocity, (n_points, 1))  
    
    # Create time series field with single time step  
    if bounds is None:  
        bounds = [[-10, -10, -10], [10, 10, 10]]  
    
    # Create sample positions for the field  
    x = np.linspace(bounds[0][0], bounds[1][0], 10, dtype=np.float32)  
    y = np.linspace(bounds[0][1], bounds[1][1], 10, dtype=np.float32)  
    z = np.linspace(bounds[0][2], bounds[1][2], 10, dtype=np.float32)  
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  
    positions = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])  
    
    # Create velocity data  
    velocities = uniform_field_func(positions)  
    velocity_snapshots = velocities[None, :, :]  # (1, N, 3)  
    times = np.array([0.0], dtype=np.float32)  
    
    return create_time_series_from_arrays(  
        velocity_snapshots=velocity_snapshots,  
        time_points=times,  
        positions=positions  
    )  


def create_vortex_field(center=(0.0, 0.0), strength=1.0, bounds=None):  
    """  
    Create a 2D vortex velocity field.  
    
    Parameters  
    ----------  
    center : tuple  
        Vortex center (x, y)  
    strength : float  
        Vortex strength (circulation)  
    bounds : array-like, optional  
        Field domain bounds  
        
    Returns  
    -------  
    BaseField  
        Vortex velocity field  
    """  
    import numpy as np  
    
    if not FIELDS_AVAILABLE:  
        raise ImportError("Fields module not available")  
    
    center = np.array(center, dtype=np.float32)  
    
    def vortex_field_func(positions, t=0.0):  
        """2D vortex field function."""  
        positions = np.asarray(positions, dtype=np.float32)  
        if positions.ndim == 1:  
            positions = positions.reshape(1, -1)  
        
        # Relative positions from center  
        dx = positions[:, 0] - center[0]  
        dy = positions[:, 1] - center[1]  
        
        # Vortex velocities: v = strength * (-y, x) / (x^2 + y^2)  
        r_squared = dx**2 + dy**2  
        r_squared = np.maximum(r_squared, 1e-10)  # Avoid singularity  
        
        vx = -strength * dy / r_squared  
        vy = strength * dx / r_squared  
        vz = np.zeros_like(vx)  
        
        return np.column_stack([vx, vy, vz])  
    
    # Create sample positions for the field  
    if bounds is None:  
        bounds = [[-5, -5, -1], [5, 5, 1]]  
    
    x = np.linspace(bounds[0][0], bounds[1][0], 20, dtype=np.float32)  
    y = np.linspace(bounds[0][1], bounds[1][1], 20, dtype=np.float32)  
    z = np.linspace(bounds[0][2], bounds[1][2], 3, dtype=np.float32)  
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  
    positions = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])  
    
    # Create velocity data  
    velocities = vortex_field_func(positions)  
    velocity_snapshots = velocities[None, :, :]  # (1, N, 3)  
    times = np.array([0.0], dtype=np.float32)  
    
    return create_time_series_from_arrays(  
        velocity_snapshots=velocity_snapshots,  
        time_points=times,  
        positions=positions  
    )  


def create_saddle_field(center=(0.0, 0.0), strength=1.0, bounds=None):  
    """  
    Create a 2D saddle point velocity field.  
    
    Parameters  
    ----------  
    center : tuple  
        Saddle center (x, y)  
    strength : float  
        Field strength  
    bounds : array-like, optional  
        Field domain bounds  
        
    Returns  
    -------  
    BaseField  
        Saddle point velocity field  
    """  
    import numpy as np  
    
    if not FIELDS_AVAILABLE:  
        raise ImportError("Fields module not available")  
    
    center = np.array(center, dtype=np.float32)  
    
    def saddle_field_func(positions, t=0.0):  
        """2D saddle field function: v = strength * (x, -y)."""  
        positions = np.asarray(positions, dtype=np.float32)  
        if positions.ndim == 1:  
            positions = positions.reshape(1, -1)  
        
        # Relative positions from center  
        dx = positions[:, 0] - center[0]  
        dy = positions[:, 1] - center[1]  
        
        # Saddle velocities  
        vx = strength * dx  
        vy = -strength * dy  
        vz = np.zeros_like(vx)  
        
        return np.column_stack([vx, vy, vz])  
    
    # Create sample positions for the field  
    if bounds is None:  
        bounds = [[-3, -3, -1], [3, 3, 1]]  
    
    x = np.linspace(bounds[0][0], bounds[1][0], 15, dtype=np.float32)  
    y = np.linspace(bounds[0][1], bounds[1][1], 15, dtype=np.float32)  
    z = np.linspace(bounds[0][2], bounds[1][2], 3, dtype=np.float32)  
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  
    positions = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])  
    
    # Create velocity data  
    velocities = saddle_field_func(positions)  
    velocity_snapshots = velocities[None, :, :]  # (1, N, 3)  
    times = np.array([0.0], dtype=np.float32)  
    
    return create_time_series_from_arrays(  
        velocity_snapshots=velocity_snapshots,  
        time_points=times,  
        positions=positions  
    )  


# Add convenience field functions to exports if fields are available  
if FIELDS_AVAILABLE:  
    __all__.extend([  
        'create_uniform_field',  
        'create_vortex_field',   
        'create_saddle_field'  
    ])  


# Convenience functions for common workflows  

def quick_track(  
    seed_positions,  
    field,  
    time_span=(0.0, 10.0),  
    n_timesteps=100,  
    integrator='rk4',  
    boundary='periodic',  
    bounds=None  
):  
    """  
    Quick particle tracking with sensible defaults.  
    
    Parameters  
    ----------  
    seed_positions : array-like  
        Initial particle positions  
    field : BaseField or callable  
        Velocity field  
    time_span : tuple  
        (start_time, end_time)   
    n_timesteps : int  
        Number of time steps  
    integrator : str  
        Integration method: 'euler', 'rk2', 'rk4'  
    boundary : str or callable  
        Boundary type: 'periodic', 'reflective', 'clamping', 'none'  
    bounds : array-like, optional  
        Domain bounds. If None, auto-detected from seeds.  
        
    Returns  
    -------  
    Trajectory  
        Particle trajectories  
    """  
    import numpy as np  
    
    # Ensure proper position format  
    seed_positions = np.asarray(seed_positions, dtype=np.float32)  
    
    # Auto-detect bounds if needed  
    if bounds is None and boundary != 'none':  
        if seed_positions.ndim == 2 and seed_positions.shape[0] > 0:  
            pos_min = np.min(seed_positions, axis=0)  
            pos_max = np.max(seed_positions, axis=0)  
            margin = 0.1 * (pos_max - pos_min)  # 10% margin  
            bounds = np.array([pos_min - margin, pos_max + margin])  
        else:  
            bounds = [[-10, -10, -10], [10, 10, 10]]  # Default large box  
    
    # Create boundary condition  
    if boundary == 'none':  
        boundary_fn = no_boundary()  
    elif boundary == 'periodic':  
        boundary_fn = periodic_boundary(bounds)  
    elif boundary == 'reflective':  
        boundary_fn = reflective_boundary(bounds)  
    elif boundary == 'clamping':  
        boundary_fn = clamping_boundary(bounds)  
    elif callable(boundary):  
        boundary_fn = boundary  
    else:  
        raise ValueError(f"Unknown boundary type: {boundary}")  
    
    # Create and run tracker  
    return track_particles_simple(  
        initial_positions=seed_positions,
        velocity_field=field,
        time_span=time_span,
        n_timesteps=n_timesteps,
        integrator=integrator,
        boundary_condition=boundary_fn
    )


def demo_tracking():
    """
    Run a demonstration of particle tracking capabilities.
    
    Returns
    -------
    dict
        Demo results with trajectories and analysis
    """
    import numpy as np
    
    print("🚀 JAXTrace Particle Tracking Demo")
    print("="*50)
    
    # Create simple vortex field
    print("Creating velocity field...")
    try:
        field = create_vortex_field(center=[0, 0], strength=1.0)
        print("✅ Created vortex field")
    except ImportError:
        # Fallback: create a simple callable field
        def simple_vortex(positions, t=0.0):
            positions = np.asarray(positions, dtype=np.float32)
            if positions.ndim == 1:
                positions = positions.reshape(1, -1)
            
            dx = positions[:, 0]
            dy = positions[:, 1]
            r_squared = dx**2 + dy**2 + 1e-10
            
            vx = -dy / r_squared
            vy = dx / r_squared
            vz = np.zeros_like(vx)
            
            return np.column_stack([vx, vy, vz])
        
        field = simple_vortex
        print("✅ Created simple vortex field (fallback)")
    
    # Generate seeds
    print("Generating seed positions...")
    seeds = circle_seeds(center=[0, 0, 0], radius=0.5, n=20)
    print(f"✅ Generated {len(seeds)} seed positions")
    
    # Track particles
    print("Tracking particles...")
    trajectory = quick_track(
        seed_positions=seeds,
        field=field,
        time_span=(0.0, 2.0),
        n_timesteps=100,
        boundary='periodic',
        bounds=[[-2, -2, -2], [2, 2, 2]]
    )
    print(f"✅ Completed tracking: {trajectory}")
    
    # Analyze results
    print("Analyzing trajectories...")
    stats = compute_trajectory_statistics(trajectory)
    print(f"✅ Analysis complete")
    
    # Summary
    results = {
        'trajectory': trajectory,
        'statistics': stats,
        'num_particles': trajectory.N,
        'num_timesteps': trajectory.T,
        'duration': trajectory.duration,
        'memory_mb': trajectory.memory_usage_mb()
    }
    
    print("\n📊 Summary:")
    print(f"  Particles: {results['num_particles']}")
    print(f"  Time steps: {results['num_timesteps']}")
    print(f"  Duration: {results['duration']:.2f}")
    print(f"  Memory: {results['memory_mb']:.1f} MB")
    print(f"  Final displacement: {stats['displacement']['mean']:.3f} ± {stats['displacement']['std']:.3f}")
    
    return results


# Helper function to create time series fields (compatible with fields module)

def create_time_series_from_arrays(velocity_snapshots, time_points, positions):
    """
    Create TimeSeriesField from velocity snapshots and positions.
    
    This is a helper function that works with the existing fields module structure.
    
    Parameters
    ----------
    velocity_snapshots : np.ndarray
        Velocity data, shape (n_times, n_points, 3)
    time_points : np.ndarray
        Time points, shape (n_times,)
    positions : np.ndarray
        Spatial positions, shape (n_points, 3)
        
    Returns
    -------
    TimeSeriesField
        Time series velocity field
    """
    if not FIELDS_AVAILABLE:
        # Return a simple callable if fields module not available
        def simple_field(pos, t=0.0):
            # Use the first time snapshot
            return velocity_snapshots[0]
        return simple_field
    
    # Try to use the actual TimeSeriesField from fields module
    try:
        return TimeSeriesField(
            velocity_snapshots=velocity_snapshots,
            time_points=time_points,
            positions=positions
        )
    except Exception:
        # Fallback to simple interpolating function
        import numpy as np
        
        def interpolating_field(pos, t=0.0):
            pos = np.asarray(pos, dtype=np.float32)
            if pos.ndim == 1:
                pos = pos.reshape(1, -1)
            
            # Find closest time index
            time_idx = np.argmin(np.abs(time_points - t))
            
            # Simple nearest neighbor interpolation
            # In a full implementation, would use proper spatial interpolation
            n_query = pos.shape[0]
            n_field = positions.shape[0]
            
            if n_query == n_field:
                # Assume same positions
                return velocity_snapshots[time_idx]
            else:
                # Return uniform field as fallback
                return np.tile(velocity_snapshots[time_idx][0], (n_query, 1))
        
        return interpolating_field


# Module-level configuration
def set_default_options(**kwargs):
    """Set default options for tracking operations."""
    global _DEFAULT_OPTIONS
    _DEFAULT_OPTIONS.update(kwargs)

# Default configuration
_DEFAULT_OPTIONS = {
    'max_memory_gb': 8.0,
    'use_jax_jit': True,
    'record_velocities': False
}

def get_default_options():
    """Get current default options."""
    return _DEFAULT_OPTIONS.copy()


# Diagnostic functions
def check_installation():
    """
    Check JAXTrace tracking module installation and dependencies.
    
    Returns
    -------
    dict
        Installation status
    """
    from ..utils.jax_utils import JAX_AVAILABLE
    
    status = {
        'jaxtrace_tracking': True,
        'jax_available': JAX_AVAILABLE,
        'numpy_available': True,
        'fields_module': FIELDS_AVAILABLE,
        'integration_module': INTEGRATION_AVAILABLE,
        'optional_dependencies': {}
    }
    
    # Check optional dependencies
    try:
        import matplotlib
        status['optional_dependencies']['matplotlib'] = True
    except ImportError:
        status['optional_dependencies']['matplotlib'] = False
    
    try:
        import scipy
        status['optional_dependencies']['scipy'] = True  
    except ImportError:
        status['optional_dependencies']['scipy'] = False
    
    try:
        import vtk
        status['optional_dependencies']['vtk'] = True
    except ImportError:
        status['optional_dependencies']['vtk'] = False
    
    try:
        import psutil
        status['optional_dependencies']['psutil'] = True
    except ImportError:
        status['optional_dependencies']['psutil'] = False
    
    return status


def print_system_info():
    """Print system information relevant to particle tracking."""
    import sys
    import platform
    
    status = check_installation()
    
    print("JAXTrace Tracking Module - System Information")
    print("=" * 50)
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"JAX Available: {'✅' if status['jax_available'] else '❌'}")
    print(f"Fields Module: {'✅' if status['fields_module'] else '❌'}")
    print(f"Integration Module: {'✅' if status['integration_module'] else '❌'}")
    print()
    
    print("Optional Dependencies:")
    for dep, available in status['optional_dependencies'].items():
        print(f"  {dep}: {'✅' if available else '❌'}")
    
    print()
    if status['jax_available']:
        try:
            import jax
            print(f"JAX Version: {jax.__version__}")
            print(f"JAX Backend: {jax.default_backend()}")
            print(f"JAX Devices: {jax.device_count()}")
        except:
            print("JAX info unavailable")
    
    # Memory info
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"System Memory: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")
    except ImportError:
        print("Memory info unavailable (install psutil)")
    
    # Check fields module compatibility
    if FIELDS_AVAILABLE:
        print(f"\nFields Module Components:")
        try:
            from ..fields import BaseField
            print(f"  BaseField: ✅")
        except ImportError:
            print(f"  BaseField: ❌")
        
        try:
            from ..fields import TimeSeriesField
            print(f"  TimeSeriesField: ✅")
        except ImportError:
            print(f"  TimeSeriesField: ❌")
            
        try:
            from ..fields import UnstructuredField
            print(f"  UnstructuredField: ✅")
        except ImportError:
            print(f"  UnstructuredField: ❌")


def create_simple_field_from_function(field_func, bounds=None):
    """
    Create a field-like object from a simple function.
    
    This provides compatibility when the full fields module is not available.
    
    Parameters
    ----------
    field_func : callable
        Function with signature field_func(positions, t) -> velocities
    bounds : array-like, optional
        Field domain bounds
        
    Returns
    -------
    object
        Field-like object with sample_at_positions method
    """
    class SimpleField:
        def __init__(self, func, bounds=None):
            self.func = func
            self.bounds = bounds
            
        def sample_at_positions(self, positions, t=0.0):
            return self.func(positions, t)
            
        def sample(self, positions, t=0.0):
            return self.func(positions, t)
            
        def __call__(self, positions, t=0.0):
            return self.func(positions, t)
    
    return SimpleField(field_func, bounds)


# Workflow utilities

def create_tracking_workflow(config):
    """
    Create a complete tracking workflow from configuration.
    
    Parameters
    ----------
    config : dict
        Workflow configuration with keys:
        - 'seeds': seed configuration
        - 'field': field configuration  
        - 'boundary': boundary configuration
        - 'tracker': tracker options
        - 'integration': integration parameters
        
    Returns
    -------
    dict
        Workflow components and results
    """
    import numpy as np
    
    workflow = {
        'config': config,
        'components': {},
        'results': None
    }
    
    # Create seeds
    seed_config = config.get('seeds', {'type': 'random', 'n': 100})
    if seed_config['type'] == 'random':
        bounds = seed_config.get('bounds', [[-1, -1, -1], [1, 1, 1]])
        seeds = random_seeds(seed_config['n'], bounds)
    elif seed_config['type'] == 'grid':
        bounds = seed_config.get('bounds', [[-1, -1, -1], [1, 1, 1]])
        resolution = seed_config.get('resolution', 10)
        seeds = uniform_grid_seeds(resolution, bounds)
    elif seed_config['type'] == 'circle':
        center = seed_config.get('center', [0, 0, 0])
        radius = seed_config.get('radius', 1.0)
        n = seed_config.get('n', 20)
        seeds = circle_seeds(center, radius, n)
    else:
        raise ValueError(f"Unknown seed type: {seed_config['type']}")
    
    workflow['components']['seeds'] = seeds
    
    # Create field
    field_config = config.get('field', {'type': 'uniform'})
    if field_config['type'] == 'uniform':
        velocity = field_config.get('velocity', [1.0, 0.0, 0.0])
        field = create_uniform_field(velocity)
    elif field_config['type'] == 'vortex':
        center = field_config.get('center', [0.0, 0.0])
        strength = field_config.get('strength', 1.0)
        field = create_vortex_field(center, strength)
    elif field_config['type'] == 'callable':
        field = field_config['function']
    else:
        raise ValueError(f"Unknown field type: {field_config['type']}")
    
    workflow['components']['field'] = field
    
    # Create boundary condition
    boundary_config = config.get('boundary', {'type': 'periodic'})
    if boundary_config['type'] == 'none':
        boundary = no_boundary()
    else:
        from .boundary import create_boundary_from_config
        boundary = create_boundary_from_config(boundary_config)
    
    workflow['components']['boundary'] = boundary
    
    # Set up integration
    integration_config = config.get('integration', {})
    time_span = integration_config.get('time_span', (0.0, 10.0))
    n_timesteps = integration_config.get('n_timesteps', 100)
    integrator = integration_config.get('method', 'rk4')
    
    # Create tracker
    tracker_config = config.get('tracker', {})
    tracker_options = TrackerOptions(**tracker_config)
    
    tracker = create_tracker(
        integrator_name=integrator,
        field=field,
        boundary_condition=boundary,
        **tracker_config
    )
    
    workflow['components']['tracker'] = tracker
    
    # Run tracking
    print("Running tracking workflow...")
    trajectory = tracker.track_particles(seeds, time_span, n_timesteps)
    workflow['results'] = trajectory
    
    # Analysis
    stats = compute_trajectory_statistics(trajectory)
    workflow['analysis'] = stats
    
    print(f"✅ Workflow completed: {trajectory}")
    return workflow


# Export workflow functions
__all__.extend([
    'create_simple_field_from_function',
    'create_tracking_workflow',
    'create_time_series_from_arrays'
])


if __name__ == "__main__":
    # Run demo when module is executed directly
    print_system_info()
    print()
    demo_tracking()