"""
JAXTrace: Memory-optimized particle tracking with JAX acceleration.

A comprehensive package for Lagrangian particle tracking in fluid flows with:
- Temporal velocity field interpolation
- Memory-efficient batch processing  
- Optional JAX acceleration with NumPy fallbacks
- Advanced density estimation (KDE/SPH)
- Static and interactive visualization

Core workflow:
1. Load velocity fields → TimeSeriesField
2. Configure integrator and tracker → ParticleTracker  
3. Run simulation → Trajectory
4. Analyze density → KDE/SPH
5. Visualize results → plotting functions
"""

from __future__ import annotations

# Version info
__version__ = "0.1.0"
__author__ = "JAXTrace Contributors"

# Core API exports
__all__ = [
    # Version
    "__version__",
    # Utilities - JAX availability check
    "JAX_AVAILABLE",
    # I/O - Data loading (no save_trajectory - doesn't exist)
    "open_dataset", 
    # Fields - Velocity field handling  
    "TimeSeriesField",
    "StructuredGridSampler", 
    "UnstructuredField",
    # Integrators - Numerical schemes
    "euler_step",
    "rk2_step", 
    "rk4_step",
    # Tracking - Core simulation
    "Trajectory",
    "ParticleTracker",
    "TrackerOptions",
    # Boundary conditions
    "periodic_boundary",
    "reflect_boundary",
    "clamp_to_bounds",
    # Seeding
    "random_seeds",
    # Analysis
    "analyze_trajectory_results",
    # Density estimation
    "KDEEstimator",
    "SPHDensityEstimator",
    # Visualization - Essential plotting (only existing functions)
    "plot_particles_2d",
    "plot_particles_3d",
    "plot_trajectory_2d", 
    "plot_trajectory_3d",
    "animate_trajectory_plotly_2d",
    # Export utilities (only existing functions)
    "render_frames_2d",
    "render_frames_3d",
    "encode_video_from_frames",
    "save_gif_from_frames",
    # System utilities
    "check_system_requirements",
    "generate_summary_report",
]

# Essential utilities - always available
try:
    from .utils.jax_utils import JAX_AVAILABLE  # noqa: F401
except Exception:
    JAX_AVAILABLE = False  # fallback

# I/O - Data loading (only what exists)
try:
    from .io.registry import open_dataset  # noqa: F401
except Exception:
    pass

# Fields - Velocity field handling with temporal interpolation
try:
    from .fields.time_series import TimeSeriesField  # noqa: F401
    from .fields.structured import StructuredGridSampler  # noqa: F401
    from .fields.unstructured import UnstructuredField  # noqa: F401
except Exception:
    pass

# Integrators - Numerical integration schemes  
try:
    from .integrators.euler import euler_step  # noqa: F401
    from .integrators.rk2 import rk2_step  # noqa: F401
    from .integrators.rk4 import rk4_step  # noqa: F401
except Exception:
    pass

# Tracking - Core particle tracking functionality
try:
    from .tracking.particles import Trajectory  # noqa: F401
    from .tracking.tracker import ParticleTracker, TrackerOptions  # noqa: F401
    from .tracking.seeding import random_seeds  # noqa: F401
    from .tracking.analysis import analyze_trajectory_results  # noqa: F401
    from .tracking.boundary import (
        periodic_boundary,  # noqa: F401
        reflect_boundary,   # noqa: F401
        clamp_to_bounds,    # noqa: F401
    )
except Exception:
    pass

# Density estimation - KDE and SPH
try:
    from .density.kde import KDEEstimator  # noqa: F401
    from .density.sph import SPHDensityEstimator  # noqa: F401
except Exception:
    pass

# Visualization - Essential static plotting (only what exists)
try:
    from .visualization.static import (
        plot_particles_2d,  # noqa: F401
        plot_particles_3d,  # noqa: F401
        plot_trajectory_2d,  # noqa: F401
        plot_trajectory_3d,  # noqa: F401
    )
except Exception:
    pass

# Visualization - Dynamic plotting
try:
    from .visualization.dynamic import (
        animate_trajectory_plotly_2d,  # noqa: F401
    )
except Exception:
    pass

# Export utilities - Only what actually exists
try:
    from .visualization.export_viz import (
        render_frames_2d,     # noqa: F401
        render_frames_3d,     # noqa: F401
        encode_video_from_frames,  # noqa: F401
        save_gif_from_frames, # noqa: F401
    )
except Exception:
    pass

# System utilities
try:
    from .utils.diagnostics import check_system_requirements  # noqa: F401
    from .utils.reporting import generate_summary_report  # noqa: F401
except Exception:
    pass

# Package-level configuration and data consistency helpers
def configure_package(
    dtype: str = "float32",
    device: str = "cpu", 
    memory_limit_gb: float = 8.0,
) -> None:
    """
    Configure JAXTrace package-wide settings for data consistency.
    
    Parameters
    ----------
    dtype : str
        Default floating point precision: 'float32' or 'float64'
        float32 recommended for JAX performance
    device : str  
        Default JAX device: 'cpu', 'gpu', or 'tpu'
    memory_limit_gb : float
        Memory limit for automatic batch size selection
        
    Notes
    -----
    This function ensures consistent data types and device placement
    across the entire tracking pipeline.
    """
    global _package_config
    
    _package_config = {
        'dtype': dtype,
        'device': device, 
        'memory_limit_gb': memory_limit_gb,
    }
    
    # Configure JAX if available
    if JAX_AVAILABLE:
        try:
            from .utils.jax_utils import configure_xla_env, to_device
            configure_xla_env()
            
            # Set default device 
            if device != "cpu":
                import jax
                devices = jax.devices(device.lower())
                if devices:
                    jax.config.update("jax_default_device", devices[0])
                    
        except Exception as e:
            print(f"Warning: JAX configuration failed: {e}")

# Default configuration            
_package_config = {
    'dtype': 'float32',
    'device': 'cpu', 
    'memory_limit_gb': 8.0,
}

def get_package_config() -> dict:
    """Get current package configuration."""
    return _package_config.copy()

# Data consistency utilities
def ensure_consistent_arrays(*arrays, dtype=None, device=None):
    """
    Ensure arrays have consistent dtype and device placement.
    
    Parameters
    ----------
    *arrays : array_like
        Arrays to make consistent
    dtype : str, optional
        Target dtype; uses package default if None
    device : str, optional  
        Target device; uses package default if None
        
    Returns
    -------
    tuple
        Converted arrays with consistent properties
    """
    if dtype is None:
        dtype = _package_config['dtype']
    if device is None:
        device = _package_config['device']
        
    if JAX_AVAILABLE:
        try:
            import jax.numpy as jnp
            from .utils.jax_utils import to_device
            
            result = []
            for arr in arrays:
                arr_jnp = jnp.asarray(arr, dtype=getattr(jnp, dtype))
                if device != "cpu":
                    arr_jnp = to_device(arr_jnp, device)
                result.append(arr_jnp)
            return tuple(result)
            
        except Exception:
            pass
    
    # NumPy fallback
    import numpy as np
    result = []
    for arr in arrays:
        arr_np = np.asarray(arr, dtype=getattr(np, dtype))
        result.append(arr_np)
    return tuple(result)


# Add these to __all__
__all__.extend([
    "configure_package",
    "get_package_config", 
    "ensure_consistent_arrays",
])

from .utils.config import configure, get_config, reset_config

__all__.extend([
    "configure",
    "get_config", 
    "reset_config",
])