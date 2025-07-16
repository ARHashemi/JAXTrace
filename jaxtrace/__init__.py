"""
JAXTrace: Memory-Optimized Particle Tracking with JAX

A high-performance particle tracking library for computational fluid dynamics
using JAX for GPU acceleration and memory optimization.
"""

__version__ = "0.1.0"
__author__ = "JAXTrace Development Team"
__email__ = "contact@jaxtrace.org"

# Core imports
from .reader import VTKReader
from .tracker import ParticleTracker
from .visualizer import ParticleVisualizer
from .utils import (
    VTKWriter,
    get_memory_config,
    monitor_memory_usage,
    get_gpu_memory_usage,
    create_custom_particle_distribution
)

# Configuration presets
MEMORY_CONFIGS = {
    'low_memory': {
        'max_time_steps': 20,
        'particle_resolution': (20, 20, 20),
        'integration_method': 'euler',
        'spatial_subsample': 2,
        'temporal_subsample': 2,
        'max_gpu_memory_gb': 4.0,
        'k_neighbors': 4,
        'shape_function': 'linear',
        'interpolation_method': 'nearest_neighbor'
    },
    'medium_memory': {
        'max_time_steps': 40,
        'particle_resolution': (30, 30, 30),
        'integration_method': 'rk2',
        'spatial_subsample': 1,
        'temporal_subsample': 1,
        'max_gpu_memory_gb': 8.0,
        'k_neighbors': 6,
        'shape_function': 'linear',
        'interpolation_method': 'finite_element'
    },
    'high_memory': {
        'max_time_steps': 100,
        'particle_resolution': (50, 50, 50),
        'integration_method': 'rk4',
        'spatial_subsample': 1,
        'temporal_subsample': 1,
        'max_gpu_memory_gb': 16.0,
        'k_neighbors': 8,
        'shape_function': 'quadratic',
        'interpolation_method': 'finite_element'
    },
    'high_accuracy': {
        'max_time_steps': 60,
        'particle_resolution': (40, 40, 40),
        'integration_method': 'rk4',
        'spatial_subsample': 1,
        'temporal_subsample': 1,
        'max_gpu_memory_gb': 12.0,
        'k_neighbors': 12,
        'shape_function': 'cubic',
        'interpolation_method': 'finite_element'
    }
}

# JAX configuration for memory optimization
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"
os.environ["JAX_PLATFORM_NAME"] = "gpu"

# Check JAX availability
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
    print(f"JAXTrace initialized with JAX version {jax.__version__}")
except ImportError:
    JAX_AVAILABLE = False
    print("WARNING: JAX not available. Some features will be disabled.")

__all__ = [
    'VTKReader',
    'ParticleTracker', 
    'ParticleVisualizer',
    'VTKWriter',
    'get_memory_config',
    'monitor_memory_usage',
    'get_gpu_memory_usage',
    'create_custom_particle_distribution',
    'MEMORY_CONFIGS',
    'JAX_AVAILABLE'
]