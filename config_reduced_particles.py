"""
Configuration for reduced particle test with resource monitoring.
"""

# REDUCED PARTICLE COUNT: 10x10x5 = 500 particles (vs 60x50x15 = 45,000)
# This is a ~99% reduction to test JAX direct interpolation feasibility

config_reduced = {
    # Particle configuration - REDUCED
    'particle_concentrations': {'x': 10, 'y': 10, 'z': 5},  # 500 total particles

    # Keep all other settings the same
    'use_direct_interpolation': True,  # Test JAX direct mode
    'max_elements_per_leaf': 32,
    'max_octree_depth': 12,
    'coarse_octree_levels': 6,
    'fine_octree_reuse': True,
    'revolution_timesteps': 40,
    'cache_size': 3,

    # Tracking parameters
    'tracking_steps': 2000,
    'dt': 0.0025,
    'integrator': 'rk4',

    # Particle region (same fractional bounds as original)
    'particle_bounds_fraction': {
        'x': (0.1, 0.3),
        'y': (0.0, 1.0),
        'z': (0.0, 1.0)
    },

    # Device configuration
    'device': 'gpu',
    'memory_limit_gb': 3.0,
}

print("="*80)
print("REDUCED PARTICLE TEST CONFIGURATION")
print("="*80)
print(f"Original:  60×50×15 = 45,000 particles")
print(f"Reduced:   10×10×5  = 500 particles")
print(f"Reduction: 99%")
print(f"Mode: JAX direct interpolation (coarse+fine octrees)")
print("="*80)
