#!/usr/bin/env python3
"""
Example Configuration File for JAXTrace

Copy this file and modify the parameters for your specific use case.

Usage:
    python run.py --config config_example.py
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

config = {
    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------
    'data_pattern': "/path/to/your/data_*.pvtu",  # REQUIRED: Path to VTK files
    'max_timesteps_to_load': 40,                  # Number of timesteps to load

    # -------------------------------------------------------------------------
    # Octree FEM Configuration
    # -------------------------------------------------------------------------
    'max_elements_per_leaf': 32,    # Octree leaf size (lower = finer tree)
    'max_octree_depth': 12,         # Maximum tree depth

    # -------------------------------------------------------------------------
    # Particle Seeding
    # -------------------------------------------------------------------------
    'particle_concentrations': {
        'x': 60,  # Particles per unit length in X direction
        'y': 50,  # Particles per unit length in Y direction
        'z': 15   # Particles per unit length in Z direction
    },

    # Particle distribution type: 'uniform', 'gaussian', 'random'
    'particle_distribution': 'uniform',

    # Gaussian distribution parameters (only used if distribution='gaussian')
    'gaussian_std': {
        'x': 0.2,  # Standard deviation as fraction of domain size in X
        'y': 0.2,  # Standard deviation as fraction of domain size in Y
        'z': 0.2   # Standard deviation as fraction of domain size in Z
    },

    # Option 1: Use entire domain for particles (default)
    'particle_bounds': None,

    # Option 2: Use fractional bounds (fraction of domain)
    # Example: Seed particles only in first 20% of X domain
    'particle_bounds_fraction': {
        'x': (0.0, 1.0),  # Full X range
        'y': (0.0, 1.0),  # Full Y range
        'z': (0.0, 1.0)   # Full Z range
    },

    # -------------------------------------------------------------------------
    # Tracking Parameters
    # -------------------------------------------------------------------------
    'n_timesteps': 2000,              # Number of tracking timesteps
    'dt': 0.0025,                     # Time step size
    'time_span': (0.0, 4.0),         # Simulation time range (t_start, t_end)
    'batch_size': 1000,               # Particles per batch
    'integrator': 'rk4',              # Integration method: 'rk4', 'euler'

    # -------------------------------------------------------------------------
    # Boundary Conditions
    # -------------------------------------------------------------------------
    'flow_axis': 'x',  # Flow direction: 'x', 'y', or 'z'

    # Inlet boundary (first wall along flow axis)
    # Options: 'continuous', 'none', 'reflective', 'periodic'
    'boundary_inlet': 'continuous',

    # Outlet boundary (last wall along flow axis)
    # Options: 'absorbing', 'reflective', 'periodic'
    'boundary_outlet': 'absorbing',

    # Inlet particle distribution (only for continuous inlet)
    'inlet_distribution': 'grid',  # 'grid' or 'random'

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    'slice_x0': None,              # X position for YZ slice (None = auto)
    'slice_levels': 20,            # Number of density contour levels
    'slice_cutoff_min': 0,         # Lower percentile cutoff for density
    'slice_cutoff_max': 95,        # Upper percentile cutoff for density

    # -------------------------------------------------------------------------
    # GPU Configuration
    # -------------------------------------------------------------------------
    'device': 'gpu',               # 'gpu' or 'cpu'
    'memory_limit_gb': 3.0,        # GPU memory limit in GB
}

# =============================================================================
# PRESET CONFIGURATIONS (Uncomment to use)
# =============================================================================

# -----------------------------------------------------------------------------
# Test Run (Fast, for debugging)
# -----------------------------------------------------------------------------
# config.update({
#     'particle_concentrations': {'x': 20, 'y': 10, 'z': 5},
#     'n_timesteps': 500,
#     'max_timesteps_to_load': 10,
#     'device': 'cpu',
# })

# -----------------------------------------------------------------------------
# High-Resolution Run (Slow, production quality)
# -----------------------------------------------------------------------------
# config.update({
#     'particle_concentrations': {'x': 100, 'y': 80, 'z': 20},
#     'n_timesteps': 5000,
#     'dt': 0.001,
#     'batch_size': 2000,
#     'device': 'gpu',
# })

# -----------------------------------------------------------------------------
# Gaussian Distribution (Concentrated in center)
# -----------------------------------------------------------------------------
# config.update({
#     'particle_distribution': 'gaussian',
#     'gaussian_std': {'x': 0.1, 'y': 0.1, 'z': 0.15},
# })

# -----------------------------------------------------------------------------
# Random Distribution (Uniform random)
# -----------------------------------------------------------------------------
# config.update({
#     'particle_distribution': 'random',
# })

# -----------------------------------------------------------------------------
# Inlet Region Only (First 20% of X domain)
# -----------------------------------------------------------------------------
# config.update({
#     'particle_bounds_fraction': {
#         'x': (0.0, 0.2),
#         'y': (0.0, 1.0),
#         'z': (0.0, 1.0)
#     }
# })

# -----------------------------------------------------------------------------
# No Inlet, Decay Mode (Particles exit but aren't replaced)
# -----------------------------------------------------------------------------
# config.update({
#     'boundary_inlet': 'none',
#     'boundary_outlet': 'absorbing',
# })

# -----------------------------------------------------------------------------
# Closed Domain (All reflective boundaries)
# -----------------------------------------------------------------------------
# config.update({
#     'boundary_inlet': 'reflective',
#     'boundary_outlet': 'reflective',
# })

# -----------------------------------------------------------------------------
# Periodic Boundaries
# -----------------------------------------------------------------------------
# config.update({
#     'boundary_inlet': 'periodic',
#     'boundary_outlet': 'periodic',
# })

# =============================================================================
# PERFORMANCE ESTIMATION
# =============================================================================

def estimate_performance(config):
    """Estimate memory usage and runtime."""
    import numpy as np

    conc = config['particle_concentrations']
    n_particles = conc['x'] * conc['y'] * conc['z']
    n_timesteps = config['n_timesteps']

    # Memory estimation
    mem_per_particle_mb = 0.0001  # ~100 bytes per particle per timestep
    total_memory_mb = n_particles * n_timesteps * mem_per_particle_mb

    # Runtime estimation (very rough)
    # Assumes ~1000 particles/second on GPU, ~100 on CPU
    device = config['device']
    particles_per_sec = 1000 if device == 'gpu' else 100
    runtime_sec = (n_particles * n_timesteps) / particles_per_sec

    print("="*80)
    print("PERFORMANCE ESTIMATION")
    print("="*80)
    print(f"Particles: {n_particles:,}")
    print(f"Timesteps: {n_timesteps:,}")
    print(f"Total operations: {n_particles * n_timesteps:,}")
    print(f"Estimated memory: {total_memory_mb:.1f} MB")
    print(f"Estimated runtime: {runtime_sec/60:.1f} minutes ({device.upper()})")
    print("="*80)

# Uncomment to see performance estimation
# estimate_performance(config)
