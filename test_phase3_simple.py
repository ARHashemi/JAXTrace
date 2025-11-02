#!/usr/bin/env python3
"""
Simple Phase 3 Hash Octree Integration Test

Quick test to verify hash octree integration works before full profiling.
"""

import os
import sys
import numpy as np

# Enable JAX 64-bit mode
import jax
jax.config.update("jax_enable_x64", True)

# GPU optimization
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

print("="*80)
print("PHASE 3: SIMPLE HASH OCTREE INTEGRATION TEST")
print("="*80)

# Test configuration
config = {
    # Data
    'data_pattern': "/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_*.pvtu",
    'max_timesteps_to_load': 10,  # Reduce for faster testing

    # Octree
    'n_coarse_levels': 6,
    'max_octree_depth': 12,
    'max_elements_per_leaf': 32,
    'revolution_timesteps': 10,

    # Phase 3: Enable hash octree
    'use_hash_octree': True,

    # Minimal particles for testing
    'particle_concentrations': {'x': 5, 'y': 5, 'z': 2},
    'particle_distribution': 'uniform',
    'particle_bounds': [
        np.array([-0.026, -0.023, -0.01]),
        np.array([-0.020, -0.018, -0.005])
    ],

    # Minimal tracking - use revolution cycle timesteps
    'n_timesteps': 50,
    'dt': 0.0025,
    'time_span': (150.0, 159.0),  # Match revolution cycle
    'batch_size': 50,
    'integrator': 'rk4',

    # Boundaries
    'flow_axis': 'x',
    'boundary_inlet': 'reflective',
    'boundary_outlet': 'reflective',

    # Skip analysis
    'perform_density_analysis': False,

    # GPU
    'device': 'gpu',
    'memory_limit_gb': 3.0,
}

print("\nConfiguration:")
print(f"  Data files: {config['max_timesteps_to_load']}")
print(f"  Hash octree: ENABLED")
print(f"  Particles: ~{config['particle_concentrations']['x'] * config['particle_concentrations']['y'] * config['particle_concentrations']['z']}")
print(f"  Timesteps: {config['n_timesteps']}")
print("\nStarting test...")
print("-"*80)

try:
    from example_workflow import main

    # Run workflow
    main(config=config)

    print("\n" + "="*80)
    print("✅ HASH OCTREE INTEGRATION TEST PASSED")
    print("="*80)
    print("\nThe hash octree successfully integrated with the workflow!")
    print("Ready for full profiling test.")

    sys.exit(0)

except Exception as e:
    print("\n" + "="*80)
    print("❌ HASH OCTREE INTEGRATION TEST FAILED")
    print("="*80)
    print(f"\nError: {e}")
    print("\nFull traceback:")
    import traceback
    traceback.print_exc()

    sys.exit(1)
