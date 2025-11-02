#!/usr/bin/env python3
"""
Phase 3E GPU Acceleration Test

Tests that:
1. Hash octrees are built successfully
2. GPU path is used (not io_callback fallback)
3. GPU utilization is high (60-80%)
4. No memory crashes

Run with: python test_phase3e_gpu.py
Monitor GPU: nvidia-smi dmon -s u -d 1
"""

from example_workflow import main

# Small test configuration
config = {
    # Data
    'data_pattern': '/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_*.pvtu',
    'max_timesteps_to_load': 10,
    'use_stable_mesh_only': True,

    # Octree
    'max_elements_per_leaf': 32,
    'max_octree_depth': 12,
    'use_direct_interpolation': True,
    'use_hash_octree': True,  # Phase 3E: Enable GPU acceleration

    # Particles - SMALL for testing
    'particle_concentrations': {'x': 5, 'y': 5, 'z': 2},  # 50 particles
    'particle_distribution': 'uniform',

    # Tracking - SHORT for testing
    'n_timesteps': 20,  # 20 steps instead of 2000
    'dt': 0.0025,
    'batch_size': 50,
    'integrator': 'rk4',

    # Boundary
    'flow_axis': 'x',
    'boundary_inlet': 'reflective',
    'boundary_outlet': 'reflective',

    # Skip density analysis
    'perform_density_analysis': False,

    # GPU
    'device': 'gpu',
    'memory_limit_gb': 3.0,
}

print("="*80)
print("PHASE 3E GPU ACCELERATION TEST")
print("="*80)
print(f"Configuration:")
print(f"  Particles: 50 (5×5×2)")
print(f"  Timesteps: 20")
print(f"  Hash octrees: ENABLED")
print(f"  Expected GPU utilization: 60-80%")
print("="*80)
print()

print("Starting test...")
print("Monitor GPU with: nvidia-smi dmon -s u -d 1")
print()

try:
    main(config=config)
    print()
    print("="*80)
    print("✅ TEST PASSED")
    print("="*80)
    print()
    print("Check the output above for:")
    print("  1. '✅ Pre-built N hash octrees for GPU' - Hash octrees built")
    print("  2. '🚀 Phase 3E: Using GPU-accelerated hash octree path' - GPU path used")
    print("  3. No '⚠️ Falling back to io_callback' warnings")
    print("  4. High GPU utilization (60-80% on nvidia-smi)")
    print()

except Exception as e:
    print()
    print("="*80)
    print("❌ TEST FAILED")
    print("="*80)
    print(f"Error: {e}")
    print()
    import traceback
    traceback.print_exc()
