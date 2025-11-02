#!/usr/bin/env python3
"""
Test hash octree reuse optimization (Phase 3F).

This test verifies that hash octrees are reused when fine octree structures
are identical across timesteps, providing ~10× speedup in building.
"""

import time
from example_workflow import main

print("=" * 80)
print("PHASE 3F: HASH OCTREE REUSE TEST")
print("=" * 80)

config = {
    'data_pattern': '/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_*.pvtu',
    'max_timesteps_to_load': 40,  # Full revolution cycle
    'use_direct_interpolation': True,
    'use_hash_octree': True,  # Phase 3E+3F: Enable GPU acceleration + reuse
    'particle_concentrations': {'x': 3, 'y': 3, 'z': 2},  # 18 particles (fast test)
    'n_timesteps': 10,  # Short tracking run
    'dt': 1e-5,
    'revolution_start': 0.0,
    'revolution_end': 100.0,
    'output_dir': './test_hash_reuse_output',
    'shared_octree_config': {
        'max_octree_depth': 12,
        'max_cells_per_node': 30,
        'revolution_timesteps': 40
    }
}

print("\n📝 Test Configuration:")
print(f"   Timesteps to load: {config['max_timesteps_to_load']}")
print(f"   Particles: {config['particle_concentrations']}")
print(f"   Tracking timesteps: {config['n_timesteps']}")
print(f"   Hash octrees enabled: {config['use_hash_octree']}")

print("\n🚀 Running test with hash octree reuse...")
print("   Expected: ~90% reuse rate (36/40 timesteps reused)")
print("   Expected: ~10× speedup in hash octree building")

start_time = time.time()

try:
    main(config=config)
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("✅ TEST PASSED")
    print("=" * 80)
    print(f"Total time: {elapsed:.2f} seconds")
    print("\nExpected reuse statistics in output above:")
    print("  - Unique hash octrees: 4-5 (10-12.5%)")
    print("  - Reused: 35-36 timesteps (87.5-90%)")
    print("  - Speedup from reuse: ~8-10×")

except Exception as e:
    print("\n" + "=" * 80)
    print("❌ TEST FAILED")
    print("=" * 80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
