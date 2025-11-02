#!/usr/bin/env python3
"""
Minimal test for Phase 3E+3F fixes - Load only 5 timesteps!
"""
import time
from example_workflow import main

print("=" * 80)
print("PHASE 3E+3F: MINIMAL TEST (5 timesteps only)")
print("=" * 80)

config = {
    'data_pattern': '/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_*.pvtu',
    'max_timesteps_to_load': 5,  # MINIMAL: Only 5 timesteps!
    'use_direct_interpolation': True,
    'use_hash_octree': True,  # Phase 3E+3F
    'particle_concentrations': {'x': 2, 'y': 2, 'z': 1},  # 4 particles only
    'n_timesteps': 5,  # Very short tracking
    'dt': 1e-5,
    'revolution_start': 0.0,
    'revolution_end': 100.0,
    'output_dir': './test_phase3f_minimal_output',
    'shared_octree_config': {
        'max_octree_depth': 12,
        'max_cells_per_node': 30,
        'revolution_timesteps': 5  # Only 5
    }
}

print("\n📝 Minimal Test Configuration:")
print(f"   Timesteps: {config['max_timesteps_to_load']} (MINIMAL!)")
print(f"   Particles: 4 (MINIMAL!)")
print(f"   Expected time: < 2 minutes")
print(f"   Goal: Verify Phase 3E+3F fixes work")

print("\n🔍 What to look for:")
print("   1. NO import errors (Phase 3E fix)")
print("   2. Hash reuse statistics (Phase 3F)")
print("   3. GPU acceleration message")

print("\n🚀 Running minimal test...\n")
start_time = time.time()

try:
    main(config=config)
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("✅ TEST PASSED")
    print("=" * 80)
    print(f"Total time: {elapsed:.2f} seconds")
    print("\n✅ Phase 3E: No import error!")
    print("✅ Phase 3F: Check reuse statistics above")

except Exception as e:
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("❌ TEST FAILED")
    print("=" * 80)
    print(f"Time before failure: {elapsed:.2f} seconds")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
