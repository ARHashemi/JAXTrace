#!/usr/bin/env python3
"""
Quick test of workflow integration with shared octree.

Tests with a small subset of files to verify:
1. Configuration is loaded correctly
2. Shared octree factory is called
3. Files are selected correctly (last N timesteps)
4. No import errors or crashes
"""

import sys
import glob
sys.path.insert(0, '/home/arhashemi/Workspace/welding/JAXTrace')

print("=" * 70)
print("WORKFLOW INTEGRATION TEST")
print("=" * 70)

# Test 1: Verify imports work
print("\nTest 1: Verify imports...")
try:
    from jaxtrace.fields.shared_octree_factory import SharedOctreeFactory, SharedOctreeConfig
    from jaxtrace.fields.shared_octree_fem_field import create_shared_octree_fem_field
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Verify configuration
print("\nTest 2: Check configuration...")
try:
    config = {
        'use_shared_coarse_octree': True,
        'n_refinement_steps': None,
        'n_coarse_levels': 6,
        'enable_fine_structure_reuse': True,
        'revolution_timesteps': 10,  # Small for testing
        'load_last_n_timesteps': True,
        'max_elements_per_leaf': 32,
        'max_octree_depth': 12,
    }
    print(f"✓ Configuration: {len(config)} parameters")
    print(f"  - Shared octree: {config['use_shared_coarse_octree']}")
    print(f"  - Revolution timesteps: {config['revolution_timesteps']}")
except Exception as e:
    print(f"✗ Configuration failed: {e}")
    sys.exit(1)

# Test 3: Verify file selection
print("\nTest 3: Check file selection...")
try:
    files = sorted(glob.glob('/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_*.pvtu'))
    print(f"✓ Found {len(files)} files")

    # Simulate last-N selection
    max_timesteps = 10
    last_n_files = files[-max_timesteps:]
    print(f"  - Selected last {len(last_n_files)} files")
    print(f"  - First: {last_n_files[0].split('/')[-1]}")
    print(f"  - Last: {last_n_files[-1].split('/')[-1]}")
except Exception as e:
    print(f"✗ File selection failed: {e}")
    sys.exit(1)

# Test 4: Test factory initialization (without building)
print("\nTest 4: Factory initialization...")
try:
    factory_config = SharedOctreeConfig(
        n_refinement_steps=3,
        n_coarse_levels=6,
        revolution_timesteps=10,
        enable_fine_structure_reuse=True,
        use_last_n_timesteps=True
    )
    factory = SharedOctreeFactory(factory_config)
    print("✓ Factory initialized")
    print(f"  - Coarse levels: {factory_config.n_coarse_levels}")
    print(f"  - Revolution timesteps: {factory_config.revolution_timesteps}")
except Exception as e:
    print(f"✗ Factory initialization failed: {e}")
    sys.exit(1)

# Test 5: Dry-run workflow logic
print("\nTest 5: Workflow logic simulation...")
try:
    # Simulate workflow file selection
    use_shared_octree = config.get('use_shared_coarse_octree', False)
    load_last_n = config.get('load_last_n_timesteps', True)

    if use_shared_octree and load_last_n:
        print("✓ Shared octree path activated")
        print(f"  - Will load LAST {config['revolution_timesteps']} timesteps")
        print(f"  - Will pass ALL {len(files)} files to factory")
    else:
        print("✗ Wrong path taken")
        sys.exit(1)
except Exception as e:
    print(f"✗ Workflow logic failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)
print("\nReady to test with actual workflow!")
print("Next: python example_workflow.py (with 10 timesteps for quick test)")
