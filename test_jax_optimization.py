#!/usr/bin/env python3
"""
Quick test to verify JAX direct interpolation optimization works.
Tests with 500 particles on Edgar/FLA dataset.
"""
import sys
sys.path.insert(0, '/home/arhashemi/Workspace/welding/JAXTrace')

from jaxtrace.fields import create_fem_time_series_field
import numpy as np

print("="*80)
print("JAX DIRECT INTERPOLATION OPTIMIZATION TEST")
print("="*80)

# Configuration
config = {
    'max_octree_depth': 12,
    'max_elements_per_node': 32,
    'n_coarse_levels': 6,
    'enable_fine_structure_reuse': True,
    'revolution_timesteps': 40,
    'use_direct_interpolation': True,  # Use optimized JAX direct mode
    'cache_size': 3,
}

print(f"\nConfiguration:")
print(f"  use_direct_interpolation: {config['use_direct_interpolation']}")
print(f"  n_coarse_levels: {config['n_coarse_levels']}")
print(f"  revolution_timesteps: {config['revolution_timesteps']}")

# Load field
print(f"\nLoading field...")
pattern = "/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_*.pvtu"

try:
    field = create_fem_time_series_field(
        pattern=pattern,
        field_name='Velocity',
        load_connectivity=True,
        timesteps_to_load=40,
        **config
    )
    print(f"✅ Field loaded: {field}")

    # Get field bounds
    bounds = field.bounds
    print(f"\n📏 Field bounds: {bounds[:3]} to {bounds[3:]}")

    # Generate 500 test particles
    print(f"\n🎯 Generating 500 test particles...")
    x = np.linspace(bounds[0], bounds[3], 10)
    y = np.linspace(bounds[1], bounds[4], 10)
    z = np.linspace(bounds[2], bounds[5], 5)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    positions = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    print(f"  Generated {len(positions)} particles")

    # Test interpolation
    print(f"\n🚀 Testing JAX direct interpolation...")
    print(f"  This is where the 2.76 TiB error would occur if optimization failed...")

    try:
        velocities = field.interpolate(positions, time=120)
        print(f"✅ SUCCESS! Interpolation completed without memory explosion!")
        print(f"  Result shape: {velocities.shape}")
        print(f"  Velocity range: [{velocities.min():.3f}, {velocities.max():.3f}]")
        print(f"  Mean velocity: {velocities.mean():.3f}")

        print(f"\n" + "="*80)
        print(f"✅ JAX OPTIMIZATION WORKS!")
        print(f"="*80)
        sys.exit(0)

    except Exception as e:
        if "2.76" in str(e) or "3038615961" in str(e):
            print(f"❌ FAILED: 2.76 TiB memory explosion still occurs!")
            print(f"  The optimization did NOT get applied properly.")
        else:
            print(f"❌ FAILED with different error: {e}")
        print(f"\n" + "="*80)
        print(f"❌ JAX OPTIMIZATION DID NOT WORK")
        print(f"="*80)
        sys.exit(1)

except Exception as e:
    print(f"❌ Error during field loading: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
