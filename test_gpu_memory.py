#!/usr/bin/env python3
"""
Quick test to verify GPU memory usage with field data.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

import numpy as np
import jax
import jax.numpy as jnp
from jaxtrace.io import load_optimized_dataset
from jaxtrace.fields import TimeSeriesField

print("="*60)
print("GPU MEMORY TEST")
print("="*60)

# Check JAX setup
print(f"\n1. JAX Configuration:")
print(f"   Devices: {jax.devices()}")
print(f"   Default backend: {jax.default_backend()}")

# Load data
print(f"\n2. Loading VTK data...")
pattern = "/home/arhashemi/Workspace/welding/Cases/004_caseCoarse.gid/post/0eule/004_caseCoarse_*.pvtu"

try:
    field = load_optimized_dataset(
        data_pattern=pattern,
        max_memory_per_timestep_mb=100.0,
        max_time_steps=10,  # Just 10 for quick test
        dtype="float32"
    )
    print(f"   ✅ Loaded field: {field}")

    # Check types BEFORE conversion
    print(f"\n3. Data Types BEFORE JAX conversion:")
    print(f"   field.data type: {type(field.data)}")
    print(f"   field.positions type: {type(field.positions)}")
    print(f"   field.times type: {type(field.times)}")

    if hasattr(field.data, 'devices'):
        print(f"   field.data devices: {field.data.devices()}")
    else:
        print(f"   field.data is NumPy array (not on GPU)")

    # Convert to JAX
    print(f"\n4. Converting to JAX arrays...")
    field.data = jnp.array(field.data)
    field.positions = jnp.array(field.positions)
    field.times = jnp.array(field.times)

    # Update internal arrays
    field._data_dev = jax.device_put(field.data)
    field._times_dev = jax.device_put(field.times)
    field._pos_dev = jax.device_put(field.positions)

    # Check types AFTER conversion
    print(f"\n5. Data Types AFTER JAX conversion:")
    print(f"   field.data type: {type(field.data)}")
    print(f"   field.positions type: {type(field.positions)}")
    print(f"   field.times type: {type(field.times)}")
    print(f"   field.data devices: {field.data.devices()}")

    # Check memory sizes
    print(f"\n6. Memory Sizes:")
    data_mb = field.data.nbytes / 1024 / 1024
    pos_mb = field.positions.nbytes / 1024 / 1024
    times_mb = field.times.nbytes / 1024 / 1024
    total_mb = data_mb + pos_mb + times_mb

    print(f"   field.data: {data_mb:.2f} MB")
    print(f"   field.positions: {pos_mb:.2f} MB")
    print(f"   field.times: {times_mb:.6f} MB")
    print(f"   Total: {total_mb:.2f} MB")

    # Test field evaluation (this is what tracker uses)
    print(f"\n7. Testing field evaluation...")
    test_pos = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    test_time = 1.0

    print(f"   Query position type: {type(test_pos)}")
    print(f"   Query position devices: {test_pos.devices()}")

    result = field.sample_at_positions(test_pos, test_time)
    print(f"   Result type: {type(result)}")
    if hasattr(result, 'devices'):
        print(f"   Result devices: {result.devices()}")
        print(f"   ✅ Field evaluation returns JAX arrays on GPU!")
    else:
        print(f"   ❌ Field evaluation returns NumPy arrays (CPU)!")

    print(f"\n8. Check if data stayed on GPU:")
    print(f"   field.data still JAX? {type(field.data).__name__}")
    print(f"   field.data devices: {field.data.devices() if hasattr(field.data, 'devices') else 'NumPy'}")

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)