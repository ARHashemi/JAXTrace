#!/usr/bin/env python3
"""
Diagnose why JIT compilation is failing.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

import jax
import jax.numpy as jnp
from jaxtrace.io import load_optimized_dataset
from jaxtrace.tracking.boundary import reflective_boundary

print("="*60)
print("JIT COMPILATION DIAGNOSTIC")
print("="*60)

# Load minimal dataset
pattern = "/home/arhashemi/Workspace/welding/Cases/004_caseCoarse.gid/post/0eule/004_caseCoarse_*.pvtu"
field = load_optimized_dataset(pattern, max_time_steps=5, dtype="float32")

# Convert to JAX
field.data = jnp.array(field.data)
field.positions = jnp.array(field.positions)
field.times = jnp.array(field.times)
field._data_dev = jax.device_put(field.data)
field._times_dev = jax.device_put(field.times)
field._pos_dev = jax.device_put(field.positions)

print(f"\n1. Field data is JAX? {type(field.data).__name__}")
print(f"   On device: {field.data.devices() if hasattr(field.data, 'devices') else 'NumPy'}")

# Create boundary
bounds = field.get_spatial_bounds()
boundary = reflective_boundary(bounds)

print(f"\n2. Boundary created")

# Test boundary directly
test_pos = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
result = boundary(test_pos)
print(f"   Boundary result type: {type(result).__name__}")
print(f"   On device: {result.devices() if hasattr(result, 'devices') else 'NumPy'}")

# Test field evaluation
print(f"\n3. Testing field evaluation...")
vel = field.sample_at_positions(test_pos, 1.0)
print(f"   Velocity type: {type(vel).__name__}")
print(f"   On device: {vel.devices() if hasattr(vel, 'devices') else 'NumPy'}")

# Now test JIT compilation of field+boundary (this is what fails)
print(f"\n4. Testing JIT compilation...")

def field_fn(x, t):
    """Field function that tracker uses."""
    return field.sample_at_positions(x, t)

try:
    # Try to JIT compile a single step
    @jax.jit
    def test_step(x, t, dt):
        v = field_fn(x, t)
        x_new = x + dt * v
        x_bounded = boundary(x_new)
        return x_bounded

    print("   Attempting JIT compilation...")
    x0 = jnp.array([[0.01, 0.0, 0.0]], dtype=jnp.float32)
    result = test_step(x0, 1.0, 0.01)
    print(f"   ✅ JIT compilation SUCCESS!")
    print(f"   Result type: {type(result).__name__}")
    print(f"   On device: {result.devices() if hasattr(result, 'devices') else 'NumPy'}")

except Exception as e:
    print(f"   ❌ JIT compilation FAILED!")
    print(f"   Error: {e}")

    # Try without JIT to see if it works at all
    print(f"\n5. Testing without JIT...")
    try:
        x0 = jnp.array([[0.01, 0.0, 0.0]], dtype=jnp.float32)
        v = field_fn(x0, 1.0)
        x_new = x0 + 0.01 * v
        x_bounded = boundary(x_new)
        print(f"   ✅ Works without JIT")
        print(f"   Result type: {type(x_bounded).__name__}")
    except Exception as e2:
        print(f"   ❌ Also fails without JIT: {e2}")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)