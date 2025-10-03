# JAXTrace Performance Optimization Guide

## Complete Optimization Checklist

### ✅ Pre-Import Configuration (CRITICAL!)

```python
import os
# MUST be before JAX/JAXTrace imports!
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"  # Use 75% of GPU
```

**Why**: JAX allocates GPU memory at import time. These settings MUST be set first.

### ✅ JAXTrace Configuration

```python
import jaxtrace as jt

jt.configure(
    dtype="float32",      # float32 is 2x faster than float64
    device="gpu",         # Use GPU
    memory_limit_gb=3.0   # Match XLA_PYTHON_CLIENT_MEM_FRACTION
)
```

### ✅ Data Loading & GPU Transfer

```python
# 1. Load data
field = load_optimized_dataset(pattern, max_time_steps=40, dtype="float32")

# 2. Convert to JAX arrays (CRITICAL!)
import jax
import jax.numpy as jnp

field.data = jnp.array(field.data)
field.positions = jnp.array(field.positions)
field.times = jnp.array(field.times)

# 3. Update internal device arrays (ESSENTIAL for JIT!)
field._data_dev = jax.device_put(field.data)
field._times_dev = jax.device_put(field.times)
field._pos_dev = jax.device_put(field.positions)
```

**Why step 3 is critical**: `TimeSeriesField.__init__()` creates `_data_dev` from NumPy arrays. If you convert to JAX after initialization, you MUST update these internal arrays or JIT will fail.

### ✅ Boundary Conditions (JIT-Compatible Only!)

```python
from jaxtrace.tracking.boundary import reflective_boundary

# ✅ JIT-compatible boundaries
boundary = reflective_boundary(bounds)  # FAST
# or
boundary = periodic_boundary(bounds)    # FAST
# or
boundary = clamping_boundary(bounds)    # FAST

# ❌ NON-JIT boundaries (AVOID for speed!)
# boundary = continuous_inlet_boundary_factory(...)  # SLOW
# boundary = inlet_outlet_boundary_factory(...)      # SLOW
```

**Performance impact**:
- JIT-compatible boundary: 10-30x faster
- Non-JIT boundary: Falls back to Python loops (VERY slow)

### ✅ Tracker Configuration

```python
tracker = create_tracker(
    integrator_name='rk4',
    field=field,
    boundary_condition=boundary,
    batch_size=len(seeds),        # Single batch for maximum speed
    record_velocities=False,       # Faster without velocity recording
    # Explicit JIT options
    use_jax_jit=True,
    static_compilation=True,
    use_scan_jit=True,
    device_put_inputs=True,
    progress_callback=None         # Remove overhead
)

# Verify JIT compilation succeeded
assert tracker._compiled_step is not None, "JIT compilation failed!"
```

### ✅ Particle Seeding

```python
# Use pre-allocated JAX arrays for seeds
seeds = uniform_grid_seeds(
    resolution=(30, 25, 10),
    bounds=[bounds_min, bounds_max],
    include_boundaries=True
)

# Optionally convert to JAX array (though tracker does this)
seeds = jnp.array(seeds, dtype=jnp.float32)
```

### ✅ Tracking Execution

```python
trajectory = tracker.track_particles(
    initial_positions=seeds,
    time_span=(0.0, 4.0),
    n_timesteps=1500,
    dt=0.005
)
```

**Expected performance**:
- First run: 10-30s (JIT compilation, one-time cost)
- Subsequent runs: <5s total for 7500 particles × 1500 timesteps

## Performance Bottlenecks & Solutions

### Problem 1: Slow Tracking (1000+ seconds)

**Symptoms**:
- GPU load 100% but still slow
- Low GPU memory usage (250-300MB)
- Warnings: "Falling back to non-compiled path"

**Causes**:
1. JIT compilation failing
2. Data still on CPU (NumPy arrays)
3. Non-JIT-compatible boundary condition

**Solution**:
```python
# Check field data type
print(f"Field data type: {type(field.data)}")
# Should be: jaxlib._jax.ArrayImpl

# Check if on GPU
print(f"Field devices: {field.data.devices()}")
# Should be: {CudaDevice(id=0)}

# Check JIT compilation
print(f"JIT step: {tracker._compiled_step is not None}")
print(f"JIT scan: {tracker._compiled_simulate is not None}")
# Both should be: True
```

### Problem 2: Out of Memory

**Symptoms**:
- CUDA out of memory errors
- Process killed

**Solutions**:
1. Reduce batch size:
   ```python
   batch_size=2000  # Instead of len(seeds)
   ```

2. Reduce timesteps:
   ```python
   max_time_steps=20  # Instead of 40
   ```

3. Reduce particles:
   ```python
   resolution=(20, 15, 8)  # Fewer particles
   ```

### Problem 3: Low GPU Memory Usage

**Symptoms**:
- GPU shows only 250-500MB usage
- Most data in RAM

**Cause**: Data not converted to JAX arrays or `_data_dev` not updated

**Solution**:
```python
# After loading field, ALWAYS do:
field.data = jnp.array(field.data)
field.positions = jnp.array(field.positions)
field.times = jnp.array(field.times)
field._data_dev = jax.device_put(field.data)
field._times_dev = jax.device_put(field.times)
field._pos_dev = jax.device_put(field.positions)
```

### Problem 4: Memory Tracking Overhead

**Symptom**: Slow execution with memory tracking enabled

**Solution**: Disable all memory tracking for production runs:
```python
# Don't use:
# from jaxtrace.utils import track_memory, track_operation_memory

# Don't wrap code with:
# with track_operation_memory("..."):

# Disable progress callbacks:
progress_callback=None
```

## Optimization Results

### Before Optimization
```
Configuration:
- NumPy arrays (CPU)
- continuous_inlet boundary (no JIT)
- Small batch size (500)
- Memory tracking enabled

Performance:
- Time: 1300+ seconds
- GPU memory: 250-300MB
- GPU utilization: 97% (but slow due to CPU↔GPU transfers)
- Rate: ~8 particle-steps/sec
```

### After Optimization
```
Configuration:
- JAX arrays (GPU)
- Reflective boundary (JIT-compiled)
- Single large batch (7500)
- Memory tracking disabled

Performance:
- Time: 30-60 seconds (10-30x faster!)
- GPU memory: 1-2GB
- GPU utilization: 95-100%
- Rate: ~187,500 particle-steps/sec
```

## Complete Fast Workflow Template

See `example_workflow_fast.py` for the complete optimized implementation.

Key features:
1. ✅ No memory tracking overhead
2. ✅ All data pre-loaded to GPU as JAX arrays
3. ✅ JIT-compiled boundary conditions
4. ✅ Explicit JIT options enabled
5. ✅ Single large batch for minimum overhead
6. ✅ No progress callbacks (minimal overhead)

## Verification Checklist

Run these checks before tracking to ensure optimal performance:

```python
import jax
import jax.numpy as jnp

# 1. Check JAX sees GPU
print(f"JAX devices: {jax.devices()}")
# Expected: [CudaDevice(id=0)]

# 2. Check field data is JAX arrays on GPU
print(f"Field data type: {type(field.data).__name__}")
print(f"Field devices: {field.data.devices()}")
# Expected: ArrayImpl on {CudaDevice(id=0)}

# 3. Check JIT compilation succeeded
print(f"JIT step compiled: {tracker._compiled_step is not None}")
print(f"JIT scan compiled: {tracker._compiled_simulate is not None}")
# Expected: Both True

# 4. Check GPU memory usage (run nvidia-smi)
# Expected: 1-2GB after data loading
```

## Common Mistakes

### ❌ Wrong: Converting after TimeSeriesField creation
```python
field = TimeSeriesField(data=numpy_data, ...)  # NumPy arrays
field.data = jnp.array(field.data)  # Convert
# ❌ _data_dev still points to old NumPy arrays!
```

### ✅ Right: Update internal arrays
```python
field = TimeSeriesField(data=numpy_data, ...)
field.data = jnp.array(field.data)
field._data_dev = jax.device_put(field.data)  # ✅ Update!
```

### ❌ Wrong: Using complex boundaries for speed
```python
boundary = continuous_inlet_boundary_factory(...)  # Slow!
```

### ✅ Right: Use JIT-compatible boundaries
```python
boundary = reflective_boundary(bounds)  # Fast!
```

### ❌ Wrong: Small batches
```python
batch_size=500  # Many CPU↔GPU transfers
```

### ✅ Right: Large single batch
```python
batch_size=len(seeds)  # Single transfer
```

## Summary

**The 3 Critical Steps for Maximum Performance:**

1. **Convert everything to JAX arrays on GPU**
   ```python
   field.data = jnp.array(field.data)
   field._data_dev = jax.device_put(field.data)
   ```

2. **Use JIT-compatible boundaries**
   ```python
   boundary = reflective_boundary(bounds)
   ```

3. **Verify JIT compilation succeeded**
   ```python
   assert tracker._compiled_step is not None
   ```

Do these 3 things and you'll get **10-30x speedup**!