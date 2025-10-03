# JAX JIT Compilation Fix

## Problem Diagnosis

### Symptoms
- 97% GPU load but only 265MB memory usage
- Progress bar delays every 10%
- Warning messages about falling back to non-compiled execution
- Very slow tracking (10-30s per batch)

### Root Cause
```
UserWarning: Falling back to step-by-step path (JAX scan failed):
The numpy.ndarray conversion method __array__() was called on traced array
```

**The velocity field data was stored as NumPy arrays instead of JAX arrays**, preventing JIT compilation.

## The Fix

### Changes Made to `example_workflow_memory_optimized.py`

1. **After loading VTK data** (line 246-261):
```python
# Convert data to JAX arrays for GPU acceleration
print("🔄 Converting field data to JAX arrays on GPU...")
import jax.numpy as jnp
import jax

field.data = jnp.array(field.data)  # Move to GPU
field.positions = jnp.array(field.positions)  # Move to GPU
field.times = jnp.array(field.times)  # Move to GPU

# Update the internal JAX device arrays (important!)
field._data_dev = jax.device_put(field.data)
field._times_dev = jax.device_put(field.times)
field._pos_dev = jax.device_put(field.positions)
```

2. **For synthetic fields** (line 317-321):
```python
# Convert to JAX arrays for GPU acceleration
import jax.numpy as jnp
velocity_data = jnp.array(velocity_data)
positions = jnp.array(positions)
times = jnp.array(times)
```

## Why This Works

### Before Fix
```
NumPy array (CPU) → JAX tries to JIT compile → Fails (can't trace NumPy)
                  ↓
              Falls back to Python for-loops (VERY SLOW)
```

### After Fix
```
JAX array (GPU) → JAX JIT compiles successfully → Compiled GPU code (FAST)
```

### Expected Performance Improvement

| Metric | Before | After |
|--------|--------|-------|
| First batch | 10-30s | 10-30s (compilation) |
| Subsequent batches | 10-30s each | 1-3s each |
| Overall speedup | 1x | **10-30x faster** |
| GPU memory | 265MB | 1-2GB |
| JIT warnings | YES | NO |

## What to Expect After Running

### During Execution

1. **First Batch (0-10%)**
   ```
   • Duration: 10-30 seconds
   • Reason: JIT compilation (one-time cost)
   • GPU memory: Gradually increases to ~1-2GB
   • No warning messages
   ```

2. **Subsequent Batches (10-100%)**
   ```
   • Duration: 1-3 seconds per 10%
   • Reason: Using compiled code
   • GPU memory: Stable at 1-2GB
   • Smooth, consistent progress
   ```

### Success Indicators

✅ **No warning messages** about "falling back to non-compiled path"
✅ **GPU memory increases** to 1-2GB (data now on GPU)
✅ **Much faster** after first batch
✅ **97% GPU utilization** (good - GPU is working hard)
✅ **Smooth progress bar** after initial compilation

## Verification

### Check if Fix Worked

1. **Run the script**:
   ```bash
   python example_workflow_memory_optimized.py
   ```

2. **Look for this message**:
   ```
   🔄 Converting field data to JAX arrays on GPU...
      ✅ Field data now on GPU: TFRT_GPU_0
      💾 GPU memory estimate: 87.2 MB
   ```

3. **Watch for NO warnings** during tracking:
   ```
   🏃 Running GPU-accelerated particle tracking...
   [Should NOT see "falling back" warnings]
   ```

4. **Monitor GPU memory** in another terminal:
   ```bash
   watch -n 1 nvidia-smi
   ```
   Expected: Memory usage increases to 1-2GB

### If You Still See Problems

#### Problem: Still see "falling back" warnings
**Check**: Did the conversion happen?
```python
# Add after loading field:
print(f"Field data type: {type(field.data)}")
print(f"Field data device: {field.data.device() if hasattr(field.data, 'device') else 'NumPy'}")
```
Expected output: `jaxlib.xla_extension.ArrayImpl` on `TFRT_GPU_0`

#### Problem: GPU memory still only 265MB
**Cause**: Data still on CPU
**Fix**: Make sure the conversion code runs without errors

#### Problem: Out of memory error
**Solution**: Reduce batch size or number of timesteps
```python
'batch_size': min(len(seeds), 1000)  # Reduce from 2000
max_time_steps=20  # Reduce from 40
```

## Technical Details

### Why Update `_data_dev` etc?

The `TimeSeriesField.__init__()` method creates these internal arrays:
```python
self._data_dev = jax.device_put(jnp.asarray(self.data))
```

When we modify `field.data` after initialization, we MUST also update `_data_dev`, otherwise:
- `sample_at_positions()` uses the OLD `_data_dev` (NumPy arrays)
- JIT compilation fails
- Falls back to slow Python execution

### Memory Layout

```
GPU Memory (4GB total):
├── System overhead: ~500MB
├── JAX runtime: ~200MB
├── Velocity field data: ~87MB
│   ├── field.data: (40, 185865, 3) float32 = 87.2MB
│   ├── field.positions: (185865, 3) float32 = 2.2MB
│   └── field.times: (40,) float32 = 160B
├── Particle trajectories: ~144MB
│   └── (2000 particles, 1500 timesteps, 3D) float32
├── Working memory: ~500MB
└── Available: ~2.5GB
```

## Performance Comparison

### Before Fix (NumPy arrays, no JIT)
```
Total tracking time: ~30 minutes
├── Batch 1: 30s
├── Batch 2: 30s
├── Batch 3: 30s
└── ... (60 batches × 30s each)
```

### After Fix (JAX arrays, JIT compiled)
```
Total tracking time: ~2-3 minutes
├── Batch 1: 30s (compilation)
├── Batch 2: 2s (compiled code!)
├── Batch 3: 2s
└── ... (59 batches × 2s each)
```

**Speedup: 10-15x faster overall!**

## Summary

The key insight: **JAX needs JAX arrays to JIT compile**. Loading data as NumPy arrays and using them in JAX functions causes:
1. JIT compilation to fail
2. Fallback to slow Python execution
3. Low GPU memory usage (data stays on CPU)
4. High GPU compute usage (GPU does computations but data transfers are bottleneck)

The fix is simple: **Convert arrays after loading**:
```python
field.data = jnp.array(field.data)
field._data_dev = jax.device_put(field.data)
```

This enables JIT compilation and keeps data on GPU for maximum performance!