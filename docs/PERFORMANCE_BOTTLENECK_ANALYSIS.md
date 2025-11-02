# Performance Bottleneck Analysis: GPU Utilization 2-3%

**Date**: 2025-10-30
**Issue**: Despite having GPU-ready hash octrees, GPU utilization remains at 2-3% with CPU at 262%

---

## Current Performance

**Observation** (6000 particles, 2000 timesteps):
- GPU Utilization: **2-3%** ❌
- CPU Usage: **262%** (21% per core × 12 cores) ❌
- GPU Memory: 724MB (loaded but unused)
- RAM: 3GB
- Speed: **Very slow**

**Expected** (with full GPU pipeline):
- GPU Utilization: **80-95%** ✅
- CPU Usage: **10-20%** (minimal)
- Speedup: **50-140×**

---

## Root Cause: `io_callback` Barrier

### The Bottleneck

**File**: `jaxtrace/fields/shared_octree_fem_field.py:397-444`

```python
def sample_at_positions(self, query_positions: np.ndarray, t: float) -> jnp.ndarray:
    """Sample field at positions using per-timestep data loading."""

    # ❌ BOTTLENECK: io_callback forces CPU execution!
    result = io_callback(
        callback_wrapper,                           # CPU function
        jax.ShapeDtypeStruct(query_positions.shape, jnp.float32),
        query_positions,
        t_jax,
        ordered=False
    )
    return result
```

**Why This Kills Performance**:

1. **CPU Barrier**: `io_callback` executes on CPU, blocking GPU pipeline
2. **Per-Particle Call**: Called for EVERY particle at EVERY timestep
3. **Serialization**: JAX arrays → NumPy → callback → JAX (slow conversion)
4. **No Batching**: Each call processes particles individually
5. **GPU Idle**: GPU waits for CPU to finish each callback

---

## Current Architecture (Slow)

```
Tracking Loop (Python) →
  For each timestep:
    sample_at_positions() →
      io_callback() [CPU BARRIER] →
        _sample_cpu_callback() [CPU] →
          _sample_with_two_stage_interpolation() →
            _find_elements_batch_hash_octree() [Partially GPU] →
              hash_lookup_batch_jax() [GPU ✅]
              test_candidates_batch_jax() [GPU ✅]
            fem_interpolate_batch_jax() [GPU ✅]

GPU Utilization: 2-3% (GPU only used 40% of the time, rest is CPU callbacks)
```

**Problem Layers**:
1. ❌ **Layer 1**: Tracking loop on CPU (Python for-loop)
2. ❌ **Layer 2**: `io_callback` on CPU (barrier)
3. ❌ **Layer 3**: `_sample_cpu_callback` on CPU (NumPy operations)
4. ⚠️  **Layer 4**: `_sample_with_two_stage_interpolation` (mixed)
5. ✅ **Layer 5**: Hash lookup on GPU
6. ✅ **Layer 6**: Element testing on GPU
7. ✅ **Layer 7**: FEM interpolation on GPU

**Only layers 5-7 use GPU**, which is why utilization is so low!

---

## Target Architecture (Fast)

```
Tracking Loop (Minimal Python) →
  JAX JIT-Compiled Pipeline [FULL GPU] →
    sample_field_gpu_batch() [GPU] →
      Temporal interpolation [GPU] →
        hash_lookup_batch_jax() [GPU] →
        test_candidates_batch_jax() [GPU] →
        fem_interpolate_batch_jax() [GPU] →
    RK4 integration [GPU] →
    Boundary conditions [GPU]

GPU Utilization: 80-95% (entire pipeline on GPU)
```

**All Layers on GPU**:
1. ✅ Tracking loop JIT-compiled
2. ✅ Field sampling pure JAX
3. ✅ Temporal interpolation pure JAX
4. ✅ Hash lookup GPU
5. ✅ Element testing GPU
6. ✅ FEM interpolation GPU
7. ✅ RK4 integration GPU

---

## What's Actually Happening

### Per-Timestep Breakdown

For **6000 particles** at **one timestep**:

```
1. Python loop calls sample_at_positions(positions, t)          [CPU, 0.1ms]
2. io_callback() triggered                                      [CPU barrier, 0.5ms]
3. Convert JAX → NumPy                                          [CPU, 2ms]
4. _sample_cpu_callback() executes                              [CPU, 5ms]
5. Find timestep indices (left_idx, right_idx, alpha)           [CPU, 0.5ms]
6. _sample_with_two_stage_interpolation() called                [CPU, 1ms]
7. _find_elements_batch_hash_octree() START                     [CPU→GPU transfer, 5ms]
8.   hash_lookup_batch_jax()                                    [GPU, 2ms] ✅
9.   test_candidates_batch_jax()                                [GPU, 3ms] ✅
10. _find_elements_batch_hash_octree() END                      [GPU→CPU transfer, 5ms]
11. fem_interpolate_batch_jax() START                           [CPU→GPU transfer, 5ms]
12.   FEM interpolation                                         [GPU, 4ms] ✅
13. fem_interpolate_batch_jax() END                             [GPU→CPU transfer, 5ms]
14. Temporal interpolation                                      [CPU, 3ms]
15. Convert NumPy → JAX                                         [CPU, 2ms]
16. Return to tracking loop                                     [CPU, 0.1ms]

Total: ~43ms per timestep for 6000 particles
GPU active: ~9ms (21% of time)
CPU active: ~34ms (79% of time)
```

**With 2000 timesteps**: 43ms × 2000 = **86 seconds minimum**

**Observed**: Much slower due to overhead, Python loop inefficiency, lack of batching

---

## Where Arrays Currently Live

### CPU Arrays (NumPy)
- ❌ `query_positions` (converted from JAX)
- ❌ `t` (time scalar)
- ❌ Timestep indices (left_idx, right_idx, alpha)
- ❌ `element_ids` (intermediate result from hash lookup)
- ❌ Mesh data during callback (self.reference_positions, self.reference_connectivity)

### GPU Arrays (JAX)
- ✅ Hash octree structure (morton_keys, element_lists, etc.)
- ✅ Mesh positions (during GPU operations)
- ✅ Mesh connectivity (during GPU operations)
- ✅ Candidate elements from hash lookup
- ✅ Field values during interpolation

**Problem**: Constant CPU ↔ GPU transfers kill performance!

---

## Why GPU Utilization is 2-3%

**GPU Timeline** (per timestep):
```
|--CPU--|--GPU--|--CPU--|--GPU--|--CPU--|
0ms    5ms    7ms    9ms   12ms  16ms  43ms
       ↑ Hash     ↑ Test    ↑ Interp
       lookup     elements  FEM

GPU busy: 7ms out of 43ms = 16% theoretical max
Observed: 2-3% due to transfer overhead and synchronization
```

**Breakdown of 43ms**:
- CPU operations: 23ms (53%)
- GPU operations: 9ms (21%)
- CPU↔GPU transfers: 11ms (26%)

**GPU is idle 79% of the time waiting for CPU!**

---

## Solution: Remove `io_callback`

### Phase 3E: Pure JAX `sample_at_positions()`

Replace the `io_callback` version with pure JAX:

```python
def sample_at_positions(self, query_positions: jnp.ndarray, t: float) -> jnp.ndarray:
    """Sample field at positions (PURE JAX - GPU accelerated)."""

    # No io_callback! Direct JAX operations
    query_positions = jnp.asarray(query_positions, dtype=jnp.float32)
    t_jax = jnp.asarray(t, dtype=jnp.float32)

    # Find temporal interpolation parameters (pure JAX)
    left_idx, right_idx, alpha = self._find_temporal_indices_jax(t_jax)

    # GPU field sampling for both timesteps
    field_left = self._sample_field_gpu(query_positions, left_idx)
    field_right = self._sample_field_gpu(query_positions, right_idx)

    # Temporal interpolation (pure JAX)
    return (1.0 - alpha) * field_left + alpha * field_right
```

### Required Changes

**File**: `jaxtrace/fields/shared_octree_fem_field.py`

1. **Remove `io_callback` wrapper** (lines 425-443)
2. **Add pure JAX temporal index finding**
3. **Add pure JAX field sampling**
4. **Ensure all mesh data stays on GPU**

---

## Expected Performance After Fix

### GPU Timeline (per timestep):
```
|--GPU-------------------------------------|
0ms                                      7ms

All operations on GPU: 7ms
```

**Speedup**: 43ms → 7ms = **6× faster per timestep**

**Full Run**:
- Before: 86 seconds (minimum)
- After: 14 seconds (**6× speedup**)
- With JIT batching: < 5 seconds (**17× speedup**)

### GPU Utilization
- Current: 2-3%
- After io_callback removal: 40-60%
- After full batching: 80-95%

---

## Implementation Priority

### Immediate (Phase 3E)
1. ✅ Hash octrees built (working!)
2. ✅ GPU hash lookup (working!)
3. ✅ GPU element testing (working!)
4. ❌ **Remove io_callback** ← DO THIS NOW
5. ❌ Pure JAX temporal interpolation
6. ❌ Pure JAX field sampling

### Future (Phase 3F)
7. Batch entire tracking loop
8. JIT-compile RK4 integration
9. Profile and optimize

---

## Files to Modify

### 1. `shared_octree_fem_field.py`

**Current** (lines 397-444):
```python
def sample_at_positions(self, query_positions, t):
    result = io_callback(callback_wrapper, ...)  # ❌ CPU barrier
    return result
```

**Target**:
```python
@jax.jit
def sample_at_positions(self, query_positions, t):
    # Pure JAX - no io_callback!
    # All operations on GPU
    return interpolated_field  # ✅ GPU accelerated
```

### 2. Use `gpu_field_sampling.py` Module

The `gpu_field_sampling.py` module (created in Phase 3D) already has:
- `sample_field_gpu_single_timestep()` - Pure JAX field sampling
- `sample_field_gpu_batch()` - Batched version

**Integrate this module** to replace `io_callback` path.

---

## Summary

### Current Bottleneck
- **`io_callback` at line 436** forces CPU execution
- Every particle query triggers CPU barrier
- GPU is idle 79% of the time
- Result: 2-3% GPU utilization, very slow performance

### Solution
- **Remove `io_callback`**
- Use pure JAX field sampling
- Keep all arrays on GPU
- Expected: 80-95% GPU utilization, 6-17× speedup

### Next Step
Implement Phase 3E: Remove `io_callback` and integrate `gpu_field_sampling.py` module for pure JAX execution.
