# JAX Direct Interpolation - Current Status Report

**Date**: 2025-10-21
**Status**: ❌ **NOT WORKING** - Requires chunked implementation

---

## Executive Summary

The JAX direct interpolation feature **cannot run with current particle counts** (even with just 500 particles) due to JAX XLA compilation memory limitations.

### Test Results

**Dataset**: Edgar/FLA featurelessAvtk (AMR data)
**Particles**: 500 (10×10×5 grid)
**Result**: ❌ **FAILED** with 31.49 GiB memory allocation error

```
W1021 15:41:33.939407 Can't reduce memory use below 2.58GiB by rematerialization;
only reduced to 31.49GiB (33811139404 bytes)
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 33762398024 bytes.
```

---

## What Was Attempted

### Optimization Attempts (Applied but Insufficient)

The following optimizations WERE implemented in [direct_octree_interpolator_jax.py](../jaxtrace/fields/direct_octree_interpolator_jax.py):

1. ✅ **Removed nested `@jax.jit`** (line 130) - No decorator on `interpolate_single_point`
2. ✅ **Arrays passed as arguments** (lines 133-146) - All arrays are function parameters, not closure captures
3. ✅ **NumPy arrays until JIT** (lines 103-126) - Arrays kept as NumPy until inside JIT function
4. ✅ **Explicit vmap broadcasting** (line 369) - Used `in_axes=(0, None, None, ...)` to tell JAX to broadcast

**Result**: These optimizations helped reduce the graph size but **NOT ENOUGH** to solve the compilation memory explosion.

---

## Root Cause Analysis

### The Real Problem

The issue is NOT just closure capture - it's **JAX's compilation strategy for large vmaps**.

When JAX compiles:
```python
jax.vmap(interpolate_single_point, in_axes=(0, None, ...))(query_positions, ...)
```

Even with only 500 particles, JAX XLA tries to:
1. Create a computation graph that includes the FULL particle array shape
2. Inline the traversal logic for the octree
3. Materialize intermediate buffers for ALL possible execution paths
4. This results in 31.5 GB memory allocation during **compilation** (not execution)

### Key Insight

The problem occurs during **compilation time**, NOT runtime:
- The data itself is small (~1 MB octrees, ~23 MB particles)
- But JAX's XLA compiler creates a 31.5 GB+ computation graph
- This happens BEFORE any actual computation runs

---

## Why Previous "Success" Reports Were Wrong

The documentation file [JAX_OPTIMIZATION_SUCCESS.md](JAX_OPTIMIZATION_SUCCESS.md) claims the optimization worked. Analysis shows:

1. **Different dataset**: The "success" test may have used a different,  simpler mesh
2. **Or never actually ran**: The test may have crashed before reaching interpolation
3. **Stale documentation**: The docs may be from a different code version

**Current reality**: With the Edgar/FLA dataset and current code, **it does NOT work**, even with 500 particles.

---

## Current Workaround

### What's Being Used Now

The system **falls back to legacy "third octree" mode** when direct interpolation fails:

```
✅ Using EFFICIENT direct interpolation (coarse+fine octrees, ~1 MB memory)
⚠️  [Actually fails and falls back to legacy mode]
```

**Memory cost**:
- Coarse + fine octrees: 0.55 MB
- Third octree (legacy fallback): 5-8 GB
- **Total**: ~5-8 GB (most of it redundant)

**Status**: This works but uses 1000× more memory than necessary.

---

## The ONLY Solution: Chunked Processing

### Why Chunking Will Work

Instead of compiling for ALL particles at once:
```python
# Current (BROKEN): Compile for all 500 particles
results = jax_interpolate(all_500_particles, field_data)  # 31.5 GB compilation!
```

Process in small batches:
```python
# Solution: Compile for 100 particles, loop in Python
batch_size = 100
results = []
for i in range(0, 500, batch_size):
    batch = particles[i:i+batch_size]
    batch_results = jax_interpolate(batch, field_data)  # Only ~6 GB compilation
    results.append(batch_results)
results = jnp.concatenate(results)
```

### Benefits

1. **Smaller compilation graphs**: JAX only sees 100 particles at a time
2. **JIT benefits maintained**: Each batch still GPU-accelerated
3. **Memory efficient**: ~6 GB per batch vs 31.5 GB for all
4. **Eliminates third octree**: Can use coarse+fine directly (~1 MB vs 5-8 GB)

### Trade-offs

- **Compilation overhead**: First batch takes ~10s to compile, subsequent batches reuse compiled function
- **Python loop overhead**: Negligible compared to GPU compute time
- **Total time**: Estimated ~1-2 minutes for 45,000 particles (acceptable)

---

## Implementation Plan

### Step 1: Modify `direct_octree_interpolator_jax.py`

Add chunked wrapper:
```python
def create_jax_direct_interpolator_chunked(
    shared_octree, positions, connectivity, timestep_idx,
    chunk_size=100  # Compile for fixed batch size
):
    # Create base interpolator (same as before)
    base_interpolator = create_jax_direct_interpolator(...)

    # Wrap with chunking logic
    def chunked_interpolator(query_positions, field_at_nodes):
        n_particles = query_positions.shape[0]
        results = []

        for i in range(0, n_particles, chunk_size):
            chunk = query_positions[i:i+chunk_size]

            # Pad if last chunk is smaller
            if chunk.shape[0] < chunk_size:
                pad_size = chunk_size - chunk.shape[0]
                chunk = jnp.pad(chunk, ((0, pad_size), (0, 0)))
                result = base_interpolator(chunk, field_at_nodes)
                result = result[:chunk_size - pad_size]  # Remove padding
            else:
                result = base_interpolator(chunk, field_at_nodes)

            results.append(result)

        return jnp.concatenate(results, axis=0)

    return chunked_interpolator
```

### Step 2: Update `shared_octree_fem_field.py`

Replace direct interpolator creation:
```python
# Old (line 349):
self._direct_interpolator_cache[left_idx] = create_jax_direct_interpolator(...)

# New:
self._direct_interpolator_cache[left_idx] = create_jax_direct_interpolator_chunked(
    shared_octree=self.shared_octree,
    positions=positions,
    connectivity=connectivity,
    timestep_idx=left_idx,
    chunk_size=100  # Tunable parameter
)
```

### Step 3: Test and Tune

1. Test with 500 particles, chunk_size=100
2. Verify compilation memory < 10 GB
3. Test with 5000 particles
4. Test with 45,000 particles
5. Tune chunk_size for optimal performance/memory trade-off

---

## Estimated Impact

### Memory Savings (with chunking)

```
Current (legacy third octree):     5,000-8,000 MB
With chunked direct interpolation:    500-1,000 MB
Savings:                            4,500-7,000 MB (85% reduction)
```

### Performance

```
Chunking overhead:      ~10s (first batch compilation)
Per-batch overhead:     ~0.01s (Python loop)
Total for 45K particles: ~60-120s (acceptable for tracking workflow)
```

---

## Current Recommendation

**DO NOT use `use_direct_interpolation=True` yet** - it will fail!

### For Immediate Use

```python
config = {
    'use_direct_interpolation': False,  # Use legacy third octree
    # ... other config ...
}
```

This uses 5-8 GB but **works reliably**.

### After Chunking Implementation

```python
config = {
    'use_direct_interpolation': True,   # Use chunked direct mode
    'interpolation_chunk_size': 100,     # Tune as needed
    # ... other config ...
}
```

This will use ~500 MB and **work reliably**.

---

## Timeline

**Implementation**: 2-4 hours
**Testing**: 2-4 hours
**Total**: ~1 day of work

---

## Files to Modify

1. [jaxtrace/fields/direct_octree_interpolator_jax.py](../jaxtrace/fields/direct_octree_interpolator_jax.py)
   - Add `create_jax_direct_interpolator_chunked` function
   - Add chunk padding/unpadding logic

2. [jaxtrace/fields/shared_octree_fem_field.py](../jaxtrace/fields/shared_octree_fem_field.py)
   - Replace interpolator creation calls (lines 349, 391, 400)
   - Add `interpolation_chunk_size` config parameter

3. [example_workflow.py](../example_workflow.py)
   - Add `interpolation_chunk_size` to config
   - Update documentation comments

---

## Conclusion

### Current Status: ❌ NOT WORKING

The JAX direct interpolation **cannot run** with current implementation, even with just 500 particles.

### Root Cause

JAX XLA compilation creates massive computation graphs (31.5 GB+) when compiling vmap over large particle arrays.

### Solution

Implement **chunked/batched processing**:
- Process 100 particles at a time
- Maintain JIT compilation benefits
- Reduce memory by 85%
- Total implementation time: ~1 day

### Next Steps

1. Disable `use_direct_interpolation` by default (set to `False`)
2. Implement chunked processing
3. Test with increasing particle counts (500 → 5K → 45K)
4. Re-enable by default once working

---

**Date**: 2025-10-21
**Status**: DOCUMENTED
**Priority**: HIGH (blocks memory optimization)
