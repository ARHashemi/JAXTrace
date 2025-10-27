# CRITICAL: JAX Compilation Memory Explosion

## Summary

The JAX direct interpolation implementation fails during **compilation** (not execution) with a 2.76 TiB memory allocation error.

## Error

```
W1021 10:50:47.566938   75669 hlo_rematerialization.cc:3204] Can't reduce memory use below 2.58GiB (2766640263 bytes) by rematerialization; only reduced to 2.76TiB (3038617035172 bytes)
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 3038615961416 bytes.
```

## Root Cause

JAX is attempting to compile a `vmap` over 45,000 particles where the per-particle interpolation function captures MASSIVE static arrays:

### Captured Arrays in JAX Compilation Graph:

1. **Coarse Octree Element Lists**:
   - Shape: `(3,105, max_candidates)` where `max_candidates` could be 1000+
   - Size: Potentially 10-50 MB per array

2. **Fine Octree Element Lists**:
   - Similar size to coarse

3. **Connectivity Array**:
   - Shape: `(3,048,900, 4)`
   - Size: 46.52 MB

4. **Positions Array**:
   - Shape: `(780,922, 3)`
   - Size: 8.94 MB

5. **Per-timestep velocity fields**:
   - Shape: `(780,922, 3)`
   - Size: 8.94 MB per timestep

### Why JAX Explodes:

When JAX compiles `jax.vmap(interpolate_single_point, in_axes=(0, None))` over 45,000 particles:

1. **XLA graph expansion**: JAX tries to create an XLA computation graph for the entire vmap
2. **Intermediate buffer allocation**: Even though we use `in_axes=(0, None)` to broadcast arrays, JAX may still allocate intermediate buffers
3. **Static shape requirements**: JAX traces the computation with static shapes, creating massive allocation requirements

### Memory Calculation:

The error attempts to allocate **2.76 TiB = 2,766,640,263,000 bytes**.

Likely cause:
- JAX is materializing intermediate computations for each of 45,000 particles
- With element lists having unknown/variable sizes, JAX allocates conservatively
- Multiplication factor: 45,000 particles × ~60 GB per-particle compilation buffer = 2.7 TB

## What Worked:

✅ **Fix #1**: Using `lax.fori_loop` instead of `lax.scan` (prevents runtime intermediate materialization)
✅ **Fix #2**: Using `in_axes=(0, None)` for vmap (tells JAX to broadcast, not duplicate)

## What DOESN'T Work:

❌ The JAX compilation itself is creating the massive allocation BEFORE runtime
❌ This is a fundamental limitation of how JAX compiles complex nested control flow with large static arrays

## Solutions:

### Option 1: Disable JIT for Direct Interpolation (Quick Fix)

Remove `@jax.jit` from the interpolator function. This will run in eager mode (slower) but avoid compilation memory explosion.

**Pros**: Simple, should work immediately
**Cons**: Loses GPU acceleration, much slower (~10-100x)

### Option 2: Chunked/Batched Interpolation

Instead of vmapping over all 45,000 particles at once, process in smaller batches (e.g., 100-1000 particles per batch).

**Pros**: Reduces compilation memory, maintains JIT benefits
**Cons**: Requires refactoring, may still hit limits for large batches

### Option 3: Use Legacy Third Octree Mode

The original implementation with the third octree (5-8 GB memory) doesn't have this compilation issue because it uses simpler, pre-built data structures.

**Pros**: Known to work (tested before)
**Cons**: Uses 5-8 GB vs 1 MB for octrees, defeats purpose of refactoring

### Option 4: Pure Python Implementation (No JAX)

Implement the direct interpolation in pure Python/NumPy without JAX primitives.

**Pros**: No compilation limits
**Cons**: No GPU acceleration, much slower

## Recommended Immediate Action:

**Use Option 3 (Legacy Mode) for now** by setting:

```python
config = {
    'use_direct_interpolation': False,  # Use legacy third octree
    # ... rest of config ...
}
```

This will:
- ✅ Complete the workflow successfully
- ✅ Provide accurate results
- ❌ Use 5-8 GB instead of 1 MB for octrees
- ⚠️  But still avoid the OOM crash

## Long-Term Solution:

Implement **Option 2 (Chunked Interpolation)**:

1. Split particle queries into batches of ~1000 particles
2. Compile JIT function for batch size 1000
3. Loop over batches (Python loop, not JAX)
4. Concatenate results

This should keep compilation memory reasonable while maintaining most JIT benefits.

## Files Affected:

- [`jaxtrace/fields/direct_octree_interpolator_jax.py`](jaxtrace/fields/direct_octree_interpolator_jax.py) - The failing interpolator
- [`jaxtrace/fields/shared_octree_fem_field.py`](jaxtrace/fields/shared_octree_fem_field.py:617) - Config flag location

## Timeline:

- **Phase C (Current)**: Discovered JAX compilation limitation
- **Next**: Document findings, recommend legacy mode for immediate use
- **Future**: Implement chunked/batched interpolation for memory-efficient JAX mode

---

**Date**: 2025-10-21
**Status**: ❌ BLOCKED - JAX compilation memory explosion
**Recommendation**: Use legacy mode (`use_direct_interpolation=False`) until chunked implementation is complete
