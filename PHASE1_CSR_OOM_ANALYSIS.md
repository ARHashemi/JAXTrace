# Phase 1 CSR Implementation - OOM Analysis and Resolution

## Problem Summary

CSR-style hash bucket search causes persistent 14 GB OOM errors when used with JAX vmap, despite multiple fix attempts.

## Error Details

```
E1209 17:54:48 gpu_hlo_schedule.cc:817] The byte size of input/output arguments (14105734144) exceeds the base limit
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 14049536000 bytes.
```

**Error location**: `level2b_heavy_csr.py:117` in `search_bucket_elements_csr`

**Context**: Called from `initial_assignment.py:421` via `jax.vmap` over particle batch

## Root Cause Analysis

### Calculation
- 14 GB ÷ 250 particles = 56 MB per particle
- Expected memory: ~200 elements × 4 bytes × 4 nodes × 3 coords = ~10 KB per particle
- **Actual is 5,600× larger than expected!**

### JAX XLA Compilation Issue

The CSR implementation uses complex nested control flow that JAX's XLA compiler cannot optimize:

```python
def search_bucket_elements_csr(...):
    # Dynamic slice (size depends on max_bucket_size)
    bucket_elements = jax.lax.dynamic_slice(sorted_elements, (bucket_start,), (max_bucket_size,))

    # Nested inside jax.lax.cond
    def search_bucket(_):
        # Nested inside lax.fori_loop
        def check_one_element(i):
            elem_id = bucket_elements[i]
            node_ids = connectivity[elem_id]  # Indexing inside loop
            tet_nodes = node_positions[node_ids]
            ...

        found_elem = jax.lax.fori_loop(0, max_bucket_size, loop_body, init)

    return jax.lax.cond(actual_size > 0, search_bucket, empty_bucket, None)
```

**When vmapped over 250 particles**, JAX XLA:
1. Cannot fold the dynamic_slice with variable `bucket_start` and `bucket_end`
2. Materializes intermediate buffers for ALL loop iterations
3. Creates separate compilation artifacts for each particle
4. Results in 14 GB memory allocation

### Failed Fix Attempts

1. **Remove jnp.arange() and batched indexing** ✗
   - Thought: Batched operations create copies
   - Fix: Fetch node data one element at a time in fori_loop
   - Result: Still OOM (problem is XLA compilation, not runtime operations)

2. **Fix vmap in_axes** ✗
   - Thought: mesh arrays being duplicated across vmap
   - Fix: `in_axes=(0, None, None, ...)` to share mesh data
   - Result: Still OOM (problem is inside function, not vmap args)

3. **Reduce batch size 250 → 32** ⏳ **TESTING NOW**
   - Thought: Reduce memory by 8× to fit in GPU
   - Fix: `BATCH_SIZE = 32` in `initial_assignment.py:377`
   - Result: Test running...

## Why CSR is Incompatible with JAX vmap

### JAX Optimization Requirements
JAX XLA compiler needs:
- Static shapes for all intermediate arrays
- Predictable memory access patterns
- No data-dependent control flow

### CSR Violates These Requirements
- `dynamic_slice` with variable offsets → cannot statically determine shapes
- `lax.cond` + `lax.fori_loop` nesting → complex control flow graph
- Loop iteration count `max_bucket_size` can be large (500-1000) → XLA unrolls or materializes

### Padded Arrays Work Better
Original padded implementation:
```python
bucket_elements = hash_bucket_elements[bucket_id]  # (max_elem_per_bucket,) static shape
valid_mask = bucket_elements >= 0  # Fixed-size mask
inside_flags = jax.vmap(check_element)(bucket_elements)  # Simple vmap, no dynamic slicing
```

This has:
- ✓ Static shapes (padding ensures fixed size)
- ✓ No dynamic slicing
- ✓ Simpler control flow

## Resolution Options

### Option 1: Reduce Batch Size (CURRENT)
**Pros**: Simple, keeps CSR implementation
**Cons**: Slower throughput (8× more batches)
**Status**: Testing with `BATCH_SIZE = 32`

### Option 2: Revert to Padded Arrays
**Pros**: Proven to work, no OOM
**Cons**: 19% more memory usage (acceptable)
**Implementation**:
1. Revert `initial_assignment.py` to use `hash_bucket.py` (padded)
2. Revert `test_octree_vs_blockwise_initialization.py` to use `build_hash_bucket_arrays`
3. Keep CSR code for future Phase 2 (octree integration)

### Option 3: Hybrid Approach
**Pros**: Best of both worlds
**Cons**: Complex, two code paths
**Implementation**:
1. Use padded arrays for initial assignment (Phase 1 & 3)
2. Use CSR only for Phase 2 octree (different search pattern, no heavy vmap)

### Option 4: Abandon CSR, Wait for Phase 2 Octree
**Pros**: Octree will give 90% memory reduction anyway (vs 19% from CSR)
**Cons**: Delays memory optimization
**Rationale**: Phase 2 octree is the real solution, CSR is a small incremental improvement

## Recommended Path Forward

### **Immediate (Phase 1 testing)**:
1. **Try batch size 32** ← Current test running
   - If works: Proceed with CSR, accept slower throughput
   - If fails: Revert to padded arrays

### **Short-term (Phase 1 completion)**:
2. **If batch=32 fails, revert to padded arrays**
   - Change: `hash_bucket.py` → `build_hash_bucket_arrays()` (padded)
   - Change: `level2b_heavy.py` → `search_level2b_hash_bucket()` (padded)
   - Keep CSR code for reference

3. **Complete Phase 1 validation**
   - Test initial assignment accuracy
   - Measure memory usage
   - Document performance

### **Medium-term (Phase 2 priority)**:
4. **Focus on Phase 2 Octree Implementation**
   - Per-block flat octree (depth 6-8)
   - Expected: 90% memory reduction (vs 19% from CSR)
   - Expected: 10× L2b speedup
   - Octree uses different search pattern (tree traversal, not vmap)

### **Long-term (Phase 3)**:
5. **Vectorize entire initial assignment**
   - Replace Python loops with single vmap over ALL particles
   - 100-500× speedup expected
   - At this point, batch size won't matter (single kernel launch)

## Key Insights

1. **CSR memory savings (19%) are not worth the complexity** when octree gives 90%
2. **JAX vmap + dynamic control flow = OOM** unless very carefully structured
3. **Padded arrays are more JAX-friendly** despite memory overhead
4. **Phase 2 octree is the real solution** for both memory and performance

## Testing Status

**Batch size 32 test**: ✗ **FAILED** - Still OOM with 429 MB allocation
- Reduced from 14 GB to 429 MB (32× reduction)
- Still 13.4 MB per particle (way too high)
- Conclusion: CSR fundamentally incompatible with JAX vmap

**Resolution**: ✓ **Reverted to padded arrays** (hash_bucket.py, level2b_heavy.py)
- Pragmatic solution that works
- Accept 19% higher memory usage (acceptable cost)
- Focus on Phase 2 octree (90% savings) instead

## Files Modified in This Investigation

1. `jaxtrace/gpu/search/level2b_heavy_csr.py` - CSR search implementation (3 iterations)
2. `jaxtrace/gpu/search/initial_assignment.py` - vmap in_axes, batch size reduction
3. `jaxtrace/gpu/search/__init__.py` - CSR exports
4. `test_octree_vs_blockwise_initialization.py` - CSR builder usage

## Next Steps Based on Test Result

### If Batch=32 Succeeds:
- ✓ Document CSR as working but slow
- ✓ Complete Phase 1 testing
- → Proceed to Phase 2 octree

### If Batch=32 Fails:
- ✗ Revert to padded arrays (`hash_bucket.py`, `level2b_heavy.py`)
- ✓ Complete Phase 1 with padded version
- → Skip directly to Phase 2 octree (90% savings)

---

**Date**: 2025-12-09
**GPU**: 4GB VRAM
**JAX Version**: Latest (2024+)
**Status**: Investigating OOM with batch size reduction
