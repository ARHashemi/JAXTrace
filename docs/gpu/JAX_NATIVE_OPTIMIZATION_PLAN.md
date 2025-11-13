# JAX-Native Optimization Plan: GPU Multi-Level Search
**Addressing Performance Bottlenecks with Memory-Safe JAX Patterns**

**Date**: 2025-11-10
**Status**: PLANNING
**Branch**: `gpu_native_implementation`
**Context**: Phase 4 multi-level search functionally correct but 54-330× slower than target

---

## Executive Summary

**Problem**: Python `for` loops in particle search prevent GPU parallelization
**Current Performance**: 150-186 p/s (multi-level), 8 p/s (initial assignment)
**Target Performance**: 10,000+ p/s
**Root Cause**: Serial CPU execution instead of JAX vmap vectorization
**Solution**: Refactor to JAX-native patterns WITH memory-safe implementations

---

## Critical Analysis: Current vs JAX-Native Implementation

### Current Implementation Pattern (BOTTLENECK)

**File**: `jaxtrace/gpu/search/multi_level_search.py:188-299`

```python
# PROBLEM: Python for loop - serial CPU execution
for i in range(n_particles):  # ❌ Serial, cannot use GPU parallelism
    pos = positions_jax[i]
    cached_elem = int(cached_element_ids[i])

    # L0: Cached element
    elem_id = search_level0_cached(pos, cached_elem, ...)
    if int(elem_id) >= 0:  # ❌ Python if/else
        element_ids[i] = int(elem_id)
        continue  # ❌ Python continue

    # L1: Neighbor elements
    if cached_elem >= 0:  # ❌ Python if/else
        elem_id = search_level1_neighbors(pos, ...)
        if int(elem_id) >= 0:
            element_ids[i] = int(elem_id)
            continue

    # ... continues with L2, L3 in same pattern
```

**Why This Is Slow**:
1. **Serial execution**: Processes ONE particle at a time on CPU
2. **No GPU utilization**: GPU sits idle while Python loop executes
3. **Repeated JIT overhead**: Each iteration triggers JIT function call overhead
4. **CPU-GPU transfers**: Data transfers on every iteration

**Performance Impact**:
- Time per particle: ~5-6 ms (including Python overhead)
- GPU utilization: <1% (idle)
- Actual throughput: 150-186 particles/s

---

## Proposed Solutions: Three Strategies

### Strategy 1: Pure JAX vmap with lax.cond (HIGHEST RISK)

**⚠️ MEMORY EXPLOSION RISK**: `lax.cond` pre-compiles BOTH branches!

```python
@jax.jit
def search_single_particle_with_laxcond(position, cached_elem, ...):
    # L0: Try cached
    elem_L0 = search_level0_cached(position, cached_elem, ...)

    # L1: Conditional execution with lax.cond
    elem_L1 = jax.lax.cond(
        elem_L0 >= 0,
        lambda: elem_L0,  # Branch 1: Found, return cached result
        lambda: search_level1_neighbors(position, ...)  # Branch 2: Search neighbors
    )

    # ... L2, L3 with more lax.cond
```

**Memory Analysis**:
```
For 1M particles with nested lax.cond:
- Each lax.cond pre-compiles BOTH branches
- L0+L1+L2+L3 = 4 levels of lax.cond
- Total branches compiled: 2^4 = 16 code paths
- Estimated memory: 500 MB - 2 GB (compilation overhead)
```

**Risk Assessment**: ⚠️ **HIGH RISK** for large particle batches (>10K)

---

### Strategy 2: Unconditional Execution with Masking ✅ RECOMMENDED

**Key Insight**: Execute ALL search levels, use masking to select first found result

```python
@jax.jit
def search_single_particle_masked(position, cached_elem, cached_block,
                                   padded_elements, elem_neighbors,
                                   block_neighbors_26, heavy_flags,
                                   node_positions, connectivity):
    """
    Execute all search levels unconditionally, mask to select first hit.

    NO lax.cond → NO memory explosion risk
    Uses jnp.where for masking instead of conditional execution
    """
    # Execute ALL levels unconditionally (in parallel on GPU)
    result_L0 = search_level0_cached(position, cached_elem, node_positions, connectivity)
    result_L1 = search_level1_neighbors(position, cached_elem, elem_neighbors,
                                        node_positions, connectivity)
    result_L2 = search_level2_dispatch(position, cached_block, heavy_flags[cached_block],
                                       padded_elements, node_positions, connectivity)
    result_L3 = search_level3_neighbor_blocks(position, cached_block, block_neighbors_26,
                                              heavy_flags, padded_elements,
                                              node_positions, connectivity)

    # Create result array: [L0, L1, L2, L3, not_found]
    candidates = jnp.array([result_L0, result_L1, result_L2, result_L3, -1], dtype=jnp.int32)

    # Create mask: which levels found a valid element (>= 0)?
    valid_mask = candidates >= 0  # [True if found, False if -1]

    # Select first valid result (priority: L0 > L1 > L2 > L3 > -1)
    # Method: Find first True index, return corresponding candidate
    first_valid_idx = jnp.argmax(valid_mask)  # Returns index of first True (or 0 if all False)

    # Return selected result
    return candidates[first_valid_idx]  # Will be -1 if all searches failed

# Vectorize over ALL particles - Single GPU kernel!
results = jax.vmap(search_single_particle_masked)(
    positions,          # (n_particles, 3)
    cached_elem_ids,    # (n_particles,)
    cached_block_ids,   # (n_particles,)
    # Broadcast static arrays to all particles
    ...
)  # Returns: (n_particles,) element IDs
```

**Memory Analysis**:
```
For 1M particles:
- No lax.cond → No branch pre-compilation
- Executes 4 search functions per particle (L0-L3)
- Each search function is JIT-compiled once (not per particle)
- Memory: ~100-200 MB (intermediate results)
```

**Performance Analysis**:
```
Cons:
- Wastes computation: Executes all 4 levels even if L0 succeeds
- Worst case: 4× redundant work

Pros:
- GPU parallelism: ALL particles processed simultaneously
- No Python loop overhead
- No conditional branching overhead
- Expected 100-300× speedup vs Python loop
- Net result: 25-75× faster than current (even with wasted work)
```

**Risk Assessment**: ✅ **LOW RISK** - Predictable memory usage

---

### Strategy 3: Hybrid Batching with Iterative Refinement ✅ BEST PERFORMANCE

**Key Insight**: Process particles in batches, filter out found particles after each level

```python
def multi_level_search_batch_iterative(
    particle_positions,      # (n_particles, 3)
    cached_element_ids,      # (n_particles,)
    cached_block_ids,        # (n_particles,)
    ...
):
    """
    Iterative multi-level search with batch refinement.

    Strategy:
        1. Try L0 on ALL particles (vmap)
        2. Filter to only particles not found in L0
        3. Try L1 on remaining particles (vmap)
        4. Filter again, continue with L2, L3

    Advantage: No wasted computation, only search unfound particles at each level
    """
    n_particles = len(particle_positions)

    # Initialize results
    element_ids = jnp.full(n_particles, -1, dtype=jnp.int32)
    search_levels = jnp.full(n_particles, -1, dtype=jnp.int32)  # Track which level found it

    # Mask: which particles still need to be found?
    active_mask = jnp.ones(n_particles, dtype=jnp.bool_)  # All active initially

    # L0: Search cached elements for ALL particles (vmap)
    @jax.jit
    def search_L0_batch(positions, cached_elems, node_pos, connectivity):
        return jax.vmap(search_level0_cached)(positions, cached_elems, node_pos, connectivity)

    results_L0 = search_L0_batch(particle_positions, cached_element_ids,
                                 node_positions, connectivity)

    # Update: which particles were found in L0?
    found_L0 = results_L0 >= 0
    element_ids = jnp.where(found_L0, results_L0, element_ids)  # Update found elements
    search_levels = jnp.where(found_L0, 0, search_levels)       # Mark as found in L0
    active_mask = active_mask & ~found_L0                       # Deactivate found particles

    # L1: Search neighbors for REMAINING particles only
    # Extract only active particles (not found in L0)
    active_indices = jnp.where(active_mask)[0]  # Indices of particles still needing search

    if jnp.sum(active_mask) > 0:  # Only proceed if there are unfound particles
        active_positions = particle_positions[active_indices]
        active_cached_elems = cached_element_ids[active_indices]

        @jax.jit
        def search_L1_batch(positions, cached_elems, elem_neighbors, node_pos, connectivity):
            return jax.vmap(search_level1_neighbors)(positions, cached_elems,
                                                     elem_neighbors, node_pos, connectivity)

        results_L1 = search_L1_batch(active_positions, active_cached_elems,
                                     element_neighbors, node_positions, connectivity)

        # Update results for active particles
        found_L1 = results_L1 >= 0
        element_ids = element_ids.at[active_indices].set(
            jnp.where(found_L1, results_L1, element_ids[active_indices])
        )
        search_levels = search_levels.at[active_indices].set(
            jnp.where(found_L1, 1, search_levels[active_indices])
        )
        active_mask = active_mask.at[active_indices].set(
            active_mask[active_indices] & ~found_L1
        )

    # L2: Search blocks for REMAINING particles
    # (Similar pattern to L1)

    # L3: Search neighbor blocks for REMAINING particles
    # (Similar pattern to L1)

    return element_ids, search_levels
```

**Memory Analysis**:
```
For 1M particles:
- L0: Process 1M particles (all) → ~50 MB
- L1: Process ~100K particles (5-15% not found in L0) → ~5 MB
- L2: Process ~10K particles (1-3% not found in L0+L1) → ~0.5 MB
- L3: Process ~1K particles (0.1-1% not found in L0-L2) → ~50 KB

Total memory: ~60 MB (vs 100-200 MB for Strategy 2)
```

**Performance Analysis**:
```
Pros:
- No wasted computation (only search unfound particles)
- Full GPU parallelism at each level
- Expected 200-500× speedup vs Python loop

Cons:
- More complex implementation
- Requires multiple vmap calls (4 levels)
- Slight overhead from mask indexing
```

**Risk Assessment**: ✅ **LOW RISK** - Most memory-efficient

---

## Comparison of Strategies

| Aspect | Strategy 1 (lax.cond) | Strategy 2 (Masked) | Strategy 3 (Iterative) |
|--------|----------------------|-------------------|----------------------|
| **Memory Risk** | ⚠️ HIGH (2-4 GB) | ✅ LOW (100-200 MB) | ✅ LOWEST (60 MB) |
| **Implementation Complexity** | Medium | Low | High |
| **Wasted Computation** | None | High (4× worst case) | None |
| **Expected Speedup** | 100-500× | 25-75× | 200-500× |
| **Code Maintainability** | Medium | High | Low |
| **Risk of Bugs** | Medium | Low | Medium |
| **Recommendation** | ❌ Avoid (memory risk) | ✅ PHASE 1 (safe, simple) | ✅ PHASE 2 (optimal) |

---

## Recommended Implementation Plan

### Phase 1: Implement Strategy 2 (Masked Execution) ✅ SAFE

**Why Start Here**:
- ✅ Low risk: No lax.cond memory explosion
- ✅ Simple to implement: Single vmap, no complex control flow
- ✅ Easy to verify: Direct comparison with current results
- ✅ Immediate gains: 25-75× speedup vs current

**Files**:
1. `jaxtrace/gpu/search/multi_level_search_v2.py` (NEW)
2. `jaxtrace/gpu/search/initial_assignment_v2.py` (NEW)

**Implementation Steps** (4-5 hours):
1. Create `search_single_particle_masked()` function (2 hours)
2. Test with small batch (100 particles) to verify correctness (1 hour)
3. Benchmark vs current implementation (1 hour)
4. Integrate into comprehensive test (30 min)
5. Document performance gains (30 min)

**Deliverables**:
- Multi-level search: 3,000-10,000 p/s (target met!)
- Initial assignment: 200-600 p/s (partial target met)
- Memory: <200 MB
- GPU utilization: 60-80%

### Phase 2: Implement Strategy 3 (Iterative Refinement) ✅ OPTIMAL

**Why Second**:
- ✅ Build on Phase 1 success
- ✅ Eliminate wasted computation
- ✅ Achieve peak performance

**Files**:
1. `jaxtrace/gpu/search/multi_level_search_v3.py` (NEW)
2. Test against Phase 1 implementation

**Implementation Steps** (6-8 hours):
1. Implement iterative refinement with masking (3 hours)
2. Optimize mask indexing and filtering (2 hours)
3. Benchmark vs Phase 1 (1 hour)
4. Integration testing (1 hour)
5. Documentation (1 hour)

**Deliverables**:
- Multi-level search: 10,000-20,000 p/s (exceeds target!)
- Initial assignment: 1,000-5,000 p/s (exceeds target!)
- Memory: <100 MB
- GPU utilization: 80-95%

---

## Memory Safety Analysis

### lax.cond Memory Explosion Risk

**Problem**: JAX pre-compiles BOTH branches of `lax.cond`

```python
# Example with nested lax.cond (4 levels):
elem = lax.cond(
    cond1,
    lambda: result1,
    lambda: lax.cond(
        cond2,
        lambda: result2,
        lambda: lax.cond(
            cond3,
            lambda: result3,
            lambda: result4
        )
    )
)
```

**Compiled Code Paths**: 2^4 = 16 different execution paths
**Memory Overhead**: Each path requires compilation buffer
**For 1M particles**: 16 paths × 1M particles × JIT overhead = **2-4 GB**

**Hardware Constraint**: NVIDIA T1000 has 4 GB VRAM
**Risk**: Memory overflow → crash or severe slowdown

### Strategy 2 & 3 Memory Safety

**No lax.cond**: Uses only `jnp.where` and `jax.vmap`
**Predictable Memory**: Linear with batch size, no branching explosion
**Tested Pattern**: L3 neighbor search already uses this pattern successfully

---

## Implementation Details

### Strategy 2: Masked Execution (RECOMMENDED FOR PHASE 1)

**Full Implementation**:

```python
# File: jaxtrace/gpu/search/multi_level_search_v2.py

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
import time

from .level0_cached import search_level0_cached
from .level1_neighbors import search_level1_neighbors
from .level2a_light import search_level2a_light_block
from .level2b_heavy import search_level2b_hash_bucket
from .level3_neighbor_blocks import search_level3_neighbor_blocks

jax.config.update("jax_enable_x64", True)


@jax.jit
def search_level2_dispatch(
    position: jax.Array,
    block_id: int,
    is_heavy: bool,
    # Light block data
    padded_block_elements: jax.Array,
    padded_block_counts: jax.Array,
    # Heavy block data (pass empty arrays if not heavy)
    hash_bucket_elements: jax.Array,
    hash_bucket_counts: jax.Array,
    hash_bucket_neighbors: jax.Array,
    n_buckets: int,
    morton_bits: int,
    block_bounds: jax.Array,
    # Mesh data
    node_positions: jax.Array,
    connectivity: jax.Array
) -> int:
    """
    Dispatch to L2a (light) or L2b (heavy) based on block classification.

    Uses jnp.where instead of lax.cond to avoid memory explosion.
    """
    # Execute BOTH searches (GPU can parallelize this)
    result_light = search_level2a_light_block(
        position, block_id,
        padded_block_elements[block_id],
        padded_block_counts[block_id],
        node_positions, connectivity
    )

    result_heavy = search_level2b_hash_bucket(
        position, block_id,
        hash_bucket_elements,
        hash_bucket_counts,
        hash_bucket_neighbors,
        n_buckets, morton_bits, block_bounds,
        node_positions, connectivity
    )

    # Select based on is_heavy flag (no branching, just masking)
    return jnp.where(is_heavy, result_heavy, result_light)


@jax.jit
def search_single_particle_masked(
    position: jax.Array,          # (3,)
    cached_elem_id: int,
    cached_block_id: int,
    # Block classification
    heavy_flags: jax.Array,        # (n_blocks,) bool
    # Padded arrays (light blocks)
    padded_block_elements: jax.Array,  # (n_blocks, max_elem)
    padded_block_counts: jax.Array,    # (n_blocks,)
    # Element neighbors
    element_neighbors: jax.Array,  # (n_elements, max_neighbors)
    # Block neighbors
    block_neighbors_26: jax.Array, # (n_blocks, 26)
    # Hash buckets (for heavy blocks - pass dummy if none)
    # ... hash bucket arrays ...
    # Mesh data
    node_positions: jax.Array,     # (n_nodes, 3)
    connectivity: jax.Array        # (n_elements, 4)
) -> int:
    """
    Search for particle using masked execution (NO lax.cond).

    Executes ALL search levels, selects first valid result.
    Memory-safe: No branching explosion.
    """
    # Execute ALL levels unconditionally
    result_L0 = search_level0_cached(
        position, cached_elem_id, node_positions, connectivity
    )

    result_L1 = search_level1_neighbors(
        position, cached_elem_id, element_neighbors,
        node_positions, connectivity
    )

    result_L2 = search_level2_dispatch(
        position, cached_block_id, heavy_flags[cached_block_id],
        padded_block_elements, padded_block_counts,
        # ... hash bucket data ...
        node_positions, connectivity
    )

    result_L3 = search_level3_neighbor_blocks(
        position, cached_block_id,
        block_neighbors_26[cached_block_id],
        heavy_flags,
        padded_block_elements,
        padded_block_counts,
        node_positions, connectivity
    )

    # Collect results: [L0, L1, L2, L3, not_found]
    candidates = jnp.array([result_L0, result_L1, result_L2, result_L3, -1],
                          dtype=jnp.int32)

    # Find first valid result (>= 0)
    valid_mask = candidates >= 0
    first_valid_idx = jnp.argmax(valid_mask)  # Index of first True

    return candidates[first_valid_idx]


def multi_level_search_batch_v2(
    particle_positions: np.ndarray,      # (n_particles, 3)
    cached_element_ids: np.ndarray,      # (n_particles,)
    cached_block_ids: np.ndarray,        # (n_particles,)
    block_classification,                # BlockClassification object
    padded_block_elements: np.ndarray,   # (n_blocks, max_elem)
    padded_block_counts: np.ndarray,     # (n_blocks,)
    element_neighbors: np.ndarray,       # (n_elements, max_neighbors)
    block_neighbors_26: np.ndarray,      # (n_blocks, 26)
    hash_bucket_data,                    # Dict[int, HashBucketArrays]
    node_positions: np.ndarray,          # (n_nodes, 3)
    connectivity: np.ndarray,            # (n_elements, 4)
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Multi-level search using JAX vmap with masked execution.

    Strategy: Execute all levels, select first valid result.
    Memory-safe: No lax.cond branching.

    Returns:
        element_ids: (n_particles,) found element IDs
        block_ids: (n_particles,) block IDs where found
        stats: Performance statistics
    """
    n_particles = len(particle_positions)

    if verbose:
        print(f"\nMulti-level search (JAX vmap): {n_particles:,} particles")

    # Convert to JAX arrays
    positions_jax = jnp.array(particle_positions, dtype=jnp.float32)
    cached_elems_jax = jnp.array(cached_element_ids, dtype=jnp.int32)
    cached_blocks_jax = jnp.array(cached_block_ids, dtype=jnp.int32)

    # Build heavy block flags array (once, not per particle!)
    n_blocks = len(padded_block_counts)
    heavy_flags = jnp.zeros(n_blocks, dtype=jnp.bool_)
    for hb_id in block_classification.heavy_blocks:
        heavy_flags = heavy_flags.at[hb_id].set(True)

    # Convert mesh data to JAX
    node_pos_jax = jnp.array(node_positions, dtype=jnp.float32)
    connectivity_jax = jnp.array(connectivity, dtype=jnp.int32)
    elem_neighbors_jax = jnp.array(element_neighbors, dtype=jnp.int32)
    padded_elements_jax = jnp.array(padded_block_elements, dtype=jnp.int32)
    padded_counts_jax = jnp.array(padded_block_counts, dtype=jnp.int32)
    block_neighbors_jax = jnp.array(block_neighbors_26, dtype=jnp.int32)

    # TODO: Prepare hash bucket arrays (if heavy blocks exist)

    # VECTORIZE over ALL particles (single GPU kernel!)
    start_time = time.time()

    element_ids_jax = jax.vmap(search_single_particle_masked)(
        positions_jax,        # (n_particles, 3)
        cached_elems_jax,     # (n_particles,)
        cached_blocks_jax,    # (n_particles,)
        heavy_flags,          # (n_blocks,) - broadcasted to all particles
        padded_elements_jax,  # (n_blocks, max_elem) - broadcasted
        padded_counts_jax,    # (n_blocks,) - broadcasted
        elem_neighbors_jax,   # (n_elements, max_neighbors) - broadcasted
        block_neighbors_jax,  # (n_blocks, 26) - broadcasted
        # ... hash bucket data ...
        node_pos_jax,         # (n_nodes, 3) - broadcasted
        connectivity_jax      # (n_elements, 4) - broadcasted
    )

    # Force computation (block until GPU completes)
    element_ids_jax.block_until_ready()
    total_time = time.time() - start_time

    # Convert back to numpy
    element_ids = np.array(element_ids_jax, dtype=np.int32)

    # Compute statistics
    n_found = np.sum(element_ids >= 0)
    n_not_found = n_particles - n_found
    throughput = n_particles / total_time if total_time > 0 else 0

    if verbose:
        print(f"\nResults:")
        print(f"  Found: {n_found:,}/{n_particles:,} ({100*n_found/n_particles:.1f}%)")
        print(f"  Time: {total_time:.2f} s")
        print(f"  Throughput: {throughput:.0f} particles/s")

    stats = {
        'n_particles': n_particles,
        'n_found': n_found,
        'n_not_found': n_not_found,
        'total_time': total_time,
        'particles_per_second': throughput
    }

    # Estimate block IDs (simplified - use cached for now)
    block_ids = cached_block_ids.copy()

    return element_ids, block_ids, stats
```

### Testing Strategy

**Unit Tests** (`tests/gpu/test_multi_level_search_v2.py`):

```python
def test_masked_execution_correctness():
    """Verify masked execution matches legacy implementation."""
    # Generate test case
    # ...

    # Run legacy (Python loop) version
    elem_ids_legacy, _, _ = multi_level_search_batch(...)

    # Run new (JAX vmap) version
    elem_ids_v2, _, _ = multi_level_search_batch_v2(...)

    # Compare results
    assert np.array_equal(elem_ids_legacy, elem_ids_v2), \
        "JAX vmap version must match legacy results exactly"


def test_masked_execution_performance():
    """Verify performance improvement."""
    # ...

    # Benchmark legacy
    start = time.time()
    multi_level_search_batch(...)
    time_legacy = time.time() - start

    # Benchmark new
    start = time.time()
    multi_level_search_batch_v2(...)
    time_v2 = time.time() - start

    speedup = time_legacy / time_v2

    print(f"Speedup: {speedup:.1f}×")
    assert speedup > 20, f"Expected >20× speedup, got {speedup:.1f}×"
```

---

## Deliverables

### Phase 1 (Strategy 2)
1. ✅ `jaxtrace/gpu/search/multi_level_search_v2.py` (JAX vmap with masking)
2. ✅ `jaxtrace/gpu/search/initial_assignment_v2.py` (JAX vmap with masking)
3. ✅ Unit tests for correctness and performance
4. ✅ Integration with comprehensive test
5. ✅ Performance report: 25-75× speedup documented
6. ✅ Memory usage report: <200 MB verified

### Phase 2 (Strategy 3) - OPTIONAL
1. ✅ `jaxtrace/gpu/search/multi_level_search_v3.py` (Iterative refinement)
2. ✅ Benchmark vs Phase 1
3. ✅ Performance report: 200-500× speedup documented
4. ✅ Memory usage report: <100 MB verified

---

## Timeline

**Phase 1 (Strategy 2)**: 4-5 hours
- Implementation: 2 hours
- Testing: 1 hour
- Benchmarking: 1 hour
- Documentation: 30 min

**Phase 2 (Strategy 3)**: 6-8 hours (OPTIONAL)
- Implementation: 3 hours
- Optimization: 2 hours
- Testing: 1 hour
- Benchmarking: 1 hour
- Documentation: 1 hour

**Total**: 4-13 hours (depending on whether Phase 2 is needed)

---

## Success Criteria

**Phase 1 (MINIMUM)**:
- ✅ Multi-level search: >3,000 particles/s (20× current)
- ✅ Initial assignment: >200 particles/s (25× current)
- ✅ Memory usage: <200 MB
- ✅ GPU utilization: >60%
- ✅ 100% correctness match with legacy

**Phase 2 (OPTIMAL - if Phase 1 insufficient)**:
- ✅ Multi-level search: >10,000 particles/s
- ✅ Initial assignment: >1,000 particles/s
- ✅ Memory usage: <100 MB
- ✅ GPU utilization: >80%

---

## Risk Mitigation

### Memory Explosion Prevention
- ✅ **NO lax.cond** in hot paths
- ✅ Pre-allocate all arrays with known sizes
- ✅ Monitor GPU memory during tests (`nvidia-smi`)
- ✅ Test with increasing batch sizes: 100 → 1K → 10K → 100K → 1M

### Correctness Validation
- ✅ Unit tests compare with legacy implementation
- ✅ Regression tests on ThreadedA mesh
- ✅ Verify hit rates match expected (L0: 85-95%, L1: 3-10%, etc.)

### Performance Validation
- ✅ Benchmark on realistic workload (ThreadedA, 1K-10K particles)
- ✅ Profile with JAX profiler if performance targets not met
- ✅ Check GPU utilization with `nvidia-smi` during execution

---

## Alignment with Original Plan

From `FINAL_EXECUTABLE_PLAN.md`:

> **Phase 4 Success Criteria**:
> - ✅ Total found: >98%
> - ✅ Memory: <500 MB for 1M particles
> - ✅ Performance: >10× speedup vs CPU
> - ✅ **Expected: > 10,000 particles/second**

**Current Status**: 150-186 p/s (54-67× BELOW target)
**After Phase 1**: 3,000-10,000 p/s (MEETS target!)
**After Phase 2**: 10,000-20,000 p/s (EXCEEDS target!)

---

## Next Steps

1. **Get approval** on Strategy 2 (masked execution) as Phase 1 approach
2. **Implement Phase 1**: JAX vmap with unconditional execution + masking
3. **Validate** correctness and performance
4. **Decide on Phase 2**: Only if Phase 1 doesn't meet 10,000 p/s target

---

**END OF PLAN**
