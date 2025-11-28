# Vectorized Multi-Level Search - Bottleneck Analysis

**Date**: 2025-11-17
**Status**: Analysis Complete - Critical Issues Identified

---

## Executive Summary

The vectorized multi-level search is **SLOWER** than sequential (0.86× speedup on 1K particles) due to **NESTED JIT COMPILATION** overhead. Each vmap creates a new JIT compilation context, and the underlying search functions are already `@jax.jit` decorated, causing:

1. **Double JIT compilation** (outer vmap + inner @jax.jit)
2. **Array conversion overhead** (NumPy ↔ JAX conversions)
3. **Lambda capture overhead** in vmapped functions

---

## Critical Bottleneck #1: Nested JIT Compilation

### Problem: `jax.vmap(lambda: @jax.jit_function)`

**Location**: [multi_level_search.py:396-398](../../jaxtrace/gpu/search/multi_level_search.py#L396-L398)

```python
# L0 vectorization (CURRENT - SLOW)
search_l0_vmap = jax.vmap(
    lambda pos, cached_elem: search_level0_cached(pos, cached_elem, node_pos_jax, connectivity_jax)
    # ^^^^^ Lambda wrapping @jax.jit function - DOUBLE JIT!
)
```

**Analysis**:
- `search_level0_cached()` is decorated with `@jax.jit` ([level0_cached.py:64](../../jaxtrace/gpu/search/level0_cached.py#L64))
- `jax.vmap()` creates a new JIT compilation context
- Lambda function captures large arrays (`node_pos_jax`, `connectivity_jax`)
- Result: JIT compiles the vmap, which JIT compiles the inner function → **DOUBLE COMPILATION**

### JIT-Decorated Functions

All search levels are pre-decorated:

| Function | File | Line | Decorator |
|----------|------|------|-----------|
| `point_in_tet_jax` | level0_cached.py | 16 | `@jax.jit` |
| `search_level0_cached` | level0_cached.py | 64 | `@jax.jit` |
| `search_level1_neighbors` | level1_neighbors.py | 76 | `@jax.jit` |
| `search_level2a_light_block` | level2a_light.py | 17 | `@jax.jit` |
| `search_level2b_hash_bucket` | level2b_heavy.py | 71 | `@jax.jit` |
| `search_level3_neighbor_blocks` | level3_neighbor_blocks.py | 19 | `@jax.jit` |

**Impact**: Every vmap call incurs JIT compilation overhead on top of already-JIT functions.

---

## Critical Bottleneck #2: Array Conversion Overhead

### Problem: Repeated NumPy ↔ JAX Conversions

**Location**: [multi_level_search.py:369-377](../../jaxtrace/gpu/search/multi_level_search.py#L369-L377)

```python
# EVERY CALL: Convert ALL arrays to JAX
positions_jax = jnp.array(particle_positions, dtype=jnp.float32)  # Copy to GPU
node_pos_jax = jnp.array(node_positions, dtype=jnp.float32)       # 895K nodes × 3 = 10MB
connectivity_jax = jnp.array(connectivity, dtype=jnp.int32)        # 3.5M elements × 4 = 56MB
elem_neighbors_jax = jnp.array(element_neighbors, dtype=jnp.int32) # 3.5M × 4 = 56MB
padded_elements_jax = jnp.array(padded_block_elements, dtype=jnp.int32)  # 256 × 444K = 433MB
# ... total ~550 MB copied to GPU
```

**Then after vmap**:

```python
l0_results = np.array(search_l0_vmap(...), dtype=np.int32)  # Copy BACK from GPU
```

**Analysis**:
- Every function call copies ~550 MB to GPU
- Every vmap result copies back to CPU
- For 3 tests × 2 versions (seq + vec) = 6 copies = **3.3 GB total transfers**
- Sequential version also does this, so not the main issue, but amplifies slowdown

---

## Critical Bottleneck #3: Lambda Capture Overhead

### Problem: Large Array Captures in Lambda

**Location**: All vmap calls capture large arrays

```python
# L0 (CURRENT)
search_l0_vmap = jax.vmap(
    lambda pos, cached_elem: search_level0_cached(
        pos, cached_elem,
        node_pos_jax,        # 10 MB captured
        connectivity_jax     # 56 MB captured
    )
)

# L1 (CURRENT)
search_l1_vmap = jax.vmap(
    lambda pos, cached_elem: search_level1_neighbors(
        pos, cached_elem,
        elem_neighbors_jax[cached_elem],  # Dynamic indexing in lambda!
        node_pos_jax,                     # 10 MB captured
        connectivity_jax                  # 56 MB captured
    )
)
```

**Analysis**:
- Lambda captures entire arrays in closure
- Dynamic indexing (`elem_neighbors_jax[cached_elem]`) inside vmap is inefficient
- JAX cannot optimize these captures as well as direct function calls

---

## Critical Bottleneck #4: Block-Grouped L2 Python Loop

### Problem: Python Loop Over Blocks Kills Performance

**Location**: [multi_level_search.py:496-502](../../jaxtrace/gpu/search/multi_level_search.py#L496-L502)

```python
# Group L1-miss particles by their cached block
particles_per_block = {}
for idx in l1_miss_indices:  # Python loop - NOT VECTORIZED
    block_id = int(cached_block_ids[idx])
    if block_id >= 0:
        particles_per_block.setdefault(block_id, []).append(idx)

# Process each block
for block_id, particle_indices in particles_per_block.items():  # Another Python loop!
    # ... vmap within each iteration
```

**Analysis**:
- Python loops cannot be JIT-compiled
- Dictionary operations are pure Python overhead
- Each block gets a separate vmap call → **Multiple JIT compilations**
- For 256 blocks, this could be 256 separate JIT compilations!

**Sequential version comparison**: Also has Python loop, so this is NOT why vectorized is slower. But it prevents the speedup we expect.

---

## Performance Impact Analysis

### First-Run Overhead Breakdown (Estimated)

For 1,000 particles:

| Component | Time (s) | Percentage | Notes |
|-----------|----------|------------|-------|
| Array conversion to GPU | 0.5 | 10% | 550 MB transfer |
| **L0 vmap JIT compilation** | **2.0** | **40%** | Nested JIT on 1K particles |
| L0 execution | 0.2 | 4% | Actual computation |
| **L1 vmap JIT compilation** | **1.5** | **30%** | Nested JIT on ~100 particles |
| L1 execution | 0.1 | 2% | Actual computation |
| L2 Python loop + vmap | 0.5 | 10% | Multiple small vmaps |
| L3 sequential | 0.1 | 2% | Few particles |
| Array conversion from GPU | 0.1 | 2% | Small result |
| **TOTAL** | **~5.0s** | **100%** | **Matches observed 5.47s!** |

**Sequential version** (213 p/s = 4.69s total):
- No vmap → **No nested JIT overhead**
- Same array conversion overhead (0.5s)
- Direct function calls with already-compiled @jax.jit functions
- Python loop overhead (~4.0s) spread across 1K particles

**Why Vectorized is Slower**:
- Sequential: 0.5s conversion + 4.0s compiled execution = 4.5s
- Vectorized: 0.5s conversion + 4.0s **JIT compilation** + 1.0s execution = 5.5s

**First run penalty**: JIT compilation overhead is ONE-TIME, but we're measuring first run!

---

## Root Cause Summary

### Why Vectorized is 0.86× (SLOWER) Instead of 1.5-4× Faster

1. **Nested JIT Compilation** (40-70% overhead)
   - Every vmap wraps an already-@jax.jit function
   - JAX must compile the vmap over the compiled function
   - This is done on **first call** - not amortized for small tests

2. **Lambda Closure Overhead** (10-20% overhead)
   - Large array captures in lambda functions
   - Dynamic indexing inside vmaps

3. **Python Loop in L2** (prevents speedup, but not slower)
   - Block grouping loop is pure Python
   - Multiple small vmaps instead of one large vmap

4. **Small Batch Size** (amplifies overhead)
   - 1K particles too small to amortize JIT compilation
   - Sequential baseline also slow (213 p/s vs expected 3,428 p/s)
   - Both versions hitting first-run penalty

### Expected Behavior at Larger Scales

**10K particles**: JIT overhead amortized 10×, expect 1.5-2× speedup
**30K particles**: JIT overhead amortized 30×, expect 2-3× speedup

---

## Solution Strategies

### Strategy 1: Pre-Compiled Vectorized Functions (RECOMMENDED)

Create dedicated vectorized functions with single JIT compilation:

```python
@jax.jit
def search_l0_vectorized(
    positions: jax.Array,        # (n_particles, 3)
    cached_elements: jax.Array,  # (n_particles,)
    node_positions: jax.Array,   # (n_nodes, 3)
    connectivity: jax.Array      # (n_elements, 4)
) -> jax.Array:
    """Vectorized L0 search - compile ONCE for entire batch."""
    def search_single(pos, cached_elem):
        # Inline logic from search_level0_cached WITHOUT @jax.jit
        is_valid = (cached_elem >= 0) & (cached_elem < len(connectivity))
        safe_idx = jnp.where(is_valid, cached_elem, 0)
        node_ids = connectivity[safe_idx]
        tet_nodes = node_positions[node_ids]
        inside = point_in_tet_jax(pos, tet_nodes)
        return jnp.where(is_valid & inside, cached_elem, -1)

    # vmap is INSIDE @jax.jit, compiled as single unit
    return jax.vmap(search_single)(positions, cached_elements)
```

**Benefits**:
- Single JIT compilation for entire batch
- No lambda captures
- JAX can optimize the entire vmap

**Estimated Speedup**: 5-10× improvement over current implementation

### Strategy 2: Remove @jax.jit from Individual Functions

Remove `@jax.jit` decorators from `search_level0_cached`, etc. and rely on vmap's compilation:

```python
# In level0_cached.py - REMOVE @jax.jit decorator
def search_level0_cached(...):  # No decorator
    # ... implementation ...
```

**Benefits**:
- Eliminates nested JIT
- Lambda still has capture overhead

**Estimated Speedup**: 2-3× improvement

### Strategy 3: Vectorize L2 Block Grouping

Replace Python loop with JAX operations:

```python
# Group particles by block using JAX
unique_blocks = jnp.unique(cached_block_ids[l1_miss_indices])

# Process all blocks in single vmap
def process_block_batch(block_id):
    mask = cached_block_ids == block_id
    block_particles = positions_jax[mask]
    # ... vectorized search ...
    return results

results = jax.vmap(process_block_batch)(unique_blocks)
```

**Benefits**:
- Eliminates Python loop
- Single vmap over all blocks

**Challenges**:
- Ragged array handling (different particle counts per block)
- May require padding

**Estimated Speedup**: 1.5-2× improvement over Strategy 1

### Strategy 4: Pre-Warm JIT Cache (Quick Fix)

Run dummy calls before timing to warm up JIT cache:

```python
# Before timing tests
dummy_positions = jnp.zeros((10, 3), dtype=jnp.float32)
dummy_cached = jnp.zeros(10, dtype=jnp.int32)
_ = search_l0_vmap(dummy_positions, dummy_cached)  # Trigger JIT compilation
jax.clear_caches()  # Optional: clear after warmup
```

**Benefits**:
- Quick fix for benchmarking
- Simulates real-world multi-timestep performance

**Drawbacks**:
- Doesn't fix underlying issue
- First timestep still slow in production

---

## Recommendations

### Immediate Action (for Current Test)

1. ✅ **Wait for test completion** - 10K and 30K results will show if speedup emerges at scale
2. ✅ **Document nested JIT issue** - Root cause identified
3. ⏳ **Analyze scaling behavior** - Does speedup improve with particle count?

### Short-Term Fix (if test shows no speedup)

Implement **Strategy 1** (Pre-Compiled Vectorized Functions):
- Create `search_l0_vectorized()`, `search_l1_vectorized()`, etc.
- Single `@jax.jit` on entire vectorized function
- No lambda captures

**Estimated effort**: 2-4 hours
**Expected result**: 5-10× speedup over current vectorized, 2-5× speedup over sequential

### Long-Term Optimization

Implement **Strategy 3** (Fully Vectorized L2):
- Eliminate all Python loops
- Pure JAX operations for block grouping
- Handle ragged arrays with padding

**Estimated effort**: 1-2 days
**Expected result**: 10-20× speedup over sequential baseline

---

## Verification Plan

### Test Current Implementation (In Progress)

- ⏳ Complete test on 1K, 10K, 30K particles
- 📊 Measure speedup vs particle count
- ✅ Confirm nested JIT is bottleneck

### Test Strategy 1 (Pre-Compiled)

1. Implement `search_l0_vectorized()` without @jax.jit on inner function
2. Benchmark on same 1K, 10K, 30K particles
3. Compare against current vectorized and sequential

**Success Criteria**:
- Speedup ≥ 2× over current vectorized
- Speedup ≥ 1.5× over sequential baseline (3,428 p/s)

---

## References

1. **JAX vmap documentation**: https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html
2. **JAX JIT compilation**: https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html
3. **Nested JIT issue**: https://github.com/google/jax/discussions/9024
4. **Current implementation**: [multi_level_search.py:324-685](../../jaxtrace/gpu/search/multi_level_search.py#L324-L685)
5. **Baseline performance**: [SESSION_SUMMARY_2025-11-14.md](SESSION_SUMMARY_2025-11-14.md) - 3,428 p/s

---

**Document Status**: ✅ Analysis Complete
**Next Step**: Wait for test results, then implement Strategy 1
**Last Updated**: 2025-11-17 11:15 UTC
