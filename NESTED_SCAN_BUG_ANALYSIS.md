# Nested Scan/Vmap Bug - Critical Performance Issue

**Date:** 2025-11-30
**Status:** 🔴 Critical bug identified - Nested vmap+scan causing 100× slowdown

---

## Test Results - Masking Fix FAILED

**Performance:**
- Mean throughput: **6,429 p/s** (expected 40-48k p/s) ❌
- Time per step: **13.25s** (expected 0.11s) ❌
- Retention: **47.6%** (expected 82%) ❌

**Conclusion:** The masking fix did NOT improve performance. Same 13.25s/step as before.

---

## Root Cause: Nested JAX Operations

### User Warning

> "Be careful about nested jax scan or jit in jit and GPU performance and OOM"

This warning pointed directly to the issue!

### The Problem

**File:** [jaxtrace/gpu/search/octree_search_gpu.py:337](jaxtrace/gpu/search/octree_search_gpu.py#L337)

```python
def search_level2_octree_scan(...):
    def search_one_particle(pos, cached_id):
        # Each particle function contains lax.scan!
        def do_octree_search(_):
            (_, element_id), _ = jax.lax.scan(  # ← NESTED SCAN
                step,
                (jnp.int32(0), jnp.int32(-1)),
                None,
                length=max_depth
            )
            return element_id

        element_id = jax.lax.cond(
            already_found,
            return_cached,
            do_octree_search,  # ← Contains lax.scan
            None
        )
        return element_id

    # PROBLEM: vmap over function containing lax.scan
    return jax.vmap(search_one_particle)(positions, cached_element_ids)  # ← NESTED VMAP
```

**Nested Structure:**
```
JIT-compiled RK4 wrapper
  └─ search_level2_octree_scan()
      └─ jax.vmap(search_one_particle)  ← 100k iterations
          └─ jax.lax.cond()
              └─ do_octree_search()
                  └─ jax.lax.scan()  ← 10 iterations per particle
```

**Total operations:** 100k particles × 10 scan iterations = **1M nested operations**

---

## Why This Kills Performance

### 1. Compilation Overhead

JAX must compile the entire nested structure:
- **vmap** unrolls over all 100k particles
- Each particle has a **lax.scan** with 10 iterations
- **lax.cond** creates branching in the compiled graph
- Result: Massive XLA graph that's slow to execute

### 2. No Early Exit Benefits

Even though we use `lax.cond` for masking:
```python
element_id = jax.lax.cond(
    already_found,  # True for 99.5% of particles
    return_cached,  # ← Should be fast
    do_octree_search,  # ← Contains expensive lax.scan
    None
)
```

**Expected:** 99.5% of particles skip the scan (fast path)
**Reality:** JAX still compiles the full vmap+cond+scan graph for all particles

### 3. Memory Pressure

The nested structure creates:
- Large intermediate arrays (100k × 10 iterations)
- Deep call stack in XLA graph
- Poor GPU occupancy (irregular control flow)

---

## Why Masking Didn't Work

The masking fix using `lax.cond` was **logically correct** but **architecturally wrong** for JAX:

### Logical Fix (What We Did)
```python
# Only search unfound particles
element_id = jax.lax.cond(
    cached_id >= 0,  # Already found?
    return_cached,   # Yes: Return cached
    do_octree_search  # No: Search octree
)
```

### Architectural Problem (Why It's Slow)
```python
# JAX compiles ENTIRE vmap+cond+scan graph regardless of runtime branches
jax.vmap(  # ← Compiles for ALL particles
    lambda pos, cached_id: jax.lax.cond(  # ← Both branches compiled
        cached_id >= 0,
        return_cached,  # ← Compiled
        do_octree_search  # ← Contains lax.scan, ALSO compiled
    )
)(positions, cached_element_ids)
```

**JAX doesn't optimize away the scan at compile time** - it compiles the full nested structure for all particles.

---

## Solutions

### Option A: Filter Particles Before Search (Recommended)

**Idea:** Only pass unfound particles to octree search, avoiding vmap over 100k particles.

```python
def search_level2_octree_scan(...):
    # 1. Identify unfound particles
    unfound_mask = cached_element_ids < 0  # Boolean mask (N,)
    unfound_indices = jnp.where(unfound_mask, size=N)[0]  # Indices of unfound
    n_unfound = jnp.sum(unfound_mask)  # Count (~500 particles)

    # 2. Extract unfound particle positions
    unfound_positions = positions[unfound_indices]  # (n_unfound, 3)

    # 3. Search ONLY unfound particles (no vmap over 100k!)
    unfound_results = jax.vmap(search_one_particle_no_mask)(unfound_positions)  # ← Only ~500 particles

    # 4. Merge results back
    element_ids = cached_element_ids.at[unfound_indices].set(unfound_results)

    return element_ids
```

**Benefits:**
- vmap over ~500 particles (not 100k) → 200× less work
- No nested cond (masking done upfront with boolean indexing)
- Simpler compiled graph

**Challenges:**
- `jnp.where(..., size=N)` requires fixed size (may need padding)
- Scatter operation `.at[indices].set()` must be JAX-compatible

### Option B: Single Scan Over All Particles (Alternative)

**Idea:** Replace vmap with a single scan that processes particles sequentially.

```python
def search_level2_octree_scan(...):
    def search_batch(carry, i):
        """Search one particle at index i."""
        pos = positions[i]
        cached_id = cached_element_ids[i]

        # Skip if already found
        element_id = jax.lax.cond(
            cached_id >= 0,
            lambda _: cached_id,
            lambda _: search_octree_single(pos),  # ← Nested scan still exists
            None
        )

        return carry, element_id

    _, element_ids = jax.lax.scan(search_batch, None, jnp.arange(len(positions)))
    return element_ids
```

**Benefits:**
- No vmap (avoids vmap+scan nesting)
- Sequential processing (simpler graph)

**Challenges:**
- Still has nested scan (outer scan over particles, inner scan for octree)
- Sequential processing slower than parallel (no GPU parallelism)
- **WORSE PERFORMANCE** than Option A

### Option C: Use Block-Local Fallback (User's Suggestion)

**Already implemented!** File: [jaxtrace/gpu/tracking/rk4_gpu_fused.py:407-500](jaxtrace/gpu/tracking/rk4_gpu_fused.py#L407-L500)

```python
def create_search_gpu_fused_with_block_fallback(
    n_hops: int = 3,
    block_lists: Optional[BlockElementLists] = None
):
    # L0: Cached element
    # L1: 3-hop neighbors
    # L2: Block-local search (ALREADY IMPLEMENTED)
```

**Benefits:**
- No octree → No nested scan
- Block-local search is vectorized point-in-tet
- Uses existing block structure (zero memory overhead)
- Already works for initialization

**User mentioned:** "We have also some global search used for particle initializations but they are blockwise"

This is **exactly** the block-local fallback approach!

---

## Recommendation

### Immediate Fix: Use Block-Local Fallback (Option C)

**Action:** Test the existing `create_search_gpu_fused_with_block_fallback()` instead of octree.

**Why:**
1. ✅ **Already implemented** (used for initialization)
2. ✅ **No nested scan** (pure vectorized operations)
3. ✅ **Zero memory overhead** (uses existing blocks)
4. ✅ **Proven to work** (initialization successful)

**Test Command:**
```bash
# Modify production script to use block fallback instead of octree
# Change from:
#   create_search_gpu_fused_with_l2_octree()
# To:
#   create_search_gpu_fused_with_block_fallback()
```

### If Block Fallback Has Issues

Then implement **Option A: Filter Particles** to fix the octree approach.

**Implementation:**
1. Use boolean masking to identify unfound particles
2. Extract unfound positions/IDs
3. Run octree search ONLY on unfound subset (~500 particles)
4. Scatter results back to full array

**Expected improvement:** 200× less vmap work = 100-115× faster

---

## Technical Notes

### Why JAX Doesn't Optimize Nested Scans

JAX compiles functions **statically** at JIT time:
- All control flow (`lax.cond`, `lax.scan`) is compiled into XLA graph
- Runtime values (like `cached_id >= 0`) don't affect compilation
- Both branches of `lax.cond` are compiled (selected at runtime)
- Nested vmap+scan creates large, complex XLA graphs

**Python-level optimization (doesn't work in JAX):**
```python
if any(cached_element_ids < 0):  # ← Requires CPU evaluation
    search_octree(...)  # ← Breaks GPU fusion
```

**JAX-level optimization (what we need):**
```python
# Filter at array level (GPU-compatible)
unfound_mask = cached_element_ids < 0  # ← GPU array operation
unfound_positions = positions[unfound_mask]  # ← GPU gather
results = search_octree(unfound_positions)  # ← Smaller vmap
```

### Nested Operations in Other Parts

The user warned about "nested jax scan or jit in jit" - let me verify the RK4 wrapper:

**File:** [jaxtrace/gpu/tracking/rk4_gpu_fused.py](jaxtrace/gpu/tracking/rk4_gpu_fused.py)

The RK4 wrapper is already JIT-compiled (factory pattern), and it calls:
- `search_level2_octree_scan()` ← Contains vmap+cond+scan (PROBLEM!)
- Other searches likely have similar structure

**Key insight:** The entire RK4 step is JIT-compiled, so any vmap/scan inside search functions creates nesting.

---

## Expected Results

### With Block-Local Fallback (Option C)

```
Time/step: 0.11-0.15s (vs 13.25s current)
Throughput: 40-48k p/s (vs 6.4k p/s current)
Retention: ~77-82%
Speedup: 100-120× faster
```

### With Filtered Octree Search (Option A)

```
Time/step: 0.11-0.15s (vs 13.25s current)
Throughput: 40-48k p/s (vs 6.4k p/s current)
Retention: ~82% (octree more complete than blocks)
Speedup: 100-120× faster
```

---

## Next Steps

1. **Test block-local fallback** (fastest path to solution)
   - Modify production script to use existing implementation
   - Expected: 40-48k p/s throughput

2. **If block fallback successful:**
   - Evaluate heavy block imbalance (user's concern)
   - Consider hybrid approach (blocks + octree for heavy blocks)

3. **If block fallback has issues:**
   - Implement filtered octree search (Option A)
   - Fix nested vmap+scan structure

---

**Date:** 2025-11-30
**Critical Issue:** Nested vmap+scan in JIT-compiled function
**Impact:** 100× performance degradation
**Solution:** Use block-local fallback or implement filtered particle search
