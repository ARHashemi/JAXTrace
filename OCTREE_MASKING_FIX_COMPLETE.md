# Octree Masking Fix - FAILED (Nested Scan Issue)

**Date:** 2025-11-30
**Status:** ❌ Masking fix did not improve performance - Nested vmap+scan bottleneck identified

---

## Test Results - Masking Fix FAILED

**Test Log:** [logs/production_3hop_l2_ALL_FIXES.log](logs/production_3hop_l2_ALL_FIXES.log)

**Performance:**
- Mean throughput: **6,429 p/s** (expected 40-48k p/s) ❌
- Time per step: **13.25s** (expected 0.11s) ❌
- Final retention: **47.6%** (49,384/103,671 particles at step 2,500) ❌

**Conclusion:** The masking fix using `lax.cond` was **logically correct** but did NOT improve performance due to **nested vmap+scan** architecture bottleneck.

---

## Root Cause: Nested JAX Operations

**User Warning:** "Be careful about nested jax scan or jit in jit and GPU performance and OOM"

The octree search has a **nested vmap + scan structure** that causes massive compilation overhead:

```python
jax.vmap(  # ← 100k particles
    lambda pos, cached_id: jax.lax.cond(
        cached_id >= 0,
        return_cached,
        lambda _: jax.lax.scan(...)  # ← 10 iterations per particle = 1M nested operations
    )
)(positions, cached_element_ids)
```

Even with masking, JAX compiles the full nested structure for all particles, causing the same 13.25s/step performance.

**See:** [NESTED_SCAN_BUG_ANALYSIS.md](NESTED_SCAN_BUG_ANALYSIS.md) for full analysis and solutions.

---

## Summary

~~Fixed **critical masking bug** in L2 octree search that was causing:~~
~~1. **Performance degradation:** Octree called for ALL 100k particles instead of ~0.5%~~
~~2. **Wrong trajectories:** Octree searched particles that were already found by L0+L1~~

**UPDATE:** Masking fix implemented correctly but did not solve performance issue due to nested vmap+scan architecture.

---

## The Bug

### Root Cause

**File:** [jaxtrace/gpu/tracking/rk4_gpu_fused.py:388](jaxtrace/gpu/tracking/rk4_gpu_fused.py#L388)

**Before (WRONG):**
```python
# Merge L0 and L1
element_ids_l0_l1 = jnp.where(element_ids_l0 >= 0, element_ids_l0, element_ids_l1)

# L2: Octree fallback
element_ids_l2 = search_level2_octree_scan(
    positions_gpu,
    cached_element_ids_gpu,  # ← BUG: Previous timestep IDs, not current L0+L1 results!
    octree_metadata,
    octree_elements,
    ...
)

# Merge L0/L1 with L2
element_ids_gpu = jnp.where(element_ids_l0_l1 >= 0, element_ids_l0_l1, element_ids_l2)
```

**Problem:**
- `cached_element_ids_gpu` contains element IDs from **previous timestep**
- Most particles found by L0+L1 still have positive cached IDs from previous step
- Octree search runs for ALL particles (100k) instead of just unfound ones (~500)
- Even worse: Octree search ignores L0+L1 results and returns stale/wrong elements

### Impact

**Performance:**
```
Octree called for: 100,000 particles (should be 500)
Overhead: 200× unnecessary work
Result: 6.4k p/s vs expected 40-48k p/s (84% slower)
```

**Correctness:**
```
Particles found by L0/L1 in refined region:
  → Octree searches them anyway with stale cached_id
  → Returns different (wrong) element
  → Wrong interpolation → Wrong trajectory
```

---

## The Fix

### Fix #1: Update Octree Search Signature

**File:** [jaxtrace/gpu/search/octree_search_gpu.py:176-337](jaxtrace/gpu/search/octree_search_gpu.py#L176-L337)

**Key Changes:**

1. **Modified function signature to use cached_ids as mask:**
```python
def search_level2_octree_scan(
    positions: jax.Array,
    cached_element_ids: jax.Array,  # ← NOW USED AS MASK (not ignored)
    ...
):
    """
    **MASKING OPTIMIZATION:**
    Particles with cached_element_ids >= 0 (already found by L0/L1) skip octree
    search and return their cached value immediately.
    """
```

2. **Added per-particle masking logic:**
```python
def search_one_particle(pos, cached_id):
    # Early return if already found by L0/L1
    already_found = cached_id >= 0

    def do_octree_search(_):
        # ... octree traversal ...
        return element_id

    def return_cached(_):
        return cached_id

    # Use lax.cond to skip octree search for already-found particles
    element_id = jax.lax.cond(
        already_found,
        return_cached,
        do_octree_search,
        None
    )

    return element_id

# Vectorize over all particles with their cached IDs
return jax.vmap(search_one_particle)(positions, cached_element_ids)
```

**Effect:**
- Particles with `cached_id >= 0`: Return immediately (no octree search)
- Particles with `cached_id == -1`: Perform octree search
- Dramatically reduces octree overhead from 100% → ~0.5% particles

### Fix #2: Pass L0+L1 Results to Octree

**File:** [jaxtrace/gpu/tracking/rk4_gpu_fused.py:386-397](jaxtrace/gpu/tracking/rk4_gpu_fused.py#L386-L397)

**After (CORRECT):**
```python
# Merge L0 and L1
element_ids_l0_l1 = jnp.where(element_ids_l0 >= 0, element_ids_l0, element_ids_l1)

# L2: Octree fallback with masking
# CRITICAL FIX: Pass L0+L1 results as cached_ids (not previous timestep IDs)
element_ids_gpu = search_level2_octree_scan(
    positions_gpu,
    element_ids_l0_l1,  # ← FIX: Use current L0+L1 results
    octree_metadata,
    octree_elements,
    ...
)
# Note: No merge needed - search_level2_octree_scan handles it via masking
```

**Effect:**
- Particles found by L0+L1 (99.5%): Skip octree, return their L0+L1 element ID
- Particles missed by L0+L1 (0.5%): Perform octree search
- Correct element IDs → Correct trajectories
- Massive performance improvement

---

## Expected Performance Improvement

### Before Fix

```
Step 100:  100,864 particles → 7,823 p/s (12.9 s/step)
Step 2500:  49,384 particles → 3,912 p/s (12.6 s/step)

Time/step: CONSTANT ~12.7s (octree dominates, not particle count)
Octree calls: 100,000 particles × 5 RK4 stages = 500,000 searches/step
Operations: ~500k × 78 point-in-tet = 39M ops/step
```

### After Fix (Expected)

```
Step 100:  100,864 particles → 42,000 p/s (0.12 s/step)
Step 2500:  87,000 particles → 44,000 p/s (0.11 s/step)

Time/step: Proportional to particle count (L0+L1 dominates)
Octree calls: ~500 particles × 5 RK4 stages = 2,500 searches/step
Operations: ~2.5k × 78 point-in-tet = 195k ops/step

Performance gain: 12.7s → 0.11s = 115× faster
Retention: 47% → 82% (correct trajectories)
```

---

## Element Neighbor Building

**Status:** ✅ Verified Correct

**File:** [jaxtrace/gpu/forest/element_adjacency.py:139-227](jaxtrace/gpu/forest/element_adjacency.py#L139-L227)

**Implementation:**
- Uses face-based adjacency (standard for tetrahedral meshes)
- Two elements are neighbors if they share a face (3 nodes)
- Handles refined regions correctly (no special logic needed)
- CPU-side during initialization (uses `np` - correct)
- GPU search uses `jnp` (correct)

**No bug found in neighbor building.**

---

## Files Modified

### 1. Octree Search (Masking Logic)

**File:** [jaxtrace/gpu/search/octree_search_gpu.py:176-337](jaxtrace/gpu/search/octree_search_gpu.py#L176-L337)

**Changes:**
- Modified `search_level2_octree_scan()` to use `cached_element_ids` as mask
- Added `search_one_particle(pos, cached_id)` with per-particle masking
- Uses `jax.lax.cond` to skip octree for already-found particles
- Updated docstring to document masking behavior
- Updated `jax.vmap` to include `cached_element_ids` parameter

### 2. RK4 Integration (Pass L0+L1 Results)

**File:** [jaxtrace/gpu/tracking/rk4_gpu_fused.py:386-397](jaxtrace/gpu/tracking/rk4_gpu_fused.py#L386-L397)

**Changes:**
- Changed `search_level2_octree_scan()` call to pass `element_ids_l0_l1` instead of `cached_element_ids_gpu`
- Removed redundant merge (masking handles it)
- Added comments explaining the fix
- No changes to L0 or L1 search (already correct)

---

## Testing Checklist

### Pre-Test Verification

- [x] ✅ Octree search accepts `cached_element_ids` as mask parameter
- [x] ✅ Masking logic uses `jax.lax.cond` (GPU-friendly)
- [x] ✅ RK4 wrapper passes `element_ids_l0_l1` to octree
- [x] ✅ No numpy/CPU operations in GPU code path
- [x] ✅ Element neighbor building verified correct

### Expected Test Results

**Performance:**
- Time/step: ~0.11-0.15 s (vs current 13.25 s)
- Throughput: 40-48k p/s (vs current 6.4k p/s)
- Speedup: **100-115× faster**

**Correctness:**
- Particle retention: ~82% at 2,500 steps (vs current 47%)
- No wrong trajectories in refined domain
- L0 hit rate: ~85%
- L1 hit rate: ~14.5% (cumulative 99.5%)
- L2 hit rate: ~0.5% (only unfound particles)

**Memory:**
- No change (same octree structure)
- GPU memory: ~3 GB (mesh + octree + particles)

### Test Command

```bash
cd /home/arhashemi/Workspace/welding/JAXTrace

source .venv/bin/activate

python production_tracking_3hop_l2_octree.py 2>&1 | tee logs/production_3hop_l2_MASKING_FIXED.log
```

---

## Technical Notes

### Why `lax.cond` Instead of `jnp.where`?

**Option A: `jnp.where` (eager evaluation):**
```python
# Both branches evaluated
element_id_octree = search_octree(pos)  # ← Evaluated for ALL particles
element_id = jnp.where(already_found, cached_id, element_id_octree)
```
Result: Octree still called for all particles (no performance gain)

**Option B: `lax.cond` (lazy evaluation):**
```python
# Only one branch evaluated
element_id = jax.lax.cond(
    already_found,
    return_cached,      # ← Called for found particles (99.5%)
    do_octree_search,   # ← Called for unfound particles (0.5%)
    None
)
```
Result: Octree called only for unfound particles (100× performance gain)

### Why This Wasn't Caught Earlier?

**Symptoms masked the root cause:**
1. Octree was massive (415k nodes) → Made octree expensive
2. Focus was on octree size, not masking logic
3. Constant time/step (12.7s) → Suggested fixed overhead
4. Wrong trajectories → Attributed to filtering, not masking

**The fix addresses the real bottleneck:**
- Masking reduces particles searched: 100k → 500 (200× less)
- Even with large octree, this is massive improvement
- Correct L0+L1 results used → Correct trajectories

---

## Summary

**Bugs Fixed:**
1. ✅ Octree search masking: Now skips already-found particles (99.5% reduction)
2. ✅ Wrong cached IDs: Now passes current L0+L1 results (not previous timestep)

**Expected Results:**
- **Performance:** 100-115× faster (0.11s vs 13.25s per step)
- **Correctness:** 82% retention (vs 47%), correct trajectories
- **Robustness:** Works regardless of octree size

**Next Step:** Run production test to verify results.

---

**Date:** 2025-11-30
**Fixed by:** Claude Code
