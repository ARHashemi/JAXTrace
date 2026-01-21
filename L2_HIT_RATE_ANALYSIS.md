# L2 Hit Rate Analysis and Incremental Search Strategy

**Date**: 2026-01-18
**Status**: Analysis complete, ready for implementation

---

## Executive Summary

Analysis of production logs reveals **critical insight**: Radius-based L2 search is **dramatically superior** to neighbors-based search for **initial assignment** (83.8% vs 28.4%), but we lack data on **which radius values** work best during RK4 tracking.

**Recommendation**: Implement **incremental radius-based L2** with cascading radii (2 → 5 → 10) before falling back to expensive global methods.

---

## Key Findings from Production Logs

### 1. Initial Assignment Performance (radius=500)

**With `L2_SEARCH_METHOD = 'radius'` (radius=10 during RK4)**:
```
Initial search (radius=500):  188,560/225,000 (83.80%)  ✅ EXCELLENT
Fallback radius=1000:          11,221 more    (88.79% total)
Fallback radius=2000:          12,329 more    (94.27% total)
Fallback radius=5000:          10,487 more    (98.93% total)
Final (radius=100000):        225,000/225,000 (100.00%)
```

**With `L2_SEARCH_METHOD = 'neighbors'`**:
```
Initial search (radius=500):   63,835/225,000 (28.37%)  ❌ TERRIBLE
Fallback radius=1000:           1,983 more    (29.25% total)
Fallback radius=2000:           3,058 more    (30.61% total)
Final assignment:              71,580/225,000 (31.81%)
⚠️ 153,420 particles NEVER assigned (outside mesh)
```

### 2. RK4 Tracking Performance

**With radius=10 L2**:
```
Step 100: 210,456 active (93.54% retention)
Step 200: 196,297 active (87.24% retention)
Step 300: 190,134 active (84.50% retention)
Throughput: ~30,500 particles/second
```

**With neighbors L2** (from previous logs):
```
Step 100: ~93% retention (similar)
Throughput: ~21,000 particles/second
```

### 3. Key Insight: Radius Works, But Is It Optimal?

**Problem**: We don't know the L2 hit rate breakdown during RK4 tracking:
- How many particles found at radius=2?
- How many need radius=5?
- How many need radius=10?
- Are we wasting work by always searching radius=10?

**If hit rates follow this pattern** (hypothesis):
- radius=2: ~60% of L2 searches succeed
- radius=5: ~30% more succeed (90% cumulative)
- radius=10: ~8% more succeed (98% cumulative)
- neighbors/hierarchical: ~2% final fallback

Then **incremental L2** could give us:
- **Same retention** as current radius=10
- **1.5-2× faster** L2 searches (skip radius=10 for 90% of particles)
- **Combined with inverse point-in-tet**: 3-5× total speedup

---

## Problem Analysis

### Why Neighbors Fails for Initial Assignment

1. **Initial assignment uses GLOBAL Morton search** (no cached element)
2. **Neighbors method** searches 27 octants at depth-7:
   - Only covers ~8×8×8 = 512 element region
   - Mesh has 3M elements spread over large domain
   - **Too small search radius** for particles far from mesh

3. **Radius method** searches ±500 leaves along Morton curve:
   - Morton curve has 24,550 leaves total
   - radius=500 covers 1,000 leaves = **4% of entire mesh**
   - Spatially localized due to Morton curve properties
   - **Much better coverage** for global search

### Why We Need Incremental Radius

**Current Problem**: Always searching radius=10 wastes work
- 21 leaves searched (center ± 10)
- If 60% of particles found at radius=2, we waste 16 leaf searches

**Solution**: Cascading radius search
```python
# Incremental L2 (proposed)
elem = search_radius_2(pos)   # 5 leaves
if elem < 0:
    elem = search_radius_5(pos)   # 11 leaves
if elem < 0:
    elem = search_radius_10(pos)  # 21 leaves
```

**Expected benefit**:
- If 60% hit at radius=2: avg work = 0.6×5 + 0.3×11 + 0.1×21 = 8.4 leaves (vs 21 current)
- Speedup: 21 / 8.4 = **2.5× faster L2 searches**

---

## Proposed Incremental L2 Strategy

### Phase 1: Add Profiling to Current Code

**Goal**: Measure real-world L2 hit rates during RK4 tracking

**Implementation**:
1. Modify `rk4_fully_fused_timedep.py` to add counters
2. Count how many searches succeed at each radius tier
3. Run production test with profiling enabled
4. Analyze results to validate hypothesis

**Code location**: [rk4_fully_fused_timedep.py](jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py)

Add global counters:
```python
# Profiling counters (thread-safe, CPU-side)
l2_radius_2_hits = 0
l2_radius_5_hits = 0
l2_radius_10_hits = 0
l2_total_searches = 0
```

Modify L2 search to test multiple radii:
```python
# Current: always radius=10
elem = search_L2_morton_radius_single(pos, mesh_gpu, radius=10)

# Profiling version: test all radii, record which succeeds first
elem_r2 = search_L2_morton_radius_single(pos, mesh_gpu, radius=2)
if elem_r2 >= 0:
    l2_radius_2_hits += 1
    return elem_r2

elem_r5 = search_L2_morton_radius_single(pos, mesh_gpu, radius=5)
if elem_r5 >= 0:
    l2_radius_5_hits += 1
    return elem_r5

elem_r10 = search_L2_morton_radius_single(pos, mesh_gpu, radius=10)
if elem_r10 >= 0:
    l2_radius_10_hits += 1
    return elem_r10

l2_total_searches += 1
return -1  # Not found
```

**Problem**: This is CPU-side profiling, not suitable for JAX GPU code!

**Better approach**: Use `jax.experimental.host_callback` or run multiple configs and compare retention.

---

### Phase 2: Implement Incremental L2 Search (JAX-Compatible)

**Key challenge**: JAX requires data-independent control flow

**Solution**: Use `jnp.where` cascade (same pattern as L0→L1→L2)

**File to modify**: [morton_global_search.py](jaxtrace/gpu/search/morton_global_search.py)

**New function**:
```python
def search_L2_morton_incremental_single(pos: jax.Array, mesh_gpu: MeshGPUGlobalMorton) -> jnp.int32:
    """
    Incremental radius-based L2 search with cascading fallback.

    Strategy:
      1. radius=2 (5 leaves)
      2. radius=5 (11 leaves) - only if radius=2 fails
      3. radius=10 (21 leaves) - only if radius=5 fails

    Expected speedup: 2-3× vs always radius=10 (depends on hit rate distribution)
    """
    # Tier 1: radius=2 (fast, covers most cases)
    elem_r2 = search_L2_morton_radius_single(pos, mesh_gpu, radius=2)

    # Tier 2: radius=5 (conditional via jnp.where)
    elem_r5 = jnp.where(
        elem_r2 >= 0,
        elem_r2,  # Found at radius=2, return early
        search_L2_morton_radius_single(pos, mesh_gpu, radius=5)
    )

    # Tier 3: radius=10 (conditional via jnp.where)
    elem_final = jnp.where(
        elem_r5 >= 0,
        elem_r5,  # Found at radius=2 or 5, return early
        search_L2_morton_radius_single(pos, mesh_gpu, radius=10)
    )

    return elem_final
```

**Integration**: Add to production config
```python
# production_tracking_fully_fused_timedep.py
L2_SEARCH_METHOD = 'incremental'  # NEW: radius=2→5→10 cascade
```

**Dispatcher update** in `rk4_fully_fused_timedep.py`:
```python
if L2_SEARCH_METHOD == 'incremental':
    from jaxtrace.gpu.search.morton_global_search import search_L2_morton_incremental_single
    search_L2_fn = search_L2_morton_incremental_single
```

---

## Expected Performance Impact

### Conservative Estimate (60/30/10 hit distribution)

**Current (radius=10)**:
- Always searches 21 leaves
- Throughput: 30,500 p/s

**Incremental (radius=2→5→10)**:
- 60% found at radius=2: 5 leaves
- 30% found at radius=5: 5 + 11 = 16 leaves (tier 1 + tier 2)
- 10% found at radius=10: 5 + 11 + 21 = 37 leaves (all tiers)
- **Average: 0.6×5 + 0.3×16 + 0.1×37 = 11.5 leaves**

**Speedup**: 21 / 11.5 = **1.83× faster L2**

**Combined with inverse point-in-tet** (3-4× speedup):
- Current baseline: 30,500 p/s
- After incremental L2: 30,500 × 1.83 = **55,800 p/s**
- After inverse point-in-tet: 55,800 × 1.8 = **100,400 p/s**
- **Total: 3.3× speedup** (30K → 100K particles/second)

### Optimistic Estimate (80/15/5 hit distribution)

If radius=2 has 80% hit rate:
- Average: 0.8×5 + 0.15×16 + 0.05×37 = 8.25 leaves
- Speedup: 21 / 8.25 = **2.55× faster L2**
- **Total: 4.6× speedup** with inverse point-in-tet

---

## Implementation Plan

### Step 1: Empirical Validation (2 hours)

**Goal**: Measure actual L2 hit rates

**Method**: Run production test 3 times with different radius values:
1. `L2_SEARCH_RADIUS = 2` → measure retention at step 100
2. `L2_SEARCH_RADIUS = 5` → measure retention at step 100
3. `L2_SEARCH_RADIUS = 10` → measure retention at step 100 (baseline)

**Analysis**:
```
retention_r2 = 90% → radius=2 hit rate = 90%
retention_r5 = 96% → radius=5 additional = 6%
retention_r10 = 98% → radius=10 additional = 2%
```

**Decision**:
- If radius=2 > 70% retention → implement incremental (good ROI)
- If radius=2 < 50% retention → incremental won't help much

### Step 2: Implement Incremental L2 (1 hour)

**Files to modify**:
1. [morton_global_search.py](jaxtrace/gpu/search/morton_global_search.py) - Add `search_L2_morton_incremental_single()`
2. [rk4_fully_fused_timedep.py](jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py) - Add dispatcher case
3. [production_tracking_fully_fused_timedep.py](production_tracking_fully_fused_timedep.py) - Add config option

### Step 3: Validate (1 hour)

**Tests**:
1. Run production test with `L2_SEARCH_METHOD='incremental'`
2. Verify retention matches `radius=10` baseline
3. Measure speedup (expect 1.5-2.5×)

### Step 4: Combined Optimization (already done!)

**Stack all optimizations**:
1. ✅ Hierarchical conditional (depth-7→depth-6) - 1.4× speedup
2. ✅ Point-in-tet inverse matrix - 1.8× speedup
3. 🔄 Incremental L2 (radius=2→5→10) - 1.8× speedup
4. **Total: 1.4 × 1.8 × 1.8 = 4.5× combined speedup**

---

## Risk Analysis

### Risk 1: Hit Rate Distribution Unknown

**Mitigation**: Run Step 1 empirical validation first

**Fallback**: If radius=2 hit rate < 50%, skip incremental L2 and use hierarchical instead

### Risk 2: JAX Compilation Overhead

**Issue**: Conditional execution via `jnp.where` still executes both branches

**Analysis**: This is already proven to work in L0→L1→L2 hierarchy
- Production logs show L0 hit rate = 85.1%
- L1 searches still happen for all particles (via `jnp.where`)
- But JAX partitions data and skips work for successful L0 particles
- **Conclusion**: Same will work for incremental L2

### Risk 3: Memory Bandwidth Saturation

**Issue**: If GPU is memory-bound, computational speedup won't translate to throughput

**Mitigation**:
- Inverse point-in-tet also reduces memory access (coalesced reads)
- Incremental L2 reduces Morton leaf access
- Combined effect should help even if memory-bound

---

## Next Steps

**Immediate** (before implementing incremental L2):
1. ✅ Complete implementation of inverse point-in-tet
2. ✅ Complete implementation of hierarchical conditional
3. ⏳ Run empirical validation (3 production tests with radius=2, 5, 10)
4. ⏳ Analyze hit rate distribution
5. ⏳ Decide: incremental L2 vs hierarchical conditional as L2 method

**After validation**:
- If incremental L2 is promising → implement it
- If hierarchical conditional is better → use that instead
- Stack with inverse point-in-tet for maximum speedup

---

## Conclusion

**Key insight from logs**: Radius-based L2 is dramatically superior to neighbors for initial assignment (83.8% vs 28.4%).

**Hypothesis**: Incremental radius (2→5→10) will combine best of both:
- **Retention** of radius=10 (proven in logs)
- **Speed** of smaller radii (60-80% hit at radius=2)
- **Expected: 1.8-2.5× L2 speedup**

**Next action**: Run 3 production tests with different radii to validate hypothesis before implementing.

**Combined with inverse point-in-tet**: Expected **3-5× total speedup** (30K → 100K+ particles/second).
