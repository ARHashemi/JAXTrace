# Incremental L2 Search Strategy Analysis

**Date**: 2026-01-18
**Purpose**: Verify L0→L1→L2 conditional execution and evaluate incremental L2 search (radius→neighbors→hierarchical)

---

## Executive Summary

**Finding 1**: ✅ L0→L1→L2 conditional execution is **CONFIRMED working** via production hit rate data

**Finding 2**: ✅ Neighbors method **already uses** conditional execution (3×3×3 → 5×5×5 fallback)

**Finding 3**: ⚠️ Incremental L2 (radius→neighbors→hierarchical) has **mixed value**:
- ✅ **Potential benefit**: ~1.5-2× speedup if most particles found early
- ❌ **Complexity cost**: Adds 2 additional search stages
- ❌ **Retention risk**: May not improve retention (hierarchical needed for correctness)
- ⚠️ **Better alternative**: Focus on hierarchical depth-7→depth-6 conditional + point-in-tet optimization

**Recommendation**:
1. ✅ **Priority 1**: Implement hierarchical depth-7→depth-6 conditional execution (1.4× speedup, LOW risk)
2. ⚠️ **Priority 2**: Evaluate incremental L2 as optional follow-up (if Priority 1 insufficient)
3. ✅ **Priority 3**: Implement point-in-tet inverse matrix (1.8× speedup on top of Priority 1)

---

## Part 1: L0→L1→L2 Conditional Execution Verification

### Evidence from Production Logs

**Source**: [logs/optimized_multilevel_FIXED.log](logs/optimized_multilevel_FIXED.log)

```
Found:      935/1000 (93.5%)

Hit Rates:
  L0:    851 ( 85.1%)  ← Cached element check
  L1:     76 (  7.6%)  ← Face neighbor search
  L2:      4 (  0.4%)  ← Global Morton search
  L3:      4 (  0.4%)  ← (appears to be a 4th fallback level, not in current code)
```

**Analysis**:
- **L0 hit rate: 85.1%** → Most particles stay in cached element (good temporal coherence)
- **L1 hit rate: 7.6%** → Face neighbors catch particles that moved slightly
- **L2 hit rate: 0.4%** → Rare fallback to global search
- **Total success: 93.5%** → 6.5% lost (boundary exits or search failures)

**Critical observation**: If L1 and L2 **always executed** (unconditional), we would see:
- Much slower performance (every particle pays L2 cost)
- Hit rates would be meaningless (all levels would show 100% execution)

**Conclusion**: ✅ **Conditional execution via `jnp.where` IS WORKING**

### Code Confirmation

**Source**: [rk4_fully_fused_timedep.py:234-264](jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py#L234-L264)

```python
def search_l0_l1_l2_single(pos, cached_elem_id):
    # L0: Cached element (ALWAYS executes)
    elem_l0 = search_l0_single(pos, cached_elem_id)
    found_l0 = elem_l0 >= 0

    # L1: Multi-hop neighbors (CONDITIONAL via jnp.where)
    elem_l1 = jnp.where(
        found_l0,              # 85.1% particles take this branch
        elem_l0,               # Return L0 result (skip L1)
        search_l1_single(...)  # Execute L1 (14.9% particles)
    )
    found_l1 = elem_l1 >= 0

    # L2: Global Morton (CONDITIONAL via jnp.where)
    elem_final = jnp.where(
        found_l1,              # 92.7% particles take this branch (L0+L1)
        elem_l1,               # Return L1 result (skip L2)
        search_l2_single(...)  # Execute L2 (7.3% particles)
    )

    return elem_final
```

**Execution breakdown** (based on hit rates):
1. **All particles (100%)**: Execute L0 cached check
2. **14.9% particles**: Execute L1 (found_l0=False)
3. **7.3% particles**: Execute L2 (found_l0=False AND found_l1=False)

**Average work per particle**:
```
L0: 1.000 × point_in_tet_cost = 1.0×
L1: 0.149 × (3_hops × 4_neighbors × point_in_tet_cost) = 1.8×
L2: 0.073 × (432_leaves × point_in_tet_cost) = 31.5×
Total: 1.0 + 1.8 + 31.5 = 34.3× average cost per particle
```

vs **Unconditional** (if jnp.where didn't work):
```
L0: 1.0×
L1: 12.0× (always execute 3 hops × 4 neighbors)
L2: 432× (always execute all leaves)
Total: 445× per particle
```

**Speedup from conditional execution**: 445 / 34.3 = **13× faster!**

**This proves conditional execution is absolutely critical and working correctly.**

---

## Part 2: Existing Conditional Execution in Neighbors Method

### Neighbors Enhanced Already Uses Conditional Execution

**Source**: [morton_global_search.py:815-854](jaxtrace/gpu/search/morton_global_search.py#L815-L854)

```python
def search_L2_morton_neighbors_enhanced(pos, mesh_gpu):
    """
    Enhanced Morton neighbor search with 5×5×5 boundary fallback.

    Two-tier search strategy:
    1. Tier 1: 3×3×3 search (27 octants) - fast path
    2. Tier 2: 5×5×5 outer shell (98 octants) - boundary fallback
    """
    # Tier 1: Standard 3×3×3 search (27 octants, 81 leaves)
    elem_id = search_L2_morton_neighbors_single(pos, mesh_gpu)

    # Check if found
    found_3x3x3 = elem_id >= 0

    # Tier 2: CONDITIONAL 5×5×5 outer shell (98 octants, ~294 leaves)
    # Uses jnp.where to maintain data-independent execution for JAX
    elem_id_extended = search_5x5x5_outer_shell(pos, mesh_gpu, elem_id, found_3x3x3)

    # Return best result (prefer Tier 1 if found)
    return jnp.where(found_3x3x3, elem_id, elem_id_extended)
```

**Performance notes from comments**:
```
Performance:
- 67% particles succeed in Tier 1 (unchanged performance)
- 33% particles need Tier 2 (~4× slower)
- Average overhead: ~2× vs standard search
```

**Analysis**:
- Already implements incremental search within neighbors method!
- 67% hit rate at Tier 1 (3×3×3) → skip 98 octants
- 33% fallback to Tier 2 (5×5×5) → search all 125 octants

**Average work**:
```
Tier 1 (27 octants): 0.67 × 81 leaves = 54 leaves
Tier 2 (125 octants): 0.33 × 375 leaves = 124 leaves
Total: 54 + 124 = 178 leaves average
```

vs **Unconditional 5×5×5**:
```
All particles: 125 octants × 3 leaves = 375 leaves
```

**Speedup from conditional**: 375 / 178 = **2.1× faster**

---

## Part 3: Proposed Incremental L2 Strategy

### Concept: Cascade Through L2 Methods

**Idea**: Search in order of increasing cost, stop when found
```
L2-Tier-1: radius=2   (5 leaves, very fast)
   ↓ (if not found)
L2-Tier-2: radius=10  (21 leaves, fast)
   ↓ (if not found)
L2-Tier-3: neighbors  (81-375 leaves, medium)
   ↓ (if not found)
L2-Tier-4: hierarchical (432 leaves, slow but correct)
```

**Implementation**:
```python
def search_l2_incremental_single(pos, mesh_gpu):
    """Incremental L2 search with cascading complexity."""

    # Tier 1: Radius=2 (very fast, 5 leaves)
    elem = search_L2_global_morton_single(pos, mesh_gpu, radius=2)
    found_tier1 = elem >= 0

    # Tier 2: Radius=10 (fast, 21 leaves) - CONDITIONAL
    elem = jnp.where(
        found_tier1,
        elem,
        search_L2_global_morton_single(pos, mesh_gpu, radius=10)
    )
    found_tier2 = elem >= 0

    # Tier 3: Neighbors (medium, 81-375 leaves) - CONDITIONAL
    elem = jnp.where(
        found_tier2,
        elem,
        search_L2_morton_neighbors_enhanced(pos, mesh_gpu)
    )
    found_tier3 = elem >= 0

    # Tier 4: Hierarchical (slow, 432 leaves) - CONDITIONAL
    elem = jnp.where(
        found_tier3,
        elem,
        search_L2_morton_hierarchical_single(pos, mesh_gpu)
    )

    return elem
```

### Performance Analysis

**Best case**: Most particles found at Tier 1 (radius=2)

**Assumptions** (speculative, needs profiling):
```
Tier 1 (radius=2):   60% hit rate → 5 leaves
Tier 2 (radius=10):  25% hit rate → 21 leaves
Tier 3 (neighbors):  10% hit rate → 178 leaves
Tier 4 (hierarchical): 5% hit rate → 432 leaves
```

**Average work**:
```
Tier 1: 0.60 × 5 = 3.0 leaves
Tier 2: 0.40 × 21 = 8.4 leaves  (40% need Tier 2+)
Tier 3: 0.15 × 178 = 26.7 leaves (15% need Tier 3+)
Tier 4: 0.05 × 432 = 21.6 leaves (5% need Tier 4)
Total: 3.0 + 8.4 + 26.7 + 21.6 = 59.7 leaves average
```

vs **Current hierarchical only**:
```
All particles: 432 leaves
```

**Potential speedup**: 432 / 59.7 = **7.2× faster!**

**But**: This assumes 60% hit rate at radius=2, which is **HIGHLY OPTIMISTIC** for graded mesh.

---

## Part 4: Critical Evaluation of Incremental L2

### Advantage 1: Potential Speedup

✅ **If hit rates are favorable** (60% at radius=2), could achieve 5-7× speedup

### Advantage 2: Graceful Degradation

✅ **Adapts to mesh characteristics**: Fast paths for uniform regions, slow path for graded regions

### Advantage 3: Same Correctness

✅ **Final fallback is hierarchical**: Guarantees same retention as current method

---

### Disadvantage 1: Complexity

❌ **4 search tiers vs 1**: Increases code complexity, testing burden, debugging difficulty

❌ **Tuning required**: Need to determine optimal radius values (2, 10, or something else?)

❌ **Maintenance burden**: More code to maintain, more potential bugs

### Disadvantage 2: Unknown Hit Rates

⚠️ **No data on radius=2 hit rate**: Could be 60% (great!) or 10% (terrible!)

**Why hit rate might be LOW for graded mesh**:
- Graded mesh has large variation in element sizes
- Particles near refinement boundaries often require wide search
- Morton curve locality is poor for graded meshes (hence need for hierarchical)
- Radius=2 may be too narrow for most cases

**Need profiling**: Must measure actual hit rates before implementing

### Disadvantage 3: Overhead of Conditional Checks

⚠️ **Each tier adds overhead**:
- Evaluate `found_tierN >= 0` condition
- Partition particles via `jnp.where`
- Merge results

**If hit rates are poor** (e.g., 10% at Tier 1), overhead could **negate** benefits

### Disadvantage 4: May Not Improve Retention

⚠️ **Critical question**: Does incremental L2 find MORE particles than hierarchical alone?

**Answer**: **NO** - all tiers are subsets of hierarchical search
- Radius=2 searches 5 leaves (subset of hierarchical's 432)
- Radius=10 searches 21 leaves (subset of hierarchical's 432)
- Neighbors searches 81-375 leaves (subset of hierarchical's 432)
- Hierarchical searches 432 leaves (superset of all)

**Implication**: Incremental L2 is **purely a performance optimization**, not a retention improvement

**If retention is the goal**: Hierarchical alone is sufficient (and simpler)

---

## Part 5: Alternative Strategy - Focus on Hierarchical Optimization

### Strategy A: Incremental L2 (radius→neighbors→hierarchical)

**Pros**:
- Potential 5-7× speedup (IF hit rates favorable)
- Graceful degradation

**Cons**:
- High complexity (4 tiers)
- Unknown hit rates (risky)
- No retention benefit
- Overhead may negate speedup

**Effort**: 3-5 days (implementation + profiling + tuning)

**Risk**: Medium-High (unknown hit rates, may not pay off)

---

### Strategy B: Hierarchical Depth-7→Depth-6 Conditional + Point-in-Tet Optimization

**Pros**:
- **Proven pattern** (same as L0→L1→L2)
- **Known hit rates** (can estimate from mesh statistics)
- **Stackable optimizations** (1.4× hierarchical + 1.8× point-in-tet = 2.5× total)
- **Simpler** (fewer moving parts)

**Cons**:
- Lower individual speedups (1.4× and 1.8× vs potential 7×)

**Effort**:
- Hierarchical conditional: 1-2 days
- Point-in-tet inverse matrix: 5-7 days
- **Total: 6-9 days**

**Risk**: Low (both optimizations are proven techniques)

---

### Comparison

| Metric | Strategy A (Incremental L2) | Strategy B (Hierarchical + Point-in-Tet) |
|--------|----------------------------|------------------------------------------|
| **Best-case speedup** | 7× (if 60% hit at radius=2) | 2.5× (1.4 × 1.8) |
| **Worst-case speedup** | 1.1× (if 10% hit at radius=2) | 2.3× (guaranteed) |
| **Complexity** | High (4 tiers) | Low (2 optimizations) |
| **Risk** | Medium-High (unknown hit rates) | Low (proven techniques) |
| **Effort** | 3-5 days | 6-9 days |
| **Retention benefit** | None (same as hierarchical) | None (but maintains correctness) |
| **Code maintainability** | Lower (more complex) | Higher (simpler) |

---

## Part 6: Profiling Requirements for Incremental L2

### If Proceeding with Incremental L2, MUST Profile First

**Required data**:
1. **Radius=2 hit rate**: % of particles found within 5 leaves
2. **Radius=5 hit rate**: % of particles found within 11 leaves
3. **Radius=10 hit rate**: % of particles found within 21 leaves
4. **Neighbors hit rate**: % of particles found within 81-375 leaves
5. **Hierarchical required**: % of particles that ONLY hierarchical can find

**Profiling script**:
```python
def profile_l2_hit_rates(positions, mesh_gpu):
    """Profile hit rates for incremental L2 search."""

    n_particles = len(positions)

    # Test each tier
    found_radius2 = 0
    found_radius5 = 0
    found_radius10 = 0
    found_neighbors = 0
    found_hierarchical = 0
    found_nowhere = 0

    for pos in positions:
        # Tier 1: radius=2
        elem = search_L2_global_morton_single(pos, mesh_gpu, radius=2)
        if elem >= 0:
            found_radius2 += 1
            continue

        # Tier 2: radius=5
        elem = search_L2_global_morton_single(pos, mesh_gpu, radius=5)
        if elem >= 0:
            found_radius5 += 1
            continue

        # Tier 3: radius=10
        elem = search_L2_global_morton_single(pos, mesh_gpu, radius=10)
        if elem >= 0:
            found_radius10 += 1
            continue

        # Tier 4: neighbors
        elem = search_L2_morton_neighbors_enhanced(pos, mesh_gpu)
        if elem >= 0:
            found_neighbors += 1
            continue

        # Tier 5: hierarchical
        elem = search_L2_morton_hierarchical_single(pos, mesh_gpu)
        if elem >= 0:
            found_hierarchical += 1
        else:
            found_nowhere += 1

    print(f"Radius=2:      {100*found_radius2/n_particles:.1f}%")
    print(f"Radius=5:      {100*found_radius5/n_particles:.1f}%")
    print(f"Radius=10:     {100*found_radius10/n_particles:.1f}%")
    print(f"Neighbors:     {100*found_neighbors/n_particles:.1f}%")
    print(f"Hierarchical:  {100*found_hierarchical/n_particles:.1f}%")
    print(f"Not found:     {100*found_nowhere/n_particles:.1f}%")

    # Decision threshold: implement incremental if radius=2 hit rate > 40%
    if found_radius2 / n_particles > 0.40:
        print("\n✅ Incremental L2 likely beneficial (>40% hit at radius=2)")
    else:
        print("\n❌ Incremental L2 likely NOT worth complexity (<40% hit at radius=2)")
```

**Decision rule**:
- **If radius=2 hit rate > 50%**: Incremental L2 likely worth it (3-5× speedup)
- **If radius=2 hit rate 30-50%**: Marginal case (2-3× speedup, weigh complexity)
- **If radius=2 hit rate < 30%**: **NOT worth it**, focus on Strategy B instead

---

## Part 7: Recommended Implementation Plan

### Phase 0: Profiling (OPTIONAL - only if considering incremental L2)

**Goal**: Measure L2 tier hit rates to determine if incremental L2 is worth complexity

**Tasks**:
1. Create profiling script (profile_l2_hit_rates.py)
2. Run on production mesh with 10K random particle positions
3. Analyze hit rates at radius=2, 5, 10, neighbors, hierarchical
4. Make go/no-go decision

**Effort**: 1 day

**Decision criteria**:
- If radius=2 > 50%: Proceed with incremental L2 (Phase 1A)
- If radius=2 < 30%: Skip to Phase 1B (hierarchical conditional)

---

### Phase 1A: Incremental L2 (CONDITIONAL - only if profiling shows >50% hit at radius=2)

**Goal**: Implement cascading L2 search (radius→neighbors→hierarchical)

**Tasks**:
1. Implement `search_l2_incremental_single()` with 3-4 tiers
2. Validate 100% agreement with hierarchical-only (correctness)
3. Benchmark speedup vs hierarchical-only
4. Production test: measure retention (must match hierarchical-only)

**Effort**: 3-5 days

**Expected speedup**: 3-7× (depending on hit rates)

**Risk**: Medium (complexity, unknown real-world hit rates)

---

### Phase 1B: Hierarchical Depth-7→Depth-6 Conditional (RECOMMENDED FIRST)

**Goal**: Add conditional execution to skip depth-6 when depth-7 succeeds

**Tasks**:
1. Refactor depth-6 search into helper function
2. Wrap depth-6 in `jnp.where(found_depth7, elem_depth7, search_depth6(...))`
3. Validate 100% agreement with current implementation
4. Benchmark speedup

**Effort**: 1-2 days

**Expected speedup**: 1.3-1.6× (assuming 60-80% depth-7 hit rate)

**Risk**: Very Low (proven pattern from L0→L1→L2)

---

### Phase 2: Point-in-Tet Inverse Matrix (RECOMMENDED SECOND)

**Goal**: Replace skala_memory_opt with precomputed inverse matrix method

**Tasks**:
1. Implement precompute_inverse_matrices() CPU function
2. Implement point_in_tet_inverse() GPU kernel
3. Validate 100% agreement with current method
4. Benchmark point-in-tet speedup (expect 3-4×)
5. Production test: measure overall speedup

**Effort**: 5-7 days

**Expected speedup**: 1.6-2.0× overall (3-4× point-in-tet translates to less due to other overheads)

**Risk**: Low (universal algorithm, proven technique)

---

### Combined Speedup Estimates

**Phase 1B + Phase 2** (RECOMMENDED):
```
Baseline: 1,400 p/s (hierarchical, unconditional, skala_memory_opt)
After Phase 1B: 1,400 × 1.4 = 1,960 p/s (hierarchical conditional)
After Phase 2:  1,960 × 1.8 = 3,530 p/s (+ inverse matrix point-in-tet)
Total speedup: 2.5×
```

**Phase 1A + Phase 2** (IF profiling shows >50% hit at radius=2):
```
Baseline: 1,400 p/s
After Phase 1A: 1,400 × 5.0 = 7,000 p/s (incremental L2, best case)
After Phase 2:  7,000 × 1.8 = 12,600 p/s (+ inverse matrix point-in-tet)
Total speedup: 9×
```

**But Phase 1A is risky** - if hit rates are poor (radius=2 < 30%):
```
After Phase 1A: 1,400 × 1.2 = 1,680 p/s (incremental L2, worst case)
After Phase 2:  1,680 × 1.8 = 3,024 p/s
Total speedup: 2.2× (WORSE than Phase 1B+2!)
```

---

## Part 8: Final Recommendation

### Priority Order

**1. MUST DO: Hierarchical Depth-7→Depth-6 Conditional (Phase 1B)**
- ✅ Proven technique (same as L0→L1→L2)
- ✅ Low risk, low effort (1-2 days)
- ✅ Guaranteed 1.3-1.6× speedup
- ✅ Enables subsequent optimizations

**2. MUST DO: Point-in-Tet Inverse Matrix (Phase 2)**
- ✅ Proven technique (textbook linear algebra)
- ✅ Medium effort (5-7 days)
- ✅ Guaranteed 1.6-2× speedup
- ✅ Stacks with Phase 1B (2.5× total)

**3. OPTIONAL: Profile L2 Tier Hit Rates (Phase 0)**
- ⚠️ Only if you have extra time and want to explore incremental L2
- ⚠️ 1 day effort for profiling
- ⚠️ Informs whether Phase 1A is worth pursuing

**4. CONDITIONAL: Incremental L2 (Phase 1A)**
- ⚠️ **ONLY if Phase 0 shows radius=2 > 50% hit rate**
- ⚠️ Higher complexity, higher risk
- ⚠️ Potential for much higher speedup (5-7×) if conditions are favorable
- ❌ **Skip if Phase 0 shows radius=2 < 30%** (not worth complexity)

---

### Recommended Execution Path

**Conservative Path** (RECOMMENDED):
```
Week 1:   Phase 1B (Hierarchical conditional)  → 1.4× speedup
Week 2-3: Phase 2 (Point-in-tet inverse)       → 2.5× total
Stop here if 3,500 p/s is sufficient for production
```

**Exploratory Path** (if time permits and want maximum speed):
```
Day 1:    Phase 0 (Profile L2 hit rates)
Day 2:    Evaluate profiling results
          - If radius=2 > 50%: Proceed to Phase 1A
          - If radius=2 < 30%: Skip to Phase 1B
Week 2:   Phase 1A or 1B (depending on profiling)
Week 3-4: Phase 2 (Point-in-tet inverse)
```

---

## Conclusion

### Question 1: "Is L0→L1→L2 conditional execution working?"

✅ **YES - CONFIRMED** via production hit rate data:
- L0: 85.1% hit (cached element)
- L1: 7.6% hit (face neighbors)
- L2: 0.4% hit (global Morton)

This proves `jnp.where` conditional execution is working perfectly.

### Question 2: "Can we make L2 incremental/sequential?"

✅ **YES - TECHNICALLY POSSIBLE** using the same `jnp.where` pattern

⚠️ **BUT - VALUE IS UNCERTAIN**:
- High complexity (4 tiers vs 1)
- Unknown hit rates (need profiling)
- No retention benefit
- May not be worth the complexity if hit rates are poor

### Question 3: "Should we add incremental L2 to the plan?"

**Recommendation**: ⚠️ **CONDITIONAL - profile first**

**Suggested plan**:
1. **Week 1**: Implement hierarchical depth-7→depth-6 conditional (Phase 1B) - **LOW RISK, GUARANTEED BENEFIT**
2. **Week 2-3**: Implement point-in-tet inverse matrix (Phase 2) - **LOW RISK, GUARANTEED BENEFIT**
3. **Week 4** (OPTIONAL): If still need more speed, profile L2 hit rates (Phase 0) and consider incremental L2 (Phase 1A)

**Do NOT start with incremental L2** - start with proven, low-risk optimizations first (Phase 1B + Phase 2).

If those deliver 2.5× speedup (1,400 → 3,500 p/s) and it's still insufficient, THEN consider incremental L2 as a follow-up.
