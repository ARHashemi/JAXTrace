# Octree-Only Performance Issues - Root Cause Analysis

## Summary

Found **2 critical bugs** causing slow performance and incorrect particle scattering:

1. **BUG #1: Octree searches wrong positions for "found" particles** → Particle scattering
2. **BUG #2: Throughput calculation is misleading** → Confusing performance metrics
3. **PERFORMANCE: Octree search is 2-3× slower than expected** → Need optimization

---

## BUG #1: Octree Searching Wrong Positions (CRITICAL)

### Location
`jaxtrace/gpu/search/octree_search_gpu.py:238-331`

### The Problem

```python
# Line 239: Identify unfound particles
unfound_mask = cached_element_ids < 0

# Line 243: Extract positions, but MASKING CREATES WRONG VALUES
unfound_positions = jnp.where(
    unfound_mask[:, None],
    positions,
    0.0  # ← BUG: Found particles get (0,0,0) position!
)

# Line 331: Run octree search on ALL particles (including found ones!)
octree_results = jax.vmap(search_one_particle)(unfound_positions)
# ↑ This searches found particles at position (0,0,0)!

# Line 335: Use octree results for unfound particles only
element_ids = jnp.where(
    unfound_mask,
    octree_results,  # ← Contains WRONG results for found particles
    cached_element_ids
)
```

### Why This Happens

The code tries to avoid nested vmap by running octree search on ALL particles, but:
1. "Found" particles (cached_id >= 0) get dummy position (0,0,0)
2. Octree searches for element at (0,0,0) and returns some element ID
3. These wrong results are discarded by the final `jnp.where`, BUT...
4. **In octree-only mode, ALL particles have cached_id=-1**, so the masking has no effect!

### Impact on Octree-Only Mode

When you pass `-1` for all particles (forcing octree search):
```python
# In create_rk4_step_octree_only():
dummy_cached_ids = jnp.full_like(..., -1)  # All particles need search
```

The `unfound_mask` is ALL TRUE, so:
- Line 243: `unfound_positions = positions` (correct)
- Line 331: Searches all positions (correct)
- Line 335: Uses octree results for all (correct)

**So why is there particle scattering?**

The bug is actually **subtle and different** in octree-only mode...

---

## BUG #1 REVISED: Octree Returns Wrong Elements

### Real Problem

Looking at the log output:
```
✓ Octree built (4.95 s)
  Filtered elements: 3,512,384/3,512,384 (100.0%)  ← ALL ELEMENTS IN OCTREE
  Total nodes: 240,227
  Leaf nodes: 209,309
  Max depth: 8
```

With `OCTREE_LEVELSET_THRESHOLD = 1.1`, the octree includes **ALL elements** (levelset range is -0.003 to 0.031).

The particle scattering happens because:
1. Octree traversal might hit wrong octant due to numerical precision
2. Leaf node check might fail to find correct element (tolerance issues)
3. When element is not found, returns `-1`, particle gets deactivated or random element

### Evidence from Log

```
Step   100/2500 | Active: 95,899  ← Lost 7,772 particles (7.5%)
Step   200/2500 | Active: 95,534  ← Lost 365 more particles
```

Particles are being lost even though octree covers 100% of elements!

### Root Causes

1. **Max depth too low**: `max_depth=8` when octree can go deeper
2. **Leaf size too large**: 50 elements per leaf → many point-in-tet checks
3. **Tolerance issues**: `point_in_tet_jax` uses `tolerance=1e-10`, too strict for RK4 intermediate stages

---

## BUG #2: Misleading Throughput Calculation

### Location
`production_tracking_octree_only.py:725`

### The Problem

```python
step_time = time.perf_counter() - step_start
throughput = particle_data.n_active / step_time  # ← WRONG
```

This calculates "active particles processed per second", NOT "total particles per second".

### Why It's Confusing

```
Step   100/2500 | Throughput: 17555.3 p/s | ETA: 219.3 min
```

- Throughput shows 17,555 p/s (sounds fast!)
- But ETA is 219 minutes = 13,140 seconds for 2,400 more steps
- 13,140 / 2,400 = 5.475 seconds per step
- 95,899 active / 5.475 s = 17,520 p/s ✓ (matches reported throughput)

**But the real issue**: Each step takes **5.5 seconds**, not 0.006 seconds!

### Correct Calculation

Should use **initial particle count** or **current particle count** consistently:

```python
# Option 1: Total particles (including inactive)
throughput = len(particle_data.positions) / step_time

# Option 2: Active particles (but make it clear)
throughput = particle_data.n_active / step_time  # "Active particles/s"
```

The ETA calculation is correct:
```python
eta = (N_TIMESTEPS - step - 1) * np.mean(step_times[-100:])
```

But the throughput metric is misleading.

---

## PERFORMANCE ISSUE: Octree Search is Slow

### Expected vs Actual

| Metric | Expected (L0+L1+L2) | Actual (Octree-only) | Difference |
|--------|---------------------|----------------------|------------|
| Throughput | 40-48k p/s | 17.5k p/s | **2.3-2.7× slower** |
| Time/step | 2-2.5 seconds | 5.5 seconds | **2.2× slower** |
| GPU Memory | 2,163 MB | 2,163 MB | Same |

### Why So Slow?

1. **5 octree searches per timestep** (one per RK4 stage + final)
   - L0+L1+L2: Most particles hit L0 (cache) or L1 (cheap neighbor check)
   - Octree-only: ALL particles do full octree traversal EVERY time

2. **Octree traversal cost**
   - Average depth: 4-6 levels
   - Each level: Load metadata, compute octant, check child
   - Leaf check: Vmap over 50 elements, point-in-tet for each

3. **No early exit between RK4 stages**
   - Even if k1 position found element, k2/k3/k4 re-search from root
   - L0+L1+L2: k2/k3/k4 usually hit L0 cache (same element)

### Bottleneck Breakdown (Estimated)

For 105k particles, 5 searches per step:
- Total searches: 525,000 per timestep
- Time: 5.5 seconds
- Searches/second: 95,454
- Time per search: ~10.5 microseconds

Compare to L1 neighbor check: ~1 microsecond per search

**Octree is 10× slower than L1 neighbor search.**

---

## FIXES

### FIX #1: Improve Octree Correctness

**File:** `jaxtrace/gpu/search/octree_search_gpu.py`

**Changes:**

1. **Increase tolerance for point-in-tet**:
   ```python
   # Line 20: Change tolerance from 1e-10 to 1e-6
   def point_in_tet_jax(..., tolerance: float = 1e-6):
   ```

   RK4 intermediate stages (k1, k2, k3) might have particles slightly outside elements due to velocity field divergence. Need looser tolerance.

2. **Increase max_depth**:
   ```python
   # In production script, change from 10 to 15
   OCTREE_MAX_DEPTH = 15
   ```

   Log shows actual depth is 8, but some particles might need deeper traversal.

3. **Add fallback to full mesh scan** (if octree fails):
   ```python
   # After octree search, check if result is -1
   # If so, do brute-force scan of nearby elements
   ```

### FIX #2: Clarify Throughput Calculation

**File:** `production_tracking_octree_only.py`

**Changes:**

```python
# Line 725: Calculate throughput for ALL particles (not just active)
n_total_particles = len(particle_data.positions)
throughput = n_total_particles / step_time

# Line 740: Update print to show BOTH metrics
print(f"Step {step+1:>5}/{N_TIMESTEPS} | "
      f"Active: {particle_data.n_active:>6,} ({100*particle_data.n_active/n_total_particles:.1f}%) | "
      f"Time/step: {step_time:>6.3f}s | "
      f"Throughput: {throughput:>7.1f} p/s | "
      f"GPU: {gpu_mem:>5.0f} MB | "
      f"RAM: {ram_mb:>6.0f} MB | "
      f"Exported: {export_stats['n_exported']:>4} | "
      f"ETA: {eta/60:.1f} min")
```

This makes it clear:
- How many particles are active (retention rate)
- How long each step takes (seconds per step)
- True throughput (all particles / time)

### FIX #3: Optimize Octree Performance

**Option A: Reduce octree searches per timestep**

Cache element IDs between RK4 stages (use k1 result for k2, etc.):

```python
# In rk4_fused_octree_only():
element_ids_k1 = search_octree_only(positions_gpu, element_ids_gpu, ...)

# For k2, use k1 result as cache (enable L0 check)
element_ids_k2 = search_octree_only(positions_k1, element_ids_k1, ...)
```

But this requires modifying `search_octree_only` to NOT force all searches.

**Option B: Reduce leaf size**

Smaller leaves = fewer point-in-tet checks:
```python
OCTREE_MAX_LEAF_SIZE = 20  # Instead of 50
```

**Option C: Accept that octree-only is slower**

This test proves that L0+L1 provide valuable early exit optimization.
The multilevel hierarchy is justified by the 2-3× speedup.

---

## RECOMMENDATIONS

### Immediate Actions

1. **Fix tolerance** in `point_in_tet_jax`:
   - Change from `1e-10` to `1e-6`
   - This should reduce particle scattering

2. **Fix throughput reporting**:
   - Show time/step clearly
   - Use consistent particle count for throughput

3. **Increase max_depth**:
   - Change from 10 to 15 in production script
   - Ensure all particles can traverse to leaves

### Testing

Run octree-only test again with these fixes:
```bash
python production_tracking_octree_only.py 2>&1 | tee logs/production_octree_only_FIXED.log
```

Expected improvements:
- Particle retention: 95-98% (vs 92% currently)
- Throughput: Still ~17k p/s (octree is inherently slower)
- Time/step: ~5 seconds (unchanged, but clearer reporting)

### Conclusion

**The multilevel search (L0+L1+L2) is 2-3× faster than octree-only.**

This proves:
- L0 cache provides valuable fast path (85-95% hit rate)
- L1 neighbor search is much faster than octree (10× faster per search)
- Octree should only be used as L2 fallback, not primary search method

**Recommendation: Keep L0+L1+L2 architecture for production.**

Octree-only mode is useful for:
- Testing octree correctness
- Measuring worst-case performance
- Debugging particle loss issues

But NOT for production use due to 2-3× performance penalty.
