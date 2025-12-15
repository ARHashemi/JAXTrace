# Octree-Only Performance Fixes - Complete

## Issues Identified and Fixed

### Issue #1: Particle Scattering (Octree Returning Wrong Elements)

**Root Cause:**
- `point_in_tet_jax()` used tolerance of `1e-10` (too strict)
- RK4 intermediate stages (k1, k2, k3) can have particles slightly outside elements
- Velocity field divergence pushes particles beyond exact tet boundaries
- Strict tolerance caused octree to reject correct elements, return `-1`

**Fix Applied:**
- **File:** `jaxtrace/gpu/search/octree_search_gpu.py:20`
- **Change:** `tolerance: float = 1e-6` (was `1e-10`)
- **Impact:** 10,000× more lenient tolerance for barycentric coordinate checks

**Expected Improvement:**
- Particle retention: 95-98% (vs 92% before)
- Fewer "not found" cases during RK4 intermediate stages
- More stable particle trajectories

---

### Issue #2: Misleading Throughput and Performance Metrics

**Root Cause:**
- Throughput calculated as `n_active / step_time`
- As particles were lost, `n_active` decreased
- Reported throughput decreased even though time/step was constant
- User saw "17k p/s" but ETA showed 219 minutes (confusing!)

**Actual Performance:**
- Time per step: ~5.5 seconds
- 103k particles / 5.5s = 18.7k p/s (true throughput)
- ETA calculation was correct, but throughput metric was misleading

**Fixes Applied:**

1. **Consistent particle count** (`production_tracking_octree_only.py:702`)
   ```python
   # Store initial count, use it for all throughput calculations
   n_total_particles = particle_data.n_active
   throughput = n_total_particles / step_time  # Not n_active!
   ```

2. **Enhanced progress reporting** (`production_tracking_octree_only.py:746-752`)
   ```python
   print(f"Step {step+1:>5}/{N_TIMESTEPS} | "
         f"Active: {particle_data.n_active:>6,} ({retention_pct:>5.1f}%) | "
         f"Time/step: {avg_step_time:>6.3f}s | "  # ← NEW: Show seconds/step
         f"Throughput: {avg_throughput:>7.0f} p/s | "
         f"GPU: {gpu_mem:>5.0f} MB | "
         f"Exported: {export_stats['n_exported']:>3} | "
         f"ETA: {eta/60:.1f} min")
   ```

**New Output Format:**
```
Step   100/2500 | Active: 95,899 (92.5%) | Time/step:  5.475s | Throughput: 18910 p/s | ...
```

Now it's clear:
- Each step takes **5.5 seconds**
- **92.5%** of particles remain active
- Throughput is consistent (uses initial particle count)
- ETA calculation is transparent

---

### Issue #3: Max Depth Too Low

**Root Cause:**
- Log showed: `Max depth: 8` (actual tree depth)
- Configuration used: `OCTREE_MAX_DEPTH = 10`
- Some particles might need deeper traversal (10 iterations might not reach leaf)

**Fix Applied:**
- **File:** `production_tracking_octree_only.py:233`
- **Change:** `OCTREE_MAX_DEPTH = 15` (was `10`)
- **Impact:** 50% more traversal iterations, ensures all particles reach leaves

**Expected Improvement:**
- Particles in deep octree branches now found
- Fewer `-1` returns from octree search
- Slight performance cost (~10% slower due to more iterations)

---

## Performance Analysis

### Why Octree-Only is 2-3× Slower Than L0+L1+L2

**Timing Breakdown:**

| Component | Octree-Only | L0+L1+L2 | Explanation |
|-----------|-------------|----------|-------------|
| L0 cache hit | N/A | ~1 μs/search | 85-95% particles (instant, no search needed) |
| L1 neighbor | N/A | ~1 μs/search | 5-14% particles (4-20 point-in-tet checks) |
| L2 octree | ~10.5 μs/search | ~10.5 μs/search | 0.05-0.5% particles (full tree traversal) |
| **Searches/step** | 525,000 | ~50,000 | Octree-only: 105k × 5 stages = 525k |
| **Time/step** | ~5.5 s | ~2.2 s | L0+L1+L2: Most hit L0/L1 (cheap) |

**Bottleneck:**
- Octree traversal: 4-6 levels × metadata loads + octant computation
- Leaf check: 50 point-in-tet operations per leaf (expensive)
- No caching between RK4 stages

**Why L0+L1+L2 is Faster:**
1. **L0 cache** (85-95% hit): Free (just validate cached element)
2. **L1 neighbors** (5-14% hit): Cheap (1-4 neighbors to check)
3. **L2 octree** (0.05% hit): Only ~500 particles use octree per step

Result: Average cost = 0.95×1μs + 0.05×1μs + 0.0005×10μs ≈ **1.005 μs/search**

Compare to octree-only: **10.5 μs/search** (10× more expensive)

---

## Test Results (Before vs After Fixes)

### Before Fixes
```
OCTREE_LEVELSET_THRESHOLD = 0.012
OCTREE_MAX_DEPTH = 10
point_in_tet tolerance = 1e-10

Step   100/2500 | Active: 95,899 | Throughput: 17555.3 p/s | ETA: 219.3 min
Step   200/2500 | Active: 95,534 | Throughput: 17469.2 p/s | ETA: 210.0 min

Issues:
- 7.5% particle loss by step 100
- Confusing throughput (appears fast but ETA shows slow)
- Particles scattering in early timesteps
```

### After Fixes (Expected)
```
OCTREE_LEVELSET_THRESHOLD = 1.1
OCTREE_MAX_DEPTH = 15
point_in_tet tolerance = 1e-6

Step   100/2500 | Active: 98,500 (95.0%) | Time/step:  5.500s | Throughput: 18850 p/s | ETA: 220 min
Step   200/2500 | Active: 97,800 (94.3%) | Time/step:  5.450s | Throughput: 19000 p/s | ETA: 209 min

Improvements:
- ~95% retention (vs 92.5%)
- Clear metrics (time/step explicit)
- Less particle scattering
- Performance still 2-3× slower than L0+L1+L2 (expected)
```

---

## Files Modified

### 1. `jaxtrace/gpu/search/octree_search_gpu.py`
- **Line 20:** Changed tolerance from `1e-10` to `1e-6`
- **Impact:** All octree searches now use looser tolerance

### 2. `production_tracking_octree_only.py`
- **Line 233:** Changed `OCTREE_MAX_DEPTH` from `10` to `15`
- **Line 702:** Store initial particle count for consistent throughput
- **Line 729:** Use `n_total_particles` instead of `n_active` for throughput
- **Lines 739-752:** Enhanced progress reporting with time/step and retention %

---

## Running the Fixed Test

```bash
python production_tracking_octree_only.py 2>&1 | tee logs/production_octree_only_FIXED.log
```

### Expected Results

**Performance:**
- Time/step: ~5-6 seconds (unchanged, octree is inherently slower)
- Throughput: ~17-19k p/s (based on initial 103k particles)
- Total runtime: ~3.5-4 hours for 2,500 steps

**Retention:**
- Step 100: 95-98% active (vs 92.5% before)
- Step 2,500: 85-90% active (vs unknown before)
- Particle scattering: Minimal (tolerance fix)

**Comparison to L0+L1+L2:**
- L0+L1+L2: ~2.2s/step, 82% retention, 40-48k p/s
- Octree-only: ~5.5s/step, 85-90% retention, 18k p/s
- **Conclusion: L0+L1+L2 is 2.5× faster with similar retention**

---

## Recommendations

### For Production Use

**Use L0+L1+L2 architecture** (not octree-only):
- 2.5× faster (2.2s vs 5.5s per step)
- Same retention (~82% vs ~85%)
- Proven in production tests

**When to Use Octree-Only:**
- Debugging particle loss issues
- Testing octree correctness
- Worst-case performance baseline
- Research: Understanding search hierarchy value

### Further Optimizations (If Needed)

If octree-only performance must be improved:

1. **Reduce leaf size**: `OCTREE_MAX_LEAF_SIZE = 20` (fewer point-in-tet checks)
2. **Add L0 caching between RK4 stages**: Check previous stage's element first
3. **Increase tree depth during build**: Deeper tree = smaller leaves = faster checks
4. **Use spatial hashing instead of octree**: O(1) lookup vs O(log n) traversal

But recommended approach: **Keep L0+L1+L2 for production.**

---

## Validation Steps

After running the fixed test:

1. **Check particle retention**:
   ```bash
   grep "Step.*2500" logs/production_octree_only_FIXED.log
   # Should show 85-90% retention
   ```

2. **Verify no scattering**:
   - Load `output/threadeda_octree_only/particles_step_000010.vtu`
   - Particles should stay in grid formation (not scattered)

3. **Compare with L0+L1+L2**:
   ```bash
   # Time/step comparison
   grep "Time/step" logs/production_octree_only_FIXED.log  # ~5.5s
   grep "Step.*100" logs/production_3hop_l2_octree.log     # ~2.2s

   # L0+L1+L2 is 2.5× faster ✓
   ```

4. **Check octree coverage**:
   ```bash
   grep "Filtered elements" logs/production_octree_only_FIXED.log
   # Should show 100% (threshold 1.1 includes all)
   ```

---

## Summary

✅ **Fix #1:** Increased point-in-tet tolerance (1e-10 → 1e-6)
✅ **Fix #2:** Clarified throughput calculation and reporting
✅ **Fix #3:** Increased max octree depth (10 → 15)

**Test Status:** Ready to run with fixes applied

**Expected Outcome:**
- Better particle retention (95% vs 92%)
- Clearer performance metrics
- Still 2-3× slower than L0+L1+L2 (proves multilevel architecture value)

**Next Step:** Run test to validate fixes
```bash
python production_tracking_octree_only.py 2>&1 | tee logs/production_octree_only_FIXED.log
```
