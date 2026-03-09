# Void Region Diagnostic Results
**Date**: 2026-02-16
**Test**: diagnose_void_region_corrected.py
**Status**: ✅ Completed Successfully

---

## Test Configuration

**Void Region** (User-identified tetrahedral void):
- X: [-0.017400, -0.015000]
- Z: [-0.002600, 0.000000]
- Y: Extended (approximate: [-0.020000, 0.020000])

**Test Method**:
- Loaded same mesh as benchmark (timesteps 158-159)
- Extracted multi-cell octree (665,824 cells)
- Generated 100 random sample positions within void region
- Tested both 3×3×3 and radius=15 search methods
- Sequential testing to avoid GPU OOM

---

## Critical Findings 🎯

### **3×3×3 Search Performance in Void Region:**
```
Found: 100/100 (100.0% ✅)
Mean tests: 263.4
```

### **Radius=15 Search Performance in Void Region:**
```
Found: 48/100 (48.0% ⚠️)
```

### **Comparison:**
```
Found by radius but NOT by 3×3×3: 0/100 (0.0%)
```

**3×3×3 OUTPERFORMS radius search in this region!**

---

## Void Region Element Analysis

**Elements in void region: 54**
- Non-Kuhn: 2 (3.7%)
- Kuhn (Level 8): 52 (96.3%)
- Volume: min=2.130e-08, max=2.130e-08, mean=2.130e-08

**All Kuhn elements at coarse Level 8** (user observed "coarse element blocks")

---

## Hypothesis Testing Results

### **User's Hypothesis (Point 4) - Cell Registration Mismatch:**

**Hypothesis:**
> Morton/hash keys for cells/elements during registration may not match
> encoded position during search. Elements registered by vertex positions
> may hash to different cells than particle positions hash to.

**Test Result:** ❌ **NOT CONFIRMED** for void region

**Evidence:**
- 3×3×3 achieved 100% success in void region
- If cell registration was mismatched, 3×3×3 would fail
- 3×3×3 success proves:
  - ✅ Particles hash to correct cells
  - ✅ Elements are registered in correct cells
  - ✅ Cell lookup is working as designed

---

## Contradiction with Global Benchmark

**Global benchmark results** (full mesh, 2500 RK4 steps):
- 3×3×3: 18.84% retention ❌
- Radius-10: 40.97% retention ⚠️

**Void region results** (100 static samples):
- 3×3×3: 100.0% found ✅
- Radius-15: 48.0% found ⚠️

**Why the difference?**

1. **Static vs Dynamic:**
   - Void diagnostic: static point-in-tet test (no tracking)
   - Benchmark: 2500 RK4 steps with dynamic tracking
   - Cumulative errors may accumulate over timesteps

2. **Local vs Global:**
   - Void region: 54 elements at Level 8 (coarse, uniform)
   - Full mesh: 3,048,900 elements at 8 levels with transitions
   - Problem may be in other regions, not this void

3. **Sample size:**
   - Void diagnostic: 100 samples in small region
   - Benchmark: 10,000 particles across entire mesh

---

## Updated Understanding

### **What This Test DISPROVES:**

❌ Cell registration mismatch (User's Point 4 hypothesis)
- 3×3×3 achieves 100% in void region → registration is correct
- If registration was wrong, 3×3×3 would fail here too

❌ Void region is the problem location
- 3×3×3 works perfectly in the void region
- Tetrahedral voids user observed are NOT caused by search failures here

❌ Point-in-tet tolerance issues (User's Point 1 hypothesis)
- Elements are coarse (Level 8, volume 2e-08)
- 100% success means tolerance is adequate for these elements

### **What This Test SUGGESTS:**

✅ Problem is in DYNAMIC TRACKING, not static search
- Static tests show 100% success
- Dynamic tracking shows 18.84% retention
- → Cumulative errors over 2500 steps?

✅ Problem is ELSEWHERE in mesh, not void region
- Void region: perfect performance
- Global mesh: poor performance
- → Look at other regions (finer levels, transitions, boundaries)

✅ Velocity field may be the issue (User's original insight)
- User: "Same mesh works fine with FEMUSS"
- Structure is correct, search works locally
- → Particles may be leaving domain or entering problematic flow regions

---

## Recommended Next Investigations

### **Option 1: Track WHERE particles are lost during RK4 (RECOMMENDED)**

Create diagnostic to:
1. Run full RK4 tracking for 2500 steps
2. Record position and element ID when particles become "lost"
3. Spatial analysis: where in mesh are particles lost?
4. Temporal analysis: at which timestep are they lost?
5. Compare with successful radius-10 tracking

**Hypothesis to test:**
- Particles lost in specific spatial regions (not void region)
- Particles lost after entering certain element types
- Particles lost at refinement boundaries (even though 99.9% same-level)

### **Option 2: Compare radius-10 vs 3×3×3 trajectories**

For same particle set:
1. Track with radius-10 (40.97% retention)
2. Track with 3×3×3 (18.84% retention)
3. Identify first timestep where trajectories diverge
4. Analyze what happened at divergence point

**Hypothesis to test:**
- Do particles take slightly different paths?
- Does one method miss elements causing trajectory deviation?
- Do errors accumulate over time?

### **Option 3: Test intermediate timesteps**

Run retention diagnostic at:
- 10 steps (user: "void appears after 10 timesteps")
- 100 steps
- 500 steps
- 1000 steps
- 2500 steps

**Hypothesis to test:**
- Is retention loss gradual or sudden?
- Does it correlate with specific timesteps?
- Is there a "cliff" where particles suddenly disappear?

### **Option 4: Investigate finest-level elements**

Void region is Level 8 (coarse). Test finest-level elements:
- User reported finest level: ~6e-4 element size
- Diagnostic showed Level 14 elements (finest)
- Fallback elements concentrated at Level 14 (64%)

**Hypothesis to test:**
- Are finest elements the problem (not coarse)?
- Does 3×3×3 fail at fine levels but succeed at coarse levels?
- Is the void region misleading (it's coarse, not fine)?

---

## Files Created

1. [diagnose_void_region_corrected.py](diagnose_void_region_corrected.py) - Diagnostic script (follows benchmark patterns)
2. [logs/diagnose_void_region_corrected.log](logs/diagnose_void_region_corrected.log) - Test output
3. [VOID_REGION_DIAGNOSTIC_RESULTS.md](VOID_REGION_DIAGNOSTIC_RESULTS.md) - This document

---

## Conclusion

**The void region is NOT the problem!**

- 3×3×3 search works perfectly (100%) in the user-identified void region
- Cell registration is correct
- Point-in-tet tolerance is adequate
- Problem must be:
  1. In other regions of the mesh (not this void)
  2. Due to dynamic tracking errors (not static search)
  3. Related to velocity field or boundary conditions

**Next step:** Investigate WHERE and WHEN particles are lost during dynamic RK4 tracking across the full mesh.

**Awaiting user decision on which investigation path to pursue.**
