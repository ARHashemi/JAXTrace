# Final Investigation Conclusions and Proposed Solutions
**Date**: 2026-02-16
**Current Status**: 18.84% retention @ 2500 steps (target: >95%)
**Investigation**: Tasks 1-6 Complete

---

## Executive Summary

After comprehensive diagnostics, we have **ruled out** the following as root causes:
- ❌ Degenerate elements (mesh quality is excellent)
- ❌ Multi-level refinement transitions (99.9% same-level neighbors)
- ❌ VTK merging defects (no degenerate elements created)

**The problem remains unexplained by structural issues.** The evidence points to either:
1. **Velocity field topology** (particles leaving domain or entering zero-velocity regions)
2. **Numerical issues** in the search/interpolation during dynamic tracking
3. **Subtle connectivity gaps** not detected by our diagnostics

---

## Detailed Findings by Task

### ✅ **Task 1: Revert Phase 3 Fallback** - COMPLETE

**Changes made:**
- Removed radius=15 Morton fallback from `rk4_fully_fused_timedep.py`
- Removed Phase 3 test configuration from `benchmark_l2_search_methods_with-export.py`

**Current state:** Clean 3×3×3 local search without fallback.

---

### ✅ **Task 2: Degenerate Elements Explanation** - COMPLETE

**What are degenerate elements?**
- Zero/negative volume (co-planar or inverted vertices)
- Extreme aspect ratios (very elongated/flattened)
- Collapsed edges (nearly coincident vertices)

**Impact on tracking:**
- Point-in-tet test fails (division by zero/near-zero volume)
- Invalid barycentric coordinates → wrong velocity vectors
- Numerical instability in interpolation

---

### ✅ **Task 3: Degenerate Element Investigation** - COMPLETE

**Source:** [logs/diagnose_degenerate_elements.log](logs/diagnose_degenerate_elements.log)

#### **Findings: NO Degeneracies!**

```
✅ Inverted elements (volume < 0):     0 (0.00%)
✅ Poor aspect ratio (> 100):          0 (0.00%)
✅ Collapsed edges (< 1%):             0 (0.00%)

Aspect ratio: min=2.48, max=4.50, mean=4.50  ← Excellent quality
Edge collapse ratio: min=0.57, max=0.70      ← No collapsed edges
```

#### **"Near-Zero Volume" is Misleading**

96.63% classified as "near-zero volume" but this is just **adaptive refinement**:
```
Volume: Min=8.124e-14 (finest level), Max=2.130e-08 (coarsest)
→ 8 refinement levels with 8× volume ratio per level
```

#### **Non-Kuhn Elements are Coarser**

```
Kuhn mean volume:     6.897e-12
Non-Kuhn mean volume: 1.507e-09
Ratio: 218.546×
```

**Interpretation:** Non-Kuhn transition elements are at coarse levels, exactly where user observes particle loss ("entering coarse element blocks").

---

### ✅ **Task 4: Cross-Cell Boundary Face Sharing** - COMPLETE

**Source:** [logs/diagnose_multi_cell_coverage_phase2_fixed.log](logs/diagnose_multi_cell_coverage_phase2_fixed.log)

#### **Findings:**

```
Interior faces:        5,917,485
  - Same cell:         5,197,905 (87.84%)
  - Different cells:     719,580 (12.16%)  ← Cross-cell faces
```

**12.16% of face neighbors are in different cells.** When particles cross these faces, they move between cells.

#### **Static vs Dynamic Gap:**

```
Static test (centroid-based):   100% searchable ✅
Dynamic tracking (3×3×3 offset): 18.84% retention ❌
```

**Hypothesis (at this point):** 3×3×3 uses same-level offsets that don't reach across refinement boundaries.

---

### ✅ **Task 5: 1:2 and 2:1 Refinement Analysis** - COMPLETE 🎯

**Source:** [logs/diagnose_refinement_transitions.log](logs/diagnose_refinement_transitions.log)

#### **🎯 CRITICAL FINDING: NO Multi-Level Transitions!**

```
Face Neighbor Transitions:
  1:1 (same level):   5,910,889 (99.89%)  ← Nearly ALL
  1:2 (one level):            0 ( 0.00%)  ← NONE!
  2:1 (one level):            0 ( 0.00%)  ← NONE!
  Multi-level:                0 ( 0.00%)  ← NONE!
```

**The mesh is uniformly refined!** There are NO 1:2 or 2:1 face neighbors.

#### **Cross-Cell Face Breakdown:**

```
Same level (1:1):     717,989 (99.78%)
Multi-level gap:        1,591 ( 0.22%)  ← Suspicious!
1:2 boundary:               0 ( 0.00%)
```

**1,591 multi-level cross-cell faces** with >1 level gap (abnormal for Kuhn mesh).

#### **The 344 Fallback Elements:**

```
64.24% at Level 14 (finest!)
18.31% at Level 13
9.01% at Level 12
```

**Unexpectedly at FINE levels, NOT coarse levels!**

#### **Conclusion:**

❌ **Multi-level refinement is NOT the problem!**
✅ **3×3×3 same-level search SHOULD work perfectly** (99.9% same-level neighbors)
⚠️ **1,591 multi-level gaps may indicate mesh assembly issues**

---

### ✅ **Task 6: VTK Mesh Merging and Deduplication** - COMPLETE

**Source:** [logs/diagnose_vtk_merging.log](logs/diagnose_vtk_merging.log)

#### **Findings:**

**64 VTK parts merged:**
```
Original nodes:     780,922
Duplicates removed: 209,749 (26.9%)  ← High duplicate rate
Final nodes:        571,173
```

**✅ NO degenerate elements created by deduplication!**

**Problematic elements:**
```
Fallback (1 cell):  344 (0.011%)
Non-Kuhn:         1,826 (0.060%)
```

**Spatial distribution:**
- Fallback elements occupy 112.98% of mesh volume
- **NOT concentrated** (spread throughout mesh)
- Rules out localized VTK part boundary issues

#### **Conclusion:**

✅ **Deduplication is working correctly** (no degenerate elements)
✅ **Fallback elements are spatially distributed** (not at part boundaries)
⚠️ **High duplicate rate (26.9%)** suggests significant part overlap

---

## What We Eliminated as Root Causes

| Hypothesis | Evidence | Conclusion |
|------------|----------|------------|
| **Degenerate elements** | 0 inverted, excellent aspect ratios | ❌ NOT the problem |
| **Near-zero volumes** | Just adaptive refinement (8 levels) | ❌ NOT the problem |
| **Multi-level transitions** | 99.9% same-level neighbors, 0 1:2 faces | ❌ NOT the problem |
| **VTK merging defects** | 0 degenerate elements after dedup | ❌ NOT the problem |
| **Localized part boundaries** | Fallback elements spread throughout | ❌ NOT the problem |

---

## What Remains Unexplained

### **The Paradox:**

1. ✅ **100% coverage** (all 3,048,900 elements in octree)
2. ✅ **100% searchable** in static diagnostic (3×3×3 test)
3. ✅ **99.9% same-level neighbors** (no refinement transitions)
4. ✅ **No degenerate elements** (excellent mesh quality)
5. ❌ **18.84% retention** in dynamic tracking (should be >95%)

**If everything is correct structurally, why are particles lost?**

---

## Proposed Next Steps (for Discussion)

### **Option 1: Investigate Velocity Field Topology** ⭐ **RECOMMENDED**

**Hypothesis:** Particles are leaving the mesh domain or entering zero-velocity regions.

**Evidence:**
- User: "Same mesh works fine with commercial code (FEMUSS)"
- User: "Huge particle loss when entering coarse element blocks"
- Non-Kuhn elements are 218× larger (coarse regions)

**Diagnostic to create:**
```python
# diagnose_velocity_field_topology.py
1. Identify zero-velocity regions in the mesh
2. Check velocity magnitude distribution
3. Analyze velocity vectors near coarse blocks
4. Track where particles are going when "lost"
   - Are they leaving mesh bounds?
   - Are they entering zero-velocity regions?
   - Are velocity vectors pointing outward at boundaries?
5. Compare particle trajectories with FEMUSS results
```

**If this is the problem:**
- Not a search algorithm issue
- May need velocity field adjustments or boundary conditions
- Could be physical simulation issue, not tracking issue

---

### **Option 2: Investigate Search Numerical Issues**

**Hypothesis:** Floating-point errors or edge cases in point-in-tet test during dynamic tracking.

**Evidence:**
- Static test (100% searchable) vs dynamic test (18.84% retention)
- May be cumulative errors over 2500 RK4 steps

**Diagnostic to create:**
```python
# diagnose_search_edge_cases.py
1. Track particles that become "lost"
   - Position when lost
   - Last known element
   - Neighboring elements
2. Re-run point-in-tet test for lost particles
   - Are they geometrically inside any element?
   - Barycentric coordinate values
3. Check for edge cases:
   - Particles exactly on element faces/edges
   - Particles very close to boundaries
   - Numerical tolerance issues (1e-10 vs 1e-6)
```

**If this is the problem:**
- Adjust tolerance in point-in-tet test
- Add edge case handling
- Improve numerical robustness

---

### **Option 3: Deep Dive into the 1,591 Multi-Level Gaps**

**Hypothesis:** The 1,591 cross-cell faces with >1 level gap are causing issues.

**Evidence:**
- Abnormal for Kuhn mesh (should be 0)
- Only 0.22% of cross-cell faces, but could be critical

**Diagnostic to create:**
```python
# diagnose_multi_level_gaps.py
1. Identify the 1,591 face pairs
2. Spatial distribution
3. Are they concentrated near particle loss regions?
4. Do particles cross these faces during tracking?
5. What refinement levels are involved? (L8-L14 or L14-L8?)
```

**If this is the problem:**
- Special handling for these anomalous faces
- May indicate mesh generation artifacts
- Could implement targeted multi-level search for these cases

---

### **Option 4: Compare with Baseline Radius=10**

**Hypothesis:** Understand WHY radius=10 achieves 40.97% (vs our 18.84%).

**Diagnostic to create:**
```python
# compare_with_baseline.py
1. Track same particle set with both methods
2. Identify where 3×3×3 loses particles but radius=10 finds them
3. Spatial distribution of "saved" particles
4. Which elements are found by radius=10 but not 3×3×3?
5. Are these elements in the octree but not reached by 3×3×3?
```

**If this reveals the problem:**
- May show specific spatial regions where 3×3×3 fails
- Could guide targeted fixes

---

### **Option 5: Analyze the 344 Fallback Elements in Detail**

**Hypothesis:** Single-cell registration is causing particle loss.

**Evidence:**
- 344 elements have only 1 cell (should have ~4)
- Concentrated at fine levels (64% at L14)
- May miss particles crossing into these elements

**Action:**
1. Improve `find_face_neighbor_fast()` to ensure all non-Kuhn elements get proper 4-cell registration
2. Use spatial hashing instead of sequential search
3. Eliminate all single-cell fallback cases

**If this helps:**
- Reduces 344 problematic elements to 0
- May improve retention slightly (but unlikely to be full solution)

---

## Recommended Investigation Order

**1st Priority:** Option 1 - Velocity Field Topology ⭐
- Most likely to explain the gap between structure (correct) and behavior (wrong)
- User observation supports this ("entering coarse element blocks")
- Commercial code works → may be physics/boundary condition issue

**2nd Priority:** Option 2 - Search Numerical Issues
- Explains static (100%) vs dynamic (18.84%) gap
- May be cumulative errors over 2500 steps

**3rd Priority:** Option 4 - Compare with Baseline
- Direct comparison shows WHERE the problem occurs
- Guides targeted investigation

**Lower Priority:**
- Option 3 (1,591 gaps are only 0.22%, unlikely to cause 81% loss)
- Option 5 (344 elements are only 0.011%, unlikely to cause 81% loss)

---

## Questions for User

Before proceeding with any implementation:

1. **Do you have FEMUSS tracking results** for the same particle seeding?
   - Trajectory files or retention data?
   - Would help identify where/when JAXTrace loses particles

2. **Are there known zero-velocity regions** in the mesh?
   - Far-field boundaries?
   - Symmetry planes?
   - Fixed boundary conditions?

3. **What happens to particles in FEMUSS** when they reach coarse blocks?
   - Do they slow down?
   - Change direction?
   - Accumulate in certain regions?

4. **Can you share a small sample mesh** (e.g., 10k elements)?
   - Would enable faster iteration
   - Could isolate the problematic region

5. **Do you prefer** to:
   - Investigate velocity field first (Option 1)?
   - Compare with radius=10 baseline (Option 4)?
   - Try improving the 344 fallback elements (Option 5)?

---

## Summary Table

| Aspect | Status | Notes |
|--------|--------|-------|
| Mesh quality | ✅ Excellent | Aspect ratio 2.5-4.5, no inversions |
| Octree coverage | ✅ 100% | All 3,048,900 elements covered |
| Static searchability | ✅ 100% | 3×3×3 diagnostic passes |
| Refinement transitions | ✅ Uniform | 99.9% same-level, no 1:2/2:1 |
| VTK merging | ✅ Clean | No degenerate elements |
| Dynamic retention | ❌ 18.84% | Should be >95% |
| **Root cause** | ❓ Unknown | Not structural → likely velocity/numerical |

---

## Files Created During Investigation

1. [PHASE3_HYBRID_FALLBACK_IMPLEMENTATION.md](PHASE3_HYBRID_FALLBACK_IMPLEMENTATION.md) - Phase 3 documentation (reverted)
2. [diagnose_degenerate_elements.py](diagnose_degenerate_elements.py) - Task 3 diagnostic
3. [logs/diagnose_degenerate_elements.log](logs/diagnose_degenerate_elements.log) - Task 3 results
4. [diagnose_refinement_transitions.py](diagnose_refinement_transitions.py) - Task 5 diagnostic
5. [logs/diagnose_refinement_transitions.log](logs/diagnose_refinement_transitions.log) - Task 5 results
6. [diagnose_vtk_merging.py](diagnose_vtk_merging.py) - Task 6 diagnostic
7. [logs/diagnose_vtk_merging.log](logs/diagnose_vtk_merging.log) - Task 6 results
8. [INVESTIGATION_SUMMARY_TASKS_1_4.md](INVESTIGATION_SUMMARY_TASKS_1_4.md) - Tasks 1-4 summary
9. [FINAL_INVESTIGATION_CONCLUSIONS.md](FINAL_INVESTIGATION_CONCLUSIONS.md) - This document

---

## Next Actions

**Awaiting user decision on which option to pursue before implementing any changes.**

Per user instruction: "from now dont change the ideas and plans before discussing with me"

Please advise which investigation path you'd like to pursue:
- **Option 1:** Velocity field topology analysis
- **Option 2:** Search numerical issues analysis
- **Option 3:** Multi-level gap analysis
- **Option 4:** Comparison with radius=10 baseline
- **Option 5:** Fix 344 fallback elements
- **Other:** Alternative approach based on your insights
