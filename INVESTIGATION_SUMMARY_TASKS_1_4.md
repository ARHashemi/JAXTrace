# Investigation Summary: Tasks 1-4
**Date**: 2026-02-16
**Status**: Phase 1+2 Complete, Retention at 18.84% (target: >95%)

---

## Task 1: ✅ Revert Phase 3 Fallback

**Changes reverted:**
- [rk4_fully_fused_timedep.py](jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py): Removed conditional radius=15 fallback
- [benchmark_l2_search_methods_with-export.py](benchmark_l2_search_methods_with-export.py): Removed Phase 3 test config

**Current state**: Clean 3×3×3 local search without fallback mechanism.

---

## Task 2: ✅ Degenerate Elements - Definition and Impact

### What Are Degenerate Elements?

**Traditional degenerate elements** have severe geometric defects:
1. **Zero/negative volume** - vertices co-planar or inverted
2. **Extreme aspect ratios** - very elongated/flattened
3. **Collapsed edges** - nearly coincident vertices

### How They Cause Particle Loss:

#### Point-in-Tet Test Failures:
```python
# Barycentric coordinate calculation
volume = det([v1-v0, v2-v0, v3-v0]) / 6
coords[i] = det([...]) / volume  # Division by zero/near-zero!
```

**For degenerate elements:**
- Zero volume → NaN/Inf in barycentric coords
- Near-zero volume → numerical amplification of floating-point errors
- Inverted volume → negative coords even for valid positions

#### Velocity Interpolation Issues:
- Invalid barycentric coords → wrong velocity vectors
- Particles "jump" to unexpected locations
- Accumulates over RK4 substeps

---

## Task 3: ✅ Degenerate Element Investigation

### **Findings: NO Traditional Degeneracies!**

From [logs/diagnose_degenerate_elements.log](logs/diagnose_degenerate_elements.log):

#### ✅ **Geometric Quality is EXCELLENT:**
```
Inverted elements (volume < 0):     0 (0.00%)
Poor aspect ratio (> 100):          0 (0.00%)
Collapsed edges (< 1%):             0 (0.00%)

Aspect ratio: min=2.48, max=4.50, mean=4.50
  → All elements well-shaped (4.5 is near-optimal for tets)

Edge collapse ratio: min=0.57, max=0.70
  → No collapsed edges (all > 57%)
```

#### ⚠️ **BUT: 96.63% Classified as "Near-Zero Volume"**

**This is MISLEADING!** It's not degeneracy - it's **adaptive mesh refinement**:

```
Volume Statistics:
  Min:    8.124e-14  ← Finest level (level 14)
  Median: 8.124e-14  ← Most elements at finest level
  Mean:   2.386e-10  ← Pulled up by coarser elements
  Max:    2.130e-08  ← Coarsest level (level 7)
```

**Refinement hierarchy:** ~8 levels with 8× volume ratio per level (2^3 for 3D halving)

#### 🔍 **Critical Finding: Non-Kuhn Elements are MUCH Coarser**

```
Kuhn elements:     mean volume = 6.897e-12  (fine)
Non-Kuhn elements: mean volume = 1.507e-09  (coarse)
Ratio: 218.546×
```

**Interpretation:**
- Non-Kuhn transition elements are at **coarse refinement levels**
- Located at boundaries between VTK parts or mesh quality transitions
- **Exactly where you observe particle loss: "entering coarse element blocks"!**

#### 📍 **Spatial Concentration of "Degenerate" Elements**

```
X: min=-0.011, max=0.010
Y: min=-0.011, max=0.011
Z: min=-0.006, max=0.000
```

**Very small spatial region** - suggests localized refinement or mesh part boundary.

### **Conclusion: Degenerate Elements are NOT the Problem**

- Mesh quality is excellent
- "Near-zero volume" is just adaptive refinement artifact
- Non-Kuhn elements are coarser transition elements (expected behavior)
- **Particle loss is NOT caused by geometric degeneracy**

---

## Task 4: 🔍 Cross-Cell Boundary Face Sharing Analysis

From [logs/diagnose_multi_cell_coverage_phase2_fixed.log](logs/diagnose_multi_cell_coverage_phase2_fixed.log):

### **The Data:**

```
Interior faces (2 elements):   5,917,485 total
  - Same cell:                 5,197,905 (87.84%)
  - Different cells:             719,580 (12.16%)  ← Cross-cell faces
```

### **What Does This Mean?**

#### **12.16% of Face Neighbors are in Different Cells**

When a particle crosses these faces during tracking:
1. Particle leaves element A (in cell X)
2. Enters element B (in cell Y, where Y ≠ X)
3. **If searching only cell X → element B not found → particle lost!**

#### **Why 3×3×3 Should Handle This:**

The 3×3×3 search covers **27 cells** centered on the current particle position:
- Center cell + all 26 immediate neighbors
- Grid offsets: `[-1, 0, 1]` in each dimension

**Question:** If 3×3×3 covers 27 cells, why are particles still lost?

### **Possible Issues:**

#### 1. **Multi-Level Refinement Complication**

The 12.16% cross-cell faces might span **multiple refinement levels**:

Example scenario:
```
Fine cell A (level 14, size = 0.0000625) contains element 1
Coarse cell B (level 7, size = 0.04) contains element 2
Elements 1 and 2 share a face (1:2 refinement)
```

**Problem:** When searching from fine cell A:
- 3×3×3 offset of ±1 fine cell = ±0.0000625 spatial distance
- Coarse cell B is 640× larger (0.04 / 0.0000625)
- **Offset of ±1 fine cell doesn't reach coarse cell B!**

Current implementation:
```python
# In mesh_aligned_point_location.py
neighbor_indices = base_indices + offset  # offset in [-1, 0, 1]
```

**This uses SAME-LEVEL offsets only!**

#### 2. **The 344 Elements with Only 1 Cell**

From diagnostic log:
```
⚠️  WARNING: 344 elements have < 4 cells
    These elements may cause particle loss at cell boundaries!
```

These are non-Kuhn elements that used fallback registration:
- Registered in only **1 coarse cell** (level 8, size=0.04)
- Should have been in **4 cells** like other elements
- **Gap in coverage** when particles move from fine → coarse regions

### **Root Cause Hypothesis:**

**Multi-level cell transitions + fallback registration gaps**

When particles move from fine to coarse mesh regions:
1. Current cell is fine (level 14)
2. 3×3×3 offset searches fine-level neighbors (±0.0000625)
3. Next element is in coarse cell (level 7-8, size 0.04)
4. **Fine-level offset too small to reach coarse-level cell**
5. Search fails → particle marked as lost

**This explains:**
- ✅ Why 100% searchable in **static diagnostic** (uses centroid-based cell lookup, not 3×3×3 offset)
- ✅ Why 18.84% retention in **dynamic tracking** (particles crossing fine→coarse boundaries)
- ✅ Why "huge particle loss when entering coarse element blocks" (user observation)

---

## Task 5: PENDING - 1:2 and 2:1 Refinement

**Key questions to investigate:**
1. How are 1:2 face neighbors currently handled in neighbor table?
2. Does multi-level search (levels 14→7) properly handle refinement transitions?
3. Are the 719,580 cross-cell faces concentrated at refinement boundaries?

---

## Task 6: PENDING - VTK Mesh Merging and Deduplication

**Key questions to investigate:**
1. Are the 344 fallback elements at VTK part boundaries?
2. Does deduplication create connectivity gaps between parts?
3. Are coarse element blocks from separate VTK parts?

---

## Proposed Solution Approaches

### **Option A: Multi-Level 3×3×3 Search**

Modify search to check **multiple refinement levels**:

```python
# For each of 27 offset cells:
for offset in offsets_3x3x3:
    # Try current level
    cell = find_cell(base_indices + offset, current_level)
    if cell found: search_elements_in_cell(cell)

    # Try coarser levels (parent cells)
    for parent_level in [current_level - 1, ..., min_level]:
        parent_cell = find_cell(base_indices // scale + offset, parent_level)
        if parent_cell found: search_elements_in_cell(parent_cell)
```

**Pros:**
- Handles fine→coarse transitions
- Covers 1:2 and 2:1 face neighbors
- No structural changes to octree

**Cons:**
- More cells to search per particle (~27 → ~54-80)
- More complex logic

### **Option B: Fix the 344 Fallback Elements**

Improve face neighbor finding for non-Kuhn elements:
- Instead of searching only recent 100 elements, use spatial hash
- Ensure all non-Kuhn elements get proper 4-cell registration
- Eliminate single-cell fallback cases

**Pros:**
- Reduces coverage gaps
- Simpler than multi-level search

**Cons:**
- Doesn't solve general fine→coarse transition problem
- 344 elements may not be the only issue

### **Option C: Radius-Based L2 Fallback (User Rejected)**

~~Use radius search as fallback when 3×3×3 fails~~

**User feedback:** "It is not a real solution"

### **Option D: Investigate Velocity Field/Domain Boundaries**

If particles are leaving the mesh domain:
- Check for zero-velocity regions (velocity closure)
- Verify mesh boundaries are properly closed
- Analyze if VTK part merging creates gaps

---

## Recommended Next Steps

**Before implementing any changes, we should:**

1. **Run Task 5 diagnostic:** Analyze 1:2/2:1 refinement distribution
   - How many of the 719,580 cross-cell faces are at refinement boundaries?
   - Are these concentrated at specific levels?

2. **Run Task 6 diagnostic:** Investigate VTK merging
   - Are the 344 fallback elements at part boundaries?
   - Spatial correlation with particle loss regions?

3. **Discuss with user:** Which option to pursue?
   - Multi-level search (comprehensive but complex)
   - Fix fallback registration (targeted but may be insufficient)
   - Other approaches based on Tasks 5-6 findings

**Do NOT implement without user approval!**

---

## Current Understanding

| Aspect | Status |
|--------|--------|
| Octree coverage | ✅ 100% (all 3,048,900 elements) |
| Static searchability | ✅ 100% (3×3×3 diagnostic) |
| Dynamic retention | ❌ 18.84% @ 2500 steps |
| Degenerate elements | ✅ None (mesh quality excellent) |
| Cross-cell faces | ⚠️ 12.16% (719,580 pairs) |
| Multi-level transitions | ⚠️ Suspected root cause |
| Fallback elements | ⚠️ 344 with single-cell registration |

**Gap:** Static tests pass, dynamic tracking fails → **search method works for same-level cells but fails at refinement transitions**
