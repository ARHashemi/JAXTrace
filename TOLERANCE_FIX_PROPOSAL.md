# Point-in-Tet Tolerance Fix Proposal
**Date**: 2026-02-16
**Root Cause**: Numerical instability in barycentric coordinate calculation for finest-level elements

---

## Problem Identified

### **Evidence from Particle Visualizations:**

**Radius-10 baseline:**
- Small tetrahedral voids (fewer particles lost)
- 40.97% retention @ 2500 steps

**Mesh-Aligned 3×3×3:**
- Large tetrahedral voids (more particles lost)
- 18.84% retention @ 2500 steps

**Key observation:** Both methods show **element-shaped voids** → particles are **skipping specific elements** during tracking.

### **Root Cause Analysis:**

From [logs/diagnose_degenerate_elements.log](logs/diagnose_degenerate_elements.log):
```
Volume Statistics (finest level):
  Min:    8.124e-14
  Median: 8.124e-14  ← 85% of elements at this level
```

From [jaxtrace/gpu/search/point_in_tet_methods.py](jaxtrace/gpu/search/point_in_tet_methods.py:193):
```python
tol = -1e-6  # Fixed tolerance
inside = (lambda0 >= tol) & (lambda1 >= tol) & (lambda2 >= tol) & (lambda3 >= tol)
```

### **The Problem:**

For elements with volume `V = 8e-14`:

```python
# Barycentric calculation:
lambda_i = V_i / V_0

# If V_0 = 8e-14 and V_i has floating-point error of ~1e-16:
error_amplification = 1e-16 / 8e-14 = 1.25e-3

# Error of 1.25e-3 >> tolerance of 1e-6
# Result: Valid particles rejected as "outside"!
```

**Numerical error gets amplified by factor of 10^13 for finest elements!**

---

## Proposed Fix

### **Adaptive Tolerance Based on Element Volume**

Instead of fixed `tol = -1e-6`, scale tolerance based on element volume:

```python
# Current (BROKEN for fine elements):
tol = -1e-6

# Proposed fix:
tol = -jnp.maximum(1e-6, jnp.abs(V0) * 1e-8)
```

### **Rationale:**

- **Large elements** (V ~ 1e-8): `tol = -1e-6` (current behavior)
- **Medium elements** (V ~ 1e-10): `tol = -1e-10 * 1e-8 = -1e-18` (tighter)
- **Fine elements** (V ~ 8e-14): `tol = -8e-14 * 1e-8 = -8e-22` (very tight)

Wait, this is **backwards**! We need **LOOSER** tolerance for fine elements:

### **Corrected Proposal:**

```python
# Scale tolerance with INVERSE of volume (looser for finer elements)
relative_tol = 1e8  # Relative tolerance factor
tol = -jnp.maximum(1e-6, relative_tol * jnp.abs(V0))
```

No, this explodes for fine elements. Let me think more carefully...

### **Actually Correct Proposal:**

The issue is that when volume is small, numerical errors in computing `V_i` get divided by small `V_0`, amplifying errors. We need tolerance to account for this:

```python
# Numerical error in V_i ≈ machine epsilon × typical edge length^3
# When divided by V_0, get: error/V_0 ≈ ε × L^3 / V_0
# For V_0 ≈ L^3: error/V_0 ≈ ε (good)
# For V_0 << L^3 (degenerate): error/V_0 >> ε (bad)

# Solution: Scale tolerance by reciprocal of volume
edge_length_characteristic = jnp.sqrt(jnp.sum(v1**2 + v2**2 + v3**2) / 3.0)
volume_characteristic = edge_length_characteristic ** 3
volume_quality = jnp.abs(V0) / jnp.maximum(volume_characteristic, 1e-15)

# If volume is much smaller than characteristic volume, use looser tolerance
tol = -1e-6 / jnp.maximum(volume_quality, 0.1)
```

This is getting complicated. Let me try **simpler approach**:

### **SIMPLEST FIX: Just Increase Tolerance**

```python
# Current:
tol = -1e-6

# Proposed:
tol = -1e-4  # 100× looser (accommodate numerical errors)
```

**Pros:**
- Simple one-line change
- Accommodates numerical errors for fine elements
- Still rejects particles clearly outside

**Cons:**
- May accept particles slightly outside (< 0.01% of element size)
- Not adaptive

### **RECOMMENDED FIX: Volume-Scaled Tolerance**

```python
# Estimate element size from edge length
edge_length_sq = jnp.sum(v1**2)  # Representative edge length squared
element_scale = jnp.sqrt(edge_length_sq)  # Characteristic length

# Scale tolerance by element size
# For element size 0.000625 (finest level): tol = -6.25e-7 (current)
# For element size 0.04 (coarsest): tol = -4e-5 (100× looser)
tol = -1e-6 * jnp.maximum(1.0, element_scale / 0.001)
```

No wait, this still doesn't address the core issue...

---

## ACTUAL ROOT CAUSE (Refined Understanding)

Looking at line 171:
```python
is_degenerate = V0_abs < 1e-12 * jnp.maximum(expected_vol, 1e-15)
```

For finest elements with `V0 = 8e-14`:
- `expected_vol ≈ (6e-4)^3 = 2.16e-10`
- Threshold: `1e-12 * 2.16e-10 = 2.16e-22`
- `8e-14 >> 2.16e-22` → **NOT flagged as degenerate** ✅

So degeneracy check is OK. The issue is purely in the tolerance.

## FINAL PROPOSED FIX

**File**: `jaxtrace/gpu/search/point_in_tet_methods.py`

**Line 193**: Change tolerance from `-1e-6` to `-1e-4`

```python
# Before:
tol = -1e-6

# After:
tol = -1e-4  # Accommodate numerical errors in fine-resolution elements
```

**Rationale:**
- Finest elements (size ~6e-4) have volume ~8e-14
- Numerical errors amplified by ~10^13 factor
- Tolerance of 1e-6 is too tight for these errors
- Tolerance of 1e-4 is ~0.016% of element size (acceptable)
- Commercial codes likely use similar tolerances

---

## Alternative Fix (More Conservative)

If `-1e-4` is too aggressive, try intermediate value:

```python
tol = -1e-5  # 10× looser than current, 10× tighter than aggressive fix
```

---

## Testing Plan

### **1. Quick Test - Change Tolerance and Re-Run Benchmark**

```bash
# Edit jaxtrace/gpu/search/point_in_tet_methods.py line 193
# Change: tol = -1e-6
# To:     tol = -1e-4

# Re-run benchmark
python benchmark_l2_search_methods_with-export.py 2>&1 | tee logs/benchmark_tolerance_fix.log
```

**Expected outcome:**
- Retention should improve from 18.84% toward 40-95%
- Tetrahedral voids should shrink or disappear
- Compare particle distribution images

### **2. Verify with Diagnostic**

Create `diagnose_tolerance_impact.py`:
```python
# For sample of particles that were "lost":
1. Run point-in-tet test with tol=-1e-6 (current)
2. Run point-in-tet test with tol=-1e-4 (proposed)
3. Count how many become "found" with looser tolerance
4. Check barycentric coordinate values for marginal cases
```

### **3. Compare with Radius-10 Baseline**

If tolerance fix brings mesh-aligned retention close to radius-10 (40.97%), then problem is solved!

---

## Expected Impact

### **Scenario 1: Tolerance is THE Problem**
- Retention improves to 35-45% (matching radius-10)
- Tetrahedral voids disappear
- Solution: This fix + optimize search further

### **Scenario 2: Tolerance is A Problem (Not THE Only One)**
- Retention improves to 25-30% (partial improvement)
- Some voids remain
- Solution: This fix + investigate remaining issues

### **Scenario 3: Tolerance is NOT the Problem**
- Retention stays at ~19% (no improvement)
- Voids unchanged
- Solution: Investigate other hypotheses

---

## Implementation Steps

1. **Make the change** (one line):
   ```python
   # jaxtrace/gpu/search/point_in_tet_methods.py:193
   tol = -1e-4
   ```

2. **Re-compile** (if needed - JAX should auto-recompile)

3. **Run benchmark**:
   ```bash
   python benchmark_l2_search_methods_with-export.py 2>&1 | tee logs/benchmark_tolerance_fix.log
   ```

4. **Check retention**:
   - Look for "Mesh-Aligned Multi-Cell + 3×3×3 Local"
   - Compare retention @ 2500 steps (was 18.84%)

5. **Visualize particles** (after 100 timesteps):
   - Check if tetrahedral voids are reduced
   - Compare with baseline radius-10

6. **If successful**: Consider even looser tolerance or adaptive scaling
7. **If unsuccessful**: Move to next hypothesis (Option 2-5)

---

## User Decision Required

**Should we proceed with this fix?**

Options:
1. **Yes, try tol=-1e-4** (aggressive fix, high chance of improvement)
2. **Yes, try tol=-1e-5** (conservative fix, moderate chance)
3. **No, investigate further first** (create diagnostic to measure tolerance impact)
4. **Other approach** (your suggestion)

Awaiting your decision before making any code changes!
