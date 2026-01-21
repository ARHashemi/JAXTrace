# RK4 Fixes Test Plan - Step by Step

**Date**: 2025-12-22
**Based on**: FULLY_FUSED_TIMEDEP_RK4_CRITICAL_REVIEW.md

---

## Overview

This document provides step-by-step instructions for testing each fix to the RK4 fully-fused implementation. After each step, you should run the production test manually and report results.

---

## Step 1: Fix L1 Return Bug ✅ COMPLETED

### What Was Fixed

**File**: `jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py`
**Line**: 126-128

**Before**:
```python
return current_elem  # BUG: Returns cached elem even if not found
```

**After**:
```python
# CRITICAL FIX: Return -1 if search failed (not found after all hops)
# This ensures L2 fallback is triggered when L1 fails
return jnp.where(found, current_elem, jnp.int32(-1))
```

### Expected Impact

- L1 now properly returns `-1` when it fails to find a containing element
- L2 fallback will be triggered correctly
- Particles should no longer get stuck in wrong elements

### Test Configuration

**File**: `production_tracking_fully_fused_timedep.py`

**Current settings** (line 61-83):
```python
PARTICLE_GRID_RESOLUTION = (20, 80, 30)  # 48,000 particles
N_STEPS = 2_500
N_HOPS = 3
L2_SEARCH_RADIUS = 10
ENABLE_L1_SEARCH = True  # ← L1 is ENABLED
```

### Run Test 1A: L1 Enabled with Fix

```bash
cd /home/arhashemi/Workspace/welding/JAXTrace
source .venv/bin/activate

# Run with current settings (L1 enabled with fix)
python production_tracking_fully_fused_timedep.py 2>&1 | tee logs/test_1a_l1_fix_enabled.log
```

### What to Check

1. **Trajectories**: Check VTK files in ParaView
   - Focus on refined region (X=30, Y=15, Z=0.3)
   - Particles should show rotating/helical motion (NOT linear)

2. **Retention**: Check log output
   - Should stay >80% throughout 2,500 steps
   - Compare to previous runs

3. **Performance**: Note throughput
   - Expected: Similar to before (~29K particles/s)

4. **L2 Fallback Rate**: Look for search statistics in log
   - L1 should fail sometimes and trigger L2
   - Check if L2 success rate is reasonable

### Expected Outcome

**If trajectories are now correct**: Bug fix worked! L1 is now properly falling back to L2.

**If trajectories still linear**: L1 itself may be insufficient (multi-hop bug or neighbor connectivity issue).

---

## Step 2: Test with L1 Disabled

### Purpose

Test if L0→L2 (bypassing L1 entirely) produces correct trajectories. This isolates whether the problem is with L1 logic or with the overall search hierarchy.

### Configuration Change

**File**: `production_tracking_fully_fused_timedep.py`

**Edit line 83**:
```python
# BEFORE:
ENABLE_L1_SEARCH = True

# AFTER:
ENABLE_L1_SEARCH = False  # ← DISABLE L1 for this test
```

### Run Test 1B: L1 Disabled

```bash
# After editing ENABLE_L1_SEARCH = False
python production_tracking_fully_fused_timedep.py 2>&1 | tee logs/test_1b_l1_disabled.log
```

### What to Check

1. **Trajectories**: Same VTK check
   - Should show rotating motion if L2 is working correctly

2. **Performance**: Will be slower
   - Expected: 20-30K particles/s (L0 misses go straight to L2)

3. **Retention**: Should still be >80%

### Expected Outcomes

**Scenario A - Trajectories correct with L1 disabled**:
- Confirms L2 is working correctly
- Confirms L1 was the problem (even with the fix)
- Suggests L1 multi-hop bug or neighbor connectivity issue

**Scenario B - Trajectories still linear with L1 disabled**:
- Problem is NOT with L1
- Problem is with L0 or L2 or point-in-tet tests
- Need to investigate L2 search radius or point-in-tet accuracy

**Scenario C - Trajectories correct with L1 disabled AND with L1 enabled (after fix)**:
- **SUCCESS!** Bug fix solved the problem
- Can proceed with optimizations

---

## Step 3: Fix L1 Multi-Hop Bug (If Needed)

### When to Apply This

- **If Test 1A fails** (trajectories still wrong with L1 enabled)
- **AND Test 1B succeeds** (trajectories correct with L1 disabled)

This confirms L1 is the problem and needs the multi-hop fix.

### What's Wrong

Current L1 implementation doesn't actually "hop" - it searches neighbors of `start_elem_id` repeatedly in each iteration, never advancing to neighbors-of-neighbors.

### The Fix

**File**: `jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py`

Replace `search_l1_single` function (lines 87-128) with the corrected version below.

### Corrected Multi-Hop Logic

```python
def search_l1_single(pos: jax.Array, start_elem_id: jax.Array) -> jax.Array:
    """L1: Multi-hop neighbor search with proper hopping.

    Fixed: Now actually advances through neighbors (hop to neighbors-of-neighbors).
    Each hop searches neighbors of the last visited element, not the original start_elem_id.
    """
    current_elem = start_elem_id
    found = False  # Force neighbor search (L0 already verified non-containment)

    for hop in range(n_hops):
        # Only search if not found yet AND current elem is valid
        should_search = (~found) & (current_elem >= 0)

        # Get neighbors of current element
        neighbors = element_neighbors[jnp.where(should_search, current_elem, 0)]

        # Check all neighbors for containment
        def check_neighbor(elem_id):
            valid = elem_id >= 0
            elem_nodes_idx = connectivity[jnp.where(valid, elem_id, 0)]
            elem_nodes = node_positions[elem_nodes_idx]
            inside = point_in_tet_jax(pos, elem_nodes, tolerance=1e-10)
            return jnp.where(valid & inside, elem_id, jnp.int32(-1))

        found_in_neighbors = jax.vmap(check_neighbor)(neighbors)
        found_mask = found_in_neighbors >= 0

        # Get first containing neighbor (if any)
        found_containing = jnp.where(
            jnp.any(found_mask),
            found_in_neighbors[jnp.argmax(found_mask)],
            jnp.int32(-1)
        )

        # KEY FIX: Get first valid neighbor (even if point not inside) for next hop
        # This allows advancing through the neighbor graph
        first_valid_neighbor = jnp.where(
            jnp.any(neighbors >= 0),  # Any valid neighbor exists
            neighbors[jnp.argmax(neighbors >= 0)],  # Pick first valid neighbor
            current_elem  # Stay at current if no valid neighbors
        )

        # Update for next hop
        # If found containing element: use it and stop hopping (found=True)
        # If not found: advance to first_valid_neighbor for next hop
        current_elem = jnp.where(
            should_search,
            jnp.where(found_containing >= 0, found_containing, first_valid_neighbor),
            current_elem
        )
        found = found | (found_containing >= 0)

    # Return -1 if search failed
    return jnp.where(found, current_elem, jnp.int32(-1))
```

### Run Test 2: Multi-Hop Fix

```bash
# After applying the multi-hop fix above
python production_tracking_fully_fused_timedep.py 2>&1 | tee logs/test_2_multihop_fix.log
```

### Expected Outcome

- Trajectories should now be correct with L1 enabled
- L1 hit rate should improve (more particles found in L1)
- L2 fallback rate should decrease

---

## Step 4: Fix Point-in-Tet Degeneracy Threshold

### When to Apply This

- **If trajectories are still slightly wrong** even with L1 fixes
- **If you see particles "sticking" at boundaries** between elements
- **If you suspect point-in-tet tests are rejecting valid elements**

### What's Wrong

Current degeneracy threshold in `point_in_tet_gpu` is absolute (`det < 1e-17`), which may be too strict for small refined elements.

For refined meshes with elements ~0.1mm:
- Edge length L ~ 0.0001 m
- Expected determinant ~ L³ ~ 1e-12
- Current threshold 1e-17 is 5 orders of magnitude smaller!
- May incorrectly reject valid small elements

### The Fix

**File**: `jaxtrace/gpu/search/morton_global_search.py`
**Line**: ~414

**Current code**:
```python
is_degenerate = jnp.abs(det) < 1e-17  # Absolute threshold
```

**Replace with** (relative threshold):
```python
# Compute relative degeneracy threshold
# For refined elements with L~0.0001m: det~L³~1e-12
# Use relative threshold based on element size
det_abs = jnp.abs(det)
edge_length_sq = jnp.sum((v1)**2)  # Typical edge length²
expected_det = edge_length_sq ** 1.5  # det scales as L³
is_degenerate = det_abs < 1e-12 * jnp.maximum(expected_det, 1e-15)
```

### Run Test 3: Degeneracy Threshold Fix

```bash
# After applying the degeneracy threshold fix
python production_tracking_fully_fused_timedep.py 2>&1 | tee logs/test_3_degeneracy_fix.log
```

### Expected Outcome

- Slightly improved accuracy at element boundaries
- Fewer false rejections of valid small elements
- May improve retention rate slightly

---

## Summary of Fixes

| Priority | Fix | File | Line | Status |
|----------|-----|------|------|--------|
| 🔴 CRITICAL | L1 return bug | rk4_fully_fused_timedep.py | 126-128 | ✅ DONE |
| 🔴 CRITICAL | Test L1 disabled | production_tracking_fully_fused_timedep.py | 83 | ⏳ PENDING |
| 🟠 MODERATE | L1 multi-hop | rk4_fully_fused_timedep.py | 87-128 | ⏳ CONDITIONAL |
| 🟡 MINOR | Point-in-tet threshold | morton_global_search.py | 414 | ⏳ CONDITIONAL |

---

## Testing Workflow

```
Step 1: Fix L1 return bug (DONE)
  ↓
Test 1A: Run with L1 enabled
  ↓
  ├─ Trajectories correct? → SUCCESS! Done.
  └─ Trajectories wrong? → Continue to Test 1B
       ↓
Test 1B: Run with L1 disabled
  ↓
  ├─ Trajectories correct? → L1 is the problem
  │    ↓
  │   Step 3: Apply multi-hop fix
  │    ↓
  │   Test 2: Run with multi-hop fix
  │    ↓
  │   Trajectories correct? → SUCCESS!
  │
  └─ Trajectories still wrong? → Problem is in L0/L2
       ↓
      Step 4: Apply degeneracy threshold fix
       ↓
      Test 3: Run with threshold fix
       ↓
      Evaluate results
```

---

## Next Steps After Testing

### If All Tests Pass

**Proceed to optimizations**:
1. Hybrid neighbors (reduce memory, enable 200K particles)
2. Octree-aligned leaves (improve L2 performance)
3. Node-based Morton (best long-term solution)

### If Tests Fail

**Additional diagnostics**:
1. Add search statistics logging (L0/L1/L2 hit rates)
2. Visualize element assignments over time
3. Check velocity field scaling/units
4. Verify mesh quality (no degenerate elements)

---

**Ready for your testing!** Start with Test 1A and report results.
