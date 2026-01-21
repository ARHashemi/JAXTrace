# Phase 1: L1 Algorithm Fix - Implementation Summary

**Date**: 2025-12-19
**Status**: ✅ IMPLEMENTED, 🧪 TESTING IN PROGRESS

---

## Quick Summary

Fixed critical bug in L1 neighbor search algorithm that prevented the neighbor search from executing. The bug caused particles in refined regions to be incorrectly assigned to cached coarse elements instead of fine elements.

**The Fix**: Changed `found = current_elem >= 0` to `found = False` to force neighbor search execution.

---

## Changes Made

### 1. Code Fix

**File**: `jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py`

**Lines 87-94**: Changed initialization
```python
# BEFORE (WRONG):
def search_l1_single(pos, start_elem_id):
    current_elem = start_elem_id
    found = current_elem >= 0  # ← BUG: Always True for valid IDs

# AFTER (CORRECT):
def search_l1_single(pos, start_elem_id):
    """L1: Multi-hop neighbor search (single particle).

    Note: L1 is only called when L0 fails, meaning start_elem_id does NOT
    contain the position. We start with found=False to force neighbor search.
    """
    current_elem = start_elem_id
    found = False  # Force neighbor search (L0 already verified non-containment)
```

**Line 124**: Changed found condition
```python
# BEFORE (WRONG):
found = (current_elem >= 0)  # Always True if current_elem is valid

# AFTER (CORRECT):
found = if_found | (found_neighbor >= 0)  # Only True if we found containing neighbor
```

### 2. Test Script

**Created**: `test_l1_fix.py`

Tests L1 fix by comparing:
- **Test 1**: L1 enabled (with fix) - Should find fine elements in refined region
- **Test 2**: L1 disabled (baseline) - Uses only L2 global search

**Metrics Tracked**:
- Element assignment rates (Fine/Medium/Coarse percentages)
- Performance (particle-steps/s)
- Success criteria for validation

### 3. Documentation

**Created**: `L1_ALGORITHM_FIX.md`

Comprehensive documentation including:
- Problem analysis and symptom description
- Root cause analysis with execution flow
- Detailed explanation of the fix
- Expected outcomes and testing plan
- Next steps based on test results

---

## Understanding the Bug

### The Problem

L1 search returned the cached coarse element unchanged, even though the particle was outside that element.

### Why It Happened

**L0+L1 Search Flow**:
```python
def search_l0_l1_l2_single(pos, cached_elem_id):
    # L0: Check cached element
    elem_l0 = search_l0_single(pos, cached_elem_id)
    found_l0 = elem_l0 >= 0

    if enable_l1_search:
        # L1: Only if L0 failed
        elem_l1 = jnp.where(
            found_l0,
            elem_l0,
            search_l1_single(pos, cached_elem_id)  # ← Called when L0 fails
        )
```

**Key Insight**: L1 is only called when L0 fails, meaning `cached_elem_id` does NOT contain the position.

**The Bug in L1**:
```python
def search_l1_single(pos, start_elem_id):
    current_elem = start_elem_id
    found = current_elem >= 0  # ← BUG: Checks validity, not containment!

    for _ in range(n_hops):
        if_found = found

        # Get neighbors (only if NOT found)
        neighbors = element_neighbors[jnp.where(~if_found & (current_elem >= 0), current_elem, 0)]
        #                                       ↑
        #                              This is FALSE when found=True!
```

**Execution with Bug**:
1. `found = current_elem >= 0` → `True` (valid ID exists)
2. Loop iteration 1: `if_found = True`
3. Condition `~if_found & (current_elem >= 0)` → `False`
4. Gets neighbors of dummy element 0 (no actual search)
5. Returns `start_elem_id` unchanged

**Result**: Particle stays in cached coarse element → Wrong velocities → Linear trajectories

### The Fix

**Corrected Initialization**:
```python
found = False  # Force neighbor search
```

**Corrected Loop Logic**:
```python
found = if_found | (found_neighbor >= 0)  # Only True if containing neighbor found
```

**Execution with Fix**:
1. `found = False` → Forces search
2. Loop iteration 1: `if_found = False`
3. Condition `~if_found & (current_elem >= 0)` → `True`
4. Gets actual neighbors of `start_elem_id`
5. Searches 4 face-neighbors
6. Returns containing neighbor if found, otherwise -1 (falls to L2) ✓

---

## Expected Outcomes

### Scenario A: Face-Neighbors Cross Refinement Levels ✓

**If face-based neighbors connect coarse → fine elements**:
- L1 succeeds frequently (finds fine neighbor in 1-2 hops)
- Fine element assignment: 70-85% (correct)
- Performance: 50-100K particles/s (faster than L2-only)

**Conclusion**: L1 works! Use it for tracking.

### Scenario B: Face-Neighbors Don't Cross Levels ✗

**If face-based neighbors stay within same refinement level**:
- L1 fails (no face-neighbor crosses coarse→fine boundary)
- L1 returns -1, falls to L2
- Fine element assignment: Same as L2-only baseline
- Performance: ~30K particles/s (same as L2-only)

**Conclusion**: L1 ineffective, proceed to Phase 2 (octree-aligned leaves).

---

## Test Results

**Status**: 🧪 Test running in background

**Test Configuration**:
- Mesh: ThreadedA (3.5M elements)
- Particles: 500 in refined region (tool center)
- Steps: 5
- Time step: 5e-6 s
- L2 search radius: 100

**Expected Results**:
- Test 1 (L1 enabled): Fine assignment rate after 5 steps
- Test 2 (L1 disabled): Fine assignment rate after 5 steps
- Performance comparison: Throughput (particles-steps/s)

**Success Criteria**:
1. ✓ L1 improves fine element assignment (or matches L2)
2. ✓ L1 reduces coarse element assignment (or matches L2)
3. ✓ L1 does not significantly degrade performance (<50% slower)

---

## Next Steps (Based on Test Results)

### If L1 Works (Scenario A)

1. **Optimize L1 Parameters**:
   - Tune `n_hops` (currently 3, try 4-5)
   - Analyze hop distribution (most found in 1-2 hops?)

2. **Production Deployment**:
   - Enable L1 in production script
   - Re-run full 225K particle tracking
   - Verify correct trajectories

3. **Performance Benchmarking**:
   - Measure L1 hit rate vs L2 fallback rate
   - Compare against L2-only baseline
   - Expected: 50-100K particles/s

### If L1 Doesn't Work (Scenario B)

1. **Investigate Neighbor Connectivity**:
   - Diagnostic: Check if face-neighbors cross refinement levels
   - For coarse element near refined region, examine neighbors

2. **Options**:
   - **Option 2A**: Node-based neighbors (20-100 neighbors, 1+ GB memory)
   - **Option 2B**: Skip topology-based search → Proceed to Phase 2

3. **Phase 2: Octree-Aligned Leaves** (Recommended):
   - Build leaves = octree cells at depth 7
   - Replace fixed 256-element segments
   - Implement 1:1 prefix→leaf mapping
   - Expected: 100-150K particles/s

---

## User's Key Concerns Addressed

### 1. Why L1 Returns Coarse Element When Particle is Outside

**User's Question**:
> "I'm wondering why L1 returns the previous coarse element, while the particle is NOT in coarse tet? It should return -1 and the early exit cannot be responsible here. Am I right?"

**Answer**: You are absolutely correct! The bug was NOT "early exit on first found medium element" (my original incorrect explanation). The actual bug was:

```python
found = current_elem >= 0  # Checks validity, not containment
```

This prevented the neighbor search from executing at all. The algorithm never looked at neighbors, so it returned the cached element unchanged.

### 2. Early Exit is NOT the Issue

**Original Wrong Explanation**: "L1 exits early when it finds medium element"

**Correct Explanation**: L1 never searches neighbors because `found = True` from the start. The early exit logic (`if_found`) is correct - the issue is that `found` is set incorrectly at initialization.

### 3. L0 Success Should NOT Trigger L1

**User's Concern**:
> "Be careful: do we put this condition as an early exit to not perform L1 for the cases that L0 was successful?"

**Current Implementation** (CORRECT):
```python
elem_l1 = jnp.where(
    found_l0,      # If L0 succeeded
    elem_l0,       # Use L0 result
    search_l1_single(pos, cached_elem_id)  # Else call L1
)
```

L1 is only called when `found_l0 = False`, which is correct. Our fix does not change this behavior.

### 4. Neighbor Construction for 1:2 Refinement

**User's Concern**:
> "We should think how to correct the elements neighbor construction to support 1:2 refinements."

**Current Status**: Face-based neighbors (3-node sharing)

**Expected Behavior**:
- If face-neighbors work: Coarse and fine share tetrahedral faces → L1 succeeds
- If face-neighbors don't work: Only edge/vertex sharing → L1 fails, need node-based

**Test Result Will Tell Us**: If L1 still doesn't improve fine assignment after fix, face-neighbors don't cross refinement levels → Need node-based or skip to Phase 2.

---

## Technical Details

### Algorithm Correctness

**Invariants (After Fix)**:
- ✓ `found = False` initially (force search)
- ✓ Neighbor search executes for each hop (until found)
- ✓ Returns containing neighbor if found
- ✓ Returns `start_elem_id` if no containing neighbor (L2 called next)

**Time Complexity**:
- Best case: O(1) - containing neighbor found in first hop
- Worst case: O(n_hops × 4) = O(12) - searches 4 neighbors per hop for 3 hops
- Average case: O(6) - found in 1-2 hops

**Space Complexity**: O(1) - fixed arrays, no recursion

### JAX Compatibility

All operations are JAX-compatible:
- No dynamic control flow (fixed number of hops)
- Uses `jnp.where` for conditional updates
- Fully JIT-compilable

---

## Files Modified/Created

### Modified
1. **`jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py`**
   - Lines 87-94: Changed initialization with docstring
   - Line 124: Changed found condition

### Created
1. **`test_l1_fix.py`**
   - Comprehensive test comparing L1 enabled vs disabled
   - Tracks element assignment rates and performance
   - Success criteria validation

2. **`L1_ALGORITHM_FIX.md`**
   - Complete documentation of bug and fix
   - Root cause analysis
   - Expected outcomes and next steps

3. **`PHASE_1_L1_FIX_SUMMARY.md`** (this file)
   - Quick reference summary
   - Key points for user review

---

## Conclusion

The L1 algorithm bug has been fixed. The test will determine whether face-based neighbors support 1:2 refinement:

- **If YES**: L1 works → Use L1 for tracking (50-100K particles/s)
- **If NO**: L1 doesn't work → Proceed to Phase 2 (octree-aligned leaves, 100-150K particles/s)

Either way, we have a clear path forward based on the test results.

---

**Waiting for test results...**
