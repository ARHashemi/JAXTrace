# L1 Algorithm Fix Documentation

**Date**: 2025-12-19
**Status**: ✅ IMPLEMENTED
**File Modified**: `jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py`

---

## Executive Summary

Fixed critical bug in L1 neighbor search algorithm that prevented neighbor search from executing, causing particles to be incorrectly assigned to cached coarse elements instead of fine elements in refined regions.

**Root Cause**: Algorithm initialized `found = current_elem >= 0`, which is `True` for valid element IDs, causing immediate early exit before neighbor search.

**Fix**: Changed initialization to `found = False` to force neighbor search, and corrected loop logic to only set `found = True` when a containing neighbor is actually found.

---

## 1. Problem Analysis

### 1.1 Symptom

Particles in refined region showed:
- **0% fine element assignment** (should be 85%)
- **100% coarse element assignment** (should be 2%)
- **Linear trajectories** instead of rotating motion

### 1.2 Expected Behavior

In graded mesh with 1:2 octree refinement:
1. Particles move from coarse to fine elements as they enter refined region
2. L0 (cached element check) fails when particle leaves cached element
3. L1 (neighbor search) should find correct fine element via face-neighbors
4. If L1 fails, L2 (Morton global) provides fallback

### 1.3 Actual Behavior (Before Fix)

1. L0 fails (particle not in cached coarse element)
2. **L1 returns cached coarse element unchanged** ❌
3. Particle incorrectly stays in coarse element
4. Velocity interpolation uses wrong (coarse) element → incorrect trajectories

---

## 2. Root Cause Analysis

### 2.1 Search Hierarchy

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
            search_l1_single(pos, cached_elem_id)  # ← Called with cached_elem_id
        )
```

**Key Insight**: L1 is only called when L0 fails, meaning `cached_elem_id` does NOT contain the position.

### 2.2 The Bug (Lines 87-124)

**Original Code** (WRONG):
```python
def search_l1_single(pos, start_elem_id):
    current_elem = start_elem_id
    found = current_elem >= 0  # ← BUG: Checks validity, not containment!

    for _ in range(n_hops):
        if_found = found

        # Get neighbors (only if NOT found)
        neighbors = element_neighbors[jnp.where(~if_found & (current_elem >= 0), current_elem, 0)]
        #                                       ↑
        #                              Condition: ~if_found is FALSE!

        # ... search neighbors ...

        # Update
        current_elem = jnp.where(~if_found & (found_neighbor >= 0), found_neighbor, current_elem)
        found = (current_elem >= 0)  # ← Always True if current_elem is valid ID

    return current_elem  # Returns unchanged start_elem_id
```

**Execution Flow**:
1. `found = current_elem >= 0` → `True` (cached element has valid ID)
2. Loop iteration 1:
   - `if_found = True`
   - Condition `~if_found & (current_elem >= 0)` → `False`
   - Gets neighbors of element 0 (dummy)
   - No neighbor search executes
   - `current_elem` unchanged
   - `found = True` (still valid ID)
3. Loop iteration 2-3: Same as iteration 1 (no search)
4. Returns `start_elem_id` unchanged

**Result**: L1 never searches neighbors, returns cached coarse element even though particle is outside.

### 2.3 Why This Is Wrong

The logic conflates two different conditions:
- **Valid element ID**: `elem_id >= 0` (element exists in mesh)
- **Contains position**: `point_in_tet(pos, elem_id)` (position is inside element)

A valid element ID does NOT imply it contains the position. Since L0 already verified non-containment, `start_elem_id` is valid but does NOT contain the position.

---

## 3. The Fix

### 3.1 Corrected Initialization

```python
def search_l1_single(pos, start_elem_id):
    """L1: Multi-hop neighbor search (single particle).

    Note: L1 is only called when L0 fails, meaning start_elem_id does NOT
    contain the position. We start with found=False to force neighbor search.
    """
    current_elem = start_elem_id
    found = False  # ✓ Force neighbor search (L0 already verified non-containment)
```

**Rationale**: Since L0 failed, we know `start_elem_id` does NOT contain the position. Setting `found = False` forces the neighbor search to execute.

### 3.2 Corrected Loop Logic

```python
    for _ in range(n_hops):
        if_found = found

        # Get neighbors (NOW executes when found=False)
        neighbors = element_neighbors[jnp.where(~if_found & (current_elem >= 0), current_elem, 0)]

        # Search through neighbors
        # ... (unchanged) ...

        # Update current element if found
        current_elem = jnp.where(~if_found & (found_neighbor >= 0), found_neighbor, current_elem)
        found = if_found | (found_neighbor >= 0)  # ✓ Only True if we found containing neighbor
```

**Key Change**: `found = if_found | (found_neighbor >= 0)` instead of `found = (current_elem >= 0)`.

This ensures `found` is only set to `True` if:
- Already found in previous hop (`if_found = True`), OR
- Found containing neighbor in current hop (`found_neighbor >= 0`)

### 3.3 Corrected Execution Flow

**With Fix**:
1. `found = False` → Forces neighbor search
2. Loop iteration 1:
   - `if_found = False`
   - Condition `~if_found & (current_elem >= 0)` → `True`
   - Gets neighbors of `start_elem_id`
   - Searches 4 face-neighbors
   - If containing neighbor found: `current_elem` = neighbor, `found = True`, exit early ✓
   - If no containing neighbor: `current_elem` unchanged, `found = False`, continue to next hop ✓
3. Loop iteration 2:
   - `if_found = False` (no neighbor found yet)
   - Gets neighbors of current element (2-hop search)
   - Searches neighbors
   - ... (continues up to n_hops=3)
4. Returns:
   - Containing neighbor ID if found ✓
   - `start_elem_id` if no containing neighbor after 3 hops → Falls to L2 ✓

---

## 4. Expected Outcomes

### 4.1 Correct Element Assignment

In refined region, particles should be assigned to:
- **Fine elements (≤0.15mm)**: 70-85% (high density near tool)
- **Medium elements (0.15-0.30mm)**: 15-25%
- **Coarse elements (>0.30mm)**: 0-5%

### 4.2 Performance Scenarios

**Scenario A: Face-Neighbors Cross Refinement Levels**
- L1 succeeds frequently (finds fine neighbor in 1-2 hops)
- L2 rarely called (only for particles far from refined region)
- **Expected**: 50-100K particles/s (faster than L2-only baseline)

**Scenario B: Face-Neighbors Don't Cross Refinement Levels**
- L1 fails (no face-neighbor crosses coarse→fine boundary)
- L1 returns -1, falls to L2
- **Expected**: ~30K particles/s (same as L2-only baseline)
- **Conclusion**: L1 is ineffective, proceed to Phase 2 (octree-aligned leaves)

---

## 5. Testing

### 5.1 Test Script

Created `test_l1_fix.py` to validate fix:

**Test Cases**:
1. **L1 Enabled (with fix)**: Track 500 particles for 5 timesteps, measure:
   - Fine/Medium/Coarse element assignment rates
   - Performance (particle-steps/s)

2. **L1 Disabled (baseline)**: Same test with `enable_l1_search=False`
   - Compare element assignment rates
   - Compare performance

**Success Criteria**:
- ✓ L1 improves fine element assignment (or matches L2)
- ✓ L1 reduces coarse element assignment (or matches L2)
- ✓ L1 does not significantly degrade performance (<50% slower)

### 5.2 Running the Test

```bash
cd /home/arhashemi/Workspace/welding/JAXTrace
python test_l1_fix.py 2>&1 | tee logs/test_l1_fix.log
```

**Expected Output**:
- Initial assignment: Coarse elements dominate (bug visible)
- With L1 fix: Fine elements dominate after tracking
- Performance comparison: L1 vs L2-only

---

## 6. Next Steps

### 6.1 Immediate (This Test)

1. Run `test_l1_fix.py` to validate fix
2. Analyze results:
   - **If fine assignment improves**: L1 works! Face-neighbors cross levels.
   - **If no improvement**: L1 ineffective, face-neighbors don't cross levels.

### 6.2 Neighbor Connectivity Investigation

If L1 still doesn't improve fine assignment:

**Question**: Do face-based neighbors support 1:2 refinement?

**Test**: Diagnostic script to check:
```python
# For coarse element on boundary with fine region:
coarse_neighbors = element_neighbors[coarse_elem_id]  # 4 neighbors

# Check sizes of neighbors
for neighbor_id in coarse_neighbors:
    neighbor_size = compute_element_size(neighbor_id)
    print(f"Neighbor {neighbor_id}: size={neighbor_size:.4f} mm")

# Expected:
# - If face-neighbors work: Some neighbors are fine (≤0.15mm)
# - If face-neighbors don't work: All neighbors are coarse (>0.30mm)
```

### 6.3 Phase 2: Octree-Aligned Leaves (If L1 Ineffective)

If face-neighbors don't cross refinement levels, proceed to Phase 2:

1. Build leaves = octree cells at depth 7 (instead of fixed 256-element segments)
2. Implement 1:1 prefix→leaf mapping
3. Expected: 100-150K particles/s (vs 30K baseline)

---

## 7. Code Changes Summary

### 7.1 File Modified

**`jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py`** (lines 87-124)

### 7.2 Changes Made

**Line 94**: Changed initialization
```python
# Before:
found = current_elem >= 0

# After:
found = False  # Force neighbor search (L0 already verified non-containment)
```

**Line 124**: Changed found condition
```python
# Before:
found = (current_elem >= 0)

# After:
found = if_found | (found_neighbor >= 0)  # Only True if we found containing neighbor
```

**Lines 88-91**: Added docstring clarification
```python
"""L1: Multi-hop neighbor search (single particle).

Note: L1 is only called when L0 fails, meaning start_elem_id does NOT
contain the position. We start with found=False to force neighbor search.
"""
```

---

## 8. References

### 8.1 Related Documents

- `MORTON_OPTIMIZATION_GUIDE.md`: Section 2.3 (L1 algorithm analysis)
- `L1_SEARCH_DISABLE_GUIDE.md`: Configuration for disabling L1
- `GRADED_MESH_REFINEMENT_SOLUTION.md`: Mesh structure and refinement

### 8.2 Key Insights from Conversation

**User's Critical Challenge** (2025-12-19):
> "I'm wondering why L1 returns the previous coarse element, while the particle is NOT in coarse tet? It should return -1 and the early exit cannot be responsible here. Am I right?"

**Answer**: You are correct! The bug was NOT "early exit on first found medium element" (my original wrong explanation). The actual bug was that the algorithm never searched neighbors at all because `found = current_elem >= 0` was True from the start.

**User's Mesh Structure Clarification**:
> "My mesh doesn't contain any nested medium/fine element. The neighbors are based on the 3 node face sharing which cannot support 1:2 refinement between levels."

**Implication**: Face-based neighbors may not cross refinement levels. If test shows L1 still ineffective after fix, we need:
1. Node-based neighbors (20-100 neighbors, 1+ GB memory), OR
2. Skip topology-based search, go directly to Phase 2 (octree-aligned leaves)

---

## 9. Appendix: Algorithm Correctness Proof

### 9.1 Invariants

**Before fix**:
- ❌ `found = True` always (if `start_elem_id >= 0`)
- ❌ Neighbor search never executes
- ❌ Returns `start_elem_id` unchanged

**After fix**:
- ✓ `found = False` initially (force search)
- ✓ Neighbor search executes for each hop (until found)
- ✓ Returns containing neighbor if found, otherwise `start_elem_id`
- ✓ Falls to L2 if `start_elem_id` unchanged after search (no containing neighbor)

### 9.2 Correctness

**Case 1: Containing neighbor found in hop k**
- Iterations 1 to k-1: `found = False`, search continues
- Iteration k: `found_neighbor >= 0`, `current_elem` updated, `found = True`
- Iterations k+1 to n_hops: `if_found = True`, early exit (no unnecessary search)
- Returns: Containing neighbor ID ✓

**Case 2: No containing neighbor found**
- All iterations: `found = False`, search continues
- After n_hops: `current_elem = start_elem_id` (unchanged)
- Returns: `start_elem_id`
- L2 called: `elem_final = search_l2_single(pos)` ✓

**Case 3: start_elem_id invalid (< 0)**
- `found = False`
- Neighbor search condition: `~if_found & (current_elem >= 0)` → `False` (invalid ID)
- No search executes (correct, no neighbors to search)
- Returns: `start_elem_id` (< 0)
- L2 called ✓

### 9.3 Performance

**Time Complexity**:
- Best case: O(1) - containing neighbor found in first hop
- Worst case: O(n_hops × 4) = O(12) - searches 4 neighbors per hop for 3 hops
- Average case: O(6) - found in 1-2 hops

**Space Complexity**:
- O(1) - fixed arrays, no recursion

---

## 10. Conclusion

The L1 algorithm bug has been fixed. The test script will determine:
1. **If L1 works**: Face-neighbors cross refinement levels → Use L1 for 50-100K particles/s
2. **If L1 doesn't work**: Face-neighbors don't cross levels → Proceed to Phase 2 (octree-aligned leaves)

**Critical Test**: Run `test_l1_fix.py` to measure element assignment rates and performance.

**Next Decision Point**: Based on test results, either:
- **Option A**: L1 works → Optimize L1 (increase n_hops, tune parameters)
- **Option B**: L1 doesn't work → Phase 2 (octree-aligned leaves, LBVH)

---

**End of Document**
