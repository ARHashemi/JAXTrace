# Octree Construction Bug - FIXED

## Summary

Fixed the critical bug in octree construction that caused 78.90% of elements to be assigned to wrong leaves.

## The Bug

**File**: [jaxtrace/gpu/search/octree_builder.py](jaxtrace/gpu/search/octree_builder.py)
**Lines**: 175-214 (before fix)

**Problem**: During octree construction, elements were assigned to octants using bbox-based masks:

```python
# OLD (BROKEN) CODE:
mask = (
    (centroids[:, 0] >= x_min) & (centroids[:, 0] < x_max) &
    (centroids[:, 1] >= y_min) & (centroids[:, 1] < y_max) &
    (centroids[:, 2] >= z_min) & (centroids[:, 2] < z_max)
)
```

This didn't match the logic used during **search** ([octree_search_gpu.py:116-125](jaxtrace/gpu/search/octree_search_gpu.py#L116-L125)):

```python
# SEARCH LOGIC:
octant = (
    (pos[0] >= bbox_mid[0]).astype(jnp.int32) +
    ((pos[1] >= bbox_mid[1]).astype(jnp.int32) << 1) +
    ((pos[2] >= bbox_mid[2]).astype(jnp.int32) << 2)
)
```

**Result**: Elements assigned to octant X during construction, but particles navigated to octant Y during search.

## The Fix

**Applied**: 2025-12-09
**Commit**: (pending user commit)

Changed octree construction to use **identical octant assignment logic** as search:

```python
# NEW (FIXED) CODE in octree_builder.py:178-187:
# Vectorized octant computation for all centroids
# Binary encoding: ix + 2*iy + 4*iz
ix = (centroids[:, 0] >= bbox_mid[0]).astype(np.int32)
iy = (centroids[:, 1] >= bbox_mid[1]).astype(np.int32)
iz = (centroids[:, 2] >= bbox_mid[2]).astype(np.int32)
octant_assignments = ix + (iy << 1) + (iz << 2)

# Group elements by octant
for target_octant in range(8):
    mask = (octant_assignments == target_octant)
    # ... assign elements with this mask to child octant
```

**Key Insight**: Construction now uses `>= bbox_mid` check (same as search), ensuring particles navigate to the exact same leaves where their elements are stored.

## Validation

### Before Fix
**Test**: [test_octree_element_assignment_bug.py](test_octree_element_assignment_bug.py)
**Log**: [logs/test_octree_element_assignment_bug.log](logs/test_octree_element_assignment_bug.log)

```
Tested 1000 elements:
  Assigned leaf == Navigated leaf: 211/1000 (21.10%)
  Assigned leaf != Navigated leaf: 789/1000 (78.90%)  ✗ BUG
```

### After Fix
**Log**: [logs/test_octree_element_assignment_bug_FIXED.log](logs/test_octree_element_assignment_bug_FIXED.log)

```
Tested 1000 elements:
  Assigned leaf == Navigated leaf: 1000/1000 (100.00%)  ✓ FIXED
  Assigned leaf != Navigated leaf: 0/1000 (0.00%)
```

**Result**: ✓ **100% of elements now assigned to correct leaves!**

## Performance Impact

The fix uses vectorized numpy operations instead of Python loops, so octree construction should be **faster** than the broken version.

**Before (broken)**: Bbox mask comparisons in Python loops
**After (fixed)**: Vectorized numpy comparisons + single mask per octant

## Expected Search Accuracy Improvement

### Before Fix
From [logs/test_octree_vs_blockwise.log](logs/test_octree_vs_blockwise.log):
```
OCTREE RESULTS:
  Found: 39392/50000 (78.78%)
  Correct: 12/50000 (0.02%)  ← 99.97% WRONG!
```

### After Fix (Expected)
```
OCTREE RESULTS:
  Found: ~50000/50000 (100%)
  Correct: ~49900/50000 (99.8%+)  ← Should match blockwise accuracy
```

**Prediction**: Octree accuracy should jump from 0.02% to >99%, matching or exceeding blockwise search.

## Testing

### Test Running
**Command**: `python test_octree_vs_blockwise_initialization.py`
**Log**: [logs/test_octree_vs_blockwise_FIXED.log](logs/test_octree_vs_blockwise_FIXED.log)
**Status**: Running (octree construction in progress)

**Note**: The test takes time because it's building an octree for 3.5M elements. Estimated time: 10-15 minutes total.

### What the Test Will Show

1. **Octree accuracy**: Should be >99% (up from 0.02%)
2. **Octree throughput**: Should be 100k-400k p/s (same as before - bug didn't affect performance)
3. **Comparison to blockwise**: Should be 2-5× faster with similar accuracy

## Impact on Scenario #2

Once the test confirms >99% accuracy:

1. **Octree is VINDICATED**: It's not the bottleneck
2. **Scenario #2 slowness**: Must be in RK4 time-stepping loop, not L2 search
3. **Next debugging step**: Profile RK4 loop for GPU syncs, export overhead, etc.

## Files Modified

### Source Code
- [jaxtrace/gpu/search/octree_builder.py](jaxtrace/gpu/search/octree_builder.py) - Lines 175-214

### Diagnostic Tests
- [test_point_in_tet_debug.py](test_point_in_tet_debug.py) - Validates point-in-tet (100% ✓)
- [test_octree_centroid_hypothesis.py](test_octree_centroid_hypothesis.py) - Found 99.1% centroid mismatch
- [test_octree_element_assignment_bug.py](test_octree_element_assignment_bug.py) - Identified 78.9% assignment bug

### Documentation
- [OCTREE_BUG_ROOT_CAUSE_FOUND.md](OCTREE_BUG_ROOT_CAUSE_FOUND.md) - Detailed root cause analysis
- [POINT_IN_TET_VALIDATED.md](POINT_IN_TET_VALIDATED.md) - Point-in-tet validation
- [OCTREE_FIX_APPLIED.md](OCTREE_FIX_APPLIED.md) - This document

### Logs
- [logs/test_point_in_tet_debug.log](logs/test_point_in_tet_debug.log) - 100% pass
- [logs/test_octree_centroid_hypothesis.log](logs/test_octree_centroid_hypothesis.log) - 99.1% mismatch found
- [logs/test_octree_element_assignment_bug.log](logs/test_octree_element_assignment_bug.log) - 78.9% wrong (before fix)
- [logs/test_octree_element_assignment_bug_FIXED.log](logs/test_octree_element_assignment_bug_FIXED.log) - 100% correct (after fix) ✓
- [logs/test_octree_vs_blockwise_FIXED.log](logs/test_octree_vs_blockwise_FIXED.log) - Running...

## Commit Message Suggestion

```
Fix octree construction bug causing 99.97% search inaccuracy

Problem:
- Octree construction used bbox masks for octant assignment
- Search used >= midpoint checks for octant navigation
- Mismatch caused 78.9% of elements assigned to wrong leaves
- Result: 99.97% of particles found wrong elements

Solution:
- Changed construction to use identical logic as search
- Vectorized octant computation: ix + 2*iy + 4*iz
- Now 100% of elements assigned to correct leaves

Validation:
- test_octree_element_assignment_bug.py: 100% match (was 21.1%)
- Expected search accuracy: >99% (was 0.02%)

Files:
- jaxtrace/gpu/search/octree_builder.py (lines 175-214)
```

## Next Steps

1. **Wait for test completion** (~5-10 more minutes)
2. **Verify accuracy improvement** (expect >99%)
3. **If accuracy is high**: Octree is production-ready for L2 fallback
4. **Debug Scenario #2**: Focus on RK4 time-stepping, not search
5. **Optional**: Run production_tracking_scenario2.py with fixed octree
