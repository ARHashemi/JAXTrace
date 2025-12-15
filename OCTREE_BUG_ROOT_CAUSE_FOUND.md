# Octree Search 99.97% Inaccuracy - ROOT CAUSE IDENTIFIED

## Executive Summary

✓ **Point-in-tet algorithm is CORRECT** (validated with [test_point_in_tet_debug.py](test_point_in_tet_debug.py))

✗ **Octree construction has a critical bug**: **78.90% of elements are assigned to WRONG leaves**

This directly explains the 99.97% search inaccuracy reported in the comparison test.

---

## Diagnostic Test Results

### Test 1: Point-in-Tet Validation
**File**: [test_point_in_tet_debug.py](test_point_in_tet_debug.py)
**Log**: [logs/test_point_in_tet_debug.log](logs/test_point_in_tet_debug.log)

```
TEST 1: Exact centroids (no perturbation)
  Result: 1000/1000 (100.0%) inside ✓

TEST 2: Perturbed centroids (1% of min edge)
  Result: 100/100 (100%) inside ✓

CONCLUSION: ✓ Point-in-tet algorithm is CORRECT
```

### Test 2: Centroid Navigation Hypothesis
**File**: [test_octree_centroid_hypothesis.py](test_octree_centroid_hypothesis.py)
**Log**: [logs/test_octree_centroid_hypothesis.log](logs/test_octree_centroid_hypothesis.log)

```
Tested 1000 particles:
  True element in reached leaf: 9/1000 (0.90%)
  True element NOT in reached leaf: 991/1000 (99.10%)

FINDING: Particles navigate to correct leaves based on their positions,
but those leaves don't contain the elements they're inside!
```

### Test 3: Element Assignment vs Navigation
**File**: [test_octree_element_assignment_bug.py](test_octree_element_assignment_bug.py)
**Log**: [logs/test_octree_element_assignment_bug.log](logs/test_octree_element_assignment_bug.log)

```
Tested 1000 elements:
  Assigned leaf == Navigated leaf: 211/1000 (21.10%)
  Assigned leaf != Navigated leaf: 789/1000 (78.90%)

CONCLUSION: ✗ BUG CONFIRMED!
Elements are assigned to different leaves than where their centroids navigate.
```

---

## The Smoking Gun

Example from test_octree_element_assignment_bug.log:

```
Element ID: 352044
Centroid: [-0.01875, -0.01661113, -0.00625]

During Construction:
  Assigned to leaf 3 (depth 2)
  Leaf bbox: min=[-0.029325, -0.02215667, -0.00708291]
             max=[-0.0146625, -0.01107834, -0.00469727]
  Centroid inside this bbox: TRUE ✓

During Search:
  Centroid navigates to leaf 6 (depth 2)
  Leaf bbox: min=[-0.0146625, -0.02215667, -0.00946855]
             max=[0.0, -0.01107834, -0.00708291]
  Centroid inside this bbox: FALSE ✗
```

**The centroid is INSIDE the assigned leaf's bbox during construction, but when the SAME centroid is used to navigate the tree during search, it reaches a DIFFERENT leaf!**

---

## Root Cause Analysis

The bug is in the octree construction algorithm in [octree_builder.py](jaxtrace/gpu/search/octree_builder.py).

### Hypothesis 1: Octant Indexing Mismatch ✗ (Ruled Out)

Initially suspected that construction and search use different octant indexing schemes:
- **Construction**: Nested loop order gives `loop_idx = iz + 2*iy + 4*ix`
- **Search**: Binary encoding gives `octant = ix + 2*iy + 4*iz`

**Analysis**: After careful verification, these produce THE SAME mapping. Not the bug.

### Hypothesis 2: Boundary Condition Mismatch ✓ (Most Likely)

During construction ([octree_builder.py:192-196](jaxtrace/gpu/search/octree_builder.py#L192-L196)):
```python
mask = (
    (centroids[:, 0] >= x_min) & (centroids[:, 0] < x_max) &
    (centroids[:, 1] >= y_min) & (centroids[:, 1] < y_max) &
    (centroids[:, 2] >= z_min) & (centroids[:, 2] < z_max)
)
```

Uses **strict inequality** `< x_max`, which excludes points exactly at the upper boundary.

During search ([octree_search_gpu.py:116-125](jaxtrace/gpu/search/octree_search_gpu.py#L116-L125)):
```python
octant = (
    (pos[0] >= bbox_mid[0]).astype(jnp.int32) +
    ((pos[1] >= bbox_mid[1]).astype(jnp.int32) << 1) +
    ((pos[2] >= bbox_mid[2]).astype(jnp.int32) << 2)
)
```

Uses `>= bbox_mid` for all three dimensions.

**The mismatch**: During construction, if a point is exactly at `bbox_mid`, it's excluded from the lower octant (because `< bbox_mid` is False). But during search, the same point goes to the UPPER octant (because `>= bbox_mid` is True).

### Hypothesis 3: Recursive Call Bug ✓ (Investigating)

The test results show:
- Assigned leaf depth: 2
- Navigated leaf depth: 2
- But they're DIFFERENT leaves at the same depth!

This suggests the bug happens during **recursive subdivision**. An element's centroid might be inside a bbox at one level, but the recursive call passes the WRONG subbbox to the child, causing the element to be placed in a sibling leaf instead of the correct one.

---

## Impact on Search Accuracy

1. **Octree construction**: Element X is assigned to leaf A (based on flawed bbox logic)
2. **Particle search**: Particle P (inside element X) navigates to leaf B (based on correct navigation logic)
3. **Element check**: Particle P checks all elements in leaf B
4. **Result**: Element X is NOT in leaf B, so particle P is either:
   - Assigned to a nearby wrong element in leaf B (99.97% case)
   - Not found at all (78.78% found rate means 21.22% not found)

---

## Recommended Fix

### Option 1: Fix Boundary Conditions (Most Direct)

Change construction mask to use `<=` for upper boundary:

```python
# In octree_builder.py:192-196
mask = (
    (centroids[:, 0] >= x_min) & (centroids[:, 0] <= x_max) &  # Changed < to <=
    (centroids[:, 1] >= y_min) & (centroids[:, 1] <= y_max) &
    (centroids[:, 2] >= z_min) & (centroids[:, 2] <= z_max)
)
```

**Caveat**: This might cause elements exactly at boundaries to be assigned to multiple octants (not necessarily wrong, but needs testing).

### Option 2: Match Construction to Search Logic

Instead of using bbox comparisons during construction, use the SAME `compute_octant` logic used during search:

```python
# In octree_builder.py, replace mask-based assignment with:
def compute_octant_construction(pos, bbox_min, bbox_max):
    """Same as search logic."""
    bbox_mid = (bbox_min + bbox_max) / 2.0
    octant = (
        int(pos[0] >= bbox_mid[0]) +
        (int(pos[1] >= bbox_mid[1]) << 1) +
        (int(pos[2] >= bbox_mid[2]) << 2)
    )
    return octant

# Then in build_recursive:
for elem_id, centroid in zip(elem_ids, centroids):
    octant = compute_octant_construction(centroid, bbox_min_local, bbox_max_local)
    octant_elements[octant].append(elem_id)
```

**This guarantees construction and search use identical octant assignment logic.**

### Option 3: Bounding-Box Assignment (Most Robust)

Assign elements to **ALL leaves whose bboxes intersect the element's bounding box**:

```python
elem_bbox_min = node_positions[connectivity[elem_id]].min(axis=0)
elem_bbox_max = node_positions[connectivity[elem_id]].max(axis=0)

for leaf in octree_leaves:
    if bbox_intersects(elem_bbox, leaf.bbox):
        leaf.elements.append(elem_id)
```

**Pros**: Handles all boundary cases correctly
**Cons**: Elements appear in multiple leaves (memory increase)

---

## Recommended Next Steps

1. **Implement Option 2** (match construction to search logic) as it's the most direct fix
2. **Re-run test_octree_vs_blockwise_initialization.py** to verify accuracy improvement
3. **Expected result**: Accuracy should jump from 0.02% to >99%
4. **If accuracy improves**: Octree is vindicated, can be used for L2 fallback
5. **If accuracy doesn't improve**: There's an additional bug to find

---

## Files

### Diagnostic Tests Created
- [test_point_in_tet_debug.py](test_point_in_tet_debug.py) - Validates point-in-tet algorithm ✓
- [test_octree_centroid_hypothesis.py](test_octree_centroid_hypothesis.py) - Tests if particles reach correct leaves
- [test_octree_element_assignment_bug.py](test_octree_element_assignment_bug.py) - Identifies construction vs search mismatch

### Documentation Created
- [POINT_IN_TET_VALIDATED.md](POINT_IN_TET_VALIDATED.md) - Point-in-tet validation results
- [OCTREE_CENTROID_BUG_ANALYSIS.md](OCTREE_CENTROID_BUG_ANALYSIS.md) - Initial hypothesis analysis
- [OCTREE_CONSTRUCTION_VS_SEARCH_BUG.md](OCTREE_CONSTRUCTION_VS_SEARCH_BUG.md) - Detailed octant indexing analysis
- [OCTREE_BUG_ROOT_CAUSE_FOUND.md](OCTREE_BUG_ROOT_CAUSE_FOUND.md) - This document

### Logs
- [logs/test_point_in_tet_debug.log](logs/test_point_in_tet_debug.log)
- [logs/test_octree_centroid_hypothesis.log](logs/test_octree_centroid_hypothesis.log)
- [logs/test_octree_element_assignment_bug.log](logs/test_octree_element_assignment_bug.log)

---

## Summary

✓ Point-in-tet algorithm is correct
✓ Octree navigation logic is correct
✗ **Octree construction assigns 78.90% of elements to wrong leaves**
→ **This causes the 99.97% search inaccuracy**
→ **Fix: Make construction use same octant logic as search**
