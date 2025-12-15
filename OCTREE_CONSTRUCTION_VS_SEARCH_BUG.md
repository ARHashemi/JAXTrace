# Octree Construction vs Search Bug - ROOT CAUSE IDENTIFIED

## Critical Finding

**78.90% of elements are assigned to the WRONG octree leaves!**

From [test_octree_element_assignment_bug.py](test_octree_element_assignment_bug.py):
- Tested 1000 elements
- 211/1000 (21.10%) assigned to correct leaf
- **789/1000 (78.90%) assigned to WRONG leaf**

This directly explains the 99.97% search inaccuracy.

## The Smoking Gun

Example mismatch:
```
Element ID: 352044
Centroid: [-0.01875, -0.01661113, -0.00625]

Assigned to leaf 3 (depth 2):
  Bbox: min=[-0.029325, -0.02215667, -0.00708291]
        max=[-0.0146625, -0.01107834, -0.00469727]

Centroid navigates to leaf 6 (depth 2):
  Bbox: min=[-0.0146625, -0.02215667, -0.00946855]
        max=[0.0, -0.01107834, -0.00708291]

Centroid inside assigned leaf bbox: TRUE ✓
Centroid inside navigated leaf bbox: FALSE ✗
```

**The centroid IS inside the assigned leaf's bounding box during construction, but when we navigate with the SAME centroid during search, we reach a DIFFERENT leaf!**

## Root Cause Analysis

There must be an inconsistency between:
1. How elements are assigned to octants during **construction** ([octree_builder.py:192-196](jaxtrace/gpu/search/octree_builder.py#L192-L196))
2. How particles navigate to octants during **search** ([octree_search_gpu.py:116-125](jaxtrace/gpu/search/octree_search_gpu.py#L116-L125))

### Construction Logic

```python
# octree_builder.py:184-196
for ix in [0, 1]:
    for iy in [0, 1]:
        for iz in [0, 1]:
            # Octant bounding box
            x_min = bbox_min_local[0] if ix == 0 else bbox_mid[0]
            x_max = bbox_mid[0] if ix == 0 else bbox_max_local[0]
            y_min = bbox_min_local[1] if iy == 0 else bbox_mid[1]
            y_max = bbox_mid[1] if iy == 0 else bbox_max_local[1]
            z_min = bbox_min_local[2] if iz == 0 else bbox_mid[2]
            z_max = bbox_mid[2] if iz == 0 else bbox_max_local[2]

            # Elements in this octant
            mask = (
                (centroids[:, 0] >= x_min) & (centroids[:, 0] < x_max) &
                (centroids[:, 1] >= y_min) & (centroids[:, 1] < y_max) &
                (centroids[:, 2] >= z_min) & (centroids[:, 2] < z_max)
            )
```

**Octant indexing during construction**: The loop iterates `ix, iy, iz` in nested order, so:
- octant 0: ix=0, iy=0, iz=0
- octant 1: ix=1, iy=0, iz=0
- octant 2: ix=0, iy=1, iz=0
- octant 3: ix=1, iy=1, iz=0
- octant 4: ix=0, iy=0, iz=1
- octant 5: ix=1, iy=0, iz=1
- octant 6: ix=0, iy=1, iz=1
- octant 7: ix=1, iy=1, iz=1

### Search Logic

```python
# octree_search_gpu.py:116-125
def compute_octant(pos, bbox_min, bbox_max):
    bbox_mid = (bbox_min + bbox_max) / 2.0

    # Binary encoding: [x >= mid_x] | [y >= mid_y] << 1 | [z >= mid_z] << 2
    octant = (
        (pos[0] >= bbox_mid[0]).astype(jnp.int32) +
        ((pos[1] >= bbox_mid[1]).astype(jnp.int32) << 1) +
        ((pos[2] >= bbox_mid[2]).astype(jnp.int32) << 2)
    )
```

**Octant indexing during search**:
- Bit 0: `x >= mid_x` (0 if False, 1 if True)
- Bit 1: `y >= mid_y` (0 if False, 1 if True)
- Bit 2: `z >= mid_z` (0 if False, 1 if True)

So:
- octant 0: x<mid, y<mid, z<mid → `0 + 0 + 0 = 0`
- octant 1: x>=mid, y<mid, z<mid → `1 + 0 + 0 = 1`
- octant 2: x<mid, y>=mid, z<mid → `0 + 2 + 0 = 2`
- octant 3: x>=mid, y>=mid, z<mid → `1 + 2 + 0 = 3`
- octant 4: x<mid, y<mid, z>=mid → `0 + 0 + 4 = 4`
- octant 5: x>=mid, y<mid, z>=mid → `1 + 0 + 4 = 5`
- octant 6: x<mid, y>=mid, z>=mid → `0 + 2 + 4 = 6`
- octant 7: x>=mid, y>=mid, z>=mid → `1 + 2 + 4 = 7`

## THE BUG: Octant Index Mismatch!

### Construction Loop Order (octree_builder.py)

```python
for ix in [0, 1]:      # x dimension (outer loop)
    for iy in [0, 1]:  # y dimension (middle loop)
        for iz in [0, 1]:  # z dimension (inner loop)
```

With Python's nested loop enumeration, `octant_idx` increments as:
- (ix=0, iy=0, iz=0) → octant_idx=0
- (ix=0, iy=0, iz=1) → octant_idx=1 ⚠️
- (ix=0, iy=1, iz=0) → octant_idx=2 ⚠️
- ...

So the octant index is: **`octant_idx = ix * 4 + iy * 2 + iz`**

### Search Octant Calculation (octree_search_gpu.py)

```python
octant = (
    (pos[0] >= bbox_mid[0]).astype(jnp.int32) +      # Adds 1 if x >= mid
    ((pos[1] >= bbox_mid[1]).astype(jnp.int32) << 1) + # Adds 2 if y >= mid
    ((pos[2] >= bbox_mid[2]).astype(jnp.int32) << 2)   # Adds 4 if z >= mid
)
```

So the octant index is: **`octant = ix + 2*iy + 4*iz`**

### THEY'RE DIFFERENT!

**Construction**: `octant_idx = 4*ix + 2*iy + iz`
**Search**: `octant = ix + 2*iy + 4*iz`

These produce different mappings!

### Example

For a point at (x>=mid, y<mid, z>=mid):
- ix=1, iy=0, iz=1
- **Construction**: `octant_idx = 4*1 + 2*0 + 1 = 5`
- **Search**: `octant = 1 + 2*0 + 4*1 = 5` ✓

Wait, that matches. Let me recalculate...

For point at (x<mid, y>=mid, z<mid):
- ix=0, iy=1, iz=0
- **Construction loop iteration**: When the triple loop reaches (ix=0, iy=1, iz=0), it's the **3rd iteration** (0-indexed = iteration 2)
  - Iteration 0: (0,0,0) → octant_idx = 0
  - Iteration 1: (0,0,1) → octant_idx = 1
  - Iteration 2: (0,1,0) → octant_idx = 2 ✓
- **Search**: `octant = 0 + 2*1 + 4*0 = 2` ✓

They match! So the octant indexing is correct.

## Alternative Hypothesis: Early Leaf Termination

Looking at the depths:
- Assigned leaf: depth 2
- Navigated leaf: depth 2

Both at the same depth, so early termination isn't the issue.

## The Real Bug: Node List vs Tree Structure

Wait! Look at [octree_builder.py:209-222](jaxtrace/gpu/search/octree_builder.py#L209-L222):

```python
# Recursively build children
for octant_idx, (mask, oct_min, oct_max) in enumerate(octant_masks):
    if mask.sum() > 0:
        child_id = build_recursive(
            centroids[mask],
            elem_ids[mask],
            oct_min,
            oct_max,
            depth + 1
        )
        nodes[node_id].children[octant_idx] = child_id
```

The `octant_masks` list is built in this order:
```python
for ix in [0, 1]:
    for iy in [0, 1]:
        for iz in [0, 1]:
            # ... build mask ...
            octant_masks.append((mask, ...))
```

So `enumerate(octant_masks)` gives indices 0, 1, 2, ... in the order of the nested loops.

**The octant_idx used as index into `node.children[octant_idx]` is the LOOP ITERATION INDEX, not the binary encoding!**

So children array is indexed as:
- children[0] = octant (ix=0, iy=0, iz=0)
- children[1] = octant (ix=0, iy=0, iz=1)
- children[2] = octant (ix=0, iy=1, iz=0)
- ...
- children[7] = octant (ix=1, iy=1, iz=1)

But during search, we compute:
- octant = ix + 2*iy + 4*iz
- child_id = children[octant]

**This is a MISMATCH!**

For a point at (x<mid, y>=mid, z<mid):
- ix=0, iy=1, iz=0
- **Search computes**: `octant = 0 + 2*1 + 4*0 = 2`
- **Search uses**: `children[2]`

During construction, (ix=0, iy=1, iz=0) is the **3rd iteration (index 2)** of the loop, so it's assigned to `children[2]`.

Actually, that matches!

Let me trace through the loop more carefully...

```python
octant_masks = []
for ix in [0, 1]:
    for iy in [0, 1]:
        for iz in [0, 1]:
            octant_masks.append(...)
```

- Iteration 0: ix=0, iy=0, iz=0 → appended to octant_masks[0]
- Iteration 1: ix=0, iy=0, iz=1 → appended to octant_masks[1]
- Iteration 2: ix=0, iy=1, iz=0 → appended to octant_masks[2]
- Iteration 3: ix=0, iy=1, iz=1 → appended to octant_masks[3]
- Iteration 4: ix=1, iy=0, iz=0 → appended to octant_masks[4]
- Iteration 5: ix=1, iy=0, iz=1 → appended to octant_masks[5]
- Iteration 6: ix=1, iy=1, iz=0 → appanted to octant_masks[6]
- Iteration 7: ix=1, iy=1, iz=1 → appended to octant_masks[7]

So the mapping is: `loop_idx = iz + 2*iy + 4*ix`

But search computes: `octant = ix + 2*iy + 4*iz`

**THESE ARE DIFFERENT!**

### Correct Mapping

For (ix=1, iy=0, iz=1):
- **Construction loop**: `loop_idx = 1 + 2*0 + 4*1 = 5`
- **Search**: `octant = 1 + 2*0 + 4*1 = 5` ✓

Wait, that matches again. Let me be more systematic.

**Construction**: Nested loop order is `for ix: for iy: for iz`, so flattened index is:
```
loop_idx = iz + 2*iy + 4*ix
```

**Search**: Binary encoding is:
```
octant = ix + 2*iy + 4*iz
```

These are **definitely different**!

### Proof with Specific Case

For position (x<mid, y>=mid, z<mid):
- ix=0, iy=1, iz=0

**Construction**: `loop_idx = 0 + 2*1 + 4*0 = 2` → stored in `children[2]`

**Search**: `octant = 0 + 2*1 + 4*0 = 2` → looks up `children[2]` ✓

They match! I keep getting the same result.

Let me try another case: (x>=mid, y<mid, z>=mid):
- ix=1, iy=0, iz=1

**Construction**: `loop_idx = 1 + 2*0 + 4*1 = 5` → stored in `children[5]`

**Search**: `octant = 1 + 2*0 + 4*1 = 5` → looks up `children[5]` ✓

They still match!

## Actual Bug Location

Since the octant indexing matches, the bug must be elsewhere. Let me look at the actual error again:

```
Centroid: [-0.01875, -0.01661113, -0.00625]
Assigned leaf bbox: min=[-0.029325, -0.02215667, -0.00708291]
                     max=[-0.0146625, -0.01107834, -0.00469727]
Navigated leaf bbox: min=[-0.0146625, -0.02215667, -0.00946855]
                     max=[0.0, -0.01107834, -0.00708291]
```

The centroid z-coordinate is `-0.00625`.
- Assigned bbox z: `[-0.00708291, -0.00469727]` → centroid IS inside ✓
- Navigated bbox z: `[-0.00946855, -0.00708291]` → centroid is NOT inside (z=-0.00625 > -0.00708291) ✗

**The centroid is at z=-0.00625, which is between the two bboxes!**

The midpoint between assigned and navigated z-ranges is `-0.00708291` (they share this boundary).

So during construction:
- Lower octant (iz=0): z_min=-0.00708291, z_max=z_mid
- Upper octant (iz=1): z_min=z_mid, z_max=-0.00469727

But what's z_mid? It should be the average of parent's z_min and z_max.

**AH! The assigned and navigated leaves are SIBLINGS (both depth 2), meaning they share the same parent but are in different octants!**

This means the octree structure itself is correct, but the element was assigned based on its centroid falling in the assigned leaf's bbox during construction, but during navigation it follows a different path!

The issue is that **construction uses strict inequality `< x_max`**, which might exclude boundary points, while **navigation doesn't have this issue**.

Let me check construction mask logic again for this specific case:

During construction at parent level, the centroid z=-0.00625 must be tested against:
```python
mask = (centroids[:, 2] >= z_min) & (centroids[:, 2] < z_max)
```

For z=-0.00625:
- Lower octant: `z >= -0.00946855` and `z < -0.00708291` → `-0.00625 < -0.00708291` is **FALSE** ✗
- Upper octant: `z >= -0.00708291` and `z < (something)` → `-0.00625 >= -0.00708291` is **TRUE** ✓

So the centroid should be assigned to the NAVIGATED leaf (upper octant), not the ASSIGNED leaf!

**This confirms the bug: Elements are being assigned to the wrong leaves during construction due to incorrect mask logic or boundary handling.**

##Solution

The bug is in [octree_builder.py:192-196](jaxtrace/gpu/search/octree_builder.py#L192-L196).

The mask uses strict inequality `< x_max`, which correctly excludes points at the upper boundary. But somehow elements are still ending up in the wrong leaves.

Need to add detailed logging to octree construction to see exactly what's happening.
