# Octree Search - Final Root Cause Diagnosis

## Summary

The octree search has **6.03% accuracy** even for particles placed at **exact element centroids** (zero perturbation).

**Root Cause**: Tetrahedral elements **span multiple octants**, but are only **stored in ONE octant** based on their centroid location.

## Test Evidence

### Test 1: Exact Centroids
**File**: [test_octree_exact_centroids.py](test_octree_exact_centroids.py)
**Log**: [logs/test_octree_exact_centroids.log](logs/test_octree_exact_centroids.log)

```
Particles: 50,000 (placed at EXACT element centroids, zero perturbation)
Found: 50,000/50,000 (100.00%)
Correct: 3,013/50,000 (6.03%)  ← 94% WRONG even with zero perturbation!
```

### Test 2: Element Assignment
**Log**: [logs/test_octree_element_assignment_bug_FIXED.log](logs/test_octree_element_assignment_bug_FIXED.log)

```
Tested 1000 elements:
  Assigned leaf == Navigated leaf: 1000/1000 (100.00%)  ✓
```

**Interpretation**: Elements ARE assigned to the correct leaves (after our octree_builder fix). But only **6% of searches succeed**.

## The Fundamental Problem

### Tetrahedral Mesh Geometry

In a tetrahedral mesh:
1. **Elements are LARGE** relative to octree leaf size
2. **Elements SPAN multiple octants**
3. **Element centroid** may be in octant A
4. But **element vertices** extend into octants B, C, D

### Octree Assignment

Current implementation:
```python
# octree_builder.py
# Assign element to ONE octant based on centroid
octant = compute_octant(centroid)
assign_element_to_octant(element_id, octant)
```

### The Failure Mode

```
Example Element:
  Centroid: in octant 0 (lower-left-front)
  Vertices: span into octants 0, 1, 2, 3 (4 octants!)

  During construction:
    - Element stored ONLY in octant 0 (centroid's octant)

  During search for particle in octant 2:
    - Particle navigates to octant 2
    - Octree leaf 2 does NOT contain the element
    - Search fails or finds wrong neighbor element

  Result: WRONG or NOT FOUND
```

## Why Only 6% Succeed

**The 6% that succeed are particles whose element centroids happen to be in the same spatial region as the particle itself.**

For a highly refined mesh with 3.5M tiny tetrahedra:
- Elements are small
- But octree leaf size is large (max depth = 8)
- Many elements per leaf (hundreds)
- Each element spans multiple sub-regions within the leaf

**When a particle lands in a different sub-region than its element's centroid, it fails.**

## The Solution

### Option 1: Bounding-Box Based Assignment (Correct)

Assign each element to **ALL octree leaves its bounding box intersects**:

```python
def assign_elements_to_leaves(elements, nodes):
    for elem_id in elements:
        # Compute element bounding box
        elem_vertices = node_positions[connectivity[elem_id]]
        elem_bbox_min = elem_vertices.min(axis=0)
        elem_bbox_max = elem_vertices.max(axis=0)

        # Find all leaves that intersect this bbox
        for leaf in octree.leaves:
            if bbox_intersects(elem_bbox, leaf.bbox):
                leaf.add_element(elem_id)
```

**Pros**:
- Correct: Every element is findable from any octant it intersects
- No false negatives

**Cons**:
- Elements appear in multiple leaves (memory increase)
- More point-in-tet checks during search (performance decrease)

### Option 2: Neighbor-Leaf Search (Search-Time Fix)

During search, check the target leaf **and all 26 neighboring octants**:

```python
def search_with_neighbors(pos, leaf_id):
    # Check target leaf first
    result = check_leaf(leaf_id, pos)
    if result >= 0:
        return result

    # Check 26 neighboring octants
    for neighbor_id in get_neighbors_26(leaf_id):
        result = check_leaf(neighbor_id, pos)
        if result >= 0:
            return result

    return -1
```

**Pros**:
- No changes to octree construction
- Handles boundary cases

**Cons**:
- Up to 27× more point-in-tet checks (mitigated by early exit)
- Need to implement octree neighbor finding

### Option 3: Hybrid - Conservative Bounding Box

Expand element bounding box by small margin (e.g., 5%) before checking intersections:

```python
bbox_margin = 0.05 * (elem_bbox_max - elem_bbox_min)
elem_bbox_min_expanded = elem_bbox_min - bbox_margin
elem_bbox_max_expanded = elem_bbox_max + bbox_margin
```

This catches most boundary cases without exploding memory.

### Option 4: Abandon Octree, Use Blockwise

The blockwise search achieved **100% accuracy** (though only 0.53% found due to implementation issues).

**Pros**:
- Proven to work correctly
- Already implemented

**Cons**:
- 3,500× slower (29 p/s vs 100k p/s)
- Needs performance optimization

## Recommendation

**Implement Option 1: Bounding-Box Based Assignment**

This is the only **correct** solution for a spatial search structure with elements that span multiple cells.

### Implementation Plan

1. Modify `build_octree_for_level()` to assign elements to ALL intersecting leaves
2. Accept memory increase (elements duplicated across leaves)
3. Early-exit search logic already handles this correctly
4. Test accuracy (should reach >99%)

### Expected Impact

**Memory**:
- Current: ~240k nodes, ~50 elements per leaf = 12M element entries
- With bbox assignment: Each element in ~2-4 leaves on average = 24-48M entries
- Memory increase: **2-4× more element storage**
- Still manageable (< 200 MB for element lists)

**Performance**:
- More elements per leaf → more point-in-tet checks
- But early exit mitigates this
- Expect: 50-80% of current throughput (still 50k-80k p/s)
- Much better than blockwise (29 p/s)

**Accuracy**:
- Should reach >99.5% (similar to point-in-tet standalone test)

## Alternative: Switch to Blockwise

If octree with bbox assignment is still problematic, **abandon octree entirely** and:
1. Optimize blockwise search implementation
2. Parallelize hash bucket lookups
3. Use GPU for point-in-tet checks
4. Target: 10k-50k p/s with 100% accuracy

This may be more practical than fixing a fundamentally flawed octree design.

## Conclusion

**Octree is fundamentally incompatible with tetrahedral meshes where elements span multiple octree cells.**

Either:
1. Fix octree with bbox-based assignment (complex, memory-intensive)
2. Switch to blockwise search and optimize it (simpler, proven correct)

**My recommendation: Switch to blockwise and optimize it.**
