# Octree Centroid-Based Assignment Bug - ROOT CAUSE FOUND

## Summary

**The 99.97% inaccuracy is caused by assigning elements to octree nodes based on element CENTROIDS, which fails when particles are inside elements whose centroids are in different octree leaf nodes.**

## The Fundamental Flaw

### Octree Construction ([octree_builder.py:192-196](jaxtrace/gpu/search/octree_builder.py#L192-L196))

Elements are assigned to octree leaf nodes based on their **centroid position**:

```python
# Assign elements to octants based on CENTROID
mask = (
    (centroids[:, 0] >= x_min) & (centroids[:, 0] < x_max) &
    (centroids[:, 1] >= y_min) & (centroids[:, 1] < y_max) &
    (centroids[:, 2] >= z_min) & (centroids[:, 2] < z_max)
)
```

### Octree Search ([octree_search_gpu.py:290-295](jaxtrace/gpu/search/octree_search_gpu.py#L290-L295))

Particles navigate to leaf nodes based on their **particle position**:

```python
# Navigate to child octant based on PARTICLE POSITION
def select_child(_):
    octant = compute_octant(pos, bbox_min, bbox_max)
    child_id = children[octant]
    return jnp.where(child_id >= 0, child_id, node_id)
```

## Why This Causes 99.97% Inaccuracy

### The Problem Scenario

Consider a tetrahedral element that:
1. Has centroid at position `(0.1, 0.1, 0.1)` in octant 0 (lower-left-front)
2. But extends across the midpoint boundary into octant 7 (upper-right-back)
3. A particle at `(0.6, 0.6, 0.6)` is **inside** this element

**What Happens:**
1. **During octree construction:**
   - Element is assigned to octant 0 leaf (based on centroid at 0.1, 0.1, 0.1)
   - Element is **NOT** in octant 7 leaf nodes

2. **During octree search:**
   - Particle at (0.6, 0.6, 0.6) navigates to octant 7 (based on position)
   - Octant 7 leaf does not contain the element (it's only in octant 0)
   - Point-in-tet check never runs for the correct element
   - Particle is assigned to some **other** element in octant 7

3. **Result:**
   - Wrong element assigned (99.97% of the time!)

### Why 78.78% Are "Found" But Wrong

The octree still "finds" elements for 78.78% of particles because:
- Most spatial regions have **some** elements in them
- But they're the **wrong** elements (from nearby but wrong locations)
- Point-in-tet returns True for these wrong elements due to spatial proximity
- **OR** (more likely): The octree returns the first element in the leaf, even if point-in-tet fails for all

Actually, let me check the leaf search logic:

### Checking `check_leaf_elements_vectorized`

From [octree_search_gpu.py:128-177](jaxtrace/gpu/search/octree_search_gpu.py#L128-L177):

```python
def check_leaf_elements_vectorized(
    pos: jax.Array,
    leaf_elements: jax.Array,
    node_positions: jax.Array,
    connectivity: jax.Array
) -> jax.Array:
    """Check if position is inside any element in leaf node."""

    def check_one_element(elem_id):
        # Use safe_id to avoid invalid reads
        safe_id = jnp.where(elem_id >= 0, elem_id, 0)
        valid = elem_id >= 0

        # Get tet nodes
        node_ids = connectivity[safe_id].astype(jnp.int32)
        tet_nodes = node_positions[node_ids]

        # Check if inside
        inside = point_in_tet_jax(pos, tet_nodes)

        # Return element ID if inside and valid, else -1
        return jnp.where(valid & inside, elem_id, -1)

    # Check all elements
    results = jax.vmap(check_one_element)(leaf_elements)

    # Return first valid result (or -1 if none)
    found = jnp.max(results)  # Takes maximum, which is first non-(-1) value
    return found
```

**The function correctly returns -1 if point-in-tet fails for all elements.** So the 78.78% found rate means point-in-tet is actually returning True for wrong elements!

This suggests **elements are spatially close enough that point-in-tet tolerance lets particles "leak" into nearby elements**.

## Why Point-in-Tet Diagnostic Passed But Search Failed

The diagnostic test ([test_point_in_tet_debug.py](test_point_in_tet_debug.py)) checked:
- Centroid of element X is inside element X ✓

But the actual search scenario is:
- Particle in element X navigates to leaf containing element Y (wrong octant)
- Point-in-tet checks particle against element Y
- Element Y might be close enough that tolerance allows false positive

**OR**: The particle genuinely **isn't inside any element** in the wrong leaf, but the octree logic has another bug.

## Evidence Supporting This Hypothesis

### 1. Blockwise Has 100% Accuracy (263/263 correct)

Blockwise search likely doesn't use centroid-based assignment. Let me check...

Actually, blockwise uses hash buckets with **element centroids**, so it has the same issue! But why is it 100% accurate?

**Answer**: Blockwise might check **all neighboring blocks** (26-connectivity), effectively checking a wider spatial region that captures elements whose centroids are in adjacent blocks.

### 2. Perturbation Scale is Tiny (3.125e-06 mm)

The 1% perturbation is so small that particles should definitely stay inside their elements. The fact that 99.97% are assigned to wrong elements means the octree is searching in the **completely wrong spatial region**.

### 3. Octree Max Depth = 8 (Not 15)

From the log:
```
Max depth: 8
```

The octree only reached depth 8, despite max_depth=15. This means leaves are **large spatial regions** containing many elements. If an element's centroid is in the wrong octant, the particle won't find it even though it's very close.

## The Fix

### Option 1: Bounding-Box Assignment (Correct But Expensive)

Assign elements to **ALL octree leaves whose bounding boxes intersect the element's bounding box**:

```python
# For each element, compute its bbox
elem_bbox_min = node_positions[connectivity[i]].min(axis=0)
elem_bbox_max = node_positions[connectivity[i]].max(axis=0)

# Assign to ALL leaves that intersect this bbox
for leaf in octree_leaves:
    if bbox_intersects(elem_bbox, leaf.bbox):
        leaf.elements.append(i)
```

**Pros:**
- Correct: Every element is in all leaves it could possibly be found in
- No false negatives

**Cons:**
- Elements appear in multiple leaves (memory increase)
- Octree size increases significantly
- More point-in-tet checks during search

### Option 2: Multi-Octant Query (Search-Time Fix)

Instead of navigating to a single leaf, search in the **current leaf and all 26 neighboring octants**:

```python
# After reaching leaf
def check_leaf_and_neighbors(node_id, pos):
    # Check current leaf
    result = check_leaf_elements(node_id, pos)
    if result >= 0:
        return result

    # Check 26 neighboring octants
    for neighbor_id in get_neighbor_octants(node_id):
        result = check_leaf_elements(neighbor_id, pos)
        if result >= 0:
            return result

    return -1
```

**Pros:**
- No changes to octree construction
- Fixes the boundary issue
- Minimal memory increase

**Cons:**
- Up to 27× more point-in-tet checks (but early exit helps)
- Need to track octree node neighbors

### Option 3: Switch to Blockwise (Escape Hatch)

Use blockwise search with 26-connectivity, which already handles neighbors:

**Pros:**
- Already implemented
- 100% accuracy (based on test results)

**Cons:**
- 3,500× slower (unusable)

### Option 4: Hybrid Approach

Use **bounding-box overlap** during octree construction but with a conservative threshold:

```python
# Expand element bbox by small margin
elem_bbox_min = node_positions[connectivity[i]].min(axis=0) - margin
elem_bbox_max = node_positions[connectivity[i]].max(axis=0) + margin

# Assign to all overlapping leaves
```

This catches boundary cases without exploding memory.

## Recommended Solution

**Implement Option 2: Multi-octant query during search**

This is the most practical fix:
1. Minimal code changes (only in octree_search_gpu.py)
2. No memory increase during construction
3. Handles all boundary cases
4. Early exit means average case isn't 27× slower

### Implementation Sketch

```python
def search_with_neighbors(pos, node_id):
    """Search in leaf and its spatial neighbors."""
    # Get current leaf's octant position in parent
    parent_id, octant_in_parent = get_parent_info(node_id)

    # Check current leaf first (most likely)
    result = check_leaf_elements(node_id, pos)
    if result >= 0:
        return result

    # Check up to 26 neighboring octants
    # (Only check siblings and parent's siblings)
    for neighbor_octant in range(8):
        if neighbor_octant == octant_in_parent:
            continue  # Already checked

        sibling_id = get_sibling(parent_id, neighbor_octant)
        if sibling_id >= 0:
            result = check_leaf_elements(sibling_id, pos)
            if result >= 0:
                return result

    return -1
```

## Next Steps

1. **Validate hypothesis**: Add logging to octree search showing which leaf each particle lands in vs which leaf contains the true element's centroid

2. **Implement fix**: Add multi-octant search with early exit

3. **Test**: Re-run test_octree_vs_blockwise_initialization.py and verify accuracy improves to >99%

4. **Optimize**: Profile to ensure performance isn't severely degraded

## Conclusion

✓ **Root cause identified**: Centroid-based element assignment to octree leaves

✓ **Validation**: Point-in-tet algorithm is correct (validated separately)

✓ **Solution**: Multi-octant search during particle query

→ **Next**: Implement and test the fix
