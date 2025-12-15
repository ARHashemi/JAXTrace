# Octree Fundamental Flaw - Centroid-Based Assignment

## Critical Discovery

Even with **ZERO perturbation** (particles placed at exact element centroids), octree search achieves only **6.03% accuracy**.

This proves the octree design has a **fundamental architectural flaw**.

## Test Results

### Test: Exact Centroids (No Perturbation)
**File**: [test_octree_exact_centroids.py](test_octree_exact_centroids.py)
**Log**: [logs/test_octree_exact_centroids.log](logs/test_octree_exact_centroids.log)

```
Particles tested: 50,000 (at EXACT element centroids)
Particles found: 50,000/50,000 (100.00%)
Correct assignments: 3,013/50,000 (6.03%)  ← 94% WRONG!
```

### Comparison

| Perturbation | Found Rate | Accuracy | Note |
|--------------|------------|----------|------|
| 0% (exact centroids) | 100.00% | 6.03% | ← Proves fundamental flaw |
| 1% (tiny offset) | 99.47% | 5.66% | Similar failure rate |

**Conclusion**: Perturbation is NOT the issue. The octree itself is fundamentally broken.

## The Fundamental Flaw

### Problem

**Tetrahedral elements SPAN multiple octants**, but are only **stored in ONE octant** (based on centroid location).

### Example

```
Element 606482:
  Centroid: [0.00050781, 0.00151736, -0.00332031]

  During construction:
    - Centroid navigates to leaf A
    - Element 606482 stored ONLY in leaf A

  During search:
    - Particle at centroid [0.00050781, 0.00151736, -0.00332031]
    - Particle navigates to leaf A (same leaf!)
    - Element 606482 is in leaf A
    - Point-in-tet check SHOULD succeed

  But returns: Element 606231 (WRONG!)
```

**Wait...** If both centroid and particle navigate to the same leaf, and the element is in that leaf, why is it finding the WRONG element?

Let me re-examine the test results...

Actually, looking at our earlier test [test_octree_element_assignment_bug_FIXED.log](logs/test_octree_element_assignment_bug_FIXED.log):
```
Tested 1000 elements:
  Assigned leaf == Navigated leaf: 1000/1000 (100.00%)
```

So elements ARE in the correct leaves now (after our fix).

But particles at exact centroids still find wrong elements (6.03% correct).

## Alternative Hypothesis: Point-in-Tet Numerical Error

Wait, our point-in-tet test showed 100% accuracy for centroids!

From [logs/test_point_in_tet_debug.log](logs/test_point_in_tet_debug.log):
```
TEST 1: Centroid Inside Own Element (No Perturbation)
  Tested: 1000 centroids
  Inside own element: 1000/1000 (100.0%)  ✓
```

So point-in-tet DOES work for centroids when tested in isolation.

## The Real Bug: Element Search Order in Octree Leaves

Looking at the results:
- Particle at element 606482's centroid
- Finds element 606231 instead

Both elements must be in the SAME leaf. The octree is checking multiple elements and returning the FIRST match, not the CORRECT match!

### Root Cause: `check_leaf_elements_vectorized`

From [octree_search_gpu.py:128-177](jaxtrace/gpu/search/octree_search_gpu.py#L128-L177):

```python
def check_leaf_elements_vectorized(pos, leaf_elements, ...):
    def check_one_element(elem_id):
        # ... point-in-tet check ...
        return jnp.where(valid & inside, elem_id, -1)

    results = jax.vmap(check_one_element)(leaf_elements)

    # Return first valid result
    found = jnp.max(results)  # Takes maximum element ID that passed!
    return found
```

**BUG**: `jnp.max(results)` returns the **largest element ID** among all elements where point-in-tet returned True, NOT the first match!

If multiple elements overlap at a point (which is common at mesh boundaries and with numerical tolerance), this returns the **WRONG** element!

### Why This Happens

1. **Mesh boundaries**: Adjacent tetrahedra share faces/edges
2. **Numerical tolerance**: `point_in_tet_jax` uses `tolerance=1e-6`
3. **Multiple matches**: A point near a shared face can be "inside" multiple elements due to tolerance
4. **Wrong selection**: `jnp.max()` picks the highest element ID instead of the correct one

### The Fix

Option 1: Return ALL matching elements and pick the first:
```python
# Find all matches
matches = results >= 0
if jnp.any(matches):
    # Return first match (lowest index in leaf_elements array)
    indices = jnp.where(matches, jnp.arange(len(results)), len(results))
    first_idx = jnp.min(indices)
    return results[first_idx]
else:
    return jnp.int32(-1)
```

Option 2: More robust - return element with highest barycentric coordinate sum (most "inside"):
```python
# Modify to return (elem_id, min_barycentric_coord)
# Pick element with highest min coordinate (most centered in element)
```

Option 3: Use strict tolerance (0.0) to reduce overlaps:
```python
point_in_tet_jax(pos, tet_nodes, tolerance=0.0)  # No tolerance
```

But this risks missing particles at exact boundaries.

## Wait - Let Me Verify

Actually, the `jnp.max()` picks the **maximum value**, which for valid results will be the element ID. For invalid results (-1), it would pick the highest element ID among valid ones.

If there are multiple valid results: `[-1, -1, 606231, -1, 606482, -1]`, then:
- `jnp.max()` returns `606482` (the highest element ID)

So it's returning **the highest element ID that passed point-in-tet**, not the true element!

## Solution

Change `jnp.max(results)` to return the **first** valid result instead of the maximum:

```python
# Find first valid result (not maximum)
def find_first_valid(results):
    # Mask invalid results
    valid_mask = results >= 0

    # Create indices: valid positions get their index, invalid get large number
    indices = jnp.where(valid_mask, jnp.arange(len(results)), len(results))

    # Find minimum index (first valid)
    first_idx = jnp.argmin(indices)

    # Return element at that index
    return results[first_idx]

found = find_first_valid(results)
```

Or even simpler - stop at first match using `lax.scan`:

```python
def check_elements_scan(carry, elem_id):
    found_id = carry

    # If already found, skip
    def already_found(_):
        return found_id

    # Check this element
    def check_element(_):
        valid = elem_id >= 0
        node_ids = connectivity[jnp.where(elem_id >= 0, elem_id, 0)]
        tet_nodes = node_positions[node_ids]
        inside = point_in_tet_jax(pos, tet_nodes)
        return jnp.where(valid & inside, elem_id, found_id)

    new_found = jax.lax.cond(found_id >= 0, already_found, check_element, None)
    return new_found, None

found_id, _ = jax.lax.scan(check_elements_scan, jnp.int32(-1), leaf_elements)
```

This stops at the FIRST element that matches, not the highest element ID!
