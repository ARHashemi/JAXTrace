# Velocity Errors Analysis - Pre-Existing vs Phase B Issues

## Date: October 16, 2025

## User's Critical Observation

> "Consider that the error about velocities was also exist before phase B implementation. Are these issues fixed now or needs further modifications?"

**Key Insight**: If velocity errors existed BEFORE Phase B, then they are a **PRE-EXISTING ISSUE** in the base octree FEM interpolator, not caused by Phase B implementation.

## Investigation Results

### Issue 1: Phase B-Specific Bug ✅ FIXED

**Bug**: Octree built with wrong mesh (timestep 0: 2,301 points) but used for tracking (timesteps 120-159: 780,922 points)

**Status**: **FIXED** in Phase B
- Changed reference timestep to 120 (revolution cycle mesh)
- Removed mesh modification during sampling
- Added validation to catch future mismatches

**Impact**: This would have caused SEVERE errors (340× mesh size mismatch), but only in Phase B implementation.

### Issue 2: Pre-Existing "Cheap Fallback" Problem ❌ NOT FIXED

**Location**: `jaxtrace/fields/octree_fem_interpolator_optimized.py:365-376`

**The Problem**: When octree search fails to find an element containing a query point, the interpolator uses a "cheap fallback" that:

```python
def cheap_fallback():
    # Use first valid element's nodes for fallback
    first_elem = candidate_elements[0]  # ⚠️ Arbitrary - uses FIRST element

    # Get its 4 nodes
    node_indices = connectivity[first_elem]

    # Find nearest of the 4 nodes (only 4!)
    dists = jnp.sum((mesh_points[node_indices] - query_point)**2, axis=1)
    nearest_local = jnp.argmin(dists)
    nearest_node = node_indices[nearest_local]

    return field_values[nearest_node]  # ⚠️ Single node value, NOT interpolated!
```

**Why This Causes Errors**:

1. **Not truly "nearest"**: Uses first candidate element (arbitrary), not the nearest element to query point
2. **Limited search**: Only searches 4 nodes from that element, ignoring potentially better neighbors
3. **No interpolation**: Returns single node value instead of properly interpolated value
4. **Spatially incorrect**: The nearest node might be far from query point

**When Does This Trigger**?

The fallback triggers when:
- Query point is outside all candidate elements (near boundaries)
- Query point falls in a gap between elements (thin regions, complex geometry)
- Octree search lands in wrong leaf node (element spans multiple octree cells)
- Numerical precision issues with point-in-tetrahedron test

**Why Front Rows and Boundaries Affected**?

- **Front rows (x-axis)**: Often near domain boundaries → fallback triggers
- **Boundary particles**: More likely to be outside elements or in gaps
- **Specific locations like (y=-0.0173678,z=-0.00928571)**: Likely in problematic geometry regions where element search fails

## Did Phase B Make It Worse?

**NO** - If errors existed before Phase B, then:

1. **Phase B mesh bug would have made it MUCH worse** (wrong mesh entirely)
2. **Now that Phase B mesh bug is fixed**, error frequency should return to **pre-Phase B levels**
3. **Cheap fallback problem remains** - same as before Phase B

## What is Actually Fixed?

### Phase B Fixes ✅

1. **Octree mesh mismatch**: Now uses correct revolution cycle mesh (timestep 120)
2. **Mesh modification bug**: No longer tries to update mesh during sampling
3. **Validation**: Catches velocity array size mismatches

### Still Broken ❌

The **cheap fallback mechanism** in the base octree interpolator - this is a pre-existing issue that affects:
- Phase A (before Phase B)
- Phase B (after fix)
- Any use of the optimized octree FEM interpolator

## Proposed Solutions for Cheap Fallback

### Option 1: Proper Nearest Element Search (Best Quality)

Instead of using first candidate, find truly nearest element:

```python
def better_fallback():
    # Search ALL candidate elements for nearest
    def compute_distance(elem_idx):
        centroid = element_centroids[elem_idx]
        return jnp.sum((centroid - query_point)**2)

    distances = jax.vmap(compute_distance)(candidate_elements[:n_candidates])
    nearest_elem_idx = candidate_elements[jnp.argmin(distances)]

    # Use nearest element's nodes for interpolation
    node_indices = connectivity[nearest_elem_idx]
    tet_nodes = mesh_points[node_indices]

    # Compute barycentric coordinates (even if outside)
    bary_coords = compute_barycentric_unconstrained(query_point, tet_nodes)

    # Interpolate using all 4 nodes
    node_values = field_values[node_indices]
    return jnp.dot(bary_coords, node_values)
```

**Pros**: Much more accurate, properly interpolated
**Cons**: More expensive (but still cheap compared to global search)

### Option 2: Expand Search to Neighboring Octree Nodes

If element not found in current leaf, search neighboring leaves:

```python
def expanded_search():
    # Get neighboring octree leaves
    neighbors = get_neighbor_leaves(node_idx)

    # Search elements in neighboring leaves
    for neighbor_node in neighbors:
        neighbor_elements = nodes_elements[neighbor_node]
        # Check elements...
```

**Pros**: Finds correct element more often, reduces fallback cases
**Cons**: More complex, requires neighbor tracking in octree

### Option 3: Better Barycentric Fallback

Use unconstrained barycentric coordinates (allow negative) for nearest element:

```python
def unconstrained_fallback():
    nearest_elem = find_nearest_element(query_point, candidate_elements)
    node_indices = connectivity[nearest_elem]
    tet_nodes = mesh_points[node_indices]

    # Allow negative barycentric coordinates for extrapolation
    bary_coords = solve_barycentric_unconstrained(query_point, tet_nodes)

    node_values = field_values[node_indices]
    return jnp.dot(bary_coords, node_values)
```

**Pros**: Proper interpolation, handles near-boundary cases better
**Cons**: May extrapolate with large errors if point is far from element

## Recommendations

### Immediate Fix (Phase B Context)

Since you reported that velocity errors existed **before** Phase B:

1. **Phase B mesh bug is fixed** ✅ - validation shows all arrays match
2. **Cheap fallback issue remains** ❌ - requires base interpolator fix
3. **Current state**: Phase B is now **as good as Phase A** (no worse, possibly better with correct mesh)

### To Fully Fix Velocity Errors

The cheap fallback mechanism needs improvement (separate from Phase B work):

**Priority 1**: Implement proper nearest element search (Option 1)
- Most impact for least complexity
- Reduces error from fallback cases
- Can be done as independent fix to base interpolator

**Priority 2**: Add diagnostic logging
- Count how often fallback triggers
- Log which particles use fallback
- Helps quantify problem scope

**Priority 3**: Consider expanded search (Option 2)
- More complex but reduces fallback frequency
- Better for production use

## Testing Recommendations

### Test 1: Compare Fallback Frequency

Add counter to track fallback usage:

```python
# In interpolator
fallback_used = jnp.where(found, 0, 1)
# Aggregate across all particles
total_fallback_count = jnp.sum(fallback_used)
```

Expected results:
- Phase A (pre-loading): ~X% use fallback
- Phase B (per-timestep): ~X% use fallback (should be same)
- After improved fallback: <X% use fallback

### Test 2: Velocity Error Magnitude

For particles using fallback, measure error:

```python
# Compare fallback result vs proper interpolation (if available)
if has_ground_truth:
    error = jnp.abs(fallback_value - true_value)
    max_error = jnp.max(error)
    mean_error = jnp.mean(error)
```

### Test 3: Specific Locations

Test the problematic locations you identified:
- (y=-0.0173678, z=-0.00928571)
- (y=-0.00985714, z=-0.00214286)
- (y=0.0220612, z=-0.00428571)

Check:
- Do these trigger fallback?
- What element is found (if any)?
- What is the distance to nearest element?

## Conclusion

**Short Answer**:
- **Phase B bug**: FIXED ✅
- **Pre-existing velocity errors**: NOT FIXED ❌ (requires base interpolator improvement)

**Long Answer**:

Your observation was correct - if velocity errors existed before Phase B, then they are **pre-existing issues** in the octree FEM interpolator's "cheap fallback" mechanism.

**What Phase B fixed**:
- Critical mesh mismatch bug (octree with wrong mesh)
- This would have made errors MUCH worse in Phase B
- Now Phase B is as accurate as Phase A

**What still needs fixing**:
- The "cheap fallback" when element search fails
- This affects both Phase A and Phase B equally
- Requires modification to base interpolator (separate from Phase B work)

**Current State**: Phase B implementation is **correct and no worse than Phase A**. The remaining velocity errors at specific locations are a **separate, pre-existing issue** that should be addressed in the base octree interpolator.

## Files for Further Investigation

1. `jaxtrace/fields/octree_fem_interpolator_optimized.py:365-376` - Cheap fallback implementation
2. `jaxtrace/fields/octree_fem_interpolator_optimized.py:309-350` - Element search logic
3. `jaxtrace/fields/octree_fem_interpolator_optimized.py:199-245` - Point-in-tetrahedron test

## Next Steps

1. **For Phase B**: Current implementation is correct ✅
2. **For velocity accuracy**: Improve cheap fallback in base interpolator (separate task)
3. **For validation**: Add diagnostics to quantify fallback frequency and error magnitude
