# Search Accuracy Analysis - Critical Issues Found

## Test Results Summary

### Octree Search:
- **Found**: 78.78% (39,392/50,000 particles)
- **Correct**: **0.02%** (12/50,000) ⚠️
- **Accuracy**: 12/39,392 = 0.03% of found particles are correct
- **Throughput**: 103,668 p/s

### Blockwise Search:
- **Found**: 0.53% (263/50,000 particles)
- **Correct**: 0.53% (all found particles correct) ✓
- **Accuracy**: 100% of found particles are correct
- **Throughput**: 29.5 p/s (extremely slow - 28 minutes!)

## Critical Findings

### Issue #1: Octree Returns Almost ALL Wrong Elements

**99.97% of found particles are assigned to wrong elements!**

This is NOT a tolerance/perturbation issue. Something is fundamentally broken in octree search.

### Issue #2: Blockwise Finds Almost Nothing

Only 0.53% of particles found, even though they're placed at element centroids + tiny perturbation.

### Issue #3: Blockwise Is 3,500× Slower

28 minutes for 50k particles = completely unusable for initialization.

## Root Cause Analysis

### Hypothesis 1: Perturbation Causes Element Change ❌ UNLIKELY

Perturbation: 3.125e-06 (0.003125 mm) = 1% of minimum element size

This is TINY. Even if particles moved to neighboring elements:
- Octree should still find SOMETHING nearby
- Blockwise should find >99% (not 0.53%)

**This is NOT the cause.**

### Hypothesis 2: Octree Element ID Mapping Bug ✓ LIKELY

Looking at [octree_search_gpu.py:165](jaxtrace/gpu/search/octree_search_gpu.py:165):

```python
return jnp.where(valid & inside, safe_id, -1)
```

The octree returns `safe_id` which is:
```python
safe_id = jnp.where(valid, elem_id, 0)
```

Where `elem_id` comes from `leaf_elements` which is populated from **filtered octree elements**.

**CRITICAL**: If octree was built with level-field filtering (`level_threshold=1.1`), the element IDs stored in octree are **FILTERED indices**, not original mesh element IDs!

From log line 47:
```
Filtered elements: 3,512,384/3,512,384
```

Wait - no filtering happened (all elements included). So this is NOT the issue.

### Hypothesis 3: Point-in-Tet Check Broken ✓ VERY LIKELY

Looking at [octree_search_gpu.py:160-163](jaxtrace/gpu/search/octree_search_gpu.py:160-163):

```python
# Get tet nodes
node_ids = connectivity[safe_id].astype(jnp.int32)
tet_nodes = node_positions[node_ids]  # ⚠️ POTENTIAL SHAPE ISSUE

# Check if inside
inside = point_in_tet_jax(pos, tet_nodes)
```

This is the **SAME PATTERN** that caused shape errors in rk4_scenario2.py!

When you do `tet_nodes = node_positions[node_ids]` with `node_ids` shape `(4,)`:
- Result should be `(4, 3)` for 4 nodes × 3 coordinates
- But JAX tracing can create unexpected shapes
- `point_in_tet_jax` expects `(4, 3)` but might receive wrong shape

**This could cause point-in-tet to always return False or always return True!**

### Hypothesis 4: Octree Traversal Bug ✓ POSSIBLE

The octree uses `max_depth=15` but actual tree depth is 8 (from log line 49).

If octree stops too early or navigates wrong:
- Lands in wrong leaf node
- Checks wrong elements
- Returns first match even if wrong

### Hypothesis 5: Blockwise Extremely Slow Due to CPU Processing

Blockwise took **1697 seconds** = 28 minutes for 50k particles.

From [initial_assignment.py](jaxtrace/gpu/search/initial_assignment.py), blockwise search is NOT fully GPU-accelerated - it's CPU-based with hash lookups.

This explains the extreme slowness but NOT the low find rate.

## Recommended Fixes

### Fix #1: CRITICAL - Debug Point-in-Tet

The point-in-tet check is likely broken. Need to:

1. **Test point-in-tet directly**:
   ```python
   # Take particle at centroid (no perturbation)
   elem_id = 100
   centroid = element_centroids[elem_id]
   nodes = connectivity[elem_id]
   tet_nodes = node_positions[nodes]

   # Should be TRUE
   result = point_in_tet_jax(centroid, tet_nodes)
   print(f"Centroid in element: {result}")  # Should be True
   ```

2. **Check tet_nodes shape**:
   ```python
   print(f"tet_nodes shape: {tet_nodes.shape}")  # Should be (4, 3)
   ```

3. **Fix indexing pattern** (if needed):
   ```python
   # WRONG (potential shape issue):
   node_ids = connectivity[safe_id].astype(jnp.int32)
   tet_nodes = node_positions[node_ids]

   # CORRECT (explicit individual indexing):
   node_ids = connectivity[safe_id].astype(jnp.int32)
   n0, n1, n2, n3 = node_ids[0], node_ids[1], node_ids[2], node_ids[3]
   p0 = node_positions[n0]
   p1 = node_positions[n1]
   p2 = node_positions[n2]
   p3 = node_positions[n3]
   tet_nodes = jnp.stack([p0, p1, p2, p3])  # Explicit (4, 3)
   ```

### Fix #2: Increase Perturbation Scale for Testing

Change from 1% to 10% of element size to ensure particles stay well inside:

```python
perturbation_scale = 0.10 * min_element_size  # 10% instead of 1%
```

This helps distinguish between:
- Tolerance issue (would be fixed by larger perturbation)
- Algorithm bug (still wrong even with larger perturbation)

### Fix #3: Test Without Perturbation First

Place particles EXACTLY at centroids (zero perturbation):

```python
particle_positions.append(centroid)  # No perturbation
```

**Expected**: 100% found, 100% correct for both methods.

If still wrong → algorithm is completely broken, not tolerance issue.

### Fix #4: Add Debug Logging to Octree

Add prints inside octree search to track:
- Which leaf node it lands in
- Which elements it checks
- Point-in-tet results for each element
- Why it returns the element it returns

## Immediate Next Steps

1. **Create simple point-in-tet test** (particles at exact centroids, no perturbation)
2. **Debug point_in_tet_jax** implementation
3. **Check if shape issue in octree_search_gpu.py:160**
4. **Compare with working Scenario #1 implementation**

## Why Blockwise Works (Sort Of)

Blockwise has **100% accuracy** on found particles, even though it only finds 0.53%.

This suggests blockwise point-in-tet is working correctly, but:
- It's checking very few elements (hence low find rate)
- It's extremely slow (CPU-based)
- It might be checking wrong blocks

## Conclusion

The octree search is **fundamentally broken** - 99.97% wrong is not a tolerance issue.

Most likely causes:
1. **Point-in-tet check always returns True** (accepts wrong elements)
2. **Point-in-tet check always returns False** (only 78% found by luck/bbox)
3. **Shape bug in tet_nodes** causes garbage comparison

Need to debug point-in-tet implementation FIRST before anything else.
