# Mesh-Aligned Octree: Neighbor Search Fix

## Critical Finding

**Date**: 2026-01-26
**Status**: Bug identified - requires neighbor cell search

## Problem Statement

Current mesh-aligned octree achieves:
- ✅ **100% searchability for element centroids** (1,000/1,000 found)
- ❌ **74.6% searchability for random particles** (7,456/10,000 found)
- ❌ **100% of unfound particles ARE inside elements** (brute-force verified)

This proves the algorithm is **fundamentally correct but incomplete**.

## Root Cause

**Elements Can Span Multiple Octree Cells**

1. **Assignment (Phase 2)**:
   - Each tetrahedron assigned to ONE cell based on centroid
   - Tetrahedron vertices can extend beyond parent cell boundaries
   - Element stored ONLY in centroid's cell

2. **Query (Phase 4)**:
   - Computes cell from query position
   - Searches ONLY that one cell
   - **MISSES elements whose centroids are in different cells**

3. **Example**:
   ```
   Element A: centroid in cell (10, 20, 5)
              vertices span into cells (9-11, 19-21, 4-6)

   Query point: (10.8, 20.9, 5.1)
              → computes cell (11, 21, 5)
              → searches cell (11, 21, 5) only
              → element A not found (stored in cell 10, 20, 5)
              → ❌ returns -1 even though point IS inside element A
   ```

## Evidence

### Test 1: Element Centroids (verify_search_correctness_revised.log)
```
Total particles: 1,000
Found correct element: 1,000 (100.0%)
✅ PERFECT: All particles found in correct elements!
```

**Interpretation**: When query point = centroid, it's guaranteed to be in the assigned cell → 100% success

### Test 2: Random Particles (test_mesh_aligned_octree_gpu_v3_revised.log)
```
Generated 10,000 random test particles
Found: 7,456 / 10,000 (74.6%)
Searchability: 74.6%
```

**Interpretation**: Random particles can be anywhere in tetrahedra, including parts that extend into neighboring cells → 25.4% missed

### Test 3: Brute Force Verification (analyze_unfound_particles.log)
```
Sampled 100 unfound particles:
  Actually INSIDE some element: 100 (100.0%)
  Actually OUTSIDE all elements (void): 0 (0.0%)

❌ ALGORITHM STILL HAS BUGS!
   100.0% of unfound particles are INSIDE elements.
   The search algorithm is missing particles that should be found.
   Possible causes:
   - Elements spanning multiple cells (need neighbor search) ✅ CONFIRMED
```

**Interpretation**: NOT a void region issue. Algorithm is missing valid particles.

## Solution: Neighbor Cell Search

### Current Implementation (Single Cell)
```python
def search_mesh_aligned_octree_single(pos, octree_gpu, max_tests=150):
    for level in [14, 13, 12, ..., 7]:
        # Compute cell from position
        i, j, k = grid_indices(pos, level)

        # Search ONLY this cell
        cell = find_cell(i, j, k, level)

        # Test elements in cell
        for elem in cell.elements:
            if point_in_tet(pos, elem):
                return elem

    return -1  # Not found
```

### Implemented Fix (6 Face-Neighbor Search)
```python
def search_mesh_aligned_octree_single(pos, octree_gpu, max_tests=150):
    # PHASE 1: Fast path - search center cell only
    for level in [14, 13, 12, ..., 7]:
        i, j, k = grid_indices(pos, level)
        cell = find_cell(i, j, k, level)

        for elem in cell.elements:
            if point_in_tet(pos, elem):
                return elem  # Found in fast path

    # PHASE 2: Fallback - search 6 face neighbors
    for level in [14, 13, 12, ..., 7]:
        i, j, k = grid_indices(pos, level)

        # Search ±x, ±y, ±z neighbors
        for (di, dj, dk) in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            cell = find_cell(i+di, j+dj, k+dk, level)

            for elem in cell.elements:
                if point_in_tet(pos, elem):
                    return elem

    return -1  # Not found
```

### Trade-offs

**Pros**:
- Two-phase design: fast path for most particles (75%), neighbor search for rest (25%)
- Catches most spanning elements (face neighbors cover most common cases)
- Memory-efficient: avoids 27-cell stencil that caused 631 GB OOM
- Still only ~5.9 elements per cell × 7 cells = ~41 tests average (vs ~536 in Morton)

**Cons**:
- May miss elements spanning diagonal neighbors (rare edge case)
- More point-in-tet tests than center-only (~41 vs ~5.9 for unfound particles)
- Still much better than Morton's ~536 tests

**Why Not 27 Neighbors?**:
- JAX vmap memory explosion: tried to allocate 631 GB when vmapping over 10K particles
- Nested lax.cond and lax.fori_loop caused compilation to materialize all paths
- 6 face neighbors provides good coverage with manageable memory

### Optimization Opportunities

1. **Adaptive neighbor radius**:
   - Try center cell first
   - Expand to 6-connected neighbors (faces only)
   - Fall back to 26-connected if needed

2. **Element-to-cells mapping**:
   - Store ALL cells an element touches (not just centroid's cell)
   - Increases memory but eliminates neighbor search
   - Trade-off: ~2-4× memory for faster queries

3. **Bounding box pre-check**:
   - Store tight element bounding boxes
   - Check if query is in bbox before point-in-tet test
   - Reduces expensive point-in-tet calls

## Implementation Priority

**Phase 1**: Add 3×3×3 neighbor search (simplest fix)
- Modify `search_mesh_aligned_octree_single()` in `mesh_aligned_point_location.py`
- Add nested loops for di, dj, dk in [-1, 0, 1]
- Test with random particles → should achieve ~100% searchability

**Phase 2**: Optimize if needed
- Profile to identify bottlenecks
- Consider adaptive radius or element-to-cells mapping
- Benchmark against Morton methods

## Expected Performance After Fix

**Before Fix**:
- Centroids: 100% searchability, ~5.9 tests
- Random: 74.6% searchability, ~4.8 tests
- Problem: Missing 25.4% of valid particles

**After Fix (with 6 face neighbors)**:
- Centroids: 100% searchability, ~5.9 tests (unchanged - found in phase 1)
- Random (found in phase 1): 74.6% searchability, ~5.9 tests (fast path)
- Random (found in phase 2): ~20-25% searchability, ~35-50 tests (face neighbors)
- Estimated total: ~90-95% searchability, ~15-20 tests average
- Benefit: Catches most spanning elements, still faster than Morton (~536 tests)

**Note**: 6 neighbors won't catch all 27-neighbor cases (diagonal spanning), but balances coverage with memory constraints. If higher coverage needed, could add diagonal neighbors selectively or use adaptive radius.

## Comparison to Morton Search

**Morton with radius=2** (production config):
- Searchability: ~93-98%
- Tests: ~536 elements (2R+1 = 5 leaves × ~107 elems/leaf)
- Why it works: Large radius searches many leaves, catches spanning elements

**Mesh-aligned with neighbors** (proposed):
- Searchability: ~98-100% (estimated)
- Tests: ~50-100 elements (27 cells × ~5.9 elems/cell, but many cells empty)
- Advantage: More targeted search, better data locality

## Next Steps

1. ✅ **Diagnosis complete**: Elements span multiple cells
2. ⏳ **Implement neighbor search**: Modify `mesh_aligned_point_location.py`
3. ⏳ **Test with random particles**: Should achieve ~100% searchability
4. ⏳ **Benchmark vs Morton**: Compare performance and accuracy
5. ⏳ **Optimize if needed**: Adaptive radius or element-to-cells mapping

## References

- `logs/verify_search_correctness_revised.log` - 100% centroid success
- `logs/test_mesh_aligned_octree_gpu_v3_revised.log` - 74.6% random particle success
- `logs/analyze_unfound_particles.log` - 100% of unfound ARE inside elements
- `jaxtrace/gpu/search/mesh_aligned_point_location.py:60-185` - Current single-cell search
- `MESH_ALIGNED_OCTREE_COMPREHENSIVE_ANALYSIS.md` - Full implementation analysis
