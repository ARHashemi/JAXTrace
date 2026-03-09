# Option B: Pre-Computed Neighbor Table - Results

**Date:** 2026-01-30
**Task:** Implement neighbor search using pre-computed lookup table
**Status:** ✅ IMPLEMENTED | ⚠️ PARTIAL SUCCESS

---

## Executive Summary

**Implemented** CPU-side neighbor table generation + GPU direct lookup to avoid JAX memory issues.

**Results:**
- ✅ **89.74% searchability** (vs 74.6% baseline) - **+15.1 percentage points**
- ⚠️ **1,504 particles/sec** throughput (vs 12,106 baseline) - **8× slower**
- ✅ **13.9 mean tests/particle** (vs 4.8 baseline) - reasonable overhead
- ✅ **No memory explosion** - stable execution

---

## Implementation

### Architecture

**CPU-Side (Preprocessing):**
1. For each cell, find its 26 spatial neighbors (3×3×3 grid - center)
2. Use (Morton code, level) lookup to find neighbor cell indices
3. Store in `cell_neighbors` array: `(n_cells, 26) int32`
4. -1 for neighbors that don't exist (boundaries)

**GPU-Side (Runtime):**
1. Find primary cell at each level (14-7)
2. Search primary cell elements
3. Search 26 pre-computed neighbors (direct array lookup - O(1))
4. Return first found element

### Files Created

**Core Implementation:**
- [jaxtrace/gpu/search/mesh_aligned_octree_with_neighbor_table.py](jaxtrace/gpu/search/mesh_aligned_octree_with_neighbor_table.py) - Neighbor table data structures and CPU builder
- [jaxtrace/gpu/search/mesh_aligned_search_with_neighbors.py](jaxtrace/gpu/search/mesh_aligned_search_with_neighbors.py) - GPU search kernel using pre-computed neighbors

**Test Scripts:**
- [test_precomputed_neighbors.py](test_precomputed_neighbors.py) - Full performance test
- [test_neighbor_debug.py](test_neighbor_debug.py) - Comparison with baseline
- [test_minimal_neighbor.py](test_minimal_neighbor.py) - Single-particle debug test

**Documentation:**
- [OPTION_B_RESULTS.md](OPTION_B_RESULTS.md) - This document

### Key Data Structures

**OctreeCellDataWithNeighbors** (CPU):
```python
@dataclass
class OctreeCellDataWithNeighbors:
    # ... all base octree fields ...
    cell_neighbors: np.ndarray  # (n_cells, 26) int32
```

**MeshAlignedOctreeGPUWithNeighbors** (GPU):
```python
@dataclass
class MeshAlignedOctreeGPUWithNeighbors:
    # ... all base octree fields ...
    cell_neighbors: jax.Array  # (n_cells, 26) int32
```

---

## Performance Results

### Neighbor Table Statistics

```
Cells: 517,069
Total neighbor lookups: 13,443,794 (517k × 26)
Found: 11,818,636 (87.9%)
Missing (boundary): 1,625,158 (12.1%)
Mean neighbors per cell: 22.9
Memory: 51.3 MB
Build time: 177.5s (CPU)
```

### Point Location Performance

**Test:** 10,000 random particles in mesh bounding box

| Method | Searchability | Tests/Particle | Throughput | Memory |
|--------|---------------|----------------|------------|--------|
| **Baseline (primary only, 8 levels)** | 74.6% | 4.8 | 12,106 p/s | 83 MB |
| **Option B (8 levels + 26 neighbors each)** | **89.74%** | **13.9** | **1,504 p/s** | **134 MB** |

**Improvement:**
- +15.1 percentage points searchability
- +51 MB memory (+61%)
- 2.9× more tests per particle
- 8× slower throughput

---

## Analysis

### Why Not 99% Searchability?

Despite searching 27 cells (primary + 26 neighbors) at 8 levels, we only achieve 89.74%.

**Possible reasons:**
1. **Particles in voids:** ~10-15% of random bbox positions fall outside the mesh
2. **Elements span beyond 26-neighbor range:** Large stretched tetrahedra may span >1 cell in multiple directions
3. **Level mismatch:** An element at level 12 won't be found by searching level 14 neighbors (finer cells)

**Analysis of the 10.26% not found:**
- Baseline finds 74.6% at primary cells only
- We find 89.74% with neighbors
- Gain: 15.14 percentage points
- Still missing: 10.26% (could be voids or extreme spanning elements)

### Why 8× Slower Throughput?

The throughput dropped from 12,106 p/s to 1,504 p/s (8× slower).

**Root cause:** Unconditional execution of all 216 cell searches (8 levels × 27 cells/level)

**Current implementation:**
```python
for level in [14, 13, 12, 11, 10, 9, 8, 7]:  # 8 levels
    search_primary_cell()
    for neighbor in range(26):  # Always execute
        search_neighbor_cell()
```

Even after finding the element, we continue searching all remaining cells! The `jnp.where` checks prevent updating the result, but the searches still execute.

**Optimization opportunities:**
1. **Early exit after found:** Use `lax.cond` to stop searching after found (but this caused memory issues before - need careful structuring)
2. **Reduce levels searched:** Search only finest levels (14-12) where most elements are
3. **Adaptive neighbor count:** Only search full 26 neighbors if primary cell fails

---

## Comparison with Previous Phase 2 Attempts

### Attempt 1: Dynamic Neighbor Computation with `lax.cond`
- **File:** `mesh_aligned_point_location_with_neighbors.py`
- **Result:** ❌ 631 GB memory allocation (RESOURCE_EXHAUSTED)
- **Cause:** Nested `lax.cond` inside `lax.fori_loop` inside `vmap`

### Attempt 2: Pure Functional with `jnp.where`
- **File:** `mesh_aligned_neighbors_simple.py`
- **Result:** ❌ Shape mismatch errors in point-in-tet
- **Cause:** Invalid element lookups when `cell_idx = -1`

### Attempt 3: Pre-Computed Neighbor Table (Option B)
- **Files:** `mesh_aligned_octree_with_neighbor_table.py`, `mesh_aligned_search_with_neighbors.py`
- **Result:** ✅ **89.74% searchability, stable execution**
- **Success factors:**
  - No dynamic neighbor computation on GPU
  - Simple O(1) array lookups
  - Unconditional execution (vmap-safe)

---

## Recommendations

### For Immediate Use

**Option B is ready for production** with caveats:

✅ **Use cases:**
- Particle tracking where 89.74% coverage is acceptable
- Combine with KD-tree for initial placement (covers the missing 10.26%)
- L2 search method in tracking pipeline

⚠️ **Limitations:**
- 8× slower than baseline (1,504 vs 12,106 p/s)
- Not suitable as standalone point location (90% coverage)
- Higher memory usage (+51 MB)

**Recommended configuration:**
```python
octree_with_neighbors = add_neighbor_table_to_octree(octree_cells)
octree_gpu = upload_octree_with_neighbors_to_gpu(connectivity, node_positions, octree_with_neighbors)

elem_ids, n_tests = search_batch_with_precomputed_neighbors(
    positions,
    octree_gpu,
    levels_to_try=(14, 13, 12),  # Reduce to 3 levels for better throughput
    max_tests_per_cell=20
)
```

### For Future Optimization

**Priority 1: Improve throughput (target: 5-10K p/s)**

1. **Reduce levels searched:**
   - Try `levels_to_try=(14, 13, 12)` instead of 8 levels
   - Expected: 3× faster (search 81 cells instead of 216)
   - May reduce searchability slightly

2. **Early exit optimization:**
   - Restructure to use `lax.scan` instead of Python for-loop
   - Use carry state to propagate "found" flag
   - Skip neighbor searches after found

3. **Adaptive neighbor search:**
   - Only search neighbors if primary cell fails
   - Expected: 75% of particles found in primary → 4× speedup for those

**Priority 2: Increase searchability (target: 95%+)**

1. **Investigate the missing 10.26%:**
   - Are they in voids (outside mesh)?
   - Are they in elements that span >26 neighbors?
   - Plot positions of unfound particles

2. **Expand neighbor range for coarse levels:**
   - Level 8-10 cells are large → may need 5×5×5 neighbors (124 neighbors)
   - Level 14 cells are tiny → 3×3×3 neighbors sufficient

3. **Hybrid approach:**
   - Use neighbor search for 90%
   - Fall back to KD-tree for remaining 10%

---

## Performance Comparison Table

| Method | Searchability | Elem/Cell | Tests/Part | Throughput | Memory | Notes |
|--------|---------------|-----------|------------|------------|--------|-------|
| **Old broken (v2)** | 17.7% | 11.5 | 4.6 | ~2K p/s | 74 MB | Min-vertex bug |
| **Phase 1 fixed (v3)** | 74.6% | 5.9 | 4.8 | 12K p/s | 83 MB | Centroid-based, primary only |
| **Option B (current)** | **89.74%** | **5.9** | **13.9** | **1.5K p/s** | **134 MB** | 8 levels + 26 neighbors |
| **Target** | 99% | 5.9 | ~20 | ~50K p/s | ~135 MB | Optimized Option B |

---

## Conclusion

✅ **Option B Successfully Implemented**
- Avoids JAX memory explosion issues
- Achieves 89.74% searchability (+15.1pp over baseline)
- Stable, production-ready code

⚠️ **Performance Trade-offs**
- 8× slower throughput (optimization needed)
- 90% searchability (not 99% target)

📋 **Next Steps**
1. **Ship Option B as-is** for tracking pipeline (acceptable performance)
2. **Optimize throughput** by reducing levels and early exit
3. **Investigate 10% unfound particles** (voids vs spanning elements)
4. **Consider hybrid approach** (neighbor search + KD-tree fallback)

---

**End of Option B Results**
