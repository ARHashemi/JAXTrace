# Implementation Plan: Phase 1 & Phase 2 Fixes

**Date**: 2026-02-13
**Goal**: Achieve >95% retention @ 100 RK4 steps
**Current**: ~80% retention

---

## Summary of Issues

| Issue | Impact | Priority |
|-------|--------|----------|
| 1,826 missing elements | 0.06% | 1 (Foundation) |
| 23.52% not searchable in 2×2×2 | 23.52% | 2 (**CRITICAL**) |
| 12.13% cross-cell faces | 12.13% | 3 (Lower - multi-cell helped!) |

**Total expected gain after Phase 1+2**: ~24% improvement → **~95%+ retention** ✅

---

## Phase 1: Cover All Elements (1,826 Missing)

### Analysis Complete ✅

**CSR Structure Verified**:
```python
# Line 308-312 in mesh_aligned_point_location.py:
cell_found_elem, cell_tests = jax.lax.fori_loop(
    0, n_elems_in_cell,  # ← VARIABLE, no padding needed!
    test_element,
    (inner_found_elem, inner_tests)
)
```

**Conclusion**: ✅ **SAFE to add elements with varying counts per cell**

### Diagnostic Running

**Script**: `diagnose_missing_elements.py`
**Purpose**: Analyze the 1,826 non-Kuhn elements:
- Geometry (axis-aligned edges, sizes)
- Spatial distribution
- Relationship to covered elements
- Fix strategy

**Expected findings**:
- >90% have face or edge neighbors (2-3 shared nodes)
- Can register in neighbor's cells
- Remaining few: use fallback grid level

### Implementation

**File to modify**: `jaxtrace/gpu/search/mesh_aligned_octree_vertex_multi.py`

**Current code (lines 131-134)**:
```python
if np.any(cell_size == 0):
    # Skip non-Kuhn elements
    n_skipped += 1
    continue  # ← Problem: 1,826 elements lost!
```

**Proposed fix**:
```python
if np.any(cell_size == 0):
    # Non-Kuhn element - register in neighbor cells
    n_non_kuhn += 1

    # Strategy 1: Find neighbor elements sharing face/edge
    neighbor_elements = find_face_or_edge_neighbors(elem_id, connectivity)

    if len(neighbor_elements) > 0:
        # Register in same cells as neighbors
        for neighbor_id in neighbor_elements:
            if neighbor_id in element_to_cells_dict:
                neighbor_cells = element_to_cells_dict[neighbor_id]
                for cell_key in neighbor_cells:
                    element_to_cells_dict[elem_id].add(cell_key)
                    cell_to_elements_dict[cell_key].add(elem_id)
                break  # Use first neighbor's cells
    else:
        # Strategy 2: Fallback to coarse grid level
        fallback_level = 8  # Coarse level
        fallback_cell_size = compute_cell_size(fallback_level)

        # Register element's centroid in coarse grid
        centroid = vertices.mean(axis=0)
        i = int(np.floor(centroid[0] / fallback_cell_size[0]))
        j = int(np.floor(centroid[1] / fallback_cell_size[1]))
        k = int(np.floor(centroid[2] / fallback_cell_size[2]))

        morton = encode_morton_3d_single(i, j, k)
        cell_key = (morton, fallback_level)

        element_to_cells_dict[elem_id].add(cell_key)
        cell_to_elements_dict[cell_key].add(elem_id)

    continue  # Skip normal vertex registration
```

**Helper function to add**:
```python
def find_face_or_edge_neighbors(elem_id, connectivity):
    """Find elements sharing 2+ nodes (edge or face)."""
    elem_nodes = set(connectivity[elem_id])
    neighbors = []

    for other_id in range(len(connectivity)):
        if other_id == elem_id:
            continue
        other_nodes = set(connectivity[other_id])
        shared = elem_nodes & other_nodes

        if len(shared) >= 2:  # Face (3) or edge (2)
            neighbors.append((other_id, len(shared)))

    # Sort by number of shared nodes (descending)
    neighbors.sort(key=lambda x: x[1], reverse=True)
    return [n[0] for n in neighbors]
```

### Testing

```bash
# 1. Run diagnostic
python diagnose_missing_elements.py

# 2. Apply fix to mesh_aligned_octree_vertex_multi.py

# 3. Verify coverage
python diagnose_multi_cell_coverage.py
# Expected: "Elements NOT in octree: 0" ✅

# 4. Benchmark
python benchmark_l2_search_methods_with-export.py
# Expected: Small retention improvement (0.06%)
```

---

## Phase 2: Expand 2×2×2 to 3×3×3

### Analysis Complete ✅

**File**: `PHASE2_3x3x3_ADAPTIVE_MESH_ANALYSIS.md`

**Key findings**:
1. ✅ Multi-level search ALREADY handles adaptive refinement
2. ✅ Tries all levels 14→7 with appropriate cell sizes
3. ✅ 3×3×3 at EACH level catches 1:2 and 2:1 neighbors
4. ❌ Current 2×2×2 misses 23.52% of elements

### Implementation

**File to modify**: `jaxtrace/gpu/search/mesh_aligned_point_location.py`

**Change 1: Offset array (line 334-343)**

```python
# OLD (2×2×2):
cell_offsets = jnp.array([
    [-1, -1, -1],
    [-1, -1,  0],
    [-1,  0, -1],
    [-1,  0,  0],
    [ 0, -1, -1],
    [ 0, -1,  0],
    [ 0,  0, -1],
    [ 0,  0,  0],
], dtype=jnp.int32)

# NEW (3×3×3):
offsets_list = []
for di in [-1, 0, 1]:
    for dj in [-1, 0, 1]:
        for dk in [-1, 0, 1]:
            offsets_list.append([di, dj, dk])

cell_offsets = jnp.array(offsets_list, dtype=jnp.int32)
```

**Change 2: Loop count (line 346)**

```python
# OLD:
level_found_elem, level_tests = jax.lax.fori_loop(
    0, 8,  # 2×2×2
    lambda i, c: try_cell(cell_offsets[i], c),
    (found_elem, total_tests)
)

# NEW:
level_found_elem, level_tests = jax.lax.fori_loop(
    0, 27,  # 3×3×3
    lambda i, c: try_cell(cell_offsets[i], c),
    (found_elem, total_tests)
)
```

**Change 3: max_tests default (line 182)**

```python
# OLD:
def search_mesh_aligned_octree_multi_local(
    pos: jax.Array,
    octree_gpu: MeshAlignedOctreeGPU,
    max_tests: jnp.int32 = 200
)

# NEW:
def search_mesh_aligned_octree_multi_local(
    pos: jax.Array,
    octree_gpu: MeshAlignedOctreeGPU,
    max_tests: jnp.int32 = 600  # 3×3×3 needs ~494 tests
)
```

**Change 4: Update comments**

```python
# Line 185-211: Update docstring
"""
Find containing element using 3×3×3 local neighborhood search.

For multi-cell vertex registration where each element is registered in ~4 cells
(where its vertices are located), we need to search a local neighborhood of cells
to find the element.

This function searches 27 cells (3×3×3 cube) centered around the particle position.
This covers all possible cells where the element's vertices could be registered,
including cases with adaptive mesh refinement (1:2 and 2:1 face neighbors).

Algorithm:
    1. For each refinement level:
        a. Compute base cell indices (i, j, k) for particle position
        b. Search 27 cells: (i+di, j+dj, k+dk) for di,dj,dk in [-1,0,1]
        c. Test elements in each cell
        d. Return first containing element found
    2. Try levels from finest to coarsest (14, 13, 12, ..., 7)
    3. Multi-level search automatically handles adaptive refinement

Args:
    pos: (3,) float32 - query position
    octree_gpu: GPU octree structure (multi-cell vertex registration)
    max_tests: Maximum elements to test across all cells (default 600)

Returns:
    (elem_id, n_tests):
        elem_id: Element ID (-1 if not found)
        n_tests: Total number of point-in-tet tests
"""
```

```python
# Line 328-333: Update comment
# Define 27 cell offsets for 3×3×3 neighborhood CENTERED on particle
# Covers ±1 cell in each direction from base cell (i,j,k).
# This handles:
#   - Elements with vertices up to 2 cells away
#   - Adaptive refinement boundaries (1:2 and 2:1 neighbors)
#   - Multi-level elements (vertices at different refinement levels)
# The multi-level search (levels 14→7) ensures we find vertices
# registered at any level by searching at that level's grid resolution.
```

### Testing

```bash
# 1. Apply fix to mesh_aligned_point_location.py

# 2. Verify searchability
python diagnose_multi_cell_coverage.py
# Expected: "NOT searchable in 3×3×3: <5%" ✅

# 3. Benchmark
python benchmark_l2_search_methods_with-export.py
# Expected: Retention ~95%+ (vs ~80% currently) ✅

# 4. Performance check
# Expected: Throughput ~10-15K p/s (vs ~37K, due to 3.38× more tests)
# Trade-off: 2.5× slower but 25% better retention
```

---

## Expected Results After Both Phases

### Before Fixes

| Method | Retention @ 100 steps |
|--------|----------------------|
| Multi-Cell 2×2×2 (current) | ~80% |

**Issues**:
- 0.06% missing elements
- 23.52% not searchable in 2×2×2
- 12.13% cross-cell faces

### After Phase 1

| Method | Retention @ 100 steps |
|--------|----------------------|
| Multi-Cell 2×2×2 | ~80.06% |

**Fixed**: 0.06% missing elements ✅
**Remaining**: 23.52% searchability issue

### After Phase 1 + Phase 2

| Method | Retention @ 100 steps |
|--------|----------------------|
| Multi-Cell 3×3×3 | **~95%+** ✅ |

**Fixed**:
- ✅ 0.06% missing elements
- ✅ 23.52% searchability (now >95% searchable)
- ✅ Most cross-cell cases covered by 3×3×3

**Performance**:
- Tests: 146 → 494 (3.38× increase)
- Throughput: ~37K → ~10-15K p/s
- **Retention: 80% → 95%+** (TARGET ACHIEVED!)

---

## Phase 3 (If Needed): Cross-Cell Faces

**Only needed if retention < 95% after Phase 2**

**Solution**: Use existing Option B (neighbor tables)
- Already implemented: `mesh_aligned_octree_with_neighbor_table.py`
- Pre-computes face-neighbor relationships
- Fallback when 3×3×3 fails

**Hybrid approach**:
1. Try 3×3×3 local search (fast)
2. If not found, use neighbor table (comprehensive)
3. Expected retention: >98%

---

## Implementation Timeline

### Step 1: Run Missing Elements Diagnostic (Running)
**Status**: Background task bfa1df5
**Output**: `logs/diagnose_missing_elements.log`

### Step 2: Implement Phase 1 Fix
**File**: `mesh_aligned_octree_vertex_multi.py`
**Complexity**: Medium (add neighbor finding + fallback)
**Time**: 1-2 hours

### Step 3: Test Phase 1
**Scripts**: Both diagnostics + benchmark
**Time**: 30 minutes

### Step 4: Implement Phase 2 Fix
**File**: `mesh_aligned_point_location.py`
**Complexity**: LOW (3 lines: offsets, loop count, max_tests)
**Time**: 10 minutes

### Step 5: Test Phase 2
**Scripts**: Both diagnostics + benchmark
**Time**: 30 minutes

### Step 6: Evaluate & Document
**Goal**: Verify >95% retention achieved
**Time**: 30 minutes

**Total estimated time**: 3-4 hours

---

## Success Criteria

✅ **Phase 1 Complete**:
- Missing elements: 1,826 → 0
- Diagnostic: "Elements NOT in octree: 0"

✅ **Phase 2 Complete**:
- Searchable: 76.48% → >95%
- Diagnostic: "NOT searchable in 3×3×3: <5%"
- Benchmark: Retention @ 100 steps: >95%

✅ **Overall Success**:
- Retention matches centroid seeding (100%)
- Retention matches Morton radius=30 baseline (~95%)
- Tetrahedral voids eliminated in visualizations

---

## Conclusion

**Your priority order is PERFECT** ✅

The plan addresses both issues correctly:
1. Clean foundation (cover all elements)
2. Fix dominant failure (23.52% searchability)
3. Adaptive refinement handled automatically by multi-level search

**No special handling needed for 1:2 face neighbors** - the existing multi-level search loop (levels 14→7) automatically covers different refinement levels by searching at appropriate grid resolutions.

**Expected outcome**: >95% retention with 3×3×3 local search! 🎯
