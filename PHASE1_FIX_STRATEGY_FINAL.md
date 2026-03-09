# Phase 1: Fix Strategy for 1,826 Missing Elements

**Date**: 2026-02-13
**Diagnostic Complete**: ✅ `logs/diagnose_missing_elements.log`

---

## Key Findings from Diagnostic

### ✅ **100% Have Face Neighbors!**

```
Neighbor analysis (1,826 elements):
  Elements with face neighbors (3 nodes): 1,826 (100.0%)
  Elements with edge neighbors (2 nodes): 0 (0.0%)
  Elements with vertex neighbors (1 node): 0 (0.0%)

Maximum shared nodes distribution:
  3 nodes: 1,826 elements (100.0%)
```

**This is PERFECT!** All 1,826 missing elements share a face (3 nodes) with at least one covered element.

### Geometry Characteristics

```
Axis-aligned edges per element:
  2 edges: 1,826 elements (100.0%)
```

**Explanation**: These are **transition elements** at refinement boundaries:
- Have exactly 2 axis-aligned edges (not 3 like Kuhn tets)
- Not aligned with the octree grid properly
- Skipped by `find_axis_aligned_edges_single()` which expects 3 AA edges

**Volume statistics**:
- Very small: median = 5.2e-12, max = 2.1e-08
- At refinement boundaries (fine → coarse transitions)

**Spatial distribution**:
- Concentrated near origin: X ∈ [-0.026, 0.028], Y ∈ [-0.019, 0.022], Z ∈ [-0.009, -0.003]
- At mesh refinement boundaries

---

## Recommended Fix Strategy

### ✅ **Strategy: Register in Face Neighbor's Cells**

Since **100%** of missing elements have face neighbors, we can simply:

1. Find face neighbor (3 shared nodes)
2. Copy that neighbor's cell registrations
3. Add missing element to those cells

**This will achieve 100% coverage!**

---

## Implementation Steps

### Step 1: Add Helper Function

**File**: `jaxtrace/gpu/search/mesh_aligned_octree_vertex_multi.py`

Add this function before `extract_octree_cells_vertex_multi()`:

```python
def find_face_neighbor_fast(elem_id: int, connectivity: np.ndarray,
                           element_to_cells_dict: dict,
                           max_search: int = 100) -> int:
    """
    Find a covered element sharing a face (3 nodes) with this element.

    Fast version: Only checks recently processed elements (likely nearby).

    Args:
        elem_id: Element ID to find neighbor for
        connectivity: Mesh connectivity
        element_to_cells_dict: Dictionary of already-processed elements
        max_search: Maximum number of recent elements to check

    Returns:
        neighbor_id: Element ID of face neighbor, or -1 if not found
    """
    elem_nodes = set(connectivity[elem_id])

    # Check recent elements (likely to be spatially nearby)
    covered_ids = list(element_to_cells_dict.keys())
    start_idx = max(0, len(covered_ids) - max_search)

    for neighbor_id in covered_ids[start_idx:]:
        neighbor_nodes = set(connectivity[neighbor_id])
        shared = elem_nodes & neighbor_nodes

        if len(shared) >= 3:  # Face neighbor
            return neighbor_id

    # If not found in recent, search ALL covered elements
    for neighbor_id in covered_ids:
        neighbor_nodes = set(connectivity[neighbor_id])
        shared = elem_nodes & neighbor_nodes

        if len(shared) >= 3:  # Face neighbor
            return neighbor_id

    return -1  # Should never happen based on diagnostic!
```

### Step 2: Modify Main Loop

**File**: `jaxtrace/gpu/search/mesh_aligned_octree_vertex_multi.py`

**Current code (lines 131-134)**:
```python
if np.any(cell_size == 0):
    # Skip non-Kuhn elements
    n_skipped += 1
    continue
```

**Replace with**:
```python
if np.any(cell_size == 0):
    # Non-Kuhn transition element - register in face neighbor's cells
    n_non_kuhn += 1

    # Find face neighbor
    neighbor_id = find_face_neighbor_fast(
        elem_id, connectivity, element_to_cells_dict, max_search=100
    )

    if neighbor_id >= 0:
        # Copy neighbor's cells
        neighbor_cells = element_to_cells_dict[neighbor_id]
        for cell_key in neighbor_cells:
            element_to_cells_dict[elem_id].add(cell_key)
            cell_to_elements_dict[cell_key].add(elem_id)
        total_vertex_cell_registrations += len(neighbor_cells)
    else:
        # Fallback: Use coarse grid level (should never happen!)
        # Based on diagnostic, all elements have face neighbors
        # But keep this for safety
        fallback_level = 8
        fallback_size = 0.04  # level 8 cell size

        centroid = vertices.mean(axis=0)
        i = int(np.floor(centroid[0] / fallback_size))
        j = int(np.floor(centroid[1] / fallback_size))
        k = int(np.floor(centroid[2] / fallback_size))

        offset = (1 << 19)
        max_coord = (1 << 20)
        i_morton = np.clip(i + offset, 0, max_coord - 1)
        j_morton = np.clip(j + offset, 0, max_coord - 1)
        k_morton = np.clip(k + offset, 0, max_coord - 1)

        morton = encode_morton_3d_single(i_morton, j_morton, k_morton, max_depth=21)
        cell_key = (morton, fallback_level)

        element_to_cells_dict[elem_id].add(cell_key)
        cell_to_elements_dict[cell_key].add(elem_id)
        total_vertex_cell_registrations += 1

        if verbose:
            print(f"    WARNING: Element {elem_id} has no face neighbor, used fallback")

    continue  # Skip normal vertex registration
```

### Step 3: Update Progress Messages

**Line 172-176**:
```python
if verbose:
    print(f"  ✅ Element→cell mapping complete!")
    print(f"    Non-Kuhn elements: {n_non_kuhn:,}")  # Changed from "Skipped"
    print(f"    Mapped {n_elements:,} elements to cells")  # All elements now!
    print(f"    Total registrations: {total_vertex_cell_registrations:,}")
    print(f"    Cells per element (mean): {total_vertex_cell_registrations / n_elements:.2f}")
```

### Step 4: Update Variable Initialization

**Line 120**:
```python
n_skipped = 0  # Remove this line
n_non_kuhn = 0  # Add this line
```

---

## Expected Results After Fix

### Before Fix

```
Mapped 3,047,074 elements to cells
Skipped 1,826 non-Kuhn elements
Elements NOT in octree: 1,826 (0.06%)
```

### After Fix

```
Mapped 3,048,900 elements to cells  ← All elements!
Non-Kuhn elements: 1,826
Elements NOT in octree: 0 (0.00%)   ← Perfect coverage!
```

---

## Testing Plan

### Test 1: Verify Coverage

```bash
python diagnose_multi_cell_coverage.py 2>&1 | tee logs/diagnose_multi_cell_coverage_phase1.log
```

**Expected output**:
```
Test 1: Multi-Cell Vertex Registration Coverage
  Total elements: 3,048,900
  Elements in octree: 3,048,900     ← Was 3,047,074
  Elements NOT in octree: 0          ← Was 1,826

  Cells per element distribution:
    0 cells: 0 elements (0.00%)      ← Was 1,826
    4 cells: 3,048,900 elements (100.00%)

  ✅ ALL elements covered!
```

### Test 2: Verify Searchability (Should be unchanged)

```
Test 2: 2×2×2 Local Search Pattern Analysis
  Searchable in 2×2×2 neighborhood: ~76%  ← Still needs Phase 2
  NOT searchable: ~23%                     ← Still needs Phase 2
```

**Why unchanged?** The 1,826 elements are now IN the octree, but 2×2×2 is still too small for the 23.52% issue. Phase 2 will fix this.

### Test 3: Benchmark

```bash
python benchmark_l2_search_methods_with-export.py
```

**Expected**: Slight improvement (~0.06%) in retention at step 0 (initial assignment).

---

## Alternative Approach (If Face Neighbor Strategy Fails)

Based on diagnostic, this should **never** be needed (100% have face neighbors), but for completeness:

### Fallback: Compute Approximate Cell from Centroid

```python
# Use element's bounding box to estimate cell size
bbox_size = bbox_max - bbox_min
estimated_level = estimate_level_from_size(bbox_size)
cell_size = level_to_cell_size(estimated_level)

# Register centroid in that level's grid
centroid = vertices.mean(axis=0)
i = floor(centroid[0] / cell_size[0])
j = floor(centroid[1] / cell_size[1])
k = floor(centroid[2] / cell_size[2])

# Add to octree
morton = encode_morton_3d(i, j, k)
cell_key = (morton, estimated_level)
register_element(elem_id, cell_key)
```

---

## Performance Impact

### Memory

```
Before: 12,188,296 registrations (3,047,074 elements × 4 cells)
After:  12,195,600 registrations (3,048,900 elements × 4 cells)

Increase: 7,304 registrations (0.06%)
Memory: ~29 KB additional (negligible)
```

### Computation

- Face neighbor finding: O(1826 × 100) = ~180K comparisons
- Added during octree construction (one-time cost)
- Construction time: +1-2 seconds (from ~330s to ~332s)
- **No impact on search performance!**

---

## Why This Fix is Correct

1. ✅ **CSR structure supports variable elements per cell** (verified in code)
2. ✅ **100% of missing elements have face neighbors** (verified in diagnostic)
3. ✅ **Face neighbors share same spatial region** → same octree cells appropriate
4. ✅ **No memory explosion** (only 0.06% more registrations)
5. ✅ **Complete coverage** → foundation for Phase 2

---

## Summary

**Fix Strategy**: ✅ **Register missing elements in face neighbor's cells**

**Implementation**:
- Add `find_face_neighbor_fast()` helper
- Modify non-Kuhn handling in main loop
- ~30 lines of code

**Testing**:
1. Run diagnostic → 0 missing elements
2. Run benchmark → verify no regression
3. Proceed to Phase 2 → fix 23.52% searchability

**Expected Outcome**: 100% element coverage, foundation for Phase 2 fix ✅
