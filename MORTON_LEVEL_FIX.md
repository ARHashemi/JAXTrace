# Morton + Level Fix: Correcting Cell Identification

## Problem Identified

**Symptom:** Elements per cell = 12.27 (expected ~5-6)

**Root Cause:** Morton codes without refinement level caused collisions.

### Why the collision occurred:

```python
# Element A at level 14, grid position (10, 20, 30)
morton_A = encode_morton(10, 20, 30)  # → 0x12345678

# Element B at level 13, grid position (10, 20, 30)
morton_B = encode_morton(10, 20, 30)  # → 0x12345678 (SAME!)

# Both elements assigned to the SAME cell!
# Result: 2× expected elements per cell
```

When a coarse cell (level 13) and fine cell (level 14) occupy the same grid position, they were treated as one cell. This caused:
- Multiple refinement levels merged into single cells
- 12.27 elements/cell instead of 5-6
- Only 248k unique cells instead of expected 500k+

## Solution

**Key Change:** Use `(morton_code, level)` tuple as cell key instead of `morton_code` alone.

### Modified Files:

1. **`mesh_aligned_octree_single_cell.py`** (Phase 2: CPU extraction)
   - Line 218-232: Changed `cell_to_elements_dict[morton]` → `cell_to_elements_dict[(morton, level)]`
   - Line 247-265: Updated CSR building to handle `(morton, level)` tuples
   - Line 278-283: Changed `element_to_cells` from Morton uint64 to cell index int32
   - Updated data structure comments

2. **`mesh_aligned_octree_gpu.py`** (Phase 3: GPU structure)
   - Added `find_cell_by_morton_and_level()` function (lines 206-260)
   - Performs binary search on sorted `(morton, level)` pairs
   - Lexicographic comparison: morton first, then level
   - Marked `find_cell_by_morton()` as deprecated

3. **`mesh_aligned_point_location.py`** (Phase 4: Point location kernel)
   - Line 119-128: Updated to call `find_cell_by_morton_and_level()`
   - Now passes both Morton code AND level to lookup
   - Import updated to include new function

4. **`__init__.py`**
   - Added `find_cell_by_morton_and_level` to exports

## Expected Results After Fix

### Phase 2 (CPU Extraction):
```
Before (v2):
  Cells per element: 1.00 ✅
  Elements per cell: 12.27 ❌
  Unique cells: 248,321

After (v3):
  Cells per element: 1.00 ✅
  Elements per cell: ~5-6 ✅ (expected)
  Unique cells: ~500k+ (more cells due to level separation)
```

### Phase 4 (GPU Point Location):
```
Before (v2):
  Searchability: Unknown (likely <100%)
  Problem: Level collisions caused incorrect cell lookups

After (v3):
  Searchability: ~100% (expected)
  Reason: Each element in exactly ONE (morton, level) cell
  Query matches assignment: same grid computation + level
```

## Why This Guarantees 100% Searchability

**For each query position P at level L:**

1. **Compute grid indices:** `(i, j, k) = floor(P / cell_size(L))`
2. **Encode to Morton:** `morton = encode_morton_3d(i, j, k)`
3. **Search for cell:** `(morton, L)` → binary search → cell_idx
4. **Test elements:** All elements in cell

**Guarantee:** If position P is inside element E, and E belongs to cell C at level L, then:
- E was assigned to cell C with key `(morton_E, level_E)`
- Query with level `L = level_E` computes same grid indices
- Query finds same cell C
- E is tested and found ✅

**Critical requirement:** Multi-level search must try ALL refinement levels present in mesh.

## Testing

### Test Script: `test_single_cube_extraction_v2.py`

Runs Phase 2 extraction with corrected `(morton, level)` keys.

**Expected output:**
```
Elements per cell: ~5-6 (should match Kuhn subdivision)
Unique cells: >300k (more than v2's 248k)
```

### Next Steps:

1. ✅ Run `test_single_cube_extraction_v2.py` to verify Phase 2 fix
2. Update `test_mesh_aligned_octree_gpu.py` to use corrected Phase 2
3. Run full GPU test to verify 100% searchability
4. Profile performance vs Morton blocks

## Technical Details

### Binary Search with (Morton, Level) Pairs

The cells are sorted lexicographically by `(morton, level)`:
```python
sorted_cell_keys = sorted(cell_to_elements_dict.keys())
# Result: [(morton1, level1), (morton2, level1), (morton2, level2), ...]
```

Binary search compares tuples:
```python
(morton_mid, level_mid) < (morton_query, level_query)
# True if morton_mid < morton_query
# OR (morton_mid == morton_query AND level_mid < level_query)
```

This ensures:
- Cells at different levels are SEPARATE
- Search finds exact (morton, level) match
- No collisions between refinement levels

### Data Structure Changes

**Before (v2):**
```python
element_to_cells: np.ndarray  # (n_elements,) uint64 - Morton codes
```

**After (v3):**
```python
element_to_cells: np.ndarray  # (n_elements,) int32 - cell indices
```

This is more efficient:
- Smaller memory (int32 vs uint64)
- Direct index lookup (no binary search needed for reverse)
- Handles skipped elements with -1

## Comparison: All Versions

### v1: Bbox Overlap (WRONG)
```
Algorithm: Find all grid cells overlapping element bbox
Result: 8 cells per element
Problem: Wrong cells (bbox corners, not parent cube)
Searchability: 2.4%
```

### v2: Single Parent Cube, Morton Only (PARTIAL)
```
Algorithm: floor(min_vertex / cell_size) * cell_size
Cell key: morton only
Result: 1 cell per element ✅, 12.27 elements per cell ❌
Problem: Refinement level collisions
Searchability: Unknown (likely <100%)
```

### v3: Single Parent Cube, (Morton, Level) (CORRECT)
```
Algorithm: floor(min_vertex / cell_size) * cell_size
Cell key: (morton, level)
Result: 1 cell per element ✅, ~5-6 elements per cell ✅
Searchability: ~100% (expected)
```

## Performance Impact

**Expected improvements:**

- **CPU extraction:** Minimal change (same algorithm, different key)
- **GPU memory:** Slightly more cells (~500k vs 248k) but still reasonable
- **GPU search:** Same search complexity (log2(n_cells) ≈ 19-20 iterations)
- **Point-in-tet tests:** Reduced from ~87 to ~5-6 per query (15× reduction!)
- **Searchability:** From 2.4% to ~100% (42× improvement)

**Overall speedup vs Morton blocks:**
```
Morton blocks: ~536 tests per query, 93-98% searchability
Mesh-aligned v3: ~5-6 tests per query, ~100% searchability
Speedup: ~90-100× reduction in tests
```

## Conclusion

The Morton + Level fix resolves the fundamental cell collision issue. By using `(morton, level)` as the cell key, we ensure:

1. ✅ Each element assigned to exactly ONE cell
2. ✅ Cells represent actual parent octree cubes at specific refinement levels
3. ✅ No collisions between different refinement levels
4. ✅ Query lookup matches assignment (same grid computation)
5. ✅ Expected 100% searchability
6. ✅ Expected ~5-6 elements per cell (Kuhn subdivision)

The fix is complete and ready for testing!
