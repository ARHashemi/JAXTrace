# Multi-Cell 2×2×2 Diagnostic Status

**Date**: 2026-02-13
**Task**: Trace `mesh_aligned_octree_multi_local` to identify particle loss sources

---

## Current Status

✅ **Complete tracing of method flow**
✅ **Identified 6 potential failure modes**
✅ **Created comprehensive diagnostic script**
🔄 **Running diagnostics** → `logs/diagnose_multi_cell_coverage.log`

---

## Files Created

1. **`diagnose_multi_cell_coverage.py`** (373 lines)
   - Test 1: Multi-cell registration coverage
   - Test 2: 2×2×2 search pattern analysis
   - Test 3: Cross-cell boundary face sharing

2. **`MULTI_CELL_2x2x2_SEARCH_ANALYSIS.md`** (Complete reference)
   - Method architecture
   - All 6 failure modes documented
   - Diagnostic plan
   - Next steps

3. **`DIAGNOSTIC_STATUS.md`** (This file)
   - Current status tracker

---

## Method Flow Traced

```
benchmark_l2_search_methods_with-export.py
├─ Config: l2_method='mesh_aligned_octree_multi_local' (line 1069)
├─ Build: extract_octree_cells_vertex_multi() (line 726)
│   └─ mesh_aligned_octree_vertex_multi.py (lines 69-268)
│       ├─ Register each element in ALL cells its vertices touch
│       └─ Creates bidirectional CSR: cell↔elements
├─ Upload: upload_mesh_aligned_octree_to_gpu() (line 760)
└─ Search: create_rk4_fully_fused_timedep() (line 547)
    └─ rk4_fully_fused_timedep.py (line 297)
        └─ search_mesh_aligned_octree_multi_local()
            └─ mesh_aligned_point_location.py (lines 179-368)
                ├─ Try levels 14, 13, ..., 7
                ├─ For each level: compute base cell (i,j,k)
                ├─ Search 8 cells: offsets [-1,-1,-1] to [0,0,0]
                └─ Test elements in each cell
```

---

## Six Potential Failure Modes

### 1. Elements Not in Octree (1,826 non-Kuhn)
**File**: `mesh_aligned_octree_vertex_multi.py:131-134`
```python
if np.any(cell_size == 0):
    n_skipped += 1
    continue  # ← Elements without axis-aligned edges
```
**Impact**: 0.06% of elements completely missing
**Test**: Count elements with `cells_per_element == 0`

### 2. Incomplete Vertex Registration
**Expected**: 4 cells per Kuhn tet (one per vertex)
**Actual**: May be < 4 due to boundary effects
**Impact**: Particles near unregistered vertices → lost
**Test**: Count elements with `cells_per_element < 4`

### 3. 2×2×2 Pattern Too Small
**Issue**: Element vertices may span > 2 cells
**Code**: `mesh_aligned_point_location.py:334-343`
```python
cell_offsets = [
    [-1,-1,-1], [-1,-1,0], [-1,0,-1], [-1,0,0],
    [0,-1,-1], [0,-1,0], [0,0,-1], [0,0,0]
]  # Only covers 2×2×2 cube
```
**Impact**: If vertex at (i_p-2, j_p, k_p) → outside 2×2×2
**Test**: Check if all vertex cells in 2×2×2 neighborhood of centroid

### 4. Cross-Cell Face Sharing
**Known from logs**: 48.75% of faces cross cell boundaries
**Issue**: Face-sharing neighbors may be > 2 cells apart
**Impact**: Particle crosses face → new element not in 2×2×2
**Test**: Build face-sharing graph, check spatial proximity

### 5. Level Mismatch
**Issue**: Element registered at one level, searched at another
**Code**: Uses `level_cell_sizes[level]` for each level
**Impact**: Grid alignment mismatch → element not found
**Test**: Verify search levels match registration levels

### 6. Morton Offset/Clipping
**Code**: `mesh_aligned_point_location.py:250-252`
```python
i_offset = jnp.clip(i + octree_gpu.morton_offset,
                    0, octree_gpu.morton_max_coord - 1)
```
**Issue**: Negative coordinates may be clipped incorrectly
**Impact**: Elements near (0,0,0) may have wrong Morton codes
**Test**: Check elements with negative coordinates

---

## Expected Diagnostic Output

### Test 1: Registration Coverage
```
Total elements: 3,048,900
Elements in octree: 3,047,074 (99.94%)
Elements NOT in octree: 1,826 (0.06%)

Cells per element distribution:
  0 cells: 1,826 elements (0.06%)  ← Non-Kuhn
  4 cells: 3,047,074 elements (99.94%)  ← Expected

Statistics:
  Mean cells/element: 4.00
  Median: 4
  Min: 4
  Max: 4
```

### Test 2: 2×2×2 Search Pattern
```
Sample: 10,000 elements

Searchable in 2×2×2: 9,500 (95.0%)
NOT searchable: 500 (5.0%)  ← PROBLEM!
Not in octree: 60 (0.6%)

⚠️ WARNING: 5% of elements NOT in 2×2×2 neighborhood!
```

### Test 3: Cross-Cell Boundaries
```
Interior faces: 5,917,485
Same cell: 3,026,070 (51.14%)
Different cells: 2,884,819 (48.75%)  ← Known issue

⚠️ CRITICAL: 2.88M face-sharing pairs in different cells
```

---

## Next Steps (After Diagnostics Complete)

1. **Analyze results** from `logs/diagnose_multi_cell_coverage.log`

2. **Identify dominant failure mode**:
   - If Test 2 shows high "NOT searchable" → expand to 3×3×3 or 4×4×4
   - If Test 3 shows neighbors too far → pre-compute neighbor tables
   - If Test 1 shows many incomplete → fix registration algorithm

3. **Implement solution**:
   - **Option A**: Expand search to 3×3×3 (27 cells) or 4×4×4 (64 cells)
   - **Option B**: Pre-compute face-neighbor tables (like Option B already exists)
   - **Option C**: Hybrid: 2×2×2 + pre-computed neighbors for boundary cases

4. **Validate**:
   - Run `benchmark_l2_search_methods_with-export.py` with fix
   - Target: >95% retention @ 100 steps
   - Compare with Morton radius=30 baseline

---

## Key Insights from Code Review

### Multi-Cell Registration IS Correct
The vertex registration algorithm correctly:
- Finds all 4 vertices per element
- Computes grid cell for each vertex
- Registers element in all touched cells
- Creates bidirectional CSR mappings

### 2×2×2 Search Pattern IS Implemented
The search correctly:
- Tries multiple levels (14→7)
- Uses exact cell sizes per level
- Searches 8-cell cube centered at particle
- Offsets are [-1,-1,-1] to [0,0,0] (correct centering)

### The Pattern MAY Be Too Small
For Kuhn tets at refinement boundaries:
- Vertices can be at (i, j, k), (i+2, j, k+2), etc.
- Centroid at (i+1, j+0.5, k+1)
- Base cell: (i+1, j, k+1)
- 2×2×2: cells (i, j-1, k) to (i+1, j, k+1)
- Vertex at (i+2, ...): OUTSIDE 2×2×2! ❌

**This is likely the dominant failure mode.**

---

## Running: Diagnostic Script

**Command**:
```bash
python3 diagnose_multi_cell_coverage.py 2>&1 | tee logs/diagnose_multi_cell_coverage.log
```

**Status**: In progress (background task b07a948)

**ETA**: ~3-5 minutes
- Load mesh: 20s
- Extract octree: 150s
- Test 1: 10s
- Test 2: 60s (10k sample)
- Test 3: 60s (face map)

**Output**: Will show exact counts for all 6 failure modes
