# Phase 2: Expanding 2×2×2 to 3×3×3 for Adaptive Mesh Refinement

**Date**: 2026-02-13
**Issue**: 23.52% of elements NOT searchable in 2×2×2 neighborhood
**Root Cause**: Adaptive mesh refinement with 1:2 and 2:1 face neighbors

---

## Current 2×2×2 Pattern Analysis

### Current Offsets (Lines 334-343 in `mesh_aligned_point_location.py`)

```python
cell_offsets = [
    [-1, -1, -1],  # 1
    [-1, -1,  0],  # 2
    [-1,  0, -1],  # 3
    [-1,  0,  0],  # 4
    [ 0, -1, -1],  # 5
    [ 0, -1,  0],  # 6
    [ 0,  0, -1],  # 7
    [ 0,  0,  0],  # 8
]
```

**Coverage**: Cells from `(i-1, j-1, k-1)` to `(i, j, k)` where `(i,j,k) = floor(pos/cell_size)`

**Centered at**: `(i-0.5, j-0.5, k-0.5)` - the "center" of the 2×2×2 cube

---

## Why 2×2×2 Fails with Adaptive Refinement

### Scenario 1: Element Spanning Refinement Boundary (1:2 Face Neighbor)

```
Fine level (level 12, cell_size = 0.01):
┌─────┬─────┬─────┬─────┐
│ F1  │ F2  │ F3  │ F4  │  Fine cells at level 12
└─────┴─────┴─────┴─────┘
      └──────┬──────┘
             │
        ┌────┴────┐
        │   C1    │             Coarse cell at level 11
        └─────────┘             (cell_size = 0.02, 2× larger)

Element E:
  - Vertex v0 in fine cell F1  (i=10, level=12)
  - Vertex v1 in fine cell F2  (i=11, level=12)
  - Vertex v2 in coarse cell C1 (i=5, level=11)  ← 2× grid spacing!
  - Vertex v3 in coarse cell C1 (i=5, level=11)

Particle at centroid:
  - Centroid ≈ (10.5, j, k) in fine grid
  - Base cell: (i=10, j, k) at level 12

2×2×2 search at level 12:
  - Searches cells (9,j-1,k-1) through (10,j,k)
  - Does NOT reach cell (5,j,k) at level 11!
  - Element NOT found! ❌
```

**Problem**: Multi-level elements have vertices at DIFFERENT refinement levels. 2×2×2 search at particle's level misses vertices at other levels.

### Scenario 2: Element Spanning 3 Cells at Same Level

```
Element vertices at same level 12:
  v0 at cell (i, j, k)
  v1 at cell (i+1, j, k)
  v2 at cell (i, j+1, k)
  v3 at cell (i+2, j, k)    ← 2 cells away in i-direction!

Centroid: (i+0.75, j+0.33, k)
Base cell: (i, j, k)

2×2×2 search: cells (i-1,j-1,k-1) to (i,j,k)
  - Covers: (i,j,k), (i-1,j,k), (i,j-1,k), etc.
  - Does NOT reach (i+2,j,k)! ❌
```

**Problem**: Even at same level, Kuhn elements can span > 2 cells in one dimension.

---

## Proposed Solution: 3×3×3 Search Pattern

### New Offsets (27 cells)

```python
cell_offsets = []
for di in [-1, 0, 1]:
    for dj in [-1, 0, 1]:
        for dk in [-1, 0, 1]:
            cell_offsets.append([di, dj, dk])
```

**Coverage**: Cells from `(i-1, j-1, k-1)` to `(i+1, j+1, k+1)`

**Centered at**: `(i, j, k)` - the particle's base cell

**Benefits**:
1. ✅ Covers ±1 cell in each direction from particle
2. ✅ Catches vertices up to 2 cells away (sufficient for Kuhn tets)
3. ✅ Works across refinement levels (searches at ALL levels 14→7)

---

## But What About Multi-Level Elements?

### Critical Implementation Detail

**File**: `mesh_aligned_point_location.py:213-224`

```python
def try_level(level_idx, carry):
    """Try searching 3×3×3 neighborhood at one refinement level."""
    level = 14 - level_idx  # Try levels 14, 13, 12, 11, 10, 9, 8, 7

    cell_size = octree_gpu.level_cell_sizes[level]  # ← Different size per level!

    # Compute base cell indices at THIS level
    i_base = floor(pos.x / cell_size.x)
    j_base = floor(pos.y / cell_size.y)
    k_base = floor(pos.z / cell_size.z)

    # Search 3×3×3 at THIS level
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                search cell (i_base+di, j_base+dj, k_base+dk) AT level
```

**Key insight**: The search **tries multiple levels** (14, 13, 12, ..., 7). For each level:
- Uses that level's cell size
- Computes grid indices at that level
- Searches 3×3×3 neighborhood at that level

**This handles adaptive refinement perfectly!**

### Example with Multi-Level Element

```
Element E with vertices:
  - v0 at cell (10, 5, 3) level 12 (fine)
  - v1 at cell (11, 5, 3) level 12 (fine)
  - v2 at cell (5, 2, 1) level 11 (coarse)  ← Different level!
  - v3 at cell (5, 3, 1) level 11 (coarse)

Element registered in 4 cells:
  - (10,5,3,level=12)
  - (11,5,3,level=12)
  - (5,2,1,level=11)
  - (5,3,1,level=11)

Particle at position (0.105, 0.049, 0.030):

Search at level 12 (cell_size = 0.01):
  i = floor(0.105 / 0.01) = 10
  j = floor(0.049 / 0.01) = 4
  k = floor(0.030 / 0.01) = 3

  3×3×3: cells (9,3,2) to (11,5,4) at level 12
  → Finds cells (10,5,3) and (11,5,3) ✅

Search at level 11 (cell_size = 0.02):
  i = floor(0.105 / 0.02) = 5
  j = floor(0.049 / 0.02) = 2
  k = floor(0.030 / 0.02) = 1

  3×3×3: cells (4,1,0) to (6,3,2) at level 11
  → Finds cells (5,2,1) and (5,3,1) ✅

Element found at level 12 or 11! ✅
```

**Conclusion**: 3×3×3 with multi-level search **automatically handles** 1:2 and 2:1 refinement!

---

## Performance Analysis

### Current 2×2×2

```
Tests per particle: 8 cells × 18.31 elem/cell = 146 tests
Searchable: 76.48%
NOT searchable: 23.52% ❌
```

### Proposed 3×3×3

```
Tests per particle: 27 cells × 18.31 elem/cell = 494 tests
Expected searchable: >95% ✅
Cost increase: 3.38× (146 → 494 tests)
```

**Trade-off**:
- 3.38× more tests
- But fixes 23.52% particle loss
- Net retention improvement: 76% → 95%+ = **25% gain**

### Alternative: 4×4×4 (If 3×3×3 Insufficient)

```
Tests per particle: 64 cells × 18.31 elem/cell = 1,172 tests
Expected searchable: >99% ✅
Cost increase: 8.03× (146 → 1,172 tests)
```

Only needed if 3×3×3 doesn't achieve >95% retention.

---

## Implementation Plan

### Step 1: Update Offset Array

**File**: `mesh_aligned_point_location.py:334-343`

```python
# OLD (2×2×2):
cell_offsets = jnp.array([
    [-1, -1, -1], [-1, -1, 0], ..., [0, 0, 0]
], dtype=jnp.int32)

# NEW (3×3×3):
cell_offsets = jnp.array([
    [di, dj, dk]
    for di in [-1, 0, 1]
    for dj in [-1, 0, 1]
    for dk in [-1, 0, 1]
], dtype=jnp.int32)
```

### Step 2: Update Loop Count

**Line 346**: Change from 8 to 27 cells

```python
# OLD:
level_found_elem, level_tests = jax.lax.fori_loop(
    0, 8,  # ← 2×2×2
    lambda i, c: try_cell(cell_offsets[i], c),
    (found_elem, total_tests)
)

# NEW:
level_found_elem, level_tests = jax.lax.fori_loop(
    0, 27,  # ← 3×3×3
    lambda i, c: try_cell(cell_offsets[i], c),
    (found_elem, total_tests)
)
```

### Step 3: Update Comments

Update documentation to reflect 3×3×3 pattern and reasoning.

### Step 4: Update max_tests

**Line 182**: Increase default max_tests

```python
# OLD:
def search_mesh_aligned_octree_multi_local(
    pos: jax.Array,
    octree_gpu: MeshAlignedOctreeGPU,
    max_tests: jnp.int32 = 200  # ← 2×2×2 needs ~146
)

# NEW:
def search_mesh_aligned_octree_multi_local(
    pos: jax.Array,
    octree_gpu: MeshAlignedOctreeGPU,
    max_tests: jnp.int32 = 600  # ← 3×3×3 needs ~494
)
```

---

## Testing Strategy

### Test 1: Diagnostic Re-run

```bash
python diagnose_multi_cell_coverage.py
```

**Expected output**:
```
Results (sample of 10,000 elements):
  Searchable in 3×3×3 neighborhood: 9,500+ (>95%)
  NOT searchable in 3×3×3: <500 (<5%)
```

### Test 2: Benchmark Re-run

```bash
python benchmark_l2_search_methods_with-export.py
```

**Expected**:
- Retention @ 100 steps: ~95%+ (vs ~80% currently)
- Throughput: ~10-15K p/s (vs ~37K currently, due to 3.38× more tests)

### Test 3: Centroid Comparison

Should still achieve 100% with centroids (verify no regression).

---

## Addressing Your Concern: Adaptive Refinement

**Your concern**: "we have adaptive mesh refinement and we have 1:2 and 2:1 elements face neighboring"

**Answer**: ✅ **Already handled!**

The multi-level search loop (lines 213-358) tries **all levels 14→7**. For each level:
- Computes grid indices using that level's cell size
- Searches in cells at that specific level
- Finds elements registered at that level

This means:
- Fine-level vertex cells found when searching at fine level
- Coarse-level vertex cells found when searching at coarse level
- 3×3×3 at EACH level ensures we catch vertices ±1 cell away

**No special handling needed for 1:2 neighbors** - the multi-level search automatically covers them!

---

## Conclusion

**Phase 2 Fix**: Expand 2×2×2 → 3×3×3

**Changes**: 2 lines of code (offset array + loop count)

**Benefits**:
- ✅ Fixes 23.52% "not searchable" issue
- ✅ Handles adaptive refinement (already in design)
- ✅ Handles 1:2 and 2:1 face neighbors (automatic)
- ✅ Expected retention: 95%+ (vs 76% currently)

**Cost**: 3.38× more tests (146 → 494), acceptable trade-off for 25% retention gain
