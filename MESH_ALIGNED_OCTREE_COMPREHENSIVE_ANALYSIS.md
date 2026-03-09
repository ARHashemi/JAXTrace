# Mesh-Aligned Octree: Comprehensive Analysis

**Date:** 2026-01-26
**Status:** Implementation Complete, Critical Bug Identified
**Searchability:** 17.7% (FAILED - should be ~100%)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Mesh-Aligned Octree Architecture](#mesh-aligned-octree-architecture)
3. [Implementation Details](#implementation-details)
4. [Search Algorithm](#search-algorithm)
5. [Critical Bug Analysis](#critical-bug-analysis)
6. [Morton Code Role](#morton-code-role)
7. [Test Results](#test-results)
8. [Root Cause Analysis](#root-cause-analysis)
9. [Proposed Solutions](#proposed-solutions)
10. [Comparison with Original Design](#comparison-with-original-design)

---

## 1. Executive Summary

### What We Built

A **mesh-aligned octree** for GPU point location that:
- Extracts the intrinsic octree structure from Kuhn tetrahedral meshes
- Assigns each tetrahedral element to its **one parent octree cube**
- Uses Morton codes as spatial hash for O(log n) cell lookup
- Achieves 1.00 cells per element (vs 8 in bbox overlap approach)
- Reduces to 11.5 elements per cell (vs 37.4 in bbox overlap)

### Critical Finding

**The implementation has a fundamental bug:** Element centroids are **NOT inside their assigned parent cubes** in 82.3% of cases.

**Verification Test Results:**
- Placed 1,000 particles at element centroids (guaranteed inside elements)
- Only 177 found (17.7%)
- 823 not found (82.3%)
- **Root cause:** Centroids fall OUTSIDE the parent cube bounds

**This means the parent cube identification algorithm is WRONG.**

---

## 2. Mesh-Aligned Octree Architecture

### 2.1 Kuhn Tetrahedral Meshes

Kuhn meshes decompose octree cubes into tetrahedra with exactly **3 axis-aligned edges**:

```
Cube subdivision patterns:
- 6-tet:  Standard Kuhn decomposition
- 12-tet: One refinement
- 24-tet: Two refinements

Key property: Each tet has 3 edges parallel to x, y, or z axes.
```

From analysis (`logs/investigate_elements_per_cell.log`):
- 11.47 elements per cell on average
- Distribution: 6-tet, 12-tet, and 24-tet patterns
- Levels 8-14 present (7 refinement levels)

### 2.2 Octree Structure

```
Level 8:      151 cells ( 0.06%)  - coarsest
Level 9:      292 cells ( 0.11%)
Level 10:     637 cells ( 0.24%)
Level 11:   1,938 cells ( 0.73%)
Level 12:   5,239 cells ( 1.97%)
Level 13:  38,738 cells (14.59%)
Level 14: 218,603 cells (82.31%)  - finest
───────────────────────────────────
Total:    265,598 cells
```

**Cell sizes** (from `logs/diagnose_cell_sizes.log`):
- Anisotropic: X/Y ratio = 0.9783, Y/Z ratio = 1.0222
- Highly uniform within each level (CV < 0.001%)
- Follow perfect octree subdivision (2× at each level)

Example at level 14:
```
X: 0.0000781250 (mean), std=1e-10
Y: 0.0000798612 (mean), std=2e-10
Z: 0.0000781250 (mean), std=1e-10
```

---

## 3. Implementation Details

### 3.1 Phase 2: CPU Extraction

**File:** `jaxtrace/gpu/search/mesh_aligned_octree_single_cell.py`

**Algorithm:**
```python
for each tetrahedral element:
    1. Find axis-aligned edges → infer cell_size and refinement level
    2. Find parent cube:
         cube_corner = floor(min_vertex / cell_size) * cell_size
         grid_indices (i, j, k) = floor(min_vertex / cell_size)
    3. Encode to Morton:
         morton = encode_morton_3d(i + offset, j + offset, k + offset)
    4. Store cell:
         cell_key = (morton, level)  # Both needed to prevent collisions!
    5. Build mapping:
         cell → list of element IDs (CSR format)
         element → cell index
```

**Key Functions:**

1. **`find_axis_aligned_edges_single()`** (lines 69-104):
   - Detects 3 axis-aligned edges from 6 edges
   - Computes cell size from edge lengths
   - Infers refinement level from Z-axis edge length

2. **`find_parent_cube()`** (lines 107-150):
   - **CRITICAL:** `cube_corner = floor(v_min / cell_size) * cell_size`
   - Returns corner, center, and grid indices

3. **`encode_morton_3d_single()`** (lines 59-66):
   - Bit interleaving of (i, j, k) with max_depth=21
   - Creates 63-bit Morton code (21 bits per dimension)

**Data Structure:**
```python
@dataclass
class OctreeCellData:
    cell_morton_codes: np.ndarray      # (n_cells,) uint64
    cell_levels: np.ndarray            # (n_cells,) uint8
    cell_sizes: np.ndarray             # (n_cells, 3) float64
    cell_grid_indices: np.ndarray      # (n_cells, 3) int32

    cell_to_elements_offsets: np.ndarray  # (n_cells + 1,) int32 - CSR
    cell_to_elements_data: np.ndarray     # (total_entries,) int32 - CSR

    element_to_cells: np.ndarray          # (n_elements,) int32
```

### 3.2 Phase 3: GPU Upload

**File:** `jaxtrace/gpu/search/mesh_aligned_octree_gpu.py`

**Enhancements Added:**
- `level_cell_sizes`: (max_level+1, 3) array storing representative cell size per level
- Computed by taking **first cell** at each level (all nearly identical within level)
- Prevents floating-point precision issues from derived formulas

**GPU Structure:**
```python
@dataclass
class MeshAlignedOctreeGPU:
    # Core mesh
    connectivity: jax.Array            # (n_elements, 4) int32
    node_positions: jax.Array          # (n_nodes, 3) float32

    # Cell structure (sorted by morton)
    cell_morton_codes: jax.Array       # (n_cells,) uint64
    cell_levels: jax.Array             # (n_cells,) uint8
    cell_sizes: jax.Array              # (n_cells, 3) float32
    cell_grid_indices: jax.Array       # (n_cells, 3) int32

    # CSR mapping
    cell_to_elements_offsets: jax.Array  # (n_cells + 1,) int32
    cell_to_elements_data: jax.Array     # (total_entries,) int32

    # Level-specific lookups
    level_cell_sizes: jax.Array        # (max_level + 1, 3) float32

    # Morton parameters
    morton_offset: jnp.int32           # 2^19 for negative coords
    morton_max_coord: jnp.int32        # 2^20 max
    max_depth: jnp.int32               # 21 bits per dimension
```

**Memory usage:** 74.1 MB for 265k cells, 3M elements

### 3.3 Phase 4: GPU Point Location

**File:** `jaxtrace/gpu/search/mesh_aligned_point_location.py`

**Algorithm:**
```python
def search_mesh_aligned_octree_single(pos, octree_gpu, max_tests):
    for level in [14, 13, 12, 11, 10, 9, 8, 7]:
        # Get exact cell size for this level
        cell_size = octree_gpu.level_cell_sizes[level]

        # Compute grid indices
        i = floor(pos[0] / cell_size[0])
        j = floor(pos[1] / cell_size[1])
        k = floor(pos[2] / cell_size[2])

        # Apply offset and encode
        i_offset = clip(i + morton_offset, 0, max_coord - 1)
        j_offset = clip(j + morton_offset, 0, max_coord - 1)
        k_offset = clip(k + morton_offset, 0, max_coord - 1)
        morton = encode_morton_3d_jax(i_offset, j_offset, k_offset)

        # Binary search for (morton, level)
        cell_idx = find_cell_by_morton_and_level(morton, level, ...)

        if cell_idx >= 0:
            # Test elements in this cell
            for elem_id in cell_elements[cell_idx]:
                if point_in_tet(pos, elem_id):
                    return elem_id

    return -1  # Not found
```

**Binary Search:**
- Searches sorted array of `(morton, level)` pairs
- Lexicographic comparison: first by morton, then by level
- Uses `lax.fori_loop` with static max_iters=25 for JAX compatibility
- O(log n) complexity

---

## 4. Search Algorithm

### 4.1 Current Implementation

**Multi-level Grid Lookup:**

```
Query position (x, y, z)
│
├─ Level 14: Compute grid(x,y,z) at finest level
│    └─ Binary search for cell(morton_14, 14)
│    └─ Test all elements in cell
│    └─ If found: RETURN
│
├─ Level 13: Compute grid(x,y,z) at next level
│    └─ Binary search for cell(morton_13, 13)
│    └─ Test all elements in cell
│    └─ If found: RETURN
│
├─ ... (continue for levels 12, 11, 10, 9, 8, 7)
│
└─ Not found: RETURN -1
```

**Assumptions:**
1. For any position, cells may exist at multiple refinement levels
2. Need to check all levels to find the containing element
3. Grid indices computed independently at each level

### 4.2 Original Design Intent

From user description:
> "Build a mesh aligned octree, then have the search limited to lowest octant/cube containing the query position, with fallbacks to sibling and parent cubes."

**Hierarchical Octree Search:**

```
Query position (x, y, z)
│
├─ Determine actual refinement level at position
│   (finest level where mesh has geometry)
│
├─ Search cell at that specific level
│    └─ If found: RETURN
│
├─ Fallback to 26 sibling cells (neighbors at same level)
│    └─ If found: RETURN
│
├─ Fallback to parent cell (one level coarser)
│    └─ Search parent's elements
│    └─ If found: RETURN
│
├─ Fallback to parent's neighbors
│    └─ If found: RETURN
│
└─ Continue up hierarchy to coarser levels
```

### 4.3 Key Differences

| Aspect | Current Implementation | Original Intent |
|--------|------------------------|-----------------|
| **Level selection** | Try all levels 14→7 | Determine actual level at position |
| **Spatial fallback** | None | Check 26 neighbors + parent |
| **Grid computation** | Independent per level | Based on local refinement |
| **Search scope** | 1 cell per level (8 total) | 1 cell + neighbors + hierarchy |
| **Efficiency** | ~4.6 tests per particle | Unknown (likely higher) |

---

## 5. Critical Bug Analysis

### 5.1 Verification Test Results

**Test:** Place particles at element centroids (guaranteed inside elements)

**File:** `logs/verify_search_correctness.log`

**Results:**
```
Total particles: 1,000
Found correct element: 177 (17.7%)
Found wrong element: 0 (0.0%)
Not found at all: 823 (82.3%)

❌ FAILED: 82.3% of particles not found
```

### 5.2 Detailed Analysis of Failures

**Example 1: Particle 0 (Element 2232962)**
```
Position (centroid): [0.00855469, -0.00307466, -0.00054688]

Element's cell:
  Level: 13
  Grid: [54, -21, -4]
  Cell size: [0.00015625, 0.00015972, 0.00015625]

Query computes:
  Grid: [54, -20, -4]  ← Y index differs by 1!

Cube bounds: [[0.00843749, -0.00335417, -0.000625],
              [0.00859374, -0.00319445, -0.00046875]]
Centroid: [0.00855469, -0.00307466, -0.00054688]

Is centroid inside cube? FALSE
  - Y coordinate: -0.00307466 is OUTSIDE [-0.00335417, -0.00319445]
  - Z coordinate: -0.00054688 is OUTSIDE [-0.000625, -0.00046875]
```

**Example 2: Particle 1 (Element 2115658)**
```
Position (centroid): [0.00298828, -0.00692795, -0.00019531]

Element's cell:
  Level: 14
  Grid: [38, -88, -3]

Query computes:
  Grid: [38, -87, -3]  ← Y index differs by 1!

Cube bounds: [[0.00296875, -0.00702776, -0.00023437],
              [0.00304687, -0.00694790, -0.00015625]]
Centroid: [0.00298828, -0.00692795, -0.00019531]

Is centroid inside cube? FALSE
  - Y coordinate: -0.00692795 is OUTSIDE [-0.00702776, -0.00694790]
```

### 5.3 Pattern Analysis

**Common failure mode:** Element centroids fall **outside** their assigned parent cube bounds.

**Grid index mismatches:**
- Y coordinate consistently off by ±1
- Also seen in X and Z coordinates
- Even when cell sizes match exactly between assignment and query

**This indicates the parent cube identification is fundamentally flawed.**

---

## 6. Morton Code Role

### 6.1 What Morton Codes DO in This Implementation

**Purpose:** Spatial hash function for fast cell lookup

```
3D Grid Position        Morton Encoding           Binary Search
────────────────   →   ─────────────────   →     ────────────
(i, j, k, level)       morton_code               Find cell index
                       uint64

Example:
(54, -21, -4, 13)  →   267642492140903770   →    cell_idx = 15234
```

**Algorithm:**
1. **Bit interleaving:** Interleave bits of i, j, k to create space-filling curve index
2. **Spatial locality:** Nearby cells have similar Morton codes
3. **Fast lookup:** Binary search in sorted array is O(log n)

**Key properties:**
- Morton code alone is NOT unique (different levels can have same grid position)
- We store `(morton, level)` tuples and search with **lexicographic comparison**
- Max depth = 21 bits per dimension = 63-bit total

### 6.2 What Morton Codes DON'T Do

**NOT used for:**
- ❌ Space-filling curve traversal
- ❌ Ray marching along curve
- ❌ Fixed-length curve segments
- ❌ Hierarchical tree structure encoding
- ❌ Neighbor finding (would need bit manipulation)

### 6.3 Comparison with Previous Morton Implementation

| Aspect | Old Morton Blocks | New Mesh-Aligned |
|--------|-------------------|------------------|
| **Purpose** | Arbitrary space partition | Hash function for grid cells |
| **Block size** | Fixed (2^20 bits) | Variable (levels 8-14) |
| **Curve usage** | Ray traversal | Just lookup |
| **Elements per block** | ~536 | ~11.5 |
| **Alignment** | None | Mesh-intrinsic octree |
| **Searchability** | 93-98% | Currently 17.7% (broken) |

### 6.4 Answer to Your Question

> "Can we divide Morton into fixed-length segments that each corresponds to a lowest level cube?"

**No**, because:
1. **Variable refinement:** Different regions have different "lowest level"
   - Region A: finest level is 14
   - Region B: finest level is 10
   - No uniform "segment length"

2. **Morton doesn't encode level:** The code `267642492140903770` could be:
   - Level 8 cell at grid (534, 123, 45)
   - Level 14 cell at grid (12, 456, 789)
   - Need the level to disambiguate

3. **Hierarchical structure:** Parent-child relationships require bit shifts:
   ```
   Child at level 14: grid (54, -21, -4)
   Parent at level 13: grid (27, -11, -2)  ← divide by 2
   ```
   But Morton codes don't have simple arithmetic relationships due to bit interleaving.

**Better approach:** Store `(morton, level)` pairs as we currently do.

---

## 7. Test Results

### 7.1 Phase 2 Extraction (CPU)

**File:** `logs/test_single_cube_extraction.log`

```
Results:
  ✅ Unique cells: 265,598
  ✅ Cells per element: 1.00 (perfect single-cube assignment)
  ✅ Elements per cell: 11.47 (down from 37.4 in bbox overlap)
  ✅ Extraction time: 145.66s

Conclusion: Phase 2 works correctly for single-cube extraction.
```

### 7.2 Grid Index Consistency (CPU)

**File:** `logs/test_grid_index_consistency.log`

```
Test: For 70 random elements, check if centroids compute correct grid indices.

Results:
  Total tested: 70
  Matches: 10 (14.3%)
  Mismatches: 60 (85.7%)

❌ Grid index computation is INCONSISTENT!
   85.7% mismatch rate

Pattern: Y coordinate off by ±1 most frequently
```

### 7.3 GPU Search (v3, v4, v5)

**File:** `logs/test_mesh_aligned_octree_gpu_v5_fixed.log`

```
Test: 10,000 random particles in bbox

Results:
  Searchability: 35.9%
  Mean tests: 4.6
  Median tests: 5
  Max tests: 48

Interpretation:
  - Only finding particles in regions where grid computation happens to work
  - ~2/3 of bbox is void (no tetrahedra)
  - Of the ~1/3 with geometry, finding ~100% where grid aligns
  - Total: ~35% of bbox = ~100% of searchable regions with correct grid
```

### 7.4 Centroid Verification (Final)

**File:** `logs/verify_search_correctness.log`

```
Test: 1,000 particles at element centroids (ground truth)

Results:
  Found correct element: 177 (17.7%)
  Not found: 823 (82.3%)

❌ CRITICAL FAILURE
   Element centroids are OUTSIDE their assigned parent cubes!

Root cause: Parent cube identification algorithm is wrong.
```

---

## 8. Root Cause Analysis

### 8.1 The Parent Cube Algorithm

**Current implementation** (`find_parent_cube()` in mesh_aligned_octree_single_cell.py:107-150):

```python
def find_parent_cube(vertices, cell_size, tolerance=1e-6):
    v_min = vertices.min(axis=0)

    # Compute grid indices
    i = int(np.floor(v_min[0] / cell_size[0]))
    j = int(np.floor(v_min[1] / cell_size[1]))
    k = int(np.floor(v_min[2] / cell_size[2]))

    # Cube corner
    cube_corner = np.array([
        i * cell_size[0],
        j * cell_size[1],
        k * cell_size[2]
    ])

    return cube_corner, ..., i, j, k
```

**Assumption:** The cube starting at `floor(v_min / cell_size) * cell_size` contains the tetrahedron.

### 8.2 Why This Is Wrong

**Counterexample from log:**

Element 2232962:
- Vertices min: approximately [0.00843749, -0.00335417, -0.000625]
- Cell size: [0.00015625, 0.00015972, 0.00015625]
- Computed grid: [54, -21, -4]
- Cube bounds: [[0.00843749, -0.00335417, -0.000625],
                 [0.00859374, -0.00319445, -0.00046875]]
- Centroid: [0.00855469, -0.00307466, -0.00054688]

**Problem:** Centroid Y = -0.00307466 is **above** cube max Y = -0.00319445

**Why:** Kuhn tetrahedra can extend **beyond** the grid cell computed from their minimum vertex!

### 8.3 Kuhn Tetrahedron Geometry

Kuhn decomposition creates tetrahedra that:
- Have 3 axis-aligned edges
- Can span multiple grid cells
- Are NOT confined to a single cube

**Example:** A tet with vertices at cube corners can have:
- Min vertex at (0, 0, 0)
- Max vertex at (dx, dy, dz)
- But body extends diagonally through space
- Centroid at (dx/4, dy/4, dz/4) may be in a DIFFERENT grid cell than v_min!

### 8.4 The Fundamental Flaw

**We assumed:** Each Kuhn tet fits in one parent cube.

**Reality:** Kuhn tets are created by subdividing cubes, but the relationship is:
- **One cube → 6, 12, or 24 tets**
- **NOT: One tet → One cube**

A tet can span multiple cubes at its refinement level!

### 8.5 Why investigate_elements_per_cell.log Was Misleading

That log showed:
```
✅ All 1000 sampled elements correctly assigned to parent cubes
✅ All elements in a cell share identical grid indices
```

**But it only checked:**
- IF elements in the SAME cell have the SAME grid indices (true, by construction)
- NOT if the assigned cell actually CONTAINS the elements (false!)

---

## 9. Proposed Solutions

### 9.1 Solution 1: Correct Parent Cube Identification

**Approach:** Find the cube that actually contains the tetrahedron.

**Algorithm:**
```python
def find_parent_cube_correct(vertices, cell_size):
    # Compute bounding box of tet
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)

    # Find all grid cells overlapped by bbox
    i_min = int(np.floor(v_min[0] / cell_size[0]))
    i_max = int(np.floor(v_max[0] / cell_size[0]))
    j_min = int(np.floor(v_min[1] / cell_size[1]))
    j_max = int(np.floor(v_max[1] / cell_size[1]))
    k_min = int(np.floor(v_min[2] / cell_size[2]))
    k_max = int(np.floor(v_max[2] / cell_size[2]))

    # Find the "most representative" cell:
    # Option A: Cell containing centroid
    centroid = vertices.mean(axis=0)
    i = int(np.floor(centroid[0] / cell_size[0]))
    j = int(np.floor(centroid[1] / cell_size[1]))
    k = int(np.floor(centroid[2] / cell_size[2]))

    # Option B: Cell containing v_min (current, known broken)

    # Option C: Store tet in ALL overlapped cells (multi-insert)
    # This is the v1 "bbox overlap" approach we rejected

    return grid_indices
```

**Pros:**
- Centroid-based: Guarantees centroid is in assigned cell
- Physically meaningful (centroid represents tet location)

**Cons:**
- May not match mesh generation algorithm
- Still only one cell per tet (misses tets spanning multiple cells)

### 9.2 Solution 2: Multi-Insert (Bbox Overlap)

**Approach:** Store each tet in ALL grid cells it overlaps.

This is the **v1 implementation** we previously rejected, but it actually might have been closer to correct!

**Algorithm:**
```python
# Compute bbox
v_min = vertices.min(axis=0)
v_max = vertices.max(axis=0)

# Find all overlapped cells
for i in range(i_min, i_max + 1):
    for j in range(j_min, j_max + 1):
        for k in range(k_min, k_max + 1):
            cell_key = (morton(i,j,k), level)
            cell_to_elements[cell_key].append(elem_id)
```

**Pros:**
- Guarantees 100% searchability (any query in tet will find it)
- Matches spatial reality (tets DO span multiple cells)

**Cons:**
- Higher memory usage (265k → ~652k cells from v1 logs)
- More elements per cell (11.5 → ~37 from v1 logs)
- More point-in-tet tests per query

**Re-evaluation of v1:**
- v1 had 2.4% searchability, but that was likely due to OTHER bugs (morton without level, wrong base sizes)
- v1's multi-insert strategy might actually be CORRECT for this mesh!

### 9.3 Solution 3: Hierarchical Search with Neighbors

**Approach:** Implement true hierarchical octree search as originally intended.

**Algorithm:**
```python
def search_hierarchical(pos, octree):
    # Start at finest level present in mesh
    level = determine_local_refinement_level(pos)

    # Compute grid cell at this level
    (i, j, k) = compute_grid(pos, level)

    # Search primary cell
    if search_cell(i, j, k, level):
        return found

    # Search 26 neighbors at same level
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                if search_cell(i+di, j+dj, k+dk, level):
                    return found

    # Go to parent level
    level -= 1
    (i, j, k) = (i//2, j//2, k//2)

    # Repeat...
```

**Pros:**
- Matches original design intent
- Handles tets spanning cell boundaries via neighbor checks
- Maintains single-insert (1 cell per tet)

**Cons:**
- Complex implementation
- More tests per query (1 + 26 neighbors + parents = potentially many)
- Need to determine local refinement level (non-trivial)

### 9.4 Solution 4: Hybrid Approach

**Approach:** Single-insert with small spatial tolerance.

**Algorithm:**
```python
def find_parent_cube_robust(vertices, cell_size, tolerance=1e-6):
    centroid = vertices.mean(axis=0)

    # Compute cell from centroid
    i = int(np.floor(centroid[0] / cell_size[0]))
    j = int(np.floor(centroid[1] / cell_size[1]))
    k = int(np.floor(centroid[2] / cell_size[2]))

    # Verify centroid is actually in this cell
    cube_corner = np.array([i, j, k]) * cell_size
    cube_max = cube_corner + cell_size

    if not (np.all(centroid >= cube_corner - tolerance) and
            np.all(centroid <= cube_max + tolerance)):
        # Fallback: try nearby cells
        # Or: insert into multiple cells
        pass

    return i, j, k
```

**During query:** Add small search radius:
```python
# Try primary cell AND immediate neighbors
for di in [-1, 0, 1]:
    for dj in [-1, 0, 1]:
        for dk in [-1, 0, 1]:
            cell_key = (morton(i+di, j+dj, k+dk), level)
            # search...
```

**Pros:**
- Fixes the immediate bug (centroids outside cubes)
- Minimal changes to existing code
- Modest increase in tests (1 → 27 cells)

**Cons:**
- Not the "pure" single-cell approach we aimed for
- Still may miss some tets

---

## 10. Comparison with Original Design

### 10.1 Your Original Specification

From user messages:
> "Build a mesh aligned octree, then have the search limited to lowest octant/cube containing the query position, with fallbacks to sibling and parent cubes."

**Key requirements:**
1. ✅ **Mesh-aligned:** Extract intrinsic octree from mesh
2. ✅ **Lowest octant:** Start search at finest level
3. ❌ **Fallback to siblings:** NOT implemented
4. ❌ **Fallback to parents:** Partially (we try coarser levels, but not as parent-child relationships)

### 10.2 What We Implemented vs What Was Intended

| Feature | Intended | Implemented | Status |
|---------|----------|-------------|--------|
| Extract mesh octree | ✅ | ✅ | Working |
| Single cube per element | ✅ | ❌ | Broken (centroids outside cubes) |
| Start at local refinement level | ✅ | ❌ | Try all levels instead |
| Check sibling cells | ✅ | ❌ | Only check one cell per level |
| Hierarchical parent fallback | ✅ | ❌ | Independent level checks |
| Morton for fast lookup | ✅ | ✅ | Working |

### 10.3 Morton Code Clarification

**Your question:** "How does Morton play a role?"

**Answer:** Morton is a **spatial hash**, not a space-filling curve for traversal.

**Usage:**
- **Assignment:** Element → parent cube grid (i,j,k) → morton hash
- **Query:** Position → grid (i,j,k) at level L → morton hash → binary search

**NOT used for:**
- Curve segments
- Ray marching
- Hierarchical traversal

**The mesh-aligned octree structure IS included,** but encoded as:
- Array of `(morton, level)` pairs (sorted)
- Array of cell sizes (one per cell)
- Not as a tree data structure

---

## 11. Recommendations

### 11.1 Immediate Fix (Recommended)

**Implement Solution 4: Centroid-based assignment + 27-cell search**

**Phase 2 change:**
```python
# Use centroid instead of v_min
centroid = vertices.mean(axis=0)
i = int(np.floor(centroid[0] / cell_size[0]))
j = int(np.floor(centroid[1] / cell_size[1]))
k = int(np.floor(centroid[2] / cell_size[2]))
```

**Phase 4 change:**
```python
# Search primary cell + 26 neighbors
for di in [-1, 0, 1]:
    for dj in [-1, 0, 1]:
        for dk in [-1, 0, 1]:
            cell_key = (morton(i+di, j+dj, k+dk), level)
            if search_cell(cell_key):
                return found
```

**Expected outcome:**
- Searchability: ~100% for centroids (verification test)
- Searchability: ~33-50% for random particles (bbox coverage)
- Mean tests: ~11.5 × 27 = ~310 per particle (high, but correct)

### 11.2 Optimal Solution (Requires More Work)

**Re-implement v1 with fixes: Multi-insert bbox overlap**

**Why:** Kuhn tets physically span multiple grid cells. Multi-insert reflects this reality.

**Changes needed:**
1. Keep v1's bbox overlap logic
2. Add `(morton, level)` tuple keys (fixed in v3)
3. Use exact `level_cell_sizes` (fixed in v5)

**Expected outcome:**
- Searchability: ~100%
- ~652k cells (vs 265k single-insert)
- ~37 elements per cell (vs 11.5)
- Mean tests: ~37-50 per particle

**Comparison with current broken state:**
- v1: 2.4% searchability (due to other bugs, NOT multi-insert)
- v5: 17.7% searchability (due to centroid-outside-cube bug)
- v1 + v3 fixes + v5 fixes: Likely ~100%

### 11.3 Long-term: True Hierarchical Search

**Implement true octree with parent-child pointers**

**Benefits:**
- Elegant hierarchical fallback
- Efficient neighbor finding
- Matches your original design intent

**Challenges:**
- Complex implementation (tree structure on GPU)
- Neighbor finding requires bit manipulation or explicit storage
- Determining local refinement level non-trivial

---

## 12. Conclusion

### 12.1 Summary of Findings

1. **Architecture:** Mesh-aligned octree extraction (Phase 2) is conceptually sound
2. **Critical Bug:** Parent cube identification is **fundamentally broken**
   - Element centroids fall outside assigned cubes in 82.3% of cases
   - Root cause: Using v_min instead of centroid, or misunderstanding Kuhn geometry
3. **Search Algorithm:** Multi-level grid lookup works, but misses spatial neighbors
4. **Morton Codes:** Used correctly as spatial hash, NOT as curve segments
5. **Test Results:** 17.7% searchability for ground-truth centroids (should be 100%)

### 12.2 Path Forward

**Immediate action required:**
- Fix parent cube identification (use centroid)
- Add spatial neighbor search (27-cell cube)
- Re-run verification test to confirm ~100% for centroids

**Consider:**
- Reverting to multi-insert (v1 strategy) with all fixes applied
- May be more robust than single-insert for this mesh type

**Future work:**
- Implement true hierarchical octree search
- Optimize neighbor finding
- Profile performance vs Morton blocks baseline

---

## Appendix A: File Reference

| File | Purpose | Status |
|------|---------|--------|
| `mesh_aligned_octree_single_cell.py` | Phase 2: CPU extraction | ❌ Bug in `find_parent_cube()` |
| `mesh_aligned_octree_gpu.py` | Phase 3: GPU upload | ✅ Working |
| `mesh_aligned_point_location.py` | Phase 4: GPU search | ⚠️ Working but incomplete (no neighbors) |
| `test_mesh_aligned_octree_gpu_v3.py` | Integration test | ✅ Test script works |
| `verify_search_correctness.py` | Centroid verification | ✅ Revealed critical bug |
| `diagnose_cell_sizes.py` | Cell size analysis | ✅ Confirmed anisotropy |
| `test_grid_index_consistency.py` | Grid computation test | ✅ Revealed 85.7% mismatch |

---

## Appendix B: Key Log Files

- `test_mesh_aligned_octree_gpu_v5_fixed.log`: 35.9% searchability (random particles)
- `verify_search_correctness.log`: 17.7% searchability (centroids) ← **CRITICAL**
- `test_grid_index_consistency.log`: 85.7% grid mismatch
- `diagnose_cell_sizes.log`: Cell size uniformity analysis
- `investigate_elements_per_cell.log`: 11.47 elements/cell distribution

---

**Document version:** 1.0
**Last updated:** 2026-01-26
**Next steps:** Implement centroid-based parent cube identification + 27-cell neighbor search
