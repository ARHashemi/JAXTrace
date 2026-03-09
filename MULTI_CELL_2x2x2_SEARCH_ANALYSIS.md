# Multi-Cell 2×2×2 Local Search Analysis

**Date**: 2026-02-13
**Method**: `mesh_aligned_octree_multi_local`
**Status**: Investigating particle loss in tetrahedral voids

---

## User Observation

Visualizations of particle distributions at timestep 100 show **tetrahedral-shaped empty regions** across ALL search methods:
- `mesh_aligned_octree_multi_local` (2×2×2 local search)
- `mesh_aligned_octree` (direct single-cell)
- `radius=30` (Morton with large radius)

**Key insight**: The voids have tetrahedral shapes, suggesting they correspond to **mesh elements that are not covered** by the search structures.

---

## Method Architecture

### Phase 1: Multi-Cell Vertex Registration

**File**: `jaxtrace/gpu/search/mesh_aligned_octree_vertex_multi.py`

**Algorithm**:
```python
for each element:
    find axis-aligned edges → cell_size, level
    for each of 4 vertices:
        grid_cell = floor(vertex / cell_size)
        register element in grid_cell
```

**Expected behavior**:
- Kuhn tetrahedra have vertices at cube corners
- Each element should be registered in ~4 cells (where its vertices are)
- Total registrations: ~12M (4× the 3M elements)

**Data structure**:
```
OctreeCellDataVertexMulti:
    cell_to_elements: CSR (cell → list of elements)
    element_to_cells: CSR (element → list of cells)  # NEW
```

### Phase 2: 2×2×2 Local Search

**File**: `jaxtrace/gpu/search/mesh_aligned_point_location.py`
**Function**: `search_mesh_aligned_octree_multi_local()`

**Algorithm**:
```python
for level in [14, 13, 12, ..., 7]:  # Finest to coarsest
    cell_size = level_cell_sizes[level]
    i_base = floor(pos.x / cell_size.x)
    j_base = floor(pos.y / cell_size.y)
    k_base = floor(pos.z / cell_size.z)

    # Search 8 cells in 2×2×2 cube CENTERED at particle
    # Offsets: [-1,-1,-1] to [0,0,0]
    for (di, dj, dk) in [(-1,-1,-1), (-1,-1,0), ..., (0,0,0)]:
        cell = (i_base+di, j_base+dj, k_base+dk)
        search elements in cell
        if found: return element
```

**Rationale**:
- Particle at position (x,y,z) is in base cell (i,j,k) = floor(pos/cell_size)
- Element vertices can be in cells (i-1,j-1,k-1) through (i,j,k)
- 2×2×2 cube centered at (i-0.5, j-0.5, k-0.5) covers all vertex locations
- Total cells searched: 8
- Total tests: ~146 (8 cells × 18.31 elements/cell)

---

## Possible Sources of Particle Loss

### Source 1: Elements Not Registered in Multi-Cell Octree

**From previous diagnostics** (`octree_retention_diagnostics.log`):
```
Uncovered elements: 1,826 (0.0599%)
- Non-Kuhn elements at refinement boundaries
- Missing axis-aligned edges
```

**Impact**: Particles in these 1,826 elements **cannot be found** by any search method.

**Test**: Check if multi-cell registration still misses these 1,826 elements.

### Source 2: Incomplete Vertex Registration (< 4 Cells per Element)

**Expected**: Each Kuhn tet should be in 4 cells (one per vertex).

**Potential issue**: If some elements are only registered in 1-3 cells, particles near unregistered vertices will be lost.

**Test**:
- Count elements with 1, 2, 3, 4 cells
- Identify elements with < 4 cells
- Check if these correspond to tetrahedral voids

### Source 3: 2×2×2 Search Pattern Doesn't Cover Element

**Scenario**:
```
Element vertices at: (i₀,j₀,k₀), (i₁,j₁,k₁), (i₂,j₂,k₂), (i₃,j₃,k₃)
Particle centroid at: (x,y,z) → base cell (i_p, j_p, k_p)

2×2×2 search covers cells:
    (i_p-1, j_p-1, k_p-1) to (i_p, j_p, k_p)

If element vertex cells are OUTSIDE this range:
    e.g., vertex at (i_p-2, j_p, k_p)
→ Element NOT found!
```

**Cause**: Element spans > 2 cells in any dimension.

**Test**:
- For each element, find centroid cell
- Check if ALL vertex cells are in 2×2×2 neighborhood
- Count elements where some vertices are outside

### Source 4: Cross-Cell Face Sharing Not Covered

**Scenario**:
```
Elements A and B share a face
A registered in cells: {(i,j,k), (i+1,j,k), ...}
B registered in cells: {(i+2,j,k), (i+3,j,k), ...}

Particle crosses from A to B across shared face
Particle now at position → base cell (i+1, j+1, k)
2×2×2 search: cells (i, j, k) to (i+1, j+1, k+1)

Element B's cells: (i+2,j,k), ... → OUTSIDE 2×2×2!
→ Particle lost!
```

**Test**:
- Find face-sharing element pairs
- Check if both elements are in each other's 2×2×2 neighborhoods
- Count cases where face-sharing neighbors are too far apart

### Source 5: Level Mismatch

**Scenario**:
```
Element registered at level 12 (cell size = 0.01)
Particle in element at position where local cell size is level 10 (cell size = 0.04)

Search tries level 14, 13, 12, ... at particle position
At level 12: computes cell based on 0.01 grid
Element IS registered at level 12, but at DIFFERENT grid cell
→ Not found!
```

**Test**:
- Check if element registration level matches search level
- Verify cell size consistency

### Source 6: Morton Offset/Clipping Issues

**Code**:
```python
i_offset = jnp.clip(i + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)
```

**Potential issue**: Negative coordinates being clipped incorrectly.

**Test**: Check elements near (0,0,0) and negative coordinates.

---

## Diagnostic Plan

**Script**: `diagnose_multi_cell_coverage.py`

### Test 1: Multi-Cell Registration Coverage
- Count elements with 0, 1, 2, 3, 4, 5+ cells
- Identify incomplete registrations
- Find completely missing elements

### Test 2: 2×2×2 Search Pattern Analysis
- For each element:
  - Compute centroid cell
  - Check if element is in ANY of 8 cells in 2×2×2 neighborhood
  - Identify elements OUTSIDE 2×2×2 pattern

### Test 3: Cross-Cell Boundary Face Sharing
- Build face-sharing graph
- Count face-sharing pairs in same cell vs different cells
- Check if cross-cell pairs are in each other's neighborhoods

### Test 4: Level-Specific Analysis (Future)
- Per-level coverage statistics
- Level mismatch detection

### Test 5: Spatial Distribution of Uncovered Elements (Future)
- Plot locations of uncovered elements
- Overlay with tetrahedral void visualizations
- Verify correlation

---

## Expected Findings

Based on previous diagnostics:

1. **1,826 elements completely missing** from octree
   - These are the non-Kuhn transition elements
   - Will appear as tetrahedral voids

2. **Some elements with < 4 cells** due to boundary effects
   - Elements at mesh boundaries may have fewer vertices registered
   - Could cause particle loss

3. **~48% of face-sharing pairs cross cell boundaries**
   - From previous: 2,884,819 / 5,917,485 faces
   - 2×2×2 search may not cover all neighbors

4. **Elements spanning > 2 cells in any dimension**
   - Kuhn tets at refinement boundaries
   - Vertices may be > 2 cells apart

---

## Next Steps

1. **Run diagnostics** and analyze results
2. **Identify dominant failure mode**:
   - Missing elements (1,826)?
   - Incomplete registration?
   - 2×2×2 pattern insufficient?
   - Cross-cell neighbors too far?

3. **Propose solution**:
   - Increase search neighborhood (3×3×3 or 4×4×4)?
   - Fix registration for missing 1,826 elements?
   - Pre-compute neighbor tables instead of spatial search?

4. **Validate fix** with particle tracking test

---

## Current Retention Rates

From benchmark results:

| Method | Description | Retention @ 100 steps |
|--------|-------------|----------------------|
| Direct single-cell | Center cell only | ~74-80% |
| Multi-cell 2×2×2 | 8-cell local search | ~80-85% |
| Radius 30 (Morton) | Large radius | ~85-90% |

**Target**: >95% retention (matching global Morton with large radius)

**Gap**: Current 2×2×2 search loses 10-20% of particles due to:
- Missing elements (0.06%)
- Incomplete spatial coverage (9.94-19.94%)

---

## References

- `ELEMENT_SPANNING_PROBLEM_ANALYSIS.md`: Root cause analysis
- `Fix_Mesh-Aligned_Octree_Retention_Loss_Multi-Cell_Vertex_Registration.md`: Multi-cell solution design
- `logs/octree_retention_diagnostics.log`: Single-cell coverage analysis
- `mesh_aligned_octree_vertex_multi.py`: Multi-cell implementation
- `mesh_aligned_point_location.py`: 2×2×2 search implementation
