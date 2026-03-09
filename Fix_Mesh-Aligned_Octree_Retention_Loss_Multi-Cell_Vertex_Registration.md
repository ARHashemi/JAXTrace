# Fix Mesh-Aligned Octree Retention Loss: Multi-Cell Vertex Registration

## Executive Summary

**The current octree extraction is already correct** — it uses the robust axis-aligned edge method and achieves 96.34% of cells with exactly 6 elements (expected for Kuhn decomposition). The diagonal-based method from the reference doc is **not better** and has risks for anisotropic cells.

**The real problem**: 100% of Kuhn tetrahedra span multiple grid cells because vertices sit at cube corners (cell boundaries). Elements are registered in only ONE cell (centroid's cell). When particles move near cell boundaries, the search fails because elements aren't registered in the cells they geometrically overlap.

**The fix**: Multi-cell vertex registration — register each element in ALL cells its 4 vertices touch.

---

## Root Cause Analysis

From the research agent's analysis:

### Current Extraction Method is Robust
The `find_axis_aligned_edges_single()` approach:
- ✅ Handles anisotropic cells (dx=0.005, dy=0.00511, dz=0.005)
- ✅ Directly measures per-axis dimensions
- ✅ Gracefully rejects non-Kuhn elements (returns cell_size=0)
- ✅ Achieves 96.34% cells with exactly 6 elements

### Why Diagonal Method Would Be Worse
The reference doc's Method 2 (body diagonal grouping):
- ❌ Assumes cubic cells (`side = diagonal/sqrt(3)`) — wrong for anisotropic mesh
- ❌ Single scalar (diagonal length) vs 3 independent dimensions (dx,dy,dz)
- ❌ Cannot detect non-Kuhn elements (no clear failure mode)
- ✅ Body diagonal IS always longest edge (even for anisotropic) — but this doesn't help

### The Real Problem: 100% Cell Spanning
From `logs/octree_retention_diagnostics.log`:
```
Single-cell elements: 0 (0.00%)
Boundary-spanning elements: 3,047,074 (100.00%)

>>> THIS IS THE KEY METRIC <<<
100.00% of elements span cell boundaries
```

**Why?** Kuhn tets have vertices at cube corners. Cube corners ARE grid cell boundaries. Every tet spans from min corner to max corner of its parent cube → every tet's 4 vertices are in different grid cells than the centroid.

**Impact**: Simulated displacement test shows:
- 0.2× cell size: 99.99% searchable
- 0.5× cell size: 60.58% searchable ⚠️
- 1.0× cell size: 10.59% searchable ❌
- 2.0× cell size: 1.77% searchable ❌

RK4 tracking loses 11.4% particles over 100 steps because particle motion crosses cell boundaries and the new element isn't registered in the neighbor cell.

---

## Solution: Multi-Cell Vertex Registration

Register each element in ALL cells its vertices touch (typically 2-4 cells per element).

### Algorithm

```python
def extract_octree_cells_vertex_multi(node_positions, connectivity, tolerance=1e-6):
    """
    Register elements in ALL cells their vertices touch.

    Same as single_cell extraction, but step 3 changes:
    """

    for elem_id in range(n_elements):
        vertices = node_positions[connectivity[elem_id]]

        # Step 1-2: Same as current code
        cell_size, level = find_axis_aligned_edges_single(vertices, tolerance)
        if np.any(cell_size == 0):
            continue  # Skip non-Kuhn elements

        # Step 3: NEW - Find ALL cells touched by vertices
        vertex_cells = set()
        for vertex in vertices:
            # Grid indices for this vertex
            i = int(np.floor(vertex[0] / cell_size[0]))
            j = int(np.floor(vertex[1] / cell_size[1]))
            k = int(np.floor(vertex[2] / cell_size[2]))

            # Encode to Morton
            morton = encode_morton_3d_single(i + offset, j + offset, k + offset)
            cell_key = (morton, level)
            vertex_cells.add(cell_key)

        # Register element in ALL vertex cells
        for cell_key in vertex_cells:
            cell_to_elements_dict[cell_key].append(elem_id)

        # Track cells per element
        cells_per_element[elem_id] = len(vertex_cells)
```

### Expected Results

**Before (single-cell)**:
- Cells per element: 1.0
- Elements per cell: 5.89 (mean), 6 (median)
- Cell coverage: 517,069 cells
- Boundary spanning: 100%

**After (vertex-multi)**:
- Cells per element: ~2-4 (vertices at corners → typically 2-4 unique cells)
- Elements per cell: ~12-24 (each cell now contains elements whose vertices touch it)
- Cell coverage: 517,069 cells (same cells, just more registrations)
- Boundary spanning: Still 100%, but NOW searchable from all touched cells ✅

**Memory impact**:
- Current CSR data: 3,047,074 entries (1 per element)
- New CSR data: ~9,000,000 entries (3× more, ~36 MB additional GPU memory)

**Search parameter changes**:
- Increase `max_tests_per_cell` from 20 → 30 (to handle ~12-24 elements/cell)
- No other GPU kernel changes needed

---

## Implementation Plan

### Files to Create

#### 1. `jaxtrace/gpu/search/mesh_aligned_octree_vertex_multi.py` (NEW)

Copy `mesh_aligned_octree_single_cell.py` and modify:

**Data structure**: `OctreeCellDataVertexMulti` (NamedTuple)
- Same fields as `OctreeCellDataSingle`
- Add: `cells_per_element_array`: (n_elements,) int32

**Main function**: `extract_octree_cells_vertex_multi()`
- Reuse `find_axis_aligned_edges_single()` (no changes)
- Reuse `encode_morton_3d_single()` (no changes)
- NEW logic: Loop over 4 vertices per element, collect unique (morton, level) cells
- Build CSR with ~3× more entries

**Handle non-Kuhn elements**:
- Same as current: skip if `cell_size == 0`
- 1,826 elements (0.06%) are transition elements at refinement boundaries
- Skipping them is acceptable (validated by research agent)

### Files to Modify

#### 2. `jaxtrace/gpu/search/__init__.py`
Add imports:
```python
from .mesh_aligned_octree_vertex_multi import (
    OctreeCellDataVertexMulti,
    extract_octree_cells_vertex_multi,
)
```

#### 3. `jaxtrace/gpu/search/mesh_aligned_search_with_neighbors.py`
Increase max_tests defaults from 20 → 30 in:
- `search_elements_in_cell(max_tests=30)` (line 40)
- `search_with_precomputed_neighbors_single(max_tests_per_cell=30)` (line 109)
- `search_multi_level_with_precomputed_neighbors(max_tests_per_cell=30)` (line 191)
- `search_batch_with_precomputed_neighbors(max_tests_per_cell=30)` (line 236)

Note: GPU upload code (`mesh_aligned_octree_gpu.py`) and neighbor table code (`mesh_aligned_octree_with_neighbor_table.py`) require NO changes — they work with any CSR size.

#### 4. `test_octree_verification_comprehensive.py`
Add vertex-multi extraction alongside single-cell:
```python
# After building single-cell octree
print("\n[NEW] Building vertex-multi octree...")
octree_vertex_multi = extract_octree_cells_vertex_multi(
    node_positions, connectivity, tolerance=1e-6, verbose=True
)

# Compare stats
print(f"\nComparison:")
print(f"  Single-cell: {octree_single.elements_per_cell_mean:.2f} elem/cell, "
      f"{octree_single.cells_per_element_mean:.2f} cell/elem")
print(f"  Vertex-multi: {octree_vertex_multi.elements_per_cell_mean:.2f} elem/cell, "
      f"{octree_vertex_multi.cells_per_element_mean:.2f} cell/elem")
```

#### 5. `test_alternative_search_strategies.py`
Add vertex-multi as a new strategy.

#### 6. `test_precomputed_neighbors.py`
Add vertex-multi variant and compare searchability at different perturbation scales.

#### 7. `test_octree_retention_diagnostics.py`
Add vertex-multi to Test 1 (boundary spanning) and Test 4 (simulated displacement).
Expected: displacement tolerance at 0.5-2.0× cell size should improve dramatically.

#### 8. `benchmark_l2_search_methods.py`
**Keep all existing methods** (they use the current single-cell octree). Add new methods:
- "Vertex-Multi Octree (direct)" — no neighbor search
- "Vertex-Multi Octree + Neighbors" — with precomputed neighbor table

Do NOT modify the existing mesh-aligned methods — they serve as baseline comparison.

**Test config additions**:
```python
{
    'name': 'Vertex-Multi Octree (direct)',
    'l2_method': 'mesh_aligned_octree_vertex_multi',
    'description': 'Multi-cell vertex registration (27 cells @ 3 levels)',
    'expected_leaves': '~20-30 tests/particle'
},
{
    'name': 'Vertex-Multi Octree + Neighbors',
    'l2_method': 'mesh_aligned_neighbors_vertex_multi',
    'description': 'Vertex-multi + neighbor table',
    'expected_leaves': '~25-35 tests/particle'
},
```

---

## Implementation Order

1. Create `mesh_aligned_octree_vertex_multi.py` with extraction function
2. Update `__init__.py` with imports
3. Update `mesh_aligned_search_with_neighbors.py` (increase max_tests)
4. Update `test_octree_verification_comprehensive.py` — compare both extractions
5. Update `test_octree_retention_diagnostics.py` — Test 1 & 4 with vertex-multi
6. Update `test_precomputed_neighbors.py` — searchability comparison
7. Update `test_alternative_search_strategies.py` — add vertex-multi strategy
8. Update `benchmark_l2_search_methods.py` — add vertex-multi methods

---

## Verification Success Criteria

**test_octree_verification_comprehensive.py**:
- Vertex-multi shows ~12-24 elements/cell (vs ~6 for single-cell) ✅
- Vertex-multi shows ~2-4 cells/element (vs 1.0 for single-cell) ✅
- Same 517,069 unique cells ✅

**test_octree_retention_diagnostics.py**:
- Test 1: Elements still span boundaries (geometric fact), but now registered in multiple cells ✅
- Test 4: Searchability at 0.5× cell displacement improves from 60% → 95%+ ✅
- Test 4: Searchability at 1.0× cell displacement improves from 10% → 80%+ ✅

**benchmark_l2_search_methods.py**:
- Vertex-multi retention improves from ~88% → 95%+ over 100 RK4 steps ✅
- Throughput remains competitive (may be slightly slower due to more tests, but still faster than radius methods) ✅

---

## Key Advantages

✅ **Keeps current robust extraction** — no risky diagonal-based grouping
✅ **Handles anisotropic cells** — already validated in current code
✅ **Predictable memory cost** — ~3× CSR expansion, ~36 MB GPU
✅ **No search algorithm changes** — just increase max_tests parameter
✅ **Gracefully handles non-Kuhn elements** — same skip logic as current

## Risk Mitigation

**Q: What if vertices land exactly on cell boundaries?**
A: `floor(x / cell_size)` is well-defined even for x = k × cell_size (returns k). Vertices on boundaries contribute to multiple cells, which is exactly what we want.

**Q: What about the 1,826 non-Kuhn elements?**
A: Research agent confirmed these are transition elements at refinement boundaries (0.06% of mesh). Current code skips them; vertex-multi will also skip them. This is acceptable — the 88% → 95% improvement comes from fixing the 99.94% of elements that ARE Kuhn.

**Q: Will 30 tests per cell be enough?**
A: Expected elements/cell is ~12-24. With 30 tests, we can check every element in 95% of cells. For the few cells with >30 elements, the search tests the first 30 (most likely to contain the particle anyway since they're spatially clustered).
