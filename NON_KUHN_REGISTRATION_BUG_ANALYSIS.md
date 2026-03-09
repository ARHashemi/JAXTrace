# Non-Kuhn Element Registration Bug: Origin, Tracing, and Fix

## 1. Context

JAXTrace implements particle tracking through a finite element mesh using RK4 integration
on GPU (JAX). The mesh is a Kuhn tetrahedral subdivision of an adaptive octree, with ~3M
elements and ~780K nodes. Particle location (finding which element contains a given point)
is the performance-critical inner loop.

The **mesh-aligned octree** approach exploits the fact that Kuhn tetrahedra have a known
relationship to their parent octree cube: each cube at refinement level L subdivides into
5-6 tetrahedra with axis-aligned edges of length 2^(-L). This allows building an octree
index where each cell maps directly to the mesh's own octree structure.

The **multi-cell vertex registration** variant registers each element in ALL octree cells
touched by its 4 vertices (~4 cells per element), enabling a compact 3x3x3 local search
that checks 27 cells per refinement level.

### The Problem

Benchmark testing showed 18.84% particle retention over 2,500 RK4 steps, far below the
>95% target. Even in targeted diagnostic tests on the mesh interior, specific positions
were consistently unfindable by the 3x3x3 search despite being inside valid mesh elements.

---

## 2. Discovery: Extended Void Region Diagnostic

**Log**: `logs/diagnose_void_region_corrected_extended.log`

An extended diagnostic test sampled 8,000 random positions across 4 spatial zones of the
mesh (interior, just-inside boundary, just-outside boundary, approach zone). The 3x3x3
search achieved near-perfect results in most zones:

| Zone        | Samples | 3x3x3 Found | Miss Rate |
|-------------|---------|-------------|-----------|
| Interior    | 2,000   | 2,000       | 0.0%      |
| Just Inside | 2,000   | 2,000       | 0.0%      |
| Just Outside| 2,000   | 2,000       | 0.0%      |
| Approach    | 2,000   | 1,981       | **1.0%**  |

**Two positions were missed** by the 3x3x3 search but found by the radius-based Morton
search (a separate, brute-force search method). Both mapped to element **39551**.

Key observation from the log:
```
Position [6316]: (-0.025374, -0.001968, -0.002550)
  Radius found: element 39551
  Element Non-Kuhn: True
  Registered in 1 cell (should be ~4)
```

This was the first clue: element 39551 is a **Non-Kuhn** element registered in only
**1 cell** instead of the expected ~4 cells.

---

## 3. Understanding Non-Kuhn Elements

In a Kuhn tetrahedral mesh derived from an octree, the vast majority of elements are
"Kuhn tetrahedra" — tetrahedra produced by a standard subdivision of axis-aligned cubes.
These have a defining property: 3 edges aligned with the coordinate axes, with lengths
equal to the parent cube's edge length.

At refinement level transitions (where adjacent cubes differ by one level), **transition
elements** are needed to maintain mesh conformity. These "Non-Kuhn" elements lack
axis-aligned edges and cannot be directly associated with a single octree cube.

Statistics from the mesh:
- **Kuhn elements**: 3,047,074 (99.94%)
- **Non-Kuhn elements**: 1,826 (0.06%)
- Non-Kuhn elements are concentrated in the weld zone: X:[-0.011, 0.010]

The function `find_axis_aligned_edges_single()` in `mesh_aligned_octree_single_cell.py`
detects axis-aligned edges and returns `cell_size = [0, 0, 0]` for Non-Kuhn elements
(no axis-aligned edges found).

---

## 4. Deep Diagnostic: Tracing the Exact Failure

**Log**: `logs/diagnose_3x3x3_nonkuhn_failure.log`

A focused diagnostic was built to:
1. Identify ALL 1,826 Non-Kuhn elements in the mesh
2. Generate 5 random interior test points per element (9,130 total)
3. Run the GPU 3x3x3 search on each
4. For missed points, replay the search on CPU to trace exactly where it fails

### Results

```
Total test points:  9,130
Found (any elem):   9,030 (98.9%)
Missed:             100 (1.1%)
Found correct elem: 9,030
Found wrong elem:   0
```

All 100 missed points belonged to just **two elements**: 39551 and 39556.

### Registration Analysis

Of the 1,826 Non-Kuhn elements:
- Registered in 0 cells: 0
- Registered in 1 cell: **344** (these are the problematic ones)
- Registered in 2+ cells: 1,482

Elements 39551 and 39556 were both registered in exactly **1 cell**: Cell 405643.

### CPU Replay Reveals the Morton Code Mismatch

For each missed point, the diagnostic replayed the 3x3x3 search on CPU, manually
computing grid indices and Morton codes at each refinement level, then checking whether
the true element was in any of the 27 searched cells.

Example from the log (element 39551):
```
Registered in 1 cell(s):
  Cell 405643: level=8, grid=(-1,0,-1), size=(0.04000000,0.04000000,0.04000000)

CPU replay of 3x3x3 search:
  Level 8: searched 27 cells, 210 elems, true_elem in cells=False
  NOT FOUND: searched 27 cells, 210 elements total
  ROOT CAUSE: Element 39551 is NOT in any cell reachable by 3x3x3
```

The critical observation: Cell 405643 has `size=0.04` at level 8, but the GPU search
uses `level_cell_sizes[8]` from the actual mesh which is approximately **0.004** (since
level 8 corresponds to 2^(-8) ~ 0.0039). The factor-of-10 discrepancy means the Morton
codes computed during registration and during search are completely different.

---

## 5. Root Cause Analysis

The bug was in `extract_octree_cells_vertex_multi()` in
`mesh_aligned_octree_vertex_multi.py`, specifically in how Non-Kuhn elements were handled.

### The Buggy Code Path

There were **two bugs** in the Non-Kuhn element registration:

#### Bug 1: Processing-Order-Dependent Neighbor Lookup

The original `find_face_neighbor_fast()` function searched for face neighbors among
**already-processed elements** using a sequential scan of `element_to_cells_dict`:

```python
def find_face_neighbor_fast(elem_id, connectivity, element_to_cells_dict, max_search=100):
    covered_ids = list(element_to_cells_dict.keys())
    start_idx = max(0, len(covered_ids) - max_search)

    for neighbor_id in covered_ids[start_idx:]:
        # Check if shares >= 3 nodes (face neighbor)
        ...
```

Problems with this approach:
- **Processing order dependency**: If a Non-Kuhn element appears before its Kuhn neighbors
  in element numbering, no Kuhn neighbor exists in `element_to_cells_dict` yet.
- **`max_search=100` limit**: Even when searching all processed elements, the linear scan
  of the last 100 elements might miss the actual neighbor.
- **No guarantee of finding a Kuhn neighbor**: Could return another Non-Kuhn element.

Result: 344 out of 1,826 Non-Kuhn elements fell through to the fallback path.

#### Bug 2: Broken Fallback with Wrong Cell Size

When `find_face_neighbor_fast()` returned -1, the fallback used **hardcoded values**:

```python
fallback_level = 8
fallback_size = 0.04  # WRONG! Actual level 8 size is ~0.004
```

This created a "phantom cell" with:
- Morton code computed using `cell_size = 0.04` → grid index `floor(pos / 0.04)`
- Registered at level 8

But the GPU search computes grid indices using `level_cell_sizes[8]` from the actual mesh
(~0.004), producing `floor(pos / 0.004)` — a completely different grid index and Morton
code. The phantom cell is unreachable.

### The Morton Code Mismatch Illustrated

For element 39551 at position (-0.026, 0.0, -0.005):

| Step | Registration (buggy) | Search (correct) |
|------|---------------------|-------------------|
| cell_size | 0.04 | ~0.004 |
| grid_x | floor(-0.026/0.04) = -1 | floor(-0.026/0.004) = -7 |
| grid_y | floor(0.0/0.04) = 0 | floor(0.0/0.004) = 0 |
| grid_z | floor(-0.005/0.04) = -1 | floor(-0.005/0.004) = -2 |
| Morton code | encode(-1+2^19, 0+2^19, -1+2^19) | encode(-7+2^19, 0+2^19, -2+2^19) |

These produce entirely different Morton codes. The search's 3x3x3 neighborhood around
(-7, 0, -2) will never reach the registration cell at (-1, 0, -1).

---

## 6. The Fix

**File**: `jaxtrace/gpu/search/mesh_aligned_octree_vertex_multi.py`

### Fix 1: Two-Pass Extraction with Node-to-Element Index

Replaced the single-pass approach with a two-pass algorithm:

- **Pass 1**: Process all Kuhn elements first, collecting their `(cell_size, level)` info
  in a `kuhn_element_info` dictionary.
- **Pass 2**: Process Non-Kuhn elements with full knowledge of ALL Kuhn neighbors.

Replaced `find_face_neighbor_fast()` (processing-order-dependent, linear scan) with
`find_kuhn_face_neighbor()` backed by a prebuilt **node-to-element index**:

```python
def build_node_to_elements(connectivity):
    """node_id -> set of element IDs sharing that node"""
    node_to_elements = defaultdict(set)
    for elem_id in range(connectivity.shape[0]):
        for node_id in connectivity[elem_id]:
            node_to_elements[int(node_id)].add(elem_id)
    return node_to_elements

def find_kuhn_face_neighbor(elem_id, connectivity, node_to_elements, kuhn_element_info):
    """Find face neighbor (>= 3 shared nodes) that is a Kuhn element."""
    elem_nodes = connectivity[elem_id]
    candidates = set()
    for node_id in elem_nodes:
        candidates.update(node_to_elements[int(node_id)])
    candidates.discard(elem_id)

    elem_node_set = set(int(n) for n in elem_nodes)
    for neighbor_id in candidates:
        if neighbor_id not in kuhn_element_info:
            continue
        shared = elem_node_set & set(int(n) for n in connectivity[neighbor_id])
        if len(shared) >= 3:
            cell_size, level = kuhn_element_info[neighbor_id]
            return neighbor_id, cell_size, level
    return -1, None, None
```

This is:
- **Order-independent**: Uses a global index, not processing order
- **Guaranteed complete**: Checks ALL elements sharing any node, not just recently processed
- **Kuhn-only**: Returns only Kuhn neighbors with valid cell_size/level

### Fix 2: Register Using Own Vertices with Neighbor's Grid Parameters

Instead of copying the neighbor's cell registrations (which may be spatially distant) or
using a broken fallback, Non-Kuhn elements are now registered using:
- Their **own vertex positions** (so the cells are spatially correct)
- The Kuhn neighbor's **cell_size and level** (so the Morton codes match the search)

```python
# Pass 2: For each Non-Kuhn element
neighbor_id, neighbor_cell_size, neighbor_level = find_kuhn_face_neighbor(
    elem_id, connectivity, node_to_elements, kuhn_element_info
)

if neighbor_id >= 0:
    # Register using OWN vertices with neighbor's grid parameters
    vertex_cells = set()
    for vertex in vertices:
        i = int(np.floor(vertex[0] / neighbor_cell_size[0]))
        j = int(np.floor(vertex[1] / neighbor_cell_size[1]))
        k = int(np.floor(vertex[2] / neighbor_cell_size[2]))
        # ... Morton encoding with neighbor_level ...
        cell_key = (morton, neighbor_level)
        vertex_cells.add(cell_key)

    for cell_key in vertex_cells:
        element_to_cells_dict[elem_id].add(cell_key)
        cell_to_elements_dict[cell_key].add(elem_id)
```

This ensures that:
1. The Non-Kuhn element is registered in cells that **physically contain its vertices**
2. The cell_size used for registration **matches** `level_cell_sizes[level]` on the GPU
3. The Morton codes computed during registration **match** those computed during search
4. The 3x3x3 neighborhood around any interior point will always include the registration cells

---

## 7. Verification

**Log**: `logs/diagnose_nonkuhn_fix_verification.log`

After applying the fix, the same diagnostic was re-run:

### Before Fix
```
Non-Kuhn registration:
  Registered in 1 cell:  344
  Registered in 2+ cells: 1,482

3x3x3 search on Non-Kuhn interiors:
  Found: 9,030/9,130 (98.9%)
  Missed: 100 (1.1%)
```

### After Fix
```
Non-Kuhn registration:
  Registered in 0 cells: 0
  Registered in 1 cell:  0
  Registered in 2+ cells: 1,826

Non-Kuhn with Kuhn neighbor: 1,826
Non-Kuhn without Kuhn neighbor: 0

3x3x3 search on Non-Kuhn interiors:
  Found: 9,130/9,130 (100.0%)
  Missed: 0 (0.0%)
  Found correct elem: 9,130
  Found wrong elem: 0
```

All 1,826 Non-Kuhn elements now have Kuhn face neighbors found (100%), all are registered
in 2+ cells, and all 9,130 interior test points are correctly found by the 3x3x3 search.

### Octree Statistics (unchanged)
- Cells: 665,842
- Elements per cell: 18.32 (mean)
- Cells per element: 4.00 (mean)

---

## 8. Summary of Changes

| Aspect | Before | After |
|--------|--------|-------|
| Non-Kuhn handling | Skip entirely / broken fallback | Two-pass with vertex registration |
| Neighbor lookup | `find_face_neighbor_fast` (order-dependent, max_search=100) | `find_kuhn_face_neighbor` (node-to-element index, order-independent) |
| Registration | Copy neighbor's cells OR hardcoded fallback (level=8, size=0.04) | Own vertices + neighbor's cell_size/level |
| Non-Kuhn in 1 cell | 344 | 0 |
| Non-Kuhn missed by 3x3x3 | 100/9,130 (1.1%) | 0/9,130 (0.0%) |
| Files modified | `mesh_aligned_octree_vertex_multi.py` | Same |

### Key Lesson

When registering elements into a spatial index, the cell parameters (size, level) used
during **registration** must exactly match those used during **search**. The GPU search
function computes grid indices as `floor(position / level_cell_sizes[level])`, where
`level_cell_sizes` is derived from actual mesh cells at each refinement level. Any
registration that uses a different cell_size will produce a different Morton code, making
the element invisible to the search — even if the cell appears at the "correct" grid
position from the registration's perspective.

---

## 9. Post-Fix Validation: Seeding Strategy Stress Test

**Script**: `diagnose_seeding_strategies.py`
**Log**: `logs/diagnose_seeding_strategies_x-perturbed_analitics.log`

After the Non-Kuhn fix was applied, a stress test was conducted to characterize the
remaining miss rate of the 3×3×3 search under increasingly aggressive particle
displacement, isolating any residual structural weaknesses.

### Test Design

50,000 random elements were sampled from the full mesh. Four seeding strategies were
tested, with all perturbations along the X axis only to permit exact boundary-crossing
analysis:

| Strategy | Perturbation | Description |
|----------|-------------|-------------|
| 1 | 0 | Exact element centroids |
| 2 | ±10% × min_edge_length along X | Tiny displacement, stays inside element |
| 3 | ±1× min_edge_length along X | Full edge-length displacement, may cross element face |
| 4 | ±2× min_edge_length along X | Double edge-length, exits element into neighbor or beyond |

For each missed particle, a CPU replay of the 3×3×3 search was performed at double
precision, tracing: position vs. mesh bounding box, source element type and registration,
per-level cell count and element count visited, and whether the true element was reachable
from the landed position.

### Results

```
Strategy                                       Found    Missed   Found%   Correct%   MeanTests
----------------------------------------------------------------------------------------------
  1. Exact centroids                          50,000         0  100.00%    100.00%       173.8
  2. ±10% elem-size along X                   50,000         0  100.00%    100.00%       173.8
  3. ±1× elem-size along X                    49,997         3   99.99%      0.00%       178.9
  4. ±2× elem-size along X                    49,995         5   99.99%      0.00%       180.6
```

Note: Correct% is 0% for strategies 3 & 4 by design — particles at ±1× to ±2× element
size along X have crossed element faces, so the ground-truth centroid element is the wrong
answer. Any adjacent element found is a valid result.

### Root Cause of All 8 Missed Particles

The CPU replay established **two distinct failure modes**:

#### Mode 1: Position Genuinely Outside the Mesh Domain (7 out of 8 cases)

All 3 misses in strategy 3 and 4 out of 5 misses in strategy 4 share the same pattern:
the particle was displaced past the mesh boundary at X = ±0.030. The mesh occupies
X ∈ [−0.030, +0.030]; the missed positions have |X| > 0.030.

The CPU replay for these cases shows:
- The true element **is found** in the searched cells (registration is correct)
- The `point_in_tet` test fails on all 27×8 checked elements
- No adjacent element contains the position either

This is the **correct behavior**: the search accurately returns "not found" for a position
that is genuinely outside all tetrahedra in the mesh. These are not false negatives — the
search is working as intended.

All boundary-miss source elements sit in the outermost column of the mesh (X ≈ ±0.025 to
±0.030 at level 8, cell size ~5 mm). A perturbation of 1× min_edge_length (~5 mm) is
sufficient to step over the mesh face into the void.

Detailed cases from the log:

| Miss | Strategy | Source Elem | Type | Source X range | Landed X | Δ from boundary |
|------|----------|-------------|------|----------------|----------|-----------------|
| 3/#1 | ±1× | 422592 | Non-Kuhn | [−0.030, −0.025] | −0.03125 | −1.25 mm |
| 3/#2 | ±1× | 39640 | Kuhn L8 | [−0.030, −0.025] | −0.03375 | −3.75 mm |
| 3/#3 | ±1× | 39586 | Kuhn L8 | [−0.030, −0.025] | −0.03250 | −2.50 mm |
| 4/#1 | ±2× | 40151 | Non-Kuhn | [+0.025, +0.030] | +0.03625 | +6.25 mm |
| 4/#3 | ±2× | 39640 | Kuhn L8 | [−0.030, −0.025] | −0.03875 | −8.75 mm |
| 4/#4 | ±2× | 39586 | Kuhn L8 | [−0.030, −0.025] | −0.03750 | −7.50 mm |
| 4/#5 | ±2× | 40142 | Kuhn L8 | [+0.025, +0.030] | +0.03750 | +7.50 mm |

CPU replay output for a representative case (strategy 3, miss #2, element 39640):
```
Landed position:    (-0.0337499976, -0.0051111151, -0.0062499996)
In mesh bbox:       False
Mesh bbox:          (-0.030000,-0.023000,-0.010000) → (0.030000,0.023000,0.000000)

Source element: 39640
  Type:    Kuhn (level 8)
  Registered in 4 octree cell(s):
    Cell 147490: level=8, grid=(-7,-1,-2)  |diff|=(0,0,0) max=0  reachable=YES
    Cell 147495: level=8, grid=(-7,-2,-1)  |diff|=(0,0,1) max=1  reachable=YES
    Cell 147499: level=8, grid=(-7,-1,-1)  |diff|=(0,0,1) max=1  reachable=YES
    Cell 147516: level=8, grid=(-6,-2,-1)  |diff|=(1,0,1) max=1  reachable=YES

CPU replay: Level 8: 12 cells, 108 elems, TRUE ELEM IN CELLS
Result: NOT FOUND — point-in-tet fails for all elements
ROOT CAUSE: Position outside mesh boundary (no element contains the point)
```

#### Mode 2: Registration Gap from Extreme Displacement (1 out of 8 cases)

Miss #2 of strategy 4 (particle 38668, source element 39686) is qualitatively different:

```
Landed position:    (-0.0324999988, -0.0191666745, -0.0087500000)
Perturbation (X):   -1.000000e-02

Source element: 39686
  Type:    Kuhn (level 8)
  Registered in 4 octree cell(s):
    Cell 147193: level=8, grid=(-5,-5,-2)  |diff|=(2,1,0) max=2  reachable=NO ← OUT OF 3×3×3
    Cell 147460: level=8, grid=(-5,-4,-2)  |diff|=(2,0,0) max=2  reachable=NO ← OUT OF 3×3×3
    Cell 147589: level=8, grid=(-4,-4,-2)  |diff|=(3,0,0) max=3  reachable=NO ← OUT OF 3×3×3
    Cell 147601: level=8, grid=(-4,-4,-1)  |diff|=(3,0,1) max=3  reachable=NO ← OUT OF 3×3×3

CPU replay: Level 8: 12 cells, 91 elems
Result: NOT FOUND — True elem NOT in any searched cell
ROOT CAUSE: Element not reachable by 3×3×3 from this position
```

Here the particle has traveled 10 mm (2× the source element's min_edge_length at level 8)
in −X, landing at grid index X = −7 while the source element's registrations are at
X = −5 to −4 — a gap of 2 to 3 grid steps in X, beyond the ±1 reach of 3×3×3. The
position is also outside the mesh boundary (X = −0.0325 vs. mesh limit −0.030), so even
if 5×5×5 search were used, no element would contain it. This is simultaneously a
registration gap (3×3×3 cannot reach source element cells) and a boundary crossing (no
element contains the point). The practical root cause is the same as Mode 1: the particle
has exited the mesh.

### Conclusion

**The 3×3×3 search is performing correctly after the Non-Kuhn registration fix.** All 8
missed particles in 100,000 total queries (strategies 3 + 4 combined) were displaced
outside the mesh bounding box. The search correctly returned "not found" in every case
because no mesh element contains those positions.

The miss rate of 0.01% at ±1× element-size displacement is entirely attributable to the
finite fraction of sampled elements that happen to sit on the outermost boundary layer of
the mesh, where a single-element-width displacement in X exits the domain. This is not a
search defect.

**Implication for production RK4 tracking**: particle loss during integration is caused
by particles physically leaving the mesh domain (which is physically correct), not by
any deficiency in the 3×3×3 spatial index. The residual 18.84% → target retention issue
must have a separate root cause (e.g., RK4 integration stepping particles outside the
mesh in a single step due to large velocity gradients near the weld zone boundary).
