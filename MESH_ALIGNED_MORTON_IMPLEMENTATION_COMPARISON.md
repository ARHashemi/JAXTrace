Detailed Comparison: Old Morton vs New Morton vs Octree
1. ORIGINAL MORTON (Element-Based)
Data Structure:

Morton codes: Computed from element centroids (3,048,900 codes)
Spatial index: Elements sorted by their centroid's Morton code
No octree involved: Pure space-filling curve ordering
How it works:


Element 0: centroid = (x0, y0, z0) → morton_code_0
Element 1: centroid = (x1, y1, z1) → morton_code_1
...
Element 3M: centroid = (x3M, y3M, z3M) → morton_code_3M

Sort all 3M elements by their Morton codes
Result: Spatially close elements have similar Morton codes
Search Algorithm:


1. Query position q = (x, y, z)
2. Compute Morton code: m_q = morton_encode(q)
3. Binary search in sorted Morton array to find "leaf" containing m_q
4. Search radius R around this leaf:
   - Check leaf at index i-R, i-R+1, ..., i, ..., i+R-1, i+R
   - Each "leaf" contains ~107 elements (after binning)
5. Test elements in these 2R+1 leaves for point-in-tet
Key characteristics:

Morton codes act as 1D spatial hash of 3D positions
Binary search finds approximate spatial location
Radius search compensates for elements not at exact Morton position
Performance: ~536 tests/particle (5 leaves × 107 elem/leaf), 93-98% retention
Why radius R works:

Elements have extent (not points)
Element centroid at Morton m₁ may contain point at Morton m₂
Radius search checks neighboring Morton regions
2. DIRECT MESH-ALIGNED OCTREE (Cell-Based, Single Cell)
Data Structure:

Octree cells: 517,069 cubes extracted from mesh structure
Cell identification: Each tetrahedron belongs to ONE parent cube
Spatial index: Cells sorted by Morton code of cell center
How octree is extracted:


For each tetrahedron:
  1. Find 3 axis-aligned edges (Kuhn property)
  2. Infer cell size from edge lengths: cell_size = |edge|
  3. Compute centroid of tetrahedron
  4. Find parent cube: grid_idx = floor(centroid / cell_size)
  5. Assign tet to this ONE cube
  
Result: 517,069 unique cubes, each containing 5-6 tets
Search Algorithm:


1. Query position q = (x, y, z)
2. Compute Morton code: m_q = morton_encode(q)
3. Binary search to find cell with Morton code ≈ m_q
4. Search ONLY this cell (no radius)
5. Test all ~6 elements in this single cell
Key characteristics:

Octree structure is intrinsic to the mesh (from generator)
Morton code is just a lookup key (spatial hash)
No radius search - assumes element is in the same cell as query point
Performance: ~6 tests/particle, 74.6% retention
Why 74.6% failure:

Elements can SPAN multiple cells
Element assigned to cell A (via centroid)
Query point in cell B (element extends into B)
Single-cell search in B finds nothing
3. NEW MESH-ALIGNED MORTON (Hybrid: Cell-Based + Radius)
Data Structure:

Octree cells: Same 517,069 cubes as direct octree
Morton codes: Computed from cell centers (not element centroids)
Spatial index: Cells sorted by Morton code
Critical difference from original Morton:


OLD Morton:
  3,048,900 elements → 3,048,900 Morton codes
  morton_i = encode(element_i.centroid)
  
NEW Morton:
  517,069 cells → 517,069 Morton codes
  morton_j = encode(cell_j.center)
  Each cell j contains multiple elements
Search Algorithm:


1. Query position q = (x, y, z)
2. Compute Morton code: m_q = morton_encode(q)
3. Binary search to find cell with Morton code ≈ m_q
4. Search radius R cells around this cell:
   - Cell at index i-R, i-R+1, ..., i, ..., i+R-1, i+R
   - Each cell contains ~6 elements
5. Test elements in these 2R+1 cells
Key characteristics:

Uses octree structure (cells) + Morton ordering (spatial hash)
Morton code finds approximate cell location
Radius search checks neighboring cells (not just center cell)
Performance: ~30 tests/particle (5 cells × 6 elem/cell), expected ~98% retention
Coordinate System & Morton Encoding
The Morton Code Mismatch (Root Cause of 0% → 23.6%)
OLD (broken) implementation:

Cell extraction (mesh_aligned_octree_single_cell.py:226-234):

# Grid indices (i, j, k) computed from global coordinates
i = floor(centroid[0] / cell_size[0])  # Can be negative!
j = floor(centroid[1] / cell_size[1])
k = floor(centroid[2] / cell_size[2])

# Add offset to handle negative indices
offset = 2^19
morton = encode(i + offset, j + offset, k + offset)
Example: If centroid = (-0.02, -0.01, -0.005):


cell_size = 0.0025
i = floor(-0.02 / 0.0025) = -8
j = floor(-0.01 / 0.0025) = -4
k = floor(-0.005 / 0.0025) = -2

morton = encode(-8 + 524288, -4 + 524288, -2 + 524288)
       = encode(524280, 524284, 524286)
       ≈ some large uint64
Search (mesh_aligned_morton_search.py:163-192):

# Normalize to [0, 1] within bbox
bbox_min = [-0.030, -0.023, -0.010]
bbox_max = [0.030, 0.023, 0.000]

normalized = (pos - bbox_min) / (bbox_max - bbox_min)
# For pos = (-0.02, -0.01, -0.005):
# normalized = (0.01/0.06, 0.013/0.046, 0.005/0.01) = (0.167, 0.283, 0.5)

u = floor(normalized * 2^21)
# u = (349525, 592445, 1048576)

morton = encode(349525, 592445, 1048576)
       ≈ completely different uint64!
Result: Binary search fails because cell Morton ≠ query Morton.

NEW (fixed) implementation:

Both cell extraction AND search use bbox-based normalization:


# BOTH builder and search do this:
bbox_min = node_positions.min(axis=0)  # Global bbox
bbox_max = node_positions.max(axis=0)

# Normalize ANY position (cell center or query) the same way
normalized = (pos - bbox_min) / (bbox_max - bbox_min)
normalized = clip(normalized, 0, 1)

u = floor(normalized * (2^21 - 1))
morton = encode(u[0], u[1], u[2])
Now consistent:

Cell center at (-0.02, -0.01, -0.005) → morton code M
Query at (-0.02, -0.01, -0.005) → same morton code M
Binary search succeeds!
From log line 96:


Morton range: [82637478876830069, 9010788041098190701]
This range now makes sense (normalized coordinates).

Why Still Only 23.6%?
Looking at the test: random particles in bbox.

Expected behavior:

~30-50% of random bbox particles fall inside the mesh (rest in void)
Of those inside, radius=2 should find ~98%
So overall: 0.4 × 0.98 = 39% expected
Actual: 23.6% → 60% of the in-mesh particles are being found

This suggests radius search is still not working correctly. Let me check the radius search implementation:

Read mesh_aligned_morton_search.py (lines 341-420)
The issue is clear! The radius search is searching along the 1D Morton curve (indices i-R to i+R), but this doesn't correspond to spatial neighbors in 3D!

The Fundamental Problem with Morton Radius Search on Cells
Morton curve properties:

Maps 3D space → 1D curve
Locally preserves proximity: nearby 3D points → nearby 1D indices
But NOT globally: distant 3D points can have nearby 1D indices
Why it works for element-based Morton:

3M elements densely fill space
Morton curve densely samples 3D space
Radius along curve ≈ radius in 3D (locally)
Why it FAILS for cell-based Morton:

Cells are sparse in Morton space (517K cells vs 2^63 possible codes)
Cell centers at indices [i-2, i-1, i, i+1, i+2] are NOT spatially adjacent!
They're just the next cells along the sorted Morton array
Example scenario:


Cell 100: Morton = 1000, center = (0.1, 0.1, 0.1), size = 0.01
Cell 101: Morton = 1001, center = (0.5, 0.5, 0.5), size = 0.02  ← COULD BE FAR AWAY!
Cell 102: Morton = 1002, center = (0.11, 0.11, 0.11), size = 0.01

Query at (0.105, 0.105, 0.105):
- Binary search finds Cell 100
- Radius=2 searches cells [98, 99, 100, 101, 102]
- But Cell 101 might be spatially FAR (different octree branch)!
The radius search needs to work in Morton code space, not array index space.

Solution: Search Morton Code Range, Not Index Range
Instead of:


# WRONG: Search array indices i-R to i+R
neighbor_cell_id = center_cell_id + offset  # Treats sorted array as spatial neighbors
We need:


# CORRECT: Search Morton codes in range [m_q - Δ, m_q + Δ]
# Where Δ depends on cell size and desired spatial radius
But this is complex because:

Cell sizes vary (different refinement levels)
Morton code distance ≠ Euclidean distance
Need to find spatially adjacent cells, not Morton-adjacent indices
Alternative: Use 3D Grid Neighbor Search
Since we already have grid indices (i, j, k) for each cell, we should search 26-neighbors in grid space:


# Center cell at grid indices (i, j, k)
# Search all cells at:
for di in [-R, ..., 0, ..., +R]:
  for dj in [-R, ..., 0, ..., +R]:
    for dk in [-R, ..., 0, ..., +R]:
      neighbor_grid = (i+di, j+dj, k+dk)
      # Find cell with these grid indices
      # Search that cell
But this requires:

Storing grid indices in GPU structure (we have this)
Building a grid index → cell ID mapping
Or: finding cells by grid coordinates (not Morton code)
This is exactly what the direct mesh-aligned octree was trying to do, but failed because elements span cells.

Summary of Architectures
Aspect	Original Morton	Direct Octree	Hybrid Morton (Current)
Spatial structure	None (flat element list)	Octree cells (517K)	Octree cells (517K)
Morton codes from	Element centroids (3M)	Cell centers (517K)	Cell centers (517K)
Search method	Radius along Morton curve	Single cell lookup	Radius along Morton curve
Radius works?	✅ Yes (dense sampling)	N/A (no radius)	❌ No (sparse cells)
Why?	Elements densely fill space	-	Cells sparse in Morton space
Tests/particle	~536	~6	~30 (if radius worked)
Retention	93-98%	74.6%	Currently 60% of in-mesh
The Real Issue
The hybrid approach incorrectly assumes:

"Cells sorted by Morton code are spatially adjacent"

Reality:

"Cells sorted by Morton code are only locally adjacent at fine granularity"

With 517K cells in a 2^63 Morton space, cells are extremely sparse. Adjacent array indices don't mean adjacent in 3D space.

To fix this properly, you need one of:

Grid-based neighbor search: Use (i,j,k) indices to find spatial neighbors
Morton range search: Search Morton codes in range, not array indices
Increase radius significantly: radius=50 might work, but defeats efficiency purpose
27-neighbor stencil: Implement the nested control flow approach (risks OOM)
The most practical fix is #1: grid-based neighbor search, which I can implement if you'd like.
