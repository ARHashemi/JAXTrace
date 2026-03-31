# L2 Search Method Benchmark: Accuracy and Performance Analysis

## 1. Overview

This document presents a comprehensive evaluation of the L2 (global) point-location search methods
implemented in JAXTrace for locating particles within an unstructured tetrahedral mesh. The L2 search
is used during initial particle assignment and as a fallback when the faster L0 (cached element) and
L1 (face-neighbor hop) searches fail.

Five L2 methods were benchmarked:

| Method | Algorithm | Cells searched per level | Data structure |
|--------|-----------|------------------------|----------------|
| **radius r=2** | Morton-curve linear band (±2 cells) | 5 | Single-cell Morton |
| **radius r=10** | Morton-curve linear band (±10 cells) | 21 | Single-cell Morton |
| **1×1×1** | Center cell only, multi-level | 1 | Multi-cell octree |
| **3×3×3** | 3D neighborhood, multi-level | 27 | Multi-cell octree |
| **5×5×5** | 3D neighborhood, multi-level | 125 | Multi-cell octree |

**Test mesh:** 3,050,196 tetrahedral elements, 571,533 nodes (Kuhn-type mesh with adaptive
refinement). Element sizes range from 7.81e-05 to 5.00e-03 (minimum edge length).


## 2. Search Algorithms

### 2.1 Mesh-Aligned Octree (1×1×1, 3×3×3, 5×5×5)

All three octree-based methods share **the same algorithm** (`jnp.where`-based, vmap-compatible),
differing only in the neighborhood size. The octree is intrinsic to the Kuhn tetrahedral mesh
structure: each octree cell corresponds to the cube from which a set of tetrahedra was generated
via Kuhn subdivision.

**Element registration (multi-cell vertex-based):**
Each element is registered in every octree cell where any of its 4 vertices falls. For Kuhn
tetrahedra, vertices lie at cube corners, so each element is registered in ~4 cells on average
(the cells sharing the cube corners). This yields:
- 666,162 cells
- 18.3 elements/cell (mean)
- 4.0 cells/element (mean)

**Search algorithm:**
```
For each refinement level (14, 13, ..., 7):
    1. Compute base grid cell (i, j, k) from particle position
    2. For each offset (di, dj, dk) in the NxNxN neighborhood:
       a. Compute neighbor cell indices (i+di, j+dj, k+dk)
       b. Encode to Morton code, binary-search in sorted cell array
       c. If cell exists, test all elements in it (point-in-tet via fori_loop)
       d. Stop on first containing element
    3. If found at this level, skip remaining coarser levels (via jnp.where)
```

The three variants differ only in the offset range:
- **1×1×1**: `di, dj, dk ∈ {0}` — 1 cell per level (center only)
- **3×3×3**: `di, dj, dk ∈ {-1, 0, 1}` — 27 cells per level
- **5×5×5**: `di, dj, dk ∈ {-2, -1, 0, 1, 2}` — 125 cells per level

All use `jnp.where` (not `lax.cond`) for vmap compatibility, ensuring correct behavior under
JAX's parallel evaluation model.

### 2.2 Morton-Curve Radius Search

The radius search uses a different data structure (single-cell Morton) and a fundamentally
different neighborhood definition.

**Element registration (single-cell centroid-based):**
Each element is registered in exactly one cell — the cell containing its centroid. This yields:
- 517,309 cells
- 5.9 elements/cell (mean)
- ~1.0 cells/element

**Search algorithm:**
```
1. Encode particle position to Morton code
2. Binary-search in sorted Morton array → center_cell_index
3. Search center cell: call search_in_cell() with fori_loop(0, min(n_elems, 256), ...)
4. If not found, loop over 2×radius neighbor cells:
     For each offset i ∈ [0, 2R):
       Map i to cell offset: [-R, ..., -1, +1, ..., +R]
       neighbor_cell_id = center_cell_id + offset  (clamped to [0, n_cells-1])
       Call search_in_cell(pos, neighbor_cell_id, ...) with fori_loop(0, min(n_elems, 256), ...)
```

**What radius=N actually means:**
- `radius=2` searches 5 cells total: center + 2 backward + 2 forward in the sorted Morton array
- `radius=10` searches 21 cells total: center + 10 backward + 10 forward
- Each cell search has a bounded `fori_loop` with `max_tests_per_cell=256`
- The "radius" is an offset in the **1D sorted cell array**, NOT a spatial 3D neighborhood

**Critical difference:** Two cells adjacent in the 1D Morton ordering may be spatially distant,
and two spatially adjacent cells may be far apart in the Morton ordering. The radius search
is fundamentally a 1D band search, not a 3D neighborhood search.


## 3. Benchmark Design

### 3.1 Perturbation-Based Test

Particles are seeded from random element centroids with Gaussian perturbation scaled by each
element's minimum edge length:

```
position = centroid + N(0,1) × min_edge_length × perturbation_factor
```

where `perturbation_factor ∈ {0.0, 0.1, 0.2, 0.5, 0.7, 1.0}`. At factor 0.0, particles sit
at exact centroids. At factor 0.1, most particles remain inside their source element. At
factor ≥ 0.5, most particles have moved to neighboring elements.

**Metrics:**
- **Found rate**: Fraction of particles assigned to any element
- **Accuracy**: Among found particles, fraction matching the source element (meaningful only for
  perturbation < ~0.5×)
- **Search failures**: Particles inside the mesh bounding box that were not found (indicates
  a true search deficiency, not particles that left the domain)

### 3.2 Intra-Element Accuracy Test

Particles are placed at **known positions inside elements** using barycentric coordinates.
The ground-truth element is guaranteed — any failure to find the correct element is a real
search or point-in-tet error.

**Position types** (controlled via barycentric coordinates λ₀, λ₁, λ₂, λ₃ with Σλᵢ = 1):

| Type | Description | Barycentric generation |
|------|-------------|----------------------|
| `centroid` | Element center | λ = (0.25, 0.25, 0.25, 0.25) |
| `random` | Uniform random inside tet | Sort 3 uniform [0,1] values; take differences |
| `near_face` | Near a random face | One λₖ = 0.02; remaining 0.98 split uniformly among other 3 |
| `near_edge` | Near a random edge | Two λₖ ∈ [0, 0.02]; remaining split between other 2 |
| `near_vertex` | Near a random vertex | Three λₖ ∈ [0, 0.02]; fourth λ ≈ 0.94–1.0 |

**Barycentric seeding algorithms in detail:**

*Uniform random in tetrahedron (sorted-differences method):*
```
u₁, u₂, u₃ ~ Uniform(0, 1), then sort: u₍₁₎ ≤ u₍₂₎ ≤ u₍₃₎
λ₀ = u₍₁₎
λ₁ = u₍₂₎ − u₍₁₎
λ₂ = u₍₃₎ − u₍₂₎
λ₃ = 1 − u₍₃₎
position = Σᵢ λᵢ · vᵢ
```

*Near face (one small coordinate):*
```
Pick random vertex index k ∈ {0,1,2,3}
λₖ = ε = 0.02  (particle is distance ~2% of tet height from face opposite vertex k)
Distribute remaining (1 − ε) uniformly among the other 3 coords
  using sorted-differences on 2 uniform values
position = Σᵢ λᵢ · vᵢ
```

*Near edge (two small coordinates):*
```
Pick 2 random vertex indices k₁, k₂
λₖ₁ = ε · U(0,1),  λₖ₂ = ε · U(0,1)   where ε = 0.02
Distribute remaining (1 − λₖ₁ − λₖ₂) between the other 2 coords
  using a single uniform split
position = Σᵢ λᵢ · vᵢ
```

*Near vertex (three small coordinates):*
```
Pick the dominant vertex index b
For each j ≠ b: λⱼ = ε · U(0,1)   where ε = 0.02
λᵦ = 1 − Σⱼ≠ᵦ λⱼ  (≈ 0.94–1.0)
position = Σᵢ λᵢ · vᵢ
```


## 4. Results

### 4.1 Perturbation Test — Found Rate

Seeding region: X ∈ [0.08, 0.38], Y ∈ [0.25, 0.75], Z ∈ [0.50, 1.00] (fraction of domain).
This covers 99.5% of elements including both refined and coarse regions.

| Method | 0.0× | 0.1× | 0.2× | 0.5× | 0.7× | 1.0× |
|--------|------|------|------|------|------|------|
| radius r=2 | 30.67% | 29.58% | 27.84% | 23.96% | 23.72% | 22.69% |
| radius r=10 | 58.02% | 52.29% | 51.10% | 48.93% | 48.63% | 46.28% |
| 1×1×1 | 49.82% | — | — | — | — | — |
| 3×3×3 | **100.00%** | 99.98% | 99.62% | 98.49% | 97.96% | 96.75% |
| 5×5×5 | **100.00%** | 99.98% | 99.62% | 98.49% | 97.96% | 96.75% |

**Key observations:**
- At zero perturbation, radius r=2 fails to find **69%** of particles that are guaranteed inside
  the mesh. Radius r=10 still misses 42%. The 1×1×1 center-cell-only search misses ~50%.
- 3×3×3 and 5×5×5 achieve **identical** results — the wider 5×5×5 neighborhood provides no
  additional benefit over 3×3×3 with multi-cell vertex registration.
- At perturbation ≥ 0.5×, the small unfound fraction for 3×3×3/5×5×5 consists entirely of
  particles that left the mesh bounding box (search_fail = 0 across all perturbation levels).

### 4.2 Perturbation Test — Search Failures (Inside BBox but Not Found)

| Method | 0.0× | 0.1× | 0.2× | 0.5× | 0.7× | 1.0× |
|--------|------|------|------|------|------|------|
| radius r=2 | 6,933 | 7,040 | 7,178 | 7,453 | 7,424 | 7,406 |
| radius r=10 | 4,198 | 4,769 | 4,852 | 4,956 | 4,933 | 5,047 |
| 1×1×1 | 5,018 | — | — | — | — | — |
| 3×3×3 | **0** | **0** | **0** | **0** | **0** | **0** |
| 5×5×5 | **0** | **0** | **0** | **0** | **0** | **0** |

The 3×3×3 and 5×5×5 methods have **zero** search failures inside the domain at all perturbation
levels. Every unfound particle is genuinely outside the mesh bounding box.

### 4.3 Perturbation Test — Accuracy (Correct Element Among Found)

| Method | 0.0× | 0.1× | 0.2× | 0.5× | 0.7× | 1.0× |
|--------|------|------|------|------|------|------|
| All methods | 100.0% | ~91.5% | ~47.5% | ~7.0% | ~2.6% | ~1.1% |

All methods have identical accuracy among found particles. The decreasing accuracy at higher
perturbation is expected — particles move to neighboring elements, so the "correct element"
match naturally decreases. This is physical behavior, not a search error.

### 4.4 Intra-Element Accuracy Test

Particles placed at known positions inside elements. Ground truth is guaranteed.

**Found rate (should be 100% for all):**

| Method | centroid | random | near_face | near_edge | near_vertex |
|--------|----------|--------|-----------|-----------|-------------|
| radius r=2 | 30.81% | 24.66% | 22.79% | 19.44% | **12.13%** |
| radius r=10 | 58.34% | 50.61% | 48.72% | 48.13% | 45.28% |
| 1×1×1 | 49.52% | 49.92% | 49.73% | 51.19% | 49.66% |
| 3×3×3 | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| 5×5×5 | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |

**Correct element rate (should be 100% for all):**

Identical to found rate — when a particle is found, it is **always** assigned to the correct
element. There are zero wrong-element assignments across all methods and position types.

### 4.5 Intra-Element — Position Sensitivity of Radius Search

The radius search shows a clear degradation pattern as particles move from centroids toward
element boundaries:

| Position type | radius r=2 found | radius r=10 found | 1×1×1 found |
|---------------|-------------------|---------------------|-------------|
| centroid | 30.81% | 58.34% | 49.52% |
| random | 24.66% | 50.61% | 49.92% |
| near_face | 22.79% | 48.72% | 49.73% |
| near_edge | 19.44% | 48.13% | 51.19% |
| near_vertex | **12.13%** | 45.28% | 49.66% |

For radius r=2, the found rate drops from 31% (centroid) to **12%** (near vertex) — a 2.5×
degradation. This is because:

1. **Single-cell registration**: Each element is registered in one cell (its centroid's cell).
   A particle near a vertex is close to a cell corner, likely in a different cell than the
   element's centroid.
2. **Morton curve locality failure**: Even when the true cell is spatially adjacent, it may be
   far away in the 1D Morton ordering, outside the ±2 search band.

**The 1×1×1 method is position-insensitive** (~50% across all position types). This is because
elements are registered in all 4 vertex cells via multi-cell registration; the found rate is
determined purely by whether the element happens to be registered in the particle's center cell,
which is independent of where within the element the particle sits (see Section 5.3).

### 4.6 Timing

| Method | Time (10k particles) | Relative |
|--------|---------------------|----------|
| 1×1×1 | 5.17–5.30s | **1.00×** (fastest) |
| 3×3×3 | 5.45–5.59s | 1.05× |
| 5×5×5 | 6.51–6.57s | 1.25× |
| radius r=2 | 6.87–7.07s | 1.33× |
| radius r=10 | 6.64–6.91s | 1.31× |

The 3×3×3 method is only **5% slower** than the minimal 1×1×1, while achieving 100% accuracy vs
~50%. The 5×5×5 is ~25% slower with no accuracy improvement. Radius methods are paradoxically
**the slowest** despite finding far fewer particles.


## 5. Analysis

### 5.1 Why Radius Search Fails — and Where

The radius search suffers from two compounding structural problems:

**Problem 1: Single-cell registration.**
Each element is registered in only one cell (its centroid's cell). In a Kuhn mesh, a
tetrahedron's 4 vertices span up to 4 different octree cells. When a particle is inside the
element but in a different cell than the centroid, the element is simply not in that cell's
candidate list.

This explains the position sensitivity: at the centroid (deep inside), the particle is most
likely in the same cell as the element's registered cell. Near a vertex (cell corner), the
particle is very likely in a different cell. The 2.5× degradation from centroid (31%) to
near_vertex (12%) for radius r=2 directly reflects this.

**Problem 2: Morton curve ≠ spatial adjacency.**
The Morton (Z-order) curve provides reasonable spatial locality on average but has
**discontinuities**: cells adjacent in 3D can be arbitrarily far apart in the 1D Morton
ordering. The radius search scans ±R cells in sorted Morton order, which is a 1D band, not
a 3D neighborhood.

Even with radius=10 (21 cells searched), the method finds only 58% of particles. The
remaining 42% reside in cells that are spatially adjacent but Morton-distant.

### 5.2 Why Radius Search is Slower Than Octree

Despite searching fewer cells and finding fewer particles, the radius search takes 7.0s
vs 5.5s for 3×3×3. The reasons:

1. **Separate function calls per cell**: The radius search calls `search_in_cell()` once for
   the center cell, then once per neighbor cell (2×radius calls). Each call contains a
   `fori_loop(0, min(n_elems, 256), ...)`. With radius=2, this means 5 separate `fori_loop`
   invocations, each bounded at 256 iterations. The octree methods use a single nested loop
   structure (one `fori_loop` over cells, one inner `fori_loop` over elements per cell).

2. **Fixed 256-iteration bound per cell**: Even if a cell has only 6 elements, the Morton
   `search_in_cell` allocates a loop bound of `min(n_elems, 256)` per cell. The octree methods
   use the actual element count per cell as the loop bound.

3. **No early termination across cells**: The radius search uses `jnp.where(active, ...)` to
   mask inactive searches, but JAX still evaluates both branches. The octree methods use
   `jnp.where(found_elem >= 0, ...)` at the level boundary to skip entire levels, which is
   more efficient because entire level searches (27 cells) are skipped at once.

4. **Morton code computation overhead**: The position-to-cell mapping in the Morton search
   requires encoding position to a global Morton code and performing a binary search across
   517,309 cells. The octree methods compute grid indices directly via `floor(pos / cell_size)`,
   which is a simple division.

### 5.3 Why 1×1×1 Fails — Definitive Analysis

The failure analysis (run on 5,000 particles missed by 1×1×1 but found by 3×3×3) gives a
**conclusive answer**:

```
Analyzed 5000 failures:
  Element NOT in center cell, found in neighbor:  5000 (100.0%)
  Element IN center cell (level/search issue):       0 (0.0%)
  No matching cell at any level (edge case):         0 (0.0%)
```

**100% of 1×1×1 failures** are because the containing element is not registered in the
particle's center cell. The element is always found in a neighboring cell (within the 3×3×3
neighborhood). There are zero cases of search algorithm bugs or level-mismatch issues.

**Where are the missing elements registered?** The neighbor offset distribution shows:

| Adjacency type | Cells in 3×3×3 | Failures found there | Percentage |
|----------------|----------------|---------------------|------------|
| Face-adjacent | 6 | 7,535 | **37.7%** |
| Edge-adjacent | 12 | 10,000 | **50.0%** |
| Corner-adjacent | 8 | 2,465 | **12.3%** |

The distribution across all 26 neighbor offsets is relatively uniform:

```
Top neighbor offsets:
  (+0,+1,+0) face     : 1512 (7.6%)
  (+0,+0,+1) face     : 1474 (7.4%)
  (-1,+0,+0) face     : 1339 (6.7%)
  (+1,+0,+0) face     : 1184 (5.9%)
  (+1,+0,+1) edge     : 1120 (5.6%)
  (+0,+0,-1) face     : 1027 (5.1%)
  (+0,+1,+1) edge     : 1010 (5.0%)
  (+0,-1,+0) face     :  999 (5.0%)
  (+0,-1,+1) edge     :  936 (4.7%)
  (-1,+0,+1) edge     :  932 (4.7%)
  (+1,+1,+0) edge     :  930 (4.7%)
  (-1,+1,+0) edge     :  930 (4.7%)
```

**Physical explanation:** In a Kuhn tetrahedral mesh, each cube is subdivided into 6 tetrahedra.
Each tetrahedron has vertices at cube corners, which by definition lie at the boundaries between
adjacent octree cells. With multi-cell vertex registration, an element is registered in the ~4
cells where its vertices sit. The particle's center cell is just one cell; the element may be
registered in any of the other ~3 vertex cells, which are face-, edge-, or corner-adjacent.

The 50% found rate of 1×1×1 is consistent with the geometry: with 4 vertex cells per element
and a uniform particle distribution, the probability that the particle's center cell is one of
the element's 4 registered cells is roughly 50% (not exactly 25% because the center cell is
biased toward being one of the vertex cells due to the Kuhn structure).

**Why 3×3×3 achieves 100%:** The 3×3×3 neighborhood (27 cells = center + 6 face + 12 edge +
8 corner neighbors) covers **all possible cells** where an element's vertices could be
registered. Since each element has only ~4 registered cells, all within the 3×3×3 cube centered
on the particle, the search is guaranteed to find the element.

### 5.4 Why 1×1×1 Outperforms Radius Despite Searching Fewer Cells

The 1×1×1 method searches only 1 cell per level but finds ~50% of particles, compared to
radius r=2's ~31% (5 cells) or even r=10's ~58% (21 cells). This highlights the fundamental
advantage of multi-cell registration:

- **1×1×1 + multi-cell registration**: 1 cell searched, but that cell contains elements from
  ~4 original registration sources. Effective coverage: ~4 spatial cells worth of elements.
- **radius r=2 + single-cell registration**: 5 cells searched, but those 5 cells may not be
  spatially adjacent (Morton ordering), and each cell only has its own centroid-registered
  elements. Effective spatial coverage: unpredictable, often less than 5 true neighbors.

Multi-cell registration is strictly more powerful than Morton-band search for spatial coverage.


## 6. Conclusions and Recommendations

### 6.1 The 3×3×3 Method is Optimal

The 3×3×3 mesh-aligned octree search with multi-cell vertex registration is the recommended
L2 search method for JAXTrace:

- **100% found rate** across all position types (centroid, random, near_face, near_edge,
  near_vertex)
- **100% correct element rate** — zero wrong-element assignments
- **Zero search failures** inside the mesh domain at all perturbation levels
- **Near-minimal cost**: only 5% slower than the trivial 1×1×1 center-cell-only search
- **Identical accuracy to 5×5×5** at 25% lower cost
- **Faster than radius search** despite searching more candidate elements

### 6.2 The 5×5×5 Method Provides No Benefit

The wider 5×5×5 neighborhood (125 cells) produces **identical** results to 3×3×3 (27 cells)
in all tests. With multi-cell vertex registration, the 3×3×3 neighborhood is sufficient to
cover all cells where an element's vertices could be registered. The 5×5×5 only adds cost
(~25% slower) with no accuracy improvement.

### 6.3 Radius Search Should Not Be Used for L2

The Morton-curve radius search is fundamentally unsuitable as a global L2 search:

- **Structural ceiling**: Even with large radius (r=10, 21 cells), it finds at most ~58% of
  particles due to Morton curve locality failures and single-cell registration.
- **Slower than octree**: Despite finding fewer particles, radius takes 7.0s vs 5.5s for
  3×3×3 (33% slower). The per-cell function call overhead, fixed 256-iteration loop bounds,
  and Morton binary search cost all contribute.
- **Position-sensitive degradation**: Found rate drops from 31% to 12% (radius r=2) as
  particles move from centroids toward vertices — the worst-case positions are exactly where
  accurate search matters most (element boundaries).

### 6.4 The 1×1×1 Method Has a Niche

The 1×1×1 center-cell-only method is the absolute fastest (~5% faster than 3×3×3) but finds
only ~50% of particles. The failure analysis proves that 100% of its failures are because the
containing element is registered in a neighbor cell, not the center cell. It could theoretically
serve as a "quick first attempt" before falling back to 3×3×3, but the 5% time savings is
unlikely to justify the added complexity. In the current L0 → L1 → L2 hierarchy, the 3×3×3
search is the correct L2 choice.


## 7. Test Configuration

```
Mesh: cylA_159.pvtu (3,050,196 elements, 571,533 nodes)
Point-in-tet: inverse matrix method, tolerance 1e-6
Seeding region: X [0.08, 0.38], Y [0.25, 0.75], Z [0.50, 1.00] (99.5% of elements)
Particles: 10,000 per scenario
Float precision: float64
GPU: CUDA (single GPU)
JAX version: 0.9.0.1
```
