# Particle Loss at Refinement Boundaries: Comprehensive Analysis

## 1. Executive Summary

After extensive diagnostic testing, **all initially hypothesized causes have been ruled out**:

- Float32 vs Float64 precision mismatch: **RULED OUT** (3x3x3 covers off-by-one)
- Cross-level search coverage gaps: **RULED OUT** (0/528 cross-boundary positions missed)
- `lax.cond` compilation artifact: **RULED OUT** (jnp.where produces identical results)
- L2 search function broken under vmap: **RULED OUT** (32/32 correct in isolation)
- L1 search causing issues: **RULED OUT** (L1 on/off produces identical loss)

**Key production observation**: L2 search has **0 hits** in production vmapped RK4, yet succeeds
at 100% in every isolated test. 19.01% of particles (61,595 / 324,000) are lost at step 1 and
the loss is essentially permanent (only ~9 additional losses over 29 steps).

**Remaining hypothesis**: The particle loss originates from the interaction between the full
RK4 integration graph and vmap compilation, NOT from the L2 search function itself. The exact
mechanism remains unidentified.

---

## 2. How the 3x3x3 Cell Search Works

### 2.1 Algorithm Overview

The search in `mesh_aligned_point_location.py` works as follows:

```
FOR level_idx = 0..7  (levels 14, 13, 12, ..., 7):
    cell_size = level_cell_sizes[level]              <-- ONE representative size per level
    i_base = floor(pos[0] / cell_size[0])            <-- float32 division on GPU
    j_base = floor(pos[1] / cell_size[1])
    k_base = floor(pos[2] / cell_size[2])

    FOR di, dj, dk in [-1, 0, 1]^3:                  <-- 27 cells in 3x3x3 neighborhood
        i = i_base + di;  j = j_base + dj;  k = k_base + dk
        morton = encode_morton_3d(i + offset, j + offset, k + offset)
        cell_idx = binary_search(morton, level)       <-- must match BOTH morton AND level
        test all elements in cell_idx via point-in-tet
```

Total: 8 levels x 27 cells = 216 cell lookups per particle (max ~600 point-in-tet tests).

### 2.2 How Neighbors Are Determined

Neighbors are determined **purely geometrically**: given the particle's grid cell `(i, j, k)` at
a given level, the algorithm searches all 27 cells `(i+di, j+dj, k+dk)` for `di, dj, dk in {-1, 0, 1}`.

This is done **independently at each refinement level**. There is no cross-level neighbor lookup.
Each level has its own grid, its own cell sizes, and its own 3x3x3 neighborhood.

### 2.3 How Elements Are Registered (Extraction Phase)

In `mesh_aligned_octree_vertex_multi.py`, element registration is done on CPU with **float64**:

```python
# CPU extraction (float64 precision)
for vertex in element_vertices:
    i = int(np.floor(vertex[0] / cell_size[0]))   # float64 / float64
    j = int(np.floor(vertex[1] / cell_size[1]))
    k = int(np.floor(vertex[2] / cell_size[2]))
    morton = encode_morton_3d(i + offset, j + offset, k + offset)
    cell_key = (morton, level)
    register element in cell_key
```

Each element is registered in ALL cells touched by its 4 vertices (~4 cells per element).
Non-Kuhn elements borrow their Kuhn face-neighbor's cell size and level.

---

## 3. The Refinement Level Structure

From diagnostic logs, the mesh has 7 refinement levels:

| Level | Elements    | Percentage | Cell Size X    | Cell Size Y    | Cell Size Z    |
|-------|------------|------------|----------------|----------------|----------------|
| 14    | 2,599,528  | 85.26%     | 7.812e-5       | **7.986e-5**   | 7.812e-5       |
| 13    | 381,987    | 12.53%     | 1.5625e-4      | **1.5972e-4**  | 1.5625e-4      |
| 12    | 43,888     | 1.44%      | 3.125e-4       | **3.1944e-4**  | 3.125e-4       |
| 11    | 14,949     | 0.49%      | ~6.25e-4       | ~6.389e-4      | ~6.25e-4       |
| 10    | 4,289      | 0.14%      | ~1.25e-3       | ~1.278e-3      | ~1.25e-3       |
| 9     | 1,594      | 0.05%      | ~2.5e-3        | ~2.556e-3      | ~2.5e-3        |
| 8     | 839        | 0.03%      | ~5.0e-3        | ~5.111e-3      | ~5.0e-3        |

**Key observation**: X and Z cell sizes are exact powers of 2 (perfectly representable in float32).
**Y cell sizes are irrational in binary** (0.00007986... cannot be exactly represented in any
floating-point format). This is where float32 precision was initially suspected.

---

## 4. Hypotheses Investigated

### 4.1 Hypothesis: Float32 vs Float64 Precision Mismatch

#### Theory

Element registration and search use **different precisions** for grid index computation:

| Phase | Location | Precision | Cell Size Type |
|-------|----------|-----------|----------------|
| Registration | `mesh_aligned_octree_vertex_multi.py:200-203` | CPU float64 | numpy float64 |
| GPU Upload | `mesh_aligned_octree_gpu.py:441-450` | - | converted to float32 |
| Search | `mesh_aligned_point_location.py:442-444` | GPU float32 | jax float32 |

The conversion happens at `mesh_aligned_octree_gpu.py:450`:
```python
level_cell_sizes_gpu = jnp.array(level_cell_sizes_cpu, dtype=jnp.float32)
```

If float32 division produces a different `floor()` result than float64, the search looks in the
wrong cell.

#### Evidence

**Test 1: Grid index comparison at boundary element centroids** (`diagnose_refinement_boundary_loss.py` Section 3)

Compared float32 vs float64 grid indices for 500 boundary element centroids:

| Level | Tests | Mismatches | Rate | Axis |
|-------|-------|------------|------|------|
| 8     | 75    | 0          | 0.00% | - |
| 9     | 250   | 0          | 0.00% | - |
| 10    | 425   | 0          | 0.00% | - |
| 11    | 250   | 10         | 4.00% | X only |

All 10 mismatches were off-by-one in X axis only (delta = +1), at level 11.

**Test 2: Systematic float32 boundary sweep** (Section 4)

Tested positions at exact cell boundaries along each axis:

| Level | Mismatches | Rate |
|-------|------------|------|
| 8     | 2,518/6,015 | 41.86% |
| 9     | 2,420/6,015 | 40.23% |
| 10    | 2,392/6,015 | 39.77% |
| 11    | 2,424/6,015 | 40.30% |
| 12    | 2,403/6,015 | 39.95% |
| 13    | 2,296/6,015 | 38.17% |
| 14    | 2,374/6,015 | 39.47% |
| **Total** | **16,827/42,105** | **39.96%** |

At exact cell boundaries, ~40% of positions have float32 vs float64 disagreement.

**Test 3: Does 3x3x3 compensate?** (Section 5)

GPU L2 search at 200 boundary element centroids:
- Found (correct element): **200**
- Found (different element): 0
- NOT FOUND: **0**

**Test 4: Cross-boundary sweep** (Section 6)

528 positions crossing from fine to coarse elements:
- Misses: **0** (0.00%)

#### Verdict: **RULED OUT**

Float32 precision does cause grid index mismatches (~40% at exact boundaries), but the 3x3x3
neighborhood search **fully compensates** for off-by-one errors. No misses were observed at
any boundary position.

---

### 4.2 Hypothesis: Cross-Level Element Reachability Gaps

#### Theory

At a refinement boundary (e.g., level-14 meets level-13), a particle might be inside a level-13
element while the 3x3x3 search at level-13 maps to the wrong cell. Since each level has its own
independent grid, the element might only be reachable through the correct cell at its own level.

#### Evidence

- 0/200 boundary centroids missed (Section 5 of diagnostic)
- 0/528 cross-boundary positions missed (Section 6 of diagnostic)
- The search iterates all 8 levels (14 down to 7), each with its own 3x3x3 neighborhood
- At the element's own level, the element is always within the 3x3x3 reach

#### Verdict: **RULED OUT**

Cross-level search works correctly. The multi-level iteration covers all 8 levels and the 3x3x3
at each level provides sufficient coverage.

---

### 4.3 Hypothesis: `lax.cond` Compilation Artifact Under vmap

#### Theory

JAX's `lax.cond` under `vmap` is lowered to `lax.select`, which evaluates both branches. This
could cause side effects or incorrect state propagation in the deeply nested conditional structure
of the L2 search (4 levels of nested `lax.cond`).

#### Evidence

**Test: Created `jnp.where` version of L2 search**

Rewrote `search_mesh_aligned_octree_multi_local` to use explicit `jnp.where` masking instead of
`lax.cond`. Named `search_mesh_aligned_octree_multi_local_where`.

**Production comparison** (`search_stats.csv`):

The lax.cond and jnp.where versions produce **byte-for-byte identical** search statistics:

| Step | n_active | n_lost | L0 hits | L1 hits | L2 hits | Misses | Miss % |
|------|----------|--------|---------|---------|---------|--------|--------|
| 1    | 262,405  | 61,595 | 1,310,278 | 1,750 | 0     | 307,972 | 19.01% |
| 2    | 262,405  | 61,595 | 1,310,270 | 1,755 | 0     | 307,975 | 19.01% |
| 3    | 262,405  | 61,595 | 1,306,604 | 5,419 | 2     | 307,975 | 19.01% |
| ...  | ...      | ...    | ...       | ...   | ...   | ...     | ...    |
| 29   | 262,396  | 61,604 | 1,310,231 | 1,749 | 0     | 308,020 | 19.01% |

Every single number is identical between the two versions across all 29 steps.

#### Verdict: **RULED OUT**

`lax.cond` lowering to `lax.select` under vmap is NOT causing the particle loss. The `jnp.where`
version, which explicitly avoids `lax.cond`, produces exactly the same results.

---

### 4.4 Hypothesis: L2 Search Function Broken Under vmap

#### Theory

The L2 search function itself might produce incorrect results when vmapped, due to indexing
issues, carry state bugs, or XLA optimization artifacts.

#### Evidence

**Test: Isolated L2 search under vmap** (`diagnose_vmapped_l2_search.py` Sections 4-5)

32 test positions vmapped through L2 search:
- All 32 matched the expected element IDs
- Result: 32/32 correct

**Test: L2 search of known positions**

Single-particle L2 search tested at known element centroids:
- Result: 100% success rate

#### Verdict: **RULED OUT**

The L2 search function produces correct results under vmap when tested in isolation.

---

### 4.5 Hypothesis: L1 Search (Face-Neighbor Multi-Hop) Causes Issues

#### Theory

The L1 search (5-hop face-neighbor traversal) might return incorrect element IDs or corrupt
state, causing downstream L2 search to receive wrong cached element information.

#### Evidence

**Test: L1 enabled vs disabled** (`diagnose_vmapped_l2_search_enhanced.py`)

Production runs with ENABLE_L1_SEARCH=True vs ENABLE_L1_SEARCH=False:
- Identical particle loss in both cases
- L1 contributes only ~0.11% of successful searches

#### Verdict: **RULED OUT**

L1 search has negligible impact and does not cause the particle loss.

---

### 4.6 Hypothesis: Single-Particle RK4 Search Failure

#### Theory

Even in single-particle (non-vmapped) mode, the RK4 sub-step positions might land at locations
where L2 search fails.

#### Evidence

**Test: Single-particle RK4 sub-step replay** (`diagnose_vmapped_l2_search.py` Section 8)

15 particles replayed through full RK4 sub-steps (k1, k2, k3, k4, final):
- Total sub-step searches: 75
- Misses: **0/75**
- All 15 particles tracked successfully

**Test: Enhanced sub-step replay** (`diagnose_vmapped_l2_search_enhanced.py` Sections 9-10)

21 particles from production output:
- 5 initially-lost + 16 surviving particles
- 21/21 tracked successfully in single-particle mode

#### Verdict: **RULED OUT (for tested positions)**

Single-particle replay succeeds at 100% for the tested positions. However, these test positions
were sampled from VTU snapshots at known-good locations, not necessarily at the exact positions
where production vmapped tracking fails.

---

## 5. Critical Production Observations

### 5.1 L2 Search Has Zero Hits in Production

The most striking finding from the production search statistics:

```
Per-step breakdown (324,000 particles x 5 sub-steps = 1,620,000 searches/step):
  L0 (cached element):  ~1,310,000  (80.88%)
  L1 (face neighbors):      ~1,750  ( 0.11%)
  L2 (global search):           ~0  ( 0.00%)  <-- NEVER succeeds!
  Miss (not found):       ~307,972  (19.01%)
```

**L2 search essentially never succeeds in production**, yet it works perfectly in every isolated
test. This is the central paradox of the investigation.

### 5.2 Loss Is Immediate and Permanent

- Step 1: 61,595 particles lost (19.01%)
- Step 29: 61,604 particles lost (19.01%)
- Only 9 additional particles lost over 28 steps

This means nearly all loss occurs at the **very first RK4 step**. Once lost (elem_id = -1),
particles receive zero velocity and remain frozen permanently.

### 5.3 Loss Pattern Is Geometric

Visualization shows particle loss concentrated at **straight lines/planes** that exactly match
the refinement boundary locations in the structured Kuhn mesh. The loss is NOT random and NOT
scattered throughout the domain.

### 5.4 lax.cond and jnp.where Are Bit-Identical

Both implementations produce the exact same search statistics down to the last integer, proving
the loss mechanism is deterministic and independent of how JAX handles conditionals.

---

## 6. Complete Evidence Table

| # | Test | Result | Rules Out |
|---|------|--------|-----------|
| 1 | L2 search under vmap (isolated, 32 positions) | 32/32 correct | L2 function itself |
| 2 | L1 enabled vs disabled | Identical loss | L1 search |
| 3 | Single-particle RK4 sub-step replay (15 particles) | 0/75 misses | Single-particle search at tested positions |
| 4 | Enhanced single-particle replay (21 particles) | 21/21 tracked | Single-particle tracking |
| 5 | lax.cond vs jnp.where production run | Bit-identical stats | lax.cond compilation artifact |
| 6 | Float32 vs float64 grid indices (500 boundary centroids) | 10 mismatches but 0 L2 misses | Float32 precision |
| 7 | Systematic boundary sweep (42,105 positions) | 40% f32/f64 mismatch but 0/200 L2 misses | Float32 causing search failure |
| 8 | Cross-boundary position sweep | 0/528 misses | Cross-level reachability |
| 9 | Production search_stats.csv | L2_hits=0, miss=19.01% | - |
| 10 | Visualization of lost particles | Loss at refinement boundary lines | Random/scattered loss pattern |

---

## 7. What Remains Unexplained

### 7.1 The Central Paradox

| Context | L2 Result |
|---------|-----------|
| L2 search alone, single particle | 100% success |
| L2 search alone, vmapped | 100% success (32/32) |
| Full RK4, single particle | 100% success (0/75 misses) |
| **Full RK4, vmapped (production)** | **0% L2 success, 19% total loss** |

The **only** configuration that fails is the complete vmapped RK4 production pipeline. Every
component works correctly when tested in isolation or in partial combinations.

### 7.2 Possible Remaining Causes

#### A. XLA Optimization Artifact in Full RK4 Graph

The full vmapped RK4 graph is extremely large: 5 sub-steps, each with L0/L1/L2 dispatch,
point-in-tet tests, velocity interpolation, position updates, and carry state. XLA may apply
optimizations (common subexpression elimination, buffer reuse, operation reordering) that
create subtle bugs in this specific graph topology.

**Evidence for**: Only the full vmapped graph fails. All subcomponents work independently.
**Evidence against**: lax.cond and jnp.where produce identical results, suggesting the issue
is not in how XLA handles conditionals.

#### B. Carry State Corruption Under vmap

The `lax.scan` loop carries state (positions, element IDs, velocities) across RK4 steps. Under
vmap, each particle has independent carry state, but XLA may compile the scan body in a way that
causes incorrect state propagation between sub-steps.

**Evidence for**: Single-particle works, vmapped fails. The loss is immediate (step 1).
**Evidence against**: No direct evidence of carry corruption has been observed.

#### C. Position/Element State Inconsistency at Initialization

If 19% of particles start with an incorrect cached element ID (e.g., elem_id = -1 or wrong
element), the first L0 test fails, L1 fails (no valid face neighbors from elem_id = -1), and
L2 is called. If L2 then also fails for these specific positions (which may be outside the mesh
or at exact boundary positions), the particles are permanently lost.

**Evidence for**: Loss is immediate and permanent. 19% is close to the fraction of particles
that might be seeded outside the mesh or at exact element boundaries.
**Evidence against**: Isolated L2 search works for boundary positions.

#### D. L2 Search Receives Corrupted Position Data Under vmap

The position passed to L2 search in the full vmapped RK4 might differ from the position computed
in isolation, due to intermediate velocity interpolation or position update steps interacting
with vmap in unexpected ways.

**Evidence for**: L2 works on correct positions but might receive wrong positions in production.
**Evidence against**: No direct evidence of position corruption has been observed.

---

## 8. Proposed Next Steps

### 8.1 Diagnose Production L2 Failure Positions

Extract the actual positions where L2 fails in production (not boundary centroids or
cross-boundary sweeps, but the exact `pos` values passed to L2 during vmapped RK4). Test these
specific positions in isolated L2 search.

### 8.2 Check Initial Element Assignment

Verify that the initial `elem_id` assignment for all 324,000 particles is correct. If 19%
start with elem_id = -1 (never found), the entire cascade follows naturally.

### 8.3 Instrument the Full RK4 Graph

Add debug outputs to the vmapped RK4 to capture, for a subset of particles:
- Position at each sub-step
- Cached elem_id at each sub-step
- L0/L1/L2 dispatch decisions
- Actual position passed to L2

Compare these with single-particle replay to identify where divergence occurs.

### 8.4 Test With Smaller Batch Size

Run vmapped RK4 with batch_size=1 (effectively single-particle but through the vmap codepath)
to determine if the issue is vmap itself or the batch size.

---

## 9. Summary

After comprehensive testing, all initially suspected causes (float32 precision, cross-level
reachability, lax.cond compilation, L2 function correctness, L1 interference) have been
**ruled out** by empirical evidence.

The particle loss at refinement boundaries is real and reproducible (19.01%, geometric pattern),
but the root cause remains in the interaction between the full RK4 integration pipeline and
vmap compilation. The key anomaly is that L2 search has **zero** successful hits in production
while achieving 100% success in every isolated test.

The next diagnostic steps should focus on capturing the **exact state** (positions, element IDs,
dispatch decisions) inside the production vmapped RK4 to identify where the divergence from
single-particle behavior occurs.
