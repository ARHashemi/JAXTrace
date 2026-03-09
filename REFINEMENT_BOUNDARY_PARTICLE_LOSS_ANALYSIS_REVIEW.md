<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# The current algorithm I'm using for searching 4 required k1, ... k4 positions of RK4 and find their corresponding elements for interpolating velocities, is like follow:

1. constructing the octree based on the intrinsic octree of the mesh with Kuhn elements, and assign the elements to the final cubic leaves/cells of the octree (~5-6 element per cube)
2. assign the few non-Kuhn elements to their (face) neighbor cells and in case not possible, consider a separate cell for non-Kuhns.
3. Morton encoding the cubic cells.
4. during the search, the position will be Morton encoded to find the corresponding cell.
5. Perform point-in-tet check for elements assigned to cell and its 3×3×3 neighborhood, by shifting the index, like j+1,j-1,...

This is a very fast method but I'm seeing tetrahedron shaped voids appears during the particle tracking. As depicted in the attached images, it happens specially at the diagonal interface of finer-to-coarser and coarser-to-finer elements.
I listed all potential origins of this kind of failure in the attached REFINEMENT_BOUNDARY_PARTICLE_LOSS_ANALYSIS.md. Evaluate it critically and if there is another additional possible source of this issue or wrong conclusions or missed evidence, explain it. Some of the search codes are attached in mesh_aligned_point_location.py

This is a rich and deeply investigated problem. Let me give you a thorough critical evaluation.

***

## Critical Evaluation of the Analysis

The document is well-structured and the ruled-out hypotheses are solidly evidenced. However, there are **several significant gaps, wrong conclusions, and overlooked mechanisms** that the current analysis misses entirely.

***

## What the Analysis Gets Right

The five ruled-out hypotheses are correctly dismissed based on sound empirical tests. The central paradox — L2 works in isolation but has zero hits inside the full vmapped RK4 — is correctly identified as the crux of the problem.[^1]

***

## Critical Gaps and Missed Causes

### 1. The Most Likely Root Cause Is NOT Investigated: Initial Element Assignment Failure

The document lists "Check Initial Element Assignment" as a *proposed next step* (§8.2), but this is almost certainly **the actual root cause**, not an open hypothesis. Consider the chain of evidence:

- Loss is **19.01% at step 1** and then almost frozen (only 9 more losses over 28 steps)[^1]
- Lost particles receive `elem_id = -1` → zero velocity → **they never move again**
- L2 search has **0 hits** in production — meaning L2 is being reached for those 307,972 miss searches but finding nothing

The critical question the analysis never asks is: **were those 61,595 particles ever found during initial seeding?** If initial assignment itself returned `elem_id = -1` for 19% of particles, then at step 1 the L0 test immediately fails (cached elem is -1), L1 fails (no face neighbors from -1), and L2 is invoked. But L2 is searching for the *step-1 position* (which is the initial position, since the particle never moved), and if that position was already unfindable during seeding, L2 will also fail — zero hits. This would perfectly explain the paradox without any vmap artifact.

The document incorrectly frames this as only "evidence for" hypothesis C, when it is actually the most parsimonious explanation for *all* observations simultaneously: the geometric pattern (boundary particles), the zero L2 hits, the permanence, and the isolation success (isolated tests sample from already-found positions, not from the failing 19%).[^1]

***

### 2. Wrong Conclusion: "Single-Particle Replay Is Ruled Out"

Hypothesis 4.6 is marked **"RULED OUT"** but the evidence does not support that verdict. The 5 initially-lost particles tested in §9–10 of the diagnostic were **sampled from VTU snapshots at known-good locations** — the document itself acknowledges this caveat.  You never tested isolated L2 on the *actual initial positions of the failing particles*. The ruling-out is premature.[^1]

***

### 3. Unnoticed Structural Bug: `lax.fori_loop` with Dynamic Bound Over Nested `lax.cond`

Looking at the code in `mesh_aligned_point_location.py`, there is a subtle and dangerous pattern:[^2]

```python
ntotest = jnp.where(should_search, jnp.minimum(n_elements_in_cell, tests_remaining), 0)
...
lax.fori_loop(0, ntotest, test_element, ...)
```

`lax.fori_loop` requires its upper bound to be a **static integer or a concrete value** in some JAX versions, but here it receives a **dynamic `ntotest`** computed with `jnp.where`. Under `vmap`, the `lax.fori_loop` body is compiled for the maximum possible bound (determined at trace time), but the *actual* number of iterations must be a concrete integer at trace time in some XLA backends. If XLA resolves `ntotest` differently inside the full RK4 vmapped graph (e.g., if `jnp.minimum` gets constant-folded to 0 due to `tests_remaining` being traced as exhausted), **all element tests in L2 would silently skip**, producing zero hits — exactly what is observed.[^2]

This is a concrete, testable mechanism the analysis completely overlooks.

***

### 4. Unnoticed: `tests_remaining = max_tests - total_tests` Can Silently Zero Out L2

In the single-level version (`search_mesh_aligned_octree_single`), there is:

```python
tests_remaining = max_tests - total_tests
ntotest = jnp.where(should_search, jnp.minimum(n_elements_in_cell, tests_remaining), 0)
```

If the RK4 pipeline calls L2 sequentially for k1, k2, k3, k4 with a **shared or incorrectly threaded `total_tests` carry**, and that carry is already at `max_tests` from a previous sub-step, then `tests_remaining = 0` and `ntotest = 0` for all subsequent sub-steps. This would produce zero L2 hits — not because L2 is broken but because the budget is silently exhausted. The analysis never checks whether `max_tests` is being correctly reset between RK4 sub-steps.[^2][^1]

***

### 5. The 3×3×3 Neighborhood May Actually *Miss* Elements at Coarse-Fine Interfaces

The analysis concludes cross-level reachability is "RULED OUT" based on 528 synthetic cross-boundary test positions. But the test used **element centroids** as test positions, which are always interior to their element. The real failure mode is different: a particle sitting exactly **on or near the geometric interface between a level-14 and level-13 region** may lie inside a level-13 element whose *vertices* were registered to level-13 cells, but whose geometric extent crosses into the level-14 grid. At the level-13 grid resolution, the particle's grid cell (i,j,k) may compute to a cell that exists in the octree; but all 27 neighbors at that level may *not* contain the straddling element, because the element's vertices at level-13 resolution fall one cell outside the 3×3×3 window when the particle is near the element's edge — not its centroid.[^1][^2]

The 528-position test was not designed to stress this: it tested positions *crossing* the boundary, not positions at the **geometric corner or edge** of a coarse element whose vertices span a distance greater than 3 cell widths at the fine level.

***

### 6. Non-Kuhn Element Handling Is a Probable Cause of the Geometric Pattern

Non-Kuhn elements "borrow their Kuhn face-neighbor's cell size and level."  At refinement boundaries, non-Kuhn elements are disproportionately concentrated (the Kuhn decomposition of a hex mesh produces non-Kuhn tets at refinement transition regions). If the borrowed cell size and level are those of the *coarser* neighbor, the non-Kuhn element is registered at a coarser grid resolution. A particle at the fine-resolution boundary inside a non-Kuhn element would compute its grid indices at level-14 (finest), look in the fine 3×3×3 neighborhood, but the element is registered only in the *coarser* level cells — which are not searched at the same (i,j,k). The multi-level loop would eventually reach that coarser level, but only if the element's coarser-level cell is within the 3×3×3 of the *particle's coarser-level* grid index, which is not guaranteed when the particle is at the fine edge of the non-Kuhn element. This is geometrically consistent with the **diagonal tetrahedral void pattern** visible in the images, since non-Kuhn tets at refinement interfaces are precisely diagonal.[^1]

***

### 7. The `maxtests=600` Cap Is a Real Risk at Refinement Boundaries

With 8 levels × 27 cells × ~5 elements/cell = 1,080 possible tests, the cap of `maxtests=600` can truncate the search before all levels are visited. The analysis never measures *how many levels are actually reached* when L2 is called from production. If coarse levels (7, 8, 9) where the non-Kuhn elements are registered are never reached due to the cap being hit on fine-level tests, those particles would be permanently missed.[^2]

***

## Summary of Issues

| \# | Issue | Status in Document | Severity |
| :-- | :-- | :-- | :-- |
| 1 | Initial seeding `elem_id=-1` as root cause | Listed as future step, not identified as primary | **Critical** |
| 2 | Single-particle replay verdict | Prematurely "RULED OUT" | **High** |
| 3 | Dynamic `ntotest` in `lax.fori_loop` silently zeroing L2 | Not investigated | **High** |
| 4 | `tests_remaining` budget not reset between RK4 sub-steps | Not investigated | **High** |
| 5 | 3×3×3 missing elements near edges (not centroids) of coarse elements | Incorrectly RULED OUT | **Medium** |
| 6 | Non-Kuhn diagonal elements at refinement boundaries | Not investigated | **Medium** |
| 7 | `maxtests=600` truncation before coarse levels reached | Not investigated | **Medium** |

The single most actionable diagnosis is: **directly inspect the initial element assignment output for all 324,000 particles before the first RK4 step**, count how many have `elem_id = -1`, and verify whether that count is exactly ≈61,595 (≈19%). If yes, the entire "vmap paradox" dissolves — the problem is in the seeding phase, not in the RK4 pipeline at all.
<span style="display:none">[^3][^4]</span>

<div align="center">⁂</div>

[^1]: REFINEMENT_BOUNDARY_PARTICLE_LOSS_ANALYSIS.md

[^2]: mesh_aligned_point_location.py

[^3]: Screenshot_20260224_154355.jpg

[^4]: Screenshot_20260225_150437.jpg

