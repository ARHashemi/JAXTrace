# The origin of the D-family lost-particle issue

**Status: root cause identified and measured. A partial fix is in; it is
confirmed incomplete and the residual is characterised below.**

Supersedes the causal account in [`05_root_cause_and_fix.md`](05_root_cause_and_fix.md),
which attributed the losses to the per-level pitch alone. That was a real but
*minority* mechanism. This document records what the measurements actually
show.

---

## 1. The mechanism, in one paragraph

The octree files an element under a cell key computed from **one** cell size,
and the search looks that cell up using a **different** cell size. Where the
two disagree, the search computes a grid index ~2 cells away from the cell that
actually holds the element. The 3×3×3 neighbourhood reaches only ±1, so the
host cell is never tested and the particle is reported as having no host
(`ElementID = -1`). It is a *lattice disagreement between registration and
lookup*, not a geometry, velocity, level-set or Morton problem.

---

## 2. Why the D family and not A/B/C

| mesh | non-Kuhn fraction | cell anisotropy | particles lost |
|---|---:|---|---:|
| A1 | 0.84% | 1.000 exactly, one size per level | **0** |
| A/B/C (15 meshes) | 0.74–1.05% | uniform | 0 |
| **D1–D4** | **50.6–59.3%** | median 1.197, max 1.457 | thousands |

"Non-Kuhn" means the tetrahedron has no axis-aligned edge on some axis
(`parent_cube.py:174`, `np.any(cell_size == 0)`), so the octree cannot place it
by its own parent cube and must borrow a neighbour's grid. The A/B/C meshes are
~1% non-Kuhn and cubic, so every level holds a single cell size and
registration and lookup agree trivially. The D meshes are >50% non-Kuhn with
cuboid cells, so the sizes form a continuum and the two lattices diverge.

This is the answer to "what is different about the D family": **not** finer
refinement, but a structurally different mesh.

---

## 3. The two lattices

**Registration**, `mesh_aligned_octree_parent_cube.py`:
- Pass 1, Kuhn elements (line 195): records each element's **own** `cell_size`.
- Pass 2, non-Kuhn elements (lines ~311-333): enumerates the element's AABB and
  files it under a key computed from a cell size. *Originally* `nbr_size`, the
  borrowed neighbour's size; now the level's lookup pitch (§5).

**Lookup**, `mesh_aligned_octree_gpu.py:452-487` and
`mesh_aligned_point_location.py:548-553`:
```python
level_cell_sizes[level] = median(all cell sizes at that level)   # per axis
i = floor(pos[0] / level_cell_sizes[level][0])                   # etc.
```
then a Morton code is formed and the sorted cell array is **binary-searched for
an exact `(morton, level)` match**. A cell is testable only if that exact key
exists — the octree stores only non-empty cells.

On D2, level 14 holds 5,875,915 cells (98% of the octree) whose stored sizes
run 3.5753e-05, 3.5839e-05, 3.6146e-05 … against a lookup pitch of 5.6117e-05.
A continuum, not a discrete family, so no single representative can match them
all.

---

## 4. The evidence

All from the D2 annulus run on the workstation (120,000 particles seeded in the
two measured loss shells, 600 steps), with the **real production search**
re-run on each sampled lost particle to confirm it still fails.

### 4.1 Attribution, 2,000 sampled lost particles

| group | n | share |
|---|---:|---:|
| reproduce as lost | 858 | |
| → no containing element | 192 | not a bug |
| → host on a **mismatched** lattice | **666** | **100.0%** |
| → host on a **matching** lattice | **0** | **0.0%** |

Perfect separation. 18.2% of all cells were mismatched, so failures are
enriched 5.5×, but the zero is the stronger statement: no failure occurs where
the lattices agree.

### 4.2 How far out of reach

Index error between the search's base cell and the nearest cell holding the
true host, **at the particle's own position**:

| | Δ ≤ 1 | Δ ≤ 2 | Δ ≤ 3 | median | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
| failing particles | **0.0%** | **98.6%** | 98.8% | 2 | 2 | 62 |

Every failure is beyond the current ±1 reach, and almost all sit in the second
ring. A thin tail reaches 62 cells (~1.3%).

Probing cell **centres** instead gave Δ ≤ 1 for 99.1% — which would lose
nothing. Centre probing is a lower bound: at a real position near a cell face
`floor()` tips one further. That reconciles the two figures, which initially
appeared to contradict each other.

### 4.3 Ruled out by measurement — do not re-investigate

| candidate | finding |
|---|---|
| mesh changes during tool revolution | **all 166 timesteps byte-identical**; the mesh is Eulerian, the tool rotates *through* it, so building the octree once is valid |
| Morton mis-mapping to far cells | **0 of 2,000** codes inconsistent with the stored grid index |
| host not registered anywhere | **0** — the `JAXTrace_stable` orphan-drop problem is fixed in the dev repo |
| host's level outside the walked range | **0** |
| point-in-tet predicate | **2 of 2,000**, at min barycentric −3.3e-04 — far beyond any tolerance change |
| `MAX_ELEMS_PER_CELL` truncation | not applicable: the `_where` variant tracking calls tests `n_elems_in_cell`, unbounded |
| `aabb` registration as the cure | cuts loss 56% but cannot fix a wrong index; and 0.2 step/s vs 0.5 |

---

## 5. The fix, and what it does *not* yet cover

Pass 2 now enumerates each non-Kuhn AABB on the **lookup lattice** —
`median(cell sizes at that level)`, derived exactly as the GPU upload derives
`level_cell_sizes` — instead of `nbr_size`.

**Cost: none.** Measured on the same mesh:

| | before | after | change |
|---|---:|---:|---:|
| cells | 5,986,798 | 5,984,124 | −0.04% |
| elements/cell (mean) | 14.2 | 14.22 | ~same |
| **max occupancy** | **95** | **79** | **−17%** |
| CSR entries | 85,087,578 | 85,078,209 | −0.01% |

Worst-case per-query work *improves*. It is build-time only: no per-step
branch, no kernel change, and a no-op on meshes whose levels hold a single size
(A/B/C). Compare 5×5×5, which would recover the same 98.6% at **4.6× the cost
of every query**.

**But the mismatch did not go to zero: 18.2% → 16.43%.**

The reason is `parent_cube.py:195`: Pass 1 records each **Kuhn** element's own
`cell_size`, and `cell_metadata` is written only `if cell_key not in
cell_metadata`. A cell first created by a Kuhn element keeps that element's
size. Since Kuhn elements at a level also vary in size on an anisotropic mesh,
the residual disagreement is inherent to Pass 1 and my Pass 2 change could not
address it.

**Open question:** whether the residual 16.43% includes the cells that actually
lose particles. The verification run decides it. If losses persist, the same
treatment must be applied to Pass 1 — or, more robustly, the lookup pitch
should be fixed per level *before* either pass registers anything, so both
passes and the search share one lattice by construction.

---

## 6. Separate defects found along the way

These are independent of the lattice issue and are recorded so they are not
rediscovered.

**Seed-box bug.** The production array seeds with
`--seed-source box-frac --seed-fraction 0.0 0.3 0.0 1.0 0.0 1.0`, i.e.
fractions of the mesh **bounding box**. The D bbox extends above the workpiece
(D1 z −6.00…+1.35 mm vs A1 −6.00…0.00), so ~18.4% of seeds land in empty air
and fail initial assignment: predicted 18,367 of 100,000, observed 17,481. Fix
by clamping the upper z fraction per case, or seeding from the FEMUSS node set.

**Stale build in the case scripts.** Each case's `run_jaxtrace.sh` points at
`JAXTrace_stable`, which predates the `orphan_fallback` fix and **drops**
non-Kuhn elements with no Kuhn neighbour (>1.3M on D2). Production used the dev
repo. A log containing `no Kuhn neighbour, skipped` is on the wrong build.

**`zero_vel` silently degrades trajectories.** RK4 combines its stages
unconditionally (`rk4_fully_fused_timedep.py:771`) and a failed substage
contributes zero for that term only (line 569), so a particle with
`ElementID = -1` is **not frozen** — it advances on whichever stages succeeded,
integrating a partially zeroed velocity biased toward stagnation. Only
`skip_step` freezes a particle. Agreed action: switch to `last_valid_vel`,
which reuses the previous stage's velocity and preserves continuity. See
[`06_pending_actions.md`](06_pending_actions.md).

**Performance, for the record.** The 1.1 → 0.5 step/s change between the June D
runs and now is **octree occupancy**, not velocity-snapshot count: D3 built
2,030,413 cells at 4.7 elem/cell, D2 builds 5,986,798 at 14.2 — 3.0× more
point-in-tet tests per query, explaining a 2.2× slowdown. The denser octree
also loses far less (36.6% → 7.5%), so it is a large net improvement.
