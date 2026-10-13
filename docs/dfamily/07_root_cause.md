# The origin of the D-family lost-particle issue

**Status: root cause identified, measured, fixed at its origin, and the fix
verified on the real D2 mesh.** Both registration passes and the search now
share one lattice by construction (§5). All three lattice-mismatch failure
categories measure **zero** after the fix; the only particles still unhosted
are the 192 that have no containing element anywhere in the mesh, which is not
a defect (§5.2).

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
- Pass 1, Kuhn elements: *originally* recorded each element's **own**
  `cell_size`; now the level's lookup pitch (§5).
- Pass 2, non-Kuhn elements: enumerates the element's AABB and files it under a
  key computed from a cell size. *Originally* `nbr_size`, the borrowed
  neighbour's size; now the level's lookup pitch (§5).

Both are described as they *were* below, since that is what the measurements in
§4 were taken against.

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

## 5. The fix

**Every registration path now files elements on the one lattice the search
divides by.** The pitch is derived in a classification sweep that runs *before*
either pass registers anything:

```python
# pass 0: classify, then derive the lattice both passes will register on
for elem_id in range(n_elements):
    cell_size, level = find_axis_aligned_edges_single(...)
    if np.any(cell_size == 0):
        non_kuhn_ids.append(elem_id); continue
    _level_sizes.setdefault(int(level), []).append(cell_size)

lookup_pitch = {lv: np.median(np.asarray(v), axis=0)
                for lv, v in _level_sizes.items()}
```

derived exactly as `upload_mesh_aligned_octree_to_gpu` derives
`level_cell_sizes`, so the two agree **by construction** rather than by
coincidence. Pass 1, the Pass 2 AABB path and the legacy centroid path all
index on `lookup_pitch[level]`.

This arrived in two steps, and the intermediate measurement is what proved
where the remainder lived.

### 5.1 Step one — Pass 2 only

Aligning the non-Kuhn AABB enumeration alone cut the cell mismatch 18.2% →
16.43% and recovered **1,164 of 2,000** previously-lost particles (58.2%). The
836 that remained separated perfectly:

| residual failure | n |
|---|---:|
| no containing element | 192 |
| host on **mismatched** cell, **Kuhn** (Pass 1) | **644** |
| host on mismatched cell, non-Kuhn (Pass 2) | **0** |
| host on matching cell | **0** |

Zero residual failures on a Pass-2 cell — the Pass-2 fix was complete on its
own terms — and zero on a matching cell, reproducing §4.1's separation. The
index error kept the same signature: median 2, p90 2, Δ ≤ 1 = 0.0%.

That isolated the remainder to `parent_cube.py` Pass 1, which recorded each
**Kuhn** element's own `cell_size` and wrote `cell_metadata` only
`if cell_key not in cell_metadata`, so a cell first created by a Kuhn element
kept that element's size. Kuhn sizes also vary at a level on an anisotropic
mesh, so the disagreement was inherent to Pass 1.

### 5.2 Step two — one lattice for both passes, and the verification

Hence pass 0. Re-running the same 2,000 previously-lost particles against the
fully-fixed octree on the real D2 mesh (10,807,677 elements, 2,802,871 nodes):

```
RESULT: of 2,000 previously-lost particles, the FIXED octree
  finds a host for : 1,808 (90.4%)
  still loses      :   192 ( 9.6%)

of the 192 still lost:
  no containing element            : 192
  host on MISMATCHED cell, KUHN    : 0   <- Pass 1 origin
  host on MISMATCHED cell, non-Kuhn: 0   <- Pass 2 origin
  host on MATCHING cell            : 0

VERDICT: residual failures are NOT lattice-related.
```

**The lattice defect is eliminated.** Recovery went 58.2% → **90.4%**, the 644
Pass-1 failures went to **0**, and all three mismatch categories are now zero.
The 192 that remain are exactly the particles that have no containing element
anywhere in the mesh — they are outside the domain, which is the expected and
correct outcome, not a search failure. This is the predicted result in §5.1
confirmed on real data.

### 5.3 Cost

Measured on the same mesh, after step one:

| | before | after step one | after step two |
|---|---:|---:|---:|
| cells | 5,986,798 | 5,984,124 | 5,982,836 |
| elements/cell (mean) | 14.2 | 14.22 | 14.22 |
| **max occupancy** | **95** | **79** | **79** |
| CSR entries | 85,087,578 | 85,078,209 | — |

Worst-case per-query work *improves* and step two costs nothing further: cells
move by 0.02%, mean load and max occupancy are unchanged. Step two adds one
extra sweep over the
elements at build time and nothing at query time: no per-step branch, no kernel
change, and a no-op on meshes whose levels hold a single size, where the median
*is* each element's own size (A/B/C). Compare widening the search to 5×5×5,
which would recover the same 98.6% at **4.6× the cost of every query**.

### 5.4 Regression guard

`scripts/dfamily/test_lattice_invariant.py` needs no case data and runs in a
second. It builds a synthetic isotropic mesh and a synthetic anisotropic one,
and asserts that every registered cell is reachable from the lattice the search
divides by. The anisotropic mesh is placed ~128 cells from the origin, because
a pitch disagreement is a *relative* error whose absolute index error grows
with distance — at the origin it is 0 and only far out does it exceed ±1.

| mesh | without the fix | with the fix |
|---|---|---|
| isotropic cubic (A/B/C-like) | PASS, 0 unreachable | PASS, 0 unreachable |
| anisotropic cuboid (D-like) | **FAIL, 66.67% unreachable, max Δ 32** | **PASS, 0 unreachable** |

The isotropic row passing either way is the direct confirmation that A/B/C are
unaffected.

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
