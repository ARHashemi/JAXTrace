# Search throughput: what was tested, what works, what is left

**Purpose.** The cell-overflow fix made the D-family search correct (loss
4.07% → 0.20%) at ~1.7× the per-step cost. This document records every
optimisation candidate examined, what each measured, which are rejected and
why, and the effort each remaining one needs — so the work can be resumed
without re-deriving any of it.

Status as of 2026-10-14. Baseline for all figures: **D2**, 10,807,677 elements,
octree 5,982,836 cells, mean occupancy 14.22, **max occupancy 79**, 8 levels
walked.

---

## 1. The cost model — and why it has three independent factors

```
tests per L2 query  =  n_levels  ×  27 cells  ×  MAX_ELEMS_PER_CELL
                    =      8     ×     27     ×        79           =  17,064
```

Every factor is paid in full on every query, because of three kernel
properties:

1. **The level loop runs to completion.** `fori_loop(0, n_levels)` with
   `should_search` masking the *result* once found — not skipping the work.
2. **The 3×3×3 neighbourhood is always fully walked**, for the same reason.
3. **The element loop is statically bounded and has no early exit.**
   `point_in_tet` is computed on every iteration and then masked, so a cell
   holding 3 elements still pays the full bound.

Each factor is therefore an independent optimisation target. **This is the key
insight for anyone resuming:** the registration-side work (factor 3) is capped
at ~1.4×, while the level loop (factor 1) is the larger untouched prize.

---

## 2. Measured and rejected

### 2.1 Split over-full cells to a finer level — REJECTED

Simulated by re-registering all 10.8M elements one level finer, exactly as the
builder does.

| metric | now | one level finer | |
|---|---:|---:|---|
| cells | 5,982,836 | 449,438,364 | **75.1×** |
| mean occupancy | 14.22 | 2.33 | good |
| **max occupancy** | **79** | **64** | **only −19%** |
| CSR entries | 85,078,209 | 1,045,441,569 | **12.3×** |
| tests per query | 17,064 | 15,552 | 0.91× |

**Verdict: 9% faster for 12.3× memory** — ~3.9 GiB of CSR plus ~13.8 GiB of
cell metadata. Does not fit a 32 GB GPU; would dominate a LUMI node.

**Why the mean falls but the max does not.** Occupancy has two contributors
that scale *oppositely* under refinement:

- **Kuhn** elements register by **parent cube**, one cell each → halving the
  pitch puts each in one of 8 sub-cells → their per-cell count drops ~8×. This
  is what collapses the mean.
- **non-Kuhn** elements register by **AABB overlap** into every cell the AABB
  touches. An AABB is a **fixed physical size**, so halving the pitch makes it
  span ~8× more cells while the number of AABBs covering any given *point* is
  unchanged. Occupancy at that point does not fall.

Measured composition of the over-full cells (3,000 sampled, 143,116 entries):
**Kuhn 95 (0.1%)** vs **non-Kuhn 143,021 (99.9%)**. So 99.9% of the tail is
precisely the refinement-immune part.

**On Morton indexing** (asked during the review): never the obstacle.
`(morton, level)` already keys every cell, 21 bits per axis leaves vast
headroom, and `find_cell_by_morton_and_level` runs 25 binary-search iterations
for a 23-iteration problem. Split cells would index and decode correctly. Note
also that splitting to the **same** level is not meaningful — `(morton, level)`
is the unique key, so two cells at the same level and grid position *are* the
same cell. Refinement necessarily adds a level, which is why the +1 level cost
is unavoidable.

### 2.2 Dynamic per-cell trip count — REJECTED

`search_mesh_aligned_octree_multi_local_where` uses `fori_loop(0, n_to_test)`
with the cell's real occupancy, so it never truncates. The intuition is that
threads finishing early would avoid waiting.

**It does not work that way under `vmap`.** Verified directly: `jax.vmap` over
trip counts `[1, 5, 50]` lowers to **one** batched while-loop with one
condition. There is no per-lane control flow, so the loop must keep iterating
until the *last* lane finishes, masking lanes already done — the effective trip
count is `max` across the batch.

With 50,000 particles per batch and 13.45% of cells over 32, the probability
that some particle touches a 79-element cell is ≈1, so the dynamic loop runs
~79 iterations **anyway** and additionally loses the unrolling. This is why
`static_where` exists.

---

## 3. Measured and viable

### 3.1 Exact tet-vs-box registration — 1.41×, VIABLE

AABB overlap is a loose proxy: a tet occupies at most ~1/6 of its AABB, so many
cells it is registered in are never actually intersected.

Measured on the **production CSR** by re-testing every element of 4,000 sampled
over-full cells against that cell's real box with an exact 13-axis
separating-axis test:

| | before | after |
|---|---:|---:|
| occupancy of over-full cells | mean 47.7, median 48, max 72 | **mean 34.0, median 35, max 56** |
| registrations kept | — | **71.3%** |
| cells dropping to ≤32 | — | 1,153 / 4,000 (28.8%) |
| **static bound** | **79** | **56** |
| tests per query | 17,064 | **12,096 → 1.41×** |
| CSR entries | 85.1M | ~61M |

**Why the win is bounded.** The 1,974 elements registered in >1000 cells each
(11.2% of the whole CSR) are *genuinely large* non-Kuhn tetrahedra: median AABB
diagonal **1.35 mm** against a mesh median of **0.0869 mm** (15.6×), sitting
where the cell pitch is 0.0469 mm. Such a tet spans ~29 cells per axis, so its
AABB covers ~24,000 cells and even an exact test leaves ~4,000. Their
registered level (14) matches their own inferred level, so nothing is
misplaced — the structure is correct, just expensive. No registration strategy
removes them; only the AABB slack is recoverable.

**The mesh is healthy**, so this is not a mesh-quality problem: AABB diagonal
p50 0.0869 mm, p99 0.3465 mm, max 5.196 mm (60× median), and **zero** elements
with volume < 1e-16.

**Effort: ~4–8 hours.** The SAT test must be vectorised (numpy over all
candidate cells of an element at once) before it is practical: ~1.1 G axis
projections across ~85 M candidate registrations would otherwise dominate a
build already taking 16–25 minutes. The test is already written and validated
against seven hand-checked cases, including the AABB corner a tet misses
(correctly separate) and a box touching one vertex (correctly intersecting) —
correctness dominates here, since dropping a cell a tet does touch would
reintroduce the symptom just fixed.

---

## 4. Identified but UNMEASURED — the larger prize

### 4.1 Early exit / reordering of the level loop — potential ~4×

The loop walks `level = max_level - level_idx`, i.e. **15, 14, 13, 12, 11, 10,
9, 8** — all eight, every query. But the level population is extremely skewed:

| level | cells | share |
|---:|---:|---:|
| 15 | 580 | 0.01% |
| **14** | **5,875,326** | **98.2%** |
| 13 | 81,899 | 1.37% |
| 8–12 | 25,031 | 0.42% |

For almost every particle the host is found on **iteration 2 of 8**, and the
remaining six iterations are masked but still execute their full
`27 × bound` point-in-tet arithmetic. That is ~75% of the level loop wasted on
a typical particle.

Options, in increasing ambition:
- **reorder** so the most populated level is tried first (14 before 15) — a
  one-line change, saves one wasted iteration;
- **true early exit**, e.g. `lax.while_loop` over levels with a found
  predicate. Under `vmap` this collapses to the batch maximum (see §2.2), so
  the gain depends on how uniform the batch is — particles in the refined
  region are nearly all at level 14, so a batch may genuinely exit early;
- **skip levels a position cannot be in**, computed from the position and the
  per-level grid extents before entering the loop.

**Potential up to ~4×, and it stacks with §3.1** (they are independent
factors). Combined ceiling ≈ 5.6×. **Unmeasured — this is the first thing to
test when resuming.**

### 4.2 Early exit inside the element loop — potential unclear

`point_in_tet` is computed unconditionally and masked, so short cells pay the
full bound. Skipping the arithmetic when `~valid` would make cost scale with
actual occupancy (mean 14.22) rather than the bound (79) — potentially ~5×.

**But it may already happen.** XLA may predicate the masked work away, and
under `vmap` a lane that could exit early still waits for its batch. Needs an
HLO inspection plus a microbenchmark before any estimate is credible.

### 4.3 Reduce the 3×3×3 neighbourhood — potential large, risk high

27 cells are walked because an element can straddle a cell boundary. With
**exact** registration an element is in every cell it genuinely touches, so the
base cell alone might suffice — up to ~27×.

**High risk**: this is exactly the search reach whose absence caused the
original lattice bug. It must not be attempted before §3.1 ships and is
verified, and any trial needs the full loss-measurement harness, not just a
throughput number.

---

## 5. Comparison table

Baseline = the shipped correct configuration (bound 79): **17,064 tests per
query**, loss 0.20%.

| # | candidate | measured gain | memory | effort | risk | status |
|---|---|---|---|---|---|---|
| 2.1 | split cells one level finer | **0.91×** (9% faster) | **+12.3× CSR** | ~1 day | low | **REJECTED** — memory |
| 2.2 | dynamic per-cell trip count | **~1.0×** (no gain) | none | ~1 h | low | **REJECTED** — `vmap` batches the trip count |
| 3.1 | exact tet-vs-box registration | **1.41×** | **−28% CSR** | ~4–8 h | low (test validated) | **VIABLE** — ready to implement |
| 4.1 | level-loop early exit / reorder | *up to ~4×* | none | ~1 h reorder, ~1 day exit | medium | **UNMEASURED** — test first |
| 4.2 | element-loop early exit | *up to ~5×, may be 1×* | none | ~2 h to measure | low | **UNMEASURED** — may already be done by XLA |
| 4.3 | shrink 3×3×3 → base cell | *up to ~27×* | none | ~2 days + loss harness | **high** | **UNMEASURED** — only after 3.1 ships |

Stacking: 3.1 and 4.1 are independent factors of the same product, so they
multiply — **~5.6× if both land**. 4.3 would multiply again but carries the
correctness risk that caused the original bug.

### Cost context

| configuration | tests/query | correctness |
|---|---:|---|
| original (bound 32) | 8,640 | **loses 4.07% of particles** |
| **shipped fix (bound 79)** | **17,064** | **correct, loss 0.20%** |
| + exact tet-box (bound 56) | 12,096 | correct |
| + level early exit (hypothetical) | ~3,000 | correct, if it works |

The shipped fix costs ~1.97× the broken version. Exact registration recovers
about half of that; the level loop could recover all of it and more.

---

## 6. Recommended order when resuming

1. **Measure 4.1 first** (level-loop reorder, one line). Highest ratio of
   potential gain to effort, and the measurement is cheap.
2. **Measure 4.2** (HLO inspection + microbenchmark). Could be free, could be
   nothing — knowing which costs two hours.
3. **Implement 3.1** if 1–2 fall short. 1.41× is banked and low-risk, but it is
   the most work of the three.
4. **Do not attempt 4.3** until 3.1 is shipped and verified, and then only with
   a loss-measurement harness attached.

Decide against the LUMI baseline (job 22815786) rather than the workstation
numbers: on LUMI the snapshot loading is ~2 h of a ~10 h run, so a 1.41× gain
on the tracking phase alone moves total wall time much less than it does on the
workstation.

---

## 7. Reproducing any of these

| script | what it measures |
|---|---|
| `scripts/dfamily/eval_cell_splitting.py` | §2.1 — occupancy composition and a direct re-registration one level finer |
| `scripts/dfamily/eval_tet_box_registration.py` | §3.1 — per-element AABB waste (note: uses the `level=14` default, so its headline figure overstates the saving; the production-CSR re-test is the trustworthy one) |
| `scripts/dfamily/verify_kernel_unchanged.py` | guards that any kernel edit leaves the production step bit-identical when the new path is off |

All are read-only and take a case `.gid` path. Each needs a ~16–25 min octree
build on D2 before it reports.

**A note on method.** Two figures in this investigation were initially
overstated because an evaluation script diverged from the production code path:
a 2.19× registration saving (the script used the `level=14` default where
production borrows a neighbour's level) and a `MAX_ELEMS_PER_CELL` refutation
(the check examined `multi_local_where`, which the `parent_cube` path does not
call). When measuring any of the above, re-test against the **production**
structure or the production function, not a reimplementation.
