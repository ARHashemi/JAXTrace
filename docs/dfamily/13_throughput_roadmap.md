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

## 4. Early exit under `vmap` — MEASURED, AND IT DOES NOT WORK

This section replaces an earlier draft that listed level-loop and element-loop
early exit as promising. **They cannot work**, and the measurement is
unambiguous.

`jax.vmap` has no per-lane control flow: a `lax.while_loop` becomes one batched
loop whose predicate is reduced across the batch, so it runs until the **last**
lane finishes. Measured, 200,000 lanes, per-lane trip-count targets:

| batch composition | mean trips | max trips | time |
|---|---:|---:|---:|
| all lanes 1 | 1.0 | 1 | **0.18 ms** |
| **1 lane 64, rest 1** | **1.0** | **64** | **3.83 ms** |
| half 1, half 64 | 32.5 | 64 | 3.84 ms |
| all lanes 64 | 64.0 | 64 | 3.70 ms |

**Cost tracks the maximum, not the mean.** One slow lane in 200,000 costs the
same as all lanes being slow — a 21× penalty for 0.0005% of the work.

Worse, the `while_loop` is **~4× slower than the equivalent `fori_loop`** at
the same trip count (3.70 ms vs 0.91 ms at 64 iterations), because it loses
unrolling and adds a batch-wide predicate evaluation per iteration.

So both of the following are rejected:

- **level-loop early exit** — cannot work; only the *reorder* below is real;
- **element-loop early exit** — cannot work, and would be slower than the
  current masked `fori_loop`.

This matches the long-standing JAX position on data-dependent control flow
under `vmap`; nothing has changed to enable it.

### 4.1 Reordering the level loop alone — NO GAIN, measured

Reordering was proposed on the assumption that masked iterations are cheap.
**They are not.** Same trip count, varying where the host is found:

| | time | vs always-work |
|---|---:|---:|
| all 8 iterations do real work | 0.476 ms | — |
| masked, host found on iteration 0 | 0.455 ms | 1.05× |
| masked, host found on iteration 1 | 0.500 ms | 0.95× |
| masked, host found on iteration 7 | 0.534 ms | 0.89× |

All within noise. `jnp.where` masks the **result**, not the work, so the
arithmetic for every level executes regardless. Changing the order changes
nothing. `lax.cond` is not an escape either — measured identical to `where`
(0.648 vs 0.645 ms at 8 iterations), confirming it lowers to a select under
`vmap`.

**The only mechanism that removes work is reducing the STATIC trip count:**

| trip count | time | |
|---:|---:|---|
| 8 (current) | 0.645 ms | — |
| 2 | 0.202 ms | **3.20×** |

Those iterations genuinely do not exist, so the saving is real.

### 4.1b Where particles actually are — the surprise that makes this tractable

Cell counts and *particle* counts are different populations, and the
difference is extreme. Probing 20,000 positions from the **production seed
grid** against the real octree:

```
levels walked by the kernel, in order: [15, 14, 13, 12, 11, 10, 9, 8]
host LEVEL distribution : level 8 -> 20,000 (100.00%)
ITERATION found at      : iteration 7 -> 20,000 (100.00%)
```

**100% of seeded particles are hosted at level 8 — the last of the eight
levels walked.** Every particle pays all eight iterations to find a host it
would have found immediately.

This is not a probe artefact. The production seed box is
`x −19.9 … −11.1 mm`, i.e. entirely **behind** the tool: 0% of seeds fall
within r ≤ 9 mm, so none start in the refined region. Level 8 (pitch 3.9 mm,
460 cells) tiles the coarse background that fills most of the domain *volume*,
while level 14 (pitch 0.061 mm, 5.88M cells) tiles only the small refined
region near the tool. Cell count is dominated by refinement; volume is not.

**Implication: a one-level fast tier could be ~8× for the coarse-region
majority**, with the full eight-level kernel as a second pass for the rest —
better than the 3.2× that an 8→2 reduction gives.

**Caveat that must be measured first.** These are *seed* positions. The
particles that matter are those that reach the tool, and they sit at fine
levels — a fast tier tuned for the coarse region would route exactly the
interesting particles to the slow path. The level distribution is needed
**during** a run, not only at seeding, before this is worth building.

### 4.2 What else works: sort particles into homogeneous batches — ~1.35×

If per-lane exit is impossible, the GPU answer is to **regroup rather than
branch**: sort particles so each batch contains lanes with similar work, which
lowers each batch's maximum toward its own mean.

Measured on a realistic skew (98% of particles need 2 levels, 2% need 3–8):

| batching | time | gain |
|---|---:|---:|
| unsorted, one batch of 200,000 | 0.61 ms | — |
| **sorted into 2 chunks** (maxes 2, 8) | **0.45 ms** | **1.35×** |
| sorted into 4 chunks | 0.51 ms | 1.20× |
| sorted into 8 chunks | 0.87 ms | 0.70× |

Two chunks is the optimum; beyond that, per-chunk kernel-launch overhead
dominates. Note this was measured on a synthetic `while_loop`; applying it to
the real search means sorting by *predicted* level, which is itself an extra
pass, so the realised gain would be smaller. **Worth measuring on the real
kernel before committing to it.**

### 4.3 Shrinking the 3×3×3 neighbourhood — ALREADY REJECTED for the paper

Removed from the candidate list. This was tested for the MALMO paper on both
timing and coverage: a 1×1×1 (base-cell-only) search gave **no significant
timing gain and failed to locate more than 40% of elements**. The 3×3×3
neighbourhood is required because an element can straddle a cell boundary, and
the cost is dominated by the element loop rather than the cell walk.

## 5. Comparison table

Baseline = the shipped correct configuration (bound 79): **17,064 tests per
query**, loss 0.20%.

| # | candidate | gain | memory | effort | risk | status |
|---|---|---|---|---|---|---|
| 3.1 | **exact tet-vs-box registration** | **1.41× measured** | **−28% CSR** | ~4–8 h | low | **VIABLE — best option** |
| 4.1 | reorder the level loop | **no gain** | none | — | — | **REJECTED** — masked iterations cost full price |
| 4.1b | **two-tier kernel: 1-level fast path + full fallback** | *up to ~8×* on coarse-region particles | none | ~1–2 days | medium | **MOST PROMISING** — needs in-run level distribution first |
| 4.2 | sort particles into homogeneous batches | ~1.35× on a synthetic proxy | none | ~1 day | medium | **NEEDS REAL MEASUREMENT** |
| 2.1 | split cells one level finer | 0.91× | **+12.3× CSR** | ~1 day | low | **REJECTED** — memory |
| 2.2 | dynamic per-cell trip count | ~1.0× | none | ~1 h | low | **REJECTED** — `vmap` batches the trip count |
| 4.0a | level-loop **early exit** | **cannot work** | none | — | — | **REJECTED** — `vmap` has no per-lane control flow |
| 4.0b | element-loop **early exit** | **cannot work, ~4× slower** | none | — | — | **REJECTED** — same, plus loses unrolling |
| 4.3 | shrink 3×3×3 → 1×1×1 | no timing gain | none | — | — | **REJECTED** — paper tests: >40% location failures |

**Realistic ceiling: 1.41× banked** (3.1), with 4.1b the only candidate that
could go materially beyond it. Two earlier drafts of this table were wrong:
one claimed ~5.6× by assuming per-lane early exit was available, the other
~1.14× from reordering. Both are now measured and rejected.

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

1. **Measure the in-run host-level distribution** (§4.1b). Cheap — add a
   level histogram to a short tracking run. It decides whether the two-tier
   kernel is worth 1–2 days: if most particle-steps are hosted at one or two
   levels throughout the run, it is the biggest available win; if the
   distribution spreads once particles reach the tool, it is not.
2. **Implement exact tet-box registration** (§3.1). 1.41× and −28% CSR,
   low-risk, independent of 4.1b so the two multiply. The SAT test is written
   and validated; it needs vectorising.
3. **Build the two-tier kernel** (§4.1b) only if step 1 supports it.
4. **Measure particle sorting** (§4.2) last; the synthetic 1.35× will not
   transfer fully, since sorting by predicted level costs an extra pass.
5. **Do not revisit** per-lane early exit, `lax.cond`, level reordering,
   dynamic trip counts, cell splitting, or a 1×1×1 neighbourhood. All six are
   measured dead ends, documented above with the numbers.

Decide against the LUMI baseline (job 22815786) rather than the workstation
numbers: on LUMI snapshot loading is ~2 h of a ~10 h run, so a 1.6× gain on the
tracking phase alone moves total wall time by much less than it does on the
workstation. **If the LUMI run is acceptable as-is, none of this is needed.**

## 7. Reproducing any of these

| script | what it measures |
|---|---|
| `scripts/dfamily/eval_cell_splitting.py` | §2.1 — occupancy composition and a direct re-registration one level finer |
| `scripts/dfamily/eval_tet_box_registration.py` | §3.1 — per-element AABB waste (note: uses the `level=14` default, so its headline figure overstates the saving; the production-CSR re-test is the trustworthy one) |
| `scripts/dfamily/verify_kernel_unchanged.py` | guards that any kernel edit leaves the production step bit-identical when the new path is off |
| `scripts/dfamily/bench_vmap_early_exit.py` | §4 — proves `while_loop` under `vmap` costs the batch maximum, and is ~4× slower than `fori_loop`. Runs in seconds, needs no case data |
| `scripts/dfamily/bench_batch_sorting.py` | §4.2 — the sorted-batch proxy measurement. Runs in seconds, needs no case data |
| `scripts/dfamily/bench_mask_cost.py` | §4.1 — proves a masked `fori_loop` iteration costs full price, so reordering gains nothing. Seconds, no case data |
| `scripts/dfamily/bench_trip_count.py` | §4.1 — `lax.cond` vs `jnp.where` (identical), and the 8→2 trip-count reduction (3.2×). Seconds, no case data |
| `scripts/dfamily/probe_host_levels.py` | §4.1b — at which level and iteration production seeds actually find their host |

All are read-only and take a case `.gid` path. Each needs a ~16–25 min octree
build on D2 before it reports.

**A note on method.** Two figures in this investigation were initially
overstated because an evaluation script diverged from the production code path:
a 2.19× registration saving (the script used the `level=14` default where
production borrows a neighbour's level) and a `MAX_ELEMS_PER_CELL` refutation
(the check examined `multi_local_where`, which the `parent_cube` path does not
call). When measuring any of the above, re-test against the **production**
structure or the production function, not a reimplementation.
