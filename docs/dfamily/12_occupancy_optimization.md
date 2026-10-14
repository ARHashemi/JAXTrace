# Reducing octree occupancy: what works and what does not

The cell-overflow fix set the static search bound to the mesh's real max
occupancy (79 on D2), which costs ~1.7× per step. This evaluates how to lower
that bound instead of paying for it.

Search cost is **`n_levels × 27 × MAX_ELEMS_PER_CELL`**, and two properties of
the kernel drive everything below:

1. **Every level is walked unconditionally** — `fori_loop(0, n_levels)` — so
   adding a level costs ~11% at 9 levels.
2. **The bound is the MAX, not the mean**, because the loop is static. A change
   that lowers mean occupancy but leaves the tail intact buys *nothing*.
3. **The inner body has no early exit.** `point_in_tet` is computed on every
   iteration and then masked, so a cell holding 3 elements still pays the full
   bound.

---

## Option 1: split over-full cells to a finer level — REJECTED by measurement

Simulated by re-registering every element one level finer, exactly as the
builder does (Kuhn by parent cube, non-Kuhn by AABB overlap):

| metric | now | one level finer | |
|---|---:|---:|---|
| cells | 5,982,836 | 449,438,364 | **75.1×** |
| mean occupancy | 14.22 | **2.33** | good |
| **MAX occupancy** | **79** | **64** | **only −19%** |
| CSR entries | 85,078,209 | 1,045,441,569 | **12.3×** |
| tests per query | 17,064 | 15,552 | 0.91× |

**Net: 9% faster queries for 12.3× the memory** — ~3.9 GiB of CSR plus ~13.8
GiB of cell metadata, which does not fit a 32 GB GPU and would dominate a LUMI
node.

### Why the mean collapses but the max does not

Occupancy has two contributors that scale **oppositely** under refinement:

- **Kuhn** elements are registered by **parent cube**, one cell each. Halve the
  pitch and each lands in one of 8 sub-cells, so their per-cell count drops
  ~8×. This is what collapses the mean 14.22 → 2.33.
- **non-Kuhn** elements are registered by **AABB overlap**, into every cell the
  AABB touches. An AABB is a **fixed physical size**, so halving the pitch
  makes it span ~8× more cells — while the number of AABBs covering any given
  *point* is unchanged. Occupancy at that point does not fall.

Measured composition of the over-full cells (3,000 sampled, 143,116 entries):

| | entries | share |
|---|---:|---:|
| Kuhn | 95 | **0.1%** |
| non-Kuhn | 143,021 | **99.9%** |

**99.9% of what fills an over-full cell is precisely the part refinement cannot
reduce.** Scale of the tail: 804,653 cells over 32 holding 38,410,022 entries
= 45.1% of the whole CSR.

### On the Morton/indexing questions

These were never the obstacle. `(morton, level)` already keys every cell, 21
bits per axis leaves vast headroom, and `find_cell_by_morton_and_level` has 25
binary-search iterations for a 23-iteration problem. Split cells would index
and decode correctly. Note also that splitting to the **same** level is not
meaningful: `(morton, level)` is the unique key, so two cells at the same level
and grid position *are* the same cell. Refinement necessarily means a new
level, which is why the +1 level cost is unavoidable.

---

## Option 2: exact tet-vs-box registration — a real 1.41x, measured

AABB overlap is a **loose proxy**: a tetrahedron occupies at most ~1/6 of its
AABB, so many cells an AABB touches are not actually intersected by the tet.
Those registrations inflate occupancy, force the bound up, and are paid for by
every query at every step.

Measured on the production CSR by re-testing every element of 4,000 sampled
over-full cells against that cell's real box with an exact 13-axis
separating-axis test:

| | before | after |
|---|---:|---:|
| occupancy of over-full cells | mean 47.7, median 48, max 72 | **mean 34.0, median 35, max 56** |
| registrations kept | — | **71.3%** |
| cells dropping to ≤32 | — | 1,153 / 4,000 (28.8%) |
| **static bound** | **79** | **56** |
| tests per query | 17,064 | **12,096** → **1.41× faster** |
| CSR entries | 85.1M | ~61M |

In context:

| configuration | tests/query | correctness |
|---|---:|---|
| original (bound 32) | 8,640 | **loses 4% of particles** |
| shipped fix (bound 79) | 17,064 | correct |
| + exact tet-box (bound 56) | 12,096 | correct |

**Exact registration recovers about half the throughput the correctness fix
cost, while keeping correctness.** It is an optimisation, not a correctness
fix — the shipped bound already recovers 93% of the lost particles.

### Why not more

The 1,974 elements registered in >1000 cells each (11.2% of the whole CSR) are
**genuinely large non-Kuhn tetrahedra**: median AABB diagonal 1.35 mm against a
mesh median of 0.0869 mm, i.e. 15.6×, sitting in a region whose cell pitch is
0.0469 mm. Such a tet spans ~29 cells per axis, so its AABB covers ~24,000
cells and even an exact test leaves ~4,000. Their registered level (14) matches
their own inferred level, so nothing is misplaced — the structure is correct,
just expensive. No registration strategy removes them; only the AABB slack can
be pruned.

### Implementation caveat

One 13-axis SAT test per candidate cell means ~1.1 G axis projections across
~85 M candidate registrations. In Python that would dominate a build already
taking 16–25 minutes, so it must be vectorised (numpy over all candidate cells
of an element at once) to be practical. That is real work, not a one-line
change.

The test itself is validated against seven hand-checked cases before use:

| case | expected | result |
|---|---|---|
| tet inside a large box | intersect | PASS |
| box far away | separate | PASS |
| **AABB corner the tet misses** | **separate** | **PASS** ← the waste removed |
| box overlapping the tet body | intersect | PASS |
| box touching one vertex only | intersect | PASS ← must not drop |
| sliver tet, box straddling it | intersect | PASS |
| sliver tet, box above it | separate | PASS |

Correctness dominates here: dropping a cell a tet really does touch would
reintroduce the symptom just fixed. Scripts:
`scripts/dfamily/eval_tet_box_registration.py` (per-element waste) and the
production-CSR re-test used for the table above.

## Option 3: early exit in the inner loop

The inner body computes `point_in_tet` unconditionally and masks the result, so
short cells pay the full bound. Skipping the arithmetic when `~valid` would
make cost scale with actual occupancy rather than with the bound.

Whether this helps on GPU is **not obvious** and must be measured: XLA may
already predicate the masked work, and under `vmap` there is no per-lane
control flow, so a lane that could exit early still waits for the batch.

---

## Why a dynamic trip count does not help

Worth recording, because it is the intuitive first suggestion.
`search_mesh_aligned_octree_multi_local_where` uses `fori_loop(0, n_to_test)`
with the cell's real occupancy, so it never truncates. But **`jax.vmap` has no
per-lane control flow**: a traced trip count lowers to a *single* batched
while-loop that must keep iterating until the last lane finishes, masking lanes
that are already done. Verified directly — `vmap` over trip counts [1, 5, 50]
produces one while-loop with one condition.

With 50,000 particles per batch and 13.45% of cells over 32, the probability
that some particle in the batch touches a 79-element cell is ~1, so the dynamic
loop runs ~79 iterations **anyway** — and loses the unrolling. The static bound
is the better choice; that is why `static_where` exists.

---

## Recommendation

1. **Keep the derived static bound** (shipped). Correct on any mesh, no larger
   than necessary — 79 on D2, 31 on A1, so A/B/C get a *shorter* loop than the
   old constant of 32.
2. **Exact tet-box registration is worth implementing**: measured 1.41x on
   query cost (bound 79 -> 56) and 85.1M -> ~61M CSR entries, which recovers
   roughly half of what the correctness fix cost. It needs the SAT test
   vectorised first, so it is a few hours of work rather than a one-liner.
3. **Do not split cells.** 9% faster for 12.3× memory.
4. Re-measure throughput on both the workstation and LUMI after any change;
   the LUMI D2 baseline with the shipped fix is job 22815786.
