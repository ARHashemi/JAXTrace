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

## Option 2: exact tet-vs-box registration — the promising direction

What refinement cannot fix, precision can. AABB overlap is a **loose proxy**: a
tetrahedron occupies at most ~1/6 of its AABB, so most cells an AABB touches
are not actually intersected by the tet. Those registrations inflate occupancy,
force the bound up, and are paid for by every query at every step.

Replacing the AABB enumeration with an exact tet-box test at registration time:

- lowers **both** the mean and the **max**, so the bound actually drops;
- is **build-time only** — the search kernel is untouched;
- is **exact**, so coverage is preserved and no element is lost.

The last point is the binding constraint: dropping a cell a tet really does
touch would reintroduce the cell-overflow bug's symptom. The test used is a
13-axis separating-axis theorem (3 box normals, 4 tet face normals, 6 edge-edge
cross products), validated against hand-checked cases before use:

| case | expected | result |
|---|---|---|
| tet inside a large box | intersect | PASS |
| box far away | separate | PASS |
| **AABB corner the tet misses** | **separate** | **PASS** ← the waste removed |
| box overlapping the tet body | intersect | PASS |
| box touching one vertex only | intersect | PASS ← must not drop |
| sliver tet, box straddling it | intersect | PASS |
| sliver tet, box above it | separate | PASS |

Evaluation script: `scripts/dfamily/eval_tet_box_registration.py`.

---

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
2. **Evaluate exact tet-box registration** as the real occupancy fix. It is the
   only option measured so far that attacks the tail rather than trading it for
   memory.
3. **Do not split cells.** 9% faster for 12.3× memory.
4. Re-measure throughput on both the workstation and LUMI after any change;
   the LUMI D2 baseline with the shipped fix is job 22815786.
