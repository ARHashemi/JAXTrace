# D-family lost particles: root cause and fix

**Status: root cause identified and fixed. Full-length verification in progress.**

This document supersedes the hypotheses in [`01_diagnosis.md`](01_diagnosis.md).
That file is kept as the record of how the investigation started, but two of its
candidate explanations turned out to be wrong, and its suggested remedy — widening
the search bands — cannot work. See [§7](#7-what-01_diagnosismd-got-wrong).

---

## 1. The short version

Particles in the D-family (D-ConcavityTilt) cases stopped moving near the tool
because **the point-location search computed the wrong octree cell index for
them**. It divided each coordinate by a per-level cell size taken from the
*first* cell encountered at that level. On the D meshes that cell was an
unrepresentative cubic outlier, while 99.4% of cells are **cuboid** — taller in
z than wide in x,y. The z index was therefore inflated by a factor of 1.197, an
error that grows with depth and reaches 14 cells at z = −4 mm. The true host
cell lay outside the 3×3×3 neighbourhood the search inspects, so the particle
was reported as having no host (`ElementID = -1`) and froze.

The fix is one line: take the **per-axis median** of the cell sizes at each
level instead of the first cell's size.

```python
# jaxtrace/gpu/search/mesh_aligned_octree_gpu.py
level_cell_sizes_cpu[level] = np.median(level_sizes, axis=0)   # was: level_sizes[0]
```

It is a **build-time change only** — nothing in the time-marching path touches
cell sizes, so there is no per-step branch and no throughput cost.

---

## 2. Why the D family and not A/B/C

The A, B and C meshes are cubic; the D meshes are cuboid. Measured from each
case's own octree build:

| mesh | distinct size-triples per level | per-cell aspect ratio | particles lost |
|---|---|---|---:|
| A1 | **1 at every level** | 1.000 exactly | **0** |
| D2 | 20 → 345 per level | median **1.197**, max 1.457 | thousands |

D2's level 14 holds 5.88M cells — 98% of the octree:

```
5,438,614 cells (92.56%)  dx,dy,dz = 4.6875e-05  4.6875e-05  5.6117e-05   cuboid
   33,991 cells ( 0.58%)  dx,dy,dz = 4.6875e-05  4.6875e-05  4.6875e-05   cubic
```

The old code picked a *cubic* cell's size as the pitch for the whole level, so
the 99.4% majority were indexed on the wrong z lattice.

Crucially the cuboid sizes form their own clean hierarchy — `dx` and `dz` each
halve exactly per level and `dz/dx` is a constant 1.1972 — so one per-axis pitch
per level is well defined. No extra octree levels are needed, which matters
because the search scans levels linearly with no early exit.

---

## 3. Why `level` collapses different sizes onto one level

`mesh_aligned_octree_single_cell.py:101-104` derives the level from the **mean**
of an anisotropic `cell_size`:

```python
avg_size = np.mean(cell_size)        # cell_size is [dx, dy, dz]
level    = round(-np.log2(avg_size))
```

So cells with different per-axis sizes share a level:

| dx, dy, dz | mean | level |
|---|---:|---:|
| 4.688e-05, 4.688e-05, 4.688e-05 | 4.6875e-05 | 14 |
| 9.375e-05, 4.688e-05, 4.688e-05 | 6.2500e-05 | 14 |

`upload_mesh_aligned_octree_to_gpu` then collapsed each level to one size
(`mesh_aligned_octree_gpu.py:452-457`, commented "all should be nearly identical
per level") and the search indexed with it
(`mesh_aligned_point_location.py:209-211`).

---

## 4. The decisive measurement

`scripts/dfamily/test_index_mismatch.py` brute-forces the element that really
contains each particle's final position, then compares the index the search
would compute against the stored grid index of the cell that element is
registered in. Lost particles are tested against an equal-sized **control
sample of survivors**:

| group | n tested | no containing element | index MATCHED | index MISMATCHED |
|---|---:|---:|---:|---:|
| lost (`ElementID<0`) | 300 | 0 | 73 (24.3%) | **227 (75.7%)** |
| survivors (control) | 300 | 0 | **300 (100.0%)** | **0 (0.0%)** |

Every surviving particle has a correct index; three-quarters of lost ones do
not. The errors were almost entirely in **z**, with x and y matching — the
signature of a z-pitch error.

---

## 5. Effect of the fix

Unfindable cells (index error exceeding the ±1 the 3×3×3 neighbourhood absorbs),
probing each cell centre:

| pitch rule | D2 | A1 |
|---|---:|---:|
| `level_sizes[0]` (old) | 57,369 (0.96%) | 0 |
| **`np.median(...)` (new)** | **9,266 (0.15%)** | **0** |

6.2× better on D2, and a **no-op on A/B/C** — A1 has one size per level, so its
median *is* its first cell. Median, dominant-mode and cluster-median all
converge on 9,266, which is why the plain median suffices.

Tracking result, D2, 288,000 particles, same seed box as the failing baseline:

| run | code | outcome |
|---|---|---|
| job 22614710 (baseline) | old pitch | first loss at **step 670**; 6,585 frozen by step 1500, 100% with `ElementID<0` |
| job 22638923 (patched) | median pitch | `active=288,000 lost=0 (+0)` through **step 800**; initial assignment 288,000/288,000 |

The patched run passed the step at which the baseline began failing, with zero
losses. It then aborted at step 899 for an unrelated reason — see
[§6](#6-known-issues-found-along-the-way) — so a full-length confirmation is
still outstanding.

The residual 9,266 cells sit at r 4.43–7.51 mm, z −2.46…+1.03 mm: a thin shell
at the shoulder *edge*. Particles were lost at r 2.51–6.86 mm, so the residual
is **not** where the losses occurred, and no special handling of it is
warranted.

---

## 6. Known issues found along the way

**`--hit-stats-log` can abort a run.** It compiles a separate GPU kernel from
the tracking step, and on the LUMI ROCm/JAX build that kernel intermittently
fails to load:

```
Failed to load HSACO: HIP_ERROR_NoBinaryForGpu
  run_tracking.py:3043 -> _hit_probe_closures.hit_probe_step(...)
```

This killed job 22638923 at step 899 of 8000 after the tracking itself had run
cleanly. `verify_pitch_fix.sh` therefore omits the flag by default; set
`HIT_STATS=1` to re-enable it.

**Two separate D-family defects, both independent of the pitch bug:**

- *Seed-box bug.* The production array seeds with
  `--seed-source box-frac --seed-fraction 0.0 0.3 0.0 1.0 0.0 1.0`, i.e.
  fractions of the mesh **bounding box**. The D bbox extends above the
  workpiece (D1: z −6.00…+1.35 mm vs A1: −6.00…0.00), so ~18.4% of seeds land
  in empty air and fail initial assignment. Predicted 18,367 of 100,000;
  observed 17,481. Fix: clamp the upper z seed fraction per case, or seed from
  the FEMUSS node set.
- *Stale build.* Each case's `run_jaxtrace.sh` sets
  `JAXTRACE=.../JAXTrace_stable`, which predates the octree `orphan_fallback`
  fix and **drops** non-Kuhn elements with no Kuhn neighbour (>1.3M on D2).
  Production used the dev repo. A log containing
  `no Kuhn neighbour, skipped` is running the wrong build.

**Registration quality matters, but is not the root cause.** `--registration
aabb` independently cut losses 8,027 → 3,540 (56%) on a 400-step sweep, because
11.2 cells/element gives a wrong index more chances to still land on a cell
listing the element. It cannot fix a wrong index, which is why a residual
survived. `aabb` is now reachable from `run_tracking.py --registration aabb`
(it was benchmark-only before).

---

## 7. What `01_diagnosis.md` got wrong

| claim there | outcome |
|---|---|
| Losses may be the level set zeroing velocity (*Explanation A*) | **Ruled out.** Frozen particles sit *outside* the tool radius; 100% of them had `ElementID<0`. |
| Losses are a search failure (*Explanation B*) | **Correct**, but attributed vaguely to "MALMO". The defect is in the per-level pitch chosen at build time, not in the search algorithm. |
| Remedy: raise `ENHANCED_SEARCH_BAND`, `L0_SKIP_BAND`, `L2_NEIGHBORHOOD` | **Cannot work.** Widening a search cannot find an element absent from the cells it visits. The index is wrong, not the radius. |

Two further hypotheses raised during the investigation were tested and
**disproved**, and are recorded here so they are not revisited:

- *`MAX_ELEMS_PER_CELL` truncation.* Real for the benchmark's static-bound
  search, but tracking calls `search_mesh_aligned_octree_multi_local_where`,
  where `max_tests` appears only in the signature and docstring and never
  truncates.
- *A factor-of-2 size mix within a level.* The max/min ratio across a level is
  indeed 2.000, but that was a red herring: the actual error is the 1.197×
  z-pitch mismatch affecting 92.6% of cells. The per-axis `dx` comparison shows
  no error at all.

---

## 8. Tooling

Under `scripts/dfamily/`, with usage and interpretation in its
[`README.md`](../../scripts/dfamily/README.md):

| script | purpose |
|---|---|
| `verify_pitch_fix.sh` | full production-length verification from a case's own `run_jaxtrace.sh` |
| `submit_dfamily_pitchfix.sh` | the same for the remaining D cases, staggered |
| `test_index_mismatch.py` | **the decisive test**: search index vs stored index, lost vs survivors |
| `eval_pitch_choice.py` | compares pitch rules (first/mean/median/mode/min/max) by unfindable-cell count |
| `why_factor2.py` | per-level distinct size-triples and per-cell anisotropy |
| `audit_octree_coverage.py` | is the true host registered in a cell the search visits? |
| `registration_sweep.sh`, `compare_registration.py` | `aabb` vs `parent_cube` vs variants |
| `analyze_frozen_particles.py`, `dfamily_diag.sh` | frozen-particle statistics for one run |
| `make_diag_run.sh`, `run_analysis_lumi.sh` | run generation and the LUMI singularity wrapper |

All analysis scripts are CPU-only and pin JAX to the CPU backend. The larger
ones need ~240 GB for a 10.8M-element mesh, so they must run as batch jobs — a
LUMI login node caps per-user memory at 96 GB and the octree build dies there
with `MemoryError`.
