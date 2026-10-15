# Build-stage cost: options, plausibility, effort

**Purpose.** The octree build dominates short runs and is entirely CPU-bound
single-threaded Python while the GPU sits idle. This maps the options the same
way [`13_throughput_roadmap.md`](13_throughput_roadmap.md) maps the search,
with measured costs rather than estimates.

Baseline: **D2**, 2,802,871 nodes (1,921,303 after dedup), 10,807,677
elements, octree 5,982,836 cells. Workstation RTX 5090. Status 2026-10-15.

---

## 1. Where the time actually goes

Measured stage times from a 600-step D2 run:

| stage | time | share |
|---|---:|---:|
| 1_load_mesh | 307.6 s | 16.7% |
| 2_precompute | 133.7 s | 7.3% |
| **3_build_upload** | **1094.4 s** | **59.5%** |
| 5_initial_assign | 11.1 s | 0.6% |
| 6_compile | 30.1 s | 1.6% |
| 7_tracking | 263.3 s | 14.3% |
| **total** | **1840.2 s** | |

**The build is four times the tracking cost**, and 85% of this run is
pre-tracking work. On a full 8000-step production run tracking grows ~13× while
the build stays constant, so the balance flips — but every diagnostic run pays
the build in full, and this investigation ran dozens of them.

### Inside the build — measured directly

Each sub-step timed on the real D2 mesh:

| sub-step | time | share of build |
|---|---:|---:|
| **`extract_octree_cells_parent_cube`** | **963.5 s** | **90.5%** |
| `build_element_neighbors_array` | 97.9 s | 9.2% |
| `upload_mesh_aligned_octree_to_gpu` | **3.2 s** | **0.3%** |
| subtotal | 1064.6 s | |

(plus `load 1 snapshot` 9.3 s and `deduplicate_nodes` 11.9 s; the run's 307.6 s
load stage is 27 snapshots, so loading scales with snapshot count as expected.)

**The host→device transfer is 3.2 s — 0.3%.** Data movement is a non-issue;
every option below is about the CPU-side construction.

### Inside the octree extract

The hot loop is `for elem_id in range(n_elements)` calling per element:

| call | measured per element | × 10.8 M elements |
|---|---:|---:|
| `find_axis_aligned_edges_single` | **14.65 µs** | **158.4 s** |
| `find_parent_cube` | 4.47 µs | 48.3 s |
| `encode_morton_3d_single` | 2.79 µs | 30.2 s |
| **subtotal** | | **236.9 s — 24.6% of the extract** |

So **~726 s (75%) of the extract is the non-Kuhn path and the accumulation**:
the face/node neighbour search over node adjacency for 50.6% of D2's elements,
the AABB cell enumeration, and `dict`/`list` insertion for every (cell,
element) pair — 85 M of them.

**Nothing here is arithmetic-bound.** 14.65 µs to examine six edges of a
tetrahedron is CPython function-call and attribute-lookup overhead; the actual
float work is a few dozen operations. That is the single most important fact
for choosing an option below.

## 1b. The profile that overturned this ranking

An earlier draft of this document ranked the options from the per-element
timings above, and guessed that the ~726 s unaccounted for was mostly the
non-Kuhn neighbour search and the dict accumulation. **cProfile on a
1.2 M-element subset says otherwise:**

| | cumulative | share | calls |
|---|---:|---:|---:|
| **`np.clip`** | **80.7 s** | **52.5%** | **22,278,630** |
| `find_axis_aligned_edges_single` | 19.9 s | 13.0% | 1,200,000 |
| `encode_morton_3d_single` | 17.7 s | 11.5% | 7,080,930 |
| `np.finfo.__init__` | 10.6 s | 6.9% | 44,557,260 |
| `find_parent_cube` | 5.0 s | 3.3% | 1,035,840 |
| `find_kuhn_face_neighbor` | 4.3 s | **2.8%** | 164,160 |

**`np.clip` alone was over half the build**, and the neighbour search — the
option this document had flagged as the likely prize — was 2.8%.

`np.clip` on a *Python scalar* routes through `_wrapfunc` → `_wrapit` →
`ndarray.clip` → `_clip` and constructs `np.finfo` objects on the way: 3.03 µs
per call for what is a pair of comparisons. The build makes three such calls
per registered cell, ~200 M on D2.

**Both are now fixed** (commit "np.clip on scalars was 52.5% of the octree
build"):

| change | measured | projected saving |
|---|---|---:|
| `_clamp_idx` instead of scalar `np.clip` | **55× faster**, identical over 200,006 values including every boundary | **~596 s** |
| magic-number bit-spreading for Morton | **3.3× faster**, bit-exact over 50,006 triples | **~152 s** |
| | | **~748 s of ~1090 s** |

Verified as pure speedups: a strict identity test rebuilds the same mesh with
the old and new implementations monkeypatched in, with and without
`exact_tet_box`, and every field of the octree is byte-identical.

**Lesson for the remaining options: profile before optimising.** Two of this
document's estimates were wrong in the same direction — the cost was in
incidental interpreter overhead, not in the algorithm that looked expensive.

## 2. Options



### 2.1 Vectorise the Kuhn classification — HIGH VALUE, MODERATE EFFORT

`find_axis_aligned_edges_single` is the largest single measured cost (158 s)
and is embarrassingly parallel: for each element, examine six edge vectors and
decide whether each axis has an axis-aligned edge. Rewritten as numpy over all
elements at once — `V[conn]` is already an `(n, 4, 3)` array — it becomes a
handful of array operations.

- **Plausibility: high.** No algorithmic change, no coverage risk; the output
  is `(cell_size, level)` per element, verifiable element-by-element against
  the current function on a sample.
- **Expected gain: ~150 s of 1094 s (14%)**, assuming numpy reaches ~0.1 µs
  per element as it does for comparable work.
- **Effort: ~4 h**, including an equivalence test against the scalar version.

### 2.2 Vectorise Morton encoding — LOW VALUE, LOW EFFORT

`encode_morton_3d_single` is a 21-iteration Python bit loop per element. The
standard magic-number bit-spreading trick vectorises it. **Measured:**

| | per code |
|---|---:|
| current Python loop | 3.491 µs |
| vectorised numpy | **0.073 µs** |
| | **48× faster** |

with identical results verified on 20,000 random triples.

- **Plausibility: certain.** Bit-exact, already benchmarked.
- **Expected gain: ~37 s of 1094 s (3%).** Small, because the function is
  called once per element, not once per registration.
- **Effort: ~1 h.** Worth doing alongside 2.1 but not on its own.

### 2.3 Replace the dict/list accumulation with array building — HIGH VALUE, MODERATE EFFORT

Registration currently appends to a `defaultdict(list)` keyed by
`(morton, level)` — 85 M dict lookups and list appends, each a CPython object
operation. The alternative is to emit a flat `(key, elem_id)` pair array and
`np.lexsort` it once, then derive the CSR offsets with `np.diff` on the sorted
keys. This is the same inversion already used in several diagnostic scripts in
`scripts/dfamily/`, where it handled 85 M entries in seconds.

- **Plausibility: high**, and it has a precedent in this codebase.
- **Expected gain: large.** The non-Kuhn path plus accumulation is ~726 s of
  the 963.5 s extract (75%), and the 85 M dict/list operations are the part of
  it that array building removes outright. Sizing the split between the
  neighbour search and the accumulation needs one more profile.
- **Effort: ~1 day**, because the metadata bookkeeping (`cell_metadata`,
  first-writer-wins semantics) has to be preserved exactly.

### 2.3b Vectorise `build_element_neighbors_array` — MEASURED 97.9 s, LOW RISK

Not separately measured before this profile: the face-adjacency build is
**97.9 s (9.2%)** of the build, and is another pure-Python pass over elements.
Face adjacency is a classic sort-based operation — emit all 4 faces per element
as sorted vertex triples, `lexsort`, and pair up adjacent duplicates — which is
fully vectorisable in numpy.

- **Plausibility: high**; standard technique, output verifiable against the
  current function exactly.
- **Expected gain: most of 97.9 s.**
- **Effort: ~4 h.**

### 2.4 Multiprocessing over element blocks — MODERATE VALUE, MODERATE EFFORT

The classification pass is independent per element, so it parallelises over
`multiprocessing.Pool` with no shared state. The workstation has many cores
and LUMI allocates 7 per task.

- **Plausibility: moderate.** Pickling `(n, 4, 3)` float arrays between
  processes costs real time, and the registration pass is *not* independent —
  it accumulates into shared dicts, so only the classification can be split.
- **Expected gain: up to ~(cores−1)× on the classification portion**, so
  bounded by 2.1's 158 s unless 2.3 lands first.
- **Effort: ~1 day**, plus a correctness concern: worker count would change
  nothing about results, but any nondeterminism in dict ordering would.
- **Verdict: do 2.1 and 2.3 first.** Vectorising is strictly better than
  parallelising an interpreted loop.

### 2.5 Move the build to the GPU — HIGH VALUE, HIGH EFFORT, HIGH RISK

The question that prompted this document: can the idle GPU do the build?

In principle yes, and the technique is well established. The standard GPU
approach for spatial structures is to compute a Morton code per primitive,
**radix-sort** the codes, then derive the hierarchy from the sorted order —
Karras' method, which constructs the whole tree in parallel rather than
top-down ([NVIDIA
research](https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees.md),
[NVIDIA developer blog](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu)).

But our structure is **not** a BVH or a pointer tree. It is a flat, Morton-keyed
*uniform grid per level* with a CSR element list — there is no hierarchy to
construct, so most of Karras' machinery does not apply. What transfers is the
useful half: *Morton code → sort → segment* is exactly the shape of our
registration, and it is the operation GPUs do best.

A JAX implementation would be:

1. `jnp` classification of all elements (vectorised, as 2.1);
2. `jnp` Morton encoding (as 2.2);
3. the AABB cell enumeration — **the hard part**, because each element emits a
   *variable* number of cells (1 to ~24,000 on D2, see
   [`12_occupancy_optimization.md`](12_occupancy_optimization.md)), and JAX
   requires static shapes. Needs either a two-pass count-then-fill with a
   fixed cap, or segment-wise processing by cell count;
4. `jnp.lexsort` on the keys and a segmented reduction for the CSR.

- **Plausibility: moderate.** Steps 1, 2 and 4 are straightforward; step 3 is
  where it becomes real work, and the variable-output-size problem is the same
  class of difficulty that killed the two-tier search (§4 of the throughput
  roadmap).
- **Expected gain: potentially 10×+** on the build, i.e. ~1000 s → ~100 s.
- **Effort: ~1–2 weeks**, and it would need the full coverage and
  lattice-invariant test suite re-run, since it reimplements the structure
  whose defects this investigation spent weeks finding.
- **Risk: high.** The build is the component with the worst defect history in
  this codebase.

### 2.6 Cache the built octree on disk — HIGH VALUE, LOW EFFORT

The mesh is **static across every loaded timestep for all 20 cases**
(certified: [`08_mesh_is_static_all_cases.md`](08_mesh_is_static_all_cases.md)),
and the octree depends only on `(node_positions, connectivity, tolerance,
registration options)`. So for a given case the octree is **the same every
run** — and this investigation rebuilt it dozens of times.

Serialise the `OctreeCellDataParentCube` arrays to `.npz` keyed by a hash of
those inputs; load instead of rebuild when the hash matches.

- **Plausibility: high.** No algorithmic change, no coverage risk. The
  arrays are plain numpy.
- **Expected gain: ~1090 s → ~10 s on every run after the first**, i.e. the
  whole build stage disappears for repeat runs.
- **Effort: ~3 h**, including the cache key (must cover tolerance,
  `orphan_fallback`, `hybrid_non_kuhn`, `exact_tet_box` and the mesh file
  mtime/size) and a `--no-octree-cache` escape.
- **Caveat:** it does not speed up the *first* run of a case, which is what
  LUMI production pays. It transforms iteration speed, not production cost.

### 2.7 Reduce `1_load_mesh` (307 s) — MODERATE VALUE, LOW EFFORT

Snapshot loading is 16.7% here and ~2 h of the LUMI production run, where it
reads 166 PVTU files from Lustre. Two cheap levers:

- **load velocity snapshots in parallel** — they are independent files, and
  the work is I/O-bound, so a thread pool would overlap the reads;
- **cache the deduplicated mesh** alongside 2.6; `deduplicate_nodes` measured
  11.9 s and the geometry is identical across timesteps.

- **Effort: ~4 h** for the thread pool. **Gain: bounded by filesystem
  bandwidth**, so it needs measuring on LUMI rather than the workstation,
  where the mesh is on local NVMe.

---

## 3. Comparison table

| # | option | gain | plausibility | effort | risk |
|---|---|---|---|---|---|
| 2.6 | **cache the built octree on disk** | **~1090 s → ~10 s on repeat runs** | high | **~3 h** | none |
| 2.1 | vectorise Kuhn classification | ~150 s (14%) | high | ~4 h | low |
| 2.0 | **`np.clip` → `_clamp_idx`** | **~596 s, 55× on the call** | certain | **done** | none |
| 2.2 | **vectorise Morton encoding** | **~152 s, 3.3× on the call** | certain | **done** | none |
| 2.3 | array-based CSR accumulation | unknown — profile first; dict ops were NOT the top cost | moderate | ~1 day | low |
| 2.3b | vectorise face-adjacency build | **97.9 s measured** | high | ~4 h | low |
| 2.7 | parallel snapshot loading | bounded by I/O; matters most on LUMI | moderate | ~4 h | none |
| 2.4 | multiprocessing the classification | ≤ (cores−1)× of 158 s | moderate | ~1 day | low |
| 2.5 | **GPU build** | potentially 10×+ | moderate | **1–2 weeks** | **high** |

### Recommended order (updated after the profile and the cache)

**Done:** the octree disk cache (2.6), `np.clip` → `_clamp_idx` (2.0, ~596 s)
and vectorised Morton (2.2, ~152 s). Together those remove ~748 s of the
~1090 s build *and* make it cacheable, so a repeat run pays ~10 s.

**Next, in order:**

1. **Re-profile.** The two fixes removed 69% of the build, so the ranking
   below is stale — `find_axis_aligned_edges_single` (19.9 s, 13.0% in the old
   profile) is probably now the top cost, and the non-Kuhn path's true share is
   unknown. One cProfile run decides what is worth doing, and this document's
   history shows guessing has a poor record here.
2. **Vectorise the Kuhn classification** (2.1) if the re-profile confirms it.
3. **Array-based CSR** (2.3) only if the dict operations actually surface. They
   were *not* the top cost, contrary to the original estimate.
4. **Do not pursue the GPU build** (2.5) unless the first-run build is still
   binding after the above. The transfer it would avoid is 3.2 s.

### Original order, before the profile

1. **2.6, octree disk cache.** Best ratio in the table by a wide margin: three
   hours of work removes the entire 1090 s stage from every repeat run, with
   no algorithmic risk. It would have saved many hours across this
   investigation alone.
2. **2.2 + 2.1 + 2.3b together.** All three are pure vectorisation of measured
   hotspots, verifiable against the scalar versions element-by-element: ~9 h
   combined for ~290 s (Morton 37 s + classification 150 s + adjacency 98 s).
3. **2.3** if the build still matters after 1–2. Profile the dict operations
   first to size it.
4. **2.5 only if the first-run build becomes the binding constraint** on LUMI
   production. It is a 1–2 week reimplementation of the component with the
   worst defect history here, and §2.5's step 3 has no clean JAX answer.

### What this does *not* fix

None of these touch tracking throughput — that is
[`13_throughput_roadmap.md`](13_throughput_roadmap.md), where the ceiling is
1.41× from exact tet-box registration. And on the full 8000-step LUMI run the
build is ~0.6 h of ~10 h, so build optimisation changes iteration speed far
more than production cost.

---

## 4. Answering "can we utilise more GPU load?"

Honestly: **the GPU is idle during the build because the build is a CPython
interpreter problem, not a parallelism problem.** The measured 14.65 µs to
inspect six edges of one tetrahedron is overhead, not computation — the same
work in numpy is ~0.1 µs on the CPU, before any GPU is involved.

So the productive order is:

1. **don't do the work at all** (2.6, cache it);
2. **do it in array form on the CPU** (2.1, 2.2, 2.3, 2.3b) — this alone could
   take the build from ~1090 s to low hundreds;
3. **only then consider the GPU** (2.5), where the remaining win is real but
   the effort is 1–2 weeks and the risk is concentrated in exactly the code
   whose defects this investigation has spent weeks establishing.

Moving an interpreted loop to the GPU without first vectorising it would mean
reimplementing it twice.

---

## 5. Reproducing the measurements

| what | how |
|---|---|
| stage times | any run prints them; see `/tmp/boundab_*.log` on the workstation |
| build sub-step times | `scripts/dfamily/profile_build_stage.py` on a case `.gid` |
| per-element function costs | the inline benchmark in §1, 20,000 synthetic level-14 tets |
| Morton vectorisation 48× | the magic-number bit-spread benchmark in §2.2, results verified bit-exact against the scalar version |

## Sources

- [Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees](https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees.md) — Karras' parallel construction method
- [Thinking Parallel, Part III: Tree Construction on the GPU](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu) — the Morton-sort-segment pattern
- [GPU-Resident Trees: BVH and Octrees](https://docs.vulkan.org/tutorial/latest/Advanced_Vulkan_Compute/06_Advanced_Data_Structures/02_gpu_resident_trees.html) — GPU-resident structure layout
