# Feasibility Analysis: Time-Dependent Meshes in MALMO

## 1. Problem Statement

The current JAXTrace/MALMO pipeline assumes a **fixed mesh topology** throughout the tracking simulation. The mesh is loaded once, the octree is built once, and only the velocity field changes per timestep. In many adaptive simulations (AMR), the mesh itself changes during the simulation — elements are refined or coarsened as the solution evolves.

The question: **can MALMO handle time-dependent meshes, and what does it cost?**

---

## 2. Current Pipeline (Fixed Mesh)

| Stage | Time (cylA, 3.05M elements) | Frequency |
|-------|---:|---|
| Mesh load (PVTU) | ~9 s | Once |
| AA metadata + inverse matrices | ~58 s | Once |
| Octree extract (centroid) | ~142 s | Once |
| Octree upload to GPU | ~0.2 s | Once |
| Element neighbours | ~150 s | Once |
| **Total preprocessing** | **~360 s** | **Once** |
| RK4 tracking (2684 steps, 293k particles) | ~1972 s | Per step |

The octree build (extract + sort + CSR assembly) is **O(N_e + N_c log N_c)** where N_e is the number of elements and N_c is the number of active cells. For centroid registration: N_c ≈ N_e/6.

---

## 3. What Changes When the Mesh Changes

When AMR refines or coarsens, the following change:

| Data | What changes | Impact |
|------|-------------|--------|
| `node_positions` | New/moved/deleted nodes | Vertex coordinates shift |
| `connectivity` | New/merged/deleted elements | Element→node mapping changes |
| `element_neighbors` | Neighbor topology changes | L1 search arrays invalid |
| Cell sizes `Δ^(ℓ)` | New refinement levels possible | Grid structure changes |
| Morton codes | Cell indices shift | Sorted array invalidated |
| CSR arrays | Element→cell mapping changes | Must rebuild |
| Inverse matrices (M_K^{-1}) | New element geometries | Must recompute for changed elements |
| Cached element IDs | Particle's cached host may no longer exist | Must invalidate |

**Key observation:** The octree is fundamentally coupled to the mesh topology. ANY change in connectivity invalidates the sorted Morton array and CSR structure.

---

## 4. Approaches

### 4.1 Full Rebuild Per Mesh Timestep

**How:** When mesh topology changes, rebuild the entire octree from scratch.

**Cost per rebuild (centroid registration, ~3M elements):**

| Stage | Time | Can skip unchanged? |
|-------|-----:|:---:|
| AA metadata | 30 s | Yes — per-element |
| Inverse matrices | 28 s | Yes — per-element |
| Octree extract + sort | 142 s | No — global sort |
| CSR assembly | included | No — global |
| GPU upload | 0.2 s | No |
| Neighbour rebuild | ~150 s | Partially |
| **Total per rebuild** | **~200–350 s** | |

**Amortisation analysis:**

| Mesh change frequency | Rebuilds (2684 steps) | Rebuild overhead | % of tracking time |
|---|---:|---:|---:|
| Every step | 2684 | 537,000 s | 27,000% — **infeasible** |
| Every 10 steps | 268 | 54,000 s | 2,700% — **infeasible** |
| Every 100 steps | 27 | 5,400 s | 274% — **marginal** |
| Every 500 steps | 5 | 1,000 s | 51% — **acceptable** |
| 3–5 distinct meshes | 3–5 | 600–1000 s | 30–50% — **good** |

**Verdict:** Feasible if mesh changes are infrequent (every 100+ steps). The octree sort is the bottleneck and cannot be easily made incremental.

**Optimisation opportunities:**
- Skip `aa_metadata` and `inverse_matrices` for unchanged elements (element-level dirty tracking)
- Pre-sort unchanged cells and merge with new cells (merge-sort instead of full sort)
- Parallelise the CPU-side build across cores

---

### 4.2 Incremental Partial Update

#### 4.2a Two-Level Array (Base + Delta Buffer)

**Concept:** Keep the main sorted octree array immutable. Store additions/deletions in a small unsorted "delta buffer." At query time, search both.

**Query-time cost:**
```
search_main(binary_search, O(log N_c))  →  found? done
    ↓ not found
search_delta(linear or binary, O(log N_delta))  →  merge results
```

**Pros:**
- No re-sort of main array
- Delta buffer small if few cells change
- Main array retains GPU-friendly properties

**Cons:**
- Query kernel complexity increases (two searches instead of one)
- Adds branching to inner loop — erodes static-loop advantage
- Must handle "shadowing" (element moved from old cell to new cell: old cell in main, new cell in delta)
- Periodic compaction needed when delta grows too large
- Significant new data structure engineering

**Query overhead estimate:** 5–10% if delta < 1% of cells; grows linearly with delta size.

**Verdict:** Architecturally clean but high implementation effort. Worth pursuing only if mesh changes are frequent AND affect a small fraction of cells.

#### 4.2b GPU-Side Radix Re-Sort

**Concept:** Identify changed cells, modify the flat array in place, then radix-sort on GPU.

**Pros:**
- GPU radix sort is O(N_c) with very small constants
- Maintains the flat sorted structure
- No query-time overhead

**Cons:**
- Identifying which cells changed requires diffing old/new mesh (non-trivial)
- CSR rebuild is a global operation — even if sort is fast, CSR offsets must be recomputed for ALL cells
- Must handle element renumbering if connectivity changes (global element IDs shift)
- JAX arrays are immutable — "in-place" modification means full array recreation

**Cost estimate:** GPU sort of 520k entries: ~1 ms. CSR rebuild: ~1–5 s (CPU-side). The CSR rebuild dominates.

**Verdict:** Feasible if a fast CSR rebuild path exists. The sort is cheap; the CSR reconstruction is the real cost.

#### 4.2c Generation Counter / Timestamp Invalidation

**Concept:** Each cell gets a generation counter. Stale cells are skipped at query time.

**Pros:**
- Minimal structural change
- No re-sort

**Cons:**
- Adds branch to innermost query loop: `if gen[cell] == current_gen`
- GPU warp divergence: if ANY thread in a warp hits a stale cell, ALL threads wait
- Stale cells accumulate, wasting binary-search work
- Does NOT handle new cells — only invalidation
- Fundamentally opposed to MALMO's design philosophy (flat, branchless, static loops)

**Verdict: Not recommended.** Erodes GPU efficiency, which is the core value proposition.

---

### 4.3 Pre-Built Octree Set (Recommended First Implementation)

**Concept:** If the mesh changes at known timesteps (typical in AMR — refinement schedule is deterministic or known post-simulation), pre-build all octrees offline and swap at runtime.

**How it works:**
1. During preprocessing, identify all distinct mesh topologies
2. Build one complete octree per topology (embarrassingly parallel)
3. Upload all octrees to GPU (or swap on demand)
4. At runtime, maintain `current_octree_index`; switch when mesh changes
5. When octree switches, invalidate all cached element IDs → force L2 search for one step

**Cost:**

| Item | Cost | Notes |
|------|------|-------|
| Build time | N_meshes × 200 s | Parallelisable across CPU cores |
| GPU memory | N_meshes × 43 MB (centroid) | 10 meshes = 430 MB |
| Runtime switch | One full L2 search per switch | ~3 s for 293k particles |
| Query overhead | Zero | Each octree is a perfect sorted flat structure |

**Pros:**
- Zero query-time overhead — each octree is identical to the current fixed-mesh structure
- Zero incremental-update complexity
- Trivial to implement: array of octrees + switching index
- Build cost is one-time and parallelisable
- Can reuse 100% of existing octree build code

**Cons:**
- GPU memory scales with number of distinct meshes
- Not suitable for continuous mesh changes (every step with unique topology)
- Requires knowing all mesh topologies upfront (post-processing scenario)
- Must store/load per-topology precomputed data (inverse matrices, neighbours)

**Verdict:** Simplest, most GPU-efficient approach. Covers the most common AMR use case where a few distinct mesh topologies alternate during the simulation.

---

## 5. Summary Comparison

| Approach | Query overhead | Build overhead per change | Impl. effort | Best for |
|----------|:---:|:---:|:---:|---|
| Full rebuild | None | ~200–350 s | Low | Infrequent changes (every 100+ steps) |
| Two-level array | 5–10% | Small (delta only) | High | Frequent changes, small affected region |
| GPU re-sort | None | ~1–5 s (CSR rebuild) | Medium | Moderate localised changes |
| Generation counter | **Significant** | Minimal | Low | **Not recommended** |
| Pre-built octree set | **None** | One-time, parallel | **Low** | Discrete AMR events (**recommended**) |

---

## 6. Recommended Implementation Plan

### Phase 1: Pre-Built Octree Set

**Target:** Support N distinct mesh topologies, pre-built offline.

**Changes needed:**
1. **Mesh loader:** Accept multiple PVTU files with different topologies
2. **Octree builder:** Loop over topologies, build one octree per mesh
3. **GPU storage:** Store array of `MeshAlignedOctreeGPU` + `MeshDataGPU`
4. **Runtime:** Add `mesh_step_to_octree_index` mapping; swap active octree when mesh changes
5. **Cache invalidation:** When octree switches, set all `element_ids = -1` to force L2 search

**Estimated effort:** 2–3 days.

### Phase 2: Full Rebuild with Selective Precomputation

**Target:** Rebuild octree at runtime when mesh changes, but skip unchanged elements.

**Changes needed:**
1. Element-level dirty tracking (diff old/new connectivity)
2. Selective `aa_metadata` and `inverse_matrices` recomputation
3. Merge-sort optimisation for partially-changed Morton arrays

**Estimated effort:** 1–2 weeks.

### Phase 3: Two-Level Delta Buffer (if needed)

**Target:** Handle frequent small mesh changes without full rebuild.

Only pursue if profiling shows Phase 1/2 are insufficient for a specific use case.

---

## 7. Information Needed from Mesh Data

To refine this analysis, we need to characterise the actual mesh variation in the FEMUSS simulation data. The following quantities are needed from the LUMI data:

1. **Per-timestep element/node counts** — do they change? By how much?
2. **Connectivity fingerprint** — hash of connectivity array per timestep; how many distinct topologies?
3. **Node position differences** — if topology is fixed, do node positions change (mesh deformation)?
4. **Refinement level distribution** — how many elements at each level per timestep?
5. **Spatial extent** — does the bounding box change?
6. **Element size statistics** — min/max/mean element edge length per timestep
7. **Change localisation** — if topology changes, WHERE in the domain (near tool? boundary? everywhere?)

The script `analyze_mesh_variation.py` (provided separately) collects this data.

---

## 8. Impact on the Paper

Depending on what the mesh data shows:

**If mesh topology is fixed** (only node positions change due to ALE/deformation):
- Current MALMO works as-is if deformation is small (vertex positions shift but octree grid stays valid)
- If deformation is large enough to invalidate cell assignments, a periodic octree rebuild suffices
- Paper can note: "The method extends to deforming meshes with periodic octree reconstruction"

**If mesh topology changes at discrete events** (AMR refinement):
- Pre-built octree set is the right approach
- Paper can note: "For adaptive meshes with N distinct topologies, N octrees are pre-built and swapped at runtime with zero query-time overhead"

**If mesh topology changes every step** (continuous remeshing):
- Full rebuild is too expensive; need the delta-buffer approach
- This is a significant research contribution and likely a separate paper

---

## 9. LUMI Mesh Analysis Results (case C3, 111 timesteps loaded)

The script `analyze_mesh_variation.py` was run on LUMI against
`/scratch/.../C-ThreadsVariations/C3.gid/post` with pattern `C3_{timestep}.pvtu`,
stride=1, intended range 0–151. The job was killed by the 2 h SLURM time limit
after **111 of 152** timesteps (~53 s per PVTU load × 152 ≈ 134 min). The
loaded data is sufficient to draw firm conclusions; a re-run with higher time
limit is **not required** for the feasibility decision.

### 9.1 Headline Findings

| Quantity | Value | Implication |
|---|---|---|
| Distinct topology hashes | **108 / 111** loaded steps | Topology changes essentially every step |
| Element count range | 8,433,298 → 8,447,088 | Variation: 13,790 (**0.16%** of total) |
| Node count range | 2,200,131 → 2,202,253 | Variation: 2,122 (**0.10%** of total) |
| Edge length range | [4.69e-05, 5.20e-03] m | Invariant across all 111 steps |
| Edge ratio (max/min) | ~111× | Confirms **7 refinement levels** at every step |
| Refinement levels | 7 | Constant across all steps |
| Avg PVTU load time | 52.6 s | I/O is the bottleneck for analysis runs |

### 9.2 Interpretation

**The mesh is a continuously-remeshed AMR mesh.** Every loaded timestep has a
unique connectivity fingerprint, but the *changes are tiny*: only ~0.16% of
elements change between consecutive steps, and the global edge-length range
(hence the refinement level count and the bounding box) is constant.

This is exactly the regime that the MALMO design philosophy was constructed
*against*: full rebuild every step is infeasible (Section 4.1), generation
counters degrade GPU efficiency (Section 4.2c). However, the small change
fraction makes the **fixed/dynamic split** discussed in Section 10 a strong
candidate.

### 9.3 What the Existing Data Already Tells Us

- A pre-built-octree-set approach (Section 4.3) **does not apply directly**:
  every timestep has a different topology, so we would need ~152 octrees,
  each ~120 MB (centroid registration on 8.4M elements → ~600 MB per
  cell-aligned arrays + 100+ MB inverse matrices) ≈ **20–30 GB GPU memory
  for the octree set alone**. This is too large for a single GPU.
- A full-rebuild-per-step approach (Section 4.1) is infeasible: ~350 s × 152
  steps = 53,000 s build time vs. ~5 s/step tracking ≈ 70× slowdown.
- The edge-length range and refinement level count are constant. Combined
  with the small fraction of changed elements, this strongly implies the
  changes are spatially localised to the AMR refinement zone (the band
  around the tool where the level-set crosses zero).

### 9.4 What is *Not* Yet Known From the Run

The console output recorded element counts, node counts, hash, and edge stats,
but **node-position differences (`pos_max_displacement`, `pos_frac_moved`)
were not printed** because the script's `pos_info` log line is only emitted
when those fields appear, and the very small per-step changes apparently
fell into a path that didn't trigger the print on this case. The CSV file
(`mesh_variation_report.csv`) on LUMI **does** contain those columns and
should be inspected directly. From the CSV we still need:

1. Maximum node displacement between step 0 and later steps (does the mesh
   *deform* or only *re-tessellate*?)
2. Whether the changed elements are localised (per-element diff vs. step 0)
3. Per-level element counts (level histogram)

These can be obtained by post-processing the CSV without re-running the
heavy PVTU loader. **No second LUMI run is needed for the high-level
decision; only for refining the implementation parameters.**

---

## 10. Cycled Velocity Sequence + Fixed/Dynamic Element Split

This section addresses two refinements that change the calculus substantially.

### 10.1 The Velocity Sequence is Cycled

The simulation provides a **limited mesh+velocity sequence** (e.g. 152 or 160
steps, ~one revolution of the tool). The full tracking run of 2,000–6,000
RK4 steps **wraps this sequence repeatedly**: step k of tracking uses mesh
sequence index `k mod N_seq`.

**Consequence:** the number of *distinct meshes the tracker ever sees* is
bounded by `N_seq`, not by the number of tracking steps. With `N_seq = 152`,
the per-step rebuild calculus from Section 4.1 changes from "rebuild 2,684
times" to "rebuild 152 times, total". And once the first cycle completes,
**zero further rebuilds are needed** — every subsequent cycle reuses
already-built octrees.

**Updated feasibility table** (sequence length 152, 2,684 tracking steps,
~5 s tracking per step):

| Approach | Build cost (one-time) | Steady-state overhead | Verdict |
|---|---|---|---|
| Pre-build all 152 octrees | 152 × 350 s = 53,000 s (CPU build, parallelisable) | 0 s | **Feasible if memory fits** |
| Rebuild each first time, cache | Same as above, distributed over first cycle | 0 s after cycle 1 | **Feasible**, lower peak memory |
| Fixed/dynamic split (Section 10.2) | 152 × (build cost of *changing* part only) | 0 s | **Recommended** |

The dominant constraint is GPU memory for the octree set. With 8.4M elements
and centroid registration:
- Cell-aligned arrays: ~600 MB
- Inverse matrices ($M_K^{-1}$, 9 floats × 8 B × 8.4M elements): ~600 MB
- Total per octree: ~1.2 GB
- 152 octrees: **~180 GB** — far exceeds a single LUMI MI250X GCD (64 GB)

So the naive "pre-build all" is memory-bound, not time-bound. The
fixed/dynamic split is what makes it tractable.

### 10.2 The Fixed/Dynamic Element Split

**Observation from the LUMI data:** only ~0.16% of elements change per step.
The vast majority of the mesh — the bulk material far from the tool — is
*topologically static*. Only the band around the tool (where AMR refinement
tracks the cutting interface) varies.

**Proposal:** partition each mesh into two element sets and treat them
separately.

| Set | Definition | Size | Octree treatment |
|---|---|---|---|
| **Fixed** | Elements unchanged across all `N_seq` cycle steps | ~99%+ | Build **once**, share across all cycle indices |
| **Dynamic** | Elements that change in at least one cycle step | ≤1–5% | Build per cycle index, swap at runtime |

**Memory estimate** (fixed=99%, dynamic=1%, sequence length 152):
- Fixed octree: ~1.2 GB (one copy)
- Dynamic octrees: 152 × 1% × 1.2 GB = **~1.8 GB total** for all 152 dynamic parts
- **Total GPU memory: ~3 GB** — comfortably fits

This is a ~60× memory reduction over the naive pre-build-all approach.

### 10.3 How to Identify the Dynamic Set

Three candidate criteria, in order of robustness:

1. **Direct topology diff** (most robust): for each cycle step, compute
   the symmetric difference of the connectivity tuples vs. step 0. The
   union over all steps defines the dynamic element set. **Requires loading
   all `N_seq` meshes once at preprocessing.**

2. **Refinement-level threshold**: dynamic = "elements at the finest L
   levels". Cheap to compute (no diff needed), and **aligns with the existing
   query loop structure** (see Section 10.5). Risk: may include some static
   fine elements (e.g. tool boundary at fixed location), increasing the
   dynamic set unnecessarily; or may miss medium-level elements that do
   change near the AMR refinement front.

3. **Level-set proximity**: dynamic = "elements with at least one node where
   |φ| < δ" for some band width δ. Conservative, geometry-aware, but requires
   the level-set field to be loaded at preprocessing.

**Recommendation:** use criterion 1 (direct diff) at preprocessing to define
the dynamic set definitively, and verify post hoc that it correlates with
criterion 2 (refinement level) so the algorithm in Section 10.5 can rely on
the level-based test at runtime.

### 10.4 Two-Tier Octree Structure

Concretely, the data structure becomes:

```
fixed_octree:           single MeshAlignedOctreeGPU (one copy, all steps share)
dynamic_octrees[k]:     MeshAlignedOctreeGPU for cycle step k = 0..N_seq-1
fixed_mesh_data:        MeshDataGPU for the static element set
dynamic_mesh_data[k]:   MeshDataGPU for the changing element set at step k
```

At query time, both octrees are searched. The host element returned can be
in either set; the L0/L1/L2 cache logic generalises straightforwardly because
element IDs are globally unique (no cross-contamination between sets).

### 10.5 Algorithmic Hook: Refinement-Level Loop Split

This is where the user's idea about iterating from finer to coarser levels
becomes powerful. The current MALMO query iterates a fixed `(2R+1)^3`
neighbourhood at each refinement level, finest-first. If we split by level:

```
for ℓ in finest..(L_dyn):           # dynamic levels (varying)
    search dynamic_octree[k] at level ℓ
for ℓ in (L_dyn-1)..coarsest:       # fixed levels (one octree forever)
    search fixed_octree at level ℓ
```

This requires **no branching inside the inner loop** — each level is bound to
exactly one octree, statically. The fused kernel keeps its branchless,
fixed-shape character (the very property that makes MALMO efficient on GPU).

**Caveat:** this works cleanly only if the fixed/dynamic split aligns with a
level boundary `L_dyn`. If a level has both static and dynamic elements
(likely for the boundary level between refinement zones), one of these
applies:
- Promote that level entirely to "dynamic" (slight memory overhead)
- Use a per-cell static/dynamic flag at that level only (one branch, but at
  outer loop, not inner)

### 10.6 Even Finer: Fixed Cells, Varying Registration

The user's last point: **even the cells (parent cubes) of the dynamic part
may stay fixed, with only the element-to-cell registration varying.**

This is plausible because:
- Cell sizes are determined by element AABBs (Section 4.1 of the paper),
  and edge-length range is invariant across timesteps (Section 9.1 here)
- Only ~0.16% of elements change → most refined cells stay populated by
  the same elements
- A new element is just a "small re-tessellation" of an already-refined
  region; it lands in cells that already exist

**If this holds (to be confirmed by the CSV diff)**, the dynamic-octree
structure simplifies even further:
- `dynamic_octrees[k]` shares the same Morton/level/CSR-offsets arrays
  across all `k`
- Only the **CSR values array** (element indices stored per cell) varies
- Memory cost drops from `N_seq × dynamic_octree_size` to roughly
  `dynamic_octree_size + N_seq × (CSR_values_size_for_changed_cells)`
- Query kernel is *identical* — only the array it indexes changes per step

This is the "two-level array" approach of Section 4.2a, but with the
*compaction problem solved upfront*: the cell layout is frozen at
preprocessing because we know all `N_seq` snapshots in advance.

**Cost estimate (ballpark):**
- Shared cell layout: same as one full octree → ~600 MB
- Per-step CSR values for changed cells: ~1% × 600 MB = ~6 MB × 152 = ~900 MB
- Static element data ($M_K^{-1}$, neighbours): ~600 MB (one copy)
- Dynamic element data: ~6 MB × 152 = ~900 MB
- **Total: ~3 GB** — same order as Section 10.2, but with simpler swap logic

### 10.7 Recommended Path Forward

1. **Phase A — Confirm the data assumption.**
   Post-process the CSV from the LUMI run to extract:
   - Maximum node displacement (deformation vs. re-tessellation)
   - Per-level element counts across all 111 steps
   - Spatial localisation of changed elements (compute centroid bounding
     box of the changed-element set)

   This requires no new heavy compute; if the existing CSV does not include
   these, write a one-shot Python post-processor that reads two PVTUs,
   diffs the connectivity, and reports the bounding box of the changed
   elements. **This is the minimum needed before any implementation.**

2. **Phase B — Implement the fixed/dynamic split** (replaces Phase 1 of
   Section 6 in this document):
   - Preprocessing: load all `N_seq` PVTUs, compute the static/dynamic
     element partition via direct diff
   - Build one fixed octree; build `N_seq` dynamic octrees
   - Runtime: tracking step `t` uses `dynamic_octree[t mod N_seq]`
   - Cache invalidation: when `t mod N_seq` advances, invalidate cached
     element IDs that point into the dynamic set only (static cache stays)

3. **Phase C — Optimise to "fixed cells, varying registration"** if Phase B
   profiling shows memory or swap cost is significant. Skip if Phase B is
   already fast enough.

### 10.8 Impact on the Paper

This is now a stronger story:

> "For cyclic velocity sequences typical of periodic-state simulations
> (rotating tools, oscillating boundaries), the mesh changes per cycle step
> are restricted to a thin AMR refinement band (verified empirically:
> <1% of elements change per step). MALMO's flat-array structure admits a
> fixed/dynamic decomposition along refinement levels: the bulk mesh is
> registered once, and only the dynamic band is rebuilt per cycle index.
> The query kernel structure — finest-to-coarsest level traversal —
> naturally accommodates this split with no inner-loop branching, preserving
> the static-shape GPU-efficiency that motivates the method."

This is a coherent extension of the fixed-mesh contribution and avoids
turning the paper into a rebuild-strategy survey.

---

## 11. Update From the Two-PVTU Diff and Full CSV (LUMI run 18020158)

After the LUMI run completed successfully, the diff (`diff_a0_b50.txt`)
and the per-timestep CSV revealed that the assumptions used in
Sections 10.2–10.6 are **not valid for this dataset**. The mesh is
**globally remeshed every timestep**, not locally adapted. This section
records the actual measurements and revises the recommended approach.

### 11.1 Measurements

**Cycle period.** The CSV row for step 150 shows `pos_max_displacement =
3.44e-3` and `pos_frac_moved = 0.022%`, while every other step shows
`max_disp ≈ 5e-2` and `frac_moved ≈ 98.7%`. Step 151 matches step 1.

> The mesh sequence is periodic with **N_seq = 150**.

**Element/node counts.** Range across 152 steps: nodes
2,200,131–2,202,253 (drift 0.10%), elements 8,433,298–8,447,088
(drift 0.16%). Bounding box invariant.

**Distinct topologies.** 152 unique connectivity hashes (each step has a
different topology, except step 150 = step 0).

**Node displacement per step.** Maximum displacement of original-index
nodes ranges from 4.5e-2 to 5.2e-2 m at every non-cycle-boundary step.
The bounding box width in the smallest axis is 6.0e-3 m, so the worst
node moves **~10× the domain depth in z**. This is a global
re-tessellation, not a small deformation: node *indices* are reused but
the *node positions they refer to are completely different*.

**Dynamic-element fraction (step 0 vs step 50).**
> 8,308,844 of 8,442,489 elements (**98.4%**) have a connectivity tuple
> that does not appear in step 0.

**Per-level dynamic distribution (step 0 vs step 50).**

| Level | #all (B) | #dynamic | %dynamic at level |
|------:|---------:|---------:|------------------:|
| 0     | 1,920    | 1,645    | 85.7%             |
| 1     | 2,045    | 1,734    | 84.8%             |
| 2     | 6,841    | 6,458    | 94.4%             |
| 3     | 25,772   | 25,443   | 98.7%             |
| 4     | 83,748   | 82,641   | 98.7%             |
| 5     | 624,565  | 613,628  | 98.2%             |
| 6     | 7,697,598| 7,577,295| 98.4%             |

The fraction is 85–98% at *every* level. The dynamic centroids span
**92% of the full bounding-box volume**. There is no spatial localisation
and no level-based localisation.

**Fixed-cell feasibility (§10.6).** Only **7.4%** of dynamic elements
land in cells that were already occupied by static elements. The
parent-cube layout would have to expand by ~92% per cycle step. The
fixed-cells-with-varying-registration approach is ruled out.

### 11.2 Implications for §10 Plan

| Idea | Status |
|------|:------|
| Cyclic sequence reuse (§10.1) | **Confirmed**, with N_seq=150 (not 152) |
| Fixed/dynamic element split (§10.2) | **Rejected**: dynamic fraction is 98%, not <1% |
| Identification by refinement level (§10.3) | **Rejected**: every level is ~98% dynamic |
| Two-tier octree (§10.4) | **Rejected** for this dataset |
| Refinement-level loop split (§10.5) | **Rejected** (no static level boundary exists) |
| Fixed cells, varying registration (§10.6) | **Rejected**: only 7% of dynamic elements reuse cells |

The mesh is being **globally regenerated** every timestep — almost
certainly because the FEMUSS simulation uses an ALE / r-adaptive remeshing
pass that follows the rotating tool. From MALMO's point of view, every
step is a fresh mesh.

### 11.3 What Remains Viable

The only assumption from §10 that survives is the **cyclic sequence**.
With N_seq = 150 distinct meshes and a 2,000–6,000-step tracking run,
each unique octree is amortised over `N_track / N_seq ≈ 13–40` cycles.

Three approaches remain on the table:

**A. Pre-build all 150 octrees and swap.**

Memory cost (centroid registration, ~8.4 M elements per octree, float32):
- Cell-aligned arrays: ~600 MB
- Inverse matrices ($M_K^{-1}$): ~600 MB (float64) / ~300 MB (float32)
- Element neighbours: ~135 MB
- **Per octree: ~1.0–1.4 GB**
- 150 octrees: **150–210 GB** — does not fit on a single LUMI MI250X GCD
  (64 GB). Multi-GPU sharding (one octree set per GPU, swap by `t mod 150`
  at host I/O) is possible but adds substantial engineering.

Build cost: 150 × ~350 s ≈ 14.5 h on one CPU, parallelisable across cores.
On a 64-core node: ~14 min. Feasible offline.

Verdict: **memory-bound on single GPU**, feasible if multi-node memory
or out-of-core swap is acceptable.

**B. Build one octree per step at runtime, free immediately.**

Total runtime build cost: 150 × ~350 s ≈ 14.5 h. Tracking is ~5 s/step ×
2,684 ≈ 3.7 h for steady-state. Build dominates by ~4×. After the first
cycle no further builds are needed only if octrees are *cached* — which
brings us back to (A).

Verdict: **infeasible** unless build cost can be reduced 10× (e.g. GPU-side
build).

**C. Build incrementally on first cycle, swap thereafter (recommended).**

- First 150 tracking steps: build octree on demand at step k, use it,
  cache to disk (or to host RAM if it fits)
- Steps 150+: load from cache (host→device upload only, ~0.2 s) and use
- Memory: only the *active* octree is on device (~1.4 GB)
- Disk cost: 150 × 1.4 GB = **210 GB** scratch — manageable on LUMI
  scratch, fits in /tmp (RAMdisk) only if node has 256+ GB
- First-cycle overhead: 150 × ~350 s = 14.5 h once
- Steady-state: zero extra cost; tracking proceeds at fixed-mesh rate

This trades disk space for memory and does the build work just once.
Feasible on a single GPU.

### 11.4 Optimisation: Reduce Per-Octree Build Cost

The 350 s figure is dominated by:
- AA metadata per element (~58 s)
- Inverse-matrix precompute (~30 s)
- Centroid extract + Morton sort + CSR (~140 s)
- Element neighbours (~150 s)

For approach (C), three optimisations are worth implementing **before**
attempting the cycled run:

1. **Skip element neighbours.** They are an L1 search optimisation. Without
   them, queries fall back to L2 search more often. With cached element
   IDs (L0) hitting >99% of queries during smooth tracking, L1 contributes
   little. Saves ~150 s per build → **build drops to ~200 s**.
2. **GPU-side build.** Morton sort and CSR assembly are fast on GPU (radix
   sort: ~1 ms for 500k cells); the bottleneck moves to PVTU I/O. Saves
   another ~100 s.
3. **Persist inverse matrices and AA metadata only.** Re-compute the octree
   on load (sort + CSR) from cached per-element data. The cache files
   become smaller.

With (1)+(2), per-octree build is ~100 s → 150 × 100 s = ~4 h on first
cycle, a tolerable one-time cost.

### 11.5 Updated Implementation Plan

**Phase 1 — On-demand build with disk cache (replaces §6 Phase 1).**

1. At preprocessing, identify N_seq = 150 from the mesh file list (one
   PVTU per cycle index)
2. At runtime, before tracking step `t`, set `k = t mod 150`. If no octree
   for `k` is cached on disk:
   - Load PVTU `k`, build mesh-aligned octree, upload to GPU
   - Serialise octree arrays to disk: `cache/octree_{k}.npz`
3. Otherwise: load from disk (~5 s host-to-device with `mmap`+upload), use
4. Invalidate L0 element-ID cache for all particles at every k transition;
   force one L2-search step

**Phase 2 — Build-time optimisations** (drop neighbours, GPU sort).

**Phase 3 — Out-of-core overlapping** (background-prefetch octree for
step k+1 while tracking step k).

### 11.6 Impact on the Paper

The original story (§10.8) is no longer supportable for this dataset.
A revised, narrower claim:

> "The MALMO structure is built per cycle index for periodic mesh sequences.
> A disk-backed octree cache populated on the first cycle and reused
> thereafter amortises the build cost over the full tracking run; the
> per-step cost during steady-state cycles is one host→device upload
> (~0.2 s) plus one full L2 search to refresh the cached element IDs.
> No query-time overhead is introduced: each cached octree is structurally
> identical to the fixed-mesh octree of Section X."

This avoids overstatement: we are not exploiting AMR locality (there is
none in this dataset), only the **periodicity** of the mesh sequence.

### 11.7 What the Paper Should *Not* Claim

- That MALMO handles arbitrary deforming or AMR meshes (it does not)
- That the fixed/dynamic split is generally applicable (it depends entirely
  on the remeshing strategy)
- That refinement-level loop splitting offers an algorithmic advantage on
  this dataset (it does not)

These would all be true for *some* AMR codes (those that locally refine
around features and leave the rest static), but **not** for r-adaptive
remeshing schemes like the one in this FEMUSS configuration.

---

## 12. Cross-Case Confirmation (LUMI runs 18124398, 18124863)

The diff (step 0 vs step 50) was repeated on two further cases:
**C4** (different thread variation, same family as C3) and
**A2** (flat-pin variation, different geometry family). All three diffs
agree to within a few percent, which strongly supports that the
behaviour is a **property of the FEMUSS r-adaptive remeshing scheme**,
not of any particular pin geometry.

| Case | Elements | Dyn. fraction | %dyn at level 0 | %dyn in existing fine cells | Bbox extent |
|------|---------:|--------------:|----------------:|----------------------------:|:-----------:|
| C3   | 8.44 M   | 98.4%         | 85.7%           | 7.4%                        | [6e-2, 3e-2, 6e-3] |
| C4   | 8.44 M   | 99.2%         | 94.3%           | 6.8%                        | [6e-2, 3e-2, 6e-3] |
| A2   | 5.97 M   | 99.995%       | 100.0%          | 2.8%                        | [6e-2, 3e-2, 6e-3] |

Notable observations:

1. **The bounding box is identical to four significant figures across all
   three cases** — `[6.0000e-02, 3.0000e-02, 6.0000e-03]`. This is the
   simulation domain, fixed by problem setup, independent of pin geometry.
   *The coarse-level octree grid is therefore architecturally fixed across
   all three datasets.*

2. **A2 is the worst case**: every element at levels 0–3 is new between
   steps 0 and 50, and 100% of level-0 elements differ. So even at the
   coarsest refinement level the *element identity* changes; what stays
   fixed is the *cell layout*, not the elements registered to those cells.

3. **The fixed-cell hit rate at the finest probe size is 2.8–7.4%**. New
   cells appear at the finest level even though the bounding box is fixed.

These results lock in the conclusion: §10.2–§10.6 cannot be salvaged. The
mesh is globally re-tessellated each step, at every level.

---

## 13. Reconsidering: Are the Parent Cubes (Octree Cells) Themselves Fixed?

The user's first proposed direction, in light of §11–§12, is whether the
**octree cell layout** stays fixed even though the elements registered to
those cells change. If yes, only the CSR-values arrays need to be rebuilt;
the cell sort, level structure, and Morton arrays are reusable.

### 13.1 What we already know from the diffs

The diff probe at the **finest** level (cell edge ~2.3e-4 m) shows that
2.8–7.4% of dynamic-element centroids fall in cells that already hosted
static elements. **At the finest level, ~93–97% of dynamic centroids land
in cells that didn't exist (or weren't occupied) in the previous mesh.**

This rules out fixed cells *at the finest level*. But for **coarser** levels
the answer is open. If the coarse cells are fixed, a hybrid scheme is
possible: static cell layout for levels 0..L_cut, dynamic cell layout for
levels L_cut+1..L_max.

### 13.2 What additional measurement is needed

The current `diff_two_meshes.py` probes a single cell size. We need to
sweep the probe size from coarse (~bbox/8 = 7.5e-3 m) to fine
(~smallest edge, 4.7e-5 m), and report **for each probe size**:

- Number of distinct cells in mesh A vs mesh B
- Number of cells in B with no element of A
- Distribution of (cell, level) pairs that change

If the curve of "fraction of B cells with no A counterpart" stays near
zero for coarse probes and rises sharply near the finest level, then a
**fixed-cells-up-to-level-L_cut** hybrid is viable.

Implementation: ~30 lines added to `diff_two_meshes.py`, runs in seconds
per probe (already loaded mesh data is reused).

### 13.3 A priori expectation

There is one strong reason to expect **partial** success:

- The simulation domain bounding box is fixed (§12, point 1) → the
  level-0 cell structure (which is just the bbox subdivided uniformly)
  must also be fixed.
- The smallest source-mesh element edge is invariant (4.69e-5 m across
  all 152 timesteps in the C3 run, see §9 / §11) → the maximum required
  refinement level is also fixed.
- *Where* the refinement is placed within the bbox is what changes (the
  refinement follows the rotating tool).

Therefore the cell layout *must* stay fixed at level 0 (the trivial
top-level cell) and *must* change at the finest level (the AMR fine zone
moves). The interesting question is the cross-over level. We expect
something like:

| Level | Cells fixed? | Reason |
|------:|:-------------|:-------|
| 0     | Yes (trivial) | Single root cell = bbox |
| 1–2   | Likely yes   | Coarse subdivision driven by bbox, not features |
| 3–4   | Maybe        | Depends on whether r-adaptivity reaches these levels |
| 5–6   | Likely no    | These are the AMR-active levels following the tool |

If the cross-over is at L_cut = 3 or 4, then ~92–98% of cells (by count)
are at the dynamic levels, so the savings from fixed coarse cells are
small. If at L_cut = 5, the savings are larger but still bounded.

### 13.4 Verdict on the parent-cube idea

**Partially viable, but the cost-benefit is unfavourable.**

Even if all coarse cells are fixed, the dynamic levels contain >99% of all
cells in the octree (because each level has ~8× more cells than the next
coarsest). The savings from skipping the sort/CSR-build for fixed coarse
cells are <1% of the total octree-build cost.

Worse, a hybrid octree (some levels static, others dynamic) **complicates
the flat-array structure that motivates MALMO**. The query kernel would
need to know per-level whether the cell layout is shared or per-cycle —
introducing exactly the kind of branching that the static-loop design
avoids.

**Recommendation:** measure the cross-over level once via the probe-sweep
extension, but do *not* implement a hybrid layout unless the savings turn
out to be unexpectedly large (e.g. L_cut = 5 or 6, which would mean the
r-adaptivity is concentrated only in the very finest zone). The expected
outcome is that a per-cycle-index octree cache (§11.5) is the cleanest
implementation.

---

## 14. The Velocity-Projection Approach (User's Idea 2)

The user's second proposal: instead of solving the time-dependent-mesh
problem in the tracker, **project the time-varying velocity field onto a
single fixed reference mesh** at preprocessing. JAXTrace then runs against
the fixed reference mesh exactly as in the current paper, with no
algorithmic change.

This turns out to be the **most attractive option** by every practical
metric, and it has a clean place in the paper narrative.

### 14.1 The idea

1. **Choose a fixed reference mesh** $\mathcal{T}_{\rm ref}$ that covers
   the simulation domain (the invariant bbox $[6e\text{-}2, 3e\text{-}2,
   6e\text{-}3]$). Resolution chosen to match the finest source-mesh
   refinement near the tool, coarser elsewhere.
2. **Build the MALMO octree on $\mathcal{T}_{\rm ref}$ once.**
3. **Preprocessing pass:** for each cycle index $k = 0, \ldots, 149$:
   a. Load source PVTU $k$, build a temporary MALMO structure on it
   b. For each node $\bm{x}_i$ of $\mathcal{T}_{\rm ref}$: locate the
      host source-element $K_k(\bm{x}_i)$ via MALMO, evaluate the source
      P1 velocity at $\bm{x}_i$, store as $\bm{u}_k(\bm{x}_i)$
   c. Discard the temporary source octree
   Result: a fixed-mesh velocity sequence
   $\{\bm{u}_k\}_{k=0}^{149}$, one P1 field per cycle index.
4. **Tracking:** run JAXTrace exactly as in the paper, against
   $\mathcal{T}_{\rm ref}$ and the precomputed velocity sequence
   $\{\bm{u}_k\}$.

### 14.2 Cost analysis

**One-time preprocessing.** For each of 150 cycle indices:
- Build source octree: ~350 s (or ~100 s with the optimisations of §11.4)
- Project velocity: $N_{\rm ref}$ point-location queries.
  Using MALMO at ~10 ns/query (centroid registration on GPU, see paper):
  - $N_{\rm ref} = 2$ M nodes → 20 ms per timestep
  - 150 × 20 ms = **3 s of GPU time** for the entire projection

So preprocessing cost is essentially $150 \times (\text{source octree
build}) + 3 \, \text{s}$. With §11.4 optimisations: ~4 h on a 64-core
node, or ~14 h without. **Once**, then never again.

**Steady-state tracking cost.** *Identical to the fixed-mesh case in
the paper.* No per-step octree rebuild, no cache invalidation, no L0
flush at cycle boundaries. The only per-step work is selecting the
correct $\bm{u}_k$ from the precomputed sequence — a single integer
indexing operation.

**Storage.** Fixed reference mesh + 150 velocity snapshots:
- Mesh: ~120 MB (fixed)
- Velocity sequence: 150 × $N_{\rm ref}$ × 3 × 4 B (float32)
  - For $N_{\rm ref} = 2$ M: 150 × 24 MB = **3.6 GB** total
  - For $N_{\rm ref} = 1.6$ M (A2-class): **2.9 GB**
- Fits comfortably in GPU memory.

**GPU memory at runtime.** ~120 MB octree + ~3.6 GB velocity sequence ≈
4 GB. Negligible compared to the 64 GB MI250X GCD.

### 14.3 Accuracy considerations

The projection introduces **one additional P1 interpolation per fixed-mesh
node per cycle index**. Specifically:

- Source velocity $\bm{u}^{\rm src}_k$ is P1 on $\mathcal{T}^{\rm src}_k$
- Projected velocity $\bm{u}^{\rm ref}_k(\bm{x}_i) := \bm{u}^{\rm
  src}_k(\bm{x}_i)$ for each ref-mesh node $\bm{x}_i$
- During tracking, the particle location $\bm{q}$ is interpolated on
  $\mathcal{T}_{\rm ref}$: $\bm{u}(\bm{q}) = \sum_j \phi_j^{\rm
  ref}(\bm{q}) \bm{u}^{\rm ref}_k(\bm{x}_j)$

This is one extra P1 evaluation compared to evaluating directly on
$\mathcal{T}^{\rm src}_k$. The interpolation error is bounded by
$\sim h^2 \|D^2 \bm{u}\|$ where $h$ is the larger of $h_{\rm src}$ and
$h_{\rm ref}$. If $\mathcal{T}_{\rm ref}$ is constructed to match the
source resolution, the additional error is at most a constant factor
(typically <2×) of the existing source-mesh discretisation error.

**Recommendation:** design $\mathcal{T}_{\rm ref}$ as the *union of source
refinement zones over the cycle*. Practically: a fine zone covering the
swept volume of the tool over one revolution (a torus-like region), with
edge length matching the finest source level (~5e-5 m), and a coarser
filling elsewhere. Total node count would be slightly larger than any
single source mesh (because the swept volume is larger than the
instantaneous fine zone), but still <3×.

### 14.4 Comparison with §11.5 cache approach

| Aspect | §11.5 Per-cycle octree cache | §14 Velocity projection |
|--------|:-----------------------------|:------------------------|
| Tracker complexity | Per-step octree swap, L0 cache flush at cycle wrap, two-tier index, per-cycle dirty tracking | **None** — fixed-mesh tracker as in the paper |
| Steady-state cost | Octree load (~5 s) + L2 search (~3 s) at every cycle wrap | **Zero overhead** beyond fixed-mesh case |
| GPU memory | ~1.4 GB active octree | ~120 MB octree + ~3.6 GB velocity sequence |
| Disk cache | ~210 GB on scratch | None at runtime |
| Preprocessing | First cycle = on-demand build (~4–14 h) | One-time projection (~4–14 h) |
| Code paths in tracker | New: cache load, dirty flush, cycle index logic | Unchanged |
| Accuracy impact | None (uses source mesh directly) | One extra P1 interpolation per node per step |
| Robustness | Sensitive to disk I/O, file integrity | Robust: just an array lookup |
| Generalises to other periodic remeshing? | Yes | **Yes**, trivially |

### 14.5 The strongest argument for §14: paper coherence

The projection idea has a remarkable property — **MALMO is the kernel of
its own preprocessing step**. Specifically:

- The source-to-reference projection is an instance of the same
  point-location problem MALMO solves (locate $\bm{x}_i \in
  \mathcal{T}^{\rm src}_k$ for each ref-mesh node)
- The runtime tracker is the fixed-mesh MALMO from the paper, unchanged
- Therefore the same algorithm and the same JAX/XLA kernel is used in
  *both* the projection (build-time) and the tracking (run-time)

This is a strong narrative: the paper's MALMO contribution is **doubly
useful** — once to build the reduced-form velocity sequence, then again
to track particles against it. No new algorithm is needed for the
time-dependent case beyond what the paper already presents.

### 14.6 Recommended Implementation Plan (revised, replaces §11.5)

**Phase 1 — Velocity projection preprocessor.**

New module `jaxtrace/preprocessing/project_velocity.py`:

1. `build_reference_mesh(source_pvtu_pattern, cycle_length, ...)` — design
   $\mathcal{T}_{\rm ref}$ from the union of source refinement zones over
   the cycle (heuristic: take the bbox of all "fine" source elements
   across all cycle indices and refine the ref-mesh accordingly)
2. `project_velocity_sequence(source_pvtu_pattern, cycle_length,
   ref_mesh) → velocity_sequence` — for each cycle index, build source
   octree, locate ref-mesh nodes, evaluate P1 source velocity at each ref
   node, store into a $(N_{\rm cycle}, N_{\rm ref}, 3)$ array
3. Save reference mesh + velocity sequence to disk

**Phase 2 — Tracker integration.**

`run_tracking.py` accepts either:
- `--mode source` (current behaviour): direct tracking on source PVTUs
- `--mode projected --ref-mesh path --vel-seq path`: tracking on the
  projected reference mesh + sequence (uses existing fixed-mesh code path
  with no algorithmic change)

**Phase 3 — Validation and sensitivity studies.**

- Compare projected-mesh trajectories against direct-mode trajectories on
  a short tracking window
- Quantify the projection error as a function of $\mathcal{T}_{\rm ref}$
  refinement
- Report the runtime ratio: fixed-mesh tracking + amortised preprocessing
  vs. on-demand octree rebuilding

### 14.7 What this does to the paper

The §10–§13 narrative was disrupted by the dataset reality. §14 fixes
that with a *positive* statement:

> "When the source mesh sequence is periodic (a common case for cyclic
> processes such as rotating-tool simulations), the time-dependent
> point-location problem can be reduced to the fixed-mesh case by
> precomputing the velocity field on a fixed reference mesh. The
> projection itself is an instance of point-location and uses the same
> MALMO kernel as the tracker. Tracking proceeds with no algorithmic
> change relative to the fixed-mesh case described above; per-cycle
> velocity selection is a single integer index into a precomputed
> array."

This is one short paragraph and it converts the time-dependent-mesh
question from a limitation into a **demonstration of MALMO's reusability**
across both stages of the workflow.

### 14.8 Final Recommendation

Implement **Approach §14 (velocity projection)** as the primary
time-dependent-mesh strategy.

Treat §11.5 (per-cycle cache) and §13 (parent-cube fixed) as
*alternatives* to be discussed briefly in the paper's outlook section,
not as primary contributions. The §14 path requires no algorithmic
extension to the tracker, runs at fixed-mesh speed, fits comfortably in
GPU memory, and uses MALMO twice over.

The probe-size sweep of §13.2 should still be done (it's cheap) for
completeness, but is not on the implementation critical path.

---

## 15. Fine-Zone Envelope Scan + Union-Octree Reference Mesh
### (LUMI run 18232497, 2026-05-07)

The third LUMI scan (`scan_fine_zone_envelope.py` over case C3, 16
samples spanning the 150-step cycle) produces results that **change
the recommended reference-mesh design** away from the two-zone uniform
mesh of §14.1.

### 15.1 Field inventory (PVTU contents)

Confirmed available point-data fields:

| Name | Components | Type |
|------|-----------:|:-----|
| `Displacement` | 3 | double |
| `Pressure`     | 1 | double |
| `Reactions`    | 3 | double |
| `Temperature`  | 1 | double |
| `LEVEL`        | 1 | double (note uppercase) |

Cell-data fields:
| Name | Components | Type |
|------|-----------:|:-----|
| `Stress` | 6 | double |
| `Strain` | 6 | double |

Nothing is named `Velocity` — the displacement field is what the
existing tracker already converts to velocity by frame differencing.
The level-set field is `LEVEL` (capitalised), not `LevelSet`.

### 15.2 Fine-zone envelope: invariant across cycle

| Quantity | Value |
|---|---:|
| Sampled steps | 0, 10, 20, ..., 149 (16 samples) |
| Fine envelope min | `[-7.227e-3, -7.148e-3, -5.988e-3]` |
| Fine envelope max | `[+7.117e-3, +7.148e-3, -1.172e-5]` |
| Fine envelope extent | `[1.434e-2, 1.430e-2, 5.977e-3]` |
| Drift in envelope across cycle | **0** (identical bytes at all 16 steps) |
| Volume fraction of bbox | **11.35%** |
| Fine-element count drift | **0.19%** (1,703,962 to 1,707,219) |
| Edge length range | `[5.66e-5, 3.99e-3]`, 7 octave levels |

**Interpretation.** The AMR refinement template is *spatially fixed* —
the fine zone is a fixed sub-cube of the bounding box, occupying the
swept volume of the rotating tool. What changes between cycle steps is
only *which specific tetrahedra* fill that fixed cube, not the cube
itself. This was previously a hypothesis; the envelope scan confirms
it as a measured property of the dataset.

### 15.3 Implication: §14 two-zone reference is suboptimal

§14.1 proposed a uniform-fine zone inside the envelope and a
uniform-coarse mesh outside. With the data in hand, this is wrong on
two counts:

1. **Inside the fine envelope, the source mesh has 5–6 octave levels
   of internal refinement** (level 5 at the envelope boundary, level 6
   immediately around the tool surface). A uniform-fine zone at level
   6 wastes ~8× memory; at level 5, it loses the level-6 detail.
2. **Outside the fine envelope, source elements are still graded** —
   level 0 at the bbox corners, levels 1–4 between corners and the fine
   envelope. A uniform-coarse outer zone would inflate elements at the
   centre of the bbox and lose features near the inflow/outflow.

The user's diagnosis is therefore correct: a two-level destination
mesh is too coarse a discretisation of the source's level structure.

### 15.4 Union-octree as the destination mesh

The clean way to formalise the user's request is the
**forest-of-octrees union construction**, well-established in the AMR
literature ([p4est](https://www.p4est.org/),
[BursteddeWilcoxGhattas11](https://p4est.github.io/papers/BursteddeWilcoxGhattas11.pdf)).
The destination mesh is built so that

> for every (level ℓ, cell index c) appearing in any source mesh of the
> cycle, the destination has a leaf cell at exactly (ℓ, c).

Properties this union construction inherits from the data:

- The union is **finite** because the source mesh has bounded leaf
  count at every level (per-level drift ≤ 13% across the cycle).
- The union is **fully resolved** wherever any source step refines:
  if the dataset's resolution at (ℓ, c) is needed at any time during
  the cycle, the destination has it permanently.
- The union is **Kuhn-tetrahedralisable** as the existing MALMO octree
  is — each leaf cube is split into 6 tets via the standard Kuhn
  triangulation. JAXTrace's existing `extract_octree_cells_parent_cube`
  / `build_global_morton_octree` pipeline applies unchanged.

### 15.5 Two algorithms to construct the union

**Algorithm U-A: octree-template construction** (recommended).

1. Start with empty destination octree
2. For each cycle step k = 0 ... N_seq-1:
   - Load source mesh k
   - Build its parent-cube octree (existing
     `extract_octree_cells_parent_cube`)
   - For each (level, cell) in source octree k: mark the destination
     octree to have a leaf at that (level, cell), refining ancestors as
     needed
3. After the loop, generate destination tetrahedral mesh: for each
   leaf (level, cell) in the union octree, emit 6 Kuhn tets

Result: a **regular Kuhn-style mesh** with exactly the union refinement
pattern. Per-cube 6-tet subdivision means destination element count =
6 × (union leaf cell count). For the 8.4M-element source, the union
should have at most ~1.2 × source_leaf_count, so destination ≈ 1.5 ×
source_tet_count = ~10–12 M tets.

This is the simplest, most predictable construction.

**Algorithm U-B: source-overlay construction**.

1. Start with source mesh 0 as the destination
2. For each cycle step k = 1 ... N_seq-1:
   - Build source octree k
   - For each (level, cell) where source k has a leaf and destination
     has a *non-leaf parent* (i.e. destination is coarser there):
     refine destination at that location by replacing the parent's
     descendant tets with source-k's tets
3. Done

Result: a **mesh that mirrors the original source tessellation**
wherever any cycle step has refinement. Destination element count
≈ source-mesh count (no Kuhn re-tessellation). Preserves any irregular
features the source mesh has.

**Tradeoffs:**

| Property | U-A (Kuhn template) | U-B (source overlay) |
|---|:---:|:---:|
| Element count | 6 × union_leaves (~10–12 M) | ≈ source count (~8.4 M) |
| Mesh regularity | Kuhn-regular at every level | Whatever the source is (probably also Kuhn) |
| Construction complexity | Low — standard octree union + Kuhn | Higher — must merge tessellations from different sources |
| Compatibility with existing octree builder | **Direct** | Indirect (would need extra dedup pass) |
| Loss of source detail | None (Kuhn covers each cube fully) | None |

For first implementation: **U-A**. It is the algorithm that the
existing JAXTrace pipeline already supports natively (the octree
builder already does Kuhn tessellation under the hood). U-B can be
considered later if U-A's element count turns out to be problematic.

### 15.6 Memory and performance estimates

**Destination mesh** (union of 150 source octrees, Kuhn-tessellated):

| Quantity | Estimate |
|---|---:|
| Destination tet count | ~10–12 M |
| Destination node count | ~3 M (Kuhn introduces some new nodes) |
| Mesh storage | ~150 MB (positions + connectivity) |
| Inverse matrices | ~1 GB (float64) |
| Octree cells / Morton arrays | ~1 GB |
| **Active GPU footprint** | **~2 GB** |

**Velocity sequence** (150 cycle indices × 3 M nodes × 3 floats × 4 B):
~5.4 GB. Plus Pressure (1 component, 1.8 GB), Temperature (1.8 GB),
LEVEL (1.8 GB), Reactions (5.4 GB) → **~16 GB total** if all fields
are projected. Velocity alone fits comfortably in GPU memory; full
multi-field set should be host-resident with on-demand upload, or
projected only for the fields the tracker actually consumes.

**Preprocessing time** (150 cycle indices, with §11.4 optimisations):
- Source octree build: ~100 s × 150 ≈ 4.2 h
- Point-locate destination nodes in source mesh: 150 × 0.2 s ≈ 30 s
- Field interpolation: 150 × 0.5 s × N_fields ≈ 5–25 min
- Union octree construction (one-time, after all sources loaded): ~5 min
- I/O and serialisation: ~30 min

Total preprocessing wall time: **~4.5–5 hours one-time** on a single
LUMI MI250X GCD.

### 15.7 Storage format evaluation

For the projected output (destination mesh + N_seq × N_field cycles),
the candidate formats:

**Per-step VTU + .pvd collection.** Per the
[ParaView discourse forum thread](https://discourse.paraview.org/t/substantial-memory-use-differences-between-vtu-and-multi-block-xdmf2-hdf5/4136),
VTU has lower memory amplification than XDMF2+HDF5 (3.45 GiB vs 599
MiB on the same 640k-element mesh, 6× ratio). VTU is the safest format
for Paraview compatibility. Disadvantage: 150 files duplicate the mesh
geometry (150 × 150 MB ≈ 23 GB on disk), and load time is dominated
by repeated mesh parsing.

**XDMF + HDF5.** Time series with shared mesh reference. The same
forum thread documents memory-blowup pathologies with the ParaView
reader; not recommended.

**VTKHDF (single-file, static mesh).** Per the
[Kitware 2025 status update](https://www.kitware.com/vtkhdf-file-format-2025-status-update/),
VTKHDF added "static mesh support in both the reader and the writer,
which allows caching geometry between time steps when scalar fields are
changing but the unstructured mesh geometry is static". This is
**designed for our exact use case**. Single file, HDF5 backend
(parallel I/O if needed), Paraview reads it natively in recent
versions.

**JAXTrace-native NPZ bundle.** Most compact, fastest GPU upload, no
external dependencies for runtime tracking. Not Paraview-readable.

**Recommendation: write BOTH** — a JAXTrace-native NPZ bundle for
runtime, and a VTKHDF archive for visualisation/external interop. The
NPZ bundle is what gets memory-mapped into GPU at tracking time;
VTKHDF is the "canonical" portable archive. This dual export is cheap:
the projected fields are computed once, then written twice.

VTU + .pvd is supported as a **third optional format** for environments
where the VTKHDF reader is unavailable (older Paraview or other tools),
but is not the default.

### 15.8 Field handling: project everything, transform nothing

Per the user's point 2: the projection step copies fields by name with
no transformation. Concretely:

- `Displacement` (source) → `Displacement` (destination), unchanged
- `Pressure` → `Pressure`
- `Temperature` → `Temperature`
- `LEVEL` → `LEVEL` (note the uppercase preservation)
- `Reactions` → `Reactions`
- `Stress`, `Strain` (cell data): not projected by default — they live
  on cells, so projecting them needs a separate cell-to-cell transfer.
  Treat as out of scope for the velocity-projection preprocessor.

The existing tracker already handles displacement-to-velocity
conversion via frame differencing. No extra step is needed in the
preprocessor.

### 15.9 Pin-velocity reconstruction: opt-in only

Per the user's point 3: `reconstruct_pin_velocity` is a FEMUSS-specific
post-processing step. The default for the projection preprocessor is
**off**; the user enables it via a CLI flag. When enabled, it runs
**at runtime** (i.e. inside the tracker loading path) on the
destination mesh, exactly as it currently runs on the source mesh — no
preprocessing change.

The only thing the preprocessor needs to know is whether to mask any
fields differently when pin-velocity is on. With the
"transform-nothing" rule, the answer is: **no, treat pin reconstruction
as a runtime option that operates on the projected `Displacement`
field, not as a preprocessing step**.

### 15.10 Updated implementation plan

This supersedes §11.5 and §14.6.

**Phase A — Reference mesh construction (offline).**

1. Load all N_seq source meshes (or stream through them one at a
   time)
2. For each, build the parent-cube octree with the existing
   `extract_octree_cells_parent_cube`
3. Compute the **union octree** (Algorithm U-A): a forest containing
   every (level, cell) that appears in any source step
4. Kuhn-tessellate each leaf cube into 6 tets to obtain the destination
   tet mesh (positions + connectivity)
5. Save destination mesh

**Phase B — Field projection (offline).**

For each cycle index k:
- Load source mesh k + named fields (`Displacement`, `Pressure`,
  `Temperature`, `LEVEL`, `Reactions`)
- Build source MALMO octree
- For each destination node, GPU-locate host source element via
  `search_l2_vectorized` vmapped over destination nodes
- Compute barycentric coords; interpolate each field linearly (P1) at
  destination node
- Stash into row k of each field's `(N_seq, N_dest_nodes, ...)` array

**Phase C — Storage.**

- Write JAXTrace-native NPZ bundle (mesh + per-field cycle stacks)
- Write VTKHDF archive (single static-mesh file with N_seq time entries)
- Optionally: per-step VTU + .pvd

**Phase D — Runtime integration.**

`run_tracking.py` accepts `--mode projected --bundle PATH`. When set:
1. Load destination mesh + cycle stacks from NPZ bundle
2. Build the destination MALMO octree once (existing fixed-mesh path)
3. Run RK4 against the cycle stack indexed by `t mod N_seq`
4. If `--reconstruct-pin` is set, apply pin-velocity reconstruction at
   runtime on the destination `Displacement` field before frame
   differencing into velocity

No algorithmic change to the tracker beyond the existing fixed-mesh
path.

### 15.11 Why this answers the user's concerns

1. **No information loss** — destination has at least the resolution
   of any source step at every spatial location (union construction).
2. **Multi-level refinement preserved** — destination has the full 7
   octave levels matching the source dataset, not 2 zones.
3. **Cubes that subdivide between cycle steps are handled** — the
   union construction places the destination at the *finest* level of
   refinement seen at that location across the cycle.
4. **No variable transformations** — fields are copied by name with P1
   interpolation only.
5. **Pin reconstruction is opt-in** — runtime flag, default off.
6. **Format choice** — VTKHDF (single-file static mesh) for
   visualisation, NPZ for runtime; both written from the same in-memory
   structures.

---

Sources for §15:
- [p4est: scalable algorithms for parallel adaptive mesh refinement on forests of octrees](https://p4est.github.io/papers/BursteddeWilcoxGhattas11.pdf)
- [VTKHDF 2025 status update — Kitware](https://www.kitware.com/vtkhdf-file-format-2025-status-update/)
- [VTU vs XDMF2+HDF5 memory comparison — ParaView Discourse](https://discourse.paraview.org/t/substantial-memory-use-differences-between-vtu-and-multi-block-xdmf2-hdf5/4136)
- LUMI run 18232497 (`logs/lumi/analyze_mesh_18232497/`): C3 case, 16-step envelope sweep, 152-step CSV, A=0/B=50 diff

---

## 16. Particle Output Format Choice (Per-Timestep Export)

The §15 evaluation chose **VTKHDF + NPZ** for the *mesh projection*
output — a single static reference mesh plus a (small) cycle stack.
The particle export workload is fundamentally different and deserves
its own evaluation.

### 16.1 The workload

`run_tracking.py` writes one particle snapshot per RK4 step when
`EXPORT_FREQ=1` (the default). With the FEMUSS-like C3 config:

| Parameter | Typical | Range |
|---|---:|---:|
| Number of particles | ~100 K | 50 K – 500 K |
| RK4 timesteps per run | 2,684 | 100 – 6,000 |
| Files per run | 2,685 | matches step count |
| Bytes per file (positions + ParticleID + Group) | ~3 MB | 1.3 MB – 13 MB |
| Bytes per file (full: + ElementID + Escaped + MaxTemperature) | ~3.5 MB | 1.7 MB – 17.5 MB |
| Total volume per run | ~7–28 GB | depends on N_particles |

For 300 K particles with all optional fields: **2,685 files × 10.5 MB
≈ 28 GB**, ~210 GB if 5 runs are kept on disk. This is the *real*
post-processing footprint per case.

### 16.2 Where the bottleneck is today

The current path (`benchmark_femuss_comparison.py:614 VTKExportThread`):

1. After each RK4 step, `positions_gpu` and `element_ids_gpu` are
   pulled to host via `np.array(...)` (~2 ms for 300 K float32).
2. A background thread (queue size 20) writes one VTU per step using
   binary-appended-raw VTK XML (the `write_vtu_binary` helper at
   `benchmark_femuss_comparison.py:507`).
3. The kernel-side RK4 step is fast (~5 ms/step at 300 K particles on
   MI250X). The host-side write is **~30–50 ms** per file, partly
   amortised by the background thread.

At `EXPORT_FREQ=1` the **export pipeline becomes the wall-time
ceiling** for high particle counts: the producer fills the queue
faster than the writer drains it, and the export queue eventually
back-pressures the main loop. Concretely on LUMI's `/flash` (2 TB/s
peak aggregate, but per-process much less), 300 K-particle runs spend
~30–50% of wall time waiting on VTU serialisation.

The transfer-out cost (LUMI → workstation) is a *second* bottleneck:
many small files over a high-latency link is the worst-case pattern
for both `rsync` (per-file fsync) and S3 (per-object overhead).

### 16.3 Candidate formats

For per-timestep particle output, the same archive-vs-archive vs
many-files axes apply, but the priorities differ from §15.7:

1. **Write throughput** — kernel can run at 200 step/s; the writer
   must keep up or back-pressure the main loop.
2. **Random per-step access in Paraview** — the user often jumps to
   step *k* without loading the others.
3. **Transfer to workstation** — long-haul, latency-sensitive.
4. **Self-describing for archival** — readable in 5 years without
   custom code.

| Format | Per-step layout | Mesh duplication | Compression |
|---|---|---|---|
| **VTU per step + .pvd** (status quo) | one `.vtu` per step + collection XML | yes (one trivial cell array per file: ~12 MB / 2685 steps ≈ 0 for points) | no |
| **VTKHDF transient PolyData** | one `.vtkhdf`, point coords flat-stacked in a Steps group; ParaView 6.0 supports this for point clouds | none (Vertices written once) | optional HDF5 gzip / lz4 |
| **HDF5 chunked stack** (custom schema) | one `.h5`, one chunked dataset per field of shape `(N_steps, N_particles, ...)` | none | gzip / lz4, lossless |
| **Parquet per N steps** (batch) | one Parquet file per 100–500 steps | none | snappy / zstd | 
| **Zarr** | dir-per-step OR chunked store, one chunk per step | none | blosc / lz4 |

VTKHDF is **specifically** designed for transient point clouds as of
ParaView 6.0 / VTK 9.3 (Kitware 2025) — the `Steps` group with offsets
into a flattened `Points` dataset is the canonical schema. See
[VTKHDF time-dependent data — Kitware](https://www.kitware.com/how-to-write-time-dependent-data-in-vtkhdf-files/).

### 16.4 Comparative table — write performance (single GPU node)

Numbers are estimates for the C3 workload (300 K particles × 2,684
steps × ~10 MB/step ≈ 28 GB raw). Write-throughput estimates assume
the kernel produces a step every ~5 ms on MI250X.

| Format | Files/run | Write speed (sustained) | Write blocks RK4? | Notes |
|---|---:|---:|:---:|---|
| VTU per step (current) | 2,685 | ~25–35 MB/s on `/flash`, ~15 MB/s on workstation NVMe | **yes**, when queue (20) fills | Each file is a fresh XML+binary blob; serialisation dominates |
| VTKHDF transient | 1 | ~150–250 MB/s (h5py + chunked datasets) | no | One open file, chunked appends; static mesh stored once |
| HDF5 chunked stack | 1 | ~150–250 MB/s | no | Identical perf to VTKHDF (it *is* HDF5); just a different schema |
| Parquet (100-step batches) | ~27 | ~120 MB/s (pyarrow) | no | Columnar; great for ad-hoc Pandas/Polars analyses |
| Zarr (chunked store) | 1 directory, 2,685 chunks | ~80–120 MB/s with blosc | no | Lots of small chunk files inside the store — same MDS hit as raw VTUs |

HDF5-backed formats (VTKHDF and custom HDF5) are roughly **5–10× faster** to write than the current VTU-per-step path, because:
- A single `H5Fcreate` + 2,685 `H5Dwrite_chunk` calls beat 2,685
  `open/fwrite/close` cycles on a parallel filesystem
  ([h5bench, LBL 2021](https://sdm.lbl.gov/~sbyna/research/papers/2021/202106-CUG_2021_h5bench.pdf)).
- Mesh-equivalent overhead (vertex cell tables) is paid **once**
  instead of 2,685 times.
- XML header construction is replaced by a fixed-shape dataset
  descriptor.

### 16.5 Comparative table — Paraview load performance

| Format | Open time (cold) | Time-slider response | Memory amplification |
|---|---:|---:|---:|
| VTU per step | ~5–10 s for 2,685 file scan | ~0.3 s/step | 1× (just the active step) |
| VTKHDF transient | ~1 s | ~0.05 s/step (precomputed offsets) | 1× (static mesh shared) |
| HDF5 custom | requires a custom reader / Python plugin | n/a | n/a |
| Parquet | requires a custom reader | n/a | n/a |
| Zarr | requires a custom reader | n/a | n/a |

VTKHDF is the only HDF5-based format ParaView reads **natively** with
time-aware behaviour. Custom HDF5/Parquet/Zarr need a Python
programmable-source filter, which works but is not user-friendly.

### 16.6 Comparative table — transfer LUMI → workstation

The transfer cost is dominated by per-object overhead. From the
[Amazon S3 small-file analysis](https://www.upsolver.com/blog/small-file-problem-s3)
and [LUMI training materials](https://lumi-supercomputer.github.io/LUMI-training-materials/2day-20241210/10-ObjectStorage/),
the rule of thumb is **~50 ms minimum overhead per object** on a
typical high-latency link, regardless of object size, until the object
exceeds a few tens of MB.

For our 2,685 × 10 MB workload at a ~1 Gbit/s effective transatlantic
link:

| Transfer method | Pattern | Estimated wall time | Notes |
|---|---|---:|---|
| `rsync` over SSH, 1 stream | sequential per file | ~30–60 min | Per-file fsync + SSH handshake dominates |
| `rsync` over SSH, --whole-file -P -j8 | parallel | ~10–15 min | Helps but still per-file overhead |
| LUMI-O upload via `s3cmd`/`rclone`, default | sequential per object | ~25–40 min | Per-object PUT overhead |
| LUMI-O via `rclone --transfers 16 --checkers 8` | parallel | ~5–10 min | Multipart + concurrency |
| **Single tarball** (`tar -cf bundle.tar dir/` then upload) | one large object | ~3–5 min | Removes per-object overhead; needs untar at destination |
| **Single HDF5/VTKHDF file** | one large object | ~3–5 min | Native — no tar/untar step |

The single-file formats are **5–10× faster to transfer** than the
2,685-file VTU collection. This dominates the choice for users who
post-process on a different machine.

[LUMI documentation on data movement](https://docs.lumi-supercomputer.eu/firststeps/movingdata/)
explicitly recommends bundling many small files into archives before
transfer for exactly this reason.

### 16.7 Comparative table — composite scoring

| Criterion | VTU/step (current) | VTKHDF transient | HDF5 custom | Parquet batched | Zarr |
|---|:---:|:---:|:---:|:---:|:---:|
| Write throughput (target ≥100 MB/s) | ❌ ~25 MB/s | ✅ ~200 MB/s | ✅ ~200 MB/s | ✅ ~120 MB/s | ⚠ ~100 MB/s |
| Paraview native time-series | ✅ via .pvd | ✅ native, faster | ❌ custom plugin | ❌ custom plugin | ❌ custom plugin |
| Single file / easy transfer | ❌ 2,685 files | ✅ 1 file | ✅ 1 file | ⚠ ~27 files | ❌ dir tree |
| Compression | ❌ raw | ✅ gzip/lz4 optional | ✅ gzip/lz4 optional | ✅ snappy/zstd default | ✅ blosc default |
| Self-describing / archival | ✅ XML readable | ✅ HDF5 standard | ⚠ schema docs needed | ⚠ schema docs needed | ⚠ schema docs needed |
| Random per-step access (Paraview slider) | ✅ trivial | ✅ via Steps offsets | ⚠ needs custom UI | ❌ batched | ⚠ chunked |
| Implementation effort | 0 (existing) | ~150 lines `h5py` | ~80 lines | ~80 lines | ~80 lines |
| Resume-on-crash | ✅ each file independent | ✅ chunked writes append | ✅ same as VTKHDF | ⚠ partial batch lost | ✅ chunk-level |
| Loss if one file corrupts | ~1 step (recoverable) | whole run (with HDF5 journal: minor) | same as VTKHDF | one batch (~100 steps) | one chunk |

**Score (5 ✅ minimum required):**

- VTU per step: 4 ✅ → status quo, write-bound
- **VTKHDF transient: 8 ✅** → **recommended**
- HDF5 custom: 6 ✅ → equivalent perf, worse interop
- Parquet: 4 ✅ → niche, good for analytics not Paraview
- Zarr: 3 ✅ → bad for this workload (many chunk files)

### 16.8 Recommendation

For per-timestep particle output:

1. **Default: VTKHDF transient PolyData**, single `.vtkhdf` file per
   run. Use `h5py` to write directly following the schema in
   [How to write time-dependent data in VTKHDF files](https://www.kitware.com/how-to-write-time-dependent-data-in-vtkhdf-files/).
   - Apply HDF5 chunking: chunk shape `(1, N_particles, 3)` for
     positions, `(1, N_particles)` for scalars. This makes per-step
     reads contiguous and per-step writes a single chunk-append.
   - Apply gzip-1 compression on field datasets only (skip on
     positions to keep writes fast); achieves ~30–40% space savings
     for slowly-changing scalars (Escaped, Group) with negligible CPU
     cost.

2. **Fallback for older Paraview installations**: keep the VTU+.pvd
   path behind a flag (`--export-format vtu-collection`). VTKHDF
   PolyData reader requires ParaView ≥ 6.0; users on 5.13 or earlier
   should opt out.

3. **Transfer**: from LUMI to workstation, **always transfer the
   single `.vtkhdf` file**. Bandwidth tests
   ([LUMI training materials, 2024](https://lumi-supercomputer.github.io/LUMI-training-materials/2day-20241210/10-ObjectStorage/))
   indicate `rclone --transfers 16` over LUMI-O reaches **~200 MB/s
   sustained** for objects ≥ 100 MB, vs ~30 MB/s for 10 MB objects.
   A 28 GB single file completes in ~2.5 min; the equivalent VTU
   collection takes ~10× longer.

### 16.9 Implementation plan

This is a self-contained add-on to the existing pipeline and does not
require any new LUMI runs to validate. Suggested order:

**Step 1.** Add `--export-format {vtu,vtkhdf}` CLI flag to
`run_tracking.py`, defaulting to `vtu` so existing workflows are
unchanged. `~5 lines.`

**Step 2.** New `jaxtrace/io/vtkhdf_writer.py` exposing a
`TransientPolyDataWriter` class with the same interface as
`VTKExportThread` (`enqueue_export(step, positions, ...)`):

- Constructor: opens an HDF5 file with a `VTKHDF` root group, type
  `PolyData`, writes the static Vertices connectivity once, sets up
  chunked extensible datasets for `Points`, `PointData/*`, and a
  `Steps` group.
- `enqueue_export(step, positions, ...)`: appends one chunk to each
  dataset and updates `Steps/Offsets`. Runs in a background thread
  same as today's writer.
- Close: writes any final metadata and flushes.

`~150 lines.`

**Step 3.** Add `--write-vtu-collection` flag for the fallback path
(off by default).

**Step 4.** Validation script: round-trip test that opens the
`.vtkhdf` in Paraview, compares the rendered point cloud at step *k*
against the equivalent `.vtu` from the legacy path. ~30 lines.

**Total new code: ~250 lines.** Single-GPU runtime risk: zero — the
RK4 kernel is unchanged. If the writer underperforms in practice, the
`--export-format vtu` fallback is one CLI flag away.

### 16.10 Sources for §16

- [VTKHDF 2025 status update — Kitware](https://www.kitware.com/vtkhdf-file-format-2025-status-update/)
- [How to write time-dependent data in VTKHDF files — Kitware](https://www.kitware.com/how-to-write-time-dependent-data-in-vtkhdf-files/)
- [h5bench: HDF5 I/O Kernel Suite — Byna et al., CUG 2021](https://sdm.lbl.gov/~sbyna/research/papers/2021/202106-CUG_2021_h5bench.pdf)
- [Analyzing the Performance of the S3 Object Storage API for HPC Workloads — MDPI 2021](https://www.mdpi.com/2076-3417/11/18/8540)
- [Small-file problem on S3 — Upsolver](https://www.upsolver.com/blog/small-file-problem-s3)
- [LUMI-O object storage training — 2024](https://lumi-supercomputer.github.io/LUMI-training-materials/2day-20241210/10-ObjectStorage/)
- [LUMI documentation — Moving data to/from LUMI](https://docs.lumi-supercomputer.eu/firststeps/movingdata/)
- [LUMI architecture (flash storage 2 TB/s)](https://lumi-supercomputer.github.io/LUMI-training-materials/1day-20240208/01_Architecture/)
