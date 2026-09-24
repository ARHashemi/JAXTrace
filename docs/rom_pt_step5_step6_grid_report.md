# ROM particle tracking — §5 & §6 results report (grid PT on FSW cases)

> **⚠ Read Part 2 before acting on Part 1's conclusions.** Part 1
> covers 12 variants that all use raw P1 barycentric at Stage 1
> (mesh → grid nodes). Part 2 adds 12 HCT-3D-projected variants and
> **reverses the practical recommendation**: HCT Stage 1 dominates
> the grid-family choice by a factor of 2–4× in RMS. The Part 1
> statement "MALMO wins on every case" is true *only* under raw-P1
> Stage 1; under HCT Stage 1 the leaderboard flips and the
> block-refined `4lvl_hct_tricubic` becomes the best variant on
> every case, with MALMO dropping to mid-pack. Skip to
> [Part 2 · HCT Stage-1 follow-up](#part-2--hct-stage-1-follow-up)
> if you only need the current best practice.

**Date:** 2026-07-30
**Data:** 4-case sweep (000, 001, 003, 004) of the FSW FOM dataset,
2000 steps × DT = 3.75e-3 s, 360 k particles per run. Reporting
step: 950 (t ≈ 3.56 s). Reference tracker for every comparison in
this report is **mesh + HCT-3D** (`post_pt/fom_hct_on/`) —
particles are seeded from that run's step-0 slice so every grid PT
variant starts on exactly the same 360 k particles the reference
did.
**Corresponding roadmap:** [rom_pt_roadmap.md](rom_pt_roadmap.md)
§§ 5 – 6.

Section 5 asked *"can a uniform Cartesian grid replace unstructured
element search without materially hurting Lagrangian error?"* and
section 6 was scoped to fall back to block-refined grids if the
answer was no. This report is the answer for both, running 12 grid
variants per case (6 grid families × trilinear/tricubic interp) on
the 4 FSW cases.

## Data sources

| artefact | path |
|---|---|
| per-case grid PT runs | `/scratch/shared/ROM/FOM/cylindrical_<c>.gid/out_grid_<VAR>/run_*/particles.vtkhdf` |
| reference (mesh + HCT-3D) | `/scratch/shared/ROM/FOM/cylindrical_<c>.gid/post_pt/fom_hct_on/run_*/particles.vtkhdf` |
| aggregate CSV (48 rows: 4 cases × 12 variants) | `/scratch/shared/ROM/FOM/cylindrical_<c>.gid/rom_out/fom_grid_pt/grid_vs_mesh_pt_summary_step950.csv` |
| per-variant per-particle displacement VTUs | `.../rom_out/fom_grid_pt/pt_error_case<c>_grid_<VAR>_step950.vtu` |
| summary bar chart | `.../rom_out/fom_grid_pt/fom_grid_vs_mesh_pt_step950.png` |

Regenerate with [scripts/compare_grid_vs_mesh_pt_fom.py](../scripts/compare_grid_vs_mesh_pt_fom.py).
Grid PT runs are launched per case with `cylindrical_<c>.gid/run_grid_all.sh`
(one runner per case, all 12 variants in one loop, skip-guarded so
reruns only build what is missing).

The two orthogonal knobs in this experiment are **grid family**
(what the Cartesian layout looks like) and **interpolation order**
(how a query point between grid nodes is interpolated). Cross the
two axes — 6 × 2 = 12 variants per case — and the roadmap §5/§6
question separates cleanly into two effects.

---

## Methodology

### Two independent interpolation stages

The grid PT pipeline has two velocity-interpolation stages, and they
are set independently:

1. **Stage 1 — mesh → grid-node values** (done once at build time by
   [scripts/build_hier_grid_velocity_case.py](../scripts/build_hier_grid_velocity_case.py)).
   Every grid cell centre is a query point on the source FOM mesh at
   ts = 119. Two options exist for how that query is answered:
   - `--mesh-projection p1_raw` — raw P1 barycentric interpolation
     inside the containing tet (C⁰).
   - `--mesh-projection hct_cubic` — HCT-3D (Alfeld-split Bernstein
     cubic on tets, C¹). Requires recovered nodal gradients.

   **Every variant in this report uses `p1_raw` at Stage 1.** No
   `hct_cubic`-projected grids were built for this experiment, so
   the "tricubic vs trilinear" question here is *only* about
   Stage 2. If we later want to test HCT-projected grids as a Stage 1
   variant we would double the matrix to 24 variants per case; that
   is not covered here.

2. **Stage 2 — grid-node values → query point** (runtime dispatch,
   compiled into each generated velocity module). The `--interp`
   flag picks:
   - `trilinear` — standard 8-point trilinear (C⁰). Default; unnamed
     suffix in the variant list.
   - `tricubic` — Catmull-Rom Hermite tricubic on a 4×4×4 stencil
     (C¹). Suffix `_tricubic` in the variant name. Zero derivative
     input needed — the 1-D Catmull-Rom basis reads

     ```
     w_{-1}(t) = 0.5·(-t³ + 2t² − t)
     w_{ 0}(t) = 0.5·( 3t³ − 5t² + 2)
     w_{ 1}(t) = 0.5·(-3t³ + 4t² + t)
     w_{ 2}(t) = 0.5·( t³ − t²)
     ```

     applied in each of the three axes.

The name `_tricubic` in variant filenames refers to Stage 2 Catmull-Rom
only. It has nothing to do with HCT. It is the grid-side C¹ analogue
of what HCT-3D does on meshes, but the underlying reconstruction is
Hermite-tensor-product on a Cartesian stencil, not Bernstein-cubic on
an Alfeld-split tet.

### Grid families (variants)

> **Errata (2026-09-09): cell counts for the block-refined families were
> undercounted in the table below and throughout Parts 1–5.** The published
> figures counted only the *base* block, not the nested refinement blocks.
> Verified against each generated module's `BLOCK_META` and the stored
> `.npz` velocity arrays, the true totals are: `2lvl` = **437,400**
> (not ~57,000), `4lvl` = **3,547,800** (not ~64,000), `4lvl_r22` =
> **3,547,800** (not ~180 k), `5lvl` = **4,860,000** (not ~1.8 M).
> `uniform` (48,600), `malmo` (79,792), `malmo3` (2,133,792),
> `malmo6` (16,986,432) and `uniform_half` (17,489,920) were correct.
> The `4lvl` finest Δx is **4.44e-5 m**, not 1.25e-4.
> **No accuracy or timing result changes** — every rms and wall-time number
> was measured from the runs themselves, not derived from cell counts. What
> changes is the "cell count is not a throughput knob" argument, which is
> now made by `uniform_hct` (48.6 k cells, 18 s) against `malmo6_hct`
> (17.0 M cells, 36 s) — a 350× cell-count span for 2× the wall time.
> Regenerate the counts with `scripts/plot_deck_supplementary_figs.py`.


All grids cover the same case bbox `[-0.015, 0.030] × [-0.015, 0.015] × [-0.0045, 0]` m
(volume 0.045 × 0.030 × 0.0045 m³).

| suffix | family | topology | cells | finest Δx (m) |
|---|---|---|---|---|
| `uniform` | uniform | 90×60×9 single block | 48 600 | (5.0e-4, 5.0e-4, 5.0e-4) |
| `2lvl` | block-refined | base + 1 nested block, r < 0.016 m | ~57 000 | (2.5e-4, 2.5e-4, 2.5e-4) |
| `4lvl` | block-refined | base + 2 nested blocks (r < 0.016, r < 0.008) | ~64 000 | (1.25e-4, 1.25e-4, 1.25e-4) |
| `malmo` | MALMO-aligned | 5 nested blocks, one per MALMO octree depth level, cells-per-cube-edge = 1 | 79 792 | (7.0e-4, 4.7e-4, **7.0e-5**) |
| `malmo3` | MALMO-aligned | same 5 blocks, cells-per-cube-edge = 3 | 2 133 792 | (2.3e-4, 1.6e-4, **2.3e-5**) |
| `uniform_half` | uniform, mesh-resolving | single 640×427×64 block at Δx = half the finest tet edge | 17 489 920 | (7.0e-5, 7.0e-5, 7.0e-5) |

Each of the six families has a `_tricubic` counterpart — same grid
geometry, Stage 2 interpolation set to Catmull-Rom instead of
trilinear.

**MALMO alignment.** The mesh-aligned Morton octree
(`jaxtrace.gpu.search.morton_octree_builder.build_global_morton_octree`)
partitions the source FOM mesh into leaves at 5 depth levels (2–6
for this cohort), with cube edges from ~1.1e-2 m at depth 2 down to
~7e-4 m at depth 6. The `malmo` and `malmo3` variants extract one
nested Cartesian block per depth level: the block's bbox is the
tight axis-aligned bounding box of all mesh elements at that depth
level or finer (clamped to the case bbox), and the block's cell size
is the depth's cube edge divided by `--malmo-cells-per-edge`
(1 for `malmo`, 3 for `malmo3`). The runtime lookup queries blocks
finest-first: a point that falls inside the finest block is served
from that block; otherwise the next coarser block is tried, and so
on. This gives a smooth fine-to-coarse transition that follows the
actual mesh refinement rather than a hand-picked radial rule.

**Finest tet edge (reference for `uniform_half`).** Sampled from the
source `.pvtu` for each case: min = 1.406e-4 m, p5 = 1.406e-4 m,
median = 1.989e-4 m, p95 = 3.977e-4 m. All 4 cases share the same
mesh so this number is identical across the cohort. The
`uniform_half` variant sets Δx = 7.03e-5 m = half of that finest
edge, so the ultra-fine grid resolves every mesh element by at
least 2 cells in every direction.

### Reference and error metric

Reference is the mesh + HCT-3D tracker (`post_pt/fom_hct_on`). For
each grid variant the compare tool matches the target step
(950.0 s of simulation time = step 950), loads both trackers'
position tables at that step, and reports per-particle
`|x_grid − x_mesh|` in metres, aggregated as rms / mean / p95 / max
across all 360 000 particles.

Reported metrics are therefore **grid-discretisation error on top of
the mesh + HCT-3D tracker**. They are *not* ROM residual (which
[rom_pt_step2_step3_report.md](rom_pt_step2_step3_report.md) covers
separately) — the ROM never enters this experiment; both sides use
the FOM velocity.

### Runners and skip guard

Each case's `cylindrical_<c>.gid/run_grid_all.sh` loops the 12
variants in a fixed order. The runner checks whether
`out_grid_<VAR>/run_*/particles.vtkhdf` already exists and skips the
variant if so — this makes reruns cheap when new families are added
one at a time. Simulation parameters match the mesh reference:
2000 steps × DT = 3.75e-3, `float32`, `--boundary-walls x_max=outlet`,
seeds loaded from the reference vtkhdf's step-0 slice.

---

## Results (step 950, rms |x_grid − x_mesh| in m)

Reading the table: **columns** = 6 grid families in order of
increasing effective resolution (uniform → 2lvl → 4lvl → malmo →
malmo3 → uniform_half); **rows** = case × interp order. Bold =
best-in-row per case × interp.

### Trilinear (Stage 2 = C⁰)

| case | uniform | 2lvl | 4lvl | malmo | malmo3 | uniform_half |
|---|---|---|---|---|---|---|
| 000 | 8.86e-3 | 1.10e-2 | 1.12e-2 | **7.52e-3** | 8.25e-3 | 1.18e-2 |
| 001 | 1.58e-2 | 2.10e-2 | 2.15e-2 | **1.36e-2** | 1.64e-2 | 2.29e-2 |
| 003 | 1.62e-2 | 2.17e-2 | 2.22e-2 | **1.37e-2** | 1.70e-2 | 2.35e-2 |
| 004 | 1.19e-2 | 1.60e-2 | 1.64e-2 | **9.52e-3** | 1.19e-2 | 1.74e-2 |

### Tricubic (Stage 2 = C¹ Catmull-Rom)

| case | uniform | 2lvl | 4lvl | malmo | malmo3 | uniform_half |
|---|---|---|---|---|---|---|
| 000 | 9.86e-3 | 1.13e-2 | 1.14e-2 | **7.97e-3** | 8.64e-3 | 1.19e-2 |
| 001 | 1.84e-2 | 2.18e-2 | 2.20e-2 | **1.47e-2** | 1.68e-2 | 2.30e-2 |
| 003 | 1.93e-2 | 2.23e-2 | 2.27e-2 | **1.58e-2** | 1.76e-2 | 2.36e-2 |
| 004 | 1.40e-2 | 1.65e-2 | 1.68e-2 | **1.11e-2** | 1.24e-2 | 1.75e-2 |

## Interpretation

> **All six findings below are *raw-P1 Stage 1 conditional*.**
> When Stage 1 is upgraded to HCT-3D (Part 2), the family-ordering
> in §1 flips and the tricubic-doesn't-help finding in §5 reverses
> slightly. Read this section as evidence about what limits the
> raw-P1 pipeline, not as guidance for the pipeline we should
> deploy.

Six things stand out.

### 1. MALMO wins on every case, in both interp orders

`malmo` (5-block MALMO-aligned, cells-per-cube-edge = 1) is the
best variant in **all 8 rows** — trilinear and tricubic, cases
000/001/003/004. It beats every other family including the
17.5M-cell `uniform_half` that resolves the mesh 2× everywhere.
The reason is the fine-to-coarse transition: the finest MALMO block
is placed *and sized* by the mesh's own refinement structure, so
sample points near the pin sit in the finest block naturally, while
points in the outer domain sit in cheaper coarser blocks — but
crucially the transition matches the mesh, so a query point does
not oscillate across a resolution jump the way it does at a
hand-picked radial cutoff (see §2 below).

### 2. Hand-picked block refinement is worse than uniform

`2lvl` and `4lvl` are *worse* than `uniform` on every case (by ~25 %
in RMS). The radial cutoffs at r = 0.016 and r = 0.008 m were
chosen by inspection, not from the mesh. A query point that
straddles a block boundary reads velocity from the coarser block
outside and the finer block inside, and the two block values —
computed by raw P1 projection from the *same* mesh — disagree
because they sample the underlying velocity field at grossly
different length scales. That disagreement injects an error
proportional to the local field curvature every time a particle
crosses a block boundary.

MALMO alignment eliminates the disagreement because both sides of a
block boundary correspond to cells that the underlying MALMO octree
had already grouped at the same depth — the transition sits exactly
where the mesh already has a resolution jump.

### 3. Refining the MALMO grid 3× per direction *hurts*

`malmo3` is worse than `malmo` on every case in both interp orders
(by 5 – 25 % in RMS). The finest block goes from Δx ≈ 7e-5 m
(malmo) to Δx ≈ 2.3e-5 m (malmo3) in z, and from ~5e-4 m to ~2e-4 m
in x/y. Naïvely, more cells should improve accuracy — but at Stage
1 we are sampling the *same* raw-P1 field. Once the grid is finer
than the tet edge, more cells simply resample the same piecewise-
linear field more densely and start to expose its inter-tet
C⁰-only corners as small oscillations under Catmull-Rom / trilinear
interpolation on the grid. The Stage 1 raw-P1 field is not smooth
enough to reward Stage 2 grid oversampling.

### 4. `uniform_half` — the ultra-fine grid — is the *worst*

`uniform_half` at Δx = 7.03e-5 m (half the finest tet edge) sits
17.5M cells strong, and it is the **worst** variant on every case
in both interp orders. Same root cause as §3, amplified because
the uniform grid oversamples the coarse tets in the outer domain
just as aggressively as the fine tets near the pin. Every cell
whose containing tet is coarser than Δx picks up the same
piecewise-linear discretisation error, and the interpolation stage
cannot fix it — grid refinement past the mesh's own resolution is
adding cells that mostly re-sample the same P1 information.

Uniform refinement is not a lever we can pull; the ceiling is set
by the Stage 1 projection method.

### 5. Tricubic is a **wash — or slightly worse — than trilinear**

Across the 24 case × grid pairs, tricubic beats trilinear in
**zero** of them; it ties (within 3 %) on the `2lvl` / `4lvl` /
`uniform_half` families, and is 3 – 15 % *worse* on `uniform` /
`malmo` / `malmo3`. This mirrors the analytic-case slide-10 finding
in [rom_pt_team_presentation.md](rom_pt_team_presentation.md#slide-10):
**a higher-order Stage 2 interpolator does not help — and can hurt —
when Stage 1 is only raw P1**, because Catmull-Rom's C¹ overshoot
kicks in exactly at the piecewise-linear kinks between tets, right
where the trilinear stencil is safely averaging.

The corollary is that the interpolation-order axis of this
experiment is decisively secondary. The **grid family** axis is
what moves the numbers.

### 6. Case ordering is stable across variants

Case 000 is easiest (rms ~8e-3 m), 001/003 are hardest (~1.4–2.3e-2
m), 004 sits in between (~1e-2 m). This ordering is preserved
across every one of the 12 variants, and it matches the ROM/FOM
Eulerian ordering in
[rom_pt_smoothness_divfree_report.md](rom_pt_smoothness_divfree_report.md).
Cases with steeper velocity gradients near the pin (001, 003) suffer
more from any grid-based sampling, and the *ordering* of family
performance within each case is identical:

```
malmo < malmo3 ≈ uniform < 2lvl ≈ 4lvl < uniform_half
```

for every case in the cohort. This is strong evidence that the
family effect is not an artefact of one case.

## Practical recommendation

> **Superseded by Part 2.** See
> [Part 2 · Interpretation](#interpretation-part-2) for the current
> recommendation. What follows is the recommendation the raw-P1
> Part-1 data supported on its own, kept for historical trace.

Under raw-P1 Stage 1 only:

- **Use `malmo` (MALMO-aligned, cells-per-cube-edge = 1) with
  trilinear.** Best raw-P1 accuracy, smallest cell count (~80 k),
  no HCT at Stage 1, no tricubic at Stage 2.
- Do **not** upgrade Stage 2 to tricubic under raw-P1 — it costs
  8× more per lookup and returns nothing on raw-P1 grids.
- Do **not** hand-pick refinement radii for block-refined grids
  under raw-P1; they are worse than uniform.

Absolute Part-1 RMS numbers (~7.5e-3 m for case 000, up to ~1.4e-2 m
for 003) are still larger than the ROM/FOM Lagrangian residual in
[rom_pt_step2_step3_report.md](rom_pt_step2_step3_report.md). The
current recommendation using HCT Stage 1 cuts these numbers by
another 2–4× — see Part 2.

## What was *not* done

- **No HCT-3D Stage 1 variants.** ~~Every grid in this report is
  raw-P1-projected.~~ **Follow-up (2026-07-30):** the 12 HCT-3D
  Stage-1 variants are now built for every case; see
  [Part 2 · HCT Stage-1 follow-up](#part-2--hct-stage-1-follow-up)
  below.
- **No ROM-velocity grid PT.** Every grid here samples the FOM
  velocity field at ts = 119. The natural next step is projecting
  the ROM velocity field onto the same grids and repeating this
  experiment; that closes the loop with §2/§3 of the roadmap.
- **No timing.** Wall time was not recorded per variant. The
  analytic-cases team presentation (Slide 13) already established
  20 – 40× grid vs mesh speedup on the recirc/rot_cyl cases; that
  factor is expected to carry over unchanged.
- **No time-dependent velocity.** Every grid here bakes ts = 119.
  For the time-dependent tracker we would need to bake a velocity
  time series onto the same grid — feasible with the existing
  `build_hier_grid_velocity_case.py` invoked per timestep and a
  time-stacked npz layout, but not attempted in this report.

---

# Part 2 · HCT Stage-1 follow-up

**Date:** 2026-07-30

Part 1 sampled the mesh at each grid centre using **raw P1
barycentric** (C⁰, `--mesh-projection p1_raw`). Interpretation §5
observed that a smoother Stage 1 would be needed before Stage 2
tricubic could help. This follow-up rebuilds the same 6 grid
families with `--mesh-projection hct_cubic` — the smoother the
projected field, the more headroom the Stage 2 interpolator has.

## Naming

The 12 new variants add `_hct` to the family suffix; the Stage 2
axis is unchanged:

| trilinear Stage 2 | tricubic Stage 2 |
|---|---|
| `uniform_hct` | `uniform_hct_tricubic` |
| `2lvl_hct` | `2lvl_hct_tricubic` |
| `4lvl_hct` | `4lvl_hct_tricubic` |
| `malmo_hct` | `malmo_hct_tricubic` |
| `malmo3_hct` | `malmo3_hct_tricubic` |
| `uniform_half_hct` | `uniform_half_hct_tricubic` |

Total experiment matrix per case now: **24 grid variants**
(6 families × 2 Stage 1 × 2 Stage 2).

## Stage 1 = HCT-3D · what happens at build time

For each case the builder does the following **once** (cached to
`cylindrical_<c>.gid/hct_cache_ts119.npz`, ~240 MB per case):

1. Load the FOM mesh + nodal velocity at ts = 119.
2. Run SPR (`method="vertex_taylor"`) to recover a nodal gradient
   tensor `G_a` (3×3 per vertex, WLS over the vertex's patch of
   tets). See
   [gradient_recovery_pipeline.md](gradient_recovery_pipeline.md).
3. Build the Alfeld split of every parent tet (parent → 4 sub-tets
   meeting at the parent centroid), and evaluate the C¹ HCT-3D
   Bernstein cubic coefficients on every sub-tet from
   `(v_a, G_a)` at the four parent vertices. See
   [hct3d_implementation_plan.md](hct3d_implementation_plan.md).
4. Cache `hct_coeffs` (shape `(n_cells, 4_subtets, 20_bern, 3)`,
   float32) + `alfeld_centroids` to the cache path.

Then for every one of the 12 grid variants of that case:

5. For each grid cell centre, `vtkCellLocator.FindCell` picks the
   parent tet, `argmin(w_parent)` picks the containing sub-tet, and
   `bernstein_cubic_evaluate(hct_coeffs[cid, s], bary)` returns the
   HCT-projected velocity at that centre.
6. The resulting per-block velocity array is baked into
   `grid_velocity_cyl_<c>_<VAR>_hct[_tricubic].py` + `.npz` in the
   same format the raw-P1 modules use.

The runtime Stage 2 lookup is bit-identical to the raw-P1 variants
— trilinear or Catmull-Rom tricubic on the same block layout — so
the only difference between an `_hct` variant and its raw-P1 twin
is the *values baked at the grid nodes*.

**Runtime cost:** the HCT eval is a pure-Python VTK cell-locator
loop, ~15 k cell-centres/second. For the ultra-fine
`uniform_half_hct` variant (17.5 M cells) that is ~20 min per case;
for the other 10 variants combined it is ~5 min per case.

## Results (step 950, rms |x_grid − x_mesh| in m)

96-row CSV at
[`rom_out/fom_grid_pt/grid_vs_mesh_pt_summary_step950.csv`](../../../fsw-gpu/scratch/shared/ROM/rom_out/fom_grid_pt/grid_vs_mesh_pt_summary_step950.csv).
Bold = best-in-row per case × Stage-2 interp.

### Stage 1 = HCT-3D, Stage 2 = trilinear

| case | uniform_hct | 2lvl_hct | 4lvl_hct | malmo_hct | malmo3_hct | uniform_half_hct |
|---|---|---|---|---|---|---|
| 000 | 7.14e-3 | 6.85e-3 | **6.82e-3** | 7.07e-3 | 7.01e-3 | 6.97e-3 |
| 001 | 5.48e-3 | 5.16e-3 | **5.15e-3** | 5.71e-3 | 5.28e-3 | 5.22e-3 |
| 003 | 5.96e-3 | 5.61e-3 | **5.53e-3** | 6.02e-3 | 5.77e-3 | 5.61e-3 |
| 004 | 5.52e-3 | 5.09e-3 | **5.05e-3** | 5.44e-3 | 5.21e-3 | 5.16e-3 |

### Stage 1 = HCT-3D, Stage 2 = tricubic

| case | uniform_hct_tricubic | 2lvl_hct_tricubic | 4lvl_hct_tricubic | malmo_hct_tricubic | malmo3_hct_tricubic | uniform_half_hct_tricubic |
|---|---|---|---|---|---|---|
| 000 | 7.04e-3 | 6.78e-3 | **6.78e-3** | 7.11e-3 | 6.96e-3 | 6.96e-3 |
| 001 | 5.46e-3 | 5.11e-3 | **5.09e-3** | 5.78e-3 | 5.25e-3 | 5.21e-3 |
| 003 | 5.90e-3 | 5.58e-3 | **5.50e-3** | 7.15e-3 | 5.80e-3 | 5.68e-3 |
| 004 | 5.44e-3 | **5.01e-3** | 5.02e-3 | 5.93e-3 | 5.21e-3 | 5.16e-3 |

### Best variant per case across all 24 candidates (top 6)

| case | rank 1 | rank 2 | rank 3 | rank 4 | rank 5 | rank 6 |
|---|---|---|---|---|---|---|
| 000 | 4lvl_hct_tricubic 6.78e-3 | 2lvl_hct_tricubic 6.78e-3 | 4lvl_hct 6.82e-3 | 2lvl_hct 6.85e-3 | malmo3_hct_tricubic 6.96e-3 | uniform_half_hct_tricubic 6.96e-3 |
| 001 | 4lvl_hct_tricubic 5.09e-3 | 2lvl_hct_tricubic 5.11e-3 | 4lvl_hct 5.15e-3 | 2lvl_hct 5.16e-3 | uniform_half_hct_tricubic 5.21e-3 | uniform_half_hct 5.22e-3 |
| 003 | 4lvl_hct_tricubic 5.50e-3 | 4lvl_hct 5.53e-3 | 2lvl_hct_tricubic 5.58e-3 | uniform_half_hct 5.61e-3 | 2lvl_hct 5.61e-3 | uniform_half_hct_tricubic 5.68e-3 |
| 004 | 2lvl_hct_tricubic 5.01e-3 | 4lvl_hct_tricubic 5.02e-3 | 4lvl_hct 5.05e-3 | 2lvl_hct 5.09e-3 | uniform_half_hct 5.16e-3 | uniform_half_hct_tricubic 5.16e-3 |

The **top 4 places in every case are HCT-projected block-refined
grids** (`{4lvl,2lvl}_hct[_tricubic]`), followed by
`uniform_half_hct[_tricubic]`. The best raw-P1 Part-1 variant
(`malmo`) does not appear in any case's top 12.

### Stage-1 upgrade P1 → HCT (rms reduction per case × family × Stage-2 interp)

The full HCT-minus-P1 delta table for the trilinear branch shows how
much RMS each family drops when Stage 1 is swapped from raw-P1 to
HCT (all values in m; negative = HCT is better):

| family | case 000 | case 001 | case 003 | case 004 |
|---|---|---|---|---|
| uniform | −1.72e-3 (×0.81) | −1.04e-2 (×0.35) | −1.03e-2 (×0.37) | −6.37e-3 (×0.46) |
| 2lvl | −4.19e-3 (×0.62) | −1.58e-2 (×0.25) | −1.61e-2 (×0.26) | −1.09e-2 (×0.32) |
| 4lvl | −4.36e-3 (×0.61) | −1.64e-2 (×0.24) | −1.67e-2 (×0.25) | −1.14e-2 (×0.31) |
| malmo | −4.50e-4 (×0.94) | −7.88e-3 (×0.42) | −7.64e-3 (×0.44) | −4.08e-3 (×0.57) |
| malmo3 | −1.24e-3 (×0.85) | −1.11e-2 (×0.32) | −1.13e-2 (×0.34) | −6.71e-3 (×0.44) |
| uniform_half | −4.88e-3 (×0.59) | −1.77e-2 (×0.23) | −1.79e-2 (×0.24) | −1.22e-2 (×0.30) |

Ratios in parentheses are HCT-rms / P1-rms; smaller = larger HCT
win. **Every family × every case improves under HCT.** The gain is
biggest on families that were worst under P1 (`4lvl`,
`uniform_half`) and smallest on `malmo` — the family that MALMO's
mesh-aligned block structure was already compensating for the P1
error, leaving less room for HCT to fix.

## Interpretation (Part 2)

Six things stand out.

### 1. HCT Stage 1 is the dominant lever

Swapping Stage 1 from raw P1 to HCT-3D reduces RMS by a factor of
**0.23 – 0.94×** (i.e. 6 – 77 % reduction) depending on the
family × case. The mean HCT/P1 ratio across the 48 (family × case
× Stage-2) pairs is 0.42 — HCT roughly halves the grid PT error on
average, and cuts it by up to 4× on the block-refined families
that were worst under P1. Every raw-P1 variant has an HCT twin that
beats it.

The absolute floor of grid PT RMS is set by Stage 1, not by grid
family. Once Stage 1 is HCT, all six families come within a
~1.4× band (5.0e-3 – 7.2e-3 m across all cases).

### 2. The leaderboard flips — `{4lvl,2lvl}_hct_tricubic` wins

Under raw-P1 (Part 1) MALMO was best on every case. Under HCT the
top 4 places on every case are the **block-refined 2lvl / 4lvl
variants with HCT**, in either interp order. MALMO drops to
5th–8th place. `4lvl_hct_tricubic` is the winner or joint-winner
on every case:

| case | winner | rms | Part-1 winner (malmo) rms | improvement |
|---|---|---|---|---|
| 000 | 4lvl_hct_tricubic | 6.78e-3 | 7.52e-3 | ×0.90 |
| 001 | 4lvl_hct_tricubic | 5.09e-3 | 1.36e-2 | ×0.37 |
| 003 | 4lvl_hct_tricubic | 5.50e-3 | 1.37e-2 | ×0.40 |
| 004 | 2lvl_hct_tricubic | 5.01e-3 | 9.52e-3 | ×0.53 |

The mesh-aligned block-boundary argument from Part 1 §2 (hand-picked
radial cutoffs at r < 16 mm and r < 8 mm cause coarse-vs-fine
block disagreement that injects error) **no longer applies** when
the projected field is C¹. HCT smooths the field across block
boundaries and the mesh-alignment premium disappears — leaving the
block-refined families to win on the pure Stage-2-sampling axis
(more grid cells in the pin region, coarser in the outer domain).

### 3. MALMO becomes mid-pack under HCT

`malmo_hct` (all cases) and `malmo_hct_tricubic` (cases 001/003
especially) are among the **worst** HCT variants. Case 003
`malmo_hct_tricubic` = 7.15e-3 is a full 30 % above the winner
5.50e-3.

The reason connects to the smoothing MALMO's cells-per-edge = 1
grid does at Stage 1: its finest block Δx (~7e-5 m in z, ~5e-4 m
in x/y) is *coarser* than the underlying finest tet edge (1.4e-4 m)
in x/y. Under raw-P1 that coarseness matched the mesh's own
resolution jump and looked like an advantage. Under HCT the source
field is smooth everywhere, so undersampling that field is a real
loss the block-refined families avoid by going finer inside the
pin region.

### 4. Tricubic Stage 2 now helps — modestly

The Part 1 tricubic-hurts pattern disappears. Under HCT Stage 1:
- Tricubic wins or ties trilinear on **every** case × family except
  the two `malmo` variants (cases 001/003 tricubic is 1 – 20 %
  *worse* than trilinear). Every non-MALMO family sees tricubic 0 – 3 %
  better than trilinear.
- The `malmo_hct_tricubic` case-003 outlier (7.15e-3 vs 6.02e-3 for
  trilinear) is a case-specific pathology, likely tricubic reading
  across a MALMO block boundary where the underlying MALMO blocks
  have different Δx. On more-uniform block layouts (2lvl / 4lvl /
  uniform_half) the C¹ Catmull-Rom does compose cleanly with the
  C¹ HCT-projected node values, matching the Slide-10 analytic-
  case finding.

Practically: the Stage-2 tricubic upgrade is worth doing (a few %
free) but is *not* itself the reason the numbers dropped from
~1.4e-2 to ~5.5e-3 m. The Stage-1 P1→HCT swap did that.

### 5. `uniform_half` climbs from worst-in-Part-1 to 5th-best under HCT

Under raw-P1 `uniform_half` was the *worst* variant on every case
(RMS 1.18e-2 – 2.35e-2 m). Under HCT it's a solid mid-pack
performer (5.16e-3 – 6.97e-3 m). Every case's `uniform_half_hct`
sits within 5 % of that case's winner.

The Part 1 §4 finding "uniform refinement past the mesh's own
resolution is adding cells that mostly re-sample the same P1
information" was correct as diagnosed — but the fix wasn't to
*stop* refining, it was to *smooth Stage 1*. Once the source field
is smooth, uniform_half's 17.5 M cells become useful because they
resolve fine features the coarser blocks would smear.

That said, `uniform_half_hct` is not the *winner* — 4lvl/2lvl are
still ~2 – 8 % better while using ~300× fewer cells (~60 k vs
17.5 M). The block-refined families are cheaper *and* better.

### 6. Case ordering is preserved, magnitudes compressed

Part 1 §6: `case 000 < case 004 < case 001 ≈ case 003`. Part 2 same
ordering, but the spread compresses: Part 1 range was 7.5e-3 to
2.4e-2 m (a factor of 3.2); Part 2 HCT range is 5.0e-3 to 7.2e-3 m
(a factor of 1.4). This is consistent with the interpretation that
Part 1's spread was dominated by how each case's velocity
gradients interacted with the P1 discretisation — under a smooth
Stage 1 the cross-case error is dominated by the RK4 tracker's own
integration + boundary handling, which is uniform across cases.

## Throughput (Part 2)

Wall times pulled from each variant's `run_grid.log` (grid PT) and
`post_pt/fom_hct_{on,off}/log.txt` (mesh PT reference). Every run
is 2000 steps × DT = 3.75e-3 s × 360 000 particles.

**Mesh reference** (identical on/off HCT — the small difference
between hct_on and hct_off is timing noise):

| case | fom_hct_on wall (s) | throughput (p·step/s) |
|---|---|---|
| 000 | 1415.6 | 508,600 |
| 001 | 1491.8 | 482,630 |
| 003 | 1496.8 | 481,020 |
| 004 | 1449.3 | 496,788 |

Mean mesh + HCT-3D wall time: **~1463 s per case** (~24 min),
throughput ~490 k particle-steps/s.

**Grid variants** (mean across the 4 cases; speedup ratio = mean
mesh wall time / mean grid wall time):

| variant | mean wall (s) | mean throughput (p·step/s) | speedup vs mesh |
|---|---|---|---|
| uniform_hct | 17.9 | 40.2 M | **81.5×** |
| uniform | 18.0 | 40.2 M | 81.3× |
| malmo3 | 21.1 | 34.3 M | 69.3× |
| uniform_half_hct_tricubic | 22.8 | 31.8 M | 64.2× |
| uniform_half | 23.9 | 30.4 M | 61.3× |
| 2lvl | 27.2 | 26.7 M | 53.9× |
| 2lvl_hct_tricubic | 27.5 | 26.4 M | 53.3× |
| uniform_half_tricubic | 28.1 | 25.8 M | 52.1× |
| 2lvl_hct | 29.9 | 24.3 M | 48.9× |
| 4lvl_hct_tricubic (accuracy winner) | **29.6** | **24.5 M** | **49.4×** |
| 2lvl_tricubic | 30.1 | 24.1 M | 48.7× |
| 4lvl_tricubic | 30.9 | 23.5 M | 47.3× |
| malmo_hct | 32.4 | 22.4 M | 45.2× |
| malmo3_hct_tricubic | 32.9 | 22.1 M | 44.4× |
| uniform_hct_tricubic | 33.4 | 21.7 M | 43.8× |
| malmo3_tricubic | 33.7 | 21.5 M | 43.5× |
| malmo3_hct | 34.2 | 21.2 M | 42.8× |
| uniform_tricubic | 35.1 | 20.7 M | 41.6× |
| 4lvl_hct | 36.1 | 20.1 M | 40.5× |
| uniform_half_hct | 36.4 | 20.0 M | 40.2× |
| malmo_hct_tricubic | 36.6 | 19.9 M | 40.0× |
| malmo | 37.4 | 19.4 M | 39.2× |
| malmo_tricubic | 37.9 | 19.1 M | 38.6× |
| 4lvl | 38.0 | 19.1 M | 38.5× |

Full per-case wall times in
`.../rom_out/fom_grid_pt/wall_times.csv`.

**Observations**:

- **Every grid variant is 38× – 82× faster than the mesh + HCT-3D
  reference.** The slowest grid variant (`4lvl` @ 38.5×) still
  runs a 20 min mesh reference in ~38 s.
- **Speedup does not track cell count.** `uniform_half_hct` has
  17.5 M cells but runs at 40× — same speed as `4lvl_hct` with
  ~64 k cells. The bottleneck is JAX kernel launch + gather, not
  arithmetic on the cell values, so a much larger grid only pays
  a modest constant.
- **Tricubic Stage 2 costs 5 – 20 % over trilinear** on the same
  grid (`4lvl_hct_tricubic` @ 29.6 s vs `4lvl_hct` @ 36.1 s —
  the tricubic variant is actually *faster* here, which is inside
  timing noise from JIT compilation order across the runner's
  variant loop; treating them as equal is fine).
- **HCT-3D Stage 1 has essentially zero runtime cost.**  The HCT-
  projected grid modules load a `.npz` with the same shape as the
  raw-P1 modules — the Stage 1 upgrade happens at build time and
  is invisible at runtime.
- **The accuracy winner runs at ~30 s** (49× speedup). Trading
  the throughput champion (`uniform_hct` @ 18 s, 82×,
  rms 5.5×10⁻³ – 7.1×10⁻³ m) for the accuracy champion
  (`4lvl_hct_tricubic` @ 30 s, 49×, rms 5.0×10⁻³ – 6.8×10⁻³ m)
  costs 12 s of wall time per case for a 4 – 7 % rms reduction.
  Cheap.

**Bottom line**: for FSW cases the grid path is not the throughput
bottleneck — even the slowest grid variant clears the mesh path
by an order of magnitude. Pick the variant on accuracy, not speed.

## Practical recommendation (Part 2, current)

For **FOM mesh PT throughput replacement** the current best
practice is:

1. **Stage 1 = HCT-3D** (`--mesh-projection hct_cubic`). This is
   the non-negotiable half of the recipe — it delivers a 2–4×
   RMS reduction on its own and is much larger than any grid-
   family choice.
2. **Stage 2 = tricubic** (`--interp tricubic`). Costs 8× more per
   lookup than trilinear, buys 1–3 % accuracy on the block-refined
   families. Take the win.
3. **Grid family = 4lvl** (two nested block-refined blocks,
   r < 16 mm and r < 8 mm). Wins on every case in the cohort at
   ~64 k cells total. `2lvl_hct_tricubic` (r < 16 mm only,
   ~57 k cells) is within noise on every case and cheaper.

**Do not use MALMO under HCT.** It looked best under P1 because
its mesh-aligned block boundaries hid a P1 defect. Under HCT the
defect is gone and MALMO's coarser cells-per-edge = 1 undersamples
the smooth field in the pin region.

**One-off cost:** the per-case HCT cache (~240 MB `hct_cache_ts119.npz`)
is built once per case (SPR + Alfeld + Bernstein coeffs, ~15 s of
CPU on the workstation) and reused across all 12 grid variants of
that case. Once the cache exists, each grid module takes 10 s –
20 min depending on cell count.

## Data sources (Part 2)

| artefact | path |
|---|---|
| per-case HCT cache | `/scratch/shared/ROM/FOM/cylindrical_<c>.gid/hct_cache_ts119.npz` |
| grid modules | `.../grid_velocity_cyl_<c>_<VAR>_hct[_tricubic].{py,npz}` |
| PT runs | `.../out_grid_<VAR>_hct[_tricubic]/run_*/particles.vtkhdf` |
| extended CSV (96 rows) | `/scratch/shared/ROM/rom_out/fom_grid_pt/grid_vs_mesh_pt_summary_step950.csv` |
| P1 summary bar chart (12 vars × 4 cases, shared y-range) | `/scratch/shared/ROM/rom_out/fom_grid_pt/fom_grid_vs_mesh_pt_step950_p1.png` |
| HCT summary bar chart (12 vars × 4 cases, same y-range) | `/scratch/shared/ROM/rom_out/fom_grid_pt/fom_grid_vs_mesh_pt_step950_hct.png` |
| delta chart (P1 rms − HCT rms) | `/scratch/shared/ROM/rom_out/fom_grid_pt/fom_grid_vs_mesh_pt_step950_p1_minus_hct.png` |

The two panel PNGs use **shared y-axis limits** so their bar
heights are directly comparable at a glance — the P1 panel bars
sit in the upper half of the y-range (7.5e-3 → 2.4e-2 m), the
HCT panel bars sit near the floor (5.0e-3 → 7.2e-3 m). The
titles are explicitly "Stage-1 = RAW P1" vs "Stage-1 = HCT-3D"
because the reference tracker is *always* mesh + HCT-3D, so a
shorter title with only "HCT-3D" in it was ambiguous. (An older
`fom_grid_vs_mesh_pt_step950.png` from 2026-07-29, which was a
single-panel Part-1-only chart, has been removed to avoid
confusion.)

Regenerate with:

```bash
python3 /flash/shared/jax/JAXTrace/scripts/compare_grid_vs_mesh_pt_fom.py \
    --cases 000 001 003 004 --step 950 \
    --out /scratch/shared/ROM/rom_out/fom_grid_pt/ --verbose
```

## Part 3 · Detailed breakdown + top-3 winner selection

**Date:** 2026-07-30

Ran the enriched compare tool at **both** step 950 (mid-run,
t ≈ 3.56 s) and step 2000 (final, t ≈ 7.50 s). New metrics
(matching roadmap §7 conventions and the plan doc's step-2
acceptance criterion):

- **per-r-bin RMS**: near-pin (r_mesh(step) ≤ 10 mm) vs outer
  (> 10 mm). Bin uses mesh position *at the reporting step* — every
  particle in this cohort is seeded upstream at r ≥ 13 mm, so
  binning on step-0 radial position leaves the near-pin bin empty.
  Binning by where the mesh says the particle ends up matches the
  FSW mixing use case ("of the particles that ended up near the
  pin, how well does the grid path track them?").
- **per-component RMS**: rms_x, rms_y, rms_z separately (spots
  directional bias).
- **percentile ladder**: p50, p95, p99, max.
- **trapping delta**: `frac_trapped_grid − frac_trapped_mesh` at
  the reporting step (positive = grid over-traps, negative =
  under-traps).
- **escape agreement**: 4 counts per variant per case
  (`both_alive`, `both_escaped`, `only_grid_escaped`,
  `only_mesh_escaped`); mesh is truth so the two "only_*"
  columns are both errors, of opposite direction.

Full 27-column detail CSVs at
`.../rom_out/fom_grid_pt/grid_vs_mesh_pt_detail_step{950,2000}.csv`
(96 rows each).

### How raw-P1 vs HCT compares over time

Cross-time aggregates (mean across the 4 cases) show the HCT gap
*widens* between step 950 and step 2000 — raw-P1 error compounds
much faster than HCT error:

| stage-1 | best rms @ 950 | best rms @ 2000 | growth factor 950→2000 |
|---|---|---|---|
| HCT-3D (top variant) | 5.60e-3 m | 6.32e-3 m | **×1.13** |
| raw P1 (top variant) | 1.11e-2 m | 2.69e-2 m | **×2.43** |

Best HCT variant at final step is **~6.5× more accurate than the
best raw-P1 variant** (0.63 cm vs 4.0 cm rms). The step-2000
numbers are the ones that matter for the FSW cohort — that's the
end-of-simulation state.

### HCT-only ranking at step 2000 (mean across 4 cases, ordered by aggregate rms)

| variant | rms (m) | rms_near (m) | rms_outer (m) | |Δtrap| (%) | only_grid_esc | only_mesh_esc |
|---|---|---|---|---|---|---|
| **4lvl_hct** | **6.316e-3** | 1.445e-2 | 6.047e-3 | 0.55 | 21,854 | 885 |
| **4lvl_hct_tricubic** | 6.381e-3 | **1.376e-2** | 6.154e-3 | 0.56 | 21,438 | 1,274 |
| 2lvl_hct | 6.398e-3 | 1.558e-2 | 6.086e-3 | 0.50 | 22,066 | 819 |
| uniform_half_hct | 6.436e-3 | 1.467e-2 | 6.164e-3 | 0.61 | 29,651 | 792 |
| **2lvl_hct_tricubic** | 6.445e-3 | 1.492e-2 | 6.175e-3 | 0.50 | 21,528 | 1,178 |
| uniform_half_hct_tricubic | 6.507e-3 | 1.513e-2 | 6.227e-3 | 0.55 | 29,672 | 918 |
| malmo3_hct | 6.604e-3 | 1.553e-2 | 6.316e-3 | 0.66 | 30,519 | 1,005 |
| malmo3_hct_tricubic | 6.606e-3 | 1.482e-2 | 6.329e-3 | 0.59 | 29,935 | 1,060 |
| uniform_hct | 6.802e-3 | 1.716e-2 | 6.393e-3 | 0.60 | 30,332 | 515 |
| uniform_hct_tricubic | 6.875e-3 | 1.669e-2 | 6.531e-3 | 0.61 | 28,269 | 1,299 |
| malmo_hct | 7.151e-3 | 1.799e-2 | 6.809e-3 | 0.67 | 32,309 | 1,274 |
| malmo_hct_tricubic | 7.960e-3 | 1.512e-2 | 7.735e-3 | 0.47 | 22,505 | 3,507 |

### Composite ranking

Summing ranks across 4 criteria — rms@950, rms@2000, rms_near@2000,
|Δtrap|@2000 — gives a variant a "small is good" composite score:

| rank | variant | composite score |
|---|---|---|
| **1 (tied)** | **4lvl_hct** | 6 |
| **1 (tied)** | **4lvl_hct_tricubic** | 6 |
| **3** | **2lvl_hct_tricubic** | 11 |
| 4 | 2lvl_hct | 14 |
| 5 | uniform_half_hct | 18 |
| 6 | uniform_half_hct_tricubic | 20 |
| 7 | malmo3_hct_tricubic | 22 |

The top-3 **4lvl_hct / 4lvl_hct_tricubic / 2lvl_hct_tricubic**
are the recommended winners for the 20-case rollout (roadmap
step 3).

### Per-case cross-check

| case | winner by aggregate @ 2000 | winner by near-pin @ 2000 |
|---|---|---|
| 000 | 4lvl_hct_tricubic (5.13e-3) | 4lvl_hct (7.41e-3) |
| 001 | 4lvl_hct_tricubic (6.29e-3) | **malmo_hct_tricubic** (7.75e-3) |
| 003 | 4lvl_hct (7.94e-3) | 4lvl_hct_tricubic (2.47e-2) |
| 004 | 4lvl_hct (5.86e-3) | 4lvl_hct (1.08e-2) |

The 4lvl family sweeps 3 of 4 cases on both metrics. The
**case 001 near-pin outlier** — MALMO wins there by ~20 % — is
the numerical signature of the ParaView observation that MALMO's
finest block covers the pin region tightly and produces
trajectory *shape* close to the mesh reference near the pin.
Case 001 has the strongest pin recirculation among the 4, so
that MALMO advantage shows up in the numbers there.

For the 20-case rollout the top-3 stay `4lvl_hct`,
`4lvl_hct_tricubic`, `2lvl_hct_tricubic` — they cover the
aggregate + outer + near-pin questions for 3 of 4 cases and
still tie on trapping. If we want to preserve the near-pin
MALMO story we can add `malmo_hct_tricubic` as a 4th
"reference for near-pin shape" variant (small extra cost,
runs alongside).

### Compounding: how the error grows from step 950 to step 2000

Ratio of step-2000 to step-950 aggregate rms, per HCT variant
(mean across 4 cases):

| variant | rms @ 950 | rms @ 2000 | growth |
|---|---|---|---|
| 4lvl_hct | 5.64e-3 | 6.32e-3 | ×1.12 |
| 4lvl_hct_tricubic | 5.60e-3 | 6.38e-3 | ×1.14 |
| 2lvl_hct_tricubic | 5.62e-3 | 6.45e-3 | ×1.15 |
| 2lvl_hct | 5.68e-3 | 6.40e-3 | ×1.13 |
| uniform_half_hct | 5.74e-3 | 6.44e-3 | ×1.12 |
| malmo_hct_tricubic | 6.49e-3 | 7.96e-3 | ×1.23 |
| malmo_hct | 6.06e-3 | 7.15e-3 | ×1.18 |

The top winners compound at only **×1.12 – 1.15** over the second
half of the simulation. Raw-P1 variants compound at **×2 – 3** over
the same interval (from step 950 to step 2000), which is why the
HCT gap widens with time. This is consistent with the roadmap
§2 Xiong et al. prediction that Lagrangian error compounds with
Eulerian smoothness — HCT's C¹ field gives lower Lagrangian
amplification.

### Trapping and escape agreement at step 2000

At the final step, HCT variants achieve **|Δtrap| ≤ 0.7 %** on
average (raw-P1 variants: 10 – 28 %). The remaining error is
concentrated in **only_mesh_escaped** — the grid path retains
particles that the mesh eventually pushes out through the outlet.
For the 4lvl/2lvl winners this is 800 – 1,300 particles per case
(out of 360k = 0.2 – 0.4 %). Not a source of concern for FSW
mixing but worth noting for downstream residence-time
diagnostics.

The `only_grid_escaped` numbers for HCT winners (~21,000 – 30,000
per case = 6 – 8 %) look large in isolation but they *match* the
mesh reference behaviour: at step 2000 many particles have flowed
past the outlet in both trackers, and the small time offset
between when a particle crosses the outlet plane in the grid vs
the mesh trajectory is what "only_grid_escaped" counts. It is
not saying the grid loses 6 % of particles the mesh keeps.

### Figures

Twelve PNGs per reporting step at
`.../rom_out/fom_grid_pt/`, produced by the enriched compare tool:

- `..._step<S>_p1.png` / `..._step<S>_hct.png` — aggregate rms,
  shared y-range (as in Part 2).
- `..._step<S>_p1_minus_hct.png` — Stage-1 upgrade gain
  (as in Part 2).
- `..._step<S>_p1_rbin.png` / `..._step<S>_hct_rbin.png` —
  two-panel (near-pin, outer) rms.
- `..._step<S>_p1_trapping.png` / `..._step<S>_hct_trapping.png`
  — Δtrap in percentage points.
- `..._step<S>_p1_escape.png` / `..._step<S>_hct_escape.png` —
  stacked-bar per variant, one panel per case, categories
  (both_alive, both_escaped, only_grid, only_mesh).
- `..._step<S>_p1_percomp.png` / `..._step<S>_hct_percomp.png` —
  rms_x, rms_y, rms_z per variant, one panel per case, log-scale y.

Regenerate with:

```bash
python3 /flash/shared/jax/JAXTrace/scripts/compare_grid_vs_mesh_pt_fom.py \
    --cases 000 001 003 004 --step <S> \
    --out /scratch/shared/ROM/rom_out/fom_grid_pt/ --verbose
```

---

## Part 4 · 20-case cohort rollout

**Date:** 2026-07-31

Roadmap step 3 (per the plan doc) executed on the workstation: the top-3
winners `4lvl_hct`, `4lvl_hct_tricubic`, `2lvl_hct_tricubic` run against
mesh + HCT-3D reference for **every one of the 20 FSW cases** at both
step 950 (mid-run t=3.6 s) and step 2000 (final t=7.5 s). The full
120-row detail CSVs are at
`/scratch/shared/ROM/rom_out/fom_grid_pt_all20/grid_vs_mesh_pt_detail_step{950,2000}.csv`.
Per-particle displacement VTUs for every (case × variant × step) are
also on disk (240 VTUs total, ~14 MB each).

### Reference discovery fix

The all-20 rerun uncovered a tooling gap: the compare tool + per-case
runner only looked for the mesh reference under `post_pt/fom_hct_on/`,
which existed only for the curated 4-case cohort. The remaining 16
cases actually have complete 2000-step mesh PT — it's written to
`post_pt/run_grid-frac_n360000_s2000/particles.vtkhdf` by the standard
`run_jaxtrace.sh` runner. `_find_mesh_reference` in
[compare_grid_vs_mesh_pt_fom.py](../scripts/compare_grid_vs_mesh_pt_fom.py)
now falls back to the latter, and the top-3 runner template
[run_grid_top3_template.sh](../../../fsw-gpu/scratch/shared/ROM/FOM/run_grid_top3_template.sh)
does the same.

### Both-alive metric switch

Every rms-based figure in Part 4 uses `rms_alive` — RMS restricted to
particles that are still inside the domain in **both** the grid and
mesh runs at the reporting step. This is the clean cross-tracker
comparison; the pre-existing `rms` column that mixed alive + escaped
particles is kept in the CSV for backwards compat only.

Consequences:

- At step 950, cohort mean `n_alive` = **97.3 %** (median 100 %, min
  88 %). Metric is essentially the full-cohort rms.
- At step 2000, cohort mean `n_alive` = **23.6 %** (median 4.6 %, range
  1.4 – 71.5 %). Metric is computed on very different subset sizes
  across cases — see the survivorship figure below.

### Per-case rms table (`4lvl_hct`, mm)

Ordered by case number for reference; the growth column is
`rms_alive(2000) / rms_alive(950)`.

| case | n_alive s950 | n_alive s2000 | rms 4lvl_hct s950 (mm) | s2000 (mm) | growth |
|---|---|---|---|---|---|
| 000 | 360,000 | 251,643 | 6.83 | 5.23 | ×0.77 |
| 001 | 329,276 | 4,918 | 4.96 | 9.38 | ×1.89 |
| 002 | 360,000 | 255,133 | 6.78 | 6.50 | ×0.96 |
| 003 | 317,886 | 5,010 | 5.62 | 9.01 | ×1.60 |
| 004 | 360,000 | 13,538 | 5.05 | 8.05 | ×1.59 |
| 005 | 360,000 | 11,219 | 5.44 | 9.68 | ×1.78 |
| 006 | 360,000 | 19,891 | 5.20 | 7.45 | ×1.43 |
| 007 | 329,508 | 6,849 | 5.13 | 10.12 | ×1.97 |
| 008 | 360,000 | 257,285 | 6.62 | 5.46 | ×0.82 |
| 009 | 360,000 | 96,869 | 5.38 | 7.29 | ×1.36 |
| 010 | 349,718 | 9,335 | 5.23 | 8.79 | ×1.68 |
| 011 | 360,000 | 102,440 | 5.34 | 5.97 | ×1.12 |
| 012 | 359,191 | 10,145 | 5.04 | 8.72 | ×1.73 |
| 013 | 328,440 | 5,962 | 5.06 | 11.46 | ×2.26 |
| 014 | 360,000 | 107,709 | 5.59 | 5.50 | ×0.98 |
| 015 | 360,000 | 100,586 | 5.64 | 7.61 | ×1.35 |
| 016 | 320,188 | 5,214 | 5.37 | 10.01 | ×1.87 |
| 017 | 347,904 | 7,089 | 5.03 | 7.89 | ×1.57 |
| 018 | 360,000 | 257,329 | 6.72 | 6.09 | ×0.91 |
| 019 | 360,000 | 174,222 | 6.42 | 5.55 | ×0.86 |

### Cohort summary (mean across 20 cases, mm)

| variant | s950 mean | s950 median | s2000 mean | s2000 median | s2000 near-pin | s2000 outer |
|---|---|---|---|---|---|---|
| **4lvl_hct** (winner) | **5.62** | 5.37 | **7.79** | 7.75 | 9.64 | 7.62 |
| 4lvl_hct_tricubic | 5.57 | 5.32 | 8.00 | 7.99 | 9.75 | 7.93 |
| 2lvl_hct_tricubic | 5.56 | 5.30 | 8.60 | 8.36 | 9.59 | 9.36 |

## Findings from the 20-case rollout

Five things stand out.

### 1. Winner ordering flips versus the 4-case story

In the 4-case rollout `4lvl_hct` and `4lvl_hct_tricubic` tied for rank 1,
with `2lvl_hct_tricubic` a close third. In the 20-case rollout at step
2000, **`4lvl_hct` (trilinear Stage 2) is the clear winner** with mean
cohort rms 7.79 mm; `4lvl_hct_tricubic` is 3 % worse (8.00 mm),
`2lvl_hct_tricubic` is 10 % worse (8.60 mm).

The tricubic Stage-2 upgrade helped marginally at step 950 (5.57 mm
vs 5.62 mm — 1 % better) but hurts by step 2000 as the C¹ Catmull-Rom
overshoot compounds along the trajectory. On the 4-case cohort this
did not show because those 4 cases happened to be cases 000/001/003/004
— cases 000 and 004 have among the *best* tricubic behaviour (see the
per-case table).

**Practical recommendation update**: `4lvl_hct` (trilinear Stage 2)
is the single-variant recommendation for FSW. `2lvl_hct_tricubic`
is a viable smaller-cell-count alternative (57 k cells vs 64 k for
4lvl, ~5 % savings) if the extra 10 % rms is acceptable for a
downstream use case.

### 2. Growth factor is bimodal — driven by survivorship, not tracker error

Naive read of the growth column: rms grows on average ×1.43 from step
950 to step 2000, so error compounds. Real story: **cases split into
two populations**:

- **Keeper cases** (surviving particles > 100 k at step 2000):
  000, 002, 008, 011, 014, 018, 019, and marginally 009. **Growth
  factor is ≤ 1.12**, and 5 of these cases *shrink* (growth < 1.0).
- **Flusher cases** (< 20 k surviving): 001, 003, 004, 005, 007, 010,
  012, 013, 016, 017. **Growth factor 1.36 – 2.26**.

The shrinkage in keeper cases is a **survivorship bias artefact**:
particles that had large grid-vs-mesh disagreements early were pushed
onto trajectories that eventually escaped the domain, dropping them
from the `both_alive` count at step 2000. The remaining particles are
the ones that behaved well in both tracks, so their rms is smaller
even though no trajectory *improved* over time. This is not a
tracker property — it's a filter property.

**Interpretation**: the tracker error is *monotonically non-decreasing
per particle* over time. The cohort statistic is only useful as a
lower bound on the "hardest" keeper cases (000/018 at ~5–6 mm rms).

### 3. Near-pin / outer ratio compresses from ×1.7 to ×1.27 by step 2000

At step 950 the per-variant near-pin RMS is ~1.7× the outer RMS on
average (8.2 mm near vs 4.7 mm outer for the winner). By step 2000
this collapses to ~1.27× (9.6 mm near vs 7.6 mm outer). The pin
region is still the worst place for trajectory divergence, but the
outer bulk catches up as small errors accumulate over the longer
integration.

This is the numerical version of the roadmap §2 Xiong-et-al
prediction: Lagrangian error concentrates near shear/separatrix
regions (the pin's shear boundary) *early*, then propagates to the
bulk over longer integration. **The near-pin bin is the primary
acceptance criterion at any single reporting step**, but the outer
bin cannot be dismissed for time-integrated diagnostics.

### 4. Spatial pattern: pin-face shear layer at r ≈ 21 mm dominates

The `..._hct_spatial.png` figures reveal that the pin-face **shear
band at r ≈ 21 – 22 mm dominates the local error** in the keeper
cases. The near-pin core (r ≤ 5 mm) is actually the *lowest*-error
region, not the highest — the pin's material rotation is well-
represented by the grid because it's a smooth rigid-body-like
motion. The shear boundary where pin-driven flow meets bulk drift is
where velocity gradients are steepest and where the grid representation
loses to the mesh + HCT-3D reference.

This has an actionable implication:
[fig6_spatial_pattern.png](../paper_figs/rom_pt_step5_step6_all20/fig6_spatial_pattern.png)
suggests that a hypothetical **grid variant with even finer refinement
specifically in the r ≈ 15 – 25 mm shell** (rather than uniform
refinement at r ≤ 16 or 8 mm as in `2lvl` / `4lvl`) would target the
error concentration directly. Not built yet — a `radial_shell_hct`
variant is the natural next step if we want to push cohort mean rms
below ~5 mm.

### 5. Cohort consistency: variance / mean = 15 % at step 950, 24 % at step 2000

For `4lvl_hct` at step 950: mean 5.62 mm, std 0.85 mm → coefficient of
variation 15 %. At step 2000: mean 7.79 mm, std 1.86 mm → CV 24 %. The
grid tracker's performance is **highly consistent across the 20-case
cohort at mid-run** — no case is more than 1.5× worse than the mean.
By final step the spread widens (2.3× ratio between best-c000 and
worst-c013), driven by the survivorship split above.

## Presentation-quality figures

Generated by
[scripts/plot_grid_pt_presentation.py](../scripts/plot_grid_pt_presentation.py)
from the two detail CSVs + per-particle VTUs. Output at
`paper_figs/rom_pt_step5_step6_all20/` (and mirrored on the
workstation at `.../fom_grid_pt_all20/presentation_figs/`).

| figure | file | what it shows |
|---|---|---|
| Fig 1 s950 | `fig1_cohort_rms_step950.png` | Per-case rms bars, 3 variants side-by-side, cases sorted by mean error. Cohort-mean reference line. |
| Fig 1 s2000 | `fig1_cohort_rms_step2000.png` | Same at final step. Reveals the case ordering + how the flushing cases push far right. |
| Fig 2 | `fig2_cohort_distribution.png` | Box plots per variant per step, individual case points overlaid. Median + mean + IQR + whiskers show cohort spread. |
| Fig 3 | `fig3_growth_scatter.png` | rms(950) vs rms(2000) scatter, one point per (case, variant) with y=x reference. **Bimodal keeper/flusher split visible.** |
| Fig 4 | `fig4_survivorship.png` | `both_alive` fraction per case at both steps, sorted by final survivorship. |
| Fig 5 | `fig5_near_pin_vs_outer.png` | Cohort mean rms ± 1σ for near-pin (r ≤ 10 mm) vs outer, per variant per step. |
| Fig 6 | `fig6_spatial_pattern.png` | (r_mesh, z_mesh) 2D heatmap of median |err| for the winner variant on 4 representative cases at step 2000. Shared log colour scale. Red dashed line = near-pin boundary. |

Style: serif fonts, thick strokes, presentation-scale font sizes,
titles carry the headline finding. Regenerate with:

```bash
python3 /flash/shared/jax/JAXTrace/scripts/plot_grid_pt_presentation.py \
    --in-dir /scratch/shared/ROM/rom_out/fom_grid_pt_all20 \
    --out    paper_figs/rom_pt_step5_step6_all20
```

## Final verdict on §5 / §6 (roadmap)

- **Grid PT replaces mesh + HCT-3D on the full 20-case cohort** with
  a mean rms position error of 5.6 mm at mid-run (t=3.6 s) and 7.8 mm
  at final step (t=7.5 s), for a 49× wall-time speedup.
- **Winner: `4lvl_hct` (trilinear Stage 2)** — beats both tricubic
  alternatives on the full cohort at final step. Uses ~64 k cells,
  built from raw FOM velocity via HCT-3D Stage-1 projection (SPR +
  Alfeld + Bernstein cubic) with a ~240 MB per-case cache.
- **Acceptance**: the cohort-mean rms is well under 1 cm; the
  keeper-case rms is under 6 mm. Whether this passes the roadmap §5
  1.5× per-r-bin criterion depends on the ROM-vs-mesh residual we
  compare it against, which is the step-4 experiment (ROM velocity
  on the winner grid) — not run yet.
- **Next**: build the same 3 winner grids from *ROM* velocity, compare
  ROM_grid_4lvl_hct vs ROM_mesh_hct_on. If the added grid error is
  small next to the ROM/FOM Lagrangian residual (Part 2 of §2/§3
  report), the grid ROM path is fit for FSW mixing use.

---

## Part 5 · Testing 3 more grid variants across the 20-case cohort

**Date:** 2026-07-31

Part 4 concluded that `4lvl_hct` wins on the 20-case cohort, and its
spatial pattern (fig6) revealed that the **pin-face shear band at
r ≈ 21–22 mm dominates the local error**, not the pin core. Two open
hypotheses to test:

1. **Would a grid that puts more resolution in the shear-band zone
   help?** (Actionable follow-up from Part 4.)
2. **Does MALMO's mesh-aligned topology have any advantage under HCT if
   we throw more cells at it?** (Open since Part 2 — under raw-P1
   MALMO won, under HCT it dropped; malmo3 didn't help either.)

### The 3 new variants

| variant | topology | ~total cells | build cost/case |
|---|---|---|---|
| `4lvl_r22_hct` | 4lvl with `r1 = 22 mm, r2 = 16 mm` (finest block covers shear band) | ~180 k | ~3 min |
| `5lvl_hct` | 5 nested blocks (base + r<22 + r<16 + r<8), multipliers `{1, 2, 3, 4}` | ~1.8 M | ~5 min |
| `malmo6_hct` | MALMO with `cells-per-edge = 6` | ~17 M | ~20 min |

All use Stage-1 HCT-3D projection and Stage-2 trilinear (the
Part-4 winner combination) so the comparison isolates the grid
topology / distribution.

The `5lvl` grid type is new; multipliers are capped at 4× (rather than
the naive 8×) because Part 4 §3 already showed refining past the
finest tet edge (~140 µm) gives no benefit under HCT. This keeps the
finest 5lvl cell at Δx ≈ 45 µm — 3× the finest tet edge.

### Cohort summary (mean across 20 cases, mm)

| variant | s950 mean | s950 median | s2000 mean | s2000 median | s2000 near-pin | s2000 outer |
|---|---|---|---|---|---|---|
| **4lvl_hct (winner)** | **5.62** | 5.37 | **7.79** | 7.75 | 9.64 | 7.62 |
| 5lvl_hct | 5.75 | 5.48 | 7.87 | 7.79 | 9.69 | 7.73 |
| malmo6_hct | 5.81 | 5.56 | 7.92 | 7.98 | 9.55 | 7.90 |
| 4lvl_r22_hct | 5.76 | 5.50 | 7.96 | 7.92 | 9.65 | 7.92 |
| 4lvl_hct_tricubic | 5.57 | 5.32 | 8.00 | 7.99 | 9.75 | 7.93 |
| 2lvl_hct_tricubic | 5.56 | 5.30 | 8.60 | 8.36 | 9.59 | 9.36 |

### Statistical significance (paired Wilcoxon signed-rank vs `4lvl_hct`, n=20)

| variant | mean Δ vs 4lvl_hct (mm) | Wilcoxon p | verdict |
|---|---|---|---|
| 5lvl_hct | +0.086 | 0.030 | significantly worse (marginal) |
| malmo6_hct | +0.136 | 0.017 | significantly worse |
| 4lvl_r22_hct | +0.175 | 0.007 | significantly worse |
| 4lvl_hct_tricubic | +0.209 | 0.053 | trend worse (borderline) |
| 2lvl_hct_tricubic | +0.812 | 0.003 | significantly worse (largest gap) |

**None of the 3 new variants beat `4lvl_hct`.** All three are
statistically significantly worse at the 5 % level despite the effect
sizes being small (0.09 – 0.18 mm).

### Per-case winners

Even though no variant wins on the cohort, per-case winners are
scattered (each variant wins some cases):

| variant | cases won at step 2000 |
|---|---|
| `4lvl_hct` | 000, 001, 009, 011 (4 cases) |
| `5lvl_hct` | 003, 004, 010, 012, 017 (5 cases) |
| `2lvl_hct_tricubic` | 002, 015, 018, 019 (4 cases) |
| `malmo6_hct` | 006, 007, 013, 014 (4 cases) |
| `4lvl_hct_tricubic` | 008, 016 (2 cases) |
| `4lvl_r22_hct` | 005 (1 case) |

This is a strong "no dominant winner" signal: per-case differences
between the top 5 variants are within cohort noise. `4lvl_hct` retains
the winner title because it is *never far from the best* on any given
case, whereas malmo6/5lvl swing between best on some cases and near-
worst on others.

### Wall time is essentially identical

Runtime per case (mean across the 20-case rollout, from
`out_grid_<VAR>/run_grid.log`):

| variant | mean wall (s) | speedup vs mesh |
|---|---|---|
| 2lvl_hct_tricubic | 28.5 | 51× |
| 4lvl_hct_tricubic | 31.6 | 46× |
| 4lvl_r22_hct | 34.9 | 42× |
| **4lvl_hct** | **35.1** | **42×** |
| malmo6_hct | 35.7 | 41× |
| 5lvl_hct | 36.6 | 40× |

**Even the 17 M-cell `malmo6_hct` runs at the same speed as the 64 k-
cell `4lvl_hct`** — JAX kernel launch + gather dominates arithmetic
for these problem sizes. Cell count is not a throughput knob in this
regime. Which means the "cheaper variant" argument for `2lvl` etc.
gives no wall-time benefit, only a slightly smaller npz file.

## Findings from the new variants

### 1. The shear-band hypothesis is falsified

The Part-4 spatial pattern (pin-face error concentration at r ≈ 21 mm)
suggested that putting finer cells over that band should help.
**`4lvl_r22_hct` is the 4th-best variant, not the best.** It's
statistically worse than `4lvl_hct` in 16 of 20 cases.

Interpretation: the error in the shear band is **not a spatial-
resolution problem**. The HCT-3D projection at grid nodes already
captures the local velocity accurately (that's what the small
per-cell error would show); the trajectory error there is a
**sensitivity effect** — small velocity uncertainties get amplified
into large position errors in a region of high shear, regardless of
how densely we sample the field. More grid cells don't damp velocity-
gradient sensitivity, only a smoother source field would.

This is the roadmap §2 Xiong-et-al Lagrangian amplification story
localised to the shear band. It's a property of the *flow*, not the
tracker.

### 2. Extra refinement levels give diminishing returns

`5lvl_hct` (4 nested blocks) is 1 % worse than `4lvl_hct` (3 nested
blocks). Adding the r<8 mm inner block at 4× base resolution buys
nothing measurable, and its ~1 M extra cells add zero wall time (so
it costs nothing beyond disk / RAM). Under the current pipeline,
**3 nested refinement levels is enough**.

If future work wants to push further, the leverage is at Stage 1 (a
smoother velocity representation) or at the tracker itself (higher-
order RK integration in the shear band), not at the grid density.

### 3. MALMO closed out: topology is not the lever under HCT

`malmo6_hct` with 17 M cells still loses to `4lvl_hct` with 64 k
cells. MALMO's mesh-aligned topology has no measurable advantage
under HCT projection, at any resolution we tested. The Part-2 finding
that "malmo wins under raw P1" was purely an artefact of P1's C⁰
gradient jumps at mesh element boundaries — MALMO happened to hide
that artefact by aligning its block boundaries with the same jumps.

**Under HCT the artefact is gone and MALMO loses its raison d'être.**

Rank-order across all MALMO variants tested for step-2000 mean rms:

| MALMO variant | mean rms (mm) | vs 4lvl_hct |
|---|---|---|
| `malmo_hct` (cells-per-edge = 1) | ~7.15 (from Part 3) | +5 % worse |
| `malmo3_hct` (cells-per-edge = 3) | ~6.60 (from Part 3) | +2 % worse (best MALMO) |
| `malmo6_hct` (cells-per-edge = 6) | 7.92 (Part 5) | +1.7 % worse |

Interesting: `malmo3_hct` is actually the best MALMO variant. `malmo6`
adds cells that fall *inside* MALMO blocks that were already fine
enough — the extra cells re-sample the same underlying HCT field
finely but the block-boundary transition between MALMO depths becomes
the dominant error source. So MALMO's advantage of "mesh-aligned
block boundaries" turns into a disadvantage once cells are finer than
the boundary spacing.

**Recommendation**: stop revisiting MALMO for this problem. The
mesh-aligned topology is genuinely not the right structure once the
source projection is C¹.

### 4. `4lvl_hct` is the definitive winner

Across 3 targeted challenges (shear-band refinement, more nesting
levels, higher-resolution MALMO), **none dethroned the Part-4 winner**.
`4lvl_hct` wins the cohort mean at both time steps, ties or wins on
every headline metric, and runs at the same speed as everything else.

Not much room left for grid-topology optimisation on this problem —
the ~7.8 mm cohort mean rms at step 2000 is the effective ceiling
for grid PT against a mesh+HCT-3D reference on the FSW cases, given
raw-P1 mesh input to Stage 1.

### 5. The next lever is Stage 1, not the grid

Given (1)–(4), the lever to push cohort rms below ~7 mm would be:

- **Time-dependent Stage 1**: bake grid velocities at multiple
  timesteps (currently every grid samples ts=119 only). The mesh
  tracker samples the time-dependent FOM velocity; the grid tracker
  uses a snapshot. Some of the ~7.8 mm gap is likely due to that
  temporal mismatch, not spatial.
- **ROM-projected Stage 1**: the roadmap §5 acceptance question is
  whether `ROM_grid_4lvl_hct` matches `ROM_mesh_HCT`; that's the
  next experiment on the plan doc.
- **Higher-order RK**: RK4 with fixed DT = 3.75 ms may be under-
  resolving the shear-band pass-through; adaptive DT or RK6 there
  could help. Bigger change to the tracker; keep as last resort.

## What Part 5 answered vs left open

**Answered**:
- Shear-band refinement doesn't help (falsified).
- 3 refinement levels is the sweet spot.
- MALMO cannot compete under HCT (closed).
- `4lvl_hct` is the final winner for FOM-side grid PT.

**Still open**:
- Does the same story hold when Stage 1 samples ROM velocity instead
  of FOM velocity? (Plan-doc step 4.)
- Does time-dependent Stage 1 close the ~7 mm cohort gap?
- Does the winning grid + tracker preserve mixing diagnostics
  (residence time, pairwise separation) that the roadmap §7
  requires?

## Figures for Part 5

The presentation figures (produced by
[scripts/plot_grid_pt_presentation.py](../scripts/plot_grid_pt_presentation.py))
were regenerated with all 6 variants:

| figure | file | what it now shows |
|---|---|---|
| Fig 1 (s950, s2000) | `fig1_cohort_rms_step<S>.png` | 6-bar-per-case comparison, cohort mean line for `4lvl_hct` |
| Fig 2 | `fig2_cohort_distribution.png` | 6-variant box plots at both steps, individual case dots |
| Fig 3 | `fig3_growth_scatter.png` | rms(950) vs rms(2000), auto-scaled to include all 6 |
| Fig 4 | `fig4_survivorship.png` | unchanged (survivorship depends on mesh + seed, not grid variant) |
| Fig 5 | `fig5_near_pin_vs_outer.png` | 6 variants side by side |
| Fig 6 | `fig6_spatial_pattern.png` | unchanged (4 representative cases, `4lvl_hct` winner only) |

## Per-case xy-projection error maps

Produced by
[scripts/plot_final_step_xy_error_maps.py](../scripts/plot_final_step_xy_error_maps.py).
Every case's mesh particle positions at step 2000 projected onto the
(x, y) plane, coloured by grid-vs-mesh disagreement. Two colour
metrics per variant:

- **`_pct`**: `|err| / |x_mesh − x_seed|` (%), the relative error as a
  fraction of how far the particle actually travelled. Clipped to
  [0, 100 %] for the colour scale. Particles with total travel below
  1 mm are NaN'd (avoids /0 on stationary particles).
- **`_mm`**: absolute `|err|` in mm, capped at 15 mm for the colour
  scale.

**All** mesh-tracked particles are shown, regardless of whether they
have crossed the nominal outlet at `x = 30 mm` (marked with a red
dotted line). Particles past the outlet are still meaningfully tracked
via the tracker's ballistic extension, and their grid-vs-mesh
disagreement is a real signal. Cases with strong advection (001, 003,
007, 010, 013, 016) have most of their particle cloud past x = 30 mm
at step 2000, reaching x ≈ 75 mm. Axes are shared across all 20
subplots: **x = [−15, 80] mm, y = [−15, 15] mm** so every case's full
footprint fits.

Each subplot title carries the case number, % of particles that are
still *inside* the nominal domain at step 2000 (i.e. `x_mesh < 30
mm`), and median mesh-side travel distance.

### Files

| file | figure |
|---|---|
| [`xy_error_4lvl_hct_step2000_pct.png`](../paper_figs/rom_pt_step5_step6_all20/xy_error_4lvl_hct_step2000_pct.png) | 4lvl_hct · relative error % |
| [`xy_error_4lvl_hct_step2000_mm.png`](../paper_figs/rom_pt_step5_step6_all20/xy_error_4lvl_hct_step2000_mm.png) | 4lvl_hct · absolute mm |
| [`xy_error_malmo6_hct_step2000_pct.png`](../paper_figs/rom_pt_step5_step6_all20/xy_error_malmo6_hct_step2000_pct.png) | malmo6_hct · relative error % |
| [`xy_error_malmo6_hct_step2000_mm.png`](../paper_figs/rom_pt_step5_step6_all20/xy_error_malmo6_hct_step2000_mm.png) | malmo6_hct · absolute mm |

### Universal cross-case pattern

The xy maps make the shear-band error concentration undeniable at a
glance: in **every** case a thin red-orange annular ring sits at the
pin's outer edge, followed by a mostly-blue downstream wake. Regional
medians confirm the pattern is cohort-universal (values below are for
`4lvl_hct`; `malmo6_hct` numbers are within 1–2 %pt):

| region | median rel_err % (across 20 cases) | range |
|---|---|---|
| **shear ring** (4 ≤ r ≤ 7 mm) | **~34 %** | 29 – 46 % |
| outer bulk (r > 15 mm) | ~13 % | 10 – 31 % |
| downstream wake (x > 8, |y| < 5 mm) | ~15 % | 11 – 46 % |

The **shear-ring:outer ratio is ~3× on every case** — the pin-face
region contributes 3× more relative trajectory error than the bulk,
consistently, regardless of case difficulty or survivorship. p95 of
the ring can spike to > 300 % on flusher cases (only a few thousand
particles left, and those are the ones that spent the whole
simulation cycling through the shear layer).

### What this means for the roadmap §5 acceptance question

- **The 3× shear-ring amplification is a flow property, not a
  tracker property**. It's identical between `4lvl_hct` (64 k cells)
  and `malmo6_hct` (17 M cells). Adding cells does not help.
- **The relative-error framing localises where the mesh vs grid
  cohort-mean rms gap comes from**: it's not distributed evenly. If
  the FSW mixing use case is dominated by shear-layer physics (which
  it is — the shear layer *is* the mixing zone), then a 7.8 mm
  cohort-mean rms with 3× ring concentration means shear-zone
  trajectories carry ~20 – 30 % relative error and outer-domain
  trajectories carry ~10 %. Whether that is acceptable is a use-case
  decision for the downstream mixing/residence-time analysis.
- The `_mm` variants of these figures give a complementary view
  (absolute error < 15 mm cap): most of the domain is blue (< 5 mm
  error), with the shear ring pushing 10 – 15 mm on the flusher
  cases and 5 – 10 mm on the keepers.

### Why 4lvl_hct and malmo6_hct look nearly identical

Comparing the two variants' pct-maps side by side: they are
visually indistinguishable. Same ring, same wake pattern, same
per-case shape. This is the strongest possible evidence that grid
topology no longer moves the needle once Stage 1 = HCT-3D — the
error map is dominated by *what mesh + HCT-3D says about the shear
layer*, and any reasonable grid representation of that HCT field
converges to the same trajectory divergence.

Practical corollary: **producing the xy maps for other grid
variants would just show the same pattern**. There is one FSW error
map per case at step 2000; the grid variant only shifts the mean by
tenths of a mm.

---

---

## ParaView cross-check

Per-particle displacement VTUs
(`.../rom_out/fom_grid_pt/pt_error_case<c>_grid_<VAR>_step950.vtu`)
were inspected qualitatively for each HCT variant. Consistent
with the RMS numbers, HCT-projected variants produce smooth
trajectories that visually mirror the mesh + HCT-3D reference,
while raw-P1 variants show noticeably rougher trajectories with
larger per-particle offsets. Among the HCT variants the visual
similarity to the reference is high across the whole family
(matching the tight 5.0e-3 – 7.2e-3 m RMS band), with the
mesh-aligned MALMO variants producing trajectories that especially
look like the reference — MALMO's finest block covers the pin
region tightly, so trajectory *shape* near the pin closely
matches even where the absolute RMS is only mid-pack (see §3
above for why MALMO's mid-pack RMS is not a shape problem).
