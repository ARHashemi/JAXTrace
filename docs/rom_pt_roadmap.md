# ROM particle tracking — roadmap

**Goal**: fast and accurate particle tracking (PT) driven by
FSW-ROM-reconstructed velocity fields, benchmarked against the
full-order (FOM) tracking on the same cohort.  This document is the
forward-looking companion to
[`rom_reconstruction_findings.md`](rom_reconstruction_findings.md),
which records the loader bug that inflated our early residual numbers
and the fix that brought them back to a usable range.

Current status is captured per section as either **DONE**, **IN
PROGRESS**, or **TODO**.

The roadmap was critically reviewed against the POD-ROM /
Lagrangian-coupling literature in
[`rom_pt_roadmap_Review.md`](rom_pt_roadmap_Review.md).  Enhancements
from that review are folded into §§ 2, 3, 4, 5 (spatially-resolved
acceptance criteria, Lagrangian consistency-error framing, FSW-
specific concern about POD low-pass filtering damping mixing
structures) and gathered into a shared conventions section (§ 7).
The literature pointers cited in the review are Xiong et al.
*Ind. Eng. Chem. Res.* 2023 (POD-ROM Eulerian-vs-Lagrangian
predictability asymmetry) and Vennell et al.'s OceanTracker work
(regular-grid throughput evidence).

## 0. Where we are today

**DONE**

- **ROM reconstruction bug fixed**.
  [`rom_reconstruction_findings.md`](rom_reconstruction_findings.md)
  errata block: the loader had `Basis_CompMode <comp> <mode>` axes
  swapped, mixing components and modes.  Fix landed in
  `jaxtrace/rom/velocity_recon.py` (commit `ff53727`).  20-case mean
  `centered` rel_rms dropped from 38.2 % → **4.04 %** with std **1.08 %**;
  best 03 at 2.62 %, worst 00 at 6.40 %.  The FEMUSS
  reconstruction convention (`v = mean + Σ_k c_k · φ_k`, no sigma
  scaling) is exactly what our `centered` formula does.

- **Reconstruction tooling deployed** (all pushed on
  `feature/analytic-velocity`):

  - Python driver `scripts/reconstruct_rom_case.py` — writes a
    single-piece PVTU that mirrors the FOM PVTU shape (same mesh,
    same `Displacement` field), suitable to feed straight to
    `run_tracking.py --velocity-source mesh`.
  - Shell wrapper `/scratch/shared/ROM/FOM/reconstruct_rom_velocities.sh`
    — bulk-reconstructs a list of cases into
    `/scratch/shared/ROM/ROM_recon_<formula>/`.

- **FOM tracking tooling regenerated for the whole cohort**:

  - `scripts/generate_jaxtrace_scripts.sh` now supports `--uniform-steps`
    (stamp the same `N_STEPS` on every case, bypass velocity-scaling).
  - Shared template `cylindrical_001.gid/run_jaxtrace.sh` extended with
    `GRADIENT_RECOVERY` + `RECOVERY_METHOD` knobs, default
    `RECOVERY_METHOD=hct_cubic` (HCT-3D cubic reconstruction on the
    Alfeld sub-tet split).
  - All 20 per-case `run_jaxtrace.sh` regenerated with uniform
    `DT=3.75e-3`, `N_STEPS=2000`, HCT-3D on, and ALL postprocess knobs
    OFF (ENABLE_UNION, N_GROUPS, EXPORT_ESCAPED_FLAG,
    TRACK_MAX_TEMPERATURE, EXPORT_TEMPERATURE all 0).  Per-case
    `PIN_RPM` and `INLET_VELOCITY` preserved from each case's
    `.som.dat`/`.som.fix`.

- **ROM PT tooling deployed** (fully symmetric with the FOM path):

  - Per-case helper `scripts/generate_run_jaxtrace_recon.sh` derives
    each case's ROM tracker from its own FOM tracker (only velocity
    source paths are swapped).
  - Bulk generator `scripts/generate_jaxtrace_recon_scripts.sh` walks
    the cohort and calls the per-case helper for every case that
    passes the preflight (ROM PVTU present).
  - Bundle launcher `scripts/launch_jaxtrace_recon.sh` runs / submits
    every `run_jaxtrace_recon.sh` (sequential on workstation, SLURM
    parallel on LUMI).
  - `OUTPUT_CASE_SUBFOLDER="post_pt/rom_<formula>"` so ROM outputs
    nest under each case's `post_pt/` folder, sitting as siblings to
    the FOM tracking output.

- **VTKHDF-based FOM-vs-ROM comparison tool**
  `scripts/compare_rom_vs_fom_tracking.py`.  Accepts `--step N`,
  `--list-steps`, `--suggest-alive-step` (auto-fallback when the
  requested step has an empty both-alive subset).  First cylA-cohort
  numbers (case 004, before the loader fix) were dominated by
  ballistic-tail linear divergence — will be re-collected with the
  fixed reconstruction.

**Currently blocked on the workstation user runs**: the FOM tracking
sweep across all 20 cases has to complete before we can produce the
FOM-vs-ROM comparison table that anchors every downstream decision.

## 1. Update the reconstruction findings doc

**DONE** — [`rom_reconstruction_findings.md`](rom_reconstruction_findings.md)
was rewritten as a single-source narrative in commit `c426cf6`.  The
old errata-plus-preserved-original layout is gone; the loader-bug
provenance now sits in §6 of the same doc, so a fresh reader gets a
clean sequential story: convention → 20-case table (4.04 % ± 1.08 %) →
time sweep (residual smallest at ts ≈ 16 – 21, drifts up by 1 – 2 pp
by ts = 119) → spatial breakdown (2 – 8 % inside the pin; the
40 – 80 % "outer-domain" numbers are a low-signal artefact of both
FOM and residual being near zero there) → historical loader bug →
recommendations.

Also materialised three permanent analysis scripts under
`tests/rom/`:

- `rom_20case_sweep.py`
- `rom_time_sweep.py`
- `rom_spatial_residual.py`

Each reads only `/scratch/shared/ROM/FOM/`, needs no JAX / GPU, and
regenerates the exact numbers cited in the findings doc's
Reproducibility section.

## 2. FOM-vs-ROM particle-tracking comparison

**TODO — the flagship experiment.**

Static-velocity setup (one reconstructed field per case, replayed for
every RK4 step; DT irrelevant to accuracy, `N_STEPS` uniform across
cases):

- **DT = 3.75e-3, N_STEPS = 2000** — already stamped on every case's
  `run_jaxtrace.sh` and `run_jaxtrace_recon.sh`.
- Same seeding, same boundary walls, same level-set, same pin velocity
  per case — everything except the velocity field is held constant.

Deliverables (per case, aggregated across the cohort):

- Final positions FOM vs ROM (at last step and at
  the `--suggest-alive-step` fallback).
- Per-particle displacement `‖rom_pos − fom_pos‖` — mean, median, rms,
  p95, p99, max, per-component rms, rms/bbox_diagonal.
- Escape-flag agreement (both-alive / both-escaped /
  only-FOM-escaped / only-ROM-escaped).
- Spatial map of displacement (VTU written by
  `compare_rom_vs_fom_tracking.py --out-vtu`).

Cohort-level cross-plot: reconstruction rel_rms (from Section 1) vs
PT rel displacement (this section) across all 20 cases.

**Prior expectation** — worth stating explicitly rather than treating
this as a purely empirical unknown.  The POD-ROM feasibility
literature (Xiong et al., *IECR* 2023, and related work; see the
review in `rom_pt_roadmap_Review.md`) documents an **asymmetry**
between Eulerian and Lagrangian ROM predictability: even when the
Eulerian velocity field reconstructs with low RMS error, the
corresponding Lagrangian trajectory error can amplify substantially,
because Lagrangian quantities are path integrals of the field and
small local errors compound along the trajectory.  The amplification
is strongest near **chaotic / separatrix regions** of the flow
— which in FSW means the shear layer that separates pin-driven
rotation from bulk drift.  So the expected outcome of this experiment
is NOT "4 % reconstruction → 4 % PT", but a "consistency error"
factor > 1 that concentrates spatially near the pin shear layer.

Concretely this means the cohort-level cross-plot alone is not
enough — we also need a **spatial breakdown of the PT displacement**
mirroring the `r`-bin breakdown from
`rom_reconstruction_findings.md` § 5:

- near-pin bin (`r ≤ 0.010 m`) — where the reconstruction already
  carries most of its residual and where FSW's mixing physics lives,
- outer-domain bin (`r > 0.010 m`) — mostly a low-signal region.

Report **per-bin PT rel displacement** at the reporting step, then
form a per-bin cross-plot against the reconstruction's per-bin
residual.  If the near-pin PT displacement is markedly worse than the
near-pin reconstruction residual (say > 2 × amplification), we've
empirically reproduced the consistency-error result and downstream
plans (Sections 3 – 5) need spatially-resolved acceptance criteria.
This is already implementable — the compare tool's `--out-vtu` gives
per-particle displacement fields and the seeding is deterministic, so
binning particles by initial radial position is a one-liner post-hoc.

Command sequence on the workstation:

```
cd /scratch/shared/ROM/FOM
bash /flash/shared/jax/JAXTrace/scripts/launch_jaxtrace.sh                # FOM, all 20
CASES="$(seq 0 19)" bash reconstruct_rom_velocities.sh                    # ROM PVTUs, all 20
bash /flash/shared/jax/JAXTrace/scripts/generate_jaxtrace_recon_scripts.sh --force
bash /flash/shared/jax/JAXTrace/scripts/launch_jaxtrace_recon.sh          # ROM, all 20
for c in 000 001 ... 019; do
    python3 /flash/shared/jax/JAXTrace/scripts/compare_rom_vs_fom_tracking.py \
        --fom-vtkhdf /scratch/shared/ROM/FOM/cylindrical_$c.gid/post_pt/<run>/particles.vtkhdf \
        --rom-vtkhdf /scratch/shared/ROM/FOM/cylindrical_$c.gid/post_pt/rom_centered/<run>/particles.vtkhdf \
        --step 500
done
```

## 3. HCT-3D ablation (with/without recovery, FOM and ROM)

**TODO** — a small 2×2 ablation embedded in the Section 2 sweep to
answer "does higher-order velocity reconstruction help FOM tracking
and/or ROM tracking?".

Four combinations per case (or a subset — start with cases 000
[worst-reconstructing] and 003 [best-reconstructing]):

| velocity source | recovery | expected behaviour |
|:---|:---|:---|
| FOM PVTU     | HCT-3D on  | reference (current default) |
| FOM PVTU     | HCT-3D off | raw P1 interpolation baseline |
| ROM recon PVTU | HCT-3D on  | current default for ROM path |
| ROM recon PVTU | HCT-3D off | raw P1 on ROM field |

Toggle by setting `GRADIENT_RECOVERY=0` (raw P1) or
`GRADIENT_RECOVERY=1` + `RECOVERY_METHOD=hct_cubic` (default) in each
runner.

Two questions to answer:

- **On FOM**: how much does HCT-3D change tracked trajectories?
  Baseline for how big an "acceptable" perturbation looks.
- **On ROM**: does HCT-3D compensate for reconstruction residual, or
  does it *smooth over* it and hide the ROM's actual error signature?

**Reporting** — the "compensate vs mask" question is a *spatial*
question, not an aggregate one.  A masking recovery can produce the
same cohort-level RMS displacement as a genuinely-compensating one
while placing the error in completely different regions
(compensating: uniformly reduced; masking: near-pin still wrong,
far-field over-corrected).  We already have the spatial tooling
required — `compare_rom_vs_fom_tracking.py --out-vtu` writes a
per-particle displacement VTU, and the `r`-binning from
`rom_reconstruction_findings.md` § 5 applies unchanged.

Concretely, for each of the four cells of the 2 × 2 table:

1. Global RMS displacement (as originally proposed).
2. Per-`r`-bin RMS displacement (near-pin + outer-domain) at the
   reporting step.
3. The spatial VTU, opened in ParaView, coloured by displacement
   magnitude.  Visual coherence of the residual pattern is the
   distinguishing signal — a compensating recovery gives a diffuse
   residual, a masking one gives a residual concentrated at the pin
   shear layer.

The cost of adding (2) and (3) is zero — the tool already produces
them. What changes is that acceptance in this ablation must include
the near-pin per-bin RMS being consistent with the compensating
hypothesis, not just the aggregate number.

## 4. "Why do ROM trajectories look neater?" investigation

**TODO** — first-pass observation from the pre-fix runs: ROM
trajectories on cylindrical looked visually smoother/neater than the
FOM ones.  Two candidate explanations:

- **The ROM basis is a smooth low-pass**: the 3 POD modes capture
  large-scale flow structure but miss high-frequency features present
  in the FOM velocity.  Tracking on the smoothed field then produces
  smoother trajectories.  Verifiable by inspecting the residual field
  `v_ROM − v_FOM` (already exportable via the recon spatial VTU) and
  showing it is dominated by fine-scale structure.
- **Coincidental**: the "neater" impression was tied to the buggy
  loader's under-shot reconstruction (dominant coefficients cancelled
  toward the mean).  Verifiable by re-running the visual comparison
  now that the loader is fixed.

**Framing caveat** — the previous version of this section called the
smoothing outcome "a legitimate feature of ROMs, not a bug".  That's
neutral phrasing that does not survive the FSW-specific context.  A
POD basis that damps high-wavenumber content damps *exactly* the
small-scale shear layers and vortical structures that drive
Lagrangian stretching / folding, i.e. the mechanism of stirring and
material mixing.  For an application whose scientific target is
mixing patterns (which FSW is), an L² -acceptable smoothed velocity
that has lost small-scale structure is not a benign trade-off — it
can invalidate the ROM for its intended downstream use.  This is a
general observation about POD-Lagrangian coupling (Xiong et al.
2023; the coherent-structures literature); see the review in
`rom_pt_roadmap_Review.md`.

Diagnostic to run once the FOM-vs-ROM PT sweep is done:

1. Load `v_FOM` and `v_ROM` for one representative case at ts = 119.
2. Compute residual = `v_FOM − v_ROM`; write as a VTU (already
   supported by the compare tool).
3. Compute the power spectrum of the residual along a couple of
   lines through the pin, and compare to the FOM velocity's own
   spectrum.  If the residual has appreciably more high-frequency
   content than the FOM, the "smoothing" hypothesis is confirmed.
4. Cross-check with FOM+HCT-3D-off: if the raw-P1 FOM trajectories
   also look neater than mesh-P1 FOM, then HCT-3D itself is
   introducing some of the aesthetic difference.
5. **Mixing-relevant Lagrangian diagnostic** — a spectral diagnostic
   on the velocity field alone does not tell us how *tracking* is
   affected.  Add one lightweight Lagrangian diagnostic that uses
   the existing PT outputs and does not require a new solve:

   - **Residence-time distribution** in an annular probe centred on
     the pin.  Bin particles by initial radial position, count how
     many are still inside the annulus at each timestep, plot the
     resulting curve for FOM PT vs ROM PT.  Systematic ROM-side
     over-retention (particles taking too long to leave the pin
     region) is the classic under-mixing signature of a low-pass
     velocity.
   - Optionally, **pairwise separation** of nearby seeded particles
     as a function of time.  Two particles seeded a few tets apart
     under a chaotic advective flow separate exponentially; the
     rate is a proxy for the top FTLE eigenvalue.  Comparing this
     rate FOM vs ROM tells us directly whether the ROM has damped
     the chaotic advection that drives FSW mixing.  Both can be
     computed as post-hoc numpy scripts from the exported
     particles.vtkhdf, no new solver code needed.

Result of this section decides whether the ROM is fit for FSW's
scientific target, independently of whether it hits an aggregate PT
displacement threshold.

## 5. Uniform-grid projection experiment

**TODO** — the main throughput lever we can pull.  Motivation: with
the mesh path (either FOM or ROM PVTU) every RK4 sub-step does a
point-in-tet search, which is the dominant cost for cylA (Section-6
benchmark hits give L0 ~99 %, but the L0 lookup itself is where the
time goes).  A uniform Cartesian grid removes the search entirely —
trilinear interpolation on `(i, j, k)` indices computed from `pos`
and a fixed origin+spacing is a handful of FLOPs per query.

Steps:

1. **Design the sampler**.  Use HCT-3D on the ROM-reconstructed field
   to sample it at every node of a uniform Cartesian grid inside the
   mesh bounding box.  HCT-3D gives a per-tet cubic representation
   that is bit-exact for quadratic fields at the parent centroid and
   spoke-edge midpoints, so its samples are the best per-tet
   approximation we can get without going back to the underlying POD
   modes.
2. **Pick a first grid size**.  Start at
   `(nx, ny, nz) = (100, 100, 40)` (400,000 cells) for cylindrical.
   The mesh diagonal is ~0.09 m so this gives ~2 mm cells — a factor
   ~4× coarser than the finest tet-edge in the pin region.
3. **New velocity provider**.  Write a `UniformGridVelocityProvider`
   analogous to the existing `MeshVelocityProvider`
   (`jaxtrace/gpu/tracking/velocity_provider.py`) that:
   - holds `(origin, spacing, n_cells, values)` on the GPU,
   - implements `sample(pos, ...)` as `jnp` trilinear interpolation
     (no search, no L0/L1/L2, no PIT test),
   - satisfies the same `is_time_dependent = False` contract.
4. **Loss experiment**.  For a representative case run PT three
   ways:
   - **FOM mesh + HCT-3D** — reference (already produced in Section 2).
   - **ROM mesh + HCT-3D** — current recon path (already in Section 2).
   - **ROM uniform-grid + trilinear** — new path.

   Compare final positions with `compare_rom_vs_fom_tracking.py`.
   Report per-particle displacement and (rom_uniform − rom_mesh) vs
   (rom_mesh − fom_mesh) — how much *extra* error does the
   uniform-grid projection introduce relative to the reconstruction
   error we already accept?
5. **Throughput**.  Time each path with the Section-6 style
   `benchmark_l2_accuracy`-analogue timing loop.  Report queries/s and
   PIT-tests-equivalent per particle per RK4 substep.

**Prior expectation and diagnostic design** — Vennell et al.'s
OceanTracker work (see `rom_pt_roadmap_Review.md`) already
demonstrates 200 – 500× speedups from replacing unstructured-grid
element search with regular-grid indexing, so the throughput half of
this experiment is well-precedented and the interesting question is
purely about accuracy.  The accuracy risk profile is asymmetric:
grid-projection error is spatially **concentrated near the pin**
where the tet mesh is finest relative to a 2 mm Cartesian cell, and
that same region is where the reconstruction residual is already
largest and where FSW's mixing physics lives.  ROM error, by
contrast, is smooth and globally distributed.  A single scalar RMS
criterion averaged over the whole domain can pass while
catastrophically failing near the pin — the exact regime where the
Section 6 fallback exists to help.

Acceptance is therefore **spatially resolved**, using the same `r`
bins as `rom_reconstruction_findings.md` § 5:

- **Near-pin (`r ≤ 0.010 m`)** — RMS FOM-vs-uniform-grid PT
  displacement must be within 1.5× the FOM-vs-mesh-ROM displacement
  in the same bin.  Failure here is a red flag for FSW use even if
  the aggregate number passes.
- **Outer-domain (`r > 0.010 m`)** — the same 1.5× ratio applied to
  the outer-bin RMS.  Failure here is less critical (the region
  matters less physically) but should still be tracked.
- **Aggregate** — the 1.5× ratio on the whole cohort.  This is the
  originally proposed criterion; kept as a sanity number, not the
  primary decision gate.

The two per-bin criteria are the primary decision gates.  If both
pass, ship the uniform-grid path.  If only the aggregate passes but
the near-pin bin fails by more than a small margin, go to Section 6
even though the "1.5× overall" reads clean — the aggregate is
misleading in this regime.  If the near-pin bin fails badly (say >
3×), we also learn how much refinement is needed there, which is
useful input for sizing the block-wise fallback.

Also mirror the mixing diagnostics from Section 4 on the uniform-grid
path: residence-time distribution and pairwise separation, compared
FOM-mesh vs ROM-mesh vs ROM-uniform-grid.  If the uniform-grid path
matches ROM-mesh on displacement but breaks the residence-time
comparison, we've quantified exactly the risk the coarse-grid
projection was supposed to have — a signal that Section 6 is needed
even more strongly than the displacement criterion suggests.

## 6. Block-wise refined grid (fallback)

**TODO — only if Section 5 doesn't pass the acceptance criterion.**

A single Cartesian grid coarse enough to fit in cache is likely too
coarse in the tool region.  Answer: a fixed *predetermined* AMR
layout that keeps the search-free property.

Two ways to seed the refinement layout:

- **User-defined regions**.  A config file listing axis-aligned
  boxes with a per-box refinement level (e.g. "refine by 4× inside
  `x ∈ [-0.01, 0.01]`, `y ∈ [-0.01, 0.01]`, `z ∈ [-0.005, 0]`").
  Simple, transparent, and identical across all 20 cases.
- **Auto-captured from the original mesh**.  Compute per-cell edge
  length in the original tet mesh; project onto a coarse Cartesian
  grid; refine any coarse cell whose finest overlapping tet is finer
  than a user threshold.  Every case gets the same layout because we
  derive it from a shared mesh.

Whichever seeding we pick, the tracking-time API must stay
search-free: given `pos`, we resolve the enclosing coarse block by
`(i, j, k) = floor((pos - origin) / coarse_spacing)`, then descend a
constant number of levels via bit-shifts to hit the leaf.  No sorting,
no binary search, no point-in-tet.  Interpolation inside each leaf
is still trilinear.

The paper Section-6 MALMO octree already implements exactly this idea
for tetrahedral meshes; we can lift the same block layout and just
replace the leaf-side "list of tets" with a leaf-side "8 corner
values" for the uniform-grid case.

## 7. Diagnostic conventions shared across sections

Sections 2, 3, 4, and 5 all rely on the same three spatial /
mixing-relevant diagnostics.  Gathering the conventions in one place
so they stay consistent as the experiments run:

**Spatial binning** — use the same radial bins as the reconstruction
spatial breakdown
(`rom_reconstruction_findings.md` § 5,
`tests/rom/rom_spatial_residual.py`):

- **near-pin bin**: nodes / particles with `r ≤ 0.010 m` (initial
  position for particles).  This is where the tet mesh is finest,
  the FOM velocity is largest, the reconstruction residual is
  largest, and FSW's mixing physics happens.
- **outer-domain bin**: `r > 0.010 m`.  Low-signal region for both
  FOM and residual.

For any acceptance criterion or comparison table involving
displacement, **report per-bin RMS separately in addition to the
aggregate**.  The aggregate alone can be misleading when the two
bins have very different signal magnitudes (as they do in
cylindrical).

**Spatial VTU dump** — `compare_rom_vs_fom_tracking.py --out-vtu`
produces a per-particle displacement field, coloured for ParaView.
Any acceptance in this roadmap should be accompanied by a visual
inspection of this VTU — the "compensating vs masking" question in
§ 3 and the "coarse-grid failure mode" question in § 5 are both
spatial questions that scalar summaries can't answer.

**Mixing-relevant Lagrangian diagnostics** — computed post-hoc from
the exported `particles.vtkhdf`, no new solver code required.  Two
recipes to run at least on the Section 2 sweep and re-run in § 4
(smoothing investigation) and § 5 (uniform-grid comparison):

1. **Residence time in an annular probe** — pick an annulus centred
   on the pin (e.g. `0.005 ≤ r ≤ 0.010`, full z range), bin
   particles by initial radial position, count how many are still
   inside at each timestep.  Compare the FOM and ROM curves.
   Systematic ROM-side over-retention (slower escape from the
   annulus) is the signature of an under-mixing velocity — the
   direct scientific concern the smoothing hypothesis raises.
2. **Pairwise separation** — pick a few hundred particle pairs
   seeded within a few tets of each other at t = 0.  Track
   `‖p_i(t) − p_j(t)‖` and plot the geometric-mean separation vs
   time in log scale.  A straight line with positive slope is
   exponential (chaotic) separation; the slope is a proxy for the
   top FTLE eigenvalue.  Compare FOM vs ROM slopes.  A markedly
   flatter ROM slope means the ROM has damped the mechanism that
   drives mixing, even if the trajectory RMS looks acceptable.

Both diagnostics are post-hoc numpy on the vtkhdf archives — plan on
~50 lines of Python for the two of them, no solver changes.

## 8. Related to-fix items pulled out of the pending work

- The static-velocity assumption (single reconstructed field per case)
  is a design choice for the current experiment.  When we move to a
  time-varying ROM (per-timestep coefficients), the reconstruction
  driver needs a `--n-times` flag and the tracking path needs a
  `VEL_START/VEL_END` range that iterates the ROM outputs.  Out of
  scope for this roadmap.
- We currently do the FOM tracking with HCT-3D on for the ROM
  comparison.  A separate FOM-with-raw-P1 baseline is part of the
  ablation in Section 3; do not confuse the two.
- The Section-2 cross-plot's *prior* (Lagrangian consistency error
  compounds beyond Eulerian reconstruction error) is a citable
  finding from Xiong et al.'s POD feasibility work.  If the Section 2
  numbers reproduce this pattern (near-pin PT rel displacement
  > 2× the reconstruction rel_rms in the same bin), we should say so
  explicitly in the writeup and cite the prior; it makes the
  experimental result stronger and more publishable than treating it
  as a novel discovery.
- The review that triggered this enhancement pass
  (`rom_pt_roadmap_Review.md`) also flagged the "smoothing is
  neutral" framing as tone-deaf for FSW; § 4 now carries an explicit
  "not neutral for FSW" caveat.  Preserve that framing in any later
  writeup — the low-pass-hides-mixing failure mode is exactly what
  the discipline expects a shallow POD basis to do.

## Timeline

The dependency chain is:

```
      [0. DONE: fix loader, deploy tooling]
                       │
                       ▼
      [1. DONE: single-source findings doc (c426cf6)]
                       │
                       ▼
      [2. FOM+ROM PT sweep, 20 cases]  (per-r-bin, not just aggregate)
                       │
             ┌─────────┴──────────┐
             ▼                    ▼
      [3. HCT-3D ablation]  [4. smoothing check + mixing diagnostics]
      (spatial VTU)         (residence time, pairwise separation)
             │                    │
             └─────────┬──────────┘
                       ▼
      [5. uniform-grid experiment]  (per-r-bin acceptance)
                       │
                       ▼
      [6. if not good enough → block-wise adaptive grid]
                       │
                       ▼
              (fast + accurate ROM PT)

      [7. shared spatial + mixing diagnostics used by 2, 3, 4, 5]
```

Section 2 gates everything downstream.  Sections 3 and 4 can run in
parallel with each other once 2 is done.  Section 5 depends on both.
Section 6 is contingent on 5 not meeting the acceptance criterion.
Section 7 is a shared conventions reference used by 2 – 5, not a
sequential step.
