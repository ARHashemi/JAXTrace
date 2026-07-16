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

**IN PROGRESS** — the errata block is in place at the top of
[`rom_reconstruction_findings.md`](rom_reconstruction_findings.md); the
buggy-loader writeup is preserved below it verbatim for provenance.  A
follow-up rewrite of the "clean" writeup will replace the original
section once we finish the FOM-vs-ROM PT comparison (Section 2 below),
so the doc ends up as a single coherent story: (i) reconstruction
convention, (ii) 20-case reconstruction residual under the fixed
loader, (iii) tracking-side impact of that residual.

Anchors for the rewrite:

- The 20-case reconstruction table now averages 4.04 % rel_rms
  (mean-only 30.9 %) — full table already in the errata.
- The "spatial breakdown" section is invalid under the fixed loader
  and should be recomputed with the corrected reconstruction on the
  worst case (`case 000`, rel_rms 6.40 %).
- The Section-7 discussion of colleague's spec vs our code stays: the
  colleague description is correct and matches the FEMUSS
  `SLEPcExternalFilter + SnapshotsMean` path exactly.

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
PT rel displacement (this section) across all 20 cases.  This is the
answer to "does a 4 % reconstruction error translate to a 4 % PT
error, or worse, or better?".

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

## 4. "Why do ROM trajectories look neater?" investigation

**TODO** — first-pass observation from the pre-fix runs: ROM
trajectories on cylindrical looked visually smoother/neater than the
FOM ones.  Two candidate explanations:

- **The ROM basis is a smooth low-pass**: the 3 POD modes capture
  large-scale flow structure but miss high-frequency features present
  in the FOM velocity.  Tracking on the smoothed field then produces
  smoother trajectories — which is a legitimate feature of ROMs, not
  a bug.  Verifiable by inspecting the residual field
  `v_ROM − v_FOM` (already exportable via the recon spatial VTU) and
  showing it is dominated by fine-scale structure.
- **Coincidental**: the "neater" impression was tied to the buggy
  loader's under-shot reconstruction (dominant coefficients cancelled
  toward the mean).  Verifiable by re-running the visual comparison
  now that the loader is fixed.

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

Acceptance criterion (proposed): the uniform-grid path is
"good enough" if the RMS FOM-vs-uniform-grid PT displacement is
within, say, 1.5× the FOM-vs-mesh-ROM displacement.  If it is, ship
it as the fast path.  If it isn't, go to Section 6.

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

## 7. Related to-fix items pulled out of the pending work

- The static-velocity assumption (single reconstructed field per case)
  is a design choice for the current experiment.  When we move to a
  time-varying ROM (per-timestep coefficients), the reconstruction
  driver needs a `--n-times` flag and the tracking path needs a
  `VEL_START/VEL_END` range that iterates the ROM outputs.  Out of
  scope for this roadmap.
- `docs/rom_reconstruction_findings.md` "Original writeup" section
  should be rewritten (not just prefixed with an errata block) once
  the Section 2 numbers are in.  Then delete the "original writeup"
  from the same document so it becomes the single source of truth.
- We currently do the FOM tracking with HCT-3D on for the ROM
  comparison.  A separate FOM-with-raw-P1 baseline is part of the
  ablation in Section 3; do not confuse the two.

## Timeline

The dependency chain is:

```
       [0. DONE: fix loader, deploy tooling]
                        │
                        ▼
       [2. FOM+ROM PT sweep, 20 cases]  ────────────► [1. rewrite findings doc]
                        │                                       │
              ┌─────────┴──────────┐                             │
              ▼                    ▼                             ▼
       [3. HCT-3D ablation]  [4. smoothing check]        (single-source doc)
              │                    │
              └─────────┬──────────┘
                        ▼
       [5. uniform-grid experiment]
                        │
                        ▼
       [6. if not good enough → block-wise adaptive grid]
                        │
                        ▼
                 (fast + accurate ROM PT)
```

Section 2 gates everything downstream.  Sections 3 and 4 can run in
parallel with each other once 2 is done.  Section 5 depends on both.
Section 6 is contingent on 5 not meeting the acceptance criterion.
