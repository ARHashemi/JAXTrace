# ROM particle tracking — next-steps plan (post-grid-experiment evaluation)

**Date:** 2026-07-30
**Context:** Grid experiment (24 variants × 4 cases) done. User
proposed a 5-step continuation plan that closes the roadmap's
main goal *"fast + accurate ROM PT for the FSW cohort"*. This
document evaluates that plan step by step, records what is
already on disk, flags the traps to avoid, and defines the
acceptance criterion each step will meet before we move to the
next.

**Sources**:
[roadmap](rom_pt_roadmap.md),
[grid report](rom_pt_step5_step6_grid_report.md),
[§2/§3 report](rom_pt_step2_step3_report.md),
[§4 report](rom_pt_smoothness_divfree_report.md),
[reconstruction findings](rom_reconstruction_findings.md).

## Where we are vs the roadmap's main goal

Two axes and two cohorts:

|  | 4-case cohort (000/001/003/004) | full 20-case cohort |
|---|---|---|
| **FOM tracker on mesh** | done, ✓ | 4 of 20 done |
| **ROM tracker on mesh** | done (under `ROM_recon_centered/`) | 4 of 20 done |
| **FOM tracker on grid** | done (24 variants) | 0 of 20 done |
| **ROM tracker on grid** | not done | 0 of 20 done |
| **Wall-time comparison** | logs exist, not tabulated yet | needs 16 more cases |
| **Per-r-bin acceptance** | not applied to grid results | not done |
| **Mixing diagnostics (residence-time, pairwise separation)** | done for mesh path | not done for grid path |

The last two rows are the roadmap's §5 acceptance criterion and
its §7 shared conventions — neither is applied yet to the grid
experiment.

The **big known gap** is that the whole 24-variant grid
experiment ran on the **FOM velocity field**, not on the ROM
reconstruction. What we have shown is
*"a grid can replace the mesh for the FOM tracker without hurting
accuracy"*.  What the roadmap actually asks (§5, line ~365) is
*"does replacing ROM-mesh with ROM-uniform-grid introduce extra
error on top of the ROM residual we already accept?"* — that is
step 4 below.

## The 5-step plan and its evaluation

### Step 1 — Extract wall-time comparison from existing logs

**What to do**: pull `Wall time:` and throughput lines out of every
`out_grid_<VAR>/run_grid.log` (grid PT logs; 24 variants × 4
cases = 96 rows) and out of `post_pt/fom_hct_{on,off}/log.txt`
(mesh PT logs; 2 variants × 4 cases = 8 rows).

**Why it's cheap**: every log already carries the two summary
lines emitted by `run_tracking.py`. No new runs needed.

**Reference cost (measured from the mesh PT logs)**:
- Mesh + HCT-3D reference: ~1400–1500 s per case, ~500 k
  particle-steps/s (identical on/off HCT).
- Grid uniform (48 k cells, trilinear): ~18 s per case,
  ~40 M particle-steps/s.
- Speedup baseline: **~80× for the simplest grid**.  Larger
  grids (`uniform_half` = 17.5 M cells) or heavier interp
  (tricubic) will scale down; the tabulation gives all 24
  numbers.

**Deliverable**: a wall-time table appended to
[rom_pt_step5_step6_grid_report.md](rom_pt_step5_step6_grid_report.md)
Part 2. Rows = 24 variants × 4 cases, columns = wall time (s),
throughput (p·step/s), speedup vs mesh + HCT-3D.

**Acceptance**: table populated + one summary paragraph noting
speedup range and the trade-off with accuracy (from Part 2 CSV).

### Step 2 — Enrich the compare tool with per-r-bin + trapping + escape breakdown

**What to add** to
[scripts/compare_grid_vs_mesh_pt_fom.py](../scripts/compare_grid_vs_mesh_pt_fom.py)
(all computable from the existing per-run `particles.vtkhdf`, no
new PT runs):

1. **Per-r-bin RMS** (matches roadmap §7 conventions):
   - `r_init ≤ 0.010 m` (near-pin, where FSW mixing lives)
   - `r_init > 0.010 m` (outer bulk)
   - "initial radial position" = `√(x0² + y0²)` at step 0, so
     bin assignment is deterministic and shared across variants.
2. **Trapping agreement**:
   - `frac_trapped_grid` = fraction of particles inside
     `r ≤ 0.010 m` at the reporting step, using grid PT positions.
   - `frac_trapped_mesh` = same, mesh PT positions.
   - Report `Δ = frac_trapped_grid − frac_trapped_mesh` per variant
     (positive = grid retains more, negative = grid loses more).
3. **Escape / alive agreement** at reporting step:
   - `both_alive`, `both_escaped`, `only_grid_escaped`,
     `only_mesh_escaped` — 4 counts.  Escaped ≡ x ≥ 0.030 m
     (matches `--boundary-walls x_max=outlet`).
4. **Per-component RMS**: rms_x, rms_y, rms_z separately.  Reveals
   whether the error is axisymmetric or has a directional bias
   (a directional bias points at a specific systematic effect
   like tricubic overshoot in a preferred axis).
5. **Percentile ladder**: p50, p95, p99, max per variant.  A
   variant may be great on mean-RMS but have long tails that
   fail the FSW mixing use case.

**New CSV**:
`.../rom_out/fom_grid_pt/grid_vs_mesh_pt_detail_step<S>.csv`
with columns:
`case, grid_type, step, n, rms, rms_x, rms_y, rms_z, mean,
p50, p95, p99, max, rms_near, rms_outer, n_near, n_outer,
frac_trapped_grid, frac_trapped_mesh, delta_trapped,
n_both_alive, n_both_escaped, n_only_grid_escaped,
n_only_mesh_escaped`.

**New figures** (all with shared y-limits within each panel-pair
and unambiguous titles that distinguish reference from Stage-1):

- `..._per_r_bin_p1.png` / `..._per_r_bin_hct.png` — two-bar
  per case × family (near-pin, outer) so each family produces
  two bars.  Same P1/HCT panel split as the aggregate chart.
- `..._trapping_p1.png` / `..._trapping_hct.png` — bar chart
  of `|Δfrac_trapped|` per (case, variant); horizontal
  reference line at the mesh + HCT-on−vs−off baseline (from
  §2/§3 report's Fig 5).
- `..._escape_agreement.png` — stacked bar with the 4 escape
  categories, one bar per variant, one panel per case.
- `..._percomp_rms.png` — grouped bars, 3 bars per variant
  (rms_x, rms_y, rms_z), one panel per case.

**Deliverable**: extended tool + updated plot script; per-case
CSVs regenerated; the two Part-2 tables in the grid report gain
a per-r-bin split.

**Acceptance**: for each of the 4 cases, the near-pin RMS bin
is reported alongside the aggregate.  The "top-3 variants"
call in step 3 uses **near-pin RMS + trapping agreement** as
the primary decision gates (aggregate RMS is a sanity check
only, per roadmap §7).

### Step 3 — Extend top-3 winners to the full 20-case cohort

**Sequencing note**: "top-3" must be decided *after* step 2
completes.  Current aggregate-RMS top-3 is `4lvl_hct_tricubic`,
`2lvl_hct_tricubic`, `4lvl_hct` — but the near-pin bin and
trapping-agreement metrics from step 2 may reshuffle the ranking.
In particular, MALMO's mesh-aligned finest block covers the pin
region tightly, so a near-pin metric may promote `malmo_hct` or
`malmo_hct_tricubic` above where they sit on aggregate RMS.

**Prerequisites on disk**:

| item | status |
|---|---|
| PVTU (`post/cylindrical_119.pvtu`) for 002 + 005–019 | ✓ present |
| `run_jaxtrace.sh` for 002 + 005–019 | ✓ present |
| `run_grid_all.sh` for 002 + 005–019 | ✗ missing (need to auto-generate) |
| grid modules for the 3 winners × 16 cases | ✗ missing (need to build) |
| `hct_cache_ts119.npz` for 002 + 005–019 | ✗ missing (built once per case on first HCT-projection call) |
| reference mesh PT (`post_pt/fom_hct_on`) for 002 + 005–019 | ✗ missing |

**Total workstation cost estimate** (all `run_grid_all.sh`-style,
skip-guarded, 8-core parallel where possible):

- Mesh PT reference for 16 cases: 16 × ~1450 s = **~6.5 h** (serial
  GPU; single GPU can only run one tracker at a time).
- Grid module builds (3 variants × 16 cases): all 3 winners are
  small grids (`4lvl` ~64 k cells, `2lvl` ~57 k cells).  With
  per-case HCT cache reused across the 3 variants: SPR (~30 s)
  + 3 × grid eval (~30 s each) = ~2 min per case × 16 = **~30 min**.
- Grid PT (3 variants × 16 cases): 48 × ~20 s = **~15 min**.
- Compare tool re-run on the enriched CSV: minutes.

**Sub-task ordering** — three helper scripts now live at
`/scratch/shared/ROM/FOM/`:

1. `run_grid_top3_template.sh` — per-case runner template
   parameterised by `CASE_STEM` (only `4lvl_hct`,
   `4lvl_hct_tricubic`, `2lvl_hct_tricubic`). Installed as
   `run_grid_top3.sh` in each case dir by the orchestrator.
2. `build_grid_top3.sh` — parallel builder for the 3 grid
   modules per case. Runs SPR once per case (~30 s cache warmup),
   then builds the two tricubic variants in parallel via
   `xargs -P NCORES`. Skip-guarded per module.
3. `launch_top3_all20.sh` — one-shot orchestrator with 4 phases:
   Phase A (mesh PT reference for 16 new cases, serial GPU
   ~6.5 h), Phase B (grid modules via #2, ~15 min at NCORES=8),
   Phase C (grid PT for top-3 × 20 cases, ~24 min), Phase D
   (enriched compare at both step 950 + step 2000).

**Actual launch**:

```bash
bash /scratch/shared/ROM/FOM/launch_top3_all20.sh 2>&1 | tee /tmp/rollout_$(date +%F).log
```

All four phases run inside the one script; each phase's own
skip-guards make interruption + restart safe. Cases 000/001/003/004
are skipped throughout because their outputs already exist.

**Winners rationale** (see grid report Part 3):
- Composite ranking summed across `rms@950`, `rms@2000`,
  `rms_near@2000`, `|Δtrap|@2000` — tied #1 = `4lvl_hct` and
  `4lvl_hct_tricubic` (score 6 each), #3 = `2lvl_hct_tricubic`
  (score 11).
- 4lvl family wins aggregate on 3 of 4 cases and near-pin on 3
  of 4 cases at step 2000. Case 001's near-pin bin is where
  `malmo_hct_tricubic` wins — an optional 4th "reference for
  near-pin shape" candidate if we care about that story per case.
- Full-cohort output lands at
  `/scratch/shared/ROM/rom_out/fom_grid_pt_all20/`.

**Step-3 outcome (2026-07-31, done)**: see
[grid report Part 4](rom_pt_step5_step6_grid_report.md#part-4--20-case-cohort-rollout)
for the 20-case numbers. Two tooling fixes were needed mid-flight:
(a) reference-discovery fallback to `run_grid-frac_n360000_s2000`,
(b) switching all rms metrics to the both-alive subset. Headline
result: **`4lvl_hct` (trilinear Stage 2) wins on the 20-case
cohort at step 2000** (mean 7.79 mm), overturning the 4-case
tie between trilinear and tricubic. The tricubic Stage-2 overshoot
compounds along longer trajectories.

**Bonus step-3b (2026-07-31, done)**: three additional variants
tested to check whether the Part-4 spatial pattern points at an
actionable optimisation — see
[grid report Part 5](rom_pt_step5_step6_grid_report.md#part-5--testing-3-more-grid-variants-across-the-20-case-cohort).
None of `4lvl_r22_hct`, `5lvl_hct`, `malmo6_hct` beat `4lvl_hct`
(all significantly worse at p ≤ 0.03, small effect sizes). Grid-
topology optimisation on this problem is effectively saturated;
the ~7.8 mm cohort mean rms at final step is the ceiling for
grid PT under raw-P1 Stage-1 mesh input.

Presentation-quality figures (6 plots) live at
`paper_figs/rom_pt_step5_step6_all20/`, produced by
[scripts/plot_grid_pt_presentation.py](../scripts/plot_grid_pt_presentation.py).

**Deliverable**: extension of Part 2 tables in the grid report to
20 cases; per-r-bin + trapping numbers for every variant.

**Acceptance**: the winner's near-pin RMS and trapping-agreement
metrics are within X of the mesh reference for **all 20 cases**,
where X is the criterion the compare tool exposes as a pass/fail
column (roadmap §5 suggested 1.5× near-pin ratio; we may relax
or tighten based on the 4-case numbers).

### Step 4 — Repeat step 3 with the ROM velocity, not the FOM velocity

**What changes**: source PVTU switches from
`cylindrical_<c>.gid/post/cylindrical_119.pvtu` (FOM) to
`/scratch/shared/ROM/ROM_recon_centered/cylindrical_<c>.gid/post/cylindrical_119.pvtu`
(reconstructed ROM).  Everything else — the grid geometry, the
HCT cache mechanics, the compare tool — is identical.

**Important reuse**: the per-case HCT cache
`hct_cache_ts119.npz` is content-hashed against the *velocity*
field, not just the mesh.  Since the ROM velocity differs from the
FOM velocity, the ROM builds will (correctly) rebuild the HCT
cache on first call.  So ROM step-4 costs **~2× step-3 build time**
(fresh SPR + Alfeld + Bernstein for the ROM velocity per case).

**Reference switches** too:
- Reference for step 3 was `FOM_mesh + HCT-3D`.
- Reference for step 4 is `ROM_mesh + HCT-3D`
  (`ROM_recon_centered/cylindrical_<c>.gid/post_pt/hct_on/...`).

**The interesting metric is *added* error**: for each variant,
report

    Δ = rms(ROM_grid vs ROM_mesh) - rms(ROM_mesh vs FOM_mesh)

If Δ is small compared to `rms(ROM_mesh vs FOM_mesh)` (the ROM
residual we already accept, ~1.5 – 2.5×10⁻² m for the 4 cases per
§2/§3 report), the grid is free from an accuracy standpoint.  If Δ
is comparable to the ROM residual, the grid is adding a second
error source and we cannot ship it without discussion.

**Deliverable**: a new report section, `Part 3 · ROM grid PT`, in
the grid report doc, with the same tables + figures but sourced
from ROM velocities.  A single-page comparison: ROM residual (in
metres) vs ROM-grid added error (in metres), per variant, per case.

**Acceptance**: winner's added error < 25 % of the ROM residual in
the near-pin bin.  Motivation: on top of a 20 % ROM PT Lagrangian
error, a further 5 % from the grid is invisible in downstream
mixing diagnostics; a further 20 % doubles the effective error.

### Step 5 — Rebuild the ROM basis from grid snapshots (frame as hypothesis, not conclusion)

**User's motivation**: mesh-side reconstruction has ~4 % Eulerian
rel_rms which amplifies to ~20 % Lagrangian PT error (matches the
Xiong et al. asymmetry).  Rebuilding the POD basis on grid-
projected snapshots may reduce one or the other.

**The hypothesis has two independent legs, and only one can be
right**:

1. **Grid-space is better-suited to POD**.  The finite element
   mesh has anisotropic refinement near the pin that gives high-
   frequency FEM modes a large L² norm they don't earn
   physically.  Projecting to a uniform-ish grid removes that
   weighting artefact, so a fixed number of grid-POD modes
   captures more of the true velocity variance than the same
   number of mesh-POD modes.  Testable by comparing singular
   value decay curves.
2. **Grid-space POD gives a smoother basis that the tracker can
   integrate more accurately**.  Independent of variance capture,
   the grid-projected snapshots are C¹ (via HCT-3D at build time
   and Catmull-Rom at query time), whereas the mesh basis is only
   C⁰ across tet faces.  A smoother basis means smaller local
   gradient jumps, which — per Xiong et al. — is exactly what
   compresses Lagrangian amplification.  Testable by rebuilding
   the ROM against the grid-POD and repeating the FOM-vs-ROM
   PT compare.

**Pre-check to avoid wasted work**: compute both SV decay curves
first.  The cheap SVD is: 20 snapshots (one per case at
ts = 119) × 3 velocity components × grid or mesh DoFs = a
`20 × M` matrix where M is 540 k (mesh, 180 k × 3) or 192 k
(grid, 64 k × 3).  Both trivial.

- If the grid-POD SV decay is materially faster (say the same 99 %
  energy in fewer modes), leg 1 is real.  Proceed to full rebuild.
- If the two SV curves are visually identical, leg 1 is null —
  grid-POD basis has the same information as mesh-POD.  The
  question is now purely about leg 2 (smoother basis =
  smaller Lagrangian error).  Still worth testing but the case is
  weaker.
- If neither leg pans out, the ~20 % Lagrangian error is dominated
  by the mode count (only 3 POD modes), not by mesh-vs-grid basis
  choice.  We would need more modes, not a different basis
  domain.

**Deliverable, phase A (SV decay check)**: a one-figure
comparison of the two SV curves.  ~50 lines of numpy.  A day.

**Deliverable, phase B (full rebuild, conditional on phase A)**:
a new `jaxtrace.rom` code path that produces a grid-space basis
+ per-case coefficients, plus a rerun of §2's FOM-vs-ROM PT
compare.  Multi-day.

**Acceptance for phase A**: SV curves emitted and interpreted.
Go/no-go decision on phase B written into this document.

**Acceptance for phase B**: the roadmap §2 near-pin PT residual
drops (relative to mesh-POD) by a factor consistent with the
SV-decay improvement from phase A.  If the two decays were
identical and phase B is still done for leg-2 reasons, the pass
criterion becomes: Lagrangian consistency error factor
(rms(PT ROM vs FOM) / rms(velocity ROM vs FOM)) drops below the
current ~5×.

---

## Review-2 revision (2026-07-31) · basis-construction alternatives to Step 5

See [rom_pt_roadmap_REVIEW2_evaluation.md](rom_pt_roadmap_REVIEW2_evaluation.md).
The literature check on Review 2's suggestions converged on a very
specific conclusion:

- Step 5's mesh-vs-grid POD comparison is **not the highest-leverage
  test** of the basis-quality hypothesis.  The literature (Christensen
  1999; Dellacasagrande 2021; Olesen 2022) shows energy-truncated
  POD's under-representation of low-KE regions is a structural
  property of the L² inner product, so **resampling the domain
  (mesh → grid) leaves the underlying energy distribution unchanged**
  and the SV decays should look nearly identical.  Step 5 phase A is
  therefore demoted from "primary hypothesis test" to "control
  experiment".
- The three basis-construction fixes below **directly** target the
  Xiong-Xie amplification we've measured, are all backed by peer-
  reviewed evidence on related problems, and are ordered by cost.

### Step 6 — Higher-mode-count POD (cheapest test of the basis-quality hypothesis)

**Cost**: hours. Rebuild the ROM basis with 5, 8, 12 modes instead of
3 and rerun §2's FOM-vs-ROM PT compare on the 4-case cohort.

**What it tests**: whether the workpiece mean-velocity bias identified
in the smoothness report §4 is a **mode-count** problem (basis too
small to represent the low-energy workpiece structure alongside the
high-energy pin structure) or a **mode-quality** problem (even with
enough modes, the L² inner product ranks them wrong).

**Predictions from the literature**:
- If it's mode-count: ROM residual should drop roughly with the
  captured energy fraction (5 modes ≈ 8–12 % Eulerian residual
  reduction; 12 modes potentially 20–30 % reduction).
- If it's mode-quality: adding modes will help less than expected —
  even 12 modes may leave the workpiece bias if the modes still
  rank by energy and the workpiece contributes little.

**Acceptance**: SV plot + FOM-vs-ROM PT compare table at each mode
count, on the 4-case cohort.  Go/no-go for Step 7 depends on whether
Step 6 closes the amplification to under ~2× (from the current ~7×).

### Step 7 — Lagrangian-inner-product POD (Xie et al. approach)

**Contingent on**: Step 6 not closing the gap.

**Cost**: multi-day. Add a new POD path in `jaxtrace.rom` that builds
the basis in a Lagrangian-weighted inner product per Xie et al.
2018/2020.  Snapshot collection unchanged; only the SVD weighting
changes.

**What it tests**: Xie et al.'s claim that Lagrangian-inner-product
POD is *"orders of magnitude more accurate"* than energy-weighted POD
at approximating Lagrangian quantities (with no closure modelling).

**Applicability caveat**: their demonstration is on QGE (2D geophysics),
not FSW.  Whether the "orders of magnitude" gain translates depends
on how much the workpiece region contributes to Lagrangian
predictability — the FSW workpiece bias suggests it contributes a lot.

**Acceptance**: on the 4-case cohort, rerun §2's FOM-vs-ROM PT
compare.  Pass if Lagrangian rel_rms drops by ≥2× vs baseline
(energy-weighted POD at same mode count).

### Step 8 — Discretely divergence-free POD basis

**Contingent on**: whether the FSW mixed u/p/e formulation output is
available, OR we're willing to project snapshots onto a div-free space
ourselves (Helmholtz-Hodge decomposition).

**What it tests**: the div-free inheritance documented in the
smoothness report §4 (ROM's div pattern is 99 %-correlated with FOM's).
Standard remedy across the incompressible-ROM literature (Akhtar 2009,
Stabile 2017, Novo & Rubino 2020, Star 2020).

**Acceptance**: div-free residual of the ROM basis drops by ≥10×;
follow-through into PT residual reduction is a separate question
(the review's claim is that this attacks the trapped-particle /
div-free-proxy anomaly at its source).

### Priority ordering (revised)

| priority | step | dev cost | expected impact |
|---|---|---|---|
| P0 | **Step 4** (ROM-on-winner-grid, unchanged) | 2 days | closes §5 acceptance for the ROM path |
| P0 | **Step 6** (higher-mode-count POD) | hours | cheapest test of basis-quality hypothesis |
| P0 | **Step 9** (integrator × interp pairing test — RK4+trilin vs RK6+tricubic on 4lvl_hct) | 2 – 3 days | Pokrajac 2002 predicts orders-of-magnitude gain if paired right; could reopen the accuracy floor Part 5 declared saturated |
| P1 | **Step 5 phase A** (SV decay check, unchanged) | half day | control experiment for domain sensitivity |
| P1 | **Step 7** (Lagrangian POD) if Step 6 falls short | multi-day | direct fix for Xiong amplification |
| P1 | **Step 11** (FTLE / LCS diagnostics on 4lvl_hct outputs) | day of numpy | closes roadmap §4 / §7 mixing-preservation question with the exact toolkit the stirred-tank literature uses (Kun Li 2022; Bashiri 2016) |
| P2 | **Step 8** (div-free POD basis) | days | attacks div-free inheritance |
| P2 | **Step 10** (temporal-derivative snapshot inclusion) | hours in static case | García-Archilla 2022 – viscosity-independent bounds |
| P3 | **Step 5 phase B** (full grid-space rebuild) | week | only if Step 5 phase A shows real gap |
| P3 | **Step 12** (paper writeup re-framing around stirred-tank literature) | writeup only | positions our work as toolkit application to a new domain, publishable |

Steps 4 and 6 can run in parallel (independent tests). Step 5
phase A is a cheap control that can run alongside either. Steps 7 and
8 are conditional on Step 6 not being sufficient. Step 9 is
independent of the basis question and answers it separately for the
integrator pairing. Step 11 uses existing PT outputs (no new runs),
so it can run any time after Part 5.

See [rom_pt_roadmap_REVIEW2_evaluation.md §"Extended literature scan"](rom_pt_roadmap_REVIEW2_evaluation.md#extended-literature-scan--additional-findings-from-4-further-consensus-searches)
for evidence backing steps 9 – 12.

**Implementation blueprints** for every step: see
[rom_pt_implementation_dossier.md](rom_pt_implementation_dossier.md).
Twelve sections covering grid design (§1), higher-mode POD (§2),
Lagrangian POD (§3), div-free basis (§4), grid-vs-mesh POD (§5),
JAX/GPU stack (§6), integrator + interp pairing (§7), volume-preserving
integrators (§8), FSW-specific mixing quantification (§9), ML-augmented
ROMs (§10), non-intrusive ROMs (§11), related-domain positioning (§12).
64 citations across the two literature passes; raw abstracts archived
in [rom_pt_literature_abstracts.md](rom_pt_literature_abstracts.md).

## Traps to avoid

- **"Top-3 by aggregate RMS" is not the same as "top-3 for FSW
  use case"**.  Aggregate rms averages over a distribution where
  a few hundred stuck-in-pin particles dominate the tail.  The
  per-r-bin split in step 2 is the fix; don't skip step 2
  before step 3.
- **The `run_grid_all.sh` for 16 missing cases needs to be
  auto-generated** — do NOT copy-paste manually, that's how case
  numbers get mismatched in `CASE_STEM`.  Sed-replace on the
  known-good 000 runner is the safe pattern.
- **Do not run more variants in step 3 than the "winners"**.
  Running all 24 variants × 16 more cases would be ~9 GB of
  npz + hours more compute for no gain.  Filter to the 3
  winners.
- **Step 4 reference is ROM-mesh, not FOM-mesh**.  Easy to get
  wrong because the grid PT run doesn't know or care what its
  reference is — the compare tool picks the reference based on
  the `--reference-variant` CLI.  Add an explicit `--reference-
  variant rom_hct_on` in the step-4 compare invocation.
- **Step 5 phase A is cheap; phase B is expensive**.  Don't
  commit to phase B until phase A shows a promising SV decay
  gap.  Otherwise the rebuild is likely to reproduce the current
  20 % Lagrangian error with a different basis but the same
  mode-count ceiling.

## Timeline expectation

Assuming a workstation with 8 GPU cores available overnight and
the local dev box for the tooling changes:

| step | dev time (this box) | compute time (workstation) |
|---|---|---|
| 1 | 1 h (parse logs, table, doc update) | 0 (existing logs) |
| 2 | 3 – 4 h (tool + figures + doc) | ~15 min (rerun compare) |
| 3 | 1 h (auto-generators) | ~7 h (16 mesh PT overnight + ~1 h grid) |
| 4 | 0.5 h (source-PVTU switch + doc) | ~2 h (same 16 cases, ROM velocities) |
| 5 phase A | 0.5 day (SVD + figure) | 0 (in-process numpy) |
| 5 phase B | 3 – 5 days if pursued | 1 – 2 days (bulk rebuild + rerun of §2) |

Steps 1 – 4 close the roadmap's stated §5 acceptance criterion for
the 20-case cohort with FOM-quality PT throughput of ~80× the
current mesh path.  Step 5 is the answer to *"is the current
mesh-space ROM basis a bottleneck for FSW mixing predictability?"*
and is worth pursuing only if the SV-decay pre-check justifies it.
