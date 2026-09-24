# ROM PT · Paused-work checkpoint

**Paused:** 2026-07-31
**Reason:** Author must revise the MALMO paper (`jaxtrace_final.tex`) in response to reviewer comments.
**When resuming:** open this file first, then follow the pointers below.


## ⚠ Update 2026-09-09 — the ROM track was closed on bad data

The shipped FEMUSS basis reconstructs this cohort at 4.04 % L2; a POD
built from the same 20 snapshots reaches 0.52 % with the same 3 modes.
Every ROM particle-tracking number that informed the decision to close
the ROM track was measured on the defective field.

Rerun on a corrected basis (5 of 20 cases so far): ROM-FOM displacement
**6.79 -> 2.32 mm at step 950** (x0.34) and **19.79 -> 4.22 mm at step
2000** (x0.21). On those cases the ROM path is now *more accurate than
the grid path* (5.62 mm).

See [`rom_pt_ourpod_rerun.md`](rom_pt_ourpod_rerun.md). The §4
under-mixing analysis — the physical argument for closing the track —
has **not** been rerun and is the most important gap.

## One-line status

**Grid PT for FSW cases:** `4lvl_hct` chosen as definitive winner
across the 20-case cohort (mean rms 5.6 mm mid-run, 7.8 mm final,
49× faster than mesh + HCT-3D). Grid-topology optimisation is
saturated per Part 5 of the grid report. Next lever is basis
construction, not grid.

## Where we left off

- **Grid experiment (§5/§6 of roadmap): DONE**
  - 24 variants tested on 4-case cohort (Parts 1–3 of grid report)
  - Top-3 rolled out to all 20 cases (Part 4)
  - 3 hypothesis-driven follow-up variants tested on all 20 (Part 5)
    — none beat `4lvl_hct` (all Wilcoxon p ≤ 0.03 worse)
  - Per-case xy error maps generated for `4lvl_hct` + `malmo6_hct`
    with extended x-axis to include ballistically-extended particles

- **Two literature reviews done**:
  - REVIEW2 evaluated with 7 Consensus searches
  - Extended literature scan with 19 more Consensus searches
  - **Consolidated 64-reference implementation dossier written**

- **Next work planned but NOT started** (see plan doc for full order):
  - Step 4 (ROM-on-winner-grid) — closes §5 acceptance for ROM path
  - Step 6 (higher-mode-count POD) — cheapest test of basis-quality
  - Step 9 (integrator × interp pairing) — RK6+tricubic on 4lvl_hct
  - Step 7 (Lagrangian-inner-product POD, Xie 2020) — if Step 6 falls short
  - Step 8 (div-free POD basis) — if the mixed u/p/e output is available

## Documents to open on resume, in order

1. [`rom_pt_roadmap.md`](rom_pt_roadmap.md) — overall status per section
2. [`rom_pt_next_steps_plan.md`](rom_pt_next_steps_plan.md) — 12-step priority table
3. [`rom_pt_step5_step6_grid_report.md`](rom_pt_step5_step6_grid_report.md) — grid experiment 5 parts + xy maps
4. [`rom_pt_implementation_dossier.md`](rom_pt_implementation_dossier.md) — implementation blueprints per step
5. [`rom_pt_roadmap_REVIEW2_evaluation.md`](rom_pt_roadmap_REVIEW2_evaluation.md) — first literature evaluation
6. [`rom_pt_literature_abstracts.md`](rom_pt_literature_abstracts.md) — raw abstract archive (offline-safe)

## Key artefacts on the workstation

- **Grid modules built** (~48 per case × 4 cases + 3 per case × 16 cases + 3 per case × 20 cases for Part 5)
  under `/scratch/shared/ROM/FOM/cylindrical_<c>.gid/grid_velocity_*.py`
- **HCT caches** (SPR + Alfeld + Bernstein coeffs, ~240 MB each)
  at `/scratch/shared/ROM/FOM/cylindrical_<c>.gid/hct_cache_ts119.npz`
- **All grid PT runs** at `/scratch/shared/ROM/FOM/cylindrical_<c>.gid/out_grid_<VAR>/`
- **Compare CSVs and figures** (96–120 rows depending on variant set)
  at `/scratch/shared/ROM/rom_out/fom_grid_pt/` (4-case) and
  `/scratch/shared/ROM/rom_out/fom_grid_pt_all20/` (20-case)
- **Presentation figures** at
  `/home/arhashemi/Workspace/welding/JAXTrace/paper_figs/rom_pt_step5_step6_all20/`
  (mirrored to `/scratch/shared/ROM/rom_out/fom_grid_pt_all20/presentation_figs/`)

## Key runner + builder scripts on disk

- `/scratch/shared/ROM/FOM/build_hct_grid_matrix.sh` — 48-variant HCT builder (Part 2 rollout)
- `/scratch/shared/ROM/FOM/build_grid_top3.sh` — 3-winner builder for 20 cases (Part 4 rollout)
- `/scratch/shared/ROM/FOM/build_grid_new3.sh` — 3-new-variant builder (Part 5 rollout)
- `/scratch/shared/ROM/FOM/launch_top3_all20.sh` — 4-phase orchestrator for 20-case top-3
- `/scratch/shared/ROM/FOM/launch_new3_all20.sh` — same for the 3 new hypothesis variants
- `/scratch/shared/ROM/FOM/cylindrical_<c>.gid/run_grid_all.sh` (24 variants) and
  `run_grid_top3.sh` (3 winners) and `run_grid_new3.sh` (3 new) per case
- `scripts/build_hier_grid_velocity_case.py` — grid builder (has `--hct-cache`, `5lvl` grid_type)
- `scripts/compare_grid_vs_mesh_pt_fom.py` — enriched compare tool (per-r-bin,
  trapping, escape, per-comp, spatial heatmap, both-alive filter)
- `scripts/plot_grid_pt_presentation.py` — presentation figures (7 PNGs, 6 variants)
- `scripts/plot_final_step_xy_error_maps.py` — per-case xy maps with relative-error colour

## Resume prompt template

When user asks "where were we on ROM PT?", answer:

> Grid PT is done: `4lvl_hct` wins at 5.6/7.8 mm cohort-mean rms
> (mid-run/final), 42-49x faster than mesh, 20 cases x 6 variants,
> topology saturated.
>
> **But the ROM track was closed on bad data.** The shipped FEMUSS basis
> reconstructs at 4.04 % L2; our own POD from the same snapshots reaches
> 0.52 % at K=3. Rerunning ROM PT on the corrected basis (5/20 cases)
> cut ROM-FOM displacement from 6.79 to 2.32 mm at step 950 and 19.79 to
> 4.22 mm at step 2000 — making the ROM path *better than the grid path*
> on those cases. See `rom_pt_ourpod_rerun.md`.
>
> Next: finish the remaining 15 cases
> (`bash /scratch/shared/ROM/FOM/run_ourpod_rom_pt_all20.sh`), then rerun
> the §4 under-mixing analysis, which was the physical argument for
> closing the ROM track and has not been re-checked.
