# Pathline-integrated damage — documents and figure generators

Working documents for the FSW void/defect-susceptibility study (Stage 1 damage
survey, Phase 4 RK4 run of 2026-09/10). Kept here so the reasoning survives the
scratch directory it was written in.

⚠️ **Figures and result files are deliberately NOT in the repo.** The source tree
they came from holds ~1.8 GB of PNG/VTU/NPZ output (`figs_traj/` alone is 1.6 GB)
plus `results/` and `vtu_final/`. Only the documents and the scripts that
regenerate the figures are tracked. The scripts read from the LUMI run directory,
not from anything stored here, so a figure is reproduced by re-running its script
against the run output — see `RUN_ON_LUMI.md`.

## What to read first

| Document | What it is |
|---|---|
| `DAMAGE_REPORT.md` | The standalone report — the primary deliverable, written to be presented from. |
| `damage_talk_meeting.md` | The meeting deck (Marp). 79 visible slides; 18 more parked in a trailing HTML comment. |
| `FUTURE_MODELS_EXPLAINED.md` | Lee–Dawson and GTN derived from scratch, every symbol and parameter; plus a ranked table of 14 candidate models. |
| `PRIMER_fsw_physics_for_me.md` | Background physics, written as self-study notes. |
| `OPEN_QUESTIONS.md` | The running log of open items, numbered (O1, O2, …). Claims retracted after checking are recorded here with their replacements. |

## The rest

Planning and process: `IMPLEMENTATION_PLAN_void_damage.md`,
`STAGED_VALIDATION_PLAN.md`, `PHASE1_GATE_AND_MEASUREMENTS.md`,
`PHASE4_HANDOVER.md`, `BENCHMARK_READINESS.md`, `DAMAGE_STORAGE_PLAN.md`,
`RUN_ON_LUMI.md`.

Literature and alternatives: `FSW_void_prediction_literature_review.md`,
`COMPARISON_with_flow_analysis.md`, `Estimate_bubbles_perplexity.md`.

Presentation material: `damage_implementation_talk.md`,
`damage_talk_FULL_reference.md` (untouched 90-slide archive),
`void_prediction_talk.md`, `CARDS_one_concept_each.md`,
`DAMAGE_PRESENTATION_GUIDE.md`, `FIGURES_INDEX.md`.

## scripts/

Figure generators and measurement tools. The ones used for the current figures:

| Script | Produces |
|---|---|
| `make_wake_metrics.py` | Wake-box metrics plus the raw-particle and continuous-field cross-sections. |
| `make_pt_vs_damage.py` | Particle-tracking vs damage comparison panels. |
| `make_density_vs_damage.py` | Separates particle density from accumulated damage (fixed-area estimator). |
| `make_regional_metrics.py` | Regional scores, added because top-layer values dominate the global ones. |
| `make_trajectory_figures.py` | Pathline figures, one representative case per tool family. |
| `make_concept_figures.py`, `make_phase3_figures.py` | Explanatory and Phase 3 figures. |
| `measure_pin_geometry.py`, `measure_delta_window.py`, `survey_lumi_cases.py` | Measurements quoted in the documents. |
| `compare_euler_rk4.py`, `make_conservation_test.py` | Order-1 vs order-4 damage quadrature; conservation checks. |
| `npz_to_vtu.py` | Converts run output for ParaView. |

PDF rendering: `make_pdf.py` (Marp deck → beamer) and `make_doc_pdf.py` (prose →
article). `check_slide_fit.py` checks whether a slide's content fits its frame —
it measures image heights from the real file aspect ratios, because `![w:1000]`
sets only the width.

⚠️ `make_doc_pdf.py` rewrites Unicode before calling pandoc. xelatex **drops** a
character its font lacks while only warning, so `8.03×10²⁶ s⁻¹` once rendered as
`8.03×10 s`. Superscripts become real LaTeX maths and symbols are wrapped in
`\ensuremath` so they also survive inside bold. Don't remove that pass.
