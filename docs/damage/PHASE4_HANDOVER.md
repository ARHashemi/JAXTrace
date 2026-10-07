# Phase 4 overnight run — what was submitted and what to check

**Job: `22476725`** — array `0-44%24`, submitted 2026-10-01 evening.
Output: `/scratch/project_465002752/hashemia/damage/phase4_rk4_30mm/<case>/`

---

## 1. Check it finished

```bash
ssh lumi
sacct -j 22476725 --format=JobID%18,State%12,ExitCode,Elapsed -X
# 45 tasks. Expect COMPLETED for all; worst case was predicted at 2.70 h
# against a 06:00:00 wall.
```

If any task shows `TIMEOUT` / `FAILED`, re-run only those:

```bash
cd /projappl/project_465002752/hashemia
MISSING_ONLY=1 bash sbatch_phase4_rk4_array.sh     # lists what is missing, no submit
MISSING_ONLY=1 sbatch --array=0-<N> sbatch_phase4_rk4_array.sh
```

`case_is_complete()` requires a non-empty npz that **opens** with the right
particle count and all-finite values — so interrupted or truncated output is
correctly treated as missing, not as done.

⚠️ **Check `/flash` is empty.** The staging dirs should all have been rsync'd
away. Anything left means a transfer failed and the results are still there:

```bash
ls -d /flash/project_465002752/hashemia/phase4_rk4_30mm_* 2>/dev/null
# expected: nothing
```

---

## 2. The one question this run exists to answer

Phase 3 found that **52–88 % of particles accumulated essentially zero damage**
because they streamed past the tool without entering the shear layer (O34), which
made the global median a measurement of the bypass fraction rather than of damage.

This run triples the travel distance (10 → 30 mm) with the seed box unchanged.

```bash
# on the workstation, with the lumi mount:
cd ~/Workspace/welding/1phase-2phase
python3 compare_euler_rk4.py \
  --euler ~/lumi/lumi_scratch/hashemia/damage/phase3_overnight_20260930 \
  --rk4   ~/lumi/lumi_scratch/hashemia/damage/phase4_rk4_30mm
```

**Read the "mean bypass reduction" line:**

| outcome | meaning | what to do next |
|---|---|---|
| **large drop** (say > 20 pts) | O34 was a DURATION problem — particles simply had not travelled far enough | proceed to regional metrics and the family regression |
| **small drop** (< 10 pts) | the bypass population is **structural**: those particles never enter the stir zone no matter how long you run | the seed box, not the duration, is what needs changing |

⚠️ **A warning sign already visible.** The rate probe (A1, 3 mm, 100k particles)
ended with **mean x = −7.82 mm** and **93.1 % bypass**. Particles moved *backwards*
relative to the nominal travel direction. That is the recirculation O34 describes.
If the 30 mm run shows the same, the answer is "structural" and the next run needs
a seed box that actually feeds the stir zone.

---

## 2b. Build the Phase 4 summary CSV (do this before the figures)

The figure scripts read a summary CSV, not the npz files. `p3summary.py` on LUMI
now takes the run directory as an argument (regression-checked: it reproduces the
Phase 3 CSV **byte-identically**):

```bash
ssh lumi
cd /projappl/project_465002752/hashemia
singularity exec --cleanenv \
  --env PYTHONPATH=/projappl/project_465002752/hashemia/JAXTrace:/projappl/project_465002752/hashemia/required-packages \
  /appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif \
  python3 p3summary.py /scratch/project_465002752/hashemia/damage/phase4_rk4_30mm
# -> .../phase4_rk4_30mm/phase4_rk4_30mm_summary.csv
```

Then regenerate every result figure against Phase 4 (the figures themselves are
run-agnostic; only the CSV path changes):

```bash
# on the workstation
cp ~/lumi/lumi_scratch/hashemia/damage/phase4_rk4_30mm/phase4_rk4_30mm_summary.csv results/
python3 make_phase3_figures.py \
  --summary results/phase4_rk4_30mm_summary.csv \
  --out figs_phase4
```

---

## 2c. ⚠️ Use the WAKE BOX, not `stir_nobc`, for Phase 4

```bash
python3 make_wake_metrics.py \
  --src ~/lumi/lumi_scratch/hashemia/damage/phase4_rk4_30mm \
  --out figs_wake --binned
# -> results/phase4_rk4_30mm_wake.csv
#    figs_wake/figW_<family>_yz.png         (scatter)
#    figs_wake/figWB_<family>_yz_binned.png (2D median field — use these)
```

At 30 mm only **2.6 %** of particles remain inside r < 7 mm, so the Phase 3
`stir_nobc` mask is nearly empty (median 2013 of 100,000) and reports `0.000`.
**That is an empty mask, not a null result.** The wake box holds ~90,000.

---

## 3. Then the regional metrics

```bash
python3 make_regional_metrics.py \
  --src ~/lumi/lumi_scratch/hashemia/damage/phase4_rk4_30mm
# -> results/phase4_rk4_30mm_regional.csv
```

Compare the family table against Phase 3's. Phase 3 at 10 mm / Euler gave:

| family | global median | **stir median** | z_top p99 | bypass |
|---|---|---|---|---|
| A-Flats | 0.0650 | 0.3824 | 358.6 | 80.6 % |
| **B-Flutes** | 0.0654 | **0.4131** | 356.1 | 80.3 % |
| C-Threads | 0.0653 | 0.3790 | 357.5 | 80.5 % |
| D-ConcavityTilt | 0.0521 | 0.3050 | **22.6** | 87.2 % |
| cohort | 0.4909 | 2.2787 | 265.7 | 57.7 % |

The A/B/C spread was **0.6 % on the global median but 9.0 % on the stir-zone
median**, with B-Flutes highest — which is the expected ordering, since B has the
deepest lobes (20–32 % of pin radius). **Does that ordering survive at 30 mm?**
That is the result worth having.

⚠️ `z_top` is reported separately on purpose: it sits against the prescribed-
velocity top surface and is partly a BC response. D's 22.6 vs ~358 for A/B/C is
the likely explanation for O32.

---

## 4. ParaView

```bash
# per-case trajectories, already exported this time (every 50 steps):
ls ~/lumi/lumi_scratch/hashemia/damage/phase4_rk4_30mm/ps_A-FlatsVariations_A1/*.vtu
```

Open the `particles_step_*.vtu` series, Representation = Points, colour by
`Damage_lnPhi` or `Damage_CL`. ParaView will group the numbered files into a
time series automatically.

⚠️ `particles_step_000000.vtu` has **no damage arrays** — it is the initial state,
written before any tracking step. Not a bug; start the animation at the second
frame.

⚠️ For a colour scale use `Damage_fracFailed` if you add it, or clamp `lnPhi`
manually: it is unbounded (O31, no saturation term) and reaches ~1270–5242, so an
auto-range makes everything look uniformly blue.

---

## 5. What was verified before submitting

| check | result |
|---|---|
| RK4 damage exports to VTU, growing monotonically | ✅ cohort 0.0134→0.0270→0.0364; PinShapes 0.00033→0.00223→0.00446 |
| `--damage-order` does not perturb position | ✅ **max abs diff = 0.000e+00 m** |
| order 4 vs order 1 effect | median **+0.01 %**, max **−17.9 %** — a **tail** correction |
| throughput at production scale | ✅ **119,525 p·step/s**, i.e. **−2.3 %** vs Phase 3 Euler (within noise) |
| flash → scratch move | ✅ clean on both normal exit and SIGTERM |
| output directory layout | ✅ flat (`$OUTDIR/$TAG/`), nesting bug fixed |
| Phase 3 results preserved | ✅ separate output directory, untouched |

⚠️ **Order 4 is a tail correction, not a general accuracy win.** A typical particle
changes by 0.05 %; the maximum drops 18 %. So it changes `frac_failed`, `p99` and
`lnphi_max` rankings and leaves **median**-based rankings essentially unchanged.
The Euler tail was overestimated by ~18 %.

---

## 6. Still open

* **O33** — the advancing-side flank asymmetry still does not reproduce. The
  bypass dilution (O34) explains the *family* ranking collapse; it does **not**
  yet explain the flank result.
* **O4** — no experimental defect ground truth exists, in this work or in the
  published FOM validation (forces, torque, thermocouples only). Both analyses
  produce a **susceptibility ranking**, not a validated defect prediction.
* **O8** — the damage work is still uncommitted on both checkouts.
* The deck `damage_implementation_talk.md` currently cites Phase 3 numbers. It
  needs the Phase 4 figures once the comparison above is run.


---

## 7. ✅ DONE 2026-10-02 — results, figures and documents

All 45 cases completed with zero failures; `/flash` drained cleanly.

| artefact | where |
|---|---|
| Wake-box metrics | `results/phase4_rk4_30mm_wake.csv` |
| Summary | `results/phase4_rk4_30mm_summary.csv` |
| Result figures | `figs_phase4/` |
| Wake cross-sections | `figs_wake/figWB_<family>_yz_field.png` |
| Trajectories | `figs_traj/figT_ps_A-FlatsVariations_A1_*` |
| **Speaker guide** | **`DAMAGE_PRESENTATION_GUIDE.md`** — share this before the talk |
| Deck | `damage_implementation_talk.md` (45 slides) |

### Headline

B-Flutes is highest on **every** wake-box statistic (mean 7.206, p99 80.9,
frac_failed 21.0 %), consistent with having the deepest lobes. The trajectories show
the mechanism: damage is acquired in **discrete capture events** in the shear layer
at r ~ 3-5 mm and z ~ -0.5..-2 mm.

### Still open

* **Flank asymmetry**: PinShapes now balances (12/20) but the cohort stays
  retreating-higher (0/25, ratio 0.641) -- unexplained.
* **D-Concavity**: only ~63,800 particles reach the wake vs ~88,000 elsewhere, so
  its low score is partly a capture-rate effect. Not yet separated.
* **No experimental defect data** -- this is a susceptibility ranking, not a
  validated prediction.

### Cheap next steps (read-only, no submission)

```bash
# does B capture more often?  does D capture at all?
python3 make_trajectory_figures.py --case <...>/phase4_rk4_30mm/ps_B-FluteVariations_B2 \
    --out figs_traj --rpm 996 --dt 3.629e-4
python3 make_trajectory_figures.py --case <...>/phase4_rk4_30mm/ps_D-ConcavityTilt_D1 \
    --out figs_traj --rpm 996 --dt 3.629e-4
```

⚠️ First run per case reads 167 VTU files over the mount (~8 min); it then caches to
`figs_traj/_cache_<case>.npz` (~300 MB) and re-plots in seconds.
