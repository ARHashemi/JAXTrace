# Commands to run on LUMI — updated 2026-09-30

> ✅ **Phases 2 and 3 are implemented and deployed** (2026-09-30). The next LUMI job
> is **§0 below: the Phase 3 damage sweep.** Earlier runs in this file are complete:
> δ sweep (job 22450486, 45 cases, O25-corrected), M5 (22404873), M4 (interactive).

Everything below is deployed and smoke-tested. Run from
`/projappl/project_465002752/hashemia`.

Read-only on the colleague's case folders; all output under
`/scratch/project_465002752/hashemia/`.

---

## 0. ⭐⭐ PHASE 3 as a JOB ARRAY — parallel across LUMI nodes

```bash
cd /projappl/project_465002752/hashemia

# finish the 10 cases the sequential job missed — ~1.3 h instead of ~10 h
MISSING_ONLY=1 sbatch --array=0-9%10 sbatch_phase3_array.sh

# or all 45 from scratch, in two waves of 24 — ~2.5 h instead of ~22 h
sbatch sbatch_phase3_array.sh
```

### Why this replaces the sequential job

The sequential run (22455819) did 35/45 in 15:49 and died on the 16 h wall. The
work is embarrassingly parallel — each case is an independent tracking run over its
own mesh — so the loop was serialising something that need not be.

Measured limits on this system:

| | |
|---|---|
| `small-g` MaxJobs / MaxSubmit | **200 / 210** |
| GCDs per node | **8**, `OverSubscribe=NO` |
| our allocation used | **212 / 18,000 GPU-h (1.2 %)** |

| | wall | GPU-hours |
|---|---|---|
| sequential | ~22 h ❌ | ~22 |
| **array, 24 concurrent** | **~2.5 h** ✅ | ~22 |
| array, 45 concurrent | ~1.3 h | ~22 |

⚠️ **Parallelism does not reduce GPU-hours, only wall time.** At 1.2 % of
allocation used, GPU-hours were never the constraint — the 16 h wall was.

`%24` caps concurrency deliberately below the 200 limit: it leaves room for other
project members, covers 45 cases in two waves, and surfaces a bad commit after 24
failures rather than 45. Raise to `%45` for a single wave.

⚠️ **`--time=02:00:00` is PER TASK.** The slowest measured case (B2, 3333 steps on
a 2.3M-node mesh) took 1 h 18 m, so 2 h gives ~35 % headroom. **A task that
overruns fails alone** — the other 44 are unaffected, which is the other advantage
over one monolithic job.

### ⚠️ "Already done" means CLEANLY finished — three independent checks

A bare "does the file exist" test is not safe here. Job 22455819 was killed
mid-case on B4 and **left a directory containing an empty `search_stats.csv`**, so
interrupted-run debris is real; and a truncated `.npz` from a cut-off write would
look like success forever.

`case_is_complete()` requires **all three**:

1. the per-case log ends with **`Done.`** — `run_tracking.py`'s own marker
   (B4's log ends mid-progress at `Step 480/2756`)
2. the log contains `[damage] wrote ...damage_models.npz`
3. **the `.npz` opens in numpy**, its `damage` array has exactly `N_PARTICLES`
   rows, all values are finite, and `positions` matches

Verified against real data:

| case | verdict |
|---|---|
| A1, B3, rom_000 (clean) | **COMPLETE** → skipped |
| B4 (killed mid-run) | **INCOMPLETE** → re-run |
| C1 (never started) | **INCOMPLETE** → re-run |
| *truncated npz + a complete-looking log* | **INCOMPLETE** ✅ caught |
| *good npz, wrong particle count* | **INCOMPLETE** ✅ caught |

The last two were tested by deliberately corrupting a copy — condition 3 is what
catches them, since 1 and 2 would both pass.

### ⚠️ One subtlety in the design

Array tasks are **independent processes**, so each one rebuilds the case list and
indexes into it. The list is therefore built with an explicit `sort` — an unsorted
glob could order differently between tasks, and two tasks would then run the same
case while another was never run at all. Verified stable across repeated
invocations.

With `MISSING_ONLY=1` the indices are **repacked**, so the 10 remaining cases are
tasks 0–9. Confirmed: exactly B4, C1–C4, D1, D2, D2old, D3, D4.

---

## 0b. (superseded) PHASE 3 sequential — smoke test, then overnight

### Step 1: smoke test (~2–3 min) — run this first

```bash
cd /projappl/project_465002752/hashemia
sbatch sbatch_phase3_smoke.sh
```

One case (`rom:000`), 10 mm travel, 20k particles → 533 steps, ~60 s of tracking
plus ~90 s fixed cost.

⚠️ **Must go through `sbatch`, not `bash`.** An interactive run on a LUMI login
node dies with *"No visible GPU devices"* — login nodes have no GPU. (I hit this
myself; the `--partition=small-g` header is what fixes it.)

**It prints PASS/FAIL at the end and tells you whether to proceed.** What it
exercises that the workstation test could not: the loop wrapper (case discovery,
PVTU selection, per-case logging, summary parser), the models path under **ROCm**
rather than CUDA, and `--damage-steps-mm` reading this case's `data/*.som.{dat,fix}`
on LUMI.

Expect roughly **25–40 % of particles inside r < 9 mm** and `rank corr(lnPhi, C)`
above 0.99.

### Step 2: overnight (~12.4 h) — all 45 cases

```bash
sbatch sbatch_phase3_overnight.sh
```

| family | n | est. |
|---|---|---|
| ROM/FOM_cases (the (v,ω) cohort) | 22 | 2.58 h |
| ROM/Validation | 3 | 0.35 h |
| PinShapes (tool geometry) | 20 | 9.50 h |
| **total** | **45** | **12.4 h** (16 h wall) |

⚠️ **Cheapest cases run first** — cohort and Validation before PinShapes. If the
wall is hit, the losses are the expensive tail rather than an arbitrary one, and
the cohort (which the ω/v_adv conclusions rest on) completes early.

### ⚠️ Why 10 mm and not 20

Cost from each case's **own** dt and v_adv, at 100k particles:

| travel | cohort | Validation | PinShapes | total |
|---|---|---|---|---|
| **10 mm** | 2.58 h | 0.35 h | 9.50 h | **12.4 h** ✅ |
| 20 mm | 4.61 h | 0.62 h | 18.49 h | 23.7 h ❌ |

PinShapes dominates because its dt is ~10× smaller (3.6e-4 vs 3.8e-3 s), so the
same *distance* costs ~10× the steps. **10 mm still carries particles through the
tool**, which is what the advancing-side signature needs; 20 mm would add wake
coverage, not stir-zone coverage.

For 20 mm on the cohort alone (4.6 h): `TRAVEL_MM=20 FAMILY=rom sbatch sbatch_phase3_overnight.sh`

### What it computes

Both damage models along every pathline, together:

```
Rice–Tracey       d(lnPhi)/dt = 0.849 · edot · exp(1.5 · eta)
Cockcroft–Latham  dC/dt       = max(s1_ratio, 0) · edot
```

The **rank correlation between them is printed per case**. Two structurally
different laws agreeing is the cheapest evidence available; the summary flags any
case below 0.9, because disagreement is diagnostic.

### ⚠️ Three things to know before quoting the output

1. **lnPhi is a RANKING, not a porosity.** Rice–Tracey has no saturation term, so
   it grows without bound past Φ=1 (`lnPhi = 9.21` for Φ₀=1e-4). The output prints
   `PAST FAILURE: x %` for exactly this reason. See O31.
2. **PinShapes rotates CCW (RPM > 0), the cohort CW (RPM < 0).** The advancing
   flank is detected per case so this is handled — but **do not pool the two
   families in one statistic.**
3. **`rank corr < 0.9` on any case** means the two models disagree there.
   Investigate before using that case's ranking.

### Two bugs found and fixed while preparing this

* **`ps:Results/*`** — `PinShapes/Results/` has no `*.gid`, and an unmatched bash
  glob stays literal, so a phantom 46th case leaked into the list. It would have
  failed at the **end** of a 12 h job. Guarded in both scripts; count now 45 exactly.
* **`val:` prefix unhandled** — the worker only knew `rom:` and `ps:`, so all 3
  Validation cases would have been silently skipped. Added.

---

## 1a. ✅ (done) δ sweep re-run — job 22450486, 45 cases

```bash
cd /projappl/project_465002752/hashemia
sbatch sbatch_delta_window.sh
```

The completed run (**job 22448098**, 65 cases, D4 recovered ✅) is now **stale in two
ways**, both fixed in the deployed scripts:

| correction | effect on the numbers |
|---|---|
| **O25 solved** — `.mat` tables are in SHEAR measures; `sigma_eq_norton` now applies √3·(√3)^m = **1.816** | the cross-check column should go **0.569 → ~1.03**; absolute η rises by 1.816× |
| **O26 answered** — `FOM_cases_PT` duplicates `FOM_cases`; dropped from `SCOPE=fom` | **65 → ~45 cases**; cohort-wide n no longer double-counted |

⚠️ **Rankings and ratios will not move** (a constant factor divides out of a ratio,
and the per-root ρ values were already computed separately). What changes is the
**absolute η scale** and the **case counts**. Both appear in figures and text, so the
re-run is worth the 30 minutes before anything is presented.

**What to check:** `Norton / solver` median should now read **≈ 1.03**, not 0.569, and
the case count should be ~45 with no `cylindrical_*` appearing twice.

---

## 1b. δ sweep — the earlier (superseded) run, for reference

```bash
cd /projappl/project_465002752/hashemia
sbatch sbatch_delta_window.sh
```

**Why re-run:** the PVTU picker now walks *down* to the newest file that actually
declares `Displacement` and `LEVEL`. D4's last two writes (`C2_200`, `C2_201`) are
truncated shells containing only `Points`; `C2_199` is complete. Verified: D4 now
resolves to `C2_199.pvtu`.

**What changes:** 64 → **65 cases**, and the D-family goes from n=4 to **n=5**, which
matters because D showed the strongest within-family correlation in the static window
(ρ = +1.000 on n=4 — a value that is nearly meaningless at that sample size).

Partition `small`, no GPU. Expect a `NOTE D4 ... skipped 2 truncated` line.

✅ **Verified standalone before you run it** (2026-09-30): D4 resolves to
`C2_199.pvtu` and gives `ratio_static` = **0.9341** against the published
phase-average **0.943** — **0.9 % agreement**, which confirms the picker chose a
physically equivalent timestep and not merely a readable one. `r_pin` = 2.17 mm,
δ = 1.49 mm, Norton/solver = 0.5689 — all consistent with its family and with the
other 64 cases.

---

## 2. M4 pin geometry — validated descriptors only (~5 min)

> ✅ **RUN 2026-09-30.** Outcome in **`BENCHMARK_READINESS.md` §3**. Short version:
> `n_lobes`/`lobe_depth` worked on **all 20** cases and separate the families
> (B-Flutes 3–8 lobes @ 0.20–0.32 depth; C-Threads 2 @ 0.03–0.08). ⚠️ The pin
> **envelope** failed on **9 of 20** — coarse CAD tessellation (A2 has 4 populated
> z-bins of 60), **not a code fault**; two fix attempts each produced a different
> wrong answer and were reverted. **Usable: 13 cases (B, C, D).** Finer STL exports
> from CAD would fix it.

```bash
cd /projappl/project_465002752/hashemia
singularity exec --cleanenv \
  --env PYTHONPATH=/projappl/project_465002752/hashemia/JAXTrace:/projappl/project_465002752/hashemia/required-packages \
  /appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif \
  python3 -u measure_pin_geometry.py \
    --cases-root /scratch/project_465002752/lorenzgl/Cases/PinShapes \
    --out /scratch/project_465002752/hashemia/damage/pin_geometry
```

Fast enough to run interactively — it reads STLs, not PVTUs.

**What ships in the CSV:** `r_max_mm`, `height_mm`, `volume_mm3`, `area_mm2`,
`sphericity`, `hull_deficit`, **`n_lobes`**, **`lobe_depth`**, `n_triangles`,
`stl_md5`, `pin_isolated`.

⚠️ **`n_lobes` + `lobe_depth` is the pair that matters** — B1 = 3 flutes at 30 %
depth, C1 = 2 at 6 %, D3 = 8 at 18 %. That is the first real capture of the
independent variable.

**What was removed from the CSV** (kept in the per-case JSON under `unvalidated`,
never tabulated): `thread_lead_mm`, `concavity`, `taper_deg`. Each produced a
plausible-looking wrong answer — the thread lead came out as **5.35 mm for both a
fluted and a threaded pin**, which was half the mis-detected domain length.

⚠️ **Expect ~3 cases to report `PIN ISOLATION FAILED`** (A1, A2, D1 — coarse STLs of
2.8k–11k triangles). Their envelope columns are **blanked rather than filled with the
shoulder radius**, because a 7.00 mm "pin radius" in a table is worse than a gap.
`n_lobes` / `lobe_depth` are still reported for them.

---

## ~~3. O26 check~~ — ✅ no longer needed

**Answered by the colleague 2026-09-30:** `FOM_cases` and `FOM_cases_PT` are the same
simulations; `FOM_cases` is the updated version. `SCOPE=fom` now excludes
`FOM_cases_PT`, which is why step 0 drops the case count from 65 to ~45.

---

## 4. Optional — O27 phase-averaged δ for the sparse cases only

Only if you want absolute per-case δ ratios rather than rankings. Four cases differ
from their revolution average by 20–48 %, and they are all low-`n_selected`:

```bash
cd /projappl/project_465002752/hashemia
MODELS=norton FAMILIES=pinshapes PHASE=revolution \
  CASES="lumi:A-FlatsVariations/A2 lumi:A-FlatsVariations/A2old \
         lumi:A-FlatsVariations/A4old lumi:D-ConcavityTilt/D2old" \
  OUTDIR=/scratch/project_465002752/hashemia/damage/o27_phaseavg \
  sbatch --job-name=o27 --partition=small --account=project_465002752 \
         --time=06:00:00 --mem=220G --cpus-per-task=16 \
         --output=/scratch/project_465002752/hashemia/logs/%x_%j.out \
         --wrap 'bash /projappl/project_465002752/hashemia/run_damage_lumi.sh'
```

⚠️ **Hours, not minutes** — a full revolution is 166 PVTU reads per case, and mesh
loading was measured at 77 % of total runtime on the big meshes (O21b). **Not needed
for Phase 3**, which works on one configuration. Needed before Phase 4.

---

## What to check when each finishes

| job | success looks like | failure mode to watch |
|---|---|---|
| δ sweep | `65 cases`, a `NOTE D4` line, `rho_delta` still ≈ −0.2 on PinShapes | if D4 is still skipped, the picker did not take effect |
| M4 | 20 rows, ~17 with `pin_isolated=True` | if **all** rows say FAILED, `PIN_R_FRAC` / junction detection regressed |
| O26 | all four `IDENTICAL` or all four `differ` | a **mix** would mean something stranger than a copy |

⚠️ `run_damage_lumi.sh` now exits **4** if nothing was produced and **5** if any case
failed, so a silent empty success is no longer possible. Exit 5 with `ok=82 failed=2`
is a *partial* success, not a crash — check the count before assuming the worst.
