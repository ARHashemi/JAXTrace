# Phase 1 gate + the measurements still open — what to run, and why

**2026-09-25.** Answers three questions: what to submit on LUMI now, what else
needs measuring to close the open items, and whether the GPU logger is needed at
this step or later.

---

## 0. What changed, so the gate measures the real thing

⚠️ The Phase 1 accumulator originally went into
`jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py`. **The FSW cases never execute
that module.** The per-case `run_jaxtrace.sh` files call `run_tracking.py`, which
gets its kernel from `create_rk4_comparison` in `benchmark_femuss_comparison.py`
— a separate inlined copy (that file is labelled "the source of truth" at
`run_tracking.py:89`).

The accumulator now lives in that kernel, and `--damage` is a flag on
`run_tracking.py`. This is why the gate below runs the case's own runner rather
than a bespoke harness: an earlier standalone harness diverged from production
(wrong locator, no mesh-aligned octree, no `M_inv`/`p0`, no pending-entry state)
and produced a number describing a configuration that never runs.

---

## 1. The gate — submit this on LUMI

```bash
cd /projappl/project_465002752/hashemia
sbatch sbatch_damage_gate.sh          # ~25 min/arm, ~2 h total
```

⚠️ **First attempt (job 22346898) timed out** after 1.5 of 4 arms. Cause: C1's runner
loads a **full revolution — 166 PVTU slices** of a 125 GB case, and mesh load was
**6520 s of an 8433 s arm (77 %)** against 670 s of actual tracking.

The gate measures *per-step* kernel cost, which does not depend on slice count (the
kernel indexes one slice per step regardless). So the script now collapses the range
to a single slice by default (`VEL_RANGE="199 199"`) and requests 12 h. Set
`VEL_RANGE=""` to inherit the full revolution — correct if you are testing the
**time-dependent** path, but that is a different experiment with its own budget.

Optional overrides:

```bash
CASE_DIR=/scratch/project_465002752/lorenzgl/Cases/PinShapes/A-Tapered/A1.gid \
N_STEPS=400 sbatch sbatch_damage_gate.sh
```

### What it does

Copies the case's own `run_jaxtrace.sh` into
`/scratch/.../hashemia/damage/phase1_gate/runners/`, makes five hardcoded
assignments env-overridable, appends an `EXTRA_ARGS` hook, and runs the copy
twice — `--damage` being the only difference.

⚠️ **The colleague's files are read and never written.** `OUTPUT_TARGET` is
forced to `scratch`, because the stock runner defaults to `case`, which would
write into their case folder. Verified on LUMI: source md5 unchanged after
patching, `bash -n` clean, hook injected at line 370 — before `srun` and before
every conditional `ARGS+=`.

### Compatibility audit (asked for explicitly)

All 21 per-case runners on LUMI, all 50 distinct flags they pass:

| target | result |
|---|---|
| updated `JAXTrace/run_tracking.py` | ✅ every flag accepted |
| `JAXTrace_stable/run_tracking.py` | ✅ every flag accepted |

* 19 runners point at `JAXTrace_stable`, 2 at `/project/${PROJECT}/${USER}/JAXTrace`.
  `/project` is a valid symlink to `/projappl`, so both resolve.
* The 20 `--density-*` flags are **absent from `_stable`** but are inside
  `if [ "${DENSITY_ENABLE:-0}" = "1" ]`, and `DENSITY_ENABLE=0` in both runners
  that define them — so they are never passed. **Not a live incompatibility**,
  but it means `_stable` would break if anyone flips that to 1.
* `--env` is a singularity flag, not a `run_tracking.py` one.

⚠️ The two `${USER}`-based runners resolve to `hashemia` only when *you* run
them; they would break for the colleague.

⚠️ **Neither LUMI repo had `--damage` before today.** `JAXTrace` and
`JAXTrace_stable` also differ from each other. The gate script therefore sets
`JAXTRACE` explicitly to the repo that has the flag; it does not rely on the
runner's default.

### Result already measured on the workstation (RTX 5090, NVIDIA)

cylindrical_000, 100k particles, 300 steps. Seven runs, both flag orders:

| state | off | on | slowdown |
|---|---|---|---|
| fast (~85 s) | 84.4 s | 84.8 s | **+0.51 %** |
| slow (~162 s) | 162.2 s | 162.5 s | **+0.15 %** |
| earlier 20k-particle pair | 162.0 s | 162.5 s | **+0.31 %** |

Gate is <3 %: **PASS**, on the production kernel, three independent measurements
in agreement.

### ✅ And on LUMI / ROCm — job 22366672, 2026-09-26

| arm | `7_tracking` |
|---|---|
| off1 | 634.3 s |
| on1 | 645.9 s |
| off2 | 635.0 s |
| on2 | 645.9 s |

**slowdown +1.77 %, spread 0.1 % / 0.0 %, verdict PASS**, accumulator 100.0 % of
288,000 particles. `on1` and `on2` damage arrays are **bitwise identical** —
deterministic, corroborating the zero spread.

| platform | slowdown | gate |
|---|---|---|
| RTX 5090 (CUDA) | +0.15 … +0.51 % | ✅ |
| **MI250X (ROCm)** | **+1.77 %** | ✅ |

**Phase 1 is closed on both platforms.**

⚠️ **The damage VALUES from a gate run are not physics.** 400 steps = 0.145 s =
1.45 mm of advance, and **0.00 %** of particles reached the 7 mm tool (median final
radius 16.7 mm — they started 11–20 mm upstream). Reaching the tool needs ~1130
steps; traversing the stir zone ~5500; the case's production `N_STEPS=8000` is
correctly sized. A timing run's damage file is a by-product. See O22.

⚠️ **The workstation has two timing states** (~85 s and ~162 s for identical
work), independent of the damage flag — `nvidia-smi` reports **persistence mode
Disabled**. Comparing arms *across* that transition yields **−23.5 %**, i.e. a
fabricated speedup from adding work. So the gate runs **two alternating rounds**
and refuses to report a ratio when two same-flag runs disagree by more than 10 %.
Enabling persistence mode on the workstation would remove the effect.

Accumulator verified live, not merely fast:

```
[damage] edot field : (1, 140461)  0.5 MB  built in 0.5s   median 53  max 3.91e3 1/s
[damage] accumulated on 99,850/100,000 particles (99.8 %)
[damage] value      : median 0.2307  mean 11.02  max 2637
[damage] alive      : 100,000/100,000
```

⚠️ **A timing gate alone would pass identically with dead damage code**, so the
liveness check is part of the gate, and the script reports `accumulator_ok:
false` if nothing accumulates.

**100 % active / 0 lost** also confirms the production locator
(`initial_assignment_mesh_aligned_multi_local`) is sound — the earlier "L2 loses
21 % of interior points" claim was a harness artefact and has been retracted.

---

## 2. GPU logger — needed now, or later?

**Needed now, but only in the light form already built into the gate script.**

The gate's claim is "damage adds <3 % cost". Wall-clock alone cannot distinguish
*no extra work* from *extra work hidden behind a stall*. If the ON arm showed the
same time at higher memory traffic or lower occupancy, the ratio would be
misleading. So the gate samples the GPU every 2 s per arm and writes
`gpu_{off,on}_<jobid>.log` — `rocm-smi` on LUMI, `nvidia-smi` on the workstation,
whichever exists.

Observed on the workstation: **mean utilisation 94–96 %, max memory 3380 MB,
identical between arms.**

`jaxtrace/damage/gpu_monitor.py` (the Python context-manager version) is kept for
in-process use and works on both backends, but is **not required** for this gate —
the shell sampler is enough and does not perturb the run. Postpone the detailed
per-kernel profiling to Phase 3, when deposition adds real memory traffic.

---

## 3. The measurements still open, in dependency order

Only M3 is closed. The rest are what the remaining open items need.

| # | measurement | closes | cost | why it must happen |
|---|---|---|---|---|
| ~~M3~~ | ~~throughput gate~~ | ~~Phase 1~~ | done | **+0.31 %, PASS** |
| **M1** | δ (shear-layer width) for all 20 PinShapes cases | O16, O10 | ~20 PVTU reads, minutes | the δ result is cylindrical-cohort only; tapered/threaded/tilted pins may differ |
| **M2** | re-run the survey with a **δ-relative** window | **the n_selected confound**, O10 | one sweep | ρ(n_selected, ratio) = **+0.571, p = 0.008** across 20 cases. Until this is done the family separation is provisional |
| **M4** | pin geometry descriptors (flat count/depth, thread lead, flute count) from the STLs | makes any geometry claim modellable | parse 20 STLs | **the benchmark's only real independent variable, and it is recorded nowhere.** "Threads beat flats" is currently a statement about four file names |
| **M5** | Norton vs Sellars–Tegart per case on LUMI | robustness of the ordering | one sweep, both models | already implemented; confirms the ranking is not rheology-dependent |

⚠️ **M2 before Phase 3.** A susceptibility field built on a confounded window
inherits the confound.

⚠️ **M4 is the significant gap.** Without descriptors there is nothing to
regress against in Phase 4.

### Deliberately deferred

| item | why |
|---|---|
| grid projection (Stage 2) | the method is completed and validated end-to-end on the **FOM mesh** first — outer loop, per your correction |
| ROM evaluation (O5, Stage 3/4) | moved to the end of the last phase |
| O14 cohort divergence | LUMI is the reference; raise with colleagues |
| O4 CT ground truth | no data identified; blocks any *validation* claim, not development |

---

## 4. Suggested order

1. **Submit the gate on LUMI** (confirms AMD/ROCm parity with the NVIDIA result
   above). Minutes.
2. **M1 + M2 together** — one sweep produces δ per case and the re-windowed
   survey. This is the highest-value item: it decides whether the headline family
   separation survives.
3. **M4** — cheap, and nothing in Phase 4 is modellable without it.
4. **M5** — cheap robustness check.
5. Then Phase 3 (damage ODEs: Rice–Tracey `ln Φ`, Cockcroft–Latham) on
   configuration A, with the accumulator now proven on the production path.
