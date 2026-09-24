---
marp: true
theme: default
size: 16:9
paginate: true
footer: 'Grid-projected particle tracking · A. R. Hashemi · 2026-09-09'
style: |
  /* Tightened for a table-and-figure-heavy deck.  Marp default
     (1280x720) does NOT auto-shrink content — slides overflow
     silently.  Base sizes below keep a full-width table or a
     480px-tall figure on one slide without per-slide overrides.
     Use `<!-- _class: tight -->` for the densest table slides. */
  section {
    font-size: 22px;
    padding: 40px 55px 55px 55px;
    justify-content: flex-start;
    line-height: 1.35;
  }
  section h1 { font-size: 34px; margin: 0 0 12px 0; }
  section h2 { font-size: 26px; margin: 0 0 10px 0; }
  section h3 { font-size: 20px; margin: 0 0 8px 0; }
  section p, section li { font-size: 20px; margin: 3px 0; }
  section table { font-size: 18px; margin: 6px 0; }
  section th, section td { padding: 4px 8px; }
  section img { max-height: 470px; display: block; margin: 6px auto; }
  section pre, section code { font-size: 16px; }

  section.lead h1 { font-size: 44px; }
  section.lead h3 { color: #666; font-weight: normal; }
  section.lead p { color: #888; font-size: 18px; }

  /* Densest tables: 20-case per-case listing, variant matrices */
  section.tight { font-size: 19px; padding-top: 30px; }
  section.tight h2 { font-size: 22px; }
  section.tight p, section.tight li { font-size: 17px; margin: 2px 0; }
  section.tight table { font-size: 14px; }
  section.tight th, section.tight td { padding: 2px 7px; }

  /* Figure + commentary side by side */
  .cols { display: grid; grid-template-columns: 1.5fr 1fr; gap: 20px;
          align-items: start; }
  .cols img { max-height: 500px; }

  /* Two figures side by side */
  .duo { display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
         align-items: start; }
  .duo img { max-height: 430px; }

  /* Placeholder box for figures not yet produced */
  .todo {
    border: 3px dashed #c44;
    background: #fff5f5;
    color: #a33;
    padding: 26px 30px;
    margin: 10px auto;
    font-size: 19px;
    line-height: 1.5;
  }
  .todo b { color: #822; }

  .win { background: #e9f0ea; font-weight: bold; }
  .bad { color: #a8503c; }
---

<!-- _class: lead -->
<!-- _paginate: skip -->

# Cartesian grids replace mesh search in FSW particle tracking

### 20-case cohort · 27 grid variants · 42–49× speedup

A. R. Hashemi — 2026-09-08

Source reports: `rom_pt_step5_step6_grid_report.md`,
`rom_pt_step2_step3_report.md`, `rom_pt_smoothness_divfree_report.md`

---

<!-- _class: tight -->

## Headline

| metric | value | note |
|---|---|---|
| **Speedup vs mesh + HCT-3D** | **42 – 49×** | 1463 s → ~35 s per case, same GPU |
| **Cohort-mean rms, mid-run** (t=3.6 s) | **5.62 mm = 0.56 D**$_{pin}$ | 20 cases, 360 k particles each |
| **Cohort-mean rms, final** (t=7.5 s) | **7.79 mm = 0.78 D**$_{pin}$ | survivorship-filtered — read with slide C5 |
| **Winner variant** | **`4lvl_hct`** | 3.55 M cells (3 blocks), HCT-3D Stage 1, trilinear Stage 2 |
| Grid variants tested | 27 | 24 on 4-case screen + 3 follow-ups on all 20 |
| Dominant design lever | **Stage-1 HCT-3D** | mean error ratio ×0.42 vs raw P1 |

**Two tracks in this work:**

- **Track A — ROM velocity surrogate: REOPENED** — was judged inaccurate on a defective basis; corrected basis cuts tracking error x0.34.<!-- -  Slides A1–A7. -->
- **Track B — grid-projected full-order tracking:**<!-- -  Slides B1–B8, C1–C6. -->

---

## The problem and the pivot

**Original goal.** A reduced-order model of the FSW velocity field, so parameter
studies would not need a full-order solve per case.

**Why it mattered.** Each FOM PT run is ~24 min at 360 k particles × 2000 steps ×
one case. Across 20 cases that is many GPU-hours per parameter study.

**What happened.** ROM velocity reconstructs well (4.04 % Eulerian) but fails as a
*tracking input*. See Track A.

**The pivot.** Keep the full-order velocity field. Kill the *element search* instead.

> Particle tracking on an unstructured tet mesh spends most of its time locating
> which element contains each particle. Project the velocity once onto a Cartesian
> grid and that search becomes arithmetic on the particle's coordinates.
> **The tracker stays full-order RK4** — no model reduction, no accuracy claim
> to defend beyond the projection itself.

---

<!-- _class: tight -->

## How to read the error numbers

Every error in this deck is a **position discrepancy against the mesh + HCT-3D
tracker**, reported two ways:

| scale | definition | why |
|---|---|---|
| **mm** | `\|x_grid − x_mesh\|`, rms over particles | the raw measurement |
| **D**$_{pin}$ | that, ÷ **10 mm** (pin diameter, r = 5 mm) | fixed, process-relevant |
| **% travel** | that, ÷ `\|x_mesh − x_seed\|` per particle | used on the xy maps (C2–C4) |

**Why the pin diameter.** It is the length scale the mixing physics is built
around — the tool feature that drives the flow — so "0.56 **D**$_{pin}$" answers
*how big is the error next to the thing that matters*. It is a single fixed
constant, so unlike a per-particle normaliser it adds no noise and stays
comparable across cases and timesteps.

**Reference points.** Domain bbox 45 × 30 × 4.5 mm (diagonal 54.3 mm) ·
plate thickness 4.5 mm · finest tet edge 0.14 mm · seeding annulus
r = 13.2 – 28.1 mm · closest approach to the axis at step 2000, r ≈ 3.7 mm.

> A useful anchor: **1.0 D**$_{pin}$ means a particle ends up one full pin
> diameter away from where the mesh tracker put it. The winner sits at
> **0.56** mid-run and **0.78** at final step.

---

<!-- _class: lead -->

# Track A — ROM velocity surrogate

### Why it was closed

---

<!-- _class: tight -->

## A1 · The reconstruction itself was never the problem

After fixing a loader bug (component/mode axes swapped in the POD basis),
20-case mean Eulerian error: **4.04 % ± 1.08 %**.

| case | centered rel_rms | case | centered rel_rms | case | centered rel_rms | case | centered rel_rms |
|---|---|---|---|---|---|---|---|
| 000 | 6.40 % | 005 | 2.82 % | 010 | 2.85 % | 015 | 3.15 % |
| 001 | 3.75 % | 006 | 4.80 % | 011 | 5.23 % | 016 | 2.75 % |
| 002 | 3.52 % | 007 | 3.30 % | 012 | 4.17 % | 017 | 4.17 % |
| 003 | **2.62 %** | 008 | 4.83 % | 013 | 4.05 % | 018 | 3.86 % |
| 004 | 3.61 % | 009 | 3.40 % | 014 | 5.61 % | 019 | 5.96 % |

| formula | mean rel_rms | std |
|---|---|---|
| mean-only | 30.90 % | 19.00 % |
| **centered** (FEMUSS convention) | **4.04 %** | **1.08 %** |
| c_over_sig | 30.56 % | 18.87 % |

Best case 003 at 2.62 %, worst 000 at 6.40 %.

> **But 4.04 % is the *shipped basis*, not a POD limit.** Our own POD from
> the same 20 snapshots reaches **0.52 %** at K=3 (0.66 % LOOCV) — see A6.

---

<!-- _class: tight -->

## A2 · Failure 1 — Eulerian accuracy does not survive integration

<div class="cols">
<div>

![Eulerian vs Lagrangian amplification](../paper_figs/rom_pt_step2_step3/step950/fig2_step950_eulerian_vs_lagrangian.png)

*Figure = **old** values only.*

</div>
<div>

Step 950, 360 k particles. **Old = shipped basis** (the figure);
**new = our POD, K=3** (2026-09-09 rerun).

| case | Eulerian old→new | Lagrangian old→new | amplif. old→new |
|---|---|---|---|
| 000 | 6.40 → **1.22 %** | 12.2 → **7.5 %** | 1.90 → **6.18×** |
| 001 | 3.75 → **1.02 %** | 11.7 → **4.5 %** | 3.12 → **4.42×** |
| 003 | **2.62 → 0.50 %** | **17.7 → 4.7 %** | 6.74 → **9.33×** |
| 004 | 3.61 → **0.39 %** | 13.2 → **2.6 %** | 3.64 → **6.62×** |

**Both columns improved 3–9×** — but Eulerian improved *more* than
Lagrangian, so **every amplification factor got worse**.

**Case 003 is still the killer:** best-reconstructing case, worst
Lagrangian gap — under both bases.

</div>
</div>

**A small Eulerian `rel_rms` is not a sufficient acceptance criterion for
tracking — and fixing the basis made that *more* true, not less.**

---

## A3 · Failure 2 — the ROM under-mixes

<div class="cols">
<div>

![Trapped particles](../paper_figs/rom_pt_step2_step3/step950/fig7_step950_trapped_particles.png)

</div>
<div>

Trapped-particle count as a **divergence-free proxy**: for an incompressible
advective field, no particle seeded far upstream should sit inside the near-pin
cylinder at a late timestep.

| case | FOM trapped | ROM trapped |
|---|---|---|
| 000 | 46.3 % (166,815) | 30.1 % (108,387) |
| 001 | 4.9 % (17,470) | 0.3 % (1,110) |
| 003 | 4.0 % (14,357) | 0.4 % (1,442) |
| 004 | 13.8 % (49,698) | 5.3 % (18,956) |

The ROM traps **fewer** particles than the FOM it was trained on.

</div>
</div>

**This is not an improvement.** The low-rank truncation damped the near-pin shear
structure that does the trapping. Under-mixing is precisely the failure mode that
invalidates a downstream FSW mixing study.

---

## A4 · Failure 2b — HCT-3D does not rescue it

<div class="duo">
<div>

![HCT gap closure](../paper_figs/rom_pt_step2_step3/step950/fig4_step950_hct_gap_closure.png)

**Δ rel_rms (HCT-on − HCT-off): −0.93 to +0.42 %pt** across all cases.

</div>
<div>

![HCT effect on trapping](../paper_figs/rom_pt_step2_step3/step950/fig7b_step950_trapping_hct_effect.png)

**Δ trapping: −0.00 to +0.96 %pt.** HCT typically makes trapping slightly *worse*.

</div>
</div>

HCT-3D is a real tracking perturbation (3–18 % near-pin on FOM, 3–44 % on ROM) but
it **neither compensates for nor masks** the ROM's Eulerian residual — the two
effects live on different scales.

---

<!-- _class: tight -->

## A5 · Failure 3 — 20 samples cannot support the regression

**Density-surrogate side.** Split between the two error sources localises the
bottleneck:

| error source | value | what it measures |
|---|---|---|
| projection error | ~12 % | basis expressiveness — can POD represent the field at all |
| **LOOCV error** | **~28 %** | true generalization — POD **+ regression** together |

- 7–10 modes retained for 90 % energy, from **20 samples** → only **10–13 effective
  DOF** for the regression step.
- `rbf`, `gp`, `poly` regressors all converge to **the same floor** — set by sample
  count, not by regressor choice.

**External review (3 rounds) confirmed the diagnosis** and flagged one overreach:
the mechanistic story we attached to it (boundary extrapolation driven by manifold
curvature, r = −0.74 Menger) rests on correlations from ~20 points and is **not
established**. The simpler "too few samples, period" hypothesis fits the same
evidence.

---

<!-- _class: tight -->

## A6 · Track A verdict — **reopened 2026-09-09**

> ### The ROM track was closed on a defective basis.

Every Track A number above used the **shipped FEMUSS basis**, which
reconstructs this cohort at **4.04 % L2**. A POD built from the *same 20
snapshots* reaches **0.52 % with the same 3 modes** — the shipped modes
simply do not span this data (its sigmas 101.30/6.09/3.68 vs 111.52/5.65/4.06
recomputed here). Not truncation, not precision, not Fortran-vs-Python —
all checked and ruled out.

| | shipped basis | our POD, K=3 |
|---|---|---|
| Eulerian L2 (cohort mean) | 4.04 % | **0.52 %** (0.66 % LOOCV) |
| ROM–FOM tracking, step 950 | 6.79 mm | **2.32 mm** (×0.34) |
| ROM–FOM tracking, step 2000 | 19.79 mm | **4.22 mm** (×0.21) |

5 of 20 cases rerun. Details: `rom_pt_ourpod_rerun.md`.

---

<!-- _class: tight -->

## A7 · What the rerun changes — and what it doesn't

**Improvement grows with integration time** — ×0.49 at step 500, ×0.34 at
950, ×0.21 at 2000. A cleaner field doesn't just shift error down; it slows
its accumulation along the trajectory.

**At 2.32 mm the ROM path now beats the grid path** (`4lvl_hct`, 5.62 mm) on
these five cases — inverting the comparison that justified closing Track A.

**The §2/§3 headline survives and is *strengthened*.** With a 3–9× better
Eulerian field, Lagrangian error fell only ~3×, so **every amplification
factor rose** (case 003: 6.74× → 9.33×; full table on slide A2). A small
Eulerian residual is an even *worse* acceptance criterion than we thought.

**Still open — do not re-close either way yet:** 15 of 20 cases unrun, and
the **§4 under-mixing analysis** (the *physical* argument for closing the
track) has **not** been rerun on the corrected field.

---

<!-- _class: lead -->

# Track B — Grid-projected full-order tracking

### The delivered result

---

<!-- _class: tight -->

## B1 · Two interpolation stages, chosen independently

| | **Stage 1 — build time** | **Stage 2 — run time** |
|---|---|---|
| what | mesh → grid nodes | grid nodes → particle |
| when | once per case | every RK4 substep |
| options | `p1_raw` — barycentric in the containing tet (C⁰)<br>`hct_cubic` — **HCT-3D**, Alfeld-split Bernstein cubic (C¹) | `trilinear` — 8-point (C⁰)<br>`tricubic` — Catmull-Rom, 4×4×4 stencil (C¹) |
| cost | HCT needs recovered nodal gradients; ~240 MB cache/case, ~15 s CPU | tricubic ≈ 8× lookup cost of trilinear |

**Reference for every comparison: the mesh + HCT-3D tracker.** Particles are seeded
from that run's own step-0 slice, so both sides start on **exactly the same 360 k
particles**. Reported numbers are therefore **grid-discretisation error alone** —
the ROM never enters Track B; both sides use FOM velocity.

**Keeping the two stages separate turned out to matter more than anything else in
this study.** → slide 12.

---

<!-- _class: tight -->

## B2 · The six grid families

All cover the same bbox `[-0.015, 0.030] × [-0.015, 0.015] × [-0.0045, 0]` m.

| suffix | topology | cells | finest Δx (m) |
|---|---|---|---|
| `uniform` | 90×60×9 single block | 48 600 | 5.0e-4 |
| `2lvl` | base + 1 nested block, r < 16 mm | 437 400 | 1.78e-4 |
| **`4lvl`** | base + 2 nested blocks, r < 16, r < 8 mm | **3 547 800** | 4.44e-5 |
| `malmo` | 5 blocks, one per MALMO octree depth, 1 cell/cube edge | 79 792 | 7.0e-5 (z) |
| `malmo3` | same 5 blocks, 3 cells/cube edge | 2 133 792 | 2.3e-5 (z) |
| `uniform_half` | single 640×427×64 block at ½ finest tet edge | 17 489 920 | 7.0e-5 |

**Finest tet edge** (reference for `uniform_half`): min 1.406e-4, median 1.989e-4,
p95 3.977e-4 m — identical across the cohort (shared mesh).

**MALMO** = mesh-aligned Morton octree; extracts one nested Cartesian block per
octree depth level, so refinement follows the *actual mesh* rather than a
hand-picked radial rule. Runtime queries blocks finest-first.

---

## B2b · What the grid families actually look like

![Grid schematics](../paper_figs/rom_pt_step5_step6_all20/figG_grid_schematic.png)

Drawn to scale from each module's baked `BLOCK_META` table (cell lines decimated
for legibility; labelled Δx is the true cell size). **Left:** the winner's three
nested blocks put 44 µm cells on the pin and shear ring. **Middle:** `uniform` is
one 500 µm block everywhere. **Right:** MALMO's five blocks follow the octree
depth levels, but its finest in-plane cell is 699 µm — **coarser than the 140 µm
finest tet edge**, which is why it undersamples the smooth HCT field.

---

<!-- _class: tight -->

## B3 · Result 1 — Stage 1 (HCT-3D) is the dominant lever

**Ratio of HCT-3D rms to raw-P1 rms** — smaller = bigger win. Trilinear Stage 2, step 950.

| grid family | case 000 | case 001 | case 003 | case 004 |
|---|---|---|---|---|
| uniform | ×0.81 | ×0.35 | ×0.37 | ×0.46 |
| 2lvl | ×0.62 | ×0.25 | ×0.26 | ×0.32 |
| **4lvl** | ×0.61 | **×0.24** | ×0.25 | ×0.31 |
| malmo | ×0.94 | ×0.42 | ×0.44 | ×0.57 |
| malmo3 | ×0.85 | ×0.32 | ×0.34 | ×0.44 |
| uniform_half | ×0.59 | **×0.23** | **×0.24** | **×0.30** |

- **Every family × every case improves under HCT.** Mean ratio across all 48
  (family × case × Stage-2) pairs: **×0.42** — HCT roughly *halves* grid PT error,
  and cuts it by up to 4× on the block-refined families.
- Once Stage 1 is HCT, **all six families collapse into a ~1.4× band**
  (5.0e-3 – 7.2e-3 m). **The floor is set by the projection, not the grid.**
- **Zero runtime cost** — the HCT-projected module loads a `.npz` of identical
  shape. The upgrade happens entirely at build time.

---

<!-- _class: tight -->

## B4 · The trap this exposed — a near-miss recommendation

**Under raw P1** (Part 1), MALMO won on **every** case, in both interp orders.
All values **D**$_{pin}$ (rms, step 950):

| case | uniform | 2lvl | 4lvl | **malmo** | malmo3 | uniform_half |
|---|---|---|---|---|---|---|
| 000 | 0.89 | 1.10 | 1.12 | **0.75** | 0.83 | 1.18 |
| 001 | 1.58 | 2.10 | 2.15 | **1.36** | 1.64 | 2.29 |
| 003 | 1.62 | 2.17 | 2.22 | **1.37** | 1.70 | 2.35 |
| 004 | 1.19 | 1.60 | 1.64 | **0.95** | 1.19 | 1.74 |

**Under HCT** (Part 2), the leaderboard **flips** — MALMO drops to 5th–8th.
Same scale, and note the whole table drops below **0.72 D**$_{pin}$:

| case | uniform_hct | 2lvl_hct | **4lvl_hct** | malmo_hct | malmo3_hct | uniform_half_hct |
|---|---|---|---|---|---|---|
| 000 | 0.71 | 0.69 | **0.68** | 0.71 | 0.70 | 0.70 |
| 001 | 0.55 | 0.52 | **0.52** | 0.57 | 0.53 | 0.52 |
| 003 | 0.60 | 0.56 | **0.55** | 0.60 | 0.58 | 0.56 |
| 004 | 0.55 | 0.51 | **0.51** | 0.54 | 0.52 | 0.52 |

> **MALMO was never better — its mesh-aligned block boundaries were *masking a P1
> defect*.** Once the projected field is C¹ the defect is gone, the mesh-alignment
> premium disappears, and MALMO's coarse cells-per-edge becomes a liability.
> **Any grid-family conclusion drawn before Stage 1 was fixed would have been wrong.**

---

## B4b · Stage 1 in the trajectories — case 000

![P1 vs HCT, case 000](../paper_figs/rom_pt_step5_step6_all20/figP_p1_vs_hct_case000_4lvl.png)

**Identical grid geometry, identical integrator, identical particles — only the
mesh→grid projection differs.** Raw P1 (middle) scatters red-to-green error across
the whole cloud; HCT-3D (right) is almost entirely dark blue.
**rms 24.93 → 5.14 mm, i.e. 2.49 → 0.51 D**$_{pin}$ **(×0.21).**

---

## B4c · Same comparison, case 003

![P1 vs HCT, case 003](../paper_figs/rom_pt_step5_step6_all20/figP_p1_vs_hct_case003_4lvl.png)

The effect is **larger** on case 003: **rms 51.59 → 7.94 mm = 5.16 → 0.79 D**$_{pin}$ **(×0.15)**.

*Note on the radius panel:* case 003 is a **flusher** (≈5 k particles alive at
step 2000), so the per-bin medians are noisy and HCT crosses above P1 in two
sparse bins. The aggregate rms and the CDF — both computed on all 360 k
particles — are the reliable read.
The error-vs-radius panel shows why the aggregate number understates it — under
raw P1 the error grows without bound in the outer domain, while HCT stays flat
across the whole radius range. The CDF makes the same point: P1's tail runs past
50 mm (5 D$_{pin}$), HCT's is fully closed by ~15 mm (1.5 D$_{pin}$).

---

<!-- _class: tight -->

## B5 · Result 2 — the 20-case winner

Top-3 from the 4-case screen + 3 hypothesis-driven follow-ups, all 20 cases.
**D**$_{pin}$ **= 10 mm** (pin diameter) — near-pin, outer and Δ columns are in
**D**$_{pin}$. Paired Wilcoxon signed-rank vs winner, n = 20.

| variant | s950 mm | s950 **D**$_{pin}$ | s2000 mm | s2000 **D**$_{pin}$ | near-pin | outer | Δ vs win | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|
| **`4lvl_hct`** | **5.62** | **0.562** | **7.79** | **0.779** | 0.964 | 0.762 | — | — |
| `5lvl_hct` | 5.75 | 0.575 | 7.87 | 0.787 | 0.969 | 0.773 | +0.009 | 0.030 |
| `malmo6_hct` | 5.81 | 0.581 | 7.92 | 0.792 | 0.955 | 0.790 | +0.014 | 0.017 |
| `4lvl_r22_hct` | 5.76 | 0.576 | 7.96 | 0.796 | 0.965 | 0.792 | +0.018 | 0.007 |
| `4lvl_hct_tricubic` | 5.57 | 0.557 | 8.00 | 0.800 | 0.975 | 0.793 | +0.021 | 0.053 |
| `2lvl_hct_tricubic` | 5.56 | 0.556 | 8.60 | 0.860 | 0.959 | 0.936 | +0.081 | 0.003 |

**Two things to read off this table:**

1. **Tricubic Stage 2 reverses sign over the run** — marginally *better* at mid-run
   (5.57 vs 5.62), clearly *worse* by final step (8.00 vs 7.79), as C¹ Catmull-Rom
   overshoot compounds along the trajectory. The 4-case screen missed this.
2. **All 3 new variants are significantly worse** — including one built specifically
   to refine the shear band where the error lives, and a 17 M-cell MALMO grid.

---

## B6 · Per-case cohort results

<div class="duo">
<div>

![Cohort rms step 950](../paper_figs/rom_pt_step5_step6_all20/fig1_cohort_rms_step950.png)

**Mid-run (t = 3.6 s).** CV = 15 %; no case more than 1.5× the mean.

</div>
<div>

![Cohort rms step 2000](../paper_figs/rom_pt_step5_step6_all20/fig1_cohort_rms_step2000.png)

**Final (t = 7.5 s).** CV = 24 %; spread widens, driven by survivorship.

</div>
</div>

Per-case bars, 3 variants side by side, cases sorted by mean error; dashed line =
cohort mean. The three top variants track each other closely on every case —
**the differences between them are within cohort noise.**

---

## B7 · Cohort distribution and no-dominant-winner

<div class="cols">
<div>

![Cohort distribution](../paper_figs/rom_pt_step5_step6_all20/fig2_cohort_distribution.png)

</div>
<div>

**Per-case winners are scattered** — every variant wins some cases:

| variant | cases won at s2000 |
|---|---|
| `4lvl_hct` | 000, 001, 009, 011 |
| `5lvl_hct` | 003, 004, 010, 012, 017 |
| `2lvl_hct_tricubic` | 002, 015, 018, 019 |
| `malmo6_hct` | 006, 007, 013, 014 |
| `4lvl_hct_tricubic` | 008, 016 |
| `4lvl_r22_hct` | 005 |

`4lvl_hct` keeps the title because it is **never far from the best on any case**,
whereas malmo6/5lvl swing between best and near-worst.

</div>
</div>

---

<!-- _class: tight -->

## B8 · Result 3 — throughput: cell count is not a knob

**Mesh reference:** ~1463 s per case (~24 min), ~490 k particle-steps/s.

| variant | mean wall (s) | speedup vs mesh | cells |
|---|---|---|---|
| `2lvl_hct_tricubic` | 28.5 | 51× | 437 k |
| `4lvl_hct_tricubic` | 31.6 | 46× | 3.55 M |
| `4lvl_r22_hct` | 34.9 | 42× | 3.55 M |
| **`4lvl_hct`** | **35.1** | **42×** | **3.55 M** |
| `malmo6_hct` | 35.7 | 41× | **~17 M** |
| `5lvl_hct` | 36.6 | 40× | 4.86 M |

- Across the full 24-variant screen, **every grid variant is 38× – 82× faster**.
- **The 17 M-cell `malmo6_hct` runs at the same speed as the 49 k-cell `uniform_hct`** —
  JAX kernel launch + gather dominate arithmetic at these problem sizes.

> **Consequence: choose the variant on accuracy alone.** There is no throughput
> argument for a smaller grid, and no accuracy argument for a larger one.

---

<!-- _class: tight -->

## B8b · Mesh vs grid, head to head

![Timing comparison](../paper_figs/rom_pt_step5_step6_all20/figT_timing_mesh_vs_grid.png)

**Left:** mesh at **1,463 s** vs accuracy winner (**36 s, 40×**) and throughput
champion (**18 s, 82×**) — log scale, **1.6 orders of magnitude** apart.
**Right:** all 24 variants clear the mesh line by 38× or more; Stage-1 choice
(blue vs orange) has **no systematic effect on runtime**.

---

<!-- _class: lead -->

# Where the residual error lives

### The cohort mean hides a strongly non-uniform distribution

---

## C1 · Near-pin vs outer, and the spatial pattern

<div class="duo">
<div>

![Near-pin vs outer](../paper_figs/rom_pt_step5_step6_all20/fig5_near_pin_vs_outer.png)

Near-pin/outer ratio **compresses ×1.7 → ×1.27** from s950 to s2000
(0.82 → 0.96 D$_{pin}$ near-pin; 0.47 → 0.76 outer): error starts
concentrated at the shear boundary, then the bulk catches up.

</div>
<div>

![Spatial pattern](../paper_figs/rom_pt_step5_step6_all20/fig6_spatial_pattern.png)

(r, z) heatmap of median |err|, shared log scale. **The pin-face shear band at
r ≈ 21–22 mm dominates** — *not* the pin core.

</div>
</div>

---

<!-- _class: tight -->

## C2 · The pin core is the *easy* part

`4lvl_hct` at step 2000, median across all 20 cases. The **% travel** column is
the per-particle relative error; the **D**$_{pin}$ and **mm** columns are the
absolute median error in each region, measured over all 20 cases.

| region | median err / travel | median **D**$_{pin}$ | median mm | mm range |
|---|---|---|---|---|
| **shear ring** (4 ≤ r ≤ 7 mm) | **~34 %** | **0.74** | 7.35 | 6.3 – 8.9 |
| downstream wake (x > 8, \|y\| < 5 mm) | ~15 % | 0.41 | 4.14 | 3.6 – 4.8 |
| outer bulk (r > 15 mm) | ~13 % | 0.40 | 3.97 | 3.4 – 4.7 |

**Shear-ring : outer ratio ≈ 3× in relative terms** (34 % vs 13 % of travel) on
every case, regardless of case difficulty or survivorship. In **absolute** terms
the ratio is **1.85×** (0.74 vs 0.40 D$_{pin}$) — the ring particles travel less
far, so the same absolute error is a much larger fraction of their journey.

**Why the core is easy:** the pin's material rotation is smooth, rigid-body-like
motion, and grids represent it well. The damage is at the **shear boundary** where
pin-driven flow meets bulk drift and velocity gradients are steepest.

p95 of the ring can spike > 300 % on flusher cases — those are the few thousand
particles that spent the whole run cycling through the shear layer.

---

## C3 · xy error maps — the pattern is cohort-universal

![xy error map, 4lvl_hct, relative](../paper_figs/rom_pt_step5_step6_all20/xy_error_4lvl_hct_step2000_pct.png)

All 20 cases, mesh particle positions at step 2000 projected to (x, y), coloured by
relative grid-vs-mesh disagreement. **In every case a thin red-orange annular ring
sits at the pin's outer edge, followed by a mostly-blue downstream wake.**
Red dotted line = nominal outlet at x = 30 mm; particles past it are still tracked
via ballistic extension.

---

## C4 · It is a flow property, not a grid property

<div class="duo">
<div>

![4lvl_hct](../paper_figs/rom_pt_step5_step6_all20/xy_error_4lvl_hct_step2000_pct.png)

**`4lvl_hct`** — 3.55 M cells

</div>
<div>

![malmo6_hct](../paper_figs/rom_pt_step5_step6_all20/xy_error_malmo6_hct_step2000_pct.png)

**`malmo6_hct`** — 17.0 M cells (**4.8× more**)

</div>
</div>

> **Visually indistinguishable.** Same ring, same wake, same per-case shape;
> regional medians agree within 1–2 %pt. The 3× shear-ring amplification is what
> **mesh + HCT-3D itself** says about the shear layer — any reasonable grid
> representation converges to the same trajectory divergence.
> **Adding cells does not help. Grid-topology optimisation is saturated.**

---

<!-- _class: tight -->

## C5 · Reading late-run statistics correctly

<div class="cols">
<div>

![Survivorship](../paper_figs/rom_pt_step5_step6_all20/fig4_survivorship.png)

</div>
<div>

`both_alive` fraction:

- **step 950:** mean **97.3 %** (median 100 %, min 88 %) — metric ≈ full cohort.
- **step 2000:** mean **23.6 %** (median 4.6 %, range 1.4 – 71.5 %).

**Cases split into two populations:**

| | surviving @ s2000 | growth factor |
|---|---|---|
| **keepers** | > 100 k | **≤ 1.12** (5 cases *shrink*) |
| **flushers** | < 20 k | **1.36 – 2.26** |

</div>
</div>

> **The shrinkage is survivorship bias, not improvement.** Particles that disagreed
> early were pushed onto trajectories that escaped, dropping them from the
> `both_alive` count. Per-particle tracker error is **monotonically non-decreasing**.
> Use **step 950 as the headline metric**; step 2000 only as a bound on keeper cases.

---

## C6 · Growth scatter — the bimodal split, visualised

<div class="cols">
<div>

![Growth scatter](../paper_figs/rom_pt_step5_step6_all20/fig3_growth_scatter.png)

</div>
<div>

rms(950) vs rms(2000), one point per (case, variant), y = x reference.

**Keeper cases** (000, 002, 008, 011, 014, 018, 019) sit **on or below** the
diagonal.

**Flusher cases** (001, 003, 004, 005, 007, 010, 012, 013, 016, 017) sit well
**above** it.

The two clusters are separated by survivorship, **not** by tracker quality —
which is why the cohort mean at s2000 needs the caveat on the previous slide.

</div>
</div>

---

<!-- _class: lead -->

# Still to add

### One ParaView snapshot — everything else is now plotted

---

## P1 · 3-D structure of the shear ring

<div class="todo">

**PLACEHOLDER — ParaView snapshot needed** (the only one left)

**Content:** per-particle displacement VTU coloured by |x_grid − x_mesh|, clipped to
a wedge or z-slice, showing the shear-ring concentration **in 3-D**.

**Source data — 240 VTUs already on disk, ~14 MB each, no new runs needed:**
`/scratch/shared/ROM/rom_out/fom_grid_pt_all20/pt_error_case<c>_grid_4lvl_hct_step<S>.vtu`
— every (case x variant x step) combination exists.

**Point to make:** slides C2–C4 show the ring in 2-D projection only. A wedge clip
would confirm it is a true annulus through the plate thickness, not an artefact of
collapsing z.

**Everything else that was pending is now generated** from the run outputs by
[scripts/plot_deck_supplementary_figs.py](../scripts/plot_deck_supplementary_figs.py):
grid schematics (B2b), mesh-vs-grid timing (B8b), and the P1-vs-HCT trajectory
comparison (B4b, B4c).

</div>

---

<!-- _class: lead -->

# Status and open questions

---

<!-- _class: tight -->

## S1 · Coverage matrix

| | 4-case cohort | full 20-case cohort |
|---|---|---|
| FOM tracker on mesh | **done** | **done** |
| ROM tracker on mesh | **done** | 4 of 20 |
| **FOM tracker on grid** | **done** (24 variants) | **done** (6 variants) |
| ROM tracker on grid | *not run* | *not run* |
| Wall-time comparison | **done** | **done** |
| Per-r-bin breakdown | **done** | **done** |
| Trapping / escape agreement | **done** | **done** |
| Mixing diagnostics (residence time, pairwise sep.) | mesh path only | *not run* |

**Two real gaps**, stated plainly:

1. **The grid experiment ran entirely on FOM velocity.** What is demonstrated is
   that a grid replaces the mesh *for the full-order tracker*. Given Track A's
   outcome this is now low priority — but it is why the roadmap's formal §5
   acceptance criterion is not formally closed.
2. **Mixing diagnostics were never ported to the grid path.** These are the
   diagnostics that would answer the acceptance question on the next slide, so
   **this is the more useful gap to close.**

---

## S2 · The open question for the team

> ### If a downstream analysis is dominated by shear-layer physics — and FSW mixing is —
>
> then a **7.8 mm = 0.78 D**$_{pin}$ **cohort-mean rms with 3× ring concentration** means:
>
> - shear-zone trajectories carry **~34 % relative error** (0.74 D$_{pin}$ absolute)
> - outer-domain trajectories carry **~13 %** (0.40 D$_{pin}$ absolute)

**Is that acceptable?**

This is a **use-case decision, not a tracker decision.** It is the one thing this
work cannot settle on its own — and the reason gap 2 above (mixing diagnostics on
the grid path) is worth closing before anyone commits.

---

<!-- _class: tight -->

## S3 · Recommended recipe

1. **Stage 1 = HCT-3D** (`--mesh-projection hct_cubic`) — **non-negotiable**. Delivers
   a 2–4× rms reduction on its own, larger than any grid-family choice, at zero
   runtime cost.
2. **Grid family = `4lvl`** — base + two nested blocks at r < 16 mm and r < 8 mm,
   3.55 M cells. Never far from best on any case (0.56 D$_{pin}$ mid-run).
3. **Stage 2 = trilinear** — tricubic helps ~1 % at mid-run but costs 3 % by final
   step as overshoot compounds.
4. **Do not use MALMO under HCT** — it looked best under P1 only because it was
   masking a P1 defect.

**One-off cost per case:** HCT cache `hct_cache_ts119.npz`, ~240 MB, ~15 s CPU,
reused across every grid variant of that case. Grid module build: 10 s – 20 min
depending on cell count.

```bash
python3 scripts/build_hier_grid_velocity_case.py \
    --grid-type 4lvl --mesh-projection hct_cubic --interp trilinear \
    --hct-cache /scratch/shared/ROM/FOM/cylindrical_<c>.gid/hct_cache_ts119.npz
```

---

<!-- _class: tight -->

## S4 · Where everything lives

**Reports** — `JAXTrace/docs/`
`rom_pt_step5_step6_grid_report.md` (grid experiment, 5 parts) ·
`rom_pt_step2_step3_report.md` (ROM vs FOM tracking) ·
`rom_pt_smoothness_divfree_report.md` (under-mixing) ·
`rom_pt_paused_state.md` (checkpoint / resume map) ·
`rom_reconstruction_findings.md` (Eulerian 20-case table)

**Figures** — `JAXTrace/paper_figs/`
`rom_pt_step5_step6_all20/` (6 cohort figs + 4 xy maps) ·
`rom_pt_step2_step3/step950/` (10 ROM-vs-FOM figs) ·
`rom_pt_step4/` (7 smoothness/mixing figs)

**Run data** — `/scratch/shared/ROM/`
`rom_out/fom_grid_pt_all20/` (120-row detail CSVs, 240 per-particle VTUs) ·
`FOM/cylindrical_<c>.gid/` (grid modules, HCT caches, all PT runs)

**Scripts** — `JAXTrace/scripts/`
`build_hier_grid_velocity_case.py` · `compare_grid_vs_mesh_pt_fom.py` ·
`plot_grid_pt_presentation.py` · `plot_final_step_xy_error_maps.py`

---

<!-- _class: lead -->

## Summary

### Track A — ROM velocity surrogate: **reopened 2026-09-09**
Closed on a defective basis. The shipped FEMUSS basis reconstructs at 4.04 %;
our own POD from the same snapshots reaches **0.52 %** at K=3, cutting ROM
tracking error from 6.79 to **2.32 mm** at step 950 (5/20 cases). The
Eulerian-does-not-predict-Lagrangian finding *survives and strengthens*
(amplification rose). §4 under-mixing not yet rerun.

### Track B — grid-projected full-order tracking: **delivered**
**`4lvl_hct` · 0.56 D**$_{pin}$ **mid-run · 0.78 D**$_{pin}$ **final · 42–49× faster · all 20 cases.**
(5.62 mm / 7.79 mm absolute.)
Stage-1 HCT-3D is the dominant lever (×0.42); grid topology is saturated.

### Open
Is ~20–30 % relative error in the shear zone acceptable for FSW mixing analysis?
