# Open questions — full detail, for reading and deciding

**2026-09-30.** Until now these lived as one-line rows in a table at line ~2200 of
`results/RESULTS_LOG.md`, which is where findings are logged chronologically. That
is fine for provenance and useless for deciding. This document gives each live item
the space to be understood: what was observed, what it does and does not affect, what
would settle it, and who can settle it.

**15 items are live** (O25, O26, O12, O30 and rotation closed; **O29, O31, O32, O33 opened**). They are grouped by what they block, because that is the
decision-relevant axis — not by number.

| group | items | one-line summary |
|---|---|---|
| **Blocks a validation claim** | O4 | no experimental ground truth exists |
| ~~Affects magnitudes~~ | ~~**O25**~~ | ✅ **SOLVED** — shear-vs-vonMises convention, factor 1.816, fixed in code |
| **Affects magnitudes, not rankings** | O7, O15 | pressure sensitivity; borrowed material tables |
| **Affects which numbers to quote** | **O27**, O22, O12, O13 | snapshot vs revolution, and fragile labels |
| **Affects the case inventory** | O14 | two diverging copies of the cohort (~~O26~~ answered) |
| **Infrastructure / housekeeping** | O8, O20, O21b | uncommitted work, a flaky GPU, an I/O cost |
| **Deferred by decision** | O5, O6 | ROM work, postponed deliberately |

> ⚠️ **O25 is now SOLVED** — the answer was in the colleague's thesis (the `.mat`
> tables are in shear measures, not von Mises). It is kept below in full because the
> reasoning matters: I had explicitly *ruled out* the correct answer, and the way
> that happened is worth not repeating. **O27** remains live.

---

## ~~O25~~ — SOLVED 2026-09-30: the tables are in SHEAR measures

**Closed.** The answer was in the colleague's thesis, not something to ask him.

### The answer

Venghaus, *Finite Elements in Analysis & Design* **224** (2023) 103986,
Publication 1, eq. (5):

```
s = 2 mu_eff e ,  mu_eff = mu0(T) * gamma^(m-1)
gamma = sqrt(2)*||e||        <- SHEAR equivalent strain rate
tau   = (sqrt2/2)*||s||      <- SHEAR equivalent stress = mu_eff*gamma
```

`mu0` **is** the `.mat` `Plastic_viscosity` table and `m` **is**
`Exponent_viscosity`. **The tables give τ(γ) in shear measures**, while our pipeline
is von Mises. Since γ = √3·ε̇_vM and σ_vM = √3·τ:

```
seq = sqrt(3) * VISCO(T) * (sqrt(3) * edot) ** EXPVI(T)
```

a total correction of **√3·(√3)^m = 1.816** at m = 0.086.

### Verified on 581,488 cells

| | ratio to the solver's own √(3J₂) |
|---|---|
| naive `VISCO·edot^EXPVI` | **0.5689** |
| thesis convention | **1.0346** |

Measured correction **1.8187** vs predicted **1.8163** — **0.13 %**. A 76 % error
becomes a 3.5 % residual, and the 0.13 % match is what makes this a confirmed
derivation rather than a fitted number.

### ⚠️ I had ruled this out. Both my arguments were wrong.

| my argument | the error |
|---|---|
| "1/0.569 = 1.7575 vs √3 = 1.7321, off by 1.47 %, outside the ±0.8 % spread" | compared against the **bare** √3; the full factor is √3·(√3)^m = 1.816 |
| "m ≈ 0.09 damps any rate redefinition to 1.05×" | true for the rate alone, but there are **two** conversions — rate *and* stress — each contributing √3. I never tested them together. |

> Correctly identifying the *category* ("a pure multiplicative convention factor")
> and then excluding the one candidate of exactly that form is worse than not
> diagnosing it at all: it sent the question to the FOM author when the answer was
> in his published paper.

### Fixed

`jaxtrace/damage/rheology.py::sigma_eq_norton` applies the conversion;
`shear_convention=False` reproduces the old numbers. All 5 analytic tests pass.

⚠️ **All previously reported absolute η and σ_eq values are low by 1.816×.** Ratios,
rankings, the tensile-flank result and everything built on them are unchanged — M5
independently confirms the ranking is rheology-insensitive (ρ = +0.9995).

⚠️ **Now relevant for Phase 3:** Cockcroft–Latham's `C = ∫⟨σ₁⟩dε̄` is *dimensional*,
so any calibrated critical value must use the corrected σ_eq.

---

## O27 — δ-window ratios come from a single snapshot

**Status: quantified. A decision about what to re-run, not a mystery.**

### What was observed

Two different things have been called "the growth ratio":

| | what it is | cost |
|---|---|---|
| **phase average** | η and ε̇ averaged over a **full tool revolution** (166 timesteps), then surveyed | 166 PVTU reads/case |
| **snapshot** | one single last timestep, surveyed directly | 1 PVTU read/case |

The published Stage-1 numbers are phase averages. The M1/M2 δ-window sweep used
snapshots, because comparing *window definitions* needs a cheap, internally
consistent measurement — and 64 cases × 166 timesteps was not affordable.

Comparing the two on the 19 PinShapes cases:

| | |
|---|---|
| median \|relative difference\| | **3.8 %** |
| A1 | +0.0 % |
| A3, D1 | +0.1 % |
| B-family | −2.5 … −10.5 % |
| C-family | +1.4 … +4.1 % |
| **A2** | **+25.6 %** |
| **D2old** | **+20.5 %** |
| **A4old** | **−21.3 %** |
| **A2old** | **+48.2 %** |

⚠️ **A1 agreed to 0.0 % and I generalised from it. That was luck.** The worst cases
are off by 20–48 %, and they are systematically the **sparse ones** — A2old captured
2,375 nodes against A1's 87,138. With few nodes, one phase is a poor estimate of the
revolution.

### Why the M2 conclusion is still sound

The M2 result compares `ratio_static` against `ratio_delta` **within each case,
computed from the same snapshot on both sides**. Whatever bias the snapshot carries
is common to both windows and cancels in the comparison. The δ-window's removal of
the confound (ρ +0.409 → −0.219) does not depend on the absolute values.

### What it affects

| | |
|---|---|
| ❌ M2's confound-removal conclusion | unaffected — same snapshot both sides |
| ❌ M5's rheology conclusion | unaffected — same snapshot, both models |
| ⚠️ **Absolute per-case δ-window ratios** | **not quotable** until re-run phase-averaged |
| ⚠️ Any regression of ratio on geometry (Phase 4) | would inherit up to 48 % noise on the sparse cases |

### The decision

**Option A — re-run the δ sweep phase-averaged.** Correct, and expensive: 65 cases ×
166 timesteps. The gate work measured PVTU loading at ~40 s/timestep on the big
meshes, so this is **hours, not minutes**, and it needs a batch job.

**Option B — re-run phase-averaged only for the sparse cases.** The 4 outliers are
all low-`n_selected`. Cheaper, and defensible if the criterion is stated in advance
(e.g. `n_selected < 20,000`).

**Option C — quote snapshots with the uncertainty attached.** "ratio = 0.91 ± 25 %
(snapshot; sparse case)". Honest, cheap, and adequate if the next step only uses
rankings.

**Recommendation: C now, A before Phase 4.** Phase 3 works on one configuration and
does not need 65 absolute ratios. Phase 4 regresses on geometry and does.

---

## What the thesis and FEMUSS source settled — and what they did not 2026-09-30

Two primary sources checked: the colleague's thesis
(`thesis_venghaus.pdf`, 192 pp, 3 included papers) and the FEMUSS source at
`/scratch/.../Femuss-Edgar/`.

### ✅ Settled

| item | answer |
|---|---|
| **O25** | **SOLVED** — the `.mat` tables are shear measures (above) |
| **O7** (alloy) | **Al6063-T6**, μ₀ and m "obtained from an extensive experimental program" (thesis Figs. 20–21). Consistent throughout, so the cohort's borrowed tables are very likely the right material. ⚠️ Still worth one confirmation that the *cohort* runs used the same file. |
| **`PinRadius`/`ShoulderRadius` in the case setup** | **unused wizard defaults.** `PinGeometry = "Arbitrary"`, `ToolDimensionsOrigin = "From CAD"`, and the STL bounding-box half-width is exactly **7.0 mm** — matching our measured `r_tool`. ⚠️ So the `4.0`/`9.0` values are decoys; **our measurements were right.** |
| von Mises yield criterion in FEMUSS | `Tau = UniaxialStressLimit` — a straight pass-through, no hidden factor in that path |
| `Mod_som_PlasticMaterial.f90` | rate-*independent* plasticity (`viscosity/deltat`) — **not** the FSW path, so not relevant |

### ⚠️ O4 — the thesis does NOT close it, but it reframes it usefully

The thesis validates against **reaction forces (Fx, Fy, Fz), torque, and K-type
thermocouple temperatures** at three material positions, converted to the Eulerian
frame. Real experimental data — but **not defect data**.

Void/wormhole/porosity appear 11 times, all in literature review (high ω → tunnel
defects, high v → wormholes, tilt suppresses voids). **No macrograph, CT scan or
sectioned weld is presented anywhere in the thesis or its three papers.**

> **The useful reframing:** the flow and thermal fields underneath our damage metric
> **are** experimentally validated, against forces, torque and temperature. The
> damage layer on top is not. That is a much better position than "nothing is
> validated", and a more honest one than implying the defect prediction is.

---

## ⚠️ O29 (new) — the "region split" recollection does not match the code

Asked to verify a colleague's description: regions for stress calculation based on
shoulder/pin ratio, an **upper 25 %** split, something about **melted vs not fully
melted**, and **advancing vs back side** treated differently.

### ✅ One part is real and precisely specifiable

`Mod_som_FswAnalysisElmope.f90:593` — the **"Modified Norton"** friction law, which is
what every case uses (`FrictionModel = "Modified Norton"` in the `.spd`):

```
a(x) = ½[ a_max + a_min + (a_max − a_min)·tanh(−6x / R_shoulder) ]
```

C1 (`a_max`=8e8, `a_min`=5.8e7, R=9 mm): **friction is 13.8× higher ahead of the tool
than in the wake**, transitioning over ~R/3 at the tool centre.

⚠️ **This is a FRONT/BACK (x) asymmetry, not advancing/retreating (±y).** Worth being
precise about: our damage analysis uses the ±y flank distinction; this is
approaching-vs-wake material. Different axes of the same process.

✅ **And it corroborates our σ_m sign determination.** Higher friction upstream ⇒ more
forging pressure upstream ⇒ exactly the **+1.70 MPa upstream / −1.35 MPa downstream**
measured in P0.2. The solver's friction model and our empirical sign test agree.

### ❌ Three parts not found

| recollection | finding |
|---|---|
| upper 25 % under the shoulder treated differently | **not found.** The only shoulder-region code (`Mod_som_FswAnalysis.f90:96–132`) is **mesh adjustment** for tilt and concavity — node displacement, not a constitutive split. Its parameters are radial (`geo_ShoulderHighpointRadius`, `shoulderFactor = 1.25`), not a depth fraction. |
| melted vs not completely melted | **contradicted by the thesis**: peak temperatures are "between 80 % and 95 % of the melting temperature … **it is not melted**" (§4, p. 66). No melt phase, no solidus/liquidus anywhere in the source. |
| based on shoulder-to-pin-length ratio | **not found as a stress criterion.** The two radii do appear together, but in the concave-tool cavity depth: `cavityDepth = (R_shoulderHighpoint + R_pin) × 0.125`. |

⚠️ **The 0.125 may be the source of the "25 %"** — it is 1/8 of a *sum* of two radii.
Plausible to half-remember as "an eighth"/"a quarter of something".

### The question

⚠️ **Do not assume the recollection is simply wrong.** It may describe a
**post-processing** convention that lives outside this source tree — e.g. how the
colleague defined sampling regions when *reporting* stresses, which is precisely the
kind of thing our own window definition (O10/M2) had to settle independently.

**Ask:** was the upper-25 %/melting distinction a *solver* setting, or a convention
used when post-processing and reporting stresses? If the latter, where is it
implemented?

---

## Discrepancies from the setup files — 2 of 3 resolved

These came out of comparing the case setup against what we measured. None threatens a
conclusion — our labels were measured from the fields, not taken from the text — but
each is a factual conflict worth resolving.

### ~~1. Rotation sense~~ — ✅ RESOLVED 2026-09-30

**The authoritative source is `data/*.som.dat` → `RPM`, signed. Positive = CCW.**
Confirmed by the colleague and verified independently from the velocity field.

⚠️ **The two families rotate in OPPOSITE senses**, which our records did not say:

| family | RPM | sense | our measured `advancing_y_sign` |
|---|---|---|---|
| **PinShapes** (20) | **+996** / +1000 | **CCW** | **+1** (19/19) |
| **ROM/FOM_cases** (22) | **−400 … −800** | **CW** | **−1** (45/45) |

**64/64 agreement** between the signed RPM and the independently measured flank.

⚠️ **A sampling trap on the way, worth remembering.** A first `u_θ` check sampled a
shell at r/R ≈ 1.0–1.15 and reported CW for *both* families — confidently wrong.
There the rotating layer has decayed to ~0.002 m/s while the traverse stream is
comparable, and the stream only cancels over an azimuthally uniform ring (which a
flatted or threaded pin is not). Resampling inside the shear layer at r/R ≈ 0.4 gives
±0.11 m/s — **50× stronger** — and a least-squares split of solid-body vs stream terms
gives **A = +0.114 for RPM +996 (CCW)** and **A = −0.119 for RPM −400 (CW)**.

> A diagnostic that samples where the signal is weakest can *invert* its sign, not
> merely blur it. Sampling location is part of a measurement's definition.

✅ **Nothing in our results changes.** `detect_orientation()` always measured the
flank per case. But P0.2's "both confirm CW" is corrected — it generalised from
cohort cases only, and had the flank been hard-coded from it, every PinShapes result
would have had its flanks swapped and the family ranking inverted.

### 2. RPM: setup says 800, the solver input says 996/1000

| source | value |
|---|---|
| `C1.spd` → `RotationalSpeed` | **800.0** |
| `data/*.som.dat` → `som_rpm` | **996** (19 cases), **1000** (B2) |

A 25 % difference. Checked whether it is a consistent rescaling — **it is not**:

| source | ω | dt | steps/rev |
|---|---|---|---|
| `C1.spd` | 800 rpm | 5.000e-4 | **150.0** (= `DegreesPerTimeStep` 2.4) |
| `som.dat` | 996 rpm | 3.629e-4 | **166.0** |

ω ratio = 1.245, dt ratio = 1.378 — **different**, so steps/rev genuinely changed
150 → 166.

✅ **Which one is operative is settled:** 166 is exactly the phase count our
full-revolution runs used (`VEL_START=34, VEL_END=199`), so the **solver** value
governs and our phase averaging was correct.

⚠️ `SpeedUp = 100.0` also appears in the setup. It is *plausibly* the thermal/advection
speed-up common in FSW models, but **that is a guess — not verified from any file**,
and it would matter for any conclusion phrased in physical time.

**The questions:** (a) which ω is physical — 800 or 996? (b) what does `SpeedUp = 100`
rescale? ⚠️ Our ω-trend conclusions used the **cohort's** ω = 400–800 from *its* own
`som.dat`, so they are internally consistent either way; this affects how ω is
*reported*, not the trend.

### ~~3. O26~~ — ✅ ANSWERED 2026-09-30

**Same simulations; `FOM_cases` is the updated version. Use only it.** (Colleague.)

⚠️ **Counts already quoted are inflated.** The completed sweeps pooled both roots:

| statistic | as reported | corrected |
|---|---|---|
| δ sweep cases | 65 | **~45** |
| M5 cases compared | 41 | **~21** |

Conclusions do not move — M5's ρ = +0.9995 and the per-root ρ values were reported
separately — but **n must be corrected wherever quoted**. `SCOPE=fom` in
`sbatch_delta_window.sh` now excludes `FOM_cases_PT`.

---

## ⚠️⚠️ O33 (new) — Phase 3 does not reproduce the Stage 1 result

> **⚠️ ROOT CAUSE FOUND 2026-10-01 — see O34.** 60–88 % of particles never
> enter the shear layer, so the global median measures the bypass fraction. With
> regional metrics (O35) the family ranking partially recovers (A/B/C spread
> 0.6 % → 9.0 %, B-Flutes highest as expected). The residence-time hypothesis
> below is superseded. **Do not quote O33 as a physics result.**

Making the result figures surfaced two disagreements. Both were checked for
plotting errors first; **both are real.**

### 1. The advancing side is not worse in accumulated damage

| | advancing wake | retreating wake |
|---|---|---|
| Stage 1 (nodal η, phase-averaged) | **94 %** of nodes tensile | 1 % |
| Phase 3 (pathline lnΦ) | **LOWER in 45/45 cases** | higher |

✅ Flank assignment verified against the **measured** `advancing_y_sign`
(PinShapes CCW ⇒ +y, cohort CW ⇒ −y). ✅ No missing data.

⚠️ **The obvious explanation was tested and FAILED.** If advancing particles simply
passed through faster, they should end farther out. Measured on A1: advancing
median radius **9.49 mm**, retreating **9.86 mm** — they ended **closer in**, the
opposite. **No replacement explanation.**

### 2. The family ranking does not reproduce

| family | Stage 1 ratio | Phase 3 lnΦ |
|---|---|---|
| C-Threads | 1.117 | 0.0653 |
| B-Flutes | 1.022 | 0.0654 |
| A-Flats | 0.892 | 0.0650 |
| D-Concavity | 0.905 | **0.0521** |

A, B and C differ by 0.6 %, less than A-Flats' own within-family spread. Only D
separates — and D is also the O32 accumulation anomaly, so that may be the same
effect rather than a damage result.

### What it does and does not mean

❌ **It does not invalidate Stage 1.** The two answer different questions: Stage 1
asks *what is the stress state at nodes in the shear layer*, Phase 3 asks *what did
a particle accumulate along its whole route*. A region can have the more dangerous
stress while particles passing through accumulate less.

⚠️ **Most likely candidate, untested:** the Phase 3 runs used 10 mm travel with
seeding in the upstream 30 % of X. If particles pass *around* the pin rather than
*through* the shear layer, the integral is dominated by the long low-damage
approach — which would dilute both the flank and the family contrast, exactly the
two symptoms observed.

**Cheap tests, neither done:**
1. re-run one case seeded close to the pin and see if the contrast returns;
2. weight the integral by residence time inside the shear layer;
3. check what fraction of each particle's path was actually inside the layer.

### ⚠️ Consequence for reporting

**Do not present the Phase 3 family ranking as confirming Stage 1.** Defensible
today: the two damage *models* agree with each other (ρ ≥ 0.98); Stage 1's η result
stands on its own and is what agrees with the colleague's flow analysis; Phase 3's
accumulated damage does not yet reproduce either finding and why is open.

---

## ⚠️ O32 (new) — the D-Concavity family accumulates ~18 % less than everything else

> **⚠️ LIKELY EXPLAINED 2026-10-01 — see O35.** The deficit is localised to the
> TOP 20 % of thickness: D's `z_top` p99 is 22.6 vs 358 for A/B/C, and its `z_root`
> p99 is 0.3–0.5 vs 15–40. A top-surface BC interaction on the only tilted family,
> not a bulk difference.

Phase 3 across all 45 cases, accumulation fraction (particles that accrued any
damage):

| case | accumulated |
|---|---|
| **D1** | **81.3 %** |
| **D4** | **81.4 %** |
| **D3** | **81.5 %** |
| **D2** | **81.7 %** |
| **D2old** | **83.0 %** |
| all other 40 cases | **97.5 % median** |

⚠️ **All five lowest are the entire D family, and no other case is below 90 %.**
That is systematic, not noise.

**The likely explanation, unverified:** D is the **only family with tool tilt** (2°,
vs 0° for A/B/C). A tilted tool changes which material passes under the shoulder and
through the shear layer, so a larger fraction of seeded particles plausibly spend
the run in near-undeformed material. Physically reasonable — but *plausible* is not
*established*.

**What it affects:**

* ⚠️ **D-family damage statistics rest on ~18 % fewer samples** than the others. Any
  cross-family comparison involving D should say so.
* It does **not** affect the within-D ranking, or the two-model agreement (D cases
  have rank corr +0.98…+1.00 like everything else).

**What would settle it:** check where the non-accumulating D particles end up. If
they sit outside the shear layer, it is the tilt geometry and expected. If they are
inside it but reading zero ε̇, that is a sampling or level-set problem on tilted
meshes and would matter much more.

⚠️ **Cheap to check** (the positions are in the npz) and worth doing before any
D-family conclusion is quoted.

---

## ⚠️ O31 (new) — Rice–Tracey has no saturation term: lnΦ is a RANKING

Phase 3 runs correctly, and its output needs one caveat stated every time it is
reported.

| | |
|---|---|
| Φ is a **volume fraction** ⇒ Φ ≤ 1 | |
| with the standard seed Φ₀ = 1e-4, failure is at | **ln(Φ/Φ₀) = 9.21** |
| measured max on cylindrical_000 (213 steps) | **1270 = 138× the threshold** |
| fraction of particles past failure | **13.7 %** |

⚠️ The ODE as the plan specifies it —
`d(lnΦ)/dt = 0.849·edot·exp(1.5η)` — has **no bounding term**. He et al. eq. (1)
includes a `1/(1−Φ)` factor which is what saturates it. So lnΦ grows without limit,
and a value of 1270 does not mean "more porous than 826" in any physical sense; both
mean *failed*.

**Not a bug** — the implementation matches the plan. **It is a reporting hazard**, and
the output now says so explicitly (`PAST FAILURE: 13.7 %`, plus two warnings).

### The decision

✅ **Use lnΦ as a susceptibility ranking.** That is exactly what the project needs — a
continuous field of *where voids are likely* — and the ranking is well-behaved (rank
correlation with Cockcroft–Latham is **+0.995**).

⚠️ **Never present lnΦ or exp(lnΦ) as a predicted porosity.** `exp(1270)` is ~10⁵⁵¹.

**Options if an actual porosity is ever needed:**

1. **Add the `1/(1−Φ)` factor** and integrate Φ directly, not lnΦ. ⚠️ Reintroduces the
   stiffness that log-space was chosen to avoid, and needs a real Φ₀ — which nobody
   measures (O-nucleation).
2. **Clamp lnΦ at ln(1/Φ₀)** and report "failed" as a saturating indicator. Cheap, and
   honest about what the model can say.
3. **Leave it as a ranking.** ← recommended, and consistent with O4: without
   experimental defect data, a calibrated porosity could not be validated anyway.

---

## ~~O30~~ — ✅ CLOSED 2026-09-30: Phase 2 implemented (kept for the reasoning)

Checked the implementation plan against the code instead of assuming Phase 1 left us
ready for Phase 3. **There is a Phase 2 in the plan that was never on the status
board**, and it is only 1/3 complete.

Phase 3's two models need three nodal fields and two accumulators:

```
Rice–Tracey     : d(lnΦ)/dt = 0.849 · edot · exp(1.5·eta)
Cockcroft–Latham: dC/dt     = max(s1_ratio, 0) · edot
```

| requirement | status |
|---|---|
| `edot` on GPU | ✅ |
| **`eta`** on GPU | ❌ computed in `fields.py`, never uploaded |
| **`s1_ratio`** on GPU | ❌ computed in `fields.py`, never uploaded |
| 2 accumulators | ❌ the kernel carries **one** |

✅ **No new physics needed** — `build_damage_fields()` already returns all three, and
they are validated by the 5 analytic tests and used across the 45-case survey. The gap
is plumbing.

⚠️ **One real consequence.** `--damage` deliberately passes `pressure=None` so it needs
no extra field. **Both `eta` and `s1_ratio` require pressure**, so Phase 2 must load it
— ending the "no extra field load" property of the Phase 1 design. Pressure is in the
same PVTU (one more array per timestep, not a second pass), but this should be stated
rather than discovered.

⚠️ **O22 still binds:** a Phase 3 run that produces meaningful *physics* needs
**N_STEPS ≥ ~5500** with the current seeding, or seeding nearer the pin. Correct ODEs
do not rescue a 400-step run where 0 % of particles reach the tool.

---

## What the colleague's internal decks changed 2026-09-30

Seven decks in `Downloaded/FSW/internal/`. Effects on open items:

| item | change |
|---|---|
| **O12** (A1 outlier) | ⚠️ **REFRAMED — it is physics.** They independently report **V_PD,A1 ≈ 1.5 × V_PD,A2** ("flat faces increase plastic deformation volume"). We measured A1 capturing **2.4×** A2's nodes in the same window and suspected a sampling artefact. Same sign, same order, two methods. **A1's large sample reflects a genuinely larger plastic zone.** |
| **O4** (no ground truth) | ⚠️ **unchanged.** Nothing in the seven decks either — all validation is forces, torque, temperature. ⚠️ But they use "**dominant transverse force (Fy > Fx) signals material stagnation and defect risk**" as a proxy, and forces *are* experimentally validated. **A possible external check on our ranking that needs no new experiment.** |
| **O5** (pressure POD) | their ROM deck sets the frame: three options for changing meshes, and they favour **reference-space L2 projection**. Our O5 sits **inside** that workstream, not beside it. |
| **new: thermal mechanism** | "Low temperatures at pin tip initiate defects" — a mechanism our η does **not** model beyond σ_eq(T). A stated limitation, not an immediate gap. |

### ⭐ Two independent confirmations of our central results

**1. The mechanism.** `PinShapePaper_sept9th`: "*Localized mass transport deficit and
flow stagnation at the rear AS cause tunnel/wormhole voids*", "*Lack of backfilling AS
indicates defects*". Reached from flow-rate and pressure post-processing — **no damage
model, no triaxiality, no pathlines.** Our filling-failure argument, independently.

**2. The family ranking**, from a completely different quantity:

| family | our damage ratio | their refill finding |
|---|---|---|
| **C-Threads** | **1.117** worst | **"perform the worst among pin profiles"** |
| A-Flats | **0.892** best | **"very effective in backfilling"** |

⚠️ M5 showed the ranking is insensitive to the *rheology*. This shows it is reproduced
by a different *physical measure*, by a different person, with neither tuned to the
other. **The strongest external support the method has.**

### ⚠️ One disagreement worth resolving

They mark their own threads result **"QUESTIONABLE"** — "*evaluation plane has to be
set closer to pin root to visualize benefit of threads*". That is **the same defect as
our O10/M2**: a sampling region fixed in the wrong place.

Our window already samples there (lower 70 % of thickness, pin-anchored radially, so
C-Threads is r/R = [0.29, 0.48] about `r_pin` = 2.07 mm). So:

* if their threads result **flips** when the plane moves down, and ours still ranks
  threads worst → a substantive disagreement to resolve;
* if theirs **confirms** worst → both methods agree at the corrected location.

**Either outcome is informative, and it is a cheap thing to ask.**

---

## Blocks a validation claim

### O4 — no experimental ground truth

No CT scan, macrograph or stop-action section has been identified for any of the 65
cases. Everything established so far is **internal consistency**:

* the predicted tensile region lands on the advancing side, where the literature
  reports defects;
* the parameter trends have the expected sign;
* two independent rheologies agree on the ranking.

⚠️ Those are *necessary* checks, not sufficient ones. **Nothing here yet
demonstrates that a high computed damage corresponds to a real void.**

**What would settle it:** any sectioned weld from one of these parameter sets, even
one. A single case with a known wormhole and a known sound case would convert the
whole thing from "physically motivated ranking" to "validated predictor".

**Decision needed:** is such data obtainable from the project's experimental side? If
not, the deliverable must be framed as a *susceptibility ranking*, not a defect
prediction. ⚠️ This is a framing decision, and it is worth making explicitly rather
than by omission.

---

## Affects magnitudes, not rankings

### O7 — the cohort's borrowed material tables

The 22 `ROM/FOM_cases` have no `.mat` of their own and use cylA's Al6063 Norton
tables. The 20 PinShapes cases each ship their own.

⚠️ If the cohort was actually run with a different alloy or a different fit, every
cohort σ_eq is wrong by an unknown factor — a *second*, unquantified scale error on
top of O25.

**What would settle it:** ask the colleague which material file the cohort runs used.
One question.

**Interim:** cohort quantitative claims are provisional; the ranking is not affected
if the same table was used for all 22.

### O15 — the metric is pressure-sensitive, not velocity-sensitive

Measured: runs agreeing on |u| to 5 % give **opposite** damage predictions, because
η = σ_m/σ_eq takes its numerator straight from pressure.

This is not a defect — it is a *prediction of the constitutive law*, and §6d of the
primer derives it: the Norton exponent m ≈ 0.09 makes σ_eq almost insensitive to
strain-rate error (a 50 % ε̇ error moves σ_eq by 1–7 %), so **η inherits essentially
all its error from σ_m, i.e. from pressure.**

**Consequence, and it is the important one for the project:** when the ROM is built,
**pressure accuracy is the binding constraint, not velocity accuracy.** A ROM that
reproduces velocity beautifully and pressure poorly will produce confident wrong
answers.

**Decision:** the pressure POD (O5) must be held to a tighter tolerance than the
velocity POD. That number should be set from a sensitivity sweep, which has not been
run.

---

## Affects which numbers to quote

### O22 — a timing gate's damage output is not physics

Job 22366672 (the Phase 1 throughput gate) wrote a `damage_edot.npz`. It is a valid
*timing* run and a meaningless *physics* one:

| | |
|---|---|
| steps | 400 → 0.145 s of path time → **1.45 mm** of tool advance |
| particles reaching the 7 mm tool | **0.00 %** (median final radius 16.7 mm) |
| steps needed to reach the tool edge | **~1130** |
| steps needed to traverse the stir zone | **~5500** |
| the case's production `N_STEPS` | **8000** ✅ correctly sized |

⚠️ **Phase 3 physics runs need N_STEPS ≥ ~5500 with this seeding, or seeding much
closer to the tool.** A damage file from a short run is a by-product, not a result.

### O12 — A1 is a family outlier

A1's ratio is **1.454** against an A-family median of **0.748** — the largest
within-family departure anywhere in the set.

⚠️ This is now more interesting, not less, because M4 found A1's STL is
**8,280 triangles** where A2/A3/A4 are ~2,790 — a *different mesh resolution*, and
A1's `r_pin` (2.39 mm) is the largest in its family (A2: 1.23 mm).

**Two hypotheses, both testable:** (a) A1 is genuinely a different geometry and
belongs in its own group; (b) the 3× finer STL changes the level-set and hence the
measured window. **Not yet distinguished.**

### O13 — the binary flank label is fragile

B1's tensile-flank label flips on a **0.0023** margin in `frac_positive`. Near
ratio ≈ 1 the binary label is a coin toss.

**Decision already taken in practice:** prefer the **continuous ratio** everywhere
and treat the flank label as a presentational convenience. ⚠️ Worth stating
explicitly in any table that shows a flank column.

---

## Affects the case inventory

### O26 — are `FOM_cases` and `FOM_cases_PT` the same simulations?

`ROM/FOM_cases` (22) and `ROM/FOM_cases_PT` (20) carry the same case names and, in
the M5 run, both produced results. If they are the same simulations, pooling them
**double-counts** and inflates every cohort-wide n.

⚠️ The M1/M2 sweep reported them separately and their ρ values differ (−0.772 vs
−0.860), which is *weak evidence they are not identical* — but that could equally be
the 2-case difference in n.

**What would settle it:** md5 the last PVTU of a few matching cases. Minutes of work,
and it is included in the commands below.

### O14 — the two cohort copies diverge

The workstation and LUMI copies of the ROM cohort are **different simulations**, and
the root cause is the **pressure field** (P median +1.3 vs +27.1 MPa, correlation
0.25 — not a rounding difference).

**Decided: LUMI is the reference.** Both kept. ⚠️ Still to be raised with the
colleague — a 20× difference in median pressure between two copies of "the same"
cohort is worth understanding, and given O15 it is exactly the field the metric is
most sensitive to.

---

## Infrastructure

### O8 — the damage work is uncommitted

`jaxtrace/damage/*`, the `run_tracking.py` and `benchmark_femuss_comparison.py` edits,
and all the analysis scripts are uncommitted on both checkouts.

⚠️ **This is the only item that can lose work.** Everything else is a question; this
is a risk. Worth clearing before any branch switch or repo reset.

### O20 — the workstation GPU has two timing states

Persistence mode is **disabled**, and identical work runs at either ~85 s or ~162 s.
Comparing arms across the transition produced a **−23.5 %** result — a fabricated
*speedup* from adding work.

**Mitigated**: the gate now runs two alternating rounds and refuses to report a ratio
if same-flag runs disagree by >10 %. **Fix at source**: enable persistence mode.

### O21b — full-revolution loading dominates any per-case sweep

C1's runner loads 166 PVTU slices; mesh load was **6520 s of an 8433 s arm (77 %)**.
Collapsing to one slice cut it to 83.5 s — **78×**.

⚠️ **Any future cohort-wide sweep that genuinely needs the full revolution** (O27
option A is exactly that) must budget for this, or use a cached/`.npz` intermediate.
65 cases × full revolution is not affordable naively.

---

## Deferred by decision, not oversight

### O5 — pressure POD not built

Deferred to outer-loop B with the rest of the ROM work. ⚠️ Given O15, this is the
**highest-stakes** deferred item: the metric's accuracy will be set by pressure POD
quality.

### O6 — cylA has pressure at only one timestep

`ts=159` only. Fine for the steady analysis it was used for; rules cylA out of any
transient study.

---

## Recently closed, for reference

| # | resolution |
|---|---|
| O1 | three rheology models implemented; Norton recommended |
| O2, O3 | depth-averaged window; all 42 cohort cases run |
| O9 | 003/A2 inversion → systematic ω transition, ρ = −0.98, crossover ~575 rpm |
| O10 | **confirmed**: `r_tool` is the shoulder; window held 72 % of nodes at r/R ≤ 0.45 |
| O11 | shear-layer edge constant at r/R ≈ 0.47 over ω = 400–800 — not an artefact |
| O16 | δ measured on 64 cases: **0.53–2.80 mm**, does NOT transfer between families |
| O17 | **RETRACTED** — "L2 loses 21 % of interior points" was my harness using the wrong entry point |
| O18, O19 | Phase 1 ported to the production kernel; gate runs through `run_tracking.py` |
| O21 | `VEL_RANGE` collapse: 6520 s → 83.5 s |
| O23, O24 | pin-anchored window implemented; δ sweep is an sbatch job |
| O28 | **D4 diagnosed**: `C2_200`/`C2_201` are truncated shells, `C2_199` is complete |
| M3 | throughput gate: CUDA +0.15…0.51 %, ROCm +1.77 %, accumulator live |
| M1, M2 | δ per case; confound removed (+0.409 → −0.219) |
| M5 | **ρ = +0.9995, 0/41 flank flips, family ranking identical** |

---

## O34 — particles BYPASS the tool; the global damage metric measures the bypass fraction

**Status:** root cause found 2026-10-01. Supersedes the residence-time hypothesis
under O33.

**What was observed** (user, from `vtu_final/`): final particles sit around or behind
the tool, having apparently not travelled far.

**What is actually wrong.** Not the step count. The FOM is solved in the
**TOOL-FIXED frame** — confirmed from `data/cylindrical.som.fix`, which prescribes
`111  5.0000000000e-03 0.0 0.0` on 71 inlet nodes: material is driven past a
stationary tool at +5 mm/s in x. In that frame a far-field particle must advect the
full travel distance. Measured instead:

| case | sim time | v_adv | expected drift | **actual mean x** |
|---|---|---|---|---|
| ps_A1 | 1.000 s | 10 mm/s | +10 mm | **−0.52 mm** |
| rom_000 | 1.999 s | 5 mm/s | +10 mm | +5.77 mm |

PinShapes particles moved **backwards by 0.5 mm** over a nominal 10 mm of travel.
They are **recirculating**, not lagging.

**The consequence that invalidates the headline metric.** Fraction of particles with
`lnPhi < 1` (i.e. essentially no damage at all):

- PinShapes: **80.3–88.4 %**
- cohort:    **52.3–64.9 %**

So the global median ln(Phi/Phi0) is dominated by particles that never entered the
shear layer. **It is a measurement of the bypass fraction, not of damage.** This is
why O33's family ranking collapsed (A/B/C within 0.6 %) and why the flank contrast
vanished: both were diluted by a ~60–88 % null population.

**Cause:** `--seed-source box-frac --seed-fraction 0.0 0.3 0.0 1.0 0.0 1.0` seeds the
upstream 30 % of x across the FULL y and z span. Most of that y-span streams past the
tool and never reaches the stir zone.

⚠️ **Tripling the step count does not fix this and would make it worse** — it
accumulates more damage in the recirculating population while the bypass particles
still contribute zero, at 3x the GPU cost. Fix the seeding first.

**What would settle it:** seed within a y-band that actually enters the stir zone
(|y| < ~1.5 R_shoulder), or seed on a streamline-traced inflow patch, and report
regional metrics (O35) rather than a global median.

---

## O35 — a single global metric is not interpretable; regional metrics recover the signal

**Status:** implemented 2026-10-01 in `make_regional_metrics.py` ->
`results/phase3_regional.csv` (45 cases, 35 columns).

Raised by the user: a max/mean per-particle damage is dominated by the top layer
(z_max), which is also where a boundary condition is applied — so the number mixes
physics with a BC artefact.

**Confirmed, and it was hiding two real results.**

Regions: radial (`pin` r<2.4mm, `shear` 2.4–7mm, `shoulder` 7–10.5mm, `far`) and
depth as a FRACTION of plate thickness (`z_top` upper 20 %, `z_mid`, `z_root` lower
20 %), plus `stir_nobc` = (r < 7 mm) AND NOT z_top — the interpretable one.

**1. Family separation improves 15x once the bypass population is excluded:**

| family | global median | **stir_nobc median** | z_top p99 | bypass |
|---|---|---|---|---|
| A-Flats | 0.0650 | 0.3824 | 358.6 | 80.6 % |
| **B-Flutes** | 0.0654 | **0.4131** | 356.1 | 80.3 % |
| C-Threads | 0.0653 | 0.3790 | 357.5 | 80.5 % |
| D-ConcavityTilt | 0.0521 | 0.3050 | **22.6** | 87.2 % |
| cohort | 0.4909 | 2.2787 | 265.7 | 57.7 % |

A/B/C spread: **0.6 % global -> 9.0 % regional**, with **B-Flutes highest**, which is
the expected ordering since B has the deepest lobes (20–32 % of pin radius, M4).

**2. The z_max artefact is real and is specific to D.** `z_top` p99 is **358 for
A/B/C but 22.6 for D** — a 16x gap confined to the top 20 % of thickness, while D's
`z_root` p99 is 0.3–0.5 vs 15–40 elsewhere. D is the only family with tool tilt (2°).
This localised top-layer difference was contaminating every global number for D, and
is the likely explanation for O32 (D's low `acc_frac`, 81–83 % vs 97.5 %).

⚠️ `z_top` must be reported separately, never folded into a headline number: it sits
against the prescribed-velocity top surface and is partly a BC response.

**Recommended headline metric going forward:** `stir_nobc` median + `stir_nobc`
frac_failed, with `z_top` and `bypass_frac` always reported alongside as diagnostics.

---

## O36 — selectable damage integration order (1 = Euler, 4 = RK4)

**Status:** implemented 2026-10-01. `--damage-order {1,4}`, default 1.
`run_tracking.py` -> `create_rk4_comparison(damage_order=...)`.

**Why it was asked for, and the correction it required.** The Phase 3 note argued
Euler was sufficient "because the driver is interpolated, so 4th order buys
nothing". The user asked whether the same argument then undermines RK4 for the
particle POSITION, since previous years' work concluded RK4 was needed. **It does
not, and the earlier phrasing invited that misreading.** Two distinct problems:

| | position | damage accumulator |
|---|---|---|
| ODE | `dx/dt = u(x(t))` | `d(lnPhi)/dt = f(eta(x), edot(x))` |
| RHS depends on state? | **yes** | **no** |
| what it is | genuine IVP | **pure quadrature along a known path** |
| why RK4 helps | the TRAJECTORY is curved; Euler cuts the corner | only centring, see below |

Measured curvature that justifies RK4 for position: at r = 3 mm the cohort sweeps
**9.0 deg and 0.47 mm of arc per step** (40 steps/rev at dt = 3.75e-3 s, 400 rpm);
PinShapes sweeps 2.17 deg / 0.114 mm. Euler would drift off the streamline element
after element. **Previous years' conclusion stands — keep RK4 for position.**

**What order 4 actually buys for the accumulator.** NOT formal 4th order: the P1
driver is C0 with a discontinuous derivative at every element face, and ANY
4th-order quadrature degrades to 1st-2nd order across a kink. The real benefit is that it
**samples the step midpoint**. Verified numerically:

| driver along the step | Euler (k1 only) | RK4 | exact |
|---|---|---|---|
| constant | 0.300000 | 0.300000 | — (agree to 5.6e-17) |
| linear in t | **0.000000** | 0.005000 | 0.005000 |

On a linearly-varying driver the k1-only estimate misses the integral **entirely**
(100 % error) because it samples the value at the step's START. At 9 deg/step that
bias is real wherever the driver has a steep gradient -- i.e. in the shear layer.

**Cost.** 3 extra `interpolate_scalar_single` gathers per scalar per step (9 total
for the 3-slot model), at elements `elem_k2/k3/k4` that the velocity step **already
located**. No extra element searches, which is what actually costs. Gathers hit
memory already in cache for the velocity read.

⚠️ **The RK4 weights are applied to the RATE, so each stage carries its OWN eta and
s1.** Weighting only `edot` while holding `eta` at k1 would be wrong, because
`exp(1.5*eta)` is the dominant and most rapidly varying factor. Each stage is
clamped before its own exponential.

**Default stays 1** so the 45-case Phase 3 survey remains reproducible.

### ✅ MEASURED on real data 2026-10-01 — the effect is a TAIL correction, not a bulk one

Controlled A/B on ps:A-FlatsVariations/A1: identical case, seed, particle count
(20,000), travel (1 mm) and n_steps (276); **only `--damage-order` differs**
(jobs 22475888 = order 1, 22475353 = order 4).

| | order 1 | order 4 | change |
|---|---|---|---|
| **position max abs diff** | — | — | **0.000e+00 m** |
| lnPhi median | 0.004456 | 0.004456 | **+0.01 %** |
| lnPhi max | 179.846 | **147.671** | **−17.9 %** |
| lnPhi mean rel. diff | — | — | 0.05 % |
| lnPhi p99 rel. diff | — | — | 0.40 % |
| CL max | 119.546 | 105.408 | −11.8 % |

**Two conclusions.**

1. ✅ **The implementation is verified.** Position difference is EXACTLY zero, so
   `--damage-order` provably does not perturb the trajectory -- which is what the
   build-flag design intended and is now proven rather than asserted.

2. ⚠️ **The effect is concentrated in the extreme tail, not the bulk.** The earlier
   synthetic linear-driver test (Euler 100 % error) made the midpoint bias look like
   a large systematic correction. On real data it is **0.05 % for a typical particle
   and −17.9 % for the maximum**. That is physically coherent -- the k1 bias matters
   where the driver has a steep gradient, i.e. for the few particles passing closest
   to the pin -- but it means order 4 **changes rankings based on tails
   (`frac_failed`, `z_top p99`, `lnphi_max`) and leaves median-based rankings
   essentially unchanged.**

⚠️ **Do not quote order 4 as "more accurate damage" in general.** It removes a
tail bias. Since `lnPhi` is already only a RANKING (O31, no saturation term), the
honest statement is: the tail of the Euler run was **overestimated by ~18 %**, and
the order-4 run is the one to use for any tail-based metric.


---

## O37 — Phase 4 run: RK4 damage, 3x travel, VTU export (submitted 2026-10-01)

`sbatch_phase4_rk4_array.sh` -> `/scratch/.../damage/phase4_rk4_30mm/`

Changes from Phase 3, and why each:

| | Phase 3 | Phase 4 | reason |
|---|---|---|---|
| damage order | 1 (Euler) | **4 (RK4)** | O36 |
| travel | 10 mm | **30 mm** | O34: at 10 mm, PinShapes particles had mean displacement **-0.52 mm** and had not resolved a wake |
| seed box | full y-span | **full y-span (unchanged)** | user's call, see below |
| export | `--no-export` | **vtu every 50 steps** | ParaView trajectories |
| staging | direct to scratch | **/flash -> rsync -> scratch** | per-step VTU on Lustre HDD tanks throughput |
| output dir | `phase3_overnight_20260930` | `phase4_rk4_30mm` | **Phase 3 results untouched** |
| wall/task | 02:00:00 | **06:00:00** | 3x steps: worst case 4656 s -> ~4.0-4.3 h |

**On keeping the full y-span.** Two fixes were available for O34; the user chose to
go further rather than narrow the seed box, on the grounds that (a) the GPU kernel is
search-bound, not particle-bound, so extra particles are nearly free, and (b) the
bypass population is itself physical -- it is the material that fills the wake -- so
regional metrics (O35) should SEPARATE it rather than the seeding DISCARD it. This
is sound and avoids baking a geometric assumption into the sampling.

⚠️ **30 mm exceeds the domain** (mesh ends at x ~ +16 mm PinShapes, ~ +25 mm cohort).
This is safe and was verified, not assumed: an exited particle gets `elem_id < 0`,
and `interpolate_scalar_single` returns `jnp.where(valid, val, 0.0)`
(benchmark_femuss_comparison.py:1274), so it accumulates **exactly zero** rather
than reading element 0. The accumulator freezes at its exit value, which is the
physically correct answer -- that particle has left the process zone.

### ⚠️ FIRST RESULT 2026-10-01 19:40 (rom_001, cohort) — mechanically fixed, but bypass is STRUCTURAL

| | Phase 3 (10 mm) | Phase 4 (30 mm) |
|---|---|---|
| steps | 267 | 800 |
| mean x | +3.88 mm | **+25.90 mm** |
| **bypass (lnPhi<1)** | 64.9 % | **58.0 %** |
| lnPhi median | 0.3097 | 0.5932 |
| lnPhi p99 | 225.7 | 760.0 |

**The 3x travel worked mechanically:** particles now advect 25.9 mm and genuinely
clear the tool, so the damage integral for particles that DO pass through the shear
layer is complete rather than truncated mid-wake (median 0.31 -> 0.59).

⚠️ **But bypass fell only 6.9 points (64.9 % -> 58.0 %).** For the cohort the null
population is **structural, not a duration artefact**: particles seeded off-axis in
y stream past and never enter the shear layer however far they travel. So:

* the **global median is still ~58 % diluted** -- regional metrics (O35,
  `stir_nobc`) remain the HEADLINE, not a refinement;
* a 4th or 5th multiple of travel would not help;
* ⚠️ **at 25.9 mm the cohort domain (~+25 mm) is already exhausted** -- a large
  fraction of these particles have EXITED and frozen their accumulators (by design,
  `jnp.where(valid, val, 0.0)`), so 30 mm is at or past the useful limit for the
  cohort.

### Confirmed across 5 cohort cases (19:50) — not an outlier

| case | steps P3->P4 | x_P3 | x_P4 | byp_P3 | byp_P4 | delta |
|---|---|---|---|---|---|---|
| rom_000 | 533 -> 1600 | +5.77 | +27.80 | 59.5 % | 48.4 % | **-11.1** |
| rom_001 | 267 -> 800 | +3.88 | +25.90 | 64.9 % | 58.0 % | -6.9 |
| rom_003 | 533 -> 1600 | +4.66 | +26.56 | 57.3 % | 52.4 % | -4.8 |
| rom_004 | 534 -> 1603 | +5.06 | +27.34 | 57.9 % | 52.3 % | -5.6 |
| rom_006 | 363 -> 1090 | +4.56 | +27.02 | 62.4 % | 55.1 % | -7.3 |

**Mean bypass reduction: -7.1 points** (range -4.8 .. -11.1), i.e. ~60 % -> ~53 %.

⚠️ **For the cohort this settles it: the bypass population is STRUCTURAL.** Tripling
the travel removed about a TENTH of it, not most of it. ~53 % of particles still
accumulate essentially nothing, so the global median remains roughly half-diluted
and the regional `stir_nobc` metric stays the headline.

**The decisive test is PinShapes**, where Phase 3 showed mean x = **-0.52 mm**
(genuine recirculation, not truncation) and 80-88 % bypass. Those are the later
array indices (25-44) and had not started at 19:50.

**Conclusion so far:** the duration fix was worth doing and improves the integral,
but it does NOT remove the dilution. To raise the signal-to-null ratio the seed
distribution has to change -- which is the (a) option deferred in favour of (b).
Both can coexist: keep the full span for the physics, and ALSO report a
stir-zone-seeded run for the ranking.

---

## O38 — the wake box: scoring only material that has been through the process

**Status:** implemented 2026-10-02, `make_wake_metrics.py` ->
`results/phase4_rk4_30mm_wake.csv` + `figs_wake/figW_<family>_yz.png`.

**✅ FIRST: the user's premise is confirmed.** Phase 4's 3x travel DID carry the
bundle clear of the tool. This supersedes the premature "bypass is structural"
reading recorded under O37, which was based on cohort cases only.

| | Phase 3 (10 mm) | Phase 4 (30 mm) |
|---|---|---|
| ps_A1 mean x | **-0.52 mm** | **+18.95 mm** |
| ps_A1 frac x > +7 mm | — | **97.2 %** |
| ps_A1 frac r < 7 mm | 22.5 % | **2.6 %** |
| PinShapes bypass | 80-87 % | 67-80 % (**-13 pts**) |
| cohort bypass | 58 % | 51 % (-6.6 pts) |

⚠️ The PinShapes drop (-13 pts) is **twice** the cohort's, which is the opposite of
what "structural bypass" predicted. PinShapes was genuinely RECIRCULATION-limited
and the longer run released it; the cohort was never trapped, only truncated.

### The box

| wall | where | why |
|---|---|---|
| x_min | **+7.0 mm** | after the shoulder: excludes the tool footprint and the ~2.6 % cap still orbiting the pin |
| x_max | max x at final step | the downstream front |
| y | full domain span | **kept** -- the advancing/retreating contrast lives here |
| z_min | domain z_min | **kept** -- the weld root is physically interesting |
| z_max | domain z_max **- 8 % of thickness** | drops the BC-contaminated top band (O35: p99 there is 5-16x the bulk) |

⚠️ **"2-3 particle layers" could not be used as specified**: seeding is RANDOM, so z
is continuous (6001 distinct z values in 100k particles) and there are no layers.
8 % of plate thickness = 0.36 mm (cohort) / 0.48 mm (PinShapes), comparable to what
a 2-3 layer band would be but mesh- and case-independent.

The box retains **~90 % of all particles**, so it is a mild, well-targeted exclusion
rather than a filter that throws the statistics away.

### Family averages over the box

| family | ln_mean | ln_med | ln_p99 | frac_failed |
|---|---|---|---|---|
| A-Flats | 6.342 | 0.2416 | 69.3 | 19.4 % |
| **B-Flutes** | **7.206** | **0.2524** | **80.9** | **21.0 %** |
| C-Threads | 6.333 | 0.2503 | 70.1 | 20.1 % |
| D-ConcavityTilt | 2.650 | 0.1746 | 39.9 | 9.0 % |
| cohort | 5.555 | 0.9148 | 49.4 | 17.6 % |

✅ **B-Flutes is highest on every statistic** -- mean, median, p99 and frac_failed --
which is the expected ordering (B has the deepest lobes, 20-32 % of pin radius, M4).
A and C are within 0.2 % of each other on the mean, consistent with their similar
shallow geometry. **This is the cleanest family separation obtained so far.**

⚠️ D-Concavity remains low (2.650 vs 6.3-7.2) even with the top band removed, so its
deficit is NOT purely the z_max BC artefact. Tilt changes which material passes
through the shear layer. O32 is therefore only PARTLY explained by O35.

---

## O39 — grid vs random seeding for continuous-field reconstruction

**Asked:** we used grid seeding before; is grid better than the current random
seeding if we want to extract a continuous 2D/3D damage field?

**Measured answer: it would not help, because advection destroys seed-time order
long before the final step.**

Final-time nearest-neighbour spacing of the 88,646 wake particles (ps_A1):

| | |
|---|---|
| mean NN distance | 0.1781 mm |
| median | 0.1755 mm |
| p99 | 0.3422 mm |
| max | 0.9556 mm |
| **coefficient of variation** | **38 %** |

A grid that survived transport would show CV ~ 0 %. **38 % means the flow has
already stretched and folded the bundle**, which is exactly what a stir zone does.
Seeding on a grid buys regularity at t=0 and loses it within a fraction of a
revolution.

**What grid seeding WOULD still buy, and it is not nothing:**

1. **Exact reproducibility** across runs and cases without relying on a fixed RNG
   seed -- useful when comparing two tool geometries particle-by-particle.
2. **A known, uniform inflow weight per particle.** Random seeding gives each
   particle equal weight only in expectation; a grid gives it exactly. For
   *integrating* a field (mass-weighted averages, deposited density) this removes a
   sampling-noise term. Poisson sampling at 100k particles has an expected largest
   empty-circle radius of ~1.9x the equivalent grid spacing -- small, but real.
3. **Cleaner-looking scatter figures** -- the speckle in `figW_*_yz.png` is Poisson
   noise, not physics.

**Recommendation:** keep random seeding for the statistics (the box averages are
unaffected -- 90k samples is far past where sampling scheme matters for a mean),
and switch to grid seeding **only if** we move to depositing a continuous field onto
a mesh, where the uniform-weight property genuinely helps. For the 2D projections,
the better fix is not grid seeding but **binned aggregation** (hexbin / 2D median),
which removes Poisson speckle regardless of seeding -- implemented as the
`--binned` option alongside the scatter.

---

## ⚠️ O40 — `stir_nobc` (O35) is a PHASE 3 metric and must not be used for Phase 4

At 10 mm travel, 22.5 % of particles were still inside r < 7 mm at the final step,
so a "stir zone" mask was a meaningful population. At 30 mm only **2.6 %** remain:
the stir_nobc counts across the 45 Phase 4 cases are **min 681, median 2013** out of
100,000 — against **~90,000** in the wake box.

`results/phase4_rk4_30mm_regional.csv` therefore reports `stir_med = 0.000` for many
cases. That is **not a bug and not a null result** -- it is an empty mask.

✅ **Use `results/phase4_rk4_30mm_wake.csv` (O38) for all Phase 4 reporting.** The
regional CSV is still useful for its `z_top` / `z_root` columns, which remain the
diagnostic for the boundary artefact.

---

## O41 — particle identity across exports is VERIFIED (and the resolution limit is particle count)

### Identity: safe to follow individual particles

Asked because the current runs use RANDOM seeding and identity was certain only
for grid seeding. **Random vs grid makes no difference** -- `ParticleID` is a
POSITIONAL INDEX created once at seeding (`np.arange(n)`, run_tracking.py:774) and
passed unchanged to every export. The kernel is a `vmap` over a fixed array and
never reorders. Three independent checks on ps_A1 (167 exports, 100k particles):

| check | result |
|---|---|
| ParticleID identical across consecutive files | ✅ True |
| row-wise displacement over 50 steps | median **0.18 mm** |
| …vs a shuffled control | **12.98 mm** (72x separation) |
| rows where lnPhi DECREASED | **0 / 100,000** |

⚠️ The last one is the strongest: lnPhi can only accumulate, so **any** row swap
would surface as a decrease. It is a physical invariant, not a heuristic.

### Resolution: the limit is particle count, not bin size

The yz field was criticised as too coarse even at the seed-grid resolution
(`SEED_GRID="60 120 40"`). Measured on ps_A1's 88,646 wake particles over ~165 mm2
(mean spacing **0.18 mm**):

| grid | dy [mm] | median n/cell | cells empty |
|---|---|---|---|
| 120x40 (seed grid) | 0.250 | 18 | 0 % |
| 200x80 | 0.150 | 5 | 0.8 % |
| 400x160 | 0.075 | 2 | **26 %** |
| 600x240 | 0.050 | 1 | **54 %** |

⚠️ **No histogram grid can be both fine and populated** -- that is a property of the
sample, not of the binning. Replaced with a **k-nearest-neighbour median**, which
decouples display resolution from cell occupancy: each output point is the median
of its k nearest particles wherever they lie.

At **600x240 (dy = 0.05 mm, 12x finer in y than the seed grid)** with k=16 the
median support radius is **0.097 mm**, nothing is blanked, and it costs 0.1 s per
panel. `--knn 8` trades smoothness for sharpness.

This revealed a **sharp advancing-side boundary at y ~ +3 mm stepping outward with
depth** that the coarse bins had smeared away entirely.

---

## O42 — following individual particles: where and when damage is acquired

`make_trajectory_figures.py` -> `figs_traj/figT_<case>_3d.png` + `_time.png`.
Read-only on the existing 167-step exports; no re-run.

**Why this beats phase-binned averaging for the "how does the tool make damage"
question.** Phase-binning needs fine export frequency to resolve intra-revolution
structure, and at `--export-freq 50` the cohort gets only **0.8 exports per
revolution** (aliased). Following a SINGLE particle has no such requirement -- you
see its own history at whatever cadence exists.

**Stratified selection**, because a random sample would mostly draw the ~60 %
bypass population and show flat lines:

| stratum | definition | what it shows |
|---|---|---|
| high | top decile of final lnPhi | went through the shear layer |
| mid | median decile | the typical processed particle |
| low | bottom decile | **the control** -- what "no damage" looks like |
| jumpy | largest single-step increment | isolates WHERE damage is picked up |

⚠️ **Not a random sample; not a population statistic.** It is a mechanism
illustration. Population numbers live in the wake CSV (O38).

The time figure pairs accumulated lnPhi with the same particles' **distance from
the tool axis**, so a step in damage can be read directly against a close approach.

### ✅ RESULT — damage is acquired in DISCRETE CAPTURE EVENTS, not gradual shear

First application, ps_A-FlatsVariations_A1 (167 exported states, 50 tool
revolutions). Four findings, all visible directly in the paired figures:

1. **Capture, not continuous shearing.** A particle approaches from upstream, is
   **captured into orbit around the pin**, circles several times while
   accumulating, then releases downstream. The `jumpy` trace rises **three orders
   of magnitude in one near-vertical step** (0.06 -> 100) and is then FLAT for the
   remaining ~45 revolutions.

2. **Final damage is essentially a COUNT of capture events.** The top-decile
   traces are staircases, with distinct bursts at steps ~1000, ~3300, ~5200 --
   each a separate close approach, separated by long flat stretches.

3. **Every jump coincides with r dropping below the 7 mm shoulder line.** The
   damage panel and the radius panel are locked together, which makes the
   association causal rather than suggestive.

4. ✅ **The bypass population is confirmed as UNPROCESSED, not weakly damaged.**
   The bottom-decile control sits flat at ~0.1 for the whole run and never
   approaches the tool. This retroactively justifies excluding it (O38) rather
   than averaging it in.

5. **The capture orbits sit at r ~ 3-5 mm** -- between the pin (2.4 mm) and the
   shoulder (7 mm), i.e. in the shear layer, NOT against the pin surface. The
   top-view panel locates the shear layer directly, without needing a velocity
   field.

⚠️ **Implication for the damage model.** Rice-Tracey and Cockcroft-Latham are rate
laws integrated along the path, and the result says the integral is dominated by a
few short, intense episodes rather than by the long approach. That means:

* the metric is sensitive to how many times a particle is **re-entrained**, which
  is a property of the TOOL GEOMETRY -- which is exactly what we want to rank;
* but it is also sensitive to the **step size inside the capture**, because that is
  where the integrand is largest and most rapidly varying. This is where the
  order-4 quadrature (O36) earns its -17.9 % correction to the tail.


---

## ⚠️ O43 — empty regions in the wake: D-Concavity's is very probably NUMERICAL

> **⚠️ REVISED 2026-10-02, same day.** The first version of this entry read the
> D-Concavity void as a direct observation of a filling failure. **That reading does
> not survive testing.** The user asked whether trapping or artificial divergence
> could explain it; both were tested and the evidence supports them. The original
> framing is replaced below, not merely annotated, because it would have been
> presented as the headline result.

### Two different empty regions

| | A-Flats (A2, A2old, A3, A3old) | D-Concavity (all 5) |
|---|---|---|
| void area | 0.7 - 1.6 mm2 | **20.4 - 25.1 mm2** |
| location | deep, advancing-side boundary | large, central |
| damage at the rim | 23.6 - 29.5 | 29.9 - 32.2 |
| **trapped at tool (r<7mm, final step)** | 2.6 - 5.2 % | **18.7 %** |
| **wake areal density vs seed plane** | 535 / 556 (-4 %) | **350 / 454 (-23 %)** |
| area at threshold 0.30 -> 0.90 mm | 2.64 -> 0.02 (**99 % lost**) | 30.2 -> 14.1 (53 % kept) |

⚠️ Note the rim damage is comparably HIGH in both (29.5 vs 32.2), so "D is surrounded
by low damage" is true of the wider field but NOT of the immediate rim. Rim damage
does not discriminate the two cases; trapping and density do.

### Why D's void is probably an artefact

1. **Trapping that does not drain.** 18.7 % of D's particles are still inside the
   shoulder radius at the final step, vs 2.6-2.8 % for every other family. Over time
   the trapped fraction rises to 26.2 % (step 3970) then **plateaus at 18.7 %**. In a
   steady incompressible flow, material entering the tool region must emerge;
   a persistent reservoir is not physical.

2. **The depletion is GLOBAL, not local.** D's wake holds 350 particles/mm2 against a
   seed density of 454 -- a **23 % deficit across the whole cross-section**. A
   physical cavity would leave its surroundings at normal density. A1: 535 vs 556.

3. **Every particle is accounted for.** D1: 62,913 wake + 18,705 trapped + 18,382
   never past the tool + 7,902 top band = 100,000 exactly. The void is particles that
   **never arrived**, not material displaced from a region.

**Mechanism:** most likely artificial divergence in the interpolated velocity field.
The flow is incompressible so a material volume is conserved, but the P1-interpolated
field is only *discretely* divergence-free; error accumulates over a ~50-revolution
path and is worst for the tilted geometry, where the tool surface is also not
mesh-aligned (so no-penetration is satisfied less exactly).

### The A-Flats voids are NOT dismissed, but are at the resolution limit

They sit where a defect would be expected -- at the advancing-side boundary,
surrounded by the highest damage in the section -- which is physically suggestive.
But **99 % of their area disappears** as the blanking radius goes 0.30 -> 0.90 mm,
i.e. they are marginal at 100k particles (mean spacing 0.18 mm).

### What would settle it

* **Re-run one D and one A case at 4x particle count.** A physical cavity keeps its
  area; a sampling artefact shrinks.
* **Measure the divergence of the interpolated velocity field directly**, and check
  whether particle count is conserved across a control surface around the tool.
* If trapping is confirmed, check whether the level-set / no-penetration treatment
  needs tightening for tilted tools.

⚠️ **Until then, no empty region may be presented as a predicted defect.** The method
produces *candidate* empty regions; distinguishing a cavity from a
particle-conservation artefact is unfinished work.

### This also revises O32 and the D damage number

D's low mean lnPhi (2.650 vs 6.3-7.2) is **not** evidence that its tool damages
material less. It is a consequence of 37 % of its particles never reaching the wake.
**D must be excluded from the family damage comparison** until the artefact is
resolved.

---

## O44 — D-Concavity exported at every 10 steps, not 50 (and 33 exports are missing)

Found 2026-10-02 while generating trajectory figures.

| family | exports | expected at freq 50 | actual spacing |
|---|---|---|---|
| A-Flats, C-Threads | 167 | 167 | 50 |
| B-Flutes (B2) | 201 | 201 | 50 |
| **D-Concavity** | **782 - 803** | **167** | **10** |

The SLURM log for every D task says `export : vtu every 50 steps`, so the override
did not come from the array script. Step indices are multiples of 10, and
`damage_models.npz` is unaffected (it is written once at the end).

⚠️ **33 of the expected 828 exports are missing**, clustered after step 6190
(spacing histogram: 767 gaps of 10, 21 of 20, 3 of 30, 2 of 40, 1 of 7). Consistent
with write pressure late in the run rather than corruption.

**Impact: low but worth knowing.**
* The final-state analyses (wake box, family ranking, void areas, O43) all use
  `damage_models.npz` and are **unaffected**.
* Trajectory figures read the actual step index from each filename, so gaps are
  handled correctly -- D simply has a few holes in its time axis, at 5x the time
  resolution of the other families.
* D's per-case output is ~2.6 GB against ~0.55 GB elsewhere, which explains part of
  the 24 GB total.

**Not yet explained:** why the frequency differed. Worth checking before the next
campaign, since an unintended 5x export rate costs wall time and disk.

---

## O45 — separating the TWO defect indicators: particle density vs accumulated damage

**Raised by the user 2026-10-02:** do the 2D damage projections depend on particle
DENSITY rather than on real accumulated damage? If so, particle tracking and damage
integration are not two independent predictors and cannot be compared.

**The concern was justified, and specifically for the case under dispute.**

### The confound, measured

The kNN-median estimator in `make_wake_metrics.py` is not density-independent: the k
nearest neighbours of a sparse point are drawn from a WIDER region, so the estimate is
smoothed over more space exactly where particles are scarce. Rank correlation between
local density and reported damage, inside the wake box:

| case | rho(density, damage) | density range |
|---|---|---|
| A-Flats A1 | -0.063 | 230 .. 1125 /mm2 |
| B-Flutes B2 | -0.073 | 202 .. 1129 /mm2 |
| **D-Concavity D1** | **-0.291** | **32 .. 879 /mm2** (27x) |

So for A and B the confound is weak, but **D's figure was partly rendering its
density deficit as a damage pattern** -- and D is the case whose void is in dispute
(O43).

### The fix: `make_density_vs_damage.py` -> `figs_compare/figX_<case>_*.png`

Three panels on a shared grid, with **no shared estimator**:

1. **DENSITY** -- particles/mm2. A pure count. The particle-tracking indicator alone.
2. **DAMAGE** -- median lnPhi on a **FIXED-AREA bin**, not kNN, so every reported cell
   is averaged over the same physical area and cannot be influenced by how far its
   neighbours lie. Cells below 8 particles report **nothing** -- "not measurable" is
   kept distinct from "low".
3. **AGREEMENT** -- both converted to within-case percentile ranks (the raw units are
   unrelated), showing where the two indicators coincide.

✅ **It works:** D's rho falls from **-0.291 to +0.015**. The confound was almost
entirely the variable-support bias.

### ⚠️ This REVISES O43 again — A-Flats is the better defect candidate, not D

With the indicators separated:

| case | void mm2 | density CV | void isolation | rim damage / median |
|---|---|---|---|---|
| A-Flats A1 | 0.15 | 0.347 | 0.25 | 67 |
| **A-Flats A2** | **3.64** | **0.353** | **0.91** | **64** |
| **A-Flats A2old** | **2.20** | **0.352** | **0.89** | **84** |
| B-Flutes B2 | 0.55 | 0.350 | 0.38 | 60 |
| D-Concavity D1 | 32.90 | 0.380 | 0.67 | 80 |
| cohort rom_000 | 15.77 | **0.625** | 0.56 | **8.9** |

* **A2 and A2old have the cleanest signature**: a SINGLE compact void (isolation 0.89
  -- 0.91, i.e. ~90 % of the empty area is one connected blob), in an otherwise
  UNIFORMLY filled cross-section, sitting directly beneath the brightest damage band
  (rim damage 64-84x the case median). **Both independent indicators point to the
  same place.** That is a defect signature.
* **D1's void is large but its surroundings are globally depleted** (O43: 18.7 %
  trapped, 23 % under-dense). The damage arc does drape over the void, so it is not
  pure noise -- but the density evidence for trapping stands.
* **rom_000 is the clear negative control**: density CV 0.625 (nearly double everyone
  else's ~0.35) and rim damage only 8.9x the median -- a ragged, scattered void with
  no damage association, i.e. a sampling artefact.

**Revised reading: the A-Flats voids are the strongest defect candidates in the
dataset, and were previously dismissed as "at the resolution limit" because the kNN
blanking threshold hid them.** D's remains ambiguous: a real void may exist, but it
sits inside a genuinely depleted region, so its area cannot be trusted.

### What this enables

This is now a **genuine two-method comparison**: particle tracking (density) and
damage integration (lnPhi) as independent predictors of the same defect. Where they
agree, confidence is high; where they disagree, one of them is wrong and the
disagreement is diagnostic. That was not possible while a single estimator produced
both.

⚠️ Still outstanding: the convergence test (4x particle count). Density CV ~0.35 at
100k particles is a sampling floor, not a physical feature.


---

## ⚠️⚠️ O46 — void area is PREDICTED BY TRAPPING across all 20 tool cases (rho = +0.98)

**This supersedes the O45 conclusion that A-Flats A2 is a credible defect candidate.
It is almost certainly not. Found 2026-10-02, a few hours after O45.**

### The measurement

Across all 20 PinShapes cases, empty (void) area in the wake cross-section against
the fraction of particles still inside r < 7 mm at the final step:

| | Spearman rho |
|---|---|
| void area vs **trapped %** | **+0.980** |
| void area vs trapped/wake damage ratio | **-0.932** |
| void area vs trapped %, **D cases EXCLUDED** | **+0.954** |
| void area vs damage ratio, **D cases EXCLUDED** | **-0.879** |

⚠️ **Excluding D does not weaken it.** The relationship holds *within* A, B and C,
so it is not "D is different" -- it is a property of the whole dataset.

### Per-case, inside the A-Flats family alone

| case | trapped % | trapped lnPhi | trap/wake ratio | void mm2 |
|---|---|---|---|---|
| A1 | 2.6 % | 259.8 | 1003 | **0.15** |
| A4old | 2.5 % | 223.6 | 897 | **0.15** |
| A4 | 2.6 % | 221.7 | 869 | **0.19** |
| A3old | 3.4 % | 88.0 | 370 | 1.49 |
| A3 | 3.8 % | 16.6 | 67 | 1.62 |
| A2old | 4.1 % | 12.0 | 54 | 2.20 |
| **A2** | **5.2 %** | **9.5** | **39** | **3.64** |

A monotone ladder: more trapping -> less damage on the trapped particles -> bigger
void. **A2, which O45 promoted as the clearest defect candidate, is simply the
A-family case with the most trapping.**

### Why the trapped-damage ratio is the diagnostic

| family | trapped % | trap/wake damage ratio |
|---|---|---|
| B-Flutes | 2.7 - 3.0 % | **868 - 1025** |
| C-Threads | 3.2 - 3.4 % | 638 - 769 |
| A-Flats | 2.5 - 5.2 % | **39 - 1003** (bimodal) |
| **D-Concavity** | **17.1 - 18.7 %** | **8 - 23** |

A particle genuinely held in the shear layer should accumulate damage **hard** --
that is what the capture mechanism (O42) shows, and B/C deliver ratios of 600-1000.
D's trapped particles sit at r ~ 3.65 mm and accumulate almost **nothing** (ratio
8-23). They are being circulated by the interpolated field **without experiencing the
strain rates that should accompany being that close to the tool**.

That is the signature of a numerical trap, and the same signature appears -- weaker
-- in A2/A2old/A3.

### Revised conclusion

⚠️ **No void in this dataset is currently a credible defect prediction.** Void area is
explained by a tracking pathology (rho = +0.98) rather than by weld physics. The O43
and O45 readings were both wrong, in opposite directions:

* O43 called D's void a direct observation of a filling failure -- wrong.
* O45 called A2's void a credible defect because two indicators agreed -- also wrong.
  **Both indicators are downstream of the same tracking failure**, so their agreement
  was not independent evidence. Density is directly depleted by trapping, and the
  damage field is shaped by which particles survive to the wake.

### What would settle it

1. **Conservation test.** ✅ **IMPLEMENTED** as `make_conservation_test.py`, reading
   the trajectory caches. Needs no re-run.

   **Reference result, ps_A-FlatsVariations_A1 (healthy):**

   | | |
   |---|---|
   | particles inside r < 7 mm | peak **22,914** -> end **2,584** (11 % of peak) |
   | late-run trend | **-88 particles per export** -> DRAINING |
   | ever entered the volume | 44,632 |
   | still inside at the final step | **2,584 (5.8 %)** |

   That is what conservation looks like: the volume fills as the bundle arrives,
   then drains as material passes through and emerges.

   ### ✅ RESULT 2026-10-02 — confirmed on the decisive metric, my prediction was
   partly wrong on the other

   | | A1 (healthy) | **D1** | prediction |
   |---|---|---|---|
   | end / peak | 0.113 | **0.711** | — |
   | **still inside / ever entered** | **5.8 %** | **49.1 %** | "far above 5.8 %" ✅ |
   | late-run trend per export | -88 | -8.5 | "near zero or positive" ⚠️ |

   ⚠️ **I predicted D would show a flat or rising trend. It is still draining.** That
   part of the prediction was wrong and is recorded as such.

   **But the per-export rate was the wrong normalisation** -- D exported every 10
   steps, A1 every 50 (O44), so D's per-export number is deflated 5x. Per STEP:

   | | drain rate | particles left | steps to empty | **in revolutions** |
   |---|---|---|---|---|
   | A1 | -1.76 /step | 2,584 | 1,465 | **8.8** |
   | **D1** | **-0.73 /step** | **18,705** | **25,662** | **154.6** |

   The run itself is 8,267 steps = **50 revolutions**. So D would need **three times
   the entire run again** to clear, while draining at under half A1's rate and holding
   7x more material.

   ✅ **The conservation failure is confirmed**: D retains **49.1 %** of everything
   that ever entered the control volume, against 5.8 % for A1 -- an **8.5x**
   difference. The control volume is not approaching steady state on any timescale
   relevant to the simulation.

   Figure: `figs_compare/figC_conservation.png` -- both curves peak together; A1 falls
   to 0.11, D1 **plateaus at 0.71**.
2. **Divergence of the interpolated field**, measured on the mesh.
3. **Convergence at 4x particle count**: a physical cavity keeps its area.
4. Check whether the level-set / no-penetration treatment leaks for tilted tools.

⚠️ Until (1) is done, the void measure must not appear in any presentation as a
defect indicator. The DAMAGE ranking (B-Flutes highest) is unaffected -- it is
computed over the wake population and does not depend on the void.


---

## O47 — the D trapping is LOCALISED to a thin annulus, not a control-volume artefact

Checked 2026-10-02 as the obvious alternative explanation for O46: a tilted tool
sweeps a wider footprint, so maybe `r < 7 mm` legitimately encloses more material for
D and the "trapping" is just a badly-sized control volume.

**It is not.** Cumulative particle fraction inside radius R at the final step:

| R [mm] | A1 | B1 | **D1** |
|---|---|---|---|
| 2.4 (pin) | 0.17 % | 0.11 % | **0.14 %** |
| **4.0** | 1.29 % | 1.36 % | **11.75 %** |
| 5.5 | 2.02 % | 2.12 % | 15.50 % |
| 7.0 | 2.58 % | 2.69 % | 18.70 % |
| 8.5 | 2.96 % | 2.99 % | 20.39 % |
| 10.0 | 3.69 % | 3.65 % | 22.17 % |

✅ **At the pin radius all three agree** (0.11 - 0.17 %), so there is no systematic
offset and no leakage into the tool itself.

⚠️ **The entire excess appears between 2.4 and 4.0 mm**: D gains 11.6 percentage
points across that annulus against ~1.2 for A and B -- a **9x excess** in a 1.6 mm
band. Beyond 4 mm the three curves rise in parallel.

This matches the measured capture-orbit radius for D's trapped particles
(median r = 3.65 mm, O46). The trapping is therefore **a localised failure in a thin
annulus just outside the pin**, not a geometric consequence of tilt widening the
footprint.

**Narrows the candidate causes.** Something about the interpolated field in the first
~1.6 mm outside the pin holds particles in closed orbits for the tilted geometry.
Worth checking there specifically:
* the level-set / no-penetration treatment where the tool surface is NOT mesh-aligned;
* whether the element-local velocity reconstruction is divergence-free in that band;
* whether the tilt puts the tool surface mid-element, so the sampled velocity blends
  tool and material nodes.

---

## ⚠️⚠️ O48 — "D-Concavity is the only TILTED family" is FALSE. No case has tilt.

**Checked 2026-10-02 while looking for the mechanism behind O46/O47.**

```
Tilt angle: 0.0    for ALL 20 PinShapes cases (A1..A4old, B1..B4, C1..C4, D1..D4)
```

A1.gid and D1.gid `.info` files are **byte-identical**. The only difference between
the cases is the tool **STL geometry**.

### What this invalidates

I inherited "D is tilted" from the FOLDER NAME `D-ConcavityTilt` and never checked
it. It was then used as the physical explanation for D's anomalies in **O32, O43,
O45, O46 and O47**, and in all three presentation documents:

* "D is the only family with tool tilt (2°)" -- **there is no 2° tilt anywhere**
* "tilt changes which material passes through the shear layer" -- unsupported
* "the tool surface is not mesh-aligned for tilted tools" -- unsupported
* "the tilt diverts particles around the tool" -- unsupported

⚠️ **Every one of those was an explanation invented to fit an observation, then
repeated until it read like a finding.** The lesson is specific: a folder name is not
a case parameter.

### What SURVIVES unchanged

All of it is measurement, none of it depended on tilt:

| | |
|---|---|
| D trapped fraction | **17.1 - 18.7 %** vs 2.5 - 5.2 % elsewhere |
| trapped/wake damage ratio | **8 - 23** vs 638 - 1025 for B/C |
| void area vs trapped fraction | **rho = +0.980** (+0.954 excluding D) |
| trapping annulus | **r = 2.4 - 4.0 mm**, 9x excess, agrees at the pin radius |

### What is now OPEN

**Why does D behave differently at all?** The remaining candidates are all geometric,
in the tool STL:

1. **Concavity.** The family name's other half. A concave pin face may create a
   closed recirculation just outside the pin -- which would match the r = 2.4-4.0 mm
   annulus exactly. ⚠️ This is a HYPOTHESIS, not a finding. It needs the STL measured,
   not assumed.
2. **STL resolution.** D1 is 558 KB and D3 is 5.5 MB, against A1 at 414 KB and B1 at
   2.7 MB -- so the family spans an order of magnitude in triangle count, yet all five
   D cases trap 17-19 %. That argues against a pure meshing artefact.
3. Whether the pin surface cuts through elements differently for this geometry.

### ✅ The concavity hypothesis was tested and FAILED

Pin radius measured from the STLs in fixed z bands inside the plate (z = -6..0 mm):

| case | z=-5 | z=-4 | z=-3 | z=-2 | z=-1 |
|---|---|---|---|---|---|
| B2 | 2.21 | 2.34 | 2.49 | 2.62 | 2.77 |
| C1 | 2.32 | 2.46 | 2.60 | 2.74 | 2.88 |
| **D3** | **2.21** | **2.34** | **2.49** | **2.62** | **2.77** |
| **D4** | **2.32** | **2.46** | **2.60** | **2.74** | **2.88** |

⚠️ **D3's pin profile is IDENTICAL to B2's at every depth, and D4's is identical to
C1's** -- yet D3 traps **18.5 %** and B2 traps **2.8 %**. A1/A2/D1/D2 return nan
because their STLs are too coarse to carry mid-pin surface (the known M4 limitation),
so they cannot be compared.

**So the trapping is NOT explained by the pin profile**, and "concavity" joins "tilt"
as a folder-name inference that does not survive measurement.

### Status: the cause of D's trapping is UNKNOWN

What is established:

* all five D cases trap **17.1 - 18.7 %** of particles against 2.5 - 5.2 % elsewhere;
* the excess is localised to **r = 2.4 - 4.0 mm** and the cases agree at the pin radius;
* trapped particles accumulate almost no damage (ratio 8 - 23 vs 638 - 1025);
* it is **not** tilt (all cases 0.0 deg) and **not** the pin radius profile (D3 == B2).

Remaining candidates, none tested:

1. Something in the **velocity field** of the D cases rather than the geometry -- i.e.
   the FOM solution differs, not the tool. ⚠️ This is now the leading candidate
   BECAUSE the geometry has been excluded, and it is checkable: compare the
   divergence and the near-pin velocity profile of D3 against B2 directly.
2. A difference in the **mesh** around the pin (element size or quality) not visible
   in `.info`.
3. Azimuthal pin features (flutes/threads/flats) that the radial r95 profile averages
   away -- D3 has 330k vertices, so it is not short of detail.

⚠️ **The correct statement for now:** *"all five D cases trap 17-19 % of particles in
a thin annulus just outside the pin. It is not tilt and not the pin radius profile.
The cause is not identified."* Do not offer a mechanism.

---

## ⚠️ O49 — every D case's FOM output carries ANOTHER case's filename stem

Found 2026-10-02 while testing the velocity-field hypothesis for O48.

| case folder | STL | **post/ stem** |
|---|---|---|
| D-ConcavityTilt/**D1** | D1.stl | **A1** |
| D-ConcavityTilt/**D2** | D2.stl | **A2** |
| D-ConcavityTilt/**D3** | D3.stl | **B4** |
| D-ConcavityTilt/**D4** | D4.stl | **C2** |
| A1, B1, B2, B4, C1 | matching | matching |

Only the D family is affected; every other case's post stem matches its folder.

### ✅ The data is NOT duplicated -- checked, not assumed

`D3.gid/post/B4_99.pvtu` and `B4.gid/post/B4_99.pvtu` have the **same md5** -- but a
`.pvtu` is only a header listing its pieces, so that proves nothing on its own. The
actual data pieces differ:

| | D3 (stem B4) | B4 |
|---|---|---|
| `B4_99_0.vtu` size | **10,445,121** | 8,554,488 |
| md5 | 109e1b11… | a5d72548… |
| piece-0 points | 18,710 | 20,677 |

Pieces 1, 7 and 33 also differ. **D3 is its own simulation on its own mesh**; only
the output NAMING was inherited, presumably from copying a template case directory.

### Why this matters anyway

1. ⚠️ **The array script picks the PVTU by globbing `post/*_[0-9]*.pvtu`** and derives
   the stem from the filename. It therefore tracked the right fields -- but by luck,
   not design. Any future script that assumes `post/<casename>_*.pvtu` will silently
   read the wrong case or find nothing.
2. It explains why `.info` is byte-identical between A1 and D1 (O48): the D
   directories were **copied from other cases** and then given new STLs. So `.info`
   describes the SOURCE case, not D -- which is exactly why its `Tilt angle: 0.0`
   and mesh settings cannot be trusted as a description of D.
3. ⚠️ **It makes the O48 conclusion weaker, not stronger.** "D3's pin profile equals
   B2's" came from the STL, which IS correctly named -- that part stands. But any
   inference drawn from a D case's `.info` is unreliable.

### Does it explain the trapping (O46/O47)?

**No, and it does not excuse it either.** The tracked velocity fields are genuinely
D's own. The trapping remains unexplained -- but the provenance of these case
directories is now a question for the colleague who produced them, and worth asking
before more effort goes into a numerical explanation.

**Suggested question for the colleague:** were the D-ConcavityTilt cases set up by
copying A1/A2/B4/C2 and replacing the tool STL? If so, (a) does anything else in the
setup still refer to the source case, and (b) is the mesh refined around the D pin or
around the SOURCE case's pin? A mesh refined for a different tool would plausibly
produce exactly the near-pin annulus artefact we measure.

---

## ⚠️ O50 — the D trapping REPRODUCES in an independent, earlier particle-tracking run

Found 2026-10-05 while building the PT-vs-damage comparison figures.

Each case folder carries a `post_pt/` directory from an **earlier, separate**
particle-tracking campaign: 288,000 **grid**-seeded particles, 8,000 steps, a
different run with different seeding and a different configuration from our Phase 4.

Final-state trapped fraction (particles still inside r < 7 mm):

| case | **old PT run** (288k grid) | **our Phase 4** (100k random) |
|---|---|---|
| **D3** | **19.98 %** | **18.5 %** |
| B2 | 6.61 % | 2.8 % |
| C1 | 2.55 % | 3.4 % |

✅ **D's anomaly is NOT an artefact of our pipeline.** An independent run, with
different seeding, reproduces it to within ~1.5 points. The cross-section figure
(`figs_compare/figP_D-ConcavityTilt_pt_vs_damage.png`) shows the **same central void**
in both rows.

### What this changes

⚠️ **O46/O47 attributed the trapping to "a tracking pathology".** That wording is now
too narrow: *our* tracker is not the cause, because another tracker does the same
thing on the same velocity field.

**The remaining explanations are upstream of any particle tracker:**

1. **The D velocity field itself** has a near-pin structure that holds material —
   either physically, or as a FOM artefact. Both trackers would faithfully reproduce
   either.
2. **The mesh around D's pin** — O49 found the D case directories were copied from
   A1/A2/B4/C2 with only the STL replaced, so the mesh may be refined for the SOURCE
   case's pin. Both trackers inherit that.

✅ **This makes the physical reading more plausible than it was**, though still not
established: if the D tool genuinely produces a closed recirculation that material
cannot escape, that IS a filling failure, and the void is real. Distinguishing that
from a mesh/field artefact needs the FOM side, not the tracker.

⚠️ **Do not over-correct.** O46's measurement stands (void area vs trapped fraction,
rho = +0.98) and it still means the void MEASURE is not independent of the trapping.
What has changed is that the trapping is no longer attributable to our code.

**Next test, in order of cost:** compare the near-pin velocity profile and divergence
of D3 against B2 directly on the FOM fields; then check whether D's mesh refinement
follows its own pin.


---

## O51 — the wake box upstream wall moved 7.0 -> 7.4 mm (tool rind excluded)

Raised 2026-10-05: does the wake box still contain the tool and particles attached
to it? Checked in the data.

The cut was at **x >= 7.0 mm**, exactly the shoulder radius. Geometrically that
guarantees r >= 7 mm, so no particle is *inside* the shoulder -- but it leaves a thin
rind of particles still hugging the tool, and they are not representative:

| x band [mm] | particles (B2) | median lnPhi |
|---|---|---|
| **7.0 - 7.2** | 30 | **7.476** |
| 7.2 - 7.5 | 67 | 2.995 |
| 7.5 - 8.0 | 216 | 0.653 |
| 9.0 - 10.0 | 1838 | **0.161** |

A **46x** contrast between the rind and the bulk.

### Why it mattered for the figures but not the tables

* **Global statistics: negligible.** Moving the cut 7.0 -> 7.4 mm removes ~100 of
  ~88,000 particles and shifts the median by **-0.02 % to -0.19 %** across all 20
  tool cases. The family ranking and `*_wake.csv` were never materially affected.
* **Cross-section fields: not negligible.** The kNN-median estimator gives each output
  point the median of its 16 nearest particles, so a handful of lnPhi ~ 7.5 particles
  at specific (y,z) locations can dominate those cells while vanishing from the global
  median.

✅ **Applied: `X_AFTER_TOOL_MM = 7.4`** in `make_wake_metrics.py`,
`make_pt_vs_damage.py` and `make_density_vs_damage.py`, with the reasoning recorded
in the first of those so the constant is not later "simplified" back to 7.0.

⚠️ The PT panel of the comparison figures is unaffected: it uses the fixed x = 15 mm
slice, which is already well clear of the tool.

✅ **The downstream wall needed no change** -- there is no upper x cut in the code, so
the box already runs to the farthest particle (x_max = 35.46 mm for B2), which is what
`wake_mask()` records as `box_x_max`.

---

## ⚠️ O52 — "under the shoulder sigma_1 < 0 almost everywhere" was FALSE. Corrected.

Raised by the user 2026-10-06: the Cockcroft-Latham slide was hard to follow and
looked likely to draw challenge. Checked against our own nodal fields. **It was
wrong**, and the error is instructive.

### The claim and the measurement

| | claimed | measured (nodes with edot>1e-3, levelset>=0, r<7mm) |
|---|---|---|
| frac sigma_1 < 0 under the shoulder | "almost everywhere" | **4.9 - 5.9 %** (8 cases) |
| median sigma_1/sigma_eq | implied negative | **+0.80** |

So the Macaulay bracket **almost never switches off**, and the stated mechanism
("C stops accumulating under the tool, resumes in the wake") does not occur.

### Why — and why it is NOT a contradiction of the literature

sigma_1 = sigma_m + sigma_eq * s1_hat. FSW flow is strongly **shear**-dominated:

| term, median under the shoulder (A1) | value |
|---|---|
| s1_hat (deviatoric) | **+0.87** |
| eta = sigma_m/sigma_eq (volumetric) | **-0.05** |
| sigma_1/sigma_eq | **+0.80** |

The deviatoric term dominates, so sigma_1 stays tensile even where the **mean** stress
is compressive -- which it is at **55.3 %** of those nodes.

⚠️ **I conflated two different quantities.** The literature
(`FSW_void_prediction_literature_review.md` §5, Saby et al. 2015; Zapara et al. 2013)
says the **hydrostatic / mean** stress is compressive under the shoulder, which our
eta measurement **confirms** (negative at 55 %). It never claimed the **largest
principal** stress is negative. I carried the first statement over to the second.

### The consequence is a BETTER argument than the one it replaces

The two laws are **not** two versions of the same indicator:

* **Rice-Tracey** reads the **volumetric** state through exp(3*eta/2) and **is**
  suppressed by compression -- consistent with the void-closure literature.
* **Cockcroft-Latham** here reads mainly the **deviatoric** state and is not.

They interrogate different parts of the stress tensor. That is a stronger
justification for running both than "agreement between two laws is evidence", which
was the previous framing.

### Fixed in

`damage_implementation_talk.md` (split into two slides),
`DAMAGE_REPORT.md` §3.3, `DAMAGE_PRESENTATION_GUIDE.md`,
`PRIMER_fsw_physics_for_me.md` §8.

---

## O53 — meeting deck: what was hidden, and one physics argument that needed fixing

`damage_talk_meeting.md` (71 slides) is the ordered meeting version;
`damage_talk_FULL_reference.md` keeps all 90 for future use. Hidden slides are parked
in an HTML comment at the end of the meeting file, not deleted.

### Hidden for this meeting

| slides | why |
|---|---|
| "accumulated damage does not reproduce the eta result" | the conclusion is not clear enough to defend; superseded by the wake-box result anyway |
| "Root cause: most particles never reach the tool" | user is not confident it is true; see the correction below |
| "Regional metrics recover the signal" | the terms (bypass, stir_nobc) were never defined for the audience |
| "What the trajectories establish" + "...and two more" | see the correction below |
| the whole void-credibility block (A2 ladder, density-vs-damage, "none are credible") | these are judgments drawn from the trapping artefact; better to show the results and let the audience reach them |
| the three placeholders | not yet generated |

### ⚠️ The physics argument for hiding the trapping slides does NOT hold as stated

The user's reason was: *"in an incompressible flow (as FSW) the particles should not
trap and rotate around the tool, and each streamline should pass the tool."*

**Half right, and the correct half matters.**

❌ **Incompressibility alone does NOT forbid closed streamlines.** div(u)=0 forbids
accumulation of *volume*, not closed orbits. Standard counter-examples in steady
incompressible flow: solid-body rotation (every streamline closed), a line vortex,
lid-driven cavity cells, Hill's spherical vortex. Presenting "FSW is incompressible,
therefore no trapping" would invite exactly the objection it is meant to pre-empt.

✅ **But a persistent NON-DRAINING reservoir in a steady through-flow IS unphysical**
-- and that is what we measured: 49.1 % of everything that entered is still inside at
the end, against 5.8 % for a healthy case, with a drain rate implying ~155 more
revolutions to clear against a 50-revolution run.

⚠️ **And the literature explicitly reports near-tool stagnation as REAL.**
Morisada et al. (2015), direct X-ray real-time imaging (lit. review §3): defects
correlate with *"stagnation on the AS"*, and measured flow velocity on the advancing
side **drops** when defects form. So slow/stagnant material near the tool is a
physical feature of FSW, not automatically a numerical artefact.

**Correct framing, if these slides are ever shown:** the issue is not that particles
linger -- they do in reality -- but that in our runs a large fraction **never leave**
over the whole simulation, which a steady through-flow cannot sustain. That is a
statement about mass conservation in the discrete field, not about incompressibility
forbidding rotation.

---

## ⚠️ O54 — O50 RETRACTED: the "independent reproduction" of the D trapping was not independent

**Raised by the user 2026-10-06** and confirmed: *"both the tracking are done by the
same jaxtrace code ... I think it is a bug and we all know we should work on it."*

### The evidence

The `post_pt/` runs carry `Group`, `ParticleID` and `Escaped` arrays in exactly the
VTU layout JAXTrace writes; ours carry the same plus `Damage_lnPhi` / `Damage_CL`.
The older campaign is **JAXTrace before the damage feature existed**, not a separate
implementation.

| | old `post_pt` run | our Phase 4 run |
|---|---|---|
| arrays | Group, ParticleID, Escaped, Temperature, MaxTemperature | Group, ParticleID, **Damage_lnPhi**, **Damage_CL** |
| code | JAXTrace | JAXTrace |

### What this invalidates

**O50 claimed the D trapping "is NOT an artefact of our pipeline" because "an
independent run, with different seeding, reproduces it."** That inference does not
hold: the same code on the same velocity field reproducing the same behaviour is a
**regression check**, not independent corroboration. Different seeding and particle
count do not make it an independent method.

⚠️ The measurement itself stands -- D3 traps 19.98 % (old) and 18.5 % (ours) against
2.5-6.6 % elsewhere, and the void appears in both. **Only the interpretation changes**,
and it changes in the conservative direction: we can no longer exclude the tracking
code as the cause.

### Current status of the D anomaly

Established:
* trapping is 17-20 % against 2.5-6 % elsewhere, in every D case;
* it is localised to r = 2.4-4.0 mm;
* trapped particles accumulate almost no damage (ratio 8-23 vs 638-1025);
* it is **not** tilt (all 20 cases report 0.0 deg) and **not** the pin radius profile
  (D3's profile is identical to B2's at every depth).

**Not** established: whether the cause is in the tracking, the velocity field, or the
mesh. All three remain open.

**Presentation stance:** state it as *a known open problem to be resolved*, not as a
result and not as an independent confirmation. The D numbers must not be read as a
tool-geometry finding until it is settled.
