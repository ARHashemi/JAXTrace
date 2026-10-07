# Predicting defect susceptibility in Friction Stir Welding

### Damage laws integrated along particle pathlines, across 45 CFD cases

<sub>Terminology used throughout: **damage law / damage model** — the two
constitutive ODEs (the literature's own term for this class). **Damage indicator** —
the accumulated value each particle carries. **Defect** — the physical void in the
weld. **Defect (or void) susceptibility** — what the ranking actually predicts. The
indicators are *not* a predicted porosity; see §3.5.</sub>

---

## Summary

We integrate two standard metal-forming damage laws **along the paths that material
particles actually take** through a friction-stir weld, using velocity and pressure
fields from an existing finite-element model. Three results:

1. **It is essentially free.** Damage accumulation adds **under 2 %** to the particle
   tracking cost, because it reuses work the tracker has already done. All 45 cases
   ran in one overnight batch.

2. **It ranks tool geometries.** Averaged over the weld wake, the **B-Flutes** family
   — which has the deepest lobes — accumulates the most damage on *every* statistic.
   Two independently-structured damage laws agree on the ordering (rank correlation
   +0.98 to +0.998).

3. **It shows the mechanism.** Following individual particles reveals that damage is
   acquired in **discrete capture events**: a particle is entrained into orbit in the
   shear layer, accumulates sharply over a few revolutions, then releases. A
   particle's final damage is essentially a count of how many times it was captured.

4. **The empty regions are not yet usable as defect predictions.** Across all 20 tool
   cases, void area is almost perfectly predicted by the fraction of particles that get
   stuck at the tool (**ρ = +0.98**, and +0.95 even excluding the D-Concavity family),
   and those particles accumulate almost no damage — the opposite of what genuine
   shear-layer capture produces. We report this as a **negative result** (§8.3b).
   ⚠️ An **earlier particle-tracking campaign shows the same effect** (§8.5). That
   campaign used the **same tracking code**, so it is a reproduction rather than an
   independent confirmation; we treat the trapping as **an unresolved defect in the
   tracking**, to be fixed before the D results are interpreted. The damage-based tool
   ranking does not depend on it.

**A negative result included deliberately.** The method produces empty regions in the
weld cross-section that look like voids. We tested them: void area correlates with
trapped-particle fraction at ρ = +0.98 (§8.3b), so the measure is not independent of
a particle-conservation problem, which an earlier run with the same code also shows
(§8.5). Reported here with the diagnosis rather than omitted; it is a known issue to
be resolved, not a result.

**What this is not.** There is no experimental defect data — no macrograph, no CT
scan — in this work or in the published validation of the underlying model. What we
produce is a **physically grounded defect-susceptibility ranking with no fitted
parameters**. It is not a validated defect prediction, and nothing here should be
read as one.

---

## 1. Why a void in FSW is not a bubble

Friction stir welding is a **solid-state** process. Peak temperature reaches only
80–95 % of melting, so nothing liquefies. A void is therefore not a bubble that
nucleates and grows — it is **material that failed to refill** the cavity behind the
advancing pin. A volumetric bookkeeping failure.

Three familiar explanations were tested and ruled out quantitatively:

| proposed mechanism | the number | verdict |
|---|---|---|
| turbulence | Reynolds number ≈ 10⁻⁷–10⁻⁴ | **7–12 orders of magnitude** below transition |
| cavitation | cavitation number ≈ 7×10⁴ (needs ≲ 1) | wrong by ~11 orders, **and wrong sign** |
| vorticity | rigid rotation gives zero stretching | large even in defect-free welds |

The last one matters practically: the tool rotates everything near it, so vorticity
cannot discriminate good welds from bad. What is needed is the **stretching** part of
the velocity gradient, not the spin part.

---

## 1b. Process geometry and terminology

![geometry](figs_concept/figC1_geometry.png)

The terms used throughout this report:

- **Advancing side** — where tool rotation and travel add; **retreating side** — where
  they oppose. Defects are reported in the literature to form preferentially on the
  advancing side.
- **Shoulder** (radius 7 mm) — contacts and constrains the top surface.
  **Pin** (radius ≈ 2.4 mm) — stirs the full 6 mm plate depth.
- **Shear layer** — a band of thickness δ ≈ 0.5–2.8 mm around the pin, where the
  strain rates that drive damage are concentrated. §9 shows that this is also where
  material is captured and where essentially all damage is acquired.
- **Wake** — the region behind the advancing tool, which the flow must refill.

---

## 2. The physical variable: stress triaxiality

Stress splits into two parts that do different jobs. The **volumetric** part
(mean stress σ_m) changes volume; the **deviatoric** part changes shape.

![stress split](figs_concept/figC2_stress_split.png)

Metal *flows* because of the shape-changing part. But voids **open or close**
because of the volume-changing part. The ratio between them is the **stress
triaxiality**:

> **η = σ_m / σ_eq**,  where σ_m = −p (pressure) and σ_eq is the von Mises stress

![triaxiality](figs_concept/figC3_triaxiality.png)

Negative η means compression dominates and voids close. Positive η means tension
dominates and voids open. The dependence is **exponential**, so a modest change in η
is a large change in void growth — which is what makes it discriminating.

**This was verified, not assumed.** Measured on the nodal fields, the mean stress is
**compressive ahead** of the tool (the forging action) and **tensile in the wake**
where voids form — median σ_m of **−6 to −7.5 MPa ahead** and **+0.4 to +2.2 MPa
behind**. ⚠️ The sign pattern, which is the part the argument rests on, holds in
**9 of the 10 cases** checked; one case (A2old) shows a weakly negative wake value.

---

## 3. The two damage laws

### 3.1 The driving quantities

Both laws are driven by the same pair, computed at every mesh node.

**Effective (von Mises equivalent) strain rate.** With **D** the stretching tensor —
the symmetric part of the velocity gradient, which carries no rigid rotation — and
**D′** its deviatoric part:

> ε̇_eff = √(⅔ D′:D′),  D = ½(∇u + ∇uᵀ),  D′ = D − ⅓(tr D)I

The ⅔ normalisation is chosen so that a uniaxial test returns exactly ε̇.

**Maximum principal stress.** σ₁ is the largest eigenvalue of the full stress tensor
σ = s + σ_m I, so that

> σ₁ / σ_eq = σ_m / σ_eq + ŝ₁

where ŝ₁ is the largest eigenvalue of the deviator normalised to unit von Mises norm.
This is obtained from D′ through the generalised-Newtonian closure s = 2μD′, so the
viscosity never has to be formed explicitly.

### 3.2 Rice–Tracey (1969)

Rice and Tracey solved the velocity field around an isolated spherical void in a
rigid-plastic matrix under remote triaxial loading, obtaining the radial growth rate

> Ṙ/R = **0.283** · ε̇_eff · exp(3η/2)

Since a volume fraction goes as R³, Φ̇/Φ = 3Ṙ/R, which gives the form used here:

> d(lnΦ)/dt = **0.849** · ε̇_eff · exp(3η/2),  0.849 = 3 × 0.283

**The 0.849 is not fitted** — it is a geometric factor of three applied to a constant
from a 1969 analytical solution. This is the property that made the law the right
first choice: it can be run with no calibration at all.

**How it indicates damage.** Φ is a physical void volume fraction — genuine porosity,
derived from the velocity field around an actual spherical void — and the law states
that voids grow *exponentially* with triaxiality: tension opens them, compression
suppresses growth.

⚠️ **But what we compute is not Φ.** The accumulator starts at zero, so the stored
quantity is ∫0.849 ε̇ exp(3η/2) dt = **ln(Φ/Φ₀)**, a *log growth factor*: it says
"porosity grew by a factor e^value", not "porosity is X". Recovering an absolute Φ
would require the initial porosity Φ₀ — of order 10⁻⁴–10⁻³ for wrought aluminium and
measurable by metallography, but **not available for this material**. A stored value
of 0.25 means Φ grew by ×1.3; 9.21 means ×10⁴.

*J. R. Rice & D. M. Tracey, On the ductile enlargement of voids in triaxial stress
fields, J. Mech. Phys. Solids 17(3), 201–217 (1969).*

### 3.3 Cockcroft–Latham (1968)

> C = ∫₀^ε̄ ⟨σ₁⟩/σ_eq dε̄   ⟹   dC/dt = max(σ₁, 0)/σ_eq · ε̇_eff

**The integrand contains no fitted parameter.** The single constant is C_crit, a
*threshold* that determines when the accumulated value counts as failure rather than
a coefficient inside the integral. It is obtained from a tensile or upset test, with
literature values for aluminium alloys around 0.2–0.5. **We have not calibrated it
for this alloy**, so C is used here as a ranking only.

⚠️ **C has no physical referent.** Unlike Φ it is not a porosity, a crack length or a
volume fraction — it is normalised plastic work, a dimensionless phenomenological
indicator whose meaning comes entirely from comparison against a calibrated C_crit for
the same alloy. **We have not calibrated one**, so C is used here as a ranking only.

**How it indicates damage.** ⟨·⟩ is the Macaulay bracket, ⟨x⟩ = max(x, 0), so damage
accrues *only where the largest principal stress is tensile*: where σ₁ ≤ 0 the
integrand is exactly zero and C stops growing.

⚠️ **Measured, and contrary to the intuitive expectation.** It is natural to assume
that the shoulder's forging action makes σ₁ compressive and switches the bracket off
under the tool. **It does not.** On the nodal fields, σ₁ > 0 at **~95 %** of active
nodes inside the shoulder radius (4.9–5.9 % negative, consistent across eight cases).

The reason is that σ₁ = σ_m + σ_eq·ŝ₁, and this flow is strongly shear-dominated:

| term, median under the shoulder | value |
|---|---|
| ŝ₁ — deviatoric contribution | **+0.87** |
| η = σ_m/σ_eq — volumetric contribution | **−0.05** |
| σ₁/σ_eq | **+0.80** |

The deviatoric term dominates, so σ₁ remains tensile even where the *mean* stress is
compressive — which it is at 55 % of those nodes.

**The consequence is worth stating, because it changes what the two laws mean here.**
Rice–Tracey responds to the **volumetric** state through exp(3η/2) and *is* suppressed
by compression. Cockcroft–Latham, in this flow, is driven mainly by the **deviatoric**
state and is not. The two are therefore not redundant: they probe different parts of
the stress tensor, which is a stronger reason to run both than the generic "two laws
agree" argument.

*M. G. Cockcroft & D. J. Latham, Ductility and the workability of metals, J. Inst.
Metals 96, 33–39 (1968).*

### 3.4 Why run both

The two are structurally different: one is exponential in triaxiality and tracks a
physical void fraction, the other is a Macaulay bracket on the maximum principal
stress and is a dimensionless indicator. Where they agree, that agreement is cheap
evidence; where they disagree, the disagreement is diagnostic. Neither was invented
for FSW — both are standard metal-forming laws with decades of use.

### 3.5 Limitations, and what remains open

- These are **growth** laws, not nucleation laws. Φ = 0 is a fixed point, so they
  require an assumed initial porosity Φ₀ that nobody measures directly.
- Rice–Tracey has **no saturation term**. A volume fraction cannot exceed 1, reached
  at ln(Φ/Φ₀) = 9.21, but values up to **5242** are measured. **ln(Φ/Φ₀) is therefore a
  defect-susceptibility ranking, not a predicted porosity**, and every figure should be read
  that way.

Two richer models were set aside for this first pass **because they require
calibration, not because they are unsuitable** — both remain open for later work:

| model | what it would add | what it requires |
|---|---|---|
| **Lee–Dawson** | the published FSW precedent — He, Dawson & Boyce (2008) applied it to this process | a coupled κ evolution ODE and ~10 constants; it clamps at σ_m ≤ 0, so it cannot represent void *closure* |
| **GTN** | full porous plasticity, with nucleation and coalescence | 7+ constants and a porous-plasticity solve inside the flow model |

He *et al.* reported a porosity change of 0.01 % → 0.017 %, a ~70 % relative change
on a very small absolute value — useful as precedent, and also as a caution about how
strongly the answer depends on the calibrated constants.

---

## 4. Why follow particles instead of using a grid

Damage is a **history integral** — it depends on everything that happened to a piece
of material, not on where it is now. A fixed grid cell has no history; material flows
through it continuously.

![pathline](figs_concept/figC5_pathline.png)

Following the particle makes the transport term vanish exactly, so the update is
simply `new = old + dt × rate`. No gradient of the damage variable is needed, and no
numerical smearing is introduced at precisely the sharp gradients we are trying to
resolve.

---

## 5. The tools

Four families of pin geometry were studied. Each dot below is a **vertex of the
tool's own CAD surface mesh** — real geometry, not a computed field. The upper and
lower rows are the same points seen from two directions.

![pin shapes](figs_concept/figC4_pin_shapes.png)

Lobe counts and depths were **measured from the CAD**, not taken from folder names:

| family | lobes | lobe depth (fraction of radius) |
|---|---|---|
| **B-Flutes** | 3, 4, 6, 8 | **20–32 %** |
| C-Threads | 2 | 3–8 % |
| D-Concavity | 8 / 2 | 18 % / 4 % |
| A-Flats | — | below the 1 % detection floor |

Two caveats:

- **"A-Flats has no lobes" is not a finding.** Its CAD files are too coarse (2,794
  triangles, no mid-pin surface) to resolve shallow flats. This is a data limitation.
- The tool-shape dataset is a **pure geometry study at a single operating point** —
  rotation speed, travel speed, plate thickness and material are effectively
  constant, so pin geometry is the only real independent variable.

---

## 6. Implementation and verification

Three nodal fields — effective strain rate, triaxiality, and normalised maximum
principal stress — are precomputed on the mesh and sampled exactly like velocity, at
the mesh element the velocity step has already located. Nothing leaves the GPU.

### 6.1 Where σ_eq comes from, and a factor-1.8 correction

The flow model uses a **Norton–Hoff** power law whose two coefficients are tabulated
against temperature in each case's own `.mat` file:

> σ_eq = VISCO(T) · ε̇^EXPVI(T)

VISCO is a viscosity-like prefactor and EXPVI the strain-rate sensitivity exponent
(m ≈ 0.086 here; m → 0 is perfectly plastic, m → 1 Newtonian).

**Using those tables directly gives the wrong answer by a factor of 1.8**, and the
way this was found is worth stating because it is the strongest single piece of
evidence that the implementation is correct.

**What was done.** We computed σ_eq from the tables the obvious way and compared it
against the stress **the solver itself exports** as a field (√(3J₂)). Over 581,488
cells our value was consistently

> ours ÷ solver's = **0.569**  (±0.8 %)

A *constant* ratio is the signature of a convention mismatch, not a coding error — a
bug would scatter.

**The cause.** "Equivalent stress" and "equivalent strain rate" have two conventions
in common use, and a power law written in one is wrong in the other:

| | von Mises (our pipeline) | **shear** (the tables) |
|---|---|---|
| strain rate | ε̇ = √(⅔ D′:D′) | γ = √2‖e‖ |
| stress | σ_vM = √(3J₂) | τ = (√2/2)‖s‖ |
| relation | — | γ = √3 ε̇,  σ_vM = √3 τ |

The solver's own paper states the shear definitions explicitly (Venghaus 2023,
eq. 5), and VISCO/EXPVI are the coefficients of *that* law. Feeding it a von Mises
rate therefore needs a √3 going in, and the shear stress it returns needs a √3 coming
out:

> σ_eq = √3 · VISCO(T) · (√3 · ε̇)^m,  factor = √3(√3)^m = **1.816** at m = 0.086

**The check.**

| | ratio to the solver's own √(3J₂) |
|---|---|
| tables used as-is | **0.569** |
| with the conversion | **1.035** |

Measured correction 1/0.569 = **1.819** against predicted **1.816** — **0.13 %**.
Predicting the factor from the published definition and then matching it to 0.13 % is
what makes this a confirmed derivation rather than a number chosen to fit.

**Can the same `.mat` file be used?** **Yes.** The tables are the right data for the
right material; they are written in the other convention, and the correction is one
constant factor applied once.

⚠️ This is the material **viscosity** convention — *not* the "modified Norton"
**friction** law, which is a separate boundary condition with a similar name.

⚠️ All absolute σ_eq and η values reported before this correction are low by 1.816×.
Ratios and rankings are unaffected, which was confirmed independently (ρ = +0.9995).

### 6.2 Integration order

**The position and the damage accumulator share the same four RK4 stages**: the
damage sampler reuses the mesh elements the velocity step has already located, so the
accumulator costs no additional element searches. But the two are solving different
problems, and "RK4" means something different in each.

| | position *x* | damage accumulator Φ |
|---|---|---|
| equation | dx/dt = u(x(t), t) | d(lnΦ)/dt = f(x(t), t) |
| does the **unknown** appear on the right? | **yes** — x does | **no** — lnΦ does not |
| so it is | an initial-value problem | a **quadrature** along an already-determined path |
| "RK4" means | the classic 4th-order integrator | the same (k₁+2k₂+2k₃+k₄)/6 weights |
| 4th-order convergence? | yes | **no** |

⚠️ **This does not mean the right-hand side is constant.** The drivers η and ε̇ vary
strongly, both in **space** — they are sampled at the particle's moving position —
and in **time**, since the nodal fields are themselves time-dependent. That variation
is precisely what the integration is capturing. The narrow point is that **lnΦ itself
never appears on the right-hand side**: along a path that the position solve has
already determined, the equation reduces to d(lnΦ)/dt = f(t). There is no feedback
from the accumulated value back into its own growth rate, which is what makes it a
quadrature rather than a coupled solve.

For the **position**, RK4 is necessary and not in question: at r = 3 mm a particle
sweeps 9.0° and 0.47 mm of arc per step, so a first-order step would cut the corner
element after element.

For the **damage accumulator** the same RK4 combination is applied: the four stages
are sampled at the four RK4 trajectory points — four *different positions* in the
mesh — and weighted (k₁+2k₂+2k₃+k₄)/6. Because the damage right-hand side does not
depend on the accumulated value, the method acts as a quadrature along the path the
position solve has already determined, rather than as a state-dependent integration.

Fourth-order *convergence*, however, is not achieved here. The driver (ε̇, η, σ₁) is
P1-interpolated, so along a path it is C⁰ with a kink at every element face, and any
fourth-order quadrature needs a smooth integrand — across a kink it degrades to first
or second order. **Its real benefit is different**: the
k1-only estimate samples the *start* of each step, while the k2 and k3 stages sample
the *midpoint*, removing a bias that matters wherever the driver has a steep
gradient — that is, in the shear layer.

**Both orders are implemented and selectable, and all 45 cases in this report were
run at order 4.** The controlled comparison — identical case, seed, particles and
steps, with only the order changed:

| | order 1 | order 4 | change |
|---|---|---|---|
| **position, maximum difference** | — | — | **0.000e+00 m** |
| lnΦ median | 0.004456 | 0.004456 | **+0.01 %** |
| lnΦ **maximum** | 179.85 | **147.67** | **−17.9 %** |

**It is a tail correction, not a bulk one.** A typical particle changes by 0.05 %;
the maximum drops 18 %. First order would be adequate for median-based rankings and
inadequate for tail metrics — `frac_failed`, p99, maximum. Since §9 shows that damage
arrives in bursts — the median particle collects 84 % of its total in 10 % of its
steps, against 10 % if uniform — the tail is exactly where the integrand is
worst behaved, so order 4 is used throughout. It costs three extra samples per scalar
at elements the velocity step has already located — no additional element searches —
and the measured throughput penalty is **−2.3 %, within noise**.

⚠️ This comparison is a single case. A sweep across families would establish whether
the 18 % tail shift is case-dependent; both orders are already implemented, so this
is cheap.

### 6.3 Cost

**Measured by running with the feature on and off, alternating:**

| platform | slowdown |
|---|---|
| RTX 5090 (CUDA) | +0.15 … +0.51 % |
| LUMI MI250X (ROCm) | +1.77 % |
| production rate with damage + full export | 119,525 particle-steps/s |

---

## 7. The simulation campaign

45 cases, **30 mm of travel**, 100,000 particles each, with full trajectory export.
All 45 completed with no failures.

The 45 divide into three sets:

| set | n | what varies | purpose |
|---|---|---|---|
| **PinShapes** | **20** | **pin geometry only** — rotation speed, travel speed, plate thickness and material are held constant | ranking tool designs |
| **reference cohort** | 22 | operating point (speeds, mesh) on a single cylindrical pin | reduced-order-model training set |
| **validation** | 3 | held-out operating points | checking that model |

The PinShapes set comprises four families: A-Flats (7 cases), B-Flutes (4),
C-Threads (4) and D-Concavity (5).

⚠️ **The sets are not comparable in absolute value** — they use different meshes,
speeds and timesteps, so comparisons are made *within* a set. PinShapes answers
*which tool*; the cohort answers *which operating point*.

An earlier attempt at 10 mm of travel was **too short** — the particle bundle had not
cleared the tool, with a mean displacement of **−0.52 mm** (i.e. still recirculating
beside the pin). Tripling the travel distance fixed this:

| | 10 mm run | 30 mm run |
|---|---|---|
| mean displacement | −0.52 mm | **+18.95 mm** |
| particles past the tool | — | **97.2 %** |
| particles still beside the pin | 22.5 % | **2.6 %** |

### Where the results are measured

Scoring uses a **wake box** containing only material that has passed through the
process and emerged:

| boundary | position | reason |
|---|---|---|
| upstream | **7 mm behind the axis** | excludes the tool footprint and the 2.6 % still orbiting |
| downstream | the particle front | — |
| lateral | full domain width | **kept** — the advancing/retreating contrast lives here |
| bottom | full depth | **kept** — the weld root is of interest |
| top | **8 % of thickness below the surface** | removes a boundary-condition artefact |

The top exclusion matters: that band sits against a prescribed-velocity surface and
its extreme values are **5–16× the bulk**, which is a numerical response rather than
physical damage. The cut is deliberately shallow (0.36–0.48 mm) because, as §9 shows,
real damage occurs at a similar depth.

**The box retains ~90 % of all particles** — a targeted exclusion, not a filter that
discards the statistics.

---

## 8. Results

### 8.1 The two damage laws agree

![models agree](figs_phase4/fig2_models_agree.png)

Rank correlation between Rice–Tracey and Cockcroft–Latham is **+0.982 to +0.998
across all 45 cases** — spanning two tool families with opposite rotation senses.
This is the implementation's self-consistency check.

### 8.2 Tool family ranking

| family | mean lnΦ | median | 99th pct | fraction past failure |
|---|---|---|---|---|
| A-Flats | 6.332 | 0.2413 | 69.0 | 19.4 % |
| **B-Flutes** | **7.166** | **0.2519** | **79.8** | **20.9 %** |
| C-Threads | 6.316 | 0.2500 | 69.7 | 20.1 % |
| D-Concavity | 2.650 | 0.1745 | 39.9 | 9.0 % |
| reference cohort | 5.554 | 0.9146 | 49.4 | 17.6 % |

**B-Flutes ranks highest on every statistic**, and it is the family with the deepest
lobes. A-Flats and C-Threads sit within 0.2 % of each other, consistent with their
similarly shallow geometry.

Two caveats:

- **D-Concavity should not be read as a clean comparison.** Only ~63,800 of its
  particles reach the wake box, against ~88,000 for the other families — so part of
  its low score is that *fewer particles arrive*, not that each is less damaged.
  §8.3b shows why: a seventh of its weld cross-section is **unfilled**. D is the only
  family whose anomaly is unexplained — ⚠️ despite the folder name, **no case in
  this dataset has tool tilt** (all 20 report `Tilt angle: 0.0`).
- The A/B/C spread is about **9 %** — a consistent ordering, not a dramatic one.

### 8.3 Where damage sits in the weld cross-section

This is the plane a metallurgical section would cut. It is presented in three stages:
the **raw particles** first, then the **continuous field** derived from them, and
finally the **comparison** against the earlier particle-tracking campaign (§8.5).

#### 8.3.1 The raw particles

Every particle in the wake box, projected onto the yz plane and coloured by the
damage it accumulated. Nothing is averaged, binned or smoothed — this is the
measurement itself. Horizontal and vertical scales are equal, so the shape is the
true weld geometry: a **shallow band widest near the top surface**, narrowing with
depth, with a sharp boundary on one flank.

⚠️ **What the raw view shows, and what it does not.** The speckled texture is
*sampling*, not physics: seeding is random, so particle positions are a Poisson
process and neighbouring particles can differ by orders of magnitude in accumulated
damage. Reading structure from individual markers would be reading noise. What *is*
trustworthy here is the **envelope** — where particles exist at all, and the extent
of the bright region.

All four families, in order:

**A-Flats**

![raw particles, A-Flats 1](figs_wake/figW_A-FlatsVariations_1_yz.png)

![raw particles, A-Flats 2](figs_wake/figW_A-FlatsVariations_2_yz.png)

![raw particles, A-Flats 3](figs_wake/figW_A-FlatsVariations_3_yz.png)
**B-Flutes**

![raw particles, B-Flutes 1](figs_wake/figW_B-FluteVariations_1_yz.png)

![raw particles, B-Flutes 2](figs_wake/figW_B-FluteVariations_2_yz.png)
**C-Threads**

![raw particles, C-Threads 1](figs_wake/figW_C-ThreadsVariations_1_yz.png)

![raw particles, C-Threads 2](figs_wake/figW_C-ThreadsVariations_2_yz.png)
**D-Concavity**

![raw particles, D-Concavity 1](figs_wake/figW_D-ConcavityTilt_1_yz.png)

![raw particles, D-Concavity 2](figs_wake/figW_D-ConcavityTilt_2_yz.png)

#### 8.3.2 From particles to a continuous field

The scatter is converted to a field so cases can be compared quantitatively. The
estimator is a **k-nearest-neighbour median**: at each point of a 600 × 240 output
grid, take the 16 nearest particles and report the median of their ln(Φ/Φ₀).

Three choices in that sentence matter:

- **Median, not mean.** lnΦ spans five decades and is heavily right-skewed, so a mean
  would map single extreme particles rather than where the bulk of the material sits.
- **k-nearest-neighbour, not a histogram bin.** With ~88,000 particles over ~165 mm²
  the mean spacing is 0.18 mm, so any bin grid fine enough to look smooth is mostly
  empty — at 600 × 240 more than half the cells would contain no particle at all. The
  kNN estimator decouples output resolution from cell occupancy: every output point
  has 16 real samples, wherever they lie.
- **Blanking.** Where the 16th neighbour is further than 0.45 mm the point is left
  **grey**, not dark. Grey means *no data*, a different statement from *low damage*.

The same four families, same order, same colour scale. The envelope is unchanged from
§8.3.1; the Poisson speckle is gone, and the internal gradient — brightest under the
shoulder, falling with depth — becomes legible.

**A-Flats**

![continuous field, A-Flats 1](figs_wake/figWB_A-FlatsVariations_1_yz_field.png)

![continuous field, A-Flats 2](figs_wake/figWB_A-FlatsVariations_2_yz_field.png)

![continuous field, A-Flats 3](figs_wake/figWB_A-FlatsVariations_3_yz_field.png)
**B-Flutes**

![continuous field, B-Flutes 1](figs_wake/figWB_B-FluteVariations_1_yz_field.png)

![continuous field, B-Flutes 2](figs_wake/figWB_B-FluteVariations_2_yz_field.png)
**C-Threads**

![continuous field, C-Threads 1](figs_wake/figWB_C-ThreadsVariations_1_yz_field.png)

![continuous field, C-Threads 2](figs_wake/figWB_C-ThreadsVariations_2_yz_field.png)
**D-Concavity**

![continuous field, D-Concavity 1](figs_wake/figWB_D-ConcavityTilt_1_yz_field.png)

![continuous field, D-Concavity 2](figs_wake/figWB_D-ConcavityTilt_2_yz_field.png)

⚠️ **D-Concavity carries a large grey region**, i.e. an area no particle reaches at
all. That is examined in §8.3b and tested against an independent method in §8.5.

### 8.3b Two independent indicators, deliberately separated

The cross-sections above colour a *k*-nearest-neighbour median of damage. That
estimator is **not independent of particle density**: the nearest neighbours of a
sparse point are drawn from a wider region, so the estimate is smoothed over more
space exactly where particles are scarce. Measured inside the wake box, the rank
correlation between local density and reported damage is −0.06 for A-Flats and
−0.07 for B-Flutes, but **−0.29 for D-Concavity**, whose density spans a 27-fold
range.

Because particle tracking and damage integration are meant to be **two independent
routes to the same prediction**, they must not share an estimator. The figures below
separate them:

- **density** is a plain count of particles per unit area;
- **damage** is a median over a **fixed-area** cell, so every reported value is
  averaged over the same physical area regardless of how far its neighbours lie.
  Cells containing fewer than 8 particles report *nothing* — "not measurable" is kept
  distinct from "low".

After this change the density–damage correlation for D-Concavity falls from −0.29 to
**+0.015**, i.e. the apparent coupling was almost entirely an artefact of the
estimator.

#### A-Flats A2 — both indicators point to the same place

![A2 density vs damage](figs_compare/figX_ps_A-FlatsVariations_A2_density_vs_damage.png)

The density panel is uniform everywhere except a **single sharply-bounded void**. The
damage panel shows that void lying **directly beneath the brightest damage band**.
The third panel, comparing the two as within-case ranks, is strongly positive all
around it.

#### D-Concavity D1 — a larger void, but in depleted surroundings

![D1 density vs damage](figs_compare/figX_ps_D-ConcavityTilt_D1_density_vs_damage.png)

The void is genuinely empty rather than merely sparse, and the damage field does drape
over its upper boundary rather than ignoring it. But the surrounding material is
**globally depleted**, which is the reason for caution set out below.

#### Quantitative comparison — and why none of these voids can be trusted

| case | void area | density CV | void isolation | rim damage ÷ median |
|---|---|---|---|---|
| A-Flats A1 | 0.15 mm² | 0.347 | 0.25 | 67 |
| A-Flats A2 | 3.64 mm² | 0.353 | 0.91 | 64 |
| A-Flats A2old | 2.20 mm² | 0.352 | 0.89 | 84 |
| B-Flutes B2 | 0.55 mm² | 0.350 | 0.38 | 60 |
| D-Concavity D1 | 32.90 mm² | 0.380 | 0.67 | 80 |
| reference cohort | 15.77 mm² | 0.625 | 0.56 | 8.9 |

On the two-indicator test alone, A-Flats A2 looks like the best candidate: a compact
void in a uniformly filled section, directly beneath a damage maximum. **That reading
does not survive a further check.**

Across all 20 tool-geometry cases, void area is almost perfectly predicted by the
fraction of particles that are still inside the shoulder radius at the end of the run:

| | Spearman ρ |
|---|---|
| void area vs **trapped fraction** | **+0.980** |
| void area vs trapped/wake damage ratio | **−0.932** |
| void area vs trapped fraction, **excluding D** | **+0.954** |

⚠️ **Excluding the D-Concavity family does not weaken the relationship**, so this is not a
property of one geometry — it holds within A-Flats, B-Flutes and C-Threads as well.
Inside the A-Flats family it is a clean monotone ladder:

| case | trapped | damage on trapped particles | ratio to wake | void |
|---|---|---|---|---|
| A1 | 2.6 % | 259.8 | 1003 | **0.15 mm²** |
| A4old | 2.5 % | 223.6 | 897 | 0.15 mm² |
| A3old | 3.4 % | 88.0 | 370 | 1.49 mm² |
| A3 | 3.8 % | 16.6 | 67 | 1.62 mm² |
| A2old | 4.1 % | 12.0 | 54 | 2.20 mm² |
| **A2** | **5.2 %** | **9.5** | **39** | **3.64 mm²** |

**A2 is simply the A-Flats case with the most trapping.**

#### The diagnostic: trapped particles that do not accumulate damage

A particle genuinely held in the shear layer should accumulate damage *hard* — that
is what the capture mechanism in §9 shows. B-Flutes and C-Threads deliver exactly
that: their trapped particles carry **600–1000×** the wake median damage.

| family | trapped | trapped ÷ wake damage |
|---|---|---|
| B-Flutes | 2.7–3.0 % | **868–1025** |
| C-Threads | 3.2–3.4 % | 638–769 |
| A-Flats | 2.5–5.2 % | 39–1003 (bimodal) |
| **D-Concavity** | **17.1–18.7 %** | **8–23** |

D's trapped particles sit at a radius of ~3.65 mm — inside the shear layer — and
accumulate **almost nothing**. They are being circulated by the interpolated velocity
field *without experiencing the strain rates that being that close to the tool
implies*. That is the signature of a numerical trap, and the same signature appears,
more weakly, in A2, A2old and A3.

⚠️ **Conclusion: no void in this dataset is currently a credible defect prediction.**
Void area is explained by a tracking pathology (ρ = +0.98) rather than by weld
physics. Critically, the agreement between the two indicators is **not** independent
evidence: particle density is directly depleted by trapping, and the damage field is
shaped by which particles survive to reach the wake — so both are downstream of the
same failure.

The direct test needs no new simulation: **count particles crossing a control surface
around the tool, in versus out.** In a steady incompressible flow they must balance.
The exported trajectory series already contains everything required.

**The damage-based tool ranking in §8.2 is unaffected** — it is computed over the wake
population and does not depend on the void measure.

### 8.4 Advancing versus retreating side — a partly open result

| | advancing | retreating | ratio |
|---|---|---|---|
| tool-shape cases | 0.2292 | 0.2317 | **0.989** |
| reference cohort | 0.7704 | 1.2013 | **0.641** |

In the earlier, too-short run the advancing side appeared lower in **all 45 cases**,
contradicting an independent analysis of the instantaneous stress state. With the
correct travel distance and the wake box, the tool-shape cases **balance out** — so
that contradiction was largely a measurement artefact.

However, a **cohort-specific excess on the retreating side survives** and is **not
currently explained**. It is stated here rather than omitted.

---

### 8.5 Comparison with the earlier particle-tracking runs

*This is the third stage of the cross-section presentation begun in §8.3: raw
particles, then the continuous field, now an independent check against a different
method.*

Each case folder also contains an **earlier, independent** particle-tracking
campaign in its `post_pt/` directory: **288,000 grid-seeded** particles over 8,000
steps, run separately from the work reported here, with different seeding and a
different configuration.

Those runs carry no damage variable. Their particles are labelled by `Group`, the
x-slab each was seeded in — five consecutive sheets of material entering the tool.
Colouring by that label shows how the tool folds the incoming material, and **a defect
appears as a region no sheet reaches**.

This gives a genuine two-method comparison: an indicator built purely from *where
material ends up*, against one built from *what happened to it along the way*.

**How each panel is constructed.** The two are drawn differently, for a reason worth
stating:

- The **damage** panel reduces each cell to a *median*, so projecting the whole
  17 mm depth of the wake box along x is harmless — more samples simply make the
  median better.
- The **particle-tracking** panel draws one marker per particle, with **every
  particle beyond x = 11.5 mm projected onto the yz plane** — the same construction
  as the damage panel, so the two cover the same material. That wall sits 4.5 mm
  clear of the 7 mm shoulder edge, so no particle still attached to the tool enters
  the projection.

⚠️ **The two panels use different upstream walls, deliberately.** The damage panel
uses **7.4 mm**, matching every other damage figure in this report (§8.3b). The PT
panel uses **11.5 mm**, because it projects *every* particle rather than taking a
per-cell median, so its wall is set further out to keep the near-tool cap out of the
picture entirely.

⚠️ **B2 is the one exception, at 6.0 mm.** Its earlier tracking run finished much
closer to the tool than the others: x_max is 24.6 mm against 29.4–29.7 mm for every
other case, and only **33.5 %** of its particles lie beyond 11.5 mm where the others
retain **74–76 %**. It holds 52,976 particles in the 6–8 mm band where B1 holds 9,377.
The common wall would discard two thirds of its cloud. Checked against all thirteen
cases, **B2 is the only one affected** — D3 also has a short x_max (22.7 mm) but
retains 71.7 %, a normal fraction, so it keeps the default. Projecting ~70,000 particles through ~13 mm leaves a median of 16
particles per plotting cell; particles are drawn in order of decreasing x so the ones
nearest the tool, which carry the folded structure, land on top.

**Viewing direction.** Both panels are orthographic — no perspective — and are drawn
as seen by an observer at the origin **looking downstream, along +x**. In a
right-handed (x, y, z) frame that places **+y on the left** of the page. ⚠️ Drawn the
other way round the figure would be the view looking back upstream, and the advancing
and retreating sides would swap places.

⚠️ Thirteen of the twenty tool cases have such a run (A1, A2old, A3old, A4old,
B1–B4, C1–C4, D3); the rest are omitted rather than partially shown. The two methods
use different particle counts and seeding schemes, so the comparison is about **where
features sit**, not how dense the clouds appear. Both panels of a pair share the same
y and z limits so they can be read against each other directly.

#### 8.5.1 B-Flutes

![B-Flutes PT vs damage](figs_compare/figP_B-FluteVariations_1_pt_vs_damage.png)

The particle-tracking panels show a gap at y ≈ +2 to +4 mm, z ≈ −3.5 to −5 mm. The
damage panels place their maximum in the same region. The two indicators, computed
from different runs by different means, agree on location.

![B-Flutes PT vs damage, continued](figs_compare/figP_B-FluteVariations_2_pt_vs_damage.png)

#### 8.5.2 A-Flats

![A-Flats PT vs damage](figs_compare/figP_A-FlatsVariations_1_pt_vs_damage.png)

![A-Flats PT vs damage, continued](figs_compare/figP_A-FlatsVariations_2_pt_vs_damage.png)

#### 8.5.3 C-Threads

![C-Threads PT vs damage](figs_compare/figP_C-ThreadsVariations_1_pt_vs_damage.png)

![C-Threads PT vs damage, continued](figs_compare/figP_C-ThreadsVariations_2_pt_vs_damage.png)

#### 8.5.4 D-Concavity — a known open problem

![D-Concavity PT vs damage](figs_compare/figP_D-ConcavityTilt_pt_vs_damage.png)

**The same central void appears in both rows.** Quantitatively, the fraction of
particles still inside the shoulder radius at the end of the run:

| case | **earlier run** (288k, grid) | **this work** (100k, random) |
|---|---|---|
| **D3** | **19.98 %** | **18.5 %** |
| B2 | 6.61 % | 2.8 % |
| C1 | 2.55 % | 3.4 % |

**This changes the interpretation of §8.3b.** That section attributed the D void to a
tracking pathology. That wording is too narrow: a different tracker, with different
seeding, run at a different time, reproduces the effect to within about 1.5
percentage points. **Our implementation is not the cause.**

The remaining explanations are upstream of any particle tracker:

1. **D's velocity field** has a near-pin structure that holds material — either
   physically, or as an artefact of the flow solution. Either would be faithfully
   reproduced by both trackers.
2. **D's mesh.** The D case directories were created by copying other cases
   (D1 from A1, D2 from A2, D3 from B4, D4 from C2 — their settings files are
   byte-identical to the source) and replacing the tool geometry. The mesh refinement
   may therefore follow the source case's pin rather than D's. Both trackers inherit
   this.

**This makes a physical reading more plausible than before**, though still not
established: if the D tool genuinely produces a recirculation that material cannot
escape, that *is* a filling failure and the void is real. Distinguishing that from a
mesh or field artefact requires examining the flow solution, not the tracker.

⚠️ The measurement in §8.3b still stands — void area correlates with trapped fraction
at ρ = +0.98, so the void *measure* remains entangled with the trapping. What has
changed is that the trapping is no longer attributable to this implementation.

---

## 9. How the tool concentrates damage along a pathline

Individual particles were followed through all 167 exported states and coloured by
the damage accumulated **so far**, so a path brightens exactly where damage is picked
up.

![trajectories](figs_traj/figT_ps_A-FlatsVariations_A1_3d.png)

![damage over time](figs_traj/figT_ps_A-FlatsVariations_A1_time.png)

Four findings:

1. **Damage is acquired in discrete capture events, not by gradual shearing.** A
   particle approaches, is **captured into orbit** around the pin, accumulates
   sharply over a few revolutions, then releases downstream. One trace rises **three
   orders of magnitude in a single step** and is then flat for the remaining ~45
   revolutions.

2. **Final damage is essentially a count of capture events.** The most-damaged traces
   are staircases — distinct bursts separated by long flat stretches, each burst a
   separate close approach.

3. **Every jump coincides with the particle passing inside the shoulder radius.** The
   upper and lower panels of the second figure are locked together.

4. **The least-damaged particles are unprocessed, not weakly processed.** They remain
   flat throughout and never approach the tool — which justifies excluding them from
   the statistics rather than averaging them in.

**Where the shear layer is.** The capture orbits sit at a radius of **3–5 mm**
(between the pin at 2.4 mm and the shoulder at 7 mm) and at a depth of **0.5–2 mm**
in a 6 mm plate — the upper third, directly beneath the shoulder. The trajectories
locate the shear layer **directly**, without needing to analyse the velocity field.

**Why this matters for the ranking.** Because the integral is dominated by a few
a small fraction of each particle's path (median: 84 % of the total in 10 % of the
steps) rather than by the long approach, the metric is effectively
measuring **how often a given tool re-entrains material** — a geometric property, and
exactly the quantity a tool-design comparison should be sensitive to.

*These trajectories are 12 particles selected across four strata (most damaged,
typical, least damaged, and largest single jump). They are a mechanism illustration,
not a population statistic; the population numbers are in §8.*

---

## 10. Limitations

**No experimental ground truth.** There is no macrograph, CT scan or sectioned weld
available — not in this work, and not in the published validation of the underlying
model, which covers forces, torque and thermocouple temperatures only. The results
are a defect-susceptibility ranking.

**ln(Φ/Φ₀) is unbounded.** Rice–Tracey has no saturation term, so the reported values are
a ranking rather than a porosity. Values past the nominal failure threshold indicate
*ranking position*, not predicted void fraction.

**Damage does not feed back.** The coupling is one-way: damage is a passive
diagnostic computed from the flow, and does not modify it.

**Families are not directly comparable in absolute value.** The tool-shape cases and
the reference cohort use different meshes, speeds and timesteps. Comparisons should
be made within a family.

**Empty regions are entangled with a particle-conservation problem.** Void area
correlates with trapped particle fraction at ρ = +0.98 across all 20 tool cases, and
the trapped particles accumulate almost no damage. ⚠️ That problem is **not specific
to one run** — an earlier campaign using the same tracking code shows it too (§8.5).
⚠️ Because the code is shared, that is a reproduction and **not** independent
evidence; the trapping is treated as an unresolved defect in the tracking. Either way
the void measure cannot currently be read as a defect prediction. The two-indicator agreement is
**not** independent
evidence, because both indicators are downstream of the same failure. This is
reported as a negative result until the conservation test in §11 is done.

**The density floor is a sampling limit.** At 100,000 particles the density
coefficient of variation is ~0.35 even in the healthiest cases, so features smaller
than roughly 1 mm² cannot be distinguished from sampling noise.

**One result remains open** — the cohort's retreating-side excess.

---

## 11. Next steps

| | |
|---|---|
| **1** | **Resolve the D trapping.** The conservation test is done (§8.3b) and the behaviour repeats with the same code (§8.5), so the cause is either in the tracking or upstream in the flow solution / mesh — it is a bug to be found, not a finding. Compare the near-pin velocity profile and divergence of D3 against B2 on the flow fields, and check whether D's mesh refinement follows its own pin or the source case's — the D directories were copied from A1/A2/B4/C2 with only the tool replaced. **This is the blocking item**: until it is settled the void measure cannot be used. |
| **2** | Extend the order-1 vs order-4 comparison (§6.2) beyond the single case tested, to establish whether the 18 % tail shift is case-dependent. Both orders are implemented, so this is cheap. |
| **3** | Resolve the cohort-specific retreating-side asymmetry. |
| **4** | Apply the independent force-ratio criterion across all cases — an external check against experimentally validated quantities, at no experimental cost. |
| **5** | Deposit the per-particle damage indicator onto a grid to produce a continuous defect-susceptibility field suitable for a reduced-order model. |

---

## Appendix — supporting figures

Radial distribution of damage relative to the tool axis:

![radial profile](figs_phase4/fig3_radial_profile.png)

Per-case advancing/retreating comparison, each case plotted against its own
advancing side (rotation sense differs between the two case sets):

![advancing wake](figs_phase4/fig4_advancing_wake.png)

Fraction of particles accumulating measurable damage, by case:

![accumulation](figs_phase4/fig5_accumulation.png)

Reference cohort wake cross-section — the first three of its 22 cases (different mesh
and operating point, so not directly comparable with the tool-shape families above):

![wake cross-section, cohort](figs_wake/figWB_cohort_1_yz_field.png)

---

## References

**Damage laws**

1. J. R. Rice & D. M. Tracey, *On the ductile enlargement of voids in triaxial stress
   fields*, **Journal of the Mechanics and Physics of Solids 17**(3), 201–217 (1969).
   Source of the 0.283 amplitude.
2. M. G. Cockcroft & D. J. Latham, *Ductility and the workability of metals*,
   **Journal of the Institute of Metals 96**, 33–39 (1968).

**FSW precedent**

3. X. He, P. R. Dawson & D. E. Boyce (2008) — Lee–Dawson damage applied to friction
   stir welding; the published precedent for pathline-integrated damage in this
   process.

**Solver and rheology**

4. L. Venghaus, **Finite Elements in Analysis & Design 224**, 103986 (2023),
   Publication 1, eq. (5) — the Norton–Hoff shear-measure definition underlying the
   √3 convention correction in §6. The accompanying thesis (§4, p. 66) is the source
   for the solid-state temperature range quoted in §1.

**Models set aside for future work**

5. A. L. Gurson (1977); V. Tvergaard & A. Needleman (1984) — the GTN porous
   plasticity model.

---

*Data: 45 finite-element cases, 100,000 particles each, 30 mm travel, full
trajectory export (24 GB). Per-case statistics in
`results/phase4_rk4_30mm_wake.csv`.*
