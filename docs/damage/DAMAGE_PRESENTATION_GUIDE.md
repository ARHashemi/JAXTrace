# Pathline-integrated damage in FSW — presentation guide

**Companion to the deck `damage_implementation_talk.md`.**
Two audiences, one document:

* **share it beforehand** — colleagues can read it on its own and arrive informed;
* **present from it** — each slide has what to say, what to expect to be asked, and
  what *not* to claim.

Everything here is measured. Where a number is uncertain or a claim does not hold,
it says so — those marks are deliberate, and the ⚠️ items are the ones most likely
to be challenged.

---

## ⚠️ Terminology — agreed wording, use it consistently

Your supervisor raised whether "defect" is better than "damage". **Both, for
different things** — and the distinction is worth stating once at the start so
nobody has to guess:

| term | use it for |
|---|---|
| **damage law / damage model** | the two constitutive ODEs. This is the literature's own term (continuum damage mechanics) — renaming them would make them harder to place |
| **damage indicator** | the accumulated lnΦ or C a particle carries |
| **defect** | the **physical void** in the weld — only where you mean one |
| **defect (or void) susceptibility** | what the ranking actually predicts |

**The rule that matters:** never say *"predicted damage"* when you mean *"predicted
defect"*. Those are different claims and we have no experimental validation for the
second.

**If asked "isn't it really porosity?"** — half right, and the half matters:
Rice–Tracey genuinely tracks Φ, a void volume fraction, so *that* one is porosity —
but with no saturation term it reaches 5242 where failure is 9.21, so it is a ranking
rather than a value. **Cockcroft–Latham is not porosity at all** — it is a
dimensionless work integral with no void in it. So "porosity" would be wrong for half
the implementation.

---

## ⚠️ Claim audit — what is verified, what is soft, what was wrong

Every quantitative claim in the talk was re-checked against our own data on
2026-10-06. This table is the result. **Read it before presenting.**

| claim | status |
|---|---|
| Re ≈ 10⁻⁷–10⁻⁴ (creeping flow) | ✅ **reproduced**: 3.8×10⁻⁵ from our own fields (ρUL/μ_eff with μ_eff = σ_eq/3ε̇) |
| Two damage laws agree, ρ = +0.982…+0.998 | ✅ from `phase4_rk4_30mm_wake.csv`, all 45 cases |
| B-Flutes highest on every wake statistic | ✅ from the same CSV |
| σ_eq correction factor √3(√3)^m = 1.816 vs 1.819 measured | ✅ 581,488 cells, 0.13 % |
| Order 4 vs order 1: median +0.01 %, max −17.9 % | ✅ controlled A/B, **but one case only** |
| σ_m compressive ahead, tensile in the wake | ⚠️ **sign pattern holds 9/10 cases**; magnitudes are case-dependent — quote the pattern, not a number |
| D trapping 17–19 % vs 2.5–5.2 % | ✅ and **reproduced by an independent earlier campaign** |
| Void area ↔ trapped fraction, ρ = +0.98 | ✅ all 20 tool cases |
| ~~"σ₁ < 0 under the shoulder, so C switches off"~~ | ❌ **FALSE — retracted.** σ₁ > 0 at ~95 % of nodes. See §3 |
| ~~D-Concavity is "the only tilted family"~~ | ❌ **FALSE — retracted.** All 20 cases report Tilt angle 0.0 |
| ~~A2's void is a credible defect~~ | ❌ **retracted** — both indicators are downstream of the same trapping |
| Cavitation number ≈ 7×10⁴ | ⚠️ **not reproduced from our fields** — carried from the earlier review. Quote as "order 10⁴, wrong sign" or drop it |

⚠️ **Three claims were retracted during preparation.** If you have shown anyone an
earlier draft, say so — it is a better look than letting it pass, and each retraction
came from checking our own data rather than being caught externally.

---

## 0. The three sentences, if you only get thirty seconds

1. We integrate two standard metal-forming damage laws **along particle pathlines**
   through the FOM velocity and pressure fields, on 45 cases, at **under 2 %**
   throughput cost.
2. The result ranks tool geometries: **B-Flutes, which have the deepest lobes,
   accumulates the most damage on every statistic**.
3. The trajectories show *why*: damage is acquired in **discrete capture events**
   when a particle is entrained into orbit in the shear layer — so the ranking is
   really counting how often a given tool re-entrains material.

⚠️ **What this is not.** It is a *susceptibility ranking*, not a validated defect
prediction. There is no experimental defect data to validate against — see §9.

---

## 1. Why a void here is not a bubble

**Say:** FSW is solid-state. Peak temperature is 80–95 % of melting, so nothing
liquefies. A void is **material that failed to refill** behind the pin — a
volumetric bookkeeping failure, not nucleation.

**The three analogies people reach for, and why each fails:**

| analogy | the number | verdict |
|---|---|---|
| turbulence | Re ≈ 10⁻⁷–10⁻⁴ | **7–12 orders** below transition |
| cavitation | σ_cav ≈ 7×10⁴, needs ≲ 1 | wrong by ~11 orders **and wrong sign** |
| vorticity | rigid rotation gives D = 0 | large even in sound welds |

**Expect:** *"Why not just use vorticity?"*
**Answer:** the tool spins everything, so vorticity cannot discriminate. Damage
needs the **stretching** part of ∇u, not the spin part.

---

## 2. The variable: stress triaxiality

η = σ_m / σ_eq, with σ_m = −p and σ_eq the von Mises stress.

**Say:** stress splits into a **volume** part and a **shape** part. Metal *flows*
because of the shape part, but voids **open or close** because of the volume part.
η is the ratio. Negative → compression → voids close. Positive → tension → voids
open.

**Say if pressed:** σ_m = −p was *verified*, not assumed — the mean stress is
**compressive ahead** of the tool (forging) and **tensile in the wake**: median σ_m
−6 to −7.5 MPa ahead, +0.4 to +2.2 MPa behind. ⚠️ Quote the **sign pattern**, which
holds in **9 of 10** cases, rather than a single pair of numbers — the magnitudes are
case- and window-dependent. One case (A2old) has a weakly negative wake value.

---

## 3. Two damage laws, run together

* **Rice–Tracey** d(lnΦ)/dt = 0.849 · ε̇ · exp(1.5 η) — no fitted parameters
* **Cockcroft–Latham** dC/dt = max(σ₁,0)/σ_eq · ε̇ — one calibrated constant

**Why both:** they have different structure — one exponential in η, one a Macaulay
bracket on σ₁. **Agreement is cheap evidence; disagreement is diagnostic.** Neither
was invented for FSW; both are standard metal-forming laws.

### ⚠️ Expect: *"Where does 0.849 come from?"* — rehearse this

**It is not fitted.** Rice and Tracey (1969) solved the velocity field around an
isolated spherical void in a rigid-plastic matrix under remote triaxial loading and
obtained the amplitude **0.283** for the radial growth rate $\dot R/R$. A volume
fraction goes as $R^3$, so $\dot\Phi/\Phi = 3\,\dot R/R$, and
$3 \times 0.283 = \mathbf{0.849}$. That is the whole origin — a geometric factor of
3 on a constant from a 1969 analytical solution.

*J. R. Rice & D. M. Tracey, J. Mech. Phys. Solids 17(3), 201–217 (1969).*

### Expect: *"And the Cockcroft–Latham constant?"*

**The integrand has no fitted parameter.** The single constant is $C_{\text{crit}}$,
a **threshold** that says when the accumulated $C$ counts as failure — not a
coefficient inside the integral. It comes from a tensile or upset test; literature
values for aluminium alloys are ~0.2–0.5. ⚠️ **Say plainly that we have not
calibrated it for this alloy, so we use $C$ as a ranking only.**

*M. G. Cockcroft & D. J. Latham, J. Inst. Metals 96, 33–39 (1968).*

### How each law indicates damage — one sentence each

* **Rice–Tracey** tracks a physical **void volume fraction** that grows
  *exponentially* with triaxiality: tension opens voids, compression suppresses them.
* **Cockcroft–Latham** accumulates **only where σ₁ is tensile** — the Macaulay
  bracket makes the integrand exactly zero where σ₁ ≤ 0.

⚠️ **Do NOT say "so it switches off under the shoulder" — we checked, and it does
not.** σ₁ > 0 at ~95 % of active nodes inside the shoulder radius. Because
σ₁ = σ_m + σ_eq·ŝ₁ and this flow is shear-dominated, the deviatoric term (ŝ₁ ≈ +0.87)
dominates the volumetric one (η ≈ −0.05), so σ₁ stays tensile even where the mean
stress is compressive (55 % of nodes).

**Say this instead, it is the stronger point:** the two laws probe *different parts of
the stress tensor* — Rice–Tracey the volumetric state (and it *is* suppressed by
compression), Cockcroft–Latham mainly the deviatoric. They are not redundant.

### Expect: *"Why not GTN or Lee–Dawson?"*

**Frame it as sequencing, not dismissal.** These two were chosen for the first pass
because they need **no calibration** — we could run them today and get a defensible
ranking. The richer models stay on the table:

| model | what it would add | what it needs |
|---|---|---|
| **Lee–Dawson** | the published FSW precedent — He, Dawson & Boyce (2008) applied it to this exact problem | a coupled κ evolution ODE and ~10 constants; ⚠️ clamps at σ_m ≤ 0, so it cannot represent void **closure** |
| **GTN** | full porous plasticity with nucleation and coalescence | 7+ constants and a porous-plasticity solve inside the FOM |

⚠️ **Useful honesty:** He *et al.*'s reported porosity change was 0.01 % → 0.017 % —
a ~70 % relative change on a very small absolute number. Worth citing both as
precedent *and* as a caution about how much the answer depends on the constants.

⚠️ **State both limitations yourself before anyone asks:**
1. These are **growth** laws, not nucleation laws. Φ = 0 is a fixed point, so they
   need an assumed initial porosity Φ₀ that nobody measures directly.
2. Rice–Tracey has **no saturation term**. Φ is a volume fraction and cannot exceed
   1, reached at lnΦ = 9.21 for Φ₀ = 10⁻⁴ — but we measure up to **5242**, i.e.
   569× past failure. **So lnΦ is a ranking, not a predicted porosity.** Always say
   this; it is the single most likely thing to be caught on.

---

## 4. The tools — and what figC4 actually plots

**Say:** each dot is a **vertex of the tool's STL surface mesh** — the real
geometry, not a computed field. The two rows are the **same points in two
projections**: (x,y) looking down the axis, (x,z) from the side.

Measured from the STLs, not assumed from folder names:

| family | lobes | lobe depth |
|---|---|---|
| **B-Flutes** | 3, 4, 6, 8 | **20–32 %** of mean radius |
| C-Threads | 2 | 3–8 % |
| D-Concavity | 8 / 2 | 18 % / 4 % |
| A-Flats | — | **below the 1 % detection floor** |

⚠️ **"A-Flats has no lobes" is NOT a finding.** Their STLs are too coarse (2,794
triangles, no mid-pin surface) to resolve shallow flats. A data limitation, not a
geometric one. Say this explicitly — it is a trap.

**Also worth saying:** PinShapes is a **pure tool-geometry study at one operating
point** — rpm, advancing speed, thickness and material are effectively constant, so
pin geometry is the only real independent variable.

---

## 5. Implementation — the parts that will be questioned

**Say:** three nodal scalars (ε̇, η, σ₁/σ_eq) are precomputed on the mesh and
sampled exactly like velocity, at the element the velocity step already found.

### 5a. Why pathlines, not a grid

Damage is a **history integral**; a grid cell has no history. Following the particle
makes the transport term vanish exactly, so the update is `new = old + dt × rate` —
no gradient of the damage variable, no numerical smearing of the sharp gradients we
are trying to resolve.

### 5b. The rheology correction — tell this story, it builds credibility

**The question someone will actually ask: "can we just use the same `.mat` file?"**
**Answer: yes — the tables are the right data, they just need one conversion.**

The `.mat` viscosity tables are written in **shear-equivalent** measures; our
pipeline is von Mises. Both the rate and the stress pick up a √3, so the factor is
√3·(√3)^m = **1.816** at m ≈ 0.086.

| | ratio to the solver's own √(3J₂) |
|---|---|
| tables used as-is | **0.569** |
| with the conversion | **1.035** |

Predicted **1.816** vs measured **1.819** — **0.13 %**, over 581,488 cells.

⚠️ **This is the material VISCOSITY convention — not the "modified Norton" FRICTION
law.** Those are two different things and the names are close enough to be confused.
The friction law is the boundary condition with the tanh profile; this is the
material model.

**Why the story is worth telling:** it was found by *measurement against the solver's
own exported stress*, not by reading documentation — and the 0.13 % agreement is what
makes it a confirmed derivation rather than a fitted number. Source: Venghaus, *Finite
Elements in Analysis & Design* **224** (2023) 103986, Publication 1, eq. (5).

⚠️ **If asked whether Henning should confirm it:** we do not need him to — the answer
is in his published paper and the 0.13 % match verifies it. Worth mentioning to him as
a courtesy, not as an open question.

### 5c. RK4 for position vs quadrature for damage

⚠️ **This one gets misunderstood — be precise.** They are different problems:

| | position *x* | damage accumulator Φ |
|---|---|---|
| ODE | dx/dt = u(x(t), t) | d(lnΦ)/dt = f(x(t), t) |
| does the **unknown** appear on the right? | **yes** — x does | **no** — lnΦ does not |
| what it is | genuine IVP | **quadrature along a known path** |

⚠️ **Expect this question, it is a fair one:** *"surely η and ε̇ are time-dependent?"*
**Yes — strongly**, in space and in time. The row does *not* say the right-hand side
is constant. It says **lnΦ does not appear on its own right-hand side**, so there is
no feedback: along a path the position solve has already fixed, the equation is
d(lnΦ)/dt = f(t). That is what makes it a quadrature rather than a coupled solve.

**RK4 for position is genuinely needed** — at r = 3 mm the cohort sweeps **9° and
0.47 mm of arc per step**, so Euler cuts the corner element after element. The
earlier decision to use RK4 stands.

For the accumulator, order 4 is **not** formally 4th order — the P1 driver is C⁰
with a kink at every element face. Its benefit is **sampling the midpoint**.
Measured A/B, identical seed and steps:

| | order 1 | order 4 |
|---|---|---|
| **position max difference** | — | **0.000e+00 m** |
| lnΦ median | 0.004456 | 0.004456 (**+0.01 %**) |
| lnΦ **max** | 179.85 | **147.67 (−17.9 %)** |

⚠️ **Say it is a TAIL correction, not a general accuracy win.** A typical particle
changes by 0.05 %; the maximum drops 18 %. It changes `frac_failed` and p99
rankings, and leaves median rankings alone.

### 5d. Cost

| platform | slowdown |
|---|---|
| RTX 5090 (CUDA) | +0.15 … +0.51 % |
| LUMI MI250X (ROCm) | +1.77 % |
| production rate, RK4 damage + VTU export | **119,525 p·step/s** (−2.3 % vs Euler — within noise) |

**Say:** the accumulator stays on the GPU for the whole run. The marginal cost is
extra gathers from memory already in cache, at elements the velocity step already
located — **no extra element searches**, which is what actually costs.

---

## 6. The run

45 cases, **30 mm** of travel, RK4 damage, VTU export every 50 steps.
**All 45 completed, zero failures.**

⚠️ **Be honest about the first attempt.** At 10 mm the bundle had **not cleared the
tool** — PinShapes particles had a mean displacement of **−0.52 mm**, i.e. they were
still recirculating beside the pin. Tripling the travel fixed it:

| | 10 mm | 30 mm |
|---|---|---|
| ps_A1 mean x | **−0.52 mm** | **+18.95 mm** |
| past x > +7 mm | — | **97.2 %** |
| still inside r < 7 mm | 22.5 % | **2.6 %** |

**If asked why not just narrow the seed box:** because the bypass particles are
physically meaningful — they are the material that fills the wake — so we separate
them in **analysis** rather than discard them in **sampling**.

---

## 7. The wake box — and why each wall is where it is

| wall | where | why |
|---|---|---|
| x_min | **+7 mm** | after the shoulder: drops the tool footprint and the 2.6 % still orbiting |
| x_max | downstream front | — |
| y | full span | **kept** — the advancing/retreating contrast lives here |
| z_min | domain | **kept** — the weld root matters |
| z_max | domain **− 8 % of thickness** | drops the BC-contaminated top band |

**On z_max, if asked:** the top band sits against the prescribed-velocity surface
and its p99 is **5–16× the bulk** — that is a boundary-condition response, not
damage. Dropping 8 % of thickness (0.36–0.48 mm) removes it.

⚠️ **"2–3 particle layers" could not be used literally** — seeding is random, so z
is continuous (6001 distinct values in 100k particles). 8 % of thickness is the same
physical scale but mesh-independent.

**The box keeps ~90 % of all particles** — a targeted exclusion, not a filter that
throws the statistics away.

---

## 8. Results — what to put weight on

### 8a. The two models agree

Rank correlation **+0.982 to +0.998** on all 45 cases. Two structurally different
laws, two tool families, opposite rotation senses. **This is the implementation
self-check**, so lead with it.

### 8b. Family ranking (averaged over the wake box)

| family | ln_mean | ln_med | ln_p99 | frac_failed |
|---|---|---|---|---|
| A-Flats | 6.332 | 0.2413 | 69.0 | 19.4 % |
| **B-Flutes** | **7.166** | **0.2519** | **79.8** | **20.9 %** |
| C-Threads | 6.316 | 0.2500 | 69.7 | 20.1 % |
| D-Concavity | 2.650 | 0.1745 | 39.9 | 9.0 % |
| cohort | 5.554 | 0.9146 | 49.4 | 17.6 % |

**Say:** B-Flutes is highest on **every** statistic, and it has the deepest lobes
(20–32 % of pin radius). A and C sit within 0.2 % of each other, matching their
similar shallow geometry. **This is the cleanest family separation we have.**

⚠️ **Two caveats to state yourself:**
1. **D-Concavity's low score and its low wake count are the SAME phenomenon** — see
   §8d. A seventh of its cross-section is unfilled, so fewer particles arrive *and*
   the average over those that do is depressed. Do not present D's damage number as
   a like-for-like comparison.
2. The A/B/C spread is ~9 %, not a factor. It is a **consistent ordering**, not a
   dramatic one.

### 8c. The flank asymmetry — a partly open result

| | advancing | retreating | ratio | adv > ret |
|---|---|---|---|---|
| **PinShapes** (CCW) | 0.2292 | 0.2317 | **0.989** | **12 / 20** |
| cohort + val (CW) | 0.7704 | 1.2013 | 0.641 | **0 / 25** |

**Say:** in the earlier short run the advancing side was lower in **45/45**, which
contradicted the Stage 1 η result. With the proper travel distance and the wake box,
**PinShapes balances out** — so that contradiction was largely a measurement
artefact. But a **cohort-specific retreating-side excess survives** and is **not
explained**. Say so plainly; do not paper over it.

---

## 8d. The void analysis — a NEGATIVE result, and how to present it

⚠️ **This section replaced two earlier versions, both wrong.** If you have shown
anyone a draft claiming either (a) the D-Concavity void is a real filling failure, or
(b) the A-Flats A2 void is a defect confirmed by two independent indicators — **both
claims are retracted**. Say so plainly if it comes up; it is a better look than
letting it pass.

### What to say

The method *does* produce empty regions in the weld cross-section. We investigated
whether they are predicted defects. **They are not** — they are a particle-tracking
artefact, and we can show that quantitatively.

Across all 20 tool cases:

| | Spearman ρ |
|---|---|
| void area vs **trapped particle fraction** | **+0.980** |
| void area vs trapped/wake damage ratio | **−0.932** |
| same, **excluding the D-Concavity family** | **+0.954** |

**Say:** excluding D does not weaken it, so this is not "one odd geometry" — it holds
within A-Flats, B-Flutes and C-Threads. Void area is predicted by how many particles
get stuck at the tool.

### The clinching argument, if someone pushes

A particle genuinely held in the shear layer should accumulate damage **hard** — which
is exactly what the trajectory analysis (§10) shows happens. B-Flutes and C-Threads
deliver that: their trapped particles carry **600–1000×** the wake median damage.

| family | trapped | trapped ÷ wake damage |
|---|---|---|
| B-Flutes | 2.7–3.0 % | **868–1025** |
| C-Threads | 3.2–3.4 % | 638–769 |
| **D-Concavity** | **17.1–18.7 %** | **8–23** |

D's trapped particles sit at r ≈ 3.65 mm — *inside* the shear layer — and accumulate
almost nothing. They are being circulated by the interpolated velocity field without
experiencing the strain rates that proximity to the tool implies. **That is a
numerical trap, not a weld feature.**

### Why "two indicators agreed" was not evidence

⚠️ **State this yourself** — it is the subtle part, and the mistake is instructive.
Particle density is *directly* depleted by trapping. The damage field is shaped by
*which particles survive to reach the wake*. **Both are downstream of the same
failure**, so their agreement corroborated nothing. Two indicators are only
independent if their failure modes are.

### What is unaffected

✅ **The damage-based tool ranking stands.** B-Flutes highest on every statistic. It
is computed over the wake population and never uses the void measure.

### The next step, if asked

**Count particles crossing a control surface around the tool, in versus out.** In a
steady incompressible flow they must balance. **This needs no new simulation** — the
exported trajectory series already contains it. Then measure the divergence of the
interpolated field, and check whether the level-set treatment leaks where the tool
surface cuts through elements.

**How to frame it:** a negative result that we found ourselves, with a quantitative
diagnosis and a concrete fix, is a stronger position than an unexamined positive one.

---

## 9. ⚠️ The honest limitation — rehearse this answer

**There is no experimental defect ground truth.** No macrograph, no CT scan, no
sectioned weld — not in this work, and not in the published FOM validation, which
covers forces, torque and thermocouples only.

**So the correct claim is:** this produces a **physically grounded susceptibility
ranking** with no fitted parameters, which agrees with an independent flow/refill
analysis. It is **not** a validated defect prediction, and we do not claim it is.

**If someone pushes:** the nearest available external check is the F_y > F_x force
criterion, which can be applied to all cases and compared against experimentally
validated quantities at no experimental cost. That is the proposed next step.

---

## 10. The mechanism — the newest and most compelling result

**Say:** we followed individual particles through all 167 exported states, coloured
by the damage accumulated so far, so a path **brightens exactly where damage is
picked up**.

**Four findings:**

1. **Capture, not gradual shear.** A particle approaches, is **captured into orbit**
   around the pin, circles several times while brightening, then releases
   downstream. One trace rises **three orders of magnitude in a single step**
   (0.06 → 100) and is then flat for ~45 more revolutions.
2. **Final damage ≈ a count of capture events.** The high traces are staircases,
   with bursts at steps ~1000, ~3300, ~5200 separated by flat stretches.
3. **Every jump coincides with r dropping below the 7 mm shoulder line.** The damage
   panel and the radius panel are locked together.
4. **The bypass particles are unprocessed, not weakly damaged** — flat at ~0.1
   throughout, never approaching the tool. This justifies excluding them.

**The payoff sentence:** the capture orbits sit at **r ≈ 3–5 mm**, between pin
(2.4 mm) and shoulder (7 mm) — so **the trajectories locate the shear layer
directly**, with no velocity field needed.

**And the side view adds the depth:** the orbits are confined to **z ≈ −0.5 to
−2 mm**, the upper third of a 6 mm plate, directly under the shoulder. **Damage is
acquired near the top surface, not through the thickness** — which is consistent
with the wake cross-section, where the damaged band is widest at the top.

⚠️ This also explains why the `z_top` band had to be excluded carefully rather than
ignored: the real damage and the boundary artefact live at the *same* depth, so a
crude top-surface cut would remove signal along with the artefact. The 8 %
thickness cut is deliberately shallow for this reason.

**Why it matters for the ranking:** the integral is strongly concentrated in time —
**the median particle collects 84 % of its damage in 10 % of its steps**, against
10 % if it were uniform, and 87 % of particles exceed 50 %. So the metric is really
measuring **how often a tool re-entrains material** — a geometric property, which is exactly what we want to rank.

⚠️ **These are 4 strata × 3 particles, deliberately not a random sample.** It is a
**mechanism illustration**; the population numbers are in the wake CSV. Say this, or
someone will reasonably object that twelve particles prove nothing.

**If asked how you know it is the same particle:** `ParticleID` is a positional
index fixed at seeding and the kernel never reorders. Verified three ways — IDs
identical across files; row displacement over 50 steps 0.18 mm vs 12.98 mm for a
shuffled control; and **lnΦ decreased in 0 of 100,000 rows**, which is a physical
invariant any row-swap would violate.

---

## 11. Likely questions, with answers

| question | answer |
|---|---|
| *Is lnΦ = 5242 a porosity?* | **No.** No saturation term; it is a ranking. Failure is at 9.21. |
| *Why random seeding, not grid?* | Measured: final-time neighbour spacing has **CV = 38 %**, so the flow destroys seed-time order within a fraction of a revolution. Grid would buy exact reproducibility and uniform inflow weight — worth it only if we deposit a continuous field. |
| *Why not finer bins in the cross-section?* | The limit is **particle count**, not bin size: at 0.18 mm mean spacing a 600×240 grid leaves **54 % of cells empty**. We use a k-nearest-neighbour median instead, which gives 0.05 mm resolution with 16 real samples per point. |
| *Does damage affect the flow?* | **No — one-way coupling.** Damage is a passive diagnostic; it does not feed back into the FOM. |
| *Why is the cohort's median so much higher than PinShapes'?* | Different meshes, speeds and dt; the families are **not** directly comparable on absolute value. Compare within a family. |
| *How do you know the advancing side is where you say?* | Rotation sense is read from the **signed RPM** in each case's own `data/*.som.dat`, and verified by sampling u_θ at r/R ≈ 0.4. PinShapes is CCW (+996), the cohort CW (−400…−800) — **opposite senses**, which is why each case is plotted against its *own* advancing side. |

---

## 12. Figure inventory

| figure | file | status |
|---|---|---|
| Pin families | `figs_concept/figC4_pin_shapes.png` | ✅ |
| Geometry / level-set trap | `figs_concept/figC1_geometry.png` | ✅ |
| Pathline schematic | `figs_concept/figC5_pathline.png` | ✅ |
| Models agree | `figs_phase4/fig2_models_agree.png` | ✅ |
| Radial profile | `figs_phase4/fig3_radial_profile.png` | ✅ |
| Advancing wake | `figs_phase4/fig4_advancing_wake.png` | ✅ |
| Accumulation | `figs_phase4/fig5_accumulation.png` | ✅ |
| Wake yz field, B-Flutes | `figs_wake/figWB_B-FluteVariations_yz_field.png` | ✅ |
| Trajectories, time | `figs_traj/figT_ps_A-FlatsVariations_A1_time.png` | ✅ |
| Trajectories, top + side view | `figs_traj/figT_ps_A-FlatsVariations_A1_3d.png` | ✅ two true-scale orthographic views (x–y and x–z) |
| Wake yz field, other families | `figs_wake/figWB_{A,C,D,cohort}_*_yz_field.png` | ✅ generated, not yet in the deck |
| Trajectories, B-Flutes case | `figs_traj/figT_ps_B-FluteVariations_B2_*` | ⏳ **not yet generated** — would test whether B's higher damage comes from *more frequent* capture |
| Trajectories, D-Concavity case | `figs_traj/figT_ps_D-ConcavityTilt_D1_*` | ⏳ **not yet generated** — would test whether D's deficit is particles never being captured |

⚠️ The two ⏳ trajectory cases are the natural next step and would directly test the
geometric explanation for the family ranking. Both are read-only on existing data.

---

## 13. Where everything lives

| | |
|---|---|
| Deck | `damage_implementation_talk.md` (Marp) |
| Run output | `/scratch/project_465002752/hashemia/damage/phase4_rk4_30mm/` (24 GB, 167 VTU/case) |
| Wake metrics | `results/phase4_rk4_30mm_wake.csv` |
| Summary | `results/phase4_rk4_30mm_summary.csv` |
| Regional | `results/phase4_rk4_30mm_regional.csv` ⚠️ `stir_nobc` is **empty** at 30 mm — use the wake CSV |
| Open items | `OPEN_QUESTIONS.md` (O1–O42) |
| Concept cards | `CARDS_one_concept_each.md` |
