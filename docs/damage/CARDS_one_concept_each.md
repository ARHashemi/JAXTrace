# Concept cards — one idea per card, no reading order

**For the moment you get stuck.** Each card is self-contained: it does not assume
you read the previous one. Jump straight to the term that stopped you, read ~10
lines, and go back to what you were doing.

The primer (`PRIMER_fsw_physics_for_me.md`) is organised for *reading through*.
This file is organised for *looking up*. Same physics, different access.

**If you only ever read one card, read [#1 Stress](#1-stress).** Everything else
in the method is built on it.

> 🖼 **Figures for these concepts** are in `figs_concept/` (see `FIGURES_INDEX.md`):
> **C1** the geometry and which side is advancing · **C2** the stress split ·
> **C3** how η's sign decides open vs close · **C4** the four real pin shapes ·
> **C5** how damage accumulates along a route. Cards 1, 3, 4 pair with C2; card 6
> with C3; cards 7 and 12 with C1; cards 8 and 9 with C5.

---

## Index

| # | card | one-line answer |
|---|---|---|
| 1 | [Stress](#1-stress) | force per unit area inside the material |
| 2 | [Strain rate](#2-strain-rate) | how fast the material is being deformed |
| 3 | [Mean stress σ_m](#3-mean-stress-m) | the squeeze/pull part — changes volume |
| 4 | [Deviatoric stress](#4-deviatoric-stress) | the shape-changing part — causes flow |
| 5 | [von Mises σ_eq](#5-von-mises-equivalent-stress-_eq) | one number for "how hard is it being sheared" |
| 6 | [Triaxiality η](#6-stress-triaxiality-) | **the key variable**: σ_m/σ_eq, sign decides voids |
| 7 | [Advancing vs retreating](#7-advancing-vs-retreating-side) | which side of the weld; defects go on advancing |
| 8 | [Pathline](#8-pathline) | the route one piece of material travels |
| 9 | [Why pathlines, not a grid](#9-why-pathlines-and-not-a-grid) | damage is a *history*, grids have none |
| 10 | [Effective viscosity](#10-effective-viscosity-_eff) | treating hot metal as a very thick fluid |
| 11 | [Norton law](#11-the-norton-law) | the formula the FOM used for strength |
| 12 | [Level-set LEVEL](#12-level-set-the-level-field) | the number that says "inside the tool" |
| 13 | [Reynolds number](#13-reynolds-number) | why turbulence is irrelevant here |
| 14 | [Filling failure](#14-filling-failure-the-actual-void-mechanism) | what actually makes a void in FSW |
| 15 | [Rice–Tracey & Cockcroft–Latham](#15-the-two-damage-models) | the two formulas we integrate |
| 16 | [FOM vs ROM](#16-fom-vs-rom) | the slow truth vs the fast surrogate |

### Part B — their vocabulary (colleagues' decks)

| # | card | one-line answer |
|---|---|---|
| B1 | [LE/RS/TE/AS](#b1-le--rs--te--as--the-four-sectors) | the four sectors they measure flux through |
| B2 | [Refill V_θ, V_z](#b2-refill-volume-v_--v_z) | **their equivalent of our η** |
| B3 | [V_PD](#b3-v_pd--plastic-deformation-zone-volume) | deformed volume; independently confirms our A1 result |
| B4 | [Forces F_x, F_y](#b4-forces-f_mag-f_x-f_y-force-angle-oscillations) | **the only experimentally validated quantity** |
| B5 | [Intra-revolution](#b5-intra-revolution-behaviour) | variation within one turn |
| B6 | [VP_high/VP_low](#b6-vp_high--vp_low--pressure-regions-in-the-deposition-cavity) | **direct overlap with our σ_m = −p** |
| B7 | [Flow depth](#b7-flow-depth) | how deep the stirring reaches |
| B8 | [Tool wear: Archard/Usui](#b8-tool-wear-archard-and-usui-models) | a separate workstream — tool, not weld |
| B9 | [SZ / TMAZ / HAZ](#b9-the-weld-zones-szwn-tmaz-haz) | the standard weld zones |
| B10 | [POD/GPR/multi-fidelity](#b10-rom-machinery-pod-gpr-multi-fidelity-reference-space-projection) | the ROM plan |

---

## 1. Stress

**Force per unit area, acting inside the material.** Pressure is the everyday
example: push on a surface, the material inside carries that load.

Why it needs 6 numbers and not 1: at any point, force can act *along* a surface
(shear, like sliding cards in a deck) or *perpendicular* to it (squeeze/pull). You
need both, on three perpendicular planes. That gives a 3×3 table, symmetric, so 6
independent numbers. Engineers call it a **tensor**; it is just a bookkeeping
table.

**The formula:**

$$\sigma_{ij} = \underbrace{\sigma_m\,\delta_{ij}}_{\text{volume part}} + \underbrace{s_{ij}}_{\text{shape part}}$$

Read `σ_ij` as "the stress table", `δ_ij` as "1 on the diagonal, 0 elsewhere", and
the two terms as the two parts below.

For our purposes it splits cleanly into **two** parts, and only the split matters:

| part | what it does | card |
|---|---|---|
| **mean stress σ_m** | changes **volume** — squeezes or pulls apart | [#3](#3-mean-stress-m) |
| **deviatoric s** | changes **shape** — makes material flow | [#4](#4-deviatoric-stress) |

> **The one sentence to keep:** metal flows because of the *shape-changing* part,
> but voids open or close because of the *volume-changing* part. That is why the
> ratio between them ([#6](#6-stress-triaxiality-)) is the variable we track.

---

## 2. Strain rate

**How fast the material is being deformed**, in units of 1/s.

Stretch a 10 mm bar to 11 mm in one second: strain is 10 %, strain *rate* is
0.1 s⁻¹. In the FSW stir zone it is **30–60 s⁻¹** — the material is being worked
enormously fast.

**The formula:**

$$\dot\varepsilon_{\text{eff}} = \sqrt{\tfrac{2}{3}\,D_{ij}D_{ij}}, \qquad D_{ij} = \tfrac12\left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right)$$

In words: take the velocity gradient, keep its **symmetric** part (that is `D`,
the stretching), square every entry, sum, times 2/3, square root. The 2/3 is chosen
so a simple pull test at rate ε̇ returns exactly ε̇.

It comes from the velocity field. If neighbouring points move at different
velocities, the material between them is deforming.

⚠️ **One subtlety that matters a lot.** Velocity gradient has two parts:

* **stretching** — genuinely deforms the material (this is strain rate)
* **spin** — rigid rotation, like a spinning coin: no distortion at all

The tool spins everything, so there is huge *spin* everywhere, even in a perfect
weld. **Only the stretching part damages material.** This is why "high vorticity"
is not a defect indicator — vorticity *is* the spin part.

**Accumulated strain** is strain rate integrated along a particle's journey. In the
stir zone it reaches **10–100** (i.e. 1,000–10,000 %), which is why the metal
recrystallises into fine grains.

---

## 3. Mean stress σ_m

**The pure squeeze-or-pull part of stress.** Average of the three
perpendicular normal stresses. It changes **volume**, not shape.

| | |
|---|---|
| σ_m **negative** = compression | material squeezed → **voids close** ✅ |
| σ_m **positive** = tension | material pulled apart → **voids open** ⚠️ |

**The formula:**

$$\sigma_m = \tfrac13\left(\sigma_{11}+\sigma_{22}+\sigma_{33}\right) = \tfrac13\sigma_{kk}$$

**Relation to pressure: σ_m = −p.** Solid mechanics counts tension as positive;
fluid solvers count compression as positive. So they are the same quantity with
opposite sign.

⚠️ **Getting this sign wrong inverts every conclusion** — voids would appear to
grow exactly where they actually close. We did not assume it; we measured it:

| region | pressure |
|---|---|
| **ahead** of the tool (material being forged) | **+1.70 MPa** |
| **behind** the tool (the wake) | **−1.35 MPa** |

Pressure is high where material is being pressed together, which confirms
compression-positive, which confirms σ_m = −p.

---

## 4. Deviatoric stress

**The shape-changing part of stress** — what is left after you remove the mean
([#3](#3-mean-stress-m)). Written **s**.

It is the part that makes metal **flow**. Squeeze a metal cube equally on all six
faces, even at enormous pressure, and it will not flow — it just shrinks slightly.
Squeeze it unequally and it deforms. That experimental fact (Bridgman, 1940s) is
why plasticity depends only on the deviatoric part.

**The formula:**

$$s_{ij} = \sigma_{ij} - \sigma_m\,\delta_{ij}, \qquad s_{kk} = 0$$

Its defining property: the three normal components sum to zero. It is "pure
distortion with no volume change".

> **The tension at the heart of this project:** flow depends only on **s**, but
> void growth depends on **σ_m** too. So σ_m drops out of the flow calculation and
> comes back — exponentially — in the damage calculation.

⚠️ Useful fact we exploited: the FOM exports `Stress` as **the deviator only**
(verified: its trace is 3e-8 Pa, i.e. machine zero). So pressure is stored
separately, and the solver's own σ_eq can be recovered from it with no material
model at all — which is how we found a 1.8× convention error in our own σ_eq.

---

## 5. von Mises equivalent stress σ_eq

**One number summarising "how hard is this material being sheared"**, built from
the deviatoric part ([#4](#4-deviatoric-stress)).

Why it exists: the deviator has 6 components, and you cannot compare 6 numbers
against a single material strength. σ_eq collapses them into one, scaled so that
in a simple pull test σ_eq equals the stress you applied.

**The formula:**

$$\sigma_{\text{eq}} = \sqrt{\tfrac{3}{2}\,s_{ij}s_{ij}} \;=\; \sqrt{3J_2}, \qquad J_2 = \tfrac12 s_{ij}s_{ij}$$

(`J_2` is the "second invariant" — the literature calls von Mises plasticity
"J2 flow theory" for this reason.)

**What it is for:** the material flows when σ_eq reaches the **flow stress** —
the strength of the material at that temperature and strain rate. In the FSW stir
zone, σ_eq ≈ 100 MPa.

⚠️ In our work σ_eq also plays a second role: it is the **denominator** of
triaxiality ([#6](#6-stress-triaxiality-)). So an error in σ_eq biases every
triaxiality number — which is exactly what happened (we were low by 1.816× from a
shear-vs-von-Mises convention in the material tables) and why checking it against
the solver's own stress mattered.

---

## 6. Stress triaxiality η

**The key variable of the whole method.** η = σ_m / σ_eq.

Read it as: *"how much volume-changing stress is there, relative to the
shape-changing stress doing the deforming?"*

| η | meaning | voids |
|---|---|---|
| **negative** | compression dominates | **close** ✅ |
| ≈ 0 | pure shear | roughly neutral |
| **positive** | tension dominates | **open** ⚠️ |

**The formula:**

$$\eta = \frac{\sigma_m}{\sigma_{\text{eq}}} = \frac{-p}{\sqrt{3J_2}}$$

**Why a ratio and not just σ_m?** Because the same tension matters more in weakly
deforming material than in strongly deforming material. Dividing by σ_eq makes it
dimensionless and comparable between regions and between cases.

**Why it is the right variable:** void growth depends on it **exponentially** —
roughly exp(1.5η). So modest changes in η produce large changes in void growth
rate, which is what makes it a discriminating indicator rather than a weak
correlate.

**Our headline measurement:** in the advancing-side wake, **94 % of nodes have
η > 0** (tension, voids open); in the retreating-side wake, **1 %**. With no fitted
parameters. That is the literature's void signature, recovered from the FOM fields.

---

## 7. Advancing vs retreating side

Jargon, but unavoidable — and the single most common thing to get backwards.

The tool both **spins** and **travels**. On one side of the weld those two motions
point the **same** way; on the other they **oppose**.

| side | rotation vs travel | |
|---|---|---|
| **advancing** | same direction → high relative speed | **defects form here** |
| **retreating** | opposite → low relative speed | material flows more easily |

⚠️ **Defined relative to TOOL TRAVEL, not material flow.** We initially had this
backwards. In the simulations the tool is stationary and material flows past it, so
you must convert: material flows +x ⇒ the tool effectively travels −x ⇒ advancing
is the flank where the tool-surface velocity points −x.

⚠️ **The two case families rotate in opposite senses** — PinShapes
counter-clockwise (RPM +996), the cohort clockwise (RPM −400…−800). So "advancing"
is on the **opposite physical side** in the two sets. Our code *measures* it per
case rather than assuming, and agrees with the signed RPM in **64 of 64** cases.

---

## 8. Pathline

**The route one piece of material actually travels.** Drop a speck of paint in the
flow and film it: that trace is a pathline.

Not to be confused with a **streamline**, which is a snapshot of flow direction at
one instant. For steady flow they coincide; for a threaded or tilted pin — where
the field changes as the tool rotates — they differ, and **only the pathline is
physically meaningful** for anything accumulated over time.

We compute them with RK4, a standard numerical integrator, taking 4 velocity
samples per step for accuracy. ~100,000 particles per case.

---

## 9. Why pathlines, and not a grid

This is the single most important design decision, and it has a clean reason.

**Damage is a history.** A piece of material becomes damaged because of everything
that happened to it along its whole journey — not because of conditions at its
current location.

```
damage(particle) = ∫ (damage rate) dt     along that particle's own route
```

⚠️ **A grid cell has no history.** It has whatever material happens to be passing
through it right now. Ask "how damaged is this cell?" and there is no answer — only
"how damaged is the material currently here", which depends on where that material
came from.

**The formal version (why it is also easier):** to track an accumulated quantity in
a fixed grid you must include an extra transport term for material flowing in and
out. Discretising it introduces numerical smearing — which would blur exactly the
sharp advancing-side gradient we are trying to resolve.

**The formula.** For any quantity φ carried by the material:

$$\underbrace{\frac{D\phi}{Dt}}_{\text{following the particle}} = \underbrace{\frac{\partial\phi}{\partial t}}_{\text{change at a fixed point}} + \underbrace{u_j\frac{\partial\phi}{\partial x_j}}_{\text{transport term}}$$

Follow the particle instead and that transport term **vanishes exactly**. The equation becomes
an ordinary `new = old + dt × rate`. That is why the implementation is one line, and
why it is correct.

---

## 10. Effective viscosity μ_eff

**A trick for treating hot metal as a very thick fluid.**

Hot aluminium in FSW is not molten (peak temperature is 80–95 % of melting), but it
flows like putty. Engineers model it as a fluid by defining

> μ_eff = (flow stress) / (3 × strain rate)

$$\mu_{\text{eff}} = \frac{\sigma_{\text{flow}}}{3\,\dot\varepsilon_{\text{eff}}}$$

which for FSW gives **10⁵–10⁶ Pa·s** — about ten million times thicker than honey.

⚠️ **It is not a material property.** It is a restatement of the material's
*strength*, so it changes with temperature and strain rate. Quoting "the viscosity
of aluminium in FSW" as a single number is a convenience, not a fact: the real
quantity is strongly **shear-thinning** (roughly μ_eff ∝ 1/strain rate), so it
falls by ~half every time the strain rate doubles.

---

## 11. The Norton law

**The formula the FOM actually used** for material strength:

> σ_eq = VISCO(T) × (strain rate)^EXPVI(T)

Both coefficients are **tabulated against temperature** in each case's `.mat` file,
measured experimentally for Al 6063-T6.

Reading the two numbers (from a real case):

| T | VISCO | EXPVI | σ_eq at 50 s⁻¹ |
|---|---|---|---|
| 25 °C | 1.13e8 | 0.023 | 124 MPa |
| 675 °C | 6.5e6 | 0.197 | 14 MPa |

**VISCO falls 17×** with temperature — that is thermal softening, and it is why
frictional heating makes the process possible at all.

**EXPVI is the rate sensitivity**, and it is *small*. Doubling the strain rate
raises strength by only **1.6 % when cold, 15 % when hot**.

⚠️ **That small exponent has a big consequence for us:** σ_eq is remarkably
insensitive to strain-rate errors. So in η = σ_m/σ_eq, almost all the error comes
from **pressure**, not velocity. This is why ROM *pressure* accuracy is the binding
constraint on the whole method — not velocity accuracy, which is what one would
naively optimise.

---

## 12. Level-set: the `LEVEL` field

A number stored at every mesh node that encodes **where the tool is**:

| LEVEL | meaning |
|---|---|
| **< 0** | inside the tool (no material here) |
| **> 0** | inside the workpiece |
| = 0 | the tool surface |

The FOM uses a fixed mesh and moves the *tool* through it, so it needs a way to
mark which parts of the mesh are currently occupied by tool rather than metal. That
is what this does.

We use it for three things: finding the tool axis and radius, excluding tool
interior from any statistic, and detecting which flank is advancing
([#7](#7-advancing-vs-retreating-side)).

⚠️ One trap we hit: the **maximum radius** of the LEVEL<0 region is the
**shoulder** (7 mm), not the pin (~2.4 mm). A sampling window sized from it was
therefore in the wrong place, which produced a spurious correlation that took real
work to find and fix.

---

## 13. Reynolds number

**The number that decides whether a flow is turbulent.** Re = (inertia) /
(viscous resistance).

$$\mathrm{Re} = \frac{\rho\,U\,L}{\mu_{\text{eff}}}$$

ρ = density, U = a characteristic speed, L = a characteristic length.

| flow | Re |
|---|---|
| water in a pipe, turbulent | > 4,000 |
| **FSW stir zone** | **10⁻⁷ – 10⁻⁴** |

**We are 7 to 12 orders of magnitude below turbulent transition.** The material is
so viscous ([#10](#10-effective-viscosity-_eff)) that inertia is utterly
negligible — this is "creeping flow", the regime of glaciers and honey, not of
mixing tanks.

**Why this matters for the project:** it is the first of three quantitative reasons
the turbulence/bubble-formation analogy does not transfer to FSW. There is no
turbulence to model. (The others: cavitation is impossible by ~11 orders of
magnitude and has the wrong sign; and vorticity is large even in perfect welds, so
it cannot indicate defects.)

---

## 14. Filling failure: the actual void mechanism

**A void in FSW is empty space the material failed to refill** — not a bubble.

```
tool moves forward  →  leaves a cavity behind the pin
                    →  material must flow around and fill it
                    →  if not enough arrives in time, the gap stays
                    →  the tool moves on, and the gap becomes a tunnel
```

⚠️ **This is NOT cavitation.** Cavitation is liquid boiling under low pressure. Here
there is no liquid (solid-state, never melted), the stir zone is under ~100 MPa
*compression* not tension, and aluminium's vapour pressure at these temperatures is
~10⁻⁹ Pa. Wrong by 11 orders of magnitude *and* the wrong sign.

So a void is a **volumetric bookkeeping failure**: material in ≠ cavity volume.

**Why it lands on the advancing side:** that is where the relative speed between
tool and material is highest, so material is swept away fastest and has least time
to flow back in. This is also where the stress state goes tensile — which is what
our η measurement detects, and why the two agree.

---

## 15. The two damage models

Both predict how an existing void grows, both borrowed from metal forming (not
invented for FSW), and we run **both** — agreement is evidence, disagreement is
diagnostic.

### Rice–Tracey — **no fitted parameters**

> d(ln Φ)/dt = 0.849 × (strain rate) × exp(1.5 η)

Φ is void volume fraction. The structure says: voids grow in proportion to **how
much deformation** is happening, multiplied by an **exponential in triaxiality**
([#6](#6-stress-triaxiality-)). The exponential is why η is so discriminating.

⚠️ **Two limitations to state honestly.** It is a *growth* law — Φ=0 stays 0
forever, so it cannot predict where voids *start*, and needs an assumed initial
porosity nobody measures. And it has **no saturation**: Φ cannot physically exceed
1, but the formula keeps growing past that. So **treat it as a ranking, not a
porosity.**

### Cockcroft–Latham — **one calibrated constant**

> dC/dt = max(largest tensile stress, 0) / σ_eq × (strain rate)

The `max(…, 0)` switches damage **off** under compression automatically, which
makes it discriminate wake from shoulder for free.

**Result:** across 35 cases the two models rank particles the same way, with rank
correlation **+0.982 to +0.998**. Two structurally different formulas agreeing is
the cheapest strong evidence available.

---

## 16. FOM vs ROM

| | what it is |
|---|---|
| **FOM** — Full-Order Model | the real finite-element simulation. Accurate, hours per case. **Our ground truth.** |
| **ROM** — Reduced-Order Model | a fast surrogate built from FOM results. Seconds per case, approximate. |
| **POD** | the maths that builds the ROM: find the few dominant patterns in many FOM results (essentially PCA on simulations) |

**The project goal:** a digital twin needs answers in seconds, so it needs the ROM.
Everything we have done is on **FOM** data deliberately — establish the physics on
the trustworthy data first, then check whether the fast surrogate preserves it.

⚠️ And we know what to watch when that happens: by
[#11](#11-the-norton-law), the damage metric is far more sensitive to **pressure**
than to velocity. A ROM that nails velocity and approximates pressure will produce
confident wrong answers.

---

## Part B — their vocabulary (the colleagues' decks)

Terms used in the internal presentations and the comparison document that are **not
part of our method**. You need them to follow their results and to connect yours.

---

### B1. LE / RS / TE / AS — the four sectors

They divide the region around the pin into four quadrants by angle and measure
material flux through each:

| | |
|---|---|
| **LE** | Leading Edge — ahead of the tool, material arriving |
| **AS** | Advancing Side — see card [#7](#7-advancing-vs-retreating-side) |
| **TE** | Trailing Edge — behind the tool, the wake |
| **RS** | Retreating Side |

⚠️ **This is their equivalent of our sampling window**, and it has the same
difficulty: where you put the boundaries changes the answer. They flagged exactly
this in their threads result ("evaluation plane has to be set closer to pin root").

### B2. Refill volume V_θ, V_z

**How much material flows back into the cavity behind the pin**, split by direction:

| | |
|---|---|
| **V_θ** | circumferential refill — material swept *around* the pin |
| **V_z** | vertical refill — material driven *downward*, e.g. by threads or tilt |

Units of m³ (a volume per revolution). **High = good backfilling = fewer voids.**

> **This is the quantity that corresponds to our η.** They measure whether material
> arrives; we measure the stress state where it did not. Theirs is the cause, ours
> the consequence — which is why the two rankings agree.

### B3. V_PD — plastic deformation zone volume

**The volume of material being actively deformed**, measured as the region where
equivalent strain rate exceeds a threshold (they use 250 s⁻¹).

Reported as the ratio **γ = V_PD / V_pin** — deformed volume per unit pin volume.

Their findings: **flat faces increase V_PD**, and **V_PD,A1 ≈ 1.5 × V_PD,A2**.

⚠️ **This independently confirms one of our numbers.** We measured A1 capturing
**2.4×** the mesh nodes of A2 in the same sampling window and suspected a sampling
artefact. Their result says A1's deforming volume is genuinely larger — so our
number was physics, not an artefact. (Same direction, same order; the two are not
the same quantity, so they should not match exactly.)

### B4. Forces: F_mag, F_x, F_y, force angle, oscillations

Reaction forces on the tool — **the only quantity in this whole project that is
experimentally validated** (against instrumented welds, in the published thesis).

| | meaning |
|---|---|
| **F_x** | traverse force — resistance to forward motion |
| **F_y** | transverse force — sideways, from the asymmetric flow |
| **F_mag** | resultant magnitude √(F_x²+F_y²+F_z²) — total energy input, bending stress |
| **force angle** | the F_x : F_y ratio — *which* direction the load acts |
| **oscillations** | variation *within* one revolution — high values mean tool fatigue / spindle vibration risk |

⚠️ **Their defect proxy: `F_y > F_x` signals material stagnation and defect risk.**

> **This is the cheapest external check available to us.** Forces are measured
> experimentally, so applying the F_y/F_x criterion to all cases and comparing
> against our damage ranking tests our result against real data — with no new
> experiment. Recorded as a next step.

### B5. Intra-revolution behaviour

The tool rotates, so everything is **periodic at ω**. "Intra-revolution" means
variation *within* one turn, as opposed to the cycle-averaged value.

Their criterion: **smooth, regular intra-revolution force waves = stable
defect-free flow**; irregular ones indicate trouble.

⚠️ Our equivalent is the **phase average** — we average η and ε̇ over all 166 steps
of a revolution. ⚠️ And this is exactly why O27 matters: a *single snapshot* differs
from the revolution average by up to 48 % on sparsely-sampled cases.

### B6. VP_high / VP_low — pressure regions in the deposition cavity

They extract high- and low-pressure volumes in the trailing half of the tool and
plot them against angle θ around the pin.

**Reading it:** high pressure behind the pin = **good consolidation** (material
being forged together). Low pressure = **void initiation likely**.

⚠️ **Direct overlap with our σ_m = −p.** Their "low pressure region" is our
"positive σ_m" is "tensile, voids open". Two names for the same physics — worth
saying explicitly in the meeting, because it is the clearest single point of
contact between the two analyses.

### B7. Flow depth

How far **down** the stirring action reaches, measured by thresholding the vertical
velocity. Threaded pins drive material deeper; a shallow flow depth leaves
unstirred material near the root.

Their finding: "stagnation zones near the bottom of AS indicate tunnel defects" —
i.e. insufficient flow depth on the advancing side is a defect mechanism.

### B8. Tool wear: Archard and Usui models

A **separate workstream** (`Dani_Wear_Model_Status_September.pdf`,
`tool_wear_*.pdf`) — not about voids at all, but you may be asked how it relates.

Two classical slip-driven wear laws:

$$\text{Archard:}\quad \dot w = k\,P\,V_{\text{slip}} \qquad\qquad \text{Usui:}\quad \dot w = k\,P\,g(T)\,V_{\text{slip}}$$

| | |
|---|---|
| **ẇ** | local wear rate — material removed from the **tool** |
| **P** | contact pressure |
| **V_slip** | relative sliding speed between tool and workpiece |
| **g(T)** | thermal activation (Usui adds temperature dependence; Archard does not) |

Their stated driver: *"pressure and thermal activation"*.

> **How it relates to our work, and the honest answer:** it shares the **inputs**
> (contact pressure, slip velocity, temperature from the same FOM) but predicts
> damage to the **tool**, where we predict damage to the **weld**. Both are
> integrals of a rate along a trajectory — structurally the same kind of
> calculation. ⚠️ But they are different physics with different validation, so do
> not claim they confirm each other.

### B9. The weld zones: SZ/WN, TMAZ, HAZ

Standard FSW microstructure classification, used descriptively throughout their work:

| zone | what happened to it |
|---|---|
| **SZ / WN** — stir zone / weld nugget | passed through the shear layer; fully recrystallised, fine equiaxed grains |
| **TMAZ** — thermo-mechanically affected | deformed by the shoulder's forging action but **not** recrystallised; grains aligned |
| **HAZ** — heat affected | thermal cycle only, no deformation |
| **BM** — base material | untouched |

⚠️ Our damage analysis samples the **SZ and the inner TMAZ** (lower 70 % of
thickness, pin-anchored radially). We do not resolve HAZ, and we do not predict
grain structure — only the stress conditions that open or close voids.

### B10. ROM machinery: POD, GPR, multi-fidelity, reference-space projection

From `PinShapePaper_ROMDevelopment.pdf` — the plan for the fast surrogate.

| term | meaning |
|---|---|
| **POD** | Proper Orthogonal Decomposition — find the few dominant patterns across many FOM results (essentially PCA on simulations) |
| **GPR** | Gaussian Process Regression — map parameters to POD coefficients, **with an uncertainty estimate** |
| **multi-fidelity** | train mostly on cheap/coarse data (80–95 %), correct with a few expensive accurate runs |
| **adaptive sampling** | let the uncertainty estimate choose where to run the next expensive case |
| **reference-space projection** | when meshes differ between cases, project everything onto one common reference mesh (their preferred option, for accuracy) |

⚠️ **Why this matters to us:** our damage metric is far more sensitive to
**pressure** than to velocity (card [#11](#11-the-norton-law)). So when the ROM is
built, **pressure accuracy is the binding constraint** — a ROM that nails velocity
and approximates pressure will produce confident wrong answers.

---

## Where to go next

| you want | go to |
|---|---|
| the full derivation of any of this | `PRIMER_fsw_physics_for_me.md` |
| what we did and why, in order | primer **§9** |
| questions the team will ask, with answers | primer **§9c** |
| a one-slide summary | primer **§9d** |
| how our results relate to the colleague's | `COMPARISON_with_flow_analysis.md` |
| every symbol with units | primer **§10b** |
