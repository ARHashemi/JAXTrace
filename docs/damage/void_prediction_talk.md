---
marp: true
theme: default
paginate: true
size: 16:9
math: katex
style: |
  section {
    font-size: 25px;
    padding: 50px 60px;
  }
  section.lead {
    text-align: center;
  }
  h1 {
    color: #1a3a5c;
    font-size: 44px;
  }
  h2 {
    color: #1a3a5c;
    font-size: 34px;
    border-bottom: 2px solid #d0d7de;
    padding-bottom: 6px;
  }
  .small { font-size: 20px; }
  .tiny { font-size: 17px; color: #555; }
  .big { font-size: 40px; font-weight: bold; color: #1a3a5c; }
  .huge { font-size: 60px; font-weight: bold; color: #1a3a5c; }
  .box {
    background: #f6f8fa;
    border-left: 4px solid #1a3a5c;
    padding: 12px 18px;
    margin: 10px 0;
  }
  table { font-size: 21px; }
  section::after {
    font-size: 15px;
    color: #888;
  }
---

<!-- _class: lead -->

# Defining Defects for the FSW Digital Twin

## Can a single-phase simulation tell us where the voids are?

**Internal meeting — 23 September 2026**

<span class="tiny">Where the fast simulation engine stands, and the open question on defect definition</span>

---

## Where this sits in the project

**Objective:** a fast simulation engine for the FSW digital twin, in two stages —

<div class="box">

**Stage 1 — classify the operational parameter space.**
Identify the "safe" zone(s) in (v_adv, ω_pin).

</div>

<div class="box">

**Stage 2 — predict the output.**
Given a point in that space, say what defects result.

</div>

**Two barriers stood in the way:**

| | Barrier | Why it blocks |
|---|---|---|
| **1** | Training data volume | Strong process nonlinearities; correlating output to input needs many samples |
| **2** | A clear definition of "defect" | Without a target variable there is nothing to predict |

---

## Status of the two barriers

<div class="box">

**Barrier 1 — data generation: addressed.**

ROM + mesh-to-grid cubic projection + JAXTrace on-grid tracking now
form a fast generator. Our own POD basis reduced velocity reconstruction
error from 4.04 % to 0.52 % (K=3), and ROM particle tracking from
6.79 mm to 2.34 mm. Sampling the parameter space is no longer the
bottleneck it was.

</div>

<div class="box">

**Barrier 2 — defect definition: open.**

Particle density from the tracking output is one candidate target.
A second question has been raised alongside it:

<strong>can a single-phase simulation identify air holes / internal voids?</strong>

</div>

---

## The question

<div class="box">

Can the location of internal voids and wormholes be estimated from a
<strong>simulation of the metal flow alone</strong>, without explicitly modelling
a second phase?

</div>

1. Is it possible at all, and what is already being done?
2. Is the multiphase/turbulence route a usable counterpart here?
3. What are the options if we want to work on this?

<span class="tiny">Basis: 29 papers screened, 2003–2025 — CFD/CEL modelling of FSW defects,
FSW flow-regime characterisation, and void closure in bulk metal forming.
Collected in Zotero (<em>FSW › FSW Void & Wormhole Prediction</em>).</span>

---

## Summary of findings

<div class="box">

**1. Yes — and it is the standard approach.**
Across the surveyed modelling work, void prediction is done by solving a
single (metal) phase and applying a criterion afterwards. Explicit gas-phase
modelling does not appear.

</div>

<div class="box">

**2. The turbulence/cavitation counterpart does not carry over.**
The flow regime differs by 7–12 orders of magnitude in Reynolds number, and
the pressure state has the opposite sign to what cavitation requires.

</div>

<div class="box">

**3. The transferable body of theory is void closure in metal forming**
— stress-triaxiality-based, not cavitation-based.

</div>

---

<!-- _class: lead -->

# Part 1
## Is there a turbulence counterpart?

---

## What the multiphase literature requires

Two mechanisms generate bubbles in liquid flows:

<div class="box">

**Cavitation** — local static pressure falls <em>below vapour pressure</em>,
typically inside a vortex core. The liquid boils locally.

</div>

<div class="box">

**Air entrainment** — near-surface turbulence deforms a free surface faster
than gravity and surface tension restore it; air is dragged under.

</div>

Both require:

| | Condition | Expressed as |
|---|---|---|
| **1** | Inertia-dominated flow | Re ≫ 1 |
| **2** | A pressure sink or gas reservoir | $p < p_v$, or a free surface |
| **3** | Weak restoring force | surface tension, gravity |

---

## Condition 1 — flow regime

Effective viscosity of plasticised aluminium has been measured experimentally:

- **Franke et al. (2017)** — capillary rheometer through a drilled backing plate:
  $\mu_{\text{eff}} = 10^5 - 5\times10^6$ Pa·s for AA6061-T6
- **Reisgen et al. (2020)** — inverse method from forge force and spindle torque
  (AA5083): same order of magnitude

With $\rho = 2700$ kg/m³, $R = 6-9$ mm, $\omega = 1000-1500$ rpm:

$$\mathrm{Re} = \frac{\rho U L}{\mu_{\text{eff}}}, \qquad U = \omega R \approx 0.6-1.4 \text{ m/s}$$

| $\mu_{\text{eff}}$ (Pa·s) | Re |
|---|---|
| $1\times10^5$ | $2\times10^{-4}$ – $7\times10^{-4}$ |
| $5\times10^6$ | $4\times10^{-6}$ – $1\times10^{-5}$ |
| $1\times10^8$ | $2\times10^{-7}$ – $7\times10^{-7}$ |

---

<!-- _class: lead -->

<span class="huge">Re ≈ 10⁻⁷ – 10⁻⁴</span>

<br>

Turbulent transition occurs around **Re ~ 10³ – 10⁵**

<br>

<div class="box">

The regime is <strong>Stokes creeping flow</strong>: inertia negligible,
velocity field slaved to the boundary conditions, no inertial vortex cores.

</div>

<span class="tiny">Consistent with how FSW flow is formulated in the modelling literature —
Seidel & Reynolds (2003) treat it as "<em>laminar, viscous flow of a non-Newtonian
fluid past a rotating circular cylinder</em>". Kadian & Biswas (2015) compare
laminar and turbulent closures directly.</span>

---

## Condition 2 — pressure state

The cavitation criterion asks whether $p$ can fall below $p_v$.

<div style="display: flex; gap: 30px; align-items: flex-start;">
<div style="flex: 1;">

**Vapour pressure of Al at ~750 K**

<span class="big">≈ 10⁻⁹ Pa</span>

Effectively a vacuum.

</div>
<div style="flex: 1;">

**Hydrostatic pressure in the stir zone**

<span class="big">+50 to +200 MPa</span>

Compressive.

</div>
</div>

<div class="box">

The cavitation number $\sigma = \dfrac{p - p_v}{\frac{1}{2}\rho U^2}$ differs from
the inception range by roughly <strong>eleven orders of magnitude</strong>.

The sign is also opposite: compressive hydrostatic stress <em>closes</em> porosity.
This is part of why the process consolidates material.

</div>

---

## Condition 3 — restoring force and gas supply

<div class="box">

Air entrainment requires an <strong>air–liquid interface</strong>. Within the stir zone
there is none; the contents of a void are vacuum rather than entrained gas.

</div>

The associated dimensionless groups therefore have no quantity to act on:

| Number | Compares | Status here |
|---|---|---|
| Fr | inertia vs gravity | no interface; gravity negligible at 100 MPa |
| We | inertia vs surface tension | surface tension immaterial at $10^6$ Pa·s |
| Bo | gravity vs surface tension | neither is the restoring force |

<div class="box">

In FSW the restoring force is the <strong>deviatoric yield stress of the solid</strong>,
which requires a different closure entirely.

</div>

---

## Partial exception

<div class="box">

At the <strong>top surface under the shoulder</strong> a genuine metal/air interface
exists. Surface lack-of-fill and flash do involve a free surface, and CEL+VOF
models track it (e.g. Das et al. 2021).

</div>

However:

- The restoring force remains **yield stress**, not surface tension
- That defect class is **surface**, distinct from internal wormholes

<div class="box">

Relevant to note, but not a basis for modelling internal voids.

</div>

---

## Assessment of the multiphase route

| Element | Status in FSW conditions |
|---|---|
| Single phase solved + criterion applied | Standard practice |
| Vorticity as an indicator | Present everywhere, including sound welds |
| Turbulence closure | Re ≈ 10⁻⁶; flow is laminar |
| Cavitation number | Differs from inception by ~10¹¹ |
| Froude / Weber / Bond | No interface in the bulk |
| Vortex circulation-flux models | Restoring force is yield stress, not σ |
| VOF / level-set | Used — tracks metal/void boundary, not gas |

<div class="box">

The <strong>methodological structure</strong> — one phase plus a post-hoc criterion —
matches what is done. The <strong>physical closure</strong> has to come from elsewhere.

</div>

---

<!-- _class: lead -->

# Part 2
## The mechanism reported in the literature

---

## Volumetric filling failure

<div class="box">

A transient cavity opens behind the advancing pin <strong>each revolution</strong>
and is refilled by material swept round from the retreating side.
Where delivery is insufficient, the deficit remains.

</div>

<svg viewBox="0 0 900 260" width="100%" style="max-height:260px">
  <rect x="40" y="40" width="820" height="180" fill="#f2f4f7" stroke="#aab" stroke-width="2"/>
  <line x1="40" y1="130" x2="860" y2="130" stroke="#ccc" stroke-dasharray="6 6"/>
  <text x="52" y="66" font-size="17" fill="#555">ADVANCING SIDE</text>
  <text x="52" y="208" font-size="17" fill="#555">RETREATING SIDE</text>
  <line x1="300" y1="26" x2="420" y2="26" stroke="#1a3a5c" stroke-width="2" marker-end="url(#ar)"/>
  <text x="430" y="31" font-size="16" fill="#1a3a5c">tool travel</text>
  <defs>
    <marker id="ar" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#1a3a5c"/>
    </marker>
    <marker id="arg" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#44618a"/>
    </marker>
    <marker id="arr" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#8a6d3b"/>
    </marker>
  </defs>
  <circle cx="420" cy="130" r="52" fill="#d8dee9" stroke="#1a3a5c" stroke-width="2.5"/>
  <circle cx="420" cy="130" r="16" fill="#1a3a5c"/>
  <text x="404" y="196" font-size="15" fill="#1a3a5c">pin</text>
  <path d="M 420 62 A 68 68 0 0 1 488 130" fill="none" stroke="#1a3a5c" stroke-width="2" marker-end="url(#ar)"/>
  <text x="496" y="106" font-size="15" fill="#1a3a5c">ω</text>
  <path d="M 300 185 Q 380 200 450 178 Q 510 158 528 132"
        fill="none" stroke="#44618a" stroke-width="3" marker-end="url(#arg)"/>
  <text x="326" y="222" font-size="16" fill="#44618a">material swept from RS</text>
  <ellipse cx="520" cy="96" rx="34" ry="20" fill="#fdfaf3" stroke="#8a6d3b" stroke-width="2.5" stroke-dasharray="5 4"/>
  <text x="560" y="86" font-size="16" fill="#8a6d3b">transient cavity</text>
  <text x="560" y="106" font-size="16" fill="#8a6d3b">(refilled each revolution)</text>
  <path d="M 556 96 L 700 96" stroke="#8a6d3b" stroke-width="3" stroke-dasharray="3 5" marker-end="url(#arr)"/>
  <text x="614" y="130" font-size="16" fill="#8a6d3b">unfilled → tunnel</text>
</svg>

Successive revolutions superimpose the deficit along the weld line,
producing a continuous wormhole or tunnel defect.

---

## Reported asymmetry

<div style="display: flex; gap: 25px;">
<div style="flex: 1;">

**Retreating side**

Tool surface velocity and traverse velocity **oppose**.

Flow turns gently; shear layer thicker.

</div>
<div style="flex: 1;">

**Advancing side**

Tool surface velocity and traverse velocity **add**.

Flow turns sharply; shear layer thinnest.

</div>
</div>

<div class="box">

Stagnation and under-filling are consistently reported on the advancing side
across the surveyed experimental and numerical work.

</div>

The asymmetry is kinematic, arising from the velocity composition
rather than from a pressure or vorticity field.

---

## Supporting observations

<div class="box">

**Morisada et al. (2015)** — two synchronised X-ray real-time imaging systems,
3D flow visualisation. Defect formation correlated with flow tilt and
stagnation on the AS; measured AS flow velocity decreased under
defect-forming conditions.

</div>

<div class="box">

**Zeng et al. (2018)** — oxide marker with stop-action welding. Reports
"<em>instantaneous void occurrence and insufficient inflow material</em>"
as the cause of preferential void formation in the top stir zone on the AS.

</div>

<div class="box">

**Ghate et al. (2020)** — per-revolution cavity filling model validated against CT:
"<em>the incoming material fills the cavity per revolution… in some cases the
cavity is partially filled, which leads to the discontinuities</em>"

</div>

---

## The transferable theory

Void evolution in **bulk metal forming**, rather than cavitation:

<div class="box">

Voids <strong>close</strong> under negative (compressive) stress triaxiality
and <strong>grow</strong> under positive triaxiality.

</div>

- **Saby, Bouchard & Bernacki (2015)** — review of void closure criteria in hot forming
- **Zapara et al. (2013)** — under compression the governing mechanism is
  void *shape change*, distinct from volumetric growth under tension
- Lode-parameter-dependent evolution models (Chen 2021, Wang 2020, Chbihi 2016)
- **Nielsen (2009)** — shear-modified Gurson model applied to an FSW specimen

<div class="box">

This provides the closure that cavitation theory supplies in the
multiphase case: a <strong>triaxiality- and Lode-dependent void evolution law</strong>.

</div>

---

<!-- _class: lead -->

# Part 3
## What is being done

---

## Five families of approach

| Family | Basis | Representative |
|---|---|---|
| **Mass balance** | flow in vs cavity volume | Arbegast 2008, Qian 2013 |
| **Pressure criterion** | threshold on contact pressure | Shi 2022 |
| **Tracer / particle** | marker advection and depletion | Dialami 2020 |
| **Streamline damage** | void-growth ODE along paths | He, Dawson & Boyce 2008 |
| **CEL + VOF** | explicit metal/void boundary | Chauhan 2018, Choudhary 2022 |

<div class="box">

All five solve a single phase and apply a criterion.
No turbulence closure appears in any of them.

</div>

---

## Mass balance

<div class="box">

**Arbegast (2008)** — <span class="tiny">Scripta Materialia, 308 citations</span>

Partitions flow into deformation zones around the pin and under the shoulder.
Defines an "excess material function" and a "forcing function" that partition
flow to fill the cavity behind the tool.

"<em>Inadequacies in this flow are related to specific FSW defect types.</em>"

</div>

<div class="box">

**Qian et al. (2013)** — <span class="tiny">Scripta Materialia</span>

Analytical model based on "<em>exactly balancing the material flowing from the
region ahead of the pin to the rear</em>". Produces optimum ω–v operating
windows consistent with experiment.

</div>

Both are volume-bookkeeping arguments, independent of CFD machinery.

---

## Pressure criterion

<div class="box">

**Shi et al. (2022)** — <span class="tiny">International Journal of Mechanical Sciences</span>

Thermal–fluid–structure coupled model with non-uniform tool/workpiece
contact pressure.

</div>

Reported criterion based on the difference between maximum and
minimum contact pressure:

<div class="box">

<span class="big">Δp &lt; 15 MPa → sound weld</span>
<span class="big">Δp &gt; 15 MPa → void</span>

</div>

<div class="box">

Calibrated for a single alloy/tool combination. The functional form may
transfer; the numerical value would require recalibration.

</div>

---

## Tracer / particle advection

<div class="box">

**Dialami, Cervera & Chiumenti (2020)** — <span class="tiny">Eur. J. Mech. A/Solids, 124 citations</span>

Two-stage strategy:

1. Speed-up stage on a fixed mesh to reach steady state
2. Periodic stage with tool rotation, advecting material tracers
   on the nodal velocity field

Reports void, wormhole, flash, joint line remnant and onion rings
from a single simulation.

</div>

<div class="box">

Structurally this is a particle-tracking post-process applied to a
precomputed velocity field — the same operation our existing tracking
code performs.

</div>

---

## Streamline-integrated damage

<div class="box">

**He, Dawson & Boyce (2008)** — <span class="tiny">J. Eng. Mater. Technol.</span>

Steady-state Eulerian FE solution (3D, 304L steel), followed by integration
of the Lee–Dawson void-growth model along the streamlines:

$$\dot{\Phi} = \psi_0\,\bar{D}\,\Phi\,\frac{\exp\!\left(c_1\sigma_m/\kappa\right)}{1-\Phi}$$

Reported <em>qualitative</em> agreement with observed void locations; highest
porosity on the advancing, trailing side of the pin.

</div>

Of the surveyed methods this is the most direct realisation of
"single phase solved, criterion applied afterwards".

---

## The void-growth equation, term by term

$$\dot{\Phi} = \psi_0\;\bar{D}\;\Phi\;\exp\!\left(c_1\sigma_m/\kappa\right)\;\frac{1}{1-\Phi}$$

| Term | What it is | Why it is there |
|---|---|---|
| $\Phi$ | current void volume fraction | growth $\propto$ existing void size → **exponential in time**; $\Phi=0$ is a fixed point, so this is a *growth* law, not nucleation |
| $\bar{D}$ | effective deformation rate | damage accrues **per unit strain**, not per unit time — sets the clock |
| $\exp(c_1\sigma_m/\kappa)$ | driving force | mean stress in units of material **strength**; exponential form from the rigid-plastic solution around a void |
| $1/(1-\Phi)$ | matrix correction | less solid carries the strain; $\approx 1$ at the $\Phi\sim10^{-4}$ of interest |
| $\psi_0$ | Arrhenius prefactor | sets the absolute rate scale; constant in their formulation |

<div class="box">

$\kappa$ (strength) is a <strong>second state variable</strong> with its own Voce-type
saturation ODE, integrated along the same streamline — the two equations are coupled.

</div>

---

## What that equation does and does not do

<div class="box">

**Requires a seed.** $\Phi=0$ is a fixed point, so an initial porosity must be
assumed — He et al. use 0.01%, "uniformly distributed". The model describes
void <em>growth</em>, not nucleation.

</div>

<div class="box">

**Does not predict closure.** Their §2: <em>"when the mean stress is hydrostatic
compression ($\sigma_m \le 0$), the void growth rate is assumed to be zero,
but never negative."</em> Porosity freezes under the shoulder; it does not heal.

</div>

<div class="box">

**Small signal.** Porosity goes from 0.01 % to ~0.017 % (smooth pin) and
~0.105 % (threaded). Agreement with experiment described as qualitative.

</div>

---

## CEL + VOF

The most widely used family in recent work.

<div class="box">

VOF here tracks the <strong>metal/void boundary</strong>. Material failing to fill
an Eulerian cell registers as a defect. The second phase is void, not gas.

</div>

- **Chauhan et al. (2018)** — validated against X-ray CT
- **Choudhary & Jain (2022)** — tunnel, void, cavity and root defects;
  tilt angle dominant (0° tunnel; 1° void+cavity; 2° defect-free; 3° defective)
- **Das et al. (2021)**, **Salloomi (2022)**, **Hosseini (2023)**, **Draper et al. (2025)**

<div class="box">

Draper et al. (2025) conclude that <strong>volume ratio</strong> is the parameter
warranting further study — again a mass-balance framing.

</div>

---

## Data-driven work

<div class="box">

**Du et al. (2019)** — <span class="tiny">npj Computational Materials, 108 experimental datasets, 3 Al alloys</span>

</div>

| Input features | Reported accuracy |
|---|---|
| Raw welding parameters | 83.3% |
| From an **analytical** FSW model | 90 – 93.3% |
| From a **rigorous numerical** model | **96.6%** |

<div class="box">

Physics-derived features from a single-phase simulation outperform raw
process parameters by ~13 percentage points. Temperature and maximum
shear stress reported as the dominant causative variables.

</div>

<span class="tiny">Force-signal monitoring (Guan 2022, Rabe 2021/2024, Ansari 2022) addresses
in-process detection rather than a priori prediction — complementary, different problem.</span>

---

## How much was built on this?

Citation record of the streamline-damage line:

| Paper | Citations | per year |
|---|---|---|
| Dialami 2020 (tracers) | 124 | 19 |
| Nielsen 2009 (Gurson, FSW specimen) | 113 | 6.4 |
| Chauhan 2018 (CEL+VOF) | 115 | 14 |
| **He, Dawson & Boyce 2008** | **11** | **0.6** |

<div class="box">

No follow-up paper extends the Lee–Dawson-along-streamlines method.
Most of the 11 are passing mentions in reviews; one is the same group
applied to a different problem.

</div>

Two readings, both plausible: the approach was sound but overtaken by CEL+VOF,
**or** it was tried and the signal proved too weak to publish. The reported
porosity change (0.01 % → 0.017 %) is consistent with either.

---

## Reported limitations

<div class="box">

**Void size is overestimated** by CEL/VOF (Zhu et al. 2017, and noted generally).
Presence is captured; magnitude is not.

</div>

<div class="box">

**Thresholds are calibrated rather than universal.** Transferability across
alloys and tool geometries is not demonstrated in the surveyed work.

</div>

<div class="box">

**Outputs are largely binary** — defect / no defect, or a region in a
cross-section. Continuous, spatially resolved susceptibility fields
are rare in the surveyed literature.

</div>

---

<!-- _class: lead -->

# Part 4
## Options

---

## Low-cost checks

Each is post-processing on simulations that already exist:

<div class="box">

**A. Reynolds number for our own parameter range**
Establishes the flow regime quantitatively for our configuration.

</div>

<div class="box">

**B. Contact-pressure differential (Shi criterion)**
Evaluate Δp on existing runs; test whether the 15 MPa threshold
separates any cases in our data.

</div>

<div class="box">

**C. Mass balance (Qian model)**
Evaluate for our (ω, v) cases and compare against observed defects.

</div>

---

## Candidate directions

<div class="box">

**Option 1 — tracer depletion**
Advect dense tracer clouds on a single-phase viscoplastic field and
identify depletion behind the tool. Reproduces an established method;
low risk; reuses existing tracking code.

</div>

<div class="box">

**Option 2 — streamline-integrated damage**
Integrate a triaxiality- and Lode-dependent void evolution law along
large numbers of GPU-tracked pathlines. Produces a continuous
susceptibility field rather than a binary flag.

</div>

<div class="box">

**Option 3 — surrogate over the parameter space**
ROM over (v_adv, ω_pin) mapping to a susceptibility field, giving a
spatially resolved process window.

</div>

<span class="tiny">The options are cumulative — 1 derisks 2, and 2 supplies the field that 3 reduces.</span>

---

## Where the existing capability fits

<div class="box">

The streamline-damage approach (He 2008) was demonstrated on a limited
set of streamlines. Throughput was the constraint at the time.

</div>

<div class="box">

Current capability — GPU particle tracking on time-dependent meshes,
plus an existing ROM pipeline over (v_adv, ω_pin) — addresses that
constraint directly.

</div>

<div class="box">

The gap identified in Part 3 — continuous, spatially resolved
susceptibility rather than binary classification — is what
high-throughput pathline integration would produce.

</div>

---

## Existing work closest to this

<div class="box">

**Fraser et al. (2018)**, *Metals* — GPU-parallelised meshfree SPH (SPHriction-3D).
Proposes a defect metric evaluating presence and severity in the weld zone,
validates on AA6061-T6, then optimises advancing speed and rpm by minimising
defect volume.

</div>

<div class="box">

**Cao et al. (2021)**, *J. Comput. Phys.* — machine learning and reduced-order
computation of an FSW model. Same group.

</div>

Both are close to the Option 2–3 ambition and predate it. What appears to remain open:

| | Status |
|---|---|
| GPU throughput + defect metric + optimisation | **done** (Fraser 2018) |
| ROM surrogate of an FSW model | **done** (Cao 2021) |
| Defect metric from an integrated **constitutive damage law** rather than particle deficiency | open |
| ROM whose **target** is a pathline-integrated damage field | open |
| Triaxiality/Lode-dependent **closure** branch in FSW | not attempted |

<span class="tiny">Both papers to be read in full before positioning any of this as novel.</span>

---

## Results — 42 cases surveyed

<div class="box">

Stage 1 complete: <strong>22 cylindrical ROM cases</strong> plus
<strong>20 featured-tool cases</strong> (flats, flutes, threads, tilt), the latter
resolved across a full tool revolution.

</div>

| finding | evidence |
|---|---|
| Advancing-wake tension is recovered from the FOM fields, no fitting | 42 cases |
| In the cylindrical cohort it is governed by ω alone | Spearman ρ = **−0.98**, crossover ≈ 575 rpm |
| Tool geometry separates the families | threads 1.20 · flutes 1.08 · tilt 0.82 · flats 0.75 |
| 8 rotational phases reproduce a 166-step revolution | within **0.021** (median 0.004) |

<div class="box">

⚠️ <strong>No ground truth yet.</strong> Every number is an internally consistent
prediction. Nothing has been compared against CT or macrographs.

</div>

---

## Phase matters — but only for some tools

<div class="box">

A featured or tilted tool is <strong>periodic at ω</strong>, so a single snapshot is
one arbitrary rotational phase. Measured spread across a revolution:

</div>

| case | sd | span |
|---|---|---|
| D1 (tilt) | 0.001 | phase-independent |
| C2, C4 (threads) | 0.011 – 0.017 | very stable |
| **A2 (flats)** | **0.451** | 0.40 → 1.69 — **flips flank mid-revolution** |
| **D2 (tilt)** | 0.320 | 0.50 → 1.31 |

<div class="box">

For A2 and D2 a single snapshot is a coin toss; for threads it barely matters.
<strong>The 8-phase probe identifies which is which at 1/20th the cost of the
full revolution.</strong>

</div>

---

## Back to Barrier 2

Candidate defect targets for the digital twin, as they now stand:

| Target | Source | Status |
|---|---|---|
| **Particle density / depletion** | existing tracking output | available now |
| **Tracer depletion behind tool** | Option 1 | reuses same output |
| **Void susceptibility (triaxiality)** | Option 2 | ✅ **implemented, 42 cases run** |
| **Contact-pressure Δp** | Shi criterion | testable on existing runs |

<div class="box">

All four are scalar or field quantities derivable from a single-phase
run — so all are compatible with the ROM/parameter-space work in Stage 1.

</div>

---

## Open questions

<div class="box">

**Validation data** — are there CT or macrograph results for welds whose
(ω, v) we also have simulations for? Without this, any susceptibility
field is unvalidated.

</div>

<div class="box">

**Target choice** — does density alone suffice for Barrier 2, or is a
dedicated void criterion worth the additional work?

</div>

<div class="box">

**Constitutive choice** — which void evolution law: Gurson-type,
Lode-dependent, or a simpler form tested first for case separation?

</div>

<div class="box">

**Sequencing** — is Option 1 worth completing before committing to Option 2?

</div>

---

## Terminology

<div class="box">

The surveyed FSW literature consistently uses
<strong>void</strong>, <strong>cavity</strong>, <strong>wormhole</strong>, <strong>tunnel defect</strong>.

</div>

<div class="box">

<strong>Bubble</strong>, <strong>cavitation</strong> and <strong>turbulence</strong> carry specific
multiphase meanings that do not apply to this regime.

</div>

The former set matches the physics described above.

---

<!-- _class: lead -->

# Discussion

<span class="small">

Literature review: `1phase-2phase/FSW_void_prediction_literature_review.md`
Zotero: **FSW › FSW Void & Wormhole Prediction** — 29 papers, tagged

</span>

<span class="tiny">
Tags: <code>KEY-PAPER</code> · <code>jaxtrace-relevant</code> · <code>correct-analogue</code> · <code>Reynolds-number-argument</code>
</span>

---

<!-- _class: lead -->

# Backup

---

## Reynolds number, in full

$$\mathrm{Re} = \frac{\rho U L}{\mu_{\text{eff}}}, \qquad U = \omega R, \qquad L = 2R$$

| Quantity | Value | Source |
|---|---|---|
| $\rho$ | 2700 kg/m³ | Al |
| $R$ | 6 – 9 mm | typical shoulder/pin |
| $\omega$ | 1000 – 1500 rpm | representative range |
| $U = \omega R$ | 0.63 – 1.41 m/s | |
| $\mu_{\text{eff}}$ | $10^5 - 5\times10^6$ Pa·s | measured, Franke et al. 2017 |

$$\mathrm{Re} = \frac{2700 \times 1.41 \times 0.018}{5\times10^6} \approx 1.4\times10^{-5}$$

<div class="box">

Taking the lowest reported viscosity ($10^5$ Pa·s) and the highest tool speed,
Re reaches only $6.9\times10^{-4}$.

</div>

---

## Source figures, by DOI

Not reproduced here; available in the collection:

| Content | Paper | DOI |
|---|---|---|
| X-ray flow visualisation, AS stagnation | Morisada 2015 | `10.1179/1362171814Y.0000000266` |
| Per-revolution cavity filling, CT | Ghate 2020 | `10.1016/j.ijmecsci.2019.105293` |
| Flow-partitioned deformation zones | Arbegast 2008 | `10.1016/j.scriptamat.2007.10.031` |
| Tracer-predicted defects | Dialami 2020 | `10.1016/j.euromechsol.2019.103912` |
| Contact-pressure Δp criterion | Shi 2022 | `10.1016/j.ijmecsci.2022.107969` |
| Streamline-integrated damage field | He 2008 | `10.1115/1.2840963` |
| Void closure vs triaxiality | Saby 2015 | `10.1016/j.jmapro.2014.05.006` |

---

## Defect taxonomy

The surveyed work distinguishes several classes:

| Defect | Mechanism | Same physics? |
|---|---|---|
| **Void / wormhole / tunnel** | volumetric under-fill, AS | subject of this survey |
| **Root flaw / kissing bond** | insufficient penetration | partly |
| **Joint line remnant** | oxide transport | no — tracer transport |
| **Flash** | excess material expelled | no — surface |
| **Surface lack-of-fill** | shoulder free surface | no — surface |

<div class="box">

Dialami et al. (2020) obtain several of these from one simulation by
tracing different particle populations — the same machinery that
predicts voids also predicts joint line remnant.

</div>

---

## Why not CEL+VOF directly

A reasonable alternative; the trade-offs:

- **Cost per parameter set.** Full CEL with tool rotation is expensive,
  which constrains any process-window or surrogate ambition
- **Void size overestimated** (Zhu et al. 2017) — presence yes, magnitude no
- **Binary output** — a region in a cross-section rather than a field
- **Limited reuse** of existing tracking and ROM infrastructure

<div class="box">

The streamline-damage route reuses current capability and targets the
gap identified in Part 3, rather than re-entering an area already
well covered by the CEL+VOF literature.

</div>
