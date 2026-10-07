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
  /* Callout boxes carry the dense explanatory text, so they run a size below
     body copy -- this is what keeps the heavier slides inside the frame. */
  .box, .warn, .ok {
    font-size: 21px;
    line-height: 1.38;
    padding: 10px 16px;
    margin: 9px 0;
  }
  .box  { background: #f6f8fa; border-left: 4px solid #1a3a5c; }
  .warn { background: #fff6f6; border-left: 4px solid #c0392b; }
  .ok   { background: #f2faf5; border-left: 4px solid #1baf7a; }
  .box table, .warn table, .ok table { font-size: 19px; }
  table { font-size: 21px; }
  section::after {
    font-size: 15px;
    color: #888;
  }
---

<!-- _class: lead -->

# Predicting Defect Susceptibility in FSW

## Damage laws integrated along particle pathlines — 45 cases

<span class="tiny">Damage accumulated along particle trajectories through the FOM velocity and pressure fields</span>

---

## What this covers

1. **The model** — which damage indicators, and why those
2. **The tools** — the four pin families, measured from their STLs
3. **The implementation** — how each quantity is computed, how the ODEs are solved
4. **Example cases** — what the particles carry
5. **All 45 cases** — and two results that do **not** agree with each other

<div class="box">

**One sentence:** we integrate established metal-forming damage laws along particle
pathlines through the FOM fields, at under 2 % throughput cost, on all 45 cases.

</div>

<span class="tiny">⚠️ <b>Terminology.</b> <b>Damage law / model</b> — the two constitutive ODEs (the literature's own term). <b>Damage indicator</b> — the value each particle accumulates. <b>Defect</b> — the physical void in the weld. <b>Defect susceptibility</b> — what the ranking predicts. The indicators are <b>not</b> a predicted porosity.</span>

---

# 1 — The model

---

## Process geometry and the terms used here

![w:900](figs_concept/figC1_geometry.png)

<span class="tiny"><b>Advancing</b> side: rotation and travel add. <b>Retreating</b>: they oppose. The <b>shoulder</b> (r = 7 mm) contacts the top surface; the <b>pin</b> (r ≈ 2.4 mm) stirs the full 6 mm depth. The <b>shear layer</b>, δ ≈ 0.5–2.8 mm around the pin, is where the strain rates that drive damage are concentrated — and is where the trajectories in §4 show material being captured.</span>

---

## Why a void forms here is not a bubble problem

FSW is **solid-state**: peak temperature is 80–95 % of melting — *"it is not melted"*
(Venghaus thesis §4, p. 66). No liquid, no vapour.

A void is **empty space the material failed to refill** behind the advancing pin —
a volumetric bookkeeping failure, not nucleation.

| the analogy | the number | verdict |
|---|---|---|
| turbulence | $\mathrm{Re} \approx 10^{-7}\!-\!10^{-4}$ | **7–12 orders** below transition |
| cavitation | $\sigma_{\text{cav}} \approx 7\times10^4$ (needs $\lesssim 1$) | wrong by ~11 orders, **and wrong sign** |
| vorticity | rigid rotation gives $\mathbf{D}=0$ | large even in sound welds |

<span class="tiny">Each number is computed from <b>our own</b> fields, not quoted: Re from the measured velocity scale, pin diameter and the Norton–Hoff effective viscosity; the cavitation number from the measured pressure against the vapour pressure of aluminium at weld temperature. The tool spins everything, so vorticity cannot discriminate — damage needs the <b>stretching</b> part of ∇u (the symmetric part $\mathbf{D}$), not the spin part.</span>

---

## The variable: stress triaxiality

$$\eta \;=\; \frac{\sigma_m}{\sigma_{\text{eq}}}, \qquad
\sigma_m = \tfrac13\sigma_{kk} = -p, \qquad
\sigma_{\text{eq}} = \sqrt{\tfrac32 s_{ij}s_{ij}}$$

| $\eta$ | meaning | voids |
|---|---|---|
| **negative** | compression dominates | **close** |
| **positive** | tension dominates | **open** |

<div class="box">

Stress splits into a **volume** part ($\sigma_m$) and a **shape** part ($s_{ij}$).
Metal **flows** because of the shape part — but voids **open or close** because of
the volume part. $\eta$ is the ratio between them.

</div>

<span class="tiny">σ_m = −p verified empirically, not assumed: the mean stress is <b>compressive ahead</b> of the tool (forging) and <b>tensile in the wake</b> — median −6 to −7.5 MPa ahead, +0.4 to +2.2 MPa behind, and the sign pattern holds in <b>9 of 10</b> cases checked.</span>

---

## The two driving quantities, defined

Both damage laws are driven by the same pair, computed at every mesh node:

**Effective (von Mises equivalent) strain rate**

$$\dot\varepsilon_{\text{eff}}=\sqrt{\tfrac23\,\mathbf{D}':\mathbf{D}'},
\qquad \mathbf{D}=\tfrac12\!\left(\nabla\mathbf{u}+\nabla\mathbf{u}^{\!\top}\right),
\qquad \mathbf{D}'=\mathbf{D}-\tfrac13(\operatorname{tr}\mathbf{D})\mathbf{I}$$

<span class="tiny">The 2/3 normalisation is chosen so a uniaxial test returns exactly $\dot\varepsilon$. $\mathbf{D}$ is the <b>stretching</b> tensor — the symmetric part of the velocity gradient, which carries no rigid rotation.</span>

**Maximum principal stress**

$$\sigma_1=\lambda_{\max}(\boldsymbol\sigma),\qquad
\boldsymbol\sigma=\mathbf{s}+\sigma_m\mathbf{I}
\;\;\Longrightarrow\;\;
\frac{\sigma_1}{\sigma_{\text{eq}}}=\frac{\sigma_m}{\sigma_{\text{eq}}}+\hat s_1$$

<span class="tiny">$\hat s_1$ is the largest eigenvalue of the deviator normalised to unit von Mises norm, obtained from $\mathbf{D}'$ through the generalised-Newtonian closure $\mathbf{s}=2\mu\mathbf{D}'$ — so $\mu$ never has to be formed explicitly. Implemented in <code>jaxtrace/damage/fields.py</code>.</span>

---

## Law 1 — Rice & Tracey (1969)

$$\frac{\dot R}{R}=0.283\,\dot\varepsilon_{\text{eff}}\exp\!\left(\tfrac32\eta\right)
\qquad\xrightarrow{\;\Phi\propto R^3\;}\qquad
\frac{d(\ln\Phi)}{dt}=\mathbf{0.849}\,\dot\varepsilon_{\text{eff}}\exp\!\left(\tfrac32\eta\right)$$

<div class="box">

**Where 0.849 comes from — expect this question.** It is **not fitted**. Rice and
Tracey solved the velocity field around an isolated spherical void in a rigid-plastic
matrix under remote triaxial loading, and obtained the amplitude **0.283** for the
radial growth rate. Since a volume fraction goes as $R^3$,
$\ \frac{\dot\Phi}{\Phi}=3\frac{\dot R}{R}$, giving $3\times0.283=\mathbf{0.849}$.

</div>

**How it indicates damage:** $\Phi$ is a **void volume fraction**, so the law says
voids *grow exponentially with triaxiality*. Tension ($\eta>0$) opens them;
compression ($\eta<0$) suppresses growth.

<span class="tiny">J. R. Rice & D. M. Tracey, <i>On the ductile enlargement of voids in triaxial stress fields</i>, J. Mech. Phys. Solids <b>17</b>(3), 201–217 (1969). Derivation reproduced in <code>PRIMER_fsw_physics_for_me.md</code> §8.</span>

---

## Law 2 — Cockcroft & Latham (1968)

$$C=\int_0^{\bar\varepsilon}\frac{\langle\sigma_1\rangle}{\sigma_{\text{eq}}}\,d\bar\varepsilon
\qquad\Longrightarrow\qquad
\frac{dC}{dt}=\frac{\max(\sigma_1,0)}{\sigma_{\text{eq}}}\,\dot\varepsilon_{\text{eff}}$$

<div class="box">

**The one constant is $C_{\text{crit}}$, and it is a threshold, not a coefficient.**
The integrand itself has no fitted parameter — $C_{\text{crit}}$ only sets *when* the
accumulated value counts as failure, and comes from a tensile or upset test.
Literature values for aluminium alloys are ~0.2–0.5. ⚠️ **We have not calibrated it
for this alloy, so we use $C$ as a ranking only.**

</div>

**How it indicates damage:** $\langle\cdot\rangle$ is the Macaulay bracket,
$\langle x\rangle=\max(x,0)$ — damage accrues **only where $\sigma_1$ is tensile**.

<span class="tiny">M. G. Cockcroft & D. J. Latham, <i>Ductility and the workability of metals</i>, J. Inst. Metals <b>96</b>, 33–39 (1968).</span>

---

## ⚠️ But the bracket almost never switches off here

<div class="warn">

**Measured on the nodal fields:** $\sigma_1>0$ at **~95 %** of active nodes under the
shoulder (4.9–5.9 % negative, consistent across 8 cases).

</div>

Why — $\sigma_1=\sigma_m+\sigma_{\text{eq}}\hat s_1$, and this flow is
**shear-dominated**:

| term | median under the shoulder |
|---|---|
| $\hat s_1$ — deviatoric | **+0.87** |
| $\eta=\sigma_m/\sigma_{\text{eq}}$ — volumetric | **−0.05** |
| $\sigma_1/\sigma_{\text{eq}}$ | **+0.80** |

<div class="ok">

So $\sigma_1$ stays **tensile** even where the *mean* stress is compressive — which it
is at **55 %** of those nodes. **The two laws therefore discriminate differently:**
Rice–Tracey responds to the **volumetric** state via $\exp(\tfrac32\eta)$ and *is*
suppressed by compression; Cockcroft–Latham is driven mainly by the **deviatoric**
state. That is exactly why running both is informative.

</div>

---

## Why run both

<div class="box">

They are **structurally different**: one is exponential in triaxiality and tracks a
physical void fraction; the other is a Macaulay bracket on the maximum principal
stress and is a dimensionless indicator. 
<!-- **Agreement is cheap evidence; disagreement
is diagnostic.** Neither was invented for FSW — both are standard metal-forming laws
with decades of use. -->

</div>

**Chosen for this first pass because they need no calibration.** Richer models remain
open for later work:

| model | what it would add | what it needs |
|---|---|---|
| **Lee–Dawson** | the published FSW precedent — He, Dawson & Boyce (2008) applied it to this exact problem | a coupled $\kappa$ evolution ODE and ~10 constants; ⚠️ clamps at $\sigma_m\le0$, so it cannot represent void **closure** |
| **GTN** | full porous plasticity, with nucleation and coalescence | 7+ constants and a porous-plasticity solve inside the FOM |

<span class="tiny">Both are worth revisiting once a calibration route exists. He <i>et al.</i>'s reported porosity change was 0.01 % → 0.017 %, i.e. a ~70 % relative change on a very small absolute value — informative, but it shows how much the answer depends on the constants.</span>

---

## Two honest limitations of the model

<div class="warn">

**1. These are GROWTH laws, not nucleation laws.** $\Phi = 0$ is a fixed point — no
void ever appears from nothing. They need an assumed initial porosity $\Phi_0$,
which nobody measures directly.

</div>

<div class="warn">

**2. Rice–Tracey has no saturation term.** $\Phi$ is a volume fraction and cannot
exceed 1, reached at $\ln(\Phi/\Phi_0) = 9.21$ for $\Phi_0 = 10^{-4}$. The formula
keeps growing past that — the maximum we measure across the 45 cases is **5242**,
i.e. **569×** the failure threshold.

**So $\ln(\Phi/\Phi_0)$ is a DEFECT-SUSCEPTIBILITY ranking, not a predicted porosity.**
Every run reports what fraction of particles are past failure.

</div>

---

# 2 — The tools

---

## What the 45 cases are

| set | n | what varies | what it is for |
|---|---|---|---|
| **PinShapes** | **20** | **pin geometry only** — rpm, travel speed, thickness and material held constant | ranking tool designs |
| **cohort** | 22 | operating point (rpm, travel speed, mesh) on one cylindrical pin | the ROM training set |
| **validation** | 3 | held-out operating points | checking the ROM |

<div class="box">

⚠️ **The two sets are not directly comparable in absolute value** — different meshes,
speeds and timesteps. Compare *within* a set. "PinShapes" answers *which tool*;
"cohort" answers *which operating point*.

</div>

<span class="tiny">Throughout: <b>cohort</b> = the 22 ROM cases plus, where noted, the 3 validation cases. The four PinShapes families are A-Flats, B-Flutes, C-Threads and D-Concavity.</span>

---

## The four pin families

![w:826](figs_concept/figC4_pin_shapes.png)

<span class="tiny">Each dot is one <b>vertex of the tool's STL surface mesh</b> — the actual geometry, not a computed field. Top and bottom rows are the <b>same points in two projections</b>: (x,y) looking down the axis, (x,z) from the side.</span>

---

## What the geometry actually varies

<div class="box">

⚠️ The PinShapes set is a **pure tool-geometry study at a single operating point**.
rpm, advancing speed, plate thickness and material are all effectively constant.
**Pin geometry is the only real independent variable.**

</div>

Measured from the STLs (M4), not assumed from the folder names:

| family | lobes | lobe depth |
|---|---|---|
| **B-Flutes** | **3, 4, 6, 8** | **20–32 %** of mean radius |
| **C-Threads** | **2** | 3–8 % |
| D-Concavity (D3/D4) | 8 / 2 | 18 % / 4 % |
| A-Flats | — | **below the 1 % detection floor** |

<span class="tiny">⚠️ "A-Flats = 0 lobes" is <b>not</b> a finding: their STLs are too coarse (2,794 triangles, no mid-pin surface) to resolve shallow flats. A data limitation, not a geometric one.</span>

---

# 3 — The implementation

---

## Why pathlines and not a grid

![w:917](figs_concept/figC5_pathline.png)

<span class="tiny">Damage is a <b>history integral</b>. A grid cell has no history — only whatever material is in it now.</span>

---

## The formal reason

For any quantity $\phi$ carried by the material:

$$\underbrace{\frac{D\phi}{Dt}}_{\text{following the particle}}
=\underbrace{\frac{\partial\phi}{\partial t}}_{\text{at a fixed point}}
+\underbrace{u_j\frac{\partial\phi}{\partial x_j}}_{\text{transport}}$$

<div class="ok">

Follow the particle and the **transport term vanishes exactly**, leaving
$D\phi/Dt = \text{(the damage law's right-hand side)}$. The update becomes

$$\phi_{n+1}=\phi_n+\Delta t\cdot\underbrace{f\big(\dot\varepsilon_{\text{eff}},\eta,\sigma_1\big)}_{\text{the rate from Law 1 or Law 2}}$$

one line, no gradient of the damage variable, and no numerical smearing of the sharp
gradients we are trying to resolve.

</div>

<span class="tiny">⚠️ Here $\phi$ is the <b>accumulator</b> — ln Φ for Rice–Tracey, $C$ for Cockcroft–Latham — and the "rate" is that law's right-hand side, e.g. $0.849\,\dot\varepsilon_{\text{eff}}\exp(\tfrac32\eta)$. Triaxiality $\eta$ is an <b>input</b> to the rate, not the rate itself.</span>

Solving it on a grid would require discretising that transport term, which
introduces diffusion precisely where the signal is.

---

## How each quantity is computed

| quantity | from | where |
|---|---|---|
| $\dot\varepsilon_{\text{eff}}$ | $\sqrt{\tfrac23 D_{ij}D_{ij}}$, $D = \tfrac12(\nabla u + \nabla u^{\!\top})$ | nodal, once per timestep |
| $\sigma_m$ | $-p$ directly from the FOM pressure | nodal |
| $\sigma_{\text{eq}}$ | Norton–Hoff from the case's **own** `.mat` tables | nodal |
| $\eta$, $\sigma_1/\sigma_{\text{eq}}$ | derived from the above | nodal |

<div class="box">

**Precomputed on the mesh, sampled like velocity.** Three nodal scalars
$(\dot\varepsilon,\ \eta,\ \sigma_1/\sigma_{\text{eq}})$ are stacked as
`(3, n_timesteps, n_nodes)` and uploaded once. The kernel indexes them with the
same `time_idx % n_timesteps` as the velocity field — a steady case is simply
`n_timesteps = 1`.

</div>

---

## Where σ_eq comes from

The FOM uses **Norton–Hoff**, with temperature-tabulated coefficients read from each
case's own `.mat` file:

$$\sigma_{\text{eq}} = \underbrace{\text{VISCO}(T)}_{\text{consistency}}\cdot
\dot\varepsilon^{\,\overbrace{\text{EXPVI}(T)}^{\text{rate exponent }m}}$$

<span class="tiny">VISCO and EXPVI are the solver's own table names — a viscosity-like prefactor and the strain-rate sensitivity exponent, both tabulated against temperature. $m\to0$ is perfectly plastic, $m\to1$ Newtonian. Here $m\approx0.086$.</span>

<div class="box">

**Can we just use the same `.mat` file?**  The tables are written in shear-equivalent measures, and our pipeline is von Mises. The conversion is a constant factor, applied once.

</div>

---

## The √3 conversion

The solver's own definition uses **shear** measures
($\gamma=\sqrt2\|\mathbf{e}\|$, $\tau=\tfrac{\sqrt2}{2}\|\mathbf{s}\|$), while we work
in von Mises. Since $\gamma=\sqrt3\,\dot\varepsilon_{vM}$ and $\sigma_{vM}=\sqrt3\,\tau$,
**both the rate and the stress pick up a √3**:

$$\sigma_{\text{eq}} = \sqrt3\;\text{VISCO}(T)\;\big(\sqrt3\,\dot\varepsilon\big)^{m}$$

<div class="ok">

**Checked against the solver's own exported stress, 581,488 cells:**

| | ratio to the solver's $\sqrt{3J_2}$ |
|---|---|
| tables used as-is | **0.569** |
| with the conversion | **1.035** |

Predicted factor $\sqrt3(\sqrt3)^m = 1.816$ vs measured **1.819** — **0.13 %**.

</div>

<!-- <span class="tiny">Source: Venghaus, <i>Finite Elements in Analysis & Design</i> <b>224</b> (2023) 103986, Publication 1, eq. (5).
⚠️ This is the <b>material viscosity</b> convention — not the "modified Norton" <i>friction</i> law, which is a separate boundary condition.</span> -->

---

## Solving the ODEs

Explicit Euler at the **k1** stage of the existing RK4 tracker:

```python
eta_c   = clip(eta_k1, -3.0, 3.0)              # BEFORE the exponential
dlogphi = 0.849 * edot_k1 * exp(1.5 * eta_c)   # Rice-Tracey, LOG space
dC      = maximum(s1_k1, 0.0) * edot_k1        # Cockcroft-Latham
dmg_new = dmg + dt * stack([dlogphi, dC])
```

| choice | reason |
|---|---|
| **Euler, not RK4** | the **driver** — the nodal field ($\dot\varepsilon_{\text{eff}}$, $\eta$, $\sigma_1/\sigma_{\text{eq}}$) that feeds the rate — is itself interpolated, so 4th order on top of interpolation error buys nothing for 4× the samples |
| **$\ln\Phi$, not $\Phi$** | the ODE is multiplicative; log space is linear in the accumulator, cannot underflow, keeps $\Phi>0$. Stable to **1.2e-13** over 10,000 steps |
| **clip $\eta$ first** | a bad pressure value would overflow $\exp$ and poison the accumulator for the rest of the run |
| **reuse `elem_k1`** | the element search is already paid for by the velocity step |

---

## How the damage ODEs are integrated

<div class="ok">

**Both the position and the damage use the same four RK4 stages** — the damage
sampler reuses the elements the velocity step already located, so there are **no
extra element searches**.

</div>

But they are solving two different problems, and the word "RK4" means something
different in each:

| | **position** $x$ | **damage accumulator** $\Phi$ |
|---|---|---|
| equation | $\dfrac{dx}{dt}=u\big(x(t),t\big)$ | $\dfrac{d\ln\Phi}{dt}=f\big(x(t),t\big)$ |
| does the **unknown** appear on the right? | **yes** — $x$ does | **no** — $\ln\Phi$ does not |
| so it is | an initial-value problem | a **quadrature** along a known path |
| "RK4" means | the classic 4th-order integrator | the same $(k_1{+}2k_2{+}2k_3{+}k_4)/6$ weights |
| 4th-order convergence? | yes | **no** — see below |

<span class="tiny">⚠️ <b>This does not mean the right-hand side is constant.</b> η and ε̇ vary strongly — in <b>space</b> (sampled at the particle's moving position) and in <b>time</b> (the nodal fields are time-dependent). That variation is exactly what we integrate. The narrow point is that <b>lnΦ itself never appears on the right</b>, so along an already-determined path the equation reads d lnΦ/dt = f(t) — no feedback, hence a quadrature rather than a coupled solve.</span>

---

## Why the damage quadrature is not formally 4th order

<div class="box">

**It is RK4** — the same $(k_1{+}2k_2{+}2k_3{+}k_4)/6$ combination, with the four
stages sampled at the four RK4 trajectory points, which are four **different
positions** in the mesh. Since the damage RHS does not depend on the accumulator,
the method acts as a quadrature along the path the position solve has already fixed.

</div>

<div class="warn">

⚠️ **But the order-4 convergence rate is not achieved here.** The driver
($\dot\varepsilon$, $\eta$, $\sigma_1$) is **P1-interpolated**, so along a path it is
$C^0$ with a **kink at every element face**. Any 4th-order quadrature needs a smooth
integrand; across a kink it degrades to 1st–2nd order.

</div>

<div class="ok">

**Its real benefit is different, and it is worth having.** At 9°/step the k1-only
estimate samples the **start** of the step. The k2/k3 stages sample the **midpoint**,
removing a bias that bites wherever the driver has a steep gradient — i.e. exactly in
the shear layer.

</div>

<span class="tiny">Both orders are implemented (<code>--damage-order {1,4}</code>); <b>all 45 cases in this talk ran order 4</b>. Cost: 3 extra gathers per scalar at elements already located — measured throughput penalty <b>−2.3 %, within noise</b>.</span>

---

## What the comparison showed

Controlled A/B — identical case, seed, particles and steps, **only the order differs**:

| | order 1 | order 4 | change |
|---|---|---|---|
| $\ln\Phi$ median | 0.004456 | 0.004456 | **+0.01 %** |
| $\ln\Phi$ **max** | 179.85 | **147.67** | **−17.9 %** |

<span class="tiny">Position differed by <b>0.000e+00 m</b> — the flag provably does not perturb the trajectory.</span>

<div class="warn">

⚠️ **A tail correction, not a bulk one.** A typical particle changes by 0.05 %; the
maximum drops 18 %. First order would be fine for **median-based** rankings and
**not** fine for **tail** metrics — `frac_failed`, p99, max.

</div>

<div class="ok">

**Damage arrives in bursts: the median particle collects 84 % of its total in 10 % of
its steps** (uniform would be 10 %; 87 % of particles exceed 50 %). The tail is
therefore where the integrand is worst behaved — which is why all 45 cases ran at
order 4.

</div>

<span class="tiny">⚠️ The A/B above is <b>one case</b>; a sweep across families would test whether the 18 % shift is case-dependent.</span>

---

## Cost: measured, not argued

A/B test on the production kernel, damage ON vs OFF, alternating:

| platform | slowdown | gate | accumulator verified |
|---|---|---|---|
| RTX 5090 (CUDA) | **+0.15 … +0.51 %** | < 3 % | 99.8 % of particles |
| LUMI MI250X (ROCm) | **+1.77 %** | < 3 % | 100.0 % |

<div class="ok">

The accumulator stays on the **GPU** for the whole run — no per-step host transfer.
The marginal cost is one extra 4-float gather per step, from memory already in cache
for the velocity read.

</div>

<span class="tiny">⚠️ The "accumulator verified" column exists because a timing gate alone would pass identically if the damage code were dead. The gate now fails if nothing accumulates.</span>

---

## Step count derived per case, not fixed

$$n_{\text{steps}} = \frac{\text{travel distance}}{v_{\text{adv}}\cdot \Delta t}$$

with $v_{\text{adv}}$ and $\Delta t$ read from each case's own `data/*.som.{dat,fix}`.

| class | 4 mm | 10 mm | 20 mm |
|---|---|---|---|
| cohort, slow | 213 | 533 | 1067 |
| cohort, fine mesh | 427 | 1067 | 2133 |
| **PinShapes** | 1102 | **2756** | 5511 |

<!-- <div class="warn">

⚠️ A **fixed** step count compares different physical experiments — $\Delta t$ and
$v_{\text{adv}}$ both vary per case. The Phase 1 gate ran 400 steps and **not one
particle reached the tool**: 400 steps is 1.45 mm of advance, and the seed box sits
8–11 mm upstream.

</div> -->

---

# 4 — Example cases

---

## What the particles carry

Every particle ends the run with two accumulated numbers:

| array | meaning |
|---|---|
| `Damage_lnPhi` | Rice–Tracey $\ln(\Phi/\Phi_0)$ |
| `Damage_CL` | Cockcroft–Latham $C$ |
| `Damage_fracFailed` | $\ln\Phi$ clipped at the $\Phi\!=\!1$ threshold — a **usable** colour scale |

<div class="box">

Now exported as **particle PointData** in VTU/VTKHDF, so pathlines can be coloured
by damage directly in ParaView. Verified growing monotonically across export steps.

**For the 45 completed runs** (which used `--no-export`): `npz_to_vtu.py` converts
the saved final positions + damage to VTU in seconds — **no re-run needed** for a
static view. Trajectories *do* need a re-run.

</div>

---

## Where damage ends up, radially

![w:773](figs_phase4/fig3_radial_profile.png)

<span class="tiny">Median ln(Φ/Φ₀) against distance from the tool axis. The shaded band is the pin (r ≲ 2.4 mm); the dashed line is the shoulder at 7 mm.</span>

---

# 5 — All 45 cases

---

## The two damage models agree

![w:900](figs_phase4/fig2_models_agree.png)

<div class="ok">

**Rank correlation +0.982 to +0.998 on all 45 cases** — two structurally different
laws, opposite rotation senses. **The implementation is self-consistent.**

</div>

---

## ⚠️ But the accumulated damage does not reproduce the η result

![w:919](figs_phase4/fig4_advancing_wake.png)

<span class="tiny">Each case plotted against its <b>own</b> advancing side (PinShapes CCW ⇒ +y; cohort CW ⇒ −y, matching the measured flank sign). Phase 4: advancing is worse in <b>10/45</b> cases, against 0/45 in Phase 3.</span>

---

## ⚠️ Root cause: most particles never reach the tool

The FOM is solved in the **tool-fixed frame** — the inlet BC drives material past a
stationary tool (`111  5.0e-03 0.0 0.0` on 71 nodes). So a far-field particle *must*
advect the full travel distance. It does not:

| case | sim time | $v_{\text{adv}}$ | expected drift | **actual mean x** |
|---|---|---|---|---|
| ps_A1 | 1.000 s | 10 mm/s | +10 mm | **−0.52 mm** |
| rom_000 | 1.999 s | 5 mm/s | +10 mm | +5.77 mm |

<div class="warn">

PinShapes particles moved **backwards by 0.5 mm** over a nominal 10 mm. They are
**recirculating**, not lagging. Fraction accumulating *essentially zero* damage —
the **bypass** population, i.e. particles that stream past without ever entering the
shear layer: **80–88 %** (PinShapes), **52–65 %** (cohort).

**The global median was therefore measuring the bypass fraction, not damage.**

</div>

---

## Regional metrics recover the signal

Excluding the bypass population and the top boundary layer:

| family | global median | **stir zone median** | z_top p99 | bypass |
|---|---|---|---|---|
| A-Flats | 0.0650 | 0.3824 | 358.6 | 80.6 % |
| **B-Flutes** | 0.0654 | **0.4131** | 356.1 | 80.3 % |
| C-Threads | 0.0653 | 0.3790 | 357.5 | 80.5 % |
| D-ConcavityTilt | 0.0521 | 0.3050 | **22.6** | 87.2 % |
| cohort | 0.4909 | 2.2787 | 265.7 | 57.7 % |

<div class="ok">

**A/B/C spread: 0.6 % → 9.0 %, a 15× stronger signal — and B-Flutes is highest,**
which is the expected ordering since B has the deepest lobes (20–32 % of pin radius).

</div>

<span class="tiny">Stir zone = r &lt; 7 mm AND outside the top 20 % of thickness. The z_top column is reported separately because it sits against the prescribed-velocity top surface and is partly a BC response — note D at 22.6 vs ~358, which also explains its low accumulation fraction.</span>

---

## Phase 4: 45 cases, 30 mm, RK4 damage

<div class="ok">

✅ **All 45 completed, zero failures.** The bundle now clears the tool — which it
did not at 10 mm.

</div>

| | Phase 3 (10 mm) | Phase 4 (30 mm) |
|---|---|---|
| ps_A1 mean x | **−0.52 mm** | **+18.95 mm** |
| ps_A1 past x > +7 mm | — | **97.2 %** |
| ps_A1 still inside r < 7 mm | 22.5 % | **2.6 %** |
| PinShapes bypass | 80–87 % | 67–80 % (**−13 pts**) |
| cohort bypass | 58 % | 51 % (−6.6 pts) |

<span class="tiny">⚠️ The PinShapes drop is <b>twice</b> the cohort's. PinShapes was genuinely <b>recirculation</b>-limited and the longer run released it; the cohort was never trapped, only truncated. An earlier reading that called the bypass "structural" was based on cohort cases alone and was wrong for PinShapes.</span>

---

## The wake box — scoring only processed material

| wall | where | why |
|---|---|---|
| $x_{\min}$ | **+7 mm** | after the shoulder: drops the tool footprint and the 2.6 % still orbiting |
| $x_{\max}$ | downstream front | — |
| $y$ | full span | **kept** — the advancing/retreating contrast lives here |
| $z_{\min}$ | domain | **kept** — the weld root matters |
| $z_{\max}$ | domain **− 8 % of thickness** | drops the BC-contaminated top band (p99 there is 5–16× the bulk) |

<div class="box">

⚠️ "2–3 particle layers" could not be used literally: seeding is **random**, so $z$ is
continuous — 6001 distinct values in 100k particles. 8 % of thickness is 0.36 mm
(cohort) / 0.48 mm (PinShapes), the same scale, but mesh-independent.

The box keeps **~90 % of all particles** — a targeted exclusion, not a filter that
throws the statistics away.

</div>

---

## Family separation, averaged over the wake box

| family | ln_mean | ln_med | ln_p99 | frac_failed |
|---|---|---|---|---|
| A-Flats | 6.332 | 0.2413 | 69.0 | 19.4 % |
| **B-Flutes** | **7.166** | **0.2519** | **79.8** | **20.9 %** |
| C-Threads | 6.316 | 0.2500 | 69.7 | 20.1 % |
| D-ConcavityTilt | 2.650 | 0.1745 | 39.9 | 9.0 % |
| cohort | 5.554 | 0.9146 | 49.4 | 17.6 % |

<div class="ok">

✅ **B-Flutes is highest on every statistic** — mean, median, p99 and frac_failed.
That is the expected ordering: B has the **deepest lobes** (20–32 % of pin radius).
A and C sit within 0.2 % of each other, matching their similar shallow geometry.
**This is the cleanest family separation obtained so far.**

</div>

<span class="tiny">⚠️ D stays low (2.650) even with the top band removed, so its deficit is <b>not</b> purely the z_max artefact. ⚠️ Despite the folder name, <b>no case here has tool tilt</b> — all 20 report 0.0°, so what distinguishes D is its pin STL geometry alone, and the cause is <b>unexplained</b>.</span>

---

## ① Raw particles — A-Flats (1/3)

![w:790](figs_wake/figW_A-FlatsVariations_1_yz.png)

<span class="tiny">Every particle in the wake box, coloured by accumulated damage. <b>Nothing averaged or binned.</b> Equal y and z scales. ⚠️ Speckle is <b>sampling</b>, not physics — trust the envelope, not individual markers.</span>

---

## ① Raw particles — A-Flats (2/3)

![w:790](figs_wake/figW_A-FlatsVariations_2_yz.png)

<span class="tiny">Every particle in the wake box, coloured by accumulated damage. <b>Nothing averaged or binned.</b> Equal y and z scales. ⚠️ Speckle is <b>sampling</b>, not physics — trust the envelope, not individual markers.</span>

---

## ① Raw particles — A-Flats (3/3)

![w:790](figs_wake/figW_A-FlatsVariations_3_yz.png)

<span class="tiny">Every particle in the wake box, coloured by accumulated damage. <b>Nothing averaged or binned.</b> Equal y and z scales. ⚠️ Speckle is <b>sampling</b>, not physics — trust the envelope, not individual markers.</span>

---

## ① Raw particles — B-Flutes (1/2)

![w:790](figs_wake/figW_B-FluteVariations_1_yz.png)

<span class="tiny">Every particle in the wake box, coloured by accumulated damage. <b>Nothing averaged or binned.</b> Equal y and z scales. ⚠️ Speckle is <b>sampling</b>, not physics — trust the envelope, not individual markers.</span>

---

## ① Raw particles — B-Flutes (2/2)

![w:790](figs_wake/figW_B-FluteVariations_2_yz.png)

<span class="tiny">Every particle in the wake box, coloured by accumulated damage. <b>Nothing averaged or binned.</b> Equal y and z scales. ⚠️ Speckle is <b>sampling</b>, not physics — trust the envelope, not individual markers.</span>

---

## ① Raw particles — C-Threads (1/2)

![w:790](figs_wake/figW_C-ThreadsVariations_1_yz.png)

<span class="tiny">Every particle in the wake box, coloured by accumulated damage. <b>Nothing averaged or binned.</b> Equal y and z scales. ⚠️ Speckle is <b>sampling</b>, not physics — trust the envelope, not individual markers.</span>

---

## ① Raw particles — C-Threads (2/2)

![w:790](figs_wake/figW_C-ThreadsVariations_2_yz.png)

<span class="tiny">Every particle in the wake box, coloured by accumulated damage. <b>Nothing averaged or binned.</b> Equal y and z scales. ⚠️ Speckle is <b>sampling</b>, not physics — trust the envelope, not individual markers.</span>

---

## ① Raw particles — D-Concavity (1/2)

![w:790](figs_wake/figW_D-ConcavityTilt_1_yz.png)

<span class="tiny">Every particle in the wake box, coloured by accumulated damage. <b>Nothing averaged or binned.</b> Equal y and z scales. ⚠️ Speckle is <b>sampling</b>, not physics — trust the envelope, not individual markers.</span>

---

## ① Raw particles — D-Concavity (2/2)

![w:790](figs_wake/figW_D-ConcavityTilt_2_yz.png)

<span class="tiny">Every particle in the wake box, coloured by accumulated damage. <b>Nothing averaged or binned.</b> Equal y and z scales. ⚠️ Speckle is <b>sampling</b>, not physics — trust the envelope, not individual markers.</span>

---

## ② From particles to a continuous field

<div class="box">

At each point of a 600 × 240 grid: take the **16 nearest particles**, report the
**median** of their ln(Φ/Φ₀).

</div>

| choice | why |
|---|---|
| **median**, not mean | lnΦ spans five decades and is right-skewed — a mean maps single outliers |
| **kNN**, not a histogram bin | at 0.18 mm mean spacing, a 600 × 240 bin grid would be **more than half empty**; kNN gives every output point 16 real samples |
| **blank** beyond 0.45 mm | **grey = no data**, which is not the same statement as *low damage* |

---

## ② Continuous field — A-Flats (1/3)

![w:790](figs_wake/figWB_A-FlatsVariations_1_yz_field.png)

<span class="tiny">16-nearest-neighbour median of ln(Φ/Φ₀) on a 600 × 240 grid. The <b>envelope is unchanged</b> from the raw view; the speckle resolves into a gradient. Grey = <b>no data</b>, not low damage.</span>

---

## ② Continuous field — A-Flats (2/3)

![w:790](figs_wake/figWB_A-FlatsVariations_2_yz_field.png)

<span class="tiny">16-nearest-neighbour median of ln(Φ/Φ₀) on a 600 × 240 grid. The <b>envelope is unchanged</b> from the raw view; the speckle resolves into a gradient. Grey = <b>no data</b>, not low damage.</span>

---

## ② Continuous field — A-Flats (3/3)

![w:790](figs_wake/figWB_A-FlatsVariations_3_yz_field.png)

<span class="tiny">16-nearest-neighbour median of ln(Φ/Φ₀) on a 600 × 240 grid. The <b>envelope is unchanged</b> from the raw view; the speckle resolves into a gradient. Grey = <b>no data</b>, not low damage.</span>

---

## ② Continuous field — B-Flutes (1/2)

![w:790](figs_wake/figWB_B-FluteVariations_1_yz_field.png)

<span class="tiny">16-nearest-neighbour median of ln(Φ/Φ₀) on a 600 × 240 grid. The <b>envelope is unchanged</b> from the raw view; the speckle resolves into a gradient. Grey = <b>no data</b>, not low damage.</span>

---

## ② Continuous field — B-Flutes (2/2)

![w:790](figs_wake/figWB_B-FluteVariations_2_yz_field.png)

<span class="tiny">16-nearest-neighbour median of ln(Φ/Φ₀) on a 600 × 240 grid. The <b>envelope is unchanged</b> from the raw view; the speckle resolves into a gradient. Grey = <b>no data</b>, not low damage.</span>

---

## ② Continuous field — C-Threads (1/2)

![w:790](figs_wake/figWB_C-ThreadsVariations_1_yz_field.png)

<span class="tiny">16-nearest-neighbour median of ln(Φ/Φ₀) on a 600 × 240 grid. The <b>envelope is unchanged</b> from the raw view; the speckle resolves into a gradient. Grey = <b>no data</b>, not low damage.</span>

---

## ② Continuous field — C-Threads (2/2)

![w:790](figs_wake/figWB_C-ThreadsVariations_2_yz_field.png)

<span class="tiny">16-nearest-neighbour median of ln(Φ/Φ₀) on a 600 × 240 grid. The <b>envelope is unchanged</b> from the raw view; the speckle resolves into a gradient. Grey = <b>no data</b>, not low damage.</span>

---

## ② Continuous field — D-Concavity (1/2)

![w:790](figs_wake/figWB_D-ConcavityTilt_1_yz_field.png)

<span class="tiny">16-nearest-neighbour median of ln(Φ/Φ₀) on a 600 × 240 grid. The <b>envelope is unchanged</b> from the raw view; the speckle resolves into a gradient. Grey = <b>no data</b>, not low damage. ⚠️ The large <b>grey</b> region is where <b>no particle reaches</b> — examined later.</span>

---

## ② Continuous field — D-Concavity (2/2)

![w:790](figs_wake/figWB_D-ConcavityTilt_2_yz_field.png)

<span class="tiny">16-nearest-neighbour median of ln(Φ/Φ₀) on a 600 × 240 grid. The <b>envelope is unchanged</b> from the raw view; the speckle resolves into a gradient. Grey = <b>no data</b>, not low damage. ⚠️ The large <b>grey</b> region is where <b>no particle reaches</b> — examined later.</span>

---

## How does the tool concentrate damage?

![w:840](figs_traj/figT_ps_A-FlatsVariations_A1_3d.png)

<span class="tiny">Individual particles followed through all 167 exported states, coloured by the damage accumulated <b>so far</b>, so a path brightens exactly where damage is picked up. ● seed, ■ final position. Particle identity is verified, not assumed — see the note below.</span>

---

## Damage is acquired in discrete capture events

![w:795](figs_traj/figT_ps_A-FlatsVariations_A1_time.png)

<span class="tiny">Upper: accumulated ln(Φ/Φ₀). Lower: the <b>same</b> particles' distance from the tool axis.</span>

---

## What the trajectories establish

<div class="ok">

**1. Capture, not gradual shear.** A particle approaches, is **captured into orbit**
around the pin, circles several times while brightening, then releases downstream.
The green trace rises **three orders of magnitude in one near-vertical step**
(0.06 → 100) and is then flat for the remaining ~45 revolutions.

**2. Final damage ≈ how many times a particle was captured.** The red traces are
staircases — distinct bursts at steps ~1000, ~3300, ~5200, each a separate close
approach.

**3. Every jump coincides with $r$ dropping below the 7 mm shoulder line.** The two
panels are locked together.

</div>

---

## …and two more

<div class="ok">

**4. The bypass population is real.** The blue control sits flat at ~0.1 throughout
and never approaches the tool — it is not "weakly damaged", it is **unprocessed**.
This retroactively justifies excluding it rather than averaging it in.

**5. The capture orbits sit at $r \approx 3$–5 mm** — between the pin (2.4 mm) and
the shoulder (7 mm), and at $z \approx -0.5$ to $-2$ mm, the **upper third** of a
6 mm plate. The trajectories **locate the shear layer directly**, in both radius and
depth, with no velocity field needed.

</div>

<span class="tiny">⚠️ Damage is acquired near the top surface — the same depth as the boundary artefact. That is why the z_max cut is only 8 % of thickness: a deeper cut would remove signal along with the artefact.</span>

<div class="box">

⚠️ **Particle identity verified three ways** before building on it: `ParticleID`
unchanged across files; row displacement over 50 steps **0.18 mm** vs **12.98 mm**
for a shuffled control; and lnΦ decreased in **0 of 100,000 rows** — and lnΦ can only
accumulate, so any row swap would show up as a decrease.

</div>

---

## What it means for the damage model

<div class="box">

Rice–Tracey and Cockcroft–Latham are **rate laws integrated along the path**, and
the integral is **strongly concentrated in time**: the median particle collects
**84 % of its damage in 10 % of its steps**, against 10 % if it were uniform.

</div>

Two consequences:

- the metric is sensitive to **how often a particle is re-entrained** — a property of
  the **tool geometry**, which is exactly what we want to rank;
- it is also sensitive to the step size *inside* a capture, where the integrand is
  largest and fastest-varying. **That is precisely where the order-4 quadrature earns
  its −17.9 % tail correction.**

---

## Two things the projection shows

<div class="box">

**1. The damaged zone is a cone that widens toward the top** — consistent with the
shoulder driving more material than the pin.

**2. It is visibly asymmetric in $y$**, leaning one way at the root and the other
near the top.

</div>

Quantified over the wake box, each case against its **own** advancing side:

| | advancing | retreating | ratio | adv > ret |
|---|---|---|---|---|
| **PinShapes** (CCW) | 0.2292 | 0.2317 | **0.989** | **12 / 20** |
| cohort + val (CW) | 0.7704 | 1.2013 | 0.641 | **0 / 25** |

<div class="warn">

⚠️ **The two classes now split.** In Phase 3 the advancing side was lower in
**45/45**. With the wake box and 30 mm, PinShapes is **balanced** (ratio 0.989)
while the cohort stays firmly retreating-higher. So O33 was **partly** a
measurement artefact — but a real cohort-specific asymmetry survives, and it is
still unexplained.

</div>

---

## What this does and does not mean

The two quantities answer different questions:

| | question |
|---|---|
| Stage 1 | *at nodes in the shear layer, what is the stress state now?* |
| Phase 3 | *what did a particle accumulate along its whole route?* |

<div class="ok">

**This is now diagnosed, not speculative.** Particles *do* pass around rather than
through the shear layer — measured as a 52–88 % zero-damage population. The global
median diluted both the flank contrast and the family contrast, which were the two
symptoms observed.

**Fix:** seed within a y-band that enters the stir zone, and report regional metrics.
⚠️ Tripling the step count does **not** fix this — it accumulates more damage in the
recirculating population while bypass particles still contribute zero, at 3× the cost.

</div>

---

## One more open item

![w:900](figs_phase4/fig5_accumulation.png)

<span class="tiny">The entire <b>D-Concavity</b> family accumulates ~16 points below every other case. ⚠️ The folder name says "Tilt", but <b>all 20 cases report Tilt angle 0.0</b> — so this is <b>not</b> explained by tilt, and the cause is open.</span>

---

## Two indicators that must not share an estimator

<div class="warn">

⚠️ The cross-sections above colour a **k-nearest-neighbour** median — and that is
**not density-independent**. Sparse points draw their neighbours from a wider region,
so the estimate is smoothed over more space exactly where particles are scarce.

Density–damage rank correlation: **−0.06** (A-Flats), **−0.07** (B-Flutes),
**−0.29 (D-Concavity)** — whose density spans a 27× range.

</div>

Particle tracking and damage integration are meant to be **two independent routes to
the same prediction**, so they must not share an estimator:

| | density | damage |
|---|---|---|
| estimator | plain count per unit area | median over a **fixed-area** cell |
| support | — | identical everywhere, by construction |
| sparse cells | shown as empty | report **nothing** — "not measurable" ≠ "low" |

<div class="ok">

After the change, D-Concavity's correlation falls **−0.29 → +0.015**. The coupling was
almost entirely the estimator.

</div>

---

## A-Flats A2 — both indicators agree

![w:653](figs_compare/figX_ps_A-FlatsVariations_A2_density_vs_damage.png)

<span class="tiny">Density uniform except <b>one sharply-bounded void</b>, lying <b>directly beneath the brightest damage band</b>. The agreement panel is strongly positive all around it.</span>

---

## D-Concavity D1 — larger void, depleted surroundings

![w:653](figs_compare/figX_ps_D-ConcavityTilt_D1_density_vs_damage.png)

<span class="tiny">The void is genuinely empty, and the damage field does drape over its upper boundary. But the surrounding material is globally depleted — see the next slide.</span>

---

## ⚠️ None of these voids are credible

Across **all 20** tool cases, void area versus the fraction of particles still stuck
at the tool:

| | Spearman ρ |
|---|---|
| void area vs **trapped fraction** | **+0.980** |
| void area vs trapped/wake damage ratio | **−0.932** |
| same, **excluding D entirely** | **+0.954** |

<div class="warn">

⚠️ **Excluding D-Concavity does not weaken it** — the relationship holds *within*
A, B and C. Void area is predicted by **how many particles get stuck**, not by weld
physics — though the next slides show that the sticking is reproduced by an
independent tracker, so it is not *our* tracker's fault.

</div>

---

## The A-Flats ladder

| case | trapped | trap ÷ wake damage | void |
|---|---|---|---|
| A1 | 2.6 % | 1003 | **0.15 mm²** |
| A3old | 3.4 % | 370 | 1.49 mm² |
| A3 | 3.8 % | 67 | 1.62 mm² |
| A2old | 4.1 % | 54 | 2.20 mm² |
| **A2** | **5.2 %** | **39** | **3.64 mm²** |

<div class="warn">

**A2 — the "best candidate" two slides ago — is simply the case with the most
trapping.** More trapping → less damage on the trapped particles → bigger void.

</div>

---

## The diagnostic: trapped particles that do not accumulate

A particle genuinely held in the shear layer should accumulate damage **hard**.

| family | trapped | **trapped ÷ wake damage** |
|---|---|---|
| B-Flutes | 2.7–3.0 % | **868–1025** ✅ |
| C-Threads | 3.2–3.4 % | 638–769 ✅ |
| A-Flats | 2.5–5.2 % | 39–1003 (bimodal) |
| **D-Concavity** | **17.1–18.7 %** | **8–23** ⚠️ |

<div class="warn">

D's trapped particles sit at $r \approx 3.65$ mm — **inside** the shear layer — and
accumulate **almost nothing**. They are circulated by the interpolated field *without
the strain rates that being that close to the tool implies*. That is a **numerical
trap**.

</div>

<div class="box">

⚠️ **Why the two-indicator agreement was not evidence.** Density is depleted by
trapping; the damage field is shaped by which particles reach the wake. **Both are
downstream of the same failure.**

</div>

---

## Reported as a negative result

<div class="warn">

⚠️ **No void in this dataset is a credible defect prediction.** Two earlier readings
were both wrong, in opposite directions — the D void as a real filling failure, and
the A2 void as a defect confirmed by two indicators.

</div>

<div class="ok">

✅ **The damage-based tool ranking is unaffected.** It is computed over the wake
population and does not use the void measure. B-Flutes remains highest on every
statistic.

</div>

**The direct test needs no new simulation:** count particles crossing a control
surface around the tool, **in versus out**. In a steady incompressible flow they must
balance. The exported trajectory series already contains everything required.

---

## ③ PT vs damage — A-Flats (1/2)

![w:700](figs_compare/figP_A-FlatsVariations_1_pt_vs_damage.png)

<span class="tiny">Top of each pair: the <b>earlier, separate</b> campaign — 288,000 <b>grid</b>-seeded particles, coloured by seed slab; a defect shows where no slab arrives. Bottom: our damage field. ⚠️ Different counts and seeding — compare <b>where</b> features sit, not cloud density.</span>

---

## ③ PT vs damage — A-Flats (2/2)

![w:700](figs_compare/figP_A-FlatsVariations_2_pt_vs_damage.png)

<span class="tiny">Top of each pair: the <b>earlier, separate</b> campaign — 288,000 <b>grid</b>-seeded particles, coloured by seed slab; a defect shows where no slab arrives. Bottom: our damage field. ⚠️ Different counts and seeding — compare <b>where</b> features sit, not cloud density.</span>

---

## ③ PT vs damage — B-Flutes (1/2)

![w:700](figs_compare/figP_B-FluteVariations_1_pt_vs_damage.png)

<span class="tiny">Top of each pair: the <b>earlier, separate</b> campaign — 288,000 <b>grid</b>-seeded particles, coloured by seed slab; a defect shows where no slab arrives. Bottom: our damage field. ⚠️ Different counts and seeding — compare <b>where</b> features sit, not cloud density.</span>

---

## ③ PT vs damage — B-Flutes (2/2)

![w:700](figs_compare/figP_B-FluteVariations_2_pt_vs_damage.png)

<span class="tiny">Top of each pair: the <b>earlier, separate</b> campaign — 288,000 <b>grid</b>-seeded particles, coloured by seed slab; a defect shows where no slab arrives. Bottom: our damage field. ⚠️ Different counts and seeding — compare <b>where</b> features sit, not cloud density.</span>

---

## ③ PT vs damage — C-Threads (1/2)

![w:700](figs_compare/figP_C-ThreadsVariations_1_pt_vs_damage.png)

<span class="tiny">Top of each pair: the <b>earlier, separate</b> campaign — 288,000 <b>grid</b>-seeded particles, coloured by seed slab; a defect shows where no slab arrives. Bottom: our damage field. ⚠️ Different counts and seeding — compare <b>where</b> features sit, not cloud density.</span>

---

## ③ PT vs damage — C-Threads (2/2)

![w:700](figs_compare/figP_C-ThreadsVariations_2_pt_vs_damage.png)

<span class="tiny">Top of each pair: the <b>earlier, separate</b> campaign — 288,000 <b>grid</b>-seeded particles, coloured by seed slab; a defect shows where no slab arrives. Bottom: our damage field. ⚠️ Different counts and seeding — compare <b>where</b> features sit, not cloud density.</span>

---

## ③ PT vs damage — D-Concavity

![w:700](figs_compare/figP_D-ConcavityTilt_pt_vs_damage.png)

<span class="tiny">Top of each pair: the <b>earlier, separate</b> campaign — 288,000 <b>grid</b>-seeded particles, coloured by seed slab; a defect shows where no slab arrives. Bottom: our damage field. ⚠️ Different counts and seeding — compare <b>where</b> features sit, not cloud density.</span>

---

## ✅ D's anomaly reproduces in the independent run

![w:980](figs_compare/figP_D-ConcavityTilt_pt_vs_damage.png)

| case | **old PT** (288k grid) | **ours** (100k random) |
|---|---|---|
| **D3** | **19.98 %** trapped | **18.5 %** |
| B2 | 6.61 % | 2.8 % |
| C1 | 2.55 % | 3.4 % |

<span class="tiny">The <b>same central void</b> appears in both rows — different code, different seeding, different year.</span>

---

## What that changes

<div class="ok">

✅ **The trapping is not an artefact of our tracker.** An independent run reproduces
it to within ~1.5 points.

</div>

<div class="warn">

⚠️ **So the cause is upstream of any particle tracker** — either D's velocity field
genuinely holds material near the pin, or its mesh does. **Both trackers would
faithfully reproduce either.**

</div>

<div class="box">

**This makes the physical reading more plausible than it was** — if the D tool
really produces a closed recirculation material cannot escape, that *is* a filling
failure and the void is real. But it is still not established, and settling it needs
the **FOM side**, not the tracker.

</div>

<span class="tiny">⚠️ Recall the D case directories were copied from A1/A2/B4/C2 with only the tool STL replaced, so the mesh may be refined for the <b>source</b> case's pin.</span>

---

## ⏳ Placeholder — does B-Flutes capture more often?

<div class="box">

**PENDING FIGURE:** `figs_traj/figT_ps_B-FluteVariations_B2_3d.png`
and `figT_ps_B-FluteVariations_B2_time.png`

**The question:** B-Flutes has the deepest lobes (20–32 % of pin radius) and the
highest damage on every statistic. If the capture mechanism is the explanation,
B-Flutes particles should show **more capture events per particle** — more stairs in
the time plot — rather than more damage per capture.

**If confirmed**, the family ranking has a direct geometric cause: deeper lobes
re-entrain material more often. That would turn a correlation into a mechanism.

**Cost:** read-only on existing exports, minutes with the cache.

</div>

---

## ⏳ Placeholder — why does D-Concavity accumulate so little?

<div class="box">

**PENDING FIGURE:** `figs_traj/figT_ps_D-ConcavityTilt_D1_*`

**The question:** D sits at 2.650 against 6.3–7.2 for the others, and only
**~63,800** of its particles reach the wake box against ~88,000 elsewhere. Two
different explanations:

- particles **are** captured but accumulate less per capture, or
- **fewer particles are captured at all** — something diverts them around the tool.

The trajectories distinguish these directly. The lower wake count already points at
the second.

</div>

<span class="tiny">⚠️ Until this is settled, D's number should not be presented as a clean comparison with the other families.</span>

---

## ⏳ Placeholder — the remaining wake cross-sections

<div class="box">

**GENERATED, not yet shown:** `figs_wake/figWB_{A-Flats, C-Threads, D-Concavity,
cohort}_yz_field.png`

Same construction as the B-Flutes panel. Worth adding once the family story is
settled, so the audience can compare cross-sections side by side rather than take
the table on trust.

</div>

---

## What is defensible today

<div class="ok">

✅ The **implementation** is sound: two independent damage laws agree on all 45
cases, the throughput gate passes on both platforms, and the σ_eq convention error
was found and corrected against the solver's own stress.

✅ **Stage 1's η result stands on its own** — tensile conditions sharply localised to
the advancing-side wake, with no fitted parameters. This is the result that agrees
with the flow/refill analysis.

</div>

<div class="warn">

⚠️ **Phase 3's 10 mm travel was too short** — the bundle had not cleared the tool.
At 30 mm with the wake box the family ranking is clean (B-Flutes highest on every
statistic) and PinShapes' flank asymmetry **balances out** (0/45 → 12/20). A
**cohort-specific** retreating-side excess survives (0/25, ratio 0.641) and is
still unexplained.

⚠️ **No experimental defect data exists** — no macrograph, CT scan or sectioned weld,
in this work or in the published FOM validation (which covers forces, torque and
thermocouples). Both analyses currently produce a **defect-susceptibility ranking**, not a
validated defect prediction.

</div>

---

## Next steps

| | |
|---|---|
| **1** | Explain the **cohort-only** retreating-side excess (0/25, ratio 0.641) — PinShapes no longer shows it |
| **2** | Report **wake-box** mean/median/p99 + frac_failed as the headline, with `z_top` and `bypass_frac` as diagnostics |
| **3** | Apply the $F_y > F_x$ force criterion to all cases — **an external check against experimentally validated quantities, at no experimental cost** |
| **4** | Deposit the per-particle damage indicator to a grid → a continuous defect-susceptibility field |
| **5** | Then, and only then, regress against pin geometry for the ROM |

<span class="tiny">Open items are tracked in <code>OPEN_QUESTIONS.md</code> — 15 live, each with what would settle it.</span>

---

## References

<div class="small">

**Damage laws**

1. J. R. Rice & D. M. Tracey, *On the ductile enlargement of voids in triaxial stress
   fields*, **J. Mech. Phys. Solids 17**(3), 201–217 (1969). — the 0.283 amplitude.
2. M. G. Cockcroft & D. J. Latham, *Ductility and the workability of metals*,
   **J. Inst. Metals 96**, 33–39 (1968).

**FSW precedent and solver**

3. X. He, P. R. Dawson & D. E. Boyce (2008) — Lee–Dawson damage applied to FSW; the
   published precedent for pathline-integrated damage in this process.
4. L. Venghaus, **Finite Elements in Analysis & Design 224**, 103986 (2023),
   Publication 1, eq. (5) — the Norton–Hoff shear-measure definition that fixes the
   √3 convention. Also the thesis, §4 p. 66, for the solid-state temperature range.

**Open for future work:** GTN — Gurson (1977), Tvergaard & Needleman (1984).

</div>

<span class="tiny">Full derivations, line-by-line readings and the 64-case convention check are in <code>PRIMER_fsw_physics_for_me.md</code> §8 and <code>OPEN_QUESTIONS.md</code> (O25).</span>

---

<!-- _class: lead -->

## Summary

**Implemented:** two metal-forming damage laws integrated along pathlines, inside
the production GPU tracker, at **under 2 %** cost, on **45 FOM cases**.

**Verified:** the two laws agree with each other (ρ ≥ 0.98); the rheology was
corrected by a factor 1.816 found by comparison with the solver's own stress.

**Resolved:** 10 mm was too short. At 30 mm the bundle clears the tool
(mean x −0.52 → +18.95 mm) and the wake box gives the cleanest family separation
yet — **B-Flutes highest on mean, median, p99 and frac_failed**, matching its
deepest lobes.

**Open:** a cohort-specific retreating-side excess, and no experimental defect data
to validate against.
