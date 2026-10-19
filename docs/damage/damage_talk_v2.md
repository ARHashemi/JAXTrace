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

<!-- _class: lead -->

# Defect Indicators in FSW

## From contact pressure to pathline integrals — what each one measures, and whether they agree

<div class="small">

16 cases · 288,000 particles each · AA6063 · 600 rpm · 600 mm/min
LUMI, 2026-10-19

</div>

---

<!-- _class: lead -->

# Part 1 — Contact pressure

## The one indicator that needs no particles

---

## Why start here

<div class="box">

Every other indicator we have is a **pathline integral**: it depends on where we
seed particles, how we step them, and how the element search behaves.

A fault anywhere in that machinery moves **all of them together**, and no amount
of agreement between them would reveal it.

</div>

$\Delta p$ reads the **pressure field at the mesh nodes**. No seeding, no
tracking, no element search.

<div class="ok">

It is therefore the only **external check** on the particle apparatus — and the
only place where a published criterion with a threshold already exists.

</div>

---

## The quantity: Shi et al. (2022)

$$\boxed{\;B \;=\; P_{\max}(\theta = 0^\circ,\ \text{front}) \;-\; P_{\min}(\theta = 180^\circ,\ \text{rear})\;}$$

<div class="small">

evaluated in an annulus around the pin, $r \in [r_\text{pin},\ 1.6\,r_\text{pin}]$,
within the pin's z-extent. $\theta$ measured from the travel direction $+x$.

</div>

| | |
|---|---|
| **sound weld** | pressure roughly uniform around the tool |
| **voided weld** | high in front, **collapsed** behind |
| **criterion** | voids form when $B > 15$ MPa |

<div class="warn">

⚠️ **Not advancing-minus-retreating.** $B$ is **front-to-rear**, along the
travel direction. We measured both: on D2 they are **+16.8** and **−11.9 MPa** —
different magnitude *and* opposite sign.

</div>

---

## ⚠️ Which plane it lives in

<div class="box">

Every other figure in this work uses a **yz** cross-section — the weld
cross-section, where the pathline indicators are compared.

$B$ **cannot** be drawn there. It is a front-to-rear difference along $+x$, and
**yz is a plane of constant $x$**.

</div>

| plane | |
|---|---|
| **xy at pin mid-depth** | ✅ contains $\theta=0$ and $\theta=180$ simultaneously |
| xz through $y=0$ | ✅ same contrast, against depth |
| **yz** | ❌ perpendicular to $B$ — would show the advancing/retreating contrast instead |

---

## Result — $B$ across the 16 cases

| case | median $B$ [MPa] | % snapshots > 15 |
|---|---|---|
| **D2** | **16.80** | **52 %** |
| **A2** | **12.82** | **36 %** |
| A3 | 6.43 | 2 % |
| D3, B4, D1, B2, D4 | 3.58 – 4.67 | 0 % |
| C1, A1, C2, A4, B1, C4, B3, C3 | 1.48 – 1.77 | 0 % |

<div class="ok">

An **11× spread** across the family — the widest of any indicator we have.

</div>

<div class="tiny">

Converged: the 16-case ranking has rank correlation +0.94 against the full set using only the first 20 of 166 snapshots, with D2 top at every subsample.

</div>

---

## Comparison — Chen et al. (2023)

<div class="small">

AA6061-T6 at **600 rpm — our rotation speed** — CFD with non-uniform contact
pressure, **validated against measured cross-sections**.

</div>

| | front | rear | drop |
|---|---|---|---|
| Chen, **sound** 600-20 | 30 | 30 | 0 |
| Chen, **voided** 600-180 | 50 | 15 | **35** |
| our high-$B$ group | 7.6 | **−0.7** | 16.8 |
| our low-$B$ group | 3.0 | +1.2 | 1.8 |

<div class="warn">

❌ **The numbers must not be compared.** Chen reports frictional **shear stress**
at the tool interface; $B$ is a **pressure** difference in the material.
Only the **pattern** transfers.

</div>

---

## The pattern does transfer — on 14 of 16 cases

<div class="box">

Chen's signature is not "a large difference" — it is the **rear collapsing**.

✅ Four of our cases have a **negative (tensile) rear pressure**: A2, A3, B4, D2.
Their median $B$ is **9.63 MPa** against **1.67** for the other twelve.

</div>

<div class="warn">

⚠️ **But 3 of 4, not 4 of 4.** Top four by $B$: D2, A2, A3, **D3**. Negative-rear
four: D2, A2, A3, **B4**.

- **B4** — rear collapses, but $B$ stays low because its *front* is also low
- **D3** — high $B$ from a *high front* with a **positive** rear: no collapse

So "large $B$" and "rear collapse" are **distinguishable statements**.

</div>

---

## ⚠️ And where our process window actually sits

$v_\text{adv} = 600$ mm/min at $\omega = 600$ rpm $\Rightarrow\ \gamma = v/\omega = 1.00$ mm/rev

| $\gamma$ [mm/rev] | case |
|---|---|
| 0.033 | Chen 600-20, **sound** |
| 0.300 | Chen 600-180, **voided** |
| **1.000** | ⚠️ **every one of our 16 cases** |

<div class="warn">

We run **3.3× beyond Chen's voided condition**. If his process-window reasoning
transfers, **all 16 cases are void-prone**, and the question is not *which tool
makes a void* but **which makes the worst one**.

</div>

<div class="tiny">

Consistent with B: all 16 show a positive front-to-rear difference. The ranking is a severity ordering inside an aggressive window, not a sound/defective split.

</div>

---

<!-- _class: lead -->

# Part 2 — Pathline indicators

## What each one integrates along a particle's own trajectory

---

## The four, and what they share

| indicator | integrand | shares inputs with the damage laws? |
|---|---|---|
| $\ln(\Phi/\Phi_0)$ Rice–Tracey | $0.849\,\dot\varepsilon\,e^{1.5\eta}$ | — (the reference) |
| $C$ Cockcroft–Latham | $\max(\sigma_1,0)/\sigma_\text{eq}\cdot\dot\varepsilon$ | ✅ heavily |
| $A_{\dot\varepsilon}$ | $\dot\varepsilon$ | ⚠️ the **control** |
| $t_\text{res}$ residence time | $H(\dot\varepsilon - \dot\varepsilon_\text{thr})$ | ❌ a **time** — no stress |
| $w$ Piwnik–Plata | $p/\bar\sigma = -\eta$ | ⚠️ partly ($\bar\sigma$) |

<div class="box">

Cost: a few FLOPs per step and **zero extra element searches** — all reuse
samples the damage path already takes. $w$ needs no new field at all.

</div>

---

## How each is drawn, and why differently

<div class="small">

Measured on the D2 wake box (268,764 particles):

</div>

| indicator | negative | exactly zero | treatment |
|---|---|---|---|
| ln_phi, C | 0 % | 0 % | log, sequential |
| $t_\text{res}$ | 0 % | **65.6 %** | linear; zeros as their own category |
| $w$ | **27.1 %** | 0 % | diverging about zero |

<div class="warn">

$t_\text{res}$: a log scale is **impossible** — two-thirds are exactly zero, and
a sequential ramp would render "never sheared" and "barely sheared" alike.
$w$: **signed**, and the sign is the physics.

</div>

---

## Per-case cross-sections

<div class="box">

One figure per case: **Group** (seeding slab), then each indicator as a k-NN
median field, on the same yz wake cross-section.

Viewed from the origin looking **downstream** (+y on the left), matching every
existing report figure.

</div>

<div class="small">

⚠️ $t_\text{res}$ also gets an **entrained-fraction** panel: a k-NN *median* of
$t_\text{res}$ is zero wherever most neighbours are zero, so that panel reads as
*coverage*. The fraction is what it is really trying to show, and it maps the
**stir-zone boundary** directly.

</div>

<div class="tiny">

figs_cases/fig_case_&lt;CASE&gt;.png — 16 figures, plus a backstage raw-scatter set.

</div>

---

## Two candidate real defects

<div class="box">

Features appearing in **several independent panels at once**, which rules out an
estimator artefact:

- **A1** — y ≈ +3, z ≈ −4.5 mm: Group gap **+** $\ln\Phi$/$C$ bright spot **+**
  $t_\text{res}$ spike **+** $w$ sign reversal
- **B2** — y ≈ +3…+4, z ≈ −4…−5 mm: Group gap, and B2 is the **density-CV
  outlier** (0.443 against a 0.273 family median)

</div>

<div class="warn">

⚠️ With detachment **off**, the field is divergence-free and
$\det\mathbf{F}\equiv 1$ — a hole is *impossible by construction*. CV 0.273 is
the **noise floor**, and any future void claim must clear it.

</div>

---

<!-- _class: lead -->

# Part 3 — Do they agree?

---

## ⚠️ The conclusion: three families, three rankings

<div class="small">

Rank correlation, pooled over **4,092,780** wake-box particles:

</div>

|            | ln_phi |     C  | A_edot | t_res  | w_bond |  $B$   |
|------------|--------|--------|--------|--------|--------|--------|
| **ln_phi** |  1.000 |  0.996 |  1.000 |  0.802 | −0.085 | −0.174 |
| **t_res**  |  0.802 |  0.802 |  0.802 |  1.000 |  0.019 | — |
| **w_bond** | −0.085 | −0.123 | −0.085 |  0.019 |  1.000 | +0.121 |
| **$B$**    | −0.174 | −0.047 | −0.174 | — | +0.121 |  1.000 |

<div class="warn">

- **damage laws** → B-Flutes highest · **bonding $w$** → D-Concavity · **$B$** → D2 and A2
- $\ln\Phi$, $C$, $A_{\dot\varepsilon}$ span only **1.17–1.19×**: one effective degree of freedom
- $\rho(B,\ \text{density CV}) = -0.518$ — **the wrong sign** for a void-predicting $B$

</div>

---

## What that means

<div class="ok">

**The pathline family broadly agrees**, as the cross-sections show by eye —
they locate the same stir zone and the same two candidate defects.

</div>

<div class="warn">

**But they are not independent evidence.** They share $\dot\varepsilon$,
$\sigma_\text{eq}$ and the same 288,000 trajectories. $B$, which shares none of
it, ranks the cases differently.

</div>

<div class="box">

⚠️ **No ranking here is independently supported**, and nothing in simulation can
adjudicate. Our own FOM is validated on **temperature and force**, not defects —
and its experimental data traces back to our own group.

</div>

---

## Next steps

| # | step |
|---|---|
| **0** | ⚠️ **One experimental cross-section** — AA6063, ≥2 pin families, void area measured (macrograph or CT) |
| **0b** | ✅ **Free:** extend the Chen mechanism test to void *location* using the cross-sections |
| 1 | Re-run with the real $A_{\dot\varepsilon}$ accumulator — the O55 control is currently partly circular |
| 2 | Understand B2's density CV = 0.443 against the 0.273 floor |
| 3 | Detachment criterion — the only route to a void that is a **mass deficit**, not a proxy |

<div class="tiny">

Detail: NEXT_STEPS_detachment_criterion.md §A.0b (two senses of independent), §A.4b (B results), §A.4c (validation references), §B (detachment).

</div>
