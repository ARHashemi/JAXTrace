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

# Independent Defect Indicators in FSW

## Testing whether the damage ranking means what we think it means

<div class="small">

16 cases · 288,000 particles each · 4,092,780 wake particles pooled
LUMI, 2026-10-18 · all jobs exit 0:0

</div>

---

## Why this work exists

<div class="warn">

The previous result rests on **two damage laws agreeing at ρ = +0.98…+0.998**.

That agreement was never corroboration. Rice–Tracey and Cockcroft–Latham share
$\dot\varepsilon$, share $\sigma_\text{eq}$, and are evaluated on the **same
pathlines**. Two indicators built from the same inputs *must* agree.

</div>

So we added indicators that share **fewer inputs**, and asked what survives.

<div class="box">

**The question for this meeting:** which indicator shows a meaningful, realistic
defect — and do they agree?

</div>

---

## What was added

| indicator | status | shares inputs with the damage laws? |
|---|---|---|
| $\ln(\Phi/\Phi_0)$ Rice–Tracey | pre-existing | — (the reference) |
| $C$ Cockcroft–Latham | pre-existing | ✅ heavily ($\dot\varepsilon$, $\sigma_\text{eq}$) |
| $t_\text{res}$ residence time | **new** | ❌ **independent** — it is a *time* |
| $w=\int p/\bar\sigma\,dt$ Piwnik–Plata | **new** | ⚠️ partly ($\bar\sigma$) |
| $A_{\dot\varepsilon}=\int\dot\varepsilon\,dt$ | derived | ⚠️ the **control** |

<div class="box">

**Two new accumulators**, not four. Cost: a few FLOPs per step and **zero extra
element searches** — both reuse samples the damage path already takes.

$w$ needs no new field at all: $p/\bar\sigma = -\eta$ **exactly**.

</div>

---

## Result 1 — the damage block is degenerate

<div class="small">

Rank correlation, pooled over **4,092,780** wake-box particles, 16 cases:

</div>

|            | ln_phi |     C  | A_edot | t_res  | w_bond |
|------------|--------|--------|--------|--------|--------|
| **ln_phi** |  1.000 |  0.996 |  1.000 |  0.802 | −0.085 |
| **C**      |  0.996 |  1.000 |  0.996 |  0.802 | −0.123 |
| **t_res**  |  0.802 |  0.802 |  0.802 |  1.000 |  0.019 |
| **w_bond** | −0.085 | −0.123 | −0.085 |  0.019 |  1.000 |

<div class="warn">

$\ln\Phi$, $C$ and $A_{\dot\varepsilon}$ are **the same quantity to three
decimals**. Across the 16 cases they span only **1.17–1.19×**.

Reporting them as three indicators **overstates the evidence**.

</div>

---

## Result 2 — ⚠️ the ranking depends on which indicator you believe

<div class="small">

$w_\text{bond}$ spans **7.03×** across the family — the only indicator that
discriminates:

</div>

| family | median $w$ | range |
|---|---|---|
| A-Flats | 0.0111 | 0.0072 – 0.0379 |
| C-Threads | 0.0241 | 0.0236 – 0.0250 |
| B-Flutes | 0.0293 | 0.0216 – 0.0345 |
| **D-Concavity** | **0.0464** | 0.0400 – 0.0508 |

<div class="warn">

**This reverses our headline.** The report has **B-Flutes highest on every wake
statistic** — on the damage laws, still true.

On the **bonding criterion, D-Concavity is highest** by a clear margin, and the
two measures **disagree** (ρ = −0.085).

</div>

---

## Why that is not a contradiction

<div class="box">

The two measures answer **different questions**:

- $\ln\Phi$, $C$ — *how hard was this material worked?*
- $w$ — *did it spend its time under the compression that makes a bond?*

A particle can be heavily worked and poorly bonded, or lightly worked and well
bonded. ρ = −0.085 says these are **nearly orthogonal** in our flow.

</div>

<div class="warn">

Which one predicts a real void **cannot be settled from simulation alone**.
It needs one experimental cross-section.

</div>

---

## Result 3 — O56: the density floor, measured for the first time

<div class="box">

With detachment **off**, the velocity field is divergence-free and
$\det\mathbf{F}\equiv 1$ — so particle density **must** be uniform.
A hole is *impossible by construction*.

</div>

Measured wake-box density CV: **median 0.273**, max **0.443** (B2).

<div class="warn">

That is **seeding and sampling noise, not voids** — and it is the **floor** any
future detachment result has to clear. A deficit must exceed CV ≈ 0.27 before it
means anything.

B2's 0.443 is the outlier to understand first.

</div>

---

## What the cross-sections show

<div class="small">

Per case: Group (seeding slab), then each indicator as a k-NN median field, on
the same yz wake cross-section. Viewed from the origin looking **downstream**
(+y on the left), matching every existing report figure.

</div>

<div class="box">

**Two candidate real defects**, each visible in several *independent* panels at once:

- **A1** — y ≈ +3, z ≈ −4.5 mm: Group gap **+** $\ln\Phi$/$C$ bright spot **+**
  $t_\text{res}$ spike **+** $w$ sign reversal
- **B2** — y ≈ +3…+4, z ≈ −4…−5 mm: Group gap, and B2 is the density-CV outlier

Co-located across indicators that share no inputs ⇒ **not an estimator artefact**.

</div>

---

## ⚠️ What we cannot yet claim

<div class="warn">

**O55 is provisionally, not conclusively, answered.**

ρ(ln_phi, $A_{\dot\varepsilon}$) = +1.000 looks decisive, but
$A_{\dot\varepsilon}$ was **derived** as $\ln\Phi/0.849$ — exact only if
$\exp(1.5\eta)\equiv 1$.

Measured: $\exp(1.5\eta)$ varies **0.758 … 1.020**, a **35 % spread**.

So +1.000 is **partly true by construction**. $A_{\dot\varepsilon}$ now has its
own accumulator; the number needs a re-run to be quoted as a measurement.

</div>

<div class="box">

The *direction* is almost certainly right — Wang et al. (2020) predict it, and
$\eta$ is small. But we should not present +1.000 as measured.

</div>

---

## Where this leaves the method

<div class="ok">

**What held up.** The pipeline is sound: 16/16 cases exit 0:0, 100 %
accumulation, every archive passes an internal-consistency check. $t_\text{res}$
separates entrained from non-entrained material by **93×** in median damage.

</div>

<div class="warn">

**What did not.** "Two damage laws agree" was never evidence. Adding one
genuinely independent indicator was enough to show the damage block carries
**one effective degree of freedom**, and to produce a **different family
ranking**.

</div>

<div class="box">

That is the exercise working as intended — but the deck's previous conclusion
needs qualifying before it is presented as settled.

</div>

---

## Next steps

| # | step | why |
|---|---|---|
| 1 | **One experimental cross-section** | ⚠️ outranks everything below; without it every ranking is unvalidated |
| 2 | Re-run with the real $A_{\dot\varepsilon}$ accumulator | makes the O55 test honest |
| 3 | $\Delta p$ (A.4) — the one unrun indicator | independent, cheap, needs a pressure-field pass |
| 4 | Understand B2's density CV = 0.443 | the outlier against the 0.273 floor |
| 5 | Detachment criterion (Part B) | the route to a void that is a **mass deficit**, not a damage proxy |

<div class="tiny">

Detail: `NEXT_STEPS_detachment_criterion.md` §A.5a (tally), §A.5b (results), §B (detachment).

</div>

---

<!-- _class: lead -->

# Backup

---

## Why the indicators need different colour treatments

<div class="small">

Measured on the D2 wake box (268,764 particles):

</div>

| indicator | min | median | max | negative | exactly zero |
|---|---|---|---|---|---|
| ln_phi | +0.0725 | +0.256 | +366.9 | 0.0 % | 0.0 % |
| C | +0.0634 | +0.240 | +342.0 | 0.0 % | 0.0 % |
| t_res | 0.0 | 0.0 | +2.169 | 0.0 % | **65.6 %** |
| w_bond | −0.0378 | +0.0457 | +0.410 | **27.1 %** | 0.0 % |

<div class="warn">

$t_\text{res}$: a log scale is **impossible** — two-thirds are exactly zero.
$w$: **signed**, and the sign is the physics. A sequential map would destroy it.

</div>

---

## Why $t_\text{res}$ gets two panels

<div class="box">

A k-NN **median** of $t_\text{res}$ is zero wherever most neighbours are zero —
so that panel reads as **coverage**, not intensity.

The **entrained fraction** — the share of neighbours with $t_\text{res}>0$ — is
the quantity it is really trying to show, and it maps the **stir-zone boundary**
directly.

</div>

Both are drawn, adjacent: the median alone misleads, the fraction alone does not
say *how long*.

<div class="tiny">

Median ln(Φ/Φ₀) for particles that never entered the layer: 0.150. For those that
did: 13.96 — a **93×** separation. $t_\text{res}$ measures what it should.

</div>
