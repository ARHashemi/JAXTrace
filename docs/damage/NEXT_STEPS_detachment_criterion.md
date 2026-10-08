# Next steps — independent indicators and a detachment-based void criterion

**Status: design document. No code written.** Written 2026-10-08, while the D-family
issues are being resolved separately. Implementation deliberately deferred.

This document covers two things:

1. **Part A** — a set of additional defect indicators that are *independent* of the
   current damage integral, so that agreement between them means something.
2. **Part B** — the detachment idea: a Niyama-style dimensionless criterion that can
   produce a **genuine** density deficit rather than a damage proxy. This is the part
   with novelty potential, and the part that needs the most care.

Everything here builds on `FUTURE_MODELS_EXPLAINED.md` (the model survey and its
ranked table §6) and `DAMAGE_REPORT.md` (the current results). References are keyed
to the reference list in `FUTURE_MODELS_EXPLAINED.md` §7 by number, e.g. **[10]**.

⚠️ **Source provenance.** Of the papers cited here only **He et al. (2008)** has been
read in full. Everything else is cited from abstracts, indexing records and secondary
discussion, as recorded in `FUTURE_MODELS_EXPLAINED.md` §7's status note. Nothing in
this document should be quoted as a verified finding of a paper we have not read.
Where a number matters to a decision below, it is flagged as needing verification.

---

## Part A — Independent indicators to add alongside the damage laws

### A.0 Why this is the first priority

The two damage laws currently in the pipeline agree at $\rho = +0.982 \ldots +0.998$
(`DAMAGE_REPORT.md`). That looked like corroboration and is not: Rice–Tracey and
Cockcroft–Latham share $\dot\varepsilon$, share $\sigma_\text{eq}$, and are evaluated
on the same pathlines. Two indicators built from the same inputs *must* agree. Recorded
as **O45**.

✅ **What breaks the circularity is an indicator that shares no inputs.** Two of the
four below qualify completely.

### A.1 Strain-rate-only ranking — do this first

$$\boxed{\;A_{\dot\varepsilon} \;=\; \int_\text{pathline} \dot\varepsilon \; dt \;}$$

Pure accumulated equivalent strain along the pathline. No triaxiality, no stress, no
material parameter, no threshold.

**Why it is the single most informative thing to add.** Wang et al. (2020) **[10]**
report that solid-state bonding depends **mainly on creep strain rate** and only
**weakly on stress triaxiality**. Our Rice–Tracey law is

$$\frac{d\ln\Phi}{dt} \;=\; 0.849 \,\dot\varepsilon \, \exp(1.5\eta)$$

so if Wang is right for our conditions, the $\exp(1.5\eta)$ factor is nearly constant
and the law reduces to a scaled strain integral. We can test that directly:

| outcome | what it means |
|---|---|
| $A_{\dot\varepsilon}$ reproduces the B-Flutes-highest ranking | the triaxiality factor contributes **nothing**; the damage apparatus is an expensive strain meter |
| the rankings **differ** | $\exp(1.5\eta)$ is doing real work, and we can say exactly how much |

⚠️ **This is a result either way**, which is rare and cheap. It is item **#2** in the
ranked table and the one I would run first.

**Expected to be near-degenerate, and that is the point.** Measured $\eta$ is small
and negative (median ≈ −0.05 under the shoulder, `DAMAGE_REPORT.md`), so
$\exp(1.5\eta) \approx 0.93$ — close to constant. The prediction is therefore that
$A_{\dot\varepsilon}$ and $\ln(\Phi/\Phi_0)$ will rank the 45 cases almost identically.
**Confirming that is a finding**, not a null result: it says our two-law apparatus has
one effective degree of freedom.

- **Independent of the damage integral?** ✅ Yes — it *is* the integral with the stress
  factor removed, which is precisely the controlled comparison.
- **Cost:** trivial. $\dot\varepsilon$ is already accumulated.
- **Calibration:** none.

### A.2 Residence time in the shear layer

$$\boxed{\;t_\text{res} \;=\; \int_\text{pathline} H\!\left(\dot\varepsilon - \dot\varepsilon_\text{thr}\right) dt\;}$$

with $H$ the Heaviside step and $\dot\varepsilon_\text{thr}$ a strain-rate level
defining "in the shear layer".

**What it separates.** The damage integral conflates *how hard* a particle was worked
with *how long*. A particle can reach the same $\int\dot\varepsilon\,dt$ by brief
intense shear or by long gentle shear, and for bonding those are not equivalent —
diffusion and recrystallisation are time-dependent. Residence time isolates the
duration axis.

- **Independent?** ✅ Yes — it is a *time*, not a work measure. Its only link to the
  damage laws is the threshold used to define the layer.
- **Cost:** low — one more accumulator.
- **Calibration:** none, but $\dot\varepsilon_\text{thr}$ needs a sensitivity sweep;
  report results across a range rather than at one value.
- ⚠️ **Sensitivity is the risk here.** Pick $\dot\varepsilon_\text{thr}$ badly and
  $t_\text{res}$ either saturates (everything is "in the layer") or vanishes. Sweep
  it over at least a decade and report the ranking's stability, not one number.

### A.3 Piwnik–Plata bonding criterion

$$\boxed{\;w \;=\; \int_\text{pathline} \frac{p}{\bar\sigma}\,dt \;,\qquad \text{bond if } w \ge w_\text{lim}\;}$$

with $p$ the contact/hydrostatic pressure and $\bar\sigma$ the flow stress. In our sign
convention $p = -\sigma_m$, both already computed in `jaxtrace/damage/fields.py`.

**Why it is worth having** — it asks the *joining* question rather than the *damage*
question. FSW is a bonding process; a void is a bonding failure. Applied to FSW by
Buffa, Pellegrino & Fratini (2014) **[8]** and compared across solid-state processes by
Fratini et al. (2016) **[9]**; $w_\text{lim}$ calibration procedure in D'Urso et al.
(2011) **[7]**.

- **Independent?** ⚠️ **Only partly** — it shares $\bar\sigma$ with our laws. Treat its
  agreement with them as weak evidence.
- **Cost:** low — one accumulator from fields already loaded.
- **Calibration:** $w_\text{lim}$ is alloy- and temperature-dependent. ⚠️ **We do not
  have it for Al6063.** Use it as a *relative* ranking across the 45 cases, not as a
  pass/fail test, unless the rolling-test calibration is actually performed.

### A.4 Contact-pressure differential

$$\boxed{\;\Delta p \;=\; p_\text{advancing} - p_\text{retreating}\;}$$

evaluated around the pin. Shi et al. (2022) **[15]** report a threshold of
**Δp < 15 MPa** for void formation.

- **Independent?** ✅ Yes — a pressure-field diagnostic, no pathline integration at all.
- **Cost:** low.
- **Calibration:** ⚠️ the 15 MPa value is alloy- and geometry-specific, and we have not
  read the paper. Verify before quoting it. Use the *sign and trend* across cases first.
- ⚠️ **Our pressure magnitudes need care.** `DAMAGE_REPORT.md` records measured
  −6 … −7.5 MPa ahead of the tool and +0.4 … +2.2 MPa behind, with the sign pattern
  holding in 9/10 cases. That is the right order for a 15 MPa threshold comparison,
  but the earlier "+1.70/−1.35 MPa" figures were **not reproducible** and were
  retracted — do not reuse them.

### A.5 What to report

A single table, 45 cases × 6 indicators, plus the rank-correlation matrix between
indicators:

| indicator | shares inputs with damage laws? |
|---|---|
| $\ln(\Phi/\Phi_0)$ Rice–Tracey | — (reference) |
| $C$ Cockcroft–Latham | ✅ heavily |
| $A_{\dot\varepsilon}$ | ⚠️ by construction (the control) |
| $t_\text{res}$ | ❌ independent |
| $w$ Piwnik–Plata | ⚠️ partly ($\bar\sigma$) |
| $\Delta p$ | ❌ independent |

✅ **The useful output is the off-diagonal structure.** If $t_\text{res}$ and $\Delta p$
— the two genuinely independent ones — agree with the damage ranking, that is real
corroboration. If they disagree, we have learned that the damage ranking reflects
work history rather than defect susceptibility, which is more valuable than another
agreeing number.

---

## Part B — The detachment criterion

### B.0 The idea in one paragraph

Particles currently follow the solenoidal velocity field for all time. Introduce a
local criterion under which a particle **stops being entrained by the rotating flow**
and instead translates with the far-field advance velocity $v_\text{adv}$. Material
that detaches early leaves the wake unfilled; the particle ensemble then exhibits a
**real** density deficit. The criterion is cast as a dimensionless group so the
threshold is a pure number rather than a per-alloy constant.

### B.1 ⚠️ First, a correction to the framing

The idea was originally phrased as particles ceasing to follow *melted* flow, with the
criterion acting as a model for *solidification*.

**That framing cannot be used, and it contradicts our own report.**
`DAMAGE_REPORT.md` §1 establishes that FSW peak temperature reaches only **80–95 % of
melting**, that nothing liquefies, and uses exactly that fact to rule out bubbles and
cavitation (Re ≈ 10⁻⁷–10⁻⁴; cavitation number ≈ 7×10⁴, wrong by ~11 orders *and* wrong
sign). Al6063 has no liquid phase anywhere in our temperature field. A criterion
presented as solidification would be rejected immediately by anyone who knows FSW, and
would contradict §1 of our own document.

✅ **The mechanism survives intact; only the vocabulary changes.** What physically
happens is not a phase change but a **flow / no-flow transition**: material stops
co-rotating when it can no longer be sheared fast enough to keep up with the tool.
That is a competition between two *rates* — which is exactly Niyama's structure, so
the analogy we wanted is still there, and on firmer ground.

| ❌ do not write | ✅ write instead |
|---|---|
| melted / molten / liquid | shear-dominated, co-rotating, entrained |
| solidification | **detachment**, loss of entrainment |
| freezes / solidifies | **detaches**, drops out of the rotating zone |
| re-melts | **re-entrains** |

⚠️ **This is not just terminology.** The thermal softening that drives detachment is
*flow-stress* softening, continuous in $T$, with no latent heat and no phase front.
Any formula must be continuous in $T$, not a step at a melting point.

### B.2 Why a detachment rule can produce a real void

This is the strongest part of the argument and worth stating precisely.

**The incompressibility objection to our current voids is correct.** For a
divergence-free field $\nabla\!\cdot\!\mathbf{u}=0$ and particles that always follow it,
the Jacobian of the flow map is identically 1:

$$\frac{d}{dt}\big(\det \mathbf{F}\big) = (\nabla\!\cdot\!\mathbf{u})\,\det\mathbf{F} = 0
\quad\Longrightarrow\quad \det\mathbf{F} \equiv 1$$

so an initially uniform particle density **stays uniform for all time**. A hole is
*impossible by construction*. Any apparent hole in our current figures is a seeding,
sampling or estimator artefact — which is exactly the D-family confound
(`dfamily-seedbox-bug`, `dfamily-root-cause-cuboid-pitch`) and why
`make_density_vs_damage.py` exists.

✅ **A detachment rule breaks this legitimately.** The ensemble velocity becomes

$$\mathbf{v}_p \;=\; \begin{cases}
\mathbf{u}(\mathbf{x}_p,t) & \text{entrained} \\[2pt]
v_\text{adv}\,\hat{\mathbf{e}}_x & \text{detached}
\end{cases}$$

This **piecewise** field is *not* divergence-free across the detachment surface — the
normal velocity jumps — so $\det\mathbf{F} \neq 1$ and a genuine density deficit can
form. The void stops being a damage proxy and becomes a **mass-bookkeeping outcome**,
which is precisely how `DAMAGE_REPORT.md` §1 defines a void: *material that failed to
refill the cavity behind the advancing pin*.

⚠️ **The physical content is in the jump, so the jump must be justified.** A reviewer
will ask what carries the momentum deficit. The honest answer is that detached material
is supported by the surrounding solid rather than by the shear layer — which is why
this is a *kinematic* post-process model, not a momentum-conserving one. State that
limitation explicitly; do not claim the piecewise field solves any balance law.

### B.3 The dimensionless groups

Niyama compares a **driving** gradient against a **resisting** rate. For FSW the
natural analogue compares the rate at which the cavity opens against the rate at which
material can be delivered into it.

**Group 1 — kinematic delivery ratio.** The pin advances at $v_\text{adv}$ while
material is swept around it at the pin periphery speed $\omega r_\text{pin}$:

$$\boxed{\;\Pi_1 \;=\; \frac{\omega\, r_\text{pin}}{v_\text{adv}}\;}$$

✅ This is already the standard FSW process group (the "weld pitch" $\omega/v$ up to
geometry), so we are on well-trodden ground. Large $\Pi_1$ = many revolutions per unit
advance = good filling. Small $\Pi_1$ = the cavity opens faster than it is fed.

**Group 2 — thermal softening margin.** Detachment is governed by how far the local
flow stress sits from the level at which material still follows the tool. With the
Norton–Hoff law already implemented (`jaxtrace/damage/rheology.py`):

$$\sigma_\text{eq}(T,\dot\varepsilon) \;=\; \sqrt{3}\,\text{VISCO}(T)\,\big(\sqrt{3}\,\dot\varepsilon\big)^{\text{EXPVI}(T)}$$

define

$$\boxed{\;\Pi_2 \;=\; \frac{\sigma_\text{eq}(T,\dot\varepsilon)}{\sigma_\text{ref}(T)}\;}$$

⚠️ **$\sigma_\text{ref}$ is the open modelling choice and the weakest link.** Options,
in order of how defensible they are:

| choice of $\sigma_\text{ref}$ | pro | con |
|---|---|---|
| the tool/workpiece **shear traction** at the pin surface | the actual competing stress; no new constant | needs a contact model we do not have |
| $\sigma_\text{eq}$ evaluated at a **fixed reference** $(T_\text{ref},\dot\varepsilon_\text{ref})$ | computable from tables we already have | $T_\text{ref}$ is a free parameter — exactly the per-alloy threshold we were trying to kill |
| the **homologous temperature** $T/T_m$ directly | zero new constants, standard in hot working | less mechanistic; ignores rate |

✅ **Recommendation: start with $T/T_m$**, i.e. $\Pi_2 = T/T_m$ with $T_m$ the Al6063
solidus. It introduces **no fitted constant**, it is the standard non-dimensionalisation
in hot deformation, and our field already spans 0.80–0.95 of it. Promote to a
stress-based $\Pi_2$ only if the homologous form proves too blunt.

⚠️ **Do not let $\Pi_2$ smuggle the threshold back in.** The entire point of the
dimensionless construction is to avoid a per-alloy constant. If the final criterion
needs a fitted $T_\text{ref}$, we have reproduced the $\Phi_0$ / $C_\text{crit}$
problem in new notation and gained nothing.

**The criterion.** Detachment when the delivery capacity, modulated by softening,
falls below a critical pure number:

$$\boxed{\;\Pi_\text{det} \;=\; \Pi_1 \cdot \Pi_2^{\,n} \;<\; \Pi_\text{crit}\;}$$

with $n$ an exponent to be determined and $\Pi_\text{crit}$ a **dimensionless** number
expected to be $O(1)$.

⚠️ **$n$ and $\Pi_\text{crit}$ are two free parameters.** Two, not zero. The claim is
not "no calibration" but "calibration to a *pure number* that should transfer across
alloys and geometries", which is Guo et al. (2015) / Kang et al. (2013) **[13]**'s
actual argument for the dimensionless Niyama. Be precise about this in any write-up:
the per-alloy threshold is replaced by a dimensionless one, not eliminated.

### B.4 The local form — what a particle actually tests

$\Pi_1$ as written is a *process* group, constant per case, so it cannot decide when
an individual particle detaches. The local analogue replaces the global rates with
local ones:

$$\Pi_1^\text{loc}(\mathbf{x}) \;=\; \frac{|\boldsymbol\omega_\text{local}|\, \ell}{|\mathbf{u}| }
\quad\text{or}\quad
\frac{\dot\varepsilon\,\ell}{|\mathbf{u}|}$$

⚠️ **Prefer the $\dot\varepsilon$ form over the vorticity form.** `DAMAGE_REPORT.md`
§1 already establishes that vorticity cannot discriminate good welds from bad — rigid
rotation gives zero stretching and is large even in defect-free welds. Using
$|\boldsymbol\omega|$ would reintroduce exactly the quantity we ruled out. The
stretching part of the velocity gradient is the discriminating one.

$\ell$ is a length scale; the shear-layer thickness is the physically meaningful choice
but must then be measured, not assumed.

### B.5 ⚠️ The re-entrainment problem

A one-way rule is **irreversible**: once detached, a particle never rejoins the
rotating flow. Real material can be picked up again by the next pass of the pin — FSW
is cyclic at $\omega$, and material is repeatedly swept, deposited and re-swept.

**Consequence: a one-way rule will over-predict voids**, and the error grows with
residence time in the wake.

Minimum viable treatment:

$$\text{detached} \;\longrightarrow\; \text{entrained} \quad\text{when}\quad \Pi_\text{det} > \Pi_\text{crit} + \Delta$$

with $\Delta>0$ a hysteresis band preventing chatter at the boundary. This makes the
rule a **two-state model with hysteresis** rather than an absorbing state.

⚠️ **Report both.** Run one-way and hysteretic variants; the gap between them bounds
the error introduced by irreversibility. If the gap is large the model is dominated by
its own switching rule and the result is not trustworthy.

### B.6 Prior art — what is novel and what is not

Be scrupulous here; this is where a novelty claim gets tested.

| already published | reference | implication for us |
|---|---|---|
| tracer advection predicting void, wormhole, flash, joint-line remnant, onion rings | Dialami, Cervera & Chiumenti (2020) **[17]**, 128 cit. | ⚠️ **our method, already published, predicting more defect types than we extract.** Not novel. |
| flow partitioning / "excess material function" for defect formation | Arbegast (2008) **[14]**, 309 cit. | the conceptual ancestor of the rate-competition idea |
| per-revolution cavity filling validated against CT | Ghate et al. (2020) **[18]** | closest to a mass-bookkeeping void model |
| dimensionless criterion removing a per-alloy threshold | Guo et al. (2015), Kang et al. (2013) **[13]** | the *pattern* being borrowed — from casting |
| bonding governed by creep rate, weakly by triaxiality | Wang et al. (2020) **[10]** | motivates A.1 and the rate-based $\Pi_2$ |

⚠️ **Sticking/slipping and shear-layer-thickness literature is the real risk.** There
is an established FSW body of work on whether material sticks to or slips past the
tool, and on shear-layer thickness. A rate-competition detachment criterion may well
already exist there under different terminology. **We have not searched it.** Until we
do, the honest claim is narrow:

> ✅ **Defensible claim:** a *dimensionless, locally-evaluated detachment criterion*
> applied to **tracer particles** to produce a mass-deficit void prediction, with the
> threshold as a pure number.
>
> ❌ **Not defensible:** "a novel void-formation mechanism", "a solidification model
> for FSW", or any claim of priority over the sticking/slipping literature.

**One targeted literature search would settle this** — see §B.8.

### B.7 How to falsify it

A criterion that cannot fail is not worth implementing. Four tests, in order:

1. **Incompressibility sanity check.** With detachment switched **off**, the particle
   density field must be uniform to within estimator noise. If it is not, the deficit
   is an artefact and must be fixed before any detachment result can be believed.
   ⚠️ **This test is mandatory and comes first** — it is the same confound as the
   D-family seed-box and cuboid-pitch bugs.
2. **Monotonicity in $\Pi_1$.** Void fraction must decrease with increasing
   $\omega r_\text{pin}/v_\text{adv}$. Our 45 cases already span the
   $(v_\text{adv},\omega)$ plane, so this is free. If void fraction is non-monotonic in
   the one group every FSW practitioner agrees on, the criterion is wrong.
3. **Threshold universality.** Fit $\Pi_\text{crit}$ on the A/B/C families and predict
   D — or fit on the PinShapes cases and predict the ROM cohort. ⚠️ **If
   $\Pi_\text{crit}$ has to be refitted per family, the dimensionless construction has
   failed** and we are back to per-case thresholds.
4. **Switching-rule dominance.** Vary $n$, $\Delta$ and $\ell$ over plausible ranges.
   If the predicted void pattern changes qualitatively, the model is reporting its own
   parameters rather than the flow.

### B.8 ⚠️ The thing that outranks all of it — unchanged

`FUTURE_MODELS_EXPLAINED.md` §6.3 ranks **obtaining one experimental cross-section**
above every model upgrade, and that is still true, with extra force here. A detachment
criterion **adds** two free parameters ($n$, $\Pi_\text{crit}$, plus $\Delta$ and
$\ell$) to a pipeline that currently has no validation target. Without a macrograph or
CT scan we would be tuning four parameters against nothing.

> **More model complexity buys precision, not accuracy** — the lesson of He et al.'s
> own 0.01 % → 0.017 % result, where the *locations* matched experiment and the
> *values* depended entirely on fitted constants.

**Honest framing for the meeting:** *"a mechanism that can produce real voids rather
than damage proxies, pending experimental validation"* — **not** *"a validated void
model"*.

---

## Recommended order of work

| # | task | effort | blocked by |
|---|---|---|---|
| 1 | **A.1** strain-rate-only ranking | trivial | — |
| 2 | **B.7.1** incompressibility sanity check (detachment off) | low | D-family fixes in progress |
| 3 | **A.2** residence time + threshold sweep | low | — |
| 4 | **A.4** $\Delta p$ (trend first, threshold only if **[15]** verified) | low | — |
| 5 | **A.5** indicator table + rank-correlation matrix | low | 1, 3, 4 |
| 6 | **B.8** one targeted search: FSW sticking/slipping & shear-layer detachment criteria | one search | — |
| 7 | **B.3–B.4** derive $\Pi_2$, settle $\sigma_\text{ref}$, local form | moderate | 6 |
| 8 | **B.5** implement two-state rule with hysteresis | moderate | 2, 7 |
| 9 | **B.7.2–4** falsification suite | moderate | 8 |
| 10 | **A.3** Piwnik–Plata (relative ranking only, pending $w_\text{lim}$) | low | — |

✅ **Items 1–5 are independent of the detachment work** and can run as soon as the
D-family issues are closed. They also produce the baseline that any detachment result
must be compared against.

⚠️ **Item 6 before item 7.** Deriving the groups before checking the sticking/slipping
literature risks rediscovering published work under new notation.

---

## New open questions

To be merged into `OPEN_QUESTIONS.md` (current highest is **O54**):

- **O55** — Does $\int\dot\varepsilon\,dt$ alone reproduce the B-Flutes-highest
  ranking? If yes, $\exp(1.5\eta)$ contributes nothing and the damage apparatus has one
  effective degree of freedom. (Wang et al. 2020 **[10]** predicts yes.)
- **O56** — With detachment off, is the particle density field uniform to estimator
  noise? Any deficit is an artefact. Blocks every detachment result.
- **O57** — Does the sticking/slipping or shear-layer-thickness literature already
  contain a rate-competition detachment criterion? Novelty claim depends on this.
- **O58** — Can $\Pi_\text{crit}$ be fitted on A/B/C and predict D, or must it be
  refitted per family? If refitted, the dimensionless construction has failed.
- **O59** — What is the correct $\sigma_\text{ref}$ in $\Pi_2$, and can it be chosen
  without introducing a fitted $T_\text{ref}$?
- **O60** — How large is the one-way vs hysteretic gap in predicted void fraction? It
  bounds the error from treating detachment as irreversible.

---

## Terminology — use consistently

Extending the terminology already fixed in `DAMAGE_REPORT.md` (damage law / damage
indicator / defect / defect susceptibility):

| term | meaning |
|---|---|
| **entrained** | particle is following the solenoidal velocity field |
| **detached** | particle translates at $v_\text{adv}$, no longer sampling the rotating flow |
| **detachment criterion** | the local dimensionless test deciding between the two |
| **re-entrainment** | detached → entrained, requires hysteresis |
| **mass-deficit void** | density hole arising from the piecewise velocity field — a genuine bookkeeping outcome, as distinct from a damage proxy |

⚠️ Never **melted**, **molten**, **solidification** or **freezing**. FSW is solid-state;
peak $T$ is 80–95 % of melting and nothing liquefies.
