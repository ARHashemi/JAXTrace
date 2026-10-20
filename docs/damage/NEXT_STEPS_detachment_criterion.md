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

$$\boxed{\;B \;=\; P_\max(\theta=0°,\ \text{front}) \;-\; P_\min(\theta=180°,\ \text{rear})\;}$$

✅ **Read in full 2026-10-19** (PDF supplied). Shi et al., *Int. J. Mech. Sci.*
**241** (2023) 107969, Eq. (19) and Fig. 21.

⚠️ **TWO ERRORS IN THE EARLIER DRAFT OF THIS SECTION, both corrected above.**

1. **The inequality was inverted.** This said "Δp **<** 15 MPa for void
   formation". The paper says the opposite: *"void defects formed in the joint
   when the value of B is **greater than** 15 MPa"*. A sound joint is formed when
   the difference is **lower** than 15 MPa.
2. **The axis was wrong.** This said $p_\text{advancing}-p_\text{retreating}$.
   Eq. (19) is a **front-to-rear** difference along the travel direction, using
   the **maximum** in front and the **minimum** in the rear — not sector means,
   and not the advancing/retreating contrast, which is perpendicular to it.
   `indicator_dp.py` computed the advancing/retreating version and has been
   corrected; it now reports both, but only $B$ may be compared to 15 MPa.

- **Independent?** ✅ Yes, and ⚠️ **in the stronger sense** — see §A.0b. It is the
  only indicator that needs no particles at all.
- **Cost:** low, using the project's threaded PVTU loader.
- **Calibration:** the 15 MPa value is from *their* alloy and tool, so the
  **ranking across our 16 cases is the primary result**; the absolute pass/fail
  is indicative. Shi states no second condition — $B$ alone is the criterion.
- ⚠️ **Our pressure magnitudes need care.** `DAMAGE_REPORT.md` records measured
  −6 … −7.5 MPa ahead of the tool and +0.4 … +2.2 MPa behind, with the sign pattern
  holding in 9/10 cases. That is the right order for a 15 MPa threshold comparison,
  but the earlier "+1.70/−1.35 MPa" figures were **not reproducible** and were
  retracted — do not reuse them.

### A.0b ⚠️ Two different senses of "independent" — and we only tested one

Worth separating, because the original motivation and what was measured are not
the same thing:

| sense | meaning | which indicators qualify |
|---|---|---|
| **(i) independent INTEGRAND** | built from different field quantities | $t_\text{res}$ (a time, no stress), $w$ (partly) |
| **(ii) independent of the PARTICLES** | does not depend on seeding, tracking or element search at all | ⚠️ **only $\Delta p$** |

⚠️ **Every indicator measured so far is sense (i), not sense (ii).** $\ln\Phi$,
$C$, $A_{\dot\varepsilon}$, $t_\text{res}$ and $w$ are *all* pathline integrals:
they share the same seeding, the same 288,000 trajectories, the same RK4 stepping
and the same L0/L1/L2 element search. A systematic error in any of those —
a seeding bias, a lost-particle pattern, a search failure — moves **all five
together**, and no amount of agreement between them would reveal it.

✅ **$\Delta p$ is the only one in sense (ii).** It reads `Pressure` from the PVTU
nodes directly and compares the advancing and retreating sides of the pin at one
instant. No particle is integrated, no element is searched, no seed box is
chosen. It would therefore be the **first genuine external check** on the whole
particle apparatus.

⚠️ **This is also the honest limit of the §A.5b result.** "Two genuinely
independent indicators agree/disagree" means independent *integrands*. The
particle machinery underneath is common to all of them, and remains untested by
anything in that table.

**Other sense-(ii) candidates**, for completeness: the flow-partitioning /
mass-balance criterion (Arbegast 2008, §6 item 6) and the Cooper–Allwood
thresholds are also evaluable on the field alone. Neither is implemented.

### A.4b ✅ MEASURED — the external check, and it disagrees with everything

16 cases on LUMI (array 22955760, all COMPLETED, 15–27 min each using the
threaded PVTU loader). Median $B$ per case, Shi threshold 15 MPa:

| case | median B [MPa] | % snapshots > 15 |  | case | median B | % > 15 |
|---|---|---|---|---|---|---|
| **D2** | **16.80** | **52 %** |  | A1 | 1.68 | 0 % |
| **A2** | **12.82** | **36 %** |  | C2 | 1.65 | 0 % |
| A3 | 6.43 | 2 % |  | A4 | 1.63 | 0 % |
| D3 | 4.67 | 0 % |  | B1 | 1.60 | 0 % |
| B4 | 4.27 | 0 % |  | C4 | 1.58 | 0 % |
| D1 | 3.67 | 0 % |  | B3 | 1.49 | 0 % |
| B2 | 3.67 | 0 % |  | C3 | 1.48 | 0 % |
| D4 | 3.58 | 0 % |  | C1 | 1.77 | 0 % |

✅ **$B$ discriminates sharply** — an **11× spread** (1.48 … 16.80) against the
1.17–1.19× of the damage block. Only **D2** and **A2** approach the threshold.

⚠️ **AND IT CORRELATES WITH NOTHING WE HAVE.** Rank correlation of $B$ against
the pathline indicators, 16 cases:

| | ln_phi | C | A_edot | w_bond | density_cv |
|---|---|---|---|---|---|
| **B** | −0.174 | −0.047 | −0.174 | **+0.121** | **−0.518** |

This is the result that matters, and it cuts both ways:

- ✅ **It is a real external check.** $B$ needs no particles — no seeding, no
  tracking, no element search (§A.0b). It is the only indicator that could have
  revealed a systematic fault in the particle apparatus, and the near-zero
  correlations mean it is **measuring something the pathline family does not
  capture at all**, not merely re-expressing it.
- ⚠️ **Three indicator families now give three different rankings.** Damage laws:
  B-Flutes highest. Bonding ($w$): D-Concavity highest. Contact pressure ($B$):
  **D2 and A2 highest, with their own families' other members near the bottom** —
  so $B$ does not even rank by *family*, it ranks by *case*.
- ⚠️ **$\rho(B, \text{density CV}) = -0.518$** is the one non-trivial link, and it
  has the **wrong sign** for comfort: cases with a *higher* contact-pressure
  difference show *lower* particle-density scatter. If $B$ predicts voids, the
  density measure should move with it, not against it.

⚠️ **None of this can be adjudicated from simulation.** Three indicators, three
rankings, no ground truth. §B.8 (one experimental cross-section) is now the
binding constraint on the whole programme, not a nice-to-have.

### A.4c Validation references — what exists, and how far each stretches

Deep search completed 2026-10-19 (205 papers) plus two full texts supplied.
⚠️ **No study matches our alloy AND our four pin families.** Ranked by usability:

| source | alloy | why it is usable | gap to our case |
|---|---|---|---|
| **Chen et al. (2023)**, *J. Manuf. Process.*, 33 cit. ✅ PDF read | **AA6061-T6, 600 rpm** | ⚠️ **the closest match we have.** Same rotation speed; CFD with *non-uniform contact pressure*; voids form on the **advancing side** when rear contact pressure drops; quantitative void width/height for six pin sizes (Table 4: 0.81–2.40 mm width); **validated against measured cross-sections** — sound at 600-20, voided at 600-180 | 6061 not 6063; varies pin *size*, not our four profile families |
| **Shash et al. (2022)**, *Proc. IMechE C* | ⚠️ **AA6063** | the only hit on **our exact alloy**, coupled experimental + numerical | 11 citations; pin geometry not varied |
| **Yadav et al. (2025)**, *JMR&T*, 8 cit. ✅ PDF read | AA6082-T6, 710 rpm | **four pin profiles** with a clear defect ordering: threaded cylindrical → *largest* tunnel; square → *smallest* | pipe butt welds, not plate; 6082; profiles differ from ours |
| **Reza-E-Rabby & Reynolds (2014, 2018)**, 53 + 38 cit. | AA6061 | **thread form / thread interruption** varied systematically — the closest published analogue to our C-Threads family | threads only |
| **Hamade & Baydoun (2019)**, *Mater. Des.*, 42 cit. | Al lap joints | **X-ray CT** defect detection — the measurement modality §B.8 asks for | lap joints, not butt |
| **Bui et al. (2026)**, *IJAMT* | **AA6061** | void formation vs welding parameters, very recent | parameters, not pin geometry |

✅ **The defensible validation plan**: use **Chen (2023)** for the *mechanism*
(advancing-side void when rear contact pressure falls — directly testable against
our $B$ and our predicted void location) and **Yadav (2025)** for the *trend*
(pin profile changes defect severity, ordering is measurable). ⚠️ Neither
validates our specific 16-case ordering, and claiming otherwise would overstate.

### A.5a ⚠️ What is actually implemented — the exact tally

Worth stating plainly, because "we added four independent indicators" is close
but not right, and the difference matters when describing the work:

| # | indicator | status | how |
|---|---|---|---|
| 1 | $\ln(\Phi/\Phi_0)$ Rice–Tracey | **pre-existing** (45-case survey) | pathline accumulator |
| 2 | $C$ Cockcroft–Latham | **pre-existing** | pathline accumulator |
| 3 | $t_\mathrm{res}$ residence time | ✅ **added here** | pathline accumulator |
| 4 | $w$ Piwnik–Plata | ✅ **added here** | pathline accumulator |
| — | $A_{\dot\varepsilon}=\int\dot\varepsilon\,dt$ (A.1) | ✅ available | ⚠️ **DERIVED**, not accumulated: recovered from $\ln\Phi$ as its degenerate limit |
| — | $\Delta p$ (A.4) | ⚠️ **NOT RUN** | `scripts/damage/indicator_dp.py` exists but is a pressure-FIELD diagnostic, never executed |

So: **two new accumulators**, not four. The accumulator width went 2 → 4
(`[ln_phi, C, t_res, w_bond]`), and the fifth column in the results table,
$A_{\dot\varepsilon}$, costs nothing because it is $\ln\Phi$ divided by a constant
— which is precisely why $\rho(\ln\Phi, A_{\dot\varepsilon}) = +1.000$ is a
measurement of $\exp(1.5\eta)$ rather than a tautology.

✅ **All four accumulators computed correctly** on all 16 cases: 100 %
accumulation, all archives pass `verify_indicator_npz.py`, every column finite.

⚠️ **$\Delta p$ remains the one unexecuted item of Part A.** It is independent of
the damage integral and cheap, so it is still worth running — but it needs a
pressure-field pass over the PVTUs, not a tracking run.

### A.5b ⚠️ Measured outcome, full family (2026-10-18)

16 cases on LUMI, 288,000 particles each, all exit 0:0, all passing
`verify_indicator_npz.py`. Pooled over **4,092,780** wake-box particles:

|            | ln_phi |     C  | A_edot | t_res  | w_bond |
|------------|--------|--------|--------|--------|--------|
| **ln_phi** |  1.000 |  0.996 |  1.000 |  0.802 | −0.085 |
| **t_res**  |  0.802 |  0.802 |  0.802 |  1.000 |  0.019 |
| **w_bond** | −0.085 | −0.123 | −0.085 |  0.019 |  1.000 |

✅ **The indicator set did its job**: it collapsed three of the five columns into
one and left exactly one genuinely new signal.

⚠️ **The damage block is degenerate.** `ln_phi`, `C` and `A_edot` are the same
quantity to three decimals. Across the 16 cases they span only **1.17–1.19×**
between the lowest and highest case — barely more than noise. Reporting them as
three indicators overstates the evidence.

✅ **`w_bond` is the one that discriminates**, spanning **7.03×** across the
family, and it separates the families in a way nothing else does:

| family | median $w$ | range |
|---|---|---|
| A-Flats | 0.0111 | 0.0072 – 0.0379 |
| C-Threads | 0.0241 | 0.0236 – 0.0250 |
| B-Flutes | 0.0293 | 0.0216 – 0.0345 |
| **D-Concavity** | **0.0464** | 0.0400 – 0.0508 |

⚠️ **This reverses the headline of the existing report.** `DAMAGE_REPORT.md` has
B-Flutes highest on every wake statistic — and on the damage laws that is still
true. But on the bonding criterion **D-Concavity is highest by a clear margin**,
and the two measures disagree ($\rho = -0.085$). Which ranking is meaningful is
an open question that only experimental validation can settle (§B.8), and the
honest statement is that *the answer depends on which indicator you believe*.

⚠️ `t_res` has a median of exactly 0 in every case, because the majority of wake
particles never enter the shear layer (65.6 % in D2). The median is therefore the
wrong summary statistic for it — use the entrained fraction or a high percentile.
See the figure note in `scripts/make_indicator_figures.py`.

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

### B.3b ⚠️ How Carlson & Beckermann actually did it — the template

✅ **Read from the full text** (`s11661-008-9715-y.pdf`, supplied 2026-10-16).
This matters because §B.3 above currently *assembles* two rates that happen to
have the right dimensions. That is not how the criterion we are imitating was
built, and a reviewer who knows this literature will see the difference.

**Their chain, in order:**

$$\boxed{\;Ny^* \;=\; \frac{Ny}{\sqrt{\dfrac{\mu_l\,\beta\,\Delta T_f}{\Delta P_{cr}\,\lambda_2^{2}}}}\;=\;\sqrt{I(g_{l,cr})}\;}$$

and, substituting the secondary dendrite arm spacing $\lambda_2 = C_\lambda \dot T^{-1/3}$,

$$Ny^* \;=\; C_\lambda\,\frac{G}{\dot T^{5/6}}\sqrt{\frac{\Delta P_{cr}}{\mu_l\,\beta\,\Delta T_f}}$$

1. **Start from a transport equation** — Darcy flow through the mushy zone,
   1-D directional solidification. Not from dimensional analysis.
2. **Group the material and process constants** into the denominator
   ($\mu_l$ viscosity, $\beta$ shrinkage, $\Delta T_f$ freezing range,
   $\Delta P_{cr}$ critical pressure drop, $\lambda_2$ arm spacing).
3. ⚠️ **The decisive step: $Ny^*$ equals $\sqrt{I(g_{l,cr})}$, an integral whose
   ONLY argument is the critical liquid fraction.** So $Ny^*$ is a function of the
   solid-fraction–temperature curve *alone*. That is what makes the threshold
   universal — not the non-dimensionalisation by itself.
4. **Invert to a physical quantity.** $Ny^* \to I(g_{l,cr}) \to g_{l,cr} \to$ pore
   volume fraction $g_p = \beta' g_{l,cr}$ from the continuity equation, assuming
   feeding ceases once porosity forms. ✅ **The output is a predicted POROSITY, not
   a flag.** They call this inversion "a novel concept that significantly enhances
   the usefulness of this new criterion".

⚠️ **The $\dot T^{5/6}$ exponent is not a fitting choice.** It follows *solely* from
the arm spacing $\lambda_2 \propto \dot T^{-1/3}$ and that spacing's effect on mushy-zone
permeability. The authors note the same dependence was found independently in three
other studies. **Lesson: the exponent fell out of the physics.** If our FSW groups
need a fitted exponent $n$ (as §B.3 currently proposes), that is a sign the
derivation is missing, not a free parameter to calibrate.

**What the FSW analogue must therefore have, to be the same kind of object:**

| their ingredient | our required analogue | do we have it? |
|---|---|---|
| Darcy flow in the mushy zone | a transport equation for material delivery into the wake cavity | ❌ **not yet derived** |
| permeability ~ $\lambda_2(\dot T)$ | how delivery rate depends on the local state | ❌ |
| $I(g_{l,cr})$ — single-argument integral | a function of one state variable only | ❌ |
| $g_p = \beta' g_{l,cr}$ | void fraction from a mass balance | ⚠️ partly: the particle-density deficit IS a mass balance |
| $\Delta P_{cr}$ calibrated once per alloy | one calibrated constant, not several | ⚠️ §B.3 currently needs $n$, $\Pi_{crit}$, $\Delta$, $\ell$ |

✅ **The one place we are already in the right shape** is the output: a
particle-density deficit is a mass-balance-derived void fraction, which is exactly
the kind of quantity $g_p$ is. ⚠️ **The gap is the front end** — we have no transport
equation, so we cannot yet claim the same rigour.

**Monroe & Beckermann (2014)** (`s11837-014-0999-7.pdf`, also supplied) is the
closest guide for adding our mechanical term: they extend the same $Ny^*$ with an
applied strain rate, which enhances the mushy-zone pressure drop. Read for *how the
mechanical term enters the pressure balance*, since that is structurally what the
tool's delivery rate must do in our case.

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

⚠️ **SEARCHED 2026-10-16. The risk was real, and the novelty claim must shrink.**
An Undermind deep search (204 papers) on exactly this question found substantial
prior art. These are now in Zotero under *FSW Void & Wormhole Prediction*, tagged
`O57`:

| what exists | reference | overlap with our proposal |
|---|---|---|
| **Sliding-to-sticking transition linked DIRECTLY to void formation** | Zhao, Pei & Liu (2022), *Int. J. Adv. Manuf. Technol.* | ⚠️ **the causal chain we proposed, already published** |
| Analytical stick-slip model, **dimensionless** functions, fraction tied to spin rate | Wang et al. (2021), *Acta Materialia*, 77 cit. | ⚠️ a dimensionless stick/slip criterion already exists |
| **Critical radius = "location of vanishing material velocity"**, non-dimensional, Bingham plastic, peak T ~80 % of melting | Buck & Langerman (2004) | ⚠️ **an analytically-derived dimensionless detachment radius, 20 years old** |
| Formal **scaling analysis** of coupled heat transfer + plastic deformation around the pin | Mendez, Tello & Lienert (2010), *Acta Materialia* | ⚠️ the non-dimensionalisation of §B.3, already done properly |
| Shear-localisation model for process-window estimation | Pei & Dong (2017, 2014) | "shear localisation" may be the established term for our "detachment" |

✅ **What this changes.** Three claims in the earlier draft are no longer available:
"a dimensionless FSW criterion is novel" (Wang 2021, Buck 2004), "deriving the groups
is novel" (Mendez 2010), and "linking a stick/slip transition to voids is novel"
(Zhao 2022). ⚠️ Note Buck & Langerman give a *yield-stress* (Bingham) basis for
material ceasing to move, which is cleaner physics than the thermal-softening ratio
$\Pi_2$ proposed in §B.3 — consider adopting it.

⚠️ **What may still be novel**, pending full-text reads: evaluating the criterion
**locally, per tracer particle, along its own trajectory**, and producing the void as
a **particle-density deficit from a piecewise-divergent ensemble** rather than from a
CEL/VOF free surface or an interface-level condition. That is a narrower claim than
the one this document originally made, and it cannot be confirmed until Zhao (2022)
and Wang (2021) are read in full.

So the honest claim is narrow:

> ✅ **Defensible claim:** a *dimensionless, locally-evaluated detachment criterion*
> applied to **tracer particles** to produce a mass-deficit void prediction, with the
> threshold as a pure number.
>
> ❌ **Not defensible:** "a novel void-formation mechanism", "a solidification model
> for FSW", or any claim of priority over the sticking/slipping literature.

**One targeted literature search would settle this** — see §B.8.

### B.6c ✅ READ 2026-10-19 — the novelty question, answered

**Zhao, Pei & Liu (2022)** read in full. ✅ **Our remaining claim survives, and
it is sharper than I expected.**

They *do* use particle tracing — but **only to compute a strain distribution**
on a cross-section 30 mm behind the pin. ⚠️ **Their void criterion is not on the
particles at all.** Two criteria, both geometric:

| model | their void criterion |
|---|---|
| CFD | position of the **TMAZ boundary** relative to the pin periphery on the advancing side |
| shear localisation | **shear-layer width** $\delta$ vs $w_{\max}$ (their Eq. 3) |

⚠️ **Their particles never detach.** They advect through the whole field, and the
void is inferred from *where a boundary sits*, not from material failing to
arrive. There is **no mass deficit and no density hole** anywhere in their method.

✅ **So the distinct contribution is intact**, and it is specifically:

> a **local, per-particle detachment rule** evaluated along each trajectory,
> producing the void as a **particle-density deficit from a piecewise-divergent
> ensemble** — rather than inferring it from the position of a boundary in the
> field.

⚠️ **What this does NOT make novel**, and the write-up must keep saying so:
linking a contact-state transition to voids (Zhao 2022), a dimensionless
stick/slip criterion (Wang 2021), the scaling groups (Mendez 2010), or a
dimensionless critical radius (Buck 2004).

**Buck & Langerman (2004)**, also read: they derive a **critical radius**, "the
location of vanishing material velocity", as a closed-form function of the
**Bingham number** $Bu$ — a yield-stress parameter — plus a geometric ratio $a$
and $R$. ✅ **This is the right basis for $\Pi_2$**: a particle stops moving when
the local stress cannot overcome the yield stress, which is a cleaner and more
defensible statement than a thermal-softening ratio. Their $Bu$ range is 0.1–5.0,
and $Bu$ is interpretable as a reciprocal dimensionless rotation rate.

### B.6d ✅ THE LITERATURE IS CLOSED — novelty verdict, and where the method comes from

Four full texts read 2026-10-19 (Zhao 2022 in B.6c; Mendez 2010, Wang 2021,
Buck 2004 here), plus a deep search run specifically on the *mechanism* rather
than on FSW keywords. The question is settled in both directions.

#### ✅ The claim, as narrowly as it can be stated — and it holds

> No located work advects **passive markers through a previously computed,
> fixed velocity field** and lets **individual particles detach by a local
> criterion**, so that the void appears as a **particle-density deficit**.

⚠️ **But "nobody has done it" is the weaker half of the finding.** The deep
search found that the mechanism *is* established — in **pore-scale colloid
filtration**, where particles are tracked through a precomputed carrier flow and
then immobilised or remobilised per-particle. ✅ **This is the same mathematical
object as ours**, and it is a far better citation than a null result:

| | colloid filtration | ours |
|---|---|---|
| carrier flow | precomputed, fixed | precomputed, fixed (FOM) |
| particles | passive, advected | passive, advected |
| local switch | hydrodynamic torque > adhesive torque | stress < yield (to be derived) |
| observable | retention / deposition profile | particle-density deficit = void |

The canonical statement of that switch is a **force/torque balance**
(Torkzaban 2007; Bradford 2010; Vanness 2019 — a *unified* force-and-torque
balance). ✅ **We should cite this as the methodological precedent**: it makes
the detachment post-process a recognised technique being transferred to a new
physics, rather than an invention needing defence from scratch.

⚠️ **Also from the search: the switch in that literature is almost always
DIMENSIONAL** (a force/torque comparison), with dimensionless groups used only
to organise regimes. ✅ **So the dimensionless form of the criterion is the
genuinely open part** — which is exactly where B.3b's Niyama-style derivation
sits. That is the contribution, not the particle bookkeeping.

#### ⚠️ What is NOT novel, and must be cited as prior art

- **Marker/tracer particle tracing in FSW** — Dialami et al. (2015, 2019, 2020).
  ⚠️ These are our own team's, and Dialami 2020 is titled *"Defect formation and
  material flow in FSW"*. The write-up must not imply tracer-based defect
  discussion is new.
- **Marker-and-cell** — Harlow & Welch (1965): markers track the liquid region,
  i.e. a free surface, not detached particles.
- **Void prediction via free-surface/VOF or explicit damage evolution** — the
  standard routes; ours is neither.
- ❌ **Particle methods (SPH / MPM / PFEM / DEM)** are a *different kind of
  thing*: there the particles **are** the discretisation and **carry mass**, so
  voids arise from the solution itself. See §B.6e.

#### ✅ Where the criterion's two groups now come from — both sourced

**$\Pi_1$, a yield/stress group — Buck & Langerman (2004).** Their critical
radius, "the location of vanishing material velocity", is a closed form in the
**Bingham number** $Bu$, over $Bu \in [0.1, 5.0]$. ✅ A particle detaches when
the local stress can no longer overcome the yield stress. This is cleaner and
more defensible than a thermal-softening ratio.

**$\Pi_2$, a kinematic/geometric group — Mendez, Tello & Lienert (2010).** Their
Eq. (30) gives the **shear-layer thickness** in closed form:

$$\delta = a\left[\tfrac{8}{3}\,\frac{\Delta T_0}{g\sigma_s}\,
\frac{\tau_R}{AB}\left(\frac{a^2}{k_0}\right)^{\!n}
\left(\frac{gK_0}{\Delta T_0}\right)\omega^{\,n+1}\right]^{1/2}$$

with the maximum shear rate $\dot{\gamma} = \omega a / \delta$ (their Eq. 35).
✅ **$\delta$ is the natural length to normalise a detachment distance against**,
and it is available from process parameters alone — no extra simulation.

⚠️ **Wang 2021 is a near-miss and must be read as a warning, not a basis.** Their
dimensionless slip parameter $\delta \in [0,1]$ and stick fraction $c/a$ are
**interfacial**, on the tool/workpiece contact. ✅ Their Eq. (10) is nonetheless
the right *functional form* to borrow — a switch on a relative velocity against a
critical $v_c$:

$$\tau_f = \mu_f p \,\mathrm{sgn}(v_{rel})\times
\begin{cases} 1 & |v_{rel}| \ge v_c\\ \tanh(|v_{rel}|/v_0) & |v_{rel}| < v_c\end{cases}$$

⚠️ **And their own stated limitation is the one we will inherit**: the predicted
stick-slip ratio **depends on the choice of $v_c$**. Any velocity-threshold
detachment rule we write has a free parameter, and the honest move is to say so
and show the sensitivity, not to tune it to a known answer.

### B.6e ⚠️ Do the particles need interaction forces? No — and adding them would break the method

**Short answer: no, and it is not a detail — it is the method's premise.**

Our particles are **passive tracers in a precomputed field**. They carry no mass
and no momentum, and the field was already solved with momentum balance and
incompressibility by the FOM. Adding interaction forces would mean:

1. ⚠️ **Double-counting.** Inter-particle forces in the FOM's material are
   *already* in the constitutive response that produced the velocity field.
   Adding them again on the tracers imposes the same physics twice.
2. ⚠️ **An inconsistent system.** The tracers would obey a momentum balance that
   the carrier field does not satisfy. With $\nabla\!\cdot u = 0$ imposed on the
   field but repulsion acting on the particles, the two disagree, and nothing
   reconciles them.
3. ❌ **It becomes a particle method.** Interaction forces + mass = SPH/MPM. That
   is a legitimate approach to FSW voids — **and it is what someone should do
   instead if they want voids to arise from the solution** — but it requires
   **re-solving momentum**, which is precisely what a post-process is for
   avoiding.

✅ **The detachment rule is the deliberate alternative to interaction forces.**
The whole reason a rule is needed is the result in §A: with $\nabla\!\cdot u=0$ and
fully entrained particles, $\det F \equiv 1$, so a void is **impossible by
construction**. The two ways out are (a) give the particles mass and re-solve —
a particle method — or (b) break entrainment with a local rule and read the void
off the resulting density deficit. ✅ **We are doing (b), on purpose.**

⚠️ **One real limitation to state, not hide.** Without interaction, detached
particles cannot *resist* each other, so the method predicts **where material
fails to arrive** — it does not predict the void's **equilibrium shape** after
surrounding material consolidates. ✅ That is a scope statement for the paper, not
a defect: the void's *existence and location* is the quantity being claimed.

### B.6f ✅ IMPLEMENTATION STEPS — in order, with what each costs

Each step is independently checkable, and **steps 1–2 produce a publishable
negative or positive result on their own.**

**Step 1 — prove the null result on our own data (½ day).** ✅ Measure
$\det F$ along our existing trajectories and confirm it is $1$ to integration
error. This is the figure that *motivates* everything else: it shows no
criterion-free post-process can ever produce a void. ⚠️ Do this first — if
$\det F$ drifts materially from 1, the mesh/interpolation error is larger than
the effect we intend to measure, and the whole plan needs rethinking.

**Step 2 — implement detachment with the simplest defensible rule (1–2 days).**
A particle detaches when the local stress cannot sustain entrainment. Start from
Buck's Bingham form, evaluated from fields we already carry:

$$\Pi_1 = \frac{\tau_{local}}{\tau_{yield}(T,\dot\varepsilon)},\qquad
\text{detach when } \Pi_1 < \Pi_1^{crit}$$

✅ Reuse the existing accumulator machinery: add one `int8` state column
(`attached`/`detached`) and freeze a detached particle's velocity contribution.
⚠️ Do **not** delete detached particles — keep them and record where they stopped;
the density deficit is computed from the *survivors*, and the stopped set is the
diagnostic.

**Step 3 — make the criterion dimensionless (the actual contribution, 2–3 days).**
Normalise with Mendez Eq. (30)'s $\delta$ and the Bingham number, then follow the
**Carlson & Beckermann template in §B.3b**: derive, don't fit, and invert to a
predicted void fraction rather than a yes/no flag. ✅ This is the part no located
work has done.

**Step 4 — sensitivity, reported honestly (1 day).** ⚠️ Wang's warning applies to
us: the threshold is a free parameter. Sweep $\Pi_1^{crit}$ and show how the void
map moves. A criterion whose answer is stable over a decade of threshold is a
result; one that is not must be reported as not.

**Step 5 — validate against the pattern, not the number (1 day).** ✅ We already
have the comparison in place (§8d.3b): $\gamma = 1.0$ mm/rev puts us $3.3\times$
beyond Chen's voided case, so **we should predict sound welds**. ⚠️ A criterion
that predicts voids everywhere on our 16 cases is falsified by that, and that is
the most valuable check available to us.

### B.6g ✅ IMPLEMENTED AND TESTED on D2 — steps 1 and 2 are done

`--damage-detachment`, D2 smoke test 2026-10-19 (50 steps, 11,200 particles,
7 snapshots, exit 0 in 432 s on the workstation RTX 5090). Both columns work.

**✅ Step 1, the null-result gate — PASSED, and this is the important number:**

```
det F = exp(ln_J)  : median 1.000000   min 0.999856   max 1.000212
|det F - 1|        : median 1.43e-05   p99 1.19e-04   max 2.12e-04
```

⚠️ **SUPERSEDED BY §B.6i — this conclusion was wrong.** On the full 8000-step
run the median is 3.17e-03 and the max 1.09e+02, because 50 steps in the
undeformed far field cannot test an accumulator that only diverges where the
material deforms. The text below is kept as the record of what the smoke test
showed and what I incorrectly concluded from it.

~~det F is 1 to 1.4e-05. The null result is confirmed *on our own data and
our own pipeline*: with $\nabla\!\cdot u=0$ and full entrainment, no
criterion-free post-process can ever produce a void. And the gate is
passed with four orders of magnitude of headroom~~ — a void fraction of
interest is O(1e-2…1e-1), against an interpolation floor of 1.4e-05. Any
detachment signal above ~1e-4 is resolvable and not numerical noise.

**Step 2, Π₁ recorded as a running minimum:**

```
Pi_1 min along path : median 0.9155   min 0.8974   max 0.9237  (11,200/11,200 finite)
detached fraction:  Pi_1_crit 0.10..0.70 -> 0.00 %     0.90 -> 0.73 %
```

⚠️ **Read this as a smoke test, not a result.** The spread is narrow
(0.897–0.924, i.e. 2.9 %) and `t_res = 0` on *every* particle, which the run
flags itself: over 50 steps the particles never reach the shear layer. ✅ So
these particles are the *undeformed far field*, where Π₁ ≈ 0.9 is exactly what
should be expected — the flow stress sits just below the reference yield
stress and nothing detaches. **The test proves the plumbing, not the physics.**

⚠️ **Π₁'s absolute scale is set by `--damage-tau-ref-edot` (default 1.0 1/s)**,
so 0.9 is not meaningful on its own: it says the local flow stress is 0.9× the
Norton stress at 1 s⁻¹. ✅ Only the *full-length* run, where particles pass
through the shear layer, can show whether Π₁ develops a usable spread.

✅ **The sweep is free, as designed.** The detached-fraction curve over
$\Pi_1^{crit}$ is printed from the recorded minimum, so §B.6f step 4 costs no
extra runs. The threshold is never applied in-kernel.

#### ⚠️ One thing the smoke test exposed, unrelated to detachment

`rank corr(lnPhi, C) = +0.8978`, which the code labels *"MODELS DISAGREE —
investigate"*. ⚠️ On the 16-case production runs this was +0.982…+0.998. The
difference is the 50-step window, not a regression: over 50 steps `ln_phi` and
`C` are both ~5e-04 and dominated by their first few increments. Worth
confirming on the full run, but it is not evidence of a problem in this change.

#### Implementation notes worth keeping

- `div_u` **costs nothing**: `tr(D)` was already computed in `fields.py` for the
  incompressibility diagnostic and thrown away.
- ⚠️ `tau_ratio` must be built against a **reference** yield stress. On the
  `models` route `sigma_eq` *is* the Norton flow stress, so dividing by the
  local value would give Π₁ ≡ 1 everywhere. It uses the Norton stress at
  `--damage-tau-ref-edot` with the **local** temperature.
- ⚠️ `tau_ratio` is **not** masked by `active`, unlike `eta`: a particle in a
  dead zone has $\dot\varepsilon\approx0$ and therefore the *lowest* flow
  stress, which is precisely the detachment condition. Masking it to 0 would
  erase the signal.
- ⚠️ `tau_min` is initialised to `+inf`, not 0, or the running minimum latches 0
  forever. All statistics exclude non-finite entries.
- ✅ The existing step-freeze reverts the whole accumulator vector, so a skipped
  step cannot latch a minimum from a position the particle never legitimately
  occupied. Verified, and the min update is ordered before the freeze.
- Field stack grows 3 → 5 slots (+67 % memory: 256.5 MB at D2 scale, 7
  snapshots); one extra Norton evaluation per timestep; 2 floats per particle.

✅ **Next: the full-length D2 run**, where particles traverse the shear layer —
that is the first run in which Π₁ can say anything physical.

### B.6h Run plan — what to march, and why the workstation run comes first

**✅ Step counts: the reference values, unchanged.** Detachment adds columns to
the same integration, so there is no reason to march differently from the
previous indicator runs. From each case's own `run_jaxtrace.sh`:

| | value (D2) |
|---|---|
| `N_STEPS` | **8000** |
| `VEL_START` / `VEL_END` | **34 / 199** (B2 is 201 — per-case, never generalise) |
| `DT` | **0.0003629** (B2 is 0.0003) |
| `SEED_GRID` | **60 120 40** = 288,000 particles |

⚠️ These differ per case and the per-case scripts are authoritative. The LUMI
submitter **injects flags into those scripts** rather than regenerating them,
precisely so facts like B2's `dt` and `VEL_END` cannot be lost.

**Measured cost. ⚠️ Two baselines, and they are not comparable:**

```
                    2026-10-16 (3 slots)   2026-10-19 (5 slots, detachment)
  commit            ea0b43e                af28eba
  Stage 1 mesh      1417 s                 1710 s
  Stage 2            140 s                   75 s
  Stage 3 damage    3539 s                 1650 s      <-- FASTER with MORE slots
  Stage 7 tracking  8574 s                 (in progress)
  TOTAL            13750 s = 3.82 h
```

⚠️ **Stage 3 halved while gaining two slots, and that is not an anomaly to
celebrate blindly — it is a different baseline.** The 16 Oct run predates
commit `0484d55` ("hoist mesh geometry out of the damage build; vectorise the
point-in-tet precompute"), verified by `git merge-base --is-ancestor`. ✅ So the
hoist's gain is real and larger than the two extra slots' cost; the two numbers
must simply never be quoted as "detachment made the build faster".

✅ The 5-slot stack measured **6083.2 MB**, matching the projection from the
3-slot figure exactly (3650 × 5/3 = 6083).

#### ⚠️ GPU memory: my projection was right about the arrays and wrong about the total

Measured in flight on the RTX 5090 (`XLA_PREALLOC=0`, on-demand allocator):

| | MB |
|---|---|
| velocity stack (166, 1.92 M, 3) | 3 650 |
| **damage stack, 5 slots** | **6 083** |
| temperature stack | 1 217 |
| octree | 723 |
| mesh | 495 |
| particles (pos + 7 accumulators) | 11 |
| **sum of persistent arrays** | **12 178** |
| **observed process usage** | **28 420** |
| ⚠️ **XLA workspace / compiled kernels** | **16 242** |

⚠️ **So the 12.2 GB figure quoted earlier was the ARRAYS, not the footprint.**
The real number is 28.4 GB of 32.6 GB — **13 % free, not 63 %**. The fused RK4
kernel's scratch space dominates, and it is ~1.3× the size of all persistent
data put together.

⚠️ **This also means the 16 Oct 3-slot run was already at ~26.0 GB (80 %).** The
two extra slots pushed an already-tight configuration to 87 %. It fits, and the
run is marching at 288,000/288,000 active with 0 lost — but there is **no room
for a third pair of columns** on this card without reducing the snapshot count
or the particle count.

✅ **On LUMI this is a non-issue**: the same footprint is 43 % of an MI250X GCD's
64 GB, leaving ~37 GB. ✅ And it reinforces why `XLA_PREALLOC=0` is mandatory on
the workstation — a 75 % preallocation (24.5 GB) would leave 8 GB for a
workspace that demonstrably wants 16 GB.

⚠️ **So the workstation cannot do this in 2 hours — it is ~4 h per case**, and
mesh loading plus the damage build is 36 % of it before a single step is
marched. ✅ This is why the family sweep belongs on LUMI, where the 16 cases run
*simultaneously* rather than 16 × 4 h in series.

**Memory, projected from the measured 3-slot figures and verified in the smoke
test:** the field stack grows 3 → 5 slots, 3650 → **6083 MB (+2433 MB)**, on
both host and device. Total GPU resident ≈ **12.2 GB** of the RTX 5090's
32.6 GB (63 % free); on MI250X's 64 GB it is comfortable. ✅ No allocator change
needed — the workstation's `XLA_PREALLOC=0` override still applies.

#### ⚠️ Why a short end-to-end run was NOT sufficient on its own

The smoke test **was** end-to-end — all 7 stages, exit 0, npz with 7 named
columns. ✅ It validated the plumbing completely. But it marched 50 steps over
7 snapshots, and `t_res = 0` on **every** particle: the particles never reached
the shear layer. ⚠️ **So Π₁ was only ever sampled in the undeformed far field**,
where it sat in a 2.9 % band around 0.9 and nothing detached.

That leaves two things untested by the smoke run, both of which only appear
where the material actually deforms:

1. ⚠️ **Whether Π₁ develops usable spread at all.** If it stays within a few
   percent even through the shear layer, the criterion has no discriminating
   power and the dimensionless form (§B.6f step 3) is attacking the wrong group.
2. ⚠️ **Whether the `clip(…, 0, 10)` upper bound on `tau_ratio` binds.** It can
   only be reached at high $\dot\varepsilon$, i.e. inside the layer. A clip that
   silently saturates would flatten the top of the distribution.

✅ **Hence: one full-length D2 run on the workstation first** (≈4 h), then the
LUMI family sweep. The cost of being wrong about either point across 16
simultaneous LUMI jobs is far higher than one workstation run.

⚠️ **The detachment plan is NOT yet complete** — this delivers §B.6f **steps 1
and 2 only**. Step 3 (the dimensionless form, the actual contribution) is not
implemented, and step 4's sweep is free *by construction* because `tau_min` is
recorded rather than thresholded. So the full-step LUMI jobs are collecting the
**inputs** the derivation needs, not executing a finished criterion.

### B.6i ⚠️ STEP 1 FAILED — the field is not discretely divergence-free

Full D2 run, 2026-10-19 (8000 steps, 288,000 particles, exit 0 in 12,119 s =
3.37 h, `lost=3,190` matching the 16-10 reference exactly). ⚠️ **The step-1 gate
does not pass, and the smoke test's result was an artefact.**

```
                         smoke (50 steps)      FULL (8000 steps, alive only)
  median |det F - 1|     1.43e-05              3.17e-03      (222x worse)
  max    |det F - 1|     2.12e-04              1.09e+02      (det F = 109.7!)
  min     det F          0.999856              0.0261
  frac   |ln_J| > 1e-2   ~0                    29.4 %
```

⚠️ **det F reaching 109.7 and 0.026 is not interpolation error — it is a factor
of 100 change in volume, which is physically impossible for this flow.** I
reported the smoke result as "✅ the null result holds" (§B.6g). ⚠️ **That
conclusion was wrong**: 50 steps in the undeformed far field cannot test an
accumulator that only diverges where the material deforms.

#### ✅ Root cause, measured directly on the field — it is NOT the integrator

`div u = tr(D)` computed per element from snapshot 34, the same way
`build_damage_fields` does it:

```
  |div u| / |D|   per element      RAW |div u| [1/s]
     p50    4.87e-02                  p50    2.31
     p75    1.86e-01                  p95   61.2
     p90    5.81e-01                  p99  321.5
     p95    8.21e-01                  p100 5938.7
     p99    1.08e+00
     p100   1.68e+00              (edot median 54.6 1/s)
  frac > 0.01 : 79.3 %      frac > 0.1 : 36.0 %
```

⚠️ **The discrete field carries a divergence of ~5 % of |D| at the median and
over 100 % in the tail.** A P1 tetrahedral velocity interpolation of an
incompressible solve is only *weakly* divergence-free — it satisfies the
constraint in an integral/variational sense, not pointwise per element. So
$\ln J=\int \nabla\!\cdot u\,dt$ integrates a **systematic discretisation
residual** over 8000 steps, and it accumulates.

✅ **This means the sign of the finding is the opposite of a bug in my code**: the
accumulator is faithfully reporting a real property of the FOM discretisation.

#### ⚠️ What this does to the plan

The step-1 gate was supposed to establish a floor that any detachment signal
must beat. ⚠️ **The floor is now 3.2e-03 median / 2.6e-01 at p99, against a void
fraction of interest of O(1e-2…1e-1).** The p99 floor *overlaps the signal*.

❌ **So "void = particle-density deficit from det F" is NOT viable as stated.**
A density deficit of a few percent cannot be distinguished from the
discretisation's own volume error. This is a real obstacle, not a tuning issue,
and §B.6f step 3 cannot be built on $\det F$ directly.

✅ **Three routes remain, in order of how much they salvage:**

1. ✅ **Use det F as a RELATIVE, not absolute, measure.** The error is
   systematic, so a *detached* particle's $\ln J$ versus the local *attached*
   mean cancels much of the common discretisation drift. This needs the
   detachment rule first, and a control population — it does not need a
   divergence-free field.
2. ✅ **Drop det F from the void definition entirely and count particles.** A
   void is then "a region the surviving particles do not reach", measured by the
   existing density/KDE machinery on the *detached-excluded* ensemble. ⚠️ This
   is in fact closer to what §B.6d claimed as the contribution — a
   particle-density deficit — and it never needs $\det F$ to be 1.
3. ⚠️ **Project the velocity onto a discretely divergence-free space** before
   tracking. Correct but expensive, invalidates comparison with every existing
   result, and is a research project in itself.

✅ **Recommendation: route 2.** It keeps the contribution intact, needs no new
field, and sidesteps the obstacle rather than fighting it. Route 1 is a useful
diagnostic to keep. ⚠️ `ln_J` should be **retained and reported as a measured
property of the FOM discretisation** — that is a publishable observation in its
own right, and it is the honest reason the method does not use $\det F$.

#### ✅ Π₁ passed its two tests

```
  Pi_1 min (alive)  median 0.8151   p01 0.0000   p02 0.7915   max 0.9111
  spread  111.8 % of the median    ✅ usable, not the 2.9 % band of the smoke test
  at clip 0 particles at tau_ratio >= 9.999    ✅ the clip never binds
```

✅ **Both open questions are answered favourably.** Π₁ develops real spread once
particles traverse the shear layer, and the `clip(…, 0, 10)` never binds.

⚠️ **But the threshold sweep is degenerate, and that is informative:**

```
  Pi_1_crit  0.10 .. 0.70  ->  1.260 %  (3,588 particles, IDENTICAL)
  Pi_1_crit  0.80          -> 11.370 %
  Pi_1_crit  0.90          -> 98.072 %
```

⚠️ `tau_min == 0` **exactly** for 3,588 particles (1.26 %), and the rest sit in a
tight band 0.79–0.91. So the distribution is **bimodal, not continuous**: a
hard-zero population plus a narrow cluster. Any threshold in 0.1–0.7 selects
exactly the zero population. ✅ **That is a usable binary classifier**, but it is
*not* a tunable continuum, so §B.6f step 4's "stability over a decade of
threshold" is satisfied trivially rather than meaningfully.

⚠️ **The zeros need explaining before Π₁ is trusted.** `tau_ratio = 0` means
`sigma_eq = 0` at that node, i.e. a node the Norton route gave zero flow
stress — plausibly outside the active material or at a level-set boundary.
Only 1,184 of the 18,787 worst-`ln_J` particles overlap them, so they are a
distinct population. ⚠️ **Until that is identified, "1.26 % detached" must not
be read as a void fraction.**

### B.6j ✅ Would HCT fix det F? Partly — and it is worth trying

Measured 2026-10-20 on D2 snapshot 34, comparing the divergence of the raw P1
field against the **SPR-recovered nodal gradient**, which is the field the HCT
cubic is built *through*:

| | raw P1 (what the det F run used) | SPR-recovered nodal |
|---|---|---|
| relative \|div u\| median | 4.87e-02 | **3.31e-03** (≈15× lower) |
| p90 | 5.81e-01 | **5.71e-02** (10× lower) |
| frac > 0.01 | 79.3 % | **32.5 %** |
| frac > 0.1 | 36.0 % | **6.4 %** |
| raw \|div u\| median | 2.31 1/s | **0.72 1/s** (3.2× lower) |

⚠️ **HCT enforces C⁰/C¹ *continuity*, never a divergence constraint** — verified
in `jaxtrace/gpu/recovery/hct3d.py`, which claims exactness for linear fields
and quadratic precision at face centroids, and nowhere a solenoidal condition.
There is **no divergence-free projection anywhere in the codebase**. So HCT
cannot make $\det F \equiv 1$ by construction.

⚠️ **SUPERSEDED IN PART BY §B.6k.** The claim below assumes $\ln J$ integrates
the divergence *of the reconstruction*. ❌ **It does not**: `div_u` is
precomputed from the raw P1 gradient before recovery is even built, so HCT
cannot change det F as currently wired. The 15× figure is a property of the
recovered gradient and remains the reason this is worth implementing properly.

~~But it should still improve $\det F$ substantially, because the quantity
that enters $\ln J$ is the divergence *of the reconstruction*, and SPR recovery
alone already reduces it by an order of magnitude.~~ A 15× smaller integrand over
the same 8000 steps would move the median $|\det F - 1|$ from 3.2e-03 toward
~2e-04, which **would clear the void fraction of interest**.

#### ⚠️ Two blockers before this can be tested

1. ❌ **HCT is steady-only.** `run_tracking.py` builds the reconstruction under
   `if n_frames == 1`, and our D2 case has **166 frames**. With a sequence it
   prints *"SKIPPED: velocity_sequence has N frames; only single-frame (steady)
   fields are supported today. Falling back to raw P1"*. ✅ It warns rather than
   failing silently — but it means **turning HCT on in the per-case script would
   have had no effect on the det F run.** The D-family scripts do not set
   `--recovery-method` at all.
2. ⚠️ **Cost.** HCT stores (n_elements, 4 subtets, 20 coeffs, 3) float32 =
   10.8 M × 240 B ≈ **2.6 GB per frame**. At 166 frames that is ~430 GB,
   impossible. Extending HCT to time-dependent fields therefore needs either
   on-the-fly rebuild per snapshot or a far more compact representation.

✅ **Cheap decisive test available:** run D2 with `--vel-range 34 34` (a single
snapshot, steady) **with and without** `--recovery-method hct_cubic`, 8000 steps
each, and compare $\det F$. That isolates the reconstruction's effect on $\ln J$
at a cost of ~1 h per arm instead of a 430 GB redesign. ⚠️ It answers a
*different* question from the production run (steady vs time-dependent field),
so it bounds the gain rather than delivering it.

#### ✅ Independent corroboration of the root cause

`rom_out/velocity_diagnostics/divergence_summary_cylindrical_*.csv`, from
earlier ROM work, measured the **FOM's own** divergence on four cases: median
|div u| of **1.64 / 1.68 / 2.99 / 2.41 1/s** for $r<10$ mm, with p99 reaching
75–151 and a max of 478–855. ✅ **That independently confirms §B.6i** — my
measured 2.31 1/s on D2 sits squarely in that range, and it was recorded before
any of this detachment work. ⚠️ **The non-zero divergence is a property of the
FOM discretisation, not of the detachment accumulator**, and it is not specific
to the D family.

### B.6k ⚠️ The HCT test ran and measured NOTHING — a design flaw in my own test

Two arms on cylA, 100 steps, 293,455 particles, both exit 0:

| | raw P1 | hct_cubic |
|---|---|---|
| median \|det F − 1\| | 9.036e-05 | **9.036e-05** |
| p99 | 7.561e-04 | 7.566e-04 |
| Π₁ min median | 0.9007 | 0.9007 |

✅ HCT genuinely built — `c1_taylor`, Bernstein coefficients
(3,050,196 × 4 × 20 × 3) = **2928 MB**, assembled in 2.78 s. Not a silent
fallback; the gradient-recovery lines confirm it.

❌ **And yet det F is IDENTICAL to five significant figures.** That is not a
null physical result — it means **the test could not have detected an effect.**

#### The reason, found in the code

`div_u` is computed inside `build_damage_fields` from the **raw P1 velocity
gradient** (`velocity_gradient_per_element` → `tr(D)` → `elements_to_nodes`),
and the damage stack is assembled at `run_tracking.py:2391` — **before** the
gradient recovery is built at line 2423. The kernel then samples `div_u` as a
plain **nodal scalar field** via `interpolate_scalar_single`.

⚠️ **So `ln_J` integrates the divergence of the raw P1 field no matter what
`--recovery-method` says.** The reconstruction affects the *velocity* used to
move particles, but the *divergence* the accumulator sees is precomputed from
P1 and is completely independent of it. My prediction in §B.6j — that HCT should
reduce det F by ~15× — was untestable by this experiment, and I should have
checked the data path before running it.

#### ✅ What would actually test it

`div_u` must be computed **from the same reconstruction that moves the
particles**. Two options:

1. ✅ **Analytic divergence of the HCT cubic**, evaluated in-kernel at the
   particle position. The Bernstein form has an exact gradient, so
   $\nabla\!\cdot u$ is available without a new field — this is the correct
   answer and it also removes the nodal-scalar round-trip for free.
2. ⚠️ **Rebuild `div_u` from the recovered nodal gradient** (`tr` of SPR's
   tensor, scattered to nodes). Cheaper to implement and ~15× lower divergence
   per §B.6j, but it is still a *projected* field, so it measures the
   gradient's divergence rather than the interpolant's.

⚠️ **Until one of these is implemented, the question "does HCT fix det F?" is
OPEN** — §B.6j's 15× is a property of the recovered gradient, not a measured
det F improvement.

#### ⚠️ A separate finding the test did produce

`max |det F − 1|` is **90.7 at 100 steps** (P1: 70.5), versus 108.7 at 8000
steps on D2. ✅ **So the pathological tail is NOT an accumulation effect** — it
is present almost immediately and barely grows. ⚠️ **I then wrongly inferred "a few bad
elements".** ❌ **Measured and refuted** — see §B.6l. The deviation is
DISTRIBUTED across essentially the whole population, so there is nothing to
exclude, and exclusion was never the right idea in any case.

⚠️ Note also that cylA's median is 9.0e-05 against D2's 3.2e-03 — a 35×
difference between cases. The det F error is strongly **case- and
mesh-dependent**, so no single floor figure can be quoted for the method.

### B.6l ❌ The det F error is DISTRIBUTED, not a few bad elements

Measured on the full D2 run, 284,810 alive particles:

| threshold | particles over it | share |
|---|---|---|
| \|det F − 1\| > 1e-4 | 278,311 | **97.72 %** |
| > 1e-3 | 224,532 | **78.84 %** |
| > 1e-2 | 83,670 | **29.38 %** |
| > 1e-1 | 18,630 | 6.54 % |
| > 1.0 | 329 | 0.116 % |
| > 10 | 18 | 0.006 % |

Percentiles: p50 3.16e-03, p75 1.72e-02, p90 7.33e-02, p95 1.20e-01,
p99 2.64e-01, p100 1.09e+02.

❌ **This refutes the "few pathological elements" reading in §B.6k.** The
distribution is smooth and heavy-tailed with **no isolated outlier population**:
97.7 % of particles carry a deviation above 1e-4 and nearly 30 % above 1e-2.
⚠️ **Element exclusion is therefore both useless and wrong** — there is no small
set to remove, and removing elements would discard real material.

✅ **The correct conclusion: the error is a systematic property of the P1
discretisation**, exactly as the field-level measurement in §B.6i says
(|div u|/|D| median 4.87e-02, 79.3 % of elements above 0.01). ✅ **So the fix must
be a better velocity representation, not data rejection** — which is the HCT
route, and is what §B.6m implements.

### B.6m ⚠️ cylA baseline, full 2684 steps — and a reporting bug of my own

Full cylA, `field` divergence + raw P1, 293,455 particles, **0 escaped**,
exit 0 in 1691 s (Stage 7 = 1577.9 s, 1.70 step/s).

```
ln_J       median -0.00026   p01 -0.1208   p99 +0.0801   min -1814.51   max +81.26
|det F-1|  median 1.865e-03  p99 2.759e-01
fraction with |det F - 1| > 1e-2 : 12.67 %
⚠️ GATE NOT PASSED
```

#### ⚠️ My reporting said the opposite, and the logic was wrong

The run printed **"✅ det F is 1 to within the interpolation error"** beside a
**max of 1.96e+35**. Two faults, both mine:

1. ❌ **The gate tested only the median.** Median 1.87e-03 passed a `> 1e-2`
   check while p99 sat at 2.76e-01 and **12.67 % of the population** exceeded
   1e-2. ✅ Now the verdict tests **p99 and the population fraction**, because a
   void fraction of interest is O(1e-2…1e-1) and what matters is how much of
   the ensemble reaches that order — not where the middle sits.
2. ⚠️ **det F was quoted without `ln_J`.** `ln_J` spans **−1814 … +81**, and
   `exp()` of that under/overflows to 0 and 2e35. So the 1e35 is a *reporting*
   artefact of ~0.07 % of particles (|ln_J| > 10: 213 of 293,455), not a
   catastrophic physical failure. ✅ Both reporters now print `ln_J` with its own
   percentiles, since **`ln_J` is the interpretable quantity** and det F loses
   resolution at both ends.

✅ **Net: the conclusion is unchanged but now correctly stated.** The field is
not incompressible enough — `|det F − 1|` reaches the void's own order for
12.7 % of particles on cylA, and 29.4 % on D2 (§B.6l).

#### ✅ Two things the baseline settles

- **cylA loses NO particles** (293,455/293,455 alive), unlike D2's 1.1 %. ✅ So
  this case isolates the divergence question from the escapee question
  entirely — a cleaner test bed than D2 for everything that follows.
- ⚠️ **Π₁'s sweep is degenerate here too**, and more extremely: thresholds
  0.10–0.40 all select **117 particles (0.04 %)**, the `tau_min == 0`
  population. Median 0.8216, p01 0.8076 — a band of ~1 % around the median
  with a hard-zero spike. ⚠️ **The same bimodal structure as D2** (§B.6i), so it
  is a property of how Π₁ is built, not of one case. The `spread 111.65 %`
  line is misleading: it is `(max−min)/median`, which the hard zeros inflate.
  ⚠️ **That spread metric needs fixing before it is quoted anywhere.**

### B.6n ✅ Performance audit — the analytic divergence is FREE, and export is already async

#### ✅ The jacfwd divergence costs nothing. Measured, not estimated.

| arm | Stage 7 | step/s |
|---|---:|---:|
| `field` (precomputed nodal P1 div) | 1577.9 s | 1.70 |
| **`analytic` (jacfwd on the interpolant)** | **1561.1 s** | **1.72** |

✅ **1 % FASTER, i.e. free within noise.** The concern that differentiating
in-kernel would cost per-step time is resolved: it does not. The reason is that
`interpolate_velocity_single` is **already called 4× per step per particle** for
the RK4 stages, so 3 forward-mode tangents on one of those calls disappears
into work the kernel was doing anyway — and it needs **no extra element
search**, which is where the real cost lives.

⚠️ **So the pre-stored 10-coefficients-per-element option (412 MB/frame on D2)
is not worth building.** It would add a second hand-derived implementation of
the HCT gradient that can drift from the evaluator, to save 0 %.

#### ✅ Export is ALREADY asynchronous and multi-threaded — and deliberately 1 worker

`VTKHDFExportThread` (`jaxtrace/io/vtkhdf_writer.py`) runs on a background
thread with a 200-deep queue. ⚠️ **It is single-worker on purpose**, measured
2026-10-16, for two independent reasons:

1. **More workers would not help.** Checked against this cylA run: GPU step is
   **582 ms**, export payload 13.4 MB, write time **119 ms** → a **20 % duty
   cycle** with 463 ms spare. The writer hides entirely behind the GPU.
2. ❌ **More workers would not be safe.** `write_step` appends to **one** HDF5
   file and mutates shared state — `Steps/NSteps`, `PointOffsets`, and every
   `PointData` dataset length, with each step's offset derived from the
   *current* length. Concurrent writers would interleave offsets and corrupt
   the archive. And `h5py` here is a **non-MPI build**, so HDF5's global lock
   would serialise parallel writes anyway: corruption risk for zero throughput.

✅ **So the throughput knob is the export CADENCE, not the thread count.**

⚠️ **And that is the real finding of this audit.** cylA ran at `EXPORT_FREQ=1`
and produced **26 GB** of particle output for a 2684-step run. The D2 indicator
runs use `EXPORT_FREQ=50`. At 50 the same case writes ~0.5 GB and transfers
0.23 GB instead of 11.7 GB. ⚠️ **For a det F / detachment study nothing needs
per-step particle output** — the quantities of interest are the final
accumulators in the npz. ✅ `EXPORT_FREQ=50` (or `NO_EXPORT=1`) should be the
default for these runs.

#### ⚠️ GPU is at 100 % — the kernel is the bottleneck, and it is already optimised

Measured during the run: **GPU utilisation 100 %**, 8.7 GB. So the 582 ms/step
is compute, not transfer, and not export. At 293,455 particles that is
**1982 ns per particle per step** for 4 searches + 4 interpolations + 3 damage
gathers.

⚠️ **The search path has already been exhaustively optimised** — see
`docs/dfamily/13_throughput_roadmap.md`, which measured and **rejected** nine
candidates (finer cells, dynamic trip counts, level-loop reordering, two-tier
kernels, per-lane early exit, 1×1×1 neighbourhoods) and states a **realistic
ceiling of 1.13×**, already implemented via exact tet-box registration. ✅ The
one candidate still open is **particle sorting into homogeneous batches**
(~1.35× on a synthetic proxy, never measured for real).

✅ **Conclusion: there is no large throughput win available in the kernel**, the
divergence work is free, and the only easy gain is cutting export cadence.
⚠️ I should not have implied the transfer was the bottleneck before checking
GPU utilisation — it is not.

### B.6o ❌ DOES HCT FIX det F? NO. The answer, finally measured correctly.

Three arms on cylA, 2684 steps, 293,455 particles, 0 escaped, all exit 0.

| arm | reconstruction | divergence | median \|det F−1\| | p99 | frac > 1e-2 | Stage 7 |
|---|---|---|---:|---:|---:|---:|
| 1 | raw P1 | `field` | **1.87e-03** | 2.76e-01 | **12.67 %** | 1577.9 s |
| 2 | raw P1 | `analytic` | 9.51e-03 | 9.46e-01 | **48.45 %** | 1561.1 s |
| 3 | **hct_cubic** | `analytic` | 5.59e-03 | 1.99e+00 | **33.42 %** | 1564.5 s |

❌ **The gate fails in all three.** HCT does not rescue it.

#### ✅ But the comparison that matters is arm 3 vs arm 2, not arm 3 vs arm 1

Arms 2 and 3 measure **the same quantity** — the divergence of the interpolant
actually moving the particles. On that fair comparison HCT **does** help:

- median: 9.51e-03 → **5.59e-03** (1.7× better)
- fraction over 1e-2: 48.45 % → **33.42 %** (1.45× better)

⚠️ **Arm 1 is not comparable to either.** Its `field` divergence is the P1
element-wise `tr(D)` **scattered to nodes and re-interpolated linearly**, which
*smooths* the divergence before integrating it. That is why it looks best: the
smoothing hides the error rather than reducing it. ✅ **Arm 2 is the honest
measurement of what the P1 tracking field actually does**, and it is 48 % of
particles over 1e-2 — four times worse than the nodal-smoothed number we had
been quoting since §B.6i.

❌ **So the situation is worse than previously recorded, not better.** The
§B.6i/§B.6m figures (3.2e-03 on D2, 1.9e-03 on cylA) understate the true
divergence error because they were computed from a smoothed field.

#### ⚠️ And HCT's tail is WORSE even though its bulk is better

p99 rises 9.46e-01 → 1.99e+00 and p01 of `ln_J` goes −0.83 → −3.63. ✅ Expected:
a C¹ cubic has larger derivatives between nodes than a piecewise-constant
gradient, so where the field is badly behaved the cubic's divergence is bigger.
⚠️ **HCT improves the median and the bulk fraction while making the extremes
worse** — it is not a uniform improvement, and a void criterion reading the tail
would be hurt by it.

#### ✅ Cost: confirmed negligible

1564.5 s vs 1561.1 s — HCT costs **0.2 %** in Stage 7, plus 2928 MB for the
Bernstein array and 2.78 s to assemble. ✅ The expensive part of HCT is memory,
not time.

#### ❌ Verdict on the det F route

Three independent measurements now say the same thing: **the discrete FOM
velocity field is not divergence-free enough to support a void defined as a
particle-density deficit from det F.** Raw P1 is 48 % over the void's own order;
HCT reduces that to 33 %; neither is usable. ⚠️ A 15× reduction in the *recovered
gradient's* divergence (§B.6j) did **not** translate into a 15× reduction in the
*interpolant's* divergence — the two are different quantities and I conflated
them.

✅ **What remains open, in order of promise:**

1. ⚠️ **Accept that det F is not the observable** and define detachment's effect
   some other way. ❌ But §B.6d's contribution claim rests on a density deficit,
   so this needs the claim restated, not just the metric swapped.
2. ✅ **A genuine divergence-free projection** (§3 of the user's question, parked):
   Scott–Vogelius is the natural fit since it uses the **same Alfeld split** HCT
   already builds. This is the only route that attacks the cause.
3. ⚠️ **Π₁ alone as a binary classifier**, with no det F at all. The sweep is
   degenerate (§B.6m: thresholds 0.1–0.4 all select the same 117 particles), so
   this needs the `tau_min == 0` population explained first.

### B.6p ✅ Divergence-free methods compared — on OUR meshes, with measurements

#### The mesh facts that constrain every option

Verified on cylA snapshot 159 (same structure on every family):

| | |
|---|---|
| cells | **100 % tetrahedra** (VTK type 10), no mixed elements |
| velocity | **P1 nodal** (`Displacement`, 3 comps at nodes) |
| pressure | **P1 nodal** (`Pressure`, 1 comp at nodes) — ⚠️ so the FOM is P1/P1 |
| extra | `Temperature`, `LEVEL` nodal; `Stress`, `Strain` per cell |
| boundary | ⚠️ **no boundary-node extraction exists in the codebase** |
| Alfeld split | ✅ **already built** by `build_alfeld_geometry` for HCT |

⚠️ **Two of these dominate the comparison.** Pure tets make every tet-based
method applicable. ❌ But no boundary-node set exists, and **every global
projection needs boundary conditions** — that is unbudgeted work in options 1
and 3 below.

#### ✅ Option A — Scott–Vogelius. Your first choice, and I agree.

**Why it fits us specifically:** SV is $P_k$ velocity / $P_{k-1}$ discontinuous
pressure on a **barycentrically refined (Alfeld) mesh**, where it is
**pointwise divergence-free**. ✅ **We already build exactly that split** — HCT's
`build_alfeld_geometry` computes the parent centroid, sub-tet vertices, edge
midpoints and sub-tet volumes per element. The geometric prerequisite is done
and cached.

| | |
|---|---|
| accuracy on our meshes | ✅ **pointwise div-free**, the only option that is |
| effect on det F | ✅ should drive `ln_J` to integration error alone |
| speed | ⚠️ one global solve per snapshot; ~10.8 M elements on D2 → **the dominant cost** |
| effort | ⚠️ **large**: assemble a Stokes-like projection, pick an iterative solver + preconditioner, extract boundaries |
| modifications | new module; HCT's Alfeld geometry reused; velocity representation changes to P2-on-sub-tets |
| risk | ⚠️ SV needs $k\ge 2$ for stability (P2/P1-disc is the usual choice). Our data is P1, so **we would be inventing the P2 field**, not projecting one. |

⚠️ **The honest caveat:** SV gives a div-free field *in its own space*. Feeding
it P1 data means solving a projection problem whose answer is only as good as
the P1 input — it cannot recover information the FOM never computed. ✅ It would
still remove the *discretisation* divergence, which is the 48 % problem.

#### ⚠️ Option B — Constrained least-squares. Measured: 1.57×, and that is not enough.

Your second choice, and I tested it rather than estimating. The natural form is
to constrain the recovered **nodal gradient** to be trace-free.

✅ **The constraint itself is trivial and nearly free:** project each nodal
gradient onto the trace-free subspace, $G_a \leftarrow G_a - \tfrac{1}{3}\mathrm{tr}(G_a)I$.
Measured on cylA: `|tr Gc|` max **3.1e-13** (exactly satisfied), and the
gradient moves by only **7.9e-04** relative at the median — a small correction,
so the fit stays faithful to the data.

❌ **But it cannot fix the divergence, and here is the measurement.** The
`vertex_taylor` interpolant's divergence splits into two terms:

$$\nabla\!\cdot u = \underbrace{\sum_a \nabla N_a\cdot(v_a + G_a(x-x_a))}_{\text{TERM 1, from VELOCITIES}} + \underbrace{\sum_a N_a\,\mathrm{tr}(G_a)}_{\text{TERM 2, from GRADIENTS}}$$

At element centroids on cylA (400 k sample):

| term | median | p99 |
|---|---:|---:|
| TERM 1 (velocities) | **1.173** | 150.9 |
| TERM 2 (gradients) | 0.148 | 47.1 |
| total | 1.836 | — |

⚠️ **TERM 2 is only 13.6 % of TERM 1.** Constraining the gradient zeroes TERM 2
and leaves TERM 1 **untouched**, giving a reduction of **1.57×** — median
1.836 → 1.173.

❌ **1.57× does not close a 48 % gap.** Scaling arm 2's fraction-over-1e-2 by
that factor lands near 30 %, i.e. roughly where HCT already is (33 %), which
also failed the gate. ⚠️ **And for raw P1 the method does not apply at all**:
there is no recovered gradient in that path, so there is nothing to constrain.

✅ **Verdict: cheap (hours, ~20 lines in `gradient_recovery`), zero risk, and
genuinely useful as a diagnostic — but insufficient as the fix.** Worth doing
only if combined with something that attacks TERM 1.

#### ⚠️ Option C — Helmholtz/Hodge projection

| | |
|---|---|
| accuracy | ⚠️ div-free **in the discrete norm of the Laplacian used** — not pointwise. That detail decides whether it helps us, and it is exactly the P1 weak-sense property we already have. |
| effect | ⚠️ **uncertain**: it enforces the same weak constraint the FOM already satisfied. It may barely move the pointwise error. |
| speed | one Poisson solve per snapshot — cheaper than SV, still global |
| effort | medium; needs boundary conditions ❌ which we do not have |
| risk | ⚠️ high chance of measuring ~1× after real work |

#### ❌ Option D — Raviart–Thomas / BDM

Div-free by construction in H(div), structurally the correct answer — ❌ but it
requires **changing the FOM's velocity discretisation**, not post-processing it.
Out of scope: we do not own the solver.

#### ❌ Option E — Stream function / vector potential

$u=\nabla\times\psi$ is identically div-free. ⚠️ In 3D $\psi$ is a vector field,
recovering it from $u$ is itself an ill-posed global solve, and boundary
handling is awkward. Worse effort-to-benefit than SV with no accuracy advantage.

#### ✅ Recommended order — and one cheap test first

| rank | option | gain | effort | verdict |
|---|---|---|---|---|
| **1** | **Scott–Vogelius** | ✅ pointwise div-free | ⚠️ large | the only option that attacks the cause; Alfeld split already built |
| 2 | Constrained LSQ | ⚠️ **1.57× measured** | ✅ hours | do it for the diagnostic, not as the fix |
| 3 | Helmholtz/Hodge | ⚠️ uncertain | medium | likely ~1× — same weak constraint we already have |
| ❌ | RT/BDM | ✅ exact | ❌ needs FOM change | out of scope |
| ❌ | Stream function | ✅ exact | ❌ ill-posed in 3D | dominated by SV |

✅ **Before building SV, run one cheap test (~1 day):** take ONE cylA snapshot,
assemble the SV projection on a **small sub-mesh** (say 10⁵ elements), and
measure the resulting pointwise divergence. ⚠️ If it does not reach ~1e-6
relative on a sub-mesh, it will not on 10.8 M elements either, and we learn that
for a day instead of weeks. ✅ This also surfaces the boundary-condition work
early, which is the part most likely to be underestimated.

### B.6q ⚠️ SV feasibility test: the LOCAL version gives 1.00×. Answers Q3 though.

Run on cylA snapshot 159, 100,000 contiguous elements
(`scripts/damage/test_scott_vogelius_submesh.py`).

| | before (raw P1) | after (local Alfeld projection) |
|---|---:|---:|
| \|div u\|/\|D\| median | 2.014e-03 | **2.014e-03** |
| p90 | 3.824e-02 | 3.824e-02 |
| frac rel > 1e-2 | 35.75 % | **35.75 %** |

❌ **Reduction 1.00×.** The centroid velocity moved by only **6.3e-08**
relative — the least-squares solution is essentially the P1 value it was
regularised toward.

#### ⚠️ Two bugs of mine on the way here, both caught by sanity tests

1. ❌ **A transposed shape-function gradient.** I built `grad N` as
   `transpose(inv(J))`; with `J = [e1 e2 e3]` as **columns**, `grad_x N_i` is
   the i-th **row of `inv(J)` used as-is**. ⚠️ `tet_shape_gradients` warns about
   precisely this: *"transposing here is a subtle and silent error (it only
   shows up on skewed elements, and it breaks incompressibility)"*. The bug made
   the projection look **47× worse** (2.0e-03 → 9.4e-02), and I nearly reported
   that as a method failure. ✅ Caught by testing an exactly-representable field:
   a linear div-free field gave `|div| = 1.3e+04` instead of ~0, and `u=(x,y,z)`
   gave 1.07 instead of 3. After the fix: 4.1e-12, and 3.000000 exactly.
2. ❌ **A wrong explanation.** I hypothesised that the volume-weighted total
   divergence is invariant to an interior DOF by the divergence theorem.
   ⚠️ **Measured false**: varying `u_vc` changes `Σ div_s·V_s` (0.0 → 1.6e-02 →
   1.44 for P1-mean, random, 100× random). So the DOF *can* change the total;
   the reason the fit does not use it is **not** a conservation identity, and I
   do not currently have a verified explanation.

⚠️ **Honest status: the local projection does not work on our data, and I have
not established why.** The most likely candidate is the regularisation
(`lam = 1e-6 * w`) dominating a nearly-singular normal matrix, which would mean
the test under-explores the method rather than refuting it. ✅ **That is worth one
more measurement** — sweep `lam` and check the condition number of `A` — before
concluding anything about SV proper.

#### ❌ What this does NOT establish

This is a **local, per-element** projection with one extra DOF — the cheapest
thing in the SV family, not SV. ⚠️ **A global SV solve couples elements through
the continuity of the velocity space and uses a P2 velocity space (SV needs
k ≥ 2); it can only do better on divergence while costing more.** So this result
rules out the *easy* version, not the method.

#### ✅ Q3 ANSWERED: PREBUILT, decisively — and this part is solid

The cost measurement is independent of whether the projection works, because it
is the same linear algebra either way: **1.63 µs/element**.

| case | elements | snapshots | solve/snapshot | total solve | store/snapshot | store all |
|---|---:|---:|---:|---:|---:|---:|
| cylA | 3,050,196 | 1 | 5.0 s | 0.1 min | 34.9 MB | 0.03 GB |
| **D2** | 10,807,677 | 166 | **17.6 s** | **48.7 min** | **123.7 MB** | **20.05 GB** |

✅ **Prebuild it, do not do it in-step.** Three reasons, all measured:

1. **17.6 s per snapshot against a 1710 s mesh load** — the projection is ~1 %
   of a stage we already pay. 48.7 min total on D2 is real but modest against a
   3.4 h run.
2. ✅ **The stored object is tiny: 3 floats per element** (the centroid
   velocity), **124 MB/snapshot** — ⚠️ **24× cheaper than HCT's Bernstein array**
   (240 B/element, 2928 MB on cylA alone). A per-snapshot rebuild is unnecessary
   when the result is this small.
3. ⚠️ **20 GB for all 166 snapshots of D2 does not fit in GPU memory**
   alongside the existing 28.4 GB footprint (§B.6n), so the realistic pattern is
   **prebuild to disk once per case, then stream or hold one snapshot at a
   time** — exactly how the velocity stack is already handled.

✅ **And in-step is ruled out on a stronger ground than cost:** the projection is
a *per-element solve*, not a per-particle evaluation. Doing it inside the step
would redo identical work for every particle in the same element — 288,000
particles over 10.8 M elements means the same 3×3 solve repeated many times per
element per step.

### B.6r ✅ Detachment implemented end to end — and the threshold, measured

`--detach` with the rule and threshold isolated in
`jaxtrace/damage/detachment.py`. ⚠️ **One file owns the policy** because the
threshold is the method's one free parameter and is not settled; the kernel
receives a resolved float and an integer mode and encodes no physics.

**Rule:** $\Pi_1 = \sigma_{eq}/\tau_{yield}^{ref} < \Pi_1^{crit}$ (Buck &
Langerman's yield-stress group).
**Kinematics:** `v_adv` (default) — the particle is carried by the workpiece's
rigid translation along the travel axis. ✅ Parameter-free: `v_adv` is already a
per-case quantity. Also `freeze`, and `flag-only` as the **diagnostic** mode
that records detachment without changing any trajectory.

Implemented as a **post-step override** modelled on the existing
ballistic/freeze escape latch, not inside the RK4 kernel: the trigger
(`tau_min`) is already sampled, the RK4 graph is untouched, and the kernel stays
byte-identical when detachment is off.

⚠️ **Three details that would be silent bugs otherwise:**
- Detachment runs **after** the escape latch — an escapee has no valid
  `tau_min` and would latch a detachment from a stale accumulator.
- A detached particle advances from its **pre-step** position; the kernel has
  already moved it with a field it no longer follows.
- Its host element id is **kept**, unlike an escapee's, so its indicators keep
  being sampled and detachment stays distinguishable from escape.

#### ✅ THE THRESHOLD: 0.5 for both cases, and it is not a tuned number

Measured $\Pi_1^{min}$ distributions on the two completed full runs:

| | cylA (2684 steps) | D2 (8000 steps) |
|---|---|---|
| exact zeros | 117 (**0.040 %**) | 3,588 (**1.260 %**) |
| smallest NONZERO | 0.8017 | 0.7859 |
| p1 / median | 0.8076 / 0.8216 | 0.0 / 0.8151 |

❌ **Π₁ is NOT a tunable continuum.** The distribution is a hard-zero spike plus
a band ~1.5 % wide. The detached fraction as the threshold moves:

| $\Pi_1^{crit}$ | cylA | D2 |
|---|---:|---:|
| 0.10 – 0.78 | **0.040 %** | **1.260 %** |
| 0.80 | 0.040 % | 11.37 % |
| 0.82 | 41.01 % | 63.48 % |
| 0.85 | 99.38 % | 97.50 % |

⚠️ **The fraction jumps 1.3 % → 63 % → 98 % over a 0.03 change.** So there is no
meaningful tuning range: the threshold is a **switch** between the hard-zero
population and essentially the whole mesh.

✅ **Hence $\Pi_1^{crit} = 0.5$ for both cases.** Any value in (0, 0.785) gives
an identical answer, so 0.5 sits mid-way through that dead zone — ✅ **one value
for two very different cases, with no per-case tuning**, which is the honest
thing to present. ✅ **And §B.6f step 4's "stability over a decade of threshold"
is satisfied**: 0.1 → 0.78 is more than a decade with a constant answer.

⚠️ **What it selects, and the caveat for the meeting.** $\Pi_1 = 0$ means
$\sigma_{eq} = 0$ — material carrying **no flow stress**, the cleanest possible
detachment signal. ⚠️ **But that population has not been independently
identified.** It is plausibly outside the active material or at a level-set
boundary, and until that is checked **"1.26 % detached" must not be presented as
a void fraction.** Present it as *"the criterion fires on a physically
identifiable population, and here is where"*, not as a quantitative prediction.

#### ⚠️ A real finding from the first launch: cylA's `v_adv` is not auto-detected

cylA keeps its `som.fix` under `data/0eule/`, not in the case parent where
`read_case_solver_params` looks, so auto-detection returns `None`. ✅ **The
policy validation refused to guess and failed in 1 second** rather than after a
mesh load — which is exactly what the fail-early design is for. The value, read
from the `Faces` constraint `111 6.6666666667e-03`, is **6.667e-03 m/s =
400 mm/min** and is now passed explicitly. ⚠️ D2's auto-detection works
(`v_adv=0.01 m/s`), so this is a cylA path quirk, not a general break.

### B.6s ⚠️ After the `tau_min` fix: cylA detaches NOTHING. The threshold is unresolved.

Re-run with the §B.6r guard (commit `2d29bbc`), cylA 2684 steps, 293,455
particles, 0 lost, exit 0 in 1557 s:

```
Pi_1 min   median 0.8216   min 0.8017   max 0.9177   (293,455 finite)
detached   0 / 293,455  (0.000 %)   rule Pi_1 < 0.5
```

❌ **The hard-zero population is GONE.** Before the fix cylA reported 117 zeros
(0.040 %) and D2 reported 3,588 (1.260 %); after it, cylA's minimum Π₁ is
**0.8017** and nothing is below 0.5.

⚠️ **So §B.6r's threshold recommendation is void.** Π₁_crit = 0.5 was chosen to
select exactly the hard-zero spike — and that spike was the artefact the guard
removed. The reasoning ("any value in (0, 0.785) gives an identical answer")
was correct arithmetic applied to contaminated data.

#### ✅ What the measurement actually shows now

Π₁ is a **narrow, well-behaved field**: 0.8017 … 0.9177, a 14 % spread with no
outlier population. ⚠️ **There is no natural threshold in it.** Any Π₁_crit has
to be placed inside a continuum, which means:

- below ~0.80 → **nothing detaches** (the cylA result)
- at ~0.82 → roughly half the particles detach
- above ~0.85 → **almost everything** detaches

❌ **This is not a criterion yet.** It is a smoothly varying field that needs a
*physically derived* cut, which is precisely what §B.6f step 3 — the
dimensionless form — was supposed to supply and which is **not yet
implemented**.

✅ **But the panel figure shows Π₁ has real spatial structure**, and that is the
useful result: low Π₁ (dark) concentrates in the stir zone and along the
advancing-side shear band, higher Π₁ (green/yellow) sits in the undeformed far
field and at depth. ✅ **The field knows where the tool worked the material.**
What is missing is the number that turns that structure into a defect call.

#### ✅ The figure (`docs/damage/figs_detach/fig_detach_cylA.png`)

Five panels on the project's standard yz convention (observer at the origin
looking +x, +y left). ⚠️ Wake clip **scaled, not copied**: D2's 7.4 mm is
0.2523 of its downstream reach, giving **12.70 mm** on cylA, which keeps
268,148 of 293,455 particles. A literal 7.4 mm would have kept 99.5 % and
isolated nothing.

⚠️ **det F structure is visible and it is a warning, not a result**: the
non-unit values cluster exactly in the stir zone, with ±7 % swings where the
material is worked hardest. ✅ That is consistent with §B.6i–§B.6o — the
divergence error is largest where the deformation is largest, i.e. exactly
where a void would form.

### B.6t ✅ Two user observations, both confirmed by measurement

#### 1 ✅ "The detachment mechanism did not work" — correct, and it is a null result

The cylA detachment run is **bit-for-bit equivalent** to a run without
detachment, because **0 of 293,455 particles detached**. The override never
fired, so every trajectory is the plain advection result. ✅ Nothing is broken;
the criterion simply never triggered at Π₁_crit = 0.5 once the `tau_min`
artefact was removed (§B.6s).

⚠️ **This is the honest state and it must be presented as such**: the machinery
is implemented and verified end to end, and the *threshold* is unresolved.

#### 2 ✅ "The original velocity already produces voids, which should not happen in a divergence-free field" — correct, and it is the sharper point

Measured on the cylA wake box (291,137 particles, 160×60 yz occupancy grid):

```
EMPTY cells inside the occupied region : 3,202 of 9,600  (33.4 %)
median occupancy of a NON-empty cell   : 53 particles
```

⚠️ **One third of the interior is empty, against a median of 53 particles per
occupied cell.** Those are large holes, not sampling noise.

❌ **And with $\nabla\!\cdot u = 0$ and full entrainment this is impossible.**
$\det F \equiv 1$ means the particle density is preserved along every
trajectory, so an initially uniform seed cannot develop interior holes. ✅ **The
holes are therefore direct, independent evidence of the divergence error**
measured in §B.6i–§B.6o — they are the *same* defect seen in position space
rather than in $\ln J$.

✅ **This matters more than the detachment null result**, for two reasons:

1. ⚠️ **It invalidates "void = particle-density deficit" as currently
   implemented.** A deficit is already present *without* any detachment rule,
   so a deficit measured *with* one cannot be attributed to detachment. The
   signal and the artefact are the same observable.
2. ✅ **It gives a far cheaper diagnostic than $\det F$.** Interior occupancy on
   a yz grid is a single histogram — no accumulator, no extra field — and it
   reports the same problem in units anyone can read ("a third of the
   cross-section is empty").

⚠️ **So the divergence work is not optional and not deferrable.** §B.6p ranked
Scott–Vogelius first on the grounds that it attacks the cause; this result says
the cause is already corrupting the primary observable, not merely adding
noise to a secondary one.

#### ⚠️ What this means for the threshold discussion

Raising Π₁_crit until particles detach is **possible and worth doing for a
demonstration**, but it must be labelled honestly:

- at ~0.82, roughly half the particles detach
- at ~0.85, almost all do

✅ Either produces a visibly different result for a showcase. ❌ **Neither is a
prediction.** The number would be chosen to make the mechanism visible, not
derived from physics — which is exactly the free-parameter problem §B.6f step 4
inherited from Wang 2021, and exactly what step 3's dimensionless form is meant
to remove.

### B.6b Full texts needed before the derivation — in priority order

⚠️ **The code is ~1 day; the derivation is the work.** §B.3b established that
$Ny^*$ comes from a *transport equation*, inverted to a physical quantity, and
that §B.3's current two-rates-by-inspection is not that. These are what the
derivation needs, ordered by what each one decides.

| # | paper | DOI | what it decides |
|---|---|---|---|
| **1** | **Zhao, C., Pei, X. & Liu, X.** (2022). *Computational investigation on void defects formation and periodic tool-workpiece sliding-to-sticking transition in self-reacting friction stir welding.* **Int. J. Adv. Manuf. Technol.**, 8075–8088 | `10.1007/s00170-022-09259-0` | ⚠️ **Whether our remaining claim survives at all.** It already links a contact-state transition to void formation — our exact causal chain. We need to know whether their criterion is evaluated *locally per material point* or at the interface. |
| **2** | **Mendez, P. F., Tello, K. E. & Lienert, T. J.** (2010). *Scaling of coupled heat transfer and plastic deformation around the pin in friction stir welding.* **Acta Materialia** 58, 6012–6026 | `10.1016/J.ACTAMAT.2010.07.019` | ⚠️ **Whether $\Pi_1,\Pi_2$ should be invented at all.** A formal scaling analysis of the pin region already exists; our groups should be expressed in *its* terms or explicitly reconciled with them. |
| **3** | **Buck, G. A. & Langerman, M. A.** (2004). *Non-dimensional characterization of the friction stir/spot welding process using a simple Couette flow model, Part I.* **AIP Conf. Proc.**, 1283–1288 | `10.1063/1.1766706` | ✅ **A cleaner physical basis for $\Pi_2$.** Gives a *critical radius* — "the location of vanishing material velocity" — from a Bingham **yield-stress** argument, at peak $T\sim 80\%$ of melting. Yield stress beats my thermal-softening ratio. |
| **4** | **Wang, X., Gao, Y.-F., Liu, X., McDonnell, M. & Feng, Z.** (2021). *Tool-workpiece stick-slip conditions and their effects on torque and heat generation rate in the friction stir welding.* **Acta Materialia**, 116969 | `10.1016/J.ACTAMAT.2021.116969` | How a dimensionless stick/slip criterion is *constructed* and tied to spin rate (Hill–Bower contact similarity). 77 cit. Same group as the Wang 2020 we already cite. |
| **5** | **Pei, X. & Dong, P.** (2014). *Shear localisation modelling of friction stir weld formation process.* **Sci. Technol. Weld. Join.** 19, 416–426 | `10.1179/1362171814Y.0000000207` | ⚠️ **Terminology.** "Shear localisation" may be the established name for our "detachment". Needed to avoid renaming an existing concept. |

**Lower priority, for completeness of the Zhao–Liu series** (the only published
line coupling contact state to voids):

| paper | DOI | note |
|---|---|---|
| Zhao, C. & Liu, X. (2021). *An alternative pressure-dependent velocity boundary condition...* **IJAMT**, 1601–1613 | `10.1007/s00170-021-07589-z` | the BC that makes the transition possible; ✅ **an open-access PDF exists** |
| Zhao, C. & Liu, X. (2020). *Computational analysis on weld formation mechanism during self-reacting FSW.* **J. Manuf. Sci. Eng.** | `10.1115/1.4048856` | earliest of the series |

⚠️ **All are self-reacting (bobbin) FSW except Mendez and Buck.** The geometry
differs from our single-shoulder configuration, so what transfers is the
*mechanism and its formulation*, not the numbers.

✅ **No further searching needed.** The O57 deep search (204 papers) already
covered this ground; these are specific PDFs, not a research question.

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
| 4 | **A.4** $\Delta p$ — ⚠️ **promoted**: the ONLY indicator independent of the particles (§A.0b), so the only external check on the tracking itself | low | — |
| 5 | **A.5** indicator table + rank-correlation matrix | low | 1, 3, 4 |
| 6 | ✅ **DONE 2026-10-16** — deep search run; see §B.6. Now: read Zhao (2022) and Wang (2021) in full | two PDFs | user to supply |
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

- **O55** — ✅ **ANSWERED 2026-10-18: YES, conclusively.** Full 16-case family on
  LUMI, 4,092,780 particles pooled: $\rho(\ln\Phi, A_{\dot\varepsilon}) = +1.000$.
  The $\exp(1.5\eta)$ factor contributes **nothing** to the ranking, exactly as
  Wang et al. (2020) **[10]** predicted. The damage apparatus has **one effective
  degree of freedom** — it is an expensive way to measure accumulated strain.
  ⚠️ This is a real result about our method, and it should be stated plainly in
  any write-up rather than buried: two damage laws, agreeing at $\rho=+0.996$,
  are both reducible to $\int\dot\varepsilon\,dt$.
- **O56** — ✅ **MEASURED 2026-10-18.** Density CV in the wake box: median
  **0.273** across 16 cases, max **0.443** (B2). With detachment off the field is
  divergence-free and $\det\mathbf{F}\equiv 1$, so this is seeding/sampling noise,
  not voids. ⚠️ It is also the **floor** against which any future detachment
  result must be judged: a detachment-induced deficit has to exceed a CV of
  ~0.27 before it means anything. B2's 0.443 is the outlier to understand first.
- **O57** — ⚠️ **PARTLY ANSWERED 2026-10-16: YES, substantially.** See §B.6. A
  dimensionless stick/slip criterion (Wang 2021), a non-dimensional critical radius
  at vanishing material velocity (Buck & Langerman 2004), a formal scaling analysis
  of the pin region (Mendez 2010) and a sliding-to-sticking transition tied to void
  formation (Zhao 2022) all exist. **Remaining question:** has anyone evaluated such
  a criterion LOCALLY per tracer particle to produce a mass-deficit void? Requires
  reading Zhao (2022) and Wang (2021) in full — both flagged `needs-fulltext` in
  Zotero.
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
