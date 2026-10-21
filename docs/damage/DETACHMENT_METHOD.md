# The detachment method for FSW void prediction

**Status 2026-10-21.** Self-contained statement of the method: what is claimed,
what is implemented, what is measured, what remains. References cited in text,
DOIs at the end. Companion document: `DIVERGENCE_FREE_PLAN.md`.

---

## 1. The claim, and what the literature says about it

### 1.1 The method in one paragraph

A previously computed FSW velocity field is post-processed by advecting passive
tracer particles. Each particle carries a local criterion along its own path;
when the criterion fires, the particle **detaches** — it stops following the
plasticised flow and is carried only by the workpiece's rigid translation at
the advancing speed $v_{adv}$. The void is then the region the surviving
(attached) particles fail to reach: a **particle-density deficit**.

### 1.2 ✅ Why a detachment rule is logically necessary

With $\nabla\!\cdot\mathbf u=0$ and fully entrained particles,
$\det\mathbf F\equiv1$ — the particle density is preserved along every
trajectory and **a void is impossible by construction**. So either particles
carry mass and momentum and the void arises from the solution (SPH/MPM/PFEM —
a *particle method*), or entrainment is broken by a local rule and the void is
read off the resulting deficit. ✅ **This method is the second, deliberately.**

### 1.3 ⚠️ Do the particles need interaction forces? No — and it is the premise

Passive tracers carry no mass, and inter-particle forces are *already* inside
the constitutive response that produced the field. Adding them would (i)
double-count that physics, (ii) make the tracers obey a momentum balance the
carrier field does not satisfy, and (iii) convert the post-process into
SPH/MPM, which requires **re-solving momentum** — exactly what a post-process
exists to avoid. ⚠️ **Scope limit to state, not hide:** without interaction the
method predicts *where material fails to arrive*, not the void's equilibrium
shape after surrounding material consolidates.

### 1.4 ✅ Novelty, established by literature search

A 187-paper deep search found **no work that advects passive markers through a
precomputed fixed field and lets individual particles detach by a local
criterion**, producing a density deficit.

- ⚠️ **[Zha22] is the closest and is not close.** They *do* use particle
  tracing, but only to compute a strain distribution 30 mm behind the pin. The
  **void criterion is not on the particles**: the CFD model tests the TMAZ
  boundary position against the pin periphery, the shear-localisation model
  compares shear-layer width to $w_{\max}$. Both geometric; their particles
  never detach.
- ✅ **The mechanism IS established elsewhere** — in pore-scale colloid
  filtration, where particles are tracked through a precomputed carrier flow
  and immobilised per-particle by a force/torque balance [Van19, Tor07, Bra10].
  Same mathematical object, different physics. ✅ **Cite as methodological
  precedent**: a recognised technique transferred to new physics, not an
  invention defended from scratch.
- ⚠️ **In that literature the switch is DIMENSIONAL** (a force/torque
  comparison); dimensionless groups only organise regimes. ✅ **So the
  dimensionless form is the genuinely open contribution.**
- ⚠️ **Prior art that must be cited, not implied new:** marker/tracer tracing in
  FSW [Dia20] — our own group, and titled *"Defect formation and material flow
  in FSW"*; marker-and-cell [Har65]; VOF/free-surface and explicit
  damage-evolution void prediction.

---

## 2. The criterion

### 2.1 ❌ What failed: Π₁ as a stress ratio

Following [Buc04], whose critical radius is *"the location of vanishing material
velocity"*, the first attempt was

$$\Pi_1=\frac{\sigma_{eq}(\dot\varepsilon_{loc},T_{loc})}{\sigma_{eq}(\dot\varepsilon_{ref},T_{loc})},
\qquad\text{detach when }\Pi_1<\Pi_1^{crit}.$$

❌ **This is degenerate.** The Norton law is
$\sigma_{eq}=\mu_0(T)\dot\gamma^{\,m}$, so with the *same local temperature* in
numerator and denominator

$$\Pi_1=\frac{\mu_0(T)\dot\varepsilon_{loc}^{\,m}}{\mu_0(T)\dot\varepsilon_{ref}^{\,m}}
=\Big(\frac{\dot\varepsilon_{loc}}{\dot\varepsilon_{ref}}\Big)^{m}.$$

⚠️ $\mu_0(T)$ **cancels exactly** — the thermal softening it was meant to carry
is gone. With $m=0.023$–$0.197$ from our tables, the power law compresses a
measured $\dot\varepsilon$ range of $10^2$–$10^4$ into a Π₁ range of 2.8×.
❌ **Measured consequence:** the criterion fired only on particles already
moving at $0.992\,v_{adv}$, 41+ mm downstream, at export 103 of 109 — a
**deceleration detector**, not a detachment criterion.

### 2.2 ✅ Fix A — the shear-layer gate (implemented)

⚠️ **Any criterion of the form $f(\text{local state})<\text{threshold}$ fires in
the wake**, because that is where material is hot and still. ❌ **But a particle
resting in the wake is CONSOLIDATED, not detached** — it arrived, filled its
place and cooled. ✅ Detachment must mean the material **failed to arrive**,
which is a statement about material *while it is being worked*.

So the criterion is sampled only inside the shear layer, reusing the same test
$t_{res}$ already uses:

$$\text{sample }\Pi\text{ only where }\dot\varepsilon>\dot\varepsilon_{thr}.$$

✅ **Measured effect:** the candidate set falls from *all* 293,455 particles to
**45,652 (15.56 %)**, confined to $\lvert y\rvert<11$ mm and the upper 5.5 mm —
the material that actually passed through the tool.

### 2.3 ✅ Fix B — the Bingham number (implemented, and physically better)

[Buc04]'s actual group is a ratio of **competing rates**, not of states:

$$Bu=\frac{\tau_y}{\mu_{eff}\,\dot\gamma}=\frac{\tau_y}{\tau_{local}} .$$

✅ **As $\dot\gamma\to0$ in the wake, $Bu\to\infty$, not zero** — a low-$Bu$
criterion **cannot** fire on quiescent material. The wake exclusion is built
into the formula rather than bolted on.

**✅ The decisive measurement.** Both criteria were accumulated on *identical*
trajectories (cylA, 2684 steps, 45,652 candidates), gated identically:

| | min | p1 | median | max | spread |
|---|---:|---:|---:|---:|---:|
| $\Pi_1^{\min}$ | 1.3623 | 1.3763 | 1.4773 | 2.0774 | 1.53× |
| $Bu^{\min}$ | 0.1678 | 0.1758 | 0.2511 | 0.3817 | **2.27×** |

rank corr$(\Pi_1,Bu)=+0.358$ — ✅ **they are not the same test.** Mean position
of the lowest 1 % by each:

| | x | y | z |
|---|---:|---:|---:|
| by $\Pi_1$ | 36.75 mm | −3.67 | −1.31 |
| by $Bu$ | **14.59 mm** | +0.68 | **−0.10** |

❌ Π₁ still points downstream even with the gate. ✅ **$Bu$ points at material
just leaving the tool, at the top surface** — where FSW voids and galling
initiate. ✅ **$Bu$ is the criterion.**

### 2.4 ✅ $\tau_y$: sourced, cross-validated, and with its caveat

[Din21] measured AA6061 in hot compression at **exactly our conditions** —
375–575 °C, 0.01–10 s⁻¹, i.e. our alloy, our temperatures, our process. Their
Sellars–Tegart fit (their Table 3: $\alpha=0.0173$ MPa⁻¹, $n=7.25$,
$Q=292$ kJ/mol, $A=1.68\times10^{21}$, RMSE 3.43 MPa) inverts to

$$\sigma=\frac1\alpha\operatorname{arcsinh}\Big[\Big(\frac{Z}{A}\Big)^{1/n}\Big],
\qquad Z=\dot\varepsilon\exp\!\Big(\frac{Q}{RT}\Big).$$

✅ **Verified against their own reported values**: 55.0 MPa computed vs 58.3
reported at 375 °C/0.01 s⁻¹; 10.9 vs 10.2 at 575 °C.

$\tau_y=\sigma_y/\sqrt3$ (von Mises) at the lowest tested rate, the closest
available proxy for a yield threshold:

| T [°C] | 425 | 457 | 475 | 500 |
|---|---:|---:|---:|---:|
| $\sigma$ at 0.01 s⁻¹ [MPa] | 35.1 | 26.6 | 22.9 | 18.7 |
| **$\tau_y$ [MPa]** | **20.3** | **15.4** | **13.2** | **10.8** |

✅ **Independent cross-check:** our own Norton `.mat` tables against [Din21]
over 425–500 °C and 0.01–10 s⁻¹ agree to a ratio of **0.81–1.09**. ⚠️ Two
independently derived constitutive descriptions of AA6061 agreeing within 20 %
validates *our rheology*, not only $\tau_y$.

⚠️ **The caveat that must travel with the number.** A 111-paper search found
**no validated temperature-resolved Bingham yield stress for AA6061/AA6063**;
[Buc04] supplies the framework but no alloy-specific value; hot-compression
studies [Din21, Moh21, Rud20] report rate- and strain-dependent flow curves,
not a separately identified yield threshold; and ❌ **whether a Bingham $\tau_y$
is even well posed near $0.9\,T_{solidus}$ is not settled either way.**
[Su19] gives quasi-static 0.2 % proof stress to 600 °C on **both** 6063-T5 and
6061-T6 as an alternative anchor. ✅ **Report $\tau_y$ as an assumption with its
source and a sensitivity sweep.** $Bu$ scales linearly in $\tau_y$, so the
**ranking** of particles — the result that matters — is unaffected.

---

## 3. Implementation

| piece | where | notes |
|---|---|---|
| policy, threshold, kinematics | `jaxtrace/damage/detachment.py` | ✅ **one file owns the rule**; the kernel gets a float and an int |
| $\Pi_1$, $Bu$, `div_u` fields | `jaxtrace/damage/fields.py` | nodal, built per snapshot |
| accumulators (10 columns) | `benchmark_femuss_comparison.py` | fused into the existing RK4 step |
| kinematics override | `run_tracking.py` | post-step, modelled on the escape latch |

**Accumulator layout:** `[ln_phi, C, t_res, w_bond, A_edot, ln_J, tau_min, pi1, bu_min, bu]`
— running minima *and* instantaneous values for both criteria, so a threshold
can be swept from the archive without re-running.

⚠️ **Details that would be silent bugs:** detachment runs **after** the escape
latch (an escapee has no valid sample); a detached particle advances from its
**pre-step** position (the kernel already moved it with a field it no longer
follows); its host element id is **kept**, unlike an escapee's, so its
indicators keep being sampled.

⚠️ **A fixed artefact worth recording:** `interpolate_scalar_single` returns 0
both for an invalid element and inside the tool. A running *minimum* cannot
distinguish that from genuine zero flow stress and latched 0 permanently —
**100 % of D2's 3,190 escapees reported $\tau_{\min}=0$**. Now fed $+\infty$ on
those steps.

---

## 4. ⚠️ What is NOT yet established

1. ❌ **No threshold is derived.** Both distributions are smooth and unimodal
   with no gap, so any $Bu^{crit}$ is a chosen percentile. ✅ This is what the
   dimensionless form (step 3 below) exists to fix.
2. ❌ **The quantitative void claim is blocked by divergence.** 33.4 % of the
   cylA wake cross-section is already empty *with no detachment applied*, so a
   deficit cannot yet be attributed to detachment. See `DIVERGENCE_FREE_PLAN.md`.
3. ⚠️ **No experimental validation.** Our weld pitch $\gamma=1.0$ mm/rev is
   3.3× beyond [Che23]'s voided case, so we should predict **sound** welds —
   a falsifiable prediction, and the cheapest available check.

---

## 5. Remaining steps

**Step 1 — switch the criterion to $Bu$ (½ day).** The columns exist; the
policy currently thresholds `tau_min`. Add a mode selecting `bu_min`, with
$\tau_y(T)$ from [Din21] evaluated per node rather than a scalar.

**Step 2 — the dimensionless form (2–3 days, the actual contribution).**
Normalise with [Men10]'s closed-form shear-layer thickness

$$\delta=a\Big[\tfrac83\tfrac{\Delta T_0}{g\sigma_s}\tfrac{\tau_R}{AB}\Big(\tfrac{a^2}{k_0}\Big)^{n}\Big(\tfrac{gK_0}{\Delta T_0}\Big)\omega^{\,n+1}\Big]^{1/2}$$

and $Bu$, then follow the [Car09] template: **derive, don't fit**, and invert
to a predicted void *fraction* rather than a yes/no flag. ✅ This is the part no
located work has done.

**Step 3 — sensitivity, reported honestly (½ day).** Sweep $Bu^{crit}$ *and*
$\tau_y$ from the archive — both are free parameters and $Bu$ scales linearly
in $\tau_y$. ⚠️ [Wan21]'s warning is ours: their stick-slip ratio depends on the
choice of $v_c$, and so does any threshold of ours.

**Step 4 — re-run on the divergence-free field** once
`DIVERGENCE_FREE_PLAN.md` step 5 lands. ✅ Only then is a density deficit
attributable to detachment.

**Step 5 — validate on pattern, not number.** $\gamma=1.0$ mm/rev should give
sound welds; a criterion that voids all 16 cases is falsified.

---

## 6. References

- [Buc04] Buck, G. A. & Langerman, M. (2004). Non-dimensional characterization
  of the friction stir/spot welding process using a simple Couette flow model.
  Part I: constant property Bingham plastic solution. *AIP Conference
  Proceedings* **712**, 1283–1288. https://doi.org/10.1063/1.1766706
- [Din21] Ding, S., Shi, Q. & Chen, G. (2021). Flow stress of 6061 aluminum
  alloy at typical temperatures during friction stir welding based on hot
  compression tests. *Metals* **11**(5), 804.
  https://doi.org/10.3390/met11050804
- [Su19] Su, M.-N. & Young, B. (2019). Material properties of normal and high
  strength aluminium alloys at elevated temperatures. *Thin-Walled Structures*
  **137**, 463–471. https://doi.org/10.1016/j.tws.2019.01.012
- [Moh21] Mohammadi, M. & Rezaei Ashtiani, H. (2021). Influence of heat
  treatment on the AA6061 and AA6063 aluminum alloys behavior at elevated
  deformation temperature. *Iranian Journal of Materials Science and
  Engineering* **18**(2). https://doi.org/10.22068/ijmse.18.2.11
- [Rud20] Rudnytskyj, A., Simon, P., Jech, M. & Gachot, C. (2020). Constitutive
  modelling of the 6061 aluminium alloy under hot rolling conditions and large
  strain ranges. *Materials & Design* **190**, 108568.
  https://doi.org/10.1016/j.matdes.2020.108568
- [Men10] Mendez, P. F., Tello, K. E. & Lienert, T. J. (2010). Scaling of
  coupled heat transfer and plastic deformation around the pin in friction stir
  welding. *Acta Materialia* **58**(18), 6012–6026.
  https://doi.org/10.1016/j.actamat.2010.07.019
- [Wan21] Wang, X., Gao, Y., Liu, X., McDonnell, M. & Feng, Z. (2021).
  Tool-workpiece stick-slip conditions and their effects on torque and heat
  generation rate in friction stir welding. *Acta Materialia* **213**, 116969.
  https://doi.org/10.1016/j.actamat.2021.116969
- [Zha22] Zhao, S., Pei, X. & Liu, W. (2022). Computational investigation on
  void defects formation and periodic tool-workpiece sliding-to-sticking
  transition in self-reacting friction stir welding. *The International Journal
  of Advanced Manufacturing Technology* **120**, 7227–7240.
  https://doi.org/10.1007/s00170-022-09259-0
- [Che23] Chen, G., Shi, Q. & Wu, Q. (2023). Numerical study on void defect
  formation in friction stir welding. *Journal of Manufacturing Processes*
  **86**, 85–97. https://doi.org/10.1016/j.jmapro.2022.12.040
- [Dia20] Dialami, N., Cervera, M. & Chiumenti, M. (2020). Defect formation and
  material flow in friction stir welding. *European Journal of Mechanics -
  A/Solids* **80**, 103912. https://doi.org/10.1016/j.euromechsol.2019.103912
- [Car09] Carlson, K. D. & Beckermann, C. (2009). Prediction of shrinkage pore
  volume fraction using a dimensionless Niyama criterion. *Metallurgical and
  Materials Transactions A* **40**, 163–175.
  https://doi.org/10.1007/s11661-008-9715-y
- [Van19] Vanness, K., Johnson, W. P. et al. (2019). A unified force and torque
  balance for colloid transport: predicting attachment and mobilization under
  favorable and unfavorable conditions. *Langmuir* **35**(27), 9061–9070.
  https://doi.org/10.1021/acs.langmuir.9b00911
- [Tor07] Torkzaban, S., Bradford, S. A. & Walker, S. L. (2007). Resolving the
  coupled effects of hydrodynamics and DLVO forces on colloid attachment in
  porous media. *Langmuir* **23**(19), 9652–9660.
  https://doi.org/10.1021/la700995e
- [Bra10] Bradford, S. A., Torkzaban, S. & Wiegmann, A. (2010). Pore-scale
  simulations to determine the applied hydrodynamic torque and colloid
  immobilization. *Vadose Zone Journal* **10**(1), 252–261.
  https://doi.org/10.2136/vzj2010.0064
- [Har65] Harlow, F. H. & Welch, J. E. (1965). Numerical calculation of
  time-dependent viscous incompressible flow of fluid with free surface.
  *Physics of Fluids* **8**, 2182–2189. https://doi.org/10.1063/1.1761178
