# Predicting Voids and Wormholes in FSW from Single-Phase Flow Simulation

**A literature reference and critical evaluation of `Estimate_bubbles_perplexity.md`**

Compiled 2026-09-22 · A. R. Hashemi
Searches run via Undermind (deep search + semantic), Consensus. Scite's relevance ranking failed on these queries (returned unrelated chemistry/physics) and was not used.

---

## 0. Executive summary

**The colleague's core instinct is right; the proposed analogy is wrong.**

- **Right:** predicting voids from a *single-phase* simulation plus a *post-hoc criterion*, without ever meshing a gas phase, is not only possible — it is the **dominant paradigm** in FSW defect modelling, with 15+ years of literature and quantitative, experimentally validated criteria.
- **Wrong:** the route there is **not** turbulence, vortex cavitation, or air entrainment. The perplexity document imports the physics of water/air free-surface flows, which does not transfer. FSW flow is **Stokes creeping flow at Re ≈ 10⁻⁷–10⁻⁴** (§2), has **no free surface** in the bulk, **no vapour phase**, and operates under **50–200 MPa compressive** hydrostatic pressure — conditions that *close* voids rather than open them.
- **The correct physics:** FSW voids are a **material-continuity / volumetric-filling failure**. The cavity that transiently opens behind the advancing pin must be refilled each revolution by material swept around from the retreating side. When the flow delivers less volume than the cavity requires, the deficit is frozen in as a void (§3). The relevant literature is **metal forming void closure and ductile damage**, not cavitation (§5).

Sections 3–6 give the criteria that are actually used, with the quantitative thresholds. Section 7 is the recommended path for us, which connects directly to JAXTrace's particle-tracking capability.

---

## 1. What the perplexity document gets right and wrong

| Claim in the doc | Verdict | Why |
|---|---|---|
| Single-phase CFD + a closure model can predict where bubbles/voids appear | **Correct, and central** | This is exactly what FSW defect modelling does — but with a *filling* closure, not a cavitation closure |
| "Vorticity alone is necessary but not sufficient" | **Correct in spirit** | In FSW, vorticity is essentially *irrelevant*; the tool imposes swirl everywhere, including in perfectly sound welds |
| Cavitation number σ = (p − p_v)/(½ρU²) as a criterion | **Not applicable** | p_v of Al at ~750 K is ~10⁻⁹ Pa. With p ≈ +100 MPa compressive, σ ~ 10¹¹. Cavitation is impossible by ~11 orders of magnitude |
| Froude / Weber / Bond numbers, air entrainment at a free surface | **Not applicable to the bulk** | There is no air–metal interface inside the stir zone. Only marginally relevant at the *top surface* under the shoulder (surface lack-of-fill / flash), a different defect class |
| Re = ρUL/μ as a first check "are vortices possible" | **Correct method, decisive negative answer** | See §2 — Re ≈ 10⁻⁷–10⁻⁴. Laminar creeping flow. Turbulence models are inappropriate |
| Vortex Reynolds/Weber numbers, Fr_Ξ² circulation-flux model (MIT/JFM) | **Not transferable** | Derived for a water–air interface with surface tension and gravity as restoring forces. In FSW, the restoring force is the **deviatoric yield stress of the solid**, a completely different closure |
| "Two-phase VOF/level-set is the gold standard" | **Half right** | VOF *is* used in FSW — but as a **free-surface tracker for the metal/void boundary in a CEL model**, not to resolve gas dynamics. The "second phase" is vacuum/void, not air |

**Bottom line:** the document is a competent summary of *hydrodynamic cavitation and air entrainment*. Essentially none of its specific criteria survive transfer to FSW. Its *structure* — simulate one phase, apply a threshold — is the right structure, and Sections 3–5 supply the correct thresholds.

---

## 2. Why turbulence/cavitation does not transfer: the numbers

Effective viscosity of plasticized aluminium during FSW has been **measured**, not just assumed:

- Franke et al. (2017) — capillary-rheometer method through a drilled backing plate: **μ_eff = 10⁵ – 5×10⁶ Pa·s** for AA6061-T6, in agreement with CFD.
- Reisgen et al. (2020) — inverse method from forge force and spindle torque in spot-FSW, AA5083: temperature- and strain-rate-dependent μ_eff of the same order.

With ρ ≈ 2700 kg/m³, tool radius R = 6–9 mm, ω = 1000–1500 rpm (U = ωR ≈ 0.6–1.4 m/s), L = 2R:

| μ_eff (Pa·s) | Re = ρUL/μ |
|---|---|
| 1×10⁵ | 2×10⁻⁴ – 7×10⁻⁴ |
| 5×10⁶ | 4×10⁻⁶ – 1×10⁻⁵ |
| 1×10⁸ | 2×10⁻⁷ – 7×10⁻⁷ |

**Re ≈ 10⁻⁷ to 10⁻⁴.** Transition to turbulence needs Re ~ 10³–10⁵. We are **7–12 orders of magnitude** away. The flow is Stokes creeping flow: inertia is negligible, the velocity field is quasi-reversible and slaved to the boundary conditions, and there are no inertial vortex cores to drop the pressure.

Kadian & Biswas (2015) compared laminar and turbulent models for FSW material flow directly — the relevant benchmark if anyone insists on a turbulence closure. Seidel & Reynolds (2003), the foundational FSW fluid-mechanics model (186 citations), states the framing explicitly: "**laminar, viscous flow of a non-Newtonian fluid past a rotating circular cylinder**."

**Consequence:** any proposal to run an FSW simulation with a turbulence model and look for vortex cavitation should be declined. It would be physically meaningless and reviewers in this field would reject it immediately.

---

## 3. The physically correct mechanism: volumetric filling failure

The consensus mechanism across the experimental and numerical literature:

1. The rotating pin carries a **shear layer** of plasticized material from the **retreating side (RS)** around the back of the tool to the **advancing side (AS)**.
2. Behind the advancing pin, a **transient cavity** opens each revolution.
3. Sound weld: incoming material fully refills the cavity within the revolution.
4. Defective weld: the cavity is only **partially filled**; the unfilled remainder is left behind, and successive revolutions superimpose it along the weld line into a continuous **tunnel/wormhole**.

Supporting evidence:

- **Ghate et al. (2020)** — explicit per-revolution cavity filling model; "the incoming material fills the cavity per revolution. However, in some cases the cavity is partially filled which leads to the discontinuities." Unfilled area larger on AS; discontinuity size grows with travel speed; tunnel defect predicted by **superposition of per-revolution discontinuities along the feed direction**. Validated against CT.
- **Morisada et al. (2015)** — direct **X-ray real-time imaging** (two synchronised systems) of flow around the tool. Defects correlate with *tilt of horizontal flow* and **stagnation on the AS**; measured flow velocity on the AS **drops** when defects form. This is the cleanest experimental confirmation.
- **Zeng et al. (2018)** — oxide-marker + stop-action welding: "instantaneous void occurrence and **insufficient inflow material**" cause preferential voids in the top SZ on the AS.
- **Zhao et al. (2019)** — CFD (Fluent), 12 mm 7N01: tunnelling at low ω correlates with large variation of plasticized-region size through thickness; at high ω, **imbalance of rotational vs longitudinal flow** around the pin gives cavity-type defects.
- **Zhu et al. (2016), Mater. & Design** — a **low-friction-force region behind the pin**; material arriving from RS decelerates there for lack of driving force and fails to reach the AS. Wormholes predicted graphically by tracer-particle distribution.

**Why AS and not RS:** on the RS, tool surface velocity and traverse velocity oppose; on the AS they add. The AS is where the flow must turn most sharply and where the shear layer is thinnest — hence stagnation and under-filling there. Every source above places voids on the AS.

---

## 4. Quantitative criteria actually used (ranked by usefulness to us)

### 4.1 Contact-pressure differential — an explicit threshold
**Shi et al. (2022)**, *Int. J. Mech. Sci.* — thermal-fluid-structure coupled model with non-uniform tool–workpiece contact pressure. Result:

> The **difference between maximum and minimum tool–workpiece contact pressure** serves as a numerical criterion: sound joint if **Δp < 15 MPa**; void defect if **Δp > 15 MPa**.

This is the most directly usable published number. Caveat: calibrated for one alloy/tool combination; treat the *form* as transferable and the *value* as requiring recalibration.

### 4.2 Mass-balance / flow-partitioning
- **Arbegast (2008)**, *Scripta Mater.* (308 citations) — **the foundational framework.** Partitions flow into distinct deformation zones around pin and under shoulder; defines an **"excess material function"** and a **"forcing function"** that partitions flow to fill the cavity behind the tool; **"inadequacies in this flow are related to specific FSW defect types."** Read this first — it is the conceptual ancestor of everything in §3.
- **Qian et al. (2013)**, *Scripta Mater.* — analytical model "based on the principle of **exactly balancing the material flowing from the region ahead of the pin to the rear** with an optimum temperature." Produces optimum ω–v operating windows consistent with experiment. Cheap and directly testable.
- **Agiwal et al. (2025)**, *JMSE* — empirical: measures stir-zone **area** from macrographs and compares against **tool advance per revolution**. Provides empirical support for the filling theories and is an easy experimental cross-check.

### 4.3 Tracer/particle-tracking (most relevant to JAXTrace)
- **Dialami, Cervera & Chiumenti (2020)**, *Eur. J. Mech. A/Solids* (124 citations) — **the key methodological paper for us.** Two-stage strategy: (i) fast speed-up stage on a fixed mesh to reach steady state; (ii) periodic stage with tool rotation modelled, in which **material tracers are advected using the nodal velocity field**. Predicts **void, wormhole, flash, joint-line remnant and onion rings in a single simulation**. This is exactly a particle-tracking post-process on a precomputed velocity field.
- **Dialami et al. (2019)**, *Int. J. Mech. Sci.* — same tracer machinery applied to oxide-layer transport → joint line remnant; validated against macrographs of anodised workpieces.
- **Chu et al. (2021)**, *IJMTM* — CEL with embedded tracing particles for bobbin-tool FSW; asynchronous horizontal/vertical flow converging on the AS.

### 4.4 Streamline-integrated damage (physically the most principled)
- **He, Dawson & Boyce (2008)**, *J. Eng. Mater. Technol.* — **steady-state Eulerian** FE flow/thermal solution, then a **void-growth model integrated along the streamlines**. Void growth rate as a function of void volume fraction, effective deformation rate, and **ratio of mean stress to material strength**. Spatial location of predicted voids agrees with experiment.

  This is the closest published analogue to "simulate one phase, then post-process with a criterion," and it is *directly* implementable on top of a JAXTrace trajectory. Also He & Dawson (2007) on 3D void growth in FSW of stainless steel.

### 4.5 CEL + Volume-of-Fluid (the de facto industry standard)
Here VOF tracks the **metal/void free boundary**, not a gas phase. Material leaving the domain or failing to fill an Eulerian cell registers as a defect.
- **Chauhan et al. (2018)**, *JMP* (115 citations) — CEL + VOF, validated against **X-ray CT**.
- **Choudhary & Jain (2022)**, *MAMS* — first systematic prediction of tunnel/void/cavity/root defects; process-parameter window; tilt angle dominant (0° → tunnel; 1° → void+cavity; 2° → defect-free; 3° → defective).
- **Zhu et al. (2017)**, *Metals* — CEL; predicts void presence accurately but **overestimates void size** (a known, general limitation).
- **Das, Bag & Pal (2021)**, *STWJ*; **Salloomi (2022)** (dissimilar Al); **Hosseini & Arezoudar (2023)** (underwater stationary shoulder); **Draper et al. (2025)** (refill FSSW, tunnel defects validated by microstructure, "volume ratio is an important parameter").

### 4.6 Machine learning on simulation-derived features
- **Du et al. (2019)**, *npj Comput. Mater.* (94 citations) — **the paradigm case for our purposes.** 108 experimental datasets, three Al alloys. Raw welding parameters → 83.3% accuracy. Features from an **analytical** FSW model (temperature, strain rate, torque, max shear stress on pin) → 90–93.3%. Same features from a **rigorous numerical model** → **96.6%** with both decision tree and neural net. **Temperature and maximum shear stress dominate.**

  The lesson: physics-derived features from a single-phase simulation beat raw parameters by 13 percentage points. This validates the whole "single-phase + derived criterion" strategy quantitatively.
- **Alhourani et al. (2025)**, *Mater. & Design* — CEL + ANN for HDPE FSW; combined framework generates defect/defect-free process maps; wormholes at 1000 rpm without preheat.
- Force-signal ML (complementary, in-process rather than predictive): **Rabe et al. (2021, 2024)** — CNN/LSTM on force feedback, >99% detection; **Guan et al. (2022)**, *Scripta Mater.* — 95.8% defect detection, 98.0% tunnel-vs-porosity classification; key signal is abnormal rise in **F_y,avg** from redundant material transported to the RS.
- **Ansari et al. (2022)**, *JMSE* — correlates process forces with void morphology in the **phase space of in-plane force vs tool rotation angle**; high-fidelity FEA linking force signatures to void structure.

---

## 5. The correct physical analogue: void closure in metal forming

This is the body of literature the perplexity document should have found, and the honest answer to "what is FSW's version of cavitation theory."

Key insight: at **negative stress triaxiality** (compressive hydrostatic stress), voids **close**; at positive triaxiality they grow. FSW under the shoulder is strongly compressive — which is precisely why FSW *consolidates* material and why voids appear only where that compression is lost.

- **Saby, Bouchard & Bernacki (2015)**, *JMP* — **"Void closure criteria for hot metal forming: A review."** The entry point to this literature.
- **Zapara, Tutyshkin & Müller (2013)** — growth and closure at negative triaxialities; mechanism is **void shape change** under compression, distinct from volumetric growth under tension.
- **Chen et al. (2021)**; **Wang et al. (2020)** — void evolution models accounting for **stress triaxiality, Lode parameter and effective strain**.
- **Chbihi et al. (2016/2018)** — influence of **Lode angle** on void closure modelling.
- **Saby et al. (2018)**; **Saby (2014)**; **Bouchard et al. (2016)** — multiscale, 3D real void closure at the mesoscale.
- **Christiansen et al. (2017)** — predicting crack onset in bulk forming via ductile damage criteria (Cockcroft–Latham, Lemaitre, etc.).
- **Nielsen (2009)** — **shear-modified Gurson model** applied to an FSW tensile specimen.
- **Ghate et al. (2020)** — **plastic limit load model** for microvoid nucleation/growth/coalescence within an FSW flow model (see §3).

**Takeaway:** the proper "closure" to bolt onto our single-phase flow field is a **triaxiality/Lode-dependent void evolution law integrated along streamlines** — the metal-forming analogue of a cavitation model. He et al. (2008) already demonstrates exactly this for FSW.

---

## 6. Where the perplexity document *is* partially salvageable

One narrow case survives: the **top surface under the shoulder**, where there genuinely is a metal/air interface. Surface lack-of-fill, flash, and surface-breaking grooves involve a real free surface, and CEL+VOF models do track it (e.g. Das et al. 2021 explicitly predicts "surface irregularities" alongside volumetric defects). But:

- The restoring force is the **yield stress**, not surface tension. Weber/Bond numbers remain inapplicable (the metal's "surface tension" is irrelevant at 10⁶ Pa·s viscosity and ~100 MPa flow stress).
- The relevant defect class (flash, lack-of-fill) is *surface*, not the internal wormholes we care about.

So: worth one sentence of acknowledgement, not a modelling strategy.

---

## 7. Recommended direction for us

Given JAXTrace's existing GPU particle-tracking on time-dependent meshes, the natural and defensible contribution is:

**Stage 1 — Reproduce the established baseline.** Take a single-phase viscoplastic FSW velocity field. Advect dense tracer clouds (this is what JAXTrace already does at scale). Reproduce the Dialami-style void/wormhole prediction from tracer depletion behind the tool. Low risk, establishes credibility.

**Stage 2 — The actual novelty: streamline-integrated damage at scale.** Follow He et al. (2008) but at a scale nobody has attempted: integrate a **triaxiality- and Lode-dependent void evolution law** (from §5) along **millions** of GPU-tracked pathlines, rather than the handful of streamlines in the 2008 paper. Output a **continuous void-susceptibility field**, not a binary defect/no-defect flag. This is where our tracking throughput is a genuine differentiator.

**Stage 3 — Surrogate/ROM.** We already have ROM machinery for (v_adv, ω_pin) → field. A ROM over the void-susceptibility field would give a near-instant process-window map — the natural successor to Du et al. (2019), but spatially resolved rather than a scalar classification.

**Quantitative checks to run first (cheap, one afternoon):**
1. Confirm Re ≪ 1 for our own parameters (§2) — closes the turbulence question definitively and gives us a citable sentence.
2. Compute the Shi et al. Δp contact-pressure differential on an existing simulation; see whether the 15 MPa threshold has any signal in our data.
3. Compute the Qian et al. mass balance for our (ω, v) cases and compare with any voids we have observed.

**What to avoid:** turbulence models, cavitation numbers, Froude/Weber criteria, and any framing of the problem as "air bubbles." Reviewers in the FSW community will read it as a category error. The word to use throughout is **void** or **cavity**, never *bubble*.

---

## 8. Reference list

Grouped as in the text. All DOIs verified through Undermind/Consensus metadata.

### Foundational framework
- Arbegast, W. J. (2008). A flow-partitioned deformation zone model for defect formation during friction stir welding. *Scripta Materialia*, 372–376. https://doi.org/10.1016/J.SCRIPTAMAT.2007.10.031
- Seidel, T. U., & Reynolds, A. P. (2003). Two-dimensional friction stir welding process model based on fluid mechanics. *Science and Technology of Welding and Joining*, 8(3), 175–183. https://doi.org/10.1179/136217103225010952
- Nunes, A. C. (2006). Metal Flow in Friction Stir Welding. https://www.semanticscholar.org/paper/ba832038de0abe9eb7790e1d3203405f51c0cad3

### Flow regime / effective viscosity (the Re argument)
- Franke, D., Morrow, J., Zinn, M., Duffie, N., & Pfefferkorn, F. (2017). Experimental Determination of the Effective Viscosity of Plasticized Aluminum Alloy 6061-T6 during Friction Stir Welding. *Procedia Manufacturing*, 218–231. https://doi.org/10.1016/J.PROMFG.2017.07.050
- Reisgen, U., Schiebahn, A., Sharma, R., Maslennikov, A., Rabe, P., & Erofeev, V. (2020). A method for evaluating dynamic viscosity of alloys during friction stir welding. *Journal of Advanced Joining Processes*. https://doi.org/10.1016/j.jajp.2019.100002
- Kadian, A., & Biswas, P. (2015). A Comparative Study of Material Flow Behavior in Friction Stir Welding Using Laminar and Turbulent Models. *JMEP*, 4119–4127. https://doi.org/10.1007/s11665-015-1520-3

### Mechanism: experimental
- Morisada, Y., Imaizumi, T., & Fujii, H. (2015). Clarification of material flow and defect formation during friction stir welding. *STWJ*, 130–137. https://doi.org/10.1179/1362171814Y.0000000266
- Zeng, X., Xue, P., Wang, D., Ni, D., Xiao, B., Wang, K., & Ma, Z. (2018). Material flow and void defect formation in friction stir welding of aluminium alloys. *STWJ*, 677–686. https://doi.org/10.1080/13621718.2018.1471844
- Agiwal, H., et al. (2025). Empirical analyses of weld zones to understand material flow and defect formation during friction stir welding. *JMSE*.
- Delgado, M., Flores, R., Santana, C., & Reyes-Osorio, L. (2023). Inspection of defects in friction stir welded Al-7075 T6 alloy. *NDT&E*, 276–292. https://doi.org/10.1080/10589759.2023.2193391

### Mechanism: numerical criteria
- Shi, L., et al. (2022). Thermal-fluid-structure coupling analysis of void defect in friction stir welding. *Int. J. Mech. Sci.* — **Δp > 15 MPa criterion**. https://doi.org/10.1016/j.ijmecsci.2022.107969
- Ghate, N., Sood, A., Srivastava, A., & Shrivastava, A. (2020). Ductile fracture based joint formation mechanism during friction stir welding. *Int. J. Mech. Sci.*, 105293. https://doi.org/10.1016/j.ijmecsci.2019.105293
- Zhao, Y., Han, J., Domblesky, J., Yang, Z., Li, Z., & Liu, X. (2019). Investigation of void formation in friction stir welding of 7N01 aluminum alloy. *JMP*. https://doi.org/10.1016/J.JMAPRO.2018.11.019
- Zhu, Y.-C., et al. (2016). Simulation of material plastic flow driven by non-uniform friction force during FSW and related defect prediction. *Materials & Design*. https://doi.org/10.1016/J.MATDES.2016.06.119
- Qian, J., et al. (2013). An analytical model to optimize rotation speed and travel speed of friction stir welding for defect-free joints. *Scripta Materialia*. https://doi.org/10.1016/J.SCRIPTAMAT.2012.10.008
- Chen, J., et al. (2023). Numerical simulation of weld formation in FSW based on non-uniform tool-workpiece interaction: An effect of tool pin size. *JMAPRO*.
- Almoussawi, M., et al. (2017). Modelling of friction stir welding of DH36 steel. *IJAMT*.
- Zhao, C., & Liu, X. (2021). An alternative pressure-dependent velocity boundary condition for modeling self-reacting friction stir welding. *IJAMT*, 1601–1613. https://doi.org/10.1007/s00170-021-07589-z

### Tracer / particle tracking  ← most relevant to JAXTrace
- Dialami, N., Cervera, M., & Chiumenti, M. (2020). Defect formation and material flow in Friction Stir Welding. *Eur. J. Mech. A/Solids*. https://doi.org/10.1016/j.euromechsol.2019.103912
- Dialami, N., Cervera, M., Chiumenti, M., & Segatori, A. (2019). Prediction of joint line remnant defect in friction stir welding. *Int. J. Mech. Sci.* https://doi.org/10.1016/J.IJMECSCI.2018.11.012
- Chu, Q., et al. (2021). In-depth understanding of material flow behavior and refinement mechanism during bobbin tool friction stir welding. *IJMTM*.
- Liu, Q., et al. (2021). Numerical investigation on thermo-mechanical and material flow characteristics in FSW for aluminum profile joint. *IJAMT* — "sparse material area" behind the pin.

### Streamline-integrated damage  ← most principled
- He, Y., Dawson, P., & Boyce, D. (2008). Modeling Damage Evolution in Friction Stir Welding Process. *J. Eng. Mater. Technol.*, 021006. https://doi.org/10.1115/1.2840963
- He, Y., & Dawson, P. (2007). Three-Dimensional Modeling of Void Growth in Friction Stir Welding of Stainless Steel.

### CEL + VOF
- Chauhan, P., Jain, R., Pal, S. K., & Singh, S. (2018). Modeling of defects in friction stir welding using coupled Eulerian and Lagrangian method. *JMP*. https://doi.org/10.1016/J.JMAPRO.2018.05.022
- Choudhary, A., & Jain, R. (2022). Numerical prediction of various defects and their formation mechanism during FSW using CEL technique. *MAMS*, 2371–2384. https://doi.org/10.1080/15376494.2022.2053911
- Zhu, Z., et al. (2017). A Finite Element Model to Simulate Defect Formation during Friction Stir Welding. *Metals*, 256. https://doi.org/10.3390/MET7070256
- Das, D., Bag, S., & Pal, S. (2021). A finite element model for surface and volumetric defects in the FSW process using a CEL approach. *STWJ*, 412–419. https://doi.org/10.1080/13621718.2021.1931760
- Salloomi, K. (2022). Defect monitoring in dissimilar friction stir welding of aluminum alloys using CEL finite element model. *AMPT*, 931–947. https://doi.org/10.1080/2374068X.2022.2106669
- Hosseini, A., & Fallahi Arezoudar, A. (2023). Determining the mechanism of defect formation and material flow characteristics in underwater stationary shoulder FSW using CEL simulation. *IJAMT*, 1755–1778. https://doi.org/10.1007/s00170-023-11513-y
- Draper, J., Fritsche, S., Amancio-Filho, S. T., Galloway, A., & Toumpis, A. (2025). A numerical modelling approach to predict material flow and defect formation in refill friction stir spot welded joints. *IJAMT*, 2907–2921. https://doi.org/10.1007/s00170-025-16092-8
- Choudhary, A., & Jain, R. (2024). Numerical simulation of material flow and defect formation during FSW to predict weld failure location. *Proc. IMechE Part B*.

### Void closure / ductile damage in metal forming  ← the correct analogue
- Saby, M., Bouchard, P.-O., & Bernacki, M. (2015). Void closure criteria for hot metal forming: A review. *JMP*, 239–250. https://doi.org/10.1016/J.JMAPRO.2014.05.006
- Zapara, M., Tutyshkin, N., & Müller, W. (2013). Growth and Closure of Voids in Metals at Negative Stress Triaxialities. *Key Eng. Mater.*, 1125–1132. https://doi.org/10.4028/www.scientific.net/KEM.554-557.1125
- Chen, et al. (2021). Void-closure behavior and a new void-evolution model for various stress states.
- Wang, et al. (2020). A void evolution model accounting for stress triaxiality, Lode parameter and effective strain for hot metal forming.
- Chbihi, A., et al. (2016/2018). Influence of Lode angle on modelling of void closure in hot metal forming processes.
- Saby, M., et al. (2018). Three-dimensional analysis of real void closure at the mesoscale during hot metal forming processes.
- Bouchard, P.-O., et al. (2016). Understanding and modeling of void closure mechanisms in hot metal forming processes: a multiscale approach.
- Christiansen, P., et al. (2017). Predicting the onset of cracks in bulk metal forming by ductile damage criteria.
- Nielsen, K. L. (2009). Effect of a shear modified Gurson model on damage development in a FSW tensile specimen.
- Parvizian, et al. (2018). Numerical analysis of void closure in metal forming.

### Machine learning
- Du, Y., et al. (2019). Conditions for void formation in friction stir welding from machine learning. *npj Computational Materials*. https://doi.org/10.1038/s41524-019-0207-y
- Alhourani, A., et al. (2025). Computational and machine learning modelling approaches for weld quality predictions in FSW of high-density polyethylene. *Materials & Design*.
- Guan, W., et al. (2022). Force data-driven machine learning for defects in friction stir welding. *Scripta Materialia*.
- Rabe, P., et al. (2021). Deep Learning approaches for force feedback based void defect detection in Friction Stir welding. *J. Adv. Joining Processes*.
- Rabe, P., et al. (2024). Development and validation of a generalized, AI-based inline void defect detection solution for FSW based on force feedback. *Welding in the World*.
- Ansari, M., Agiwal, H., Zinn, M., Pfefferkorn, F., & Rudraraju, S. (2022). Novel correlations between process forces and void morphology for effective detection and minimization of voids during friction stir welding. *JMSE*. https://doi.org/10.1115/1.4054338
- Hunt, J. B., et al. (2022). A Generalized Method for In-Process Defect Detection in Friction Stir Welding. *JMMP*.

### Reviews
- Das, D., et al. (2024). A review on phenomenological model subtleties for defect assessment in friction stir welding. *JMAPRO*. https://doi.org/10.1016/j.jmapro.2024.04.063
- Fathi, M. (2025). Review of Friction Stir Welding Defects: Causative Parameters Detection and Repairing. *Trans. Indian Inst. Metals*.
- Kulkarni, B., et al. (2025). Machine learning techniques in monitoring and controlling friction stir welding process: a critical review. *Discover Applied Sciences*.
- Wahab, M. A., et al. (2019). Challenges in the detection of weld-defects in friction-stir-welding (FSW). *AMPT*.

---

## 9. Search provenance

- **Undermind deep search** (completed, ~2 min): "Prediction of void, wormhole, and tunnel defect formation in FSW using CFD material flow simulation…" — [workspace link](https://app.undermind.ai/projects/ad91d850-93a8-48f2-a729-f94d9279ccdf)
- **Undermind semantic searches**: (a) void closure criteria / stress triaxiality / ductile damage in forming and solid-state joining; (b) FSW effective viscosity and laminar vs turbulent flow regime.
- **Consensus**: four queries — single-phase CFD void prediction; pressure/stagnation criteria; ML defect prediction; process-window & swept-volume criteria.
- **Scite**: attempted; relevance ranking returned unrelated chemistry/physics/linguistics papers for FSW queries. Not used. Worth retrying later for citation-context analysis on specific DOIs (e.g. how Arbegast 2008 is cited).

**Not yet done / open leads:**
- Full text of Shi et al. (2022) to check how the 15 MPa Δp threshold was calibrated and how transferable it is.
- Full text of He et al. (2008) for the exact void-growth ODE integrated along streamlines — this is the equation we would implement in Stage 2.
- Arbegast (2008) full text — PDF is available in the Undermind workspace.
