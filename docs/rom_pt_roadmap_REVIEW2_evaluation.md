# Evaluation of `rom_pt_roadmap_REVIEW2.md` against the ROM/Lagrangian literature

**Date:** 2026-07-31
**Sources:** literature search via Consensus (`mcp__claude_ai_Consensus__search`),
covering Xie/Iliescu Lagrangian POD, discretely-divergence-free
POD-Galerkin methods, weighted / non-Euclidean POD for low-energy
regions, and FSW-specific POD work.

## TL;DR

Review 2 identifies three literature-grounded fixes that our next-steps
plan doesn't yet consider:

1. **Lagrangian-inner-product POD** (Xie et al.) — targets the
   Eulerian/Lagrangian amplification we've measured. Directly relevant.
2. **Discretely divergence-free POD basis** — targets the div-free
   violation we've measured. Standard remedy exists; low-effort path.
3. **Mode-count / weighted-inner-product POD** — targets the workpiece
   mean-velocity bias we've measured. This is a documented,
   citable failure mode of energy-weighted POD.

All three point at **basis construction**, not grid representation, as
the highest-leverage lever now that grid-topology optimisation is
saturated (Part 5 of the grid report). The literature check confirms
the review's diagnosis: all three fixes have been demonstrated on
related problems, all three are implementable with our current
infrastructure, and the mode-count observation in particular is a
structural property of L² POD that our current 3-mode basis is
sitting right on top of.

## Claim-by-claim evaluation

### Claim 1 · Lagrangian-inner-product POD (Xie et al.) is a direct fix for the amplification we measured

**Literature check**: **confirmed with strong evidence.**

The review cites [Xie et al. 2018/2020, "Lagrangian Reduced Order
Modeling of Finite Time Lyapunov Exponents"](https://consensus.app/papers/details/f52431fea1f35c768a6adac63b482fb1/?utm_source=claude_desktop)
[1, 2] as proposing a POD variant that builds the basis to maximise
Lagrangian-quantity accuracy (they specifically target FTLE) using
Lagrangian inner products, rather than energy-optimal Eulerian L²
inner products.

The abstract's headline claim is faithfully summarised in Review 2:

> the new Lagrangian ROMs are orders of magnitude more accurate than
> the standard Eulerian ROMs \[...] in approximating not only
> Lagrangian fields (e.g., the finite time Lyapunov exponent (FTLE)),
> but also Eulerian fields (e.g., the streamfunction) \[...] the new
> Lagrangian ROMs do not employ any closure modeling \[...] the
> dramatic increase in the new Lagrangian ROMs' accuracy is entirely
> due to the novel Lagrangian inner products used to build the
> Lagrangian ROM basis.

**Applicability to FSW**: their demonstration is on the quasi-geostrophic
equations, a different physics regime (large-scale, 2D). The mechanism
however is generic — the basis is constructed to preserve trajectories,
not variance — and directly addresses the case-003-style pattern we
have measured (2.62 % Eulerian error → 17.7 % Lagrangian residual,
6.7× amplification). Whether the "orders of magnitude" gain translates
to FSW-scale amplification (2 – 7×) depends on how strongly the
FSW velocity field's dynamically-important regions overlap with its
low-energy regions — the workpiece mean-velocity bias suggests they
do overlap in exactly the way that would make Xie's approach effective.

**Cost to implement**: modest. Our snapshot collection stays the same;
the POD step changes from a standard SVD on `(snapshots × nodes)` to
an SVD in a Lagrangian-weighted inner product. Xie et al. give the
exact matrix formulation. Compatible with our existing `jaxtrace.rom`
pipeline as an alternative POD path, not a replacement.

**Related work worth citing alongside Xie**: [Parish et al. 2022, "On
the impact of dimensionally-consistent and physics-based inner
products for POD-Galerkin"](https://consensus.app/papers/details/9fdc346d0b4d500f8636b37839cf2b58/?utm_source=claude_desktop)
[3] shows that inner-product choice systematically affects ROM
accuracy and robustness. It's compressible Euler, not incompressible
FSW, but the underlying "the inner product matters" argument is the
same one Xie makes.

### Claim 2 · Discretely divergence-free POD is a standard remedy for the div-free violation we measured

**Literature check**: **confirmed with multiple established references.**

Review 2 cites the general observation that building the POD basis on
a discretely divergence-free velocity space (via inf-sup-stable
mixed FEM or explicit projection onto div-free) removes the divergence
inheritance that our smoothness report §4 documented (ROM's div
pattern is 99 % correlated with FOM's).

The literature I checked confirms:

- [Akhtar et al. 2009, "On the stability and extension of reduced-order Galerkin models in incompressible flows"](https://consensus.app/papers/details/933ae415d39650048d32ea52c27aea12/?utm_source=claude_desktop)
  [4] uses "divergence-free velocity and pressure modes" for the POD of
  cylinder wake flow — the standard practice.
- [Stabile et al. 2017, "Finite volume POD-Galerkin stabilised reduced order methods for the parametrised incompressible Navier–Stokes equations"](https://consensus.app/papers/details/ead7a2245d1a501795770d835ad3ed3a/?utm_source=claude_desktop)
  [5] discusses two pressure-stabilisation strategies: supremizer
  enrichment vs. pressure-Poisson, both compatible with div-free
  bases.
- [Novo & Rubino 2020, "Error analysis of POD stabilized methods for incompressible flows"](https://consensus.app/papers/details/def9702c3df45a6c9277be5bb5334e1e/?utm_source=claude_desktop)
  [6] proves error bounds for POD methods when the snapshots are
  discretely divergence-free from an inf-sup stable FEM, and explicitly
  notes: *"since the snapshots are discretely divergence-free, the
  pressure can be removed from the formulation of the POD approximation
  to the velocity"*. Direct, low-effort path.
- [Gräßle et al. 2019](https://consensus.app/papers/details/06db025349d25ee89613a2a744f9f09d/?utm_source=claude_desktop)
  [7] use either projection onto a common div-free space or
  supremizer enrichment — same recipe.
- [Star et al. 2020, "Reduced order models for incompressible Navier-Stokes on collocated grids using a discretize-then-project approach"](https://consensus.app/papers/details/2afcea692ec7596ebba5f821ef31604d/?utm_source=claude_desktop)
  [8] demonstrates that the "consistent flux method" (produces
  div-free velocity at both FOM and ROM level) is *"slightly more
  accurate"* than the inconsistent method that doesn't preserve
  div-freeness. Small effect on rms, but real.

**Applicability to FSW**: The review's specific suggestion is
compelling — *if a companion project already uses a mixed u/p/e
formulation, build the POD on that formulation's velocity output.*
Worth verifying whether such an FSW-mixed-formulation output exists
and can be surfaced.

**Cost to implement**: low if the mixed-formulation output already
exists; medium if we have to construct the projection ourselves. The
projection is a standard Helmholtz-Hodge operation.

**Caveat from the literature**: [Lee et al. 2020](https://consensus.app/papers/details/6ae4c72d22f251cca054e5e19ea5ae11/?utm_source=claude_desktop)
[9] flags that "several sources of error [...] call into question if
a POD basis for an incompressible flow is divergence-free" — so even
with a div-free FOM input, the POD basis may not preserve it exactly.
Worth measuring how much of the div-free inheritance we can actually
eliminate.

### Claim 3 · The 3-mode ceiling is likely the dominant error source, not mesh-vs-grid basis choice

**Literature check**: **confirmed as a structural property, not
domain-specific.**

Review 2 argues that energy-weighted POD systematically under-resolves
low-energy regions, which in FSW is the workpiece bulk (the pin
rotation carries most of the kinetic energy). Multiple sources
support this directly:

- [Christensen et al. 1999, "Evaluation of POD-Based Decomposition Techniques"](https://consensus.app/papers/details/113e721cea5d5808a3f03c6228e403f8/?utm_source=claude_desktop)
  [10] proposes "weighted POD (w-POD) as an alternative to give higher
  priority to low-energetic or important modes by simply weighting"
  and "predefined POD (p-POD)" for a priori-known important modes.
  Exactly the failure mode the review names.
- [Dellacasagrande et al. 2021, "Identification of coexisting dynamics in boundary layer flows through POD with weighting matrices"](https://consensus.app/papers/details/d91633c6ea3c5c19947e9e8f45fe24cb/?utm_source=claude_desktop)
  [11] proposes a non-Euclidean POD (NE-POD) with the explicit goal of
  *"emphasize fluctuation events localized in spatio-temporal regions
  with low kinetic energy magnitude, which are not highlighted by the
  classic POD"*. Also confirms our own diagnosis.
- [Olesen et al. 2022, "Dissipation-optimized POD"](https://consensus.app/papers/details/04148b54bb475ca4a76689a5135d0bc1/?utm_source=claude_desktop)
  [12] shows that swapping the L² inner product for a dissipation-
  weighted one shifts POD modes toward different physical regions:
  *"TKE and dissipation are reconstructed more efficiently in the
  dissipation-rich near-wall region using d-POD modes, and in the
  TKE-rich bulk using e-POD modes."* Same lesson, different weighting.

The pattern across all three references is consistent: **the inner
product acts as a per-region importance weight, and the standard L²
choice gives all weight to high-KE regions**. This is exactly the
mechanism that biases our 3-mode basis toward the pin at the expense
of the workpiece mean.

**Implication for our Step 5 plan**: The plan's phase A is a
comparison of singular-value decay curves for mesh-POD vs grid-POD.
The literature above predicts that **both decays will be dominated by
the same high-KE pin structure**, so the two SV curves will look
essentially identical — resampling the domain doesn't change the
energy distribution across the flow. If phase A confirms this
prediction, the natural next steps are:

- **(a)** run POD with more modes (5, 8, 12) and measure whether the
  workpiece bias reduces — this is the cheapest, most direct test.
- **(b)** replace the L² inner product with either a Lagrangian
  inner product (Xie) or a spatially-weighted inner product that
  up-weights the workpiece region (Christensen / Dellacasagrande).
- Domain-of-POD (mesh vs grid) is a distant third and should probably
  be deprioritised.

### Claim 4 · Grid experiment "current best practice" is provisional

**Literature check**: not really a literature claim, but worth
reinforcing.

The review points out that we had one topology-ranking flip already
(Part 1 raw-P1 → Part 2 HCT, MALMO's ordering reversed) and that a
second reshuffle at 20 cases was plausible. The 20-case rollout
(Parts 4 and 5, completed after Review 2 was written) confirmed the
provisional judgment held up: `4lvl_hct` remained the winner, and 3
further hypothesis-driven variants (`4lvl_r22_hct`, `5lvl_hct`,
`malmo6_hct`) all failed to beat it by statistically-significant but
small margins.

So the review's caution was warranted at the time it was written, and
the 20-case data has now largely resolved the uncertainty in the
FOM-side. The provisional flag now moves downstream — to the
ROM-side rerun of the same experiment.

### Claim 5 · FSW-specific POD work already exists but doesn't address our question

The Consensus search surfaced [Cao et al. 2021, "Machine learning and reduced order computation of a friction stir welding model"](https://consensus.app/papers/details/23e2a9dc5e165c12b1dbeb4c0f7d3fb0/?utm_source=claude_desktop)
[13] which does POD-ROM for the FSW heat + Navier-Stokes system with
shear-dependent viscosity. That work targets **parametric FOM
speedup** (POD of the coupled thermo-mechanical field), not the
downstream particle-tracking / mixing question we're solving. It is
worth citing as prior art for POD-in-FSW but does not overlap our
Lagrangian question.

## What Review 2 did *not* pick up (that the 20-case data now shows)

Two things worth adding to the writeup that Review 2 could not have
known:

1. **The grid-topology saturation is now proven, not conjectured.**
   Part 5 tested 3 more grid variants including MALMO at cells-per-edge
   = 6 (17M cells); none beat `4lvl_hct` (64 k cells), all
   statistically worse at p ≤ 0.03. The review anticipated this
   ("the SV decay pre-check in Step 5 phase A is likely to show very
   similar decay curves"); we now have the empirical confirmation
   from the *tracker-side* that grid representation is not the lever.
   This strengthens the case for moving effort to basis construction.
2. **The universal 3× shear-ring amplification** at r ≈ 4 – 7 mm,
   visible on every one of the 20 cases in both `4lvl_hct` and
   `malmo6_hct`, is a **flow-property signature**, not a tracker
   property. This is the spatial fingerprint of the Xiong-et-al
   amplification the review discussed. The Xie Lagrangian-basis
   approach targets exactly this pattern by weighting the basis
   toward trajectory preservation in the shear zone.

## Revised priority ordering for Step 5

The plan doc's Step 5 currently frames the phase-A hypothesis test as
"is grid-space POD better than mesh-space POD?" and treats mode-count
increases as a fallback. Based on Review 2 + the literature above +
the Part 5 saturation finding, I'd revise the priority order:

### Tier 1 · high leverage, low-medium effort

1. **Increase POD mode count** (5 → 8 → 12 modes) and re-run FOM-vs-ROM
   PT on the 4-case cohort. Cheapest possible test of whether the
   basis-construction argument holds. If more modes reduce the
   workpiece bias visibly, that's the winning fix and we don't need
   any of the more sophisticated approaches.
2. **Lagrangian-inner-product POD** (Xie et al. approach). If the
   mode-count test doesn't close the gap, the next best step is to
   construct the POD basis under a Lagrangian inner product. Cost:
   moderate; a new `jaxtrace.rom` code path building `Ψ` via
   Xie's weighted formulation. Testable on the same 4-case cohort;
   compare Eulerian rel_rms and Lagrangian PT residual against the
   current baseline.

### Tier 2 · high leverage, medium-high effort

3. **Discretely divergence-free POD basis.** Contingent on whether a
   mixed u/p/e formulation FSW output is available. If not, we'd need
   to construct the projection ourselves — a Helmholtz-Hodge
   decomposition on the FOM snapshots. Attack the div-free inheritance
   documented in the smoothness report §4.
4. **Spatially-weighted POD** (Christensen / Dellacasagrande) with an
   explicit workpiece-region up-weight in the inner product. A
   simpler alternative to full Lagrangian inner product; targets the
   same low-energy-region under-representation but with a hand-tuned
   weight rather than a physics-derived one. Interesting comparison
   point rather than a shipping candidate.

### Tier 3 · lower priority

5. **Grid-space POD** (current Step 5 phase A / phase B). Deprioritised
   because both the review and the literature predict it will show
   little vs mesh-space POD, and Part 5 of the grid report has already
   demonstrated grid-topology saturation on the tracker side. Still
   worth doing as a control experiment (phase A SV decay is cheap),
   but not the primary hypothesis.
6. **ROM-on-winner-grid experiment** (plan doc step 4). Still worth
   running because it closes the roadmap §5 acceptance question for
   the *ROM* path, but it does not address the underlying basis-quality
   issue. Run in parallel with Tier 1.

## Concrete plan-doc updates

The `docs/rom_pt_next_steps_plan.md` file should be revised with:

- A new **Step 5b** (Lagrangian-inner-product POD) inserted between
  the current Step 4 (ROM-on-winner-grid) and Step 5 (grid-space
  POD rebuild).
- Step 5 phase A retained but demoted from "primary hypothesis test"
  to "control experiment for basis-domain sensitivity".
- Step 5 phase B (full grid-space rebuild) conditional on both phase A
  and Step 5b showing a promising signal.
- A new **Step 5c** (discretely divergence-free POD basis) added,
  conditional on the mixed-formulation output being available or
  cheap to construct.
- A new **Step 6** (higher-mode-count POD) added *before* Step 5b,
  because it's the cheapest test of the basis-quality hypothesis.

## References

The Consensus search covered peer-reviewed papers, preprints, and
technical reports. Numbered references cited above:

[1] [Xie X, Nolan PJ, Ross SD, Mou C, Iliescu T (2020), "Lagrangian Reduced Order Modeling of Finite Time Lyapunov Exponents"](https://consensus.app/papers/details/f52431fea1f35c768a6adac63b482fb1/?utm_source=claude_desktop)
    (4 citations · journal version)

[2] [Xie X, Nolan PJ, Ross SD, Mou C, Iliescu T (2018), "Lagrangian Data-Driven Reduced Order Modeling of Finite Time Lyapunov Exponents"](https://consensus.app/papers/details/7a2aeb5a0cfa5cc494df041cfd3a8636/?utm_source=claude_desktop)
    (10 citations · arXiv preprint of [1])

[3] [Parish EJ, Rizzi F, Blonigan PJ (2022), "On the impact of dimensionally-consistent and physics-based inner products for POD-Galerkin and least-squares model reduction of compressible flows"](https://consensus.app/papers/details/9fdc346d0b4d500f8636b37839cf2b58/?utm_source=claude_desktop)
    (25 citations · ArXiv)

[4] [Akhtar I, Nayfeh AH, Ribbens CJ (2009), "On the stability and extension of reduced-order Galerkin models in incompressible flows"](https://consensus.app/papers/details/933ae415d39650048d32ea52c27aea12/?utm_source=claude_desktop)
    (224 citations · Theoretical and Computational Fluid Dynamics)

[5] [Stabile G, Rozza G (2017), "Finite volume POD-Galerkin stabilised reduced order methods for the parametrised incompressible Navier–Stokes equations"](https://consensus.app/papers/details/ead7a2245d1a501795770d835ad3ed3a/?utm_source=claude_desktop)
    (219 citations · Computers & Fluids)

[6] [Novo J, Rubino S (2020), "Error analysis of POD stabilized methods for incompressible flows"](https://consensus.app/papers/details/def9702c3df45a6c9277be5bb5334e1e/?utm_source=claude_desktop)
    (36 citations · ArXiv)

[7] [Gräßle C, Hinze M, Ulbrich S (2019), "Model order reduction for space-adaptive simulations of Navier-Stokes"](https://consensus.app/papers/details/06db025349d25ee89613a2a744f9f09d/?utm_source=claude_desktop)
    (PAMM)

[8] [Star S, Sanderse B, Stabile G, Rozza G, Degroote J (2020), "Reduced order models for the incompressible Navier-Stokes equations on collocated grids using a 'discretize-then-project' approach"](https://consensus.app/papers/details/2afcea692ec7596ebba5f821ef31604d/?utm_source=claude_desktop)
    (9 citations · Int. J. Numer. Methods Fluids)

[9] [Lee MW et al. (2020), "On the Importance of Numerical Error in Constructing POD-based Reduced-Order Models of Nonlinear Fluid Flows"](https://consensus.app/papers/details/6ae4c72d22f251cca054e5e19ea5ae11/?utm_source=claude_desktop)

[10] [Christensen EA, Brøns M, Sørensen JN (1999), "Evaluation of Proper Orthogonal Decomposition-Based Decomposition Techniques Applied to Parameter-Dependent Nonturbulent Flows"](https://consensus.app/papers/details/113e721cea5d5808a3f03c6228e403f8/?utm_source=claude_desktop)
    (121 citations · SIAM J. Sci. Comput.)

[11] [Dellacasagrande M, Guardone A, Simoni D (2021), "Identification of coexisting dynamics in boundary layer flows through POD with weighting matrices"](https://consensus.app/papers/details/d91633c6ea3c5c19947e9e8f45fe24cb/?utm_source=claude_desktop)
    (Meccanica)

[12] [Olesen PJ, Hodžić A, Andersen SJ, Sørensen NN, Velte CM (2022), "Dissipation-optimized Proper Orthogonal Decomposition"](https://consensus.app/papers/details/04148b54bb475ca4a76689a5135d0bc1/?utm_source=claude_desktop)
    (12 citations · Physics of Fluids)

[13] [Cao X et al. (2021), "Machine learning and reduced order computation of a friction stir welding model"](https://consensus.app/papers/details/23e2a9dc5e165c12b1dbeb4c0f7d3fb0/?utm_source=claude_desktop)
    (4 citations · J. Comput. Phys.)

*The Consensus search tool prompts an upgrade to Consensus Pro for
20 results/search + study-design metadata. This evaluation used the
free 10-results-per-search tier; the citations above are what
surfaced in the top-10 for each query.*

---

## Extended literature scan · additional findings from 4 further Consensus searches

Review 2 addressed basis-construction issues. Four more targeted
literature searches surfaced findings across four axes that Review 2
did not cover but that directly bear on our roadmap:

### A. Temporal-derivative snapshots and grad-div stabilisation

[García-Archilla, Novo, Rubino 2022](https://consensus.app/papers/details/c9d5fcdb253d5e449950094e5defb2bb/?utm_source=claude_desktop)
[14] shows that **including snapshots that approach the velocity time
derivative** (in addition to the velocity snapshots themselves)
improves POD-ROM accuracy for incompressible Navier-Stokes and gives
error bounds *independent of inverse powers of the viscosity*. The
same authors extended this to pressure error bounds in [2023](https://consensus.app/papers/details/6c4e950b974853f9b6f84210849f5aa9/?utm_source=claude_desktop)
[15] and to continuous-in-time analysis in [2024](https://consensus.app/papers/details/9ef2af95b09b5a95bc263fa192e10dfa/?utm_source=claude_desktop)
[16], showing that **a small number of well-chosen snapshots plus
temporal derivatives can accurately approximate the full time
interval**.

Both these works rely on **grad-div stabilisation** in the FOM and the
ROM to get viscosity-independent bounds. That is a standard remedy
in the modern incompressible-ROM literature (also in [4], [5], [6]).

**Applicability to our roadmap**: currently our ROM uses one snapshot
per case at ts = 119 (static-velocity assumption per plan doc §8).
That's not a time-derivative POD — but the *principle* transfers: if
the ROM will ever be extended to time-dependent (which the roadmap
§8 acknowledges as future work), the García-Archilla path is the
established one for velocity error bounds that don't blow up as
viscosity → 0. Even in the current static case, **including a "temporal
tendency" snapshot for each case (e.g. `v(ts+1) − v(ts-1)`) may
improve the basis without changing the ROM structure**. Worth
prototyping alongside Step 6.

### B. Lagrangian coherent structures for mixing quantification (FSW-analogue evidence)

The closest FSW-analogue in the literature is the mechanically
agitated / stirred-tank community, which has both the pin-driven
rotational geometry and the mixing-as-scientific-target framing.
Multiple hits directly parallel our questions:

- [Kun Li et al. 2022](https://consensus.app/papers/details/29f22ac5f77855d986d8ece317dd8efd/?utm_source=claude_desktop)
  [17] uses PEPT-derived Lagrangian trajectories from a mechanically
  agitated vessel to compute forward + backward FTLE and extract
  attracting / repelling LCS ridges. They show that **hidden LCSs
  organise the chaotic behaviour of fluid particle paths that
  underpin mixing through the exchange of fluid between zones of
  different kinematics**. This is exactly the story we want for FSW —
  and the roadmap §7 already lists FTLE-style Lagrangian diagnostics
  (residence time + pairwise separation) as required for
  §4 / §5 / §7 validation.
- [Bashiri et al. 2016](https://consensus.app/papers/details/f6d548bfa3ea579cad6a25120b72eaa6/?utm_source=claude_desktop)
  [18] uses Radioactive Particle Tracking (RPT) in a Rushton-turbine
  stirred tank to compute Lagrangian trajectories, Poincaré maps,
  and mixing indices based on stochastic independence + memory loss.
  Our FSW residence-time / pairwise-separation diagnostics are the
  same class of Lagrangian mixing metric.
- [Shadden et al. 2005](https://consensus.app/papers/details/f3e7724c147b537ab134203df30ecf9e/?utm_source=claude_desktop)
  [19] (1439 citations) is the canonical LCS reference — should
  cite in any FSW-mixing writeup that uses FTLE.

**Applicability**: our shear-band spatial pattern (Part 5 §4 spatial
finding) is a textbook LCS ridge. FSW's shear band **is** the
mixing zone, and our per-particle displacement error concentrates
there because that's where the FTLE ridge is. Framing our
"where does the error come from" story in the LCS/FTLE vocabulary
would strengthen the writeup considerably. Not a code change, but a
framing / writeup upgrade for the paper draft.

### C. Integrator + interpolation order pairing

Highly relevant to our tricubic-vs-trilinear finding (Part 5 §1: the
tricubic Stage-2 Catmull-Rom compounds worse than trilinear over
2000 steps):

- [Pokrajac et al. 2002](https://consensus.app/papers/details/a5ef30b6fa025145a95a68d5b8de485c/?utm_source=claude_desktop)
  [20] tests all Runge–Kutta orders 1 – 6(4) combined with 1st – 5th
  order exit polynomials for particle tracking in FE meshes. **Two
  headline results:**
  1. **Quadratic velocity interpolation on a coarse mesh often gives
     more accurate paths than linear interpolation on a fine mesh
     while being cheaper.** Direct parallel to our HCT
     Stage-1 finding: better interpolation > more cells.
  2. **The accuracy of the interpolation must match the ODE order** —
     RK5(4) or RK6(4) paired with 5th-order exit polynomial gives
     *several orders of magnitude* accuracy improvement over the
     Pollock method. **RK4 + trilinear is nearly optimal for our
     grid PT**; RK4 + tricubic is a mismatched pairing that may
     be the reason tricubic Stage-2 hurts at step 2000.
- [Beznosov et al. 2025](https://consensus.app/papers/details/770bc529ee665f828855fad69f0c61a3/?utm_source=claude_desktop)
  [21] shows that **insufficient derivative continuity of the
  interpolant degrades RK-scheme accuracy at low error tolerances,
  introducing discontinuity-induced truncation errors**. Our raw-P1
  Stage-1 field is C⁰ (jump at tet faces); HCT-projected is C¹.
  The paper proposes vector-potential Hermite interpolation to
  guarantee C^m continuity for high-order adaptive RK. Directly
  supports the "smoother source field is the lever" theme of our
  Part 5 conclusion.
- [Rössler et al. 2018](https://consensus.app/papers/details/9da7d93f3adc53009325a5c891c76f6d/?utm_source=claude_desktop)
  [22] benchmarks 6 explicit RK schemes on 5000+ ECMWF wind-field
  trajectories: **3rd-order RK with 170 s dt matches 4th-order RK
  in efficiency for tropospheric transport**, and truncation errors
  fall into three groups by RK order. **DT choice matters as much as
  order**. Our fixed DT = 3.75 ms is likely conservative; an
  adaptive-DT test (or DT sweep at fixed order) is a cheap follow-up.
- [Coppola et al. 2001](https://consensus.app/papers/details/e4c41a132aaa5219997d90d0f9ae35b4/?utm_source=claude_desktop)
  [23] discusses **high-order polynomial velocity approximation on
  unstructured meshes** — the mesh-side counterpart to our HCT-3D
  approach. Should be cited in any Stage-1 discussion of the
  mesh path.
- [Yeung & Pope 1988](https://consensus.app/papers/details/7692f4fdedfe5db1b0442df2be5584ce/?utm_source=claude_desktop)
  [24] (353 citations) is the classic reference for **cubic-spline vs
  Taylor-series interpolation of DNS Eulerian fields for Lagrangian
  particle tracking**. Cubic splines give higher interpolation
  accuracy, Taylor series is easier to implement — the same
  trade-off our HCT-3D-vs-P1 Stage-1 comparison shows.

**Applicability to roadmap**: opens a new Tier-2 experiment we hadn't
scoped —

> **Step 9 (proposed): pair-appropriate integrator + interpolation
> order test.** RK4 + trilinear on 4lvl_hct (current baseline) vs
> RK6(4) + tricubic on 4lvl_hct (matched order). Pokrajac's finding
> predicts a several-orders-of-magnitude improvement in
> single-particle accuracy if the pairing is done right — testable
> without any change to the ROM basis.

### D. Related-domain ROM-PT prior art

The stirred-tank + Rushton-turbine POD-ROM literature is the closest
FSW analogue we have. Key hits:

- [Mikhaylov et al. 2023](https://consensus.app/papers/details/ef509961cc0a5608999805d1a476f09f/?utm_source=claude_desktop)
  [25] applies POD to 3-D Rushton-turbine LES data (Re = 30000) and
  finds **the four leading POD modes come in pairs, corresponding to
  precessing Macro-instabilities that rotate opposite to the
  impeller**. Their reduced-order model of mean + first-two-modes
  captures the largest flow structures but *"does not reproduce the
  finer features"* — verbatim the pattern we see in our 3-mode ROM
  missing the workpiece bulk.
- [Mikhaylov et al. 2021](https://consensus.app/papers/details/c7337d91c01b56509c904cacb993f229/?utm_source=claude_desktop)
  [26] combines POD + N4SID system identification for real-time
  reconstruction of blade-wake vortical structures from sparse
  sensors. First pair of POD modes reconstructable from a single
  sensor. Suggests our ROM basis is under-specified per case —
  same "3-mode ceiling" story as Review 2.
- [Arosemena et al. 2023](https://consensus.app/papers/details/560762d290bd53899bd0352200d8f89d/?utm_source=claude_desktop)
  [27] does POD on LES of a baffled stirred tank with Newtonian +
  shear-thinning rheology. Decomposes into mean + most energetic
  periodic + fluctuating. Framework transferable to FSW where the
  rotational impeller is the tool pin.
- [Jiang et al. 2023](https://consensus.app/papers/details/f58e418b94c05c99a99f91daa61bcf4f/?utm_source=claude_desktop)
  [28] builds a SVD-based ROM for solid-liquid mixing in a stirred
  tank, showing **3 orders of magnitude compute reduction** — same
  scale as the speedup we're chasing on the FSW ROM PT path.
- [Peace et al. 2025](https://consensus.app/papers/details/f3539021842b5618b761b89b88cc464c/?utm_source=claude_desktop)
  [29] applies a **Lagrangian-based method for the first time to
  assess mixing performance of aerated stirred tanks** using PEPT.
  This is a very recent (2025) methodological analogue for our
  roadmap §7 Lagrangian diagnostics.

**Applicability to our writeup**: the stirred-tank community has
essentially the same PT-driven mixing question we do, and their
methodological toolkit (POD basis analysis, PEPT / RPT validation,
Poincaré maps, LCS/FTLE) provides ready templates. Our FSW paper
should cite this literature as the near-neighbour rather than only
POD-ROM CFD generic references. The parallel is close enough that
**the observed 3-mode ceiling issue we have is a documented pattern
in that literature too** ([25], [26]), which raises the confidence
that the mode-count fix will help.

### E. FTLE / LCS computation methodology (useful if we implement mixing diagnostics)

- [Qian et al. 2023](https://consensus.app/papers/details/29f44035f49f5b15a1bdca79a5157e25/?utm_source=claude_desktop)
  [30] presents the Lagrangian-Eulerian Stabilized Collocation
  Method (LESCM) for **accurate FTLE computation in viscous
  incompressible flows**, handling 16M particles on a single
  workstation. Directly usable if we implement FTLE on our
  360k-particle FSW runs — well below their scale limit.
- [Lagares et al. 2023](https://consensus.app/papers/details/e7dcf10b8c3c505b87d599d3a28c282f/?utm_source=claude_desktop)
  [31] presents a **GPU-accelerated particle advection method for 3D
  LCS in DNS datasets** with strong scaling to 62 V100 GPUs. Their
  owning-cell locator would map naturally onto our
  mesh-aligned-octree work; the methodology is what we would want
  when scaling from 360k particles to 20M+ for statistically
  converged FTLE.
- [Raben et al. 2013](https://consensus.app/papers/details/3eeb3a7021d15e29a6b6d7ecccff317a/?utm_source=claude_desktop)
  [32] compares three FTLE-from-PIV methods (VFI vs PFM vs FMC) and
  finds that **direct pathline-based FTLE computation matches
  separatrices ~80× better than velocity-field integration for
  noisy data**. Applicable to any experimental FSW validation.

## Consolidated new priority actions from the extended scan

Adding to Section "Revised priority ordering for Step 5" above:

| priority | new step | cost | rationale |
|---|---|---|---|
| P0 | **Step 9** integrator × interp pairing test (RK4+trilin vs RK6(4)+tricubic on 4lvl_hct) | days | Pokrajac 2002 predicts orders-of-magnitude improvement if paired right; falsifies (or confirms) our Part-5 tricubic-hurts finding |
| P1 | **Step 10** temporal-derivative snapshot inclusion | hours (if static) | García-Archilla 2022 – viscosity-independent bounds |
| P2 | **Step 11** LCS/FTLE mixing diagnostics on the 4lvl_hct outputs | day of numpy | Directly answers roadmap §4/§7 whether the ROM/grid PT damages mixing prediction — using the same tools the stirred-tank literature validates against |
| P3 | **Step 12** re-frame writeup around the stirred-tank literature ([17], [18], [25], [26], [29]) | writeup only | Positions our work as an application of a known toolkit to a new domain (FSW), not novel invention. Publishable framing. |

Step 9 in particular is worth prioritising: if RK6(4) + tricubic on
4lvl_hct beats RK4 + trilinear by even 2×, that would flip our Part-5
"tricubic hurts" verdict and open the accuracy floor we thought was
saturated at 7.8 mm cohort-mean rms.

## Extended references (Section E extension)

[14] [García-Archilla B, Novo J, Rubino S (2022), "POD-ROMs for incompressible flows including snapshots of the temporal derivative of the full order solution"](https://consensus.app/papers/details/c9d5fcdb253d5e449950094e5defb2bb/?utm_source=claude_desktop)
     (11 citations · SIAM J. Numer. Anal.)

[15] [García-Archilla B, Novo J, Rubino S (2023), pressure error bounds extension of [14]](https://consensus.app/papers/details/6c4e950b974853f9b6f84210849f5aa9/?utm_source=claude_desktop)
     (4 citations · J. Numerical Math.)

[16] [García-Archilla B, Novo J, Rubino S (2024), "POD-ROM methods: from a finite set of snapshots to continuous-in-time approximations"](https://consensus.app/papers/details/9ef2af95b09b5a95bc263fa192e10dfa/?utm_source=claude_desktop)
     (4 citations · ArXiv)

[17] [Li K et al. (2022), "Computation of Lagrangian Coherent Structures from Experimental Fluid Trajectory Measurements in a Mechanically Agitated Vessel"](https://consensus.app/papers/details/29f22ac5f77855d986d8ece317dd8efd/?utm_source=claude_desktop)
     (25 citations · Chemical Engineering Science)

[18] [Bashiri H et al. (2016), "Investigation of turbulent fluid flows in stirred tanks using a non-intrusive particle tracking technique"](https://consensus.app/papers/details/f6d548bfa3ea579cad6a25120b72eaa6/?utm_source=claude_desktop)
     (36 citations · Chemical Engineering Science)

[19] [Shadden S, Lekien F, Marsden JE (2005), "Definition and properties of Lagrangian coherent structures from finite-time Lyapunov exponents in two-dimensional aperiodic flows"](https://consensus.app/papers/details/f3e7724c147b537ab134203df30ecf9e/?utm_source=claude_desktop)
     (1439 citations · Physica D — canonical LCS/FTLE reference)

[20] [Pokrajac D, Lazic R (2002), "An efficient algorithm for high accuracy particle tracking in finite elements"](https://consensus.app/papers/details/a5ef30b6fa025145a95a68d5b8de485c/?utm_source=claude_desktop)
     (26 citations · Advances in Water Resources)

[21] [Beznosov O et al. (2025), "High order interpolation of magnetic fields with vector potential reconstruction for particle simulations"](https://consensus.app/papers/details/770bc529ee665f828855fad69f0c61a3/?utm_source=claude_desktop)
     (1 citation · Comput. Phys. Commun.)

[22] [Rössler T et al. (2018), "Trajectory errors of different numerical integration schemes diagnosed with the MPTRAC advection module driven by ECMWF operational analyses"](https://consensus.app/papers/details/9da7d93f3adc53009325a5c891c76f6d/?utm_source=claude_desktop)
     (23 citations · Geoscientific Model Development)

[23] [Coppola G et al. (2001), "Nonlinear particle tracking for high-order elements"](https://consensus.app/papers/details/e4c41a132aaa5219997d90d0f9ae35b4/?utm_source=claude_desktop)
     (45 citations · J. Comput. Phys.)

[24] [Yeung PK, Pope SB (1988), "An algorithm for tracking fluid particles in numerical simulations of homogeneous turbulence"](https://consensus.app/papers/details/7692f4fdedfe5db1b0442df2be5584ce/?utm_source=claude_desktop)
     (353 citations · J. Comput. Phys.)

[25] [Mikhaylov K et al. (2023), "Three-dimensional characterisation of macro-instabilities in a turbulent stirred tank flow and reconstruction from sparse measurements using machine learning methods"](https://consensus.app/papers/details/ef509961cc0a5608999805d1a476f09f/?utm_source=claude_desktop)
     (5 citations · Chemical Engineering Research and Design)

[26] [Mikhaylov K et al. (2021), "Reconstruction of large-scale flow structures in a stirred tank from limited sensor data"](https://consensus.app/papers/details/c7337d91c01b56509c904cacb993f229/?utm_source=claude_desktop)
     (18 citations · AIChE Journal)

[27] [Arosemena A et al. (2023), "Proper orthogonal decomposition modal analysis in a baffled stirred tank: a base tool for the study of structures"](https://consensus.app/papers/details/560762d290bd53899bd0352200d8f89d/?utm_source=claude_desktop)
     (2 citations · Flow)

[28] [Jiang Y et al. (2023), "Reduced-order modeling of solid-liquid mixing in a stirred tank using data-driven singular value decomposition"](https://consensus.app/papers/details/f58e418b94c05c99a99f91daa61bcf4f/?utm_source=claude_desktop)
     (18 citations · Chemical Engineering Research and Design)

[29] [Peace W et al. (2025), "Study of the dispersed and flooded aeration regimes in two-phase gas-liquid stirred tanks using Positron Emission Particle Tracking"](https://consensus.app/papers/details/f3539021842b5618b761b89b88cc464c/?utm_source=claude_desktop)
     (3 citations · Chemical Engineering Research and Design)

[30] [Qian Z et al. (2023), "Extraction of Lagrangian Coherent Structures in the framework of the Lagrangian–Eulerian Stabilized Collocation Method (LESCM)"](https://consensus.app/papers/details/29f44035f49f5b15a1bdca79a5157e25/?utm_source=claude_desktop)
     (14 citations · Computer Methods in Applied Mechanics and Engineering)

[31] [Lagares C et al. (2023), "A GPU-Accelerated Particle Advection Methodology for 3D Lagrangian Coherent Structures in High-Speed Turbulent Boundary Layers"](https://consensus.app/papers/details/e7dcf10b8c3c505b87d599d3a28c282f/?utm_source=claude_desktop)
     (7 citations · Energies)

[32] [Raben SG et al. (2013), "Computation of finite-time Lyapunov exponents from time-resolved particle image velocimetry data"](https://consensus.app/papers/details/3eeb3a7021d15e29a6b6d7ecccff317a/?utm_source=claude_desktop)
     (39 citations · Experiments in Fluids)
