<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# I've revised the roadmap and attached some recent results and also think about the next steps as attached. Evaluate critically and provides suggestions based on high impact publications and literature

This is an unusually rigorous, self-correcting body of work — the team caught and reversed their own hypothesis in §4, and the grid experiment's Part 2 addendum is a textbook example of not shipping a conclusion before checking the obvious confound (Stage-1 projection method). That said, measured against the ROM/Lagrangian and incompressible-ROM literature, there are several places where the interpretation could be sharper or the next-steps plan is chasing a less-likely fix. Here's the critical breakdown.

## What Holds Up Well Against the Literature

**The Eulerian/Lagrangian amplification finding is a clean empirical confirmation of a known asymmetry.** Case 003 — best Eulerian reconstruction (2.62%), worst Lagrangian gap (17.7%, 6.74× amplification)  — is exactly the counter-intuitive pattern the consistency-error literature predicts: because Lagrangian trajectories are path integrals of the velocity field, low point-wise error doesn't guarantee low trajectory error, and the two can even be *anti-correlated* across cases when the residual sits in dynamically sensitive regions. This is good, hard evidence, not just an assumption carried over from the earlier review.[^1]

**Self-correcting the smoothing hypothesis in §4 is exactly the right scientific behavior**, and the revised diagnosis — a coherent low-*k* mean-velocity bias in the workpiece (ROM overestimates |v| by 20–50% at r > 10mm) rather than high-*k* structure loss [^2] — is well-supported by three independent signals (radial profile offset, FFT plateau at low k, and faster particle transport through the annular probe). This directly matches a documented failure mode in POD literature: **energy-based mode truncation systematically under-resolves low-magnitude regions**, because modes are ranked by their contribution to total kinetic energy, and the workpiece's low-velocity bulk flow contributes little energy relative to the high-velocity pin rotation, so a 3-mode POD basis is essentially fit to the pin and the workpiece mean gets whatever residual falls out of that fit [^3][^4]. This is a specific, citable mechanism for the bias — not just "the ROM is imperfect," but "this is what energy-weighted POD structurally does when a field has regions with wildly different velocity magnitudes."

**The `uniform_half` result — the ultra-fine 17.5M-cell grid being the *worst* variant — is a genuine and well-explained finding**, and it's consistent with basic interpolation theory: oversampling a C⁰ piecewise-linear (raw P1) field with a higher-order interpolant doesn't create smoothness that wasn't there; it just resolves the field's inherent kinks more finely, and a smooth interpolant reacting to an unsmoothed kink is precisely the mechanism behind Runge-type overshoot phenomena. The report's own explanation (Stage-1 ceiling limits what Stage-2 refinement can extract) is correct and matches why Part 2's Stage-1 HCT-3D upgrade reverses the grid-family ranking.[^5]

## Where the Interpretation Should Go Further

**The workpiece mean-bias diagnosis correctly identifies the mechanism but stops short of naming the fix the literature already has for this exact problem.** The next-steps plan's Step 5 proposes two hypotheses — grid-space POD reduces anisotropic mesh-weighting artifacts, or a smoother (C¹) basis reduces Lagrangian amplification — but both treat the fix as a side effect of *where* the POD is computed (grid vs. mesh) rather than addressing *how* the POD basis is optimized. There's a more direct, higher-impact fix documented[^6]
<span style="display:none">[^10][^11][^12][^13][^14][^15][^16][^17][^18][^19][^20][^21][^22][^23][^7][^8][^9]</span>


<div align="center">⁂</div>

[^1]: rom_pt_step2_step3_report-2.md

[^2]: rom_pt_smoothness_divfree_report-3.md

[^3]: https://arxiv.org/html/2605.27756v2

[^4]: https://core.ac.uk/download/pdf/195294899.pdf

[^5]: rom_pt_step5_step6_grid_report-4.md

[^6]: rom_pt_next_steps_plan-5.md

[^7]: rom_feasibility_report.pdf

[^8]: recovery_and_interpolation_methods.md

[^9]: rom_pt_roadmap.md

[^10]: rom_pt_roadmap.md

[^11]: https://arxiv.org/abs/1808.05635

[^12]: https://arxiv.org/abs/1307.7888

[^13]: https://dspace.mit.edu/bitstream/handle/1721.1/130083/ReducedFTLE.pdf?sequence=1\&isAllowed=y

[^14]: https://www.sciencedirect.com/science/article/pii/S004578252400032X

[^15]: https://www.icas.org/icas_archive/ICAS2016/data/papers/2016_0409_paper.pdf

[^16]: https://pubs-en.cstam.org.cn/article/doi/10.6052/0459-1879-21-464

[^17]: https://egusphere.copernicus.org/preprints/2024/egusphere-2024-1171/egusphere-2024-1171-ATC1.pdf

[^18]: https://digital.csic.es/bitstream/10261/25697/1/tesina.pdf

[^19]: https://www.ross.aoe.vt.edu/papers/nolan-serra-ross-2020.pdf

[^20]: https://www.mdpi.com/2311-5521/5/4/189

[^21]: http://www.cds.caltech.edu/~marsden/bib/2005/19-ShLeMa2005/ShLeMa2005.pdf

[^22]: https://arts.units.it/retrieve/handle/11368/2992596/394790/2006.14428.pdf

[^23]: http://georgehaller.com/reprints/smallestFTLE.pdf

There's a more direct, higher-impact fix documented in the ROM literature that targets exactly the failure mode diagnosed in §4, rather than treating it as a side-effect of *where* the POD is computed. Xie, Nolan, Ross, Mou \& Iliescu propose **Lagrangian inner products** for constructing the ROM basis itself: instead of building the POD basis to maximize captured *kinetic energy* under the standard Eulerian L² inner product, they build it to maximize accuracy of *Lagrangian quantities* (they specifically target FTLE fields). Their result is striking and directly relevant here — the Lagrangian-basis ROM is "orders of magnitude more accurate" than the standard Eulerian ROM at predicting both Lagrangian fields *and* the underlying Eulerian field, with **no closure modeling of discarded modes at all** — the entire accuracy gain comes from which modes are prioritized in the basis. This reframes the roadmap's Step 5 hypothesis-testing: rather than asking "does moving the POD from mesh-space to grid-space help" (legs 1/2 in the plan ), a more literature-grounded leg 3 would be "does replacing the standard energy-weighted POD with a Lagrangian-inner-product POD fix the workpiece mean-bias directly, since that bias is precisely a case of low-energy-but-dynamically-important structure being discarded by the standard basis." This is a stronger candidate fix because it's purpose-built for exactly the Eulerian/Lagrangian consistency-error problem the team has already characterized, rather than a basis-domain change that only *might* incidentally help.[^1][^2]

**The divergence-free findings (§4, points 4–5) deserve a similar "there's a known fix" follow-up rather than being logged as a neutral observation.** The report correctly notes the FOM itself is not divergence-free at a non-negligible level, and the ROM inherits this structure with 99% cell-by-cell correlation to the FOM's divergence pattern. But the ROM/incompressible-flow literature has a standard remedy for exactly this: building the POD basis from a **discretely divergence-free velocity space** (e.g., a mixed/inf-sup-stable FEM velocity-pressure formulation, or explicitly projecting snapshots onto a divergence-free subspace before the SVD) rather than from raw nodal velocity snapshots. Given that a companion project already uses a mixed u/p/e formulation for FSW specifically to improve strain accuracy, there's a natural, low-effort path here: build the POD basis on the same discretely divergence-free velocity space that formulation already produces, rather than the raw reconstructed nodal velocities. This would attack the trapped-particle/div-free-proxy anomaly at its source instead of only diagnosing it.[^3][^4][^5]

**The "only 3 POD modes" ceiling is under-examined relative to how much weight the plan puts on basis-domain experiments.** The next-steps plan itself acknowledges this possibility only as a fallback ("if neither leg pans out... we would need more modes, not a different basis domain" ), but given how POD mode ranking works — energy-based selection will always favor the high-velocity pin region over the low-velocity workpiece bulk when only a handful of modes are kept — a 3-mode basis is close to a worst-case scenario for exactly the workpiece bias already found. The literature on POD mode-energy redistribution shows this is a structural property of energy truncation, not a domain-specific quirk, which means the SV-decay pre-check in Step 5 phase A is likely to show **very similar decay curves for mesh-space and grid-space POD** (since the underlying energy distribution across the flow doesn't change by resampling domain), and the team should weight that outcome as the *most probable* result rather than a fallback case, and prioritize a mode-count or basis-construction fix (more modes, or Lagrangian-weighted modes) over further grid-domain experiments.[^2][^6][^7]

## Where the Grid Experiment's Statistical Reach Should Be Tempered

**The MALMO-vs-block-refined reversal under HCT-3D Stage-1 is a genuinely important empirical finding, but it rests on only 4 cases, and the plan's own Step 3 sequencing note already flags a case-001 exception** (near-pin bin favoring a different winner than the aggregate). Given that reversal risk was just demonstrated once already (Part 1 → Part 2 raw-P1 vs HCT-3D flip), the team should treat the 4-case "top-3" ranking as provisional with real probability of another reshuffle at 20 cases, not just as a formality to confirm — the plan already schedules this correctly, but the report language ("current best practice") in the grid report risks understating that uncertainty for anyone reading only the summary.[^8][^2]

## Summary Table

| Finding | What the docs show | Literature-grounded refinement |
| :-- | :-- | :-- |
| Eulerian/Lagrangian amplification | Confirmed empirically, consistency-error framing correct [^9] | Consider Xie et al.'s Lagrangian-inner-product ROM basis [^1] as a direct fix, not just a diagnostic frame |
| Workpiece mean-velocity bias | Correctly diagnosed as coherent low-*k* bias, not smoothing [^3] | This is a textbook energy-truncation artifact [^6][^7]; likely persists regardless of mesh-vs-grid POD domain |
| Divergence-free violations | FOM and ROM both violate it; ROM inherits FOM's pattern [^3] | Discretely divergence-free POD bases are an established remedy [^4][^5], not yet on the roadmap |
| Grid Stage-1/Stage-2 interaction | HCT Stage-1 reverses the grid-family ranking [^8] | Consistent with approximation theory (can't extract more smoothness than the sampled field has); good methodological catch |
| 3-mode ceiling | Treated as a fallback explanation in Step 5 | Should be treated as the most likely outcome given how energy-based POD mode selection works |
| 4-case grid winners | Provisional, one exception already flagged | Treat "current best practice" language as provisional pending the 20-case rollout, not settled |

The overall trajectory of this work is unusually self-critical for an internal roadmap — reversing your own hypothesis in §4 rather than confirming it is exactly the right instinct. The main opportunity now is to stop treating the ROM basis itself as fixed and start testing basis-construction fixes (Lagrangian inner products, divergence-free projection, or simply more modes) alongside the grid/interpolation-engineering track, since the diagnostic work has already pointed quite specifically at the basis, not the interpolation pipeline, as the dominant error source.
<span style="display:none">[^10][^11][^12][^13][^14][^15][^16][^17][^18][^19][^20][^21]</span>

<div align="center">⁂</div>

[^1]: https://arxiv.org/abs/1808.05635

[^2]: rom_pt_next_steps_plan-5.md

[^3]: rom_pt_smoothness_divfree_report-3.md

[^4]: https://ir.cwi.nl/pub/30840/30840.pdf

[^5]: https://www.sciencedirect.com/science/article/abs/pii/S0045782511001617

[^6]: https://arxiv.org/html/2605.27756v2

[^7]: https://core.ac.uk/download/pdf/195294899.pdf

[^8]: rom_pt_step5_step6_grid_report-4.md

[^9]: rom_pt_step2_step3_report-2.md

[^10]: https://ar5iv.labs.arxiv.org/html/1810.03892

[^11]: https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2022.905392/full

[^12]: https://cwrowley.princeton.edu/papers/CompPOD.pdf

[^13]: https://www.sciencedirect.com/science/article/abs/pii/S0021999120302874

[^14]: https://link.springer.com/article/10.1007/s40314-025-03344-2

[^15]: https://arxiv.org/pdf/2010.06701.pdf

[^16]: https://ir.cwi.nl/pub/30087/30087.pdf

[^17]: https://iris.sissa.it/retrieve/handle/20.500.11767/124561/146945/PhDThesisZancanaroMatteo.pdf

[^18]: https://www.wias-berlin.de/people/john/BETREUUNG/diss_giere.pdf

[^19]: https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.5927

[^20]: https://www.sandia.gov/app/uploads/sites/127/2021/10/rom_jcp08.pdf

[^21]: https://www.academia.edu/616751/POD_based_Reduced_Order_Modelling_of_a_compressible_forced_cavity_flow



