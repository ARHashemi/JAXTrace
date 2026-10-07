<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# I also tested the reduced order velocity fields for each case as the inputs of the regressor, explained in the updated attached document. No improvement. What is the origin of this inaccuracy and how can be resolved? Does it mean that the manifold of data for both density and particle are nonlinear that svd is not successful? Or we should conclude the regression method does not capture well the manifold or parameter space? Or anything else?

This is the most important question in your entire project so far. The answer requires distinguishing three separate failure modes clearly, because they have completely different fixes.

***

## The Three Candidate Explanations — Evaluated

### Explanation 1: The output manifold is nonlinear (Kolmogorov barrier)

This is the question of whether the *density or particle cloud*, as a function of $(v_{adv}, \omega_{pin})$, lives on a nonlinear manifold that SVD cannot span with few modes.

**Evidence from your data: mostly NO for the output manifold, but YES for the density field structure.**

The projection error at K=7 is ~12% for density. This means the 17 training snapshots *can* be represented well in a 7-dimensional linear subspace — the output manifold of your 17 points is *not grossly nonlinear*. If the manifold were strongly nonlinear, you would see the projection error itself staying high regardless of K, or requiring 15+ modes to reach 12%. You do not see this.[^1][^2]

However, the slow SVD decay (9 modes for 90% energy, no sharp elbow) is a genuine Kolmogorov barrier signal. This is distinct from the output manifold nonlinearity — it reflects that the **density field as a spatial object** is transport-dominated: blobs at different $v_{adv}$ values are spatially shifted relative to each other and are nearly orthogonal in $L^2$, requiring many modes to span the ensemble.  This is the problem that co-moving frame subtraction (sPOD) was designed to address — it would almost certainly collapse the 9-mode 90%-energy threshold to 3–4 modes. The fact that you did not apply it means you are paying a Kolmogorov tax in mode count, but it does not explain the LOOCV error.[^3][^1]

**Verdict:** The output manifold nonlinearity is *not* the primary bottleneck at this stage.

***

### Explanation 2: The regression method fails to capture the manifold

This is the question of whether the relationship between inputs and the $K$ POD coefficients is nonlinear in a way that your regressors (RBF thin-plate spline, GP, polynomial) cannot capture with 17 points.

**Evidence from your data: partially YES, but not fixable by switching regressors.**

The active subspace result from §7.2 is decisive here. The dominant input direction is essentially pure $v_{adv}$, with eigenvalue ratio $\lambda_1/\lambda_2 = 8.3$ (89% of sensitivity in one direction). This means the regression problem is effectively **1D** — you are predicting POD coefficients along a single input axis.[^4]

For a 1D regression problem with 17 points covering a $\sim 2\times$ range of $v_{adv}$, a thin-plate spline RBF should be near-optimal. There is nothing a more sophisticated regressor can do that the RBF cannot, because the function to be approximated has resolution limited by sample density, not by regressor flexibility. This is confirmed by the identical performance of polynomial, GP, and RBF — when the function is smooth and the sample count is the bottleneck, all reasonable regressors converge to the same error floor.[^5][^6]

The weld pitch test (§7.1) further confirms this: $p_w$-only is *worse* than $v_{adv}$-only. The response is not pitch-dominated — it is $v_{adv}$-dominated — and substituting the wrong 1D active coordinate costs accuracy. This is a physically interesting finding: the density distribution is more sensitive to how fast the workpiece feeds through the tool than to how fast the tool rotates.[^4]

**Verdict:** The regression method is adequate given the data. Switching to a more complex regressor will not help.

***

### Explanation 3: Sample starvation (the true bottleneck)

**Evidence from your data: unambiguously YES.**

Consider the quantitative decomposition of your LOOCV error:

$$
\underbrace{28.2\%}_{\text{LOOCV}} = \underbrace{12\%}_{\text{projection}} + \underbrace{16.2\%}_{\text{regression gap}}
$$

The regression gap of 16.2% comes from a specific, identifiable source: in LOOCV, when case $i$ is held out, the remaining 16 points must interpolate to an unseen $v_{adv}^{(i)}$ value. For the cases in the high-$|\omega|$/low-$v_{adv}$ corner, the nearest training point in $v_{adv}$-space is far away — exactly where the spatial density structure changes most rapidly (the wake length scales linearly with $v_{adv}$). [^4][^6]

The first-stage ROM coefficient experiment (§7.3) provides the cleanest confirmation. Displacement Mode 1 has **R² = 1.0 with $\omega$** and Mode 2 has **R² = 0.93 with $v_{adv}$**. The first-stage modes are essentially perfect linear functions of the two scalar inputs. This means your 10-dimensional first-stage feature vector carries *no new information* beyond $(v_{adv}, \omega_{pin})$ — it is a reparametrisation. If better input features cannot help, and if better regressors cannot help, the constraint is the data itself.[^4]

**Verdict:** The bottleneck is sample count and placement. Adding ~5–8 points in the high-$|\omega|$/low-$v_{adv}$ corner is the correct next action.

***

## The Deeper Structural Picture

Here is a unified explanation of why all three input strategies (raw scalars, weld pitch, first-stage coefficients) gave the same answer. All three are measuring the same underlying thing:

```
v_adv  ──────────────────→  Displacement Mode 2  ──┐
                            (R² = 0.93)             │
                                                    ├──→  density/particle modes
ω_pin  ──────────────────→  Displacement Mode 1  ──┘
                            (R² = 1.0)
```

Every input encoding you tested is a near-invertible transformation of $(v_{adv}, \omega_{pin})$. The information content is identical. Adding the first-stage coefficients is like adding extra copies of the same two numbers with slight noise — it cannot reduce the fundamental interpolation error, and adding too many copies causes overfitting (as you observed: 12 features → K collapses to 3).[^7][^4]

This is the **curse of dimensionality** in a very small-data regime. With 17 points in a 2D space that is effectively 1D (due to the active subspace), the effective sample density is ~17 points on a 1D line. Your worst held-out cases (48.8% error) are those at the extremes of this line — the convex hull boundary cases where LOOCV must extrapolate rather than interpolate.[^8][^5]

***

## What Can Actually Reduce the Error

Given the diagnosis, the interventions are ordered by expected impact:


| Intervention | Expected ΔLOOCV | Rationale |
| :-- | :-- | :-- |
| **Add 5–8 samples in high-\( | \omega | \)/low-$v_{adv}$ corner** |
| **Co-moving density preprocessing (sPOD)** | −3 to −5% projection error; small LOOCV effect | Reduces mode count needed for 90% energy; helps the regressor fit fewer, more physically coherent coefficients |
| **Near-pin domain restriction (already done)** | −2% (observed) | Removes the wake, whose length depends on $v_{adv}$ and inflates regression difficulty |
| **Autoencoder latent space instead of SVD** | Unknown; possibly none at this sample count | Nonlinear compression helps only if the output manifold is genuinely nonlinear — your data does not yet confirm this |
| **Better regressor (GP with ARD, neural net)** | ~0% | All regressors converge to the same error at 17 samples in a 1D active subspace |
| **More velocity/temperature input features** | ~0% (confirmed empirically in §7.3) | First-stage coefficients are linear reparametrisations of the two scalars |

The autoencoder question deserves a specific comment: at 17–20 training samples, a neural-network-based autoencoder will almost certainly **overfit** or **underfit** systematically, because it has far more parameters than data. The crossover point where nonlinear compression beats linear SVD typically requires ~50–100 training snapshots for a 2D parameter space of this complexity.  It is not the right tool for the current dataset size — but will become the right tool once you have added the additional samples.[^9][^7]
<span style="display:none">[^10][^11][^12][^13][^14][^15][^16]</span>

<div align="center">⁂</div>

[^1]: https://arxiv.org/html/2602.01397v1

[^2]: https://www.sciencedirect.com/science/article/abs/pii/S0377042722002758

[^3]: https://ris.utwente.nl/ws/portalfiles/portal/479692374/1-s2.0-S0167278924002501-main.pdf

[^4]: rom_feasibility_report.md

[^5]: http://www2.stat.duke.edu/~banks/611-lectures.dir/lect19.pdf

[^6]: https://core.ac.uk/download/pdf/10128737.pdf

[^7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9712510/

[^8]: https://sites.stat.washington.edu/courses/stat527/s14/slides/multivarkernels-projectionpursuit.pdf

[^9]: https://stefaniafresca.com/publication/journal-article-17/

[^10]: https://arxiv.org/html/2408.12415v1

[^11]: https://mathematical-coffees.github.io/slides/mc08-delon.pdf

[^12]: https://www.research-collection.ethz.ch/server/api/core/bitstreams/a4624237-c5db-46a3-b826-fba958cd61da/content

[^13]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5415687/

[^14]: https://arxiv.org/pdf/2603.19702.pdf

[^15]: https://www.nature.com/articles/s41592-018-0019-x

[^16]: https://www.mat.univie.ac.at/~mnonino/MonicaNonino.pdf

