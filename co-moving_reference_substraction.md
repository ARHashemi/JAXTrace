<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Before continue, I want you to evaluate something and make it clear for implementation. For working with particles, I observe that there are a significant regions in the domain that the particles just affected by advancing velocity and simply walking like x=v_adv*t. So, I'm thinking to somehow remove them from the domain or do something to make the other variations more significant. For example, subtract the x=v_adv*t from their displacement. As well, for the density field, I'm thinking to subtract the initial density or something like that from the density field. I'm not sure how to do it to the union of particles (from all time steps) and the time averaged density field?

Your instinct is exactly right and connects to a well-studied problem called the **transport-dominated SVD failure**. Here is the full formalisation and implementation plan.

***

## Why Vanilla SVD Fails Without This Preprocessing

The SVD finds the best *linear* subspace approximating all 20 particle clouds. But if most particles simply travel with $V_{adv}$ (a rigid translation of the whole cloud), then the snapshot matrix looks like:

$$
\mathbf{X}^{(k)} = \underbrace{\mathbf{X}^{ref}(t) + V_{adv}^{(k)} \cdot t \cdot \hat{\mathbf{e}}_x}_{\text{dominant, trivial transport}} + \underbrace{\boldsymbol{\xi}^{(k)}(t)}_{\text{small, physically interesting perturbation}}
$$

The dominant singular modes of the raw snapshot matrix capture the bulk translation — i.e., the leading POD modes all look like "everyone moving to the right." The physically interesting variation (stirring near the pin, mixing width differences between configurations) is buried in small singular values and gets truncated. This is exactly the **Kolmogorov n-width problem** for transport-dominated systems.[^1][^2]

The solution is to **remove the trivial transport before compression**, so that SVD operates only on the non-trivial residual. This is the core idea behind Shifted POD (sPOD).[^3][^1]

***

## Part 1 — Particles: What to Subtract and How

### The Correct Reference Frame Transformation

For particle $i$ in configuration $k$, define the **co-moving displacement** (the position in the frame that travels with the bulk advection):

$$
\boxed{\boldsymbol{\xi}_i^{(k)}(t) = \mathbf{x}_i^{(k)}(t) - \mathbf{x}_{i,0} - V_{adv}^{(k)}\, t\, \hat{\mathbf{e}}_x}
$$

where:

- $\mathbf{x}_i^{(k)}(t)$ — raw particle position in configuration $k$ at time $t$
- $\mathbf{x}_{i,0}$ — initial seed position (same for all configurations if seeding is identical)
- $V_{adv}^{(k)}\, t\, \hat{\mathbf{e}}_x$ — the trivial bulk advection contribution

This gives you the **perturbation displacement** $\boldsymbol{\xi}_i^{(k)}$: how much each particle deviates from where it would be if it had simply travelled with the weld advance. Particles far from the tool have $\boldsymbol{\xi}_i^{(k)} \approx \mathbf{0}$ for all $k$ — they are genuinely uninteresting and compress to near zero automatically. Particles near the pin have large $\boldsymbol{\xi}_i^{(k)}$ that vary strongly with $(V_{adv}, \omega_{pin})$.[^4][^1]

### What About the $y$ and $z$ Components?

In the $y$ (transverse) and $z$ (depth) directions there is no bulk transport — the reference frame subtraction is zero for those components. The full transformation is:

$$
\xi_{i,x}^{(k)}(t) = x_i^{(k)}(t) - x_{i,0} - V_{adv}^{(k)}\,t
$$

$$
\xi_{i,y}^{(k)}(t) = y_i^{(k)}(t) - y_{i,0}
$$

$$
\xi_{i,z}^{(k)}(t) = z_i^{(k)}(t) - z_{i,0}
$$

### For the Union of Particles Across All Time Steps

If you stack all snapshots in time (not just the final state), the snapshot matrix for the ROM has one column per $(k, t_j)$ pair:

$$
\mathbf{A}_{particles} = \bigl[\boldsymbol{\xi}^{(1)}(t_0),\; \boldsymbol{\xi}^{(1)}(t_1),\; \ldots,\; \boldsymbol{\xi}^{(20)}(t_{N_t})\bigr] \in \mathbb{R}^{3N_p \times (20 \cdot N_t)}
$$

Critically: before stacking, **normalise time** to $\tau = t/T_{pass}^{(k)}$ where $T_{pass}^{(k)}$ is the time for the tool to fully pass the seeding zone in configuration $k$. This ensures the time axis is comparable across configurations with different $V_{adv}$. Without this, the column at index $j$ represents "t = 2 seconds" for a fast config but "t = 5 seconds" for a slow config — not the same physical event.

### The Particle Masking Question — Which Particles to Keep?

After subtracting the bulk advection, particles far from the tool have $\|\boldsymbol{\xi}_i\|_2 \approx 0$ across all configurations. Including them in the SVD dilutes the signal — they inflate $N_p$ without contributing information. Three options:


| Approach | How to implement | Trade-off |
| :-- | :-- | :-- |
| **Radial mask** | Keep only particles within a fixed distance $r_{mask}$ from the tool path centreline | Simple; $r_{mask}$ is a tunable parameter |
| **Variance threshold** | Keep particle $i$ if $\max_k \|\boldsymbol{\xi}_i^{(k)}(T)\|_2 > \epsilon_{mask}$ | Adaptive to actual data; no geometric assumption |
| **Weighted SVD** | Keep all particles but weight them by their variance across configurations | No data loss; computationally heavier |

The **variance threshold** is recommended because it requires no knowledge of the domain geometry and directly selects particles that vary between configurations — the particles the ROM needs to represent.

***

## Part 2 — Density Field: What to Subtract and How

### The Three Candidate Reference Fields

For the density field, there are three natural choices of what to subtract, each with a different physical meaning:

**Option D1 — Subtract the initial (uniform) density:**

$$
\rho'(\mathbf{x}, t) = \rho(\mathbf{x}, t) - \rho_0
$$

This removes the background level and centres the field around zero. It helps numerically but does not remove any spatial structure — the whole transported blob is still present in $\rho'$. This is the weakest preprocessing.

**Option D2 — Subtract the time-averaged density (across all configs):**

$$
\bar{\rho}(\mathbf{x}) = \frac{1}{20 \cdot N_t} \sum_{k,j} \rho^{(k)}(\mathbf{x}, t_j), \qquad \rho''^{(k)}(\mathbf{x}, t) = \rho^{(k)}(\mathbf{x}, t) - \bar{\rho}(\mathbf{x})
$$

This removes the mean spatial pattern shared across all configurations and all times. What remains is the configuration-to-configuration variation and the time evolution. This is the standard POD preprocessing (the "method of snapshots" mean subtraction).  The SVD of $\rho''$ represents deviations from the ensemble mean, which is exactly what the ROM needs to interpolate between configs.[^5]

**Option D3 — Subtract the co-moving reference density (sPOD approach):**

$$
\rho_{ref}(\mathbf{x}, t) = \rho_0\bigl(\mathbf{x} - V_{adv}^{(k)}\, t\, \hat{\mathbf{e}}_x\bigr)
$$

where $\rho_0(\mathbf{x})$ is the initial density distribution. This is the exact analogue of the particle co-moving frame: it tracks where the density blob *would* be if it simply advected rigidly at $V_{adv}$. The residual $\rho^{(k)} - \rho_{ref}$ captures only the distortion of the density blob due to stirring, mixing, and shear near the pin.[^2][^1]

This is the most powerful preprocessing but requires that you can evaluate $\rho_0$ at shifted spatial locations — straightforward if the density is computed on a regular grid (just index-shift the array), harder on an unstructured mesh (requires interpolation).

### For the Time-Averaged Density Specifically

If you want to build a ROM for the **time-averaged density** (one field per configuration, not time-dependent), the reference subtraction is:

$$
\langle\rho\rangle^{(k)}(\mathbf{x}) = \frac{1}{N_t}\sum_j \rho^{(k)}(\mathbf{x}, t_j), \qquad \text{subtract: } \overline{\langle\rho\rangle}(\mathbf{x}) = \frac{1}{20}\sum_k \langle\rho\rangle^{(k)}(\mathbf{x})
$$

The snapshot matrix for the SVD is then the $20$ mean-subtracted time-averaged fields, one column per configuration. This is the cleanest and most directly interpretable setup: the POD modes of the residual describe *how the average material distribution pattern changes as you vary the welding parameters*.[^6][^5]

***

## Summary: Recommended Preprocessing Pipeline

```
Raw particles {x_i^(k)(t)}          Raw density ρ^(k)(x,t)
        |                                     |
  Subtract bulk advection              Subtract ensemble mean
  ξ_i^(k)(t) = x_i^(k)(t)           ρ''(x,t) = ρ(x,t) - ρ̄(x)
             - x_i,0                          |
             - V_adv^(k)·t·ê_x        For time-avg density:
        |                              subtract config-mean
  Normalise time axis                  of time-averaged fields
  τ = t / T_pass^(k)                          |
        |                                     |
  Apply variance threshold             Apply co-moving shift (D3)
  mask: keep particles with            if SVD decay is still slow
  max_k ||ξ_i^(k)(T)||₂ > ε                  |
        |                                     |
  Stack into snapshot matrix           Stack into snapshot matrix
  A ∈ R^{3N_p_masked × (20·N_t)}      A ∈ R^{N_grid × 20}
        |                                     |
       SVD → check singular               SVD → check singular
       value decay                        value decay
```

The key diagnostic at the end is the same for both: if the preprocessed singular value decay reaches 95% energy within 5–8 modes, linear SVD-based ROM is viable. If not, the autoencoder architecture is needed — but the same preprocessing must still be applied first, because even an autoencoder trains far better on the perturbation field than on the raw advection-dominated data.[^7][^4]
<span style="display:none">[^10][^11][^12][^13][^14][^15][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://www.cfd.tu-berlin.de/~reiss/ReissSchulzeSesterhennMehrmann2018.pdf

[^2]: https://ar5iv.labs.arxiv.org/html/1803.01805

[^3]: https://d-nb.info/1156018617/34

[^4]: https://arxiv.org/html/2603.19702v1

[^5]: https://www.sandia.gov/app/uploads/sites/127/2021/10/FY16_SAND_report.pdf

[^6]: https://www.sciencedirect.com/science/article/pii/S2666821126002541

[^7]: https://ar5iv.labs.arxiv.org/html/2204.12670

[^8]: https://d-nb.info/1239158734/34

[^9]: https://pubs.acs.org/doi/10.1021/acs.iecr.3c01477

[^10]: https://gmd.copernicus.org/articles/16/5339/2023/

[^11]: https://web.stanford.edu/group/frg/course_work/CME345/CA-AA216-CME345-Ch4.pdf

[^12]: https://www.zora.uzh.ch/id/eprint/79218/1/pub_f.pdf

[^13]: https://www.osti.gov/servlets/purl/1843556

[^14]: http://maeresearch.ucsd.edu/tartakovsky/Papers/lu-2020-lagrangian.pdf

[^15]: https://tpfm.it.cas.cz/im/im/proceeding/2022/16

