# PCA / POD via SVD — the density-surrogate pipeline

This document describes, step by step, exactly what
[`jaxtrace/rom/pca.py`](../jaxtrace/rom/pca.py) does to reduce the FOM
density snapshots to a small linear basis, and the precise meaning of
every mean / normalization factor. It is the math behind the elbow and
coverage plots produced by [`run_rom_pca.py`](../run_rom_pca.py).

---

## 0. What goes in

We have **`n` density snapshots** (after dropping the outlier cases, `n = 17`).
Each snapshot is one full 3D `mean_density` field, resampled onto a common
grid and flattened to a vector of length **`m`** (`m ≈ 1.46 million` voxels
after trimming).

Stack them as rows of a **snapshot matrix**

$$
X \in \mathbb{R}^{n \times m}, \qquad
X_{i,:} = \rho_i \;\; \text{(case } i \text{'s flattened field)} .
$$

- rows = cases (one per `(v_adv, ω_pin)` pair), `n = 17`
- columns = voxels, `m ≈ 1.46e6`

This matrix is **short and fat** (`m ≫ n`). That single fact drives two
choices later: at most `n−1` non-trivial modes exist, and we factor a
small `n×n` matrix instead of the wide `n×m` one.

---

## The pipeline at a glance

```
X  (n×m, raw density)
│
├─ step 1  per-case scale      (optional)   X ← X / s_case      [rows]
├─ step 2  log transform       (optional)   X ← log1p(X)
├─ step 3  per-voxel centring  (ALWAYS)     Xc ← X − mean       [the POD reference]
├─ step 4  global scale        (optional)   Xc ← Xc / g         [one scalar]
│
├─ step 5  Gram matrix                      G ← Xc Xcᵀ          (n×n)
├─ step 6  SVD of G                         G = V diag(σ²) Vᵀ
├─ step 7  recover spatial modes            Φ ← diag(1/σ) Vᵀ Xc
└─ step 8  coefficients                     A ← V diag(σ)        (= Xc Φᵀ)
```

Steps 1–4 are the **normalization chain**, applied in this exact order.
Which of the optional steps fire is selected by the single
`normalize=` argument. Steps 5–8 are the SVD itself and never change.

---

## Step 1 — per-case scale (optional)

Divide **each snapshot (row) by one scalar computed from that row**,
*before* anything else. This removes per-case *amplitude* so the modes
describe *shape*. The scalar `s_i` depends on the option:

| option | scalar `s_i` for case `i` | meaning |
|---|---|---|
| `per_case_l2` | $s_i = \lVert \rho_i \rVert_2 = \sqrt{\sum_v \rho_{i,v}^2}$ | Euclidean length of the field vector |
| `per_case_mass` | $s_i = \sum_v \rho_{i,v}$ | total "mass" (sum of all voxels) |
| `per_case_max` | $s_i = \max_v \rho_{i,v}$ | peak density |

Then

$$
X_{i,:} \leftarrow \frac{X_{i,:}}{s_i}, \qquad
s_i \leftarrow 1 \text{ if } s_i \le 0 .
$$

Concretely:

- **`l2`** makes every snapshot a **unit-length vector**. Two fields with
  the same spatial pattern but different intensity become identical rows.
- **`mass`** makes every snapshot **integrate to 1** — like turning each
  density field into a probability distribution. Useful when the *where*
  matters but the *how much* does not.
- **`max`** makes every snapshot **peak at 1**. Sensitive to a single hot
  voxel, so it is the least robust of the three.

The `s_i` are stored as `per_case_scale` and **re-multiplied back in**
during reconstruction, so the surrogate still predicts physical density.
In a full ROM you would also regress `s_i` on `(v_adv, ω_pin)` as a
separate scalar output.

> **Why this is the only normalization that changes the model.** It
> reweights the *rows* relative to each other, which changes the
> covariance structure and therefore the modes. In this dataset the
> snapshot peaks vary ~5× across cases, so the leading modes of the raw
> matrix are tilted toward the few high-peak cases; per-case scaling
> removes that tilt.

For every other `normalize` value, step 1 is skipped (`s_i = 1`).

---

## Step 2 — log transform (optional, `normalize="log"`)

Replace each entry by

$$
X \leftarrow \log(1 + X), \qquad X \leftarrow \max(X, 0) \text{ first.}
$$

`log1p(x) = log(1+x)` is used instead of `log(x)` so that empty voxels
(`x = 0`) map to `0` instead of `−∞`, and small values stay well-behaved.

**What it does:** density spans several orders of magnitude — a bright
stir-zone core and a faint wake. Plain PCA chases *variance*, which is
dominated by the loud core, so the faint-but-physically-important wake
structure barely registers. The log compresses the dynamic range:

$$
\underbrace{10^6}_{\text{core}} \rightarrow \log(1+10^6) \approx 13.8,
\qquad
\underbrace{10^3}_{\text{wake}} \rightarrow \log(1+10^3) \approx 6.9 .
$$

A 1000× ratio in linear space becomes a 2× ratio in log space, so the
wake now contributes comparably to the modes. This is why `log` reaches a
given coverage in **fewer** modes here — it is a genuinely different,
often more efficient basis for multi-scale fields.

> **Caveat:** the modes now live in log-density space. `reconstruct()`
> does *not* invert the log, so `log` is offered as a spectral-shape
> diagnostic, not a physical-units reconstruction. (Inverting would be
> `expm1`, easy to add if you decide to model in log space.)

`log` and `per_case_*` are mutually exclusive in the current code (one
`normalize` value selects one path).

---

## Step 3 — per-voxel mean centring (ALWAYS)

Compute the **mean field** — the per-column average over the `n` cases:

$$
\bar{\rho}_v = \frac{1}{n}\sum_{i=1}^{n} X_{i,v}, \qquad
\bar{\rho} \in \mathbb{R}^{m},
$$

and subtract it from every row:

$$
X_c = X - \mathbf{1}\,\bar{\rho}^{\top}, \qquad
(X_c)_{i,:} = \rho_i - \bar{\rho}.
$$

This is the single most important step, and the place where intuition
most often goes wrong, so to be explicit:

- **The mean is a per-voxel *vector* of length `m`, not a single scalar.**
  It is the average *density field*, with full spatial structure (high in
  the stir zone, ~0 at the edges). It is literally "mode 0" / the POD
  reference field.
- PCA/POD decomposes the **fluctuations about this mean field**,
  $\rho_i - \bar{\rho}$. Subtracting a single global scalar instead would
  leave the whole mean shape masquerading as a fluctuation and waste the
  first real mode reproducing it.

So: **mean → per-voxel vector (required). Always on, for every
`normalize` option.** It is computed *after* steps 1–2, i.e. on whatever
field those steps produced.

---

## Step 4 — single global scale (optional)

Divide the **entire centred matrix by one scalar** `g`:

| option | scalar `g` | definition |
|---|---|---|
| `global_std` | std of all centred entries | $g = \operatorname{std}(X_c)$ |
| `global_max` | largest centred magnitude | $g = \max_{i,v} \lvert (X_c)_{i,v} \rvert$ |
| `global_frobenius` | Frobenius norm | $g = \lVert X_c \rVert_F = \sqrt{\sum_{i,v}(X_c)_{i,v}^2}$ |

$$
X_c \leftarrow \frac{X_c}{g}, \qquad g \leftarrow 1 \text{ if } g \le 0 .
$$

**This is a cosmetic rescale.** Dividing the whole matrix by a constant
multiplies every singular value `σ` by `1/g` but leaves the **modes Φ**,
the **relative spectrum**, and the **coverage curve completely
unchanged**. We verified this empirically: `global_std`, `global_max`,
`global_frobenius` and `none` all give *identical* 10-mode reconstruction
error.

So its only purpose is to make the coefficients land in a tame O(1) range
for a downstream regressor. It is a single scalar — never per-voxel,
never per-case — exactly as a uniform unit-change should be. `g` is
stored as `global_scale` and undone in reconstruction.

> **z-score warning.** A "standardize each feature" PCA would divide each
> *column* by its own std (per-voxel). We deliberately do **not** do this:
> with 1.46M voxels of the *same* physical quantity, per-voxel scaling
> would inflate near-empty edge voxels to unit variance and drown the
> signal. Per-voxel scaling is for features with *different units*; a
> single homogeneous field uses mean-centring (+ optional global scalar).

---

## What "centred" matrix the SVD actually sees

After steps 1–4 the matrix handed to the SVD is

$$
Z = \frac{1}{g}\Big( T(X / s) - \overline{T(X/s)} \Big),
$$

where `s` is the per-case scaling (step 1, identity unless `per_case_*`),
`T` is the optional log (step 2, identity unless `log`), the overbar is
the per-voxel mean (step 3), and `g` is the global scalar (step 4,
`1` unless `global_*`). For the default `normalize="none"`, this reduces
to the textbook POD centring $Z = X - \bar{\rho}$.

---

## Step 5 — Gram matrix (method of snapshots)

The honest, direct thing would be to SVD `Z` itself:
$Z = U \Sigma \Phi$ with `Φ` the spatial modes. But `Z` is `17 × 1.46M`,
and a GPU `gesvd` on a matrix that wide is both wasteful and numerically
fragile (it actually failed with a cuSolver internal error). The
*method of snapshots* avoids it.

Form the small **Gram matrix**

$$
G = Z Z^{\top} \in \mathbb{R}^{n \times n} \quad (17 \times 17),
$$

i.e. $G_{ij} = \langle z_i, z_j \rangle$, the inner products between
centred snapshots. `G` is symmetric and positive semi-definite.

---

## Step 6 — SVD of the small matrix

Apply `jax.numpy.linalg.svd` to `G`. Because `G` is symmetric PSD, its
SVD coincides with its eigendecomposition:

$$
G = V \, \operatorname{diag}(\sigma_1^2, \dots, \sigma_n^2) \, V^{\top},
$$

so the SVD's singular values of `G` are the **squared** singular values
of `Z`, and the singular vectors `V` are the snapshot-space eigenvectors.
Recover the singular values of `Z`:

$$
\sigma_i = \sqrt{\max(\text{Sg}_i, 0)} .
$$

(The clip to 0 guards against tiny negative round-off in the PSD
eigenvalues.) These `σ_i` are the `singular_values` plotted in the elbow.

Because we mean-centred `n` snapshots, `G` has rank at most `n−1`, so the
**last singular value is ~0** (the null direction). That is the cliff you
see at mode 17 in the plots — it is expected, not a bug.

---

## Step 7 — recover the spatial modes

The right singular vectors of `Z` (the actual 3D mode shapes, length `m`)
come back via

$$
\phi_i = \frac{1}{\sigma_i}\, Z^{\top} v_i ,
$$

stacked as `modes` $\Phi \in \mathbb{R}^{n \times m}$ (row `i` is mode
`i`). In code this is the single matmul
`(V * (1/σ)).T @ Z`, with `1/σ` set to 0 for the null mode. The `φ_i`
are orthonormal density-field patterns; reshaped to the grid they are
plottable fields.

---

## Step 8 — coefficients (mode amplitudes)

Each snapshot's coordinates in the mode basis:

$$
A = V \operatorname{diag}(\sigma) \in \mathbb{R}^{n \times n},
\qquad A = Z \Phi^{\top},
$$

`coeffs` row `i` = how much of each mode case `i` contains. **These are
the regression targets** of the next stage: learn the map
$(v_{adv}, \omega_{pin}) \mapsto A_{i,:}$, then a new parameter pair
predicts coefficients → reconstruct a field.

The exact (truncated) reconstruction in the SVD's working space is

$$
Z_i \approx \sum_{k=1}^{K} A_{ik}\, \phi_k .
$$

---

## Reconstruction back to physical density

`PCAResult.reconstruct(n_modes=K)` inverts the chain in reverse:

1. fluctuation in working space: $f = A_{:, :K}\,\Phi_{:K}$
2. undo the global scale and re-add the mean field:
   $\;Y = \bar{\rho} + g\, f$  (note: `g` multiplies the **fluctuation
   only**, never the mean)
3. undo the per-case scale: $\;\rho_i = s_i\, Y_i$

This recovers physical density for all options **except `log`**, whose
`expm1` inverse is intentionally not applied (see step 2). Full-rank
(`K = n`) reconstruction round-trips to ~`1e-15` for every non-log
option — the numerical proof the chain is consistent.

---

## Energy, coverage, and the elbow

The "energy" of mode `k` is its captured variance $\sigma_k^2$. The
diagnostics are:

$$
\text{per-mode fraction} = \frac{\sigma_k^2}{\sum_j \sigma_j^2},
\qquad
\text{cumulative coverage}(K) = \frac{\sum_{k=1}^{K}\sigma_k^2}{\sum_j \sigma_j^2}.
$$

`n_modes_for(c)` returns the smallest `K` whose cumulative coverage
reaches `c`. The **elbow plot** shows `σ_k` (log scale) vs `k` — a sharp
elbow would mean a few modes suffice; a gentle slope (what we see) means
the response is genuinely multi-mode. The **coverage plot** shows
cumulative coverage vs `K`, annotated at 90/95/99 %.

Because global scaling (step 4) cancels in every ratio above, the
coverage curve is invariant to `global_*` — only `none` vs `per_case_*`
vs `log` produce distinct curves. That is exactly what
`run_rom_pca.py --compare-normalize` overlays.

---

## Quick reference — which `normalize` to use

| value | step 1 | step 2 | step 4 | changes modes? | physical reconstruction? |
|---|---|---|---|---|---|
| `none` (default) | – | – | – | baseline | yes |
| `global_std/max/frobenius` | – | – | ÷ scalar | **no** (cosmetic) | yes |
| `per_case_l2/mass/max` | ÷ row scalar | – | – | **yes** (shape vs amplitude) | yes |
| `log` | – | log1p | – | **yes** (multi-scale) | no (no expm1 yet) |

For the feasibility elbow, `none` is the correct standard POD. `log` is
the most interesting alternative to compare, since it best handles the
order-of-magnitude spread between the stir-zone core and the wake.

---

# Part II — measuring surrogate error

The PCA above gives a *basis*. A *surrogate* must, given only the two
inputs $(v_{adv}, \omega_{pin})$ of an **unseen** case, produce its full
density field. There are two very different error numbers, and conflating
them is the most common mistake — so they get their own section.

## Projection (truncation) error — basis expressiveness only

Fit PCA on **all** `n` cases, then reconstruct case `i` from its first
`K` modes:

$$
\rho_i^{\text{proj}} = \bar{\rho} + \sum_{k=1}^{K} A_{ik}\,\phi_k,
\qquad
e_i^{\text{proj}} = \frac{\lVert \rho_i^{\text{proj}} - \rho_i \rVert_2}
                         {\lVert \rho_i \rVert_2}.
$$

Crucially, the coefficients $A_{ik}$ here are the **true** ones, obtained
by projecting case `i`'s *own* density onto the modes. Case `i` was in the
training set. So this measures only **how well `K` modes span the data** —
it always decreases as `K` grows and says *nothing* about predicting a new
case. On this dataset, $K=7$ gives ~**11.8 %** mean projection error.

Use it to answer "is the linear subspace rich enough?" — not "is the
surrogate accurate?".

## LOOCV (leave-one-out cross-validation) — true generalization

To predict an **unseen** case we cannot project its density (we don't have
it). We only have its inputs. So the surrogate needs a second piece: a
**regressor** $g:(v_{adv}, \omega_{pin}) \mapsto (A_1,\dots,A_K)$ that
predicts the mode coefficients from the inputs. **PCA + regressor together
is the surrogate**, and LOOCV tests exactly that:

```
for each case i = 1..n:
    train = all cases except i
    1. PCA on `train`            -> mean ρ̄, modes Φ, training coeffs A_train
    2. fit regressor g on train  : inputs_train -> A_train
    3. predict held-out coeffs   : Â_i = g(inputs_i)        ← inputs only!
    4. reconstruct               : ρ̂_i = ρ̄ + Σ_k Â_ik φ_k   (chain undone)
    5. error                     : e_i = ‖ρ̂_i − ρ_i‖ / ‖ρ_i‖
```

The held-out case never touches the PCA fit *or* the regressor fit, so
$e_i$ is an honest generalization error. Sweeping `K` and averaging $e_i$
over the folds gives the **error-vs-modes** curve; plotting $e_i$ at the
best `K` over the $(v_{adv}, \omega_{pin})$ plane gives the **2D error
map**. (This is the standard LOOCV your colleagues use; "train on n−1,
test on 1, repeat" is its definition.)

> **Why LOOCV ≫ projection.** Projection at $K=7$ is ~12 %; LOOCV at
> $K=7$ is ~28 %. The gap is the *regression* error: the modes represent
> the fields well, but predicting their amplitudes from just two scalars
> with only 16 training points is the hard part. That gap — not the
> projection number — is the real feasibility signal.

> **Why the error-vs-modes curve has a dip.** Adding modes first helps
> (less truncation) but eventually hurts: the higher modes carry little
> energy and their coefficients are noisy functions of the inputs, so the
> regressor overfits them. The minimum of the LOOCV curve is the
> bias/variance sweet spot — here $K \approx 4$–$7$.

### The per-case-scale subtlety under `per_case_*`

When `normalize="per_case_*"`, the training coefficients live in
amplitude-removed space, so the regressor predicts a *shape*. The removed
amplitude $s_i$ is itself regressed on the inputs (a second, scalar
regression) and multiplied back in at reconstruction, so $\hat\rho_i$ is
in physical units. For `none`/`global_*`, $s_i\equiv1$ and this step is a
no-op. For `log`, the error is computed in log-density space (the log is
not inverted), so **`log` LOOCV numbers are not comparable** to the
linear-space ones — the comparison plot flags this and puts `log` on its
own axis/scale.

## The three regressors

All three map standardized inputs (each of $v_{adv},\omega_{pin}$ shifted
to zero mean / unit std on the training fold) to the coefficient vector.
They are deliberately lightweight — 16 training points cannot support a
heavy model.

**1. `rbf` — radial basis function interpolation**
(`scipy.interpolate.RBFInterpolator`, thin-plate-spline kernel,
`smoothing=0`). Writes each coefficient as a weighted sum of radial
kernels centred on the training points:

$$
\hat A(p) = \sum_{j=1}^{n_{\text{train}}} w_j\,\varphi(\lVert p - p_j\rVert)
            + (\text{low-order polynomial}),
\qquad \varphi(r)=r^2\log r .
$$

The weights $w_j$ are solved so the interpolant passes **exactly** through
every training value. Smooth, tuning-free, and well-suited to scattered
low-dimensional data — the best performer here. Weakness: extrapolation
outside the convex hull of the samples is unreliable.

**2. `gp` — Gaussian process regression**
(a small anisotropic-RBF GP implemented in NumPy — no scikit-learn
dependency). Models the coefficient as a Gaussian process with a squared-
exponential kernel:

$$
k(p,p') = \exp\!\Big(-\tfrac{\lVert p - p'\rVert^2}{2\,\ell^2}\Big),
\qquad
\hat A(p) = k(p, P)\,\big[K(P,P) + \sigma_n^2 I\big]^{-1} A_{\text{train}} .
$$

The length scale $\ell$ is set from the median pairwise distance of the
training points; a tiny noise $\sigma_n^2$ keeps the solve stable. A GP
also yields a predictive variance (uncertainty), useful later for active
sampling. With only 16 points it is slightly noisier than RBF here.

**3. `poly` — least-squares polynomial**
Fits the coefficients as a degree-2 polynomial in $(v_{adv}, \omega_{pin})$:

$$
\hat A(p) = c_0 + c_1 v + c_2 \omega + c_3 v^2 + c_4 v\omega + c_5 \omega^2,
$$

with $c$ from ordinary least squares. The simplest, most interpretable
baseline; underfits if the true response is more wiggly than quadratic,
but it is a useful floor — if a fancy regressor can't beat the polynomial,
the limitation is the data, not the model. On this dataset it lands
between RBF and GP.

## Reading the two LOOCV plots

`run_rom_loocv.py` produces:

- **`rom_loocv_error_vs_modes.png`** — mean held-out error vs `K`, one
  curve per regressor. Pick the regressor and `K` at the lowest dip.
- **`rom_loocv_error_maps.png`** — the held-out error at the best `K`
  drawn as a filled contour over the $(v_{adv}, \omega_{pin})$ plane, with
  the 17 cases marked. This shows *where* in parameter space the surrogate
  is weak — typically the sparsely sampled corners (here the high-$|\omega|$
  / low-$v_{adv}$ region) and the convex-hull edges.

`run_rom_loocv.py --compare-normalize` instead fixes one regressor and
overlays the LOOCV result across normalizations
(`rom_loocv_normalize_compare.png`), to see whether `per_case_*` or `log`
buys anything over plain `none`. (On this data they do not move the linear-
space error much — confirming the bottleneck is the regression, not the
field normalization.)
