# Data-driven surrogate for FSW — POD/SVD feasibility report

**Scope.** Can a reduced-order model (ROM) predict the full 3D outcome of
a friction-stir-welding (FSW) simulation from just two process inputs —
the advance velocity `v_adv` (INLET_VELOCITY) and the pin rotation
`ω_pin` (PIN_RPM)? This report covers the first-step feasibility study
using **PCA / Proper Orthogonal Decomposition (POD) via SVD**, on two
output representations:

1. the **Eulerian mean-density field** (a 3D voxel field per case), and
2. the **Lagrangian particle cloud** (per-particle displacement per case).

For each, it documents the full pipeline (pre-SVD preparation → SVD →
post-SVD validation) and the results of every variant we tried. The
detailed per-step math of the SVD and the normalization options lives in
the companion document [`rom_pca_svd.md`](rom_pca_svd.md); this report is
the higher-level, shareable summary.

> **Bottom line.** Linear POD is *feasible but not yet accurate*. With 17
> cases the density surrogate reaches ~28 % held-out error; restricting to
> the near-pin region improves it to ~26 %. The particle-displacement
> surrogate is harder (~37 %) and is dominated by data-quality issues (one
> runaway case). The dominant error source in every variant is the
> **regression from 2 inputs to mode amplitudes**, not the POD basis —
> i.e. the limiting factor is *sample count*, not representation.

---

## 1. Dataset

20 full-order FSW simulations on a Latin-hypercube-like sampling of
`(v_adv, ω_pin)`, with `v_adv ∈ [5.0e-3, 1.0e-2]` m/s and
`ω_pin ∈ [-800, -400]` RPM.

| quantity | density study | particle study |
|---|---|---|
| source file | `…/union/particles_union_density.vtkhdf` | `…/run_grid-frac_n*_s*/particles.vtkhdf` |
| field | `mean_density` (ImageData) | per-particle positions (trajectory) |
| snapshot | one static 3D field | final-step displacement of 360k particles |
| cases available | 19 (002 has no density file) | 20 (all cases) |
| cases used | 17 (see §2.1) | 20, or 19 excluding the 001 outlier |

**Authoritative parameters** come from each case's `run_jaxtrace.sh`
(`INLET_VELOCITY`, `DT`, `N_STEPS`, `PIN_RPM`). The `dt` column in
`case_parameters.csv` is inconsistent with the actual runs and must not be
used for timing.

A notable design property surfaced during the particle study:
**`v_adv · t_max = v_adv · DT · N_STEPS = 0.075` for every case** — the
runs were length-matched (higher `v_adv` → fewer steps) so all cases
advect the same total distance.

---

## 2. The Eulerian density-field study

### 2.1 Pre-SVD preparation

**(a) Common grid (correspondence).** PCA needs every snapshot as a vector
on *one shared grid*, so that "column j" means the same physical voxel in
every case. The native density grids differ per case (extent/origin/
spacing scale with `v_adv`), so we:

- built a **common reference grid** = the per-axis *intersection* bounding
  box of the kept cases, at a fixed resolution, and
- **trilinearly resampled** every case onto it
  (`jax.scipy.ndimage.map_coordinates`, GPU).

**(b) Case selection.** Cases **000 and 001** are x-extent outliers — a
strict all-case intersection would collapse the box to a ~34-voxel x-slab.
Dropping them (002 already missing) keeps the full wake. → **17 snapshots**
on a **319×72×73** grid.

**(c) Boundary trimming.** A margin is trimmed after resampling to remove
boundary artefacts: 1 voxel off every wall, plus **6 voxels off the z_max
wall** where an artificial high-density top layer forms. → **317×70×66 ≈
1.46 M voxels**.

**(d) Centring / normalization.** PCA always subtracts the **per-voxel mean
field** (the POD reference; a length-1.46 M *vector*, not a scalar — it has
real spatial structure). Optional further normalization was tested as a
variant (§2.4). Default: mean-centre only ("none").

### 2.2 SVD

Because the matrix is short-and-fat (17 × 1.46 M), we use the **method of
snapshots**: form the small Gram matrix `G = Xc Xcᵀ` (17×17), apply
`jax.numpy.linalg.svd` to it, and recover the spatial modes. This is
identical to a direct SVD of the centred matrix but avoids the GPU
`gesvd` failure on the wide matrix. Computation is float64. Full
derivation in [`rom_pca_svd.md`](rom_pca_svd.md).

### 2.3 Results — spectrum and coverage (elbow)

Mean-centred POD, 17 snapshots → 16 non-trivial modes (the 17th is the
null mode from centring).

| domain | grid | mode-1 energy | **modes for 90 % / 95 % / 99 %** |
|---|---|---|---|
| full | 317×70×66 | 38.7 % | **9 / 12 / 15** |
| first 20 % of x (near-pin) | 63×70×66 | 37.7 % | **10 / 13 / 15** |

There is **no sharp elbow** — the spectrum decays gently, meaning the
density response is genuinely multi-mode (≈8–11 modes for a faithful
field). Restricting to the near-pin region makes the elbow *slightly
harder* (the stir zone is where variance is least concentrated), but —
see §2.5 — it makes the field *easier to predict*.

Plots: `rom_pca_elbow_coverage.png`, `rom_pca_elbow_coverage_x20.png`.

### 2.4 Results — normalization comparison

Tested normalization options (applied before the SVD; see the math doc):

| normalize | what it does | 90 %/95 %/99 % modes | LOOCV (rbf) |
|---|---|---|---|
| `none` (default) | mean-centre only | 9 / 12 / 15 | 28.24 % |
| `per_case_l2` | ÷ each snapshot's L2 norm (shape vs amplitude) | 9 / 12 / 15 | **27.85 %** |
| `per_case_mass` | ÷ each snapshot's total mass | 9 / 12 / 15 | 28.27 % |
| `global_*` | ÷ one scalar (std/max/Frobenius) | identical to none | identical |
| `log` | `log1p(ρ)` (compress dynamic range) | 4 / 5 / 11 | 78.7 %* |

\* `log` error is measured in log-density space (not inverted), so it is
**not comparable** to the linear-space numbers; it is shown for spectral
shape only. `global_*` provably leaves modes and coverage unchanged
(it only rescales σ).

**Finding:** normalization barely moves the held-out error
(`per_case_l2` is marginally best). The field representation is not the
bottleneck. Plots: `rom_pca_normalize_compare*.png`,
`rom_loocv_normalize_compare*.png`.

### 2.5 Results — held-out accuracy (LOOCV)

See §4 for what LOOCV means and why it differs from projection error.
Surrogate = PCA basis + a regressor `(v_adv, ω_pin) → mode coefficients`.

| domain | regressor | best K | **mean held-out error** | per-case min–max |
|---|---|---|---|---|
| full | **rbf** | 7 | **28.2 %** | 19.3 – 48.8 % |
| full | poly | 4 | 29.3 % | 19.7 – 48.5 % |
| full | gp | 4 | 31.1 % | 20.8 – 51.1 % |
| near-pin (x20) | **rbf** | 7 | **25.9 %** | 17.6 – 48.2 % |
| near-pin (x20) | poly | 3 | 26.8 % | 18.4 – 47.0 % |
| near-pin (x20) | gp | 3 | 27.5 % | 18.6 – 47.5 % |

For reference, the **projection error** (train on all 17, basis
expressiveness only) at K=7 is ~**12 %** — so roughly *half* the held-out
error is POD truncation and half is the input→coefficient regression.

**Spatial pattern (2D error map).** The error is consistently largest in
the **high-|ω| / low-v_adv corner** (cases 015, 018, 008) and along the
sampling convex-hull edges — i.e. where the 17 points sample most
sparsely. Plots: `rom_loocv_error_maps*.png`, `rom_loocv_error_vs_modes*.png`.

**Why near-pin is better.** Cropping to the first 20 % of x removes the
downstream wake, whose *length* depends strongly on `v_adv` and is hard to
regress from two inputs with 16 training points. The near-pin stir-zone
structure generalizes better.

---

## 3. The Lagrangian particle-cloud study

### 3.1 Pre-SVD preparation

**(a) Correspondence — free.** Every case seeds the **same 360 000
particles** on the **same t=0 uniform grid** (60×120×50) with **identical
ParticleID ordering** (verified byte-identical). So particle *i* is the
same seed location in every case, and a per-particle displacement vector
stacks directly across cases — no resampling needed. Snapshot = final-step
displacement, concatenated over components: `[Δx₁..Δx_N, Δy.., Δz..]`.

**(b) Representations tested.** Four candidate snapshot definitions:

| name | definition | intent |
|---|---|---|
| `final` | final positions `x_final` | raw, includes seed layout |
| `raw` | `Δ = x_final − x_seed` | displacement |
| `comoving` | `ξ = Δ − v_adv·t_max·ê_x` | co-moving frame (subtract bulk drift) |
| `seedrel` | per-particle seed-relative (empirical drift) | data-calibrated drift |

The drift `v_adv·t_max = 0.075` (constant; from `run_jaxtrace.sh`).

**(c) Centring.** As always, PCA subtracts the per-particle mean
displacement before the SVD.

### 3.2 Two structural findings (important)

**Finding 1 — drift subtraction does NOT change the SVD modes.** Because
correspondence is exact, subtracting a *constant per-case* x-shift (and the
shared seed grid) is a rigid offset that is **fully absorbed by the
per-snapshot mean-centring** PCA already does. Consequently `final`, `raw`,
and `comoving` give **identical singular values and modes** — the three
curves lie exactly on top of each other. Drift subtraction matters only for
*interpreting/visualizing* the mean field (it becomes a physical residual),
not for the mode structure. Plot: `rom_particles_elbow_compare*.png`.

**Finding 2 — "eliminating unaffected particles" is moot at this scale.**
The seed spacing is tiny (Δp ≈ 1.7e-4) compared with typical co-moving
displacements (~5e-3 … 7e-2), so **>97 % of particles exceed even
|ξ| > 5·Δp**. There is essentially no "unaffected" set to threshold away.
Any |ξ|-threshold either removes almost nothing or, if pushed high,
destroys the shared-column correspondence. Plot:
`rom_particles_displacement_stats*.png`.

### 3.3 Results — spectrum, coverage, accuracy

**Case 001 is a runaway outlier** (`max|ξ| ≈ 2.0` vs ~0.07 for all
others). It alone accounts for **94 % of the total variance**, which made
the all-case spectrum look misleadingly compressible (1 mode = 90 %).
Excluding it gives the honest picture:

| set | mode-1 energy | 90 %/95 %/99 % modes | LOOCV (rbf) |
|---|---|---|---|
| all 20 cases | 94.4 % | 1 / 2 / 8 | 99 % (unusable) |
| **excluding 001** | 38.8 % | **11 / 14 / 18** | **36.9 %** (K=8) |

LOOCV excluding 001: rbf 36.9 % (K=8), gp 37.9 % (K=3), poly 39.0 % (K=4).

**Interpretation.** Excluding the outlier, the particle-displacement
spectrum resembles the density one (mode-1 ≈ 39 %, gentle decay), but the
**held-out error is worse (~37 % vs ~28 %)**: predicting a per-particle
*Lagrangian* displacement field from two scalars is harder than predicting
the *Eulerian* density. The runaway case 001 should be investigated as a
data-quality issue before the particle ROM is trusted.

---

## 4. Methodology — projection vs LOOCV, and the regressors

Two error numbers appear throughout; conflating them is the most common
mistake.

**Projection (truncation) error** — fit PCA on *all* cases, reconstruct
case *i* from its *own* (true) coefficients and the first K modes. Measures
**only how well K modes span the data**; always improves with K; says
nothing about predicting a new case. (~12 % at K=7 for density.)

**LOOCV (leave-one-out cross-validation)** — the real generalization test:

```
for each case i:
    PCA on the other n-1 cases            → mean, modes Φ
    fit regressor g: inputs → coefficients (on the n-1)
    predict held-out coeffs   Â_i = g(inputs_i)     ← inputs only
    reconstruct               ρ̂_i = mean + Σ_k Â_ik φ_k
    error_i = ‖ρ̂_i − ρ_i‖ / ‖ρ_i‖
```

The held-out case touches neither the PCA nor the regressor fit. Because a
surrogate cannot project an unseen field (it has no access to it), **LOOCV
necessarily tests PCA + regression together** — that is the actual
surrogate. The gap between projection (~12 %) and LOOCV (~28 %) is the
regression error.

**The three regressors** (all on standardized inputs; deliberately
lightweight for ≤17 points):

- **`rbf`** — radial-basis-function interpolation (thin-plate spline,
  scipy). Smooth, exact at samples, tuning-free; best performer here.
- **`gp`** — a small anisotropic-RBF Gaussian process (NumPy, no sklearn).
  Adds predictive uncertainty; slightly noisier with so few points.
- **`poly`** — least-squares degree-2 polynomial in `(v_adv, ω_pin)`.
  Interpretable baseline.

The **error-vs-modes curve** has a characteristic dip: too few modes
underfit, too many overfit the noisy high-mode coefficients. The minimum
(K ≈ 4–7 here) is the bias/variance sweet spot.

---

## 5. Summary of all variants

| study | variant | snapshots | 90 % modes | best LOOCV | notes |
|---|---|---|---|---|---|
| density | full domain, none | 17 | 9 | **28.2 %** (rbf) | baseline |
| density | full domain, per_case_l2 | 17 | 9 | 27.9 % (rbf) | normalization ≈ no effect |
| density | near-pin (first 20 % x) | 17 | 10 | **25.9 %** (rbf) | best density result |
| density | log normalization | 17 | 4* | n/a | log-space, not comparable |
| particles | comoving, all cases | 20 | 1 | 99 % | dominated by 001 outlier |
| particles | comoving, excl. 001 | 19 | 11 | **36.9 %** (rbf) | honest particle result |
| particles | final / raw / comoving | — | identical | — | drift subtraction ⇒ same modes |

---

## 6. Conclusions and recommendations

1. **Feasibility: yes, accuracy: not yet.** Linear POD + simple regression
   gives ~26–28 % held-out density error from two inputs. Promising for a
   first pass, not production-accurate.

2. **The bottleneck is data, not representation.** Projection error is ~12 %
   while LOOCV is ~28 %, and neither normalization nor the choice among
   reasonable representations moves the needle. The limiting factor is
   **17 training points in a 2D input space**, sparsest exactly where error
   peaks (high-|ω| / low-v_adv corner).

3. **Region matters.** Focusing on the near-pin region (dropping the
   v_adv-sensitive wake) measurably improves predictability.

4. **Particles are harder and noisier.** The Lagrangian displacement ROM is
   ~37 % and is contaminated by a runaway case (001). Drift subtraction —
   though physically motivated — does not change the POD modes here, and
   |ξ|-thresholding cannot separate "affected" particles at this seed
   spacing.

**Recommended next steps**

- **Add samples**, prioritising the high-|ω| / low-v_adv corner and the
  convex-hull edges where LOOCV error concentrates.
- **Investigate case 001** (runaway particles) before trusting the particle
  ROM; cross-check against the known vmapped-RK4 particle-loss issue.
- For particles, try a **seed-region mask** (particles seeded near the pin)
  instead of a |ξ|-threshold — a shared, correspondence-preserving way to
  isolate the stir zone.
- Beyond linear POD: **kernel-POD / autoencoder** latent spaces, or a GP
  with uncertainty to *drive* the next sampling (active learning).

---

## Appendix — reproducing the results

All code is on branch `feature/rom-svd`, package `jaxtrace/rom/`.

```bash
# Density: elbow + normalization comparison, then LOOCV
python run_rom_pca.py    --out-dir rom_out
python run_rom_loocv.py  --out-dir rom_out
python run_rom_loocv.py  --out-dir rom_out --compare-normalize --compare-regressor rbf
# Density: near-pin first 20% of x (separate filenames via --tag)
python run_rom_pca.py    --out-dir rom_out --x-keep-fraction 0.2 --tag x20
python run_rom_loocv.py  --out-dir rom_out --x-keep-fraction 0.2 --tag x20

# Particles: stats + elbow comparison + LOOCV (exclude the 001 outlier)
python run_rom_particles.py --out-dir rom_out --mode comoving
python run_rom_particles.py --out-dir rom_out --mode comoving --exclude 001 --tag no001
```

Output files (`rom_out/`, regenerable; not committed):

- Density: `rom_pca_*`, `rom_loocv_*` (with `_x20` for the near-pin run).
- Particles: `rom_particles_*` (with `_no001` for the outlier-excluded run).

Math details: [`rom_pca_svd.md`](rom_pca_svd.md).
