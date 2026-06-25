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
| density | weld-pitch inputs (§7.1) | 17 | — | 28.3 % | no gain over baseline |
| density | velocity-coeff inputs (§7.3) | 17 | — | 27.6 % | marginal gain (~0.6 %) |
| density | all first-stage fields (§7.3) | 17 | — | 29.5 % | overfits (12 features) |
| particles | velocity-coeff inputs (§7.3) | 19 | — | 37.8 % | no gain |
| particles | manifold geometry (§8) | 18 | 11 lin / ~3 intr | — | curved manifold, ω-led MDS axis 1 |
| density | manifold geometry (§8) | 17 | 9 lin / ~2.5 intr | — | curved manifold, mixed-param axis 2 |

---

## 6. Conclusions and recommendations

1. **Feasibility: yes, accuracy: not yet.** Linear POD + simple regression
   gives ~26–28 % held-out density error from two inputs. Promising for a
   first pass, not production-accurate.

2. **The bottleneck is data, not representation.** Projection error is ~12 %
   while LOOCV is ~28 %, and neither normalization, input features (weld
   pitch, log, first-stage ROM coefficients — §7), nor the choice among
   reasonable representations moves the needle. **Active-subspace analysis
   (§7.2) confirms the response is `v_adv`-dominated and low-rank in the
   inputs**, so no input combination unlocks accuracy. The limiting factor is
   **17 training points in a 2D input space**, sparsest exactly where error
   peaks (high-|ω| / low-v_adv corner).

3. **Region matters.** Focusing on the near-pin region (dropping the
   v_adv-sensitive wake) measurably improves predictability.

4. **Particles are harder and noisier.** The Lagrangian displacement ROM is
   ~37 % and is contaminated by a runaway case (001). Drift subtraction —
   though physically motivated — does not change the POD modes here, and
   |ξ|-thresholding cannot separate "affected" particles at this seed
   spacing.

5. **The data manifold is curved, not flat (§8).** Independent
   intrinsic-dimension estimates (~2–3) sit well below the linear PC count
   (9–11 for 90 %) — the signature of a smooth but *nonlinear* manifold,
   cleanly organised by `(v_adv, ω)`. This explains the slow SVD decay and
   identifies nonlinear reduction (autoencoder / kernel-POD) as the
   *eventual* — not current, given the sample count — route to lower error.

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

## 7. Follow-up studies (post-review)

After an external review, four additional lines of work were carried out.
Each was a hypothesis aimed at the diagnosed bottleneck (the input →
mode-coefficient regression). The honest summary up front: **none of the
input-side changes materially beat the two-scalar baseline** — which is
itself an important, consistent result confirming the bottleneck is sample
count, not the input representation.

### 7.1 Richer input features (weld pitch, log)

We tested mapping the raw inputs to physics-motivated features before the
regression — most importantly the **weld pitch** `p_w = v_adv / |ω_pin|`,
the advance-per-revolution often cited as the dominant FSW transport
parameter.

| feature set | density LOOCV (rbf) | particles LOOCV (rbf) |
|---|---|---|
| `(v_adv, ω)` identity (baseline) | **28.2 %** | **36.9 %** |
| `(p_w, ω)` | 28.3 % | 37.8 % |
| `(p_w, v_adv)` | 30.6 % | 38.4 % |
| `p_w` only | 36.8 % | diverges |
| `(log v_adv, log|ω|)` | 29.1 % | 36.8 % |

**Weld pitch does not help, and `p_w`-only is much worse.** This refutes
the intuition that the response collapses onto weld pitch.

### 7.2 Active-subspace analysis — *why* pitch fails

To explain 7.1 we computed the first-order **active subspace** of the
input → coefficient map: a linear sensitivity per mode, energy-weighted,
eigendecomposed to find the dominant input direction.

| dataset | eigenvalue ratio λ₁/λ₂ | activity share | dominant raw direction |
|---|---|---|---|
| density full | 8.3 | 89 % / 11 % | ≈ **pure `v_adv`** `[1, 0]` |
| density near-pin | 8.9 | 90 % / 10 % | ≈ **pure `v_adv`** `[1, 0]` |
| particles (excl 001) | 3.0 | 75 % / 25 % | ≈ `v_adv`, but ω matters more |

**The map is nearly 1-D — but the active direction is essentially `v_adv`,
not weld pitch.** The coefficients barely depend on ω, so collapsing onto
`p_w` (which mixes ω in) is the *wrong* reduction, exactly explaining why
`pitch_only` degrades. Particles are genuinely more 2-D (ω contributes
more), consistent with them being harder to predict.

### 7.3 First-stage ROM coefficients as inputs

A colleague's **first-stage ROM** is a POD of the FOM Displacement
(velocity), Pressure and Temperature fields, stored as per-case reduced
coefficients (`cylindrical.som.fswrom.romdata`: Displacement 3 modes,
Pressure 4, Temperature 3 → up to 10 coordinates per case). The idea: use
these physics-rich coordinates instead of, or in addition to, the two raw
scalars for the *second-stage* density/particle regression.

*Verification first.* The files carry no case-ordering metadata; we
confirmed the 20 coefficient rows are in case order 000–019 by correlation
(Displacement Mode 1 vs ω = **+0.999**; Mode 2 vs v_adv = **−0.931**).

All three first-stage fields were tested as inputs — velocity
(Displacement), temperature, and pressure — alone and appended to
`(v_adv, ω)`:

| input | density LOOCV (rbf) | particles LOOCV (rbf) |
|---|---|---|
| `(v_adv, ω)` baseline | 28.2 % | 36.9 % |
| velocity coeffs **alone** (3) | **27.6 %** | 37.8 % |
| `(v,ω)` + velocity (5) | 28.0 % | 37.9 % |
| temperature coeffs alone (3) | 27.8 % | 37.5 % |
| `(v,ω)` + temperature (5) | 28.6 % | 37.6 % |
| all fields alone (10) | 28.9 % | 38.4 % |
| `(v,ω)` + all fields (12) | 29.5 % | 38.8 % |

**Marginal at best.** Velocity coefficients alone are the best of the set
but give only a small real gain for density (27.6 % vs 28.2 %) and none
for particles; piling on all fields **overfits** (12 features, 17–19
points → best-K collapses to 3). A sensitivity check explains it:
first-stage **modes 1–2 of each field are nearly linear in `(v_adv, ω)`**
(R² 0.87–1.0, i.e. redundant), and only **mode ≥ 3 carries new
information** (89–97 %), which does not strongly drive the density/particle
modes. The first-stage coefficients are largely a *reparametrisation* of
the two scalars we already have.

#### How the coefficients are passed to the regressor

The second-stage regressor accepts an arbitrary feature matrix via the
`extra_features` / `use_base_features` arguments of `loocv`. The
first-stage coefficients are loaded and aligned to the study's case order,
then handed in directly — **no spatial basis is involved** (see the note
below):

```python
from jaxtrace.rom import load_dataset, loocv
from jaxtrace.rom.first_stage import load_first_stage_coeffs

ds = load_dataset(verbose=False)                  # 2nd-stage target (density)
fs = load_first_stage_coeffs()                    # reads .romdata only

# (n_cases, d) feature matrix, rows aligned to ds.case_numbers.
# modes=None keeps every retained mode of each field.
ef = fs.select(fields=["Displacement"],           # velocity only
               modes={"Displacement": 3},          # cap modes if desired
               case_numbers=ds.case_numbers)

# (a) coefficients ALONE as inputs (drop the raw scalars):
loocv(ds.matrix, ds.params, ds.case_numbers, regressor="rbf",
      extra_features=ef, use_base_features=False)

# (b) APPEND coefficients to (v_adv, omega):
loocv(ds.matrix, ds.params, ds.case_numbers, regressor="rbf",
      extra_features=ef, use_base_features=True)
```

`extra_features` is a plain `(n_cases, d)` array, so any external per-case
descriptor can be used the same way. `fs.select(fields=[...])` chooses
which fields and how many modes to include; the driver
`run_rom_first_stage.py` simply sweeps the configurations in the table
above.

#### Is the `.basis` file needed? No — and why

The first-stage POD writes a field as

> field_i(x) ≈ mean(x) + Σ_k a_ik · φ_k(x)

where the **coefficients `a_ik`** (scalars, in `.romdata`, ~218 KB) are
the per-case reduced coordinates and the **basis `φ_k(x)`** (vectors of
length 180 461, in `.basis`, ~30 MB) are the spatial mode shapes, *shared
by all cases*.

- The second-stage regressor takes a **low-dimensional description of each
  case as input**. That description *is* the coefficient vector `a_i` — a
  handful of numbers. So only `.romdata` is read.
- The **basis is only needed to reconstruct or visualise a physical field**
  (`coeffs × basis + mean → field(x)`), e.g. to evaluate a predicted
  velocity at a point or plot a mode shape. The surrogate does none of
  that, so the 30 MB basis is never loaded.
- One subtlety that justifies using the coefficients directly: scalar
  coefficients are only comparable across cases when they refer to the
  **same basis**. Here the colleague built a single global first-stage POD,
  so all 20 cases share one basis — which is exactly why their `a_ik` are
  meaningful, alignable features. We rely on that guarantee but do not need
  to load the basis to use it.

In short: **`.romdata` suffices for the second-stage ROM; `.basis` would
only be required to turn predicted coefficients back into full fields.**

### 7.4 Cases 000/001 reruns (in progress)

The two cases dropped from the studies (000/001) are being re-run.
Diagnosis: their tracking scripts were structurally correct (only INPUT /
N_STEPS / INLET_VELOCITY differed from good cases, with `DT = 3.75e-3`
matching their RPM = −400). Fresh `run_jaxtrace.sh` / `run_union.sh` were
regenerated by cloning a known-good case (014) and substituting the
per-case values, keeping the original DT (a clean control). If the
particle runaway (case 001, `max|ξ| ≈ 2.0`) persists after a clean rerun,
the next step is a finer DT (1.875e-3). Results pending; once available,
all studies should be re-run with the fuller dataset (≈19 density / 20
particle cases).

### 7.5 Net conclusion of the follow-up

Three independent input-side levers — **weld pitch**, **log inputs**, and
**first-stage ROM coefficients** — each failed to beat the two-scalar
baseline by more than ~0.5 %. The **active-subspace** result ties this
together: the response is `v_adv`-dominated and effectively low-rank in
the inputs, so no clever input *combination* unlocks accuracy. The
limiting factor is unambiguously the **number and placement of training
samples**, reinforcing the §6 recommendation to add cases in the
high-|ω| / low-v_adv corner.

---

## 8. Manifold-structure diagnostics

Sections 2–7 measured *surrogate accuracy*. This section instead probes
the *geometry* of the data itself: each case is one high-dimensional
snapshot (a 360 000×3 particle cloud, or a flattened density field), and
the 20 cases trace out a **manifold** in that space, parametrised by
`(v_adv, ω_pin)`. Understanding that manifold's dimension and linearity
explains *why* linear POD behaves as it does, and whether a nonlinear
method could ever help.

**Method order — linear/metric first, nonlinear embeddings last.** t-SNE
and UMAP are deliberately *not* used as the first tool: they are nonlinear
embeddings that can manufacture clusters and curvature which are not
present in the data, so their output is untrustworthy without a linear
anchor. We therefore use only methods that are cheap, deterministic, and
faithful (numpy/scipy, no sklearn/umap), in increasing sophistication.
Runaway-dominated cases 000/001 (particles) and 000/001/002 (density) are
excluded, since their non-physical outliers otherwise dominate every
pairwise distance.

### 8.1 The five diagnostics

1. **Linear spectrum** — mean-centred PCA explained-variance curve. How
   many linear modes are needed for 90 % energy = the *linear*
   dimensionality.
2. **Linearity residual** — relative L2 reconstruction error using the
   first *k* PCs, for all *k*. A fast drop to zero ⇒ linear & low-dim.
3. **Intrinsic dimension** — two estimators that do **not** assume
   linearity: *TwoNN* (ratio of 1st/2nd nearest-neighbour distances) and
   the *correlation dimension* (Grassberger–Procaccia: slope of
   `log C(r)` vs `log r`). These return a single number — the true
   dimension of the surface the data lies on, regardless of curvature.
4. **Pairwise distance matrix** — the full *n×n* case-to-case Euclidean
   distances; reveals clustering directly.
5. **Classical (Torgerson) MDS** — a *linear*, distance-preserving 2-D
   layout (see §8.3 for exactly how its axes arise), coloured by the
   parameters.

### 8.2 Results

| quantity | particles (18 cases) | density (17 cases) |
|---|---|---|
| PC1 explained variance | 36.4 % | 38.7 % |
| **linear PCs for 90 %** | **11** | **9** |
| **intrinsic dim (correlation)** | **≈ 3.4** | **≈ 2.5** |
| intrinsic dim (TwoNN) | 13.6* | 8.5* |
| MDS axis-1 corr (v_adv, ω) | (−0.45, **+0.86**) | (+0.18, **+0.88**) |
| MDS axis-2 corr (v_adv, ω) | (**−0.87**, −0.47) | (−0.72, +0.44) |

\* TwoNN is unreliable at *n* ≈ 17–18 points (it needs many samples); the
correlation dimension is the trustworthy estimate here. Both agree
qualitatively that the intrinsic dimension is **small** (≈ 2–3).

**The headline signature: high linear dimension, low intrinsic
dimension.** Both manifolds need ~9–11 linear PCs for 90 % energy, yet
their intrinsic dimension is only ~2–3. That gap is the *definition* of a
**curved (nonlinear) manifold**: the data lies on a low-dimensional
surface that is *bent* through the high-dimensional space, so a linear
basis can only wrap it by stacking many modes. (If the manifold were
flat, linear-PC count and intrinsic dimension would agree.) This is an
independent confirmation of the slow-SVD-decay / Kolmogorov-barrier
observation in §2.3 and the review — now quantified by a dimension
estimate rather than inferred from the spectrum alone.

### 8.3 How the two MDS axes are chosen — and why they differ

This is worth spelling out because it is the most easily misread plot.

**Construction.** Classical MDS does *not* pick two of the original
coordinates. From the *n×n* distance matrix `D` it forms the
double-centred matrix

> `B = −½ · J · D² · J`,  with `J = I − (1/n)·11ᵀ` (the centring operator),

and eigendecomposes `B = V Λ Vᵀ`. The embedding coordinates are
`xᵢ = √λᵢ · vᵢ` for the **two largest eigenvalues** `λ₁ ≥ λ₂`. So:

- **Axis 1 is the single direction of greatest spread** among the cases
  (largest λ); axis 2 is the orthogonal direction of next-greatest spread.
- They are ordered by *variance captured*, exactly like PCA — in fact for
  Euclidean `D`, classical MDS is PCA of the cases, so axis 1/2 are the
  first two principal directions of the case cloud.
- The axes are therefore **data-driven, not parameter-driven**: nothing
  forces them to align with `v_adv` or `ω`. We *measure* their alignment
  afterwards by correlating each axis with the two inputs (the table
  above).

**Why a near-1 negative eigenvalue appears.** Both manifolds show exactly
one small negative MDS eigenvalue. Classical MDS yields negatives only
when `D` is *not* exactly Euclidean-embeddable — i.e. when the manifold is
curved. A single tiny negative eigenvalue is a mild, expected curvature
signal, consistent with §8.2.

**Why the axes differ between particles and density.** The axes are the
principal directions of *different objects*, so they need not match:

- **Particles** — the snapshot is a Lagrangian point cloud. Its dominant
  variation (axis 1, λ₁) tracks **ω** (corr **+0.86**), and axis 2 tracks
  **v_adv** almost purely (corr **−0.87**). The two axes are *cleanly
  separated* into one-parameter-each coordinates: the particle manifold is
  curved but its two leading directions are nearly an orthogonal
  `(ω, v_adv)` chart.
- **Density** — the snapshot is the Eulerian field. Axis 1 also tracks
  **ω** (corr **+0.88**), but axis 2 is a **mixture** of `v_adv` (−0.72)
  *and* `ω` (+0.44) — the parameters are *not* cleanly separated onto
  orthogonal axes. Physically: the density field's spatial structure
  couples the two inputs (the wake length depends on `v_adv` while its
  internal pattern depends on `ω`), so no single MDS axis isolates
  `v_adv`.

The common feature — **axis 1 ≈ ω for both** — says pin rotation is the
single largest source of snapshot-to-snapshot variation. The difference —
**particles give a cleaner `(ω, v_adv)` split than density** — is
consistent with the active-subspace result (§7.2): the density response
is more strongly entangled with `v_adv` through the wake, so its
parameter directions are more mixed.

**How to read the MDS plot.** Points are the 18/17 cases laid out so that
on-page distance ≈ true high-dimensional distance. Colour = `v_adv`, point
size = `|ω|`. A smooth colour/size gradient across the layout (which both
plots show) means the manifold is **smoothly parametrised by the inputs**
— there are no isolated clusters or folds, just a continuous curved sheet.
That is the *good* news for ROM feasibility: the mapping
`(v_adv, ω) → snapshot` is continuous; it is merely *nonlinear*, which is
why linear POD pays a mode-count tax and why the bottleneck is sample
density on a curved surface rather than discontinuity.

### 8.4 Implication

The diagnosis "curved but smooth, intrinsic-dim ≈ 2–3, cleanly
parameter-organised" is precisely the regime where a **nonlinear**
reduction (autoencoder / kernel-POD / manifold-aware interpolation) can
eventually beat linear POD — *once there are enough samples* to constrain
it (the review estimates ~50–100 for a 2-D parameter space). At the
current 17–20 cases it would overfit, so the immediate lever remains
adding samples (§6); but §8 establishes that the *eventual* payoff route
is nonlinear compression, and gives the target intrinsic dimension (~3) a
latent space should aim for.

> Nonlinear embeddings (UMAP / t-SNE) can be added as a final confirmation
> step once `umap-learn` / `scikit-learn` are installed; any structure they
> show should already be visible in the MDS layout, or it is an artefact.

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

# Follow-up (§7): input features and first-stage ROM coefficients
python run_rom_loocv.py --out-dir rom_out --feature-transform pitch_omega   # §7.1
python run_rom_first_stage.py --target density   --out-dir rom_out          # §7.3
python run_rom_first_stage.py --target particles --out-dir rom_out          # §7.3

# Manifold-structure diagnostics (§8): spectrum, intrinsic dim, MDS
python run_rom_manifold.py --target particles --exclude 000 001 --out-dir rom_out
python run_rom_manifold.py --target density   --out-dir rom_out
```

Active subspace (§7.2) and feature transforms (§7.1) are library calls in
`jaxtrace.rom` (`active_subspace`, `FEATURE_TRANSFORMS`); first-stage
loading is `load_first_stage_coeffs`.

Output files (`rom_out/`, regenerable; not committed):

- Density: `rom_pca_*`, `rom_loocv_*` (with `_x20` for the near-pin run).
- Particles: `rom_particles_*` (with `_no001` for the outlier-excluded run).
- Follow-up: `rom_firststage_{density,particles}.{png,npz}`.
- Manifold (§8): `rom_manifold_{particles,density}.{png,npz}`.

Math details: [`rom_pca_svd.md`](rom_pca_svd.md).
