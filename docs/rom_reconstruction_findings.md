# FSW-ROM velocity reconstruction — findings

> ## 2026-07-16 ERRATA — earlier numbers were wrong due to a loader bug
>
> Everything below the "Original writeup" heading was produced with a
> broken `load_basis`.  The Fortran writer emits basis vectors as
> `Basis_CompMode <component>  <mode>` (the *first* integer is x/y/z,
> the *second* is the mode index), documented at
> `Mod_som_FswROM.f90:486–487`.  Our loader iterated the two integers
> in the wrong order, producing a Frankenstein "mode k" that mixed the
> x-component of three different real modes into a single row.
> Reconstructions with those Frankenstein modes had 3–80 % rel_rms
> against the FOM, wildly dependent on case; the residual asymmetries
> we blamed on mass-weighted SVD, basis truncation, and case-specific
> transients were **all** downstream of this axis swap.
>
> The colleague's own loader
> (`DTFSW/RBFforFSW/rbfrom/loader.py::_extract_basis`) uses the
> correct regex.  Cross-checking the sources led straight to the fix.
>
> **After the fix**, on the same 20 cases at ts = 119:
>
> | Metric | Buggy loader | Fixed loader |
> |:---|---:|---:|
> | mean `centered` rel_rms | 38.18 % | **4.04 %** |
> | std | 23.32 % | **1.08 %** |
> | best case | 04 at 4.06 % | 03 at **2.62 %** |
> | worst case | 01 at 80.13 % | 00 at **6.40 %** |
> | mode norms `||φ_k||_2` | `[1.34, 1.34, 0.086]` (junk) | `[1.10, 1.10, 1.09]` (proper orthonormal) |
>
> Every claim in the "Original writeup" below about mass-weighted
> SVD, non-Euclidean-orthonormal modes, or 63 % truncation floor is
> **retracted**.  The 3-mode POD basis captures the FSW physics well;
> the FEMUSS reconstruction convention is exactly what our `centered`
> formula does; and our loader is now correct.
>
> Fix: `jaxtrace/rom/velocity_recon.py` axis order in `load_basis`.
>
> Recommendation on which formula to use:
>
> * `centered` (colleague convention, no sigma scaling) — averages
>   4.04 % on cylindrical.  This is the correct choice.
> * `c_over_sig` (previously the "best" empirically) — now the *worst*
>   under the fixed loader because the axis-swap bug happened to
>   damp its contribution.  Do not use.
>
> The rest of this document is preserved verbatim for provenance.
> Read below only if you want the diagnostic trail.

---

## Original writeup (2026-07-14, buggy loader, superseded)

**Scope**: cross-check the `--velocity-source rom` path in `run_tracking.py`
against the FEMUSS FSW-ROM implementation, quantify the residual error
across all 20 stored cases, identify where the residual is worst in
space and time, and diagnose the root cause.

**Files inspected**:
- `jaxtrace/rom/velocity_recon.py` — our loader/reconstructor
- `FemussROM/Sources/modules/solme/FrictionStirWelding/Mod_som_FswROM.f90` — colleague's FSW ROM driver
- `FemussROM/Sources/ParallelLibraries/SLEPc/Mod_SLEPcMatrix.f90` — projection + reconstruction primitives
- `FemussROM/Sources/ParallelLibraries/SLEPc/Mod_SVDMatrix.f90` — SVD + optional mass-weighting step

**Data**:
- `/scratch/shared/ROM/FOM/cylindrical.som.fswrom.basis` — SnapshotsMean + 3 stored modes on 180,461 nodes
- `/scratch/shared/ROM/FOM/cylindrical.som.fswrom.romdata` — 20 sigma values + 3×20 coefficient matrix
- `cylindrical_000.gid` … `cylindrical_019.gid` — full FOM PVTU sequences, timesteps 0–119

---

## TL;DR

1. **Our code exactly matches FEMUSS's convention** for reconstruction:
   `v(x) = SnapshotsMean(x) + Σ_k c_k · φ_k(x)` (no sigma scaling, no
   normalisation, plain unweighted sum). No bug in
   `jaxtrace/rom/velocity_recon.py`.
2. **The stored basis is truncated to 3 modes out of 20**. Sigmas 4–20
   were dropped from the basis file, so any linear combination of the
   3 remaining modes has a hard-floor residual equal to the L² norm of
   the discarded modes. The best-fit projection onto the 3 modes still
   leaves ~63 % of the FOM norm unexplained on the cases we tested.
3. **The stored basis was built with lumped-mass-weighted SVD**
   (`LUMPE ON` at training time). The Basis_CompMode arrays are
   `φ_k = M⁻¹ u_k`, mass-weighted (not Euclidean-orthonormal). Any
   downstream reconstructor working only from the basis file — including
   FEMUSS itself for a *new* snapshot — cannot recover the correct
   coefficients without the mass matrix.
4. **On the 20 stored cases, the `centered` (colleague-spec) formula
   averages 38.2 % rel_rms and is consistently ~7 pp *worse* than
   using SnapshotsMean alone**. `c_over_sig` (dividing c by sigma
   before contracting) averages 30.8 %, essentially tied with mean-alone,
   because the sigma-scaled correction is negligible.
5. **Spatially the residual is concentrated near the tool region** and
   near the top surface (z = 0). Outside the pin radius (r > 0.010) the
   FOM has near-zero motion; the ROM spuriously predicts motion there,
   which shows up as very large *relative* error even though the absolute
   error is small.
6. **Temporally, cases with slow-transient FOM (e.g. cases 0, 1) get
   worse over time** (from ~70 % at ts=11 to ~80 % at ts=116). Cases that
   reach steady state early (e.g. 4, 7) are essentially flat over time.
   This is expected — the ROM stores one coefficient vector per case,
   which represents an average / late-time approximation of that case.

---

## 1. Colleague spec vs. our code

Your colleague's description ("for each node, multiply the coefs and
basis and then add the corresponding mean available in
cylindrical.som.fswrom.basis") is exactly:

    v(node, case) = mean(node) + Σ_k c_k(case) · φ_k(node)

Our `centered` formula in
[jaxtrace/rom/velocity_recon.py:297–321](../jaxtrace/rom/velocity_recon.py#L297-L321):

```python
w = c
v = np.einsum("k,knj->nj", w, basis.modes[:n_use])   # Σ_k c_k · φ_k
v = v + basis.mean                                   # + mean
```

**Verbatim identical.** The code is correct.

FEMUSS itself reconstructs via
[SLEPcExternalFilter](../../FemussROM/Sources/ParallelLibraries/SLEPc/Mod_SLEPcMatrix.f90#L738-L759)
called from
[som_fswrom_ComputeROMSolutioninFOMspace](../../FemussROM/Sources/modules/solme/FrictionStirWelding/Mod_som_FswROM.f90#L413-L449):

```fortran
FilteredVector = 0.0_rp
do idofr = 1, a%ndofr
   call VecGetArrayF90(a%BasisV(idofr), auxpointer, ierr)
   do ipoin = 1, a%npoinLocal
      do idofn = 1, a%ndofn
         ispos = (ipoin-1)*a%ndofn + idofn
         FilteredVector(idofn,ipoin) =
             FilteredVector(idofn,ipoin) + auxpointer(ispos) * Vector(idofr)
      end do
   end do
end do
```

Plain unweighted sum `Σ_k c_k · φ_k`. Then in the FSW driver:

```fortran
if (a%FswROMData%kfl_substractmean) then
   snapshot(:,1:npoinlocal) = snapshot(:,1:npoinlocal) &
       + a%FswROMData%SnapshotsMean(:,1:npoinlocal)
end if
```

Mean added at the end. Our loader and reconstructor match this exactly.

**No sigma scaling**: sigmas are stored (`Sigma_Mode k`) but never used
in reconstruction — only informational (basis energy retained).

---

## 2. What was actually stored in the basis file

The FSW-ROM training pipeline for the cylindrical case followed this
route (line numbers refer to `Mod_som_FswROM.f90` / `Mod_SVDMatrix.f90` /
`Mod_SLEPcMatrix.f90`):

1. Assemble snapshot matrix S (`nsnap` × `n_nodes*3`) — one snapshot
   per training case (20 total).
2. Subtract SnapshotsMean (line 254 of `Mod_som_FswROM.f90`, gated by
   `kfl_substractmean`).
3. Call SLEPc `SVDSolve` — computes left singular vectors `u_i`,
   Euclidean-orthonormal (line 231 of `Mod_SVDMatrix.f90`).
4. **If `LUMPE ON` at input**: apply `SVDMassMultBasis` (line 258 of
   `Mod_SVDMatrix.f90`):
   ```fortran
   call BVMatMult(auxBasis, a%invLMass, a%Basis, ierr)
   ! Basis <- invLMass * Basis
   ```
   After this line, the stored modes are `φ_k = M⁻¹ u_k` — **not**
   Euclidean-orthonormal.
5. Truncate to the top-3 modes (line 293–317 of `Mod_som_FswROM.f90`)
   based on either `basis_energy_target` or an explicit
   `basis_number`.
6. Compute per-snapshot coefficients using
   [ExternalProj](../../FemussROM/Sources/ParallelLibraries/SLEPc/Mod_SLEPcMatrix.f90#L712-L736):
   ```fortran
   call VecMDot(auxvec, a%ndofr, a%BasisV, ProjVector, ierr)
   ! c_k = <snapshot, φ_k>_Euclidean
   ```
   This is a **plain Euclidean inner product** against the
   already-mass-weighted φ_k. Note: this is *not* the correct
   mass-weighted projection `<snapshot, φ_k>_M`, which would recover
   the SLEPc singular-vector amplitude exactly. FEMUSS accepts the
   inconsistency because the coefficients are used only to reconstruct
   the training snapshots they were computed from — and there
   `Σ_k c_k φ_k ≈ snapshot` up to the truncation error.
7. Write `Basis_CompMode k` (the mass-weighted modes),
   `Sigma_Mode k` (the Euclidean singular values), and
   `SnapshotsMean` to disk (lines 351–363 of `Mod_som_FswROM.f90`).

---

## 3. Empirical evidence that this is what happened on cylindrical

Diagnostic Gram matrix of the stored modes (Euclidean inner product):

```
G[i,j] = <φ_i, φ_j>_Euclidean =
  [[ 1.792   0.206   0.005]
   [ 0.206   1.806  -0.008]
   [ 0.005  -0.008   0.007]]
```

- Diagonal ≠ 1 → modes are **not Euclidean-orthonormal**.
- Off-diagonal ≠ 0 → modes are not even Euclidean-orthogonal.
- Diagonal magnitudes 1.79, 1.81, 0.0074 → mode 3 is much smaller in
  Euclidean norm than modes 1–2, consistent with a mass-weighted SVD
  where φ_3 = M⁻¹ u_3 has been compressed.

Projection experiment on case 1's own FOM sequence, projecting the
centred FOM onto φ_k using `c_k = <u, φ_k>_Euclidean`:

| Timestep | Best-fit projection `c_k`   | ‖proj − stored c‖ |
|---------:|:----------------------------|------------------:|
|   30     | `[16.09, -0.29,  0.26]`     | 13.83             |
|   60     | `[16.41, -0.42,  0.27]`     | 13.51             |
|   90     | `[16.50, -0.45,  0.28]`     | 13.43             |
|  119     | `[16.53, -0.46,  0.28]`     | 13.40             |
| **stored (case 1)** | **`[29.77, -0.50, -1.75]`** | (reference)       |

The stored `c₁` for case 1 is **~1.8× larger** than any Euclidean
projection of the case's FOM snapshots onto the same φ_k. That gap is
exactly what mass-weighted training vs Euclidean-only downstream
reconstruction produces.

Also: mode 3 has `sign(c₃_stored) = −1` but every Euclidean projection
gives `sign(c₃) = +1` — the Euclidean and mass-weighted projections
disagree even on the sign of the smallest mode.

---

## 4. 20-case residual sweep at final timestep (ts = 119)

Reconstructing every case using its own stored coefficients and
comparing against the case's own FOM at ts = 119:

| Case | ‖FOM‖_rms  | mean_rel | centered_rel (colleague) | c/sig_rel | stored c₁ | c₂     | c₃     |
|-----:|-----------:|---------:|-------------------------:|----------:|----------:|-------:|-------:|
|   0  | 7.17e-02   |  62.20 % | **76.02 %**              |  61.96 %  |   27.93   |  1.546 | −0.055 |
|   1  | 7.09e-02   |  64.97 % | **80.13 %**              |  64.70 %  |   29.77   | −0.502 | −1.745 |
|   2  | 1.60e-01   |  29.60 % |    37.84 %               |  29.46 %  |  −33.29   |  2.365 | −0.736 |
|   3  | 1.58e-01   |  28.34 % |    36.31 %               |  28.20 %  |  −30.85   | −2.043 | −0.831 |
|   4  | 1.15e-01   |   3.65 % |     **4.06 %**           |   3.61 %  |   −1.66   | −0.483 |  1.092 |
|   5  | 1.59e-01   |  28.92 % |    36.94 %               |  28.78 %  |  −31.96   |  0.135 | −0.785 |
|   6  | 7.35e-02   |  58.46 % | **71.72 %**              |  58.23 %  |   27.31   |  0.379 | −0.681 |
|   7  | 1.13e-01   |   4.78 % |     **4.83 %**           |   4.63 %  |    0.11   | −2.038 |  0.649 |
|   8  | 1.08e-01   |   7.29 % |     **7.91 %**           |   7.32 %  |    2.21   |  1.047 |  1.540 |
|   9  | 1.37e-01   |  17.53 % |    22.87 %               |  17.45 %  |  −17.44   |  0.566 |  0.557 |
|  10  | 1.37e-01   |  17.03 % |    22.16 %               |  16.93 %  |  −16.51   | −1.435 |  0.345 |
|  11  | 8.85e-02   |  30.81 % |    36.91 %               |  30.70 %  |   16.26   |  0.461 |  0.791 |
|  12  | 9.65e-02   |  19.96 % |    23.48 %               |  19.88 %  |   11.12   | −0.873 |  0.609 |
|  13  | 9.15e-02   |  27.02 % |    32.40 %               |  26.89 %  |   15.02   | −1.625 | −0.124 |
|  14  | 7.16e-02   |  62.53 % | **76.64 %**              |  62.28 %  |   28.31   |  1.078 | −0.391 |
|  15  | 1.60e-01   |  29.40 % |    37.54 %               |  29.26 %  |  −32.84   |  1.506 | −0.769 |
|  16  | 1.37e-01   |  16.94 % |    22.08 %               |  16.83 %  |  −16.22   | −2.338 |  0.221 |
|  17  | 7.61e-02   |  53.23 % | **65.29 %**              |  53.02 %  |   25.83   | −0.463 | −0.999 |
|  18  | 1.42e-01   |  20.39 % |    26.56 %               |  20.30 %  |  −20.97   |  1.653 |  0.397 |
|  19  | 8.58e-02   |  35.00 % |    41.99 %               |  34.88 %  |   17.87   |  1.065 |  0.916 |

**Aggregate**:

| Formula      | mean rel_rms | std   |
|:-------------|-------------:|------:|
| mean-only    |     30.90 %  |  —    |
| **centered** |   **38.18 %**| 23.32 |
| c_over_sig   |     30.76 %  |  —    |

**Best case**: 04 at 4.06 %. **Worst case**: 01 at 80.13 %.

Notes:

- `centered` is consistently 6–15 pp worse than mean-alone. Adding the
  scaled 3-mode correction pushes the reconstruction further from the
  truth than doing nothing.
- `c_over_sig` (dividing c_k by σ_k before contracting) is essentially
  a rounding difference from mean-alone. The 3-mode correction, when
  divided by σ_k, contributes almost nothing.
- The correlation between `|c₁|` and the residual is strong: cases with
  large `|c₁|` (∼28–33) uniformly fail worst. Cases with small `|c₁|`
  (< 3, e.g. 4, 7, 8) reconstruct within single-digit percent.
  Interpretation: the mass-weighting error is proportional to the
  coefficient magnitude.

---

## 5. Time sweep — worst timestep per case

Reconstructing with the (single, case-specific) stored coefficient vector
and comparing against every timestep of the same case:

| Case | best-ts / rel | worst-ts / rel | pattern                    |
|-----:|--------------:|---------------:|:---------------------------|
|   4  | 21 / 3.38 %   | 11 / 5.32 %    | flat, always small         |
|   7  | 16 / 3.45 %   | 116 / 4.83 %   | flat, always small         |
|  15  | 81 / 37.54 %  | 11 / 38.07 %   | flat, always poor          |
|   0  | 11 / 69.67 %  | 116 / 76.01 %  | **grows monotonically**    |
|   1  | 11 / 74.96 %  | 116 / 80.13 %  | **grows monotonically**    |

Interpretation:

- The stored coefficient vector is a **single number per case per mode**,
  not a per-timestep trajectory. It represents an aggregate / late-time
  approximation of the case's FOM.
- Cases that reach steady state before ts = 119 (4, 7, 15) have a flat
  residual profile — the ROM captures the steady state as well as the
  3 modes allow.
- Cases still in transient at ts = 119 (0, 1) show the residual
  **growing** over time. The ROM captures a mean trajectory that lags
  behind the actively evolving physics.

**No single timestep is "worst" across all cases**. The temporal
residual pattern is case-dependent.

---

## 6. Spatial breakdown — where does the residual live?

At the worst point (case 1, ts = 119), splitting nodes by radial
distance from the tool axis (r = √(x² + y²)):

| r bin (m)        | # nodes | ‖resid‖_rms (centered) | ‖FOM‖_rms   | relative |
|:-----------------|--------:|----------------------:|------------:|---------:|
| 0.000 – 0.003    |  44 622 |       5.82e-02        |   1.14e-01  |  50.9 %  |
| 0.003 – 0.007    |  83 879 |       9.99e-02        |   1.27e-01  |  78.6 %  |
| 0.007 – 0.010    |  49 562 |       1.23e-01        |   1.26e-01  |  98.1 %  |
| **0.010 – 0.013**| 1 498   |     **4.40e-02**      |   1.01e-02  | **437 %**|
| 0.013 – 0.017    |    340  |       4.37e-02        |   9.89e-03  | 442 %    |
| 0.017 – 0.020    |    157  |       4.31e-02        |   9.84e-03  | 438 %    |
| 0.020 – 0.023    |    128  |       4.23e-02        |   9.70e-03  | 436 %    |
| 0.023 – 0.027    |    105  |       4.21e-02        |   9.66e-03  | 435 %    |
| 0.027 – 0.030    |    100  |       4.20e-02        |   9.65e-03  | 435 %    |
| 0.030 – 0.034    |     60  |       4.20e-02        |   9.66e-03  | 435 %    |

Vertical (z), same case-1 ts-119:

| z bin (m)             | # nodes | ‖resid‖_rms | ‖FOM‖_rms   | relative |
|:----------------------|--------:|------------:|------------:|---------:|
| bottom, z ≈ -0.004    |  22 081 |   6.60e-02  |   8.57e-02  |  ~76 %   |
| mid, z ≈ -0.002       |  32 629 |   8.15e-02  |   9.65e-02  |  ~85 %   |
| upper mid, z ≈ -0.001 |  24 223 |   7.13e-02  |   1.06e-01  |  ~71 %   |
| **top, z ≈ 0**        |  78 841 |   1.14e-01  |   1.51e-01  |  ~76 %   |

Interpretation:

- **Under the pin (r ≤ 0.010)** the reconstruction absolute-error scales
  with the FOM velocity — relative error is 50–98 %. This is the
  region the ROM is fitting.
- **Outside the pin radius (r > 0.010)** the FOM has essentially zero
  velocity (‖FOM‖_rms ≈ 0.01, one order below the pin region). But
  the `centered` reconstruction stubbornly predicts a non-trivial
  residual (‖resid‖_rms ≈ 0.04) because the stored basis vectors have
  support in that outer region too — leftover from the mass-weighted
  training. This produces **relative errors of 435–440 %** in a region
  that should read as "no motion".
- **`c_over_sig` collapses the outer-domain error** from ~440 % back to
  ~11–23 % — it correctly recognises that mode-1's contribution to the
  outer domain should be small.
- Vertically, the top surface (z ≈ 0, the FSW plate-surface interface)
  is where absolute error is largest.

The **best-case spatial pattern (case 4)** shows the same qualitative
shape: inside r ≤ 0.010 the reconstruction is 2–5 % off, outside the
pin the relative error jumps to 40–54 %. Just the amplitude of the
error is much smaller than case 1.

Two VTU files written for interactive inspection (open in ParaView):

- `case01_ts119_residual.vtu` — worst case; fields: `fom`, `mean`,
  `centered`, `c_over_sig`, `resid_centered`, `resid_cos`,
  `resid_centered_mag`, `resid_cos_mag`, `fom_mag`
- `case04_ts119_residual.vtu` — best case; same fields

(Currently under
`/tmp/claude-1000/…/scratchpad/vtu_out/`. Move them wherever convenient.)

---

## 7. Root cause summary

| Piece                              | Status |
|:-----------------------------------|:-------|
| Colleague's spec                   | correctly describes `centered` formula |
| Our `jaxtrace/rom/velocity_recon.py` `centered` path | **matches FEMUSS exactly** |
| FEMUSS's own reconstruction (`ExternalFilter`)       | plain unweighted `Σ_k c_k φ_k + mean` |
| FEMUSS's own projection (`ExternalProj`)             | plain `<u, φ_k>_Euclidean` — **not** the correct mass-weighted projection |
| Basis stored on disk               | `φ_k = M⁻¹ u_k` (mass-weighted) — assumes downstream reader can apply `M` |
| Number of stored modes             | **3** out of 20 — hard-floor residual ≥ ‖dropped_modes‖ |
| Mass matrix `M`                    | **not exported** — downstream reconstructor cannot fix the metric mismatch |
| SnapshotsMean                      | correctly stored, correctly added at end |

**Two independent limitations produce the observed error**:

1. **Truncation**: only 3 of 20 modes stored. σ₁ = 101, σ₂ = 6.1,
   σ₃ = 3.7, σ₄..σ₂₀ = 1.3 down to 4.6e-14. Even a perfect
   reconstruction from these 3 modes has a hard-floor residual bounded
   below by ‖dropped_modes‖. Sanity check on case 1 ts=119: even the
   optimal Euclidean projection onto the 3 stored modes leaves 63 % of
   the FOM norm — so the truncation floor is not the whole story.
2. **Metric mismatch**: stored modes are mass-weighted (`M⁻¹ u_k`),
   coefficients were computed against them with the Euclidean inner
   product (which is what our `centered` code and FEMUSS's
   `ExternalFilter` both do). This makes the reconstruction identity
   `Σ_k c_k φ_k ≈ centred snapshot` non-orthogonal — it fits the
   training snapshots by construction but propagates poorly.

The metric mismatch is the dominant contributor for the badly-fitting
cases: they are precisely the cases where the true FOM has significant
motion under the pin (large `|c₁|` needed) and the mass-weighted vs
Euclidean discrepancy in mode-1 shows up strongly.

---

## 8. Recommendations

Pick one or several:

### A. Ask the ROM producer to re-export

The cleanest fix. Ask your colleague to either:

- **Turn `LUMPE OFF` at ROM training** and re-run the offline stage.
  This gives back Euclidean-orthonormal basis vectors, and both
  FEMUSS and our loader can reconstruct new snapshots correctly.
- **Or**, keep `LUMPE ON` but also export the lumped mass vector `M_ii`
  (one float per node component) in the `.basis` file. Then downstream
  code can compute `c_k = Σ_i M_ii · u_i · φ_k,i` (mass-weighted
  projection) at load time.
- **Or**, keep the current basis but export more modes. Re-training
  with `basis_energy_target ≥ 0.95` should store 10–15 modes and
  reduce the truncation floor to a few percent.

### B. Fall back to `c_over_sig` inside our loader

Empirically `c_over_sig` never does worse than mean-alone, and often
gives a small improvement. Currently `--rom-formula centered` is the
default in `run_tracking.py`; switching the default to `c_over_sig`
would be a one-line change with negligible risk. Not a fix — just a
better idle-state.

### C. Re-project against a reference FOM at load time

If a single reference FOM snapshot is available for each case (e.g. the
last timestep), we can compute Euclidean projection coefficients
ourselves:

```
c_k = <FOM_ref - mean, φ_k>_Euclidean / <φ_k, φ_k>_Euclidean
```

(or solve the full 3×3 Gram-matrix system to handle non-orthogonality),
then reconstruct with `mean + Σ_k c_k φ_k`. This gives the *best
possible* linear-in-modes reconstruction of the reference snapshot from
the 3 stored modes. It's ad-hoc — the tracking would use a static
velocity field, not one that follows the ROM's own coefficient — but if
we just want a plausible steady-state field to run tracking against, it
is by far the highest-fidelity option available from the current basis
file.

### D. Bypass the ROM entirely

For particle tracking in Section 7, the ROM-derived field is only
useful if it faithfully represents the tracked physics. Given the 4–80 %
residual spread across cases, an alternative is to use the FOM directly
(the PVTU sequences are already on disk) via
`--velocity-source mesh`. This is what the workstation Sec.6 benchmark
already does and it needs no coefficient rescaling.

---

## Reproducibility

Scripts used (currently in the session scratchpad):

- `rom_sweep.py` — 20-case residual table at ts = 119
- `rom_time_sweep.py` — per-timestep residual for cases 0, 1, 4, 7, 15
- `rom_spatial_residual.py` — radial + vertical breakdown, writes VTU
- Existing `tests/rom/compare_rom_recon.py` — original single-case
  comparison harness

To rerun everything from scratch:

```bash
python3 rom_sweep.py              # ~2 min, prints table
python3 rom_time_sweep.py         # ~5 min, prints best/worst ts per case
python3 rom_spatial_residual.py \
    --case 1 --ts 119 \
    --out case01_ts119_residual.vtu
python3 rom_spatial_residual.py \
    --case 4 --ts 119 \
    --out case04_ts119_residual.vtu
```

All three scripts read only `/scratch/shared/ROM/FOM/` and produce
plain text + a VTU. No GPU or JAX required.
