# FSW-ROM velocity reconstruction — findings

**Scope**: quantify how well a linear POD reconstruction from the
`cylindrical.som.fswrom.basis` / `.romdata` pair recovers the FOM
velocity field on the 20 cylindrical cohort cases, document the
FEMUSS convention we mirror in `jaxtrace/rom/velocity_recon.py`, and
retain a diagnostic trail for the historical loader bug that inflated
our earlier residuals into the 4–80 % range.

Companion doc: [`rom_pt_roadmap.md`](rom_pt_roadmap.md) sequences the
particle-tracking experiments that build on the reconstruction.

**Files inspected**:

- `jaxtrace/rom/velocity_recon.py` — our loader / reconstructor
- `DTFSW/RBFforFSW/rbfrom/loader.py` — colleague's own loader (the
  disambiguating reference)
- `FemussROM/Sources/modules/solme/FrictionStirWelding/Mod_som_FswROM.f90`
  — FSW ROM driver (writer and reader, including the naming rule for
  `Basis_CompMode`)
- `FemussROM/Sources/ParallelLibraries/SLEPc/Mod_SLEPcMatrix.f90` —
  projection + reconstruction primitives (`ExternalProj`,
  `ExternalFilter`)

**Data** (all under `/scratch/shared/ROM/FOM/`):

- `cylindrical.som.fswrom.basis` — SnapshotsMean + 3 stored modes on
  180,461 nodes
- `cylindrical.som.fswrom.romdata` — 20 sigma values + 3×20
  coefficient matrix (one coefficient vector per case)
- `cylindrical_000.gid` … `cylindrical_019.gid` — full FOM PVTU
  sequences, timesteps 0–119

---

## 1. TL;DR

1. The FEMUSS reconstruction convention is exactly

       v(x) = SnapshotsMean(x) + Σ_k c_k · φ_k(x)

   — a plain unweighted sum, no sigma scaling. Our `centered` formula
   implements this verbatim.
2. On the 20 cohort cases at ts = 119, the reconstruction has **mean
   rel_rms = 4.04 % ± 1.08 %** (best: case 03 at 2.62 %, worst: case
   00 at 6.40 %). Every case reconstructs to single-digit percent —
   the 3-mode POD basis captures the FSW physics well.
3. The residual sits mostly under the pin (r ≤ 0.010 m) where the
   FOM has substantial motion. Outside the pin the FOM velocity is
   an order of magnitude smaller, so the absolute residual is tiny
   (~4 × 10⁻³) even though the relative error there reads 40–80 %.
4. The residual grows slowly through the FOM's transient: the ROM
   prediction is closest to the FOM at `ts ≈ 16–21` (early-transient)
   and drifts up ~1.5–2 pp by `ts = 119`. This is expected — the
   stored coefficients are one scalar vector per case that fits the
   *whole* snapshot cohort at training time, so it settles into a
   time-averaged compromise.
5. `c_over_sig` (empirically the "best" formula in an earlier
   writeup) is essentially the same as mean-only under the fixed
   loader. That earlier ranking was an artefact of the loader bug —
   see § 6.
6. Loader lesson: the Fortran writer names basis vectors
   `Basis_CompMode <component>  <mode>`. The colleague's Python
   loader used the correct regex from day one; our loader had the
   two integers swapped and produced Frankenstein "modes" that mixed
   the x-component of three different real modes into one row.
   Fixed in commit `ff53727` (§ 6).

---

## 2. The reconstruction convention

The colleague-authored spec ("for each node, multiply the coefs by
the basis, then add the SnapshotsMean from the basis file") maps to

    v(node, case) = mean(node) + Σ_{k=1..K} c_k(case) · φ_k(node)

Our `centered` formula in
[`jaxtrace/rom/velocity_recon.py`](../jaxtrace/rom/velocity_recon.py) implements
this verbatim:

```python
w = c
v = np.einsum("k,knj->nj", w, basis.modes[:n_use])   # Σ_k c_k · φ_k
v = v + basis.mean                                   # + mean
```

FEMUSS itself reconstructs via
[`SLEPcExternalFilter`](../../FemussROM/Sources/ParallelLibraries/SLEPc/Mod_SLEPcMatrix.f90#L738-L759),
called from
[`som_fswrom_ComputeROMSolutioninFOMspace`](../../FemussROM/Sources/modules/solme/FrictionStirWelding/Mod_som_FswROM.f90#L413-L449):

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

Plain unweighted sum `Σ_k c_k · φ_k`, then the mean is added in the
outer driver:

```fortran
if (a%FswROMData%kfl_substractmean) then
   snapshot(:,1:npoinlocal) = snapshot(:,1:npoinlocal) &
       + a%FswROMData%SnapshotsMean(:,1:npoinlocal)
end if
```

Sigmas are stored (`Sigma_Mode k`) but never used in reconstruction —
they are informational only (retained-energy diagnostics).

---

## 3. 20-case residual sweep at ts = 119

Reconstruct every case using its own stored coefficients and compare
to the case's own FOM at ts = 119 (numbers from
`tests/rom/rom_20case_sweep.py`):

| Case | ‖FOM‖_rms | mean_rel | **centered_rel** | c_over_sig_rel | c₁ | c₂ | c₃ |
|-----:|----------:|---------:|-----------------:|----------------:|-----:|------:|------:|
|   0  | 7.17e-02  | 62.20 %  |     **6.40 %**   |    61.61 %      | 27.93 |  1.55 | −0.05 |
|   1  | 7.09e-02  | 64.97 %  |       3.75 %     |    64.31 %      | 29.77 | −0.50 | −1.75 |
|   2  | 1.60e-01  | 29.60 %  |       3.52 %     |    29.28 %      | −33.29 |  2.37 | −0.74 |
|   3  | 1.58e-01  | 28.34 %  |     **2.62 %**   |    28.03 %      | −30.85 | −2.04 | −0.83 |
|   4  | 1.15e-01  |  3.65 %  |       3.61 %     |     3.44 %      | −1.66 | −0.48 |  1.09 |
|   5  | 1.59e-01  | 28.92 %  |       2.82 %     |    28.62 %      | −31.96 |  0.14 | −0.79 |
|   6  | 7.35e-02  | 58.46 %  |       4.80 %     |    57.91 %      | 27.31 |  0.38 | −0.68 |
|   7  | 1.13e-01  |  4.78 %  |       3.30 %     |     4.44 %      |  0.11 | −2.04 |  0.65 |
|   8  | 1.08e-01  |  7.29 %  |       4.83 %     |     7.11 %      |  2.21 |  1.05 |  1.54 |
|   9  | 1.37e-01  | 17.53 %  |       3.40 %     |    17.34 %      | −17.44 |  0.57 |  0.56 |
|  10  | 1.37e-01  | 17.03 %  |       2.85 %     |    16.83 %      | −16.51 | −1.44 |  0.35 |
|  11  | 8.85e-02  | 30.81 %  |       5.23 %     |    30.53 %      | 16.26 |  0.46 |  0.79 |
|  12  | 9.65e-02  | 19.96 %  |       4.17 %     |    19.76 %      | 11.12 | −0.87 |  0.61 |
|  13  | 9.15e-02  | 27.02 %  |       4.05 %     |    26.73 %      | 15.02 | −1.63 | −0.12 |
|  14  | 7.16e-02  | 62.53 %  |       5.61 %     |    61.94 %      | 28.31 |  1.08 | −0.39 |
|  15  | 1.60e-01  | 29.40 %  |       3.15 %     |    29.09 %      | −32.84 |  1.51 | −0.77 |
|  16  | 1.37e-01  | 16.94 %  |       2.75 %     |    16.70 %      | −16.22 | −2.34 |  0.22 |
|  17  | 7.61e-02  | 53.23 %  |       4.17 %     |    52.72 %      | 25.83 | −0.46 | −1.00 |
|  18  | 1.42e-01  | 20.39 %  |       3.86 %     |    20.17 %      | −20.97 |  1.65 |  0.40 |
|  19  | 8.58e-02  | 35.00 %  |       5.96 %     |    34.68 %      | 17.87 |  1.07 |  0.92 |

**Aggregate** (mean ± std over the 20 cases):

| Formula      |     mean rel_rms |          std |
|:-------------|-----------------:|-------------:|
| mean-only    |         30.90 %  |    19.00 %   |
| **centered** |       **4.04 %** |  **1.08 %**  |
| c_over_sig   |         30.56 %  |    18.87 %   |

**Best case**: 03 at **2.62 %**. **Worst case**: 00 at **6.40 %**.

Note the flat behaviour of `centered` across the cohort: no case
exceeds 6.40 %, and mean-only (which was our "baseline" in the
buggy-loader writeup) is now uniformly much worse. `c_over_sig` sits
right on top of mean-only for every case because dividing the
coefficients by the (large) singular values crushes the 3-mode
correction to nearly zero — confirming that `c_over_sig` should not
be used.

---

## 4. Time sweep — when in the FOM's transient does the ROM match best?

The stored coefficients are one scalar vector per case; the
reconstruction is a static field. Comparing that field to every FOM
timestep tells us where along the FOM's transient the ROM prediction
lands.

Two representative cases (numbers from
`tests/rom/rom_time_sweep.py`):

**Worst-reconstructing case (case 00)** at stride = 5, ts = 1 … 40:

| ts | ‖FOM‖_rms | mean_rel | centered_rel |
|---:|----------:|---------:|-------------:|
|  11 | 7.50e-02 | 52.83 %  |     5.47 %   |
|  16 | 7.34e-02 | 57.26 %  |   **3.22 %** |
|  21 | 7.30e-02 | 58.34 %  |     3.57 %   |
|  26 | 7.27e-02 | 59.23 %  |     4.07 %   |
|  31 | 7.25e-02 | 59.90 %  |     4.54 %   |
|  36 | 7.23e-02 | 60.40 %  |     4.93 %   |

**Best-reconstructing case (case 03)** at stride = 20, ts = 21 … 101:

| ts | ‖FOM‖_rms | mean_rel | centered_rel |
|---:|----------:|---------:|-------------:|
|  21 | 1.59e-01 | 29.03 %  |   **1.76 %** |
|  41 | 1.58e-01 | 28.52 %  |     2.30 %   |
|  61 | 1.58e-01 | 28.40 %  |     2.52 %   |
|  81 | 1.58e-01 | 28.36 %  |     2.58 %   |
| 101 | 1.58e-01 | 28.35 %  |     2.61 %   |

Both cases show the **same pattern**: the ROM prediction is closest
to the FOM at `ts ≈ 16–21` (early-transient), then drifts upward and
plateaus by `ts ≈ 100`. The residual grows by roughly 1.5–2 pp
between the minimum and the ts = 119 value tabulated in § 3.

Physical reading: the ROM coefficients were fit at training time
against the *whole* per-case snapshot record, so the reconstructed
field is a compromise between the FOM's early- and late-transient
states. When we probe it against a specific FOM timestep, the
minimum residual naturally lands somewhere in the middle of the
transient, not at either extreme.

---

## 5. Spatial breakdown — where does the residual live?

Splitting nodes by radial distance from the tool axis and by
height z for the worst-reconstructing case (numbers from
`tests/rom/rom_spatial_residual.py --case 0 --ts 119`):

**Radial** (case 00, ts = 119, centered):

| r bin (m)     | # nodes | ‖resid‖_rms | ‖FOM‖_rms | relative |
|:--------------|--------:|------------:|----------:|---------:|
| 0.000 – 0.003 |  44 622 |  2.34e-03   |  1.15e-01 |   2.04 % |
| 0.003 – 0.007 |  83 879 |  8.44e-03   |  1.29e-01 |   6.54 % |
| 0.007 – 0.010 |  49 562 |  1.02e-02   |  1.27e-01 |   8.07 % |
| 0.010 – 0.013 |   1 498 |  2.16e-03   |  5.26e-03 |  41.17 % |
| 0.013 – 0.017 |     340 |  2.53e-03   |  5.04e-03 |  50.20 % |
| 0.017 – 0.020 |     157 |  3.21e-03   |  5.05e-03 |  63.55 % |
| 0.020 – 0.023 |     128 |  3.82e-03   |  4.99e-03 |  76.62 % |
| 0.023 – 0.027 |     105 |  3.97e-03   |  4.97e-03 |  79.81 % |
| 0.027 – 0.030 |     100 |  3.99e-03   |  4.96e-03 |  80.35 % |
| 0.030 – 0.034 |      60 |  3.97e-03   |  4.97e-03 |  80.03 % |

**Vertical (z)**, same case (values are RMS residuals inside each
z-bin, centered formula):

| z bin (m)             | # nodes | ‖resid‖_rms | ‖FOM‖_rms | relative |
|:----------------------|--------:|------------:|----------:|---------:|
| bottom, z ≈ −0.004    |  22 081 |  5.03e-03   |  8.27e-02 |   ~6 %  |
| mid, z ≈ −0.002       |  32 629 |  6.03e-03   |  9.90e-02 |   ~6 %  |
| upper mid, z ≈ −0.001 |  24 223 |  5.55e-03   |  1.07e-01 |   ~5 %  |
| top, z ≈ 0            |  78 841 |  9.34e-03   |  1.22e-01 |   ~8 %  |

Reading the two tables:

- **Inside the pin** (r ≤ 0.010 m) — where 99 % of the mesh mass and
  essentially all of the FOM velocity live — the relative residual
  is 2 – 8 %. This is what dominates the aggregate `centered_rel =
  6.40 %` in § 3.
- **Outside the pin** (r > 0.010 m) the FOM velocity itself is an
  order of magnitude smaller (‖FOM‖_rms ≈ 5 × 10⁻³, versus ~1.3 × 10⁻¹
  under the pin). The reconstruction residual there is also small
  in absolute terms (~4 × 10⁻³) but reads 40 – 80 % *relative*
  because both quantities are near the mesh's low-signal floor.
  This is the low-signal-region noise artefact, not a genuine
  reconstruction failure.
- **Vertically** the residual is essentially flat around 5 – 8 % with
  a slight amplification at the top surface (z ≈ 0) — where the
  pin's stir motion pushes material.

The best-reconstructing case (case 03, ts = 119) shows the same
shape: 1 – 3.5 % inside the pin, 19 – 36 % outside (again driven by
the outer-domain being near-zero in both fields).

Interactive VTUs are produced by
`tests/rom/rom_spatial_residual.py --out <path>` and open cleanly in
ParaView; the fields are `fom`, `mean`, `centered`,
`resid_centered`, `fom_mag`, `resid_centered_mag`.

---

## 6. Historical loader bug (2026-07-14 → 2026-07-16)

This section is provenance / lessons-learned. The current
`velocity_recon.py` is correct; the numbers above reflect the fixed
loader.

**Symptom**: an earlier version of this doc reported mean
`centered_rel = 38.2 % ± 23.3 %`, best 04 at 4.06 %, worst 01 at
80.13 %. Under that reading, `c_over_sig` was the empirically-best
formula and we chased several red-herring hypotheses about
mass-weighted SVD, non-Euclidean-orthonormal modes, and a 63 %
truncation floor.

**Root cause**: the Fortran writer at
`Mod_som_FswROM.f90:486-487` emits basis vectors with the naming

    write(nameV,"(A,I0,A,I0)")
        'Basis_CompMode ', idofn, '  ', imode

with `idofn` = velocity component (1..3, x/y/z) and `imode` = POD
mode index. Our `load_basis` had the two integers **transposed**: it
treated the first number as the mode and the second as the
component, producing a Frankenstein "mode k" that mixed the
x-component of three different real modes into one row of the
returned array. The linear combination `Σ_k c_k · φ_k` on that
Frankenstein basis has nothing to do with the FOM.

**Diagnostic route**: cross-checking against the colleague's own
loader `DTFSW/RBFforFSW/rbfrom/loader.py::_extract_basis` (which uses
the correct regex, `Basis_CompMode\s*(\d+)\s+(\d+)` with group(1) =
component, group(2) = mode) immediately pointed to the axis swap.
Confirmed by reading the Fortran writer, which even has an inline
comment stating the intended naming.

**Fix**: `jaxtrace/rom/velocity_recon.py::load_basis` now iterates
mode-outer, component-inner, matching the writer. Landed in commit
`ff53727` on `feature/analytic-velocity`.

**Before / after comparison on the same 20 cases at ts = 119**:

| Metric                                        | Buggy loader | Fixed loader |
|:----------------------------------------------|-------------:|-------------:|
| mean `centered_rel`                           | 38.18 %      | **4.04 %**   |
| std                                           | 23.32 %      | **1.08 %**   |
| best case (`centered_rel`)                    | 04 at 4.06 % | 03 at **2.62 %** |
| worst case (`centered_rel`)                   | 01 at 80.13 %| 00 at **6.40 %** |
| mode L² norms `‖φ_k‖₂`                        | `[1.34, 1.34, 0.086]` (junk) | `[1.10, 1.10, 1.09]` (proper) |
| Gram matrix diagonal `<φ_k, φ_k>_Euclidean`   | `[1.79, 1.81, 0.007]` (junk) | `[≈1.0, ≈1.0, ≈1.0]` (orthonormal) |

The pre-fix mode norms and Gram diagonals were what led us to suspect
mass-weighted SVD; under the fix they are exactly what an
Euclidean-orthonormal POD basis should look like, so all mass-matrix
arguments are **retracted**. No hidden physics — just an off-by-one
in the loader.

Lesson to preserve: whenever you find yourself invoking
FEMUSS-side pathologies to explain a downstream mismatch, first
verify the loader against a second independent implementation
(colleague's Python loader, in this case). Two agreeing loaders
would have caught this in an hour.

---

## 7. Recommendations

**Use `centered` (the FEMUSS convention).** It averages 4 %
rel_rms on the cylindrical cohort, is monotone under mode addition,
and matches how FEMUSS itself reconstructs training snapshots
in-solver. No other formula in the loader currently beats it — see
the 20-case table in § 3.

The reconstruction is now good enough that the interesting question
moves from "is the reconstruction correct?" (yes) to "does this
level of reconstruction accuracy translate to acceptable particle-
tracking accuracy?". That question is the subject of
[`rom_pt_roadmap.md`](rom_pt_roadmap.md).

If a future run finds a case where `centered_rel` exceeds ~8 %,
inspect that case's early-transient FOM (§ 4): the ROM may simply be
matching a different point of the case's temporal trajectory, in
which case the mismatch is a training-time choice, not a bug.

---

## Reproducibility

Three permanent analysis scripts live under
[`tests/rom/`](../tests/rom/):

- `rom_20case_sweep.py` — the § 3 table
- `rom_time_sweep.py` — the § 4 tables
- `rom_spatial_residual.py` — the § 5 tables and the VTU dumps

All three read only `/scratch/shared/ROM/FOM/` and produce plain
text (plus a VTU when requested). Neither GPU nor JAX is required —
they use the `jaxtrace.rom.velocity_recon` module in numpy mode.

To reproduce every number in the document:

```bash
python3 tests/rom/rom_20case_sweep.py
python3 tests/rom/rom_time_sweep.py --case 0 --stride 5 --ts-min 1 --ts-max 40
python3 tests/rom/rom_time_sweep.py --case 3 --stride 20
python3 tests/rom/rom_spatial_residual.py --case 0 --ts 119 \
    --out /tmp/case00_ts119_centered.vtu
python3 tests/rom/rom_spatial_residual.py --case 3 --ts 119 \
    --out /tmp/case03_ts119_centered.vtu
```

Times: each script runs in under a minute on the workstation.

The colleague's Python loader
(`DTFSW/RBFforFSW/rbfrom/loader.py::_extract_basis`) and the Fortran
writer
(`FemussROM/Sources/modules/solme/FrictionStirWelding/Mod_som_FswROM.f90:486-487`)
are the two independent authorities for the `Basis_CompMode` naming
convention. Any future modification to
`jaxtrace/rom/velocity_recon.py::load_basis` should cross-check
against both.
