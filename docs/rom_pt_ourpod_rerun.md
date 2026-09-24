# ROM particle tracking, rerun on a corrected POD basis

**Date:** 2026-09-09
**Status:** 5 of 20 cases done (000, 001, 002, 003, 004). Cases 001/003/004
plus 000 have a shipped-basis counterpart, so the head-to-head below covers
four cases; case 002 is new-only.
**Supersedes the ROM numbers in:** [`rom_pt_step2_step3_report.md`](rom_pt_step2_step3_report.md),
[`rom_pt_smoothness_divfree_report.md`](rom_pt_smoothness_divfree_report.md)
(the latter has not been rerun — see *What is still open*).

## Why the old ROM tracking numbers were wrong

Every previous ROM particle-tracking result was measured on a velocity
field reconstructed from the **shipped FEMUSS basis**
(`cylindrical.som.fswrom.basis`). That file reconstructs the ts=119
cohort at **4.04 % L2**. Rebuilding POD directly from the same 20
snapshots reaches **0.52 % with the same 3 modes**.

The cause is not truncation, precision, or Fortran-vs-Python — all of
which were checked and ruled out:

| candidate cause | verdict |
|---|---|
| too few modes | **no** — K=3 suffices when the basis is built from this data |
| float precision | **no** — both sides float64 |
| Fortran vs Python | **no** — formula verified identical to `som_fswrom_ComputeROMSolutioninFOMspace` |
| L1 vs L2 norm convention | **no** — 3.77 % vs 4.04 %, cannot explain 8x |
| **snapshot-set mismatch** | **yes** — shipped sigmas 101.30/6.09/3.68 vs 111.52/5.65/4.06 recomputed here |

Even an *optimal* least-squares projection onto the shipped modes still
gives 3.16 %, so the shipped mode shapes do not span this data. The
shipped basis also fits no timestep in 60..119 better than ~4.7 %, so it
is not a timestep-selection issue either.

## Headline: how much better is the tracking?

`rms |x_ROM − x_FOM|` over all 360 k particles, reference
`post_pt/fom_hct_on/`, identical seeds, identical integrator settings.

| step | shipped basis | our POD K=3 | ratio |
|---|---|---|---|
| 500 (t≈1.9 s) | 2.96 mm | **1.46 mm** | **×0.49** |
| 950 (t≈3.6 s) | 6.79 mm | **2.32 mm** | **×0.34** |
| 2000 (t≈7.5 s) | 19.79 mm | **4.22 mm** | **×0.21** |

Mean over cases 000/001/003/004. **The improvement grows with
integration time** — 2x at mid-early, 3x at the report's main step, and
nearly 5x by the final step. That is the signature of a per-step velocity
error that compounds along the trajectory: a cleaner field does not just
shift the error down, it slows its accumulation.

### Per case

| case | step 500 old → new | step 950 old → new | step 2000 old → new |
|---|---|---|---|
| 000 | 1.46 → 1.74 mm (×1.19) | 5.02 → 3.10 mm (×0.62) | 19.25 → 4.59 mm (×0.24) |
| 001 | 3.15 → 1.54 mm (×0.49) | 6.25 → 2.41 mm (×0.39) | 18.00 → 4.52 mm (×0.25) |
| 003 | 4.22 → 1.96 mm (×0.47) | 9.65 → 2.56 mm (×0.26) | 24.08 → 5.39 mm (×0.22) |
| 004 | 3.02 → 0.61 mm (×0.20) | 6.23 → 1.21 mm (×0.19) | 17.82 → 2.39 mm (×0.13) |

Case 000 is the one exception — very slightly *worse* at step 500
(1.46 → 1.74 mm) before improving substantially later. It is also the
worst-reconstructing case under our POD at K=3 (1.22 % vs a 0.52 %
cohort mean), so this is consistent rather than anomalous.

Case 003 — the case the old report singled out as "field-accurate but
flow-sensitive" — improves most at step 950 (×0.26).

### In the deck's units

D<sub>pin</sub> = 10 mm (pin diameter). At step 950 the cohort mean falls
from **0.68 D<sub>pin</sub>** to **0.23 D<sub>pin</sub>**.

For scale, grid PT (`4lvl_hct`) sits at 5.62 mm = 0.56 D<sub>pin</sub> at
the same step over the full 20-case cohort. **On these five cases the ROM
path on a correct basis is now more accurate than the grid path** —
which inverts the picture that led to closing the ROM track. See the
caveat below before acting on that.

## What this does to the §2/§3 conclusions

Recomputed in the old report's own metric (`rms / FOM bbox diagonal`),
step 950. The "old" columns reproduce the published values exactly, which
validates the comparison.

| case | old Lagr. | new Lagr. | old Eulerian | new Eulerian | old amplif. | new amplif. |
|---|---|---|---|---|---|---|
| 000 | 12.17 % | 7.53 % | 6.40 % | 1.22 % | 1.90× | **6.18×** |
| 001 | 11.69 % | 4.51 % | 3.75 % | 1.02 % | 3.12× | **4.42×** |
| 003 | 17.67 % | 4.68 % | 2.62 % | 0.50 % | 6.74× | **9.33×** |
| 004 | 13.16 % | 2.55 % | 3.61 % | 0.39 % | 3.64× | **6.62×** |

**The headline §2/§3 finding survives — and is strengthened.** The report
argued that Eulerian accuracy does not predict Lagrangian accuracy. With
a 3–9× better Eulerian field, Lagrangian error fell only ~3×, so the
**amplification factors rose in every case**. A small Eulerian residual
is even less sufficient as an acceptance criterion than the original
report claimed.

What *is* superseded is the specific claim that the ROM–FOM Lagrangian
gap is large in absolute terms. At 2.32 mm mean it is now comparable to
grid-discretisation error, not dominant over it.

## What is still open

1. **15 of 20 cases not yet run.** Everything above is 4–5 cases. Case
   003 improved most and it drove the original narrative, so the cohort
   picture could shift. Run:
   `bash /scratch/shared/ROM/FOM/run_ourpod_rom_pt_all20.sh` (skips the
   five already done).
2. **The §4 under-mixing analysis has not been rerun.** Trapped-particle
   counts, residence time and pairwise separation were all measured on
   the defective field. The under-mixing finding was the *physical*
   reason for closing the ROM track, so it is the one that most needs
   rechecking — a better-resolved field may retain more of the near-pin
   shear structure that does the trapping.
3. **In-sample vs out-of-sample.** Our POD is built from all 20
   snapshots, so reconstructing case *c* uses a basis that saw case *c*.
   Leave-one-out gives 0.66 % at K=3 (vs 0.52 % in-sample), so the
   penalty is small — but a deployable surrogate also needs a
   parameter→coefficient regression, which adds error this comparison
   does not include.

## Reproducing

```bash
# build/export the basis (already done; writes .npz + a report)
python3 scripts/build_our_pod_basis.py \
    --rom-root /scratch/shared/ROM/FOM \
    --out      /scratch/shared/ROM/rom_out/our_pod

# reconstruct + generate runners + track
bash /scratch/shared/ROM/FOM/run_ourpod_rom_pt_all20.sh

# compare (head-to-head against the shipped-basis runs)
python3 scripts/compare_rom_pt_sweep.py \
    --rom-root /scratch/shared/ROM \
    --variant ourpod3 --baseline centered --cases 0 1 2 3 4
```

**Two traps when reproducing:**

- **Reference choice dominates.** Two FOM references exist per case and
  they differ from each other by 9–11 mm at step 950 — comparable to the
  signal. Use `post_pt/fom_hct_on/`, which matches the ROM runs'
  `GRADIENT_RECOVERY=1`. The generic
  `post_pt/run_grid-frac_n360000_s2000/` gives 8.99 mm instead of
  2.32 mm for the same data.
- **Radial binning.** Particles are seeded in an annulus at r ≥ 13.2 mm,
  so a near-pin bin defined on the *initial* radius is always empty. Bin
  on the radius at the reporting step.

## Data

| artefact | path |
|---|---|
| our POD basis | `/scratch/shared/ROM/rom_out/our_pod/our_pod_ts119.npz` |
| basis report | `/scratch/shared/ROM/rom_out/our_pod/our_pod_ts119.md` |
| reconstructed fields | `/scratch/shared/ROM/ROM_recon_ourpod3/<case>.gid/post/cylindrical_0.pvtu` |
| ROM PT runs | `/scratch/shared/ROM/ROM_recon_ourpod3/<case>.gid/post_pt/rom_ourpod3/run_*/particles.vtkhdf` |
| shipped-basis PT runs | `/scratch/shared/ROM/ROM_recon_centered/<case>.gid/post_pt/rom_centered_hct_on/` |
| logs | `/scratch/shared/ROM/FOM/ourpod_pt_logs/ourpod3/` |
