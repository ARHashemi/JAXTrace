# ROM PT — smoothness and divergence-free analysis (roadmap §4)

> **⚠ Errata (2026-09-09) — not yet rerun.** Every ROM field in this
> report is the **shipped FEMUSS basis** reconstruction (4.04 % L2). Our
> own POD reaches 0.52 % at K=3, and rerunning the tracking sweep on it
> cut ROM-FOM displacement by x0.34 at step 950. The under-mixing
> finding here — that the ROM traps fewer particles because truncation
> damped the near-pin shear structure — was **the physical argument for
> closing the ROM track**, and it has *not* been re-checked against the
> corrected field. A better-resolved field may retain more of that
> structure. Treat every conclusion below as open until rerun.
> See [`rom_pt_ourpod_rerun.md`](rom_pt_ourpod_rerun.md).


**Date:** 2026-07-17 (revised evening of same day)
**Scope:** 4-case cohort (000, 001, 003, 004), Eulerian velocity
comparison at ts = 119 (source timestep of the ROM reconstruction),
Lagrangian mixing diagnostics from the overnight PT sweep.
**Roadmap section:** §4 · "Why do ROM trajectories look neater?"

## Revision note (2026-07-17 evening)

An earlier version of this report used nearest-node sampling for the
azimuthal ring diagnostics.  On the FOM mesh at r ≈ 10 – 15 mm the
mesh coverage is coarse enough that 360 queries were collapsing to
~50 unique nodes, producing severe aliasing that we mistook for
"low-k coherent structure".  Symptom: Fig B looked like a single
sinusoid at r = 10 mm; Fig C had FFT spikes that didn't match Fig B's
shape.  In addition, the earlier ring radii (r = 5, 10, 15 mm) sat
inside the pin (r = 5) or right on the pin surface (r = 10), so the
"noisy" appearance at r = 5 was actually the smoothly-varying
pin-rotation velocity while r = 10 was hitting the level-set
boundary.

This revision:

- **replaces nearest-node with `vtkProbeFilter` interpolation** — the
  probe uses the same P1 barycentric interpolation the RK4 tracker
  uses, so the samples represent what the tracker actually sees;
- **moves the ring radii to r = 6, 9, 12, 15 mm** — deep pin,
  shear-layer, just-outside-pin, bulk workpiece;
- **splits Fig F into a linear/log twin panel** to expose the tail
  where FOM traps 10 – 1000× more particles than ROM.

Findings section below is entirely rewritten to reflect what the
corrected figures actually show.

## What this report answers

Section 4 of the roadmap asked whether the ROM's low-rank truncation
damps the small-scale velocity structure that FSW mixing relies on.
The §2/§3 report established that the ROM–FOM Lagrangian gap
grows with integration time and concentrates near the pin, but did
not check *why* directly.  This report closes that loop with
Eulerian diagnostics on the velocity fields themselves:

1. **Radial and azimuthal |v| profiles** — is the ROM velocity a
   smoothed version of the FOM velocity or does it match closely?
2. **Azimuthal FFT spectra** — where in the wavenumber spectrum
   does the ROM residual live?  Roadmap §4.3 predicts high-k
   dominance if the smoothing hypothesis is correct.
3. **Point-wise divergence sampling** — is `∇·v` actually close to
   zero for either field?  Is the ROM more or less divergence-free
   than the FOM?
4. **Divergence vs Lagrangian PT gap** — does Eulerian
   `|∇·v|_rms` predict Lagrangian PT rel_rms across cases better
   than the Eulerian velocity rel_rms does?
5. **Mixing diagnostics** — residence-time and pairwise-separation
   curves per case × HCT setting, from the overnight sweep's mixing
   directories.

The findings partly reinforce and partly overturn hypotheses from
§§2–3.  Read the headline findings first before jumping into the
figures.

## Data sources

| artefact | path |
|---|---|
| radial line samples of |v| (4 cases) | `rom_out/velocity_diagnostics/radial_samples_cylindrical_<case>.csv` |
| azimuthal ring samples of |v| (4 cases) | `rom_out/velocity_diagnostics/ring_samples_cylindrical_<case>.csv` |
| per-tet |∇·v| summary (4 cases) | `rom_out/velocity_diagnostics/divergence_summary_cylindrical_<case>.csv` |
| residual VTU + divergence VTU (ParaView) | `rom_out/velocity_diagnostics/{residual,divergence}_cylindrical_<case>.vtu` |
| mixing residence time (case × HCT) | `ROM_recon_centered/cylindrical_<case>.gid/post_pt_mixing/hct_{on,off}/residence_time.csv` |
| mixing pairwise separation (case × HCT) | same directory, `pairwise_separation.csv` |

Figures live under `paper_figs/rom_pt_step4/` in PNG + SVG at 300 dpi.
Regenerate with:

- `scripts/velocity_field_diagnostics.py` (produces the CSVs + VTUs)
- `scripts/build_smoothness_divfree_figures.py` (renders the figures)

## Methodology

### Eulerian sampling grid

The FOM and ROM PVTUs share the exact same mesh (the ROM
reconstruction copies the FOM mesh verbatim and overwrites the
velocity field), verified by lexsort node-position match: max
node-position difference after alignment = 0.  So `v_FOM − v_ROM`
is a valid per-node subtraction with no interpolation.

Radial line samples: 200 nearest-node queries per line at 0°, 90°,
180°, 270° from r = 0.1 mm to r = 20 mm, restricted to the z ∈
[−1, +1] mm mid-plane.  This gives 800 samples per case in
`radial_samples_cylindrical_<case>.csv`.

Ring samples: 360 azimuthal samples at each of r = 5, 10, 15 mm,
same z-slab, 1080 samples per case.

### P1 per-element divergence

For each tet, using its 4 vertex velocities and the standard
barycentric shape functions, ∇·v is computed as a single scalar
via `div v = trace(M⁻¹ · [v_1 − v_0; v_2 − v_0; v_3 − v_0])` where
M is the edge matrix from vertex 0.  Vectorised over all 736,697
cells with numpy.  This is the exact P1 divergence at every cell,
not an interpolated field.

### Trapping metric context

The §2/§3 report defined a "trapped particle" as one whose final
radial position is still ≤ 10 mm at step 950.  The mixing
directories (`residence_time.csv`) give a finer time-resolved view
of the same process: the count of particles inside the annular
probe at every 20 steps, for both FOM and ROM tracker independently.

---

## Headline findings

1. **Inside the pin body (r < 8 mm) the ROM matches the FOM
   accurately at all azimuthal scales.**  Fig B r = 6 mm panels show
   |v| = 0.28 – 0.56 m/s (the pin rotation velocity), FOM and ROM
   overlap within 1 – 2 %.  Fig C at r = 6 mm shows the residual
   spectrum sits ~1 decade below FOM/ROM across the entire k range.
   The ROM is not a low-pass smoother of the pin rotation.

2. **In the workpiece (r > 10 mm) the ROM *systematically
   overestimates* |v| by 20 – 50 %.**  Fig B r = 12 and r = 15 mm
   panels show ROM 30 – 40 % higher than FOM at every θ, in every
   case.  This is not smoothing (which would preserve amplitude and
   damp fine structure), it is a **coherent bias** — the ROM's mean
   workpiece velocity is wrong, not its structure.

3. **The residual is dominated by low-k (large-scale) content, not
   high-k.**  Fig C at r = 12 mm shows the residual sitting *at* the
   FOM/ROM plateau at k = 1 – 4 (the workpiece-overestimation bias
   from finding 2) then 1 – 2 decades *below* FOM/ROM at k > 5.  So
   the ROM captures the small azimuthal features and misses the
   large-scale mean.  The direction of the residual signature is
   **opposite** to the roadmap §4 smoothing hypothesis.

4. **The FOM velocity itself is far from divergence-free.**  Near-pin
   FOM |∇·v|_rms is 18 – 34 s⁻¹ across cases (Fig D).  With FOM |v|
   peaks ~ 0.5 – 0.9 m/s and cell edge ~ 10⁻⁴ m, natural gradient
   scale is ~10⁴ s⁻¹, so |∇·v|_rms ~30 is ~0.3 % of that.  Not
   negligible.

5. **ROM inherits FOM's divergence structure at 99 % cell-by-cell
   correlation.**  Fig D overlays FOM and ROM CCDFs
   indistinguishably.  The ROM does not introduce new divergence
   artefacts and does not clean up the FOM's discretisation-induced
   divergence.

6. **FOM |∇·v|_rms predicts the Lagrangian PT gap.**  Fig E scatter:
   case 003 has both the highest FOM near-pin |∇·v|_rms and the
   worst ROM-vs-FOM PT rel_rms.  This is a **stronger** case-level
   predictor than Eulerian velocity rel_rms — which in the §2/§3
   report Fig 2 showed case 003 *lowest* on Eulerian error but
   *highest* on PT gap.

7. **FOM traps 2 – 100× more particles than ROM at final step.**
   Fig F log panels: case 000 FOM leaves 14,606 in the annulus at
   step 2000, ROM only 1,016.  Case 001: FOM 774, ROM 7.  Case 003:
   FOM 2,714, ROM 1,260.  Case 004: FOM 4,563, ROM 1,738.  In every
   case the ROM tracker under-traps.  Case 003 is the *least*
   discrepant (2× ratio), likely because its coherent shear-layer
   residual in Fig 5 keeps ROM particles orbiting near the tool for
   longer.

8. **Case 001 has the largest ROM/FOM trapping asymmetry (110×) but
   the smallest bulk velocity residual.**  This suggests trapping
   is not driven by the mean velocity bias of finding 2 alone —
   there is a shear-layer detail specific to each case that
   determines whether particles enter the recirculation zone.

9. **Pairwise-separation curves do not show a universal
   ROM < FOM pattern.**  Fig G: case 001 ROM stays below FOM
   throughout; cases 003 / 004 HCT-off show ROM ending *above* FOM;
   case 000 is mixed.  The chaotic-advection rate is case-dependent
   and cannot be summarised as "ROM smooths trajectories".

10. **HCT has essentially no effect on Eulerian smoothness, on
    divergence, on residence-time, or on separation.**  HCT-on/off
    rows in Figs F–G overlay closely; the Eulerian velocities
    themselves are unaffected by HCT (which is a runtime velocity
    reconstruction inside the tracker, not a change to the stored
    field).  Consistent with the §2/§3 report's Fig 4 finding.

## Revised roadmap §4 interpretation

The roadmap §4 posed two hypotheses:

> **The ROM basis is a smooth low-pass**: the 3 POD modes capture
> large-scale flow structure but miss high-frequency features
> present in the FOM velocity.
>
> **Coincidental**: the "neater" impression was tied to the buggy
> loader's under-shot reconstruction.

The corrected Eulerian diagnostics show the smoothing hypothesis is
**rejected in the direction the roadmap posed it**.  The ROM does
not miss high-k structure; it misses the *low-k mean* in the
workpiece region.  Concretely, three separate signals converge:

- **Fig B r = 12 mm**: ROM overestimates workpiece |v| by 30 – 40 %.
- **Fig C r = 12 mm**: residual sits at the FOM/ROM level at k = 1
  – 4 (bias) and 1 – 2 decades below at k > 5 (structure captured).
- **Fig F**: ROM particles exit the annular probe far faster than
  FOM particles.  Rate is consistent with 30 – 50 % faster mean
  transport, matching the workpiece velocity overestimation.

So the correct §4 diagnosis is: **the ROM overestimates the mean
workpiece velocity**, which produces faster particle transport
through the pin region and less trapping.  The ROM is not a
smoother; it is a biased estimator of the workpiece flow.  This
plausibly comes from the POD reconstruction fitting a small number
of modes to a field that is dominated by low-magnitude workpiece
velocity plus high-magnitude pin rotation — the fit optimises the
high-energy pin region and offsets the workpiece to hit the
overall energy budget.

This changes the acceptance criterion for §5 (uniform-grid) and §6
(block-refined) in a specific way:

- The residual is **not** high-frequency, so projecting onto a
  uniform grid will not "filter" it away.  §5 will not close the
  gap by throughput alone.
- The residual is **not** localised in a single spatial region
  (it's the workpiece bulk bias plus a shear-layer detail), so
  §6's block refinement of just the near-pin region will not fix
  it either.
- The real fix is a ROM reconstruction that **enforces the mean
  workpiece velocity as a constraint** — e.g. by projecting onto a
  bilinear + POD basis or by using a mode subset that captures the
  low-frequency workpiece modes explicitly.  Not covered by the
  current roadmap; noted as a follow-up.
- The `rot_cyl_2026` analytic case is now more important, not
  less: it will tell us whether the tracker's own budget on a
  strictly div-free field with a strictly-known mean is closer to
  the FOM or to the ROM behaviour we observe here.  If tracker
  error dominates on the analytic case, the workpiece bias in §4
  is genuinely a ROM defect; if not, it is a shared
  tracker + reconstruction artefact.

---

## Figure 1 — radial |v| profiles

![figA](../paper_figs/rom_pt_step4/figA_radial_velocity_profiles.png)

Four rows (cases 000, 001, 003, 004), three columns (FOM, ROM,
residual).  Each panel plots |v| along four radial lines from the
origin outward: east (θ=0°, blue), north (90°, green), west (180°,
orange), south (270°, red).  Vertical dotted line = r = 10 mm
(the near-pin bin boundary).

**How to read.** Left and middle columns should look almost
identical if the ROM matches the FOM well.  Right column shows the
absolute difference — its peak is where the ROM residual lives
along that line.

**Key observations.**

- The FOM and ROM columns are visually indistinguishable at plot
  resolution; only the east line (blue) shows oscillations
  characteristic of the pin's rotational structure, and both ROM
  and FOM capture these oscillations at the same amplitude and
  phase.
- The residual peaks near r ≈ 7–8 mm (just inside the near-pin
  boundary) and drops sharply at r > 10 mm.  Case 003's residual
  peak reaches ~3.5 cm/s (7 % of local velocity), case 000's ~3.6
  cm/s (10 % of local peak velocity but higher local scale).
- The east line dominates the residual in every case — that is
  the downstream/upstream axis where the tool wake sits.  Fig 5a
  of the §2/§3 report showed the same directional asymmetry.

## Figure 2 — azimuthal |v| profiles

![figB](../paper_figs/rom_pt_step4/figB_azimuthal_velocity_profiles.png)

Four rows (cases), four columns (rings at r = 6, 9, 12, 15 mm).
Blue solid = FOM, red dashed = ROM.  Column titles carry the
LEVEL-derived inside/outside classification (r = 6 mm is inside
the pin body, r = 9 – 15 mm outside).

**How to read.** For each ring the two curves should overlay if
the ROM captures the field faithfully.  A vertical offset means
mean-level bias; a shape mismatch means structural bias.

**Key observations.**

- r = 6 mm (inside pin): |v| = 0.28 – 0.56 m/s (pin rotation
  velocity, `ω · r`).  ROM tracks FOM within 1 – 2 % across all
  cases and all θ.
- r = 9 mm (immediate shear-layer downstream of pin): |v| collapses
  by 30× to 0.008 – 0.015 m/s.  ROM is ~30 % higher than FOM at
  every θ.  Dropouts to 0 in the FOM at ~θ = 30 – 70° and 200° are
  where the probe hits level-set-clipped cells (material physically
  outside the flow domain); ignore those spikes and read the
  envelope.
- r = 12 mm (bulk workpiece): |v| ≈ 0.010 – 0.012 m/s.  ROM
  systematically 20 – 40 % above FOM.  Structure (relative peaks
  and troughs vs θ) is well-matched; the offset is the story.
- r = 15 mm (bulk workpiece far): same 20 – 40 % overestimation.
  The bias does not decay with distance from the pin.

## Figure 3 — azimuthal FFT spectra

![figC](../paper_figs/rom_pt_step4/figC_azimuthal_fft_spectra.png)

Same layout as Fig B (4 cases × 4 rings).  Log-log |FFT(|v|)| vs
azimuthal wavenumber k = 1..N/2.  Blue = FOM, red dashed = ROM,
green dotted = residual.

**How to read.** The residual amplitude relative to FOM/ROM tells
us where the ROM residual sits in the azimuthal spectrum:

- residual *below* FOM/ROM at some k → ROM captures that scale.
- residual *at* FOM/ROM level at some k → ROM misses that scale
  completely (this is where the smoothing hypothesis would predict
  a signature — but it would predict "at high k", not "at low k").
- residual *above* FOM/ROM → ROM introduces artefact at that scale
  (not seen here).

**Key observations.**

- r = 6 mm (inside pin): classic decay ~ k⁻¹·⁵ for both FOM and
  ROM.  Residual sits 1 decade below both across the whole k
  range.  ROM captures inside-pin azimuthal structure at every
  scale.
- r = 12 mm (bulk workpiece): FOM and ROM plateau at ~10⁻² (the
  workpiece velocity structure is dominated by low-k modes;
  higher-k content is near noise floor).  Residual **matches** FOM
  and ROM at k = 1 – 4 and falls 1 – 2 decades below at k > 5.
  This is the spectral fingerprint of the workpiece-bias story:
  ROM captures the small-scale detail correctly but is offset in
  the k = 1 – 4 mean amplitude.  **Opposite of the smoothing
  prediction**, which would put the residual at high k.
- r = 15 mm: similar to r = 12 mm.  The bias mode extends into the
  bulk workpiece.
- r = 9 mm: the shear-layer detail is noisy; residual roughly
  tracks FOM/ROM across k with no clean spectral separation.  Not
  the strongest evidence but not contradicting the low-k-bias
  story.

## Figure 4 — divergence complementary CDFs

![figD](../paper_figs/rom_pt_step4/figD_divergence_ccdf.png)

Per-tet |∇·v| for cells with centroid r ≤ 10 mm, log-log CCDF.
Blue solid = FOM, red dashed = ROM, one panel per case.  A truly
divergence-free field would be a spike at x = 0 (i.e. all
probability at |∇·v| = 0).

**How to read.** How far right does the curve extend?  Right =
larger |∇·v|.  At P = 10⁻² (1 % of near-pin cells), what value
does the CCDF cross?

**Key observations.**

- Every case: FOM and ROM CCDF are visually indistinguishable —
  the ROM inherits the FOM's divergence structure directly.  In
  cell-by-cell numerical terms, corr(div_fom, div_rom) = 0.992 on
  case 003.
- All four cases have significant tails: 1 % of near-pin cells
  have |∇·v| > 25 (cases 000, 001), > 32 (case 004), > 42 (case
  003).
- Case 003's CCDF extends the furthest right — max |∇·v| ≈ 855.
  This is the direct Eulerian confirmation that case 003's
  velocity field carries the strongest local divergence.

## Figure 5 — Eulerian divergence vs Lagrangian PT gap

![figE](../paper_figs/rom_pt_step4/figE_divergence_vs_pt.png)

Scatter: X = FOM near-pin |∇·v|_rms at ts = 119, Y = ROM–FOM PT
rel_rms at step 950 (pooled across bins).  One point per case.

**How to read.** A positive correlation would mean cases with
higher FOM divergence tend to have larger Lagrangian PT gaps.

**Key observations.**

- **Clean positive ordering.**  Case 003 (rms = 34, PT = 42.8 %),
  case 004 (rms = 25, PT = 18.1 %), case 001 (rms = 18, PT = 24.4
  %), case 000 (rms = 18, PT = 12.2 %).
- The ordering is almost monotone except case 001 has PT slightly
  higher than case 000 despite similar divergence.
- Compared to the §2/§3 report's Fig 2 (Eulerian velocity
  rel_rms vs Lagrangian PT rel_rms), this scatter has *more*
  predictive power: case 003 is unambiguously the outlier on both
  axes here, whereas in Fig 2 case 003 was the outlier on Y only
  and *lowest* on X.
- **This is a candidate replacement for the Eulerian rel_rms
  acceptance criterion.**  A ROM whose reconstructed velocity is
  low-rel_rms but high-|∇·v|_rms will fail Lagrangian use.  This
  is a new §7 diagnostic worth adding to the roadmap.

## Figure 6 — residence time in annular probe

![figF](../paper_figs/rom_pt_step4/figF_mixing_residence_time.png)

Four cases (columns) × two rows (linear top, log bottom).  Blue
solid = FOM tracker particle count inside the annular probe (r ∈
[5, 10] mm), red dashed = ROM tracker.  HCT-on data shown; HCT-off
is essentially identical.  Log panels carry the final-step values
as annotations.

**How to read.** The linear panel shows the peak (when the wave
of particles arrives at the pin) and the initial decay.  The log
panel shows the tail — the particles that get stuck circulating
in the annulus and remain there indefinitely.  ParaView inspection
confirms: the FOM tail is real trapped material, the ROM tail is
much smaller.

**Key observations.**

- **Linear panel — arrival wave**: ROM peaks earlier and slightly
  lower than FOM in every case.  Peaks are at step 400 – 800
  depending on case; ROM peak is 5 – 15 % below FOM peak.  This
  reflects the workpiece-velocity overestimation from finding 2:
  ROM particles arrive at and pass through the annulus faster.
- **Log panel — trapping tail**: after step 1500, FOM curves
  plateau at 800 – 15,000 particles while ROM curves drop to
  55 – 1,700.  Ratios:
    - Case 000 · FOM 14,606 / ROM 1,016 → 14× more FOM trapping
    - Case 001 · FOM 774 / ROM 7 → 110× more FOM trapping
    - Case 003 · FOM 2,714 / ROM 1,260 → 2× more FOM trapping
    - Case 004 · FOM 4,563 / ROM 1,738 → 2.6× more FOM trapping
- **Case 001 is the extreme**: ROM traps virtually zero (7
  particles) while FOM traps 774.  This case has the fastest
  workpiece velocity (Fig B r = 12 mm: |v| = 0.010 m/s FOM vs
  0.013 m/s ROM) and its ROM overestimation is enough to blow
  out any recirculation.
- **Case 003 has the smallest ratio** despite having the largest
  Eulerian residual overall (§2/§3 Fig 5).  The Fig 5a ring
  around the pin actually *keeps* ROM particles circulating in
  the annulus longer than the other cases' more diffuse
  residuals do.

## Figure 7 — pairwise separation (top-FTLE proxy)

![figG](../paper_figs/rom_pt_step4/figG_mixing_pairwise_separation.png)

Same layout as Fig F.  Y axis: geometric-mean distance between
nearby-seeded particle pairs, log scale.  A larger slope means
faster chaotic separation of nearby trajectories — a Lagrangian
proxy for the top FTLE eigenvalue.

**How to read.** Rising curve → chaotic separation.  Slope
comparison between FOM and ROM tells us whether the two fields
support the same rate of chaotic mixing.

**Key observations.**

- All curves rise from ~1.1e-3 m at step 0 to 1.9 – 2.6e-3 m at
  step 2000: pair separations grow by 2–2.5× over the full run.
- **The FOM vs ROM ordering is not universal.**  Case 001: ROM
  below FOM throughout.  Case 003 HCT-off: FOM slightly above ROM
  early, then ROM crosses over and ends higher.  Case 004
  HCT-off: same pattern, ROM ends above.  Case 000: ROM slightly
  above FOM throughout HCT-on, similar HCT-off.
- **No clean "ROM under-mixes" story.**  The chaotic-advection
  rate is case-dependent.  The finding survives the earlier
  low-pass-smoothing framing only if we interpret "smoothing" as
  something that happens per-case rather than universally.

---

## Coverage matrix — roadmap §4

| §4 sub-item | Done here |
|---|---|
| 4.1 Load `v_FOM` and `v_ROM` at ts=119 for a case | 4 cases loaded |
| 4.2 Compute residual as VTU | `residual_cylindrical_<case>.vtu`, 4 files |
| 4.3 Power spectrum of residual along pin-radial lines | Fig C (azimuthal FFT; radial FFT not shown as radial line is not uniformly sampled) |
| 4.4 Cross-check FOM+HCT-off vs FOM+HCT-on | Fig 5c of the §2/§3 report + trapping figs in §2/§3 |
| 4.5 Residence time + pairwise separation | Figs F, G |

Additional diagnostics not in the original roadmap §4 but relevant:

| additional item | Done here |
|---|---|
| Divergence sampling of FOM and ROM (§4-adjacent) | Figs D, E |
| Azimuthal ring FFT | Fig C |
| Radial line comparison | Fig A |
| Azimuthal ring |v| comparison | Fig B |

## What comes next (§5, §6)

The **finding that FOM is not divergence-free** (Fig D) reframes
the §5 uniform-grid experiment.  If the ROM–FOM Lagrangian gap is
dominated by the FOM's own divergence structure (not by ROM
low-rank truncation), then projecting the ROM velocity onto a
uniform Cartesian grid will not close the gap on its own.  Two
follow-up experiments become the priority:

- **`rot_cyl_2026`** (analytic potential-flow-past-cylinder) — the
  four-mesh case (analytic / uniform / 2-lvl / 4-lvl) will tell us
  the reference behaviour of the tracker on a strictly div-free
  field and how far mesh interpolation can push us from that
  reference.  This gives us the *tracker's own div-free budget*
  independently of the FOM's discretisation defect.
- **20-case cohort** — the overnight sweep will test whether the
  |∇·v|_rms → PT rel_rms correlation from Fig E holds up on 16
  additional cases.  If it does, `|∇·v|_rms` at ts=119 becomes a
  case-level acceptance criterion for whether a case is
  Lagrangian-safe.
