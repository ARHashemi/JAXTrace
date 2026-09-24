# ROM particle tracking — §2 & §3 results report

**Date:** 2026-07-17
**Data:** 4-case sweep (000, 001, 003, 004), 2000 steps × DT = 3.75e-3 s,
360 k particles per run. Reporting step: 950 (t ≈ 3.56 s) for the main
figures; step 500 (t ≈ 1.88 s) kept as a mid-run reference.
**Corresponding roadmap:** `docs/rom_pt_roadmap.md` §§ 2 – 3.


> ## ⚠ Errata (2026-09-09) — these numbers used a defective ROM basis
>
> Every ROM result in this report was measured on velocity fields
> reconstructed from the **shipped FEMUSS basis**
> (`cylindrical.som.fswrom.basis`), which reconstructs this cohort at
> **4.04 % L2**. A POD built directly from the same 20 snapshots reaches
> **0.52 % with the same 3 modes** — the shipped modes simply do not span
> this data (its stored sigmas 101.30/6.09/3.68 differ from what these
> snapshots produce: 111.52/5.65/4.06). See
> [`rom_pt_ourpod_rerun.md`](rom_pt_ourpod_rerun.md).
>
> **The tracking sweep was rerun on a corrected basis.** ROM–FOM
> displacement dropped by **×0.34 at step 950** (6.79 → 2.32 mm mean over
> cases 000/001/003/004) and **×0.21 at step 2000** (19.79 → 4.22 mm).
>
> **What survives:** the *qualitative* findings below — near-pin
> concentration, HCT being a small perturbation, the ROM under-mixing —
> have not been re-checked against the corrected runs. Treat them as
> open.
>
> **What is overturned:** Finding 1's specific amplification factors are
> superseded. The headline claim that "Eulerian rel_rms does not predict
> Lagrangian rel_rms" is *strengthened*, not weakened: with a ~5-8x better
> Eulerian field the Lagrangian error fell only ~3x, so the amplification
> factors actually *rose* (case 003: 6.74x -> 9.33x).

## Data sources

| artefact | path |
|---|---|
| per-r-bin displacement table (rom-vs-fom, step 950) | `rom_out/pt_r_binning_step950.csv` |
| per-r-bin displacement table (HCT ablation, step 950) | `rom_out/pt_r_binning_hct_ablation_step950.csv` |
| same at step 500 | `rom_out/pt_r_binning_step500.csv`, `rom_out/pt_r_binning_hct_ablation_step500.csv` |
| compare VTUs (16 files, step 950) | `.../<case>.gid/post_pt_compare/<label>/<label>_step950.vtu` |
| compare VTUs (step 500) | `.../<case>.gid/post_pt_compare/<label>/<label>_step500.vtu` |
| aggregate compare logs | `.../post_pt_compare/rom_vs_fom_hct_{on,off}/compare.log` |

All step-950 figures live under `paper_figs/rom_pt_step2_step3/step950/`
in PNG + SVG at 300 dpi. Step-500 figures live under
`paper_figs/rom_pt_step2_step3/` at the same DPI.
Regenerate with `scripts/build_step2_step3_figures_step950.py` (step
950) or `scripts/build_step2_step3_figures.py` (step 500).

---

## Methodology

### Particle subset selection

Trackers apply **ballistic and inlet-plane extensions** when a particle
leaves the mesh interior — an escaped particle continues on a
physically defined trajectory using the last valid velocity and
inlet-extended geometry. So an escaped particle is not gone from the
physics; it is still evolving under the same extension rule for FOM
and ROM. Restricting the ROM-vs-FOM statistics to particles that both
trackers agree are still inside the mesh (`both_alive` in the compare
tool) has two problems:

1. It **hides** the ROM-vs-FOM residual that accumulates on the
   extension. That residual is small per step but real, and it
   matters most exactly where the compare tool would drop the
   particle from the sample.
2. It **biases the near-pin bin**. Particles that escaped through the
   outlet (all of them were seeded upstream and moved outward through
   the tool) are dropped from the outer-domain sample, so the outer
   bin's count no longer reflects the seed distribution.

The step-950 figures therefore use the **full particle set** (all
360 k). This changes numbers vs the earlier step-500 report that used
`both_alive`; the corrected story is stronger, not weaker (see Fig 2).

### Trapped-particle definition (divergence-free proxy)

For an **incompressible advective field** with proper inlet/outlet
handling, every streamline seeded upstream of the tool should pass
around the tool and reach the outlet. In particular, no particle
seeded far upstream of the tool should be found *inside* the near-pin
cylinder at a late timestep, unless the numerical velocity field has
drifted from divergence-free at the tracker's spatial scale.

We define a particle as **trapped** at the reporting step if its final
radial position satisfies

    r_final = sqrt(x**2 + y**2) ≤ r_near_pin       (default 10 mm)

evaluated separately for each tracker (FOM, ROM) at step 950. Because
all particles were seeded far upstream of the tool, this final-step
check is sufficient — the trapped set is exactly the set of particles
that failed to advect past the tool.

For a truly div-free field these counts would be near zero. Any
non-zero value is a Lagrangian proxy for how far off div-free the
numerical velocity behaves under RK4 integration, and comparing FOM
vs ROM at the same case tells us whether the ROM's low-rank
truncation *improves* or *degrades* the div-free character.

### rel_rms convention

Every "rel_rms" reported here is normalised against the **FOM
bounding-box diagonal** of the reporting-step particle cloud, so
comparisons across cases and reporting steps are on the same scale.
This matches the "rms / FOM diagonal" line the compare tool prints
in its aggregate block.

---

## Headline findings

1. **Eulerian rel_rms does not predict Lagrangian rel_rms.** At step
   950 the amplification factors are: case 000 = 1.90×, case 001 =
   3.12×, case 003 = **6.74×**, case 004 = 3.64× (Fig 2).  Case 003 —
   the best-reconstructing case at Eulerian 2.62 % — has the *worst*
   Lagrangian gap at 17.7 %.  Cases 001/003/004 sit ~3–7× above y = x;
   case 000 sits below in step-500 but crosses to 1.9× at step 950.
   The amplification grows with integration time.

2. **Near-pin displacement dominates the ROM–FOM gap in every case
   except 000.** Cases 001, 003, 004 all show near-pin rel_rms 2–3×
   the outer-domain value (Fig 1).  Case 003 is extreme: near-pin
   43 %.  Case 000 is uniform ~12 % across bins, and its spatial map
   (Fig 5a) shows why — the residual is not concentrated on any
   coherent structure; it is a diffuse background of trapped particles
   (see Finding 4).

3. **HCT-3D is a small tracking perturbation.** On the FOM side it
   shifts trajectories by 3–18 % near-pin (Fig 3 left), on the ROM
   side by 3–44 % near-pin (Fig 3 right).  These are large in absolute
   terms but they do **not** close the ROM–FOM gap: Fig 4 shows Δ
   (rel_rms with HCT-on minus rel_rms with HCT-off) between −0.93 and
   +0.42 %pt across all cases.  HCT is neither compensating for the
   ROM's Eulerian residual nor masking it; the two effects live on
   different scales.

4. **The ROM tracker is more "divergence-free" than the FOM tracker
   for the same case, but neither is fully div-free.** Fig 7 shows
   that in every case fewer ROM particles are trapped than FOM
   particles at step 950. Case 000 is extreme: FOM traps 46 % of
   seeded particles (166,815), ROM traps 30 % (108,387). Cases 001
   and 003 have very low ROM trapping (0.3 – 0.4 %) but 4 – 5 % FOM
   trapping. Case 004 sits in the middle at FOM 14 % / ROM 5 %. So:

   - Every case violates the div-free proxy to some extent.
   - The ROM's smoothed velocity field is a *better* advecter through
     the tool region than the FOM field it is trained on. This is
     *not* evidence that the ROM is more physically correct — it may
     simply mean the ROM has damped the near-pin shear structure that
     traps particles in the FOM tracker. This is exactly the
     under-mixing signature that would concern a downstream FSW mixing
     analysis.

5. **HCT-3D does not improve trapping.** Fig 7b shows Δ (HCT-on
   minus HCT-off trapping) is between −0.00 %pt and +0.96 %pt across
   every case and both trackers.  Turning HCT on typically makes
   trapping slightly *worse*.  So HCT-3D as currently implemented is
   not a divergence-free-improving mechanism.

6. **Case 003 is the "field-accurate but flow-sensitive" case.**
   Best Eulerian error, worst Lagrangian gap, biggest HCT effect on
   the ROM tracker (44 % near-pin, Fig 3 right), most coherent
   spatial residual pattern (Fig 5a/b: bright ring surrounding the
   pin).  This is the canonical case that motivates §4 of the
   roadmap.

---

## Figure 1 — near-pin vs outer, per case and HCT setting

![fig1](../paper_figs/rom_pt_step2_step3/step950/fig1_step950_r_bin_bars.png)

Grouped bars, all-particles subset. X = case (000, 001, 003, 004).
Y = ROM–FOM displacement rel_rms as % of the FOM bbox diagonal.
Colour: red = near-pin bin (r ≤ 10 mm), blue = outer (r > 10 mm).
Fill vs hatch: solid = HCT-on, diagonal = HCT-off.

**How to read.** For each case, four bars stand next to each other:
red-solid, blue-solid (HCT-on: near, outer), red-hatched, blue-hatched
(HCT-off: near, outer). A taller red than blue means "near-pin is
worse than outer". Solid and hatched next to each other are nearly
identical means "HCT-on and HCT-off give the same PT error".

**Key observations.**

- Cases 001, 003, 004 all show the near-pin bin dominating.  Case 003
  is dominant: near-pin 43 % vs outer 16 %.
- Case 000 is *uniform*: both bins ~12 %.  There is no near-pin
  concentration to speak of.  This is because most of case 000's
  particles are still inside the mesh interior — they never made it
  out to the outer bin (see Fig 7).
- HCT on/off makes almost no visible difference — within-case red-solid
  vs red-hatched are the same height, ditto blue.

## Figure 2 — Eulerian → Lagrangian amplification scatter

![fig2](../paper_figs/rom_pt_step2_step3/step950/fig2_step950_eulerian_vs_lagrangian.png)

X = per-case Eulerian rel_rms of the ROM velocity reconstruction
(from `docs/rom_reconstruction_findings.md`, "centered" formula).
Y = aggregate Lagrangian PT rel_rms at step 950 (pooled from the two
r-bins in the CSV, weighted by particle count).  Circle = HCT-on,
square = HCT-off.  Dotted line = y = x.  Each case is annotated once
with its amplification factor = Y / X.

**How to read.** A point on the diagonal means PT error equals
Eulerian error.  Above the diagonal means PT is *worse* than Eulerian.
The vertical distance is the "consistency error" of the ROM as a
tracking-input field.

**Key observations.**

- All four cases sit above y = x at step 950 (compared to case 000
  being below at step 500).  This is direct evidence that the
  amplification **grows with integration time**: as trajectories get
  longer, the Eulerian residual compounds into a bigger Lagrangian
  error.
- The amplification is not monotone in the Eulerian error.  The
  smallest Eulerian error (case 003) has the largest Lagrangian error.
  So the ROM's Eulerian acceptance criterion (small `rel_rms`) is
  *not* sufficient for tracking use.
- HCT on/off points are nearly on top of each other for every case,
  matching Fig 4 — HCT does not move Lagrangian error appreciably.

## Figure 3 — HCT-3D tracking perturbation

![fig3](../paper_figs/rom_pt_step2_step3/step950/fig3_step950_hct_ablation_bars.png)

Two panels, same y-axis scale. Left: turn HCT-on vs off in the FOM
tracker with the FOM velocity — how much does HCT alone move
trajectories?  Right: same on the ROM tracker with the ROM velocity.
Red = near-pin, blue = outer.

**How to read.** These are on-vs-off comparisons using the *same*
velocity source, so any bar height is purely the HCT recovery effect
on tracking. Compare to Fig 1 for scale: Fig 3 bars are the "HCT
tracking overhead", Fig 1 bars are the "ROM-vs-FOM gap".

**Key observations.**

- Case 003 dominates both panels.  HCT-3D on the ROM field shifts
  case-003 near-pin trajectories by 44 % — an enormous perturbation.
- Near-pin bars are 5–10× outer bars in every case, on both trackers.
- The ROM side (right) has bigger HCT effects than the FOM side
  (left).  The ROM's smoother field means HCT has more room to
  interpolate finer detail into the tracker, which changes
  trajectories more.

## Figure 4 — does HCT close the ROM/FOM gap?

![fig4](../paper_figs/rom_pt_step2_step3/step950/fig4_step950_hct_gap_closure.png)

Near-pin ROM–FOM rel_rms per case, with HCT-on (green) and HCT-off
(red) shown side by side. Δ labels give (on − off) in percentage
points.

**How to read.** If HCT is *compensating* for the ROM's Eulerian
residual, HCT-on should be substantially lower than HCT-off (Δ ≪ 0).
If HCT is *masking* the residual (making a defect look small at the
tracker level without fixing it), the two are equal by coincidence
but Fig 5 would show incoherent residual patterns.  If HCT is neither,
Δ ≈ 0 and Fig 5 shows coherent residuals — which is what we see.

**Key observations.**

- Δ ∈ [−0.93, +0.42] %pt across all 4 cases — all much smaller than
  the base 12 – 43 % rel_rms values.  HCT is a small perturbation on
  the ROM–FOM gap, not a rescue.
- Case 003 is the only slightly-negative Δ (−0.93 %pt); every other
  case is slightly positive.

## Figure 5 — spatial residual maps

Four separate figures, one per comparison.  In each, particles are
plotted at their FOM final (x, y) position and coloured by
ROM–FOM displacement magnitude.  All-particles subset.  White dashed
circle = r = 10 mm near-pin boundary.  Colour scale is clipped at the
99th percentile of the pooled displacement across the four case panels
of that figure.  A subsample of ~15 k particles is drawn per case for
render speed.

### Fig 5a — ROM (HCT-on) vs FOM (HCT-on)

![fig5a](../paper_figs/rom_pt_step2_step3/step950/fig5a_step950_spatial_rom_vs_fom_hct_on.png)

**The primary §2 answer.**  How much do ROM and FOM tracking agree
when both use HCT-3D?

- Case 000: a **large trapped population** sits inside the near-pin
  circle (dark blue clouds) — the ROM-vs-FOM residual there is
  modest because both trackers are stuck in similar orbits.  The
  outer domain is mostly empty because most particles never made it
  past the tool.
- Case 001, 003, 004: particles have advected through the tool
  region and formed a downstream cloud on the right (outer bin).
  A bright yellow-green residual sits on a coherent arc surrounding
  the pin cavity on the left side (particles that circulated the
  pin before exiting).  This is the near-pin residual dominating.
- Case 003 shows the brightest, widest ring — matching its 43 %
  near-pin rel_rms.

### Fig 5b — ROM (HCT-off) vs FOM (HCT-off)

![fig5b](../paper_figs/rom_pt_step2_step3/step950/fig5b_step950_spatial_rom_vs_fom_hct_off.png)

Same spatial structure as Fig 5a — HCT-off does not change *where*
the residual sits, only marginally *how much*.  This is the visual
proof of the "HCT is a perturbation, not a rescue" claim from Fig 4.

### Fig 5c — FOM HCT-on vs FOM HCT-off

![fig5c](../paper_figs/rom_pt_step2_step3/step950/fig5c_step950_spatial_fom_hct_on_vs_off.png)

Same reference field, HCT toggled.  This is the **isolated HCT
tracking effect** on the reference tracker.  The residual is
concentrated on the shear layer around the pin, matching the physical
expectation: HCT-3D changes tracking most where velocity gradients
are largest.

### Fig 5d — ROM HCT-on vs ROM HCT-off

![fig5d](../paper_figs/rom_pt_step2_step3/step950/fig5d_step950_spatial_rom_hct_on_vs_off.png)

Same as 5c but on the ROM field.  Case 000's panel is particularly
striking: the HCT on/off residual traces a hook-shaped structure
inside the pin cavity that mirrors the trapped-particle streamline
pattern.  Case 003's dominant residual sits *outside* the near-pin
boundary — HCT is shifting particles through the shear layer more
than inside the cavity itself.

## Figure 6 — displacement CDFs

![fig6](../paper_figs/rom_pt_step2_step3/step950/fig6_step950_displacement_cdf.png)

Cumulative distribution of per-particle |ROM − FOM| at step 950,
log x-axis, one panel per case, near-pin (red) vs outer (blue).

**How to read.** If the red curve sits to the right of the blue
curve, near-pin particles have larger displacements than outer
particles.  The horizontal separation between the curves at fixed
F is the "near-pin residual excess" at that percentile.

**Key observations.**

- Cases 001 and 003 show clean red-shifted separation — the near-pin
  distribution is uniformly ~1 decade to the right of the outer
  distribution.  Case 003 has the biggest separation.
- Case 000 has red and blue almost overlapping — the residual is
  uniform, consistent with Fig 1 uniform bars.  This is the trapped-
  particle signature: near-pin and outer are drawn from the same
  distribution because most particles never made the trip out.
- Case 004 has near-overlap up to ~10⁻³ then a heavier near-pin tail
  above that.  Middle case.

## Figure 7 — trapped-particle count (divergence-free proxy)

![fig7](../paper_figs/rom_pt_step2_step3/step950/fig7_step950_trapped_particles.png)

Per case, per tracker (red = FOM, blue = ROM): particles with
r_final ≤ 10 mm at step 950 as a percentage of seeded (360 k).
Bar labels give the absolute count and percentage.  Data is from
the rom_vs_fom_hct_on comparison; the r values are computed
independently on FOM and ROM final positions, so this is a genuine
per-tracker measurement.

**How to read.** For a truly divergence-free field with proper
inlet/outlet handling, every bar should be ≈ 0.  Non-zero bars
indicate the tracker is either *literally* trapping particles
(numerical stagnation) or *effectively* trapping them (recirculation
in a region that should have net outflow).

**Key observations.**

- **Case 000** is dramatically off from div-free: FOM traps 46 %,
  ROM traps 30 %.  This is a case-specific problem, not a ROM problem —
  the FOM velocity field itself has structure that keeps almost half
  the seeded particles from ever leaving the tool region.
- **Cases 001 and 003** are close to div-free on the ROM side
  (0.3 %, 0.4 %) but not on the FOM side (5 %, 4 %).  The ROM's
  smoothed field is *better* at passing streamlines through the tool
  than the FOM.  This should be taken carefully — it does not mean
  the ROM is more physically correct; it likely means the ROM has
  damped the small-scale shear structure that traps particles.
- **Case 004** sits in between: FOM 14 %, ROM 5 %.
- **In every case ROM < FOM.**  The 3-mode POD truncation removes
  the sharpest local velocity structure — which happens to be
  exactly the structure responsible for near-tool recirculation and
  trapping in the FOM.

## Figure 7b — HCT effect on trapping

![fig7b](../paper_figs/rom_pt_step2_step3/step950/fig7b_step950_trapping_hct_effect.png)

Per case, per tracker, HCT-on (green) vs HCT-off (red).
Δ = (on − off) trapping percentage.

**Key observations.**

- Every Δ is between −0.00 %pt and +0.96 %pt.  HCT-3D as
  implemented has no meaningful effect on the div-free proxy.
- Δ is slightly *positive* in most cases (HCT-on traps marginally
  more), which is the opposite of the naïve "higher-order recovery →
  more div-free" expectation.  A likely explanation: HCT introduces
  intra-element velocity structure that is not calibrated for
  divergence-free-ness in the finite-element sense; it improves
  velocity smoothness but at the tracker's spatial scale, that
  slight extra structure can push a marginal particle back inside
  the near-pin circle.

---

## Reference: step-500 figures

The step-500 figures are the same set of visuals at t ≈ 1.88 s.  They
tell an earlier chapter of the same story: the amplification is
smaller (case 003 at 3.86× vs 6.74× at step 950), near-pin
concentration is present but less extreme, and trapped-particle
counts are irrelevant because particles are still in transit.
Kept for reference at `paper_figs/rom_pt_step2_step3/fig*.png` /
`fig*.svg`.

---

## Coverage matrix — §2 and §3 deliverables

| Roadmap deliverable | Location |
|---|---|
| §2 · particles.vtkhdf for FOM + ROM, HCT-on + HCT-off, all 4 cases | FOM/ROM trees under `<case>.gid/post_pt/<variant>/` |
| §2 · compare.log + per-particle displacement VTU (step 500 + 950) | `<case>.gid/post_pt_compare/<label>/` |
| §2 · per-r-bin displacement | Fig 1 |
| §2 · cohort Eulerian vs Lagrangian | Fig 2 |
| §2 · CDF / spatial coherence | Fig 5a, Fig 6 |
| §2 · divergence-free proxy (trapping) | Fig 7 |
| §3 · four combinations (FOM/ROM × HCT-on/off) tracker archives | on disk |
| §3 · per-r-bin HCT ablation stats | Fig 3 |
| §3 · masking-vs-compensating decision | Fig 4, Fig 5c/d, Fig 7b |

## What comes next (§4)

- Residual velocity field: `v_FOM − v_ROM` at ts = 119, case 003
  (natural pick — brightest Fig 5 ring).  Roadmap §4.2.
- 1D power spectrum of residual along pin-radial lines.  Roadmap §4.3.
- Mixing diagnostics (residence-time + pairwise separation) — already
  on disk under `ROM_recon_centered/<case>.gid/post_pt_mixing/`.
  Roadmap §4.5.
- **New question raised by Fig 7:** why does the ROM tracker trap
  fewer particles than the FOM tracker?  Candidate hypotheses:
  (i) POD truncation removes near-tool shear structure required for
  physical recirculation; (ii) POD reconstruction is more div-free
  than the FOM per-element approximation on cylindrical, at least in
  some cases.  Discriminating between these is a small experiment
  using the reconstructed PVTUs already on disk.
