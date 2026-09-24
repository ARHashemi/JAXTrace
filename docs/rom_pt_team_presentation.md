---
marp: true
theme: default
size: 16:9
paginate: true
footer: 'ROM particle tracking · A. R. Hashemi · 2026-07-20'
style: |
  /* Global tightening for a science-heavy 20-slide deck.
     Marp default (1280×720) does NOT auto-shrink content — slides
     overflow silently, which is what we saw on this deck.  The rules
     below reduce base font-size and image cap so every slide fits
     without needing to touch per-slide directives, except for the
     ones marked ".tight" or ".two-col" below.  */
  section {
    font-size: 22px;
    padding: 40px 55px 55px 55px;   /* leave room for footer + page num */
    justify-content: flex-start;
    line-height: 1.35;
  }
  section h1 { font-size: 34px; margin: 0 0 12px 0; }
  section h2 { font-size: 26px; margin: 0 0 10px 0; }
  section h3 { font-size: 20px; margin: 0 0 8px 0; }
  section p, section li { font-size: 20px; margin: 3px 0; }
  section table { font-size: 18px; margin: 6px 0; }
  section th, section td { padding: 4px 8px; }
  section img { max-height: 480px; display: block; margin: 6px auto; }
  section pre, section code { font-size: 16px; }

  /* Title slide */
  section.lead h1 { font-size: 44px; }
  section.lead h3 { color: #666; font-weight: normal; }

  /* Extra-tight variant for text-heavy slides.  Applied via
     `<!-- _class: tight -->` on the slide's first line. */
  section.tight { font-size: 19px; padding-top: 30px; }
  section.tight h2 { font-size: 22px; }
  section.tight p, section.tight li { font-size: 17px; margin: 2px 0; }
  section.tight table { font-size: 15px; }

  /* Two-column layout for image + text slides.  Wrap the two blocks in
     a <div class="cols"><div>...</div><div>...</div></div>. */
  .cols { display: grid; grid-template-columns: 1.4fr 1fr; gap: 20px;
          align-items: start; }
  .cols img { max-height: 520px; }

  /* De-emphasise the "presenter meta" line at the top of slide 1 */
  section.lead p { color: #888; font-size: 18px; }
---

<!-- _class: lead -->
<!-- _paginate: skip -->

# ROM particle tracking — progress report

### 4-case cohort · pipeline in place · headline findings

A. R. Hashemi — 2026-07-20 · 30-min team update

Roadmap: `docs/rom_pt_roadmap.md`
Deep-dive reports: `rom_reconstruction_findings.md`,
`rom_pt_step2_step3_report.md`, `rom_pt_smoothness_divfree_report.md`

---

<!-- _class: tight -->

## Slide 2 — Where we are today (roadmap §0)

**Goal.** Fast + accurate particle tracking on ROM-reconstructed velocity fields — replacing the FOM as the tracker's input for downstream FSW analysis.

**Why.** Each FOM PT run is ~20 min at 360 k particles × 2000 steps × single case. Across the 20-case cohort that is many GPU-hours per parameter study. A ROM-driven tracker would cost 1/10 of that.

**Roadmap sections.**

| # | Topic | Status |
|---|---|---|
| 1 | Reconstruction findings doc | **done** |
| 2 | FOM-vs-ROM PT comparison | **done for 4 cases**, 20-case sweep launched |
| 3 | HCT-3D ablation | **done for 4 cases** |
| 4 | Smoothness + divergence analysis | **done for 4 cases** |
| 5 | Uniform-grid experiment | `recirc_2026` **done**, `rot_cyl_2026` v1 done + fresh run tonight, **FSW all 20 cases done** with 6 winner-candidate variants (top-3 from Part 3 + 3 hypothesis-driven from Part 5). See [rom_pt_step5_step6_grid_report.md](rom_pt_step5_step6_grid_report.md). **Definitive winner: `4lvl_hct`** at 5.6 mm (mid-run) / 7.8 mm (final), ~42× faster than mesh. Grid topology is saturated: shear-band refinement, more nesting, and higher-resolution MALMO all *fail* to beat 4lvl_hct. |
| 6 | Block-refined grid | **done** (same report). Block-refined 4lvl with HCT-3D Stage-1 + trilinear Stage-2 wins. Trilinear beats tricubic on the 20-case cohort (Catmull-Rom overshoot compounds), and adding a 4th nested block gives no measurable gain (~1 % worse). |
| 7 | Diagnostic conventions | shared, reused across §§2–4 |

---

<!-- _class: tight -->

## Slide 3 — What "cohort" means

4 cases sampled from a 20-case FSW parameter grid (v_adv, ω_pin variations).

Each case, at ts = 119 (steady-ish state), has:
- FOM velocity field on a 180 k-node × 736 k-tet mesh
- ROM reconstruction via SLEPc + our loader with the `centered` formula (3 modes)
- Level-set field marking pin body vs workpiece
- 360 k particles seeded upstream of the tool, tracked for 2000 RK4 steps at Δt = 3.75 e-3 s

**Cases picked for the initial deep-dive: 000, 001, 003, 004.**
Selection covers best-reconstructing (003, 2.62 % Eulerian rel_rms) and worst-reconstructing (000, 6.40 %).

The overnight sweep now extending to all 20 cases (launched 2026-07-17 evening; runs into 2026-07-19).

---

## Slide 4 — §1 · ROM reconstruction quality

**Done in `docs/rom_reconstruction_findings.md`.**

Twenty-case cohort mean `rel_rms = 4.04 % ± 1.08 %`.

Best case: 003 at **2.62 %**.
Worst case: 000 at **6.40 %**.

Spatial breakdown: reconstruction residual concentrates near the pin (2–8 %); outer-domain residuals (40–80 %) are low-signal artefacts because both FOM and residual are near zero there.

Three reproducibility scripts live in `tests/rom/`:
- `rom_20case_sweep.py`
- `rom_time_sweep.py`
- `rom_spatial_residual.py`

**Bottom line: the ROM's Eulerian velocity error is small (~4 %) and case-dependent.**

---

## Slide 5 — §2/§3 · Question we are asking

"Given a low Eulerian reconstruction error, does the ROM produce equally good particle trajectories?"

Two things to test:
1. **§2**: how far do ROM-tracked particles drift from FOM-tracked particles?
2. **§3**: does the HCT-3D higher-order velocity recovery either **compensate** for the ROM's Eulerian residual (Δ negative) or **mask** it (Δ zero + incoherent residual pattern)?

Setup per case:
- Same seeding, same boundary walls, same integrator, same 2000 steps.
- **Four variants**: FOM × HCT-on, FOM × HCT-off, ROM × HCT-on, ROM × HCT-off.
- Reporting steps: 500 (mid-run) and 950 (late-run).

Analysis of both timesteps in `docs/rom_pt_step2_step3_report.md`.

---

<!-- _class: tight -->

## Slide 6 — What "gradient recovery" and "HCT-3D" mean

<div class="cols">
<div>

**The problem.** A P1 tet field has one gradient per element — piecewise constant, **jumps across every face**. The tracker's RK4 substep queries velocity at arbitrary interior points; a jumpy gradient produces kinked trajectories and integration error even when nodal values are perfect.

**Gradient recovery** (Zienkiewicz-Zhu 1992, SPR / PPR family). Reconstruct a **smooth nodal gradient** $\mathbf G_a$ at every mesh node $a$ by least-squares fitting a low-order polynomial through the raw per-element gradients in a patch of neighbouring elements. SPR is *superconvergent* on regular meshes — the recovered $\mathbf G_a$ converges **one order faster** than the raw P1 gradient.

**In this deck: HCT-3D on** = use the reconstructed cubic; **HCT-3D off** = raw P1 barycentric interpolation.

</div>
<div>

**HCT-3D** = **Hsieh–Clough–Tocher, 3-D variant**. A $C^1$ piecewise-cubic macro-element on the *Alfeld split* of each parent tet (parent split into 4 sub-tets meeting at the centroid). Inside each sub-tet: cubic Bernstein polynomial. Across sub-tet interfaces: $C^1$. At parent vertices: interpolates the recovered $(\mathbf v_a, \mathbf G_a)$. Simplified form used by the tracker:

$$
\mathbf v(\mathbf p) = \sum_{a}\! N_a(\mathbf p)\bigl(\mathbf v_a\!+\!\mathbf G_a(\mathbf p\!-\!\mathbf x_a)\bigr) + \text{cubic bubble}
$$

**Why we use it.** Exact for **quadratic** velocity fields at parent vertices, centroid, spoke-edge midpoints, and face centroids. On the recirc test field: beats raw P1 by **~5× mean error, ~9× max**. Cost is one 16×10 LS solve per parent tet, stored once, then a polynomial eval per RK4 substep.

</div>
</div>

---


<!-- _class: tight -->

## Slide 7 — Analytic reference cases — why we need them

<div class="cols">
<div>

![recirc_2026 error curves](../paper_figs/rom_pt_analytic/recirc_2026_error.png)

Max (left) and RMS (right) trajectory error vs step for **recirc_2026**, 9 variants: {uniform, 2-lvl, 4-lvl} × {raw P1, vertex_taylor, HCT-cubic}. Reference is the mesh-free analytic solve.

</div>
<div>

**Why we need analytic cases.** §§2–4 tells us ROM tracks differ from FOM tracks, but **not how much of that gap is the tracker's own budget** on a mesh-interpolated velocity vs real ROM error.

**Fix.** Run the tracker on a **known div-free analytic velocity** on three progressively-refined meshes. Analytic-vs-mesh **is** the tracker's own budget. Anything on top is real ROM error.

Two cases in `FOM_analytic/`:
- **recirc_2026** (§A) — Gaussian × sin recirculation. Validation case (this slide).
- **rot_cyl_2026** (§B) — potential flow past rotating cylinder = closest analog to the FSW pin (next slide).

**recirc results tell us:**
- HCT-cubic separates cleanly from raw P1: **~5–6× smaller RMS** at step 6000.
- Vertex-Taylor is a much smaller step over raw P1 (they overlap in the plot).
- **Reconciles Slide 12 tension:** HCT looks useless in the FSW cohort because the ROM/FOM gap dominates. On a known-good field it delivers the designed ~6× improvement. **HCT works — it's just too small a correction next to the ROM residual.**

</div>
</div>

---

<!-- _class: tight -->

## Slide 8 — rot_cyl_2026 · the FSW-analog analytic case

<div class="cols">
<div>

**Field.** Potential flow past a rotating cylinder with circulation Γ (Venghaus §B, Magnus solution):

$$
\mathbf v = V_{\text{ref}}\!\left(1 - \tfrac{a^2(x^2-y^2)}{r^4}\right)\!\hat{\mathbf x} - \tfrac{\Gamma}{2\pi}\tfrac{y}{r^2}\hat{\mathbf x} + \ldots
$$

- Divergence-free **by construction** — cylinder surface `r = a` is a streamline.
- Free stream past a rotating solid = FSW's advection past a rotating pin. Same tool-in-flow topology as the cohort cases.
- Parameters retuned (`a` shrunk 4×, `V_ref` bumped 2.5×) so particles cross the domain and probe the near-cylinder shear layer.

**Refinement mirrors the roadmap §5/§6 plan:** shells at r ≤ a, 2a, 4a.

</div>
<div>

**What it will decompose:**

$$
\underbrace{e_{\text{ROM/FOM}}}_{\text{observed}} \;=\; \underbrace{e_{\text{tracker}}(\text{mesh})}_{\text{rot\_cyl gives this}} \;+\; \underbrace{e_{\text{ROM}}(\text{workpiece bias})}_{\text{residual}}
$$

- If **rot_cyl uniform-mesh error ≈ cohort ROM/FOM gap** → ROM is only as bad as any mesh-interpolated tracker; uniform-grid projection is a good fix.
- If **rot_cyl uniform ≪ cohort ROM/FOM gap** → ROM adds a real bias on top; the §4 workpiece-mean-bias finding stands and §5 uniform-grid alone won't close the gap.

**Reference is mesh-free.** `run_analytic.sh` uses `--velocity-source analytic` — velocity is `velocity_fn(x, y, z)` evaluated at every RK4 substage directly. **No mesh, no octree, no element search.** That is exactly why grid-based PT (§5) is attractive: a uniform grid also skips element search — one `floor((x − x₀) / Δx)` per query instead of L0/L1/L2 tet traversal.

**Status.** v1 runs completed (`summary.json`) but hit CFL saturation. Fresh runs with retuned `a` / `V_ref` land tonight.

</div>
</div>

---

<!-- _class: tight -->

<!-- _class: tight -->

## Slide 9 — §5 pipeline · mesh + 3 target grids · setup

<div class="cols">
<div>

**Source mesh: 4-lvl FEM tet meshes** (from `generate_meshes.sh`).

| case | nodes | tets | median edge (m) |
|---|---:|---:|---:|
| recirc_2026 | 1.36 M | 7.78 M | 3.0e-4 |
| rot_cyl_2026 | 83 k | 428 k | 3.9e-3 |

Source velocity for the projection tests is the **analytic** field evaluated at each mesh node — so the mesh-side error being projected is only the mesh's own P1 discretisation, no ROM residual.

</div>
<div>

**3 target grids per case** (all Cartesian, uniform cell size within each block, base at the mesh's natural resolution):

| grid | recirc cells | rot_cyl cells |
|---|---:|---:|
| **uniform_base** | 16 384 | 65 536 |
| **blockref_2lvl** (base + 2× block r ≤ r₁) | 139 264 | 589 312 |
| **blockref_4lvl** (base + 2× + 4× blocks r ≤ r₂ nested) | 1.19 M | 4.78 M |

Refine radii **r₁ / r₂**:
- recirc: 0.031 / 0.016 m (centred on Gaussian peak)
- rot_cyl: 0.016 / 0.008 m (multiples of scaled cylinder radius a')

**3 projection methods** onto each grid: raw P1 barycentric · SPR-recovered vertex-Taylor · **HCT-3D cubic** (Alfeld-split Bernstein).

Tool: `scripts/analytic_grid_projection.py`.

</div>
</div>

---

<!-- _class: tight -->

<!-- _class: tight -->

## Slide 10 — Mesh → grid projection error (aggregate, all 3 methods)

<div class="cols">
<div>

![projection error bars](../paper_figs/rom_pt_analytic/projection_error_bars.png)

Log-scale rel_rms per (case × grid × method).  Bars floored at 10⁻⁴; "≈ 0" annotations mark exact-zero cells.

</div>
<div>

**Findings.**

- **Uniform base grid**: rel_rms ≈ **0.00 % (recirc)** / **0.02 % (rot_cyl)** for every method — the grid is coarser than the mesh, all methods equivalent.
- **Block-refined grids · recirc**: raw P1 = **0.06 – 0.10 %**, vertex_taylor slightly higher, **HCT-3D jumps to 0.44 – 1.37 %** — 7–13× *worse* than raw P1.
- **Block-refined grids · rot_cyl**: all three methods within 4% of each other (~1.0 % on 2lvl, ~0.65 % on 4lvl).

**Interpretation.** On block-refined grids **finer than the source mesh**, HCT-3D's cubic reconstruction *inside* each parent tet overshoots at query points that raw P1 handles fine.  HCT was designed for query points **coarser** than the mesh, not finer.

The aggregate rel_rms hides *where* the error lives — Slide 11 splits it by region and reveals a very different story.

</div>
</div>

---

<!-- _class: tight -->

## Slide 11 — Where HCT wins vs loses · region-stratified

<div class="cols">
<div>

![projection error regions](../paper_figs/rom_pt_analytic/projection_error_regions.png)

Same 3 methods, but split into **near-feature** (hatched, r ≤ 10 mm) vs **outer** (solid).  Log-scale RMS |error|, per (case × grid × method).

</div>
<div>

**The answer to "when does HCT matter?":**

- **HCT wins in the OUTER region** — on rot_cyl at every grid, HCT and vertex_taylor slash outer-region RMS error by **1 – 2 orders of magnitude** vs raw P1.  Same trend on recirc uniform_base (all 3 methods hit ≈ 0 outer).
- **HCT loses NEAR the feature** — the hatched bars in every "block-refined" panel show P1 lowest, HCT highest.  Cubic overshoot dominates the aggregate rel_rms because near-feature cells are more numerous in the refined regions.

**Recipe.**  Use HCT where the field is *smooth and away from singularities/steep gradients*; use raw P1 in the refined shells around features.  A mixed strategy — HCT in outer blocks, P1 in inner blocks — would combine the strengths.

</div>
</div>

---

<!-- _class: tight -->

<!-- _class: tight -->

## Slide 12 — Grid vs mesh · wall time per refinement level

<div class="cols">
<div>

![wall time grid vs mesh](../paper_figs/rom_pt_analytic/grid_vs_mesh_time.png)

Grid PT vs mesh (HCT-3D) at 3 refinement levels — 6 grid runs (2 cases × 3 grids) launched via `run_grid_all.sh`. Wall time from each variant's tracker log.

</div>
<div>

**Take-home.**

- **Grid PT is 20 – 40× faster than mesh (HCT-3D)** at every refinement level.
- **Grid wall time is essentially flat across refinement**: recirc 92 → 133 s, rot_cyl 116 → 121 s. Adding refined blocks costs almost nothing because the JAX kernel keeps the same fixed O(1) inner loop; only the shared memory footprint grows.
- **Mesh wall time is also flat** (~2700 s recirc, ~4700 s rot_cyl) — dominated by L0/L1/L2 element search rather than the number of cells.

Bottom line: **grid PT wins the throughput race** regardless of refinement level.

</div>
</div>

---

<!-- _class: tight -->

## Slide 13 — Grid vs mesh · velocity projection accuracy

<div class="cols">
<div>

![velocity projection](../paper_figs/rom_pt_analytic/grid_vs_mesh_velocity.png)

Eulerian projection error at grid cell centres (raw P1 interpolation from 4-lvl mesh) compared to the analytic velocity at the same centres.

</div>
<div>

**Take-home.**

- **Uniform grid at the mesh's natural resolution ≈ 0 velocity error** — recirc = **0 %**, rot_cyl = **0.024 %**. The coarse grid preserves the mesh's P1 information essentially perfectly.
- **Block-refined grids introduce projection error** because they oversample the mesh's own resolution — recirc: **0.06 → 0.10 %**, rot_cyl: **1.04 → 0.68 %**. The finer the grid past the mesh, the more it "sees" the P1 gradient jumps.
- **This is the Eulerian half of the story** — the *velocity field itself* is essentially exact at coarse grids and degrades slightly at finer grids.

**Consequence.** For §5, the **coarsest grid** is what we want — accurate at zero cost.

</div>
</div>

---

<!-- _class: tight -->

## Slide 14 — Grid vs mesh · PT trajectory accuracy at step 2500

<div class="cols">
<div>

![PT step 2500](../paper_figs/rom_pt_analytic/grid_vs_mesh_pt_step2500.png)

Trajectory RMS error at step 2500 (t = 0.125 s scaled) — grid PT vs mesh (HCT-3D) per grid type, log scale.

</div>
<div>

**Take-home — case-dependent, timestep-dependent story.**

- **rot_cyl at step 2500**: grid is **~148× more accurate** than mesh+HCT at every refinement level (grid 8.4e-4 m vs mesh 0.124 m).  The mesh tracker is saturated by discretisation error near the cylinder wall; the grid trilinear captures it far better.
- **recirc at step 2500**: mesh+HCT is **~14× more accurate** than grid (grid 2.7e-3 m vs mesh 2.0e-4 m).  HCT's cubic recovery pays off on the smooth analytic field at mid-time; the grid trilinear can't match it.
- **Grid variants are essentially identical** — uniform ≈ 2lvl ≈ 4lvl at step 2500 for both cases.  Refining the grid past uniform gives no PT accuracy benefit.

At step 6000 the picture flips on recirc: grid becomes **~2× better** than mesh+HCT as the mesh's per-face gradient jumps accumulate.  So **the winner at step 2500 is not necessarily the winner at step 6000** — timescale matters.

</div>
</div>

---

<!-- _class: tight -->

## Slide 15 — When does HCT-3D matter? (final synthesis)

<div class="cols">
<div>

**Empirical evidence from four experiments:**

- **§3 FSW cohort** (Slide 19, HCT gap-closure): HCT changes ROM/FOM near-pin gap by ±0.9 %pt — negligible against the ~10–40 % ROM residual.
- **§7 recirc time series** (Slide 7): on tet-mesh PT with the smooth analytic field, HCT reduces final-step RMS **~6×** vs raw P1.  Clean positive validation.
- **§10 mesh→grid aggregate** (Slide 10): HCT is **7-13× worse than P1** on block-refined recirc grids finer than the mesh.
- **§11 region-stratified** (Slide 11): HCT wins in outer/smooth cells (1-2 orders of magnitude); loses near singularities and refined regions.

</div>
<div>

**When HCT-3D wins:**

1. **Query points are AS DENSE OR COARSER than the mesh** — HCT's cubic recovery smooths the P1 gradient jumps between tets.
2. **Field is smooth and has strong local curvature** (recirc's Gaussian × sin).
3. **Long trajectories on a tet mesh** — HCT correction is per-substep, compounds visibly over 6000 RK4 steps.

**When HCT-3D loses or is neutral:**

1. **Query points are FINER than the mesh** (block-refined grid over a coarse-in-the-outer-shell FEM mesh) — cubic reconstruction overshoots between P1 tet vertices.
2. **Field has singularities** (rot_cyl's 1/r² near-cylinder).
3. **When a different dominant error swamps it** (ROM/FOM Eulerian residual → ROM adds 10× more error than HCT ever removes).

**For §5 (grid PT), HCT is not applicable inside the tracker** (grid uses trilinear, not tet HCT).  The grid_uniform result on Slide 12 shows that grid PT already wins on trajectory error *without needing HCT-style recovery* — the mesh's P1 information collapses onto the coarse grid at zero loss.

</div>
</div>

---

## Slide 16 — Fig 1 · Near-pin dominates the ROM–FOM gap

<div class="cols">
<div>

![step-950 r-bin bars](../paper_figs/rom_pt_step2_step3/step950/fig1_step950_r_bin_bars.png)

</div>
<div>

**Read:** grouped bars per case;
red = near-pin (r ≤ 10 mm), blue = outer.
Solid = HCT-on, hatched = HCT-off.

- **Every case except 000** shows near-pin ~ 2× worse than outer.
- **Case 003 extreme**: 43 % near-pin vs 16 % outer.
- **HCT on/off makes almost no difference** at plot resolution.

</div>
</div>

---

## Slide 17 — Fig 2 · Eulerian → Lagrangian amplification

<div class="cols">
<div>

![step-950 amplification scatter](../paper_figs/rom_pt_step2_step3/step950/fig2_step950_eulerian_vs_lagrangian.png)

</div>
<div>

**X**: Eulerian rel_rms of the ROM reconstruction.
**Y**: Lagrangian PT rel_rms at step 950 (both-alive).
**Dotted**: y = x.

- **Every point sits above y = x** — trajectory error worse than velocity error.
- Case 003 amplifies **6.7×**. Case 001: 3.1×. Case 004: 3.6×. Case 000: 1.9×.
- **Small Eulerian error does not imply small Lagrangian error.**
- Direct empirical confirmation of the Xiong et al. 2023 POD-Lagrangian consistency-error phenomenon.

</div>
</div>

---

## Slide 18 — Fig 3 · Spatial residual (case 003 at step 500)

<div class="cols">
<div>

![spatial residual](../paper_figs/rom_pt_step2_step3/fig5_spatial_residual.png)

</div>
<div>

4-panel scatter, particle position coloured by ROM–FOM displacement magnitude. Dashed circle = r = 10 mm near-pin boundary.

- Residual is **not spatially random**.
- Case 003's coherent bright ring surrounds the pin cavity — exactly the shear-layer topology where FSW mixing physics lives.
- Case 000 shows a smaller localised residual; case 001/004 show intermediate patterns.

Strongest single visual for the roadmap §2 spatial-consistency-error prediction.

</div>
</div>

---

## Slide 19 — Fig 4 · HCT-3D is not compensating, not masking

<div class="cols">
<div>

![HCT gap closure](../paper_figs/rom_pt_step2_step3/step950/fig4_step950_hct_gap_closure.png)

</div>
<div>

Near-pin ROM–FOM rel_rms per case. Green = HCT-on, red = HCT-off.

- Δ (on − off) is between **−0.93 %pt and +0.42 %pt** across all cases.
- If HCT were **compensating**, Δ would be strongly negative.
- If HCT were **masking**, Δ ≈ 0 but with incoherent residual patterns.
- We observe Δ ≈ 0 **and** coherent residuals (Slide 18) — HCT and the ROM residual **live on different scales** (HCT within-element, ROM inter-element).

**HCT-3D is neither a fix nor a mask for the ROM's Eulerian residual.**

</div>
</div>

---

## Slide 20 — Fig 7 · Trapped particles — divergence-free proxy

<div class="cols">
<div>

![trapped particles](../paper_figs/rom_pt_step2_step3/step950/fig7_step950_trapped_particles.png)

</div>
<div>

At step 950, count particles with r ≤ 10 mm (still trapped near the tool). For an **incompressible field with proper outlet handling all bars should be ≈ 0**.

- **Case 000**: FOM traps **46 %** of seeded particles; ROM traps **30 %**.
- Cases 001, 003: FOM traps 4–5 %, ROM traps 0.3–0.4 %.
- **ROM traps fewer than FOM in every case.**

Two candidate interpretations:
- ROM smoothing damps small-scale shear that traps particles.
- ROM velocity reconstruction more div-free than FOM discretised velocity.

**§4 addresses which of these it is.**

</div>
</div>

---

## Slide 21 — §4 · Deeper look — is the ROM a smoother?

Roadmap §4 hypotheses:
- **H1 (smoothing)**: 3-mode POD damps high-k velocity structure, so tracker sees a smoothed flow.
- **H2 (coincidental)**: earlier "neater trajectories" were a loader-bug artefact.

**Test**: reconstruct v_FOM and v_ROM at ts = 119, subtract, characterise the residual per case with:
- radial and azimuthal |v| profiles
- azimuthal FFT spectra at multiple radii
- per-tet ∇·v distribution
- residence-time and pairwise-separation curves from the overnight mixing runs

Full analysis in `docs/rom_pt_smoothness_divfree_report.md`.

---

## Slide 22 — Fig B · Azimuthal |v| profiles per ring

<div class="cols">
<div>

![azimuthal profiles](../paper_figs/rom_pt_step4/figB_azimuthal_velocity_profiles.png)

</div>
<div>

Four rings per case: r = 6, 9, 12, 15 mm (inside pin / shear layer / just outside / bulk).

- **r = 6 mm (inside pin)**: ROM matches FOM within 1–2 % across all θ.
- **r = 12, 15 mm (workpiece)**: ROM **systematically 20–40 % higher than FOM** at every θ.
- Dropouts to zero at r = 9 mm come from level-set-clipped cells (real geometry, not artefact).

**Key finding: the ROM's error is a coherent mean bias in the workpiece, not a smoothing.**

</div>
</div>

---

## Slide 23 — Fig C · Azimuthal FFT spectra

<div class="cols">
<div>

![azimuthal FFT](../paper_figs/rom_pt_step4/figC_azimuthal_fft_spectra.png)

</div>
<div>

|FFT(|v|)| vs azimuthal wavenumber k.
Blue = FOM, red = ROM, green = residual.

- **r = 6 mm (inside pin)**: residual ~1 decade **below** FOM/ROM across all k → ROM captures every scale.
- **r = 12 mm (workpiece)**: residual **at** FOM/ROM level at k = 1–4 (large-scale bias) and **below** at k > 5 (small-scale captured).
- Direction is **opposite** to the smoothing prediction.

**H1 (smoothing) is rejected. The ROM misses the workpiece mean, not the high-frequency detail.**

</div>
</div>

---

## Slide 24 — Fig D · Is the ROM more divergence-free than FOM?

<div class="cols">
<div>

![divergence CCDF](../paper_figs/rom_pt_step4/figD_divergence_ccdf.png)

</div>
<div>

Per-tet |∇·v| complementary CDF, near-pin subset (r ≤ 10 mm). Log-log.

- **FOM curve and ROM curve overlap essentially perfectly** in every case.
- Cell-by-cell correlation between div_FOM and div_ROM is **0.99**.
- Near-pin |∇·v|_rms is 18–34 s⁻¹ across cases — the **FOM itself is not divergence-free** at the P1-tet level (FEMUSS stabilised P1/P1 mixed formulation artefact, ~0.3 % of natural gradient scale but not zero).

**The ROM inherits the FOM's divergence structure directly — it neither fixes nor worsens it.**

</div>
</div>

---

## Slide 25 — Fig E · New case-level predictor

<div class="cols">
<div>

![divergence vs PT gap](../paper_figs/rom_pt_step4/figE_divergence_vs_pt.png)

</div>
<div>

X: FOM near-pin |∇·v|_rms (Eulerian).
Y: ROM–FOM PT rel_rms at step 950 (Lagrangian).

- **Clean positive ordering.** Case 003 highest on both axes; cases 001/000/004 follow.
- **Stronger predictor than Eulerian velocity rel_rms** (where case 003 was *lowest* on X — Slide 10).

**A ROM whose reconstructed velocity is low-rel_rms but high-|∇·v|_rms will fail Lagrangian use.**

Candidate new §7 acceptance criterion.

</div>
</div>

---

<!-- _class: tight -->

## Slide 26 — Fig F · Trapping over time (linear vs log)

<div class="cols">
<div>

![residence time](../paper_figs/rom_pt_step4/figF_mixing_residence_time.png)

</div>
<div>

Particles inside annular probe (r ∈ [5, 10] mm) vs step.
Top row linear, bottom row log.

- **Linear**: ROM peaks 5–15 % lower than FOM and decays faster (consistent with ROM's overestimated workpiece velocity → faster transport).
- **Log** exposes the tail:

| case | FOM trapped | ROM trapped | ratio |
|---|---:|---:|---:|
| 000 | 14 606 | 1 016 | **14×** |
| 001 | 774 | 7 | **110×** |
| 003 | 2 714 | 1 260 | **2×** |
| 004 | 4 563 | 1 738 | **2.6×** |

**FOM keeps material recirculating near the tool. ROM releases it.**

</div>
</div>

---

<!-- _class: tight -->

## Slide 27 — Combined story so far

- ROM captures inside-pin structure at every scale (Fig B, r = 6 mm).
- ROM overestimates workpiece mean velocity by 20–40 % (Fig B, r = 12/15 mm).
- Residual is **low-k coherent** in the workpiece, not high-k noise (Fig C).
- ROM inherits FOM's divergence structure at 99 % correlation (Fig D).
- FOM's near-pin |∇·v|_rms is a better case-level PT-error predictor than Eulerian rel_rms (Fig E).
- FOM traps 2–110× more particles than ROM (Fig F / §2 Fig 7).
- HCT-3D is a small perturbation with no meaningful impact on any of the above.

**Reframed §4 diagnosis** — the ROM is a **biased estimator of workpiece velocity**, not a smoother:

- §5 uniform-grid projection **won't close the gap** — projection doesn't fix a coherent low-k mean bias.
- §6 block refinement of near-pin **won't close the gap either** — the bias is in the workpiece.
- Real fix: reconstruction that enforces the mean workpiece velocity as a constraint.

---

## Slide 28 — What's running now — overnight sweep

Launched 2026-07-17 evening, resumes with `SKIP_TRACKING` sentinels:

- **20-case ROM-vs-FOM sweep** — cases 2, 5–19 (16 new cases × 4 variants = 64 tracker runs)
- ~27 h total, will complete by 2026-07-19
- Uses `scripts/overnight_all20_cases.sh` (wrapper) which delegates to `scripts/overnight_rom_vs_fom_sweep.sh`
- Sentinel-based resume: any variant that already has `particles.vtkhdf` is skipped

**When it completes**:
- Re-run r-binning + figure builders → cohort-level scatter plots go from 4 to 20 points.
- Refresh §2/§3 report + §4 report from the extended data.
- Test whether the divergence-vs-PT predictor (Fig E) holds up on 20 cases.

---
## Slide 29 — What's next — §5 uniform-grid experiment

**Motivation (§4 reframed).** Uniform-grid projection removes mesh-interpolation error but **cannot fix a coherent low-k mean bias**. The rot_cyl_2026 decomposition on Slide 21 will tell us how much of the ROM/FOM gap that projection actually erases before we invest in §6 (block refinement) or a mean-corrected reconstruction.

**Runtime budget so far.**

| step | wall-clock | status |
|---|---:|---|
| §§2–4 for 4 cases | ~5.5 h | done |
| overnight all-20 sweep | ~27 h | running, resume-safe |
| rot_cyl_2026 (fresh params) | ~20 min | ready to launch |
| §5 uniform-grid on cohort | ~4 × cohort re-tracking (~30 h) | conditional on rot_cyl outcome |

**If rot_cyl_2026 says the tracker on div-free input is already at the observed ROM/FOM gap** → §5 will not close the gap; jump directly to a mean-corrected ROM reconstruction.
**If rot_cyl_2026 says the tracker is well below the gap** → §5 has room to work; run it on the cohort.

---

## Slide 30 — Recap · progress + solid findings

**Progress**

- §§1–4 complete for 4 cases, extending to 20 tonight.
- Comprehensive tooling in place: r-binning, spatial figs, divergence analysis, mixing diagnostics — all one-command-away for the extended cohort.
- Three self-documenting reports capture all findings.

**Solid findings**

- ROM–FOM Lagrangian gap grows with integration time and concentrates near-pin.
- Small Eulerian error ≠ small Lagrangian error (up to 6.7× amplification).
- HCT-3D delivers the expected ~6× tracker-error reduction on a known-good analytic field (recirc), but the improvement is too small to affect the ROM/FOM cohort gap.
- ROM does **not** smooth away high-k; ROM overestimates workpiece mean.
- FOM's own divergence structure predicts Lagrangian error better than the Eulerian rel_rms does.

---

## Slide 31 — Open questions + where to look

**Open questions for the team**

1. Is the workpiece bias intrinsic to 3-mode POD, or would 5–10 modes fix it? (cheap test)
2. Does the divergence-vs-PT predictor (Fig E) hold across 20 cases? (2 days away)
3. Does the analytic case show the tracker recovering the reference on div-free input? (tonight)
4. Should we invest in a POD reconstruction that enforces a workpiece-mean constraint?

**Deep-dive detail** — three reports in `docs/`

| section | file |
|---|---|
| §1 | `rom_reconstruction_findings.md` |
| §§2–3 | `rom_pt_step2_step3_report.md` |
| §4 | `rom_pt_smoothness_divfree_report.md` |
