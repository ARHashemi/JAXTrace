# LUMI cases as reference benchmark — readiness audit

**2026-09-25.** What the 20 PinShapes cases + 22 FOM cohort cases can and cannot
support, what is already compatible, and what still has to be measured.

---

## 1. What the LUMI cases actually vary

Read from `data/*.som.dat` (solver input, authoritative) across all 20 PinShapes
cases:

| parameter | status | values |
|---|---|---|
| `som_rpm` | varies trivially | 996 (19 cases), 1000 (B2) |
| `som_tilt_deg` | **varies** | 0 (A,B,C) · 2 (D) |
| advancing speed | **constant** | 600 mm/min = 0.01 m/s |
| BC velocity (`som.fix`) | **constant** | 0.01 m/s |
| plate height | **constant** | 6 mm |
| steps/rev | **constant** | 150 |
| plunge, vertical load | **constant** | 0.0001, 0 |
| material | **constant** | Al6063, own Norton tables in every case |
| **pin geometry (STL)** | **the real variable** | A1.stl 414 KB … C1.stl 10.1 MB |

> ### ⚠️ Consequence for the benchmark
>
> **The PinShapes set is a pure tool-geometry study at a single operating
> point.** It cannot test the (v_adv, ω) parameter-space behaviour at all — those
> are fixed. Any safe-zone map over process parameters must come from the FOM
> cohort (which does vary ω = 400–800, v = 5–10 mm/s), not from PinShapes.
>
> The two sets answer different questions and must not be pooled.

✅ **O7 resolved for PinShapes:** every case ships its own `Al6063` `.mat` with
`Plastic_viscosity` / `Exponent_viscosity`. No borrowed tables. (The 22 FOM
cohort cases still borrow cylA's — O7 stands for them.)

---

## 2. ⚠️ A confound found while auditing: sampled volume predicts the answer

`n_selected` — how many nodes the survey window captures — spans **2,375 to
114,260 across cases with identical `r_tool` (7.0 mm)**, a 48× range.

| | Spearman ρ(n_selected, growth_ratio) | p |
|---|---|---|
| **all 20 cases** | **+0.571** | **0.008** |
| excluding the 3 sparse "old" cases | +0.431 | 0.084 |
| within family A (n=7) | −0.036 | 0.94 |
| within family B (n=4) | −0.200 | 0.80 |
| within family C (n=4) | −0.800 | 0.20 |
| within family D (n=5) | **+1.000** | **0.00** |

### ✅ Reproduced independently 2026-09-28

Re-extracted from the 20 per-case Stage-1 JSONs (`survey/n_selected` and
`survey/growth_ratio_adv_over_ret`) with `results/extract_confound.py`:

| | published | re-extracted |
|---|---|---|
| ρ(n_selected, growth_ratio), n=20 | +0.571 | **+0.571** ✅ |
| n_selected range | 2,375 – 114,260 (48×) | **2,375 – 114,260 (48×)** ✅ |
| within family A (n=7) | −0.036 | **−0.036** ✅ |
| within family B (n=4) | −0.200 | **−0.200** ✅ |
| within family C (n=4) | −0.800 | **−0.800** ✅ |
| within family D (n=5) | +1.000 | **+1.000** ✅ |

Every figure reproduces exactly. **The static-window result is confirmed and is
retained as the reference for team discussion** — the δ-relative work below is
reported *alongside* it, not as a replacement.

**Reading:** the cross-family correlation is largely because families differ in
*both* sampling density and ratio — C-Threads captures ~62k nodes and scores
high, the sparse A-old cases capture 2–8k and score mid. Within most families the
correlation vanishes or reverses, which argues against a pure artefact.

But it is **not excludable**, and family D shows ρ = +1.00 on n = 5.

⚠️ **The family separation in Figure 1 is therefore provisional.** It may be
partly a sampling-density effect rather than pure geometry.

### ✅ RESOLVED 2026-09-28 — it is geometry, not sampling density

A pin-anchored window (`r ∈ [r_pin, r_pin + 1.5(r50 − r_pin)]`, with `r_pin`
detected per case) removes the correlation:

| population | n | ρ static | ρ δ-window |
|---|---|---|---|
| **PinShapes pooled** | 19 | **+0.409** | **−0.219** |
| A-Flats | 7 | +0.179 | **0.000** |
| D-Concavity | 4 | +0.400 | **0.000** |

Mechanism, measured: `r_tool` = 7 mm is the **shoulder**; the pin surface sits at
**0.86–3.47 mm** depending on geometry. A window fixed at r/R ∈ [0.40, 0.70]
cannot follow it, and on A1 **72 %** of that window's nodes lie at r/R ≤ 0.45 with
only **10 %** beyond 0.50 — an effective thin annulus whose position moves with
pin shape.

**The family separation stands.** ⚠️ Absolute per-case δ ratios still need a
phase-averaged re-run (O27): single snapshots differ from 166-phase averages by a
median 3.8 % but up to **+48 %** on the sparse `old` cases.

**This is the same root cause as O10 and O11:** the window is defined in tool
geometry (`r/R` of the LEVEL<0 extent) while the physics lives in the shear
layer, and different pin shapes present very different amounts of material inside
the same annulus.

### The fix

Make the window **shear-layer-relative** and **volume-normalised**:

1. Measure δ per case (the `measure_shear_layer.py` tool already does this).
2. Define the window as `r ∈ [0.5 δ, 1.2 δ]` rather than `r/R ∈ [0.4, 0.7]`.
3. Report `n_selected` alongside every ratio, and check the confound again.

⚠️ **Do this before Phase 4** (the parameter map). Until then, quote the
family separation with the caveat attached.

---

## 3. What still has to be measured on the LUMI cases

| # | measurement | why | cost |
|---|---|---|---|
| M1 | **δ(case) for all 20 PinShapes** | the δ result is cylindrical-cohort only; tapered/threaded/tilted pins may differ (O16) | ~20 PVTU reads |
| M2 | **`n_selected` after a δ-relative window** | does the confound in §2 disappear? | re-run survey |
| M3 | **throughput A/B, damage on vs off** | Phase 1 gate, <3 % | 2 tracking runs |
| M4 | **pin geometry descriptors** (flat count/depth, thread lead, flute count) | the only real independent variable, and it is **not currently captured anywhere** | parse the STLs |

> ### M4 RUN 2026-09-30 — ⚠️ 9 of 20 cases cannot yield a pin envelope
>
> Ran as-is. Outcome worse than predicted (I said ~3 failures; it is **9**):
>
> | family | envelope OK | `n_lobes` / `lobe_depth` |
> |---|---|---|
> | **A-Flats** (7) | **0 of 7** ❌ | 0 lobes, depth ≤ 0.010 |
> | **B-Flutes** (4) | **4 of 4** ✅ | **3, 6, 4, 8** lobes at depth **0.20–0.32** |
> | **C-Threads** (4) | **4 of 4** ✅ | **2** lobes at depth 0.03–0.08 |
> | **D-Concavity** (5) | **2 of 5** (D3, D4) | D3: **8** @ 0.18; D4: 2 @ 0.04 |
>
> **Cause (diagnosed, not guessed):** the failing STLs are **coarse CAD
> tessellations**. A2 (2,794 triangles) has only **4 populated z-bins out of 60** —
> vertices exist at the pin tip (z = −5.61, r = 2.16), the shoulder underside
> (z = −0.08, r = 6.97) and two above. **There is no mid-pin surface to profile**, so
> `height`, `volume` and `hull_deficit` are not recoverable from the STL at any cut.
>
> ⚠️ **Not** a shared-geometry issue — each failing case has a distinct STL. (The only
> true duplicate is **D2 = D2old**, byte-identical.)
>
> **Two fix attempts, both rejected because they produced plausible wrong answers:**
>
> | attempt | result |
> |---|---|
> | relax the sparse-profile guard (`len(rp) < 6` → 2) | `r_pin` became **correct** (2.16–2.21 mm, matching M1) but `height` became **86 mm on a 5.4 mm pin** — the mask swept up stray shank vertices |
> | cut at the step midpoint instead of the bin centre | same failure |
>
> ⚠️ **Reverted to failing honestly.** Envelope columns are blanked, not filled with
> the shoulder radius. A gap in a table beats a number that looks usable and is not.
>
> ### What M4 delivered anyway — the part that matters
>
> `n_lobes` + `lobe_depth` **worked on all 20 cases** (they come from the *azimuthal*
> profile, which needs no z-resolution) and they separate the families cleanly:
>
> | family | lobes | depth | reading |
> |---|---|---|---|
> | A-Flats | **0** | ≤ 0.010 | ⚠️ flats are **too shallow to register** — see below |
> | B-Flutes | 3–8 | **0.20–0.32** | deep, few-lobed |
> | C-Threads | **2** | 0.03–0.08 | shallow, consistent |
> | D3 / D4 | 8 / 2 | 0.18 / 0.04 | mixed |
>
> ⚠️ **A-Flats reporting 0 lobes is itself informative but needs care.** Depth ≤ 1 %
> of mean radius is below the `MIN_LOBE_DEPTH` gate, so the detector correctly
> refuses to name a lobe count. Whether that is because the flats are genuinely
> shallow, or because a 2,794-triangle STL cannot resolve them, **is not
> distinguishable from these data.** Do not report "A-Flats have no lobes".
>
> ### Decision
>
> ✅ **Use `n_lobes` and `lobe_depth` for B, C and D only** — 13 cases with
> resolvable features. That is enough to make "deep flutes vs shallow threads" a
> number instead of a folder name.
>
> ⚠️ **To get A-Flats descriptors, the STLs must be re-tessellated from CAD** — this
> is a data limitation, not a code limitation, and no amount of post-processing
> fixes it. Worth one question: is a finer STL export available?
>
> ---
>
> ### (superseded) M4 status 2026-09-30 — ⚠️ PARTIAL. `measure_pin_geometry.py` is written and
> tested; **4 of 7 descriptors ship, 3 are withheld.**
>
> **✅ Ships in the CSV** (verified sane on B1/C1/D3, and `r_max` agrees with M1's
> independently measured `r_pin`):
>
> | descriptor | B1 | C1 | D3 |
> |---|---|---|---|
> | **`n_lobes`** | **3** flutes | **2** | **8** |
> | **`lobe_depth`** | **0.302** | 0.063 | 0.180 |
> | `r_max_mm` (pin) | 2.96 | 2.97 | 2.96 |
> | `height_mm` | 5.44 | 5.44 | 5.44 |
>
> plus `volume_mm3`, `area_mm2`, `sphericity`, `hull_deficit`, `n_triangles`,
> `stl_md5`, `pin_isolated`.
>
> **`n_lobes` + `lobe_depth` is the pair that answers M4's question** — "3 flutes at
> 30 % depth" vs "2 threads at 6 %" is a modellable number where "C-Threads" was only
> a folder name.
>
> **❌ Withheld** (computed, written only to the per-case JSON under an
> `unvalidated` key, never tabulated):
>
> | descriptor | why |
> |---|---|
> | `thread_lead_mm` | gave **5.35 mm for BOTH a fluted and a threaded pin** — half the mis-detected domain length, i.e. the FFT fundamental. After fixing the span: 777 mm. A 1-D r(z) profile averages the helix away; a real lead needs (θ,z) jointly. |
> | `concavity` | returns **+0.0000 for every case** — no information as written |
> | `taper_deg` | sign flipped between runs as the pin mask changed (C1: −2.7 → +8.1°) |
>
> ⚠️ **3 of 20 cases report `PIN ISOLATION FAILED`** (A1, A2, D1 — coarse STLs of
> 2.8k–11k triangles). Their envelope columns are **blanked, not filled with the
> shoulder radius**: a gap in a table beats a plausible wrong number.
> `n_lobes`/`lobe_depth` are still valid for them.
>
> **Decision needed — three options:**
>
> 1. **Run it as-is and use the 4 working descriptors.** ⚠️ Recommended. `n_lobes` and
>    `lobe_depth` are what Phase 4 needs to regress against; the withheld three were
>    never going to be the regressors. ~5 min.
> 2. **Fix the thread lead first** — a 2-D (θ,z) helix detector. Real work, and only
>    C-Threads (4 cases) would gain a descriptor they can already be identified by.
> 3. **Fix pin isolation for the 3 coarse STLs** — would complete the envelope columns
>    for A1/A2/D1. ⚠️ Worth it *if* A1 stays an outlier (O12), because its 8,280-triangle
>    STL vs A2–A4's ~2,790 is a candidate explanation for that outlier.
>
> ⚠️ **What M4 cannot do at all:** `D2` and `D2old` have **byte-identical STLs**, so they
> share a pin and are not independent geometry samples. Any geometry regression must
> account for that.
| M5 | **σ_eq route cross-check** (Norton vs Sellars–Tegart per case) | already have both; confirm ordering is rheology-independent on LUMI as it was on the workstation | re-run with `MODELS="norton sellars_tegart"` |

⚠️ **M4 is the significant gap.** The benchmark's independent variable is pin
geometry, and we currently record only `r_tool`. Without descriptors, "threads
score higher than flats" is an observation about four file names, not a
relationship anyone can model or extrapolate.

### Thresholds and tolerances — current status

| constant | value | basis | status |
|---|---|---|---|
| `EDOT_FLOOR` | 0.01 s⁻¹ | masks where σ_eq → 0 | ✅ fine — median ε̇ is 30–60 s⁻¹, so this only removes genuinely dead material |
| `ETA_CLAMP` | ±3 | prevents `exp` overflow | ✅ never reached on LUMI (max \|η\| = 0.96 on A2) |
| `R_LO_FRAC, R_HI_FRAC` | 0.40, 0.70 | of `r_tool` | ⚠️ **the confound in §2** — should become δ-relative |
| `Z_LO_FRAC, Z_HI_FRAC` | 0.0, 0.70 | of thickness | ⚠️ tuned on cylA (10 mm plate); LUMI plates are **6 mm** — re-check it still excludes the shoulder |
| `EDOT_MIN` (survey) | 1.0 s⁻¹ | active material only | ✅ fine |
| phase probe | 8 phases | matches 166-step revolution to 0.021 | ✅ **validated on LUMI** |
| σ_eq model | Norton | FOM's own tables | ✅ every PinShapes case has its own |

⚠️ **`Z_HI_FRAC = 0.70` on a 6 mm plate puts the window top at z = −1.8 mm.**
The tool z-extent should be checked per family: if the shoulder reaches deeper
than 1.8 mm on these cases, the window is contaminated exactly as it was on cylA.

---

## 4. Storage plan — where damage lives

### The three representations, and why all three exist

```
  (a) PER-PARTICLE          (b) DEPOSITED FIELD        (c) CASE SCALAR
  carried on the pathline → on a grid over the      → one number per case
  during tracking            swept region              for the parameter map

  shape (n_particles, k)     shape (nx, ny, nz)        shape ()
  ~4 MB at 360k particles    ~10-50 MB                 bytes
  the ONLY place the         what you plot and          what the ROM in
  history integral exists    compare between cases      Phase 4 regresses
```

**(a) is authoritative.** Damage is a *history integral along a pathline* — it
only has meaning per particle. A grid cell has no history; it has whatever
particles happened to pass through it.

**(b) is derived by deposition**, reusing the existing union/density machinery.
Each particle deposits its accumulated scalars into the cells it visited. Two
reductions are meaningful and different:

| reduction | meaning |
|---|---|
| **max** over particles in a cell | worst history any material passing here endured — the defect-risk reading |
| **mean** over particles | typical exposure — smoother, better ROM target |
| **count** | coverage; a cell with 3 particles is not comparable to one with 300 |

⚠️ **Carry the count.** Without it, a sparsely-visited cell's max is noise, and
that is exactly the confound §2 shows at case level.

**(c) is a reduction of (b)** over the region of interest, which is what Phase 4
regresses against (v_adv, ω).

### What gets carried per particle

| slot | quantity | dtype | notes |
|---|---|---|---|
| 0 | `ebar` accumulated strain ∫ε̇ dt | f32 | Phase 1 — **built** |
| 1 | `ln Φ` Rice–Tracey | f32 | log space (stability verified to 1.2e-13 over 10k steps) |
| 2 | `C` Cockcroft–Latham | f32 | |

12 bytes/particle → **4.3 MB at 360k particles**. Negligible.

### Domain coverage — the honest limitation

Deposition only covers the **swept region**. Material never visited by a seeded
particle has no damage value, and that is not the same as zero damage.

So the field must ship with its coverage mask, and any case-level reduction must
state the region it covers. Comparing a case where particles swept 80 % of the
stir zone against one where they swept 40 % is the case-level version of the §2
confound.

**Practical consequence:** seed density and seeding region must be identical
across cases being compared. Worth fixing now rather than discovering later.

---

## 5. Recommended order

1. **M3 throughput gate** — closes Phase 1, cheapest, unblocks everything.
2. **M1 + M2** — δ per case, then a δ-relative window; resolves the §2 confound,
   O10, O11-for-PinShapes and O16 together.
3. ~~**M4 geometry descriptors**~~ → **run 2026-09-30, partial by a data limit.**
   `n_lobes`/`lobe_depth` work on all 20 cases and separate the families
   (B-Flutes 3–8 lobes @ 0.20–0.32 depth; C-Threads 2 @ 0.03–0.08). ⚠️ The pin
   **envelope** fails on 9 of 20 — coarse CAD tessellation, not a code fault; A2 has
   4 populated z-bins of 60. **Usable: 13 cases (B, C, D).** Finer STL exports would
   fix it.
4. **M5 rheology cross-check** — cheap, and it either confirms robustness or
   finds something.

Phases 3–4 of the implementation plan should wait for M1–M2, because a
susceptibility field built on a confounded window inherits the confound.
