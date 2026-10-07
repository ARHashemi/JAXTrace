# Figures — what each one shows, and what it is for

Two sets. **Concept figures** explain what the quantities mean; **result figures**
show what the 45 cases measured. Each is PNG (for slides) and SVG (for documents).

Regenerate: `python3 make_concept_figures.py` · `python3 make_phase3_figures.py`

---

## Concept figures — `figs_concept/`

For understanding the method. Schematic, except C4 which uses the real tool STLs.

| figure | what it shows | the point |
|---|---|---|
| **C1 geometry** | tool + plate, plan and section | where **advancing/retreating** are, and that `r_tool` from LEVEL<0 is the **shoulder (7 mm)**, not the pin (2.4 mm) |
| **C2 stress split** | σ = volume part + shape part | metal **flows** from the shape part, voids **open/close** from the volume part — which is why their ratio η is the variable |
| **C3 triaxiality** | the η number line | η<0 closes voids, η>0 opens them, and the dependence is **exponential** |
| **C4 pin shapes** | the four families, **from the real STLs** | the only independent variable in the PinShapes study |
| **C5 pathline** | one particle's route + its accumulating lnΦ | damage is a **history**, which is why we track particles and not a grid |

⚠️ **C4 labels A-Flats as "STL too coarse to show the pin surface"** — that is not a
rendering failure. A1 has 7,560 vertices all at the pin tip and **none** between
z = −5 and −1 mm; the side surface is simply not tessellated. The same limitation
made M4's envelope descriptors fail on 9 of 20 cases.

---

## Result figures — `figs_phase3/`

From `results/phase3_summary.csv`, reduced from the 45 `damage_models.npz` files.

| figure | what it shows |
|---|---|
| **fig1 family ranking** | accumulated lnΦ per case, grouped by pin family |
| **fig2 models agree** | Rice–Tracey vs Cockcroft–Latham, and the rank-correlation histogram |
| **fig3 radial profile** | where damage sits relative to the pin and shoulder |
| **fig4 advancing wake** | the two wake flanks, each case against its **own** advancing side |
| **fig5 accumulation** | fraction of particles accumulating — the D-family anomaly |

### ⚠️ What fig1 and fig4 actually say — read before presenting them

**fig2 is the good news.** The two damage models rank particles identically on all
45 cases (ρ = 0.982–0.998). The implementation is self-consistent.

**fig1 and fig4 contradict the Stage-1 result**, and the figures say so:

| | Stage 1 (nodal η) | Phase 3 (pathline lnΦ) |
|---|---|---|
| advancing side | **94 %** tensile vs 1 % | **LOWER in 45/45 cases** |
| family ranking | 1.117 / 1.022 / 0.905 / 0.892 | A, B, C all ≈ 0.065 — **indistinguishable** |

Both were checked for plotting errors first. The flank assignment is correct
(PinShapes CCW ⇒ advancing = +y; cohort CW ⇒ −y, matching the measured
`advancing_y_sign`), and no data is missing.

⚠️ **I tested the obvious explanation and it failed.** If advancing-side particles
simply passed through faster, they should end farther out. Measured on A1:
advancing median radius **9.49 mm** vs retreating **9.86 mm** — they ended *closer
in*. The residence-time story is wrong and I have no replacement. **O33.**

**So the defensible statements today are:**

* ✅ the two damage models agree with each other;
* ✅ Stage 1's nodal η result stands on its own, and is what agrees with the
  colleague's flow analysis;
* ⚠️ Phase 3's **accumulated** damage does not yet reproduce the flank asymmetry or
  the family ranking, and **why is unresolved**.

**Do not present fig1 as confirming the Stage-1 family ranking.**

### The most likely candidate, untested

The Phase 3 runs used **10 mm travel** with seeding in the upstream 30 % of X.
Particles may pass *around* the pin rather than *through* the shear layer, so the
integral is dominated by the long low-damage approach. That would dilute both the
flank contrast and the family contrast — exactly the two symptoms. Cheap to test:
re-run one case seeded near the pin, or weight the integral by time inside the
shear layer.

---

## Where the underlying data is

| | |
|---|---|
| per-case summary (45 rows) | `results/phase3_summary.csv` |
| full per-particle arrays | `/home/arhashemi/lumi/lumi_scratch/hashemia/damage/phase3_overnight_20260930/*/damage_models.npz` (175 MB) |
| pin descriptors | `results/pin_geometry.csv` |
| δ / window measurements | `results/delta_window_45.csv` |

⚠️ With `--export-format vtu` a run now also writes `Damage_lnPhi` and `Damage_CL`
as particle PointData, so pathlines can be coloured by damage directly in ParaView.
The 45 completed runs used `--no-export`, so re-run a case with export on if you
want that.


---

## Phase 4 (30 mm, RK4 damage) — added 2026-10-02

⚠️ **The Phase 3 figures in `figs_phase3/` are superseded.** They were produced at
10 mm travel, where the particle bundle had not yet cleared the tool (PinShapes mean
displacement **-0.52 mm**). Use `figs_phase4/` for anything presented.

| figure | what it shows | watch out for |
|---|---|---|
| `figs_phase4/fig2_models_agree.png` | Rice-Tracey vs Cockcroft-Latham, all 45 cases | rank corr +0.982..+0.998 -- this is the implementation self-check |
| `figs_phase4/fig3_radial_profile.png` | median lnPhi vs distance from tool axis | at 30 mm only 2.6 % of particles remain inside r < 7 mm |
| `figs_phase4/fig4_advancing_wake.png` | each case against its OWN advancing side | mixes PinShapes and cohort, which behave differently -- prefer the split table in the guide |
| `figs_phase4/fig5_accumulation.png` | fraction of particles accumulating | D-Concavity is low partly because fewer of its particles reach the wake |

### Wake box (`figs_wake/`)

| figure | what it shows |
|---|---|
| `figWB_<family>_yz_field.png` | yz cross-section of the wake, 16-NN median of lnPhi at 0.05 mm resolution |
| `figW_<family>_yz.png` | the same thing as a raw scatter -- shows the Poisson speckle the field version removes |

⚠️ **Equal y and z scales.** The weld is 30 mm wide and 5.5 mm deep; an auto aspect
ratio turns a shallow band into a false cone. Earlier versions of this figure had
that defect.

⚠️ **Not a histogram.** At 0.18 mm mean particle spacing a 600x240 bin grid leaves
**54 % of cells empty**, so no bin size is both fine and populated. The k-nearest-
neighbour median decouples display resolution from cell occupancy.

### Trajectories (`figs_traj/`)

| figure | what it shows |
|---|---|
| `figT_<case>_3d.png` | top (x-y) and side (x-z) views, true scale, coloured by damage accumulated so far |
| `figT_<case>_time.png` | lnPhi vs step, paired with the same particles' distance from the tool axis |

⚠️ **4 strata x 3 particles, deliberately not a random sample.** A mechanism
illustration, not a population statistic -- the population numbers are in
`results/phase4_rk4_30mm_wake.csv`.

### Pending

| figure | why it is worth making |
|---|---|
| `figT_ps_B-FluteVariations_B2_*` | does B capture MORE OFTEN? would give the family ranking a geometric cause |
| `figT_ps_D-ConcavityTilt_D1_*` | are D's particles captured at all, or diverted by the tilt? |
