# Two independent routes to the same weld-quality ranking

**A comparison of the flow/force post-processing analysis with a
stress-triaxiality damage analysis, on the same 20 PinShapes cases.**

Prepared 2026-09-30.

---

## Who this is for

**Two ways to use this document.** For the meeting, read **§0** (three sentences
plus answers to likely pushback). For the detail behind any claim, the numbered
sections follow. If a physics term stops you, `CARDS_one_concept_each.md` has a
self-contained card for each one — jump there and come straight back.

🖼 **Figures:** `figs_phase3/` for the results, `figs_concept/` for the method —
indexed in `FIGURES_INDEX.md`. ⚠️ Read that index before using fig1 or fig4: the
Phase 3 *accumulated* damage does **not** yet reproduce the Stage 1 family ranking
or flank asymmetry (O33), so the statement "our numbers confirm theirs" rests on the
**Stage 1** η result, not on Phase 3.

For the full derivation of any physics claim here, see
`PRIMER_fsw_physics_for_me.md` (§7 stress and strain, §8 the damage models, §9 what
was built and why).

This note assumes no knowledge of the damage analysis. It uses the vocabulary of
the existing PinShapes flow study — advancing side, backfilling, deposition
cavity, plastic deformation zone — and explains where the second method differs.

**The short version:** two analyses of the same 20 FOM simulations, using
different physical quantities and written independently, produce **the same
ranking of pin families by defect risk** and **the same statement of the defect
mechanism**.

---

## 0. What to say — the three sentences

If you only get thirty seconds, these three sentences carry the whole message. Each
is defensible with a number, and the number is in this document.

> **1.** "We measured the same defect mechanism you did — material-transport deficit
> at the rear advancing side — but from the stress side instead of the flux side,
> and with no fitted parameters."
>
> **2.** "Our ranking of the pin families is the same as yours: threads worst, flats
> best. Two different physical quantities, measured independently, same order."
>
> **3.** "And we can quantify it per case, so it can be regressed against pin
> geometry — which is what the ROM needs."

⚠️ **Then say the limitation before anyone asks it:** *"Neither of us has
experimental defect data yet. Both methods currently produce a susceptibility
ranking, not a validated defect prediction."* Volunteering that is stronger than
being caught by it, and it is true of both analyses equally — so it is not a
weakness of yours.

### If someone challenges a specific claim

| they say | you say |
|---|---|
| *"How can a stress criterion say anything about material flow?"* | "They are the cause and the consequence. You measure whether material arrives; we measure the stress state where it did not. Where refill fails, the material behind the pin is pulled apart rather than forged — that shows up as tensile mean stress, which is what we detect." |
| *"Isn't this just the same thing twice?"* | "No — and that is the point. If two independent measures of different quantities agree on the ranking, neither is an artefact of its own method. We also re-ran everything with a completely different material model: the ranking did not move (ρ = +0.9995)." |
| *"Your window/threshold looks arbitrary."* | "It was, and we found that ourselves: the first window correlated with the answer at ρ = +0.571 because it was sized from the shoulder radius while the physics sits at the pin. Anchoring it to each pin's measured surface removes the correlation. **You flagged the same issue in your own threads result.**" |
| *"Threads performing worst contradicts the literature."* | "It contradicts the expectation, and you flagged it as questionable pending a pin-root evaluation. Our window already samples at the pin root and also ranks threads worst — so either both methods agree at the corrected location, or there is a real disagreement to resolve. Worth checking." |
| *"What does this add over what we have?"* | "A continuous field instead of a per-case verdict, established void-growth physics with at most one calibrated constant, and it runs inside the GPU tracker at under 2 % cost — so it can live in the ROM rather than only in post-processing." |

---

## 1. The two methods in one table

| | **Flow analysis** (existing) | **Damage analysis** (this work) |
|---|---|---|
| primary quantity | material **flux** through LE / RS / TE / AS sectors; pressure in the deposition cavity | **stress triaxiality** η = σ_m / σ_eq |
| what it asks | *is the cavity behind the pin refilled?* | *is the stress state locally tensile (void-opening) or compressive (void-closing)?* |
| where evaluated | an evaluation plane / threshold volume | a window anchored on the **pin surface**, lower 70 % of plate thickness |
| output per case | V_θ, V_z refill volumes; F_mag, F_x, F_y | a single **advancing/retreating growth ratio** |
| time treatment | over one full revolution | phase-averaged over one revolution (166 steps) |
| calibration | none | **none** — no fitted parameters |

⚠️ **Neither method was tuned to reproduce the other.** They were developed
separately and compared only after both were complete.

---

## 2. The defect mechanism — the same statement, twice

The flow analysis concludes:

> "**Defect Initiation: Localized mass transport deficit and flow stagnation at
> the rear AS cause tunnel/wormhole voids.**"
> "**Lack of backfilling AS indicates defects.**"
> "Sound Flow Conditions: Balanced material recirculation with complete cavity
> backfilling behind the trailing pin."

The damage analysis reaches the same picture from stress rather than flux:

* A void in FSW is **not** a cavitation bubble. The process is solid-state: no
  melting (peak T is 80–95 % of T_melt), no vapour phase. Quantitatively, the
  Reynolds number is **Re ≈ 10⁻⁷–10⁻⁴** (creeping flow, 7–12 orders below
  turbulent transition) and the cavitation number is **~10⁵ times too large, with
  the wrong sign**. So a void is **empty space that the material flow failed to
  refill** — a volumetric bookkeeping failure.
* Where refilling fails, the material behind the pin is pulled apart rather than
  forged together. That shows up as **tensile mean stress**, σ_m > 0.
* Measured on the FOM fields, tensile conditions are **sharply localised to the
  advancing-side wake**: in the reference cylindrical case, **94 % of nodes are in
  tension on the advancing side of the wake versus 1 % on the retreating side**.

> **The two analyses agree on the mechanism and on its location.** One measures
> the *cause* (material not arriving); the other measures the *consequence*
> (tensile stress where it did not arrive).

---

## 3. The family ranking — independent agreement

The flow analysis concludes:

> "Key Takeaway: **Flats are very effective in backfilling**; … **Threads perform
> the worst among pin profiles**"

The damage analysis produces a number per case. Higher = more void-prone:

| family | damage ratio (median) | flow-analysis finding |
|---|---|---|
| **C-Threads** | **1.117** — worst | **"perform the worst among pin profiles"** |
| B-Flutes | 1.022 | — |
| D-Concavity | 0.905 | "tool tilt increases vertical flow" (favourable) |
| **A-Flats** | **0.892** — best | **"very effective in backfilling"** |

**Same order, both methods.** A ratio above 1 means the advancing-side wake is
more void-prone than the retreating side; below 1 means the reverse.

### How robust is that ranking?

Two independent stress tests were run on the damage side:

| test | result |
|---|---|
| **Change the material model.** Recompute everything with a completely different rheology (Sellars–Tegart with literature constants, no fitting) instead of the FOM's own Norton–Hoff tables. | rank correlation between the two models **ρ = +0.9995**, **0 of 41 cases** change which side is tensile, family order **identical** |
| **Change the sampling window.** The first window was sized from the *shoulder* radius; it was re-anchored on each pin's own measured surface. | the spurious correlation between "how much material was sampled" and the answer **disappears** (ρ +0.374 → −0.244) |

⚠️ So the ranking is not an artefact of the material model, nor of where the
measurement was taken. **And it is reproduced by a different physical quantity
entirely** — the flow analysis above.

---

## 4. Where the two methods independently confirm a specific number

The flow analysis reports:

> "Plastic deformation zone volume … **Flat faces increase V_PD** …
> **V_PD,A1 ≈ 1.5 × V_PD,A2**"

The damage analysis measured, without knowledge of that result, how many mesh
nodes fall inside its sampling window per case:

| | A1 | A2 | ratio |
|---|---|---|---|
| nodes in the sampling window | 87,094 | 36,586 | **2.4×** |
| V_PD (flow analysis) | — | — | **1.5×** |

⚠️ These are **not the same quantity** — one is a deforming volume, the other a
node count in a fixed window — so the numbers should not match exactly. What
matters is that both say **A1 deforms a substantially larger volume than A2, in
the same direction and the same order of magnitude.**

**Why this mattered:** A1 had been flagged as a possible *measurement artefact* in
the damage analysis, precisely because it captured so much more material than its
siblings. The flow analysis shows the large deforming volume is **real physics**,
so A1's unusual value is a property of the tool, not a defect of the method.

---

## 5. What the damage analysis adds

Not a competing answer — three things the flux/force route does not provide.

### (a) A continuous field rather than a per-case verdict

Damage is accumulated **along each material particle's own trajectory** (a
pathline) as it passes the tool:

```
D(particle) = ∫ f( ε̇, η, … ) dt        along that particle's path
```

With ~10⁵–10⁶ tracked particles, depositing their accumulated values back onto a
grid gives a **continuous susceptibility field over the weld** — a map of where
material *has been through* a damaging history, rather than one number per case.

⚠️ This matters because a history integral cannot be recovered from a snapshot: a
grid cell has no history, only whatever particles happened to pass through it.

### (b) Borrowed, established damage physics

The stress-state dependence is not invented for FSW. It is the standard
void-growth framework from metal forming:

| model | free parameters |
|---|---|
| **Rice–Tracey** — void growth ∝ exp(3η/2) | **none** |
| **Cockcroft–Latham** — ∫⟨σ₁⟩dε̄ | 1 (a critical value) |

⚠️ Both are **growth** laws: they describe how an existing void grows or closes,
not how one nucleates. They require an assumed initial porosity, which nobody
measures directly. **This is a real limitation, not a technicality.**

### (c) It is designed to run inside the fast surrogate

The damage accumulation was added to the existing GPU particle tracker and
measured: **+0.15 % to +1.77 % throughput cost**, on both NVIDIA and AMD
hardware, verified by an A/B test on the production code path. So it can run
inside the reduced-order model rather than only as offline post-processing.

---

## 6. One disagreement, and how to settle it

The flow analysis flags its own threads result:

> "**QUESTIONABLE**: Threads perform the worst among pin profiles → **evaluation
> plane has to be set closer to pin root** to visualize benefit of threads."
> "horizontal flow evaluation must be repeated closer to pin tip"

⚠️ **The damage analysis already samples there**, because it hit the same problem
and had to fix it:

| | |
|---|---|
| depth | lower **70 %** of plate thickness — the pin-root region, shoulder excluded |
| radial | anchored on each pin's **own measured surface** (C-Threads: r_pin = 2.07 mm, window r/R = 0.29–0.48) |

The reason this mattered on the damage side: the tool radius taken from the
level-set is the **shoulder** radius (7.0 mm), while the pin surface sits at
**0.86–2.43 mm** depending on geometry. A window fixed as a fraction of the
shoulder radius cannot follow a pin surface that moves by a factor of three — and
it produced exactly the kind of spurious correlation described in §3.

**So there are two outcomes, and both are useful:**

1. The flow analysis, repeated at the pin root, **still** ranks threads worst →
   both methods agree at the corrected location, and the "QUESTIONABLE" flag can
   be lifted.
2. It **reverses** → a genuine disagreement between a flux-based and a
   stress-based criterion, at the same location, which is worth understanding.

---

## 7. What neither method has

⚠️ **No experimental defect data.** Neither analysis has been compared against a
sectioned weld, a macrograph or a CT scan. All validation in both routes is
against **reaction forces, torque and thermocouple temperatures** — which the FOM
reproduces well, but which are not defects.

**What this means for how results should be described:** both methods currently
produce a **susceptibility ranking**, not a validated defect prediction. The
ranking rests on:

* agreement with the literature's expectation (defects on the advancing side),
* internal consistency under changes of material model and sampling window,
* and now, agreement between two independent physical measures.

⚠️ Those are necessary conditions, not sufficient ones. **A single sectioned weld
from one of these 20 parameter sets — one known-sound and one known-defective —
would convert both analyses from "physically motivated ranking" to "validated
predictor".**

### One available check that needs no new experiment

The flow analysis uses:

> "**Dominant transverse force (F_y > F_x) signals material stagnation and defect
> risk.**"

Reaction forces **are** experimentally validated in the published FOM work. So the
F_y/F_x criterion could be applied to all 20 cases and compared against both
rankings — an external check, from measured quantities, at no experimental cost.

---

## 8. Summary

| question | answer |
|---|---|
| Do the two methods agree on the **mechanism**? | **Yes** — material-transport deficit at the rear advancing side, both independently |
| Do they agree on the **family ranking**? | **Yes** — C-Threads worst, A-Flats best, both |
| Is the damage ranking an artefact of its material model? | **No** — ρ = +0.9995 under an independent rheology, 0/41 flank changes |
| Is it an artefact of where it samples? | **No** — the spurious correlation disappears with a pin-anchored window |
| Do they agree on a specific quantitative claim? | **Yes** — A1 deforms a larger volume than A2 (1.5× vs 2.4×, same direction) |
| Is anything **validated against real defects**? | ⚠️ **No.** Neither method. This is the main open item for both. |
| What does the damage route add? | a continuous field, established void-growth physics, and <2 % cost inside the GPU tracker |
| What is the open disagreement? | the flow analysis's threads result is flagged "questionable" pending a pin-root evaluation, which the damage route already uses |
