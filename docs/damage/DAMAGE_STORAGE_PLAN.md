# Where damage lives — storage and projection plan

**2026-09-25.** Answers: store the damage scalars and the final defect indicator
per particle, or as a field over the domain? Then project to mesh or grid?

The short answer: **per particle is authoritative, everything else is derived** —
and almost all the machinery for the derivation already exists in this repo and is
currently unused.

---

## 1. Why per-particle is not a choice

Damage is a **history integral along a pathline**:

```
D(particle) = ∫ f(ε̇, η, σ₁/σ_eq, …) dt      along that particle's own path
```

A grid cell has no history. It has whatever particles happened to pass through it.
So the per-particle array is the only representation where the quantity is
*defined*; a field is a **reduction** of it, and which reduction you pick changes
the meaning.

This is not a storage-efficiency argument — it is what the quantity is. Any
pipeline that deposits first and integrates second is computing something else.

### Cost — it is not the constraint

| slot | quantity | phase |
|---|---|---|
| 0 | `ebar` = ∫ε̇ dt accumulated strain | 1 — **built, on the production kernel** |
| 1 | `ln Φ` Rice–Tracey (log space) | 3 |
| 2 | `C` Cockcroft–Latham | 3 |

12 bytes/particle → **4.3 MB at 360k particles**, one transfer at the end of the
run. Negligible against the 0.5 MB `edot` field and the mesh itself.

⚠️ Rice–Tracey is multiplicative, so slot 1 carries **ln Φ**, not Φ. Verified over
10,000 steps: log-space agrees with direct multiplication to **1.2e-13**, and
cannot underflow.

---

## 2. What already exists (checked, not assumed)

This is the part that changes the plan from "build a deposition pipeline" to
"wire up three existing pieces".

| piece | where | state |
|---|---|---|
| per-particle scalars into exported VTU/VTKHDF | `run_tracking.py` `extra_scalars` dict (already carries `Group`, `Temperature`, `MaxTemperature`, `Escaped`, `Density`) | ✅ works — add one key |
| swept-region union with **max-reduction** | `jaxtrace/density/union.py`, `reduce_max_fields=` on `dedup_batch` / `dedup_incremental` | ✅ implemented — **and called by nothing** |
| ROI restriction | same, `roi_lo` / `roi_hi` | ✅ |
| voxel-grid deposition, weighted | `jaxtrace/density/estimator.py`, per-particle `weights` | ✅ |
| post-processing driver + launchers | `run_density_postprocess.py`, `scripts/run_lumi_union.sh`, `scripts/run_workstation_density_union.sh` | ✅ LUMI **and** workstation |

> `reduce_max_fields` exists with the docstring "useful for *hottest temperature
> ever reached in this cell*". That is structurally identical to "worst damage any
> material passing through here endured". It was built for a different field and
> never used; damage is its first real consumer.

**Verified by test, not by reading.** Two particles into one cell with damage 1.0
and 9.0, a third elsewhere with 5.0:

| call | result |
|---|---|
| `reduce_max_fields=['Damage']` | `[5.0, 9.0]` — **max, correct** |
| no `reduce_max_fields` | `[1.0, 5.0]` — first-seen |

So the reduction choice genuinely changes the answer on identical input, which is
why §3 insists it be stated rather than defaulted.

Two API details found only by running it:
* the result attribute is **`point_data`**, not `fields`;
* `reduce_max_fields` handles 2-D arrays component-wise, so all three damage slots
  can be carried in one `(n, 3)` array in a single pass.

⚠️ **There is no per-cell count field yet.** `union.py:233` calls `np.unique(...)`
with `return_index` and `return_inverse` but **not `return_counts`** — which is the
count. Adding it is one argument plus one field in `UnionResult`, not new
machinery. §3 treats the count as mandatory, so this is a required (small) change,
not an optional extra.

⚠️ **The union is a post-processing step, not part of the tracking loop.** It reads
exported trajectories. That is the right shape here: it keeps the tracking loop
free of per-step host transfers (the constraint you set), and it means the
reduction can be changed and re-run without re-tracking.

---

## 3. The three representations

```
  (a) PER PARTICLE            (b) SWEPT-REGION FIELD        (c) CASE SCALAR
  carried on the pathline  →  union / voxel grid over     →  one number per case
  during tracking             the region particles visited    for the parameter map

  (n_particles, k)            union cloud, or (nx,ny,nz)     ()
  4.3 MB @ 360k               10–50 MB                       bytes
  the ONLY place the          what you plot and compare      what Phase 4
  history integral exists     between cases                  regresses on (v,ω)
```

### (a) → (b): which reduction, and why it matters

Three are meaningful and they answer different questions:

| reduction | meaning | use |
|---|---|---|
| **max** over particles in a cell | worst history any material passing here endured | **the defect-risk reading** — a void forms where *some* material failed, not where the average did |
| **mean** | typical exposure | smoother, better ROM regression target |
| **count** | how many particles visited | **coverage / trustworthiness** |

⚠️ **Carry the count alongside whichever reduction you report.** A cell visited by
3 particles and one visited by 300 are not comparable, and `max` over 3 samples is
noise. This is the same failure as the `n_selected` confound at case level
(ρ = +0.571, p = 0.008): a statistic that silently depends on sample density.

**Recommendation: max for the defect indicator, mean for the ROM target, count
always.** Store all three — they come from one pass.

### (b) → mesh or grid?

**Grid (voxel), not mesh.** Reasons, in order:

1. The deposition machinery is voxel-based and already exists.
2. Comparing cases requires a **common** support. The 20 PinShapes cases have
   different meshes; a voxel grid in tool-relative coordinates is shared.
3. Nodal projection would need a scatter onto an unstructured mesh with its own
   volume weighting — new code, and no question currently needs it.

⚠️ Mesh projection becomes worth doing only if damage has to be fed *back* into a
solver. Nothing in the current plan does that, so it is out of scope until it is.

---

## 4. The honest limitation — coverage

Deposition covers **only the swept region**. Material no seeded particle visited
has no damage value, and **that is not the same as zero damage.**

Consequences that must be handled rather than noted:

1. **Ship the coverage mask** (the `count` field) with every damage field. A cell
   with count 0 is *unknown*, not *safe*.
2. **Any case-level reduction must state the region it covers.** Comparing a case
   where particles swept 80 % of the stir zone against one that swept 40 % is the
   case-level version of the `n_selected` confound.
3. ⚠️ **Seed density and seeding region must be identical across compared cases.**
   Worth fixing now: it is cheap to standardise and expensive to discover later.
   The current gate uses `--seed-source box-frac --seed-fraction 0.0 0.2 …`, i.e.
   the upstream 20 % of X, which is an inflow seeding — fine for a throughput
   measurement, **not** validated as adequate stir-zone coverage.

**Open question for the cohort runs:** does inflow seeding actually populate the
advancing-side wake, where the tensile signature lives? Measurable directly from
the union's count field. Until measured, no case-level damage comparison should be
quoted.

---

## 5. Implementation — three steps, in order

### Step 1 — export damage per particle *(small)*

`run_tracking.py` already builds `step_extras` each export step. Add:

```python
if args.damage and damage_gpu is not None:
    step_extras['Damage'] = np.asarray(damage_gpu, dtype=np.float32)
```

⚠️ This *does* add a per-step device→host copy, so it must be gated on the export
interval (as `Density` and `Temperature` already are), not done every step. The
tracking loop's no-transfer property is preserved because exports are periodic.

The end-of-run `damage_<driver>.npz` (already written) stays as the authoritative
final state.

### Step 2 — union with max-reduction *(small — first user of an existing API)*

In the union post-processing call:

```python
dedup_batch(..., field_names=['Damage', 'ParticleID'],
            reduce_max_fields=['Damage'])
```

Gives the swept-region cloud with, per surviving cell, the **worst** damage of all
particles collapsed into it. `count` comes from the same pass.

### Step 3 — voxel grid for cross-case comparison *(moderate)*

Deposit the union cloud onto a fixed grid in **tool-relative** coordinates so the
20 PinShapes cases share a support. Reuse `DensityRunnerConfig` with
`bounds_mode="explicit"`.

⚠️ Tool-relative matters: cases differ in tool position and D-family has 2° tilt.
Depositing in lab coordinates would compare different physical regions.

### Sequencing against the phase plan

Steps 1–2 belong with **Phase 3** (when there are real damage ODEs worth
depositing, not just ∫ε̇ dt). Step 3 belongs with **Phase 4** (the parameter map).

⚠️ **Do not build the field before M1/M2.** A susceptibility field on a confounded
window inherits the confound — the same reason Phase 3 waits.

---

## 6. Summary

| question | answer |
|---|---|
| per particle or field? | **per particle is authoritative** — damage is a path integral; a field is a reduction |
| store what per particle? | `ebar` now; `ln Φ`, `C` in Phase 3. 12 B/particle |
| then project where? | **voxel grid, tool-relative**, via the existing union + density pipeline |
| which reduction? | **max** for defect risk, **mean** for the ROM target, **count always** |
| mesh projection? | not needed — only if damage feeds back into a solver |
| how much new code? | little: `reduce_max_fields` already exists and is unused; one `extra_scalars` key; grid bounds config |
| biggest risk | **coverage** — unvisited ≠ undamaged; identical seeding across compared cases is a prerequisite, not a detail |
