# The vertical lines of stopped particles on the pin wall

**Not a search failure, and not a defect.** They are particles that crossed
microscopically into the tool, had their velocity forced to zero by the
level-set mask, and can never leave — because a zero velocity cannot transport
them back out.

Investigated 2026-10-15 on the LUMI D2 run (job 22815786, 288,000 particles,
8000 steps, final loss 3,044 = 1.06%).

---

## 1. The reported observation

Two z-aligned lines of stationary particles on the pin wall, with the rest of
the result acceptable.

## 2. The search-failure hypothesis is ruled out

The natural first guess is that those particles lost their host element. The
exported `ElementID` settles it directly:

| | wedge (−160…−140°, the line) | rest of the pin band |
|---|---:|---:|
| count | **3,087** | 4,139 |
| `ElementID < 0` (search failed) | **0 (0.00%)** | 0 (0.00%) |
| `ElementID >= 0` (hosted) | 3,087 | 4,139 |
| `escaped_ever` set | **0** | 0 |
| distinct host elements | **792** | 3,091 |

Not one has a failed search, none ever escaped, and they occupy 792 distinct
elements with the top ten holding only 10.6% — so they are not piled into a
few pathological cells either.

Separately, **all 3,044 genuinely lost particles sit at r 7.02–19.15 mm,
z 0.00–0.24 mm** — at or outside the shoulder edge on the top surface. **Zero
lost particles are anywhere near the pin** (r 1.5–4.0 mm: 0). The two
populations do not overlap at all.

## 3. What is actually happening

Nearest-node `LEVEL` and nodal velocity at the particle positions, against two
controls:

| | LEVEL median | inside tool (LEVEL<0) | nodal \|v\| median | T median |
|---|---:|---:|---:|---:|
| **wedge (the line)** | **−0.000008** | **67.2%** | **0.258 m/s** | **439.4** |
| rest of the pin band | +0.000961 | 5.5% | 0.146 m/s | 510.8 |
| control, r > 12 mm | +0.011048 | 0.0% | 0.010 m/s | 315.8 |

Two facts settle the mechanism:

1. **67.2% of the wedge particles are inside the tool**, against 5.5% for
   their immediate neighbours — a 12× enrichment. Their median `LEVEL` is
   −8e-06, i.e. *microscopically* inside the pin wall. They straddle the
   surface.
2. **The fluid there is not slow.** Nodal |v| is 0.258 m/s, the *highest* of
   the three groups — the pin wall is where rotational speed is greatest — and
   0.0% of them are near a stagnation point. So this is not a physical
   stagnation zone.

The velocity they *receive* is nevertheless zero, because of the mask
(`rk4_fully_fused_timedep.py`, the `interpolate_velocity_single` body):

```python
ls_val = b0*node_ls[0] + b1*node_ls[1] + b2*node_ls[2] + b3*node_ls[3]
vel = jnp.where(ls_val >= 0.0, vel, jnp.zeros(3, ...))
```

**A particle that drifts a few microns inside the pin gets zero velocity, and
zero velocity cannot carry it back out.** It is parked permanently. The
temperature confirms the sequence: 439 K against 511 K for its neighbours —
it stopped accumulating path history the moment it was masked.

### Why two lines rather than a cylindrical shell

The accumulation is azimuthally concentrated: 2,432 of 7,226 pin-band
particles in a single 10° sector (−160…−150°) and 655 in the adjacent one,
~43% in a 20° wedge spanning the full pin height (z −5.7…−0.3 mm), against
~180 per sector elsewhere.

The tool rotates at −600 RPM, so its surface sweeps every azimuth ten times a
second; a *fixed* angular accumulation therefore cannot be a feature of the
tool. It must be fixed in the lab frame, and −160…−140° is the trailing/wake
region given `v_adv` along +x. Particles arriving from the seed box (which
sits at x −19.9…−11.1 mm, behind the tool) reach the pin from that side, touch
the wall, and stick where they first make contact.

## 4. Is it a bug?

**No — it is faithful to the reference implementation.** `config.py` records
that `RK4_LEVELSET_MODE = 'zero_vel'` is deliberately FEMUSS-equivalent:

> FEMUSS reference (`Mod_ParticleTracer.f90`, line 1646):
> `if (levelSetValue < 0.0_rp) AddFluidInteraction = .false.`

So FEMUSS parks these particles too. Reproducing that is the default's
purpose.

## 5. Options, if the lines are unwanted

None of these is a correctness fix; they are modelling choices.

| option | effect | cost |
|---|---|---|
| **leave it** | matches FEMUSS exactly | the lines remain; they are 1.1% of particles and visibly artefactual |
| **`--levelset-mode skip_step`** | a particle whose substage enters the tool has the *whole step* discarded, so it stays at its last outside position instead of being zeroed | already implemented; more aggressive than FEMUSS, and it still parks the particle, just outside the wall rather than inside |
| **reflect or project out of the tool** | on detecting `LEVEL<0`, push the particle back along the level-set gradient to the surface and continue | not implemented; needs a gradient and a step-back rule, and departs from FEMUSS |
| **`RK4_L0_SKIP_BOUNDARY_ELEMENTS`** (already `True`) | forces a fresh search at boundary elements so the mask is evaluated against the correct element | already on; it makes the masking *more* faithful, not less frequent |

**Recommendation: leave the default.** The parked particles are a known
consequence of the FEMUSS masking rule, they are confined to the tool surface,
they are 1.1% of the population, and they are trivially identifiable in
post-processing (inside the tool, `LEVEL < 0`, temperature below their
neighbours). Filtering them at visualisation time is cheaper and more honest
than changing the physics. If they must not exist at all, reflection (option 3)
is the only option that genuinely frees them, and it should be a documented
deviation from FEMUSS rather than a silent default.

## 6. Reproducing

- `scripts/dfamily/wedge_check.py` — ElementID / escaped / host-element statistics
  from `final_state.npz` alone, no mesh needed.
- `scripts/dfamily/wedge_phys.py` — nearest-node `LEVEL` and nodal velocity at the
  particle positions, with controls. Needs one snapshot of the case mesh; runs
  on the workstation against a copied `final_state.npz`.
