# Why particles stop in the D-family cases

**A stand-alone analysis.** Written 2026-10-07. No case folder was modified;
everything below comes from reading the existing output files.

> **SUPERSEDED — read [`05_root_cause_and_fix.md`](05_root_cause_and_fix.md)
> for the resolved answer.**
>
> This file is kept as the record of how the investigation started. Two of its
> conclusions did not survive measurement:
>
> - *Explanation A* (the level set zeroing velocity) was **ruled out**: frozen
>   particles sit outside the tool radius and 100% of them had `ElementID<0`.
> - The suggested remedy — raising `ENHANCED_SEARCH_BAND`, `L0_SKIP_BAND` or
>   `L2_NEIGHBORHOOD` — **cannot work**. Widening a search does not help when
>   the search is computing the wrong cell index in the first place.
>
> The actual cause: the octree's per-level cell size was taken from the *first*
> cell at each level, which on the cuboid D meshes is an unrepresentative cubic
> outlier. The z index was inflated 1.197×, growing with depth to 14 cells, so
> the true host fell outside the 3×3×3 neighbourhood. Fixed by using the
> per-axis median.

---

## 0. Vocabulary (read this first if the terms are unfamiliar)

The particle tracker moves massless "tracer" points through a velocity field
that FEMUSS computed on a tetrahedral mesh. Four terms recur below:

- **Host element.** The single tetrahedron that currently contains a
  particle. To move a particle you must first know which tetrahedron it sits
  in, because the velocity is only defined at the four corners of that
  tetrahedron and is interpolated inside it.
- **Point location / search.** The act of finding the host element. JAXTrace
  does this in three stages: **L0** re-checks the tetrahedron the particle was
  in last step (almost always correct and nearly free); **L1** checks that
  tetrahedron's immediate neighbours; **L2** (this is MALMO) is the global
  fallback that searches a 3x3x3 block of grid cells. If all three fail, the
  particle has no host.
- **`element_id = -1`.** The marker written when no host was found. The
  particle is then frozen: it keeps its last position forever and is excluded
  from the physics.
- **Level set (`LEVEL` field).** A signed distance that marks where the
  rotating tool is. `LEVEL < 0` means "inside the tool", where there is no
  material and therefore no meaningful velocity. The run is configured with
  `LEVELSET_MODE=zero_vel`, meaning a particle found inside the tool is given
  **zero velocity** for that step: it stops.

---

## 1. What we are trying to explain

In families A (Flats), B (Flutes) and C (Threads), tracking works. In family
D (ConcavityTilt), a large fraction of particles stop moving and pile up
around the tool, in the refined part of the mesh.

Measured, in `phase4_rk4_30mm`:

| family | acc_frac | fraction within 7 mm of tool | median radius |
|---|---|---|---|
| A-Flats | 0.998 | 0.035 | 21.2 mm |
| B-Flute | 0.997 | 0.028 | 21.3 mm |
| C-Threads | 0.997 | 0.033 | 21.3 mm |
| **D-Concav** | **0.818** | **0.181** | **18.5 mm** |

Counting particles whose final `element_id` is negative:

| case | lost | % |
|---|---|---|
| A1 | 0 | 0.00% |
| C1 | 0 | 0.00% |
| D1 | 35,941 | 35.9% |
| D4 | 35,369 | 35.4% |

A and C lose **zero**. So this is not a dial that is slightly mistuned; D
does something the others never do.

---

## 2. The decisive measurement: when do they stop?

I compared particle positions between exported steps for D2 and defined a
particle as "frozen" when it moves less than 1e-6 mm.

```
dead on arrival (never moved, step 0 -> 100)   18,327
froze later during the run                     17,635
total frozen at the final step                 35,962  (36.0%)
```

**This is two different problems, not one.** They need separate fixes.

### 2a. The 18,327 that never moved at all

Where were they?

```
of the 18,327 dead-on-arrival:   z > 0 : 17,929  (97.8%)
                                 z <= 0:    398  ( 2.2%)

of ALL particles seeded at z > 0  (18,281):  98.1% are dead
of ALL particles seeded at z <= 0 (81,719):   0.49% are dead
```

Essentially **every particle that starts above z = 0 is dead immediately**,
and almost every particle below z = 0 is fine. That is not a search
accuracy issue - a search that was merely struggling would fail
intermittently, not at a 98% rate on one side of a plane.

The reason is geometric. Full mesh bounds:

| mesh | z range |
|---|---|
| A1 | -6.00 .. **0.00** mm |
| D1 | -6.00 .. **+1.35** mm |

x and y are identical (-20..40, -15..15). **Only the D mesh extends above
z = 0.** The D tool has a concave, tilted shoulder, so the mesh bounding box
has to be tall enough to contain it - but the *material* does not fill that
extra height. The region between the real top surface and z = +1.35 mm is
mostly empty space inside the bounding box.

The seeding grid is specified as a box and then filled uniformly. For A, the
box and the material agree, so every seed lands in material. For D, the same
box now spans up to +1.35 mm, so roughly 18% of the seeds are placed in the
empty space above the sloped surface. They have no host element on step one
and are marked lost before the simulation does anything.

Confirming the surface is tilted, not flat: fitting a plane through those
dead particles gives

```
z = -0.00993*x - 0.00106*y + 0.511     ->  -0.57 degrees about y
```

and fitting the mesh's own nodes above z = 0 gives

```
z = -0.00462*x - 0.00063*y + 0.279     ->  -0.27 degrees about y
```

Same axis, same sign. This is the "Tilt" of ConcavityTilt, recovered from
the particle data.

**Consequence:** ~18% of the 100,000 particles are wasted before tracking
begins. This alone accounts for over half of the 36% loss, and it drags
`acc_frac` from 0.998 to 0.82. It is a **seeding** problem, not a MALMO
problem.

### 2b. The 17,635 that froze during the run - this is the real question

These started in material, moved normally, then stopped. Their final
positions:

```
z <= 0 : 15,848   median radius 3.44 mm   median z -2.88 mm
z > 0  :  1,787
```

So ~15,800 particles are frozen **deep inside the material** (3 mm below the
surface), in a tight ring close to the tool. This matches what you see in
`particles_step_008267.vtu`. Their radial distribution:

```
  2- 3 mm   2,284  ############
  3- 4 mm   9,489  ##################################################
  4- 5 mm   1,460  #######
  5- 6 mm     154
  6- 7 mm   2,065  ##########
  7- 8 mm     272
```

Over 60% sit in a 1 mm-wide band at **3-4 mm radius**. For reference, in the
A1 mesh the level-set tool region (`LEVEL < 0`) reaches **r = 2.97 mm**.

**The freeze ring sits immediately outside the tool boundary.**

---

## 2c. NEW EVIDENCE: the D tool is more than twice as wide near the surface

After the first draft I was able to read the `LEVEL` field for D2 directly.
(The earlier attempt failed only because D2's mesh files are prefixed `A2_`,
not `A1_`.) Comparing the tool region (`LEVEL < 0`) by depth:

| depth band | A1 tool radius | **D2 tool radius** |
|---|---|---|
| z -6 .. -4 mm | 2.38 mm | 2.49 mm |
| z -4 .. -2 mm | 2.70 mm | 2.49 mm |
| **z -2 .. 0 mm** | **2.97 mm** | **6.73 mm** |
| z 0 .. +1.5 mm | 2.49 mm (89 nodes) | 6.73 mm (9,911 nodes) |

Deep down the two tools are the same size - a ~2.5 mm pin. But in the top
2 mm, **D2 flares out to 6.73 mm**: that is the concave shoulder, and A1
simply does not have one. This is the single biggest geometric difference
between the families, and it was invisible in the summary metrics.

### What this does to the freeze ring

Splitting the 15,848 late freezers by depth and asking how many lie inside
the D2 tool radius *for their own depth*:

| depth band | frozen | inside tool region | median radius |
|---|---|---|---|
| z -6 .. -4 mm | 5,575 | 57 (**1.0%**) | 3.50 mm |
| z -4 .. -2 mm | 4,621 | 52 (**1.1%**) | 3.10 mm |
| z -2 .. 0 mm | 5,652 | 4,442 (**78.6%**) | 4.13 mm |

This is a clean separation, and it splits the "real problem" again:

- **Near the surface (z > -2 mm): 78.6% of the frozen particles are inside
  the tool region.** For these, `LEVELSET_MODE=zero_vel` is doing exactly
  what it is configured to do - they entered the shoulder footprint and had
  their velocity zeroed. The host element is almost certainly fine. This is
  **Explanation A**, and it is not a bug in the tracker; it is the tool
  sweeping a much larger area than in A/C.
- **Deeper down (z < -2 mm): only ~1% are inside the tool.** About 10,200
  particles freeze at a median radius of ~3.1-3.5 mm while sitting *outside*
  the 2.49 mm pin. The level set does not explain these. This is the
  population that still points at **Explanation B** - a search failure in the
  refined mesh just outside the pin.

So the final accounting for D2's 35,962 frozen particles is:

| population | count | cause |
|---|---|---|
| seeded above the material | 18,327 | seeding box vs tilted mesh (section 2a) |
| frozen inside the shoulder footprint | ~4,400 | level set, working as configured |
| frozen near the surface but outside tool | ~1,200 | unclear, likely level-set boundary |
| **frozen deep, outside the pin** | **~10,200** | **unexplained - the real defect** |
| frozen above z = 0 after moving | 1,787 | tilted free surface |

**The genuinely unexplained group is ~10,200 particles (10% of the total),
not 36%.** That is the one worth chasing.

---

## 3. Two candidate explanations for 2b, and how to tell them apart

The freeze ring is consistent with either of two causes. We cannot yet
distinguish them from the data we have, and they need different fixes.

### Explanation A - the velocity field stops them (physics / level set)

`LEVELSET_MODE=zero_vel` means: if a particle is judged to be inside the
tool, its velocity is set to zero for that step. If the level set for the D
tool is slightly too generous - for instance because the concave shoulder
makes the signed distance inaccurate near the pin - then particles sitting
just *outside* the real tool get zero velocity anyway. Once a particle's
velocity is zero it never moves again, so it cannot escape: the condition is
self-locking.

This would produce exactly a thin ring just outside the tool radius, which
is what we see.

Evidence for: the band is narrow and at a fixed radius, which is what a
geometric criterion produces. The tilt means the tool's vertical position
varies with x, so a level set computed for an untilted tool would be
systematically wrong in a band.

Evidence against: nothing yet rules it out.

### Explanation B - the search loses them (MALMO / L0-L1-L2)

The mesh is heavily refined near the tool. In refined regions tetrahedra are
small, so a particle crosses several of them per step, and L0 and L1 are more
likely to miss. L2 (MALMO) is the backstop, running with
`L2_NEIGHBORHOOD=3` - a 3x3x3 block of cells. If the local cells are much
smaller than the particle's step, the true host can lie outside that block
and the search returns -1.

Evidence for: the ring is in the most refined part of the mesh; D1 has 14%
more cells than A1 (7.92 M vs 6.96 M), so its refinement is finer.

Evidence against: A and C use the same search on similarly refined meshes and
lose **zero** particles. That is a strong argument that the search itself is
sound and something about D is different.

### The measurement that separates them

Re-run one D case with **`EXPORT_ELEMENT_IDS=1`**. That writes each
particle's host element per step. Then:

- If a particle's `element_id` stays **valid** (>= 0) while it stops moving,
  the search is working and the **velocity is zero** -> Explanation A, the
  level set.
- If `element_id` flips to **-1** at the step it stops, the **search failed**
  -> Explanation B, MALMO.

This is a single diagnostic run and it is decisive. Everything else is
guesswork until we have it.

---

## 4. What the proposed settings actually do

You asked what the suggested fixes mean. All three already exist in
`run_jaxtrace.sh`; all three are currently **off** in these runs.

### `ENHANCED_SEARCH_BAND` (currently 0.0, suggested 1e-3)

"Within this distance of the tool surface, use a more thorough search."
Specifically it switches L1 from face-neighbours to node-neighbours (checking
every tetrahedron sharing a corner, not just a face - many more candidates)
and widens L2 from a 3x3x3 to a 5x5x5 block of cells. Setting it to `1e-3`
means "apply this within +/-1 mm of the tool surface".

It costs speed, but only in the band. **It helps only if the cause is
Explanation B.**

### `L0_SKIP_BAND` (currently 0.0, suggested 0.5e-3)

L0 is the cheap check "is the particle still in the same tetrahedron as last
step?". Near the tool boundary that cached answer can be stale - the particle
may have crossed into the tool region. This setting says "within +/-0.5 mm of
the tool surface, don't trust the cache; search properly." Again, it only
matters for Explanation B.

### `L2_NEIGHBORHOOD=5` (currently 3)

Widens MALMO's global search from a 3x3x3 to a 5x5x5 block of cells
everywhere, not just near the tool. Slower everywhere. A blunter version of
the first option.

**None of these touch Explanation A.** If the level set is zeroing the
velocity, the host element is found correctly and no amount of extra
searching changes anything. That is precisely why the diagnostic run in
section 3 has to come first.

---

## 5. The seeding problem (2a) has a separate, simple fix

Independent of which explanation wins, ~18,000 particles are being thrown
away at seeding. Three options:

1. **Lower the seed box top.** `SEED_BOX` currently ends at z = -0.00019 m.
   Because the box is cropped against the *mesh bounding box*, and D's box
   is taller, seeds end up above the material. Pinning the top to the real
   material surface would fix it.
2. **Drop seeds with no host element.** Check at step 0 and reseed them.
   Robust regardless of geometry.
3. **Seed by fraction of the material, not the bounding box.** Avoids the
   problem for any future tilted-tool case.

Option 2 is the most general and the least likely to need revisiting.

---

## 6. Summary

- D loses ~36% of particles; A and C lose none.
- **Half of the loss (18,327) is a seeding artefact**: the D mesh is taller
  than A/C because the tilted, concave tool needs the headroom, so ~18% of
  seeds are placed in empty space above the sloped surface and die on step
  one. Fixable immediately, independent of everything else.
- **The other half (17,635) is the real problem** you identified: particles
  that start correctly, move, then freeze in a 1 mm-wide ring at 3-4 mm
  radius, 3 mm deep in the material - just outside the tool boundary
  (r ~ 2.97 mm in A1).
- Reading D2's level set then split that ring in two. **D2's tool flares to
  6.73 mm in the top 2 mm** (the concave shoulder); A1's never exceeds
  2.97 mm. Near the surface, 78.6% of the frozen particles are *inside* that
  footprint - the level set is zeroing their velocity by design, not failing.
- **Deeper than z = -2 mm, only ~1% are inside the tool.** ~10,200 particles
  freeze at r ~ 3.1-3.5 mm while outside the 2.49 mm pin. Nothing in the
  configuration explains those.
- **One diagnostic run with `EXPORT_ELEMENT_IDS=1` settles whether those
  ~10,200 are a search failure or a velocity-field artefact.** Until then,
  changing search settings is a guess.

### What I have NOT done

- No case folder was modified; no job was submitted; no ssh was used.
- The D2 level set has now been read directly (section 2c), so the earlier
  caveat about assuming A1's tool radius is resolved - and the assumption
  was wrong, which is why section 2c exists.
- Measurements in section 2c use 6 of D2's mesh pieces, not all 128. The
  radius figures are lower bounds; reading the full mesh may widen them
  slightly. The qualitative split (shoulder vs deep pin) is unaffected.
- The ~10,200 deep freezers remain unexplained. Do not assume MALMO until
  the `EXPORT_ELEMENT_IDS=1` run confirms it.
