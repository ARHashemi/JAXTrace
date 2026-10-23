# Two Routes to a Uniform Particle Field

**Why the divergence-free correction did not give uniform streamlines, and the
two things that would — each derived from scratch, with the effort each costs
us given what we already have.**

Written 2026-10-23. Companion to `DIVERGENCE_FREE_METHOD.md`, which documents
the DTP (deviatoric trace projection) itself.

---

## 0. The observation that prompted this

After making the FSW velocity field divergence-free, the final particle
distribution in ParaView looked **essentially unchanged** — same clumps, same
gaps. The expectation was that an incompressible field should carry a uniform
seed cloud to a uniform final cloud, with no voids.

Measured on cylA (2,684 steps, 293,455 particles), the coefficient of
variation of nearest-neighbour spacing — lower is more uniform:

| model | CV at the final step |
|---|---|
| raw P1 | **0.1226** |
| DTP | 0.1305 |
| HCT | 0.1306 |
| DTP+HCT | 0.1300 |

The DTP made uniformity **marginally worse**, not better. ⚠️ This document
explains why that is *expected* rather than a bug, and sets out the two
methods that would actually address it.

### 0.1 Why `det F = 1` does not imply uniform spacing

This is the crux, and it is worth being very precise.

`det F` is the **volume ratio of an infinitesimal material element** carried
along one pathline. The DTP drives it to 1 within 6.1e-06 on cylA. That is a
statement about **each particle separately**.

Particle *spacing* is a statement about **pairs of particles**. The two are
different, and a divergence-free flow controls only the first.

Consider the simplest incompressible flow, pure shear:

$$u = (\gamma y,\ 0,\ 0), \qquad \nabla\cdot u = 0 \ \text{exactly}$$

A fluid blob keeps its volume for all time — `det F = 1` — but after time `t`
a particle at height `y` has moved `γ y t` while its neighbour at `y + δ` has
moved `γ(y+δ)t`. Their separation along `x` grows **without bound**. The blob
becomes a long thin filament of the same volume.

So in any sheared incompressible flow, an initially uniform lattice **must**
become non-uniform. Nothing about the velocity field can prevent it.

### 0.2 What we measured, which narrows the cause further

Three candidate explanations, tested on cylA:

| hypothesis | test | verdict |
|---|---|---|
| the seed was never uniform | CV at step 0 | ❌ **0.0003** — a near-perfect lattice |
| shear stretches into filaments | local neighbourhood axis ratio `λ₁/λ₃` | ❌ only **1.56** — nearly round |
| it develops early then saturates | CV vs step | ✅ **see below** |

| step | raw CV | DTP CV |
|---|---|---|
| 0 | 0.0003 | 0.0003 |
| 25 % | 0.0878 | 0.0906 |
| 50 % | 0.1195 | 0.1251 |
| 100 % | 0.1226 | 0.1305 |

⚠️ Most of the non-uniformity appears in the **first quarter** and then
plateaus. That is the signature of a cloud **relaxing from an artificial
lattice onto the flow's own structure** — not of an error accumulating, which
would grow steadily with path length.

**So the diagnosis is: we seeded on a grid, and the flow does not organise
itself on a grid.** The two fixes below attack that from opposite ends — one
changes where we put the particles, the other changes the field they follow.

---

## 1. Flow-adapted seeding

### 1.1 The idea in one sentence

Instead of placing particles on a uniform spatial lattice, place them so that
**equal numbers of particles carry equal amounts of flux**, so the flow itself
keeps them evenly spread.

### 1.2 Why a uniform lattice is the wrong starting point

Take an inlet plane `A` with normal `n`. A uniform grid of `N` particles over
`A` assigns each particle the same **area** `ΔA = |A|/N`. But the amount of
fluid passing through that patch per unit time is the **flux**

$$\Phi(\Delta A) = \int_{\Delta A} u\cdot n \; dA \;\approx\; (u\cdot n)\,\Delta A$$

which varies with `u·n`. In FSW the velocity across an inlet plane varies by
orders of magnitude between the shear layer and the far field.

So a uniform grid gives each particle **unequal flux**. Particles seeded where
the flow is fast carry far more material than those in slow regions, and the
cloud immediately reorganises: fast particles race apart, slow ones bunch.
**That reorganisation is the 0.0003 → 0.09 jump in the first quarter of the
run.**

### 1.3 The fix: seed by cumulative flux, not by area

Choose seed positions so that each particle represents the **same flux**:

$$\Phi_{\text{total}} = \int_A u\cdot n\; dA, \qquad
\phi = \frac{\Phi_{\text{total}}}{N}$$

In one dimension across the plane (say along `y`), define the **cumulative
flux function**

$$F(y) = \int_{y_0}^{y} (u\cdot n)(y')\; dy'$$

which is monotonically increasing wherever `u·n > 0`. Then place particle `k`
at the position `y_k` solving

$$\boxed{\,F(y_k) = \left(k - \tfrac12\right)\frac{F(y_1)}{N},
\qquad k = 1\ldots N\,}$$

i.e. **invert the cumulative flux** at `N` equally spaced levels. This is
exactly inverse-transform sampling, with `u·n` playing the role of a
probability density.

In 2D over an inlet plane the same construction applies to the joint
distribution: sample `(y,z)` with density proportional to `u·n`, either by
inverting a 2D cumulative or by rejection sampling.

### 1.4 Why this is the *right* invariant to impose

For a steady divergence-free field, the flux through a **stream tube** is
constant along the tube:

$$\frac{d}{ds}\int_{A(s)} u\cdot n\; dA = 0$$

by the divergence theorem, since no flux crosses the tube's walls. So if each
particle starts carrying flux `φ`, it **continues** to represent flux `φ`
everywhere downstream.

✅ The particle number density then satisfies, at every downstream station,

$$n(x) \;\propto\; \frac{|u(x)|}{\phi}$$

— the particles are *dense where the flow is fast* and *sparse where it is
slow*, which is precisely the stationary distribution the advection itself
produces. **Seeded this way, the cloud does not reorganise, because it starts
in the configuration the flow would have driven it to anyway.**

✅ **Verified numerically** on a 1D channel with a varying speed profile
(`u = 0.2 + 2 sin²(πy)`, 5,000 particles):

| seeding | CV of flux per particle |
|---|---|
| uniform grid | **0.5894** |
| flux-matched | **0.0000** |

and the resulting seed density tracks `|u|` to a CV of **0.0052**, confirming
`n ∝ |u|` as claimed.

⚠️ **This does not give uniform spacing in physical space.** It gives
*stationary* spacing: no clumping or void formation *develops*, because the
distribution is already the flow's own. If what you want is a literally
even-spaced cloud at the outlet, that is incompatible with an incompressible
flow of varying speed, and no seeding can deliver it.

### 1.5 The time-dependent complication

Our FOM is unsteady (166 snapshots cycled). `u·n` on the inlet plane changes
between snapshots, so a flux-matched seeding is only exact for the snapshot it
was built from.

Two defensible options:
1. **Seed from the time-averaged flux**, `⟨u·n⟩_t`. Simple, one seed set, and
   correct in the mean.
2. **Re-seed per snapshot** if particles are injected continuously. Correct
   instantaneously, but changes the experiment into a continuous-injection one
   rather than a fixed cohort.

For the current fixed-cohort experiments option 1 is the right choice.

### 1.6 What it costs us

| piece | status | effort |
|---|---|---|
| read `u·n` on a seeding plane | ✅ we already load the nodal velocity | — |
| build `F(y)` and invert it | new, ~60 lines of numpy | **hours** |
| feed positions to the tracker | ✅ `--seed-source file` exists and is tested (used for the rot_cyl common seed set) | — |
| validate | re-run cylA, compare CV at step 0 and at the end | one 26-min arm |

⚠️ **No kernel change, no GPU work, no memory cost.** The seeding happens once
on the host before Stage 4, and `run_tracking.py` already accepts an arbitrary
`(N,3)` array through `--seed-file`. This is the cheapest of the two options by
a wide margin.

**Estimated: half a day**, most of it validation rather than implementation.

### 1.7 ⚠️ What it will and will not change

- ✅ Will remove the **initial reorganisation transient** (the 0.0003 → 0.09
  jump), because the cloud starts in the flow's own distribution.
- ✅ Will make the particle density a **meaningful sampling of the flux**,
  which matters if void fraction is ever computed as a particle-count density.
- ❌ Will **not** give evenly-spaced particles at the outlet. That is
  incompatible with incompressible flow at varying speed.
- ❌ Will **not** change `det F`, the damage indicators, or any per-particle
  quantity. It changes *which* material points we follow, not the physics
  along each path.

---

## 2. The flux-continuity repair (Cordes–Kinzelbach)

### 2.1 The defect it fixes

Measured on cylA, 101,612 interior faces, comparing the normal velocity seen
from each side of a face **at the same point**:

| field | faces with opposite-sign `u·n` | relative jump p99 |
|---|---|---|
| raw P1 | **0 / 101,612 (0.000 %)** | 0.0000 |
| **DTP** | **1,043 / 101,612 (1.026 %)** | **1.0000** |

✅ The raw P1 field is **exactly flux-continuous** across faces. ❌ **The DTP
breaks it**, because it removes each element's trace independently and two
neighbours get different corrections.

Selzer & Cirpka [Sel20] name the consequence precisely:

> "the non-conforming property of the velocity field leads to severe numerical
> artifacts, possibly including **particle stagnation** because the
> approximated normal velocity component **points into opposite directions at
> the two sites of an element interface**"

A particle arriving at such a face is pushed back from both sides. 1.03 % of
faces being traps is a plausible contributor to the extra 0.008 CV the DTP
added over raw.

### 2.2 The idea in one sentence

Rebuild the velocity field on **patches around nodes** rather than elements,
choosing the patch fluxes so that mass balance holds **exactly at every node**
— which makes the normal flux continuous across every internal boundary, using
only neighbouring elements and **no global solve**.

### 2.3 The construction, step by step

Following Cordes & Kinzelbach [Cor92], in our notation.

**Step 1 — the patch.** Around each node `i`, form the patch `P_i` by joining
the sub-regions of every element touching `i`. In 2D a triangle is split by
its medians into three sub-triangles, one per vertex; in 3D a tetrahedron
splits into four sub-volumes, one per vertex. Every element contributes
exactly one sub-volume to each of its vertices' patches, so the patches
**tile the domain without overlap**.

**Step 2 — the element velocities we start from.** On each element the P1
field gives one constant velocity

$$v^e = -K^e\,\nabla h^e \quad\text{(their Darcy form)},
\qquad\text{for us simply } v^e = A^e x + b^e \ \text{at the centroid}$$

**Step 3 — the flux a patch receives from each element.** For element `e`
overlapping patch `i`, the flux across the part of the patch boundary lying
inside `e` is

$$Q_i^e = (v^e\cdot n)\;\ell_i^e\;t$$

with `ℓ` the length of that boundary segment (an area in 3D) and `t` the
thickness. ⚠️ This is the flux of the *existing* element velocity through the
*new* patch boundary — no new unknowns yet.

**Step 4 — impose mass balance at every node.** Require

$$\boxed{\ \sum_{e=1}^{M} Q_i^e \;=\; R_i\ }$$

where `R_i` is any source/sink at node `i` (zero for us — FSW has no injection
or extraction in the bulk). ⚠️ **This is the condition the standard P1
Galerkin solution already satisfies**, which is the paper's key observation:
*"standard finite element flow models yield exact water balances, which in the
usual approach to path line computation are not fully used."*

**Step 5 — close the remaining freedom.** Mass balance alone does not
determine the patch velocities uniquely. The second condition is that the
field remain **irrotational in a weak sense**, i.e. a potential still exists:

$$\oint_X \nabla h\cdot dx = 0$$

on a closed path `X` around node `i`, which for piecewise-constant gradients
gives

$$\boxed{\ \sum_{e=1}^{M} (K^e)^{-1} v^e\cdot x^e \;=\; 0\ }$$

the sum running over the patch's elements. ⚠️ For us, **this is where the
method must be adapted**: FSW is not a potential flow, so the irrotationality
condition is wrong. A rotational closure is needed — see §2.5.

**Step 6 — the result.** Solving the small per-patch system (M unknowns, M
equations, M = the patch's element count, typically 5–20) gives patch
velocities `v_i^e` whose **normal component is continuous across every patch
boundary** and which **conserve mass exactly at every node**. Pathlines can
then be integrated normally.

⚠️ **The paper also offers a streamfunction interpolation giving pathlines in
closed form with no time stepping — and that part does NOT transfer to us.**
It is 2D-specific, and the related single-step constructions of [Pol88] and
[Lu94] depend on a block-centred finite-difference cell where each velocity
component is a function of its own coordinate alone. Our tetrahedral affine
map `u = Ax + b` has no such separation. **The benefit of this route for us is
the flux continuity, nothing more.**

### 2.4 Why this is the right shape of fix for us

| property | DTP | Cordes–Kinzelbach |
|---|---|---|
| local, no global solve | ✅ | ✅ (per-patch, M×M) |
| divergence-free | ✅ pointwise per element | ✅ exact nodal mass balance |
| **flux-continuous across faces** | ❌ **broken, 1.03 %** | ✅ **by construction** |
| preserves the gradient | ✅ cos 0.99998 | ⚠️ unmeasured for us |
| cost | ~0 | small solve per node |

It is the only candidate that targets **exactly** the property we measured
broken, while keeping the locality that makes the DTP affordable. [Sel20]'s
RTN₀ projection also restores conformity but at the price of a
**rotation-blind** element (§11.1 of the method document), which is
disqualifying for a rotation-dominated FSW field.

### 2.5 ⚠️ Four things that must be adapted before this works for us

The paper is 2D, groundwater, potential flow. Each gap is real and I would
rather name them now than discover them mid-implementation.

1. **2D → 3D.** The paper says *"the generalization to three dimensions is
   indicated"* but develops only the 2D case. Tetrahedral patches split into
   four sub-volumes per element; the boundary segments become polygonal faces.
   ✅ We already have the face topology (`build_face_topology`) and node→element
   patches (`build_node_patches`) from the BDM₁ and SPR work, so the geometry
   is largely in hand.

2. **Potential flow → rotational flow.** Step 5's closure assumes
   `rot(K⁻¹v) = 0`. FSW is **rotation-dominated** — a rotating pin is the
   entire point — so this condition is physically wrong here. ⚠️ **A
   replacement closure must be chosen and justified.** The natural candidate
   is least-squares: among all patch velocities satisfying nodal mass balance,
   take the one closest to the original element velocities,
   `min Σ_e |v_i^e − v^e|²`. That is well posed, local, and mirrors the
   minimum-norm principle that made the DTP defensible.

3. **Darcy → Navier–Stokes.** The `K⁻¹` weighting is a hydraulic conductivity.
   For us it drops out (or becomes a viscosity weighting if we want the
   closure to respect the local rheology).

4. **Steady → unsteady.** They treat transient flow as steady per time step
   with storage lumped into `R_i`. For us `R_i = 0` and each snapshot is
   treated independently — which matches how the DTP already works.

### 2.6 What it costs us

| piece | status | effort |
|---|---|---|
| node→element patches | ✅ `build_node_patches` exists (SPR uses it) | — |
| face topology & adjacency | ✅ `build_face_topology`, `element_adjacency` | — |
| 3D patch sub-volume geometry | **new** — the medial split of a tet and its boundary faces | days |
| per-patch M×M assembly and solve | **new**, but tiny and embarrassingly parallel | days |
| **choosing and justifying the closure** | **new, and the real risk** (§2.5.2) | ⚠️ **open-ended** |
| storing the result | ⚠️ patch-based, not element-based: a **new sampler** in the kernel, and the field is no longer one affine map per element | days |
| validation | rot_cyl against analytic truth + cylA det F + the face-continuity test we already wrote | 1–2 days |

**Estimated: 2–3 weeks**, and that assumes the closure question resolves
cleanly. ⚠️ The honest risk is item 5: the paper's closure does not transfer,
and the replacement is a modelling choice we would have to defend, not a
transcription.

⚠️ **A second cost that is easy to miss:** the output is a **patch-wise** field,
whereas everything downstream (the kernel sampler, `divergence_single`, the
damage accumulators) assumes **one affine map per element**. That is a deeper
change than the DTP, which rode entirely on the existing `centroid_taylor`
plumbing.

### 2.7 ⚠️ And the question that should be asked first

Before any of that: **does flux discontinuity actually matter for our
results?**

What we know:
- the DTP creates sign-flipped faces at **1.03 %**
- it added **+0.008** to the final CV versus raw (0.1226 → 0.1305)
- trajectory accuracy against analytic ground truth changed by **0.3 %**
  (rot_cyl, step 3000)

So the measured harm is small. The premise-first rule that has served this
project well (§11 of the method document, where every literature method was
refuted by testing its premise) says: **measure the damage before building the
repair.** A cheap test exists — take the 1,043 bad faces on cylA, find the
particles whose paths cross them, and compare their trajectories and
indicators against the rest. If they are not measurably worse, the repair is
not worth three weeks.

---

## 3. Side by side

| | flow-adapted seeding | Cordes–Kinzelbach repair |
|---|---|---|
| **what it fixes** | the cloud starts in the wrong distribution | the DTP's broken face continuity |
| **mechanism** | seed ∝ flux, not ∝ area | patch-wise field with exact nodal mass balance |
| **removes clumping?** | ✅ the *transient*; ❌ not the steady-state variation | ❌ not directly — it removes stagnation traps |
| **touches the kernel?** | ❌ none | ✅ new sampler, patch-based field |
| **memory** | none | patch field, size TBD |
| **changes `det F`?** | ❌ no | ⚠️ should preserve it; unverified |
| **main risk** | none — it is a pre-process | the closure (§2.5.2) does not transfer from potential flow |
| **effort** | **~half a day** | **~2–3 weeks** |

### 3.1 Recommendation

**Do the seeding first.** It is half a day, touches nothing, is reversible,
and directly addresses the mechanism the measurements point at — the cloud
reorganising in the first quarter of the run because it was seeded on a grid
the flow does not use.

**Then measure whether the 1.03 % of bad faces actually harms anything**
(§2.7) before committing weeks to the repair.

⚠️ **And one expectation to set now:** neither method will produce an
evenly-spaced particle field at the outlet. In an incompressible flow with
spatially varying speed, particle density *must* vary as `1/|u|` along a
stream tube. A void in the final cloud is not necessarily a numerical defect —
it may be where the material genuinely moves fastest. If the goal is to detect
*physical* voids, the quantity to look at is `det F` along each path (which
the DTP fixed, from `inf` to 6.1e-06) and the damage indicators — **not** the
visual density of the particle cloud.

---

---

## 4. Detecting a void WITHOUT relying on markers

⚠️ **The objection that prompted this section:** `det F` and the damage
indicators are carried *by particles*. If a region empties of particles, there
is nothing there to carry an indicator — the very place we most want to
measure is the place we have no data. And a particle-density map cannot
distinguish a real void from the `1/|u|` advection artefact of §3.1.

Both objections are answerable, and the fix is cheap.

### 4.1 The separation principle

For steady incompressible flow, mass conservation along a stream tube gives

$$\rho\,|u|\,A = \text{const} \;\;\Longrightarrow\;\; |u|\,A = \text{const}$$

so the tube **narrows where the flow is fast**, and a purely advected particle
cloud has number density

$$n(x) \;\propto\; \frac{1}{A(x)} \;\propto\; |u(x)|$$

⚠️ Therefore **low particle density is ambiguous**: it means *either* the flow
is slow there (an artefact) *or* material genuinely left (a void).

The resolution is to divide the artefact out. Define the **flux-normalised
density**

$$\boxed{\;\hat\rho(x) \;=\; \frac{n(x)}{|u(x)|}\;}$$

which is **constant** for a purely advected incompressible cloud. Any
*deviation from constant* is then a genuine material deficit, independent of
how fast the flow is.

✅ **Verified on a 1D stream tube** with a known synthetic void (60 % deep, at
`x = 0.65`) superimposed on a varying speed profile:

| field | minimum value | location of minimum |
|---|---|---|
| raw particle density | 0.167 | **x = 0.000** ❌ (merely the slowest point) |
| **flux-normalised** `n/\|u\|` | **0.400** | **x = 0.649** ✅ |

The raw map points at the wrong place; the normalised map finds the void and
recovers its depth exactly (`0.40 = 1 − 0.6`).

### 4.2 Why this needs no new machinery

`run_tracking.py` already ships an **online density-field estimator**
(`--density-enable`) that deposits particles onto a voxel grid with an SPH
kernel (Wendland C2 by default, with fixed/Scott/Silverman/kNN bandwidths).
That gives `n(x)` on an Eulerian grid.

And `|u(x)|` is available on the same grid from the FOM velocity field, which
is already loaded.

⚠️ So the whole diagnostic is **one division of two fields we already have**:

```
rho_hat = density_voxels / max(|u|_voxels, eps)
```

normalised so that the far field reads 1. ✅ **No new tracking, no kernel
change, no extra memory** — it is a post-process on outputs that exist.

### 4.3 ⚠️ The limits of this, stated plainly

1. **An empty cell gives no signal at all.** If a region is *completely*
   evacuated, `n = 0` and `\hat\rho = 0`, which is a detection — but nothing
   carries an indicator there, so we learn *that* it is empty, not the damage
   history of the material that left. That is intrinsic to a Lagrangian
   method, not a defect of this normalisation.
2. **It assumes steady flow per snapshot.** The stream-tube argument uses
   `|u| A = const` along a tube, which holds for a steady field. With 166
   cycled snapshots it holds per snapshot and approximately in the mean.
3. **It cannot distinguish "material left" from "material never arrived".**
   A region downstream of a stagnation point may be sparse because no pathline
   from the seed plane reaches it. ⚠️ Flow-adapted seeding (§1) helps here by
   making the seed cover the flux rather than the area, but a coverage check
   (which seed cells map where) is the honest companion diagnostic.
4. ⚠️ **It is a KINEMATIC indicator, not a mechanism.** It says where material
   is missing, not why. Pairing it with the pathline indicators — `det F`
   along the surviving paths, `ln_phi`, `w_bond` — is what connects a deficit
   to a cause.

### 4.4 The Eulerian alternative, for completeness

A fully marker-free route exists: solve a scalar **advection equation for a
material fraction** `φ` on the mesh,

$$\frac{\partial\varphi}{\partial t} + u\cdot\nabla\varphi = 0,
\qquad \varphi(t=0)=1$$

with `φ = 0` entering at the inlet of any region that should be empty. For a
divergence-free `u` this conserves `∫φ` exactly, and `φ < 1` marks a deficit
directly on the mesh with no particles at all.

⚠️ **Why I do not recommend it here.** It needs a stabilised advection solver
(the FOM's own P1/P1 stabilisation problem all over again), it is numerically
diffusive, and it would smear exactly the sharp deficits we are hunting. The
flux-normalised particle density gets the same information from machinery that
already exists and already works.

### 4.5 ✅ MEASURED ON cylA — and the apparent voids are an artefact

`scripts/damage/flux_normalised_density.py` and
`plot_flux_normalised_density.py`, run on the cylA DTP arm (293,455
particles, 2,684 steps) against the FOM snapshot 159.

⚠️ **Two things had to be fixed before the numbers meant anything**, and both
are the kind of error that produces confident nonsense:

1. **The cloud is a thin slab** (extent ratios 6 : 4.6 : 1). A cubic voxel
   grid left **0.7 % of voxels occupied** at 2.7 particles each — sampling
   noise, not a density field. Anisotropic bins (≈cubic *in space*) raised
   occupancy to **29.1 %**.
2. **Escaped particles** sit frozen at their last position and pile a false
   spike at the outlet. Dropped.

**The result, with the boundary control that the first pass lacked:**

| measure | deficit < 30 %, all occupied | **interior only** |
|---|---|---|
| raw density `n` | 4.91 % | **0.00 %** |
| flux-normalised `n/\|u\|` | 5.27 % | **0.00 %** |

✅ **Every apparent deficit is a cloud-BOUNDARY voxel** — partly filled for a
purely geometric reason. Requiring all six face-neighbours to be occupied
(51.7 % of occupied voxels qualify) leaves **zero** deficits in either
measure. The panel-5/panel-6 pair shows it directly: 133 red voxels tracing
the cloud outline, then an empty panel.

⚠️ **And the normalisation is a NO-OP on this case, for a measurable reason.**
Over the *occupied* voxels the speed spans only

| | `\|u\|` |
|---|---|
| p01 | 6.34e-03 |
| p50 | 6.59e-03 |
| p99 | 6.82e-03 |
| **range** | **1.08×** |

Dividing by a quantity that varies by 8 % cannot change the map, which is
exactly what the `(n/\|u\|)/n` panel shows — flat grey at 1.0. The mesh-wide
speed range is far larger, but **the particles sit in a narrow band of it**.
So this case does not test the discrimination; it only establishes that there
is nothing to discriminate.

**Conclusion for the original question:** the voids visible in ParaView on
cylA are **not** material deficits, and they are **not** a `1/\|u\|` artefact
either. They are the **edge of the particle cloud** — the seeded block has
simply been deformed, and its boundary looks like a void when rendered as
points. ✅ This is consistent with §0.2 (the cloud relaxing from a lattice)
and with `det F` being clean at 6.1e-06.

### 4.6 ⚠️ Recommendation

**Do §4.2 now, in post-processing.** It is a division of two existing fields,
it directly answers "is this void real or is the flow just fast here", and it
costs an afternoon rather than a method.

Concretely:
1. re-run (or reuse) one cylA/A-family arm with `--density-enable`;
2. sample `|u|` onto the same voxel grid from the FOM snapshot;
3. form `\hat\rho = n/|u|`, normalise to the far field;
4. overlay the surviving particles' `det F` and `ln_phi` to connect deficits
   to damage history.

If `\hat\rho` is flat everywhere, the apparent voids in ParaView are **pure
advection artefact** and the search for a velocity-field fix is over. If it
shows localised deficits, those are the candidate void sites — and they can
then be compared against the indicators carried by whatever particles remain
nearby.
## References

- **[Cor92]** Cordes, C. & Kinzelbach, W. (1992). Continuous groundwater
  velocity fields and path lines in linear, bilinear, and trilinear finite
  elements. *Water Resources Research* **28**(11), 2903–2911.
  https://doi.org/10.1029/92WR01686
  — The patch construction of §2. ✅ Fulltext in the local collection and read
  for this document.
- **[Sel20]** Selzer, P. & Cirpka, O. A. (2020). Postprocessing of standard
  finite element velocity fields for accurate particle tracking applied to
  groundwater flow. *Computational Geosciences*.
  https://doi.org/10.1007/s10596-020-09969-y
  — States the conformity requirement and names the particle-stagnation
  failure mode measured in §2.1. ✅ Fulltext read.
- **[Ver00]** Verma, V., Kao, D. & Pang, A. (2000). A flow-guided streamline
  seeding strategy. *Proceedings Visualization 2000*, 163–170.
  https://doi.org/10.1109/VISUAL.2000.885690
  — ✅ Fulltext read. Seeds by **flow features**: identify critical points,
  segment the field into regions containing one each, seed each with a
  template matched to the critical-point type, then fill the remainder with a
  **Poisson-disk** distribution *"to minimize closely spaced seed points"*.
  ⚠️ **Its goal is not ours.** It optimises for *visual* coverage of flow
  structure with few streamlines — explicitly *"aesthetically pleasing"* — not
  for each marker carrying equal material. The Poisson-disk step is
  nevertheless directly reusable as a refinement of §1.3: after inverting the
  cumulative flux, a Poisson-disk relaxation within the flux-matched
  distribution would remove residual clustering without disturbing the
  flux-per-particle invariant.
- **[Pol88]** Pollock, D. W. (1988). Semianalytical computation of path lines
  for finite-difference models. *Ground Water* **26**(6), 743–750.
  https://doi.org/10.1111/j.1745-6584.1988.tb00425.x
  — The flux-conserving semi-analytical pathline method that [Cor92]
  generalises from finite differences to finite elements. ✅ Fulltext in the
  local collection as a **scan without text**; read here via `tesseract` OCR
  of the 300 dpi page images.
  **Its core assumption:** *"each directional velocity component varies
  linearly within a grid cell in its own coordinate direction"* — the
  x-velocity varies linearly in x but is independent of y and z. That makes
  each component separately integrable in closed form, so *"given the initial
  position of a particle anywhere in a cell, the coordinates of any other
  point along its path line within the cell, and the time of travel between
  them, can be computed directly"*, and for steady flow the exit point is
  obtained **in a single step** with no time stepping.
  ⚠️ **It does not transfer to us directly:** the construction is for
  block-centred finite-difference *cells*, where a component can be made a
  function of its own coordinate alone. On an unstructured tetrahedral mesh
  with a full affine map `u = Ax + b` there is no such separation, so the
  single-step exit point is not available. What *does* transfer is the idea
  behind §2: use the flux information the flow solution already contains
  rather than re-deriving velocity by differentiation.
- **[Lu94]** Lu, N. (1994). A semianalytical method of path line computation
  for transient finite-difference groundwater flow models. *Water Resources
  Research* **30**(8), 2449–2459. https://doi.org/10.1029/94WR01219
  — ✅ Fulltext read. Extends [Pol88] to **transient** flow by interpolating
  the particle velocity **linearly in both space and time** within each cell
  and time step, then integrating the total derivative analytically:
  *"errors are much smaller than in numerical integration schemes."*
  ⚠️ **Why this one matters for us specifically:** our FOM is unsteady (166
  cycled snapshots) and §2.5.4 flagged that [Cor92] treats transience only as
  a sequence of steady steps. [Lu94] is the published answer to exactly that
  gap. It shares [Pol88]'s finite-difference cell restriction, so it is a
  guide to the *time* treatment rather than a drop-in method.

### ⚠️ What reading [Pol88], [Lu94] and [Ver00] changed

All three are now read. The updates above are the substance; the summary:

- **The semi-analytical route is narrower than I implied.** [Pol88] and
  [Lu94] both depend on a **block-centred finite-difference cell**, where each
  velocity component can be made a function of its own coordinate alone. Our
  tetrahedral affine map `u = Ax + b` has no such separation, so "pathlines in
  closed form, no time stepping" — which I listed as an attraction of
  [Cor92] — **does not transfer**. [Cor92]'s streamfunction interpolation is
  2D-specific for the same reason. ⚠️ **The attraction of §2 is the flux
  continuity alone**, and I should not have advertised the closed-form
  pathlines.
- **[Lu94] is the right reference for the unsteady gap** flagged in §2.5.4,
  even though its method is not directly usable: interpolate velocity linearly
  in space *and time*, then integrate the total derivative.
- **[Ver00]'s Poisson-disk step is a genuine refinement of §1**, usable on top
  of flux-matched seeding.

### Still worth having

1. Anything on **3D** Cordes-type patch postprocessing on **simplices** — the
   1992 paper only indicates the 3D case, and §2.5.1 remains the gap.
2. A treatment of flux-continuous reconstruction for **rotational** flow,
   which is the §2.5.2 closure problem and the one real risk in route 2.
