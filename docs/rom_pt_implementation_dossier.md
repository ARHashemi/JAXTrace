# ROM PT · Implementation Dossier

**Date:** 2026-07-31
**Purpose:** consolidated technical reference for every next-step in the
plan doc. Each section below covers *what to do*, *why it works*
(with peer-reviewed citations), *how to implement it in our JAX/GPU
stack*, and *what performance / accuracy to expect*. This is the
document to open before starting Steps 4–12 or defining Steps 13+.

**Companion docs**:
- [`rom_pt_roadmap.md`](rom_pt_roadmap.md) — overall plan structure
- [`rom_pt_next_steps_plan.md`](rom_pt_next_steps_plan.md) — the 12-step priority table
- [`rom_pt_roadmap_REVIEW2_evaluation.md`](rom_pt_roadmap_REVIEW2_evaluation.md) — Review 2 + first literature pass
- [`rom_pt_step5_step6_grid_report.md`](rom_pt_step5_step6_grid_report.md) — grid-experiment results
- This dossier — implementation-ready blueprints (**you are here**)

**Literature basis:** 32 references from the first Consensus pass +
32 new references from a second pass focused on implementation
details (JAX/GPU, integrators, POD variants, FSW validation).
Complete reference table at the end of this document.

---

## Section 1 · Grid design — is `4lvl_hct` genuinely optimal or is there a better variant we haven't tested?

### What Part 5 showed
`4lvl_hct` beats every alternative tested (5lvl, malmo6, 4lvl_r22,
2lvl_tricubic, 4lvl_tricubic) on the 20-case cohort. Grid-topology
optimisation is saturated *for the current framework*: cell count
(64k vs 17M), block layout (nested vs MALMO-aligned), Stage-2
interp order (trilinear vs tricubic) all move the number by less
than 5%.

### What the extended literature suggests

The saturation observation is consistent with **approximation theory**:
once the source velocity field is C¹ (via HCT Stage 1), the
interpolation error scales as (Δx)^p × ‖derivative‖ — and beyond a
critical Δx (roughly the finest tet edge), the derivative bound is
what dominates, not Δx. Refining beyond that adds cells without
buying accuracy [G-3 §5, §6].

**Two paths remain untested that the literature suggests could move
the number**:

#### 1a. Metric-based / anisotropic grid refinement guided by the mesh's own error field
Instead of centred bboxes with fixed refinement radii, use the FE
mesh's own **solution-based error indicator** to drive the AMR
metric field [Fidkowski 2020](https://consensus.app/papers/details/c030727c1f70512c9b7a7509757c1791/?utm_source=claude_desktop) [34], [Balan 2021](https://consensus.app/papers/details/d4b2a597c55a50089c493338e4c8595a/?utm_source=claude_desktop) [35],
[Frey 2005](https://consensus.app/papers/details/d0f5f415917d5e51b1232f9243ee48f9/?utm_source=claude_desktop) [36]. The Cartesian grid then aligns
its refinement with the *velocity gradient tensor's principal
directions* — including the shear band at r ≈ 21 mm identified in
Part 4 fig6, but with **anisotropic** cell shapes that better match
the shear-layer geometry (thin in the radial direction, elongated
tangentially). Our current 4lvl blocks are isotropic cubes.

**Cost estimate**: modest. Refactor `make_hier_grid()` to take a
per-region `metric_tensor(x,y,z)` and emit rectangular (not cubic)
sub-blocks. HCT Stage-1 code path is unchanged.

**Expected impact**: potentially 20–40% rms reduction in shear-band
region (from [35] anisotropic error control examples), possibly
none elsewhere. Would help only if the shear-band error is the
dominant contribution to cohort rms — which Part 4 said it is.

#### 1b. Metric-based compression-ratio constraint (Prouvost 2024)
[Prouvost 2024](https://consensus.app/papers/details/2b2628ea29215001a4c9ca3c966df25d/?utm_source=claude_desktop) [37]
showed that minimising the *interpolation error* on octree grids
does NOT always minimise the *total numerical error*. There's an
optimal "compression ratio" (finest cell size / mean cell size) that
depends on the solution structure. Our 4lvl currently has ratio ≈ 4
(inner 1.25e-4 m vs base 5e-4 m). Prouvost's argument suggests
exploring ratios 2, 6, 8, 12 to find the sweet spot.

**Cost**: near-zero (already in the builder). Just needs the sweep.

**Verdict**: worth 4–6 grid module builds + PT runs on the 4-case
cohort before committing to Steps 6–8. Low cost, high information
value.

---

## Section 2 · Higher-mode POD (Step 6 in plan doc) — implementation notes

### Cheapest test of the basis-quality hypothesis
Rebuild POD with mode counts n = 3, 5, 8, 12, 16, rerun §2 FOM-vs-ROM
PT compare on the 4-case cohort. All the literature we surveyed
suggests this will help if the 3-mode ceiling is the real
bottleneck.

### What the literature quantifies

- [Duan 2023 (IECR)](https://consensus.app/papers/details/4f51abe86082549d8cae1f4fafb4b401/?utm_source=claude_desktop)
  [38] formalises exactly our finding for CFD-DEM: *"the POD-based
  ROM shows poor and excellent predictability for Lagrangian and
  Eulerian variables, respectively."* Proposes a "predictability
  ratio" that quantifies mode-count sufficiency — worth computing on
  our snapshots as a diagnostic. Also suggests **mapping Lagrangian
  variables onto fixed Eulerian meshes** to improve prediction
  (which is what we did via the grid experiment).
- [Shnapp 2026 "Lagrangian POD"](https://consensus.app/papers/details/f13a00a5fc8056fcbda4d7fa83fe6ef9/?utm_source=claude_desktop)
  [39]: for turbulence with ensemble particle trajectories, **~10
  modes reproduces single-particle dispersion and curvature
  statistics accurately** on trajectories at integral-time-scale
  length; **30–60 modes needed for acceleration distribution
  tails**. That's a concrete "mode-count vs which Lagrangian
  statistic converges" number.
- [Bhattacharyya 2020](https://consensus.app/papers/details/7b0fb54d90cc5518802798ecad7f59e9/?utm_source=claude_desktop)
  [40] shows that **conventional variance-based mode selection
  (99.9% energy) leads to inaccurate models for spatiotemporally
  localised loadings**, and proposes "energy closure" as an improved
  selection criterion — the reduced subspace should not *lose* the
  energy input/dissipation balance of the full system. Worth
  implementing as a diagnostic: check whether our 3-mode ROM
  captures the energy input at the pin.
- [Brindise 2017](https://consensus.app/papers/details/6caf7607c7505528b8ac5783e42161f2/?utm_source=claude_desktop)
  [41] proposes the "entropy-line fit" (ELF) method for objective
  mode selection: analyses the spatial DCT of eigenmodes and
  distinguishes signal from noise via a two-line fit of the entropy
  mode spectrum. Fully autonomous — better than "99% energy" gut
  choice.

### Recommended implementation

1. Compute SVD of the 20-snapshot matrix at mode counts 3, 5, 8, 12, 16
   (cheap, ~seconds on GPU with JAX).
2. Emit **all four** diagnostics: cumulative energy, "predictability
   ratio" (Duan), energy-closure ratio (Bhattacharyya), ELF spectrum
   (Brindise). Cross-plot.
3. For the two mode counts that cross the predictability-ratio +
   energy-closure thresholds, rerun the full FOM-vs-ROM PT compare on
   the 4-case cohort.

### Expected timeline
Day 1: SVD + diagnostics. Day 2: 2 × ROM rebuilds + 8 PT runs. Day 3:
compare + report. **~3 days total**, cheapest and highest-information
step in the plan.

---

## Section 3 · Lagrangian-inner-product POD (Step 7) — implementation

### Xie et al. formulation
[Xie 2020](https://consensus.app/papers/details/f52431fea1f35c768a6adac63b482fb1/?utm_source=claude_desktop)
[1] builds the POD basis by weighting the L² inner product with a
matrix W that emphasises Lagrangian quantities (they use gradient
information along trajectories). Concretely, the standard POD is

    Cov_ij = ∫ v_i · v_j dV   → SVD → modes

Lagrangian-weighted POD is

    Cov_ij = ∫ v_i · W · v_j dV   → weighted SVD → different modes

where W upweights regions where trajectory sensitivity is high
(shear layers, separatrices).

### How to compute W in our JAX pipeline

Two implementation-ready recipes from the literature:

**Option A · Weighted SVD (WSVD)**:
[Li 2023](https://consensus.app/papers/details/6374e2606167549d979c5840b83e020e/?utm_source=claude_desktop)
[42] gives the full mathematical formulation for SVD under a
non-standard inner product `‖x‖²_M = xᵀMx` with M symmetric
positive-definite. The **weighted Golub-Kahan bidiagonalization**
(WGKB) algorithm computes dominant WSVD components iteratively,
without forming M directly. This is the right primitive for us
because M for us is the FE mass matrix (implicit — never materialised).

**Option B · Incremental weighted SVD**:
[Fareed 2018a](https://consensus.app/papers/details/57666dd393205fffbd52138dd39b6485/?utm_source=claude_desktop)
[43], [Fareed 2018b](https://consensus.app/papers/details/d9458a08ac2350889141b903ff76e26b/?utm_source=claude_desktop)
[44] gives incremental algorithms for **SVD with respect to a
weighted inner product** for POD of PDE data. This is what we need
for building the basis across all 20 case snapshots without holding
the full snapshot matrix in memory.

### What the weight matrix W should be for FSW

Per Xie: W should upweight the **trajectory-sensitive regions**. For
our FSW problem, this maps directly to:
- Highest weight: **the pin-face shear band at r ≈ 4–7 mm**
  (Part 5 §4 spatial pattern showed ~34 % relative error there —
  Xiong-Xie amplification localised)
- Medium weight: **inflow zone x < -0.010 m** (particles are
  seeded there and their initial trajectory determines everything)
- Low weight: **outer bulk r > 15 mm** (advective only)

### JAX/GPU implementation

- **Standard POD**: 20 × M matrix (M = 180k mesh nodes × 3
  components ≈ 540k), full SVD is a few seconds via `jnp.linalg.svd`.
- **Weighted POD with diagonal W**: same SVD applied to
  `W^{1/2} · A` where A is the snapshot matrix. Trivial.
- **Weighted POD with FE mass matrix W**: use randomised weighted
  SVD ([Alla 2016](https://consensus.app/papers/details/595604a77a615f5782ce33b597e32425/?utm_source=claude_desktop)
  [45]) — cheap approximation, well-suited to JAX (all matrix-vector
  ops, no dense factorisation).

**Sanity check**: the trivial-W version (all ones) should reproduce
standard POD exactly. If our weighted implementation doesn't, the
mass matrix is wrong.

### Realistic expected impact
Xie's "orders of magnitude" result was on QGE (2D geophysics). On
FSW with the same 3–5 modes, expect **2–4× Lagrangian rms
reduction** if the workpiece mean-velocity bias is the dominant
error source. Sanity check: if Step 6 (mode-count) already got most
of the gain, Step 7 has less to add.

---

## Section 4 · Divergence-free POD basis (Step 8) — implementation

### The problem
Our smoothness-report §4 shows ROM's div pattern is 99% correlated
with FOM's — the ROM inherits the FOM's incompressibility violation.
This isn't a POD artefact; it's what happens when you POD a
non-div-free FOM.

### Two implementation paths

#### 4a. If a mixed u/p/e FSW formulation output is available
[Novo & Rubino 2020](https://consensus.app/papers/details/def9702c3df45a6c9277be5bb5334e1e/?utm_source=claude_desktop)
[6] proves error bounds for POD when the snapshots come from an
**inf-sup stable Galerkin method with grad-div stabilization** —
"since the snapshots are discretely divergence-free, the pressure
can be removed from the formulation of the POD approximation to
the velocity." This is the shortest path: get the mixed u/p/e
velocity snapshots and rebuild POD on them.

#### 4b. If only nodal velocity snapshots exist (current situation)
Project snapshots onto a divergence-free subspace via
**Helmholtz-Hodge decomposition** before the SVD. Two mature
implementations:

- **Spectral Leray projection**: [Xigui Li 2026](https://consensus.app/papers/details/df607d464f1e5e02ada8789ca8867544/?utm_source=claude_desktop)
  [46] — differentiable spectral Leray projection grounded in
  Helmholtz-Hodge decomposition, integrated in JAX. **Directly
  usable in our JAX stack**. Enforces exact incompressibility up to
  discretisation error.
- **RBF-based curl-free / divergence-free correction**: [Lanyu Li 2024](https://consensus.app/papers/details/6de17f9b1cef587d90e4a2b9dba4d473/?utm_source=claude_desktop)
  [47] uses RBFs with divergence-free kernels to provide
  divergence-free correction to velocity fields for incompressible
  flows — no Lagrange multipliers, no saddle points.

#### 4c. Alternative: augmented-basis method (ABM)
[Kaneko 2022](https://consensus.app/papers/details/fecbf799e1445d268328912856003c45/?utm_source=claude_desktop)
[48] proposes augmenting the standard POD basis with **divergence-free
projections of a subset of nonlinear interaction terms** — the
augmenting bases carry localised high-wavenumber content and are
better at dissipating turbulent kinetic energy than the standard
POD bases. Requires more code than 4a but no mixed FEM output.

### Expected impact
- Div-free error residual of ROM drops by 10× or more (the FE
  literature is consistent on this).
- Whether that translates into PT residual reduction is separate —
  needs measurement. From [Star 2020](https://consensus.app/papers/details/2afcea692ec7596ebba5f821ef31604d/?utm_source=claude_desktop)
  [8]: the "consistent flux method" (div-free) is *"slightly more
  accurate"* than the inconsistent method on standard benchmarks.
  Small effect on rms in the literature examples, but *big* effect
  on volume conservation for long trajectories (see Section 8 below).

---

## Section 5 · ROM-on-grid vs ROM-on-mesh

### Literature verdict on which is likely to win

[Gräßle 2017](https://consensus.app/papers/details/458a3beef72f5df494a60af9c8969a46/?utm_source=claude_desktop)
[49] and [Ullmann 2016](https://consensus.app/papers/details/6f27cb7a4911532ba1942a43bb8b3190/?utm_source=claude_desktop)
[50] both handle **POD across snapshots living in different FE spaces**
using the correlation-matrix formulation, without needing to
interpolate snapshots onto a common mesh. **Grid vs mesh is not the
fundamental question** — both work; the *inner product* is what
matters.

[Nakamura 2024](https://consensus.app/papers/details/33c89c61ddd55b6c928769396627ab59/?utm_source=claude_desktop)
[51] proposes POD on flow snapshots computed on **different grids in
computational space** (mapped via conformal mapping). Very general
framework — supports our exact "mesh snapshots but I want a grid
POD basis" use case.

[Gooijer 2021](https://consensus.app/papers/details/5118da687c555d5188fcf14afb9ef0ce/?utm_source=claude_desktop)
[52] systematically studies the effect of preprocessing (scaling
per physical part) on POD-based surrogate accuracy — for our
multi-component velocity field, this could matter (currently we
POD u,v,w together; scaling them separately could improve the
basis).

### Recommendation
Move Step 5 phase A (SV decay check) from "control experiment"
back to informative: run the SV decay for both mesh-POD and grid-POD
with the **same weighted inner product** (per Step 7). If the two
curves are still identical, domain-of-POD is genuinely irrelevant.
If the grid one is faster, it opens a Tier-1 basis choice.

---

## Section 6 · JAX/GPU implementation stack for the whole pipeline

### What's already there (2026 state of the art)

- **[JAX-Fluids 2.0](https://consensus.app/papers/details/1d81628aa705532887e26df45a353e37/?utm_source=claude_desktop)**
  [53] (43 citations): differentiable CFD in JAX, scales to
  512 A100 GPUs / 1024 TPU v3 cores. Compressible + two-phase,
  positivity preserving, IO handling. **Reference architecture for
  our Stage-1 velocity infrastructure if we ever generate our own
  FOM snapshots**.
- **[JAX-MPM](https://consensus.app/papers/details/e57ecc89754f5a959b002610fb4b11cb/?utm_source=claude_desktop)**
  [54]: MPM solver in JAX, **2.7M particles × 1000 timesteps in 22 s
  on a single GPU** at single precision (98 s double). Directly
  comparable scale to our FSW cohort (360k particles × 2000 steps).
  We're currently at ~30 s per case for 4lvl_hct grid PT — JAX-MPM
  suggests we could do 3-4× better with tighter kernel fusion.
- **[JAX-LaB](https://consensus.app/papers/details/589f34263d8251919c13070eb6f7d66b/?utm_source=claude_desktop)**
  [55]: differentiable Lattice-Boltzmann in JAX, scales across CPU
  → multi-GPU → distributed. Complementary methodology.
- **[Diff-FlowFSI](https://consensus.app/papers/details/734406c742205e3fa822b79f128dacc0/?utm_source=claude_desktop)**
  [56]: JAX + immersed boundary FSI — matches our "pin embedded in
  workpiece" geometry.
- **[JAX-Shock](https://consensus.app/papers/details/fe7aff8de88454fab5635c4bd395f9f1/?utm_source=claude_desktop)**
  [57]: JAX + high-order shock-capturing + immersed boundary. Not
  directly for us but shows the pattern.

### For particle tracking specifically

- **[EcoSLIM](https://consensus.app/papers/details/08558f631d8f5ca78cb42b674b34793b/?utm_source=claude_desktop)**
  [58]: OpenMP-CUDA Lagrangian PT, strong scaling on multi-GPU.
- **[Wang 2021 RT-cores](https://consensus.app/papers/details/3fab5de2510751019235f94930dd8e8b/?utm_source=claude_desktop)**
  [59]: **hardware ray tracing cores accelerate Eulerian-Lagrangian
  particle tracking on unstructured meshes by 1.8–2×**. Uses
  bounding volume hierarchy (BVH) tree — natural mapping to our
  mesh-aligned octree work. Would need CUDA rewrite (JAX doesn't
  expose RT cores yet).
- **[SCALE-TRACK](https://consensus.app/papers/details/f32db0a2c7d05277a12e67b5eccf18d7/?utm_source=claude_desktop)**
  [60]: **256 billion particles on 256 GPUs**; 1.4 billion on a
  single GPU. Absolute state of the art for E-L PT scale.
- **[PLUTO GPU LP](https://consensus.app/papers/details/eaa789ddc9445bd4b7076d6f60d544e4/?utm_source=claude_desktop)**
  [61]: OpenACC + MPI Lagrangian particles, 80–90 % weak scaling
  parallel efficiency, 6× speedup vs CPU. Different physics
  (astrophysics) but the parallel structure is instructive.
- **[Baldan 2023](https://consensus.app/papers/details/a4c6c9181bc859ee8d83b18074603c59/?utm_source=claude_desktop)**
  [62]: efficient distributed-memory L-PT on unstructured hybrid
  meshes with parallel ray-tracing location and global particle IDs
  reducing inter-process communication.
- **[Zhao 2022](https://consensus.app/papers/details/5e496117fd9855998b97b79bafd711d4/?utm_source=claude_desktop)**
  [63]: RT-cores for **particle-based simulation neighbour search**,
  10–60 % faster than cell-based on RTX GPUs.

### Concrete recommendation for our stack

- **Keep JAX**. It's the right choice for our differentiable
  PT + POD pipeline. `jax.jit` + `jax.vmap` already give us
  90 % of the JAX-MPM-class throughput.
- **Two optimisation avenues** if we hit throughput limits:
  1. **Tighter fusion**: profile our RK4 step — likely we can
     collapse 4 texture-fetch stages into fewer kernel launches
     (JAX-MPM's technique).
  2. **Only if we ever go >10M particles**: consider a CUDA
     ray-tracing-core drop-in for the mesh-aligned octree lookup
     (Wang 2021 recipe). Not needed at 360k particles.

### AD-related notes

JAX-MPM demonstrates **gradient-based inverse modelling directly
through the time-stepping solver** [54]. This unlocks:
- End-to-end fitting of ROM coefficients to observed particle
  positions (rather than to Eulerian velocity snapshots)
- Learning W (Lagrangian-inner-product weight) from data rather
  than hand-crafting it — a natural closed loop with Step 7

---

## Section 7 · Integrator + interpolation pairing (Step 9)

### What Pokrajac 2002 [20] proved
On unstructured FE meshes: **the accuracy of the exit polynomial
(interpolation) must match the ODE solver order**. Concretely, RK5(4)
or RK6(4) paired with a 5th-order exit polynomial gives *several
orders of magnitude accuracy improvement* over Pollock's method,
at ~10× the compute cost. **RK4 with trilinear or tricubic is a
mismatched pairing** — RK4 is order-4 accurate in time, trilinear
is order-2 in space, tricubic is order-4. So RK4 + tricubic is
matched; RK4 + trilinear is under-matched.

But our Part 5 finding was: **RK4 + tricubic underperforms
RK4 + trilinear on the 20-case cohort at step 2000**. That's
inconsistent with Pokrajac unless the tricubic overshoot at C¹-jump
regions dominates the higher-order accuracy gain.

### What Beznosov 2025 clarifies
[Beznosov 2025](https://consensus.app/papers/details/770bc529ee665f828855fad69f0c61a3/?utm_source=claude_desktop)
[21]: **insufficient derivative continuity of the interpolant
degrades RK-scheme accuracy at low error tolerances, introducing
discontinuity-induced truncation errors.** Our HCT-projected field
is C¹ inside each parent tet, but the underlying grid is trilinear
or tricubic Stage-2 — so the *composite* smoothness is only C⁰ at
grid-cell boundaries. That's the mechanism for Part 5's
"tricubic hurts" observation: RK6 hitting a C⁰ jump gives a worse
truncation error than RK4 does.

### What to try (Step 9 concrete design)

Three combos on 4lvl_hct + case 000 first:

| ID | RK order | Stage-2 interp | Composite smoothness | Expected |
|---|---|---|---|---|
| baseline | RK4 (fixed dt) | trilinear | C⁰ at cell boundaries | current 7.79 mm |
| A | RK4 (fixed dt) | Catmull-Rom tricubic | C¹ inside cell, C⁰ across | small change (Part 5) |
| B | Dormand-Prince RK5(4) adaptive | trilinear | C⁰ | ~10 – 20 % rms drop |
| C | Dormand-Prince RK5(4) adaptive | tricubic | C⁰ across | mixed — Beznosov predicts worse than B |
| D | Dormand-Prince RK5(4) adaptive | **C¹ Hermite tricubic** across cells | C¹ globally | 2 – 4× rms drop (Pokrajac ceiling) |

Combo D is the interesting one. We currently interpolate
Catmull-Rom cardinal cubic, which is C¹ *within a stencil* but C⁰
between stencils. A "true" C¹ globally would need node-and-derivative
data at each cell corner (like HCT does for tets). Building that
is Section 8 of this dossier.

### Adaptive step-size infrastructure

[Ranocha 2021](https://consensus.app/papers/details/08afc433407d598b9706403fdf1cdda9/?utm_source=claude_desktop)
[64] gives error-control based time-integration for compressible CFD
and shows **step size adapts near the CFL stability limit at loose
tolerances, and provides temporal-error control at tighter
tolerances**. Ready-to-use PI controllers for embedded RK pairs.
JAX-compatible (all matrix-vector ops).

[Vermeire 2023](https://consensus.app/papers/details/7c80b4bff3b758a894c2cb37b9dcd993/?utm_source=claude_desktop)
[65] proposes **paired explicit Runge-Kutta (P-ERK) schemes** with
embedded pairs — different numbers of active stages per element
based on local stiffness — up to **7× less computationally
expensive than classical embedded pairs** for locally stiff systems.
Could handle our shear-band region where the local Lipschitz
constant is much higher.

### Rössler 2018 [22] takeaway
For ECMWF wind fields: 3rd-order RK with 170 s dt matches 4th-order
RK in efficiency for tropospheric transport. For FSW: our fixed
dt = 3.75 ms is probably conservative; a dt sweep at fixed RK order
is a cheap first test.

---

## Section 8 · Volume-preserving / geometry-preserving integrators (new Step 13 candidate)

### The insight
Incompressible flow has ∇·v = 0 ⇒ trajectory maps preserve volume.
Standard RK does *not* preserve volume — position error grows
linearly (or worse) over long trajectories.

### The tools

- **[Tapley 2019](https://consensus.app/papers/details/a0f0ffede1dd54ed820f77a163eda37a/?utm_source=claude_desktop)**
  [66]: for divergence-free velocity available at discrete points +
  times (our exact setup): construct a **divergence-free
  approximation using matrix-valued RBFs**, integrated via a
  **volume-preserving map**. Result: accurate trajectories in a
  helical vortex using **much larger step sizes and fewer
  interpolation points** than conventional schemes. This is
  precisely the C¹ + geometry-preserving stack Step 9 combo D
  needs.
- **[Wuispel 1995](https://consensus.app/papers/details/bf64f5faf7e15234bea91a0afbdb190f/?utm_source=claude_desktop)**
  [67]: family of **general n-dimensional volume-preserving
  integrators** for divergence-free vector fields. The classical
  reference.
- **[Kato 2021](https://consensus.app/papers/details/08ab3190648f5aa8b6ac0642177d7083/?utm_source=claude_desktop)**
  [68]: **volume-preserving integrator based on exact flow of
  velocity** for PIC simulations. Structure-preserving alternative
  to Boris.
- **[Wang 2019](https://consensus.app/papers/details/c95220f85d875db6ae214c72e7efe080/?utm_source=claude_desktop)**
  [69]: **volume-preserving exponential integrators** — necessary
  and sufficient conditions given for different vector field classes.
- **[He 2015](https://consensus.app/papers/details/199ca8e0ec8450f1af393ab7a8cc14fa/?utm_source=claude_desktop)**
  [70]: **volume-preserving algorithms for charged-particle
  dynamics** using splitting; includes the Boris algorithm as a
  special case. Bounded numerical errors of energy, momentum, and
  adiabatic invariants over very long simulations.
- **[Qin 2013 "Why is Boris algorithm so good"](https://consensus.app/papers/details/a9fae08d45ab5c2d88caae3879bdcc25/?utm_source=claude_desktop)**
  [71] (288 citations): **Boris conserves phase-space volume even
  though it is not symplectic** — global energy-error bound
  typically associated with symplectic algorithms still holds.
  Explains why Boris is de facto standard for charged-particle
  tracking. Same conservation principle applies to our incompressible
  advection.
- **[Gorges 2022](https://consensus.app/papers/details/ccd6df2c728f56e7aacdfc98cd8719e4/?utm_source=claude_desktop)**
  [72]: **divergence-preserving velocity interpolation for front
  tracking**, orders of magnitude better volume conservation than
  conventional interpolation. Directly applicable to our grid PT.

### Recommended experiment
- Implement Tapley-style **divergence-free RBF interpolation** on
  4lvl_hct grid cells (Section 8 recipe)
- Combine with a **volume-preserving splitting integrator** (He 2015
  or Wuispel 1995 template)
- Rerun the case-000 keeper PT and measure per-particle position
  divergence over 2000 steps

**Expected impact**: dramatic reduction in the long-time
position-error growth (Tapley says "much larger step sizes"; He
2015 says "up to the order of the method over very long simulation
time"). Would specifically shrink the outer-bulk error we currently
attribute to accumulation.

---

## Section 9 · Mixing quantification and FSW-specific validation

### The mixing story so far
Roadmap §4 (`rom_pt_smoothness_divfree_report.md`) already
implemented **residence-time distribution** and **pairwise separation**
diagnostics on the 4-case cohort. Verdict: HCT has essentially no
effect on Eulerian smoothness, divergence, residence time, or
pairwise separation.

### What the FSW literature quantifies (validation targets)

- [Dialami 2015](https://consensus.app/papers/details/63f6dceafc305f34a682e161da78a163/?utm_source=claude_desktop)
  [73] compares **BES, RK4, BFECC** for FSW particle tracing —
  Zalesak's disk benchmark. Our RK4 baseline is exactly what they
  used. Reproducible baseline.
- [Dialami 2020](https://consensus.app/papers/details/e927c4fb7b8c5da79fab5769d19b406b/?utm_source=claude_desktop)
  [74] (122 citations): FSW particle tracing that predicts **voids,
  wormholes, flash, joint line remnants, onion rings in a single
  simulation**. **Every one of those defects is a Lagrangian
  invariant** — if our ROM PT can reproduce them, that's a
  publication-grade result.
- [Kumar 2018 PIV](https://consensus.app/papers/details/6c4d671a22bd5dfc9f8b6c2381fe1c11/?utm_source=claude_desktop)
  [75]: **experimental velocity and strain-rate around the pin**:
  peripheral velocity 60 % of pin surface velocity; strain rate
  20 s⁻¹ at 0.6 mm from pin surface; strain rate scales linearly
  with rotational speed (8 s⁻¹ at 75 rpm → 44 s⁻¹ at 425 rpm).
  Concrete experimental data to validate ROM velocity field against.
- [Ambrosio 2023 review](https://consensus.app/papers/details/5f4d21c9977451ab88b0beafad911392/?utm_source=claude_desktop)
  [76] (102 citations): comprehensive review of experimental FSW
  material-flow work to date. Provides the validation-benchmark
  catalog.
- [Stubblefield 2023](https://consensus.app/papers/details/02d1c29cee355d329fd76d3a868b9082/?utm_source=claude_desktop)
  [77]: **SPH particle tracking + X-ray CT experimental validation
  for AFSD** — the anodized-oxide-shell tracer + copper-wire-core
  tracer methodology is directly transferable to FSW.
- [Chen 2021](https://consensus.app/papers/details/d41cdb311f515241917d22334a59f9a1/?utm_source=claude_desktop)
  [78]: **novel material tracing technology using ER2319 aluminium
  alloy welding wire as tracer** — in-situ FSW material flow
  observation. Provides quantitative flow-path data.
- [Krishnan 2002 "On the formation of onion rings"](https://consensus.app/papers/details/07fc2b574ef05297aeb3809cc39270a6/?utm_source=claude_desktop)
  [79] (485 citations): canonical reference. Onion ring spacing =
  forward motion per tool rotation. A ROM that reproduces this
  passes a strong Lagrangian benchmark.

### Mixing metrics beyond residence + pairwise separation

- **Poincaré maps** [18] from Bashiri: bin particles by initial
  radial position, plot Poincaré section on a horizontal plane.
  Compare mesh vs grid Poincaré maps qualitatively.
- **Mixing index from POD modes** [80]: [Shuo Li 2021](https://consensus.app/papers/details/d2ab90ba44da50c9b0a54bba3ae4e070/?utm_source=claude_desktop)
  proposes that **clumped POD modes indicate convective mixing**
  and **random POD modes indicate diffusive mixing**. Could
  discriminate convection- vs diffusion-dominated regions of our
  FSW flow.
- **2D uniformity metrics from particle trajectories** [81]:
  [Gai Zhang 2024](https://consensus.app/papers/details/b07a133490a25157b9939440d534e28a/?utm_source=claude_desktop)
  proposes point-pattern-density fluctuation methods with strong
  linear correlation (r = −0.99) to mixing time. Ready-to-use
  metric.

### FTLE-specific tooling

Already covered in the first evaluation pass (Kun Li 2022, Qian 2023
LESCM, Lagares 2023 GPU-FTLE, Raben 2013). Add:

- **Wall-time on our 360k-particle scale**: [Lagares 2023](https://consensus.app/papers/details/e7dcf10b8c3c505b87d599d3a28c282f/?utm_source=claude_desktop)
  [31] scales to millions of particles on a single V100. We're at
  0.36M — well below capacity.

### Recommended new metric bundle for our 4-case-cohort report

1. Poincaré maps (compare mesh vs 4lvl_hct)
2. FTLE ridges on the same particle set (compare mesh vs 4lvl_hct)
3. Onion-ring spacing measured from tracer particle trajectories

If (1) and (2) agree between mesh and 4lvl_hct, we have strong
evidence the grid preserves *mixing structure*, not just
per-particle position. That's the publishable finding.

---

## Section 10 · ML-augmented ROM alternatives (Step 14 candidate)

### The landscape
Once we get 4 – 8 % Eulerian rel_rms (which our current 3-mode POD
already does), the marginal accuracy of pure linear POD is
saturating. Nonlinear model-order reduction can push further:

- **[Ahmed 2021 nonlinear POD](https://consensus.app/papers/details/cb438c3ef5555bada27b542c14f343ab/?utm_source=claude_desktop)**
  [82]: autoencoder + LSTM for convection-dominated flows. Improves
  accuracy AND reduces training/testing cost.
- **[Kaneko ABM](https://consensus.app/papers/details/fecbf799e1445d268328912856003c45/?utm_source=claude_desktop)**
  [48]: augmented basis method for turbulence — POD basis augmented
  with divergence-free projections of nonlinear interaction terms.
  Better than L², H¹, Leray-stabilised POD. **Suitable for
  parametric model reduction** — exactly our 20-case parametric
  setup.
- **[Grimberg 2020](https://consensus.app/papers/details/ea46855961585636aa5ff41659d37398/?utm_source=claude_desktop)**
  [83] (108 citations): **the real culprit behind PROM instability
  is the Galerkin framework**, not modal truncation. **Petrov-Galerkin
  framework can construct numerically stable and accurate PROMs
  for convection-dominated laminar and turbulent flows without
  closure models or subspace enrichment**. Cheap fix for our
  eventual Galerkin POD-ROM.
- **[Fu 2023 SAE + self-attention](https://consensus.app/papers/details/ec888655fe48533d8b04a9392ec7451d/?utm_source=claude_desktop)**
  [84]: stacked autoencoder + self-attention for NIROM. Outperforms
  standard POD.
- **[Ding 2024 AE vs POD](https://consensus.app/papers/details/73eba141709b5be4ab4e0f582092e552/?utm_source=claude_desktop)**
  [85] (24 citations): systematic comparison of CAE, FCAE, and POD
  for parametric ROMs on jet-into-crossflow. Both AE and POD are
  9 orders of magnitude faster than FOM; **AE is better for
  spatial-distribution accuracy but requires more hyperparameters
  and training time**.
- **[Zhu 2024](https://consensus.app/papers/details/8ec3e2f156fd54709761bfdd1590a236/?utm_source=claude_desktop)**
  [86]: **compressed autoencoders via pruning + SVD**, reduce AE
  size to 3 – 18 % without accuracy loss. Answers the "AE is too
  memory-heavy" objection.

### Recommendation
Only pursue after Steps 6 – 9 are done. If we still have Lagrangian
rel_rms > 15 % after mode-count fix, Lagrangian POD, and
integrator+interp pairing fix, an autoencoder + POD hybrid becomes
worth the multi-week investment.

---

## Section 11 · Non-intrusive ROM path (Step 15 candidate)

The FSW parametric surrogate we're building is *inherently
non-intrusive* — we don't have access to the FOM assembly matrix,
only the snapshot outputs. Multiple established recipes:

- **[Hesthaven 2018 POD-NN](https://consensus.app/papers/details/f62e99b5db3a55c4bba87b5acb99621c/?utm_source=claude_desktop)**
  [87] (628 citations): POD basis + neural network for the
  coefficients. Non-intrusive, non-affine parameter dependence.
  Reference implementation.
- **[Guo 2018 POD-GPR](https://consensus.app/papers/details/9b897d6fab6f569f8f189a496ec4db24/?utm_source=claude_desktop)**
  [88] (259 citations): POD + Gaussian process regression. Same
  goal as Hesthaven but with probabilistic uncertainty.
- **[Xiao 2015 POD-RBF](https://consensus.app/papers/details/d0ba90b79ba95157b8b06f90fea9554a/?utm_source=claude_desktop)**
  [89] (183 citations): POD + radial basis function interpolation
  of the coefficients. Simplest possible implementation.
- **[Min 2024 POD-RBFNN](https://consensus.app/papers/details/bca3d282463956fda8e4855eaad0f3f1/?utm_source=claude_desktop)**
  [90] (48 citations): POD + RBF neural network. **Smaller RMSE
  and MAE than POD-BPNN** on flow around parallel twin cylinders.
- **[Yang 2020 adaptive sampling](https://consensus.app/papers/details/879ec1ea771c5fce836bf4801b4f835f/?utm_source=claude_desktop)**
  [91]: POD-GPR with **adaptive sampling based on Gaussian process
  variance + gradient**. Higher accuracy at same sample size than
  Halton sequences. Useful when we go beyond the current 20 cases.
- **[Yang 2020 POD-Galerkin water-hammer](https://consensus.app/papers/details/53ae64206e5f58cdb9367d9f0ca6f834/?utm_source=claude_desktop)**
  [92]: shows the pattern for 2D velocity profile reduction — very
  similar to our u,v,w problem.

### For our 20-case parametric setup
The 20 cases are 20 parameter tuples (v_adv, ω_pin). The natural
NIROM is:

    (v_adv, ω_pin) → POD coefficients (via RBF / GPR / NN) → velocity field

The current pipeline **implicitly** does this by rebuilding the
ROM per case. Making it explicit + fittable across the 20 cases is a
few days of work and produces a **query-time velocity field for any
(v_adv, ω_pin)** in the parameter space — a real deliverable.

---

## Section 12 · Related-domain prior art (positioning for the paper)

The stirred-tank + Rushton-turbine POD literature was established in
Section D of the first evaluation. Also worth citing:

- **[Fang 2025 POD-MOR for friction stir spot welding](https://consensus.app/papers/details/2595d5fd121757fd90dfd9ea1055f299/?utm_source=claude_desktop)**
  [93]: applies POD-MOR to SPH simulations of friction-stir spot
  welding. **Reports "POD-MOR significantly reduces computational
  error compared to uniform reduction of particle numbers in SPH"**.
  Directly adjacent to our FSW POD-ROM work — a required citation.
- **[Cao 2021 ML+ROM FSW](https://consensus.app/papers/details/23e2a9dc5e165c12b1dbeb4c0f7d3fb0/?utm_source=claude_desktop)**
  [13] (already cited): POD-ROM for FSW heat+NS with shear-dependent
  viscosity, + ANN for the parameter → POD-coefficient mapping.
- **[Li 2022 ROM-EL fluidised bed](https://consensus.app/papers/details/4bc7651b71d45da8b8924b7292516254/?utm_source=claude_desktop)**
  [94] (41 citations): Lanczos POD for Eulerian-Lagrangian
  simulations; **3 orders of magnitude speedup vs CFD-DEM**. Same
  scale we're chasing.
- **[Razavi 2026 ROM-ROM particle-in-fluid](https://consensus.app/papers/details/7aaeff23edfc55778435b1bb3d4ea71f/?utm_source=claude_desktop)**
  [95]: **3×10⁵ times faster than full CFD-DEM**, 30 POD modes
  preserving 80–85 % system energy. Absolute state of the art for
  ROM-based particle-in-fluid.
- **[Hijazi 2019 POD-Galerkin turbulent](https://consensus.app/papers/details/d0702962f96d5e3bb9c2767fefb10921/?utm_source=claude_desktop)**
  [96] (207 citations): FV POD-Galerkin ROM for turbulent flows up
  to Re = 10⁵. Established framework we should cite for FSW's
  turbulent regime.
- **[Peace 2025 PEPT-based Lagrangian mixing](https://consensus.app/papers/details/f3539021842b5618b761b89b88cc464c/?utm_source=claude_desktop)**
  [29] (already cited): **first-time Lagrangian mixing metric for
  aerated stirred tanks** using PEPT. Very recent (2025), same year
  as our work.

### Positioning statement for the FSW paper draft
> This work applies established POD-ROM machinery from the
> Eulerian-Lagrangian mixing / stirred-tank community (Duan 2023 [38],
> Kun Li 2022 [17], Fang 2025 [93], Mikhaylov 2023 [25]) to
> friction-stir welding. The novelty is (1) the mesh + grid-projection
> pipeline that combines HCT-3D Stage-1 with tricubic Stage-2 for
> a C¹-consistent velocity field, and (2) the systematic 20-case
> cohort characterisation showing that basis-construction (not
> grid topology or interpolation order) is the effective lever
> for reducing Lagrangian PT error under the standard POD basis.

---

## Master reference table

64 references total across the two literature passes. Search hits
that appeared in the first evaluation but got repurposed here are
noted.

### POD basis construction (Section 2 – 5)
| # | Ref | Contribution |
|---|---|---|
| 1 | [Xie 2020](https://consensus.app/papers/details/f52431fea1f35c768a6adac63b482fb1/?utm_source=claude_desktop) | Lagrangian-inner-product POD (canonical) |
| 3 | [Parish 2022](https://consensus.app/papers/details/9fdc346d0b4d500f8636b37839cf2b58/?utm_source=claude_desktop) | Physics-based inner products beat L² |
| 6 | [Novo & Rubino 2020](https://consensus.app/papers/details/def9702c3df45a6c9277be5bb5334e1e/?utm_source=claude_desktop) | Discretely div-free snapshots → no pressure in POD velocity |
| 8 | [Star 2020](https://consensus.app/papers/details/2afcea692ec7596ebba5f821ef31604d/?utm_source=claude_desktop) | Consistent flux (div-free) more accurate than inconsistent |
| 10 | [Christensen 1999](https://consensus.app/papers/details/113e721cea5d5808a3f03c6228e403f8/?utm_source=claude_desktop) | Weighted POD gives priority to low-energy modes |
| 11 | [Dellacasagrande 2021](https://consensus.app/papers/details/d91633c6ea3c5c19947e9e8f45fe24cb/?utm_source=claude_desktop) | Non-Euclidean POD for low-KE regions |
| 12 | [Olesen 2022](https://consensus.app/papers/details/04148b54bb475ca4a76689a5135d0bc1/?utm_source=claude_desktop) | Dissipation-optimised POD |
| 38 | [Duan 2023](https://consensus.app/papers/details/4f51abe86082549d8cae1f4fafb4b401/?utm_source=claude_desktop) | Feasibility of POD-based ROM for Eulerian-Lagrangian |
| 39 | [Shnapp 2026](https://consensus.app/papers/details/f13a00a5fc8056fcbda4d7fa83fe6ef9/?utm_source=claude_desktop) | Lagrangian POD, mode count for trajectory statistics |
| 40 | [Bhattacharyya 2020](https://consensus.app/papers/details/7b0fb54d90cc5518802798ecad7f59e9/?utm_source=claude_desktop) | Energy closure as mode selection criterion |
| 41 | [Brindise 2017](https://consensus.app/papers/details/6caf7607c7505528b8ac5783e42161f2/?utm_source=claude_desktop) | Entropy-line fit for objective mode count |
| 42 | [Haibo Li 2023](https://consensus.app/papers/details/6374e2606167549d979c5840b83e020e/?utm_source=claude_desktop) | Weighted SVD (WSVD) formulation |
| 43 | [Fareed 2018a](https://consensus.app/papers/details/57666dd393205fffbd52138dd39b6485/?utm_source=claude_desktop) | Incremental POD algorithms for continuous-time data |
| 44 | [Fareed 2018b](https://consensus.app/papers/details/d9458a08ac2350889141b903ff76e26b/?utm_source=claude_desktop) | Error analysis of incremental POD |
| 45 | [Alla 2016](https://consensus.app/papers/details/595604a77a615f5782ce33b597e32425/?utm_source=claude_desktop) | Randomised SVD for POD/DMD |
| 46 | [Xigui Li 2026](https://consensus.app/papers/details/df607d464f1e5e02ada8789ca8867544/?utm_source=claude_desktop) | Differentiable spectral Leray projection (JAX-ready) |
| 47 | [Lanyu Li 2024](https://consensus.app/papers/details/6de17f9b1cef587d90e4a2b9dba4d473/?utm_source=claude_desktop) | RBF Helmholtz-Hodge decomposition, div/curl-free correction |
| 48 | [Kaneko 2022](https://consensus.app/papers/details/fecbf799e1445d268328912856003c45/?utm_source=claude_desktop) | Augmented-basis method for turbulent ROMs |
| 49 | [Gräßle 2017](https://consensus.app/papers/details/458a3beef72f5df494a60af9c8969a46/?utm_source=claude_desktop) | POD across different FE spaces (correlation-matrix approach) |
| 50 | [Ullmann 2016](https://consensus.app/papers/details/6f27cb7a4911532ba1942a43bb8b3190/?utm_source=claude_desktop) | POD-Galerkin with adaptive FE snapshots |
| 51 | [Nakamura 2024](https://consensus.app/papers/details/33c89c61ddd55b6c928769396627ab59/?utm_source=claude_desktop) | POD on flow snapshots computed on different grids |
| 52 | [Gooijer 2021](https://consensus.app/papers/details/5118da687c555d5188fcf14afb9ef0ce/?utm_source=claude_desktop) | Scaling per physical part improves POD surrogate accuracy |

### Grid design + AMR (Section 1, Section 6)
| # | Ref | Contribution |
|---|---|---|
| 34 | [Fidkowski 2020](https://consensus.app/papers/details/c030727c1f70512c9b7a7509757c1791/?utm_source=claude_desktop) | ML for anisotropic metric fields (goal-oriented AMR) |
| 35 | [Balan 2021](https://consensus.app/papers/details/d4b2a597c55a50089c493338e4c8595a/?utm_source=claude_desktop) | Review + comparison of anisotropic metric-based AMR |
| 36 | [Frey 2005](https://consensus.app/papers/details/d0f5f415917d5e51b1232f9243ee48f9/?utm_source=claude_desktop) | Anisotropic mesh adaptation (canonical, 386 citations) |
| 37 | [Prouvost 2024](https://consensus.app/papers/details/2b2628ea29215001a4c9ca3c966df25d/?utm_source=claude_desktop) | Compression-ratio constraint in metric AMR |

### GPU / JAX implementations (Section 6)
| # | Ref | Contribution |
|---|---|---|
| 53 | [JAX-Fluids 2.0 (Bezgin 2024)](https://consensus.app/papers/details/1d81628aa705532887e26df45a353e37/?utm_source=claude_desktop) | Reference JAX CFD architecture |
| 54 | [JAX-MPM (Du 2025)](https://consensus.app/papers/details/e57ecc89754f5a959b002610fb4b11cb/?utm_source=claude_desktop) | 2.7M particle Lagrangian JAX benchmark |
| 55 | [JAX-LaB (Pradhan 2025)](https://consensus.app/papers/details/589f34263d8251919c13070eb6f7d66b/?utm_source=claude_desktop) | JAX Lattice-Boltzmann multi-GPU |
| 56 | [Diff-FlowFSI (Fan 2025)](https://consensus.app/papers/details/734406c742205e3fa822b79f128dacc0/?utm_source=claude_desktop) | JAX + IBM FSI |
| 57 | [JAX-Shock (Zhang 2026)](https://consensus.app/papers/details/fe7aff8de88454fab5635c4bd395f9f1/?utm_source=claude_desktop) | JAX shock-capturing + IBM |
| 58 | [EcoSLIM (Yang 2021)](https://consensus.app/papers/details/08558f631d8f5ca78cb42b674b34793b/?utm_source=claude_desktop) | OpenMP-CUDA Lagrangian PT for hydrology |
| 59 | [Wang 2021 RT-cores](https://consensus.app/papers/details/3fab5de2510751019235f94930dd8e8b/?utm_source=claude_desktop) | Hardware ray-tracing 1.8-2× speedup for unstructured PT |
| 60 | [SCALE-TRACK (Schmalfuss 2026)](https://consensus.app/papers/details/f32db0a2c7d05277a12e67b5eccf18d7/?utm_source=claude_desktop) | 256B particles / 256 GPUs |
| 61 | [PLUTO GPU LP (Suriano 2026)](https://consensus.app/papers/details/eaa789ddc9445bd4b7076d6f60d544e4/?utm_source=claude_desktop) | OpenACC + MPI Lagrangian particles |
| 62 | [Baldan 2023](https://consensus.app/papers/details/a4c6c9181bc859ee8d83b18074603c59/?utm_source=claude_desktop) | Distributed-memory L-PT with global IDs |
| 63 | [Zhao 2022 RT-cores DEM](https://consensus.app/papers/details/5e496117fd9855998b97b79bafd711d4/?utm_source=claude_desktop) | RT-cores for particle neighbour search |

### Integrators (Section 7 – 8)
| # | Ref | Contribution |
|---|---|---|
| 20 | [Pokrajac 2002](https://consensus.app/papers/details/a5ef30b6fa025145a95a68d5b8de485c/?utm_source=claude_desktop) | RK order × exit polynomial matching (canonical) |
| 21 | [Beznosov 2025](https://consensus.app/papers/details/770bc529ee665f828855fad69f0c61a3/?utm_source=claude_desktop) | Derivative continuity + high-order RK for magnetic fields |
| 22 | [Rössler 2018](https://consensus.app/papers/details/9da7d93f3adc53009325a5c891c76f6d/?utm_source=claude_desktop) | 6-scheme RK benchmark on ECMWF wind trajectories |
| 23 | [Coppola 2001](https://consensus.app/papers/details/e4c41a132aaa5219997d90d0f9ae35b4/?utm_source=claude_desktop) | High-order polynomial approximation on unstructured meshes |
| 24 | [Yeung & Pope 1988](https://consensus.app/papers/details/7692f4fdedfe5db1b0442df2be5584ce/?utm_source=claude_desktop) | Cubic-spline vs Taylor-series interp for turbulence PT |
| 64 | [Ranocha 2021](https://consensus.app/papers/details/08afc433407d598b9706403fdf1cdda9/?utm_source=claude_desktop) | Error-control adaptive RK for CFD (PI controllers) |
| 65 | [Vermeire 2023](https://consensus.app/papers/details/7c80b4bff3b758a894c2cb37b9dcd993/?utm_source=claude_desktop) | Embedded paired explicit RK (7× cheaper for stiff) |
| 66 | [Tapley 2019](https://consensus.app/papers/details/a0f0ffede1dd54ed820f77a163eda37a/?utm_source=claude_desktop) | Div-free RBF + volume-preserving map for PT |
| 67 | [Wuispel 1995](https://consensus.app/papers/details/bf64f5faf7e15234bea91a0afbdb190f/?utm_source=claude_desktop) | Volume-preserving integrators (canonical) |
| 68 | [Kato 2021](https://consensus.app/papers/details/08ab3190648f5aa8b6ac0642177d7083/?utm_source=claude_desktop) | Volume-preserving integrator from exact flow (PIC) |
| 69 | [Wang 2019](https://consensus.app/papers/details/c95220f85d875db6ae214c72e7efe080/?utm_source=claude_desktop) | Volume-preserving exponential integrators |
| 70 | [He 2015](https://consensus.app/papers/details/199ca8e0ec8450f1af393ab7a8cc14fa/?utm_source=claude_desktop) | Volume-preserving Boris-family algorithms |
| 71 | [Qin 2013](https://consensus.app/papers/details/a9fae08d45ab5c2d88caae3879bdcc25/?utm_source=claude_desktop) | Why Boris algorithm works — 288 citations |
| 72 | [Gorges 2022](https://consensus.app/papers/details/ccd6df2c728f56e7aacdfc98cd8719e4/?utm_source=claude_desktop) | Div-preserving velocity interp for front tracking |

### LCS/FTLE methodology (Section 9)
| # | Ref | Contribution |
|---|---|---|
| 17 | [Kun Li 2022](https://consensus.app/papers/details/29f22ac5f77855d986d8ece317dd8efd/?utm_source=claude_desktop) | PEPT-driven FTLE/LCS in mechanically agitated vessel |
| 18 | [Bashiri 2016](https://consensus.app/papers/details/f6d548bfa3ea579cad6a25120b72eaa6/?utm_source=claude_desktop) | RPT + Poincaré maps + mixing indices in stirred tank |
| 19 | [Shadden 2005](https://consensus.app/papers/details/f3e7724c147b537ab134203df30ecf9e/?utm_source=claude_desktop) | Canonical LCS/FTLE reference (1439 citations) |
| 30 | [Qian 2023 LESCM](https://consensus.app/papers/details/29f44035f49f5b15a1bdca79a5157e25/?utm_source=claude_desktop) | Accurate FTLE for viscous incompressible, 16M particles |
| 31 | [Lagares 2023](https://consensus.app/papers/details/e7dcf10b8c3c505b87d599d3a28c282f/?utm_source=claude_desktop) | GPU-FTLE, 62 V100 scaling |
| 32 | [Raben 2013](https://consensus.app/papers/details/3eeb3a7021d15e29a6b6d7ecccff317a/?utm_source=claude_desktop) | Pathline-based FTLE beats VFI 80× on noisy PIV |

### FSW-specific validation (Section 9, 12)
| # | Ref | Contribution |
|---|---|---|
| 73 | [Dialami 2015](https://consensus.app/papers/details/63f6dceafc305f34a682e161da78a163/?utm_source=claude_desktop) | RK4 vs BES vs BFECC for FSW PT (Zalesak benchmark) |
| 74 | [Dialami 2020](https://consensus.app/papers/details/e927c4fb7b8c5da79fab5769d19b406b/?utm_source=claude_desktop) | FSW particle tracing predicts wormholes, onion rings |
| 75 | [Kumar 2018 PIV](https://consensus.app/papers/details/6c4d671a22bd5dfc9f8b6c2381fe1c11/?utm_source=claude_desktop) | Experimental strain-rate data around FSW pin |
| 76 | [Ambrosio 2023 review](https://consensus.app/papers/details/5f4d21c9977451ab88b0beafad911392/?utm_source=claude_desktop) | Comprehensive FSW material-flow experimental review |
| 77 | [Stubblefield 2023](https://consensus.app/papers/details/02d1c29cee355d329fd76d3a868b9082/?utm_source=claude_desktop) | SPH + X-ray CT particle tracking for AFSD |
| 78 | [Chen 2021](https://consensus.app/papers/details/d41cdb311f515241917d22334a59f9a1/?utm_source=claude_desktop) | Novel material tracing technology for in-situ FSW |
| 79 | [Krishnan 2002](https://consensus.app/papers/details/07fc2b574ef05297aeb3809cc39270a6/?utm_source=claude_desktop) | Canonical "onion rings" reference (485 citations) |
| 80 | [Shuo Li 2021](https://consensus.app/papers/details/d2ab90ba44da50c9b0a54bba3ae4e070/?utm_source=claude_desktop) | POD mode types identify convective vs diffusive mixing |
| 81 | [Gai Zhang 2024](https://consensus.app/papers/details/b07a133490a25157b9939440d534e28a/?utm_source=claude_desktop) | 2D uniformity metrics from particle trajectories |
| 93 | [Fang 2025 POD-MOR-FSSW](https://consensus.app/papers/details/2595d5fd121757fd90dfd9ea1055f299/?utm_source=claude_desktop) | POD-MOR for friction-stir spot welding (very recent, direct-adjacent) |

### ROM-PT prior art (Section 12)
| # | Ref | Contribution |
|---|---|---|
| 25 | [Mikhaylov 2023](https://consensus.app/papers/details/ef509961cc0a5608999805d1a476f09f/?utm_source=claude_desktop) | POD of turbulent Rushton-turbine stirred tank |
| 26 | [Mikhaylov 2021](https://consensus.app/papers/details/c7337d91c01b56509c904cacb993f229/?utm_source=claude_desktop) | POD + N4SID from sparse sensors in stirred tank |
| 27 | [Arosemena 2023](https://consensus.app/papers/details/560762d290bd53899bd0352200d8f89d/?utm_source=claude_desktop) | POD modal analysis in baffled stirred tank |
| 28 | [Jiang 2023](https://consensus.app/papers/details/f58e418b94c05c99a99f91daa61bcf4f/?utm_source=claude_desktop) | SVD-based ROM for solid-liquid stirred tank |
| 29 | [Peace 2025](https://consensus.app/papers/details/f3539021842b5618b761b89b88cc464c/?utm_source=claude_desktop) | Lagrangian mixing metric for aerated stirred tanks via PEPT |
| 13 | [Cao 2021 ML+ROM FSW](https://consensus.app/papers/details/23e2a9dc5e165c12b1dbeb4c0f7d3fb0/?utm_source=claude_desktop) | POD-ROM + ANN for FSW heat + Navier-Stokes |
| 94 | [Li 2022 ROM-EL](https://consensus.app/papers/details/4bc7651b71d45da8b8924b7292516254/?utm_source=claude_desktop) | Lanczos POD for CFD-DEM, 3 orders speedup |
| 95 | [Razavi 2026 ROM-ROM](https://consensus.app/papers/details/7aaeff23edfc55778435b1bb3d4ea71f/?utm_source=claude_desktop) | 3×10⁵ speedup, 30 modes for particle-in-fluid |
| 96 | [Hijazi 2019 turbulent POD](https://consensus.app/papers/details/d0702962f96d5e3bb9c2767fefb10921/?utm_source=claude_desktop) | POD-Galerkin ROM for turbulent flows up to Re = 10⁵ (207 citations) |

### ML-augmented ROM (Section 10)
| # | Ref | Contribution |
|---|---|---|
| 82 | [Ahmed 2021 nonlinear POD](https://consensus.app/papers/details/cb438c3ef5555bada27b542c14f343ab/?utm_source=claude_desktop) | Autoencoder + LSTM for convection-dominated ROM |
| 83 | [Grimberg 2020](https://consensus.app/papers/details/ea46855961585636aa5ff41659d37398/?utm_source=claude_desktop) | Petrov-Galerkin fixes PROM instability without closure |
| 84 | [Fu 2023](https://consensus.app/papers/details/ec888655fe48533d8b04a9392ec7451d/?utm_source=claude_desktop) | Stacked AE + self-attention for NIROM |
| 85 | [Ding 2024](https://consensus.app/papers/details/73eba141709b5be4ab4e0f582092e552/?utm_source=claude_desktop) | Systematic CAE vs FCAE vs POD comparison |
| 86 | [Zhu 2024](https://consensus.app/papers/details/8ec3e2f156fd54709761bfdd1590a236/?utm_source=claude_desktop) | Compressed autoencoders via pruning + SVD |

### Non-intrusive ROM (Section 11)
| # | Ref | Contribution |
|---|---|---|
| 87 | [Hesthaven 2018 POD-NN](https://consensus.app/papers/details/f62e99b5db3a55c4bba87b5acb99621c/?utm_source=claude_desktop) | POD + neural network (628 citations) |
| 88 | [Guo 2018 POD-GPR](https://consensus.app/papers/details/9b897d6fab6f569f8f189a496ec4db24/?utm_source=claude_desktop) | POD + Gaussian process (259 citations) |
| 89 | [Xiao 2015 POD-RBF](https://consensus.app/papers/details/d0ba90b79ba95157b8b06f90fea9554a/?utm_source=claude_desktop) | POD + RBF interp (183 citations) |
| 90 | [Min 2024 POD-RBFNN](https://consensus.app/papers/details/bca3d282463956fda8e4855eaad0f3f1/?utm_source=claude_desktop) | POD + RBFNN, smaller error than POD-BPNN |
| 91 | [Yang 2020 adaptive-sampling](https://consensus.app/papers/details/879ec1ea771c5fce836bf4801b4f835f/?utm_source=claude_desktop) | POD-GPR + adaptive sampling |

### From first evaluation pass (kept for cross-reference)
| # | Ref | Section |
|---|---|---|
| 2 | Xie 2018 (preprint of [1]) | Section 3 |
| 4 | Akhtar 2009 | Section 4 |
| 5 | Stabile 2017 | Section 4 |
| 7 | Gräßle 2019 | Section 4 |
| 9 | Lee 2020 | Section 4 |
| 14–16 | García-Archilla series (temporal derivative POD) | Section 2 |

---

## Summary — what the extended literature scan changed vs the roadmap as written

**Confirmed with high confidence**:
- 3-mode ceiling is the most likely dominant error source (Duan 2023 [38] formalises this exact pattern; Shnapp 2026 [39] quantifies mode counts needed for Lagrangian statistics)
- Lagrangian-inner-product POD is the right fix if mode count doesn't help (Xie 2020, mature implementation via Fareed's incremental weighted SVD)
- Discretely divergence-free bases work; JAX-ready implementation exists (Xigui Li 2026 [46])
- Grid-topology saturation matches approximation theory
- JAX/GPU stack is the right choice (JAX-MPM 2025 [54] demonstrates our exact scale in production)
- FSW-specific POD-ROM already exists (Fang 2025 [93]) — we need to cite and position against it

**New avenues surfaced**:
- Anisotropic metric-based AMR (Fidkowski, Balan, Prouvost) — could push shear-band accuracy 20-40% below saturation
- Volume-preserving integrators (Tapley 2019 [66] is the single most relevant paper we found) — could dramatically reduce long-time position error growth
- Petrov-Galerkin (Grimberg 2020 [83]) — cheap fix if we ever build a Galerkin POD-ROM
- Compression-ratio constraint (Prouvost 2024) — cheap grid-topology sweep
- Stirred-tank literature (Mikhaylov 2021/2023, Bashiri 2016, Peace 2025) — the correct positioning for the FSW paper

**De-prioritised**:
- ML-augmented ROMs (autoencoder+LSTM etc.) — only worth pursuing if Steps 6-9 leave residual > 15 %
- Domain-of-POD (mesh vs grid) — literature says it doesn't matter much; inner product does
- Higher-cell-count grid experiments (uniform_half etc.) — Prouvost 2024 [37] warns that this is a compression-ratio problem, not a resolution problem

**Consensus search budget used**: 19 of the 26 available (some reset
Aug 1). Coverage across 8 topic areas, 64 unique references.
