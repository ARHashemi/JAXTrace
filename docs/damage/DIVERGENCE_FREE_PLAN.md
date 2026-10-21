# Making the FSW velocity field divergence-free, while preserving it

**Status 2026-10-21.** Self-contained: the problem, the measurements that
establish it, the mathematics of the chosen method, what must be adapted for
our data, the cost, and the implementation steps. References are cited in text
and listed with DOIs at the end.

---

## 1. The problem, and the measurements that establish it

### 1.1 Why a divergence-free field is a prerequisite, not a refinement

The void model advects passive tracers through a precomputed field and lets
particles **detach** by a local criterion. The void is then a **particle-density
deficit**. For that to mean anything, the *attached* ensemble must conserve
density exactly.

Let $\mathbf{F}$ be the deformation gradient of the flow map. Along a
trajectory,

$$\frac{d}{dt}\ln\det\mathbf{F} \;=\; \nabla\!\cdot\mathbf{u},
\qquad\Longrightarrow\qquad
\det\mathbf{F}(t)=\exp\!\Big(\int_0^t \nabla\!\cdot\mathbf{u}\,d\tau\Big).$$

With $\nabla\!\cdot\mathbf{u}\equiv0$ and full entrainment, $\det\mathbf{F}\equiv1$:
**an initially uniform seed cannot develop interior holes.** ⚠️ Any hole that
appears is then either detachment (the signal) or discretisation error (the
artefact). **If the second is as large as the first, the method cannot work.**

### 1.2 What we measured — the field is not divergence-free

Per element, $\nabla\!\cdot\mathbf{u}=\operatorname{tr}\mathbf{D}$ with
$\mathbf{D}=\tfrac12(\mathbf{L}+\mathbf{L}^{\mathsf T})$,
$\mathbf{L}=\sum_a \mathbf{u}_a\otimes\nabla N_a$:

| quantity | D2, snapshot 34 | cylA, snapshot 159 |
|---|---:|---:|
| median $\lvert\nabla\!\cdot\mathbf{u}\rvert/\lvert\mathbf{D}\rvert$ | 4.87e-02 | 2.49e-02 |
| p90 | 5.81e-01 | — |
| fraction of elements > 0.01 | **79.3 %** | — |
| fraction > 0.1 | 36.0 % | — |
| median raw $\lvert\nabla\!\cdot\mathbf{u}\rvert$ | 2.31 s⁻¹ | 1.49 s⁻¹ |
| median $\dot\varepsilon$ for scale | 54.6 s⁻¹ | — |

✅ **Independently corroborated** by earlier ROM work
(`rom_out/velocity_diagnostics/divergence_summary_cylindrical_*.csv`), which
measured the FOM's own divergence on four further cases: median 1.64–2.99 s⁻¹
for $r<10$ mm, p99 75–151, max 478–855. ⚠️ **Not specific to one case or one
pin family.**

### 1.3 The consequence, measured two independent ways

**(a) In $\ln J$.** Integrating $\nabla\!\cdot\mathbf{u}$ along real
trajectories (`--damage-detachment`, column `ln_J`):

| cylA, 2684 steps | raw P1, `field` div | raw P1, `analytic` div | HCT cubic, `analytic` |
|---|---:|---:|---:|
| median $\lvert\det F-1\rvert$ | 1.87e-03 | 9.51e-03 | 5.59e-03 |
| p99 | 2.76e-01 | 9.46e-01 | 1.99e+00 |
| **fraction > 1e-2** | 12.67 % | **48.45 %** | 33.42 % |

⚠️ The `field` column is the P1 divergence **scattered to nodes and
re-interpolated linearly**, which *smooths* the error before integrating it.
✅ **The honest number is the `analytic` one: 48 % of particles exceed 1e-2**,
the order of the void fraction itself.

⚠️ The error is **distributed, not a few bad elements**: on D2, 97.7 % of
particles exceed 1e-4 and 29.4 % exceed 1e-2, in a smooth heavy-tailed
distribution with no isolated outlier population. ❌ **Element exclusion is
therefore both useless and wrong.**

**(b) In position space — the decisive one.** Occupancy of the cylA wake box
(291,137 particles, 160×60 yz grid):

```
EMPTY interior cells : 3,202 of 9,600  (33.4 %)
median occupancy of a non-empty cell : 53 particles
```

❌ **A third of the cross-section is already empty, with no detachment rule
applied at all.** Those holes cannot exist in a divergence-free field. ⚠️ **So
the artefact and the signal are the same observable**, and no detachment result
is quantitatively interpretable until this is fixed.

---

## 2. What does NOT work, and why — measured, not assumed

### 2.1 ❌ Higher-order reconstruction (HCT) — 1.45×, and for the wrong reason

A C¹ cubic Hsieh–Clough–Tocher reconstruction on the Alfeld split was built and
run (two cylA arms, 2684 steps). ✅ It does reduce the interpolant's divergence
— fraction over 1e-2 falls 48.45 % → 33.42 % — but ❌ **it fails the gate
anyway**, and its *tail* is worse (p99 rises 9.46e-01 → 1.99e+00), because a
cubic has larger between-node derivatives than a piecewise-constant gradient.

⚠️ **HCT enforces C⁰/C¹ continuity, never a divergence constraint.** Verified in
`jaxtrace/gpu/recovery/hct3d.py`. Cost is negligible (0.2 % of Stage 7) but
memory is not (2928 MB per snapshot on cylA).

### 2.2 ❌ Constrained least squares on the nodal gradient — 1.57×, structurally capped

Constraining the recovered nodal gradient to be trace-free is trivially
satisfiable (measured $\lvert\operatorname{tr}\mathbf{G}_c\rvert_{\max}=3.1$e-13)
and barely perturbs the fit (7.9e-04 relative). ❌ But the `vertex_taylor`
divergence splits as

$$\nabla\!\cdot\mathbf{u}=\underbrace{\sum_a\nabla N_a\cdot(\mathbf{v}_a+\mathbf{G}_a(x-x_a))}_{\text{TERM 1, from VELOCITIES}}+\underbrace{\sum_a N_a\operatorname{tr}\mathbf{G}_a}_{\text{TERM 2, from GRADIENTS}}$$

and only TERM 2 can be constrained. Measured at element centroids on cylA
(400 k sample): TERM 1 median **1.173**, TERM 2 median **0.148** — ⚠️ TERM 2 is
**13.6 %** of TERM 1, giving a reduction of **1.57×**. ❌ Not enough.

### 2.3 ❌ One extra centroid DOF ("local Scott–Vogelius") — exactly 1.00×, a theorem

Giving each parent tet one extra velocity DOF at the Alfeld centroid
$\mathbf{u}_{vc}$ and minimising $\sum_s V_s(\mathbf{g}_s\cdot\mathbf{u}_{vc}+r_s)^2$
produces **no change whatsoever**. Diagnosed exactly:

```
sum_s V_s * g_s                     = 2.5e-16   (identically 0)
||grad of the objective at u_P1||   = 2.5e-15   (identically 0)
cond(A) median 52, rank 3           (NOT a numerical problem)
```

1. ⚠️ $\sum_s V_s\mathbf{g}_s=\mathbf{0}$: by the divergence theorem an
   **interior** DOF cannot change net flux through the parent boundary.
2. ⚠️ The P1 centroid value is **already** the least-squares minimiser — the
   objective is stationary there by construction.

❌ **And it was never Scott–Vogelius.** Real SV is $P_k$ velocity /
$P_{k-1}$ **discontinuous** pressure, inf-sup stable on an Alfeld split only
for $k\ge3$ in three dimensions [Joh20, Ain21]. We attempted $k=1$.

---

## 3. ✅ The chosen method: RTN₀ projection (Selzer & Cirpka 2020)

### 3.1 Why this one

[Sel20] states our defect in our own words: *"Accurate particle tracking
requires element-wise mass-conservative, conforming velocity fields. **This
condition is not fulfilled by the standard linear Galerkin finite element
method.**"* ✅ Same discretisation (P1 on simplices), same application (particle
tracking), same defect, and it is built **for tetrahedra**.

| | Scott–Vogelius | **RTN₀ projection** |
|---|---|---|
| 3D order needed | ❌ $P_3$ on Alfeld split | ✅ **lowest order** |
| guarantees | pointwise div-free | ✅ zero divergence per element, continuous normal flux |
| designed for | solving Stokes | ✅ **particle tracking** |
| bonus | — | ✅ semi-analytical element-wise trajectories |

### 3.2 The space

The lowest-order Raviart–Thomas–Nédélec space [RT77, Ned80] on a tetrahedron
$E$ has one degree of freedom per face — the **normal flux**
$q_{F}^{\perp}$ — and basis functions

$$\boldsymbol{\psi}_{F_m}(\mathbf{x})=\frac{\lvert F_m\rvert}{d\,\lvert E\rvert}\,(\mathbf{x}-\mathbf{x}_{n_m}),$$

where $\mathbf{x}_{n_m}$ is the vertex opposite face $F_m$ and $d=3$. Two
properties make the space the right target:

1. ✅ **$\nabla\!\cdot\boldsymbol{\psi}_{F_m}$ is constant per element**, so
   $\nabla\!\cdot\mathbf{u}_h$ is a single number per element and can be set to
   zero **exactly**.
2. ✅ **The normal component is continuous across faces**, so the field is
   *conforming* in $H(\mathrm{div})$ and a particle crossing a face sees no
   jump in normal velocity — which is what makes trajectories well defined.

### 3.3 The projection

Unknowns: the vector of face fluxes $\mathbf{q}=(q_{F_m}^{\perp})_{F_m\in\mathcal F}\in\mathbb R^{M}$.

**Objective** (their Eq. 22) — stay as close as possible to the P1 solution, in
the velocity at element centroids:

$$\min_{\mathbf q}\;\tfrac12\,\lVert\mathbf N\mathbf q-\mathbf g\rVert_2^2 ,$$

where $\mathbf g\in\mathbb R^{Nd}$ stacks the P1 element-centroid velocities and
$\mathbf N\in\mathbb R^{Nd\times M}$ assembles
$\mathbf N_{im}=\boldsymbol{\psi}_{F_m}(\mathbf x_{c,i})$ for $F_m\subset\partial E_i$,
zero otherwise.

**Constraint** (their Eq. 24) — element-wise mass conservation:

$$\mathbf M\mathbf q=\mathbf f,\qquad
\mathbf M_{im}=\boldsymbol{\nu}_{F_m}\!\cdot\mathbf n_{E_i}\,\lvert F_m\rvert,\qquad
f_i=\int_{E_i} f\,dx ,$$

with $\boldsymbol\nu_{F_m}$ the face's reference normal, $\mathbf n_{E_i}$ the
outward normal of $E_i$, and $f_i$ the source strength. ✅ **For us
$\mathbf f=\mathbf 0$**: the FSW workpiece has no volumetric source, so the
constraint is exactly $\nabla\!\cdot\mathbf u_h=0$ per element.

**Lagrangian** (their Eq. 26) and its stationarity conditions (Eq. 27):

$$\mathcal L(\mathbf q,\boldsymbol\lambda)=\tfrac12\lVert\mathbf N\mathbf q-\mathbf g\rVert^2+\boldsymbol\lambda^{\mathsf T}(\mathbf M\mathbf q-\mathbf f)$$

$$\begin{bmatrix}\mathbf N^{\mathsf T}\mathbf N & \mathbf M^{\mathsf T}\\[2pt] \mathbf M & \mathbf 0\end{bmatrix}
\begin{bmatrix}\mathbf q\\ \boldsymbol\lambda\end{bmatrix}
=\begin{bmatrix}\mathbf N^{\mathsf T}\mathbf g\\ \mathbf f\end{bmatrix}$$

a sparse symmetric **saddle-point system**, one multiplier per element.

### 3.4 ⚠️ The one adaptation we must make, and what it costs

[Sel20] weight the objective by the inverse hydraulic conductivity,
$\mathbf N_{im}=\mathbf K^{-1}_{E_i}\boldsymbol\psi_{F_m}(\mathbf x_c)$ (their
Eq. 23), because in groundwater the primary unknown is the head and the natural
metric is the hydraulic gradient. ⚠️ **We have no $\mathbf K$** — our velocity
is a computed FSW flow field, not a Darcy flux.

✅ **Set $\mathbf K=\mathbf I$.** The objective becomes *"find the RTN₀ field
whose element-centroid velocity is closest in $L_2$ to the P1 centroid velocity,
subject to exact element-wise mass conservation."* ⚠️ **What we lose:** their
weighting is physically optimal in heterogeneous media. ✅ For a
single-material workpiece that weighting is the identity anyway.

### 3.5 ⚠️ What "preserving the field" can and cannot mean

RTN₀ is **lowest order**: the normal flux is constant per face and the
tangential components are reconstructed, not independent. ⚠️ **So the projection
necessarily loses tangential detail relative to P1.** It trades velocity
accuracy for exact mass conservation.

✅ **For this application that trade is the right way round**, because §1.3(b)
shows the mass error is already corrupting the primary observable. ⚠️ **But it
must be quantified and reported**, not assumed benign — see step 4 of §5.

---

## 4. Cost, sized on our meshes

| | cylA | **D2** |
|---|---:|---:|
| elements $N$ | 3,050,196 | 10,807,677 |
| face unknowns $M$ | ~6.1 M | **~21.6 M** |
| saddle-point size $M+N$ | ~9.2 M | **~32.4 M** |
| nonzeros | ~24 M | ~86 M (**~1 GB**) |
| storage, 1 float32 per face | 24 MB | **86 MB/snapshot** |
| all snapshots | 24 MB | **14 GB** (166 snapshots) |

✅ **Memory is comparable to the octree we already build and cache.**
⚠️ **The cost is the solve** — global, sparse, per snapshot, ×166 on D2.

✅ **Therefore the projection MUST be prebuilt**, not evaluated in-step. That is
a stronger statement than the earlier per-element estimate implied: a global
solve cannot run inside a particle step at all. ✅ Storage streams exactly like
the velocity stack.

---

## 4b. ⚠️ STEP 1 RESULT (2026-10-21) — RTN₀ conserves mass but DISCARDS ROTATION

The sub-mesh proof was built and run
(`scripts/damage/rtn0_submesh_proof.py`, cylA, 20,000 contiguous elements).
✅ **The projection works.** ❌ **But the field it returns is not usable for
FSW, and the reason is structural.**

### What the solver does

| attempt | basis | scaling | constraint residual | preservation |
|---|---|---|---:|---:|
| 1 | with $\lvert F
vert$ | none | **6.8e-16** ✅ | 84 % ❌ |
| 2 | no $\lvert F
vert$ | none | 9.3e+04 ❌ | huge ❌ |
| 3 | no $\lvert F
vert$ | row-scaled | 1.3e-08 ⚠️ | 89 % ❌ |

⚠️ **Attempt 1 satisfied the constraint at machine precision** — the
saddle-point assembly, the global mass-balance correction and the boundary
pinning are all correct. ❌ Its basis was wrong (see below), and attempts 2–3
use the right basis but suffer $1/(3V)$ with $V\sim10^{-12}$, which wrecks the
conditioning.

⚠️ **Two implementation bugs found and worth recording.** (i) With $q_F$ the
**flux** rather than the flux density, the basis is
$oldsymbol\psi_F=(\mathbf x-\mathbf x_{opp})/(3V)$ with **no $\lvert F
vert$
factor**; a unit test on a constant field — which RTN₀ contains exactly —
returned 50 % error with it and 3.5e-17 without, and took seconds to write.
(ii) **Global mass balance is a consistency condition**: demanding
$
abla\!\cdot\mathbf u=0$ in every element forces the net boundary flux to
vanish, and the P1 field's does not, so the constrained system is insoluble
until the imbalance is distributed over the boundary.

### ❌ The decisive finding: RTN₀ cannot represent rotation

Tested with **no solver at all** — exact face fluxes of a known field, then
reconstruct:

| field | relative error |
|---|---:|
| constant | **3.5e-17** ✅ |
| linear with divergence ($\mathbf u=\mathbf x$) | **1.1e-16** ✅ |
| **rotation** ($\mathbf u=(-y,x,0)$) | **3.3e-01** ❌ |
| shear ($\mathbf u=(z,0,0)$) | 3.3e-01 ❌ |
| linear divergence-free ($\mathbf u=(x,-y,0)$) | **1.00** ❌ |

✅ **On one tetrahedron, RTN₀ spans only $\{	ext{constants}\}\oplus\{\mathbf x\}$**
— four degrees of freedom, three translations plus one radial (divergence)
mode. ❌ **Rotation is not in the space.**

⚠️ **FSW flow near the tool is dominated by rotation.** So the ~85 %
"preservation" loss, nearly identical across three very different numerical
setups, is **a property of the space, not of the implementation**. ❌ RTN₀
conserves mass by throwing away exactly the motion that defines the process.

⚠️ **This does not contradict [Sel20].** Their application is groundwater,
where flow is gradient-driven and nearly irrotational, so lowest-order RTN₀
loses little. ✅ Our flow is the opposite case, and the method does not
transfer for that specific reason.

### ✅ What this changes

❌ **Lowest-order RTN₀ is rejected** — on a measurement, not a guess.
The remaining routes, re-ranked:

1. ✅ **Next-order Raviart–Thomas (RT₁) or BDM₁.** BDM₁ contains **all linear
   fields**, including rotation, and is still $H(\mathrm{div})$-conforming with
   element-wise exact mass conservation. ⚠️ 3 DOF per face instead of 1 — about
   65 M unknowns on D2, so the solve grows but the structure is unchanged.
2. ⚠️ **Scott–Vogelius $P_3$ on the Alfeld split**, as originally ranked — more
   expensive but pointwise divergence-free and rotation-complete.
3. ❌ Lowest-order RTN₀ — **rejected, rotation-blind.**

✅ **The infrastructure built here transfers directly to option 1**: the face
topology, orientation convention, mass-balance correction, boundary pinning and
saddle-point assembly are all basis-independent. ⚠️ Only the basis functions and
the per-face DOF count change.

---

## 4c. ✅ BDM₁ vs SV-P₃ — tested the cheap way FIRST (2026-10-21)

⚠️ RTN₀ was built, solved, and only *then* found rotation-blind. ✅ **Both
candidates were therefore tested for representation content before any solver
was written** (`scripts/damage/test_fe_space_content.py`).

### ✅ BDM₁ recovers every linear field, including rotation

Recovering a field from its P1 normal traces on the four faces:

| field | relative error |
|---|---:|
| constant | 4.1e-16 ✅ |
| **rotation about z** | **7.5e-16** ✅ |
| **rotation about x** | **2.9e-16** ✅ |
| shear | 1.3e-15 ✅ |
| linear divergence-free | 8.4e-16 ✅ |
| linear with divergence | 1.0e-15 ✅ |
| quadratic | 1.00 ❌ (expected, and not needed) |

✅ **BDM₁ on a tet has 12 DOF — exactly the 12 coefficients of a P1 vector
field — so it contains ALL linear fields.** The rank of the trace map is 12,
i.e. full. ⚠️ Quadratics are absent, which does not matter: the source field is
itself P1, hence linear per element.

### ✅ And the divergence-free subspace keeps what we need

For $\mathbf u=\mathbf A\mathbf x+\mathbf b$ in BDM₁, $
abla\!\cdot\mathbf u
=\operatorname{tr}\mathbf A$ is a **constant per element**, so $
abla\!\cdot
\mathbf u=0$ is **one scalar equation per element** — the same constraint
structure as RTN₀, exactly enforceable by one Lagrange multiplier.

Measured $\operatorname{tr}\mathbf A$ after projection
(`scripts/damage/test_bdm1_divfree_subspace.py`):

| field | tr A | |
|---|---:|---|
| rotation about z | +1.4e-15 | ✅ already divergence-free |
| shear | −1.2e-15 | ✅ already divergence-free |
| radial ($\mathbf u=\mathbf x$) | **+3.000** | ⚠️ carries the divergence |

✅ **Imposing $
abla\!\cdot\mathbf u=0$ removes ONLY the radial mode —
precisely the spurious part — and leaves rotation and shear untouched.**
12 DOF − 1 constraint = **11 free parameters**, against RTN₀'s 4 − 1 = 3.
❌ That one line explains the whole RTN₀ failure: its three surviving DOF were
the translations, and rotation was never available to begin with.

### Cost

| | BDM₁ | SV-P₃ |
|---|---:|---:|
| content | ✅ all linear fields | ✅ all cubics |
| DOF per face | 3 | — (nodal) |
| cylA | ~18.3 M | ~91.5 M |
| **D2** | **~64.8 M** | **~324 M (5.0×)** |
| constraint | 1 scalar/element | 1 per sub-tet |

✅ **BDM₁ is chosen**: it has everything the FSW field needs at **one fifth**
the cost, and it reuses the RTN₀ infrastructure unchanged (face topology,
orientation, mass balance, boundary pinning, saddle-point assembly) — only the
basis and the per-face DOF count differ.

⚠️ **SV-P₃ is kept as the fallback**, not discarded: it is *pointwise*
divergence-free where BDM₁ is element-wise, which matters only if the
element-wise guarantee proves insufficient for $\det F$.

---

## 4d. ✅ BDM₁ WORKS — divergence solved; ⚠️ the solver needs conditioning work

`scripts/damage/bdm1_submesh_proof.py`, cylA, contiguous sub-meshes.

### ✅ The divergence problem is SOLVED

| | median $\lvert\nabla\!\cdot\mathbf u\rvert/\lvert\nabla\mathbf u\rvert$ |
|---|---:|
| P1 (the problem) | 2.76e-03 |
| **BDM₁** | **1.0e-16** |

✅ **Machine precision, and stable at every problem size tested** (2 k → 20 k
elements: 9.1e-17, 9.5e-17, 1.0e-16). ✅ **That is a factor of $2.8\times10^{13}$,
and it is the result this whole line of work was for.**

### ✅ And unlike RTN₀, the field survives

| size | BDM₁ interior median | RTN₀ |
|---|---:|---:|
| 2,000 el | **0.0100** | ~0.85 |
| 5,000 el | **0.0249** | ~0.85 |
| 20,000 el | 1.62 ⚠️ | ~0.85 |

✅ At 2 k elements the field changes by **1 %** while becoming exactly
divergence-free — ✅ **85× better preservation than RTN₀**, and it confirms
§4c's prediction that removing only the radial mode leaves rotation intact.

### ⚠️ But preservation degrades with problem size — a conditioning issue

❌ **0.010 → 0.025 → 1.62 as the sub-mesh grows, while the divergence stays at
1e-16.** ⚠️ That signature — constraint satisfied perfectly, objective
increasingly violated — is a **badly conditioned saddle point**, not a physical
effect. The direct LU is losing accuracy in the objective block as the system
grows.

### ✅ FIXED (2026-10-21): Tikhonov toward the P1 trace

⚠️ **$\mathbf N^{\mathsf T}\mathbf N$ is singular by construction.** The
objective sees only the 3-component centroid velocity per element while the
element carries 12 DOF, so **9 directions per element are invisible to it** and
are pinned only through face sharing. The constraint was satisfied and the
objective was free to wander in that null space — exactly the observed
signature.

✅ **Fix: a Tikhonov term pulling every DOF toward its P1 normal trace**, i.e.
*"stay as close to the input field as the divergence constraint allows"*. That
makes the objective block positive definite and the saddle point well posed.
⚠️ [Sel20] never meet this because RTN₀ has **one** DOF per face (4 per
element), so their objective is not rank deficient.

**Weight sweep at 20,000 elements (interior):**

| `--reg` | divergence | preservation p50 | p99 |
|---|---:|---:|---:|
| 0 | 1.0e-16 | 1.624 ❌ | 1.97e+12 ❌ |
| **1e-8** | **8.5e-17** ✅ | **0.097** ✅ | **1.86** ✅ |
| 1e-6 | 4.2e-16 ✅ | 0.101 ✅ | 1.97 |
| 1e-4 | 5.7e-14 | 2.004 ❌ | 2.87e+01 |
| 1e-2 | 5.9e-12 | 197.7 ❌ | 2.76e+03 |

⚠️ **Too much regularisation is worse, not better** — above ~1e-6 it fights the
objective instead of only fixing its null space. ✅ **The usable window is
1e-8 … 1e-6.**

**✅ And it now SCALES**, which was the whole problem:

| elements | divergence | preservation p50 | solve (direct LU) |
|---|---:|---:|---:|
| 5,000 | 8.2e-17 | 0.1005 | 0.4 s |
| 20,000 | 8.5e-17 | 0.0974 | 7.1 s |
| 50,000 | 8.7e-17 | **0.0675** | 25.1 s |

✅ **Preservation improves with size** (0.100 → 0.068) instead of blowing up,
and divergence is pinned at 8.5e-17 throughout. ✅ **BDM₁ with `--reg 1e-8` is
a working divergence-free projection that preserves the field to ~7 %.**

#### ⚠️ Remaining (superseded causes, kept for the record)
1. The constraint row scaling is a single global factor; a **per-row** scaling
   (by $\lVert\mathbf M_i\rVert$) is the standard fix.
2. $\mathbf N^{\mathsf T}\mathbf N$ is **singular by construction** — the
   objective only sees the centroid velocity (3 numbers) per element while the
   element has 12 DOF, so 9 directions per element are unconstrained by the
   objective and are fixed only through face sharing. ⚠️ **A regularisation
   term $\epsilon\lVert\mathbf q-\mathbf q_{P1}\rVert^2$ pulling the unseen
   directions toward the P1 trace is the principled fix**, and it is what
   [Sel20] avoid by having only one DOF per face.
3. An iterative solver (MINRES with a block preconditioner) rather than LU.

✅ **None of these threatens the method** — the space is right and the
divergence result is exact. ⚠️ **It is solver engineering, and it is not
finished.**

### Cost note

BDM₁ at 20 k elements: 31,910 × 31,910 saddle point, 620 k nonzeros, 7.0 s by
direct LU. ⚠️ Extrapolating to D2's 10.8 M elements is not meaningful for a
direct solver; step 3 of §5 (iterative + preconditioner) is mandatory, as
already stated.

---

## 4e. ✅ Would SV-P₃ preserve the field better than BDM₁? NO — proved

⚠️ **The ~7 % change is not a BDM₁ limitation.** BDM₁ contains *all* linear
fields and our source is P1 — linear per element — so BDM₁ could reproduce it
**exactly**. The change is the price of the **constraint**, not of the space.

### The per-element lower bound, and why it is not attainable

For $\mathbf u=\mathbf A\mathbf x+\mathbf b$ the nearest divergence-free linear
field removes the isotropic part, $\mathbf A'=\mathbf A-\tfrac13(\operatorname{tr}\mathbf A)\mathbf I$,
giving a change of $\tfrac13\lvert\operatorname{tr}\mathbf A\rvert\,\lVert\mathbf x-\mathbf x_c\rVert$.
At our measured divergence that is **4.2e-06** — four orders below BDM₁'s
6.8e-02.

❌ **But that bound is unattainable by ANY conforming method**, and the reason
is a one-line proof (`scripts/damage/test_conforming_obstruction.py`): P1 is
only $C^0$, so $\mathbf A$ **jumps across faces**. Projecting each element
independently then gives a normal component that is **double-valued on the
shared face**:

```
normal component on a shared face, per-element ideal projection
   vertex 0 : side A +0.546713   side B +0.546713   jump 0.0
   vertex 1 : side A +0.009760   side B -0.224813   jump 2.35e-01
   vertex 2 : side A +1.127831   side B +1.050673   jump 7.72e-02
```

⚠️ **A field with a jump in the normal component is not in $H(\mathrm{div})$**,
so it is in neither BDM₁ **nor Scott–Vogelius** — SV is $H^1$-conforming, which
is *stronger*. ✅ **Both methods face the same obstruction**, and the extra
change above 4.2e-06 is the cost of reconciling neighbouring elements'
incompatible gradients into one conforming field.

### ✅ Verdict

❌ **SV-P₃ cannot be expected to preserve the field better in the median.** It
removes the same divergence and satisfies a *stricter* continuity requirement.

⚠️ **Where it could still differ:** it is **pointwise** divergence-free where
BDM₁ is element-wise, so if residual within-element divergence proves to matter
for $\det F$, SV-P₃ would help. ✅ **But that is a measurable question** —
run $\det F$ on the BDM₁ field first; only if it fails does SV-P₃ earn its 5×
cost.

---

## 4f. ⚠️ The full-mesh solver — where it stands (2026-10-21)

### ✅ The method is proven; the SOLVER is the open item

| elements | divergence | preservation p50 | direct LU |
|---|---:|---:|---:|
| 5,000 | 8.2e-17 | 0.1005 | 0.4 s |
| 20,000 | 8.5e-17 | 0.0974 | 7.1 s |
| 50,000 | 8.7e-17 | **0.0675** | 25.1 s |

### ❌ The iterative path does not yet work — four attempts, recorded

| attempt | outcome |
|---|---|
| MINRES + diagonal preconditioner | ⚠️ 20,000 iterations, relative residual **1.3e-02**, divergence degraded to 5.1e-04 |
| ILU of the raw saddle point | ❌ *"Factor is exactly singular"* — the (2,2) block is exactly zero |
| ILU + shift scaled by max\|diag A\| | ❌ still singular; three `--reg` values gave **byte-identical** output, i.e. the shift was never large enough to matter |
| ILU + shift scaled by \|M\| (5.1e-05) | ❌ **still singular** |

⚠️ **A generic saddle point of the same block structure factorises fine** —
tested. ⚠️ **So our matrix is genuinely near-singular, and the cause is a null
space I have not characterised.** ❌ Further preconditioner tuning is the wrong
move; the structure needs diagnosing first.

### ✅ The pragmatic route: domain decomposition

Measured LU scaling is $t\sim n^{1.82}$, so a monolithic cylA solve
extrapolates to **~14 hours** and would likely exhaust RAM. ✅ But subdomains of
~50 k elements cost 25 s each, and cylA needs 61 of them:

| | serial | 16 threads |
|---|---:|---:|
| cylA, one snapshot | 25.4 min | **1.6 min** |

✅ **This reuses only the solver already proven**, and is embarrassingly
parallel across both subdomains and snapshots. ⚠️ **The cost is overlap**:
subdomain boundaries must be pinned to the P1 trace, so a particle crossing one
sees a small flux mismatch — the same over-constraint measured on the sub-mesh
cut boundary (§4d). ⚠️ With 61 subdomains the affected fraction is far smaller
than the 29 % seen on a single 20 k block, but it is not zero and must be
measured, not assumed.

### Decision point

| route | effort | risk |
|---|---|---|
| **A** — domain decomposition with the proven LU | ~½ day | ⚠️ overlap error, measurable |
| **B** — characterise the null space, then precondition properly | ~½–1 day | ⚠️ may not resolve quickly; four attempts already failed |
| **C** — external solver (PETSc/MUMPS fieldsplit) | ~1 day + dependency | ✅ the standard answer, ❌ new dependency on LUMI |

---

## 4g. ✅ Is BDM₁ good enough? Two worries, both measured

⚠️ Two concerns were raised: that BDM₁ *"does not provide continuity on the
element faces"* and is *"not point-wise divergence free"*. ✅ **Both are
testable, and both are largely unfounded** — though one of them was my own
sloppy wording.

### ✅ Worry 1 — face continuity: the NORMAL component is exactly continuous

`scripts/damage/test_bdm1_face_continuity.py`, two tets sharing a face, DOFs
shared with the correct orientation sign:

```
outward normals on the shared face : tet A (0,0,-1), tet B (0,0,+1)  ✅ opposite
NORMAL component jump              : 0.000e+00     ✅ CONTINUOUS
```

✅ **BDM₁ is $H(\mathrm{div})$-conforming by construction** — the normal flux a
particle sees crossing a face is **single valued**. ⚠️ **The TANGENTIAL
component does jump**, and that is what $H(\mathrm{div})$ means; it is true of
RT, BDM and every $H(\mathrm{div})$ space. Only SV, being $H^1$-conforming, is
fully continuous.

⚠️ **A caveat on the first version of this test:** matching DOFs by *local*
index gave a spurious jump of 5.97. The two tets order the shared face
differently and their outward normals are opposite, so the sign must be handled
— exactly as the real assembly does via sorted global node ids.

### ✅ Worry 2 — pointwise divergence: BDM₁ IS pointwise divergence-free

⚠️ **My earlier "element-wise, not pointwise" wording was misleading, and I
withdraw it.** For $\mathbf u=\mathbf A\mathbf x+\mathbf b$,
$\nabla\!\cdot\mathbf u=\operatorname{tr}\mathbf A$ is a **constant over the
element**. Setting it to zero therefore makes the field **pointwise**
divergence-free *inside every element*, not merely on average. ✅ The measured
8.5e-17 is exactly that.

### ✅ So does the tangential jump hurt $\det F$? No — and this is decisive

$$\ln J=\int \nabla\!\cdot\mathbf u\,dt$$

A trajectory spends **100 % of its time inside elements**; faces are a
**measure-zero** set. With $\nabla\!\cdot\mathbf u=0$ pointwise inside every
element, the integral is **exactly zero regardless of any face behaviour**.
✅ **$\det F=1$ to integration error, and the tangential jump cannot affect it.**

⚠️ What the jump *does* affect is the **trajectory**: a particle crossing a
face sees its tangential velocity change discontinuously, so the path is
slightly kinked. ⚠️ Bounded by the same ~7 % field change already measured.

✅ **And the comparison that settles it:** the P1 field is $C^0$ — *fully*
continuous in every component — and its $\det F$ **fails catastrophically**
(48 % of particles over 1e-2). ❌ **Continuity is not what makes $\det F$ work;
the divergence is.** BDM₁ fixes the property that matters and degrades one that
demonstrably does not.

### Verdict

| | continuity | divergence | $\det F$ | cost |
|---|---|---|---|---|
| P1 (now) | ✅ full | ❌ 2.8e-03 | ❌ broken | — |
| **BDM₁** | ✅ normal, ⚠️ tangential jumps | ✅ **8.5e-17 pointwise** | ✅ **exact** | 1× |
| SV-P₃ | ✅ full | ✅ pointwise | ✅ exact | ⚠️ 5×, unproven roadmap |

✅ **Continue with BDM₁.** ⚠️ SV-P₃'s only remaining advantage is tangential
continuity, which affects trajectory smoothness and **not** the quantity the
whole exercise exists to fix. ✅ **The honest test is empirical**: run $\det F$
and the occupancy check on the BDM₁ field. If they pass, SV-P₃ buys nothing for
5× the cost and an unproven roadmap.

---

## 4h. ✅ The solver question — diagnosed, and the literature found a FOURTH option

### ✅ The null space is characterised exactly

On a 1,500-element block: $\operatorname{rank}(\mathbf M)=1{,}498$ of 1,500,
**deficiency exactly 2**, with two singular values at 5.17e-12. ⚠️ The left
null vectors were **not** the constant vector, so it is *not* the classic
enclosed-domain pressure mode.

✅ **Cause found and it matches exactly: 2 elements have all four faces on the
sub-mesh cut.** Every DOF is pinned, so the $\nabla\!\cdot\mathbf u=0$ row acts
on no free unknown and is identically zero. ✅ **Dropping those rows is exact,
not a regularisation** — they carry no information. ⚠️ **And on a FULL mesh
they cannot occur**: an element with four true-boundary faces would be a
detached tetrahedron. A sub-mesh artefact.

### ⚠️ But that was not what blocked ILU

❌ At 20,000 elements there are **no** empty rows and ILU still fails
*"exactly singular"*. ✅ A synthetic saddle point of the same block structure
factorises fine (cond 8.2e+02, ILU succeeds at three drop tolerances). ⚠️ **So
our matrix is a CONDITIONING problem, not a structural one**, and more
preconditioner tuning is the wrong response.

### ✅ What the literature says (Undermind, 146 papers)

> *"At this scale, the best-supported direction is a matrix-free Krylov method
> for the constrained least-squares KKT system, with a problem-specific Schur-
> complement preconditioner; Stokes/Darcy H(div) solvers are useful design
> precedents, not drop-in preconditioners for your distinct $N^TN+\epsilon I$
> block."*

Key points:
- ✅ $\mathbf A$ is **positive definite for $\epsilon>0$**, so the natural
  reduced operator is the Schur complement
  $\mathbf S=\mathbf M\mathbf A^{-1}\mathbf M^{\mathsf T}$.
- ⚠️ *"The Jacobi stall indicates that a diagonal scaling alone is
  inadequate"* — confirming our measurement.
- ✅ **Hybridisation + AMG** [Dob18, Lee16] and **auxiliary-space AMG for
  H(div)** [Kol12] have scalable results, including parallel implementations;
  ✅ matrix-free GPU saddle-point solvers for H(div) exist [Paz23].
- ✅ **"The repeated snapshots share the operator, making setup reuse and
  Krylov recycling especially valuable"** — 166 snapshots on one mesh, so one
  factorisation amortises 166×. ⚠️ **We had not exploited this.**

### ✅✅ THE FOURTH OPTION — local patchwise reconstruction, no global solve

⚠️ **This was not among the three I offered, and it may be better than all of
them.** [Led16] and [Coc07] construct **divergence-free $H(\mathrm{div})$
reconstruction operators patchwise** — element-local or vertex-patch problems
that assemble into a globally conforming, exactly divergence-free field.

✅ **If it applies, it removes the global solve entirely**: tiny dense local
solves, embarrassingly parallel, no preconditioner, no conditioning problem,
and it runs per snapshot at near-zero cost.

⚠️ **The caveat the search states plainly:** patchwise reconstructions *"do not
generally reproduce this global least-squares minimiser"* — the field would be
divergence-free and close to P1, but not the *closest*. ✅ For our purpose that
is acceptable: we need $\det F=1$ and a faithful field, not optimality.

⚠️ Also found: [Scu16b] *"Mass-conservative reconstruction of Galerkin velocity
fields for transport simulations"* and [Cor92] on continuous velocity fields
and path lines in finite elements — **both are our exact application**, and
neither was in the earlier searches.

### Revised plan

| rank | route | why |
|---|---|---|
| **1** | ✅ **Local patchwise reconstruction** [Led16, Coc07, Scu16b] | no global solve at all; needs full texts to confirm applicability |
| 2 | ✅ Schur-complement CG with setup reuse across 166 snapshots | the literature's direct recommendation; $\mathbf A$ is SPD for $\epsilon>0$ |
| 3 | ⚠️ Hybridisation + AMG [Dob18, Lee16] | proven scalable, but a larger implementation |
| 4 | ⚠️ Domain decomposition with the proven LU | pragmatic fallback, overlap error |
| ❌ | more ILU tuning | the matrix is ill-conditioned, not singular |

**Full texts wanted:** [Led16], [Coc07], [Scu16b], [Dob18].

---

## 4i. ✅✅ RE-RANKED AFTER READING — Larson–Niklasson wins, and it is not close

Full texts read: [Scu16b], [Led16], [Coc07], [Dob18], [Paz23], [Cor92].

### ✅ [Scu16b] is our exact problem, solved locally

Their abstract: *"a local **Larson–Niklasson** post-processing procedure for
reconstructing mass-conservative velocities from a linear (**P1**) Galerkin
finite element solution … modifies the standard finite element velocities by
**imposing local conservation on element patches**. The resulting reconstructed
flow field is characterized by **continuous fluxes on element edges**."*

✅ Same input (P1 Galerkin), same defect, same requirement (continuous normal
flux for transport), **and no global solve**.

**The method.** For each node $i$, the elements sharing it form the **element
patch** $E_i$. A small system is assembled on that patch and solved; the
conservative face flux is then the average of the per-node contributions,
$q_F=\tfrac13\sum_i q_F^{(i)}$. ⚠️ Each patch system has **rank $N_{E_i}-1$** —
Eq. (14) still holds multiplied by a constant — fixed by setting one unknown to
zero. ✅ **That is the same rank-1 deficiency we diagnosed, with a known cure.**

**The cost, and it is decisive:**

| | cylA | D2 |
|---|---:|---:|
| local solves | 571,533 | 1,921,303 |
| dimension each | ≤ 18 | ≤ 18 |
| work | $O(m^3)$ dense LU | same |
| **estimated, 16 threads** | **~0.2 s/snapshot** | **~0.7 s/snapshot** |
| **BDM₁ global saddle point** | **~14 h/snapshot** | worse |

✅ Their own comparison: P1+LN costs **55,000 AXPY** against **175,000** for
mixed hybrid FE — *"a strong computational advantage"*. ✅ **"Intrinsically
parallelizable"**, no global matrix, no preconditioner, **no conditioning
problem** — the thing that has blocked us for a day disappears entirely.

⚠️ **Known weakness, and why it does not apply to us:** element patches at
strong **permeability contrasts** trap mass; Schiavazzi (2011) adds a second
correction step. ✅ **A single-material FSW workpiece has no such contrast.**

### The others, in one line each

- ✅ **[Paz23]** — matrix-free GPU saddle-point solvers for $H(\mathrm{div})$,
  with *"block-preconditioned MINRES converging in a number of iterations
  independent of mesh size and polynomial degree"* and an approximate Schur
  complement that is an **M-matrix graph Laplacian**, so AMG-friendly.
  ⚠️ **This is the right answer if a global solve is ever needed** — and it is
  GPU-native, matching our stack. ❌ But it solves a problem LN avoids.
- ⚠️ **[Dob18]** hybridisation + AMG, **[Led16]** patchwise divergence-free
  reconstruction, **[Coc07]** locally conservative fluxes for CG — all sound,
  all more machinery than [Scu16b] for the same end.
- ✅ **[Cor92]** (186 citations) — continuous velocity fields and path lines in
  finite elements, i.e. our application two decades earlier.

### ✅ Final ranking

| rank | route | cost/snapshot | status |
|---|---|---:|---|
| **1** | ✅ **Larson–Niklasson patch reconstruction** [Scu16b] | **~0.2–0.7 s** | ✅ **proceed** |
| 2 | Matrix-free GPU $H(\mathrm{div})$ MINRES [Paz23] | — | fallback if LN's accuracy disappoints |
| 3 | BDM₁ global saddle point | ~14 h | ❌ **superseded** |
| 4 | Domain decomposition / ILU tuning | — | ❌ **abandoned** |

⚠️ **The BDM₁ work is not wasted**: the face topology, orientation convention,
mass-balance consistency check, boundary pinning and the two verdict tests
(divergence + preservation) all transfer directly. ✅ And it established the
*target*: machine-precision divergence with ~7 % field change is achievable, so
LN can be judged against a known-good reference rather than against hope.

---

## 4j. ❌ LN DOES NOT APPLY — our defect is not the one it fixes

Implemented and debugged. ❌ **Larson–Niklasson cannot help us, for a reason
that is structural and that I should have checked before building it.**

### The measurement

Over 3,000 interior faces of cylA, comparing the two one-sided P1 fluxes:

```
|f_A + f_B| / |f_A| :  median 0.0000e+00   max 0.0000e+00
```

On an individual face: elem 2266 gives $+1.027232\times10^{-10}$, elem 2283
gives $-1.027232\times10^{-10}$ — **exactly opposite**, same face-centroid
velocity from both sides, same area, opposite normals.

✅ **Our P1 velocity is $C^0$, so its value at a face centroid is single
valued and the face fluxes are ALREADY CONTINUOUS.**

### ❌ Why that kills LN

[Scu16b] state their premise explicitly: *"the element-wise **constant**
velocity field calculated in (8) yields **discontinuous normal fluxes** across
element faces."*

⚠️ **Their input is an element-wise constant velocity** — a Darcy flux
$-\mathbf K\nabla h$, which is piecewise constant when the head is P1.
⚠️ **Ours is a nodal P1 velocity**, which is continuous. ❌ **So the defect LN
repairs does not exist in our data**: the averaging step is a no-op, the
residual it corrects is identically zero, and the patch solve has nothing to
do.

✅ **The "exactly 0.0" verdict was therefore CORRECT** — it was faithfully
measuring a quantity that is zero by construction for a $C^0$ field. ⚠️ I was
right to distrust it, but for the wrong reason.

### ⚠️ The distinction that matters, and that I had blurred

There are **two different defects**, and only one is ours:

| defect | who has it | fix |
|---|---|---|
| **face-flux discontinuity** | element-wise constant velocities (Darcy) | ✅ LN |
| **nonzero divergence INSIDE each element** | ✅ **ours** — $\sum_F \mathbf v\cdot\mathbf n\,A=\nabla\!\cdot\mathbf v\,V\neq0$ | BDM₁ / SV |

✅ **Verified identity on real elements:** net face flux $=\nabla\!\cdot
\mathbf v\cdot|V|$ to within 0.3 % (ratios 0.997–1.003 over ten elements). So
the net flux *is* the divergence — our field conserves nothing per element even
though its faces match perfectly.

⚠️ **Hence Scudeler being "our exact application" was true of the APPLICATION
(particle tracking on a P1 field) and false of the DEFECT.** The abstract's
*"this condition is not fulfilled by the standard linear Galerkin FEM"* refers
to local mass conservation, which they obtain from a *flux* that is piecewise
constant — not from a continuous nodal velocity.

### ✅ Ranking restored

| rank | route | status |
|---|---|---|
| **1** | ✅ **BDM₁ global projection** | ✅ **proven**: divergence 8.5e-17, preservation ~7 %, scaling verified to 50 k elements |
| 2 | Matrix-free GPU $H(\mathrm{div})$ MINRES [Paz23] | the right full-scale solver for BDM₁ |
| 3 | SV-P₃ | fallback; §4e showed it cannot preserve better |
| ❌ | Larson–Niklasson | **does not apply** — wrong defect |

✅ **BDM₁ remains the only method measured to fix our actual defect**, and the
remaining work is its solver, for which [Paz23] is the literature's answer.

---

## 5. Implementation steps

**Step 1 — sub-mesh proof (½–1 day).** Assemble the $\mathbf K=\mathbf I$
saddle-point system on ~10⁵ contiguous elements of one cylA snapshot; solve
directly (SciPy sparse LU); measure the pointwise divergence of the result.

⚠️ **The pass/fail is sharp.** The constraint is enforced *exactly*, so the
element divergence must be at round-off, **~1e-10 relative or better**. ❌ If it
is not, the assembly is wrong. (The earlier attempt would have been caught in
minutes by this test — it is the step that was skipped.)

**Step 2 — boundary conditions (1–2 days, the underestimated part).** The
system needs $q_F^\perp$ prescribed on $\partial\Omega$. ⚠️ **No boundary-node
extraction exists in the codebase.** Needed: identify boundary faces from the
tet connectivity (a face appearing in exactly one element), classify them
(inlet / outlet / tool / free surface, using the existing `LEVEL` field for the
tool), and impose the P1 normal flux as a Dirichlet condition on each.
⚠️ **Mass must balance globally** or the constrained system is inconsistent:
$\sum_F q_F^\perp=0$ over the closed boundary.

**Step 3 — iterative solve at full scale (2–3 days).** Direct factorisation is
out at 32 M unknowns. Use MINRES or a Schur-complement CG with a block
preconditioner; [Far20] give a Reynolds-robust augmented-Lagrangian
preconditioner for the closely related SV saddle point and is the reference to
follow if convergence is poor.

**Step 4 — ✅ verify BOTH properties, not just one (½ day).**
- divergence: element $\lvert\nabla\!\cdot\mathbf u_h\rvert$ at round-off;
- ⚠️ **preservation**: $\lVert\mathbf u_{RTN_0}-\mathbf u_{P1}\rVert$ at
  centroids, reported as a distribution. **A projection that conserves mass by
  destroying the flow field is not a success.** This is the number that answers
  "while preserving it".

**Step 5 — re-run $\det F$ (1 h + run time).** Track on the projected field with
`--damage-divergence analytic`. ✅ Expect $\lvert\det F-1\rvert$ to fall from
48 % of particles over 1e-2 to integration error alone. ✅ **And re-run the
occupancy test of §1.3(b)** — the 33.4 % empty-cell fraction is the
end-to-end check that the holes were the artefact.

**Step 6 — optional, later: semi-analytical trajectories.** [Sel20] §2.4 gives
element-wise exact exit points for divergent and non-divergent flow in 3D.
✅ On an RTN₀ field this makes $\det F$ exact *by construction* rather than
approximately. ⚠️ A different integrator from our RK4; not a drop-in.

---

## 6. References

- [Sel20] Selzer, P. & Cirpka, O. A. (2020). Postprocessing of standard finite
  element velocity fields for accurate particle tracking applied to groundwater
  flow. *Computational Geosciences* **24**, 1605–1624.
  https://doi.org/10.1007/s10596-020-09969-y
- [RT77] Raviart, P.-A. & Thomas, J. M. (1977). A mixed finite element method
  for 2nd order elliptic problems. In *Mathematical Aspects of Finite Element
  Methods*, Lecture Notes in Mathematics **606**, 292–315.
  https://doi.org/10.1007/BFb0064470
- [Ned80] Nédélec, J.-C. (1980). Mixed finite elements in $\mathbb R^3$.
  *Numerische Mathematik* **35**, 315–341. https://doi.org/10.1007/BF01396415
- [Far20] Farrell, P. E., Mitchell, L., Scott, L. R. & Wechsung, F. (2021). A
  Reynolds-robust preconditioner for the Scott–Vogelius discretization of the
  stationary incompressible Navier–Stokes equations. *SMAI Journal of
  Computational Mathematics* **7**, 75–96. https://doi.org/10.5802/smai-jcm.72
- [Ain21] Ainsworth, M. & Parker, C. (2022). The Scott–Vogelius method for
  Stokes problem on anisotropic meshes. arXiv:2109.14780.
  https://doi.org/10.48550/arXiv.2109.14780
- [Joh20] John, V., Linke, A., Merdon, C., Neilan, M. & Rebholz, L. G. (2017).
  On the divergence constraint in mixed finite element methods for
  incompressible flows. *SIAM Review* **59**(3), 492–544.
  https://doi.org/10.1137/15M1047696
- [Har65] Harlow, F. H. & Welch, J. E. (1965). Numerical calculation of
  time-dependent viscous incompressible flow of fluid with free surface.
  *Physics of Fluids* **8**, 2182–2189. https://doi.org/10.1063/1.1761178
