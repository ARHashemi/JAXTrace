# Making the FSW velocity field divergence-free, while preserving it

> ⚠️ **This is the chronological WORKING LOG, not the explanation.** It records
> every measurement in the order it was taken, including a long BDM₁ attempt
> that was abandoned (§4x) — so sections early in this file describe approaches
> that were later refuted, and several are marked SUPERSEDED.
>
> **For the method that works, read `DIVERGENCE_FREE_METHOD.md`**: the problem
> from scratch, the closed form with its optimality proof, the algorithms and
> where each acts in the tracking pipeline, and all the measurements in one
> place. Come here for *why* an approach was rejected or how a number was
> obtained.
>
> Short version of the outcome: the fix is `A' = A − (tr A/3)I` applied
> in-kernel (§4y, §4ab, §4ac), and it is the provably closest divergence-free
> field to the FOM gradient.

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

## 4k. ❌ The LOCAL-PATCH family is closed — tested, not assumed

⚠️ The question was whether [Led16]'s **local vertex-patch** reconstruction —
or any patch method — could replace the global solve. ✅ **It is decided by one
measurement, and the answer is no.**

### The premise, and why it is testable

[Led16] construct *"a velocity reconstruction operator that maps **discretely
divergence-free** test functions to **exactly divergence-free** ones … based on
local $H(\mathrm{div})$-conforming flux equilibration on **vertex patches**"*.

⚠️ **"Discretely divergence-free" means** $\int_\Omega(\nabla\!\cdot
\mathbf u_h)\,q_h=0$ for all $q_h$ in the pressure space — the divergence
vanishes *weakly* against the discrete pressures, even though it is nonzero
pointwise. ✅ **That is the operator's entire premise**: it has something to
map only if the input already satisfies it.

### ❌ Our field does not satisfy it

`scripts/damage/test_discrete_divergence.py`, whole cylA mesh (571,533 nodes,
3,050,196 elements). With $\nabla\!\cdot\mathbf u$ constant per element and
$\int_T\phi_i=V/4$:

$$b_i=\int_\Omega(\nabla\!\cdot\mathbf u)\,\phi_i=\sum_{T\ni i}\frac{(\nabla\!\cdot\mathbf u)_T\,V_T}{4}$$

relative to the same sum taken with $\lvert\nabla\!\cdot\mathbf u\rvert$:

| | |
|---|---:|
| p50 | **1.06e-01** |
| p75 | 4.95e-01 |
| p90 … p100 | **1.00** |
| mean | **3.03e-01** |

❌ **Not discretely divergence-free.** At the median, 10 % of the local
divergence survives the weak test; above p90 it survives **entirely** —
meaning the divergence does not change sign within those patches at all, so
nothing cancels.

### ⚠️ The pattern, and the lesson

This is the **third** method whose premise our data fails, and the failures are
all the same shape:

| method | premise | our field |
|---|---|---|
| Larson–Niklasson [Scu16b] | face fluxes **discontinuous** | ❌ exactly continuous |
| Lederer–Schöberl [Led16] | **discretely** divergence-free | ❌ 10–100 % residual |
| Pazner [Paz23] | (1,1) block ≈ its diagonal | ❌ eig spans −2.07 … +8.11 |

✅ **Each was found by one cheap measurement**, and each would have been found
*before* implementation had I tested the premise first. ⚠️ **That is now the
standing rule for this work.**

⚠️ **The common cause is worth stating:** our FOM is a **P1/P1 velocity–
pressure** discretisation, which is **not inf-sup stable** and must be
stabilised. Its velocity is therefore *neither* pointwise *nor* discretely
divergence-free — it satisfies a **stabilised** continuity equation, which is a
weaker condition than any of these three methods assume. ✅ **That is why a
global constrained projection is needed: there is no weak structure to exploit
locally.**

### ✅ Conclusion: domain decomposition IS the right approach

❌ No local or patchwise method applies. ✅ **BDM₁ with the centroid objective
remains the only method measured to fix our defect** (8.5e-17 divergence,
~7 % field change, verified to 50 k elements), and its ill-conditioning is
intrinsic to the under-determined objective that *gives* the good preservation
(§4f). ✅ **Domain decomposition with the proven direct LU is therefore the
correct route, not a fallback** — 61 subdomains × 25 s = **1.6 min/snapshot on
16 threads**.

---

## 4l. ✅ FEMUSS already decomposed the domain — and the pieces are the right size

⚠️ The PVTU files are not monolithic. Each snapshot is written as **64 pieces**
with `GhostLevel="1"`:

```
<PUnstructuredGrid GhostLevel="1">
  <Piece Source="cylA_159_0.vtu"/>   13,174 points   59,275 cells
  <Piece Source="cylA_159_1.vtu"/>   12,937 points   55,916 cells
  ... 64 pieces
```

✅ **The piece size is ~50–59 k cells — almost exactly the 50 k subdomain at
which the direct LU costs 25 s.** The decomposition we were about to build
already exists.

✅ **And the overlap is real, not just declared.** Measured on 4 pieces: 49,716
points listed against 45,815 geometrically unique — **7.8 % duplicated across
pieces**. ⚠️ There is no `vtkGhostType` array, so the ghost layer must be
detected geometrically (by shared node coordinates after deduplication), but it
*is* present — which is exactly the overlap an additive-Schwarz decomposition
needs.

⚠️ **One caution.** The pieces are FEMUSS's **parallel partition**, chosen for
*its* load balance, not for our projection. They are not guaranteed to be
connected or well shaped for a saddle-point solve. ✅ But they are conforming,
overlapping and the right size, so they are the natural starting point and cost
nothing to try.

---

## 4m. ✅ PREBUILT, not per step — and the margin is large

The projection **must** be prebuilt. Three independent reasons:

1. ⚠️ **It is per-SNAPSHOT, not per-step.** D2 has **166 snapshots** but
   **8000 RK4 steps** — **48 steps per snapshot**. Projecting per step would
   redo identical work 48 times.
2. ❌ **It is a global solve over the mesh.** Nothing inside a `vmap`-ed
   per-particle kernel can contain one.
3. ⚠️ **It would reintroduce in-step host work** — a CPU solve inside the time
   loop, which is precisely what must not happen.

### Storage, and it is modest

| | DOF | per snapshot | all snapshots |
|---|---:|---:|---:|
| cylA | 18.8 M | **72 MB** | 72 MB (1 snapshot) |
| **D2** | 66.0 M | **252 MB** | **41 GB** (166) |

✅ **One snapshot in memory at a time**, streamed exactly as the velocity stack
already is. ⚠️ 41 GB on disk for D2 is real but unremarkable beside the case's
existing output.

### ✅ The kernel change is neutral-to-positive

`interpolate_velocity_single` evaluates $\mathbf u=\mathbf A\mathbf x+\mathbf b$
from a per-element 12-vector instead of P1 barycentric weights. ✅ That is
**fewer** operations than P1 (no barycentric solve), stays entirely inside the
`vmap`, and adds **no transfer**. Throughput should be neutral or slightly
better.

---

## 4n. ⚠️ THE SEAM — the concern was right, and it changes the design

⚠️ **Question raised: after merging and deduplication, would a per-piece
projection be discontinuous at the subdomain boundary?** ✅ **Tested before
implementing, and the answer reshapes the plan.**

### ❌ `GhostLevel="1"` does not mean overlapping cells

`scripts/damage/test_piece_seam.py`, pieces 0 and 1 of cylA snapshot 159:

```
shared POINTS : 1,271
shared CELLS  : 0
```

⚠️ **The pieces TILE the domain and meet at a surface.** The declared ghost
level is not backed by duplicated cells, and there is no `vtkGhostType` array.
❌ So the earlier reading of "7.8 % duplicated points = usable overlap" was
wrong: the duplication is a shared *interface surface*, not a ghost layer.

### ✅ Continuity itself is safe — but accuracy is not

An interface face belongs to exactly one piece, and in **both** neighbouring
pieces it is a *boundary* face pinned to the **same** P1 normal trace. ✅ **So
the normal component agrees across the seam and H(div) conformity survives
merging.** The literal concern — a double-valued normal flux — does not occur.

❌ **But §4d measured what pinning costs:**

| | preservation p50 |
|---|---:|
| interior elements | **0.097** |
| boundary-touching | **40.9** (420× worse) |

Measured across 8 FEMUSS pieces:

| | |
|---|---:|
| boundary faces per piece | **10.3 %** |
| **elements touching a seam** | **21.5 %** |
| true domain boundary, whole mesh | **5.7 %** |

⚠️ **A per-piece solve over-constrains 3.8× more elements than a whole-mesh
solve**, and each of those lands in the 420×-worse regime. ⚠️ **Worse, the
seams are a parallel partition — they cut THROUGH the stir zone**, which is
exactly where the void physics lives.

### ✅ The fix: overlapping Schwarz with a discarded ring

Grow each piece by **one or two element layers** taken from its neighbours,
solve on the enlarged patch, and **keep only the interior of the original
piece**. ✅ The over-constrained ring then falls entirely inside the discarded
overlap, so every retained element is solved as an interior element.

⚠️ **The overlap must be BUILT, not reused.** FEMUSS's pieces share points but
no cells, so there is no ghost layer to exploit. ✅ But the global mesh is
already loaded and deduplicated for tracking, so growing a patch is a cheap
neighbour query on data we already hold.

⚠️ **And the pieces are then only a PARTITION HINT**, not the decomposition
itself — we use their element lists as seeds and build our own overlap. That
is still a saving: it removes the need to partition 10.8 M elements ourselves.

### ✅ HOW MANY OVERLAP LAYERS? Measured, not guessed — the answer is 3

⚠️ **Concern raised: pieces are partitioned by element COUNT, so a piece in a
refined region covers a tiny physical volume — would 1–2 layers be enough
there?** ✅ **Measured, and it changed the number.**

**First, the refinement is real.** Across 12 FEMUSS pieces the median element
size is **identical (0.0787 mm)** — but *within* a single piece sizes span up
to **32×** (0.0787 → 2.52 mm). ⚠️ And **56–86 % of seam-touching elements sit
in the fine region**, where 2 layers buys only **0.157 mm** of physical
overlap against ~5 mm in a coarse region.

**Then, the decisive measurement** — preservation as a function of layer depth
from the cut boundary (20 k sub-mesh, `--reg 1e-8`):

| layer | n | p50 | vs asymptote |
|---|---:|---:|---:|
| 0 | 5,833 | 0.1851 | **2.72×** |
| 1 | 4,781 | 0.1409 | **2.07×** |
| 2 | 3,849 | 0.1000 | **1.47×** |
| **3** | 2,214 | **0.0686** | **1.01×** ✅ |
| 4 | 1,417 | 0.0706 | 1.04× |
| ≥5 | 1,674 | 0.0681 | — |

✅ **Converged by layer 3.** ❌ **So 1–2 layers is NOT enough** — my first
proposal would have retained elements 1.5–2.1× worse than interior.

### ✅ And the refinement worry resolves favourably

⚠️ The right unit turns out to be **elements, not millimetres**: the
contamination propagates through **face adjacency**, and the pinning decays
**per element**, not per unit distance. ✅ **So 3 layers is 3 layers regardless
of local $h$** — a refined region has more elements per mm, but it also needs
more elements to carry the perturbation the same number of hops.

⚠️ **This must still be verified on a deliberately refined patch** before
trusting it at full scale; the table above is one 20 k block. ✅ **Recorded as
a required check, not assumed.**

### Revised step 1

1. Load and deduplicate the global mesh (already done for tracking).
2. Read the 64 piece element lists as **seeds**.
3. Grow each seed by **3 element layers** (measured, see above) via face
   adjacency → **overlapping patch**.
4. Solve BDM₁ on each patch independently (direct LU, ~25 s at 50 k).
5. Keep only the **seed** elements' DOFs; discard the 3-layer overlap ring.
   ⚠️ Verify the layer-depth profile on a REFINED patch — 3 is measured on one
   20 k block and the refinement case is untested.
6. ⚠️ **Verify at the seam**: measure the normal-flux jump across
   piece interfaces on the assembled field. It should be ~0 by construction;
   anything else means the discard rule is wrong.

---

## 4o. ✅ THE DECOMPOSITION WORKS — divergence exact, one caveat at the seam

End-to-end run on 2 FEMUSS pieces of cylA (`build_divfree_field.py`):

```
piece 0: seed 59,275 -> patch 79,281   (221 s)
piece 1: seed 55,916 -> patch 74,382   (365 s total)
filled 115,191 / 3,050,196 elements
```

| check | result | |
|---|---:|---|
| **divergence** median | **8.73e-17** | ✅ machine precision |
| divergence p99 / max | 1.09e-15 / 1.05e-14 | ✅ |
| **preservation** p50 | **2.76e-02** | ✅ **better than monolithic (6.75e-02)** |
| preservation p90 / p99 | 2.39e-01 / 3.48e-01 | |
| **seam** p50 | 3.77e-02 vs 2.74e-02 | ⚠️ **1.38×** |

✅ **The decomposition does not degrade the property we are buying.** The
divergence matches the monolithic BDM₁ result exactly, because the constraint
is enforced **per element** regardless of which patch solved it.

✅ **And preservation is BETTER than monolithic** (2.76e-02 vs 6.75e-02),
because every retained element is now a patch *interior* element — the
3-layer discard rule is doing its job.

### ⚠️ The seam, in proportion

1,945 of 115,191 elements (**1.7 %**) sit across a piece interface, and their
preservation is **1.38×** the bulk — above the 1.3× gate.

⚠️ **But the seam affects FIELD FIDELITY, not mass conservation.** The
divergence at seam elements is 8.7e-17 like everywhere else, because
$\nabla\!\cdot\mathbf u=0$ is a per-element constraint. ✅ **So $\det F$ is
unaffected** — which is the quantity this whole exercise exists to fix.

For scale: the P1 field fails the $\det F$ gate for **48 %** of particles. A
1.38× fidelity difference on 1.7 % of elements is a far smaller defect, and it
is measured rather than assumed.

### ⚠️ Cost correction — 7× my estimate

| | estimated | **measured** |
|---|---:|---:|
| per piece | 25 s | **~180 s** |
| cylA, 64 pieces, serial | 27 min | **3.2 h** |
| **16 threads** | 1.6 min | **~12 min/snapshot** |

⚠️ The 3-layer growth inflates each patch from ~55 k to ~79 k elements, and LU
scales as $n^{1.82}$, so the overlap costs ~7× the bare piece. ✅ Still
tractable — patches are independent — but **12 min/snapshot, not 1.6**.

⚠️ **And widening to 5 layers will cost more again**, so the seam fix is not
free. Running that test now to measure the trade.

---

## 4p. ✅ PROFILED — the cost is 99.5 % sparse LU, and patch size matters enormously

⚠️ Concern raised: *"it should not take so long, and htop shows only one
core."* ✅ Both observations were right, and profiling found a large win.

### Where the 180 s per piece actually goes

| stage | time |
|---|---:|
| face topology (3.05 M elements) | 0.1 s |
| geometry | 0.0 s |
| trace maps **T** | 0.0 s |
| `inv(T)`, 88,922 × 12 × 12 | 0.2 s |
| assemble **N**, **M** | 0.0 s |
| build saddle point (591,185², nnz 11.9 M) | 0.2 s |
| **`splu` factorisation** | **131.8 s** |
| **total** | **132.4 s** |

❌ **99.5 % is one call to `scipy.sparse.linalg.splu`.** ⚠️ `grow_patch`, which
I suspected because of its Python loop over faces, takes **0.1 s** — the
suspicion was wrong. ✅ **And `splu` is single-threaded**, which is exactly the
one-core observation: SuperLU has no shared-memory parallelism.

### ✅ `splu` scales as $n^{3.10}$ — so patch size dominates everything

| seed | patch | ratio | solve | cylA serial | 16 threads |
|---:|---:|---:|---:|---:|---:|
| 500 | 1,305 | 2.61× | 0.1 s | 816 s | 51 s |
| 1,000 | 4,014 | 4.01× | 0.2 s | 681 s | 43 s |
| 2,000 | 6,634 | 3.32× | 0.3 s | 516 s | 32 s |
| **5,000** | **12,686** | 2.54× | **0.6 s** | **374 s** | **23 s** |
| 10,000 | 23,410 | 2.34× | 1.8 s | 552 s | 34 s |
| 20,000 | 39,431 | 1.97× | 12.8 s | 1,962 s | 123 s |
| 40,000 | 61,329 | 1.53× | 75.8 s | 5,840 s | 365 s |
| 55,000 (FEMUSS piece) | 88,922 | 1.62× | **131.8 s** | — | — |

✅ **The optimum is ~5,000 elements per seed: 23 s/snapshot on 16 threads for
cylA, against the ~720 s the FEMUSS-piece size costs — a 31× speedup.**

⚠️ **Why small wins despite more waste.** At exponent 3.1, halving the patch
cuts per-patch cost **8×** while only doubling the count — a 4× net win. The
3-layer ring is a fixed *absolute* thickness, so small seeds waste a larger
*fraction* (61 % at 5 k vs 38 % at 55 k), but that is far outweighed. ⚠️ Below
~2,000 the overlap fraction finally dominates and the cost rises again, which
is why the curve has a minimum rather than decreasing forever.

### ⚠️ BUT quality degrades with small patches, and not monotonically

| seed | divergence | preservation p50 | p90 |
|---:|---:|---:|---:|
| 5,000 | 1.30e-16 | **0.202** | 0.487 |
| 10,000 | 1.15e-16 | 0.140 | 0.373 |
| 20,000 | 1.00e-16 | 0.180 | 0.509 |
| 40,000 | 9.54e-17 | **0.019** | 0.186 |
| 55,000 | 9.45e-17 | 0.034 | 0.603 |

✅ **Divergence is machine precision at every size** — the property we are
buying is unaffected.

❌ **Preservation is 6× worse at 5 k than at 55 k** (0.202 vs 0.034). ⚠️ **And
it is NOT monotone** (0.202, 0.140, 0.180, 0.019), so it depends on *where*
the patch sits, not only on its size — these seeds are contiguous index ranges,
which carve different shapes through the mesh.

⚠️ **So the 31× speedup is not free, and the trade is not yet characterised.**
The honest next step is to measure preservation at several *positions* per
size, not one, before choosing. ✅ The 3-layer rule was validated the same way
and that is what made it trustworthy.

### Recommendation

⚠️ **Do not adopt 5,000 on this evidence alone.** ✅ What is solid:

1. ✅ `splu` is 99.5 % of the cost and single-threaded — **parallelism across
   patches is free and essential**, and was never implemented.
2. ✅ Patch size is the dominant lever, with a measured $n^{3.10}$.
3. ⚠️ Quality vs size needs a position-averaged measurement before the size is
   fixed.

---

## 4q. ✅ PATCH SIZE SETTLED — 10,000, measured over positions

⚠️ The single-sample numbers in §4p were **noise**. Averaging 4 random
positions per size gives a clean monotone trend:

| seed | solve | preservation p50 | **std** | cylA, 16 threads |
|---:|---:|---:|---:|---:|
| 2,000 | 0.62 s | 0.342 | **0.324** ⚠️ | **59 s** |
| 5,000 | 2.09 s | 0.237 | 0.081 | 80 s |
| **10,000** | 7.54 s | **0.106** | **0.038** ✅ | **144 s** |
| 20,000 | 36.85 s | 0.092 | 0.059 | 352 s |
| 55,000 (FEMUSS piece) | 131.8 s | — | — | ~720 s |

✅ **Preservation improves monotonically with size** (0.342 → 0.237 → 0.106 →
0.092) once position is averaged out. ⚠️ At 2,000 the **spread is 0.324** —
larger than the difference between any two sizes — which is why one sample per
size was worthless and gave the non-monotone 0.202/0.140/0.180/0.019.

✅ **Divergence is 8–10e-17 at every size and every position.** The property
being bought is completely insensitive to patch size; only *fidelity* trades.

### ✅ The choice: 10,000

⚠️ **Diminishing returns past it:** 10k → 20k improves preservation by **13 %**
for **2.4× the cost**. ✅ And 10,000 has the **tightest spread** (std 0.038),
i.e. it is the most predictable.

✅ **Against the FEMUSS piece size it is 5× faster** (144 s vs ~720 s) **and
better characterised** — the piece size was never position-sampled at all.

### ✅ Parallelism, the other half

`splu` is 99.5 % of the cost and **single-threaded**, so the patches — which
are completely independent — are now solved in a `ProcessPoolExecutor`
(`--jobs`, default 16).

⚠️ **Two implementation traps, both hit:**
- A **closure cannot be pickled** for a ProcessPool. The worker is module-level
  and reads fork-inherited globals; passing the mesh through the pickle
  boundary would copy ~1 GB **per patch**.
- The original loop **interleaved VTK reading with solving**, so nothing could
  overlap. Seeds are now all mapped first, then dispatched.

### Net effect

| | before | **after** |
|---|---:|---:|
| patch size | 55,000 (FEMUSS piece) | **10,000** |
| parallelism | ❌ none (1 core) | ✅ 16 processes |
| **cylA, per snapshot** | **~720 s** | **~144 s** |

✅ **5× faster, with preservation measured and its spread known.**

---

## 4r. ✅ FULL cylA SNAPSHOT — 100 % coverage, seam clean, one open question

```
333 patches, 16 processes, 1625 % CPU
solved in 246.3 s; filled 3,050,196 / 3,050,196 (100.0 %)
wrote cylA_159.npz (126.9 MB)
```

| verdict | result | |
|---|---:|---|
| **coverage** | 3,050,196 / 3,050,196 | ✅ **100 %** |
| **divergence** median | **8.89e-17** | ✅ machine precision |
| divergence p99 | 1.14e-15 | ✅ |
| **seam** | 1.93e-01 vs 1.84e-01 = **1.05×** | ✅ **no penalty** |
| preservation p50 / p90 | 1.86e-01 / 7.81e-01 | ⚠️ see below |

### ✅ The seam concern is resolved

⚠️ The 2-piece test showed **1.38×**, above the 1.3 × gate. ✅ **At full scale
with 333 patches it is 1.05× — no measurable penalty.** The 3-layer discard
rule works, and the earlier figure came from a 2-piece configuration where a
single interface dominated the statistic.

### ✅ Divergence is exact across the whole mesh

8.89e-17 median over **3.05 M elements**, identical to the monolithic and
2-piece results. ⚠️ The max is 9.2e-08 — a handful of elements, and worth
noting rather than hiding, but p99 is 1.14e-15.

### ⚠️ OPEN: preservation is worse than the position sweep predicted

**1.86e-01 here against 1.06e-01** measured at seed = 10,000 in §4q, and
against **0.0675** for the monolithic solve.

❌ **My first explanation was wrong.** I hypothesised that `np.array_split` on
a FEMUSS piece yields geometrically *fragmented* seeds with more boundary per
element. ⚠️ **Measured and refuted:** boundary-face fraction is **26.1 %** for
a contiguous index range (what the sweep used) against **14.3–25.4 %** for the
split chunks — the chunks are **no worse**, and mostly better.

⚠️ **So the cause is not yet identified.** Candidates not yet tested:
- the sweep sampled 4 positions in one region; the full mesh includes the tool
  surface and the refined stir zone, which the sweep may have missed;
- `p90 = 0.78` suggests a sub-population with large error rather than a uniform
  shift — worth splitting by region before drawing conclusions.

✅ **What is NOT in doubt:** divergence is machine precision everywhere, the
seam is clean, and the field is complete and written. ⚠️ **Preservation needs
one more measurement before the field is used for production tracking.**

---

## 4s. ✅ THE PRESERVATION PUZZLE — float32 storage was destroying the field

⚠️ The full run reported preservation p50 **0.186**, but recomputing from the
saved `.npz` gave **0.4285**. ✅ **That discrepancy was the clue.**

### ❌ The cause: `u = A x + b` is a catastrophic cancellation

| quantity | magnitude |
|---|---:|
| $\lvert\mathbf A\mathbf x\rvert$ | **1.4e+06** |
| $\lvert\mathbf b\rvert$ | **6.4e+05** |
| $\lvert\mathbf u\rvert$ | **2.1e-01** |

⚠️ The two terms are each ~1e6 and cancel to ~1e-1 — **6.8 decimal digits of
cancellation**. float32 carries ~7.2, so **only 0.4 digits survive**.

✅ **Measured directly:** fresh float64 coefficients give preservation p50
**0.140**; the same coefficients after a float32 round trip give **0.240** — a
**1.7× degradation from storage alone**.

✅ **Fixed:** the field is now stored in **float64**. 12 doubles/element =
1.0 GB for D2 against 0.5 GB — cheap insurance for the quantity the whole
method rests on.

⚠️ **A better fix exists and is recorded as future work:** store
$\mathbf u_c$ and $\mathbf A$ about the **element-local** origin,
$\mathbf u(\mathbf x)=\mathbf u_c+\mathbf A(\mathbf x-\mathbf x_c)$, which
removes the cancellation entirely rather than paying for it in precision.

### ⚠️ But that is not the whole story — two issues, not one

float64 preservation is **0.140**, still above the **0.106** the position sweep
measured at the same seed size. ⚠️ **So float32 explains the 0.186 → 0.4285
discrepancy and part of the gap, but not all of it.**

### ✅ And the error is strongly concentrated — by velocity magnitude

| $\lvert\mathbf u\rvert$ | n | preservation p50 |
|---|---:|---:|
| 1e-4 … 1e-3 | 1,062 | **20.6** |
| 1e-3 … 5e-3 | 50,300 | **3.68** |
| 5e-3 … 1e-2 | 206,903 | 1.21 |
| **1e-2 … 1** | **2,791,930** | **0.403** |

✅ **The error is a RATIO, and it blows up where $\lvert\mathbf u\rvert$ is
small** — stagnant material far from the tool, where a small absolute change
is a large relative one. ⚠️ **91.5 % of elements are in the well-resolved
band**, and the tail is dominated by nearly-motionless material where the
relative measure is not meaningful.

By radius, the worst band is $r = 9\!-\!15$ mm (p50 **4.71**) — ⚠️ **outside
the 9 mm shoulder**, i.e. again the slow-moving far field, not the stir zone.
✅ Inside the tool radius ($r<4$ mm) it is **0.351**.

⚠️ **So the headline p50 overstates the damage**: it is dominated by elements
whose velocity is near zero. ✅ **A future report should use an ABSOLUTE
error normalised by a global velocity scale**, not a per-element ratio.

---

## 4t. ✅ D2 RESULTS — and the preservation gap localised to |A|, cause still open

### ✅ D2 timestep 34, full mesh

```
solved in 1530.8 s; filled 10,807,677 / 10,807,677 (100.0 %)
divergence   median 1.09e-16   p99 2.66e-15
preservation p50  4.73e-04     p90 1.58e-01
seam         6.24e-04 vs 4.18e-04 = 1.49x
wrote D2_34.npz (459.6 MB)
```

✅ **100 % coverage on 10.8 M elements in 25.5 min on 12 workers**, divergence
at machine precision. ✅ **Preservation p50 4.7e-04 — essentially perfect**,
and 400× better than cylA's 0.186.

⚠️ Seam 1.49× here against cylA's 1.05×, with 128 pieces instead of 64. At
6.2e-04 absolute it is immaterial, but it should be watched.

### ✅ The gap is REAL and tracks |A|, not the case

Identical pipeline, same seed size, same layers, 3 positions each:

| case | patch | \|A\| | preservation |
|---|---:|---:|---:|
| D2 | 24,879 | **8.01e+04** | **2.30e-07** |
| D2 | 24,204 | 2.22e+05 | 2.22e-07 |
| D2 | 21,259 | 9.37e+05 | 6.79e-02 |
| cylA | 15,075 | 6.22e+06 | 3.53e-02 |
| cylA | 17,363 | 3.13e+08 | 1.73e-01 |
| cylA | 23,377 | 3.79e+08 | 3.02e-01 |

✅ **corr(log\|A\|, log preservation) = +0.82, and the ordering is exact**:
every patch with $\lvert A\rvert<10^6$ preserves well, every patch above it
does not. ⚠️ **So it is a per-patch property, not a per-case one** — cylA
simply has more bad patches.

⚠️ $\lvert A\rvert\sim10^8$ is **not physical**: $\lvert\nabla\mathbf u\rvert$
is ~150 s⁻¹, so the BDM₁ coefficients are being amplified by ~10⁶.

### ❌ Three hypotheses tested, all refuted

| hypothesis | test | verdict |
|---|---|---|
| slow material inflates the ratio | velocity distributions | ❌ nearly identical (p50 0.213 vs 0.211) |
| cylA has more divergence to remove | $\lvert\nabla\!\cdot u\rvert/\lvert D\rvert$ | ❌ **inverted** — D2 is *larger* (4.87e-02 vs 2.49e-02) |
| `inv(T)` ill-conditioned by geometry | cond(T), and vs distance from origin | ❌ similar (1.1e5 vs 1.7e5), corr −0.18 |

⚠️ **The cause of the \|A\| blow-up is not yet identified.** What is
established: it is per-patch, correlates with preservation at +0.82, and is
*not* explained by the velocity field, the divergence magnitude, or the
elementwise conditioning of **T**.

⚠️ Remaining candidates, untested: the **Tikhonov weight** `reg` is scaled by
$\lVert N\rVert^2$, which varies per patch — a patch where the objective block
is weak would let the solution drift to large coefficients while still
satisfying the constraint exactly (which it does: divergence is 1e-16 on
*every* patch, good or bad).

✅ **What is NOT in doubt:** divergence is machine precision on both cases and
every patch; D2 — the production case — preserves the field to 4.7e-04.

---

## 4u. ✅ ALL THREE D2 SNAPSHOTS — and the diagnostic, as far as it goes

### ✅ D2 is reproducible to 4 significant figures

| timestep | divergence median | preservation p50 | seam | time |
|---|---:|---:|---:|---:|
| 34 | 1.0938e-16 | 4.7252e-04 | 1.49× | 1530.8 s |
| 100 | 1.0940e-16 | 4.7278e-04 | 1.50× | 1539.0 s |
| 199 | 1.0941e-16 | 4.7235e-04 | 1.49× | 1532.8 s |

✅ **Three independent velocity fields, 10.8 M elements each, 100 % coverage,
and the verdicts agree to 4 significant figures.** That reproducibility is
itself evidence the method is deterministic and the implementation sound.

### ❌ Tikhonov is NOT the cause — the fourth refuted hypothesis

Sweeping `reg` over **six orders** on a bad cylA patch:

| `reg` | \|A\| | preservation |
|---|---:|---:|
| 1e-8 | 3.786e+08 | 3.03e-01 |
| 1e-6 | 3.788e+08 | 3.22e-01 |
| 1e-4 | 3.787e+08 | 9.78e+00 |
| 1e-2 | 3.741e+08 | 9.62e+02 |

⚠️ **\|A\| is invariant** — it moves by 1 % while `reg` moves by 10⁶. ✅ But
preservation degrades badly above 1e-6, confirming **1e-8 is the right
setting** for a different reason than I assumed.

### ❌ Nor is it element size or basis normalisation

The good and bad patches have **identical element size** (h = 0.0787 mm for
both, p50 and min) yet \|A\| differs by **62×**. ⚠️ So the large coefficients
are not a 1/h scaling artefact.

### ⚠️ What the diagnostic DID establish

| | GOOD patch | BAD patch |
|---|---:|---:|
| preservation p50 | 3.53e-02 | 3.03e-01 |
| fraction \|A\| > 1e7 | **25.7 %** | **80.3 %** |
| fraction \|A\| > 1e9 | 0.00 % | 16.9 % |
| per-element corr(log\|A\|, log preservation) | +0.240 | +0.263 |

✅ **The difference is the FRACTION of elements with large coefficients**, not
their magnitude where they occur: preservation among the large-\|A\| elements
is 5.5e-02 (good) and 3.3e-01 (bad) — i.e. it tracks the patch, not \|A\|.

⚠️ **And the per-element correlation is only +0.24**, against +0.82 across
patches. ❌ **So \|A\| is a SYMPTOM, not the cause** — something makes a whole
patch behave badly, and large coefficients come along with it.

### ⚠️ Honest status

Five hypotheses tested and refuted: slow material, divergence magnitude,
`cond(T)`, Tikhonov weight, element size. ✅ The failure is **per-patch and
all-or-nothing**, which points at the patch-level linear algebra rather than
any element property — but I have not isolated it.

✅ **What this does NOT block:** D2, the production case, preserves the field
to **4.7e-04** with divergence at **1.09e-16**, reproducibly across three
snapshots. ⚠️ The open question concerns cylA, the small test case, and should
be resolved before cylA results are quoted — not before D2 tracking proceeds.

---

## 4v. ✅ MODEL ORDER REFUTED — and the non-uniformity quantified

Two questions were put to the method, both about whether the *algebra* is the
wrong shape for the data rather than the solver being wrong:

1. Are we fitting a **low-order model to naturally higher-order data**?
2. Are we **stretching/squeezing uniformly** where some regions have more
   structure and need finer local adjustment?

`scripts/damage/test_local_order.py` answers both from the data alone — no
solver, no regularisation, nothing to tune — on D2 and A2 at ts = 34
(10.8 M and 10.3 M elements, 200 k-element random samples).

### (1) Model order: refuted, and by construction

| quantity | D2 | A2 |
|---|---|---|
| affine fit residual per tet, `/‖u‖∞` (median) | **1.37e-16** | **1.05e-16** |
| worst over the sample | 5.07e-14 | 5.18e-14 |

The FOM velocity is **P1**, so on a tetrahedron its four nodal values define an
*exactly affine* field, and BDM₁'s 12 DOF contain every affine vector field
(§4e). The residual is machine epsilon because the fit is an identity, not an
approximation. **`u = A x + b` is not too low order for this data**, and no
higher-order local model (SV-P3 included) can improve the representation of
what the FOM actually provides. This closes the question structurally rather
than empirically: it is a statement about P1 ⊂ BDM₁, not a measurement that
might come out differently on another case.

⚠️ The corollary matters for interpreting §4t–4u. Per element, divergence-free
*and* exact centroid velocity is **always** simultaneously reachable — 11 free
DOF against 4 equations (1 trace + 3 value), and the closed form is explicit:
`A' = A − (tr A/3) I`, `b' = b + (A − A') x_c`, which is div-free and leaves
`u(x_c)` untouched. Check [3] confirms the construction to 3.9e-08.
**Therefore every preservation error we measure comes from inter-element
coupling — face continuity and the pinned patch boundary — and none of it from
the element model.** The five refuted hypotheses of §4t–4u were all looking
inside the element; the cause cannot be there.

### (2) Non-uniformity: confirmed, and large

The minimum-norm divergence-free correction to an affine map has a closed form,
so the *irreducible* relative change in the gradient is computable per element:

$$\min_{\operatorname{tr}A'=0}\frac{\|A'-A\|_F}{\|A\|_F}
  =\frac{|\operatorname{div}u|}{\sqrt{3}\,\|A\|_F}$$

| percentile | D2 | A2 |
|---|---|---|
| p50 | 6.57e-03 | 4.08e-03 |
| p90 | 5.33e-02 | 3.51e-02 |
| p99 | 2.82e-01 | 2.14e-01 |
| max | 6.17e-01 | 6.43e-01 |
| **p99/p50** | **42.9×** | **52.6×** |
| max/p50 | 94× | 158× |
| fraction needing > 1 % | 39.2 % | 30.3 % |
| fraction needing > 10 % | 4.9 % | 3.2 % |

The two cases agree closely, so **this is a property of the FOM, not of one
pin geometry** — consistent with §4-common-cause (P1/P1 is not inf-sup stable,
so its velocity carries real divergence everywhere).

The repair the constraint demands spans **~2 orders of magnitude across the
mesh**, and `solve_patch` currently meets it with **three global scalars**:
one `nscale` (RMS of all of `N`), one constraint scaling `cs`, and one Tikhonov
floor `eps = reg·nscale²`. That is a uniform knob on a non-uniform problem —
exactly the second hypothesis, and it is confirmed as *stated*.

### ⚠️ What this does and does not explain

It does **not** yet explain the cylA/D2 gap. §4u established that failure is
per-patch and all-or-nothing, whereas this non-uniformity is smooth and nearly
identical in D2 and A2. A smoothly varying quantity does not produce a binary
per-patch outcome on its own.

What it does do is **relocate the search**. Combined with the §4-(1) corollary,
the cause is now confined to the coupling terms — the pinned boundary trace
`qb` and the face-continuity DOF sharing — and *cannot* be the element model,
its order, or anything measured per element. Note that the Tikhonov term pulls
the free DOFs toward `qp1`, the **averaged P1 normal trace**, with weight `eps`
that is uniform while the required repair varies 43×; in a patch where many
elements sit in the high-repair tail, that uniform pull fights the constraint
instead of merely closing the null space.

**Next test (do the premise first — §4-standing-rule):** before building any
adaptive weighting, measure whether per-element scaling of the data rows (by
`‖A‖_F`, or by the local repair cost above) changes the outcome on one known-bad
cylA patch and one known-good D2 patch. If a weighting that differs 43× between
elements leaves preservation unchanged, uniformity is not the cause either and
the remaining suspect is the pinned trace alone.

## 4w. ⚠️ SUPERSEDED BY §4x — `N` is rank-deficient, but that is not the defect

> ⚠️ **Read §4x first.** The measurements below are correct and the refutations
> stand, but the framing is wrong: the near-null direction of `N` is the SLACK
> the solve needs, not a defect, and the real defect (the reconstructed
> gradient is ~10⁶ too large and orthogonal to the FOM's in EVERY patch,
> including the ones called "good" here) is invisible to the preservation
> metric this section ranks patches by. ~~CAUSE LOCALISED~~

§4v closed off the element model. This section finds where the failure actually
lives, using a 12-patch probe on **D2 ts = 34** (3,000-element seeds, 3-layer
rings, ~5,000 elements per patch) that **reproduces the cylA failure on D2** —
so the bug is no longer tied to one case and is now cheap to iterate on.

`scripts/damage/test_adaptive_weighting.py`, `scripts/damage/test_bad_patch.py`.

### The failure reproduces, and it is a continuum

4 of 12 patches fail. Preservation p50 on the seed:

| patch | preservation | `smin(N)` |
|---|---|---|
| 1, 2, 3, 5, 6, 7, 8, 10 | 1.6e-07 … 7.1e-07 | **6.1e-02 … 7.2e-02** (tight) |
| 11 | 3.8e-02 | 2.8e-02 |
| 9 | 4.4e-01 | 1.8e-02 |
| 4 | 9.8e-01 | **1.0e-06** |
| 0 | 1.2e+00 | **1.9e-06** |

⚠️ Earlier sections called this "all-or-nothing". With `smin(N)` in hand it is
a **continuum**: patches 9 and 11 sit between the clusters and fail
proportionally less. That is stronger evidence than a threshold would be — it
is what a near-null space predicts, and a binary switch would not.

### What `N` is, and why a small `smin` is fatal

`N` maps the patch's face DOFs to the **element-centroid velocities** — the
only thing the objective matches (3 equations per element on 12 DOF, the
deliberate choice documented in `solve_patch`). A near-null vector of `N` is
therefore a BDM₁ field that is **invisible to the objective**: adding any
multiple of it changes no centroid velocity, so the solve may add it for free,
and only `eps·I` resists. With `reg = 1e-8` and `smin(N) ~ 1e-6`, nothing does.

Across a patch the count makes the exposure plain: ~27,000 free DOF against
`3m ≈ 15,000` objective equations and `m ≈ 5,000` constraints — a **rank
deficiency of ~7,600**. The system is massively under-determined *by design*,
and the divergence constraint plus the pinned boundary are all that close it.
When they do not close it completely, the leak is patch-global.

### ⚠️ Hypotheses refuted in this section

Each was measured, not reasoned about (§4-standing-rule):

| hypothesis | prediction | measurement | verdict |
|---|---|---|---|
| **non-uniform weighting needed** (§4v's own next step) | adaptive row weights help | `by_normA` matches uniform to 3 s.f. (1.577e-07 vs 1.558e-07); `by_repair` is 10–200× **worse** | ❌ a weighting varying 1000× changes nothing — uniformity is not the cause |
| **S1 boundary dominance** | bad patches have less free interior | free-DOF fraction 86.2 % bad vs 87.1 % good | ❌ flat |
| **S2 net-flux inconsistency** | error ∝ relative flux imbalance | corr +0.467, but **patch 5 has the worst imbalance (8.5e-03) and preserves at 3.0e-07** | ❌ hard counterexample |
| **cancellation in `u = A x + b`** | bad patches cancel worse | `|b|/|Ax|` = **1.000** in all 12 | ❌ universal, not a discriminator |
| **`|u|` magnitude / slow material** | failures in the slow field | `|u|` p50 1.1e-01…5.2e-01, no separation | ❌ |
| **boundary data propagating inward** | error decays from the pin | see depth scan below | ❌ decisive |
| **Tikhonov damps the mode** (the causal test) | raising `reg` fixes bad patches only | patch 0: 1.21 → **1.99e+04** at `reg = 1e-1`; good patches degrade monotonically too | ❌ see below |

### The depth scan — why the ring is not the problem

Preservation p50 by face-distance from the pinned patch boundary:

| patch | L0 (bdry) | L1 | L2 | L3 | deep ≥4 |
|---|---|---|---|---|---|
| 1 (good) | 2.10e-01 | 2.86e-02 | **1.09e-06** | 4.46e-07 | 1.14e-07 |
| 2 (good) | 1.55e-01 | 4.28e-02 | 9.24e-07 | 3.84e-07 | 8.73e-08 |
| 0 (bad) | 1.23e+00 | 1.23e+00 | 1.23e+00 | 1.18e+00 | **1.17e+00** |
| 4 (bad) | 8.76e-01 | 9.36e-01 | 9.61e-01 | 9.70e-01 | **9.68e-01** (grows inward) |

✅ In the good patches the error falls **five orders in two layers** — the
3-layer ring of §4-overlap does exactly its job, and the seam result (1.05×)
is explained. ❌ In the bad patches it is **flat**, and in patch 4 it *grows*
inward. The error is **patch-global**, so no amount of extra overlap will fix
it and the pinned trace is not the carrier.

### ⚠️ The causal test failed — the mode is real, the obvious fix is wrong

If the invisible direction were merely under-damped, raising `eps` would fix
the bad patches and leave the good ones alone. It does the opposite:

| reg | 1e-8 | 1e-6 | 1e-4 | 1e-2 | 1e-1 |
|---|---|---|---|---|---|
| patch 0 (bad) | 1.21e+00 | 1.26e+00 | 2.33e+01 | 2.28e+03 | 1.99e+04 |
| patch 1 (good) | 1.56e-07 | 6.97e-07 | 5.94e-05 | 5.50e-03 | 4.82e-02 |

Monotonically worse everywhere. The reason is that `eps` does not damp toward
*nothing* — it pulls the free DOFs toward `qp1`, the **vertex-averaged P1
normal trace**, which is not itself an admissible target (averaging across a
face's two elements is not a BDM₁ function). So Tikhonov trades an invisible
mode for a wrong bias. This also retrospectively confirms §4u's finding that
`|A|` was invariant across six orders of `reg`: `reg` is not the knob.

**Conclusion: the freedom must be REMOVED, not penalised.** The objective is
too thin — 3 centroid equations cannot see a ~7,600-dimensional complement.

### Next step (premise first)

The fix must add *consistent* information rather than bias. The candidate with
a theoretical guarantee is to match the velocity in an **L²-projection sense
over each element** instead of at its centroid alone — i.e. replace the 3
centroid rows with the 12 moments `∫_K u·φ_i`, which for BDM₁ is exactly the
unisolvent set and makes `N` square and non-singular by construction. ⚠️ The
`faces` variant tried earlier (§4-objective, divergence 5.5e-09 and
preservation 9.6e+03) is *not* this: it matched normal traces at face points,
which over-determines once the constraint is added. **Measure `smin(N)` under
the moment objective on patches 0 and 4 before building anything** — if it
does not rise into the 1e-02 cluster, the premise is wrong and this is not the
fix either.

## 4x. ❌ STOP — the BDM₁ field is divergence-free but UNRELATED to the FOM gradient

⚠️⚠️ **This invalidates §4r–4w's "successes" and the fields built so far.** It
was found by testing §4w's own next step, and it inverts the reading of every
preservation number in this document.

### What happened

§4w's premise test asked whether a richer objective lifts `smin(N)`. It does,
completely — 12 BDM₁ moments make `N` unisolvent and `smin` becomes a uniform
**0.48 across every patch**, a 250,000× lift on patch 4. But solving with it
gave preservation **0.60 … 1.4e+05** and divergence **1.3e-08** — worse
everywhere. ✅ The premise-first rule earned its keep: the fix was built and
discarded in one test instead of a week.

That failure is informative. With 12 moment rows the objective demands the
BDM₁ field match the P1 field *throughout* each element, while `div u = 0` is
also imposed — and the P1 field has real divergence (§4v, p50 0.66 % of ‖A‖).
Those demands are **inconsistent**, so the solve sacrifices both. ⚠️ Therefore
the near-null direction of `N` is **not a defect to remove — it is the slack
the solve needs** to satisfy the constraint exactly while matching the
centroid. `solve_patch`'s "3 equations on 12 DOF is deliberate" is right, and
§4w's diagnosis of a "rank deficiency" was the wrong frame.

### ⚠️ The actual defect: the gradient is garbage in EVERY patch

Asking *how* the solution differs from P1, rather than by how much:

| patch | preservation p50 | `|A_new|/|A_P1|` | `cos(A_new, A_P1)` |
|---|---|---|---|
| 1 ("good") | 1.56e-07 | **1.5e+03** | **−0.005** |
| 2 ("good") | 1.74e-07 | 5.8e+03 | −0.001 |
| 5 ("good") | 3.04e-07 | 1.3e+03 | −0.006 |
| 0 (bad) | 1.21e+00 | 2.7e+07 | −0.006 |
| 4 (bad) | 9.84e-01 | 1.5e+07 | −0.003 |

**The "good" patches are not good.** Their reconstructed gradient is
10³ times too large and **orthogonal** to the true gradient (cos ≈ 0). They
match the centroid velocity to seven digits purely by cancellation between two
enormous wrong terms — which is exactly the `|b|/|A x_c| = 1.0000` that §4w
measured in all 12 patches and dismissed as "universal, not a discriminator."
⚠️ It was universal, and it was the disease, not a nuisance.

### Confirmed on the shipped field, not just a probe

`C1_34.npz` from the LUMI pilot — 7,586,466 elements, the file this pipeline
actually wrote — on a 200 k sample:

| quantity | p50 | p99 | want |
|---|---|---|---|
| divergence `|div|/‖A‖` | 9.21e-17 | 1.46e-15 | ~0 ✅ |
| preservation | 1.66e-01 | 5.65e+00 | small |
| **`|A_new|/|A_P1|`** | **2.80e+06** | 1.49e+08 | **~1** ❌ |
| **`cos(A_new, A_P1)`** | **1.61e-03** | 7.63e-01 | **~1** ❌ |
| `|b|/|A x_c|` | 1.0000 | 1.0002 | ≪ 1 ❌ |

Elements with a roughly correct gradient direction (`cos > 0.9`): **0.115 %**.
Elements correctly scaled (`0.5 < ratio < 2`): **0.001 %**.

**Why this is fatal rather than cosmetic:** tracking integrates the
**gradient**. `det F = exp(∫ div u dt)` and every damage accumulator
(Rice–Tracey, Cockcroft–Latham, Piwnik–Plata) is driven by `A`, not by
`u(x_c)`. A field whose `A` is 10⁶ too large and pointing the wrong way is
unusable for the purpose it was built for, however clean its divergence.

### Actions taken

- **LUMI sweep 23059351 cancelled** after the pilot, rather than producing 17
  cases × 3 snapshots of the same artefact. Allocation preserved.
- **A gradient gate added to `build_divfree_field.py`'s verdict.** It reports
  `|A_new|/|A_P1|`, `cos(A_new, A_P1)` and `|b|/|A x_c|`, and prints an
  explicit ❌ FAIL telling the reader not to use the field for tracking.
  ⚠️ `preservation` must never again be read as sufficient: it is blind to the
  gradient by construction, and that blindness is what let §4r–4u call these
  fields a success.

### ⚠️ What this means for the method

The BDM₁-on-patches formulation is **not salvageable by tuning**. The
objective is fundamentally too weak: matching one vector per element (3
equations) cannot pin down a 12-DOF affine map, so the solve is free to pick
any member of a 9-dimensional family — and with `reg = 1e-8` it picks an
arbitrary huge one. Strengthening the objective to pin `A` down makes it
inconsistent with exact `div u = 0` (the moment test above). **The two
requirements cannot both be met in this discrete space**, which is the same
obstruction §4-conforming found from the other direction.

The honest conclusion: the FOM velocity is a stabilised **P1/P1** field whose
divergence is a genuine ~0.66 % of its gradient, and no local projection can
remove it while preserving the gradient. The remaining options, in order of
how much they concede:

1. **Report `div u` as a diagnostic and stop projecting.** The §4v closed form
   `A' = A − (tr A/3) I` is the *minimum-norm* divergence-free correction per
   element and changes the gradient by a measured p50 0.66 % / p99 28 %. It is
   discontinuous across faces, so it is not an H(div) field — but for a
   pathline integrator that samples `A` per element, continuity of the normal
   trace was never actually required. ⚠️ Test this against the det F gate
   before believing it; it is the cheapest option by far and was never tried.
2. **Scott–Vogelius P3** on a barycentrically-refined mesh — pointwise
   divergence-free *and* H¹-conforming, so the gradient is well-defined. Cost
   was the objection (§4-sv), never correctness.
3. **Accept the FOM as-is** and carry `ln J = ∫ div u dt` as an error bar on
   det F rather than forcing it to zero.

**Recommendation: (1) first**, because it is a closed form, costs nothing, and
its effect on the gradient is already measured. If the det F gate passes with
it, the whole patch-solve apparatus is unnecessary.

## 4y. ✅ SOLVED — the closed-form per-element correction, and it beats BDM₁ on everything

§4x's recommendation (1), tested immediately because it costs nothing.
`scripts/damage/test_closed_form_divfree.py`.

### The correction

For the affine map `u = A x + b` that the P1 data defines exactly on each tet
(§4v), the **minimum-norm** divergence-free correction is closed-form:

$$A' = A - \frac{\operatorname{tr}A}{3}\,I,
\qquad b' = b + (A - A')\,x_c$$

- `tr A' = tr A − 3·(tr A/3) = 0` — divergence-free **by construction**, to
  round-off, with no solve.
- `u'(x_c) = A' x_c + b' = A x_c + b = u(x_c)` — the centroid velocity is
  **exactly** unchanged.
- `‖A′−A‖_F/‖A‖_F = |div u|/(√3‖A‖_F)` is **provably minimal** (§4v) — p50
  0.66 %, p99 28 % on D2. No divergence-free field is closer to the FOM's
  gradient.

### Measured, all four families at ts = 34

| case | elements | divergence | preservation | `|A'|/|A_P1|` | `cos` | aligned | time |
|---|---|---|---|---|---|---|---|
| D2 | 10,807,677 | 2.98e-18 | 3.88e-08 | 0.99998 | 0.99998 | 99.74 % | **13.4 s** |
| A2 | 10,335,240 | 2.56e-18 | 3.82e-08 | 0.99999 | 0.99999 | 99.85 % | 14.1 s |
| B1 | 7,462,783 | 4.45e-18 | 4.34e-08 | 0.99995 | 0.99995 | 97.92 % | 71.2 s |
| C1 | 7,586,466 | 4.06e-18 | 4.41e-08 | 0.99997 | 0.99997 | 99.87 % | 12.1 s |
| D1 | 7,922,599 | 4.72e-18 | 4.70e-08 | 0.99993 | 0.99993 | 98.10 % | 9.3 s |

Correctly scaled (`0.5 < ratio < 2`): **100.000 %** in every case. ✅ Gradient
gate passes everywhere.

### Against the BDM₁ patch solve it replaces

| | BDM₁ (§4r–4w) | closed form | factor |
|---|---|---|---|
| divergence | 9.2e-17 | **3.0e-18** | 30× better |
| preservation | 1.7e-01 | **3.9e-08** | 4.3 million× better |
| `|A_new|/|A_P1|` | 2.8e+06 | **0.99998** | — |
| `cos(A_new, A_P1)` | 0.0016 | **0.99998** | — |
| gradient gate | ❌ 0.115 % aligned | ✅ 99.74 % | — |
| cost / snapshot | 1511 s on 16 cores, 821 patches | **13.4 s, one pass** | 113× |
| machinery | face topology, adjacency, patch growth, overlap ring, sparse LU, process pool, seam check | `np.linalg.solve` on 4×4 blocks | — |

⚠️ It is better on **every** axis simultaneously, including the two that
motivated the whole patch apparatus. The 3-layer overlap, the 10,000-element
patch tuning, the Schwarz driver and the seam analysis (§4-overlap, §4p–4r)
are all **unnecessary**, not merely slower.

### ⚠️ What it gives up, and why that is acceptable here

The corrected field is **discontinuous across element faces**: it is not
H(div)-conforming and not a finite-element function in the usual sense. That
was the stated reason for pursuing BDM₁ in the first place.

It does not matter for this application, for a concrete reason: **the pathline
integrator samples `A` and `b` inside whichever element contains the particle
and never evaluates a normal trace or a flux**. Conformity buys well-posedness
for a *variational problem*; there is no variational problem here, only a
pointwise field evaluation per element. §4x reached the same conclusion from
the other side — the P1/P1 FOM is not inf-sup stable, so there is no weak
structure to preserve.

⚠️ **Open, and the honest caveat:** the field is divergence-free *within* each
element but a particle crossing a face sees a jump in `A`. Whether the
accumulated `ln J = ∫ div u dt` along a pathline is therefore zero to the same
precision is **not** established by the per-element numbers above — it needs
the det F gate on a real tracking run. That is the next test, and it is the
one that decides this. Per-element divergence of 3e-18 is necessary, not
sufficient.

### Next

1. **Run the det F gate** on D2 with this field (`--damage-divergence field`)
   and compare against the analytic baseline. This is the decisive test.
2. If it passes, delete nothing yet but mark §4e–4w as superseded method and
   keep `divfree.py` only for the record.
3. ⚠️ Do **not** re-run the LUMI sweep until (1) passes. Cancelling 23059351
   was right; a sweep of a field that fails the pathline gate is the same
   mistake at 17× the cost.

## 4z. ✅ THE DECISIVE TEST PASSES — `ln J` vanishes along the pathline

§4y left one thing open, and it was the thing that mattered: the corrected
field is divergence-free *within* each element, but a particle crossing a face
sees a **jump in `A`**. Per-element divergence of 4e-18 is necessary, not
sufficient — what tracking integrates is `ln J = ∫ div u dt` **along a
pathline**, sampled at the particle position each RK4 substep
(`benchmark_femuss_comparison.py:1855`).

`scripts/damage/test_pathline_lnj.py` marches the same seeds through the raw
P1 field and the corrected field, accumulating `ln J` with the RK4 weights
exactly as the kernel does. ✅ Pure numpy, single process, no JAX — it is
deliberately **independent of the production integrator it validates**.

### Result: 200 seeds × 300 steps, three families

| case | raw `|ln J|` p50 | raw max `|det F−1|` | **corrected `|ln J|` p50** | corrected p99 | improvement |
|---|---|---|---|---|---|
| D2 | 7.63e-04 | 0.072 | **1.66e-18** | 2.06e-17 | 4.6e+14× |
| C1 | 1.30e-02 | 0.587 | **2.98e-18** | 3.77e-17 | 4.4e+15× |
| B1 | 1.49e-01 | **2.734** | **5.55e-18** | 5.34e-17 | 2.7e+16× |

`|det F − 1|` on the corrected field is **exactly 0.0** in float64 in all
three cases. ✅ PASS at p99 < 1e-10 by eight orders of margin.

### ⚠️ The face jump was fully exercised, not avoided

Each particle crossed a median of **59 (D2) / 72 (C1) / 78 (B1)** element
faces, up to 140. The discontinuity in `A` is therefore not a hypothetical
exposure that the test happened to miss — it was hit dozens of times per
pathline and did not matter. The reason is simple once stated: every substep
samples *some* element, and in **every** element `div u = 0` to round-off. The
jump changes *which* zero-divergence map is sampled, never that it is zero.

### The trajectory barely moves — so the damage map survives

The second failure mode §4y flagged: `det F` could be clean while particles
visit materially different elements, moving the damage map.

| case | trajectory shift p50 | distance travelled p50 | **relative shift** |
|---|---|---|---|
| D2 | 6.39e-09 | 8.95e-04 | **8.19e-06** |
| C1 | 7.19e-08 | 1.08e-03 | **9.59e-05** |
| B1 | 2.84e-07 | 1.15e-03 | **2.90e-04** |

Relative shift ≤ 3e-04. ✅ Both fields kept **200/200 particles alive**, so
neither failure mode occurred.

### ⚠️ How badly the raw field was failing

B1's raw `|det F − 1|` reaches **2.734** — a 273 % volume error on an
incompressible material. The step-1 gate was not marginally failing on these
cases, it was **meaningless**: `det F` was measuring the FOM's stabilisation
error, not material volume change. This retrospectively explains §B.6g (the
gate failing at 3.17e-03 median on the full D2 run while the 50-step smoke
test showed 1.43e-05 — the smoke test simply had not accumulated enough path).

### Verdict

The closed-form correction is **validated end to end**: divergence-free per
element (§4y), gradient preserved to 0.99998 (§4y), and now `ln J` exact along
the pathline with trajectories unchanged. There is no remaining open question
about the *method*. What remains is **plumbing**, and §4aa settles it.

## 4aa. ⚠️ SUPERSEDED BY §4ab — host-side plumbing, correct but steady-only

> ⚠️ **Read §4ab.** The host-side projection below works and its refutation of
> the nodal fold-in stands, but it rides the **steady-only** gradient-recovery
> path, and production is 166 frames. Its claim that per-frame streaming makes
> this practical is **wrong** — no such streaming exists; the velocity stack is
> uploaded whole, and stored gradients would be 64.6 GB. §4ab does the
> projection in-kernel instead, needing no per-frame array at all.
> ~~PLUMBING — it rides the existing centroid_taylor path~~

§4z validated the method. This section is how it reaches the GPU without
violating the standing constraint: **no in-step CPU–GPU transfer, prebuilt
where possible.**

### The key observation

The kernel's `centroid_taylor` reconstruction already evaluates

```
vel = vc + Gc @ (pos - xc)          # benchmark_femuss_comparison.py:1284
```

which is **algebraically identical** to the corrected affine map
`u'(x) = A' x + b'` with `Gc = A'` and `vc = u(x_c)`. And the projection leaves
the centroid velocity *exactly* unchanged (that is what `b' = b + (A−A') x_c`
buys), so **only `element_gradient` moves**:

| array | changed? |
|---|---|
| `element_centroid` (geometry) | no |
| `element_v_centroid` | **no** — exactly invariant under the projection |
| `element_gradient` | yes — trace removed |

✅ So the implementation is: project the prebuilt `(n_elem, 3, 3)` gradient on
the host, once, before `jax.device_put`. **No new GPU array, no new sampler,
no kernel edit, zero in-step cost.** `--damage-divergence analytic` then
differentiates that same closure through `jacfwd`, so `div u` picks up
`tr(A') = 0` automatically — the two can never drift apart.

Implemented as `jaxtrace/damage/divfree_projection.py` (the single source of
truth, as `detachment.py` is for the detachment rule) plus `--divergence-free`
in `run_tracking.py`.

### ⚠️ Refuted: folding the correction into the NODAL velocity

The tempting simplification — correct the nodal velocity once and leave every
downstream path untouched — **cannot work, and was measured rather than
assumed**. A node is shared by ~20 elements with *different* `tr A`, so no
nodal field has a P1 gradient equal to `A'` in every element. A global
least-squares nodal fit on a 1.53 M-element sub-box of D2:

| | `|div u|/‖A‖` p50 |
|---|---|
| raw P1 | 1.03e-02 |
| **nodal least-squares fit** | **1.03e-01** (10× WORSE) |
| per-element closed form | 4.32e-18 |

and it changed the nodal velocity by **100 %** (p50 0.9999) doing it. ❌ The
per-element path is **necessary**, not merely convenient.

### ⚠️ Two guards, because a silent no-op is the expensive failure

`--divergence-free` acts on an array only the `centroid_taylor` path uploads.
On any other reconstruction the flag would do **nothing** while the run
completed normally and still integrated a divergent field. Both checks are at
**parse time**, so they fail in a second rather than after a mesh load:

- `--divergence-free` without `--gradient-recovery 1` → error.
- `--divergence-free` with `vertex_taylor` or any `hct_*` → error naming the
  reason (those reconstruct from nodal data, where a per-element trace-free
  gradient has nothing to attach to).

### ⚠️ OPEN AND BLOCKING FOR PRODUCTION: recovery is steady-only

`run_tracking.py:2575` builds gradient recovery only under `if n_frames == 1`,
printing `SKIPPED ... falling back to raw P1` otherwise. D2's production script
is `VEL_START=34 VEL_END=199` — **166 frames** — so a production run today
would silently get no projection. The det F gate
(`scripts/damage/test_divfree_detf.sh`) therefore forces `--vel-range 34 34`.
**That makes it a steady-field validation: it proves the plumbing and bounds
the gain, it does not license the 166-frame runs.**

Extending it is feasible, and unlike HCT the numbers are comfortable:

| | per frame | all 166 frames | streaming 1 frame |
|---|---|---|---|
| `centroid_taylor` (`v_centroid` + `gradient`) | 0.52 GB | 86.1 GB ❌ > 32.6 GB HBM | **0.65 GB ✅** |
| HCT Bernstein (reference) | 10.4 GB | 1.72 TB ❌ | 10.4 GB |

✅ Streaming one frame at a time — exactly what the velocity stack already does
— is **0.65 GB resident**, including the time-independent geometry. So
time-dependent `centroid_taylor` + projection is practical where
time-dependent HCT is not.

**Next, in order:**
1. The det F gate on the steady arm (running) — does the production kernel
   reproduce §4z's 1.7e-18?
2. Extend recovery to time-dependent fields, streaming per frame alongside the
   velocity stack. ✅ Until then `--divergence-free` is a steady-case flag, and
   a **third guard** closes the one case the parse-time checks cannot see:
   `n_frames` is unknown at parse time, so the multi-frame no-op is caught at
   the `n_frames == 1` branch itself, where the run now **aborts** instead of
   silently integrating a divergent field. Its message names the workaround
   (`--vel-range <t> <t>`) and the memory figure that makes the real fix
   practical.

## 4ab. ✅ IN-KERNEL — no per-frame array, so it works at production frame counts

§4aa's host-side plumbing was correct but **steady-only**, which is useless for
production: D2 runs `VEL_START=34 VEL_END=199`, 166 frames. This section
replaces it.

### ⚠️ What §4aa got wrong

§4aa claimed time-dependent recovery was practical because "streaming one
frame is 0.65 GB resident". **That streaming mechanism does not exist.**
`velocity_sequence_gpu = jax.device_put(velocity_sequence)` uploads the stack
**whole** (`run_tracking.py:2410`) and the kernel indexes
`velocity_fields_gpu[vel_idx]`. Extending recovery the same way means:

| array, D2 at 166 frames | size |
|---|---|
| `velocity_sequence` (already uploaded) | 3.83 GB |
| `element_v_centroid` (n_fr, n_el, 3) | 21.5 GB |
| `element_gradient` (n_fr, n_el, 3, 3) | **64.6 GB** |
| **total** | **86.1 GB** vs a 32.6 GB card ❌ |

Even storing just `tr A` as a scalar per element per frame is 7.18 GB. The
stored-gradient route does not fit at production frame counts, full stop.

### The resolution: the kernel can compute `tr A` itself

The projection needs only two things, and neither has to be stored per frame:

1. **`tr A`** — recoverable in-kernel from data already in registers. For raw
   P1, `A = Σ_a u_a ⊗ ∇N_a`, and the kernel already gathers the 4 nodal
   velocities and holds `M_inv`. ✅ Verified against the exact affine map to
   **8.9e-16**. More generally `jax.jacfwd` of the active interpolant gives
   the full Jacobian — which the `analytic` divergence path already computes.
2. **`x_c`** — the element centroid, **time-independent geometry**:
   (n_el, 3) float32 = **0.13 GB on D2 at any frame count**.

So the corrected field is, in-kernel,

$$u'(x) = u(x) - \frac{\operatorname{tr}A}{3}\,(x - x_c)$$

⚠️ **The centroid anchor is not a detail — it is what makes this a field.**
Anchoring at the particle position instead gives `u'(x_pos) = u(x_pos)` for
every sample point, i.e. no correction at all. (I drafted exactly that first;
it is wrong and was caught before running.)

✅ Verified equal to the host-side closed form of
`jaxtrace/damage/divfree_projection.py` to **5.0e-16** over interior points,
with `tr(jacfwd(corrected)) = -1.1e-16` against `tr(jacfwd(raw)) = 1.608`, and
the centroid value preserved **exactly** (0.0e+00).

### Implementation

`benchmark_femuss_comparison.py` gains `divergence_free: bool = False`. When
set, `interpolate_velocity_single` is **wrapped**, not reimplemented:

```python
def velocity_divfree_single(pos, elem_id, velocity_field,
                            _raw=interpolate_velocity_single):
    u   = _raw(pos, elem_id, velocity_field)
    Jac = jax.jacfwd(_raw, argnums=0)(pos, elem_id, velocity_field)
    tr3 = jnp.trace(Jac) / 3.0
    xc  = element_centroid_gpu[elem_id]
    ok  = (elem_id >= 0) & (elem_id < len(connectivity))
    return jnp.where(ok, u - tr3 * (pos - xc), u)
```

Wrapping means it inherits **any** reconstruction (raw P1, centroid_taylor,
vertex_taylor, HCT) and cannot drift from the field it corrects — the same
argument that makes the `analytic` divergence path correct. All four RK4 stages
and the analytic divergence then see the corrected field with no separate code
path.

⚠️ **`_raw` is a DEFAULT ARGUMENT, not a closed-over name.** Python resolves a
closed-over name at *call* time, so rebinding `interpolate_velocity_single`
below while the body referred to that name makes the function call **itself** —
infinite recursion. Confirmed in isolation (`RecursionError`) before writing
the real thing.

⚠️ **No host-side projection on the recovery path.** Projecting
`element_gradient` there *as well* would apply the correction **twice** on
`centroid_taylor`. §4aa's host-side block is removed, and
`divfree_projection.py` is kept as the reference implementation and the unit
test's oracle.

### Cost

3 extra forward-mode tangents per velocity call. ✅ That is **compute, not an
in-step host transfer**, so the vmap is untouched and the standing constraint
holds.

⚠️ **Do not understate this.** The `analytic` divergence path builds a Jacobian
on the **k1 stage only** (`benchmark_femuss_comparison.py:1916`), whereas the
wrapper puts one on **all four RK4 stages**. So the marginal cost is three
additional Jacobians per step per particle, not "near zero" — an earlier draft
of this section claimed the latter and was wrong. XLA should fuse the Jacobian
with the velocity evaluation it differentiates (same gathers, same registers),
so the real figure is expected well below 4×, but **it is measured, not
assumed**: the two gate arms differ only in this flag, so their `Stage 7`
times give the throughput cost directly.

### The one guard that matters now

`--divergence-free` requires `--damage-divergence analytic` (**the default**).
`field` integrates a *precomputed nodal* `div_u` from the raw P1 gradient and
cannot respond to the projection, so `ln J` would silently ignore it while the
velocity was corrected — a wrong `det F` with no visible symptom. Checked at
parse time.

⚠️ §4aa's other two guards are **removed as obsolete**: the in-kernel path
needs neither `--gradient-recovery 1` nor `centroid_taylor`, and keeping them
would block the production case this section exists to enable. Centroids are
uploaded independently when recovery does not supply them.

## 4ac. ✅ det F GATE PASSES IN THE PRODUCTION KERNEL

Job 23061561 on LUMI, D2, two arms differing **only** in `--divergence-free`.
Both: 4,000 particles, 200 steps, `--vel-range 34 34`, raw P1 (no gradient
recovery), `--damage-divergence analytic`, **3,200/4,000 alive (80.0 %) in
each** — so they are strictly comparable.

| | `ln_J` median | `ln_J` p01 / p99 | `|det F − 1|` median | p99 | max | Stage 7 |
|---|---|---|---|---|---|---|
| raw P1 | +3.3e-05 | −1.6e-03 / +1.7e-03 | **4.39e-04** | 1.91e-03 | 5.84e-03 | 268.5 s |
| **divergence-free** | 0.00000 | 0.0 / 0.0 | **0.000e+00** | 0.000e+00 | 0.000e+00 | 255.4 s |

✅ The in-kernel projection **compiles and runs under `vmap`** (exit 0), and
the centroid upload behaved as designed: `(10807677, 3) float32 (0.13 GB)`,
logged as "time-independent geometry, so this works at any frame count".

✅ The production kernel reproduces the numpy pathline prediction of §4z.

### ⚠️ The zero is a dtype FLOOR, not exactness

`ln_J` accumulates in the run's working precision, and in **float32**
`exp(x)` rounds to exactly 1.0 for `|x| < ~6e-8`. So the `df` arm's
`0.000e+00` means **`|ln_J| < 6e-8`** — a real pass, but a floor.

What makes it meaningful is that the raw arm's **4.39e-04 is four orders
ABOVE** that floor, so the metric demonstrably resolves the quantity it is
being asked about; the projection pushes it below detection. The float64
pathline measurement (1.7e-18, §4z) sits ten orders below the floor and is the
actual value.

`run_tracking.py` now prints this floor and its dtype-derived value whenever
`max|det F − 1|` is identically zero (6.0e-08 for float32, 1.1e-16 for
float64) and points at `--precision float64`. ⚠️ The gate is the number this
whole method is judged on; it must not report a precision the storage cannot
support.

### ⚠️ Throughput: do NOT read a speedup from this

Stage 7 was **255.4 s (df) vs 268.5 s (raw)** — the projection measured **5 %
FASTER**, against my prediction of a cost from three extra Jacobians per step.

**That is not evidence of a speedup.** At 4,000 particles and 0.74–0.78
step/s, the run is kernel-launch-latency dominated, not compute-bound: 200
steps in ~260 s is ~1.3 s per step for 4,000 particles, which is nowhere near
arithmetic throughput. The 5 % is within run-to-run noise on a shared node.
What this *does* establish is that the projection is **not catastrophically
slower** — no 4× blowup from the extra Jacobians, consistent with XLA fusing
each one with the velocity evaluation it differentiates. ⚠️ The real
throughput number needs the full-length run (8,000 steps, 288k particles),
where the kernel is compute-bound.

### ⚠️ Two caveats on this run, both pre-existing

- **800 of 4,000 particles (20 %) not found** — outside mesh or octree
  coverage. Identical in both arms, and a consequence of the coarse
  `20 20 10` smoke-test seed grid over the fractional box, not of the
  projection.
- **100 % recorded `t_res = 0`** (`edot` never exceeded 1 s⁻¹). The 200-step
  march never reaches the shear layer, which is exactly §B.6g's warning that a
  short run does not exercise the physics. ⚠️ So this gate validates the
  **plumbing and the divergence**, not the damage indicators.

### Next: the full-length run is the number that decides

A 200-step march **understates** `ln_J` because it is a path integral: §B.6g
measured the gate at 1.43e-05 on a 50-step smoke test and **3.17e-03** on the
full D2 run, because a short march never leaves the undeformed far field. The
raw arm here reads 4.39e-04 at 200 steps, between those two, consistent with
that scaling.

So the full-length arms (8,000 steps, 166 velocity frames, production seed
density) remain **the** test, and the staged job is held until this result is
recorded. ⚠️ It should also run `--precision float64` for at least the `df`
arm, or the gate will again report a floor instead of a value.

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
