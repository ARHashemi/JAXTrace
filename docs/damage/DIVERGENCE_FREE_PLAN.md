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
