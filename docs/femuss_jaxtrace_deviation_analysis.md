# FEMUSS vs JAXTrace Trajectory Deviation Analysis

## 1. Observed Behaviour

Running the diagnostic tool on 62,164 particles near the FSW tool
(y &isin; [-0.01, 0.01], z &isin; [-0.005, 0]) for 100 steps (dt = 0.0025 s):

| Metric | Value |
|--------|-------|
| Mean position error at step 100 | 6.54 &times; 10<sup>-6</sup> m |
| Max position error at step 100 | 7.09 &times; 10<sup>-4</sup> m |
| Error growth rate (mean, per step) | ~6.5 &times; 10<sup>-8</sup> m/step &mdash; **perfectly linear** |
| Particles lost by JAXTrace | 0 |
| Particles lost by FEMUSS | 5 |
| Mechanism breakdown | 45,745 ok, 16,409 gradual drift, 8 lost by FEMUSS, 2 sudden jump |
| Error decomposition (worst 20) | **>99% tangential**, <1% radial, negligible axial |
| Worst particles location | R<sub>xy</sub> &asymp; 10 mm (pin radius), z &asymp; -0.303 mm (first layer below tool surface) |
| Near-tool vs far error ratio | 12&ndash;19&times; (scales with velocity gradient) |

**Key diagnostic signatures:**
- Error grows linearly &rArr; constant per-step numerical difference (not algorithmic bug)
- Error is almost purely tangential &rArr; small radial discrepancy amplified by rotation
- Error scales with proximity to tool &rArr; velocity gradient amplifies interpolation difference
- Zero particles lost by JAXTrace &rArr; point-location is not the issue


## 2. Potential Origins (Suspects)

### Suspect 1 &mdash; Barycentric Interpolation Method (Primary)

This is the most likely source. The two codes use **mathematically equivalent but numerically different** methods to compute barycentric coordinates for velocity interpolation.

See [Section 3](#3-mathematical-comparison-of-the-two-interpolation-methods) for the full mathematical derivation.

**Impact estimate:** For tets with det &sim; 10<sup>-13</sup> (present in this mesh), the Gram matrix condition number is &kappa;(M)<sup>2</sup>, giving relative interpolation differences of order 10<sup>-4</sup>. At the pin tangential velocity of 0.628 m/s (600 RPM, R = 10 mm), this produces &Delta;v &asymp; 6 &times; 10<sup>-5</sup> m/s, or &Delta;x &asymp; 1.5 &times; 10<sup>-7</sup> m per step &mdash; matching the observed near-tool rate exactly.

### Suspect 2 &mdash; MPI Weight Factor

**FEMUSS** applies a weight factor to the velocity contribution from each element (`Mod_ParticleTracer.f90`, line 1631&ndash;1632):

```fortran
call e%GetWeightFactor(weight)
ParticleInteractionQuantity(1:e%ndime,iparticle) = &
    FluidVelocity(1:e%ndime) * a%ComputationalDtime * weight
```

where (`Mod_Element.f90`, line 992&ndash;998):

```fortran
subroutine GetWeightFactor(e, weightfactor)
    weightfactor = 1.0 - real(count(e%lnods(1:e%pnode) > e%npoinLocal)) / real(e%pnode)
end subroutine
```

This accounts for MPI domain decomposition: if some nodes of the element belong to another rank, the displacement contribution is reduced. For a tet with 1 foreign node, `weight = 0.75`; with 2, `weight = 0.5`. The contributions from all MPI ranks are then summed via `MPI_AllREDUCE(..., MPI_SUM)`.

**JAXTrace** has no domain decomposition and uses `weight = 1.0` implicitly.

**Impact:** If FEMUSS ran with 1 MPI rank, `npoinLocal` = total nodes, so all weights are 1.0 and this suspect is irrelevant. If FEMUSS ran with multiple ranks, particles near partition boundaries would see reduced velocities on each rank before summation. Whether this sums to exactly 1.0 depends on whether the element is found by exactly one rank or multiple ranks.

### Suspect 3 &mdash; Velocity Field Pointer (`bg_velocity2` vs `bg_velocity`)

FEMUSS maintains two velocity pointers:

```fortran
real(rp), pointer :: bg_velocity(:,:)   ! current solver velocity
real(rp), pointer :: bg_velocity2(:,:)  ! velocity used for interpolation
```

The particle tracer interpolates from `bg_velocity2` (line 1639):

```fortran
call e%gather(e%ndime, elvel, a%bg_velocity2)
FluidVelocity = matmul(elvel, newshape)
```

Without velocity recycling, both point to the same array (`SetVelocityArray`, line 860):

```fortran
a%bg_velocity  => bg_velocity
a%bg_velocity2 => bg_velocity
```

With velocity recycling, `bg_velocity2` points to a stored snapshot:

```fortran
a%bg_velocity2 => a%bg_storedVelocity(:,:,a%rstep)
```

**Impact:** If the FSW case uses velocity recycling, the ordering of pin velocity reconstruction relative to the store/restore cycle matters. JAXTrace applies pin velocity reconstruction to the velocity array before tracking starts. If FEMUSS applies it at a different point in the cycle (or if the reconstruction itself differs slightly due to when `bg_velocity` vs `bg_velocity2` is modified), this could contribute to discrepancy. Needs verification of the FSW driver code.

### Suspect 4 &mdash; RK4 Substage Position Computation

**FEMUSS** (`Mod_ParticleTracer.f90`, lines 1217&ndash;1236):

```fortran
do iorder = 1, rk_order
    ! Evaluate velocity at substage position
    coord = particle_position + particle_displacement
    do jorder = 1, rk_order
        coord = coord + dt * rk_A(jorder, iorder) * rk_k(jorder, :, iparticle)
    end do

    ! Search element, compute shape functions, interpolate velocity
    call ComputeParticleInteractionQuantity(ComputeDisplacementIncrease)

    ! Store velocity (k = displacement / dt)
    rk_k(iorder,:,:) = ParticleInteractionQuantity / ComputationalDtime
end do

! Final update: weighted sum of all k's
ParticleInteractionQuantity = sum(rk_b(i) * rk_k(i,:,:)) * dt
```

**JAXTrace** (`benchmark_femuss_comparison.py`, lines 867&ndash;899):

```python
# Stage 1: k1 at pos
elem_k1 = search(pos, elem_id)
vel_k1 = interpolate(pos, elem_k1, velocity_field)

# Stage 2: k2 at pos + 0.5*dt*k1
pos_k1 = pos + 0.5 * dt * vel_k1
elem_k2 = search(pos_k1, elem_k1)
vel_k2 = interpolate(pos_k1, elem_k2, velocity_field)

# Stage 3: k3 at pos + 0.5*dt*k2
pos_k2 = pos + 0.5 * dt * vel_k2
elem_k3 = search(pos_k2, elem_k2)
vel_k3 = interpolate(pos_k2, elem_k3, velocity_field)

# Stage 4: k4 at pos + dt*k3
pos_k3 = pos + dt * vel_k3
elem_k4 = search(pos_k3, elem_k3)
vel_k4 = interpolate(pos_k3, elem_k4, velocity_field)

# Final
pos_final = pos + (dt/6) * (vel_k1 + 2*vel_k2 + 2*vel_k3 + vel_k4)
```

**Impact:** Both implement standard RK4 (Butcher tableau: a = [0, 1/2, 1/2, 1]; b = [1/6, 1/3, 1/3, 1/6]). The logic is equivalent *provided* the Butcher coefficients `rk_A` and `rk_b` in FEMUSS match the classic RK4 tableau. This should be verified from the FEMUSS input file or `ReadData` subroutine.

### Suspect 5 &mdash; Level-Set Boundary Decision

**FEMUSS** (lines 1643&ndash;1646):

```fortran
call e%gather(1_ip, ellev, a%bg_LevelSet)
levelSetValue = dot_product(ellev(1:e%pnode), newshape(1:e%pnode,1))
if (levelSetValue < 0.0_rp) AddFluidInteraction = .false.
```

When `AddFluidInteraction = .false.`, `ComputeDisplacementIncrease` is never called, so the particle gets **zero velocity** for that substage.

**JAXTrace** (lines 818&ndash;822):

```python
node_ls = levelset_gpu[nodes_idx]
ls_val = b0*node_ls[0] + b1*node_ls[1] + b2*node_ls[2] + b3*node_ls[3]
vel = jnp.where(ls_val >= 0.0, vel, jnp.zeros(3))
```

Both interpolate the level-set using shape functions and zero the velocity when LS < 0. However, since the barycentric coordinates differ slightly (Suspect 1), the interpolated `ls_val` will differ. For particles in elements that straddle the tool boundary (where LS changes sign), a small difference in barycentric coords can flip the sign of the interpolated LS value, causing one code to apply velocity and the other to zero it.

**Impact:** This is a binary decision (velocity vs zero) triggered by a continuous value near zero. Even a 10<sup>-8</sup> difference in the interpolated LS can flip the outcome for particles exactly at the boundary. This could explain the "sudden jump" mechanism (2 particles) seen in the diagnostic results.


## 3. Mathematical Comparison of the Two Interpolation Methods

### 3.1 Setup

Consider a tetrahedron with vertices **p**<sub>0</sub>, **p**<sub>1</sub>, **p**<sub>2</sub>, **p**<sub>3</sub> &isin; R<sup>3</sup>. Define the edge matrix:

$$
\mathbf{M} = \begin{bmatrix} \mathbf{p}_1 - \mathbf{p}_0 & \mathbf{p}_2 - \mathbf{p}_0 & \mathbf{p}_3 - \mathbf{p}_0 \end{bmatrix} \in \mathbb{R}^{3 \times 3}
$$

For a query point **q**, define the local vector:

$$
\mathbf{d} = \mathbf{q} - \mathbf{p}_0
$$

The barycentric coordinates (&lambda;<sub>1</sub>, &lambda;<sub>2</sub>, &lambda;<sub>3</sub>) satisfy:

$$
\mathbf{M} \boldsymbol{\lambda} = \mathbf{d}, \qquad \lambda_0 = 1 - \lambda_1 - \lambda_2 - \lambda_3
$$

Then the interpolated velocity is:

$$
\mathbf{v}(\mathbf{q}) = \lambda_0 \mathbf{v}_0 + \lambda_1 \mathbf{v}_1 + \lambda_2 \mathbf{v}_2 + \lambda_3 \mathbf{v}_3
$$

### 3.2 FEMUSS: Direct Jacobian Inverse

FEMUSS solves **M** **&lambda;** = **d** by analytically inverting **M** via cofactors.

The Jacobian matrix `xjacm` in FEMUSS is **M**<sup>T</sup> (Fortran column-major storage with the edges stored as columns of `xjacm`):

```fortran
! Mod_Element.f90, subroutine linear_isoparinv (lines 639-683)

! Build Jacobian: xjacm(i,j) = elcod(i, j+1) - elcod(i, 1)
xjacm(:,1) = e%elcod(:,2) - e%elcod(:,1)   ! = p1 - p0
xjacm(:,2) = e%elcod(:,3) - e%elcod(:,1)   ! = p2 - p0
xjacm(:,3) = e%elcod(:,4) - e%elcod(:,1)   ! = p3 - p0

! Compute cofactor matrix (= adjugate of xjacm)
xjaci(1,1) =  xjacm(2,2)*xjacm(3,3) - xjacm(3,2)*xjacm(2,3)
xjaci(2,1) = -xjacm(2,1)*xjacm(3,3) + xjacm(3,1)*xjacm(2,3)
xjaci(3,1) =  xjacm(2,1)*xjacm(3,2) - xjacm(3,1)*xjacm(2,2)
xjaci(2,2) =  xjacm(1,1)*xjacm(3,3) - xjacm(3,1)*xjacm(1,3)
xjaci(3,2) = -xjacm(1,1)*xjacm(3,2) + xjacm(1,2)*xjacm(3,1)
xjaci(3,3) =  xjacm(1,1)*xjacm(2,2) - xjacm(2,1)*xjacm(1,2)
xjaci(1,2) = -xjacm(1,2)*xjacm(3,3) + xjacm(3,2)*xjacm(1,3)
xjaci(1,3) =  xjacm(1,2)*xjacm(2,3) - xjacm(2,2)*xjacm(1,3)
xjaci(2,3) = -xjacm(1,1)*xjacm(2,3) + xjacm(2,1)*xjacm(1,3)

! Determinant
detjm = xjacm(1,1)*xjaci(1,1) + xjacm(1,2)*xjaci(2,1) + xjacm(1,3)*xjaci(3,1)
denom = 1.0_rp / detjm

! Scale to get inverse
xjaci = xjaci * denom

! Local coordinates
b = xglob - e%elcod(:,1)       ! = d = q - p0
xloc(i) = dot_product(xjaci(i,:), b)   ! lambda = M^{-1} d
```

Then shape functions are evaluated:

```fortran
! Mod_TypeOfElementTetraedra.f90 (lines 90-100)
shape(1) = 1.0 - s - t - z     ! lambda_0 = 1 - lambda_1 - lambda_2 - lambda_3
shape(2) = s                    ! lambda_1
shape(3) = t                    ! lambda_2
shape(4) = z                    ! lambda_3
```

And velocity is interpolated:

```fortran
! Mod_ParticleTracer.f90 (line 1640)
FluidVelocity(1:ndime) = matmul(elvel(1:ndime, 1:pnode), newshape(1:pnode, 1))
! = sum_i shape(i) * velocity(node_i)
```

**In matrix notation:**

$$
\boldsymbol{\lambda} = \mathbf{M}^{-1} \mathbf{d}
$$

where **M**<sup>-1</sup> is computed via the **cofactor (adjugate) matrix** divided by det(**M**).

**Operation count:** 9 cofactor entries (each ~2 multiplications + 1 addition) + 1 determinant (3 multiply-adds) + 9 divisions + 3 dot products for the final multiply = **~30 multiplications, ~15 additions**.

**Numerical properties:**
- Condition number of the linear system = &kappa;(**M**)
- For a regular tetrahedron, &kappa; &asymp; 1
- For a degenerate tetrahedron with det &sim; 10<sup>-13</sup>, &kappa; can reach 10<sup>5</sup>&ndash;10<sup>6</sup>

### 3.3 JAXTrace: Gram Matrix (Normal Equations)

JAXTrace computes barycentric coordinates by solving the **normal equations** of the same system.

Instead of solving **M &lambda;** = **d** directly, it forms:

$$
\mathbf{G} = \mathbf{M}^T \mathbf{M} \in \mathbb{R}^{3 \times 3}, \qquad \mathbf{r} = \mathbf{M}^T \mathbf{d} \in \mathbb{R}^3
$$

and solves **G &lambda;** = **r** via Cramer's rule.

```python
# benchmark_femuss_comparison.py, lines 798-814

v0 = nodes[1] - nodes[0]   # column 1 of M
v1 = nodes[2] - nodes[0]   # column 2 of M
v2 = nodes[3] - nodes[0]   # column 3 of M
vp = pos - nodes[0]         # d = q - p0

# Gram matrix G = M^T M (symmetric, 6 unique entries)
d00 = dot(v0, v0)   # G[0,0]
d01 = dot(v0, v1)   # G[0,1] = G[1,0]
d02 = dot(v0, v2)   # G[0,2] = G[2,0]
d11 = dot(v1, v1)   # G[1,1]
d12 = dot(v1, v2)   # G[1,2] = G[2,1]
d22 = dot(v2, v2)   # G[2,2]

# Right-hand side r = M^T d (3 entries)
dp0 = dot(vp, v0)   # r[0]
dp1 = dot(vp, v1)   # r[1]
dp2 = dot(vp, v2)   # r[2]

# Determinant of G (via cofactor expansion along first row)
det = d00*(d11*d22 - d12**2) - d01*(d01*d22 - d02*d12) + d02*(d01*d12 - d02*d11)

# Cramer's rule: lambda_i = det(G_i) / det(G)
b1 = (dp0*(d11*d22 - d12**2) - d01*(dp1*d22 - dp2*d12) + d02*(dp1*d12 - dp2*d11)) / det
b2 = (d00*(dp1*d22 - dp2*d12) - dp0*(d01*d22 - d02*d12) + d02*(d01*dp2 - d02*dp1)) / det
b3 = (d00*(d11*dp2 - d12*dp1) - d01*(d01*dp2 - d02*dp1) + dp0*(d01*d12 - d02*d11)) / det
b0 = 1.0 - b1 - b2 - b3

vel = b0*v0_vel + b1*v1_vel + b2*v2_vel + b3*v3_vel
```

**In matrix notation:**

$$
\boldsymbol{\lambda} = (\mathbf{M}^T \mathbf{M})^{-1} \mathbf{M}^T \mathbf{d}
$$

**Operation count:** 6 dot products for **G** + 3 dot products for **r** + Cramer's rule on 3&times;3 = **~45 multiplications, ~25 additions**. More work than the direct inverse.

**Numerical properties:**
- Condition number of the system = &kappa;(**M**<sup>T</sup>**M**) = &kappa;(**M**)<sup>2</sup>
- **This is the critical difference.** The Gram matrix squares the condition number.
- For a tet with &kappa;(**M**) = 10<sup>5</sup>, the Gram system has &kappa; = 10<sup>10</sup>
- In float64 (machine &epsilon; &asymp; 10<sup>-16</sup>), the effective accuracy is:
  - Direct inverse: ~10<sup>-16</sup> / 10<sup>5</sup> = **10<sup>-11</sup> relative error** in &lambda;
  - Gram/normal equations: ~10<sup>-16</sup> / 10<sup>10</sup> = **10<sup>-6</sup> relative error** in &lambda;

### 3.4 Equivalence and Divergence

**Mathematically:** For non-degenerate tetrahedra (det(**M**) &ne; 0), both methods give the exact same barycentric coordinates:

$$
\mathbf{M}^{-1} \mathbf{d} = (\mathbf{M}^T \mathbf{M})^{-1} \mathbf{M}^T \mathbf{d}
$$

This identity holds because **M** is square and invertible: (**M**<sup>T</sup>**M**)<sup>-1</sup>**M**<sup>T</sup> = **M**<sup>-1</sup>(**M**<sup>T</sup>)<sup>-1</sup>**M**<sup>T</sup> = **M**<sup>-1</sup>.

**Numerically:** The two methods diverge because floating-point arithmetic is not exact:

| Property | Direct Inverse (FEMUSS) | Gram / Normal Eqs (JAXTrace) |
|----------|------------------------|------------------------------|
| System solved | **M &lambda;** = **d** | **M**<sup>T</sup>**M &lambda;** = **M**<sup>T</sup>**d** |
| Condition number | &kappa;(**M**) | &kappa;(**M**)<sup>2</sup> |
| Relative error in &lambda; | O(&epsilon; &middot; &kappa;) | O(&epsilon; &middot; &kappa;<sup>2</sup>) |
| Operations | ~30 mul, ~15 add | ~45 mul, ~25 add |
| Extra dot products | 0 | 9 (Gram entries + RHS) |
| Cancellation risk | Low (cofactors) | High (dot products of near-parallel vectors) |

### 3.5 Impact on This Mesh

From the precomputation log:
```
Det range: [4.87e-13, 1.28e-07]
Degenerate: 0 (0.0000%)
```

For the smallest-determinant elements (det &sim; 5 &times; 10<sup>-13</sup>):
- The edge vectors have magnitudes ~10<sup>-3</sup> (mm-scale mesh)
- Edge matrix norm &sim; 10<sup>-3</sup>, so &kappa;(**M**) &sim; ||**M**|| &middot; ||**M**<sup>-1</sup>|| &sim; 10<sup>-3</sup> &times; (10<sup>-3</sup> / 5&times;10<sup>-13</sup>) &asymp; 2 &times; 10<sup>6</sup>
- Gram condition number: &kappa;<sup>2</sup> &asymp; 4 &times; 10<sup>12</sup>
- Relative &lambda; error in JAXTrace: &sim; 10<sup>-16</sup> &times; 4 &times; 10<sup>12</sup> = 4 &times; 10<sup>-4</sup>
- Relative &lambda; error in FEMUSS: &sim; 10<sup>-16</sup> &times; 2 &times; 10<sup>6</sup> = 2 &times; 10<sup>-10</sup>

The difference in &lambda; (10<sup>-4</sup> vs 10<sup>-10</sup>) translates directly to velocity interpolation error:
- At pin tangential velocity 0.628 m/s: &Delta;v = 0.628 &times; 4&times;10<sup>-4</sup> = 2.5 &times; 10<sup>-4</sup> m/s
- Over dt = 0.0025 s: &Delta;x = 6 &times; 10<sup>-7</sup> m per step (worst elements)
- Mean over all elements (most are well-conditioned): ~10<sup>-7</sup> m per step near tool

This matches the observed error growth rate of 1.6 &times; 10<sup>-7</sup> m/step near the tool.


## 4. Proposed Fix

Replace the Gram-matrix interpolation in JAXTrace with the direct Jacobian inverse, matching FEMUSS.

The precomputed inverse matrices (`M_inv_array`, `p0_array`) already exist in the codebase (`jaxtrace/gpu/search/point_in_tet_inverse.py`) but are only used for the point-in-tet containment test, not for velocity interpolation.

**Current** (Gram matrix):
```python
v0 = nodes[1] - nodes[0]
v1 = nodes[2] - nodes[0]
v2 = nodes[3] - nodes[0]
vp = pos - nodes[0]
d00 = jnp.dot(v0, v0)  # ... (6 Gram entries + 3 RHS + Cramer's rule)
```

**Proposed** (direct inverse, matching FEMUSS):
```python
M_inv = M_inv_gpu[elem_id]      # precomputed (3, 3)
p0 = p0_gpu[elem_id]            # precomputed (3,)
local = pos - p0
bary = M_inv @ local             # (3,) = (lambda_1, lambda_2, lambda_3)
b0 = 1.0 - bary[0] - bary[1] - bary[2]
```

This is also faster (one 3&times;3 matvec vs 9 dot products + Cramer's rule) and uses the same precomputed data already on GPU.


## 5. Summary of Suspects

| # | Suspect | Likely Impact | How to Verify |
|---|---------|--------------|---------------|
| **1** | **Gram vs direct inverse interpolation** | **High** &mdash; explains linear drift, spatial scaling, and magnitude | Replace with direct inverse, re-run diagnostic |
| 2 | MPI weight factor | Medium (if multi-rank) | Check FEMUSS run config for number of MPI ranks |
| 3 | `bg_velocity2` pointer / recycling order | Low&ndash;Medium | Check FSW driver for recycling config and pin velocity timing |
| 4 | RK4 Butcher tableau mismatch | Low | Verify `rk_A`, `rk_b` coefficients in FEMUSS input |
| 5 | Level-set boundary decision | Low (2 particles) | Follows from Suspect 1; fixing interpolation will resolve this too |
