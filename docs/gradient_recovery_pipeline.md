# Gradient Recovery and Velocity Reconstruction Pipeline for Particle Tracking

## Overview

This document details the complete post-processing pipeline to improve particle tracking accuracy
on a linear (P1) FEM velocity field, using Superconvergent Patch Recovery (SPR) / Polynomial-Preserving
Recovery (PPR) followed by a Hermite-type velocity reconstruction, feeding into RK4 integration.

The pipeline operates entirely as a post-process on the converged (periodic steady-state) FEM solution,
not inside the nonlinear solver or time-marching loop.

## Implementation status

Steps 1–4 (raw element gradient, patch build, SPR fit, C0 reassembly) and a
first-order Step 5 (Taylor form: `v(p) = v_c + G_c @ (p - x_c)` per element)
are implemented in `jaxtrace/gpu/recovery/gradient_recovery.py`.

`run_tracking.py` exposes two flags:

* `--gradient-recovery {0,1}` — default `1`. When `1`, the mesh path runs
  the full Step 1–5 pipeline once on the loaded velocity field
  (whether from FOM PVTU, `--velocity-source rom`, or an analytic field
  projected onto a mesh) before the RK4 loop, and the JIT'd kernel
  samples the Taylor reconstruction instead of raw P1 barycentric interp.
* `--recovery-method {taylor}` — selects the Step 5 form. Currently
  `taylor` is the only supported method; a future patch adds
  `hct_cubic` (a Hsieh-Clough-Tocher / macro-element cubic Hermite
  reconstruction) that will implement the full accuracy target
  described below in Step 5.

The Taylor form is exact for linear velocity fields (recovers the raw
nodal velocities everywhere), first-order in the recovered gradient
elsewhere. On highly non-linear regions of the field it may not
outperform raw P1 sampled at nodes; the full Hermite reconstruction
described in Step 5 below will.

---

## Step 0: Prerequisites and Inputs

Required data per node, from the FEM output:

- Nodal velocity vector `u_i = (u_x, u_y, u_z)` at every mesh node `i`
- Mesh connectivity (element-to-node table)
- Nodal coordinates `x_i`
- (Optional, partial) Element/Gauss-point deviatoric strain rate tensor (symmetric part of gradient only)

Not required but useful for validation:

- Stored strain rate field, to cross-check the symmetric part of your regenerated raw gradient

---

## Step 1: Compute the Raw Element-Wise Velocity Gradient

For each linear (P1) simplex element (triangle in 2D, tetrahedron in 3D), the velocity gradient is
constant within the element, since it is the derivative of a linear shape function times nodal values.

1.1. For each element `e`, assemble the shape function gradient matrix `B_e` (standard FEM operation,
     identical to what is used to build the element stiffness matrix).

1.2. Compute the full (not just symmetric) velocity gradient tensor per element:

    grad_u_e = sum_over_nodes_a ( u_a (x) grad(N_a) )

    where `grad(N_a)` is the constant gradient of shape function `N_a` in element `e`, and `(x)` denotes
    the outer product (each velocity component times each spatial derivative).

1.3. Store the full 3x3 (or 2x2) gradient tensor per element — this includes both:
     - the symmetric part (equivalent to the strain rate you may already have stored)
     - the antisymmetric part (vorticity/spin tensor), which is NOT in your existing strain output and
       must be generated fresh from nodal velocities.

1.4. (Validation) Compare the symmetric part of `grad_u_e` against your solver's stored strain rate
     at the element center. They should match closely; large discrepancies indicate a bug in gradient
     reconstruction or inconsistent quadrature point locations.

Output of Step 1: one full gradient tensor per element (piecewise-constant field).

---

## Step 2: Build Nodal Patches

2.1. For every mesh node `i`, identify all elements that share that node (its "patch" Omega_i).

2.2. Build patch-to-element and node-to-patch maps once, as a preprocessing pass over the mesh
     connectivity (this only needs to be built once per mesh, not once per SPR call).

2.3. Record, for each element in the patch, the location of its sample point:
     - For P1 elements, since the gradient is constant, any representative point works
       (commonly the centroid); no true "Gauss point" distinction exists here as it does for
       higher-order elements.

Output of Step 2: for each node, a list of (element gradient value, sample location) pairs.

---

## Step 3: Fit a Smooth Polynomial per Node (SPR Core Step)

For each node `i`, and for each gradient tensor component independently (i.e. repeat this fitting
9 times in 3D — once per tensor entry, or 6 times if you only recover the symmetric part):

3.1. Assume a local polynomial model of the same degree as your shape functions. For P1 elements,
     this is a linear polynomial in space:

    sigma*(x, y, z) = a0 + a1*x + a2*y + a3*z

3.2. Collect the sampled gradient-component values `sigma_h^(k)` from all `m` elements in the patch,
     at their sample locations `(x_k, y_k, z_k)`.

3.3. Solve the least-squares problem:

    minimize over (a0, a1, a2, a3):  sum_{k=1}^{m} [ sigma*(x_k, y_k, z_k) - sigma_h^(k) ]^2

    This is a standard small linear least-squares system (design matrix has `m` rows, 4 columns in 3D).
    Solve via normal equations or QR decomposition — `m` is typically small (5-20 elements per patch).

3.4. Evaluate the fitted polynomial AT THE NODE itself:

    sigma*(x_i, y_i, z_i) = a0 + a1*x_i + a2*y_i + a3*z_i

3.5. Store this as the recovered nodal value for that gradient component.

3.6. Repeat for every node and every gradient tensor component.

Output of Step 3: a full recovered gradient tensor at every node (smooth, continuous field once
reassembled with shape functions).

### PPR Variant (Optional Refinement)

If using PPR instead of plain SPR: modify the least-squares fit (Step 3.3) or the sample point
selection so that the fit exactly reproduces the underlying polynomial when the true solution is
already a polynomial of the target degree. This typically involves a different weighting or
constrained least-squares formulation; consult the PPR reference for the exact constrained system.
This variant is recommended if your mesh has strongly irregular/distorted elements, where plain SPR's
superconvergence assumptions weaken.

---

## Step 4: Reassemble the Recovered Gradient as a Continuous Field

4.1. With recovered nodal gradient values available at every node, treat them exactly like any other
     nodal FEM field.

4.2. Interpolate between nodes using the SAME shape functions as your original FEM solution:

    grad_u*(x) = sum_a ( N_a(x) * grad_u*_a )

Output of Step 4: a continuous (C0), smoother, more accurate gradient field over the whole domain,
one full order of accuracy better than the raw piecewise-constant gradient.

---

## Step 5: Build a Hermite-Type Velocity Reconstruction per Element

This step uses BOTH the original nodal velocities AND the newly recovered nodal gradients to build
a higher-order (cubic) reconstruction of the velocity field itself — this is what actually gets
sampled during RK4, not the gradient.

5.1. For each element, and each velocity component independently:

5.2. In 1D (edge-local direction), the cubic Hermite interpolant between two nodes i and i+1,
     separated by distance h, with local coordinate xi in [0,1], is:

    u(xi) = h00(xi)*u_i + h10(xi)*h*u'_i + h01(xi)*u_(i+1) + h11(xi)*h*u'_(i+1)

    where the Hermite basis functions are:

    h00(xi) = 2*xi^3 - 3*xi^2 + 1
    h10(xi) = xi^3 - 2*xi^2 + xi
    h01(xi) = -2*xi^3 + 3*xi^2
    h11(xi) = xi^3 - xi^2

    and u'_i, u'_(i+1) are the directional derivatives (projections of the recovered gradient
    vector onto the edge direction) at each node.

5.3. Generalize to 2D/3D simplex elements: use a corresponding multivariate Hermite/cubic
     reconstruction (e.g. a cubic polynomial per element, constrained to match nodal velocity
     values and nodal recovered gradients at each vertex). This requires more coefficients than
     the 1D case (a full cubic in 2D has 10 coefficients per component; constraints from
     3 vertices x 3 conditions each, value + 2 gradient components, gives 9 constraints,
     requiring one additional consistency condition or a reduced cubic basis, e.g. a
     Hsieh-Clough-Tocher or Powell-Sabin macro-element approach for triangles, or an
     equivalent tetrahedral macro-element in 3D).

5.4. Store the reconstruction coefficients per element (precomputed once, reused for every
     particle and every RK4 stage that queries a point inside that element).

Output of Step 5: a smooth, gradient-consistent, higher-order velocity reconstruction, valid for
sampling anywhere inside each element — no spurious curvature invented, since the reconstruction
is anchored to the physically justified recovered gradient, not an arbitrary global spline fit.

---

## Step 6: Integrate Particle Trajectories with RK4

6.1. For each particle, at each RK4 stage, locate the containing element (via existing point-location
     algorithm).

6.2. Evaluate the Hermite-reconstructed velocity field (Step 5 output) at the particle's current
     position — NOT the raw linear FEM velocity, and NOT the gradient directly.

6.3. Proceed with standard RK4 stage combination:

    k1 = v(x_n, t_n)
    k2 = v(x_n + (dt/2)*k1, t_n + dt/2)
    k3 = v(x_n + (dt/2)*k2, t_n + dt/2)
    k4 = v(x_n + dt*k3, t_n + dt)
    x_(n+1) = x_n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    where v(...) always means "evaluate the Hermite reconstruction," never the gradient field.

6.4. If tracking through time-varying velocity fields (not just the single periodic steady-state
     snapshot), repeat Steps 1-5 once per stored time snapshot, or reuse the single steady-state
     reconstruction if using the periodic steady-state recycling strategy already implemented in
     your framework.

---

## Summary Table of the Full Pipeline

| Step | Input | Operation | Output |
|---|---|---|---|
| 1 | Nodal velocities, mesh | Element-wise gradient computation | Piecewise-constant full gradient tensor |
| 2 | Mesh connectivity | Build node patches | Patch-to-element map |
| 3 | Patch gradient samples | Least-squares polynomial fit (SPR/PPR) | Recovered nodal gradient |
| 4 | Recovered nodal gradients | Shape-function interpolation | Continuous recovered gradient field |
| 5 | Nodal velocities + recovered gradients | Hermite/cubic reconstruction per element | Smooth velocity reconstruction |
| 6 | Reconstructed velocity field | RK4 time integration | Particle trajectories |

---

## Practical Notes

- Steps 1-5 are a ONE-TIME post-process per stored velocity snapshot — do not repeat inside the
  nonlinear FEM solve or time-marching loop.
- Your existing strain rate output only gives the symmetric part of the gradient (Step 1.3);
  the antisymmetric (vorticity) part must be freshly computed from nodal velocities.
- Validate Step 1 output against your solver's own ZZ-error-estimator strain projection, if
  accessible, as a sanity check before proceeding to SPR.
- Consider PPR over plain SPR specifically in regions with distorted or highly stretched elements
  (e.g., near the tool-workpiece interface in FSW), where plain SPR's superconvergence assumption
  is weaker.
