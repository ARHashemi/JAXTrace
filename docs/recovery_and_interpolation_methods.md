# Gradient recovery and velocity interpolation methods in JAXTrace

**Scope.** This document is the single reference for every method
JAXTrace's RK4 kernel can use to look up a velocity at a query
position inside a P1 tetrahedral element. It covers:

1. What "gradient recovery" and "interpolation" mean in this
   codebase, and how they compose.
2. Every method currently implemented, with its formula, its
   accuracy contract, its computational cost, and when to use it.
3. A dedicated deep dive on the HCT-3D piecewise-cubic
   reconstruction (Alfeld split + Taylor-fit variant), since it
   is the newest and most involved method.

Everything below is grounded in the code under
`jaxtrace/gpu/recovery/` and the RK4 kernel in
`benchmark_femuss_comparison.py::create_rk4_comparison`. Line
references are provided where the shipped code is the ground
truth.

Companion documents:

* `docs/gradient_recovery_pipeline.md` — the "why" (mathematical
  motivation).
* `docs/hct3d_implementation_plan.md` — the design roadmap for
  HCT-3D and the numerical construction we chose.
* `docs/velocity_sources.md` — how these methods interact with the
  three velocity sources (mesh, analytic, ROM).

---

## 1. What is happening at each RK4 sub-stage

Given: a P1 tetrahedral mesh with per-node velocity samples
`v_a ∈ ℝ³`, a query point `p` inside a known element with local
vertices `x_0, x_1, x_2, x_3`, and the option of a **recovered
nodal gradient tensor** `G_a ∈ ℝ^{3×3}` at each node.

The RK4 kernel calls one function — `interpolate_velocity_single(p,
elem_id, velocity_field)` — to produce the velocity `v(p)`. The
choice of method determines which formula runs inside that call
and what precomputed arrays feed it.

Every method in JAXTrace fits into a two-stage pipeline:

```
┌──────────────────────────┐          ┌────────────────────────────┐
│ 1. Recovery (build-time) │  ───>    │ 2. Interpolation (RK4 hot) │
│                          │          │                            │
│  * SPR nodal gradients   │          │  Method-specific evaluator │
│  * Alfeld-split cubic    │          │  reads gathered arrays and │
│    coefficients          │          │  returns v(p).             │
└──────────────────────────┘          └────────────────────────────┘
```

**Recovery** runs **once** after the velocity field is loaded and
produces per-node or per-element arrays that are uploaded to the
GPU. **Interpolation** runs **once per particle per RK4 sub-stage
per step** — typically 4 × N_particles × N_steps ≈ 10⁹ evaluations
in a production run. So the interpolation stage must be tightly
JIT-compiled and gather-efficient; the recovery stage can afford
NumPy or a batched JAX solve.

The available methods, in order of increasing accuracy:

| Method | Recovery arrays | Interpolation cost | Kernel branch |
|---|---|---|---|
| Raw P1 | (none) | 1 direct-inverse solve + 1 dot | fallback |
| `centroid_taylor` | `(x_c, v_c, G_c)` per element | 1 matvec | `use_centroid_taylor` |
| `vertex_taylor` | `G_a` per node | 1 direct-inverse solve + 4 matvecs + 1 P1 blend | `use_vertex_taylor` |
| `hct_cubic_c0` | 20 Bernstein coeffs per sub-tet × 4 sub-tets per element | direct-inverse solve + argmin + 20-term Bernstein eval | `use_hct_cubic` |
| `hct_cubic_quad_faces` | same shape, different values | same as above | `use_hct_cubic` |
| `hct_cubic` | same shape, different values | same as above | `use_hct_cubic` |

**Priority order in the kernel** (highest fidelity that has all
its arrays uploaded wins):

```
hct_cubic > vertex_taylor > centroid_taylor > raw P1
```

The kernel branch is a Python-level `if` chain in
`benchmark_femuss_comparison.py` around lines 944–966; each branch
is inside a single `jax.jit`, so at run time exactly one method's
code path is compiled.

---

## 2. Steps 1–4 — SPR-based gradient recovery

Every method except raw P1 needs a **nodal gradient tensor** `G_a`
at each mesh node. The raw P1 element-average gradient is
piecewise-constant and jumps across every face, so we smooth it
via **Superconvergent Patch Recovery** (Zienkiewicz–Zhu 1992).

The implementation lives in
`jaxtrace/gpu/recovery/gradient_recovery.py`:

### Step 1 — raw element gradient tensor

`compute_element_gradients(node_positions, connectivity,
node_velocities)` computes the 3×3 gradient tensor on each parent
tet:

$$G_e = \sum_a v_a \otimes \nabla N_a$$

where `∇N_a` is the constant P1 shape-function gradient of vertex
`a` on element `e`. Vectorised over elements with `numpy.linalg.inv`
on the batched Jacobian.

**Cost**: NumPy vectorised, ~10 ms on 100k tets. Storage:
`(n_elements, 3, 3)` — kept only for diagnostics after the SPR
step consumes it.

### Step 2 — patch table

`build_node_patches(n_nodes, connectivity)` computes, for each
node `i`, the list of element ids that share vertex `i`. Returned
as a CSR-style pair `(patch_offsets, patch_elements)`.

**Cost**: NumPy, negligible.

### Step 3 — patch-local least-squares fit (SPR)

`spr_recover_nodal_gradients(...)` fits a **linear polynomial in
space** to the patch of raw element gradient tensors around each
node, then evaluates that polynomial at the node position.

For each node `i` and each of the 9 gradient-tensor components:

1. Collect the `(n_patch, 9)` array `S` of raw element gradients
   at their element centroids `(x_e, y_e, z_e)`.
2. Centre the centroids at the node: `X_c = X_patch - x_i`.
3. Build the design matrix `A = [1 | X_c]` (shape `(n_patch, 4)`).
4. Solve `A · [a₀, a₁, a₂, a₃] = S` via `numpy.linalg.lstsq`.
5. The recovered gradient value at node `i` is the constant term
   `a₀`.

**Numerical gotcha handled in the code**: at symmetric corner
nodes (Kuhn-tetrahedralised hex grids), the patch centroids lie
in a plane, making the full 4-column design matrix
numerically singular even though `X_c` is rank-3. The code checks
`np.linalg.cond(A) > 1e10` and falls back to the arithmetic mean
of the patch. This is the correct answer for constant fields and
a stable estimator for higher-order fields.

**Cost**: NumPy, ~1–3 s for 180k nodes (see the design doc for
why it stays in NumPy).

### Step 4 — reassemble as a nodal C⁰ field

Trivial: the output of Step 3 IS a nodal field; interpreting it
through P1 shape functions gives a continuous piecewise-linear
gradient tensor over the mesh. No further code required.

**All downstream methods consume the tensor `G_a ∈ ℝ^{3×3}` at
each node.** SPR runs unconditionally on the mesh path whenever
gradient recovery is enabled.

---

## 3. Step 5 — velocity reconstruction methods

Each method below produces `v(p)` from `(p, elem_id,
velocity_field)` plus method-specific precomputed arrays.

### 3.1 Raw P1 barycentric interpolation (baseline)

**Formula**:
$$v(p) = \sum_{a=0}^{3} N_a(p) \, v_a$$

where `N_a(p) = b_a` are the barycentric weights of `p` in the
element. The 4 weights come from the direct-inverse formula:

```
b_{123} = M⁻¹ (p − x₀)      (precomputed M⁻¹ per element)
b_0     = 1 − b_1 − b_2 − b_3
```

**Accuracy contract**: bit-exact for **linear** velocity fields;
first-order accurate elsewhere. The gradient of the reconstructed
field is piecewise-constant with jumps across every face — this
is the fundamental limitation that all the other methods below
try to fix.

**Cost**: 3×3 solve + 4 multiplies + 1 dot per query. Cheapest
possible. `M⁻¹` (shape `(n_elements, 3, 3)`) uploaded to GPU;
`velocity_field` (shape `(n_nodes, 3)`) is the raw nodal data.

**Kernel location**: `benchmark_femuss_comparison.py:1037-1075`
(the fallback branch of `interpolate_velocity_single`).

---

### 3.2 `centroid_taylor` (legacy alias: `taylor`)

**Formula**:
$$v(p) = v_c + G_c \cdot (p - x_c)$$

where `x_c` is the element centroid, `v_c` is the P1-blended
velocity at `x_c` (which for a P1 tet equals the mean of the 4
nodal velocities), and `G_c` is the P1-blended recovered gradient
at `x_c` (mean of 4 nodal gradients).

**Accuracy contract**: bit-exact for **linear** fields; first-order
in the recovered gradient elsewhere. Because `G_c` is a P1 blend
of smoothed nodal gradients, it captures cross-element smoothness
the raw P1 method misses — but the same `G_c` value is used
everywhere inside the element, so it does not adapt to the query
position.

**Cost**: one `(3, 3)` gather + one 3×3 matvec + one 3-vector add.
Very cheap. Storage: `(n_elements, 3)` centroid, `(n_elements, 3)`
`v_c`, `(n_elements, 3, 3)` `G_c`.

**When to use**: when you need something better than raw P1 with
minimal memory or compute overhead. Kept as the default in
`--recovery-method` for backward compatibility.

**Kernel location**: `benchmark_femuss_comparison.py:1009-1035`.

---

### 3.3 `vertex_taylor`

**Formula**:
$$v(p) = \sum_{a=0}^{3} N_a(p) \cdot \left[ v_a + G_a \cdot (p - x_a) \right]$$

Per-vertex Taylor expansion blended with the P1 shape functions.
For each of the 4 element vertices, we compute a first-order
Taylor expansion **anchored at that vertex** using its recovered
gradient `G_a`, then blend the four expansions with the query
point's P1 barycentric weights.

**Key property**: **exact at nodes**. Setting `p = x_b` gives
`N_a(x_b) = δ_{ab}` and `p - x_b = 0`, so `v(x_b) = v_b`.

**Accuracy contract**: bit-exact for **linear** fields. On
higher-order fields it is materially more accurate than
`centroid_taylor` because each vertex contributes its own
gradient (rather than the four vertex gradients being averaged
into one centroid tensor); the query-position blending then picks
up locally-appropriate gradient information.

Empirical measurement on the recirc_2026 § A field (32×16×4 mesh,
200 random interior samples): mean absolute error 3× lower than
`centroid_taylor`, max error 3.5× lower.

**Cost**: 4 gradient gathers (each `(3, 3)`) + 4 matvecs + 4
scaled adds + one 3-vector sum. ~4× the interpolation compute of
`centroid_taylor` but negligible next to the search cost that
dominates real runs. Storage: `(n_nodes, 3, 3)` for `G_a`
(uploaded once, gathered inline per query).

**When to use**: strong default for smooth fields with
appreciable curvature. Recommended over `centroid_taylor` unless
the recovered nodal gradient tensor exceeds VRAM budget (rare —
6 MB for a 180 k-node mesh).

**Kernel location**: `benchmark_femuss_comparison.py:962-1007`.

---

### 3.4 HCT-3D family: `hct_cubic_c0`, `hct_cubic_quad_faces`, `hct_cubic`

The three variants share a common structure — piecewise cubic
Bernstein–Bézier polynomials on the Alfeld split of each parent
tetrahedron — and differ only in **how the coefficients are
computed**. Section 4 below is the deep dive; the summary rows are:

| Variant | (μ, γ) at parent centroid | Face-centroid coefficients | Exactness contract |
|---|---|---|---|
| `hct_cubic_c0` | P1 average | P1 blend of 3 face-vertex values | linear-exact everywhere |
| `hct_cubic_quad_faces` | P1 average | Farin quadratic-precision formula | + quadratic-exact at parent face centroids |
| `hct_cubic` | Taylor-fit LS solve | Farin quadratic-precision formula | + quadratic-exact at parent centroid, spoke-edge midpoints, parent face centroids |

All three produce a `(n_elements, 4_subtets, 20_coeffs, 3_components)`
float32 array. All three share **the same kernel evaluator**:
argmin sub-tet detection, sub-tet barycentric via closed form,
Bernstein cubic eval via multi-index basis. Only the coefficient
values change.

**Empirical accuracy on the recirc_2026 § A field** (32×16×4 mesh,
200 random interior samples, mean absolute error):

| Method | Mean err | Max err | Relative to `centroid_taylor` |
|---|---|---|---|
| Raw P1 | 3.9e-1 | 6.7 | 0.9 × |
| `centroid_taylor` | 4.4e-1 | 7.5 | 1.0 × |
| `vertex_taylor` | 1.5e-1 | 2.4 | 3.0 × better |
| `hct_cubic_c0` | 3.8e-1 | 6.6 | 1.1 × better |
| `hct_cubic_quad_faces` | 2.1e-1 | 5.0 | 2.1 × better |
| `hct_cubic` | 8.2e-2 | 0.85 | **5.4× / 9× better** |

**When to use HCT**: smooth velocity fields with strong internal
curvature where the interpolation error (not the discretisation
error of the mesh) is the dominant contribution to trajectory
drift. On the recirc_2026 field, `hct_cubic` reduces mean error
by 5.4× vs `centroid_taylor` and reduces max error by 9× — a
qualitative improvement in worst-case behaviour.

**When NOT to use HCT**: on very coarse meshes where the P1
approximation itself is dominant (HCT reduces interpolation error,
not discretisation error), or when `vertex_taylor` already meets
the accuracy target and the extra ~800 MB VRAM (on a 900k-tet
mesh) is inconvenient.

**Kernel location**: `benchmark_femuss_comparison.py:914-960`.

---

## 4. HCT-3D deep dive

This section explains, in enough depth to reproduce or audit, what
`hct_cubic` actually does. Companion doc:
`docs/hct3d_implementation_plan.md` (design and complexity budget).

### 4.1 Motivation — why a single cubic per tet is not enough

A cubic polynomial on one tetrahedron has 20 Bernstein–Bézier
control values (one per multi-index `α = (α₀, α₁, α₂, α₃)` with
`|α| = 3`): 4 at vertices, 12 on edges (2 per edge × 6 edges), 4
at face centroids.

If we try to determine all 20 coefficients per element from mesh
data:

* 4 come naturally from vertex values.
* 12 come from vertex gradients (three per vertex × four vertices)
  via the 1-D Hermite along-edge formula.
* That leaves 4 face-centroid coefficients.

Between two neighbouring tetrahedra sharing a face, a single
cubic surface on each side would have to agree at the shared
face **as a bivariate cubic** — 10 coefficients. Since only 6
of those (3 vertex + 3 edges × 1 edge coefficient per side, taken
from the shared face vertex data) are automatically the same on
both sides, the two elements will disagree at the 3 face-edge
"other" coefficients and at the face centroid. So a single cubic
per tet cannot be even **C⁰-continuous** across faces without
compromising accuracy inside each element. This is the classical
"multivariate spline dimension counting" problem.

**The Alfeld split fixes this by adding degrees of freedom
without adding boundary vertices**: each parent tet is subdivided
into 4 sub-tets meeting at the parent centroid `v_c`. Each
sub-tet then holds its own cubic (20 coefficients each = 80
total). The 60 "extra" coefficients relative to a single cubic
are precisely the freedom needed to enforce continuity within the
parent AND allow shared boundary polynomial values to match a
consistent choice on neighbouring parents.

### 4.2 The Alfeld split

Given a parent tet with vertices `p_0, p_1, p_2, p_3`, compute
the centroid:

$$v_c = \tfrac{1}{4}(p_0 + p_1 + p_2 + p_3)$$

Form 4 sub-tets, each replacing one parent vertex with `v_c`:

```
Sub-tet 0:  (v_c, p_1, p_2, p_3)   —  base triangle opposite p_0
Sub-tet 1:  (v_c, p_0, p_2, p_3)   —  base triangle opposite p_1
Sub-tet 2:  (v_c, p_0, p_1, p_3)   —  base triangle opposite p_2
Sub-tet 3:  (v_c, p_0, p_1, p_2)   —  base triangle opposite p_3
```

The exact vertex ordering matters: JAXTrace's
`ALFELD_SUBTET_PARENT_VERTS` constant reuses
`TET_FACES` (outward-oriented face triangles) so all 4 sub-tets
inherit positive volume from a right-handed parent. This is
verified by `test_alfeld_subtet_volumes_all_positive_for_right_handed_parents`
(the test that caught an early bug where using raw vertex indices
gave alternating-sign volumes).

Geometry precompute — `build_alfeld_geometry` in `hct3d.py`:

* Parent centroid per element
* 6 edge midpoints per element (ordered per `TET_EDGES`)
* 4 face centroids per element (ordered per `TET_FACES`)
* Signed volume of each sub-tet
* Signed volume of the parent tet

Cost: NumPy vectorised, ~30 ms on 100 k elements.

### 4.3 What sits at each Bernstein multi-index

On each sub-tet with local vertices `(w_0, w_1, w_2, w_3)` and
barycentric coordinates `(β_0, β_1, β_2, β_3)`, the cubic Bézier
polynomial is:

$$u(β) = \sum_{|α|=3} c_α \cdot B^3_α(β), \quad B^3_α(β) = \frac{3!}{\prod α_i!} \prod_i β_i^{α_i}$$

The 20 multi-indices `α` group into three families:

| Group | Multi-index shape | Count | Physical location |
|---|---|---|---|
| Vertex | `3 e_i` | 4 | Sub-tet vertex `w_i` |
| Edge | `2 e_i + e_j` (i≠j) | 12 | Two per edge, 1/3 and 2/3 along |
| Face | `e_i + e_j + e_k` | 4 | Centroid of the face opposite `w_ℓ` |

Cubic Bézier tetrahedra have no interior control points.

JAXTrace stores multi-indices in the canonical order:

* Rows 0–3: vertex α = `(3,0,0,0), (0,3,0,0), (0,0,3,0), (0,0,0,3)`
* Rows 4–15: edge α values, 2 per edge, ordered per `TET_EDGES`
* Rows 16–19: face α = `(0,1,1,1), (1,0,1,1), (1,1,0,1), (1,1,1,0)`

The table is materialised once at import as `BERN_INDICES`
(shape `(20, 4)` int32). Lookup helpers
`_VERTEX_COEFF_INDEX`, `_EDGE_COEFF_INDEX`,
`_FACE_COEFF_INDEX`, and `_FACE_EDGE_COEFF_INDEX` map each
geometric role to a row in `BERN_INDICES`, so the JIT'd
assembly body has zero runtime lookup overhead.

### 4.4 Assembling the coefficients — variants

All three HCT variants share the vertex and edge coefficient
formulas; they diverge on face coefficients and on the value of
`(μ, γ) = (u(v_c), ∇u(v_c))`.

**Vertex coefficients** (all variants):

* `c_{3e_i}` at parent vertex `p_i` = nodal velocity `v_i`
* `c_{3e_0}` at centroid vertex `v_c` = `μ` (variant-specific)

**Edge coefficients** (all variants) — 1-D cubic Hermite along
each sub-tet edge from endpoint `A` to endpoint `B`:

$$c_{P→⅓} = u(A) + \tfrac{1}{3}\, \nabla u(A) \cdot (B - A)$$

At parent-vertex endpoints, `(u, ∇u)` are `(v_a, G_a)` — nodal
velocity and recovered nodal gradient. At `v_c` endpoints, they
are `(μ, γ)`.

**Face coefficients** — three formulas, one per variant:

#### `hct_cubic_c0` — linear-precision face rule

$$c^{\text{face}}_{111} = \tfrac{1}{3}(u_{v_1} + u_{v_2} + u_{v_3})$$

The P1 blend of the three face vertex values. Exact for linear
fields; adds no quadratic information.

#### `hct_cubic_quad_faces` — Farin's quadratic-precision rule

$$c^{\text{face}}_{111} = -\tfrac{1}{6} \sum_{\text{3 face-vertex coeffs}} + \tfrac{1}{4} \sum_{\text{6 face-edge coeffs}}$$

Derived by requiring the cubic to reproduce quadratic monomials
exactly. Verified symbolically with rational arithmetic on
`u = β_1^2` and `u = β_1 β_2` (see the Phase 3a-plus commit
message for the derivation) — the unique solution to c-value
quadratic exactness is `(x_v, x_e) = (-1/6, 1/4)`. Extends
`hct_cubic_c0`'s exactness to include **all parent face
centroids** for quadratic velocity fields.

#### `hct_cubic` — Farin face + Taylor-fit `(μ, γ)`

Same face rule as `hct_cubic_quad_faces`, but replaces the P1
average for `(μ, γ)` at the parent centroid with a **local
Taylor-fit** that captures quadratic information at `v_c` too.

### 4.5 Taylor-fit (μ, γ) — how `hct_cubic` decides the centroid data

For each parent tet, per velocity component, fit a **local
degree-2 Taylor expansion around v_c**:

$$u(x) \approx \mu + \gamma \cdot (x - v_c) + \tfrac{1}{2} (x - v_c)^\top H (x - v_c)$$

Against 16 constraints per component:

* 4 parent-vertex values: `u(p_i) = v_i` for `i = 0..3`
* 12 parent-vertex gradients: `∇u(p_i) = G_i` (row j gives
  `∂u/∂x_j` at `p_i`) for `i = 0..3, j = 0..2`

10 unknowns per component: `(μ, γ_x, γ_y, γ_z, H_{xx}, H_{yy},
H_{zz}, H_{xy}, H_{xz}, H_{yz})` (the symmetric 3×3 Hessian).

The system is overdetermined (16 rows for 10 unknowns), solved
by normal equations:

$$X = (A^\top A)^{-1} A^\top b$$

with `A` (shape `(16, 10)`) built from vertex-offset polynomials
(1, dx, dy, dz, ½dx², …) and `b` (shape `(16, 3)`) built from
concatenated vertex values and vertex gradients. Value rows are
scaled by `1/L` (mean vertex-to-centroid distance) so the value
and gradient constraints have comparable magnitude and the LS
system stays well-conditioned across mesh resolutions.

**Batched over all elements** as one `jax.jit`'d function:
`_build_taylor_lstsq_body` in `hct3d.py`.
`jnp.linalg.solve` on the batched `(n_elements, 10, 10)` normal
matrix.

**Only `μ` and `γ` are used downstream** — the Hessian `H` is a
free parameter of the LS fit and is discarded. This is
intentional: the cubic Bézier degrees of freedom don't include a
Hessian at the centroid, so we're using the fit only to nail down
the vertex-and-gradient information at `v_c`.

### 4.6 Degenerate-element fallback (Phase 6)

The 16×10 LS system can fail (produce NaN or wild values) on
near-degenerate parent tets: near-flat elements, near-colinear
neighbours, extreme aspect ratios. Left unchecked these would
corrupt the Bernstein coefficients and propagate into RK4
trajectories as silent errors.

Gate applied per-element inside the JIT'd LS body:

* `is_finite`: all 10 solution components finite (no NaN or Inf).
* `is_bounded`: max absolute component `< 1e6` (a safe
  overestimate; real Taylor coefficients on reasonable meshes are
  far below this).

When either check fails, the element's `(μ, γ)` are silently
replaced with the Phase 3a P1-average rule. This is a conservative
fallback: it sacrifices quadratic exactness on that element but
preserves linear exactness and continuity, and it prevents the
LS blowup from contaminating trajectories on the ~99% of healthy
neighbouring elements.

`build_hct_bernstein_c1_taylor` reports the per-mesh degenerate
count and percentage. On a scaled Kuhn hex-grid smoke mesh (12
288 tets) the gate fires on 912 elements (7.42 %) — the corner
and edge tets where the LS system is genuinely ill-conditioned
by symmetry. Tracking still runs to completion with 0 lost
particles.

### 4.7 Kernel-side evaluation

Given a query point `p` inside parent element `elem_id` with
its P1 barycentric weights `(b_0, b_1, b_2, b_3)` (already
computed by the direct-inverse formula for the raw P1 fallback
branch; reused):

**Step 1 — sub-tet detection.** The point is in sub-tet `s`
where `s = argmin_i b_i`. Rationale: sub-tet `s` has base
triangle opposite parent vertex `s`, and the point is closest to
that base (and farthest from `p_s`) precisely when `b_s` is the
smallest of the four. Verified: 500/500 matches at random
interior points of a reference tet. Cost in the kernel: one
`jnp.argmin`.

**Step 2 — sub-tet barycentric via closed form.** With `fs =
ALFELD_SUBTET_PARENT_VERTS[s]`:

```
β_0     = 4 · b_s                    (weight on v_c)
β_{k+1} = b_{fs[k]} - b_s             (weight on parent vertex fs[k]),  k=0,1,2
```

Verified bit-exact against ground-truth sub-tet barycentric
solves (max error < 1e-15). Cost: 4 subtractions + 1 multiply.

**Step 3 — Bernstein cubic evaluation.** Precompute per-axis
powers `β_j^0 = 1, β_j^1 = β_j, β_j^2 = β_j², β_j^3 = β_j³` for
each of the 4 barycentric axes. Then the Bernstein basis at 20
multi-indices is:

$$\text{basis}[k] = \text{mult}[k] \cdot β_0^{α_{k,0}} \cdot β_1^{α_{k,1}} \cdot β_2^{α_{k,2}} \cdot β_3^{α_{k,3}}$$

with `mult[k] = 6 / (α_{k,0}! α_{k,1}! α_{k,2}! α_{k,3}!)`
precomputed at kernel-build time and baked into the JIT'd graph
as a static constant.

Final value: single einsum over the coefficient tensor at the
selected sub-tet:

$$v(p) = \sum_{k=0}^{19} \text{basis}[k] \cdot c_{k}[s]$$

Cost in the kernel: 4 lookups (β-power tables) + 20 multiplies
+ 20-term dot product, all bound to XLA-fused code.

### 4.8 Cost summary for `hct_cubic`

Per query at run time:

| Stage | Cost |
|---|---|
| Direct-inverse solve for parent barycentric | 1 `(3, 3)` matvec + adds |
| `argmin` sub-tet detection | 1 |
| Sub-tet barycentric closed form | 4 subs + 1 mul |
| Bernstein basis (20 values) | ~50 multiplies |
| Coefficient gather and dot product | 20-way `einsum` |
| **Total** | ~1.3 × the raw-P1 cost |

The compilation graph size adds ~1 s to the JIT compile once per
process; the runtime cost per RK4 step is dominated by the
mesh-search phase (not by this interpolation), so overall
throughput is comparable to `vertex_taylor` in end-to-end runs.

Precompute (once per mesh + velocity field, at build time):

| Stage | Cost on 900k tets |
|---|---|
| SPR nodal gradient recovery | ~1–3 s (NumPy) |
| Alfeld geometry precompute | ~50 ms |
| Edge midpoint gradients | ~10 ms |
| Taylor-fit LS solve, batched | ~200 ms (JAX-JIT) |
| Bernstein coefficient assembly | ~1.5 s (JAX-JIT) |
| **Total** | ~5 s |

VRAM footprint: `(n_elements, 4, 20, 3)` at float32 = ~865 MB on
a 900k-tet mesh. This is the practical upper bound on mesh size
that can accommodate HCT within a 32 GB VRAM budget with room
for the RK4 kernel, particles, and search structures.

---

## 5. Cross-cutting concerns

### 5.1 Interaction with the RK4 kernel

`interpolate_velocity_single` in
`benchmark_femuss_comparison.py::create_rk4_comparison` reads the
following optional keyword arguments, in priority order:

```python
create_rk4_comparison(
    ...,
    # HCT (highest priority)
    hct_bernstein_gpu=None,          # (n_elements, 4, 20, 3) float32
    # vertex_taylor
    node_gradient_gpu=None,          # (n_nodes, 3, 3) float32
    # centroid_taylor
    element_centroid_gpu=None,       # (n_elements, 3) float32
    element_v_centroid_gpu=None,     # (n_elements, 3) float32
    element_gradient_gpu=None,       # (n_elements, 3, 3) float32
    # raw P1 fallback if all above are None
)
```

`run_tracking.py::main` builds the appropriate array based on
`--recovery-method` and passes it through. Only one method's
arrays are uploaded to the GPU per run.

### 5.2 Interaction with velocity sources

Recovery + interpolation works identically regardless of where
the per-node velocity data came from:

| Source | Recovery inputs | Notes |
|---|---|---|
| FOM mesh PVTU | `node_velocities` from the loaded PVTU | Standard path. |
| ROM reconstruction | `node_velocities` from `--velocity-source rom` | ROM reconstructs at mesh nodes, then Steps 1–5 apply. |
| Analytic mesh projection | `node_velocities` sampled from `--velocity-source analytic --project-to-mesh` (if used) | Analytic-source runs bypass recovery entirely by default because there is no mesh field to recover gradients of. |

Details in `docs/velocity_sources.md`.

### 5.3 What is verified by the test suite

`tests/recovery/`:

* `test_hct3d_geometry.py` — 16 tests: `TET_EDGES`/`TET_FACES`
  invariants, sub-tet volume sanity, Alfeld geometry
  correctness.
* `test_hct3d_bernstein.py` — 28 tests: multi-index catalog
  invariants, Bernstein evaluator correctness, per-variant
  exactness contracts on constant/linear/quadratic fields at
  various geometric locations (parent vertices, parent centroid,
  spoke-edge midpoints, parent face centroids), C⁰ continuity at
  spoke and parent-face edges, degenerate-element fallback.

Total: 44 tests, all passing. Run with:

```bash
python tests/recovery/test_hct3d_geometry.py
python tests/recovery/test_hct3d_bernstein.py
```

Each file has a `__main__` runner so you don't need pytest.

---

## 6. Choosing a method — decision tree

```
Does the field have significant internal curvature (not just linear-in-x)?
├── No → raw P1 or centroid_taylor
└── Yes
    ├── Do you have SPR-recovered nodal gradients built already?
    │   ├── No → centroid_taylor (cheapest smoothing option)
    │   └── Yes
    │       ├── Is the recirc/HCT accuracy (5×+ over centroid) needed?
    │       │   ├── No → vertex_taylor (fast, exact at nodes)
    │       │   └── Yes
    │       │       ├── Does the mesh fit within the VRAM budget for the
    │       │       │   Bernstein coefficient array (~1 kB / element)?
    │       │       │   ├── No → vertex_taylor
    │       │       │   └── Yes → hct_cubic
```

For most production workflows: **default to `hct_cubic` on
smooth analytic-source problems, `vertex_taylor` on FOM/ROM
problems where the mesh dominates the field's roughness.** The
CLI default is `centroid_taylor` for backward compatibility;
override with `--recovery-method hct_cubic` or
`--recovery-method vertex_taylor` as appropriate.

---

## 7. Provenance and changelog

| Feature | Landed | Reference |
|---|---|---|
| SPR + centroid_taylor Step 5 | `54b6f55` | pipeline doc |
| vertex_taylor evaluator | `96bac64` | commit message |
| HCT-3D design roadmap | `7478a93` | `hct3d_implementation_plan.md` |
| HCT-3D Phase 1+2 (Alfeld geometry + edge DOFs) | `babf2eb` | `hct3d.py` |
| HCT-3D Phase 3a (C⁰ Bernstein) | `4abeee5` | `build_hct_bernstein_c0` |
| HCT-3D Phase 3a JAX migration | `8bc5d04` | 5× speedup |
| HCT-3D Phase 3a-plus (Farin faces) | `e932955` | `build_hct_bernstein_quad_faces` |
| HCT-3D Phase 3b (Taylor-fit `(μ, γ)`) | `9dcd469` | `build_hct_bernstein_c1_taylor` |
| HCT-3D Phase 4 (kernel evaluator + CLI) | `a9547ac` | `--recovery-method hct_cubic` |
| HCT-3D Phase 5+6 (fallback + reporting + docs) | `824f10d` | this doc, degenerate-gate |

Full commit series lives on `feature/analytic-velocity`.
