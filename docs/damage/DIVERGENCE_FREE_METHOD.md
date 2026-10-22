# Making the FSW Velocity Field Divergence-Free

**A complete, self-contained explanation of the problem, the method that works,
and how it runs inside the tracking pipeline.**

Status: method validated end to end. Full-length production gate running at
time of writing (LUMI job 23062898).
Companion: `DIVERGENCE_FREE_PLAN.md` is the chronological working log with
every measurement and every refuted approach. This document is the explanation.

---

## Table of contents

1. [What the problem is, in plain terms](#1-what-the-problem-is-in-plain-terms)
2. [Why it matters for void prediction](#2-why-it-matters-for-void-prediction)
3. [The mathematics, built from scratch](#3-the-mathematics-built-from-scratch)
4. [The method: a three-line closed form](#4-the-method-a-three-line-closed-form)
5. [Why this is the *best possible* correction](#5-why-this-is-the-best-possible-correction)
6. [What it gives up, honestly](#6-what-it-gives-up-honestly)
7. [How it is implemented in the pipeline](#7-how-it-is-implemented-in-the-pipeline)
8. [Algorithms and pseudocode](#8-algorithms-and-pseudocode)
9. [Where each piece lives in the code](#9-where-each-piece-lives-in-the-code)
10. [All the measurements](#10-all-the-measurements)
11. [What we tried that did not work, and why](#11-what-we-tried-that-did-not-work-and-why)
12. [How to run it](#12-how-to-run-it)
13. [Open items](#13-open-items)

---

## 1. What the problem is, in plain terms

### 1.1 The physical claim

Aluminium being friction-stir welded is **incompressible**. Squeeze it in one
direction and it must expand in another; the volume of any blob of material
cannot change. If we track a small parcel of metal through the weld and its
computed volume grows by 50%, that is not physics — it is numerical error.

### 1.2 Where the trouble starts

Our velocity field comes from FEMUSS, a finite-element CFD solver. It gives us,
at every **node** of a tetrahedral mesh, a velocity vector. For incompressible
flow the solver *should* produce a field satisfying

$$\nabla\cdot u = \frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y} + \frac{\partial u_z}{\partial z} = 0$$

everywhere. **It does not.** Measured on case D2, snapshot 34:

| quantity | value |
|---|---|
| typical `\|∇·u\|` | 2.29 s⁻¹ |
| typical gradient magnitude `‖A‖` | 151.6 s⁻¹ |
| **relative divergence** | **~0.7 % – 1.5 %** |

So roughly one part in a hundred of the velocity gradient is spurious
volume creation or destruction.

### 1.3 Why the solver does this (not a bug)

FEMUSS uses **P1/P1** elements: velocity and pressure both linear on each
tetrahedron. That combination is famously **not inf-sup stable** — it violates
the Ladyzhenskaya–Babuška–Brezzi condition, which is the mathematical
requirement for a velocity–pressure pair to work. Used naively it produces
wild checkerboard pressure oscillations.

The standard fix is **stabilisation**: add a small artificial term to the
equations that suppresses the oscillation. It works, the pressures come out
smooth, and the price is that the velocity field is **no longer exactly
divergence-free**. The divergence we measure *is* the stabilisation.

> **Key consequence, which cost us a lot of effort to appreciate:** the
> divergence is not a small random error sprinkled on a good field. It is a
> structural property of a stabilised discretisation. That is why every method
> from the literature that assumes a *stable* mixed discretisation failed on
> our data (§11).

---

## 2. Why it matters for void prediction

We track particles through the weld and accumulate damage along each pathline.
The central kinematic quantity is the **volume ratio**

$$\det F = \frac{\text{volume now}}{\text{volume at the start}}$$

where `F` is the deformation gradient. For incompressible material
`det F = 1` exactly, always.

There is a beautiful identity connecting this to divergence:

$$\boxed{\ \ln(\det F) = \int_{\text{pathline}} (\nabla\cdot u)\;dt\ }$$

So `det F` is the exponential of the accumulated divergence along the path.
This gives us a **free self-test**, which we call the *step-1 gate*:

> Track fully-entrained particles. If the field were truly divergence-free,
> every particle would report `det F = 1` to round-off. Whatever deviation we
> see is the **error floor of the entire pipeline** — and any void signal we
> later claim must be *larger* than that floor, or we are just reporting noise.

### 2.1 The gate was failing badly

| case | max `\|det F − 1\|` on the raw field |
|---|---|
| D2 | 0.072 (7 %) |
| C1 | 0.587 (59 %) |
| **B1** | **2.734 (273 %)** |

A 273 % volume change on incompressible aluminium. The gate was not marginally
failing — it was **meaningless**. `det F` was measuring the solver's
stabilisation error, not material volume change.

⚠️ And because `ln J` is a **path integral**, the error *grows with path
length*. A short test march understates it badly: the same D2 configuration
measured 1.43e-05 over 50 steps and 3.17e-03 over the full 8,000-step run.
Any validation must use production path lengths.

### 2.2 Why not just ignore it?

Because every damage model we use is driven by the velocity **gradient**, not
the velocity value:

- Rice–Tracey void growth — driven by `η = σ_m/σ_eq`, from the stress, from the strain rate, from `A`
- Cockcroft–Latham — same
- Piwnik–Plata bonding — same
- strain rate `ε̇`, residence time, the Bingham number — all from `A`

If `A` is wrong, every indicator is wrong. The divergence is the one piece of
`A` we can prove is wrong, so it is the obvious thing to fix.

---

## 3. The mathematics, built from scratch

This section assumes nothing beyond matrices and partial derivatives.

### 3.1 The velocity field on one tetrahedron is exactly affine

The FOM gives velocity at nodes, and the finite-element interpolation inside
an element is **linear** (that is what "P1" means). On a tetrahedron with
vertices `x₀…x₃` and nodal velocities `u₀…u₃`, there is exactly one affine map

$$u(x) = A\,x + b$$

reproducing all four — because four points in 3D determine an affine map
uniquely. Solve the 4×4 system

$$\begin{pmatrix} x_0^T & 1\\ x_1^T & 1\\ x_2^T & 1\\ x_3^T & 1\end{pmatrix}
  \begin{pmatrix} A^T \\ b^T \end{pmatrix}
  = \begin{pmatrix} u_0^T\\ u_1^T\\ u_2^T\\ u_3^T\end{pmatrix}$$

Here `A` is the 3×3 **velocity gradient**, `A_ij = ∂u_i/∂x_j`, and `b` is a
constant offset.

✅ **Verified**: the residual of this fit is **1.4e-16** (machine epsilon) on
10.8 M elements. It is an identity, not an approximation.

> **This single fact is the foundation of everything that follows.** We are not
> approximating the FOM field — on each element we have it *exactly*, in three
> numbers times three plus three.

### 3.2 Divergence is the trace

For an affine field, every derivative is a constant entry of `A`:

$$\nabla\cdot u = \frac{\partial u_x}{\partial x}+\frac{\partial u_y}{\partial y}+\frac{\partial u_z}{\partial z} = A_{11}+A_{22}+A_{33} = \operatorname{tr}A$$

So **"divergence-free" means exactly "`A` has zero trace"**. The whole problem
reduces to: our matrices have a non-zero trace, and we want trace zero.

### 3.3 The decomposition that solves it

Any square matrix splits uniquely into a trace-free part plus a multiple of the
identity:

$$A = \underbrace{\left(A - \frac{\operatorname{tr}A}{3}I\right)}_{\text{trace-free (deviatoric)}} + \underbrace{\frac{\operatorname{tr}A}{3}I}_{\text{pure expansion}}$$

Check the first term's trace: `tr(A) − (tr A/3)·tr(I) = tr A − 3·(tr A/3) = 0`. ✓

The second term, `(tr A/3)·I`, is **pure isotropic expansion** — uniform
growth or shrinkage in all directions. That is precisely the unphysical part.
The first term carries all the shear and rotation, which is the physics we want
to keep.

**So the fix is to delete the second term.**

### 3.4 Keeping the velocity value unchanged

If we change `A → A'`, then `u(x) = A'x + b` gives different velocities
everywhere, which would move particles to the wrong places. We want to change
the **gradient** without moving the field. Pick a reference point `x_c` and
demand `u'(x_c) = u(x_c)`:

$$A' x_c + b' = A x_c + b \quad\Longrightarrow\quad b' = b + (A - A')\,x_c$$

Substituting `A − A' = (tr A/3)I` gives the compact form

$$b' = b + \frac{\operatorname{tr}A}{3}\,x_c$$

### 3.5 Which reference point?

We use the **element centroid** `x_c = (x₀+x₁+x₂+x₃)/4`.

⚠️ **This choice is not cosmetic — it is what makes the result a field.** If we
anchored at the *particle's current position* instead, then by construction
`u'(x_pos) = u(x_pos)` at every point we ever evaluate, which means **no
correction at all**. The corrected field must be defined independently of where
we sample it. (This mistake was made and caught before running.)

The centroid is the natural choice: it is the element's own geometric centre,
it is independent of time and of the velocity, and the resulting field is
exactly the minimum-norm one (§5).

---

## 4. The method: a three-line closed form

For each tetrahedron, with `A, b` from §3.1 and centroid `x_c`:

$$\boxed{\begin{aligned}
s &= \tfrac{1}{3}\operatorname{tr}A = \tfrac{1}{3}(A_{11}+A_{22}+A_{33})\\
A' &= A - s\,I\\
b' &= b + s\,x_c
\end{aligned}}$$

and the corrected velocity field is `u'(x) = A'x + b'`, or equivalently and
more usefully for implementation:

$$\boxed{\ u'(x) = u(x) - s\,(x - x_c)\ }$$

Those two forms are algebraically identical — verified numerically to
**5.4e-15** over 40 random tetrahedra. The first is convenient on the host when
you hold `A` and `b`; the second is convenient inside the GPU kernel, where you
have `u(x)` and can get `s` by differentiating.

### 4.1 Three properties, all exact

| property | why | verified |
|---|---|---|
| `∇·u' = 0` | `tr A' = tr A − 3s = 0` by construction | 3.0e-18 |
| `u'(x_c) = u(x_c)` | by the choice of `b'` | exactly 0.0 |
| minimal change to `A` | §5 | 0.66 % typical |

### 4.2 What it costs

**Nothing.** No linear solve, no iteration, no communication between elements,
no stored field. It is a trace, a subtraction and a scalar multiply — about
ten floating-point operations per element. Applied to a whole 10.8 M-element
snapshot on one CPU core it takes **13 seconds**; applied inside the GPU kernel
it costs a few extra FLOPs per velocity evaluation and **zero extra memory**.

---

## 5. Why this is the *best possible* correction

Not "a good heuristic" — provably optimal, and worth understanding because it
is what justifies not looking for anything better.

**Claim.** Among all trace-free matrices, `A' = A − (tr A/3)I` is the one
closest to `A` in the Frobenius norm.

**Proof.** We minimise `‖X − A‖²_F` subject to `tr X = 0`. Lagrangian:

$$L = \|X-A\|_F^2 - 2\lambda\operatorname{tr}X$$

Setting `∂L/∂X = 0` gives `2(X − A) − 2λI = 0`, so `X = A + λI`. Imposing the
constraint: `tr X = tr A + 3λ = 0`, hence `λ = −(tr A)/3`, giving
`X = A − (tr A/3)I`. ∎

The geometric reading: trace-free matrices form a linear subspace, and the
closest point in a subspace is the **orthogonal projection** onto it. Our
formula *is* that projection. (Indeed `⟨A', I⟩ = tr A' = 0`, so the removed
part is orthogonal to what remains.)

**The size of the unavoidable change** follows immediately:

$$\frac{\|A'-A\|_F}{\|A\|_F} = \frac{\|(tr A/3)I\|_F}{\|A\|_F} = \frac{|tr A|\sqrt{3}/3}{\|A\|_F} = \frac{|\nabla\cdot u|}{\sqrt{3}\,\|A\|_F}$$

Measured on D2 and A2 — note these are **lower bounds for any method
whatsoever**:

| percentile | D2 | A2 |
|---|---|---|
| p50 | 0.66 % | 0.41 % |
| p90 | 5.3 % | 3.5 % |
| p99 | 28 % | 21 % |
| max | 62 % | 64 % |

> So no divergence-free field can be closer to the FOM gradient than ours.
> Any other method is either equal or worse. **This retires the search.**

---

## 6. What it gives up, honestly

The corrected field is **discontinuous across element faces**.

Each element gets its own `s = tr A/3`, and two elements sharing a face
generally have different `s`. So a particle crossing a face sees `A` jump. In
finite-element language the field is no longer `H(div)`-conforming, and not a
finite-element function in the usual sense.

**This was the stated reason we spent weeks on BDM₁** (§11), which *is*
conforming. So why is it acceptable now?

### 6.1 Because our integrator never looks at a face

A pathline integrator does exactly one thing with the velocity field: it
**evaluates `u` at a point inside whichever element contains the particle**. It
never evaluates a normal trace, never integrates a flux across a face, never
solves a variational problem.

Conformity exists to make *variational problems* well-posed. We do not have a
variational problem. We have pointwise evaluation.

### 6.2 And because we measured it rather than argued it

The honest concern was: the per-element divergence is zero, but does the
**path integral** `ln J = ∫(∇·u)dt` stay zero when the particle keeps jumping
between elements with different corrections?

We tested it directly with an integrator written independently of the
production code (pure numpy, no JAX):

| case | raw `\|ln J\|` p50 | **corrected `\|ln J\|` p50** | face crossings per particle |
|---|---|---|---|
| D2 | 7.63e-04 | **1.66e-18** | 59 (max 105) |
| C1 | 1.30e-02 | **2.98e-18** | 72 (max 132) |
| B1 | 1.49e-01 | **5.55e-18** | 78 (max 140) |

`|det F − 1|` came out **exactly 0.0** in float64 in all three.

The face jump was exercised **dozens of times per pathline** and did not
matter. The reason, once stated, is obvious: every RK4 substage lands inside
*some* element, and in **every** element the divergence is zero to round-off.
The jump changes *which* zero-divergence map we sample, never that it is zero.

### 6.3 And the trajectories barely move

| case | trajectory shift | distance travelled | **relative** |
|---|---|---|---|
| D2 | 6.4e-09 | 8.9e-04 | **8.2e-06** |
| C1 | 7.2e-08 | 1.1e-03 | **9.6e-05** |
| B1 | 2.8e-07 | 1.1e-03 | **2.9e-04** |

Relative shift ≤ 3e-04, and 200/200 particles stayed alive in both fields. So
the damage map is preserved while `det F` becomes exact — we are not trading
one error for another.

⚠️ **Do not reuse this field for anything needing continuity** — a flux
computation, a conservative remap, an FE post-process. It is correct for
pointwise pathline evaluation and nothing else.

---

## 7. How it is implemented in the pipeline

### 7.1 Two possible places, and why we chose the second

**Option A — host-side, before upload.** Correct `A` once per element on the
CPU, upload the corrected gradients. Clean, zero kernel changes.
❌ **Rejected.** It needs a stored `(n_frames, n_elements, 3, 3)` array. On D2
at 166 frames in float32 that is **64.6 GB** against a **32.6 GB** GPU. It
only works for a single steady snapshot, and production is 166 frames.

**Option B — in-kernel, on the fly.** ✅ **Chosen.** The kernel can compute
`s = tr A/3` itself from data it already has in registers, so **nothing is
stored per frame**.

### 7.2 Why the kernel can do this for free

The velocity interpolation already gathers the four nodal velocities and holds
the inverse element matrix `M_inv`. From those, `A` follows:

$$A = \sum_{a=0}^{3} u_a \otimes \nabla N_a$$

where `∇N_a` are the shape-function gradients, which are exactly the rows of
`M_inv` (and `∇N₀ = −Σ∇N_a`). ✅ Verified against the exact affine map to
**8.9e-16**.

In practice we get it even more cleanly: `jax.jacfwd` differentiates the
velocity interpolation closure directly, giving the full 3×3 Jacobian. That
has a real advantage — it **wraps** the active reconstruction rather than
hard-coding P1 algebra, so the correction can never drift from the field it
corrects.

⚠️ **But it is only EXACT for a reconstruction whose gradient is constant per
element** — raw P1 and `centroid_taylor`. See §7.6; `vertex_taylor` and the
`hct_*` variants are blocked at parse time.

The only other input is `x_c`, the element centroid. That is **pure geometry**:
independent of velocity, independent of time. One `(n_elements, 3)` array,
**0.13 GB on D2 at any frame count**.

| approach | per-frame memory | works at 166 frames? |
|---|---|---|
| stored corrected gradients | 0.39 GB/frame → 64.6 GB | ❌ |
| **in-kernel + centroids** | **0 (+0.13 GB once)** | **✅** |
| HCT Bernstein (for scale) | 10.4 GB/frame → 1.72 TB | ❌ |

### 7.3 The seven pipeline stages, and where the correction appears

```
[1/7] Load mesh           PVTU → nodes, connectivity, velocity sequence
                          (166 frames × 3 fields; dedup piece boundaries)
[2/7] Precompute metadata element geometry, M_inv, p0, volumes, AABBs
[3/7] Build + upload      octree; device_put everything
      ★ CENTROID UPLOAD HERE — if --divergence-free and the recovery path
        did not already supply them, compute x_c = mean(nodes[conn]) and
        upload. 0.13 GB, once, time-independent.
[4/7] Seed particles      grid-frac / box / femuss → initial positions
[5/7] Initial assignment  locate each particle's starting element
[6/7] Build RK4           ★ THE WRAPPER IS INSTALLED HERE — at kernel BUILD
        time (trace time), interpolate_velocity_single is replaced by the
        corrected version. Costs nothing at run time: the projection is
        baked into the compiled graph as a few extra FLOPs.
[7/7] Run N steps         the vmapped RK4 loop; every velocity evaluation
                          (all 4 substages) now returns the corrected field
```

⚠️ The key design point: the correction is installed **once, at build time**,
by swapping a Python function reference *before* JAX traces the kernel. There
is no per-step branch, no runtime flag check, and no host interaction. XLA
compiles the corrected arithmetic directly into the fused kernel.

### 7.4 How it coexists with `vmap`

The standing constraint on this project is: **no in-step CPU↔GPU transfer, do
not break the single-vmap architecture.** The design respects both.

The tracking kernel is one `jax.vmap` over a per-particle function:

```python
positions_final, element_ids_final, vels_out, dmg_out = jax.vmap(
    rk4_single_particle
)(positions_gpu, element_ids_gpu, vels_in, dmg_in)
```

`rk4_single_particle` calls `interpolate_velocity_single` four times (k1…k4).
We replaced that one function with a wrapper. Consequences:

- ✅ **The vmap structure is untouched** — same single vmap, same signature,
  same batching. The projection is *inside* the mapped function.
- ✅ **No host transfer** — `s` is computed from registers; `x_c` is a gathered
  row of a resident array.
- ✅ **No new control flow** — no `lax.cond`, no branch. (This matters here:
  nested `lax.cond` under `vmap` has caused real compilation artefacts in this
  codebase.) The only conditional is a `jnp.where` on element validity, which
  vectorises cleanly.
- ⚠️ **Cost**: three extra forward-mode Jacobians per step per particle. The
  `analytic` divergence path already built one on the k1 stage; the wrapper
  puts one on all four. This is **compute**, not transfer. XLA should fuse each
  Jacobian with the velocity evaluation it differentiates (same gathers, same
  registers), and the measured 200-step smoke test showed no slowdown — but
  that run was launch-latency bound, so the real figure awaits the
  compute-bound full run.

### 7.5 How `ln J` picks up the correction automatically

This is the most elegant part. The damage accumulator computes divergence by

```python
def divergence_single(pos, elem_id, velocity_field):
    J = jax.jacfwd(interpolate_velocity_single, argnums=0)(pos, elem_id, velocity_field)
    return jnp.trace(J)
```

It reads `interpolate_velocity_single` **by name, at call time**. Since we
rebound that name to the wrapper at build time, `divergence_single`
automatically differentiates the **corrected** field and returns
`tr A' = 0`. ✅ No separate code path, and the velocity and its divergence
**cannot** drift apart by construction.

Verified in isolation: `tr(jacfwd(corrected)) = −1.1e-16` versus
`tr(jacfwd(raw)) = 1.608`.

### 7.6 ⚠️ Which reconstructions it is exact for, and why

The wrapper subtracts `s·(x − x_c)` with `s = tr(∇u)/3` evaluated at the
sample point. Take the divergence of the result:

$$\nabla\cdot u' = \nabla\cdot u - \nabla\cdot\big[s(x)\,(x-x_c)\big]
 = \nabla\cdot u - \big[3s(x) + \nabla s\cdot(x-x_c)\big]
 = -\,\nabla s\cdot(x-x_c)$$

using `3s = ∇·u`. ✅ Verified numerically to 8 significant figures.

So the leftover divergence is **exactly the term from `s` varying in space**,
and it vanishes **iff `∇s = 0`**, i.e. iff the reconstruction's gradient is
constant on each element.

| reconstruction | gradient within an element | projection exact? |
|---|---|---|
| raw P1 (`--gradient-recovery 0`) | constant | ✅ yes |
| `centroid_taylor` | constant (`v_c + G_c·(x−x_c)`) | ✅ yes |
| `vertex_taylor` | **varies** (blended per-vertex Taylor) | ❌ no |
| `hct_cubic` and variants | **varies** (piecewise cubic) | ❌ no |

On a synthetic `vertex_taylor` field, **~1.2 % of the original divergence
survived** the projection. That is far better than nothing but it is not zero,
and this flag exists to deliver exactly zero — a run quietly returning 98.8 %
of a divergence-free field would be reported as a pass.

**So the two varying-gradient methods are blocked at parse time**, with a
message naming the reason and the alternative. ⚠️ An earlier draft of this
document claimed the wrapper "inherits whatever reconstruction is active"; that
is true of the *value* but false of the *divergence*, and the claim was
corrected after testing all four rather than only the one that had been run.

If a higher-order reconstruction is ever needed together with this flag, the
fix is to subtract the isotropic part of the Jacobian **pointwise** in the
integrator's gradient rather than as an `(x − x_c)` field — which changes what
`A` means downstream and is a separate design question.

### 7.7 ✅ It does not disturb the masked tool region

The kernel zeroes velocity inside the tool with a level-set mask,
`val = where(ls_val >= 0, val, 0)` — "no material here, so no flow and no
damage".

The concern was real enough to test: the wrapper returns
`u − (tr(Jac)/3)(pos − x_c)`, and inside the tool `u = 0` but `(pos − x_c)` is
**not** zero. If `tr(Jac)` were non-zero there, the correction would **inject
velocity into a region that must stay stationary** — and the mask is a step
function, so an element straddling the interface mixes masked and unmasked
nodes.

✅ **Measured clean**, including the straddling case:

| case | mean `‖u_raw‖` | mean `‖u_wrapped‖` | injected where `u_raw = 0` |
|---|---|---|---|
| all 4 nodes inside the tool | 0.000e+00 | 0.000e+00 | **0.000e+00** |
| all 4 outside | 7.07e-01 | 7.37e-01 | 0.000e+00 |
| **straddling the interface** | 3.52e-01 | 3.81e-01 | **0.000e+00** |

The reason is structural rather than lucky: `jnp.where` has **zero gradient on
the masked side**, so `tr(Jac) = 0` exactly where `u = 0`, and the correction
term vanishes together with the velocity. Nothing is injected.

⚠️ This is also why `--divergence-free` **requires `--damage-divergence
analytic`** (the default). The alternative, `field`, reads a *precomputed
nodal* `div_u` built from the raw P1 gradient — it cannot respond to the
projection, so `ln J` would silently integrate the uncorrected divergence while
the velocity was corrected. That would be a wrong `det F` with no visible
symptom, so it is blocked at argument-parse time.

---

## 8. Algorithms and pseudocode

### 8.1 The core projection (the whole method)

```
ALGORITHM 1 — Trace-free projection of one element's affine velocity map
───────────────────────────────────────────────────────────────────────────
INPUT   A  : 3x3 velocity gradient,  A[i][j] = du_i/dx_j
        b  : 3-vector offset,  so that u(x) = A x + b
        xc : 3-vector element centroid
OUTPUT  A', b'  with  tr(A') = 0  and  u'(xc) = u(xc)

1.  s  <-  ( A[0][0] + A[1][1] + A[2][2] ) / 3        # one third of divergence
2.  A' <-  A
3.  for i in 0,1,2:  A'[i][i] <- A'[i][i] - s          # remove the trace
4.  b' <-  b + s * xc                                  # keep u(xc) fixed
5.  return A', b'

COST     ~10 flops. No solve, no neighbours, no iteration.
EXACT    tr(A') = tr(A) - 3s = 0 identically.
⚠️ Do steps 1 and 3 in float64 even if storage is float32: u = A x + b cancels
   ~6.8 decimal digits and float32 carries only ~7.2.
```

### 8.2 The equivalent in-kernel form

```
ALGORITHM 2 — The same correction, evaluated pointwise (what the GPU runs)
───────────────────────────────────────────────────────────────────────────
INPUT   pos        : particle position (3-vector)
        elem_id    : index of the containing element (-1 if outside)
        vel_field  : nodal velocities for the current frame
        RAW        : the uncorrected interpolation function
        centroids  : resident (n_elements, 3) array
OUTPUT  corrected velocity at pos

1.  u    <-  RAW(pos, elem_id, vel_field)              # the velocity we'd use
2.  Jac  <-  jacobian_of(RAW) at pos                   # 3x3, forward mode
3.  s    <-  trace(Jac) / 3
4.  xc   <-  centroids[elem_id]
5.  ok   <-  (elem_id >= 0) AND (elem_id < n_elements)
6.  return  ok ? ( u - s * (pos - xc) )  :  u          # escapees unchanged

WHY EQUIVALENT TO ALGORITHM 1
    u'(x) = A x + b - s(x - xc)
          = (A - sI) x + (b + s xc)
          = A' x + b'                                   ✓ verified 5.4e-15
WHY NO STORED GRADIENT
    step 2 differentiates a function of data already in registers.
⚠️ step 5: an escaped particle must not read element 0's centroid. RAW already
   returns 0 for invalid elements, so we match that rather than inventing one.
```

### 8.3 Installing it (build time, once)

```
ALGORITHM 3 — Kernel construction with the projection enabled
───────────────────────────────────────────────────────────────────────────
AT STAGE [6/7], before JAX traces anything:

1.  define RAW = interpolate_velocity_single        # P1 / taylor / HCT / ...
2.  define divergence_single(pos, e, V):
        return trace( jacobian_of( <the name
                      'interpolate_velocity_single'> ) )
3.  define WRAPPER(pos, e, V, _raw = RAW):          # capture RAW as a DEFAULT
        u   = _raw(pos, e, V)
        Jac = jacobian_of(_raw) at pos
        s   = trace(Jac)/3
        return where(valid(e), u - s*(pos - centroids[e]), u)
4.  if divergence_free:
        require centroids is not None   else ERROR
        interpolate_velocity_single <- WRAPPER        # rebind the NAME
5.  trace & compile the vmapped RK4 using the (possibly rebound) name

⚠️ STEP 3, THE DEFAULT ARGUMENT, IS LOAD-BEARING. Python resolves a
   closed-over name at CALL time. If WRAPPER referred to
   'interpolate_velocity_single' from the enclosing scope, then after step 4
   it would call ITSELF -- infinite recursion. Confirmed experimentally
   (RecursionError) before writing the real version.

✅ STEP 2 NEEDS NO CHANGE and that is the point: it resolves the name at call
   time, so after step 4 it differentiates the CORRECTED field and returns 0.
   Velocity and divergence cannot drift apart.
```

### 8.4 Where it sits in the RK4 step

```
ALGORITHM 4 — One RK4 step for one particle (inside jax.vmap)
───────────────────────────────────────────────────────────────────────────
INPUT  pos, elem, dmg      (per-particle state)
       dt, velocity_field  (shared)

   # ---- stage k1 -------------------------------------------------------
1. elem1 <- SEARCH(pos, elem)                   # L0 cache / L1 hop / L2 global
2. vel1  <- VELOCITY(pos, elem1)                # ★ corrected here
3. div1  <- trace(jacobian VELOCITY at pos)     # ★ = 0 automatically
4. dmg   <- dmg + dt * weights * f(vel1, div1, ...)   # ln_J, lnPhi, C, t_res...
5. p1    <- pos + 0.5*dt*vel1

   # ---- stages k2, k3, k4 ----------------------------------------------
6. elem2 <- SEARCH(p1, elem1);  vel2 <- VELOCITY(p1, elem2)      # ★ corrected
7. p2    <- pos + 0.5*dt*vel2
8. elem3 <- SEARCH(p2, elem2);  vel3 <- VELOCITY(p2, elem3)      # ★ corrected
9. p3    <- pos + dt*vel3
10.elem4 <- SEARCH(p3, elem3);  vel4 <- VELOCITY(p3, elem4)      # ★ corrected

   # ---- combine ---------------------------------------------------------
11.pos'  <- pos + (dt/6)*(vel1 + 2*vel2 + 2*vel3 + vel4)
12.return pos', elem4, dmg

★ = the four places the projection acts. All inside the mapped function, so
    jax.vmap batches them over all particles with no change to its structure.
⚠️ The divergence is sampled at the k1 stage only (damage_order=1 default),
    which is why ln_J is a k1-weighted integral, not an RK4-weighted one. That
    is pre-existing behaviour and unchanged by this work.
```

### 8.5 The pipeline, end to end

```
ALGORITHM 5 — Full tracking run with --divergence-free
───────────────────────────────────────────────────────────────────────────
 [parse]   if divergence_free and damage_divergence != "analytic": ERROR
           # 'field' uses a precomputed nodal div_u from the RAW gradient and
           # would silently ignore the projection

 [1/7]     nodes, conn, vel_seq <- load_pvtu(case, 166 frames)
           nodes, conn, vel_seq <- deduplicate_nodes(...)     # piece seams

 [2/7]     M_inv, p0, volumes, aabb <- precompute_geometry(nodes, conn)

 [3/7]     octree <- build_or_load_cache(nodes, conn)
           upload: vel_seq, M_inv, p0, conn, levelset, temperature, octree
           if divergence_free and centroids not already uploaded:
               centroids <- mean(nodes[conn], axis=1)          # (n_el, 3)
               upload centroids                                # 0.13 GB, once

 [4/7]     particles <- seed(mode, bounds, n)

 [5/7]     elem[p] <- locate(particles[p])  for all p

 [6/7]     rk4_step <- build_kernel(..., divergence_free=True)   # ALGORITHM 3

 [7/7]     for step in 1..N_STEPS:
               frame <- step * dt / frame_dt        mod 166     # cyclic
               pos, elem, dmg <- vmap(rk4_single)(pos, elem, dmg)  # ALGORITHM 4
               if step % export_freq == 0: write_vtkhdf(pos, ...)

 [report]  ln_J  <- dmg[:, 5]  for ALIVE particles only
           detF  <- exp(ln_J)
           GATE: |detF - 1| median / p99 / max,  fraction > 1e-2
           ⚠️ if max == 0 exactly, report the DTYPE FLOOR, not "exact":
              float32 exp() rounds to 1.0 for |x| < 6e-8
```

---

## 9. Where each piece lives in the code

| file | role |
|---|---|
| `jaxtrace/damage/divfree_projection.py` | **Single source of truth** for the maths. `project_element_gradient_divfree()` (host-side, Algorithm 1) and `divergence_report()` (the gradient diagnostics). Also the unit-test oracle. |
| `benchmark_femuss_comparison.py` | The kernel. `divergence_free` kwarg; `velocity_divfree_single` (Algorithm 2) installed by rebinding at build time (Algorithm 3). |
| `run_tracking.py` | CLI `--divergence-free`; the parse-time guard; the centroid upload at stage [3/7]; the `det F` gate report with the dtype-floor disclosure. |
| `scripts/damage/test_divfree_kernel_equiv.py` | Pins Algorithm 2 against Algorithm 1 (5.4e-15 over 40 tets), with a sanity check that the test fields actually carry divergence so the test cannot be vacuous. |
| `scripts/damage/test_pathline_lnj.py` | The independent validator: pure numpy, no JAX, marches the same seeds through raw and corrected fields and compares `ln J` and trajectories. |
| `scripts/damage/test_divfree_reconstructions.py` | Pins §7.6: which reconstructions the projection is exact for, and that the leftover divergence equals `−(∇s·(x−x_c))`. Guards against silently unblocking `vertex_taylor` / `hct_*`. |
| `scripts/damage/test_divfree_levelset.py` | Pins §7.7: nothing is injected into the level-set-masked tool region, including an element straddling the interface. |
| `scripts/damage/test_closed_form_divfree.py` | Whole-snapshot verification across cases: divergence, preservation, gradient alignment. |
| `scripts/damage/test_local_order.py` | Proves the P1 data is exactly affine (model order is not the problem) and quantifies the irreducible repair cost. |
| `scripts/damage/test_divfree_detf.sh` | The two-arm `det F` gate on LUMI. Detects the PVTU stem from disk. |
| `docs/damage/DIVERGENCE_FREE_PLAN.md` | The chronological working log — every measurement, every refuted approach, §4a–§4ac. |
| ⚠️ `jaxtrace/damage/divfree.py`, `scripts/damage/build_divfree_field.py`, `scripts/damage/sbatch_divfree_sweep.sh` | **Superseded BDM₁ machinery.** Refuse to run without `ALLOW_SUPERSEDED=1`. Kept only because the diagnostics built on them are the evidence for §11. |

---

## 10. All the measurements

### 10.1 The projection, whole snapshots (CPU, float64)

| case | elements | `\|div u\|/‖A‖` | `‖u'−u_P1‖/‖u_P1‖` | `‖A'‖/‖A_P1‖` | `cos(A',A_P1)` | aligned | time |
|---|---|---|---|---|---|---|---|
| D2 | 10,807,677 | 2.98e-18 | 3.88e-08 | 0.99998 | 0.99998 | 99.74 % | 13.4 s |
| A2 | 10,335,240 | 2.56e-18 | 3.82e-08 | 0.99999 | 0.99999 | 99.85 % | 14.1 s |
| B1 | 7,462,783 | 4.45e-18 | 4.34e-08 | 0.99995 | 0.99995 | 97.92 % | 71.2 s |
| C1 | 7,586,466 | 4.06e-18 | 4.41e-08 | 0.99997 | 0.99997 | 99.87 % | 12.1 s |
| D1 | 7,922,599 | 4.72e-18 | 4.70e-08 | 0.99993 | 0.99993 | 98.10 % | 9.3 s |

Correctly scaled (`0.5 < ‖A'‖/‖A‖ < 2`): **100.000 %** in every case.

### 10.2 Pathlines (independent numpy integrator, 200 seeds × 300 steps)

| case | raw `\|ln J\|` p50 | corrected p50 | corrected p99 | improvement | crossings |
|---|---|---|---|---|---|
| D2 | 7.63e-04 | 1.66e-18 | 2.06e-17 | 4.6e+14× | 59 |
| C1 | 1.30e-02 | 2.98e-18 | 3.77e-17 | 4.4e+15× | 72 |
| B1 | 1.49e-01 | 5.55e-18 | 5.34e-17 | 2.7e+16× | 78 |

`|det F − 1|` exactly 0.0 in float64; 200/200 alive in both arms.

### 10.3 Production kernel, 200-step gate (LUMI job 23061561)

| arm | `ln J` median | `\|det F−1\|` median | p99 | max | `>1e-2` | Stage 7 |
|---|---|---|---|---|---|---|
| raw P1 | +3.3e-05 | 4.39e-04 | 1.91e-03 | 5.84e-03 | 0.00 % | 268.5 s |
| **divergence-free** | 0.00000 | **0.000e+00** | 0.000e+00 | 0.000e+00 | 0.00 % | 255.4 s |

Both arms 4,000 particles, 3,200 alive (80.0 %) — strictly comparable.

⚠️ **Three caveats on this run, stated plainly:**
1. The `df` zero is a **float32 dtype floor** (`|ln J| < 6e-8`), not exactness.
   It is meaningful because the raw arm's 4.39e-04 is **four orders above** the
   floor, so the metric demonstrably resolves the quantity.
2. The 5 % Stage 7 difference is **noise, not a speedup** — at 4,000 particles
   and ~1.3 s/step the run is kernel-launch-latency bound. What it does show is
   no catastrophic slowdown from the extra Jacobians.
3. 100 % of particles recorded `t_res = 0` (never reached the shear layer), and
   20 % were not found (coarse smoke seed grid). So this validated the
   **plumbing**, not the damage indicators.

### 10.4 Unit tests

| test | result |
|---|---|
| in-kernel vs host closed form, 40 tets × 8 points | 5.4e-15 |
| `tr(jacfwd(corrected))` | −1.1e-16 (vs raw 1.608) |
| `u'(x_c) = u(x_c)` | exactly 0.0 |
| `A` from nodal velocities + `M_inv` vs exact affine | 8.9e-16 |
| trace-free after projection, random matrices | 4.8e-16 |
| minimality (no trace-free candidate closer) | confirmed |

### 10.5 Memory

| array | per frame | 166 frames | note |
|---|---|---|---|
| velocity stack float32 | 23 MB | 3.83 GB | already uploaded |
| velocity stack float64 | 46 MB | 7.65 GB | the float64 gate arm |

⚠️ **What `--precision float64` actually costs, measured on arm 1 of the
full-length gate** (D2, 166 frames), against the float32 200-step run:

| stage | float32 | float64 | precision-bound? |
|---|---|---|---|
| [1/7] load 166 frames | ~78 s (1 frame) | **2098 s** | ✅ yes — 12.9 s/frame, host arrays double |
| [2/7] precompute geometry | 197.7 s | **202.7 s** | ❌ no — within noise |

So the cost of float64 is concentrated almost entirely in the **PVTU load**,
not the geometry work. An earlier estimate here assumed the precompute would
scale too; it does not.
| **centroids (our only addition)** | — | **0.13 GB** | time-independent |
| stored corrected gradients (rejected) | 389 MB | 64.6 GB | ❌ > 32.6 GB HBM |
| HCT Bernstein (for scale) | 10.4 GB | 1.72 TB | ❌ |

---

## 11. What we tried that did not work, and why

This section is longer than the method itself, deliberately. The failures are
what justify the simplicity of the answer, and two of them produced *plausible
numbers* before being caught.

### 11.1 The common cause

Every method taken from the literature assumes a **stable** mixed
discretisation. Ours is **stabilised P1/P1**, which is a different object. Each
paper's structural hypothesis is false on our data — and each was cheap to
check and expensive to assume:

| method | its premise | our field | verdict |
|---|---|---|---|
| Larson–Niklasson | face fluxes are discontinuous | exactly continuous (jump 0.0) | ❌ |
| Lederer–Schöberl | field is *discretely* divergence-free | 10–100 % residual | ❌ |
| Pazner | (1,1) block ≈ its diagonal | eigenvalues −2.07 … +8.11 | ❌ |
| RTN₀ element | — | rotation-blind (spans only constants ⊕ x) | ❌ |

> **Standing rule adopted from this:** before implementing a method from a
> paper, write the one test that measures its stated premise on our data, and
> run that first.

### 11.2 The big one: BDM₁ overlapping Schwarz

We built a full domain-decomposition solver: BDM₁ elements (12 DOF per
tetrahedron, `H(div)`-conforming), patches of ~10,000 elements grown by 3
face-adjacent layers, sparse LU per patch, 16 parallel processes, seam
analysis. ~1500 s per snapshot on 16 cores.

It appeared to work: divergence 9.2e-17, seam ratio 1.05×, 100 % coverage.

**It was producing garbage, and the metric we were watching could not see it.**

| quantity | shipped C1 field | should be |
|---|---|---|
| divergence | 9.2e-17 ✅ | ~0 |
| "preservation" `‖u_new−u_P1‖/‖u_P1‖` | 1.66e-01 | small |
| **`‖A_new‖/‖A_P1‖`** | **2.80e+06** ❌ | ~1 |
| **`cos(A_new, A_P1)`** | **0.0016** ❌ | ~1 |
| `\|b\|/\|A x_c\|` | 1.0000 ❌ | ≪ 1 |
| elements with `cos > 0.9` | **0.115 %** | ~100 % |

The gradient was a **million times too large and orthogonal** to the truth. The
field matched the centroid velocity only through **catastrophic cancellation**
between two enormous wrong terms — which is exactly what `|b|/|A x_c| = 1.0000`
means, a number we had measured in all 12 test patches and dismissed as
"universal, not a discriminator." It *was* universal. It was the disease.

#### Why the metric hid it

"Preservation" compares only `u(x_c)`. And `u(x_c) = A x_c + b` can be right
while `A` and `b` are each enormous and wrong and cancel. **The metric was
blind to the gradient by construction** — and the gradient is the thing
tracking integrates.

> **The lesson, now enforced in code:** never gate on a value-only metric. The
> `det F` report and `divergence_report()` both check `‖A'‖/‖A‖`,
> `cos(A', A_P1)` and the cancellation ratio.

#### Why it could not be fixed by tuning

Five hypotheses were measured and refuted (slow material, divergence
magnitude, `cond(T)`, Tikhonov weight, element size). Then:

- **Rank deficiency found**: `smin(N)` separated good patches (6.1e-02 … 7.2e-02)
  from catastrophic ones (1.0e-06, 1.9e-06) — a 65,000× gap. `N` maps face DOFs
  to centroid velocities, so a near-null vector is a field the objective
  **cannot see**, and the solve adds it for free.
- **Raising the regularisation made it worse**, not better: patch 0 went from
  1.21 to 1.99e+04. `eps` pulls toward the vertex-averaged P1 trace, which is
  not an admissible BDM₁ target — it trades an invisible mode for a wrong bias.
- **Strengthening the objective also failed**: 12 BDM₁ moments made `N`
  unisolvent (`smin` → a uniform 0.48, a 250,000× lift) and made preservation
  **1e5× worse**. Because matching the P1 field *throughout* an element while
  being exactly divergence-free is **inconsistent** — the P1 field has real
  divergence.

That last result is the key insight: **the near-null direction is not a defect,
it is the slack the solve needs.** The two requirements cannot both be met in
that discrete space. The BDM₁ approach was not salvageable by tuning.

### 11.3 Folding the correction into the nodal velocity

The tempting simplification: correct the nodal velocities once, leave every
downstream path untouched.

❌ **Impossible, and measured rather than assumed.** A node is shared by ~20
elements with *different* `tr A`, so no nodal field has a P1 gradient equal to
`A'` in every element. A global least-squares nodal fit on 1.53 M elements:

| | `\|div u\|/‖A‖` p50 |
|---|---|
| raw P1 | 1.03e-02 |
| **nodal least-squares** | **1.03e-01** (10× WORSE) |
| per-element closed form | 4.32e-18 |

and it changed the nodal velocity by **100 %** doing it.

### 11.4 Two hypotheses about the *algebra* that turned out to be wrong

Asked directly: are we fitting a low-order model to higher-order data, and are
we applying a uniform correction where some regions need finer adjustment?

- **Low order?** ❌ Refuted *by construction*. The FOM is P1, so on a tet its
  four nodal values define an **exactly affine** field (residual 1.4e-16), and
  any affine-complete element space contains it. No higher-order local model —
  SV-P3 included — can represent the FOM better than exactly.
- **Uniformity?** The non-uniformity is **real** (the required repair spans 43×
  from p50 to p99 against three global scalars) but ❌ **not the cause**:
  adaptive row weighting varying 1000× between elements changed nothing
  (`by_normA` matched uniform to 3 significant figures; `by_repair` was 10–200×
  worse).

Both were worth asking — the first retired SV-P3, and chasing the second is
what uncovered the gradient defect in §11.2.

### 11.5 Methods not pursued, and why that is now defensible

| option | status |
|---|---|
| **Scott–Vogelius P3** on barycentrically-refined mesh | Not needed. It would be pointwise divergence-free *and* conforming, but §11.4 shows the FOM data is exactly affine, so a cubic space adds nothing to represent, and §5 shows our correction is already the closest possible. Conformity is the only thing it adds, and §6 shows we do not need it. |
| **Global constrained projection** (PETSc/MUMPS fieldsplit) | Not needed, same reason: it could not beat the minimum-norm bound, and it would cost a global solve per snapshot. |
| **Accept the FOM and carry `ln J` as an error bar** | Superseded — we can simply remove the error instead. |

---

## 12. How to run it

### 12.1 Normal use

```bash
python3 run_tracking.py \
    --input /path/to/case.gid \
    --output /path/to/out \
    --mesh-subdir "" \
    --mesh-pattern "A2_{timestep}.pvtu" \
    --vel-range 34 199 \
    --n-steps 8000 --dt 0.0003629 \
    --damage --damage-driver models --damage-indicators \
    --damage-detachment \
    --divergence-free                 # <-- the whole feature
```

`--damage-divergence analytic` is the default and is required; it is enforced.

### 12.2 What you should see in the log

```
[divergence-free] element centroids uploaded (10807677, 3) float32 (0.13 GB)
    -- time-independent geometry, so this works at any frame count
...
  [damage] ln_J = int(div u dt)    : median +0.00000  ...
  [damage] det F = exp(ln_J)       : median 1.000000  ...
  [damage]   |det F - 1| : median 0.000e+00  p99 0.000e+00  max 0.000e+00
  [damage]   ⚠️  |det F - 1| is identically 0 in float32: this is the dtype
             FLOOR (|ln_J| < 6.0e-08), not an exact zero. Re-run with
             --precision float64 to resolve the true value.
```

### 12.3 Guards you may hit

| message | meaning |
|---|---|
| `--divergence-free requires --damage-divergence analytic` | `field` mode reads a precomputed nodal `div_u` from the **raw** gradient and would silently ignore the projection. |
| `divergence_free=True requires element_centroid_gpu` | Internal: the kernel was built without centroids. `run_tracking.py` uploads them automatically. |
| `--divergence-free is exact only for a reconstruction whose gradient is CONSTANT per element` | You combined it with `vertex_taylor` or an `hct_*` method, where ~1.2 % of the divergence would survive (§7.6). Use `centroid_taylor` or `--gradient-recovery 0`. |
| `REFUSING TO RUN` from `build_divfree_field.py` | You invoked the **superseded** BDM₁ builder. Use `--divergence-free` instead. |

### 12.4 Verifying it yourself

```bash
# 1. The maths: in-kernel form == host closed form (seconds, no GPU)
python3 scripts/damage/test_divfree_kernel_equiv.py

# 2. A whole snapshot: divergence, gradient alignment (minutes, CPU)
python3 scripts/damage/test_closed_form_divfree.py \
    --case <post dir> --pattern 'A2_{timestep}.pvtu' --timestep 34

# 3. Pathlines, independent of the production kernel (minutes, CPU)
python3 scripts/damage/test_pathline_lnj.py \
    --case <post dir> --pattern 'A2_{timestep}.pvtu' --timestep 34

# 4. The det F gate through the real kernel (GPU, LUMI)
bash scripts/damage/test_divfree_detf.sh          # both arms
```

---

## 13. Open items

1. **The full-length `det F` gate** — LUMI job 23062898, three arms (df/float64,
   raw/float64, raw/float32), 8,000 steps over 166 frames with production seed
   density. ⚠️ This is the number the method is finally judged on, because
   `ln J` is a path integral and shorter marches understate it (1.43e-05 at 50
   steps vs 3.17e-03 at 8,000 on the same configuration). Running at time of
   writing.
2. **Compute-bound throughput.** The 200-step gate was launch-latency bound, so
   the cost of three extra Jacobians per step is not yet measured. The
   full-length arms differ only in the flag, so their Stage 7 times will give it.
3. **The damage indicators themselves.** Every validation so far concerns the
   *kinematics* (`ln J`, `det F`, trajectories). Whether the corrected field
   changes the void predictions is a separate question, and needs a run long
   enough to reach the shear layer (`t_res > 0`).
4. **`ln J` uses the k1 stage only** (`damage_order=1`). That is pre-existing
   and unchanged here, but it means `ln J` is not an RK4-weighted integral. With
   a divergence-free field it hardly matters — every stage would contribute
   zero — but it is worth knowing.
5. **Other cases.** The projection is verified on A2, B1, C1, D1, D2. The
   remaining pin-shape cases are untested, though nothing in the method is
   case-specific.
6. ⚠️ **The octree cache key includes the node-position dtype**, so a
   `--precision float64` run MISSES a cache built at float32 and rebuilds
   (~20 min) and then stores a second ~570 MB entry for *the same geometry*.
   (Cost is modest — even with the miss, a three-arm gate is ~11.8 h against a
   24 h limit — so this is a storage point, not a scheduling one.)
   `compute_cache_key()` hashes `node_positions` content **and dtype**
   (`jaxtrace/gpu/search/octree_cache.py:144`), and the loader casts positions
   to the run's precision (`mesh_loader_timedep.py:327`). The octree is a
   purely geometric structure, so this is finer-grained than it needs to be.
   Found while projecting the timeline of the float64 gate arms; not a
   correctness problem, but it doubles cache storage per case whenever
   precision is varied. Worth keying on positions cast to a canonical dtype.

---

## Appendix A — Symbols

| symbol | meaning |
|---|---|
| `u(x)` | velocity field, m/s |
| `A` | velocity gradient, `A_ij = ∂u_i/∂x_j`, s⁻¹ |
| `b` | affine offset, so `u = Ax + b` |
| `x_c` | element centroid |
| `s` | `tr(A)/3`, one third of the divergence |
| `A'`, `b'` | the corrected gradient and offset |
| `I` | 3×3 identity |
| `‖·‖_F` | Frobenius norm, `√Σ A_ij²` |
| `∇·u` | divergence, `= tr A` for an affine field |
| `F` | deformation gradient |
| `det F` | volume ratio; `= 1` for incompressible material |
| `ln J` | `∫(∇·u)dt` along a pathline; `det F = exp(ln J)` |
| `N_a` | P1 shape function of node `a` |
| `M_inv` | inverse element matrix, gives shape-function gradients |
| `smin(N)` | smallest singular value (BDM₁ diagnostics only) |

## Appendix B — The derivation on one page

```
GIVEN  a tetrahedron with nodes x0..x3 and nodal velocities u0..u3.

1. The P1 interpolation is affine and exact:
        u(x) = A x + b,     from the 4x4 solve of  [x_a | 1] [A^T; b^T] = [u_a]

2. For an affine field, divergence is the trace:
        div u = du_x/dx + du_y/dy + du_z/dz = A11 + A22 + A33 = tr A

3. Split A uniquely into trace-free + isotropic:
        A = (A - (tr A/3) I)  +  (tr A/3) I
             \_____________/      \_______/
              shear+rotation       pure expansion  <- the unphysical part

4. Delete the isotropic part:
        A' = A - (tr A/3) I        =>  tr A' = tr A - 3(tr A/3) = 0   ✓

5. Restore the velocity at the centroid so particles are not displaced:
        A' x_c + b' = A x_c + b    =>  b' = b + (tr A/3) x_c          ✓

6. Equivalently, pointwise:
        u'(x) = u(x) - (tr A/3)(x - x_c)                               ✓

7. Optimality: minimise ||X - A||_F^2 s.t. tr X = 0.
        L = ||X-A||^2 - 2L tr X;   dL/dX = 0  =>  X = A + L I
        tr X = 0  =>  L = -(tr A)/3  =>  X = A - (tr A/3) I   = A'    ✓
   So A' is the ORTHOGONAL PROJECTION of A onto the trace-free subspace:
   no divergence-free field is closer to the FOM gradient.

8. The unavoidable change:
        ||A'-A||_F / ||A||_F = |div u| / (sqrt(3) ||A||_F)
   measured: 0.66 % typical, 28 % at p99 (D2).
```
