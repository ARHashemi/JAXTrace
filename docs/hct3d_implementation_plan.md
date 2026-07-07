# Full HCT-3D reconstruction — implementation plan

Draft plan for a C¹ piecewise-cubic Hermite velocity reconstruction on
tetrahedral meshes, to be added to the recovery pipeline as
`--recovery-method hct_cubic` alongside the existing `centroid_taylor`
and `vertex_taylor`.

The construction is the 3D analogue of the 2D Hsieh-Clough-Tocher
macro-element (Ciarlet 1978, §6.1) with the Alfeld split (Alfeld 1984)
plus the full C¹ vertex/face-normal-derivative constraints of
Worsey-Farin (1987).

**Status**: not yet implemented. This document is the roadmap; the
existing `vertex_taylor` remains the highest-accuracy method until the
phases below land.

---

## Why C¹, and why cubic

A P1 tet has:

* **Value** given by 4 nodal DOFs (linear interpolation).
* **Gradient** piecewise constant, with **jumps across all interior
  faces**. This is the discontinuity SPR smooths at nodes.

A raw cubic on a single parent tet (a "P3 element") has 20 DOFs per
component. It could interpolate

* 4 vertex values                    = 4
* 4 vertex gradients × 3 components  = 12
* 4 face centroids × value           = 4

= 20 constraints, uniquely determined. But two neighbouring P3 tets
sharing a face would need to match all four face-centroid values plus
the whole face polynomial — over-constrained. So a **single cubic per
parent tet cannot be C⁰ across parent faces** unless we throw away
information we need.

The macro-element fix: **split each parent tet into K sub-tets**, put
a smaller cubic on each sub-tet, and use the extra DOFs freed by the
split to satisfy C¹ continuity at sub-tet interfaces.

---

## The Alfeld split

Split parent tet T with vertices `v0, v1, v2, v3` and centroid
`vc = (v0+v1+v2+v3)/4` into **4 sub-tets** T₀..T₃, each formed by
replacing one parent vertex with the centroid:

    T_i = (vc, v_j, v_k, v_l)       for i ∉ {j,k,l}

Each parent face `F_i = (v_j, v_k, v_l)` is a face of exactly one
sub-tet. Each of the 4 interior sub-tet faces is a triangle
`(vc, v_a, v_b)` shared by two sub-tets.

Alfeld's split is the minimal-vertex refinement that admits a global
C¹ piecewise cubic. Worsey and Farin proved the reduced-parameter
system yielding uniqueness.

---

## Step-by-step recipe

### Phase 1 — Precompute geometry

Per parent tet:

* `vc` — centroid
* `Fc_i` — centroid of parent face i (i = 0..3)
* `Ec_ij` — midpoint of parent edge (i,j)
* Sub-tet vertex tables: which four points make up each of the 4
  sub-tets.

Wall-clock: a few ms per 100k elements, one-shot.

Store as a per-element dataclass carried alongside the recovery data.

### Phase 2 — Enforce DOF layout on the reduced macro-element

Following Worsey-Farin §3, the C¹ piecewise cubic has the following
DOFs on the parent tet:

| Label            | Location         | Type            | Count |
|------------------|------------------|-----------------|-------|
| Vertex value     | v0..v3           | value           | 4     |
| Vertex gradient  | v0..v3           | grad · 3 comps  | 12    |
| Edge-midpoint    | midpoints of 6 edges | normal deriv   | 6     |

Total = **22 DOFs per component** for the whole macro-element (not
per sub-tet). Each of the 4 sub-tets has 20 Bernstein DOFs = 80 DOFs,
but the C⁰ + C¹ constraints across the 4 interior faces (6 per face =
24 total) plus the 4 face-centroid interior conditions collapse this
to 22 free parameters.

The 22 DOFs are the ones our pipeline naturally exposes:

* **Vertex values** and **vertex gradients** come from the existing
  Steps 1-4 SPR pipeline.
* **Edge-midpoint normal derivatives**: for each parent edge, project
  the average of the two endpoint SPR gradients onto the edge-normal
  plane, take the midpoint value. **This is the new piece** we need
  to compute and store.

### Phase 3 — Solve for Bernstein coefficients

Standard approach: express each sub-tet's cubic in Bernstein-Bézier
form on its 4 vertices. A cubic on a tet has 20 B-coefficients per
component: 4 vertex + 12 edge (2 per edge × 6 edges) + 4 face
centroid + 0 interior.

Structure of the solve on the Alfeld-split macro-element (per parent
tet, per velocity component):

1. **Parent-vertex B-coefficients (`c_{3e_i}`)**: come directly from
   the loaded nodal velocity component. 4 knowns.
2. **Centroid vertex B-coefficient (`c_{3000}` at vc)**: the value
   `u(vc)` at the parent centroid. 1 unknown (call it `μ`).
3. **Parent-edge B-coefficients** (2 per edge × 6 parent edges = 12):
   from the 1D cubic Hermite along-edge formula using vertex value
   and vertex gradient at the endpoint. Known.
4. **Spoke-edge B-coefficients** (2 per edge × 4 spoke edges from vc
   = 8, but 4 of them ARE the endpoint vc so they equal `μ`; the
   other 4 involve `∇u(vc) =: γ`, 3 unknowns).
5. **Parent-face centroid B-coefficients** (`c_{0111}` on the outer
   triangle of each sub-tet, 4 of them): from Farin's
   quadratic-precision formula in terms of parent-vertex and
   parent-edge B-coefficients. Known.
6. **Interior-face centroid B-coefficients** (`c_{0111}` on each
   vc-containing triangle, 6 shared across the 4 sub-tets): from
   Farin's formula in terms of the spoke-edge and parent-vertex/
   parent-edge B-coefficients plus the edge-midpoint gradient DOF
   from Phase 2.
7. **C¹ continuity across the 4 interior sub-tet faces**: 4
   equations per component linking the outer and interior face
   B-coefficients. Together with the 4 unknowns (μ, γx, γy, γz)
   this closes the system.

Following Farin (2002, §17.3) the whole system reduces to a **4×4
linear solve per parent tet per component**. Precompute all knowns
first, then solve for (μ, γ) once per component. Wall-clock
estimate: ~50 μs per element in NumPy. For a 900k-tet mesh, ~45 s.
Fine for a one-shot precompute.

**Development sequencing.** Because the full C¹-with-macro-element
construction has enough moving parts to be a bug-collection risk,
Phase 3 lands in three sub-commits:

* **Commit 2a** *(DONE, 4abeee5)*: cubic reconstruction with C⁰
  interior continuity. Face-centroid B-coefficient uses P1 vertex
  average (linear-precision only). (μ, γ) computed via P1 average
  rules. Testable properties: linear exactness everywhere, C⁰
  continuity at spoke and parent-face edges.
* **Commit 2a-plus** *(DONE, this commit)*: quadratic-precision
  face-centroid coefficient via Farin's formula
  `c_{111}^face = -1/6·Σ_vertex + 1/4·Σ_edge` on the face's 3 vertex
  and 6 edge B-coefficients. All other coefficients unchanged from
  2a. Testable properties: linear exactness still holds; quadratic
  exactness at parent face centroids added. Verified numerically
  against the reference NumPy `bernstein_cubic_evaluate` at 200
  face-centroid samples with `u = (x², xy, y² + 0.3 z²)`.
* **Commit 2b** *(DONE, this commit, Taylor-fit variant)*: instead of
  the full Worsey-Farin C¹ 4×4 solve (which requires closed-form
  cross-face constraints that are non-trivial to derive from
  scratch), we ship a lightweight batched **16×10 least-squares
  Taylor-fit** per element per component. For each parent tet, fit
  a local degree-2 Taylor expansion
  `u(x) ≈ μ + γ·(x-vc) + ½(x-vc)^T H (x-vc)`
  against the 16 constraints: 4 parent-vertex values + 12
  parent-vertex gradient components. Solve via
  `jnp.linalg.solve(A^T A, A^T b)` batched across elements.
  Then use `μ` and `γ` in the standard Phase 3a-plus Bernstein
  assembly pipeline. Testable properties gained:
  - Bit-exact reconstruction for **quadratic velocity fields** at
    parent centroid `vc`, spoke-edge midpoints, parent face
    centroids, and parent vertices.
  - All Phase 3a and Phase 3a-plus properties preserved.
  This is not full C¹ (interior sub-tet face conditions are not
  imposed) but delivers 5.4× lower mean error than Phase 3a and
  3× lower than Phase 3a-plus on the recirc_2026 field with
  minimal added compute.

**Storage**: `(n_elements, 4_sub_tets, 20_bern_coeffs, 3_comps)` in
float32 = **864 MB VRAM for 900k tets**. Uploads once, never touched
again after build_recovery.

### Phase 4 — Kernel-side evaluation

Per query at position `p`:

1. **Locate the containing sub-tet** (4 candidates per parent). Each
   parent face F_i separates the parent into two half-spaces via a
   plane containing (vc, v_a, v_b). Compute signed volumes of
   (p, vc, v_j, v_k), (p, vc, v_k, v_l), (p, vc, v_l, v_j) with the
   parent face F_i; the point is in sub-tet T_i iff all three
   signed volumes have the same sign as the reference. Fast:
   4 vec cross products + 4 scalar signs. Total 4 branches, JAX-
   friendly (implement via jnp.where cascade or bit-packed index).

2. **Compute barycentric coordinates in the sub-tet** using the
   direct-inverse formula we already have (compute the sub-tet's
   Jacobian inverse on the fly from vc + 3 sub-tet vertices).

3. **Evaluate the Bernstein cubic** using de Casteljau or direct
   Bernstein basis:

       v(p) = sum_{i+j+k+l = 3, i,j,k,l >= 0}
              C_{ijkl} * B^3_{ijkl}(bary)

   where `B^3_{ijkl}(b) = (3! / (i! j! k! l!)) * b_0^i b_1^j b_2^k b_3^l`.

   Direct evaluation is 20 multiply-adds per component; de Casteljau
   is 3 rounds of vertex-averaging, slightly cheaper.

Wall-clock estimate: with n_elements = 900k and float32 gather-heavy
kernel, expect ~4-6x the vertex_taylor per-step cost. Still << search
+ RK4 substage overhead for the recirc_2026 case.

### Phase 5 — Validation gates

Correctness tests, in order:

1. **Linear field**: recovery must be bit-exact at nodes AND anywhere
   inside each element. Vertex values match; vertex gradients match;
   edge-normal derivatives match; Bernstein coefficients degenerate
   to the linear-interpolation solution.

2. **Quadratic field** (e.g. `u = x² + xy`): reconstruction should be
   exact everywhere (the cubic can represent quadratics). This is
   the strongest correctness check.

3. **Cubic field** (e.g. `u = x³ + y²z`): reconstruction should be
   exact everywhere.

4. **C¹ continuity at random face samples**: pick 100 points on
   sub-tet interfaces; evaluate the cubic from both sides; check
   value AND gradient (in the face-normal direction) match to
   float32 noise.

5. **Recirc field comparison**: repeat the mean/max err test we
   already have for centroid_taylor and vertex_taylor, at 200 random
   interior samples. Expected: hct_cubic should beat vertex_taylor
   by another ~3-5× on this field.

### Phase 6 — Fallback

At parent tets where the DOF system is degenerate (near-singular
Jacobian, extreme aspect ratios, near-colinear vertices), the
Bernstein solve returns NaN or huge coefficients. **Fall back to
vertex_taylor** for those elements — same call signature, cheap. Log
the fraction of degenerate elements at build time.

---

## Complexity budget

### Wall-clock estimates (900k-tet recirc_2026 4-lvl mesh)

Measured on the host CPU (JAX cpu backend). GPU (CUDA) will be
substantially faster still.

| Phase | Estimate | Actual (measured after implementation) |
|---|---|---|
| Phase 1 (geometry) | ~5 s | **~50 ms** (NumPy, vectorised) |
| Phase 2 (edge DOFs) | ~15 s | **~10 ms** (NumPy, vectorised) |
| Phase 3a (Bernstein C⁰, JAX steady) | — | **~1 s** |
| Phase 3a-plus (quadratic face upgrade, JAX steady) | — | **~1.05 s** (adds ~50 ms to Phase 3a) |
| Phase 3b (Taylor-fit LS + Bernstein assembly, JAX steady) | — | **~1.5 s** (adds ~0.5 s for the 16x10 LS batch) |
| SPR nodal gradient | — | ~5 s |
| Total precompute (all above) | ~65 s | **<10 s** |
| Per-step overhead vs raw P1 | ~4-6× on interpolation, ~1.3× on total step | tbd (Phase 4) |

### CPU (NumPy) vs GPU (JAX) — where each wins here

The design started with the intuition that all precomputes should be
NumPy (one-shot, so JIT overhead not amortized), and only the RK4
kernel's per-query hot path should be JAX. Measurement changed that:

* **Phase 1 (geometry)** and **Phase 2 (edge DOFs)** are trivial
  vectorised element-wise NumPy ops that finish in tens of
  milliseconds. Migrating them to JAX would add ~1 s of JIT-compile
  overhead per unique input shape for no benefit. **Keep NumPy.**
* **Phase 3a (Bernstein assembly)** naïvely has 4 sub-tet × 6 edge ×
  20 lookup passes — a lot of Python-side bookkeeping around a
  batched einsum. In NumPy this takes ~5.6 s on 900k tets. Migrated
  to JAX with static index tables folded into the JIT'd graph, it
  runs in ~1 s steady-state (~5× faster). **Use JAX.**
* **SPR nodal gradient recovery** has a per-node Python for-loop over
  variable-size patches, each running a small `np.linalg.lstsq`.
  Migrating to JAX requires solving the variable-patch-size problem
  (pad-and-mask or size-grouping), each of which needs its own test
  coverage. Current wall time on 68k nodes is 1.8 s and it's not
  the bottleneck. **Leave NumPy** and revisit only if
  end-to-end profiling on the workstation shows it dominates.
* **Phase 4 (RK4 kernel evaluator)** — inherently JAX; runs inside
  the JIT'd `interpolate_velocity_single`, called per particle per
  substage per step. **JAX all the way.**

### VRAM budget (32 GB total, 58 GB host RAM)

Per-element coefficient storage `(n_elements, 4_sub_tets,
20_bern_coeffs, 3_comps)` at float32:

| Mesh | Tets | VRAM |
|---|---|---|
| uniform (recirc_2026) | 98 k | 94 MB |
| 2lvl (recirc_2026) | 200 k | 192 MB |
| 4lvl (recirc_2026) | 900 k | 864 MB |

The 4lvl case fits comfortably in the workstation's 32 GB VRAM: the
existing mesh + search octree + velocity + particles use ~3 GB, and
JAX-XLA compilation of the RK4 kernel reserves 4-8 GB of transient
working memory. That leaves ~21 GB free for the HCT-3D tables.
No mitigations needed at this mesh size; a size-cap fallback (per
Phase 6) is a defensive measure for future 10⁷-tet meshes.

### JAX-XLA compile-memory risk

A meaty risk with a cubic-Bernstein kernel is that the JIT'd RK4
step's XLA-HLO graph balloons in size and the compile phase itself
runs out of GPU memory during autotuning. Empirically the current
vertex_taylor kernel compiles in ~4 GB working set; a naïve
Bernstein-cubic kernel expressed as 20 separate `where + matmul`
statements can easily push the graph to 8-12 GB compile working set
and trip OOM on smaller GPUs.

Mitigations we will apply from the start:

1. **Static Bernstein basis lookup**: precompute the 4×20 Bernstein
   basis-value tensor at each of 27 canonical barycentric samples
   (used for autotuning), stored as a module-level `jnp.array`.
   The kernel reads coefficient tables + basis values and does a
   single `einsum` per query — one HLO node, not 20.
2. **Sub-tet detection via a scan**, not 4 branches: a single
   `fori_loop` over 4 candidate sub-tets returns the winning
   sub-tet index and its barycentric coordinates in one pass.
   Avoids the 2⁴ = 16-way branch fan-out an XLA compiler generates
   from nested `jnp.where`.
3. **Coefficients gathered as one contiguous read**: shape
   `(n_elements, 4, 20, 3)`; per query, one `jnp.take` yields
   the (4, 20, 3) slab, then sub-tet index picks the (20, 3) matrix,
   then `einsum` folds against the basis vector. Two gather stages
   maximum.
4. **Float32 throughout**, no mixed precision: keeps the HLO
   type-inference tables small.

If the compiled kernel still OOMs at build time on the 4lvl mesh, we
can fall back to a **compile-time chunk** strategy (batch particles
in the RK4 loop instead of vmap'ing over all 360k at once). This is a
known JAX pattern and has been used elsewhere in the codebase.

## Development effort

| Task | Time |
|---|---|
| Phase 1 (geometry precompute) | 0.5 day |
| Phase 2 (edge-midpoint gradient DOFs) | 0.5 day |
| Phase 3 (Bernstein-coefficient solve) — the meaty piece | **2 days** |
| Phase 4 (kernel-side sub-tet detection + Bernstein eval in JAX) | **1.5 days** |
| Phase 5 (unit tests: linear, quadratic, cubic, C¹ continuity) | 1 day |
| Phase 6 (degenerate-element fallback) | 0.5 day |
| Documentation + case-config update | 0.5 day |
| **Total** | **~6-8 days** |

If Phase 3 uses the Worsey-Farin closed-form expressions (which
requires careful reading of the paper's Appendix), that number is
achievable. If we end up building a generic constrained-LS solver as
a fallback, add 2-3 days.

---

## Sequencing options

Given the total is ~1-2 weeks of focused work, we can:

* **Option 1**: do all of it in one commit series over ~2 weeks.
  Pro: one coherent PR to review. Con: nothing lands until it's all
  done; can't A/B-test progress against vertex_taylor.
* **Option 2**: land as 4 commits corresponding to Phases 1+2, 3,
  4+5, and 6. Each intermediate commit produces a working "partial"
  method (e.g. after Phase 3, `--recovery-method hct_cubic` is
  available but only correctness-tested on toy meshes; the recirc
  case still uses vertex_taylor by default).
* **Option 3**: land the Alfeld C⁰ variant first as
  `--recovery-method alfeld_cubic` (the "Option A" from the previous
  question — a partial C⁰ subset of Phase 3 with no C¹ constraints,
  ~1-2 days), get feedback on trajectory accuracy, then decide
  whether the extra weeks for full C¹ are worth it.

## Open questions

1. **Do we care about C¹ continuity in practice?** For particle
   tracking, an RK4 substage that crosses a sub-tet face at C⁰
   sees a slope discontinuity in the velocity but not the position.
   The RK4 error incurred is O(dt²) per substage where dt is the
   sub-tet transit time. At CFL < 1 this is typically dominated by
   the RK4 truncation error of the step itself. So **C⁰ + smooth
   nodal gradients may already give ~90% of the accuracy of full C¹**
   — measurable but not spectacular.

2. **Can we use fewer sub-tets?** The classical alternative is the
   Powell-Sabin split (24 sub-tets in 3D), which admits QUADRATIC C¹
   continuity with fewer DOFs. But 24 sub-tets is 6× more VRAM and
   kernel overhead. Alfeld's 4-sub-tet cubic is the sweet spot.

3. **Do we need to handle refined-mesh transitions specially?** In
   the recirc_2026 case, refined-mesh cells have hanging nodes at
   level boundaries. The HCT cubic on the coarse side and the four
   HCT cubics on the fine side won't naturally agree at the T-junction
   face. This is a **known limitation**; particles crossing level
   boundaries will see the same C⁰ discontinuity there as they do
   with raw P1. Not fixable without a full conforming refinement.

---

## References

* Alfeld, P. (1984). *A trivariate Clough-Tocher scheme for
  tetrahedral data.* Computer Aided Geometric Design, 1(2), 169–181.
* Worsey, A. J., & Farin, G. (1987). *An n-dimensional Clough-Tocher
  interpolant.* Constructive Approximation, 3(1), 99–110.
* Farin, G. (2002). *Curves and Surfaces for CAGD: A Practical
  Guide.* 5th ed., Morgan Kaufmann. Chapters 17-18 cover
  multivariate Bernstein-Bezier basics.
* Ciarlet, P. G. (1978). *The Finite Element Method for Elliptic
  Problems.* North-Holland. §6.1 for the 2D HCT construction we
  are extending.
