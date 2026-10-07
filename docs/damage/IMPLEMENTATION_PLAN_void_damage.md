# Implementation Roadmap — Pathline-Integrated Void Damage in JAXTrace

**Reference document. Written against the code as of 2026-09-23.**

Goal: carry damage state variables along GPU-tracked pathlines and deposit them to a
grid, producing a continuous void-susceptibility field — **without adding any per-step
host↔device transfer** and without degrading throughput.

Physics background: `PRIMER_fsw_physics_for_me.md` §7–§9.

⚠️ **This document is executed end to end, Phase 0 → 5, on ONE data
configuration at a time.** The four configurations — FOM mesh, ROM mesh, FOM
grid, ROM grid — are the outer loop: complete this plan on FOM mesh, validate,
write the report, and only then repeat the *same finished method* on the next
configuration. See `STAGED_VALIDATION_PLAN.md` § "How this work is organised".

**Current position — configuration A (FOM mesh):** Phase 0 is complete. Phases 2
and 3 were partly built ahead of order as pointwise field diagnostics (that is
where the 62 case-results come from), but **Phase 1 — carrying damage state along
a pathline — has not been built, and phases 3–4 cannot be completed without it.**
Phase 1 is the next thing to do.

---

## 0. The performance question, answered up front

**The current loop is already fully asynchronous and stays on device.** Verified by
reading the code:

- [`run_tracking.py:2671`](../JAXTrace/run_tracking.py#L2671) — the marching loop is a
  Python `for`, but every operand is a device array and `rk4_step` is `@jax.jit`. JAX
  dispatches asynchronously; the host never blocks.
- [`run_tracking.py:2720`](../JAXTrace/run_tracking.py#L2720) — the **only** host transfer
  is gated behind `do_log or do_export or do_density`, i.e. every `LOG_INTERVAL` /
  `EXPORT_FREQUENCY` steps, never per step.
- [`rk4_fully_fused_timedep.py:628`](../JAXTrace/jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py#L628)
  — one `@jax.jit` wrapping one `jax.vmap` over all particles.

**Therefore the rule for this work is simply: do not break what is already correct.**

Three invariants to preserve:

> **I1.** Damage state lives in device arrays that are passed in and returned by the step
> function, exactly like `positions_gpu` / `element_ids_gpu`. Never materialised on host
> inside the loop.
>
> **I2.** No new `.item()`, `float()`, `int()`, `np.array()`, `np.sum()` or
> `block_until_ready()` on a damage array inside the loop. Any of these forces a
> synchronisation and serialises the pipeline.
>
> **I3.** No Python-level `if` on a traced damage value. Use `jnp.where`. (The codebase
> already follows this — see the `use_skip_step_on_fail` block at
> [`rk4_fully_fused_timedep.py:707`](../JAXTrace/jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py#L707),
> which uses `jnp.where`, not branching. Note the project's known issue with nested
> `lax.cond` under `vmap` — stay with `jnp.where`.)

**The cheap-by-construction argument.** The damage update is:

- **pointwise** — no neighbour coupling, no scatter, no communication;
- **~20 flops per particle per step** (two ODE updates, one `exp`);
- **already-paid memory traffic** — the nodal scalars it needs are read at the same
  `connectivity[elem_id]` indices the velocity interpolation already touches, so the
  gather is served from cache.

The RK4 step costs **five L0/L1/L2 element searches** plus four barycentric solves. That
is hundreds of flops and several irregular memory gathers. Adding ~20 flops is expected
to be **<2% overhead** — and Phase 1 measures this rather than assuming it.

---

## Phase 0 — Prerequisites and go/no-go checks

Do these before writing integration code. Each can kill or reshape the plan.

### 0.1 ✅ RESOLVED — pressure is present in the FOM

**Verified 2026-09-23** by reading `cylindrical_119.pvtu` directly: `Pressure` is a
Float64 point array, range −7.40e7 … +6.44e7 Pa, on all 180,461 nodes. Present at all
120 timesteps in the 20-case cohort, and at ts=159 in `cylA.gid`.

See `STAGED_VALIDATION_PLAN.md` §0 for the full field inventory and three ⚠️ findings
that change this plan (velocity naming, pressure sign, steady state).

Whether the *mesh2grid projection* carries it is an **outer-loop C** question and
does not block anything here: the method is built on FOM mesh data throughout.

### 0.2 Fallback, only relevant once the outer loop starts

Pressure exists in the FOM, so the whole inner loop is unblocked. These fallbacks
matter only at **outer-loop B**, if the ROM cannot reconstruct pressure well enough:

1. **Build our own pressure POD** over the 20 cases — preferred, mirrors
   `build_our_pod_basis.py`.
2. **Use the colleague's pressure modes** (4 retained) — exists, but verify against the
   known shipped-basis mismatch.
3. **Strain-rate-only surrogate** — drop the $\exp(c_1\sigma_m/\kappa)$ factor and
   accumulate $\bar\varepsilon$ alone. Not a damage model, but a legitimate field, and
   Phase 1 still validates the machinery.

⚠️ O15 makes this more pressing than it looks: the metric is far more sensitive to
pressure than to velocity, so the pressure-accuracy target for outer-loop B must be
*measured*, not assumed.

### 0.3 ⚠️ Sign and orientation — two checks, both cheap, both silently fatal

**The naive version of this check was wrong.** An earlier draft asserted the stir zone is
uniformly compressive and told you to `assert p.mean() > 0`. **The data disproves that:**
pressure is roughly symmetric about zero (60 % positive, median +1.3 MPa), because it is
a deviatoric/incompressible-solve pressure, not an absolute one including the forge load.

✅ *Resolved:* `Displacement` **is** velocity in m/s — confirmed by the user, no conversion.

Both are now resolved (staged plan §0.3b) — kept here as the checks to repeat
whenever a new case family is added:

| # | Question | How to settle it |
|---|---|---|
| 1 | Which angular sector is the advancing side? | from `INLET_VELOCITY` / `PIN_RPM` sign |
| 2 | Is $\sigma_m = +P$ or $-P$? | pick the sign making material **ahead of** the tool compressive |

Check 2 is decidable from the data: the pressure field shows a clean dipole
(−17.6 MPa at [−90°,−45°) vs +9.0 MPa at [0°,+45°)), so one sector is being forged and the
other released. Choose the sign convention that puts compression ahead of the tool, and
**record the reasoning in the code**.

### 0.4 Read the prior art

Before claiming novelty (see primer §8b):

- **Fraser et al. 2018**, `10.3390/met8020101` — GPU SPH + defect metric + parameter
  optimisation. Closest prior art to this entire plan.
- **Cao et al. 2021**, `10.1016/j.jcp.2021.110863` — ML + ROM of an FSW model.

### 0.5 Validation data inventory

Without ground truth the output is an unfalsifiable picture. Establish now:

- Any CT scans or macrographs for welds whose $(\omega, v_{\text{adv}})$ we simulated?
- If none: can we get even **one** defective/sound pair? One pair beats zero.

---

## Phase 1 — Minimal viable damage (strain accumulator)

**Goal:** prove state can be carried with no measurable performance loss, before any
physics complexity. One scalar, no new fields, no new inputs.

### 1.1 Extend the step signature

In `_build_rk4_step`, add one input and one output:

```python
@jax.jit
def rk4_step(
    positions_gpu, element_ids_gpu, dt, velocity_fields_gpu, time_idx,
    last_valid_velocities=None,
    ebar_gpu=None,          # NEW: (n_particles,) accumulated strain
):
```

Inside `rk4_single_particle`, add `ebar` as a vmapped argument and return the updated
value. Follow the existing `last_vel` pattern exactly
([`rk4_fully_fused_timedep.py:641`](../JAXTrace/jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py#L641)) —
it is already a carried-state argument with a dummy-array fallback, so the plumbing
is proven.

### 1.2 Gate it behind a flag

Mirror `use_levelset_mask` / `use_extended_domain`: a module-level boolean closed over at
build time, so when damage is off **the traced graph is byte-identical to today's**. This
guarantees no regression for existing runs and makes A/B benchmarking exact.

```python
use_damage = damage_config is not None
```

### 1.3 The accumulation point

Update **once per step, using the k1 stage quantities**, not per RK4 sub-stage:

```python
# after vel_k1 / elem_k1 are known
if use_damage:
    edot_k1 = sample_scalar(pos, elem_k1, edot_field)   # see Phase 2
    ebar_new = ebar + dt * edot_k1
    # freeze accumulation on skipped steps, mirroring position handling
    if use_skip_step_on_fail:
        ebar_new = jnp.where(any_failed, ebar, ebar_new)
    if use_extended_domain:
        ebar_new = jnp.where(already_exited, ebar, ebar_new)
```

**Why k1 and not a full RK4 of the damage ODE:** damage is a slowly-varying accumulator
driven by a field that is itself interpolated. Fourth-order accuracy on it is meaningless
when the driving field carries interpolation error. First-order (explicit Euler at k1)
costs one sample instead of four. Revisit only if Phase 4 shows time-step sensitivity.

### 1.4 Benchmark gate

```bash
# A/B on an existing case, damage off vs on
python run_tracking.py <existing args>                    # baseline
python run_tracking.py <existing args> --damage-model strain
```

**Pass criterion: throughput loss < 3%.** If it exceeds that, stop and profile before
adding physics — something is wrong with the plumbing, not the flops.

Also verify: results with `--damage-model none` are **bitwise identical** to the
pre-change binary.

---

## Phase 2 — Derived nodal fields (the key design decision)

### 2.1 Precompute on the mesh, sample like velocity

$\dot\varepsilon_{\text{eff}}$, $\sigma_m$, $\sigma_{\text{eq}}$ are **functions of the
field, not of the particle**. So compute them **once per case, per timestep, on the mesh
nodes** — never per particle per step.

```
FOM/ROM fields  ──►  ∇u, D, eigenvalues  ──►  nodal scalars  ──►  GPU  ──►  sampled
   (u, p, T)          (once, on mesh)         (n_nodes,)       (upload)    (per step)
```

⚠️ **Keep the time dimension even though current cases are steady.** Threaded tools with
time-dependent fields are planned, so mirror the velocity contract exactly: store
`(n_timesteps, n_nodes)` and index with `time_idx % n_timesteps`. A steady case is
`n_timesteps == 1` and the modulo makes every step read index 0 — no branch, no separate
path (staged plan §0.4c).

Three nodal scalar arrays, each `(n_timesteps, n_nodes)` float32:

| Array | Content | From |
|---|---|---|
| `edot_nodal` | $\dot\varepsilon_{\text{eff}} = \sqrt{\tfrac23 D_{ij}D_{ij}}$ | $\nabla u$ |
| `sigm_nodal` | $\sigma_m = -p$ | pressure |
| `sigeq_nodal` | $\sigma_{\text{eq}} = 3\mu_{\text{eff}}\dot\varepsilon_{\text{eff}}$ | $\mu_{\text{eff}}$ or Sellars–Tegart |

Store $\eta = \sigma_m/\sigma_{\text{eq}}$ **precomputed** rather than dividing per
particle — one division per node instead of per particle per step, and the masking
(§2.3) is applied once at source.

### 2.2 Sampling: reuse the barycentric weights already computed

**This is where the performance win is.** `interpolate_velocity_single`
([`rk4_fully_fused_timedep.py:498`](../JAXTrace/jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py#L498))
already computes `b0, b1, b2, b3` and `nodes_idx`. The level-set mask at line 553 is the
exact template:

```python
node_ls = levelset_gpu[nodes_idx]                      # (4,)
ls_val  = b0*node_ls[0] + b1*node_ls[1] + b2*node_ls[2] + b3*node_ls[3]
```

**Refactor `interpolate_velocity_single` to optionally return `(b0..b3, nodes_idx)`**, or
add a `damage_scalars` argument returning the interpolated triple alongside velocity.
Then sampling three scalars costs:

- **0 extra element searches** (the expensive part — reused)
- **0 extra barycentric solves** (reused)
- 3 gathers of 4 floats at indices already in cache
- 12 multiply-adds

⚠️ **Do not** write a separate `sample_scalar(pos, elem, field)` that recomputes
barycentric coordinates. That would roughly double the interpolation cost. The whole
design rests on reusing the weights.

### 2.3 Masking at source

$\eta$ divides by $\sigma_{\text{eq}}$, which → 0 away from the tool. Mask **when building
the nodal array**, not per particle:

```python
active = edot_nodal > EDOT_FLOOR          # e.g. 0.01 s^-1
eta_nodal = np.where(active, sigm_nodal / np.maximum(sigeq_nodal, SIGEQ_FLOOR), 0.0)
```

Zero $\eta$ gives $\exp(0)=1$, a neutral multiplier — safe. This keeps the GPU kernel
branch-free.

### 2.4 Memory budget — scales with the number of timesteps

Keep the layout `(n_timesteps, n_nodes)` float32 per scalar, indexed by
`time_idx % n_timesteps` exactly as velocity is (staged plan §0.4c). A steady case is
`n_timesteps == 1`; no special-casing.

| Case | per scalar | three scalars |
|---|---|---|
| **Steady** (`n_ts = 1`, current cohort) | 0.72 MB | **2.2 MB** — negligible |
| Transient (`n_ts = 120`, threaded tools) | 86 MB | 260 MB |

At 120 timesteps this is the same order as `velocity_sequence_gpu` itself (3 × 86 MB),
so it roughly **doubles** field memory. Not a concern now, but check headroom before the
first transient run; if tight:

- store `eta_nodal` and `edot_nodal` only (drop intermediates) — 172 MB
- or float16 for $\eta$ (it is $O(1)$ and feeds an exponential — precision is not critical)

---

## Phase 3 — The damage models

Implement **two independent models** and run them together. Agreement between two
different laws is the cheapest available evidence; disagreement is diagnostic.

### 3.1 Rice–Tracey (no calibration)

$$\frac{d\ln\Phi}{dt} = 0.849\,\dot\varepsilon_{\text{eff}}\exp\!\left(\tfrac32\eta\right)$$

⚠️ **Integrate $\ln\Phi$, not $\Phi$.** The ODE is multiplicative (primer §8, factor C),
so log-space makes it linear in the accumulator, removes stiffness over thousands of
steps, and keeps $\Phi>0$ by construction.

```python
logphi_new = logphi + dt * 0.849 * edot * jnp.exp(1.5 * eta)
```

`jnp.exp` on `(n_particles,)` is one fast transcendental — negligible beside five element
searches.

⚠️ **Overflow guard.** If $\eta$ is large and positive from a bad pressure field,
$\exp$ can blow up. Clamp: `eta = jnp.clip(eta, ETA_MIN, ETA_MAX)` with e.g. $\pm 3$.
Cheap insurance.

### 3.2 Cockcroft–Latham (one calibrated constant)

$$\frac{dC}{dt} = \frac{\max(\sigma_1,0)}{\sigma_{\text{eq}}}\,\dot\varepsilon_{\text{eff}}$$

```python
C_new = C + dt * jnp.maximum(s1_ratio, 0.0) * edot
```

Needs a fourth nodal scalar `s1_ratio_nodal` = $\sigma_1/\sigma_{\text{eq}}$, from the
largest eigenvalue of $\sigma_{ij}$ (primer §7.6 — use the invariant form, no eigensolve).

**Why this one suits FSW:** the Macaulay bracket switches damage off under compression
automatically, so it discriminates wake-vs-shoulder for free.

### 3.3 Not now: Lee–Dawson and GTN

**Lee–Dawson** (He et al.) needs the coupled $\kappa$ ODE — a second carried state plus a
Voce saturation law with ~10 material constants, and it *clamps at compression* so it
cannot show closure. **GTN** needs 7+ constants and strictly a porous-plasticity solve.

Defer both until Phase 4 shows the cheap models separate cases. If Phase 4 fails, adding
parameters will not rescue it.

### 3.4 State layout

Carry a single `(n_particles, 3)` float32 array rather than three separate `(n,)` arrays —
one argument, one output, contiguous, better coalescing:

```python
damage_state[:, 0] = ebar      # accumulated strain
damage_state[:, 1] = logphi    # log void fraction (Rice-Tracey)
damage_state[:, 2] = C         # Cockcroft-Latham integral
```

Cost: 12 bytes/particle. At 360k particles: **4.3 MB**. Negligible.

---

## Phase 4 — Deposition, validation, and the honest test

### 4.1 Deposit to a grid

Reuse the existing density/union machinery — the per-particle scalars deposit exactly as
particle density already does. Produce, per case:

- `mean_logphi` / `max_logphi` over the union
- `mean_C`, `max_C`
- `mean_ebar`

Export alongside the existing `particles_union_density.vtkhdf` fields.

### 4.2 The three questions, in order

**Q1 — Does it reproduce the known spatial signature?**
Damage should concentrate on the **advancing side**, **behind** the tool. This is
established across the whole literature (primer §6) and He et al. reproduced it. If our
field does not show it, something is wrong — most likely the pressure sign (§0.3).

*This is the single most important check and it needs no experimental data.*

**Q2 — Does it separate defective from sound cases?**
Across the 20-case cohort, does peak/mean damage rank the cases the way observed defects
do? Even without CT, if some cases are known defective, a monotonic ranking is real signal.

**Q3 — Does the ROM reduce it?**
Only after Q1 and Q2. Add the damage field as a ROM target over $(v_{\text{adv}}, \omega)$,
same pipeline as density. Expect it to be *harder* than density (it is a history integral,
so it inherits pathline sensitivity — compare the density-vs-particles gap already
documented: 20.6% vs 41.4%).

### 4.3 Kill criteria — decide these now

State them before seeing results:

| Observation | Conclusion |
|---|---|
| Damage field is spatially flat | signal too weak — stop, report negative |
| Concentrates on the retreating side | sign error or field problem — debug, do not publish |
| No monotonic relation to known defective cases | not predictive — stop or rethink target |
| Rice–Tracey and Cockcroft–Latham strongly disagree | one is misconfigured — diagnose before proceeding |

**He et al.'s porosity changed 0.01% → 0.017%.** That is a ~70% relative change on a tiny
absolute number. A real possibility is that the signal is too weak to be useful — which is
one plausible reading of why nobody followed the method in 18 years (primer §8b).
Phase 4 exists to find that out cheaply.

---

## Phase 5 — If it works

- Lode-dependent closure branch (genuinely unattempted in FSW — primer §8b)
- Time-dependent rather than steady fields. He et al. used steady; the filling mechanism
  is *per-revolution*, so a steady field may average away the cause. **Our time-dependent
  tracking is a real differentiator here** — worth testing explicitly.
- Second-stage ROM → instant spatially-resolved process window
- Compare against Fraser's SPH particle-deficiency metric as an independent measure

---

## Appendix A — Performance checklist

Run through this before every benchmark:

- [ ] No `.item()`, `float()`, `int()`, `np.array()` on a damage array inside the loop
- [ ] No `block_until_ready()` except at the deliberate sync points
- [ ] Damage arrays are `jnp` device arrays, created once before the loop
- [ ] `damage_state` passed in and returned — not closed over, not global
- [ ] No Python `if` on traced values — `jnp.where` only
- [ ] No `lax.cond` under `vmap` (known compilation artifacts in this codebase)
- [ ] Damage off ⇒ traced graph identical to baseline (verify with `jax.make_jaxpr`)
- [ ] `nodal` arrays uploaded once before the loop, indexed by `time_idx` inside
- [ ] Barycentric weights reused, not recomputed
- [ ] float32 throughout (match `config.FLOAT_DTYPE_JNP`)

## Appendix B — Diagnosing a slowdown

If throughput drops more than ~3%:

1. **Check for a hidden sync.** Time the loop with and without the damage output being
   read. If reading it is what costs, the transfer is the problem, not the compute.
2. **Check the jaxpr.** `jax.make_jaxpr(rk4_step)(...)` — look for unexpected
   `convert_element_type` (dtype promotion to float64) or a recomputed barycentric block.
3. **Check register pressure.** The kernel is already large. Three extra live scalars
   across a long function could spill. If so, compute damage in a *separate* small
   `@jax.jit` kernel taking `(pos, elem_id, damage_state)` — still no host transfer,
   still one dispatch, but its own register budget. This is the main fallback.
4. **Check memory bandwidth.** If the nodal arrays do not fit alongside
   `velocity_sequence_gpu`, you will thrash. Drop to float16 for $\eta$.

## Appendix C — Files likely to change

| File | Change |
|---|---|
| `jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py` | carried damage state; reuse barycentric weights |
| `jaxtrace/gpu/tracking/mesh_data_gpu.py` | upload nodal damage scalars |
| `jaxtrace/gpu/mesh_loader_timedep.py` | load pressure/temperature alongside velocity |
| **new** `jaxtrace/damage/fields.py` | build nodal $\dot\varepsilon$, $\eta$, $\sigma_1/\sigma_{\text{eq}}$ from $u,p,T$ |
| **new** `jaxtrace/damage/models.py` | Rice–Tracey, Cockcroft–Latham update rules |
| `run_tracking.py` | `--damage-model` flag; thread state through the loop |
| `jaxtrace/density/union.py` | deposit damage scalars |

**Order of work:** 0.1 → 0.3 → 1.x (benchmark gate) → 2.x → 3.x → 4.x.
Do not skip the Phase 1 benchmark gate; it is what protects the performance property.
