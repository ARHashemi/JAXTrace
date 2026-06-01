# Density Estimation: Bottlenecks, Optimizations, and Why We Reverted

This document explains, in order, every performance bottleneck we
identified in the density-estimation pipeline, what we did about each
one, why the first round of optimizations had to be reverted, and what
remains on the plan. It is meant to be read top-to-bottom by anyone
inheriting this work — you should come away knowing which bottlenecks
are real, which are not, and why.

The companion mathematics-reference is in
[`docs/density_estimation.md`](density_estimation.md). This document
assumes the algorithm is understood; here we only talk about cost.

---

## 0. Quick orientation

The density estimator computes, per timestep,

```
ρ(q_m) = Σ_i K(‖q_m − x_i‖, h_i) · w_i        (then optionally /N for "pdf")
```

over `M` voxel-grid centres and `N` particle positions. The dominant
cost per step is the `O(M·N)` outer pair sweep — for the current case
that is `2.1M voxels × 360k particles ≈ 7.5 × 10¹¹` distance/kernel
evaluations per timestep.

The pipeline runs that sweep entirely on the GPU through JAX/XLA. The
particle trajectory is read from disk in a background thread; the
density slabs are written to disk in another background thread; the
main thread only orchestrates.

Everything below is about where time goes inside that pipeline and how
to reduce it.

---

## 1. The bottlenecks, ranked by impact

We enumerated ten candidate bottlenecks. Listed here roughly by impact
size, with which are **real**, **myths**, and **mitigated**.

| # | Candidate | Status | Impact | Notes |
|---|---|---|---|---|
| 1 | Per-particle kernel sum (`N×N`) dominates over grid pass (`N×M`) | **Real** | ~40 % of GPU time | Folds into octree work — becomes ~free after #4 |
| 2 | The `(M, N, 3)` "diff tensor" wastes HBM | **Myth** (see §3) | none | XLA fuses it away on its own |
| 3 | `r² = ‖a‖² + ‖b‖² − 2 a·b` can hit tensor cores | **Myth at our workload** | none | Workload is bandwidth-bound, tensor cores idle |
| 4 | Octree backend mis-sizes hash cells; falls back to brute | **Real (the big one)** | up to **~150× theoretical** on grid pass | The proper fix is the focus of §4 below |
| 5 | Stencil candidate buffer truncates at 256 → silent under-count | **Real**, paired with #4 | correctness, not just speed | Replace with fori_loop accumulator |
| 6 | Per-particle eval done as a second brute pass | **Real**, paired with #4 | once #4 lands, this is ~free | Re-use the same particle hash |
| 7 | Hybrid 2D-spatial + 1D-brute for thin-axis geometries | **Conditional** | ~3× on top of #4 in thin domains | Only matters when one bbox axis collapses to ≤ 1 cell |
| 8 | Accumulator update fires four separate kernels | Real but tiny | < 1 % | Trivial fuse-into-one-jit |
| 9 | `pad_particles` allocates per step | Real but tiny | < 1 % | Cache the padded buffers |
| 10 | Spatial query bucketing (Morton-sort queries) | Real but small | 1.3–1.7× on brute paths | Helpful only if brute is still in play after #4 |

#2 and #3 were the first round of optimization we attempted. They
turned out not to be bottlenecks at all — explained in §3.

#4 is the real prize. The full plan is in §4.

---

## 2. What we observed in production

Two end-to-end production runs (cylindrical_009, N = 360 000 particles,
M = 128³ = 2 097 152 voxels, brute backend, full 2000 timesteps, on
the workstation NVIDIA GPU):

| Run | Steady-state per-step | Total | Notes |
|---|---|---|---|
| **Baseline (naive brute)** | 2.88 s/step | 5798 s (~96 min) | XLA-fused single kernel |
| **Tiled + matmul (#2+#3)** | 3.61 s/step | 7223 s (~120 min) | **Reverted** — see §3 |

The 25 % slowdown of the "optimization" was the smoking gun.

These numbers depend on **N · M only**, not on the kernel or the
particle distribution (for compactly-supported kernels at this h, every
pair is computed regardless of whether it contributes). That is what
makes the brute backend a fixed cost: as long as both N and M are
held, the per-step time is locked.

The corollary: **shrinking N · M is the only way to actually go faster
on this workload**. That is exactly what the octree is supposed to do.

---

## 3. The first round of optimizations and why we reverted

We attempted Strategy C #2 ("tile the particle dimension") and #3 ("use
the matmul identity for distance") together in commit `0b5605f`,
followed by `97c11ac` for kernel-tolerance bookkeeping. Production
runtime got *worse*.

### What we thought was happening

The naive brute kernel is:

```python
diff = Q[:, None, :] - P[None, :, :]    # (M_chunk, N, 3) — looks like 34 GB
r    = sqrt(sum(diff*diff, -1))
K    = kernel(r, h, d)
rho  = sum(K * w, -1)
```

Eyeballing this you see a `(M_chunk × N × 3)` intermediate tensor that
for `M_chunk=8192, N=360k` is `~34 GB`. Naturally one assumes that
tensor is materialised in HBM, streamed back in, and the bandwidth on
that streaming dominates. So you tile `N` into smaller blocks, keep
the working set in L2, and use `r² = ‖Q‖² + ‖P‖² − 2 Q·P` to route
distance through GEMM so tensor cores get to do work.

### What was actually happening

XLA's fusion pass on a modern NVIDIA GPU rewrites the *entire*
expression `Q[:,None,:] - P[None,:,:] → ... → sum(K*w, -1)` into a
**single fused streaming kernel** that reads each `(Q_i, P_j)` pair
once, computes the kernel value into a register, and accumulates into
`rho_i` directly. The "34 GB intermediate" is a Python-level analysis
fiction; on-device it never exists.

So the naive layout was already optimal: one fused launch per chunk,
streaming HBM once.

### What our "optimization" actually did

`jax.lax.scan` over `N / particle_tile = 360k / 4096 ≈ 88` tiles is
**not a fused kernel**. Each scan iteration is a separate node in the
XLA compute graph that cannot be fused with the next one (the
accumulator carries state across iterations, which forces a
synchronisation). So we traded:

- **1 fused launch streaming HBM once**,

for

- **88 sequential launches**, each reloading its tile of `P` from
  HBM, and the matmul trick adding an extra `jnp.maximum(r², 0)`
  elementwise op per iteration that the naive layout did not have.

This workload is **HBM-bandwidth-bound**, not compute-bound. Tensor
cores were idle the entire time the matmul ran because we were
waiting for memory, not arithmetic. The extra elementwise op simply
added bandwidth pressure. The dispatch overhead from 88 sequential
launches added more on top.

Net effect: 25 % slower. Reverted in commit `7386b33`.

### The general lesson

> When an existing kernel is a single expression that XLA can fuse
> end-to-end, manual restructuring almost always pessimizes. The right
> intervention is **algorithmic** (reducing operations performed), not
> **memory-layout** (rearranging the same operations).

This is the principle that pushes us straight to the octree work in §4.

---

## 4. The proper octree cell-sizing fix (Strategy C #4 + #5)

This is the real opportunity. Unlike #2/#3, the octree backend
**reduces FLOPs**: it skips pairs whose distance exceeds the kernel
support, which in our geometry is ~97 % of all pairs.

The current octree code is in [estimator.py](../jaxtrace/density/estimator.py)
under `_build_particle_hash` and `_make_octree_kernel`. It has two
real bugs and one heuristic that needs replacement.

### 4.1 The current implementation, briefly

The octree builds a uniform 3D hash over the particle cloud:

```python
cell_size = max(extent / cells_per_dim, support_radius)
```

where `support_radius = SUPPORT × h_max`. This guarantees one hash
cell is at least as wide as the kernel's reach, so a 3×3×3 stencil
around the query cell is sufficient to cover all neighbours that could
possibly contribute.

For each grid query, the kernel:

1. Locates the query's hash cell.
2. Visits the 27 cells in the 3×3×3 stencil.
3. Collects up to `max_neighbors = 256` particles into a fixed-size
   candidate buffer (in stencil order).
4. Computes the kernel sum over the candidates.

That last step is the bug.

### 4.2 Bug #1: Mis-sized cells force enormous occupancy

In our case the geometry is anisotropic:

```
extent           = (0.073, 0.030, 0.0045) m
support_radius   = 2 × h = 2 × 4.5 mm     = 0.009 m
cells_per_dim    = 64

extent / 64     = (0.00114, 0.00046, 7e-5) m    — much smaller than support
max(..., 0.009) = (0.009,   0.009,   0.009) m   — clipped up everywhere
dims            = (9, 4, 1) hash cells          — only 36 cells total
particles/cell  = 360 000 / 36 ≈ 10 000
```

So the hash has **only 36 cells covering 360 000 particles**. Each
cell averages 10 000 particles. The kernel then collects at most 256
of them.

**This is not a small bias.** It means each grid query sees a
near-arbitrary 256-particle subset of its ~270 000 neighbours,
biased toward whichever cell in the 3×3×3 stencil comes first in
the iteration order. The density value at that voxel is roughly
`256/270000 ≈ 0.1 %` of the correct value, and the spatial pattern
is dominated by the stencil-traversal order rather than the actual
particle distribution. That is exactly the visual artifact that
caused the earlier "thin curtain along y_min" failure mode.

### 4.3 Bug #2: This is not strictly a grid problem

The bug arises from the *interaction* of:

- The particle cloud's geometry (anisotropic, thin in z).
- The kernel's bandwidth (`h = 4.5 mm`, support `2h = 9 mm`).
- The cell-size heuristic (`max(extent/N, support)`).

For an isotropic, well-spread cloud (e.g. a unit cube uniformly seeded),
the math gives `dims = 64×64×64 = 262 144 cells` with ~1 particle each
and the bug doesn't trigger. The 256-buffer is more than enough.

For our weld-pool geometry, where z is **130× thinner** than x, the
cell size is forced up to `support` along *every* axis (because z
extent / 64 < support, and the heuristic clamps), collapsing the hash
to 36 cells. That is the trigger.

So the bug is **partly** "anisotropic domain" and **partly** "the cell
heuristic was wrong for any anisotropic geometry". A round cubic
domain would hide the bug; ours exposes it.

### 4.4 Bug #3: The candidate-buffer is a hard truncation

Even if we fix the cell sizing, capping the per-query candidate count
at 256 is a silent failure mode. Any future configuration where a
single hash cell happens to contain > 256 particles drops the
remainder. This breaks the "exact within float32" guarantee the brute
backend gives.

The fix is to **never collect into a fixed buffer**. Instead, do a
running kernel-sum accumulation over the per-cell particle list
in-place. JAX's `fori_loop` is the right primitive — it doesn't need
to know `n_in_cell` at trace time, only at runtime, and accumulates a
scalar `rho` correctly regardless of cell occupancy.

### 4.5 The fix (Strategy C #4 + #5 combined)

Three coordinated changes:

**Change A — Replace the cell-size heuristic.**
Drop the `max(..., support_radius)` clamp. Instead, target a known
average occupancy and size the cell to match:

```
target_n_per_cell ≈ max_neighbors / stencil_volume       (typically 256/27 ≈ 9.5)
cs_target ≈ (V_bbox × target_n_per_cell / N) ** (1/3)    (isotropic in *physical* units)
```

For our case (`V_bbox ≈ 9.9e-6 m³`, `N = 360 000`):

```
cs_target ≈ (9.9e-6 × 9.5 / 360 000) ** (1/3) ≈ 6.4e-4 m
```

This gives `dims = (114, 47, 7)` cells, ~37 000 cells with ~9
particles each on average — well under any reasonable buffer cap.

**Change B — Make the stencil per-axis adaptive.**
With `cs < support` along some axes, the 3×3×3 stencil no longer
covers the full kernel reach. Compute the needed stencil radius per
axis:

```
stencil_radius[a] = ceil(support_radius / cs[a])
stencil_volume   = ∏ (2 stencil_radius[a] + 1)
```

For our case with `cs ≈ 6.4e-4 m` isotropic and `support = 9 mm`:

```
stencil_radius = (15, 15, 15)            — 31×31×31 = 29 791 cells
```

That seems wild but it isn't: each cell holds ~9 particles, total
candidates per query ≈ 270 000, identical to the true neighbour
count — which is what we want. Iteration cost is unchanged from "scan
all 270k neighbours" but **we only scan the ones that actually have
particles**, not all 360 000. For sparser regions of the bbox the
savings are larger.

**Change C — Replace the candidate buffer with a fori_loop sum.**
Per cell visited, do an inner fori_loop over the particles in that
cell and accumulate the kernel contribution directly into `rho`. No
buffer, no cap, no silent truncation:

```python
def visit_cell(carry, cell_id):
    s = cell_starts[cell_id]
    e = cell_starts[cell_id + 1]

    def acc_one(j, partial):
        pid = sorted_idx[s + j]
        r   = jnp.linalg.norm(q - P[pid])
        return partial + kernel(r, h[pid], d) * w[pid]

    return jax.lax.fori_loop(0, e - s, acc_one, carry), None

rho, _ = jax.lax.scan(visit_cell, jnp.float32(0.0), stencil_cell_ids)
return rho
```

The `scan` and `fori_loop` are XLA loop primitives that compile into
a single device kernel, so we don't pay the dispatch overhead that
killed #2.

**Engine-selector update.** Add a runtime check that refuses to use
the octree when it can't actually help — specifically when
`dims_min < 3` (degenerate axis can't even hold a 3-cell stencil),
or when `expected_neighbours > 0.5 × N` (cloud is too dense for the
octree's saving to overcome its construction cost):

```python
expected_neighbours = N × support_volume / V_bbox
cost_octree         ≈ M × expected_neighbours
cost_brute          ≈ M × N

use_octree iff (cost_octree < 0.1 × cost_brute) and (dims_min ≥ 3)
```

### 4.6 Expected speedup, end-to-end

For our case:

| Path | Work per step | Time per step (proj.) |
|---|---|---|
| Current brute (baseline) | `M × N = 7.5 × 10¹¹` | 2.88 s |
| Octree (post-fix) on grid | `M × n_avg ≈ 2.1M × 270k = 5.7 × 10¹¹`* | ~2.2 s |
| Octree (post-fix) when avg is realistic** | `M × 30k = 6.3 × 10¹⁰` | **~0.2–0.4 s** |
| Including #6 (per-particle reuses hash) | + `N × 30k = 1.1 × 10¹⁰` | **~0.3–0.5 s** |

(*) Worst case at this anisotropy with a generous bandwidth: the
support ball happens to contain almost everything, so the octree
doesn't save much. We may want to revisit the bandwidth choice.

(**) Average over the run, accounting for the fact that most voxels
are at the edge of the cloud, not in the dense centre. The
~30 000-neighbour estimate is what the brute pass actually convolves
against in non-central voxels.

So the realistic per-step speedup is **6–15×**, taking total runtime
from ~96 minutes to ~6–15 minutes for the full 2000-step run, with
**no precision compromise** (the fori_loop sum is bit-identical to
brute when the stencil covers the full kernel support).

### 4.7 Verification plan

The plan that survives is the one from before, slightly adapted:

1. Implement Changes A + B + C as one commit on `feature/density`.
2. Re-add a correctness test (`tests/test_octree_vs_brute.py`) that
   runs both backends on a small `N=10k, M=32³` test and asserts
   `max | ρ_octree - ρ_brute | < 1e-5 × max | ρ_brute |`.
3. Run the workstation production case (`STEP_TAIL=20`,
   `RESOLUTION=64`) with `ENGINE=octree` and compare timing + the
   marginal sums (x, y, z) against the known-good baseline.
4. Only if both correctness and speedup check out: run full
   `STEP_TAIL=2000` at `RESOLUTION=128` and compare against the
   2.88 s/step brute baseline.

If anything goes sideways, the brute backend remains the safe default
(it's the `engine=brute` path used today), so a revert costs only
the octree commit.

---

## 5. So is the bottleneck "the mesh" or "the kernel" or what?

This is what motivated the question, so here is an explicit answer:

- **It is not the mesh.** The mesh-aligned octree is only used to build
  the inside-mesh voxel mask (currently disabled for this case because
  the velocity mesh is a moving-tool snapshot). It is not used during
  the density computation itself.

- **It is not the kernel function.** All six kernels in
  [`kernels.py`](../jaxtrace/density/kernels.py) are evaluated per
  particle-pair at the same cost (`~10 FLOPs/pair`). The bottleneck
  doesn't care which one you use.

- **It is partly the bandwidth.** With `h = 4.5 mm` and `support = 9
  mm` against a 30 mm × 30 mm cloud, ~30 % of all particles are
  in-support of each central voxel — that is what makes the octree
  win less dramatically here than it would for a tighter bandwidth.

- **It is partly the grid resolution.** `M = 128³ = 2.1 M` queries is
  large. Halving it to `64³ = 262k` would already give 8× speedup
  with no algorithmic change, at the cost of spatial resolution.

- **It is mostly the algorithm choice (brute vs. octree).** Brute is
  `O(M × N)`; octree (when correct) is `O(M × n_avg)` where `n_avg`
  is the average per-query in-support count. At our geometry,
  `n_avg / N ≈ 0.08`, so the upper-bound speedup is ~12×. The bug in
  the current octree is the reason we are stuck on brute.

The cleanest mental model: **brute force is doing 10–15× the work it
needs to**. The octree exists exactly to skip that work. Once it
works correctly, the bottleneck moves on to whatever the next-largest
thing is — likely the per-particle eval (#6, mitigated by reusing the
hash), then the actual write throughput (which we already de-bottlenecked
with the background-thread writer + gzip compression).

---

## 6. What is in the repo right now (state on `feature/density`)

| Commit | What it does |
|---|---|
| `fb766aa` | Density implementation, brute + buggy octree |
| `0b5605f` | Strategy #2+#3 (tiled + matmul brute) — **reverted by `7386b33`** |
| `97c11ac` | Tolerance bookkeeping for #2+#3 — **reverted by `7386b33`** |
| `7386b33` | Revert #2+#3, return to the XLA-fused naive brute kernel |

Current state: same naive brute as `fb766aa`, baseline ~2.88 s/step.

Octree code is present but the auto-selector keeps it disabled until
the cell-sizing fix lands.

---

## 7. Next concrete step

Implement Strategy C #4 + #5 as a single commit on `feature/density`,
verify against brute on a small `N=10k, M=32³` test, then production-
benchmark the 20-step case. If the projected ~6–15× speedup holds,
proceed to #6 (per-particle reuses the same hash for ~free).

If something unexpected comes out of the small-N verification, the
plan branches back to either reworking the cell-size heuristic
(#7 hybrid) or running the brute backend at lower grid resolution
for production while we iterate.
