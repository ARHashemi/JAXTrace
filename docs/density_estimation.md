# JAXTrace — Density Estimation: Mathematics & Implementation

This document is a concise reference for the density-field estimation
pipeline introduced under [`jaxtrace.density`](../jaxtrace/density/). It
covers the kernel SPH/KDE estimator, bandwidth strategies, voxel grid,
backends, normalization, time accumulation, and on-disk layout.

The implementation lives in:

- [`kernels.py`](../jaxtrace/density/kernels.py) — kernel functions
- [`bandwidth.py`](../jaxtrace/density/bandwidth.py) — bandwidth resolution
- [`grid.py`](../jaxtrace/density/grid.py) — voxel grid + I/O helpers
- [`inside_mesh.py`](../jaxtrace/density/inside_mesh.py) — voxel masking
- [`estimator.py`](../jaxtrace/density/estimator.py) — brute / octree backends
- [`time_accumulator.py`](../jaxtrace/density/time_accumulator.py) — time-average
- [`writers.py`](../jaxtrace/density/writers.py) — VTI / VTKHDF output
- [`runner.py`](../jaxtrace/density/runner.py) — orchestrator

---

## 1. The main algorithm

For each time step `t_k` and a particle cloud `{x_i ∈ R³}` with optional
per-particle weights `w_i ≥ 0` and smoothing lengths `h_i > 0`, we
estimate a continuous scalar field `ρ(x, t_k)` evaluated at the centres
of a uniform voxel grid `{q_m ∈ R³}`:

```
ρ(q_m, t_k) = (1/W) · Σ_i  w_i · K(‖q_m − x_i‖, h_i; d)
```

where `K(r, h; d)` is a kernel function (compactly-supported in most
cases) and `W` is a normalization that depends on the chosen mode (§5).
The same kernel is also evaluated *at the particle positions themselves*
(`q_m ↦ x_j`) to produce a per-particle `density` scalar that flows
through to the existing trajectory export.

The pipeline is fully GPU-resident under JAX. Each per-step call is one
JIT-compiled kernel that fuses the broadcast difference, distance,
kernel evaluation, and weighted reduction into one device launch (brute
backend) or one launch per chunk of queries (octree backend).

---

## 2. Kernel functions

All kernels share the signature `K(r, h, d)` with `r` and `h`
broadcasting against each other; `d ∈ {2, 3}` is the spatial
dimensionality. All are normalized so that
`∫_{R^d} K(‖x‖, h) dx = 1`. Each kernel has a compact-support radius
expressed in units of `h` (`SUPPORT * h`); contributions at `r ≥
SUPPORT * h` are exactly zero.

Let `q := r/h`.

### Gaussian
```
K(r, h, d) = (2π h²)^(−d/2) · exp(−q² / 2)
```
SUPPORT = 3 (numerical truncation; mathematically infinite).
Smooth, infinitely differentiable; classical KDE kernel.

### Cubic spline (M₄ B-spline)
```
                  σ_d · [1 − 1.5 q² + 0.75 q³],     0 ≤ q < 1
K(r, h, 3) =      σ_d · 0.25 · (2 − q)³,           1 ≤ q < 2
                  0,                               q ≥ 2
```
SUPPORT = 2. Coefficients `σ_2 = 10/(7π h²)`, `σ_3 = 1/(π h³)`.

### Wendland C² (default)
```
K(r, h, 3) = σ_d · (1 − q/2)⁴ · (1 + 2q),   0 ≤ q < 2
```
SUPPORT = 2.  `σ_2 = 7/(4π h²)`, `σ_3 = 21/(16π h³)`.
Compactly-supported, C² continuous, less prone to particle pairing than
the cubic spline. The current default for both online and offline
pipelines.

### Wendland C⁴
```
K(r, h, 3) = σ_d · (1 − q/2)⁶ · (1 + 3q + (35/12)q²),   0 ≤ q < 2
```
SUPPORT = 2. `σ_2 = 9/(4π h²)`, `σ_3 = 495/(256π h³)`.
Smoother than C²; useful for fields where second derivatives matter.

### Epanechnikov
```
K(r, h, d) = σ_d · max(0, 1 − q²)
```
SUPPORT = 1. `σ_2 = 2/(π h²)`, `σ_3 = 15/(8π h³)`.
Theoretically AMISE-optimal in KDE; cheapest to evaluate.

### Quintic spline (M₆ B-spline)
```
K(r, h, 3) = σ_d · [(3 − q)⁵ − 6 (2 − q)₊⁵ + 15 (1 − q)₊⁵]
```
SUPPORT = 3. `σ_2 = 7/(478π h²)`, `σ_3 = 1/(120π h³)`.
Highest-order spline currently provided; smoother fields, larger reach.

### Kernel reach in physical units

For a given `h` the kernel reach in physical space is `support × h`.
With anisotropic voxel grids the reach is *isotropic in physical
space*, not in voxel units. On strongly anisotropic grids
(e.g. `dz << dx`), the reach measured in voxels along z is much larger
than in x — this is the correct behaviour and matches what the
underlying density represents.

---

## 3. Bandwidth resolution

Modes are exposed via `DensityRunnerConfig.bandwidth_mode` and resolved
in [`bandwidth.resolve_bandwidth`](../jaxtrace/density/bandwidth.py).
They all return a `(N,)` JAX array of per-particle smoothing lengths.

### `fixed`
```
h_i = bandwidth                         if explicitly set
    = bandwidth_factor · vs             otherwise, with vs := max(voxel spacing)
```
Same `h` for all particles. `vs = max(spacing)` (not min) ensures the
kernel spans multiple voxels in every direction on anisotropic grids;
using `min(spacing)` produces near-empty fields on thin axes.

### Scott's rule
```
h = σ · n^(−1/(d+4))
```
where `σ` is the mean per-dimension standard deviation of the particle
cloud and `n = N`. Returned as a constant `(N,)` vector.

### Silverman's rule
```
h = (4/(d+2))^(1/(d+4)) · σ · n^(−1/(d+4))
```
Mild correction over Scott for Gaussian-like distributions. Both
Scott/Silverman are *AMISE-optimal under the assumption that the
underlying distribution is Gaussian*; treat as a reasonable
data-adaptive default, not as ground truth.

### `knn_adaptive`
```
h_i = knn_safety · dist(i, k-th nearest neighbour),     k = knn_k
```
Per-particle bandwidth scaled to the local crowding. Implemented as a
brute-force JAX `top_k` over chunks of `knn_chunk` queries — fully on
the GPU. Cost is `O(N · knn_chunk)` per chunk; cheap up to ~50 k
particles, gets expensive beyond. (A particle-octree path for k-NN is
on the wishlist.)

### Refresh cadence

`bandwidth_refresh_every = N` recomputes `h` every N steps. `0` (default)
computes once and caches.

---

## 4. Voxel grid

The voxel grid is uniform, axis-aligned, defined by:

- bounding box `(bbox_min, bbox_max) ∈ R³ × R³`
- per-axis resolution `(Nx, Ny, Nz)` or an absolute `voxel_size`

Spacing:
```
spacing[a] = (bbox_max[a] − bbox_min[a]) / N_a
origin[a]  = bbox_min[a] + 0.5 · spacing[a]              (first voxel center)
```

Voxel centres are arranged as `(Nx, Ny, Nz)` with `indexing="ij"` and
flattened in C order, giving a flat `(M, 3)` device array of size
`M = Nx · Ny · Nz`.

### Bounds resolution (`bounds_mode`)

- `explicit` — caller supplies `bounds` directly.
- `mesh` — use the velocity mesh's deduplicated-node bbox (host code).
- `particles` — initial particle positions bbox (online mode only).
- `prepass` (offline) — two-pass scan of the trajectory `particles.vtkhdf`
  to compute the union over all steps.

### Inside-mesh masking

Optional: when a `MeshAlignedOctreeGPU` is supplied, every voxel centre
is point-located in the mesh via the same
`search_mesh_aligned_octree_multi_local_where` used by the RK4 kernel.
The result is a `(M,)` boolean mask; voxels outside the mesh receive
no density contribution and are written as zeros.

The mask is precomputed once at runner construction. If the velocity
mesh is *time-dependent* (moving / tool-following), the static
snapshot zeros out regions the particles legitimately occupy at other
times. Disable masking in that case (`mask_inside_mesh=False`).

---

## 5. Normalization

Three modes (`DensityRunnerConfig.normalization`):

- `unnormalized`: returns `Σ_i K(‖q − x_i‖, h_i)`. Units = inverse
  volume (kernel units).
- `mass`: returns `Σ_i w_i · K(...)`. If `w_i` are masses, this is the
  conventional SPH mass density (kg / m³ in 3D).
- `pdf`: returns `Σ_i (w_i / W) · K(...)` where `W = Σ_j w_j` is the
  total weight (= N for uniform weights). With this normalization
  `∫_{R³} ρ(x) dx = 1` exactly. Values are then in units of
  *inverse-volume probability density*.

### Why `pdf` peaks at ~10⁵ even though it integrates to 1

For a tight particle cloud the local density can be very large in
absolute (per-volume) units. As a worked example, the current run uses
Wendland C² in 3D, `h = 4.5 × 10⁻³ m`. The kernel value at `r = 0` is

```
W(0, h) = 21 / (16π · h³) ≈ 4.6 × 10⁶ m⁻³
```

With `N = 360 000` particles, a query in the middle of a dense cluster
sees ~10 % of particles (≈ 36 000) within support, and each contributes
on average a fraction of the peak. After the `1/N` pdf normalization
the result is on the order of

```
ρ_peak ≈ (1/N) · Σ K(r_i, h) ≈ (1/360 000) · 36 000 · ~0.3 · 4.6 × 10⁶
        ≈ 1.4 × 10⁵ m⁻³
```

which is exactly the magnitude observed (`max ≈ 4.2 × 10⁵`). The
integral over the whole domain still equals 1 because the *support
volume* `(4π/3)(2h)³ ≈ 3 × 10⁻⁶ m³` is tiny — the field is sharply
peaked. So "pdf" doesn't mean "values ≤ 1"; it means "the field
*integrates* to 1 over space".

If you want the values in `m⁻³` to be of order 1 (e.g. for
human-readable visualisation), increase `h`: the peak value scales as
`h⁻³`. Tripling `h` divides the peak by ~27.

If you want **per-voxel probability mass** instead, multiply the field
by `voxel_volume = dx · dy · dz` post-hoc. With the current grid
(`spacing ≈ (5.7 × 10⁻⁴, 2.3 × 10⁻⁴, 3.5 × 10⁻⁵) m`,
`voxel_volume ≈ 4.6 × 10⁻¹² m³`) the peak per-voxel probability is
`~4.2 × 10⁵ · 4.6 × 10⁻¹² ≈ 2 × 10⁻⁶`, which sums over voxels to ~1.

---

## 6. Backends

Two estimator backends, selected by `EstimatorConfig.engine`:

### Brute force (default for `cost < auto_threshold`)
```
ρ_chunk = Σ_i K(‖Q_chunk[:, None, :] − P[None, :, :]‖, h, d) · w
```
One JIT-compiled function over `(query_chunk, all_particles)`. Memory
peaks at `chunk × N × 3` floats for the `diff` tensor; chunked so the
peak fits in HBM. Exact — sums against every particle. Used in your
current test runs.

### Morton-hash octree
A uniform 3D cell hash over the particle cloud is built each step on
the host:

```
cell_size = max(extent / cells_per_dim, support_radius)
cell_id   = (i · Ny_h + j) · Nz_h + k
```
followed by a stable sort of particles by `cell_id` to produce
CSR-style offsets. The jitted kernel then, per query:

1. Locates the query's cell.
2. Visits the 27 cells of the 3×3×3 stencil.
3. Concatenates up to `octree_max_neighbors` particles from these
   cells into a fixed-size buffer.
4. Evaluates the kernel sum against the buffer.

Known limitation: when the cell occupancy exceeds `max_neighbors`,
particles from later stencil cells are silently dropped, biasing the
result. Until the cell sizing is reworked to bound occupancy
proportionally to `max_neighbors`, the brute backend is the safer
choice (the default `auto_threshold = 1e12` keeps brute selected for
typical workstation grids).

### Padded particle bucket

Both backends pad `N` up to a multiple of `particle_bucket` (default
4096) by appending zero-weight ghost particles. This keeps the JIT
trace shape-stable across runs even when `N` varies slightly (e.g.
inlet/outlet flows), so the kernel only compiles once per session.

---

## 7. Time accumulation

The accumulator ([time_accumulator.py](../jaxtrace/density/time_accumulator.py))
maintains four per-voxel running statistics on the GPU:

```
S_dt       = Σ_k Δt_k                      (scalar)
S_ρ_dt[m]  = Σ_k ρ_k[m] · Δt_k
S_act[m]   = Σ_k (ρ_k[m] > ε) · Δt_k
peak_ρ[m]  = max_k ρ_k[m]
peak_t[m]  = argmax_k ρ_k[m]
```

Finalised fields (written to `density_time_average.vtkhdf`):

```
mean_density[m]      = S_ρ_dt[m] / S_dt
coverage_fraction[m] = S_act[m]  / S_dt          ∈ [0, 1]
peak_density[m]      = peak_ρ[m]
peak_time[m]         = peak_t[m]
```

`coverage_fraction` is the fraction of the simulation time during which
voxel `m` had any density above `ε` — a "footprint" map. `peak_time`
tells you *when* the front passed through. `mean_density` is the
genuine time average.

This integration is built into the per-step hook so it costs one
elementwise add over the grid per step on the GPU; the accumulator
arrays never leave HBM.

---

## 8. On-disk layout (VTKHDF)

Two writer modes:

- `vti`: one `.vti` per step + a `.pvd` index. Universal but
  scattered.
- `vtkhdf` (default): a single transient `ImageData` archive with
  per-step PointData slabs.

### Transient ImageData

PointData arrays are stored as 4-D HDF5 datasets shaped
`(NSteps, Nz, Ny, Nx)`, where each `(Nz, Ny, Nx)` slab is one step.
This is what `vtkHDFReader ≥ 9.4` requires for transient ImageData
(`ndims ≥ 4` check). The runner produces `(Nx, Ny, Nz)`-ordered arrays
internally; the writer transposes to `(Nz, Ny, Nx)` so x varies
fastest in memory.

Companion arrays:

```
VTKHDF/Steps/Values               (NSteps,) float64 — time values
VTKHDF/Steps/PointDataOffsets/<name> (NSteps,) int64 — step index along axis 0
VTKHDF/Steps/PartOffsets          (NSteps,) int64
VTKHDF/Steps/NumberOfParts        (NSteps,) int64
```

Geometry attributes:

```
VTKHDF.attrs.Type        = b"ImageData"
VTKHDF.attrs.WholeExtent = [0, Nx-1, 0, Ny-1, 0, Nz-1]
VTKHDF.attrs.Origin      = (origin_x, origin_y, origin_z)
VTKHDF.attrs.Spacing     = (dx, dy, dz)
VTKHDF.attrs.Direction   = identity 3×3 (flattened)
```

### Steady ImageData (time average)

Same convention without the `Steps/` group. Each PointData array is a
single `(Nz, Ny, Nx)` dataset.

### Compression

`gzip` is the default (and the only filter ParaView's bundled
`vtkhdf5` can decompress). `lzf` and `blosc` are accepted but require
a custom HDF5 install on the reader side; see [`writers.py`](../jaxtrace/density/writers.py)
for the caveats.

---

## 9. Per-particle density export

When `eval_at_particles = True` (default), the estimator also computes
`ρ(x_i, t_k)` at the particle positions themselves (using the same
kernel and bandwidth as the grid pass). The result is shipped as a
`(N,)` scalar to either the offline post-processor's particle export
or, in online mode, the existing `particles.vtkhdf` writer's
`extra_scalars["Density"]` field.

In online mode `run_tracking.py` only runs the density step on the
existing export cadence (`--density-export-freq`, defaulting to
`--export-freq`), so the per-particle field always lines up with the
per-step trajectory data.

---

## 10. Notation summary

| Symbol | Meaning |
|---|---|
| `x_i` | Particle position (m, 3-vector) |
| `q_m` | Voxel-centre position (m, 3-vector) |
| `r`   | `‖q − x‖` Euclidean distance |
| `h_i` | Smoothing length of particle i (m) |
| `q`   | Normalised distance `r/h` |
| `w_i` | Particle weight (mass or 1) |
| `d`   | Spatial dimensionality (2 or 3) |
| `K(r, h, d)` | Normalised kernel |
| `ρ(x, t)` | Estimated density field |
| `N` | Number of particles |
| `M` | Number of active voxels |
| `Δt_k` | Time-integration weight for step k |
| `S_dt`, `S_ρ_dt`, `S_act`, `peak_ρ`, `peak_t` | Accumulator state |
