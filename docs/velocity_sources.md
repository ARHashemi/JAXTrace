# Velocity sources: mesh, analytic, and ROM

`run_tracking.py` now supports three sources for the per-substep
velocity used by the RK4 kernel. All three share the same fully-fused
kernel, the same seeding, the same boundary handling, and the same
VTKHDF (or VTU) particle output — they differ only in **where the
velocity value at a given position comes from**.

```
--velocity-source mesh       (default)   FOM PVTU field read at nodes,
                                         P1-interpolated per substep
                                         via the L0/L1/L2 host-element
                                         search.
--velocity-source analytic               v = velocity_fn(x, y, z[, t])
                                         evaluated exactly per substep;
                                         no mesh, no interpolation.
--velocity-source rom                    v = POD-reconstructed field
                                         at mesh nodes, then treated
                                         as a normal mesh field (same
                                         search + P1 interp as `mesh`).
```

This document describes the analytic and ROM paths; the mesh path is
the pre-existing default and is documented under
`docs/phase1/`, `docs/phase2/`, etc.

---

## 1. Analytic velocity source

### 1.1 Motivation

For validation, benchmarking and PDF-appendix reproductions, the
tracking kernel needs to run against a **known analytic vector field**
without an intermediate mesh. This isolates:

* **RK4 truncation** from **P1 interpolation error** — the analytic
  path has zero interpolation error by construction, so any difference
  between analytic and mesh trajectories on the same field is
  attributable to interpolation alone.
* **Search and dedup code paths** from the field's mathematics — the
  analytic path skips both, so bugs in the mesh geometry pipeline
  cannot masquerade as physics.

### 1.2 CLI surface

```
--velocity-source analytic
--velocity-module PATH        # required
--domain-bbox XMIN XMAX YMIN YMAX ZMIN ZMAX
                              # optional override of the module's default bbox
```

The user supplies a Python file that exports a single factory
function:

```python
# my_field.py
import jax.numpy as jnp
from jaxtrace.gpu.tracking.velocity_provider import AnalyticVelocityProvider


def velocity_fn(pos):
    """Return velocity at position pos, shape (3,), JAX-pure."""
    x, y, z = pos[0], pos[1], pos[2]
    ...
    return jnp.stack([u, v, w])


def build_provider(domain_bbox=None, dt=0.0, t_start=0.0):
    return AnalyticVelocityProvider(
        velocity_fn=velocity_fn,
        is_time_dependent=False,
        level_set_fn=None,
        domain_bbox=domain_bbox or ((-1., 1.), (-1., 1.), (-1., 1.)),
        meta={"name": "my_field"},
    )
```

The provider is loaded exactly once, cached, then evaluated per
RK4 substage inside the JIT-compiled kernel.

### 1.3 Included fields

Under `jaxtrace/analytic_fields/`:

| File | Description |
|---|---|
| `uniform.py` | `v = (V_ref, 0, 0)`. Sanity/regression check; closed-form trajectories. |
| `divergence_free_recirculation.py` | Streamfunction-derived field from the FSW Internal Summary (Venghaus 2026), appendix §A. Divergence-free by construction, extruded through z. |

### 1.4 The similarity-scaling gotcha

JAXTrace's spatial search currently iterates only octree levels
[7..14]. A structured mesh of cell size `cs` sits at level
`round(-log2(cs))`. For the raw PDF §A bbox `(-4,4) × (-2,2) × (-1,1)`
with a 64×32×8 mesh, cs = 0.125 ⇒ level 3 — outside the search range,
so no particle is ever assigned.

The fix is a similarity transform `x' = x/S, t' = t/S, v' = v/S`
with `S = 64`. The right similarity **scales every velocity
parameter, not just the geometric ones**:

```
V_ref' = V_ref / S = 5/64 ≈ 0.078   (background flow speed)
A'     = A     / S = 5/64           (disturbance amplitude)
L'     = L     / S = 1/64           (Gaussian half-width)
H'     = H     / S = 1/64           (recirculation wavelength)
bbox'  = bbox  / S                  (spatial extent)
```

Common mistake: dividing only `L` and `H` while leaving `V_ref` and
`A` at 5. That amplifies the velocity-gradient term `A·(2π/H)` by
a factor of S, giving peak speeds of ~2000 m/s in the scaled bbox
and trajectories that fly hundreds of cells per step. Documented in
`jaxtrace/analytic_fields/divergence_free_recirculation.py` and in
the recirc_2026 case's README.

### 1.5 The case study: recirc_2026

Located at `/scratch/shared/ROM/FOM_analytic/recirc_2026/`, this
case validates JAXTrace's P1 interpolation error against the
analytic path for the PDF §A field:

* `recirculation_field.py` — the scaled §A field (SCALE=64).
* `case_config.sh` — dt=5e-5, N=2000, seed grid 60×120×50.
* `generate_meshes.sh` — builds three test meshes:
  - `mesh_uniform/`  (uniform 64×32×8, ~100 k tets, no refinement)
  - `mesh_2lvl/`     (1 refinement pass, R=1.5 raw ⇒ ~200 k tets)
  - `mesh_4lvl/`     (3 nested refinement passes, ~1 M tets)
* `run_analytic.sh`, `run_mesh_uniform.sh`, `run_mesh_2lvl.sh`,
  `run_mesh_4lvl.sh` — the four tracking runs.
* `compare.py` — per-step max/rms error of each mesh variant vs the
  analytic reference; writes summary.json + error.png.

The adaptive refinement code lives at
`tests/analytic_velocity/generate_test_mesh.py`. Each pass identifies
hex cells whose centroid lies within radius R of the center and
subdivides each into 8 child cells at the next level. T-junctions
between refinement levels are intentional and handled by the L2
search (which walks neighbouring cells across all levels).

### 1.6 Output format parity

Both paths write identical VTKHDF (or VTU) archives via the same
`VTKHDFExportThread` / `VTKExportThread` used by the mesh path.
`Group`, `ParticleID` and `Escaped` scalars are all supported on
the analytic path. Downstream tools that consume mesh-path output
work unchanged on analytic-path output.

---

## 2. ROM velocity source

### 2.1 Motivation

The colleague's first-stage FSW-ROM produces a truncated POD basis
of the FOM velocity field over 20 cases in the cohort cylindrical
dataset. Reconstructing the velocity at every mesh node from this
basis lets us:

* Run particle tracking with the ROM-approximated velocity and
  compare trajectories against the actual FOM run — a direct
  quantification of how much the ROM's velocity errors propagate
  into Lagrangian dynamics.
* Provide a template for downstream ROM-driven tracking (e.g.
  during ROM-based FSW parametric studies) with no per-step ROM
  reconstruction cost, since the reconstructed field is fixed for
  a given case.

### 2.2 CLI surface

```
--velocity-source rom
--rom-basis         PATH        # *.fswrom.basis (HDF5)
--rom-romdata       PATH        # *.fswrom.romdata (HDF5)
--rom-case-idx      N           # 0..19 for cohort cylindrical dataset
--rom-formula       NAME        # centered | sigma_c | c_over_sig |
                                # no_mean | no_mean_sig
--rom-field-group   NAME        # HDF5 group; default 'Displacement'
```

`--input` still points at a case directory containing a real PVTU;
this is where the mesh geometry (nodes + connectivity) comes from.
The FOM velocity in that PVTU is **loaded and then discarded** — it
gets replaced by the ROM reconstruction before dedup. Downstream
(dedup, GPU upload, tracking) is completely unchanged.

The reconstruction happens **once**, before the RK4 loop. The
resulting `(n_nodes, 3)` velocity field is stuffed into a
single-timestep `velocity_sequence` and served to every RK4 step
via the standard cyclic-index path.

### 2.3 Data file structure

For the cohort cylindrical dataset:

```
cylindrical.som.fswrom.basis
└── ROMDATA/cylindrical.som/
    ├── Displacement/
    │   ├── SnapshotsMean 1/2/3        (180,461,) each
    │   ├── Basis_CompMode 1 1/2/3     (180,461,) each  ← mode 1
    │   ├── Basis_CompMode 2 1/2/3     (180,461,) each  ← mode 2
    │   └── Basis_CompMode 3 1/2/3     (180,461,) each  ← mode 3
    ├── Pressure/  (same layout, 1 mode)
    └── Temperature/  (same layout, 3 modes)

cylindrical.som.fswrom.romdata
└── ROMDATA/cylindrical.som/
    └── Displacement/
        ├── BasisCoefficients_Mode1    (20,)   ← per-case coefficient
        ├── BasisCoefficients_Mode2    (20,)
        ├── BasisCoefficients_Mode3    (20,)
        └── Sigma_Mode{1..20}          (1,) each  ← singular values
```

Only 3 spatial modes are stored, but 20 singular values are listed.
For the cylindrical dataset the top-3 sigmas are 101, 6.1, 3.7; the
remaining 17 sit at 1.3 and below.

### 2.4 Reconstruction formulas

The exact scaling convention used to store the basis modes isn't
documented in the .fswrom files. `jaxtrace/rom/velocity_recon.py`
supports five formulas and lets the user pick empirically:

| Formula | Reconstruction |
|---|---|
| `centered`    | `v = mean + Σ_k c_k · φ_k`                |
| `sigma_c`     | `v = mean + Σ_k σ_k · c_k · φ_k`          |
| `c_over_sig`  | `v = mean + Σ_k (c_k / σ_k) · φ_k`        |
| `no_mean`     | `v =        Σ_k c_k · φ_k`                |
| `no_mean_sig` | `v =        Σ_k σ_k · c_k · φ_k`          |

The scorer (`tests/rom/compare_rom_recon.py`) runs each formula
against a real FOM snapshot and ranks by RMS error.

### 2.5 The case study: rom_test_001

Located at `/scratch/shared/ROM/FOM/rom_test_001/`, this case
validates the ROM reconstruction end-to-end:

* `case_config.sh` — points at `cylindrical_001.gid/post` for the
  mesh and the FOM reference; sets the ROM basis+romdata paths;
  matches the cohort case's dt, N, seed grid.
* `run_fom.sh` — tracks the reference case with the FOM PVTU
  velocity (loads `cylindrical_119.pvtu`, uses it directly).
* `run_rom.sh` — tracks with the ROM-reconstructed velocity,
  same mesh, same seeds.
* `compare.py` — node-level formula scoring plus per-step
  particle-trajectory error between the FOM and ROM runs;
  writes summary.json + error.png.

Sample scoring output on the cohort:

```
formula          max_abs        rms       cosine    rel_rms
c_over_sig     3.06e-01     4.59e-02      0.98      0.65    ← best
centered       5.82e-01     5.68e-02      0.86      0.80
no_mean        6.55e-01     1.04e-01     -0.38      1.47
sigma_c        5.92e+01     5.47e+00     -0.36     77.16
no_mean_sig    5.92e+01     5.51e+00     -0.38     77.78
```

**Best formula (`c_over_sig`) has cosine similarity 0.98 but
relative RMS error 65 % of |FOM|_rms** — the reconstruction gets
the flow direction right but underestimates the magnitude. The
mean-alone baseline gets the same 65 %, so the top-3 modes barely
change the fit against a single-timestep FOM snapshot.

Interpretation: `SnapshotsMean` looks like a time- and case-averaged
velocity, and the truncated basis captures only the largest inter-case
variations — not the transient specifics of any given timestep. This
is expected behaviour of a low-rank POD trained on the full snapshot
stack; the reconstruction is only exact at snapshots that lie in the
span of the stored basis, and step 119 doesn't. Working with the
colleague to identify the exact scaling convention is an open task.

Trajectory-level error observed on a full 360k-particle × 2000-step
run:

```
per-step max err  final: 8.67e-02, global: 8.67e-02
per-step rms err  final: 2.36e-02, global: 2.36e-02
```

Interpretation: over 2000 steps the ~65 % velocity error accumulates
into a max position drift of ~87 mm and rms drift of ~24 mm across
360k particles. Meaningful for structural comparisons of Lagrangian
statistics; not a match for individual trajectories.

### 2.6 Performance

The ROM reconstruction is a single dense matrix product across
`(n_modes × n_nodes × 3)` values, plus a per-node addition of the
mean field. For the cohort cylindrical dataset:

| Item | Cost |
|---|---|
| Load basis + romdata (HDF5) | ~0.5 s (30 MB basis, 220 kB romdata) |
| Reconstruct per-node velocity | <0.05 s (180 k nodes × 3 modes × 3 components) |
| Upload to GPU | same as mesh path (~10 s for 180 k nodes) |
| Per-step RK4 tracking | **identical to mesh path** |

Since reconstruction happens once before the RK4 loop, tracking
throughput is byte-for-byte the same as the FOM mesh path. There is
no per-step ROM overhead.

---

## 3. Testing

Locations of the validation harnesses:

| Path | Purpose |
|---|---|
| `tests/test_velocity_provider_parity.py` | Layer 1 contract tests for the mesh vs analytic velocity provider abstraction. |
| `tests/test_analytic_path.py` | Layer 2 correctness of the analytic RK4 kernel against closed-form and scipy DOP853 references. |
| `tests/analytic_velocity/run_validation.py` | Three-way harness (analytic vs mesh vs scipy) at multiple mesh resolutions. |
| `tests/rom/compare_rom_recon.py` | ROM formula scorer against a real FOM snapshot. |

---

## 4. Change log

| Commit | Description |
|---|---|
| 9fb659d | Phase 1: velocity_provider abstraction (no behavioural change). |
| a89e0ff | Phase 2a: analytic RK4 kernel + reference fields + unit tests. |
| 8324e88 | Phase 2b: `--velocity-source analytic` CLI + `run_workstation.sh` wiring. |
| ccfb5f4 | Phase 3: mesh generator + three-way validation harness. |
| 2e76808 | Validation harness: cap subprocess VRAM + surface failing log inline. |
| 4c13b44 | Analytic path: emit VTKHDF (and VTU) like the mesh path. |
| ab3ba32 | `generate_test_mesh.py`: add adaptive Kuhn refinement. |
| a919e6e | `run_tracking.py`: `--velocity-source rom` for FSW-ROM velocity reconstruction. |
