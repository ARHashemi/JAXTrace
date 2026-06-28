# Analytic-velocity validation harness

Phase 3 of the analytic-velocity feature: tools to validate the new
`--velocity-source analytic` path against the existing mesh path.

## What's in here

| File | Role |
|---|---|
| `generate_test_mesh.py` | Build a Kuhn-tetrahedralised structured tetrahedral mesh covering a user-specified bbox and write it as a JAXTrace-compatible PVTU. Evaluates a user-supplied analytic velocity function at every node. |
| `recirculation_scaled.py` | Scaled-down copy of `jaxtrace/analytic_fields/divergence_free_recirculation.py`. Cell sizes land in the search code's `[7..14]` level range. Use for mesh-vs-analytic comparison. |
| `run_validation.py` | Three-way harness: analytic path vs mesh path vs scipy DOP853 reference. Emits a summary.json + per-resolution error table. |

## The level-range constraint

JAXTrace's spatial search currently iterates over octree levels 7..14
(hard-coded in `jaxtrace/gpu/search/mesh_aligned_point_location.py`).
Cells outside that range never get visited.

A structured hex mesh's level is set by its cell_size, where

```
level ≈ round(-log2(cell_size))
```

A 64-cell-wide mesh of width 8 (e.g. the PDF appendix bbox
`[-4,4] × [-2,2] × [-0.25,0.25]`) has cell_size ≈ 0.125 → **level 3**,
outside the search range. Result: zero particles found.

The fix for testing is `recirculation_scaled.py` — same flow geometry
as the published field but with `L`, `H`, `xc` divided by 64, used with
a 64× shrunk bbox. Cell sizes then land at level ~10, comfortably in
range.

The longer-term fix is to make the search code iterate over the
**actual** levels present in the octree (instead of the hard-coded
`[7..14]`). That's a separate refactor.

## Quick start

```bash
cd /flash/shared/jax/JAXTrace   # workstation root

# 1. Generate a test mesh:
python tests/analytic_velocity/generate_test_mesh.py \
    --velocity-module tests/analytic_velocity/recirculation_scaled.py \
    --bbox -0.0625 0.0625 -0.03125 0.03125 -0.00390625 0.00390625 \
    --n-cells 64 32 8 \
    --output /tmp/mesh_64x32x8 \
    --stem mesh_0

# 2. Run the validation harness (analytic + scipy + one mesh resolution):
python tests/analytic_velocity/run_validation.py \
    --mesh-resolutions 64 \
    --n-steps 50 \
    --output /tmp/validation_run

# 3. Inspect summary:
cat /tmp/validation_run/summary.json
```

The harness reports for each mesh resolution:

```
  analytic:        max=1.2e-14,  rms=4.1e-15   <- pure RK4 truncation
  mesh [64,32,8]:  max=5.3e-04,  rms=2.1e-04   <- RK4 + interpolation
                   Δmax=+5.3e-04                <- the interpolation
                                                   contribution
```

The `mesh - analytic` delta isolates the mesh-induced trajectory error,
since RK4 truncation is shared between paths.

## Convergence sweep

```bash
python tests/analytic_velocity/run_validation.py \
    --mesh-resolutions 32 64 128 \
    --n-steps 50 \
    --output /tmp/validation_sweep
```

Each refinement halves the cell size and is expected to reduce
interpolation error roughly proportionally (P1 interpolation is `O(h²)`
in space). Plot from `summary.json`.

## What the validation actually proves

The two paths share:
* the same fully-fused RK4 step kernel structure
* the same velocity_provider abstraction (the provider object differs;
  the call site is identical)
* the same boundary handling (sub-step bbox clamp, boundary projection)

What differs:
* the mesh path's velocity comes from L0/L1/L2 host-element search
  followed by P1 barycentric interpolation
* the analytic path evaluates `velocity_fn(pos)` exactly

So if the analytic-vs-scipy error is round-off (~1e-14) and the
mesh-vs-scipy error is some larger value, the difference **must** be
interpolation error: it's the only piece that differs between paths.

## Known limitations

* `generate_test_mesh.py` only supports steady velocity fields. For
  time-dependent fields the PVTU would need one file per timestep.
* `run_validation.py` uses subprocess invocations of run_tracking.py
  for both paths, which pays a ~10s JIT-compile cost each. For a
  convergence sweep this adds up; future improvement: run everything
  in-process.
* The level-range workaround means we can't directly compare against
  the cohort's published parameters. The scaled field reproduces the
  same flow geometry; absolute distances and times scale by `1/SCALE`.

## Phase 3 status

| Stage | What | Status |
|---|---|---|
| 1 | velocity_provider abstraction | done (Phase 1) |
| 2a | analytic kernel + reference fields | done |
| 2b | CLI flags + run_workstation.sh wiring | done |
| 3 | **mesh generator + validation harness** | **this folder** |
| — | level-range fix in search code | deferred |
