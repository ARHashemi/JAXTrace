# JAXTrace

GPU-accelerated Lagrangian particle tracking on unstructured tetrahedral
meshes, written in Python on top of JAX/XLA. The point-location kernel
is a *mesh-aligned multi-level octree* (MALMO) that resolves the host
element of each particle in a single fused RK4 step.

The intended workload is post-processing of finite-element flow
solutions: given a sequence of node-centred velocity snapshots on the
same mesh, integrate large batches of passive tracers through the
domain.

[![License: EUPL-1.2 with Additional Terms](https://img.shields.io/badge/license-EUPL--1.2%20with%20Additional%20Terms-blue.svg)](LICENSE)

---

## What it does

* **Reads** time-dependent velocity fields from PVTU file sequences
  (the FEMUSS export layout is the default; any compatible PVTU stack
  works).
* **Builds** a mesh-aligned octree over the source mesh
  (centroid-registered parent cubes with a static 3×3×3 search
  neighbourhood — see references below).
* **Tracks** particles via a fully-fused RK4 step on GPU. The integration
  loop, host-element search, P1 interpolation, and boundary handling
  are JIT-compiled into a single XLA graph.
* **Writes** particle positions and per-step element IDs to VTU /
  VTP files; an optional HDF5 path is also available.
* Validates against FEMUSS's own embedded particle tracker for the same
  data.

Limitations: tested on a fixed mesh topology with a periodic velocity
sequence; r-adaptive (per-step remeshed) datasets are not yet handled
end-to-end by the runtime — see the design notes branch for that work.

---

## Requirements

Python 3.10+ and a working JAX install for your platform:

* CPU only (fallback, slow): `pip install -r requirements.txt`
* NVIDIA CUDA: install JAX per
  [Google's instructions](https://jax.readthedocs.io/en/latest/installation.html)
  before `pip install -e .`
* AMD ROCm (LUMI): use the container image referenced in
  `scripts/run_lumi.sh`.

Core dependencies: `numpy`, `jax`, `jaxlib`, `vtk`, `h5py`, `scipy`.

---

## Install

```bash
git clone --branch release/stable https://github.com/ARHashemi/JAXTrace.git
cd JAXTrace
pip install -e .
```

For day-to-day production work, **`release/stable`** is the branch to
clone — it contains only the runtime code (no design notes, run logs,
or diagnostic scripts). The full development history is on
`feature/lumi-benchmark`.

---

## Running a tracking job

The production entry point is `run_tracking.py`. It expects a directory
of PVTU files indexed by time step and produces VTU output for the
tracked particles.

```bash
python run_tracking.py \
    --input  /path/to/case.gid/post \
    --output /path/to/results \
    --n-steps 2684
```

By default the loader auto-detects the FEMUSS case stem from the input
directory; pass `--mesh-dir` / `--femuss-dir` if your layout differs.

### Particle seeding

Pick one of six modes via `--seed-source`:

| Mode | Bounds | Distribution | Required args |
|------|--------|--------------|---------------|
| `femuss`    | from FEMUSS PVTU | as in source | `--femuss-start` |
| `box`       | `--seed-box XMIN XMAX YMIN YMAX ZMIN ZMAX` | uniform random | `--n-particles` |
| `grid`      | `--seed-box` | uniform grid | `--seed-grid NX NY NZ` |
| `box-frac`  | fractions of mesh bbox via `--seed-fraction XLO XHI YLO YHI ZLO ZHI` | uniform random | `--n-particles` |
| `grid-frac` | `--seed-fraction` | uniform grid | `--seed-grid NX NY NZ` |
| `file`      | from `.npy` / `.npz` of shape `(N, 3)` | from file | `--seed-file` |

Examples:

```bash
# Seed 100k particles uniformly at random in an absolute box
python run_tracking.py --input <dir> --output <dir> \
    --seed-source box \
    --seed-box -0.01 0.01 -0.005 0.005 0.0 0.002 \
    --n-particles 100000 --n-steps 2684

# 50x70x30 = 105k particles on a regular grid covering the first 20%
# of X (full Y, full Z) of the mesh bounding box
python run_tracking.py --input <dir> --output <dir> \
    --seed-source grid-frac \
    --seed-fraction 0.0 0.2 0.0 1.0 0.0 1.0 \
    --seed-grid 50 70 30 --n-steps 2684
```

### Cluster wrappers

* `scripts/run_lumi.sh` — SLURM batch script for LUMI (AMD MI250X).
  Edit the `[1]`–`[6]` configuration blocks at the top, then `sbatch`.
* `scripts/run_workstation.sh` — local equivalent for an NVIDIA
  workstation. Sets the XLA memory cap and forwards the same CLI
  flags as the LUMI script.

Both expose the seeding modes above through `SEED_SOURCE`, `SEED_BOX`,
`SEED_FRACTION`, `SEED_GRID`, `N_PARTICLES`.

### Validating against FEMUSS

If your input directory contains a FEMUSS particle-tracking PVTU,
add `--femuss-compare` (and use `--seed-source femuss`). The driver
will read the FEMUSS reference at the final step and report
per-particle position errors. See
`diagnose_femuss_deviation.py` for a step-by-step deviation
analyser.

---

## Project layout

```
run_tracking.py                  # production driver
benchmark_femuss_comparison.py   # reference benchmark + RK4 builder
benchmark_l2_accuracy.py         # search-correctness microbenchmark
diagnose_femuss_deviation.py     # per-step deviation analyser
jaxtrace/                        # library code
├── config.py                    # precision and RK4 policy flags
├── tracking/                    # seeding, integrators, boundaries
└── gpu/                         # JAX/GPU implementation
    ├── mesh_loader.py           # PVTU → flat arrays
    ├── mesh_loader_timedep.py   # time-series loader
    ├── search/                  # MALMO octree (build + search kernels)
    └── tracking/                # mesh upload, fused RK4 step
scripts/                         # cluster + workstation runners
tests/                           # unit + integration tests
config/                          # reference YAML configs
examples/gpu/                    # minimal GPU usage examples
```

---

## How MALMO works (brief)

For a query point **q**, the search needs to return the host
tetrahedral element K* ∈ T_h. MALMO builds an octree over axis-aligned
parent cubes derived from the mesh's natural refinement levels; each
tetrahedron is registered to the parent cube containing its centroid.
At query time, the kernel computes **q**'s Morton code, locates the
candidate cell, and tests the (≤8) elements registered there plus a
26-cell halo (3×3×3). The whole loop is statically shaped, allowing
JAX/XLA to fuse it into a single RK4 substep with no host
round-trips.

The 3×3×3 neighbourhood is sufficient (and necessary) for the
class of meshes we target; see the paper draft for the geometric
argument and validation.

---

## License

EUPL-1.2 (with Additional Terms) — see [LICENSE](LICENSE).

---

## Citation

```bibtex
@software{jaxtrace,
  author = {Hashemi, A. R.},
  title  = {JAXTrace: GPU-accelerated particle tracking on unstructured
            tetrahedral meshes},
  url    = {https://github.com/ARHashemi/JAXTrace},
  year   = {2026}
}
```
