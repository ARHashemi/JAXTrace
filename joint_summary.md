# Joint MALMO + SCONE + RTXAdvect benchmark

**Mesh cohort**: 10 meshes across three groups (Kim fuel-pin cohort of 8, cross-methodology sanity check on the Stanford Bunny, and the paper's target application mesh — a 3.05 M-tet friction-stir-welding domain). Scale spans 60 tets to 3 050 196 tets. There is no established community-standard tet-mesh cohort for point-location benchmarks; the meshes cited by Wang 2022, Morrical 2020 and Wald 2019 are gated interactive-rendering datasets 10-2000× larger than the FSW target and would not run to completion for SCONE.

**Frameworks**:
- **MALMO** — this paper's mesh-aligned multi-level octree (GPU / JAX). In-mesh barycentric sampling with ground-truth host IDs. `correct%` = fraction of queries where MALMO returned the SAME tet the point was drawn from.
- **SCONE** — Kim et al. 2026 (Cambridge Nuclear) reference implementation of AMLG-Patch and its predecessors, four acceleration methods (`none`/`octree`/`patchSingle`/`patchMulti`), CPU + OpenMP (1 and 8 threads). Runs on the pad-fix sweep with `BOX_PAD=5`. SCONE's `rayVolPhysicsPackage` is a material-tally Monte-Carlo package, not a point-location tool: it reports cycle-level statistics (ray speed, elapsed time, per-material relative volume) but never per-particle host-cell hits, so both `correct%` and `found%` show `—`. Interpret the `status` column instead: `OK` means every ray contributed to the tally (implicitly all found valid material); `FAIL(ec=1)` means SCONE's own "Ray has lost correct material" error terminated tracking mid-cycle; `SEGV` means the acceleration structure could not be built.
- **RTXAdvect** — Wang et al. 2022 (CPC), ported to CUDA 13.3 + OptiX 9.1 on RTX 5090 / Blackwell (compute capability 12.0). BVH-based host-tet locator using RT-cores. Runs 1M particles × 100 steps. `found%` = 1 - (out-of-domain / N_seeded). Sampling is **in-mesh barycentric** (matching MALMO) when a rerun is available under `rtxadvect_runs_inmesh/`; otherwise axis-aligned bounding-box seeding, which biases `found%` downward on meshes with low fill-ratio. Correct-tet ground truth is not compared because RTXAdvect does not report per-particle host-tet IDs back to the CPU.

**Column key**
- `hw` — CPU or GPU
- `par` — parallel mode (JAX / OptiX_RT / OMP_N / serial)
- `n_tets` — mesh size for reference
- `correct%` / `found%` — see framework notes above; `—` = not measurable
- `build (s)` — one-shot preprocessing (octree / BVH / patch grid) that amortises across queries
- `query (s)` — per-batch query wall time (best of 3 for MALMO)
- `RSS (MB)` — peak resident set size of the process
- `tput (Mq/s)` — throughput in million-queries-per-second (`n_points_queried / query_seconds`)

# Kim cohort (fuel pins, SCONE-native)

## FinalFuelPinTet63 (63 tets)

| framework | method | hw | par | status | correct% | found% | build (s) | query (s) | RSS (MB) | PIT | tput (Mq/s) |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| MALMO | `aabb` | GPU | JAX | OK | 100.00 | 100.00 | 0.006 | 0.148 | 1196 | 46.5 | 0.675 |
| MALMO | `centroid` | GPU | JAX | OK | 100.00 | 100.00 | 0.006 | 0.106 | 1197 | 48.9 | 0.942 |
| MALMO | `vertex_multi` | GPU | JAX | OK | 100.00 | 100.00 | 0.004 | 0.087 | 1197 | 30.5 | 1.152 |
| SCONE | `none` | CPU | OMP_1 | OK | — | — | — | 2.147 | 6 | — | — |
| SCONE | `none` | CPU | OMP_8 | OK | — | — | — | 0.344 | 6 | — | — |
| SCONE | `octree` | CPU | OMP_1 | OK | — | — | — | 0.510 | 7 | — | — |
| SCONE | `octree` | CPU | OMP_8 | OK | — | — | — | 0.138 | 7 | — | — |
| SCONE | `patchMulti` | CPU | OMP_1 | SEGV | — | — | — | 3.126 | 716 | — | — |
| SCONE | `patchMulti` | CPU | OMP_8 | SEGV | — | — | — | 3.126 | 715 | — | — |
| SCONE | `patchSingle` | CPU | OMP_1 | FAIL(ec=1) | — | — | — | 0.014 | 13 | — | — |
| SCONE | `patchSingle` | CPU | OMP_8 | FAIL(ec=1) | — | — | — | 0.015 | 14 | — | — |
| RTXAdvect | `RTX_BVH` | GPU | OptiX_RT | OK | — | 100.00 | 0.157 | 0.697 | 405 | — | 100.000 |

## FinalFuelPinTet137 (137 tets)

| framework | method | hw | par | status | correct% | found% | build (s) | query (s) | RSS (MB) | PIT | tput (Mq/s) |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| MALMO | `aabb` | GPU | JAX | OK | 100.00 | 100.00 | 0.015 | 0.431 | 1197 | 96.4 | 0.232 |
| MALMO | `centroid` | GPU | JAX | OK | 100.00 | 100.00 | 0.015 | 0.984 | 1197 | 94.1 | 0.102 |
| MALMO | `vertex_multi` | GPU | JAX | OK | 100.00 | 100.00 | 0.009 | 0.400 | 1196 | 42.4 | 0.250 |
| SCONE | `none` | CPU | OMP_1 | OK | — | — | — | 4.416 | 6 | — | — |
| SCONE | `none` | CPU | OMP_8 | OK | — | — | — | 0.656 | 7 | — | — |
| SCONE | `octree` | CPU | OMP_1 | OK | — | — | — | 0.595 | 8 | — | — |
| SCONE | `octree` | CPU | OMP_8 | OK | — | — | — | 0.178 | 8 | — | — |
| SCONE | `patchMulti` | CPU | OMP_1 | SEGV | — | — | — | 0.024 | 16 | — | — |
| SCONE | `patchMulti` | CPU | OMP_8 | SEGV | — | — | — | 0.023 | 16 | — | — |
| SCONE | `patchSingle` | CPU | OMP_1 | OK | — | — | — | 72.614 | 4837 | — | — |
| SCONE | `patchSingle` | CPU | OMP_8 | OK | — | — | — | 71.333 | 4837 | — | — |
| RTXAdvect | `RTX_BVH` | GPU | OptiX_RT | OK | — | 100.00 | 0.159 | 0.692 | 405 | — | 100.000 |

## FinalFuelPinTet298 (298 tets)

| framework | method | hw | par | status | correct% | found% | build (s) | query (s) | RSS (MB) | PIT | tput (Mq/s) |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| MALMO | `aabb` | GPU | JAX | OK | 100.00 | 100.00 | 0.043 | 0.779 | 1196 | 103.5 | 0.128 |
| MALMO | `centroid` | GPU | JAX | OK | 100.00 | 100.00 | 0.043 | 0.934 | 1195 | 101.9 | 0.107 |
| MALMO | `vertex_multi` | GPU | JAX | OK | 100.00 | 100.00 | 0.019 | 0.892 | 1195 | 47.6 | 0.112 |
| SCONE | `none` | CPU | OMP_1 | OK | — | — | — | 10.187 | 8 | — | — |
| SCONE | `none` | CPU | OMP_8 | OK | — | — | — | 1.529 | 7 | — | — |
| SCONE | `octree` | CPU | OMP_1 | OK | — | — | — | 0.759 | 11 | — | — |
| SCONE | `octree` | CPU | OMP_8 | OK | — | — | — | 0.241 | 11 | — | — |
| SCONE | `patchMulti` | CPU | OMP_1 | SEGV | — | — | — | 0.035 | 20 | — | — |
| SCONE | `patchMulti` | CPU | OMP_8 | SEGV | — | — | — | 0.036 | 20 | — | — |
| SCONE | `patchSingle` | CPU | OMP_1 | OK | — | — | — | 29.195 | 2079 | — | — |
| SCONE | `patchSingle` | CPU | OMP_8 | OK | — | — | — | 28.622 | 2079 | — | — |
| RTXAdvect | `RTX_BVH` | GPU | OptiX_RT | OK | — | 100.00 | 0.160 | 0.707 | 405 | — | 100.000 |

## FinalFuelPinTet1820 (1,820 tets)

| framework | method | hw | par | status | correct% | found% | build (s) | query (s) | RSS (MB) | PIT | tput (Mq/s) |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| MALMO | `aabb` | GPU | JAX | OK | 100.00 | 100.00 | 0.328 | 1.614 | 1196 | 122.6 | 0.062 |
| MALMO | `centroid` | GPU | JAX | OK | 99.99 | 99.99 | 0.328 | 1.461 | 1196 | 122.5 | 0.068 |
| MALMO | `vertex_multi` | GPU | JAX | OK | 99.90 | 99.90 | 0.117 | 1.458 | 1194 | 46.7 | 0.069 |
| SCONE | `none` | CPU | OMP_1 | OK | — | — | — | 89.408 | 16 | — | — |
| SCONE | `none` | CPU | OMP_8 | OK | — | — | — | 13.253 | 16 | — | — |
| SCONE | `octree` | CPU | OMP_1 | OK | — | — | — | 2.321 | 34 | — | — |
| SCONE | `octree` | CPU | OMP_8 | OK | — | — | — | 1.214 | 34 | — | — |
| SCONE | `patchMulti` | CPU | OMP_1 | SEGV | — | — | — | 0.138 | 48 | — | — |
| SCONE | `patchMulti` | CPU | OMP_8 | SEGV | — | — | — | 0.140 | 47 | — | — |
| SCONE | `patchSingle` | CPU | OMP_1 | OK | — | — | — | 295.670 | 15833 | — | — |
| SCONE | `patchSingle` | CPU | OMP_8 | OK | — | — | — | 294.191 | 15833 | — | — |
| RTXAdvect | `RTX_BVH` | GPU | OptiX_RT | OK | — | 100.00 | 0.161 | 0.697 | 404 | — | 100.000 |

## FinalFuelPinTet2856 (2,856 tets)

| framework | method | hw | par | status | correct% | found% | build (s) | query (s) | RSS (MB) | PIT | tput (Mq/s) |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| MALMO | `aabb` | GPU | JAX | OK | 100.00 | 100.00 | 0.567 | 1.350 | 1201 | 82.1 | 0.074 |
| MALMO | `centroid` | GPU | JAX | OK | 100.00 | 99.9990 | 0.579 | 1.492 | 1201 | 82.0 | 0.067 |
| MALMO | `vertex_multi` | GPU | JAX | OK | 99.97 | 99.98 | 0.183 | 1.409 | 1195 | 36.4 | 0.071 |
| SCONE | `none` | CPU | OMP_1 | OK | — | — | — | 152.599 | 22 | — | — |
| SCONE | `none` | CPU | OMP_8 | OK | — | — | — | 23.682 | 22 | — | — |
| SCONE | `octree` | CPU | OMP_1 | OK | — | — | — | 3.709 | 49 | — | — |
| SCONE | `octree` | CPU | OMP_8 | OK | — | — | — | 2.171 | 49 | — | — |
| SCONE | `patchMulti` | CPU | OMP_1 | SEGV | — | — | — | 0.331 | 90 | — | — |
| SCONE | `patchMulti` | CPU | OMP_8 | SEGV | — | — | — | 0.330 | 90 | — | — |
| SCONE | `patchSingle` | CPU | OMP_1 | FAIL(ec=1) | — | — | — | 0.042 | 30 | — | — |
| SCONE | `patchSingle` | CPU | OMP_8 | FAIL(ec=1) | — | — | — | 0.041 | 30 | — | — |
| RTXAdvect | `RTX_BVH` | GPU | OptiX_RT | OK | — | 100.00 | 0.164 | 0.703 | 405 | — | 100.000 |

## FinalFuelPinPoly264 (1,404 tets)

| framework | method | hw | par | status | correct% | found% | build (s) | query (s) | RSS (MB) | PIT | tput (Mq/s) |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| MALMO | `aabb` | GPU | JAX | OK | 100.00 | 100.00 | 0.110 | 0.281 | 1196 | 276.6 | 0.356 |
| MALMO | `centroid` | GPU | JAX | OK | 100.00 | 100.00 | 0.116 | 0.359 | 1195 | 280.6 | 0.279 |
| MALMO | `vertex_multi` | GPU | JAX | OK | 100.00 | 100.00 | 0.091 | 0.192 | 1195 | 174.7 | 0.522 |
| SCONE | `none` | CPU | OMP_1 | OK | — | — | — | 7.684 | 8 | — | — |
| SCONE | `none` | CPU | OMP_8 | OK | — | — | — | 1.102 | 8 | — | — |
| SCONE | `octree` | CPU | OMP_1 | FAIL(ec=1) | — | — | — | 0.339 | 17 | — | — |
| SCONE | `octree` | CPU | OMP_8 | OK | — | — | — | 0.233 | 11 | — | — |
| SCONE | `patchMulti` | CPU | OMP_1 | SEGV | — | — | — | 0.052 | 39 | — | — |
| SCONE | `patchMulti` | CPU | OMP_8 | SEGV | — | — | — | 0.057 | 39 | — | — |
| SCONE | `patchSingle` | CPU | OMP_1 | OK | — | — | — | 10.679 | 2214 | — | — |
| SCONE | `patchSingle` | CPU | OMP_8 | OK | — | — | — | 10.263 | 2214 | — | — |
| RTXAdvect | `RTX_BVH` | GPU | OptiX_RT | OK | — | 100.00 | 0.159 | 0.697 | 405 | — | 100.000 |

## FinalFuelPinPoly940 (5,220 tets)

| framework | method | hw | par | status | correct% | found% | build (s) | query (s) | RSS (MB) | PIT | tput (Mq/s) |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| MALMO | `aabb` | GPU | JAX | OK | 100.00 | 100.00 | 0.449 | 1.455 | 1198 | 233.5 | 0.069 |
| MALMO | `centroid` | GPU | JAX | OK | 99.64 | 99.64 | 0.460 | 1.695 | 1199 | 247.6 | 0.059 |
| MALMO | `vertex_multi` | GPU | JAX | OK | 100.00 | 100.00 | 0.334 | 1.362 | 1200 | 141.8 | 0.073 |
| SCONE | `none` | CPU | OMP_1 | OK | — | — | — | 25.220 | 15 | — | — |
| SCONE | `none` | CPU | OMP_8 | OK | — | — | — | 3.753 | 15 | — | — |
| SCONE | `octree` | CPU | OMP_1 | OK | — | — | — | 1.079 | 25 | — | — |
| SCONE | `octree` | CPU | OMP_8 | OK | — | — | — | 0.446 | 24 | — | — |
| SCONE | `patchMulti` | CPU | OMP_1 | SEGV | — | — | — | 0.178 | 114 | — | — |
| SCONE | `patchMulti` | CPU | OMP_8 | SEGV | — | — | — | 0.175 | 115 | — | — |
| SCONE | `patchSingle` | CPU | OMP_1 | OK | — | — | — | 22.567 | 3908 | — | — |
| SCONE | `patchSingle` | CPU | OMP_8 | OK | — | — | — | 21.935 | 3908 | — | — |
| RTXAdvect | `RTX_BVH` | GPU | OptiX_RT | OK | — | 100.00 | 0.162 | 0.703 | 405 | — | 100.000 |

## FinalFuelPinPoly1560 (8,712 tets)

| framework | method | hw | par | status | correct% | found% | build (s) | query (s) | RSS (MB) | PIT | tput (Mq/s) |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| MALMO | `aabb` | GPU | JAX | OK | 100.00 | 100.00 | 0.491 | 0.797 | 1200 | 190.1 | 0.125 |
| MALMO | `centroid` | GPU | JAX | OK | 100.00 | 100.00 | 0.753 | 1.625 | 1202 | 360.1 | 0.062 |
| MALMO | `vertex_multi` | GPU | JAX | OK | 100.00 | 100.00 | 0.554 | 1.273 | 1204 | 211.5 | 0.079 |
| SCONE | `none` | CPU | OMP_1 | OK | — | — | — | 46.794 | 20 | — | — |
| SCONE | `none` | CPU | OMP_8 | OK | — | — | — | 6.905 | 20 | — | — |
| SCONE | `octree` | CPU | OMP_1 | OK | — | — | — | 1.724 | 39 | — | — |
| SCONE | `octree` | CPU | OMP_8 | OK | — | — | — | 0.888 | 38 | — | — |
| SCONE | `patchMulti` | CPU | OMP_1 | SEGV | — | — | — | 0.308 | 195 | — | — |
| SCONE | `patchMulti` | CPU | OMP_8 | SEGV | — | — | — | 0.312 | 194 | — | — |
| SCONE | `patchSingle` | CPU | OMP_1 | OK | — | — | — | 42.416 | 7045 | — | — |
| SCONE | `patchSingle` | CPU | OMP_8 | OK | — | — | — | 41.628 | 7045 | — | — |
| RTXAdvect | `RTX_BVH` | GPU | OptiX_RT | OK | — | 100.00 | 0.166 | 0.706 | 406 | — | 100.000 |

# Cross-methodology check

## StanfordBunny_LowPoly (379 tets)

| framework | method | hw | par | status | correct% | found% | build (s) | query (s) | RSS (MB) | PIT | tput (Mq/s) |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| MALMO | `aabb` | GPU | JAX | OK | 100.00 | 100.00 | 0.021 | 0.246 | 1195 | 298.5 | 0.406 |
| MALMO | `centroid` | GPU | JAX | OK | 100.00 | 100.00 | 0.020 | 0.247 | 1195 | 298.5 | 0.405 |
| MALMO | `vertex_multi` | GPU | JAX | OK | 96.12 | 96.12 | 0.023 | 4.911 | 1194 | 227.9 | 0.020 |
| SCONE | `none` | CPU | OMP_1 | OK | — | — | — | 5.018 | 8 | — | — |
| SCONE | `none` | CPU | OMP_8 | OK | — | — | — | 0.741 | 8 | — | — |
| SCONE | `octree` | CPU | OMP_1 | OK | — | — | — | 0.436 | 13 | — | — |
| SCONE | `octree` | CPU | OMP_8 | OK | — | — | — | 0.248 | 13 | — | — |
| SCONE | `patchMulti` | CPU | OMP_1 | FAIL(ec=1) | — | — | — | 0.017 | 16 | — | — |
| SCONE | `patchMulti` | CPU | OMP_8 | FAIL(ec=1) | — | — | — | 0.018 | 16 | — | — |
| SCONE | `patchSingle` | CPU | OMP_1 | FAIL(ec=2) | — | — | — | 0.017 | 16 | — | — |
| SCONE | `patchSingle` | CPU | OMP_8 | FAIL(ec=2) | — | — | — | 0.018 | 15 | — | — |
| RTXAdvect | `RTX_BVH` | GPU | OptiX_RT | OK | — | 99.9999 | 0.136 | 0.896 | — | — | 100.000 |

# Paper's target application mesh

## FSW_paper (3,050,196 tets)

| framework | method | hw | par | status | correct% | found% | build (s) | query (s) | RSS (MB) | PIT | tput (Mq/s) |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| MALMO | `aabb` | GPU | JAX | OK | 100.00 | 100.00 | 240.937 | 1.600 | 2772 | 115.5 | 0.062 |
| MALMO | `centroid` | GPU | JAX | OK | 100.00 | 100.00 | 83.600 | 2.426 | 2538 | 78.8 | 0.041 |
| MALMO | `vertex_multi` | GPU | JAX | OK | 100.00 | 100.00 | 197.424 | 6.366 | 3229 | 124.3 | 0.016 |
| SCONE | `none` | CPU | OMP_1 | FAIL(ec=1) | — | — | — | 43.338 | 17277 | — | — |
| SCONE | `none` | CPU | OMP_8 | FAIL(ec=1) | — | — | — | 44.789 | 17277 | — | — |
| SCONE | `octree` | CPU | OMP_1 | FAIL(ec=1) | — | — | — | 16314.671 | 19084 | — | — |
| SCONE | `octree` | CPU | OMP_8 | FAIL(ec=1) | — | — | — | 16315.866 | 19084 | — | — |
| SCONE | `patchMulti` | CPU | OMP_1 | FAIL(ec=1) | — | — | — | 34.480 | 17678 | — | — |
| SCONE | `patchMulti` | CPU | OMP_8 | FAIL(ec=1) | — | — | — | 34.767 | 17679 | — | — |
| SCONE | `patchSingle` | CPU | OMP_1 | FAIL(ec=1) | — | — | — | 34.165 | 17285 | — | — |
| SCONE | `patchSingle` | CPU | OMP_8 | FAIL(ec=1) | — | — | — | 34.179 | 17285 | — | — |
| RTXAdvect | `RTX_BVH` | GPU | OptiX_RT | OK | — | 99.05 | 2.508 | 1.070 | — | — | 6.667 |
