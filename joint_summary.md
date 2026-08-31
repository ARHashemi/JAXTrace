# Joint MALMO + SCONE + RTXAdvect benchmark

**Meshes**: 9 Kim fuel-pin / bunny meshes (63 – 8712 tets, plus StanfordBunny) + FSW paper mesh (cylA_119, 3 050 196 tets).

**Frameworks**:
- **MALMO** — this paper's mesh-aligned multi-level octree; in-mesh barycentric sampling with ground-truth host IDs. `correct%` = fraction of queries where MALMO returned the SAME tet the point was drawn from.
- **SCONE** — Kim et al. 2026 (Cambridge Nuclear) reference implementation of AMLG-Patch and its predecessors (`none`/`octree`/`patchMulti`/`patchSingle`). Runs on the pad-fix sweep with BOX_PAD=5. SCONE has no query-list driver so `correct%` is not applicable (it ray-traces with internal validation).
- **RTXAdvect** — Wang et al. 2022 (CPC), ported to CUDA 13.3 + OptiX 9.1 on RTX 5090 (Blackwell). BVH-based host-tet locator using RT-cores. Runs 1M particles × 100 steps; `found%` = 1 - (out-of-domain / N_seeded). Sampling is **in-mesh barycentric** (matching MALMO) when a rerun is available under `rtxadvect_runs_inmesh/`; otherwise it is axis-aligned bounding-box seeding, which biases `found%` downward on meshes with low fill-ratio (e.g. Stanford Bunny at 27.2%). Correct-tet ground truth is not compared because RTXAdvect does not report per-particle host-tet IDs back to the CPU.

## FinalFuelPinTet63

| framework | method | omp | status | correct% | found% | wall (s) | RSS (MB) | PIT | tput (Mq/s) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| MALMO | `aabb` | 1 | OK | 100.00 | 100.00 | 0.148 | 1196 | 46.5 | 0.675 |
| MALMO | `centroid` | 1 | OK | 100.00 | 100.00 | 0.106 | 1197 | 48.9 | 0.942 |
| MALMO | `vertex_multi` | 1 | OK | 100.00 | 100.00 | 0.087 | 1197 | 30.5 | 1.152 |
| SCONE | `none` | 1 | OK | — | — | 2.147 | 6 | — | — |
| SCONE | `none` | 8 | OK | — | — | 0.344 | 6 | — | — |
| SCONE | `octree` | 1 | OK | — | — | 0.510 | 7 | — | — |
| SCONE | `octree` | 8 | OK | — | — | 0.138 | 7 | — | — |
| SCONE | `patchMulti` | 1 | SEGV | — | — | 3.126 | 716 | — | — |
| SCONE | `patchMulti` | 8 | SEGV | — | — | 3.126 | 715 | — | — |
| SCONE | `patchSingle` | 1 | FAIL(ec=1) | — | — | 0.014 | 13 | — | — |
| SCONE | `patchSingle` | 8 | FAIL(ec=1) | — | — | 0.015 | 14 | — | — |
| RTXAdvect | `RTX_BVH` | 1 | OK | — | 100.00 | 1.588 | 405 | — | 100.000 |

## FinalFuelPinTet137

| framework | method | omp | status | correct% | found% | wall (s) | RSS (MB) | PIT | tput (Mq/s) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| MALMO | `aabb` | 1 | OK | 100.00 | 100.00 | 0.431 | 1197 | 96.4 | 0.232 |
| MALMO | `centroid` | 1 | OK | 100.00 | 100.00 | 0.984 | 1197 | 94.1 | 0.102 |
| MALMO | `vertex_multi` | 1 | OK | 100.00 | 100.00 | 0.400 | 1196 | 42.4 | 0.250 |
| SCONE | `none` | 1 | OK | — | — | 4.416 | 6 | — | — |
| SCONE | `none` | 8 | OK | — | — | 0.656 | 7 | — | — |
| SCONE | `octree` | 1 | OK | — | — | 0.595 | 8 | — | — |
| SCONE | `octree` | 8 | OK | — | — | 0.178 | 8 | — | — |
| SCONE | `patchMulti` | 1 | SEGV | — | — | 0.024 | 16 | — | — |
| SCONE | `patchMulti` | 8 | SEGV | — | — | 0.023 | 16 | — | — |
| SCONE | `patchSingle` | 1 | OK | — | — | 72.614 | 4837 | — | — |
| SCONE | `patchSingle` | 8 | OK | — | — | 71.333 | 4837 | — | — |
| RTXAdvect | `RTX_BVH` | 1 | OK | — | 100.00 | 1.579 | 405 | — | 100.000 |

## FinalFuelPinTet298

| framework | method | omp | status | correct% | found% | wall (s) | RSS (MB) | PIT | tput (Mq/s) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| MALMO | `aabb` | 1 | OK | 100.00 | 100.00 | 0.779 | 1196 | 103.5 | 0.128 |
| MALMO | `centroid` | 1 | OK | 100.00 | 100.00 | 0.934 | 1195 | 101.9 | 0.107 |
| MALMO | `vertex_multi` | 1 | OK | 100.00 | 100.00 | 0.892 | 1195 | 47.6 | 0.112 |
| SCONE | `none` | 1 | OK | — | — | 10.187 | 8 | — | — |
| SCONE | `none` | 8 | OK | — | — | 1.529 | 7 | — | — |
| SCONE | `octree` | 1 | OK | — | — | 0.759 | 11 | — | — |
| SCONE | `octree` | 8 | OK | — | — | 0.241 | 11 | — | — |
| SCONE | `patchMulti` | 1 | SEGV | — | — | 0.035 | 20 | — | — |
| SCONE | `patchMulti` | 8 | SEGV | — | — | 0.036 | 20 | — | — |
| SCONE | `patchSingle` | 1 | OK | — | — | 29.195 | 2079 | — | — |
| SCONE | `patchSingle` | 8 | OK | — | — | 28.622 | 2079 | — | — |
| RTXAdvect | `RTX_BVH` | 1 | OK | — | 100.00 | 1.602 | 405 | — | 100.000 |

## FinalFuelPinTet1820

| framework | method | omp | status | correct% | found% | wall (s) | RSS (MB) | PIT | tput (Mq/s) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| MALMO | `aabb` | 1 | OK | 100.00 | 100.00 | 1.614 | 1196 | 122.6 | 0.062 |
| MALMO | `centroid` | 1 | OK | 99.99 | 99.99 | 1.461 | 1196 | 122.5 | 0.068 |
| MALMO | `vertex_multi` | 1 | OK | 99.90 | 99.90 | 1.458 | 1194 | 46.7 | 0.069 |
| SCONE | `none` | 1 | OK | — | — | 89.408 | 16 | — | — |
| SCONE | `none` | 8 | OK | — | — | 13.253 | 16 | — | — |
| SCONE | `octree` | 1 | OK | — | — | 2.321 | 34 | — | — |
| SCONE | `octree` | 8 | OK | — | — | 1.214 | 34 | — | — |
| SCONE | `patchMulti` | 1 | SEGV | — | — | 0.138 | 48 | — | — |
| SCONE | `patchMulti` | 8 | SEGV | — | — | 0.140 | 47 | — | — |
| SCONE | `patchSingle` | 1 | OK | — | — | 295.670 | 15833 | — | — |
| SCONE | `patchSingle` | 8 | OK | — | — | 294.191 | 15833 | — | — |
| RTXAdvect | `RTX_BVH` | 1 | OK | — | 100.00 | 1.587 | 404 | — | 100.000 |

## FinalFuelPinTet2856

| framework | method | omp | status | correct% | found% | wall (s) | RSS (MB) | PIT | tput (Mq/s) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| MALMO | `aabb` | 1 | OK | 100.00 | 100.00 | 1.350 | 1201 | 82.1 | 0.074 |
| MALMO | `centroid` | 1 | OK | 100.00 | 99.9990 | 1.492 | 1201 | 82.0 | 0.067 |
| MALMO | `vertex_multi` | 1 | OK | 99.97 | 99.98 | 1.409 | 1195 | 36.4 | 0.071 |
| SCONE | `none` | 1 | OK | — | — | 152.599 | 22 | — | — |
| SCONE | `none` | 8 | OK | — | — | 23.682 | 22 | — | — |
| SCONE | `octree` | 1 | OK | — | — | 3.709 | 49 | — | — |
| SCONE | `octree` | 8 | OK | — | — | 2.171 | 49 | — | — |
| SCONE | `patchMulti` | 1 | SEGV | — | — | 0.331 | 90 | — | — |
| SCONE | `patchMulti` | 8 | SEGV | — | — | 0.330 | 90 | — | — |
| SCONE | `patchSingle` | 1 | FAIL(ec=1) | — | — | 0.042 | 30 | — | — |
| SCONE | `patchSingle` | 8 | FAIL(ec=1) | — | — | 0.041 | 30 | — | — |
| RTXAdvect | `RTX_BVH` | 1 | OK | — | 100.00 | 1.603 | 405 | — | 100.000 |

## FinalFuelPinPoly264

| framework | method | omp | status | correct% | found% | wall (s) | RSS (MB) | PIT | tput (Mq/s) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| MALMO | `aabb` | 1 | OK | 100.00 | 100.00 | 0.281 | 1196 | 276.6 | 0.356 |
| MALMO | `centroid` | 1 | OK | 100.00 | 100.00 | 0.359 | 1195 | 280.6 | 0.279 |
| MALMO | `vertex_multi` | 1 | OK | 100.00 | 100.00 | 0.192 | 1195 | 174.7 | 0.522 |
| SCONE | `none` | 1 | OK | — | — | 7.684 | 8 | — | — |
| SCONE | `none` | 8 | OK | — | — | 1.102 | 8 | — | — |
| SCONE | `octree` | 1 | FAIL(ec=1) | — | — | 0.339 | 17 | — | — |
| SCONE | `octree` | 8 | OK | — | — | 0.233 | 11 | — | — |
| SCONE | `patchMulti` | 1 | SEGV | — | — | 0.052 | 39 | — | — |
| SCONE | `patchMulti` | 8 | SEGV | — | — | 0.057 | 39 | — | — |
| SCONE | `patchSingle` | 1 | OK | — | — | 10.679 | 2214 | — | — |
| SCONE | `patchSingle` | 8 | OK | — | — | 10.263 | 2214 | — | — |
| RTXAdvect | `RTX_BVH` | 1 | OK | — | 100.00 | 1.583 | 405 | — | 100.000 |

## FinalFuelPinPoly940

| framework | method | omp | status | correct% | found% | wall (s) | RSS (MB) | PIT | tput (Mq/s) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| MALMO | `aabb` | 1 | OK | 100.00 | 100.00 | 1.455 | 1198 | 233.5 | 0.069 |
| MALMO | `centroid` | 1 | OK | 99.64 | 99.64 | 1.695 | 1199 | 247.6 | 0.059 |
| MALMO | `vertex_multi` | 1 | OK | 100.00 | 100.00 | 1.362 | 1200 | 141.8 | 0.073 |
| SCONE | `none` | 1 | OK | — | — | 25.220 | 15 | — | — |
| SCONE | `none` | 8 | OK | — | — | 3.753 | 15 | — | — |
| SCONE | `octree` | 1 | OK | — | — | 1.079 | 25 | — | — |
| SCONE | `octree` | 8 | OK | — | — | 0.446 | 24 | — | — |
| SCONE | `patchMulti` | 1 | SEGV | — | — | 0.178 | 114 | — | — |
| SCONE | `patchMulti` | 8 | SEGV | — | — | 0.175 | 115 | — | — |
| SCONE | `patchSingle` | 1 | OK | — | — | 22.567 | 3908 | — | — |
| SCONE | `patchSingle` | 8 | OK | — | — | 21.935 | 3908 | — | — |
| RTXAdvect | `RTX_BVH` | 1 | OK | — | 100.00 | 1.611 | 405 | — | 100.000 |

## FinalFuelPinPoly1560

| framework | method | omp | status | correct% | found% | wall (s) | RSS (MB) | PIT | tput (Mq/s) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| MALMO | `aabb` | 1 | OK | 100.00 | 100.00 | 0.797 | 1200 | 190.1 | 0.125 |
| MALMO | `centroid` | 1 | OK | 100.00 | 100.00 | 1.625 | 1202 | 360.1 | 0.062 |
| MALMO | `vertex_multi` | 1 | OK | 100.00 | 100.00 | 1.273 | 1204 | 211.5 | 0.079 |
| SCONE | `none` | 1 | OK | — | — | 46.794 | 20 | — | — |
| SCONE | `none` | 8 | OK | — | — | 6.905 | 20 | — | — |
| SCONE | `octree` | 1 | OK | — | — | 1.724 | 39 | — | — |
| SCONE | `octree` | 8 | OK | — | — | 0.888 | 38 | — | — |
| SCONE | `patchMulti` | 1 | SEGV | — | — | 0.308 | 195 | — | — |
| SCONE | `patchMulti` | 8 | SEGV | — | — | 0.312 | 194 | — | — |
| SCONE | `patchSingle` | 1 | OK | — | — | 42.416 | 7045 | — | — |
| SCONE | `patchSingle` | 8 | OK | — | — | 41.628 | 7045 | — | — |
| RTXAdvect | `RTX_BVH` | 1 | OK | — | 100.00 | 1.614 | 406 | — | 100.000 |

## StanfordBunny_LowPoly

| framework | method | omp | status | correct% | found% | wall (s) | RSS (MB) | PIT | tput (Mq/s) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| MALMO | `aabb` | 1 | OK | 100.00 | 100.00 | 0.246 | 1195 | 298.5 | 0.406 |
| MALMO | `centroid` | 1 | OK | 100.00 | 100.00 | 0.247 | 1195 | 298.5 | 0.405 |
| MALMO | `vertex_multi` | 1 | OK | 96.12 | 96.12 | 4.911 | 1194 | 227.9 | 0.020 |
| SCONE | `none` | 1 | OK | — | — | 5.018 | 8 | — | — |
| SCONE | `none` | 8 | OK | — | — | 0.741 | 8 | — | — |
| SCONE | `octree` | 1 | OK | — | — | 0.436 | 13 | — | — |
| SCONE | `octree` | 8 | OK | — | — | 0.248 | 13 | — | — |
| SCONE | `patchMulti` | 1 | FAIL(ec=1) | — | — | 0.017 | 16 | — | — |
| SCONE | `patchMulti` | 8 | FAIL(ec=1) | — | — | 0.018 | 16 | — | — |
| SCONE | `patchSingle` | 1 | FAIL(ec=2) | — | — | 0.017 | 16 | — | — |
| SCONE | `patchSingle` | 8 | FAIL(ec=2) | — | — | 0.018 | 15 | — | — |
| RTXAdvect | `RTX_BVH` | 1 | OK | — | 99.9999 | 2.302 | — | — | 100.000 |

## FSW_paper

| framework | method | omp | status | correct% | found% | wall (s) | RSS (MB) | PIT | tput (Mq/s) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| MALMO | `aabb` | 1 | OK | 100.00 | 100.00 | 1.600 | 2772 | 115.5 | 0.062 |
| MALMO | `centroid` | 1 | OK | 100.00 | 100.00 | 2.426 | 2538 | 78.8 | 0.041 |
| MALMO | `vertex_multi` | 1 | OK | 100.00 | 100.00 | 6.366 | 3229 | 124.3 | 0.016 |
| SCONE | `none` | 1 | FAIL(ec=1) | — | — | 43.338 | 17277 | — | — |
| SCONE | `none` | 8 | FAIL(ec=1) | — | — | 44.789 | 17277 | — | — |
| SCONE | `octree` | 1 | FAIL(ec=1) | — | — | 16314.671 | 19084 | — | — |
| SCONE | `octree` | 8 | FAIL(ec=1) | — | — | 16315.866 | 19084 | — | — |
| SCONE | `patchMulti` | 1 | FAIL(ec=1) | — | — | 34.480 | 17678 | — | — |
| SCONE | `patchMulti` | 8 | FAIL(ec=1) | — | — | 34.767 | 17679 | — | — |
| SCONE | `patchSingle` | 1 | FAIL(ec=1) | — | — | 34.165 | 17285 | — | — |
| SCONE | `patchSingle` | 8 | FAIL(ec=1) | — | — | 34.179 | 17285 | — | — |
| RTXAdvect | `RTX_BVH` | 1 | OK | — | 99.05 | 7.071 | — | — | 6.667 |
