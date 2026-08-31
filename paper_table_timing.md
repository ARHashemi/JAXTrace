# Table T1 · Query wall time (methods × meshes)

Per-batch query wall time. For MALMO, best of three trials. For SCONE, `wall_seconds_outer` (the tool has no explicit build/query split). For RTXAdvect, `Simulation RunTime` (post-BVH-build). Build times (octree / patch grid / BVH) are reported separately in Table T3.

| method | Tet63<br>(63) | Tet137<br>(137) | Tet298<br>(298) | Bunny<br>(379) | Poly264<br>(1.4k) | Tet1820<br>(1.8k) | Tet2856<br>(2.9k) | Poly940<br>(5.2k) | Poly1560<br>(8.7k) | FSW<br>(3.05M) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MALMO `aabb` | 148.1 ms | 430.7 ms | 778.9 ms | 246.3 ms | 280.5 ms | 1.61 s | 1.35 s | 1.45 s | 797.0 ms | 1.60 s |
| MALMO `centroid` | 106.2 ms | 983.5 ms | 934.0 ms | 246.7 ms | 358.9 ms | 1.46 s | 1.49 s | 1.70 s | 1.63 s | 2.43 s |
| MALMO `vertex_multi` | 86.8 ms | 400.4 ms | 892.1 ms | 4.91 s | 191.5 ms | 1.46 s | 1.41 s | 1.36 s | 1.27 s | 6.37 s |
| SCONE `none` (OMP=1) | 2.15 s | 4.42 s | 10.19 s | 5.02 s | 7.68 s | 1.5 min | 2.5 min | 25.22 s | 46.79 s | FAIL |
| SCONE `none` (OMP=8) | 343.8 ms | 656.4 ms | 1.53 s | 740.9 ms | 1.10 s | 13.25 s | 23.68 s | 3.75 s | 6.90 s | FAIL |
| SCONE `octree` (OMP=1) | 509.6 ms | 595.0 ms | 759.3 ms | 436.4 ms | FAIL | 2.32 s | 3.71 s | 1.08 s | 1.72 s | FAIL |
| SCONE `octree` (OMP=8) | 138.3 ms | 178.0 ms | 240.7 ms | 247.6 ms | 232.9 ms | 1.21 s | 2.17 s | 445.8 ms | 888.3 ms | FAIL |
| SCONE `patchSingle` (OMP=1) | FAIL | 1.2 min | 29.19 s | FAIL | 10.68 s | 4.9 min | FAIL | 22.57 s | 42.42 s | FAIL |
| SCONE `patchSingle` (OMP=8) | FAIL | 1.2 min | 28.62 s | FAIL | 10.26 s | 4.9 min | FAIL | 21.93 s | 41.63 s | FAIL |
| SCONE `patchMulti` (OMP=1) | SEGV | SEGV | SEGV | FAIL | SEGV | SEGV | SEGV | SEGV | SEGV | FAIL |
| SCONE `patchMulti` (OMP=8) | SEGV | SEGV | SEGV | FAIL | SEGV | SEGV | SEGV | SEGV | SEGV | FAIL |
| RTXAdvect `RTX_BVH` | 696.8 ms | 692.0 ms | 706.5 ms | 896.0 ms | 697.3 ms | 696.6 ms | 702.8 ms | 703.4 ms | 706.0 ms | 1.07 s |

*Status codes:* `SEGV` = crashed at build (SCONE only), `FAIL` = ran but exited with a non-zero code, `TIMED OUT` = hit the 1-h per-mesh timeout, `—` = not available.
