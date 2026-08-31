# Table T2 · Location correctness (methods × meshes)

Location correctness on the R2-6 mesh cohort for the two frameworks that report per-particle hits. MALMO reports `correct%` (marked `*`), the fraction of queries returning the exact ground-truth host tet. RTXAdvect reports `found%`, the fraction of particles that landed in a valid host tet (its BVH inside-test does not identify which one). Both use in-mesh barycentric sampling with a common random seed. Rates at or above 99.99 % (fewer than 1 miss in 10 000) are reported as 100.00 % — those misses are numerical edge cases from points landing exactly on shared faces. SCONE does not report per-particle hits (see Table T1 for its outcomes).

| method | Tet63<br>(63) | Tet137<br>(137) | Tet298<br>(298) | Bunny<br>(379) | Poly264<br>(1.4k) | Tet1820<br>(1.8k) | Tet2856<br>(2.9k) | Poly940<br>(5.2k) | Poly1560<br>(8.7k) | FSW<br>(3.05M) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MALMO `aabb` | 100.00 %* | 100.00 %* | 100.00 %* | 100.00 %* | 100.00 %* | 100.00 %* | 100.00 %* | 100.00 %* | 100.00 %* | 100.00 %* |
| MALMO `centroid` | 100.00 %* | 100.00 %* | 100.00 %* | 100.00 %* | 100.00 %* | 99.99 %* | 100.00 %* | 99.64 %* | 100.00 %* | 100.00 %* |
| MALMO `vertex_multi` | 100.00 %* | 100.00 %* | 100.00 %* | 96.12 %* | 100.00 %* | 99.90 %* | 99.97 %* | 100.00 %* | 100.00 %* | 100.00 %* |
| RTXAdvect `RTX_BVH` | 100.00 % | 100.00 % | 100.00 % | 100.00 % | 100.00 % | 100.00 % | 100.00 % | 100.00 % | 100.00 % | 99.05 % |

*Column key:* `correct%*` = fraction of queries returning the exact ground-truth host tet (MALMO only). `found%` = fraction of particles that landed in a valid host tet (RTXAdvect). `SEGV` / `FAIL` = SCONE ran but did not produce a usable tally.
