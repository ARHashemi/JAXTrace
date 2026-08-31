# Table T5 · Peak resident memory (methods × meshes)

Peak resident set size (RSS) recorded by `/usr/bin/time -v` for the whole process (mesh load + acceleration structure build + query cost).  Includes MALMO's constant ~1.2 GB JAX/CUDA runtime allocation.  RTXAdvect memory was not captured on the sweep runs (the driver script did not wrap the process in `time -v`) and is reported as `—`; the process footprint is dominated by mesh + BVH storage, empirically ~400 MB on Kim meshes and ~1.2 GB on FSW per the smoke test.

| method | Tet63<br>(63) | Tet137<br>(137) | Tet298<br>(298) | Bunny<br>(379) | Poly264<br>(1.4k) | Tet1820<br>(1.8k) | Tet2856<br>(2.9k) | Poly940<br>(5.2k) | Poly1560<br>(8.7k) | FSW<br>(3.05M) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MALMO `aabb` | 1.17 GB | 1.17 GB | 1.17 GB | 1.17 GB | 1.17 GB | 1.17 GB | 1.17 GB | 1.17 GB | 1.17 GB | 2.71 GB |
| MALMO `centroid` | 1.17 GB | 1.17 GB | 1.17 GB | 1.17 GB | 1.17 GB | 1.17 GB | 1.17 GB | 1.17 GB | 1.17 GB | 2.48 GB |
| MALMO `vertex_multi` | 1.17 GB | 1.17 GB | 1.17 GB | 1.17 GB | 1.17 GB | 1.17 GB | 1.17 GB | 1.17 GB | 1.18 GB | 3.15 GB |
| SCONE `none` (OMP=1) | 6 MB | 6 MB | 8 MB | 8 MB | 8 MB | 16 MB | 22 MB | 15 MB | 20 MB | FAIL |
| SCONE `none` (OMP=8) | 6 MB | 7 MB | 7 MB | 8 MB | 8 MB | 16 MB | 22 MB | 15 MB | 20 MB | FAIL |
| SCONE `octree` (OMP=1) | 7 MB | 8 MB | 11 MB | 13 MB | FAIL | 34 MB | 49 MB | 25 MB | 39 MB | FAIL |
| SCONE `octree` (OMP=8) | 7 MB | 8 MB | 11 MB | 13 MB | 11 MB | 34 MB | 49 MB | 24 MB | 38 MB | FAIL |
| SCONE `patchSingle` (OMP=1) | FAIL | 4.72 GB | 2.03 GB | FAIL | 2.16 GB | 15.46 GB | FAIL | 3.82 GB | 6.88 GB | FAIL |
| SCONE `patchSingle` (OMP=8) | FAIL | 4.72 GB | 2.03 GB | FAIL | 2.16 GB | 15.46 GB | FAIL | 3.82 GB | 6.88 GB | FAIL |
| SCONE `patchMulti` (OMP=1) | SEGV | SEGV | SEGV | FAIL | SEGV | SEGV | SEGV | SEGV | SEGV | FAIL |
| SCONE `patchMulti` (OMP=8) | SEGV | SEGV | SEGV | FAIL | SEGV | SEGV | SEGV | SEGV | SEGV | FAIL |
| RTXAdvect `RTX_BVH` | 405 MB | 405 MB | 405 MB | — | 405 MB | 404 MB | 405 MB | 405 MB | 406 MB | — |

*Status codes:* SCONE's `patchSingle` peaks in the 2–16 GB range on the Kim cohort; SCONE's `patchMulti` needs 10¹²–10⁶ octree cells (peta-byte scale) and SEGVs before building.  MALMO stays under 3.3 GB even on the 3.05 M-tet FSW mesh.
