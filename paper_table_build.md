# Table T3 · One-shot build cost (methods × meshes)

Preprocessing time — MALMO's octree build, RTXAdvect's BVH build. This cost is paid once per mesh and amortises across all subsequent query batches; combined with Table T1 it shows the amortisation break-even. For example on the FSW mesh, MALMO `aabb` builds in ~240 s and queries in ~1.6 s, vs RTXAdvect's ~2.5 s build and ~1.07 s query; MALMO wins after ~450 query batches (a single 2 684-step FSW tracking run is 6× past the break-even). SCONE reports no build/query split (its `wall_seconds_outer` is a combined initialise+ray-trace time, tabulated in T1).

| method | Tet63<br>(63) | Tet137<br>(137) | Tet298<br>(298) | Bunny<br>(379) | Poly264<br>(1.4k) | Tet1820<br>(1.8k) | Tet2856<br>(2.9k) | Poly940<br>(5.2k) | Poly1560<br>(8.7k) | FSW<br>(3.05M) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MALMO `aabb` | 5.6 ms | 15.0 ms | 42.7 ms | 20.9 ms | 109.8 ms | 328.0 ms | 566.9 ms | 449.0 ms | 491.2 ms | 4.0 min |
| MALMO `centroid` | 6.1 ms | 15.0 ms | 42.7 ms | 20.0 ms | 116.1 ms | 328.2 ms | 579.5 ms | 459.6 ms | 752.8 ms | 1.4 min |
| MALMO `vertex_multi` | 4.1 ms | 9.2 ms | 19.4 ms | 23.4 ms | 91.2 ms | 117.0 ms | 182.7 ms | 333.9 ms | 554.4 ms | 3.3 min |
| RTXAdvect `RTX_BVH` | 156.8 ms | 158.9 ms | 159.8 ms | 135.5 ms | 158.7 ms | 161.0 ms | 163.6 ms | 162.2 ms | 166.5 ms | 2.51 s |

*Status codes:* `FAIL` / `SEGV` = the run did not complete the build.  `—` = build/query split not reported (SCONE).
