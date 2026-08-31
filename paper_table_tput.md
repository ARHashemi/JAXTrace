# Table T4 · Fair per-batch query throughput (methods × meshes)

Per-batch throughput, computed uniformly as **`n_points_queried / query_seconds`** — MALMO's 100 k-particle batches and RTXAdvect's 1 M-particle batches divided by their respective query wall-times.  RTXAdvect's own log convention reports particle-steps per second (inflating by n_steps=100), which we have de-normalised here so the units match MALMO's query batch.  SCONE does not run per-particle queries — its ray-tracing population is a Monte-Carlo tally sampler — so `throughput` is not defined and cells show `—`.

| method | Tet63<br>(63) | Tet137<br>(137) | Tet298<br>(298) | Bunny<br>(379) | Poly264<br>(1.4k) | Tet1820<br>(1.8k) | Tet2856<br>(2.9k) | Poly940<br>(5.2k) | Poly1560<br>(8.7k) | FSW<br>(3.05M) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MALMO `aabb` | 0.68 Mq/s | 0.23 Mq/s | 0.13 Mq/s | 0.41 Mq/s | 0.36 Mq/s | 62.0 kq/s | 74.1 kq/s | 68.7 kq/s | 0.13 Mq/s | 62.5 kq/s |
| MALMO `centroid` | 0.94 Mq/s | 0.10 Mq/s | 0.11 Mq/s | 0.41 Mq/s | 0.28 Mq/s | 68.5 kq/s | 67.0 kq/s | 59.0 kq/s | 61.5 kq/s | 41.2 kq/s |
| MALMO `vertex_multi` | 1.15 Mq/s | 0.25 Mq/s | 0.11 Mq/s | 20.4 kq/s | 0.52 Mq/s | 68.6 kq/s | 71.0 kq/s | 73.4 kq/s | 78.5 kq/s | 15.7 kq/s |
| RTXAdvect `RTX_BVH` | 1.44 Mq/s | 1.45 Mq/s | 1.42 Mq/s | 1.12 Mq/s | 1.43 Mq/s | 1.44 Mq/s | 1.42 Mq/s | 1.42 Mq/s | 1.42 Mq/s | 0.94 Mq/s |

*Column key:* Mq/s = million queries per second per batch. For MALMO on FSW the batch is 100 k queries; for RTXAdvect 1 M particles per BVH-traversal step.  Higher is better.
