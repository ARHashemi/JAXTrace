# Table T6 · Batch-size scaling (kernel-only), FinalFuelPinPoly1560

Per-variant query wall time (best of 3 trials, `jax.block_until_ready()` synchronised) on the FinalFuelPinPoly1560 mesh.  Below $N_p \approx 50\,000$ the kernel is launch-latency dominated; above, the GPU transitions into the compute-bound regime where cross-variant comparisons are meaningful.

| variant | $N_p=10,000$ | $N_p=50,000$ | $N_p=100,000$ | $N_p=200,000$ | $N_p=500,000$ |
|---|---:|---:|---:|---:|---:|
| MALMO$^\mathrm{A}$ | 12.6 ms | 62.9 ms | 126.4 ms | 252.9 ms | 630.3 ms |
| MALMO$^\mathrm{C}$ | 23.5 ms | 117.6 ms | 234.9 ms | 469.5 ms | 1176.3 ms |
| MALMO$^\mathrm{V}$ | 19.5 ms | 96.9 ms | 195.5 ms | 395.4 ms | 986.5 ms |


_Throughput (Mq/s) equivalent:_

| variant | $N_p=10,000$ | $N_p=50,000$ | $N_p=100,000$ | $N_p=200,000$ | $N_p=500,000$ |
|---|---:|---:|---:|---:|---:|
| MALMO$^\mathrm{A}$ | 0.80 Mq/s | 0.79 Mq/s | 0.79 Mq/s | 0.79 Mq/s | 0.79 Mq/s |
| MALMO$^\mathrm{C}$ | 0.43 Mq/s | 0.43 Mq/s | 0.43 Mq/s | 0.43 Mq/s | 0.43 Mq/s |
| MALMO$^\mathrm{V}$ | 0.51 Mq/s | 0.52 Mq/s | 0.51 Mq/s | 0.51 Mq/s | 0.51 Mq/s |
