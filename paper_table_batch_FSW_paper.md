# Table T6 · Batch-size scaling (kernel-only), FSW_paper

Per-variant query wall time (best of 3 trials, `jax.block_until_ready()` synchronised) on the FSW_paper mesh.  Below $N_p \approx 50\,000$ the kernel is launch-latency dominated; above, the GPU transitions into the compute-bound regime where cross-variant comparisons are meaningful.

| variant | $N_p=10,000$ | $N_p=50,000$ | $N_p=100,000$ | $N_p=200,000$ | $N_p=500,000$ |
|---|---:|---:|---:|---:|---:|
| MALMO$^\mathrm{A}$ | 119.2 ms | 615.7 ms | 1253.6 ms | 2479.5 ms | 6236.2 ms |
| MALMO$^\mathrm{C}$ | 51.1 ms | 256.5 ms | 513.3 ms | 1020.0 ms | 2561.4 ms |
| MALMO$^\mathrm{V}$ | 110.7 ms | 551.4 ms | 1107.7 ms | 2212.9 ms | 5542.4 ms |


_Throughput (Mq/s) equivalent:_

| variant | $N_p=10,000$ | $N_p=50,000$ | $N_p=100,000$ | $N_p=200,000$ | $N_p=500,000$ |
|---|---:|---:|---:|---:|---:|
| MALMO$^\mathrm{A}$ | 0.08 Mq/s | 0.08 Mq/s | 0.08 Mq/s | 0.08 Mq/s | 0.08 Mq/s |
| MALMO$^\mathrm{C}$ | 0.20 Mq/s | 0.19 Mq/s | 0.19 Mq/s | 0.20 Mq/s | 0.20 Mq/s |
| MALMO$^\mathrm{V}$ | 0.09 Mq/s | 0.09 Mq/s | 0.09 Mq/s | 0.09 Mq/s | 0.09 Mq/s |
