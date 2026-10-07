<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Can I get best performance on LUMI with the attached scripts for running a JAX python code? Can the monitoring reduce the performance speed?

Your script is **close to a good production setup**, but it is probably **not yet the best-performance configuration** for LUMI. The monitoring can reduce speed a bit, especially the repeated `find` and `du` on the output directory, while some of your JAX/XLA memory settings are also more conservative than maximal-performance settings.[^1]

## What is already good

You are doing several things right for LUMI: using the tested JAX ROCm container, running on `small-g` with one GPU and one task, keeping active output on `/flash`, and moving results to `/scratch` afterward.  You also set the MIOpen cache in `/tmp`, use `float32`, and pass a clean environment into the container, which are all sensible choices for stable GPU runs.[^1]

## Likely speed limiters

The biggest likely limiter is that you disable JAX preallocation with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and force `XLA_PYTHON_CLIENT_ALLOCATOR=platform`, which often improves memory flexibility but can hurt raw throughput compared with the default pooled allocator.  Another obvious limit is that the job uses only **one GPU** on LUMI-G, so if your code can be parallelized across devices, you are leaving most of the node unused.[^1]

A third possible limiter is `HSA_ENABLE_SDMA=0`, because that disables SDMA engines and can reduce transfer performance when host-device or device-device copies matter.  Your script also writes VTU output every step with `EXPORT_FREQ=1`, and with thousands of steps that can make I/O and metadata overhead dominate the runtime even if the compute kernels are fast.[^1]

## Monitoring overhead

The monitoring itself is **not catastrophic**, but it is not free.  `rocm-smi` and `free -h` every 30 seconds are usually minor overheads, but the repeated `find $FLASH_OUT -name '*.vtu'` and `du -sh $FLASH_OUT` can become noticeably expensive when the directory contains many files and the simulation is exporting frequently.[^1]

If you want the cleanest timing runs, disable the monitor entirely or at least remove the `find` and `du` calls.  A good compromise is to keep GPU monitoring but log file count and disk usage only every 10–30 minutes, or only once after the run ends.[^1]

## Best changes to try

For a pure performance test, I would first change only these settings:

- Set `XLA_PYTHON_CLIENT_PREALLOCATE=true` or simply remove both XLA memory lines and use JAX defaults.[^1]
- Temporarily remove `XLA_PYTHON_CLIENT_ALLOCATOR=platform`.[^1]
- Temporarily remove `HSA_ENABLE_SDMA=0` and benchmark both ways.[^1]
- Set `EXPORT_FREQ` much larger, or use `NO_EXPORT=1` for compute-only timing.[^1]
- Disable the background monitor during benchmarking, especially the `find` and `du` parts.[^1]

I would keep `float32`, `/flash` output, and the `/tmp` MIOpen cache as they are.  If your code is single-GPU by design, `small-g` is fine; if it can shard work, the next big step is scaling beyond one GPU rather than tuning shell settings.[^1]

## Recommended benchmark setup

Use two benchmark runs with the same input: one **compute-only** run with `NO_EXPORT=1` and no monitor, and one **production-like** run with exports enabled.  That split will tell you whether your bottleneck is compute, memory allocation, or output/monitoring overhead.[^1]

My short recommendation is: keep the script structure, but for speed testing remove the monitor, reduce export frequency, and stop overriding JAX memory allocation unless you actually need those safeguards.  If you want, I can rewrite `run_lumi.sh` into a **benchmark version** and a **production version** side by side.[^1]
<span style="display:none">[^2][^3][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: run_lumi.sh

[^2]: Numerical Methods in Fluids - 2008 - Lu - Weighted essential non‐oscillatory schemes for tidal bore on unstructured meshes.pdf

[^3]: 1-s2.0-S0021999125002311-main.pdf

[^4]: Daum_DA_2004.pdf

[^5]: 1-s2.0-S0743731521001337-main.pdf

[^6]: karras2012hpg_paper.pdf

[^7]: 2492045.2492055.pdf

[^8]: Computer Graphics Forum - 2003 - Kipfer - Local Exact Particle Tracing on Unstructured Grids.pdf

[^9]: Communications in Numerical Methods in Engineering - 2008 - Macpherson - Particle tracking in unstructured arbitrary.pdf

