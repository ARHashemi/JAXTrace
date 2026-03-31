<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# The code on lumi is so slower than our workstation with nvidia 5000. Is there any way to improve the code speed?

This is a common surprise with AMD MI250X vs NVIDIA. Let's diagnose and fix systematically.

***

## Why LUMI Can Feel Slower Initially

The MI250X is theoretically more powerful than an RTX 5000, but several common issues cause poor performance out of the box:


| Issue | Impact | Fix |
| :-- | :-- | :-- |
| **MIOpen kernel compilation** | First run is 10–30× slower (compiling GPU kernels) | Persistent kernel cache |
| **JAX JIT compilation** | First call compiles, subsequent calls are fast | Warm-up runs |
| **Wrong data types** | float64 is ~16× slower than float32 on GPU | Use float32 where possible |
| **Small problem size** | GPU underutilized, overhead dominates | Scale up batch/problem size |
| **Single GCD vs full node** | You're using 1 of 8 available GPU dies | Use all 8 GCDs |
| **Host↔GPU data transfers** | Unnecessary `.block_until_ready()` or `np` conversions | Keep data on GPU |


***

## Step 1 — Fix MIOpen Kernel Cache (Biggest Win)

MIOpen recompiles kernels **every run** unless you give it a persistent cache directory on a fast filesystem. Add this to your Slurm script:

```bash
# Persistent MIOpen cache on flash (survives between jobs)
export MIOPEN_USER_DB_PATH="/flash/project_465002752/hashemia/miopen_cache"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH
```

First run will still be slow while it compiles. Every subsequent run with the same operations will be **dramatically faster** (often 5–10× speedup on first-call latency).

***

## Step 2 — Profile Where Time Is Actually Spent

Add this to your script to measure real GPU vs CPU time:

```python
import jax
import jax.numpy as jnp
import time

# Always block_until_ready() when timing GPU code
# otherwise you measure dispatch time, not compute time

def benchmark(fn, *args, n_warmup=3, n_runs=10):
    # Warmup — triggers JIT compilation
    for _ in range(n_warmup):
        result = fn(*args)
        result.block_until_ready()

    # Actual timing
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args)
        result.block_until_ready()
        times.append(time.perf_counter() - t0)

    print(f"  Mean: {1000*sum(times)/len(times):.2f} ms")
    print(f"  Min:  {1000*min(times):.2f} ms")
    return result
```


***

## Step 3 — Use float32 Instead of float64

Your `jaxtrace.config` sets `jax_enable_x64=True`. The MI250X (like most GPUs) runs float32 at **2× the throughput** of float64. Check if you truly need float64:

```python
# In your config or script:
import jax
# Only enable x64 where precision is critical
# For most ML/tracing operations float32 is sufficient

# Cast inputs explicitly:
x = x.astype(jnp.float32)
```

If your physics requires float64, keep it — but be aware of the performance cost.

***

## Step 4 — Use All 8 GCDs (8 GPU Dies)

You've been testing with `--gpus-per-node=1`. The MI250X node has **8 GCDs** — using all 8 gives ~8× throughput for parallelizable work:

```python
# In your script, shard computation across all devices
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

devices = jax.devices()
print(f"Using {len(devices)} devices")  # Should print 8

sharding = PositionalSharding(devices)
# Shard your arrays:
x_sharded = jax.device_put(x, sharding.reshape(-1, 1))
```

Update your Slurm script to use all 8:

```bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7   # 7 CPU cores per GCD
```


***

## Step 5 — Avoid Python Loops, Use JAX Primitives

Common performance killers:

```python
# ❌ Slow — Python loop with GPU calls
for i in range(1000):
    x = jnp.dot(x, W)

# ✅ Fast — JAX scan (single compiled loop on GPU)
def step(x, _):
    return jnp.dot(x, W), None

x_final, _ = jax.lax.scan(step, x_init, None, length=1000)
```

```python
# ❌ Slow — converts to numpy (GPU→CPU transfer)
for val in np.array(jax_array):
    process(val)

# ✅ Fast — stay on GPU
result = jax.vmap(process)(jax_array)
```


***

## Step 6 — Use JAX Profiler to Find Bottlenecks

```python
import jax

# Profile a section of your code
with jax.profiler.trace("/tmp/jax_trace", create_perfetto_link=True):
    result = your_function(inputs)
    result.block_until_ready()
```

Then copy the trace to your local machine and open it in [ui.perfetto.dev](https://ui.perfetto.dev) to see exactly where time is spent.

***

## Quick Diagnostic Test

Run this on LUMI to get a baseline and compare with your workstation:

```bash
srun --account=project_465002752 --partition=small-g \
     --nodes=1 --gpus-per-node=8 --time=00:15:00 \
     singularity exec --cleanenv \
     --env PYTHONPATH=$JAXTRACE:$PKGS \
     $SIF \
     python -c "
import jax
import jax.numpy as jnp
import time

print('Devices:', jax.devices())

# Warmup
x = jnp.ones((4096, 4096))
jnp.dot(x, x).block_until_ready()

# Benchmark
times = []
for _ in range(20):
    t0 = time.perf_counter()
    jnp.dot(x, x).block_until_ready()
    times.append(time.perf_counter() - t0)

print(f'matmul 4096x4096: {1000*min(times):.2f} ms (best)')
print(f'matmul 4096x4096: {1000*sum(times)/len(times):.2f} ms (mean)')
"
```

Paste the output and tell me what your **specific slow operation is** (time-stepping loop, mesh operations, I/O?) and I can give you targeted optimizations for JAXTrace.

