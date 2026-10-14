"""If early exit cannot work per-lane, can it work per-BATCH?

The cost of a while_loop under vmap is set by the batch maximum. So if
particles are SORTED so that each batch contains only lanes with similar trip
counts, each batch's maximum falls to near its own mean. That is the standard
GPU answer to divergence: regroup, do not branch.

Tested: one big mixed batch vs several homogeneous sub-batches with the same
total work.
"""
import time
import jax, jax.numpy as jnp
import numpy as np

N = 200_000

def body_work(c):
    x = c
    for _ in range(8):
        x = x * 1.0000001 + 1e-9
    return x

def while_lane(seed_target):
    seed, tgt = seed_target
    def cond(s):
        i, c = s
        return i < tgt
    def bod(s):
        i, c = s
        return i + 1, body_work(c)
    return jax.lax.while_loop(cond, bod, (0, seed))[1]

g = jax.jit(jax.vmap(while_lane))

seeds = np.random.rand(N).astype(np.float32)
# realistic skew: 98% of particles need 2 levels, 2% need up to 8
tgt = np.where(np.random.rand(N) < 0.98, 2, np.random.randint(3, 9, N)).astype(np.int32)
print(f"trip counts: mean {tgt.mean():.2f}  max {tgt.max()}  "
      f"(98% need 2, 2% need 3-8)")
print()

# unsorted: one batch
sj, tj = jnp.asarray(seeds), jnp.asarray(tgt)
g((sj, tj)).block_until_ready()
t = time.perf_counter()
for _ in range(10):
    g((sj, tj)).block_until_ready()
unsorted_t = (time.perf_counter() - t) / 10
print(f"UNSORTED, one batch of {N:,}      : {unsorted_t*1e3:7.2f} ms")

# sorted by trip count, split into homogeneous chunks
order = np.argsort(tgt, kind="stable")
ss, ts = seeds[order], tgt[order]
for nchunk in (2, 4, 8):
    chunks = [(jnp.asarray(ss[a:b]), jnp.asarray(ts[a:b]))
              for a, b in zip(range(0, N, N // nchunk),
                              list(range(N // nchunk, N + 1, N // nchunk)))]
    for c in chunks:
        g(c).block_until_ready()
    t = time.perf_counter()
    for _ in range(10):
        for c in chunks:
            g(c).block_until_ready()
    dt = (time.perf_counter() - t) / 10
    maxes = [int(ts[a:b].max()) for a, b in
             zip(range(0, N, N // nchunk), range(N // nchunk, N + 1, N // nchunk))]
    print(f"SORTED into {nchunk} chunks, maxes {maxes}: {dt*1e3:7.2f} ms  "
          f"({unsorted_t/dt:.2f}x)")
print()
print("NOTE: chunking also costs kernel-launch overhead per chunk, which is")
print("why too many chunks stops helping.")
