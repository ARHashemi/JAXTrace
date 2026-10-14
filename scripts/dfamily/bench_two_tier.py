#!/usr/bin/env python3
"""Does a two-launch (two-tier) search actually beat one 8-level kernel?

The idea is NOT a masked branch inside one kernel -- that would execute both
branches for every lane and cost MORE (1 + 8 > 8). It is two separate kernel
launches with a compaction between them:

    elem = fast_kernel(pos)                 # launch 1: few levels, ALL N
    idx  = nonzero(elem < 0)                # compaction (host-visible size)
    elem = elem.at[idx].set(full_kernel(pos[idx]))   # launch 2: 8 levels, M<<N

Each kernel is straight-line code; the selection happens *between* launches,
so no GPU lane ever evaluates a branch it does not need. Total work is
N*1 + M*8 rather than N*8.

The catch this benchmark measures: `nonzero` produces a DYNAMICALLY SHAPED
array, which cannot live inside a jitted function. So the compaction has to
happen at the Python level between two jitted calls, costing a device-host
sync every step. That sync may eat the gain.

Three variants are timed at several miss fractions:
  one_kernel   : the current design, 8 levels over all N
  masked_both  : the naive two-loop-with-mask (expected to be worse)
  two_launch   : compaction between two jitted launches (the real proposal)
"""
import time
import numpy as np
import jax
import jax.numpy as jnp

N = 200_000
TABLE = jnp.asarray(np.random.rand(4_000_000).astype(np.float32))


def _work(x, i):
    idx = (jnp.int32(x * 1000).astype(jnp.int32) + i * 7919) % 4_000_000
    y = TABLE[idx]
    for _ in range(32):
        y = y * 1.0000001 + 1e-9
    return y


def kernel(n_levels):
    """A stand-in for the level loop: n_levels iterations of real work.
    Reports -1 when the lane's host level is deeper than n_levels."""
    def f(pos_hostlevel):
        pos, host = pos_hostlevel
        def body(i, c):
            return _work(c, i)
        val = jax.lax.fori_loop(0, n_levels, body, pos)
        found = host < n_levels
        return jnp.where(found, val, jnp.float32(-1.0))
    return jax.jit(jax.vmap(f))


def masked_both(n_fast, n_full):
    def f(pos_hostlevel):
        pos, host = pos_hostlevel
        fast = jax.lax.fori_loop(0, n_fast, lambda i, c: _work(c, i), pos)
        full = jax.lax.fori_loop(0, n_full, lambda i, c: _work(c, i), pos)
        return jnp.where(host < n_fast, fast, full)
    return jax.jit(jax.vmap(f))


def bench(fn, arg, reps=10):
    fn(arg)[0].block_until_ready() if isinstance(fn(arg), tuple) else fn(arg).block_until_ready()
    t = time.perf_counter()
    for _ in range(reps):
        r = fn(arg)
        (r[0] if isinstance(r, tuple) else r).block_until_ready()
    return (time.perf_counter() - t) / reps


pos = jnp.asarray(np.random.rand(N).astype(np.float32))
k_full = kernel(8)
k_fast = kernel(1)
k_mask = masked_both(1, 8)

print(f"{N:,} lanes; 'miss' = lane needs more than 1 level\n")
print(f"{'miss %':>7} {'one_kernel':>12} {'masked_both':>12} "
      f"{'two_launch':>12} {'speedup':>9}")

for frac in (0.0, 0.01, 0.05, 0.20, 0.50, 1.0):
    host = np.where(np.random.rand(N) < frac, 7, 0).astype(np.int32)
    hj = jnp.asarray(host)
    arg = (pos, hj)

    t_one = bench(lambda a: k_full(a), arg)
    t_mask = bench(lambda a: k_mask(a), arg)

    # two-launch with compaction between jitted calls
    def two_launch(a):
        p, h = a
        out = k_fast((p, h))
        miss = np.asarray(out) < 0          # forces a sync, as it must
        idx = np.flatnonzero(miss)
        if len(idx):
            ij = jnp.asarray(idx)
            sub = k_full((p[ij], h[ij]))
            out = out.at[ij].set(sub)
        return out

    t_two = bench(two_launch, arg)
    print(f"{frac*100:6.0f}% {t_one*1e3:11.3f}ms {t_mask*1e3:11.3f}ms "
          f"{t_two*1e3:11.3f}ms {t_one/t_two:8.2f}x")

print()
print("Reading: masked_both should always be the worst (both loops run).")
print("two_launch wins only when the miss fraction is low AND the")
print("compaction sync is cheaper than the levels it saves.")
