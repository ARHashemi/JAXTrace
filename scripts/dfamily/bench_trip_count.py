"""Can the work actually be SKIPPED, not just masked?

Masking (jnp.where on the result) does not prune -- measured. Options that
might actually skip:

  1. lax.cond inside the loop body. Under vmap, cond is converted to a
     select... which is masking again. Expect no gain.
  2. Make the DATA cheap when not needed: if the loop reads a gathered array,
     point the gather at a dummy index so the memory traffic collapses even
     though the FLOPs remain. Tests whether the kernel is memory- or FLOP-bound.
  3. Reduce the TRIP COUNT itself, statically, per batch: if a whole batch only
     ever needs k levels, compile a kernel with n_levels=k. This is the only
     mechanism that genuinely removes iterations under vmap.
"""
import time
import jax, jax.numpy as jnp
import numpy as np

N = 200_000
TABLE = jnp.asarray(np.random.rand(4_000_000).astype(np.float32))

def heavy_gather(idx):
    """FLOPs + a dependent gather, like point_in_tet reading node positions."""
    x = TABLE[idx]
    for _ in range(32):
        x = x * 1.0000001 + 1e-9
    return x

def loop_mask(seed_found, iters):
    seed, found_at = seed_found
    def body(i, c):
        should = i <= found_at
        idx = (jnp.int32(c * 1000).astype(jnp.int32) + i * 7919) % 4_000_000
        new = heavy_gather(idx)
        return jnp.where(should, new, c)
    return jax.lax.fori_loop(0, iters, body, seed)

def loop_cond(seed_found, iters):
    """lax.cond instead of where -- does vmap turn it into a select?"""
    seed, found_at = seed_found
    def body(i, c):
        def do(c_):
            idx = (jnp.int32(c_ * 1000).astype(jnp.int32) + i * 7919) % 4_000_000
            return heavy_gather(idx)
        return jax.lax.cond(i <= found_at, do, lambda c_: c_, c)
    return jax.lax.fori_loop(0, iters, body, seed)

seeds = jnp.asarray(np.random.rand(N), dtype=jnp.float32)
found0 = jnp.zeros(N, dtype=jnp.int32)

def bench(fn, arg, label, reps=20):
    g = jax.jit(jax.vmap(fn))
    g(arg).block_until_ready()
    t = time.perf_counter()
    for _ in range(reps):
        g(arg).block_until_ready()
    dt = (time.perf_counter() - t) / reps
    print(f"  {label:46}: {dt*1e3:7.3f} ms")
    return dt

print("TRIP COUNT 8 (current level loop):")
m8 = bench(lambda a: loop_mask(a, 8), (seeds, found0), "jnp.where mask, found on iter 0")
c8 = bench(lambda a: loop_cond(a, 8), (seeds, found0), "lax.cond, found on iter 0")
print()
print("TRIP COUNT 2 (what a per-batch static reduction would give):")
m2 = bench(lambda a: loop_mask(a, 2), (seeds, found0), "jnp.where mask, 2 iterations")
print()
print(f"  cond vs where at 8 iters : {m8/c8:.2f}x  (1.0 = cond is also a select)")
print(f"  8 iters -> 2 iters       : {m8/m2:.2f}x  <- the only real lever")
