"""Can a while_loop under vmap+jit actually exit early, or does it run to the
batch maximum? Measured, not argued."""
import time
import jax, jax.numpy as jnp
import numpy as np

N = 200_000

def body_work(c):
    # non-trivial arithmetic so the iteration cost is visible
    x = c
    for _ in range(8):
        x = x * 1.0000001 + 1e-9
    return x

def fori_fixed(n_iter):
    def f(seed):
        return jax.lax.fori_loop(0, n_iter, lambda i, c: body_work(c), seed)
    return f

def while_earlyexit(target):
    """Exit as soon as the counter reaches this lane's own target."""
    def f(seed_target):
        seed, tgt = seed_target
        def cond(s):
            i, c = s
            return i < tgt
        def bod(s):
            i, c = s
            return i + 1, body_work(c)
        return jax.lax.while_loop(cond, bod, (0, seed))[1]
    return f

seeds = jnp.asarray(np.random.rand(N), dtype=jnp.float32)

print("A  fixed-trip fori_loop, cost vs trip count")
for n in (1, 8, 64):
    g = jax.jit(jax.vmap(fori_fixed(n)))
    g(seeds).block_until_ready()
    t = time.perf_counter()
    for _ in range(5):
        g(seeds).block_until_ready()
    dt = (time.perf_counter() - t) / 5
    print(f"   {n:3d} iterations: {dt*1e3:8.2f} ms")

print()
print("B  while_loop with PER-LANE targets under vmap")
print("   if early exit works, cost tracks the MEAN target;")
print("   if not, it tracks the MAX target.")
for label, tgt in (("all lanes 1",        np.ones(N)),
                   ("all lanes 64",       np.full(N, 64)),
                   ("1 lane 64, rest 1",  np.concatenate([[64], np.ones(N-1)])),
                   ("half 1 half 64",     np.where(np.arange(N) % 2 == 0, 1, 64))):
    tj = jnp.asarray(tgt, dtype=jnp.int32)
    g = jax.jit(jax.vmap(while_earlyexit(None)))
    g((seeds, tj)).block_until_ready()
    t = time.perf_counter()
    for _ in range(5):
        g((seeds, tj)).block_until_ready()
    dt = (time.perf_counter() - t) / 5
    print(f"   {label:18} mean={tgt.mean():5.1f} max={tgt.max():3.0f}  ->  {dt*1e3:8.2f} ms")
