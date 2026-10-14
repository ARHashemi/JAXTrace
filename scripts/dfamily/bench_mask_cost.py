"""Does a MASKED fori_loop iteration actually cost anything?

The level loop runs all 8 levels and uses `should_search` to mask the result
once found. If XLA still executes the arithmetic, reordering saves nothing in
the masked iterations -- it only helps if the mask makes the work cheap.

Two regimes compared at the same trip count:
  (a) every iteration does real work
  (b) iterations after the first are masked by a jnp.where on the SAME pattern
      the kernel uses (should_search = not found yet)
If (b) == (a), masking is free of charge in the wrong direction: the work is
still done, so order does not matter. If (b) < (a), the mask does prune.
"""
import time
import jax, jax.numpy as jnp
import numpy as np

N = 200_000
ITERS = 8

def heavy(x):
    # stand-in for 27 cells x bound point-in-tet tests
    for _ in range(64):
        x = x * 1.0000001 + 1e-9
    return x

def always_work(seed):
    def body(i, c):
        return heavy(c)
    return jax.lax.fori_loop(0, ITERS, body, seed)

def masked_after_first(seed_found):
    """found_at tells each lane which iteration 'finds' its host."""
    seed, found_at = seed_found
    def body(i, c):
        should = i <= found_at              # same shape as should_search
        new = heavy(c)
        return jnp.where(should, new, c)    # mask the RESULT, as the kernel does
    return jax.lax.fori_loop(0, ITERS, body, seed)

seeds = jnp.asarray(np.random.rand(N), dtype=jnp.float32)

g1 = jax.jit(jax.vmap(always_work))
g1(seeds).block_until_ready()
t = time.perf_counter()
for _ in range(20):
    g1(seeds).block_until_ready()
a = (time.perf_counter() - t) / 20
print(f"(a) all {ITERS} iterations do real work        : {a*1e3:7.3f} ms")

for label, fa in (("found on iter 0 (best case)", 0),
                  ("found on iter 1 (level 14 2nd)", 1),
                  ("found on iter 7 (worst case)", 7)):
    fj = jnp.full(N, fa, dtype=jnp.int32)
    g2 = jax.jit(jax.vmap(masked_after_first))
    g2((seeds, fj)).block_until_ready()
    t = time.perf_counter()
    for _ in range(20):
        g2((seeds, fj)).block_until_ready()
    b = (time.perf_counter() - t) / 20
    print(f"(b) masked, {label:30}: {b*1e3:7.3f} ms  ({a/b:.2f}x vs (a))")

print()
print("If all (b) rows are the same, the mask does NOT prune work and")
print("reordering the level loop cannot help. If 'found on iter 0' is")
print("much faster, the mask prunes and order matters.")
