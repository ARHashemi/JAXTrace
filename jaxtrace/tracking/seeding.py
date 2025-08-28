from __future__ import annotations
from typing import Optional

import jax
import jax.numpy as jnp


def random_seeds(n: int, bounds: jnp.ndarray, rng_seed: int = 0, dtype=jnp.float32) -> jnp.ndarray:
    """
    Uniformly sample N seed positions within bounds.

    bounds: (2,3) [min,max]
    """
    key = jax.random.PRNGKey(rng_seed)
    u = jax.random.uniform(key, shape=(n, 3), dtype=dtype)
    lo = jnp.asarray(bounds[0], dtype=dtype)
    hi = jnp.asarray(bounds[1], dtype=dtype)
    return lo + u * (hi - lo)
