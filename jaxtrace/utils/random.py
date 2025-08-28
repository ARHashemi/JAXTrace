from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np

# Guarded JAX import following the codebase’s optional pattern[^8,^9]
try:
    import jax
    import jax.numpy as jnp
    import jax.random as jrand
    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False
    jax = None  # type: ignore
    jnp = None  # type: ignore
    jrand = None  # type: ignore


# -------------------------
# Keys and splitting
# -------------------------

def rng_key(seed: Optional[int] = None):
    """
    Create a RNG handle:
    - JAX: returns PRNGKey(seed or 0)
    - NumPy: returns Generator(PCG64(seed or 0))
    """
    s = int(seed) if seed is not None else 0
    if JAX_AVAILABLE:
        return jrand.PRNGKey(s)
    return np.random.default_rng(s)

def split_keys(key, num: int):
    """
    Split a RNG handle:
    - JAX: returns array/list of keys
    - NumPy: returns list of new Generators using SeedSequence
    """
    num = int(num)
    if JAX_AVAILABLE:
        return jrand.split(key, num)
    ss = np.random.SeedSequence(np.random.PCG64(key).state["state"] if hasattr(key, "bit_generator") else int(np.random.SeedSequence().entropy))
    return [np.random.default_rng(s) for s in ss.spawn(num)]


# -------------------------
# Distributions
# -------------------------

def uniform(key, shape: Sequence[int], *, low: float = 0.0, high: float = 1.0, dtype=None):
    if JAX_AVAILABLE:
        return jrand.uniform(key, shape=tuple(shape), minval=low, maxval=high, dtype=dtype or jnp.float32)
    rng = key if hasattr(key, "random") else np.random.default_rng()
    return rng.uniform(low=low, high=high, size=tuple(shape)).astype(dtype or np.float32)

def normal(key, shape: Sequence[int], *, mean: float = 0.0, std: float = 1.0, dtype=None):
    if JAX_AVAILABLE:
        return mean + std * jrand.normal(key, shape=tuple(shape), dtype=dtype or jnp.float32)
    rng = key if hasattr(key, "random") else np.random.default_rng()
    return (mean + std * rng.standard_normal(size=tuple(shape))).astype(dtype or np.float32)

def shuffle(key, x):
    """
    Shuffle along the first axis.
    """
    x = np.asarray(x)
    if JAX_AVAILABLE:
        idx = jrand.permutation(key, x.shape[0])
        return np.asarray(jnp.asarray(x)[idx])
    rng = key if hasattr(key, "random") else np.random.default_rng()
    y = x.copy()
    rng.shuffle(y, axis=0)
    return y
