from __future__ import annotations
import jax
import jax.numpy as jnp
from .base import FieldFn


@jax.jit
def euler_step(x: jnp.ndarray, t: float, dt: float | jnp.ndarray, field_fn: FieldFn) -> jnp.ndarray:
    """
    Explicit Euler step:
        x_{n+1} = x_n + dt * v(x_n, t_n)

    - x: (N,3)
    - t: scalar
    - dt: scalar or (N,)
    - field_fn: (x, t) -> (N,3)
    """
    v = field_fn(x, t)
    dtv = dt[..., None] * v
    return x + dtv
