from __future__ import annotations
import jax
import jax.numpy as jnp
from .base import FieldFn


@jax.jit
def rk2_step(x: jnp.ndarray, t: float, dt: float | jnp.ndarray, field_fn: FieldFn) -> jnp.ndarray:
    """
    Second-order Runge–Kutta (midpoint/Heun):
        k1 = v(x, t)
        k2 = v(x + 0.5*dt*k1, t + 0.5*dt)
        x' = x + dt * k2

    - Delegates temporal interpolation to field_fn(x, t), consistent with your
      existing approach to blend velocities between time slices for intermediate
      sub-steps[^5].
    """
    k1 = field_fn(x, t)
    x2 = x + 0.5 * dt[..., None] * k1
    k2 = field_fn(x2, t + 0.5 * float(jnp.asarray(dt).mean()))
    return x + dt[..., None] * k2
