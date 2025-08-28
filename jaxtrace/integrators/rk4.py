from __future__ import annotations
import jax
import jax.numpy as jnp
from .base import FieldFn


@jax.jit
def rk4_step(x: jnp.ndarray, t: float, dt: float | jnp.ndarray, field_fn: FieldFn) -> jnp.ndarray:
    """
    Classical fourth-order Runge–Kutta:
        k1 = v(x, t)
        k2 = v(x + 0.5*dt*k1, t + 0.5*dt)
        k3 = v(x + 0.5*dt*k2, t + 0.5*dt)
        k4 = v(x + dt*k3, t + dt)
        x' = x + (dt/6) * (k1 + 2k2 + 2k3 + k4)

    - Matches your current RK4 pattern using velocity queries at intermediate times,
      which in your previous tracker were implemented via temporal interpolation[^5,^3].
    """
    dt_s = float(jnp.asarray(dt).mean())
    k1 = field_fn(x, t)
    k2 = field_fn(x + 0.5 * dt[..., None] * k1, t + 0.5 * dt_s)
    k3 = field_fn(x + 0.5 * dt[..., None] * k2, t + 0.5 * dt_s)
    k4 = field_fn(x + dt[..., None] * k3, t + dt_s)
    incr = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return x + dt[..., None] * incr
