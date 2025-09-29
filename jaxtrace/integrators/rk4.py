# jaxtrace/integrators/rk4.py

from __future__ import annotations
from typing import Callable, Union
from ..utils.jax_utils import JAX_AVAILABLE

if JAX_AVAILABLE:
    try:
        import jax
        import jax.numpy as jnp
    except Exception:
        JAX_AVAILABLE = False

if not JAX_AVAILABLE:
    import numpy as jnp  # type: ignore


def _ensure_float32_shape(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=jnp.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"Positions must have shape (N,3), got {x.shape}")
    return x


def _ensure_consistent_types(x: jnp.ndarray, t: float, dt: Union[float, jnp.ndarray]) -> tuple:
    x = _ensure_float32_shape(x)
    t = jnp.asarray(t, dtype=jnp.float32)
    dt = jnp.asarray(dt, dtype=jnp.float32)
    return x, t, dt


if JAX_AVAILABLE:
    @jax.jit
    def _rk4_combine_static(
        x: jnp.ndarray,
        dt_scalar: jnp.ndarray,
        k1: jnp.ndarray,
        k2: jnp.ndarray,
        k3: jnp.ndarray,
        k4: jnp.ndarray,
    ) -> jnp.ndarray:
        return x + dt_scalar / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
else:
    def _rk4_combine_static(x, dt_scalar, k1, k2, k3, k4):
        return x + dt_scalar / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _rk4_combine_dynamic(
    x: jnp.ndarray,
    dt_array: jnp.ndarray,
    k1: jnp.ndarray,
    k2: jnp.ndarray,
    k3: jnp.ndarray,
    k4: jnp.ndarray,
) -> jnp.ndarray:
    dt_expanded = dt_array.reshape(-1, 1) if dt_array.ndim == 1 else dt_array
    return x + dt_expanded / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_step(
    x: jnp.ndarray,
    t: float,
    dt: Union[float, jnp.ndarray],
    field_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """
    Runge-Kutta 4 integrator.

    Parameters
    ----------
    x : (N, 3) positions
    t : scalar time (Python float or JAX 0-D array)
    dt : scalar or per-particle array
    field_fn : callable(positions, time) -> velocities, JAX-compatible

    Returns
    -------
    x_next : (N, 3) next positions
    """
    x, t_j, dt_j = _ensure_consistent_types(x, t, dt)

    # k1 at (x, t)
    k1 = field_fn(x, t_j)
    k1 = _ensure_float32_shape(k1)

    if dt_j.ndim == 0:
        # Scalar step size: do exact RK4 times
        dt_half = 0.5 * dt_j
        t_half = t_j + dt_half
        t_full = t_j + dt_j

        x2 = x + dt_half * k1
        k2 = field_fn(x2, t_half)
        k2 = _ensure_float32_shape(k2)

        x3 = x + dt_half * k2
        k3 = field_fn(x3, t_half)
        k3 = _ensure_float32_shape(k3)

        x4 = x + dt_j * k3
        k4 = field_fn(x4, t_full)
        k4 = _ensure_float32_shape(k4)

        return _rk4_combine_static(x, dt_j, k1, k2, k3, k4)

    else:
        # Per-particle dt: use per-particle positions, but a scalar time
        # derived from the mean step size to keep field_fn signature scalar in time.
        dt_expanded = dt_j.reshape(-1, 1) if dt_j.ndim == 1 else dt_j
        dt_half_expanded = 0.5 * dt_expanded

        dt_mean = jnp.mean(dt_j)
        t_half = t_j + 0.5 * dt_mean
        t_full = t_j + dt_mean

        x2 = x + dt_half_expanded * k1
        k2 = field_fn(x2, t_half)
        k2 = _ensure_float32_shape(k2)

        x3 = x + dt_half_expanded * k2
        k3 = field_fn(x3, t_half)
        k3 = _ensure_float32_shape(k3)

        x4 = x + dt_expanded * k3
        k4 = field_fn(x4, t_full)
        k4 = _ensure_float32_shape(k4)

        return _rk4_combine_dynamic(x, dt_j, k1, k2, k3, k4)