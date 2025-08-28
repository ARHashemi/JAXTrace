from __future__ import annotations
import jax.numpy as jnp


def apply_periodic(x: jnp.ndarray, bounds: jnp.ndarray) -> jnp.ndarray:
    """
    Apply periodic wrapping to coordinates x within AABB bounds.

    x: (N,3)
    bounds: (2,3) [min,max]
    """
    lo = bounds[0]
    hi = bounds[1]
    width = jnp.maximum(hi - lo, 0.0)
    # Avoid division by zero when width=0 (degenerate axis): leave that axis unchanged
    w_safe = jnp.where(width > 0, width, 1.0)
    return lo + jnp.mod(x - lo, w_safe)


def periodic_boundary(bounds: jnp.ndarray):
    """
    Factory returning a callable f(x) that applies periodic wrapping within bounds.
    """
    def _f(x: jnp.ndarray) -> jnp.ndarray:
        return apply_periodic(x, bounds)
    return _f


def clamp_to_bounds(x: jnp.ndarray, bounds: jnp.ndarray) -> jnp.ndarray:
    """
    Clamp coordinates to [lo, hi] bounds.
    """
    return jnp.clip(x, a_min=bounds[0], a_max=bounds[1])


def reflect_boundary(x: jnp.ndarray, bounds: jnp.ndarray) -> jnp.ndarray:
    """
    Reflective boundary conditions within [lo,hi] on each axis.
    Uses a simple sawtooth reflection mapping.
    """
    lo = bounds[0]
    hi = bounds[1]
    width = jnp.maximum(hi - lo, 0.0)
    # Map to [0, 2*width]
    y = jnp.mod(x - lo, 2.0 * jnp.where(width > 0, width, 1.0))
    # Reflect back to [0, width]
    y_ref = jnp.where(y <= width, y, 2.0 * width - y)
    return lo + y_ref
