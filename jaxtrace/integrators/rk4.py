# jaxtrace/integrators/rk4.py
"""
Fourth-order Runge-Kutta integration with JAX optimization.

Provides high-accuracy RK4 integration with memory management
and both static/dynamic time step support.
"""

from __future__ import annotations
from typing import Callable, Union

# Import JAX utilities with fallback
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
    """Ensure array is float32 and has proper shape (N,3)."""
    x = jnp.asarray(x, dtype=jnp.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)  # Single particle case
    elif x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"Positions must have shape (N,3), got {x.shape}")
    return x


def _ensure_consistent_types(x: jnp.ndarray, t: float, dt: Union[float, jnp.ndarray]) -> tuple:
    """Ensure all inputs have consistent float32 types."""
    x = _ensure_float32_shape(x)
    t = jnp.asarray(t, dtype=jnp.float32)
    dt = jnp.asarray(dt, dtype=jnp.float32)
    return x, t, dt


if JAX_AVAILABLE:
    @jax.jit
    def _rk4_step_static_jit(
        x: jnp.ndarray,
        t: jnp.ndarray,
        dt_scalar: jnp.ndarray,
        k1: jnp.ndarray,
        k2: jnp.ndarray,
        k3: jnp.ndarray,
        k4: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        JIT-compiled RK4 step for static dt (scalar).
        
        Parameters
        ----------
        x : jnp.ndarray
            Particle positions, shape (N, 3)
        t : jnp.ndarray
            Current time (scalar as array)
        dt_scalar : jnp.ndarray
            Time step (scalar as array)
        k1, k2, k3, k4 : jnp.ndarray
            RK4 slope estimates, each shape (N, 3)
            
        Returns
        -------
        jnp.ndarray
            Updated positions, shape (N, 3)
        """
        return x + dt_scalar / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _rk4_step_dynamic_fallback(
        x: jnp.ndarray,
        t: jnp.ndarray,
        dt_array: jnp.ndarray,
        k1: jnp.ndarray,
        k2: jnp.ndarray,
        k3: jnp.ndarray,
        k4: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Non-JIT RK4 step for dynamic dt (array).
        
        Parameters
        ----------
        x : jnp.ndarray
            Particle positions, shape (N, 3)
        t : jnp.ndarray
            Current time
        dt_array : jnp.ndarray
            Per-particle time steps, shape (N,) or (N, 1)
        k1, k2, k3, k4 : jnp.ndarray
            RK4 slope estimates, each shape (N, 3)
            
        Returns
        -------
        jnp.ndarray
            Updated positions, shape (N, 3)
        """
        dt_expanded = dt_array.reshape(-1, 1) if dt_array.ndim == 1 else dt_array
        return x + dt_expanded / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

else:
    # NumPy fallback implementations
    def _rk4_step_static_jit(x, t, dt_scalar, k1, k2, k3, k4):
        return x + dt_scalar / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
    def _rk4_step_dynamic_fallback(x, t, dt_array, k1, k2, k3, k4):
        dt_expanded = dt_array.reshape(-1, 1) if dt_array.ndim == 1 else dt_array
        return x + dt_expanded / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_step(
    x: jnp.ndarray,
    t: float,
    dt: Union[float, jnp.ndarray],
    field_fn: Callable[[jnp.ndarray, float], jnp.ndarray],
) -> jnp.ndarray:
    """
    Fourth-order Runge-Kutta integration step with high accuracy.

    Classical RK4 method with automatic JIT optimization for scalar time steps.
    Provides excellent accuracy for smooth velocity fields.

    Parameters
    ----------
    x : jnp.ndarray
        Current particle positions, shape (N, 3)
    t : float
        Current time
    dt : float or jnp.ndarray
        Time step size. If scalar, uses JIT version for speed.
    field_fn : Callable[[jnp.ndarray, float], jnp.ndarray]
        Velocity field function returning shape (N, 3)

    Returns
    -------
    jnp.ndarray
        Updated particle positions, shape (N, 3)

    Notes
    -----
    - Requires 4 field evaluations per step (most expensive)
    - Highest accuracy of available methods
    - Automatically uses float32 for JAX performance
    - Memory usage is ~4x higher than Euler method
    """
    # Ensure consistent types and shapes
    x, t_arr, dt_arr = _ensure_consistent_types(x, t, dt)
    t_float = float(t_arr)
    
    # Compute RK4 slope estimates
    if dt_arr.ndim == 0:  # Scalar dt - more efficient computation
        dt_float = float(dt_arr)
        dt_half = dt_float * 0.5
        
        # k1 = v(x_n, t_n)
        k1 = field_fn(x, t_float)
        k1 = _ensure_float32_shape(k1)
        
        # k2 = v(x_n + dt/2 * k1, t_n + dt/2)
        k2 = field_fn(x + dt_half * k1, t_float + dt_half)
        k2 = _ensure_float32_shape(k2)
        
        # k3 = v(x_n + dt/2 * k2, t_n + dt/2)
        k3 = field_fn(x + dt_half * k2, t_float + dt_half)
        k3 = _ensure_float32_shape(k3)
        
        # k4 = v(x_n + dt * k3, t_n + dt)
        k4 = field_fn(x + dt_float * k3, t_float + dt_float)
        k4 = _ensure_float32_shape(k4)
        
        # Final step using JIT version
        return _rk4_step_static_jit(x, t_arr, dt_arr, k1, k2, k3, k4)
        
    else:  # Array dt - more complex handling
        dt_half = dt_arr * 0.5
        dt_half_expanded = dt_half.reshape(-1, 1) if dt_half.ndim == 1 else dt_half
        dt_expanded = dt_arr.reshape(-1, 1) if dt_arr.ndim == 1 else dt_arr
        
        # Use average time steps for temporal increments
        dt_avg = float(jnp.mean(dt_arr))
        dt_half_avg = dt_avg * 0.5
        
        # k1 = v(x_n, t_n)
        k1 = field_fn(x, t_float)
        k1 = _ensure_float32_shape(k1)
        
        # k2 = v(x_n + dt/2 * k1, t_n + dt/2)
        k2 = field_fn(x + dt_half_expanded * k1, t_float + dt_half_avg)
        k2 = _ensure_float32_shape(k2)
        
        # k3 = v(x_n + dt/2 * k2, t_n + dt/2)
        k3 = field_fn(x + dt_half_expanded * k2, t_float + dt_half_avg)
        k3 = _ensure_float32_shape(k3)
        
        # k4 = v(x_n + dt * k3, t_n + dt)
        k4 = field_fn(x + dt_expanded * k3, t_float + dt_avg)
        k4 = _ensure_float32_shape(k4)
        
        # Final step using dynamic version
        return _rk4_step_dynamic_fallback(x, t_arr, dt_arr, k1, k2, k3, k4)


def rk4_step_batch(
    x_batch: jnp.ndarray,
    t: float,
    dt: float,
    field_fn: Callable[[jnp.ndarray, float], jnp.ndarray],
    batch_size: int = 2500,
) -> jnp.ndarray:
    """
    Memory-efficient batched RK4 integration for large particle counts.

    RK4 requires 4x memory compared to Euler, so much smaller default batch size.

    Parameters
    ----------
    x_batch : jnp.ndarray
        Particle positions, shape (N, 3)
    t : float
        Current time
    dt : float
        Time step size
    field_fn : Callable
        Velocity field function
    batch_size : int
        Particles per batch (smallest due to high memory needs)

    Returns
    -------
    jnp.ndarray
        Updated positions, shape (N, 3)
    """
    x_batch = _ensure_float32_shape(x_batch)
    n_particles = x_batch.shape[0]
    
    if n_particles <= batch_size:
        return rk4_step(x_batch, t, dt, field_fn)
    
    # Process in batches
    results = []
    for i in range(0, n_particles, batch_size):
        end_idx = min(i + batch_size, n_particles)
        x_chunk = x_batch[i:end_idx]
        x_next_chunk = rk4_step(x_chunk, t, dt, field_fn)
        results.append(x_next_chunk)
    
    return jnp.concatenate(results, axis=0)