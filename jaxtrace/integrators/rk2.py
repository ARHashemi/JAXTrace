# jaxtrace/integrators/rk2.py  
"""
Second-order Runge-Kutta (midpoint) integration with JAX optimization.

Provides high-performance RK2 integration with memory management
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
    def _rk2_step_static_jit(
        x: jnp.ndarray,
        t: jnp.ndarray,
        dt_scalar: jnp.ndarray,
        v1: jnp.ndarray,
        v2: jnp.ndarray
    ) -> jnp.ndarray:
        """
        JIT-compiled RK2 step for static dt (scalar).
        
        Parameters
        ----------
        x : jnp.ndarray
            Particle positions, shape (N, 3)
        t : jnp.ndarray
            Current time (scalar as array)
        dt_scalar : jnp.ndarray
            Time step (scalar as array)
        v1 : jnp.ndarray
            Velocity at x, t, shape (N, 3)
        v2 : jnp.ndarray
            Velocity at x + dt/2 * v1, t + dt/2, shape (N, 3)
            
        Returns
        -------
        jnp.ndarray
            Updated positions, shape (N, 3)
        """
        return x + dt_scalar * v2

    def _rk2_step_dynamic_fallback(
        x: jnp.ndarray,
        t: jnp.ndarray, 
        dt_array: jnp.ndarray,
        v1: jnp.ndarray,
        v2: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Non-JIT RK2 step for dynamic dt (array).
        
        Parameters
        ----------
        x : jnp.ndarray
            Particle positions, shape (N, 3)
        t : jnp.ndarray
            Current time
        dt_array : jnp.ndarray
            Per-particle time steps, shape (N,) or (N, 1)
        v1 : jnp.ndarray
            Velocity at x, t, shape (N, 3)
        v2 : jnp.ndarray
            Velocity at midpoint, shape (N, 3)
            
        Returns
        -------
        jnp.ndarray
            Updated positions, shape (N, 3)
        """
        dt_expanded = dt_array.reshape(-1, 1) if dt_array.ndim == 1 else dt_array
        return x + dt_expanded * v2

else:
    # NumPy fallback implementations
    def _rk2_step_static_jit(x, t, dt_scalar, v1, v2):
        return x + dt_scalar * v2
        
    def _rk2_step_dynamic_fallback(x, t, dt_array, v1, v2):
        dt_expanded = dt_array.reshape(-1, 1) if dt_array.ndim == 1 else dt_array
        return x + dt_expanded * v2


def rk2_step(
    x: jnp.ndarray,
    t: float,
    dt: Union[float, jnp.ndarray],
    field_fn: Callable[[jnp.ndarray, float], jnp.ndarray],
) -> jnp.ndarray:
    """
    Second-order Runge-Kutta (midpoint method) integration step.

    Performs: x_{n+1} = x_n + dt * v(x_n + dt/2 * v(x_n, t_n), t_n + dt/2)
    
    Higher accuracy than Euler method with only one additional field evaluation.
    Automatically chooses JIT or dynamic version based on dt type.

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
    - Automatically converts inputs to float32 for JAX performance
    - Enforces shape (N, 3) for all position arrays
    - More accurate than Euler but requires 2x field evaluations
    """
    # Ensure consistent types and shapes
    x, t_arr, dt_arr = _ensure_consistent_types(x, t, dt)
    t_float = float(t_arr)
    
    # Step 1: Evaluate velocity at current position
    v1 = field_fn(x, t_float)
    v1 = _ensure_float32_shape(v1)
    
    # Step 2: Compute midpoint position and time
    if dt_arr.ndim == 0:  # Scalar dt
        dt_half = dt_arr * 0.5
        x_mid = x + dt_half * v1
        t_mid = t_float + float(dt_half)
    else:  # Array dt
        dt_half = dt_arr * 0.5
        dt_half_expanded = dt_half.reshape(-1, 1) if dt_half.ndim == 1 else dt_half
        x_mid = x + dt_half_expanded * v1
        t_mid = t_float + float(jnp.mean(dt_half))  # Use average for time
    
    # Step 3: Evaluate velocity at midpoint
    v2 = field_fn(x_mid, t_mid)
    v2 = _ensure_float32_shape(v2)
    
    # Step 4: Final update using midpoint velocity
    if dt_arr.ndim == 0:  # Scalar dt - use fast JIT version
        return _rk2_step_static_jit(x, t_arr, dt_arr, v1, v2)
    else:  # Array dt - use dynamic version
        return _rk2_step_dynamic_fallback(x, t_arr, dt_arr, v1, v2)


def rk2_step_batch(
    x_batch: jnp.ndarray,
    t: float,
    dt: float,
    field_fn: Callable[[jnp.ndarray, float], jnp.ndarray],
    batch_size: int = 5000,
) -> jnp.ndarray:
    """
    Memory-efficient batched RK2 integration for large particle counts.

    RK2 requires 2x memory compared to Euler, so smaller default batch size.

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
        Particles per batch (smaller than Euler due to memory needs)

    Returns
    -------
    jnp.ndarray
        Updated positions, shape (N, 3)
    """
    x_batch = _ensure_float32_shape(x_batch)
    n_particles = x_batch.shape[0]
    
    if n_particles <= batch_size:
        return rk2_step(x_batch, t, dt, field_fn)
    
    # Process in batches
    results = []
    for i in range(0, n_particles, batch_size):
        end_idx = min(i + batch_size, n_particles)
        x_chunk = x_batch[i:end_idx]
        x_next_chunk = rk2_step(x_chunk, t, dt, field_fn)
        results.append(x_next_chunk)
    
    return jnp.concatenate(results, axis=0)