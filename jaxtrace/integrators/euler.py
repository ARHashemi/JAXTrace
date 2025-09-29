# jaxtrace/integrators/euler.py
"""
Forward Euler integration with JAX optimization.

Provides both static (JIT-compiled) and dynamic (flexible) versions
for optimal performance under different use cases.
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
    def _euler_step_static_jit(
        x: jnp.ndarray, 
        t: jnp.ndarray, 
        dt_scalar: jnp.ndarray, 
        v: jnp.ndarray
    ) -> jnp.ndarray:
        """
        JIT-compiled Euler step for static dt (scalar).
        
        Parameters
        ----------
        x : jnp.ndarray
            Particle positions, shape (N, 3)
        t : jnp.ndarray
            Current time (scalar as array)
        dt_scalar : jnp.ndarray  
            Time step (scalar as array)
        v : jnp.ndarray
            Velocity at positions, shape (N, 3)
            
        Returns
        -------
        jnp.ndarray
            Updated positions, shape (N, 3)
        """
        return x + dt_scalar * v

    def _euler_step_dynamic_fallback(
        x: jnp.ndarray, 
        t: jnp.ndarray, 
        dt_array: jnp.ndarray, 
        v: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Non-JIT Euler step for dynamic dt (array).
        
        Parameters
        ----------
        x : jnp.ndarray
            Particle positions, shape (N, 3)
        t : jnp.ndarray
            Current time
        dt_array : jnp.ndarray
            Per-particle time steps, shape (N,) or (N, 1)
        v : jnp.ndarray
            Velocity at positions, shape (N, 3)
            
        Returns
        -------
        jnp.ndarray
            Updated positions, shape (N, 3)
        """
        dt_expanded = dt_array.reshape(-1, 1) if dt_array.ndim == 1 else dt_array
        return x + dt_expanded * v

else:
    # NumPy fallback implementations
    def _euler_step_static_jit(x, t, dt_scalar, v):
        return x + dt_scalar * v
        
    def _euler_step_dynamic_fallback(x, t, dt_array, v):
        dt_expanded = dt_array.reshape(-1, 1) if dt_array.ndim == 1 else dt_array
        return x + dt_expanded * v


def euler_step(
    x: jnp.ndarray,
    t: float,
    dt: Union[float, jnp.ndarray],
    field_fn: Callable[[jnp.ndarray, float], jnp.ndarray],
) -> jnp.ndarray:
    """
    Forward Euler integration step: x_{n+1} = x_n + dt * v(x_n, t_n).

    Automatically chooses between static (JIT-compiled) and dynamic versions
    based on the dt argument type for optimal performance.

    Parameters
    ----------
    x : jnp.ndarray
        Current particle positions, shape (N, 3)
    t : float
        Current time
    dt : float or jnp.ndarray
        Time step size. If scalar, uses fast JIT version.
        If array (N,), uses flexible dynamic version.
    field_fn : Callable[[jnp.ndarray, float], jnp.ndarray]
        Velocity field function returning shape (N, 3)

    Returns
    -------
    jnp.ndarray
        Updated particle positions, shape (N, 3)

    Notes
    -----
    - Input arrays are automatically converted to float32 for JAX performance
    - Shape (N, 3) is enforced for all position arrays
    - Uses JIT compilation when dt is scalar for maximum speed
    """
    # Ensure consistent types and shapes
    x, t_arr, dt_arr = _ensure_consistent_types(x, t, dt)
    
    # Sample velocity field
    v = field_fn(x, float(t_arr))
    v = _ensure_float32_shape(v)
    
    # Choose implementation based on dt type
    if dt_arr.ndim == 0:  # Scalar dt - use fast JIT version
        return _euler_step_static_jit(x, t_arr, dt_arr, v)
    else:  # Array dt - use dynamic version
        return _euler_step_dynamic_fallback(x, t_arr, dt_arr, v)


def euler_step_batch(
    x_batch: jnp.ndarray,
    t: float,
    dt: float,
    field_fn: Callable[[jnp.ndarray, float], jnp.ndarray],
    batch_size: int = 10000,
) -> jnp.ndarray:
    """
    Memory-efficient batched Euler integration for large particle counts.

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
        Particles per batch to control memory usage

    Returns
    -------
    jnp.ndarray
        Updated positions, shape (N, 3)
    """
    x_batch = _ensure_float32_shape(x_batch)
    n_particles = x_batch.shape[0]
    
    if n_particles <= batch_size:
        return euler_step(x_batch, t, dt, field_fn)
    
    # Process in batches
    results = []
    for i in range(0, n_particles, batch_size):
        end_idx = min(i + batch_size, n_particles)
        x_chunk = x_batch[i:end_idx]
        x_next_chunk = euler_step(x_chunk, t, dt, field_fn)
        results.append(x_next_chunk)
    
    return jnp.concatenate(results, axis=0)