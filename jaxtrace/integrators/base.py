# jaxtrace/integrators/base.py
"""
Base types and protocols for numerical integration.

Defines the function signatures used by all integrators in JAXTrace.
All integrators follow the common pattern:

    new_positions = integrator_step(positions, time, dt, field_function)
"""

from __future__ import annotations
from typing import Callable, Protocol, Union
import numpy as np

# Import JAX utilities
from ..utils.jax_utils import JAX_AVAILABLE

if JAX_AVAILABLE:
    try:
        import jax.numpy as jnp
        ArrayLike = Union[np.ndarray, jnp.ndarray]
    except Exception:
        JAX_AVAILABLE = False
        
if not JAX_AVAILABLE:
    import numpy as jnp  # type: ignore
    ArrayLike = np.ndarray


# Field function signature: takes positions and time, returns velocities
FieldFn = Callable[[jnp.ndarray, float], jnp.ndarray]
"""
Field function protocol.

Parameters
----------
positions : jnp.ndarray
    Particle positions, shape (N, 3)
time : float
    Current simulation time

Returns
-------
jnp.ndarray
    Velocity vectors at positions, shape (N, 3)
"""


class IntegratorFn(Protocol):
    """
    Protocol for integrator step functions.
    
    All integrators must implement this signature for interoperability
    with ParticleTracker and other components.
    """
    
    def __call__(
        self, 
        x: jnp.ndarray, 
        t: float, 
        dt: Union[float, jnp.ndarray], 
        field_fn: FieldFn
    ) -> jnp.ndarray:
        """
        Advance particles by one time step.
        
        Parameters
        ----------
        x : jnp.ndarray
            Current particle positions, shape (N, 3)
        t : float
            Current simulation time
        dt : float or jnp.ndarray
            Time step size. If array, shape (N,) for per-particle time steps
        field_fn : FieldFn
            Function that computes velocity field at (positions, time)
            
        Returns
        -------
        jnp.ndarray
            New particle positions after time step, shape (N, 3)
        """
        ...


# Utility type aliases for convenience
TimestepLike = Union[float, jnp.ndarray]
"""Time step that can be scalar or per-particle array."""

IntegratorRegistry = dict[str, IntegratorFn]
"""Registry mapping integrator names to functions."""