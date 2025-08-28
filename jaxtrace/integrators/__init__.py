"""
JAXTrace Integrators

Explicit time-stepping methods for particle advection. Each stepper follows the
signature:

    new_x = step(x, t, dt, field_fn)

where:
- x: (N,3) particle positions
- t: scalar time
- dt: scalar or (N,) time step sizes
- field_fn: callable (x, t) -> (N,3) velocities
"""

from .base import IntegratorFn
from .euler import euler_step
from .rk2 import rk2_step
from .rk4 import rk4_step

__all__ = [
    "IntegratorFn",
    "euler_step",
    "rk2_step",
    "rk4_step",
]
