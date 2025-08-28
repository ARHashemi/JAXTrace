from __future__ import annotations
from typing import Callable, Protocol

import jax.numpy as jnp


# Field callback: takes positions and time, returns velocities
FieldFn = Callable[[jnp.ndarray, float], jnp.ndarray]


class IntegratorFn(Protocol):
    def __call__(self, x: jnp.ndarray, t: float, dt: float | jnp.ndarray, field_fn: FieldFn) -> jnp.ndarray: ...
