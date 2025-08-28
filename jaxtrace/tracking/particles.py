from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp


@dataclass
class Trajectory:
    """
    A minimal trajectory container.

    - If positions_over_time is None, only last_positions and times are meaningful.
    - times are the sampling times at which positions were recorded.
    """
    last_positions: jnp.ndarray              # (N,3)
    times: jnp.ndarray                       # (T,)
    positions_over_time: Optional[jnp.ndarray] = None  # (T,N,3) or None

    def positions_last(self) -> jnp.ndarray:
        return self.last_positions

    def positions_over_time_array(self) -> jnp.ndarray:
        if self.positions_over_time is None:
            raise RuntimeError("Trajectory does not contain recorded positions_over_time")
        return self.positions_over_time
