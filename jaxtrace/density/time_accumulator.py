# jaxtrace/density/time_accumulator.py
"""
GPU-resident accumulators for time-averaged density and related summary fields.

Maintains, per voxel, four running sums updated each step:

    sum_dt           : sum of dt_k                         -> denominator
    sum_rho_dt       : sum of rho_k * dt_k                 -> mean density
    sum_active_dt    : sum of (rho_k > eps) * dt_k         -> coverage fraction
    peak_rho         : running max of rho_k                -> peak density
    peak_time        : argmax time                         -> peak time

Output finalized as:
    mean_density       = sum_rho_dt / max(sum_dt, eps)
    coverage_fraction  = sum_active_dt / max(sum_dt, eps)
    peak_density       = peak_rho
    peak_time          = peak_time

All updates are jit-able; the accumulator holds JAX device arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class TimeAccumulator:
    n_voxels: int
    eps_active: float = 1e-12
    sum_dt: float = 0.0  # scalar, kept on host (cheap) since it's identical for all voxels
    sum_rho_dt: jnp.ndarray = None
    sum_active_dt: jnp.ndarray = None
    peak_rho: jnp.ndarray = None
    peak_time: jnp.ndarray = None

    def __post_init__(self):
        if self.sum_rho_dt is None:
            self.sum_rho_dt = jnp.zeros((self.n_voxels,), dtype=jnp.float32)
            self.sum_active_dt = jnp.zeros((self.n_voxels,), dtype=jnp.float32)
            self.peak_rho = jnp.zeros((self.n_voxels,), dtype=jnp.float32)
            self.peak_time = jnp.zeros((self.n_voxels,), dtype=jnp.float32)

    def update(self, rho: jnp.ndarray, dt: float, t: float) -> None:
        """In-place update with a per-step density array."""
        dt32 = jnp.float32(dt)
        t32 = jnp.float32(t)
        eps = jnp.float32(self.eps_active)

        self.sum_rho_dt = self.sum_rho_dt + rho * dt32
        self.sum_active_dt = self.sum_active_dt + (rho > eps).astype(jnp.float32) * dt32
        new_peak = rho > self.peak_rho
        self.peak_rho = jnp.where(new_peak, rho, self.peak_rho)
        self.peak_time = jnp.where(new_peak, t32, self.peak_time)
        self.sum_dt += float(dt)

    def finalize(self) -> dict[str, np.ndarray]:
        """Return host numpy arrays for writing."""
        denom = max(self.sum_dt, self.eps_active)
        mean_density = np.asarray(self.sum_rho_dt / jnp.float32(denom))
        coverage = np.asarray(self.sum_active_dt / jnp.float32(denom))
        return {
            "mean_density": mean_density.astype(np.float32),
            "coverage_fraction": coverage.astype(np.float32),
            "peak_density": np.asarray(self.peak_rho, dtype=np.float32),
            "peak_time": np.asarray(self.peak_time, dtype=np.float32),
            "total_time": np.float32(self.sum_dt),
        }
