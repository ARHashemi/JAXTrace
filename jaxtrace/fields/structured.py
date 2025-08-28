from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp

from .base import GridMeta


def _floor_clip(i, lo, hi):
    i0 = jnp.floor(i).astype(jnp.int32)
    i0 = jnp.clip(i0, lo, hi - 1)
    i1 = jnp.clip(i0 + 1, lo, hi - 1)
    w = jnp.clip(i - i0.astype(i.dtype), 0.0, 1.0)
    return i0, i1, w


@jax.jit
def trilinear_sample(values: jnp.ndarray, x: jnp.ndarray, origin: jnp.ndarray, spacing: jnp.ndarray):
    """
    Trilinear interpolation on structured grid.

    values: (Nx,Ny,Nz,C)
    x: (N,3)
    origin: (3,)
    spacing: (3,)
    returns: (N,C)
    """
    Nx, Ny, Nz, C = values.shape
    # index in grid space
    g = (x - origin) / spacing
    ix, iy, iz = g[:, 0], g[:, 1], g[:, 2]

    i0x, i1x, wx = _floor_clip(ix, 0, Nx)
    i0y, i1y, wy = _floor_clip(iy, 0, Ny)
    i0z, i1z, wz = _floor_clip(iz, 0, Nz)

    def gather(ii, jj, kk):
        return values[ii, jj, kk, :]  # (N,C)

    c000 = gather(i0x, i0y, i0z)
    c100 = gather(i1x, i0y, i0z)
    c010 = gather(i0x, i1y, i0z)
    c110 = gather(i1x, i1y, i0z)
    c001 = gather(i0x, i0y, i1z)
    c101 = gather(i1x, i0y, i1z)
    c011 = gather(i0x, i1y, i1z)
    c111 = gather(i1x, i1y, i1z)

    c00 = c000 * (1 - wx)[:, None] + c100 * wx[:, None]
    c10 = c010 * (1 - wx)[:, None] + c110 * wx[:, None]
    c01 = c001 * (1 - wx)[:, None] + c101 * wx[:, None]
    c11 = c011 * (1 - wx)[:, None] + c111 * wx[:, None]

    c0 = c00 * (1 - wy)[:, None] + c10 * wy[:, None]
    c1 = c01 * (1 - wy)[:, None] + c11 * wy[:, None]

    c = c0 * (1 - wz)[:, None] + c1 * wz[:, None]
    return c  # (N,C)


@dataclass
class StructuredGridSampler:
    """
    Spatial sampler for structured grids using trilinear interpolation.
    """
    meta: GridMeta

    def sample_given_values(self, x: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
        return trilinear_sample(values, x, self.meta.origin, self.meta.spacing)
