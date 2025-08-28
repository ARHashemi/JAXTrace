from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Tuple, Optional

import jax.numpy as jnp


@dataclass
class GridMeta:
    """Structured grid metadata."""
    origin: jnp.ndarray   # (3,)
    spacing: jnp.ndarray  # (3,)
    shape: Tuple[int, int, int]  # (Nx, Ny, Nz)
    bounds: jnp.ndarray   # (2,3), min/max


class Field(Protocol):
    """Spatial field on a static topology."""
    def sample(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Sample field at positions x.

        x: (N,3)
        returns: (N,C)
        """
        ...


class TimeDependentField(Protocol):
    """Time-dependent field with temporal interpolation and prefetching."""
    def sample_t(self, x: jnp.ndarray, t: float) -> jnp.ndarray:
        """
        Sample field at positions x and time t.

        x: (N,3)
        t: float or scalar array
        returns: (N,C)
        """
        ...

    def bounds(self) -> jnp.ndarray:
        """Return spatial bounds as (2,3)."""
        ...


# ---------- Barycentric helpers for triangles and tetrahedra ----------

def barycentric_coords_triangle(p: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray):
    """
    Barycentric coords of p in triangle (a,b,c). Works in 2D or 3D (planar).
    Returns (l1, l2, l3).
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = jnp.dot(v0, v0)
    d01 = jnp.dot(v0, v1)
    d11 = jnp.dot(v1, v1)
    d20 = jnp.dot(v2, v0)
    d21 = jnp.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    # safe inverse
    inv = jnp.where(denom != 0.0, 1.0 / denom, 0.0)
    v = (d11 * d20 - d01 * d21) * inv
    w = (d00 * d21 - d01 * d20) * inv
    u = 1.0 - v - w
    return jnp.stack([u, v, w], axis=0)  # (3,)


def barycentric_coords_tetra(p: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, d: jnp.ndarray):
    """
    Barycentric coords of p in tetrahedron (a,b,c,d).
    Returns (l1, l2, l3, l4).
    """
    m = jnp.stack([b - a, c - a, d - a], axis=1)  # 3x3
    v = p - a
    # Solve m * [l2,l3,l4] = v, then l1 = 1 - sum(l2..l4)
    # Use robust solve via least squares
    sol = jnp.linalg.lstsq(m, v, rcond=None)[0]
    l2, l3, l4 = sol
    l1 = 1.0 - (l2 + l3 + l4)
    return jnp.stack([l1, l2, l3, l4], axis=0)  # (4,)


# ---------- Quadratic shape functions on reference elements ----------

def tri6_shape(lmbda: jnp.ndarray):
    """
    6-node quadratic triangle shape functions from barycentric coords.
    lmbda: (3,) barycentrics [l1,l2,l3]
    returns: (6,) [N1..N6]
    """
    l1, l2, l3 = lmbda
    return jnp.stack([
        l1 * (2.0 * l1 - 1.0),
        l2 * (2.0 * l2 - 1.0),
        l3 * (2.0 * l3 - 1.0),
        4.0 * l1 * l2,
        4.0 * l2 * l3,
        4.0 * l3 * l1
    ], axis=0)


def tet10_shape(lmbda: jnp.ndarray):
    """
    10-node quadratic tetra shape functions from barycentric coords.
    lmbda: (4,)
    returns: (10,)
    Node order: [v1,v2,v3,v4, e12,e23,e31,e14,e24,e34] using 4 vertex + 6 edge midpoints.
    """
    l1, l2, l3, l4 = lmbda
    return jnp.stack([
        l1 * (2.0 * l1 - 1.0),
        l2 * (2.0 * l2 - 1.0),
        l3 * (2.0 * l3 - 1.0),
        l4 * (2.0 * l4 - 1.0),
        4.0 * l1 * l2,
        4.0 * l2 * l3,
        4.0 * l3 * l1,
        4.0 * l1 * l4,
        4.0 * l2 * l4,
        4.0 * l3 * l4
    ], axis=0)
