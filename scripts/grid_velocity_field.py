"""
grid_velocity_field.py --- template for a uniform-grid velocity field
module that plugs into JAXTrace's --velocity-source analytic path.

The trick: the tracker's analytic path already accepts any pure
callable `velocity_fn(pos: jnp.ndarray[3])`.  If we back that callable
by an O(1) uniform-grid lookup instead of an actual analytic formula,
we get grid-based particle tracking WITHOUT any element search, WITHOUT
octree traversal, WITHOUT touching the mesh code path.

This module is NOT meant to be imported directly.  It is a template:
the caller (see `scripts/build_grid_velocity_case.py`) generates a
concrete instance for a given case by

  1. Loading a source velocity field (mesh, analytic, or ROM).
  2. Sampling it onto a uniform grid via analytic_grid_projection.py.
  3. Baking the grid points + velocity values into constants at the top
     of a fresh copy of this file.
  4. Writing the copy alongside the case's other run scripts.

Example instantiation:

  GRID_XMIN = (-0.0625, -0.03125, -0.015625)
  GRID_XMAX = ( 0.0625,  0.03125,  0.015625)
  GRID_N    = (128, 64, 8)
  GRID_V    = <flattened (N, 3) numpy array baked into the module>

Then `run_tracking.py --velocity-source analytic --velocity-module
this_file.py --domain-bbox ...` runs particle tracking on the grid
velocity with no element search per RK4 substep.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jaxtrace.gpu.tracking.velocity_provider import AnalyticVelocityProvider


# ---------------------------------------------------------------------------
# Parameters -- populated by scripts/build_grid_velocity_case.py.
# The defaults below make it possible to import this module for testing.
# ---------------------------------------------------------------------------

GRID_XMIN: tuple[float, float, float] = (-1.0, -1.0, -1.0)
GRID_XMAX: tuple[float, float, float] = ( 1.0,  1.0,  1.0)
GRID_N:    tuple[int,   int,   int]   = ( 2,    2,    2)
GRID_V:    np.ndarray = np.zeros((8, 3), dtype=np.float32)

# Interpolation mode: 'nearest' (O(1) hash) or 'trilinear' (O(1) but 8
# neighbours per query).  Trilinear is usually what you want; 'nearest'
# is a diagnostic mode that shows how coarse the grid actually is.
INTERP_MODE: str = "trilinear"


def _build_lookup():
    """Precompute JAX-friendly grid constants."""
    xmin = jnp.asarray(GRID_XMIN, dtype=jnp.float32)
    xmax = jnp.asarray(GRID_XMAX, dtype=jnp.float32)
    n    = jnp.asarray(GRID_N,    dtype=jnp.int32)
    dx   = (xmax - xmin) / n.astype(jnp.float32)
    v    = jnp.asarray(
        np.asarray(GRID_V, dtype=np.float32).reshape(
            int(GRID_N[0]), int(GRID_N[1]), int(GRID_N[2]), 3
        )
    )
    return xmin, xmax, n, dx, v


_XMIN, _XMAX, _N, _DX, _V = _build_lookup()


def _clamp_index(idx, n):
    return jnp.clip(idx, 0, n - 1)


def _velocity_fn_nearest(pos):
    """Zero-order (nearest-cell) lookup.  O(1) per query."""
    frac = (pos - _XMIN) / _DX
    idx = jnp.floor(frac).astype(jnp.int32)
    ix = _clamp_index(idx[0], _N[0])
    iy = _clamp_index(idx[1], _N[1])
    iz = _clamp_index(idx[2], _N[2])
    return _V[ix, iy, iz]


def _velocity_fn_trilinear(pos):
    """Trilinear (weighted 8-corner) interpolation.  Still O(1)."""
    # Convert to normalised fractional index space
    frac = (pos - _XMIN) / _DX - 0.5     # cell centres at half-integer
    i0 = jnp.floor(frac).astype(jnp.int32)
    t = frac - i0.astype(jnp.float32)     # (3,) in [0, 1]
    i1 = i0 + 1
    ix0 = _clamp_index(i0[0], _N[0]); ix1 = _clamp_index(i1[0], _N[0])
    iy0 = _clamp_index(i0[1], _N[1]); iy1 = _clamp_index(i1[1], _N[1])
    iz0 = _clamp_index(i0[2], _N[2]); iz1 = _clamp_index(i1[2], _N[2])
    tx, ty, tz = t[0], t[1], t[2]

    c000 = _V[ix0, iy0, iz0]
    c100 = _V[ix1, iy0, iz0]
    c010 = _V[ix0, iy1, iz0]
    c110 = _V[ix1, iy1, iz0]
    c001 = _V[ix0, iy0, iz1]
    c101 = _V[ix1, iy0, iz1]
    c011 = _V[ix0, iy1, iz1]
    c111 = _V[ix1, iy1, iz1]

    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0  = c00  * (1 - ty) + c10  * ty
    c1  = c01  * (1 - ty) + c11  * ty
    return c0 * (1 - tz) + c1 * tz


def velocity_fn(pos):
    """Public entry point: dispatches on INTERP_MODE at import time."""
    if INTERP_MODE == "nearest":
        return _velocity_fn_nearest(pos)
    return _velocity_fn_trilinear(pos)


DOMAIN_BBOX = (
    (float(GRID_XMIN[0]), float(GRID_XMAX[0])),
    (float(GRID_XMIN[1]), float(GRID_XMAX[1])),
    (float(GRID_XMIN[2]), float(GRID_XMAX[2])),
)


def build_provider(domain_bbox=None, dt=0.0, t_start=0.0):
    del dt, t_start
    if domain_bbox is None:
        domain_bbox = DOMAIN_BBOX
    return AnalyticVelocityProvider(
        velocity_fn=velocity_fn,
        is_time_dependent=False,
        level_set_fn=None,
        domain_bbox=domain_bbox,
        meta={
            "name":         "grid_velocity_field",
            "grid_xmin":    tuple(GRID_XMIN),
            "grid_xmax":    tuple(GRID_XMAX),
            "grid_n":       tuple(GRID_N),
            "interp_mode":  INTERP_MODE,
            "n_cells":      int(np.prod(GRID_N)),
            "note":         "O(1) grid lookup: no element search, no octree.",
        },
    )
