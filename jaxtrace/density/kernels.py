# jaxtrace/density/kernels.py
"""
SPH/KDE kernel functions, pure JAX, **with anisotropic per-axis bandwidth**.

Each kernel is decomposed into

  - a *shape function* ``W_shape(q, d)`` that depends only on the
    normalised distance ``q`` and the spatial dimensionality ``d``. The
    shape already includes the kernel's dimensionless constant (e.g.
    ``21/(16 π)`` for the 3-D Wendland C2).
  - a *normalisation factor* that depends only on the bandwidth and
    makes the kernel integrate to 1:

        normalisation(h) = 1 / (h_x * h_y * h_z)        in 3-D
                         = 1 / (h_x * h_y)              in 2-D

This decoupling lets us write the kernel **anisotropically** without
duplicating the shape logic per kernel. The normalised distance is

    q = sqrt( Σ_a (Δx_a / h_a)^2 )

i.e. the Euclidean distance in coordinates scaled per-axis by ``h_a``.
For isotropic ``h_a == h`` this collapses to the familiar ``q = r/h``.

Public API
----------

``evaluate_kernel(name, diff, h, d)``
    The single anisotropic entry point. ``diff`` has shape ``(..., d)``
    and is the displacement vector ``q - x``. ``h`` has shape ``(..., d)``
    and broadcasts against ``diff``. Returns the kernel value of shape
    ``(...,)``.

``kernel_support(name)`` / ``kernel_has_compact_support(name)``
    Unchanged from the isotropic API. The "support radius" is in units
    of ``h`` along *each* axis: a point with ``q > SUPPORT`` contributes
    zero (or numerically negligibly, for Gaussian).
"""

from __future__ import annotations

import math
from typing import Callable, Dict

import jax
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# Kernel registry
# -----------------------------------------------------------------------------

KERNEL_NAMES = (
    "gaussian",
    "cubic_spline",
    "wendland_c2",
    "wendland_c4",
    "epanechnikov",
    "quintic_spline",
)

# Compact-support radius in units of h. For anisotropic h this still means
# "q < SUPPORT" where q is the normalised distance — equivalently, the
# kernel is zero outside the ellipsoid {Δx : Σ_a (Δx_a/h_a)^2 < SUPPORT^2}.
#
# Gaussian has no analytic compact support; 5.0 is the numerical-truncation
# radius at which exp(-q²/2) ~ 4e-6, small enough that stencil-based culling
# never drops more than ~1e-5 from the integral.
KERNEL_SUPPORT = {
    "gaussian":       5.0,
    "cubic_spline":   2.0,
    "wendland_c2":    2.0,
    "wendland_c4":    2.0,
    "epanechnikov":   1.0,
    "quintic_spline": 3.0,
}

KERNEL_HAS_COMPACT_SUPPORT = {
    "gaussian":       False,
    "cubic_spline":   True,
    "wendland_c2":    True,
    "wendland_c4":    True,
    "epanechnikov":   True,
    "quintic_spline": True,
}


def kernel_support(name: str) -> float:
    """Compact-support radius in units of h for the named kernel."""
    if name not in KERNEL_SUPPORT:
        raise ValueError(f"unknown kernel {name!r}; choose from {KERNEL_NAMES}")
    return KERNEL_SUPPORT[name]


def kernel_has_compact_support(name: str) -> bool:
    """True iff the kernel is exactly zero past ``SUPPORT * h``."""
    if name not in KERNEL_HAS_COMPACT_SUPPORT:
        raise ValueError(f"unknown kernel {name!r}; choose from {KERNEL_NAMES}")
    return KERNEL_HAS_COMPACT_SUPPORT[name]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_TINY_H = 1e-30


def _safe_h(h):
    """Floor ``h`` to avoid divide-by-zero in degenerate (zero-bandwidth) cases."""
    return jnp.maximum(h, _TINY_H)


def _normalised_q(diff, h):
    """
    Normalised distance ``q = sqrt(Σ (Δx_a / h_a)^2)``.

    ``diff`` and ``h`` broadcast against each other; the last axis is the
    spatial axis. Output drops the last axis.
    """
    hs = _safe_h(h)
    scaled = diff / hs
    return jnp.sqrt(jnp.sum(scaled * scaled, axis=-1))


def _inv_volume(h, d: int):
    """
    Anisotropic normalisation factor ``1 / (h_x * h_y * ... )`` over the
    last ``d`` axes of ``h``. For isotropic h (last-axis broadcastable to
    a scalar) this collapses to ``1/h^d``.

    Always uses the floor on h to keep the division finite when an axis
    has a vanishing bandwidth (e.g. for placeholder ghost particles).
    """
    hs = _safe_h(h)
    return jnp.prod(1.0 / hs, axis=-1)


# -----------------------------------------------------------------------------
# Per-kernel shape functions (dimensionless in q)
#
# Each returns the kernel value INCLUDING the dimensionless leading constant
# (e.g. 21/(16 π) for 3-D Wendland C2). The caller multiplies by 1/(h_x h_y h_z)
# to get the properly anisotropic normalised kernel.
# -----------------------------------------------------------------------------

def _shape_gaussian(q, d: int):
    return (2.0 * math.pi) ** (-0.5 * d) * jnp.exp(-0.5 * q * q)


def _shape_cubic_spline(q, d: int):
    if d == 2:
        c = 10.0 / (7.0 * math.pi)
    elif d == 3:
        c = 1.0 / math.pi
    else:
        raise ValueError("cubic_spline supports d in {2,3}")
    w1 = 1.0 - 1.5 * q * q + 0.75 * q ** 3
    w2 = 0.25 * (2.0 - q) ** 3
    out = jnp.where(q < 1.0, w1, jnp.where(q < 2.0, w2, 0.0))
    return c * out


def _shape_wendland_c2(q, d: int):
    if d == 2:
        c = 7.0 / (4.0 * math.pi)
    elif d == 3:
        c = 21.0 / (16.0 * math.pi)
    else:
        raise ValueError("wendland_c2 supports d in {2,3}")
    t = jnp.maximum(1.0 - 0.5 * q, 0.0)
    return c * (t ** 4) * (1.0 + 2.0 * q) * jnp.where(q < 2.0, 1.0, 0.0)


def _shape_wendland_c4(q, d: int):
    if d == 2:
        c = 9.0 / (4.0 * math.pi)
    elif d == 3:
        c = 495.0 / (256.0 * math.pi)
    else:
        raise ValueError("wendland_c4 supports d in {2,3}")
    t = jnp.maximum(1.0 - 0.5 * q, 0.0)
    inner = 1.0 + 3.0 * q + (35.0 / 12.0) * q * q
    return c * (t ** 6) * inner * jnp.where(q < 2.0, 1.0, 0.0)


def _shape_epanechnikov(q, d: int):
    if d == 2:
        c = 2.0 / math.pi
    elif d == 3:
        c = 15.0 / (8.0 * math.pi)
    else:
        raise ValueError("epanechnikov supports d in {2,3}")
    return c * jnp.maximum(1.0 - q * q, 0.0)


def _shape_quintic_spline(q, d: int):
    if d == 2:
        c = 7.0 / (478.0 * math.pi)
    elif d == 3:
        c = 1.0 / (120.0 * math.pi)
    else:
        raise ValueError("quintic_spline supports d in {2,3}")
    a = jnp.maximum(3.0 - q, 0.0) ** 5
    b = jnp.where(q < 2.0, 6.0 * jnp.maximum(2.0 - q, 0.0) ** 5, 0.0)
    c_term = jnp.where(q < 1.0, 15.0 * jnp.maximum(1.0 - q, 0.0) ** 5, 0.0)
    return c * (a - b + c_term) * jnp.where(q < 3.0, 1.0, 0.0)


_SHAPE_DISPATCH: Dict[str, Callable] = {
    "gaussian":       _shape_gaussian,
    "cubic_spline":   _shape_cubic_spline,
    "wendland_c2":    _shape_wendland_c2,
    "wendland_c4":    _shape_wendland_c4,
    "epanechnikov":   _shape_epanechnikov,
    "quintic_spline": _shape_quintic_spline,
}


# -----------------------------------------------------------------------------
# Public anisotropic kernel API
# -----------------------------------------------------------------------------

def evaluate_kernel(name: str, diff, h, d: int):
    """
    Anisotropic kernel evaluator.

    Parameters
    ----------
    name : str
        Kernel name; one of :data:`KERNEL_NAMES`.
    diff : array, shape ``(..., d)``
        Displacement vector ``q - x`` per pair.
    h : array, shape ``(..., d)``
        Per-axis bandwidth, broadcastable against ``diff``. For isotropic
        bandwidth the caller can pass an array of shape ``(..., 1)`` or
        broadcast ``(..., d)`` with identical entries.
    d : int
        Spatial dimensionality (2 or 3). Must match ``diff.shape[-1]``.

    Returns
    -------
    array, shape ``(...,)``
        The kernel value at each pair.
    """
    if name not in _SHAPE_DISPATCH:
        raise ValueError(f"unknown kernel {name!r}; choose from {KERNEL_NAMES}")
    q = _normalised_q(diff, h)
    return _SHAPE_DISPATCH[name](q, d) * _inv_volume(h, d)


def evaluate_kernel_from_q(name: str, q, h, d: int):
    """
    Variant for callers that have already computed the normalised distance
    ``q`` (e.g. the octree backend, which evaluates per-pair distances
    in a fori_loop). ``h`` is still required to apply the anisotropic
    1/(h_x h_y …) normalisation.
    """
    if name not in _SHAPE_DISPATCH:
        raise ValueError(f"unknown kernel {name!r}; choose from {KERNEL_NAMES}")
    return _SHAPE_DISPATCH[name](q, d) * _inv_volume(h, d)


# -----------------------------------------------------------------------------
# Back-compat shims (deprecated). The earlier API took ``r`` (scalar
# distance) and an isotropic ``h``; we keep small wrappers so any
# stragglers don't break. New code should use :func:`evaluate_kernel`.
# -----------------------------------------------------------------------------

def gaussian(r, h, d: int):
    """Isotropic Gaussian (back-compat). Prefer ``evaluate_kernel``."""
    return _SHAPE_DISPATCH["gaussian"](r / _safe_h(h), d) / (_safe_h(h) ** d)


def cubic_spline(r, h, d: int):
    return _SHAPE_DISPATCH["cubic_spline"](r / _safe_h(h), d) / (_safe_h(h) ** d)


def wendland_c2(r, h, d: int):
    return _SHAPE_DISPATCH["wendland_c2"](r / _safe_h(h), d) / (_safe_h(h) ** d)


def wendland_c4(r, h, d: int):
    return _SHAPE_DISPATCH["wendland_c4"](r / _safe_h(h), d) / (_safe_h(h) ** d)


def epanechnikov(r, h, d: int):
    return _SHAPE_DISPATCH["epanechnikov"](r / _safe_h(h), d) / (_safe_h(h) ** d)


def quintic_spline(r, h, d: int):
    return _SHAPE_DISPATCH["quintic_spline"](r / _safe_h(h), d) / (_safe_h(h) ** d)
