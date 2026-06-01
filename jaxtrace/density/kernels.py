# jaxtrace/density/kernels.py
"""
SPH/KDE kernel functions, pure JAX.

All kernels have the signature::

    W(r, h, d) -> array

where ``r`` and ``h`` broadcast against each other (scalar h or per-particle h),
and ``d`` is the spatial dimensionality (2 or 3). All kernels are normalized so
that ``integral_{R^d} W(|x|, h) dx == 1``.

Two normalization conventions for the *output* of a density estimator are
expressed *outside* this module:

  - "pdf"  mode:  rho(x) = sum_i W(|x - x_i|, h_i, d) / N
  - "mass" mode:  rho(x) = sum_i m_i * W(|x - x_i|, h_i, d)

This file only provides the per-pair kernel evaluations.
"""

from __future__ import annotations

import math

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

# Compact-support radius (in units of h) for each kernel. A query at distance r
# from a particle with smoothing length h contributes only if r < SUPPORT * h
# — but only for kernels whose mathematical form is EXACTLY ZERO past that
# radius. Gaussian has no compact support; the 3*h figure here is a
# numerical-truncation convenience for radius queries, NOT a hard analytic
# cutoff. Code that wants to skip pairs / cells outside a radius (e.g. the
# octree per-cell pre-filter) must check ``kernel_has_compact_support`` first
# or it will silently truncate the Gaussian tails.
KERNEL_SUPPORT = {
    "gaussian":       3.0,
    "cubic_spline":   2.0,
    "wendland_c2":    2.0,
    "wendland_c4":    2.0,
    "epanechnikov":   1.0,
    "quintic_spline": 3.0,
}

# Whether the kernel is mathematically zero beyond ``SUPPORT * h``.
# False for Gaussian (asymptotic decay only); True for everything else.
KERNEL_HAS_COMPACT_SUPPORT = {
    "gaussian":       False,
    "cubic_spline":   True,
    "wendland_c2":    True,
    "wendland_c4":    True,
    "epanechnikov":   True,
    "quintic_spline": True,
}


def kernel_support(name: str) -> float:
    """Return the compact-support radius in units of h for the named kernel.

    For Gaussian (no compact support) this returns the 3*h numerical
    truncation radius; callers that need exact arithmetic must consult
    :func:`kernel_has_compact_support` before using this for cutoff logic.
    """
    if name not in KERNEL_SUPPORT:
        raise ValueError(f"unknown kernel {name!r}; choose from {KERNEL_NAMES}")
    return KERNEL_SUPPORT[name]


def kernel_has_compact_support(name: str) -> bool:
    """True iff the kernel evaluates to exactly zero for r >= SUPPORT * h."""
    if name not in KERNEL_HAS_COMPACT_SUPPORT:
        raise ValueError(f"unknown kernel {name!r}; choose from {KERNEL_NAMES}")
    return KERNEL_HAS_COMPACT_SUPPORT[name]


# -----------------------------------------------------------------------------
# Per-kernel evaluators (JAX, broadcasting)
# -----------------------------------------------------------------------------

def _safe_h(h):
    return jnp.maximum(h, jnp.asarray(1e-30, dtype=h.dtype) if hasattr(h, "dtype") else 1e-30)


def gaussian(r, h, d: int):
    """Isotropic Gaussian: W = (2*pi*h^2)^(-d/2) * exp(-r^2 / (2 h^2))."""
    hs = _safe_h(h)
    sigma = (2.0 * math.pi) ** (-0.5 * d) * hs ** (-d)
    return sigma * jnp.exp(-0.5 * (r / hs) ** 2)


def cubic_spline(r, h, d: int):
    """M4 cubic spline with compact support 2h."""
    hs = _safe_h(h)
    q = r / hs
    if d == 2:
        sigma = 10.0 / (7.0 * math.pi) * hs ** (-2)
    elif d == 3:
        sigma = 1.0 / math.pi * hs ** (-3)
    else:
        raise ValueError("cubic_spline supports d in {2,3}")
    w1 = 1.0 - 1.5 * q ** 2 + 0.75 * q ** 3
    w2 = 0.25 * (2.0 - q) ** 3
    out = jnp.where(q < 1.0, w1, jnp.where(q < 2.0, w2, 0.0))
    return sigma * out


def wendland_c2(r, h, d: int):
    """Wendland C2 with compact support 2h: (1 - q/2)^4 * (1 + 2 q)."""
    hs = _safe_h(h)
    q = r / hs
    if d == 2:
        sigma = 7.0 / (4.0 * math.pi) * hs ** (-2)
    elif d == 3:
        sigma = 21.0 / (16.0 * math.pi) * hs ** (-3)
    else:
        raise ValueError("wendland_c2 supports d in {2,3}")
    t = jnp.maximum(1.0 - 0.5 * q, 0.0)
    return sigma * (t ** 4) * (1.0 + 2.0 * q) * jnp.where(q < 2.0, 1.0, 0.0)


def wendland_c4(r, h, d: int):
    """Wendland C4 with compact support 2h: (1 - q/2)^6 * (1 + 3 q + 35/12 q^2)."""
    hs = _safe_h(h)
    q = r / hs
    if d == 2:
        sigma = 9.0 / (4.0 * math.pi) * hs ** (-2)
    elif d == 3:
        sigma = 495.0 / (256.0 * math.pi) * hs ** (-3)
    else:
        raise ValueError("wendland_c4 supports d in {2,3}")
    t = jnp.maximum(1.0 - 0.5 * q, 0.0)
    inner = 1.0 + 3.0 * q + (35.0 / 12.0) * q ** 2
    return sigma * (t ** 6) * inner * jnp.where(q < 2.0, 1.0, 0.0)


def epanechnikov(r, h, d: int):
    """Epanechnikov kernel with compact support 1*h: max(0, 1 - q^2)."""
    hs = _safe_h(h)
    q = r / hs
    if d == 2:
        sigma = 2.0 / math.pi * hs ** (-2)
    elif d == 3:
        sigma = 15.0 / (8.0 * math.pi) * hs ** (-3)
    else:
        raise ValueError("epanechnikov supports d in {2,3}")
    return sigma * jnp.maximum(1.0 - q ** 2, 0.0)


def quintic_spline(r, h, d: int):
    """M6 quintic spline with compact support 3h."""
    hs = _safe_h(h)
    q = r / hs
    if d == 2:
        sigma = 7.0 / (478.0 * math.pi) * hs ** (-2)
    elif d == 3:
        sigma = 1.0 / (120.0 * math.pi) * hs ** (-3)
    else:
        raise ValueError("quintic_spline supports d in {2,3}")
    a = jnp.maximum(3.0 - q, 0.0) ** 5
    b = jnp.where(q < 2.0, 6.0 * jnp.maximum(2.0 - q, 0.0) ** 5, 0.0)
    c = jnp.where(q < 1.0, 15.0 * jnp.maximum(1.0 - q, 0.0) ** 5, 0.0)
    return sigma * (a - b + c) * jnp.where(q < 3.0, 1.0, 0.0)


_DISPATCH = {
    "gaussian": gaussian,
    "cubic_spline": cubic_spline,
    "wendland_c2": wendland_c2,
    "wendland_c4": wendland_c4,
    "epanechnikov": epanechnikov,
    "quintic_spline": quintic_spline,
}


def evaluate_kernel(name: str, r, h, d: int):
    """Dispatch by name. Use this from JIT-traced code with a static name."""
    if name not in _DISPATCH:
        raise ValueError(f"unknown kernel {name!r}; choose from {KERNEL_NAMES}")
    return _DISPATCH[name](r, h, d)
