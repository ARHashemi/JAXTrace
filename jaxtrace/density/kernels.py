from __future__ import annotations
from typing import Tuple
import math

import numpy as np

# Optional JAX
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False


# ---------- Bandwidth rules ----------

def scott_bandwidth(data: np.ndarray) -> float:
    """
    Scott's rule: h ~ n^{-1/(d+4)} * sigma
    """
    x = np.asarray(data)
    n, d = x.shape
    sigma = np.std(x, axis=0, ddof=1).mean()
    return sigma * n ** (-1.0 / (d + 4.0))


def silverman_bandwidth(data: np.ndarray) -> float:
    """
    Silverman's rule: h ~ ( (4/(d+2))^(1/(d+4)) ) * n^{-1/(d+4)} * sigma
    """
    x = np.asarray(data)
    n, d = x.shape
    sigma = np.std(x, axis=0, ddof=1).mean()
    factor = (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0))
    return factor * sigma * n ** (-1.0 / (d + 4.0))


# ---------- Gaussian kernel (for KDE) ----------

def gaussian_kernel(r2_over_h2, d: int):
    """
    Unnormalized Gaussian kernel exp(-0.5 * r^2 / h^2) and normalization for dimension d.
    Returns kernel value with proper normalization so integral over R^d = 1.
    """
    if JAX_AVAILABLE and isinstance(r2_over_h2, (jnp.ndarray, jax.Array)):
        norm = (2.0 * math.pi) ** (-0.5 * d)
        return norm * jnp.exp(-0.5 * r2_over_h2)
    r2_over_h2 = np.asarray(r2_over_h2)
    norm = (2.0 * math.pi) ** (-0.5 * d)
    return norm * np.exp(-0.5 * r2_over_h2)


# ---------- SPH kernels (cubic spline, Wendland C2) ----------

def cubic_spline_kernel_2d(r: np.ndarray, h: float) -> np.ndarray:
    """
    2D cubic spline kernel W(r,h) with normalization 10/(7*pi*h^2).
    """
    q = r / h
    sigma = 10.0 / (7.0 * math.pi * h * h)
    res = np.zeros_like(r)
    m1 = (q >= 0) & (q < 1)
    m2 = (q >= 1) & (q < 2)
    res[m1] = 1.0 - 1.5 * q[m1]**2 + 0.75 * q[m1]**3
    res[m2] = 0.25 * (2.0 - q[m2])**3
    return sigma * res

def cubic_spline_kernel_3d(r: np.ndarray, h: float) -> np.ndarray:
    """
    3D cubic spline kernel W(r,h) with normalization 1/(pi*h^3).
    """
    q = r / h
    sigma = 1.0 / (math.pi * h**3)
    res = np.zeros_like(r)
    m1 = (q >= 0) & (q < 1)
    m2 = (q >= 1) & (q < 2)
    res[m1] = 1.0 - 1.5 * q[m1]**2 + 0.75 * q[m1]**3
    res[m2] = 0.25 * (2.0 - q[m2])**3
    return sigma * res

def wendland_c2_kernel_2d(r: np.ndarray, h: float) -> np.ndarray:
    """
    2D Wendland C2 kernel with compact support 2h, normalized.
    W(q) = (7/(4*pi*h^2)) * (1 - q/2)^4 * (1 + 2q),  for 0 <= q <= 2
    """
    q = r / h
    sigma = 7.0 / (4.0 * math.pi * h * h)
    m = (q >= 0) & (q <= 2.0)
    t = (1.0 - 0.5 * q[m])
    return np.where(m, sigma * (t**4) * (1.0 + 2.0 * q[m]), 0.0)

def wendland_c2_kernel_3d(r: np.ndarray, h: float) -> np.ndarray:
    """
    3D Wendland C2 kernel with compact support 2h, normalized.
    W(q) = (21/(2*pi*h^3)) * (1 - q/2)^4 * (1 + 2q),  for 0 <= q <= 2
    """
    q = r / h
    sigma = 21.0 / (2.0 * math.pi * h**3)
    m = (q >= 0) & (q <= 2.0)
    t = (1.0 - 0.5 * q[m])
    return np.where(m, sigma * (t**4) * (1.0 + 2.0 * q[m]), 0.0)
