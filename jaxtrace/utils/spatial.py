# jaxtrace/utils/spatial.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

try:
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False
    jnp = None  # type: ignore
    def jit(x):  # type: ignore
        return x

Array = np.ndarray

# -------------------------
# AABB
# -------------------------

@dataclass
class AABB:
    """
    Axis-aligned bounding box in 2D/3D.

    Attributes
    ----------
    lo : (D,) lower corner
    hi : (D,) upper corner
    """
    lo: Array
    hi: Array

    def __post_init__(self):
        self.lo = np.asarray(self.lo, dtype=float)
        self.hi = np.asarray(self.hi, dtype=float)
        if self.lo.shape != self.hi.shape:
            raise ValueError("lo and hi must have same shape")
        if not np.all(self.lo <= self.hi):
            lo = np.minimum(self.lo, self.hi)
            hi = np.maximum(self.lo, self.hi)
            self.lo, self.hi = lo, hi

    @property
    def dim(self) -> int:
        return self.lo.shape[0]

    def size(self) -> Array:
        return self.hi - self.lo

    def contains(self, pts: Array) -> Array:
        pts = np.asarray(pts)
        return np.logical_and(np.all(pts >= self.lo, axis=-1), np.all(pts <= self.hi, axis=-1))

    def intersects(self, other: "AABB") -> bool:
        return bool(np.all(self.hi >= other.lo) and np.all(other.hi >= self.lo))

    def expand(self, margin: float) -> "AABB":
        m = float(margin)
        return AABB(self.lo - m, self.hi + m)

    def clamp_points(self, pts: Array) -> Array:
        pts = np.asarray(pts)
        return np.clip(pts, self.lo, self.hi)

# -------------------------
# Transforms
# -------------------------

def translate(pts: Array, t: Array) -> Array:
    pts = np.asarray(pts); t = np.asarray(t)
    return pts + t

def scale(pts: Array, s: Array | float) -> Array:
    pts = np.asarray(pts); s = np.asarray(s)
    return pts * s

def rotate2d(pts: Array, theta: float) -> Array:
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return pts @ R.T

def _Rx(a: float) -> Array:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

def _Ry(a: float) -> Array:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

def _Rz(a: float) -> Array:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

def rotate3d_euler(pts: Array, angles: Tuple[float, float, float], order: str = "xyz") -> Array:
    """Rotate by Euler angles in radians with the given order."""
    ax = { "x": _Rx, "y": _Ry, "z": _Rz }
    R = np.eye(3)
    for k, a in zip(order.lower(), angles):
        R = ax[k](a) @ R
    return pts @ R.T

def transform_points(pts: Array, *, s: Array | float = 1.0, R: Optional[Array] = None, t: Optional[Array] = None) -> Array:
    """
    Apply similarity transform: y = (pts * s) @ R^T + t
    Any subset can be provided.
    """
    out = scale(pts, s)
    if R is not None:
        out = out @ np.asarray(R).T
    if t is not None:
        out = out + np.asarray(t)
    return out

# -------------------------
# Grid hashing
# -------------------------

def _pack_hash(ix: Array, iy: Array, iz: Optional[Array] = None) -> Array:
    """
    Pack integer cell coordinates into a 64-bit hash using mixed primes.
    Works for 2D (omit iz) and 3D. Collisions are rare but not impossible.
    """
    ix = ix.astype(np.int64); iy = iy.astype(np.int64)
    if iz is None:
        return (ix * 73856093) ^ (iy * 19349663)
    iz = iz.astype(np.int64)
    return (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)

def grid_hash(points: Array, cell_size: Array | float, origin: Array | float = 0.0):
    """
    Hash points onto a uniform grid.

    Parameters
    ----------
    points : (N,D)
    cell_size : scalar or (D,)
    origin : scalar or (D,)

    Returns
    -------
    keys : (N,) int64 hash keys
    cells : (N,D) int32 grid indices
    """
    P = np.asarray(points, dtype=float)
    cs = np.asarray(cell_size, dtype=float)
    og = np.asarray(origin, dtype=float)
    if cs.ndim == 0: cs = np.full(P.shape[1], cs)
    if og.ndim == 0: og = np.full(P.shape[1], og)
    idx = np.floor((P - og) / cs).astype(np.int32)
    if P.shape[1] == 2:
        keys = _pack_hash(idx[:, 0], idx[:, 1])
    elif P.shape[1] == 3:
        keys = _pack_hash(idx[:, 0], idx[:, 1], idx[:, 2])
    else:
        raise ValueError("points must be 2D or 3D")
    return keys.astype(np.int64), idx

# -------------------------
# Reductions
# -------------------------

def unique_ids(keys: Array) -> Tuple[Array, Array]:
    """
    Return unique keys and an inverse map such that unique[inv[i]] == keys[i].
    """
    keys = np.asarray(keys)
    uniq, inv = np.unique(keys, return_inverse=True)
    return uniq, inv

def segment_sum(values: Array, segment_ids: Array, num_segments: Optional[int] = None) -> Array:
    """
    Sum values over segments. Works for 1D or ND values.

    - NumPy: uses add.at
    - JAX: uses .at[].add which lowers to an efficient scatter add
    """
    v = np.asarray(values)
    ids = np.asarray(segment_ids).astype(np.int64)
    if num_segments is None:
        num_segments = int(ids.max() + 1) if ids.size > 0 else 0
    out_shape = (num_segments,) + v.shape[1:]
    if JAX_AVAILABLE:
        import jax.numpy as jnp
        out = jnp.zeros(out_shape, dtype=v.dtype)
        out = out.at[ids].add(v)
        return np.asarray(out)
    out = np.zeros(out_shape, dtype=v.dtype)
    np.add.at(out, ids, v)
    return out

def bincount_sum(ids: Array, weights: Optional[Array] = None, minlength: Optional[int] = None) -> Array:
    """
    Convenience wrapper for 1D segment counts/sums.

    If weights is None: counts per ID
    Else: weighted sum per ID
    """
    i = np.asarray(ids).astype(np.int64)
    size = int(i.max() + 1) if minlength is None and i.size > 0 else int(minlength or 0)
    if JAX_AVAILABLE:
        import jax.numpy as jnp
        out = jnp.zeros((size,), dtype=(weights.dtype if weights is not None else np.int64))
        if weights is None:
            out = out.at[i].add(1)
        else:
            out = out.at[i].add(np.asarray(weights))
        return np.asarray(out)
    return np.bincount(i, weights=weights, minlength=size)