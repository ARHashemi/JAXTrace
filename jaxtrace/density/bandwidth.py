# jaxtrace/density/bandwidth.py
"""
Smoothing-length / bandwidth resolution strategies. All paths are JAX-based
so the resulting ``h`` array is a JAX device array ready for the estimator.

Public API
----------

resolve_bandwidth(positions, mode, kernel, voxel_size, *,
                  fixed_h=None, bandwidth_factor=2.0,
                  knn_k=32, knn_safety=1.2,
                  particle_octree=None) -> jnp.ndarray  shape (N,)

Modes:
  - "fixed":         per-particle h = fixed_h (or bandwidth_factor * voxel_size)
  - "scott":         h_scalar = sigma * N^{-1/(d+4)}
  - "silverman":     h_scalar = (4/(d+2))^{1/(d+4)} * sigma * N^{-1/(d+4)}
  - "knn_adaptive":  per-particle h_i = knn_safety * dist(i, k-th neighbor)

For "knn_adaptive" a particle octree may be passed; if None, a brute-force
k-NN is used (acceptable for N <= ~50k). The brute-force path is fully JIT-ed
and chunked over query particles to bound memory.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def resolve_bandwidth(
    positions: jnp.ndarray,            # (N, 3) device array, float32
    mode: str,
    *,
    voxel_size: Optional[float] = None,
    fixed_h: Optional[float] = None,
    bandwidth_factor: float = 2.0,
    knn_k: int = 32,
    knn_safety: float = 1.2,
    knn_chunk: int = 4096,
    d: int = 3,
) -> jnp.ndarray:
    """
    Return per-particle smoothing lengths, shape (N,), float32, on device.
    """
    if mode == "fixed":
        if fixed_h is None:
            if voxel_size is None:
                raise ValueError("fixed bandwidth requires fixed_h or voxel_size")
            h_val = float(bandwidth_factor) * float(voxel_size)
        else:
            h_val = float(fixed_h)
        return jnp.full((positions.shape[0],), h_val, dtype=jnp.float32)

    if mode == "scott":
        h_val = _scott_silverman(positions, rule="scott", d=d)
        return jnp.full((positions.shape[0],), h_val, dtype=jnp.float32)

    if mode == "silverman":
        h_val = _scott_silverman(positions, rule="silverman", d=d)
        return jnp.full((positions.shape[0],), h_val, dtype=jnp.float32)

    if mode == "knn_adaptive":
        return _knn_bandwidth_bruteforce(
            positions, k=int(knn_k), safety=float(knn_safety), chunk=int(knn_chunk),
        )

    raise ValueError(f"unknown bandwidth mode {mode!r}")


# -----------------------------------------------------------------------------
# Scott / Silverman
# -----------------------------------------------------------------------------

@jax.jit
def _per_dim_std(P: jnp.ndarray) -> jnp.ndarray:
    """Population std along axis 0 with ddof=1, returned per dim."""
    n = jnp.maximum(P.shape[0] - 1, 1)
    mean = jnp.mean(P, axis=0)
    diff = P - mean
    var = jnp.sum(diff * diff, axis=0) / n
    return jnp.sqrt(var)


def _scott_silverman(P: jnp.ndarray, *, rule: str, d: int) -> float:
    sigma = float(jnp.mean(_per_dim_std(P)))
    n = int(P.shape[0])
    factor = 1.0
    if rule == "silverman":
        factor = (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0))
    return factor * sigma * n ** (-1.0 / (d + 4.0))


# -----------------------------------------------------------------------------
# k-NN adaptive bandwidth, brute-force JAX
# -----------------------------------------------------------------------------

def _knn_bandwidth_bruteforce(
    P: jnp.ndarray, *, k: int, safety: float, chunk: int,
) -> jnp.ndarray:
    """
    Compute h_i = safety * dist(i, k-th nearest neighbor) over the particle
    cloud, fully on GPU, chunked over query particles to bound memory.

    For each chunk we compute (chunk, N) pairwise squared distances, then
    take the (k+1)-th smallest (self-distance is at index 0).
    """
    N = int(P.shape[0])
    k_eff = int(min(max(k, 1), max(N - 1, 1)))

    @jax.jit
    def chunk_h(Q: jnp.ndarray) -> jnp.ndarray:
        # Q: (m, 3); pairwise squared dist to all P
        diff = Q[:, None, :] - P[None, :, :]
        d2 = jnp.sum(diff * diff, axis=-1)           # (m, N)
        # k+1 smallest (incl. self at 0). top_k on -d2 then reverse.
        neg_top = jax.lax.top_k(-d2, k_eff + 1)[0]   # (m, k+1)
        d2_kth = -neg_top[:, -1]                     # (m,)
        return jnp.sqrt(jnp.maximum(d2_kth, 0.0))

    out = []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        out.append(chunk_h(P[s:e]))
    h = jnp.concatenate(out, axis=0).astype(jnp.float32) * jnp.float32(safety)
    # Avoid pathological zero h (coincident particles).
    return jnp.maximum(h, jnp.float32(1e-8))


# -----------------------------------------------------------------------------
# Pre-pass utilities
# -----------------------------------------------------------------------------

def particle_bbox(P: jnp.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (min_xyz, max_xyz) of the particle cloud as host numpy arrays."""
    lo = np.asarray(jnp.min(P, axis=0)).astype(np.float32)
    hi = np.asarray(jnp.max(P, axis=0)).astype(np.float32)
    return lo, hi
