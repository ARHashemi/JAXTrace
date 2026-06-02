# jaxtrace/density/bandwidth.py
"""
Smoothing-length / bandwidth resolution strategies.

This module produces per-particle bandwidths of shape ``(N, d)`` —
i.e. **per-axis**, anisotropic. For modes whose math is naturally
isotropic (a scalar ``h``) we still return ``(N, d)`` so the rest of
the pipeline only has to deal with one shape; the per-axis entries are
identical for isotropic modes.

Modes
-----
- ``fixed``         : per-particle ``h_i = (hx, hy, hz)``.
                      Scalar input is broadcast across all axes; an
                      explicit 3-vector overrides.
- ``scott``         : per-axis Scott's rule, ``h_a = σ_a · N^{-1/(d+4)}``,
                      where ``σ_a`` is the per-dimension std of the cloud.
                      Identical to the scalar Scott if the cloud is
                      isotropic, but adapts to anisotropic clouds.
- ``silverman``     : per-axis Silverman's rule,
                      ``h_a = (4/(d+2))^{1/(d+4)} · σ_a · N^{-1/(d+4)}``.
- ``knn_adaptive``  : per-particle scalar
                      ``h_i = safety · dist(i, k-th NN)``, broadcast
                      across axes. The fully per-axis k-NN extension
                      (anisotropic neighbourhood ellipsoid) is not
                      implemented yet — TODO.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np


# Per-axis bandwidth input: either a scalar (isotropic) or a 3-vector.
HInput = Union[float, Sequence[float], np.ndarray, jnp.ndarray, None]


def _broadcast_to_per_axis(value: HInput, d: int) -> np.ndarray:
    """Normalise a scalar / 3-vector input to a host (d,) numpy array."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        return np.full((d,), float(arr[0]), dtype=np.float32)
    if arr.size != d:
        raise ValueError(
            f"per-axis bandwidth must be a scalar or a {d}-vector, got shape {arr.shape}"
        )
    return arr


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def resolve_bandwidth(
    positions: jnp.ndarray,            # (N, d) device array, float32
    mode: str,
    *,
    voxel_size: HInput = None,          # scalar or per-axis (d,)
    fixed_h: HInput = None,             # scalar or per-axis (d,)
    bandwidth_factor: float = 2.0,
    knn_k: int = 32,
    knn_safety: float = 1.2,
    knn_chunk: int = 4096,
    d: int = 3,
) -> jnp.ndarray:
    """
    Return per-particle, per-axis smoothing lengths of shape ``(N, d)``,
    float32, on device.

    ``fixed_h`` and ``voxel_size`` may each be a scalar (isotropic) or a
    per-axis sequence of length ``d``. For modes other than ``fixed``,
    these arguments are unused; Scott/Silverman compute per-axis ``σ``
    from the cloud, k-NN uses a scalar per particle.
    """
    N = positions.shape[0]

    if mode == "fixed":
        fixed_arr = _broadcast_to_per_axis(fixed_h, d)
        if fixed_arr is None:
            vs_arr = _broadcast_to_per_axis(voxel_size, d)
            if vs_arr is None:
                raise ValueError("fixed bandwidth requires fixed_h or voxel_size")
            h_axis = float(bandwidth_factor) * vs_arr
        else:
            h_axis = fixed_arr
        # Broadcast to (N, d) — identical row per particle.
        return jnp.broadcast_to(jnp.asarray(h_axis, dtype=jnp.float32), (N, d))

    if mode == "scott":
        h_axis = _scott_silverman_per_axis(positions, rule="scott", d=d)
        return jnp.broadcast_to(jnp.asarray(h_axis, dtype=jnp.float32), (N, d))

    if mode == "silverman":
        h_axis = _scott_silverman_per_axis(positions, rule="silverman", d=d)
        return jnp.broadcast_to(jnp.asarray(h_axis, dtype=jnp.float32), (N, d))

    if mode == "knn_adaptive":
        # Scalar per-particle bandwidth, broadcast across axes. A true
        # anisotropic k-NN (ellipsoidal neighbourhood) would compute the
        # per-axis std of the k nearest displacements; not implemented yet.
        h_scalar = _knn_bandwidth_bruteforce(
            positions, k=int(knn_k), safety=float(knn_safety), chunk=int(knn_chunk),
        )                                       # (N,)
        return jnp.broadcast_to(h_scalar[:, None], (N, d)).astype(jnp.float32)

    raise ValueError(f"unknown bandwidth mode {mode!r}")


# -----------------------------------------------------------------------------
# Scott / Silverman (per-axis)
# -----------------------------------------------------------------------------

@jax.jit
def _per_dim_std(P: jnp.ndarray) -> jnp.ndarray:
    """Population std along axis 0 with ddof=1, returned per dim — shape ``(d,)``."""
    n = jnp.maximum(P.shape[0] - 1, 1)
    mean = jnp.mean(P, axis=0)
    diff = P - mean
    var = jnp.sum(diff * diff, axis=0) / n
    return jnp.sqrt(var)


def _scott_silverman_per_axis(P: jnp.ndarray, *, rule: str, d: int) -> np.ndarray:
    """Per-axis Scott/Silverman, returning a host ``(d,)`` numpy array."""
    sigma = np.asarray(_per_dim_std(P))            # (d,)
    n = int(P.shape[0])
    factor = 1.0
    if rule == "silverman":
        factor = (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0))
    return (factor * sigma * n ** (-1.0 / (d + 4.0))).astype(np.float32)


# -----------------------------------------------------------------------------
# k-NN adaptive bandwidth, brute-force JAX (returns scalar per particle)
# -----------------------------------------------------------------------------

def _knn_bandwidth_bruteforce(
    P: jnp.ndarray, *, k: int, safety: float, chunk: int,
) -> jnp.ndarray:
    """
    Compute ``h_i = safety · dist(i, k-th nearest neighbour)`` over the
    particle cloud, fully on GPU, chunked over query particles to bound
    memory.

    Returns a scalar bandwidth per particle, shape ``(N,)``. The caller
    broadcasts it across axes if a per-axis array is needed.
    """
    N = int(P.shape[0])
    k_eff = int(min(max(k, 1), max(N - 1, 1)))

    @jax.jit
    def chunk_h(Q: jnp.ndarray) -> jnp.ndarray:
        diff = Q[:, None, :] - P[None, :, :]
        d2 = jnp.sum(diff * diff, axis=-1)
        neg_top = jax.lax.top_k(-d2, k_eff + 1)[0]
        d2_kth = -neg_top[:, -1]
        return jnp.sqrt(jnp.maximum(d2_kth, 0.0))

    out = []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        out.append(chunk_h(P[s:e]))
    h = jnp.concatenate(out, axis=0).astype(jnp.float32) * jnp.float32(safety)
    return jnp.maximum(h, jnp.float32(1e-8))


# -----------------------------------------------------------------------------
# Pre-pass utilities
# -----------------------------------------------------------------------------

def particle_bbox(P: jnp.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (min_xyz, max_xyz) of the particle cloud as host numpy arrays."""
    lo = np.asarray(jnp.min(P, axis=0)).astype(np.float32)
    hi = np.asarray(jnp.max(P, axis=0)).astype(np.float32)
    return lo, hi
