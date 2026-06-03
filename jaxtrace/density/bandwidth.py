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
    initial_positions=None,             # (N, d) seeding for "initial_spacing" mode
    d: int = 3,
) -> jnp.ndarray:
    """
    Return per-particle, per-axis smoothing lengths of shape ``(N, d)``,
    float32, on device.

    Modes
    -----
    ``fixed``
        Per-axis or scalar bandwidth via ``fixed_h``; if absent, default
        to ``bandwidth_factor * voxel_size`` (also scalar or per-axis).
    ``scott`` / ``silverman``
        Per-axis bandwidth from the cloud's per-dimension std × the
        usual asymptotic factor. Anisotropic for free.
    ``knn_adaptive``
        Scalar per-particle bandwidth from k-NN distances, broadcast
        across axes.
    ``initial_spacing``
        Per-axis bandwidth = ``bandwidth_factor * Δp_axis``, where
        ``Δp_axis`` is the per-axis inter-particle spacing of the
        ``initial_positions`` argument (see :func:`initial_particle_spacing`).
        This is the standard SPH choice for a uniform initial seeding
        and gives a kernel that resolves the particle scale by
        construction. The reference positions are taken once at runner
        startup and the resulting h is held fixed for the run.
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
        return jnp.broadcast_to(jnp.asarray(h_axis, dtype=jnp.float32), (N, d))

    if mode == "scott":
        h_axis = _scott_silverman_per_axis(positions, rule="scott", d=d)
        return jnp.broadcast_to(jnp.asarray(h_axis, dtype=jnp.float32), (N, d))

    if mode == "silverman":
        h_axis = _scott_silverman_per_axis(positions, rule="silverman", d=d)
        return jnp.broadcast_to(jnp.asarray(h_axis, dtype=jnp.float32), (N, d))

    if mode == "knn_adaptive":
        h_scalar = _knn_bandwidth_bruteforce(
            positions, k=int(knn_k), safety=float(knn_safety), chunk=int(knn_chunk),
        )
        return jnp.broadcast_to(h_scalar[:, None], (N, d)).astype(jnp.float32)

    if mode == "initial_spacing":
        ref_pos = initial_positions if initial_positions is not None else positions
        delta_p_axis = initial_particle_spacing(ref_pos)      # (d,)
        h_axis = (float(bandwidth_factor) * delta_p_axis).astype(np.float32)
        return jnp.broadcast_to(jnp.asarray(h_axis, dtype=jnp.float32), (N, d))

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


# -----------------------------------------------------------------------------
# Initial inter-particle spacing
# -----------------------------------------------------------------------------

def initial_particle_spacing(P) -> np.ndarray:
    """
    Estimate the per-axis inter-particle spacing for an initial seeding.

    For a uniform cartesian grid seeding whose cell aspect matches the
    bbox aspect, the per-axis spacing is

        Delta_p_a = extent_a / N ** (1/d)

    Derivation: if the seeding fills the bbox with N cells whose
    aspect matches the bbox (so that ``Delta_p_a / Delta_p_b ==
    extent_a / extent_b``), then by volume conservation

        Delta_p_x * Delta_p_y * Delta_p_z == V_bbox / N    (3-D)

    combined with the aspect constraint gives
    ``Delta_p_a = extent_a / N ** (1/d)``. Equivalent statements:

      - the equivalent seeding has ``N ** (1/d)`` cells per axis;
      - the per-axis spacing is ``(V_bbox / N) ** (1/d) * (extent_a / geom_mean(extent))``;
      - on a cubic bbox this collapses to the isotropic ``(V/N)^{1/3}``.

    Parameters
    ----------
    P : array, shape (N, d)
        Particle positions (host or device).

    Returns
    -------
    np.ndarray of shape (d,), float32
        Per-axis inter-particle spacing in physical units.
    """
    P_np = np.asarray(P, dtype=np.float64)
    if P_np.ndim != 2:
        raise ValueError(f"expected (N, d), got shape {P_np.shape}")
    N, d = P_np.shape
    if N <= 1:
        raise ValueError(f"need at least 2 particles to estimate spacing, got N={N}")
    lo = P_np.min(axis=0)
    hi = P_np.max(axis=0)
    extent = np.maximum(hi - lo, 1e-30)            # avoid div-by-zero on degenerate axes
    return (extent / float(N) ** (1.0 / d)).astype(np.float32)

