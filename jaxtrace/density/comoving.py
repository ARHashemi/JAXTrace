"""
Co-moving (shifted-frame) preprocessing for transport-dominated flows.

The unified particle cloud and time-averaged density of a welding-like
simulation are dominated by the bulk advection V_adv: most particles
simply travel as x(t) ≈ x_0 + V_adv·t, and the time-averaged density
is essentially a smeared copy of the initial seeding along V_adv.
This bulk transport carries no information about stirring near the
tool, so it dominates the leading SVD/POD modes of any downstream
ROM and buries the physically interesting structure.

This module computes:

  * Particle co-moving displacement
        ξ_i = x_i - x_{i,0} - V_adv · (t_i - t_0)

    where x_{i,0} is the survivor's step-0 seed position (looked up
    via ParticleID) and t_i is its first-seen step time. Particles
    that follow pure bulk advection collapse to ξ_i ≈ 0; deviations
    reveal stirring near the pin.

  * Co-moving reference density and signed residual
        ρ_ref(x, t) = ρ_0(x - V_adv·(t - t_0))
        Δρ̄(x) = ρ̄(x) - ⟨ρ_ref⟩(x)

    where ρ_0 is the density of the step-0 cloud on the same voxel
    grid and ⟨·⟩ averages over the same time interval used to form
    ρ̄. On a uniform Cartesian grid the time integral reduces to a
    linear sweep of trilinear samples along V_adv.

References
----------
* Reiss et al., "The shifted proper orthogonal decomposition"
  https://www.cfd.tu-berlin.de/~reiss/ReissSchulzeSesterhennMehrmann2018.pdf
* The repo design note: ``co-moving_reference_substraction.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Particle co-moving displacement
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CoMovingParticleResult:
    """
    Per-survivor co-moving displacement.

    ``xi`` is the (N, 3) vector ξ_i = x_i - x_{i,0} - V·(t_i - t_0).
    ``xi_magnitude`` is its Euclidean norm (N,).
    """
    xi: np.ndarray
    xi_magnitude: np.ndarray
    drift_velocity: np.ndarray   # (3,) the V used
    time_origin: float           # t_0 used
    n_matched: int               # number of survivors that found their seed
    n_unmatched: int             # survivors with no matching x_{i,0}


def compute_comoving_displacement(
    survivor_positions: np.ndarray,            # (N, 3) lab-frame positions
    survivor_particle_ids: np.ndarray,         # (N,) int — keys into seed table
    survivor_first_seen_time: np.ndarray,      # (N,) float
    seed_positions: np.ndarray,                # (N0, 3) step-0 positions
    seed_particle_ids: np.ndarray,             # (N0,) int — same key space
    drift_velocity: np.ndarray,                # (3,) float
    time_origin: float,
) -> CoMovingParticleResult:
    """
    ξ_i = x_i - x_{i,0} - V · (t_i - t_0)

    Survivors whose ParticleID cannot be matched in the seed table
    (e.g. lost particles, ID truncation, or a re-run on a file that
    lacks them) are assigned ξ_i = (NaN, NaN, NaN) so they are
    visually distinguishable in ParaView and statistically excluded
    by any downstream tool that respects NaN. The matched/unmatched
    counts are returned.
    """
    survivor_positions = np.asarray(survivor_positions, dtype=np.float64)
    survivor_particle_ids = np.asarray(survivor_particle_ids).ravel()
    survivor_first_seen_time = np.asarray(survivor_first_seen_time, dtype=np.float64).ravel()
    drift_velocity = np.asarray(drift_velocity, dtype=np.float64).reshape(3)

    # Build seed table: ParticleID -> row.
    seed_ids = np.asarray(seed_particle_ids).ravel()
    seed_pos = np.asarray(seed_positions, dtype=np.float64)
    # Use a sorted index lookup; ID range can be sparse so np.searchsorted
    # against a sorted index is the right tool.
    order = np.argsort(seed_ids)
    sorted_ids = seed_ids[order]
    pos = np.searchsorted(sorted_ids, survivor_particle_ids)
    pos_clipped = np.clip(pos, 0, sorted_ids.size - 1)
    matched = sorted_ids[pos_clipped] == survivor_particle_ids
    seed_rows = order[pos_clipped]
    x0 = seed_pos[seed_rows]                                   # (N, 3)

    dt = survivor_first_seen_time - float(time_origin)         # (N,)
    drift = drift_velocity[None, :] * dt[:, None]              # (N, 3)
    xi = survivor_positions - x0 - drift                       # (N, 3)
    xi[~matched] = np.nan
    xi_mag = np.linalg.norm(xi, axis=1)

    return CoMovingParticleResult(
        xi=xi.astype(np.float32),
        xi_magnitude=xi_mag.astype(np.float32),
        drift_velocity=drift_velocity.astype(np.float32),
        time_origin=float(time_origin),
        n_matched=int(matched.sum()),
        n_unmatched=int((~matched).sum()),
    )


# -----------------------------------------------------------------------------
# Density co-moving residual on a regular grid
# -----------------------------------------------------------------------------

def _trilinear_sample(
    field: np.ndarray,            # (Nx, Ny, Nz) — physical (i,j,k) order
    origin: np.ndarray,           # (3,) world coord of voxel center [0,0,0]
    spacing: np.ndarray,          # (3,) voxel size
    query: np.ndarray,            # (M, 3) world-space query points
    fill: float = 0.0,
) -> np.ndarray:
    """
    Trilinear interpolation of ``field`` at world-space ``query`` points.

    Points outside the grid get ``fill``. Output is shape ``(M,)``.
    """
    Nx, Ny, Nz = field.shape
    # Convert world -> index (continuous).
    idx = (query - origin[None, :]) / spacing[None, :]
    ix0 = np.floor(idx[:, 0]).astype(np.int64)
    iy0 = np.floor(idx[:, 1]).astype(np.int64)
    iz0 = np.floor(idx[:, 2]).astype(np.int64)
    fx = idx[:, 0] - ix0
    fy = idx[:, 1] - iy0
    fz = idx[:, 2] - iz0
    ix1, iy1, iz1 = ix0 + 1, iy0 + 1, iz0 + 1

    inside = (
        (ix0 >= 0) & (ix1 < Nx) &
        (iy0 >= 0) & (iy1 < Ny) &
        (iz0 >= 0) & (iz1 < Nz)
    )
    # Clamp indices used by the gather so the array access is safe;
    # results outside `inside` are overwritten with `fill` at the end.
    ix0c = np.clip(ix0, 0, Nx - 1); ix1c = np.clip(ix1, 0, Nx - 1)
    iy0c = np.clip(iy0, 0, Ny - 1); iy1c = np.clip(iy1, 0, Ny - 1)
    iz0c = np.clip(iz0, 0, Nz - 1); iz1c = np.clip(iz1, 0, Nz - 1)

    c000 = field[ix0c, iy0c, iz0c]
    c100 = field[ix1c, iy0c, iz0c]
    c010 = field[ix0c, iy1c, iz0c]
    c110 = field[ix1c, iy1c, iz0c]
    c001 = field[ix0c, iy0c, iz1c]
    c101 = field[ix1c, iy0c, iz1c]
    c011 = field[ix0c, iy1c, iz1c]
    c111 = field[ix1c, iy1c, iz1c]

    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    out = c0 * (1 - fz) + c1 * fz
    out = np.where(inside, out, fill)
    return out


# -----------------------------------------------------------------------------
# Uniform-reference density (sPOD residual against a perfect uniform seeding)
# -----------------------------------------------------------------------------

def build_uniform_reference_cloud(
    bbox_lo: np.ndarray,        # (3,) lower corner of the reference box
    bbox_hi: np.ndarray,        # (3,) upper corner
    delta_p: np.ndarray,        # (3,) anisotropic per-axis spacing
) -> np.ndarray:
    """
    Synthesize a Cartesian uniform grid of point positions inside the
    given box with per-axis spacing ``delta_p``. Returns ``(N, 3)``
    float32.

    The grid is built with at least one point per axis. Spacing is
    enforced so that ``ceil(extent / delta_p)`` cells fit; the actual
    spacing per axis is then ``extent / (n_axis - 1)`` when
    ``n_axis > 1`` so the corners sit exactly on the box.
    """
    bbox_lo = np.asarray(bbox_lo, dtype=np.float64).reshape(3)
    bbox_hi = np.asarray(bbox_hi, dtype=np.float64).reshape(3)
    delta_p = np.asarray(delta_p, dtype=np.float64).reshape(3)
    extent = bbox_hi - bbox_lo
    n_axis = np.maximum(1, np.ceil(extent / np.maximum(delta_p, 1e-30)).astype(np.int64)) + 1
    axes = []
    for a in range(3):
        if n_axis[a] == 1:
            axes.append(np.array([0.5 * (bbox_lo[a] + bbox_hi[a])], dtype=np.float64))
        else:
            axes.append(np.linspace(bbox_lo[a], bbox_hi[a], int(n_axis[a]), dtype=np.float64))
    GX, GY, GZ = np.meshgrid(*axes, indexing="ij")
    return np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=-1).astype(np.float32)


@dataclass(frozen=True)
class UniformReferenceResult:
    """
    Result of the uniform-reference density pass.

    ``reference_density`` is ρ_unif(x), the kernel-smoothed density of a
    perfectly uniform Cartesian grid (spacing Δp) filling the reference
    box, evaluated on the same voxel grid as ρ̄.

    ``residual`` is the signed difference ρ̄ − ρ_unif: positive in
    regions material drifted *into* that the seeding did not cover,
    negative in regions depleted relative to the uniform baseline,
    ~zero in pure-advection regions inside the seeding.

    ``normalized`` is ρ̄ / max(ρ_unif, ε) — a ratio centered around 1
    for pure-advection regions. ε is set so a tiny fraction of
    ρ_unif.max() does not produce huge ratios; voxels far outside the
    reference box where ρ_unif ≈ 0 still get bounded ratios.
    """
    reference_density: np.ndarray
    residual: np.ndarray
    normalized: np.ndarray
    n_reference_points: int
    bbox_lo: np.ndarray
    bbox_hi: np.ndarray
    delta_p: np.ndarray


@dataclass(frozen=True)
class CoMovingDensityResult:
    """
    Time-averaged co-moving reference density and residual on the
    same voxel grid the underlying ρ̄ was built on.

    ``reference_density`` is ⟨ρ_ref⟩(x), the time-average of
    ρ_0(x - V·(t - t_0)) over the same time samples used to form ρ̄.
    ``residual`` is the signed difference ρ̄ - ⟨ρ_ref⟩.
    All arrays are shaped (Nx, Ny, Nz) — matching the per-step
    write_time_average output.
    """
    reference_density: np.ndarray
    residual: np.ndarray
    drift_velocity: np.ndarray
    time_origin: float
    t_start: float
    t_end: float


def compute_comoving_reference_density(
    rho_mean: np.ndarray,                # (Nx, Ny, Nz) the existing time-avg
    rho_initial: np.ndarray,             # (Nx, Ny, Nz) density of step-0 cloud
    grid_origin: np.ndarray,             # (3,) voxel-center origin
    grid_spacing: np.ndarray,            # (3,) voxel size
    drift_velocity: np.ndarray,          # (3,) V
    times: np.ndarray,                   # (M,) time samples used for ρ̄
    time_origin: float,
    weights: Optional[np.ndarray] = None,  # (M,) dt weights, defaults to uniform
) -> CoMovingDensityResult:
    """
    Build the time-averaged co-moving reference field on the same
    grid as ``rho_mean`` and return the residual ``rho_mean - <rho_ref>``.

    For each time sample t_j the reference is ρ_0(x - V·(t_j - t_0)),
    which on a uniform grid we evaluate by trilinearly sampling
    ``rho_initial`` at the shifted voxel-center positions. The shift
    is constant per slab — so we do one trilinear pass per sample,
    plus a weighted sum across samples.

    The ``weights`` argument supports non-uniform time integration
    (e.g. variable dt). If ``None`` the average is the simple mean
    over ``times``.
    """
    Nx, Ny, Nz = rho_mean.shape
    grid_origin = np.asarray(grid_origin, dtype=np.float64).reshape(3)
    grid_spacing = np.asarray(grid_spacing, dtype=np.float64).reshape(3)
    drift_velocity = np.asarray(drift_velocity, dtype=np.float64).reshape(3)
    times = np.asarray(times, dtype=np.float64).ravel()
    if weights is None:
        w = np.ones_like(times) / max(times.size, 1)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        w = w / max(w.sum(), 1e-30)

    # World-space voxel centers (Nx*Ny*Nz, 3).
    ix = np.arange(Nx); iy = np.arange(Ny); iz = np.arange(Nz)
    cx = grid_origin[0] + ix * grid_spacing[0]
    cy = grid_origin[1] + iy * grid_spacing[1]
    cz = grid_origin[2] + iz * grid_spacing[2]
    CX, CY, CZ = np.meshgrid(cx, cy, cz, indexing="ij")
    centers = np.stack([CX.ravel(), CY.ravel(), CZ.ravel()], axis=-1)  # (M, 3)

    rho_initial_f = np.asarray(rho_initial, dtype=np.float64)

    accum = np.zeros((Nx * Ny * Nz,), dtype=np.float64)
    for t, wj in zip(times, w):
        shift = drift_velocity * (float(t) - float(time_origin))
        sample = _trilinear_sample(
            rho_initial_f, grid_origin, grid_spacing,
            centers - shift[None, :], fill=0.0,
        )
        accum += wj * sample

    ref = accum.reshape(Nx, Ny, Nz)
    residual = rho_mean.astype(np.float64) - ref
    return CoMovingDensityResult(
        reference_density=ref.astype(np.float32),
        residual=residual.astype(np.float32),
        drift_velocity=drift_velocity.astype(np.float32),
        time_origin=float(time_origin),
        t_start=float(times[0]) if times.size else float(time_origin),
        t_end=float(times[-1]) if times.size else float(time_origin),
    )
