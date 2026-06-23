"""
PCA / POD of the snapshot matrix via ``jax.numpy.linalg.svd``.

Given a snapshot matrix ``X`` of shape ``(n_cases, n_voxels)`` (one
density field per row), PCA finds an orthonormal basis (POD modes) that
captures the most field variance with the fewest modes. With only
``n_cases`` snapshots the matrix is short-and-fat, so at most
``n_cases - 1`` non-trivial modes exist after mean-centring.

We mean-centre, then SVD the centred matrix::

    X_c = X - mean(X, axis=0)          # (n, m)
    X_c = U @ diag(S) @ Vt             # economy SVD

The right singular vectors (rows of ``Vt``) are the spatial POD modes
(length ``n_voxels``); ``S`` are the singular values. The variance
"energy" captured by mode ``k`` is ``S[k]**2``, and cumulative energy
coverage is ``cumsum(S**2) / sum(S**2)`` — the elbow plot used to pick
the number of retained modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PCAResult:
    """Outputs of a mean-centred SVD of the snapshot matrix."""

    mean: np.ndarray            # (n_voxels,) snapshot mean (the POD "mode 0")
    modes: np.ndarray           # (n_modes, n_voxels) spatial POD modes (Vt)
    singular_values: np.ndarray  # (n_modes,) singular values S
    coeffs: np.ndarray          # (n_cases, n_modes) per-case mode amplitudes U*S
    n_cases: int

    @property
    def energy(self) -> np.ndarray:
        """Variance captured per mode (S**2)."""
        return self.singular_values ** 2

    @property
    def energy_fraction(self) -> np.ndarray:
        """Per-mode fraction of total variance."""
        e = self.energy
        total = e.sum()
        return e / total if total > 0 else e

    @property
    def cumulative_energy(self) -> np.ndarray:
        """Cumulative fraction of variance covered by the first k modes."""
        return np.cumsum(self.energy_fraction)

    def n_modes_for(self, coverage: float) -> int:
        """Smallest #modes whose cumulative energy reaches ``coverage``."""
        cum = self.cumulative_energy
        idx = int(np.searchsorted(cum, coverage) + 1)
        return min(idx, len(self.singular_values))

    def reconstruct(self, n_modes: Optional[int] = None) -> np.ndarray:
        """
        Rebuild the snapshot matrix from the first ``n_modes`` modes.

        Returns (n_cases, n_voxels). With ``n_modes=None`` uses all modes
        (exact up to round-off).
        """
        k = len(self.singular_values) if n_modes is None else int(n_modes)
        return self.mean[None, :] + self.coeffs[:, :k] @ self.modes[:k]


def fit_pca(matrix: np.ndarray, dtype=np.float64) -> PCAResult:
    """
    Mean-centred PCA/POD of ``matrix`` (n_cases, n_voxels) via the
    *method of snapshots*.

    For ``n_voxels >> n_cases`` a direct SVD of the wide centred matrix
    is both wasteful and numerically fragile on GPU solvers. Instead we
    apply ``jax.numpy.linalg.svd`` to the small ``(n_cases, n_cases)``
    Gram matrix ``G = Xc @ Xc.T`` and recover the spatial modes::

        G = V diag(sigma**2) V.T   (svd of symmetric PSD G)
        sigma_i = sqrt(Sg_i)
        phi_i = (1 / sigma_i) * Xc.T @ v_i     # spatial POD mode

    This gives identical singular values / modes to the economy SVD of
    ``Xc`` (up to sign) while only ever factoring an ``n×n`` matrix.
    Computation is float64 for spectral stability.
    """
    import jax
    import jax.numpy as jnp

    # float64 needs x64 enabled; do it before any array is created.
    if dtype == np.float64:
        jax.config.update("jax_enable_x64", True)

    X = jnp.asarray(np.asarray(matrix, dtype=dtype))
    n_cases = int(X.shape[0])

    mean = jnp.mean(X, axis=0)
    Xc = X - mean[None, :]

    # Small symmetric Gram matrix (n×n). Its SVD is cheap and robust on
    # GPU (unlike gesvd on the wide m≈1.7M matrix). For a symmetric PSD
    # matrix, svd(G) = Vg @ diag(Sg) @ Vg.T with Sg = sigma**2 and the
    # left singular vectors == right == the snapshot eigenvectors.
    G = Xc @ Xc.T
    Vg, Sg, _ = jnp.linalg.svd(G)  # jax.numpy.linalg.svd, n×n
    evecs = Vg
    S = jnp.sqrt(jnp.clip(Sg, 0.0))

    # Spatial modes phi_i = Xc.T @ v_i / sigma_i, transposed to (n, m).
    inv_s = jnp.where(S > 0, 1.0 / S, 0.0)
    # modes (n, m): each row k is (Xc.T @ evecs[:,k]) * inv_s[k] -> use
    # (evecs * inv_s).T @ Xc = ((evecs * inv_s).T) @ Xc.
    modes = (evecs * inv_s[None, :]).T @ Xc  # (n, m)

    # Per-case amplitudes coeffs = U @ diag(S) = Xc @ modes.T = evecs * S.
    coeffs = evecs * S[None, :]

    return PCAResult(
        mean=np.asarray(mean, dtype=np.float64),
        modes=np.asarray(modes, dtype=np.float64),
        singular_values=np.asarray(S, dtype=np.float64),
        coeffs=np.asarray(coeffs, dtype=np.float64),
        n_cases=n_cases,
    )
