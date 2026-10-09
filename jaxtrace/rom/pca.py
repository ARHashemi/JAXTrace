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
    """Outputs of a mean-centred (optionally scaled) SVD of the snapshot
    matrix.

    The SVD is applied to ``X_scaled = (X - mean) / global_scale`` after
    an optional ``per_case_scale`` was divided out of each snapshot first.
    ``reconstruct`` undoes both so the returned fields are in physical
    density units. See ``fit_pca`` for the precise normalization chain.
    """

    mean: np.ndarray            # (n_voxels,) snapshot mean (the POD "mode 0")
    modes: np.ndarray           # (n_modes, n_voxels) spatial POD modes (Vt)
    singular_values: np.ndarray  # (n_modes,) singular values S
    coeffs: np.ndarray          # (n_cases, n_modes) per-case mode amplitudes U*S
    n_cases: int
    normalize: str = "none"     # which normalization was applied
    global_scale: float = 1.0   # single scalar divided out before SVD
    # (n_cases,) per-snapshot scale divided out *before* centring; all-ones
    # unless normalize == "per_case_*".
    per_case_scale: Optional[np.ndarray] = None

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
        Rebuild the snapshot matrix in physical density units from the
        first ``n_modes`` modes, undoing the normalization chain.

        Returns (n_cases, n_voxels). With ``n_modes=None`` uses all modes
        (exact up to round-off).
        """
        k = len(self.singular_values) if n_modes is None else int(n_modes)
        # coeffs @ modes live in the *globally-scaled, centred* space:
        #     (Y - mean) / g  ≈  coeffs @ modes
        # where Y is the (optionally per-case-scaled, optionally logged)
        # field and `mean` is its per-voxel mean (NOT scaled by g). Undo
        # the global scale on the fluctuation only, then re-add the mean:
        #     Y ≈ mean + g * (coeffs @ modes)
        fluct = self.coeffs[:, :k] @ self.modes[:k]
        recon = self.mean[None, :] + self.global_scale * fluct
        # Undo the per-case scale (applied before centring).
        if self.per_case_scale is not None:
            recon = recon * self.per_case_scale[:, None]
        return recon


# Normalization options for ``fit_pca``. See the docstring for the
# precise effect of each.
NORMALIZE_CHOICES = (
    "none",            # mean-centre only (covariance POD; standard)
    "global_std",      # ÷ single global std of all centred entries
    "global_max",      # ÷ single global max |centred entry|
    "global_frobenius",  # ÷ Frobenius norm of the centred matrix
    "per_case_l2",     # ÷ per-snapshot L2 norm before centring
    "per_case_mass",   # ÷ per-snapshot sum (mass) before centring
    "per_case_max",    # ÷ per-snapshot max before centring
    "log",             # SVD of log1p-transformed field (then mean-centre)
)


def fit_pca(
    matrix: np.ndarray,
    normalize: str = "none",
    dtype=np.float64,
) -> PCAResult:
    """
    Mean-centred (optionally scaled) PCA/POD of ``matrix``
    (n_cases, n_voxels) via the *method of snapshots*.

    Normalization chain (in order)::

        1. per-case scale   : Y = X / s_case[:, None]   (only per_case_*)
        2. log transform    : Y = log1p(Y)              (only normalize="log")
        3. per-voxel centre : Yc = Y - mean(Y, axis=0)  (ALWAYS)
        4. global scale     : Z = Yc / g                (global_* options)

    The SVD is applied to ``Z``. Steps 1 and 4 are undone by
    ``reconstruct``; the log transform (step 2) is *not* inverted there
    (reconstruction would live in log-density space) — it is offered as a
    spectral-shape diagnostic, not a physical-units reconstruction.

    ``normalize`` options
    ---------------------
    ``none``
        Mean-centre only. The standard covariance POD; modes stay in
        physical density units. **Default and recommended for the elbow.**
    ``global_std`` / ``global_max`` / ``global_frobenius``
        Divide every centred entry by a *single scalar* (global std,
        global max-abs, or Frobenius norm). This does **not** change the
        modes or the relative spectrum — it only rescales the singular
        values / coefficients to O(1). Useful only to keep coefficients
        numerically tame for downstream regression.
    ``per_case_l2`` / ``per_case_mass`` / ``per_case_max``
        Divide each snapshot by its own L2 norm / total mass / peak
        *before* centring. This **does** change the model: it removes
        per-case amplitude (the ~5× peak variation across cases) so the
        modes capture *shape* variation, with the scalar amplitude
        regressed separately.
    ``log``
        SVD of ``log1p`` of the field. Emphasises low-density structure
        when density spans orders of magnitude.

    For ``n_voxels >> n_cases`` a direct SVD of the wide centred matrix
    is both wasteful and numerically fragile on GPU solvers, so we apply
    ``jax.numpy.linalg.svd`` to the small ``(n_cases, n_cases)`` Gram
    matrix ``G = Zc @ Zc.T`` and recover the spatial modes::

        G = V diag(sigma**2) V.T   (svd of symmetric PSD G)
        sigma_i = sqrt(Sg_i)
        phi_i = (1 / sigma_i) * Zc.T @ v_i     # spatial POD mode

    Computation is float64 for spectral stability.
    """
    if normalize not in NORMALIZE_CHOICES:
        raise ValueError(
            f"normalize={normalize!r} not in {NORMALIZE_CHOICES}"
        )

    import jax
    import jax.numpy as jnp

    # float64 needs x64 enabled; do it before any array is created.
    if dtype == np.float64:
        jax.config.update("jax_enable_x64", True)

    X = jnp.asarray(np.asarray(matrix, dtype=dtype))
    n_cases = int(X.shape[0])

    # --- step 1: per-case scale (before centring) ---
    per_case_scale_arr: Optional[np.ndarray] = None
    if normalize == "per_case_l2":
        s_case = jnp.linalg.norm(X, axis=1)
    elif normalize == "per_case_mass":
        s_case = jnp.sum(X, axis=1)
    elif normalize == "per_case_max":
        s_case = jnp.max(X, axis=1)
    else:
        s_case = None
    if s_case is not None:
        s_case = jnp.where(s_case > 0, s_case, 1.0)
        X = X / s_case[:, None]
        per_case_scale_arr = np.asarray(s_case, dtype=np.float64)

    # --- step 2: log transform ---
    if normalize == "log":
        X = jnp.log1p(jnp.clip(X, 0.0))

    # --- step 3: per-voxel mean centring (always) ---
    mean = jnp.mean(X, axis=0)
    Xc = X - mean[None, :]

    # --- step 4: single global scalar ---
    if normalize == "global_std":
        g = jnp.std(Xc)
    elif normalize == "global_max":
        g = jnp.max(jnp.abs(Xc))
    elif normalize == "global_frobenius":
        g = jnp.linalg.norm(Xc)
    else:
        g = jnp.asarray(1.0, dtype=Xc.dtype)
    g = jnp.where(g > 0, g, 1.0)
    Xc = Xc / g
    global_scale = float(g)

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
        normalize=normalize,
        global_scale=global_scale,
        per_case_scale=per_case_scale_arr,
    )
