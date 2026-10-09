"""
Leave-one-out cross-validation (LOOCV) of the PCA + regression surrogate.

PCA alone cannot *predict* a held-out field: reconstructing case *i*
needs its mode coefficients, which depend on the held-out density. The
surrogate therefore couples PCA with a regressor that maps the two
process inputs ``(v_adv, omega_pin)`` to the mode coefficients. LOOCV
tests that whole pipeline:

  for each case i:
    1. fit PCA on the other n-1 cases            -> mean, modes Phi
    2. fit regressor (inputs -> coeffs) on them
    3. predict held-out coeffs from i's inputs
    4. reconstruct rho_hat_i = mean + sum_k A_ik Phi_k  (chain undone)
    5. error_i = ||rho_hat_i - rho_i|| / ||rho_i||

This isolates *generalization* error, unlike the train-on-all projection
error (which only measures basis expressiveness).

Regressors (all dependency-light, fit one coefficient at a time):
  * ``rbf``    - scipy RBFInterpolator (thin-plate / multiquadric)
  * ``gp``     - a small anisotropic-RBF Gaussian process in NumPy
  * ``poly``   - least-squares polynomial in (v_adv, omega_pin)

The per-case amplitude removed by a ``per_case_*`` normalization is
itself regressed on the inputs so reconstructions return physical units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .pca import fit_pca


# ---------------------------------------------------------------------------
# Input normalization (so regressors see O(1) features)
# ---------------------------------------------------------------------------

def _standardize_fit(P: np.ndarray):
    """Return (mu, sd) for column-wise standardization of inputs."""
    mu = P.mean(axis=0)
    sd = P.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def _standardize(P: np.ndarray, mu, sd) -> np.ndarray:
    return (P - mu) / sd


# ---------------------------------------------------------------------------
# Regressors: each is fit(X_train, Y_train) -> predict(X_query)
# Y is (n_train, n_targets); we fit all targets together where possible.
# ---------------------------------------------------------------------------

def _make_rbf(kernel: str = "thin_plate_spline"):
    from scipy.interpolate import RBFInterpolator

    def fit(Xtr: np.ndarray, Ytr: np.ndarray) -> Callable:
        # smoothing=0 -> exact interpolation at the training points.
        interp = RBFInterpolator(Xtr, Ytr, kernel=kernel, smoothing=0.0)
        return lambda Xq: np.asarray(interp(Xq))

    return fit


def _make_poly(degree: int = 2):
    import itertools

    def _features(X: np.ndarray) -> np.ndarray:
        # X is (n, d) standardized features. Build all monomials up to
        # `degree` over the d columns (works for d = 1 or 2).
        n, d = X.shape
        cols = [np.ones(n)]
        for deg in range(1, degree + 1):
            for combo in itertools.combinations_with_replacement(range(d), deg):
                term = np.ones(n)
                for j in combo:
                    term = term * X[:, j]
                cols.append(term)
        return np.stack(cols, axis=1)

    def fit(Xtr: np.ndarray, Ytr: np.ndarray) -> Callable:
        Ftr = _features(Xtr)
        # Least squares for all targets at once: (F^+ Y).
        coef, *_ = np.linalg.lstsq(Ftr, Ytr, rcond=None)
        return lambda Xq: _features(Xq) @ coef

    return fit


def _make_gp(length_scale: Optional[float] = None, noise: float = 1e-6):
    """
    Minimal Gaussian-process regressor with an isotropic RBF kernel on the
    standardized inputs. One GP shared across all coefficient targets
    (same kernel, different right-hand sides), which is the standard
    multi-output GP with a shared covariance.
    """

    def _kernel(A: np.ndarray, B: np.ndarray, ell: float) -> np.ndarray:
        d2 = (
            np.sum(A * A, axis=1)[:, None]
            + np.sum(B * B, axis=1)[None, :]
            - 2.0 * A @ B.T
        )
        return np.exp(-0.5 * np.maximum(d2, 0.0) / (ell * ell))

    def fit(Xtr: np.ndarray, Ytr: np.ndarray) -> Callable:
        n = Xtr.shape[0]
        # Heuristic length scale: median pairwise distance (standardized).
        if length_scale is None:
            d2 = (
                np.sum(Xtr * Xtr, axis=1)[:, None]
                + np.sum(Xtr * Xtr, axis=1)[None, :]
                - 2.0 * Xtr @ Xtr.T
            )
            dists = np.sqrt(np.maximum(d2[np.triu_indices(n, 1)], 0.0))
            ell = float(np.median(dists)) if dists.size else 1.0
            ell = ell if ell > 0 else 1.0
        else:
            ell = float(length_scale)

        K = _kernel(Xtr, Xtr, ell) + noise * np.eye(n)
        L = np.linalg.cholesky(K)
        # alpha = K^{-1} Ytr  (solve once for all targets)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Ytr))

        def predict(Xq: np.ndarray) -> np.ndarray:
            Ks = _kernel(Xq, Xtr, ell)
            return Ks @ alpha

        return predict

    return fit


REGRESSORS: Dict[str, Callable] = {
    "rbf": _make_rbf(),
    "gp": _make_gp(),
    "poly": _make_poly(degree=2),
}


# ---------------------------------------------------------------------------
# LOOCV
# ---------------------------------------------------------------------------

@dataclass
class LOOCVResult:
    """Per-case held-out errors for one regressor, swept over mode counts."""

    regressor: str
    normalize: str
    k_values: np.ndarray            # (n_k,) mode counts swept
    # (n_k, n_cases) relative L2 error of the held-out reconstruction.
    rel_error: np.ndarray
    params: np.ndarray              # (n_cases, 2) [v_adv, omega_pin]
    case_numbers: List[str]

    @property
    def mean_error(self) -> np.ndarray:
        """(n_k,) mean held-out error vs mode count."""
        return self.rel_error.mean(axis=1)

    @property
    def best_k(self) -> int:
        return int(self.k_values[int(np.argmin(self.mean_error))])

    def error_at_best_k(self) -> np.ndarray:
        """(n_cases,) per-case error at the LOOCV-optimal mode count."""
        ki = int(np.argmin(self.mean_error))
        return self.rel_error[ki]


def _reconstruct_holdout(
    pca,
    coeffs_pred: np.ndarray,   # (K,) predicted working-space coeffs
    scale_pred: float,         # predicted per-case scale (1.0 unless per_case_*)
    k: int,
) -> np.ndarray:
    """Reconstruct one held-out field from predicted coefficients."""
    fluct = coeffs_pred[:k] @ pca.modes[:k]          # working space
    y = pca.mean + pca.global_scale * fluct          # undo global scale, add mean
    return scale_pred * y                            # undo per-case scale


def loocv(
    matrix: np.ndarray,
    params: np.ndarray,
    case_numbers: List[str],
    regressor: str = "rbf",
    normalize: str = "none",
    k_values: Optional[np.ndarray] = None,
    feature_transform: str = "identity",
    extra_features: Optional[np.ndarray] = None,
    use_base_features: bool = True,
    verbose: bool = True,
) -> LOOCVResult:
    """
    Leave-one-out CV of the PCA+regressor surrogate over a sweep of mode
    counts ``k_values`` (default 1..n-1).

    ``feature_transform`` selects how the raw ``(v_adv, omega_pin)`` inputs
    are mapped to regressor features (see ``features.FEATURE_TRANSFORMS``);
    e.g. ``pitch_omega`` regresses on the weld pitch instead of v_adv.

    ``extra_features`` is an optional (n_cases, d) matrix appended to the
    regressor inputs — e.g. first-stage ROM velocity/temperature
    coefficients. Set ``use_base_features=False`` to regress on the extra
    features ALONE (dropping the transformed (v_adv, omega) entirely).
    All features are standardised per fold before the regressor sees them.

    Errors are relative L2 in the SVD working space's *physical* units
    (the per-case and global scales are undone). For ``normalize="log"``
    the comparison is in log-density space (the log is intentionally not
    inverted), so treat those numbers as relative, not absolute.
    """
    if regressor not in REGRESSORS:
        raise ValueError(f"regressor={regressor!r} not in {list(REGRESSORS)}")
    fit_reg = REGRESSORS[regressor]

    from .features import FEATURE_TRANSFORMS
    if feature_transform not in FEATURE_TRANSFORMS:
        raise ValueError(f"feature_transform={feature_transform!r} not in "
                         f"{list(FEATURE_TRANSFORMS)}")
    feat = FEATURE_TRANSFORMS[feature_transform]

    X = np.asarray(matrix, dtype=np.float64)
    P = feat(np.asarray(params, dtype=np.float64))   # (n, d) base features
    if not use_base_features:
        if extra_features is None:
            raise ValueError("use_base_features=False requires extra_features")
        P = np.empty((X.shape[0], 0))
    if extra_features is not None:
        ef = np.asarray(extra_features, dtype=np.float64)
        if ef.shape[0] != X.shape[0]:
            raise ValueError(f"extra_features rows {ef.shape[0]} != "
                             f"n_cases {X.shape[0]}")
        P = np.concatenate([P, ef], axis=1) if P.shape[1] else ef
    n_cases = X.shape[0]
    if k_values is None:
        # At most n-1 non-trivial modes from n-1 training snapshots, and
        # the mean drops one more -> n-2 available per fold. Cap there.
        k_values = np.arange(1, n_cases - 1)
    k_values = np.asarray(k_values, dtype=int)
    k_max = int(k_values.max())

    # Held-out comparison should be in the same space the surrogate
    # predicts: for normalize="log", compare log1p(rho); else raw rho.
    use_log = normalize == "log"

    rel_error = np.zeros((len(k_values), n_cases), dtype=np.float64)

    for i in range(n_cases):
        train = np.array([j for j in range(n_cases) if j != i])
        Xtr, Ptr = X[train], P[train]
        xi, pi = X[i], P[i][None, :]

        # PCA on the training fold.
        pca = fit_pca(Xtr, normalize=normalize)

        # Standardize inputs on the fold, fit regressor coeffs(P)->A.
        mu, sd = _standardize_fit(Ptr)
        Ptr_s = _standardize(Ptr, mu, sd)
        pi_s = _standardize(pi, mu, sd)

        ncoef = pca.coeffs.shape[1]
        kk = min(k_max, ncoef)
        predict = fit_reg(Ptr_s, pca.coeffs[:, :kk])
        coeffs_pred = np.asarray(predict(pi_s)).ravel()  # (kk,)

        # Predict the per-case amplitude too (1.0 unless per_case_*).
        if pca.per_case_scale is not None:
            spred_fn = fit_reg(Ptr_s, pca.per_case_scale[:, None])
            scale_pred = float(np.asarray(spred_fn(pi_s)).ravel()[0])
        else:
            scale_pred = 1.0

        # Truth in the comparison space.
        truth = np.log1p(np.maximum(xi, 0.0)) if use_log else xi
        denom = np.linalg.norm(truth)
        denom = denom if denom > 0 else 1.0

        for ik, k in enumerate(k_values):
            kc = min(int(k), kk)
            rho_hat = _reconstruct_holdout(pca, coeffs_pred, scale_pred, kc)
            pred = np.log1p(np.maximum(rho_hat, 0.0)) if use_log else rho_hat
            rel_error[ik, i] = np.linalg.norm(pred - truth) / denom

        if verbose:
            ebest = rel_error[:, i].min()
            print(f"[cv] {regressor}/{normalize} fold {i:2d} "
                  f"(case {case_numbers[i]}): best rel-err={ebest:.3f}")

    return LOOCVResult(
        regressor=regressor,
        normalize=normalize,
        k_values=k_values,
        rel_error=rel_error,
        params=P,
        case_numbers=list(case_numbers),
    )
