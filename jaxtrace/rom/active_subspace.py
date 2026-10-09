"""
Active-subspace analysis of the input -> mode-coefficient map.

The surrogate's hard part is the regression g: (v_adv, omega_pin) -> POD
coefficients A. Active subspaces ask: is there a dominant *direction* in
the 2D input space along which A varies most? If one eigenvalue of the
gradient-outer-product matrix C dominates, the map is effectively 1D and
that eigenvector is the feature to regress on (e.g. it may align with the
weld-pitch direction); if the two eigenvalues are comparable, the problem
is genuinely 2D and no single combined input will collapse it.

We have only ~17 scattered samples, so gradients are estimated by a local
linear (least-squares) fit of each coefficient vs the *standardised*
inputs — i.e. the global linear sensitivity, which for a small LHS design
is the honest first-order active-subspace estimate:

    A_k ≈ b_k0 + b_k · z,   z = standardised (v_adv, omega_pin)
    C = sum_k w_k b_k b_k^T  (w_k = mode energy weight)
    eigendecompose C -> (eigenvalues, eigenvectors)

Eigenvalue ratio lambda_1 / lambda_2 >> 1  =>  near-1D (active direction
= eigvec_1). Ratio ~ 1  =>  genuinely 2D.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .pca import fit_pca


@dataclass
class ActiveSubspaceResult:
    eigenvalues: np.ndarray         # (2,) descending
    eigenvectors: np.ndarray        # (2, 2), columns are directions in (v,omega) std space
    input_mean: np.ndarray          # (2,) for de-standardising
    input_std: np.ndarray           # (2,)
    n_modes_used: int

    @property
    def eigval_ratio(self) -> float:
        e = self.eigenvalues
        return float(e[0] / e[1]) if e[1] > 0 else float("inf")

    @property
    def activity_share(self) -> np.ndarray:
        """Fraction of activity per direction (eigenvalue share)."""
        e = self.eigenvalues
        return e / e.sum() if e.sum() > 0 else e

    def active_direction_raw(self) -> np.ndarray:
        """Dominant direction expressed in raw (v_adv, omega) units
        (un-standardised, normalised)."""
        d = self.eigenvectors[:, 0] / self.input_std
        return d / np.linalg.norm(d)


def active_subspace(
    matrix: np.ndarray,
    params: np.ndarray,
    n_modes: Optional[int] = None,
    energy_weighted: bool = True,
    normalize: str = "none",
) -> ActiveSubspaceResult:
    """
    First-order active subspace of the (v_adv, omega_pin) -> coefficient
    map, using a global linear sensitivity fit per mode.

    ``n_modes`` caps how many leading POD coefficients are included
    (default: all non-trivial). ``energy_weighted`` weights each mode's
    gradient contribution by its singular-value energy (so dominant modes
    count more), matching how reconstruction error is dominated.
    """
    X = np.asarray(matrix, dtype=np.float64)
    P = np.asarray(params, dtype=np.float64)[:, :2]

    pca = fit_pca(X, normalize=normalize)
    A = pca.coeffs                                   # (n, n_modes)
    nmax = A.shape[1]
    k = nmax if n_modes is None else min(int(n_modes), nmax)

    mu = P.mean(axis=0)
    sd = P.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (P - mu) / sd                                # standardised inputs
    G = np.column_stack([np.ones(Z.shape[0]), Z])    # design [1, zv, zw]

    weights = pca.energy_fraction[:k] if energy_weighted else np.ones(k)

    C = np.zeros((2, 2))
    for j in range(k):
        # least-squares linear fit A[:,j] = b0 + b·z ; gradient = b (2,)
        coef, *_ = np.linalg.lstsq(G, A[:, j], rcond=None)
        b = coef[1:]                                 # sensitivity to (zv, zw)
        C += weights[j] * np.outer(b, b)

    evals, evecs = np.linalg.eigh(C)                 # ascending
    evals = evals[::-1]
    evecs = evecs[:, ::-1]
    return ActiveSubspaceResult(
        eigenvalues=evals, eigenvectors=evecs,
        input_mean=mu, input_std=sd, n_modes_used=k,
    )
