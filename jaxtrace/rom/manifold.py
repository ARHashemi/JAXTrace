"""
Manifold-insight diagnostics for the ROM "case manifold".

Each FOM case is one high-dimensional snapshot (e.g. a 360k*3 particle
cloud, or a flattened density field). The 20 cases trace out a manifold
in that space, parametrised by (v_adv, omega_pin). Before reaching for a
nonlinear embedding (t-SNE / UMAP) — which can manufacture clusters and
curvature that are not real — these *linear and metric* diagnostics
characterise the manifold's dimension, linearity, and organisation with
methods that are cheap and trustworthy:

  * pca_spectrum        linear dimensionality (explained-variance curve)
  * pairwise_distances  the 20x20 case-to-case distance matrix
  * classical_mds       distance-preserving 2D layout (linear, faithful)
  * intrinsic_dimension TwoNN + correlation-dimension estimates (a number)
  * linearity_residual  reconstruction error vs #PCs (curvature signal)
  * parameter_alignment how well each PC correlates with (v_adv, omega)

All depend only on numpy / scipy. UMAP/t-SNE are deliberately left out
(optional, separate) so the trustworthy diagnostics come first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Robust standardisation (cases 000/001 have runaway particles)
# ---------------------------------------------------------------------------

def clip_outlier_features(
    X: np.ndarray, percentile: float = 99.9
) -> np.ndarray:
    """
    Clip extreme per-feature values to a symmetric percentile band. This
    tames runaway-particle coordinates that would otherwise dominate
    Euclidean distances, without dropping any case.
    """
    lo = np.percentile(X, 100 - percentile, axis=0)
    hi = np.percentile(X, percentile, axis=0)
    return np.clip(X, lo, hi)


# ---------------------------------------------------------------------------
# 1. Linear spectrum
# ---------------------------------------------------------------------------

@dataclass
class SpectrumResult:
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative: np.ndarray

    def n_components_for(self, frac: float) -> int:
        return int(np.searchsorted(self.cumulative, frac) + 1)


def pca_spectrum(X: np.ndarray) -> SpectrumResult:
    """Mean-centred SVD spectrum of the case matrix (n_cases x n_features)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # economy SVD via the small Gram matrix (n x n).
    G = Xc @ Xc.T
    w = np.linalg.eigvalsh(G)[::-1]
    w = np.clip(w, 0, None)
    s = np.sqrt(w)
    evr = w / w.sum() if w.sum() > 0 else w
    return SpectrumResult(s, evr, np.cumsum(evr))


# ---------------------------------------------------------------------------
# 2. Pairwise distances + classical MDS
# ---------------------------------------------------------------------------

def pairwise_distances(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Full (n x n) distance matrix between cases."""
    if metric == "euclidean":
        sq = np.sum(X * X, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        return np.sqrt(np.maximum(d2, 0.0))
    if metric == "cosine":
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        norm = np.where(norm > 0, norm, 1.0)
        Xn = X / norm
        return 1.0 - Xn @ Xn.T
    raise ValueError(f"unknown metric {metric!r}")


def classical_mds(D: np.ndarray, n_components: int = 2):
    """
    Classical (Torgerson) MDS: a *linear*, distance-preserving embedding.
    Returns (coords (n, k), eigenvalues). Negative trailing eigenvalues
    quantify how non-Euclidean the distances are (manifold curvature).
    """
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order]
    k = n_components
    pos = np.clip(w[:k], 0, None)
    coords = V[:, :k] * np.sqrt(pos)[None, :]
    return coords, w


# ---------------------------------------------------------------------------
# 3. Intrinsic dimension
# ---------------------------------------------------------------------------

def intrinsic_dimension_twonn(X: np.ndarray) -> float:
    """
    TwoNN intrinsic-dimension estimator (Facco et al. 2017). Uses the
    ratio of 2nd- to 1st-nearest-neighbour distances; robust on small
    samples and does not assume linearity.
    """
    D = pairwise_distances(X)
    np.fill_diagonal(D, np.inf)
    # two nearest neighbours per point
    part = np.sort(D, axis=1)
    r1 = part[:, 0]
    r2 = part[:, 1]
    good = r1 > 0
    mu = (r2[good] / r1[good])
    mu = mu[np.isfinite(mu) & (mu > 1)]
    if mu.size < 2:
        return float("nan")
    # ML estimate: d = N / sum(log mu)
    return float(mu.size / np.sum(np.log(mu)))


def correlation_dimension(X: np.ndarray) -> float:
    """
    Grassberger-Procaccia correlation dimension: slope of log C(r) vs
    log r over the central range of pairwise distances. Another
    nonlinear-aware intrinsic-dimension estimate, as a cross-check.
    """
    D = pairwise_distances(X)
    iu = np.triu_indices_from(D, 1)
    d = np.sort(D[iu])
    d = d[d > 0]
    if d.size < 5:
        return float("nan")
    radii = np.geomspace(d[max(1, d.size // 10)], d[-max(1, d.size // 10)], 20)
    C = np.array([(d < r).mean() for r in radii])
    m = (C > 0) & (C < 1)
    if m.sum() < 3:
        return float("nan")
    slope = np.polyfit(np.log(radii[m]), np.log(C[m]), 1)[0]
    return float(slope)


# ---------------------------------------------------------------------------
# 4. Linearity / parameter alignment
# ---------------------------------------------------------------------------

def linearity_residual(X: np.ndarray) -> np.ndarray:
    """
    Relative reconstruction error of the case matrix using the first k
    principal components, for k = 1..n-1. A fast drop to ~0 means the
    manifold is essentially linear and low-dimensional.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    G = Xc @ Xc.T
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    w = np.clip(w[order], 0, None)
    total = w.sum()
    # residual energy after k components = sum_{>k} w / total
    cum = np.cumsum(w)
    resid = 1.0 - cum / total if total > 0 else np.zeros_like(w)
    return np.sqrt(np.clip(resid, 0, None))  # relative L2 residual


def parameter_alignment(
    coords: np.ndarray, params: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Correlation of each embedding axis with v_adv and omega_pin. High
    |corr| means that axis is essentially a parameter coordinate, i.e.
    the manifold is smoothly parametrised by the inputs.
    """
    out = {}
    names = ["v_adv", "omega_pin"]
    for j in range(coords.shape[1]):
        cs = []
        for p in range(params.shape[1]):
            c = np.corrcoef(coords[:, j], params[:, p])[0, 1]
            cs.append(c)
        out[f"axis{j+1}"] = np.array(cs)
    out["param_names"] = np.array(names)
    return out


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

@dataclass
class ManifoldReport:
    spectrum: SpectrumResult
    distances: np.ndarray
    mds_coords: np.ndarray
    mds_eigenvalues: np.ndarray
    id_twonn: float
    id_correlation: float
    residual: np.ndarray
    alignment: Dict[str, np.ndarray]
    case_numbers: List[str]
    params: np.ndarray


def analyze_manifold(
    X: np.ndarray,
    params: np.ndarray,
    case_numbers: List[str],
    clip_percentile: Optional[float] = 99.9,
) -> ManifoldReport:
    """Run the full linear/metric manifold diagnostic suite."""
    Xa = clip_outlier_features(X, clip_percentile) if clip_percentile else X
    spec = pca_spectrum(Xa)
    D = pairwise_distances(Xa)
    coords, eigs = classical_mds(D, n_components=2)
    return ManifoldReport(
        spectrum=spec,
        distances=D,
        mds_coords=coords,
        mds_eigenvalues=eigs,
        id_twonn=intrinsic_dimension_twonn(Xa),
        id_correlation=correlation_dimension(Xa),
        residual=linearity_residual(Xa),
        alignment=parameter_alignment(coords, params),
        case_numbers=list(case_numbers),
        params=np.asarray(params),
    )
