"""
Discrete curvature diagnostics for the ROM case manifold.

Complements the intrinsic-dimension / MDS analysis in ``manifold.py`` by
measuring *how curved* the parameter-to-snapshot map is, using discrete
secant-vector geometry directly in the high-dimensional snapshot space
(NOT the low-D MDS embedding — the embedding's curvature is not the
manifold's). See docs/manifold_curvature.md for the literature.

Three estimators, in increasing robustness / decreasing interpretability:

1. ``turning_angle_curvature`` — for points ordered along a 1D path
   (e.g. fix omega, sweep v_adv): the angle between consecutive secant
   vectors, theta_i = angle(p_{i+1}-p_i, p_i-p_{i-1}). The discrete
   curve-curvature; 0 = locally straight/linear.
2. ``menger_curvature`` — for a triple of points, kappa = 4*Area /
   (|a||b||c|) = 1/R of the circumscribed circle. The exact
   cross-product formalisation of "angle between secants"; computed
   along the same ordered 1D paths (consecutive triples).
3. ``local_pca_curvature`` — for each point, k nearest neighbours, fit a
   local tangent via PCA; curvature = trailing-eigenvalue energy beyond
   the intrinsic dimension. Most robust for *unordered* clouds, but
   FRAGILE at n=20 (needs a local neighbourhood) — treat as a relative,
   comparative signal only.

At n~20 all of these are *relative/comparative* diagnostics (is region A
more curved than B? is the particle manifold more curved than density?),
not absolute geometric quantities. High-D secants are computed pairwise
so the ambient dimension (~10^6) never needs materialising per triple.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Parameter-path ordering (natural 1D paths through the 2D input grid)
# ---------------------------------------------------------------------------

def parameter_paths(
    params: np.ndarray,
    tol_rel: float = 0.02,
) -> Dict[str, List[np.ndarray]]:
    """
    Group case indices into 1D paths along each parameter axis.

    Returns {"vary_v": [...paths...], "vary_w": [...]} where each path is
    an array of case indices with one parameter ~constant and the other
    increasing. A path needs >=3 points to admit a curvature.

    ``tol_rel`` is the relative tolerance (of each axis' range) for
    treating a parameter as "constant" within a path. Because the sampling
    is Latin-hypercube-like (few exact ties), we also emit the *global*
    monotone orderings (all points sorted by one axis) as fallback paths.
    """
    v = params[:, 0]
    w = params[:, 1]
    out: Dict[str, List[np.ndarray]] = {"vary_v": [], "vary_w": []}

    # exact/near ties on the held-fixed axis
    def _grouped(fix, move, key):
        rng = np.ptp(fix) if np.ptp(fix) > 0 else 1.0
        tol = tol_rel * rng
        seen = np.zeros(len(fix), bool)
        for i in range(len(fix)):
            if seen[i]:
                continue
            grp = np.where(np.abs(fix - fix[i]) <= tol)[0]
            seen[grp] = True
            if grp.size >= 3:
                out[key].append(grp[np.argsort(move[grp])])

    _grouped(w, v, "vary_v")   # fix omega, sweep v_adv
    _grouped(v, w, "vary_w")   # fix v_adv, sweep omega

    # Fallback global monotone paths (always >=3 pts): the whole set
    # sorted by one axis. Useful when there are no exact ties.
    out["vary_v"].append(np.argsort(v))
    out["vary_w"].append(np.argsort(w))
    return out


# ---------------------------------------------------------------------------
# 1. Turning-angle curvature
# ---------------------------------------------------------------------------

def _secant(X: np.ndarray, i: int, j: int) -> np.ndarray:
    return X[j] - X[i]


def turning_angle_curvature(
    X: np.ndarray, path: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    *Turning* angle (radians) at each interior point of an ordered
    ``path`` — the deviation of the direction of travel, so **0 = the path
    goes perfectly straight** and pi = it reverses.

      s_in  = p_k - p_{k-1},  s_out = p_{k+1} - p_k
      turn_k = angle *between the two secant directions*
             = arccos( <s_in, s_out> / (|s_in||s_out|) )

    (This is the exterior/turning angle: two co-directional secants give
    0; orthogonal give pi/2; anti-parallel give pi. In high ambient
    dimension, unrelated secants tend toward pi/2, so read values
    comparatively, not against a Euclidean-plane intuition.)

    Returns (interior_case_indices, turns).
    """
    idx = np.asarray(path, int)
    turns = []
    inner = []
    for k in range(1, len(idx) - 1):
        s_in = X[idx[k]] - X[idx[k - 1]]
        s_out = X[idx[k + 1]] - X[idx[k]]
        ni = np.linalg.norm(s_in)
        no = np.linalg.norm(s_out)
        if ni <= 0 or no <= 0:
            continue
        c = float(np.dot(s_in, s_out) / (ni * no))
        turns.append(np.arccos(np.clip(c, -1.0, 1.0)))
        inner.append(idx[k])
    return np.asarray(inner, int), np.asarray(turns, float)


# ---------------------------------------------------------------------------
# 2. Menger curvature
# ---------------------------------------------------------------------------

def menger_curvature_triple(
    X: np.ndarray, i: int, j: int, k: int
) -> float:
    """
    Menger curvature of the triple (i, j, k): kappa = 4*Area / (a*b*c),
    the reciprocal of the circumradius. 0 => collinear (locally flat).

    Area is from the two edge vectors at j via the (high-D) parallelogram
    identity Area = 0.5*sqrt(|u|^2|v|^2 - <u,v>^2) — no cross product
    needed, so it works in any ambient dimension.
    """
    u = X[i] - X[j]
    v = X[k] - X[j]
    a = np.linalg.norm(X[i] - X[j])
    b = np.linalg.norm(X[k] - X[j])
    c = np.linalg.norm(X[i] - X[k])
    if a <= 0 or b <= 0 or c <= 0:
        return 0.0
    gram = (np.dot(u, u) * np.dot(v, v)) - np.dot(u, v) ** 2
    area = 0.5 * np.sqrt(max(gram, 0.0))
    return float(4.0 * area / (a * b * c))


def menger_curvature_path(
    X: np.ndarray, path: np.ndarray, scale: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Menger curvature for each consecutive triple along ``path``.

    Raw Menger kappa = 1/circumradius has units of 1/length, so it is not
    comparable across manifolds with different snapshot magnitudes (e.g.
    density ~1e6 vs particle displacement ~1e-2). Passing ``scale`` (a
    characteristic length, e.g. the median secant length of the manifold)
    returns the **dimensionless** kappa*scale — "bend per step" — which IS
    comparable across manifolds. ``scale=None`` returns the raw kappa.

    Returns (interior_case_indices, kappas)."""
    idx = np.asarray(path, int)
    s = scale if (scale is not None and scale > 0) else 1.0
    inner, kappas = [], []
    for k in range(1, len(idx) - 1):
        kappas.append(
            menger_curvature_triple(X, idx[k - 1], idx[k], idx[k + 1]) * s)
        inner.append(idx[k])
    return np.asarray(inner, int), np.asarray(kappas, float)


# ---------------------------------------------------------------------------
# 3. Local PCA curvature (fragile at n~20 — comparative only)
# ---------------------------------------------------------------------------

def local_pca_curvature(
    X: np.ndarray,
    k: int = 5,
    intrinsic_dim: int = 2,
    distances: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Per-point local curvature = fraction of local variance NOT captured by
    the leading ``intrinsic_dim`` principal directions of the k-NN
    neighbourhood:  sigma_i = sum_{>d} lambda / sum lambda.

    0 => neighbourhood lies in a flat d-plane (locally linear); larger =>
    more curved. WARNING: at n~20 with k~5 this is a noisy, relative
    signal — compare across manifolds/regions, do not read absolutely.
    """
    n = X.shape[0]
    if distances is None:
        from .manifold import pairwise_distances
        distances = pairwise_distances(X)
    D = distances.copy()
    np.fill_diagonal(D, np.inf)
    sig = np.zeros(n)
    kk = min(k, n - 1)
    for i in range(n):
        nbr = np.argsort(D[i])[:kk]
        pts = X[np.concatenate([[i], nbr])]          # (kk+1, m)
        c = pts - pts.mean(axis=0, keepdims=True)
        # covariance eigenvalues via the small (kk+1)x(kk+1) Gram matrix
        g = c @ c.T
        ev = np.linalg.eigvalsh(g)
        ev = np.clip(ev[::-1], 0, None)
        tot = ev.sum()
        if tot <= 0:
            continue
        d = min(intrinsic_dim, len(ev) - 1)
        sig[i] = float(ev[d:].sum() / tot)
    return sig


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

@dataclass
class CurvatureReport:
    case_numbers: List[str]
    params: np.ndarray
    # turning-angle: per interior case, aggregated over all paths it sits in
    turning_angle_deg: Dict[str, float]        # case -> mean turning angle [deg]
    turning_angle_by_axis: Dict[str, Dict[str, float]]  # axis -> {case: deg}
    # menger: per interior case
    menger: Dict[str, float]                    # case -> mean menger kappa
    menger_by_axis: Dict[str, Dict[str, float]]
    # local PCA
    local_pca: Dict[str, float]                 # case -> sigma_i
    local_pca_k: int
    local_pca_dim: int
    # scalar summaries
    summary: Dict[str, float] = field(default_factory=dict)


def analyze_curvature(
    X: np.ndarray,
    params: np.ndarray,
    case_numbers: List[str],
    local_pca_k: int = 5,
    intrinsic_dim: int = 2,
    clip_percentile: Optional[float] = 99.9,
) -> CurvatureReport:
    """
    Full discrete-curvature diagnostic on the high-dimensional snapshot
    matrix ``X`` (n_cases x n_features).

    ``intrinsic_dim`` should match the manifold's estimated intrinsic
    dimension (~2 for density, ~3 for particles) for the local-PCA
    estimator. ``clip_percentile`` tames residual outlier features (as in
    ``manifold.py``); pass None to disable.
    """
    from .manifold import clip_outlier_features, pairwise_distances

    Xa = clip_outlier_features(X, clip_percentile) if clip_percentile else X
    Xa = np.asarray(Xa, float)
    cn = list(case_numbers)

    paths = parameter_paths(params)

    # Characteristic length: median pairwise distance. Menger kappa is
    # multiplied by this so it becomes dimensionless ("bend per step") and
    # comparable across manifolds of different magnitude.
    Dfull = pairwise_distances(Xa)
    scale = float(np.median(Dfull[np.triu_indices_from(Dfull, 1)]))

    # --- turning angle + menger, aggregated per case over its paths ---
    ta_axis: Dict[str, Dict[str, List[float]]] = {"vary_v": {}, "vary_w": {}}
    mg_axis: Dict[str, Dict[str, List[float]]] = {"vary_v": {}, "vary_w": {}}
    for axis, plist in paths.items():
        for p in plist:
            ii, th = turning_angle_curvature(Xa, p)
            for c, t in zip(ii, np.degrees(th)):
                ta_axis[axis].setdefault(cn[c], []).append(float(t))
            im, km = menger_curvature_path(Xa, p, scale=scale)
            for c, kv in zip(im, km):
                mg_axis[axis].setdefault(cn[c], []).append(float(kv))

    def _mean_over_axes(byaxis):
        agg: Dict[str, List[float]] = {}
        for axis in byaxis:
            for c, vals in byaxis[axis].items():
                agg.setdefault(c, []).extend(vals)
        return {c: float(np.mean(v)) for c, v in agg.items()}

    ta_axis_mean = {a: {c: float(np.mean(v)) for c, v in d.items()}
                    for a, d in ta_axis.items()}
    mg_axis_mean = {a: {c: float(np.mean(v)) for c, v in d.items()}
                    for a, d in mg_axis.items()}
    turning = _mean_over_axes(ta_axis)
    menger = _mean_over_axes(mg_axis)

    # --- local PCA curvature ---
    lpca = local_pca_curvature(Xa, k=local_pca_k,
                               intrinsic_dim=intrinsic_dim, distances=Dfull)
    local_pca = {cn[i]: float(lpca[i]) for i in range(len(cn))}

    summary = {
        "turning_angle_mean_deg": float(np.mean(list(turning.values())))
        if turning else float("nan"),
        "turning_angle_max_deg": float(np.max(list(turning.values())))
        if turning else float("nan"),
        "menger_mean": float(np.mean(list(menger.values())))
        if menger else float("nan"),
        "menger_max": float(np.max(list(menger.values())))
        if menger else float("nan"),
        "local_pca_mean": float(np.mean(lpca)),
        "local_pca_max": float(np.max(lpca)),
        "length_scale": scale,
    }

    return CurvatureReport(
        case_numbers=cn, params=np.asarray(params),
        turning_angle_deg=turning, turning_angle_by_axis=ta_axis_mean,
        menger=menger, menger_by_axis=mg_axis_mean,
        local_pca=local_pca, local_pca_k=local_pca_k,
        local_pca_dim=intrinsic_dim, summary=summary,
    )
