# jaxtrace/density/kde.py
"""
Kernel Density Estimation with adaptive bandwidth selection.

Provides Gaussian KDE with Scott/Silverman bandwidth rules, grid evaluation,
and optional JAX acceleration with chunked processing for large datasets.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np

# Import JAX utilities with fallback
from ..utils.jax_utils import JAX_AVAILABLE

if JAX_AVAILABLE:
    try:
        import jax
        import jax.numpy as jnp
        from jax import jit
    except Exception:
        JAX_AVAILABLE = False

if not JAX_AVAILABLE:
    import numpy as jnp  # type: ignore
    # Mock jax.jit for NumPy fallback
    class MockJit:
        def __call__(self, func):
            return func
    jit = MockJit()

# Optional SciPy fallback
try:
    from scipy.stats import gaussian_kde as scipy_gaussian_kde
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

from .kernels import scott_bandwidth, silverman_bandwidth, gaussian_kernel


@dataclass
class KDEEstimator:
    """
    Gaussian KDE with Scott/Silverman bandwidth rules, 2D/3D grids,
    chunked evaluation and optional JAX acceleration.

    Attributes
    ----------
    positions : np.ndarray
        Data points, shape (N, D)
    bandwidth : float, optional
        Fixed bandwidth; auto-determined if None
    bandwidth_rule : str
        Bandwidth selection rule: 'scott' or 'silverman'
    normalize : bool
        Whether to normalize density estimates
    plane : str
        Projection plane for 2D extraction: 'xy', 'xz', or 'yz'
    slab_position : float
        Position of 2D slice through 3D data
    slab_thickness : float
        Thickness of 2D slice
    resolution : int or tuple
        Grid resolution for density evaluation
    bounds : tuple, optional
        Domain bounds ((xmin,xmax), (ymin,ymax), (zmin,zmax))
    """
    positions: np.ndarray         # (N, D) with D in {2,3}
    bandwidth: Optional[float] = None
    bandwidth_rule: str = "scott"  # 'scott' | 'silverman'
    normalize: bool = True
    # Grid / slicing
    plane: str = "xy"
    slab_position: float = 0.0
    slab_thickness: float = 0.1
    resolution: Union[int, Tuple[int, int, int]] = 100
    bounds: Optional[Tuple[Tuple[float, float], ...]] = None  # ((xmin,xmax),(ymin,ymax),(zmin,zmax))

    def __post_init__(self):
        P = np.asarray(self.positions, dtype=np.float32)
        if P.ndim != 2 or P.shape[1] not in (2, 3):
            raise ValueError("positions must be (N,2) or (N,3)")
        self.N, self.D = int(P.shape[0]), int(P.shape[1])
        self.P = P

        # Infer bandwidth if not provided
        if self.bandwidth is None:
            if self.bandwidth_rule == "scott":
                self.h = float(scott_bandwidth(P))
            elif self.bandwidth_rule == "silverman":
                self.h = float(silverman_bandwidth(P))
            else:
                raise ValueError("bandwidth_rule must be 'scott' or 'silverman' if bandwidth is None")
        else:
            self.h = float(self.bandwidth)

    # ---------- Internal helpers ----------

    def _axes_for_plane(self) -> Tuple[int, int, int]:
        plane = self.plane.lower()
        axis_map = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}
        if plane not in axis_map:
            raise ValueError("plane must be one of 'xy','xz','yz'")
        return axis_map[plane]

    def _slice_2d_points(self) -> np.ndarray:
        """Extract 2D slice from 3D points using a slab filter; fallback to projection if empty."""
        if self.D == 2:
            return self.P
        i, j, k = self._axes_for_plane()
        z0 = float(self.slab_position)
        th = float(self.slab_thickness)
        mask = np.abs(self.P[:, k] - z0) <= (th * 0.5)
        pts2 = self.P[mask][:, [i, j]]
        if pts2.size == 0:
            pts2 = self.P[:, [i, j]]  # fallback to pure projection
        return pts2.astype(np.float32, copy=False)

    # ---------- Public pointwise evaluation ----------

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate KDE density at arbitrary points.

        Parameters
        ----------
        points : (M, d)
            Query points. If the estimator was built with 2D data, d must be 2.
            If the estimator was built with 3D data:
              - If d==3, use 3D.
              - If d==2 and plane is set, project the 3D data to 2D and use the 2D evaluator.
              - If d==3 but the estimator is 2D (positions were 2D), we project points to 2D via 'plane'.

        Returns
        -------
        (M,)
            Density values at the query points.
        """
        Q = np.asarray(points, dtype=np.float32)
        if Q.ndim != 2:
            raise ValueError(f"points must be 2D, got {Q.shape}")

        # Choose working dimensionality and data
        if self.D == 2:
            # Estimator trained on 2D; ensure points are 2D
            if Q.shape[1] == 3:
                i, j, _ = self._axes_for_plane()
                grid = Q[:, [i, j]]
            elif Q.shape[1] == 2:
                grid = Q
            else:
                raise ValueError(f"Expected points with 2 or 3 columns, got {Q.shape[1]}")
            pts = self.P  # (N,2)
        else:
            # Estimator trained on 3D; accept either 3D directly or 2D via plane
            if Q.shape[1] == 3:
                grid = Q
                pts = self.P  # (N,3)
            elif Q.shape[1] == 2:
                # Project the 3D data to 2D plane and evaluate in 2D
                pts = self._slice_2d_points()  # (N2,2)
                i, j, _ = self._axes_for_plane()
                grid = Q[:, [0, 1]]  # already (x,y) on the chosen plane
            else:
                raise ValueError(f"Expected points with 2 or 3 columns, got {Q.shape[1]}")

        return self._evaluate_grid(grid, pts)

    # ---------- Grid builders ----------

    def _make_grid_2d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pts = self.P[:, :2] if self.D == 2 else self._slice_2d_points()
        if self.bounds is None:
            lo = pts.min(axis=0)
            hi = pts.max(axis=0)
        else:
            lo = np.array([self.bounds[0][0], self.bounds[1][0]], dtype=float)
            hi = np.array([self.bounds[0][1], self.bounds[1][1]], dtype=float)
        res = self.resolution if isinstance(self.resolution, int) else self.resolution[:2]
        nx = ny = int(res)
        x = np.linspace(lo[0], hi[0], nx, dtype=np.float32)
        y = np.linspace(lo[1], hi[1], ny, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="xy")
        return X, Y, pts

    def _make_grid_3d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.bounds is None:
            lo = self.P.min(axis=0)
            hi = self.P.max(axis=0)
        else:
            lo = np.array([b[0] for b in self.bounds], dtype=float)
            hi = np.array([b[1] for b in self.bounds], dtype=float)
        res = self.resolution if not isinstance(self.resolution, int) else (self.resolution,)*3
        nx, ny, nz = map(int, res)
        x = np.linspace(lo[0], hi[0], nx, dtype=np.float32)
        y = np.linspace(lo[1], hi[1], ny, dtype=np.float32)
        z = np.linspace(lo[2], hi[2], nz, dtype=np.float32)
        X, Y, Z = np.meshgrid(x, y, z, indexing="xy")
        return X, Y, Z, self.P

    # ---------- Grid evaluation APIs ----------

    def evaluate_2d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, Y, pts = self._make_grid_2d()
        grid = np.stack([X.ravel(), Y.ravel()], axis=1)
        dens = self._evaluate_grid(grid, pts)
        Z = dens.reshape(X.shape)

        if self.normalize:
            dx = (X.max() - X.min()) / max(X.shape[1] - 1, 1)
            dy = (Y.max() - Y.min()) / max(Y.shape[0] - 1, 1)
            cell_area = float(dx * dy)
            total = float((Z * cell_area).sum())
            if total > 0:
                Z = Z * (self.N / total)
            domain_area = float((X.max() - X.min()) * (Y.max() - Y.min()))
            avg = self.N / max(domain_area, 1e-12)
            Z = Z / max(avg, 1e-12)
        return X, Y, Z

    def evaluate_3d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X, Y, Z, pts = self._make_grid_3d()
        grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        dens = self._evaluate_grid(grid, pts)
        D = dens.reshape(X.shape)

        if self.normalize:
            dx = (X.max() - X.min()) / max(X.shape[1] - 1, 1)
            dy = (Y.max() - Y.min()) / max(Y.shape[0] - 1, 1)
            dz = (Z.max() - Z.min()) / max(Z.shape[2] - 1, 1)
            cell_vol = float(dx * dy * dz)
            total = float((D * cell_vol).sum())
            if total > 0:
                D = D * (self.N / total)
            domain_vol = float((X.max() - X.min()) * (Y.max() - Y.min()) * (Z.max() - Z.min()))
            avg = self.N / max(domain_vol, 1e-12)
            D = D / max(avg, 1e-12)
        return X, Y, Z, D

    # ---------- Core engine ----------

    def _evaluate_grid(self, grid_xy_or_xyz: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Evaluate KDE at grid points using JAX, SciPy, or NumPy backend."""
        h = float(self.h)
        d = int(grid_xy_or_xyz.shape[1])

        if JAX_AVAILABLE:
            P = jnp.asarray(pts, dtype=jnp.float32)          # (N,d)
            Q = jnp.asarray(grid_xy_or_xyz, dtype=jnp.float32)  # (M,d)
            inv_h2 = jnp.asarray(1.0 / (h * h), dtype=jnp.float32)

            @jit
            def eval_chunk(qchunk):
                # qchunk: (M,d)
                diffs = qchunk[:, None, :] - P[None, :, :]       # (M,N,d)
                r2 = jnp.sum(diffs * diffs, axis=-1) * inv_h2    # (M,N)
                k = gaussian_kernel(r2, d)                        # normalized
                dens = jnp.mean(k, axis=1)
                return dens

            # Chunk to avoid OOM
            M = int(Q.shape[0])
            chunk = 100_000 if d == 2 else 50_000
            out = []
            for s in range(0, M, chunk):
                e = min(s + chunk, M)
                out.append(np.asarray(eval_chunk(Q[s:e])))
            return np.concatenate(out, axis=0).astype(np.float32, copy=False)

        # SciPy fallback
        if SCIPY_AVAILABLE:
            kde = scipy_gaussian_kde(pts.T, bw_method=h)
            return kde(grid_xy_or_xyz.T).astype(np.float32, copy=False)

        # Pure NumPy fallback
        diffs = grid_xy_or_xyz[:, None, :] - pts[None, :, :]
        r2 = np.sum(diffs * diffs, axis=-1) / (h * h)
        k = gaussian_kernel(r2, d)
        dens = np.mean(k, axis=1)
        return dens.astype(np.float32, copy=False)