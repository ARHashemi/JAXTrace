# jaxtrace/density/sph.py
"""
Smoothed Particle Hydrodynamics (SPH) density estimation.

Provides SPH density computation with various kernels, adaptive smoothing,
and efficient neighbor search using hash grids or brute force methods.
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
        from jax import jit, vmap
    except Exception:
        JAX_AVAILABLE = False

if not JAX_AVAILABLE:
    import numpy as jnp  # type: ignore
    # Mock jax functions for NumPy fallback
    class MockJit:
        def __call__(self, func):
            return func
    class MockVmap:
        def __call__(self, func):
            def vectorized(x):
                return np.array([func(xi) for xi in x])
            return vectorized
    jit = MockJit()
    vmap = MockVmap()

from .neighbors import HashGridNeighbors


@dataclass
class SPHDensityEstimator:
    """
    SPH density estimator with cubic-spline or Wendland kernels.

    - dimensions: '2d' or '3d' (for 2D, a slab slice is taken from (N,3) positions)
    - adaptive: if True, estimate per-particle h from k-NN distances (with safeguards)
    - neighbor_search: 'hashgrid' (fast, JAX) or 'bruteforce'
    - normalize: normalize density to per-area (2D) or per-volume (3D) average
    
    Attributes
    ----------
    positions : np.ndarray
        Particle positions, shape (N, 3)
    dimensions : str
        Spatial dimensionality: '2d' or '3d'
    plane : str
        Projection plane for 2D: 'xy', 'xz', or 'yz'
    position : float
        Position along normal axis for 2D slice
    slab_thickness : float
        Thickness of 2D slice
    smoothing_length : float, optional
        Fixed smoothing length; auto-determined if None
    kernel_type : str
        Kernel function: 'cubic_spline', 'wendland', or 'gaussian'
    adaptive : bool
        Whether to use adaptive smoothing lengths
    n_neighbors : int
        Number of neighbors for adaptive smoothing
    masses : np.ndarray, optional
        Particle masses; uniform if None
    max_particles_for_adaptive : int
        Maximum particles for adaptive smoothing
    neighbor_search : str
        Search method: 'hashgrid' or 'bruteforce'
    per_cell_cap : int
        Maximum particles per hash grid cell
    max_neighbors : int
        Maximum neighbors to consider
    normalize : bool
        Whether to normalize density estimates
    resolution : int or tuple
        Grid resolution for evaluation
    bounds : tuple, optional
        Domain bounds for evaluation
    """
    positions: np.ndarray
    dimensions: str = "2d"
    plane: str = "xy"
    position: float = 0.0
    slab_thickness: float = 0.1
    smoothing_length: Optional[float] = None
    kernel_type: str = "cubic_spline"  # 'cubic_spline'|'wendland'|'gaussian'
    adaptive: bool = False
    n_neighbors: int = 32
    masses: Optional[np.ndarray] = None
    max_particles_for_adaptive: int = 5000
    neighbor_search: str = "hashgrid"  # 'hashgrid' or 'bruteforce'
    per_cell_cap: int = 64
    max_neighbors: int = 512
    normalize: bool = True
    resolution: Union[int, Tuple[int, int, int]] = 100
    bounds: Optional[Tuple[Tuple[float, float], ...]] = None

    def __post_init__(self):
        if not JAX_AVAILABLE:
            print("Warning: JAX not available, SPH will use NumPy fallback with reduced performance")

        P3 = np.asarray(self.positions)
        if self.dimensions.lower() == "2d":
            if P3.shape[1] != 3:
                raise ValueError("2D SPH requires (N,3) positions for slicing")
            self.P = self._extract_2d(P3)
            self.D = 2
        else:
            if P3.shape[1] != 3:
                raise ValueError("3D SPH requires (N,3) positions")
            self.P = P3
            self.D = 3

        self.N = int(self.P.shape[0])
        if self.N < 2:
            raise ValueError("At least two particles are required")

        if self.masses is None:
            self.masses = np.ones((self.N,), dtype=np.float32)
        else:
            self.masses = np.asarray(self.masses, dtype=np.float32)
            if self.masses.shape[0] != self.N:
                raise ValueError("masses must be length N")

        self.positions_jax = jnp.asarray(self.P, dtype=jnp.float32)
        self.masses_jax = jnp.asarray(self.masses, dtype=jnp.float32)

        if self.smoothing_length is not None:
            self.smoothing_lengths = jnp.full(self.N, float(self.smoothing_length), dtype=jnp.float32)
        else:
            self.smoothing_lengths = self._calculate_smoothing_lengths()

        self._mean_h = float(jnp.mean(self.smoothing_lengths))

        # Build neighbor backend if requested
        self._neighbor_backend = "bruteforce"
        if self.neighbor_search == "hashgrid" and JAX_AVAILABLE:
            try:
                self._hash = HashGridNeighbors(
                    positions=self.positions_jax,
                    cell_size=max(self._mean_h, 1e-6),
                    per_cell_cap=int(self.per_cell_cap),
                )
                self._neighbor_backend = "hashgrid"
            except Exception as e:
                print(f"Warning: HashGrid initialization failed ({e}), using brute force")

        if JAX_AVAILABLE:
            self._evaluate_jit = jit(self._evaluate_density)
        else:
            self._evaluate_jit = self._evaluate_density

    # ---------- Slicing helpers ----------

    def _extract_2d(self, P3: np.ndarray) -> np.ndarray:
        """Extract 2D slice from 3D particle positions."""
        plane = self.plane.lower()
        axis_map = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}
        if plane not in axis_map:
            raise ValueError("plane must be one of 'xy','xz','yz'")
        i, j, k = axis_map[plane]
        z0 = float(self.position)
        th = float(self.slab_thickness)
        mask = np.abs(P3[:, k] - z0) <= (th * 0.5)
        pts2 = P3[mask][:, [i, j]]
        if pts2.size == 0:
            pts2 = P3[:, [i, j]]
        return pts2.astype(np.float32, copy=False)

    # ---------- Adaptive smoothing ----------

    def _calculate_smoothing_lengths(self) -> jnp.ndarray:
        """Calculate smoothing lengths using adaptive or fixed approach."""
        if self.adaptive and self.N <= int(self.max_particles_for_adaptive):
            try:
                return self._adaptive_smoothing_lengths()
            except Exception as e:
                print(f"Warning: Adaptive smoothing failed ({e}). Using fixed smoothing.")
                return self._fixed_smoothing_length()
        return self._fixed_smoothing_length()

    def refresh_smoothing_lengths(self) -> None:
        """Recalculate smoothing lengths and update neighbor search structures."""
        self.smoothing_lengths = self._calculate_smoothing_lengths()
        self._mean_h = float(jnp.mean(self.smoothing_lengths))
        if self._neighbor_backend == "hashgrid" and JAX_AVAILABLE:
            self._hash = HashGridNeighbors(
                positions=self.positions_jax,
                cell_size=max(self._mean_h, 1e-6),
                per_cell_cap=int(self.per_cell_cap),
            )

    def _fixed_smoothing_length(self) -> jnp.ndarray:
        """Calculate fixed smoothing length based on particle density."""
        n_per_dim = self.N ** (1.0 / self.D)
        pos_min = jnp.min(self.positions_jax, axis=0)
        pos_max = jnp.max(self.positions_jax, axis=0)
        domain_size = jnp.mean(pos_max - pos_min)
        h = domain_size / n_per_dim * 2.0
        h = jnp.clip(h, 0.01, domain_size * 0.1)
        return jnp.full(self.N, h, dtype=jnp.float32)

    def _adaptive_smoothing_lengths(self) -> jnp.ndarray:
        """Calculate adaptive smoothing lengths based on k-NN distances."""
        k = int(min(self.n_neighbors, max(self.N - 1, 1)))
        idxs = jnp.arange(self.N)

        def kth_dist(i):
            pi = self.positions_jax[i]
            d = jnp.linalg.norm(self.positions_jax - pi[None, :], axis=1)
            nn = jnp.argpartition(d, k)
            return 1.2 * d[nn[k]]  # small safety factor

        return vmap(kth_dist)(idxs).astype(jnp.float32)

    # ---------- Kernels ----------

    def _kernel_cubic(self, r: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        """Cubic spline kernel."""
        q = r / jnp.maximum(h, 1e-12)
        if self.D == 2:
            sigma = 10.0 / (7.0 * jnp.pi * jnp.maximum(h, 1e-12) ** 2)
        else:
            sigma = 1.0 / (jnp.pi * jnp.maximum(h, 1e-12) ** 3)
        w1 = sigma * (1.0 - 1.5 * q**2 + 0.75 * q**3)
        w2 = sigma * 0.25 * (2.0 - q) ** 3
        return jnp.where(q <= 1.0, w1, jnp.where(q <= 2.0, w2, 0.0))

    def _kernel_wendland(self, r: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        """Wendland C2 kernel."""
        q = r / jnp.maximum(h, 1e-12)
        if self.D == 2:
            sigma = 7.0 / (jnp.pi * jnp.maximum(h, 1e-12) ** 2)
        else:
            sigma = 21.0 / (2.0 * jnp.pi * jnp.maximum(h, 1e-12) ** 3)
        t = jnp.maximum(1.0 - q, 0.0)
        return sigma * (t**4) * (1.0 + 4.0 * q)

    def _kernel_gaussian(self, r: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        """Gaussian kernel."""
        q2 = (r / jnp.maximum(h, 1e-12)) ** 2
        if self.D == 2:
            sigma = 1.0 / (jnp.pi * jnp.maximum(h, 1e-12) ** 2)
        else:
            sigma = 1.0 / ((jnp.pi * jnp.maximum(h, 1e-12) ** 2) ** 1.5)
        return sigma * jnp.exp(-q2)

    def _kernel(self, r: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        """Dispatch to selected kernel function."""
        if self.kernel_type == "cubic_spline":
            return self._kernel_cubic(r, h)
        if self.kernel_type == "wendland":
            return self._kernel_wendland(r, h)
        if self.kernel_type == "gaussian":
            return self._kernel_gaussian(r, h)
        raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    # ---------- Evaluation ----------

    def _evaluate_density(self, eval_points: jnp.ndarray) -> jnp.ndarray:
        """Evaluate SPH density at given points."""
        Q = eval_points
        if self._neighbor_backend == "hashgrid":
            radius = 2.0 * self._mean_h
            idxs, dists = self._hash.query_many(Q, radius=radius, max_neighbors=int(self.max_neighbors))
            valid = (idxs >= 0)
            masses = jnp.where(valid, self.masses_jax[idxs], 0.0)
            hi = jnp.where(valid, self.smoothing_lengths[idxs], self._mean_h)
            w = self._kernel(dists, hi)
            return jnp.sum(masses * w, axis=1)

        # Fallback: bruteforce cutoff
        def single(point):
            d = jnp.linalg.norm(self.positions_jax - point[None, :], axis=1)
            avg_h = jnp.mean(self.smoothing_lengths)
            cutoff = 3.0 * avg_h
            mask = d <= cutoff
            di = d[mask]
            mi = self.masses_jax[mask]
            hi = self.smoothing_lengths[mask]
            return jnp.sum(mi * self._kernel(di, hi))

        return vmap(single)(Q)

    def evaluate(self, eval_points: np.ndarray) -> np.ndarray:
        """
        Evaluate SPH density at given evaluation points.
        
        Parameters
        ----------
        eval_points : np.ndarray
            Points to evaluate density at, shape (M, D)
            
        Returns
        -------
        np.ndarray
            Density values at evaluation points, shape (M,)
        """
        Q = jnp.asarray(eval_points, dtype=jnp.float32)
        chunk = 50_000 if self.D == 2 else 25_000
        outs = []
        for s in range(0, Q.shape[0], chunk):
            e = min(s + chunk, Q.shape[0])
            outs.append(self._evaluate_jit(Q[s:e]))
        rho = jnp.concatenate(outs, axis=0)

        if self.normalize:
            if self.D == 2:
                xmin, xmax = float(jnp.min(Q[:, 0])), float(jnp.max(Q[:, 0]))
                ymin, ymax = float(jnp.min(Q[:, 1])), float(jnp.max(Q[:, 1]))
                area = max((xmax - xmin) * (ymax - ymin), 1e-12)
                avg = self.N / area
                rho = rho / avg
            else:
                xmin, xmax = float(jnp.min(Q[:, 0])), float(jnp.max(Q[:, 0]))
                ymin, ymax = float(jnp.min(Q[:, 1])), float(jnp.max(Q[:, 1]))
                zmin, zmax = float(jnp.min(Q[:, 2])), float(jnp.max(Q[:, 2]))
                vol = max((xmax - xmin) * (ymax - ymin) * (zmax - zmin), 1e-12)
                avg = self.N / vol
                rho = rho / avg

        return np.asarray(rho)