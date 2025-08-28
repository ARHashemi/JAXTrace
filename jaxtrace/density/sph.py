# density/sph.py (replace or extend your SPHDensityEstimator accordingly)

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False

from .neighbors import HashGridNeighbors
try:
    from .neighbors import JaxHashGridNeighbors
except Exception:
    JaxHashGridNeighbors = None


@dataclass
class SPHDensityEstimator:
    # ... keep your current signature and defaults ...
    positions: np.ndarray
    dimensions: str = "2d"
    plane: str = "xy"
    position: float = 0.0
    slab_thickness: float = 0.1
    smoothing_length: Optional[float] = None
    kernel_type: str = "cubic_spline"
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
        # 1) Positions and dimensionality
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

        self.N = self.P.shape[0]
        if self.N < 2:
            raise ValueError("At least two particles are required")

        # 2) Masses
        if self.masses is None:
            self.masses = np.ones((self.N,), dtype=np.float32)
        else:
            self.masses = np.asarray(self.masses, dtype=np.float32)
            if self.masses.shape[0] != self.N:
                raise ValueError("masses must be length N")

        # 3) JAX arrays
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for SPHDensityEstimator")
        self.positions_jax = jnp.asarray(self.P)
        self.masses_jax = jnp.asarray(self.masses)

        # 4) Smoothing lengths (cached)
        if self.smoothing_length is not None:
            self.smoothing_lengths = jnp.full(self.N, float(self.smoothing_length))
        else:
            self.smoothing_lengths = self._calculate_smoothing_lengths()
        # cache for refresh bookkeeping
        self._h_cache_version = 0

        # 5) Build neighbor search
        self._mean_h = float(jnp.mean(self.smoothing_lengths))
        self._neighbor_backend = "bruteforce"
        if self.neighbor_search == "hashgrid" and JAX_AVAILABLE and JaxHashGridNeighbors is not None:
            self._jax_hash = JaxHashGridNeighbors(
                positions=self.positions_jax,
                cell_size=max(self._mean_h, 1e-6),
                per_cell_cap=int(self.per_cell_cap),
            )
            self._neighbor_backend = "hashgrid"

        # 6) Compile evaluators
        self._evaluate_jit = jit(self._evaluate_density)

    # ---------- Adaptive smoothing (existing design, made explicit as cache) ----------

    def _calculate_smoothing_lengths(self) -> jnp.ndarray:
        """
        Choose adaptive or fixed smoothing lengths with memory safeguards, matching your
        adaptive path and fallback behavior
        """
        if self.adaptive and self.N <= int(self.max_particles_for_adaptive):
            try:
                return self._adaptive_smoothing_lengths()
            except Exception as e:
                print(f"Warning: Adaptive smoothing failed ({e}). Using fixed smoothing")
                self.adaptive = False
                return self._fixed_smoothing_length()
        return self._fixed_smoothing_length()

    def refresh_smoothing_lengths(self, force: bool = False):
        """
        Recompute hi and rebuild neighbor grid if needed.
        Call this after positions change or n_neighbors changes.
        """
        if not self.adaptive and not force:
            return
        self.smoothing_lengths = self._calculate_smoothing_lengths()
        self._mean_h = float(jnp.mean(self.smoothing_lengths))
        if self._neighbor_backend == "hashgrid":
            # rebuild device grid with new average cell size
            self._jax_hash = JaxHashGridNeighbors(
                positions=self.positions_jax,
                cell_size=max(self._mean_h, 1e-6),
                per_cell_cap=int(self.per_cell_cap),
            )
        self._h_cache_version += 1

    def _fixed_smoothing_length(self) -> jnp.ndarray:
        # same idea as your fixed path: estimate from domain/particle spacing
        n_per_dim = self.N ** (1.0 / self.D)
        pos_min = jnp.min(self.positions_jax, axis=0)
        pos_max = jnp.max(self.positions_jax, axis=0)
        domain_size = jnp.mean(pos_max - pos_min)
        h = domain_size / n_per_dim * 2.0
        h = jnp.clip(h, 0.01, domain_size * 0.1)
        return jnp.full(self.N, h)

    def _adaptive_smoothing_lengths(self) -> jnp.ndarray:
        """
        Compute hi from k-th nearest neighbor distances with chunking, as in your current code
        """
        k = int(min(self.n_neighbors, max(self.N - 1, 1)))
        chunk_size = min(500, self.N)
        outs = []

        for s in range(0, self.N, chunk_size):
            e = min(s + chunk_size, self.N)
            idx = jnp.arange(s, e)

            def nth_dist(i):
                pi = self.positions_jax[i]
                d = jnp.linalg.norm(self.positions_jax - pi[None, :], axis=1)
                nn = jnp.argpartition(d, k)
                kth = d[nn[k]]
                return 1.2 * kth  # small safety factor

            outs.append(vmap(nth_dist)(idx))

        return jnp.concatenate(outs)

    # ---------- Kernels (reusing your definitions/normalizations) ----------

    def _kernel_cubic(self, r: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        q = r / jnp.maximum(h, 1e-12)
        if self.D == 2:
            sigma = 10.0 / (7.0 * jnp.pi * jnp.maximum(h, 1e-12) ** 2)
        else:
            sigma = 1.0 / (jnp.pi * jnp.maximum(h, 1e-12) ** 3)
        w1 = sigma * (1.0 - 1.5 * q**2 + 0.75 * q**3)
        w2 = sigma * 0.25 * (2.0 - q) ** 3
        return jnp.where(q <= 1.0, w1, jnp.where(q <= 2.0, w2, 0.0))

    def _kernel_wendland(self, r: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        q = r / jnp.maximum(h, 1e-12)
        if self.D == 2:
            sigma = 7.0 / (jnp.pi * jnp.maximum(h, 1e-12) ** 2)
        else:
            sigma = 21.0 / (2.0 * jnp.pi * jnp.maximum(h, 1e-12) ** 3)
        t = jnp.maximum(1.0 - q, 0.0)
        return sigma * (t**4) * (1.0 + 4.0 * q)

    def _kernel(self, r: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        if self.kernel_type == "cubic_spline":
            return self._kernel_cubic(r, h)
        elif self.kernel_type == "wendland":
            return self._kernel_wendland(r, h)
        elif self.kernel_type == "gaussian":
            # normalized Gaussian with h as std-like scale
            if self.D == 2:
                sigma = 1.0 / (jnp.pi * jnp.maximum(h, 1e-12) ** 2)
            else:
                sigma = 1.0 / ((jnp.pi * jnp.maximum(h, 1e-12) ** 2) ** 1.5)
            q2 = (r / jnp.maximum(h, 1e-12)) ** 2
            return sigma * jnp.exp(-q2)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    # ---------- Evaluation (JAX hash-grid path + fallback) ----------

    def _evaluate_density(self, eval_points: jnp.ndarray) -> jnp.ndarray:
        """
        Use the device hash-grid when available; otherwise fall back to the current
        distance-scan approach with cutoff and chunking
        """
        Q = eval_points

        if self._neighbor_backend == "hashgrid":
            # Single radius for candidate collection; per-particle hi still applied in kernel
            radius = 2.0 * self._mean_h
            idxs, dists = self._jax_hash.query_many(Q, radius=radius, max_neighbors=int(self.max_neighbors))
            # Gather per-neighbor masses and hi
            valid = (idxs >= 0)
            masses = jnp.where(valid, self.masses_jax[idxs], 0.0)
            hi = jnp.where(valid, self.smoothing_lengths[idxs], self._mean_h)

            w = self._kernel(dists, hi)
            dens = jnp.sum(masses * w, axis=1)
            return dens

        # Fallback: your existing cutoff-based O(N) scan path, JITed
        def single_point_density(point):
            d = jnp.linalg.norm(self.positions_jax - point[None, :], axis=1)
            avg_h = jnp.mean(self.smoothing_lengths)
            cutoff = 3.0 * avg_h
            mask = d <= cutoff
            idx = jnp.where(mask, jnp.arange(self.N), -1)
            idx = idx[idx >= 0]
            # If too few, include all (as in your code)
            idx = jax.lax.cond(idx.size < 10, lambda _: jnp.arange(self.N), lambda _: idx, operand=None)
            di = d[idx]
            mi = self.masses_jax[idx]
            hi = self.smoothing_lengths[idx]
            wi = self._kernel(di, hi)
            return jnp.sum(mi * wi)

        return vmap(single_point_density)(Q)

    # ---------- Public API ----------

    def evaluate(self, eval_points: np.ndarray) -> np.ndarray:
        """
        Chunked evaluation driver, matching your current approach to avoid OOM
        """
        Q = jnp.asarray(eval_points)
        chunk = 1000 if self.D == 2 else 500
        outs = []
        for s in range(0, Q.shape[0], chunk):
            e = min(s + chunk, Q.shape[0])
            try:
                outs.append(self._evaluate_jit(Q[s:e]))
            except Exception as ex:
                # Fallback to a simple weighting, as in your current code
                outs.append(self._fallback_density_evaluation(Q[s:e]))
        rho = jnp.concatenate(outs, axis=0)

        # Dimensionless normalization (same style as your module)
        if self.normalize:
            if self.D == 2:
                # Estimate area from query grid
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

    # keep your existing _fallback_density_evaluation and slicing helpers
    def _fallback_density_evaluation(self, eval_points: jnp.ndarray) -> jnp.ndarray:
        """
        Simple unoptimized fallback for density evaluation, matching your current code
        """
        Q = eval_points
        chunk_size = 100 if self.D == 2 else 50
        outs = []

        for s in range(0, Q.shape[0], chunk_size):
            e = min(s + chunk_size, Q.shape[0])
            idx = jnp.arange(s, e)

            def single_point_density(i):
                point = Q[i]
                d = jnp.linalg.norm(self.positions_jax - point[None, :], axis=1)
                avg_h = jnp.mean(self.smoothing_lengths)
                cutoff = 3.0 * avg_h
                mask = d <= cutoff
                idx = jnp.where(mask, jnp.arange(self.N), -1)
                idx = idx[idx >= 0]
                # If too few, include all (as in your code)
                idx = jax.lax.cond(idx.size < 10, lambda _: jnp.arange(self.N), lambda _: idx, operand=None)
                di = d[idx]
                mi = self.masses_jax[idx]
                hi = self.smoothing_lengths[idx]
                wi = self._kernel(di, hi)
                return jnp.sum(mi * wi)

            outs.append(vmap(single_point_density)(idx))

        return jnp.concatenate(outs)