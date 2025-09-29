# jaxtrace/density/neighbors.py
"""
Fast neighbor search using uniform hash grids.

Provides device-optimized spatial data structures for range queries
with JAX acceleration and JIT-friendly fixed-size output arrays.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

# Import JAX utilities with fallback
from ..utils.jax_utils import JAX_AVAILABLE

if JAX_AVAILABLE:
    try:
        import jax
        import jax.numpy as jnp
    except Exception:
        JAX_AVAILABLE = False

if not JAX_AVAILABLE:
    import numpy as jnp  # type: ignore
    # Mock jax functions for NumPy fallback
    class MockJax:
        def vmap(self, func):
            def vectorized(args):
                if isinstance(args, (list, tuple)):
                    return [func(arg) for arg in args]
                return np.array([func(arg) for arg in args])
            return vectorized
        def jit(self, func):
            return func
    jax = MockJax()


@dataclass
class HashGridNeighbors:
    """
    Device-side uniform hash-grid for fast range queries under jit/vmap.

    - Packs integer cells into 64-bit keys.
    - CSR-like storage: unique cell ids + (start,count) into a sorted point index array.
    - Query returns fixed-size candidate sets with -1 padding for JIT-friendliness.
    
    Attributes
    ----------
    positions : jnp.ndarray
        Point coordinates, shape (N, D) where D=2 or 3
    cell_size : float
        Spatial hash grid cell size
    per_cell_cap : int
        Maximum candidates gathered per cell for memory control
    """
    positions: jnp.ndarray   # (N,D)
    cell_size: float
    per_cell_cap: int = 64      # max candidates gathered per cell

    def __post_init__(self):
        if not JAX_AVAILABLE:
            print("Warning: JAX not available, HashGridNeighbors will use NumPy fallback")
            
        P = jnp.asarray(self.positions)
        if P.ndim != 2 or P.shape[1] not in (2, 3):
            raise ValueError("positions must be (N,2) or (N,3)")
        self.positions = P
        self.D = int(P.shape[1])
        self.inv = 1.0 / jnp.maximum(self.cell_size, 1e-12)
        self.origin = jnp.min(P, axis=0)

        C = jnp.floor((P - self.origin) * self.inv).astype(jnp.int32)
        Cmin = jnp.min(C, axis=0)
        Csh = C - Cmin  # non-negative
        self.Cmin = Cmin

        if self.D == 2:
            self._pack = lambda c: (c[:, 0].astype(jnp.int64) << 32) | c[:, 1].astype(jnp.int64)
            self._pack_single = lambda c: (c[0].astype(jnp.int64) << 32) | c[1].astype(jnp.int64)
            offs = jnp.array([[di, dj] for di in (-1, 0, 1) for dj in (-1, 0, 1)], dtype=jnp.int32)
        else:
            self._pack = lambda c: ((c[:, 0].astype(jnp.int64) << 42) |
                                    (c[:, 1].astype(jnp.int64) << 21) |
                                     c[:, 2].astype(jnp.int64))
            self._pack_single = lambda c: ((c[0].astype(jnp.int64) << 42) |
                                           (c[1].astype(jnp.int64) << 21) |
                                            c[2].astype(jnp.int64))
            offs = jnp.array([[di, dj, dk] for di in (-1, 0, 1)
                                           for dj in (-1, 0, 1)
                                           for dk in (-1, 0, 1)], dtype=jnp.int32)
        self.neighbor_offsets = offs

        cell_ids = self._pack(Csh)
        order = jnp.argsort(cell_ids)
        self.pt_indices_sorted = order
        self.cell_ids_sorted = cell_ids[order]

        uniq, first_idx = jnp.unique(self.cell_ids_sorted, return_index=True)
        counts = jnp.diff(jnp.concatenate([first_idx, jnp.array([self.cell_ids_sorted.size])], axis=0))
        self.unique_cell_ids = uniq
        self.unique_starts = first_idx
        self.unique_counts = counts

    def _cell_of(self, q: jnp.ndarray) -> jnp.ndarray:
        """Convert query point to grid cell coordinates."""
        return jnp.floor((q - self.origin) * self.inv).astype(jnp.int32) - self.Cmin

    def query_many(self, Q: jnp.ndarray, radius: float, max_neighbors: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Batched range query for multiple points.

        Parameters
        ----------
        Q : jnp.ndarray
            Query points, shape (M, D)
        radius : float
            Search radius
        max_neighbors : int
            Maximum neighbors to return per query point

        Returns
        -------
        idxs : jnp.ndarray
            Neighbor indices, shape (M, K), -1 padded
        dists : jnp.ndarray
            Distances to neighbors, shape (M, K), inf for padded
        """
        Q = jnp.asarray(Q)
        r = jnp.asarray(radius)
        Kcells = self.neighbor_offsets.shape[0]
        per_cell_cap = int(self.per_cell_cap)
        Nsorted = self.pt_indices_sorted.size

        def query_one(q):
            cq = self._cell_of(q)
            ngh = cq[None, :] + self.neighbor_offsets  # (Kcells,D)
            ngh_ids = jax.vmap(self._pack_single)(ngh)

            pos = jnp.searchsorted(self.unique_cell_ids, ngh_ids, side="left")
            present = (pos < self.unique_cell_ids.size) & (self.unique_cell_ids[pos] == ngh_ids)
            starts = jnp.where(present, self.unique_starts[pos], -1)
            counts = jnp.where(present, self.unique_counts[pos], 0)

            ar = jnp.arange(per_cell_cap, dtype=jnp.int32)
            base = starts[:, None] + ar[None, :]
            valid_slot = (starts[:, None] >= 0) & (ar[None, :] < counts[:, None])
            base_safe = jnp.clip(base, 0, jnp.maximum(Nsorted - 1, 0))
            cand_sorted = jnp.where(valid_slot, base_safe, 0)
            cand_pts = self.pt_indices_sorted[cand_sorted].reshape(-1)
            cand_valid = valid_slot.reshape(-1)

            pts = self.positions[cand_pts]
            d = jnp.linalg.norm(pts - q[None, :], axis=1)
            in_r = d <= r
            valid = cand_valid & in_r

            d_sel = jnp.where(valid, d, jnp.inf)
            order = jnp.argsort(d_sel)[:max_neighbors]
            sel_idx = cand_pts[order].astype(jnp.int32)
            sel_dist = d_sel[order].astype(self.positions.dtype)

            pad = max_neighbors - sel_idx.size
            sel_idx = jnp.pad(sel_idx, (0, pad), constant_values=-1)
            sel_dist = jnp.pad(sel_dist, (0, pad), constant_values=jnp.inf)
            return sel_idx, sel_dist

        return jax.vmap(query_one)(Q)