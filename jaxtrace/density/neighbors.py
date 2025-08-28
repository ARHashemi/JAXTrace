# density/neighbors.py (append)

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False


@dataclass
class JaxHashGridNeighbors:
    """
    Device-side uniform hash-grid for fast range queries under jit/vmap.

    - Packs integer cells into 64-bit keys.
    - CSR-like storage: unique cell ids + (start,count) into a sorted point index array.
    - Query returns fixed-size candidate sets with -1 padding for JIT-friendliness.
    """
    positions: "jax.Array"
    cell_size: float
    per_cell_cap: int = 64  # max candidates gathered per cell

    def __post_init__(self):
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is required for JaxHashGridNeighbors")
        P = jnp.asarray(self.positions)
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

    def _cell_of(self, q: "jax.Array") -> "jax.Array":
        return jnp.floor((q - self.origin) * self.inv).astype(jnp.int32) - self.Cmin

    def query_many(self, Q: "jax.Array", radius: float, max_neighbors: int):
        """
        Batched range query.

        Returns
        -------
        idxs  : (M, K) int32, -1 padded
        dists : (M, K) float32, inf for padded
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
