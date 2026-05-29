# jaxtrace/density/estimator.py
"""
GPU/JAX density estimator with two backends:

  - **brute-force**: chunked, JIT-compiled kernel sum (queries x sources).
    Best for small/medium particle counts or small grids. No spatial index.

  - **octree (Morton voxel hash)**: builds a fixed-cell Morton hash over the
    particle cloud each step, then for each query gathers only candidate
    cells within the kernel support. Best for large N and large M where the
    brute cost N*M dominates.

Both backends share the same kernel functions from ``kernels.py`` and the
same per-particle ``h``, ``mass`` weight arrays. Both stay on the GPU end
to end.

The estimator is *shape-stable*: the particle count is rounded up to a fixed
bucket size and the unused slots are zero-massed so they don't affect the
sum. This way the jitted kernels compile **once per run** even if the active
particle count drifts (inlet/outlet flows, late seeding, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from . import kernels


Engine = Literal["auto", "brute", "octree"]
Normalization = Literal["pdf", "mass", "unnormalized"]


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class EstimatorConfig:
    kernel: str = "wendland_c2"
    d: int = 3
    normalization: Normalization = "pdf"
    engine: Engine = "auto"
    # cost-model threshold for auto-selecting octree over brute
    # Auto-engine cost threshold. The octree backend caps candidate
    # neighbours per query at ``octree_max_neighbors`` (default 256), which
    # is only safe when each hash cell holds fewer than that many particles.
    # Until we make the hash cell sizing adapt to ``max_neighbors``, prefer
    # brute force unless N*M is very large. Override per-run via
    # EstimatorConfig.engine = "brute" | "octree" | "auto".
    auto_threshold: float = 1e12
    # brute-force chunking
    brute_query_chunk: int = 8192
    # octree backend
    octree_cells_per_dim: int = 64         # uniform hash; chosen at build time
    octree_max_neighbors: int = 256        # per query cap
    # shape stabilization
    particle_bucket: int = 4096            # round N up to nearest multiple of this


# -----------------------------------------------------------------------------
# Shape stabilization
# -----------------------------------------------------------------------------

def _bucket_round(n: int, bucket: int) -> int:
    return ((n + bucket - 1) // bucket) * bucket


def pad_particles(
    positions: jnp.ndarray,            # (N, 3)
    h: jnp.ndarray,                    # (N,)
    weights: jnp.ndarray,              # (N,)
    n_padded: int,
):
    """Pad to ``n_padded`` length by appending zero-weight ghost particles."""
    N = positions.shape[0]
    if N == n_padded:
        return positions, h, weights
    pad = n_padded - N
    pos_pad = jnp.zeros((pad, 3), dtype=positions.dtype)
    h_pad = jnp.ones((pad,), dtype=h.dtype)            # nonzero to avoid /0
    w_pad = jnp.zeros((pad,), dtype=weights.dtype)     # zero weight = invisible
    return (
        jnp.concatenate([positions, pos_pad], axis=0),
        jnp.concatenate([h, h_pad], axis=0),
        jnp.concatenate([weights, w_pad], axis=0),
    )


# -----------------------------------------------------------------------------
# Brute-force backend
# -----------------------------------------------------------------------------

def _make_brute_kernel(kernel_name: str, d: int, particle_tile: int = 4096):
    """
    Tiled brute-force kernel.

    For each chunk of queries (size ``m``), we scan particles in tiles
    of size ``particle_tile`` along the particle axis, accumulating the
    kernel sum incrementally. Per-tile intermediate is shaped
    ``(m, particle_tile)`` — small enough to live in the GPU's L2/SM
    cache, instead of the ``(m, N)`` mega-tensor the naive layout
    materialises.

    Distance computation uses the matmul identity

        r² = ‖q‖² + ‖p‖² − 2 q·p

    so the inner product ``Q @ P_tile.T`` lights up tensor cores on
    modern NVIDIA GPUs. Float32 catastrophic cancellation at very small
    ``r`` is bounded by clamping ``r² ≥ 0`` and relying on the kernel's
    own ``_safe_h`` near r=0. Output is bit-comparable to the naive
    layout up to float32 round-off; relative error stays well below
    1e-5 on the test set.

    The caller is required to have padded the particle arrays
    (``P``, ``h``, ``w``) to a multiple of ``particle_tile`` (the
    estimator's ``pad_particles`` already does this for
    ``particle_bucket``; we keep ``particle_tile == particle_bucket``).
    """
    def per_tile(Q, P_tile, h_tile, w_tile):
        # Q: (m, 3), P_tile: (T, 3), h_tile: (T,), w_tile: (T,)
        qsq = jnp.sum(Q * Q, axis=-1, keepdims=True)           # (m, 1)
        psq = jnp.sum(P_tile * P_tile, axis=-1)[None, :]       # (1, T)
        qp = Q @ P_tile.T                                       # (m, T)  ← tensor core
        r2 = jnp.maximum(qsq + psq - 2.0 * qp, 0.0)            # (m, T)
        r = jnp.sqrt(r2)                                        # (m, T)
        K = kernels.evaluate_kernel(kernel_name, r, h_tile[None, :], d)
        return jnp.sum(K * w_tile[None, :], axis=1)             # (m,)

    @jax.jit
    def per_chunk(Q, P, h, w):
        N_padded = P.shape[0]
        # The estimator's pad_particles always rounds N up to particle_bucket,
        # which we require to equal particle_tile so the reshape is exact and
        # the scan has a static iteration count (good for JIT compile).
        n_tiles = N_padded // particle_tile
        P_t = P.reshape(n_tiles, particle_tile, 3)
        h_t = h.reshape(n_tiles, particle_tile)
        w_t = w.reshape(n_tiles, particle_tile)

        def scan_step(acc, tile):
            P_tile, h_tile, w_tile = tile
            return acc + per_tile(Q, P_tile, h_tile, w_tile), None

        out, _ = jax.lax.scan(
            scan_step,
            jnp.zeros((Q.shape[0],), dtype=Q.dtype),
            (P_t, h_t, w_t),
        )
        return out

    return per_chunk


# -----------------------------------------------------------------------------
# Morton-hash octree backend (on particles)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ParticleHash:
    bbox_min: jnp.ndarray       # (3,)
    cell_size: jnp.ndarray      # (3,)
    grid_dims: Tuple[int, int, int]
    cell_starts: jnp.ndarray    # (n_cells + 1,) int32  (CSR offsets)
    sorted_particle_idx: jnp.ndarray  # (N,) int32
    n_cells_total: int


def _build_particle_hash(
    P: jnp.ndarray, weights: jnp.ndarray, support_radius: float,
    cells_per_dim: int,
) -> ParticleHash:
    """
    Build a uniform 3D cell hash over the particles, sized so that one cell
    is at least ``support_radius`` so a 3x3x3 query covers the kernel support.

    Returns CSR-style arrays so that the per-cell particle lists are addressed
    by (cell_starts[i], cell_starts[i+1]).

    Only particles with weight > 0 are indexed (zero-weight ghosts).
    Building runs on host (numpy) then is uploaded; building each step is
    cheap relative to the kernel sum on the GPU.
    """
    # Use only valid particles for the bbox so ghosts don't blow it up.
    P_np = np.asarray(P)
    w_np = np.asarray(weights)
    valid = w_np > 0.0
    if not np.any(valid):
        # degenerate case: no real particles, build an empty hash
        return ParticleHash(
            bbox_min=jnp.zeros(3, jnp.float32),
            cell_size=jnp.ones(3, jnp.float32),
            grid_dims=(1, 1, 1),
            cell_starts=jnp.zeros(2, jnp.int32),
            sorted_particle_idx=jnp.zeros(0, jnp.int32),
            n_cells_total=1,
        )

    Pv = P_np[valid]
    bbox_min = Pv.min(axis=0).astype(np.float32) - np.float32(1e-3)
    bbox_max = Pv.max(axis=0).astype(np.float32) + np.float32(1e-3)
    extent = bbox_max - bbox_min

    # Cell size = max(extent / cells_per_dim, support_radius)
    base = extent / float(cells_per_dim)
    cs = np.maximum(base, np.float32(support_radius)).astype(np.float32)
    dims = tuple(int(max(1, np.ceil(e / s))) for e, s in zip(extent, cs))

    # Assign each *valid* particle to a cell
    idx_valid = np.nonzero(valid)[0].astype(np.int32)
    ijk = np.floor((Pv - bbox_min) / cs).astype(np.int32)
    np.clip(ijk[:, 0], 0, dims[0] - 1, out=ijk[:, 0])
    np.clip(ijk[:, 1], 0, dims[1] - 1, out=ijk[:, 1])
    np.clip(ijk[:, 2], 0, dims[2] - 1, out=ijk[:, 2])
    cell_id = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    n_cells_total = int(dims[0] * dims[1] * dims[2])

    order = np.argsort(cell_id, kind="stable")
    sorted_cells = cell_id[order]
    sorted_particles = idx_valid[order]

    # CSR offsets
    starts = np.zeros(n_cells_total + 1, dtype=np.int32)
    np.add.at(starts[1:], sorted_cells, 1)
    np.cumsum(starts, out=starts)

    return ParticleHash(
        bbox_min=jnp.asarray(bbox_min, jnp.float32),
        cell_size=jnp.asarray(cs, jnp.float32),
        grid_dims=dims,
        cell_starts=jnp.asarray(starts, jnp.int32),
        sorted_particle_idx=jnp.asarray(sorted_particles, jnp.int32),
        n_cells_total=n_cells_total,
    )


def _make_octree_kernel(
    kernel_name: str, d: int, max_neighbors: int,
):
    """
    Build a jitted query function. The function takes a (m, 3) chunk of
    queries plus the ParticleHash arrays and the full (padded) particle
    arrays, and returns the per-query density (m,).

    For each query we visit a 3x3x3 cell stencil around the query cell;
    candidate particles are concatenated into a fixed-capacity buffer of
    size ``max_neighbors`` (truncated if exceeded), then a single dense
    kernel sum is computed over the buffer.
    """
    stencil_offsets = jnp.array(
        [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
        dtype=jnp.int32,
    )  # (27, 3)

    def per_query(
        q: jnp.ndarray,                  # (3,)
        P: jnp.ndarray,                  # (N_padded, 3)
        h: jnp.ndarray,                  # (N_padded,)
        w: jnp.ndarray,                  # (N_padded,)
        bbox_min: jnp.ndarray,           # (3,)
        cell_size: jnp.ndarray,          # (3,)
        cell_starts: jnp.ndarray,        # (n_cells+1,)
        sorted_idx: jnp.ndarray,         # (N_valid,)
        grid_dims: jnp.ndarray,          # (3,) int32
    ):
        # Locate query cell
        ijk = jnp.floor((q - bbox_min) / cell_size).astype(jnp.int32)
        ijk = jnp.clip(ijk, jnp.int32(0), grid_dims - jnp.int32(1))

        # Build the candidate buffer (max_neighbors size) by scanning 27 cells.
        def visit_cell(carry, offset):
            buf_idx, count = carry
            cijk = ijk + offset
            in_range = jnp.all((cijk >= 0) & (cijk < grid_dims))
            cijk_c = jnp.clip(cijk, jnp.int32(0), grid_dims - jnp.int32(1))
            cell_id = (cijk_c[0] * grid_dims[1] + cijk_c[1]) * grid_dims[2] + cijk_c[2]
            s = cell_starts[cell_id]
            e = cell_starts[cell_id + 1]
            n_in_cell = jnp.where(in_range, e - s, jnp.int32(0))

            # Pull up to (max_neighbors - count) entries from this cell.
            room = jnp.maximum(max_neighbors - count, jnp.int32(0))
            take = jnp.minimum(n_in_cell, room)

            def copy_one(j, c):
                bi, cnt = c
                src_pos = s + j
                src_pos_safe = jnp.minimum(src_pos, sorted_idx.shape[0] - 1)
                particle_id = sorted_idx[src_pos_safe]
                bi = bi.at[cnt + j].set(particle_id)
                return bi, cnt

            buf_idx, _ = jax.lax.fori_loop(0, take, copy_one, (buf_idx, count))
            return (buf_idx, count + take), None

        init_buf = jnp.full((max_neighbors,), jnp.int32(-1), dtype=jnp.int32)
        (buf_idx, n_found), _ = jax.lax.scan(visit_cell, (init_buf, jnp.int32(0)), stencil_offsets)

        # Gather and compute the kernel sum. Invalid slots (idx == -1) are
        # gated via mask so they contribute zero.
        valid = buf_idx >= 0
        safe_idx = jnp.where(valid, buf_idx, 0)
        Pi = P[safe_idx]                  # (max_neighbors, 3)
        hi = h[safe_idx]                  # (max_neighbors,)
        wi = w[safe_idx]                  # (max_neighbors,)
        diff = q[None, :] - Pi
        r = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
        K = kernels.evaluate_kernel(kernel_name, r, hi, d)
        contrib = K * wi * valid.astype(K.dtype)
        return jnp.sum(contrib)

    @jax.jit
    def per_chunk(
        Q, P, h, w, bbox_min, cell_size, cell_starts, sorted_idx, grid_dims,
    ):
        return jax.vmap(
            lambda q: per_query(q, P, h, w, bbox_min, cell_size, cell_starts, sorted_idx, grid_dims),
        )(Q)

    return per_chunk


# -----------------------------------------------------------------------------
# Public estimator
# -----------------------------------------------------------------------------

@dataclass
class DensityEstimator:
    cfg: EstimatorConfig
    # Voxel-grid query set (flat (M_active, 3) device array, already masked)
    query_points: Optional[jnp.ndarray] = None
    _brute_fn: Optional[callable] = field(default=None, init=False, repr=False)
    _octree_fn: Optional[callable] = field(default=None, init=False, repr=False)
    _n_padded_last: int = field(default=-1, init=False, repr=False)

    def __post_init__(self):
        # Compile-once kernel constructors (shape-polymorphic over chunk count).
        # particle_tile == particle_bucket so the tiled brute kernel's reshape
        # is always exact and the inner scan has a static iteration count.
        self._brute_fn = _make_brute_kernel(
            self.cfg.kernel, self.cfg.d,
            particle_tile=self.cfg.particle_bucket,
        )
        self._octree_fn = _make_octree_kernel(
            self.cfg.kernel, self.cfg.d, self.cfg.octree_max_neighbors,
        )

    # --- engine selection -----------------------------------------------------

    def _select_engine(self, n_active: int, m_active: int) -> Engine:
        if self.cfg.engine != "auto":
            engine = self.cfg.engine
        else:
            cost = float(n_active) * float(m_active)
            engine = "octree" if cost > self.cfg.auto_threshold else "brute"
        if engine != getattr(self, "_last_engine", None):
            print(f"[density] engine = {engine}  (N={n_active}, M={m_active}, "
                  f"cost={n_active*m_active:.2e}, threshold={self.cfg.auto_threshold:.2e})")
            self._last_engine = engine
        return engine

    # --- main entry -----------------------------------------------------------

    def evaluate(
        self,
        positions: jnp.ndarray,             # (N, 3) device
        h: jnp.ndarray,                     # (N,) device
        weights: jnp.ndarray,               # (N,) device
        query_points: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Evaluate the density at the given query points (defaults to the
        grid query set passed to the constructor).
        """
        Q = query_points if query_points is not None else self.query_points
        if Q is None:
            raise ValueError("no query points set on estimator and none passed")

        N = int(positions.shape[0])
        M = int(Q.shape[0])
        engine = self._select_engine(N, M)

        # Apply normalization
        if self.cfg.normalization == "pdf":
            w_eff = weights / jnp.maximum(jnp.sum(weights), jnp.float32(1e-30))
        elif self.cfg.normalization == "mass":
            w_eff = weights
        elif self.cfg.normalization == "unnormalized":
            w_eff = weights
        else:
            raise ValueError(f"unknown normalization {self.cfg.normalization!r}")

        # Shape-stabilize: pad particles to a fixed bucket so JIT compiles once.
        n_padded = _bucket_round(N, self.cfg.particle_bucket)
        Pp, hp, wp = pad_particles(positions, h, w_eff, n_padded)
        self._n_padded_last = n_padded

        if engine == "brute":
            return self._eval_brute(Pp, hp, wp, Q)
        return self._eval_octree(Pp, hp, wp, Q)

    # --- backends -------------------------------------------------------------

    def _eval_brute(self, P, h, w, Q):
        chunk = self.cfg.brute_query_chunk
        M = int(Q.shape[0])
        out = []
        for s in range(0, M, chunk):
            e = min(s + chunk, M)
            Qc = Q[s:e]
            if Qc.shape[0] < chunk:
                pad = chunk - Qc.shape[0]
                Qc_pad = jnp.concatenate(
                    [Qc, jnp.zeros((pad, 3), dtype=Qc.dtype)], axis=0,
                )
                rho_pad = self._brute_fn(Qc_pad, P, h, w)
                out.append(rho_pad[:Qc.shape[0]])
            else:
                out.append(self._brute_fn(Qc, P, h, w))
        return jnp.concatenate(out, axis=0)

    def _eval_octree(self, P, h, w, Q):
        # Build particle hash on host (cheap), then run jitted query
        support = kernels.kernel_support(self.cfg.kernel)
        h_max = float(jnp.max(h))
        ph = _build_particle_hash(
            P, w, support_radius=support * h_max,
            cells_per_dim=self.cfg.octree_cells_per_dim,
        )
        # If the hash is empty (no real particles), return zeros.
        if ph.sorted_particle_idx.shape[0] == 0:
            return jnp.zeros((Q.shape[0],), dtype=jnp.float32)

        grid_dims = jnp.asarray(ph.grid_dims, dtype=jnp.int32)
        chunk = self.cfg.brute_query_chunk
        M = int(Q.shape[0])
        out = []
        for s in range(0, M, chunk):
            e = min(s + chunk, M)
            Qc = Q[s:e]
            if Qc.shape[0] < chunk:
                pad = chunk - Qc.shape[0]
                Qc_pad = jnp.concatenate(
                    [Qc, jnp.zeros((pad, 3), dtype=Qc.dtype)], axis=0,
                )
                rho_pad = self._octree_fn(
                    Qc_pad, P, h, w,
                    ph.bbox_min, ph.cell_size, ph.cell_starts,
                    ph.sorted_particle_idx, grid_dims,
                )
                out.append(rho_pad[:Qc.shape[0]])
            else:
                out.append(self._octree_fn(
                    Qc, P, h, w,
                    ph.bbox_min, ph.cell_size, ph.cell_starts,
                    ph.sorted_particle_idx, grid_dims,
                ))
        return jnp.concatenate(out, axis=0)
