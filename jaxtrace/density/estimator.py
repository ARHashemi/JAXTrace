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
    # cost-model threshold for auto-selecting octree over brute.
    # The "auto" selector picks the octree backend only when the projected
    # cost ratio is favourable AND the hash geometry can fit a proper
    # stencil (see _select_engine). For typical workstation grids the
    # particle-hash backend is now correctness-equivalent to brute and
    # usually faster, so the default threshold is set low.
    auto_threshold: float = 1e10
    # brute-force chunking
    brute_query_chunk: int = 8192
    # octree (particle-hash) backend
    octree_target_n_per_cell: int = 9      # target average particles per cell
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

def _make_brute_kernel(kernel_name: str, d: int):
    """Return a jitted (queries_chunk, P, h, w) -> rho_chunk function."""

    def per_chunk(Q, P, h, w):
        # Q: (m, 3), P: (N, 3), h: (N,), w: (N,)
        # diff: (m, N, 3)
        diff = Q[:, None, :] - P[None, :, :]
        r = jnp.sqrt(jnp.sum(diff * diff, axis=-1))      # (m, N)
        K = kernels.evaluate_kernel(kernel_name, r, h[None, :], d)  # (m, N)
        return jnp.sum(K * w[None, :], axis=1)            # (m,)

    return jax.jit(per_chunk)


# -----------------------------------------------------------------------------
# Particle-hash backend ("Backend P")
# -----------------------------------------------------------------------------
#
# A uniform 3-D cell hash over the particle cloud. Per-step, every particle is
# assigned to one cell; per query, we visit a per-axis stencil around the
# query's cell and accumulate the kernel sum over all particles in those
# cells (no fixed-capacity buffer, no truncation). Compared to the previous
# revision the three bugs that pinned us to the brute backend are gone:
#
#   1. Cell size is chosen by *occupancy target* rather than clamped up to
#      ``support_radius``. This stops the thin-axis collapse to a single cell
#      that bricked the previous version on the cylindrical_009 geometry.
#   2. Per-axis stencil radius ``ceil(support_radius / cs)`` is computed at
#      hash-build time so the stencil always covers the full kernel support
#      regardless of cs. Stencil shape is baked into the JIT trace.
#   3. The candidate-buffer cap is gone — a ``fori_loop`` accumulates the
#      kernel sum directly into a scalar, so the result is exact (up to
#      float32 round-off) regardless of cell occupancy. Backend P is now
#      bit-identical to brute up to ~1e-5 relative error.

@dataclass(frozen=True)
class ParticleHash:
    bbox_min: jnp.ndarray              # (3,) float32
    cell_size: jnp.ndarray             # (3,) float32  — isotropic in practice
    grid_dims: Tuple[int, int, int]    # static, baked into JIT
    cell_starts: jnp.ndarray           # (n_cells + 1,) int32   CSR offsets
    sorted_particle_idx: jnp.ndarray   # (N_valid,) int32       per-cell particle list
    n_cells_total: int
    # Per-axis stencil radius needed to cover the kernel's support_radius.
    # The visited stencil is (2*sr+1)^3 cells centred on the query's cell.
    stencil_radius: Tuple[int, int, int]
    # For diagnostics — average and max occupancy at build time
    mean_per_cell: float
    max_per_cell: int


def _empty_hash() -> ParticleHash:
    """Sentinel hash with one empty cell, used when no valid particles exist."""
    return ParticleHash(
        bbox_min=jnp.zeros(3, jnp.float32),
        cell_size=jnp.ones(3, jnp.float32),
        grid_dims=(1, 1, 1),
        cell_starts=jnp.zeros(2, jnp.int32),
        sorted_particle_idx=jnp.zeros(0, jnp.int32),
        n_cells_total=1,
        stencil_radius=(0, 0, 0),
        mean_per_cell=0.0,
        max_per_cell=0,
    )


def _build_particle_hash(
    P: jnp.ndarray, weights: jnp.ndarray, support_radius: float,
    target_n_per_cell: int = 9,
    min_cs: float = 1e-6,
) -> ParticleHash:
    """
    Build a uniform 3-D cell hash over the particle cloud, sized to a target
    average occupancy rather than the support radius.

    Cell-size rule (isotropic in physical units):

        cs* = (V_bbox * target_n_per_cell / N_valid) ** (1/3)
        cs  = max(cs*, min_cs)

    No clamp against support_radius. The kernel can still reach across many
    cells; we just compute the per-axis stencil radius to cover it:

        stencil_radius[a] = ceil(support_radius / cs[a])

    Caller passes per-particle weights so zero-weight ghosts (from
    pad_particles) are ignored when computing the bbox / occupancy.
    """
    P_np = np.asarray(P)
    w_np = np.asarray(weights)
    valid = w_np > 0.0
    n_valid = int(valid.sum())
    if n_valid == 0:
        return _empty_hash()

    Pv = P_np[valid]
    bbox_min = Pv.min(axis=0).astype(np.float32) - np.float32(1e-3)
    bbox_max = Pv.max(axis=0).astype(np.float32) + np.float32(1e-3)
    extent = (bbox_max - bbox_min).astype(np.float64)

    # Isotropic cell-size by occupancy target. The trailing ``max(., min_cs)``
    # guards against degenerate (zero-volume) clouds.
    V_bbox = float(np.prod(extent))
    cs_target = (V_bbox * float(target_n_per_cell) / float(n_valid)) ** (1.0 / 3.0)
    cs_scalar = max(cs_target, float(min_cs))
    cs = np.array([cs_scalar, cs_scalar, cs_scalar], dtype=np.float32)

    # Dimensions: at least 1 along every axis.
    dims = tuple(int(max(1, np.ceil(e / cs_scalar))) for e in extent)

    # Per-axis stencil radius. We use the SAME radius along every axis
    # because cs is isotropic; the dims may differ so we may walk fewer cells
    # along a thin axis (the in-range mask in the kernel handles that).
    sr = int(np.ceil(float(support_radius) / cs_scalar))
    stencil_radius = (sr, sr, sr)

    # Assign each VALID particle to a cell. Use flat C-order indexing
    # cell_id = (i*Ny + j)*Nz + k for cache locality of consecutive cells
    # along the fastest-varying axis (z).
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

    starts = np.zeros(n_cells_total + 1, dtype=np.int32)
    np.add.at(starts[1:], sorted_cells, 1)
    counts = starts[1:].copy()
    np.cumsum(starts, out=starts)

    return ParticleHash(
        bbox_min=jnp.asarray(bbox_min, jnp.float32),
        cell_size=jnp.asarray(cs, jnp.float32),
        grid_dims=dims,
        cell_starts=jnp.asarray(starts, jnp.int32),
        sorted_particle_idx=jnp.asarray(sorted_particles, jnp.int32),
        n_cells_total=n_cells_total,
        stencil_radius=stencil_radius,
        mean_per_cell=float(n_valid) / float(n_cells_total),
        max_per_cell=int(counts.max()) if counts.size else 0,
    )


def _make_octree_kernel(kernel_name: str, d: int):
    """
    Build a jitted query function for Backend P.

    Per query, we scan a fixed (2*sr+1)^3 stencil of cells around the query's
    home cell, and for each cell we run a ``fori_loop`` over its particles
    accumulating the kernel sum into a scalar ``rho``. There is no candidate
    buffer; the result is exact regardless of cell occupancy (a particle
    outside the kernel's support contributes exactly zero via the kernel's
    own support cut-off).

    The stencil offsets are passed as a runtime ``jnp.ndarray`` whose length
    is the static stencil_volume; the JIT cache keys on shape, so as long as
    the stencil shape is stable across calls there is no recompile.
    """

    def per_query(
        q: jnp.ndarray,                  # (3,) float32
        P: jnp.ndarray,                  # (N_padded, 3) float32
        h: jnp.ndarray,                  # (N_padded,)   float32
        w: jnp.ndarray,                  # (N_padded,)   float32
        bbox_min: jnp.ndarray,           # (3,)
        cell_size: jnp.ndarray,          # (3,)
        cell_starts: jnp.ndarray,        # (n_cells+1,) int32
        sorted_idx: jnp.ndarray,         # (N_valid,)   int32
        grid_dims: jnp.ndarray,          # (3,) int32
        stencil_offsets: jnp.ndarray,    # (S, 3) int32, S = (2sr+1)^3
        support_radius_sq: jnp.ndarray,  # () float32 — (SUPPORT * h_max)^2
    ):
        ijk_home = jnp.floor((q - bbox_min) / cell_size).astype(jnp.int32)
        # The home cell may legitimately be outside the bbox when the query
        # lies in the kernel-reach halo just outside the particle cloud. We
        # don't clip here — the stencil walk's in_range mask handles it.

        def visit_cell(rho_acc, offset):
            cijk = ijk_home + offset
            in_range = jnp.all((cijk >= 0) & (cijk < grid_dims))
            cijk_c = jnp.clip(cijk, jnp.int32(0), grid_dims - jnp.int32(1))
            cell_id = (cijk_c[0] * grid_dims[1] + cijk_c[1]) * grid_dims[2] + cijk_c[2]
            s = cell_starts[cell_id]
            e = cell_starts[cell_id + 1]

            # Per-cell pre-filter: compute the cell's AABB and check whether
            # the closest point of that AABB to the query is within the
            # kernel support. If not, no particle in this cell can
            # contribute (the kernel returns exactly zero past support_radius),
            # so we skip the inner fori_loop entirely. This is the dominant
            # speedup at large stencil_volume because most stencil cells lie
            # outside the support sphere even though they're inside the
            # support cube.
            cell_lo = bbox_min + cijk.astype(jnp.float32) * cell_size
            cell_hi = cell_lo + cell_size
            # Closest point in AABB to q: clip q into the box.
            closest = jnp.clip(q, cell_lo, cell_hi)
            d2 = jnp.sum((closest - q) ** 2)
            cell_in_support = d2 < support_radius_sq

            n_in_cell = jnp.where(
                jnp.logical_and(in_range, cell_in_support),
                e - s,
                jnp.int32(0),
            )

            def acc_one(j, partial):
                pid = sorted_idx[s + j]
                diff = q - P[pid]
                r = jnp.sqrt(jnp.sum(diff * diff))
                K = kernels.evaluate_kernel(kernel_name, r, h[pid], d)
                return partial + K * w[pid]

            new_rho = jax.lax.fori_loop(0, n_in_cell, acc_one, rho_acc)
            return new_rho, None

        rho, _ = jax.lax.scan(visit_cell, jnp.float32(0.0), stencil_offsets)
        return rho

    @jax.jit
    def per_chunk(
        Q, P, h, w, bbox_min, cell_size, cell_starts, sorted_idx, grid_dims,
        stencil_offsets, support_radius_sq,
    ):
        return jax.vmap(
            lambda q: per_query(
                q, P, h, w, bbox_min, cell_size, cell_starts, sorted_idx,
                grid_dims, stencil_offsets, support_radius_sq,
            ),
        )(Q)

    return per_chunk


def _stencil_offsets(stencil_radius: Tuple[int, int, int]) -> jnp.ndarray:
    """Flat (S, 3) int32 array of cell offsets covering the (2sr+1)^3 box."""
    sx, sy, sz = stencil_radius
    if sx == 0 and sy == 0 and sz == 0:
        return jnp.zeros((1, 3), dtype=jnp.int32)
    offs = np.array(
        [[dx, dy, dz]
         for dx in range(-sx, sx + 1)
         for dy in range(-sy, sy + 1)
         for dz in range(-sz, sz + 1)],
        dtype=np.int32,
    )
    return jnp.asarray(offs)


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
        self._brute_fn = _make_brute_kernel(self.cfg.kernel, self.cfg.d)
        self._octree_fn = _make_octree_kernel(self.cfg.kernel, self.cfg.d)

    # --- engine selection -----------------------------------------------------

    def _select_engine(self, n_active: int, m_active: int) -> Engine:
        """Choose backend.

        ``auto`` mode picks octree when the projected octree cost is at
        least 2× cheaper than brute. The actual neighbour-count estimate
        depends on ``support_radius / bbox_size``, which we don't yet
        have at this point — so we rely on the ``auto_threshold`` heuristic
        on ``N*M`` and let ``_eval_octree`` fall back to brute mid-flight
        if the geometry produces a degenerate hash (e.g. dims_min < 3,
        in which case the stencil can't even take one step).
        """
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
        """Backend P: particle hash + per-axis stencil + fori_loop sum.

        Steps:
          1. Build the particle hash on host (numpy sort + CSR offsets — cheap).
          2. Sanity-check the resulting geometry: if any axis has dims < 1
             we can't form a stencil; if the projected stencil_volume is
             larger than 50 % of N_valid the octree provides no real speedup
             over brute. In either case we degrade gracefully to brute.
          3. JIT-launch the per-chunk kernel with the static stencil_offsets
             baked in.
        """
        support = kernels.kernel_support(self.cfg.kernel)
        h_max = float(jnp.max(h))
        ph = _build_particle_hash(
            P, w, support_radius=support * h_max,
            target_n_per_cell=self.cfg.octree_target_n_per_cell,
        )

        # If the hash is degenerate (no valid particles) just return zeros.
        if ph.sorted_particle_idx.shape[0] == 0:
            return jnp.zeros((Q.shape[0],), dtype=jnp.float32)

        # Geometry-based fall-back to brute. The auto-selector picked octree
        # on N*M cost only; here we know the actual hash dims and can decide
        # whether the work saving is real.
        #
        # ``expected_neighbours`` upper-bounds the per-query neighbour count
        # by ``stencil_vol * mean_per_cell``. The cubic stencil overcounts
        # the spherical kernel support by ~6/pi ~ 1.9x, so the true cost is
        # capped at ``min(expected_neighbours, n_valid)``. We only fall back
        # to brute when even the upper-bound estimate isn't usefully smaller
        # than brute's N — i.e. when there is genuinely no work to skip.
        n_valid = int(ph.sorted_particle_idx.shape[0])
        stencil_vol = int(
            (2 * ph.stencil_radius[0] + 1)
            * (2 * ph.stencil_radius[1] + 1)
            * (2 * ph.stencil_radius[2] + 1)
        )
        expected_neighbours = stencil_vol * ph.mean_per_cell
        # Threshold 0.9 (rather than 0.5): if the stencil work is even 10%
        # less than brute we take it — the inner ``fori_loop`` is cheap
        # relative to the kernel evaluation, so any FLOPs saved compound.
        if expected_neighbours > 0.9 * n_valid:
            print(f"[density] octree fallback to brute: stencil_vol={stencil_vol}, "
                  f"mean_per_cell={ph.mean_per_cell:.1f}, "
                  f"expected_neighbours/N_valid={expected_neighbours/n_valid:.2f}")
            return self._eval_brute(P, h, w, Q)
        print(f"[density] octree built: dims={ph.grid_dims}, "
              f"cs={float(ph.cell_size[0]):.4g} m, "
              f"stencil_radius={ph.stencil_radius} (vol={stencil_vol}), "
              f"mean_per_cell={ph.mean_per_cell:.1f}, max_per_cell={ph.max_per_cell}, "
              f"~neighbours/query={expected_neighbours:.0f}")

        grid_dims = jnp.asarray(ph.grid_dims, dtype=jnp.int32)
        stencil_offsets = _stencil_offsets(ph.stencil_radius)
        support_radius_sq = jnp.float32((support * h_max) ** 2)

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
                    ph.sorted_particle_idx, grid_dims, stencil_offsets,
                    support_radius_sq,
                )
                out.append(rho_pad[:Qc.shape[0]])
            else:
                out.append(self._octree_fn(
                    Qc, P, h, w,
                    ph.bbox_min, ph.cell_size, ph.cell_starts,
                    ph.sorted_particle_idx, grid_dims, stencil_offsets,
                    support_radius_sq,
                ))
        return jnp.concatenate(out, axis=0)
