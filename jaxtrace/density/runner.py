# jaxtrace/density/runner.py
"""
High-level glue class that ties together:

  - voxel grid construction
  - inside-mesh masking (optional)
  - per-step density evaluation on the masked grid
  - per-particle density samples (optional)
  - time accumulation (optional)
  - background writer for per-step files
  - finalization (time-averaged file)

This is the single object both ``run_tracking.py`` and the offline CLI
instantiate. The contract is:

  * ``__init__`` does *all* the up-front setup (grid build, mask, estimator
    compile / warmup).
  * ``step(positions_device, dt, time_value, step_index)`` is the per-step
    hook. It is safe to call inside the tracking loop. When the density
    feature is disabled at the higher level, this class is simply not
    instantiated, so there is zero overhead.
  * ``close()`` flushes any background writer and writes the time-average.

All the heavy work is in JAX device kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from . import bandwidth, grid as grid_mod, inside_mesh, writers
from .estimator import DensityEstimator, EstimatorConfig
from .time_accumulator import TimeAccumulator


BoundsMode = Literal["mesh", "particles", "explicit", "prepass"]


@dataclass
class DensityRunnerConfig:
    # --- grid -----------------------------------------------------------------
    bounds_mode: BoundsMode = "mesh"        # mesh | particles | explicit | prepass
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None
    resolution: Optional[Union[int, Tuple[int, int, int]]] = 128
    voxel_size: Optional[float] = None
    pad_fraction: float = 0.0

    # --- inside-mesh masking --------------------------------------------------
    mask_inside_mesh: bool = True

    # --- kernel / bandwidth ---------------------------------------------------
    kernel: str = "wendland_c2"
    bandwidth_mode: str = "fixed"            # fixed | scott | silverman | knn_adaptive
    bandwidth: Optional[float] = None        # for fixed mode
    bandwidth_factor: float = 2.0
    knn_k: int = 32
    knn_safety: float = 1.2

    # --- weighting ------------------------------------------------------------
    use_per_particle_mass: bool = False
    normalization: str = "pdf"               # pdf | mass | unnormalized

    # --- engine ---------------------------------------------------------------
    engine: str = "auto"                     # auto | brute | octree
    auto_threshold: float = 5e10
    brute_query_chunk: int = 8192
    octree_cells_per_dim: int = 64
    octree_max_neighbors: int = 256
    particle_bucket: int = 4096

    # --- outputs --------------------------------------------------------------
    eval_on_grid: bool = True
    eval_at_particles: bool = True
    write_per_step: bool = True
    write_time_average: bool = True
    output_format: str = "vtkhdf"            # vtkhdf | vti
    output_dir: str = "density_out"
    filename_stem: str = "density"
    queue_size: int = 64
    compression: str = "gzip"                 # gzip | lzf | blosc | none
    compression_opts: int = 1                 # gzip level (1-9) or blosc clevel
    blosc_threads: int = 4                    # blosc multi-thread count
    # If True, also return the grid density as a host numpy array from step().
    # Default False — saves a per-step device-to-host copy on the main thread;
    # the writer thread still copies its own host buffer for I/O.
    return_grid_to_host: bool = False

    # --- bandwidth refresh ----------------------------------------------------
    bandwidth_refresh_every: int = 0         # 0 = compute once; N = recompute every N steps


@dataclass
class DensityRunner:
    cfg: DensityRunnerConfig
    mesh_octree_gpu: Optional[object] = None  # MeshAlignedOctreeGPU, optional
    mesh_bbox_min: Optional[np.ndarray] = None  # used when bounds_mode == "mesh"
    mesh_bbox_max: Optional[np.ndarray] = None
    initial_positions: Optional[np.ndarray] = None  # for "particles" bbox or warmup

    voxel_grid: Optional[grid_mod.VoxelGrid] = field(init=False, default=None)
    active_indices: Optional[jnp.ndarray] = field(init=False, default=None)  # (M_active,) int32
    inside_mask_flat: Optional[jnp.ndarray] = field(init=False, default=None)  # (M,) bool
    query_points: Optional[jnp.ndarray] = field(init=False, default=None)
    estimator: Optional[DensityEstimator] = field(init=False, default=None)
    accumulator: Optional[TimeAccumulator] = field(init=False, default=None)
    writer: Optional[writers.DensityWriterThread] = field(init=False, default=None)
    _h_cache: Optional[jnp.ndarray] = field(init=False, default=None)
    _steps_since_h_refresh: int = field(init=False, default=0)

    def __post_init__(self):
        bb_min, bb_max = self._resolve_bbox()
        self.voxel_grid = grid_mod.make_voxel_grid(
            bb_min, bb_max,
            resolution=self.cfg.resolution,
            voxel_size=self.cfg.voxel_size,
            pad_fraction=self.cfg.pad_fraction,
        )

        # Build inside-mesh mask if requested and a mesh octree was provided.
        centers_flat = self.voxel_grid.centers_flat
        M = int(centers_flat.shape[0])
        if self.cfg.mask_inside_mesh and self.mesh_octree_gpu is not None:
            self.inside_mask_flat = inside_mesh.compute_inside_mesh_mask(
                centers_flat, self.mesh_octree_gpu,
            )
            mask_np = np.asarray(self.inside_mask_flat)
            self.active_indices = jnp.asarray(np.nonzero(mask_np)[0], dtype=jnp.int32)
        else:
            self.inside_mask_flat = jnp.ones((M,), dtype=jnp.bool_)
            self.active_indices = jnp.arange(M, dtype=jnp.int32)

        self.query_points = centers_flat[self.active_indices]

        est_cfg = EstimatorConfig(
            kernel=self.cfg.kernel,
            d=3,
            normalization=self.cfg.normalization,
            engine=self.cfg.engine,
            auto_threshold=self.cfg.auto_threshold,
            brute_query_chunk=self.cfg.brute_query_chunk,
            octree_cells_per_dim=self.cfg.octree_cells_per_dim,
            octree_max_neighbors=self.cfg.octree_max_neighbors,
            particle_bucket=self.cfg.particle_bucket,
        )
        self.estimator = DensityEstimator(cfg=est_cfg, query_points=self.query_points)

        if self.cfg.write_time_average:
            self.accumulator = TimeAccumulator(n_voxels=M)

        if self.cfg.write_per_step:
            wcfg = writers.DensityWriterConfig(
                output_dir=Path(self.cfg.output_dir),
                format=self.cfg.output_format,  # type: ignore[arg-type]
                filename_stem=self.cfg.filename_stem,
                queue_size=self.cfg.queue_size,
                compression=self.cfg.compression,
                compression_opts=self.cfg.compression_opts,
                blosc_threads=self.cfg.blosc_threads,
            )
            self.writer = writers.DensityWriterThread(wcfg, self.voxel_grid)
            self.writer.start()

    # -------------------------------------------------------------------------
    # Bbox resolution
    # -------------------------------------------------------------------------
    def _resolve_bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        if cfg.bounds_mode == "explicit":
            if cfg.bounds is None:
                raise ValueError("bounds_mode='explicit' requires bounds=...")
            bb = np.asarray(cfg.bounds, dtype=np.float32)
            return bb[:, 0].copy(), bb[:, 1].copy()
        if cfg.bounds_mode == "mesh":
            if self.mesh_bbox_min is None or self.mesh_bbox_max is None:
                raise ValueError("bounds_mode='mesh' needs mesh_bbox_min / mesh_bbox_max")
            return (np.asarray(self.mesh_bbox_min, dtype=np.float32),
                    np.asarray(self.mesh_bbox_max, dtype=np.float32))
        if cfg.bounds_mode == "particles":
            if self.initial_positions is None:
                raise ValueError("bounds_mode='particles' needs initial_positions")
            return grid_mod.positions_bbox(self.initial_positions)
        if cfg.bounds_mode == "prepass":
            # User must precompute and pass via bounds=...
            if cfg.bounds is None:
                raise ValueError(
                    "bounds_mode='prepass' requires bounds=... (precomputed by caller)"
                )
            bb = np.asarray(cfg.bounds, dtype=np.float32)
            return bb[:, 0].copy(), bb[:, 1].copy()
        raise ValueError(f"unknown bounds_mode {cfg.bounds_mode!r}")

    # -------------------------------------------------------------------------
    # Bandwidth handling
    # -------------------------------------------------------------------------
    def _compute_h(self, positions: jnp.ndarray) -> jnp.ndarray:
        # Reference voxel size for the bandwidth default. Use the largest axis
        # spacing rather than the smallest so the kernel always spans multiple
        # voxels even on anisotropic grids — picking min(spacing) on a strongly
        # anisotropic bbox (e.g. dz << dx, dy) leaves the kernel narrower than
        # the in-plane voxel size and produces an almost-empty density field.
        vs = float(np.max(self.voxel_grid.spacing))
        return bandwidth.resolve_bandwidth(
            positions,
            mode=self.cfg.bandwidth_mode,
            voxel_size=vs,
            fixed_h=self.cfg.bandwidth,
            bandwidth_factor=self.cfg.bandwidth_factor,
            knn_k=self.cfg.knn_k,
            knn_safety=self.cfg.knn_safety,
            d=3,
        )

    # -------------------------------------------------------------------------
    # Per-step API
    # -------------------------------------------------------------------------
    def step(
        self,
        positions: jnp.ndarray,        # (N, 3) device float32
        dt: float,
        time_value: float,
        step_index: int,
        weights: Optional[jnp.ndarray] = None,   # (N,) device, or None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Evaluate density for this step. Returns ``(rho_grid_3d, rho_particles)``,
        each as host numpy arrays or ``None`` if not requested.

        The grid result is shaped (Nx, Ny, Nz); masked voxels are filled with 0.
        """
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be (N,3), got {positions.shape}")

        # Bandwidth: compute or refresh.
        need_refresh = (
            self._h_cache is None
            or (self.cfg.bandwidth_refresh_every > 0
                and self._steps_since_h_refresh >= self.cfg.bandwidth_refresh_every)
        )
        if need_refresh:
            self._h_cache = self._compute_h(positions)
            self._steps_since_h_refresh = 0
        else:
            # Fixed-h shape is N-dependent only if particle count changes.
            if self._h_cache.shape[0] != positions.shape[0]:
                self._h_cache = self._compute_h(positions)
        self._steps_since_h_refresh += 1

        if weights is None:
            weights = jnp.ones((positions.shape[0],), dtype=jnp.float32)

        # Grid evaluation. We deliberately keep ``rho_full`` as a device array
        # all the way through accumulation + writer enqueue. The writer thread
        # calls ``np.asarray`` itself, so the device-to-host copy happens off
        # the GPU's critical path and the main thread can immediately start
        # the next step.
        rho_grid_device: Optional[jnp.ndarray] = None
        if self.cfg.eval_on_grid:
            rho_active = self.estimator.evaluate(
                positions, self._h_cache, weights, query_points=self.query_points,
            )
            rho_full = jnp.zeros((self.voxel_grid.n_voxels,), dtype=jnp.float32)
            rho_full = rho_full.at[self.active_indices].set(rho_active)
            rho_grid_device = rho_full.reshape(self.voxel_grid.resolution)

            if self.accumulator is not None:
                # In-place GPU update; no host sync.
                self.accumulator.update(rho_full, dt=dt, t=time_value)

            if self.writer is not None:
                # Enqueue the device array; writer thread does the copy.
                self.writer.enqueue(step_index, time_value, rho_grid_device)

        # Per-particle evaluation. We pull this to host because callers (e.g.
        # run_tracking.py) feed it into a numpy extra_scalars dict for the
        # particles export. It is itself a small array (N float32), so the
        # device-to-host cost is negligible compared to grid eval.
        rho_part_np: Optional[np.ndarray] = None
        if self.cfg.eval_at_particles:
            rho_part = self.estimator.evaluate(
                positions, self._h_cache, weights, query_points=positions,
            )
            rho_part_np = np.asarray(rho_part)

        # Return shapes match the prior contract: 3D numpy (Nx,Ny,Nz) or None.
        # We materialise the grid result only if the caller asked for it.
        rho_grid_3d_np = (
            np.asarray(rho_grid_device).reshape(self.voxel_grid.resolution)
            if rho_grid_device is not None and self.cfg.return_grid_to_host
            else None
        )
        return rho_grid_3d_np, rho_part_np

    # -------------------------------------------------------------------------
    # Finalization
    # -------------------------------------------------------------------------
    def close(self) -> None:
        if self.writer is not None:
            self.writer.stop()

        if self.accumulator is not None and self.cfg.write_time_average:
            fields = self.accumulator.finalize()
            # Drop scalars; keep per-voxel fields
            voxel_fields = {
                k: v for k, v in fields.items()
                if isinstance(v, np.ndarray) and v.ndim == 1 and v.shape[0] == self.voxel_grid.n_voxels
            }
            writers.write_time_average(
                Path(self.cfg.output_dir), self.voxel_grid, voxel_fields,
                fmt=self.cfg.output_format,
                filename_stem=f"{self.cfg.filename_stem}_time_average",
                compression=self.cfg.compression,
                compression_opts=self.cfg.compression_opts,
                blosc_threads=self.cfg.blosc_threads,
            )
