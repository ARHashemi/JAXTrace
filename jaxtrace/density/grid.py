# jaxtrace/density/grid.py
"""
Uniform Cartesian voxel grid for density evaluation.

The grid is defined by a bounding box (xmin..xmax, ymin..ymax, zmin..zmax)
and a resolution (Nx, Ny, Nz). Voxel *centers* are at::

    origin = (xmin + dx/2, ymin + dy/2, zmin + dz/2)
    dx = (xmax - xmin) / Nx           (analogous for dy, dz)
    voxel_center[i, j, k] = origin + (i*dx, j*dy, k*dz)

We store the grid centers as a flat (M, 3) device array of size M = Nx*Ny*Nz.
The estimator masks "active" voxels (e.g. inside the fluid domain); the active
set is what's actually evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np


BBox = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]


@dataclass(frozen=True)
class VoxelGrid:
    bbox_min: np.ndarray         # (3,) float32, host
    bbox_max: np.ndarray         # (3,) float32, host
    resolution: Tuple[int, int, int]
    spacing: np.ndarray          # (3,) float32, host  (dx, dy, dz)
    origin: np.ndarray           # (3,) float32, host  (first voxel center)
    centers_flat: jnp.ndarray    # (M, 3) float32, device  (row-major i,j,k)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.resolution

    @property
    def n_voxels(self) -> int:
        return int(self.resolution[0] * self.resolution[1] * self.resolution[2])

    @property
    def voxel_volume(self) -> float:
        return float(self.spacing[0] * self.spacing[1] * self.spacing[2])

    def index_to_xyz(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Host arrays of voxel-center coordinates along each axis."""
        ox, oy, oz = self.origin
        dx, dy, dz = self.spacing
        nx, ny, nz = self.resolution
        x = ox + dx * np.arange(nx, dtype=np.float32)
        y = oy + dy * np.arange(ny, dtype=np.float32)
        z = oz + dz * np.arange(nz, dtype=np.float32)
        return x, y, z


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------

def make_voxel_grid(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    *,
    resolution: Optional[Union[int, Tuple[int, int, int]]] = None,
    voxel_size: Optional[Union[float, Tuple[float, float, float]]] = None,
    pad_fraction: float = 0.0,
) -> VoxelGrid:
    """
    Build a uniform voxel grid.

    Either ``resolution`` or ``voxel_size`` must be given. If both are given,
    ``voxel_size`` wins and resolution is recomputed.

    Both arguments accept either a scalar (isotropic) or a 3-tuple
    ``(x, y, z)`` (per-axis). A per-axis ``voxel_size`` lets you make the
    grid finer along one axis without changing the others; per-axis
    ``resolution`` is the same idea expressed as cell counts.

    ``pad_fraction`` enlarges the bbox by that fraction in each direction.
    """
    bbox_min = np.asarray(bbox_min, dtype=np.float32).reshape(3)
    bbox_max = np.asarray(bbox_max, dtype=np.float32).reshape(3)
    if np.any(bbox_max <= bbox_min):
        raise ValueError(f"bbox_max must be > bbox_min, got {bbox_min} .. {bbox_max}")

    if pad_fraction > 0.0:
        extent = bbox_max - bbox_min
        bbox_min = (bbox_min - pad_fraction * extent).astype(np.float32)
        bbox_max = (bbox_max + pad_fraction * extent).astype(np.float32)

    extent = bbox_max - bbox_min
    if voxel_size is not None:
        vs_arr = np.asarray(voxel_size, dtype=np.float32).reshape(-1)
        if vs_arr.size == 1:
            vs_per_axis = np.full((3,), float(vs_arr[0]), dtype=np.float32)
        elif vs_arr.size == 3:
            vs_per_axis = vs_arr.astype(np.float32)
        else:
            raise ValueError(
                f"voxel_size must be a scalar or a 3-vector, got shape {vs_arr.shape}"
            )
        nx, ny, nz = (max(1, int(np.ceil(e / vs))) for e, vs in zip(extent, vs_per_axis))
    elif resolution is not None:
        if isinstance(resolution, (int, np.integer)):
            nx = ny = nz = int(resolution)
        else:
            res_arr = np.asarray(resolution).reshape(-1)
            if res_arr.size == 1:
                nx = ny = nz = int(res_arr[0])
            elif res_arr.size == 3:
                nx, ny, nz = (int(v) for v in res_arr)
            else:
                raise ValueError(
                    f"resolution must be a scalar or a 3-vector, got shape {res_arr.shape}"
                )
    else:
        raise ValueError("provide either resolution or voxel_size")

    spacing = np.array([extent[0] / nx, extent[1] / ny, extent[2] / nz], dtype=np.float32)
    origin = (bbox_min + 0.5 * spacing).astype(np.float32)

    # Build flat (M, 3) on device using a JAX meshgrid. We use indexing="ij"
    # to guarantee row-major (i,j,k) ordering matching VTI's "Fortran" expectation
    # when we flatten with x fastest. We'll handle byte-order in the writer.
    xs = origin[0] + spacing[0] * jnp.arange(nx, dtype=jnp.float32)
    ys = origin[1] + spacing[1] * jnp.arange(ny, dtype=jnp.float32)
    zs = origin[2] + spacing[2] * jnp.arange(nz, dtype=jnp.float32)
    X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
    centers_flat = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(jnp.float32)

    return VoxelGrid(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        resolution=(int(nx), int(ny), int(nz)),
        spacing=spacing,
        origin=origin,
        centers_flat=centers_flat,
    )


# -----------------------------------------------------------------------------
# Bbox helpers
# -----------------------------------------------------------------------------

def bbox_union(
    a_min: np.ndarray, a_max: np.ndarray,
    b_min: np.ndarray, b_max: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    return (np.minimum(a_min, b_min).astype(np.float32),
            np.maximum(a_max, b_max).astype(np.float32))


def positions_bbox(positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bbox of (N,3) positions (host or device array)."""
    P = np.asarray(positions)
    return P.min(axis=0).astype(np.float32), P.max(axis=0).astype(np.float32)


def trajectory_bbox_union_from_vtkhdf(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-pass: scan a particles.vtkhdf file and return the union bbox over all
    timesteps. Reads positions in chunks; never holds the full trajectory in
    memory.
    """
    import h5py

    lo = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    hi = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    with h5py.File(path, "r") as f:
        pts = f["/VTKHDF/Points"]
        chunk = 1_000_000
        total = pts.shape[0]
        for s in range(0, total, chunk):
            e = min(s + chunk, total)
            block = pts[s:e]
            lo = np.minimum(lo, block.min(axis=0))
            hi = np.maximum(hi, block.max(axis=0))

    return lo.astype(np.float32), hi.astype(np.float32)


def iterate_vtkhdf_steps(
    path: str,
    *,
    step_indices=None,           # optional iterable of step indices to read
):
    """
    Yield ``(step_index, time_value, positions_np)`` per step from a
    particles.vtkhdf file written by :class:`TransientPolyDataWriter`.

    If ``step_indices`` is provided, only those steps are yielded (in order).
    ``positions_np`` is a numpy float32 array of shape (n_particles_at_step, 3).
    """
    import h5py

    with h5py.File(path, "r") as f:
        root = f["/VTKHDF"]
        n_steps = int(root["Steps"].attrs["NSteps"])
        n_points = root["NumberOfPoints"][:]            # (n_steps,)
        point_offsets = root["Steps/PointOffsets"][:]   # (n_steps,)
        times = root["Steps/Values"][:]                 # (n_steps,)
        pts = root["Points"]                            # (sum_npts, 3)

        idxs = range(n_steps) if step_indices is None else list(step_indices)
        for step in idxs:
            if step < 0 or step >= n_steps:
                continue
            start = int(point_offsets[step])
            count = int(n_points[step])
            block = pts[start:start + count]
            yield step, float(times[step]), np.asarray(block, dtype=np.float32)


def prefetch_vtkhdf_steps(
    path: str,
    *,
    step_indices=None,
    prefetch: int = 4,
):
    """
    Same yield contract as :func:`iterate_vtkhdf_steps`, but the next
    ``prefetch`` blocks are read on a background thread while the main
    thread is busy. Reading particles.vtkhdf is gzip-decompression-bound,
    so this typically eliminates the read stall between GPU steps.

    The reader thread holds the HDF5 file open for the duration of the
    iteration; it exits when the consumer exhausts the iterator or stops
    consuming.
    """
    import threading
    import queue as _queue

    q: "_queue.Queue" = _queue.Queue(maxsize=max(int(prefetch), 1))
    _SENTINEL = object()
    _stop = threading.Event()

    def _producer():
        try:
            for item in iterate_vtkhdf_steps(path, step_indices=step_indices):
                if _stop.is_set():
                    break
                q.put(item)
        except Exception as e:
            q.put(("__error__", e))
        finally:
            q.put(_SENTINEL)

    t = threading.Thread(target=_producer, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is _SENTINEL:
                return
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__error__":
                raise item[1]
            yield item
    finally:
        _stop.set()
        # drain so the producer can exit cleanly
        try:
            while True:
                q.get_nowait()
        except Exception:
            pass
        t.join(timeout=5.0)
