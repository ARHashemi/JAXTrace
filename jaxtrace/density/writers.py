# jaxtrace/density/writers.py
"""
Density field writers.

Two formats are supported, selected via :class:`DensityWriterConfig.format`:

  - ``"vti"``:    one VTK XML ImageData (.vti) per step + a ParaView .pvd index.
  - ``"vtkhdf"``: a single VTKHDF ImageData (.vtkhdf) transient file with
                  per-step PointData arrays appended.

The writer runs on a background thread, mirroring
:class:`jaxtrace.io.vtkhdf_writer.VTKHDFExportThread`. The main loop never
blocks on disk: ``enqueue(step, time, rho_3d)`` is non-blocking up to the
queue limit; back-pressure kicks in only if disk can't keep up.

Time-averaged outputs (mean_density, coverage_fraction, peak_density,
peak_time) are written synchronously at run end via :func:`write_time_average`.
"""

from __future__ import annotations

import queue
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from .grid import VoxelGrid


Format = Literal["vti", "vtkhdf"]

# Compression options for VTKHDF PointData.
#
# Default = "gzip" because that is the only filter ParaView's bundled
# vtkhdf5 has linked in (LZF and blosc are h5py-side / hdf5plugin-side
# filters that VTK's reader cannot decompress; trying to open an LZF or
# blosc-compressed VTKHDF file in ParaView fails with
# ``H5Dread start: ... count: ... Error reading array``).
#
# Use "lzf" only if you read the file back exclusively via h5py / a custom
# HDF5 install with the LZF filter registered. Use "blosc" (via hdf5plugin)
# similarly, when the consumer has the blosc plugin loaded. Use "none" for
# fastest writer at the cost of larger files (~3x bigger).
CompressionName = Literal["gzip", "lzf", "blosc", "none"]


def _resolve_compression(
    name: CompressionName | str | None,
    opts: int = 1,
    blosc_threads: int = 4,
) -> dict:
    """
    Return the kwargs dict to pass to h5py.create_dataset(...) for the chosen
    compression mode. The returned dict can be **-unpacked into create_dataset
    and always contains a subset of {compression, compression_opts}.

    Falls back to lzf (with a printed warning) if blosc is requested but
    hdf5plugin is not available.
    """
    if name is None or name == "none":
        return {}
    if name == "gzip":
        return {"compression": "gzip", "compression_opts": int(opts)}
    if name == "lzf":
        return {"compression": "lzf"}
    if name == "blosc":
        try:
            import hdf5plugin  # type: ignore
            # blosc:zstd is the best speed/ratio sweet spot with multi-thread.
            # See: https://github.com/silx-kit/hdf5plugin
            return {
                **hdf5plugin.Blosc(
                    cname="zstd",
                    clevel=max(1, min(int(opts), 9)),
                    shuffle=hdf5plugin.Blosc.SHUFFLE,
                ),
            }
        except Exception as e:
            print(f"[density] blosc compression unavailable ({e}); falling back to lzf")
            return {"compression": "lzf"}
    raise ValueError(
        f"unknown compression {name!r} (expected one of gzip|lzf|blosc|none)"
    )


@dataclass
class DensityWriterConfig:
    output_dir: Path
    format: Format = "vtkhdf"
    filename_stem: str = "density"
    queue_size: int = 64
    # Compression: gzip|lzf|blosc|none. Default lzf — fastest single-thread.
    compression: str | None = "gzip"
    compression_opts: int = 1
    blosc_threads: int = 4


# -----------------------------------------------------------------------------
# Synchronous VTI helpers
# -----------------------------------------------------------------------------

def _vti_write_step(
    path: Path,
    grid: VoxelGrid,
    arrays: Dict[str, np.ndarray],   # name -> (Nx, Ny, Nz) float32
) -> None:
    """Write a single ImageData (.vti) file with one or more PointData arrays."""
    try:
        import vtk  # type: ignore
        from vtk.util.numpy_support import numpy_to_vtk  # type: ignore
    except Exception as e:
        raise ImportError(f"vtk not available for VTI write: {e}")

    nx, ny, nz = grid.resolution
    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetOrigin(float(grid.origin[0]), float(grid.origin[1]), float(grid.origin[2]))
    img.SetSpacing(float(grid.spacing[0]), float(grid.spacing[1]), float(grid.spacing[2]))

    for name, arr in arrays.items():
        if arr.shape != (nx, ny, nz):
            raise ValueError(f"array {name!r} shape {arr.shape} != grid shape {(nx, ny, nz)}")
        # VTI expects flat (Nx*Ny*Nz,) with x fastest. Our grids are (i,j,k) -> flatten with x fastest:
        # Using order="F" so axis 0 (i/x) varies fastest in memory.
        flat = np.ascontiguousarray(arr, dtype=np.float32).flatten(order="F")
        vtk_arr = numpy_to_vtk(flat, deep=True)
        vtk_arr.SetName(name)
        img.GetPointData().AddArray(vtk_arr)

    # Set the first scalar array as the active one.
    if arrays:
        img.GetPointData().SetActiveScalars(next(iter(arrays.keys())))

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(img)
    writer.SetCompressorTypeToZLib()
    writer.Write()


def _pvd_write(pvd_path: Path, entries: list[Tuple[float, str]]) -> None:
    """Write a ParaView .pvd index pointing at the per-step .vti files."""
    root = ET.Element(
        "VTKFile",
        attrib={"type": "Collection", "version": "0.1", "byte_order": "LittleEndian"},
    )
    coll = ET.SubElement(root, "Collection")
    for time_val, filename in entries:
        ET.SubElement(
            coll, "DataSet",
            attrib={"timestep": f"{time_val:.9g}", "group": "", "part": "0", "file": filename},
        )
    tree = ET.ElementTree(root)
    pvd_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(pvd_path, xml_declaration=True, encoding="UTF-8")


# -----------------------------------------------------------------------------
# Synchronous VTKHDF ImageData helpers
# -----------------------------------------------------------------------------

def _to_zyx(arr: np.ndarray, resolution: Tuple[int, int, int]) -> np.ndarray:
    """
    Reshape a per-voxel array of shape ``(Nx, Ny, Nz)`` to VTK's expected
    ``(Nz, Ny, Nx)`` PointData layout (slowest-varying axis first, x fastest).

    Callers always pass the runner's (Nx, Ny, Nz) layout; we unconditionally
    transpose. (An earlier heuristic accepted both shapes via shape equality,
    but that silently no-ops on a cubic grid where Nx==Ny==Nz, swapping the
    x and z axes in the on-disk file.)
    """
    nx, ny, nz = resolution
    if arr.shape != (nx, ny, nz):
        raise ValueError(
            f"expected (Nx, Ny, Nz) = {(nx, ny, nz)}, got {arr.shape}"
        )
    return np.ascontiguousarray(np.transpose(arr, (2, 1, 0)), dtype=np.float32)


class _TransientImageDataWriter:
    """
    Minimal VTKHDF ImageData transient writer. Each PointData array is stored
    as a 3-D dataset of shape ``(NSteps * Nz, Ny, Nx)``; the leading axis is
    sliced by ``Steps/PointDataOffsets/<name>`` to recover each step's
    ``(Nz, Ny, Nx)`` block.

    Layout matches what VTK 9.4+'s ``vtkHDFReader`` expects for transient
    ImageData; the reader emits ``Expecting ndims >= 4`` if any field is
    stored as a flat 1-D array. We use the v110 file format envelope, same
    as the PolyData writer, for ParaView compatibility.
    """

    def __init__(
        self,
        path: Path,
        grid: VoxelGrid,
        compression: str | None = "gzip",
        compression_opts: int = 1,
        blosc_threads: int = 4,
    ) -> None:
        import h5py

        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._h5py = h5py
        self._comp_kwargs = _resolve_compression(compression, compression_opts, blosc_threads)
        self.n_voxels = grid.n_voxels
        self._resolution = grid.resolution

        self._file = h5py.File(str(path), "w", libver=("earliest", "v110"))
        root = self._file.create_group("VTKHDF")
        root.attrs["Version"] = np.array([2, 0], dtype="i8")
        root.attrs.create("Type", b"ImageData", dtype=h5py.string_dtype("ascii", 9))

        # ImageData geometry
        # WholeExtent: (xmin, xmax, ymin, ymax, zmin, zmax) in *point* indices
        nx, ny, nz = grid.resolution
        root.attrs["WholeExtent"] = np.array([0, nx - 1, 0, ny - 1, 0, nz - 1], dtype="i8")
        root.attrs["Origin"] = np.asarray(grid.origin, dtype="f8")
        root.attrs["Spacing"] = np.asarray(grid.spacing, dtype="f8")
        root.attrs["Direction"] = np.eye(3, dtype="f8").ravel()

        self._point_data = root.create_group("PointData")

        steps = root.create_group("Steps")
        steps.attrs.create("NSteps", 0, dtype="i8")
        steps.create_dataset("Values", (0,), maxshape=(None,), dtype="f8")
        steps.create_dataset("PartOffsets", (0,), maxshape=(None,), dtype="i8")
        steps.create_dataset("NumberOfParts", (0,), maxshape=(None,), dtype="i8")
        self._point_data_offsets = steps.create_group("PointDataOffsets")

        self._root = root
        self._steps = steps
        self._n_steps = 0

    def _ensure_field(self, name: str):
        """Get-or-create the (data, offsets) pair for a PointData field.

        For transient ImageData, VTK's ``vtkHDFReader`` (≥9.4) requires each
        PointData array to be a 4-D dataset shaped ``(NSteps, Nz, Ny, Nx)``.
        A 3-D shape like ``(NSteps*Nz, Ny, Nx)`` triggers
        ``Expecting ndims >= 4, got: 3``. Each per-step block is one
        ``(Nz, Ny, Nx)`` slab along the leading (time) axis.
        """
        if name in self._point_data:
            return self._point_data[name], self._point_data_offsets[name]
        nx, ny, nz = self._resolution
        # Chunk = one full step. Cap at ~1 MiB if a single (Nz,Ny,Nx) slab is
        # larger than that, by reducing Nz per chunk to keep h5py happy.
        slab_bytes = nz * ny * nx * 4
        if slab_bytes <= (1 << 20):
            chunk_nz = nz
        else:
            chunk_nz = max(1, (1 << 20) // max(ny * nx * 4, 1))
        ds = self._point_data.create_dataset(
            name, (0, nz, ny, nx),
            maxshape=(None, nz, ny, nx),
            dtype="f4",
            chunks=(1, chunk_nz, ny, nx),
            **self._comp_kwargs,
        )
        off = self._point_data_offsets.create_dataset(
            name, (0,), maxshape=(None,), dtype="i8",
        )
        return ds, off

    def write_step(self, time_value: float, arrays: Dict[str, np.ndarray]):
        for name, arr in arrays.items():
            block = _to_zyx(arr, self._resolution)         # (Nz, Ny, Nx)
            ds, off = self._ensure_field(name)
            # Append one (Nz, Ny, Nx) slab along the leading time axis.
            step_idx = ds.shape[0]
            ds.resize((step_idx + 1, ds.shape[1], ds.shape[2], ds.shape[3]))
            ds[step_idx, :, :, :] = block
            # Per-step offset: the step index itself (units = "steps").
            n_off = off.shape[0]
            off.resize((n_off + 1,))
            off[n_off] = step_idx

        # Steps bookkeeping
        n = self._n_steps
        for ds_name in ("Values", "PartOffsets", "NumberOfParts"):
            ds = self._steps[ds_name]
            ds.resize((n + 1,))
        self._steps["Values"][n] = float(time_value)
        self._steps["PartOffsets"][n] = 0
        self._steps["NumberOfParts"][n] = 1
        self._n_steps += 1
        self._steps.attrs["NSteps"] = self._n_steps

    def flush(self):
        self._file.flush()

    def close(self):
        try:
            self._file.flush()
        finally:
            self._file.close()


class _StaticImageDataWriter:
    """
    Steady (non-transient) VTKHDF ImageData writer for the time-averaged
    output. Each PointData array is a single ``(Nz, Ny, Nx)`` dataset. No
    ``Steps`` group is present, which is what the reader uses to distinguish
    transient from static files.
    """

    def __init__(
        self,
        path: Path,
        grid: VoxelGrid,
        compression: str | None = "gzip",
        compression_opts: int = 1,
        blosc_threads: int = 4,
    ) -> None:
        import h5py

        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._h5py = h5py
        self._comp_kwargs = _resolve_compression(compression, compression_opts, blosc_threads)
        self._resolution = grid.resolution

        self._file = h5py.File(str(path), "w", libver=("earliest", "v110"))
        root = self._file.create_group("VTKHDF")
        root.attrs["Version"] = np.array([2, 0], dtype="i8")
        root.attrs.create("Type", b"ImageData", dtype=h5py.string_dtype("ascii", 9))

        nx, ny, nz = grid.resolution
        root.attrs["WholeExtent"] = np.array([0, nx - 1, 0, ny - 1, 0, nz - 1], dtype="i8")
        root.attrs["Origin"] = np.asarray(grid.origin, dtype="f8")
        root.attrs["Spacing"] = np.asarray(grid.spacing, dtype="f8")
        root.attrs["Direction"] = np.eye(3, dtype="f8").ravel()

        self._point_data = root.create_group("PointData")
        self._root = root

    def write_fields(self, arrays: Dict[str, np.ndarray]):
        for name, arr in arrays.items():
            block = _to_zyx(arr, self._resolution)        # (Nz, Ny, Nx)
            self._point_data.create_dataset(
                name, data=block, dtype="f4",
                **self._comp_kwargs,
            )

    def close(self):
        try:
            self._file.flush()
        finally:
            self._file.close()


# -----------------------------------------------------------------------------
# Background writer thread
# -----------------------------------------------------------------------------

class DensityWriterThread:
    """
    Background-thread density writer with the same surface as the existing
    :class:`VTKHDFExportThread`: ``start()``, ``enqueue(...)``, ``stop()``.

    When the format is ``"vti"`` it writes one .vti per step and a .pvd index
    at stop time. When the format is ``"vtkhdf"`` it appends to a single
    transient ImageData file.
    """

    _STOP = object()
    _FLUSH = object()

    def __init__(self, cfg: DensityWriterConfig, grid: VoxelGrid) -> None:
        self.cfg = cfg
        self.grid = grid
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=cfg.queue_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._n_written = 0

        # Format-specific state
        self._vti_entries: list[Tuple[float, str]] = []
        self._hdf_writer: Optional[_TransientImageDataWriter] = None
        if cfg.format == "vtkhdf":
            hdf_path = cfg.output_dir / f"{cfg.filename_stem}.vtkhdf"
            self._hdf_writer = _TransientImageDataWriter(
                hdf_path, grid,
                compression=cfg.compression,
                compression_opts=cfg.compression_opts,
                blosc_threads=cfg.blosc_threads,
            )

    def start(self) -> None:
        self._thread.start()

    @property
    def n_written(self) -> int:
        return self._n_written

    def enqueue(self, step: int, time_value: float, rho_3d) -> None:
        """Submit a per-step density field. Non-blocking up to queue_size.

        ``rho_3d`` may be a numpy array (already on host) or a JAX device
        array. In the latter case the device-to-host copy is performed on
        the writer thread so the main GPU pipeline is not blocked.
        """
        try:
            self._queue.put((step, time_value, rho_3d), timeout=30.0)
        except queue.Full:
            print(f"[density] queue full at step {step}, skipping")

    def flush_async(self) -> None:
        try:
            self._queue.put_nowait(self._FLUSH)
        except queue.Full:
            pass

    def stop(self) -> None:
        self._queue.put(self._STOP)
        self._stop_event.set()
        self._thread.join()
        if self.cfg.format == "vti":
            pvd_path = self.cfg.output_dir / f"{self.cfg.filename_stem}.pvd"
            _pvd_write(pvd_path, self._vti_entries)
        elif self.cfg.format == "vtkhdf" and self._hdf_writer is not None:
            self._hdf_writer.close()

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is self._STOP or item is None:
                break
            if item is self._FLUSH:
                if self._hdf_writer is not None:
                    try:
                        self._hdf_writer.flush()
                    except Exception as e:
                        print(f"[density] flush error: {e}")
                self._queue.task_done()
                continue
            step, time_value, rho_3d = item
            try:
                # Materialise to host on this thread. For a JAX device array
                # this is the implicit ``block_until_ready`` + DMA copy; the
                # main thread is not blocked because it already moved on.
                rho_host = np.asarray(rho_3d)
                if self.cfg.format == "vti":
                    fname = f"{self.cfg.filename_stem}_{step:06d}.vti"
                    _vti_write_step(self.cfg.output_dir / fname, self.grid, {"density": rho_host})
                    self._vti_entries.append((float(time_value), fname))
                elif self.cfg.format == "vtkhdf" and self._hdf_writer is not None:
                    self._hdf_writer.write_step(time_value, {"density": rho_host})
                self._n_written += 1
            except Exception as e:
                print(f"[density] write error at step {step}: {e}")
            finally:
                self._queue.task_done()


# -----------------------------------------------------------------------------
# Time-averaged output
# -----------------------------------------------------------------------------

def write_time_average(
    output_dir: Path,
    grid: VoxelGrid,
    fields: Dict[str, np.ndarray],
    *,
    fmt: Format = "vtkhdf",
    filename_stem: str = "density_time_average",
    compression: str | None = "gzip",
    compression_opts: int = 1,
    blosc_threads: int = 4,
) -> Path:
    """
    Write the finalized time-average fields as a single ImageData file.

    ``fields`` keys are arbitrary names; each value must be (Nx, Ny, Nz).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = grid.resolution
    fields_3d = {}
    for name, arr in fields.items():
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 1:
            a = a.reshape((nx, ny, nz))
        if a.shape != (nx, ny, nz):
            raise ValueError(f"field {name!r} shape {a.shape} != grid {(nx, ny, nz)}")
        fields_3d[name] = a

    if fmt == "vti":
        path = output_dir / f"{filename_stem}.vti"
        _vti_write_step(path, grid, fields_3d)
        return path

    if fmt == "vtkhdf":
        path = output_dir / f"{filename_stem}.vtkhdf"
        writer = _StaticImageDataWriter(
            path, grid,
            compression=compression,
            compression_opts=compression_opts,
            blosc_threads=blosc_threads,
        )
        writer.write_fields(fields_3d)
        writer.close()
        return path

    raise ValueError(f"unknown format {fmt!r}")
