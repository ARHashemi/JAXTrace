"""Transient VTKHDF PolyData writer for particle trajectories.

A single ``.vtkhdf`` file (HDF5 under the hood) holds the full time series
for one tracking run. Per step, the writer appends:

  * the point coordinates for that step into ``/VTKHDF/Points``,
  * each per-particle scalar/vector field into ``/VTKHDF/PointData/<name>``,
  * one entry per step in the ``/VTKHDF/Steps`` offset structures.

Topology is trivial (one vertex cell per particle), and the connectivity
array is **identical** every step when the particle count is constant.
Following the official VTKHDF guidance for static topology, we write the
``Vertices`` arrays **once** and emit zero deltas into
``CellOffsets`` / ``ConnectivityIdOffsets`` for every step — readers
interpret a zero delta as "reuse the previously written topology", so
the file stays compact regardless of step count.

The class :class:`VTKHDFExportThread` is a drop-in replacement for the
existing :class:`VTKExportThread` used in :mod:`benchmark_femuss_comparison`:
same ``start()`` / ``enqueue_export(step, positions, particle_ids,
element_ids, extra_scalars)`` / ``stop()`` interface, but it writes one
``.vtkhdf`` archive instead of one VTU per step.

ParaView ≥ 6.0 / VTK ≥ 9.4 read these files natively as time-varying
point clouds.

References
----------
* `VTKHDF format specification
  <https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/vtkhdf_specifications.html>`_
* `How to write time-dependent data in VTKHDF files
  <https://www.kitware.com/how-to-write-time-dependent-data-in-vtkhdf-files/>`_
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "h5py is required for VTKHDF export. Install with `pip install h5py`."
    ) from e


# Default chunk in particles. Tuned for ~100k-300k particles per step:
# one chunk per step keeps per-step reads contiguous in Paraview while
# avoiding pathologically small HDF5 chunks for low particle counts.
_MIN_CHUNK = 4_096


def _chunk_for(n_particles: int) -> int:
    """Pick an HDF5 chunk size that is one full step's worth of data,
    but at least 4 K elements so very small runs don't pay overhead."""
    return max(_MIN_CHUNK, n_particles)


def _append(dset: "h5py.Dataset", values: np.ndarray) -> None:
    """Resize a 1- or 2-D extensible dataset and write ``values`` at the end."""
    old = dset.shape[0]
    new = old + values.shape[0]
    dset.resize(new, axis=0)
    dset[old:new] = values


class TransientPolyDataWriter:
    """Synchronous writer. Use :class:`VTKHDFExportThread` for the
    background-thread interface that mirrors :class:`VTKExportThread`."""

    def __init__(
        self,
        output_path: Path,
        n_particles_hint: int = 0,
        compression: str | None = "gzip",
        compression_opts: int = 1,
        flush_interval: int = 1,
    ) -> None:
        """Open the output file and initialise the VTKHDF skeleton.

        Parameters
        ----------
        output_path
            File path (``.vtkhdf`` extension recommended). Parent directory
            is created if missing.
        n_particles_hint
            Expected particle count. Used to size the HDF5 chunk; if zero,
            chunks fall back to ``_MIN_CHUNK`` and resize when the first
            step is written.
        compression
            HDF5 compression filter for field data. ``"gzip"`` (default) or
            ``"lzf"``. Pass ``None`` to disable. Compression is applied to
            point data only — ``Points`` itself is left uncompressed because
            float32 coordinates compress poorly and the kernel writes them
            on every step.
        compression_opts
            Level for gzip (1–9). Ignored for lzf.
        flush_interval
            Call ``H5Fflush`` after every Nth step (default 1, i.e. flush
            on every step). HDF5 metadata (dataset extents, the ``NSteps``
            attribute, field offsets) lives in a process-local cache until
            close; periodic flushes make the file consistent on disk so a
            SIGTERM-killed job (e.g. SLURM timeout) leaves a recoverable
            archive containing every committed step. The cost is one
            metadata flush per N steps — negligible compared to chunk
            writes themselves. Set to 0 to disable periodic flushing
            (close-time flush only — risky on shared HPC queues).

            For the use case of "I want every committed step on disk even
            if the run dies", keep this at 1. For pure benchmark runs where
            metadata-write latency dominates, bump it to 10 or 100.
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self._chunk = _chunk_for(n_particles_hint)
        self._comp = compression
        self._comp_opts = compression_opts
        self._flush_interval = max(int(flush_interval), 0)
        self._steps_since_flush = 0

        # libver="latest" enables HDF5 1.10+ format features (required by
        # SWMR and by VTKHDF Version 2). It does not lock SWMR on; we'd
        # only call swmr_mode = True if a separate reader process were
        # tailing the file — but using "latest" makes the on-disk format
        # consistent with the schema we declare.
        self._file = h5py.File(str(self.output_path), "w", libver="latest")
        self._root = self._file.create_group("VTKHDF")
        self._root.attrs["Version"] = np.array([2, 0], dtype="i8")
        # The Type attribute must be a fixed-length ASCII string per the
        # VTKHDF spec; h5py's default str dtype would write variable-length.
        self._root.attrs.create(
            "Type", b"PolyData",
            dtype=h5py.string_dtype("ascii", 8),
        )

        # Per-step counts and concatenated point coordinates.
        self._root.create_dataset(
            "NumberOfPoints", (0,), maxshape=(None,), dtype="i8",
        )
        self._root.create_dataset(
            "Points", (0, 3), maxshape=(None, 3), dtype="f4",
            chunks=(self._chunk, 3),
        )

        # Vertices subgroup — written once for static topology.
        verts = self._root.create_group("Vertices")
        verts.create_dataset(
            "Connectivity", (0,), maxshape=(None,), dtype="i8",
            chunks=(self._chunk,),
        )
        verts.create_dataset(
            "Offsets", (0,), maxshape=(None,), dtype="i8",
            chunks=(self._chunk,),
        )
        verts.create_dataset(
            "NumberOfConnectivityIds", (0,), maxshape=(None,), dtype="i8",
        )
        verts.create_dataset(
            "NumberOfCells", (0,), maxshape=(None,), dtype="i8",
        )
        # The other PolyData topologies are required even when empty.
        for name in ("Lines", "Polygons", "Strips"):
            g = self._root.create_group(name)
            g.create_dataset("Connectivity", (0,), maxshape=(None,), dtype="i8")
            g.create_dataset("Offsets", (0,), maxshape=(None,), dtype="i8")
            g.create_dataset(
                "NumberOfConnectivityIds", (0,), maxshape=(None,), dtype="i8",
            )
            g.create_dataset(
                "NumberOfCells", (0,), maxshape=(None,), dtype="i8",
            )

        # PointData root group — datasets are lazily created on first write
        # so we don't need to know field names up front.
        self._point_data = self._root.create_group("PointData")

        # Steps subgroup.
        self._steps = self._root.create_group("Steps")
        self._steps.attrs.create("NSteps", 0, dtype="i8")
        self._steps.create_dataset(
            "Values", (0,), maxshape=(None,), dtype="f8",
        )
        self._steps.create_dataset(
            "PointOffsets", (0,), maxshape=(None,), dtype="i8",
        )
        self._steps.create_dataset(
            "CellOffsets", (0, 4), maxshape=(None, 4), dtype="i8",
        )
        self._steps.create_dataset(
            "ConnectivityIdOffsets", (0, 4), maxshape=(None, 4), dtype="i8",
        )
        self._steps.create_dataset(
            "PartOffsets", (0,), maxshape=(None,), dtype="i8",
        )
        self._steps.create_dataset(
            "NumberOfParts", (0,), maxshape=(None,), dtype="i8",
        )
        self._point_data_offsets = self._steps.create_group("PointDataOffsets")

        self._topology_written = False
        self._n_field_arrays: dict[str, int] = {}  # name -> running length

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_topology_payload(self, n_particles: int) -> None:
        """Write the trivial vertex-cell ``Connectivity`` / ``Offsets`` arrays
        once. ``NumberOfCells`` and ``NumberOfConnectivityIds`` are NOT touched
        here — those grow with one entry per step (handled in ``write_step``).

        With ``CellOffsets[step, *] == 0`` and ``ConnectivityIdOffsets[step,
        *] == 0`` every step, the reader reuses the connectivity/offsets
        arrays written here for every step (static-topology trick from the
        VTKHDF transient documentation).
        """
        if self._topology_written:
            return
        verts = self._root["Vertices"]
        connectivity = np.arange(n_particles, dtype="i8")
        offsets = np.arange(n_particles + 1, dtype="i8")
        _append(verts["Connectivity"], connectivity)
        _append(verts["Offsets"], offsets)
        self._topology_written = True

    def _append_topology_counts(self, n_particles: int) -> None:
        """Append one row per step to every topology group's ``NumberOfCells``
        and ``NumberOfConnectivityIds``. ``Vertices`` gets the real count; the
        other three (``Lines``, ``Polygons``, ``Strips``) get zero. VTK 9.4+
        expects these arrays to have shape ``(NPartitions * NSteps,)``.

        For empty topologies (Lines/Polygons/Strips) we also keep ``Offsets``
        non-empty: VTK's reader dereferences ``Offsets[i+1] - Offsets[i]`` to
        determine the per-step cell count; with zero ``NumberOfCells`` we
        just append a trailing zero to keep that slice valid.
        """
        for name in ("Lines", "Polygons", "Strips"):
            g = self._root[name]
            _append(g["NumberOfCells"], np.array([0], dtype="i8"))
            _append(g["NumberOfConnectivityIds"], np.array([0], dtype="i8"))
            # Initialise Offsets with a single zero on first use, then leave
            # it alone — every step's CellOffsets[*, this_topology] is 0
            # (static-topology trick), so the reader uses Offsets[0:1].
            if g["Offsets"].shape[0] == 0:
                _append(g["Offsets"], np.array([0], dtype="i8"))
        verts = self._root["Vertices"]
        _append(verts["NumberOfCells"], np.array([n_particles], dtype="i8"))
        _append(verts["NumberOfConnectivityIds"], np.array([n_particles], dtype="i8"))

    def _field_dataset(
        self, name: str, sample: np.ndarray, n_prior_steps: int, n_particles: int,
    ) -> "h5py.Dataset":
        """Get-or-create the per-step PointData dataset for ``name``.

        On creation, the dataset is sized to hold ``n_prior_steps`` zero-filled
        steps so its length stays aligned with ``Steps/Values`` for fields that
        appear mid-run. The companion ``Steps/PointDataOffsets/<name>`` array
        is similarly pre-populated with zeros pointing at the leading padding
        region (any value < dset length is a valid offset since the data is
        all-zeros there)."""
        if name in self._point_data:
            return self._point_data[name]

        if sample.ndim == 1:
            shape: tuple[int, ...] = (0,)
            maxshape: tuple[int | None, ...] = (None,)
            chunks: tuple[int, ...] = (self._chunk,)
            pad_shape: tuple[int, ...] = (n_prior_steps * n_particles,)
        elif sample.ndim == 2:
            comps = sample.shape[1]
            shape = (0, comps)
            maxshape = (None, comps)
            chunks = (self._chunk, comps)
            pad_shape = (n_prior_steps * n_particles, comps)
        else:
            raise ValueError(
                f"PointData '{name}' must be 1-D or 2-D, got shape {sample.shape}"
            )

        # Numeric dtype mapping: keep int32/uint8/float32 native; promote
        # int64 -> int32 to match VTK's default integer width for PointData.
        if sample.dtype == np.bool_:
            dtype = "u1"
        elif sample.dtype == np.uint8:
            dtype = "u1"
        elif sample.dtype in (np.int32, np.int64):
            dtype = "i4"
        elif sample.dtype == np.float64:
            dtype = "f8"
        else:
            dtype = "f4"

        kwargs: dict[str, Any] = dict(chunks=chunks, dtype=dtype)
        if self._comp is not None:
            kwargs["compression"] = self._comp
            if self._comp == "gzip":
                kwargs["compression_opts"] = self._comp_opts

        dset = self._point_data.create_dataset(
            name, shape, maxshape=maxshape, **kwargs,
        )
        self._point_data_offsets.create_dataset(
            name, (0,), maxshape=(None,), dtype="i8",
        )
        if n_prior_steps > 0:
            # Pad the data with zeros for the steps that were committed before
            # this field existed.
            _append(dset, np.zeros(pad_shape, dtype=dtype))
            # All prior offsets point to 0 — the leading padding is valid data.
            _append(
                self._point_data_offsets[name],
                np.zeros(n_prior_steps, dtype="i8"),
            )
        return dset

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_step(
        self,
        step: int,
        positions: np.ndarray,
        particle_ids: np.ndarray | None = None,
        element_ids: np.ndarray | None = None,
        extra_scalars: dict[str, np.ndarray] | None = None,
        time_value: float | None = None,
    ) -> None:
        """Append one timestep.

        Parameters
        ----------
        step
            Integer step index. Stored as the time value when
            ``time_value`` is not provided.
        positions
            ``(N, 3)`` float array of particle coordinates.
        particle_ids, element_ids
            Optional ``(N,)`` int arrays. Written as ``ParticleID`` and
            ``ElementID`` respectively.
        extra_scalars
            Optional mapping ``{name: array}`` of additional per-particle
            fields. Arrays may be 1-D ``(N,)`` or 2-D ``(N, C)``.
        time_value
            Optional physical time value; defaults to ``float(step)``.
        """
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be (N,3), got {positions.shape}")
        n = positions.shape[0]

        self._ensure_topology_payload(n)
        self._append_topology_counts(n)

        # --- Steps bookkeeping ---
        steps_grp = self._steps
        # NSteps BEFORE this step has been committed; the field-creation
        # backfill in _field_dataset uses this as the number of zero-padded
        # prior steps to emit.
        n_prior_steps = int(steps_grp.attrs["NSteps"])
        point_offset = int(self._root["Points"].shape[0])

        # Per-step geometry appends.
        _append(self._root["NumberOfPoints"], np.array([n], dtype="i8"))
        _append(self._root["Points"], positions)

        # Per-step Steps metadata.
        _append(steps_grp["Values"],
                np.array([float(step) if time_value is None else time_value],
                         dtype="f8"))
        _append(steps_grp["PointOffsets"],
                np.array([point_offset], dtype="i8"))
        # Static topology: zero deltas mean "reuse what was already written".
        _append(steps_grp["CellOffsets"],
                np.zeros((1, 4), dtype="i8"))
        _append(steps_grp["ConnectivityIdOffsets"],
                np.zeros((1, 4), dtype="i8"))
        _append(steps_grp["PartOffsets"], np.array([0], dtype="i8"))
        _append(steps_grp["NumberOfParts"], np.array([1], dtype="i8"))

        # Commit the step. Field writes below leave the dataset lengths at
        # exactly (NSteps * n) rows for every PointData array.
        steps_grp.attrs.modify("NSteps", np.int64(n_prior_steps + 1))

        # --- Point data writes ---
        provided: dict[str, np.ndarray] = {}
        if particle_ids is not None:
            provided["ParticleID"] = np.asarray(particle_ids, dtype=np.int32)
        if element_ids is not None:
            provided["ElementID"] = np.asarray(element_ids, dtype=np.int32)
        if extra_scalars:
            for k, v in extra_scalars.items():
                provided[k] = np.asarray(v)

        def _write_named_field(name: str, arr: np.ndarray) -> None:
            if arr.ndim not in (1, 2) or arr.shape[0] != n:
                raise ValueError(
                    f"Field '{name}' must be 1- or 2-D with leading dim "
                    f"n_particles={n}, got shape {arr.shape}"
                )
            arr = np.ascontiguousarray(arr)
            dset = self._field_dataset(
                name, arr,
                n_prior_steps=n_prior_steps,
                n_particles=n,
            )
            offset_dset = self._point_data_offsets[name]
            offset_value = int(dset.shape[0])
            _append(dset, arr)
            _append(offset_dset, np.array([offset_value], dtype="i8"))

        for name, arr in provided.items():
            _write_named_field(name, arr)

        # For PointData arrays created on a previous step but not provided
        # now, pad this step's slot with zeros so the dataset length stays
        # at NSteps * n.
        for name in list(self._point_data.keys()):
            if name in provided:
                continue
            dset = self._point_data[name]
            offset_dset = self._point_data_offsets[name]
            pad_shape = (n,) + dset.shape[1:]
            offset_value = int(dset.shape[0])
            _append(dset, np.zeros(pad_shape, dtype=dset.dtype))
            _append(offset_dset, np.array([offset_value], dtype="i8"))

        # Periodic flush: commits dataset extents, the NSteps attribute,
        # and all PointDataOffsets entries to disk so a SIGTERM-killed run
        # leaves a recoverable file with every committed step intact.
        self._steps_since_flush += 1
        if self._flush_interval > 0 and self._steps_since_flush >= self._flush_interval:
            self.flush()

    def flush(self) -> None:
        """Force HDF5 to write all buffered metadata and chunks to disk.
        Safe to call from a signal handler — does not allocate."""
        if self._file:
            self._file.flush()
            self._steps_since_flush = 0

    def close(self) -> None:
        """Flush and close the file."""
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None  # type: ignore[assignment]

    def __enter__(self) -> "TransientPolyDataWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class VTKHDFExportThread:
    """Background-thread wrapper with the same surface as
    ``benchmark_femuss_comparison.VTKExportThread``.

    Construct, ``start()``, ``enqueue_export(...)`` per step, then
    ``stop()``. Internally serialises every queued step through a single
    :class:`TransientPolyDataWriter` so the on-disk file has steps in
    enqueue order.
    """

    def __init__(
        self,
        output_dir: Path,
        queue_size: int = 200,
        filename: str = "particles.vtkhdf",
        n_particles_hint: int = 0,
        compression: str | None = "gzip",
        compression_opts: int = 1,
        flush_interval: int = 1,
    ) -> None:
        """Spawn a background writer thread.

        Parameters
        ----------
        queue_size
            Maximum number of pending exports. The main loop blocks in
            ``enqueue_export`` once the queue is full; pick a value large
            enough that the writer drains fast enough to never fill up at
            the kernel's step rate. With 100-300k particles and gzip-1
            compression, ~30-50 ms per step is typical; a 200-deep queue
            absorbs ~10 s of GPU bursts before back-pressuring.
        flush_interval
            Forwarded to :class:`TransientPolyDataWriter`. Default 1
            (flush every step) so a SIGTERM-killed run leaves a complete
            file up to the last committed step.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / filename
        self.export_queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self.worker_thread = threading.Thread(
            target=self._worker, daemon=True,
        )
        self.stop_event = threading.Event()
        self.n_exported = 0
        self._writer = TransientPolyDataWriter(
            self.output_path,
            n_particles_hint=n_particles_hint,
            compression=compression,
            compression_opts=compression_opts,
            flush_interval=flush_interval,
        )

    def start(self) -> None:
        self.worker_thread.start()

    def enqueue_export(
        self,
        step: int,
        positions: np.ndarray,
        particle_ids: np.ndarray | None = None,
        element_ids: np.ndarray | None = None,
        extra_scalars: dict[str, np.ndarray] | None = None,
    ) -> None:
        try:
            self.export_queue.put(
                (step, positions, particle_ids, element_ids, extra_scalars),
                timeout=30.0,
            )
        except queue.Full:
            print(f"Warning: Export queue full at step {step}, skipping")

    # Sentinel objects placed on the queue to instruct the worker.
    _STOP = object()
    _FLUSH = object()

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                item = self.export_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is self._STOP or item is None:
                break
            if item is self._FLUSH:
                try:
                    self._writer.flush()
                except Exception as e:  # pragma: no cover
                    print(f"VTKHDF flush error: {e}")
                finally:
                    self.export_queue.task_done()
                continue
            step, positions, particle_ids, element_ids, extra_scalars = item
            try:
                self._writer.write_step(
                    step=step,
                    positions=positions,
                    particle_ids=particle_ids,
                    element_ids=element_ids,
                    extra_scalars=extra_scalars,
                )
                self.n_exported += 1
            except Exception as e:  # pragma: no cover - logging only
                print(f"VTKHDF export error at step {step}: {e}")
            finally:
                self.export_queue.task_done()

    def flush_async(self) -> None:
        """Request the writer to flush the file at its next opportunity
        (after any already-queued writes complete). Non-blocking; safe to
        call from a signal handler."""
        try:
            self.export_queue.put_nowait(self._FLUSH)
        except queue.Full:
            # Queue is full; the next stop() will flush via close() anyway.
            pass

    def stop(self) -> None:
        self.export_queue.put(self._STOP)
        self.stop_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=60.0)
        self._writer.close()
