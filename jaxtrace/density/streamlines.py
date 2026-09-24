"""
Streamline-union output for a particle trajectory.

Each particle's full position-vs-time path is collected into a polyline,
yielding a static VTKHDF PolyData file whose ``Lines`` group contains one
line strip per particle (vertices in step order). ParaView renders this
as a line bundle that can be coloured by per-vertex Temperature or
MaxTemperature, animated along the line by VertexTime, or thresholded
per-line by ParticleID / seed group.

This output is independent of the dedup union: it is a Lagrangian
visualisation, not a spatial-snapshot one. Memory is bounded by sub-
sampling particles (``particle_stride``) and / or steps
(``step_stride``); typical 5 % × stride-20 over 4 k steps gives a
~50 MB file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class StreamlineResult:
    """
    Collected streamlines.

    Shapes:
      positions      -- (N_lines, N_vertices_per_line, 3) float32
      vertex_time    -- (N_lines, N_vertices_per_line) float32
      vertex_fields  -- dict[name] -> (N_lines, N_vertices_per_line, [C]) ...
      line_fields    -- dict[name] -> (N_lines,) ...
      particle_ids   -- (N_lines,) int  (ParticleID per line)
      step_indices   -- (N_vertices_per_line,) int  (selected step idx per vertex)
    """
    positions: np.ndarray
    vertex_time: np.ndarray
    vertex_fields: dict
    line_fields: dict
    particle_ids: np.ndarray
    step_indices: np.ndarray
    bbox_lo: np.ndarray
    bbox_hi: np.ndarray

    @property
    def n_lines(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_vertices_per_line(self) -> int:
        return int(self.positions.shape[1])


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------

def select_streamline_particles(
    seed_positions: np.ndarray,         # (N_seed, 3)
    seed_particle_ids: np.ndarray,      # (N_seed,) int
    particle_stride: int,
    *,
    roi_lo: Optional[np.ndarray] = None,
    roi_hi: Optional[np.ndarray] = None,
    roi_mode: str = "seed",
) -> np.ndarray:
    """
    Pick which particles to include in the streamline output.

    Returns a sorted ``(N_lines,)`` int64 array of ParticleIDs.

    ``particle_stride``
        Take every Nth particle from the (post-ROI) seed list. 1 = all.

    ``roi_mode``
        ``"seed"``   — only particles whose step-0 seed position falls in
        the ROI box. Their full trajectory is still written (they may
        wander outside later).
        ``"none"``   — ROI ignored for particle selection.
        ``"any"``    — not implemented at construction; treated as
        ``"none"`` because we do not yet know each particle's full
        trajectory.
    """
    seed_positions = np.asarray(seed_positions, dtype=np.float32)
    seed_particle_ids = np.asarray(seed_particle_ids, dtype=np.int64).ravel()
    if roi_mode == "seed" and roi_lo is not None and roi_hi is not None:
        mask = np.all(
            (seed_positions >= roi_lo) & (seed_positions <= roi_hi), axis=1,
        )
        ids_in_roi = seed_particle_ids[mask]
    else:
        ids_in_roi = seed_particle_ids
    stride = max(1, int(particle_stride))
    return np.sort(ids_in_roi[::stride])


def collect_streamlines_from_vtkhdf(
    particles_path: str,
    keep_particle_ids: np.ndarray,
    step_indices: Sequence[int],
    *,
    vertex_field_names: Sequence[str] = (),
) -> StreamlineResult:
    """
    Read selected steps from ``particles.vtkhdf`` and pack positions
    (and any requested vertex PointData fields) into per-line arrays.

    The trajectory is read step by step. For each step we look up the
    rows in ``ParticleID`` that match ``keep_particle_ids`` and copy
    their (position, fields) into the pre-allocated output. The lookup
    uses ``np.searchsorted`` against a sorted ID list, so a step with
    ~360 k particles and ~18 k kept IDs runs in O(N log K) per step.

    Returns a fully-materialised ``StreamlineResult``; memory cost is
    ``N_lines × N_steps_selected × (3 + extra_fields) × 4 bytes``.
    """
    from .grid import iterate_vtkhdf_steps
    keep_ids = np.asarray(keep_particle_ids, dtype=np.int64).ravel()
    n_lines = int(keep_ids.size)
    sel_steps = list(step_indices)
    n_vert = len(sel_steps)

    positions = np.empty((n_lines, n_vert, 3), dtype=np.float32)
    vertex_time = np.empty((n_lines, n_vert), dtype=np.float32)
    vertex_fields: dict[str, np.ndarray] = {}

    keep_sort = np.argsort(keep_ids)
    keep_ids_sorted = keep_ids[keep_sort]
    # `inv_keep_sort` maps "row in sorted unique keep" -> "row in output line array"
    # so a hit at sorted-rank r writes to the original-order slot.
    inv_keep_sort = np.argsort(keep_sort)

    fill_count = 0
    last_t = 0.0
    for vert_idx, item in enumerate(
        iterate_vtkhdf_steps(
            particles_path, step_indices=sel_steps,
            fields=list(vertex_field_names) + ["ParticleID"],
            return_fields=True,
        )
    ):
        step_idx, t, P, fdict = item
        last_t = float(t)
        pids = fdict.get("ParticleID")
        if pids is None:
            # No ParticleID in the trajectory; fall back to row-index map.
            # In that case the keep list is interpreted as row indices.
            row_idx = keep_ids
            valid = (row_idx >= 0) & (row_idx < P.shape[0])
            positions[:, vert_idx, :] = 0.0
            positions[valid, vert_idx, :] = P[row_idx[valid]]
            for fname in vertex_field_names:
                if fname not in fdict:
                    continue
                if fname not in vertex_fields:
                    vertex_fields[fname] = np.zeros(
                        (n_lines, n_vert), dtype=np.float32,
                    )
                vertex_fields[fname][valid, vert_idx] = fdict[fname][row_idx[valid]]
        else:
            pids = np.asarray(pids, dtype=np.int64).ravel()
            pos_in_step = np.searchsorted(pids, keep_ids_sorted)
            pos_clamped = np.clip(pos_in_step, 0, pids.size - 1)
            matched = pids[pos_clamped] == keep_ids_sorted
            # Reorder back to caller's ID order.
            matched_unsorted = matched[inv_keep_sort]
            rows = pos_clamped[inv_keep_sort]
            positions[:, vert_idx, :] = 0.0
            positions[matched_unsorted, vert_idx, :] = P[rows[matched_unsorted]]
            for fname in vertex_field_names:
                if fname not in fdict:
                    continue
                if fname not in vertex_fields:
                    vertex_fields[fname] = np.zeros(
                        (n_lines, n_vert), dtype=np.float32,
                    )
                vals = np.asarray(fdict[fname])
                if vals.ndim == 1:
                    vertex_fields[fname][matched_unsorted, vert_idx] = \
                        vals[rows[matched_unsorted]].astype(np.float32)
                else:
                    # Skip vector fields for now; ParaView line colormap
                    # uses scalars per vertex.
                    pass
        vertex_time[:, vert_idx] = float(t)
        fill_count += 1

    if fill_count == 0:
        return StreamlineResult(
            positions=np.zeros((0, 0, 3), dtype=np.float32),
            vertex_time=np.zeros((0, 0), dtype=np.float32),
            vertex_fields={}, line_fields={},
            particle_ids=keep_ids,
            step_indices=np.asarray(sel_steps, dtype=np.int64),
            bbox_lo=np.zeros(3, dtype=np.float32),
            bbox_hi=np.zeros(3, dtype=np.float32),
        )

    if positions.size:
        flat = positions.reshape(-1, 3)
        bbox_lo = flat.min(axis=0).astype(np.float32)
        bbox_hi = flat.max(axis=0).astype(np.float32)
    else:
        bbox_lo = np.zeros(3, dtype=np.float32)
        bbox_hi = np.zeros(3, dtype=np.float32)
    return StreamlineResult(
        positions=positions,
        vertex_time=vertex_time,
        vertex_fields=vertex_fields,
        line_fields={"ParticleID": keep_ids.astype(np.int32)},
        particle_ids=keep_ids,
        step_indices=np.asarray(sel_steps, dtype=np.int64),
        bbox_lo=bbox_lo,
        bbox_hi=bbox_hi,
    )


# -----------------------------------------------------------------------------
# VTKHDF PolyData writer (Lines variant)
# -----------------------------------------------------------------------------

def _hdf5_dtype_for(arr: np.ndarray) -> str:
    if arr.dtype == np.bool_ or arr.dtype == np.uint8:
        return "u1"
    if arr.dtype in (np.int32, np.int64):
        return "i4"
    if arr.dtype == np.float64:
        return "f8"
    return "f4"


def write_streamlines_vtkhdf(
    out_path: Path,
    result: StreamlineResult,
    *,
    compression: Optional[str] = "gzip",
    compression_opts: int = 1,
    file_attrs: Optional[dict] = None,
) -> Path:
    """
    Write the streamlines as a static PolyData VTKHDF file. Each particle
    becomes one line strip with ``n_vertices_per_line`` vertices in step
    order. Per-vertex PointData and per-line CellData (e.g. ParticleID)
    are written alongside the geometry.

    ParaView opens this directly; the same file also round-trips through
    h5py for any post-processing.
    """
    import h5py

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    P = np.ascontiguousarray(result.positions.reshape(-1, 3), dtype=np.float32)
    n_lines = result.n_lines
    n_vert_per_line = result.n_vertices_per_line
    n_points = int(P.shape[0])

    comp_kwargs: dict = {}
    if compression == "gzip":
        comp_kwargs = {"compression": "gzip", "compression_opts": int(compression_opts)}
    elif compression == "lzf":
        comp_kwargs = {"compression": "lzf"}

    # Lines: each line is a strip of consecutive vertex indices.
    # Connectivity: [0,1,...,L-1, L,L+1,...,2L-1, ...]; Offsets: [0, L, 2L, ..., N_points]
    connectivity = np.arange(n_points, dtype=np.int64)
    offsets = np.arange(0, n_points + 1, n_vert_per_line, dtype=np.int64)

    with h5py.File(str(out_path), "w", libver=("earliest", "v110")) as f:
        root = f.create_group("VTKHDF")
        root.attrs["Version"] = np.array([2, 0], dtype="i8")
        root.attrs.create(
            "Type", b"PolyData",
            dtype=h5py.string_dtype("ascii", 8),
        )
        for k, v in (file_attrs or {}).items():
            root.attrs[k] = v
        root.create_dataset("NumberOfPoints", data=np.array([n_points], dtype="i8"))
        root.create_dataset("Points", data=P, dtype="f4", **comp_kwargs)

        lines = root.create_group("Lines")
        lines.create_dataset("Connectivity", data=connectivity, **comp_kwargs)
        lines.create_dataset("Offsets", data=offsets)
        lines.create_dataset("NumberOfConnectivityIds",
                             data=np.array([n_points], dtype="i8"))
        lines.create_dataset("NumberOfCells",
                             data=np.array([n_lines], dtype="i8"))
        for name in ("Vertices", "Polygons", "Strips"):
            g = root.create_group(name)
            g.create_dataset("Connectivity", data=np.zeros((0,), dtype="i8"))
            g.create_dataset("Offsets", data=np.array([0], dtype="i8"))
            g.create_dataset("NumberOfConnectivityIds",
                             data=np.array([0], dtype="i8"))
            g.create_dataset("NumberOfCells", data=np.array([0], dtype="i8"))

        # Per-vertex PointData: VertexTime always written, plus any extras.
        pd = root.create_group("PointData")
        vt = np.ascontiguousarray(result.vertex_time, dtype=np.float32).reshape(-1)
        pd.create_dataset("VertexTime", data=vt, dtype="f4", **comp_kwargs)
        for name, arr in (result.vertex_fields or {}).items():
            if arr.size == 0:
                continue
            flat = np.ascontiguousarray(arr).reshape(-1)
            if flat.shape[0] != n_points:
                continue
            pd.create_dataset(name, data=flat, dtype=_hdf5_dtype_for(flat),
                              **comp_kwargs)

        # Per-line CellData (one value per line).
        cd = root.create_group("CellData")
        for name, arr in (result.line_fields or {}).items():
            flat = np.ascontiguousarray(arr).reshape(-1)
            if flat.shape[0] != n_lines:
                continue
            cd.create_dataset(name, data=flat, dtype=_hdf5_dtype_for(flat),
                              **comp_kwargs)

    return out_path
