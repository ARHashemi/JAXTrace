# jaxtrace/density/union.py
"""
Time-union and deduplication of a particle trajectory.

The goal is a single static "material distribution" point cloud: the
union of every selected time step's particle positions, with
duplicates removed. A duplicate, here, means **two particles whose
positions fall within the same anisotropic dedup cell**:

    cell_id(x) = floor((x - bbox_lo) / tol_axis)

where ``tol_axis`` is a per-axis tolerance, by default a fraction
(typically 0.5) of the initial inter-particle spacing
:func:`jaxtrace.density.bandwidth.initial_particle_spacing`.

Two dedup modes are exposed:

  - ``batch``: read the whole selected union into memory, then run a
    single uniqueness pass via ``cell_id`` ``int64`` keys. Output is
    deterministic; the survivor within each occupied cell is the
    particle with the lowest position in the read order
    (= step-major, then particle-id within step).

  - ``incremental``: process steps one at a time. A running
    ``set`` of seen ``cell_id`` keys decides whether each incoming
    particle is a duplicate of an earlier-step survivor; survivors
    are appended to a running list. Memory peak is the same as batch;
    advantage is that very large trajectories can be streamed
    through without forming the full union first.

Neither mode dedups *within* a single step — particles within a
single step are by construction distinct under RK4 advection.

Per-particle PointData fields (ParticleID, ElementID, Group,
Temperature, MaxTemperature, Escaped, …) are carried through the
union: each survivor inherits its first-seen attributes by default.
Specific scalar fields can instead be max-reduced across the
collapsed group via the ``reduce_max_fields`` argument — useful for
"hottest temperature ever reached in this cell".

A region-of-interest box can be applied at read time: particles
outside the ROI are dropped before dedup, so the resulting cloud
and the optional voxel-grid density both live entirely in the ROI.

The output is a single static PolyData VTKHDF file
``<stem>_union.vtkhdf`` plus, optionally, a companion ``.npy`` for
downstream Python tooling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Cell-id encoding for the dedup hash
# -----------------------------------------------------------------------------

def _flat_cell_id(
    P: np.ndarray,            # (N, 3) float32
    bbox_lo: np.ndarray,      # (3,) float32
    tol_axis: np.ndarray,     # (3,) float32
    dims_yz: Tuple[int, int],  # (Ny, Nz)
) -> np.ndarray:
    """
    Compute flat int64 cell ids for each particle.

    The hash is anisotropic: cell size differs per axis. We pack
    (i, j, k) -> i*Ny*Nz + j*Nz + k. With reasonable Ny, Nz
    (< 10^6 each) the flat id fits comfortably in int64.
    """
    ny, nz = dims_yz
    ijk = np.floor((P - bbox_lo) / tol_axis).astype(np.int64)
    np.maximum(ijk, 0, out=ijk)
    return ijk[:, 0] * (ny * nz) + ijk[:, 1] * nz + ijk[:, 2]


def _bbox_and_dims(
    bbox_min: np.ndarray, bbox_max: np.ndarray, tol_axis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Return (padded bbox_lo, padded bbox_hi, Ny, Nz) for the cell hash.
    """
    lo = np.asarray(bbox_min, dtype=np.float32) - tol_axis.astype(np.float32)
    hi = np.asarray(bbox_max, dtype=np.float32) + tol_axis.astype(np.float32)
    extent = hi - lo
    ny = int(np.ceil(extent[1] / tol_axis[1])) + 1
    nz = int(np.ceil(extent[2] / tol_axis[2])) + 1
    return lo, hi, ny, nz


# -----------------------------------------------------------------------------
# ROI mask
# -----------------------------------------------------------------------------

def roi_box_from_fraction(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    fractions: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map ``[xlo, xhi, ylo, yhi, zlo, zhi]`` fractions in [0, 1] to absolute
    metres against the given full-domain bbox.

    Returns ``(roi_lo, roi_hi)`` as ``(3,)`` float32 arrays.
    """
    f = np.asarray(fractions, dtype=np.float64).reshape(3, 2)
    lo = np.asarray(bbox_min, dtype=np.float64)
    hi = np.asarray(bbox_max, dtype=np.float64)
    extent = hi - lo
    roi_lo = (lo + f[:, 0] * extent).astype(np.float32)
    roi_hi = (lo + f[:, 1] * extent).astype(np.float32)
    return roi_lo, roi_hi


def _roi_mask(
    P: np.ndarray, roi_lo: np.ndarray, roi_hi: np.ndarray,
) -> np.ndarray:
    """Return a boolean (N,) mask of particles inside [roi_lo, roi_hi]."""
    return np.all((P >= roi_lo) & (P <= roi_hi), axis=1)


# -----------------------------------------------------------------------------
# Public API: dedup modes
# -----------------------------------------------------------------------------

@dataclass
class UnionResult:
    """Output of a union+dedup pass."""
    positions: np.ndarray            # (N_unique, 3) float32
    point_data: dict                 # name -> (N_unique, ...) numpy array
    n_input: int                     # total particles read (post-ROI)
    n_kept: int                      # survivors after dedup
    bbox_lo: np.ndarray              # padded dedup-hash lo (3,)
    bbox_hi: np.ndarray              # padded dedup-hash hi (3,)
    tol_axis: np.ndarray             # (3,)
    dedup_mode: str                  # "batch" | "incremental" | "none"
    n_steps_processed: int = 0
    roi_lo: Optional[np.ndarray] = None
    roi_hi: Optional[np.ndarray] = None


def _accumulate_max(
    out: dict, name: str, values: np.ndarray, indices: np.ndarray,
) -> None:
    """
    Apply ``out[name][indices] = max(out[name][indices], values)`` in
    place. ``values`` and ``indices`` must be the same length. Used by
    the max-reduction path so the survivor's chosen scalar field
    becomes the maximum across all particles collapsed into its cell.
    """
    np.maximum.at(out[name], indices, values)


def dedup_batch(
    positions_per_step: Iterable,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    tol_axis: np.ndarray,
    *,
    field_names: Optional[Sequence[str]] = None,
    reduce_max_fields: Optional[Sequence[str]] = None,
    roi_lo: Optional[np.ndarray] = None,
    roi_hi: Optional[np.ndarray] = None,
) -> UnionResult:
    """
    Batch dedup: read the whole union into memory, then run a single
    uniqueness pass.

    ``positions_per_step`` must yield either ``(N, 3)`` arrays (legacy
    contract) or ``(positions, fields_dict)`` pairs when
    ``field_names`` is provided.

    Survivors inherit per-particle attributes from the first-seen
    particle in each cell. Names listed in ``reduce_max_fields`` are
    additionally max-reduced across the collapsed group.

    ``roi_lo`` / ``roi_hi`` (when provided) restrict the union to
    particles inside the given absolute box; particles outside are
    discarded before the hash. The dedup-hash bbox should already be
    the ROI when ROI is active.
    """
    arrs: list[np.ndarray] = []
    field_arrs: dict[str, list[np.ndarray]] = (
        {n: [] for n in field_names} if field_names else {}
    )
    n_steps = 0
    have_roi = roi_lo is not None and roi_hi is not None

    for item in positions_per_step:
        if field_names is not None:
            P, fdict = item
        else:
            P, fdict = item, None
        P = np.asarray(P, dtype=np.float32)
        if have_roi and P.size:
            mask = _roi_mask(P, roi_lo, roi_hi)
            P = P[mask]
            if fdict is not None:
                fdict = {k: v[mask] for k, v in fdict.items() if k in field_arrs}
        arrs.append(P)
        if field_arrs and fdict is not None:
            for n in field_arrs:
                if n in fdict:
                    field_arrs[n].append(fdict[n])
                else:
                    # PointData field missing for this step → backfill zeros
                    field_arrs[n].append(_zeros_like_field(n, field_arrs[n], P.shape[0]))
        n_steps += 1

    if not arrs:
        return UnionResult(
            positions=np.zeros((0, 3), dtype=np.float32),
            point_data={n: np.zeros((0,), dtype=np.float32) for n in field_arrs},
            n_input=0, n_kept=0,
            bbox_lo=np.asarray(bbox_min, dtype=np.float32),
            bbox_hi=np.asarray(bbox_max, dtype=np.float32),
            tol_axis=np.asarray(tol_axis, dtype=np.float32),
            dedup_mode="batch", n_steps_processed=0,
            roi_lo=roi_lo, roi_hi=roi_hi,
        )

    P_all = np.concatenate(arrs, axis=0)
    n_input = int(P_all.shape[0])
    fields_all = {n: np.concatenate(field_arrs[n], axis=0) for n in field_arrs}

    lo, _, ny, nz = _bbox_and_dims(bbox_min, bbox_max, tol_axis)
    flat = _flat_cell_id(P_all, lo, tol_axis, (ny, nz))
    _, first_idx, inverse = np.unique(flat, return_index=True, return_inverse=True)
    # Restore read order for survivors so the output ordering is
    # deterministic and matches the streaming/incremental mode.
    order = np.argsort(first_idx)
    kept_positions = P_all[first_idx[order]]

    # First-seen: index each field by `first_idx[order]`. Then,
    # for fields named in reduce_max_fields, max-reduce across the
    # collapsed group via scatter-max with the inverse map.
    survivor_of_first: np.ndarray = np.empty_like(first_idx)
    survivor_of_first[order] = np.arange(order.shape[0])
    # survivor_of_first[u] = the survivor row corresponding to unique key u
    kept_fields: dict[str, np.ndarray] = {}
    rmax = set(reduce_max_fields or ())
    for n, arr in fields_all.items():
        first_view = arr[first_idx[order]]
        kept_fields[n] = first_view.copy()
        if n in rmax:
            # arr is 1- or 2-D. For 2-D, max element-wise across components.
            if arr.ndim == 1:
                survivor_row = survivor_of_first[inverse]
                _accumulate_max({n: kept_fields[n]}, n, arr, survivor_row)
            else:
                # Process each component independently.
                survivor_row = survivor_of_first[inverse]
                for c in range(arr.shape[1]):
                    np.maximum.at(kept_fields[n][:, c], survivor_row, arr[:, c])

    return UnionResult(
        positions=kept_positions.astype(np.float32, copy=False),
        point_data=kept_fields,
        n_input=n_input,
        n_kept=int(kept_positions.shape[0]),
        bbox_lo=lo.astype(np.float32),
        bbox_hi=np.asarray(bbox_max, dtype=np.float32) + tol_axis.astype(np.float32),
        tol_axis=tol_axis.astype(np.float32),
        dedup_mode="batch",
        n_steps_processed=n_steps,
        roi_lo=roi_lo, roi_hi=roi_hi,
    )


def _zeros_like_field(name: str, prior: list[np.ndarray], n: int) -> np.ndarray:
    """Synth a zero-filled slot when a step is missing a field."""
    if not prior:
        return np.zeros((n,), dtype=np.float32)
    proto = prior[-1]
    if proto.ndim == 1:
        return np.zeros((n,), dtype=proto.dtype)
    return np.zeros((n, proto.shape[1]), dtype=proto.dtype)


def dedup_incremental(
    positions_per_step: Iterable,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    tol_axis: np.ndarray,
    log_every: int = 100,
    *,
    field_names: Optional[Sequence[str]] = None,
    reduce_max_fields: Optional[Sequence[str]] = None,
    roi_lo: Optional[np.ndarray] = None,
    roi_hi: Optional[np.ndarray] = None,
) -> UnionResult:
    """
    Incremental dedup: process each step against the running set of
    cell ids already claimed by earlier steps. Survivors are appended
    to the running list.

    Same first-seen semantics as ``dedup_batch``. With
    ``reduce_max_fields``, the scalar field of a survivor is updated
    in place when a later-step duplicate carries a larger value.
    """
    tol_axis = np.asarray(tol_axis, dtype=np.float32)
    lo, _, ny, nz = _bbox_and_dims(bbox_min, bbox_max, tol_axis)
    have_roi = roi_lo is not None and roi_hi is not None
    rmax = set(reduce_max_fields or ())

    # seen[key] -> row index of survivor in the running list
    seen: dict = {}
    survivor_positions: List[np.ndarray] = []
    survivor_fields: dict[str, list[np.ndarray]] = (
        {n: [] for n in field_names} if field_names else {}
    )
    n_input = 0
    n_steps = 0

    for step_idx, item in enumerate(positions_per_step):
        if field_names is not None:
            P, fdict = item
        else:
            P, fdict = item, None
        P = np.asarray(P, dtype=np.float32)
        if have_roi and P.size:
            mask = _roi_mask(P, roi_lo, roi_hi)
            P = P[mask]
            if fdict is not None:
                fdict = {k: v[mask] for k, v in fdict.items() if k in survivor_fields}
        if P.size == 0:
            n_steps += 1
            continue
        n_input += int(P.shape[0])
        flat = _flat_cell_id(P, lo, tol_axis, (ny, nz))
        flat_list = flat.tolist()

        # Vectorised "new vs existing" partition.
        new_mask = np.fromiter(
            (k not in seen for k in flat_list),
            dtype=bool, count=flat.size,
        )

        # --- Update existing survivors via max-reduction (if requested). ---
        if rmax and not new_mask.all():
            existing_idx = np.where(~new_mask)[0]
            existing_keys = flat[existing_idx]
            survivor_rows = np.fromiter(
                (seen[int(k)] for k in existing_keys.tolist()),
                dtype=np.int64, count=existing_keys.size,
            )
            for n in rmax:
                if fdict is None or n not in fdict:
                    continue
                # The running survivor_fields[n] is a list of per-step
                # chunks. Flatten on demand to scatter-max into.
                # Mutating list-of-chunks is awkward, so we collapse to
                # a single backing array lazily on first reduce-max hit.
                if not survivor_fields[n] or not isinstance(
                    survivor_fields[n][0], np.ndarray
                ):
                    continue
                # Concatenate the chunks into one backing array so we
                # can index by absolute survivor row.
                flat_arr = np.concatenate(survivor_fields[n], axis=0)
                vals = fdict[n][existing_idx]
                if flat_arr.ndim == 1:
                    np.maximum.at(flat_arr, survivor_rows, vals)
                else:
                    for c in range(flat_arr.shape[1]):
                        np.maximum.at(
                            flat_arr[:, c], survivor_rows, vals[:, c],
                        )
                # Repack as a single-element list.
                survivor_fields[n] = [flat_arr]

        # --- Append new survivors. ---
        if new_mask.any():
            new_flat = flat[new_mask]
            # Within-step dedup: keep first occurrence per unique new key.
            _, first_idx_local = np.unique(new_flat, return_index=True)
            first_idx_local.sort()
            new_kept_keys = new_flat[first_idx_local]
            new_positions = P[new_mask][first_idx_local]
            current_count = sum(s.shape[0] for s in survivor_positions)
            survivor_positions.append(new_positions)
            if survivor_fields and fdict is not None:
                for n in survivor_fields:
                    if n in fdict:
                        chunk = fdict[n][new_mask][first_idx_local]
                    else:
                        chunk = _zeros_like_field(
                            n, survivor_fields[n], new_positions.shape[0]
                        )
                    survivor_fields[n].append(chunk)
            for offset, k in enumerate(new_kept_keys.tolist()):
                seen[int(k)] = current_count + offset

        n_steps += 1
        if log_every and (n_steps % log_every == 0):
            n_so_far = sum(s.shape[0] for s in survivor_positions)
            print(f"  [union] step {step_idx}: read {n_input:,}, "
                  f"kept {n_so_far:,} ({n_so_far / max(n_input, 1) * 100:.1f}%)")

    if survivor_positions:
        kept = np.concatenate(survivor_positions, axis=0)
    else:
        kept = np.zeros((0, 3), dtype=np.float32)
    kept_fields = {
        n: (np.concatenate(parts, axis=0)
            if parts else np.zeros((0,), dtype=np.float32))
        for n, parts in survivor_fields.items()
    }

    return UnionResult(
        positions=kept,
        point_data=kept_fields,
        n_input=n_input,
        n_kept=int(kept.shape[0]),
        bbox_lo=lo,
        bbox_hi=np.asarray(bbox_max, dtype=np.float32) + tol_axis.astype(np.float32),
        tol_axis=tol_axis,
        dedup_mode="incremental",
        n_steps_processed=n_steps,
        roi_lo=roi_lo, roi_hi=roi_hi,
    )


def union_no_dedup(
    positions_per_step: Iterable,
    bbox_min: np.ndarray, bbox_max: np.ndarray,
    *,
    field_names: Optional[Sequence[str]] = None,
    roi_lo: Optional[np.ndarray] = None,
    roi_hi: Optional[np.ndarray] = None,
) -> UnionResult:
    """
    Concatenate without dedup — useful when the user wants the raw
    union as input to a downstream tool that will dedup itself.
    """
    arrs: list[np.ndarray] = []
    field_arrs: dict[str, list[np.ndarray]] = (
        {n: [] for n in field_names} if field_names else {}
    )
    n_steps = 0
    have_roi = roi_lo is not None and roi_hi is not None
    for item in positions_per_step:
        if field_names is not None:
            P, fdict = item
        else:
            P, fdict = item, None
        P = np.asarray(P, dtype=np.float32)
        if have_roi and P.size:
            mask = _roi_mask(P, roi_lo, roi_hi)
            P = P[mask]
            if fdict is not None:
                fdict = {k: v[mask] for k, v in fdict.items() if k in field_arrs}
        arrs.append(P)
        if field_arrs and fdict is not None:
            for n in field_arrs:
                if n in fdict:
                    field_arrs[n].append(fdict[n])
                else:
                    field_arrs[n].append(_zeros_like_field(n, field_arrs[n], P.shape[0]))
        n_steps += 1

    P_all = np.concatenate(arrs, axis=0) if arrs else np.zeros((0, 3), dtype=np.float32)
    kept_fields = {
        n: (np.concatenate(field_arrs[n], axis=0)
            if field_arrs[n] else np.zeros((0,), dtype=np.float32))
        for n in field_arrs
    }
    return UnionResult(
        positions=P_all,
        point_data=kept_fields,
        n_input=int(P_all.shape[0]),
        n_kept=int(P_all.shape[0]),
        bbox_lo=np.asarray(bbox_min, dtype=np.float32),
        bbox_hi=np.asarray(bbox_max, dtype=np.float32),
        tol_axis=np.zeros(3, dtype=np.float32),
        dedup_mode="none",
        n_steps_processed=n_steps,
        roi_lo=roi_lo, roi_hi=roi_hi,
    )


# -----------------------------------------------------------------------------
# Output writers
# -----------------------------------------------------------------------------

def _hdf5_dtype_for(arr: np.ndarray) -> str:
    """Mirror TransientPolyDataWriter's PointData dtype policy."""
    if arr.dtype == np.bool_ or arr.dtype == np.uint8:
        return "u1"
    if arr.dtype in (np.int32, np.int64):
        return "i4"
    if arr.dtype == np.float64:
        return "f8"
    return "f4"


def write_union_vtkhdf(
    out_path: Path,
    result: UnionResult,
    compression: str | None = "gzip",
    compression_opts: int = 1,
    *,
    extra_point_data: Optional[dict] = None,
    file_attrs: Optional[dict] = None,
) -> Path:
    """
    Write the deduplicated cloud as a static PolyData VTKHDF
    (single-step, one vertex per particle), including every
    per-particle PointData field carried in ``result.point_data``.

    ``extra_point_data`` lets callers attach derived fields
    (CoMovingDisplacement, CoMovingMagnitude, …) without mutating
    ``result``. ``file_attrs`` is a flat dict written as h5 attrs
    on ``/VTKHDF`` — used for time metadata (t_start, t_end, t_pass,
    drift_velocity).
    """
    import h5py

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    P = np.ascontiguousarray(result.positions, dtype=np.float32)
    N = int(P.shape[0])

    comp_kwargs: dict = {}
    if compression == "gzip":
        comp_kwargs = {"compression": "gzip", "compression_opts": int(compression_opts)}
    elif compression == "lzf":
        comp_kwargs = {"compression": "lzf"}

    with h5py.File(str(out_path), "w", libver=("earliest", "v110")) as f:
        root = f.create_group("VTKHDF")
        root.attrs["Version"] = np.array([2, 0], dtype="i8")
        root.attrs.create(
            "Type", b"PolyData",
            dtype=h5py.string_dtype("ascii", 8),
        )
        for k, v in (file_attrs or {}).items():
            root.attrs[k] = v
        root.create_dataset("NumberOfPoints", data=np.array([N], dtype="i8"))
        root.create_dataset("Points", data=P, dtype="f4", **comp_kwargs)

        verts = root.create_group("Vertices")
        connectivity = np.arange(N, dtype="i8")
        offsets = np.arange(N + 1, dtype="i8")
        verts.create_dataset("Connectivity", data=connectivity)
        verts.create_dataset("Offsets", data=offsets)
        verts.create_dataset(
            "NumberOfConnectivityIds", data=np.array([N], dtype="i8"),
        )
        verts.create_dataset("NumberOfCells", data=np.array([N], dtype="i8"))
        for name in ("Lines", "Polygons", "Strips"):
            g = root.create_group(name)
            g.create_dataset("Connectivity", data=np.zeros((0,), dtype="i8"))
            g.create_dataset("Offsets", data=np.array([0], dtype="i8"))
            g.create_dataset(
                "NumberOfConnectivityIds", data=np.array([0], dtype="i8"),
            )
            g.create_dataset("NumberOfCells", data=np.array([0], dtype="i8"))

        pd = root.create_group("PointData")
        all_pd = dict(result.point_data or {})
        if extra_point_data:
            all_pd.update(extra_point_data)
        for name, arr in all_pd.items():
            if arr.shape[0] != N:
                continue
            arr_c = np.ascontiguousarray(arr)
            pd.create_dataset(
                name, data=arr_c,
                dtype=_hdf5_dtype_for(arr_c),
                **comp_kwargs,
            )

    return out_path


def write_union_npy(out_path: Path, result: UnionResult) -> Path:
    """
    Dump the deduplicated cloud as NumPy files.

    Writes ``<out_path>`` for positions and, when ``result.point_data``
    is non-empty, sibling files ``<stem>_<field>.npy`` per field.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, np.ascontiguousarray(result.positions, dtype=np.float32))
    stem = out_path.with_suffix("")
    for name, arr in (result.point_data or {}).items():
        np.save(stem.with_name(f"{stem.name}_{name}.npy"),
                np.ascontiguousarray(arr))
    return out_path


def read_union_vtkhdf(in_path: Path) -> "UnionResult":
    """
    Reverse of :func:`write_union_vtkhdf`. Loads the static PolyData
    file into a :class:`UnionResult` whose ``point_data`` mirrors the
    file's ``/VTKHDF/PointData`` group.

    File-level attributes (drift_velocity, t_start, t_end, t_pass, …)
    end up on the returned object via the dict-like ``UnionResult``
    ``__dict__`` — callers that want them should read the file's
    ``VTKHDF.attrs`` directly.
    """
    import h5py
    in_path = Path(in_path)
    with h5py.File(str(in_path), "r") as f:
        root = f["/VTKHDF"]
        P = np.asarray(root["Points"][:], dtype=np.float32)
        pd: dict[str, np.ndarray] = {}
        if "PointData" in root:
            for name in sorted(root["PointData"].keys()):
                pd[name] = np.asarray(root["PointData"][name][:])
    return UnionResult(
        positions=P,
        point_data=pd,
        n_input=int(P.shape[0]),
        n_kept=int(P.shape[0]),
        bbox_lo=P.min(axis=0) if P.size else np.zeros(3, dtype=np.float32),
        bbox_hi=P.max(axis=0) if P.size else np.zeros(3, dtype=np.float32),
        tol_axis=np.zeros(3, dtype=np.float32),
        dedup_mode="loaded",
        n_steps_processed=0,
    )


def read_union_attrs(in_path: Path) -> dict:
    """Return ``/VTKHDF.attrs`` as a plain dict (drops h5py wrappers)."""
    import h5py
    out: dict = {}
    with h5py.File(str(in_path), "r") as f:
        for k, v in f["/VTKHDF"].attrs.items():
            out[k] = v.tolist() if hasattr(v, "tolist") else v
    return out
