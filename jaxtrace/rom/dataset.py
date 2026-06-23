"""
Discover FOM cases and assemble a common-grid snapshot matrix.

Each FSW case produces ONE static 3D mean-density field, stored as a
VTKHDF ImageData file. Across cases the voxel grids differ (extent,
origin, spacing) because the advected domain length scales with
``v_adv``. PCA requires every snapshot as a vector on a single, shared
grid, so this module:

1. discovers the cases and their ``(v_adv, omega_pin)`` parameters,
2. builds a common *reference grid* (intersection of the kept cases'
   bounding boxes at a fixed resolution), and
3. trilinearly resamples each native snapshot onto that reference grid.

The resampling runs on JAX (``jax.scipy.ndimage.map_coordinates``) so it
uses the GPU when available, matching the rest of JAXTrace.
"""

from __future__ import annotations

import csv
import glob
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Default location of the FOM dataset on the mounted remote workstation.
DEFAULT_FOM_ROOT = Path(
    "/home/arhashemi/fsw-gpu/scratch/shared/ROM/FOM"
)

# Cases excluded from the common-grid PCA. 000 and 001 are x-extent
# outliers that would clip the intersection box to a thin slab; 002 has
# no density output. See the rom-fom-dataset project memory.
DEFAULT_EXCLUDE = ("000", "001", "002")

_CASE_RE = re.compile(r"cylindrical_(\d+)\.gid")


@dataclass(frozen=True)
class ReferenceGrid:
    """A regular axis-aligned voxel grid (the common PCA grid)."""

    origin: np.ndarray          # (3,) world-space lower corner (x, y, z)
    spacing: np.ndarray         # (3,) voxel size (x, y, z)
    shape: Tuple[int, int, int]  # (nx, ny, nz) number of points per axis

    @property
    def n_voxels(self) -> int:
        return int(self.shape[0] * self.shape[1] * self.shape[2])

    def axis_coords(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World coordinates of the grid points along each axis."""
        return tuple(  # type: ignore[return-value]
            self.origin[a] + self.spacing[a] * np.arange(self.shape[a])
            for a in range(3)
        )


@dataclass
class CaseSnapshot:
    """One FOM case: its parameters and native density field metadata."""

    case_number: str
    v_adv: float
    omega_pin: float
    path: Path
    native_origin: np.ndarray
    native_spacing: np.ndarray
    native_shape_zyx: Tuple[int, int, int]  # as stored (z, y, x)


@dataclass
class SnapshotDataset:
    """The assembled common-grid snapshot matrix and its metadata."""

    snapshots: List[CaseSnapshot]
    grid: ReferenceGrid
    # (n_cases, n_voxels) matrix of resampled mean_density, row per case.
    matrix: np.ndarray
    # (n_cases, 2) inputs [v_adv, omega_pin] aligned with matrix rows.
    params: np.ndarray
    field_name: str = "mean_density"

    @property
    def n_cases(self) -> int:
        return self.matrix.shape[0]

    @property
    def case_numbers(self) -> List[str]:
        return [s.case_number for s in self.snapshots]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _read_native_header(path: Path):
    """Return (origin, spacing, shape_zyx) for a VTKHDF ImageData file."""
    import h5py

    with h5py.File(str(path), "r") as f:
        g = f["/VTKHDF"]
        origin = np.asarray(g.attrs["Origin"], dtype=np.float64)
        spacing = np.asarray(g.attrs["Spacing"], dtype=np.float64)
        shape_zyx = tuple(int(s) for s in g["PointData/mean_density"].shape)
    return origin, spacing, shape_zyx


def discover_cases(
    fom_root: Path = DEFAULT_FOM_ROOT,
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
    params_csv: Optional[Path] = None,
) -> List[CaseSnapshot]:
    """
    Find every ``particles_union_density.vtkhdf`` under ``fom_root`` and
    attach its ``(v_adv, omega_pin)`` parameters.

    Parameters are read from ``case_parameters.csv`` when present (faster
    and authoritative); otherwise from each case's ``run_jaxtrace.sh``.
    Cases whose number is in ``exclude`` are skipped.
    """
    fom_root = Path(fom_root)
    params_csv = params_csv or (fom_root / "case_parameters.csv")
    param_map = _load_param_map(fom_root, params_csv)

    pattern = str(
        fom_root / "cylindrical_*.gid" / "post_pt" / "*" / "union"
        / "particles_union_density.vtkhdf"
    )
    snapshots: List[CaseSnapshot] = []
    for p in sorted(glob.glob(pattern)):
        path = Path(p)
        m = _CASE_RE.search(p)
        if m is None:
            continue
        case = m.group(1)
        if case in exclude:
            continue
        if case not in param_map:
            raise KeyError(
                f"case {case} found on disk but has no (v_adv, omega_pin) "
                f"in {params_csv} or run_jaxtrace.sh"
            )
        v_adv, omega = param_map[case]
        origin, spacing, shape_zyx = _read_native_header(path)
        snapshots.append(
            CaseSnapshot(
                case_number=case,
                v_adv=v_adv,
                omega_pin=omega,
                path=path,
                native_origin=origin,
                native_spacing=spacing,
                native_shape_zyx=shape_zyx,
            )
        )
    if not snapshots:
        raise FileNotFoundError(
            f"no density files matched under {fom_root} (pattern: {pattern})"
        )
    return snapshots


def _load_param_map(
    fom_root: Path, params_csv: Path
) -> Dict[str, Tuple[float, float]]:
    """case_number -> (v_adv, omega_pin), preferring the CSV."""
    out: Dict[str, Tuple[float, float]] = {}
    if params_csv.is_file():
        with open(params_csv) as fh:
            for row in csv.DictReader(fh):
                m = _CASE_RE.search(row.get("case_name", ""))
                case = m.group(1) if m else row.get("case_number", "").strip()
                if not case:
                    continue
                out[case] = (
                    float(row["inlet_velocity"]),
                    float(row["pin_rpm"]),
                )
        return out
    # Fallback: parse run_jaxtrace.sh per case.
    for sh in sorted(glob.glob(str(fom_root / "cylindrical_*.gid" / "run_jaxtrace.sh"))):
        m = _CASE_RE.search(sh)
        if m is None:
            continue
        case = m.group(1)
        v_adv = omega = None
        with open(sh) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("INLET_VELOCITY="):
                    v_adv = float(line.split("=", 1)[1].split()[0])
                elif line.startswith("PIN_RPM="):
                    omega = float(line.split("=", 1)[1].split()[0])
        if v_adv is not None and omega is not None:
            out[case] = (v_adv, omega)
    return out


# ---------------------------------------------------------------------------
# Common reference grid
# ---------------------------------------------------------------------------

def build_reference_grid(
    snapshots: Sequence[CaseSnapshot],
    resolution: Optional[Tuple[int, int, int]] = None,
) -> ReferenceGrid:
    """
    Intersection bounding box of all ``snapshots`` at a fixed resolution.

    The box is the per-axis overlap of every case's native extent, so
    every reference point falls inside every case's domain and resampling
    never extrapolates. When ``resolution`` is None it is chosen as
    span / median(native spacing), rounded.
    """
    los, his, spacings = [], [], []
    for s in snapshots:
        # native_shape_zyx is (z, y, x); convert to (nx, ny, nz) points.
        nz, ny, nx = s.native_shape_zyx
        npts = np.array([nx, ny, nz], dtype=np.float64)
        lo = s.native_origin
        # ImageData WholeExtent had npts cells -> hi = origin + spacing*npts.
        hi = s.native_origin + s.native_spacing * npts
        los.append(lo)
        his.append(hi)
        spacings.append(s.native_spacing)
    lo = np.max(np.asarray(los), axis=0)
    hi = np.min(np.asarray(his), axis=0)
    if np.any(hi <= lo):
        raise ValueError(
            f"empty intersection box: lo={lo}, hi={hi}. "
            f"Check the excluded cases."
        )
    if resolution is None:
        sp_med = np.median(np.asarray(spacings), axis=0)
        res = np.maximum(np.round((hi - lo) / sp_med).astype(int), 2)
        resolution = (int(res[0]), int(res[1]), int(res[2]))
    spacing = (hi - lo) / (np.asarray(resolution, dtype=np.float64) - 1)
    return ReferenceGrid(origin=lo, spacing=spacing, shape=tuple(resolution))


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def _read_field_zyx(path: Path, field_name: str) -> np.ndarray:
    import h5py

    with h5py.File(str(path), "r") as f:
        return np.asarray(
            f[f"/VTKHDF/PointData/{field_name}"][...], dtype=np.float32
        )


def resample_to_reference(
    snap: CaseSnapshot,
    grid: ReferenceGrid,
    field_name: str = "mean_density",
) -> np.ndarray:
    """
    Trilinearly resample one case's field onto ``grid``.

    Returns a flat (n_voxels,) float32 vector in C order over (x, y, z)
    point indices (x fastest), matching ``grid.shape`` raveled as
    ``(nx, ny, nz)``.
    """
    import jax
    import jax.numpy as jnp

    field_zyx = _read_field_zyx(snap.path, field_name)  # (nz, ny, nx)

    # Reference grid world coordinates per axis (x, y, z).
    xs, ys, zs = grid.axis_coords()
    # Convert world coords -> native fractional index along each axis.
    # native index_a = (world - native_origin_a) / native_spacing_a.
    ix = (xs - snap.native_origin[0]) / snap.native_spacing[0]
    iy = (ys - snap.native_origin[1]) / snap.native_spacing[1]
    iz = (zs - snap.native_origin[2]) / snap.native_spacing[2]

    # map_coordinates indexes the array dims in order; field is (z, y, x),
    # so coords must be stacked as (iz, iy, ix).
    IZ, IY, IX = jnp.meshgrid(
        jnp.asarray(iz), jnp.asarray(iy), jnp.asarray(ix), indexing="ij"
    )  # each (nz_ref, ny_ref, nx_ref) in reference (z, y, x) order
    coords = jnp.stack([IZ.ravel(), IY.ravel(), IX.ravel()], axis=0)

    sampled = jax.scipy.ndimage.map_coordinates(
        jnp.asarray(field_zyx), coords, order=1, mode="nearest"
    )  # flat, in reference (z, y, x) C order

    nx, ny, nz = grid.shape
    # reshape to (nz, ny, nx) then transpose to (nx, ny, nz) and ravel.
    arr = np.asarray(sampled, dtype=np.float32).reshape(nz, ny, nx)
    arr = np.transpose(arr, (2, 1, 0))  # -> (nx, ny, nz)
    return arr.ravel(order="C")


def load_dataset(
    fom_root: Path = DEFAULT_FOM_ROOT,
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
    resolution: Optional[Tuple[int, int, int]] = None,
    field_name: str = "mean_density",
    params_csv: Optional[Path] = None,
    verbose: bool = True,
) -> SnapshotDataset:
    """
    End-to-end: discover cases, build the common grid, resample every
    snapshot onto it, and stack into a (n_cases, n_voxels) matrix.
    """
    snapshots = discover_cases(fom_root, exclude=exclude, params_csv=params_csv)
    grid = build_reference_grid(snapshots, resolution=resolution)
    if verbose:
        print(f"[rom] {len(snapshots)} cases (excluded {list(exclude)})")
        print(f"[rom] reference grid shape (nx,ny,nz) = {grid.shape}, "
              f"n_voxels = {grid.n_voxels:,}")
        print(f"[rom]   origin  = {grid.origin}")
        print(f"[rom]   spacing = {grid.spacing}")

    rows = np.empty((len(snapshots), grid.n_voxels), dtype=np.float32)
    params = np.empty((len(snapshots), 2), dtype=np.float64)
    for i, snap in enumerate(snapshots):
        rows[i] = resample_to_reference(snap, grid, field_name=field_name)
        params[i] = (snap.v_adv, snap.omega_pin)
        if verbose:
            print(f"[rom]   resampled case {snap.case_number}  "
                  f"v_adv={snap.v_adv:.4e} omega={snap.omega_pin:+.0f}  "
                  f"rho[min,max]=[{rows[i].min():.4g},{rows[i].max():.4g}]")

    return SnapshotDataset(
        snapshots=snapshots,
        grid=grid,
        matrix=rows,
        params=params,
        field_name=field_name,
    )
