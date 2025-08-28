"""
VTK I/O: readers for time-varying structured grids and writers for particles/density.

- VTKSeries: load structured grid time slices (.vti/.vts/.vtr)
- Writers:
  - write_particles_as_polydata(...)
  - write_scalar_field_as_image(...)
  - write_particle_series(...)

This merges VTK reading and exporting into one module to simplify usage.
"""

from __future__ import annotations
import os
import glob
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np

# Optional VTK import
try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    VTK_AVAILABLE = True
except Exception:
    VTK_AVAILABLE = False
    warnings.warn("VTK not available. VTK I/O will be disabled")

# Minimal GridMeta here to avoid circular imports with fields.base
@dataclass
class GridMeta:
    origin: np.ndarray    # (3,)
    spacing: np.ndarray   # (3,)
    shape: Tuple[int, int, int]  # (Nx, Ny, Nz)
    bounds: np.ndarray    # (2,3) min/max


def _read_vti_to_numpy(filename: str) -> Tuple[np.ndarray, GridMeta]:
    """
    Read a .vti structured grid with a 3-component vector at points or cells.

    Returns
    -------
    values: np.ndarray
        Array shaped (Nx, Ny, Nz, C) where C=1 or 3.
    meta: GridMeta
    """
    if not VTK_AVAILABLE:
        raise RuntimeError("VTK not available; cannot read VTK files")

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    image = reader.GetOutput()

    dims = image.GetDimensions()           # (Nx, Ny, Nz)
    spacing = np.array(image.GetSpacing()) # (dx, dy, dz)
    origin = np.array(image.GetOrigin())   # (ox, oy, oz)

    # Prefer point data; fallback to cell data
    pd = image.GetPointData()
    cd = image.GetCellData()

    arr = None
    if pd and pd.GetNumberOfArrays() > 0:
        arr = pd.GetArray(0)
    elif cd and cd.GetNumberOfArrays() > 0:
        arr = cd.GetArray(0)
    else:
        raise ValueError(f"No data arrays found in {filename}")

    np_arr = vtk_to_numpy(arr)
    num_comp = arr.GetNumberOfComponents()

    # Reshape: VTK stores in Fortran-like flattening with dims order
    # We reshape to (Nx, Ny, Nz, C)
    if pd and pd.GetNumberOfArrays() > 0:
        Nx, Ny, Nz = dims
    else:
        # For cell data, dims are (Nx-1, Ny-1, Nz-1)
        Nx, Ny, Nz = dims[0]-1, dims[1]-1, dims[2]-1

    np_arr = np_arr.reshape((Nx, Ny, Nz, num_comp), order="F")
    bounds = np.stack([origin, origin + spacing * np.array([Nx-1, Ny-1, Nz-1])], axis=0)
    meta = GridMeta(origin=origin, spacing=spacing, shape=(Nx, Ny, Nz), bounds=bounds.astype(float))
    return np_arr, meta


def _read_vts_vtr_to_numpy(filename: str) -> Tuple[np.ndarray, GridMeta]:
    """
    Read .vts (structured grid) or .vtr (rectilinear grid).
    Note: implementations are placeholders; adapt as needed.
    """
    if not VTK_AVAILABLE:
        raise RuntimeError("VTK not available; cannot read VTK files")

    ext = os.path.splitext(filename)[1].lower()
    if ext == ".vts":
        reader = vtk.vtkXMLStructuredGridReader()
    elif ext == ".vtr":
        reader = vtk.vtkXMLRectilinearGridReader()
    else:
        raise ValueError(f"Unsupported extension for structured/rectilinear: {ext}")

    reader.SetFileName(filename)
    reader.Update()
    ds = reader.GetOutput()

    # Extract array
    pd = ds.GetPointData()
    cd = ds.GetCellData()
    arr = pd.GetArray(0) if pd and pd.GetNumberOfArrays() else (cd.GetArray(0) if cd and cd.GetNumberOfArrays() else None)
    if arr is None:
        raise ValueError(f"No data arrays found in {filename}")

    np_arr = vtk_to_numpy(arr)
    num_comp = arr.GetNumberOfComponents()

    # Dimensions and spacing/origin depend on dataset type
    if ext == ".vts":
        dims = ds.GetDimensions()
        # Spacing approximated using bounds; refine if exact spacing available
        bounds = np.array(ds.GetBounds()).reshape(3, 2).T  # (2,3)
        origin = bounds[0]
        size = bounds[1] - bounds[0]
        Nx, Ny, Nz = dims
        spacing = size / np.maximum([Nx-1, Ny-1, Nz-1], 1)
    else:
        # Rectilinear grid with coordinate arrays
        Nx = ds.GetXCoordinates().GetNumberOfTuples()
        Ny = ds.GetYCoordinates().GetNumberOfTuples()
        Nz = ds.GetZCoordinates().GetNumberOfTuples()
        x0 = ds.GetXCoordinates().GetTuple1(0); x1 = ds.GetXCoordinates().GetTuple1(Nx-1)
        y0 = ds.GetYCoordinates().GetTuple1(0); y1 = ds.GetYCoordinates().GetTuple1(Ny-1)
        z0 = ds.GetZCoordinates().GetTuple1(0); z1 = ds.GetZCoordinates().GetTuple1(Nz-1)
        origin = np.array([x0, y0, z0], dtype=float)
        spacing = np.array([(x1-x0)/max(Nx-1, 1), (y1-y0)/max(Ny-1, 1), (z1-z0)/max(Nz-1, 1)], dtype=float)

    np_arr = np_arr.reshape((Nx, Ny, Nz, num_comp), order="F")
    bounds = np.stack([origin, origin + spacing * np.array([Nx-1, Ny-1, Nz-1])], axis=0)
    meta = GridMeta(origin=origin, spacing=spacing, shape=(Nx, Ny, Nz), bounds=bounds.astype(float))
    return np_arr, meta


class VTKSeries:
    """
    VTK series reader for structured grids.

    Accepts:
    - A single filename
    - A glob pattern (e.g., 'flow_t*.vti')
    - A list of filenames
    """

    def __init__(self, spec: Union[str, Iterable[str]]):
        if isinstance(spec, (list, tuple)):
            self._files = list(spec)
        elif isinstance(spec, str):
            if any(ch in spec for ch in "*?[]"):
                self._files = sorted(glob.glob(spec))
            else:
                self._files = [spec]
        else:
            raise TypeError("spec must be a filename, glob pattern, or list of filenames")

        if not self._files:
            raise FileNotFoundError("No VTK files provided or matched")

        self._ext = os.path.splitext(self._files[0])[1].lower()
        if self._ext not in (".vti", ".vts", ".vtr"):
            raise ValueError(f"Unsupported VTK extension: {self._ext}")

        self._meta_cache: Optional[GridMeta] = None

    def __len__(self) -> int:
        return len(self._files)

    @property
    def filenames(self) -> List[str]:
        return self._files

    def load_slice(self, i: int) -> np.ndarray:
        """
        Load a time slice as numpy array shaped (Nx, Ny, Nz, C).
        """
        filename = self._files[i]
        if self._ext == ".vti":
            arr, meta = _read_vti_to_numpy(filename)
        else:
            arr, meta = _read_vts_vtr_to_numpy(filename)
        # Cache meta from first slice
        if self._meta_cache is None:
            self._meta_cache = meta
        return arr

    def grid_meta(self) -> GridMeta:
        """
        Return grid metadata for the series (from the first slice).
        """
        if self._meta_cache is None:
            # Trigger load of first slice metadata without holding its array
            _arr, meta = (_read_vti_to_numpy(self._files[0]) if self._ext == ".vti"
                          else _read_vts_vtr_to_numpy(self._files[0]))
            self._meta_cache = meta
        return self._meta_cache


# --------------------------
# VTK Writers (exports)
# --------------------------

def write_particles_as_polydata(
    positions: np.ndarray,
    scalars: Optional[Dict[str, np.ndarray]] = None,
    path: str = "particles.vtp"
) -> None:
    """
    Write particle positions as VTK PolyData (.vtp).

    Parameters
    ----------
    positions : (N, 3) float array
    scalars : dict of name -> array
        Optional scalar/vector arrays per point. Arrays must have length N.
    path : output filename with .vtp extension
    """
    if not VTK_AVAILABLE:
        raise RuntimeError("VTK not available; cannot write VTK files")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be (N, 3)")

    N = positions.shape[0]

    # Create points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(positions.astype(np.float32)))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    # Add id connectivity as vertices (one vertex per point)
    vertices = vtk.vtkCellArray()
    vertices.Allocate(N)
    for i in range(N):
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(i)
    polydata.SetVerts(vertices)

    # Add optional scalar data
    if scalars:
        for name, arr in scalars.items():
            arr = np.asarray(arr)
            if len(arr) != N:
                raise ValueError(f"Scalar '{name}' length {len(arr)} != N={N}")
            vtk_arr = numpy_to_vtk(arr.astype(np.float32) if arr.dtype.kind == "f" else arr)
            vtk_arr.SetName(name)
            polydata.GetPointData().AddArray(vtk_arr)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(polydata)
    writer.Write()


def write_scalar_field_as_image(
    grid_meta: GridMeta,
    values: np.ndarray,
    path: str = "field.vti",
    name: str = "scalar"
) -> None:
    """
    Write a scalar field on a structured grid as .vti.

    Parameters
    ----------
    grid_meta : GridMeta with origin, spacing, shape
    values : (Nx, Ny, Nz) or (Nx, Ny, Nz, C)
    path : output .vti filename
    name : array name
    """
    if not VTK_AVAILABLE:
        raise RuntimeError("VTK not available; cannot write VTK files")

    Nx, Ny, Nz = grid_meta.shape
    img = vtk.vtkImageData()
    img.SetDimensions(Nx, Ny, Nz)
    img.SetSpacing(*grid_meta.spacing.tolist())
    img.SetOrigin(*grid_meta.origin.tolist())

    arr = np.asarray(values)
    if arr.ndim == 3:
        num_comp = 1
    elif arr.ndim == 4:
        num_comp = arr.shape[-1]
    else:
        raise ValueError("values must be (Nx,Ny,Nz) or (Nx,Ny,Nz,C)")

    flat = np.ascontiguousarray(arr.reshape(-1, num_comp, order="F"))
    vtk_arr = numpy_to_vtk(flat, deep=True)
    vtk_arr.SetNumberOfComponents(num_comp)
    vtk_arr.SetName(name)

    img.GetPointData().AddArray(vtk_arr)
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(img)
    writer.Write()


def write_particle_series(
    positions_over_time: Union[np.ndarray, Sequence[np.ndarray]],
    scalars_series: Optional[Sequence[Optional[Dict[str, np.ndarray]]]] = None,
    out_dir: str = "vtk_out",
    filename_fmt: str = "particles_t{t:05d}.vtp"
) -> List[str]:
    """
    Write a time series of particle positions as .vtp files.

    Parameters
    ----------
    positions_over_time : (T, N, 3) array or a sequence of (N,3) arrays
    scalars_series : sequence of dicts or None per time step
    out_dir : output directory
    filename_fmt : format string with {t} placeholder

    Returns
    -------
    List of written filenames in chronological order
    """
    os.makedirs(out_dir, exist_ok=True)
    files = []

    if isinstance(positions_over_time, np.ndarray):
        if positions_over_time.ndim != 3 or positions_over_time.shape[-1] != 3:
            raise ValueError("positions_over_time must be (T, N, 3)")
        T = positions_over_time.shape[0]
        for t in range(T):
            path = os.path.join(out_dir, filename_fmt.format(t=t))
            scalars = scalars_series[t] if scalars_series is not None else None
            write_particles_as_polydata(positions_over_time[t], scalars, path)
            files.append(path)
    else:
        for t, pos in enumerate(positions_over_time):
            path = os.path.join(out_dir, filename_fmt.format(t=t))
            scalars = scalars_series[t] if scalars_series is not None else None
            write_particles_as_polydata(np.asarray(pos), scalars, path)
            files.append(path)

    return files
