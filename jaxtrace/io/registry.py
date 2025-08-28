"""
Registry for dataset opening by file extension or glob pattern.
"""

from __future__ import annotations
import os
import glob
from typing import Union, Iterable, Optional

# Lazy imports of backends
# These imports are safe (lightweight) because modules gate heavy deps inside functions
from .vtk_io import VTKSeries
from .hdf5_io import H5Series


def _first_match(pattern: str) -> Optional[str]:
    files = sorted(glob.glob(pattern))
    return files[0] if files else None


def open_dataset(spec: Union[str, Iterable[str]]):
    """
    Open a dataset series from a file, a glob pattern, or an iterable of filenames.

    Parameters
    ----------
    spec:
        - A single filename (e.g., 'flow_t0000.vti' or 'data.h5')
        - A glob pattern (e.g., 'flow_t*.vti')
        - An iterable of filenames (e.g., a list of .vti files)

    Returns
    -------
    VTKSeries | H5Series
        A series object exposing __len__() and load_slice(i) -> np.ndarray.

    Notes
    -----
    - VTK series: returns velocity fields as numpy arrays (Nx, Ny, Nz, C),
      where C is 3 for vector fields.
    - HDF5 series: expects a default dataset name 'velocity' unless overridden.
    """
    if isinstance(spec, (list, tuple)):
        if not spec:
            raise ValueError("Empty iterable passed to open_dataset")
        ext = os.path.splitext(spec[0])[1].lower()
        if ext in (".vti", ".vts", ".vtr"):
            return VTKSeries(spec)
        elif ext in (".h5", ".hdf5"):
            return H5Series(spec)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    if isinstance(spec, str):
        # If a glob, resolve to first file to dispatch
        if any(ch in spec for ch in "*?[]"):
            first = _first_match(spec)
            if first is None:
                raise FileNotFoundError(f"No files match pattern: {spec}")
            ext = os.path.splitext(first)[1].lower()
            if ext in (".vti", ".vts", ".vtr"):
                return VTKSeries(spec)
            elif ext in (".h5", ".hdf5"):
                return H5Series(spec)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        else:
            ext = os.path.splitext(spec)[1].lower()
            if ext in (".vti", ".vts", ".vtr"):
                return VTKSeries(spec)
            elif ext in (".h5", ".hdf5"):
                return H5Series(spec)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

    raise TypeError("spec must be a str or an iterable of str")
