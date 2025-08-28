"""
I/O backends for datasets and exports.

- vtk_io: VTK time-series readers/writers (.vti/.vts/.vtr/.vtp)
- hdf5_io: HDF5 time-series with chunked access
- registry: open_dataset(...) dispatch by file type
"""

from .registry import open_dataset
