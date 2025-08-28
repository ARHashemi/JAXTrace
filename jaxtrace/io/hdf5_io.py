"""
HDF5 I/O: chunked datasets for large velocity fields and trajectories.

- H5Series: load time slices by index with minimal memory
- Trajectory writing utilities

Optional dependency: h5py
"""

from __future__ import annotations
import os
import warnings
from typing import Optional, Tuple, Sequence, Union, Iterable, List, Dict

import numpy as np

try:
    import h5py
    HDF5_AVAILABLE = True
except Exception:
    HDF5_AVAILABLE = False
    warnings.warn("h5py not available. HDF5 I/O will be disabled")


class H5Series:
    """
    HDF5 time series for velocity fields.

    Assumes a dataset layout like:
      /velocity/t0000  (Nx, Ny, Nz, C)
      /velocity/t0001
    or a single 5D dataset:
      /velocity  (T, Nx, Ny, Nz, C)

    You can pass either a filename or a list of filenames (one per time slice).
    """

    def __init__(self, spec: Union[str, Iterable[str]], dataset: str = "velocity"):
        if not HDF5_AVAILABLE:
            raise RuntimeError("h5py not available; cannot open HDF5 datasets")

        if isinstance(spec, (list, tuple)):
            self._files = list(spec)
            self._multi_file = True
        elif isinstance(spec, str):
            self._files = [spec]
            self._multi_file = False
        else:
            raise TypeError("spec must be a filename or list of filenames")

        self._dataset = dataset
        self._length = self._infer_length()

    def _infer_length(self) -> int:
        if self._multi_file:
            return len(self._files)
        # Single file: detect 5D dataset or group of slices
        fname = self._files[0]
        with h5py.File(fname, "r") as f:
            if self._dataset in f:
                d = f[self._dataset]
                if d.ndim == 5:
                    return int(d.shape[0])
                else:
                    raise ValueError("Expected /velocity to be 5D (T,Nx,Ny,Nz,C)")
            elif "velocity" in f and isinstance(f["velocity"], h5py.Group):
                grp = f["velocity"]
                keys = sorted(k for k in grp.keys() if k.startswith("t"))
                return len(keys)
            else:
                raise ValueError("Cannot infer series length; dataset not found")

    def __len__(self) -> int:
        return self._length

    def load_slice(self, i: int) -> np.ndarray:
        """
        Load time slice i as numpy array shaped (Nx, Ny, Nz, C).
        """
        if self._multi_file:
            fname = self._files[i]
            with h5py.File(fname, "r") as f:
                if self._dataset in f:
                    arr = np.asarray(f[self._dataset][...])
                else:
                    raise ValueError(f"Dataset '{self._dataset}' not found in {fname}")
            return arr

        fname = self._files[0]
        with h5py.File(fname, "r") as f:
            if self._dataset in f:
                d = f[self._dataset]
                if d.ndim != 5:
                    raise ValueError("Expected 5D dataset (T,Nx,Ny,Nz,C)")
                arr = np.asarray(d[i, ...])
                return arr
            elif "velocity" in f and isinstance(f["velocity"], h5py.Group):
                grp = f["velocity"]
                key = f"t{i:04d}"
                if key not in grp:
                    raise KeyError(f"Slice '{key}' not found in group '/velocity'")
                arr = np.asarray(grp[key][...])
                return arr
            else:
                raise ValueError(f"Dataset '{self._dataset}' not found")

    # --------------------------
    # Trajectory I/O utilities
    # --------------------------

    @staticmethod
    def write_trajectories(
        path: str,
        positions_over_time: Union[np.ndarray, Sequence[np.ndarray]],
        ids: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
        group: str = "trajectories",
        chunks: Optional[Tuple[int, int, int]] = None,
        compression: Optional[str] = "gzip",
        compression_opts: int = 4
    ) -> str:
        """
        Write trajectories to HDF5.

        positions_over_time:
            - (T, N, 3) array or sequence of (N,3) arrays
        ids: (N,) optional particle ids
        times: (T,) optional time stamps
        """
        if not HDF5_AVAILABLE:
            raise RuntimeError("h5py not available; cannot write HDF5 files")

        if isinstance(positions_over_time, np.ndarray):
            if positions_over_time.ndim != 3 or positions_over_time.shape[-1] != 3:
                raise ValueError("positions_over_time must be (T, N, 3)")
            T, N, _ = positions_over_time.shape
        else:
            seq = [np.asarray(p) for p in positions_over_time]
            T = len(seq)
            N = seq[0].shape[0]
            positions_over_time = np.stack(seq, axis=0)

        with h5py.File(path, "w") as f:
            grp = f.require_group(group)
            ds_pos = grp.create_dataset(
                "positions",
                shape=(T, N, 3),
                dtype="f4",
                data=positions_over_time.astype("f4"),
                chunks=chunks if chunks is not None else (1, min(N, 65536), 3),
                compression=compression,
                compression_opts=compression_opts,
            )
            if ids is not None:
                grp.create_dataset("ids", data=np.asarray(ids, dtype=np.int64))
            if times is not None:
                grp.create_dataset("times", data=np.asarray(times, dtype=np.float64))
        return path

    @staticmethod
    def append_trajectories(
        path: str,
        new_positions: np.ndarray,
        group: str = "trajectories"
    ) -> None:
        """
        Append new time steps to an existing trajectories dataset.

        new_positions: (T_new, N, 3)
        """
        if not HDF5_AVAILABLE:
            raise RuntimeError("h5py not available; cannot write HDF5 files")

        new_positions = np.asarray(new_positions)
        if new_positions.ndim != 3 or new_positions.shape[-1] != 3:
            raise ValueError("new_positions must be (T_new, N, 3)")

        with h5py.File(path, "a") as f:
            grp = f[group]
            ds = grp["positions"]
            T_old = ds.shape[0]
            T_new = new_positions.shape[0]
            ds.resize((T_old + T_new, ds.shape[1], 3))
            ds[T_old:T_old + T_new, :, :] = new_positions.astype("f4")
