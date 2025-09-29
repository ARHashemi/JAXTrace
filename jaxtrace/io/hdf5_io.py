# jaxtrace/io/hdf5_io.py
"""
HDF5 I/O utilities for chunked time-series data and trajectory storage.

Supports reading HDF5 time series with 5D datasets (T,Nx,Ny,Nz,C) or
groups of per-time datasets, and writing particle trajectories with
chunking and compression.
"""

from __future__ import annotations
import warnings
from typing import Optional, Tuple, Sequence, Union, Iterable, List
import numpy as np

try:
    import h5py
    HDF5_AVAILABLE = True
except Exception:
    HDF5_AVAILABLE = False
    warnings.warn("h5py not available. HDF5 I/O will be disabled", stacklevel=2)


class H5Series:
    """
    HDF5 time series reader for velocity fields.

    Supports multiple HDF5 layouts:
    1. Single 5D dataset: /dataset (T, Nx, Ny, Nz, C)
    2. Group of per-time datasets: /dataset/t0000, /dataset/t0001, ... each (Nx, Ny, Nz, C)
    3. Multi-file series: list of files, each containing a 4D dataset at /dataset
    """
    
    def __init__(self, spec: Union[str, Iterable[str]], dataset: str = "velocity"):
        """
        Initialize HDF5 series reader.
        
        Parameters
        ----------
        spec : str or Iterable[str]
            File specification:
            - Single file: "data.h5"
            - File list: ["data_001.h5", "data_002.h5", ...]
        dataset : str
            Dataset path within HDF5 files
        """
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
        """Infer number of time steps from file structure."""
        if self._multi_file:
            return len(self._files)
        
        # Single file - check internal structure
        fname = self._files[0]
        if not fname:
            raise ValueError("Empty filename provided")
            
        try:
            with h5py.File(fname, "r") as f:
                if self._dataset not in f:
                    raise ValueError(f"Dataset '{self._dataset}' not found in {fname}")
                
                obj = f[self._dataset]
                
                if isinstance(obj, h5py.Dataset):
                    # Single 5D dataset (T, Nx, Ny, Nz, C)
                    if obj.ndim == 5:
                        return int(obj.shape[0])
                    else:
                        raise ValueError(
                            f"Expected 5D dataset (T,Nx,Ny,Nz,C) at '/{self._dataset}', "
                            f"got {obj.ndim}D"
                        )
                        
                elif isinstance(obj, h5py.Group):
                    # Group with time-step datasets
                    keys = sorted(k for k in obj.keys() if k.startswith("t"))
                    if not keys:
                        raise ValueError(f"No timestep datasets (t****) found in group '/{self._dataset}'")
                    return len(keys)
                else:
                    raise ValueError(f"Unsupported object type at '/{self._dataset}'")
                    
        except Exception as e:
            if isinstance(e, (ValueError, KeyError)):
                raise
            else:
                raise RuntimeError(f"Failed to read HDF5 file {fname}: {e}") from e

    def __len__(self) -> int:
        """Number of time steps in the series."""
        return self._length

    def load_slice(self, i: int) -> np.ndarray:
        """
        Load time slice i as numpy array.
        
        Parameters
        ----------
        i : int
            Time index (0 to len(self)-1)
            
        Returns
        -------
        np.ndarray
            Grid data array, shape (Nx, Ny, Nz, C)
        """
        if not (0 <= i < self._length):
            raise IndexError(f"Time index {i} out of range [0, {self._length})")
        
        if self._multi_file:
            # Multi-file: each file contains one time step
            fname = self._files[i]
            try:
                with h5py.File(fname, "r") as f:
                    if self._dataset in f:
                        arr = np.asarray(f[self._dataset][...])
                    else:
                        raise ValueError(f"Dataset '{self._dataset}' not found in {fname}")
            except Exception as e:
                raise RuntimeError(f"Failed to read {fname}: {e}") from e
            
            return arr

        # Single file with multiple time steps
        fname = self._files[0]
        try:
            with h5py.File(fname, "r") as f:
                obj = f[self._dataset]
                
                if isinstance(obj, h5py.Dataset):
                    # 5D dataset: extract time slice
                    if obj.ndim != 5:
                        raise ValueError(f"Expected 5D dataset, got {obj.ndim}D")
                    arr = np.asarray(obj[i, ...])
                    return arr
                    
                elif isinstance(obj, h5py.Group):
                    # Group: load specific timestep dataset
                    key = f"t{i:04d}"
                    if key not in obj:
                        # Try alternative naming schemes
                        alt_keys = [f"t{i:03d}", f"t{i:05d}", f"step_{i}", str(i)]
                        key = next((k for k in alt_keys if k in obj), None)
                        
                        if key is None:
                            available_keys = sorted(k for k in obj.keys() if k.startswith("t"))
                            raise KeyError(
                                f"Timestep {i} not found in group '/{self._dataset}'. "
                                f"Available keys: {available_keys[:5]}{'...' if len(available_keys) > 5 else ''}"
                            )
                    
                    arr = np.asarray(obj[key][...])
                    return arr
                else:
                    raise ValueError(f"Unsupported object type at '/{self._dataset}'")
                    
        except Exception as e:
            if isinstance(e, (ValueError, KeyError, IndexError)):
                raise
            else:
                raise RuntimeError(f"Failed to read {fname}: {e}") from e

    def load_timestep(self, index: int) -> np.ndarray:
        """Alias for load_slice() to match TimeSeriesReader protocol."""
        return self.load_slice(index)

    def get_times(self) -> List[float]:
        """Get time values (placeholder - HDF5 files may or may not store time info)."""
        # Could be enhanced to read actual time values from metadata
        return list(range(len(self)))

    def close(self) -> None:
        """Close the dataset (no-op for H5Series since we open/close per access)."""
        pass

    # ---------- Static Methods for Writing ----------

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
        Write particle trajectories to HDF5 file.
        
        Parameters
        ----------
        path : str
            Output filename
        positions_over_time : np.ndarray or Sequence[np.ndarray]
            Trajectory data, shape (T, N, 3) or sequence of (N, 3) arrays
        ids : np.ndarray, optional
            Particle IDs, shape (N,)
        times : np.ndarray, optional
            Time values, shape (T,)
        group : str
            HDF5 group name for trajectory data
        chunks : tuple, optional
            Chunk shape (T_chunk, N_chunk, 3). If None, auto-determined.
        compression : str, optional
            Compression algorithm ('gzip', 'lzf', 'szip')
        compression_opts : int
            Compression level (0-9 for gzip)
            
        Returns
        -------
        str
            Path to written file
        """
        if not HDF5_AVAILABLE:
            raise RuntimeError("h5py not available; cannot write HDF5 files")
        
        # Convert input to (T, N, 3) array
        if isinstance(positions_over_time, np.ndarray):
            if positions_over_time.ndim != 3 or positions_over_time.shape[-1] != 3:
                raise ValueError("positions_over_time must be (T, N, 3)")
            traj_data = positions_over_time
        else:
            # Sequence of arrays - stack them
            seq = [np.asarray(p) for p in positions_over_time]
            if not seq:
                raise ValueError("Empty sequence provided")
            
            # Validate shapes
            N = seq[0].shape[0]
            if seq[0].ndim != 2 or seq[0].shape[1] != 3:
                raise ValueError("Each position array must be (N, 3)")
            
            for i, p in enumerate(seq[1:], 1):
                if p.shape != (N, 3):
                    raise ValueError(f"Position array {i} has shape {p.shape}, expected ({N}, 3)")
            
            traj_data = np.stack(seq, axis=0)
        
        T, N, _ = traj_data.shape
        
        # Auto-determine chunk size if not provided
        if chunks is None:
            # Use reasonable chunk sizes: ~1MB chunks, but ensure at least 1 timestep
            chunk_t = max(1, min(T, 64))  # 1-64 timesteps per chunk
            chunk_n = max(1, min(N, 65536))  # Up to 64k particles per chunk
            chunks = (chunk_t, chunk_n, 3)
        
        try:
            with h5py.File(path, "w") as f:
                grp = f.require_group(group)
                
                # Write main positions dataset
                ds_pos = grp.create_dataset(
                    "positions",
                    shape=(T, N, 3),
                    maxshape=(None, N, 3),  # Allow time extension
                    dtype="f4",
                    data=traj_data.astype("f4"),
                    chunks=chunks,
                    compression=compression,
                    compression_opts=compression_opts,
                )
                
                # Add metadata
                ds_pos.attrs["description"] = "Particle trajectories"
                ds_pos.attrs["units"] = "length units (unspecified)"
                
                # Write particle IDs if provided
                if ids is not None:
                    ids_array = np.asarray(ids, dtype=np.int64)
                    if len(ids_array) != N:
                        raise ValueError(f"ids length {len(ids_array)} != N={N}")
                    grp.create_dataset("ids", data=ids_array)
                
                # Write time values if provided
                if times is not None:
                    times_array = np.asarray(times, dtype=np.float64)
                    if len(times_array) != T:
                        raise ValueError(f"times length {len(times_array)} != T={T}")
                    grp.create_dataset("times", data=times_array)
                
                # Add global metadata
                grp.attrs["num_particles"] = N
                grp.attrs["num_timesteps"] = T
                grp.attrs["format_version"] = "1.0"
                
        except Exception as e:
            raise RuntimeError(f"Failed to write HDF5 file {path}: {e}") from e
        
        return path

    @staticmethod
    def append_trajectories(
        path: str,
        new_positions: np.ndarray,
        group: str = "trajectories"
    ) -> None:
        """
        Append new trajectory data to existing HDF5 file.
        
        Parameters
        ----------
        path : str
            Existing HDF5 filename
        new_positions : np.ndarray
            New trajectory data to append, shape (T_new, N, 3)
        group : str
            HDF5 group name containing trajectory data
        """
        if not HDF5_AVAILABLE:
            raise RuntimeError("h5py not available; cannot write HDF5 files")
        
        new_positions = np.asarray(new_positions)
        if new_positions.ndim != 3 or new_positions.shape[-1] != 3:
            raise ValueError("new_positions must be (T_new, N, 3)")
        
        try:
            with h5py.File(path, "a") as f:
                if group not in f:
                    raise ValueError(f"Group '{group}' not found in {path}")
                
                grp = f[group]
                if "positions" not in grp:
                    raise ValueError(f"Dataset 'positions' not found in group '{group}'")
                
                ds = grp["positions"]
                
                # Check compatibility
                if ds.shape[1:] != new_positions.shape[1:]:
                    raise ValueError(
                        f"Shape mismatch: existing {ds.shape[1:]} vs new {new_positions.shape[1:]}"
                    )
                
                # Check if dataset is resizable
                if ds.maxshape[0] is not None and ds.maxshape[0] < ds.shape[0] + new_positions.shape[0]:
                    raise RuntimeError(
                        "Dataset not resizable along time dimension. "
                        "Was it created with maxshape=(None, N, 3)?"
                    )
                
                # Resize and append
                T_old = ds.shape[0]
                T_new = new_positions.shape[0]
                ds.resize((T_old + T_new, ds.shape[1], 3))
                ds[T_old:T_old + T_new, :, :] = new_positions.astype("f4")
                
                # Update metadata
                grp.attrs["num_timesteps"] = T_old + T_new
                
        except Exception as e:
            raise RuntimeError(f"Failed to append to HDF5 file {path}: {e}") from e


# Backward compatibility aliases  
HDF5TimeSeriesReader = H5Series
HDF5TimeSeriesWriter = H5Series.write_trajectories
create_hdf5_timeseries = H5Series.write_trajectories
read_hdf5_timeseries = H5Series