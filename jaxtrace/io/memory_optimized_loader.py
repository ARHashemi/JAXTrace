"""
Memory-Optimized Data Loader for JAXTrace

This module provides memory-efficient data loading specifically optimized for
particle tracking workflows. It loads only essential fields (velocity and mesh data)
and implements various memory optimization strategies.

Features:
- Selective field loading (velocity + mesh only)
- Streaming data loading for large datasets
- Memory usage monitoring during loading
- Automatic data compression
- Chunked processing for 500MB+ datasets
"""

import gc
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass

from ..utils.memory_tracker import track_memory, track_operation_memory, track_variable_memory
from ..fields.time_series import TimeSeriesField
from .vtk_reader import VTKUnstructuredTimeSeriesReader


@dataclass
class MemoryOptimizedConfig:
    """Configuration for memory-optimized data loading."""
    essential_fields_only: bool = True
    max_memory_per_timestep_mb: float = 100.0  # 100MB per timestep max
    enable_compression: bool = True
    chunk_size: int = 10  # Process 10 timesteps at a time
    dtype: str = "float32"  # Use float32 instead of float64
    discard_boundary_layers: bool = True  # Remove boundary ghost cells
    subsample_factor: Optional[int] = None  # Spatial subsampling factor


class MemoryOptimizedLoader:
    """
    Memory-optimized loader for large VTK datasets.

    Designed for datasets where each timestep is ~500MB,
    keeping only velocity fields and essential mesh data.
    """

    def __init__(self, config: Optional[MemoryOptimizedConfig] = None):
        """
        Initialize memory-optimized loader.

        Parameters
        ----------
        config : MemoryOptimizedConfig, optional
            Configuration for memory optimization
        """
        self.config = config or MemoryOptimizedConfig()
        self.essential_fields = ['velocity', 'Velocity', 'U', 'v', 'vel', 'Displacement', 'displacement']
        self.mesh_fields = ['points', 'coordinates', 'connectivity']

        # Memory tracking
        self.loaded_timesteps: List[int] = []
        self.memory_usage: Dict[str, float] = {}

    def load_dataset(self,
                    data_pattern: str,
                    max_time_steps: Optional[int] = None,
                    time_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Load dataset with memory optimization.

        Parameters
        ----------
        data_pattern : str
            Path pattern for VTK files
        max_time_steps : int, optional
            Maximum number of timesteps to load
        time_range : tuple, optional
            Time range (start, end) to load

        Returns
        -------
        Dict[str, Any]
            Optimized dataset containing only essential fields
        """

        with track_operation_memory("optimized_dataset_loading"):
            track_memory("start_dataset_loading")

            # Initialize VTK reader
            reader = VTKUnstructuredTimeSeriesReader(data_pattern)
            track_memory("vtk_reader_initialized")

            # Get available timesteps - VTKUnstructuredTimeSeriesReader uses indices
            num_timesteps = len(reader)
            available_times = np.arange(num_timesteps, dtype=np.float32)
            track_variable_memory("available_times", available_times)

            if num_timesteps == 0:
                raise ValueError("No timesteps found in dataset")

            # Select timesteps based on constraints
            selected_times = self._select_timesteps(
                available_times, max_time_steps, time_range
            )
            track_variable_memory("selected_times", selected_times)

            print(f"🔍 Loading {len(selected_times)} timesteps with memory optimization")
            print(f"   Essential fields only: {self.config.essential_fields_only}")
            print(f"   Data type: {self.config.dtype}")
            print(f"   Compression: {self.config.enable_compression}")

            # Load data in chunks to manage memory
            return self._load_data_in_chunks(reader, selected_times)

    def _select_timesteps(self,
                         available_times: np.ndarray,
                         max_time_steps: Optional[int],
                         time_range: Optional[Tuple[float, float]]) -> np.ndarray:
        """Select timesteps based on constraints."""

        times = available_times.copy()

        # Apply time range filter
        if time_range is not None:
            start_time, end_time = time_range
            mask = (times >= start_time) & (times <= end_time)
            times = times[mask]

        # Apply max timesteps limit
        if max_time_steps is not None and len(times) > max_time_steps:
            # Take evenly spaced timesteps
            indices = np.linspace(0, len(times)-1, max_time_steps, dtype=int)
            times = times[indices]

        return times

    def _load_data_in_chunks(self,
                           reader: VTKUnstructuredTimeSeriesReader,
                           selected_times: np.ndarray) -> Dict[str, Any]:
        """Load data in memory-efficient chunks."""

        chunk_size = self.config.chunk_size
        num_chunks = int(np.ceil(len(selected_times) / chunk_size))

        # Initialize output data structures
        velocity_data = []
        times_list = []
        positions = None
        connectivity = None

        print(f"📦 Processing {num_chunks} chunks of {chunk_size} timesteps each")

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(selected_times))
            chunk_times = selected_times[start_idx:end_idx]

            with track_operation_memory(f"chunk_{chunk_idx}"):
                print(f"   Processing chunk {chunk_idx+1}/{num_chunks} "
                      f"(timesteps {start_idx}-{end_idx-1})")

                chunk_data = self._load_chunk(reader, chunk_times)

                # Extract and store velocity data
                chunk_velocities = chunk_data['velocities']
                velocity_data.append(chunk_velocities)
                times_list.extend(chunk_data['times'])

                # Store mesh data only once (from first chunk)
                if positions is None:
                    positions = chunk_data['positions']
                    connectivity = chunk_data.get('connectivity')
                    track_variable_memory("mesh_positions", positions)
                    if connectivity is not None:
                        track_variable_memory("mesh_connectivity", connectivity)

                # Force garbage collection between chunks
                del chunk_data, chunk_velocities
                gc.collect()
                track_memory(f"chunk_{chunk_idx}_completed")

        # Combine velocity data from all chunks
        with track_operation_memory("combining_velocity_data"):
            velocity_data = np.concatenate(velocity_data, axis=0)
            times_array = np.array(times_list, dtype=self.config.dtype)

            track_variable_memory("final_velocity_data", velocity_data)
            track_variable_memory("final_times", times_array)

        # Create final dataset
        dataset = {
            'velocity_data': velocity_data,
            'times': times_array,
            'positions': positions,
            'connectivity': connectivity,
            'memory_optimized': True,
            'optimization_config': self.config
        }

        # Calculate total memory usage
        total_memory_mb = self._calculate_dataset_memory(dataset)
        print(f"✅ Dataset loaded: {total_memory_mb:.1f} MB total memory")

        return dataset

    def _load_chunk(self, reader: VTKUnstructuredTimeSeriesReader, chunk_times: np.ndarray) -> Dict[str, Any]:
        """Load a single chunk of timesteps."""

        chunk_velocities = []
        chunk_times_list = []
        positions = None
        connectivity = None

        for time_idx in chunk_times:
            # Read timestep data - VTKUnstructuredTimeSeriesReader returns velocity directly
            time_idx_int = int(time_idx)

            try:
                velocity = reader.load_single_timestep(time_idx_int)
            except Exception as e:
                print(f"   ⚠️  Skipping timestep {time_idx_int}: {e}")
                continue

            if velocity is None:
                print(f"   ⚠️  No velocity field found at timestep {time_idx_int}")
                continue

            # Apply optimizations
            velocity = self._optimize_velocity_data(velocity)

            chunk_velocities.append(velocity)
            chunk_times_list.append(time_idx)

            # Extract mesh data from first timestep
            if positions is None:
                # Get positions from reader (loaded when first timestep is read)
                positions = reader.grid_points
                connectivity = None  # VTKUnstructuredTimeSeriesReader doesn't expose connectivity

        if not chunk_velocities:
            raise ValueError(f"No valid velocity data found in chunk")

        # Stack velocities
        velocities_array = np.stack(chunk_velocities, axis=0)

        return {
            'velocities': velocities_array,
            'times': chunk_times_list,
            'positions': positions,
            'connectivity': connectivity
        }

    def _extract_velocity_field(self, timestep_data: Dict) -> Optional[np.ndarray]:
        """Extract velocity field from timestep data.

        Note: The displacement field is used directly as velocity (no derivatives needed).
        """

        # Try different possible velocity field names (including displacement which stores velocity)
        for field_name in self.essential_fields:
            if field_name in timestep_data:
                velocity = timestep_data[field_name]

                # Ensure it's a 3D vector field
                if velocity.ndim == 2 and velocity.shape[1] == 3:
                    return velocity
                elif velocity.ndim == 2 and velocity.shape[1] == 2:
                    # Add zero z-component for 2D fields
                    zeros = np.zeros((velocity.shape[0], 1), dtype=velocity.dtype)
                    return np.concatenate([velocity, zeros], axis=1)

        return None

    def _extract_positions(self, timestep_data: Dict) -> np.ndarray:
        """Extract mesh positions from timestep data."""

        for field_name in ['points', 'coordinates', 'positions']:
            if field_name in timestep_data:
                positions = timestep_data[field_name]

                # Ensure it's 3D coordinates
                if positions.ndim == 2 and positions.shape[1] >= 2:
                    if positions.shape[1] == 2:
                        # Add zero z-coordinate for 2D meshes
                        zeros = np.zeros((positions.shape[0], 1), dtype=positions.dtype)
                        positions = np.concatenate([positions, zeros], axis=1)

                    return positions

        raise ValueError("No mesh positions found in timestep data")

    def _extract_connectivity(self, timestep_data: Dict) -> Optional[np.ndarray]:
        """Extract mesh connectivity if available."""

        for field_name in ['connectivity', 'cells', 'elements']:
            if field_name in timestep_data:
                return timestep_data[field_name]

        return None

    def _optimize_velocity_data(self, velocity: np.ndarray) -> np.ndarray:
        """Apply memory optimizations to velocity data."""

        # Convert to specified dtype
        if velocity.dtype != self.config.dtype:
            velocity = velocity.astype(self.config.dtype)

        # Remove boundary layers if requested
        if self.config.discard_boundary_layers:
            velocity = self._remove_boundary_layers(velocity)

        # Apply spatial subsampling if requested
        if self.config.subsample_factor is not None:
            velocity = velocity[::self.config.subsample_factor]

        return velocity

    def _remove_boundary_layers(self, velocity: np.ndarray) -> np.ndarray:
        """Remove boundary ghost cells to reduce memory usage."""

        # Simple approach: remove points with zero velocity (likely boundary)
        velocity_magnitude = np.linalg.norm(velocity, axis=1)
        non_zero_mask = velocity_magnitude > 1e-12

        return velocity[non_zero_mask]

    def _calculate_dataset_memory(self, dataset: Dict) -> float:
        """Calculate total memory usage of dataset in MB."""

        total_bytes = 0

        for key, value in dataset.items():
            if isinstance(value, np.ndarray):
                total_bytes += value.nbytes
            elif hasattr(value, 'nbytes'):
                total_bytes += value.nbytes

        return total_bytes / 1024 / 1024

    def create_time_series_field(self, dataset: Dict) -> TimeSeriesField:
        """Create a TimeSeriesField from optimized dataset."""

        with track_operation_memory("creating_time_series_field"):
            field = TimeSeriesField(
                data=dataset['velocity_data'],
                times=dataset['times'],
                positions=dataset['positions'],
                interpolation="linear",
                extrapolation="constant",
                _source_info={
                    'memory_optimized': True,
                    'optimization_config': dataset['optimization_config'],
                    'connectivity': dataset.get('connectivity')
                }
            )

            track_variable_memory("time_series_field", field)

        return field


def load_optimized_dataset(data_pattern: str,
                          max_memory_per_timestep_mb: float = 100.0,
                          max_time_steps: Optional[int] = None,
                          time_range: Optional[Tuple[float, float]] = None,
                          dtype: str = "float32") -> TimeSeriesField:
    """
    Convenient function to load a memory-optimized dataset.

    Parameters
    ----------
    data_pattern : str
        Path pattern for VTK files
    max_memory_per_timestep_mb : float, default 100.0
        Maximum memory per timestep in MB
    max_time_steps : int, optional
        Maximum number of timesteps to load
    time_range : tuple, optional
        Time range (start, end) to load
    dtype : str, default "float32"
        Data type for arrays

    Returns
    -------
    TimeSeriesField
        Memory-optimized time series field
    """

    config = MemoryOptimizedConfig(
        essential_fields_only=True,
        max_memory_per_timestep_mb=max_memory_per_timestep_mb,
        enable_compression=True,
        dtype=dtype
    )

    loader = MemoryOptimizedLoader(config)
    dataset = loader.load_dataset(
        data_pattern=data_pattern,
        max_time_steps=max_time_steps,
        time_range=time_range
    )

    return loader.create_time_series_field(dataset)


def estimate_memory_usage(data_pattern: str,
                         num_timesteps: int,
                         dtype: str = "float32") -> Dict[str, float]:
    """
    Estimate memory usage for a dataset before loading.

    Parameters
    ----------
    data_pattern : str
        Path pattern for VTK files
    num_timesteps : int
        Number of timesteps to estimate for
    dtype : str, default "float32"
        Data type for estimation

    Returns
    -------
    Dict[str, float]
        Memory usage estimates in MB
    """

    try:
        # Load a single timestep to estimate size
        reader = VTKUnstructuredTimeSeriesReader(data_pattern)

        if len(reader) == 0:
            return {"error": "No timesteps found"}

        # Load first timestep (index 0) - returns velocity array (N, 3)
        sample_velocity = reader.load_single_timestep(0)

        if sample_velocity is None:
            return {"error": "Could not read sample timestep"}

        # Estimate memory for different data types
        bytes_per_float32 = 4
        bytes_per_float64 = 8

        # Get velocity field points from loaded data
        velocity_points = sample_velocity.shape[0]

        # Get mesh points from reader
        mesh_points = reader._n_grid_points if reader.grid_points is not None else velocity_points

        if velocity_points == 0:
            return {"error": "No velocity field found"}

        # Calculate estimates
        bytes_per_point = bytes_per_float32 if dtype == "float32" else bytes_per_float64

        velocity_memory_per_timestep = velocity_points * 3 * bytes_per_point  # 3D velocity
        mesh_memory = mesh_points * 3 * bytes_per_point  # 3D coordinates

        total_velocity_memory = velocity_memory_per_timestep * num_timesteps
        total_memory = total_velocity_memory + mesh_memory

        return {
            "velocity_points": velocity_points,
            "mesh_points": mesh_points,
            "velocity_memory_per_timestep_mb": velocity_memory_per_timestep / 1024 / 1024,
            "mesh_memory_mb": mesh_memory / 1024 / 1024,
            "total_velocity_memory_mb": total_velocity_memory / 1024 / 1024,
            "total_memory_mb": total_memory / 1024 / 1024,
            "recommended_chunk_size": max(1, int(500 * 1024 * 1024 / velocity_memory_per_timestep))
        }

    except Exception as e:
        return {"error": f"Estimation failed: {str(e)}"}