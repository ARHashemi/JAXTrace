"""
Temporal Batching Field with On-Demand Loading

Loads velocity fields on-demand for temporal windows.
Handles variable mesh topology (AMR data).
"""

import numpy as np
import jax.numpy as jnp
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from typing import List, Tuple, Optional
from pathlib import Path
import glob

from .grid_hash_field import build_grid_hash_mesh, create_grid_hash_interpolator, GridHashMesh


class TemporalBatchingField:
    """
    Field with on-demand loading for temporal batching.

    Designed for AMR data where mesh topology changes per timestep.
    Uses grid hash for fast spatial queries.
    """

    def __init__(self,
                 data_pattern: str,
                 grid_resolution: int = 32,
                 cache_size: int = 3,
                 streaming: bool = True,
                 batch_size: int = 1000):
        """
        Initialize temporal batching field.

        Parameters
        ----------
        data_pattern : str
            Glob pattern for VTK files (e.g., "/path/data_*.pvtu")
        grid_resolution : int
            Grid hash resolution (default: 32)
        cache_size : int
            Number of timesteps to keep in cache (default: 3)
        streaming : bool
            If True, use CPU-based streaming interpolation (low memory)
            If False, use batched GPU interpolation (balanced memory/speed)
        batch_size : int
            Particles per GPU batch when streaming=False (default: 1000)
        """

        self.data_pattern = data_pattern
        self.grid_resolution = grid_resolution
        self.cache_size = cache_size
        self.streaming = streaming
        self.batch_size = batch_size

        # Find all files
        self.files = sorted(glob.glob(data_pattern))
        if not self.files:
            raise ValueError(f"No files found matching pattern: {data_pattern}")

        self.n_timesteps = len(self.files)

        print(f"📁 Temporal batching field initialized:")
        print(f"   Pattern: {data_pattern}")
        print(f"   Files found: {self.n_timesteps}")
        print(f"   Grid resolution: {grid_resolution}³ cells")
        print(f"   Cache size: {cache_size} timesteps")

        # Cache for loaded timesteps
        self._cache = {}  # {timestep_idx: GridHashMesh}
        self._cache_order = []  # LRU tracking

        # Store domain bounds (computed from first file)
        self._domain_bounds = None

    def load_timestep(self, timestep_idx: int) -> GridHashMesh:
        """
        Load single timestep and build grid hash.

        Parameters
        ----------
        timestep_idx : int
            Timestep index (0 to n_timesteps-1)

        Returns
        -------
        GridHashMesh
            Grid hash mesh for this timestep
        """

        # Check cache
        if timestep_idx in self._cache:
            return self._cache[timestep_idx]

        # Load VTK file
        filename = self.files[timestep_idx]

        reader = vtk.vtkXMLPUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        mesh = reader.GetOutput()

        # Extract data
        points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float32)

        # Get connectivity
        cells = mesh.GetCells()
        connectivity_vtk = vtk_to_numpy(cells.GetData())

        # Parse tetrahedral connectivity
        n_cells = mesh.GetNumberOfCells()
        connectivity_list = []
        idx = 0

        for _ in range(n_cells):
            n_points = connectivity_vtk[idx]
            if n_points == 4:  # Tetrahedron
                tet = connectivity_vtk[idx+1:idx+5]
                connectivity_list.append(tet)
            idx += n_points + 1

        connectivity = np.array(connectivity_list, dtype=np.int32)

        # Get velocity field
        point_data = mesh.GetPointData()

        # Try common velocity field names
        velocity_array = None
        for name in ['Velocity', 'velocity', 'VELOCITY', 'v', 'u']:
            velocity_array = point_data.GetArray(name)
            if velocity_array is not None:
                break

        if velocity_array is None:
            # Use first array with 3 components
            for i in range(point_data.GetNumberOfArrays()):
                arr = point_data.GetArray(i)
                if arr.GetNumberOfComponents() == 3:
                    velocity_array = arr
                    break

        if velocity_array is None:
            raise ValueError(f"No velocity field found in {filename}")

        velocity = vtk_to_numpy(velocity_array).astype(np.float32)

        # Build grid hash mesh
        grid_hash = build_grid_hash_mesh(
            points,
            connectivity,
            velocity,
            grid_resolution=self.grid_resolution
        )

        # Update cache
        self._cache[timestep_idx] = grid_hash
        self._cache_order.append(timestep_idx)

        # Remove oldest if cache full
        if len(self._cache) > self.cache_size:
            oldest = self._cache_order.pop(0)
            if oldest in self._cache:
                del self._cache[oldest]

        # Store domain bounds from first load
        if self._domain_bounds is None:
            self._domain_bounds = (
                grid_hash.grid_min,
                grid_hash.grid_max
            )

        return grid_hash

    def load_window(self, timestep_indices: List[int]) -> List[GridHashMesh]:
        """
        Load multiple timesteps for temporal window.

        Parameters
        ----------
        timestep_indices : List[int]
            List of timestep indices to load

        Returns
        -------
        List[GridHashMesh]
            List of grid hash meshes
        """

        meshes = []
        for idx in timestep_indices:
            if idx < 0 or idx >= self.n_timesteps:
                raise ValueError(f"Timestep index {idx} out of range [0, {self.n_timesteps})")
            meshes.append(self.load_timestep(idx))

        return meshes

    def get_spatial_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get spatial domain bounds.

        Returns
        -------
        bounds_min, bounds_max : np.ndarray
            Domain bounding box
        """

        if self._domain_bounds is None:
            # Load first timestep to get bounds
            self.load_timestep(0)

        return (
            np.array(self._domain_bounds[0]),
            np.array(self._domain_bounds[1])
        )

    def sample_at_positions_temporal(self,
                                    query_positions: np.ndarray,
                                    timestep_idx: int) -> jnp.ndarray:
        """
        Sample velocity at positions for specific timestep.

        Parameters
        ----------
        query_positions : np.ndarray
            Query positions (N, 3)
        timestep_idx : int
            Timestep index

        Returns
        -------
        jnp.ndarray
            Interpolated velocities (N, 3)
        """

        # Load timestep
        mesh = self.load_timestep(timestep_idx)

        # Create interpolator (pass streaming mode and batch size)
        interpolator = create_grid_hash_interpolator(
            mesh,
            streaming=self.streaming,
            batch_size=self.batch_size
        )

        # Interpolate
        query_positions_jax = jnp.asarray(query_positions, dtype=jnp.float32)
        velocities = interpolator(query_positions_jax)

        return velocities

    def interpolate_temporal(self,
                           query_positions: np.ndarray,
                           t_query: float,
                           dt_data: float) -> jnp.ndarray:
        """
        Interpolate velocity at positions with temporal interpolation.

        Uses linear interpolation between adjacent timesteps.

        Parameters
        ----------
        query_positions : np.ndarray
            Query positions (N, 3)
        t_query : float
            Query time
        dt_data : float
            Time interval between velocity field timesteps

        Returns
        -------
        jnp.ndarray
            Interpolated velocities (N, 3)
        """

        # Find bracketing timesteps
        t_idx_float = t_query / dt_data
        t_idx_left = int(np.floor(t_idx_float))
        t_idx_right = t_idx_left + 1

        # Handle boundaries
        t_idx_left = np.clip(t_idx_left, 0, self.n_timesteps - 1)
        t_idx_right = np.clip(t_idx_right, 0, self.n_timesteps - 1)

        # Interpolation weight
        alpha = t_idx_float - t_idx_left
        alpha = np.clip(alpha, 0.0, 1.0)

        # Sample at both timesteps
        v_left = self.sample_at_positions_temporal(query_positions, t_idx_left)

        if t_idx_right != t_idx_left:
            v_right = self.sample_at_positions_temporal(query_positions, t_idx_right)
            # Linear interpolation
            velocities = (1.0 - alpha) * v_left + alpha * v_right
        else:
            velocities = v_left

        return velocities

    def __repr__(self) -> str:
        return (
            f"TemporalBatchingField("
            f"n_timesteps={self.n_timesteps}, "
            f"grid_resolution={self.grid_resolution}, "
            f"cache_size={self.cache_size}, "
            f"cached={len(self._cache)})"
        )
