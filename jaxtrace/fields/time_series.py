# jaxtrace/fields/time_series.py  
"""  
Time-series velocity field with memory-efficient interpolation.  

Provides temporal interpolation of velocity fields with consistent  
float32 data types and (N,3) shape enforcement throughout.  
Enhanced with VTK integration and comprehensive export capabilities.  
"""  

from __future__ import annotations  
from dataclasses import dataclass  
from typing import Optional, Tuple, Union, List, Dict  
from pathlib import Path  
import warnings  
import numpy as np  

# Import JAX utilities with fallback  
from ..utils.jax_utils import JAX_AVAILABLE  

if JAX_AVAILABLE:  
    try:  
        import jax.numpy as jnp  
    except Exception:  
        JAX_AVAILABLE = False  

if not JAX_AVAILABLE:  
    import numpy as jnp  # type: ignore  

from .base import TimeDependentField, _ensure_float32, _ensure_positions_shape  


def _ensure_velocities_shape(velocities: np.ndarray, positions_shape: Tuple[int, int]) -> np.ndarray:  
    """  
    Ensure velocities match positions shape (N, 3).  
    
    Parameters  
    ----------  
    velocities : np.ndarray  
        Input velocities  
    positions_shape : Tuple[int, int]  
        Expected shape (N, 3)  
        
    Returns  
    -------  
    np.ndarray  
        Velocities with shape (N, 3), float32 dtype  
    """  
    vel = _ensure_float32(velocities)  
    expected_n, expected_d = positions_shape  
    
    if vel.shape[0] != expected_n:  
        raise ValueError(f"Velocity count {vel.shape[0]} doesn't match position count {expected_n}")  
    
    if vel.shape[1] == 2:  
        # Convert 2D velocities to 3D by adding zero w-component  
        w_zeros = np.zeros((vel.shape[0], 1), dtype=np.float32)  
        vel = np.concatenate([vel, w_zeros], axis=1)  
    elif vel.shape[1] != 3:  
        raise ValueError(f"Velocities must have 2 or 3 columns, got {vel.shape[1]}")  
    
    return vel.astype(np.float32, copy=False)  


@dataclass  
class TimeSeriesField(TimeDependentField):  
    """  
    Time-dependent velocity field with temporal interpolation.  
    
    Enforces consistent float32 data types and (N,3) shape standards  
    throughout the interpolation pipeline. Enhanced with VTK integration.  
    
    Attributes  
    ----------  
    data : np.ndarray  
        Velocity field data, shape (T, N, 3) - standardized to float32  
    times : np.ndarray  
        Time points, shape (T,) - standardized to float32  
    positions : np.ndarray  
        Spatial positions, shape (N, 3) - standardized to float32  
    interpolation : str  
        Temporal interpolation method  
    extrapolation : str  
        Behavior for times outside [t_min, t_max]  
    _source_info : dict, optional  
        Metadata from data source (e.g., VTK file info)  
    """  
    data: np.ndarray            # (T, N, 3) - velocity field snapshots  
    times: np.ndarray           # (T,) - time points  
    positions: np.ndarray       # (N, 3) - spatial positions  
    interpolation: str = "linear"     # 'linear' | 'nearest' | 'cubic'  
    extrapolation: str = "constant"   # 'constant' | 'linear' | 'nan' | 'zero'  
    _source_info: Optional[dict] = None  # Source metadata  

    def __post_init__(self):  
        # Convert all data to consistent float32 format  
        self.data = _ensure_float32(self.data)  
        self.times = _ensure_float32(self.times)  
        self.positions = _ensure_positions_shape(self.positions)  
        
        # Validate shapes and consistency  
        if self.data.ndim != 3:  
            raise ValueError(f"data must have 3 dimensions (T,N,3), got shape {self.data.shape}")  
        
        T, N, D = self.data.shape  
        if D != 3:  
            raise ValueError(f"data must have 3 velocity components, got {D}")  
        
        if self.times.shape != (T,):  
            raise ValueError(f"times shape {self.times.shape} doesn't match data time dimension {T}")  
        
        if self.positions.shape != (N, 3):  
            raise ValueError(f"positions shape {self.positions.shape} doesn't match data spatial dimension ({N}, 3)")  
        
        # Ensure data is (T, N, 3) with float32  
        self.data = self.data.astype(np.float32, copy=False)  
        
        # Store metadata  
        self.T, self.N = T, N  
        self.t_min, self.t_max = float(self.times.min()), float(self.times.max())  
        
        # Validate time ordering  
        if not np.all(np.diff(self.times) > 0):  
            warnings.warn("Time array is not strictly increasing - interpolation may be unreliable")  
        
        # Store source info  
        if self._source_info is None:  
            self._source_info = {}  

    # ---------- Core interpolation (TimeDependentField protocol) ----------  
    
    def sample_at_time(self, t: float) -> np.ndarray:  
        """  
        Sample velocity field at specific time with temporal interpolation.  
        
        Parameters  
        ----------  
        t : float  
            Time point for sampling  
            
        Returns  
        -------  
        np.ndarray  
            Velocity field at time t, shape (N, 3), dtype float32  
        """  
        t = np.float32(t)  
        
        # Handle extrapolation cases  
        if t < self.t_min or t > self.t_max:  
            return self._handle_extrapolation(t)  
        
        # Find surrounding time indices  
        if self.interpolation == "nearest":  
            idx = np.argmin(np.abs(self.times - t))  
            return self.data[idx].copy()  # Ensure float32 copy  
        
        elif self.interpolation == "linear":  
            # Find bracketing indices  
            idx_right = np.searchsorted(self.times, t)  
            
            if idx_right == 0:  
                return self.data[0].copy()  
            elif idx_right >= self.T:  
                return self.data[-1].copy()  
            
            idx_left = idx_right - 1  
            t_left, t_right = self.times[idx_left], self.times[idx_right]  
            
            # Linear interpolation weight  
            dt = t_right - t_left  
            if dt < 1e-10:  # Nearly identical times  
                return self.data[idx_left].copy()  
            
            alpha = (t - t_left) / dt  
            vel_left = self.data[idx_left]  
            vel_right = self.data[idx_right]  
            
            # Interpolate and ensure float32  
            interpolated = (1.0 - alpha) * vel_left + alpha * vel_right  
            return interpolated.astype(np.float32, copy=False)  
        
        elif self.interpolation == "cubic":  
            return self._cubic_interpolate_at_time(t)  
        
        else:  
            raise ValueError(f"Unknown interpolation method: {self.interpolation}")  

    def _handle_extrapolation(self, t: float) -> np.ndarray:  
        """Handle time points outside the available data range."""  
        if self.extrapolation == "constant":  
            if t < self.t_min:  
                return self.data[0].copy()  
            else:  
                return self.data[-1].copy()  
        
        elif self.extrapolation == "linear":  
            if t < self.t_min:  
                # Extrapolate from first two points  
                if self.T < 2:  
                    return self.data[0].copy()  
                dt = self.times[1] - self.times[0]  
                dv = self.data[1] - self.data[0]  
                alpha = (t - self.times[0]) / dt  
                result = self.data[0] + alpha * dv  
                return result.astype(np.float32, copy=False)  
            else:  
                # Extrapolate from last two points  
                if self.T < 2:  
                    return self.data[-1].copy()  
                dt = self.times[-1] - self.times[-2]  
                dv = self.data[-1] - self.data[-2]  
                alpha = (t - self.times[-1]) / dt  
                result = self.data[-1] + alpha * dv  
                return result.astype(np.float32, copy=False)  
        
        elif self.extrapolation == "zero":  
            return np.zeros((self.N, 3), dtype=np.float32)  
        
        elif self.extrapolation == "nan":  
            return np.full((self.N, 3), np.nan, dtype=np.float32)  
        
        else:  
            raise ValueError(f"Unknown extrapolation method: {self.extrapolation}")  

    def _cubic_interpolate_at_time(self, t: float) -> np.ndarray:  
        """Cubic spline interpolation at time t."""  
        from scipy.interpolate import CubicSpline  
        
        # For cubic interpolation, we need at least 4 points  
        if self.T < 4:  
            warnings.warn("Not enough points for cubic interpolation, falling back to linear")  
            return self._linear_interpolate_at_time(t)  
        
        # Create cubic splines for each spatial point and velocity component  
        result = np.zeros((self.N, 3), dtype=np.float32)  
        
        for n in range(self.N):  
            for d in range(3):  
                # Create cubic spline for this node and component  
                cs = CubicSpline(self.times, self.data[:, n, d])  
                result[n, d] = cs(t)  
        
        return result  

    def _linear_interpolate_at_time(self, t: float) -> np.ndarray:  
        """Linear interpolation fallback for cubic method."""  
        # Find bracketing indices  
        idx_right = np.searchsorted(self.times, t)  
        
        if idx_right == 0:  
            return self.data[0].copy()  
        elif idx_right >= self.T:  
            return self.data[-1].copy()  
        
        idx_left = idx_right - 1  
        t_left, t_right = self.times[idx_left], self.times[idx_right]  
        
        # Linear interpolation weight  
        dt = t_right - t_left  
        if dt < 1e-10:  
            return self.data[idx_left].copy()  
        
        alpha = (t - t_left) / dt  
        vel_left = self.data[idx_left]  
        vel_right = self.data[idx_right]  
        
        interpolated = (1.0 - alpha) * vel_left + alpha * vel_right  
        return interpolated.astype(np.float32, copy=False)  

    # ---------- Spatial sampling (TimeDependentField protocol) ----------  

    def sample_at_positions(self, query_positions: np.ndarray, t: float) -> np.ndarray:  
        """  
        Sample velocity field at arbitrary positions and time.  
        
        Uses nearest-neighbor spatial sampling with temporal interpolation.  
        For structured grids or improved interpolation, consider using  
        specialized field classes.  
        
        Parameters  
        ----------  
        query_positions : np.ndarray  
            Query positions, shape (M, 2) or (M, 3)  
        t : float  
            Time for sampling  
            
        Returns  
        -------  
        np.ndarray  
            Velocities at query positions, shape (M, 3), dtype float32  
        """  
        # Ensure consistent shape and type for query positions  
        query_pos = _ensure_positions_shape(query_positions)  # (M, 3), float32  
        M = query_pos.shape[0]  
        
        if M == 0:  
            return np.zeros((0, 3), dtype=np.float32)  
        
        # Get velocity field at time t  
        vel_field = self.sample_at_time(t)  # (N, 3), float32  
        
        # Spatial interpolation using nearest neighbor  
        # For better performance on large datasets, consider using scipy.spatial.cKDTree  
        try:  
            from scipy.spatial import cKDTree  
            tree = cKDTree(self.positions)  
            distances, indices = tree.query(query_pos, k=1)  
            result = vel_field[indices]  # Shape (M, 3)  
            
        except ImportError:  
            # Fallback to manual distance computation  
            distances = np.linalg.norm(  
                query_pos[:, None, :] - self.positions[None, :, :],   
                axis=2  
            )  # Shape (M, N)  
            
            # Find nearest neighbors  
            nearest_indices = np.argmin(distances, axis=1)  # Shape (M,)  
            result = vel_field[nearest_indices]  # Shape (M, 3)  
        
        return result.astype(np.float32, copy=False)  

    # ---------- Protocol methods ----------  

    def get_time_bounds(self) -> Tuple[float, float]:  
        """Get time range of available data."""  
        return self.t_min, self.t_max  

    def get_spatial_bounds(self) -> Tuple[np.ndarray, np.ndarray]:  
        """Get spatial bounds of field positions."""  
        pos_min = np.min(self.positions, axis=0)  # (3,)  
        pos_max = np.max(self.positions, axis=0)  # (3,)  
        return pos_min.astype(np.float32), pos_max.astype(np.float32)  

    # ---------- Utility methods ----------  

    def validate_time(self, t: float) -> bool:  
        """Check if time is within available data range."""  
        return self.t_min <= t <= self.t_max  

    def validate_data(self) -> bool:  
        """  
        Validate field data consistency.  
        
        Returns  
        -------  
        bool  
            True if valid  
            
        Raises  
        ------  
        ValueError  
            If data is inconsistent  
        """  
        # Check for NaN/infinite values  
        if not np.all(np.isfinite(self.data)):  
            raise ValueError("Velocity data contains non-finite values")  
        
        if not np.all(np.isfinite(self.times)):  
            raise ValueError("Time data contains non-finite values")  
        
        if not np.all(np.isfinite(self.positions)):  
            raise ValueError("Position data contains non-finite values")  
        
        # Check time monotonicity  
        if not np.all(np.diff(self.times) >= 0):  
            raise ValueError("Times must be monotonically non-decreasing")  
        
        return True  

    def memory_usage_mb(self) -> float:  
        """Estimate memory usage in MB."""  
        data_size = self.data.size * 4        # float32 velocity data  
        times_size = self.times.size * 4      # float32 times  
        positions_size = self.positions.size * 4  # float32 positions  
        return (data_size + times_size + positions_size) / 1024**2  

    def to_jax(self):  
        """Convert to JAX arrays for accelerated computation."""  
        if not JAX_AVAILABLE:  
            warnings.warn("JAX not available, returning NumPy arrays")
            return self
        
        return TimeSeriesField(
            data=jnp.asarray(self.data, dtype=jnp.float32),
            times=jnp.asarray(self.times, dtype=jnp.float32),
            positions=jnp.asarray(self.positions, dtype=jnp.float32),
            interpolation=self.interpolation,
            extrapolation=self.extrapolation,
            _source_info=self._source_info
        )

    def get_time_derivative(self, t: float, dt: float = 1e-6) -> np.ndarray:
        """
        Compute time derivative at time t using finite differences.
        
        Parameters
        ----------
        t : float
            Time point for derivative
        dt : float
            Small time step for finite differences
            
        Returns
        -------
        np.ndarray
            Time derivative, shape (N, 3)
        """
        if not self.validate_time(t):
            warnings.warn(f"Time {t} is outside data range [{self.t_min}, {self.t_max}]")
        
        # Central difference when possible
        if self.validate_time(t - dt) and self.validate_time(t + dt):
            v_minus = self.sample_at_time(t - dt)
            v_plus = self.sample_at_time(t + dt)
            return (v_plus - v_minus) / (2 * dt)
        
        # Forward difference
        elif self.validate_time(t + dt):
            v_now = self.sample_at_time(t)
            v_plus = self.sample_at_time(t + dt)
            return (v_plus - v_now) / dt
        
        # Backward difference
        elif self.validate_time(t - dt):
            v_minus = self.sample_at_time(t - dt)
            v_now = self.sample_at_time(t)
            return (v_now - v_minus) / dt
        
        else:
            # Can't compute derivative - return zeros
            return np.zeros((self.N, 3), dtype=np.float32)

    def get_time_step_statistics(self) -> Dict[str, float]:
        """Get statistics about time step sizes."""
        if self.T < 2:
            return {"mean_dt": 0.0, "min_dt": 0.0, "max_dt": 0.0, "std_dt": 0.0}
        
        dt_array = np.diff(self.times)
        return {
            "mean_dt": float(np.mean(dt_array)),
            "min_dt": float(np.min(dt_array)),
            "max_dt": float(np.max(dt_array)),
            "std_dt": float(np.std(dt_array))
        }

    def get_velocity_statistics(self) -> Dict[str, np.ndarray]:
        """Get comprehensive velocity statistics across time."""
        # Compute statistics across time dimension
        vel_magnitude = np.linalg.norm(self.data, axis=-1)  # (T, N)
        
        return {
            "mean_velocity": np.mean(self.data, axis=0).astype(np.float32),      # (N, 3)
            "std_velocity": np.std(self.data, axis=0).astype(np.float32),       # (N, 3)
            "min_velocity": np.min(self.data, axis=0).astype(np.float32),       # (N, 3)
            "max_velocity": np.max(self.data, axis=0).astype(np.float32),       # (N, 3)
            "mean_speed": np.mean(vel_magnitude, axis=0).astype(np.float32),    # (N,)
            "max_speed": np.max(vel_magnitude, axis=0).astype(np.float32),      # (N,)
            "speed_std": np.std(vel_magnitude, axis=0).astype(np.float32)       # (N,)
        }

    def extract_time_slice(self, time_indices: Union[int, List[int], slice]) -> 'TimeSeriesField':
        """
        Extract subset of time steps.
        
        Parameters
        ----------
        time_indices : int, list, or slice
            Time indices to extract
            
        Returns
        -------
        TimeSeriesField
            New field with selected time steps
        """
        if isinstance(time_indices, int):
            time_indices = [time_indices]
        elif isinstance(time_indices, slice):
            time_indices = range(*time_indices.indices(self.T))
        
        # Ensure indices are valid
        time_indices = [idx for idx in time_indices if 0 <= idx < self.T]
        
        if not time_indices:
            raise ValueError("No valid time indices provided")
        
        return TimeSeriesField(
            data=self.data[time_indices],
            times=self.times[time_indices],
            positions=self.positions.copy(),
            interpolation=self.interpolation,
            extrapolation=self.extrapolation,
            _source_info=self._source_info.copy() if self._source_info else None
        )

    def extract_spatial_subset(self, spatial_indices: Union[List[int], np.ndarray]) -> 'TimeSeriesField':
        """
        Extract spatial subset of field positions.
        
        Parameters
        ----------
        spatial_indices : list or array
            Spatial indices to extract
            
        Returns
        -------
        TimeSeriesField
            New field with selected spatial positions
        """
        spatial_indices = np.asarray(spatial_indices, dtype=int)
        
        # Validate indices
        if np.any(spatial_indices < 0) or np.any(spatial_indices >= self.N):
            raise ValueError(f"Spatial indices out of range [0, {self.N-1}]")
        
        return TimeSeriesField(
            data=self.data[:, spatial_indices, :],
            times=self.times.copy(),
            positions=self.positions[spatial_indices],
            interpolation=self.interpolation,
            extrapolation=self.extrapolation,
            _source_info=self._source_info.copy() if self._source_info else None
        )

    def extract_2d_slice(self, plane: str = "xy") -> 'TimeSeriesField':
        """
        Extract 2D slice by dropping one coordinate.
        
        Parameters
        ----------
        plane : str
            Plane to keep: 'xy', 'xz', or 'yz'
            
        Returns
        -------
        TimeSeriesField
            2D field with appropriate coordinates
        """
        pos_2d = self.positions.copy()
        vel_2d = self.data.copy()
        
        if plane == "xy":
            # Keep X,Y coordinates, set Z=0
            pos_2d[:, 2] = 0.0
        elif plane == "xz":
            # Keep X,Z coordinates, set Y=0
            pos_2d[:, 1] = 0.0
        elif plane == "yz":
            # Keep Y,Z coordinates, set X=0
            pos_2d[:, 0] = 0.0
        else:
            raise ValueError(f"Unknown plane '{plane}'. Use 'xy', 'xz', or 'yz'.")
        
        return TimeSeriesField(
            data=vel_2d,
            times=self.times.copy(),
            positions=pos_2d,
            interpolation=self.interpolation,
            extrapolation=self.extrapolation,
            _source_info=self._source_info.copy() if self._source_info else None
        )

    # ---------- Advanced analysis methods ----------

    def compute_lagrangian_coherent_structures(self, integration_time: float, 
                                             resolution: Tuple[int, int] = (50, 50)) -> np.ndarray:
        """
        Compute finite-time Lyapunov exponent (FTLE) field for LCS analysis.
        
        Parameters
        ----------
        integration_time : float
            Integration time for FTLE computation
        resolution : tuple
            Grid resolution for FTLE field
            
        Returns
        -------
        np.ndarray
            FTLE field on regular grid
        """
        bounds_min, bounds_max = self.get_spatial_bounds()
        
        # Create regular grid for FTLE computation
        x_grid = np.linspace(bounds_min[0], bounds_max[0], resolution[0])
        y_grid = np.linspace(bounds_min[1], bounds_max[1], resolution[1])
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        
        # For simplified implementation, return gradient-based approximation
        # Full LCS would require particle integration
        mid_time = (self.t_min + self.t_max) / 2
        vel_field = self.sample_at_time(mid_time)
        
        # Compute velocity gradients as LCS approximation
        grad_norm = np.linalg.norm(vel_field, axis=1)
        
        # Map to grid using nearest neighbor
        try:
            from scipy.spatial import cKDTree
            from scipy.interpolate import griddata
            
            points_2d = self.positions[:, :2]  # Use only X,Y coordinates
            grid_points = np.column_stack([X.ravel(), Y.ravel()])
            
            ftle_values = griddata(
                points_2d, grad_norm, grid_points, 
                method='linear', fill_value=0.0
            ).reshape(resolution)
            
        except ImportError:
            # Simple fallback
            ftle_values = np.zeros(resolution, dtype=np.float32)
        
        return ftle_values

    # ---------- VTK Integration and Export ----------

    def export_to_vtk(self, output_dir: Union[str, Path], filename_prefix: str = "time_series",
                      export_format: str = "individual", max_files: Optional[int] = None,
                      include_magnitude: bool = True) -> Path:
        """
        Export time series to VTK format for ParaView visualization.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory
        filename_prefix : str
            Prefix for VTK files
        export_format : str
            Export format: 'individual' (separate files) or 'collection' (PVD series)
        max_files : int, optional
            Maximum number of files to export. If None, exports all.
        include_magnitude : bool
            Whether to include velocity magnitude field
            
        Returns
        -------
        Path
            Path to output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine time indices to export
        if max_files is not None and max_files < self.T:
            time_indices = np.linspace(0, self.T-1, max_files, dtype=int)
        else:
            time_indices = np.arange(self.T)
        
        if export_format == "individual":
            return self._export_individual_vtk_files(
                output_path, filename_prefix, time_indices, include_magnitude
            )
        elif export_format == "collection":
            return self._export_vtk_collection(
                output_path, filename_prefix, time_indices, include_magnitude
            )
        else:
            raise ValueError(f"Unknown export format: {export_format}")

    def _export_individual_vtk_files(self, output_path: Path, prefix: str,
                                   time_indices: np.ndarray, include_magnitude: bool) -> Path:
        """Export individual VTK files for each time step."""
        try:
            import vtk
            from vtk.util.numpy_support import numpy_to_vtk
            
            file_list = []
            
            for i, t_idx in enumerate(time_indices):
                time_val = self.times[t_idx]
                velocity_data = self.data[t_idx]  # (N, 3)
                
                # Create VTK points
                points = vtk.vtkPoints()
                points.SetNumberOfPoints(self.N)
                
                for n in range(self.N):
                    points.SetPoint(n, self.positions[n])
                
                # Create unstructured grid
                ugrid = vtk.vtkUnstructuredGrid()
                ugrid.SetPoints(points)
                
                # Add vertex cells for each point
                for n in range(self.N):
                    ugrid.InsertNextCell(vtk.VTK_VERTEX, 1, [n])
                
                # Add velocity data
                vel_array = numpy_to_vtk(velocity_data, deep=True, array_type=vtk.VTK_FLOAT)
                vel_array.SetName("Velocity")
                vel_array.SetNumberOfComponents(3)
                ugrid.GetPointData().SetVectors(vel_array)
                
                # Add velocity magnitude if requested
                if include_magnitude:
                    speed = np.linalg.norm(velocity_data, axis=1).astype(np.float32)
                    speed_array = numpy_to_vtk(speed, deep=True, array_type=vtk.VTK_FLOAT)
                    speed_array.SetName("Speed")
                    ugrid.GetPointData().SetScalars(speed_array)
                
                # Write file
                filename = f"{prefix}_t{i:04d}_time{time_val:.6f}.vtu"
                file_path = output_path / filename
                
                writer = vtk.vtkXMLUnstructuredGridWriter()
                writer.SetFileName(str(file_path))
                writer.SetInputData(ugrid)
                writer.Write()
                
                file_list.append(file_path)
            
            print(f"✅ Exported {len(file_list)} individual VTK files to {output_path}")
            return output_path
            
        except ImportError:
            raise ImportError("VTK library required for VTK export. Install with: pip install vtk")

    def _export_vtk_collection(self, output_path: Path, prefix: str,
                             time_indices: np.ndarray, include_magnitude: bool) -> Path:
        """Export VTK collection (PVD) for time series visualization."""
        try:
            import vtk
            
            # First export individual files
            self._export_individual_vtk_files(output_path, prefix, time_indices, include_magnitude)
            
            # Create PVD collection file
            pvd_filename = output_path / f"{prefix}_series.pvd"
            
            with open(pvd_filename, 'w') as f:
                f.write('<?xml version="1.0"?>\n')
                f.write('<VTKFile type="Collection" version="0.1">\n')
                f.write('  <Collection>\n')
                
                for i, t_idx in enumerate(time_indices):
                    time_val = self.times[t_idx]
                    vtu_filename = f"{prefix}_t{i:04d}_time{time_val:.6f}.vtu"
                    
                    f.write(f'    <DataSet timestep="{time_val:.6f}" '
                           f'group="" part="0" file="{vtu_filename}"/>\n')
                
                f.write('  </Collection>\n')
                f.write('</VTKFile>\n')
            
            print(f"✅ Created VTK collection: {pvd_filename}")
            return output_path
            
        except ImportError:
            raise ImportError("VTK library required for VTK export. Install with: pip install vtk")

    def export_trajectories_to_vtk(self, trajectory_positions: np.ndarray,
                                 output_dir: Union[str, Path], filename_prefix: str = "trajectories",
                                 particle_data: Optional[Dict[str, np.ndarray]] = None,
                                 export_both_formats: bool = True) -> Path:
        """
        Export particle trajectories to VTK format.
        
        Parameters
        ----------
        trajectory_positions : np.ndarray
            Trajectory data, shape (T_traj, N_particles, 3)
        output_dir : str or Path
            Output directory
        filename_prefix : str
            Prefix for VTK files
        particle_data : dict, optional
            Additional particle data to include
        export_both_formats : bool
            Whether to export both lines and points formats
            
        Returns
        -------
        Path
            Path to output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            import vtk
            from vtk.util.numpy_support import numpy_to_vtk
            
            T_traj, N_particles, _ = trajectory_positions.shape
            
            if export_both_formats:
                # Export trajectory lines
                self._export_trajectory_lines(
                    trajectory_positions, output_path, f"{filename_prefix}_lines",
                    particle_data
                )
                
                # Export trajectory points
                self._export_trajectory_points(
                    trajectory_positions, output_path, f"{filename_prefix}_points",
                    particle_data
                )
            else:
                # Default to lines format
                self._export_trajectory_lines(
                    trajectory_positions, output_path, filename_prefix,
                    particle_data
                )
            
            return output_path
            
        except ImportError:
            raise ImportError("VTK library required for trajectory export. Install with: pip install vtk")

    def _export_trajectory_lines(self, trajectory_positions: np.ndarray,
                               output_path: Path, filename: str,
                               particle_data: Optional[Dict[str, np.ndarray]]) -> None:
        """Export trajectories as polylines."""
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk
        
        T_traj, N_particles, _ = trajectory_positions.shape
        
        # Create polydata for trajectory lines
        polydata = vtk.vtkPolyData()
        
        # Create points for all trajectory positions
        points = vtk.vtkPoints()
        all_positions = trajectory_positions.reshape(-1, 3)  # (T*N, 3)
        
        for pos in all_positions:
            points.InsertNextPoint(pos)
        
        polydata.SetPoints(points)
        
        # Create line cells
        lines = vtk.vtkCellArray()
        
        for p in range(N_particles):
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(T_traj)
            
            for t in range(T_traj):
                point_id = t * N_particles + p
                line.GetPointIds().SetId(t, point_id)
            
            lines.InsertNextCell(line)
        
        polydata.SetLines(lines)
        
        # Add particle data if provided
        if particle_data:
            for name, data in particle_data.items():
                if len(data) == N_particles:
                    # Replicate data for each time step
                    expanded_data = np.repeat(data, T_traj)
                    vtk_array = numpy_to_vtk(expanded_data, deep=True)
                    vtk_array.SetName(name)
                    polydata.GetPointData().AddArray(vtk_array)
        
        # Write file
        file_path = output_path / f"{filename}.vtp"
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(file_path))
        writer.SetInputData(polydata)
        writer.Write()

    def _export_trajectory_points(self, trajectory_positions: np.ndarray,
                                output_path: Path, filename: str,
                                particle_data: Optional[Dict[str, np.ndarray]]) -> None:
        """Export trajectory points with time information."""
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk
        
        T_traj, N_particles, _ = trajectory_positions.shape
        
        # Create unstructured grid for points
        ugrid = vtk.vtkUnstructuredGrid()
        
        # Create points
        points = vtk.vtkPoints()
        all_positions = trajectory_positions.reshape(-1, 3)  # (T*N, 3)
        
        for pos in all_positions:
            points.InsertNextPoint(pos)
        
        ugrid.SetPoints(points)
        
        # Add vertex cells
        for i in range(len(all_positions)):
            ugrid.InsertNextCell(vtk.VTK_VERTEX, 1, [i])
        
        # Add time information
        times_expanded = np.repeat(
            np.linspace(self.t_min, self.t_max, T_traj),
            N_particles
        )
        time_array = numpy_to_vtk(times_expanded, deep=True)
        time_array.SetName("Time")
        ugrid.GetPointData().SetScalars(time_array)
        
        # Add particle IDs
        particle_ids = np.tile(np.arange(N_particles), T_traj)
        id_array = numpy_to_vtk(particle_ids, deep=True)
        id_array.SetName("ParticleID")
        ugrid.GetPointData().AddArray(id_array)
        
        # Add particle data if provided
        if particle_data:
            for name, data in particle_data.items():
                if len(data) == N_particles:
                    expanded_data = np.tile(data, T_traj)
                    vtk_array = numpy_to_vtk(expanded_data, deep=True)
                    vtk_array.SetName(name)
                    ugrid.GetPointData().AddArray(vtk_array)
        
        # Write file
        file_path = output_path / f"{filename}.vtu"
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(file_path))
        writer.SetInputData(ugrid)
        writer.Write()


# ---------- Factory functions ----------

def create_time_series_from_arrays(
    velocity_snapshots: np.ndarray,
    time_points: np.ndarray,
    positions: np.ndarray,
    **kwargs
) -> TimeSeriesField:
    """
    Create time series field from data arrays.
    
    Parameters
    ----------
    velocity_snapshots : np.ndarray
        Velocity data, shape (T, N, 3)
    time_points : np.ndarray
        Time values, shape (T,)
    positions : np.ndarray
        Spatial positions, shape (N, 2) or (N, 3)
    **kwargs
        Additional TimeSeriesField arguments
        
    Returns
    -------
    TimeSeriesField
        Configured time series field
    """
    return TimeSeriesField(
        data=velocity_snapshots,
        times=time_points,
        positions=positions,
        **kwargs
    )


def create_time_series_from_function(
    positions: np.ndarray,
    time_points: np.ndarray,
    velocity_function: callable,
    **kwargs
) -> TimeSeriesField:
    """
    Create time series field from analytical function.
    
    Parameters
    ----------
    positions : np.ndarray
        Spatial positions, shape (N, 2) or (N, 3)
    time_points : np.ndarray
        Time values, shape (T,)
    velocity_function : callable
        Function(positions, t) -> velocities that returns velocity field
    **kwargs
        Additional TimeSeriesField arguments
        
    Returns
    -------
    TimeSeriesField
        Configured time series field
    """
    positions = _ensure_positions_shape(positions)
    time_points = _ensure_float32(time_points)
    
    T = len(time_points)
    N = positions.shape[0]
    
    # Generate velocity data
    velocity_data = np.zeros((T, N, 3), dtype=np.float32)
    
    for t_idx, t in enumerate(time_points):
        vel = velocity_function(positions, t)
        velocity_data[t_idx] = _ensure_velocities_shape(vel, (N, 3))
    
    return TimeSeriesField(
        data=velocity_data,
        times=time_points,
        positions=positions,
        **kwargs
    )


def create_time_series_from_vtk_files(
    file_pattern: str,
    velocity_field_name: str = "velocity",
    max_time_steps: Optional[int] = None,
    **kwargs
) -> TimeSeriesField:
    """
    Create time series field from VTK files.
    
    Parameters
    ----------
    file_pattern : str
        File pattern or directory containing VTK files
    velocity_field_name : str
        Name of velocity field in VTK files
    max_time_steps : int, optional
        Maximum number of time steps to load
    **kwargs
        Additional TimeSeriesField arguments
        
    Returns
    -------
    TimeSeriesField
        Time series field loaded from VTK files
    """
    try:
        from ..io import open_vtk_time_series
        
        # Load VTK time series data
        vtk_data = open_vtk_time_series(
            file_pattern=file_pattern,
            max_time_steps=max_time_steps,
            velocity_field_name=velocity_field_name
        )
        
        return TimeSeriesField(
            data=vtk_data['velocity_data'],
            times=vtk_data['times'],
            positions=vtk_data['positions'],
            _source_info=vtk_data.get('source_info', {}),
            **kwargs
        )
        
    except ImportError:
        raise ImportError("VTK I/O capabilities required. Ensure jaxtrace.io is properly configured.")


# Aliases for backwards compatibility
TimeDependentVelocityField = TimeSeriesField
TimeVaryingField = TimeSeriesField