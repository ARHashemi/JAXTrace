# jaxtrace/io/vtk_writer.py
"""
VTK writer module for exporting JAXTrace results.

Supports both structured and unstructured grid formats with
consistent JAXTrace data handling.
"""

from __future__ import annotations
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np

try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    warnings.warn("VTK not available - VTK file writing will be disabled")

from ..utils.config import get_config


def _ensure_float32(data: np.ndarray) -> np.ndarray:
    """Convert data to float32 for consistency."""
    return np.asarray(data, dtype=np.float32)


def _validate_positions(positions: np.ndarray) -> np.ndarray:
    """Validate and ensure positions are (N, 3) float32."""
    pos = _ensure_float32(positions)
    
    if pos.ndim != 2:
        raise ValueError(f"Positions must be 2D array (N, 3), got shape {pos.shape}")
    
    if pos.shape[1] != 3:
        raise ValueError(f"Positions must have 3 coordinates, got {pos.shape[1]}")
    
    return pos


def _validate_velocity(velocity: np.ndarray) -> np.ndarray:
    """Validate and ensure velocity is (N, 3) float32."""
    vel = _ensure_float32(velocity)
    
    if vel.ndim != 2:
        raise ValueError(f"Velocity must be 2D array (N, 3), got shape {vel.shape}")
    
    if vel.shape[1] != 3:
        raise ValueError(f"Velocity must have 3 components, got {vel.shape[1]}")
    
    return vel


class VTKTrajectoryWriter:
    """
    Write particle trajectories to VTK format.
    
    Exports trajectory data as VTK PolyData with lines connecting
    particle positions over time, compatible with ParaView visualization.
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize trajectory writer.
        
        Parameters
        ----------
        output_dir : str or Path
            Directory to save VTK files
        """
        if not VTK_AVAILABLE:
            raise ImportError("VTK not available - cannot write VTK files")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"VTKTrajectoryWriter initialized:")
        print(f"  📁 Output directory: {self.output_dir.absolute()}")
    
    def write_trajectory_points(
        self,
        trajectory_positions: np.ndarray,  # (T, N, 3)
        times: np.ndarray,                 # (T,)
        filename: str = "particle_trajectories.vtp",
        particle_data: Optional[Dict[str, np.ndarray]] = None,
        time_subset: Optional[slice] = None
    ):
        """
        Write trajectory as point cloud time series.
        
        Each particle position at each time step becomes a separate point
        with time and particle ID information.
        
        Parameters
        ----------
        trajectory_positions : np.ndarray
            Trajectory positions, shape (T, N, 3)
        times : np.ndarray
            Time values, shape (T,)
        filename : str
            Output filename (should end with .vtp)
        particle_data : dict, optional
            Additional per-particle data to include:
            - Keys: data field names
            - Values: arrays of shape (N,) or (T, N)
        time_subset : slice, optional
            Time slice to export (None = all times)
        """
        T, N, _ = trajectory_positions.shape
        
        if time_subset is not None:
            trajectory_positions = trajectory_positions[time_subset]
            times = times[time_subset]
            T = trajectory_positions.shape[0]
            
            # Also subset time-varying particle data
            if particle_data:
                for key, data in particle_data.items():
                    if data.ndim == 2 and data.shape[0] == len(times) + len(range(*time_subset.indices(len(times)))):
                        particle_data[key] = data[time_subset]
        
        # Create VTK PolyData for all time steps
        poly_data = vtk.vtkPolyData()
        
        # Flatten all positions into single array
        all_positions = trajectory_positions.reshape(-1, 3)  # (T*N, 3)
        all_positions = _validate_positions(all_positions)
        
        # Create VTK points
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(all_positions))
        poly_data.SetPoints(vtk_points)
        
        # Create vertices for all points
        vertices = vtk.vtkCellArray()
        for i in range(T * N):
            vertices.InsertNextCell(1, [i])
        poly_data.SetVerts(vertices)
        
        # Add time data
        time_array = np.repeat(times, N)  # (T*N,)
        time_vtk = numpy_to_vtk(_ensure_float32(time_array))
        time_vtk.SetName("Time")
        poly_data.GetPointData().AddArray(time_vtk)
        poly_data.GetPointData().SetActiveScalars("Time")
        
        # Add particle IDs
        particle_ids = np.tile(np.arange(N, dtype=np.float32), T)  # (T*N,)
        pid_vtk = numpy_to_vtk(particle_ids)
        pid_vtk.SetName("ParticleID")
        poly_data.GetPointData().AddArray(pid_vtk)
        
        # Add time step indices
        time_indices = np.repeat(np.arange(T, dtype=np.float32), N)  # (T*N,)
        tid_vtk = numpy_to_vtk(time_indices)
        tid_vtk.SetName("TimeStep")
        poly_data.GetPointData().AddArray(tid_vtk)
        
        # Add additional particle data
        if particle_data:
            for name, data in particle_data.items():
                try:
                    data = _ensure_float32(data)
                    
                    if data.ndim == 1 and len(data) == N:
                        # Per-particle data -> repeat for all time steps
                        expanded_data = np.tile(data, T)  # (T*N,)
                    elif data.ndim == 2 and data.shape == (T, N):
                        # Time-varying per-particle data
                        expanded_data = data.ravel()  # (T*N,)
                    else:
                        warnings.warn(f"Skipping data '{name}' - incompatible shape {data.shape}")
                        continue
                    
                    data_vtk = numpy_to_vtk(expanded_data)
                    data_vtk.SetName(name)
                    poly_data.GetPointData().AddArray(data_vtk)
                    
                except Exception as e:
                    warnings.warn(f"Failed to add particle data '{name}': {e}")
        
        # Write to file
        output_path = self.output_dir / filename
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(output_path))
        writer.SetInputData(poly_data)
        writer.SetCompressorTypeToZLib()  # Compress for smaller files
        writer.Write()
        
        print(f"✅ Trajectory points written to: {output_path}")
        print(f"   📊 Points: {T * N} (T={T}, N={N})")
        print(f"   🕒 Time range: {times[0]:.3f} to {times[-1]:.3f}")
        print(f"   📏 File size: {output_path.stat().st_size / 1024**2:.1f} MB")
    
    def write_trajectory_lines(
        self,
        trajectory_positions: np.ndarray,  # (T, N, 3)
        times: np.ndarray,                 # (T,)
        filename: str = "particle_paths.vtp",
        max_particles: Optional[int] = None,
        particle_subset: Optional[np.ndarray] = None
    ):
        """
        Write trajectory as polylines connecting each particle's path.
        
        Creates one polyline per particle, connecting its positions across time.
        
        Parameters
        ----------
        trajectory_positions : np.ndarray
            Trajectory positions, shape (T, N, 3)
        times : np.ndarray
            Time values, shape (T,)
        filename : str
            Output filename (should end with .vtp)
        max_particles : int, optional
            Maximum number of particles to include (for large datasets)
        particle_subset : np.ndarray, optional
            Specific particle indices to include (overrides max_particles)
        """
        T, N, _ = trajectory_positions.shape
        
        # Determine which particles to include
        if particle_subset is not None:
            particle_indices = particle_subset
            N_selected = len(particle_indices)
        elif max_particles is not None and N > max_particles:
            particle_indices = np.linspace(0, N-1, max_particles, dtype=int)
            N_selected = max_particles
            print(f"  📉 Limited to {N_selected} particles for visualization")
        else:
            particle_indices = np.arange(N)
            N_selected = N
        
        # Extract selected trajectories
        selected_trajectories = trajectory_positions[:, particle_indices, :]  # (T, N_selected, 3)
        
        # Create VTK PolyData
        poly_data = vtk.vtkPolyData()
        
        # Flatten positions
        all_positions = selected_trajectories.reshape(-1, 3)  # (T*N_selected, 3)
        all_positions = _validate_positions(all_positions)
        
        # Create VTK points
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(all_positions))
        poly_data.SetPoints(vtk_points)
        
        # Create polylines for each particle
        lines = vtk.vtkCellArray()
        for particle_idx in range(N_selected):
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(T)
            for time_idx in range(T):
                point_id = time_idx * N_selected + particle_idx
                line.GetPointIds().SetId(time_idx, point_id)
            lines.InsertNextCell(line)
        
        poly_data.SetLines(lines)
        
        # Add time data to points
        time_array = np.repeat(times, N_selected)  # (T*N_selected,)
        time_vtk = numpy_to_vtk(_ensure_float32(time_array))
        time_vtk.SetName("Time")
        poly_data.GetPointData().SetScalars(time_vtk)
        
        # Add particle IDs
        particle_ids = np.tile(particle_indices.astype(np.float32), T)  # (T*N_selected,)
        pid_vtk = numpy_to_vtk(particle_ids)
        pid_vtk.SetName("ParticleID")
        poly_data.GetPointData().AddArray(pid_vtk)
        
        # Add path length information to cells (lines)
        path_lengths = np.zeros(N_selected, dtype=np.float32)
        for i in range(N_selected):
            path = selected_trajectories[:, i, :]  # (T, 3)
            diffs = np.diff(path, axis=0)  # (T-1, 3)
            path_lengths[i] = np.sum(np.linalg.norm(diffs, axis=1))
        
        path_vtk = numpy_to_vtk(path_lengths)
        path_vtk.SetName("PathLength")
        poly_data.GetCellData().AddArray(path_vtk)
        poly_data.GetCellData().SetActiveScalars("PathLength")
        
        # Write to file
        output_path = self.output_dir / filename
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(output_path))
        writer.SetInputData(poly_data)
        writer.SetCompressorTypeToZLib()
        writer.Write()
        
        print(f"✅ Trajectory lines written to: {output_path}")
        print(f"   📊 Lines: {N_selected}, Points: {T * N_selected}")
        print(f"   🕒 Time range: {times[0]:.3f} to {times[-1]:.3f}")
        print(f"   📏 File size: {output_path.stat().st_size / 1024**2:.1f} MB")

class VTKFieldWriter:
    """
    Write velocity fields and scalar data to VTK format.
    
    Supports both structured and unstructured grids for velocity field
    visualization and analysis.
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize field writer.
        
        Parameters
        ----------
        output_dir : str or Path
            Directory to save VTK files
        """
        if not VTK_AVAILABLE:
            raise ImportError("VTK not available - cannot write VTK files")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"VTKFieldWriter initialized:")
        print(f"  📁 Output directory: {self.output_dir.absolute()}")
    
    def write_unstructured_field(
        self,
        positions: np.ndarray,           # (N, 3)
        velocity_data: np.ndarray,       # (N, 3) or (T, N, 3)
        filename: str = "velocity_field.vtu",
        scalar_data: Optional[Dict[str, np.ndarray]] = None,
        time_value: Optional[float] = None,
        cell_connectivity: Optional[np.ndarray] = None
    ):
        """
        Write velocity field on unstructured grid.
        
        Parameters
        ----------
        positions : np.ndarray
            Grid positions, shape (N, 3)
        velocity_data : np.ndarray
            Velocity data, shape (N, 3) or (T, N, 3)
        filename : str
            Output filename (should end with .vtu)
        scalar_data : dict, optional
            Additional scalar data to include:
            - Keys: field names
            - Values: arrays of shape (N,)
        time_value : float, optional
            Time value for this snapshot
        cell_connectivity : np.ndarray, optional
            Cell connectivity for creating actual mesh cells
            Shape depends on cell type (e.g., (N_cells, 4) for tetrahedra)
        """
        positions = _validate_positions(positions)
        N = positions.shape[0]
        
        # Handle time series velocity data
        if velocity_data.ndim == 3:
            # Time series data - take first timestep or average
            T = velocity_data.shape[0]
            if T == 1:
                velocity_data = velocity_data[0]  # (N, 3)
            else:
                velocity_data = np.mean(velocity_data, axis=0)  # (N, 3)
                print(f"  📊 Averaged velocity over {T} time steps")
        
        velocity_data = _validate_velocity(velocity_data)
        
        if velocity_data.shape[0] != N:
            raise ValueError(f"Velocity data size {velocity_data.shape[0]} != positions size {N}")
        
        # Create VTK UnstructuredGrid
        grid = vtk.vtkUnstructuredGrid()
        
        # Set points
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(positions))
        grid.SetPoints(vtk_points)
        
        # Create cells
        if cell_connectivity is not None:
            # Use provided connectivity
            self._add_cells_from_connectivity(grid, cell_connectivity)
        else:
            # Create vertex cells for each point (point cloud)
            for i in range(N):
                grid.InsertNextCell(vtk.VTK_VERTEX, 1, [i])
        
        # Add velocity field
        velocity_vtk = numpy_to_vtk(velocity_data)
        velocity_vtk.SetName("velocity")
        grid.GetPointData().SetVectors(velocity_vtk)
        
        # Add velocity magnitude
        velocity_magnitude = np.linalg.norm(velocity_data, axis=1)
        mag_vtk = numpy_to_vtk(_ensure_float32(velocity_magnitude))
        mag_vtk.SetName("velocity_magnitude")
        grid.GetPointData().AddArray(mag_vtk)
        grid.GetPointData().SetActiveScalars("velocity_magnitude")
        
        # Add time if provided
        if time_value is not None:
            time_array = np.full(N, time_value, dtype=np.float32)
            time_vtk = numpy_to_vtk(time_array)
            time_vtk.SetName("Time")
            grid.GetPointData().AddArray(time_vtk)
        
        # Add scalar data
        if scalar_data:
            for name, data in scalar_data.items():
                try:
                    data = _ensure_float32(data)
                    if len(data) != N:
                        warnings.warn(f"Skipping scalar '{name}' - size mismatch: {len(data)} != {N}")
                        continue
                    
                    scalar_vtk = numpy_to_vtk(data)
                    scalar_vtk.SetName(name)
                    grid.GetPointData().AddArray(scalar_vtk)
                    
                except Exception as e:
                    warnings.warn(f"Failed to add scalar data '{name}': {e}")
        
        # Write to file
        output_path = self.output_dir / filename
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(output_path))
        writer.SetInputData(grid)
        writer.SetCompressorTypeToZLib()
        writer.Write()
        
        print(f"✅ Unstructured field written to: {output_path}")
        print(f"   📊 Points: {N}")
        print(f"   🎯 Fields: velocity, velocity_magnitude + {len(scalar_data or {})} scalars")
        print(f"   📏 File size: {output_path.stat().st_size / 1024**2:.1f} MB")
    
    def _add_cells_from_connectivity(self, grid: vtk.vtkUnstructuredGrid, connectivity: np.ndarray):
        """Add cells to grid based on connectivity array."""
        n_cells, n_points_per_cell = connectivity.shape
        
        # Determine VTK cell type based on points per cell
        if n_points_per_cell == 3:
            cell_type = vtk.VTK_TRIANGLE
        elif n_points_per_cell == 4:
            cell_type = vtk.VTK_TETRA
        elif n_points_per_cell == 8:
            cell_type = vtk.VTK_HEXAHEDRON
        else:
            raise ValueError(f"Unsupported cell type with {n_points_per_cell} points")
        
        for i in range(n_cells):
            grid.InsertNextCell(cell_type, n_points_per_cell, connectivity[i])
    
    def write_structured_field(
        self,
        grid_data: np.ndarray,          # (Nx, Ny, Nz, 3) or (Nx, Ny, Nz, C)
        origin: np.ndarray,             # (3,) - grid origin
        spacing: np.ndarray,            # (3,) - grid spacing
        filename: str = "structured_field.vti",
        field_name: str = "velocity",
        time_value: Optional[float] = None
    ):
        """
        Write structured grid velocity field to VTK ImageData format.
        
        Parameters
        ----------
        grid_data : np.ndarray
            Grid data, shape (Nx, Ny, Nz, C) where C is number of components
        origin : np.ndarray
            Grid origin coordinates (3,)
        spacing : np.ndarray
            Grid spacing (3,)
        filename : str
            Output filename (should end with .vti)
        field_name : str
            Name for the vector field
        time_value : float, optional
            Time value for this snapshot
        """
        if grid_data.ndim != 4:
            raise ValueError(f"Grid data must be 4D (Nx, Ny, Nz, C), got shape {grid_data.shape}")
        
        Nx, Ny, Nz, C = grid_data.shape
        origin = _ensure_float32(origin)
        spacing = _ensure_float32(spacing)
        
        # Create VTK ImageData
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(Nx, Ny, Nz)
        image_data.SetOrigin(origin)
        image_data.SetSpacing(spacing)
        
        # Flatten and convert data (VTK uses Fortran order)
        flat_data = grid_data.reshape(-1, C, order='F')  # (Nx*Ny*Nz, C)
        flat_data = _ensure_float32(flat_data)
        
        # Create VTK array
        vtk_array = numpy_to_vtk(flat_data)
        vtk_array.SetName(field_name)
        
        # Set as vectors if 3-component, otherwise as scalars
        if C == 3:
            image_data.GetPointData().SetVectors(vtk_array)
            
            # Also add magnitude
            magnitude = np.linalg.norm(flat_data, axis=1)
            mag_vtk = numpy_to_vtk(_ensure_float32(magnitude))
            mag_vtk.SetName(f"{field_name}_magnitude")
            image_data.GetPointData().AddArray(mag_vtk)
            image_data.GetPointData().SetActiveScalars(f"{field_name}_magnitude")
        else:
            image_data.GetPointData().SetScalars(vtk_array)
        
        # Add time if provided
        if time_value is not None:
            time_array = np.full(Nx*Ny*Nz, time_value, dtype=np.float32)
            time_vtk = numpy_to_vtk(time_array)
            time_vtk.SetName("Time")
            image_data.GetPointData().AddArray(time_vtk)
        
        # Write to file
        output_path = self.output_dir / filename
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(str(output_path))
        writer.SetInputData(image_data)
        writer.SetCompressorTypeToZLib()
        writer.Write()
        
        print(f"✅ Structured field written to: {output_path}")
        print(f"   📊 Grid: {Nx}×{Ny}×{Nz} = {Nx*Ny*Nz} points")
        print(f"   🎯 Field: {field_name} ({C}D)")
        print(f"   📏 File size: {output_path.stat().st_size / 1024**2:.1f} MB")
    
    def write_time_series_collection(
        self,
        time_series_data: Dict[str, Any],
        collection_name: str = "velocity_time_series",
        max_time_steps: Optional[int] = None,
        field_name: str = "velocity"
    ):
        """
        Write time series as VTK collection (.pvd file + individual time steps).
        
        Parameters
        ----------
        time_series_data : dict
            Dictionary with JAXTrace time series format:
            - 'velocity_data': (T, N, 3) 
            - 'positions': (N, 3)
            - 'times': (T,)
        collection_name : str
            Base name for the collection
        max_time_steps : int, optional
            Maximum time steps to export
        field_name : str
            Name for velocity field
        """
        velocity_data = time_series_data['velocity_data']  # (T, N, 3)
        positions = time_series_data['positions']          # (N, 3)
        times = time_series_data['times']                  # (T,)
        
        T, N, _ = velocity_data.shape
        
        if max_time_steps is not None and T > max_time_steps:
            # Subsample time steps
            indices = np.linspace(0, T-1, max_time_steps, dtype=int)
            velocity_data = velocity_data[indices]
            times = times[indices]
            T = max_time_steps
            print(f"  📉 Subsampled to {T} time steps for export")
        
        # Create individual VTU files
        vtu_files = []
        for t_idx in range(T):
            vtu_filename = f"{collection_name}_{t_idx:04d}.vtu"
            
            # Calculate velocity magnitude
            velocity_snapshot = velocity_data[t_idx]  # (N, 3)
            scalar_data = {
                f"{field_name}_magnitude": np.linalg.norm(velocity_snapshot, axis=1)
            }
            
            self.write_unstructured_field(
                positions=positions,
                velocity_data=velocity_snapshot,
                filename=vtu_filename,
                scalar_data=scalar_data,
                time_value=float(times[t_idx])
            )
            
            vtu_files.append(vtu_filename)
            
            if (t_idx + 1) % max(1, T // 10) == 0:
                print(f"  📊 Exported {t_idx + 1}/{T} time steps")
        
        # Create PVD collection file
        self._write_pvd_collection(collection_name, vtu_files, times)
        
        print(f"✅ Time series collection written:")
        print(f"   📊 Files: {T} VTU files + 1 PVD collection")
        print(f"   🕒 Time range: {times[0]:.3f} to {times[-1]:.3f}")
    
    def _write_pvd_collection(self, collection_name: str, vtu_files: List[str], times: np.ndarray):
        """Write PVD collection file for time series visualization."""
        pvd_path = self.output_dir / f"{collection_name}.pvd"
        
        with open(pvd_path, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1">\n')
            f.write('  <Collection>\n')
            
            for vtu_file, time_val in zip(vtu_files, times):
                f.write(f'    <DataSet timestep="{time_val:.6f}" file="{vtu_file}"/>\n')
            
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')
        
        print(f"   📋 Collection file: {pvd_path}")


# Legacy structured grid writer (migrated from vtk_io.py)
def write_structured_grid(
    grid_data: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    filename: str,
    field_name: str = "velocity"
):
    """
    Legacy function for writing structured grids.
    
    Parameters
    ----------
    grid_data : np.ndarray
        Grid data, shape (Nx, Ny, Nz, C)
    origin : np.ndarray
        Grid origin (3,)
    spacing : np.ndarray
        Grid spacing (3,)
    filename : str
        Output filename
    field_name : str
        Field name
    """
    output_dir = Path(filename).parent
    writer = VTKFieldWriter(output_dir)
    writer.write_structured_field(
        grid_data=grid_data,
        origin=origin,
        spacing=spacing,
        filename=Path(filename).name,
        field_name=field_name
    )