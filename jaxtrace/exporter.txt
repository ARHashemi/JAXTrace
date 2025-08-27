"""
VTK Export Module for JAXTrace

Advanced VTK exporter with 2D/3D support for particle tracking results and density fields.
"""

import numpy as np
import os
from typing import Dict, Tuple, Optional, Union
import warnings

# VTK imports
try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    warnings.warn("VTK not available. VTK export functionality will be disabled.")


class VTKExporter:
    """Advanced VTK exporter with 2D/3D support for particle tracking results and density fields."""
    
    def __init__(self, output_directory: str = "vtk_output"):
        """
        Initialize VTK exporter.
        
        Args:
            output_directory: Directory to save VTK files
        """
        if not VTK_AVAILABLE:
            raise ImportError("VTK is required for VTKExporter. Please install VTK.")
        
        self.output_directory = output_directory
        os.makedirs(output_directory, exist_ok=True)
        print(f"VTKExporter initialized. Output directory: {output_directory}")
    
    def export_particle_positions(self, 
                                 positions: np.ndarray,
                                 filename: str,
                                 point_data: Optional[Dict[str, np.ndarray]] = None,
                                 time_value: Optional[float] = None,
                                 dimensions: str = '3d',
                                 plane: str = 'xy',
                                 position: float = 0.0,
                                 slab_thickness: float = 0.1) -> str:
        """
        Export particle positions as VTK points with 2D/3D support.
        
        Args:
            positions: Particle positions (N, 3) for 3D or (N, 2) for 2D
            filename: Output filename (without .vtp extension)
            point_data: Additional point data arrays {name: array}
            time_value: Time value for temporal data
            dimensions: '2d' or '3d' export
            plane: Cross-section plane for 2D ('xy', 'xz', 'yz')
            position: Position along the third axis for 2D slicing
            slab_thickness: Thickness of the slice for 2D
            
        Returns:
            Full path to saved file
        """
        # Process positions based on dimensions
        if dimensions == '2d':
            if positions.shape[1] == 3:
                # Extract 2D positions from 3D data
                export_positions, export_point_data = self._extract_2d_data(
                    positions, point_data, plane, position, slab_thickness
                )
            else:
                # Already 2D data - convert to 3D for VTK
                export_positions = self._convert_2d_to_3d(positions, plane, position)
                export_point_data = point_data
        else:
            # 3D export
            if positions.shape[1] != 3:
                raise ValueError("3D export requires 3D positions (N, 3)")
            export_positions = positions
            export_point_data = point_data
        
        # Create VTK points
        points = vtk.vtkPoints()
        for i in range(len(export_positions)):
            points.InsertNextPoint(export_positions[i])
        
        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        # Create vertices for visualization
        vertices = vtk.vtkCellArray()
        for i in range(len(export_positions)):
            vertices.InsertNextCell(1, [i])
        polydata.SetVerts(vertices)
        
        # Add point data
        if export_point_data is not None:
            for name, data in export_point_data.items():
                if len(data) != len(export_positions):
                    warnings.warn(f"Point data '{name}' length mismatch. Skipping.")
                    continue
                
                # Convert to VTK array
                if data.ndim == 1:
                    vtk_array = numpy_to_vtk(data)
                    vtk_array.SetName(name)
                    polydata.GetPointData().AddArray(vtk_array)
                elif data.ndim == 2 and data.shape[1] == 3:
                    # Vector data
                    vtk_array = numpy_to_vtk(data)
                    vtk_array.SetName(name)
                    vtk_array.SetNumberOfComponents(3)
                    polydata.GetPointData().AddArray(vtk_array)
                    if name.lower() in ['velocity', 'displacement']:
                        polydata.GetPointData().SetVectors(vtk_array)
                else:
                    warnings.warn(f"Skipping point data '{name}' with unsupported shape {data.shape}")
        
        # Add metadata
        if time_value is not None:
            polydata.GetFieldData().AddArray(numpy_to_vtk(np.array([time_value], dtype=np.float32), deep=True))
            polydata.GetFieldData().GetArray(0).SetName("TimeValue")
        if dimensions == '2d':
            polydata.GetFieldData().AddArray(numpy_to_vtk(np.array([plane, position, slab_thickness], dtype=np.float32), deep=True))
            polydata.GetFieldData().GetArray(1).SetName("Plane")
            polydata.GetFieldData().GetArray(2).SetName("Position")
            polydata.GetFieldData().GetArray(3).SetName("SlabThickness")
        
        # self._add_metadata(polydata, dimensions, plane, position, time_value)
        
        # Write file
        filepath = os.path.join(self.output_directory, f"{filename}.vtp")
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filepath)
        writer.SetInputData(polydata)
        writer.Write()
        
        export_type = f"{dimensions.upper()} ({plane} plane)" if dimensions == '2d' else "3D"
        print(f"Exported {export_type} particle positions: {filepath} ({len(export_positions)} particles)")
        return filepath
    
    def export_density_field(self,
                             density_data: Dict,
                             filename: str,
                             additional_fields: Optional[Dict[str, np.ndarray]] = None) -> str:
        """
        Export density field as VTK structured grid with 2D/3D support.
        
        Args:
            density_data: Dictionary containing density information:
                For 2D: {'X': X_grid, 'Y': Y_grid, 'density': density_values, 
                        'dimensions': '2d', 'plane': 'xy', 'position': 0.0}
                For 3D: {'X': X_grid, 'Y': Y_grid, 'Z': Z_grid, 'density': density_values,
                        'dimensions': '3d'}
            filename: Output filename (without extension)
            additional_fields: Additional scalar fields {name: array}
            
        Returns:
            Full path to saved file
        """
        dimensions = density_data.get('dimensions', '2d')
        
        if dimensions == '2d':
            return self._export_2d_density(density_data, filename, additional_fields)
        else:
            return self._export_3d_density(density_data, filename, additional_fields)
    
    def _export_2d_density(self, density_data: Dict, filename: str, additional_fields: Optional[Dict]) -> str:
        """Export 2D density field as VTK structured grid."""
        X = density_data['X']
        Y = density_data['Y']
        density = density_data['density']
        plane = density_data.get('plane', 'xy')
        position = density_data.get('position', 0.0)
        
        if X.shape != Y.shape or X.shape != density.shape:
            raise ValueError("X, Y, and density arrays must have the same shape")
        
        ny, nx = X.shape
        nz = 1  # 2D slice
        
        # Create structured grid
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(nx, ny, nz)
        
        # Create points - convert 2D to 3D based on plane
        points = vtk.vtkPoints()
        
        if plane == 'xy':
            # XY plane at fixed Z
            for j in range(ny):
                for i in range(nx):
                    points.InsertNextPoint(X[j, i], Y[j, i], position)
        elif plane == 'xz':
            # XZ plane at fixed Y
            for j in range(ny):  # j represents Z
                for i in range(nx):  # i represents X
                    points.InsertNextPoint(X[j, i], position, Y[j, i])
        elif plane == 'yz':
            # YZ plane at fixed X
            for j in range(ny):  # j represents Z
                for i in range(nx):  # i represents Y
                    points.InsertNextPoint(position, X[j, i], Y[j, i])
        else:
            raise ValueError("plane must be 'xy', 'xz', or 'yz'")
        
        grid.SetPoints(points)
        
        # Add density as point data
        density_flat = density.flatten()
        density_array = numpy_to_vtk(density_flat)
        density_array.SetName("Density")
        grid.GetPointData().SetScalars(density_array)
        
        # Add additional fields
        if additional_fields is not None:
            for name, field in additional_fields.items():
                if field.shape != density.shape:
                    warnings.warn(f"Skipping field '{name}' with shape {field.shape} (expected {density.shape})")
                    continue
                
                field_flat = field.flatten()
                field_array = numpy_to_vtk(field_flat)
                field_array.SetName(name)
                grid.GetPointData().AddArray(field_array)
        
        # Add metadata
        self._add_field_metadata(grid, '2d', plane, position)
        
        # Write file
        filepath = os.path.join(self.output_directory, f"{filename}.vti")
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(filepath)
        writer.SetInputData(grid)
        writer.Write()
        
        print(f"Exported 2D density field ({plane} plane): {filepath} ({nx}x{ny} grid)")
        return filepath
    
    def _export_3d_density(self, density_data: Dict, filename: str, additional_fields: Optional[Dict]) -> str:
        """Export 3D density field as VTK structured grid."""
        X = density_data['X']
        Y = density_data['Y'] 
        Z = density_data['Z']
        density = density_data['density']
        
        if X.shape != Y.shape or X.shape != Z.shape or X.shape != density.shape:
            raise ValueError("X, Y, Z, and density arrays must have the same shape")
        
        nz, ny, nx = X.shape  # Assuming indexing='ij' from meshgrid
        
        # Create structured grid
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(nx, ny, nz)
        
        # Create points
        points = vtk.vtkPoints()
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    points.InsertNextPoint(X[k, j, i], Y[k, j, i], Z[k, j, i])
        
        grid.SetPoints(points)
        
        # Add density as point data
        density_flat = density.flatten(order='F')  # Fortran order for VTK
        density_array = numpy_to_vtk(density_flat)
        density_array.SetName("Density")
        grid.GetPointData().SetScalars(density_array)
        
        # Add additional fields
        if additional_fields is not None:
            for name, field in additional_fields.items():
                if field.shape != density.shape:
                    warnings.warn(f"Skipping field '{name}' with shape {field.shape} (expected {density.shape})")
                    continue
                
                field_flat = field.flatten(order='F')
                field_array = numpy_to_vtk(field_flat)
                field_array.SetName(name)
                grid.GetPointData().AddArray(field_array)
        
        # Add metadata
        self._add_field_metadata(grid, '3d')
        
        # Write file
        filepath = os.path.join(self.output_directory, f"{filename}.vti")
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(filepath)
        writer.SetInputData(grid)
        writer.Write()
        
        print(f"Exported 3D density field: {filepath} ({nx}x{ny}x{nz} grid)")
        return filepath
    
    def export_particle_trajectories(self,
                                   trajectories: np.ndarray,
                                   filename: str,
                                   time_values: Optional[np.ndarray] = None,
                                   particle_ids: Optional[np.ndarray] = None,
                                   dimensions: str = '3d') -> str:
        """
        Export particle trajectories as VTK polylines with 2D/3D support.
        
        Args:
            trajectories: Particle trajectories (N_particles, N_timesteps, 2/3)
            filename: Output filename (without .vtp extension)
            time_values: Time values for each timestep
            particle_ids: Particle IDs
            dimensions: '2d' or '3d' export
            
        Returns:
            Full path to saved file
        """
        n_particles, n_timesteps, spatial_dims = trajectories.shape
        
        # Ensure 3D coordinates for VTK
        if spatial_dims == 2 and dimensions == '3d':
            # Pad 2D trajectories with zeros in Z
            traj_3d = np.zeros((n_particles, n_timesteps, 3))
            traj_3d[:, :, :2] = trajectories
            trajectories = traj_3d
        elif spatial_dims == 2:
            # Convert 2D to 3D with Z=0
            traj_3d = np.zeros((n_particles, n_timesteps, 3))
            traj_3d[:, :, :2] = trajectories
            trajectories = traj_3d
        
        # Create polydata
        polydata = vtk.vtkPolyData()
        
        # Create points from all trajectory points
        points = vtk.vtkPoints()
        point_idx = 0
        
        # Create polylines for trajectories
        lines = vtk.vtkCellArray()
        
        # Arrays for point data
        particle_id_array = vtk.vtkIntArray()
        particle_id_array.SetName("ParticleID")
        
        time_array = vtk.vtkFloatArray()
        time_array.SetName("Time")
        
        for particle_idx in range(n_particles):
            # Create line for this particle
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(n_timesteps)
            
            for time_idx in range(n_timesteps):
                pos = trajectories[particle_idx, time_idx]
                points.InsertNextPoint(pos)
                line.GetPointIds().SetId(time_idx, point_idx)
                
                # Add point data
                pid = particle_ids[particle_idx] if particle_ids is not None else particle_idx
                particle_id_array.InsertNextValue(pid)
                
                time_val = time_values[time_idx] if time_values is not None else time_idx
                time_array.InsertNextValue(time_val)
                
                point_idx += 1
            
            lines.InsertNextCell(line)
        
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        polydata.GetPointData().AddArray(particle_id_array)
        polydata.GetPointData().AddArray(time_array)
        
        # Add metadata
        self._add_metadata(polydata, dimensions, time_range=(0, n_timesteps-1))
        
        # Write file
        filepath = os.path.join(self.output_directory, f"{filename}.vtp")
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filepath)
        writer.SetInputData(polydata)
        writer.Write()
        
        print(f"Exported {dimensions.upper()} trajectories: {filepath} ({n_particles} particles, {n_timesteps} timesteps)")
        return filepath
    
    def export_complete_simulation(self,
                                  positions: np.ndarray,
                                  base_filename: str = "simulation_results",
                                  initial_positions: Optional[np.ndarray] = None,
                                  trajectories: Optional[np.ndarray] = None,
                                  density_method: str = 'jax_kde',
                                  dimensions: str = '3d',
                                  plane: str = 'xy',
                                  position: float = 0.0,
                                  slab_thickness: float = 0.1,
                                  density_resolution: Union[int, Tuple] = 100,
                                  time_value: Optional[float] = None,
                                  normalize_density: bool = True,
                                  calculate_density: bool = True,
                                  **density_kwargs) -> Dict[str, str]:
        """
        Export complete simulation results with automatic density calculation.
        
        Args:
            positions: Final particle positions (N, 3)
            base_filename: Base filename for all outputs
            initial_positions: Initial particle positions (N, 3)
            trajectories: Full particle trajectories (N, timesteps, 3)
            density_method: Density calculation method ('jax_kde', 'sph')
            dimensions: '2d' or '3d' export and density calculation
            plane: Cross-section plane for 2D ('xy', 'xz', 'yz')
            position: Position along the third axis for 2D
            slab_thickness: Thickness of the slice for 2D
            density_resolution: Resolution for density grid
            time_value: Time value for the final state
            normalize_density: Whether to normalize density
            calculate_density: Whether to calculate and export density
            **density_kwargs: Additional parameters for density calculation
            
        Returns:
            Dictionary mapping data type to saved file path
        """
        saved_files = {}
        
        # Calculate point data (displacement if initial positions available)
        point_data = {}
        if initial_positions is not None:
            # For 2D export, calculate displacement in 2D or 3D
            if dimensions == '2d':
                final_2d, _ = self._extract_2d_data(positions, None, plane, position, slab_thickness)
                initial_2d, _ = self._extract_2d_data(initial_positions, None, plane, position, slab_thickness)
                # point_data['InitialPositions'] = initial_2d
                point_data['FinalPositions'] = final_2d
                if len(final_2d) == len(initial_2d):
                    displacement_2d = final_2d - initial_2d
                    displacement_magnitude = np.linalg.norm(displacement_2d, axis=1)
                    
                    # Convert back to 3D for VTK export
                    displacement_3d = self._convert_2d_displacement_to_3d(displacement_2d, plane)
                    point_data['Displacement'] = displacement_3d
                    point_data['DisplacementMagnitude'] = displacement_magnitude
            else:
                # 3D displacement
                # point_data['InitialPositions'] = initial_positions
                point_data['FinalPositions'] = positions
                displacement = positions - initial_positions
                displacement_magnitude = np.linalg.norm(displacement, axis=1)
                point_data['Displacement'] = displacement
                point_data['DisplacementMagnitude'] = displacement_magnitude
        
        # Export final positions
        final_file = self.export_particle_positions(
            positions,
            f"{base_filename}_final",
            point_data=point_data,
            time_value=time_value,
            dimensions=dimensions,
            plane=plane,
            position=position,
            slab_thickness=slab_thickness
        )
        saved_files['final_positions'] = final_file
        
        # Export initial positions if provided
        if initial_positions is not None:
            initial_file = self.export_particle_positions(
                initial_positions,
                f"{base_filename}_initial",
                time_value=0.0 if time_value is not None else None,
                dimensions=dimensions,
                plane=plane,
                position=position,
                slab_thickness=slab_thickness
            )
            saved_files['initial_positions'] = initial_file
        
        # Export trajectories if provided
        if trajectories is not None:
            traj_file = self.export_particle_trajectories(
                trajectories,
                f"{base_filename}_trajectories",
                dimensions=dimensions
            )
            saved_files['trajectories'] = traj_file
        
        # Calculate and export density if requested
        if calculate_density:
            try:
                density_data = self._calculate_density_for_export(
                    positions, density_method, dimensions, plane, position, 
                    slab_thickness, density_resolution, normalize_density, **density_kwargs
                )
                
                if density_data is not None:
                    density_file = self.export_density_field(
                        density_data,
                        f"{base_filename}_density"
                    )
                    saved_files['density_field'] = density_file
                
            except Exception as e:
                print(f"Warning: Density calculation failed: {e}")
                print("Continuing without density export...")
        
        # Create ParaView state file
        # state_file = self.create_paraview_state_file(saved_files, f"{base_filename}_state.pvsm")
        # saved_files['paraview_state'] = state_file
        
        print(f"\nComplete simulation export finished:")
        print(f"Base filename: {base_filename}")
        print(f"Export type: {dimensions.upper()}")
        if dimensions == '2d':
            print(f"Plane: {plane}, Position: {position}, Thickness: {slab_thickness}")
        print(f"Files created: {len(saved_files)}")
        for data_type, filepath in saved_files.items():
            print(f"  {data_type}: {os.path.basename(filepath)}")
        
        return saved_files
    
    def _calculate_density_for_export(self, positions, method, dimensions, plane, position, 
                                    slab_thickness, resolution, normalize, **kwargs):
        """Calculate density data for export using the density module."""
        try:
            from .density import DensityCalculator
        except ImportError:
            print("Warning: Density module not available. Skipping density calculation.")
            return None
        
        calculator = DensityCalculator(positions)
        
        if dimensions == '2d':
            X, Y, density = calculator.calculate_density(
                method=method,
                dimensions='2d',
                plane=plane,
                position=position,
                slab_thickness=slab_thickness,
                resolution=resolution,
                normalize=normalize,
                **kwargs
            )
            
            return {
                'X': X,
                'Y': Y,
                'density': density,
                'dimensions': '2d',
                'plane': plane,
                'position': position
            }
        else:
            X, Y, Z, density = calculator.calculate_density(
                method=method,
                dimensions='3d',
                resolution=resolution,
                normalize=normalize,
                **kwargs
            )
            
            return {
                'X': X,
                'Y': Y,
                'Z': Z,
                'density': density,
                'dimensions': '3d'
            }
    
    def _extract_2d_data(self, positions, point_data, plane, position, slab_thickness):
        """Extract 2D positions and point data from 3D data."""
        if positions.shape[1] != 3:
            return positions, point_data
        
        # Define plane mapping
        plane_maps = {
            'xy': (0, 1, 2),
            'xz': (0, 2, 1), 
            'yz': (1, 2, 0)
        }
        
        axis1, axis2, axis3 = plane_maps[plane]
        
        # Filter particles in slab
        mask = np.abs(positions[:, axis3] - position) <= slab_thickness / 2
        pos_2d = positions[mask][:, [axis1, axis2]]
        
        # Filter point data if provided
        filtered_point_data = {}
        if point_data is not None:
            for name, data in point_data.items():
                if len(data) == len(positions):
                    filtered_point_data[name] = data[mask]
                else:
                    filtered_point_data[name] = data
        
        return pos_2d, filtered_point_data if point_data is not None else None
    
    def _convert_2d_to_3d(self, positions_2d, plane, position):
        """Convert 2D positions to 3D for VTK export."""
        n_particles = len(positions_2d)
        positions_3d = np.zeros((n_particles, 3))
        
        