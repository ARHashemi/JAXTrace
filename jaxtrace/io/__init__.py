# jaxtrace/io/__init__.py  
"""  
JAXTrace I/O module with comprehensive VTK support.  

Supports both structured and unstructured grids, time series data,  
and trajectory export with consistent JAXTrace data formats.  

Main entry points:  
- open_dataset() - Universal dataset opener with auto-format detection  
- VTK readers/writers - Direct VTK functionality  
- Registry system - Extensible format support  
"""  

import numpy as np  
import warnings  
from pathlib import Path  

# Import registry functionality (your main I/O interface)  
from .registry import (  
    open_dataset,  
    list_supported_formats,  
    format_info,  
    get_format_capabilities,  
    print_registry_status,  
    validate_file_access,  
    # Backwards compatibility functions  
    open_vtk_series,  
    open_hdf5_series  
)  

# Enhanced VTK functionality (conditional import with comprehensive error handling)  
VTK_IO_AVAILABLE = False  
VTK_STRUCTURED_AVAILABLE = False  

try:  
    from .vtk_reader import (  
        VTKUnstructuredTimeSeriesReader,  
        VTKStructuredSeries,  
        open_vtk_time_series,  
        open_vtk_structured_series,  
        open_vtk_dataset,  
        _ensure_float32,  
        _ensure_3d_positions,  
        _ensure_3d_velocity,  
        _natural_sort_key,  
        GridMeta  
    )  
    
    # Try to import VTK writers, create basic versions if not available  
    try:  
        from .vtk_writer import (  
            VTKTrajectoryWriter,  
            VTKFieldWriter,  
            write_structured_grid  
        )  
    except ImportError:  
        # Create basic VTK writers if the full module isn't available  
        pass  # We'll define them below  
    
    VTK_IO_AVAILABLE = True  
    VTK_STRUCTURED_AVAILABLE = True  
    
except ImportError as e:  
    warnings.warn(f"Enhanced VTK I/O functionality not available: {e}")  
    VTK_IO_AVAILABLE = False  
    VTK_STRUCTURED_AVAILABLE = False  


# Define VTK writer classes and functions  
if VTK_IO_AVAILABLE:  
    try:  
        import vtk  
        from vtk.util import numpy_support  
        
        class VTKTrajectoryWriter:  
            """Enhanced VTK trajectory writer with comprehensive features."""  
            
            def __init__(self):  
                self.format = 'xml'  
                self.compression = True  
            
            def write_trajectory(self, trajectory, filename, include_velocities=True,   
                               include_particle_ids=True, include_timestep_data=True, format='xml'):  
                """  
                Write trajectory to VTK polydata format.  
                
                Args:  
                    trajectory: JAXTrace trajectory object  
                    filename: Output filename  
                    include_velocities: Include velocity data if available  
                    include_particle_ids: Include particle IDs  
                    include_timestep_data: Include timestep information  
                    format: 'xml' or 'binary'  
                """  
                print(f"Writing trajectory to {filename}...")  
                
                # Create polydata for trajectory lines  
                polydata = vtk.vtkPolyData()  
                points = vtk.vtkPoints()  
                lines = vtk.vtkCellArray()  
                
                # Point and cell data arrays  
                if include_particle_ids:  
                    particle_id_array = vtk.vtkIntArray()  
                    particle_id_array.SetName("ParticleID")  
                
                if include_timestep_data:  
                    timestep_array = vtk.vtkFloatArray()  
                    timestep_array.SetName("TimeStep")  
                
                if include_velocities and trajectory.velocities is not None:  
                    velocity_array = vtk.vtkFloatArray()  
                    velocity_array.SetNumberOfComponents(3)  
                    velocity_array.SetName("Velocity")  
                
                point_id = 0  
                
                # Add trajectory data for each particle  
                for particle_idx in range(trajectory.N):  
                    # Create line for this particle's trajectory  
                    line = vtk.vtkPolyLine()  
                    line.GetPointIds().SetNumberOfIds(trajectory.T)  
                    
                    for time_idx in range(trajectory.T):  
                        pos = trajectory.positions[time_idx, particle_idx]  
                        points.InsertNextPoint(pos[0], pos[1], pos[2])  
                        
                        # Add point data  
                        if include_particle_ids:  
                            particle_id_array.InsertNextValue(particle_idx)  
                        
                        if include_timestep_data:  
                            timestep_array.InsertNextValue(time_idx)  
                        
                        if include_velocities and trajectory.velocities is not None:  
                            vel = trajectory.velocities[time_idx, particle_idx]  
                            velocity_array.InsertNextTuple3(vel[0], vel[1], vel[2])  
                        
                        line.GetPointIds().SetId(time_idx, point_id)  
                        point_id += 1  
                    
                    lines.InsertNextCell(line)  
                
                # Set up polydata  
                polydata.SetPoints(points)  
                polydata.SetLines(lines)  
                
                # Add point data  
                if include_particle_ids:  
                    polydata.GetPointData().AddArray(particle_id_array)  
                
                if include_timestep_data:  
                    polydata.GetPointData().AddArray(timestep_array)  
                
                if include_velocities and trajectory.velocities is not None:  
                    polydata.GetPointData().SetVectors(velocity_array)  
                
                # Write to file  
                if format == 'xml':  
                    writer = vtk.vtkXMLPolyDataWriter()  
                else:  
                    writer = vtk.vtkPolyDataWriter()  
                
                writer.SetFileName(filename)  
                writer.SetInputData(polydata)  
                
                if format == 'xml' and self.compression:  
                    writer.SetCompressorTypeToZLib()  
                
                writer.Write()  
                print(f"Trajectory written successfully to {filename}")  
            
            def write_time_series(self, trajectory, output_directory, filename_pattern="trajectory_{:04d}.vtu",  
                                include_velocities=True, format='xml'):  
                """  
                Write trajectory as time series of particle positions.  
                
                Args:  
                    trajectory: JAXTrace trajectory object  
                    output_directory: Output directory path  
                    filename_pattern: Filename pattern with time index  
                    include_velocities: Include velocity data if available  
                    format: 'xml' or 'binary'  
                """  
                output_path = Path(output_directory)  
                output_path.mkdir(exist_ok=True)  
                
                print(f"Writing trajectory time series to {output_directory}...")  
                
                for time_idx in range(trajectory.T):  
                    filename = output_path / filename_pattern.format(time_idx)  
                    
                    # Create unstructured grid for particles at this timestep  
                    ugrid = vtk.vtkUnstructuredGrid()  
                    points = vtk.vtkPoints()  
                    
                    # Add points (particles)  
                    for particle_idx in range(trajectory.N):  
                        pos = trajectory.positions[time_idx, particle_idx]  
                        points.InsertNextPoint(pos[0], pos[1], pos[2])  
                    
                    ugrid.SetPoints(points)  
                    
                    # Add particle IDs as point data  
                    particle_id_array = vtk.vtkIntArray()  
                    particle_id_array.SetName("ParticleID")  
                    for particle_idx in range(trajectory.N):  
                        particle_id_array.InsertNextValue(particle_idx)  
                    ugrid.GetPointData().AddArray(particle_id_array)  
                    
                    # Add velocities if available  
                    if include_velocities and trajectory.velocities is not None:  
                        velocity_array = vtk.vtkFloatArray()  
                        velocity_array.SetNumberOfComponents(3)  
                        velocity_array.SetName("Velocity")  
                        
                        for particle_idx in range(trajectory.N):  
                            vel = trajectory.velocities[time_idx, particle_idx]  
                            velocity_array.InsertNextTuple3(vel[0], vel[1], vel[2])  
                        
                        ugrid.GetPointData().SetVectors(velocity_array)  
                    
                    # Create vertex cells for visualization  
                    for particle_idx in range(trajectory.N):  
                        vertex = vtk.vtkVertex()  
                        vertex.GetPointIds().SetId(0, particle_idx)  
                        ugrid.InsertNextCell(vertex.GetCellType(), vertex.GetPointIds())  
                    
                    # Write to file  
                    if format == 'xml':  
                        writer = vtk.vtkXMLUnstructuredGridWriter()  
                    else:  
                        writer = vtk.vtkUnstructuredGridWriter()  
                    
                    writer.SetFileName(str(filename))  
                    writer.SetInputData(ugrid)  
                    
                    if format == 'xml' and self.compression:  
                        writer.SetCompressorTypeToZLib()  
                    
                    writer.Write()  
                
                print(f"Time series written: {trajectory.T} files")  
            
            def write_particles_at_time(self, positions, velocities=None, time=0, filename=None, format='xml'):  
                """  
                Write particles at a specific time to VTK file.  
                
                Args:  
                    positions: Particle positions (N, 3)  
                    velocities: Particle velocities (N, 3), optional  
                    time: Time value  
                    filename: Output filename  
                    format: 'xml' or 'binary'  
                """  
                ugrid = vtk.vtkUnstructuredGrid()  
                points = vtk.vtkPoints()  
                
                # Add points  
                for pos in positions:  
                    points.InsertNextPoint(pos[0], pos[1], pos[2])  
                ugrid.SetPoints(points)  
                
                # Add time information  
                time_array = vtk.vtkFloatArray()  
                time_array.SetName("Time")  
                time_array.InsertNextValue(time)  
                ugrid.GetFieldData().AddArray(time_array)  
                
                # Add velocities if provided  
                if velocities is not None:  
                    velocity_array = vtk.vtkFloatArray()  
                    velocity_array.SetNumberOfComponents(3)  
                    velocity_array.SetName("Velocity")  
                    
                    for vel in velocities:  
                        velocity_array.InsertNextTuple3(vel[0], vel[1], vel[2])  
                    
                    ugrid.GetPointData().SetVectors(velocity_array)  
                
                # Create vertex cells  
                for i in range(len(positions)):  
                    vertex = vtk.vtkVertex()  
                    vertex.GetPointIds().SetId(0, i)  
                    ugrid.InsertNextCell(vertex.GetCellType(), vertex.GetPointIds())  
                
                # Write file  
                if format == 'xml':  
                    writer = vtk.vtkXMLUnstructuredGridWriter()  
                else:  
                    writer = vtk.vtkUnstructuredGridWriter()  
                
                writer.SetFileName(filename)  
                writer.SetInputData(ugrid)  
                
                if format == 'xml' and self.compression:  
                    writer.SetCompressorTypeToZLib()  
                
                writer.Write()  


        class VTKFieldWriter:  
            """Enhanced VTK field writer for structured data."""  
            
            def __init__(self):  
                self.format = 'xml'  
                self.compression = True  
            
            def write_structured_2d(self, X, Y, field_data, filename, field_name="field", format='xml'):  
                """  
                Write 2D structured field data to VTK image data format.  
                
                Args:  
                    X, Y: Coordinate meshgrids  
                    field_data: Field values on the grid  
                    filename: Output filename  
                    field_name: Name for the field data  
                    format: 'xml' or 'binary'  
                """  
                print(f"Writing structured field to {filename}...")  
                
                # Create image data (structured grid)  
                image_data = vtk.vtkImageData()  
                
                # Set dimensions  
                nx, ny = X.shape  
                image_data.SetDimensions(nx, ny, 1)  
                
                # Set spacing and origin  
                dx = (np.max(X) - np.min(X)) / (nx - 1) if nx > 1 else 1.0  
                dy = (np.max(Y) - np.min(Y)) / (ny - 1) if ny > 1 else 1.0  
                
                image_data.SetSpacing(dx, dy, 1.0)  
                image_data.SetOrigin(np.min(X), np.min(Y), 0.0)  
                
                # Add field data  
                field_array = numpy_support.numpy_to_vtk(field_data.ravel(order='F'), deep=True)  
                field_array.SetName(field_name)  
                image_data.GetPointData().SetScalars(field_array)  
                
                # Write to file  
                if format == 'xml':  
                    writer = vtk.vtkXMLImageDataWriter()  
                else:  
                    writer = vtk.vtkImageDataWriter()  
                
                writer.SetFileName(filename)  
                writer.SetInputData(image_data)  
                
                if format == 'xml' and self.compression:  
                    writer.SetCompressorTypeToZLib()  
                
                writer.Write()  
                print(f"Structured field written to {filename}")  
            
            def write_structured_3d(self, X, Y, Z, field_data, filename, field_name="field", format='xml'):  
                """  
                Write 3D structured field data to VTK.  
                
                Args:  
                    X, Y, Z: Coordinate meshgrids  
                    field_data: Field values on the grid  
                    filename: Output filename  
                    field_name: Name for the field data  
                    format: 'xml' or 'binary'  
                """  
                image_data = vtk.vtkImageData()  
                
                # Set dimensions  
                nx, ny, nz = X.shape  
                image_data.SetDimensions(nx, ny, nz)  
                
                # Set spacing and origin  
                dx = (np.max(X) - np.min(X)) / (nx - 1) if nx > 1 else 1.0  
                dy = (np.max(Y) - np.min(Y)) / (ny - 1) if ny > 1 else 1.0  
                dz = (np.max(Z) - np.min(Z)) / (nz - 1) if nz > 1 else 1.0  
                
                image_data.SetSpacing(dx, dy, dz)  
                image_data.SetOrigin(np.min(X), np.min(Y), np.min(Z))  
                
                # Add field data  
                field_array = numpy_support.numpy_to_vtk(field_data.ravel(order='F'), deep=True)  
                field_array.SetName(field_name)  
                image_data.GetPointData().SetScalars(field_array)  
                
                # Write to file  
                if format == 'xml':                    
                    writer = vtk.vtkXMLImageDataWriter()
                else:
                    writer = vtk.vtkImageDataWriter()
                
                writer.SetFileName(filename)
                writer.SetInputData(image_data)
                
                if format == 'xml' and self.compression:
                    writer.SetCompressorTypeToZLib()
                
                writer.Write()
                print(f"3D structured field written to {filename}")
            
            def write_unstructured_field(self, positions, field_data, filename, field_name="field", format='xml'):
                """
                Write unstructured field data to VTK.
                
                Args:
                    positions: Point positions (N, 3)
                    field_data: Field values at points (N,)
                    filename: Output filename
                    field_name: Name for the field data
                    format: 'xml' or 'binary'
                """
                ugrid = vtk.vtkUnstructuredGrid()
                points = vtk.vtkPoints()
                
                # Add points
                for pos in positions:
                    points.InsertNextPoint(pos[0], pos[1], pos[2])
                ugrid.SetPoints(points)
                
                # Add field data
                if field_data.ndim == 1:
                    # Scalar field
                    field_array = numpy_support.numpy_to_vtk(field_data, deep=True)
                    field_array.SetName(field_name)
                    ugrid.GetPointData().SetScalars(field_array)
                elif field_data.ndim == 2 and field_data.shape[1] == 3:
                    # Vector field
                    field_array = numpy_support.numpy_to_vtk(field_data, deep=True)
                    field_array.SetName(field_name)
                    field_array.SetNumberOfComponents(3)
                    ugrid.GetPointData().SetVectors(field_array)
                
                # Create vertex cells
                for i in range(len(positions)):
                    vertex = vtk.vtkVertex()
                    vertex.GetPointIds().SetId(0, i)
                    ugrid.InsertNextCell(vertex.GetCellType(), vertex.GetPointIds())
                
                # Write file
                if format == 'xml':
                    writer = vtk.vtkXMLUnstructuredGridWriter()
                else:
                    writer = vtk.vtkUnstructuredGridWriter()
                
                writer.SetFileName(filename)
                writer.SetInputData(ugrid)
                
                if format == 'xml' and self.compression:
                    writer.SetCompressorTypeToZLib()
                
                writer.Write()
                print(f"Unstructured field written to {filename}")


        def write_structured_grid(X, Y, field_data, filename, field_name="field", format='xml'):
            """
            Convenience function to write 2D structured grid.
            
            Args:
                X, Y: Coordinate meshgrids
                field_data: Field values on the grid
                filename: Output filename
                field_name: Name for the field data
                format: 'xml' or 'binary'
            """
            writer = VTKFieldWriter()
            writer.write_structured_2d(X, Y, field_data, filename, field_name, format)

    except ImportError:
        # VTK not available - create dummy classes
        class VTKTrajectoryWriter:
            def __init__(self):
                pass
            
            def write_trajectory(self, *args, **kwargs):
                raise ImportError("VTK not available - cannot write trajectory files")
            
            def write_time_series(self, *args, **kwargs):
                raise ImportError("VTK not available - cannot write time series files")
            
            def write_particles_at_time(self, *args, **kwargs):
                raise ImportError("VTK not available - cannot write particle files")

        class VTKFieldWriter:
            def __init__(self):
                pass
            
            def write_structured_2d(self, *args, **kwargs):
                raise ImportError("VTK not available - cannot write structured fields")
            
            def write_structured_3d(self, *args, **kwargs):
                raise ImportError("VTK not available - cannot write structured fields")
            
            def write_unstructured_field(self, *args, **kwargs):
                raise ImportError("VTK not available - cannot write unstructured fields")

        def write_structured_grid(*args, **kwargs):
            raise ImportError("VTK not available - cannot write structured grids")

else:
    # VTK I/O not available at all - create dummy classes
    class VTKTrajectoryWriter:
        def __init__(self):
            pass
        
        def write_trajectory(self, *args, **kwargs):
            raise ImportError("VTK I/O not available - install vtk package")
        
        def write_time_series(self, *args, **kwargs):
            raise ImportError("VTK I/O not available - install vtk package")
        
        def write_particles_at_time(self, *args, **kwargs):
            raise ImportError("VTK I/O not available - install vtk package")

    class VTKFieldWriter:
        def __init__(self):
            pass
        
        def write_structured_2d(self, *args, **kwargs):
            raise ImportError("VTK I/O not available - install vtk package")
        
        def write_structured_3d(self, *args, **kwargs):
            raise ImportError("VTK I/O not available - install vtk package")
        
        def write_unstructured_field(self, *args, **kwargs):
            raise ImportError("VTK I/O not available - install vtk package")

    def write_structured_grid(*args, **kwargs):
        raise ImportError("VTK I/O not available - install vtk package")


# def export_trajectory_to_vtk(trajectory, filename, include_velocities=True, 
#                            include_metadata=True, time_series=False, format='xml'):
#     """
#     Export trajectory to VTK format (main export function).
    
#     Args:
#         trajectory: JAXTrace trajectory object
#         filename: Output filename
#         include_velocities: Include velocity data if available
#         include_metadata: Include metadata (particle IDs, timesteps)
#         time_series: Export as time series if True, single file if False
#         format: 'xml' or 'binary'
#     """
#     if not VTK_IO_AVAILABLE:
#         raise ImportError("VTK not available - cannot export trajectory")
    
#     writer = VTKTrajectoryWriter()
    
#     if time_series:
#         # Export as time series to directory
#         output_dir = Path(filename).parent / (Path(filename).stem + "_series")
#         writer.write_time_series(
#             trajectory=trajectory,
#             output_directory=str(output_dir),
#             include_velocities=include_velocities,
#             format=format
#         )
#         print(f"Trajectory exported as time series to {output_dir}")
#     else:
#         # Export as single trajectory file
#         writer.write_trajectory(
#             trajectory=trajectory,
#             filename=filename,
#             include_velocities=include_velocities,
#             include_particle_ids=include_metadata,
#             include_timestep_data=include_metadata,
#             format=format
#         )
#         print(f"Trajectory exported to {filename}")
def export_trajectory_to_vtk(trajectory, filename, include_velocities=True, 
                           include_metadata=True, time_series=False, format='xml'):
    """
    Export trajectory to VTK format (main export function).

    Returns
    -------
    dict
        - If time_series=False:
            {'mode': 'single', 'file': '<path>'}
        - If time_series=True:
            {'mode': 'series', 'directory': '<dir>', 'count': <int>}
    """
    if not VTK_IO_AVAILABLE:
        raise ImportError("VTK not available - cannot export trajectory")
    
    writer = VTKTrajectoryWriter()
    
    if time_series:
        output_dir = Path(filename).parent / (Path(filename).stem + "_series")
        writer.write_time_series(
            trajectory=trajectory,
            output_directory=str(output_dir),
            include_velocities=include_velocities,
            format=format
        )
        print(f"Trajectory exported as time series to {output_dir}")
        return {'mode': 'series', 'directory': str(output_dir), 'count': int(trajectory.T)}
    else:
        writer.write_trajectory(
            trajectory=trajectory,
            filename=filename,
            include_velocities=include_velocities,
            include_particle_ids=include_metadata,
            include_timestep_data=include_metadata,
            format=format
        )
        print(f"Trajectory exported to {filename}")
        return {'mode': 'single', 'file': str(filename)}

# Status and utility functions
def get_vtk_status():
    """Get VTK functionality status."""
    status = {
        'vtk_available': VTK_IO_AVAILABLE,
        'structured_grids': VTK_STRUCTURED_AVAILABLE,
        'trajectory_export': VTK_IO_AVAILABLE,
        'field_export': VTK_IO_AVAILABLE
    }
    
    if VTK_IO_AVAILABLE:
        try:
            import vtk
            status['vtk_version'] = vtk.vtkVersion.GetVTKVersion()
        except:
            status['vtk_version'] = 'unknown'
    
    return status


def get_io_status():
    """Get comprehensive I/O system status."""
    return {
        'registry_available': True,  # Always available
        'vtk_io': get_vtk_status(),
        'supported_formats': list_supported_formats(),
        'enhanced_features': {
            'trajectory_writers': VTK_IO_AVAILABLE,
            'field_writers': VTK_IO_AVAILABLE,
            'time_series_export': VTK_IO_AVAILABLE,
            'structured_grids': VTK_STRUCTURED_AVAILABLE
        }
    }


def print_io_summary():
    """Print comprehensive I/O system status summary."""
    print("JAXTrace I/O System Status:")
    print("-" * 30)
    
    status = get_io_status()
    
    # Registry status
    print(f"Registry system: ✅ Available")
    formats = status['supported_formats']
    print(f"Supported formats: {', '.join(formats)}")
    
    # VTK status
    vtk_status = status['vtk_io']
    if vtk_status['vtk_available']:
        print(f"VTK I/O: ✅ Available (v{vtk_status.get('vtk_version', 'unknown')})")
        print(f"  - Trajectory export: {'✅' if vtk_status['trajectory_export'] else '❌'}")
        print(f"  - Field export: {'✅' if vtk_status['field_export'] else '❌'}")
        print(f"  - Structured grids: {'✅' if vtk_status['structured_grids'] else '❌'}")
    else:
        print(f"VTK I/O: ❌ Not available (install vtk package)")
    
    # Enhanced features
    enhanced = status['enhanced_features']
    available_count = sum(1 for v in enhanced.values() if v)
    total_count = len(enhanced)
    
    print(f"Enhanced features: {available_count}/{total_count} available")


# Backwards compatibility
def export_trajectory_vtk(*args, **kwargs):
    """Backwards compatibility alias."""
    return export_trajectory_to_vtk(*args, **kwargs)


# Export the main interface
__all__ = [
    # Registry system (main interface)
    'open_dataset',
    'list_supported_formats', 
    'format_info',
    'get_format_capabilities',
    'print_registry_status',
    'validate_file_access',
    
    # VTK functionality
    'export_trajectory_to_vtk',
    'VTKTrajectoryWriter',
    'VTKFieldWriter', 
    'write_structured_grid',
    
    # VTK readers (if available)
    'VTKUnstructuredTimeSeriesReader' if VTK_IO_AVAILABLE else None,
    'VTKStructuredSeries' if VTK_STRUCTURED_AVAILABLE else None,
    'open_vtk_time_series' if VTK_IO_AVAILABLE else None,
    'open_vtk_structured_series' if VTK_STRUCTURED_AVAILABLE else None,
    'open_vtk_dataset' if VTK_IO_AVAILABLE else None,
    
    # Status functions
    'get_vtk_status',
    'get_io_status', 
    'print_io_summary',
    
    # Status flags
    'VTK_IO_AVAILABLE',
    'VTK_STRUCTURED_AVAILABLE',
    
    # Backwards compatibility
    'open_vtk_series',
    'open_hdf5_series',
    'export_trajectory_vtk'
]

# Remove None entries from __all__
__all__ = [item for item in __all__ if item is not None]