# jaxtrace/fields/__init__.py
"""
JAXTrace velocity field classes with comprehensive VTK integration.

This module provides a unified interface for different types of velocity fields:
- BaseField: Base protocol for all field types
- StructuredGridSampler: Regular grid fields with trilinear interpolation
- UnstructuredField: Unstructured mesh fields with barycentric interpolation
- TimeSeriesField: Time-dependent fields with temporal interpolation

Enhanced with VTK I/O capabilities and factory functions for easy field creation.
"""

from .base import (
    BaseField, 
    Field,
    TimeDependentField,
    GridMeta,
    barycentric_coords_triangle,
    barycentric_coords_tetrahedron,
    tri6_shape_functions,
    tet10_shape_functions
)

from .structured import (
    StructuredGridSampler,
    create_structured_field_from_arrays,
    create_structured_field_from_vtk_data,
    create_uniform_grid
)

from .unstructured import (
    UnstructuredField,
    UnstructuredMesh,
    ElementType,
    create_unstructured_field_2d,
    create_unstructured_field_3d,
    create_unstructured_from_vtk_data,
    precompute_element_data,
    optimized_element_search
)

from .time_series import (
    TimeSeriesField,
    create_time_series_from_arrays,
    create_time_series_from_function,
    create_time_series_from_vtk_files
)

# Import VTK availability status
from ..utils.jax_utils import JAX_AVAILABLE

try:
    from ..io import VTK_IO_AVAILABLE
except ImportError:
    VTK_IO_AVAILABLE = False

# Re-export everything with backward compatibility aliases
__all__ = [
    # Base protocols and utilities
    'BaseField',
    'Field', 
    'TimeDependentField',
    'GridMeta',
    
    # Field implementations
    'StructuredGridSampler',
    'UnstructuredField', 
    'TimeSeriesField',
    
    # Unstructured mesh support
    'UnstructuredMesh',
    'ElementType',
    
    # Factory functions
    'create_field_from_vtk',
    'create_field_from_data',
    'create_structured_field_from_arrays',
    'create_structured_field_from_vtk_data', 
    'create_uniform_grid',
    'create_unstructured_field_2d',
    'create_unstructured_field_3d',
    'create_unstructured_from_vtk_data',
    'create_time_series_from_arrays',
    'create_time_series_from_function',
    'create_time_series_from_vtk_files',
    
    # Utilities
    'barycentric_coords_triangle',
    'barycentric_coords_tetrahedron',
    'tri6_shape_functions',
    'tet10_shape_functions',
    'precompute_element_data',
    'optimized_element_search',
    
    # Backward compatibility aliases
    'StructuredGridField',
    'StructuredVelocityField',
    'StructuredSampler',
    'UnstructuredSampler',
    'TimeDependentVelocityField',
    'TimeVaryingField',
]

# Backward compatibility aliases
StructuredGridField = StructuredGridSampler
StructuredVelocityField = StructuredGridSampler
StructuredSampler = StructuredGridSampler
UnstructuredSampler = UnstructuredField
TimeDependentVelocityField = TimeSeriesField
TimeVaryingField = TimeSeriesField


def create_field_from_vtk(file_pattern, field_type="auto", **kwargs):
    """
    Create appropriate field from VTK data with auto-detection.
    
    Parameters
    ----------
    file_pattern : str
        VTK file pattern or single file path
    field_type : str
        Field type: 'auto', 'time_series', 'structured', 'unstructured'
    **kwargs
        Additional arguments passed to field constructors
        
    Returns
    -------
    BaseField
        Appropriate field instance based on VTK data structure
        
    Raises
    ------
    ImportError
        If VTK is not available
    FileNotFoundError
        If no VTK files found matching pattern
    ValueError
        If unsupported VTK data format
    """
    if not VTK_IO_AVAILABLE:
        raise ImportError("VTK I/O not available - install VTK or check io module configuration")
    
    from ..io import open_vtk_dataset, open_vtk_time_series
    import glob
    from pathlib import Path
    
    # Get file list for analysis
    pattern_path = Path(file_pattern)
    if '*' in file_pattern or '?' in file_pattern:
        files = glob.glob(file_pattern)
    elif pattern_path.is_dir():
        # Directory provided - look for VTK files
        vtk_extensions = ['*.vtu', '*.vtp', '*.vti', '*.vts', '*.vtr', '*.vtk']
        files = []
        for ext in vtk_extensions:
            files.extend(glob.glob(str(pattern_path / ext)))
    elif pattern_path.is_file():
        files = [file_pattern]
    else:
        files = glob.glob(file_pattern)
    
    if not files:
        raise FileNotFoundError(f"No VTK files found: {file_pattern}")
    
    # Sort files to ensure proper temporal ordering
    files = sorted(files)
    sample_file = files[0].lower()
    
    print(f"🎯 Analyzing VTK data:")
    print(f"   📁 Pattern: {file_pattern}")
    print(f"   📊 Files found: {len(files)}")
    print(f"   📋 Sample file: {Path(files[0]).name}")
    
    # Auto-detect field type if requested
    if field_type == "auto":
        if len(files) > 1:
            field_type = "time_series"
        elif any(ext in sample_file for ext in ['.vti', '.vts', '.vtr']):
            field_type = "structured"  
        else:
            field_type = "unstructured"
        
        print(f"   🔍 Auto-detected type: {field_type}")
    
    # Create appropriate field type
    if field_type == "time_series":
        if len(files) == 1:
            # Single file - create time series with single time step
            print("   ⚠️  Creating single-timestep time series from one file")
            
        return create_time_series_from_vtk_files(file_pattern, **kwargs)
    
    elif field_type == "structured":
        # Load single structured file
        if len(files) > 1:
            print("   ⚠️  Multiple files provided for structured field - using first file")
        
        return create_structured_field_from_vtk_data(files[0], **kwargs)
    
    elif field_type == "unstructured":
        # For single unstructured files, load as single-timestep time series
        if len(files) > 1:
            print("   ⚠️  Multiple files provided for unstructured field - using first file")
        
        # Load single file and create unstructured field
        vtk_data = open_vtk_dataset(files[0])
        return create_unstructured_from_vtk_data(vtk_data, **kwargs)
    
    else:
        raise ValueError(f"Unknown field type: {field_type}")


def create_field_from_data(data, field_type="auto", **kwargs):
    """
    Create field from raw data arrays or dictionaries.
    
    Parameters
    ----------
    data : dict or np.ndarray
        Data in various formats:
        - dict: VTK-style data dictionary with keys like 'velocity_data', 'times', 'positions'
        - np.ndarray: Velocity data array with various possible shapes
    field_type : str
        Field type to create: 'auto', 'time_series', 'structured', 'unstructured'
    **kwargs
        Additional field parameters
        
    Returns
    -------
    BaseField
        Created field instance
        
    Raises
    ------
    ValueError
        If data format is not recognized or incompatible
    TypeError
        If unsupported data type provided
    """
    import numpy as np
    
    if isinstance(data, dict):
        # Dictionary format (e.g., from VTK reader or manual construction)
        print(f"🎯 Creating field from dictionary data:")
        print(f"   📋 Keys: {list(data.keys())}")
        
        if 'velocity_data' in data and 'times' in data and 'positions' in data:
            # Time series format
            print("   🔍 Detected time series format")
            return create_time_series_from_arrays(
                velocity_snapshots=data['velocity_data'],
                time_points=data['times'],
                positions=data['positions'],
                **kwargs
            )
        
        elif 'points' in data and 'connectivity' in data:
            # Unstructured mesh format
            print("   🔍 Detected unstructured mesh format")
            return create_unstructured_from_vtk_data(data, **kwargs)
        
        elif 'velocity_data' in data and 'grid_x' in data:
            # Structured grid format
            print("   🔍 Detected structured grid format")
            return create_structured_field_from_arrays(
                velocity_data=data['velocity_data'],
                x_coords=data['grid_x'],
                y_coords=data['grid_y'],
                z_coords=data.get('grid_z'),
                **kwargs
            )
        
        else:
            available_keys = list(data.keys())
            raise ValueError(
                f"Dictionary data format not recognized. Available keys: {available_keys}. "
                "Expected formats:\n"
                "- Time series: 'velocity_data', 'times', 'positions'\n"
                "- Unstructured: 'points', 'connectivity'\n"
                "- Structured: 'velocity_data', 'grid_x', 'grid_y', ['grid_z']"
            )
    
    elif isinstance(data, np.ndarray):
        print(f"🎯 Creating field from array data:")
        print(f"   📊 Shape: {data.shape}")
        print(f"   📋 dtype: {data.dtype}")
        
        if field_type == "auto":
            # Auto-detect based on array dimensions
            if data.ndim == 3:
                field_type = "time_series"
            elif data.ndim == 4:
                field_type = "structured"
            else:
                raise ValueError(f"Cannot auto-detect field type from array shape {data.shape}")
            
            print(f"   🔍 Auto-detected type: {field_type}")
        
        if field_type == "time_series":
            if data.ndim == 3:
                # Assume (T, N, 3) time series format
                times = kwargs.pop('times', np.arange(data.shape[0], dtype=np.float32))
                positions = kwargs.pop('positions', None)
                
                if positions is None:
                    raise ValueError(
                        "Positions required for time series field creation. "
                        "Provide positions via positions=array argument."
                    )
                
                print(f"   📊 Time steps: {data.shape[0]}")
                print(f"   📊 Spatial points: {data.shape[1]}")
                
                return create_time_series_from_arrays(
                    velocity_snapshots=data,
                    time_points=times,
                    positions=positions,
                    **kwargs
                )
            else:
                raise ValueError(f"Time series requires 3D array (T,N,3), got shape {data.shape}")
        
        elif field_type == "structured":
            if data.ndim == 4:
                # Assume (Nx, Ny, Nz, 3) structured format
                x_coords = kwargs.pop('x_coords', np.arange(data.shape[0], dtype=np.float32))
                y_coords = kwargs.pop('y_coords', np.arange(data.shape[1], dtype=np.float32))
                z_coords = kwargs.pop('z_coords', np.arange(data.shape[2], dtype=np.float32))
                
                print(f"   📊 Grid size: {data.shape[:3]}")
                
                return create_structured_field_from_arrays(
                    velocity_data=data,
                    x_coords=x_coords,
                    y_coords=y_coords,
                    z_coords=z_coords,
                    **kwargs
                )
            else:
                raise ValueError(f"Structured field requires 4D array (Nx,Ny,Nz,3), got shape {data.shape}")
        
        elif field_type == "unstructured":
            # For unstructured, need additional mesh information
            nodes = kwargs.pop('nodes', None)
            elements = kwargs.pop('elements', None)
            
            if nodes is None:
                raise ValueError("Node positions required for unstructured field")
            
            if data.ndim == 2:
                # Single time step (N, 3) 
                # Convert to time series format with single timestep
                data_3d = data[None, :, :]  # (1, N, 3)
                times = np.array([0.0], dtype=np.float32)
                
                return create_time_series_from_arrays(
                    velocity_snapshots=data_3d,
                    time_points=times,
                    positions=nodes,
                    **kwargs
                )
            else:
                raise ValueError(f"Unstructured field from array requires 2D array (N,3), got shape {data.shape}")
        
        else:
            raise ValueError(f"Unknown field type: {field_type}")
    
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected dict or np.ndarray.")


def detect_field_type(data_source):
    """
    Detect appropriate field type from various data sources.
    
    Parameters
    ----------
    data_source : str, dict, or np.ndarray
        Data source to analyze
        
    Returns
    -------
    str
        Detected field type: 'time_series', 'structured', 'unstructured'
    """
    import numpy as np
    from pathlib import Path
    import glob
    
    if isinstance(data_source, str):
        # File or pattern analysis
        if Path(data_source).is_file():
            file_ext = Path(data_source).suffix.lower()
            if file_ext in ['.vti', '.vts', '.vtr']:
                return 'structured'
            elif file_ext in ['.vtu', '.vtp']:
                return 'unstructured'
            else:
                return 'time_series'  # Default for unknown formats
        else:
            # Pattern - check if multiple files
            files = glob.glob(data_source)
            return 'time_series' if len(files) > 1 else 'unstructured'
    
    elif isinstance(data_source, dict):
        if 'times' in data_source and len(np.atleast_1d(data_source['times'])) > 1:
            return 'time_series'
        elif 'grid_x' in data_source:
            return 'structured'
        else:
            return 'unstructured'
    
    elif isinstance(data_source, np.ndarray):
        if data_source.ndim == 3:
            return 'time_series'
        elif data_source.ndim == 4:
            return 'structured'
        else:
            return 'unstructured'
    
    else:
        return 'unstructured'  # Default fallback


def list_available_fields():
    """
    List all available field types and their capabilities.
    
    Returns
    -------
    dict
        Dictionary describing available field types
    """
    return {
        'StructuredGridSampler': {
            'description': 'Regular grid with trilinear interpolation',
            'supports_time': False,
            'supports_2d': True,
            'supports_3d': True,
            'interpolation_method': 'trilinear',
            'memory_efficient': True
        },
        
        'UnstructuredField': {
            'description': 'Unstructured mesh with barycentric interpolation',
            'supports_time': False,
            'supports_2d': True, 
            'supports_3d': True,
            'interpolation_method': 'barycentric/kNN',
            'element_types': ['triangle', 'tetrahedra'],
            'element_orders': [1, 2]
        },
        
        'TimeSeriesField': {
            'description': 'Time-dependent field with temporal interpolation',
            'supports_time': True,
            'supports_2d': True,
            'supports_3d': True,
            'temporal_interpolation': ['linear', 'nearest', 'cubic'],
            'spatial_interpolation': 'nearest_neighbor',
            'extrapolation_modes': ['constant', 'linear', 'nan', 'zero']
        }
    }


def get_capabilities():
    """
    Get information about available capabilities and dependencies.
    
    Returns
    -------
    dict
        Capability information
    """
    capabilities = {
        'jax_available': JAX_AVAILABLE,
        'vtk_io_available': VTK_IO_AVAILABLE,
        'field_types': list(list_available_fields().keys()),
        'supported_formats': {
            'vtk': VTK_IO_AVAILABLE,
            'numpy_arrays': True,
            'dictionary_data': True
        }
    }
    
    if JAX_AVAILABLE:
        capabilities['jax_features'] = [
            'accelerated_interpolation',
            'automatic_differentiation', 
            'vectorized_operations'
        ]
    
    if VTK_IO_AVAILABLE:
        capabilities['vtk_formats'] = [
            'vtu', 'vtp',  # Unstructured
            'vti', 'vts', 'vtr',  # Structured
            'vtk',  # Legacy
            'pvd', 'pvtu', 'pvtp'  # Parallel/time series
        ]
    
    return capabilities


def print_field_summary():
    """Print a summary of available field types and capabilities."""
    print("🌊 JAXTrace Fields Summary")
    print("=" * 50)
    
    capabilities = get_capabilities()
    fields = list_available_fields()
    
    print(f"🔧 Dependencies:")
    print(f"   JAX: {'✅' if capabilities['jax_available'] else '❌'}")
    print(f"   VTK I/O: {'✅' if capabilities['vtk_io_available'] else '❌'}")
    print()
    
    print(f"📊 Available Field Types:")
    for name, info in fields.items():
        print(f"   • {name}")
        print(f"     └─ {info['description']}")
        if 'interpolation_method' in info:
            print(f"     └─ Interpolation: {info['interpolation_method']}")
        if info.get('supports_time'):
            print(f"     └─ Time-dependent: ✅")
    print()
    
    if capabilities['vtk_io_available']:
        print(f"🗂️  Supported VTK Formats:")
        for fmt in capabilities['vtk_formats']:
            print(f"   • .{fmt}")
    
    print()
    print("💡 Quick Start:")
    print("   from jaxtrace.fields import create_field_from_vtk")
    print("   field = create_field_from_vtk('data/*.vtu')")
    print("   velocities = field.sample_at_positions(positions, time)")


# Optional: Print summary on import for development
if __name__ == "__main__":
    print_field_summary()