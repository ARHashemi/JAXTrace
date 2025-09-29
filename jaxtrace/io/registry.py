# jaxtrace/io/registry.py
"""
JAXTrace I/O registry for automatic format detection and loading.

This module provides a centralized registry system for different data formats,
with automatic format detection and appropriate reader selection.
Uses enhanced VTK support only - no legacy dependencies.
"""

import os
import glob
import warnings
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

# Enhanced VTK support (only modern readers)
ENHANCED_VTK_AVAILABLE = False
HDF5_AVAILABLE = False

# Enhanced VTK readers (our primary VTK support)
try:
    from .vtk_reader import (
        VTKStructuredSeries,
        VTKUnstructuredTimeSeriesReader,
        open_vtk_time_series,
        open_vtk_structured_series,
        open_vtk_dataset
    )
    ENHANCED_VTK_AVAILABLE = True
    print("✅ Enhanced VTK I/O available")
except ImportError as e:
    VTKStructuredSeries = None
    VTKUnstructuredTimeSeriesReader = None
    open_vtk_time_series = None
    open_vtk_structured_series = None
    open_vtk_dataset = None
    warnings.warn(f"VTK support not available: {e}")

# HDF5 support
try:
    from .hdf5_io import H5Series
    HDF5_AVAILABLE = True
    print("✅ HDF5 I/O available")
except ImportError:
    H5Series = None
    warnings.warn("HDF5 support not available - install h5py package")

# VTK is available only if enhanced support is available
VTK_AVAILABLE = ENHANCED_VTK_AVAILABLE

# Registry of supported formats and their characteristics
_format_registry = {
    "vtk": {
        "extensions": [".vti", ".vts", ".vtr", ".pvtu", ".vtu", ".vtk", ".vtp"],
        "description": "VTK (Visualization Toolkit) format",
        "reader_class": None,  # Determined dynamically
        "supports_time_series": True,
        "supports_structured": True,
        "supports_unstructured": True,
        "available": VTK_AVAILABLE
    },
    "hdf5": {
        "extensions": [".h5", ".hdf5"],
        "description": "HDF5 hierarchical data format", 
        "reader_class": H5Series,
        "supports_time_series": True,
        "supports_structured": True,
        "supports_unstructured": False,
        "available": HDF5_AVAILABLE
    }
}

# File extension to format mapping
_extension_to_format = {}
for format_name, info in _format_registry.items():
    for ext in info["extensions"]:
        _extension_to_format[ext.lower()] = format_name


def _detect_vtk_format_type(files: List[str]) -> str:
    """
    Detect specific VTK format type from file extensions.
    
    Returns
    -------
    str
        'structured', 'unstructured', 'polydata', or 'mixed'
    """
    structured_exts = {'.vti', '.vts', '.vtr'}
    unstructured_exts = {'.pvtu', '.vtu', '.vtk'}
    polydata_exts = {'.vtp'}
    
    file_exts = set()
    for file in files:
        ext = os.path.splitext(file.lower())[1]
        file_exts.add(ext)
    
    has_structured = bool(file_exts & structured_exts)
    has_unstructured = bool(file_exts & unstructured_exts)
    has_polydata = bool(file_exts & polydata_exts)
    
    if has_structured and has_unstructured:
        return 'mixed'
    elif has_structured:
        return 'structured'
    elif has_unstructured:
        return 'unstructured'
    elif has_polydata:
        return 'polydata'
    else:
        return 'unknown'


def _select_vtk_reader(files: List[str], vtk_format_type: str) -> Any:
    """
    Select the appropriate VTK reader for the given files.
    
    Parameters
    ----------
    files : List[str]
        List of VTK files
    vtk_format_type : str
        VTK format type ('structured', 'unstructured', etc.)
        
    Returns
    -------
    Any
        Appropriate VTK reader instance or data
    """
    if not VTK_AVAILABLE:
        raise ImportError("VTK support not available - install vtk package")
    
    # For time series of multiple files
    if len(files) > 1:
        if vtk_format_type in ('unstructured', 'mixed'):
            # Use enhanced unstructured reader for time series
            file_pattern = _create_file_pattern(files)
            return open_vtk_time_series(file_pattern)
        elif vtk_format_type == 'structured':
            # Use enhanced structured reader
            return open_vtk_structured_series(files)
        else:
            # Try with the first file as pattern
            file_pattern = _create_file_pattern(files)
            return open_vtk_dataset(file_pattern)
    
    # For single files
    else:
        single_file = files[0]
        return open_vtk_dataset(single_file)


def _create_file_pattern(files: List[str]) -> str:
    """
    Create a glob pattern from a list of files for time series reading.
    
    Parameters
    ----------
    files : List[str]
        List of files
        
    Returns
    -------
    str
        Glob pattern that matches the files
    """
    if not files:
        return ""
    
    # Get the directory and try to create a pattern
    first_file = files[0]
    directory = os.path.dirname(first_file)
    basename = os.path.basename(first_file)
    
    # Try to find common pattern
    # Look for numeric sequences in filename
    import re
    
    # Replace numeric sequences with wildcards
    pattern = re.sub(r'\d+', '*', basename)
    
    if directory:
        return os.path.join(directory, pattern)
    else:
        return pattern


def _get_file_list(path_spec: Union[str, List[str]]) -> List[str]:
    """
    Get list of files from various path specifications.
    
    Parameters
    ----------
    path_spec : str or List[str]
        Path specification (file, glob pattern, or list)
        
    Returns
    -------
    List[str]
        List of files
    """
    if isinstance(path_spec, list):
        return path_spec
    elif isinstance(path_spec, str):
        # Check if it's a glob pattern
        if any(char in path_spec for char in ['*', '?', '[']):
            files = glob.glob(path_spec)
            return sorted(files)
        else:
            # Single file or directory
            path = Path(path_spec)
            if path.is_file():
                return [str(path)]
            elif path.is_dir():
                # Get all supported files in directory
                all_files = []
                for ext in _extension_to_format.keys():
                    all_files.extend(path.glob(f"*{ext}"))
                return [str(f) for f in sorted(all_files)]
            else:
                return [path_spec]  # File might not exist yet
    else:
        raise TypeError(f"Unsupported path specification type: {type(path_spec)}")


def detect_format(path_spec: Union[str, List[str]]) -> Optional[str]:
    """
    Automatically detect data format from file path(s).
    
    Parameters
    ----------
    path_spec : str or List[str]
        Path(s) to data file(s)
        
    Returns
    -------
    Optional[str]
        Detected format name, or None if unknown
    """
    files = _get_file_list(path_spec)
    
    if not files:
        return None
    
    # Check first file's extension
    first_file = files[0]
    ext = os.path.splitext(first_file.lower())[1]
    
    return _extension_to_format.get(ext)


def open_dataset(
    path_spec: Union[str, List[str]], 
    format_type: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Open dataset with automatic format detection.
    
    Parameters
    ----------
    path_spec : str or List[str]
        Path specification:
        - Single file: "/path/to/data.vti"
        - Glob pattern: "/path/to/data_*.pvtu" 
        - File list: ["file1.vtu", "file2.vtu"]
        - Directory: "/path/to/data/" (opens all supported files)
    format_type : str, optional
        Force specific format ('vtk', 'hdf5'). If None, auto-detect.
    **kwargs
        Additional arguments passed to the reader
        
    Returns
    -------
    Any
        Dataset reader instance or loaded data
        
    Raises
    ------
    FileNotFoundError
        If no files found matching the specification
    ValueError
        If format cannot be determined or is unsupported
    ImportError
        If required backend is not available
    """
    # Get file list
    files = _get_file_list(path_spec)
    
    if not files:
        raise FileNotFoundError(f"No files found matching: {path_spec}")
    
    # Filter to existing files only
    existing_files = [f for f in files if os.path.exists(f)]
    
    if not existing_files:
        raise FileNotFoundError(f"No existing files found from: {files}")
    
    files = existing_files
    
    # Detect format if not specified
    if format_type is None:
        format_type = detect_format(files)
        if format_type is None:
            # Try to infer from content or give helpful error
            sample_file = files[0]
            ext = os.path.splitext(sample_file.lower())[1]
            supported_exts = list(_extension_to_format.keys())
            raise ValueError(
                f"Unknown file format '{ext}' for file: {sample_file}\n"
                f"Supported extensions: {supported_exts}"
            )
    
    # Check if format is supported
    if format_type not in _format_registry:
        raise ValueError(f"Unsupported format: {format_type}")
    
    format_info = _format_registry[format_type]
    
    if not format_info["available"]:
        raise ImportError(
            f"{format_type.upper()} backend not available. "
            f"Install required package for {format_info['description']}"
        )
    
    print(f"📁 Opening {format_type.upper()} dataset: {len(files)} file(s)")
    
    # Handle different formats
    try:
        if format_type == "vtk":
            vtk_format_type = _detect_vtk_format_type(files)
            print(f"   🎯 VTK format type: {vtk_format_type}")
            return _select_vtk_reader(files, vtk_format_type)
            
        elif format_type == "hdf5":
            if len(files) == 1:
                return H5Series(files[0], **kwargs)
            else:
                return H5Series(files, **kwargs)
        
        else:
            raise ValueError(f"Format '{format_type}' recognized but not implemented")
    
    except Exception as e:
        raise RuntimeError(
            f"Failed to open {format_type.upper()} dataset: {e}\n"
            f"Files: {files[:3]}{'...' if len(files) > 3 else ''}"
        ) from e


def list_supported_formats() -> Dict[str, List[str]]:
    """
    List all supported data formats and their extensions.
    
    Returns
    -------
    Dict[str, List[str]]
        Mapping from format name to list of supported extensions
    """
    return {
        name: info["extensions"] 
        for name, info in _format_registry.items()
        if info["available"]
    }


def format_info(format_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific format.
    
    Parameters
    ----------
    format_name : str
        Format name (e.g., 'vtk', 'hdf5')
        
    Returns
    -------
    Dict[str, Any]
        Format information including capabilities and availability
    """
    if format_name not in _format_registry:
        raise ValueError(f"Unknown format: {format_name}")
    
    return _format_registry[format_name].copy()


def get_format_capabilities() -> Dict[str, Dict[str, Any]]:
    """
    Get capabilities matrix for all available formats.
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Format capabilities organized by format name
    """
    capabilities = {}
    
    for format_name, info in _format_registry.items():
        if info["available"]:
            capabilities[format_name] = {
                'extensions': info["extensions"],
                'time_series': info["supports_time_series"],
                'structured': info["supports_structured"], 
                'unstructured': info["supports_unstructured"],
                'description': info["description"]
            }
    
    return capabilities


# Backwards compatibility functions (maintained from original)
def open_vtk_series(path_spec: Union[str, List[str]], **kwargs) -> Any:
    """
    Open VTK series (backwards compatibility).
    
    Parameters
    ----------
    path_spec : str or List[str]
        VTK file specification
    **kwargs
        Additional reader arguments
        
    Returns
    -------
    Any
        VTK dataset
    """
    return open_dataset(path_spec, format_type="vtk", **kwargs)


def open_hdf5_series(path_spec: Union[str, List[str]], **kwargs) -> Any:
    """
    Open HDF5 series (backwards compatibility).
    
    Parameters
    ----------
    path_spec : str or List[str]
        HDF5 file specification  
    **kwargs
        Additional reader arguments
        
    Returns
    -------
    Any
        HDF5 dataset
    """
    return open_dataset(path_spec, format_type="hdf5", **kwargs)


# Status and diagnostic functions
def print_registry_status():
    """Print detailed registry status information."""
    print("JAXTrace I/O Registry Status")
    print("=" * 40)
    
    print(f"Enhanced VTK: {'✅' if ENHANCED_VTK_AVAILABLE else '❌'}")
    print(f"HDF5:         {'✅' if HDF5_AVAILABLE else '❌'}")
    print()
    
    available_formats = [name for name, info in _format_registry.items() if info["available"]]
    print(f"Available formats: {len(available_formats)}")
    
    if available_formats:
        capabilities = get_format_capabilities()
        
        for format_name in available_formats:
            info = capabilities[format_name]
            print(f"\n{format_name.upper()}:")
            print(f"  Extensions: {', '.join(info['extensions'])}")
            print(f"  Time series: {'✅' if info['time_series'] else '❌'}")
            print(f"  Structured: {'✅' if info['structured'] else '❌'}")
            print(f"  Unstructured: {'✅' if info['unstructured'] else '❌'}")
            print(f"  Description: {info['description']}")
    
    if not available_formats:
        print("\n⚠️  No I/O backends available!")
        print("Install backends with:")
        print("  pip install vtk       # For VTK support")
        print("  pip install h5py      # For HDF5 support")


def validate_file_access(path_spec: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Validate file access and provide diagnostic information.
    
    Parameters
    ----------
    path_spec : str or List[str]
        Path specification to validate
        
    Returns
    -------
    Dict[str, Any]
        Validation results and diagnostic information
    """
    validation = {
        'valid': False,
        'files_found': [],
        'files_missing': [],
        'detected_format': None,
        'format_supported': False,
        'errors': []
    }
    
    try:
        files = _get_file_list(path_spec)
        validation['files_requested'] = files
        
        for file in files:
            if os.path.exists(file):
                validation['files_found'].append(file)
            else:
                validation['files_missing'].append(file)
        
        if validation['files_found']:
            try:
                format_type = detect_format(validation['files_found'])
                validation['detected_format'] = format_type
                
                if format_type:
                    format_info_dict = _format_registry.get(format_type, {})
                    validation['format_supported'] = format_info_dict.get('available', False)
                
            except Exception as e:
                validation['errors'].append(f"Format detection failed: {e}")
        
        validation['valid'] = (
            len(validation['files_found']) > 0 and
            validation['format_supported'] and
            len(validation['errors']) == 0
        )
        
    except Exception as e:
        validation['errors'].append(f"File validation failed: {e}")
    
    return validation


# Initialize registry status on import
if __name__ == "__main__":
    print_registry_status()