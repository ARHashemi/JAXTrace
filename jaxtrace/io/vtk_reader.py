# jaxtrace/io/vtk_reader.py  
"""  
Comprehensive VTK reader supporting structured and unstructured grids.  

Handles time series data from various VTK formats with memory optimization  
and consistent JAXTrace data formats.  
"""  

from __future__ import annotations  
import os  
import glob  
import gc  
import re  
import warnings  
from pathlib import Path  
from dataclasses import dataclass  
from typing import List, Dict, Optional, Tuple, Any, Union, Iterable  
import numpy as np  

try:  
    import psutil  
    PSUTIL_AVAILABLE = True  
except ImportError:  
    PSUTIL_AVAILABLE = False  

try:  
    from scipy.spatial import cKDTree  
    SCIPY_AVAILABLE = True  
except ImportError:  
    SCIPY_AVAILABLE = False  

# VTK imports with error handling  
try:  
    import vtk  
    from vtk.util.numpy_support import vtk_to_numpy  
    VTK_AVAILABLE = True  
except ImportError:  
    VTK_AVAILABLE = False  
    warnings.warn("VTK not available - VTK file reading will be disabled")  

# JAXTrace configuration  
from ..utils.config import get_config  


# ============================================================================  
# Utility Functions  
# ============================================================================  

def _ensure_float32(data: np.ndarray) -> np.ndarray:  
    """Convert data to float32 for JAXTrace consistency."""  
    return np.asarray(data, dtype=np.float32)  


def _ensure_3d_positions(positions: np.ndarray) -> np.ndarray:  
    """Ensure positions have shape (N, 3) with float32 dtype."""  
    pos = _ensure_float32(positions)  
    
    if pos.ndim != 2:  
        raise ValueError(f"Positions must be 2D array (N, D), got shape {pos.shape}")  
    
    if pos.shape[1] == 2:  
        # 2D positions -> add zero z-coordinate  
        n_points = pos.shape[0]  
        pos_3d = np.zeros((n_points, 3), dtype=np.float32)  
        pos_3d[:, :2] = pos  
        return pos_3d  
    elif pos.shape[1] == 3:  
        # Already 3D  
        return pos.astype(np.float32, copy=False)  
    elif pos.shape[1] > 3:  
        # Take first 3 components  
        return pos[:, :3].astype(np.float32, copy=False)  
    else:  
        raise ValueError(f"Invalid position dimensionality: {pos.shape}")  


def _ensure_3d_velocity(velocity: np.ndarray) -> np.ndarray:  
    """Ensure velocity has shape (N, 3) with float32 dtype."""  
    vel = _ensure_float32(velocity)  
    
    if vel.ndim != 2:  
        raise ValueError(f"Velocity must be 2D array (N, D), got shape {vel.shape}")  
    
    if vel.shape[1] == 1:  
        # 1D velocity -> pad to 3D  
        n_points = vel.shape[0]  
        vel_3d = np.zeros((n_points, 3), dtype=np.float32)  
        vel_3d[:, 0] = vel[:, 0]  
        return vel_3d  
    elif vel.shape[1] == 2:  
        # 2D velocity -> add zero w-component  
        n_points = vel.shape[0]  
        vel_3d = np.zeros((n_points, 3), dtype=np.float32)  
        vel_3d[:, :2] = vel  
        return vel_3d  
    elif vel.shape[1] == 3:  
        # Already 3D  
        return vel.astype(np.float32, copy=False)  
    elif vel.shape[1] > 3:  
        # Take first 3 components  
        return vel[:, :3].astype(np.float32, copy=False)  
    else:  
        raise ValueError(f"Invalid velocity shape: {vel.shape}")  


def _natural_sort_key(filename: str) -> tuple:  
    """  
    Natural sorting key for time-series files.  
    
    Extracts numeric parts for proper ordering:  
    002_caseCoarse_0.pvtu -> (2, 0)  
    002_caseCoarse_159.pvtu -> (2, 159)  
    """  
    basename = os.path.basename(filename)  
    # Extract all numbers from filename  
    numbers = tuple(int(x) for x in re.findall(r'\d+', basename))  
    return numbers if numbers else (basename,)  


def _get_memory_usage() -> float:  
    """Get current process memory usage in GB."""  
    if PSUTIL_AVAILABLE:  
        try:  
            process = psutil.Process(os.getpid())  
            return process.memory_info().rss / (1024**3)  
        except Exception:  
            pass  
    return 0.0  


# ============================================================================  
# Structured Grid Support (Migrated from vtk_io.py)  
# ============================================================================  

@dataclass(frozen=True)  
class GridMeta:  
    """Structured grid metadata for VTK datasets."""  
    origin: np.ndarray    # (3,) grid origin coordinates  
    spacing: np.ndarray   # (3,) grid spacing dx,dy,dz  
    shape: Tuple[int, int, int]  # (Nx, Ny, Nz) grid dimensions  
    bounds: np.ndarray    # (2,3) [[xmin,ymin,zmin], [xmax,ymax,zmax]]  


def _read_vti_to_numpy(filename: str, dtype: np.dtype = np.float32) -> Tuple[np.ndarray, GridMeta]:  
    """Read VTK ImageData (.vti) file to numpy array with grid metadata."""  
    if not VTK_AVAILABLE:  
        raise RuntimeError("VTK not available; cannot read VTK files")  
    
    reader = vtk.vtkXMLImageDataReader()  
    reader.SetFileName(filename)  
    reader.Update()  
    
    image = reader.GetOutput()  
    dims = image.GetDimensions()           # (Nx, Ny, Nz)  
    spacing = np.array(image.GetSpacing(), dtype=float)  # (dx, dy, dz)  
    origin = np.array(image.GetOrigin(), dtype=float)    # (ox, oy, oz)  
    
    # Try to get point data first, then cell data  
    pd = image.GetPointData()  
    cd = image.GetCellData()  
    arr = None  
    
    if pd and pd.GetNumberOfArrays() > 0:  
        arr = pd.GetArray(0)  
        Nx, Ny, Nz = dims  # Point data uses full dimensions  
    elif cd and cd.GetNumberOfArrays() > 0:  
        arr = cd.GetArray(0)  
        Nx, Ny, Nz = dims[0]-1, dims[1]-1, dims[2]-1  # Cell data is one less  
    
    if arr is None:  
        raise ValueError(f"No data arrays found in {filename}")  
    
    np_arr = vtk_to_numpy(arr)  
    num_comp = arr.GetNumberOfComponents()  
    
    # Reshape to (Nx, Ny, Nz, C) - VTK uses Fortran ordering  
    np_arr = np_arr.reshape((Nx, Ny, Nz, num_comp), order="F").astype(dtype, copy=False)  
    
    # Calculate bounds  
    bounds = np.stack([origin, origin + spacing * np.array([Nx-1, Ny-1, Nz-1])], axis=0)  
    
    meta = GridMeta(  
        origin=origin,   
        spacing=spacing,   
        shape=(Nx, Ny, Nz),   
        bounds=bounds.astype(float)  
    )  
    
    return np_arr, meta  


def _read_vts_vtr_to_numpy(filename: str, dtype: np.dtype = np.float32) -> Tuple[np.ndarray, GridMeta]:  
    """Read VTK StructuredGrid (.vts) or RectilinearGrid (.vtr) to numpy array."""  
    if not VTK_AVAILABLE:  
        raise RuntimeError("VTK not available; cannot read VTK files")  
    
    ext = os.path.splitext(filename)[1].lower()  
    
    if ext == ".vts":  
        reader = vtk.vtkXMLStructuredGridReader()  
    elif ext == ".vtr":  
        reader = vtk.vtkXMLRectilinearGridReader()  
    else:  
        raise ValueError(f"Unsupported extension for structured/rectilinear: {ext}")  
    
    reader.SetFileName(filename)  
    reader.Update()  
    ds = reader.GetOutput()  
    
    # Get data array (prefer point data over cell data)  
    pd = ds.GetPointData()  
    cd = ds.GetCellData()  
    arr = None  
    
    if pd and pd.GetNumberOfArrays() > 0:  
        arr = pd.GetArray(0)  
    elif cd and cd.GetNumberOfArrays() > 0:  
        arr = cd.GetArray(0)  
    
    if arr is None:  
        raise ValueError(f"No data arrays found in {filename}")  
    
    np_arr = vtk_to_numpy(arr)  
    num_comp = arr.GetNumberOfComponents()  
    
    # Get grid dimensions and spacing  
    if ext == ".vts":  # StructuredGrid  
        dims = ds.GetDimensions()  
        bounds = np.array(ds.GetBounds()).reshape(3, 2).T  # (2,3)  
        origin = bounds[0]  
        size = bounds[1] - bounds[0]  
        Nx, Ny, Nz = dims  
        spacing = size / np.maximum([Nx-1, Ny-1, Nz-1], 1)  
    else:  # RectilinearGrid  
        Nx = ds.GetXCoordinates().GetNumberOfTuples()  
        Ny = ds.GetYCoordinates().GetNumberOfTuples()  
        Nz = ds.GetZCoordinates().GetNumberOfTuples()  
        
        x0 = ds.GetXCoordinates().GetTuple1(0)  
        x1 = ds.GetXCoordinates().GetTuple1(Nx-1) if Nx > 1 else x0  
        y0 = ds.GetYCoordinates().GetTuple1(0)  
        y1 = ds.GetYCoordinates().GetTuple1(Ny-1) if Ny > 1 else y0  
        z0 = ds.GetZCoordinates().GetTuple1(0)  
        z1 = ds.GetZCoordinates().GetTuple1(Nz-1) if Nz > 1 else z0  
        
        origin = np.array([x0, y0, z0], dtype=float)  
        spacing = np.array([  
            (x1-x0)/max(Nx-1, 1),   
            (y1-y0)/max(Ny-1, 1),   
            (z1-z0)/max(Nz-1, 1)  
        ], dtype=float)  
    
    # Reshape array  
    np_arr = np_arr.reshape((Nx, Ny, Nz, num_comp), order="F").astype(dtype, copy=False)  
    
    # Calculate bounds  
    bounds = np.stack([origin, origin + spacing * np.array([Nx-1, Ny-1, Nz-1])], axis=0)  
    
    meta = GridMeta(  
        origin=origin,   
        spacing=spacing,   
        shape=(Nx, Ny, Nz),   
        bounds=bounds.astype(float)  
    )  
    
    return np_arr, meta  


class VTKStructuredSeries:  
    """  
    Time series reader for VTK structured grid files.  
    
    Supports single files, file lists, and glob patterns for  
    .vti (ImageData), .vts (StructuredGrid), and .vtr (RectilinearGrid) files.  
    """  
    
    def __init__(self, spec: Union[str, Iterable[str]]):  
        """  
        Initialize VTK structured series from file specification.  
        
        Parameters  
        ----------  
        spec : str or Iterable[str]  
            File specification:  
            - Single file: "data.vti"  
            - Glob pattern: "data_*.vti"  
            - File list: ["data_001.vti", "data_002.vti"]  
        """  
        if not VTK_AVAILABLE:  
            raise RuntimeError("VTK not available; cannot read VTK files")  
            
        if isinstance(spec, (list, tuple)):  
            self._files = list(spec)  
        elif isinstance(spec, str):  
            if any(ch in spec for ch in "*?[]"):  # Glob pattern  
                self._files = sorted(glob.glob(spec))  
            else:  # Single file  
                self._files = [spec]  
        else:  
            raise TypeError("spec must be a filename, glob pattern, or list of filenames")  
        
        if not self._files:  
            raise FileNotFoundError(f"No VTK files found matching: {spec}")  
        
        # Validate first file and determine extension  
        if not os.path.exists(self._files[0]):  
            raise FileNotFoundError(f"First file not found: {self._files[0]}")  
            
        self._ext = os.path.splitext(self._files[0])[1].lower()  
        if self._ext not in (".vti", ".vts", ".vtr"):  
            raise ValueError(f"Unsupported VTK extension: {self._ext}")  
        
        # Cache for grid metadata (loaded on first access)  
        self._meta_cache: Optional[GridMeta] = None  

    def __len__(self) -> int:  
        """Number of time steps in the series."""  
        return len(self._files)  

    @property  
    def filenames(self) -> List[str]:  
        """List of filenames in the series."""  
        return self._files.copy()  

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
        if not (0 <= i < len(self._files)):  
            raise IndexError(f"Time index {i} out of range [0, {len(self._files)})")  
            
        filename = self._files[i]  
        
        if self._ext == ".vti":  
            arr, meta = _read_vti_to_numpy(filename)  
        else:  # .vts or .vtr  
            arr, meta = _read_vts_vtr_to_numpy(filename)  
        
        # Cache metadata from first successful load  
        if self._meta_cache is None:  
            self._meta_cache = meta  
            
        return arr  

    def load_timestep(self, index: int) -> np.ndarray:  
        """Alias for load_slice() to match TimeSeriesReader protocol."""  
        return self.load_slice(index)  

    def get_times(self) -> List[float]:  
        """Get time values (placeholder - VTK files don't typically store time info)."""  
        return list(range(len(self._files)))  

    def close(self) -> None:  
        """Close the dataset (no-op for VTK files)."""  
        pass  

    def grid_meta(self) -> GridMeta:
        """
        Get grid metadata (loads first file if not cached).
        
        Returns
        -------
        GridMeta
            Grid metadata including origin, spacing, shape, and bounds
        """
        if self._meta_cache is None:
            # Load first file to get metadata
            self.load_slice(0)
        return self._meta_cache

    def load_full_series(self, dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Load entire time series as single array.
        
        Parameters
        ----------
        dtype : np.dtype
            Data type for output array
            
        Returns
        -------
        np.ndarray
            Time series array, shape (T, Nx, Ny, Nz, C)
        """
        if len(self._files) == 0:
            raise ValueError("No files in series")
        
        # Load first slice to get dimensions
        first_slice = self.load_slice(0).astype(dtype, copy=False)
        T = len(self._files)
        
        # Pre-allocate full array
        full_shape = (T,) + first_slice.shape
        time_series = np.zeros(full_shape, dtype=dtype)
        time_series[0] = first_slice
        
        # Load remaining slices
        for i in range(1, T):
            time_series[i] = self.load_slice(i).astype(dtype, copy=False)
        
        return time_series

    def __getitem__(self, i: int) -> np.ndarray:
        """Load time slice using indexing."""
        return self.load_slice(i)

    def __repr__(self) -> str:
        """String representation."""
        return f"VTKStructuredSeries({len(self._files)} files, ext={self._ext})"


# ============================================================================
# Unstructured Grid Support (Enhanced from previous implementation)
# ============================================================================

class VTKUnstructuredTimeSeriesReader:
    """
    Memory-optimized reader for VTK unstructured grid time series (.pvtu/.vtu).
    
    Designed for time sequence data with consistent grid structure.
    Features:
    - Handles .pvtu (parallel) files preferentially over .vtu (serial)
    - Memory-efficient caching with configurable limits
    - Auto-detection of velocity field names
    - Consistent JAXTrace data formats: (T,N,3) velocities, (N,3) positions
    """
    
    def __init__(
        self,
        file_pattern: str,
        max_time_steps: Optional[int] = None,
        velocity_field_name: Optional[str] = None,
        cache_size_limit: Optional[int] = None
    ):
        """
        Initialize VTK unstructured time series reader.
        
        Parameters
        ----------
        file_pattern : str
            Pattern to match VTK files (e.g., "/path/to/data/*.pvtu")
        max_time_steps : int, optional
            Maximum number of time steps to load (None = all)
        velocity_field_name : str, optional
            Name of velocity field in VTK files (auto-detected if None)
        cache_size_limit : int, optional
            Maximum time steps to cache (uses config if None)
        """
        if not VTK_AVAILABLE:
            raise ImportError("VTK not available - cannot read VTK files")
        
        self.file_pattern = file_pattern
        self.max_time_steps = max_time_steps
        self.velocity_field_name = velocity_field_name
        
        # Configure cache size from global config
        config = get_config()
        if cache_size_limit is None:
            # Estimate based on memory limit (conservative)
            estimated_cache = max(int(config.memory_limit_gb * 0.2), 3)
            self.cache_size_limit = min(estimated_cache, 15)
        else:
            self.cache_size_limit = cache_size_limit
        
        # Find and validate time series files
        self.time_files = self._find_time_files()
        if not self.time_files:
            raise ValueError(f"No VTK files found matching pattern: {file_pattern}")
        
        # Grid and cache initialization
        self.grid_points = None          # (N, 3) - spatial positions, float32
        self.grid_tree = None            # KDTree for spatial queries
        self.velocity_cache = {}         # time_idx -> (N, 3) velocity data
        
        # Metadata
        self._grid_bounds = None         # (xmin, ymin, zmin, xmax, ymax, zmax)
        self._n_grid_points = None       # Total grid points
        self._detected_field_name = None # Auto-detected velocity field name
        
        print(f"VTKUnstructuredTimeSeriesReader initialized:")
        print(f"  📁 Pattern: {file_pattern}")
        print(f"  📊 Time steps: {len(self.time_files)}")
        print(f"  🗄️ Cache limit: {self.cache_size_limit}")
        print(f"  💾 Memory limit: {config.memory_limit_gb} GB")
    
    def _find_time_files(self) -> List[str]:
        """Find and sort VTK time series files."""
        # Handle different path formats
        if os.path.isabs(self.file_pattern):
            files = glob.glob(self.file_pattern)
        else:
            files = glob.glob(self.file_pattern)
            if not files:
                files = glob.glob(os.path.abspath(self.file_pattern))
        
        if not files:
            # Try expanding user path
            expanded = os.path.expanduser(self.file_pattern)
            files = glob.glob(expanded)
        
        # Separate .pvtu and .vtu files
        pvtu_files = [f for f in files if f.lower().endswith('.pvtu')]
        vtu_files = [f for f in files if f.lower().endswith('.vtu')]
        
        # Filter out .vtu files that have corresponding .pvtu files
        if pvtu_files:
            # Remove .vtu files that have corresponding .pvtu
            pvtu_bases = set()
            for pvtu_file in pvtu_files:
                # Extract base name pattern (e.g., "002_caseCoarse_0" from "002_caseCoarse_0.pvtu")
                base = os.path.splitext(pvtu_file)[0]
                pvtu_bases.add(base)
            
            # Keep only .vtu files without corresponding .pvtu
            vtu_files = [f for f in vtu_files if os.path.splitext(f)[0] not in pvtu_bases]
            
            selected_files = pvtu_files
            print(f"  🔍 Using parallel VTU files (.pvtu): {len(selected_files)} found")
        else:
            selected_files = vtu_files
            print(f"  🔍 Using serial VTU files (.vtu): {len(selected_files)} found")
        
        # Natural sort for proper time ordering
        selected_files.sort(key=_natural_sort_key)
        
        # Apply time step limit
        if self.max_time_steps is not None and len(selected_files) > self.max_time_steps:
            selected_files = selected_files[-self.max_time_steps:]
            print(f"  ⏳ Limited to last {self.max_time_steps} time steps")
        VTKUnstructuredTimeSeriesReader
        return selected_files
    
    def _cleanup_cache(self):
        """Clean up velocity cache to maintain memory limits."""
        if len(self.velocity_cache) > self.cache_size_limit:
            # FIFO removal
            items_to_remove = len(self.velocity_cache) - self.cache_size_limit
            keys_to_remove = list(self.velocity_cache.keys())[:items_to_remove]
            
            for key in keys_to_remove:
                del self.velocity_cache[key]
            
            gc.collect()
            # Only print cleanup message occasionally to avoid spam
            if len(keys_to_remove) > 1:
                print(f"  🧹 Cleaned cache: removed {items_to_remove} time steps")
    
    def _detect_velocity_field(self, vtk_output) -> str:
        """Auto-detect velocity field name in VTK file."""
        if self.velocity_field_name is not None:
            return self.velocity_field_name
        
        if self._detected_field_name is not None:
            return self._detected_field_name
        
        # Extended list of common velocity field names
        velocity_names = [
            'velocity', 'Velocity', 'VELOCITY',
            'U', 'u', 'vel', 'Vel', 'VEL',
            'Displacement', 'displacement', 'DISPLACEMENT',
            'VelocityField', 'velocity_field',
            'FlowVelocity', 'flow_velocity',
            'v', 'V', 'fluid_velocity',
            'Result', 'RESULT',  # Generic solver output names
            'NodeVel', 'NODEVEL'  # Some FEM solvers use this
        ]
        
        point_data = vtk_output.GetPointData()
        
        # Check each potential name
        for name in velocity_names:
            velocity_array = point_data.GetArray(name)
            if velocity_array is not None:
                n_components = velocity_array.GetNumberOfComponents()
                if n_components >= 1:  # At least 1D velocity
                    print(f"  🎯 Auto-detected velocity field: '{name}' ({n_components}D)")
                    self._detected_field_name = name
                    self.velocity_field_name = name
                    return name
        
        # If not found, list available fields for debugging
        available_fields = []
        for i in range(point_data.GetNumberOfArrays()):
            array_name = point_data.GetArrayName(i)
            if array_name:
                array = point_data.GetArray(i)
                n_comp = array.GetNumberOfComponents() if array else 0
                available_fields.append(f"{array_name} ({n_comp}D)")
        
        raise ValueError(
            f"Velocity field not found. Available point data fields: {available_fields}\n"
            f"Specify velocity_field_name explicitly if using non-standard field names."
        )
    
    def load_single_timestep(self, time_idx: int) -> np.ndarray:
        """
        Load velocity data for a single time step.
        
        Parameters
        ----------
        time_idx : int
            Time step index
            
        Returns
        -------
        np.ndarray
            Velocity field array, shape (N, 3), dtype float32
        """
        if time_idx < 0 or time_idx >= len(self.time_files):
            raise IndexError(f"Time index {time_idx} out of range [0, {len(self.time_files)-1}]")
        
        # Check cache first
        if time_idx in self.velocity_cache:
            return self.velocity_cache[time_idx]
        
        file_path = self.time_files[time_idx]
        
        # Select appropriate VTK reader
        if file_path.lower().endswith('.pvtu'):
            reader = vtk.vtkXMLPUnstructuredGridReader()
        elif file_path.lower().endswith('.vtu'):
            reader = vtk.vtkXMLUnstructuredGridReader()
        elif file_path.lower().endswith('.vtk'):
            reader = vtk.vtkUnstructuredGridReader()
        else:
            raise ValueError(f"Unsupported VTK file format: {file_path}")
        
        reader.SetFileName(file_path)
        reader.Update()
        output = reader.GetOutput()
        
        # Load grid points only once (assume consistent grid across time)
        if self.grid_points is None:
            points_data = output.GetPoints().GetData()
            self.grid_points = _ensure_3d_positions(vtk_to_numpy(points_data))
            
            self._n_grid_points = self.grid_points.shape[0]
            self._grid_bounds = self._calculate_bounds()
            
            # Build spatial index if SciPy available
            if SCIPY_AVAILABLE:
                self.grid_tree = cKDTree(self.grid_points)
            
            print(f"  📍 Loaded grid: {self._n_grid_points} points")
            print(f"  📏 Bounds: x[{self._grid_bounds[0]:.2f}, {self._grid_bounds[3]:.2f}], "
                  f"y[{self._grid_bounds[1]:.2f}, {self._grid_bounds[4]:.2f}], "
                  f"z[{self._grid_bounds[2]:.2f}, {self._grid_bounds[5]:.2f}]")
        
        # Load velocity field
        velocity_field_name = self._detect_velocity_field(output)
        velocity_array = output.GetPointData().GetArray(velocity_field_name)
        
        if velocity_array is None:
            raise ValueError(f"Velocity field '{velocity_field_name}' not found in {file_path}")
        
        # Convert to numpy and ensure 3D format
        velocity = vtk_to_numpy(velocity_array)  # (N, components)
        velocity = _ensure_3d_velocity(velocity)  # (N, 3), float32
        
        # Validate consistency
        if velocity.shape[0] != self._n_grid_points:
            raise ValueError(
                f"Velocity field size mismatch in {file_path}: "
                f"expected {self._n_grid_points}, got {velocity.shape[0]}"
            )
        
        # Cache the velocity field with memory management
        self.velocity_cache[time_idx] = velocity
        self._cleanup_cache()
        
        return velocity
    
    def _calculate_bounds(self) -> Tuple[float, ...]:
        """Calculate spatial bounds of the grid."""
        if self.grid_points is None:
            return None
        
        min_coords = np.min(self.grid_points, axis=0)  # (3,)
        max_coords = np.max(self.grid_points, axis=0)  # (3,)
        
        # Return as (xmin, ymin, zmin, xmax, ymax, zmax)
        return (
            float(min_coords[0]), float(min_coords[1]), float(min_coords[2]),
            float(max_coords[0]), float(max_coords[1]), float(max_coords[2])
        )
    
    def load_time_series(self, start_idx: int = 0, end_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Load velocity time series data in JAXTrace format.
        
        Parameters
        ----------
        start_idx : int
            Starting time step index
        end_idx : int, optional
            Ending time step index (None for all remaining)
            
        Returns
        -------
        dict
            Dictionary with JAXTrace-compatible time series data:
            - 'velocity_data': (T, N, 3) velocity field time series
            - 'positions': (N, 3) grid positions
            - 'times': (T,) time values
            - 'source': 'vtk_unstructured_series'
            - 'file_pattern': original file pattern
            - 'velocity_field_name': detected field name
        """
        if end_idx is None:
            end_idx = len(self.time_files)
        
        end_idx = min(end_idx, len(self.time_files))
        n_timesteps = end_idx - start_idx
        
        if n_timesteps <= 0:
            raise ValueError(f"Invalid time range: [{start_idx}, {end_idx})")
        
        print(f"📥 Loading time series: steps {start_idx} to {end_idx-1}")
        
        # Load first timestep to get grid information
        if self.grid_points is None:
            self.load_single_timestep(start_idx)
        
        n_points = self._n_grid_points
        
        # Pre-allocate arrays in JAXTrace format
        velocity_data = np.zeros((n_timesteps, n_points, 3), dtype=np.float32)  # (T, N, 3)
        
        # Load each timestep with progress reporting
        for i, time_idx in enumerate(range(start_idx, end_idx)):
            velocity = self.load_single_timestep(time_idx)  # (N, 3)
            velocity_data[i] = velocity
            
            # Progress reporting
            if n_timesteps > 20 and (i + 1) % max(1, n_timesteps // 10) == 0:
                print(f"  📊 Loaded {i+1}/{n_timesteps} time steps ({100*(i+1)/n_timesteps:.1f}%)")
            elif n_timesteps <= 20 and (i + 1) % 5 == 0:
                print(f"  📊 Loaded {i+1}/{n_timesteps} time steps")
        
        # Generate time values (use step indices as time values)
        times = np.arange(start_idx, end_idx, dtype=np.float32)
        
        print(f"✅ Time series loaded:")
        print(f"  📊 Shape: {velocity_data.shape} (T,N,3)")
        print(f"  🕒 Time range: {times[0]:.0f} to {times[-1]:.0f}")
        print(f"  💾 Memory usage: {_get_memory_usage():.2f} GB")
        
        return {
            'velocity_data': velocity_data,      # (T, N, 3)
            'positions': self.grid_points,       # (N, 3)  
            'times': times,                      # (T,)
            'source': 'vtk_unstructured_series',
            'file_pattern': self.file_pattern,
            'velocity_field_name': self.velocity_field_name or self._detected_field_name
        }
 
    def get_bounds(self) -> Tuple[float, ...]:
        """Get spatial bounds: (xmin, ymin, zmin, xmax, ymax, zmax)."""
        if self._grid_bounds is None:
            self.load_single_timestep(0)
        return self._grid_bounds
    
    def get_grid_info(self) -> Dict[str, Any]:
        """Get comprehensive grid information."""
        if self.grid_points is None:
            self.load_single_timestep(0)
        
        return {
            'grid_points': self.grid_points,
            'n_points': self._n_grid_points,
            'n_timesteps': len(self.time_files),
            'bounds': self._grid_bounds,
            'time_files': self.time_files,
            'velocity_field_name': self.velocity_field_name or self._detected_field_name,
            'cache_size': len(self.velocity_cache),
            'memory_usage_gb': _get_memory_usage(),
            'has_spatial_index': self.grid_tree is not None
        }
    
    def validate_files(self) -> Dict[str, Any]:
        """Validate VTK files for consistency."""
        print("🔍 Validating VTK time series files...")
        
        validation_report = {
            'total_files': len(self.time_files),
            'valid_files': 0,
            'invalid_files': [],
            'grid_consistency': True,
            'velocity_field_consistency': True,
            'errors': []
        }
        
        first_n_points = None
        first_velocity_field = None
        
        # Validate first 5 files to avoid excessive loading
        files_to_check = min(5, len(self.time_files))
        
        for i, file_path in enumerate(self.time_files[:files_to_check]):
            try:
                # Select appropriate reader
                if file_path.lower().endswith('.pvtu'):
                    reader = vtk.vtkXMLPUnstructuredGridReader()
                elif file_path.lower().endswith('.vtu'):
                    reader = vtk.vtkXMLUnstructuredGridReader()
                elif file_path.lower().endswith('.vtk'):
                    reader = vtk.vtkUnstructuredGridReader()
                else:
                    continue
                
                reader.SetFileName(file_path)
                reader.Update()
                output = reader.GetOutput()
                
                # Check grid point count consistency
                n_points = output.GetNumberOfPoints()
                if first_n_points is None:
                    first_n_points = n_points
                elif n_points != first_n_points:
                    validation_report['grid_consistency'] = False
                    validation_report['errors'].append(
                        f"Grid size mismatch in step {i}: expected {first_n_points}, got {n_points}"
                    )
                
                # Check velocity field consistency
                try:
                    velocity_name = self._detect_velocity_field(output)
                    if first_velocity_field is None:
                        first_velocity_field = velocity_name
                    elif velocity_name != first_velocity_field:
                        validation_report['velocity_field_consistency'] = False
                        validation_report['errors'].append(
                            f"Velocity field name mismatch in step {i}: "
                            f"expected '{first_velocity_field}', got '{velocity_name}'"
                        )
                except Exception as e:
                    validation_report['errors'].append(
                        f"Velocity field detection failed in step {i}: {str(e)}"
                    )
                
                validation_report['valid_files'] += 1
                
            except Exception as e:
                validation_report['invalid_files'].append((file_path, str(e)))
                validation_report['errors'].append(
                    f"Error reading step {i} ({os.path.basename(file_path)}): {str(e)}"
                )
        
        # Summary
        is_valid = (validation_report['valid_files'] > 0 and
                   validation_report['grid_consistency'] and
                   validation_report['velocity_field_consistency'])
        
        print(f"{'✅' if is_valid else '❌'} Validation complete: "
              f"{validation_report['valid_files']}/{files_to_check} files checked")
        
        if validation_report['errors']:
            print("⚠️  Validation issues found:")
            for error in validation_report['errors'][:3]:  # Show first 3 errors
                print(f"    {error}")
            if len(validation_report['errors']) > 3:
                print(f"    ... and {len(validation_report['errors']) - 3} more errors")
        
        return validation_report
    
    # Utility and convenience methods
    def get_time_range(self) -> Tuple[int, int]:
        """Get available time step range."""
        return (0, len(self.time_files) - 1)
    
    def preload_timesteps(self, start_idx: int = 0, end_idx: Optional[int] = None):
        """Preload multiple time steps into cache."""
        if end_idx is None:
            end_idx = len(self.time_files)
        
        end_idx = min(end_idx, len(self.time_files))
        
        print(f"📥 Preloading time steps {start_idx} to {end_idx-1}...")
        
        for i in range(start_idx, end_idx):
            if i not in self.velocity_cache:
                self.load_single_timestep(i)
                if end_idx - start_idx > 10 and i % 10 == 0:
                    print(f"  📊 Loaded timestep {i}/{end_idx-1}")
        
        print(f"✅ Preloading complete. Cache size: {len(self.velocity_cache)}")
    
    def clear_cache(self):
        """Clear velocity field cache to free memory."""
        self.velocity_cache.clear()
        gc.collect()
        print("🧹 Velocity cache cleared")
    
    def get_velocity_statistics(self, time_idx: int) -> Dict[str, Any]:
        """Get velocity field statistics for a given time step."""
        velocity = self.load_single_timestep(time_idx)  # (N, 3)
        velocity_magnitude = np.linalg.norm(velocity, axis=1)  # (N,)
        
        return {
            'time_idx': time_idx,
            'file_path': self.time_files[time_idx],
            'mean_velocity': np.mean(velocity, axis=0),      # (3,)
            'std_velocity': np.std(velocity, axis=0),        # (3,)
            'mean_magnitude': float(np.mean(velocity_magnitude)),
            'max_magnitude': float(np.max(velocity_magnitude)),
            'min_magnitude': float(np.min(velocity_magnitude)),
            'velocity_range': {
                'x': (float(np.min(velocity[:, 0])), float(np.max(velocity[:, 0]))),
                'y': (float(np.min(velocity[:, 1])), float(np.max(velocity[:, 1]))),
                'z': (float(np.min(velocity[:, 2])), float(np.max(velocity[:, 2])))
            }
        }
    
    # Magic methods for convenience
    def __len__(self) -> int:
        """Number of time steps."""
        return len(self.time_files)
    
    def __getitem__(self, time_idx: int) -> np.ndarray:
        """Get velocity field for time step using array indexing."""
        return self.load_single_timestep(time_idx)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"VTKUnstructuredTimeSeriesReader(files={len(self.time_files)}, "
                f"field='{self.velocity_field_name or self._detected_field_name}', "
                f"cache={len(self.velocity_cache)}/{self.cache_size_limit})")


# ============================================================================
# Factory Functions and Convenience API
# ============================================================================

def open_vtk_time_series(
    file_pattern: str,
    max_time_steps: Optional[int] = None,
    velocity_field_name: Optional[str] = None,
    validate_files: bool = True,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Open VTK time series files and return JAXTrace-compatible data.
    
    Parameters
    ----------
    file_pattern : str
        Glob pattern for VTK files (e.g., "/path/to/data/*.pvtu")
    max_time_steps : int, optional
        Maximum time steps to load
    velocity_field_name : str, optional  
        Name of velocity field (auto-detected if None)
    validate_files : bool
        Whether to validate files before loading
    **kwargs
        Additional arguments for VTKUnstructuredTimeSeriesReader
        
    Returns
    -------
    dict
        JAXTrace-compatible time series data with consistent shapes
    """
    reader = VTKUnstructuredTimeSeriesReader(
        file_pattern=file_pattern,
        max_time_steps=max_time_steps,
        velocity_field_name=velocity_field_name,
        **kwargs
    )
    
    # Validate files first if requested
    if validate_files:
        validation = reader.validate_files()
        if not validation['grid_consistency']:
            warnings.warn("Grid inconsistency detected - results may be unreliable")
        if validation['valid_files'] == 0:
            raise RuntimeError("No valid VTK files found - cannot proceed")
    
    # Load all data
    return reader.load_time_series()


def open_vtk_structured_series(
    file_spec: Union[str, Iterable[str]],
    **kwargs
) -> VTKStructuredSeries:
    """
    Open VTK structured grid series.
    
    Parameters
    ----------
    file_spec : str or Iterable[str]
        File specification (single file, glob pattern, or file list)
    **kwargs
        Additional arguments (currently unused)
        
    Returns
    -------
    VTKStructuredSeries
        Structured grid series reader
    """
    return VTKStructuredSeries(file_spec)


# Auto-detection convenience function
def open_vtk_dataset(file_pattern: str, **kwargs):
    """
    Open VTK dataset with auto-detection of structured vs unstructured format.
    
    Parameters
    ----------
    file_pattern : str
        File pattern or path to VTK files
    **kwargs
        Additional arguments passed to appropriate reader
        
    Returns
    -------
    dict or VTKStructuredSeries
        Loaded VTK data in appropriate format
    """
    import glob
    from pathlib import Path
    
    # Determine file type from pattern
    if isinstance(file_pattern, str):
        if any(pattern in file_pattern for pattern in ['*', '?', '[']):
            # Glob pattern
            files = glob.glob(file_pattern)
        else:
            # Single file or directory
            p = Path(file_pattern)
            if p.is_dir():
                files = list(p.glob('*.vt*'))
            else:
                files = [str(p)]
    else:
        files = list(file_pattern)
    
    if not files:
        raise FileNotFoundError(f"No VTK files found: {file_pattern}")
    
    # Check file extensions to determine reader type
    sample_file = files[0].lower()
    
    if sample_file.endswith(('.pvtu', '.vtu', '.vtk')):
        # Unstructured grid time series
        return open_vtk_time_series(file_pattern, **kwargs)
    
    elif sample_file.endswith(('.vti', '.vts', '.vtr')):
        # Structured grid series
        return open_vtk_structured_series(file_pattern, **kwargs)
    
    else:
        raise ValueError(f"Unsupported VTK file type: {sample_file}")