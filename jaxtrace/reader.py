"""
VTK Reader Module for JAXTrace

Memory-optimized VTK file reader for particle tracking applications.
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import glob
from scipy.spatial import cKDTree
from typing import List, Dict, Optional, Tuple
import os
import gc
import psutil
import weakref


class VTKReader:
    """Memory-optimized VTK reader that loads only necessary data."""
    
    def __init__(self, 
                 file_pattern: str, 
                 max_time_steps: Optional[int] = None,
                 velocity_field_name: Optional[str] = None,
                 cache_size_limit: int = 5):
        """
        Initialize memory-optimized VTK reader.
        
        Args:
            file_pattern: Pattern to match VTK files (e.g., "*.pvtu", "*_case_*.vtu")
            max_time_steps: Maximum number of time steps to keep in memory (None = all)
            velocity_field_name: Name of velocity field in VTK files (auto-detected if None)
            cache_size_limit: Maximum number of time steps to cache in memory
        """
        self.file_pattern = file_pattern
        self.max_time_steps = max_time_steps
        self.velocity_field_name = velocity_field_name
        self.cache_size_limit = cache_size_limit
        
        # Find and validate files
        self.time_files = self._find_time_files()
        if not self.time_files:
            raise ValueError(f"No VTK files found matching pattern: {file_pattern}")
        
        # Memory management
        self.grid_points = None
        self.grid_tree = None
        self.velocity_cache = {}
        
        # Grid metadata
        self._grid_bounds = None
        self._n_grid_points = None
        
        # Use weak references for cleanup
        self._cleanup_refs = []
        
        print(f"VTKReader initialized with {len(self.time_files)} time steps")
        
    def _find_time_files(self) -> List[str]:
        """Find VTK files, optionally limiting to last n time steps."""
        files = glob.glob(self.file_pattern)
        if not files:
            # Try absolute path
            files = glob.glob(os.path.abspath(self.file_pattern))
            
        files.sort()
        
        if self.max_time_steps is not None:
            files = files[-self.max_time_steps:]
            print(f"Limited to last {self.max_time_steps} time steps")
        
        return files
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**3
    
    def _cleanup_cache(self):
        """Clean up velocity cache to free memory."""
        if len(self.velocity_cache) > self.cache_size_limit:
            # Remove oldest entries
            keys_to_remove = list(self.velocity_cache.keys())[:-self.cache_size_limit]
            for key in keys_to_remove:
                del self.velocity_cache[key]
            gc.collect()
            # print(f"Cleaned cache, removed {len(keys_to_remove)} time steps")
    
    def _detect_velocity_field(self, vtk_output) -> str:
        """Auto-detect velocity field name in VTK file."""
        if self.velocity_field_name is not None:
            return self.velocity_field_name
            
        # Common velocity field names
        velocity_names = ['velocity', 'Velocity', 'U', 'vel', 'Displacement', 
                         'VelocityField', 'FlowVelocity', 'v']
        
        for name in velocity_names:
            velocity_array = vtk_output.GetPointData().GetArray(name)
            if velocity_array is not None:
                print(f"Auto-detected velocity field: '{name}'")
                self.velocity_field_name = name
                return name
        
        # If not found, list available fields
        point_data = vtk_output.GetPointData()
        available_fields = []
        for i in range(point_data.GetNumberOfArrays()):
            array_name = point_data.GetArrayName(i)
            if array_name:
                available_fields.append(array_name)
        
        raise ValueError(f"Velocity field not found. Available fields: {available_fields}")
    
    def load_single_timestep(self, time_idx: int) -> np.ndarray:
        """
        Load velocity data for a single time step.
        
        Args:
            time_idx: Time step index
            
        Returns:
            Velocity field array (n_points, 3)
        """
        if time_idx < 0 or time_idx >= len(self.time_files):
            raise ValueError(f"Time index {time_idx} out of range [0, {len(self.time_files)-1}]")
            
        # Check cache first
        if time_idx in self.velocity_cache:
            return self.velocity_cache[time_idx]
        
        file_path = self.time_files[time_idx]
        
        # Read VTK file
        if file_path.endswith('.pvtu'):
            reader = vtk.vtkXMLPUnstructuredGridReader()
        elif file_path.endswith('.vtu'):
            reader = vtk.vtkXMLUnstructuredGridReader()
        elif file_path.endswith('.vtk'):
            reader = vtk.vtkUnstructuredGridReader()
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        reader.SetFileName(file_path)
        reader.Update()
        
        output = reader.GetOutput()
        
        # Load grid points only once
        if self.grid_points is None:
            self.grid_points = vtk_to_numpy(output.GetPoints().GetData())
            self.grid_tree = cKDTree(self.grid_points)
            self._n_grid_points = len(self.grid_points)
            self._grid_bounds = self._calculate_bounds()
            print(f"Loaded grid with {self._n_grid_points} points")
        
        # Detect and load velocity field
        velocity_field_name = self._detect_velocity_field(output)
        velocity_array = output.GetPointData().GetArray(velocity_field_name)
        
        if velocity_array is None:
            raise ValueError(f"Velocity field '{velocity_field_name}' not found")
        
        velocity = vtk_to_numpy(velocity_array)
        
        # Ensure 3D velocity (pad with zeros if needed)
        if velocity.shape[1] < 3:
            velocity_3d = np.zeros((velocity.shape[0], 3))
            velocity_3d[:, :velocity.shape[1]] = velocity
            velocity = velocity_3d
        
        # Cache with memory management
        self.velocity_cache[time_idx] = velocity
        self._cleanup_cache()
        
        return velocity
    
    def _calculate_bounds(self) -> Tuple[float, ...]:
        """Calculate spatial bounds of the grid."""
        if self.grid_points is None:
            return None
        
        min_coords = np.min(self.grid_points, axis=0)
        max_coords = np.max(self.grid_points, axis=0)
        
        return (min_coords[0], max_coords[0], 
                min_coords[1], max_coords[1], 
                min_coords[2], max_coords[2])
    
    def get_grid_info(self) -> Dict:
        """
        Get comprehensive grid information.
        
        Returns:
            Dictionary containing grid metadata
        """
        # Load first file to get grid structure if not already loaded
        if self.grid_points is None:
            self.load_single_timestep(0)
        
        return {
            'grid_points': self.grid_points,
            'n_points': self._n_grid_points,
            'n_timesteps': len(self.time_files),
            'bounds': self._grid_bounds,
            'time_files': self.time_files,
            'velocity_field_name': self.velocity_field_name,
            'cache_size': len(self.velocity_cache),
            'memory_usage_gb': self._get_memory_usage()
        }
    
    def get_bounds(self) -> Tuple[float, ...]:
        """
        Get spatial bounds of the grid.
        
        Returns:
            Tuple of (xmin, xmax, ymin, ymax, zmin, zmax)
        """
        if self._grid_bounds is None:
            # Load first timestep to get bounds
            self.load_single_timestep(0)
        return self._grid_bounds
    
    def get_time_range(self) -> Tuple[int, int]:
        """
        Get available time step range.
        
        Returns:
            Tuple of (first_timestep, last_timestep)
        """
        return (0, len(self.time_files) - 1)
    
    def preload_timesteps(self, start_idx: int = 0, end_idx: Optional[int] = None):
        """
        Preload multiple time steps into cache.
        
        Args:
            start_idx: Starting time step index
            end_idx: Ending time step index (None for all remaining)
        """
        if end_idx is None:
            end_idx = len(self.time_files)
        
        end_idx = min(end_idx, len(self.time_files))
        
        print(f"Preloading time steps {start_idx} to {end_idx-1}...")
        
        for i in range(start_idx, end_idx):
            if i not in self.velocity_cache:
                self.load_single_timestep(i)
                if i % 10 == 0:
                    print(f"  Loaded timestep {i}/{end_idx-1}")
        
        print(f"Preloading complete. Cache size: {len(self.velocity_cache)}")
    
    def clear_cache(self):
        """Clear velocity field cache to free memory."""
        self.velocity_cache.clear()
        gc.collect()
        print("Velocity cache cleared")
    
    def get_velocity_statistics(self, time_idx: int) -> Dict:
        """
        Get velocity field statistics for a given time step.
        
        Args:
            time_idx: Time step index
            
        Returns:
            Dictionary with velocity statistics
        """
        velocity = self.load_single_timestep(time_idx)
        velocity_magnitude = np.linalg.norm(velocity, axis=1)
        
        return {
            'time_idx': time_idx,
            'mean_velocity': np.mean(velocity, axis=0),
            'std_velocity': np.std(velocity, axis=0),
            'mean_magnitude': np.mean(velocity_magnitude),
            'max_magnitude': np.max(velocity_magnitude),
            'min_magnitude': np.min(velocity_magnitude),
            'velocity_range': {
                'x': (np.min(velocity[:, 0]), np.max(velocity[:, 0])),
                'y': (np.min(velocity[:, 1]), np.max(velocity[:, 1])),
                'z': (np.min(velocity[:, 2]), np.max(velocity[:, 2]))
            }
        }
    
    def validate_files(self) -> Dict:
        """
        Validate all VTK files and check for consistency.
        
        Returns:
            Validation report dictionary
        """
        print("Validating VTK files...")
        
        validation_report = {
            'total_files': len(self.time_files),
            'valid_files': 0,
            'invalid_files': [],
            'grid_consistency': True,
            'velocity_field_consistency': True,
            'errors': []
        }
        
        first_grid_points = None
        first_velocity_field = None
        
        for i, file_path in enumerate(self.time_files):
            try:
                # Try to load the file
                if file_path.endswith('.pvtu'):
                    reader = vtk.vtkXMLPUnstructuredGridReader()
                elif file_path.endswith('.vtu'):
                    reader = vtk.vtkXMLUnstructuredGridReader()
                elif file_path.endswith('.vtk'):
                    reader = vtk.vtkUnstructuredGridReader()
                
                reader.SetFileName(file_path)
                reader.Update()
                output = reader.GetOutput()
                
                # Check grid consistency
                current_points = vtk_to_numpy(output.GetPoints().GetData())
                if first_grid_points is None:
                    first_grid_points = current_points
                elif not np.allclose(first_grid_points, current_points, rtol=1e-10):
                    validation_report['grid_consistency'] = False
                    validation_report['errors'].append(f"Grid mismatch in file {i}: {file_path}")
                
                # Check velocity field
                velocity_name = self._detect_velocity_field(output)
                if first_velocity_field is None:
                    first_velocity_field = velocity_name
                elif velocity_name != first_velocity_field:
                    validation_report['velocity_field_consistency'] = False
                    validation_report['errors'].append(f"Velocity field name mismatch in file {i}: {file_path}")
                
                validation_report['valid_files'] += 1
                
            except Exception as e:
                validation_report['invalid_files'].append((file_path, str(e)))
                validation_report['errors'].append(f"Error loading file {i}: {file_path} - {str(e)}")
        
        print(f"Validation complete: {validation_report['valid_files']}/{validation_report['total_files']} files valid")
        
        if validation_report['errors']:
            print("Validation errors found:")
            for error in validation_report['errors']:
                print(f"  - {error}")
        
        return validation_report
    
    def __len__(self) -> int:
        """Return number of time steps."""
        return len(self.time_files)
    
    def __getitem__(self, time_idx: int) -> np.ndarray:
        """Get velocity field for a time step using array indexing."""
        return self.load_single_timestep(time_idx)
    
    def __repr__(self) -> str:
        """String representation of the VTKReader."""
        return (f"VTKReader(pattern='{self.file_pattern}', "
                f"n_timesteps={len(self.time_files)}, "
                f"velocity_field='{self.velocity_field_name}', "
                f"cache_size={len(self.velocity_cache)})")
