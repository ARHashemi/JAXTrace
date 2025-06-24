import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import glob
from scipy.spatial import cKDTree
from typing import List, Dict

class VTKVelocityReader:
    """Class to read time-dependent velocity fields from VTK files."""
    
    def __init__(self, file_pattern: str):
        """
        Initialize VTK reader.
        
        Args:
            file_pattern: Pattern to match VTK files (e.g., "*_caseCoarse_*.pvtu")
        """
        self.file_pattern = file_pattern
        self.time_files = self._find_time_files()
        self.grid_data = None
        self.velocity_data = None
        self.time_steps = None
        
    def _find_time_files(self) -> List[str]:
        """Find all VTK files matching the pattern."""
        files = glob.glob(self.file_pattern)
        files.sort()  # Assume files are named in time order
        return files
    
    def load_velocity_data(self) -> Dict:
        """
        Load velocity data from all time steps.
        
        Returns:
            Dictionary containing grid coordinates, velocity data, time info, and grid bounds
        """
        velocity_fields = []
        time_values = []
        
        print(f"Loading {len(self.time_files)} VTK files...")
        
        for i, file_path in enumerate(self.time_files):
            # Read VTK file
            if file_path.endswith('.pvtu'):
                reader = vtk.vtkXMLPUnstructuredGridReader()
            elif file_path.endswith('.vtu'):
                reader = vtk.vtkXMLUnstructuredGridReader()
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            reader.SetFileName(file_path)
            reader.Update()
            
            output = reader.GetOutput()
            points = vtk_to_numpy(output.GetPoints().GetData())
            
            # Get grid bounds and structure information
            bounds = output.GetBounds()  # Returns (xmin, xmax, ymin, ymax, zmin, zmax)
            n_cells = output.GetNumberOfCells()
            cell_types = set()
            for j in range(n_cells):
                cell_types.add(output.GetCellType(j))
            
            # Try to find velocity field (common names)
            velocity_names = ['velocity', 'Velocity', 'U', 'vel', 'Displacement', 'displacement']
            velocity_array = None
            
            for name in velocity_names:
                velocity_array = output.GetPointData().GetArray(name)
                if velocity_array is not None:
                    velocity_field_name = name
                    break
            
            if velocity_array is None:
                # List available arrays
                point_data = output.GetPointData()
                available_arrays = [point_data.GetArrayName(i) 
                                  for i in range(point_data.GetNumberOfArrays())]
                raise ValueError(f"Velocity field not found. Available arrays: {available_arrays}")
            
            velocity = vtk_to_numpy(velocity_array)
            
            # Print compact information
            bounds_str = f"[{bounds[0]:.1f}:{bounds[1]:.1f}, {bounds[2]:.1f}:{bounds[3]:.1f}, {bounds[4]:.1f}:{bounds[5]:.1f}]"
            cell_types_str = ','.join([str(ct) for ct in sorted(cell_types)])
            filename = file_path.split('/')[-1]  # Get just the filename
            
            print(f"  [{i+1:2d}/{len(self.time_files)}] {filename:<25} | "
                  f"Points: {len(points):6d} | Cells: {n_cells:6d} (types: {cell_types_str}) | "
                  f"Bounds: {bounds_str} | Field: '{velocity_field_name}'")
            
            if i == 0:
                self.grid_points = points
                self.grid_tree = cKDTree(points)  # For interpolation
                self.grid_bounds = bounds
                
                # Create interpolator for velocity field
                self.interpolator = vtk.vtkProbeFilter()
                self.interpolator.SetSourceData(output)
            
            velocity_fields.append(velocity)
            time_values.append(i)  # Use index as time if no time data available
        
        self.velocity_data = np.array(velocity_fields)
        self.time_steps = np.array(time_values)
        
        print(f"Successfully loaded velocity data: {self.velocity_data.shape[0]} time steps, "
              f"{self.velocity_data.shape[1]} points, {self.velocity_data.shape[2]}D velocity")
        
        return {
            'points': self.grid_points,
            'velocity': self.velocity_data,
            'times': self.time_steps,
            'bounds': self.grid_bounds
        }