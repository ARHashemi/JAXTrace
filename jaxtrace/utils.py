"""
Utility Functions for JAXTrace

Memory management, configuration presets, and helper functions.
"""

import numpy as np
import functools
import psutil
import os
from typing import Dict, Tuple, Optional, List, Union, Callable
import warnings

# VTK imports for output writing
try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    warnings.warn("VTK not available. VTK output functionality will be disabled.")

# JAX imports with error handling
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class VTKWriter:
    """VTK writer for particle tracking results and density fields."""
    
    def __init__(self, output_directory: str = "vtk_output"):
        """
        Initialize VTK writer.
        
        Args:
            output_directory: Directory to save VTK files
        """
        if not VTK_AVAILABLE:
            raise ImportError("VTK is required for VTKWriter. Please install VTK.")
        
        self.output_directory = output_directory
        os.makedirs(output_directory, exist_ok=True)
        print(f"VTKWriter initialized. Output directory: {output_directory}")
    
    def save_particle_positions(self, 
                               positions: np.ndarray,
                               filename: str,
                               point_data: Optional[Dict[str, np.ndarray]] = None,
                               time_value: Optional[float] = None) -> str:
        """
        Save particle positions as VTK points.
        
        Args:
            positions: Particle positions (N, 3)
            filename: Output filename (without .vtp extension)
            point_data: Additional point data arrays {name: array}
            time_value: Time value for temporal data
            
        Returns:
            Full path to saved file
        """
        if positions.shape[1] != 3:
            raise ValueError("Positions must be Nx3 array")
        
        # Create VTK points
        points = vtk.vtkPoints()
        for i in range(len(positions)):
            points.InsertNextPoint(positions[i])
        
        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        # Create vertices for visualization
        vertices = vtk.vtkCellArray()
        for i in range(len(positions)):
            vertices.InsertNextCell(1, [i])
        polydata.SetVerts(vertices)
        
        # Add point data
        if point_data is not None:
            for name, data in point_data.items():
                if len(data) != len(positions):
                    raise ValueError(f"Point data '{name}' length {len(data)} doesn't match positions length {len(positions)}")
                
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
        
        # Add time information if provided
        if time_value is not None:
            time_array = vtk.vtkFloatArray()
            time_array.SetName("TimeValue")
            time_array.SetNumberOfTuples(1)
            time_array.SetValue(0, time_value)
            polydata.GetFieldData().AddArray(time_array)
        
        # Write file
        filepath = os.path.join(self.output_directory, f"{filename}.vtp")
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filepath)
        writer.SetInputData(polydata)
        writer.Write()
        
        print(f"Saved particle positions to {filepath} ({len(positions)} particles)")
        return filepath
    
    def save_density_field(self,
                          X: np.ndarray,
                          Y: np.ndarray, 
                          Z: np.ndarray,
                          density: np.ndarray,
                          filename: str,
                          plane: str = 'xy',
                          position: float = 0.0,
                          additional_fields: Optional[Dict[str, np.ndarray]] = None) -> str:
        """
        Save density field as VTK structured grid.
        
        Args:
            X: X coordinates meshgrid
            Y: Y coordinates meshgrid  
            Z: Z values (density)
            density: Density values (same shape as X, Y)
            filename: Output filename (without .vti extension)
            plane: Plane of the density field ('xy', 'xz', 'yz')
            position: Position along the third axis
            additional_fields: Additional scalar fields {name: array}
            
        Returns:
            Full path to saved file
        """
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
        
        # Write file
        filepath = os.path.join(self.output_directory, f"{filename}.vti")
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(filepath)
        writer.SetInputData(grid)
        writer.Write()
        
        print(f"Saved density field to {filepath} ({nx}x{ny} grid)")
        return filepath
    
    def save_particle_trajectories(self,
                                  trajectories: np.ndarray,
                                  filename: str,
                                  time_values: Optional[np.ndarray] = None,
                                  particle_ids: Optional[np.ndarray] = None) -> str:
        """
        Save particle trajectories as VTK polylines.
        
        Args:
            trajectories: Particle trajectories (N_particles, N_timesteps, 3)
            filename: Output filename (without .vtp extension)
            time_values: Time values for each timestep
            particle_ids: Particle IDs
            
        Returns:
            Full path to saved file
        """
        n_particles, n_timesteps, _ = trajectories.shape
        
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
        
        # Write file
        filepath = os.path.join(self.output_directory, f"{filename}.vtp")
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filepath)
        writer.SetInputData(polydata)
        writer.Write()
        
        print(f"Saved particle trajectories to {filepath} ({n_particles} particles, {n_timesteps} timesteps)")
        return filepath
    
    def save_combined_results(self,
                             final_positions: np.ndarray,
                             density_data: Optional[Dict] = None,
                             trajectories: Optional[np.ndarray] = None,
                             initial_positions: Optional[np.ndarray] = None,
                             base_filename: str = "particle_tracking_results",
                             time_value: Optional[float] = None) -> Dict[str, str]:
        """
        Save all particle tracking results in one call.
        
        Args:
            final_positions: Final particle positions (N, 3)
            density_data: Dictionary with density field data
                         {'X': X_grid, 'Y': Y_grid, 'density': density_values, 
                          'plane': 'xy', 'position': 0.0}
            trajectories: Full particle trajectories (N, timesteps, 3)
            initial_positions: Initial particle positions (N, 3)
            base_filename: Base filename for all outputs
            time_value: Time value for the final state
            
        Returns:
            Dictionary mapping data type to saved file path
        """
        saved_files = {}
        
        # Save final positions
        point_data = {}
        if initial_positions is not None:
            displacement = final_positions - initial_positions
            displacement_magnitude = np.linalg.norm(displacement, axis=1)
            point_data['Displacement'] = displacement
            point_data['DisplacementMagnitude'] = displacement_magnitude
        
        final_file = self.save_particle_positions(
            final_positions,
            f"{base_filename}_final",
            point_data=point_data,
            time_value=time_value
        )
        saved_files['final_positions'] = final_file
        
        # Save initial positions if provided
        if initial_positions is not None:
            initial_file = self.save_particle_positions(
                initial_positions,
                f"{base_filename}_initial",
                time_value=0.0 if time_value is not None else None
            )
            saved_files['initial_positions'] = initial_file
        
        # Save trajectories if provided
        if trajectories is not None:
            traj_file = self.save_particle_trajectories(
                trajectories,
                f"{base_filename}_trajectories"
            )
            saved_files['trajectories'] = traj_file
        
        # Save density field if provided
        if density_data is not None:
            density_file = self.save_density_field(
                density_data['X'],
                density_data['Y'],
                density_data['density'],
                density_data.get('density', np.zeros_like(density_data['X'])),
                f"{base_filename}_density",
                plane=density_data.get('plane', 'xy'),
                position=density_data.get('position', 0.0)
            )
            saved_files['density_field'] = density_file
        
        print(f"Saved {len(saved_files)} files to {self.output_directory}")
        return saved_files
    
    def create_paraview_state_file(self,
                                  saved_files: Dict[str, str],
                                  state_filename: str = "paraview_state.pvsm") -> str:
        """
        Create a basic ParaView state file for easy loading.
        
        Args:
            saved_files: Dictionary from save_combined_results
            state_filename: Name of the state file
            
        Returns:
            Path to the created state file
        """
        state_content = f"""<?xml version="1.0"?>
<ParaView version="5.11.0">
  <ServerManagerState version="5.11.0">
    <GlobalPropertiesManager group_="misc" type="GlobalPropertiesManager" global_properties_version="1" camera_properties_version="1" />
"""
        
        # Add data sources
        for data_type, filepath in saved_files.items():
            rel_path = os.path.relpath(filepath, self.output_directory)
            if data_type == 'final_positions':
                state_content += f"""
    <Proxy group="sources" type="XMLPolyDataReader" id="1" servers="1">
      <Property name="FileName" id="1.FileName" number_of_elements="1">
        <Element index="0" value="{rel_path}"/>
      </Property>
    </Proxy>"""
            elif data_type == 'density_field':
                state_content += f"""
    <Proxy group="sources" type="XMLStructuredGridReader" id="2" servers="1">
      <Property name="FileName" id="2.FileName" number_of_elements="1">
        <Element index="0" value="{rel_path}"/>
      </Property>
    </Proxy>"""
        
        state_content += """
  </ServerManagerState>
</ParaView>"""
        
        state_filepath = os.path.join(self.output_directory, state_filename)
        with open(state_filepath, 'w') as f:
            f.write(state_content)
        
        print(f"Created ParaView state file: {state_filepath}")
        return state_filepath


def get_memory_config(scenario: str) -> Dict:
    """
    Get memory configuration for different computational scenarios.
    
    Args:
        scenario: Scenario type ('low_memory', 'medium_memory', 'high_memory', 'high_accuracy')
        
    Returns:
        Dictionary with configuration parameters
    """
    configs = {
        'low_memory': {
            'max_time_steps': 20,
            'particle_resolution': (20, 20, 20),
            'integration_method': 'euler',
            'spatial_subsample': 2,
            'temporal_subsample': 2,
            'max_gpu_memory_gb': 4.0,
            'k_neighbors': 4,
            'shape_function': 'linear',
            'interpolation_method': 'nearest_neighbor',
            'cache_size_limit': 3
        },
        'medium_memory': {
            'max_time_steps': 40,
            'particle_resolution': (30, 30, 30),
            'integration_method': 'rk2',
            'spatial_subsample': 1,
            'temporal_subsample': 1,
            'max_gpu_memory_gb': 8.0,
            'k_neighbors': 6,
            'shape_function': 'linear',
            'interpolation_method': 'finite_element',
            'cache_size_limit': 5
        },
        'high_memory': {
            'max_time_steps': 100,
            'particle_resolution': (50, 50, 50),
            'integration_method': 'rk4',
            'spatial_subsample': 1,
            'temporal_subsample': 1,
            'max_gpu_memory_gb': 16.0,
            'k_neighbors': 8,
            'shape_function': 'quadratic',
            'interpolation_method': 'finite_element',
            'cache_size_limit': 10
        },
        'high_accuracy': {
            'max_time_steps': 60,
            'particle_resolution': (40, 40, 40),
            'integration_method': 'rk4',
            'spatial_subsample': 1,
            'temporal_subsample': 1,
            'max_gpu_memory_gb': 12.0,
            'k_neighbors': 12,
            'shape_function': 'cubic',
            'interpolation_method': 'finite_element',
            'cache_size_limit': 7
        }
    }
    
    if scenario not in configs:
        available_scenarios = list(configs.keys())
        raise ValueError(f"Unknown scenario '{scenario}'. Available scenarios: {available_scenarios}")
    
    return configs[scenario].copy()


def monitor_memory_usage(func):
    """
    Decorator to monitor memory usage of functions.
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function with memory monitoring
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024**3
        
        result = func(*args, **kwargs)
        
        mem_after = process.memory_info().rss / 1024**3
        print(f"{func.__name__}: Memory usage: {mem_before:.2f} -> {mem_after:.2f} GB "
              f"(Δ{mem_after - mem_before:+.2f} GB)")
        
        return result
    return wrapper


def get_gpu_memory_usage() -> Tuple[Optional[float], Optional[float]]:
    """
    Get current GPU memory usage.
    
    Returns:
        Tuple of (used_memory_gb, total_memory_gb) or (None, None) if unavailable
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024**3, info.total / 1024**3
    except ImportError:
        warnings.warn("pynvml not available. GPU memory monitoring disabled.")
        return None, None
    except Exception as e:
        warnings.warn(f"Error getting GPU memory info: {e}")
        return None, None


def get_system_info() -> Dict:
    """
    Get comprehensive system information for performance tuning.
    
    Returns:
        Dictionary with system information
    """
    info = {
        'cpu_count': psutil.cpu_count(),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'memory_total_gb': psutil.virtual_memory().total / 1024**3,
        'memory_available_gb': psutil.virtual_memory().available / 1024**3,
        'memory_percent_used': psutil.virtual_memory().percent,
        'jax_available': JAX_AVAILABLE
    }
    
    # GPU information
    gpu_used, gpu_total = get_gpu_memory_usage()
    if gpu_used is not None:
        info['gpu_memory_used_gb'] = gpu_used
        info['gpu_memory_total_gb'] = gpu_total
        info['gpu_memory_percent_used'] = (gpu_used / gpu_total) * 100
    else:
        info['gpu_available'] = False
    
    # JAX device information
    if JAX_AVAILABLE:
        try:
            devices = jax.devices()
            info['jax_devices'] = [str(device) for device in devices]
            info['jax_default_backend'] = jax.default_backend()
        except Exception as e:
            info['jax_error'] = str(e)
    
    return info


def create_custom_particle_distribution(box_bounds: Tuple[Tuple[float, float], ...],
                                       distribution_type: str = 'uniform',
                                       n_particles: int = 10000,
                                       random_seed: int = 42) -> np.ndarray:
    """
    Create custom particle distributions for specialized initial conditions.
    
    Args:
        box_bounds: Bounds as ((xmin, xmax), (ymin, ymax), (zmin, zmax))
        distribution_type: 'uniform', 'random', 'gaussian', 'stratified', 'clustered'
        n_particles: Number of particles to create
        random_seed: Random seed for reproducibility
        
    Returns:
        Particle positions array (n_particles, 3)
    """
    if len(box_bounds) != 3:
        raise ValueError("box_bounds must have 3 dimensions")
    
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = box_bounds
    np.random.seed(random_seed)
    
    if distribution_type == 'uniform':
        # Uniform grid distribution
        n_per_dim = int(np.ceil(n_particles**(1/3)))
        x = np.linspace(xmin, xmax, n_per_dim)
        y = np.linspace(ymin, ymax, n_per_dim)
        z = np.linspace(zmin, zmax, n_per_dim)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        positions = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
    elif distribution_type == 'random':
        # Random uniform distribution
        positions = np.random.rand(n_particles, 3)
        positions[:, 0] = positions[:, 0] * (xmax - xmin) + xmin
        positions[:, 1] = positions[:, 1] * (ymax - ymin) + ymin
        positions[:, 2] = positions[:, 2] * (zmax - zmin) + zmin
        
    elif distribution_type == 'gaussian':
        # Gaussian distribution centered in domain
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        center_z = (zmin + zmax) / 2
        
        std_x = (xmax - xmin) / 6  # 3-sigma within bounds
        std_y = (ymax - ymin) / 6
        std_z = (zmax - zmin) / 6
        
        positions = np.random.normal(
            loc=[center_x, center_y, center_z],
            scale=[std_x, std_y, std_z],
            size=(n_particles, 3)
        )
        
        # Clip to bounds
        positions[:, 0] = np.clip(positions[:, 0], xmin, xmax)
        positions[:, 1] = np.clip(positions[:, 1], ymin, ymax)
        positions[:, 2] = np.clip(positions[:, 2], zmin, zmax)
        
    elif distribution_type == 'stratified':
        # Stratified sampling for better coverage
        n_per_dim = int(np.ceil(n_particles**(1/3)))
        positions = []
        
        for i in range(n_per_dim):
            for j in range(n_per_dim):
                for k in range(n_per_dim):
                    if len(positions) >= n_particles:
                        break
                    
                    # Stratified sample within each cell
                    x = xmin + (i + np.random.rand()) * (xmax - xmin) / n_per_dim
                    y = ymin + (j + np.random.rand()) * (ymax - ymin) / n_per_dim
                    z = zmin + (k + np.random.rand()) * (zmax - zmin) / n_per_dim
                    
                    positions.append([x, y, z])
        
        positions = np.array(positions[:n_particles])
        
    elif distribution_type == 'clustered':
        # Multiple Gaussian clusters
        n_clusters = min(5, max(2, n_particles // 1000))  # 2-5 clusters
        positions = []
        particles_per_cluster = n_particles // n_clusters
        
        for i in range(n_clusters):
            # Random cluster center
            center_x = np.random.uniform(xmin + 0.1*(xmax-xmin), xmax - 0.1*(xmax-xmin))
            center_y = np.random.uniform(ymin + 0.1*(ymax-ymin), ymax - 0.1*(ymax-ymin))
            center_z = np.random.uniform(zmin + 0.1*(zmax-zmin), zmax - 0.1*(zmax-zmin))
            
            # Random cluster size
            std_x = np.random.uniform(0.05, 0.15) * (xmax - xmin)
            std_y = np.random.uniform(0.05, 0.15) * (ymax - ymin)
            std_z = np.random.uniform(0.05, 0.15) * (zmax - zmin)
            
            # Generate cluster
            if i == n_clusters - 1:  # Last cluster gets remaining particles
                n_cluster_particles = n_particles - len(positions)
            else:
                n_cluster_particles = particles_per_cluster
            
            cluster_positions = np.random.normal(
                loc=[center_x, center_y, center_z],
                scale=[std_x, std_y, std_z],
                size=(n_cluster_particles, 3)
            )
            
            # Clip to bounds
            cluster_positions[:, 0] = np.clip(cluster_positions[:, 0], xmin, xmax)
            cluster_positions[:, 1] = np.clip(cluster_positions[:, 1], ymin, ymax)
            cluster_positions[:, 2] = np.clip(cluster_positions[:, 2], zmin, zmax)
            
            positions.extend(cluster_positions.tolist())
        
        positions = np.array(positions)
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}. "
                        f"Available types: 'uniform', 'random', 'gaussian', 'stratified', 'clustered'")
    
    print(f"Created {len(positions)} particles with '{distribution_type}' distribution")
    return positions


def estimate_computational_requirements(n_particles: int,
                                      n_timesteps: int,
                                      integration_method: str = 'euler',
                                      interpolation_method: str = 'finite_element',
                                      k_neighbors: int = 8) -> Dict:
    """
    Estimate computational requirements for a particle tracking simulation.
    
    Args:
        n_particles: Number of particles
        n_timesteps: Number of time steps
        integration_method: Integration method ('euler', 'rk2', 'rk4')
        interpolation_method: Interpolation method ('nearest_neighbor', 'finite_element')
        k_neighbors: Number of neighbors for finite element method
        
    Returns:
        Dictionary with estimated requirements
    """
    # Base memory requirements (bytes per particle)
    base_memory_per_particle = 3 * 4  # 3D position in float32
    
    # Integration method multipliers
    integration_multipliers = {
        'euler': 1.0,
        'rk2': 2.0,
        'rk4': 4.0
    }
    
    # Interpolation method multipliers
    interpolation_multipliers = {
        'nearest_neighbor': 1.0,
        'finite_element': k_neighbors / 4.0  # Scales with neighbors
    }
    
    int_mult = integration_multipliers.get(integration_method, 1.0)
    interp_mult = interpolation_multipliers.get(interpolation_method, 1.0)
    
    # Memory estimates
    particle_memory_gb = (n_particles * base_memory_per_particle * int_mult * interp_mult) / 1024**3
    grid_memory_estimate_gb = 0.5  # Rough estimate for grid data
    total_memory_gb = particle_memory_gb + grid_memory_estimate_gb
    
    # Runtime estimates (very rough)
    base_ops_per_particle_per_step = 100  # Basic operations
    interp_ops_multiplier = k_neighbors if interpolation_method == 'finite_element' else 1
    int_ops_multiplier = {'euler': 1, 'rk2': 2, 'rk4': 4}.get(integration_method, 1)
    
    total_operations = n_particles * n_timesteps * base_ops_per_particle_per_step * interp_ops_multiplier * int_ops_multiplier
    estimated_runtime_seconds = total_operations / 1e9  # Assume 1 GFLOPS
    
    # Recommendations
    recommendations = []
    
    if total_memory_gb > 16:
        recommendations.append("Consider using spatial/temporal subsampling for memory reduction")
    
    if n_particles > 100000:
        recommendations.append("Use batch processing for large particle counts")
    
    if integration_method == 'rk4' and n_particles > 50000:
        recommendations.append("Consider RK2 or Euler for better performance with many particles")
    
    if interpolation_method == 'finite_element' and k_neighbors > 12:
        recommendations.append("High k_neighbors may impact performance significantly")
    
    return {
        'n_particles': n_particles,
        'n_timesteps': n_timesteps,
        'integration_method': integration_method,
        'interpolation_method': interpolation_method,
        'k_neighbors': k_neighbors,
        'estimated_memory_gb': total_memory_gb,
        'particle_memory_gb': particle_memory_gb,
        'grid_memory_gb': grid_memory_estimate_gb,
        'estimated_runtime_seconds': estimated_runtime_seconds,
        'estimated_runtime_hours': estimated_runtime_seconds / 3600,
        'total_operations': total_operations,
        'recommendations': recommendations
    }


def validate_simulation_parameters(config: Dict) -> Tuple[bool, List[str]]:
    """
    Validate simulation parameters and provide warnings/errors.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, warnings/errors)
    """
    warnings_errors = []
    is_valid = True
    
    # Required parameters
    required_params = ['particle_resolution', 'integration_method', 'max_gpu_memory_gb']
    for param in required_params:
        if param not in config:
            warnings_errors.append(f"ERROR: Missing required parameter '{param}'")
            is_valid = False
    
    # Validate particle resolution
    if 'particle_resolution' in config:
        resolution = config['particle_resolution']
        if not isinstance(resolution, (tuple, list)) or len(resolution) != 3:
            warnings_errors.append("ERROR: particle_resolution must be a 3-tuple (nx, ny, nz)")
            is_valid = False
        elif any(r <= 0 for r in resolution):
            warnings_errors.append("ERROR: particle_resolution values must be positive")
            is_valid = False
        elif np.prod(resolution) > 1000000:
            warnings_errors.append("WARNING: Very large particle count may cause memory issues")
    
    # Validate integration method
    if 'integration_method' in config:
        valid_methods = ['euler', 'rk2', 'rk4']
        if config['integration_method'] not in valid_methods:
            warnings_errors.append(f"ERROR: integration_method must be one of {valid_methods}")
            is_valid = False
    
    # Validate interpolation method
    if 'interpolation_method' in config:
        valid_interp = ['nearest_neighbor', 'finite_element']
        if config['interpolation_method'] not in valid_interp:
            warnings_errors.append(f"ERROR: interpolation_method must be one of {valid_interp}")
            is_valid = False
    
    # Validate shape function
    if 'shape_function' in config:
        valid_shapes = ['linear', 'quadratic', 'cubic', 'gaussian']
        if config['shape_function'] not in valid_shapes:
            warnings_errors.append(f"ERROR: shape_function must be one of {valid_shapes}")
            is_valid = False
    
    # Memory validation
    if 'max_gpu_memory_gb' in config:
        max_mem = config['max_gpu_memory_gb']
        if max_mem <= 0:
            warnings_errors.append("ERROR: max_gpu_memory_gb must be positive")
            is_valid = False
        elif max_mem < 2:
            warnings_errors.append("WARNING: Very low GPU memory limit may cause issues")
    
    # k_neighbors validation
    if 'k_neighbors' in config:
        k = config['k_neighbors']
        if k < 1:
            warnings_errors.append("ERROR: k_neighbors must be at least 1")
            is_valid = False
        elif k > 20:
            warnings_errors.append("WARNING: High k_neighbors may significantly impact performance")
    
    # Subsampling validation
    for param in ['spatial_subsample', 'temporal_subsample']:
        if param in config:
            value = config[param]
            if value < 1:
                warnings_errors.append(f"ERROR: {param} must be at least 1")
                is_valid = False
    
    return is_valid, warnings_errors


def optimize_parameters_for_system(base_config: Dict, 
                                  target_memory_gb: Optional[float] = None,
                                  target_runtime_hours: Optional[float] = None) -> Dict:
    """
    Automatically optimize parameters based on system constraints.
    
    Args:
        base_config: Base configuration dictionary
        target_memory_gb: Target memory usage in GB (None for auto-detect)
        target_runtime_hours: Target runtime in hours (None for no constraint)
        
    Returns:
        Optimized configuration dictionary
    """
    config = base_config.copy()
    
    # Auto-detect available memory if not specified
    if target_memory_gb is None:
        system_info = get_system_info()
        if 'gpu_memory_total_gb' in system_info:
            target_memory_gb = system_info['gpu_memory_total_gb'] * 0.8  # Use 80% of GPU memory
        else:
            target_memory_gb = system_info['memory_available_gb'] * 0.5  # Use 50% of system memory
    
    # Calculate current requirements
    if 'particle_resolution' in config:
        n_particles = np.prod(config['particle_resolution'])
        n_timesteps = config.get('max_time_steps', 50)
        
        current_req = estimate_computational_requirements(
            n_particles, n_timesteps,
            config.get('integration_method', 'euler'),
            config.get('interpolation_method', 'finite_element'),
            config.get('k_neighbors', 8)
        )
        
        # Optimize if memory exceeds target
        if current_req['estimated_memory_gb'] > target_memory_gb:
            print(f"Optimizing: Current memory estimate {current_req['estimated_memory_gb']:.2f} GB > target {target_memory_gb:.2f} GB")
            
            # Try reducing particle resolution first
            scale_factor = (target_memory_gb / current_req['estimated_memory_gb']) ** (1/3)
            new_resolution = tuple(max(5, int(r * scale_factor)) for r in config['particle_resolution'])
            config['particle_resolution'] = new_resolution
            print(f"Reduced particle resolution to {new_resolution}")
            
            # If still too large, add subsampling
            new_n_particles = np.prod(new_resolution)
            new_req = estimate_computational_requirements(
                new_n_particles, n_timesteps,
                config.get('integration_method', 'euler'),
                config.get('interpolation_method', 'finite_element'),
                config.get('k_neighbors', 8)
            )
            
            if new_req['estimated_memory_gb'] > target_memory_gb:
                subsample_factor = int(np.ceil(new_req['estimated_memory_gb'] / target_memory_gb))
                config['spatial_subsample'] = max(config.get('spatial_subsample', 1), subsample_factor)
                print(f"Added spatial subsampling factor: {subsample_factor}")
        
        # Optimize for runtime if specified
        if target_runtime_hours is not None:
            if current_req['estimated_runtime_hours'] > target_runtime_hours:
                print(f"Optimizing runtime: Current estimate {current_req['estimated_runtime_hours']:.2f} h > target {target_runtime_hours:.2f} h")
                
                # Switch to faster integration method
                if config.get('integration_method') == 'rk4':
                    config['integration_method'] = 'rk2'
                    print("Switched from RK4 to RK2 integration")
                elif config.get('integration_method') == 'rk2':
                    config['integration_method'] = 'euler'
                    print("Switched from RK2 to Euler integration")
                
                # Reduce k_neighbors for finite element
                if config.get('interpolation_method') == 'finite_element':
                    config['k_neighbors'] = min(config.get('k_neighbors', 8), 6)
                    print("Reduced k_neighbors to 6 for performance")
    
    print("Parameter optimization complete")
    return config


def create_progress_callback(update_frequency: int = 100, 
                           show_memory: bool = True,
                           show_time: bool = True) -> Callable:
    """
    Create a progress callback function for particle tracking.
    
    Args:
        update_frequency: How often to show updates
        show_memory: Whether to show memory usage
        show_time: Whether to show timing information
        
    Returns:
        Progress callback function
    """
    import time
    start_time = time.time()
    
    def progress_callback(step: int, positions: np.ndarray):
        if step % update_frequency == 0 or step == 1:
            message_parts = [f"Step {step}"]
            
            if show_time:
                elapsed = time.time() - start_time
                message_parts.append(f"elapsed: {elapsed:.1f}s")
            
            if show_memory:
                try:
                    process = psutil.Process(os.getpid())
                    mem_gb = process.memory_info().rss / 1024**3
                    message_parts.append(f"RAM: {mem_gb:.2f} GB")
                    
                    gpu_used, gpu_total = get_gpu_memory_usage()
                    if gpu_used is not None:
                        message_parts.append(f"GPU: {gpu_used:.2f}/{gpu_total:.2f} GB")
                except:
                    pass
            
            # Position statistics
            pos_mean = np.mean(positions, axis=0)
            pos_std = np.std(positions, axis=0)
            message_parts.append(f"pos_mean: [{pos_mean[0]:.2f}, {pos_mean[1]:.2f}, {pos_mean[2]:.2f}]")
            
            print(" | ".join(message_parts))
    
    return progress_callback


def save_simulation_config(config: Dict, filepath: str):
    """
    Save simulation configuration to file.
    
    Args:
        config: Configuration dictionary
        filepath: Output file path (JSON format)
    """
    import json
    import datetime
    
    # Add metadata
    config_with_meta = config.copy()
    config_with_meta['_metadata'] = {
        'created_at': datetime.datetime.now().isoformat(),
        'jaxtrace_version': '0.1.0',
        'system_info': get_system_info()
    }
    
    with open(filepath, 'w') as f:
        json.dump(config_with_meta, f, indent=2, default=str)
    
    print(f"Configuration saved to {filepath}")


def load_simulation_config(filepath: str) -> Dict:
    """
    Load simulation configuration from file.
    
    Args:
        filepath: Input file path (JSON format)
        
    Returns:
        Configuration dictionary
    """
    import json
    
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    # Remove metadata if present
    if '_metadata' in config:
        del config['_metadata']
    
    print(f"Configuration loaded from {filepath}")
    return config


def benchmark_interpolation_methods(n_particles: int = 10000,
                                   n_evaluations: int = 1000,
                                   methods: List[str] = None) -> Dict:
    """
    Benchmark different interpolation methods for performance comparison.
    
    Args:
        n_particles: Number of particles for benchmark
        n_evaluations: Number of interpolation evaluations to perform
        methods: List of methods to benchmark (None for all available)
        
    Returns:
        Dictionary with benchmark results
    """
    if not JAX_AVAILABLE:
        print("JAX not available, skipping benchmark")
        return {}
    
    import time
    
    if methods is None:
        methods = ['nearest_neighbor', 'finite_element']
    
    # Generate test data
    np.random.seed(42)
    grid_points = jnp.array(np.random.randn(n_particles, 3))
    velocity_field = jnp.array(np.random.randn(n_particles, 3))
    eval_points = jnp.array(np.random.randn(n_evaluations, 3))
    
    results = {}
    
    for method in methods:
        print(f"Benchmarking {method}...")
        
        if method == 'nearest_neighbor':
            @jax.jit
            def interpolate_fn(point):
                distances = jnp.linalg.norm(grid_points - point, axis=1)
                nearest_idx = jnp.argmin(distances)
                return velocity_field[nearest_idx]
        
        elif method == 'finite_element':
            k_neighbors = 8
            @jax.jit
            def interpolate_fn(point):
                distances = jnp.linalg.norm(grid_points - point, axis=1)
                neighbor_indices = jnp.argpartition(distances, k_neighbors)[:k_neighbors]
                neighbor_distances = distances[neighbor_indices]
                neighbor_velocities = velocity_field[neighbor_indices]
                safe_distances = jnp.maximum(neighbor_distances, 1e-10)
                weights = 1.0 / safe_distances
                weights = weights / jnp.sum(weights)
                return jnp.sum(weights[:, None] * neighbor_velocities, axis=0)
        else:
            continue
        
        # Warm up JIT compilation
        _ = interpolate_fn(eval_points[0])
        
        # Benchmark
        start_time = time.time()
        for point in eval_points:
            _ = interpolate_fn(point)
        end_time = time.time()
        
        total_time = end_time - start_time
        time_per_eval = total_time / n_evaluations
        evals_per_second = n_evaluations / total_time
        
        results[method] = {
            'total_time_seconds': total_time,
            'time_per_evaluation_ms': time_per_eval * 1000,
            'evaluations_per_second': evals_per_second,
            'n_particles': n_particles,
            'n_evaluations': n_evaluations
        }
        
        print(f"  {method}: {time_per_eval*1000:.3f} ms/eval, {evals_per_second:.0f} eval/s")
    
    return results


# Export all utility functions
__all__ = [
    'VTKWriter',
    'get_memory_config',
    'monitor_memory_usage', 
    'get_gpu_memory_usage',
    'get_system_info',
    'create_custom_particle_distribution',
    'estimate_computational_requirements',
    'validate_simulation_parameters',
    'optimize_parameters_for_system',
    'create_progress_callback',
    'save_simulation_config',
    'load_simulation_config',
    'benchmark_interpolation_methods'
]