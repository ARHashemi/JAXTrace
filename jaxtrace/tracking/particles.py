# jaxtrace/tracking/particles.py  
"""  
Particle trajectory storage with consistent data types and shapes.  

Enforces (T,N,3) float32 format throughout with memory-efficient  
storage and JAX-compatible data structures.  
"""  

from __future__ import annotations  
from dataclasses import dataclass, field  
from typing import Optional, List, Tuple, Union, Dict, Any  
import numpy as np  
import warnings  

# Import JAX utilities with fallback  
from ..utils.jax_utils import JAX_AVAILABLE  

if JAX_AVAILABLE:  
    try:  
        import jax.numpy as jnp  
    except Exception:  
        JAX_AVAILABLE = False  

if not JAX_AVAILABLE:  
    import numpy as jnp  # type: ignore  


def _ensure_float32(data: np.ndarray) -> np.ndarray:  
    """Convert data to float32 for consistency and JAX performance."""  
    return np.asarray(data, dtype=np.float32)  


def _ensure_trajectory_shape(positions: np.ndarray) -> np.ndarray:  
    """  
    Ensure trajectory data has shape (T, N, 3) with float32 dtype.  
    
    Parameters  
    ----------  
    positions : np.ndarray  
        Position data in various formats  
        
    Returns  
    -------  
    np.ndarray  
        Trajectory with shape (T, N, 3), dtype float32  
    """  
    pos = _ensure_float32(positions)  
    
    if pos.ndim == 1:  
        # Single particle, single time: (3,) -> (1, 1, 3)  
        pos = pos.reshape(1, 1, -1)  
    elif pos.ndim == 2:  
        if pos.shape[1] == 2:  
            # 2D trajectory: (T, 2) -> (T, 1, 3)  
            z_zeros = np.zeros((pos.shape[0], 1), dtype=np.float32)  
            pos = np.concatenate([pos, z_zeros], axis=1)  
            pos = pos[:, None, :]  # Add particle dimension  
        elif pos.shape[1] == 3:  
            # Single particle trajectory: (T, 3) -> (T, 1, 3)  
            pos = pos[:, None, :]  
        else:  
            # Multiple particles, single time: (N, 3) -> (1, N, 3)  
            if pos.shape[1] == 3:  
                pos = pos[None, :, :]  
            else:  
                raise ValueError(f"Cannot interpret 2D array with shape {pos.shape}")  
    elif pos.ndim == 3:  
        # Already correct format: (T, N, D)  
        T, N, D = pos.shape  
        if D == 2:  
            # Convert 2D to 3D by adding zero z-component  
            z_zeros = np.zeros((T, N, 1), dtype=np.float32)  
            pos = np.concatenate([pos, z_zeros], axis=2)  
        elif D != 3:  
            raise ValueError(f"Position data must have 2 or 3 spatial dimensions, got {D}")  
    else:  
        raise ValueError(f"Position data must be 1D, 2D, or 3D array, got {pos.ndim}D")  
    
    return pos.astype(np.float32, copy=False)  


@dataclass  
class Trajectory:  
    """  
    Container for particle trajectory data with consistent format.  
    
    Enforces (T,N,3) float32 storage format throughout for optimal  
    JAX compatibility and memory efficiency.  
    
    Attributes  
    ----------  
    positions : np.ndarray  
        Particle positions, shape (T, N, 3), dtype float32  
    times : np.ndarray  
        Time points, shape (T,), dtype float32  
    velocities : np.ndarray, optional  
        Particle velocities, shape (T, N, 3), dtype float32  
    metadata : dict  
        Additional trajectory information  
    """  
    positions: np.ndarray          # (T, N, 3) - standardized format  
    times: np.ndarray             # (T,) - time points  
    velocities: Optional[np.ndarray] = None  # (T, N, 3) - optional velocities  
    metadata: Dict[str, Any] = field(default_factory=dict)  

    def __post_init__(self):  
        # Ensure consistent data types and shapes  
        self.positions = _ensure_trajectory_shape(self.positions)  
        self.times = _ensure_float32(self.times)  
        
        # Get standardized dimensions  
        self.T, self.N, self.D = self.positions.shape  
        if self.D != 3:  
            raise ValueError(f"Positions must have 3 spatial dimensions after processing")  
        
        # Validate time array  
        if self.times.shape != (self.T,):  
            raise ValueError(f"Times shape {self.times.shape} doesn't match trajectory length {self.T}")  
        
        # Process optional velocities  
        if self.velocities is not None:  
            self.velocities = _ensure_trajectory_shape(self.velocities)  
            if self.velocities.shape != (self.T, self.N, 3):  
                raise ValueError(f"Velocities shape {self.velocities.shape} doesn't match positions shape {self.positions.shape}")  
        
        # Store metadata  
        self.metadata.setdefault('format_version', '1.0')  
        self.metadata.setdefault('dtype', 'float32')  
        self.metadata.setdefault('coordinate_system', 'cartesian')  

    # ---------- Core accessors ----------  
    
    def __len__(self) -> int:  
        """Number of time steps."""  
        return self.T  
    
    def __getitem__(self, key: Union[int, slice, tuple]) -> 'Trajectory':  
        """Slice trajectory in time and/or particles."""  
        if isinstance(key, int):  
            # Single time step  
            return Trajectory(  
                positions=self.positions[key:key+1],  # Keep 3D shape  
                times=self.times[key:key+1],  
                velocities=self.velocities[key:key+1] if self.velocities is not None else None,  
                metadata=self.metadata.copy(),  
            )  
        elif isinstance(key, slice):  
            # Time slice  
            return Trajectory(  
                positions=self.positions[key],  
                times=self.times[key],  
                velocities=self.velocities[key] if self.velocities is not None else None,  
                metadata=self.metadata.copy(),  
            )  
        elif isinstance(key, tuple) and len(key) == 2:  
            # Time and particle slice  
            t_key, p_key = key  
            return Trajectory(  
                positions=self.positions[t_key, p_key],  
                times=self.times[t_key],  
                velocities=self.velocities[t_key, p_key] if self.velocities is not None else None,  
                metadata=self.metadata.copy(),  
            )  
        else:  
            raise TypeError(f"Invalid key type for Trajectory indexing: {type(key)}")  

    @property   
    def num_particles(self) -> int:  
        """Number of particles."""  
        return self.N  
    
    @property  
    def num_timesteps(self) -> int:  
        """Number of time steps."""  
        return self.T  

    @property  
    def duration(self) -> float:  
        """Total trajectory duration."""  
        return float(self.times[-1] - self.times[0]) if self.T > 1 else 0.0  

    @property  
    def dt_mean(self) -> float:  
        """Average time step size."""  
        return float(np.mean(np.diff(self.times))) if self.T > 1 else 0.0  

    # ---------- Data extraction ----------  

    def get_positions_at_time(self, t_idx: int) -> np.ndarray:  
        """  
        Get positions at specific time index.  
        
        Parameters  
        ----------  
        t_idx : int  
            Time index  
            
        Returns  
        -------  
        np.ndarray  
            Positions at time t_idx, shape (N, 3), dtype float32  
        """  
        if not 0 <= t_idx < self.T:  
            raise IndexError(f"Time index {t_idx} out of range [0, {self.T})")  
        
        return self.positions[t_idx].copy()  # (N, 3)  

    def get_particle_trajectory(self, p_idx: int) -> Tuple[np.ndarray, np.ndarray]:  
        """  
        Get trajectory for specific particle.  
        
        Parameters  
        ----------  
        p_idx : int  
            Particle index  
            
        Returns  
        -------  
        positions : np.ndarray  
            Particle trajectory, shape (T, 3), dtype float32  
        times : np.ndarray  
            Time points, shape (T,), dtype float32  
        """  
        if not 0 <= p_idx < self.N:  
            raise IndexError(f"Particle index {p_idx} out of range [0, {self.N})")  
        
        return self.positions[:, p_idx].copy(), self.times.copy()  

    def extract_2d_slice(self, plane: str = "xy") -> 'Trajectory':  
        """  
        Extract 2D trajectory from 3D data.  
        
        Parameters  
        ----------  
        plane : str  
            Plane to extract: 'xy', 'xz', or 'yz'  
            
        Returns  
        -------  
        Trajectory  
            2D trajectory (still stored in (T,N,3) format with zeros)  
        """  
        plane = plane.lower()  
        if plane not in ['xy', 'xz', 'yz']:  
            raise ValueError("plane must be 'xy', 'xz', or 'yz'")  
        
        # Extract relevant coordinates but maintain (T,N,3) format  
        pos_2d = self.positions.copy()  # (T, N, 3)  
        
        if plane == "xy":  
            pos_2d[:, :, 2] = 0.0  # Zero out z-component  
        elif plane == "xz":  
            pos_2d[:, :, 1] = 0.0  # Zero out y-component  
        elif plane == "yz":  
            pos_2d[:, :, 0] = 0.0  # Zero out x-component  
        
        vel_2d = None  
        if self.velocities is not None:  
            vel_2d = self.velocities.copy()  
            if plane == "xy":  
                vel_2d[:, :, 2] = 0.0  
            elif plane == "xz":  
                vel_2d[:, :, 1] = 0.0  
            elif plane == "yz":  
                vel_2d[:, :, 0] = 0.0  
        
        metadata = self.metadata.copy()  
        metadata['extracted_plane'] = plane  
        
        return Trajectory(  
            positions=pos_2d,  
            times=self.times.copy(),  
            velocities=vel_2d,  
            metadata=metadata,  
        )  

    # ---------- Statistics ----------  

    def compute_displacement(self) -> np.ndarray:  
        """  
        Compute total displacement for each particle.  
        
        Returns  
        -------  
        np.ndarray  
            Total displacements, shape (N,), dtype float32  
        """  
        if self.T < 2:  
            return np.zeros(self.N, dtype=np.float32)  
        
        start_pos = self.positions[0]   # (N, 3)  
        end_pos = self.positions[-1]    # (N, 3)  
        displacement = np.linalg.norm(end_pos - start_pos, axis=1)  # (N,)  
        return displacement.astype(np.float32)  

    def compute_path_length(self) -> np.ndarray:  
        """  
        Compute total path length for each particle.  
        
        Returns  
        -------  
        np.ndarray  
            Path lengths, shape (N,), dtype float32  
        """  
        if self.T < 2:  
            return np.zeros(self.N, dtype=np.float32)  
        
        # Compute step lengths  
        pos_diff = np.diff(self.positions, axis=0)  # (T-1, N, 3)  
        step_lengths = np.linalg.norm(pos_diff, axis=2)  # (T-1, N)  
        path_lengths = np.sum(step_lengths, axis=0)  # (N,)  
        
        return path_lengths.astype(np.float32)  

    def compute_speeds(self) -> np.ndarray:  
        """  
        Compute instantaneous speeds for all particles.  
        
        Returns  
        -------  
        np.ndarray  
            Speeds, shape (T-1, N), dtype float32  
        """  
        if self.T < 2:  
            return np.zeros((0, self.N), dtype=np.float32)  
        
        # Compute velocities if not stored  
        if self.velocities is not None:  
            speeds = np.linalg.norm(self.velocities, axis=2)  # (T, N)  
            return speeds[:-1].astype(np.float32)  # Exclude last point  
        else:  
            # Compute from position differences  
            pos_diff = np.diff(self.positions, axis=0)  # (T-1, N, 3)  
            dt_diff = np.diff(self.times)  # (T-1,)  
            
            velocities = pos_diff / dt_diff[:, None, None]  # (T-1, N, 3)  
            speeds = np.linalg.norm(velocities, axis=2)  # (T-1, N)  
            
            return speeds.astype(np.float32)  

    # ---------- Conversion utilities ----------  

    def to_jax(self) -> 'Trajectory':  
        """Convert to JAX arrays for accelerated computation."""  
        if not JAX_AVAILABLE:  
            warnings.warn("JAX not available, returning copy with NumPy arrays")  
            return Trajectory(  
                positions=self.positions.copy(),  
                times=self.times.copy(),  
                velocities=self.velocities.copy() if self.velocities is not None else None,  
                metadata=self.metadata.copy(),  
            )  
        
        return Trajectory(  
            positions=np.asarray(jnp.asarray(self.positions, dtype=jnp.float32)),  
            times=np.asarray(jnp.asarray(self.times, dtype=jnp.float32)),  
            velocities=np.asarray(jnp.asarray(self.velocities, dtype=jnp.float32)) if self.velocities is not None else None,  
            metadata=self.metadata.copy(),  
        )  

    def to_numpy(self) -> 'Trajectory':  
        """Ensure all data is NumPy arrays (for saving/compatibility)."""  
        return Trajectory(  
            positions=np.asarray(self.positions, dtype=np.float32),  
            times=np.asarray(self.times, dtype=np.float32),  
            velocities=np.asarray(self.velocities, dtype=np.float32) if self.velocities is not None else None,  
            metadata=self.metadata.copy(),  
        )  

    # ---------- Memory management ----------  

    def memory_usage_mb(self) -> float:  
        """Estimate memory usage in MB."""  
        pos_size = self.positions.nbytes  
        time_size = self.times.nbytes  
        vel_size = self.velocities.nbytes if self.velocities is not None else 0  
        total_bytes = pos_size + time_size + vel_size  
        return total_bytes / (1024 * 1024)  

    def subsample_time(self, step: int = 2) -> 'Trajectory':  
        """  
        Subsample trajectory in time to reduce memory usage.  
        
        Parameters  
        ----------  
        step : int  
            Time subsampling step  
            
        Returns  
        -------  
        Trajectory  
            Subsampled trajectory  
        """  
        return Trajectory(  
            positions=self.positions[::step],  
            times=self.times[::step],  
            velocities=self.velocities[::step] if self.velocities is not None else None,  
            metadata={**self.metadata, 'subsampled': step},  
        )  

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:  
        """  
        Get spatial bounds of all trajectories.  
        
        Returns  
        -------  
        bounds_min : np.ndarray  
            Minimum coordinates, shape (3,)  
        bounds_max : np.ndarray  
            Maximum coordinates, shape (3,)  
        """  
        bounds_min = np.min(self.positions.reshape(-1, 3), axis=0)  # (3,)  
        bounds_max = np.max(self.positions.reshape(-1, 3), axis=0)  # (3,)  
        return bounds_min, bounds_max  

    # ---------- Analysis methods ----------  

    def analyze_convergence(self) -> Dict[str, Any]:  
        """Analyze trajectory convergence properties."""  
        if self.T < 10:  
            return {'status': 'insufficient_data', 'num_timesteps': self.T}  
        
        # Compute displacement over time  
        initial_pos = self.positions[0]  # (N, 3)  
        displacements = np.linalg.norm(  
            self.positions - initial_pos[None, :, :],   
            axis=2  
        )  # (T, N)  
        
        # Statistics  
        final_displacement = displacements[-1]  # (N,)  
        max_displacement = np.max(displacements, axis=0)  # (N,)  
        
        return {  
            'status': 'success',  
            'num_timesteps': self.T,  
            'num_particles': self.N,  
            'duration': self.duration,  
            'mean_final_displacement': float(np.mean(final_displacement)),  
            'max_final_displacement': float(np.max(final_displacement)),  
            'mean_max_displacement': float(np.mean(max_displacement)),  
            'trajectory_bounds': self.get_bounds(),  
            'memory_usage_mb': self.memory_usage_mb(),  
        }  

    # ---------- Export and Visualization ----------  

    def export_to_vtk(self, filename: str, time_indices: Optional[List[int]] = None) -> None:  
        """  
        Export trajectory to VTK format for ParaView visualization.  
        
        Parameters  
        ----------  
        filename : str  
            Output VTK filename (with .vtu extension)  
        time_indices : list, optional  
            Specific time indices to export. If None, exports all.  
        """  
        try:  
            import vtk  
            from vtk.util.numpy_support import numpy_to_vtk  
            
            if time_indices is None:  
                time_indices = list(range(self.T))  
            
            # Create unstructured grid  
            ugrid = vtk.vtkUnstructuredGrid()  
            
            # Create points for all selected time steps  
            n_points = len(time_indices) * self.N  
            points = vtk.vtkPoints()  
            points.SetNumberOfPoints(n_points)  
            
            # Velocity and time data  
            velocity_data = np.zeros((n_points, 3), dtype=np.float32)  
            time_data = np.zeros(n_points, dtype=np.float32)  
            particle_ids = np.zeros(n_points, dtype=np.int32)  
            
            point_idx = 0  
            for t_idx in time_indices:  
                for p_idx in range(self.N):  
                    # Set point position  
                    pos = self.positions[t_idx, p_idx]  
                    points.SetPoint(point_idx, pos)  
                    
                    # Set velocity  
                    if self.velocities is not None:  
                        velocity_data[point_idx] = self.velocities[t_idx, p_idx]  
                    else:  
                        # Compute velocity from position differences  
                        if t_idx > 0:  
                            dt = self.times[t_idx] - self.times[t_idx-1]  
                            dpos = self.positions[t_idx, p_idx] - self.positions[t_idx-1, p_idx]  
                            velocity_data[point_idx] = dpos / dt  
                        else:  
                            velocity_data[point_idx] = [0.0, 0.0, 0.0]  
                    
                    # Set time and particle ID  
                    time_data[point_idx] = self.times[t_idx]  
                    particle_ids[point_idx] = p_idx  
                    
                    point_idx += 1  
            
            ugrid.SetPoints(points)  
            
            # Add vertex cells  
            for i in range(n_points):  
                ugrid.InsertNextCell(vtk.VTK_VERTEX, 1, [i])  
            
            # Add data arrays  
            vel_array = numpy_to_vtk(velocity_data, deep=True, array_type=vtk.VTK_FLOAT)  
            vel_array.SetName("Velocity")  
            vel_array.SetNumberOfComponents(3)  
            ugrid.GetPointData().SetVectors(vel_array)  
            
            time_array = numpy_to_vtk(time_data, deep=True, array_type=vtk.VTK_FLOAT)  
            time_array.SetName("Time")  
            ugrid.GetPointData().SetScalars(time_array)  
            
            id_array = numpy_to_vtk(particle_ids, deep=True, array_type=vtk.VTK_INT)  
            id_array.SetName("ParticleID")  
            ugrid.GetPointData().AddArray(id_array)  
            
            # Add speed  
            speed_data = np.linalg.norm(velocity_data, axis=1).astype(np.float32)  
            speed_array = numpy_to_vtk(speed_data, deep=True, array_type=vtk.VTK_FLOAT)  
            speed_array.SetName("Speed")  
            ugrid.GetPointData().AddArray(speed_array)  
            
            # Write file  
            writer = vtk.vtkXMLUnstructuredGridWriter()  
            writer.SetFileName(filename)  
            writer.SetInputData(ugrid)  
            writer.Write()  
            
            print(f"✅ Exported trajectory to {filename}")  
            
        except ImportError:  
            raise ImportError("VTK library required for VTK export. Install with: pip install vtk")  

    def plot_2d(self, plane: str = "xy", particle_indices: Optional[List[int]] = None,  
                show_arrows: bool = False, figsize: Tuple[int, int] = (10, 8)):  
        """  
        Create 2D trajectory plot.  
        
        Parameters  
        ----------  
        plane : str  
            Projection plane: 'xy', 'xz', or 'yz'  
        particle_indices : list, optional  
            Specific particles to plot. If None, plots all.  
        show_arrows : bool  
            Whether to show velocity arrows  
        figsize : tuple  
            Figure size  
        """  
        try:  
            import matplotlib.pyplot as plt  
            import matplotlib.colors as mcolors  
            
            if particle_indices is None:  
                particle_indices = list(range(min(self.N, 20)))  # Limit for visibility  
            
            fig, ax = plt.subplots(figsize=figsize)  
            
            # Extract coordinates based on plane  
            if plane == "xy":  
                x_data = self.positions[:, particle_indices, 0]  # (T, n_particles)  
                y_data = self.positions[:, particle_indices, 1]  
                x_label, y_label = "X", "Y"  
            elif plane == "xz":  
                x_data = self.positions[:, particle_indices, 0]  
                y_data = self.positions[:, particle_indices, 2]  
                x_label, y_label = "X", "Z"  
            elif plane == "yz":  
                x_data = self.positions[:, particle_indices, 1]  
                y_data = self.positions[:, particle_indices, 2]  
                x_label, y_label = "Y", "Z"  
            else:  
                raise ValueError(f"Unknown plane: {plane}")  
            
            # Color map for particles  
            colors = plt.cm.tab10(np.linspace(0, 1, len(particle_indices)))  
            
            # Plot trajectories  
            for i, p_idx in enumerate(particle_indices):  
                ax.plot(x_data[:, i], y_data[:, i],   
                       color=colors[i], alpha=0.7, linewidth=1.5,  
                       label=f"Particle {p_idx}")  
                
                # Mark start and end  
                ax.scatter(x_data[0, i], y_data[0, i],   
                          color=colors[i], marker='o', s=50, alpha=0.8)  
                ax.scatter(x_data[-1, i], y_data[-1, i],
                          color=colors[i], marker='s', s=50, alpha=0.8)
            
            # Add velocity arrows if requested
            if show_arrows and self.velocities is not None:
                # Subsample for arrows to avoid clutter
                arrow_step = max(1, self.T // 20)
                for i, p_idx in enumerate(particle_indices):
                    for t in range(0, self.T, arrow_step):
                        if plane == "xy":
                            dx, dy = self.velocities[t, p_idx, 0], self.velocities[t, p_idx, 1]
                        elif plane == "xz":
                            dx, dy = self.velocities[t, p_idx, 0], self.velocities[t, p_idx, 2]
                        elif plane == "yz":
                            dx, dy = self.velocities[t, p_idx, 1], self.velocities[t, p_idx, 2]
                        
                        ax.arrow(x_data[t, i], y_data[t, i], dx*0.1, dy*0.1,
                                head_width=0.02, head_length=0.02, 
                                fc=colors[i], ec=colors[i], alpha=0.5)
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f"Particle Trajectories - {plane.upper()} Plane")
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            if len(particle_indices) <= 10:
                ax.legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            raise ImportError("Matplotlib required for plotting. Install with: pip install matplotlib")

    def __repr__(self) -> str:
        return (f"Trajectory(T={self.T}, N={self.N}, "
                f"duration={self.duration:.3f}, "
                f"memory={self.memory_usage_mb():.1f}MB)")


# ---------- Factory functions ----------

def create_trajectory_from_positions(
    positions: np.ndarray,
    times: Optional[np.ndarray] = None,
    velocities: Optional[np.ndarray] = None,
    **metadata
) -> Trajectory:
    """
    Create trajectory from position data.
    
    Parameters
    ----------
    positions : np.ndarray
        Position data in various formats
    times : np.ndarray, optional
        Time points. If None, uses sequential integers.
    velocities : np.ndarray, optional
        Velocity data matching position format
    **metadata
        Additional trajectory metadata
        
    Returns
    -------
    Trajectory
        Standardized trajectory object
    """
    # Ensure proper trajectory shape
    positions = _ensure_trajectory_shape(positions)
    T, N, _ = positions.shape
    
    # Create time array if not provided
    if times is None:
        times = np.arange(T, dtype=np.float32)
    else:
        times = _ensure_float32(times)
        if times.shape != (T,):
            raise ValueError(f"Times shape {times.shape} doesn't match trajectory length {T}")
    
    # Process velocities if provided
    if velocities is not None:
        velocities = _ensure_trajectory_shape(velocities)
        if velocities.shape != (T, N, 3):
            raise ValueError(f"Velocities shape {velocities.shape} doesn't match positions {(T, N, 3)}")
    
    return Trajectory(
        positions=positions,
        times=times,
        velocities=velocities,
        metadata=metadata
    )


def create_single_particle_trajectory(
    positions: np.ndarray,
    times: Optional[np.ndarray] = None,
    velocities: Optional[np.ndarray] = None,
    **metadata
) -> Trajectory:
    """
    Create trajectory for a single particle.
    
    Parameters
    ----------
    positions : np.ndarray
        Position data, shape (T, 2) or (T, 3)
    times : np.ndarray, optional
        Time points, shape (T,)
    velocities : np.ndarray, optional
        Velocity data, shape (T, 2) or (T, 3)
    **metadata
        Additional metadata
        
    Returns
    -------
    Trajectory
        Single-particle trajectory
    """
    # Convert to multi-particle format
    positions = _ensure_float32(positions)
    if positions.ndim != 2:
        raise ValueError(f"Single particle positions must be 2D (T, D), got {positions.ndim}D")
    
    T, D = positions.shape
    if D == 2:
        # Add zero z-component
        z_zeros = np.zeros((T, 1), dtype=np.float32)
        positions = np.concatenate([positions, z_zeros], axis=1)
    elif D != 3:
        raise ValueError(f"Positions must have 2 or 3 spatial dimensions, got {D}")
    
    # Add particle dimension: (T, 3) -> (T, 1, 3)
    positions = positions[:, None, :]
    
    # Process velocities similarly if provided
    if velocities is not None:
        velocities = _ensure_float32(velocities)
        if velocities.ndim != 2 or velocities.shape[0] != T:
            raise ValueError(f"Velocities shape {velocities.shape} doesn't match positions time dimension {T}")
        
        if velocities.shape[1] == 2:
            z_zeros = np.zeros((T, 1), dtype=np.float32)
            velocities = np.concatenate([velocities, z_zeros], axis=1)
        
        velocities = velocities[:, None, :]  # (T, 1, 3)
    
    # Create time array if needed
    if times is None:
        times = np.arange(T, dtype=np.float32)
    else:
        times = _ensure_float32(times)
    
    return Trajectory(
        positions=positions,
        times=times,
        velocities=velocities,
        metadata={**metadata, 'particle_type': 'single'}
    )


def create_empty_trajectory(num_particles: int, num_timesteps: int) -> Trajectory:
    """
    Create empty trajectory with specified dimensions.
    
    Parameters
    ----------
    num_particles : int
        Number of particles
    num_timesteps : int
        Number of time steps
        
    Returns
    -------
    Trajectory
        Empty trajectory filled with zeros
    """
    positions = np.zeros((num_timesteps, num_particles, 3), dtype=np.float32)
    times = np.zeros(num_timesteps, dtype=np.float32)
    
    return Trajectory(
        positions=positions,
        times=times,
        metadata={'status': 'empty', 'initialized': True}
    )


def merge_trajectories(trajectories: List[Trajectory], axis: str = "time") -> Trajectory:
    """
    Merge multiple trajectories.
    
    Parameters
    ----------
    trajectories : list of Trajectory
        Trajectories to merge
    axis : str
        Merge axis: 'time' or 'particles'
        
    Returns
    -------
    Trajectory
        Merged trajectory
    """
    if not trajectories:
        raise ValueError("No trajectories provided for merging")
    
    if len(trajectories) == 1:
        return trajectories[0]
    
    if axis == "time":
        # Concatenate along time dimension
        # All trajectories must have same number of particles
        N_particles = [traj.N for traj in trajectories]
        if len(set(N_particles)) != 1:
            raise ValueError(f"All trajectories must have same number of particles for time merge: {N_particles}")
        
        positions = np.concatenate([traj.positions for traj in trajectories], axis=0)
        times = np.concatenate([traj.times for traj in trajectories], axis=0)
        
        # Merge velocities if all have them
        velocities = None
        if all(traj.velocities is not None for traj in trajectories):
            velocities = np.concatenate([traj.velocities for traj in trajectories], axis=0)
        
        # Merge metadata
        metadata = trajectories[0].metadata.copy()
        metadata['merged_axis'] = 'time'
        metadata['merged_count'] = len(trajectories)
        
    elif axis == "particles":
        # Concatenate along particle dimension
        # All trajectories must have same time points
        T_times = [traj.T for traj in trajectories]
        if len(set(T_times)) != 1:
            raise ValueError(f"All trajectories must have same number of timesteps for particle merge: {T_times}")
        
        # Check time consistency
        ref_times = trajectories[0].times
        for i, traj in enumerate(trajectories[1:], 1):
            if not np.allclose(traj.times, ref_times, atol=1e-6):
                warnings.warn(f"Trajectory {i} has different time points - using first trajectory's times")
        
        positions = np.concatenate([traj.positions for traj in trajectories], axis=1)
        times = ref_times.copy()
        
        # Merge velocities if all have them
        velocities = None
        if all(traj.velocities is not None for traj in trajectories):
            velocities = np.concatenate([traj.velocities for traj in trajectories], axis=1)
        
        # Merge metadata
        metadata = trajectories[0].metadata.copy()
        metadata['merged_axis'] = 'particles'
        metadata['merged_count'] = len(trajectories)
        
    else:
        raise ValueError(f"Unknown merge axis: {axis}. Use 'time' or 'particles'.")
    
    return Trajectory(
        positions=positions,
        times=times,
        velocities=velocities,
        metadata=metadata
    )


# ---------- Analysis utilities ----------

def compute_trajectory_statistics(trajectory: Trajectory) -> Dict[str, Any]:
    """
    Compute comprehensive trajectory statistics.
    
    Parameters
    ----------
    trajectory : Trajectory
        Trajectory to analyze
        
    Returns
    -------
    dict
        Trajectory statistics
    """
    stats = {
        'basic': {
            'num_particles': trajectory.N,
            'num_timesteps': trajectory.T,
            'duration': trajectory.duration,
            'dt_mean': trajectory.dt_mean,
            'memory_mb': trajectory.memory_usage_mb()
        }
    }
    
    if trajectory.T > 1:
        # Displacement statistics
        displacements = trajectory.compute_displacement()  # (N,)
        path_lengths = trajectory.compute_path_length()    # (N,)
        
        stats['displacement'] = {
            'mean': float(np.mean(displacements)),
            'std': float(np.std(displacements)),
            'min': float(np.min(displacements)),
            'max': float(np.max(displacements))
        }
        
        stats['path_length'] = {
            'mean': float(np.mean(path_lengths)),
            'std': float(np.std(path_lengths)),
            'min': float(np.min(path_lengths)),
            'max': float(np.max(path_lengths))
        }
        
        # Speed statistics if available
        if trajectory.T > 1:
            speeds = trajectory.compute_speeds()  # (T-1, N)
            if speeds.size > 0:
                stats['speed'] = {
                    'mean': float(np.mean(speeds)),
                    'std': float(np.std(speeds)),
                    'min': float(np.min(speeds)),
                    'max': float(np.max(speeds))
                }
        
        # Spatial bounds
        bounds_min, bounds_max = trajectory.get_bounds()
        stats['bounds'] = {
            'min': bounds_min.tolist(),
            'max': bounds_max.tolist(),
            'extent': (bounds_max - bounds_min).tolist()
        }
    
    return stats


def validate_trajectory_data(trajectory: Trajectory, strict: bool = False) -> Dict[str, Any]:
    """
    Validate trajectory data consistency and quality.
    
    Parameters
    ----------
    trajectory : Trajectory
        Trajectory to validate
    strict : bool
        Whether to perform strict validation
        
    Returns
    -------
    dict
        Validation results
    """
    results = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Basic shape validation
    if trajectory.positions.shape != (trajectory.T, trajectory.N, 3):
        results['errors'].append(f"Position shape {trajectory.positions.shape} doesn't match (T,N,3) = ({trajectory.T},{trajectory.N},3)")
        results['valid'] = False
    
    if trajectory.times.shape != (trajectory.T,):
        results['errors'].append(f"Times shape {trajectory.times.shape} doesn't match (T,) = ({trajectory.T},)")
        results['valid'] = False
    
    # Data type validation
    if trajectory.positions.dtype != np.float32:
        if strict:
            results['errors'].append(f"Positions dtype {trajectory.positions.dtype} is not float32")
            results['valid'] = False
        else:
            results['warnings'].append(f"Positions dtype {trajectory.positions.dtype} is not optimal float32")
    
    # Check for NaN/infinite values
    if not np.all(np.isfinite(trajectory.positions)):
        results['errors'].append("Positions contain non-finite values (NaN/inf)")
        results['valid'] = False
    
    if not np.all(np.isfinite(trajectory.times)):
        results['errors'].append("Times contain non-finite values (NaN/inf)")
        results['valid'] = False
    
    # Time monotonicity
    if trajectory.T > 1:
        if not np.all(np.diff(trajectory.times) >= 0):
            results['errors'].append("Time array is not monotonically non-decreasing")
            results['valid'] = False
        
        # Check for very small time steps
        dt_min = np.min(np.diff(trajectory.times))
        if dt_min < 1e-10:
            results['warnings'].append(f"Very small minimum time step: {dt_min}")
    
    # Velocity validation if present
    if trajectory.velocities is not None:
        if trajectory.velocities.shape != (trajectory.T, trajectory.N, 3):
            results['errors'].append(f"Velocities shape {trajectory.velocities.shape} doesn't match positions")
            results['valid'] = False
        
        if not np.all(np.isfinite(trajectory.velocities)):
            results['errors'].append("Velocities contain non-finite values")
            results['valid'] = False
    
    # Memory usage warning
    memory_mb = trajectory.memory_usage_mb()
    if memory_mb > 1000:  # 1 GB
        results['warnings'].append(f"Large memory usage: {memory_mb:.1f} MB")
    
    return results