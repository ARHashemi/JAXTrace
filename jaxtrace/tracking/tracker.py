# jaxtrace/tracking/tracker.py  
"""  
High-performance particle tracker with memory management and JAX optimization.  

Provides adaptive batch processing, automatic memory management, and  
JIT-compiled integration with consistent float32 data types.  
"""  

from __future__ import annotations  
from dataclasses import dataclass, field  
from typing import Callable, Optional, List, Dict, Any, Tuple, Union  
import numpy as np  
import warnings  
import time  

# Import JAX utilities with fallback  
from ..utils.jax_utils import JAX_AVAILABLE  

if JAX_AVAILABLE:  
    try:  
        import jax  
        import jax.numpy as jnp  
        from jax import jit, vmap  
    except Exception:  
        JAX_AVAILABLE = False  

if not JAX_AVAILABLE:  
    import numpy as jnp  # type: ignore  
    # Mock JAX functions  
    class MockJit:  
        def __call__(self, func):  
            return func  
    class MockVmap:  
        def __call__(self, func, **kwargs):  
            def vectorized(x):  
                return np.array([func(xi) for xi in x])  
            return vectorized  
    jit = MockJit()  
    vmap = MockVmap()  

from .particles import Trajectory, create_trajectory_from_positions  


def _ensure_float32(data: np.ndarray) -> np.ndarray:  
    """Convert data to float32 for consistency."""  
    return np.asarray(data, dtype=np.float32)  


def _ensure_positions_shape(positions: np.ndarray) -> np.ndarray:  
    """Ensure positions have shape (N, 3) with float32 dtype."""  
    pos = _ensure_float32(positions)  
    
    if pos.ndim == 1:  
        pos = pos.reshape(1, -1)  
    
    if pos.ndim != 2:  
        raise ValueError(f"Positions must be 2D array, got shape {pos.shape}")  
    
    if pos.shape[1] == 2:  
        # Convert 2D to 3D  
        z_zeros = np.zeros((pos.shape[0], 1), dtype=np.float32)  
        pos = np.concatenate([pos, z_zeros], axis=1)  
    elif pos.shape[1] != 3:  
        raise ValueError(f"Positions must have 2 or 3 columns, got {pos.shape[1]}")  
    
    return pos  


def _estimate_memory_usage(n_particles: int, n_timesteps: int, has_velocities: bool = False) -> float:  
    """Estimate memory usage in GB for trajectory storage."""  
    # Positions: (T, N, 3) * 4 bytes (float32)  
    pos_size = n_timesteps * n_particles * 3 * 4  
    # Times: (T,) * 4 bytes  
    time_size = n_timesteps * 4  
    # Optional velocities  
    vel_size = pos_size if has_velocities else 0  
    # Integration workspace (temporary arrays)  
    workspace_size = n_particles * 3 * 4 * 4  # ~4 integration stages  
    
    total_bytes = pos_size + time_size + vel_size + workspace_size  
    return total_bytes / (1024**3)  # Convert to GB  


@dataclass  
class TrackerOptions:  
    """  
    Configuration options for particle tracking.  
    
    All memory and performance settings for consistent batch processing  
    with automatic adaptation to available system resources.  
    """  
    # Memory management  
    max_memory_gb: float = 8.0          # Maximum memory usage  
    batch_size: Optional[int] = None    # Auto-determined if None  
    oom_recovery: bool = True           # Automatic batch size reduction  
    
    # Performance settings  
    use_jax_jit: bool = True            # Use JAX JIT compilation  
    max_batch_size: int = 100_000       # Maximum particles per batch  
    min_batch_size: int = 100           # Minimum particles per batch  
    
    # Recording options  
    record_velocities: bool = False     # Store velocity data  
    recording_interval: int = 1         # Time step interval for recording  
    
    # Progress monitoring  
    progress_callback: Optional[Callable[[float], None]] = None  
    
    # JAX compilation options  
    static_compilation: bool = True     # Pre-compile for fixed batch sizes  
    
    # Advanced options  
    adaptive_dt: bool = False           # Adaptive time stepping  
    error_tolerance: float = 1e-6       # For adaptive schemes  

    def estimate_batch_size(self, n_particles: int, n_timesteps: int) -> int:  
        """Estimate optimal batch size based on memory constraints."""  
        if self.batch_size is not None:  
            return min(self.batch_size, self.max_batch_size)  
        
        # Start with all particles and reduce if necessary  
        batch_size = min(n_particles, self.max_batch_size)  
        
        while batch_size >= self.min_batch_size:  
            mem_usage = _estimate_memory_usage(batch_size, n_timesteps, self.record_velocities)  
            if mem_usage <= self.max_memory_gb:  
                break  
            batch_size = max(batch_size // 2, self.min_batch_size)  
        
        return max(batch_size, self.min_batch_size)  


@dataclass  
class ParticleTracker:  
    """  
    High-performance particle tracker with adaptive memory management.  
    
    Features:  
    - Automatic batch size selection based on memory constraints  
    - JAX JIT compilation for maximum performance  
    - Consistent float32 data types throughout  
    - (T,N,3) trajectory storage format  
    - Out-of-memory recovery with automatic batch size reduction  
    
    Attributes  
    ----------  
    integrator : Callable  
        Integration function (euler_step, rk2_step, rk4_step)  
    field_fn : Callable  
        Velocity field function: field_fn(positions, time) -> velocities  
    boundary_fn : Callable  
        Boundary condition function: boundary_fn(positions) -> positions  
    options : TrackerOptions  
        Tracking configuration options  
    """  
    integrator: Callable                # Integration scheme  
    field_fn: Callable                  # Velocity field  
    boundary_fn: Callable               # Boundary conditions  
    options: TrackerOptions = field(default_factory=TrackerOptions)  
    
    def __post_init__(self):  
        # Validate functions  
        if not callable(self.integrator):  
            raise ValueError("integrator must be callable")  
        if not callable(self.field_fn):  
            raise ValueError("field_fn must be callable")  
        if not callable(self.boundary_fn):  
            raise ValueError("boundary_fn must be callable")  
        
        # Prepare JIT-compiled functions if JAX is available  
        self._compiled_step = None  
        self._setup_jax_compilation()  
    
    def _setup_jax_compilation(self):  
        """Setup JAX compilation for integration steps."""  
        if not JAX_AVAILABLE or not self.options.use_jax_jit:  
            return  
        
        try:  
            # Create a compiled step function for fixed batch sizes  
            if self.options.static_compilation:  
                @jit  
                def compiled_step(x_batch, t, dt):  
                    # Single integration step with boundary conditions  
                    x_new = self.integrator(x_batch, t, dt, self.field_fn)  
                    x_bounded = self.boundary_fn(x_new)  
                    return x_bounded  
                
                self._compiled_step = compiled_step  
        except Exception as e:  
            warnings.warn(f"JAX compilation setup failed: {e}")  
            self._compiled_step = None  

    def _integration_step(self, x_batch: jnp.ndarray, t: float, dt: float) -> jnp.ndarray:  
        """Single integration step with optimal performance."""  
        # Convert to consistent types  
        x_batch = jnp.asarray(x_batch, dtype=jnp.float32)  
        
        if self._compiled_step is not None:  
            try:  
                return self._compiled_step(x_batch, t, dt)  
            except Exception:  
                # Fallback to non-compiled version  
                pass  
        
        # Non-compiled version  
        x_new = self.integrator(x_batch, t, dt, self.field_fn)  
        x_bounded = self.boundary_fn(x_new)  
        return x_bounded  

    def track_particles(  
        self,  
        initial_positions: np.ndarray,  
        time_span: Tuple[float, float],  
        n_timesteps: int,
        dt: Optional[float] = None,  
        **kwargs  
    ) -> Trajectory:  
        """  
        Track particles through velocity field.  
        
        Parameters  
        ----------  
        initial_positions : np.ndarray  
            Initial particle positions, shape (N, 2) or (N, 3)  
        time_span : Tuple[float, float]  
            (t_start, t_end) integration time span  
        n_timesteps : int  
            Number of time steps  
        
        Returns  
        -------  
        Trajectory  
            Particle trajectories with shape (T, N, 3)  
        """  
        # Ensure consistent input format  
        x0 = _ensure_positions_shape(initial_positions)  # (N, 3), float32  
        n_particles = x0.shape[0]  
        
        # Setup time grid  
        t_start, t_end = time_span  
        times = np.linspace(t_start, t_end, n_timesteps, dtype=np.float32)  
        if dt is None:
            dt = float(times[1] - times[0]) if n_timesteps > 1 else 0.0  
        else:
            dt = float(dt)
    
        
        # Estimate batch size  
        batch_size = self.options.estimate_batch_size(n_particles, n_timesteps)  
        
        if batch_size >= n_particles:  
            # Single batch - most efficient  
            return self._track_single_batch(x0, times, dt)  
        else:  
            # Multiple batches with memory management  
            return self._track_multi_batch(x0, times, dt, batch_size)  

    def _track_single_batch(  
        self,   
        x0: np.ndarray,   
        times: np.ndarray,   
        dt: float  
    ) -> Trajectory:  
        """Track particles in a single batch (most efficient path)."""  
        n_particles = x0.shape[0]  
        n_timesteps = len(times)  
        
        # Pre-allocate trajectory storage  
        positions = np.zeros((n_timesteps, n_particles, 3), dtype=np.float32)  
        velocities = None  
        if self.options.record_velocities:  
            velocities = np.zeros((n_timesteps, n_particles, 3), dtype=np.float32)  
        
        # Initialize  
        x_current = x0.copy()  
        positions[0] = x_current  
        
        if self.options.record_velocities:  
            v_initial = self.field_fn(x_current, times[0])  
            v_initial = jnp.asarray(v_initial, dtype=jnp.float32)  
            velocities[0] = np.asarray(v_initial)  
        
        # Integration loop  
        for i in range(1, n_timesteps):  
            t_current = times[i-1]  
            
            # Integration step  
            x_current = self._integration_step(x_current, t_current, dt)  
            
            # Record trajectory  
            if i % self.options.recording_interval == 0:  
                positions[i] = np.asarray(x_current, dtype=np.float32)  
                
                if self.options.record_velocities:  
                    v_current = self.field_fn(x_current, times[i])  
                    v_current = jnp.asarray(v_current, dtype=jnp.float32)  
                    velocities[i] = np.asarray(v_current)  
            
            # Progress callback  
            if self.options.progress_callback is not None:  
                progress = i / (n_timesteps - 1)  
                self.options.progress_callback(progress)  
        
        return Trajectory(  
            positions=positions,  
            times=times,  
            velocities=velocities,  
            metadata={  
                'integrator': str(self.integrator),  
                'batch_processing': 'single_batch',  
                'n_particles': n_particles,  
                'n_timesteps': n_timesteps,  
                'dt': dt,  
                'jax_compiled': self._compiled_step is not None,  
            }  
        )  

    def _track_multi_batch(  
        self,   
        x0: np.ndarray,   
        times: np.ndarray,   
        dt: float,  
        batch_size: int  
    ) -> Trajectory:  
        """Track particles using multiple batches for memory management."""  
        n_particles = x0.shape[0]  
        n_timesteps = len(times)  
        
        # Pre-allocate full trajectory storage  
        positions = np.zeros((n_timesteps, n_particles, 3), dtype=np.float32)  
        velocities = None  
        if self.options.record_velocities:  
            velocities = np.zeros((n_timesteps, n_particles, 3), dtype=np.float32)  
        
        # Initialize  
        positions[0] = x0  
        
        if self.options.record_velocities:  
            v_initial = self.field_fn(x0, times[0])  
            velocities[0] = np.asarray(v_initial, dtype=np.float32)  
        
        # Process in batches  
        n_batches = (n_particles + batch_size - 1) // batch_size  
        
        try:  
            for batch_idx in range(n_batches):  
                start_idx = batch_idx * batch_size  
                end_idx = min(start_idx + batch_size, n_particles)  
                
                # Extract batch  
                x_batch = positions[0, start_idx:end_idx].copy()  # (batch_n, 3)  
                
                # Track batch  
                batch_trajectory = self._track_single_batch(  
                    x_batch, times, dt  
                )  
                
                # Store results  
                positions[:, start_idx:end_idx] = batch_trajectory.positions  
                if self.options.record_velocities and batch_trajectory.velocities is not None:  
                    velocities[:, start_idx:end_idx] = batch_trajectory.velocities  
                
                # Progress callback  
                if self.options.progress_callback is not None:  
                    batch_progress = (batch_idx + 1) / n_batches  
                    self.options.progress_callback(batch_progress)  
        
        except MemoryError as e:  
            if self.options.oom_recovery and batch_size > self.options.min_batch_size:  
                # Reduce batch size and retry  
                new_batch_size = max(batch_size // 2, self.options.min_batch_size)  
                warnings.warn(f"OOM encountered, reducing batch size from {batch_size} to {new_batch_size}")  
                return self._track_multi_batch(x0, times, dt, new_batch_size)  
            else:  
                raise e  
        
        return Trajectory(  
            positions=positions,  
            times=times,  
            velocities=velocities,  
            metadata={  
                'integrator': str(self.integrator),  
                'batch_processing': 'multi_batch',  
                'batch_size': batch_size,  
                'n_batches': n_batches,  
                'n_particles': n_particles,  
                'n_timesteps': n_timesteps,  
                'dt': dt,  
                'jax_compiled': self._compiled_step is not None,  
            }  
        )  

    def estimate_runtime(
        self,
        n_particles: int,
        n_timesteps: int,
        calibration_particles: int = 1000,
        calibration_steps: int = 10
    ) -> Dict[str, Any]:
        """
        Estimate runtime for particle tracking job.
        
        Parameters
        ----------
        n_particles : int
            Number of particles to track
        n_timesteps : int
            Number of time steps
        calibration_particles : int
            Number of particles for timing calibration
        calibration_steps : int
            Number of steps for timing calibration
            
        Returns
        -------
        dict
            Runtime estimation results
        """
        # Create test positions
        test_positions = np.random.uniform(-1, 1, size=(calibration_particles, 3)).astype(np.float32)
        test_times = np.linspace(0, 1, calibration_steps, dtype=np.float32)
        dt = float(test_times[1] - test_times[0]) if calibration_steps > 1 else 0.01
        
        # Calibration run
        start_time = time.time()
        try:
            # Track small batch for timing
            calibration_result = self._track_single_batch(test_positions, test_times, dt)
            calibration_time = time.time() - start_time
            
            # Extrapolate to full job
            time_per_particle_step = calibration_time / (calibration_particles * calibration_steps)
            total_operations = n_particles * n_timesteps
            estimated_time = total_operations * time_per_particle_step
            
            # Memory estimation
            batch_size = self.options.estimate_batch_size(n_particles, n_timesteps)
            memory_gb = _estimate_memory_usage(batch_size, n_timesteps, self.options.record_velocities)
            
            # Batch analysis
            if batch_size >= n_particles:
                processing_mode = "single_batch"
                n_batches = 1
            else:
                processing_mode = "multi_batch"
                n_batches = (n_particles + batch_size - 1) // batch_size
            
            return {
                'success': True,
                'estimated_runtime_seconds': estimated_time,
                'estimated_runtime_minutes': estimated_time / 60,
                'estimated_runtime_hours': estimated_time / 3600,
                'processing_mode': processing_mode,
                'batch_size': batch_size,
                'n_batches': n_batches,
                'estimated_memory_gb': memory_gb,
                'time_per_particle_step': time_per_particle_step,
                'calibration_time': calibration_time,
                'calibration_particles': calibration_particles,
                'jax_available': JAX_AVAILABLE,
                'jax_compiled': self._compiled_step is not None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }

    def benchmark_performance(
        self,
        test_sizes: List[int] = [100, 500, 1000, 5000],
        n_timesteps: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark tracking performance across different problem sizes.
        
        Parameters
        ----------
        test_sizes : list
            List of particle counts to test
        n_timesteps : int
            Number of time steps for each test
            
        Returns
        -------
        dict
            Benchmark results
        """
        results = {
            'test_sizes': test_sizes,
            'n_timesteps': n_timesteps,
            'timing_results': {},
            'memory_usage': {},
            'throughput': {},
            'scalability': {}
        }
        
        for n_particles in test_sizes:
            print(f"Benchmarking {n_particles} particles...")
            
            # Create test data
            x0 = np.random.uniform(-1, 1, size=(n_particles, 3)).astype(np.float32)
            times = np.linspace(0, 1, n_timesteps, dtype=np.float32)
            dt = float(times[1] - times[0])
            
            # Time the tracking
            start_time = time.time()
            try:
                batch_size = self.options.estimate_batch_size(n_particles, n_timesteps)
                
                if batch_size >= n_particles:
                    trajectory = self._track_single_batch(x0, times, dt)
                else:
                    trajectory = self._track_multi_batch(x0, times, dt, batch_size)
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                # Calculate metrics
                total_operations = n_particles * n_timesteps
                throughput = total_operations / elapsed_time  # operations per second
                memory_mb = trajectory.memory_usage_mb()
                
                results['timing_results'][n_particles] = elapsed_time
                results['memory_usage'][n_particles] = memory_mb
                results['throughput'][n_particles] = throughput
                
                print(f"  Time: {elapsed_time:.3f}s, Throughput: {throughput:.0f} ops/s, Memory: {memory_mb:.1f}MB")
                
            except Exception as e:
                print(f"  Failed: {e}")
                results['timing_results'][n_particles] = None
                results['memory_usage'][n_particles] = None
                results['throughput'][n_particles] = None
        
        # Analyze scalability
        valid_sizes = [s for s in test_sizes if results['timing_results'][s] is not None]
        if len(valid_sizes) >= 2:
            # Compute scaling exponent
            import math
            size_ratios = [valid_sizes[i+1] / valid_sizes[i] for i in range(len(valid_sizes)-1)]
            time_ratios = [results['timing_results'][valid_sizes[i+1]] / results['timing_results'][valid_sizes[i]] 
                          for i in range(len(valid_sizes)-1)]
            
            if all(r > 0 for r in size_ratios + time_ratios):
                log_size_ratios = [math.log(r) for r in size_ratios]
                log_time_ratios = [math.log(r) for r in time_ratios]
                
                # Simple linear regression for scaling exponent
                n = len(log_size_ratios)
                if n > 0:
                    mean_x = sum(log_size_ratios) / n
                    mean_y = sum(log_time_ratios) / n
                    
                    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_size_ratios, log_time_ratios))
                    denominator = sum((x - mean_x)**2 for x in log_size_ratios)
                    
                    if denominator > 0:
                        scaling_exponent = numerator / denominator
                        results['scalability']['scaling_exponent'] = scaling_exponent
                        results['scalability']['complexity'] = f"O(N^{scaling_exponent:.2f})"
        
        return results

    def analyze_field_sampling(
        self,
        test_positions: np.ndarray,
        time_points: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze field sampling performance and characteristics.
        
        Parameters
        ----------
        test_positions : np.ndarray
            Test positions, shape (N, 3)
        time_points : np.ndarray
            Test time points, shape (T,)
            
        Returns
        -------
        dict
            Field analysis results
        """
        test_positions = _ensure_positions_shape(test_positions)
        time_points = _ensure_float32(time_points)
        
        results = {
            'num_positions': test_positions.shape[0],
            'num_times': len(time_points),
            'sampling_stats': {},
            'performance': {},
            'field_properties': {}
        }
        
        # Sample field at test points
        velocities_by_time = []
        sampling_times = []
        
        for t in time_points:
            start_time = time.time()
            try:
                velocities = self.field_fn(test_positions, t)
                velocities = np.asarray(velocities, dtype=np.float32)
                sampling_time = time.time() - start_time
                
                velocities_by_time.append(velocities)
                sampling_times.append(sampling_time)
            except Exception as e:
                results['field_properties']['sampling_error'] = str(e)
                return results
        
        # Performance analysis
        mean_sampling_time = np.mean(sampling_times)
        sampling_rate = test_positions.shape[0] / mean_sampling_time  # positions per second
        
        results['performance'] = {
            'mean_sampling_time': mean_sampling_time,
            'sampling_rate_per_second': sampling_rate,
            'total_samples': len(test_positions) * len(time_points)
        }
        
        # Field statistics
        if velocities_by_time:
            all_velocities = np.concatenate(velocities_by_time, axis=0)  # (N*T, 3)
            
            # Magnitude statistics
            magnitudes = np.linalg.norm(all_velocities, axis=1)
            
            results['field_properties'] = {
                'velocity_magnitude': {
                    'mean': float(np.mean(magnitudes)),
                    'std': float(np.std(magnitudes)),
                    'min': float(np.min(magnitudes)),
                    'max': float(np.max(magnitudes))
                },
                'velocity_components': {
                    'x': {'mean': float(np.mean(all_velocities[:, 0])), 'std': float(np.std(all_velocities[:, 0]))},
                    'y': {'mean': float(np.mean(all_velocities[:, 1])), 'std': float(np.std(all_velocities[:, 1]))},
                    'z': {'mean': float(np.mean(all_velocities[:, 2])), 'std': float(np.std(all_velocities[:, 2]))}
                },
                'finite_values': int(np.sum(np.isfinite(all_velocities).all(axis=1))),
                'infinite_values': int(np.sum(np.isinf(all_velocities).any(axis=1))),
                'nan_values': int(np.sum(np.isnan(all_velocities).any(axis=1)))
            }
        
        return results

    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate tracker configuration and dependencies.
        
        Returns
        -------
        dict
            Validation results
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'configuration': {}
        }
        
        # Test functions with dummy data
        test_pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        test_time = 0.0
        test_dt = 0.01
        
        # Test field function
        try:
            test_vel = self.field_fn(test_pos, test_time)
            test_vel = np.asarray(test_vel, dtype=np.float32)
            if test_vel.shape != (1, 3):
                results['errors'].append(f"field_fn returned wrong shape: {test_vel.shape}, expected (1, 3)")
                results['valid'] = False
            if not np.all(np.isfinite(test_vel)):
                results['warnings'].append("field_fn returned non-finite values")
        except Exception as e:
            results['errors'].append(f"field_fn test failed: {str(e)}")
            results['valid'] = False
        
        # Test boundary function
        try:
            test_bounded = self.boundary_fn(test_pos)
            test_bounded = np.asarray(test_bounded, dtype=np.float32)
            if test_bounded.shape != test_pos.shape:
                results['errors'].append(f"boundary_fn changed shape: {test_bounded.shape} != {test_pos.shape}")
                results['valid'] = False
        except Exception as e:
            results['errors'].append(f"boundary_fn test failed: {str(e)}")
            results['valid'] = False
        
        # Test integrator
        try:
            test_integrated = self.integrator(test_pos, test_time, test_dt, self.field_fn)
            test_integrated = np.asarray(test_integrated, dtype=np.float32)
            if test_integrated.shape != test_pos.shape:
                results['errors'].append(f"integrator changed shape: {test_integrated.shape} != {test_pos.shape}")
                results['valid'] = False
        except Exception as e:
            results['errors'].append(f"integrator test failed: {str(e)}")
            results['valid'] = False
        
        # JAX compilation status
        if self.options.use_jax_jit and not JAX_AVAILABLE:
            results['warnings'].append("JAX JIT requested but JAX not available")
        
        if self._compiled_step is None and self.options.use_jax_jit and JAX_AVAILABLE:
            results['warnings'].append("JAX compilation failed, using fallback")
        
        # Configuration summary
        results['configuration'] = {
            'jax_available': JAX_AVAILABLE,
            'jax_compilation_enabled': self.options.use_jax_jit,
            'jax_compiled': self._compiled_step is not None,
            'max_memory_gb': self.options.max_memory_gb,
            'max_batch_size': self.options.max_batch_size,
            'record_velocities': self.options.record_velocities
        }
        
        return results

    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'available': True,
                'rss_mb': memory_info.rss / (1024**2),
                'vms_mb': memory_info.vms / (1024**2),
                'percent': process.memory_percent(),
                'system_available_mb': psutil.virtual_memory().available / (1024**2),
                'system_total_mb': psutil.virtual_memory().total / (1024**2)
            }
        except ImportError:
            return {
                'available': False,
                'message': 'psutil not available for memory monitoring'
            }


# ---------- Factory functions ----------

def create_tracker(
    integrator_name: str,
    field,
    boundary_condition,
    **options
) -> ParticleTracker:
    """
    Factory function to create particle tracker with named integrator.
    
    Parameters
    ----------
    integrator_name : str
        Name of integrator: 'euler', 'rk2', 'rk4'
    field : BaseField or callable
        Velocity field
    boundary_condition : BoundaryCondition or callable
        Boundary condition function
    **options
        Additional options for TrackerOptions
        
    Returns
    -------
    ParticleTracker
        Configured particle tracker
    """
    # Import integration functions
    from ..integrators import euler_step, rk2_step, rk4_step
    
    # Select integrator
    integrators = {
        'euler': euler_step,
        'rk2': rk2_step, 
        'rk4': rk4_step
    }
    
    if integrator_name not in integrators:
        raise ValueError(f"Unknown integrator: {integrator_name}. Available: {list(integrators.keys())}")
    
    integrator = integrators[integrator_name]
    
    # Handle field interface
    if hasattr(field, 'sample_at_positions'):
        # BaseField interface
        field_fn = lambda positions, t: field.sample_at_positions(positions, t)
    elif hasattr(field, 'sample'):
        # Alternative field interface
        field_fn = lambda positions, t: field.sample(positions)
    elif callable(field):
        # Direct callable
        field_fn = field
    else:
        raise TypeError("field must be BaseField instance or callable")
    
    # Handle boundary condition
    if callable(boundary_condition):
        boundary_fn = boundary_condition
    else:
        raise TypeError("boundary_condition must be callable")
    
    # Create tracker options
    tracker_options = TrackerOptions(**options)
    
    return ParticleTracker(
        integrator=integrator,
        field_fn=field_fn,
        boundary_fn=boundary_fn,
        options=tracker_options
    )


# ---------- Utility functions ----------

def track_particles_simple(
    initial_positions: np.ndarray,
    velocity_field,
    time_span: Tuple[float, float],
    n_timesteps: int,
    integrator: str = 'rk4',
    boundary_condition = None,
    dt: Optional[float] = None,
    **tracker_options
) -> Trajectory:
    """
    Simple interface for particle tracking.
    
    Parameters
    ----------
    initial_positions : np.ndarray
        Initial positions, shape (N, 2) or (N, 3)
    velocity_field : BaseField or callable
        Velocity field
    time_span : tuple
        (t_start, t_end) integration time
    n_timesteps : int
        Number of time steps
    integrator : str
        Integration method: 'euler', 'rk2', 'rk4'
    boundary_condition : callable, optional
        Boundary condition function. If None, uses no boundaries.
    **tracker_options
        Additional options for tracker
        
    Returns
    -------
    Trajectory
        Particle trajectories
    """
    # Default boundary condition
    if boundary_condition is None:
        from .boundary import no_boundary
        boundary_condition = no_boundary()
    
    # Create and run tracker
    tracker = create_tracker(
        integrator_name=integrator,
        field=velocity_field,
        boundary_condition=boundary_condition,
        **tracker_options
    )
    
    return tracker.track_particles(initial_positions, time_span, n_timesteps, dt=dt)


def compare_integrators(
    initial_positions: np.ndarray,
    velocity_field,
    time_span: Tuple[float, float],
    n_timesteps: int,
    integrators: List[str] = ['euler', 'rk2', 'rk4'],
    dt: Optional[float] = None,
    **options
) -> Dict[str, Any]:
    """
    Compare different integration methods on the same problem.
    
    Parameters
    ----------
    initial_positions : np.ndarray
        Initial positions
    velocity_field : BaseField or callable
        Velocity field  
    time_span : tuple
        Integration time span
    n_timesteps : int
        Number of time steps
    integrators : list
        List of integrator names to compare
    **options
        Tracker options
        
    Returns
    -------
    dict
        Comparison results with trajectories and timing
    """
    from .boundary import no_boundary
    
    results = {
        'integrators': integrators,
        'trajectories': {},
        'timing': {},
        'comparison': {}
    }
    
    # Default boundary
    boundary_fn = no_boundary()
    
    for integrator_name in integrators:
        print(f"Testing {integrator_name} integrator...")
        
        try:
            # Create tracker
            tracker = create_tracker(
                integrator_name=integrator_name,
                field=velocity_field,
                boundary_condition=boundary_fn,
                **options
            )
            
            # Time the integration
            start_time = time.time()
            trajectory = tracker.track_particles(initial_positions, time_span, n_timesteps, dt=dt)
            end_time = time.time()
            
            # Store results
            results['trajectories'][integrator_name] = trajectory
            results['timing'][integrator_name] = end_time - start_time
            
            print(f"  Completed in {end_time - start_time:.3f}s")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results['trajectories'][integrator_name] = None
            results['timing'][integrator_name] = None
    
    # Compare final positions if we have results
    valid_results = {k: v for k, v in results['trajectories'].items() if v is not None}
    
    if len(valid_results) >= 2:
        integrator_names = list(valid_results.keys())
        ref_name = integrator_names[0]
        ref_final = valid_results[ref_name].positions[-1]  # (N, 3)
        
        for integrator_name in integrator_names[1:]:
            test_final = valid_results[integrator_name].positions[-1]
            
            # Compute differences
            differences = np.linalg.norm(test_final - ref_final, axis=1)  # (N,)
            
            results['comparison'][f'{integrator_name}_vs_{ref_name}'] = {
                'mean_difference': float(np.mean(differences)),
                'max_difference': float(np.max(differences)),
                'std_difference': float(np.std(differences))
            }
    
    return results


# Export main classes and functions
__all__ = [
    'ParticleTracker',
    'TrackerOptions', 
    'create_tracker',
    'track_particles_simple',
    'compare_integrators'
]