import os
# Disable JAX memory preallocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Allow JAX to grow GPU memory usage as needed
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"  # Use 90% of GPU memory
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure JAX sees your NVIDIA GPU
import jax
print(jax.devices())  # Should list NVIDIA GPU
print(jax.lib.xla_bridge.get_backend().platform)
import jax.numpy as jnp
from jax import jit, vmap, lax
from scipy.spatial import cKDTree
import numpy as np
from typing import Tuple, Optional, Callable
import functools

class MemoryOptimizedJAXParticleAdvection:
    """Memory-optimized JAX particle advection with streaming and checkpointing."""
    
    def __init__(self, grid_points: np.ndarray, velocity_data: np.ndarray, 
                 time_steps: np.ndarray, static_time_step: int = 0):
        """
        Initialize particle advection.
        
        Args:
            grid_points: Grid points coordinates (N, 3)
            velocity_data: Velocity field data (n_times, N, 3) or (1, N, 3) for static
            time_steps: Time step values
            static_time_step: Time step index for static fields
        """
        self.grid_points = jnp.array(grid_points)
        self.velocity_data = jnp.array(velocity_data)
        self.time_steps = jnp.array(time_steps)
        self.static = (self.velocity_data.shape[0] == 1)
        self.static_time_step = static_time_step
        self.grid_tree = cKDTree(grid_points)
        
        # Memory management settings
        self.max_gpu_memory_gb = 8.0  # Adjust based on your GPU
        self.checkpoint_every = 100   # Checkpoint frequency for gradient checkpointing
        
        self._compile_functions()
    
    def _compile_functions(self):
        """Compile JAX functions for GPU execution."""
        
        @jit
        def interpolate_velocity(position: jnp.ndarray, time_idx: int) -> jnp.ndarray:
            """Interpolate velocity at given position and time."""
            safe_time_idx = self.static_time_step if self.static else jnp.clip(time_idx, 0, len(self.velocity_data) - 1)
            distances = jnp.linalg.norm(self.grid_points - position, axis=1)
            nearest_idx = jnp.argmin(distances)
            return self.velocity_data[safe_time_idx, nearest_idx]
        
        @jit
        def rk4_step(position: jnp.ndarray, dt: float, time_idx: int) -> jnp.ndarray:
            """Single 4th-order Runge-Kutta step."""
            k1 = interpolate_velocity(position, time_idx)
            k2 = interpolate_velocity(position + 0.5 * dt * k1, time_idx)
            k3 = interpolate_velocity(position + 0.5 * dt * k2, time_idx)
            k4 = interpolate_velocity(position + dt * k3, time_idx)
            return position + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Store compiled functions
        self.interpolate_velocity = interpolate_velocity
        self.rk4_step = rk4_step
        self._compiled_advectors = {}
    
    def _estimate_memory_usage(self, n_particles: int, n_steps: int, store_full_trajectory: bool = True) -> float:
        """Estimate GPU memory usage in GB."""
        bytes_per_float = 4  # float32
        
        if store_full_trajectory:
            trajectory_memory = n_particles * n_steps * 3 * bytes_per_float
        else:
            trajectory_memory = n_particles * 3 * bytes_per_float  # Only current positions
        
        # Add velocity data and grid points memory
        velocity_memory = self.velocity_data.nbytes
        grid_memory = self.grid_points.nbytes
        
        total_bytes = trajectory_memory + velocity_memory + grid_memory
        return total_bytes / (1024**3)  # Convert to GB
    
    def _calculate_optimal_batch_size(self, n_particles: int, n_steps: int, 
                                    store_full_trajectory: bool = True) -> int:
        """Calculate optimal batch size based on available GPU memory."""
        # Start with full batch and reduce until memory fits
        for batch_size in [n_particles, n_particles//2, n_particles//4, n_particles//8, 1000, 500, 100]:
            if batch_size <= 0:
                break
            memory_usage = self._estimate_memory_usage(batch_size, n_steps, store_full_trajectory)
            if memory_usage < self.max_gpu_memory_gb * 0.8:  # Use 80% of available memory
                return min(batch_size, n_particles)
        return 100  # Minimum batch size
    
    def create_particle_grid(self, box_bounds: Tuple[Tuple[float, float], ...], 
                           resolution: Tuple[int, int, int]) -> jnp.ndarray:
        """Create initial particle positions in a box."""
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = box_bounds
        nx, ny, nz = resolution
        
        x = jnp.linspace(xmin, xmax, nx)
        y = jnp.linspace(ymin, ymax, ny)
        z = jnp.linspace(zmin, zmax, nz)
        
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        positions = jnp.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        return positions
    
    def _make_streaming_advector(self, chunk_size: int):
        """Create a streaming advector that processes particles in temporal chunks."""
        
        @jit
        def advect_chunk(positions: jnp.ndarray, dt: float, start_time_idx: int, 
                        n_chunk_steps: int) -> jnp.ndarray:
            """Advect particles for a chunk of time steps."""
            def step_fn(carry, i):
                pos = carry
                time_idx = start_time_idx + i if not self.static else self.static_time_step
                new_pos = self.rk4_step(pos, dt, time_idx)
                return new_pos, new_pos
            
            final_positions, _ = lax.scan(step_fn, positions, jnp.arange(n_chunk_steps))
            return final_positions
        
        return vmap(advect_chunk, in_axes=(0, None, None, None))
    
    def advect_particles_streaming(self, initial_positions: jnp.ndarray, dt: float, 
                                 n_steps: int, chunk_size: int = 100,
                                 callback: Optional[Callable] = None) -> jnp.ndarray:
        """
        Advect particles using temporal chunking to reduce memory usage.
        
        Args:
            initial_positions: Initial particle positions (N, 3)
            dt: Time step size
            n_steps: Total number of time steps
            chunk_size: Number of time steps per chunk
            callback: Optional callback function called after each chunk
            
        Returns:
            Final particle positions (N, 3)
        """
        advect_chunk = self._make_streaming_advector(chunk_size)
        n_particles = initial_positions.shape[0]
        
        # Calculate optimal batch size
        batch_size = self._calculate_optimal_batch_size(n_particles, chunk_size, store_full_trajectory=False)
        
        print(f"Using streaming advection with chunk_size={chunk_size}, batch_size={batch_size}")
        print(f"Estimated memory usage: {self._estimate_memory_usage(batch_size, chunk_size, False):.2f} GB")
        
        current_positions = initial_positions
        n_chunks = (n_steps + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(n_chunks):
            start_step = chunk_idx * chunk_size
            remaining_steps = min(chunk_size, n_steps - start_step)
            
            print(f"Processing temporal chunk {chunk_idx + 1}/{n_chunks} "
                  f"(steps {start_step} to {start_step + remaining_steps})")
            
            # Process particles in spatial batches
            n_batches = (n_particles + batch_size - 1) // batch_size
            new_positions = []
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_particles)
                batch_positions = current_positions[start_idx:end_idx]
                
                # Advect this batch for the current time chunk
                batch_final = advect_chunk(batch_positions, dt, start_step, remaining_steps)
                new_positions.append(batch_final)
            
            current_positions = jnp.concatenate(new_positions, axis=0)
            
            if callback:
                callback(chunk_idx, current_positions)
        
        return current_positions
    
    def _make_checkpointed_advector(self, n_steps: int, checkpoint_every: int):
        """Create advector with gradient checkpointing for memory efficiency."""
        
        @functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.dots_with_no_batch_dims)
        @jit
        def advect_segment(positions: jnp.ndarray, dt: float, start_idx: int, 
                          segment_steps: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Advect particles for a segment with checkpointing."""
            trajectory_segment = jnp.zeros((segment_steps + 1, positions.shape[0], 3))
            trajectory_segment = trajectory_segment.at[0].set(positions)
            
            def step_fn(carry, i):
                pos = carry
                time_idx = start_idx + i if not self.static else self.static_time_step
                new_pos = self.rk4_step(pos, dt, time_idx)
                return new_pos, new_pos
            
            final_pos, all_pos = lax.scan(step_fn, positions, jnp.arange(segment_steps))
            
            # Store trajectory segment
            trajectory_segment = trajectory_segment.at[1:].set(all_pos.transpose(1, 0, 2))
            
            return final_pos, trajectory_segment
        
        return vmap(advect_segment, in_axes=(0, None, None, None))
    
    def advect_particles_checkpointed(self, initial_positions: jnp.ndarray, dt: float, 
                                    n_steps: int, save_trajectory: bool = False,
                                    batch_size: Optional[int] = None) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Advect particles using gradient checkpointing to reduce memory usage.
        
        Args:
            initial_positions: Initial particle positions (N, 3)
            dt: Time step size
            n_steps: Number of time steps
            save_trajectory: Whether to save full trajectory (uses more memory)
            batch_size: Batch size (auto-calculated if None)
            
        Returns:
            Tuple of (final_positions, trajectories) where trajectories is None if save_trajectory=False
        """
        n_particles = initial_positions.shape[0]
        
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size(n_particles, n_steps, save_trajectory)
        
        print(f"Using checkpointed advection with batch_size={batch_size}")
        print(f"Estimated memory usage: {self._estimate_memory_usage(batch_size, n_steps, save_trajectory):.2f} GB")
        
        # Create checkpointed advector
        advect_segment = self._make_checkpointed_advector(n_steps, self.checkpoint_every)
        
        n_batches = (n_particles + batch_size - 1) // batch_size
        final_positions = []
        all_trajectories = [] if save_trajectory else None
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_particles)
            batch_positions = initial_positions[start_idx:end_idx]
            
            print(f"Processing batch {batch_idx + 1}/{n_batches}")
            
            # Use segments for checkpointing
            n_segments = (n_steps + self.checkpoint_every - 1) // self.checkpoint_every
            current_positions = batch_positions
            batch_trajectory = []
            
            for seg_idx in range(n_segments):
                start_step = seg_idx * self.checkpoint_every
                segment_steps = min(self.checkpoint_every, n_steps - start_step)
                
                final_pos, trajectory_seg = advect_segment(
                    current_positions, dt, start_step, segment_steps
                )
                
                current_positions = final_pos
                if save_trajectory:
                    batch_trajectory.append(trajectory_seg)
            
            final_positions.append(current_positions)
            if save_trajectory:
                # Concatenate trajectory segments
                full_batch_trajectory = jnp.concatenate(batch_trajectory, axis=0)
                all_trajectories.append(full_batch_trajectory)
        
        final_result = jnp.concatenate(final_positions, axis=0)
        trajectory_result = jnp.concatenate(all_trajectories, axis=1) if save_trajectory else None
        
        return final_result, trajectory_result
    
    def advect_particles_minimal_memory(self, initial_positions: jnp.ndarray, dt: float, 
                                      n_steps: int, output_frequency: int = 100,
                                      output_callback: Optional[Callable] = None) -> jnp.ndarray:
        """
        Ultra-minimal memory advection - only stores positions at output frequency.
        
        Args:
            initial_positions: Initial particle positions (N, 3)
            dt: Time step size
            n_steps: Number of time steps
            output_frequency: How often to output positions
            output_callback: Function to call with (step, positions) at each output
            
        Returns:
            Final particle positions only
        """
        @jit
        def advect_single_step(positions: jnp.ndarray, time_idx: int) -> jnp.ndarray:
            """Single time step for all particles."""
            return vmap(lambda pos: self.rk4_step(pos, dt, time_idx))(positions)
        
        current_positions = initial_positions
        
        for step in range(n_steps):
            time_idx = step if not self.static else self.static_time_step
            current_positions = advect_single_step(current_positions, time_idx)
            
            # Output at specified frequency
            if (step + 1) % output_frequency == 0 or step == n_steps - 1:
                print(f"Step {step + 1}/{n_steps}")
                if output_callback:
                    output_callback(step + 1, current_positions)
        
        return current_positions

# Example usage functions
def example_streaming_usage():
    """Example of using streaming advection for large particle sets."""
    # Create dummy data
    grid_points = np.random.rand(1000, 3) * 10
    velocity_data = np.random.rand(1, 1000, 3)  # Static field
    time_steps = np.linspace(0, 10, 100)
    
    advector = MemoryOptimizedJAXParticleAdvection(grid_points, velocity_data, time_steps)
    
    # Create large particle set
    initial_positions = advector.create_particle_grid(
        box_bounds=((0, 10), (0, 10), (0, 10)),
        resolution=(50, 50, 50)  # 125k particles
    )
    
    print(f"Advecting {initial_positions.shape[0]} particles...")
    
    # Use streaming for memory efficiency
    final_positions = advector.advect_particles_streaming(
        initial_positions, dt=0.1, n_steps=1000, chunk_size=50
    )
    
    print(f"Final positions shape: {final_positions.shape}")
    return final_positions

def example_checkpointed_usage():
    """Example of using checkpointed advection with trajectory saving."""
    grid_points = np.random.rand(500, 3) * 5
    velocity_data = np.random.rand(100, 500, 3)  # Time-varying field
    time_steps = np.linspace(0, 10, 100)
    
    advector = MemoryOptimizedJAXParticleAdvection(grid_points, velocity_data, time_steps)
    
    initial_positions = advector.create_particle_grid(
        box_bounds=((0, 5), (0, 5), (0, 5)),
        resolution=(20, 20, 20)  # 8k particles
    )
    
    # Use checkpointing with trajectory saving
    final_pos, trajectories = advector.advect_particles_checkpointed(
        initial_positions, dt=0.1, n_steps=100, save_trajectory=True
    )
    
    print(f"Final positions: {final_pos.shape}")
    if trajectories is not None:
        print(f"Trajectories: {trajectories.shape}")
    
    return final_pos, trajectories