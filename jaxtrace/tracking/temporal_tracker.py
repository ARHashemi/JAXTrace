"""
Temporal Batching Particle Tracker

Advances ALL particles through temporal windows on GPU.
Much better GPU utilization than spatial batching.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Optional
import time

from ..fields.temporal_field import TemporalBatchingField, create_grid_hash_interpolator
from .particles import Trajectory


class TemporalBatchingTracker:
    """
    Particle tracker using temporal batching.

    Advances all particles through time windows on GPU for better performance.
    """

    def __init__(self,
                 integrator: Callable,
                 field: TemporalBatchingField,
                 boundary_fn: Callable,
                 temporal_window_size: int = 20,
                 record_velocities: bool = False):
        """
        Initialize temporal batching tracker.

        Parameters
        ----------
        integrator : Callable
            Integration function (e.g., rk4_step)
        field : TemporalBatchingField
            Temporal field with on-demand loading
        boundary_fn : Callable
            Boundary condition function
        temporal_window_size : int
            Number of velocity field timesteps per window (default: 20)
        record_velocities : bool
            Whether to record velocities (default: False)
        """

        self.integrator = integrator
        self.field = field
        self.boundary_fn = boundary_fn
        self.temporal_window_size = temporal_window_size
        self.record_velocities = record_velocities

        print(f"🚀 Temporal batching tracker initialized:")
        print(f"   Window size: {temporal_window_size} velocity timesteps")
        print(f"   Record velocities: {record_velocities}")

    def track_particles(self,
                       initial_positions: np.ndarray,
                       n_tracking_steps: int,
                       dt_tracking: float,
                       dt_data: float,
                       progress_callback: Optional[Callable] = None) -> Trajectory:
        """
        Track particles using temporal batching.

        Parameters
        ----------
        initial_positions : np.ndarray
            Initial particle positions (N, 3)
        n_tracking_steps : int
            Number of tracking timesteps
        dt_tracking : float
            Tracking time step
        dt_data : float
            Time interval between velocity field timesteps
        progress_callback : Callable, optional
            Progress callback function

        Returns
        -------
        Trajectory
            Particle trajectory
        """

        initial_positions = np.asarray(initial_positions, dtype=np.float32)
        n_particles = initial_positions.shape[0]

        print(f"🏃 Temporal batching tracking:")
        print(f"   Particles: {n_particles:,}")
        print(f"   Tracking steps: {n_tracking_steps:,}")
        print(f"   dt_tracking: {dt_tracking}")
        print(f"   dt_data: {dt_data}")

        # Allocate trajectory storage
        positions = np.zeros((n_tracking_steps, n_particles, 3), dtype=np.float32)
        velocities = None
        if self.record_velocities:
            velocities = np.zeros((n_tracking_steps, n_particles, 3), dtype=np.float32)

        positions[0] = initial_positions

        # Compute how many data timesteps are needed
        t_final = (n_tracking_steps - 1) * dt_tracking
        n_data_steps_needed = int(np.ceil(t_final / dt_data)) + 1
        n_data_steps_needed = min(n_data_steps_needed, self.field.n_timesteps)

        print(f"   Data timesteps needed: {n_data_steps_needed} / {self.field.n_timesteps}")

        # Process in temporal windows
        n_windows = int(np.ceil(n_data_steps_needed / self.temporal_window_size))

        print(f"   Temporal windows: {n_windows}")
        print()

        start_time = time.time()
        current_positions = jnp.array(initial_positions)

        tracking_step = 0

        for window_idx in range(n_windows):
            # Determine data timesteps for this window
            data_start = window_idx * self.temporal_window_size
            data_end = min(data_start + self.temporal_window_size, n_data_steps_needed)
            data_indices = list(range(data_start, data_end))

            # Determine tracking steps covered by this window
            t_data_start = data_start * dt_data
            t_data_end = data_end * dt_data

            tracking_start = tracking_step
            tracking_end = min(n_tracking_steps, int(np.ceil(t_data_end / dt_tracking)) + 1)

            print(f"   Window {window_idx + 1}/{n_windows}:")
            print(f"      Data timesteps: {data_start} to {data_end - 1}")
            print(f"      Tracking steps: {tracking_start} to {tracking_end - 1}")

            # Load velocity fields for this window
            window_start_load = time.time()
            meshes = self.field.load_window(data_indices)
            load_time = time.time() - window_start_load

            print(f"      Loaded {len(meshes)} timesteps in {load_time:.2f}s")

            # Create interpolators
            interpolators = [create_grid_hash_interpolator(mesh) for mesh in meshes]

            # Advance particles through this window
            window_start_compute = time.time()

            positions_window, velocities_window = self._advance_window(
                current_positions,
                interpolators,
                tracking_start,
                tracking_end,
                dt_tracking,
                dt_data,
                t_data_start
            )

            compute_time = time.time() - window_start_compute

            # Store results
            n_steps_window = tracking_end - tracking_start
            positions[tracking_start:tracking_end] = np.array(positions_window)

            if self.record_velocities and velocities_window is not None:
                velocities[tracking_start:tracking_end] = np.array(velocities_window)

            # Update current positions for next window
            current_positions = positions_window[-1]

            tracking_step = tracking_end

            print(f"      Computed {n_steps_window} steps in {compute_time:.2f}s")
            print(f"      Speed: {n_particles * n_steps_window / compute_time:.0f} particle-steps/sec")

            # Progress callback
            if progress_callback is not None:
                progress = tracking_step / n_tracking_steps
                progress_callback(progress)

            # Break if we've covered all tracking steps
            if tracking_step >= n_tracking_steps:
                break

        total_time = time.time() - start_time

        print()
        print(f"✅ Temporal batching complete:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Overall speed: {n_particles * n_tracking_steps / total_time:.0f} particle-steps/sec")

        # Create times array
        times = np.arange(n_tracking_steps, dtype=np.float32) * dt_tracking

        # Create trajectory
        trajectory = Trajectory(
            positions=positions,
            times=times,
            velocities=velocities,
            metadata={
                'integrator': 'temporal_batching',
                'n_particles': n_particles,
                'n_timesteps': n_tracking_steps,
                'dt_tracking': dt_tracking,
                'dt_data': dt_data,
                'temporal_window_size': self.temporal_window_size
            }
        )

        return trajectory

    def _advance_window(self,
                       positions_start: jnp.ndarray,
                       interpolators: list,
                       tracking_start: int,
                       tracking_end: int,
                       dt_tracking: float,
                       dt_data: float,
                       t_window_start: float) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Advance particles through one temporal window on GPU.

        Parameters
        ----------
        positions_start : jnp.ndarray
            Starting positions (N, 3)
        interpolators : list
            List of grid hash interpolators for this window
        tracking_start : int
            Starting tracking step index
        tracking_end : int
            Ending tracking step index
        dt_tracking : float
            Tracking time step
        dt_data : float
            Data time step
        t_window_start : float
            Time at start of this window

        Returns
        -------
        positions, velocities : jnp.ndarray
            Trajectory through window
        """

        n_steps = tracking_end - tracking_start
        n_particles = positions_start.shape[0]

        # Allocate storage for this window
        positions_window = jnp.zeros((n_steps, n_particles, 3), dtype=jnp.float32)
        velocities_window = None
        if self.record_velocities:
            velocities_window = jnp.zeros((n_steps, n_particles, 3), dtype=jnp.float32)

        # Single step advance (NOT JIT-compiled due to dynamic interpolator indexing)
        def advance_step(pos, t_current):
            """Advance one tracking step."""

            # Temporal interpolation between data timesteps
            t_idx_float = (t_current - t_window_start) / dt_data
            t_idx_left = int(np.floor(t_idx_float))
            t_idx_right = t_idx_left + 1

            # Clamp indices
            n_data = len(interpolators)
            t_idx_left = max(0, min(t_idx_left, n_data - 1))
            t_idx_right = max(0, min(t_idx_right, n_data - 1))

            alpha = float(t_idx_float - t_idx_left)
            alpha = max(0.0, min(alpha, 1.0))

            # Field function with temporal interpolation
            def field_fn(positions, t):
                v_left = interpolators[t_idx_left](positions)
                v_right = interpolators[t_idx_right](positions)
                return (1.0 - alpha) * v_left + alpha * v_right

            # RK4 step
            pos_next = self.integrator(pos, t_current, dt_tracking, field_fn)

            # Apply boundary conditions
            pos_next = self.boundary_fn(pos_next)

            return pos_next

        # Advance through window (on GPU)
        current_pos = positions_start

        for step_idx in range(n_steps):
            t_current = (tracking_start + step_idx) * dt_tracking

            # Store current position
            positions_window = positions_window.at[step_idx].set(current_pos)

            # Advance
            if step_idx < n_steps - 1:  # Don't advance on last step
                current_pos = advance_step(current_pos, t_current)

        return positions_window, velocities_window
