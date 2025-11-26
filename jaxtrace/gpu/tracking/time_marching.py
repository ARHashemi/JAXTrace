"""
Complete Time-Marching Pipeline for Particle Tracking

Integrates:
1. Element Search (Phase 1 batch processor)
2. Velocity Interpolation (block-local barycentric coordinates)
3. Time Integration (Forward Euler / RK4)

Pipeline stages for single timestep:
  Current State → Interpolate Velocities → Integrate Positions → Search New Elements → New State

Performance target: >2,500 p/s with Forward Euler (bottlenecked by element search @ 3,416 p/s)
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Optional, Callable
from dataclasses import replace

from ..particles import ParticleData
from ..forest import PaddedArrays
from ..batching import BatchConfig

from .velocity_interpolation import batch_interpolate_velocities
from .time_integration import (
    forward_euler_step,
    forward_euler_step_with_active_mask,
    compute_adaptive_timestep,
    rk4_step_with_search,
    rk4_step_simple
)


class ParticleTimeMarcher:
    """
    GPU-accelerated particle time-marching with element search.

    Combines velocity interpolation, time integration, and element search
    into a complete time-marching pipeline.

    Examples
    --------
    >>> # Setup
    >>> marcher = ParticleTimeMarcher(padded_arrays, config)
    >>>
    >>> # Define time-dependent velocity field
    >>> def get_velocity_field(time):
    ...     # Load or compute velocity field at time t
    ...     return velocity_field_array  # (n_blocks, max_nodes, 3)
    >>>
    >>> # March particles
    >>> results = marcher.march_forward_euler(
    ...     particle_data,
    ...     get_velocity_field,
    ...     n_timesteps=100,
    ...     dt=0.01,
    ...     start_time=0.0
    ... )
    >>>
    >>> print(f"Final throughput: {results['avg_throughput']:.0f} p/s")
    """

    def __init__(
        self,
        padded_arrays: PaddedArrays,
        connectivity: np.ndarray,
        node_positions: np.ndarray,
        config: BatchConfig,
        verbose: bool = True
    ):
        """
        Initialize time-marching pipeline.

        Parameters
        ----------
        padded_arrays : PaddedArrays
            Block-local mesh structure from Phase 1
        connectivity : np.ndarray
            Element connectivity array (n_elements, 4), int32
        node_positions : np.ndarray
            Node positions (n_nodes, 3), float32
        config : BatchConfig
            Batch processing configuration
        verbose : bool
            Print progress and statistics
        """
        self.padded_arrays = padded_arrays
        self.config = config
        self.verbose = verbose

        # Transfer mesh to GPU once at initialization
        if verbose:
            print("📊 Transferring mesh to GPU...")
        self.connectivity_gpu = jax.device_put(connectivity)
        self.node_positions_gpu = jax.device_put(node_positions)
        if verbose:
            print("✅ Mesh on GPU")

    def _interpolate_velocities_for_block(
        self,
        particle_indices: np.ndarray,
        particle_data: ParticleData,
        block_id: int,
        velocity_field_all_blocks: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate velocities for particles in a single block.

        Parameters
        ----------
        particle_indices : np.ndarray
            Indices of particles in this block
        particle_data : ParticleData
            Full particle data
        block_id : int
            Block ID
        velocity_field_all_blocks : np.ndarray
            Velocity fields for all blocks (n_blocks, max_nodes, 3)

        Returns
        -------
        velocities : np.ndarray
            Interpolated velocities (len(particle_indices), 3)
        """

        if len(particle_indices) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # Extract particle data for this block
        block_positions = particle_data.positions[particle_indices]
        block_element_ids = particle_data.element_ids[particle_indices]

        # Upload to GPU
        block_positions_gpu = jax.device_put(block_positions)
        block_element_ids_gpu = jax.device_put(block_element_ids)
        block_velocity_field_gpu = jax.device_put(velocity_field_all_blocks[block_id])

        # Interpolate on GPU (using global connectivity and node_positions)
        velocities = batch_interpolate_velocities(
            block_positions_gpu,
            block_element_ids_gpu,
            self.connectivity_gpu,
            self.node_positions_gpu,
            block_velocity_field_gpu
        )

        # Transfer back to CPU
        return np.array(velocities)

    def interpolate_velocities(
        self,
        particle_data: ParticleData,
        velocity_field: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate velocities for all particles.

        Processes particles block-by-block for efficiency.

        Parameters
        ----------
        particle_data : ParticleData
            Current particle state
        velocity_field : np.ndarray
            Velocity field at all nodes (n_blocks, max_nodes, 3)

        Returns
        -------
        velocities : np.ndarray
            Interpolated velocities (N, 3)
        """

        n_particles = len(particle_data.positions)
        velocities = np.zeros((n_particles, 3), dtype=np.float32)

        # Group particles by block
        from ..batching.block_grouping import group_particles_by_block
        grouping = group_particles_by_block(
            particle_data.block_ids,
            self.padded_arrays.block_sizes
        )

        # Process each block
        for block_id, particle_indices in grouping.groups.items():
            if len(particle_indices) == 0:
                continue

            block_velocities = self._interpolate_velocities_for_block(
                np.array(particle_indices),
                particle_data,
                block_id,
                velocity_field
            )

            velocities[particle_indices] = block_velocities

        return velocities


    def march_single_timestep_euler(
        self,
        particle_data: ParticleData,
        velocity_field: np.ndarray,
        dt: float,
        search_fn: Callable,
        use_active_mask: bool = True
    ) -> Tuple[ParticleData, Dict]:
        """
        Single Forward Euler timestep: interpolate → integrate → search.

        Parameters
        ----------
        particle_data : ParticleData
            Current particle state
        velocity_field : np.ndarray
            Velocity field (n_blocks, max_nodes, 3)
        dt : float
            Time step size
        search_fn : Callable
            Element search function: search_fn(particle_data) -> (particle_data, search_stats)
        use_active_mask : bool
            Only update active particles

        Returns
        -------
        new_particle_data : ParticleData
            Updated particle state
        step_stats : dict
            Statistics for this timestep
        """

        t_start = time.time()

        # Stage 1: Interpolate velocities
        t_interp_start = time.time()
        velocities = self.interpolate_velocities(particle_data, velocity_field)
        t_interp = time.time() - t_interp_start

        # Stage 2: Forward Euler integration
        t_integrate_start = time.time()
        if use_active_mask:
            new_positions = forward_euler_step_with_active_mask(
                jnp.array(particle_data.positions),
                jnp.array(velocities),
                jnp.array(particle_data.active_mask),
                dt
            )
        else:
            new_positions = forward_euler_step(
                jnp.array(particle_data.positions),
                jnp.array(velocities),
                dt
            )
        new_positions = np.array(new_positions)
        t_integrate = time.time() - t_integrate_start

        # Update particle positions
        particle_data.positions = new_positions

        # Stage 3: Search for new elements
        t_search_start = time.time()
        particle_data, search_stats = search_fn(particle_data)
        t_search = time.time() - t_search_start

        t_total = time.time() - t_start

        # Compile statistics
        n_particles = len(particle_data.positions)
        step_stats = {
            'time_interpolation': t_interp,
            'time_integration': t_integrate,
            'time_search': t_search,
            'time_total': t_total,
            'throughput': n_particles / t_total if t_total > 0 else 0,
            'search_stats': search_stats,
            'n_particles': n_particles,
            'n_active': int(np.sum(particle_data.active_mask))
        }

        return particle_data, step_stats

    def march_forward_euler(
        self,
        particle_data: ParticleData,
        velocity_field_fn: Callable[[float], np.ndarray],
        search_fn: Callable,
        n_timesteps: int,
        dt: float,
        start_time: float = 0.0,
        checkpoint_interval: Optional[int] = None,
        checkpoint_callback: Optional[Callable] = None
    ) -> Dict:
        """
        March particles for multiple timesteps using Forward Euler.

        Parameters
        ----------
        particle_data : ParticleData
            Initial particle state
        velocity_field_fn : callable
            Function(time) -> velocity_field (n_blocks, max_nodes, 3)
        search_fn : callable
            Element search function: search_fn(particle_data) -> (particle_data, search_stats)
        n_timesteps : int
            Number of timesteps to march
        dt : float
            Time step size
        start_time : float
            Initial simulation time
        checkpoint_interval : int, optional
            Save checkpoint every N steps
        checkpoint_callback : callable, optional
            Function(step, time, particle_data, stats) called at checkpoints

        Returns
        -------
        results : dict
            Overall statistics and final particle state
        """

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"FORWARD EULER TIME-MARCHING")
            print(f"{'='*80}")
            print(f"Particles: {len(particle_data.positions):,}")
            print(f"Timesteps: {n_timesteps}")
            print(f"dt: {dt}")
            print(f"Total time: {n_timesteps * dt:.3f}")
            print()

        all_stats = []
        current_time = start_time

        for step in range(n_timesteps):
            if self.verbose and step % 10 == 0:
                print(f"Step {step}/{n_timesteps} (t={current_time:.3f})...")

            # Get velocity field at current time
            velocity_field = velocity_field_fn(current_time)

            # Single timestep
            particle_data, step_stats = self.march_single_timestep_euler(
                particle_data, velocity_field, dt, search_fn
            )

            step_stats['step'] = step
            step_stats['time'] = current_time
            all_stats.append(step_stats)

            # Checkpoint
            if checkpoint_interval and (step + 1) % checkpoint_interval == 0:
                if checkpoint_callback:
                    checkpoint_callback(step, current_time, particle_data, step_stats)

            current_time += dt

        # Aggregate results
        avg_throughput = np.mean([s['throughput'] for s in all_stats])
        total_time = np.sum([s['time_total'] for s in all_stats])

        if self.verbose:
            print()
            print(f"{'='*80}")
            print(f"TIME-MARCHING COMPLETE")
            print(f"{'='*80}")
            print(f"Average throughput: {avg_throughput:.0f} p/s")
            print(f"Total compute time: {total_time:.2f} s")
            print(f"Final active particles: {np.sum(particle_data.active_mask):,}")
            print()

        results = {
            'particle_data': particle_data,
            'all_stats': all_stats,
            'avg_throughput': avg_throughput,
            'total_time': total_time,
            'final_time': current_time
        }

        return results

    def march_single_timestep_rk4(
        self,
        particle_data: ParticleData,
        velocity_field_fn: Callable[[float], np.ndarray],
        dt: float,
        current_time: float,
        search_fn: Callable,
        use_intermediate_searches: bool = True
    ) -> Tuple[ParticleData, Dict]:
        """
        Single RK4 timestep with optional intermediate element searches.

        Parameters
        ----------
        particle_data : ParticleData
            Current particle state
        velocity_field_fn : callable
            Function(time) -> velocity_field (n_blocks, max_nodes, 3)
        dt : float
            Time step size
        current_time : float
            Current simulation time
        search_fn : Callable
            Element search function: search_fn(particle_data) -> (particle_data, search_stats)
        use_intermediate_searches : bool
            If True, performs element search at k2, k3, k4 intermediate positions (accurate but slow)
            If False, uses current element_ids for all 4 stages (fast but less accurate)

        Returns
        -------
        new_particle_data : ParticleData
            Updated particle state
        step_stats : dict
            Statistics for this timestep
        """

        t_start = time.time()

        if use_intermediate_searches:
            # Full RK4 with intermediate searches (most accurate)

            # Create velocity interpolator that matches expected signature
            def velocity_interpolator(pdata, t):
                vf = velocity_field_fn(t)
                return self.interpolate_velocities(pdata, vf)

            # Use rk4_step_with_search
            particle_data, rk4_stats = rk4_step_with_search(
                particle_data,
                velocity_interpolator,
                search_fn,
                dt,
                current_time
            )

            t_total = time.time() - t_start

            # Compile statistics
            n_particles = len(particle_data.positions)
            step_stats = {
                'time_total': t_total,
                'throughput': n_particles / t_total if t_total > 0 else 0,
                'rk4_stats': rk4_stats,
                'n_particles': n_particles,
                'n_active': int(np.sum(particle_data.active_mask)),
                'n_searches': rk4_stats['n_searches']
            }

        else:
            # Simplified RK4 (no intermediate searches - faster but less accurate)

            # Create velocity interpolator
            def velocity_interpolator(pdata, t):
                vf = velocity_field_fn(t)
                return self.interpolate_velocities(pdata, vf)

            # Use rk4_step_simple
            t_rk4_start = time.time()
            particle_data = rk4_step_simple(
                particle_data,
                velocity_interpolator,
                dt,
                current_time
            )
            t_rk4 = time.time() - t_rk4_start

            # Convert JAX arrays to numpy before search
            import jax.numpy as jnp
            if isinstance(particle_data.positions, jnp.ndarray):
                particle_data = replace(
                    particle_data,
                    positions=np.array(particle_data.positions),
                    velocities=np.array(particle_data.velocities)
                )

            # Final search at new positions
            t_search_start = time.time()
            particle_data, search_stats = search_fn(particle_data)
            t_search = time.time() - t_search_start

            t_total = time.time() - t_start

            # Compile statistics
            n_particles = len(particle_data.positions)
            step_stats = {
                'time_rk4': t_rk4,
                'time_search': t_search,
                'time_total': t_total,
                'throughput': n_particles / t_total if t_total > 0 else 0,
                'search_stats': search_stats,
                'n_particles': n_particles,
                'n_active': int(np.sum(particle_data.active_mask)),
                'n_searches': 1
            }

        return particle_data, step_stats

    def march_rk4(
        self,
        particle_data: ParticleData,
        velocity_field_fn: Callable[[float], np.ndarray],
        search_fn: Callable,
        n_timesteps: int,
        dt: float,
        start_time: float = 0.0,
        use_intermediate_searches: bool = True,
        checkpoint_interval: Optional[int] = None,
        checkpoint_callback: Optional[Callable] = None
    ) -> Dict:
        """
        March particles for multiple timesteps using RK4.

        Parameters
        ----------
        particle_data : ParticleData
            Initial particle state
        velocity_field_fn : callable
            Function(time) -> velocity_field (n_blocks, max_nodes, 3)
        search_fn : callable
            Element search function: search_fn(particle_data) -> (particle_data, search_stats)
        n_timesteps : int
            Number of timesteps to march
        dt : float
            Time step size
        start_time : float
            Initial simulation time
        use_intermediate_searches : bool
            If True: Full RK4 with searches at k2, k3, k4 (accurate, ~800 p/s)
            If False: Simplified RK4 with single final search (fast, ~2,500 p/s)
        checkpoint_interval : int, optional
            Save checkpoint every N steps
        checkpoint_callback : callable, optional
            Function(step, time, particle_data, stats) called at checkpoints

        Returns
        -------
        results : dict
            Overall statistics and final particle state
        """

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"RK4 TIME-MARCHING")
            if use_intermediate_searches:
                print(f"Mode: Full (with intermediate searches)")
            else:
                print(f"Mode: Simplified (single final search)")
            print(f"{'='*80}")
            print(f"Particles: {len(particle_data.positions):,}")
            print(f"Timesteps: {n_timesteps}")
            print(f"dt: {dt}")
            print(f"Total time: {n_timesteps * dt:.3f}")
            print()

        all_stats = []
        current_time = start_time

        for step in range(n_timesteps):
            if self.verbose and step % 10 == 0:
                print(f"Step {step}/{n_timesteps} (t={current_time:.3f})...")

            # Single RK4 timestep
            particle_data, step_stats = self.march_single_timestep_rk4(
                particle_data,
                velocity_field_fn,
                dt,
                current_time,
                search_fn,
                use_intermediate_searches
            )

            step_stats['step'] = step
            step_stats['time'] = current_time
            all_stats.append(step_stats)

            # Checkpoint
            if checkpoint_interval and (step + 1) % checkpoint_interval == 0:
                if checkpoint_callback:
                    checkpoint_callback(step, current_time, particle_data, step_stats)

            current_time += dt

        # Aggregate results
        avg_throughput = np.mean([s['throughput'] for s in all_stats])
        total_time = np.sum([s['time_total'] for s in all_stats])

        if self.verbose:
            print()
            print(f"{'='*80}")
            print(f"RK4 TIME-MARCHING COMPLETE")
            print(f"{'='*80}")
            print(f"Average throughput: {avg_throughput:.0f} p/s")
            print(f"Total compute time: {total_time:.2f} s")
            print(f"Final active particles: {np.sum(particle_data.active_mask):,}")
            if use_intermediate_searches:
                total_searches = sum([s['n_searches'] for s in all_stats])
                print(f"Total element searches: {total_searches}")
            print()

        results = {
            'particle_data': particle_data,
            'all_stats': all_stats,
            'avg_throughput': avg_throughput,
            'total_time': total_time,
            'final_time': current_time
        }

        return results


# ============================================================================
# Convenience Functions
# ============================================================================

def create_constant_velocity_field(
    padded_arrays: PaddedArrays,
    velocity: np.ndarray,
    node_positions: np.ndarray
) -> np.ndarray:
    """
    Create constant velocity field for testing.

    Parameters
    ----------
    padded_arrays : PaddedArrays
        Mesh structure
    velocity : np.ndarray
        Constant velocity vector (3,)
    node_positions : np.ndarray
        Original node positions (n_nodes, 3)

    Returns
    -------
    velocity_field : np.ndarray
        Velocity field (n_blocks, max_nodes, 3)
    """

    n_blocks = padded_arrays.n_blocks
    n_nodes = node_positions.shape[0]

    # Create velocity field with shape (n_blocks, n_nodes, 3)
    # Each block sees all nodes (replicated across blocks)
    velocity_field = np.tile(velocity, (n_blocks, n_nodes, 1)).astype(np.float32)

    return velocity_field


def create_time_dependent_velocity_field_fn(
    padded_arrays: PaddedArrays,
    node_positions: np.ndarray,
    base_velocity: np.ndarray,
    amplitude: float = 0.5,
    frequency: float = 1.0
) -> Callable[[float], np.ndarray]:
    """
    Create sinusoidal time-dependent velocity field for testing.

    v(t) = base_velocity * (1 + amplitude * sin(2π * frequency * t))

    Parameters
    ----------
    padded_arrays : PaddedArrays
        Mesh structure
    node_positions : np.ndarray
        Original node positions (n_nodes, 3)
    base_velocity : np.ndarray
        Base velocity vector (3,)
    amplitude : float
        Oscillation amplitude (default: 0.5 → 50% variation)
    frequency : float
        Oscillation frequency in Hz (default: 1.0)

    Returns
    -------
    velocity_field_fn : callable
        Function(time) -> velocity_field
    """

    def velocity_field_fn(time):
        # Sinusoidal modulation
        factor = 1.0 + amplitude * np.sin(2 * np.pi * frequency * time)
        velocity = base_velocity * factor

        # Create constant field at this time
        return create_constant_velocity_field(padded_arrays, velocity, node_positions)

    return velocity_field_fn


# ============================================================================
# Performance Benchmarking
# ============================================================================

def benchmark_time_marching_pipeline(
    padded_arrays: PaddedArrays,
    particle_data: ParticleData,
    config: BatchConfig,
    n_timesteps: int = 10,
    dt: float = 0.01
) -> Dict:
    """
    Benchmark time-marching pipeline performance.

    Tests with constant velocity field to isolate pipeline overhead.

    Parameters
    ----------
    padded_arrays : PaddedArrays
        Mesh structure
    particle_data : ParticleData
        Test particles
    config : BatchConfig
        Configuration
    n_timesteps : int
        Number of test timesteps
    dt : float
        Time step size

    Returns
    -------
    benchmark_results : dict
        Detailed performance breakdown
    """

    print(f"\n{'='*80}")
    print(f"TIME-MARCHING PIPELINE BENCHMARK")
    print(f"{'='*80}")

    # Create marcher
    marcher = ParticleTimeMarcher(padded_arrays, config, verbose=False)

    # Constant velocity field [1, 0, 0] m/s
    velocity_field = create_constant_velocity_field(
        padded_arrays, np.array([1.0, 0.0, 0.0])
    )

    def vel_fn(t):
        return velocity_field

    # Run benchmark
    print(f"Running {n_timesteps} timesteps...")
    t_start = time.time()

    results = marcher.march_forward_euler(
        particle_data,
        vel_fn,
        n_timesteps=n_timesteps,
        dt=dt
    )

    t_total = time.time() - t_start

    # Analyze bottlenecks
    all_stats = results['all_stats']
    avg_interp = np.mean([s['time_interpolation'] for s in all_stats])
    avg_integrate = np.mean([s['time_integration'] for s in all_stats])
    avg_search = np.mean([s['time_search'] for s in all_stats])

    print(f"\nPERFORMANCE BREAKDOWN:")
    print(f"  Interpolation: {avg_interp*1000:.1f} ms/step ({avg_interp/results['all_stats'][0]['time_total']*100:.1f}%)")
    print(f"  Integration:   {avg_integrate*1000:.1f} ms/step ({avg_integrate/results['all_stats'][0]['time_total']*100:.1f}%)")
    print(f"  Element Search: {avg_search*1000:.1f} ms/step ({avg_search/results['all_stats'][0]['time_total']*100:.1f}%)")
    print(f"\nOverall Throughput: {results['avg_throughput']:.0f} p/s")
    print(f"Total Time: {t_total:.2f} s")

    return {
        'results': results,
        'avg_time_interpolation': avg_interp,
        'avg_time_integration': avg_integrate,
        'avg_time_search': avg_search,
        'total_time': t_total
    }
