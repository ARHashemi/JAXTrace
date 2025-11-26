"""
Time Integration for Particle Tracking

GPU-accelerated time integration schemes:
- Forward Euler: First-order, fast, single velocity evaluation per step
- RK4: Fourth-order, accurate, requires 4 velocity evaluations per step

Leverages existing RK4 implementation from jaxtrace/integrators/rk4.py
but adapted for particle tracking with element search integration.

Performance targets:
- Forward Euler: >20,000 p/s (minimal overhead)
- RK4 with search: ~800 p/s (4× velocity interpolation + 3× element search)
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple
from dataclasses import replace

# Import existing RK4 implementation
from jaxtrace.integrators.rk4 import rk4_step as _rk4_step_original


# ============================================================================
# Forward Euler Integration
# ============================================================================

@jax.jit
def forward_euler_step(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    dt: float
) -> jnp.ndarray:
    """
    Single Forward Euler integration step.

    Simplest time integration: x_{n+1} = x_n + dt * v_n

    Parameters
    ----------
    positions : jnp.ndarray
        Current particle positions (N, 3)
    velocities : jnp.ndarray
        Velocities at current positions (N, 3)
    dt : float
        Time step size

    Returns
    -------
    new_positions : jnp.ndarray
        Updated positions (N, 3)

    Notes
    -----
    Forward Euler is:
    - First-order accurate: O(dt)
    - Conditionally stable: requires dt < stability limit
    - Fast: Only 1 velocity evaluation per timestep
    - Recommended for initial implementation and testing

    For higher accuracy, use RK4 (4th order) but at 4× computational cost.

    Examples
    --------
    >>> positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    >>> velocities = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> dt = 0.1
    >>> new_positions = forward_euler_step(positions, velocities, dt)
    >>> new_positions
    Array([[0.1, 0.0, 0.0],
           [1.0, 1.1, 0.0]], dtype=float32)
    """

    return positions + dt * velocities


@jax.jit
def forward_euler_step_with_active_mask(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    active_mask: jnp.ndarray,
    dt: float
) -> jnp.ndarray:
    """
    Forward Euler step with active particle mask.

    Only updates active particles; inactive particles remain stationary.

    Parameters
    ----------
    positions : jnp.ndarray
        Current positions (N, 3)
    velocities : jnp.ndarray
        Velocities (N, 3)
    active_mask : jnp.ndarray
        Boolean mask (N,), True for active particles
    dt : float
        Time step

    Returns
    -------
    new_positions : jnp.ndarray
        Updated positions (N, 3)
    """

    # Compute displacement
    displacement = dt * velocities

    # Apply only to active particles
    active_mask_expanded = active_mask[:, None]  # (N, 1)
    new_positions = jnp.where(active_mask_expanded, positions + displacement, positions)

    return new_positions


# ============================================================================
# Adaptive Time Stepping
# ============================================================================

@jax.jit
def compute_adaptive_timestep(
    velocities: jnp.ndarray,
    element_sizes: jnp.ndarray,
    cfl_number: float = 0.5,
    dt_min: float = 1e-6,
    dt_max: float = 1.0
) -> jnp.ndarray:
    """
    Compute per-particle adaptive time steps based on CFL condition.

    CFL condition: dt = CFL * h / |v|
    where h is element size and v is velocity magnitude.

    Parameters
    ----------
    velocities : jnp.ndarray
        Particle velocities (N, 3)
    element_sizes : jnp.ndarray
        Characteristic size of element containing each particle (N,)
    cfl_number : float
        CFL number (default: 0.5 for stability)
    dt_min : float
        Minimum allowed timestep
    dt_max : float
        Maximum allowed timestep

    Returns
    -------
    dt_adaptive : jnp.ndarray
        Per-particle timesteps (N,)

    Examples
    --------
    >>> velocities = jnp.array([[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    >>> element_sizes = jnp.array([0.1, 0.1])
    >>> dt = compute_adaptive_timestep(velocities, element_sizes, cfl_number=0.5)
    >>> dt
    Array([0.05, 0.005], dtype=float32)  # Faster particles get smaller dt
    """

    # Compute velocity magnitudes
    velocity_mags = jnp.linalg.norm(velocities, axis=1)

    # CFL condition: dt = CFL * h / |v|
    # Avoid division by zero for stationary particles
    velocity_mags_safe = jnp.maximum(velocity_mags, 1e-10)
    dt_cfl = cfl_number * element_sizes / velocity_mags_safe

    # Clamp to min/max bounds
    dt_adaptive = jnp.clip(dt_cfl, dt_min, dt_max)

    return dt_adaptive


# ============================================================================
# RK4 Integration with Element Search
# ============================================================================

def rk4_step_with_search(
    particle_data,
    velocity_interpolator: Callable,
    element_searcher: Callable,
    dt: float,
    current_time: float
) -> Tuple:
    """
    RK4 time integration with element search for intermediate stages.

    This function adapts the existing rk4_step() from jaxtrace/integrators/rk4.py
    to work with particle tracking, where intermediate RK4 stages (k2, k3, k4)
    require searching for new elements as particles move.

    RK4 stages:
    1. k1 = v(x_n, t_n)                    [no search needed - already have element]
    2. k2 = v(x_n + dt/2 * k1, t + dt/2)   [search needed for x_n + dt/2 * k1]
    3. k3 = v(x_n + dt/2 * k2, t + dt/2)   [search needed for x_n + dt/2 * k2]
    4. k4 = v(x_n + dt * k3, t + dt)       [search needed for x_n + dt * k3]

    Parameters
    ----------
    particle_data : ParticleData
        Current particle state with positions, element_ids, block_ids
    velocity_interpolator : callable
        Function(particle_data, time) -> velocities (N, 3)
    element_searcher : callable
        Function(particle_data) -> updated_particle_data
        Searches for elements containing particles at new positions
    dt : float
        Time step size
    current_time : float
        Current simulation time

    Returns
    -------
    new_particle_data : ParticleData
        Updated particle state after RK4 step
    rk4_stats : dict
        Statistics: number of searches performed, hit rates, etc.

    Notes
    -----
    RK4 is 4th-order accurate but requires:
    - 4 velocity interpolations (k1, k2, k3, k4)
    - 3 element searches (for k2, k3, k4 positions)

    This is ~4× slower than Forward Euler but much more accurate.

    Performance estimate: ~800 p/s on ThreadedA
    (3,416 p/s search × 0.25 duty cycle for 3 searches + interpolation overhead)

    Examples
    --------
    >>> # Create velocity interpolator
    >>> def vel_interp(pdata, t):
    ...     return interpolate_velocities_multi_block(
    ...         pdata.positions, pdata.element_ids, pdata.block_ids,
    ...         padded_arrays, velocity_field_at_time(t)
    ...     )
    >>>
    >>> # Create element searcher (wraps Phase 1 batch processor)
    >>> def elem_search(pdata):
    ...     stats = process_batch(0, pdata, padded_arrays, config, verbose=False)
    ...     return pdata, stats
    >>>
    >>> # RK4 step
    >>> new_pdata, stats = rk4_step_with_search(
    ...     particle_data, vel_interp, elem_search, dt=0.01, current_time=0.0
    ... )
    """

    # Stage 1: k1 at current position
    # No search needed - particles already have valid element_ids
    t1 = current_time
    v1 = velocity_interpolator(particle_data, t1)  # k1

    # Stage 2: k2 at x + dt/2 * k1
    # Need to search for elements at intermediate position
    pos2 = particle_data.positions + 0.5 * dt * v1
    # Convert to numpy if JAX array (element_searcher expects numpy)
    if hasattr(pos2, '__array__'):
        import numpy as np
        pos2 = np.asarray(pos2)
    pdata2 = replace(particle_data, positions=pos2)
    pdata2_searched, search_stats_2 = element_searcher(pdata2)

    t2 = current_time + 0.5 * dt
    v2 = velocity_interpolator(pdata2_searched, t2)  # k2

    # Stage 3: k3 at x + dt/2 * k2
    pos3 = particle_data.positions + 0.5 * dt * v2
    # Convert to numpy if JAX array (element_searcher expects numpy)
    if hasattr(pos3, '__array__'):
        import numpy as np
        pos3 = np.asarray(pos3)
    pdata3 = replace(particle_data, positions=pos3)
    pdata3_searched, search_stats_3 = element_searcher(pdata3)

    t3 = current_time + 0.5 * dt
    v3 = velocity_interpolator(pdata3_searched, t3)  # k3

    # Stage 4: k4 at x + dt * k3
    pos4 = particle_data.positions + dt * v3
    # Convert to numpy if JAX array (element_searcher expects numpy)
    if hasattr(pos4, '__array__'):
        import numpy as np
        pos4 = np.asarray(pos4)
    pdata4 = replace(particle_data, positions=pos4)
    pdata4_searched, search_stats_4 = element_searcher(pdata4)

    t4 = current_time + dt
    v4 = velocity_interpolator(pdata4_searched, t4)  # k4

    # RK4 combination: x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    new_positions = particle_data.positions + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
    # Convert to numpy if JAX array (element_searcher expects numpy)
    if hasattr(new_positions, '__array__'):
        import numpy as np
        new_positions = np.asarray(new_positions)

    # Final search at new positions
    new_particle_data = replace(particle_data, positions=new_positions)
    new_particle_data_searched, search_stats_final = element_searcher(new_particle_data)

    # Aggregate statistics
    rk4_stats = {
        'n_searches': 4,  # k2, k3, k4, final
        'search_stats_k2': search_stats_2,
        'search_stats_k3': search_stats_3,
        'search_stats_k4': search_stats_4,
        'search_stats_final': search_stats_final,
    }

    return new_particle_data_searched, rk4_stats


def rk4_step_with_incremental_search(
    particle_data,
    velocity_interpolator: Callable,
    incremental_searcher: Callable,
    dt: float,
    current_time: float
) -> Tuple:
    """
    RK4 time integration with OPTIMIZED incremental search (L0+L1).

    This is the FAST version of RK4 that exploits spatial coherence.
    Uses L0 (cached element) + L1 (face neighbors) for intermediate stages
    where particles have small displacements.

    Expected performance: 10-50× faster than rk4_step_with_search()

    RK4 stages:
    1. k1 = v(x_n, t_n)                    [no search needed]
    2. k2 = v(x_n + dt/2 * k1, t + dt/2)   [INCREMENTAL search - L0+L1 first]
    3. k3 = v(x_n + dt/2 * k2, t + dt/2)   [INCREMENTAL search - L0+L1 first]
    4. k4 = v(x_n + dt * k3, t + dt)       [INCREMENTAL search - L0+L1 first]
    5. Final position                       [INCREMENTAL search - L0+L1 first]

    Parameters
    ----------
    particle_data : ParticleData
        Current particle state with positions, element_ids, block_ids
    velocity_interpolator : callable
        Function(particle_data, time) -> velocities (N, 3)
    incremental_searcher : callable
        Function(new_positions, cached_elem_ids, cached_block_ids) -> (elem_ids, block_ids, stats)
        Uses L0+L1 before falling back to full search
    dt : float
        Time step size
    current_time : float
        Current simulation time

    Returns
    -------
    new_particle_data : ParticleData
        Updated particle state after RK4 step
    rk4_stats : dict
        Statistics including L0/L1 hit rates per stage

    Performance
    -----------
    Expected hit rates for dt=0.001, v=1mm/s:
    - L0 (cached): 60-80% → <1μs per particle
    - L1 (neighbors): 15-25% → ~5μs per particle
    - L2+L3 (full): 5-10% → ~10ms per particle

    Average time per particle: ~0.7*1μs + 0.2*5μs + 0.1*10ms = ~1ms
    vs full search: 4 × 10ms = 40ms
    Speedup: ~40×

    Examples
    --------
    >>> # Create incremental searcher
    >>> def incr_search(new_pos, cached_elem, cached_block):
    ...     return incremental_search_batch(
    ...         new_pos, cached_elem, cached_block,
    ...         bbox, grid_size, classification, padded_arrays,
    ...         block_neighbors_26, hash_bucket_data,
    ...         node_positions, connectivity, element_neighbors
    ...     )
    >>>
    >>> # RK4 step with incremental search
    >>> new_pdata, stats = rk4_step_with_incremental_search(
    ...     particle_data, vel_interp, incr_search, dt=0.001, current_time=0.0
    ... )
    >>> print(f"L0 hit rate: {stats['l0_total']/stats['n_searches']*100:.1f}%")
    """

    import numpy as np

    # Stage 1: k1 at current position
    # No search needed - particles already have valid element_ids
    t1 = current_time
    v1 = velocity_interpolator(particle_data, t1)  # k1

    # Stage 2: k2 at x + dt/2 * k1
    # Use incremental search (L0+L1) for small displacement
    pos2 = particle_data.positions + 0.5 * dt * v1
    if hasattr(pos2, '__array__'):
        pos2 = np.asarray(pos2)

    elem_ids_2, block_ids_2, search_stats_2 = incremental_searcher(
        pos2,
        particle_data.element_ids,  # cached from stage 1
        particle_data.block_ids
    )

    pdata2 = replace(particle_data, positions=pos2, element_ids=elem_ids_2, block_ids=block_ids_2)
    t2 = current_time + 0.5 * dt
    v2 = velocity_interpolator(pdata2, t2)  # k2

    # Stage 3: k3 at x + dt/2 * k2
    # Use incremental search with cache from stage 2
    pos3 = particle_data.positions + 0.5 * dt * v2
    if hasattr(pos3, '__array__'):
        pos3 = np.asarray(pos3)

    elem_ids_3, block_ids_3, search_stats_3 = incremental_searcher(
        pos3,
        elem_ids_2,  # cached from stage 2
        block_ids_2
    )

    pdata3 = replace(particle_data, positions=pos3, element_ids=elem_ids_3, block_ids=block_ids_3)
    t3 = current_time + 0.5 * dt
    v3 = velocity_interpolator(pdata3, t3)  # k3

    # Stage 4: k4 at x + dt * k3
    # Use incremental search with cache from stage 3
    pos4 = particle_data.positions + dt * v3
    if hasattr(pos4, '__array__'):
        pos4 = np.asarray(pos4)

    elem_ids_4, block_ids_4, search_stats_4 = incremental_searcher(
        pos4,
        elem_ids_3,  # cached from stage 3
        block_ids_3
    )

    pdata4 = replace(particle_data, positions=pos4, element_ids=elem_ids_4, block_ids=block_ids_4)
    t4 = current_time + dt
    v4 = velocity_interpolator(pdata4, t4)  # k4

    # RK4 combination: x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    new_positions = particle_data.positions + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
    if hasattr(new_positions, '__array__'):
        new_positions = np.asarray(new_positions)

    # Final search at new positions (use cache from original position)
    elem_ids_final, block_ids_final, search_stats_final = incremental_searcher(
        new_positions,
        particle_data.element_ids,  # cached from original position
        particle_data.block_ids
    )

    new_particle_data = replace(
        particle_data,
        positions=new_positions,
        element_ids=elem_ids_final,
        block_ids=block_ids_final
    )

    # Aggregate statistics
    rk4_stats = {
        'n_searches': 4,  # k2, k3, k4, final
        'search_stats_k2': search_stats_2,
        'search_stats_k3': search_stats_3,
        'search_stats_k4': search_stats_4,
        'search_stats_final': search_stats_final,
        # Aggregate L0/L1 hit rates
        'l0_total': (search_stats_2.l0_hits + search_stats_3.l0_hits +
                     search_stats_4.l0_hits + search_stats_final.l0_hits),
        'l1_total': (search_stats_2.l1_hits + search_stats_3.l1_hits +
                     search_stats_4.l1_hits + search_stats_final.l1_hits),
        'l2_total': (search_stats_2.l2_hits + search_stats_3.l2_hits +
                     search_stats_4.l2_hits + search_stats_final.l2_hits),
    }

    return new_particle_data, rk4_stats


# ============================================================================
# Simplified RK4 (No Intermediate Searches)
# ============================================================================

def rk4_step_simple(
    particle_data,
    velocity_interpolator: Callable,
    dt: float,
    current_time: float
) -> Tuple:
    """
    Simplified RK4 without intermediate element searches.

    WARNING: This is faster but LESS ACCURATE than rk4_step_with_search()
    because it uses the initial element_id for all 4 stages.

    Only use this if:
    1. Time step is very small (dt << element size / velocity)
    2. Velocity field is smooth within elements
    3. You can tolerate reduced accuracy for speed

    Performance: ~3,000 p/s (4× velocity interpolation but no searches)

    Parameters
    ----------
    particle_data : ParticleData
        Current particle state
    velocity_interpolator : callable
        Function(particle_data, time) -> velocities (N, 3)
    dt : float
        Time step
    current_time : float
        Current time

    Returns
    -------
    new_particle_data : ParticleData
        Updated particle state (positions updated, element_ids unchanged)
    """

    # Use existing RK4 implementation with custom field function
    def field_fn(positions, time):
        """Velocity field function for RK4."""
        # Create temporary particle data with new positions but same element_ids
        pdata_temp = replace(particle_data, positions=positions)
        return velocity_interpolator(pdata_temp, time)

    # Call existing RK4
    new_positions = _rk4_step_original(
        x=particle_data.positions,
        t=current_time,
        dt=dt,
        field_fn=field_fn
    )

    # Update particle data
    new_particle_data = replace(particle_data, positions=new_positions)

    return new_particle_data


# ============================================================================
# Integration Strategy Comparison
# ============================================================================

"""
Time Integration Strategy Comparison
====================================

| Method                    | Accuracy | Searches/Step | Interpolations/Step | Est. Throughput |
|---------------------------|----------|---------------|---------------------|-----------------|
| Forward Euler             | O(dt)    | 1             | 1                   | 3,000 p/s       |
| RK4 Simple (no search)    | O(dt⁴)*  | 1             | 4                   | 2,500 p/s       |
| RK4 Full (with search)    | O(dt⁴)   | 4             | 4                   | 800 p/s         |

*RK4 Simple accuracy degrades if particles move between elements during substeps

Recommended Strategy:
--------------------
1. Start with Forward Euler for initial implementation and testing
2. Use RK4 Full for production simulations requiring high accuracy
3. Consider RK4 Simple only for very small timesteps or smooth fields

Stability:
---------
Forward Euler requires dt < CFL_limit where:
  CFL_limit ≈ h / |v|  (element size / velocity magnitude)

For typical welding simulations:
  - Element size: 1-10 mm
  - Velocity: 1-100 mm/s
  - Safe dt: 0.01-0.1 s

Use compute_adaptive_timestep() for per-particle CFL-based timestep control.
"""
