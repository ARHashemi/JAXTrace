"""
Temporally-Batched Scenario #2: Layered RK4 with GPU-Resident Data

Optimizations over rk4_scenario2.py:
1. Temporal batching: Process N timesteps in a batch with GPU-resident data
2. Eliminated CPU-GPU transfers between timesteps within a batch
3. Only download at end of batch (for export or statistics)

Key design:
- Accept GPU arrays as input, return GPU arrays as output
- No forced uploads/downloads inside the function
- Caller controls when to transfer (for batching)
"""

import time
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, Dict

from jaxtrace.gpu.tracking.mesh_data_gpu import MeshDataGPU
from jaxtrace.gpu.particles import ParticleData


# Import search functions from original scenario2
from jaxtrace.gpu.tracking.rk4_scenario2 import (
    search_L0_batch,
    search_L1_batch,
    search_L2_octree_batch,
    interpolate_velocity_batch
)


@dataclass
class RK4StatsScenario2:
    """Statistics for Scenario #2 RK4 step"""
    # k1 stage
    k1_l0_hits: int
    k1_l1_hits: int
    k1_l2_hits: int

    # k2 stage
    k2_l0_hits: int
    k2_l1_hits: int
    k2_l2_hits: int

    # k3 stage (not reported in original, but computed)
    k3_l0_hits: int
    k3_l1_hits: int
    k3_l2_hits: int

    # final stage
    final_l0_hits: int
    final_l1_hits: int
    final_l2_hits: int

    # Timing
    total_time: float


def rk4_step_scenario2_gpu_resident(
    positions_gpu: jax.Array,
    element_ids_gpu: jax.Array,
    velocity_field_gpu: jax.Array,
    dt: float,
    mesh_gpu: MeshDataGPU,
    octree_metadata_gpu: jax.Array,
    octree_elements_gpu: jax.Array,
    n_hops: int = 3,
    max_octree_depth: int = 15,
    current_time: float = 0.0
) -> Tuple[jax.Array, jax.Array, Dict]:
    """
    Single RK4 timestep with GPU-resident data (no forced CPU-GPU transfers).

    This is the core function for temporal batching. It:
    - Accepts JAX arrays on GPU (no upload)
    - Returns JAX arrays on GPU (no download)
    - Caller decides when to transfer data

    Parameters
    ----------
    positions_gpu : jax.Array, shape (N, 3)
        Particle positions on GPU
    element_ids_gpu : jax.Array, shape (N,)
        Cached element IDs on GPU
    velocity_field_gpu : jax.Array, shape (n_nodes, 3)
        Velocity field on GPU
    dt : float
        Timestep size
    mesh_gpu : MeshDataGPU
        Mesh data on GPU
    octree_metadata_gpu : jax.Array
        Octree metadata on GPU
    octree_elements_gpu : jax.Array
        Octree elements on GPU
    n_hops : int
        Number of neighbor hops for L1 search
    max_octree_depth : int
        Maximum octree depth for L2 search
    current_time : float
        Current simulation time (unused, for compatibility)

    Returns
    -------
    positions_final_gpu : jax.Array, shape (N, 3)
        Updated positions on GPU
    element_ids_final_gpu : jax.Array, shape (N,)
        Updated element IDs on GPU
    stats : Dict
        Search statistics
    """
    t_total = time.time()

    n_particles = len(positions_gpu)

    # ========================================================================
    # Stage k1: Evaluate at current position
    # ========================================================================

    # L0 search: Check cached elements
    elem_ids_k1_l0 = search_L0_batch(
        positions_gpu,
        element_ids_gpu,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions
    )
    n_k1_l0_hits = jnp.sum(elem_ids_k1_l0 >= 0)

    # L1 search: Multi-hop neighbors (filtered to unfound particles)
    unfound_l0_k1 = elem_ids_k1_l0 < 0
    n_residual_l0_k1 = jnp.sum(unfound_l0_k1)

    elem_ids_k1_l1_filtered = search_L1_batch(
        positions_gpu[unfound_l0_k1],
        element_ids_gpu[unfound_l0_k1],
        mesh_gpu.element_neighbors,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        n_hops
    )

    elem_ids_k1_l1_full = jnp.full(n_particles, -1, dtype=jnp.int32)
    elem_ids_k1_l1_full = elem_ids_k1_l1_full.at[unfound_l0_k1].set(elem_ids_k1_l1_filtered)

    # Merge L0 and L1
    elem_ids_k1_l0_l1 = jnp.where(elem_ids_k1_l0 >= 0, elem_ids_k1_l0, elem_ids_k1_l1_full)
    n_k1_l1_hits = jnp.sum((elem_ids_k1_l0 < 0) & (elem_ids_k1_l1_full >= 0))

    # L2 search: Octree fallback (filtered to unfound particles)
    unfound_l1_k1 = elem_ids_k1_l0_l1 < 0
    n_residual_l1_k1 = jnp.sum(unfound_l1_k1)

    elem_ids_k1_l2_filtered = search_L2_octree_batch(
        positions_gpu[unfound_l1_k1],
        octree_metadata_gpu,
        octree_elements_gpu,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        max_octree_depth
    )

    elem_ids_k1_l2_full = jnp.full(n_particles, -1, dtype=jnp.int32)
    elem_ids_k1_l2_full = elem_ids_k1_l2_full.at[unfound_l1_k1].set(elem_ids_k1_l2_filtered)

    elem_ids_k1 = jnp.where(elem_ids_k1_l0_l1 >= 0, elem_ids_k1_l0_l1, elem_ids_k1_l2_full)
    n_k1_l2_hits = jnp.sum((elem_ids_k1_l0_l1 < 0) & (elem_ids_k1_l2_full >= 0))

    # Interpolate velocity at k1
    velocities_k1 = interpolate_velocity_batch(
        positions_gpu,
        elem_ids_k1,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        velocity_field_gpu
    )

    # ========================================================================
    # Stage k2: Evaluate at t + dt/2, x + k1*dt/2
    # ========================================================================

    positions_k2 = positions_gpu + 0.5 * dt * velocities_k1

    # L0 search
    elem_ids_k2_l0 = search_L0_batch(
        positions_k2,
        elem_ids_k1,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions
    )
    n_k2_l0_hits = jnp.sum(elem_ids_k2_l0 >= 0)

    # L1 search (filtered)
    unfound_l0_k2 = elem_ids_k2_l0 < 0

    elem_ids_k2_l1_filtered = search_L1_batch(
        positions_k2[unfound_l0_k2],
        elem_ids_k1[unfound_l0_k2],
        mesh_gpu.element_neighbors,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        n_hops
    )

    elem_ids_k2_l1_full = jnp.full(n_particles, -1, dtype=jnp.int32)
    elem_ids_k2_l1_full = elem_ids_k2_l1_full.at[unfound_l0_k2].set(elem_ids_k2_l1_filtered)

    elem_ids_k2_l0_l1 = jnp.where(elem_ids_k2_l0 >= 0, elem_ids_k2_l0, elem_ids_k2_l1_full)
    n_k2_l1_hits = jnp.sum((elem_ids_k2_l0 < 0) & (elem_ids_k2_l1_full >= 0))

    # L2 search (filtered)
    unfound_l1_k2 = elem_ids_k2_l0_l1 < 0

    elem_ids_k2_l2_filtered = search_L2_octree_batch(
        positions_k2[unfound_l1_k2],
        octree_metadata_gpu,
        octree_elements_gpu,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        max_octree_depth
    )

    elem_ids_k2_l2_full = jnp.full(n_particles, -1, dtype=jnp.int32)
    elem_ids_k2_l2_full = elem_ids_k2_l2_full.at[unfound_l1_k2].set(elem_ids_k2_l2_filtered)

    elem_ids_k2 = jnp.where(elem_ids_k2_l0_l1 >= 0, elem_ids_k2_l0_l1, elem_ids_k2_l2_full)
    n_k2_l2_hits = jnp.sum((elem_ids_k2_l0_l1 < 0) & (elem_ids_k2_l2_full >= 0))

    # Interpolate velocity at k2
    velocities_k2 = interpolate_velocity_batch(
        positions_k2,
        elem_ids_k2,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        velocity_field_gpu
    )

    # ========================================================================
    # Stage k3: Evaluate at t + dt/2, x + k2*dt/2
    # ========================================================================

    positions_k3 = positions_gpu + 0.5 * dt * velocities_k2

    # L0 search
    elem_ids_k3_l0 = search_L0_batch(
        positions_k3,
        elem_ids_k2,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions
    )
    n_k3_l0_hits = jnp.sum(elem_ids_k3_l0 >= 0)

    # L1 search (filtered)
    unfound_l0_k3 = elem_ids_k3_l0 < 0

    elem_ids_k3_l1_filtered = search_L1_batch(
        positions_k3[unfound_l0_k3],
        elem_ids_k2[unfound_l0_k3],
        mesh_gpu.element_neighbors,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        n_hops
    )

    elem_ids_k3_l1_full = jnp.full(n_particles, -1, dtype=jnp.int32)
    elem_ids_k3_l1_full = elem_ids_k3_l1_full.at[unfound_l0_k3].set(elem_ids_k3_l1_filtered)

    elem_ids_k3_l0_l1 = jnp.where(elem_ids_k3_l0 >= 0, elem_ids_k3_l0, elem_ids_k3_l1_full)
    n_k3_l1_hits = jnp.sum((elem_ids_k3_l0 < 0) & (elem_ids_k3_l1_full >= 0))

    # L2 search (filtered)
    unfound_l1_k3 = elem_ids_k3_l0_l1 < 0

    elem_ids_k3_l2_filtered = search_L2_octree_batch(
        positions_k3[unfound_l1_k3],
        octree_metadata_gpu,
        octree_elements_gpu,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        max_octree_depth
    )

    elem_ids_k3_l2_full = jnp.full(n_particles, -1, dtype=jnp.int32)
    elem_ids_k3_l2_full = elem_ids_k3_l2_full.at[unfound_l1_k3].set(elem_ids_k3_l2_filtered)

    elem_ids_k3 = jnp.where(elem_ids_k3_l0_l1 >= 0, elem_ids_k3_l0_l1, elem_ids_k3_l2_full)
    n_k3_l2_hits = jnp.sum((elem_ids_k3_l0_l1 < 0) & (elem_ids_k3_l2_full >= 0))

    # Interpolate velocity at k3
    velocities_k3 = interpolate_velocity_batch(
        positions_k3,
        elem_ids_k3,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        velocity_field_gpu
    )

    # ========================================================================
    # Stage k4: Evaluate at t + dt, x + k3*dt
    # ========================================================================

    positions_k4 = positions_gpu + dt * velocities_k3

    # No search at k4 (not used in final position update)
    # Just interpolate velocity using k3 element IDs
    velocities_k4 = interpolate_velocity_batch(
        positions_k4,
        elem_ids_k3,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        velocity_field_gpu
    )

    # ========================================================================
    # Final Update: x_new = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    # ========================================================================

    positions_final_gpu = positions_gpu + (dt / 6.0) * (
        velocities_k1 + 2.0 * velocities_k2 + 2.0 * velocities_k3 + velocities_k4
    )

    # Final search for updated positions
    # L0 search
    elem_ids_final_l0 = search_L0_batch(
        positions_final_gpu,
        elem_ids_k3,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions
    )
    n_final_l0_hits = jnp.sum(elem_ids_final_l0 >= 0)

    # L1 search (filtered)
    unfound_l0_final = elem_ids_final_l0 < 0

    elem_ids_final_l1_filtered = search_L1_batch(
        positions_final_gpu[unfound_l0_final],
        elem_ids_k3[unfound_l0_final],
        mesh_gpu.element_neighbors,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        n_hops
    )

    elem_ids_final_l1_full = jnp.full(n_particles, -1, dtype=jnp.int32)
    elem_ids_final_l1_full = elem_ids_final_l1_full.at[unfound_l0_final].set(elem_ids_final_l1_filtered)

    elem_ids_final_l0_l1 = jnp.where(elem_ids_final_l0 >= 0, elem_ids_final_l0, elem_ids_final_l1_full)
    n_final_l1_hits = jnp.sum((elem_ids_final_l0 < 0) & (elem_ids_final_l1_full >= 0))

    # L2 search (filtered)
    unfound_l1_final = elem_ids_final_l0_l1 < 0

    elem_ids_final_l2_filtered = search_L2_octree_batch(
        positions_final_gpu[unfound_l1_final],
        octree_metadata_gpu,
        octree_elements_gpu,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        max_octree_depth
    )

    elem_ids_final_l2_full = jnp.full(n_particles, -1, dtype=jnp.int32)
    elem_ids_final_l2_full = elem_ids_final_l2_full.at[unfound_l1_final].set(elem_ids_final_l2_filtered)

    elem_ids_final_gpu = jnp.where(elem_ids_final_l0_l1 >= 0, elem_ids_final_l0_l1, elem_ids_final_l2_full)
    n_final_l2_hits = jnp.sum((elem_ids_final_l0_l1 < 0) & (elem_ids_final_l2_full >= 0))

    # ========================================================================
    # Return GPU arrays directly (no download)
    # ========================================================================

    t_total = time.time() - t_total

    # Build statistics dictionary (convert to Python int for compatibility)
    stats = {
        'k1_l0_hits': int(n_k1_l0_hits),
        'k1_l1_hits': int(n_k1_l1_hits),
        'k1_l2_hits': int(n_k1_l2_hits),
        'k2_l0_hits': int(n_k2_l0_hits),
        'k2_l1_hits': int(n_k2_l1_hits),
        'k2_l2_hits': int(n_k2_l2_hits),
        'k3_l0_hits': int(n_k3_l0_hits),
        'k3_l1_hits': int(n_k3_l1_hits),
        'k3_l2_hits': int(n_k3_l2_hits),
        'final_l0_hits': int(n_final_l0_hits),
        'final_l1_hits': int(n_final_l1_hits),
        'final_l2_hits': int(n_final_l2_hits),
        'total_time': t_total
    }

    return positions_final_gpu, elem_ids_final_gpu, stats


def rk4_temporal_batch_scenario2(
    positions_gpu: jax.Array,
    element_ids_gpu: jax.Array,
    velocity_field_gpu: jax.Array,
    dt: float,
    mesh_gpu: MeshDataGPU,
    octree_metadata_gpu: jax.Array,
    octree_elements_gpu: jax.Array,
    n_steps: int = 3,
    n_hops: int = 3,
    max_octree_depth: int = 15,
    start_time: float = 0.0
) -> Tuple[jax.Array, jax.Array, list]:
    """
    Process multiple timesteps in a batch with GPU-resident data.

    This function processes N consecutive timesteps without any CPU-GPU
    transfers between steps. Data stays on GPU throughout the batch.

    Parameters
    ----------
    positions_gpu : jax.Array, shape (N, 3)
        Initial particle positions on GPU
    element_ids_gpu : jax.Array, shape (N,)
        Initial cached element IDs on GPU
    velocity_field_gpu : jax.Array, shape (n_nodes, 3)
        Velocity field on GPU
    dt : float
        Timestep size
    mesh_gpu : MeshDataGPU
        Mesh data on GPU
    octree_metadata_gpu : jax.Array
        Octree metadata on GPU
    octree_elements_gpu : jax.Array
        Octree elements on GPU
    n_steps : int
        Number of timesteps in batch (default: 3)
    n_hops : int
        Number of neighbor hops for L1 search
    max_octree_depth : int
        Maximum octree depth for L2 search
    start_time : float
        Starting simulation time

    Returns
    -------
    positions_final_gpu : jax.Array, shape (N, 3)
        Final positions after N steps (on GPU)
    element_ids_final_gpu : jax.Array, shape (N,)
        Final element IDs after N steps (on GPU)
    all_stats : list of Dict
        List of statistics dictionaries (one per step)
    """
    all_stats = []

    # Keep data on GPU throughout the batch
    pos = positions_gpu
    elem_ids = element_ids_gpu

    for i in range(n_steps):
        pos, elem_ids, stats = rk4_step_scenario2_gpu_resident(
            pos,
            elem_ids,
            velocity_field_gpu,
            dt,
            mesh_gpu,
            octree_metadata_gpu,
            octree_elements_gpu,
            n_hops=n_hops,
            max_octree_depth=max_octree_depth,
            current_time=start_time + i * dt
        )
        all_stats.append(stats)

    return pos, elem_ids, all_stats
