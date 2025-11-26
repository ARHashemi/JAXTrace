"""
GPU-Fused RK4 Time Integration

This module implements a fully GPU-resident RK4 integration where all intermediate
states (positions, velocities, element IDs) remain on GPU throughout all 4 RK4 stages.

Key optimization: Eliminate CPU-GPU transfers between RK4 stages by keeping everything
on GPU from initial upload to final download.

Performance impact:
- Baseline: 10-20 MB transfers per timestep (upload/download each stage)
- Fused: ~2 MB transfers per timestep (upload initial state, download final state)
- Expected speedup: 2-3× overall throughput due to reduced transfer overhead
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, Optional

from jaxtrace.gpu.tracking.mesh_data_gpu import MeshDataGPU
from jaxtrace.gpu.search.level0_cached import point_in_tet_jax
from jaxtrace.gpu.search.incremental_search_vectorized import (
    search_level0_vectorized,
    search_level1_extended_vectorized,
    search_level1_multihop_vectorized
)


@dataclass
class RK4GPUState:
    """
    GPU-resident state for RK4 integration.

    All arrays are JAX DeviceArrays living on GPU.
    """
    positions: jax.Array      # (N, 3) float32
    element_ids: jax.Array    # (N,) int32
    velocities: jax.Array     # (N, 3) float32 - current velocity
    active_mask: jax.Array    # (N,) bool - active particles


@jax.jit
def interpolate_velocity_batch_gpu(
    positions_gpu: jax.Array,      # (N, 3)
    element_ids_gpu: jax.Array,    # (N,)
    mesh_gpu_connectivity: jax.Array,
    mesh_gpu_node_positions: jax.Array,
    velocity_field_gpu: jax.Array
) -> jax.Array:
    """
    Batch velocity interpolation entirely on GPU.

    This is a JIT-compiled version that stays on GPU.
    No CPU-GPU transfers within this function.

    Parameters
    ----------
    positions_gpu : jax.Array, shape (N, 3)
        Particle positions (GPU-resident)
    element_ids_gpu : jax.Array, shape (N,)
        Element IDs for each particle (GPU-resident)
    mesh_gpu_connectivity : jax.Array
        Element connectivity (GPU-resident)
    mesh_gpu_node_positions : jax.Array
        Node positions (GPU-resident)
    velocity_field_gpu : jax.Array
        Velocity at nodes (GPU-resident)

    Returns
    -------
    velocities_gpu : jax.Array, shape (N, 3)
        Interpolated velocities (GPU-resident)
    """
    def interpolate_single(position, element_id):
        """Interpolate velocity at a single particle."""
        # Get element connectivity (4 nodes for tet)
        # Cast element_id to int32 for indexing
        elem_id_int = element_id.astype(jnp.int32)
        elem_nodes = mesh_gpu_connectivity[elem_id_int]

        # Cast elem_nodes to int32 for indexing (connectivity might be float32)
        elem_nodes_int = elem_nodes.astype(jnp.int32)

        # Get node coordinates and velocities by indexing individual nodes
        # Extract each node ID explicitly to avoid any indexing ambiguity
        # elem_nodes_int has shape (4,) containing node IDs [n0, n1, n2, n3]
        n0 = elem_nodes_int[0]
        n1 = elem_nodes_int[1]
        n2 = elem_nodes_int[2]
        n3 = elem_nodes_int[3]

        # Get coordinates for each node (each is shape (3,))
        p0 = mesh_gpu_node_positions[n0]  # (3,)
        p1 = mesh_gpu_node_positions[n1]  # (3,)
        p2 = mesh_gpu_node_positions[n2]  # (3,)
        p3 = mesh_gpu_node_positions[n3]  # (3,)

        # Get velocities for each node (each is shape (3,))
        v0 = velocity_field_gpu[n0]  # (3,)
        v_1 = velocity_field_gpu[n1]  # (3,)
        v_2 = velocity_field_gpu[n2]  # (3,)
        v_3 = velocity_field_gpu[n3]  # (3,)

        # Compute barycentric coordinates using vectors from p0
        vec1 = p1 - p0
        vec2 = p2 - p0
        vec3 = p3 - p0

        A = jnp.stack([vec1, vec2, vec3], axis=1)  # (3, 3)
        dp = position - p0
        lambda_123 = jnp.linalg.solve(A, dp)  # (3,)
        lambda_0 = 1.0 - jnp.sum(lambda_123)

        lambdas = jnp.array([lambda_0, lambda_123[0], lambda_123[1], lambda_123[2]])  # (4,)

        # Interpolate velocity
        velocity = lambda_0 * v0 + lambda_123[0] * v_1 + lambda_123[1] * v_2 + lambda_123[2] * v_3  # (3,)

        return velocity

    # Vectorize over all particles
    return jax.vmap(interpolate_single)(positions_gpu, element_ids_gpu)


def create_search_gpu_fused(n_hops: int = 3):
    """
    Create a JIT-compiled GPU search function with specified hop count.

    Parameters
    ----------
    n_hops : int, default=3
        Number of hops for L1 neighbor search:
        - 2: ~20 neighbors (95-98% hit rate, fastest)
        - 3: ~84 neighbors (98-99.5% hit rate, recommended)
        - 4: ~340 neighbors (99.5-99.9% hit rate, most thorough)

    Returns
    -------
    search_func : callable
        JIT-compiled search function
    """
    @jax.jit
    def search_gpu_fused_impl(
        positions_gpu: jax.Array,      # (N, 3)
        cached_element_ids_gpu: jax.Array,  # (N,)
        mesh_gpu_node_positions: jax.Array,
        mesh_gpu_connectivity: jax.Array,
        mesh_gpu_element_neighbors: jax.Array
    ) -> jax.Array:
        """
        Fused GPU search: L0 + L1 multi-hop, all on GPU.

        No CPU-GPU transfers. Returns updated element IDs on GPU.

        Parameters
        ----------
        positions_gpu : jax.Array, shape (N, 3)
            Particle positions (GPU-resident)
        cached_element_ids_gpu : jax.Array, shape (N,)
            Cached element IDs from previous step (GPU-resident)
        mesh_gpu_* : jax.Array
            GPU-resident mesh data

        Returns
        -------
        element_ids_gpu : jax.Array, shape (N,)
            Updated element IDs (GPU-resident)
        """
        # L0: Check cached elements
        element_ids_l0 = search_level0_vectorized(
            positions_gpu,
            cached_element_ids_gpu,
            mesh_gpu_node_positions,
            mesh_gpu_connectivity
        )

        # L1: Check neighbors for ALL particles (vectorized, multi-hop)
        # For particles that succeeded in L0, L1 will return same result or -1
        # For particles that failed L0, L1 searches neighbors
        element_ids_l1 = search_level1_multihop_vectorized(
            positions_gpu,
            cached_element_ids_gpu,
            mesh_gpu_element_neighbors,
            mesh_gpu_node_positions,
            mesh_gpu_connectivity,
            n_hops=n_hops
        )

        # Merge results: use L0 if found, else use L1
        element_ids_gpu = jnp.where(element_ids_l0 >= 0, element_ids_l0, element_ids_l1)

        return element_ids_gpu

    return search_gpu_fused_impl


# Default search function (3-hop)
search_gpu_fused = create_search_gpu_fused(n_hops=3)


@jax.jit
def rk4_stage_gpu(
    pos_gpu: jax.Array,            # (N, 3) - current positions
    elem_ids_gpu: jax.Array,       # (N,) - current element IDs
    v_prev_gpu: jax.Array,         # (N, 3) - velocity from previous stage
    dt: float,
    alpha: float,                   # RK4 coefficient (0.5 for k2/k3, 1.0 for k4)
    mesh_gpu_connectivity: jax.Array,
    mesh_gpu_node_positions: jax.Array,
    mesh_gpu_element_neighbors: jax.Array,
    velocity_field_gpu: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Single RK4 stage entirely on GPU.

    Computes: pos_new = pos + alpha*dt*v_prev
    Then: elem_ids_new = search(pos_new, elem_ids)
    Then: v_new = interpolate(pos_new, elem_ids_new)

    All operations stay on GPU.

    Returns
    -------
    pos_new_gpu : jax.Array, shape (N, 3)
    elem_ids_new_gpu : jax.Array, shape (N,)
    v_new_gpu : jax.Array, shape (N, 3)
    """
    # Compute new positions
    pos_new_gpu = pos_gpu + alpha * dt * v_prev_gpu

    # Search for new element IDs
    elem_ids_new_gpu = search_gpu_fused(
        pos_new_gpu,
        elem_ids_gpu,  # Use previous elem_ids as cache
        mesh_gpu_node_positions,
        mesh_gpu_connectivity,
        mesh_gpu_element_neighbors
    )

    # Interpolate velocity at new positions
    v_new_gpu = interpolate_velocity_batch_gpu(
        pos_new_gpu,
        elem_ids_new_gpu,
        mesh_gpu_connectivity,
        mesh_gpu_node_positions,
        velocity_field_gpu
    )

    return pos_new_gpu, elem_ids_new_gpu, v_new_gpu


@jax.jit
def rk4_step_gpu_fused(
    positions_initial_gpu: jax.Array,      # (N, 3)
    element_ids_initial_gpu: jax.Array,    # (N,)
    dt: float,
    mesh_gpu_connectivity: jax.Array,
    mesh_gpu_node_positions: jax.Array,
    mesh_gpu_element_neighbors: jax.Array,
    velocity_field_gpu: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """
    Complete RK4 step entirely on GPU.

    All 4 RK4 stages execute on GPU without CPU-GPU transfers.
    Only the initial state is uploaded and final state is downloaded.

    RK4 formula:
    k1 = v(x_n, t_n)
    k2 = v(x_n + dt/2 * k1, t_n + dt/2)
    k3 = v(x_n + dt/2 * k2, t_n + dt/2)
    k4 = v(x_n + dt * k3, t_n + dt)
    x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    Parameters
    ----------
    positions_initial_gpu : jax.Array, shape (N, 3)
        Initial particle positions (GPU-resident)
    element_ids_initial_gpu : jax.Array, shape (N,)
        Initial element IDs (GPU-resident)
    dt : float
        Time step size
    mesh_gpu_* : jax.Array
        GPU-resident mesh data
    velocity_field_gpu : jax.Array
        GPU-resident velocity field

    Returns
    -------
    positions_final_gpu : jax.Array, shape (N, 3)
        Final particle positions (GPU-resident)
    element_ids_final_gpu : jax.Array, shape (N,)
        Final element IDs (GPU-resident)
    """
    # Stage 1: k1 at x_n
    v1_gpu = interpolate_velocity_batch_gpu(
        positions_initial_gpu,
        element_ids_initial_gpu,
        mesh_gpu_connectivity,
        mesh_gpu_node_positions,
        velocity_field_gpu
    )

    # Stage 2: k2 at x_n + dt/2 * k1
    pos2_gpu, elem_ids_2_gpu, v2_gpu = rk4_stage_gpu(
        positions_initial_gpu,
        element_ids_initial_gpu,
        v1_gpu,
        dt,
        0.5,  # alpha for k2
        mesh_gpu_connectivity,
        mesh_gpu_node_positions,
        mesh_gpu_element_neighbors,
        velocity_field_gpu
    )

    # Stage 3: k3 at x_n + dt/2 * k2
    pos3_gpu, elem_ids_3_gpu, v3_gpu = rk4_stage_gpu(
        positions_initial_gpu,
        elem_ids_2_gpu,  # Use elem_ids from stage 2
        v2_gpu,
        dt,
        0.5,  # alpha for k3
        mesh_gpu_connectivity,
        mesh_gpu_node_positions,
        mesh_gpu_element_neighbors,
        velocity_field_gpu
    )

    # Stage 4: k4 at x_n + dt * k3
    pos4_gpu, elem_ids_4_gpu, v4_gpu = rk4_stage_gpu(
        positions_initial_gpu,
        elem_ids_3_gpu,  # Use elem_ids from stage 3
        v3_gpu,
        dt,
        1.0,  # alpha for k4
        mesh_gpu_connectivity,
        mesh_gpu_node_positions,
        mesh_gpu_element_neighbors,
        velocity_field_gpu
    )

    # RK4 combination: x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    positions_final_gpu = positions_initial_gpu + (dt / 6.0) * (
        v1_gpu + 2.0*v2_gpu + 2.0*v3_gpu + v4_gpu
    )

    # Final search at new positions
    element_ids_final_gpu = search_gpu_fused(
        positions_final_gpu,
        element_ids_initial_gpu,  # Use initial elem_ids as cache
        mesh_gpu_node_positions,
        mesh_gpu_connectivity,
        mesh_gpu_element_neighbors
    )

    return positions_final_gpu, element_ids_final_gpu


def rk4_step_gpu_fused_wrapper(
    positions: np.ndarray,
    element_ids: np.ndarray,
    dt: float,
    mesh_gpu: MeshDataGPU,
    velocity_field: np.ndarray,
    n_hops: int = 3
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Wrapper for GPU-fused RK4 that handles CPU-GPU transfers.

    This function:
    1. Uploads initial state to GPU once
    2. Calls fully GPU-resident RK4
    3. Downloads final state from GPU once

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Initial particle positions (CPU)
    element_ids : np.ndarray, shape (N,)
        Initial element IDs (CPU)
    dt : float
        Time step size
    mesh_gpu : MeshDataGPU
        GPU-resident mesh data
    velocity_field : np.ndarray
        Velocity field (will be uploaded to GPU)
    n_hops : int, default=3
        Number of hops for L1 neighbor search (2-4)

    Returns
    -------
    positions_final : np.ndarray, shape (N, 3)
        Final particle positions (CPU)
    element_ids_final : np.ndarray, shape (N,)
        Final element IDs (CPU)
    stats : dict
        Timing statistics
    """
    # Create search function with specified hop count
    search_func = create_search_gpu_fused(n_hops=n_hops)

    # Create RK4 function with this search
    @jax.jit
    def rk4_fused_with_search(
        positions_gpu,
        element_ids_gpu,
        dt,
        connectivity_gpu,
        node_positions_gpu,
        element_neighbors_gpu,
        velocity_field_gpu
    ):
        """GPU-fused RK4 with configurable search."""
        # Stage 1: k1 = f(t, y)
        element_ids_k1 = search_func(
            positions_gpu,
            element_ids_gpu,
            node_positions_gpu,
            connectivity_gpu,
            element_neighbors_gpu
        )
        velocities_k1 = interpolate_velocity_batch_gpu(
            positions_gpu,
            element_ids_k1,
            connectivity_gpu,
            node_positions_gpu,
            velocity_field_gpu
        )
        positions_k1 = positions_gpu + 0.5 * dt * velocities_k1

        # Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
        element_ids_k2 = search_func(
            positions_k1,
            element_ids_k1,
            node_positions_gpu,
            connectivity_gpu,
            element_neighbors_gpu
        )
        velocities_k2 = interpolate_velocity_batch_gpu(
            positions_k1,
            element_ids_k2,
            connectivity_gpu,
            node_positions_gpu,
            velocity_field_gpu
        )
        positions_k2 = positions_gpu + 0.5 * dt * velocities_k2

        # Stage 3: k3 = f(t + dt/2, y + dt/2 * k2)
        element_ids_k3 = search_func(
            positions_k2,
            element_ids_k2,
            node_positions_gpu,
            connectivity_gpu,
            element_neighbors_gpu
        )
        velocities_k3 = interpolate_velocity_batch_gpu(
            positions_k2,
            element_ids_k3,
            connectivity_gpu,
            node_positions_gpu,
            velocity_field_gpu
        )
        positions_k3 = positions_gpu + dt * velocities_k3

        # Stage 4: k4 = f(t + dt, y + dt * k3)
        element_ids_k4 = search_func(
            positions_k3,
            element_ids_k3,
            node_positions_gpu,
            connectivity_gpu,
            element_neighbors_gpu
        )
        velocities_k4 = interpolate_velocity_batch_gpu(
            positions_k3,
            element_ids_k4,
            connectivity_gpu,
            node_positions_gpu,
            velocity_field_gpu
        )

        # Final position: y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        positions_final_gpu = positions_gpu + (dt / 6.0) * (
            velocities_k1 + 2.0 * velocities_k2 + 2.0 * velocities_k3 + velocities_k4
        )

        # Final search at new positions
        element_ids_final_gpu = search_func(
            positions_final_gpu,
            element_ids_gpu,
            node_positions_gpu,
            connectivity_gpu,
            element_neighbors_gpu
        )

        return positions_final_gpu, element_ids_final_gpu
    t_total = time.time()

    # Upload initial state to GPU (ONE upload)
    t_upload = time.time()
    positions_gpu = jax.device_put(positions.astype(np.float32))
    element_ids_gpu = jax.device_put(element_ids.astype(np.int32))
    velocity_field_gpu = jax.device_put(velocity_field.astype(np.float32))
    t_upload = time.time() - t_upload

    # Execute GPU-fused RK4 (all on GPU, no transfers)
    t_compute = time.time()
    positions_final_gpu, element_ids_final_gpu = rk4_fused_with_search(
        positions_gpu,
        element_ids_gpu,
        dt,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        mesh_gpu.element_neighbors,
        velocity_field_gpu
    )
    # Force GPU computation to complete
    positions_final_gpu.block_until_ready()
    t_compute = time.time() - t_compute

    # Download final state from GPU (ONE download)
    t_download = time.time()
    positions_final = np.array(positions_final_gpu, dtype=np.float32)
    element_ids_final = np.array(element_ids_final_gpu, dtype=np.int32)
    t_download = time.time() - t_download

    t_total = time.time() - t_total

    stats = {
        'time_upload': t_upload,
        'time_compute': t_compute,
        'time_download': t_download,
        'time_total': t_total,
        'n_particles': len(positions)
    }

    return positions_final, element_ids_final, stats


def rk4_step_gpu_fused_for_production(
    particle_data,
    velocity_field: np.ndarray,
    dt: float,
    mesh_gpu: MeshDataGPU,
    current_time: float = 0.0,
    n_hops: int = 3
):
    """
    GPU-fused RK4 wrapper for production script.

    Matches the interface of rk4_step_with_incremental_search() but keeps
    everything on GPU for maximum performance.

    This eliminates 8 CPU-GPU round trips per timestep:
    - OLD: 5× interpolation + 3× search = 8 round trips
    - NEW: 1× upload + GPU computation + 1× download = 2 transfers

    Parameters
    ----------
    particle_data : ParticleData
        Current particle state
    velocity_field : np.ndarray
        Velocity field at nodes
    dt : float
        Time step size
    mesh_gpu : MeshDataGPU
        GPU-resident mesh
    current_time : float
        Current time (not used, kept for interface compatibility)
    n_hops : int, default=3
        Number of hops for L1 neighbor search:
        - 2: ~20 neighbors (95-98% hit rate, fastest)
        - 3: ~84 neighbors (98-99.5% hit rate, recommended)
        - 4: ~340 neighbors (99.5-99.9% hit rate, most thorough)

    Returns
    -------
    new_particle_data : ParticleData
        Updated particle state
    rk4_stats : dict
        Statistics
    """
    from dataclasses import replace

    # Call GPU-fused RK4 with specified hop count
    positions_new, element_ids_new, stats = rk4_step_gpu_fused_wrapper(
        particle_data.positions,
        particle_data.element_ids,
        dt,
        mesh_gpu,
        velocity_field,
        n_hops=n_hops
    )

    # Update particle data (keep velocities unchanged - will be computed next step)
    new_particle_data = replace(
        particle_data,
        positions=positions_new,
        element_ids=element_ids_new
    )

    return new_particle_data, stats
