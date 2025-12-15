"""
Fully-Fused RK4 with Global Morton L2 Search

This module implements a **fully-fused** RK4 integrator where ALL operations
(L0/L1/L2 search + velocity interpolation) for ALL 5 RK4 stages are fused into
a SINGLE vmap over particles.

Key Differences from rk4_global_morton.py:
- **Single vmap**: One vmap(rk4_single_particle) instead of separate vmaps per stage
- **No CPU-GPU transfers**: Data stays on GPU between timesteps
- **Single kernel launch**: All 5 stages fused into one GPU kernel per timestep

Expected Performance:
- Throughput: 60-120k p/s (2-3× improvement over rk4_global_morton.py)
- Per-timestep overhead: <1ms (single kernel launch, no transfers)
- Memory: Same as rk4_global_morton.py (~40-100 MB)

Architecture:
- L0: Cached element (point-in-tet test)
- L1: Multi-hop neighbor search (3-5 hops)
- L2: Global Morton search (binary search + bounded leaf scan)
"""

import jax
import jax.numpy as jnp
from typing import Tuple

from jaxtrace.gpu.search.level0_cached import point_in_tet_jax
from jaxtrace.gpu.search.morton_global_search import (
    MeshGPUGlobalMorton,
    position_to_leaf_id_octree,
    search_in_leaf_global,
    point_in_tet_gpu
)


def create_rk4_fully_fused_global_morton(
    mesh_gpu_connectivity: jax.Array,
    mesh_gpu_node_positions: jax.Array,
    mesh_gpu_element_neighbors: jax.Array,
    mesh_gpu_global_morton: MeshGPUGlobalMorton,
    n_hops: int = 3,
    l2_search_radius: int = 2
):
    """
    Create fully-fused RK4 integrator with Global Morton L2 search.

    This version fuses ALL operations into a single vmap over particles:
    - All 5 RK4 stages (k1, k2, k3, k4, final)
    - All 5 L0+L1+L2 searches
    - All 4 velocity interpolations

    Result: SINGLE GPU kernel launch per timestep instead of 10+ launches.

    Parameters
    ----------
    mesh_gpu_connectivity : jax.Array
        Element connectivity array (n_elements, 4)
    mesh_gpu_node_positions : jax.Array
        Node position array (n_nodes, 3)
    mesh_gpu_element_neighbors : jax.Array
        Element neighbors array (n_elements, 4)
    mesh_gpu_global_morton : MeshGPUGlobalMorton
        GPU-resident global Morton structure
    n_hops : int, default=3
        Number of hops for L1 neighbor search
    l2_search_radius : int, default=2
        Search ±radius leaves in L2

    Returns
    -------
    rk4_step_func : callable
        Function with signature (positions_gpu, element_ids_gpu, dt, velocity_field_gpu, ...)
        Returns: (positions_final_gpu, element_ids_final_gpu)
    """

    # Pre-extract mesh arrays for direct access
    connectivity = mesh_gpu_connectivity
    node_positions = mesh_gpu_node_positions
    element_neighbors = mesh_gpu_element_neighbors

    # ============================================================================
    # Single-Particle Helper Functions
    # ============================================================================

    def search_l0_single(pos: jax.Array, cached_elem_id: jax.Array) -> jax.Array:
        """L0: Check if particle still in cached element (single particle)."""
        is_valid = (cached_elem_id >= 0) & (cached_elem_id < len(connectivity))

        # Get element nodes
        elem_nodes_idx = connectivity[jnp.where(is_valid, cached_elem_id, 0)]
        elem_nodes = node_positions[elem_nodes_idx]

        # Point-in-tet test
        inside = point_in_tet_jax(pos, elem_nodes, tolerance=1e-10)

        # Return cached element if inside, else -1
        return jnp.where(is_valid & inside, cached_elem_id, jnp.int32(-1))

    def search_l1_single(
        pos: jax.Array,
        start_elem_id: jax.Array
    ) -> jax.Array:
        """L1: Multi-hop neighbor search (single particle)."""
        # Start from L0 result
        current_elem = start_elem_id
        found = current_elem >= 0

        # Multi-hop search (unrolled for JIT)
        for _ in range(n_hops):
            # If already found, skip
            if_found = found

            # Get neighbors of current element
            neighbors = element_neighbors[jnp.where(~if_found & (current_elem >= 0), current_elem, 0)]

            # Search through neighbors
            def check_neighbor(elem_id):
                valid = elem_id >= 0
                elem_nodes_idx = connectivity[jnp.where(valid, elem_id, 0)]
                elem_nodes = node_positions[elem_nodes_idx]
                inside = point_in_tet_jax(pos, elem_nodes, tolerance=1e-10)
                return jnp.where(valid & inside, elem_id, jnp.int32(-1))

            # Check all 4 neighbors
            found_in_neighbors = jax.vmap(check_neighbor)(neighbors)

            # Find first valid neighbor
            found_mask = found_in_neighbors >= 0
            found_neighbor = jnp.where(
                jnp.any(found_mask),
                found_in_neighbors[jnp.argmax(found_mask)],
                jnp.int32(-1)
            )

            # Update current element if found
            current_elem = jnp.where(if_found, current_elem, found_neighbor)
            found = found | (found_neighbor >= 0)

        return current_elem

    def search_l2_single(pos: jax.Array) -> jax.Array:
        """L2: Global Morton search (single particle)."""
        # Find leaf containing this position
        leaf_id = position_to_leaf_id_octree(pos, mesh_gpu_global_morton)

        # Search center leaf
        elem_id = search_in_leaf_global(pos, leaf_id, mesh_gpu_global_morton)

        # If found, return
        found = elem_id >= 0

        # Search neighbor leaves (radius search)
        def search_neighbor_leaf(offset):
            # Skip center leaf (offset=0, already searched)
            skip_center = offset == 0

            neighbor_leaf = leaf_id + offset
            valid = (neighbor_leaf >= 0) & (neighbor_leaf < mesh_gpu_global_morton.n_leaves) & (~skip_center)
            result = jnp.where(
                valid,
                search_in_leaf_global(pos, neighbor_leaf, mesh_gpu_global_morton),
                jnp.int32(-1)
            )
            return result

        # Search ±radius leaves (including 0, but search_neighbor_leaf will skip it)
        offsets = jnp.arange(-l2_search_radius, l2_search_radius + 1)
        neighbor_results = jax.vmap(search_neighbor_leaf)(offsets)

        # Find first valid neighbor result
        neighbor_mask = neighbor_results >= 0
        found_in_neighbor = jnp.where(
            jnp.any(neighbor_mask),
            neighbor_results[jnp.argmax(neighbor_mask)],
            jnp.int32(-1)
        )

        # Return center result if found, else neighbor result
        return jnp.where(found, elem_id, found_in_neighbor)

    def search_l0_l1_l2_single(
        pos: jax.Array,
        cached_elem_id: jax.Array
    ) -> jax.Array:
        """Full L0+L1+L2 search hierarchy (single particle)."""
        # L0: Cached element
        elem_l0 = search_l0_single(pos, cached_elem_id)

        # L1: Multi-hop neighbors (only if L0 failed)
        elem_l1 = jnp.where(
            elem_l0 >= 0,
            elem_l0,
            search_l1_single(pos, cached_elem_id)
        )

        # L2: Global Morton (only if L0+L1 failed)
        elem_l2 = jnp.where(
            elem_l1 >= 0,
            elem_l1,
            search_l2_single(pos)
        )

        return elem_l2

    def interpolate_velocity_single(
        pos: jax.Array,
        elem_id: jax.Array,
        velocity_field: jax.Array
    ) -> jax.Array:
        """Interpolate velocity at position (single particle)."""
        valid = elem_id >= 0

        # Get element nodes
        elem_nodes_idx = connectivity[jnp.where(valid, elem_id, 0)]
        node_coords = node_positions[elem_nodes_idx]  # (4, 3)
        node_velocities = velocity_field[elem_nodes_idx]  # (4, 3)

        # Compute barycentric coordinates
        p0, p1, p2, p3 = node_coords[0], node_coords[1], node_coords[2], node_coords[3]
        v1, v2, v3 = p1 - p0, p2 - p0, p3 - p0
        vp = pos - p0

        # Solve [v1 v2 v3] * [b1, b2, b3]^T = vp
        det = (v1[0] * (v2[1] * v3[2] - v2[2] * v3[1]) -
               v1[1] * (v2[0] * v3[2] - v2[2] * v3[0]) +
               v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]))

        det_inv = jnp.where(jnp.abs(det) < 1e-12, 0.0, 1.0 / det)

        b1 = ((vp[0] * (v2[1] * v3[2] - v2[2] * v3[1]) -
               vp[1] * (v2[0] * v3[2] - v2[2] * v3[0]) +
               vp[2] * (v2[0] * v3[1] - v2[1] * v3[0])) * det_inv)

        b2 = ((v1[0] * (vp[1] * v3[2] - vp[2] * v3[1]) -
               v1[1] * (vp[0] * v3[2] - vp[2] * v3[0]) +
               v1[2] * (vp[0] * v3[1] - vp[1] * v3[0])) * det_inv)

        b3 = ((v1[0] * (v2[1] * vp[2] - v2[2] * vp[1]) -
               v1[1] * (v2[0] * vp[2] - v2[2] * vp[0]) +
               v1[2] * (v2[0] * vp[1] - v2[1] * vp[0])) * det_inv)

        b0 = 1.0 - b1 - b2 - b3

        # Interpolate velocity
        vel = (b0 * node_velocities[0] +
               b1 * node_velocities[1] +
               b2 * node_velocities[2] +
               b3 * node_velocities[3])

        # Return zero velocity if invalid
        return jnp.where(valid, vel, jnp.zeros(3, dtype=jnp.float32))

    # ============================================================================
    # Fully-Fused RK4 Function
    # ============================================================================

    @jax.jit
    def rk4_fully_fused_step(
        positions_gpu: jax.Array,        # (N, 3) float32
        element_ids_gpu: jax.Array,      # (N,) int32
        dt: float,
        velocity_field_gpu: jax.Array    # (n_nodes, 3) float32
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Fully-fused RK4 step (SINGLE vmap over particles).

        All 5 RK4 stages + 5 searches + 4 interpolations are fused into
        a single GPU kernel via vmap.

        Parameters
        ----------
        positions_gpu : jax.Array, shape (N, 3)
            Particle positions on GPU
        element_ids_gpu : jax.Array, shape (N,)
            Cached element IDs on GPU
        dt : float
            Time step size
        velocity_field_gpu : jax.Array, shape (n_nodes, 3)
            Velocity field on GPU

        Returns
        -------
        positions_final : jax.Array, shape (N, 3)
            Updated positions on GPU
        element_ids_final : jax.Array, shape (N,)
            Updated element IDs on GPU
        """

        def rk4_single_particle(pos: jax.Array, elem_id: jax.Array) -> Tuple[jax.Array, jax.Array]:
            """Fused RK4 for a single particle (all stages inline)."""

            # Stage 1: k1 = f(t, y)
            elem_k1 = search_l0_l1_l2_single(pos, elem_id)
            vel_k1 = interpolate_velocity_single(pos, elem_k1, velocity_field_gpu)
            pos_k1 = pos + 0.5 * dt * vel_k1

            # Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
            elem_k2 = search_l0_l1_l2_single(pos_k1, elem_k1)
            vel_k2 = interpolate_velocity_single(pos_k1, elem_k2, velocity_field_gpu)
            pos_k2 = pos + 0.5 * dt * vel_k2

            # Stage 3: k3 = f(t + dt/2, y + dt/2 * k2)
            elem_k3 = search_l0_l1_l2_single(pos_k2, elem_k2)
            vel_k3 = interpolate_velocity_single(pos_k2, elem_k3, velocity_field_gpu)
            pos_k3 = pos + dt * vel_k3

            # Stage 4: k4 = f(t + dt, y + dt * k3)
            elem_k4 = search_l0_l1_l2_single(pos_k3, elem_k3)
            vel_k4 = interpolate_velocity_single(pos_k3, elem_k4, velocity_field_gpu)

            # Final position: y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            pos_final = pos + (dt / 6.0) * (vel_k1 + 2.0 * vel_k2 + 2.0 * vel_k3 + vel_k4)

            # Search at final position for next timestep
            elem_final = search_l0_l1_l2_single(pos_final, elem_k4)

            return pos_final, elem_final

        # SINGLE vmap over all particles (fuses all stages)
        positions_final, element_ids_final = jax.vmap(rk4_single_particle)(
            positions_gpu, element_ids_gpu
        )

        return positions_final, element_ids_final

    return rk4_fully_fused_step
