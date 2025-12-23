#!/usr/bin/env python3
"""
Fully-Fused RK4 with Time-Dependent Velocity - Phase 5B Extension

This is a time-dependent version of the fully-fused RK4 integrator.
Uses cyclic velocity field sequence loaded on GPU for transient simulations.

Key features:
- All velocity timesteps pre-loaded on GPU (no per-step transfers)
- Cyclic indexing for periodic velocity sequences
- Zero performance overhead vs static velocity version
- Maintains single vmap architecture over all particles
"""

import jax
import jax.numpy as jnp
from jaxtrace.gpu.search.level0_cached import point_in_tet_jax
from jaxtrace.gpu.search.morton_global_search import (
    MeshGPUGlobalMorton,
    search_in_leaf_global,
    position_to_leaf_id_octree,
    position_to_leaf_id_linear,
    point_in_tet_gpu
)


def create_rk4_fully_fused_timedep(
    mesh_gpu_connectivity: jax.Array,
    mesh_gpu_node_positions: jax.Array,
    mesh_gpu_element_neighbors: jax.Array,
    mesh_gpu_global_morton: MeshGPUGlobalMorton,
    n_hops: int = 3,
    l2_search_radius: int = 2,
    enable_l1_search: bool = True
):
    """
    Create fully-fused RK4 integrator with time-dependent velocity.

    This version accepts a sequence of velocity fields and uses cyclic indexing
    to implement periodic/transient velocity boundary conditions.

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
    enable_l1_search : bool, default=True
        Enable L1 neighbor search. If False, search hierarchy becomes L0→L2 (skip L1).
        Useful for testing or when L1 is known to be ineffective (e.g., graded refinement).

    Returns
    -------
    rk4_step_func : callable
        Function with signature:
            (positions_gpu, element_ids_gpu, dt, velocity_fields_gpu, time_idx)
        Returns: (positions_final_gpu, element_ids_final_gpu)
    """

    # Pre-extract mesh arrays for direct access
    connectivity = mesh_gpu_connectivity
    node_positions = mesh_gpu_node_positions
    element_neighbors = mesh_gpu_element_neighbors

    # ============================================================================
    # Single-Particle Helper Functions (Time-Dependent)
    # ============================================================================

    def search_l0_single(pos: jax.Array, cached_elem_id: jax.Array) -> jax.Array:
        """L0: Check if particle still in cached element (single particle)."""
        is_valid = (cached_elem_id >= 0) & (cached_elem_id < len(connectivity))

        # Use GPU-optimized point-in-tet with JIT compilation and relative degeneracy threshold
        inside = jnp.where(
            is_valid,
            point_in_tet_gpu(pos, cached_elem_id, connectivity, node_positions),
            False
        )

        return jnp.where(inside, cached_elem_id, jnp.int32(-1))

    def search_l1_single(pos: jax.Array, start_elem_id: jax.Array) -> jax.Array:
        """L1: Multi-hop neighbor search with proper hopping (single particle).

        Note: L1 is only called when L0 fails, meaning start_elem_id does NOT
        contain the position. We start with found=False to force neighbor search.

        Multi-hop strategy:
        - If containing element found: stop and return it
        - If not found: advance to first valid neighbor for next hop
        - This allows traversing the neighbor graph (neighbors-of-neighbors)
        """
        current_elem = start_elem_id
        found = False  # Force neighbor search (L0 already verified non-containment)

        # Multi-hop search (unrolled for JIT)
        for _ in range(n_hops):
            should_search = (~found) & (current_elem >= 0)

            # Get neighbors of current element
            neighbors = element_neighbors[jnp.where(should_search, current_elem, 0)]

            # Search through neighbors
            def check_neighbor(elem_id):
                valid = elem_id >= 0
                # Use GPU-optimized point-in-tet with JIT compilation
                inside = jnp.where(
                    valid,
                    point_in_tet_gpu(pos, elem_id, connectivity, node_positions),
                    False
                )
                return jnp.where(inside, elem_id, jnp.int32(-1))

            # Check all neighbors (vmap over neighbor dimension)
            found_in_neighbors = jax.vmap(check_neighbor)(neighbors)
            found_mask = found_in_neighbors >= 0

            # Get first containing neighbor (if any)
            found_containing = jnp.where(
                jnp.any(found_mask),
                found_in_neighbors[jnp.argmax(found_mask)],
                jnp.int32(-1)
            )

            # MULTI-HOP FIX: Get first valid neighbor (even if point not inside) for next hop
            # This allows advancing through the neighbor graph
            first_valid_neighbor = jnp.where(
                jnp.any(neighbors >= 0),
                neighbors[jnp.argmax(neighbors >= 0)],
                current_elem  # Stay at current if no valid neighbors
            )

            # Update for next hop:
            # - If found containing element: use it and set found=True (stops hopping)
            # - If not found: advance to first_valid_neighbor for next hop
            current_elem = jnp.where(
                should_search,
                jnp.where(found_containing >= 0, found_containing, first_valid_neighbor),
                current_elem
            )
            found = found | (found_containing >= 0)

        # CRITICAL FIX: Return -1 if search failed (not found after all hops)
        # This ensures L2 fallback is triggered when L1 fails
        return jnp.where(found, current_elem, jnp.int32(-1))

    def search_l2_single(pos: jax.Array) -> jax.Array:
        """L2: Global Morton search (single particle)."""
        # Map position to leaf
        leaf_id = jnp.where(
            mesh_gpu_global_morton.table_depth > 0,
            position_to_leaf_id_octree(pos, mesh_gpu_global_morton),
            position_to_leaf_id_linear(pos, mesh_gpu_global_morton)
        )

        # Search center leaf
        elem_id = search_in_leaf_global(pos, leaf_id, mesh_gpu_global_morton)
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

        # Return center result if found, otherwise neighbor result
        return jnp.where(found, elem_id, found_in_neighbor)

    def search_l0_l1_l2_single(pos: jax.Array, cached_elem_id: jax.Array) -> jax.Array:
        """Full L0+L1+L2 search hierarchy for single particle."""
        # L0: Cached element
        elem_l0 = search_l0_single(pos, cached_elem_id)
        found_l0 = elem_l0 >= 0

        if enable_l1_search:
            # L1: Multi-hop neighbors (only if L0 failed)
            elem_l1 = jnp.where(
                found_l0,
                elem_l0,
                search_l1_single(pos, cached_elem_id)
            )
            found_l1 = elem_l1 >= 0

            # L2: Global Morton (only if L0+L1 failed)
            elem_final = jnp.where(
                found_l1,
                elem_l1,
                search_l2_single(pos)
            )
        else:
            # L1 disabled: L0→L2 search hierarchy
            # L2: Global Morton (only if L0 failed)
            elem_final = jnp.where(
                found_l0,
                elem_l0,
                search_l2_single(pos)
            )

        return elem_final

    def interpolate_velocity_single(
        pos: jax.Array,
        elem_id: jax.Array,
        velocity_field: jax.Array  # (n_nodes, 3) - single timestep
    ) -> jax.Array:
        """
        Barycentric velocity interpolation for single particle.

        Args:
            pos: (3,) particle position
            elem_id: scalar element ID
            velocity_field: (n_nodes, 3) velocity at nodes for this timestep

        Returns:
            vel: (3,) interpolated velocity
        """
        valid = (elem_id >= 0) & (elem_id < len(connectivity))

        # Get element nodes
        nodes_idx = connectivity[elem_id]  # (4,)
        nodes = node_positions[nodes_idx]  # (4, 3)
        node_vels = velocity_field[nodes_idx]  # (4, 3)

        # Barycentric coordinates
        v0 = nodes[1] - nodes[0]
        v1 = nodes[2] - nodes[0]
        v2 = nodes[3] - nodes[0]
        vp = pos - nodes[0]

        d00 = jnp.dot(v0, v0)
        d01 = jnp.dot(v0, v1)
        d02 = jnp.dot(v0, v2)
        d11 = jnp.dot(v1, v1)
        d12 = jnp.dot(v1, v2)
        d22 = jnp.dot(v2, v2)

        dp0 = jnp.dot(vp, v0)
        dp1 = jnp.dot(vp, v1)
        dp2 = jnp.dot(vp, v2)

        # Solve 3x3 system for barycentric coords
        det = d00 * (d11*d22 - d12*d12) - d01 * (d01*d22 - d02*d12) + d02 * (d01*d12 - d02*d11)
        det = jnp.where(jnp.abs(det) < 1e-12, 1e-12, det)

        b1 = (dp0 * (d11*d22 - d12*d12) - d01 * (dp1*d22 - dp2*d12) + d02 * (dp1*d12 - dp2*d11)) / det
        b2 = (d00 * (dp1*d22 - dp2*d12) - dp0 * (d01*d22 - d02*d12) + d02 * (d01*dp2 - d02*dp1)) / det
        b3 = (d00 * (d11*dp2 - d12*dp1) - d01 * (d01*dp2 - d02*dp1) + dp0 * (d01*d12 - d02*d11)) / det
        b0 = 1.0 - b1 - b2 - b3

        # Interpolate velocity
        vel = b0 * node_vels[0] + b1 * node_vels[1] + b2 * node_vels[2] + b3 * node_vels[3]

        return jnp.where(valid, vel, jnp.zeros(3, dtype=jnp.float32))

    # ============================================================================
    # Fully-Fused RK4 Step (Time-Dependent)
    # ============================================================================

    @jax.jit
    def rk4_fully_fused_step_timedep(
        positions_gpu: jax.Array,         # (N, 3) float32
        element_ids_gpu: jax.Array,       # (N,) int32
        dt: float,
        velocity_fields_gpu: jax.Array,   # (n_timesteps, n_nodes, 3) float32
        time_idx: int                      # Current time index (cycles with modulo)
    ):
        """
        Single RK4 timestep with time-dependent velocity (fully fused).

        All operations fused into single vmap over particles:
        - All 5 RK4 stages (k1, k2, k3, k4, final)
        - All 5 L0+L1+L2 searches
        - All 4 velocity interpolations

        Args:
            positions_gpu: (N, 3) particle positions
            element_ids_gpu: (N,) cached element IDs
            dt: timestep size
            velocity_fields_gpu: (n_timesteps, n_nodes, 3) velocity sequence
            time_idx: index into velocity sequence (cyclic with modulo)

        Returns:
            positions_final: (N, 3) updated positions
            element_ids_final: (N,) updated element IDs
        """
        n_timesteps = velocity_fields_gpu.shape[0]

        # Cyclic indexing for velocity
        vel_idx = time_idx % n_timesteps
        velocity_field = velocity_fields_gpu[vel_idx]

        # Single-particle RK4 with all stages fused
        def rk4_single_particle(pos: jax.Array, elem_id: jax.Array):
            """RK4 for single particle with all stages inline."""

            # Stage 1: k1 = f(t, y)
            elem_k1 = search_l0_l1_l2_single(pos, elem_id)
            vel_k1 = interpolate_velocity_single(pos, elem_k1, velocity_field)
            pos_k1 = pos + 0.5 * dt * vel_k1

            # Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
            elem_k2 = search_l0_l1_l2_single(pos_k1, elem_k1)
            vel_k2 = interpolate_velocity_single(pos_k1, elem_k2, velocity_field)
            pos_k2 = pos + 0.5 * dt * vel_k2

            # Stage 3: k3 = f(t + dt/2, y + dt/2 * k2)
            elem_k3 = search_l0_l1_l2_single(pos_k2, elem_k2)
            vel_k3 = interpolate_velocity_single(pos_k2, elem_k3, velocity_field)
            pos_k3 = pos + dt * vel_k3

            # Stage 4: k4 = f(t + dt, y + dt * k3)
            elem_k4 = search_l0_l1_l2_single(pos_k3, elem_k3)
            vel_k4 = interpolate_velocity_single(pos_k3, elem_k4, velocity_field)

            # Final position: y_n+1 = y_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            pos_final = pos + (dt / 6.0) * (vel_k1 + 2.0*vel_k2 + 2.0*vel_k3 + vel_k4)

            # Final element search
            elem_final = search_l0_l1_l2_single(pos_final, elem_k4)

            return pos_final, elem_final

        # SINGLE vmap over all particles (fully fused)
        positions_final, element_ids_final = jax.vmap(rk4_single_particle)(
            positions_gpu, element_ids_gpu
        )

        return positions_final, element_ids_final

    return rk4_fully_fused_step_timedep
