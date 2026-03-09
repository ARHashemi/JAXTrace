#!/usr/bin/env python3
"""
Diagnostic RK4 with Time-Dependent Velocity — Intermediate Position Export

IDENTICAL numerics to rk4_fully_fused_timedep.py, but returns all intermediate
positions and element IDs from the 5 RK4 sub-steps so we can diagnose exactly
where and why particles are lost.

For each particle, one RK4 step produces:
  Stage 0 (input):   pos_in,    elem_in     ← input position and cached element
  Stage 1 (k1):      elem_k1                ← search result at pos_in
                      vel_k1                 ← interpolated velocity
                      pos_k1 = pos + 0.5*dt*vel_k1
  Stage 2 (k2):      elem_k2                ← search result at pos_k1
                      vel_k2
                      pos_k2 = pos + 0.5*dt*vel_k2
  Stage 3 (k3):      elem_k3                ← search result at pos_k2
                      vel_k3
                      pos_k3 = pos + dt*vel_k3
  Stage 4 (k4):      elem_k4                ← search result at pos_k3
                      vel_k4
  Final:              pos_final = pos + (dt/6)*(k1+2*k2+2*k3+k4)
                      elem_final             ← search result at pos_final

Output arrays per step:
  positions_stages:   (N, 5, 3)  — [pos_in, pos_k1, pos_k2, pos_k3, pos_final]
  element_ids_stages: (N, 6)     — [elem_in, elem_k1, elem_k2, elem_k3, elem_k4, elem_final]
  velocities_stages:  (N, 4, 3)  — [vel_k1, vel_k2, vel_k3, vel_k4]

WARNING: This is ~30x more memory and ~2x slower than the production RK4.
         Use only for short diagnostic runs (10-50 steps).
"""

import jax
import jax.numpy as jnp
from typing import Optional
from jaxtrace.gpu.search.level0_cached import point_in_tet_jax
from jaxtrace.gpu.search.morton_global_search import (
    MeshGPUGlobalMorton,
    search_in_leaf_global,
    position_to_leaf_id_octree,
    position_to_leaf_id_linear,
    point_in_tet_gpu,
    search_L2_global_morton_single,
    search_L2_morton_incremental_single,
    search_L2_morton_neighbors_single,
    search_L2_morton_neighbors_enhanced,
    search_L2_morton_hierarchical_single
)
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import MeshAlignedOctreeGPU
from jaxtrace.gpu.search.mesh_aligned_point_location import (
    search_mesh_aligned_octree_single,
    search_mesh_aligned_octree_multi_local
)
from jaxtrace.gpu.search.mesh_aligned_morton_search import (
    MeshAlignedMortonGPU,
    search_L2_mesh_aligned_morton_single,
    search_L2_mesh_aligned_morton_incremental_single,
)
import jaxtrace.config as config


def create_rk4_diagnostic_timedep(
    mesh_gpu_connectivity: jax.Array,
    mesh_gpu_node_positions: jax.Array,
    mesh_gpu_element_neighbors: jax.Array,
    mesh_gpu_element_volumes: jax.Array,
    mesh_gpu_global_morton: MeshGPUGlobalMorton,
    n_hops: int = 3,
    l2_search_radius: int = 2,
    enable_l1_search: bool = True,
    l2_search_method: str = 'radius',
    l2_incremental_radii: tuple = (2, 5, 10),
    mesh_aligned_octree: Optional[MeshAlignedOctreeGPU] = None,
    mesh_aligned_morton: Optional[MeshAlignedMortonGPU] = None,
    mesh_aligned_octree_neighbors=None,
    mesh_aligned_octree_use_multi_local: bool = False,
    kdtree_gpu=None,
    kdtree_k_nearest: int = 3,
    kdtree_max_tests: int = 256
):
    """
    Create diagnostic RK4 integrator that exports intermediate sub-step data.

    Same parameters as create_rk4_fully_fused_timedep.
    Returns a function with additional outputs for diagnostic analysis.
    """

    # Pre-extract mesh arrays
    connectivity = mesh_gpu_connectivity
    node_positions = mesh_gpu_node_positions
    element_neighbors = mesh_gpu_element_neighbors

    # ========================================================================
    # Single-Particle Helpers (IDENTICAL to production)
    # ========================================================================

    def search_l0_single(pos, cached_elem_id):
        is_valid = (cached_elem_id >= 0) & (cached_elem_id < len(connectivity))
        inside = jnp.where(
            is_valid,
            point_in_tet_gpu(pos, cached_elem_id, connectivity, node_positions),
            False
        )
        return jnp.where(inside, cached_elem_id, jnp.int32(-1))

    def search_l1_single(pos, start_elem_id):
        current_elem = start_elem_id
        found = False

        start_elem_valid = start_elem_id >= 0
        start_volume = jnp.where(
            start_elem_valid,
            mesh_gpu_element_volumes[start_elem_id],
            jnp.float32(1.0)
        )

        neighbors_of_start = element_neighbors[jnp.where(start_elem_valid, start_elem_id, 0)]
        valid_neighbor_mask = neighbors_of_start >= 0
        neighbor_volumes = jnp.where(
            valid_neighbor_mask,
            mesh_gpu_element_volumes[jnp.where(valid_neighbor_mask, neighbors_of_start, 0)],
            start_volume
        )
        median_neighbor_volume = jnp.median(neighbor_volumes)
        size_ratio = start_volume / (median_neighbor_volume + 1e-10)

        n_hops_adaptive = jnp.where(
            size_ratio < 0.1,
            jnp.int32(6),
            jnp.int32(n_hops)
        )

        for hop_idx in range(6):
            hop_enabled = hop_idx < n_hops_adaptive
            should_search = (~found) & (current_elem >= 0) & hop_enabled
            neighbors = element_neighbors[jnp.where(should_search, current_elem, 0)]
            found_containing = jnp.int32(-1)

            for neighbor_idx in range(4):
                elem_id = neighbors[neighbor_idx]
                valid = elem_id >= 0
                check_this = (found_containing < 0) & valid
                inside = jnp.where(
                    check_this,
                    point_in_tet_gpu(pos, elem_id, connectivity, node_positions),
                    False
                )
                found_containing = jnp.where(
                    inside & check_this, elem_id, found_containing
                )

            first_valid_neighbor = jnp.where(
                jnp.any(neighbors >= 0),
                neighbors[jnp.argmax(neighbors >= 0)],
                current_elem
            )
            current_elem = jnp.where(
                should_search,
                jnp.where(found_containing >= 0, found_containing, first_valid_neighbor),
                current_elem
            )
            found = found | (found_containing >= 0)

        return jnp.where(found, current_elem, jnp.int32(-1))

    def search_l2_single(pos):
        use_mesh_aligned_morton = (
            config.L2_SEARCH_METHOD == 'mesh_aligned_morton' and
            mesh_aligned_morton is not None
        )
        use_mesh_aligned_octree = (
            config.L2_SEARCH_METHOD == 'mesh_aligned_octree' and
            mesh_aligned_octree is not None
        )
        use_mesh_aligned_neighbors = (
            config.L2_SEARCH_METHOD == 'mesh_aligned_neighbors' and
            mesh_aligned_octree_neighbors is not None
        )

        if use_mesh_aligned_morton:
            if l2_search_method == 'incremental':
                elem_id = search_L2_mesh_aligned_morton_incremental_single(
                    pos, mesh_aligned_morton,
                    radii=l2_incremental_radii, max_tests_per_cell=jnp.int32(256)
                )
            else:
                elem_id = search_L2_mesh_aligned_morton_single(
                    pos, mesh_aligned_morton,
                    search_radius=jnp.int32(l2_search_radius), max_tests_per_cell=jnp.int32(256)
                )
            return elem_id
        elif use_mesh_aligned_octree:
            if mesh_aligned_octree_use_multi_local:
                elem_id, _ = search_mesh_aligned_octree_multi_local(
                    pos, mesh_aligned_octree, max_tests=jnp.int32(600)
                )
            else:
                elem_id, _ = search_mesh_aligned_octree_single(
                    pos, mesh_aligned_octree, max_tests=jnp.int32(150)
                )
            return elem_id
        elif use_mesh_aligned_neighbors:
            from jaxtrace.gpu.search.mesh_aligned_search_with_neighbors import search_multi_level_with_precomputed_neighbors
            elem_id, _ = search_multi_level_with_precomputed_neighbors(
                pos, mesh_aligned_octree_neighbors,
                levels_to_try=(14, 13, 12), max_tests_per_cell=jnp.int32(20)
            )
            return elem_id
        elif l2_search_method == 'hierarchical':
            return search_L2_morton_hierarchical_single(pos, mesh_gpu_global_morton)
        elif l2_search_method == 'incremental':
            return search_L2_morton_incremental_single(pos, mesh_gpu_global_morton, radii=l2_incremental_radii)
        elif l2_search_method == 'neighbors':
            return search_L2_morton_neighbors_enhanced(pos, mesh_gpu_global_morton)
        elif l2_search_method == 'kdtree':
            from jaxtrace.gpu.search.kdtree_node_search import search_L2_kdtree_single
            return search_L2_kdtree_single(pos, kdtree_gpu, k_nearest=kdtree_k_nearest, max_tests=kdtree_max_tests)
        else:
            return search_L2_global_morton_single(pos, mesh_gpu_global_morton, l2_search_radius)

    def search_l0_l1_l2_single(pos, cached_elem_id):
        elem_l0 = search_l0_single(pos, cached_elem_id)
        found_l0 = elem_l0 >= 0

        if enable_l1_search:
            elem_l1 = jnp.where(found_l0, elem_l0, search_l1_single(pos, cached_elem_id))
            found_l1 = elem_l1 >= 0
            elem_final = jnp.where(found_l1, elem_l1, search_l2_single(pos))
        else:
            elem_final = jnp.where(found_l0, elem_l0, search_l2_single(pos))

        return elem_final

    def interpolate_velocity_single(pos, elem_id, velocity_field):
        valid = (elem_id >= 0) & (elem_id < len(connectivity))
        nodes_idx = connectivity[elem_id]
        nodes = node_positions[nodes_idx]
        node_vels = velocity_field[nodes_idx]

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

        det = d00 * (d11*d22 - d12*d12) - d01 * (d01*d22 - d02*d12) + d02 * (d01*d12 - d02*d11)
        det = jnp.where(jnp.abs(det) < 1e-12, 1e-12, det)

        b1 = (dp0 * (d11*d22 - d12*d12) - d01 * (dp1*d22 - dp2*d12) + d02 * (dp1*d12 - dp2*d11)) / det
        b2 = (d00 * (dp1*d22 - dp2*d12) - dp0 * (d01*d22 - d02*d12) + d02 * (d01*dp2 - d02*dp1)) / det
        b3 = (d00 * (d11*dp2 - d12*dp1) - d01 * (d01*dp2 - d02*dp1) + dp0 * (d01*d12 - d02*d11)) / det
        b0 = 1.0 - b1 - b2 - b3

        vel = b0 * node_vels[0] + b1 * node_vels[1] + b2 * node_vels[2] + b3 * node_vels[3]
        return jnp.where(valid, vel, jnp.zeros(3, dtype=jnp.float32))

    # ========================================================================
    # Diagnostic RK4 Step
    # ========================================================================

    @jax.jit
    def rk4_diagnostic_step_timedep(
        positions_gpu: jax.Array,         # (N, 3)
        element_ids_gpu: jax.Array,       # (N,)
        dt: float,
        velocity_fields_gpu: jax.Array,   # (n_timesteps, n_nodes, 3)
        time_idx: int
    ):
        """
        Diagnostic RK4 step — same numerics as production, but returns
        all intermediate positions, element IDs, and velocities.

        Returns:
            positions_final:    (N, 3)    — same as production
            element_ids_final:  (N,)      — same as production
            positions_stages:   (N, 5, 3) — [pos_in, pos_k1, pos_k2, pos_k3, pos_final]
            element_ids_stages: (N, 6)    — [elem_in, elem_k1, elem_k2, elem_k3, elem_k4, elem_final]
            velocities_stages:  (N, 4, 3) — [vel_k1, vel_k2, vel_k3, vel_k4]
        """
        n_timesteps = velocity_fields_gpu.shape[0]
        vel_idx = time_idx % n_timesteps
        velocity_field = velocity_fields_gpu[vel_idx]

        def rk4_single_particle_diagnostic(pos: jax.Array, elem_id: jax.Array):
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

            # Final
            pos_final = pos + (dt / 6.0) * (vel_k1 + 2.0*vel_k2 + 2.0*vel_k3 + vel_k4)
            elem_final = search_l0_l1_l2_single(pos_final, elem_k4)

            # Pack diagnostics
            positions_stages = jnp.stack([pos, pos_k1, pos_k2, pos_k3, pos_final])     # (5, 3)
            element_ids_stages = jnp.array([elem_id, elem_k1, elem_k2, elem_k3, elem_k4, elem_final])  # (6,)
            velocities_stages = jnp.stack([vel_k1, vel_k2, vel_k3, vel_k4])             # (4, 3)

            return pos_final, elem_final, positions_stages, element_ids_stages, velocities_stages

        # vmap over all particles
        (positions_final, element_ids_final,
         positions_stages, element_ids_stages, velocities_stages) = jax.vmap(
            rk4_single_particle_diagnostic
        )(positions_gpu, element_ids_gpu)

        return (positions_final, element_ids_final,
                positions_stages, element_ids_stages, velocities_stages)

    return rk4_diagnostic_step_timedep
