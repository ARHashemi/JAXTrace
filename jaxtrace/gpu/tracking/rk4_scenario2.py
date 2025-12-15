"""
True Scenario #2: Layered Batched RK4 with Explicit Residual Filtering

This implementation follows the architecture where each subprocess is a separate
GPU-parallelized operation with no nested JIT/vmap/scan.

Key design:
- Each search level (L0, L1, L2) is a separate JIT-compiled function
- Residual filtering happens between levels using jnp.where
- No single monolithic JIT function wrapping the entire step
- Explicit GPU parallelization at each stage
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass, replace
from typing import Tuple, Dict

from jaxtrace.gpu.tracking.mesh_data_gpu import MeshDataGPU
from jaxtrace.gpu.particles import ParticleData


# ============================================================================
# Individual GPU-Parallel Search Functions (No Nesting)
# ============================================================================

@jax.jit
def search_L0_batch(
    positions: jax.Array,
    cached_element_ids: jax.Array,
    connectivity: jax.Array,
    node_positions: jax.Array
) -> jax.Array:
    """
    L0 search: Check if particles are still in cached elements.

    Pure GPU-parallel operation over all particles.
    No nested vmap/scan.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Particle positions
    cached_element_ids : jax.Array, shape (N,)
        Cached element IDs from previous timestep
    connectivity : jax.Array, shape (n_elements, 4)
        Element-to-node connectivity
    node_positions : jax.Array, shape (n_nodes, 3)
        Node coordinates

    Returns
    -------
    element_ids : jax.Array, shape (N,)
        Element IDs (>=0 if found, -1 if not in cached element)
    """
    from jaxtrace.gpu.search.octree_search_gpu import point_in_tet_jax

    def check_single_particle(position, element_id):
        """Check single particle in cached element."""
        # Cast element_id to int32 for indexing
        elem_id_int = element_id.astype(jnp.int32)
        elem_nodes = connectivity[elem_id_int]

        # Cast elem_nodes to int32 for indexing
        elem_nodes_int = elem_nodes.astype(jnp.int32)

        # Extract each node ID explicitly
        n0 = elem_nodes_int[0]
        n1 = elem_nodes_int[1]
        n2 = elem_nodes_int[2]
        n3 = elem_nodes_int[3]

        # Get coordinates for each node
        p0 = node_positions[n0]  # (3,)
        p1 = node_positions[n1]  # (3,)
        p2 = node_positions[n2]  # (3,)
        p3 = node_positions[n3]  # (3,)

        # Stack into tet_nodes array for point_in_tet_jax
        tet_nodes = jnp.stack([p0, p1, p2, p3])  # (4, 3)

        # Check if inside
        inside = point_in_tet_jax(position, tet_nodes, tolerance=1e-6)

        # Check validity
        is_valid = (element_id >= 0) & (element_id < len(connectivity))

        return jnp.where(is_valid & inside, element_id, jnp.int32(-1))

    # Vectorize over all particles
    return jax.vmap(check_single_particle)(positions, cached_element_ids)


def search_L1_batch(
    positions: jax.Array,
    cached_element_ids: jax.Array,
    element_neighbors: jax.Array,
    connectivity: jax.Array,
    node_positions: jax.Array,
    n_hops: int = 3
) -> jax.Array:
    """
    L1 search: Multi-hop neighbor search.

    Pure GPU-parallel operation over subset of particles.
    No nested vmap/scan (uses fixed unrolled hops).

    Creates JIT-compiled function with n_hops baked in at compile time.
    This avoids TracerBoolConversionError by evaluating n_hops outside JIT boundary.

    Parameters
    ----------
    positions : jax.Array, shape (N_residual, 3)
        Positions of particles that failed L0
    cached_element_ids : jax.Array, shape (N_residual,)
        Cached element IDs
    element_neighbors : jax.Array, shape (n_elements, 4)
        Face neighbor connectivity
    connectivity : jax.Array, shape (n_elements, 4)
        Element-to-node connectivity
    node_positions : jax.Array, shape (n_nodes, 3)
        Node coordinates
    n_hops : int, default=3
        Number of neighbor hops (fixed, not traced)

    Returns
    -------
    element_ids : jax.Array, shape (N_residual,)
        Element IDs (>=0 if found, -1 if not found)
    """
    from jaxtrace.gpu.search.octree_search_gpu import point_in_tet_jax

    @jax.jit
    def search_single_particle(pos, cached_id):
        """Search single particle in neighbors."""
        # Start with cached element
        result = jnp.int32(-1)

        # Get initial neighbors (hop 1)
        initial_valid = (cached_id >= 0) & (cached_id < len(element_neighbors))
        safe_cached_id = jnp.where(initial_valid, cached_id, 0)
        neighbors_hop1 = jnp.where(
            initial_valid,
            element_neighbors[safe_cached_id],
            jnp.full(4, -1, dtype=jnp.int32)
        )

        # Check hop 1 neighbors (vectorized over 4 neighbors)
        def check_neighbor(nbr_id):
            # Cast element_id to int32 for indexing
            nbr_id_int = nbr_id.astype(jnp.int32)
            elem_nodes = connectivity[nbr_id_int]

            # Cast elem_nodes to int32 for indexing
            elem_nodes_int = elem_nodes.astype(jnp.int32)

            # Extract each node ID explicitly
            n0 = elem_nodes_int[0]
            n1 = elem_nodes_int[1]
            n2 = elem_nodes_int[2]
            n3 = elem_nodes_int[3]

            # Get coordinates for each node
            p0 = node_positions[n0]  # (3,)
            p1 = node_positions[n1]  # (3,)
            p2 = node_positions[n2]  # (3,)
            p3 = node_positions[n3]  # (3,)

            # Stack into tet_nodes array for point_in_tet_jax
            tet_nodes = jnp.stack([p0, p1, p2, p3])  # (4, 3)

            inside = point_in_tet_jax(pos, tet_nodes, tolerance=1e-6)

            # Check validity
            valid = (nbr_id >= 0) & (nbr_id < len(connectivity))

            return jnp.where(valid & inside, nbr_id, jnp.int32(-1))

        results_hop1 = jax.vmap(check_neighbor)(neighbors_hop1)
        result = jnp.max(results_hop1)  # Take first found (max of -1 and valid IDs)

        # If n_hops >= 2, expand to hop 2
        if n_hops >= 2:
            # Get hop 2 neighbors (expand from hop 1)
            def get_hop2_from_hop1(hop1_id):
                valid = (hop1_id >= 0) & (hop1_id < len(element_neighbors))
                safe_id = jnp.where(valid, hop1_id, 0)
                return jnp.where(
                    valid,
                    element_neighbors[safe_id],
                    jnp.full(4, -1, dtype=jnp.int32)
                )

            neighbors_hop2_nested = jax.vmap(get_hop2_from_hop1)(neighbors_hop1)  # (4, 4)
            neighbors_hop2 = neighbors_hop2_nested.reshape(-1)  # (16,)

            # Only search if not found yet
            def check_hop2():
                results_hop2 = jax.vmap(check_neighbor)(neighbors_hop2)
                return jnp.max(results_hop2)

            result = jnp.where(result >= 0, result, check_hop2())

        # If n_hops >= 3, expand to hop 3
        if n_hops >= 3:
            # Expand from hop 2 to hop 3 (16 -> 64)
            def get_hop3_from_hop2(hop2_id):
                valid = (hop2_id >= 0) & (hop2_id < len(element_neighbors))
                safe_id = jnp.where(valid, hop2_id, 0)
                return jnp.where(
                    valid,
                    element_neighbors[safe_id],
                    jnp.full(4, -1, dtype=jnp.int32)
                )

            neighbors_hop2_for_expansion = neighbors_hop2_nested.reshape(-1)  # (16,)
            neighbors_hop3_nested = jax.vmap(get_hop3_from_hop2)(neighbors_hop2_for_expansion)  # (16, 4)
            neighbors_hop3 = neighbors_hop3_nested.reshape(-1)  # (64,)

            def check_hop3():
                results_hop3 = jax.vmap(check_neighbor)(neighbors_hop3)
                return jnp.max(results_hop3)

            result = jnp.where(result >= 0, result, check_hop3())

        return result

    # Vectorize over residual particles
    return jax.vmap(search_single_particle)(positions, cached_element_ids)


def search_L2_octree_batch(
    positions: jax.Array,
    octree_metadata: jax.Array,
    octree_elements: jax.Array,
    connectivity: jax.Array,
    node_positions: jax.Array,
    max_depth: int = 10
) -> jax.Array:
    """
    L2 search: Octree spatial search.

    Pure GPU-parallel operation over subset of particles.
    Uses scan internally but this is the ONLY place scan is used.

    Creates JIT-compiled function with max_depth baked in at compile time.
    This avoids TracerBoolConversionError by evaluating max_depth outside JIT boundary.

    Parameters
    ----------
    positions : jax.Array, shape (N_residual, 3)
        Positions of particles that failed L0+L1
    octree_metadata : jax.Array, shape (n_nodes, 15)
        Octree node metadata
    octree_elements : jax.Array, shape (n_nodes, max_leaf_size)
        Octree element arrays
    connectivity : jax.Array, shape (n_elements, 4)
        Element-to-node connectivity
    node_positions : jax.Array, shape (n_nodes, 3)
        Node coordinates
    max_depth : int, default=10
        Maximum octree traversal depth

    Returns
    -------
    element_ids : jax.Array, shape (N_residual,)
        Element IDs (>=0 if found, -1 if not found)
    """
    from jaxtrace.gpu.search.octree_search_gpu import (
        point_in_tet_jax,
        compute_octant,
        check_leaf_elements_vectorized
    )

    @jax.jit
    def search_single_particle(pos):
        """Search single particle in octree."""
        def step(carry, _):
            """Single octree traversal step."""
            node_id, found_id = carry

            # Load node metadata
            node_meta = octree_metadata[node_id]
            is_leaf = node_meta[0] > 0.5
            bbox_min = node_meta[1:4]
            bbox_max = node_meta[4:7]
            children = node_meta[7:15].astype(jnp.int32)

            # If leaf: Check elements
            def check_leaf(_):
                elements = octree_elements[node_id]
                return check_leaf_elements_vectorized(
                    pos, elements, node_positions, connectivity
                )

            # If branch: Select child
            def select_child(_):
                octant = compute_octant(pos, bbox_min, bbox_max)
                child_id = children[octant]
                return jnp.where(child_id >= 0, child_id, node_id)

            # Branch based on leaf status
            leaf_result = jax.lax.cond(
                is_leaf,
                check_leaf,
                lambda _: jnp.int32(-1),
                None
            )

            child_id = jax.lax.cond(
                is_leaf,
                lambda _: node_id.astype(jnp.int32),
                select_child,
                None
            )

            # Early exit: if found, keep node and result
            new_node_id = jnp.where(found_id >= 0, node_id, child_id)
            new_found_id = jnp.where(found_id >= 0, found_id, leaf_result)

            return (new_node_id, new_found_id), None

        # Scan for max_depth iterations
        (_, element_id), _ = jax.lax.scan(
            step,
            (jnp.int32(0), jnp.int32(-1)),
            None,
            length=max_depth
        )

        return element_id

    # Vectorize over residual particles
    return jax.vmap(search_single_particle)(positions)


@jax.jit
def interpolate_velocity_batch(
    positions: jax.Array,
    element_ids: jax.Array,
    connectivity: jax.Array,
    node_positions: jax.Array,
    velocity_field: jax.Array
) -> jax.Array:
    """Interpolate velocity at particle positions (GPU-parallel)."""
    def interpolate_single(position, element_id):
        """Interpolate velocity at a single particle."""
        # Get element connectivity (4 nodes for tet)
        # Cast element_id to int32 for indexing
        elem_id_int = element_id.astype(jnp.int32)
        elem_nodes = connectivity[elem_id_int]

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
        p0 = node_positions[n0]  # (3,)
        p1 = node_positions[n1]  # (3,)
        p2 = node_positions[n2]  # (3,)
        p3 = node_positions[n3]  # (3,)

        # Get velocities for each node (each is shape (3,))
        v0 = velocity_field[n0]  # (3,)
        v_1 = velocity_field[n1]  # (3,)
        v_2 = velocity_field[n2]  # (3,)
        v_3 = velocity_field[n3]  # (3,)

        # Compute barycentric coordinates using vectors from p0
        vec1 = p1 - p0
        vec2 = p2 - p0
        vec3 = p3 - p0

        A = jnp.stack([vec1, vec2, vec3], axis=1)  # (3, 3)
        dp = position - p0
        lambda_123 = jnp.linalg.solve(A, dp)  # (3,)
        lambda_0 = 1.0 - jnp.sum(lambda_123)

        # Interpolate velocity
        velocity = lambda_0 * v0 + lambda_123[0] * v_1 + lambda_123[1] * v_2 + lambda_123[2] * v_3  # (3,)

        return velocity

    # Vectorize over all particles
    return jax.vmap(interpolate_single)(positions, element_ids)


# ============================================================================
# True Scenario #2: Layered RK4 Step
# ============================================================================

def rk4_step_scenario2(
    particle_data: ParticleData,
    velocity_field_gpu: jax.Array,
    dt: float,
    mesh_gpu: MeshDataGPU,
    octree_metadata: jax.Array,
    octree_elements: jax.Array,
    n_hops: int = 3,
    max_octree_depth: int = 10,
    current_time: float = 0.0
) -> Tuple[ParticleData, dict]:
    """
    True Scenario #2: RK4 step with explicit layered search and residual filtering.

    Architecture:
    - Each subprocess is a separate GPU-parallelized JIT function
    - Residual filtering happens between levels
    - No single monolithic JIT wrapping everything
    - No nested vmap/scan except within individual level functions

    Parameters
    ----------
    particle_data : ParticleData
        Particle data with positions, element_ids, velocities, etc.
    velocity_field_gpu : jax.Array
        Velocity field on GPU (already uploaded)
    dt : float
        Time step size
    mesh_gpu : MeshDataGPU
        GPU-resident mesh data
    octree_metadata : jax.Array
        Octree metadata on GPU
    octree_elements : jax.Array
        Octree elements on GPU
    n_hops : int
        Number of hops for L1 search
    max_octree_depth : int
        Maximum octree depth
    current_time : float
        Current time (for logging)

    Returns
    -------
    particle_data_updated : ParticleData
        Updated particle data
    stats : dict
        Statistics with timing and hit rates
    """
    t_total = time.time()

    # Extract data
    positions = particle_data.positions
    element_ids = particle_data.element_ids
    velocities = particle_data.velocities
    n_particles = len(positions)

    # Upload to GPU if needed
    t_upload = time.time()
    if isinstance(positions, np.ndarray):
        positions_gpu = jax.device_put(positions)
    else:
        positions_gpu = positions

    if isinstance(element_ids, np.ndarray):
        element_ids_gpu = jax.device_put(element_ids)
    else:
        element_ids_gpu = element_ids
    t_upload = time.time() - t_upload

    t_compute = time.time()

    # ========================================================================
    # Stage k1
    # ========================================================================

    # Interpolate velocity at current position (no search needed)
    velocities_k1 = interpolate_velocity_batch(
        positions_gpu,
        element_ids_gpu,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        velocity_field_gpu
    )

    # Calculate positions_k1
    positions_k1 = positions_gpu + 0.5 * dt * velocities_k1

    # L0 search for positions_k1 (all particles)
    elem_ids_k1_l0 = search_L0_batch(
        positions_k1,
        element_ids_gpu,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions
    )

    # Find L0 residuals
    unfound_l0_k1 = elem_ids_k1_l0 < 0
    n_residual_l0_k1 = jnp.sum(unfound_l0_k1)

    # L1 search for L0 residuals only
    # Filter to only unfound particles (boolean indexing OUTSIDE JIT is allowed)
    # NOTE: Always run search (no Python if check to avoid GPU→CPU sync)
    elem_ids_k1_l1_filtered = search_L1_batch(
        positions_k1[unfound_l0_k1],  # FILTERED: only unfound particles
        element_ids_gpu[unfound_l0_k1],  # FILTERED: only unfound cached IDs
        mesh_gpu.element_neighbors,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        n_hops
    )

    # Scatter results back to full array
    elem_ids_k1_l1_full = jnp.full(n_particles, -1, dtype=jnp.int32)
    elem_ids_k1_l1_full = elem_ids_k1_l1_full.at[unfound_l0_k1].set(elem_ids_k1_l1_filtered)

    # Merge L0 and L1 results
    elem_ids_k1_l0_l1 = jnp.where(elem_ids_k1_l0 >= 0, elem_ids_k1_l0, elem_ids_k1_l1_full)

    # Find L1 residuals
    unfound_l1_k1 = elem_ids_k1_l0_l1 < 0
    n_residual_l1_k1 = jnp.sum(unfound_l1_k1)

    # L2 search for L1 residuals only
    # NOTE: Always run search (no Python if check to avoid GPU→CPU sync)
    elem_ids_k1_l2_filtered = search_L2_octree_batch(
        positions_k1[unfound_l1_k1],  # FILTERED: only L1 residuals
        octree_metadata,
        octree_elements,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        max_octree_depth
    )

    # Scatter results back
    elem_ids_k1_l2_full = jnp.full(n_particles, -1, dtype=jnp.int32)
    elem_ids_k1_l2_full = elem_ids_k1_l2_full.at[unfound_l1_k1].set(elem_ids_k1_l2_filtered)

    # Final element IDs for k1
    elem_ids_k1 = jnp.where(elem_ids_k1_l0_l1 >= 0, elem_ids_k1_l0_l1, elem_ids_k1_l2_full)

    # Interpolate velocity at positions_k1
    velocities_k2 = interpolate_velocity_batch(
        positions_k1,
        elem_ids_k1,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        velocity_field_gpu
    )

    # ========================================================================
    # Stage k2
    # ========================================================================

    # Calculate positions_k2
    positions_k2 = positions_gpu + 0.5 * dt * velocities_k2

    # L0 search for positions_k2
    elem_ids_k2_l0 = search_L0_batch(
        positions_k2,
        elem_ids_k1,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions
    )

    # Find L0 residuals
    unfound_l0_k2 = elem_ids_k2_l0 < 0
    n_residual_l0_k2 = jnp.sum(unfound_l0_k2)

    # L1 search - filtered
    # NOTE: Always run search (no Python if check to avoid GPU→CPU sync)
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

    # Merge L0 and L1
    elem_ids_k2_l0_l1 = jnp.where(elem_ids_k2_l0 >= 0, elem_ids_k2_l0, elem_ids_k2_l1_full)

    # Find L1 residuals
    unfound_l1_k2 = elem_ids_k2_l0_l1 < 0
    n_residual_l1_k2 = jnp.sum(unfound_l1_k2)

    # L2 search - filtered
    # NOTE: Always run search (no Python if check to avoid GPU→CPU sync)
    elem_ids_k2_l2_filtered = search_L2_octree_batch(
        positions_k2[unfound_l1_k2],
        octree_metadata,
        octree_elements,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        max_octree_depth
    )
    elem_ids_k2_l2_full = jnp.full(n_particles, -1, dtype=jnp.int32)
    elem_ids_k2_l2_full = elem_ids_k2_l2_full.at[unfound_l1_k2].set(elem_ids_k2_l2_filtered)

    # Final element IDs for k2
    elem_ids_k2 = jnp.where(elem_ids_k2_l0_l1 >= 0, elem_ids_k2_l0_l1, elem_ids_k2_l2_full)

    # Interpolate velocity at positions_k2
    velocities_k3 = interpolate_velocity_batch(
        positions_k2,
        elem_ids_k2,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        velocity_field_gpu
    )

    # ========================================================================
    # Stage k3
    # ========================================================================

    # Calculate positions_k3
    positions_k3 = positions_gpu + dt * velocities_k3

    # L0 search
    elem_ids_k3_l0 = search_L0_batch(
        positions_k3,
        elem_ids_k2,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions
    )

    # L1 search for residuals - filtered
    unfound_l0_k3 = elem_ids_k3_l0 < 0
    n_residual_l0_k3 = jnp.sum(unfound_l0_k3)

    # NOTE: Always run search (no Python if check to avoid GPU→CPU sync)
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

    # Merge
    elem_ids_k3_l0_l1 = jnp.where(elem_ids_k3_l0 >= 0, elem_ids_k3_l0, elem_ids_k3_l1_full)

    # L2 search for residuals - filtered
    unfound_l1_k3 = elem_ids_k3_l0_l1 < 0
    n_residual_l1_k3 = jnp.sum(unfound_l1_k3)

    # NOTE: Always run search (no Python if check to avoid GPU→CPU sync)
    elem_ids_k3_l2_filtered = search_L2_octree_batch(
        positions_k3[unfound_l1_k3],
        octree_metadata,
        octree_elements,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        max_octree_depth
    )
    elem_ids_k3_l2_full = jnp.full(n_particles, -1, dtype=jnp.int32)
    elem_ids_k3_l2_full = elem_ids_k3_l2_full.at[unfound_l1_k3].set(elem_ids_k3_l2_filtered)

    elem_ids_k3 = jnp.where(elem_ids_k3_l0_l1 >= 0, elem_ids_k3_l0_l1, elem_ids_k3_l2_full)

    # Interpolate velocity at positions_k3
    velocities_k4 = interpolate_velocity_batch(
        positions_k3,
        elem_ids_k3,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        velocity_field_gpu
    )

    # ========================================================================
    # RK4 Combination
    # ========================================================================

    positions_final_gpu = positions_gpu + (dt / 6.0) * (
        velocities_k1 + 2.0 * velocities_k2 + 2.0 * velocities_k3 + velocities_k4
    )

    # ========================================================================
    # Final Element Update
    # ========================================================================

    # L0 search
    elem_ids_final_l0 = search_L0_batch(
        positions_final_gpu,
        elem_ids_k3,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions
    )

    # L1 search for residuals - filtered
    unfound_l0_final = elem_ids_final_l0 < 0
    n_residual_l0_final = jnp.sum(unfound_l0_final)

    # NOTE: Always run search (no Python if check to avoid GPU→CPU sync)
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

    # Merge
    elem_ids_final_l0_l1 = jnp.where(elem_ids_final_l0 >= 0, elem_ids_final_l0, elem_ids_final_l1_full)

    # L2 search for residuals - filtered
    unfound_l1_final = elem_ids_final_l0_l1 < 0
    n_residual_l1_final = jnp.sum(unfound_l1_final)

    # NOTE: Always run search (no Python if check to avoid GPU→CPU sync)
    elem_ids_final_l2_filtered = search_L2_octree_batch(
        positions_final_gpu[unfound_l1_final],
        octree_metadata,
        octree_elements,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        max_octree_depth
    )
    elem_ids_final_l2_full = jnp.full(n_particles, -1, dtype=jnp.int32)
    elem_ids_final_l2_full = elem_ids_final_l2_full.at[unfound_l1_final].set(elem_ids_final_l2_filtered)

    elem_ids_final = jnp.where(elem_ids_final_l0_l1 >= 0, elem_ids_final_l0_l1, elem_ids_final_l2_full)

    # Force computation
    elem_ids_final.block_until_ready()
    t_compute = time.time() - t_compute

    # ========================================================================
    # Download Results
    # ========================================================================

    t_download = time.time()
    positions_final = np.array(positions_final_gpu)
    element_ids_final = np.array(elem_ids_final)
    t_download = time.time() - t_download

    t_total = time.time() - t_total

    # Update particle data
    particle_data_updated = ParticleData(
        positions=positions_final.astype(np.float32),
        element_ids=element_ids_final.astype(np.int32),
        velocities=velocities,  # Keep original velocities (not updated in RK4 step)
        block_ids=particle_data.block_ids,  # Keep block IDs
        active_mask=particle_data.active_mask  # Keep active mask
    )

    # Collect statistics
    stats = {
        'time_upload': t_upload,
        'time_compute': t_compute,
        'time_download': t_download,
        'time_total': t_total,
        'n_particles': len(positions),
        # Hit rates for each stage
        'k1_l0_hits': int(jnp.sum(~unfound_l0_k1)),
        'k1_l1_hits': int(jnp.sum(unfound_l0_k1 & ~unfound_l1_k1)),
        'k1_l2_hits': int(jnp.sum(unfound_l1_k1)),
        'k2_l0_hits': int(jnp.sum(~unfound_l0_k2)),
        'k2_l1_hits': int(jnp.sum(unfound_l0_k2 & ~unfound_l1_k2)),
        'k2_l2_hits': int(jnp.sum(unfound_l1_k2)),
        'final_l0_hits': int(jnp.sum(~unfound_l0_final)),
        'final_l1_hits': int(jnp.sum(unfound_l0_final & ~unfound_l1_final)),
        'final_l2_hits': int(jnp.sum(unfound_l1_final)),
    }

    return particle_data_updated, stats
