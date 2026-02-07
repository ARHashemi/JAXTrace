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


def create_rk4_fully_fused_timedep(
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
    mesh_aligned_octree_neighbors = None,  # Optional[MeshAlignedOctreeGPUWithNeighbors]
    mesh_aligned_octree_use_multi_local: bool = False,  # NEW: Use 2×2×2 local search
    kdtree_gpu = None,  # Optional[NodeKDTreeGPU]
    kdtree_k_nearest: int = 3,
    kdtree_max_tests: int = 256
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
    mesh_gpu_element_volumes : jax.Array
        Element volumes array (n_elements,) - used for adaptive L1 hop count
    mesh_gpu_global_morton : MeshGPUGlobalMorton
        GPU-resident global Morton structure
    n_hops : int, default=3
        Number of hops for L1 neighbor search
    l2_search_radius : int, default=2
        Search ±radius leaves in L2
    enable_l1_search : bool, default=True
        Enable L1 neighbor search. If False, search hierarchy becomes L0→L2 (skip L1).
        Useful for testing or when L1 is known to be ineffective (e.g., graded refinement).
    l2_search_method : str, default='radius'
        L2 search method:
        - 'radius': Linear ±radius search along Morton curve (original method)
        - 'incremental': Cascading radius search with configurable tiers
        - 'neighbors': Morton neighbor arithmetic (26 spatial neighbors at single depth)
        - 'hierarchical': Multi-depth Morton neighbors (depth 7 + depth 6 fallback)
                         Best for variable-depth octree leaves
    l2_incremental_radii : tuple, default=(2, 5, 10)
        Cascading radii for 'incremental' L2 search method (2-5 tiers supported).
        Each radius=R searches 2R+1 leaves (symmetric band: [-R,...,0,...,+R]).
        Example: (2, 5, 10) → search 5 leaves, then 11, then 21 if needed.
        Ignored if l2_search_method != 'incremental'.
    mesh_aligned_octree : Optional[MeshAlignedOctreeGPU], default=None
        GPU-resident mesh-aligned octree structure for L2 search.
        If provided and config.L2_SEARCH_METHOD == 'mesh_aligned_octree',
        uses mesh-aligned octree instead of Morton curve search.
        Only works with Kuhn tetrahedral meshes (axis-aligned tets).
        Provides ~100% searchability with ~5.9 elements per cell.

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
        """L1: Multi-hop neighbor search with ADAPTIVE hop count (Phase 1.3).

        Note: L1 is only called when L0 fails, meaning start_elem_id does NOT
        contain the position. We start with found=False to force neighbor search.

        Multi-hop strategy:
        - If containing element found: stop and return it
        - If not found: advance to first valid neighbor for next hop
        - This allows traversing the neighbor graph (neighbors-of-neighbors)

        Adaptive hop count (Phase 1.3 fix):
        - Detect refinement boundary crossings by comparing element volumes
        - If start element is 10× smaller than median neighbor → likely refined→coarse boundary
        - Use extended hop count (6) for boundary cases, normal (3) otherwise
        - This handles particles crossing from refined to coarse regions
        """
        current_elem = start_elem_id
        found = False  # Force neighbor search (L0 already verified non-containment)

        # PHASE 1.3: Adaptive hop count based on element size ratio
        # Detect refinement boundary by comparing start element volume with neighbor volumes
        start_elem_valid = start_elem_id >= 0
        start_volume = jnp.where(
            start_elem_valid,
            mesh_gpu_element_volumes[start_elem_id],
            jnp.float32(1.0)  # Default to avoid division issues
        )

        # Get neighbor volumes
        neighbors_of_start = element_neighbors[jnp.where(start_elem_valid, start_elem_id, 0)]
        valid_neighbor_mask = neighbors_of_start >= 0

        # Compute neighbor volumes (use safe indexing)
        neighbor_volumes = jnp.where(
            valid_neighbor_mask,
            mesh_gpu_element_volumes[jnp.where(valid_neighbor_mask, neighbors_of_start, 0)],
            start_volume  # Use start volume for invalid neighbors
        )

        # Median neighbor volume (robust to outliers)
        median_neighbor_volume = jnp.median(neighbor_volumes)

        # Size ratio: start_volume / median_neighbor_volume
        # If ratio < 0.1 → start element is 10× SMALLER than neighbors (refined→coarse boundary)
        size_ratio = start_volume / (median_neighbor_volume + 1e-10)  # Avoid division by zero

        # Adaptive hop count:
        # - Small→Large transition (size_ratio < 0.1): Use 6 hops (extended search)
        # - Normal case: Use n_hops (default 3)
        n_hops_adaptive = jnp.where(
            size_ratio < 0.1,
            jnp.int32(6),  # Extended search for refinement boundary
            jnp.int32(n_hops)  # Normal search
        )

        # Multi-hop search (unrolled for maximum hop count = 6)
        # Use masking to skip extra iterations when n_hops_adaptive < 6
        for hop_idx in range(6):
            # Skip this hop if hop_idx >= n_hops_adaptive (adaptive masking)
            hop_enabled = hop_idx < n_hops_adaptive
            should_search = (~found) & (current_elem >= 0) & hop_enabled

            # Get neighbors of current element
            neighbors = element_neighbors[jnp.where(should_search, current_elem, 0)]

            # FIXED: Remove nested vmap - use sequential search with jnp.where
            # This eliminates vmap-in-vmap overhead and allows logical early-exit
            found_containing = jnp.int32(-1)

            # Unroll 4-neighbor check (sequential, not vmapped)
            for neighbor_idx in range(4):
                elem_id = neighbors[neighbor_idx]
                valid = elem_id >= 0

                # Only check if not found yet and valid
                check_this = (found_containing < 0) & valid

                inside = jnp.where(
                    check_this,
                    point_in_tet_gpu(pos, elem_id, connectivity, node_positions),
                    False
                )

                # Update found_containing if inside
                found_containing = jnp.where(
                    inside & check_this,
                    elem_id,
                    found_containing
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
        """L2: Global search (single particle) - method selected by config."""
        # Check if mesh-aligned Morton should be used (HYBRID APPROACH - NEW)
        use_mesh_aligned_morton = (
            config.L2_SEARCH_METHOD == 'mesh_aligned_morton' and
            mesh_aligned_morton is not None
        )

        # Check if direct mesh-aligned octree should be used
        use_mesh_aligned_octree = (
            config.L2_SEARCH_METHOD == 'mesh_aligned_octree' and
            mesh_aligned_octree is not None
        )

        # Check if mesh-aligned octree with neighbors should be used (Option B - NEW)
        use_mesh_aligned_neighbors = (
            config.L2_SEARCH_METHOD == 'mesh_aligned_neighbors' and
            mesh_aligned_octree_neighbors is not None
        )

        if use_mesh_aligned_morton:
            # MESH-ALIGNED MORTON (HYBRID): Morton radius search over cell centers
            # Combines intrinsic mesh structure (5.9 elem/cell) + proven radius search (93-98% retention)
            # Morton codes from CELL CENTERS (not element centroids)
            # Handles elements spanning multiple cells via radius search
            # Expected ~98% retention with ~30 tests (radius=2: 5 cells × 5.9 elem/cell)
            if l2_search_method == 'incremental':
                # Use incremental radius search
                elem_id = search_L2_mesh_aligned_morton_incremental_single(
                    pos,
                    mesh_aligned_morton,
                    radii=l2_incremental_radii,
                    max_tests_per_cell=jnp.int32(256)
                )
            else:
                # Default: fixed radius search
                elem_id = search_L2_mesh_aligned_morton_single(
                    pos,
                    mesh_aligned_morton,
                    search_radius=jnp.int32(l2_search_radius),
                    max_tests_per_cell=jnp.int32(256)
                )
            return elem_id
        elif use_mesh_aligned_octree:
            # MESH-ALIGNED OCTREE: Single-cell or multi-cell local search
            # Extracts octree structure from Kuhn tetrahedral mesh
            # Only works with axis-aligned Kuhn tets
            if mesh_aligned_octree_use_multi_local:
                # MULTI-CELL LOCAL (2×2×2): Search 8-cell neighborhood
                # For multi-cell vertex registration (~4 cells per element)
                # Searches 8 cells to cover all vertex locations
                # ~146 tests/particle (8 cells × 18.31 elem/cell)
                elem_id, _ = search_mesh_aligned_octree_multi_local(
                    pos,
                    mesh_aligned_octree,
                    max_tests=jnp.int32(200)
                )
            else:
                # SINGLE-CELL (DIRECT): Center cell only
                # 74.6% retention (elements span multiple cells)
                # ~5.9 elements per cell
                elem_id, _ = search_mesh_aligned_octree_single(
                    pos,
                    mesh_aligned_octree,
                    max_tests=jnp.int32(150)
                )
            return elem_id
        elif use_mesh_aligned_neighbors:
            # MESH-ALIGNED NEIGHBORS (OPTION B): Pre-computed neighbor table
            # Extracts octree structure + builds CPU neighbor table
            # Searches primary cell + 26 spatial neighbors at 3 levels
            # 99.95% searchability for particles inside mesh
            # ~13.9 tests/particle for uniform random, ~5.6 for centroids
            # ~8,190-8,832 particles/sec
            # Only works with axis-aligned Kuhn tets
            from jaxtrace.gpu.search.mesh_aligned_search_with_neighbors import search_multi_level_with_precomputed_neighbors
            elem_id, _ = search_multi_level_with_precomputed_neighbors(
                pos,
                mesh_aligned_octree_neighbors,
                levels_to_try=(14, 13, 12),
                max_tests_per_cell=jnp.int32(20)
            )
            return elem_id
        elif l2_search_method == 'hierarchical':
            # Hierarchical Morton neighbor search (depth 7 + depth 6 fallback)
            # Searches at multiple octree depths for variable-depth leaves
            # Requires table_depth > 0 (octree prefix table)
            return search_L2_morton_hierarchical_single(pos, mesh_gpu_global_morton)
        elif l2_search_method == 'incremental':
            # INCREMENTAL: Cascading radius search with configurable tiers
            # Each tier searches a symmetric band: radius=R → 2R+1 leaves
            # Conditional execution: skip later tiers if found at earlier tier
            # Example (2,5,10): 5 leaves → 11 leaves → 21 leaves
            # Expected: 1.8-2.5× speedup vs always using max radius
            return search_L2_morton_incremental_single(pos, mesh_gpu_global_morton, radii=l2_incremental_radii)
        elif l2_search_method == 'neighbors':
            # ENHANCED: Morton neighbor arithmetic with 5×5×5 boundary fallback
            # Tier 1: 3×3×3 (27 octants) - fast path
            # Tier 2: 5×5×5 outer shell (98 octants) - boundary fallback
            # Requires table_depth > 0 (octree prefix table)
            return search_L2_morton_neighbors_enhanced(pos, mesh_gpu_global_morton)
        elif l2_search_method == 'kdtree':
            # KD-TREE: Node-based search using K nearest nodes
            # Find K nearest mesh nodes, test all connected elements
            # Expected: ~95-100% retention, ~64 tests (K=3 × ~21 elem/node)
            # No spatial structure needed, works with any mesh
            from jaxtrace.gpu.search.kdtree_node_search import search_L2_kdtree_single
            return search_L2_kdtree_single(pos, kdtree_gpu, k_nearest=kdtree_k_nearest, max_tests=kdtree_max_tests)
        else:
            # Default: radius-based search (linear ±radius along Morton curve)
            return search_L2_global_morton_single(pos, mesh_gpu_global_morton, l2_search_radius)

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
