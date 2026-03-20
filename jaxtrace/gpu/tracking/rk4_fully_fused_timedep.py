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
    search_mesh_aligned_octree_multi_local,
    search_mesh_aligned_octree_multi_local_where
)
from jaxtrace.gpu.search.mesh_aligned_morton_search import (
    MeshAlignedMortonGPU,
    search_L2_mesh_aligned_morton_single,
    search_L2_mesh_aligned_morton_incremental_single,
)
import jaxtrace.config as config


def _create_rk4_fully_fused_timedep_impl(
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
    mesh_aligned_octree_neighbors = None,
    mesh_aligned_octree_use_multi_local: bool = False,
    mesh_aligned_octree_use_where: bool = False,
    kdtree_gpu = None,
    kdtree_k_nearest: int = 3,
    kdtree_max_tests: int = 256,
    mesh_bbox_min: Optional[jax.Array] = None,
    mesh_bbox_max: Optional[jax.Array] = None,
    levelset_gpu: Optional[jax.Array] = None,
    boundary_elements_gpu: Optional[jax.Array] = None,
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
        If provided and L2_SEARCH_METHOD == 'mesh_aligned_octree' at creation time,
        uses mesh-aligned octree instead of Morton curve search.
        Only works with Kuhn tetrahedral meshes (axis-aligned tets).
        Provides ~100% searchability with ~5.9 elements per cell.
    boundary_elements_gpu : Optional[jax.Array], default=None
        Boolean array (n_elements,) marking elements at the tool boundary (mixed
        level-set sign). When RK4_L0_SKIP_BOUNDARY_ELEMENTS is True, L0 caching
        is bypassed for these elements, forcing fresh L1/L2 search each substage.
        Matches FEMUSS behavior which always does fresh octree search.

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

    # Capture L2 search method at creation time (NOT at JIT trace time).
    # This avoids a race condition where config.L2_SEARCH_METHOD is temporarily
    # set by the caller but restored before the first JIT call triggers tracing.
    l2_search_method_config = config.L2_SEARCH_METHOD

    # Capture RK4 sub-step recovery config at creation time.
    use_bbox_clamp = config.RK4_SUBSTEP_BBOX_CLAMP and mesh_bbox_min is not None
    use_boundary_projection = config.RK4_BOUNDARY_PROJECTION and mesh_bbox_min is not None
    boundary_projection_tol = config.RK4_BOUNDARY_PROJECTION_TOL
    use_levelset_mask = config.RK4_LEVELSET_MASK and levelset_gpu is not None
    levelset_mode = config.RK4_LEVELSET_MODE if use_levelset_mask else None

    # Failed substage policy: backward compat with deprecated RK4_SUBSTEP_LAST_VALID_VEL
    failed_substage_policy = config.RK4_FAILED_SUBSTAGE_POLICY
    if config.RK4_SUBSTEP_LAST_VALID_VEL:
        failed_substage_policy = 'last_valid_vel'
    use_last_valid_vel = (failed_substage_policy == 'last_valid_vel')
    use_skip_step_on_fail = (failed_substage_policy == 'skip_step')
    use_skip_step_on_tool = (levelset_mode == 'skip_step')

    # L0 skip for boundary elements (mixed level-set sign at tool boundary)
    use_l0_skip_boundary = (
        config.RK4_L0_SKIP_BOUNDARY_ELEMENTS
        and boundary_elements_gpu is not None
    )

    # Per-wall boundary control: build (3,) boolean masks for min/max walls.
    # True = apply clamping on this wall, False = outlet (no treatment).
    # Resolved at trace time — zero runtime cost for disabled walls.
    wall_config = config.RK4_BOUNDARY_WALLS
    if wall_config is not None and not isinstance(wall_config, dict):
        wall_config = None  # non-dict value (e.g. 'clamp') → default all-clamp
    if wall_config is None:
        # Default: all walls clamped
        clamp_min_mask = jnp.array([True, True, True])
        clamp_max_mask = jnp.array([True, True, True])
    else:
        clamp_min_mask = jnp.array([
            wall_config.get('x_min', 'clamp') == 'clamp',
            wall_config.get('y_min', 'clamp') == 'clamp',
            wall_config.get('z_min', 'clamp') == 'clamp',
        ])
        clamp_max_mask = jnp.array([
            wall_config.get('x_max', 'clamp') == 'clamp',
            wall_config.get('y_max', 'clamp') == 'clamp',
            wall_config.get('z_max', 'clamp') == 'clamp',
        ])

    # Extended domain: check if any wall is set to 'extended'
    use_extended_domain = (
        wall_config is not None and
        'extended' in wall_config.values()
    )

    # ============================================================================
    # Single-Particle Helper Functions (Time-Dependent)
    # ============================================================================

    def search_l0_single(pos: jax.Array, cached_elem_id: jax.Array) -> jax.Array:
        """L0: Check if particle still in cached element (single particle).

        When use_l0_skip_boundary is enabled, boundary elements (mixed level-set
        sign) are treated as L0 miss, forcing a fresh L1/L2 search. This matches
        FEMUSS behavior which always does a fresh octree search with no caching.
        """
        is_valid = (cached_elem_id >= 0) & (cached_elem_id < len(connectivity))

        # Skip L0 for boundary elements: force fresh search near tool boundary
        if use_l0_skip_boundary:
            is_boundary = jnp.where(
                is_valid,
                boundary_elements_gpu[cached_elem_id],
                False
            )
            is_valid = is_valid & (~is_boundary)

        # Use GPU-optimized point-in-tet with JIT compilation and relative degeneracy threshold
        inside = jnp.where(
            is_valid,
            point_in_tet_gpu(pos, cached_elem_id, connectivity, node_positions),
            False
        )

        return jnp.where(inside, cached_elem_id, jnp.int32(-1))

    # Determine neighbor count from array shape (static at JIT trace time)
    _n_neighbors_per_elem = element_neighbors.shape[1]
    _use_node_l1 = (_n_neighbors_per_elem > 4)

    def search_l1_single(pos: jax.Array, start_elem_id: jax.Array) -> jax.Array:
        """L1: Neighbor search — adapts to face-based (4) or node-based (20-100+).

        Face-based (4 neighbors): Multi-hop with adaptive hop count.
        Node-based (>4 neighbors): Single hop with fori_loop over all neighbors.
        """
        start_elem_valid = start_elem_id >= 0

        if _use_node_l1:
            # Node-based: single hop, test all neighbors via fori_loop
            neighbors = element_neighbors[jnp.where(start_elem_valid, start_elem_id, 0)]
            n_valid = jnp.where(start_elem_valid, jnp.sum(neighbors >= 0), jnp.int32(0))

            def test_neighbor(idx, carry):
                found_elem = carry
                elem_id = neighbors[idx]
                valid = (elem_id >= 0) & (found_elem < 0)
                inside = jnp.where(
                    valid,
                    point_in_tet_gpu(pos, elem_id, connectivity, node_positions),
                    False
                )
                return jnp.where(inside & valid, elem_id, found_elem)

            return jax.lax.fori_loop(0, n_valid, test_neighbor, jnp.int32(-1))
        else:
            # Face-based: multi-hop with adaptive hop count
            current_elem = start_elem_id
            found = False

            start_volume = jnp.where(
                start_elem_valid,
                mesh_gpu_element_volumes[start_elem_id],
                config.FLOAT_DTYPE_JNP(1.0)
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
                        inside & check_this,
                        elem_id,
                        found_containing
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

    def search_l2_single(pos: jax.Array) -> jax.Array:
        """L2: Global search (single particle) - method selected at creation time."""
        # Check if mesh-aligned Morton should be used (HYBRID APPROACH - NEW)
        use_mesh_aligned_morton = (
            l2_search_method_config == 'mesh_aligned_morton' and
            mesh_aligned_morton is not None
        )

        # Check if direct mesh-aligned octree should be used
        use_mesh_aligned_octree = (
            l2_search_method_config == 'mesh_aligned_octree' and
            mesh_aligned_octree is not None
        )

        # Check if mesh-aligned octree with neighbors should be used (Option B - NEW)
        use_mesh_aligned_neighbors = (
            l2_search_method_config == 'mesh_aligned_neighbors' and
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
            if mesh_aligned_octree_use_multi_local and mesh_aligned_octree_use_where:
                # MULTI-CELL LOCAL (3×3×3) with jnp.where: avoids lax.cond vmap artifacts
                elem_id, _ = search_mesh_aligned_octree_multi_local_where(
                    pos,
                    mesh_aligned_octree,
                    max_tests=jnp.int32(600)
                )
            elif mesh_aligned_octree_use_multi_local:
                # MULTI-CELL LOCAL (3×3×3): Search 27-cell neighborhood
                # For multi-cell vertex registration (~4 cells per element)
                # Searches 27 cells to cover all vertex locations including adaptive refinement
                # ~494 tests/particle (27 cells × 18.31 elem/cell)
                elem_id, _ = search_mesh_aligned_octree_multi_local(
                    pos,
                    mesh_aligned_octree,
                    max_tests=jnp.int32(600)
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

    def search_l0_l1_l2_with_level(pos: jax.Array, cached_elem_id: jax.Array):
        """
        Full L0+L1+L2 search hierarchy, also returns which level found the element.

        Returns:
            (elem_id, hit_level) where hit_level is int8:
                0 = L0 hit (cached element still valid)
                1 = L1 hit (neighbor search found it)
                2 = L2 hit (global search found it)
               -1 = miss (not found at any level)
        """
        elem_l0 = search_l0_single(pos, cached_elem_id)
        found_l0 = elem_l0 >= 0

        if enable_l1_search:
            elem_l1_raw = search_l1_single(pos, cached_elem_id)
            elem_l1 = jnp.where(found_l0, elem_l0, elem_l1_raw)
            found_l1 = elem_l1 >= 0

            elem_l2 = search_l2_single(pos)
            elem_final = jnp.where(found_l1, elem_l1, elem_l2)
            found_l2 = elem_l2 >= 0

            hit_level = jnp.where(
                found_l0, jnp.int8(0),
                jnp.where(found_l1, jnp.int8(1),
                          jnp.where(found_l2, jnp.int8(2), jnp.int8(-1)))
            )
        else:
            elem_l2 = search_l2_single(pos)
            elem_final = jnp.where(found_l0, elem_l0, elem_l2)
            found_l2 = elem_l2 >= 0

            hit_level = jnp.where(
                found_l0, jnp.int8(0),
                jnp.where(found_l2, jnp.int8(2), jnp.int8(-1))
            )

        return elem_final, hit_level

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
        det = jnp.where(jnp.abs(det) < config.INTERPOLATION_DET_MIN, config.INTERPOLATION_DET_MIN, det)

        b1 = (dp0 * (d11*d22 - d12*d12) - d01 * (dp1*d22 - dp2*d12) + d02 * (dp1*d12 - dp2*d11)) / det
        b2 = (d00 * (dp1*d22 - dp2*d12) - dp0 * (d01*d22 - d02*d12) + d02 * (d01*dp2 - d02*dp1)) / det
        b3 = (d00 * (d11*dp2 - d12*dp1) - d01 * (d01*dp2 - d02*dp1) + dp0 * (d01*d12 - d02*d11)) / det
        b0 = 1.0 - b1 - b2 - b3

        # Interpolate velocity
        vel = b0 * node_vels[0] + b1 * node_vels[1] + b2 * node_vels[2] + b3 * node_vels[3]

        # Level-set masking: zero velocity inside tool (level-set < 0)
        if use_levelset_mask:
            node_ls = levelset_gpu[nodes_idx]  # (4,)
            ls_val = b0 * node_ls[0] + b1 * node_ls[1] + b2 * node_ls[2] + b3 * node_ls[3]
            vel = jnp.where(ls_val >= 0.0, vel, jnp.zeros(3, dtype=config.FLOAT_DTYPE_JNP))

        return jnp.where(valid, vel, jnp.zeros(3, dtype=config.FLOAT_DTYPE_JNP))

    def check_inside_tool(pos, elem_id):
        """Check if position is inside tool (level-set < 0). Returns boolean."""
        valid = (elem_id >= 0) & (elem_id < len(connectivity))
        nodes_idx = connectivity[elem_id]
        nodes = node_positions[nodes_idx]
        node_ls = levelset_gpu[nodes_idx]

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
        det = jnp.where(jnp.abs(det) < config.INTERPOLATION_DET_MIN, config.INTERPOLATION_DET_MIN, det)

        b1 = (dp0 * (d11*d22 - d12*d12) - d01 * (dp1*d22 - dp2*d12) + d02 * (dp1*d12 - dp2*d11)) / det
        b2 = (d00 * (dp1*d22 - dp2*d12) - dp0 * (d01*d22 - d02*d12) + d02 * (d01*dp2 - d02*dp1)) / det
        b3 = (d00 * (d11*dp2 - d12*dp1) - d01 * (d01*dp2 - d02*dp1) + dp0 * (d01*d12 - d02*d11)) / det
        b0 = 1.0 - b1 - b2 - b3

        ls_val = b0 * node_ls[0] + b1 * node_ls[1] + b2 * node_ls[2] + b3 * node_ls[3]
        return valid & (ls_val < 0.0)

    # ============================================================================
    # Per-wall clamping helpers (resolved at trace time via masks)
    # ============================================================================

    def clamp_to_bbox(pos):
        """Clamp position to mesh bbox, applying +tol inward only on the clamped component.

        For each axis, if the position is below bbox_min, clamp to bbox_min + tol.
        If above bbox_max, clamp to bbox_max - tol. Components inside the bbox are untouched.
        The tolerance is applied only in the inward-normal direction of the crossed wall.
        Per-wall outlet mask: walls with mask=False are not clamped.
        """
        tol = boundary_projection_tol
        # Min walls: clamp to bbox_min + tol (only where pos < bbox_min AND wall is active)
        hit_min = clamp_min_mask & (pos < mesh_bbox_min)
        clamped = jnp.where(hit_min, mesh_bbox_min + tol, pos)
        # Max walls: clamp to bbox_max - tol (only where pos > bbox_max AND wall is active)
        hit_max = clamp_max_mask & (clamped > mesh_bbox_max)
        clamped = jnp.where(hit_max, mesh_bbox_max - tol, clamped)
        return clamped

    # ============================================================================
    # Fully-Fused RK4 Step (Time-Dependent)
    # ============================================================================

    # ========================================================================
    # Unified RK4 Step Builder — single implementation, optional stats
    # ========================================================================
    # Both step functions (with and without stats) share identical numerics.
    # The `collect_stats` bool is resolved at JIT trace time, so when False
    # the level-tracking code is completely eliminated from the compiled kernel.

    def _build_rk4_step(collect_stats: bool):
        """Build a JIT-compiled RK4 step function with or without stats."""

        search_fn = search_l0_l1_l2_with_level if collect_stats else search_l0_l1_l2_single

        @jax.jit
        def rk4_step(
            positions_gpu: jax.Array,
            element_ids_gpu: jax.Array,
            dt: float,
            velocity_fields_gpu: jax.Array,
            time_idx: int,
            last_valid_velocities: jax.Array = None,
        ):
            n_timesteps = velocity_fields_gpu.shape[0]
            vel_idx = time_idx % n_timesteps
            velocity_field = velocity_fields_gpu[vel_idx]

            def rk4_single_particle(pos, elem_id, last_vel):
                # --- Extended domain: ballistic propagation for exited particles ---
                if use_extended_domain:
                    already_exited = elem_id < 0
                    # Ballistic: pos += dt * last_vel (for exited particles)
                    pos_ballistic = pos + dt * last_vel

                # --- Full RK4 integration (always computed; selected via jnp.where) ---
                # Stage 1: k1 = f(t, y)
                search_result_k1 = search_fn(pos, elem_id)
                if collect_stats:
                    elem_k1, lvl_k1 = search_result_k1
                else:
                    elem_k1 = search_result_k1
                vel_k1 = interpolate_velocity_single(pos, elem_k1, velocity_field)
                pos_k1 = pos + 0.5 * dt * vel_k1

                # Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
                if use_bbox_clamp:
                    pos_k1 = clamp_to_bbox(pos_k1)
                search_result_k2 = search_fn(pos_k1, elem_k1)
                if collect_stats:
                    elem_k2, lvl_k2 = search_result_k2
                else:
                    elem_k2 = search_result_k2
                vel_k2 = interpolate_velocity_single(pos_k1, elem_k2, velocity_field)
                if use_last_valid_vel:
                    vel_k2 = jnp.where(elem_k2 >= 0, vel_k2, vel_k1)
                pos_k2 = pos + 0.5 * dt * vel_k2

                # Stage 3: k3 = f(t + dt/2, y + dt/2 * k2)
                if use_bbox_clamp:
                    pos_k2 = clamp_to_bbox(pos_k2)
                search_result_k3 = search_fn(pos_k2, elem_k2)
                if collect_stats:
                    elem_k3, lvl_k3 = search_result_k3
                else:
                    elem_k3 = search_result_k3
                vel_k3 = interpolate_velocity_single(pos_k2, elem_k3, velocity_field)
                if use_last_valid_vel:
                    vel_k3 = jnp.where(elem_k3 >= 0, vel_k3, vel_k2)
                pos_k3 = pos + dt * vel_k3

                # Stage 4: k4 = f(t + dt, y + dt * k3)
                if use_bbox_clamp:
                    pos_k3 = clamp_to_bbox(pos_k3)
                search_result_k4 = search_fn(pos_k3, elem_k3)
                if collect_stats:
                    elem_k4, lvl_k4 = search_result_k4
                else:
                    elem_k4 = search_result_k4
                vel_k4 = interpolate_velocity_single(pos_k3, elem_k4, velocity_field)
                if use_last_valid_vel:
                    vel_k4 = jnp.where(elem_k4 >= 0, vel_k4, vel_k3)

                # Final position: y_n+1 = y_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
                pos_final = pos + (dt / 6.0) * (vel_k1 + 2.0*vel_k2 + 2.0*vel_k3 + vel_k4)

                # Final element search
                search_result_final = search_fn(pos_final, elem_k4)
                if collect_stats:
                    elem_final, lvl_final = search_result_final
                else:
                    elem_final = search_result_final

                # Skip-step policy: discard entire step if any substage failed
                if use_skip_step_on_fail:
                    any_failed = (elem_k1 < 0) | (elem_k2 < 0) | (elem_k3 < 0) | (elem_k4 < 0)
                    pos_final = jnp.where(any_failed, pos, pos_final)
                    elem_final = jnp.where(any_failed, elem_id, elem_final)
                    if collect_stats:
                        lvl_final = jnp.where(any_failed, jnp.int8(-1), lvl_final)

                # Skip-step policy: discard entire step if any substage is inside tool
                if use_skip_step_on_tool:
                    any_inside = (
                        check_inside_tool(pos, elem_k1) |
                        check_inside_tool(pos_k1, elem_k2) |
                        check_inside_tool(pos_k2, elem_k3) |
                        check_inside_tool(pos_k3, elem_k4)
                    )
                    pos_final = jnp.where(any_inside, pos, pos_final)
                    elem_final = jnp.where(any_inside, elem_id, elem_final)
                    if collect_stats:
                        lvl_final = jnp.where(any_inside, jnp.int8(-1), lvl_final)

                # Boundary projection: clamp to bbox and re-search if lost
                if use_boundary_projection:
                    pos_clamped = clamp_to_bbox(pos_final)
                    search_result_clamped = search_fn(pos_clamped, elem_k4)
                    if collect_stats:
                        elem_clamped, lvl_clamped = search_result_clamped
                    else:
                        elem_clamped = search_result_clamped
                    lost = elem_final < 0
                    pos_final = jnp.where(lost, pos_clamped, pos_final)
                    elem_final = jnp.where(lost, elem_clamped, elem_final)
                    if collect_stats:
                        lvl_final = jnp.where(lost, lvl_clamped, lvl_final)

                # --- Extended domain: merge ballistic vs RK4 results ---
                if use_extended_domain:
                    # For exited particles: use ballistic position, keep elem=-1
                    pos_final = jnp.where(already_exited, pos_ballistic, pos_final)
                    elem_final = jnp.where(already_exited, jnp.int32(-1), elem_final)
                    # Update last_vel: use vel_k1 if particle is alive, keep old if exited
                    vel_out = jnp.where(already_exited, last_vel, vel_k1)
                    if collect_stats:
                        lvl_final = jnp.where(already_exited, jnp.int8(-1), lvl_final)
                else:
                    vel_out = last_vel  # pass-through (unused when extended domain off)

                if collect_stats:
                    hit_levels = jnp.array([lvl_k1, lvl_k2, lvl_k3, lvl_k4, lvl_final], dtype=jnp.int8)
                    return pos_final, elem_final, hit_levels, vel_out
                else:
                    return pos_final, elem_final, vel_out

            # Provide zero velocities if not passed (non-extended mode)
            if use_extended_domain:
                # last_valid_velocities must be provided by caller
                vels_in = last_valid_velocities
            else:
                # Dummy array — not used in computation, just satisfies vmap signature
                vels_in = jnp.zeros_like(positions_gpu)

            if collect_stats:
                positions_final, element_ids_final, all_hit_levels, vels_out = jax.vmap(
                    rk4_single_particle
                )(positions_gpu, element_ids_gpu, vels_in)

                # Aggregate counts across all particles and all 5 sub-steps
                l0_hits = jnp.sum(all_hit_levels == 0).astype(jnp.int32)
                l1_hits = jnp.sum(all_hit_levels == 1).astype(jnp.int32)
                l2_hits = jnp.sum(all_hit_levels == 2).astype(jnp.int32)
                misses  = jnp.sum(all_hit_levels == -1).astype(jnp.int32)

                if use_extended_domain:
                    return positions_final, element_ids_final, (l0_hits, l1_hits, l2_hits, misses), vels_out
                else:
                    return positions_final, element_ids_final, (l0_hits, l1_hits, l2_hits, misses)
            else:
                positions_final, element_ids_final, vels_out = jax.vmap(
                    rk4_single_particle
                )(positions_gpu, element_ids_gpu, vels_in)

                if use_extended_domain:
                    return positions_final, element_ids_final, vels_out
                else:
                    return positions_final, element_ids_final

        return rk4_step

    rk4_fully_fused_step_timedep = _build_rk4_step(collect_stats=False)
    rk4_fully_fused_step_timedep_with_stats = _build_rk4_step(collect_stats=True)

    return rk4_fully_fused_step_timedep, rk4_fully_fused_step_timedep_with_stats


def create_rk4_fully_fused_timedep(
    mesh_gpu_connectivity,
    mesh_gpu_node_positions,
    mesh_gpu_element_neighbors,
    mesh_gpu_element_volumes,
    mesh_gpu_global_morton,
    n_hops: int = 3,
    l2_search_radius: int = 2,
    enable_l1_search: bool = True,
    l2_search_method: str = 'radius',
    l2_incremental_radii: tuple = (2, 5, 10),
    mesh_aligned_octree=None,
    mesh_aligned_morton=None,
    mesh_aligned_octree_neighbors=None,
    mesh_aligned_octree_use_multi_local: bool = False,
    mesh_aligned_octree_use_where: bool = False,
    kdtree_gpu=None,
    kdtree_k_nearest: int = 3,
    kdtree_max_tests: int = 256,
    mesh_bbox_min=None,
    mesh_bbox_max=None,
    levelset_gpu=None,
    boundary_elements_gpu=None,
):
    """
    Create fully-fused RK4 integrator with time-dependent velocity.
    Returns only the production step function (no stats overhead).
    See create_rk4_fully_fused_timedep_with_stats for the stats variant.
    """
    step_fn, _ = _create_rk4_fully_fused_timedep_impl(
        mesh_gpu_connectivity, mesh_gpu_node_positions,
        mesh_gpu_element_neighbors, mesh_gpu_element_volumes,
        mesh_gpu_global_morton, n_hops, l2_search_radius,
        enable_l1_search, l2_search_method, l2_incremental_radii,
        mesh_aligned_octree, mesh_aligned_morton, mesh_aligned_octree_neighbors,
        mesh_aligned_octree_use_multi_local, mesh_aligned_octree_use_where,
        kdtree_gpu, kdtree_k_nearest, kdtree_max_tests,
        mesh_bbox_min, mesh_bbox_max, levelset_gpu, boundary_elements_gpu,
    )
    return step_fn


def create_rk4_fully_fused_timedep_with_stats(
    mesh_gpu_connectivity,
    mesh_gpu_node_positions,
    mesh_gpu_element_neighbors,
    mesh_gpu_element_volumes,
    mesh_gpu_global_morton,
    n_hops: int = 3,
    l2_search_radius: int = 2,
    enable_l1_search: bool = True,
    l2_search_method: str = 'radius',
    l2_incremental_radii: tuple = (2, 5, 10),
    mesh_aligned_octree=None,
    mesh_aligned_morton=None,
    mesh_aligned_octree_neighbors=None,
    mesh_aligned_octree_use_multi_local: bool = False,
    mesh_aligned_octree_use_where: bool = False,
    kdtree_gpu=None,
    kdtree_k_nearest: int = 3,
    kdtree_max_tests: int = 256,
    mesh_bbox_min=None,
    mesh_bbox_max=None,
    levelset_gpu=None,
    boundary_elements_gpu=None,
):
    """
    Create fully-fused RK4 integrator with time-dependent velocity.
    Returns (step_fn, step_fn_with_stats) where step_fn_with_stats returns
    (positions, element_ids, (l0_hits, l1_hits, l2_hits, misses)) per step.

    Hit counts are summed across all N particles × 5 RK4 sub-step searches.
    hit_level encoding: 0=L0, 1=L1, 2=L2, -1=miss (not found at any level).
    """
    step_fn, step_fn_with_stats = _create_rk4_fully_fused_timedep_impl(
        mesh_gpu_connectivity, mesh_gpu_node_positions,
        mesh_gpu_element_neighbors, mesh_gpu_element_volumes,
        mesh_gpu_global_morton, n_hops, l2_search_radius,
        enable_l1_search, l2_search_method, l2_incremental_radii,
        mesh_aligned_octree, mesh_aligned_morton, mesh_aligned_octree_neighbors,
        mesh_aligned_octree_use_multi_local, mesh_aligned_octree_use_where,
        kdtree_gpu, kdtree_k_nearest, kdtree_max_tests,
        mesh_bbox_min, mesh_bbox_max, levelset_gpu, boundary_elements_gpu,
    )
    return step_fn, step_fn_with_stats
