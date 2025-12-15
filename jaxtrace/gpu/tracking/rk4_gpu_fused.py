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
    search_level1_multihop_vectorized,
    search_level1_multihop_hierarchical
)
from jaxtrace.gpu.search.block_local_search import (
    BlockElementLists,
    create_search_with_block_fallback
)
from jaxtrace.gpu.search.octree_search_gpu import search_level2_octree_scan
from jaxtrace.gpu.search.level2_block_morton import create_level2_block_morton_search


@jax.jit
def compute_block_id_from_position(
    position: jax.Array,
    domain_bounds: jax.Array,
    grid_size: Tuple[int, int, int]
) -> jax.Array:
    """
    Compute block ID for a single particle position.

    Maps 3D position to coarse block grid index, then to linear block ID.

    Parameters
    ----------
    position : jax.Array, shape (3,)
        Particle position [x, y, z]
    domain_bounds : jax.Array, shape (6,)
        Domain bounds [xmin, xmax, ymin, ymax, zmin, zmax]
    grid_size : Tuple[int, int, int]
        Grid dimensions (nx, ny, nz)

    Returns
    -------
    block_id : jax.Array, scalar int32
        Block ID (0 to n_blocks-1), or -1 if outside domain

    Notes
    -----
    Block ID mapping: block_id = i + j*nx + k*nx*ny
    This matches the block_grid.py convention.
    """
    nx, ny, nz = grid_size

    # Compute block sizes
    dx = (domain_bounds[1] - domain_bounds[0]) / nx
    dy = (domain_bounds[3] - domain_bounds[2]) / ny
    dz = (domain_bounds[5] - domain_bounds[4]) / nz

    # Compute grid indices
    i = jnp.floor((position[0] - domain_bounds[0]) / dx).astype(jnp.int32)
    j = jnp.floor((position[1] - domain_bounds[2]) / dy).astype(jnp.int32)
    k = jnp.floor((position[2] - domain_bounds[4]) / dz).astype(jnp.int32)

    # Check bounds
    valid = (
        (i >= 0) & (i < nx) &
        (j >= 0) & (j < ny) &
        (k >= 0) & (k < nz)
    )

    # Compute linear block ID
    block_id = i + j * nx + k * nx * ny

    # Return -1 if outside domain
    return jnp.where(valid, block_id, jnp.int32(-1))


@jax.jit
def compute_block_ids_batch(
    positions: jax.Array,
    domain_bounds: jax.Array,
    grid_size: Tuple[int, int, int]
) -> jax.Array:
    """
    Compute block IDs for a batch of particles.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Particle positions
    domain_bounds : jax.Array, shape (6,)
        Domain bounds [xmin, xmax, ymin, ymax, zmin, zmax]
    grid_size : Tuple[int, int, int]
        Grid dimensions (nx, ny, nz)

    Returns
    -------
    block_ids : jax.Array, shape (N,), int32
        Block IDs for each particle
    """
    # Single vmap over particles
    return jax.vmap(
        lambda pos: compute_block_id_from_position(pos, domain_bounds, grid_size)
    )(positions)


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
    block_ids: jax.Array      # (N,) int32 - coarse block ID per particle (for L2 search)


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


def create_search_gpu_fused_hierarchical(n_hops: int = 5):
    """
    Create a JIT-compiled GPU search function with hierarchical early-exit multi-hop.

    This uses the hierarchical early-exit implementation instead of concatenation,
    which enables 4-hop and 5-hop search without GPU memory overflow.

    Memory comparison for 105k particles:
    - Naive 5-hop concatenation: 1,364 neighbors × 105k = 572 MB → OOM
    - Hierarchical early-exit: avg ~25 neighbors × 105k = 10 MB ✅

    Performance trade-off:
    - Throughput: 8-15k p/s (vs 23k for 3-hop concatenated)
    - Hit rate: 99.99% (vs 99.9% for 3-hop)
    - Retention: 82% at 2,500 steps (vs 16% for 3-hop)

    Parameters
    ----------
    n_hops : int, default=5
        Number of hops for L1 neighbor search:
        - 3: ~84 neighbors (98-99.5% hit rate, baseline)
        - 4: ~256 neighbors max (99.9% hit rate, better)
        - 5: ~1,024 neighbors max (99.99% hit rate, recommended)

    Returns
    -------
    search_func : callable
        JIT-compiled search function with hierarchical early-exit
    """
    @jax.jit
    def search_gpu_fused_hierarchical_impl(
        positions_gpu: jax.Array,      # (N, 3)
        cached_element_ids_gpu: jax.Array,  # (N,)
        mesh_gpu_node_positions: jax.Array,
        mesh_gpu_connectivity: jax.Array,
        mesh_gpu_element_neighbors: jax.Array
    ) -> jax.Array:
        """
        Fused GPU search with hierarchical early-exit: L0 + L1 multi-hop, all on GPU.

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

        # L1: Hierarchical multi-hop search with early exit
        element_ids_l1 = search_level1_multihop_hierarchical(
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

    return search_gpu_fused_hierarchical_impl


def create_search_gpu_fused_with_l2_octree(
    n_hops: int = 4,
    octree_node_metadata: Optional[jax.Array] = None,
    octree_node_elements: Optional[jax.Array] = None,
    max_octree_depth: int = 10
):
    """
    Create a JIT-compiled GPU search function with L2 octree fallback.

    This uses hierarchical early-exit multi-hop search (L0 + L1) with scan-based
    octree search as L2 fallback for particles that miss both L0 and L1.

    Three-tier search hierarchy:
    - L0: Check cached elements (85-95% hit rate)
    - L1: Hierarchical multi-hop neighbor search (99.9-99.95% cumulative hit rate)
    - L2: Scan-based octree search (99.99% cumulative hit rate)

    Memory comparison for 105k particles:
    - Hierarchical 4-hop: 10 MB
    - 4-hop + L2 octree: 12 MB (+2 MB for octree)

    Performance comparison:
    - 4-hop only: 40-48k p/s, 60% retention at 2,500 steps
    - 4-hop + L2: 40-48k p/s, 82% retention at 2,500 steps (<1% overhead)

    Parameters
    ----------
    n_hops : int, default=4
        Number of hops for L1 neighbor search:
        - 3: ~84 neighbors (98-99.5% hit rate)
        - 4: ~256 neighbors max (99.9-99.95% hit rate, recommended)
        - 5: ~1,024 neighbors max (99.99% hit rate, but slower)
    octree_node_metadata : jax.Array, optional, shape (n_nodes, 15)
        Octree metadata (GPU-resident). If None, L2 octree is disabled.
    octree_node_elements : jax.Array, optional, shape (n_nodes, max_leaf_size)
        Octree element arrays (GPU-resident). If None, L2 octree is disabled.
    max_octree_depth : int, default=10
        Maximum octree traversal depth

    Returns
    -------
    search_func : callable
        JIT-compiled search function with L0 + L1 + L2 hierarchy
    """
    @jax.jit
    def search_gpu_fused_with_l2_impl(
        positions_gpu: jax.Array,      # (N, 3)
        cached_element_ids_gpu: jax.Array,  # (N,)
        mesh_gpu_node_positions: jax.Array,
        mesh_gpu_connectivity: jax.Array,
        mesh_gpu_element_neighbors: jax.Array
    ) -> jax.Array:
        """
        Fused GPU search with L2 octree fallback: L0 + L1 + L2, all on GPU.

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

        # L1: Hierarchical multi-hop search with early exit
        element_ids_l1 = search_level1_multihop_hierarchical(
            positions_gpu,
            cached_element_ids_gpu,
            mesh_gpu_element_neighbors,
            mesh_gpu_node_positions,
            mesh_gpu_connectivity,
            n_hops=n_hops
        )

        # Merge L0 and L1
        element_ids_l0_l1 = jnp.where(element_ids_l0 >= 0, element_ids_l0, element_ids_l1)

        # L2: Octree fallback (only if octree is provided)
        if octree_node_metadata is not None and octree_node_elements is not None:
            # CRITICAL FIX: Pass L0+L1 results as cached_ids (not previous timestep IDs)
            # This enables masking: particles with element_ids_l0_l1 >= 0 skip octree search
            element_ids_gpu = search_level2_octree_scan(
                positions_gpu,
                element_ids_l0_l1,  # ← FIX: Use current L0+L1 results, not previous timestep
                octree_node_metadata,
                octree_node_elements,
                mesh_gpu_node_positions,
                mesh_gpu_connectivity,
                max_depth=max_octree_depth
            )
            # Note: No merge needed - search_level2_octree_scan handles it via masking
        else:
            # No L2 octree - just use L0/L1 results
            element_ids_gpu = element_ids_l0_l1

        return element_ids_gpu

    return search_gpu_fused_with_l2_impl


def create_search_gpu_fused_with_l2_block_morton(
    n_hops: int = 3,
    search_l2_morton = None
):
    """
    Create a JIT-compiled GPU search function with L2 block Morton fallback.

    This provides three-tier search hierarchy optimized for fused RK4:
    1. L0: Cached element check (85-95% hit rate)
    2. L1: Multi-hop neighbor search (99.9-99.95% cumulative hit rate)
    3. L2: Block-local Morton search (99.99% cumulative hit rate)

    Key advantages over global octree L2:
    - Memory: ~8 MB vs 6,500 MB
    - Bounded search: O(max_elements_per_block) ~ O(50)
    - JAX-compatible: No nested vmap, no CSR, pure padded arrays
    - Architecture-aligned: Uses existing coarse block structure

    Parameters
    ----------
    n_hops : int, default=3
        Number of hops for L1 neighbor search:
        - 2: ~20 neighbors (95-98% hit rate, fastest)
        - 3: ~84 neighbors (98-99.5% hit rate, recommended)
        - 4: ~256 neighbors max (99.9-99.95% hit rate)
    search_l2_morton : callable, optional
        L2 block Morton search function created by create_level2_block_morton_search().
        If None, L2 is disabled and only L0+L1 are used.

    Returns
    -------
    search_func : callable
        JIT-compiled search function with signature:
        search_func(positions, cached_ids, block_ids, node_positions, connectivity, element_neighbors) -> element_ids

    Notes
    -----
    The returned search function expects block_ids as a parameter. These should be computed
    from particle positions using compute_block_ids_batch().

    Expected performance (with 3-hop + L2 Morton):
    - Throughput: 40-48k p/s
    - Hit rate: >99.95%
    - Retention: >80% at 2,500 steps
    - Memory overhead: <1%
    """
    @jax.jit
    def search_gpu_fused_with_l2_morton_impl(
        positions_gpu: jax.Array,           # (N, 3)
        cached_element_ids_gpu: jax.Array,  # (N,)
        block_ids_gpu: jax.Array,           # (N,) - NEW: block assignment
        mesh_gpu_node_positions: jax.Array,
        mesh_gpu_connectivity: jax.Array,
        mesh_gpu_element_neighbors: jax.Array
    ) -> jax.Array:
        """
        Fused GPU search with L2 block Morton fallback: L0 + L1 + L2, all on GPU.

        No CPU-GPU transfers. Returns updated element IDs on GPU.

        Parameters
        ----------
        positions_gpu : jax.Array, shape (N, 3)
            Particle positions (GPU-resident)
        cached_element_ids_gpu : jax.Array, shape (N,)
            Cached element IDs from previous step (GPU-resident)
        block_ids_gpu : jax.Array, shape (N,)
            Block ID for each particle (GPU-resident)
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

        # L1: Hierarchical multi-hop search with early exit
        element_ids_l1 = search_level1_multihop_hierarchical(
            positions_gpu,
            cached_element_ids_gpu,
            mesh_gpu_element_neighbors,
            mesh_gpu_node_positions,
            mesh_gpu_connectivity,
            n_hops=n_hops
        )

        # Merge L0 and L1
        element_ids_l0_l1 = jnp.where(element_ids_l0 >= 0, element_ids_l0, element_ids_l1)

        # L2: Block Morton fallback (only if search function is provided)
        if search_l2_morton is not None:
            element_ids_gpu = search_l2_morton(
                positions_gpu,
                block_ids_gpu,
                element_ids_l0_l1  # Pass L0+L1 results as cached_ids
            )
        else:
            # No L2 Morton - just use L0/L1 results
            element_ids_gpu = element_ids_l0_l1

        return element_ids_gpu

    return search_gpu_fused_with_l2_morton_impl


def create_search_gpu_fused_with_block_fallback(
    n_hops: int = 3,
    block_lists: Optional[BlockElementLists] = None
):
    """
    Create a JIT-compiled GPU search function with block-local fallback.

    This provides two-tier search:
    1. L0 + L1 multi-hop (fast, 99.9% hit rate)
    2. Block-local fallback (catches remaining failures, pure GPU)

    Parameters
    ----------
    n_hops : int, default=3
        Number of hops for L1 neighbor search
    block_lists : BlockElementLists, optional
        Block element lists for fallback. If None, uses standard search without fallback.

    Returns
    -------
    search_func : callable
        JIT-compiled search function with fallback
    """
    if block_lists is None:
        # No fallback, use standard search
        return create_search_gpu_fused(n_hops=n_hops)

    # Create search with block fallback
    search_with_fallback = create_search_with_block_fallback(
        n_hops=n_hops,
        block_lists=block_lists
    )

    @jax.jit
    def search_gpu_fused_with_fallback_impl(
        positions_gpu: jax.Array,           # (N, 3)
        cached_element_ids_gpu: jax.Array,  # (N,)
        block_ids_gpu: jax.Array,           # (N,) - NEW: block assignment
        mesh_gpu_node_positions: jax.Array,
        mesh_gpu_connectivity: jax.Array,
        mesh_gpu_element_neighbors: jax.Array
    ) -> jax.Array:
        """
        Fused GPU search with block-local fallback.

        Three tiers:
        1. L0: Check cached element (fastest)
        2. L1: Multi-hop neighbor search (fast, 99.9% success)
        3. Block-local: Search all elements in particle's block (slow, 100% success in block)

        All operations stay on GPU, no CPU-GPU transfers.

        Parameters
        ----------
        positions_gpu : jax.Array, shape (N, 3)
            Particle positions
        cached_element_ids_gpu : jax.Array, shape (N,)
            Cached element IDs from previous step
        block_ids_gpu : jax.Array, shape (N,)
            Block ID for each particle
        mesh_gpu_* : jax.Array
            GPU-resident mesh data

        Returns
        -------
        element_ids_gpu : jax.Array, shape (N,)
            Updated element IDs
        """
        # L0: Check cached elements
        element_ids_l0 = search_level0_vectorized(
            positions_gpu,
            cached_element_ids_gpu,
            mesh_gpu_node_positions,
            mesh_gpu_connectivity
        )

        # L1 + Block fallback: Multi-hop search with automatic fallback
        element_ids_l1_fallback = search_with_fallback(
            positions_gpu,
            cached_element_ids_gpu,
            block_ids_gpu,
            mesh_gpu_node_positions,
            mesh_gpu_connectivity,
            mesh_gpu_element_neighbors
        )

        # Merge L0 and L1+fallback results
        element_ids_gpu = jnp.where(element_ids_l0 >= 0, element_ids_l0, element_ids_l1_fallback)

        return element_ids_gpu

    return search_gpu_fused_with_fallback_impl


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
    velocity_field,  # Can be np.ndarray OR jax.Array
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
    velocity_field : np.ndarray or jax.Array
        Velocity field at nodes. If numpy array, will be uploaded to GPU.
        If jax.Array, assumes already on GPU (avoids repeated uploads).
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

    # Upload initial state to GPU (ONE upload per timestep)
    t_upload = time.time()
    positions_gpu = jax.device_put(positions.astype(np.float32))
    element_ids_gpu = jax.device_put(element_ids.astype(np.int32))

    # Only upload velocity field if it's a numpy array (not already on GPU)
    if isinstance(velocity_field, np.ndarray):
        velocity_field_gpu = jax.device_put(velocity_field.astype(np.float32))
    else:
        # Already a jax.Array on GPU, no upload needed
        velocity_field_gpu = velocity_field
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


def rk4_step_gpu_fused_wrapper_hierarchical(
    positions: np.ndarray,
    element_ids: np.ndarray,
    dt: float,
    mesh_gpu: MeshDataGPU,
    velocity_field,  # Can be np.ndarray OR jax.Array
    n_hops: int = 5
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Wrapper for GPU-fused RK4 with HIERARCHICAL early-exit multi-hop search.

    This function uses the hierarchical early-exit implementation instead of
    concatenation, enabling 5-hop search without GPU memory overflow.

    Memory comparison for 105k particles:
    - Naive 5-hop concatenation: 1,364 neighbors × 105k = 572 MB → OOM ❌
    - Hierarchical 5-hop early-exit: avg ~25 neighbors × 105k = 10 MB ✅

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
    velocity_field : np.ndarray or jax.Array
        Velocity field at nodes. If numpy array, will be uploaded to GPU.
        If jax.Array, assumes already on GPU (avoids repeated uploads).
    n_hops : int, default=5
        Number of hops for hierarchical L1 neighbor search:
        - 3: ~84 neighbors max (baseline for comparison)
        - 4: ~256 neighbors max (99.95% hit rate)
        - 5: ~1,024 neighbors max (99.99% hit rate, recommended)

    Returns
    -------
    positions_final : np.ndarray, shape (N, 3)
        Final particle positions (CPU)
    element_ids_final : np.ndarray, shape (N,)
        Final element IDs (CPU)
    stats : dict
        Timing statistics

    Expected Performance:
    - Throughput: 8-15k p/s (vs 23k for 3-hop concatenated)
    - Hit rate: 99.99% (vs 99.9% for 3-hop)
    - Retention: 82% at 2,500 steps (vs 16% for 3-hop)
    - Memory: 10 MB (vs 572 MB OOM for naive 5-hop)
    """
    # Create hierarchical search function
    search_func = create_search_gpu_fused_hierarchical(n_hops=n_hops)

    # Create RK4 function with hierarchical search
    @jax.jit
    def rk4_fused_with_hierarchical_search(
        positions_gpu,
        element_ids_gpu,
        dt,
        connectivity_gpu,
        node_positions_gpu,
        element_neighbors_gpu,
        velocity_field_gpu
    ):
        """GPU-fused RK4 with hierarchical early-exit search."""
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

        # Combine stages: y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        positions_new = positions_gpu + (dt / 6.0) * (
            velocities_k1 + 2.0 * velocities_k2 + 2.0 * velocities_k3 + velocities_k4
        )

        # Final element search for new positions
        element_ids_new = search_func(
            positions_new,
            element_ids_k4,
            node_positions_gpu,
            connectivity_gpu,
            element_neighbors_gpu
        )

        return positions_new, element_ids_new

    # Upload initial state to GPU (ONE upload)
    t_total = time.time()

    t_upload = time.time()
    positions_gpu = jax.device_put(positions.astype(np.float32))
    element_ids_gpu = jax.device_put(element_ids.astype(np.int32))

    # Upload velocity field if not already on GPU
    if isinstance(velocity_field, np.ndarray):
        velocity_field_gpu = jax.device_put(velocity_field.astype(np.float32))
    else:
        velocity_field_gpu = velocity_field  # Already on GPU
    t_upload = time.time() - t_upload

    # Run GPU-fused RK4 with hierarchical search
    t_compute = time.time()
    positions_final_gpu, element_ids_final_gpu = rk4_fused_with_hierarchical_search(
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


def create_rk4_step_gpu_fused_for_production_with_l2_octree(
    n_hops: int = 3,
    octree_metadata: Optional[jax.Array] = None,
    octree_elements: Optional[jax.Array] = None,
    max_octree_depth: int = 10
):
    """
    Create production RK4 wrapper with L2 octree fallback.

    This factory function creates the search function ONCE and returns
    a wrapper that can be called multiple times without re-JIT compilation.

    Parameters
    ----------
    n_hops : int, default=3
        Number of hops for L1 neighbor search
    octree_metadata : jax.Array, optional
        Octree node metadata on GPU
    octree_elements : jax.Array, optional
        Octree element lists on GPU
    max_octree_depth : int, default=10
        Maximum octree traversal depth

    Returns
    -------
    rk4_step_func : callable
        Function with signature (particle_data, velocity_field, dt, mesh_gpu, current_time)
    """
    # Create search function ONCE (cached)
    search_func = create_search_gpu_fused_with_l2_octree(
        n_hops=n_hops,
        octree_node_metadata=octree_metadata,
        octree_node_elements=octree_elements,
        max_octree_depth=max_octree_depth
    )

    def rk4_step_gpu_fused_for_production_with_l2_octree(
        particle_data: dict,
        velocity_field,  # Can be np.ndarray OR jax.Array
        dt: float,
        mesh_gpu: MeshDataGPU,
        current_time: float = 0.0
    ) -> Tuple[dict, dict]:
        """
        Production wrapper for GPU-fused RK4 with L2 octree fallback.

        Three-tier search hierarchy:
        - L0: Cached element check (~85-95% hit rate)
        - L1: Multi-hop neighbor expansion (n_hops, ~99.9% cumulative)
        - L2: Octree spatial search (catches remaining particles in refined regions)

        This wrapper matches the signature of rk4_step_gpu_fused_for_production_hierarchical()
        but adds L2 octree fallback for improved retention in refined mesh regions.

        Parameters
        ----------
        particle_data : dict
            Particle data with keys 'positions', 'element_ids', 'particle_ids', 'active_mask'
        velocity_field : np.ndarray or jax.Array
            Velocity field at nodes. If numpy array, will be uploaded to GPU.
            If jax.Array, assumes already on GPU (avoids repeated uploads).
        dt : float
            Time step size
        mesh_gpu : MeshDataGPU
            GPU-resident mesh data
        current_time : float, optional
            Current simulation time (for logging/diagnostics)

        Returns
        -------
        particle_data_updated : dict
            Updated particle data with new positions and element IDs
        stats : dict
            Timing and performance statistics

        Expected Performance (with 3-hop + L2):
        - Throughput: ~40-48k p/s (similar to 4-hop hierarchical)
        - Hit rate: >99.95% (L2 catches refined region particles)
        - Retention: ~82% at 2,500 steps
        - Overhead: <1% vs 3-hop only (for real mesh with sparse refined regions)
        """
        # Extract particle data
        positions = particle_data.positions
        element_ids = particle_data.element_ids

        # Create RK4 function with L2 octree search
        @jax.jit
        def rk4_fused_with_l2_search(
            positions_gpu,
            element_ids_gpu,
            dt,
            connectivity_gpu,
            node_positions_gpu,
            element_neighbors_gpu,
            velocity_field_gpu
        ):
            """GPU-fused RK4 with L0 + L1 (multi-hop) + L2 (octree) search."""
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

        # Upload initial state to GPU (ONE upload per timestep)
        t_upload = time.time()
        positions_gpu = jax.device_put(positions.astype(np.float32))
        element_ids_gpu = jax.device_put(element_ids.astype(np.int32))

        # Only upload velocity field if it's a numpy array (not already on GPU)
        if isinstance(velocity_field, np.ndarray):
            velocity_field_gpu = jax.device_put(velocity_field.astype(np.float32))
        else:
            # Already a jax.Array on GPU, no upload needed
            velocity_field_gpu = velocity_field
        t_upload = time.time() - t_upload

        # Run GPU-fused RK4 with L2 octree search
        t_compute = time.time()
        positions_final_gpu, element_ids_final_gpu = rk4_fused_with_l2_search(
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

        # Update particle data
        from dataclasses import replace
        particle_data_updated = replace(
            particle_data,
            positions=positions_final,
            element_ids=element_ids_final
        )

        stats = {
            'time_upload': t_upload,
            'time_compute': t_compute,
            'time_download': t_download,
            'time_total': t_total,
            'n_particles': len(positions)
        }

        return particle_data_updated, stats

    # Return the inner function (reusable wrapper)
    return rk4_step_gpu_fused_for_production_with_l2_octree


def create_rk4_step_gpu_fused_for_production_with_l2_block_morton(
    n_hops: int = 3,
    block_element_ids_gpu: Optional[jax.Array] = None,
    node_positions_gpu: Optional[jax.Array] = None,
    connectivity_gpu: Optional[jax.Array] = None,
    max_elements_per_block: int = 50,
    domain_bounds: Optional[jax.Array] = None,
    grid_size: Optional[Tuple[int, int, int]] = None
):
    """
    Create production RK4 wrapper with L2 block Morton fallback.

    This factory function creates the search function ONCE and returns
    a wrapper that can be called multiple times without re-JIT compilation.

    Three-tier search hierarchy:
    - L0: Cached element check (85-95% hit rate)
    - L1: Multi-hop neighbor search (99.9-99.95% cumulative)
    - L2: Block-local Morton search (99.99% cumulative)

    Key advantages over global octree L2:
    - Memory: ~8 MB vs 6,500 MB
    - Bounded search: O(50) vs O(depth + leaf_size)
    - JAX-compatible: No nested vmap, pure padded arrays
    - Architecture-aligned: Uses existing coarse block structure

    Parameters
    ----------
    n_hops : int, default=3
        Number of hops for L1 neighbor search
    block_element_ids_gpu : jax.Array, optional, shape (n_blocks, max_elements_per_block)
        Block Morton structures on GPU (from block_morton_builder)
    node_positions_gpu : jax.Array, optional
        Node positions on GPU (from mesh_gpu)
    connectivity_gpu : jax.Array, optional
        Element connectivity on GPU (from mesh_gpu)
    max_elements_per_block : int, default=50
        Maximum elements per block
    domain_bounds : jax.Array, optional, shape (6,)
        Domain bounds [xmin, xmax, ymin, ymax, zmin, zmax]
    grid_size : Tuple[int, int, int], optional
        Grid dimensions (nx, ny, nz)

    Returns
    -------
    rk4_step_func : callable
        Function with signature (particle_data, velocity_field, dt, mesh_gpu, current_time)

    Notes
    -----
    If block Morton structures are not provided (None), L2 is disabled and only L0+L1 are used.
    """
    # Create L2 Morton search function ONCE (if structures provided)
    if (block_element_ids_gpu is not None and
        node_positions_gpu is not None and
        connectivity_gpu is not None):
        search_l2_morton = create_level2_block_morton_search(
            block_element_ids_gpu,
            node_positions_gpu,
            connectivity_gpu,
            max_elements_per_block
        )
    else:
        search_l2_morton = None

    # Create combined search function ONCE
    search_func = create_search_gpu_fused_with_l2_block_morton(
        n_hops=n_hops,
        search_l2_morton=search_l2_morton
    )

    # Capture mesh arrays at creation time (not passed as dynamic args)
    # This avoids re-JIT compilation and prevents JAX from materializing the entire mesh
    mesh_connectivity_captured = connectivity_gpu
    mesh_node_positions_captured = node_positions_gpu

    # Create RK4 function with L2 Morton search ONCE (closure captures mesh arrays)
    @jax.jit
    def rk4_fused_with_l2_morton_search(
        positions_gpu,
        element_ids_gpu,
        block_ids_gpu,
        dt,
        element_neighbors_gpu,
        velocity_field_gpu
    ):
        """GPU-fused RK4 with L0 + L1 + L2 (block Morton) search.

        Mesh arrays (connectivity, node_positions) captured in closure to avoid
        passing as dynamic arguments (prevents JAX from materializing entire mesh).
        """
        # Stage 1: k1 = f(t, y)
        element_ids_k1 = search_func(
            positions_gpu,
            element_ids_gpu,
            block_ids_gpu,
            mesh_node_positions_captured,
            mesh_connectivity_captured,
            element_neighbors_gpu
        )
        velocities_k1 = interpolate_velocity_batch_gpu(
            positions_gpu,
            element_ids_k1,
            mesh_connectivity_captured,
            mesh_node_positions_captured,
            velocity_field_gpu
        )
        positions_k1 = positions_gpu + 0.5 * dt * velocities_k1

        # Update block IDs for k1 positions (if domain provided)
        if domain_bounds is not None and grid_size is not None:
            block_ids_k1 = compute_block_ids_batch(positions_k1, domain_bounds, grid_size)
        else:
            block_ids_k1 = block_ids_gpu

        # Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
        element_ids_k2 = search_func(
            positions_k1,
            element_ids_k1,
            block_ids_k1,
            mesh_node_positions_captured,
            mesh_connectivity_captured,
            element_neighbors_gpu
        )
        velocities_k2 = interpolate_velocity_batch_gpu(
            positions_k1,
            element_ids_k2,
            mesh_connectivity_captured,
            mesh_node_positions_captured,
            velocity_field_gpu
        )
        positions_k2 = positions_gpu + 0.5 * dt * velocities_k2

        # Update block IDs for k2 positions
        if domain_bounds is not None and grid_size is not None:
            block_ids_k2 = compute_block_ids_batch(positions_k2, domain_bounds, grid_size)
        else:
            block_ids_k2 = block_ids_gpu

        # Stage 3: k3 = f(t + dt/2, y + dt/2 * k2)
        element_ids_k3 = search_func(
            positions_k2,
            element_ids_k2,
            block_ids_k2,
            mesh_node_positions_captured,
            mesh_connectivity_captured,
            element_neighbors_gpu
        )
        velocities_k3 = interpolate_velocity_batch_gpu(
            positions_k2,
            element_ids_k3,
            mesh_connectivity_captured,
            mesh_node_positions_captured,
            velocity_field_gpu
        )
        positions_k3 = positions_gpu + dt * velocities_k3

        # Update block IDs for k3 positions
        if domain_bounds is not None and grid_size is not None:
            block_ids_k3 = compute_block_ids_batch(positions_k3, domain_bounds, grid_size)
        else:
            block_ids_k3 = block_ids_gpu

        # Stage 4: k4 = f(t + dt, y + dt * k3)
        element_ids_k4 = search_func(
            positions_k3,
            element_ids_k3,
            block_ids_k3,
            mesh_node_positions_captured,
            mesh_connectivity_captured,
            element_neighbors_gpu
        )
        velocities_k4 = interpolate_velocity_batch_gpu(
            positions_k3,
            element_ids_k4,
            mesh_connectivity_captured,
            mesh_node_positions_captured,
            velocity_field_gpu
        )

        # Final position: y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        positions_final_gpu = positions_gpu + (dt / 6.0) * (
            velocities_k1 + 2.0 * velocities_k2 + 2.0 * velocities_k3 + velocities_k4
        )

        # Update block IDs for final positions
        if domain_bounds is not None and grid_size is not None:
            block_ids_final = compute_block_ids_batch(positions_final_gpu, domain_bounds, grid_size)
        else:
            block_ids_final = block_ids_gpu

        # Final search at new positions
        element_ids_final_gpu = search_func(
            positions_final_gpu,
            element_ids_gpu,
            block_ids_final,
            mesh_node_positions_captured,
            mesh_connectivity_captured,
            element_neighbors_gpu
        )

        return positions_final_gpu, element_ids_final_gpu

    def rk4_step_gpu_fused_for_production_with_l2_block_morton_impl(
        particle_data: dict,
        velocity_field,  # Can be np.ndarray OR jax.Array
        dt: float,
        mesh_gpu: MeshDataGPU,
        current_time: float = 0.0
    ) -> Tuple[dict, dict]:
        """
        Production wrapper for GPU-fused RK4 with L2 block Morton fallback.

        Parameters
        ----------
        particle_data : dict
            Particle data with keys 'positions', 'element_ids', 'particle_ids', 'active_mask'
        velocity_field : np.ndarray or jax.Array
            Velocity field at nodes
        dt : float
            Time step size
        mesh_gpu : MeshDataGPU
            GPU-resident mesh data
        current_time : float, optional
            Current simulation time

        Returns
        -------
        particle_data_updated : dict
            Updated particle data
        stats : dict
            Timing statistics

        Expected Performance (with 3-hop + L2 Morton):
        - Throughput: ~40-48k p/s
        - Hit rate: >99.95%
        - Retention: >80% at 2,500 steps
        - Memory overhead: <1%
        """
        # Extract particle data
        positions = particle_data.positions
        element_ids = particle_data.element_ids

        # Compute block IDs from positions (if domain_bounds and grid_size provided)
        if domain_bounds is not None and grid_size is not None:
            # Upload positions to GPU for block ID computation
            if isinstance(positions, np.ndarray):
                positions_gpu_temp = jax.device_put(positions.astype(np.float32))
            else:
                positions_gpu_temp = positions

            # Compute block IDs
            block_ids_gpu = compute_block_ids_batch(
                positions_gpu_temp,
                domain_bounds,
                grid_size
            )
        else:
            # No block computation - use zeros (L2 will be disabled)
            n_particles = len(positions)
            block_ids_gpu = jnp.zeros(n_particles, dtype=jnp.int32)

        # NO MORE @jax.jit HERE - function already created above
        # Just call the pre-compiled function
        t_total = time.time()

        # Upload initial state to GPU (ONE upload per timestep)
        t_upload = time.time()
        if isinstance(positions, np.ndarray):
            positions_gpu = jax.device_put(positions.astype(np.float32))
        else:
            positions_gpu = positions

        if isinstance(element_ids, np.ndarray):
            element_ids_gpu = jax.device_put(element_ids.astype(np.int32))
        else:
            element_ids_gpu = element_ids

        # Only upload velocity field if it's a numpy array (not already on GPU)
        if isinstance(velocity_field, np.ndarray):
            velocity_field_gpu = jax.device_put(velocity_field.astype(np.float32))
        else:
            velocity_field_gpu = velocity_field
        t_upload = time.time() - t_upload

        # Run GPU-fused RK4 with L2 Morton search
        t_compute = time.time()
        positions_final_gpu, element_ids_final_gpu = rk4_fused_with_l2_morton_search(
            positions_gpu,
            element_ids_gpu,
            block_ids_gpu,
            dt,
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

        # Update particle data
        from dataclasses import replace
        particle_data_updated = replace(
            particle_data,
            positions=positions_final,
            element_ids=element_ids_final
        )

        stats = {
            'time_upload': t_upload,
            'time_compute': t_compute,
            'time_download': t_download,
            'time_total': t_total,
            'n_particles': len(positions)
        }

        return particle_data_updated, stats

    return rk4_step_gpu_fused_for_production_with_l2_block_morton_impl


def rk4_step_gpu_fused_with_block_fallback(
    positions: np.ndarray,
    element_ids: np.ndarray,
    block_ids: np.ndarray,        # NEW: block IDs for each particle
    dt: float,
    mesh_gpu: MeshDataGPU,
    velocity_field,               # Can be np.ndarray OR jax.Array
    n_hops: int = 3,
    block_lists: Optional[BlockElementLists] = None
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    GPU-fused RK4 with block-local fallback support.

    This function adds block-local fallback to catch particles that fail
    L1 multi-hop search, preventing particle loss in refined regions.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Initial particle positions (CPU)
    element_ids : np.ndarray, shape (N,)
        Initial element IDs (CPU)
    block_ids : np.ndarray, shape (N,)
        Block ID for each particle (CPU)
    dt : float
        Time step size
    mesh_gpu : MeshDataGPU
        GPU-resident mesh data
    velocity_field : np.ndarray or jax.Array
        Velocity field at nodes
    n_hops : int, default=3
        Number of hops for L1 neighbor search
    block_lists : BlockElementLists, optional
        Block element lists for fallback. If None, no fallback is used.

    Returns
    -------
    positions_final : np.ndarray, shape (N, 3)
        Final particle positions (CPU)
    element_ids_final : np.ndarray, shape (N,)
        Final element IDs (CPU)
    stats : dict
        Timing statistics
    """
    # Create search function with block fallback
    search_func = create_search_gpu_fused_with_block_fallback(
        n_hops=n_hops,
        block_lists=block_lists
    )

    # Determine if we need block IDs
    needs_block_ids = block_lists is not None

    # Create RK4 function with this search
    if needs_block_ids:
        @jax.jit
        def rk4_fused_with_search_and_fallback(
            positions_gpu,
            element_ids_gpu,
            block_ids_gpu,        # NEW
            dt,
            connectivity_gpu,
            node_positions_gpu,
            element_neighbors_gpu,
            velocity_field_gpu
        ):
            """GPU-fused RK4 with block fallback."""
            # Stage 1: k1 = f(t, y)
            element_ids_k1 = search_func(
                positions_gpu,
                element_ids_gpu,
                block_ids_gpu,  # NEW
                node_positions_gpu,
                connectivity_gpu,
                element_neighbors_gpu
            )
            velocities_k1 = interpolate_velocity_batch_gpu(
                positions_gpu, element_ids_k1,
                connectivity_gpu, node_positions_gpu, velocity_field_gpu
            )
            positions_k1 = positions_gpu + 0.5 * dt * velocities_k1

            # Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
            element_ids_k2 = search_func(
                positions_k1, element_ids_k1, block_ids_gpu,
                node_positions_gpu, connectivity_gpu, element_neighbors_gpu
            )
            velocities_k2 = interpolate_velocity_batch_gpu(
                positions_k1, element_ids_k2,
                connectivity_gpu, node_positions_gpu, velocity_field_gpu
            )
            positions_k2 = positions_gpu + 0.5 * dt * velocities_k2

            # Stage 3: k3 = f(t + dt/2, y + dt/2 * k2)
            element_ids_k3 = search_func(
                positions_k2, element_ids_k2, block_ids_gpu,
                node_positions_gpu, connectivity_gpu, element_neighbors_gpu
            )
            velocities_k3 = interpolate_velocity_batch_gpu(
                positions_k2, element_ids_k3,
                connectivity_gpu, node_positions_gpu, velocity_field_gpu
            )
            positions_k3 = positions_gpu + dt * velocities_k3

            # Stage 4: k4 = f(t + dt, y + dt * k3)
            element_ids_k4 = search_func(
                positions_k3, element_ids_k3, block_ids_gpu,
                node_positions_gpu, connectivity_gpu, element_neighbors_gpu
            )
            velocities_k4 = interpolate_velocity_batch_gpu(
                positions_k3, element_ids_k4,
                connectivity_gpu, node_positions_gpu, velocity_field_gpu
            )

            # Final position: y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            positions_final_gpu = positions_gpu + (dt / 6.0) * (
                velocities_k1 + 2.0 * velocities_k2 + 2.0 * velocities_k3 + velocities_k4
            )

            # Final search at new positions
            element_ids_final_gpu = search_func(
                positions_final_gpu, element_ids_gpu, block_ids_gpu,
                node_positions_gpu, connectivity_gpu, element_neighbors_gpu
            )

            return positions_final_gpu, element_ids_final_gpu

        rk4_func = rk4_fused_with_search_and_fallback
    else:
        # No fallback, use standard RK4 (same as rk4_step_gpu_fused_wrapper)
        raise ValueError("Block fallback requires block_lists to be provided")

    t_total = time.time()

    # Upload initial state to GPU
    t_upload = time.time()
    positions_gpu = jax.device_put(positions.astype(np.float32))
    element_ids_gpu = jax.device_put(element_ids.astype(np.int32))
    block_ids_gpu = jax.device_put(block_ids.astype(np.int32))  # NEW

    # Upload velocity field if needed
    if isinstance(velocity_field, np.ndarray):
        velocity_field_gpu = jax.device_put(velocity_field.astype(np.float32))
    else:
        velocity_field_gpu = velocity_field
    t_upload = time.time() - t_upload

    # Execute GPU-fused RK4 with fallback (all on GPU, no transfers)
    t_compute = time.time()
    positions_final_gpu, element_ids_final_gpu = rk4_func(
        positions_gpu,
        element_ids_gpu,
        block_ids_gpu,  # NEW
        dt,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        mesh_gpu.element_neighbors,
        velocity_field_gpu
    )
    # Force GPU computation to complete
    positions_final_gpu.block_until_ready()
    t_compute = time.time() - t_compute

    # Download final state from GPU
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


def rk4_step_gpu_fused_for_production_hierarchical(
    particle_data,
    velocity_field: np.ndarray,
    dt: float,
    mesh_gpu: MeshDataGPU,
    current_time: float = 0.0,
    n_hops: int = 5
):
    """
    GPU-fused RK4 wrapper for production script with HIERARCHICAL early-exit search.

    This uses the hierarchical early-exit implementation to enable 5-hop search
    without GPU memory overflow.

    Matches the interface of rk4_step_with_incremental_search() but uses hierarchical
    5-hop search for better particle retention.

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
    n_hops : int, default=5
        Number of hops for hierarchical L1 neighbor search:
        - 3: ~84 neighbors max (baseline for comparison)
        - 4: ~256 neighbors max (99.95% hit rate)
        - 5: ~1,024 neighbors max (99.99% hit rate, recommended)

    Returns
    -------
    new_particle_data : ParticleData
        Updated particle state
    rk4_stats : dict
        Statistics

    Expected Performance:
    - Throughput: 8-15k p/s (vs 23k for 3-hop)
    - Hit rate: 99.99% (vs 99.9% for 3-hop)
    - Retention: 82% at 2,500 steps (vs 16% for 3-hop)
    - Memory: 10 MB (vs 572 MB OOM for naive 5-hop)
    """
    from dataclasses import replace

    # Call GPU-fused RK4 with hierarchical search
    positions_new, element_ids_new, stats = rk4_step_gpu_fused_wrapper_hierarchical(
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


def rk4_step_gpu_fused_for_production_with_block_fallback(
    particle_data,
    velocity_field: np.ndarray,
    dt: float,
    mesh_gpu: MeshDataGPU,
    block_lists: Optional[BlockElementLists] = None,
    current_time: float = 0.0,
    n_hops: int = 3
):
    """
    GPU-fused RK4 wrapper with block-local fallback for production script.

    Same as rk4_step_gpu_fused_for_production() but adds block-local fallback
    for particles that fail L1 multi-hop search.

    Block-local fallback:
    - Searches only within particle's assigned block (1-450k elements)
    - Avoids expensive global search (3.5M elements)
    - Targets refined regions where 80-90% of failures occur
    - Expected improvement: 99.91% → 99.99% hit rate (77.9% vs 7.8% retention)
    - Performance impact: ~7% slower (42k vs 45k p/s)

    Parameters
    ----------
    particle_data : ParticleData
        Current particle state (must have block_ids attribute)
    velocity_field : np.ndarray
        Velocity field at nodes
    dt : float
        Time step size
    mesh_gpu : MeshDataGPU
        GPU-resident mesh
    block_lists : BlockElementLists, optional
        Block element lists for fallback. If None, no fallback is used.
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

    # Call GPU-fused RK4 with block fallback
    positions_new, element_ids_new, stats = rk4_step_gpu_fused_with_block_fallback(
        particle_data.positions,
        particle_data.element_ids,
        particle_data.block_ids,  # NEW: Pass block IDs
        dt,
        mesh_gpu,
        velocity_field,
        n_hops=n_hops,
        block_lists=block_lists
    )

    # Update particle data (keep velocities unchanged - will be computed next step)
    new_particle_data = replace(
        particle_data,
        positions=positions_new,
        element_ids=element_ids_new
    )

    return new_particle_data, stats


def create_rk4_step_octree_only(
    octree_metadata: jax.Array,
    octree_elements: jax.Array,
    max_octree_depth: int = 10
):
    """
    Create production RK4 wrapper with octree-only search (no L0, no L1).
    
    This tests pure octree search performance by bypassing L0 cache and L1 neighbor search.
    All particles use direct octree search at every RK4 stage.
    
    Purpose:
    - Measure baseline octree search performance
    - Compare with multilevel search (L0+L1+L2)
    - Understand overhead of hierarchical search
    
    Parameters
    ----------
    octree_metadata : jax.Array
        Octree node metadata on GPU
    octree_elements : jax.Array
        Octree element lists on GPU
    max_octree_depth : int, default=10
        Maximum octree traversal depth
    
    Returns
    -------
    rk4_step_func : callable
        Function with signature (particle_data, velocity_field, dt, mesh_gpu, current_time)
    """
    
    @jax.jit
    def search_octree_only(
        positions_gpu: jax.Array,
        cached_element_ids_gpu: jax.Array,
        mesh_gpu_node_positions: jax.Array,
        mesh_gpu_connectivity: jax.Array,
        mesh_gpu_element_neighbors: jax.Array
    ) -> jax.Array:
        """
        Octree-only search: Skip L0 and L1, go directly to octree.
        
        This function ignores cached_element_ids and always searches the octree.
        """
        # Pass -1 as cached_element_ids to force octree search for all particles
        # (particles with cached_id < 0 will always trigger octree search)
        dummy_cached_ids = jnp.full_like(cached_element_ids_gpu, -1, dtype=jnp.int32)
        
        # Direct octree search for all particles
        element_ids_gpu = search_level2_octree_scan(
            positions_gpu,
            dummy_cached_ids,  # Force all particles through octree
            octree_metadata,
            octree_elements,
            mesh_gpu_node_positions,
            mesh_gpu_connectivity,
            max_depth=max_octree_depth
        )
        
        return element_ids_gpu
    
    def rk4_step_octree_only_impl(
        particle_data: dict,
        velocity_field,
        dt: float,
        mesh_gpu: MeshDataGPU,
        current_time: float = 0.0
    ) -> Tuple[dict, dict]:
        """
        Production wrapper for GPU-fused RK4 with octree-only search.
        
        Parameters
        ----------
        particle_data : dict
            Particle data with keys 'positions', 'element_ids', 'particle_ids', 'active_mask'
        velocity_field : np.ndarray or jax.Array
            Velocity field at nodes
        dt : float
            Time step size
        mesh_gpu : MeshDataGPU
            GPU-resident mesh data
        current_time : float, optional
            Current simulation time
        
        Returns
        -------
        particle_data_updated : dict
            Updated particle data
        stats : dict
            Timing statistics
        """
        # Extract particle data
        positions = particle_data.positions
        element_ids = particle_data.element_ids
        
        # Create RK4 function with octree-only search
        @jax.jit
        def rk4_fused_octree_only(
            positions_gpu,
            element_ids_gpu,
            dt,
            connectivity_gpu,
            node_positions_gpu,
            element_neighbors_gpu,
            velocity_field_gpu
        ):
            """GPU-fused RK4 with octree-only search (no L0, no L1)."""
            # Stage 1: k1 = f(t, y)
            element_ids_k1 = search_octree_only(
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
            element_ids_k2 = search_octree_only(
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
            element_ids_k3 = search_octree_only(
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
            element_ids_k4 = search_octree_only(
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
            
            # Combine stages: y_new = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            positions_final = positions_gpu + (dt / 6.0) * (
                velocities_k1 + 2.0 * velocities_k2 + 2.0 * velocities_k3 + velocities_k4
            )
            
            # Final element search at new position
            element_ids_final = search_octree_only(
                positions_final,
                element_ids_k4,
                node_positions_gpu,
                connectivity_gpu,
                element_neighbors_gpu
            )
            
            return positions_final, element_ids_final
        
        # Timing: Upload to GPU (ONE upload)
        t_total = time.time()
        t_upload = time.time()
        
        # Upload positions and element IDs to GPU (if not already on GPU)
        if isinstance(positions, np.ndarray):
            positions_gpu = jax.device_put(positions.astype(np.float32))
        else:
            positions_gpu = positions
            
        if isinstance(element_ids, np.ndarray):
            element_ids_gpu = jax.device_put(element_ids.astype(np.int32))
        else:
            element_ids_gpu = element_ids
        
        # Upload velocity field if needed (or assume already on GPU)
        if isinstance(velocity_field, np.ndarray):
            velocity_field_gpu = jax.device_put(velocity_field.astype(np.float32))
        else:
            velocity_field_gpu = velocity_field
        
        t_upload = time.time() - t_upload
        
        # Timing: GPU computation (all 4 RK4 stages + 5 searches)
        t_compute = time.time()
        positions_final_gpu, element_ids_final_gpu = rk4_fused_octree_only(
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
        
        # Timing: Download from GPU (ONE download)
        t_download = time.time()
        positions_final = np.array(positions_final_gpu, dtype=np.float32)
        element_ids_final = np.array(element_ids_final_gpu, dtype=np.int32)
        t_download = time.time() - t_download
        
        t_total = time.time() - t_total
        
        # Update particle data
        from dataclasses import replace
        particle_data_updated = replace(
            particle_data,
            positions=positions_final,
            element_ids=element_ids_final
        )
        
        stats = {
            'time_upload': t_upload,
            'time_compute': t_compute,
            'time_download': t_download,
            'time_total': t_total,
            'n_particles': len(positions)
        }
        
        return particle_data_updated, stats
    
    # Return the inner function (reusable wrapper)
    return rk4_step_octree_only_impl


# ============================================================================
# HOT Morton Integration
# ============================================================================

def create_rk4_step_gpu_fused_for_production_with_hot_morton(
    mesh_gpu_hot,
    n_hops: int = 3
):
    """
    Create production RK4 wrapper with HOT Morton L2 search.

    This version uses HOT Morton with LOCAL connectivity per leaf to solve the
    JAX OOM issue from Phase 2 block Morton (4.88 TiB allocation).

    Key Innovation: Search accesses only PRE-FETCHED local arrays (fixed-size),
    avoiding dynamic global mesh indexing that causes JAX OOM.

    Architecture:
    - L0: Cached element (point-in-tet test)
    - L1: Multi-hop neighbor search (3-5 hops)
    - L2: HOT Morton with local connectivity (OOM-safe)

    Expected Performance:
    - Throughput: 40-50k p/s (similar to Phase 2 target)
    - L0+L1 Hit Rate: 99.9% (same as baseline)
    - L2 Hit Rate: >99.9% (local search)
    - Retention: >95% at 2,500 steps
    - Memory: ~100-800 MB (vs 8 MB Phase 2, but OOM-safe)

    Parameters
    ----------
    mesh_gpu_hot : MeshGPUHOT
        GPU-resident HOT Morton structures with local connectivity
    n_hops : int, default=3
        Number of hops for L1 neighbor search:
        - 2: ~20 neighbors (95-98% hit rate, fastest)
        - 3: ~84 neighbors (98-99.5% hit rate, recommended)
        - 4: ~340 neighbors (99.5-99.9% hit rate)
        - 5: ~1,024 neighbors (99.99% hit rate, hierarchical only)

    Returns
    -------
    rk4_step_func : callable
        Function with signature (particle_data, velocity_field, dt, mesh_gpu, current_time)
    """
    from jaxtrace.gpu.search.hot_morton_search import search_hot_morton_single_particle

    # Create search function with L0 + L1 + L2 HOT Morton
    if n_hops <= 4:
        # Use standard multi-hop search
        l1_search = search_level1_multihop_vectorized
    else:
        # Use hierarchical early-exit search for 5+ hops
        l1_search = search_level1_multihop_hierarchical

    @jax.jit
    def search_l0_l1_l2_hot(
        positions_gpu: jax.Array,
        cached_element_ids_gpu: jax.Array,
        block_ids_gpu: jax.Array,
        mesh_gpu_node_positions: jax.Array,
        mesh_gpu_connectivity: jax.Array,
        mesh_gpu_element_neighbors: jax.Array
    ) -> jax.Array:
        """
        L0 + L1 + L2 HOT Morton search with local connectivity.

        Args:
            positions_gpu: (N, 3) float32
            cached_element_ids_gpu: (N,) int32 - from previous timestep
            block_ids_gpu: (N,) int32 - coarse block IDs
            mesh_gpu_*: standard mesh arrays

        Returns:
            element_ids: (N,) int32
        """
        # L0: Cached element (point-in-tet test)
        element_ids_l0 = search_level0_vectorized(
            positions_gpu,
            cached_element_ids_gpu,
            mesh_gpu_node_positions,
            mesh_gpu_connectivity
        )

        # L1: Multi-hop neighbor search (for particles that missed L0)
        element_ids_l1 = l1_search(
            positions_gpu,
            element_ids_l0,
            mesh_gpu_node_positions,
            mesh_gpu_connectivity,
            mesh_gpu_element_neighbors,
            n_hops=n_hops
        )

        # L2: HOT Morton with local connectivity (for particles that missed L0+L1)
        def search_l2_single(pos, elem_id_l1, block_id):
            # Only search if L1 failed
            need_l2 = elem_id_l1 < 0

            # Compute Morton code
            from jaxtrace.gpu.search.hot_morton_search import (
                compute_morton_code_from_position_jax,
                find_leaf_for_morton_code
            )

            morton_code = compute_morton_code_from_position_jax(
                pos,
                mesh_gpu_hot.domain_bounds[:3],  # min
                mesh_gpu_hot.domain_bounds[3:]   # max
            )

            # Find leaf
            leaf_id = find_leaf_for_morton_code(morton_code, block_id, mesh_gpu_hot)

            # Search leaf using local connectivity (OOM-safe)
            elem_id_l2 = search_hot_morton_single_particle(pos, block_id, leaf_id, mesh_gpu_hot)

            # Return L2 result if L1 failed, otherwise keep L1 result
            return jnp.where(need_l2, elem_id_l2, elem_id_l1)

        # Vmap L2 search over all particles
        element_ids_final = jax.vmap(search_l2_single)(positions_gpu, element_ids_l1, block_ids_gpu)

        return element_ids_final

    def rk4_step_hot_morton_impl(
        particle_data,
        velocity_field,
        dt: float,
        mesh_gpu: MeshDataGPU,
        current_time: float = 0.0
    ):
        """
        Production wrapper for GPU-fused RK4 with HOT Morton L2 search.

        Parameters
        ----------
        particle_data : ParticleData
            Particle data with positions, element_ids, block_ids
        velocity_field : np.ndarray or jax.Array
            Velocity field at nodes
        dt : float
            Time step size
        mesh_gpu : MeshDataGPU
            GPU-resident mesh data (standard mesh arrays)
        current_time : float
            Current simulation time

        Returns
        -------
        particle_data_updated : ParticleData
            Updated particle data
        stats : dict
            Timing statistics
        """
        # Extract particle data
        positions = particle_data.positions
        element_ids = particle_data.element_ids
        block_ids = particle_data.block_ids

        # Create RK4 function with HOT Morton search
        @jax.jit
        def rk4_fused_hot_morton(
            positions_gpu,
            element_ids_gpu,
            block_ids_gpu,
            dt,
            connectivity_gpu,
            node_positions_gpu,
            element_neighbors_gpu,
            velocity_field_gpu
        ):
            """GPU-fused RK4 with L0+L1+L2 HOT Morton search."""

            # Stage 1: k1 = f(t, y)
            element_ids_k1 = search_l0_l1_l2_hot(
                positions_gpu,
                element_ids_gpu,
                block_ids_gpu,
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

            # Recompute block IDs at k1 positions
            from jaxtrace.gpu.search.hot_morton_search import compute_block_id_from_position_hot
            block_ids_k1 = jax.vmap(
                lambda pos: compute_block_id_from_position_hot(
                    pos, mesh_gpu_hot.domain_bounds, mesh_gpu_hot.grid_size
                )
            )(positions_k1)

            # Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
            element_ids_k2 = search_l0_l1_l2_hot(
                positions_k1,
                element_ids_k1,
                block_ids_k1,
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

            # Recompute block IDs at k2 positions
            block_ids_k2 = jax.vmap(
                lambda pos: compute_block_id_from_position_hot(
                    pos, mesh_gpu_hot.domain_bounds, mesh_gpu_hot.grid_size
                )
            )(positions_k2)

            # Stage 3: k3 = f(t + dt/2, y + dt/2 * k2)
            element_ids_k3 = search_l0_l1_l2_hot(
                positions_k2,
                element_ids_k2,
                block_ids_k2,
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

            # Recompute block IDs at k3 positions
            block_ids_k3 = jax.vmap(
                lambda pos: compute_block_id_from_position_hot(
                    pos, mesh_gpu_hot.domain_bounds, mesh_gpu_hot.grid_size
                )
            )(positions_k3)

            # Stage 4: k4 = f(t + dt, y + dt * k3)
            element_ids_k4 = search_l0_l1_l2_hot(
                positions_k3,
                element_ids_k3,
                block_ids_k3,
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

            # Combine stages: y_new = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            positions_final = positions_gpu + (dt / 6.0) * (
                velocities_k1 + 2.0 * velocities_k2 + 2.0 * velocities_k3 + velocities_k4
            )

            # Recompute block IDs at final positions
            block_ids_final = jax.vmap(
                lambda pos: compute_block_id_from_position_hot(
                    pos, mesh_gpu_hot.domain_bounds, mesh_gpu_hot.grid_size
                )
            )(positions_final)

            # Final element search at new position
            element_ids_final = search_l0_l1_l2_hot(
                positions_final,
                element_ids_k4,
                block_ids_final,
                node_positions_gpu,
                connectivity_gpu,
                element_neighbors_gpu
            )

            return positions_final, element_ids_final, block_ids_final

        # Timing: Upload to GPU
        t_total = time.time()
        t_upload = time.time()

        # Upload positions, element IDs, block IDs to GPU
        if isinstance(positions, np.ndarray):
            positions_gpu = jax.device_put(positions.astype(np.float32))
        else:
            positions_gpu = positions

        if isinstance(element_ids, np.ndarray):
            element_ids_gpu = jax.device_put(element_ids.astype(np.int32))
        else:
            element_ids_gpu = element_ids

        if isinstance(block_ids, np.ndarray):
            block_ids_gpu = jax.device_put(block_ids.astype(np.int32))
        else:
            block_ids_gpu = block_ids

        # Upload velocity field if needed
        if isinstance(velocity_field, np.ndarray):
            velocity_field_gpu = jax.device_put(velocity_field.astype(np.float32))
        else:
            velocity_field_gpu = velocity_field

        t_upload = time.time() - t_upload

        # Timing: GPU computation (all 4 RK4 stages + 5 searches)
        t_compute = time.time()
        positions_final_gpu, element_ids_final_gpu, block_ids_final_gpu = rk4_fused_hot_morton(
            positions_gpu,
            element_ids_gpu,
            block_ids_gpu,
            dt,
            mesh_gpu.connectivity,
            mesh_gpu.node_positions,
            mesh_gpu.element_neighbors,
            velocity_field_gpu
        )
        # Force GPU computation to complete
        positions_final_gpu.block_until_ready()
        t_compute = time.time() - t_compute

        # Timing: Download from GPU
        t_download = time.time()
        positions_final = np.array(positions_final_gpu, dtype=np.float32)
        element_ids_final = np.array(element_ids_final_gpu, dtype=np.int32)
        block_ids_final = np.array(block_ids_final_gpu, dtype=np.int32)
        t_download = time.time() - t_download

        t_total = time.time() - t_total

        # Update particle data
        from dataclasses import replace
        particle_data_updated = replace(
            particle_data,
            positions=positions_final,
            element_ids=element_ids_final,
            block_ids=block_ids_final
        )

        stats = {
            'time_upload': t_upload,
            'time_compute': t_compute,
            'time_download': t_download,
            'time_total': t_total,
            'n_particles': len(positions)
        }

        return particle_data_updated, stats

    # Return the inner function (reusable wrapper)
    return rk4_step_hot_morton_impl
