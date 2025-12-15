"""
HOT Morton GPU Search - JAX-Compatible Search with Local Connectivity

This module implements the GPU search kernel for HOT Morton with LOCAL connectivity
per leaf, solving the JAX OOM issue from Phase 2.

Key Innovation: Search accesses only PRE-FETCHED local arrays (fixed-size, static shapes),
avoiding dynamic global mesh indexing that causes JAX to materialize 4.88 TiB.

Search Strategy:
1. Compute block ID from particle position
2. Find octree leaf in block using Morton code binary search
3. Search elements in leaf using LOCAL connectivity (OOM-safe)
4. Return global element ID if found, -1 otherwise
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class MeshGPUHOT:
    """GPU mesh data for HOT Morton search with local connectivity."""

    # Block metadata
    n_blocks: int
    block_bbox_min: jax.Array  # (n_blocks, 3) float32
    block_bbox_max: jax.Array  # (n_blocks, 3) float32

    # Octree leaf metadata (per block)
    n_leaves_per_block: jax.Array  # (n_blocks,) int32
    max_leaves_per_block: int
    max_leaf_capacity: int

    # Leaf structures (n_blocks, max_leaves_per_block)
    leaf_morton_start: jax.Array   # (n_blocks, max_leaves, 2) int64 - [low, high]
    leaf_elem_count: jax.Array     # (n_blocks, max_leaves) int32

    # LOCAL connectivity per leaf (CRITICAL - avoids global mesh access)
    leaf_local_connectivity: jax.Array  # (n_blocks, max_leaves, max_capacity, 4) int32
    leaf_node_coords: jax.Array         # (n_blocks, max_leaves, max_local_nodes, 3) float32
    leaf_n_local_nodes: jax.Array       # (n_blocks, max_leaves) int32
    leaf_global_elem_ids: jax.Array     # (n_blocks, max_leaves, max_capacity) int32 - for result mapping

    max_local_nodes: int
    grid_size: Tuple[int, int, int]
    domain_bounds: jax.Array  # (6,) float32 - [xmin, xmax, ymin, ymax, zmin, zmax]


# ============================================================================
# Morton Code Utilities (GPU)
# ============================================================================

@jax.jit
def interleave_bits_3d_jax(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
    """
    Interleave 3 x 21-bit integers into a 63-bit Morton code (JAX version).

    This is a JAX-compatible version using only JAX ops.
    """
    def expand_bits_jax(v: jax.Array) -> jax.Array:
        """Expand bits: abc → a00b00c00 (JAX version)"""
        v = v & 0x1fffff  # 21 bits
        v = (v | (v << 32)) & 0x1f00000000ffff
        v = (v | (v << 16)) & 0x1f0000ff0000ff
        v = (v | (v << 8)) & 0x100f00f00f00f00f
        v = (v | (v << 4)) & 0x10c30c30c30c30c3
        v = (v | (v << 2)) & 0x1249249249249249
        return v

    return expand_bits_jax(x) | (expand_bits_jax(y) << 1) | (expand_bits_jax(z) << 2)


@jax.jit
def compute_morton_code_from_position_jax(
    position: jax.Array,
    bbox_min: jax.Array,
    bbox_max: jax.Array,
    morton_resolution: int = 2097151  # 2^21 - 1
) -> jax.Array:
    """
    Compute Morton code for a 3D position (JAX version).

    Args:
        position: (3,) float32
        bbox_min: (3,) float32 - domain bounds min
        bbox_max: (3,) float32 - domain bounds max
        morton_resolution: quantization resolution

    Returns:
        morton_code: int64 scalar
    """
    # Normalize to [0, 1]
    normalized = (position - bbox_min) / (bbox_max - bbox_min)
    normalized = jnp.clip(normalized, 0.0, 1.0)

    # Quantize to integer grid
    x = (normalized[0] * morton_resolution).astype(jnp.int64)
    y = (normalized[1] * morton_resolution).astype(jnp.int64)
    z = (normalized[2] * morton_resolution).astype(jnp.int64)

    return interleave_bits_3d_jax(x, y, z)


# ============================================================================
# Block ID Computation
# ============================================================================

@jax.jit
def compute_block_id_from_position_hot(
    position: jax.Array,
    domain_bounds: jax.Array,
    grid_size: Tuple[int, int, int]
) -> jax.Array:
    """
    Compute block ID from 3D position.

    Args:
        position: (3,) float32
        domain_bounds: (6,) float32 - [xmin, xmax, ymin, ymax, zmin, zmax]
        grid_size: (nx, ny, nz)

    Returns:
        block_id: int32 scalar (-1 if out of bounds)
    """
    nx, ny, nz = grid_size

    dx = (domain_bounds[1] - domain_bounds[0]) / nx
    dy = (domain_bounds[3] - domain_bounds[2]) / ny
    dz = (domain_bounds[5] - domain_bounds[4]) / nz

    i = jnp.floor((position[0] - domain_bounds[0]) / dx).astype(jnp.int32)
    j = jnp.floor((position[1] - domain_bounds[2]) / dy).astype(jnp.int32)
    k = jnp.floor((position[2] - domain_bounds[4]) / dz).astype(jnp.int32)

    valid = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (k >= 0) & (k < nz)
    block_id = i + j * nx + k * nx * ny

    return jnp.where(valid, block_id, jnp.int32(-1))


# ============================================================================
# Octree Leaf Lookup
# ============================================================================

@jax.jit
def find_leaf_for_morton_code(
    morton_code: jax.Array,
    block_id: jax.Array,
    mesh_gpu: MeshGPUHOT
) -> jax.Array:
    """
    Find octree leaf index for a given Morton code using binary search.

    Args:
        morton_code: int64 scalar
        block_id: int32 scalar
        mesh_gpu: MeshGPUHOT with leaf metadata

    Returns:
        leaf_id: int32 scalar (-1 if not found or out of bounds)
    """
    # Check block validity
    valid_block = (block_id >= 0) & (block_id < mesh_gpu.n_blocks)

    n_leaves = mesh_gpu.n_leaves_per_block[block_id]

    # Binary search through leaves
    def binary_search_body(carry):
        low, high, found_leaf = carry

        mid = (low + high) // 2

        # Get Morton range for this leaf
        morton_range = mesh_gpu.leaf_morton_start[block_id, mid]  # (2,) - [low, high]
        morton_low = morton_range[0]
        morton_high = morton_range[1]

        # Check if morton_code is in range
        in_range = (morton_code >= morton_low) & (morton_code <= morton_high)

        # Update search range
        new_found = jnp.where(in_range, mid, found_leaf)
        new_low = jnp.where(in_range, low, jnp.where(morton_code < morton_low, low, mid + 1))
        new_high = jnp.where(in_range, high, jnp.where(morton_code < morton_low, mid - 1, high))

        return (new_low, new_high, new_found)

    def binary_search_cond(carry):
        low, high, found_leaf = carry
        return (low <= high) & (found_leaf == -1)

    # Initial state
    init_carry = (jnp.int32(0), n_leaves - 1, jnp.int32(-1))

    # Run binary search (bounded by max_leaves_per_block)
    final_low, final_high, final_leaf = lax.while_loop(
        binary_search_cond,
        binary_search_body,
        init_carry
    )

    # Return leaf ID if found and block is valid
    return jnp.where(valid_block & (final_leaf >= 0), final_leaf, jnp.int32(-1))


# ============================================================================
# Point-in-Tet Test (Using Local Connectivity)
# ============================================================================

@jax.jit
def point_in_tet_local_jax(position: jax.Array, tet_nodes: jax.Array) -> jax.Array:
    """
    Point-in-tetrahedron test using barycentric coordinates.

    Args:
        position: (3,) float32 - query point
        tet_nodes: (4, 3) float32 - tetrahedron vertices

    Returns:
        inside: bool scalar
    """
    # Compute barycentric coordinates using Cramer's rule
    v0 = tet_nodes[0]
    v1 = tet_nodes[1]
    v2 = tet_nodes[2]
    v3 = tet_nodes[3]

    # Edge vectors
    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0
    p = position - v0

    # Compute determinants
    def det3x3(a, b, c):
        return jnp.dot(a, jnp.cross(b, c))

    d0 = det3x3(e1, e2, e3)
    d1 = det3x3(p, e2, e3)
    d2 = det3x3(e1, p, e3)
    d3 = det3x3(e1, e2, p)

    # Barycentric coordinates
    tol = 1e-10
    safe_d0 = jnp.where(jnp.abs(d0) < tol, 1.0, d0)

    lambda1 = d1 / safe_d0
    lambda2 = d2 / safe_d0
    lambda3 = d3 / safe_d0
    lambda0 = 1.0 - lambda1 - lambda2 - lambda3

    # Check if all coordinates are non-negative (with tolerance)
    tol_bary = -1e-6
    inside = (lambda0 >= tol_bary) & (lambda1 >= tol_bary) & (lambda2 >= tol_bary) & (lambda3 >= tol_bary)

    return inside


# ============================================================================
# L2 HOT Morton Search (Using Local Connectivity)
# ============================================================================

@jax.jit
def search_hot_morton_single_particle(
    position: jax.Array,
    block_id: jax.Array,
    leaf_id: jax.Array,
    mesh_gpu: MeshGPUHOT
) -> jax.Array:
    """
    Search for containing element in a single HOT Morton leaf using LOCAL connectivity.

    This function is OOM-SAFE because it accesses only PRE-FETCHED local arrays:
    - leaf_local_connectivity[block_id, leaf_id]: (max_capacity, 4) int32
    - leaf_node_coords[block_id, leaf_id]: (max_local_nodes, 3) float32

    NO dynamic global mesh access → NO JAX OOM!

    Args:
        position: (3,) float32 - particle position
        block_id: int32 scalar
        leaf_id: int32 scalar
        mesh_gpu: MeshGPUHOT with local connectivity

    Returns:
        element_id: int32 scalar - global element ID if found, -1 otherwise
    """
    # Check validity
    valid = (block_id >= 0) & (block_id < mesh_gpu.n_blocks) & (leaf_id >= 0)

    # Get LOCAL structures for this leaf (pre-fetched, fixed-size)
    local_connectivity = mesh_gpu.leaf_local_connectivity[block_id, leaf_id]  # (max_capacity, 4)
    local_node_coords = mesh_gpu.leaf_node_coords[block_id, leaf_id]          # (max_local_nodes, 3)
    global_elem_ids = mesh_gpu.leaf_global_elem_ids[block_id, leaf_id]       # (max_capacity,)
    elem_count = mesh_gpu.leaf_elem_count[block_id, leaf_id]

    # Bounded loop over elements in leaf
    def test_one_elem(j, found_elem):
        # Check if this slot is valid
        valid_elem = j < elem_count
        global_elem_id = global_elem_ids[j]

        # Get LOCAL connectivity (NO global mesh access!)
        local_node_ids = local_connectivity[j]  # (4,) int32 - local indices

        # Get node coords using LOCAL indexing (static shape)
        tet_nodes = local_node_coords[local_node_ids]  # (4, 3) float32

        # Point-in-tet test
        inside = point_in_tet_local_jax(position, tet_nodes) & valid_elem

        # Return global element ID if found (first match)
        return jnp.where(inside & (found_elem == -1), global_elem_id, found_elem)

    # Bounded loop (JAX-compatible)
    found_elem = lax.fori_loop(
        0,
        mesh_gpu.max_leaf_capacity,
        test_one_elem,
        jnp.int32(-1)
    )

    return jnp.where(valid, found_elem, jnp.int32(-1))


# ============================================================================
# Complete L2 HOT Morton Search Function
# ============================================================================

def create_level2_hot_morton_search(
    mesh_gpu: MeshGPUHOT
) -> callable:
    """
    Create L2 HOT Morton search function with mesh data captured in closure.

    Args:
        mesh_gpu: MeshGPUHOT with all preprocessed structures

    Returns:
        search_func: (positions, block_ids) → element_ids
    """

    @jax.jit
    def search_hot_morton_batch(
        positions: jax.Array,
        block_ids: jax.Array
    ) -> jax.Array:
        """
        Search for containing elements using HOT Morton with local connectivity.

        Args:
            positions: (N, 3) float32
            block_ids: (N,) int32

        Returns:
            element_ids: (N,) int32
        """

        def search_single_particle(pos, block_id):
            # Compute Morton code
            morton_code = compute_morton_code_from_position_jax(
                pos,
                mesh_gpu.domain_bounds[:3],  # min
                mesh_gpu.domain_bounds[3:]   # max
            )

            # Find leaf
            leaf_id = find_leaf_for_morton_code(morton_code, block_id, mesh_gpu)

            # Search leaf using local connectivity
            elem_id = search_hot_morton_single_particle(pos, block_id, leaf_id, mesh_gpu)

            return elem_id

        # Vmap over particles
        element_ids = jax.vmap(search_single_particle)(positions, block_ids)

        return element_ids

    return search_hot_morton_batch


def create_level2_hot_morton_search_unconditional(
    mesh_gpu: MeshGPUHOT
) -> callable:
    """
    Create UNCONDITIONAL L2 HOT Morton search (for initial particle assignment).

    This version computes block IDs internally and returns element IDs directly.

    Args:
        mesh_gpu: MeshGPUHOT with all preprocessed structures

    Returns:
        search_func: (positions,) → element_ids
    """

    @jax.jit
    def search_hot_morton_unconditional(positions: jax.Array) -> jax.Array:
        """
        Search for containing elements (unconditional version).

        Args:
            positions: (N, 3) float32

        Returns:
            element_ids: (N,) int32
        """

        def search_single_particle(pos):
            # Compute block ID
            block_id = compute_block_id_from_position_hot(
                pos, mesh_gpu.domain_bounds, mesh_gpu.grid_size
            )

            # Compute Morton code
            morton_code = compute_morton_code_from_position_jax(
                pos,
                mesh_gpu.domain_bounds[:3],  # min
                mesh_gpu.domain_bounds[3:]   # max
            )

            # Find leaf
            leaf_id = find_leaf_for_morton_code(morton_code, block_id, mesh_gpu)

            # Search leaf using local connectivity
            elem_id = search_hot_morton_single_particle(pos, block_id, leaf_id, mesh_gpu)

            return elem_id

        # Vmap over particles
        element_ids = jax.vmap(search_single_particle)(positions)

        return element_ids

    return search_hot_morton_unconditional


# ============================================================================
# Upload HOT Morton Structures to GPU
# ============================================================================

def upload_hot_morton_structures_to_gpu(
    hot_structures,
    domain_bounds: np.ndarray,
    grid_size: Tuple[int, int, int]
) -> MeshGPUHOT:
    """
    Upload HOT Morton structures to GPU.

    Args:
        hot_structures: HOTMortonStructures from preprocessing
        domain_bounds: (6,) float32 - [xmin, xmax, ymin, ymax, zmin, zmax]
        grid_size: (nx, ny, nz)

    Returns:
        MeshGPUHOT with all arrays on GPU
    """
    return MeshGPUHOT(
        n_blocks=hot_structures.n_blocks,
        block_bbox_min=jax.device_put(hot_structures.block_bbox_min),
        block_bbox_max=jax.device_put(hot_structures.block_bbox_max),
        n_leaves_per_block=jax.device_put(hot_structures.n_leaves_per_block),
        max_leaves_per_block=hot_structures.max_leaves_per_block,
        max_leaf_capacity=hot_structures.max_leaf_capacity,
        leaf_morton_start=jax.device_put(hot_structures.leaf_morton_start),
        leaf_elem_count=jax.device_put(hot_structures.leaf_elem_count),
        leaf_local_connectivity=jax.device_put(hot_structures.leaf_local_connectivity),
        leaf_node_coords=jax.device_put(hot_structures.leaf_node_coords),
        leaf_n_local_nodes=jax.device_put(hot_structures.leaf_n_local_nodes),
        leaf_global_elem_ids=jax.device_put(hot_structures.leaf_global_elem_ids),
        max_local_nodes=hot_structures.max_local_nodes,
        grid_size=grid_size,
        domain_bounds=jax.device_put(domain_bounds)
    )
