#!/usr/bin/env python3
"""
Global Morton Search - Phase 2: GPU Search Kernel

JAX-compatible functions for Morton-based element search on GPU.
Works with global Morton-sorted element list (NO blocks).

Architecture:
- Position → Morton code → Leaf ID → Bounded search
- Single-particle functions (vmapped externally)
- No nested jit/vmap
- Bounded loops with lax.fori_loop
"""

import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass
from typing import Tuple
import numpy as np


# ============================================================================
# GPU Data Structure
# ============================================================================

@dataclass
class MeshGPUGlobalMorton:
    """GPU-resident global Morton structure for L2 search."""

    # Core mesh data (already on GPU)
    connectivity: jax.Array          # (n_elements, 4) int32
    node_positions: jax.Array        # (n_nodes, 3) float32

    # Morton structure (uploaded from GlobalMortonStructure)
    elem_ids_sorted: jax.Array       # (n_elements,) int32 - global Morton order
    morton_sorted: jax.Array         # (n_elements,) uint64 - sorted Morton codes
    leaf_start: jax.Array            # (n_leaves,) int32 - start index in elem_ids_sorted
    leaf_length: jax.Array           # (n_leaves,) int32 - elements in this leaf

    # Octree prefix table for position→leaf mapping (NEW - Phase 5)
    prefix_start: jax.Array          # (8^D,) int32 - prefix→first_leaf_id lookup
    prefix_length: jax.Array         # (8^D,) int32 - prefix→num_leaves lookup
    table_depth: jnp.int32           # Number of octree levels in prefix table

    # Morton parameters
    morton_min: jnp.uint64           # Minimum Morton code in mesh
    morton_max: jnp.uint64           # Maximum Morton code in mesh
    bbox_min: jax.Array              # (3,) float32 - global bbox
    bbox_max: jax.Array              # (3,) float32

    # Configuration
    n_leaves: jnp.int32              # Total number of leaves
    max_depth: jnp.int32             # Morton encoding depth (bits per dimension)
    leaf_capacity: jnp.int32         # Maximum elements per leaf

    # L0/L1 data (if needed, can be added later)
    # element_neighbors: jax.Array  # (n_elements, 4) int32
    # extended_neighbors: jax.Array # (n_elements, max_extended) int32


# ============================================================================
# Morton Encoding (JAX-compatible)
# ============================================================================

def interleave_bits_3d_jax(x: jnp.uint32, y: jnp.uint32, z: jnp.uint32) -> jnp.uint64:
    """
    Interleave bits of (x, y, z) to create Morton code.

    JAX-compatible version using shifts and masks.
    Supports up to 21 bits per dimension (63 bits total).

    Args:
        x, y, z: Unsigned 32-bit integers in range [0, 2^21 - 1]

    Returns:
        Morton code as uint64
    """
    # Expand bits using magic masks (spreads bits with 2 zeros between each bit)
    # This is the fast bit-twiddling method

    # Convert to uint64 for operations
    x = x.astype(jnp.uint64)
    y = y.astype(jnp.uint64)
    z = z.astype(jnp.uint64)

    # Expand x (position 0, 3, 6, 9, ...)
    x = (x | (x << 32)) & jnp.uint64(0x001f00000000ffff)
    x = (x | (x << 16)) & jnp.uint64(0x001f0000ff0000ff)
    x = (x | (x <<  8)) & jnp.uint64(0x100f00f00f00f00f)
    x = (x | (x <<  4)) & jnp.uint64(0x10c30c30c30c30c3)
    x = (x | (x <<  2)) & jnp.uint64(0x1249249249249249)

    # Expand y (position 1, 4, 7, 10, ...)
    y = (y | (y << 32)) & jnp.uint64(0x001f00000000ffff)
    y = (y | (y << 16)) & jnp.uint64(0x001f0000ff0000ff)
    y = (y | (y <<  8)) & jnp.uint64(0x100f00f00f00f00f)
    y = (y | (y <<  4)) & jnp.uint64(0x10c30c30c30c30c3)
    y = (y | (y <<  2)) & jnp.uint64(0x1249249249249249)

    # Expand z (position 2, 5, 8, 11, ...)
    z = (z | (z << 32)) & jnp.uint64(0x001f00000000ffff)
    z = (z | (z << 16)) & jnp.uint64(0x001f0000ff0000ff)
    z = (z | (z <<  8)) & jnp.uint64(0x100f00f00f00f00f)
    z = (z | (z <<  4)) & jnp.uint64(0x10c30c30c30c30c3)
    z = (z | (z <<  2)) & jnp.uint64(0x1249249249249249)

    # Interleave: x at bit 0, y at bit 1, z at bit 2, repeat
    return x | (y << 1) | (z << 2)


def morton_encode_position_jax(
    pos: jax.Array,
    bbox_min: jax.Array,
    bbox_max: jax.Array,
    max_depth: jnp.int32
) -> jnp.uint64:
    """
    Encode 3D position to Morton code on GPU.

    Args:
        pos: (3,) float32 - position in world coordinates
        bbox_min: (3,) float32 - global bounding box minimum
        bbox_max: (3,) float32 - global bounding box maximum
        max_depth: int32 - bits per dimension (typically 21)

    Returns:
        Morton code as uint64
    """
    # Normalize position to [0, 1] within bbox
    normalized = (pos - bbox_min) / (bbox_max - bbox_min)

    # Clamp to [0, 1] to handle boundary cases
    normalized = jnp.clip(normalized, 0.0, 1.0)

    # Scale to integer grid [0, 2^max_depth - 1]
    grid_max = (2 ** max_depth) - 1
    u = jnp.floor(normalized * grid_max).astype(jnp.uint32)

    # Interleave bits
    return interleave_bits_3d_jax(u[0], u[1], u[2])


# ============================================================================
# Position to Leaf Mapping
# ============================================================================

def morton_binary_search_leaf(
    morton_code: jnp.uint64,
    morton_sorted: jax.Array,
    leaf_capacity: jnp.int32
) -> jnp.int32:
    """
    Binary search to find leaf containing given Morton code.

    Since leaves are fixed-capacity segments of the sorted Morton array,
    we search for the index where morton_code would be inserted, then
    compute leaf_id = index // leaf_capacity.

    Uses lax.while_loop for JAX compatibility.

    Args:
        morton_code: uint64 - query Morton code
        morton_sorted: (n_elements,) uint64 - sorted Morton codes
        leaf_capacity: int32 - elements per leaf

    Returns:
        leaf_id: int32 - leaf containing elements near this Morton code
    """
    n_elements = morton_sorted.shape[0]

    # Binary search state: (left, right)
    def cond_fun(state):
        left, right = state
        return left < right

    def body_fun(state):
        left, right = state
        mid = (left + right) // 2
        mid_morton = morton_sorted[mid]

        # If query < mid, search left half
        # If query >= mid, search right half
        new_left = jnp.where(morton_code < mid_morton, left, mid + 1)
        new_right = jnp.where(morton_code < mid_morton, mid, right)

        return (new_left, new_right)

    # Initial state
    init_state = (jnp.int32(0), jnp.int32(n_elements))

    # Run binary search
    final_left, final_right = lax.while_loop(cond_fun, body_fun, init_state)

    # final_left is the insertion point
    # Clamp to valid element index range
    idx = jnp.clip(final_left, 0, n_elements - 1)

    # Compute leaf ID
    leaf_id = idx // leaf_capacity

    # Clamp to valid leaf range
    n_leaves = (n_elements + leaf_capacity - 1) // leaf_capacity
    return jnp.clip(leaf_id, 0, n_leaves - 1)


def position_to_leaf_id_octree(
    pos: jax.Array,
    mesh_gpu: MeshGPUGlobalMorton
) -> jnp.int32:
    """
    Map position to leaf ID using octree prefix table with range search.

    Since multiple leaves at depth D+1 can share the same prefix at depth D,
    the prefix table stores a range of leaves. We then search within that
    range to find the exact leaf containing the position.

    Algorithm:
    1. Compute Morton code for position
    2. Extract prefix bits (top table_depth * 3 bits)
    3. Get leaf range from prefix_start[prefix] and prefix_length[prefix]
    4. Search within range for leaf containing this Morton code

    Args:
        pos: (3,) float32 - query position
        mesh_gpu: GPU-resident Morton structure with prefix_start/prefix_length

    Returns:
        leaf_id: int32 in range [0, n_leaves - 1]
    """
    # 1. Compute Morton code for position
    m = morton_encode_position_jax(
        pos,
        mesh_gpu.bbox_min,
        mesh_gpu.bbox_max,
        mesh_gpu.max_depth
    )

    # 2. Extract prefix bits (top table_depth * 3 bits)
    table_depth_int = int(mesh_gpu.table_depth)
    prefix_bits_int = table_depth_int * 3
    shift_amount = 63 - prefix_bits_int

    # Right-shift Morton code to extract prefix
    prefix = lax.shift_right_logical(m, jnp.uint64(shift_amount))
    prefix = prefix.astype(jnp.int32)

    # 3. Get leaf range from prefix table
    prefix = jnp.clip(prefix, 0, mesh_gpu.prefix_start.shape[0] - 1)
    first_leaf = mesh_gpu.prefix_start[prefix]
    num_leaves = mesh_gpu.prefix_length[prefix]

    # 4. Search within leaf range for exact match
    # If only one leaf, return it directly
    # If multiple leaves, find which one contains this Morton code
    def check_leaf(leaf_idx):
        """Check if Morton code m is in this leaf's range."""
        start_idx = mesh_gpu.leaf_start[leaf_idx]
        length = mesh_gpu.leaf_length[leaf_idx]

        # Check if m is in [morton_sorted[start], morton_sorted[start+length-1]]
        morton_first = mesh_gpu.morton_sorted[start_idx]
        morton_last = mesh_gpu.morton_sorted[start_idx + jnp.maximum(length - 1, 0)]

        return (m >= morton_first) & (m <= morton_last)

    # Linear search through candidate leaves (usually ≤8 leaves)
    best_leaf = first_leaf

    # Unroll loop for JAX (max 8 iterations for depth-7 leaves sharing depth-6 prefix)
    for offset in range(8):
        leaf_idx = first_leaf + offset
        # Only check if within valid range
        is_valid = (offset < num_leaves) & (leaf_idx < mesh_gpu.n_leaves)
        matches = is_valid & check_leaf(leaf_idx)
        # Update best_leaf if this one matches
        best_leaf = jnp.where(matches, leaf_idx, best_leaf)

    # Safety clipping
    leaf_id = jnp.clip(best_leaf, 0, mesh_gpu.n_leaves - 1)

    return leaf_id


# DEPRECATED: Binary search fallback (kept for backward compatibility)
def position_to_leaf_id(
    pos: jax.Array,
    mesh_gpu: MeshGPUGlobalMorton
) -> jnp.int32:
    """
    Map position to leaf ID using binary search on Morton codes.

    DEPRECATED: This assumes fixed-capacity leaves, which is incorrect.
    Use position_to_leaf_id_octree() instead for octree-based leaves.

    More accurate than linear approximation, but still wrong for adaptive octree.

    Args:
        pos: (3,) float32 - query position
        mesh_gpu: GPU-resident Morton structure

    Returns:
        leaf_id: int32 in range [0, n_leaves - 1]
    """
    # Compute Morton code for position
    m = morton_encode_position_jax(
        pos,
        mesh_gpu.bbox_min,
        mesh_gpu.bbox_max,
        mesh_gpu.max_depth
    )

    # Binary search in sorted Morton array
    leaf_id = morton_binary_search_leaf(
        m,
        mesh_gpu.morton_sorted,
        mesh_gpu.leaf_capacity
    )

    return leaf_id


# Deprecated: linear approximation (kept for reference)
def position_to_leaf_id_linear(
    pos: jax.Array,
    mesh_gpu: MeshGPUGlobalMorton
) -> jnp.int32:
    """
    Map position to leaf ID using linear approximation.

    DEPRECATED: Only works for uniformly distributed Morton codes.
    Use position_to_leaf_id() instead (binary search).

    Args:
        pos: (3,) float32 - query position
        mesh_gpu: GPU-resident Morton structure

    Returns:
        leaf_id: int32 in range [0, n_leaves - 1]
    """
    # Compute Morton code for position
    m = morton_encode_position_jax(
        pos,
        mesh_gpu.bbox_min,
        mesh_gpu.bbox_max,
        mesh_gpu.max_depth
    )

    # Normalize to [0, 1] along Morton curve
    morton_range = mesh_gpu.morton_max - mesh_gpu.morton_min + jnp.uint64(1)
    t = (m - mesh_gpu.morton_min).astype(jnp.float32) / morton_range.astype(jnp.float32)

    # Map to leaf index
    leaf_id_float = t * mesh_gpu.n_leaves.astype(jnp.float32)
    leaf_id = jnp.floor(leaf_id_float).astype(jnp.int32)

    # Clamp to valid range
    return jnp.clip(leaf_id, 0, mesh_gpu.n_leaves - 1)


# ============================================================================
# Point-in-Tetrahedron Test (GPU)
# ============================================================================

def point_in_tet_gpu(
    pos: jax.Array,
    elem_id: jnp.int32,
    connectivity: jax.Array,
    node_positions: jax.Array
) -> jnp.bool_:
    """
    Test if position is inside tetrahedron using barycentric coordinates.

    Returns True if point is inside (all barycentric coords >= 0).

    Args:
        pos: (3,) float32 - query position
        elem_id: int32 - element ID to test
        connectivity: (n_elements, 4) int32
        node_positions: (n_nodes, 3) float32

    Returns:
        inside: bool - True if pos is in element
    """
    # Get node indices
    nodes = connectivity[elem_id]  # (4,)

    # Get node positions
    p0 = node_positions[nodes[0]]  # (3,)
    p1 = node_positions[nodes[1]]
    p2 = node_positions[nodes[2]]
    p3 = node_positions[nodes[3]]

    # Compute vectors from p0
    v1 = p1 - p0
    v2 = p2 - p0
    v3 = p3 - p0
    vp = pos - p0

    # Solve for barycentric coordinates using Cramer's rule
    # [v1 v2 v3] * [b1, b2, b3]^T = vp
    # b0 = 1 - b1 - b2 - b3

    # Compute 3x3 determinant: det([v1 v2 v3])
    det = (v1[0] * (v2[1] * v3[2] - v2[2] * v3[1]) -
           v1[1] * (v2[0] * v3[2] - v2[2] * v3[0]) +
           v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]))

    # Handle degenerate tetrahedron
    is_degenerate = jnp.abs(det) < 1e-12

    # Compute barycentric coordinates
    det_inv = jnp.where(is_degenerate, 1.0, 1.0 / det)

    # b1 = det([vp v2 v3]) / det
    b1 = ((vp[0] * (v2[1] * v3[2] - v2[2] * v3[1]) -
           vp[1] * (v2[0] * v3[2] - v2[2] * v3[0]) +
           vp[2] * (v2[0] * v3[1] - v2[1] * v3[0])) * det_inv)

    # b2 = det([v1 vp v3]) / det
    b2 = ((v1[0] * (vp[1] * v3[2] - vp[2] * v3[1]) -
           v1[1] * (vp[0] * v3[2] - vp[2] * v3[0]) +
           v1[2] * (vp[0] * v3[1] - vp[1] * v3[0])) * det_inv)

    # b3 = det([v1 v2 vp]) / det
    b3 = ((v1[0] * (v2[1] * vp[2] - v2[2] * vp[1]) -
           v1[1] * (v2[0] * vp[2] - v2[2] * vp[0]) +
           v1[2] * (v2[0] * vp[1] - v2[1] * vp[0])) * det_inv)

    # b0 = 1 - b1 - b2 - b3
    b0 = 1.0 - b1 - b2 - b3

    # Check if all barycentric coordinates are non-negative
    # Use small tolerance for numerical stability
    tol = -1e-6
    inside = (b0 >= tol) & (b1 >= tol) & (b2 >= tol) & (b3 >= tol) & (~is_degenerate)

    return inside


# ============================================================================
# Bounded Leaf Search (Fixed-capacity loop)
# ============================================================================

def search_in_leaf_global(
    pos: jax.Array,
    leaf_id: jnp.int32,
    mesh_gpu: MeshGPUGlobalMorton
) -> jnp.int32:
    """
    Search for element containing pos within a single leaf.

    Uses bounded lax.fori_loop with fixed upper bound (leaf_capacity).
    Masks inactive iterations when j >= actual leaf_length.

    This is the core bounded search pattern for JAX/GPU.

    Args:
        pos: (3,) float32 - query position
        leaf_id: int32 - which leaf to search
        mesh_gpu: GPU-resident Morton structure

    Returns:
        elem_id: int32 - found element ID, or -1 if not found
    """
    # Get leaf parameters
    start = mesh_gpu.leaf_start[leaf_id]
    length = mesh_gpu.leaf_length[leaf_id]

    # Bounded loop body
    def body(j: jnp.int32, found_elem: jnp.int32) -> jnp.int32:
        """
        Check element j in leaf if not yet found.

        Args:
            j: iteration index [0, leaf_capacity)
            found_elem: current found element (-1 if none)

        Returns:
            Updated found_elem
        """
        # Active only if: (1) not yet found, (2) j < actual length
        active = (found_elem == -1) & (j < length)

        # Get global element ID
        idx = start + j
        elem_id = jnp.where(active, mesh_gpu.elem_ids_sorted[idx], jnp.int32(0))

        # Test point-in-tet (masked by active)
        inside = jnp.where(
            active,
            point_in_tet_gpu(pos, elem_id, mesh_gpu.connectivity, mesh_gpu.node_positions),
            False
        )

        # Update found_elem if inside and active
        return jnp.where(inside & active, elem_id, found_elem)

    # Run bounded loop from 0 to leaf_capacity
    init = jnp.int32(-1)
    found_elem = lax.fori_loop(0, mesh_gpu.leaf_capacity, body, init)

    return found_elem


# ============================================================================
# L2 Search (Single Particle)
# ============================================================================

def search_L2_global_morton_single(
    pos: jax.Array,
    mesh_gpu: MeshGPUGlobalMorton,
    search_radius: jnp.int32 = jnp.int32(1)
) -> jnp.int32:
    """
    L2 search using global Morton structure for SINGLE particle.

    Searches the predicted leaf and its neighbors along the Morton curve.
    This accounts for the fact that a point may be inside an element whose
    centroid's Morton code is in a different leaf.

    IMPORTANT: This function is NOT @jax.jit decorated.
    It will be vmapped externally in the search hierarchy.

    Steps:
    1. Position → Morton code → Leaf ID (binary search)
    2. Search within leaf and ±search_radius neighbors (bounded loops)

    Args:
        pos: (3,) float32 - query position
        mesh_gpu: GPU-resident Morton structure
        search_radius: int32 - search ±radius leaves (default 1)

    Returns:
        elem_id: int32 - found element, or -1 if not found
    """
    # Map position to leaf using appropriate method
    # Use octree prefix table if available (O(1)), otherwise fallback to binary search
    center_leaf_id = jnp.where(
        mesh_gpu.table_depth > 0,
        position_to_leaf_id_octree(pos, mesh_gpu),
        position_to_leaf_id(pos, mesh_gpu)
    )

    # Search center leaf first
    elem_id = search_in_leaf_global(pos, center_leaf_id, mesh_gpu)

    # If found, return immediately
    found = elem_id >= 0

    # Search neighboring leaves if not found
    # Check leaves: center-radius, ..., center-1, center+1, ..., center+radius
    def search_neighbor(i, state):
        elem_id, found = state

        # Skip if already found
        active = ~found

        # Compute neighbor offset: maps i ∈ [0, 2*radius] to offsets [-radius, -1, +1, +radius]
        # i=0 → offset=-radius, i=1 → offset=-radius+1, ..., i=radius-1 → offset=-1
        # i=radius → offset=+1, ..., i=2*radius-1 → offset=+radius
        offset = jnp.where(i < search_radius,
                          i - search_radius,  # [-radius, ..., -1]
                          i - search_radius + 1)  # [+1, ..., +radius]

        neighbor_leaf_id = center_leaf_id + offset

        # Clamp to valid leaf range
        neighbor_leaf_id = jnp.clip(neighbor_leaf_id, 0, mesh_gpu.n_leaves - 1)

        # Search in neighbor leaf (masked by active)
        elem_neighbor = jnp.where(
            active,
            search_in_leaf_global(pos, neighbor_leaf_id, mesh_gpu),
            jnp.int32(-1)
        )

        # Update if found
        improve = (elem_neighbor >= 0) & active
        elem_id = jnp.where(improve, elem_neighbor, elem_id)
        found = found | improve

        return (elem_id, found)

    # Search 2*search_radius neighbors
    init_state = (elem_id, found)
    final_elem_id, final_found = lax.fori_loop(
        0,
        2 * search_radius,
        search_neighbor,
        init_state
    )

    return final_elem_id


# ============================================================================
# Upload Function
# ============================================================================

def upload_global_morton_to_gpu(
    morton_struct,  # GlobalMortonStructure from morton_global_builder OR morton_octree_builder
    connectivity: np.ndarray,
    node_positions: np.ndarray
) -> MeshGPUGlobalMorton:
    """
    Upload global Morton structure to GPU.

    Supports both:
    - OLD: Fixed-capacity leaves (from morton_global_builder)
    - NEW: Adaptive octree leaves (from morton_octree_builder)

    Args:
        morton_struct: GlobalMortonStructure from CPU preprocessing
            Must have: elem_ids_sorted, morton_sorted, leaf_start, leaf_length,
                      bbox_min, bbox_max, max_depth, leaf_capacity, n_leaves
            Optional (for octree): prefix_start, prefix_length, table_depth
        connectivity: (n_elements, 4) int32 - mesh connectivity
        node_positions: (n_nodes, 3) float32 - node coordinates

    Returns:
        MeshGPUGlobalMorton with all data on GPU
    """
    # Check if octree structure (has prefix_start/prefix_length)
    has_prefix_table = hasattr(morton_struct, 'prefix_start') and morton_struct.prefix_start is not None

    # Create prefix table arrays (use dummy if not available)
    if has_prefix_table:
        prefix_start_gpu = jax.device_put(morton_struct.prefix_start.astype(np.int32))
        prefix_length_gpu = jax.device_put(morton_struct.prefix_length.astype(np.int32))
        table_depth_val = jnp.int32(morton_struct.table_depth)
    else:
        # Dummy prefix table for backward compatibility
        prefix_start_gpu = jax.device_put(np.array([0], dtype=np.int32))
        prefix_length_gpu = jax.device_put(np.array([0], dtype=np.int32))
        table_depth_val = jnp.int32(0)

    return MeshGPUGlobalMorton(
        # Core mesh data
        connectivity=jax.device_put(connectivity.astype(np.int32)),
        node_positions=jax.device_put(node_positions.astype(np.float32)),

        # Morton structure
        elem_ids_sorted=jax.device_put(morton_struct.elem_ids_sorted.astype(np.int32)),
        morton_sorted=jax.device_put(morton_struct.morton_sorted.astype(np.uint64)),
        leaf_start=jax.device_put(morton_struct.leaf_start.astype(np.int32)),
        leaf_length=jax.device_put(morton_struct.leaf_length.astype(np.int32)),

        # Octree prefix table (NEW - Phase 5)
        prefix_start=prefix_start_gpu,
        prefix_length=prefix_length_gpu,
        table_depth=table_depth_val,

        # Morton parameters
        morton_min=jnp.uint64(morton_struct.morton_sorted.min()),
        morton_max=jnp.uint64(morton_struct.morton_sorted.max()),
        bbox_min=jax.device_put(morton_struct.bbox_min.astype(np.float32)),
        bbox_max=jax.device_put(morton_struct.bbox_max.astype(np.float32)),

        # Configuration
        n_leaves=jnp.int32(morton_struct.n_leaves),
        max_depth=jnp.int32(morton_struct.max_depth),
        leaf_capacity=jnp.int32(morton_struct.leaf_capacity)
    )


# ============================================================================
# Utility: Vectorized Morton Encoding (for batch positions)
# ============================================================================

@jax.jit
def morton_encode_positions_batch(
    positions: jax.Array,
    bbox_min: jax.Array,
    bbox_max: jax.Array,
    max_depth: jnp.int32
) -> jax.Array:
    """
    Vectorized Morton encoding for batch of positions.

    Args:
        positions: (n_particles, 3) float32
        bbox_min, bbox_max: (3,) float32
        max_depth: int32

    Returns:
        morton_codes: (n_particles,) uint64
    """
    encode_one = lambda p: morton_encode_position_jax(p, bbox_min, bbox_max, max_depth)
    return jax.vmap(encode_one)(positions)


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'MeshGPUGlobalMorton',
    'interleave_bits_3d_jax',
    'morton_encode_position_jax',
    'morton_encode_positions_batch',
    'morton_binary_search_leaf',
    'position_to_leaf_id',
    'position_to_leaf_id_linear',  # Deprecated
    'point_in_tet_gpu',
    'search_in_leaf_global',
    'search_L2_global_morton_single',
    'upload_global_morton_to_gpu',
]
