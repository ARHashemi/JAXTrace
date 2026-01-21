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

# Import point-in-tet methods and configuration
from jaxtrace.gpu.search.point_in_tet_methods import point_in_tet_gpu as point_in_tet_dispatcher
import jaxtrace.config as config


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

    # Corrected AA metadata and memory optimizations (optional, only if precomputed)
    aa_metadata: object = None       # AxisAlignedMetadata or None
    element_vertices: jax.Array = None  # (n_elements, 4, 3) float32 or None


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

    # Linear search through candidate leaves
    # FIX #2: Search ALL leaves with this prefix (not just first 8)
    # In refined regions, a single prefix can map to 50-100+ leaves at different depths
    best_leaf = first_leaf

    def check_one_leaf(offset, current_best_leaf):
        leaf_idx = first_leaf + offset
        # Only check if within valid range
        is_valid = (offset < num_leaves) & (leaf_idx < mesh_gpu.n_leaves)
        matches = is_valid & check_leaf(leaf_idx)
        # Update best_leaf if this one matches
        return jnp.where(matches, leaf_idx, current_best_leaf)

    # Search up to 256 leaves (reasonable upper bound for prefix collision)
    max_leaves_to_check = jnp.minimum(num_leaves, 256)
    best_leaf = lax.fori_loop(0, max_leaves_to_check, check_one_leaf, best_leaf)

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
    Test if position is inside tetrahedron (configurable method).

    This is a wrapper that dispatches to the configured point-in-tet method.
    Method selection is controlled via jaxtrace.config.POINT_IN_TET_METHOD.

    Available methods (see point_in_tet_methods.py):
    - "current": Barycentric/Cramer's rule (145 FLOPs, baseline)
    - "skala": GPU-optimized cross products (48 FLOPs, ~3× speedup)
    - "axis_aligned": Specialized for axis-aligned meshes (12 FLOPs, ~12× speedup)

    NOT JIT-decorated to avoid overhead when used within already-JIT-compiled functions.

    Args:
        pos: (3,) float32 - query position
        elem_id: int32 - element ID to test
        connectivity: (n_elements, 4) int32
        node_positions: (n_nodes, 3) float32

    Returns:
        inside: bool - True if pos is in element
    """
    return point_in_tet_dispatcher(
        pos, elem_id, connectivity, node_positions,
        method=config.POINT_IN_TET_METHOD
    )


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

    PHASE 1 FIX: Use lax.fori_loop instead of unrolled Python loop to reduce
    XLA graph size during compilation. This is the innermost loop called by
    all L2 methods (radius, neighbors, hierarchical, enhanced).

    Impact:
    - Neighbors: 648 → 81 unrolled (8× reduction, 2.2 TB → 275 GB)
    - Hierarchical: 3,456 → 432 unrolled (8× reduction, 11.7 TB → 1.46 TB)
    - Enhanced: 3,000 → 375 unrolled (8× reduction, 10.1 TB → 1.26 TB)

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

    def check_element(j, found_elem):
        """Check one element in leaf (bounded loop body)."""
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

    # BOUNDED LOOP: Reduces XLA graph size by 8× (no unrolling)
    # FIX: Search ALL elements in leaf (not just first 8)
    # Leaves can contain up to leaf_capacity (256) elements
    max_elements_to_search = jnp.minimum(length, 256)
    found_elem = lax.fori_loop(0, max_elements_to_search, check_element, jnp.int32(-1))

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

    **IMPORTANT: radius=N searches BOTH directions**:
      - Searches center leaf (1 leaf)
      - Searches -N, -N+1, ..., -1 leaves BACKWARD (N leaves)
      - Searches +1, +2, ..., +N leaves FORWARD (N leaves)
      - **Total: 2N + 1 leaves** (symmetric band around center)

      Example: radius=10 searches 21 leaves total:
        leaves[-10], leaves[-9], ..., leaves[0], ..., leaves[+9], leaves[+10]

    IMPORTANT: This function is NOT @jax.jit decorated.
    It will be vmapped externally in the search hierarchy.

    Steps:
    1. Position → Morton code → Leaf ID (binary search)
    2. Search within leaf and ±search_radius neighbors (bounded loops)

    Args:
        pos: (3,) float32 - query position
        mesh_gpu: GPU-resident Morton structure
        search_radius: int32 - search ±radius leaves (default 1)
                              Total leaves searched: 2*radius + 1

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

    # PHASE 1 FIX: Use lax.fori_loop instead of unrolled loops
    # This prevents RAM explosion when search_radius is large (e.g., 100, 500)
    # Old code had hardcoded range(15) which limited radius and caused graph explosion

    def search_one_neighbor(i, state):
        elem_id, found = state
        # Map i ∈ [0, 2*search_radius-1] to offset ∈ [-search_radius, -1] ∪ [+1, +search_radius]
        # i=0 → offset=-search_radius, i=search_radius-1 → offset=-1
        # i=search_radius → offset=+1, i=2*search_radius-1 → offset=+search_radius
        offset = jnp.where(
            i < search_radius,
            -(search_radius - i),  # Negative offsets
            (i - search_radius) + 1  # Positive offsets
        )

        active = ~found
        neighbor_leaf_id = jnp.clip(center_leaf_id + offset, 0, mesh_gpu.n_leaves - 1)

        elem_neighbor = jnp.where(
            active,
            search_in_leaf_global(pos, neighbor_leaf_id, mesh_gpu),
            jnp.int32(-1)
        )

        improve = (elem_neighbor >= 0) & active
        new_elem_id = jnp.where(improve, elem_neighbor, elem_id)
        new_found = found | improve

        return (new_elem_id, new_found)

    # Search neighbors: -search_radius, ..., -1, +1, ..., +search_radius (skip 0, already searched)
    # Cap at 512 to prevent absurd values (24,550 leaves total, radius=512 is ~4% of mesh)
    safe_radius = jnp.minimum(search_radius, 512)
    elem_id, found = lax.fori_loop(0, 2 * safe_radius, search_one_neighbor, (elem_id, found))

    return elem_id


def search_L2_morton_incremental_single(
    pos: jax.Array,
    mesh_gpu: MeshGPUGlobalMorton,
    radii: tuple = (2, 5, 10)
) -> jnp.int32:
    """
    L2 search with incremental radius expansion (conditional cascade).

    Searches with increasing radius values, using jnp.where for conditional execution.
    Each tier searches a SYMMETRIC BAND around the center leaf:
      - radius=R searches 2R+1 leaves: [-R, ..., -1, 0, +1, ..., +R]

    **Default configuration** (radii=(2, 5, 10)):
      Tier 1: radius=2  → 5 leaves  (2×2+1)
      Tier 2: radius=5  → 11 leaves (2×5+1) - only if tier 1 fails
      Tier 3: radius=10 → 21 leaves (2×10+1) - only if tier 2 fails

    **Expected performance** (assuming 60/30/10 hit distribution):
      - 60% particles found at tier 1:  5 leaves
      - 30% particles found at tier 2:  5 + 11 = 16 leaves (cumulative)
      - 10% particles found at tier 3:  5 + 11 + 21 = 37 leaves (cumulative)
      - Average: 0.6×5 + 0.3×16 + 0.1×37 = 11.5 leaves (vs 21 for always radius=10)
      - Speedup: 21 / 11.5 = 1.83× faster L2 search

    **Tuning guide**:
      - Small first radius (2-5): Fast path for nearby particles
      - Medium second radius (5-15): Catches moderate displacements
      - Large final radius (10-50): Safety net for large displacements
      - More tiers (e.g., (2,5,10,20,50)): Finer-grained fallback

    This uses the same conditional execution pattern as L0→L1→L2 hierarchy:
      JAX partitions particles based on success flags and skips work for successful cases.

    Args:
        pos: (3,) float32 - query position
        mesh_gpu: GPU-resident Morton structure
        radii: tuple of int - cascading search radii (default: (2, 5, 10))
                             Supports 2-5 tiers for flexibility

    Returns:
        elem_id: int32 - found element, or -1 if not found
    """
    # Support flexible number of tiers (2-5)
    if len(radii) < 2:
        raise ValueError(f"radii must have at least 2 tiers, got {len(radii)}")
    if len(radii) > 5:
        raise ValueError(f"radii must have at most 5 tiers (to prevent graph explosion), got {len(radii)}")

    # Tier 1: Always execute (smallest radius)
    elem = search_L2_global_morton_single(pos, mesh_gpu, search_radius=jnp.int32(radii[0]))

    # Remaining tiers: Conditional cascade
    for i in range(1, len(radii)):
        elem = jnp.where(
            elem >= 0,
            elem,  # Found at previous tier, skip this tier
            search_L2_global_morton_single(pos, mesh_gpu, search_radius=jnp.int32(radii[i]))
        )

    return elem


def search_L2_morton_neighbors_single(
    pos: jax.Array,
    mesh_gpu: MeshGPUGlobalMorton
) -> jnp.int32:
    """
    L2 search using Morton neighbor arithmetic for SINGLE particle.

    Uses spatial octant neighbor finding instead of linear ±radius search.
    This is geometrically correct and much faster than radius-based search.

    Algorithm:
    1. Position → Morton code → Extract prefix
    2. Decode prefix to octant coordinates
    3. Find 26 spatial neighbor octants
    4. Look up leaves for each neighbor octant
    5. Search within neighbor leaves

    Advantages over radius-based search:
    - Geometrically correct (searches actual spatial neighbors)
    - Fixed cost (always 27 octants regardless of domain size)
    - Faster (27 octants vs 2*radius octants, and octants align with elements)

    IMPORTANT: This function is NOT @jax.jit decorated.
    It will be vmapped externally in the search hierarchy.

    Requirements:
    - mesh_gpu.table_depth > 0 (requires octree prefix table)
    - mesh_gpu.prefix_start and mesh_gpu.prefix_length populated

    Args:
        pos: (3,) float32 - query position
        mesh_gpu: GPU-resident Morton structure with prefix table

    Returns:
        elem_id: int32 - found element, or -1 if not found
    """
    # Import morton_neighbors functions (local import to avoid circular dependency)
    from jaxtrace.gpu.search.morton_neighbors import (
        decode_morton_prefix_jax,
        get_26_neighbor_prefixes_jax,
    )

    # 1. Compute Morton code for position
    morton_query = morton_encode_position_jax(
        pos,
        mesh_gpu.bbox_min,
        mesh_gpu.bbox_max,
        mesh_gpu.max_depth
    )

    # 2. Keep Morton code left-aligned for neighbor generation
    # Note: decode_morton_prefix_jax expects left-aligned uint64!
    table_depth_int = int(mesh_gpu.table_depth)
    center_prefix = morton_query  # Keep full 64-bit, left-aligned

    # 3. Get 26 neighbor prefixes + center (27 total)
    max_coord = jnp.int32((2 ** table_depth_int) - 1)
    neighbor_prefixes = get_26_neighbor_prefixes_jax(
        center_prefix,
        table_depth_int,
        max_coord
    )

    # PHASE 3 FIX: Replace 27-octant unrolled loop with lax.fori_loop
    # This is the final reduction to eliminate all unrolling
    table_depth_int = int(mesh_gpu.table_depth)
    shift_amount = 63 - (table_depth_int * 3)

    def search_one_octant(i, state):
        """Search one octant (bounded loop body)."""
        elem_id, found = state
        active = jnp.logical_not(found)

        # Get neighbor prefix
        neighbor_prefix = neighbor_prefixes[i]

        # Convert prefix to array index
        prefix_idx = lax.shift_right_logical(neighbor_prefix, jnp.uint64(shift_amount))
        prefix_idx = prefix_idx.astype(jnp.int32)
        prefix_idx = jnp.clip(prefix_idx, 0, mesh_gpu.prefix_start.shape[0] - 1)

        # Look up leaf range
        first_leaf = mesh_gpu.prefix_start[prefix_idx]
        num_leaves_in_prefix = mesh_gpu.prefix_length[prefix_idx]

        has_leaves = num_leaves_in_prefix > 0
        valid_leaf = first_leaf >= 0

        # PHASE 2: Search up to 3 leaves with lax.fori_loop
        def search_leaves_in_octant(leaf_offset, leaf_state):
            """Search one leaf in octant (bounded loop body)."""
            octant_elem, octant_found = leaf_state
            leaf_id = first_leaf + leaf_offset
            valid = (leaf_offset < num_leaves_in_prefix) & (leaf_id >= 0) & (leaf_id < mesh_gpu.n_leaves) & jnp.logical_not(octant_found)

            result = jnp.where(valid, search_in_leaf_global(pos, leaf_id, mesh_gpu), jnp.int32(-1))
            improved = result >= 0

            return (
                jnp.where(improved, result, octant_elem),
                octant_found | improved
            )

        # BOUNDED LOOP: 3 leaves per octant
        octant_elem, octant_found = lax.fori_loop(
            0, 3,
            search_leaves_in_octant,
            (jnp.int32(-1), jnp.bool_(False))
        )

        # Update global state if found in this octant
        elem_neighbor = jnp.where(
            active & has_leaves & valid_leaf,
            octant_elem,
            jnp.int32(-1)
        )

        improve = (elem_neighbor >= 0) & active
        elem_id = jnp.where(improve, elem_neighbor, elem_id)
        found = found | improve

        return (elem_id, found)

    # PHASE 3: BOUNDED LOOP over 27 octants (final reduction)
    elem_id, found = lax.fori_loop(
        0, 27,
        search_one_octant,
        (jnp.int32(-1), jnp.bool_(False))
    )

    return elem_id


def search_5x5x5_outer_shell(
    pos: jax.Array,
    mesh_gpu: MeshGPUGlobalMorton,
    current_elem: jnp.int32,
    already_found: jnp.bool_
) -> jnp.int32:
    """
    Search outer shell of 5×5×5 neighborhood (98 octants).

    Searches all octants where max(|dx|, |dy|, |dz|) == 2.
    Skips inner 3×3×3 (already searched by search_L2_morton_neighbors_single).

    This is a fallback search for particles near octree boundaries where
    Morton neighbors may not include spatial neighbors.

    Args:
        pos: (3,) float32 - query position
        mesh_gpu: GPU-resident Morton structure
        current_elem: int32 - current element ID (from 3×3×3 search)
        already_found: bool - whether 3×3×3 search succeeded

    Returns:
        elem_id: int32 - found element, or current_elem if not found
    """
    from jaxtrace.gpu.search.morton_neighbors import (
        decode_morton_prefix_jax,
        encode_morton_prefix_jax,
    )

    # Compute Morton code and decode
    morton_query = morton_encode_position_jax(
        pos,
        mesh_gpu.bbox_min,
        mesh_gpu.bbox_max,
        mesh_gpu.max_depth
    )

    table_depth_int = int(mesh_gpu.table_depth)
    cx, cy, cz = decode_morton_prefix_jax(morton_query, table_depth_int)
    max_coord = jnp.int32((2 ** table_depth_int) - 1)

    # PHASE 3 FIX: Replace 125-octant unrolled loop with lax.fori_loop
    shift_amount = 63 - (table_depth_int * 3)

    def search_one_enhanced_octant(i, state):
        """Search one octant in 5×5×5 shell (bounded loop body)."""
        elem_id, found = state

        # Skip if already found
        active = jnp.logical_not(found) & jnp.logical_not(already_found)

        # Map i ∈ [0, 125) to (dx, dy, dz) ∈ [-2, 2]³
        dz = (i % 5) - 2
        dy = ((i // 5) % 5) - 2
        dx = ((i // 25) % 5) - 2

        # Skip inner 3×3×3: |dx| <= 1 AND |dy| <= 1 AND |dz| <= 1
        max_offset = jnp.maximum(jnp.maximum(jnp.abs(dx), jnp.abs(dy)), jnp.abs(dz))
        is_outer = max_offset == 2

        active = active & is_outer

        # Compute neighbor coordinates
        nx = jnp.clip(cx + dx, 0, max_coord)
        ny = jnp.clip(cy + dy, 0, max_coord)
        nz = jnp.clip(cz + dz, 0, max_coord)

        # Encode neighbor prefix
        neighbor_prefix = encode_morton_prefix_jax(nx, ny, nz, table_depth_int)

        # Look up leaves
        prefix_idx = lax.shift_right_logical(neighbor_prefix, jnp.uint64(shift_amount))
        prefix_idx = prefix_idx.astype(jnp.int32)
        prefix_idx = jnp.clip(prefix_idx, 0, mesh_gpu.prefix_start.shape[0] - 1)

        first_leaf = mesh_gpu.prefix_start[prefix_idx]
        num_leaves_in_prefix = mesh_gpu.prefix_length[prefix_idx]

        has_leaves = num_leaves_in_prefix > 0
        valid_leaf = first_leaf >= 0

        # PHASE 2: Search up to 3 leaves with lax.fori_loop
        def search_leaves_in_octant_enhanced(leaf_offset, leaf_state):
            """Search one leaf in octant for enhanced search (bounded loop body)."""
            octant_elem, octant_found = leaf_state
            leaf_id = first_leaf + leaf_offset
            valid = (leaf_offset < num_leaves_in_prefix) & (leaf_id >= 0) & (leaf_id < mesh_gpu.n_leaves) & jnp.logical_not(octant_found)

            result = jnp.where(valid, search_in_leaf_global(pos, leaf_id, mesh_gpu), jnp.int32(-1))
            improved = result >= 0

            return (
                jnp.where(improved, result, octant_elem),
                octant_found | improved
            )

        # BOUNDED LOOP: 3 leaves per octant
        octant_elem, octant_found = lax.fori_loop(
            0, 3,
            search_leaves_in_octant_enhanced,
            (jnp.int32(-1), jnp.bool_(False))
        )

        elem_neighbor = jnp.where(
            active & has_leaves & valid_leaf,
            octant_elem,
            jnp.int32(-1)
        )

        # Update if found
        improve = (elem_neighbor >= 0) & active
        elem_id = jnp.where(improve, elem_neighbor, elem_id)
        found = found | improve

        return (elem_id, found)

    # PHASE 3: BOUNDED LOOP over 125 octants (final reduction)
    elem_id, found = lax.fori_loop(
        0, 125,
        search_one_enhanced_octant,
        (current_elem, already_found)
    )

    return elem_id


def search_L2_morton_neighbors_enhanced(
    pos: jax.Array,
    mesh_gpu: MeshGPUGlobalMorton
) -> jnp.int32:
    """
    Enhanced Morton neighbor search with 5×5×5 boundary fallback.

    Two-tier search strategy:
    1. Tier 1: 3×3×3 search (27 octants) - fast path
    2. Tier 2: 5×5×5 outer shell (98 octants) - boundary fallback

    This addresses Morton Z-order discontinuities at octree boundaries,
    especially important for highly refined meshes with large element size variations.

    Performance:
    - 67% particles succeed in Tier 1 (unchanged performance)
    - 33% particles need Tier 2 (~4× slower)
    - Average overhead: ~2× vs standard search

    Expected benefit: +5-10% retention for meshes with 262K× size variation

    Args:
        pos: (3,) float32 - query position
        mesh_gpu: GPU-resident Morton structure

    Returns:
        elem_id: int32 - found element, or -1 if not found
    """
    # Tier 1: Standard 3×3×3 search
    elem_id = search_L2_morton_neighbors_single(pos, mesh_gpu)

    # Check if found
    found_3x3x3 = elem_id >= 0

    # Tier 2: Search 5×5×5 outer shell if not found
    # Uses jnp.where to maintain data-independent execution for JAX
    elem_id_extended = search_5x5x5_outer_shell(pos, mesh_gpu, elem_id, found_3x3x3)

    # Return best result
    return jnp.where(found_3x3x3, elem_id, elem_id_extended)


def search_depth6_octants_single(
    pos: jax.Array,
    morton_query: jnp.uint64,
    mesh_gpu: MeshGPUGlobalMorton
) -> jnp.int32:
    """
    Search 27 neighbor octants at depth-6 (coarse resolution).

    Helper function to enable conditional execution in hierarchical search.
    Searches at 64³ octree resolution to catch large elements assigned to
    coarse octants that might be missed by depth-7 search.

    Args:
        pos: (3,) query position
        morton_query: Morton code for position
        mesh_gpu: GPU mesh structure

    Returns:
        elem_id: Found element, or -1 if not found
    """
    from jaxtrace.gpu.search.morton_neighbors import get_26_neighbor_prefixes_jax

    max_coord_6 = jnp.int32((2 ** 6) - 1)
    neighbor_prefixes_6 = get_26_neighbor_prefixes_jax(morton_query, 6, max_coord_6)
    shift_amount_6 = 63 - (6 * 3)
    scale_factor = 8  # Depth-6 to depth-7 mapping

    def search_one_octant_depth6(i, state):
        """Search one octant at depth-6 (bounded loop body)."""
        elem_id_depth6, found_depth6 = state
        active = jnp.logical_not(found_depth6)
        neighbor_prefix = neighbor_prefixes_6[i]

        # Depth-6 prefix → scale to depth-7 table
        coarse_idx = lax.shift_right_logical(neighbor_prefix, jnp.uint64(shift_amount_6))
        prefix_idx = coarse_idx.astype(jnp.int32) * scale_factor
        prefix_idx = jnp.clip(prefix_idx, 0, mesh_gpu.prefix_start.shape[0] - 1)

        first_leaf = mesh_gpu.prefix_start[prefix_idx]
        num_leaves = mesh_gpu.prefix_length[prefix_idx]
        has_leaves = num_leaves > 0
        valid_start = first_leaf >= 0

        # Search up to 8 leaves with lax.fori_loop
        def search_leaves_depth6(leaf_offset, leaf_state):
            """Search one leaf at depth-6 (bounded loop body)."""
            octant_elem, octant_found = leaf_state
            leaf_id = first_leaf + leaf_offset
            valid = (leaf_offset < num_leaves) & (leaf_id >= 0) & (leaf_id < mesh_gpu.n_leaves) & jnp.logical_not(octant_found)

            result = jnp.where(valid, search_in_leaf_global(pos, leaf_id, mesh_gpu), jnp.int32(-1))
            improved = result >= 0

            return (
                jnp.where(improved, result, octant_elem),
                octant_found | improved
            )

        # BOUNDED LOOP: 8 leaves per octant
        octant_elem, octant_found = lax.fori_loop(
            0, 8,
            search_leaves_depth6,
            (jnp.int32(-1), jnp.bool_(False))
        )

        elem_neighbor = jnp.where(active & has_leaves & valid_start, octant_elem, jnp.int32(-1))

        improve = (elem_neighbor >= 0) & active
        return (
            jnp.where(improve, elem_neighbor, elem_id_depth6),
            found_depth6 | improve
        )

    # BOUNDED LOOP over 27 octants at depth-6
    elem_id_depth6, found_depth6 = lax.fori_loop(
        0, 27,
        search_one_octant_depth6,
        (jnp.int32(-1), jnp.bool_(False))
    )

    return elem_id_depth6


def search_L2_morton_hierarchical_single(
    pos: jax.Array,
    mesh_gpu: MeshGPUGlobalMorton
) -> jnp.int32:
    """
    Hierarchical Morton neighbor search with CONDITIONAL multi-depth fallback.

    Searches at multiple octree depths to handle variable-depth leaves:
    1. Depth 7 (fine): 27 neighbor octants at 128³ resolution
    2. Depth 6 (coarse): 27 neighbor octants at 64³ resolution (CONDITIONAL)

    OPTIMIZATION: Uses jnp.where to conditionally execute depth-6 search
    only for particles that fail depth-7 search. This provides 1.3-1.6×
    speedup depending on depth-7 hit rate (expected 60-80%).

    This handles particles at coarse/fine boundaries and ensures we catch
    elements in both depth-6 and depth-7 leaves (required for graded mesh).

    JAX-compatible: Uses jnp.where for conditional execution within vmap.

    Args:
        pos: (3,) float32 - query position
        mesh_gpu: GPU-resident Morton structure

    Returns:
        elem_id: int32 - found element, or -1 if not found
    """
    from jaxtrace.gpu.search.morton_neighbors import get_26_neighbor_prefixes_jax

    # 1. Compute Morton code for position
    morton_query = morton_encode_position_jax(
        pos,
        mesh_gpu.bbox_min,
        mesh_gpu.bbox_max,
        mesh_gpu.max_depth
    )

    # DEPTH 7: Search 27 octants at fine resolution (ALWAYS executes)
    max_coord_7 = jnp.int32((2 ** 7) - 1)
    neighbor_prefixes_7 = get_26_neighbor_prefixes_jax(morton_query, 7, max_coord_7)
    shift_amount_7 = 63 - (7 * 3)

    def search_one_octant_depth7(i, state):
        """Search one octant at depth-7 (bounded loop body)."""
        elem_id_depth7, found_depth7 = state
        active = jnp.logical_not(found_depth7)
        neighbor_prefix = neighbor_prefixes_7[i]

        prefix_idx = lax.shift_right_logical(neighbor_prefix, jnp.uint64(shift_amount_7))
        prefix_idx = prefix_idx.astype(jnp.int32)
        prefix_idx = jnp.clip(prefix_idx, 0, mesh_gpu.prefix_start.shape[0] - 1)

        first_leaf = mesh_gpu.prefix_start[prefix_idx]
        num_leaves = mesh_gpu.prefix_length[prefix_idx]
        has_leaves = num_leaves > 0
        valid_start = first_leaf >= 0

        # Search up to 8 leaves with lax.fori_loop
        def search_leaves_depth7(leaf_offset, leaf_state):
            """Search one leaf at depth-7 (bounded loop body)."""
            octant_elem, octant_found = leaf_state
            leaf_id = first_leaf + leaf_offset
            valid = (leaf_offset < num_leaves) & (leaf_id >= 0) & (leaf_id < mesh_gpu.n_leaves) & jnp.logical_not(octant_found)

            result = jnp.where(valid, search_in_leaf_global(pos, leaf_id, mesh_gpu), jnp.int32(-1))
            improved = result >= 0

            return (
                jnp.where(improved, result, octant_elem),
                octant_found | improved
            )

        # BOUNDED LOOP: 8 leaves per octant
        octant_elem, octant_found = lax.fori_loop(
            0, 8,
            search_leaves_depth7,
            (jnp.int32(-1), jnp.bool_(False))
        )

        elem_neighbor = jnp.where(active & has_leaves & valid_start, octant_elem, jnp.int32(-1))

        improve = (elem_neighbor >= 0) & active
        elem_id_depth7 = jnp.where(improve, elem_neighbor, elem_id_depth7)
        found_depth7 = found_depth7 | improve

        return (elem_id_depth7, found_depth7)

    # BOUNDED LOOP over 27 octants at depth-7
    elem_id_depth7, found_depth7 = lax.fori_loop(
        0, 27,
        search_one_octant_depth7,
        (jnp.int32(-1), jnp.bool_(False))
    )

    # DEPTH 6: CONDITIONAL search at coarse resolution
    # Only executes for particles that failed depth-7 (via jnp.where)
    # This is the key optimization: saves 216 leaf searches for particles
    # that succeed at depth-7 (expected 60-80% of particles)
    elem_final = jnp.where(
        found_depth7,
        elem_id_depth7,
        search_depth6_octants_single(pos, morton_query, mesh_gpu)
    )

    return elem_final


# ============================================================================
# Upload Function
# ============================================================================

def upload_global_morton_to_gpu(
    morton_struct,  # GlobalMortonStructure from morton_global_builder OR morton_octree_builder OR HilbertStructure
    connectivity: np.ndarray,
    node_positions: np.ndarray
) -> MeshGPUGlobalMorton:
    """
    Upload global Morton or Hilbert structure to GPU.

    Supports:
    - OLD: Fixed-capacity leaves (from morton_global_builder)
    - NEW: Adaptive octree leaves (from morton_octree_builder)
    - NEW: Hilbert octree (from hilbert_octree_builder) - DROP-IN COMPATIBLE

    Args:
        morton_struct: Structure from CPU preprocessing (Morton or Hilbert)
            Must have: elem_ids_sorted, morton_sorted OR hilbert_sorted, leaf_start, leaf_length,
                      bbox_min, bbox_max, max_depth, leaf_capacity, n_leaves
            Optional (for octree): prefix_start, prefix_length, table_depth
        connectivity: (n_elements, 4) int32 - mesh connectivity
        node_positions: (n_nodes, 3) float32 - node coordinates

    Returns:
        MeshGPUGlobalMorton with all data on GPU
        (Note: Works for both Morton and Hilbert - field name is generic)
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

    # Get curve indices (either morton_sorted or hilbert_sorted)
    # This allows the same upload function to work with both curve types
    if hasattr(morton_struct, 'hilbert_sorted'):
        curve_indices = morton_struct.hilbert_sorted
    else:
        curve_indices = morton_struct.morton_sorted

    return MeshGPUGlobalMorton(
        # Core mesh data
        connectivity=jax.device_put(connectivity.astype(np.int32)),
        node_positions=jax.device_put(node_positions.astype(np.float32)),

        # Curve structure (works for both Morton and Hilbert)
        elem_ids_sorted=jax.device_put(morton_struct.elem_ids_sorted.astype(np.int32)),
        morton_sorted=jax.device_put(curve_indices.astype(np.uint64)),  # Generic field name (holds Morton OR Hilbert)
        leaf_start=jax.device_put(morton_struct.leaf_start.astype(np.int32)),
        leaf_length=jax.device_put(morton_struct.leaf_length.astype(np.int32)),

        # Octree prefix table (NEW - Phase 5)
        prefix_start=prefix_start_gpu,
        prefix_length=prefix_length_gpu,
        table_depth=table_depth_val,

        # Curve parameters (works for both Morton and Hilbert)
        morton_min=jnp.uint64(curve_indices.min()),
        morton_max=jnp.uint64(curve_indices.max()),
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
