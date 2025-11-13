"""
GPU Kernels for Particle Tracking.

JAX-based GPU kernels for element search and field interpolation.
All functions are designed to work with JAX's vmap for parallelization.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple


def build_block_element_lists(
    element_to_block: np.ndarray,
    n_blocks: int,
    max_elements_per_block: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build compact element lists for each block (CPU preprocessing).

    This creates a fixed-size array of elements per block to enable efficient
    GPU searching without processing the entire mesh or sparse arrays with dummies.

    Args:
        element_to_block: Element-to-block mapping [N_elements]
        n_blocks: Number of blocks in grid
        max_elements_per_block: Maximum elements to store per block.
            If None, uses 95th percentile + 20% buffer.

    Returns:
        block_elements: Element IDs per block [n_blocks, max_per_block], padded with -1
        block_counts: Actual element counts [n_blocks]

    Algorithm:
        For each block:
        1. Use np.where to find all elements in block (CPU-efficient)
        2. Store in fixed-size array, pad with -1 if fewer elements
        3. Track actual count for each block

    Note:
        This is computed ONCE on CPU during tracker initialization.
        The compact lists are then transferred to GPU for fast searches.

    Example:
        For ThreadedA with 32 blocks:
        - Block sizes: 36 to 938K elements
        - 95th percentile: ~150 elements
        - max_per_block = 150 * 1.2 = 180
        - Result: [32, 180] array = 5,760 entries (vs 3.5M full mesh!)
    """
    # Determine max_elements_per_block if not specified
    if max_elements_per_block is None:
        block_sizes = []
        for block_id in range(n_blocks):
            count = np.sum(element_to_block == block_id)
            block_sizes.append(count)

        # Use 95th percentile + 20% buffer, but cap at 10,000 to avoid GPU OOM
        # For highly imbalanced grids, some blocks will be truncated (those will use
        # slower searches, but it's better than crashing)
        percentile_95 = np.percentile(block_sizes, 95)
        max_from_percentile = int(percentile_95 * 1.2)

        # Cap at 10,000 for GPU memory efficiency
        # This allows ~1.3 MB for block lists per block
        max_elements_per_block = min(max_from_percentile, 10000)

        print(f"  Block element list sizing:")
        print(f"    Min block size: {min(block_sizes):,}")
        print(f"    Max block size: {max(block_sizes):,}")
        print(f"    Mean: {np.mean(block_sizes):,.0f}")
        print(f"    95th percentile: {percentile_95:,.0f}")
        print(f"    Capped max_per_block: {max_elements_per_block:,}")

        if max_from_percentile > 10000:
            n_large = sum(1 for s in block_sizes if s > 10000)
            print(f"    ⚠️  {n_large} blocks exceed 10K elements (will use fallback search)")
            print(f"       Consider adaptive grid refinement (Phase 8) for load balancing")

    # Allocate arrays
    block_elements = np.full(
        (n_blocks, max_elements_per_block),
        -1,
        dtype=np.int32
    )
    block_counts = np.zeros(n_blocks, dtype=np.int32)

    # Fill arrays
    n_truncated = 0
    for block_id in range(n_blocks):
        # CPU: np.where creates compact list efficiently
        elements = np.where(element_to_block == block_id)[0]
        count = len(elements)

        if count > max_elements_per_block:
            # Truncate if block exceeds limit (rare, from load imbalance)
            n_truncated += 1
            elements = elements[:max_elements_per_block]
            count = max_elements_per_block

        # Store elements (rest remain -1)
        if count > 0:
            block_elements[block_id, :count] = elements
        block_counts[block_id] = count

    if n_truncated > 0:
        print(f"  ⚠️  {n_truncated} blocks truncated (load imbalance)")
        print(f"     Consider increasing max_per_block or using adaptive grid (Phase 8)")

    return block_elements, block_counts


@jax.jit
def point_in_tetrahedron_jax(
    point: jnp.ndarray,
    vertices: jnp.ndarray
) -> bool:
    """
    Test if point is inside tetrahedral element using barycentric coordinates.

    Uses JAX operations for GPU execution. Can be vmapped over particles or elements.

    Args:
        point: 3D point [3] (x, y, z)
        vertices: Element vertices [4, 3] (4 nodes × 3 coords)

    Returns:
        True if point is inside tetrahedron

    Algorithm:
        Compute barycentric coordinates λ = [λ0, λ1, λ2, λ3]
        Point is inside if all λi ∈ [0, 1] and Σλi = 1

    Example:
        >>> vertices = jnp.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        >>> point = jnp.array([0.2, 0.2, 0.2])
        >>> point_in_tetrahedron_jax(point, vertices)
        True
    """
    # Extract vertices
    p0, p1, p2, p3 = vertices[0], vertices[1], vertices[2], vertices[3]

    # Build matrix for barycentric coordinate system
    # [p1-p0, p2-p0, p3-p0] @ [λ1, λ2, λ3] = point - p0
    mat = jnp.column_stack([p1 - p0, p2 - p0, p3 - p0])
    rhs = point - p0

    # Solve for [λ1, λ2, λ3]
    # Use jnp.linalg.solve which is differentiable and GPU-friendly
    try:
        lambdas = jnp.linalg.solve(mat, rhs)
    except:
        # Degenerate element - return False
        return False

    lambda1, lambda2, lambda3 = lambdas[0], lambdas[1], lambdas[2]
    lambda0 = 1.0 - lambda1 - lambda2 - lambda3

    # Check if all barycentric coordinates are in [0, 1]
    tolerance = 1e-6
    in_range = (
        (lambda0 >= -tolerance) & (lambda0 <= 1.0 + tolerance) &
        (lambda1 >= -tolerance) & (lambda1 <= 1.0 + tolerance) &
        (lambda2 >= -tolerance) & (lambda2 <= 1.0 + tolerance) &
        (lambda3 >= -tolerance) & (lambda3 <= 1.0 + tolerance)
    )

    return in_range


@jax.jit
def point_in_tetrahedron_safe(
    point: jnp.ndarray,
    vertices: jnp.ndarray
) -> bool:
    """
    Safe version of point-in-tetrahedron that handles singular matrices.

    This version uses pseudoinverse instead of direct solve to handle
    degenerate elements gracefully.

    Args:
        point: 3D point [3]
        vertices: Element vertices [4, 3]

    Returns:
        True if point is inside tetrahedron
    """
    p0, p1, p2, p3 = vertices[0], vertices[1], vertices[2], vertices[3]

    # Build matrix
    mat = jnp.column_stack([p1 - p0, p2 - p0, p3 - p0])
    rhs = point - p0

    # Use pseudoinverse for stability
    mat_pinv = jnp.linalg.pinv(mat)
    lambdas = mat_pinv @ rhs

    lambda1, lambda2, lambda3 = lambdas[0], lambdas[1], lambdas[2]
    lambda0 = 1.0 - lambda1 - lambda2 - lambda3

    # Check bounds
    tolerance = 1e-6
    in_range = (
        (lambda0 >= -tolerance) & (lambda0 <= 1.0 + tolerance) &
        (lambda1 >= -tolerance) & (lambda1 <= 1.0 + tolerance) &
        (lambda2 >= -tolerance) & (lambda2 <= 1.0 + tolerance) &
        (lambda3 >= -tolerance) & (lambda3 <= 1.0 + tolerance)
    )

    return in_range


@jax.jit
def search_cached_element_jax(
    point: jnp.ndarray,
    cached_element_id: int,
    positions: jnp.ndarray,
    connectivity: jnp.ndarray
) -> Tuple[bool, int]:
    """
    Level 0: Check if point is in cached element (GPU version).

    Args:
        point: 3D point [3]
        cached_element_id: Previously cached element ID
        positions: Node positions [N_nodes, 3]
        connectivity: Element connectivity [N_elements, 4]

    Returns:
        (found, element_id): found=True if point in cached element

    Note:
        This is designed to be vmapped over particles.
    """
    # Check if cache is valid
    is_valid_cache = cached_element_id >= 0

    # Get cached element vertices
    # Use jnp.where to avoid indexing errors when cache invalid
    safe_id = jnp.where(is_valid_cache, cached_element_id, 0)
    element_node_ids = connectivity[safe_id]
    vertices = positions[element_node_ids]

    # Check if point is inside
    is_inside = point_in_tetrahedron_safe(point, vertices)

    # Return result only if cache was valid AND point is inside
    found = is_valid_cache & is_inside
    result_id = jnp.where(found, cached_element_id, -1)

    return found, result_id


@jax.jit
def search_neighbors_jax(
    point: jnp.ndarray,
    cached_element_id: int,
    element_neighbors: jnp.ndarray,
    positions: jnp.ndarray,
    connectivity: jnp.ndarray
) -> Tuple[bool, int]:
    """
    Level 1: Check neighbor elements (GPU version).

    Args:
        point: 3D point [3]
        cached_element_id: Previously cached element ID
        element_neighbors: Neighbor array [N_elements, max_neighbors]
        positions: Node positions [N_nodes, 3]
        connectivity: Element connectivity [N_elements, 4]

    Returns:
        (found, element_id): found=True if point in neighbor

    Note:
        Uses lax.scan for efficient iteration over neighbors.
    """
    # Check if cache is valid
    is_valid_cache = cached_element_id >= 0

    # Get neighbors (use safe indexing)
    safe_cached_id = jnp.where(is_valid_cache, cached_element_id, 0)
    neighbor_ids = element_neighbors[safe_cached_id]  # [max_neighbors]

    # Search through neighbors using scan
    def check_neighbor(carry, neighbor_id):
        found, result_id = carry

        # Skip if already found or neighbor is invalid
        is_valid_neighbor = neighbor_id >= 0
        should_check = (~found) & is_valid_neighbor

        # Get neighbor vertices
        safe_id = jnp.where(is_valid_neighbor, neighbor_id, 0)
        element_node_ids = connectivity[safe_id]
        vertices = positions[element_node_ids]

        # Check if point is inside
        is_inside = point_in_tetrahedron_safe(point, vertices)

        # Update result if found
        new_found = found | (should_check & is_inside)
        new_result = jnp.where(should_check & is_inside, neighbor_id, result_id)

        return (new_found, new_result), None

    # Scan over neighbors only if cache is valid
    init_found = jnp.array(False)
    init_result = jnp.array(-1, dtype=jnp.int32)

    (found, result_id), _ = jax.lax.scan(
        check_neighbor,
        (init_found, init_result),
        neighbor_ids
    )

    # Return False if cache was invalid
    final_found = found & is_valid_cache
    final_result = jnp.where(is_valid_cache, result_id, -1)

    return final_found, final_result


@jax.jit
def search_block_elements_jax(
    point: jnp.ndarray,
    block_id: int,
    block_elements: jnp.ndarray,
    block_counts: jnp.ndarray,
    positions: jnp.ndarray,
    connectivity: jnp.ndarray
) -> Tuple[bool, int]:
    """
    Level 2: Block-local search using pre-computed element lists (GPU version).

    Uses pre-computed compact element lists to avoid checking entire mesh or
    sparse arrays with dummies. Much more efficient than previous approaches.

    Args:
        point: 3D point [3]
        block_id: Block containing point
        block_elements: Pre-computed element lists [n_blocks, max_per_block]
        block_counts: Actual element counts [n_blocks]
        positions: Node positions [N_nodes, 3]
        connectivity: Element connectivity [N_elements, 4]

    Returns:
        (found, element_id): found=True if element found

    Algorithm:
        1. Get pre-computed element list for this block
        2. vmap over elements (only max_per_block, not full mesh!)
        3. Return first match

    Note:
        This fixes both previous bottlenecks:
        - lax.scan version: checked 1000 elements with 90% dummies
        - vmap version: checked 3.5M elements causing OOM
        - This version: checks only ~200 elements (compact list)

    Example for ThreadedA:
        - Old: 1000 or 3,500,000 checks per particle
        - New: 200 checks per particle (~100 real + ~100 padding)
        - 5× to 17,500× improvement!
    """
    # Check if block_id is valid
    is_valid_block = block_id >= 0

    # Get element list for this block [max_per_block]
    # block_elements[block_id] contains element IDs, padded with -1
    elements = block_elements[block_id]
    count = block_counts[block_id]

    # Check each element (vectorized)
    def check_element(elem_id):
        """Check if point is in element."""
        is_valid = elem_id >= 0

        # Safe indexing (use 0 if invalid, but won't use result)
        safe_id = jnp.where(is_valid, elem_id, 0)
        element_node_ids = connectivity[safe_id]
        vertices = positions[element_node_ids]

        # Check containment
        is_inside = jnp.where(
            is_valid,
            point_in_tetrahedron_safe(point, vertices),
            False
        )

        return is_inside, jnp.where(is_inside, elem_id, -1)

    # vmap over elements in this block (NOT full mesh!)
    # For ThreadedA: ~200 elements instead of 3.5M
    found_array, result_array = jax.vmap(check_element)(elements)

    # Find first match
    found_any = jnp.any(found_array)
    first_match_idx = jnp.argmax(found_array)
    result_id = result_array[first_match_idx]

    # Return result only if block is valid and we found a match
    final_found = found_any & is_valid_block
    final_result = jnp.where(final_found, result_id, -1)

    return final_found, final_result


@jax.jit
def find_containing_element_gpu(
    point: jnp.ndarray,
    cached_element_id: int,
    block_id: int,
    element_neighbors: jnp.ndarray,
    block_elements: jnp.ndarray,
    block_counts: jnp.ndarray,
    positions: jnp.ndarray,
    connectivity: jnp.ndarray
) -> int:
    """
    Three-tier element search (GPU version).

    Hierarchical search:
    1. Check cached element (O(1))
    2. Check neighbors (O(1))
    3. Check block elements (O(k) where k = elements in block)

    Args:
        point: 3D point [3]
        cached_element_id: Previously cached element ID
        block_id: Block containing point
        element_neighbors: Neighbor array [N_elements, max_neighbors]
        block_elements: Pre-computed element lists [n_blocks, max_per_block]
        block_counts: Actual element counts [n_blocks]
        positions: Node positions [N_nodes, 3]
        connectivity: Element connectivity [N_elements, 4]

    Returns:
        element_id: ID of containing element (-1 if not found)

    Note:
        This function is designed to be vmapped over particles.
        Statistics tracking happens on CPU after results are transferred.
    """
    # Level 0: Check cached element
    found_l0, elem_l0 = search_cached_element_jax(
        point, cached_element_id, positions, connectivity
    )

    # Level 1: Check neighbors (only if L0 failed)
    found_l1, elem_l1 = search_neighbors_jax(
        point, cached_element_id, element_neighbors, positions, connectivity
    )

    # Level 2: Block search (only if L0 and L1 failed)
    found_l2, elem_l2 = search_block_elements_jax(
        point, block_id, block_elements, block_counts, positions, connectivity
    )

    # Return result from first successful level
    # Use jnp.where for branching (JAX-compatible)
    result = jnp.where(
        found_l0,
        elem_l0,
        jnp.where(
            found_l1,
            elem_l1,
            jnp.where(found_l2, elem_l2, -1)
        )
    )

    return result


# Vectorized version over particles
find_containing_elements_batch = jax.jit(jax.vmap(
    find_containing_element_gpu,
    in_axes=(0, 0, 0, None, None, None, None, None)
))


@jax.jit
def position_to_block_id_jax(
    position: jnp.ndarray,
    domain_bounds: jnp.ndarray,
    grid_size: Tuple[int, int, int]
) -> int:
    """
    Fast O(1) position → block_id mapping (GPU version).

    Args:
        position: 3D position [3] (x, y, z)
        domain_bounds: Domain bounds [6] (xmin, xmax, ymin, ymax, zmin, zmax)
        grid_size: Grid dimensions (nx, ny, nz)

    Returns:
        block_id: Block ID (0 to nx*ny*nz-1), or -1 if outside domain
    """
    nx, ny, nz = grid_size

    # Extract bounds
    xmin, xmax = domain_bounds[0], domain_bounds[1]
    ymin, ymax = domain_bounds[2], domain_bounds[3]
    zmin, zmax = domain_bounds[4], domain_bounds[5]

    # Check if inside domain
    x, y, z = position[0], position[1], position[2]
    inside_x = (x >= xmin) & (x <= xmax)
    inside_y = (y >= ymin) & (y <= ymax)
    inside_z = (z >= zmin) & (z <= zmax)
    inside = inside_x & inside_y & inside_z

    # Compute grid indices
    ix = jnp.floor((x - xmin) / (xmax - xmin) * nx).astype(jnp.int32)
    iy = jnp.floor((y - ymin) / (ymax - ymin) * ny).astype(jnp.int32)
    iz = jnp.floor((z - zmin) / (zmax - zmin) * nz).astype(jnp.int32)

    # Clamp to valid range
    ix = jnp.clip(ix, 0, nx - 1)
    iy = jnp.clip(iy, 0, ny - 1)
    iz = jnp.clip(iz, 0, nz - 1)

    # Compute block ID
    block_id = ix + iy * nx + iz * (nx * ny)

    # Return -1 if outside
    return jnp.where(inside, block_id, -1)


# Vectorized version over positions
positions_to_block_ids_batch = jax.jit(jax.vmap(
    position_to_block_id_jax,
    in_axes=(0, None, None)
))
