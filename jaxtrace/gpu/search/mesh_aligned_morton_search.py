#!/usr/bin/env python3
"""
Mesh-Aligned Morton Search - Phase 5: GPU Search Kernel

JAX-compatible Morton radius search over mesh-aligned cell centers.

This hybrid approach combines:
  1. Intrinsic mesh octree structure (5.9 elements/cell)
  2. Proven Morton radius search algorithm (93-98% retention)

Key Differences from Direct Mesh-Aligned Octree:
  - Direct octree: Single-cell lookup (74.6% retention, elements span cells)
  - This: Morton radius search over cells (expected ~98% retention)

Key Differences from Original Morton:
  - Original: Morton codes from element centroids
  - This: Morton codes from cell centers (fewer Morton leaves, better locality)

Architecture:
  Position → Morton code → Binary search → Cell → Search radius cells → Elements

No nested control flow - single vmap over particles.
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
class MeshAlignedMortonGPU:
    """GPU-resident mesh-aligned Morton structure for L2 search."""

    # Core mesh data (already on GPU)
    connectivity: jax.Array          # (n_elements, 4) int32
    node_positions: jax.Array        # (n_nodes, 3) float32

    # Morton structure (cell-based)
    cell_morton_codes: jax.Array     # (n_cells,) uint64 - sorted Morton codes
    cell_levels: jax.Array           # (n_cells,) uint8 - refinement level per cell
    cell_grid_indices: jax.Array     # (n_cells, 3) int32 - grid indices (i, j, k)
    cell_sizes: jax.Array            # (n_cells, 3) float32 - cell sizes

    # Cell -> elements mapping (CSR format)
    cell_to_elements_offsets: jax.Array  # (n_cells+1,) int32
    cell_to_elements_data: jax.Array     # (total_elements,) int32

    # Morton parameters
    morton_min: jnp.uint64           # Minimum Morton code
    morton_max: jnp.uint64           # Maximum Morton code
    bbox_min: jax.Array              # (3,) float32 - global bbox
    bbox_max: jax.Array              # (3,) float32

    # Configuration
    n_cells: jnp.int32               # Number of cells (= "leaves" in Morton terminology)
    max_depth: jnp.int32             # Morton encoding depth (21)


# ============================================================================
# Upload to GPU
# ============================================================================

def upload_mesh_aligned_morton_to_gpu(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    structure,  # MeshAlignedMortonStructure
    verbose: bool = True
) -> MeshAlignedMortonGPU:
    """
    Upload mesh-aligned Morton structure to GPU.

    Args:
        node_positions: (n_nodes, 3) float64
        connectivity: (n_elements, 4) int32
        structure: MeshAlignedMortonStructure from builder
        verbose: Print upload status

    Returns:
        MeshAlignedMortonGPU ready for search
    """

    if verbose:
        print(f"\nUploading Mesh-Aligned Morton Structure to GPU...")

    mesh_gpu = MeshAlignedMortonGPU(
        connectivity=jnp.array(connectivity, dtype=jnp.int32),
        node_positions=jnp.array(node_positions, dtype=jnp.float32),
        cell_morton_codes=jnp.array(structure.cell_morton_codes, dtype=jnp.uint64),
        cell_levels=jnp.array(structure.cell_levels, dtype=jnp.uint8),
        cell_grid_indices=jnp.array(structure.cell_grid_indices, dtype=jnp.int32),
        cell_sizes=jnp.array(structure.cell_sizes, dtype=jnp.float32),
        cell_to_elements_offsets=jnp.array(structure.cell_to_elements_offsets, dtype=jnp.int32),
        cell_to_elements_data=jnp.array(structure.cell_to_elements_data, dtype=jnp.int32),
        morton_min=jnp.uint64(structure.morton_min),
        morton_max=jnp.uint64(structure.morton_max),
        bbox_min=jnp.array(structure.bbox_min, dtype=jnp.float32),
        bbox_max=jnp.array(structure.bbox_max, dtype=jnp.float32),
        n_cells=jnp.int32(structure.n_cells),
        max_depth=jnp.int32(structure.max_depth),
    )

    if verbose:
        print(f"  ✅ Upload complete!")
        print(f"    Cells: {structure.n_cells:,}")
        print(f"    Elements per cell: {structure.elements_per_cell_mean:.1f} (mean), {structure.elements_per_cell_max} (max)")

    return mesh_gpu


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
# Position to Cell Mapping
# ============================================================================

def morton_binary_search_cell(
    morton_code: jnp.uint64,
    cell_morton_codes: jax.Array
) -> jnp.int32:
    """
    Binary search to find cell containing given Morton code.

    Since cells are sorted by Morton code, we search for the index
    where morton_code would be inserted.

    Uses lax.while_loop for JAX compatibility.

    Args:
        morton_code: uint64 - query Morton code
        cell_morton_codes: (n_cells,) uint64 - sorted Morton codes

    Returns:
        cell_id: int32 - cell containing elements near this Morton code
    """
    n_cells = cell_morton_codes.shape[0]

    # Binary search state: (left, right)
    def cond_fun(state):
        left, right = state
        return left < right

    def body_fun(state):
        left, right = state
        mid = (left + right) // 2
        mid_morton = cell_morton_codes[mid]

        # If query < mid, search left half
        # If query >= mid, search right half
        new_left = jnp.where(morton_code < mid_morton, left, mid + 1)
        new_right = jnp.where(morton_code < mid_morton, mid, right)

        return (new_left, new_right)

    # Initial state
    init_state = (jnp.int32(0), jnp.int32(n_cells))

    # Run binary search
    final_left, final_right = lax.while_loop(cond_fun, body_fun, init_state)

    # final_left is the insertion point
    # Clamp to valid cell index range
    cell_id = jnp.clip(final_left, 0, n_cells - 1)

    return cell_id


def position_to_cell_id(
    pos: jax.Array,
    mesh_gpu: MeshAlignedMortonGPU
) -> jnp.int32:
    """
    Map position to cell ID using binary search on Morton codes.

    Args:
        pos: (3,) float32 - query position
        mesh_gpu: GPU-resident Morton structure

    Returns:
        cell_id: int32 in range [0, n_cells - 1]
    """
    # Compute Morton code for position
    m = morton_encode_position_jax(
        pos,
        mesh_gpu.bbox_min,
        mesh_gpu.bbox_max,
        mesh_gpu.max_depth
    )

    # Binary search in sorted Morton array
    cell_id = morton_binary_search_cell(m, mesh_gpu.cell_morton_codes)

    return cell_id


# ============================================================================
# Bounded Cell Search
# ============================================================================

def search_in_cell(
    pos: jax.Array,
    cell_id: jnp.int32,
    mesh_gpu: MeshAlignedMortonGPU,
    max_tests: jnp.int32 = 256
) -> jnp.int32:
    """
    Search for element containing pos within a single cell.

    Uses bounded lax.fori_loop to prevent graph explosion.

    Args:
        pos: (3,) float32 - query position
        cell_id: int32 - which cell to search
        mesh_gpu: GPU-resident Morton structure
        max_tests: Maximum elements to test (default 256)

    Returns:
        elem_id: int32 - found element ID, or -1 if not found
    """
    # Get cell parameters
    start = mesh_gpu.cell_to_elements_offsets[cell_id]
    end = mesh_gpu.cell_to_elements_offsets[cell_id + 1]
    length = end - start

    def check_element(j, found_elem):
        """Check one element in cell (bounded loop body)."""
        # Active only if: (1) not yet found, (2) j < actual length
        active = (found_elem == -1) & (j < length)

        # Get global element ID
        idx = start + j
        elem_id = jnp.where(active, mesh_gpu.cell_to_elements_data[idx], jnp.int32(0))

        # Test point-in-tet (masked by active)
        inside = jnp.where(
            active,
            point_in_tet_dispatcher(
                pos, elem_id, mesh_gpu.connectivity, mesh_gpu.node_positions,
                method=config.POINT_IN_TET_METHOD
            ),
            False
        )

        # Update found_elem if inside and active
        return jnp.where(inside & active, elem_id, found_elem)

    # Bounded loop to prevent graph explosion
    n_to_test = jnp.minimum(length, max_tests)
    found_elem = lax.fori_loop(0, n_to_test, check_element, jnp.int32(-1))

    return found_elem


# ============================================================================
# L2 Search (Single Particle) - Morton Radius
# ============================================================================

def search_L2_mesh_aligned_morton_single(
    pos: jax.Array,
    mesh_gpu: MeshAlignedMortonGPU,
    search_radius: jnp.int32 = jnp.int32(2),
    max_tests_per_cell: jnp.int32 = 256
) -> jnp.int32:
    """
    L2 search using mesh-aligned Morton structure with radius search.

    This is the HYBRID APPROACH combining:
      1. Intrinsic mesh octree structure (cell-based)
      2. Proven Morton radius search algorithm

    Searches the predicted cell and its neighbors along the Morton curve.
    Radius search accounts for elements spanning multiple cells.

    **IMPORTANT: radius=N searches BOTH directions**:
      - Searches center cell (1 cell)
      - Searches -N, -N+1, ..., -1 cells BACKWARD (N cells)
      - Searches +1, +2, ..., +N cells FORWARD (N cells)
      - **Total: 2N + 1 cells** (symmetric band around center)

    Example: radius=2 searches 5 cells total:
      cells[-2], cells[-1], cells[0], cells[+1], cells[+2]

    With 5.9 elements/cell (mean), radius=2 searches ~30 elements (5 × 5.9).
    Compare to original Morton radius=2: ~536 elements (5 leaves × 107 elem/leaf).

    IMPORTANT: This function is NOT @jax.jit decorated.
    It will be vmapped externally in the RK4 integrator.

    Args:
        pos: (3,) float32 - query position
        mesh_gpu: GPU-resident Morton structure
        search_radius: int32 - search ±radius cells (default 2)
        max_tests_per_cell: int32 - max elements to test per cell (default 256)

    Returns:
        elem_id: int32 - found element, or -1 if not found
    """
    # Map position to cell
    center_cell_id = position_to_cell_id(pos, mesh_gpu)

    # Search center cell first
    elem_id = search_in_cell(pos, center_cell_id, mesh_gpu, max_tests_per_cell)

    # If found, return immediately
    found = elem_id >= 0

    def search_one_neighbor(i, state):
        elem_id, found = state

        # Map i ∈ [0, 2*search_radius-1] to offset ∈ [-search_radius, -1] ∪ [+1, +search_radius]
        offset = jnp.where(
            i < search_radius,
            -(search_radius - i),  # Negative offsets
            (i - search_radius) + 1  # Positive offsets
        )

        active = ~found
        neighbor_cell_id = jnp.clip(center_cell_id + offset, 0, mesh_gpu.n_cells - 1)

        elem_neighbor = jnp.where(
            active,
            search_in_cell(pos, neighbor_cell_id, mesh_gpu, max_tests_per_cell),
            jnp.int32(-1)
        )

        improve = (elem_neighbor >= 0) & active
        new_elem_id = jnp.where(improve, elem_neighbor, elem_id)
        new_found = found | improve

        return (new_elem_id, new_found)

    # Search neighbors: -search_radius, ..., -1, +1, ..., +search_radius
    # Cap at 512 to prevent absurd values
    safe_radius = jnp.minimum(search_radius, 512)
    elem_id, found = lax.fori_loop(0, 2 * safe_radius, search_one_neighbor, (elem_id, found))

    return elem_id


# ============================================================================
# Grid-Based Neighbor Search (TRUE SPATIAL NEIGHBORS)
# ============================================================================

def search_L2_mesh_aligned_grid_neighbors_single(
    pos: jax.Array,
    mesh_gpu: MeshAlignedMortonGPU,
    grid_radius: jnp.int32 = jnp.int32(1),
    max_tests_per_cell: jnp.int32 = 256
) -> jnp.int32:
    """
    L2 search using grid-based 3D neighbor search (TRUE spatial neighbors).

    This is the CORRECT hybrid approach:
      1. Find center cell from position (grid indices)
      2. Search (2*grid_radius + 1)^3 neighbors in grid space
      3. Use cell size to determine which cells to check

    **Simplified approach**: Instead of searching specific (i,j,k) neighbors,
    we search a wider window in the Morton-sorted array (±window) and filter
    by Euclidean distance to find spatial neighbors.

    Example: grid_radius=1 → search ±100 cells in array, keep distance < 1.5*cell_size
             grid_radius=2 → search ±200 cells in array, keep distance < 2.5*cell_size

    This leverages Morton locality while avoiding expensive grid lookups.

    Args:
        pos: (3,) float32 - query position
        mesh_gpu: GPU-resident Morton structure
        grid_radius: Grid search radius (1 = ~27 cells, 2 = ~125 cells)
        max_tests_per_cell: Maximum tests per cell

    Returns:
        elem_id: int32 - found element, or -1 if not found
    """
    # Step 1: Compute position's grid indices to determine expected cell
    # Use first cell's size as reference (most cells have similar size)
    ref_cell_size = mesh_gpu.cell_sizes[0]

    pos_i = jnp.floor(pos[0] / ref_cell_size[0]).astype(jnp.int32)
    pos_j = jnp.floor(pos[1] / ref_cell_size[1]).astype(jnp.int32)
    pos_k = jnp.floor(pos[2] / ref_cell_size[2]).astype(jnp.int32)

    # Step 2: Search nearby cells using distance-based filtering
    # Morton codes preserve spatial locality, so we search a window around the binary search result
    center_cell_id = position_to_cell_id(pos, mesh_gpu)

    # Compute center position of the found cell
    center_i = mesh_gpu.cell_grid_indices[center_cell_id, 0]
    center_j = mesh_gpu.cell_grid_indices[center_cell_id, 1]
    center_k = mesh_gpu.cell_grid_indices[center_cell_id, 2]
    center_cell_size = mesh_gpu.cell_sizes[center_cell_id]

    # Search center cell first
    elem_id = search_in_cell(pos, center_cell_id, mesh_gpu, max_tests_per_cell)
    found = elem_id >= 0

    # Step 3: Search nearby cells using distance-based filtering
    # Search window: ±(50 * grid_radius) cells in sorted array
    window_size = 50 * grid_radius
    search_start = jnp.clip(center_cell_id - window_size, 0, mesh_gpu.n_cells - 1)
    search_end = jnp.clip(center_cell_id + window_size + 1, 0, mesh_gpu.n_cells)
    n_cells_to_check = search_end - search_start

    def check_nearby_cell(offset, state):
        """Check one cell in the window."""
        elem_id, found = state

        # Skip if already found
        active = ~found

        cell_idx = search_start + offset

        # Skip center cell (already checked)
        is_center = (cell_idx == center_cell_id)

        # Get this cell's grid indices
        cell_i = mesh_gpu.cell_grid_indices[cell_idx, 0]
        cell_j = mesh_gpu.cell_grid_indices[cell_idx, 1]
        cell_k = mesh_gpu.cell_grid_indices[cell_idx, 2]

        # Check if cell is within grid radius
        di = jnp.abs(cell_i - center_i)
        dj = jnp.abs(cell_j - center_j)
        dk = jnp.abs(cell_k - center_k)

        within_grid = (di <= grid_radius) & (dj <= grid_radius) & (dk <= grid_radius)

        # Search this cell if it's within grid radius, not center, and we're active
        should_search = within_grid & (~is_center) & active

        elem_neighbor = jnp.where(
            should_search,
            search_in_cell(pos, cell_idx, mesh_gpu, max_tests_per_cell),
            jnp.int32(-1)
        )

        improve = (elem_neighbor >= 0) & active
        new_elem_id = jnp.where(improve, elem_neighbor, elem_id)
        new_found = found | improve

        return (new_elem_id, new_found)

    # Search all cells in window
    safe_n_cells = jnp.minimum(n_cells_to_check, 500)  # Cap at 500 cells to prevent explosion
    elem_id, found = lax.fori_loop(0, safe_n_cells, check_nearby_cell, (elem_id, found))

    return elem_id


def search_L2_mesh_aligned_morton_incremental_single(
    pos: jax.Array,
    mesh_gpu: MeshAlignedMortonGPU,
    radii: tuple = (2, 5, 10),
    max_tests_per_cell: jnp.int32 = 256
) -> jnp.int32:
    """
    L2 search with incremental radius expansion (conditional cascade).

    Searches with increasing radius values, using conditional execution.
    Each tier searches a SYMMETRIC BAND around the center cell.

    **Default configuration** (radii=(2, 5, 10)):
      Tier 1: radius=2  → 5 cells  (2×2+1)
      Tier 2: radius=5  → 11 cells (2×5+1) - only if tier 1 fails
      Tier 3: radius=10 → 21 cells (2×10+1) - only if tier 2 fails

    With 5.9 elements/cell:
      Tier 1: ~30 tests (5 × 5.9)
      Tier 2: ~65 tests (11 × 5.9)
      Tier 3: ~124 tests (21 × 5.9)

    Args:
        pos: (3,) float32 - query position
        mesh_gpu: GPU-resident Morton structure
        radii: Tuple of radii to try (e.g., (2, 5, 10))
        max_tests_per_cell: Maximum tests per cell

    Returns:
        elem_id: int32 - found element, or -1 if not found
    """
    # Try tier 1
    elem_id_1 = search_L2_mesh_aligned_morton_single(
        pos, mesh_gpu, jnp.int32(radii[0]), max_tests_per_cell
    )
    found_1 = elem_id_1 >= 0

    # Try tier 2 (only if tier 1 failed)
    if len(radii) > 1:
        elem_id_2 = jnp.where(
            found_1,
            elem_id_1,
            search_L2_mesh_aligned_morton_single(
                pos, mesh_gpu, jnp.int32(radii[1]), max_tests_per_cell
            )
        )
        found_2 = elem_id_2 >= 0
    else:
        elem_id_2 = elem_id_1
        found_2 = found_1

    # Try tier 3 (only if tier 2 failed)
    if len(radii) > 2:
        elem_id_3 = jnp.where(
            found_2,
            elem_id_2,
            search_L2_mesh_aligned_morton_single(
                pos, mesh_gpu, jnp.int32(radii[2]), max_tests_per_cell
            )
        )
    else:
        elem_id_3 = elem_id_2

    return elem_id_3


# ============================================================================
# Batch Search (for testing/benchmarking)
# ============================================================================

def search_L2_mesh_aligned_grid_neighbors_batch(
    positions: jax.Array,
    mesh_gpu: MeshAlignedMortonGPU,
    grid_radius: jnp.int32 = jnp.int32(1),
    max_tests_per_cell: jnp.int32 = 256
) -> jax.Array:
    """
    Batch grid-based neighbor search using vmap.

    Args:
        positions: (n_particles, 3) float32 - query positions
        mesh_gpu: GPU-resident Morton structure
        grid_radius: Grid search radius (1 = 3×3×3)
        max_tests_per_cell: Maximum tests per cell

    Returns:
        elem_ids: (n_particles,) int32 - found elements (-1 if not found)
    """
    elem_ids = jax.vmap(
        lambda pos: search_L2_mesh_aligned_grid_neighbors_single(
            pos, mesh_gpu, grid_radius, max_tests_per_cell
        ),
        in_axes=0
    )(positions)

    return elem_ids


def search_L2_mesh_aligned_morton_batch(
    positions: jax.Array,
    mesh_gpu: MeshAlignedMortonGPU,
    search_radius: jnp.int32 = jnp.int32(2),
    max_tests_per_cell: jnp.int32 = 256
) -> jax.Array:
    """
    Batch search using vmap (for testing/benchmarking).

    Args:
        positions: (n_particles, 3) float32 - query positions
        mesh_gpu: GPU-resident Morton structure
        search_radius: Search radius
        max_tests_per_cell: Maximum tests per cell

    Returns:
        elem_ids: (n_particles,) int32 - found elements (-1 if not found)
    """
    elem_ids = jax.vmap(
        lambda pos: search_L2_mesh_aligned_morton_single(
            pos, mesh_gpu, search_radius, max_tests_per_cell
        ),
        in_axes=0
    )(positions)

    return elem_ids
