"""
Phase 3: Mesh-Aligned Octree GPU Structure

GPU-compatible octree structure for mesh-aligned cell-based point location.

Architecture:
- Position → Cell grid indices → Morton code → Element candidates
- Direct cell lookup (no tree traversal needed!)
- Multi-insert: elements stored in all overlapping cells
- JAX-compatible for GPU execution

Key Difference from Morton Global Search:
- Uses mesh's intrinsic octree cells (not arbitrary space partition)
- ~37 elements per cell vs ~536 in Morton blocks
- ~144× reduction in point-in-tet tests
"""

import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import jaxtrace.config as config


# ============================================================================
# GPU Data Structure
# ============================================================================

@dataclass
class MeshAlignedOctreeGPU:
    """
    GPU-resident mesh-aligned octree structure for fast point location.

    This structure uses the mesh's intrinsic octree cells (from Kuhn
    tetrahedral decomposition) for spatial indexing.

    Attributes
    ----------
    Core mesh data (GPU-resident):
        connectivity : jax.Array, shape (n_elements, 4), int32
            Element connectivity
        node_positions : jax.Array, shape (n_nodes, 3), float32
            Node coordinates

    Cell structure (CSR format for cell → elements mapping):
        cell_morton_codes : jax.Array, shape (n_cells,), uint64
            Morton codes for each cell (sorted)
        cell_levels : jax.Array, shape (n_cells,), uint8
            Octree level for each cell
        cell_sizes : jax.Array, shape (n_cells, 3), float32
            Cell dimensions (dx, dy, dz) for each cell
        cell_grid_indices : jax.Array, shape (n_cells, 3), int32
            Grid position (i, j, k) for each cell

        cell_to_elements_offsets : jax.Array, shape (n_cells + 1,), int32
            CSR offsets: cell i contains elements in range
            [offsets[i], offsets[i+1])
        cell_to_elements_data : jax.Array, shape (total_entries,), int32
            CSR data: flattened list of element IDs

    Morton encoding parameters:
        morton_offset : jnp.int32
            Offset for negative coordinates (typically 2^19)
        morton_max_coord : jnp.int32
            Maximum coordinate value (typically 2^20)
        max_depth : jnp.int32
            Morton encoding depth (typically 21 bits per dimension)

    Bounding box:
        bbox_min : jax.Array, shape (3,), float32
            Global mesh bounding box minimum
        bbox_max : jax.Array, shape (3,), float32
            Global mesh bounding box maximum

    Statistics:
        n_cells : jnp.int32
            Number of unique octree cells
        n_elements : jnp.int32
            Number of mesh elements
        mean_elements_per_cell : jnp.float32
            Average elements per cell (~37 for typical mesh)
    """
    # Core mesh data
    connectivity: jax.Array  # (n_elements, 4) int32
    node_positions: jax.Array  # (n_nodes, 3) float32

    # Cell structure (sorted by Morton code)
    cell_morton_codes: jax.Array  # (n_cells,) uint64
    cell_levels: jax.Array  # (n_cells,) uint8
    cell_sizes: jax.Array  # (n_cells, 3) float32
    cell_grid_indices: jax.Array  # (n_cells, 3) int32

    # Cell → elements mapping (CSR)
    cell_to_elements_offsets: jax.Array  # (n_cells + 1,) int32
    cell_to_elements_data: jax.Array  # (total_entries,) int32

    # Morton parameters
    morton_offset: jnp.int32
    morton_max_coord: jnp.int32
    max_depth: jnp.int32

    # Bounding box
    bbox_min: jax.Array  # (3,) float32
    bbox_max: jax.Array  # (3,) float32

    # Level-specific cell sizes for query lookup
    # Maps level -> representative cell size for that level
    # Shape: (max_level + 1, 3) float32, with unused levels set to 0
    level_cell_sizes: jax.Array  # (max_level + 1, 3) float32

    # Mesh's actual refinement-level range (extracted from the mesh at build
    # time, not hardcoded in the search kernel). Used by the search's
    # multi-level loop so MALMO adapts to any conforming tet mesh
    # regardless of scale — see paper Sec. 4 / Sec. 5.
    min_level: jnp.int32
    max_level: jnp.int32

    # Statistics
    n_cells: jnp.int32
    n_elements: jnp.int32
    mean_elements_per_cell: jnp.float32


# ============================================================================
# Morton Encoding (JAX-compatible, matches Phase 2)
# ============================================================================

def encode_morton_3d_jax(i: jnp.int32, j: jnp.int32, k: jnp.int32) -> jnp.uint64:
    """
    Encode grid indices (i, j, k) to Morton code on GPU.

    Uses bit interleaving to create space-filling curve index.
    Supports signed coordinates via offset (applied before calling).

    Args:
        i, j, k: Grid indices (after offset applied)

    Returns:
        Morton code as uint64
    """
    # Convert to uint64
    i = i.astype(jnp.uint64)
    j = j.astype(jnp.uint64)
    k = k.astype(jnp.uint64)

    # Interleave bits (unrolled for 21 bits)
    morton = jnp.uint64(0)

    # Manual unrolling for JAX compatibility
    for bit in range(21):
        morton |= ((i >> bit) & 1) << (3 * bit)
        morton |= ((j >> bit) & 1) << (3 * bit + 1)
        morton |= ((k >> bit) & 1) << (3 * bit + 2)

    return morton


def position_to_grid_indices(
    pos: jax.Array,
    cell_size: jax.Array,
) -> Tuple[jnp.int32, jnp.int32, jnp.int32]:
    """
    Convert world position to grid indices.

    Args:
        pos: (3,) float32 - position in world coordinates
        cell_size: (3,) float32 - cell dimensions (dx, dy, dz)

    Returns:
        (i, j, k): Grid indices (can be negative)
    """
    # Floor division to get grid cell
    i = jnp.floor(pos[0] / cell_size[0]).astype(jnp.int32)
    j = jnp.floor(pos[1] / cell_size[1]).astype(jnp.int32)
    k = jnp.floor(pos[2] / cell_size[2]).astype(jnp.int32)

    return i, j, k


def position_to_morton_code(
    pos: jax.Array,
    cell_size: jax.Array,
    morton_offset: jnp.int32,
    morton_max_coord: jnp.int32,
) -> jnp.uint64:
    """
    Convert world position to Morton code (single octree level).

    This assumes a single refinement level. For multi-level octrees,
    we need to determine the appropriate cell size first.

    Args:
        pos: (3,) float32 - position in world coordinates
        cell_size: (3,) float32 - cell dimensions at this level
        morton_offset: int32 - offset for negative coordinates
        morton_max_coord: int32 - maximum coordinate value

    Returns:
        Morton code as uint64
    """
    # Get grid indices
    i, j, k = position_to_grid_indices(pos, cell_size)

    # Apply offset and clamp
    i = jnp.clip(i + morton_offset, 0, morton_max_coord - 1)
    j = jnp.clip(j + morton_offset, 0, morton_max_coord - 1)
    k = jnp.clip(k + morton_offset, 0, morton_max_coord - 1)

    # Encode
    return encode_morton_3d_jax(i, j, k)


# ============================================================================
# Cell Lookup (Binary Search in Sorted Morton Array)
# ============================================================================

def find_cell_by_morton_and_level(
    morton_code: jnp.uint64,
    level: jnp.uint8,
    cell_morton_codes: jax.Array,
    cell_levels: jax.Array,
) -> jnp.int32:
    """
    Find cell index by (Morton code, level) pair using binary search.

    CRITICAL: Cells are sorted by (morton, level) tuples. We must find
    a cell where BOTH morton AND level match.

    Args:
        morton_code: Query Morton code
        level: Query refinement level
        cell_morton_codes: (n_cells,) uint64, sorted
        cell_levels: (n_cells,) uint8, sorted by (morton, level)

    Returns:
        Cell index, or -1 if not found
    """
    max_iters = 25

    def search_step(i, carry):
        left, right, found_idx = carry
        is_active = left < right

        mid = (left + right) // 2
        mid_morton = jnp.where(is_active, cell_morton_codes[mid], jnp.uint64(0))
        mid_level = jnp.where(is_active, cell_levels[mid], jnp.uint8(0))

        # Compare (morton, level) tuples lexicographically
        # First compare morton codes
        mid_less = jnp.logical_or(
            mid_morton < morton_code,
            jnp.logical_and(mid_morton == morton_code, mid_level < level)
        )
        mid_greater = jnp.logical_or(
            mid_morton > morton_code,
            jnp.logical_and(mid_morton == morton_code, mid_level > level)
        )

        # Update bounds
        new_left = jnp.where(
            jnp.logical_and(is_active, mid_less),
            mid + 1,
            left
        )
        new_right = jnp.where(
            jnp.logical_and(is_active, mid_greater),
            mid,
            right
        )

        # Check if found (both morton AND level match)
        is_found = jnp.logical_and(
            jnp.logical_and(is_active, mid_morton == morton_code),
            mid_level == level
        )
        new_found_idx = jnp.where(is_found, mid, found_idx)

        return (new_left, new_right, new_found_idx)

    n_cells = cell_morton_codes.shape[0]
    init_state = (jnp.int32(0), jnp.int32(n_cells), jnp.int32(-1))
    final_state = lax.fori_loop(0, max_iters, search_step, init_state)

    _, _, found_idx = final_state
    return found_idx


def find_cell_by_morton(
    morton_code: jnp.uint64,
    cell_morton_codes: jax.Array,
) -> jnp.int32:
    """
    Find cell index by Morton code using binary search.

    DEPRECATED: Use find_cell_by_morton_and_level() instead.
    This function is kept for backward compatibility only.

    Args:
        morton_code: Query Morton code
        cell_morton_codes: (n_cells,) uint64, sorted Morton codes

    Returns:
        Cell index, or -1 if not found
    """
    # Binary search with fixed iteration count
    # Maximum cells: ~652k → log2(652k) ≈ 20 iterations
    # Use 25 iterations for safety
    max_iters = 25

    def search_step(i, carry):
        left, right, found_idx = carry

        # Check if search space is exhausted
        is_active = left < right

        mid = (left + right) // 2
        mid_code = jnp.where(is_active, cell_morton_codes[mid], jnp.uint64(0))

        # Update bounds
        new_left = jnp.where(
            jnp.logical_and(is_active, mid_code < morton_code),
            mid + 1,
            left
        )
        new_right = jnp.where(
            jnp.logical_and(is_active, mid_code > morton_code),
            mid,
            right
        )

        # Check if found
        is_found = jnp.logical_and(is_active, mid_code == morton_code)
        new_found_idx = jnp.where(is_found, mid, found_idx)

        return (new_left, new_right, new_found_idx)

    # Initial state: (left, right, found_idx)
    n_cells = cell_morton_codes.shape[0]
    init_state = (jnp.int32(0), jnp.int32(n_cells), jnp.int32(-1))

    # Run binary search with fori_loop (static iteration count)
    final_state = lax.fori_loop(0, max_iters, search_step, init_state)

    _, _, found_idx = final_state
    return found_idx


def get_cell_elements(
    cell_idx: jnp.int32,
    cell_to_elements_offsets: jax.Array,
    cell_to_elements_data: jax.Array,
) -> Tuple[jnp.int32, jnp.int32]:
    """
    Get element range for a cell using CSR format.

    Args:
        cell_idx: Cell index
        cell_to_elements_offsets: (n_cells + 1,) int32
        cell_to_elements_data: (total_entries,) int32

    Returns:
        (start_idx, length): Range in cell_to_elements_data
                            Elements are data[start_idx:start_idx+length]
    """
    # Handle invalid cell
    start_idx = jnp.where(
        cell_idx >= 0,
        cell_to_elements_offsets[cell_idx],
        0
    )
    end_idx = jnp.where(
        cell_idx >= 0,
        cell_to_elements_offsets[cell_idx + 1],
        0
    )
    length = end_idx - start_idx

    return start_idx, length


# ============================================================================
# Upload Function
# ============================================================================

def upload_mesh_aligned_octree_to_gpu(
    connectivity: np.ndarray,
    node_positions: np.ndarray,
    octree_cells,  # OctreeCellData from Phase 2
    verbose: bool = True
) -> MeshAlignedOctreeGPU:
    """
    Upload mesh-aligned octree to GPU for point location.

    Parameters
    ----------
    connectivity : ndarray, shape (n_elements, 4), int32
        Element connectivity
    node_positions : ndarray, shape (n_nodes, 3), float32
        Node coordinates
    octree_cells : OctreeCellData
        Phase 2 octree cell structure (CPU arrays)
    verbose : bool
        Print upload statistics

    Returns
    -------
    octree_gpu : MeshAlignedOctreeGPU
        GPU-resident octree structure
    """
    if verbose:
        print("Uploading mesh-aligned octree to GPU...")

    # Upload mesh data
    connectivity_gpu = jnp.array(connectivity, dtype=jnp.int32)
    node_positions_gpu = jnp.array(node_positions, dtype=config.FLOAT_DTYPE_JNP)

    # Upload cell structure
    cell_morton_codes_gpu = jnp.array(octree_cells.cell_morton_codes, dtype=jnp.uint64)
    cell_levels_gpu = jnp.array(octree_cells.cell_levels, dtype=jnp.uint8)
    cell_sizes_gpu = jnp.array(octree_cells.cell_sizes, dtype=config.FLOAT_DTYPE_JNP)
    cell_grid_indices_gpu = jnp.array(octree_cells.cell_grid_indices, dtype=jnp.int32)

    # Upload CSR structure
    cell_to_elements_offsets_gpu = jnp.array(
        octree_cells.cell_to_elements_offsets, dtype=jnp.int32
    )
    cell_to_elements_data_gpu = jnp.array(
        octree_cells.cell_to_elements_data, dtype=jnp.int32
    )

    # Compute bounding box
    bbox_min = node_positions.min(axis=0).astype(config.FLOAT_DTYPE_NP)
    bbox_max = node_positions.max(axis=0).astype(config.FLOAT_DTYPE_NP)
    bbox_min_gpu = jnp.array(bbox_min)
    bbox_max_gpu = jnp.array(bbox_max)

    # Morton parameters (matching Phase 2)
    morton_offset = jnp.int32(1 << 19)  # 2^19 for signed coordinates
    morton_max_coord = jnp.int32(1 << 20)  # 2^20 max coordinate
    max_depth = jnp.int32(21)  # 21 bits per dimension

    # Compute representative cell size for each level
    # CRITICAL: Use actual cell sizes from mesh, not derived formulas!
    # This ensures grid index computation matches between assignment and query.
    unique_levels = np.unique(octree_cells.cell_levels)
    min_level_int = int(np.min(unique_levels))
    max_level_int = int(np.max(unique_levels))
    level_cell_sizes_cpu = np.zeros((max_level_int + 1, 3), dtype=config.FLOAT_DTYPE_NP)

    # Per-axis MEDIAN, not the first cell's size.
    #
    # The old code took level_sizes[0] on the stated assumption that "all
    # should be nearly identical per level". That holds for a cubic mesh but
    # NOT for an anisotropic (cuboid) one. On the D-family FSW meshes the
    # tetrahedra near the tilted concave shoulder are cuboid with a constant
    # aspect ratio dz/dx = 1.197, and `level` is derived from the MEAN of
    # [dx,dy,dz] (mesh_aligned_octree_single_cell.py:101-104), so cells of
    # different size land on the same level. Measured on D2: 99.42% of the
    # 5.88M level-14 cells are cuboid (dx=dy=4.6875e-05, dz=5.6117e-05) and
    # only 0.58% are cubic -- and level_sizes[0] happened to be one of the
    # cubic outliers. The search then divided z by 4.6875e-05 while the real
    # cells were 5.6117e-05 tall, inflating every z index by 1.197x. The error
    # grows with depth and reaches |dk| = 14 at z = -4 mm, far outside the +-1
    # that the 3x3x3 neighbourhood can absorb, so the true host cell was never
    # visited and the particle was reported lost.
    #
    # The median is robust to both the minority outliers and the ~1% per-cell
    # jitter that mesh motion introduces. Measured unfindable-cell counts on
    # D2 (probing each cell centre, so a lower bound):
    #     first (old)      57,369   0.96%
    #     median (new)      9,266   0.15%     <- 6.2x better
    #     mean             13,139   0.22%
    # Median, dominant-mode and cluster-median all converge on 9,266, which is
    # why the simple median is enough. On a cubic mesh (A1, every level 100%
    # one size) median == first, so this is a no-op for the A/B/C families:
    # 0 unfindable cells before and after.
    #
    # No kernel change is needed: level_cell_sizes is already (max_level+1, 3)
    # and every search site divides each axis by its own pitch, so cuboid
    # cells are already supported. This is purely a build-time choice.
    low_share_levels = []
    for level in unique_levels:
        level_mask = octree_cells.cell_levels == level
        level_sizes = octree_cells.cell_sizes[level_mask]
        level_cell_sizes_cpu[level] = np.median(level_sizes, axis=0)

        # Warn when a level is not dominated by one size family, i.e. when the
        # one-pitch-per-level model is a poor fit for this mesh. Cluster in
        # RELATIVE terms (5%) so mesh-motion jitter does not fragment a family.
        if len(level_sizes) > 1:
            key = np.round(np.log2(np.maximum(level_sizes, 1e-300))
                           / np.log2(1.05))
            _, counts = np.unique(key, axis=0, return_counts=True)
            share = counts.max() / len(level_sizes)
            if share < 0.60:
                low_share_levels.append((int(level), len(level_sizes), share))

    level_cell_sizes_gpu = jnp.array(level_cell_sizes_cpu, dtype=config.FLOAT_DTYPE_JNP)
    min_level_gpu = jnp.int32(min_level_int)
    max_level_gpu = jnp.int32(max_level_int)

    if verbose:
        print(f"  octree level range on this mesh: [{min_level_int}, {max_level_int}] "
              f"(kernel will iterate {max_level_int - min_level_int + 1} levels)")
        # Report anisotropy so a cuboid mesh is visible in the log rather than
        # silently relying on the median being a good fit.
        aspect = level_cell_sizes_cpu[unique_levels]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = aspect.max(axis=1) / np.where(aspect.min(axis=1) > 0,
                                                  aspect.min(axis=1), np.nan)
        worst = float(np.nanmax(ratio)) if len(ratio) else 1.0
        if worst > 1.01:
            print(f"  cell aspect ratio (max/min per level): up to {worst:.4f} "
                  f"-> ANISOTROPIC (cuboid) cells; per-axis pitch in use")
        if low_share_levels:
            print("  WARNING: these levels are not dominated by a single cell "
                  "size, so one pitch per level is an imperfect fit:")
            for lv, n, sh in low_share_levels:
                print(f"    level {lv}: {n:,} cells, largest size family only "
                      f"{100*sh:.1f}%")
            print("    Point location may miss hosts in those levels. See "
                  "scripts/dfamily/eval_pitch_choice.py to quantify.")

    # Statistics
    n_cells = jnp.int32(octree_cells.n_cells)
    n_elements = jnp.int32(octree_cells.n_elements)
    mean_elements_per_cell = config.FLOAT_DTYPE_JNP(octree_cells.elements_per_cell_mean)

    # Create GPU structure
    octree_gpu = MeshAlignedOctreeGPU(
        connectivity=connectivity_gpu,
        node_positions=node_positions_gpu,
        cell_morton_codes=cell_morton_codes_gpu,
        cell_levels=cell_levels_gpu,
        cell_sizes=cell_sizes_gpu,
        cell_grid_indices=cell_grid_indices_gpu,
        cell_to_elements_offsets=cell_to_elements_offsets_gpu,
        cell_to_elements_data=cell_to_elements_data_gpu,
        morton_offset=morton_offset,
        morton_max_coord=morton_max_coord,
        max_depth=max_depth,
        bbox_min=bbox_min_gpu,
        bbox_max=bbox_max_gpu,
        level_cell_sizes=level_cell_sizes_gpu,
        min_level=min_level_gpu,
        max_level=max_level_gpu,
        n_cells=n_cells,
        n_elements=n_elements,
        mean_elements_per_cell=mean_elements_per_cell,
    )

    if verbose:
        # Compute memory usage
        memory_mb = (
            connectivity_gpu.nbytes +
            node_positions_gpu.nbytes +
            cell_morton_codes_gpu.nbytes +
            cell_levels_gpu.nbytes +
            cell_sizes_gpu.nbytes +
            cell_grid_indices_gpu.nbytes +
            cell_to_elements_offsets_gpu.nbytes +
            cell_to_elements_data_gpu.nbytes
        ) / (1024 ** 2)

        print(f"  GPU memory: {memory_mb:.1f} MB")
        print(f"  Cells: {octree_cells.n_cells:,}")
        print(f"  Elements: {octree_cells.n_elements:,}")
        print(f"  Mean elements/cell: {octree_cells.elements_per_cell_mean:.1f}")
        print(f"  CSR data entries: {len(octree_cells.cell_to_elements_data):,}")

    return octree_gpu


# ============================================================================
# Helper: Infer Cell Size from Position
# ============================================================================

def infer_cell_size_at_position(
    pos: jax.Array,
    octree_gpu: MeshAlignedOctreeGPU,
    default_level: jnp.int32 = 14
) -> jax.Array:
    """
    Infer the cell size at a given position.

    For multi-level octrees, we need to determine which refinement level
    applies at the query position. This is a simplified version that uses
    a default level or nearby cell information.

    Args:
        pos: (3,) float32 - query position
        octree_gpu: GPU octree structure
        default_level: Fallback refinement level

    Returns:
        cell_size: (3,) float32 - cell dimensions at this position
    """
    # For now, use a simple heuristic: try the most common cell size
    # (could be improved by looking at nearby cells)

    # Use the first cell's size as default (assumes relatively uniform refinement)
    # In production, we'd want a more sophisticated approach
    default_size = octree_gpu.cell_sizes[0]

    return default_size
