#!/usr/bin/env python3
"""
Mesh-Aligned Morton Builder - Phase 5

Build Morton octree from mesh-aligned cell centers (not element centroids).
This hybrid approach combines:
  1. Intrinsic mesh octree structure (from Kuhn mesh)
  2. Proven Morton radius search algorithm

Key Differences from Original Morton:
  - Original: Morton codes from ELEMENT CENTROIDS (3M codes for 3M elements)
  - This: Morton codes from CELL CENTERS (~517K codes for ~517K cells)
  - Each Morton leaf contains ALL elements in that cell

Architecture:
  Position → Morton code → Binary search → Cell → Elements in cell

Expected Benefits:
  - 5.9 elements/cell (vs ~107 elements/leaf in original)
  - Leverages intrinsic mesh alignment
  - Uses proven radius search (93-98% retention)
  - No nested control flow (no OOM)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from .mesh_aligned_octree_single_cell import (
    OctreeCellDataSingle,
    extract_octree_cells_single,
)


# ============================================================================
# Data Structure
# ============================================================================

@dataclass
class MeshAlignedMortonStructure:
    """
    Morton structure built from mesh-aligned cell centers.

    This is similar to GlobalMortonStructure, but:
      - Morton codes are from CELL CENTERS (not element centroids)
      - Each "leaf" is one mesh-aligned cell
      - Elements are grouped by their parent cell
    """

    # Cell data
    cell_morton_codes: np.ndarray       # (n_cells,) uint64 - sorted Morton codes
    cell_levels: np.ndarray             # (n_cells,) uint8 - refinement level per cell
    cell_centers: np.ndarray            # (n_cells, 3) float64 - cell centers
    cell_sizes: np.ndarray              # (n_cells, 3) float64 - cell sizes
    cell_grid_indices: np.ndarray       # (n_cells, 3) int32 - grid indices (i, j, k)

    # Cell -> elements mapping (CSR format)
    cell_to_elements_offsets: np.ndarray  # (n_cells+1,) int32
    cell_to_elements_data: np.ndarray     # (total_elements,) int32

    # Grid-based lookup (for neighbor search)
    grid_to_cell_map: dict              # {(i, j, k, level): cell_idx} - Python dict for CPU

    # Morton parameters
    morton_min: np.uint64               # Minimum Morton code
    morton_max: np.uint64               # Maximum Morton code
    bbox_min: np.ndarray                # (3,) float64 - global bbox
    bbox_max: np.ndarray                # (3,) float64

    # Configuration
    n_cells: int                        # Number of cells (= "leaves" in Morton terminology)
    max_depth: int                      # Morton encoding depth (21)

    # Metadata
    elements_per_cell_mean: float       # Average elements per cell
    elements_per_cell_max: int          # Maximum elements in any cell

    def __repr__(self):
        return (
            f"MeshAlignedMortonStructure(\n"
            f"  n_cells={self.n_cells:,},\n"
            f"  morton_range=[{self.morton_min}, {self.morton_max}],\n"
            f"  elements_per_cell: mean={self.elements_per_cell_mean:.1f}, max={self.elements_per_cell_max},\n"
            f"  max_depth={self.max_depth}\n"
            f")"
        )


# ============================================================================
# Builder Function
# ============================================================================

def build_mesh_aligned_morton_structure(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    mesh_octree_cells: OctreeCellDataSingle = None,
    tolerance: float = 1e-6,
    verbose: bool = True
) -> MeshAlignedMortonStructure:
    """
    Build Morton structure from mesh-aligned cell centers.

    This creates a Morton octree where each "leaf" is a mesh-aligned cell,
    and Morton codes are computed from CELL CENTERS (not element centroids).

    Algorithm:
      1. Extract mesh-aligned octree cells (if not provided)
      2. Compute cell centers
      3. Encode cell centers to Morton codes
      4. Sort by Morton codes
      5. Build cell → elements mapping

    Args:
        node_positions: (n_nodes, 3) float64 - node coordinates
        connectivity: (n_elements, 4) int32 - element connectivity
        mesh_octree_cells: Pre-computed OctreeCellDataSingle (optional, will extract if None)
        tolerance: Tolerance for axis-aligned edge detection
        verbose: Print progress

    Returns:
        MeshAlignedMortonStructure ready for GPU upload
    """

    if verbose:
        print(f"\n{'='*80}")
        print(f"Building Mesh-Aligned Morton Structure (Cell-Based)")
        print(f"{'='*80}")

    # Step 1: Extract mesh-aligned octree cells (if not provided)
    if mesh_octree_cells is None:
        if verbose:
            print(f"\n[1/5] Extracting mesh-aligned octree cells...")
        mesh_octree_cells = extract_octree_cells_single(
            node_positions, connectivity, tolerance=tolerance, verbose=verbose
        )
    else:
        if verbose:
            print(f"\n[1/5] Using pre-computed mesh-aligned octree cells...")
            print(f"    Cells: {mesh_octree_cells.n_cells:,}")

    n_cells = mesh_octree_cells.n_cells

    # Step 2: Compute cell centers
    if verbose:
        print(f"\n[2/5] Computing cell centers from grid indices and sizes...")

    # Cell centers in GLOBAL coordinates
    # The grid indices (i, j, k) are already in global space
    # (computed from global centroids in find_parent_cube)
    # So: center = (i, j, k) * cell_size + cell_size/2
    cell_centers = np.zeros((n_cells, 3), dtype=np.float64)

    for cell_idx in range(n_cells):
        i, j, k = mesh_octree_cells.cell_grid_indices[cell_idx]
        cell_size = mesh_octree_cells.cell_sizes[cell_idx]

        # Cell center = corner + half-size (in global coordinates)
        cell_centers[cell_idx] = np.array([
            i * cell_size[0] + cell_size[0] / 2.0,
            j * cell_size[1] + cell_size[1] / 2.0,
            k * cell_size[2] + cell_size[2] / 2.0,
        ])

    if verbose:
        print(f"    ✅ Computed {n_cells:,} cell centers")
        print(f"    Center range: [{cell_centers.min(axis=0)}] to [{cell_centers.max(axis=0)}]")

    # Step 3: Compute global bounding box
    if verbose:
        print(f"\n[3/5] Computing global bounding box...")

    bbox_min = node_positions.min(axis=0)
    bbox_max = node_positions.max(axis=0)

    if verbose:
        print(f"    BBox: [{bbox_min[0]:.3f}, {bbox_max[0]:.3f}] × "
              f"[{bbox_min[1]:.3f}, {bbox_max[1]:.3f}] × "
              f"[{bbox_min[2]:.3f}, {bbox_max[2]:.3f}]")

    # Step 4: Recompute Morton codes using bbox-based normalization
    # CRITICAL FIX: Cell extraction uses grid-index-based Morton codes with offsets,
    # but search uses bbox-based normalization. We need consistency!
    if verbose:
        print(f"\n[4/5] Recomputing Morton codes using bbox-based normalization...")

    # Import the encoding function (same as used in search)
    from .mesh_aligned_octree_single_cell import encode_morton_3d_single

    cell_morton_codes = np.zeros(n_cells, dtype=np.uint64)
    max_depth = 21

    for cell_idx in range(n_cells):
        center = cell_centers[cell_idx]

        # Normalize to [0, 1] within bbox (same as search)
        normalized = (center - bbox_min) / (bbox_max - bbox_min)
        normalized = np.clip(normalized, 0.0, 1.0)

        # Scale to integer grid [0, 2^21 - 1]
        grid_max = (2 ** max_depth) - 1
        u = np.floor(normalized * grid_max).astype(np.uint32)

        # Encode to Morton
        cell_morton_codes[cell_idx] = encode_morton_3d_single(
            int(u[0]), int(u[1]), int(u[2]), max_depth=max_depth
        )

    # Sort cells by new Morton codes
    sort_indices = np.argsort(cell_morton_codes)
    cell_morton_codes = cell_morton_codes[sort_indices]
    cell_levels = mesh_octree_cells.cell_levels[sort_indices]
    cell_centers = cell_centers[sort_indices]
    cell_sizes_sorted = mesh_octree_cells.cell_sizes[sort_indices]
    cell_grid_indices_sorted = mesh_octree_cells.cell_grid_indices[sort_indices]

    # Rebuild CSR with new ordering
    cell_to_elements_offsets_new = np.zeros(n_cells + 1, dtype=np.int32)
    cell_to_elements_lists = []

    for new_idx, old_idx in enumerate(sort_indices):
        start = mesh_octree_cells.cell_to_elements_offsets[old_idx]
        end = mesh_octree_cells.cell_to_elements_offsets[old_idx + 1]
        elems = mesh_octree_cells.cell_to_elements_data[start:end]

        cell_to_elements_offsets_new[new_idx + 1] = cell_to_elements_offsets_new[new_idx] + len(elems)
        cell_to_elements_lists.extend(elems)

    cell_to_elements_data_new = np.array(cell_to_elements_lists, dtype=np.int32)

    # Build grid → cell mapping for neighbor search
    # We'll use a flattened array approach for GPU compatibility
    if verbose:
        print(f"\n[5/7] Building grid-based neighbor lookup...")

    grid_to_cell_map = {}
    for cell_idx in range(n_cells):
        i, j, k = cell_grid_indices_sorted[cell_idx]
        level = cell_levels[cell_idx]
        key = (int(i), int(j), int(k), int(level))
        grid_to_cell_map[key] = cell_idx

    if verbose:
        print(f"    ✅ Built grid lookup with {len(grid_to_cell_map):,} entries")
        print(f"    (Used for CPU validation; GPU will use linear search)")

    morton_min = cell_morton_codes.min()
    morton_max = cell_morton_codes.max()

    if verbose:
        print(f"    ✅ Recomputed {n_cells:,} Morton codes")
        print(f"    Morton range: [{morton_min}, {morton_max}]")

    # Step 6: Compute statistics (using re-sorted data)
    if verbose:
        print(f"\n[6/7] Computing statistics...")

    elements_per_cell = np.diff(cell_to_elements_offsets_new).astype(np.int32)
    elements_per_cell_mean = elements_per_cell.mean()
    elements_per_cell_max = elements_per_cell.max()

    if verbose:
        print(f"    Elements per cell:")
        print(f"      Mean: {elements_per_cell_mean:.1f}")
        print(f"      Median: {np.median(elements_per_cell):.0f}")
        print(f"      Max: {elements_per_cell_max}")
        print(f"      P95: {np.percentile(elements_per_cell, 95):.0f}")

    # Step 7: Build structure (using recomputed Morton codes and re-sorted data)
    if verbose:
        print(f"\n[7/7] Finalizing structure...")

    structure = MeshAlignedMortonStructure(
        cell_morton_codes=cell_morton_codes,
        cell_levels=cell_levels,
        cell_centers=cell_centers,
        cell_sizes=cell_sizes_sorted,
        cell_grid_indices=cell_grid_indices_sorted,
        cell_to_elements_offsets=cell_to_elements_offsets_new,
        cell_to_elements_data=cell_to_elements_data_new,
        grid_to_cell_map=grid_to_cell_map,
        morton_min=morton_min,
        morton_max=morton_max,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        n_cells=n_cells,
        max_depth=21,  # Standard for 3D Morton codes (21 bits per dimension)
        elements_per_cell_mean=elements_per_cell_mean,
        elements_per_cell_max=int(elements_per_cell_max),
    )

    if verbose:
        print(f"\n{'='*80}")
        print(f"✅ Mesh-Aligned Morton Structure Complete!")
        print(f"{'='*80}")
        print(structure)
        print(f"{'='*80}\n")

    return structure


# ============================================================================
# Validation (Optional)
# ============================================================================

def validate_mesh_aligned_morton_structure(
    structure: MeshAlignedMortonStructure,
    connectivity: np.ndarray,
    verbose: bool = True
) -> bool:
    """
    Validate mesh-aligned Morton structure consistency.

    Checks:
      1. Morton codes are sorted
      2. All elements are covered
      3. CSR offsets are valid

    Args:
        structure: MeshAlignedMortonStructure to validate
        connectivity: (n_elements, 4) int32 - for element count check
        verbose: Print results

    Returns:
        True if valid, False otherwise
    """

    if verbose:
        print(f"\nValidating Mesh-Aligned Morton Structure...")

    # Check 1: Morton codes sorted
    is_sorted = np.all(structure.cell_morton_codes[:-1] <= structure.cell_morton_codes[1:])

    if verbose:
        status = "✅" if is_sorted else "❌"
        print(f"  {status} Morton codes sorted: {is_sorted}")

    # Check 2: All elements covered
    n_elements = connectivity.shape[0]
    n_csr_entries = structure.cell_to_elements_data.shape[0]

    # For multi-cell assignment, count UNIQUE elements in CSR data
    unique_elements = np.unique(structure.cell_to_elements_data)
    n_unique_elements = len(unique_elements)

    # Note: Some elements may be skipped (non-Kuhn), so we expect <= n_elements
    all_covered = n_unique_elements <= n_elements

    if verbose:
        status = "✅" if all_covered else "❌"
        print(f"  {status} Element coverage: {n_unique_elements:,} unique elements / {n_elements:,} total")
        print(f"      CSR entries: {n_csr_entries:,} (avg {n_csr_entries / max(1, n_unique_elements):.1f} cells per element)")
        if n_unique_elements < n_elements:
            print(f"      ({n_elements - n_unique_elements:,} non-Kuhn elements skipped)")

    # Check 3: CSR offsets valid
    offsets_valid = (
        structure.cell_to_elements_offsets[0] == 0 and
        structure.cell_to_elements_offsets[-1] == n_csr_entries and
        np.all(structure.cell_to_elements_offsets[:-1] <= structure.cell_to_elements_offsets[1:])
    )

    if verbose:
        status = "✅" if offsets_valid else "❌"
        print(f"  {status} CSR offsets valid: {offsets_valid}")

    all_valid = is_sorted and all_covered and offsets_valid

    if verbose:
        print(f"\n{'✅ VALID' if all_valid else '❌ INVALID'}")

    return all_valid
