#!/usr/bin/env python3
"""
Compare Block-Wise Leaves vs Multi-Cell Vertex Registration

This script properly analyzes the two approaches for improving retention:

Approach A: Block-Wise Leaves
  - Use 2×2×2 parent cubes (8 cells) as octree leaves
  - Each leaf contains ~6 Kuhn tetrahedra
  - Elements fully contained within their block

Approach B: Multi-Cell Vertex Registration
  - Register each element in ALL cells its vertices touch (~4 cells)
  - Keep current single-cell granularity
  - Larger element→cell mapping

This analysis calculates:
1. Number of blocks for Approach A
2. Elements per block distribution
3. GPU memory for both approaches
4. Search performance implications
"""

import numpy as np
from pathlib import Path
from collections import defaultdict

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import (
    extract_octree_cells_single,
    find_axis_aligned_edges_single
)


def analyze_block_grouping():
    """Analyze how elements group into 2×2×2 parent blocks."""

    print("="*80)
    print("Loading mesh...")
    print("="*80)

    MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")

    node_positions, connectivity, _ = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern="featurelessAvtk_{timestep}.pvtu",
        timestep_range=(158, 159),
        field_name='Displacement',
        verbose=False
    )

    node_positions, connectivity, _, _ = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=None, verbose=False
    )

    print(f"  Mesh: {connectivity.shape[0]:,} elements, {node_positions.shape[0]:,} nodes")

    print("\n" + "="*80)
    print("Extracting Current Single-Cell Octree...")
    print("="*80)

    octree_single = extract_octree_cells_single(
        node_positions, connectivity, tolerance=1e-6, verbose=True
    )

    n_cells = octree_single.n_cells
    n_elements = octree_single.n_elements
    elements_per_cell_mean = octree_single.elements_per_cell_mean

    print(f"\n  Current octree:")
    print(f"    Cells: {n_cells:,}")
    print(f"    Elements: {n_elements:,}")
    print(f"    Elements/cell (mean): {elements_per_cell_mean:.2f}")

    print("\n" + "="*80)
    print("Approach A: Block-Wise Leaves Analysis")
    print("="*80)

    # Build block→elements mapping
    # A "block" is a 2×2×2 group of cells
    # Block key: (block_i, block_j, block_k, level) where block_i = cell_i // 2

    block_to_elements = defaultdict(list)
    element_to_block = {}

    for elem_id in range(len(connectivity)):
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]

        cell_size, level = find_axis_aligned_edges_single(vertices, tolerance=1e-6)
        if np.any(cell_size == 0):
            continue  # Skip non-Kuhn

        # Find the 2×2×2 block that contains this element
        # Use the minimum grid coordinates of the element's vertices
        grid_coords = np.floor(vertices / cell_size).astype(int)
        min_grid = grid_coords.min(axis=0)

        # Block index: floor(cell_index / 2)
        # This groups each 2×2×2 region of cells into one block
        block_i = min_grid[0] // 2
        block_j = min_grid[1] // 2
        block_k = min_grid[2] // 2

        block_key = (block_i, block_j, block_k, level)
        block_to_elements[block_key].append(elem_id)
        element_to_block[elem_id] = block_key

    n_blocks = len(block_to_elements)
    elements_per_block = [len(elems) for elems in block_to_elements.values()]
    elements_per_block_mean = np.mean(elements_per_block)
    elements_per_block_median = np.median(elements_per_block)

    print(f"\n  Block-wise grouping results:")
    print(f"    Total blocks: {n_blocks:,}")
    print(f"    Elements per block (mean): {elements_per_block_mean:.2f}")
    print(f"    Elements per block (median): {elements_per_block_median:.0f}")
    print(f"    Elements per block (min, max): ({int(np.min(elements_per_block))}, {int(np.max(elements_per_block))})")

    # Distribution
    elements_per_block_hist = defaultdict(int)
    for count in elements_per_block:
        elements_per_block_hist[count] += 1

    print(f"\n  Elements-per-block distribution (top 10):")
    for count, n_blocks_with_count in sorted(elements_per_block_hist.items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = 100.0 * n_blocks_with_count / n_blocks
        print(f"    {count:2d} elements: {n_blocks_with_count:7,} blocks ({pct:5.2f}%)")

    print("\n" + "="*80)
    print("Approach B: Multi-Cell Vertex Registration Analysis")
    print("="*80)

    # Estimate multi-cell vertex registration size
    # Each element touches ~4 cells (as shown in comprehensive test)
    cells_per_element = 4  # From octree_verification_comprehensive_enhanced.log

    total_element_registrations = n_elements * cells_per_element

    print(f"\n  Multi-cell vertex registration estimates:")
    print(f"    Cells: {n_cells:,} (unchanged)")
    print(f"    Elements: {n_elements:,} (unchanged)")
    print(f"    Cells per element: {cells_per_element}")
    print(f"    Total element registrations: {total_element_registrations:,}")
    print(f"    Elements per cell (estimated): {total_element_registrations / n_cells:.2f}")

    print("\n" + "="*80)
    print("GPU Memory Comparison")
    print("="*80)

    # Current single-cell octree
    current_cell_metadata = n_cells * (8 + 1 + 3*8 + 3*4)  # morton(8B) + level(1B) + size(3×8B) + grid(3×4B)
    current_cell_to_elements_offsets = (n_cells + 1) * 4  # int32
    current_cell_to_elements_data = n_elements * 4  # int32, ~1 cell per element
    current_total = current_cell_metadata + current_cell_to_elements_offsets + current_cell_to_elements_data

    print(f"\n  Current (Single-Cell Registration):")
    print(f"    Cell metadata: {current_cell_metadata / 1e6:.2f} MB")
    print(f"    Cell→elements offsets: {current_cell_to_elements_offsets / 1e6:.2f} MB")
    print(f"    Cell→elements data: {current_cell_to_elements_data / 1e6:.2f} MB")
    print(f"    Total: {current_total / 1e6:.2f} MB")

    # Approach A: Block-wise leaves
    block_cell_metadata = n_blocks * (8 + 1 + 3*8 + 3*4)  # Same structure as cells
    block_to_elements_offsets = (n_blocks + 1) * 4
    block_to_elements_data = n_elements * 4  # Each element in exactly 1 block
    block_total = block_cell_metadata + block_to_elements_offsets + block_to_elements_data

    print(f"\n  Approach A (Block-Wise Leaves):")
    print(f"    Block metadata: {block_cell_metadata / 1e6:.2f} MB")
    print(f"    Block→elements offsets: {block_to_elements_offsets / 1e6:.2f} MB")
    print(f"    Block→elements data: {block_to_elements_data / 1e6:.2f} MB")
    print(f"    Total: {block_total / 1e6:.2f} MB")
    print(f"    Reduction vs current: {(current_total - block_total) / 1e6:.2f} MB ({100.0 * (current_total - block_total) / current_total:.1f}%)")

    # Approach B: Multi-cell vertex registration
    multicell_cell_metadata = n_cells * (8 + 1 + 3*8 + 3*4)  # Same number of cells
    multicell_cell_to_elements_offsets = (n_cells + 1) * 4
    multicell_cell_to_elements_data = total_element_registrations * 4  # ~4× larger
    multicell_element_to_cells_offsets = (n_elements + 1) * 4  # NEW
    multicell_element_to_cells_data = total_element_registrations * 4  # NEW
    multicell_total = (multicell_cell_metadata + multicell_cell_to_elements_offsets +
                       multicell_cell_to_elements_data + multicell_element_to_cells_offsets +
                       multicell_element_to_cells_data)

    print(f"\n  Approach B (Multi-Cell Vertex Registration):")
    print(f"    Cell metadata: {multicell_cell_metadata / 1e6:.2f} MB")
    print(f"    Cell→elements offsets: {multicell_cell_to_elements_offsets / 1e6:.2f} MB")
    print(f"    Cell→elements data: {multicell_cell_to_elements_data / 1e6:.2f} MB")
    print(f"    Element→cells offsets: {multicell_element_to_cells_offsets / 1e6:.2f} MB")
    print(f"    Element→cells data: {multicell_element_to_cells_data / 1e6:.2f} MB")
    print(f"    Total: {multicell_total / 1e6:.2f} MB")
    print(f"    Increase vs current: {(multicell_total - current_total) / 1e6:.2f} MB ({100.0 * (multicell_total - current_total) / current_total:.1f}%)")

    print("\n" + "="*80)
    print("Search Performance Implications")
    print("="*80)

    print(f"\n  Current (Single-Cell):")
    print(f"    Tests per particle: ~{elements_per_cell_mean:.1f} (direct) + ~{5 * elements_per_cell_mean:.1f} (neighbors)")
    print(f"    Issue: Elements not found when particle crosses to adjacent cell")

    print(f"\n  Approach A (Block-Wise Leaves):")
    print(f"    Tests per particle: ~{elements_per_block_mean:.1f} (direct) + ~{5 * elements_per_block_mean:.1f} (neighbors)")
    print(f"    Pro: Coarser granularity → fewer blocks to search")
    print(f"    Pro: Elements fully contained → better retention")
    print(f"    Con: More tests per block ({elements_per_block_mean:.1f} vs {elements_per_cell_mean:.1f})")
    print(f"    Con: Coarser spatial resolution → may search irrelevant blocks")

    print(f"\n  Approach B (Multi-Cell Vertex Registration):")
    print(f"    Tests per particle: ~{total_element_registrations / n_cells:.1f} (direct) + ~{5 * total_element_registrations / n_cells:.1f} (neighbors)")
    print(f"    Pro: Finer granularity → better spatial resolution")
    print(f"    Pro: Elements in all touching cells → better retention")
    print(f"    Con: More memory ({multicell_total / 1e6:.1f} MB vs {block_total / 1e6:.1f} MB)")
    print(f"    Con: More total tests (~{total_element_registrations / n_cells * 6:.1f} vs ~{elements_per_block_mean * 6:.1f})")

    print("\n" + "="*80)
    print("Recommendation")
    print("="*80)

    # Calculate efficiency metrics
    block_memory_reduction = 100.0 * (current_total - block_total) / current_total
    block_tests_increase = 100.0 * (elements_per_block_mean - elements_per_cell_mean) / elements_per_cell_mean

    multicell_memory_increase = 100.0 * (multicell_total - current_total) / current_total
    multicell_tests_increase = 100.0 * ((total_element_registrations / n_cells) - elements_per_cell_mean) / elements_per_cell_mean

    print(f"\n  Approach A (Block-Wise Leaves):")
    print(f"    ✅ Memory: {block_memory_reduction:.1f}% REDUCTION")
    print(f"    ✅ Tests per block: {elements_per_block_mean:.1f} (reasonable)")
    print(f"    ⚠️  Spatial resolution: {100.0 * n_blocks / n_cells:.1f}% of current")
    print(f"    ✅ Implementation: Simpler (no multi-cell tracking)")

    print(f"\n  Approach B (Multi-Cell Vertex Registration):")
    print(f"    ❌ Memory: {multicell_memory_increase:.1f}% INCREASE")
    print(f"    ⚠️  Tests per cell: {total_element_registrations / n_cells:.1f} (higher)")
    print(f"    ✅ Spatial resolution: 100% (unchanged)")
    print(f"    ⚠️  Implementation: More complex (bidirectional mapping)")

    print(f"\n  Overall Assessment:")
    if block_memory_reduction > 0 and elements_per_block_mean < 15:
        print(f"    ✅ RECOMMEND Approach A (Block-Wise Leaves)")
        print(f"       - Better memory efficiency")
        print(f"       - Reasonable search cost")
        print(f"       - Simpler implementation")
        print(f"       - Should achieve similar retention (elements fully in blocks)")
    else:
        print(f"    ✅ RECOMMEND Approach B (Multi-Cell Vertex Registration)")
        print(f"       - Better spatial resolution")
        print(f"       - More precise element coverage")
        print(f"       - Worth the memory cost")

    print(f"\n  Suggested Next Steps:")
    print(f"    1. Implement the recommended approach")
    print(f"    2. Run benchmark_l2_search_methods.py to measure actual retention")
    print(f"    3. If retention still insufficient, implement the other approach as backup")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    analyze_block_grouping()
