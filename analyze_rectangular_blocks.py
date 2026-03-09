#!/usr/bin/env python3
"""
Analyze Rectangular Block Structure for Kuhn Elements

This script investigates whether we can use rectangular blocks (bounding boxes
of Kuhn tetrahedra) as octree leaves instead of single cells.

Key Questions:
1. What are the typical dimensions of these blocks? (2×2×1, 2×2×2, etc.)
2. Do these blocks fully contain their tetrahedra volumes?
3. Can we group 6 Kuhn tets into their parent 2×2×2 block?
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


def tetrahedron_volume(vertices):
    """Compute volume of tetrahedron using cross product formula."""
    v0, v1, v2, v3 = vertices
    return abs(np.dot(v0 - v3, np.cross(v1 - v3, v2 - v3))) / 6.0


def analyze_block_structure():
    """Analyze rectangular block structure of Kuhn elements."""

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
    print("Analyzing Rectangular Block Structure")
    print("="*80)

    # Sample random elements
    np.random.seed(42)
    n_sample = 10000
    sample_elem_ids = np.random.choice(len(connectivity), size=n_sample, replace=False)

    # Statistics
    block_dimensions = defaultdict(int)  # (dx_cells, dy_cells, dz_cells) -> count
    block_volumes = []
    tet_volumes = []
    coverage_ratios = []

    # Detailed samples
    sample_blocks = []

    print(f"\nAnalyzing {n_sample:,} random elements...")

    for idx, elem_id in enumerate(sample_elem_ids):
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]

        # Get cell size
        cell_size, level = find_axis_aligned_edges_single(vertices, tolerance=1e-6)

        if np.any(cell_size == 0):
            continue  # Skip non-Kuhn

        # Find bounding box in grid coordinates
        grid_coords = np.floor(vertices / cell_size).astype(int)

        # Compute block dimensions
        min_grid = grid_coords.min(axis=0)
        max_grid = grid_coords.max(axis=0)
        block_dims = tuple((max_grid - min_grid + 1).tolist())  # +1 for inclusive range

        block_dimensions[block_dims] += 1

        # Compute volumes
        tet_vol = tetrahedron_volume(vertices)
        block_vol = np.prod(cell_size * np.array(block_dims))

        tet_volumes.append(tet_vol)
        block_volumes.append(block_vol)
        coverage_ratios.append(tet_vol / block_vol if block_vol > 0 else 0)

        # Sample detailed info
        if len(sample_blocks) < 5:
            sample_blocks.append({
                'elem_id': int(elem_id),
                'level': level,
                'block_dims': block_dims,
                'grid_min': tuple(min_grid.tolist()),
                'grid_max': tuple(max_grid.tolist()),
                'tet_vol': float(tet_vol),
                'block_vol': float(block_vol),
                'coverage': float(tet_vol / block_vol if block_vol > 0 else 0)
            })

    print(f"\n{'='*80}")
    print("Results: Block Dimension Distribution")
    print("="*80)

    total_analyzed = len(coverage_ratios)

    print(f"\nTotal analyzed: {total_analyzed:,} elements")
    print(f"\nBlock dimensions (sorted by frequency):")

    for dims, count in sorted(block_dimensions.items(), key=lambda x: x[1], reverse=True)[:15]:
        pct = 100.0 * count / total_analyzed
        n_cells = dims[0] * dims[1] * dims[2]
        print(f"  {dims[0]}×{dims[1]}×{dims[2]} ({n_cells:2d} cells): {count:6,} elements ({pct:5.2f}%)")

    print(f"\n{'='*80}")
    print("Results: Volume Coverage Analysis")
    print("="*80)

    coverage_ratios = np.array(coverage_ratios)

    print(f"\nTet volume / Block volume ratio:")
    print(f"  Mean:   {coverage_ratios.mean():.6f}")
    print(f"  Median: {np.median(coverage_ratios):.6f}")
    print(f"  Min:    {coverage_ratios.min():.6f}")
    print(f"  Max:    {coverage_ratios.max():.6f}")

    print(f"\nPercentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(coverage_ratios, p)
        print(f"  {p:3d}%: {val:.6f}")

    print(f"\n{'='*80}")
    print("Results: Block Cell Count Distribution")
    print("="*80)

    # Group by total cell count
    cells_count = defaultdict(int)
    for dims, count in block_dimensions.items():
        n_cells = dims[0] * dims[1] * dims[2]
        cells_count[n_cells] += count

    print(f"\nBlocks grouped by total cell count:")
    for n_cells in sorted(cells_count.keys()):
        count = cells_count[n_cells]
        pct = 100.0 * count / total_analyzed
        print(f"  {n_cells:2d}-cell blocks: {count:6,} elements ({pct:5.2f}%)")

    # Key blocks
    blocks_8cell = block_dimensions.get((2, 2, 2), 0)
    blocks_4cell_221 = block_dimensions.get((2, 2, 1), 0)
    blocks_4cell_212 = block_dimensions.get((2, 1, 2), 0)
    blocks_4cell_122 = block_dimensions.get((1, 2, 2), 0)

    print(f"\nKey block types:")
    print(f"  2×2×2 (8 cells):  {blocks_8cell:6,} ({100.0*blocks_8cell/total_analyzed:5.2f}%)")
    print(f"  2×2×1 (4 cells):  {blocks_4cell_221:6,} ({100.0*blocks_4cell_221/total_analyzed:5.2f}%)")
    print(f"  2×1×2 (4 cells):  {blocks_4cell_212:6,} ({100.0*blocks_4cell_212/total_analyzed:5.2f}%)")
    print(f"  1×2×2 (4 cells):  {blocks_4cell_122:6,} ({100.0*blocks_4cell_122/total_analyzed:5.2f}%)")

    print(f"\n{'='*80}")
    print("Sample Block Details")
    print("="*80)

    for i, block in enumerate(sample_blocks):
        print(f"\nElement {block['elem_id']} (level {block['level']}):")
        print(f"  Block dims: {block['block_dims']} = {np.prod(block['block_dims'])} cells")
        print(f"  Grid range: {block['grid_min']} → {block['grid_max']}")
        print(f"  Tet volume: {block['tet_vol']:.6e}")
        print(f"  Block volume: {block['block_vol']:.6e}")
        print(f"  Coverage: {block['coverage']:.4f} ({100*block['coverage']:.2f}%)")

    print(f"\n{'='*80}")
    print("Interpretation")
    print("="*80)

    # Check if typical Kuhn block is 2×2×2
    pct_8cell = 100.0 * blocks_8cell / total_analyzed
    pct_4cell = 100.0 * sum(cells_count[n] for n in [4]) / total_analyzed

    mean_coverage = coverage_ratios.mean()

    print(f"\n1. Block Structure:")
    if pct_8cell > 50:
        print(f"   ✅ Majority ({pct_8cell:.1f}%) are 2×2×2 blocks (8 cells)")
        print(f"   → Kuhn tets DO come from parent cubes")
    elif pct_4cell > 50:
        print(f"   ⚠️  Majority ({pct_4cell:.1f}%) are 4-cell blocks (not full cubes)")
        print(f"   → Blocks are rectangular, not cubic")
    else:
        print(f"   ❌ Mixed block sizes - no clear pattern")

    print(f"\n2. Volume Coverage:")
    if mean_coverage < 0.2:
        print(f"   ❌ Poor coverage ({mean_coverage:.4f}) - blocks waste space")
        print(f"   → NOT suitable as octree leaves")
    elif mean_coverage < 0.5:
        print(f"   ⚠️  Moderate coverage ({mean_coverage:.4f}) - some waste")
        print(f"   → May work but not optimal")
    else:
        print(f"   ✅ Good coverage ({mean_coverage:.4f})")
        print(f"   → Blocks efficiently contain tets")

    print(f"\n3. Recommendation:")
    if pct_8cell > 80 and mean_coverage > 0.15:
        print(f"   ✅ Use 2×2×2 parent cubes as octree leaves")
        print(f"   → Each leaf contains ~6 Kuhn tets")
        print(f"   → Better than multi-cell vertex registration")
    else:
        print(f"   ⚠️  Proceed with multi-cell vertex registration (Phase 2)")
        print(f"   → Rectangular blocks don't form clean octree structure")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    analyze_block_structure()
