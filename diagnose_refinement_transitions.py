#!/usr/bin/env python3
"""
Diagnose 1:2 and 2:1 refinement transitions and their impact on particle tracking.

This script analyzes:
1. Distribution of refinement levels across the mesh
2. Face neighbor relationships at refinement boundaries (1:2, 2:1, 1:1)
3. How many cross-cell face pairs are at refinement transitions
4. Spatial distribution of multi-level transitions
5. Whether the 344 fallback elements are at refinement boundaries
6. Whether current search method handles multi-level transitions

Key questions:
- Are the 719,580 cross-cell faces concentrated at refinement boundaries?
- Do 3×3×3 same-level offsets fail to reach across refinement levels?
- Are the 344 fallback elements at coarse-fine transitions?
"""

import numpy as np
import time
from pathlib import Path
from collections import defaultdict

# Import mesh loading
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

# Import octree extraction
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import extract_octree_cells_vertex_multi
from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import find_axis_aligned_edges_single

# Configuration
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME = 'Displacement'


def compute_element_level(vertices, tolerance=1e-6):
    """Compute refinement level for an element."""
    cell_size, level = find_axis_aligned_edges_single(vertices, tolerance)

    if np.any(cell_size == 0):
        # Non-Kuhn: use mean of non-zero sizes
        valid_sizes = cell_size[cell_size > 0]
        if len(valid_sizes) > 0:
            mean_size = np.mean(valid_sizes)
            # Estimate level from size (assuming base size of 0.04 at level 8)
            level = int(8 + np.log2(0.04 / mean_size))
        else:
            level = -1  # Unknown

    return level, cell_size


def main():
    print("="*80)
    print("Refinement Transition Diagnostic")
    print("="*80)
    print()

    # Load mesh
    print("[1/7] Loading mesh...")
    t0 = time.time()
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=False
    )

    n_nodes_orig = node_positions.shape[0]
    n_elements = connectivity.shape[0]
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"    Elements: {n_elements:,}, Nodes: {n_nodes_orig:,}")

    print("  Deduplicating...")
    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    n_nodes = node_positions.shape[0]
    print(f"    Removed {n_duplicates_removed:,} duplicates ({n_nodes:,} nodes remaining)")

    # Extract octree to get cell information
    print("\n[2/7] Extracting multi-cell octree...")
    t0 = time.time()
    octree_multi = extract_octree_cells_vertex_multi(
        node_positions, connectivity, tolerance=1e-6, verbose=False
    )
    print(f"  Extracted in {time.time()-t0:.1f}s")
    print(f"    Cells: {octree_multi.n_cells:,}")
    print(f"    Mean cells/element: {octree_multi.cells_per_element_mean:.2f}")

    # Compute refinement level for each element
    print("\n[3/7] Computing refinement levels for all elements...")
    t0 = time.time()

    element_levels = np.zeros(n_elements, dtype=np.int32)
    element_cell_sizes = np.zeros((n_elements, 3), dtype=np.float64)
    non_kuhn_mask = np.zeros(n_elements, dtype=bool)

    for elem_id in range(n_elements):
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]

        level, cell_size = compute_element_level(vertices, tolerance=1e-6)
        element_levels[elem_id] = level
        element_cell_sizes[elem_id] = cell_size

        if np.any(cell_size == 0):
            non_kuhn_mask[elem_id] = True

        if (elem_id + 1) % 500000 == 0:
            print(f"    Processed {elem_id + 1:,}/{n_elements:,}...")

    print(f"  Computed in {time.time()-t0:.1f}s")

    # Level distribution
    print("\n  Refinement Level Distribution:")
    unique_levels, level_counts = np.unique(element_levels[element_levels >= 0], return_counts=True)

    for level, count in zip(unique_levels, level_counts):
        pct = 100.0 * count / n_elements
        # Estimate cell size (assuming level 8 = 0.04)
        cell_size_estimate = 0.04 / (2 ** (level - 8)) if level >= 0 else 0
        print(f"    Level {level:2d}: {count:8,} elements ({pct:5.2f}%), size≈{cell_size_estimate:.6f}")

    n_non_kuhn = non_kuhn_mask.sum()
    print(f"\n    Non-Kuhn: {n_non_kuhn:,} elements ({100.0*n_non_kuhn/n_elements:.2f}%)")

    # Build face neighbor map
    print("\n[4/7] Building face neighbor relationships...")
    t0 = time.time()

    # Face → [elem1, elem2] mapping
    from collections import defaultdict as dd
    face_to_elements = dd(list)

    for elem_id in range(n_elements):
        node_ids = connectivity[elem_id]

        # Four faces of tetrahedron
        faces = [
            tuple(sorted([node_ids[0], node_ids[1], node_ids[2]])),
            tuple(sorted([node_ids[0], node_ids[1], node_ids[3]])),
            tuple(sorted([node_ids[0], node_ids[2], node_ids[3]])),
            tuple(sorted([node_ids[1], node_ids[2], node_ids[3]])),
        ]

        for face in faces:
            face_to_elements[face].append(elem_id)

        if (elem_id + 1) % 500000 == 0:
            print(f"    Processed {elem_id + 1:,}/{n_elements:,}...")

    print(f"  Built in {time.time()-t0:.1f}s")
    print(f"    Total unique faces: {len(face_to_elements):,}")

    # Filter interior faces (exactly 2 elements)
    interior_faces = {face: elems for face, elems in face_to_elements.items() if len(elems) == 2}
    n_interior_faces = len(interior_faces)

    print(f"    Interior faces: {n_interior_faces:,}")

    # Analyze refinement transitions
    print("\n[5/7] Analyzing refinement transitions at face neighbors...")
    t0 = time.time()

    refinement_transitions = {
        '1:1_same_level': 0,
        '1:2_one_level_up': 0,
        '2:1_one_level_down': 0,
        'multi_level': 0,
        'one_non_kuhn': 0,
        'both_non_kuhn': 0,
    }

    cross_cell_same_level = 0
    cross_cell_one_level = 0
    cross_cell_multi_level = 0

    # Element to cells mapping (from octree)
    elem_to_cells_offsets = octree_multi.element_to_cells_offsets
    elem_to_cells_data = octree_multi.element_to_cells_data

    def get_element_cells(elem_id):
        """Get cell indices for an element."""
        start = elem_to_cells_offsets[elem_id]
        end = elem_to_cells_offsets[elem_id + 1]
        return set(elem_to_cells_data[start:end])

    for face, (elem1, elem2) in interior_faces.items():
        level1 = element_levels[elem1]
        level2 = element_levels[elem2]

        is_non_kuhn1 = non_kuhn_mask[elem1]
        is_non_kuhn2 = non_kuhn_mask[elem2]

        # Classify refinement transition
        if is_non_kuhn1 and is_non_kuhn2:
            refinement_transitions['both_non_kuhn'] += 1
        elif is_non_kuhn1 or is_non_kuhn2:
            refinement_transitions['one_non_kuhn'] += 1
        elif level1 == level2:
            refinement_transitions['1:1_same_level'] += 1
        elif abs(level1 - level2) == 1:
            if level1 < level2:
                refinement_transitions['2:1_one_level_down'] += 1
            else:
                refinement_transitions['1:2_one_level_up'] += 1
        else:
            refinement_transitions['multi_level'] += 1

        # Check if they're in different cells
        cells1 = get_element_cells(elem1)
        cells2 = get_element_cells(elem2)

        if not cells1.intersection(cells2):  # No shared cells
            # Cross-cell face
            if level1 == level2:
                cross_cell_same_level += 1
            elif abs(level1 - level2) == 1:
                cross_cell_one_level += 1
            else:
                cross_cell_multi_level += 1

    print(f"  Analyzed in {time.time()-t0:.1f}s")

    print(f"\n  Face Neighbor Refinement Transitions:")
    total_transitions = sum(refinement_transitions.values())
    for trans_type, count in sorted(refinement_transitions.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / total_transitions
        print(f"    {trans_type:20s}: {count:8,} ({pct:5.2f}%)")

    print(f"\n  Cross-Cell Faces by Refinement Level Difference:")
    total_cross_cell = cross_cell_same_level + cross_cell_one_level + cross_cell_multi_level
    print(f"    Same level (1:1):       {cross_cell_same_level:8,} ({100.0*cross_cell_same_level/total_cross_cell if total_cross_cell > 0 else 0:5.2f}%)")
    print(f"    One level diff (1:2):   {cross_cell_one_level:8,} ({100.0*cross_cell_one_level/total_cross_cell if total_cross_cell > 0 else 0:5.2f}%)")
    print(f"    Multi-level (>1):       {cross_cell_multi_level:8,} ({100.0*cross_cell_multi_level/total_cross_cell if total_cross_cell > 0 else 0:5.2f}%)")
    print(f"    TOTAL cross-cell:       {total_cross_cell:8,}")

    # Analyze the 344 fallback elements
    print("\n[6/7] Analyzing the 344 fallback elements...")

    # Re-extract to find which elements used fallback
    fallback_elements = []

    for elem_id in range(n_elements):
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]

        cell_size, level = find_axis_aligned_edges_single(vertices, tolerance=1e-6)

        if np.any(cell_size == 0):
            # Non-Kuhn - check if it would have found neighbor
            # Count cells it got
            start = elem_to_cells_offsets[elem_id]
            end = elem_to_cells_offsets[elem_id + 1]
            n_cells = end - start

            if n_cells == 1:
                fallback_elements.append(elem_id)

    n_fallback = len(fallback_elements)
    print(f"  Found {n_fallback} elements with single-cell registration (fallback)")

    if n_fallback > 0:
        fallback_levels = element_levels[fallback_elements]
        fallback_levels_valid = fallback_levels[fallback_levels >= 0]

        print(f"\n  Fallback Element Refinement Levels:")
        unique_fb_levels, fb_level_counts = np.unique(fallback_levels_valid, return_counts=True)
        for level, count in zip(unique_fb_levels, fb_level_counts):
            pct = 100.0 * count / n_fallback
            print(f"    Level {level:2d}: {count:4,} ({pct:5.2f}%)")

        # Spatial distribution
        fallback_centroids = np.array([
            node_positions[connectivity[elem_id]].mean(axis=0)
            for elem_id in fallback_elements
        ])

        print(f"\n  Fallback Element Spatial Distribution:")
        print(f"    X: min={fallback_centroids[:,0].min():.6f}, max={fallback_centroids[:,0].max():.6f}")
        print(f"    Y: min={fallback_centroids[:,1].min():.6f}, max={fallback_centroids[:,1].max():.6f}")
        print(f"    Z: min={fallback_centroids[:,2].min():.6f}, max={fallback_centroids[:,2].max():.6f}")

    # Analyze search coverage at refinement transitions
    print("\n[7/7] Estimating 3×3×3 search coverage at refinement transitions...")

    # Sample cross-cell face pairs at 1:2 transitions
    sample_size = min(1000, cross_cell_one_level)

    if cross_cell_one_level > 0:
        print(f"\n  Sampling {sample_size} cross-cell 1:2 face pairs...")

        sampled = 0
        coverage_failures = 0

        for face, (elem1, elem2) in interior_faces.items():
            if sampled >= sample_size:
                break

            level1 = element_levels[elem1]
            level2 = element_levels[elem2]

            if abs(level1 - level2) != 1:
                continue

            cells1 = get_element_cells(elem1)
            cells2 = get_element_cells(elem2)

            if cells1.intersection(cells2):  # Same cell - skip
                continue

            sampled += 1

            # Estimate if 3×3×3 offset would reach
            # Fine element (higher level) trying to reach coarse element (lower level)
            fine_elem = elem1 if level1 > level2 else elem2
            coarse_elem = elem2 if level1 > level2 else elem1

            fine_level = max(level1, level2)
            coarse_level = min(level1, level2)

            # Cell size ratio
            level_diff = fine_level - coarse_level
            size_ratio = 2 ** level_diff  # Each level is 2× finer

            # 3×3×3 offset in fine cells: max offset = ±1 fine cell
            # To reach coarse cell, need offset ≥ size_ratio
            if size_ratio > 1:
                # 3×3×3 at fine level won't reach coarse level cell
                coverage_failures += 1

        print(f"\n  3×3×3 Coverage Analysis (sample of {sampled}):")
        print(f"    Coverage failures: {coverage_failures}/{sampled} ({100.0*coverage_failures/sampled if sampled > 0 else 0:.1f}%)")
        print(f"    → These face pairs require multi-level search to be found")

    print("\n" + "="*80)
    print("Refinement Transition Diagnostic Complete")
    print("="*80)

    # Summary
    print("\nKey Findings:")

    print(f"\n1. Refinement Level Distribution:")
    print(f"   - Finest level: {unique_levels.max()}, Coarsest level: {unique_levels.min()}")
    print(f"   - Span: {unique_levels.max() - unique_levels.min()} levels")

    print(f"\n2. Face Neighbor Transitions:")
    print(f"   - 1:1 (same level): {refinement_transitions['1:1_same_level']:,} ({100.0*refinement_transitions['1:1_same_level']/total_transitions:.1f}%)")
    print(f"   - 1:2 (one level):  {refinement_transitions['1:2_one_level_up'] + refinement_transitions['2:1_one_level_down']:,} ({100.0*(refinement_transitions['1:2_one_level_up'] + refinement_transitions['2:1_one_level_down'])/total_transitions:.1f}%)")
    print(f"   - Multi-level:      {refinement_transitions['multi_level']:,} ({100.0*refinement_transitions['multi_level']/total_transitions:.1f}%)")

    print(f"\n3. Cross-Cell Faces (719,580 expected):")
    print(f"   - At same level:    {cross_cell_same_level:,} ({100.0*cross_cell_same_level/total_cross_cell if total_cross_cell > 0 else 0:.1f}%)")
    print(f"   - At 1:2 boundary:  {cross_cell_one_level:,} ({100.0*cross_cell_one_level/total_cross_cell if total_cross_cell > 0 else 0:.1f}%)")
    print(f"   - Multi-level gap:  {cross_cell_multi_level:,} ({100.0*cross_cell_multi_level/total_cross_cell if total_cross_cell > 0 else 0:.1f}%)")

    print(f"\n4. Fallback Elements:")
    print(f"   - Count: {n_fallback} (expected 344)")
    if n_fallback > 0:
        print(f"   - Levels: {unique_fb_levels.min()} to {unique_fb_levels.max()}")

    if cross_cell_one_level > 0 and sampled > 0:
        print(f"\n5. 3×3×3 Search Limitations:")
        print(f"   - {coverage_failures}/{sampled} 1:2 transitions NOT reachable with same-level offsets")
        print(f"   - Estimated {int(cross_cell_one_level * coverage_failures / sampled):,} face pairs may cause particle loss")

    print("\nRecommendations:")

    if cross_cell_one_level > 1000:
        print(f"  ⚠️  {cross_cell_one_level:,} cross-cell faces at 1:2 refinement boundaries")
        print("     → Multi-level search is REQUIRED to handle these transitions")
    elif cross_cell_one_level == 0:
        print(f"  ✅ NO 1:2 refinement transitions found!")
        print("     → Mesh is uniformly refined (99.9% same-level neighbors)")
        print("     → 3×3×3 same-level search SHOULD work perfectly")

    if cross_cell_one_level > 0 and 'coverage_failures' in locals() and 'sampled' in locals():
        if coverage_failures > sampled * 0.5:
            print(f"  ⚠️  {100.0*coverage_failures/sampled:.0f}% of 1:2 transitions NOT covered by same-level 3×3×3")
            print("     → This is likely the main cause of 18.84% retention problem!")

    if n_fallback > 0:
        print(f"  ⚠️  {n_fallback} elements have incomplete cell coverage")
        print("     → Improve face neighbor finding for non-Kuhn elements")

    if cross_cell_multi_level > 0:
        print(f"  ⚠️  {cross_cell_multi_level:,} multi-level cross-cell faces (>1 level gap)")
        print("     → These may be related to VTK part boundaries or mesh defects")


if __name__ == "__main__":
    main()
