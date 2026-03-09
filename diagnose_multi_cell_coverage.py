#!/usr/bin/env python3
"""
Multi-Cell Octree Coverage Diagnostics

Traces the mesh_aligned_octree_multi_local search method to identify sources
of particle loss in tetrahedral voids.

Tests:
1. Multi-cell vertex registration coverage (4 cells per element)
2. 2×2×2 local search pattern effectiveness
3. Cross-cell boundary particle tracking
4. Missing elements in local search neighborhoods
5. Level-specific coverage analysis
"""

import numpy as np
import time
from pathlib import Path
from collections import defaultdict

# Import mesh loading
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

# Import octree extraction
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import (
    extract_octree_cells_vertex_multi,
    OctreeCellDataVertexMulti
)

# Configuration
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME = 'Displacement'


def encode_morton_3d_np(i, j, k):
    """Numpy version of Morton encoding."""
    morton = 0
    for bit in range(21):
        morton |= ((i & (1 << bit)) << (2 * bit)) | \
                  ((j & (1 << bit)) << (2 * bit + 1)) | \
                  ((k & (1 << bit)) << (2 * bit + 2))
    return morton


def test_multi_cell_registration_coverage(
    connectivity, node_positions, octree_multi: OctreeCellDataVertexMulti
):
    """
    Test 1: Verify multi-cell vertex registration covers all elements.

    Checks:
    - How many elements are registered in 1, 2, 3, 4 cells
    - Which elements have < 4 cells (incomplete registration)
    - Element coverage vs expected 4 cells for Kuhn tetrahedra
    """
    print("\n" + "="*80)
    print("Test 1: Multi-Cell Vertex Registration Coverage")
    print("="*80)

    n_elements = connectivity.shape[0]

    # Build element→cells mapping
    print("  Building element→cells mapping from octree...")
    element_to_cells = defaultdict(set)

    for cell_idx in range(octree_multi.n_cells):
        start = octree_multi.cell_to_elements_offsets[cell_idx]
        end = octree_multi.cell_to_elements_offsets[cell_idx + 1]
        elem_ids = octree_multi.cell_to_elements_data[start:end]

        for elem_id in elem_ids:
            element_to_cells[elem_id].add(cell_idx)

    # Analyze coverage
    cells_per_element = np.array([len(element_to_cells.get(i, set())) for i in range(n_elements)])

    print(f"\n  Results:")
    print(f"    Total elements: {n_elements:,}")
    print(f"    Elements in octree: {len(element_to_cells):,}")
    print(f"    Elements NOT in octree: {n_elements - len(element_to_cells):,}")

    print(f"\n  Cells per element distribution:")
    for n_cells in range(6):
        count = np.sum(cells_per_element == n_cells)
        pct = 100 * count / n_elements
        if count > 0:
            print(f"    {n_cells} cells: {count:,} elements ({pct:.2f}%)")

    print(f"\n  Statistics:")
    covered_elements = cells_per_element[cells_per_element > 0]
    if len(covered_elements) > 0:
        print(f"    Mean cells/element: {covered_elements.mean():.2f}")
        print(f"    Median cells/element: {np.median(covered_elements):.0f}")
        print(f"    Min cells/element: {covered_elements.min():.0f}")
        print(f"    Max cells/element: {covered_elements.max():.0f}")

    # Find elements with incomplete registration (< 4 cells)
    incomplete_elements = np.where(np.logical_and(cells_per_element > 0, cells_per_element < 4))[0]
    if len(incomplete_elements) > 0:
        print(f"\n  ⚠️  WARNING: {len(incomplete_elements):,} elements have < 4 cells")
        print(f"      These elements may cause particle loss at cell boundaries!")
        print(f"      Sample element IDs: {incomplete_elements[:10].tolist()}")

    # Find elements with 0 cells (completely missing)
    missing_elements = np.where(cells_per_element == 0)[0]
    if len(missing_elements) > 0:
        print(f"\n  ❌ CRITICAL: {len(missing_elements):,} elements NOT in octree")
        print(f"      Particles in these elements CANNOT be found!")
        print(f"      Sample element IDs: {missing_elements[:10].tolist()}")

    return element_to_cells, cells_per_element


def test_3x3x3_local_search_pattern(
    connectivity, node_positions, octree_multi: OctreeCellDataVertexMulti,
    element_to_cells
):
    """
    Test 2: Analyze 3×3×3 local search pattern effectiveness.

    For each element:
    - Find the cell where particle centroid is located
    - Check if element is in ANY of the 27 cells in 3×3×3 neighborhood
    - Identify cases where element is OUTSIDE the 3×3×3 neighborhood
    """
    print("\n" + "="*80)
    print("Test 2: 3×3×3 Local Search Pattern Analysis")
    print("="*80)

    print("\n  Computing element centroids...")
    n_elements = connectivity.shape[0]
    element_centroids = np.zeros((n_elements, 3), dtype=np.float32)
    for elem_id in range(n_elements):
        if elem_id % 500000 == 0 and elem_id > 0:
            print(f"    Processed {elem_id:,}/{n_elements:,}...")
        node_ids = connectivity[elem_id]
        element_centroids[elem_id] = node_positions[node_ids].mean(axis=0)

    print("\n  Building cell spatial index (grid_indices → cell_idx)...")
    cell_grid_to_idx = {}
    for cell_idx in range(octree_multi.n_cells):
        morton = octree_multi.cell_morton_codes[cell_idx]
        level = octree_multi.cell_levels[cell_idx]
        i, j, k = octree_multi.cell_grid_indices[cell_idx]
        key = (int(i), int(j), int(k), int(level))
        cell_grid_to_idx[key] = cell_idx

    print("\n  Testing 3×3×3 search pattern for sample elements...")
    n_sample = min(10000, n_elements)
    sample_indices = np.random.choice(n_elements, n_sample, replace=False)

    searchable_in_3x3x3 = 0
    not_searchable = 0
    no_cells = 0

    not_searchable_elem_ids = []

    for elem_id in sample_indices:
        # Skip if element not in octree
        if elem_id not in element_to_cells:
            no_cells += 1
            continue

        # Get element's cells
        elem_cells = element_to_cells[elem_id]

        # Find centroid cell
        centroid = element_centroids[elem_id]

        # Try each level where element is registered
        found_in_neighborhood = False

        for cell_idx in elem_cells:
            level = octree_multi.cell_levels[cell_idx]
            cell_size = octree_multi.cell_sizes[cell_idx]

            # Compute base grid indices for centroid
            i_base = int(np.floor(centroid[0] / cell_size[0]))
            j_base = int(np.floor(centroid[1] / cell_size[1]))
            k_base = int(np.floor(centroid[2] / cell_size[2]))

            # Check 3×3×3 neighborhood centered at [-1,-1,-1] to [1,1,1]
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    for dk in [-1, 0, 1]:
                        i = i_base + di
                        j = j_base + dj
                        k = k_base + dk

                        # Check if this cell contains the element
                        key = (i, j, k, int(level))
                        if key in cell_grid_to_idx:
                            neighbor_cell_idx = cell_grid_to_idx[key]

                            # Check if element is in this neighbor cell
                            start = octree_multi.cell_to_elements_offsets[neighbor_cell_idx]
                            end = octree_multi.cell_to_elements_offsets[neighbor_cell_idx + 1]
                            cell_elements = octree_multi.cell_to_elements_data[start:end]

                            if elem_id in cell_elements:
                                found_in_neighborhood = True
                                break
                    if found_in_neighborhood:
                        break
                if found_in_neighborhood:
                    break

            if found_in_neighborhood:
                break

        if found_in_neighborhood:
            searchable_in_3x3x3 += 1
        else:
            not_searchable += 1
            if len(not_searchable_elem_ids) < 100:
                not_searchable_elem_ids.append(elem_id)

    print(f"\n  Results (sample of {n_sample:,} elements):")
    print(f"    Searchable in 3×3×3 neighborhood: {searchable_in_3x3x3:,} ({100*searchable_in_3x3x3/n_sample:.2f}%)")
    print(f"    NOT searchable in 3×3×3: {not_searchable:,} ({100*not_searchable/n_sample:.2f}%)")
    print(f"    Not in octree at all: {no_cells:,} ({100*no_cells/n_sample:.2f}%)")

    if not_searchable > 0:
        print(f"\n  ⚠️  WARNING: {100*not_searchable/n_sample:.2f}% of elements NOT in 3×3×3 neighborhood!")
        print(f"      Particles in these elements will be LOST!")
        print(f"      Sample element IDs: {not_searchable_elem_ids[:20]}")
    else:
        print(f"\n  ✅ SUCCESS: All sampled elements searchable in 3×3×3 neighborhood!")


def test_cross_cell_boundary_tracking(
    connectivity, node_positions, octree_multi: OctreeCellDataVertexMulti,
    element_to_cells
):
    """
    Test 3: Analyze cross-cell boundary face sharing.

    For elements sharing faces:
    - How many share faces across cell boundaries?
    - Are both elements in each other's 2×2×2 neighborhoods?
    """
    print("\n" + "="*80)
    print("Test 3: Cross-Cell Boundary Face Sharing")
    print("="*80)

    print("\n  Building element face map...")
    n_elements = connectivity.shape[0]

    # Build face map (sorted vertex tuple → element IDs)
    face_to_elements = defaultdict(list)

    for elem_id in range(n_elements):
        if elem_id % 500000 == 0 and elem_id > 0:
            print(f"    Processed {elem_id:,}/{n_elements:,}...")

        nodes = connectivity[elem_id]
        # 4 faces of tetrahedron
        faces = [
            tuple(sorted([nodes[0], nodes[1], nodes[2]])),
            tuple(sorted([nodes[0], nodes[1], nodes[3]])),
            tuple(sorted([nodes[0], nodes[2], nodes[3]])),
            tuple(sorted([nodes[1], nodes[2], nodes[3]])),
        ]

        for face in faces:
            face_to_elements[face].append(elem_id)

    print("\n  Analyzing face-sharing across cell boundaries...")

    same_cells = 0
    different_cells = 0
    one_missing = 0
    both_missing = 0

    cross_cell_not_in_neighborhood = 0
    cross_cell_sample_pairs = []

    for face, elem_ids in face_to_elements.items():
        if len(elem_ids) != 2:
            continue  # Boundary face or invalid

        elem_a, elem_b = elem_ids

        # Check if both in octree
        if elem_a not in element_to_cells and elem_b not in element_to_cells:
            both_missing += 1
            continue
        elif elem_a not in element_to_cells or elem_b not in element_to_cells:
            one_missing += 1
            continue

        cells_a = element_to_cells[elem_a]
        cells_b = element_to_cells[elem_b]

        # Check if they share any cells
        if cells_a & cells_b:
            same_cells += 1
        else:
            different_cells += 1

            # Check if they're in each other's 2×2×2 neighborhood
            # This requires spatial proximity check - simplified for now
            if len(cross_cell_sample_pairs) < 100:
                cross_cell_sample_pairs.append((elem_a, elem_b))

    total_interior_faces = same_cells + different_cells + one_missing

    print(f"\n  Results:")
    print(f"    Interior faces (2 elements): {total_interior_faces:,}")
    print(f"    Same cell: {same_cells:,} ({100*same_cells/total_interior_faces:.2f}%)")
    print(f"    Different cells: {different_cells:,} ({100*different_cells/total_interior_faces:.2f}%)")
    print(f"    One element missing: {one_missing:,} ({100*one_missing/total_interior_faces:.2f}%)")
    print(f"    Both missing: {both_missing:,}")

    if different_cells > 0:
        print(f"\n  ℹ️  INFO: {different_cells:,} face-sharing pairs in DIFFERENT cells")
        print(f"      When particle crosses these faces, it moves between cells!")
        print(f"      3×3×3 search should cover both cells to maintain tracking.")


def main():
    print("="*80)
    print("Multi-Cell Octree Coverage Diagnostics")
    print("="*80)
    print(f"Mesh: {MESH_BASE_PATH / MESH_FILE_PATTERN.format(timestep=158)}")
    print()

    # Load mesh (exact copy from benchmark)
    print("[1/5] Loading mesh...")
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

    # Extract multi-cell octree
    print("\n[2/5] Extracting multi-cell octree (vertex registration)...")
    t0 = time.time()
    octree_multi = extract_octree_cells_vertex_multi(
        node_positions, connectivity, tolerance=1e-6, verbose=True
    )
    print(f"  Extracted in {time.time()-t0:.1f}s")

    # Run tests
    print("\n[3/5] Running coverage tests...")
    element_to_cells, cells_per_element = test_multi_cell_registration_coverage(
        connectivity, node_positions, octree_multi
    )

    print("\n[4/5] Testing 3×3×3 search pattern...")
    test_3x3x3_local_search_pattern(
        connectivity, node_positions, octree_multi, element_to_cells
    )

    print("\n[5/5] Testing cross-cell boundary tracking...")
    test_cross_cell_boundary_tracking(
        connectivity, node_positions, octree_multi, element_to_cells
    )

    print("\n" + "="*80)
    print("Diagnostics Complete")
    print("="*80)


if __name__ == "__main__":
    main()
