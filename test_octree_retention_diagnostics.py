#!/usr/bin/env python3
"""
Octree Retention Diagnostics

Investigates WHY the mesh-aligned octree search loses ~11.4% of particles
over 100 RK4 steps. Specifically quantifies each failure mode:

1. Element boundary spanning: Tets whose vertices extend into neighbor cells
2. Non-standard cells: Cells with != 6 elements (incomplete Kuhn subdivision)
3. Uncovered elements: Non-Kuhn tets skipped during octree extraction
4. Cross-cell particle motion: Simulated small displacements that cross cell boundaries
5. Vertex sharing analysis: How many elements share vertices across cell boundaries

Key insight from benchmark:
- All 3 mesh-aligned methods (direct, morton r=2, neighbors) give IDENTICAL 88.59%
- This means the neighbor search is not helping during RK4 tracking
- The loss must come from elements NOT being registered in neighboring cells
"""

import time
import numpy as np
from pathlib import Path
from collections import defaultdict

from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import (
    extract_octree_cells_single,
    find_axis_aligned_edges_single,
    find_parent_cube,
)
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes


class RetentionDiagnostics:
    """Diagnose root causes of particle retention loss in mesh-aligned octree."""

    def __init__(self, node_positions, connectivity, octree_cells):
        self.node_positions = node_positions
        self.connectivity = connectivity
        self.octree = octree_cells
        self.n_elements = connectivity.shape[0]
        self.n_cells = octree_cells.n_cells

        # Build reverse lookup: cell_idx -> list of element_ids
        self.cell_to_elements = {}
        for cell_idx in range(self.n_cells):
            start = octree_cells.cell_to_elements_offsets[cell_idx]
            end = octree_cells.cell_to_elements_offsets[cell_idx + 1]
            self.cell_to_elements[cell_idx] = octree_cells.cell_to_elements_data[start:end].tolist()

        # Build spatial lookup: (level, i, j, k) -> cell_idx
        self.grid_to_cell = {}
        for cell_idx in range(self.n_cells):
            level = int(octree_cells.cell_levels[cell_idx])
            gi, gj, gk = octree_cells.cell_grid_indices[cell_idx]
            self.grid_to_cell[(level, int(gi), int(gj), int(gk))] = cell_idx

    def run_all_diagnostics(self):
        """Run all diagnostic tests."""
        print("\n" + "=" * 80)
        print("OCTREE RETENTION LOSS DIAGNOSTICS")
        print("=" * 80)
        print(f"Mesh: {self.n_elements:,} elements, {self.node_positions.shape[0]:,} nodes")
        print(f"Octree: {self.n_cells:,} cells")
        print()

        self.test_1_element_boundary_spanning()
        self.test_2_vertex_cell_distribution()
        self.test_3_non_standard_cells_detail()
        self.test_4_simulated_displacement_search()
        self.test_5_uncovered_elements_neighbors()
        self.test_6_cross_boundary_element_pairs()

        print("\n" + "=" * 80)
        print("DIAGNOSTICS COMPLETE")
        print("=" * 80)

    def test_1_element_boundary_spanning(self):
        """
        Test 1: How many elements have vertices in different cells?

        For each element, check if ALL 4 vertices map to the same cell as the centroid.
        If vertices are in different cells, the element spans cell boundaries.
        This is the primary suspected cause of retention loss.
        """
        print("\n" + "=" * 80)
        print("Test 1: Element Boundary Spanning Analysis")
        print("=" * 80)

        n_spanning = 0
        n_single_cell = 0
        n_skipped = 0
        spanning_details = []  # Store details for first few

        sample_size = min(self.n_elements, self.n_elements)  # Check all

        for elem_id in range(sample_size):
            cell_idx = self.octree.element_to_cells[elem_id]
            if cell_idx < 0:
                n_skipped += 1
                continue

            # Get element info
            elem_nodes = self.connectivity[elem_id]
            vertices = self.node_positions[elem_nodes]
            level = int(self.octree.cell_levels[cell_idx])
            cell_size = self.octree.cell_sizes[cell_idx]
            centroid_grid = self.octree.cell_grid_indices[cell_idx]

            # Check each vertex: which cell would it map to?
            vertex_cells = set()
            vertex_grid_indices = []
            for v_idx in range(4):
                vi = int(np.floor(vertices[v_idx, 0] / cell_size[0]))
                vj = int(np.floor(vertices[v_idx, 1] / cell_size[1]))
                vk = int(np.floor(vertices[v_idx, 2] / cell_size[2]))
                vertex_cells.add((vi, vj, vk))
                vertex_grid_indices.append((vi, vj, vk))

            if len(vertex_cells) > 1:
                n_spanning += 1

                # Check how many unique cells the vertices span
                if len(spanning_details) < 10:
                    spanning_details.append({
                        'elem_id': elem_id,
                        'level': level,
                        'centroid_grid': tuple(centroid_grid),
                        'n_unique_cells': len(vertex_cells),
                        'vertex_cells': list(vertex_cells),
                    })
            else:
                n_single_cell += 1

            if (elem_id + 1) % 500000 == 0:
                print(f"    Processed {elem_id + 1:,}/{sample_size:,}...")

        total_checked = n_spanning + n_single_cell
        spanning_pct = 100.0 * n_spanning / total_checked if total_checked > 0 else 0

        print(f"\n  Results:")
        print(f"    Elements checked: {total_checked:,}")
        print(f"    Skipped (non-Kuhn): {n_skipped:,}")
        print(f"    Single-cell elements: {n_single_cell:,} ({100.0 * n_single_cell / total_checked:.2f}%)")
        print(f"    Boundary-spanning elements: {n_spanning:,} ({spanning_pct:.2f}%)")

        if spanning_details:
            # Count how many cells each spanning element covers
            n_cells_distribution = defaultdict(int)
            for d in spanning_details:
                n_cells_distribution[d['n_unique_cells']] += 1

            print(f"\n  Sample boundary-spanning elements:")
            for d in spanning_details[:5]:
                print(f"    Element {d['elem_id']}: level={d['level']}, "
                      f"centroid_grid={d['centroid_grid']}, "
                      f"vertices span {d['n_unique_cells']} cells: {d['vertex_cells']}")

        # Count spanning distribution for ALL spanning elements
        print(f"\n  >>> THIS IS THE KEY METRIC <<<")
        print(f"  {spanning_pct:.2f}% of elements span cell boundaries")
        print(f"  When a particle crosses into the vertex-only region of a spanning element,")
        print(f"  the mesh-aligned search will NOT find it (element registered at centroid cell only)")

    def test_2_vertex_cell_distribution(self):
        """
        Test 2: For spanning elements, which neighbor cells contain their vertices?

        This tells us: if we search neighbor cells, would we find the element?
        The element is only registered in the centroid cell.
        A vertex in neighbor cell means: particle near that vertex won't find this element.
        """
        print("\n" + "=" * 80)
        print("Test 2: Vertex-to-Cell Distribution for Spanning Elements")
        print("=" * 80)

        # For each spanning element, check if the neighbor cells actually exist in octree
        # and whether the element is registered there
        n_vertex_in_existing_neighbor = 0
        n_vertex_in_missing_neighbor = 0
        n_spanning_analyzed = 0

        for elem_id in range(self.n_elements):
            cell_idx = self.octree.element_to_cells[elem_id]
            if cell_idx < 0:
                continue

            elem_nodes = self.connectivity[elem_id]
            vertices = self.node_positions[elem_nodes]
            level = int(self.octree.cell_levels[cell_idx])
            cell_size = self.octree.cell_sizes[cell_idx]
            centroid_grid = tuple(self.octree.cell_grid_indices[cell_idx])

            # Check each vertex
            vertex_grids = set()
            for v_idx in range(4):
                vi = int(np.floor(vertices[v_idx, 0] / cell_size[0]))
                vj = int(np.floor(vertices[v_idx, 1] / cell_size[1]))
                vk = int(np.floor(vertices[v_idx, 2] / cell_size[2]))
                vertex_grids.add((vi, vj, vk))

            if len(vertex_grids) <= 1:
                continue  # Not spanning

            n_spanning_analyzed += 1

            # Check neighbor cells (excluding centroid cell)
            for vg in vertex_grids:
                if vg == centroid_grid:
                    continue

                neighbor_key = (level, vg[0], vg[1], vg[2])
                if neighbor_key in self.grid_to_cell:
                    n_vertex_in_existing_neighbor += 1
                else:
                    n_vertex_in_missing_neighbor += 1

            if (elem_id + 1) % 500000 == 0:
                print(f"    Processed {elem_id + 1:,}/{self.n_elements:,}...")

        print(f"\n  Spanning elements analyzed: {n_spanning_analyzed:,}")
        print(f"  Vertex-neighbor cells that EXIST in octree: {n_vertex_in_existing_neighbor:,}")
        print(f"  Vertex-neighbor cells that DON'T exist: {n_vertex_in_missing_neighbor:,}")
        print(f"\n  >>> IMPLICATION <<<")
        print(f"  The element IS in the neighbor cell's spatial region but is NOT registered there.")
        print(f"  Searching neighbor cells only finds elements registered there (by centroid),")
        print(f"  NOT elements whose vertices happen to be in the neighbor cell.")

    def test_3_non_standard_cells_detail(self):
        """
        Test 3: Detailed analysis of non-standard cells (!=6 elements)

        For cells with fewer than 6 elements:
        - Are they at refinement boundaries?
        - What level are they at?
        - Do they have neighbor cells at different levels?
        """
        print("\n" + "=" * 80)
        print("Test 3: Non-Standard Cells Detailed Analysis")
        print("=" * 80)

        non_standard = defaultdict(list)  # elem_count -> list of cell_idx

        for cell_idx in range(self.n_cells):
            start = self.octree.cell_to_elements_offsets[cell_idx]
            end = self.octree.cell_to_elements_offsets[cell_idx + 1]
            n_elems = end - start

            if n_elems != 6:
                non_standard[n_elems].append(cell_idx)

        total_non_standard = sum(len(v) for v in non_standard.values())
        total_elements_in_non_standard = 0

        print(f"\n  Non-standard cells: {total_non_standard:,} / {self.n_cells:,} "
              f"({100.0 * total_non_standard / self.n_cells:.2f}%)")

        for n_elems in sorted(non_standard.keys()):
            cells = non_standard[n_elems]
            levels = [int(self.octree.cell_levels[c]) for c in cells]
            level_dist = defaultdict(int)
            for l in levels:
                level_dist[l] += 1

            total_elements_in_non_standard += n_elems * len(cells)

            print(f"\n  {n_elems} elements/cell: {len(cells):,} cells")
            print(f"    Level distribution: {dict(sorted(level_dist.items()))}")

            # Check if these cells are at refinement boundaries
            n_at_boundary = 0
            for cell_idx in cells[:min(1000, len(cells))]:
                level = int(self.octree.cell_levels[cell_idx])
                gi, gj, gk = self.octree.cell_grid_indices[cell_idx]

                # Check if any 26 neighbors are at a different level
                has_different_level_neighbor = False
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        for dk in [-1, 0, 1]:
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            # Check at same level
                            nkey = (level, int(gi) + di, int(gj) + dj, int(gk) + dk)
                            if nkey not in self.grid_to_cell:
                                # Missing neighbor could mean different level exists there
                                has_different_level_neighbor = True
                                break
                        if has_different_level_neighbor:
                            break
                    if has_different_level_neighbor:
                        break

                if has_different_level_neighbor:
                    n_at_boundary += 1

            checked = min(1000, len(cells))
            print(f"    At refinement boundary: {n_at_boundary}/{checked} "
                  f"({100.0 * n_at_boundary / checked:.1f}%)")

        print(f"\n  Total elements in non-standard cells: {total_elements_in_non_standard:,} "
              f"({100.0 * total_elements_in_non_standard / self.n_elements:.2f}% of mesh)")

    def test_4_simulated_displacement_search(self):
        """
        Test 4: Simulated small displacements from centroids

        For a sample of elements, perturb the centroid by small amounts and check
        if the mesh-aligned search would find the correct element.

        This simulates what happens during RK4 tracking when a particle moves
        slightly from its current position.
        """
        print("\n" + "=" * 80)
        print("Test 4: Simulated Displacement Search Test")
        print("=" * 80)

        np.random.seed(42)
        n_sample = 50000
        n_perturbations_per_element = 10

        # Displacement scales (fractions of cell size at level 14)
        # Level 14 cell size is approximately 7.8e-05
        displacement_scales = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

        # Select sample elements (only from covered elements)
        valid_elements = np.where(self.octree.element_to_cells >= 0)[0]
        sample_elements = np.random.choice(valid_elements, min(n_sample, len(valid_elements)), replace=False)

        print(f"  Sample elements: {len(sample_elements):,}")
        print(f"  Perturbations per element: {n_perturbations_per_element}")
        print(f"  Displacement scales (fraction of cell size): {displacement_scales}")

        # Compute reference cell size (level 14)
        level_14_mask = self.octree.cell_levels == 14
        if np.any(level_14_mask):
            ref_cell_size = self.octree.cell_sizes[level_14_mask][0]
        else:
            ref_cell_size = self.octree.cell_sizes[0]

        print(f"  Reference cell size (level 14): {ref_cell_size}")

        for scale in displacement_scales:
            displacement_mag = scale * np.mean(ref_cell_size)
            n_found_primary = 0
            n_found_neighbor = 0
            n_not_found = 0
            n_total = 0

            for elem_id in sample_elements:
                cell_idx = self.octree.element_to_cells[elem_id]
                if cell_idx < 0:
                    continue

                elem_nodes = self.connectivity[elem_id]
                vertices = self.node_positions[elem_nodes]
                centroid = vertices.mean(axis=0)
                level = int(self.octree.cell_levels[cell_idx])
                cell_size = self.octree.cell_sizes[cell_idx]
                centroid_grid = tuple(self.octree.cell_grid_indices[cell_idx])

                for _ in range(n_perturbations_per_element):
                    # Random displacement
                    direction = np.random.randn(3)
                    direction /= np.linalg.norm(direction)
                    displacement = direction * displacement_mag
                    query_pos = centroid + displacement

                    # Where does query position map to?
                    qi = int(np.floor(query_pos[0] / cell_size[0]))
                    qj = int(np.floor(query_pos[1] / cell_size[1]))
                    qk = int(np.floor(query_pos[2] / cell_size[2]))
                    query_grid = (qi, qj, qk)

                    n_total += 1

                    if query_grid == centroid_grid:
                        # Same cell as centroid -> would find element
                        n_found_primary += 1
                    else:
                        # Different cell -> check if element is registered in that cell
                        query_cell_key = (level, qi, qj, qk)
                        if query_cell_key in self.grid_to_cell:
                            query_cell_idx = self.grid_to_cell[query_cell_key]
                            # Is our element registered in that cell?
                            elements_in_query_cell = self.cell_to_elements.get(query_cell_idx, [])
                            if elem_id in elements_in_query_cell:
                                n_found_neighbor += 1
                            else:
                                n_not_found += 1
                        else:
                            n_not_found += 1

            found_pct = 100.0 * (n_found_primary + n_found_neighbor) / n_total if n_total > 0 else 0
            primary_pct = 100.0 * n_found_primary / n_total if n_total > 0 else 0
            neighbor_pct = 100.0 * n_found_neighbor / n_total if n_total > 0 else 0
            not_found_pct = 100.0 * n_not_found / n_total if n_total > 0 else 0

            print(f"\n  Scale={scale:.2f} (displacement={displacement_mag:.6e}):")
            print(f"    Total queries: {n_total:,}")
            print(f"    Found in primary cell: {n_found_primary:,} ({primary_pct:.2f}%)")
            print(f"    Found in neighbor cell: {n_found_neighbor:,} ({neighbor_pct:.2f}%)")
            print(f"    NOT found anywhere: {n_not_found:,} ({not_found_pct:.2f}%)")
            print(f"    Total searchable: {found_pct:.2f}%")

    def test_5_uncovered_elements_neighbors(self):
        """
        Test 5: What happens to particles in/near uncovered (non-Kuhn) elements?

        Analyze the spatial distribution of uncovered elements and their
        relationship to covered neighbors.
        """
        print("\n" + "=" * 80)
        print("Test 5: Uncovered Elements Analysis")
        print("=" * 80)

        uncovered_ids = np.where(self.octree.element_to_cells < 0)[0]
        n_uncovered = len(uncovered_ids)

        print(f"  Uncovered elements: {n_uncovered:,} ({100.0 * n_uncovered / self.n_elements:.4f}%)")

        if n_uncovered == 0:
            print("  No uncovered elements - skipping")
            return

        # Analyze uncovered elements
        # Check if they share faces/edges with covered elements
        n_share_node_with_covered = 0
        n_isolated = 0

        # Build node-to-element lookup (for uncovered elements' nodes)
        uncovered_nodes = set()
        for elem_id in uncovered_ids:
            for node_id in self.connectivity[elem_id]:
                uncovered_nodes.add(int(node_id))

        # Check if these nodes are shared with covered elements
        # (Sample for speed)
        sample_uncovered = uncovered_ids[:min(500, n_uncovered)]

        for elem_id in sample_uncovered:
            elem_nodes = set(self.connectivity[elem_id].tolist())
            shares_node = False

            # Check a neighborhood of elements
            for node_id in elem_nodes:
                # Check nearby elements (brute force for small sample)
                for other_elem in range(max(0, elem_id - 100), min(self.n_elements, elem_id + 100)):
                    if other_elem == elem_id:
                        continue
                    if self.octree.element_to_cells[other_elem] >= 0:
                        other_nodes = set(self.connectivity[other_elem].tolist())
                        if elem_nodes & other_nodes:
                            shares_node = True
                            break
                if shares_node:
                    break

            if shares_node:
                n_share_node_with_covered += 1
            else:
                n_isolated += 1

        checked = len(sample_uncovered)
        print(f"  Sample analyzed: {checked}")
        print(f"    Share nodes with covered elements: {n_share_node_with_covered} "
              f"({100.0 * n_share_node_with_covered / checked:.1f}%)")
        print(f"    Isolated: {n_isolated} ({100.0 * n_isolated / checked:.1f}%)")

        # Analyze why they're non-Kuhn
        print(f"\n  Why are they non-Kuhn? Analyzing edge structure...")
        n_degenerate = 0
        n_no_axis_aligned = 0
        n_other = 0

        for elem_id in sample_uncovered[:100]:
            elem_nodes = self.connectivity[elem_id]
            vertices = self.node_positions[elem_nodes]

            cell_size, level = find_axis_aligned_edges_single(vertices, tolerance=1e-6)
            if np.any(cell_size == 0):
                # Check if degenerate (zero volume)
                v0 = vertices[1] - vertices[0]
                v1 = vertices[2] - vertices[0]
                v2 = vertices[3] - vertices[0]
                det = np.abs(np.dot(v0, np.cross(v1, v2)))
                if det < 1e-15:
                    n_degenerate += 1
                else:
                    n_no_axis_aligned += 1
            else:
                n_other += 1

        checked_100 = min(100, len(sample_uncovered))
        print(f"  Detailed analysis of {checked_100} uncovered elements:")
        print(f"    Degenerate (zero volume): {n_degenerate}")
        print(f"    No axis-aligned edges: {n_no_axis_aligned}")
        print(f"    Other: {n_other}")

    def test_6_cross_boundary_element_pairs(self):
        """
        Test 6: Shared-face element pairs that are in DIFFERENT cells

        When an element shares a face with an element in a different cell,
        a particle crossing that face will need to find the new element
        in a different cell. This is the exact RK4 failure scenario.

        This test counts how many face-sharing element pairs span different cells.
        """
        print("\n" + "=" * 80)
        print("Test 6: Cross-Cell Face-Sharing Element Pairs")
        print("=" * 80)

        # Build face-to-element mapping
        # A face is defined by 3 sorted node indices
        face_to_elements = defaultdict(list)

        print("  Building face-to-element map...")
        t0 = time.time()

        for elem_id in range(self.n_elements):
            nodes = self.connectivity[elem_id]
            # 4 faces of a tetrahedron (each defined by 3 of 4 nodes)
            faces = [
                tuple(sorted([nodes[0], nodes[1], nodes[2]])),
                tuple(sorted([nodes[0], nodes[1], nodes[3]])),
                tuple(sorted([nodes[0], nodes[2], nodes[3]])),
                tuple(sorted([nodes[1], nodes[2], nodes[3]])),
            ]
            for face in faces:
                face_to_elements[face].append(elem_id)

            if (elem_id + 1) % 500000 == 0:
                print(f"    Processed {elem_id + 1:,}/{self.n_elements:,}...")

        t1 = time.time()
        print(f"  Face map built in {t1 - t0:.1f}s")

        # Find shared faces (interior faces have exactly 2 elements)
        n_interior_faces = 0
        n_same_cell = 0
        n_different_cell = 0
        n_one_uncovered = 0
        n_both_uncovered = 0
        n_different_level = 0

        different_cell_details = []

        for face, elements in face_to_elements.items():
            if len(elements) != 2:
                continue  # Boundary face or error

            n_interior_faces += 1
            elem_a, elem_b = elements

            cell_a = self.octree.element_to_cells[elem_a]
            cell_b = self.octree.element_to_cells[elem_b]

            if cell_a < 0 and cell_b < 0:
                n_both_uncovered += 1
                continue
            elif cell_a < 0 or cell_b < 0:
                n_one_uncovered += 1
                continue

            if cell_a == cell_b:
                n_same_cell += 1
            else:
                n_different_cell += 1

                # Check if different levels
                level_a = int(self.octree.cell_levels[cell_a])
                level_b = int(self.octree.cell_levels[cell_b])
                if level_a != level_b:
                    n_different_level += 1

                if len(different_cell_details) < 10:
                    different_cell_details.append({
                        'elem_a': elem_a, 'elem_b': elem_b,
                        'cell_a': cell_a, 'cell_b': cell_b,
                        'level_a': level_a, 'level_b': level_b,
                    })

        total_classified = n_same_cell + n_different_cell + n_one_uncovered + n_both_uncovered

        print(f"\n  Results:")
        print(f"    Total interior faces: {n_interior_faces:,}")
        print(f"    Same cell: {n_same_cell:,} ({100.0 * n_same_cell / n_interior_faces:.2f}%)")
        print(f"    Different cell: {n_different_cell:,} ({100.0 * n_different_cell / n_interior_faces:.2f}%)")
        print(f"      Of which, different level: {n_different_level:,}")
        print(f"    One element uncovered: {n_one_uncovered:,} ({100.0 * n_one_uncovered / n_interior_faces:.4f}%)")
        print(f"    Both elements uncovered: {n_both_uncovered:,}")

        print(f"\n  >>> KEY INSIGHT <<<")
        print(f"  {n_different_cell:,} interior faces connect elements in DIFFERENT cells.")
        print(f"  When a particle crosses any of these faces, the NEW element is NOT in the current cell.")
        print(f"  The L2 search must find it in a neighbor cell.")
        print(f"  If the new element's CENTROID is not in an adjacent cell (at same level),")
        print(f"  the neighbor search will fail.")

        if different_cell_details:
            print(f"\n  Sample cross-cell face pairs:")
            for d in different_cell_details[:5]:
                grid_a = tuple(self.octree.cell_grid_indices[d['cell_a']])
                grid_b = tuple(self.octree.cell_grid_indices[d['cell_b']])
                print(f"    Elements ({d['elem_a']}, {d['elem_b']}): "
                      f"cells ({d['cell_a']}, {d['cell_b']}), "
                      f"levels ({d['level_a']}, {d['level_b']}), "
                      f"grids ({grid_a}, {grid_b})")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Loading mesh...")
    print("=" * 80)

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

    # Extract octree
    print("\n" + "=" * 80)
    print("Extracting octree...")
    print("=" * 80)

    t0 = time.time()
    octree_cells = extract_octree_cells_single(
        node_positions, connectivity, verbose=True
    )
    t1 = time.time()

    print(f"\n  Octree extracted in {t1 - t0:.1f}s")
    print(f"    Cells: {octree_cells.n_cells:,}")
    print(f"    Elements/cell: {octree_cells.elements_per_cell_mean:.2f}")

    # Run diagnostics
    diag = RetentionDiagnostics(node_positions, connectivity, octree_cells)
    diag.run_all_diagnostics()
