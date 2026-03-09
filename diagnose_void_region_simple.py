#!/usr/bin/env python3
"""
Simplified void region diagnostic - test element coverage without GPU search.

This version analyzes the void region WITHOUT running GPU searches,
to avoid JIT/shape mismatch issues.
"""

import numpy as np
import time
from pathlib import Path

# Import mesh loading
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

# Import octree
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import extract_octree_cells_vertex_multi
from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import find_axis_aligned_edges_single

# Configuration
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME = 'Displacement'

# Void region boundaries (from user observation)
VOID_X_MIN = -0.0174
VOID_X_MAX = -0.015
VOID_Z_MIN = -0.0026
VOID_Z_MAX = 0.0
VOID_Y_MIN = -0.02
VOID_Y_MAX = 0.02


def point_in_tet_numpy(pos, vertices):
    """Simple numpy point-in-tet test."""
    v0, v1, v2, v3 = vertices

    # Edge vectors from v0
    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0
    vp = pos - v0

    # Compute volume
    V0 = np.dot(e1, np.cross(e2, e3))

    if abs(V0) < 1e-15:
        return False

    # Barycentric coordinates
    V1 = np.dot(vp, np.cross(e2, e3))
    V2 = np.dot(e1, np.cross(vp, e3))
    V3 = np.dot(e1, np.cross(e2, vp))

    lambda1 = V1 / V0
    lambda2 = V2 / V0
    lambda3 = V3 / V0
    lambda0 = 1.0 - lambda1 - lambda2 - lambda3

    tol = -1e-6
    return (lambda0 >= tol) and (lambda1 >= tol) and (lambda2 >= tol) and (lambda3 >= tol)


def main():
    print("="*80)
    print("Simplified Void Region Diagnostic")
    print("="*80)
    print(f"\nVoid region:")
    print(f"  X: [{VOID_X_MIN:.6f}, {VOID_X_MAX:.6f}]")
    print(f"  Z: [{VOID_Z_MIN:.6f}, {VOID_Z_MAX:.6f}]")
    print()

    # Load mesh
    print("[1/4] Loading mesh...")
    t0 = time.time()
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=False
    )

    n_elements = connectivity.shape[0]
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"    Elements: {n_elements:,}")

    print("  Deduplicating...")
    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    print(f"    Removed {n_duplicates_removed:,} duplicates")

    # Extract octree
    print("\n[2/4] Extracting multi-cell octree...")
    t0 = time.time()
    octree_multi = extract_octree_cells_vertex_multi(
        node_positions, connectivity, tolerance=1e-6, verbose=False
    )
    print(f"  Extracted in {time.time()-t0:.1f}s")

    # Find elements in void region
    print("\n[3/4] Finding elements in void region...")
    void_elements = []

    for elem_id in range(n_elements):
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]

        # Element bounding box
        elem_min = vertices.min(axis=0)
        elem_max = vertices.max(axis=0)

        # Check overlap
        x_overlap = (elem_max[0] >= VOID_X_MIN) and (elem_min[0] <= VOID_X_MAX)
        y_overlap = (elem_max[1] >= VOID_Y_MIN) and (elem_min[1] <= VOID_Y_MAX)
        z_overlap = (elem_max[2] >= VOID_Z_MIN) and (elem_min[2] <= VOID_Z_MAX)

        if x_overlap and y_overlap and z_overlap:
            cell_size, level = find_axis_aligned_edges_single(vertices, tolerance=1e-6)
            is_non_kuhn = np.any(cell_size == 0)

            # Get cells
            start = octree_multi.element_to_cells_offsets[elem_id]
            end = octree_multi.element_to_cells_offsets[elem_id + 1]
            n_cells = end - start

            void_elements.append({
                'elem_id': elem_id,
                'level': level,
                'is_non_kuhn': is_non_kuhn,
                'n_cells': n_cells,
                'vertices': vertices,
                'cell_size': cell_size
            })

    print(f"  Found {len(void_elements):,} elements in void region")

    # Sample and test
    print("\n[4/4] Testing sample positions...")
    np.random.seed(42)
    n_samples = 100

    sample_positions = np.column_stack([
        np.random.uniform(VOID_X_MIN, VOID_X_MAX, n_samples),
        np.random.uniform(VOID_Y_MIN, VOID_Y_MAX, n_samples),
        np.random.uniform(VOID_Z_MIN, VOID_Z_MAX, n_samples)
    ])

    # For each sample, check if any void element contains it
    found_count = 0
    found_by_non_kuhn = 0
    found_by_single_cell = 0

    for i, pos in enumerate(sample_positions):
        found = False
        found_elem = None

        for elem_info in void_elements:
            if point_in_tet_numpy(pos, elem_info['vertices']):
                found = True
                found_elem = elem_info
                break

        if found:
            found_count += 1
            if found_elem['is_non_kuhn']:
                found_by_non_kuhn += 1
            if found_elem['n_cells'] == 1:
                found_by_single_cell += 1

    print(f"\n  Results:")
    print(f"    Positions with containing element: {found_count}/{n_samples}")
    print(f"    Found in Non-Kuhn elements: {found_by_non_kuhn}")
    print(f"    Found in single-cell elements: {found_by_single_cell}")

    # Analyze void elements
    print(f"\n  Void element analysis:")
    non_kuhn_count = sum(1 for e in void_elements if e['is_non_kuhn'])
    single_cell_count = sum(1 for e in void_elements if e['n_cells'] == 1)

    print(f"    Non-Kuhn: {non_kuhn_count}/{len(void_elements)} ({100.0*non_kuhn_count/len(void_elements):.1f}%)")
    print(f"    Single-cell: {single_cell_count}/{len(void_elements)} ({100.0*single_cell_count/len(void_elements):.1f}%)")

    # Show sample mismatches
    print(f"\n  Checking cell registration for first 10 void elements...")
    for i, elem_info in enumerate(void_elements[:10]):
        elem_id = elem_info['elem_id']
        vertices = elem_info['vertices']
        cell_size = elem_info['cell_size']
        is_non_kuhn = elem_info['is_non_kuhn']

        # Get element's registered cells
        start = octree_multi.element_to_cells_offsets[elem_id]
        end = octree_multi.element_to_cells_offsets[elem_id + 1]
        cell_indices = octree_multi.element_to_cells_data[start:end]
        cell_grid_indices = octree_multi.cell_grid_indices[cell_indices]

        print(f"\n    Element {elem_id} ({'Non-Kuhn' if is_non_kuhn else f'Level {elem_info[\"level\"]}'})")
        print(f"      Registered in {len(cell_indices)} cells:")
        for cell_idx, grid_idx in zip(cell_indices, cell_grid_indices):
            print(f"        Cell {cell_idx}: grid ({grid_idx[0]}, {grid_idx[1]}, {grid_idx[2]})")

        # Compute where vertices map to
        if not is_non_kuhn:
            print(f"      Vertex grid positions:")
            for j, vertex in enumerate(vertices):
                vi = int(np.floor(vertex[0] / cell_size[0]))
                vj = int(np.floor(vertex[1] / cell_size[1]))
                vk = int(np.floor(vertex[2] / cell_size[2]))

                # Check if this matches any registered cell
                matches = any(np.array_equal([vi, vj, vk], grid_idx) for grid_idx in cell_grid_indices)
                marker = "✓" if matches else "✗"
                print(f"        Vertex {j}: ({vi}, {vj}, {vk}) {marker}")

    print("\n" + "="*80)
    print("Diagnostic Complete")
    print("="*80)


if __name__ == "__main__":
    main()
