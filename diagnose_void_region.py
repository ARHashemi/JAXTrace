#!/usr/bin/env python3
"""
Diagnose the specific tetrahedral void region identified by user.

Void characteristics:
- Right-angle triangle extruded along Y
- X: -0.0174 to -0.015 (range: 0.0024)
- Z: -0.0026 to 0.0 (range: 0.0026)
- Y: extended
- Appears after 10 timesteps

Investigation:
1. Find all elements in this spatial region
2. Check if they're in the octree
3. Check if they're Non-Kuhn
4. Test 3×3×3 search for sample positions in this region
5. Compare with radius search
6. Identify why 3×3×3 fails but radius succeeds
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
from pathlib import Path

# Import mesh loading
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

# Import octree
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import extract_octree_cells_vertex_multi
from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import find_axis_aligned_edges_single
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu

# Import search methods
from jaxtrace.gpu.search.mesh_aligned_point_location import search_mesh_aligned_octree_multi_local
from jaxtrace.gpu.search.morton_global_search import search_L2_global_morton_single, MeshGPUGlobalMorton

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
VOID_Y_MIN = -0.02  # Extended, approximate
VOID_Y_MAX = 0.02   # Extended, approximate


def main():
    print("="*80)
    print("Void Region Diagnostic")
    print("="*80)
    print(f"\nVoid region:")
    print(f"  X: [{VOID_X_MIN:.6f}, {VOID_X_MAX:.6f}]")
    print(f"  Y: [{VOID_Y_MIN:.6f}, {VOID_Y_MAX:.6f}] (approximate)")
    print(f"  Z: [{VOID_Z_MIN:.6f}, {VOID_Z_MAX:.6f}]")
    print()

    # Load mesh
    print("[1/6] Loading mesh...")
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

    # Find elements in void region
    print("\n[2/6] Finding elements in void region...")
    t0 = time.time()

    void_elements = []
    void_elements_info = []

    for elem_id in range(n_elements):
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]

        # Element bounding box
        elem_min = vertices.min(axis=0)
        elem_max = vertices.max(axis=0)

        # Check if element overlaps void region
        x_overlap = (elem_max[0] >= VOID_X_MIN) and (elem_min[0] <= VOID_X_MAX)
        y_overlap = (elem_max[1] >= VOID_Y_MIN) and (elem_min[1] <= VOID_Y_MAX)
        z_overlap = (elem_max[2] >= VOID_Z_MIN) and (elem_min[2] <= VOID_Z_MAX)

        if x_overlap and y_overlap and z_overlap:
            void_elements.append(elem_id)

            # Compute element properties
            centroid = vertices.mean(axis=0)
            volume = np.abs(np.linalg.det(np.column_stack([
                vertices[1] - vertices[0],
                vertices[2] - vertices[0],
                vertices[3] - vertices[0]
            ])) / 6.0)

            cell_size, level = find_axis_aligned_edges_single(vertices, tolerance=1e-6)
            is_non_kuhn = np.any(cell_size == 0)

            void_elements_info.append({
                'elem_id': elem_id,
                'centroid': centroid,
                'volume': volume,
                'level': level,
                'is_non_kuhn': is_non_kuhn,
                'bbox_min': elem_min,
                'bbox_max': elem_max
            })

        if (elem_id + 1) % 500000 == 0:
            print(f"    Checked {elem_id + 1:,}/{n_elements:,}...")

    n_void_elements = len(void_elements)
    print(f"  Found in {time.time()-t0:.1f}s")
    print(f"    Elements in void region: {n_void_elements:,}")

    if n_void_elements == 0:
        print("\n  No elements found in void region!")
        print("  Try adjusting VOID_* boundaries or check particle export coordinates.")
        return

    # Analyze void elements
    print(f"\n  Void element properties:")

    non_kuhn_count = sum(1 for info in void_elements_info if info['is_non_kuhn'])
    print(f"    Non-Kuhn: {non_kuhn_count}/{n_void_elements} ({100.0*non_kuhn_count/n_void_elements:.1f}%)")

    levels = [info['level'] for info in void_elements_info if not info['is_non_kuhn']]
    if levels:
        level_counts = {}
        for level in levels:
            level_counts[level] = level_counts.get(level, 0) + 1

        print(f"    Refinement levels:")
        for level in sorted(level_counts.keys()):
            count = level_counts[level]
            print(f"      Level {level:2d}: {count:4,} elements")

    volumes = np.array([info['volume'] for info in void_elements_info])
    print(f"    Volume: min={volumes.min():.3e}, max={volumes.max():.3e}, mean={volumes.mean():.3e}")

    # Extract octree
    print("\n[3/6] Extracting multi-cell octree...")
    t0 = time.time()
    octree_multi = extract_octree_cells_vertex_multi(
        node_positions, connectivity, tolerance=1e-6, verbose=False
    )
    print(f"  Extracted in {time.time()-t0:.1f}s")

    # Upload to GPU
    print("\n[4/6] Uploading to GPU...")
    t0 = time.time()
    octree_gpu = upload_mesh_aligned_octree_to_gpu(
        node_positions, connectivity, octree_multi, verbose=False
    )
    connectivity_gpu = jnp.array(connectivity, dtype=jnp.int32)
    node_positions_gpu = jnp.array(node_positions, dtype=jnp.float32)
    jax.block_until_ready(connectivity_gpu)
    print(f"  Uploaded in {time.time()-t0:.1f}s")

    # Build Morton structure for radius search (comparison)
    print("\n[5/6] Building Morton structure for comparison...")
    t0 = time.time()
    from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
    from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu

    octree_struct = build_global_morton_octree(
        node_positions=node_positions,
        connectivity=connectivity,
        leaf_capacity=256,
        max_depth=21,
        verbose=False
    )

    morton_gpu = upload_global_morton_to_gpu(
        octree_struct, connectivity, node_positions
    )

    jax.block_until_ready(morton_gpu.morton_sorted)
    print(f"  Built in {time.time()-t0:.1f}s")

    # Test searches
    print("\n[6/6] Testing searches for sample positions in void region...")

    # Generate sample positions in void region
    n_samples = 100
    np.random.seed(42)

    sample_positions = np.column_stack([
        np.random.uniform(VOID_X_MIN, VOID_X_MAX, n_samples),
        np.random.uniform(VOID_Y_MIN, VOID_Y_MAX, n_samples),
        np.random.uniform(VOID_Z_MIN, VOID_Z_MAX, n_samples)
    ]).astype(np.float32)

    sample_positions_gpu = jnp.array(sample_positions)

    # Test 3×3×3 search
    print(f"\n  Testing 3×3×3 search on {n_samples} sample positions...")
    t0 = time.time()

    @jax.jit
    def test_3x3x3(pos):
        elem_id, n_tests = search_mesh_aligned_octree_multi_local(
            pos, octree_gpu, max_tests=jnp.int32(600)
        )
        return elem_id, n_tests

    results_3x3x3 = jax.vmap(test_3x3x3)(sample_positions_gpu)
    results_3x3x3 = jax.block_until_ready(results_3x3x3)

    elem_ids_3x3x3, n_tests_3x3x3 = results_3x3x3
    n_found_3x3x3 = int(jnp.sum(elem_ids_3x3x3 >= 0))

    print(f"    3×3×3 search: {n_found_3x3x3}/{n_samples} found ({100.0*n_found_3x3x3/n_samples:.1f}%)")
    print(f"    Mean tests: {float(n_tests_3x3x3.mean()):.1f}")

    # Test radius search
    print(f"\n  Testing radius=15 search on {n_samples} sample positions...")

    @jax.jit
    def test_radius(pos):
        elem_id = search_L2_global_morton_single(pos, morton_gpu, search_radius=jnp.int32(15))
        return elem_id

    elem_ids_radius = jax.vmap(test_radius)(sample_positions_gpu)
    elem_ids_radius = jax.block_until_ready(elem_ids_radius)

    n_found_radius = int(jnp.sum(elem_ids_radius >= 0))

    print(f"    Radius=15 search: {n_found_radius}/{n_samples} found ({100.0*n_found_radius/n_samples:.1f}%)")

    # Analysis of differences
    print(f"\n  Comparison:")

    found_by_radius_not_3x3x3 = (elem_ids_radius >= 0) & (elem_ids_3x3x3 < 0)
    n_diff = int(jnp.sum(found_by_radius_not_3x3x3))

    print(f"    Found by radius but NOT by 3×3×3: {n_diff}/{n_samples} ({100.0*n_diff/n_samples:.1f}%)")

    if n_diff > 0:
        print(f"\n  Analyzing positions found by radius but not 3×3×3...")

        diff_indices = np.where(found_by_radius_not_3x3x3)[0][:10]  # First 10

        for idx in diff_indices:
            pos = sample_positions[idx]
            elem_id_radius = int(elem_ids_radius[idx])

            print(f"\n    Position [{idx}]: ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})")
            print(f"      Radius found: element {elem_id_radius}")

            # Get element properties
            elem_nodes = connectivity[elem_id_radius]
            elem_vertices = node_positions[elem_nodes]
            elem_centroid = elem_vertices.mean(axis=0)

            elem_cell_size, elem_level = find_axis_aligned_edges_single(elem_vertices, tolerance=1e-6)
            elem_is_non_kuhn = np.any(elem_cell_size == 0)

            print(f"      Element centroid: ({elem_centroid[0]:.6f}, {elem_centroid[1]:.6f}, {elem_centroid[2]:.6f})")
            print(f"      Element level: {elem_level}")
            print(f"      Element Non-Kuhn: {elem_is_non_kuhn}")

            # Check which cells this element is registered in
            elem_to_cells_offsets = octree_multi.element_to_cells_offsets
            elem_to_cells_data = octree_multi.element_to_cells_data

            start = elem_to_cells_offsets[elem_id_radius]
            end = elem_to_cells_offsets[elem_id_radius + 1]
            n_cells = end - start

            print(f"      Registered in {n_cells} cells")

            if n_cells < 4:
                print(f"      ⚠️  WARNING: Element registered in <4 cells (should be ~4)")

            # Compute which cell particle SHOULD be in
            if not elem_is_non_kuhn:
                particle_i = int(np.floor(pos[0] / elem_cell_size[0]))
                particle_j = int(np.floor(pos[1] / elem_cell_size[1]))
                particle_k = int(np.floor(pos[2] / elem_cell_size[2]))

                print(f"      Particle grid cell: ({particle_i}, {particle_j}, {particle_k})")

                # Get element's cells
                cell_indices = elem_to_cells_data[start:end]
                cell_grid_indices = octree_multi.cell_grid_indices[cell_indices]

                print(f"      Element registered in cells:")
                for cell_idx, grid_idx in zip(cell_indices, cell_grid_indices):
                    print(f"        Cell {cell_idx}: grid ({grid_idx[0]}, {grid_idx[1]}, {grid_idx[2]})")

                # Check if particle cell is in element's cells
                particle_in_element_cells = any(
                    np.array_equal([particle_i, particle_j, particle_k], grid_idx)
                    for grid_idx in cell_grid_indices
                )

                if not particle_in_element_cells:
                    print(f"      🚨 MISMATCH: Particle cell NOT in element's registered cells!")
                    print(f"         → This is why 3×3×3 failed!")

    print("\n" + "="*80)
    print("Void Region Diagnostic Complete")
    print("="*80)

    print("\nSummary:")
    print(f"  Elements in void region: {n_void_elements:,}")
    print(f"    Non-Kuhn: {non_kuhn_count} ({100.0*non_kuhn_count/n_void_elements:.1f}%)")
    print(f"\n  Search performance in void region:")
    print(f"    3×3×3: {n_found_3x3x3}/{n_samples} ({100.0*n_found_3x3x3/n_samples:.1f}%)")
    print(f"    Radius=15: {n_found_radius}/{n_samples} ({100.0*n_found_radius/n_samples:.1f}%)")
    print(f"    Difference: {n_diff} positions")

    if n_diff > 0:
        print(f"\n  🚨 FOUND THE PROBLEM!")
        print(f"     Radius search finds elements that 3×3×3 misses")
        print(f"     Check mismatch analysis above for root cause")


if __name__ == "__main__":
    main()
