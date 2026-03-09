#!/usr/bin/env python3
"""
Deep diagnostic: Why does mesh_aligned_octree_multi_local (3×3×3) miss elements?

Previous finding:
  - 2 positions found by radius but missed by 3×3×3
  - Both map to element 39551 which is Non-Kuhn, registered in only 1 cell

This diagnostic:
  1. Load mesh + build octree (same as benchmark)
  2. For EVERY Non-Kuhn element: show where it is vs where it's registered
  3. Generate sample positions inside Non-Kuhn elements specifically
  4. CPU-side replay of the 3×3×3 search to trace exactly what it looks at
  5. Identify the exact failure mechanism
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'

import numpy as np
import jax
import jax.numpy as jnp
import time
from pathlib import Path
from collections import defaultdict

# Import mesh loading (following benchmark pattern)
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

# Import octree
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import extract_octree_cells_vertex_multi
from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import (
    find_axis_aligned_edges_single,
    encode_morton_3d_single,
)
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu

# Import search methods
from jaxtrace.gpu.search.mesh_aligned_point_location import search_mesh_aligned_octree_multi_local

# Import point-in-tet setup
from jaxtrace.gpu.search.aa_detection import precompute_aa_metadata, precompute_element_vertices
from jaxtrace.gpu.search.point_in_tet_methods import set_corrected_metadata, set_inverse_matrices_gpu
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
import jaxtrace.config as config

# Configuration (matching benchmark)
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME = 'Displacement'


def point_in_tet_numpy(pos, vertices, tol=-1e-6):
    """CPU point-in-tet using barycentric coordinates."""
    v0, v1, v2, v3 = vertices
    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0
    vp = pos - v0

    V0 = np.dot(e1, np.cross(e2, e3))
    if abs(V0) < 1e-30:
        return False

    V1 = np.dot(vp, np.cross(e2, e3))
    V2 = np.dot(e1, np.cross(vp, e3))
    V3 = np.dot(e1, np.cross(e2, vp))

    l1 = V1 / V0
    l2 = V2 / V0
    l3 = V3 / V0
    l0 = 1.0 - l1 - l2 - l3

    return (l0 >= tol) and (l1 >= tol) and (l2 >= tol) and (l3 >= tol)


def main():
    print("=" * 80)
    print("3×3×3 Search Failure Deep Diagnostic")
    print("=" * 80)

    # ==================================================================
    # [1] Load mesh
    # ==================================================================
    print("\n[1/6] Loading mesh...")
    t0 = time.time()
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=False
    )
    n_elements = connectivity.shape[0]
    print(f"  Loaded in {time.time()-t0:.1f}s  ({n_elements:,} elements)")

    print("  Deduplicating...")
    node_positions, connectivity, n_dup, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    print(f"  Removed {n_dup:,} duplicates")

    # ==================================================================
    # [2] Precompute inverse matrices (needed for GPU search)
    # ==================================================================
    print("\n[2/6] Precomputing inverse point-in-tet metadata...")
    t0 = time.time()
    aa_metadata = precompute_aa_metadata(connectivity, node_positions, verbose=False)
    element_vertices_arr = precompute_element_vertices(connectivity, node_positions, verbose=False)
    M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)

    from jaxtrace.gpu.search.aa_detection import AxisAlignedMetadata
    aa_metadata_gpu = AxisAlignedMetadata(
        base_vertex_indices=jax.device_put(aa_metadata.base_vertex_indices),
        base_vertices=jax.device_put(aa_metadata.base_vertices),
        inv_edge_lengths=jax.device_put(aa_metadata.inv_edge_lengths),
        axis_indices=jax.device_put(aa_metadata.axis_indices),
        is_axis_aligned=jax.device_put(aa_metadata.is_axis_aligned)
    )
    element_vertices_gpu = jax.device_put(element_vertices_arr)
    M_inv_gpu = jax.device_put(M_inv_array)
    p0_gpu = jax.device_put(p0_array)

    set_corrected_metadata(aa_metadata_gpu, element_vertices_gpu)
    set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)
    config.POINT_IN_TET_METHOD = 'inverse'
    print(f"  Done in {time.time()-t0:.1f}s")

    # ==================================================================
    # [3] Extract octree + identify all Non-Kuhn elements
    # ==================================================================
    print("\n[3/6] Extracting multi-cell octree...")
    t0 = time.time()
    octree_multi = extract_octree_cells_vertex_multi(
        node_positions, connectivity, tolerance=1e-6, verbose=False
    )
    print(f"  Extracted in {time.time()-t0:.1f}s")
    print(f"    Cells: {octree_multi.n_cells:,}")
    print(f"    Elements per cell: {octree_multi.elements_per_cell_mean:.2f}")
    print(f"    Cells per element: {octree_multi.cells_per_element_mean:.2f}")

    # Identify Non-Kuhn elements
    print("\n  Scanning all elements for Non-Kuhn...")
    t0 = time.time()
    non_kuhn_elements = []
    kuhn_by_level = defaultdict(int)

    for elem_id in range(n_elements):
        vertices = node_positions[connectivity[elem_id]]
        cell_size, level = find_axis_aligned_edges_single(vertices, tolerance=1e-6)
        if np.any(cell_size == 0):
            non_kuhn_elements.append(elem_id)
        else:
            kuhn_by_level[level] += 1
        if (elem_id + 1) % 500000 == 0:
            print(f"    Scanned {elem_id+1:,}/{n_elements:,}...")

    print(f"  Scanned in {time.time()-t0:.1f}s")
    print(f"  Non-Kuhn elements: {len(non_kuhn_elements):,} / {n_elements:,} ({100.0*len(non_kuhn_elements)/n_elements:.3f}%)")
    print(f"  Kuhn elements by level:")
    for lvl in sorted(kuhn_by_level.keys()):
        print(f"    Level {lvl:2d}: {kuhn_by_level[lvl]:>10,}")

    # ==================================================================
    # [4] Analyze Non-Kuhn element registration
    # ==================================================================
    print("\n[4/6] Analyzing Non-Kuhn element cell registration...")

    # For each Non-Kuhn element: where is it registered vs where is it spatially?
    elem_to_cells_offsets = octree_multi.element_to_cells_offsets
    elem_to_cells_data = octree_multi.element_to_cells_data
    cell_levels_cpu = octree_multi.cell_levels
    cell_grid_indices_cpu = octree_multi.cell_grid_indices
    cell_sizes_cpu = octree_multi.cell_sizes

    n_registered_0 = 0
    n_registered_1 = 0
    n_registered_multi = 0
    misregistered_elements = []  # Elements whose centroid is NOT in any of their registered cells

    for elem_id in non_kuhn_elements:
        start = elem_to_cells_offsets[elem_id]
        end = elem_to_cells_offsets[elem_id + 1]
        n_cells = end - start

        if n_cells == 0:
            n_registered_0 += 1
            misregistered_elements.append((elem_id, 'unregistered', 0))
            continue
        elif n_cells == 1:
            n_registered_1 += 1
        else:
            n_registered_multi += 1

        # Get registered cell info
        cell_indices = elem_to_cells_data[start:end]
        reg_levels = cell_levels_cpu[cell_indices]
        reg_grid = cell_grid_indices_cpu[cell_indices]
        reg_sizes = cell_sizes_cpu[cell_indices]

        # Element spatial info
        vertices = node_positions[connectivity[elem_id]]
        centroid = vertices.mean(axis=0)
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)

        # Check: does ANY registered cell spatially overlap the element's bbox?
        has_overlap = False
        for ci in range(len(cell_indices)):
            cs = reg_sizes[ci]
            gi = reg_grid[ci]
            # Cell bbox
            cell_min = gi * cs  # approximate - grid_index * cell_size
            cell_max = cell_min + cs

            # Check overlap
            if np.all(cell_max >= bbox_min) and np.all(cell_min <= bbox_max):
                has_overlap = True
                break

        # Also check: would 3×3×3 from centroid reach any registered cell?
        # i.e., is the registered cell within ±1 of centroid's grid cell at any level?
        centroid_reachable = False
        for ci in range(len(cell_indices)):
            cs = reg_sizes[ci]
            lvl = reg_levels[ci]
            gi = reg_grid[ci]

            if np.any(cs == 0) or np.any(cs < 1e-15):
                continue

            # Centroid grid cell at this level
            ci_grid = np.floor(centroid / cs).astype(int)

            # Distance in grid cells
            grid_diff = np.abs(gi - ci_grid)

            if np.all(grid_diff <= 1):
                centroid_reachable = True
                break

        if not centroid_reachable:
            misregistered_elements.append((elem_id, 'centroid_unreachable', n_cells))

    print(f"\n  Non-Kuhn registration summary:")
    print(f"    Registered in 0 cells: {n_registered_0}")
    print(f"    Registered in 1 cell:  {n_registered_1}")
    print(f"    Registered in 2+ cells: {n_registered_multi}")
    print(f"    Misregistered (centroid unreachable by 3×3×3): {len(misregistered_elements)}")

    # Show details for misregistered elements (up to 20)
    if misregistered_elements:
        print(f"\n  Misregistered Non-Kuhn elements (first 20):")
        for elem_id, reason, n_cells in misregistered_elements[:20]:
            vertices = node_positions[connectivity[elem_id]]
            centroid = vertices.mean(axis=0)
            bbox_min = vertices.min(axis=0)
            bbox_max = vertices.max(axis=0)

            start = elem_to_cells_offsets[elem_id]
            end = elem_to_cells_offsets[elem_id + 1]
            cell_indices = elem_to_cells_data[start:end]

            print(f"\n    Element {elem_id} ({reason}, {n_cells} cells)")
            print(f"      Centroid: ({centroid[0]:.7f}, {centroid[1]:.7f}, {centroid[2]:.7f})")
            print(f"      BBox: ({bbox_min[0]:.7f},{bbox_min[1]:.7f},{bbox_min[2]:.7f})"
                  f" → ({bbox_max[0]:.7f},{bbox_max[1]:.7f},{bbox_max[2]:.7f})")

            if n_cells > 0:
                for ci in range(min(len(cell_indices), 4)):
                    idx = cell_indices[ci]
                    cs = cell_sizes_cpu[idx]
                    gi = cell_grid_indices_cpu[idx]
                    lvl = cell_levels_cpu[idx]
                    cell_min = gi * cs
                    cell_max = cell_min + cs
                    print(f"      Registered cell {ci}: level={lvl}, grid=({gi[0]},{gi[1]},{gi[2]}), "
                          f"size=({cs[0]:.7f},{cs[1]:.7f},{cs[2]:.7f})")
                    print(f"        Cell bbox: ({cell_min[0]:.7f},{cell_min[1]:.7f},{cell_min[2]:.7f})"
                          f" → ({cell_max[0]:.7f},{cell_max[1]:.7f},{cell_max[2]:.7f})")

                    # Where would centroid map at this level?
                    if np.all(cs > 1e-15):
                        centroid_grid = np.floor(centroid / cs).astype(int)
                        grid_diff = gi - centroid_grid
                        print(f"        Centroid grid at this level: ({centroid_grid[0]},{centroid_grid[1]},{centroid_grid[2]})"
                              f"  diff=({grid_diff[0]},{grid_diff[1]},{grid_diff[2]})")

    # ==================================================================
    # [5] Generate test points INSIDE Non-Kuhn elements
    # ==================================================================
    print("\n\n[5/6] Generating test points inside Non-Kuhn elements...")

    # For each Non-Kuhn element, generate random interior points
    test_positions = []
    test_elem_ids = []
    test_is_misregistered = []
    misreg_set = set(e[0] for e in misregistered_elements)

    np.random.seed(42)
    points_per_element = 5

    for elem_id in non_kuhn_elements:
        vertices = node_positions[connectivity[elem_id]]

        # Generate random points inside tetrahedron using barycentric sampling
        for _ in range(points_per_element):
            # Random barycentric coordinates (uniform inside tet)
            r = np.random.random(4)
            r = -np.log(r + 1e-30)  # Exponential
            r /= r.sum()
            pos = (r[:, None] * vertices).sum(axis=0)

            # Verify point is actually inside
            if point_in_tet_numpy(pos, vertices):
                test_positions.append(pos.astype(np.float32))
                test_elem_ids.append(elem_id)
                test_is_misregistered.append(elem_id in misreg_set)

    n_test = len(test_positions)
    n_misreg_test = sum(test_is_misregistered)
    print(f"  Generated {n_test} test points inside {len(non_kuhn_elements)} Non-Kuhn elements")
    print(f"  Of these, {n_misreg_test} are inside misregistered elements")

    if n_test == 0:
        print("  No valid test points generated. Exiting.")
        return

    test_positions_np = np.array(test_positions)

    # ==================================================================
    # [6] Run 3×3×3 GPU search on Non-Kuhn test points
    # ==================================================================
    print("\n[6/6] Running 3×3×3 search on Non-Kuhn element test points...")

    # Upload octree to GPU
    octree_gpu = upload_mesh_aligned_octree_to_gpu(
        connectivity, node_positions, octree_multi, verbose=False
    )

    @jax.jit
    def search_3x3x3(pos):
        elem_id, n_tests = search_mesh_aligned_octree_multi_local(
            pos, octree_gpu, max_tests=jnp.int32(600)
        )
        return elem_id, n_tests

    # Run search sequentially to avoid OOM
    found_ids = []
    n_tests_list = []

    t0 = time.time()
    for i in range(n_test):
        pos_gpu = jnp.array(test_positions_np[i])
        eid, nt = search_3x3x3(pos_gpu)
        found_ids.append(int(jax.block_until_ready(eid)))
        n_tests_list.append(int(jax.block_until_ready(nt)))
        if (i + 1) % 1000 == 0:
            print(f"    Searched {i+1}/{n_test}...")

    print(f"  Search done in {time.time()-t0:.1f}s")

    found_ids = np.array(found_ids)
    n_tests_arr = np.array(n_tests_list)
    test_elem_ids_np = np.array(test_elem_ids)
    test_misreg_np = np.array(test_is_misregistered)

    # Results
    n_found = int((found_ids >= 0).sum())
    n_missed = n_test - n_found
    n_found_correct = int(((found_ids >= 0) & (found_ids == test_elem_ids_np)).sum())
    n_found_wrong = int(((found_ids >= 0) & (found_ids != test_elem_ids_np)).sum())

    print(f"\n  === RESULTS: 3×3×3 on Non-Kuhn element interiors ===")
    print(f"  Total test points:  {n_test}")
    print(f"  Found (any elem):   {n_found} ({100.0*n_found/n_test:.1f}%)")
    print(f"  Missed:             {n_missed} ({100.0*n_missed/n_test:.1f}%)")
    print(f"  Found correct elem: {n_found_correct}")
    print(f"  Found wrong elem:   {n_found_wrong}")

    # Break down by misregistered vs properly registered
    if n_misreg_test > 0:
        misreg_found = int((found_ids[test_misreg_np] >= 0).sum())
        misreg_missed = n_misreg_test - misreg_found
        print(f"\n  Misregistered Non-Kuhn elements:")
        print(f"    Test points:  {n_misreg_test}")
        print(f"    Found:        {misreg_found} ({100.0*misreg_found/n_misreg_test:.1f}%)")
        print(f"    Missed:       {misreg_missed} ({100.0*misreg_missed/n_misreg_test:.1f}%)")

    proper_mask = ~test_misreg_np
    n_proper_test = int(proper_mask.sum())
    if n_proper_test > 0:
        proper_found = int((found_ids[proper_mask] >= 0).sum())
        proper_missed = n_proper_test - proper_found
        print(f"\n  Properly registered Non-Kuhn elements:")
        print(f"    Test points:  {n_proper_test}")
        print(f"    Found:        {proper_found} ({100.0*proper_found/n_proper_test:.1f}%)")
        print(f"    Missed:       {proper_missed} ({100.0*proper_missed/n_proper_test:.1f}%)")

    # ==================================================================
    # Detail analysis of missed points
    # ==================================================================
    missed_mask = found_ids < 0
    missed_indices = np.where(missed_mask)[0]

    if len(missed_indices) > 0:
        print(f"\n  === DETAILED ANALYSIS OF MISSED POINTS (first 10) ===")

        for rank, idx in enumerate(missed_indices[:10]):
            pos = test_positions_np[idx]
            true_elem = test_elem_ids_np[idx]
            is_misreg = test_misreg_np[idx]

            vertices = node_positions[connectivity[true_elem]]
            centroid = vertices.mean(axis=0)

            print(f"\n  --- Missed point {rank+1} (index {idx}) ---")
            print(f"  Position:    ({pos[0]:.8f}, {pos[1]:.8f}, {pos[2]:.8f})")
            print(f"  True elem:   {true_elem} (misregistered={is_misreg})")
            print(f"  Elem centroid: ({centroid[0]:.8f}, {centroid[1]:.8f}, {centroid[2]:.8f})")

            # Element registration
            start = elem_to_cells_offsets[true_elem]
            end = elem_to_cells_offsets[true_elem + 1]
            n_cells = end - start
            cell_indices = elem_to_cells_data[start:end]

            print(f"  Registered in {n_cells} cell(s):")
            for ci in range(n_cells):
                c_idx = cell_indices[ci]
                cs = cell_sizes_cpu[c_idx]
                gi = cell_grid_indices_cpu[c_idx]
                lvl = cell_levels_cpu[c_idx]
                print(f"    Cell {c_idx}: level={lvl}, grid=({gi[0]},{gi[1]},{gi[2]}), "
                      f"size=({cs[0]:.8f},{cs[1]:.8f},{cs[2]:.8f})")

            # CPU replay of 3×3×3 search
            print(f"\n  CPU REPLAY of 3×3×3 search:")
            level_cell_sizes = {}
            for lvl in range(15):
                mask = cell_levels_cpu == lvl
                if mask.any():
                    level_cell_sizes[lvl] = cell_sizes_cpu[mask][0]

            # Build morton lookup dict for CPU replay
            # cell_key = (morton, level) -> cell_idx
            cell_morton_codes_cpu = octree_multi.cell_morton_codes
            cell_lookup = {}
            for c_idx in range(octree_multi.n_cells):
                key = (int(cell_morton_codes_cpu[c_idx]), int(cell_levels_cpu[c_idx]))
                cell_lookup[key] = c_idx

            offset = (1 << 19)
            max_coord = (1 << 20)

            cells_searched = 0
            elements_tested = 0
            found_in_replay = False

            for level_idx in range(8):  # levels 14 → 7
                level = 14 - level_idx
                if level not in level_cell_sizes:
                    continue

                cs = level_cell_sizes[level]
                if np.any(cs < 1e-15):
                    continue

                i_base = int(np.floor(pos[0] / cs[0]))
                j_base = int(np.floor(pos[1] / cs[1]))
                k_base = int(np.floor(pos[2] / cs[2]))

                level_found = False
                level_cells = 0
                level_elems = 0

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        for dk in [-1, 0, 1]:
                            i = i_base + di
                            j = j_base + dj
                            k = k_base + dk

                            i_m = max(0, min(i + offset, max_coord - 1))
                            j_m = max(0, min(j + offset, max_coord - 1))
                            k_m = max(0, min(k + offset, max_coord - 1))

                            morton = encode_morton_3d_single(i_m, j_m, k_m, max_depth=21)
                            key = (morton, level)

                            if key in cell_lookup:
                                c_idx = cell_lookup[key]
                                c_start = octree_multi.cell_to_elements_offsets[c_idx]
                                c_end = octree_multi.cell_to_elements_offsets[c_idx + 1]
                                n_elems = c_end - c_start
                                cells_searched += 1
                                level_cells += 1

                                for ei in range(n_elems):
                                    e_id = octree_multi.cell_to_elements_data[c_start + ei]
                                    elements_tested += 1
                                    level_elems += 1

                                    v = node_positions[connectivity[e_id]]
                                    if point_in_tet_numpy(pos.astype(np.float64), v):
                                        found_in_replay = True
                                        print(f"    Level {level}: FOUND in elem {e_id} "
                                              f"(cell grid=({i},{j},{k}), offset=({di},{dj},{dk}))")
                                        break

                                    # Check if true_elem is in this cell
                                    if e_id == true_elem:
                                        print(f"    Level {level}: true elem {true_elem} IS in cell ({i},{j},{k}) "
                                              f"but point_in_tet FAILED!")

                            if found_in_replay:
                                break
                        if found_in_replay:
                            break
                    if found_in_replay:
                        break

                if level_cells > 0 and not found_in_replay:
                    # Check if the true element's registered cell was among the searched ones
                    for ci in range(n_cells):
                        c_idx = cell_indices[ci]
                        c_lvl = int(cell_levels_cpu[c_idx])
                        if c_lvl == level:
                            c_gi = cell_grid_indices_cpu[c_idx]
                            in_range = (abs(c_gi[0] - i_base) <= 1 and
                                       abs(c_gi[1] - j_base) <= 1 and
                                       abs(c_gi[2] - k_base) <= 1)
                            print(f"    Level {level}: searched {level_cells} cells, {level_elems} elems. "
                                  f"True elem's cell ({c_gi[0]},{c_gi[1]},{c_gi[2]}) "
                                  f"{'IN' if in_range else 'NOT IN'} 3×3×3 around ({i_base},{j_base},{k_base})")

                if found_in_replay:
                    break

            if not found_in_replay:
                print(f"    NOT FOUND after searching {cells_searched} cells, {elements_tested} elements")
                print(f"    → The true element ({true_elem}) is NOT reachable by 3×3×3 at any level!")

                # Show where it IS registered
                for ci in range(n_cells):
                    c_idx = cell_indices[ci]
                    c_lvl = int(cell_levels_cpu[c_idx])
                    c_gi = cell_grid_indices_cpu[c_idx]
                    cs = cell_sizes_cpu[c_idx]

                    if np.any(cs < 1e-15):
                        print(f"    Registered cell: level={c_lvl}, grid=({c_gi[0]},{c_gi[1]},{c_gi[2]}), "
                              f"size=ZERO!")
                        continue

                    # Where would the particle map at this level?
                    pi = int(np.floor(pos[0] / cs[0]))
                    pj = int(np.floor(pos[1] / cs[1]))
                    pk = int(np.floor(pos[2] / cs[2]))

                    dist = np.array([abs(c_gi[0] - pi), abs(c_gi[1] - pj), abs(c_gi[2] - pk)])
                    print(f"    Registered cell: level={c_lvl}, grid=({c_gi[0]},{c_gi[1]},{c_gi[2]})")
                    print(f"      Particle grid at level {c_lvl}: ({pi},{pj},{pk})")
                    print(f"      Grid distance: ({dist[0]},{dist[1]},{dist[2]}) → max={dist.max()}")
                    if dist.max() > 1:
                        print(f"      ⚠ DISTANCE > 1: 3×3×3 cannot reach this cell!")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Non-Kuhn elements: {len(non_kuhn_elements):,}")
    print(f"  Misregistered (unreachable by 3×3×3): {len(misregistered_elements)}")
    print(f"  Test points inside Non-Kuhn elements: {n_test}")
    print(f"  3×3×3 missed: {n_missed} ({100.0*n_missed/n_test:.1f}%)")
    if n_missed > 0:
        print(f"\n  ROOT CAUSE: Non-Kuhn elements registered in face neighbor's cells,")
        print(f"  which can be spatially far from the element's actual location.")
        print(f"  3×3×3 only searches ±1 cell neighborhood → misses these elements.")


if __name__ == "__main__":
    main()
