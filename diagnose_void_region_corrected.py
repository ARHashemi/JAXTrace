#!/usr/bin/env python3
"""
Deep diagnostic: Why does mesh_aligned_octree_multi_local (3×3×3) miss elements?

Previous finding (extended region diagnostic):
  - 2 out of 8000 positions found by radius but missed by 3×3×3
  - Both map to element 39551 which is Non-Kuhn, registered in only 1 cell
  - The registered cell is spatially far from the element's actual position

This diagnostic focuses exclusively on 3×3×3 search failures:
  1. Load mesh + build octree (same pattern as benchmark)
  2. Identify ALL Non-Kuhn elements globally + analyze their cell registration
  3. Generate test points INSIDE Non-Kuhn elements
  4. Run 3×3×3 GPU search on those points
  5. For failures: CPU-side replay to trace exactly what the search looked at
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

# Import octree builders (following benchmark pattern)
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import extract_octree_cells_vertex_multi
from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import (
    find_axis_aligned_edges_single,
    encode_morton_3d_single,
)
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu

# Import search methods
from jaxtrace.gpu.search.mesh_aligned_point_location import search_mesh_aligned_octree_multi_local

# Import point-in-tet setup (following benchmark lines 66-67, 808-809)
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
    """CPU point-in-tet using barycentric coordinates (double precision)."""
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
    l1, l2, l3 = V1/V0, V2/V0, V3/V0
    l0 = 1.0 - l1 - l2 - l3
    return (l0 >= tol) and (l1 >= tol) and (l2 >= tol) and (l3 >= tol)


def main():
    print("=" * 80)
    print("3x3x3 Search Failure Deep Diagnostic")
    print("  Focus: Non-Kuhn element cell registration mismatch")
    print("=" * 80)

    # ========================================================================
    # [1] Load Mesh (EXACT pattern from benchmark lines 660-676)
    # ========================================================================
    print("\n[1/7] Loading mesh...")
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
    print(f"    Elements: {n_elements:,}, Nodes: {node_positions.shape[0]:,}")

    # ========================================================================
    # [2] Deduplicate (EXACT pattern from benchmark lines 678-683)
    # ========================================================================
    print("\n[2/7] Deduplicating...")
    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    print(f"  Removed {n_duplicates_removed:,} duplicates")

    # ========================================================================
    # [2b] Precompute metadata for inverse method (EXACT pattern from benchmark)
    # ========================================================================
    print("\n[2b/7] Precomputing metadata for inverse point-in-tet...")
    t0 = time.time()
    aa_metadata = precompute_aa_metadata(connectivity, node_positions, verbose=False)
    element_vertices_precomp = precompute_element_vertices(connectivity, node_positions, verbose=False)
    M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)

    from jaxtrace.gpu.search.aa_detection import AxisAlignedMetadata
    aa_metadata_gpu = AxisAlignedMetadata(
        base_vertex_indices=jax.device_put(aa_metadata.base_vertex_indices),
        base_vertices=jax.device_put(aa_metadata.base_vertices),
        inv_edge_lengths=jax.device_put(aa_metadata.inv_edge_lengths),
        axis_indices=jax.device_put(aa_metadata.axis_indices),
        is_axis_aligned=jax.device_put(aa_metadata.is_axis_aligned)
    )
    element_vertices_gpu = jax.device_put(element_vertices_precomp)
    M_inv_gpu = jax.device_put(M_inv_array)
    p0_gpu = jax.device_put(p0_array)

    set_corrected_metadata(aa_metadata_gpu, element_vertices_gpu)
    set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)
    config.POINT_IN_TET_METHOD = 'inverse'
    print(f"  Metadata ready in {time.time()-t0:.1f}s")

    # ========================================================================
    # [3] Extract octree (EXACT pattern from benchmark lines 726-734)
    # ========================================================================
    print("\n[3/7] Extracting multi-cell octree...")
    t0 = time.time()
    octree_multi = extract_octree_cells_vertex_multi(
        node_positions, connectivity, tolerance=1e-6, verbose=False
    )
    print(f"  Extracted in {time.time()-t0:.1f}s")
    print(f"    Cells: {octree_multi.n_cells:,}")
    print(f"    Elements per cell: {octree_multi.elements_per_cell_mean:.2f}")
    print(f"    Cells per element: {octree_multi.cells_per_element_mean:.2f}")

    # ========================================================================
    # [4] Upload to GPU (correct signature: connectivity, node_positions, octree_cells)
    # ========================================================================
    print("\n[4/7] Uploading octree to GPU...")
    t0 = time.time()
    octree_gpu = upload_mesh_aligned_octree_to_gpu(
        connectivity, node_positions, octree_multi, verbose=False
    )
    print(f"  Uploaded in {time.time()-t0:.1f}s")

    # ========================================================================
    # [5] Identify ALL Non-Kuhn elements + analyze registration
    # ========================================================================
    print("\n[5/7] Scanning all elements for Non-Kuhn + analyzing cell registration...")
    t0 = time.time()

    elem_to_cells_offsets = octree_multi.element_to_cells_offsets
    elem_to_cells_data = octree_multi.element_to_cells_data
    cell_levels_cpu = octree_multi.cell_levels
    cell_grid_indices_cpu = octree_multi.cell_grid_indices
    cell_sizes_cpu = octree_multi.cell_sizes
    cell_morton_codes_cpu = octree_multi.cell_morton_codes

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
    print(f"  Non-Kuhn elements: {len(non_kuhn_elements):,} / {n_elements:,}"
          f" ({100.0*len(non_kuhn_elements)/n_elements:.3f}%)")
    print(f"  Kuhn elements by level:")
    for lvl in sorted(kuhn_by_level.keys()):
        print(f"    Level {lvl:2d}: {kuhn_by_level[lvl]:>10,}")

    # Analyze each Non-Kuhn element's registration
    n_registered_0 = 0
    n_registered_1 = 0
    n_registered_multi = 0
    misregistered = []

    for elem_id in non_kuhn_elements:
        start = elem_to_cells_offsets[elem_id]
        end = elem_to_cells_offsets[elem_id + 1]
        n_cells = end - start

        if n_cells == 0:
            n_registered_0 += 1
            misregistered.append((elem_id, 'unregistered'))
            continue
        elif n_cells == 1:
            n_registered_1 += 1
        else:
            n_registered_multi += 1

        # Check: would the 3×3×3 from centroid reach the registered cell?
        vertices = node_positions[connectivity[elem_id]]
        centroid = vertices.mean(axis=0)

        cell_indices = elem_to_cells_data[start:end]
        centroid_reachable = False
        for ci in cell_indices:
            cs = cell_sizes_cpu[ci]
            gi = cell_grid_indices_cpu[ci]
            if np.any(cs < 1e-15):
                continue
            centroid_grid = np.floor(centroid / cs).astype(int)
            grid_diff = np.abs(gi - centroid_grid)
            if np.all(grid_diff <= 1):
                centroid_reachable = True
                break

        if not centroid_reachable:
            misregistered.append((elem_id, 'unreachable'))

    print(f"\n  Non-Kuhn registration summary:")
    print(f"    Registered in 0 cells: {n_registered_0}")
    print(f"    Registered in 1 cell:  {n_registered_1}")
    print(f"    Registered in 2+ cells: {n_registered_multi}")
    print(f"    Misregistered (unreachable by 3x3x3 from centroid): {len(misregistered)}")

    # Show details for first 20 misregistered elements
    if misregistered:
        print(f"\n  Details of misregistered elements (first 20):")
        for elem_id, reason in misregistered[:20]:
            vertices = node_positions[connectivity[elem_id]]
            centroid = vertices.mean(axis=0)
            start = elem_to_cells_offsets[elem_id]
            end = elem_to_cells_offsets[elem_id + 1]
            n_cells = end - start
            cell_indices = elem_to_cells_data[start:end]

            print(f"\n    Element {elem_id} ({reason}, {n_cells} cells)")
            print(f"      Centroid: ({centroid[0]:.8f}, {centroid[1]:.8f}, {centroid[2]:.8f})")
            print(f"      BBox: ({vertices.min(0)[0]:.7f},{vertices.min(0)[1]:.7f},{vertices.min(0)[2]:.7f})"
                  f" -> ({vertices.max(0)[0]:.7f},{vertices.max(0)[1]:.7f},{vertices.max(0)[2]:.7f})")

            for ci_idx in range(min(n_cells, 4)):
                ci = cell_indices[ci_idx]
                cs = cell_sizes_cpu[ci]
                gi = cell_grid_indices_cpu[ci]
                lvl = cell_levels_cpu[ci]
                print(f"      Cell {ci}: level={lvl}, grid=({gi[0]},{gi[1]},{gi[2]}), "
                      f"size=({cs[0]:.8f},{cs[1]:.8f},{cs[2]:.8f})")
                if np.all(cs > 1e-15):
                    centroid_grid = np.floor(centroid / cs).astype(int)
                    diff = gi - centroid_grid
                    print(f"        Centroid grid at level {lvl}: ({centroid_grid[0]},{centroid_grid[1]},{centroid_grid[2]})"
                          f"  diff=({diff[0]},{diff[1]},{diff[2]})")

    # ========================================================================
    # [6] Generate test points INSIDE Non-Kuhn elements + run 3×3×3 search
    # ========================================================================
    print("\n\n[6/7] Generating test points inside Non-Kuhn elements and running 3x3x3 search...")

    np.random.seed(42)
    POINTS_PER_ELEMENT = 5

    test_positions = []
    test_true_elem = []
    test_is_misreg = []
    misreg_set = set(e[0] for e in misregistered)

    for elem_id in non_kuhn_elements:
        vertices = node_positions[connectivity[elem_id]]
        for _ in range(POINTS_PER_ELEMENT):
            # Uniform random inside tetrahedron via exponential barycentric coords
            r = np.random.random(4)
            r = -np.log(r + 1e-30)
            r /= r.sum()
            pos = (r[:, None] * vertices).sum(axis=0)
            if point_in_tet_numpy(pos.astype(np.float64), vertices):
                test_positions.append(pos.astype(np.float32))
                test_true_elem.append(elem_id)
                test_is_misreg.append(elem_id in misreg_set)

    n_test = len(test_positions)
    n_misreg_test = sum(test_is_misreg)
    print(f"  Generated {n_test} verified interior points in {len(non_kuhn_elements)} Non-Kuhn elements")
    print(f"  Of these, {n_misreg_test} are in misregistered elements")

    if n_test == 0:
        print("  No valid test points. Exiting.")
        return

    test_positions_np = np.array(test_positions)
    test_true_elem_np = np.array(test_true_elem, dtype=np.int32)
    test_is_misreg_np = np.array(test_is_misreg)

    # JIT-compiled 3×3×3 search
    @jax.jit
    def search_3x3x3(pos):
        elem_id, n_tests = search_mesh_aligned_octree_multi_local(
            pos, octree_gpu, max_tests=jnp.int32(600)
        )
        return elem_id, n_tests

    found_ids = []
    n_tests_list = []

    t0 = time.time()
    for i in range(n_test):
        pos_gpu = jnp.array(test_positions_np[i])
        eid, nt = search_3x3x3(pos_gpu)
        found_ids.append(int(jax.block_until_ready(eid)))
        n_tests_list.append(int(jax.block_until_ready(nt)))
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            remaining = (n_test - i - 1) / rate
            print(f"    Searched {i+1}/{n_test}... ({rate:.0f}/s, ~{remaining:.0f}s remaining)")

    print(f"  GPU search done in {time.time()-t0:.1f}s")

    found_ids_np = np.array(found_ids, dtype=np.int32)

    n_found = int((found_ids_np >= 0).sum())
    n_missed = n_test - n_found
    n_correct = int(((found_ids_np >= 0) & (found_ids_np == test_true_elem_np)).sum())
    n_wrong = int(((found_ids_np >= 0) & (found_ids_np != test_true_elem_np)).sum())

    print(f"\n  === 3x3x3 Results on Non-Kuhn Element Interiors ===")
    print(f"  Total test points:  {n_test}")
    print(f"  Found (any elem):   {n_found} ({100.0*n_found/n_test:.1f}%)")
    print(f"  Missed:             {n_missed} ({100.0*n_missed/n_test:.1f}%)")
    print(f"  Found correct elem: {n_correct}")
    print(f"  Found wrong elem:   {n_wrong}")

    # Breakdown: misregistered vs properly registered
    if n_misreg_test > 0:
        misreg_found = int((found_ids_np[test_is_misreg_np] >= 0).sum())
        misreg_missed = n_misreg_test - misreg_found
        print(f"\n  Misregistered Non-Kuhn ({len(misregistered)} elems, {n_misreg_test} points):")
        print(f"    Found: {misreg_found} ({100.0*misreg_found/n_misreg_test:.1f}%)")
        print(f"    Missed: {misreg_missed} ({100.0*misreg_missed/n_misreg_test:.1f}%)")

    proper_mask = ~test_is_misreg_np
    n_proper_test = int(proper_mask.sum())
    if n_proper_test > 0:
        proper_found = int((found_ids_np[proper_mask] >= 0).sum())
        proper_missed = n_proper_test - proper_found
        print(f"\n  Properly registered Non-Kuhn ({len(non_kuhn_elements)-len(misregistered)} elems, {n_proper_test} points):")
        print(f"    Found: {proper_found} ({100.0*proper_found/n_proper_test:.1f}%)")
        print(f"    Missed: {proper_missed} ({100.0*proper_missed/n_proper_test:.1f}%)")

    # ========================================================================
    # [7] CPU replay for missed points - trace exactly what 3×3×3 looked at
    # ========================================================================
    missed_indices = np.where(found_ids_np < 0)[0]

    if len(missed_indices) > 0:
        print(f"\n\n[7/7] CPU replay of 3x3x3 search for {min(len(missed_indices), 10)} missed points...")

        # Build level_cell_sizes lookup from octree
        level_cell_sizes = {}
        for lvl in range(21):
            mask = cell_levels_cpu == lvl
            if mask.any():
                level_cell_sizes[lvl] = cell_sizes_cpu[mask][0]

        # Build Morton lookup for CPU replay: (morton, level) -> cell_idx
        cell_lookup = {}
        for c_idx in range(octree_multi.n_cells):
            key = (int(cell_morton_codes_cpu[c_idx]), int(cell_levels_cpu[c_idx]))
            cell_lookup[key] = c_idx

        morton_offset = (1 << 19)
        max_coord = (1 << 20)

        for rank, idx in enumerate(missed_indices[:10]):
            pos = test_positions_np[idx].astype(np.float64)
            true_elem = test_true_elem_np[idx]
            is_misreg = test_is_misreg_np[idx]

            true_verts = node_positions[connectivity[true_elem]]
            centroid = true_verts.mean(axis=0)

            print(f"\n  --- Missed point {rank+1} (test index {idx}) ---")
            print(f"  Position:     ({pos[0]:.10f}, {pos[1]:.10f}, {pos[2]:.10f})")
            print(f"  True element: {true_elem} (misregistered={is_misreg})")
            print(f"  Elem centroid: ({centroid[0]:.10f}, {centroid[1]:.10f}, {centroid[2]:.10f})")
            print(f"  Elem vertices:")
            for vi in range(4):
                v = true_verts[vi]
                print(f"    V{vi}: ({v[0]:.10f}, {v[1]:.10f}, {v[2]:.10f})")

            # Show registration
            start = elem_to_cells_offsets[true_elem]
            end = elem_to_cells_offsets[true_elem + 1]
            n_cells = end - start
            cell_indices = elem_to_cells_data[start:end]

            print(f"  Registered in {n_cells} cell(s):")
            for ci_idx in range(n_cells):
                ci = cell_indices[ci_idx]
                cs = cell_sizes_cpu[ci]
                gi = cell_grid_indices_cpu[ci]
                lvl = cell_levels_cpu[ci]
                print(f"    Cell {ci}: level={lvl}, grid=({gi[0]},{gi[1]},{gi[2]}), "
                      f"size=({cs[0]:.8f},{cs[1]:.8f},{cs[2]:.8f})")

                if np.all(cs > 1e-15):
                    pos_grid = np.floor(pos / cs).astype(int)
                    diff = np.abs(gi - pos_grid)
                    print(f"      Particle grid at level {lvl}: ({pos_grid[0]},{pos_grid[1]},{pos_grid[2]})"
                          f"  |diff|=({diff[0]},{diff[1]},{diff[2]}) max={diff.max()}")
                    if diff.max() > 1:
                        print(f"      ** UNREACHABLE: max grid distance {diff.max()} > 1")

            # CPU replay of search
            print(f"\n  CPU replay of 3x3x3 search:")
            total_cells_searched = 0
            total_elems_tested = 0
            found_in_replay = False

            for level_idx in range(8):
                level = 14 - level_idx
                if level not in level_cell_sizes:
                    continue

                cs = level_cell_sizes[level]
                if np.any(cs < 1e-15):
                    continue

                i_base = int(np.floor(pos[0] / cs[0]))
                j_base = int(np.floor(pos[1] / cs[1]))
                k_base = int(np.floor(pos[2] / cs[2]))

                level_cells = 0
                level_elems = 0
                true_elem_found_in_cell = False

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        for dk in [-1, 0, 1]:
                            i = i_base + di
                            j = j_base + dj
                            k = k_base + dk

                            i_m = max(0, min(i + morton_offset, max_coord - 1))
                            j_m = max(0, min(j + morton_offset, max_coord - 1))
                            k_m = max(0, min(k + morton_offset, max_coord - 1))

                            morton = encode_morton_3d_single(i_m, j_m, k_m, max_depth=21)
                            key = (morton, level)

                            if key in cell_lookup:
                                c_idx = cell_lookup[key]
                                c_start = octree_multi.cell_to_elements_offsets[c_idx]
                                c_end = octree_multi.cell_to_elements_offsets[c_idx + 1]
                                n_elems = c_end - c_start
                                total_cells_searched += 1
                                level_cells += 1
                                level_elems += n_elems
                                total_elems_tested += n_elems

                                # Check if true element is in this cell
                                elems_in_cell = octree_multi.cell_to_elements_data[c_start:c_end]
                                if true_elem in elems_in_cell:
                                    true_elem_found_in_cell = True

                                # Test all elements
                                for ei in range(n_elems):
                                    e_id = octree_multi.cell_to_elements_data[c_start + ei]
                                    v = node_positions[connectivity[e_id]]
                                    if point_in_tet_numpy(pos, v):
                                        found_in_replay = True
                                        print(f"    Level {level}: FOUND elem {e_id} at offset=({di},{dj},{dk})")
                                        break

                            if found_in_replay:
                                break
                        if found_in_replay:
                            break
                    if found_in_replay:
                        break

                if level_cells > 0:
                    print(f"    Level {level}: searched {level_cells} cells, {level_elems} elems, "
                          f"true_elem in cells={true_elem_found_in_cell}")

                if found_in_replay:
                    break

            if not found_in_replay:
                print(f"    NOT FOUND: searched {total_cells_searched} cells, {total_elems_tested} elements total")
                print(f"    ROOT CAUSE: Element {true_elem} is NOT in any cell reachable by 3x3x3")
    else:
        print("\n[7/7] No missed points! All Non-Kuhn element interiors found by 3x3x3.")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Non-Kuhn elements:       {len(non_kuhn_elements):,} ({100.0*len(non_kuhn_elements)/n_elements:.3f}%)")
    print(f"  Misregistered (unreachable): {len(misregistered)}")
    print(f"  Test points (inside Non-Kuhn): {n_test}")
    print(f"  3x3x3 found:            {n_found} ({100.0*n_found/n_test:.1f}%)")
    print(f"  3x3x3 missed:           {n_missed} ({100.0*n_missed/n_test:.1f}%)")
    if n_missed > 0:
        print(f"\n  ROOT CAUSE: Non-Kuhn elements are registered in their face neighbor's cells,")
        print(f"  which can be spatially far from the element itself. The 3x3x3 search only")
        print(f"  checks +-1 cell neighborhood, so it cannot reach these misregistered cells.")
        print(f"  FIX: Register Non-Kuhn elements using their OWN vertex positions with the")
        print(f"  face neighbor's cell_size, instead of copying the neighbor's cells blindly.")


if __name__ == "__main__":
    main()
