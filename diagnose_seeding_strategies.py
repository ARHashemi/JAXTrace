#!/usr/bin/env python3
"""
Seeding Strategy Diagnostic for mesh_aligned_octree_multi_local (3×3×3)

Tests 4 particle seeding strategies over all elements to measure search success rate:
  1. Element centroids (exact center, guaranteed to be found)
  2. Centroids ± 10% element-size along X (tiny perturbation, stays inside)
  3. Centroids ± 1× element-size along X (full edge-length, may cross face)
  4. Centroids ± 2× element-size along X (large perturbation, exits element)

All strategies use the 3×3×3 mesh_aligned_octree_multi_local search exclusively.
Reports found/missed/correct/wrong counts per strategy.

Patterns match diagnose_void_region_corrected.py and benchmark_l2_search_methods_with-export.py.
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

# Import octree builder and uploader (following benchmark pattern)
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import extract_octree_cells_vertex_multi
from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import (
    find_axis_aligned_edges_single,
    encode_morton_3d_single,
)
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu

# Import search function
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

# Number of particles per seeding strategy (subsample of elements in the seed region)
N_PARTICLES = 50_000
SEED = 42

# Max elements to test in 3×3×3 search
MAX_TESTS = 600

# Spatial seeding region — fractional bounds within the mesh domain.
# Each axis: (min_fraction, max_fraction) relative to (domain_min, domain_max).
# Use (0.0, 1.0) on all axes to sample from the entire mesh.
# Matches the PARTICLE_BOUNDS_FRACTION convention in benchmark_l2_search_methods_with-export.py.
SEED_REGION = {
    'x': (0.15, 0.25),  # full domain in X
    'y': (0.0, 1.0),  # full domain in Y
    'z': (0.0, 1.0),  # full domain in Z
}


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


def compute_element_edge_length(vertices):
    """Compute minimum edge length of a tetrahedron."""
    min_len = np.inf
    for i in range(4):
        for j in range(i + 1, 4):
            length = np.linalg.norm(vertices[i] - vertices[j])
            if length < min_len:
                min_len = length
    return min_len


def analyze_missed_positions(
    strategy_name,
    missed_indices,
    positions,
    found_ids,
    true_elem_ids,
    perturbations,
    octree_multi,
    node_positions,
    connectivity,
    cell_lookup,
    level_cell_sizes,
    mesh_bbox,
):
    """
    For each missed position, report:
      - Position coordinates
      - Source element (ID, centroid, vertices, Kuhn/Non-Kuhn, level, registration)
      - Perturbation applied
      - Whether position is inside the mesh bounding box
      - CPU 3×3×3 replay: what cells searched, whether true elem reachable
      - Root cause summary
    """
    if len(missed_indices) == 0:
        return

    print(f"\n{'='*80}")
    print(f"MISSED POSITION ANALYSIS: {strategy_name}")
    print(f"  {len(missed_indices)} missed particle(s)")
    print(f"{'='*80}")

    morton_offset = (1 << 19)
    max_coord = (1 << 20)

    elem_to_cells_offsets = octree_multi.element_to_cells_offsets
    elem_to_cells_data = octree_multi.element_to_cells_data
    cell_sizes_cpu = octree_multi.cell_sizes
    cell_grid_indices_cpu = octree_multi.cell_grid_indices
    cell_levels_cpu = octree_multi.cell_levels

    mesh_min, mesh_max = mesh_bbox

    for rank, idx in enumerate(missed_indices):
        pos = positions[idx].astype(np.float64)
        true_elem = int(true_elem_ids[idx])
        perturb = perturbations[idx]

        true_verts = node_positions[connectivity[true_elem]].astype(np.float64)
        centroid = true_verts.mean(axis=0)

        # Kuhn/Non-Kuhn classification of source element
        cell_size_src, level_src = find_axis_aligned_edges_single(true_verts, tolerance=1e-6)
        is_non_kuhn = bool(np.any(cell_size_src == 0))

        # Registration of source element
        e_start = elem_to_cells_offsets[true_elem]
        e_end = elem_to_cells_offsets[true_elem + 1]
        n_reg_cells = e_end - e_start
        reg_cell_indices = elem_to_cells_data[e_start:e_end]

        # Inside mesh bbox?
        in_bbox = bool(np.all(pos >= mesh_min) and np.all(pos <= mesh_max))

        print(f"\n  ─── Missed #{rank+1} (particle index {idx}) ───")
        print(f"  Landed position:    ({pos[0]:.10f}, {pos[1]:.10f}, {pos[2]:.10f})")
        print(f"  Perturbation (X):   {perturb[0]:+.6e}  (Y: {perturb[1]:+.6e}, Z: {perturb[2]:+.6e})")
        print(f"  In mesh bbox:       {in_bbox}")
        print(f"  Mesh bbox:          ({mesh_min[0]:.6f},{mesh_min[1]:.6f},{mesh_min[2]:.6f})"
              f" → ({mesh_max[0]:.6f},{mesh_max[1]:.6f},{mesh_max[2]:.6f})")

        print(f"\n  Source element: {true_elem}")
        print(f"    Type:    {'Non-Kuhn' if is_non_kuhn else f'Kuhn (level {level_src})'}")
        print(f"    Centroid: ({centroid[0]:.10f}, {centroid[1]:.10f}, {centroid[2]:.10f})")
        print(f"    BBox:    ({true_verts.min(0)[0]:.8f},{true_verts.min(0)[1]:.8f},{true_verts.min(0)[2]:.8f})"
              f" → ({true_verts.max(0)[0]:.8f},{true_verts.max(0)[1]:.8f},{true_verts.max(0)[2]:.8f})")
        print(f"    Vertices:")
        for vi in range(4):
            v = true_verts[vi]
            print(f"      V{vi}: ({v[0]:.10f}, {v[1]:.10f}, {v[2]:.10f})")
        print(f"    Registered in {n_reg_cells} octree cell(s):")
        for ci_idx in range(n_reg_cells):
            ci = reg_cell_indices[ci_idx]
            cs = cell_sizes_cpu[ci]
            gi = cell_grid_indices_cpu[ci]
            lvl = cell_levels_cpu[ci]
            pos_grid = np.floor(pos / cs).astype(int) if np.all(cs > 1e-15) else np.array([0, 0, 0])
            diff = np.abs(gi - pos_grid) if np.all(cs > 1e-15) else np.array([-1, -1, -1])
            reachable = bool(diff.max() <= 1)
            print(f"      Cell {ci}: level={lvl}, grid=({gi[0]},{gi[1]},{gi[2]}), "
                  f"size=({cs[0]:.6e},{cs[1]:.6e},{cs[2]:.6e})")
            print(f"        Particle grid at level {lvl}: ({pos_grid[0]},{pos_grid[1]},{pos_grid[2]})"
                  f"  |diff|=({diff[0]},{diff[1]},{diff[2]}) max={diff.max()}"
                  f"  reachable={'YES' if reachable else 'NO ← OUT OF 3×3×3'}")

        # CPU replay of 3×3×3 search
        print(f"\n  CPU replay of 3×3×3 search (levels 14 → 7):")
        total_cells_searched = 0
        total_elems_tested = 0
        true_elem_reachable = False
        cpu_found_elem = -1

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
            true_elem_in_level = False
            found_this_level = False

            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    for dk in [-1, 0, 1]:
                        ii = i_base + di
                        jj = j_base + dj
                        kk = k_base + dk

                        i_m = max(0, min(ii + morton_offset, max_coord - 1))
                        j_m = max(0, min(jj + morton_offset, max_coord - 1))
                        k_m = max(0, min(kk + morton_offset, max_coord - 1))

                        morton = encode_morton_3d_single(i_m, j_m, k_m, max_depth=21)
                        key = (morton, level)

                        if key in cell_lookup:
                            c_idx = cell_lookup[key]
                            c_start = octree_multi.cell_to_elements_offsets[c_idx]
                            c_end = octree_multi.cell_to_elements_offsets[c_idx + 1]
                            n_elems = c_end - c_start
                            level_cells += 1
                            level_elems += n_elems
                            total_cells_searched += 1
                            total_elems_tested += n_elems

                            elems_in_cell = octree_multi.cell_to_elements_data[c_start:c_end]
                            if true_elem in elems_in_cell:
                                true_elem_in_level = True
                                true_elem_reachable = True

                            for ei in range(n_elems):
                                e_id = int(octree_multi.cell_to_elements_data[c_start + ei])
                                v = node_positions[connectivity[e_id]].astype(np.float64)
                                if point_in_tet_numpy(pos, v):
                                    cpu_found_elem = e_id
                                    found_this_level = True
                                    break

                        if found_this_level:
                            break
                    if found_this_level:
                        break
                if found_this_level:
                    break

            status = f"{level_cells} cells, {level_elems} elems"
            if true_elem_in_level:
                status += ", TRUE ELEM IN CELLS"
            if level_cells > 0:
                print(f"    Level {level}: {status}")
            if found_this_level:
                break

        # Final verdict
        print(f"\n  CPU replay result:")
        if cpu_found_elem >= 0:
            print(f"    CPU FOUND elem {cpu_found_elem} (GPU missed it — possible JIT/float32 divergence)")
        else:
            print(f"    NOT FOUND by CPU replay either")
            print(f"    Total searched: {total_cells_searched} cells, {total_elems_tested} elements")
            if true_elem_reachable:
                print(f"    True elem IS in the searched cells → point-in-tet test fails (position outside element)")
                print(f"    ROOT CAUSE: Position has crossed the mesh boundary or into a void region")
            else:
                print(f"    True elem NOT in any searched cell → registration/Morton code mismatch")
                print(f"    ROOT CAUSE: Element not reachable by 3×3×3 from this position")

        # Check if position is inside any mesh element that co-registers in the same cells
        # as the source element (cheap neighborhood via registered cells)
        neighbor_elems = set()
        for ci_idx in range(n_reg_cells):
            ci = reg_cell_indices[ci_idx]
            c_start = octree_multi.cell_to_elements_offsets[ci]
            c_end = octree_multi.cell_to_elements_offsets[ci + 1]
            for ei in range(c_start, c_end):
                neighbor_elems.add(int(octree_multi.cell_to_elements_data[ei]))

        found_neighbor = -1
        for e_id in neighbor_elems:
            v = node_positions[connectivity[e_id]].astype(np.float64)
            if point_in_tet_numpy(pos, v):
                found_neighbor = e_id
                break

        if found_neighbor >= 0:
            n_kuhn_src = not np.any(find_axis_aligned_edges_single(
                node_positions[connectivity[found_neighbor]].astype(np.float64), 1e-6)[0] == 0)
            print(f"    Point IS inside adjacent elem {found_neighbor} "
                  f"({'Kuhn' if n_kuhn_src else 'Non-Kuhn'}) — valid result in neighbor cell")
        else:
            print(f"    Point is NOT inside any cell-neighbor element → likely outside mesh")


def run_seeding_strategy(
    strategy_name,
    positions,
    true_elem_ids,
    search_fn,
):
    """
    Run the 3×3×3 search on a batch of positions and report results.

    Args:
        strategy_name: Name of the seeding strategy (for printing)
        positions: (N, 3) float32 array of query positions
        true_elem_ids: (N,) int32 array of ground-truth element IDs (or -1 if unknown)
        search_fn: JIT-compiled search function pos -> (elem_id, n_tests)

    Returns:
        dict with counts: n_total, n_found, n_missed, n_correct, n_wrong, mean_tests
    """
    n_total = len(positions)
    print(f"\n  --- Strategy: {strategy_name} ---")
    print(f"  Particles: {n_total:,}")

    found_ids = np.empty(n_total, dtype=np.int32)
    n_tests_arr = np.empty(n_total, dtype=np.int32)

    t0 = time.time()
    for i in range(n_total):
        pos_gpu = jnp.array(positions[i])
        eid, nt = search_fn(pos_gpu)
        found_ids[i] = int(jax.block_until_ready(eid))
        n_tests_arr[i] = int(jax.block_until_ready(nt))

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n_total - i - 1) / rate
            print(f"    Searched {i+1:,}/{n_total:,}... ({rate:.0f}/s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"  Searched in {elapsed:.1f}s ({n_total/elapsed:.0f} p/s)")

    n_found = int((found_ids >= 0).sum())
    n_missed = n_total - n_found
    mean_tests = float(n_tests_arr.mean())

    # Correctness: only meaningful when true_elem_ids are known (>= 0)
    has_truth = true_elem_ids >= 0
    n_with_truth = int(has_truth.sum())
    if n_with_truth > 0:
        n_correct = int(((found_ids == true_elem_ids) & has_truth).sum())
        n_wrong = int(((found_ids >= 0) & (found_ids != true_elem_ids) & has_truth).sum())
    else:
        n_correct = 0
        n_wrong = 0

    print(f"  Found:   {n_found:,}/{n_total:,} ({100.0*n_found/n_total:.2f}%)")
    print(f"  Missed:  {n_missed:,}/{n_total:,} ({100.0*n_missed/n_total:.2f}%)")
    print(f"  Correct: {n_correct:,}/{n_with_truth:,} ({100.0*n_correct/max(n_with_truth,1):.2f}%) "
          f"[of those with ground truth]")
    print(f"  Wrong:   {n_wrong:,}/{n_with_truth:,} ({100.0*n_wrong/max(n_with_truth,1):.2f}%)")
    print(f"  Mean tests/particle: {mean_tests:.1f}")

    return {
        'n_total': n_total,
        'n_found': n_found,
        'n_missed': n_missed,
        'n_correct': n_correct,
        'n_wrong': n_wrong,
        'n_with_truth': n_with_truth,
        'mean_tests': mean_tests,
        'found_ids': found_ids,  # (N,) int32, -1 if not found
    }


def main():
    print("=" * 80)
    print("Seeding Strategy Diagnostic: mesh_aligned_octree_multi_local (3×3×3)")
    print("=" * 80)
    print(f"Strategies (perturbations along X axis only):")
    print(f"  1. Exact element centroids")
    print(f"  2. Centroids ± 10% element-size along X")
    print(f"  3. Centroids ± 1× element-size along X")
    print(f"  4. Centroids ± 2× element-size along X")
    print(f"N particles per strategy: {N_PARTICLES:,}")
    print(f"Seed region (fractional): X{SEED_REGION['x']}  Y{SEED_REGION['y']}  Z{SEED_REGION['z']}")

    # ========================================================================
    # [1] Load Mesh (EXACT pattern from benchmark lines 660-676)
    # ========================================================================
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
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Elements: {n_elements:,}, Nodes: {node_positions.shape[0]:,}")

    # ========================================================================
    # [2] Deduplicate (EXACT pattern from benchmark lines 678-683)
    # ========================================================================
    print("\n[2/6] Deduplicating...")
    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    n_elements = connectivity.shape[0]
    print(f"  Removed {n_duplicates_removed:,} duplicates")
    print(f"  Nodes after dedup: {node_positions.shape[0]:,}")

    # ========================================================================
    # [3] Precompute metadata for inverse method (EXACT pattern from benchmark)
    # ========================================================================
    print("\n[3/6] Precomputing metadata for inverse point-in-tet...")
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
    # [4] Extract and upload octree (EXACT pattern from benchmark lines 726-764)
    # ========================================================================
    print("\n[4/6] Building and uploading mesh-aligned octree (multi-cell)...")
    t0 = time.time()
    octree_multi = extract_octree_cells_vertex_multi(
        node_positions, connectivity, tolerance=1e-6, verbose=False
    )
    print(f"  Extracted in {time.time()-t0:.1f}s")
    print(f"    Cells: {octree_multi.n_cells:,}")
    print(f"    Elements per cell: {octree_multi.elements_per_cell_mean:.2f}")
    print(f"    Cells per element: {octree_multi.cells_per_element_mean:.2f}")

    t0 = time.time()
    # Correct argument order: (connectivity, node_positions, octree_cells)
    octree_gpu = upload_mesh_aligned_octree_to_gpu(
        connectivity, node_positions, octree_multi, verbose=False
    )
    print(f"  Uploaded in {time.time()-t0:.1f}s")

    # Build CPU-side lookup structures for missed-position analysis
    print("  Building CPU lookup tables for miss analysis...")
    cell_levels_cpu = octree_multi.cell_levels
    cell_sizes_cpu = octree_multi.cell_sizes
    cell_morton_codes_cpu = octree_multi.cell_morton_codes

    # level -> representative cell_size (float64 array of shape (3,))
    level_cell_sizes = {}
    for lvl in range(21):
        mask = cell_levels_cpu == lvl
        if mask.any():
            level_cell_sizes[lvl] = cell_sizes_cpu[mask][0].astype(np.float64)

    # (morton, level) -> cell_idx
    cell_lookup = {}
    for c_idx in range(octree_multi.n_cells):
        key = (int(cell_morton_codes_cpu[c_idx]), int(cell_levels_cpu[c_idx]))
        cell_lookup[key] = c_idx

    # Mesh bounding box (for checking if missed positions are outside the mesh)
    mesh_bbox = (node_positions.min(axis=0).astype(np.float64),
                 node_positions.max(axis=0).astype(np.float64))
    print(f"  Lookup ready: {len(level_cell_sizes)} levels, {len(cell_lookup)} cells")

    # ========================================================================
    # [5] Filter elements to seed region, sample N_PARTICLES, compute geometry
    # ========================================================================
    print(f"\n[5/6] Filtering to seed region and sampling {N_PARTICLES:,} elements...")
    t0 = time.time()
    np.random.seed(SEED)

    # Compute domain bounds from node positions (matching benchmark pattern)
    domain_min = node_positions.min(axis=0).astype(np.float64)
    domain_max = node_positions.max(axis=0).astype(np.float64)
    domain_size = domain_max - domain_min

    # Compute absolute seed region bounds from SEED_REGION fractions
    region_min = np.array([
        domain_min[i] + SEED_REGION[ax][0] * domain_size[i]
        for i, ax in enumerate(['x', 'y', 'z'])
    ], dtype=np.float64)
    region_max = np.array([
        domain_min[i] + SEED_REGION[ax][1] * domain_size[i]
        for i, ax in enumerate(['x', 'y', 'z'])
    ], dtype=np.float64)

    print(f"  Domain:      ({domain_min[0]:.6f},{domain_min[1]:.6f},{domain_min[2]:.6f})"
          f" → ({domain_max[0]:.6f},{domain_max[1]:.6f},{domain_max[2]:.6f})")
    print(f"  Seed region fractions: "
          f"X{SEED_REGION['x']}  Y{SEED_REGION['y']}  Z{SEED_REGION['z']}")
    print(f"  Seed region (abs): ({region_min[0]:.6f},{region_min[1]:.6f},{region_min[2]:.6f})"
          f" → ({region_max[0]:.6f},{region_max[1]:.6f},{region_max[2]:.6f})")

    # Compute all element centroids and filter by region
    # (centroid-based filter, same as benchmark PARTICLE_SEEDING=='centroids' path)
    print(f"  Scanning {n_elements:,} elements for centroid-in-region filter...")
    all_centroids = node_positions[connectivity].mean(axis=1)  # (n_elements, 3), float64
    in_region = (
        (all_centroids[:, 0] >= region_min[0]) & (all_centroids[:, 0] <= region_max[0]) &
        (all_centroids[:, 1] >= region_min[1]) & (all_centroids[:, 1] <= region_max[1]) &
        (all_centroids[:, 2] >= region_min[2]) & (all_centroids[:, 2] <= region_max[2])
    )
    region_elem_ids = np.where(in_region)[0]
    n_region_elements = len(region_elem_ids)
    print(f"  Elements in seed region: {n_region_elements:,} / {n_elements:,}"
          f" ({100.0 * n_region_elements / n_elements:.2f}%)")

    if n_region_elements == 0:
        print("  ERROR: No elements in seed region. Check SEED_REGION fractions.")
        return

    # Sample N_PARTICLES from region elements (with replacement if region is smaller)
    selected_elem_ids = region_elem_ids[
        np.random.choice(n_region_elements, size=N_PARTICLES, replace=(N_PARTICLES > n_region_elements))
    ]
    if N_PARTICLES > n_region_elements:
        print(f"  WARNING: N_PARTICLES ({N_PARTICLES:,}) > region elements ({n_region_elements:,}),"
              f" sampling with replacement")

    # Compute centroids and minimum edge length for each selected element
    centroids = np.zeros((N_PARTICLES, 3), dtype=np.float64)
    min_edge_lengths = np.zeros(N_PARTICLES, dtype=np.float64)

    for i, elem_id in enumerate(selected_elem_ids):
        vertices = node_positions[connectivity[elem_id]]  # (4, 3)
        centroids[i] = vertices.mean(axis=0)
        min_edge_lengths[i] = compute_element_edge_length(vertices)

        if (i + 1) % 10000 == 0:
            print(f"    Computed {i+1:,}/{N_PARTICLES:,}...")

    centroids_f32 = centroids.astype(np.float32)
    true_elem_ids = selected_elem_ids.astype(np.int32)

    print(f"  Computed in {time.time()-t0:.1f}s")
    print(f"  Element edge length stats (sampled {N_PARTICLES:,} from {n_region_elements:,} in region):")
    print(f"    Min:    {min_edge_lengths.min():.6e}")
    print(f"    Max:    {min_edge_lengths.max():.6e}")
    print(f"    Mean:   {min_edge_lengths.mean():.6e}")
    print(f"    Median: {np.median(min_edge_lengths):.6e}")

    # ========================================================================
    # [6] Run all 4 seeding strategies
    # ========================================================================
    print("\n[6/6] Running 4 seeding strategies with 3×3×3 search...")
    print("=" * 80)

    # JIT-compile the 3×3×3 search once (shared across all strategies)
    @jax.jit
    def search_3x3x3(pos):
        elem_id, n_tests = search_mesh_aligned_octree_multi_local(
            pos, octree_gpu, max_tests=jnp.int32(MAX_TESTS)
        )
        return elem_id, n_tests

    # Warmup (compile)
    print("\n  Warming up JIT...")
    _ = search_3x3x3(jnp.array(centroids_f32[0]))
    jax.block_until_ready(_)
    print("  JIT ready.")

    results = {}

    # ------------------------------------------------------------------
    # Strategy 1: Exact centroids
    # ------------------------------------------------------------------
    positions_1 = centroids_f32.copy()
    perturbations_1 = np.zeros((N_PARTICLES, 3), dtype=np.float64)
    results['1_centroids'] = run_seeding_strategy(
        "Strategy 1: Exact centroids",
        positions_1,
        true_elem_ids,
        search_3x3x3,
    )
    missed_1 = np.where(results['1_centroids']['found_ids'] < 0)[0]
    analyze_missed_positions(
        "Strategy 1: Exact centroids", missed_1, positions_1, results['1_centroids']['found_ids'],
        true_elem_ids, perturbations_1, octree_multi, node_positions, connectivity,
        cell_lookup, level_cell_sizes, mesh_bbox,
    )

    # ------------------------------------------------------------------
    # Strategy 2: 10% element-size perturbed centroids (X axis only)
    # ------------------------------------------------------------------
    np.random.seed(SEED + 1)
    signs_2 = np.sign(np.random.randn(N_PARTICLES)).astype(np.float64)  # ±1 along X
    perturbations_2 = np.zeros((N_PARTICLES, 3), dtype=np.float64)
    perturbations_2[:, 0] = signs_2 * (0.1 * min_edge_lengths)
    positions_2 = (centroids + perturbations_2).astype(np.float32)
    results['2_perturbed_10pct'] = run_seeding_strategy(
        "Strategy 2: 10% element-size perturbed (X only)",
        positions_2,
        true_elem_ids,
        search_3x3x3,
    )
    missed_2 = np.where(results['2_perturbed_10pct']['found_ids'] < 0)[0]
    analyze_missed_positions(
        "Strategy 2: 10% element-size perturbed (X only)", missed_2, positions_2,
        results['2_perturbed_10pct']['found_ids'], true_elem_ids, perturbations_2,
        octree_multi, node_positions, connectivity, cell_lookup, level_cell_sizes, mesh_bbox,
    )

    # ------------------------------------------------------------------
    # Strategy 3: 1× element-size perturbed centroids (X axis only)
    # ------------------------------------------------------------------
    np.random.seed(SEED + 2)
    signs_3 = np.sign(np.random.randn(N_PARTICLES)).astype(np.float64)  # ±1 along X
    perturbations_3 = np.zeros((N_PARTICLES, 3), dtype=np.float64)
    perturbations_3[:, 0] = signs_3 * min_edge_lengths
    positions_3 = (centroids + perturbations_3).astype(np.float32)
    results['3_perturbed_1x'] = run_seeding_strategy(
        "Strategy 3: 1× element-size perturbed (X only)",
        positions_3,
        true_elem_ids,
        search_3x3x3,
    )
    missed_3 = np.where(results['3_perturbed_1x']['found_ids'] < 0)[0]
    analyze_missed_positions(
        "Strategy 3: 1× element-size perturbed (X only)", missed_3, positions_3,
        results['3_perturbed_1x']['found_ids'], true_elem_ids, perturbations_3,
        octree_multi, node_positions, connectivity, cell_lookup, level_cell_sizes, mesh_bbox,
    )

    # ------------------------------------------------------------------
    # Strategy 4: 2× element-size perturbed centroids (X axis only)
    # ------------------------------------------------------------------
    np.random.seed(SEED + 3)
    signs_4 = np.sign(np.random.randn(N_PARTICLES)).astype(np.float64)  # ±1 along X
    perturbations_4 = np.zeros((N_PARTICLES, 3), dtype=np.float64)
    perturbations_4[:, 0] = signs_4 * (2.0 * min_edge_lengths)
    positions_4 = (centroids + perturbations_4).astype(np.float32)
    results['4_perturbed_2x'] = run_seeding_strategy(
        "Strategy 4: 2× element-size perturbed (X only)",
        positions_4,
        true_elem_ids,
        search_3x3x3,
    )
    missed_4 = np.where(results['4_perturbed_2x']['found_ids'] < 0)[0]
    analyze_missed_positions(
        "Strategy 4: 2× element-size perturbed (X only)", missed_4, positions_4,
        results['4_perturbed_2x']['found_ids'], true_elem_ids, perturbations_4,
        octree_multi, node_positions, connectivity, cell_lookup, level_cell_sizes, mesh_bbox,
    )

    # ========================================================================
    # Summary Table
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: 3×3×3 search results across seeding strategies")
    print("=" * 80)
    print(f"  Octree: multi-cell vertex registration ({octree_multi.n_cells:,} cells, "
          f"{octree_multi.elements_per_cell_mean:.1f} elem/cell, "
          f"{octree_multi.cells_per_element_mean:.1f} cells/elem)")
    print(f"  Max tests per search: {MAX_TESTS}")
    print()

    header = f"{'Strategy':<42s}  {'Found':>8s}  {'Missed':>8s}  {'Found%':>7s}  {'Correct%':>9s}  {'MeanTests':>10s}"
    print(header)
    print("-" * len(header))

    strategy_labels = {
        '1_centroids':        "1. Exact centroids",
        '2_perturbed_10pct':  "2. ±10% elem-size along X",
        '3_perturbed_1x':     "3. ±1× elem-size along X",
        '4_perturbed_2x':     "4. ±2× elem-size along X",
    }

    for key, label in strategy_labels.items():
        r = results[key]
        found_pct = 100.0 * r['n_found'] / r['n_total']
        correct_pct = (100.0 * r['n_correct'] / r['n_with_truth']
                       if r['n_with_truth'] > 0 else float('nan'))
        print(f"  {label:<40s}  {r['n_found']:>8,}  {r['n_missed']:>8,}  "
              f"{found_pct:>6.2f}%  {correct_pct:>8.2f}%  {r['mean_tests']:>10.1f}")

    print()
    print("Notes:")
    print("  Found%:   fraction of queries where any element was returned")
    print("  Correct%: fraction of found results matching the ground-truth centroid element")
    print("            (perturbations along X only — all particles displaced in +X or -X)")
    print("  For strategies 3 & 4, particles may have crossed an element face along X,")
    print("  so a 'wrong' result may be a geometrically valid adjacent element.")
    print("=" * 80)


if __name__ == "__main__":
    main()
