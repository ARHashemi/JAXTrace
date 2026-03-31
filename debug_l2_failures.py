#!/usr/bin/env python3
"""
Debug L2 search failures near the refined region.

Generates particles from element centroids with small perturbation (0.1x min edge),
runs all L2 methods, and analyzes WHERE failures occur:
  - Outside mesh bounding box?
  - Inside bbox but at a refinement boundary?
  - What octree levels do failing particles map to?
  - Do the failing particles' cells have elements registered?
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ.setdefault('JAX_PLATFORMS', 'cuda,rocm,cpu')

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import jaxtrace.config as config
import jax
import jax.numpy as jnp

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import extract_octree_cells_vertex_multi
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu
from jaxtrace.gpu.search.aa_detection import precompute_aa_metadata, precompute_element_vertices
from jaxtrace.gpu.search.point_in_tet_methods import set_corrected_metadata, set_inverse_matrices_gpu
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
from jaxtrace.gpu.search.mesh_aligned_point_location import (
    search_mesh_aligned_octree_multi_local_where,
    search_mesh_aligned_octree_5x5x5_where,
)


# =============================================================================
# Config
# =============================================================================
MESH_BASE_PATH = Path("/media/arhashemi/HDD2TB/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "cylA_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (159, 159)
VELOCITY_FIELD_NAME = 'Displacement'

N_PARTICLES = 10000
SEED = 42


def compute_element_sizes(connectivity, node_positions):
    v = node_positions[connectivity]
    edges = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    edge_lengths = np.zeros((v.shape[0], 6), dtype=np.float64)
    for i, (a, b) in enumerate(edges):
        edge_lengths[:, i] = np.linalg.norm(v[:, a] - v[:, b], axis=1)
    return edge_lengths.min(axis=1)


def brute_force_search(positions, connectivity, node_positions, M_inv, p0, tol=1e-6):
    """
    CPU brute-force point-in-tet for a small set of particles.
    Uses barycentric coordinates via inverse matrix method.
    Returns element IDs (-1 if not found).
    """
    n_particles = positions.shape[0]
    n_elements = connectivity.shape[0]
    result = np.full(n_particles, -1, dtype=np.int32)

    for p_idx in range(n_particles):
        pos = positions[p_idx]
        for e_idx in range(n_elements):
            # Barycentric: lambda = M_inv @ (pos - p0)
            diff = pos - p0[e_idx]
            lam = M_inv[e_idx] @ diff  # (3,)
            lam4 = 1.0 - lam[0] - lam[1] - lam[2]
            if (lam[0] >= -tol and lam[1] >= -tol and lam[2] >= -tol and lam4 >= -tol):
                result[p_idx] = e_idx
                break

    return result


def main():
    print("=" * 80)
    print("Debug L2 Search Failures Near Refined Region")
    print("=" * 80)

    # ---- Load mesh ----
    print("\n[1/5] Loading mesh...")
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=False,
    )
    node_positions, connectivity, n_dup, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    connectivity = connectivity.astype(np.int32)
    n_elements = connectivity.shape[0]
    n_nodes = node_positions.shape[0]
    print(f"  Elements: {n_elements:,}, Nodes: {n_nodes:,}")

    # ---- Precompute ----
    print("\n[2/5] Precomputing...")
    config.POINT_IN_TET_METHOD = 'inverse'
    config.POINT_IN_TET_TOLERANCE = 1e-6

    aa_metadata = precompute_aa_metadata(connectivity, node_positions, verbose=False)
    element_vertices = precompute_element_vertices(connectivity, node_positions, verbose=False)
    M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)
    element_sizes = compute_element_sizes(connectivity, node_positions)

    # ---- Build octree & upload ----
    print("\n[3/5] Building octree...")
    mesh_octree_cells = extract_octree_cells_vertex_multi(
        node_positions, connectivity, tolerance=1e-6, verbose=False
    )
    octree_gpu = upload_mesh_aligned_octree_to_gpu(
        connectivity, node_positions, mesh_octree_cells, verbose=False
    )

    aa_gpu = type(aa_metadata)(
        base_vertex_indices=jax.device_put(aa_metadata.base_vertex_indices),
        base_vertices=jax.device_put(aa_metadata.base_vertices),
        inv_edge_lengths=jax.device_put(aa_metadata.inv_edge_lengths),
        axis_indices=jax.device_put(aa_metadata.axis_indices),
        is_axis_aligned=jax.device_put(aa_metadata.is_axis_aligned),
    )
    set_corrected_metadata(aa_gpu, jax.device_put(element_vertices))
    set_inverse_matrices_gpu(jax.device_put(M_inv_array), jax.device_put(p0_array))

    # ---- Generate particles from refined region with 0.1x perturbation ----
    print("\n[4/5] Generating particles in refined region...")
    domain_min = node_positions.min(axis=0)
    domain_max = node_positions.max(axis=0)
    domain_extent = domain_max - domain_min

    # Same region as the log: seed-x 0.35 0.65, seed-y 0.25 0.75, seed-z 0.5 1.0
    seed_frac = np.array([[0.35, 0.25, 0.50],
                          [0.65, 0.75, 1.00]])
    seed_min = domain_min + seed_frac[0] * domain_extent
    seed_max = domain_min + seed_frac[1] * domain_extent

    all_centroids = node_positions[connectivity].mean(axis=1)
    in_region = np.all((all_centroids >= seed_min) & (all_centroids <= seed_max), axis=1)
    valid_eids = np.where(in_region)[0]
    print(f"  Region: X=[{seed_min[0]:.6f},{seed_max[0]:.6f}], "
          f"Y=[{seed_min[1]:.6f},{seed_max[1]:.6f}], "
          f"Z=[{seed_min[2]:.6f},{seed_max[2]:.6f}]")
    print(f"  Elements in region: {len(valid_eids):,}")

    # Element size stats in this region
    region_sizes = element_sizes[valid_eids]
    print(f"  Element sizes in region: min={region_sizes.min():.2e}, "
          f"mean={region_sizes.mean():.2e}, max={region_sizes.max():.2e}")

    # Generate particles with 0.1x perturbation
    rng = np.random.default_rng(SEED)
    source_elems = rng.choice(valid_eids, size=N_PARTICLES, replace=True).astype(np.int32)
    verts = node_positions[connectivity[source_elems]]
    centroids = verts.mean(axis=1)

    perturbation_factor = 0.1
    sizes = element_sizes[source_elems]
    perturbation = rng.standard_normal((N_PARTICLES, 3)) * (sizes[:, None] * perturbation_factor)
    positions = centroids + perturbation

    # Check bbox
    in_bbox = np.all((positions >= domain_min) & (positions <= domain_max), axis=1)
    print(f"\n  Perturbation factor: {perturbation_factor}x (of min edge length)")
    print(f"  Mean displacement: {np.linalg.norm(perturbation, axis=1).mean():.2e}")
    print(f"  In mesh bbox: {in_bbox.sum()}/{N_PARTICLES}")

    # ---- Run 3x3x3 search ----
    print("\n[5/5] Running 3x3x3 search and analyzing failures...")
    positions_gpu = jax.device_put(positions.astype(config.FLOAT_DTYPE_NP))
    max_tests = jnp.int32(600)

    @jax.jit
    def search_batch(pos_batch):
        def single(pos):
            elem_id, n_tests = search_mesh_aligned_octree_multi_local_where(
                pos, octree_gpu, max_tests=max_tests
            )
            return elem_id, n_tests
        return jax.vmap(single)(pos_batch)

    found_eids_all = np.full(N_PARTICLES, -1, dtype=np.int32)
    n_tests_all = np.zeros(N_PARTICLES, dtype=np.int32)

    batch_size = 50000
    for start in range(0, N_PARTICLES, batch_size):
        end = min(start + batch_size, N_PARTICLES)
        eids, tests = search_batch(positions_gpu[start:end])
        found_eids_all[start:end] = np.array(jax.block_until_ready(eids), dtype=np.int32)
        n_tests_all[start:end] = np.array(jax.block_until_ready(tests), dtype=np.int32)

    n_found = (found_eids_all >= 0).sum()
    n_not_found = N_PARTICLES - n_found
    print(f"\n  3x3x3 results: found={n_found}, not_found={n_not_found}")

    # Classify failures
    not_found_mask = found_eids_all < 0
    outside_bbox = ~in_bbox
    inside_bbox_not_found = not_found_mask & in_bbox

    print(f"\n  Failure breakdown:")
    print(f"    Outside mesh bbox:            {(not_found_mask & outside_bbox).sum()}")
    print(f"    Inside bbox but NOT found:    {inside_bbox_not_found.sum()}")

    if inside_bbox_not_found.sum() == 0:
        print("\n  All failures are particles outside the mesh bbox.")
        print("  No search failure inside the domain — L2 is accurate.")
        return

    # ---- Deep analysis of inside-bbox failures ----
    fail_indices = np.where(inside_bbox_not_found)[0]
    n_fail = len(fail_indices)
    print(f"\n  Analyzing {n_fail} inside-bbox failures...")

    fail_positions = positions[fail_indices]
    fail_source_elems = source_elems[fail_indices]
    fail_perturbations = perturbation[fail_indices]
    fail_sizes = sizes[fail_indices]
    fail_n_tests = n_tests_all[fail_indices]

    # How far did they move relative to element size?
    fail_displacements = np.linalg.norm(fail_perturbations, axis=1)
    fail_relative_disp = fail_displacements / fail_sizes

    print(f"    Displacement stats:")
    print(f"      Absolute: min={fail_displacements.min():.2e}, "
          f"mean={fail_displacements.mean():.2e}, max={fail_displacements.max():.2e}")
    print(f"      Relative (disp/min_edge): min={fail_relative_disp.min():.2f}, "
          f"mean={fail_relative_disp.mean():.2f}, max={fail_relative_disp.max():.2f}")

    # Source element sizes
    print(f"    Source element min-edge: min={fail_sizes.min():.2e}, "
          f"mean={fail_sizes.mean():.2e}, max={fail_sizes.max():.2e}")

    # n_tests exhausted?
    print(f"    Point-in-tet tests used: min={fail_n_tests.min()}, "
          f"mean={fail_n_tests.mean():.0f}, max={fail_n_tests.max()}")
    n_hit_max = (fail_n_tests >= 590).sum()
    print(f"    Hit max_tests limit (>=590): {n_hit_max}/{n_fail}")

    # Brute-force verify: are these particles actually inside ANY element?
    print(f"\n    Brute-force CPU search on {min(n_fail, 100)} failure cases...")
    n_to_verify = min(n_fail, 100)
    verify_positions = fail_positions[:n_to_verify]

    t0 = time.time()
    bf_results = brute_force_search(
        verify_positions, connectivity, node_positions, M_inv_array, p0_array, tol=1e-6
    )
    elapsed = time.time() - t0

    bf_found = (bf_results >= 0).sum()
    bf_not_found = n_to_verify - bf_found
    print(f"    Brute-force: found={bf_found}/{n_to_verify} in {elapsed:.1f}s")
    print(f"    Truly outside mesh (not in any element): {bf_not_found}")
    print(f"    Inside mesh but L2 missed:               {bf_found}")

    if bf_found > 0:
        # For the ones brute-force found, what element did it find?
        bf_found_mask = bf_results >= 0
        bf_elem_ids = bf_results[bf_found_mask]
        bf_elem_sizes = element_sizes[bf_elem_ids]
        print(f"\n    Elements that brute-force found (L2 missed):")
        print(f"      Element sizes: min={bf_elem_sizes.min():.2e}, "
              f"mean={bf_elem_sizes.mean():.2e}, max={bf_elem_sizes.max():.2e}")

        # Check what octree cells these elements are registered in
        # vs what cells the query positions map to
        print(f"\n    Checking octree cell registration for missed elements...")

        # For each missed particle, check what octree level/cell its position maps to
        # vs what cells the true element is registered in
        cell_levels = np.array(mesh_octree_cells.cell_levels)
        cell_morton = np.array(mesh_octree_cells.cell_morton_codes)
        offsets = np.array(mesh_octree_cells.cell_to_elements_offsets)
        data = np.array(mesh_octree_cells.cell_to_elements_data)

        # Build element -> cells reverse map (which cells contain each element?)
        from collections import defaultdict
        elem_to_cells = defaultdict(list)
        for cell_idx in range(len(offsets) - 1):
            for j in range(offsets[cell_idx], offsets[cell_idx + 1]):
                elem_to_cells[int(data[j])].append(cell_idx)

        # For first 10 missed cases, print diagnostics
        n_diag = min(10, bf_found)
        bf_found_indices = np.where(bf_found_mask)[0]

        for i in range(n_diag):
            idx = bf_found_indices[i]
            pos = verify_positions[idx]
            true_elem = bf_results[idx]
            source_elem = fail_source_elems[idx]
            cells_for_true_elem = elem_to_cells.get(int(true_elem), [])
            cells_for_source = elem_to_cells.get(int(source_elem), [])

            true_elem_size = element_sizes[true_elem]
            source_elem_size = element_sizes[source_elem]

            # What levels are these cells at?
            true_levels = [int(cell_levels[c]) for c in cells_for_true_elem]
            source_levels = [int(cell_levels[c]) for c in cells_for_source]

            print(f"\n    Case {i+1}:")
            print(f"      Position: [{pos[0]:.8e}, {pos[1]:.8e}, {pos[2]:.8e}]")
            print(f"      Source elem: {source_elem} (min_edge={source_elem_size:.2e})")
            print(f"        Registered in {len(cells_for_source)} cells at levels {sorted(set(source_levels))}")
            print(f"      True elem (brute-force): {true_elem} (min_edge={true_elem_size:.2e})")
            print(f"        Registered in {len(cells_for_true_elem)} cells at levels {sorted(set(true_levels))}")

            # Check if source and true element share any cells
            common_cells = set(cells_for_true_elem) & set(cells_for_source)
            print(f"      Shared cells: {len(common_cells)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
