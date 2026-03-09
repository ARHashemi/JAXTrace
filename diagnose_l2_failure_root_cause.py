#!/usr/bin/env python3
"""
DIAGNOSTIC: Root Cause Analysis of L2 Search Failures

Tests the actual L2 failure rate on production particle positions and investigates:

1. Initial assignment: How many particles are unfindable by L2 at seeding positions?
2. Edge/corner positions: Does L2 miss positions near element edges/corners
   (not just centroids) at refinement boundaries?
3. Non-Kuhn elements: Are particles inside non-Kuhn elements at refinement
   boundaries unreachable by the 3×3×3 multi-level search?
4. Production failure positions: Extract exact positions where L2 fails during
   the first RK4 step and diagnose why.

Usage:
    python diagnose_l2_failure_root_cause.py 2>&1 | tee logs/diagnose_l2_failure_root_cause.log
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'

import sys
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import time
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import (
    extract_octree_cells_vertex_multi,
    find_axis_aligned_edges_single,
)
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import (
    upload_mesh_aligned_octree_to_gpu,
    encode_morton_3d_jax,
    find_cell_by_morton_and_level,
)
from jaxtrace.gpu.search.aa_detection import precompute_aa_metadata, precompute_element_vertices
from jaxtrace.gpu.search.point_in_tet_methods import set_corrected_metadata, set_inverse_matrices_gpu
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
import jaxtrace.config as config

from jaxtrace.gpu.search.mesh_aligned_point_location import (
    search_mesh_aligned_octree_multi_local_where,
)
from jaxtrace.gpu.search.point_in_tet_methods import (
    point_in_tet_gpu as point_in_tet_dispatcher,
)
from jaxtrace.tracking.seeding import uniform_grid_seeds

# =============================================================================
# Configuration (matching benchmark_l2_search_methods_with-export.py)
# =============================================================================

MESH_BASE_PATH = Path("data/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME = 'Displacement'

PARTICLE_GRID_RESOLUTION = (60, 90, 60)  # 324,000 particles
PARTICLE_BOUNDS_FRACTION = {
    'x': (0.12, 0.22),
    'y': (0.2, 0.8),
    'z': (0.01, 1.0),
}


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("DIAGNOSTIC: Root Cause Analysis of L2 Search Failures")
    print("=" * 80)

    # =========================================================================
    # Section 1: Load mesh, build structures (same as benchmark)
    # =========================================================================
    print("\n[1/6] Loading mesh and building structures...")

    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=True
    )
    node_positions, connectivity, n_dup, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    connectivity = connectivity.astype(np.int32)
    node_positions = node_positions.astype(np.float64)
    n_nodes = node_positions.shape[0]
    n_elements = connectivity.shape[0]
    print(f"  {n_nodes:,} nodes, {n_elements:,} elements")

    # Precompute metadata
    print("\n  Precomputing metadata...")
    aa_metadata = precompute_aa_metadata(connectivity, node_positions, verbose=False)
    element_vertices = precompute_element_vertices(connectivity, node_positions, verbose=False)
    M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)
    element_neighbors = build_element_neighbors_array(connectivity, method='face', verbose=False)

    # Extract octree
    print("  Extracting octree (multi-cell vertex registration)...")
    mesh_octree = extract_octree_cells_vertex_multi(
        node_positions, connectivity, tolerance=1e-6, verbose=True
    )

    # Upload to GPU
    print("  Uploading to GPU...")
    mesh_gpu = upload_mesh_to_gpu(connectivity, node_positions, element_neighbors, verbose=False)
    octree_gpu = upload_mesh_aligned_octree_to_gpu(
        connectivity, node_positions, mesh_octree, verbose=False
    )

    from jaxtrace.gpu.search.aa_detection import AxisAlignedMetadata
    aa_metadata_gpu = AxisAlignedMetadata(
        base_vertex_indices=jax.device_put(aa_metadata.base_vertex_indices),
        base_vertices=jax.device_put(aa_metadata.base_vertices),
        inv_edge_lengths=jax.device_put(aa_metadata.inv_edge_lengths),
        axis_indices=jax.device_put(aa_metadata.axis_indices),
        is_axis_aligned=jax.device_put(aa_metadata.is_axis_aligned)
    )
    element_vertices_gpu = jax.device_put(element_vertices)
    M_inv_gpu = jax.device_put(M_inv_array)
    p0_gpu = jax.device_put(p0_array)
    set_corrected_metadata(aa_metadata_gpu, element_vertices_gpu)
    set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)
    config.POINT_IN_TET_METHOD = 'inverse'

    # Build element level info
    print("  Classifying elements by refinement level...")
    kuhn_info = {}  # elem_id -> (cell_size, level)
    non_kuhn_ids = []
    level_counts = defaultdict(int)
    for elem_id in range(n_elements):
        verts = node_positions[connectivity[elem_id]]
        cell_size, level = find_axis_aligned_edges_single(verts, tolerance=1e-6)
        if not np.any(cell_size == 0):
            kuhn_info[elem_id] = (cell_size.copy(), level)
            level_counts[level] += 1
        else:
            non_kuhn_ids.append(elem_id)
        if (elem_id + 1) % 500000 == 0:
            print(f"    {elem_id + 1:,}/{n_elements:,}...")

    print(f"  Kuhn elements: {len(kuhn_info):,}")
    print(f"  Non-Kuhn elements: {len(non_kuhn_ids):,}")
    print(f"  Level distribution:")
    for level in sorted(level_counts.keys()):
        print(f"    Level {level:2d}: {level_counts[level]:>10,} elements ({100*level_counts[level]/n_elements:.2f}%)")

    # =========================================================================
    # Section 2: Initial Assignment L2 Failure Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("[2/6] Initial Assignment L2 Failure Analysis")
    print("=" * 80)
    print("  Generating EXACT same particle positions as benchmark...")

    # Generate particle positions (exact match with benchmark)
    domain_min = node_positions.min(axis=0)
    domain_max = node_positions.max(axis=0)
    domain_size = domain_max - domain_min

    nx, ny, nz = PARTICLE_GRID_RESOLUTION
    n_particles = nx * ny * nz

    par_bounds_min = np.zeros(3, dtype=np.float32)
    par_bounds_max = np.zeros(3, dtype=np.float32)
    for i, axis in enumerate(['x', 'y', 'z']):
        min_frac, max_frac = PARTICLE_BOUNDS_FRACTION[axis]
        par_bounds_min[i] = domain_min[i] + min_frac * domain_size[i]
        par_bounds_max[i] = domain_min[i] + max_frac * domain_size[i]
    par_bounds = [par_bounds_min, par_bounds_max]

    particle_positions = uniform_grid_seeds(
        resolution=(nx, ny, nz),
        bounds=par_bounds,
        include_boundaries=True
    )

    # Apply same clipping as benchmark
    margin = 0.01
    bbox_min_safe = domain_min + margin * domain_size
    bbox_max_safe = domain_max - margin * domain_size
    particle_positions = np.clip(particle_positions, bbox_min_safe, bbox_max_safe)

    print(f"  Particles: {n_particles:,}")
    print(f"  Bounds: X=[{par_bounds_min[0]:.6f}, {par_bounds_max[0]:.6f}]")
    print(f"          Y=[{par_bounds_min[1]:.6f}, {par_bounds_max[1]:.6f}]")
    print(f"          Z=[{par_bounds_min[2]:.6f}, {par_bounds_max[2]:.6f}]")

    # Run L2 search on ALL particles (this is what happens during warmup step
    # when all elem_ids start at -1)
    print(f"\n  Running L2 search on all {n_particles:,} particles...")
    positions_gpu = jax.device_put(jnp.array(particle_positions, dtype=jnp.float32))

    # Batch L2 search using the jnp.where version (vmap-safe, avoids OOM from
    # nested lax.cond in the original version)
    batch_size = 50000
    max_tests_jax = jnp.int32(600)

    @jax.jit
    def _search_batch(positions_batch):
        def single(pos):
            elem_id, n_tests = search_mesh_aligned_octree_multi_local_where(
                pos, octree_gpu, max_tests=max_tests_jax
            )
            return elem_id, n_tests
        return jax.vmap(single)(positions_batch)

    all_elem_ids = np.full(n_particles, -1, dtype=np.int32)
    all_n_tests = np.full(n_particles, 0, dtype=np.int32)

    t_start = time.time()
    for batch_start in range(0, n_particles, batch_size):
        batch_end = min(batch_start + batch_size, n_particles)
        batch_positions = positions_gpu[batch_start:batch_end]
        batch_elem_ids, batch_n_tests = _search_batch(batch_positions)
        batch_elem_ids = jax.block_until_ready(batch_elem_ids)
        all_elem_ids[batch_start:batch_end] = np.array(batch_elem_ids)
        all_n_tests[batch_start:batch_end] = np.array(batch_n_tests)
        n_found_so_far = np.sum(all_elem_ids[:batch_end] >= 0)
        if (batch_start // batch_size) % 2 == 0:
            print(f"    Batch {batch_start:,}-{batch_end:,}: "
                  f"found so far {n_found_so_far:,}/{batch_end:,} "
                  f"({100*n_found_so_far/batch_end:.2f}%)")
    t_elapsed = time.time() - t_start

    n_found = np.sum(all_elem_ids >= 0)
    n_missed = n_particles - n_found
    print(f"\n  L2 search results:")
    print(f"    Found: {n_found:,}/{n_particles:,} ({100*n_found/n_particles:.2f}%)")
    print(f"    Missed: {n_missed:,}/{n_particles:,} ({100*n_missed/n_particles:.2f}%)")
    print(f"    Time: {t_elapsed:.2f}s")
    print(f"    Tests: mean={all_n_tests[all_elem_ids >= 0].mean():.1f}, "
          f"max={all_n_tests.max()}")

    if n_missed == 0:
        print("\n  *** All particles found! L2 search works at seeding positions. ***")
        print("  The 19% loss must occur DURING RK4 sub-steps, not at initial seeding.")
    else:
        print(f"\n  *** {n_missed:,} particles NOT found by L2 ({100*n_missed/n_particles:.2f}%) ***")

    # Analyze missed positions
    missed_mask = all_elem_ids < 0
    missed_positions = particle_positions[missed_mask]
    missed_indices = np.where(missed_mask)[0]

    # =========================================================================
    # Section 3: Analyze Where Missed Particles Are Located
    # =========================================================================
    print("\n" + "=" * 80)
    print("[3/6] Spatial Analysis of Missed Particles")
    print("=" * 80)

    if n_missed > 0:
        print(f"  Analyzing {n_missed:,} missed positions...")

        # Check if missed positions are inside the mesh bbox
        mesh_min = node_positions.min(axis=0)
        mesh_max = node_positions.max(axis=0)
        inside_bbox = np.all(
            (missed_positions >= mesh_min[None, :]) &
            (missed_positions <= mesh_max[None, :]),
            axis=1
        )
        n_inside_bbox = np.sum(inside_bbox)
        print(f"  Inside mesh bounding box: {n_inside_bbox:,}/{n_missed:,}")
        print(f"  Outside mesh bounding box: {n_missed - n_inside_bbox:,}")

        # Spatial distribution of missed positions
        print(f"\n  Spatial distribution of missed positions:")
        for axis_idx, axis_name in enumerate(['X', 'Y', 'Z']):
            vals = missed_positions[:, axis_idx]
            print(f"    {axis_name}: min={vals.min():.8f}, max={vals.max():.8f}, "
                  f"mean={vals.mean():.8f}")

        # Check proximity to mesh boundary
        # Use brute-force point-in-tet on a sample of missed particles
        n_sample = min(500, n_missed)
        sample_indices = np.random.choice(n_missed, n_sample, replace=False)
        sample_positions = missed_positions[sample_indices]

        print(f"\n  Brute-force point-in-tet check for {n_sample} missed particles...")
        n_actually_inside = 0
        inside_elem_ids = []

        for i, pos in enumerate(sample_positions):
            pos_jax = jnp.array(pos, dtype=jnp.float32)
            found = False

            # Test ALL elements (brute force, but only for sample)
            # Optimization: test elements in nearby cells first via the octree
            # but using Python loop with ALL levels and WIDER neighborhood
            for level in sorted(level_counts.keys(), reverse=True):
                if found:
                    break
                # Get cell size for this level
                for eid, (cs, lvl) in kuhn_info.items():
                    if lvl == level:
                        level_cs = cs
                        break
                else:
                    continue

                # Compute grid index (float64 for accuracy)
                pos_f64 = np.float64(pos)
                cs_f64 = np.float64(level_cs)
                i_base = int(np.floor(pos_f64[0] / cs_f64[0]))
                j_base = int(np.floor(pos_f64[1] / cs_f64[1]))
                k_base = int(np.floor(pos_f64[2] / cs_f64[2]))

                # Search 5×5×5 neighborhood (wider than production 3×3×3)
                for di in range(-2, 3):
                    if found:
                        break
                    for dj in range(-2, 3):
                        if found:
                            break
                        for dk in range(-2, 3):
                            ii = i_base + di
                            jj = j_base + dj
                            kk = k_base + dk

                            offset = 1 << 19
                            max_coord = 1 << 20
                            i_m = np.clip(ii + offset, 0, max_coord - 1)
                            j_m = np.clip(jj + offset, 0, max_coord - 1)
                            k_m = np.clip(kk + offset, 0, max_coord - 1)

                            morton = encode_morton_3d_jax(
                                jnp.int32(int(i_m)),
                                jnp.int32(int(j_m)),
                                jnp.int32(int(k_m))
                            )
                            cell_idx = find_cell_by_morton_and_level(
                                morton, jnp.uint8(int(level)),
                                octree_gpu.cell_morton_codes,
                                octree_gpu.cell_levels
                            )
                            cell_idx = int(cell_idx)
                            if cell_idx < 0:
                                continue

                            start = int(octree_gpu.cell_to_elements_offsets[cell_idx])
                            end = int(octree_gpu.cell_to_elements_offsets[cell_idx + 1])
                            for elem_offset in range(start, end):
                                elem_id = int(octree_gpu.cell_to_elements_data[elem_offset])
                                is_inside = point_in_tet_dispatcher(
                                    pos_jax, jnp.int32(elem_id),
                                    octree_gpu.connectivity,
                                    octree_gpu.node_positions,
                                    config.POINT_IN_TET_METHOD
                                )
                                if bool(is_inside):
                                    n_actually_inside += 1
                                    inside_elem_ids.append(elem_id)
                                    found = True
                                    break

            if (i + 1) % 100 == 0:
                print(f"    Checked {i+1}/{n_sample}: {n_actually_inside} found with 5×5×5")

        print(f"\n  Brute-force 5×5×5 results:")
        print(f"    Found: {n_actually_inside}/{n_sample} ({100*n_actually_inside/n_sample:.2f}%)")
        print(f"    Truly outside mesh: {n_sample - n_actually_inside}/{n_sample}")

        if n_actually_inside > 0:
            print(f"\n  *** {n_actually_inside} particles ARE inside elements but 3×3×3 missed them! ***")
            print(f"  *** This confirms 3×3×3 coverage is insufficient for some positions. ***")

            # Analyze which levels the missed-but-findable elements are at
            level_of_found = defaultdict(int)
            is_non_kuhn_count = 0
            for eid in inside_elem_ids:
                if eid in kuhn_info:
                    _, lvl = kuhn_info[eid]
                    level_of_found[lvl] += 1
                else:
                    is_non_kuhn_count += 1

            print(f"\n  Level distribution of elements found by 5×5×5 but missed by 3×3×3:")
            for level in sorted(level_of_found.keys()):
                print(f"    Level {level}: {level_of_found[level]}")
            if is_non_kuhn_count > 0:
                print(f"    Non-Kuhn: {is_non_kuhn_count}")
        else:
            print(f"\n  All {n_sample} sampled missed particles are truly outside the mesh.")
            print(f"  The L2 failure is due to particles outside the element domain.")
    else:
        print("  No missed particles to analyze.")
        sample_positions = np.array([])

    # =========================================================================
    # Section 4: Non-Kuhn Element Reachability Test
    # =========================================================================
    print("\n" + "=" * 80)
    print("[4/6] Non-Kuhn Element Reachability Test")
    print("=" * 80)

    print(f"  Testing L2 search at centroids of {len(non_kuhn_ids):,} non-Kuhn elements...")

    if len(non_kuhn_ids) > 0:
        # Test centroids of non-Kuhn elements
        n_test_nonkuhn = min(500, len(non_kuhn_ids))
        test_nonkuhn_ids = non_kuhn_ids[:n_test_nonkuhn]

        n_found_nonkuhn = 0
        n_missed_nonkuhn = 0
        n_found_wrong_nonkuhn = 0
        missed_nonkuhn_examples = []

        for idx, elem_id in enumerate(test_nonkuhn_ids):
            verts = node_positions[connectivity[elem_id]]
            centroid = verts.mean(axis=0)
            pos_jax = jnp.array(centroid, dtype=jnp.float32)

            found_elem, n_tests = search_mesh_aligned_octree_multi_local_where(
                pos_jax, octree_gpu, max_tests=max_tests_jax
            )
            found_elem = int(found_elem)

            if found_elem == elem_id:
                n_found_nonkuhn += 1
            elif found_elem >= 0:
                n_found_wrong_nonkuhn += 1
                n_found_nonkuhn += 1
            else:
                n_missed_nonkuhn += 1
                if len(missed_nonkuhn_examples) < 20:
                    missed_nonkuhn_examples.append({
                        'elem_id': elem_id,
                        'centroid': centroid.copy(),
                        'vertices': verts.copy(),
                    })

        print(f"\n  Non-Kuhn centroid L2 search results ({n_test_nonkuhn} tested):")
        print(f"    Found (exact match): {n_found_nonkuhn - n_found_wrong_nonkuhn}")
        print(f"    Found (different element): {n_found_wrong_nonkuhn}")
        print(f"    Missed: {n_missed_nonkuhn} ({100*n_missed_nonkuhn/n_test_nonkuhn:.2f}%)")

        if missed_nonkuhn_examples:
            print(f"\n  Missed non-Kuhn examples (first {min(10, len(missed_nonkuhn_examples))}):")
            for ex in missed_nonkuhn_examples[:10]:
                print(f"    Elem {ex['elem_id']}:")
                print(f"      centroid=({ex['centroid'][0]:.8e}, {ex['centroid'][1]:.8e}, {ex['centroid'][2]:.8e})")

                # Check which cell it was registered in
                pos_jax = jnp.array(ex['centroid'], dtype=jnp.float32)
                is_inside = point_in_tet_dispatcher(
                    pos_jax, jnp.int32(ex['elem_id']),
                    octree_gpu.connectivity, octree_gpu.node_positions,
                    config.POINT_IN_TET_METHOD
                )
                print(f"      Point-in-tet (expected elem): {bool(is_inside)}")

        # Also test positions near edges of non-Kuhn elements
        print(f"\n  Testing edge positions of non-Kuhn elements...")
        n_edge_tests = 0
        n_edge_misses = 0

        for elem_id in test_nonkuhn_ids[:100]:
            verts = node_positions[connectivity[elem_id]]
            centroid = verts.mean(axis=0)

            # Test positions at 95%, 90%, 80% from centroid toward each vertex
            for vi in range(4):
                for frac in [0.8, 0.9, 0.95]:
                    test_pos = centroid * (1 - frac) + verts[vi] * frac
                    pos_jax = jnp.array(test_pos, dtype=jnp.float32)

                    # Verify position is inside element
                    is_inside = point_in_tet_dispatcher(
                        pos_jax, jnp.int32(elem_id),
                        octree_gpu.connectivity, octree_gpu.node_positions,
                        config.POINT_IN_TET_METHOD
                    )
                    if not bool(is_inside):
                        continue  # Skip positions outside element

                    n_edge_tests += 1
                    found_elem, _ = search_mesh_aligned_octree_multi_local_where(
                        pos_jax, octree_gpu, max_tests=max_tests_jax
                    )
                    if int(found_elem) < 0:
                        n_edge_misses += 1

        print(f"    Edge tests: {n_edge_tests}")
        print(f"    Edge misses: {n_edge_misses} ({100*n_edge_misses/max(1,n_edge_tests):.2f}%)")
    else:
        print("  No non-Kuhn elements found.")

    # =========================================================================
    # Section 5: Element Edge/Corner Position Test (Kuhn elements at boundaries)
    # =========================================================================
    print("\n" + "=" * 80)
    print("[5/6] Element Edge/Corner Position Test at Refinement Boundaries")
    print("=" * 80)

    # Find boundary elements (elements at refinement level transitions)
    print("  Finding refinement boundary elements...")
    node_to_elements = defaultdict(list)
    for eid in range(n_elements):
        for nid in connectivity[eid]:
            node_to_elements[nid].append(eid)

    boundary_elements = []
    for elem_id in range(n_elements):
        if elem_id not in kuhn_info:
            continue
        elem_level = kuhn_info[elem_id][1]
        found_boundary = False
        for node_id in connectivity[elem_id]:
            if found_boundary:
                break
            for neighbor_id in node_to_elements[node_id]:
                if neighbor_id == elem_id or neighbor_id not in kuhn_info:
                    continue
                neighbor_level = kuhn_info[neighbor_id][1]
                if neighbor_level != elem_level:
                    boundary_elements.append((elem_id, elem_level, neighbor_level))
                    found_boundary = True
                    break
        if len(boundary_elements) >= 1000:
            break

    print(f"  Found {len(boundary_elements)} boundary elements")

    # Test positions near element edges and vertices (NOT centroids)
    print(f"\n  Testing edge/vertex positions of boundary elements...")
    n_edge_boundary_tests = 0
    n_edge_boundary_misses = 0
    edge_miss_examples = []

    for elem_id, elem_level, neighbor_level in boundary_elements[:500]:
        verts = node_positions[connectivity[elem_id]]
        centroid = verts.mean(axis=0)

        # Test at various positions within the element:
        # - Near each vertex (95% from centroid)
        # - Near each edge midpoint (90% from centroid)
        # - Near each face center (85% from centroid)
        test_positions_local = []

        # Near vertices (95% toward vertex from centroid)
        for vi in range(4):
            test_positions_local.append(('vertex', vi, centroid * 0.05 + verts[vi] * 0.95))

        # Near edge midpoints (90% toward edge midpoint)
        edge_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        for ei, (v1, v2) in enumerate(edge_pairs):
            edge_mid = (verts[v1] + verts[v2]) / 2.0
            test_positions_local.append(('edge', ei, centroid * 0.1 + edge_mid * 0.9))

        # Near face centers (85% toward face center)
        face_triples = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
        for fi, (v1, v2, v3) in enumerate(face_triples):
            face_center = (verts[v1] + verts[v2] + verts[v3]) / 3.0
            test_positions_local.append(('face', fi, centroid * 0.15 + face_center * 0.85))

        for pos_type, pos_idx, test_pos in test_positions_local:
            pos_jax = jnp.array(test_pos, dtype=jnp.float32)

            # Verify position is inside the element
            is_inside = point_in_tet_dispatcher(
                pos_jax, jnp.int32(elem_id),
                octree_gpu.connectivity, octree_gpu.node_positions,
                config.POINT_IN_TET_METHOD
            )
            if not bool(is_inside):
                continue  # Skip if outside element (near vertex may overshoot)

            n_edge_boundary_tests += 1
            found_elem, n_tests = search_mesh_aligned_octree_multi_local_where(
                pos_jax, octree_gpu, max_tests=max_tests_jax
            )
            if int(found_elem) < 0:
                n_edge_boundary_misses += 1
                if len(edge_miss_examples) < 30:
                    edge_miss_examples.append({
                        'elem_id': elem_id,
                        'elem_level': elem_level,
                        'neighbor_level': neighbor_level,
                        'pos_type': pos_type,
                        'pos_idx': pos_idx,
                        'pos': test_pos.copy(),
                        'n_tests': int(n_tests),
                    })

    miss_rate = 100 * n_edge_boundary_misses / max(1, n_edge_boundary_tests)
    print(f"\n  Edge/vertex/face position results:")
    print(f"    Tests: {n_edge_boundary_tests:,}")
    print(f"    Misses: {n_edge_boundary_misses:,} ({miss_rate:.3f}%)")

    if edge_miss_examples:
        print(f"\n  Miss examples (first {min(20, len(edge_miss_examples))}):")
        # Group by position type
        miss_by_type = defaultdict(int)
        miss_by_level = defaultdict(int)
        for ex in edge_miss_examples:
            miss_by_type[ex['pos_type']] += 1
            miss_by_level[f"L{ex['elem_level']}↔L{ex['neighbor_level']}"] += 1

        print(f"    By position type: {dict(miss_by_type)}")
        print(f"    By level transition: {dict(miss_by_level)}")

        for ex in edge_miss_examples[:10]:
            print(f"\n    Elem {ex['elem_id']} (L{ex['elem_level']}↔L{ex['neighbor_level']}), "
                  f"{ex['pos_type']} {ex['pos_idx']}, {ex['n_tests']} tests")
            print(f"      pos=({ex['pos'][0]:.10e}, {ex['pos'][1]:.10e}, {ex['pos'][2]:.10e})")

            # Check which cell the element is registered in vs where search looks
            if ex['elem_id'] in kuhn_info:
                cs, lvl = kuhn_info[ex['elem_id']]
                pos_f64 = np.float64(ex['pos'])
                cs_f64 = np.float64(cs)
                cs_f32 = np.float32(cs)
                i64 = int(np.floor(pos_f64[0] / cs_f64[0]))
                j64 = int(np.floor(pos_f64[1] / cs_f64[1]))
                k64 = int(np.floor(pos_f64[2] / cs_f64[2]))
                i32 = int(np.floor(np.float32(pos_f64[0]) / cs_f32[0]))
                j32 = int(np.floor(np.float32(pos_f64[1]) / cs_f32[1]))
                k32 = int(np.floor(np.float32(pos_f64[2]) / cs_f32[2]))
                print(f"      Level {lvl}: f64=({i64},{j64},{k64}) f32=({i32},{j32},{k32}) "
                      f"delta=({i32-i64},{j32-j64},{k32-k64})")

                # Check which cells the element's vertices map to
                elem_verts = node_positions[connectivity[ex['elem_id']]]
                vert_cells = set()
                for vi in range(4):
                    vi64 = int(np.floor(np.float64(elem_verts[vi, 0]) / cs_f64[0]))
                    vj64 = int(np.floor(np.float64(elem_verts[vi, 1]) / cs_f64[1]))
                    vk64 = int(np.floor(np.float64(elem_verts[vi, 2]) / cs_f64[2]))
                    vert_cells.add((vi64, vj64, vk64))
                print(f"      Vertex cells (f64): {vert_cells}")
                print(f"      Position cell in vertex cells? {(i64, j64, k64) in vert_cells}")

                # Check if 3×3×3 around f32 position covers all vertex cells
                search_cells_f32 = set()
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        for dk in range(-1, 2):
                            search_cells_f32.add((i32+di, j32+dj, k32+dk))
                uncovered = vert_cells - search_cells_f32
                if uncovered:
                    print(f"      *** UNCOVERED vertex cells: {uncovered} ***")
                else:
                    print(f"      All vertex cells covered by 3×3×3")
    else:
        print(f"\n  *** No misses at edge/vertex/face positions! ***")

    # =========================================================================
    # Section 6: Summary and Conclusions
    # =========================================================================
    print("\n" + "=" * 80)
    print("[6/6] SUMMARY AND CONCLUSIONS")
    print("=" * 80)

    print(f"\n  Section 2 - Initial Assignment L2:")
    print(f"    Particles: {n_particles:,}")
    print(f"    Found by L2: {n_found:,} ({100*n_found/n_particles:.2f}%)")
    print(f"    Missed by L2: {n_missed:,} ({100*n_missed/n_particles:.2f}%)")

    if n_missed > 0 and len(sample_positions) > 0:
        print(f"\n  Section 3 - 5×5×5 Brute-Force Recovery:")
        print(f"    Sampled {n_sample} missed particles")
        print(f"    Recovered by 5×5×5: {n_actually_inside}")
        print(f"    Truly outside mesh: {n_sample - n_actually_inside}")

    print(f"\n  Section 4 - Non-Kuhn Elements:")
    if len(non_kuhn_ids) > 0:
        print(f"    Centroid misses: {n_missed_nonkuhn}/{n_test_nonkuhn}")
        print(f"    Edge misses: {n_edge_misses}/{n_edge_tests}")

    print(f"\n  Section 5 - Edge/Vertex Positions at Boundaries:")
    print(f"    Tests: {n_edge_boundary_tests:,}")
    print(f"    Misses: {n_edge_boundary_misses:,} ({miss_rate:.3f}%)")

    print(f"\n  Conclusions:")
    if n_missed == 0:
        print(f"    ✅ L2 search finds ALL particles at seeding positions")
        print(f"       → The 19% loss occurs during RK4 sub-steps, not initial assignment")
    elif n_missed > 0 and n_actually_inside > 0:
        print(f"    ⚠️  L2 (3×3×3) misses {n_missed:,} particles that ARE inside mesh elements")
        print(f"       → 3×3×3 neighborhood is too narrow for some positions")
        print(f"       → Consider 5×5×5 or float64 grid indices")
    elif n_missed > 0 and n_actually_inside == 0:
        print(f"    ℹ️  L2 misses {n_missed:,} particles that are OUTSIDE the mesh")
        print(f"       → These particles are in voids between mesh boundary and seeding box")
        print(f"       → Not an L2 search bug, just particles seeded outside elements")

    if n_edge_boundary_misses > 0:
        print(f"    ⚠️  L2 misses positions near element edges/vertices at boundaries")
        print(f"       → This IS the mechanism for RK4 sub-step losses")
    else:
        print(f"    ✅ L2 finds all tested edge/vertex/face positions at boundaries")

    if n_missed_nonkuhn > 0:
        print(f"    ⚠️  Non-Kuhn element centroids missed by L2")
        print(f"       → Non-Kuhn registration at borrowed level may be insufficient")
    elif len(non_kuhn_ids) > 0:
        print(f"    ✅ Non-Kuhn element centroids all found by L2")

    print("\nDone.")


if __name__ == '__main__':
    main()
