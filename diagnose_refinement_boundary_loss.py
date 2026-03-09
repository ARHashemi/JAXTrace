#!/usr/bin/env python3
"""
DIAGNOSTIC: Particle Loss at Refinement Boundaries

Tests two hypotheses for particle loss at coarse:fine element boundaries:

1. Float32 vs Float64 Precision Mismatch
   - Element registration uses float64 grid index computation (CPU)
   - Search uses float32 grid index computation (GPU)
   - At cell boundaries, floor() can produce different results

2. Cross-Level Element Reachability
   - At level 14/13 boundaries, a particle may be inside a level-13 element
   - The 3×3×3 search at level 13 must find the correct cell
   - If float32 precision shifts the base cell, the 3×3×3 may miss it

Test Plan:
  Section 1: Load mesh, extract octree, identify refinement boundary positions
  Section 2: Float32 vs Float64 grid index comparison at all refinement levels
  Section 3: Cross-level element reachability test
  Section 4: Targeted L2 search at boundary positions
  Section 5: RK4 sub-step replay at boundary positions
  Section 6: Float64 L2 search comparison
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
    search_mesh_aligned_octree_multi_local,
    search_mesh_aligned_octree_multi_local_where,
)
from jaxtrace.gpu.search.point_in_tet_methods import (
    point_in_tet_gpu as point_in_tet_dispatcher,
)

# =============================================================================
# Configuration
# =============================================================================

MESH_BASE_PATH = Path("data/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME = 'Displacement'

# =============================================================================
# Helper Functions
# =============================================================================

def compute_grid_index_f64(pos, cell_size, morton_offset=1 << 19, morton_max_coord=1 << 20):
    """Compute grid index using float64 (matching registration)."""
    pos = np.float64(pos)
    cell_size = np.float64(cell_size)
    i = int(np.floor(pos[0] / cell_size[0]))
    j = int(np.floor(pos[1] / cell_size[1]))
    k = int(np.floor(pos[2] / cell_size[2]))
    return i, j, k


def compute_grid_index_f32(pos, cell_size, morton_offset=1 << 19, morton_max_coord=1 << 20):
    """Compute grid index using float32 (matching GPU search)."""
    pos = np.float32(pos)
    cell_size = np.float32(cell_size)
    i = int(np.floor(np.float32(pos[0]) / np.float32(cell_size[0])))
    j = int(np.floor(np.float32(pos[1]) / np.float32(cell_size[1])))
    k = int(np.floor(np.float32(pos[2]) / np.float32(cell_size[2])))
    return i, j, k


def find_boundary_elements(connectivity, node_positions, kuhn_info, n_sample=1000):
    """Find elements at refinement level boundaries.

    Returns list of (elem_id, elem_level, neighbor_level, boundary_type) tuples
    where boundary_type is 'fine_to_coarse' or 'coarse_to_fine'.
    """
    n_elements = connectivity.shape[0]

    # Build node-to-element map
    node_to_elements = defaultdict(list)
    for elem_id in range(n_elements):
        for node_id in connectivity[elem_id]:
            node_to_elements[node_id].append(elem_id)

    boundary_elements = []
    checked = 0

    for elem_id in range(n_elements):
        if elem_id not in kuhn_info:
            continue

        elem_level = kuhn_info[elem_id][1]

        # Check face neighbors for different refinement level
        for node_id in connectivity[elem_id]:
            for neighbor_id in node_to_elements[node_id]:
                if neighbor_id == elem_id:
                    continue
                if neighbor_id not in kuhn_info:
                    continue
                neighbor_level = kuhn_info[neighbor_id][1]
                if neighbor_level != elem_level:
                    btype = 'fine_to_coarse' if elem_level > neighbor_level else 'coarse_to_fine'
                    boundary_elements.append((elem_id, elem_level, neighbor_level, btype))
                    break
            else:
                continue
            break

        checked += 1
        if len(boundary_elements) >= n_sample:
            break

    return boundary_elements


def generate_boundary_positions(connectivity, node_positions, boundary_elements, n_per_element=5):
    """Generate test positions near refinement boundaries.

    For each boundary element, generate positions:
    - At the centroid
    - Near each face that borders a different-level element
    - At vertex positions (exact boundary)
    """
    positions = []
    metadata = []

    for elem_id, elem_level, neighbor_level, btype in boundary_elements:
        verts = node_positions[connectivity[elem_id]]
        centroid = verts.mean(axis=0)

        # Centroid
        positions.append(centroid)
        metadata.append({
            'elem_id': elem_id, 'level': elem_level,
            'neighbor_level': neighbor_level, 'type': btype,
            'pos_type': 'centroid'
        })

        # Points near each vertex (95% toward centroid from vertex)
        for vi in range(4):
            near_vertex = verts[vi] * 0.05 + centroid * 0.95
            positions.append(near_vertex)
            metadata.append({
                'elem_id': elem_id, 'level': elem_level,
                'neighbor_level': neighbor_level, 'type': btype,
                'pos_type': f'near_v{vi}'
            })

    return np.array(positions, dtype=np.float64), metadata


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("DIAGNOSTIC: Particle Loss at Refinement Boundaries")
    print("=" * 80)

    # =========================================================================
    # Section 1: Load mesh and build structures
    # =========================================================================
    print("\n[1/7] Loading mesh...")
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

    # Precompute search structures
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

    # =========================================================================
    # Section 2: Identify refinement boundary elements
    # =========================================================================
    print("\n[2/7] Identifying refinement boundary elements...")

    # Build Kuhn element info (level per element)
    print("  Classifying elements by refinement level...")
    kuhn_info = {}  # elem_id -> (cell_size, level)
    level_counts = defaultdict(int)
    for elem_id in range(n_elements):
        verts = node_positions[connectivity[elem_id]]
        cell_size, level = find_axis_aligned_edges_single(verts, tolerance=1e-6)
        if not np.any(cell_size == 0):
            kuhn_info[elem_id] = (cell_size.copy(), level)
            level_counts[level] += 1
        if (elem_id + 1) % 500000 == 0:
            print(f"    {elem_id + 1:,}/{n_elements:,}...")

    print(f"  Level distribution:")
    for level in sorted(level_counts.keys()):
        print(f"    Level {level:2d}: {level_counts[level]:>10,} elements ({100*level_counts[level]/n_elements:.2f}%)")

    # Find boundary elements
    print(f"\n  Finding refinement boundary elements...")
    boundary_elements = find_boundary_elements(connectivity, node_positions, kuhn_info, n_sample=500)
    print(f"  Found {len(boundary_elements)} boundary elements")

    # Categorize
    boundary_types = defaultdict(int)
    for _, elem_level, neighbor_level, btype in boundary_elements:
        key = f"L{elem_level}↔L{neighbor_level}"
        boundary_types[key] += 1
    print(f"  Boundary types:")
    for key, count in sorted(boundary_types.items()):
        print(f"    {key}: {count}")

    # =========================================================================
    # Section 3: Float32 vs Float64 Grid Index Comparison
    # =========================================================================
    print("\n[3/7] Float32 vs Float64 grid index comparison...")

    # Get cell sizes per level from the octree
    level_cell_sizes = {}
    for level in sorted(level_counts.keys()):
        # Find first element at this level
        for elem_id, (cs, lvl) in kuhn_info.items():
            if lvl == level:
                level_cell_sizes[level] = cs.copy()
                break

    print(f"\n  Cell sizes per level:")
    for level in sorted(level_cell_sizes.keys()):
        cs = level_cell_sizes[level]
        print(f"    Level {level:2d}: X={cs[0]:.10e}, Y={cs[1]:.10e}, Z={cs[2]:.10e}")

    # Generate test positions at boundary elements
    print(f"\n  Generating boundary test positions...")
    test_positions, test_metadata = generate_boundary_positions(
        connectivity, node_positions, boundary_elements[:100], n_per_element=5
    )
    print(f"  Generated {len(test_positions)} test positions")

    # Compare grid indices
    n_mismatches_per_level = defaultdict(lambda: {'total': 0, 'x': 0, 'y': 0, 'z': 0})
    n_tests_per_level = defaultdict(int)

    total_mismatches = 0
    mismatch_examples = []

    for pos_idx, (pos, meta) in enumerate(zip(test_positions, test_metadata)):
        elem_level = meta['level']

        # Test at THIS element's level AND neighboring level
        levels_to_test = set([elem_level, meta['neighbor_level']])

        for level in levels_to_test:
            if level not in level_cell_sizes:
                continue

            cs = level_cell_sizes[level]
            n_tests_per_level[level] += 1

            i64, j64, k64 = compute_grid_index_f64(pos, cs)
            i32, j32, k32 = compute_grid_index_f32(pos, cs)

            mismatch_x = i64 != i32
            mismatch_y = j64 != j32
            mismatch_z = k64 != k32
            any_mismatch = mismatch_x or mismatch_y or mismatch_z

            n_mismatches_per_level[level]['total'] += int(any_mismatch)
            n_mismatches_per_level[level]['x'] += int(mismatch_x)
            n_mismatches_per_level[level]['y'] += int(mismatch_y)
            n_mismatches_per_level[level]['z'] += int(mismatch_z)

            if any_mismatch:
                total_mismatches += 1
                if len(mismatch_examples) < 20:
                    mismatch_examples.append({
                        'pos': pos.copy(), 'level': level,
                        'i64': (i64, j64, k64), 'i32': (i32, j32, k32),
                        'cell_size': cs.copy(), 'meta': meta,
                    })

    print(f"\n  Grid index mismatch results (f32 vs f64):")
    print(f"  {'Level':>6s}  {'Tests':>8s}  {'Mismatches':>10s}  {'Rate':>8s}  {'X':>5s}  {'Y':>5s}  {'Z':>5s}")
    print(f"  {'-'*55}")
    for level in sorted(n_tests_per_level.keys()):
        n = n_tests_per_level[level]
        m = n_mismatches_per_level[level]
        rate = 100 * m['total'] / n if n > 0 else 0
        print(f"  {level:6d}  {n:8d}  {m['total']:10d}  {rate:7.2f}%  {m['x']:5d}  {m['y']:5d}  {m['z']:5d}")

    print(f"\n  Total mismatches: {total_mismatches} across {len(test_positions)} positions × levels")

    if mismatch_examples:
        print(f"\n  Example mismatches (first {len(mismatch_examples)}):")
        for ex in mismatch_examples[:10]:
            print(f"    Level {ex['level']}, {ex['meta']['pos_type']} of elem {ex['meta']['elem_id']}")
            print(f"      pos = ({ex['pos'][0]:.10e}, {ex['pos'][1]:.10e}, {ex['pos'][2]:.10e})")
            print(f"      cell_size = ({ex['cell_size'][0]:.10e}, {ex['cell_size'][1]:.10e}, {ex['cell_size'][2]:.10e})")
            print(f"      f64 indices: ({ex['i64'][0]}, {ex['i64'][1]}, {ex['i64'][2]})")
            print(f"      f32 indices: ({ex['i32'][0]}, {ex['i32'][1]}, {ex['i32'][2]})")
            delta = tuple(a - b for a, b in zip(ex['i32'], ex['i64']))
            print(f"      delta (f32-f64): ({delta[0]}, {delta[1]}, {delta[2]})")
    else:
        print(f"\n  *** No mismatches found in boundary positions! ***")
        print(f"  This suggests float32 precision is NOT the primary issue for these positions.")

    # =========================================================================
    # Section 4: Systematic Float32 Boundary Sweep
    # =========================================================================
    print("\n[4/7] Systematic float32 boundary sweep...")
    print("  Testing positions at exact cell boundaries along each axis.")

    # For each level, generate positions that fall exactly on cell boundaries
    boundary_mismatches = 0
    boundary_tests = 0

    for level in sorted(level_cell_sizes.keys()):
        cs = level_cell_sizes[level]
        cs_f32 = np.float32(cs)
        cs_f64 = np.float64(cs)

        level_mismatches = 0
        level_tests = 0

        # Sweep along each axis at multiple positions
        for axis in range(3):
            axis_name = 'XYZ'[axis]
            # Generate positions near cell boundaries
            # Cell boundary at: n * cell_size[axis] for integer n
            for n in range(-200, 201):
                # Exact boundary position (float64)
                boundary_pos = np.float64(n) * cs_f64[axis]

                # Test positions slightly above and below boundary
                for eps_mult in [-1e-8, -1e-10, 0, 1e-10, 1e-8]:
                    test_val = boundary_pos + eps_mult * cs_f64[axis]

                    # Float64 index
                    idx_f64 = int(np.floor(np.float64(test_val) / cs_f64[axis]))
                    # Float32 index
                    idx_f32 = int(np.floor(np.float32(test_val) / cs_f32[axis]))

                    level_tests += 1
                    if idx_f64 != idx_f32:
                        level_mismatches += 1
                        boundary_mismatches += 1

        boundary_tests += level_tests
        rate = 100 * level_mismatches / level_tests if level_tests > 0 else 0
        print(f"    Level {level:2d}: {level_mismatches:6d}/{level_tests:6d} mismatches ({rate:.2f}%)")

    print(f"\n  Total boundary sweep: {boundary_mismatches:,}/{boundary_tests:,} mismatches "
          f"({100*boundary_mismatches/boundary_tests:.2f}%)")

    # =========================================================================
    # Section 5: Cross-Level Element Reachability
    # =========================================================================
    print("\n[5/7] Cross-level element reachability test...")
    print("  For each boundary element, test whether L2 search finds the correct element")
    print("  using both float32 and float64 grid indices.")

    # Use GPU L2 search (float32 - current production)
    n_found_gpu = 0
    n_missed_gpu = 0
    n_found_wrong_gpu = 0
    missed_examples = []

    # Test centroid of each boundary element
    boundary_centroids = []
    boundary_elem_ids = []
    for elem_id, elem_level, neighbor_level, btype in boundary_elements[:200]:
        verts = node_positions[connectivity[elem_id]]
        centroid = verts.mean(axis=0)
        boundary_centroids.append(centroid)
        boundary_elem_ids.append(elem_id)

    print(f"  Testing {len(boundary_centroids)} boundary element centroids...")

    for idx, (pos, expected_elem) in enumerate(zip(boundary_centroids, boundary_elem_ids)):
        # GPU L2 search (float32)
        pos_jax = jnp.array(pos, dtype=jnp.float32)
        found_elem, n_tests = search_mesh_aligned_octree_multi_local(
            pos_jax, octree_gpu, max_tests=jnp.int32(600)
        )
        found_elem = int(found_elem)

        if found_elem == expected_elem:
            n_found_gpu += 1
        elif found_elem >= 0:
            # Found a different element - check if position is really inside it
            n_found_wrong_gpu += 1
            n_found_gpu += 1  # Still found something
        else:
            n_missed_gpu += 1
            if len(missed_examples) < 20:
                missed_examples.append({
                    'pos': pos.copy(),
                    'expected_elem': expected_elem,
                    'elem_level': boundary_elements[idx][1],
                    'neighbor_level': boundary_elements[idx][2],
                })

    print(f"\n  GPU L2 search at boundary centroids:")
    print(f"    Found (correct element): {n_found_gpu - n_found_wrong_gpu}")
    print(f"    Found (different element): {n_found_wrong_gpu}")
    print(f"    NOT FOUND: {n_missed_gpu} *** THIS IS THE BUG ***")

    if missed_examples:
        print(f"\n  Missed examples:")
        for ex in missed_examples[:10]:
            print(f"    Elem {ex['expected_elem']} (L{ex['elem_level']}↔L{ex['neighbor_level']}): "
                  f"pos=({ex['pos'][0]:.8e}, {ex['pos'][1]:.8e}, {ex['pos'][2]:.8e})")

    # =========================================================================
    # Section 6: Test positions between boundary element pairs
    # =========================================================================
    print("\n[6/7] Testing positions BETWEEN boundary element pairs...")
    print("  Generate positions that cross from fine to coarse elements.")

    cross_boundary_tests = 0
    cross_boundary_misses = 0
    cross_boundary_examples = []

    for elem_id, elem_level, neighbor_level, btype in boundary_elements[:100]:
        if btype != 'fine_to_coarse':
            continue

        verts = node_positions[connectivity[elem_id]]
        centroid = verts.mean(axis=0)

        # Generate positions stepping from centroid toward each vertex
        # (vertices are at cell boundaries)
        for vi in range(4):
            for frac in [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]:
                test_pos = centroid * (1 - frac) + verts[vi] * frac
                cross_boundary_tests += 1

                pos_jax = jnp.array(test_pos, dtype=jnp.float32)
                found_elem, n_tests = search_mesh_aligned_octree_multi_local(
                    pos_jax, octree_gpu, max_tests=jnp.int32(600)
                )
                found_elem = int(found_elem)

                if found_elem < 0:
                    cross_boundary_misses += 1
                    if len(cross_boundary_examples) < 20:
                        cross_boundary_examples.append({
                            'pos': test_pos.copy(),
                            'elem_id': elem_id,
                            'level': elem_level,
                            'neighbor_level': neighbor_level,
                            'frac': frac,
                            'vertex_idx': vi,
                        })

    print(f"\n  Cross-boundary sweep:")
    print(f"    Tests: {cross_boundary_tests}")
    print(f"    Misses: {cross_boundary_misses} ({100*cross_boundary_misses/max(1,cross_boundary_tests):.2f}%)")

    if cross_boundary_examples:
        print(f"\n  Miss examples:")
        for ex in cross_boundary_examples[:10]:
            print(f"    Elem {ex['elem_id']} (L{ex['level']}→L{ex['neighbor_level']}), "
                  f"frac={ex['frac']:.2f}, vertex {ex['vertex_idx']}")
            print(f"      pos=({ex['pos'][0]:.10e}, {ex['pos'][1]:.10e}, {ex['pos'][2]:.10e})")

            # Detailed diagnosis: check grid indices at both levels
            for check_level in [ex['level'], ex['neighbor_level']]:
                if check_level in level_cell_sizes:
                    cs = level_cell_sizes[check_level]
                    i64, j64, k64 = compute_grid_index_f64(ex['pos'], cs)
                    i32, j32, k32 = compute_grid_index_f32(ex['pos'], cs)
                    match = "MATCH" if (i64, j64, k64) == (i32, j32, k32) else "MISMATCH"
                    print(f"      Level {check_level}: f64=({i64},{j64},{k64}) f32=({i32},{j32},{k32}) [{match}]")

            # Check point-in-tet for expected element
            pos_jax = jnp.array(ex['pos'], dtype=jnp.float32)
            is_inside = point_in_tet_dispatcher(
                pos_jax,
                jnp.int32(ex['elem_id']),
                octree_gpu.connectivity,
                octree_gpu.node_positions,
                config.POINT_IN_TET_METHOD
            )
            print(f"      Point-in-tet for expected elem {ex['elem_id']}: {bool(is_inside)}")
    else:
        print(f"  *** No misses in cross-boundary sweep! ***")

    # =========================================================================
    # Section 7: Float64 L2 Search Comparison
    # =========================================================================
    print("\n[7/7] Float64 grid index L2 search test...")
    print("  Manually performing L2 search with float64 grid indices")
    print("  to check if precision difference causes the failure.")

    # For positions that the GPU missed, do manual float64 search
    all_missed_positions = []
    for ex in missed_examples:
        all_missed_positions.append(ex['pos'])
    for ex in cross_boundary_examples:
        all_missed_positions.append(ex['pos'])

    if all_missed_positions:
        print(f"\n  Testing {len(all_missed_positions)} missed positions with float64 grid indices...")

        n_found_f64 = 0
        for pos in all_missed_positions:
            found_any = False
            # Try each level with float64 grid indices
            for level in sorted(level_cell_sizes.keys(), reverse=True):
                cs_f64 = np.float64(level_cell_sizes[level])
                i_base, j_base, k_base = compute_grid_index_f64(pos, cs_f64)

                # Search 3×3×3 neighborhood with float64 indices
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        for dk in [-1, 0, 1]:
                            i = i_base + di
                            j = j_base + dj
                            k = k_base + dk

                            offset = 1 << 19
                            max_coord = 1 << 20
                            i_m = np.clip(i + offset, 0, max_coord - 1)
                            j_m = np.clip(j + offset, 0, max_coord - 1)
                            k_m = np.clip(k + offset, 0, max_coord - 1)

                            morton = encode_morton_3d_jax(
                                jnp.int32(i_m), jnp.int32(j_m), jnp.int32(k_m)
                            )
                            cell_idx = find_cell_by_morton_and_level(
                                morton, jnp.uint8(level),
                                octree_gpu.cell_morton_codes,
                                octree_gpu.cell_levels
                            )
                            cell_idx = int(cell_idx)
                            if cell_idx < 0:
                                continue

                            # Test elements in cell
                            start = int(octree_gpu.cell_to_elements_offsets[cell_idx])
                            end = int(octree_gpu.cell_to_elements_offsets[cell_idx + 1])
                            for elem_offset in range(start, end):
                                elem_id = int(octree_gpu.cell_to_elements_data[elem_offset])
                                pos_jax = jnp.array(pos, dtype=jnp.float32)
                                is_inside = point_in_tet_dispatcher(
                                    pos_jax, jnp.int32(elem_id),
                                    octree_gpu.connectivity,
                                    octree_gpu.node_positions,
                                    config.POINT_IN_TET_METHOD
                                )
                                if bool(is_inside):
                                    found_any = True
                                    break
                            if found_any:
                                break
                        if found_any:
                            break
                    if found_any:
                        break
                if found_any:
                    break

            if found_any:
                n_found_f64 += 1

        print(f"\n  Float64 grid index search results:")
        print(f"    Found: {n_found_f64}/{len(all_missed_positions)}")
        print(f"    Still missed: {len(all_missed_positions) - n_found_f64}")

        if n_found_f64 > 0:
            print(f"\n  *** Float64 found {n_found_f64} positions that float32 missed! ***")
            print(f"  *** This CONFIRMS float32 precision is a contributing factor. ***")
        else:
            print(f"\n  Float64 didn't help — the issue is NOT precision.")
            print(f"  The elements may not be registered in any reachable cell.")
    else:
        print(f"  No missed positions to test (L2 search succeeded everywhere).")
        print(f"  The boundary centroids and cross-boundary positions are all findable.")
        print(f"  The loss may only occur at specific RK4 intermediate positions.")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Boundary elements found: {len(boundary_elements)}")
    print(f"  Float32 vs float64 grid index mismatches at boundary positions: {total_mismatches}")
    print(f"  Float32 boundary sweep mismatches: {boundary_mismatches}/{boundary_tests}")
    print(f"  GPU L2 missed boundary centroids: {n_missed_gpu}/{len(boundary_centroids)}")
    print(f"  Cross-boundary sweep misses: {cross_boundary_misses}/{cross_boundary_tests}")
    if all_missed_positions:
        print(f"  Float64 search recovered: {n_found_f64}/{len(all_missed_positions)} missed positions")

    print(f"\n  Conclusions:")
    if total_mismatches > 0:
        print(f"    ⚠️  Float32 precision causes grid index mismatches at {total_mismatches} boundary positions")
    else:
        print(f"    ✅ Float32 precision OK for tested boundary positions")

    if n_missed_gpu > 0:
        print(f"    ⚠️  GPU L2 search misses {n_missed_gpu} boundary centroids")
    else:
        print(f"    ✅ GPU L2 search finds all boundary centroids")

    if cross_boundary_misses > 0:
        print(f"    ⚠️  L2 search misses {cross_boundary_misses} cross-boundary positions")
    else:
        print(f"    ✅ L2 search finds all cross-boundary positions")

    print("\nDone.")


if __name__ == '__main__':
    main()
