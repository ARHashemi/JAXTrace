#!/usr/bin/env python3
"""
Verification test for parent-cube octree registration + static 3x3x3 search.

Tests:
1. Build statistics: elements-per-cell distribution, max <= expected cap
2. Found rate: 100% on intra-element particles (centroid, random, near_face,
   near_edge, near_vertex) — particles seeded inside elements at EACH
   refinement level separately
3. Comparison with vertex-multi 3x3x3 (dynamic loop) as reference
"""

import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp

import jaxtrace.config as config
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import extract_octree_cells_parent_cube
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import extract_octree_cells_vertex_multi
from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import find_axis_aligned_edges_single
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu
from jaxtrace.gpu.search.mesh_aligned_point_location import (
    search_mesh_aligned_octree_multi_local_where,
    search_mesh_aligned_octree_static_where,
)
from jaxtrace.gpu.search.aa_detection import precompute_aa_metadata, precompute_element_vertices, AxisAlignedMetadata
from jaxtrace.gpu.search.point_in_tet_methods import set_corrected_metadata, set_inverse_matrices_gpu
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices


# ============================================================================
# Intra-element particle generation (from benchmark_l2_accuracy.py)
# ============================================================================

def generate_intra_element_particles(connectivity, node_positions, n_particles, rng,
                                     valid_element_ids=None, position_type='random'):
    if valid_element_ids is None:
        n_choices = connectivity.shape[0]
        source_elements = rng.integers(0, n_choices, size=n_particles).astype(np.int32)
    else:
        source_elements = rng.choice(valid_element_ids, size=n_particles, replace=True).astype(np.int32)

    verts = node_positions[connectivity[source_elements]]  # (n_particles, 4, 3)

    if position_type == 'centroid':
        positions = verts.mean(axis=1)

    elif position_type == 'random':
        u = np.sort(rng.random((n_particles, 3)), axis=1)
        lam = np.zeros((n_particles, 4), dtype=np.float64)
        lam[:, 0] = u[:, 0]
        lam[:, 1] = u[:, 1] - u[:, 0]
        lam[:, 2] = u[:, 2] - u[:, 1]
        lam[:, 3] = 1.0 - u[:, 2]
        positions = np.einsum('ni,nid->nd', lam, verts)

    elif position_type == 'near_face':
        eps = 0.02
        small_idx = rng.integers(0, 4, size=n_particles)
        u = np.sort(rng.random((n_particles, 2)), axis=1)
        lam = np.zeros((n_particles, 4), dtype=np.float64)
        remaining = 1.0 - eps
        for i in range(n_particles):
            si = small_idx[i]
            lam[i, si] = eps
            others = [j for j in range(4) if j != si]
            lam[i, others[0]] = u[i, 0] * remaining
            lam[i, others[1]] = (u[i, 1] - u[i, 0]) * remaining
            lam[i, others[2]] = (1.0 - u[i, 1]) * remaining
        positions = np.einsum('ni,nid->nd', lam, verts)

    elif position_type == 'near_edge':
        eps = 0.02
        lam = np.zeros((n_particles, 4), dtype=np.float64)
        for i in range(n_particles):
            small = rng.choice(4, size=2, replace=False)
            lam[i, small[0]] = eps * rng.random()
            lam[i, small[1]] = eps * rng.random()
            remaining = 1.0 - lam[i, small[0]] - lam[i, small[1]]
            others = [j for j in range(4) if j not in small]
            split = rng.random()
            lam[i, others[0]] = split * remaining
            lam[i, others[1]] = (1.0 - split) * remaining
        positions = np.einsum('ni,nid->nd', lam, verts)

    elif position_type == 'near_vertex':
        eps = 0.02
        lam = np.zeros((n_particles, 4), dtype=np.float64)
        big_idx = rng.integers(0, 4, size=n_particles)
        for i in range(n_particles):
            bi = big_idx[i]
            total_small = 0.0
            for j in range(4):
                if j != bi:
                    val = eps * rng.random()
                    lam[i, j] = val
                    total_small += val
            lam[i, bi] = 1.0 - total_small
        positions = np.einsum('ni,nid->nd', lam, verts)

    else:
        raise ValueError(f"Unknown position_type: {position_type}")

    return positions, source_elements


# ============================================================================
# Batch search wrappers
# ============================================================================

def search_static_batch(positions_gpu, octree_gpu, max_elems_per_cell, batch_size=50000):
    n = positions_gpu.shape[0]
    all_eids = []
    all_tests = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = positions_gpu[start:end]
        eids, tests = jax.vmap(
            lambda p: search_mesh_aligned_octree_static_where(
                p, octree_gpu, max_elems_per_cell=max_elems_per_cell
            )
        )(batch)
        all_eids.append(eids)
        all_tests.append(tests)
    return jnp.concatenate(all_eids), jnp.concatenate(all_tests)


def search_dynamic_batch(positions_gpu, octree_gpu, batch_size=50000):
    n = positions_gpu.shape[0]
    all_eids = []
    all_tests = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = positions_gpu[start:end]
        eids, tests = jax.vmap(
            lambda p: search_mesh_aligned_octree_multi_local_where(
                p, octree_gpu, max_tests=jnp.int32(600)
            )
        )(batch)
        all_eids.append(eids)
        all_tests.append(tests)
    return jnp.concatenate(all_eids), jnp.concatenate(all_tests)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Test parent-cube octree registration")
    parser.add_argument("--input", required=True, help="Path to mesh directory")
    parser.add_argument("--n-particles", type=int, default=2000,
                        help="Particles per position type PER LEVEL")
    parser.add_argument("--max-elems", type=int, default=8,
                        help="MAX_ELEMS_PER_CELL for static loop")
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["float32", "float64"], default="float64")
    return parser.parse_args()


def main():
    args = parse_args()

    config.set_precision(args.precision == "float64")
    config.POINT_IN_TET_METHOD = "corrected_aa"
    config.POINT_IN_TET_TOLERANCE = 1e-6

    print("=" * 80)
    print("Parent-Cube Octree Verification Test")
    print("=" * 80)
    print(f"  Input: {args.input}")
    print(f"  Particles per type per level: {args.n_particles}")
    print(f"  MAX_ELEMS_PER_CELL: {args.max_elems}")
    print(f"  Precision: {args.precision}")

    # ------------------------------------------------------------------
    # 1. Load mesh
    # ------------------------------------------------------------------
    print(f"\n[1/6] Loading mesh...")
    from pathlib import Path
    mesh_base = Path(args.input)

    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=mesh_base / "0eule",
        file_pattern="cylA_{timestep}.pvtu",
        timestep_range=(158, 159),
        field_name='Displacement',
        verbose=True,
    )
    node_positions, connectivity, n_dup, _ = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    connectivity = np.asarray(connectivity, dtype=np.int32)
    node_positions = np.asarray(node_positions, dtype=np.float64)
    n_elements = connectivity.shape[0]
    print(f"  Elements: {n_elements:,}, Nodes: {node_positions.shape[0]:,} "
          f"(removed {n_dup:,} duplicates)")

    # ------------------------------------------------------------------
    # 2. Precompute PIT data
    # ------------------------------------------------------------------
    print(f"\n[2/6] Precomputing PIT metadata...")
    config.POINT_IN_TET_METHOD = 'inverse'
    aa_metadata = precompute_aa_metadata(connectivity, node_positions, verbose=False)
    element_vertices = precompute_element_vertices(connectivity, node_positions, verbose=False)
    M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)

    aa_metadata_gpu = AxisAlignedMetadata(
        base_vertex_indices=jax.device_put(aa_metadata.base_vertex_indices),
        base_vertices=jax.device_put(aa_metadata.base_vertices),
        inv_edge_lengths=jax.device_put(aa_metadata.inv_edge_lengths),
        axis_indices=jax.device_put(aa_metadata.axis_indices),
        is_axis_aligned=jax.device_put(aa_metadata.is_axis_aligned),
    )
    element_vertices_gpu = jax.device_put(element_vertices)
    M_inv_gpu = jax.device_put(M_inv_array)
    p0_gpu = jax.device_put(p0_array)

    set_corrected_metadata(aa_metadata_gpu, element_vertices_gpu)
    set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)
    print("  Done")

    # ------------------------------------------------------------------
    # 3. Build both octrees
    # ------------------------------------------------------------------
    print(f"\n[3/6] Building octrees...")
    t0 = time.time()

    octree_pc = extract_octree_cells_parent_cube(
        node_positions, connectivity, tolerance=1e-6, verbose=True
    )

    octree_vm = extract_octree_cells_vertex_multi(
        node_positions, connectivity, tolerance=1e-6, verbose=False
    )
    print(f"  Vertex-multi: {octree_vm.n_cells:,} cells, "
          f"{octree_vm.elements_per_cell_mean:.1f} elem/cell")
    print(f"  Build time: {time.time() - t0:.1f}s")

    # Upload both to GPU
    octree_pc_gpu = upload_mesh_aligned_octree_to_gpu(
        connectivity, node_positions, octree_pc, verbose=False
    )
    octree_vm_gpu = upload_mesh_aligned_octree_to_gpu(
        connectivity, node_positions, octree_vm, verbose=False
    )

    # ------------------------------------------------------------------
    # 4. Classify elements by level
    # ------------------------------------------------------------------
    print(f"\n[4/6] Classifying elements by level...")
    element_levels = np.zeros(n_elements, dtype=np.int32)
    n_non_kuhn = 0
    for elem_id in range(n_elements):
        vertices = node_positions[connectivity[elem_id]]
        cell_size, level = find_axis_aligned_edges_single(vertices, 1e-6)
        if np.any(cell_size == 0):
            element_levels[elem_id] = -1
            n_non_kuhn += 1
        else:
            element_levels[elem_id] = level

    unique_levels = np.unique(element_levels[element_levels >= 0])
    print(f"  Levels present: {sorted(unique_levels)}")
    for lvl in unique_levels:
        n = np.sum(element_levels == lvl)
        print(f"    Level {lvl:2d}: {n:>8,} elements ({100*n/n_elements:.2f}%)")
    if n_non_kuhn > 0:
        print(f"    Non-Kuhn: {n_non_kuhn:>8,} elements ({100*n_non_kuhn/n_elements:.4f}%)")

    # ------------------------------------------------------------------
    # 5. Generate particles and run searches
    # ------------------------------------------------------------------
    print(f"\n[5/6] Running search tests...")

    position_types = ['centroid', 'random', 'near_face', 'near_edge', 'near_vertex']
    rng = np.random.default_rng(args.seed)

    # Results tables
    results_pc = {}   # parent-cube + static
    results_vm = {}   # vertex-multi + dynamic (reference)

    test_levels = list(unique_levels) + ['all']

    for level in test_levels:
        if level == 'all':
            valid_ids = np.arange(n_elements, dtype=np.int32)
            level_label = "ALL"
        else:
            valid_ids = np.where(element_levels == level)[0].astype(np.int32)
            level_label = f"L{level}"

        if len(valid_ids) == 0:
            continue

        n_per_type = min(args.n_particles, len(valid_ids))
        results_pc[level_label] = {}
        results_vm[level_label] = {}

        for pt in position_types:
            positions, source_elems = generate_intra_element_particles(
                connectivity, node_positions, n_per_type, rng,
                valid_element_ids=valid_ids, position_type=pt,
            )

            positions_gpu = jax.device_put(jnp.array(positions, dtype=config.FLOAT_DTYPE_JNP))

            # Parent-cube + static search
            t0 = time.time()
            eids_pc, tests_pc = search_static_batch(
                positions_gpu, octree_pc_gpu, args.max_elems, args.batch_size
            )
            jax.block_until_ready(eids_pc)
            t_pc = time.time() - t0

            eids_pc_np = np.asarray(eids_pc)
            tests_pc_np = np.asarray(tests_pc)
            found_pc = np.sum(eids_pc_np >= 0)
            correct_pc = np.sum(eids_pc_np == source_elems)

            # Vertex-multi + dynamic search
            t0 = time.time()
            eids_vm, tests_vm = search_dynamic_batch(
                positions_gpu, octree_vm_gpu, args.batch_size
            )
            jax.block_until_ready(eids_vm)
            t_vm = time.time() - t0

            eids_vm_np = np.asarray(eids_vm)
            tests_vm_np = np.asarray(tests_vm)
            found_vm = np.sum(eids_vm_np >= 0)
            correct_vm = np.sum(eids_vm_np == source_elems)

            results_pc[level_label][pt] = {
                'found': found_pc, 'correct': correct_pc, 'total': n_per_type,
                'mean_tests': float(tests_pc_np[eids_pc_np >= 0].mean()) if found_pc > 0 else 0,
                'time': t_pc,
            }
            results_vm[level_label][pt] = {
                'found': found_vm, 'correct': correct_vm, 'total': n_per_type,
                'mean_tests': float(tests_vm_np[eids_vm_np >= 0].mean()) if found_vm > 0 else 0,
                'time': t_vm,
            }

            status = "OK" if found_pc == n_per_type else "FAIL"
            print(f"  {level_label:>4s} {pt:>12s}: "
                  f"PC={found_pc}/{n_per_type} ({status}) "
                  f"PIT={results_pc[level_label][pt]['mean_tests']:.0f}  "
                  f"VM={found_vm}/{n_per_type} "
                  f"PIT={results_vm[level_label][pt]['mean_tests']:.0f}")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    print(f"\nOctree build statistics:")
    print(f"  Parent-cube:  {octree_pc.n_cells:,} cells, "
          f"mean={octree_pc.elements_per_cell_mean:.2f}, "
          f"max={octree_pc.max_elements_per_cell} elem/cell")
    print(f"  Vertex-multi: {octree_vm.n_cells:,} cells, "
          f"mean={octree_vm.elements_per_cell_mean:.2f} elem/cell")
    print(f"  Static loop bound: {args.max_elems}")

    if octree_pc.max_elements_per_cell > args.max_elems:
        print(f"\n  WARNING: max_elements_per_cell ({octree_pc.max_elements_per_cell}) > "
              f"MAX_ELEMS_PER_CELL ({args.max_elems})")
        print(f"  Some elements may be truncated. Consider increasing --max-elems.")

    print(f"\nFound rate per level (Parent-Cube / Vertex-Multi):")
    print(f"  {'Level':>5s}  {'centroid':>10s}  {'random':>10s}  {'near_face':>10s}  "
          f"{'near_edge':>10s}  {'near_vtx':>10s}")
    print(f"  {'-'*65}")

    all_pass = True
    for level_label in results_pc:
        row = ""
        for pt in position_types:
            r_pc = results_pc[level_label][pt]
            r_vm = results_vm[level_label][pt]
            pct_pc = 100 * r_pc['found'] / r_pc['total']
            pct_vm = 100 * r_vm['found'] / r_vm['total']
            row += f"  {pct_pc:5.1f}/{pct_vm:5.1f}"
            if pct_pc < 100.0:
                all_pass = False
        print(f"  {level_label:>5s}{row}")

    print(f"\nMean PIT tests per query (Parent-Cube / Vertex-Multi):")
    print(f"  {'Level':>5s}  {'centroid':>10s}  {'random':>10s}  {'near_face':>10s}  "
          f"{'near_edge':>10s}  {'near_vtx':>10s}")
    print(f"  {'-'*65}")
    for level_label in results_pc:
        row = ""
        for pt in position_types:
            r_pc = results_pc[level_label][pt]
            r_vm = results_vm[level_label][pt]
            row += f"  {r_pc['mean_tests']:5.0f}/{r_vm['mean_tests']:5.0f}"
        print(f"  {level_label:>5s}{row}")

    print()
    if all_pass:
        print("RESULT: PASS - 100% found rate on all levels and position types")
    else:
        print("RESULT: FAIL - some queries not found (see details above)")

    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()
