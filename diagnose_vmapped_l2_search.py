#!/usr/bin/env python3
"""
DEFINITIVE DIAGNOSTIC: Does vmapped L2 search produce different results
than single-particle L2 search?

The hypothesis: search_mesh_aligned_octree_multi_local contains deeply nested
lax.cond calls that get lowered to SELECT under vmap, potentially producing
different results than non-vmapped execution.

Test Plan:
1. Load mesh and build octree (same as production)
2. Get positions of vanishing particles at known timestep
3. Run L2 search for each particle individually (no vmap)
4. Run L2 search for ALL particles vmapped together
5. Compare results: if vmap produces different elem_ids → ROOT CAUSE FOUND

This isolates the L2 search from RK4, L0, L1, and velocity interpolation.
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'

import sys
import re
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import extract_octree_cells_vertex_multi
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import (
    upload_mesh_aligned_octree_to_gpu,
)
from jaxtrace.gpu.search.aa_detection import precompute_aa_metadata, precompute_element_vertices
from jaxtrace.gpu.search.point_in_tet_methods import set_corrected_metadata, set_inverse_matrices_gpu
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
import jaxtrace.config as config

from jaxtrace.gpu.search.mesh_aligned_point_location import (
    search_mesh_aligned_octree_multi_local,
)
from jaxtrace.gpu.search.point_in_tet_methods import (
    point_in_tet_gpu as point_in_tet_dispatcher,
)

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError:
    print("ERROR: vtk not available")
    sys.exit(1)


# =============================================================================
# GPU single-particle helpers (float32, matching diagnose_gpu_vs_cpu_mismatch.py)
# =============================================================================

def gpu_search_l2(pos_f32, octree_gpu):
    """GPU L2 search (single particle, no vmap)."""
    pos_jax = jnp.array(pos_f32, dtype=jnp.float32)
    elem_id, n_tests = search_mesh_aligned_octree_multi_local(
        pos_jax, octree_gpu, max_tests=jnp.int32(600)
    )
    return int(elem_id), int(n_tests)


def gpu_point_in_tet(pos_f32, elem_id, connectivity_gpu, node_positions_gpu):
    """GPU point-in-tet (single element, matching production dispatcher)."""
    pos_jax = jnp.array(pos_f32, dtype=jnp.float32)
    result = point_in_tet_dispatcher(
        pos_jax,
        jnp.int32(elem_id),
        connectivity_gpu,
        node_positions_gpu,
        method=config.POINT_IN_TET_METHOD
    )
    return bool(result)


def gpu_interpolate_velocity(pos_f32, elem_id, connectivity_gpu, node_positions_gpu, velocity_field_gpu):
    """GPU velocity interpolation (float32 barycentric, matching production)."""
    if elem_id < 0:
        return np.zeros(3, dtype=np.float32)

    pos = jnp.array(pos_f32, dtype=jnp.float32)
    nodes_idx = connectivity_gpu[elem_id]
    nodes = node_positions_gpu[nodes_idx]
    node_vels = velocity_field_gpu[nodes_idx]

    v0 = nodes[1] - nodes[0]
    v1 = nodes[2] - nodes[0]
    v2 = nodes[3] - nodes[0]
    vp = pos - nodes[0]

    d00 = jnp.dot(v0, v0); d01 = jnp.dot(v0, v1); d02 = jnp.dot(v0, v2)
    d11 = jnp.dot(v1, v1); d12 = jnp.dot(v1, v2); d22 = jnp.dot(v2, v2)
    dp0 = jnp.dot(vp, v0); dp1 = jnp.dot(vp, v1); dp2 = jnp.dot(vp, v2)

    det = d00*(d11*d22-d12*d12) - d01*(d01*d22-d02*d12) + d02*(d01*d12-d02*d11)
    det = jnp.where(jnp.abs(det) < 1e-12, 1e-12, det)

    b1 = (dp0*(d11*d22-d12*d12) - d01*(dp1*d22-dp2*d12) + d02*(dp1*d12-dp2*d11)) / det
    b2 = (d00*(dp1*d22-dp2*d12) - dp0*(d01*d22-d02*d12) + d02*(d01*dp2-d02*dp1)) / det
    b3 = (d00*(d11*dp2-d12*dp1) - d01*(d01*dp2-d02*dp1) + dp0*(d01*d12-d02*d11)) / det
    b0 = 1.0 - b1 - b2 - b3

    vel = b0*node_vels[0] + b1*node_vels[1] + b2*node_vels[2] + b3*node_vels[3]
    return np.array(vel, dtype=np.float32)


def cpu_point_in_tet(pos, elem_id, connectivity_cpu, node_positions_cpu, tol=1e-6):
    """CPU point-in-tet (float64)."""
    nodes_idx = connectivity_cpu[elem_id]
    nodes = node_positions_cpu[nodes_idx].astype(np.float64)
    p = np.array(pos, dtype=np.float64)
    v0 = nodes[1] - nodes[0]
    v1 = nodes[2] - nodes[0]
    v2 = nodes[3] - nodes[0]
    vp = p - nodes[0]
    M = np.column_stack([v0, v1, v2])
    try:
        bary = np.linalg.solve(M, vp)
    except np.linalg.LinAlgError:
        return False, np.full(4, np.nan)
    b1, b2, b3 = bary
    b0 = 1.0 - b1 - b2 - b3
    coords = np.array([b0, b1, b2, b3])
    return bool(np.all(coords >= -tol)), coords


# =============================================================================
# Configuration (matching diagnose_gpu_vs_cpu_mismatch.py exactly)
# =============================================================================

MESH_BASE_PATH          = Path("data/FLA/post/0eule")
MESH_FILE_PATTERN       = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME     = 'Displacement'

EXPORT_DIR = Path(
    "output/benchmark_with_export_L-hits/"
    "Mesh-Aligned_Multi-Cell_+_3×3×3_Local_(Option_A_-_Phase_2)"
)

# Steps to analyse
STEP_RANGE = (1683, 1690)

# Specific particle IDs to force-inspect
FOCUS_PIDS = [265608]

# Also inspect up to N randomly-sampled vanishing particles
MAX_RANDOM = 50
RANDOM_SEED = 42

DT = 0.0025
POINT_IN_TET_METHOD = 'inverse'


# =============================================================================
# VTU loading (exact copy from diagnose_gpu_vs_cpu_mismatch.py)
# =============================================================================

def load_vtu(filepath):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(filepath))
    reader.Update()
    output = reader.GetOutput()
    pts = vtk_to_numpy(output.GetPoints().GetData()).astype(np.float32)
    pd = output.GetPointData()
    pid_arr = pd.GetArray('ParticleID')
    eid_arr = pd.GetArray('ElementID')
    if pid_arr is None or eid_arr is None:
        raise ValueError(f"Missing ParticleID or ElementID in {filepath}")
    return {
        'positions':    pts,
        'particle_ids': vtk_to_numpy(pid_arr).astype(np.int32),
        'element_ids':  vtk_to_numpy(eid_arr).astype(np.int32),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("DEFINITIVE DIAGNOSTIC: vmapped vs single-particle L2 search")
    print("=" * 80)

    # =========================================================================
    # Section 1: Load mesh (same pattern as diagnose_gpu_vs_cpu_mismatch.py)
    # =========================================================================
    print("\n[1/8] Loading mesh...")
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=False
    )
    node_positions, connectivity, _, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    # Ensure correct dtypes (deduplicate_nodes may change them)
    connectivity = connectivity.astype(np.int32)
    node_positions = node_positions.astype(np.float64)  # keep float64 for CPU ops
    n_nodes = node_positions.shape[0]
    n_elements = connectivity.shape[0]
    print(f"  {n_nodes:,} nodes, {n_elements:,} elements")

    # =========================================================================
    # Section 2: Build search structures (same pattern as production)
    # =========================================================================
    print("\n[2/8] Building search structures...")
    aa_metadata = precompute_aa_metadata(connectivity, node_positions, verbose=False)
    element_vertices = precompute_element_vertices(connectivity, node_positions, verbose=False)
    M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)
    element_neighbors = build_element_neighbors_array(connectivity, method='face', verbose=False)
    mesh_gpu = upload_mesh_to_gpu(connectivity, node_positions, element_neighbors, verbose=False)

    mesh_octree = extract_octree_cells_vertex_multi(
        node_positions, connectivity, tolerance=1e-6, verbose=False
    )
    octree_gpu = upload_mesh_aligned_octree_to_gpu(
        connectivity, node_positions, mesh_octree, verbose=False
    )
    print(f"  Octree: {mesh_octree.n_cells:,} cells, {mesh_octree.elements_per_cell_mean:.2f} elem/cell")

    # Compute element volumes (same as benchmark)
    e1 = node_positions[connectivity[:, 1]] - node_positions[connectivity[:, 0]]
    e2 = node_positions[connectivity[:, 2]] - node_positions[connectivity[:, 0]]
    e3 = node_positions[connectivity[:, 3]] - node_positions[connectivity[:, 0]]
    det = np.sum(e1 * np.cross(e2, e3), axis=1)
    element_volumes = np.abs(det) / 6.0

    # Upload metadata to GPU (same as benchmark and diagnosis scripts)
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
    p0_gpu    = jax.device_put(p0_array)
    element_volumes_gpu = jax.device_put(element_volumes.astype(np.float32))
    set_corrected_metadata(aa_metadata_gpu, element_vertices_gpu)
    set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)
    config.POINT_IN_TET_METHOD = POINT_IN_TET_METHOD

    velocity_field_gpu = jax.device_put(jnp.array(velocity_sequence[0], dtype=jnp.float32))
    velocity_fields_gpu = jax.device_put(jnp.array(velocity_sequence, dtype=jnp.float32))

    print(f"  Velocity fields shape: {velocity_fields_gpu.shape}")

    # =========================================================================
    # Section 3: Load VTU snapshots & collect test positions
    # =========================================================================
    print(f"\n[3/8] Loading VTU snapshots for steps {STEP_RANGE}...")

    step_pat = re.compile(r'particles_step_(\d+)\.vtu$')
    vtu_files = sorted(EXPORT_DIR.glob("particles_step_*.vtu"))

    vtu_with_steps = []
    for f in vtu_files:
        m = step_pat.search(f.name)
        if m:
            vtu_with_steps.append((int(m.group(1)), f))
    vtu_with_steps.sort()
    print(f"  Found {len(vtu_with_steps)} VTU files in {EXPORT_DIR}")

    # Find vanishing particles (same logic as diagnose_gpu_vs_cpu_mismatch.py)
    test_positions = []
    test_labels = []
    test_elem_ids = []  # element IDs from VTU (for L0 cache)

    snapshots = {}
    for step_num, path in vtu_with_steps:
        if STEP_RANGE[0] <= step_num <= STEP_RANGE[1]:
            snapshots[step_num] = load_vtu(str(path))

    if not snapshots:
        print("  WARNING: No VTU files in step range. Using known position only.")
    else:
        print(f"  Loaded steps: {sorted(snapshots.keys())}")

        # Find particles that vanish between consecutive steps
        steps_sorted = sorted(snapshots.keys())
        vanishing_particles = []

        for idx in range(len(steps_sorted) - 1):
            s_curr = steps_sorted[idx]
            s_next = steps_sorted[idx + 1]
            snap_curr = snapshots[s_curr]
            snap_next = snapshots[s_next]

            pids_curr = set(snap_curr['particle_ids'])
            pids_next = set(snap_next['particle_ids'])
            vanished = pids_curr - pids_next

            for pid in vanished:
                mask = snap_curr['particle_ids'] == pid
                pos = snap_curr['positions'][mask][0]
                eid = snap_curr['element_ids'][mask][0]
                vanishing_particles.append((pid, s_curr, pos, eid))

        print(f"  Total vanishing particles in range: {len(vanishing_particles)}")

        # Collect focus particles
        for pid_target in FOCUS_PIDS:
            for pid, step, pos, eid in vanishing_particles:
                if pid == pid_target:
                    test_positions.append(pos)
                    test_labels.append(f"PID={pid}_step={step}")
                    test_elem_ids.append(eid)
                    print(f"    Focus PID={pid} at step {step}: pos=({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}), elem={eid}")

            # Also check snapshots directly if not found as vanishing
            if not any(pid_target == p[0] for p in vanishing_particles):
                for step_num in steps_sorted:
                    snap = snapshots[step_num]
                    mask = snap['particle_ids'] == pid_target
                    if np.any(mask):
                        pos = snap['positions'][mask][0]
                        eid = snap['element_ids'][mask][0]
                        test_positions.append(pos)
                        test_labels.append(f"PID={pid_target}_step={step_num}")
                        test_elem_ids.append(eid)
                        print(f"    Focus PID={pid_target} at step {step_num}: pos=({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}), elem={eid}")
                        break

        # Collect random vanishing particles
        rng = np.random.RandomState(RANDOM_SEED)
        remaining = [p for p in vanishing_particles if p[0] not in FOCUS_PIDS]
        if remaining:
            n_sample = min(MAX_RANDOM, len(remaining))
            sample_idx = rng.choice(len(remaining), n_sample, replace=False)
            for i in sample_idx:
                pid, step, pos, eid = remaining[i]
                test_positions.append(pos)
                test_labels.append(f"vanishing_PID={pid}_step={step}")
                test_elem_ids.append(eid)
            print(f"    Added {n_sample} random vanishing particles")

    # Fallback if no VTU data
    if not test_positions:
        print("  No VTU data available, using known particle position + jitter")
        base = np.array([0.0130553, -0.0090768, -0.00186707], dtype=np.float32)
        test_positions.append(base)
        test_labels.append("PID=265608_approx")
        test_elem_ids.append(-1)

        rng = np.random.RandomState(RANDOM_SEED)
        for i in range(49):
            jitter = rng.uniform(-0.001, 0.001, 3).astype(np.float32)
            test_positions.append(base + jitter)
            test_labels.append(f"jittered_{i}")
            test_elem_ids.append(-1)

    test_positions_np = np.array(test_positions, dtype=np.float32)
    test_elem_ids_np = np.array(test_elem_ids, dtype=np.int32)
    N = len(test_positions_np)
    print(f"\n  Total test positions: {N}")

    # =========================================================================
    # Section 4: Run L2 search INDIVIDUALLY (no vmap)
    # =========================================================================
    print("\n[4/8] Individual (non-vmapped) L2 search...")

    # JIT compile the single-particle search
    @jax.jit
    def search_single(pos):
        return search_mesh_aligned_octree_multi_local(pos, octree_gpu, max_tests=jnp.int32(600))

    # Warmup
    _ = search_single(jnp.array(test_positions_np[0]))

    individual_results = []
    t0 = time.time()
    for i in range(N):
        pos_gpu = jnp.array(test_positions_np[i])
        elem_id, n_tests = search_single(pos_gpu)
        individual_results.append((int(elem_id), int(n_tests)))
    t1 = time.time()

    individual_elem_ids = np.array([r[0] for r in individual_results])
    individual_n_tests = np.array([r[1] for r in individual_results])

    found_individual = np.sum(individual_elem_ids >= 0)
    print(f"  Individual search: {found_individual}/{N} found ({100*found_individual/N:.1f}%)")
    print(f"  Time: {t1-t0:.2f}s")
    print(f"  Mean tests: {np.mean(individual_n_tests):.1f}")

    # Show first few results
    for i in range(min(10, N)):
        print(f"    [{test_labels[i]}] elem={individual_elem_ids[i]}, tests={individual_n_tests[i]}")

    # =========================================================================
    # Section 5: Run L2 search VMAPPED (batch) with increasing batch sizes
    # =========================================================================
    print("\n[5/8] Vmapped (batch) L2 search...")

    # Create vmapped search
    @jax.jit
    def search_vmapped(positions_batch):
        return jax.vmap(
            lambda pos: search_mesh_aligned_octree_multi_local(pos, octree_gpu, max_tests=jnp.int32(600))
        )(positions_batch)

    # Try with increasing batch sizes to find OOM threshold
    vmapped_elem_ids = np.full(N, -999, dtype=np.int32)  # -999 = not tested
    vmapped_n_tests = np.full(N, -1, dtype=np.int32)

    batch_sizes_to_try = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    max_working_batch = 0

    for batch_size in batch_sizes_to_try:
        if batch_size > N:
            break
        print(f"\n  Testing batch_size={batch_size}...")
        try:
            positions_batch = jnp.array(test_positions_np[:batch_size])
            t0 = time.time()
            elem_ids_batch, n_tests_batch = search_vmapped(positions_batch)
            elem_ids_batch.block_until_ready()
            t1 = time.time()

            vmapped_elem_ids[:batch_size] = np.array(elem_ids_batch)
            vmapped_n_tests[:batch_size] = np.array(n_tests_batch)

            # Compare with individual results
            matches = vmapped_elem_ids[:batch_size] == individual_elem_ids[:batch_size]
            n_match = int(np.sum(matches))
            n_mismatch = batch_size - n_match

            print(f"    Time: {t1-t0:.2f}s")
            print(f"    Matches: {n_match}/{batch_size}, Mismatches: {n_mismatch}/{batch_size}")

            if n_mismatch > 0:
                print(f"\n    *** MISMATCHES DETECTED (batch_size={batch_size}) ***")
                for i in range(batch_size):
                    if not matches[i]:
                        print(f"      [{test_labels[i]}] individual={individual_elem_ids[i]}, vmapped={vmapped_elem_ids[i]}")

            max_working_batch = batch_size

        except Exception as e:
            err_str = str(e)
            if 'RESOURCE_EXHAUSTED' in err_str:
                print(f"    OOM at batch_size={batch_size}")
            else:
                print(f"    Error at batch_size={batch_size}: {type(e).__name__}: {e}")
            break

    # =========================================================================
    # Section 6: Full comparison at maximum working batch size
    # =========================================================================
    print(f"\n[6/8] Full comparison at batch_size={max_working_batch}")

    if max_working_batch >= 2:
        n_compare = min(N, max_working_batch)

        vmapped_final = vmapped_elem_ids[:n_compare]
        individual_final = individual_elem_ids[:n_compare]

        matches = vmapped_final == individual_final
        n_match = int(np.sum(matches))
        n_mismatch = n_compare - n_match

        # Category analysis
        both_found = (vmapped_final >= 0) & (individual_final >= 0)
        vmap_lost = (vmapped_final < 0) & (individual_final >= 0)
        individual_lost = (vmapped_final >= 0) & (individual_final < 0)
        both_lost = (vmapped_final < 0) & (individual_final < 0)

        print(f"  Results for {n_compare} particles:")
        print(f"    Both found same element: {np.sum(matches & both_found)}")
        print(f"    Both found but DIFFERENT element: {np.sum(~matches & both_found)}")
        print(f"    Individual found, vmap LOST: {np.sum(vmap_lost)} *** THIS IS THE BUG ***")
        print(f"    Vmap found, individual lost: {np.sum(individual_lost)}")
        print(f"    Both lost: {np.sum(both_lost)}")

        if np.sum(vmap_lost) > 0:
            print(f"\n  *** CONFIRMED: vmap causes particle loss in L2 search! ***")
            print(f"  {np.sum(vmap_lost)} particles found by individual search but LOST by vmapped search")
            print(f"\n  Detailed mismatches (individual found, vmap lost):")
            for i in range(n_compare):
                if vmap_lost[i]:
                    print(f"    [{test_labels[i]}] pos=({test_positions_np[i][0]:.6f}, {test_positions_np[i][1]:.6f}, {test_positions_np[i][2]:.6f})")
                    print(f"      individual: elem={individual_final[i]}, vmap: elem={vmapped_final[i]}")

        if np.sum(~matches & both_found) > 0:
            print(f"\n  *** WARNING: vmap returns DIFFERENT element for {np.sum(~matches & both_found)} particles! ***")
            for i in range(n_compare):
                if not matches[i] and both_found[i]:
                    print(f"    [{test_labels[i]}] individual={individual_final[i]}, vmap={vmapped_final[i]}")
    else:
        print("  Cannot compare: batch_size too small")

    # =========================================================================
    # Section 7: Test full L0+L1+L2 hierarchy under vmap (L1 on vs off)
    # =========================================================================
    print(f"\n[7/10] Test full L0+L1+L2 hierarchy under vmap (L1 on vs off)...")
    print("  (This tests whether L1→L2 fallback is corrupted by vmap)")

    from jaxtrace.gpu.tracking.rk4_fully_fused_timedep import create_rk4_fully_fused_timedep_with_stats
    from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
    from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu

    # Build Morton octree (needed for create_rk4 even if not used)
    octree_struct = build_global_morton_octree(node_positions, connectivity)
    mesh_gpu_octree = upload_global_morton_to_gpu(octree_struct, connectivity, node_positions)

    # Create RK4 with L1 ENABLED (production config)
    original_l2_method = config.L2_SEARCH_METHOD
    config.L2_SEARCH_METHOD = 'mesh_aligned_octree'

    _, step_fn_stats = create_rk4_fully_fused_timedep_with_stats(
        mesh_gpu_connectivity=mesh_gpu.connectivity,
        mesh_gpu_node_positions=mesh_gpu.node_positions,
        mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
        mesh_gpu_element_volumes=element_volumes_gpu,
        mesh_gpu_global_morton=mesh_gpu_octree,
        n_hops=5,
        enable_l1_search=True,  # L1 ENABLED
        l2_search_method='radius',  # fallback (not used)
        mesh_aligned_octree=octree_gpu,
        mesh_aligned_octree_use_multi_local=True,
    )

    # Create RK4 with L1 DISABLED
    _, step_fn_stats_no_l1 = create_rk4_fully_fused_timedep_with_stats(
        mesh_gpu_connectivity=mesh_gpu.connectivity,
        mesh_gpu_node_positions=mesh_gpu.node_positions,
        mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
        mesh_gpu_element_volumes=element_volumes_gpu,
        mesh_gpu_global_morton=mesh_gpu_octree,
        n_hops=5,
        enable_l1_search=False,  # L1 DISABLED
        l2_search_method='radius',
        mesh_aligned_octree=octree_gpu,
        mesh_aligned_octree_use_multi_local=True,
    )

    config.L2_SEARCH_METHOD = original_l2_method

    # Get particles that were found by individual L2 search
    found_mask = individual_elem_ids >= 0
    n_found = int(np.sum(found_mask))
    print(f"  Testing with {n_found} particles found by individual L2")

    if n_found > 0:
        rk4_pos_gpu = jnp.array(test_positions_np[found_mask], dtype=jnp.float32)
        rk4_elem_gpu = jnp.array(individual_elem_ids[found_mask], dtype=jnp.int32)

        # Test with L1 enabled — single step
        print("\n  RK4 step with L1 ENABLED:")
        try:
            pos_after_l1, elem_after_l1, stats_l1 = step_fn_stats(
                rk4_pos_gpu, rk4_elem_gpu, DT, velocity_fields_gpu, 0
            )
            pos_after_l1.block_until_ready()

            lost_l1 = int(jnp.sum(elem_after_l1 < 0))
            l0_hits, l1_hits, l2_hits, misses = stats_l1
            print(f"    Lost after 1 step: {lost_l1}/{n_found}")
            print(f"    L0={int(l0_hits)}, L1={int(l1_hits)}, L2={int(l2_hits)}, miss={int(misses)}")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            lost_l1 = -1

        # Test with L1 DISABLED — single step
        print("\n  RK4 step with L1 DISABLED (L0 → L2 only):")
        try:
            pos_after_no_l1, elem_after_no_l1, stats_no_l1 = step_fn_stats_no_l1(
                rk4_pos_gpu, rk4_elem_gpu, DT, velocity_fields_gpu, 0
            )
            pos_after_no_l1.block_until_ready()

            lost_no_l1 = int(jnp.sum(elem_after_no_l1 < 0))
            l0_h2, l1_h2, l2_h2, miss_2 = stats_no_l1
            print(f"    Lost after 1 step: {lost_no_l1}/{n_found}")
            print(f"    L0={int(l0_h2)}, L1={int(l1_h2)}, L2={int(l2_h2)}, miss={int(miss_2)}")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            lost_no_l1 = -1

        # Compare
        if lost_l1 >= 0 and lost_no_l1 >= 0:
            print(f"\n  *** COMPARISON (1 step) ***")
            print(f"    L1 enabled:  {lost_l1} particles lost, {int(misses)} misses")
            print(f"    L1 disabled: {lost_no_l1} particles lost, {int(miss_2)} misses")
            if lost_l1 > lost_no_l1:
                print(f"    → L1 CAUSES {lost_l1 - lost_no_l1} EXTRA PARTICLE LOSSES!")
            elif lost_l1 < lost_no_l1:
                print(f"    → L1 helps retain {lost_no_l1 - lost_l1} more particles")
            else:
                print(f"    → Same particle loss (issue not in L1)")

        # Multi-step cascade test
        print(f"\n  Multi-step cascade test (10 steps):")
        n_steps = 10

        print("    L1 ENABLED:")
        pos_curr = rk4_pos_gpu
        elem_curr = rk4_elem_gpu
        try:
            for step in range(n_steps):
                pos_curr, elem_curr, stats_step = step_fn_stats(
                    pos_curr, elem_curr, DT, velocity_fields_gpu, step % velocity_fields_gpu.shape[0]
                )
                lost = int(jnp.sum(elem_curr < 0))
                l0h, l1h, l2h, mss = stats_step
                print(f"      Step {step}: lost={lost}/{n_found}, L0={int(l0h)}, L1={int(l1h)}, L2={int(l2h)}, miss={int(mss)}")
        except Exception as e:
            print(f"      FAILED: {type(e).__name__}: {e}")

        print("\n    L1 DISABLED:")
        pos_curr = rk4_pos_gpu
        elem_curr = rk4_elem_gpu
        try:
            for step in range(n_steps):
                pos_curr, elem_curr, stats_step = step_fn_stats_no_l1(
                    pos_curr, elem_curr, DT, velocity_fields_gpu, step % velocity_fields_gpu.shape[0]
                )
                lost = int(jnp.sum(elem_curr < 0))
                l0h, l1h, l2h, mss = stats_step
                print(f"      Step {step}: lost={lost}/{n_found}, L0={int(l0h)}, L1={int(l1h)}, L2={int(l2h)}, miss={int(mss)}")
        except Exception as e:
            print(f"      FAILED: {type(e).__name__}: {e}")

    # =========================================================================
    # Section 8: RK4 sub-step replay — WHERE exactly do searches fail?
    # =========================================================================
    print(f"\n[8/10] RK4 sub-step replay (single-particle GPU, no vmap)...")
    print("  Replaying RK4 sub-steps one at a time to capture failing positions.")
    print("  For each failure: brute-force check if position is inside ANY element.\n")

    connectivity_gpu = mesh_gpu.connectivity
    node_positions_gpu = mesh_gpu.node_positions
    neighbors_gpu = mesh_gpu.element_neighbors

    connectivity_cpu = np.array(connectivity, dtype=np.int32)
    node_positions_cpu = np.array(node_positions, dtype=np.float64)

    # Precompute centroids for brute-force search
    centroids_cpu = node_positions_cpu[connectivity_cpu].mean(axis=1)

    # Mesh bounding box
    bbox_min = node_positions_cpu.min(axis=0)
    bbox_max = node_positions_cpu.max(axis=0)
    print(f"  Mesh bbox: min={bbox_min}, max={bbox_max}")

    STAGE_NAMES = ['k1', 'k2', 'k3', 'k4', 'final']
    MAX_REPLAY_PARTICLES = 15  # Limit to keep output manageable

    # Select particles for replay: focus PIDs first, then random vanishing
    replay_indices = []
    found_labels = [test_labels[i] for i in range(N) if found_mask[i]]
    found_positions = test_positions_np[found_mask]
    found_elems = individual_elem_ids[found_mask]

    # Focus PIDs first
    for i, label in enumerate(found_labels):
        if any(f"PID={pid}" in label for pid in FOCUS_PIDS):
            replay_indices.append(i)

    # Then add random vanishing until we hit the limit
    remaining = [i for i in range(n_found) if i not in replay_indices]
    rng_replay = np.random.RandomState(RANDOM_SEED)
    if remaining:
        n_add = min(MAX_REPLAY_PARTICLES - len(replay_indices), len(remaining))
        replay_indices.extend(rng_replay.choice(remaining, n_add, replace=False).tolist())

    print(f"  Replaying {len(replay_indices)} particles through RK4 sub-steps\n")

    # Counters for summary
    total_substeps = 0
    total_misses = 0
    misses_outside_bbox = 0
    misses_inside_mesh_bf = 0
    misses_outside_mesh_bf = 0
    miss_n_tests_list = []
    miss_displacement_list = []

    for ri in replay_indices:
        pos0 = found_positions[ri].copy()
        elem0 = int(found_elems[ri])
        label = found_labels[ri]

        print(f"  {'='*70}")
        print(f"  {label}")
        print(f"  pos  = [{pos0[0]:.8e}, {pos0[1]:.8e}, {pos0[2]:.8e}]")
        print(f"  elem = {elem0}")

        # Verify starting element
        if elem0 >= 0:
            inside_start = gpu_point_in_tet(pos0, elem0, connectivity_gpu, node_positions_gpu)
            print(f"  GPU PIT at start: inside={inside_start}")

        # RK4 sub-step replay
        pos = pos0.copy()
        cached = elem0
        vel_k1 = vel_k2 = vel_k3 = vel_k4 = np.zeros(3, dtype=np.float32)

        for si, stage_name in enumerate(STAGE_NAMES):
            total_substeps += 1

            # Compute search position based on RK4 stage
            if stage_name == 'k1':
                search_pos = pos.copy()
            elif stage_name == 'k2':
                search_pos = (pos + np.float32(0.5) * np.float32(DT) * vel_k1).astype(np.float32)
            elif stage_name == 'k3':
                search_pos = (pos + np.float32(0.5) * np.float32(DT) * vel_k2).astype(np.float32)
            elif stage_name == 'k4':
                search_pos = (pos + np.float32(DT) * vel_k3).astype(np.float32)
            elif stage_name == 'final':
                search_pos = (pos + (np.float32(DT) / np.float32(6.0)) *
                              (vel_k1 + np.float32(2.0)*vel_k2 +
                               np.float32(2.0)*vel_k3 + vel_k4)).astype(np.float32)

            # Displacement from start
            displacement = np.linalg.norm(search_pos - pos0)

            # Check bounding box
            inside_bbox = np.all(search_pos >= bbox_min - 1e-6) and np.all(search_pos <= bbox_max + 1e-6)

            # L2 search (GPU, single particle)
            l2_elem, l2_tests = gpu_search_l2(search_pos, octree_gpu)

            # L0 search (cached element)
            l0_elem = -1
            if cached >= 0:
                l0_inside = gpu_point_in_tet(search_pos, cached, connectivity_gpu, node_positions_gpu)
                l0_elem = cached if l0_inside else -1

            # Determine final elem (L0 → L2, skip L1 for clarity)
            found_elem = l0_elem if l0_elem >= 0 else l2_elem
            level_str = 'L0' if l0_elem >= 0 else ('L2' if l2_elem >= 0 else 'MISS')

            # If MISS: brute-force check
            bf_elem = -1
            bf_candidates = 0
            if found_elem < 0:
                total_misses += 1
                miss_n_tests_list.append(l2_tests)
                miss_displacement_list.append(displacement)

                if not inside_bbox:
                    misses_outside_bbox += 1
                else:
                    # Brute-force: check all elements with centroid within 2mm
                    dists = np.linalg.norm(centroids_cpu - search_pos.astype(np.float64), axis=1)
                    candidates = np.where(dists < 0.002)[0]
                    bf_candidates = len(candidates)
                    for ceid in candidates:
                        ok, bary = cpu_point_in_tet(search_pos, int(ceid), connectivity_cpu, node_positions_cpu)
                        if ok:
                            bf_elem = int(ceid)
                            break
                    if bf_elem >= 0:
                        misses_inside_mesh_bf += 1
                    else:
                        # Try larger radius
                        candidates2 = np.where(dists < 0.01)[0]
                        bf_candidates = len(candidates2)
                        for ceid in candidates2:
                            ok, bary = cpu_point_in_tet(search_pos, int(ceid), connectivity_cpu, node_positions_cpu)
                            if ok:
                                bf_elem = int(ceid)
                                break
                        if bf_elem >= 0:
                            misses_inside_mesh_bf += 1
                        else:
                            misses_outside_mesh_bf += 1

            # Velocity interpolation
            vel = gpu_interpolate_velocity(
                search_pos, found_elem, connectivity_gpu, node_positions_gpu, velocity_field_gpu
            )

            # Format output
            miss_flag = ""
            if found_elem < 0:
                if not inside_bbox:
                    miss_flag = " *** MISS (OUTSIDE BBOX) ***"
                elif bf_elem >= 0:
                    miss_flag = f" *** MISS but BF found elem={bf_elem} (L2 COVERAGE GAP!) ***"
                else:
                    miss_flag = f" *** MISS (not in any element, {bf_candidates} candidates checked) ***"

            print(f"    {stage_name:>5s}: cached={cached:>8d} → {level_str:>4s} elem={found_elem:>8d}  "
                  f"|v|={np.linalg.norm(vel):.4e}  disp={displacement:.4e}  "
                  f"L2({l2_tests:3d}t)={l2_elem:>8d}  bbox={'Y' if inside_bbox else 'N'}"
                  f"{miss_flag}")

            # Update for next stage
            cached = found_elem
            if stage_name == 'k1':
                vel_k1 = vel
            elif stage_name == 'k2':
                vel_k2 = vel
            elif stage_name == 'k3':
                vel_k3 = vel
            elif stage_name == 'k4':
                vel_k4 = vel

    # =========================================================================
    # Section 9: Analysis of L2 coverage at failing positions
    # =========================================================================
    print(f"\n[9/10] Analysis of L2 coverage at failing positions...")

    if total_misses > 0:
        print(f"  Total sub-steps: {total_substeps}")
        print(f"  Total misses: {total_misses} ({100*total_misses/total_substeps:.1f}%)")
        print(f"  Misses outside bounding box: {misses_outside_bbox}")
        print(f"  Misses inside mesh (BF found): {misses_inside_mesh_bf} ← L2 COVERAGE GAPS")
        print(f"  Misses truly outside mesh (BF not found): {misses_outside_mesh_bf}")

        if miss_n_tests_list:
            print(f"\n  L2 test budget at miss positions:")
            print(f"    Mean tests: {np.mean(miss_n_tests_list):.1f}")
            print(f"    Max tests:  {np.max(miss_n_tests_list)}")
            print(f"    Min tests:  {np.min(miss_n_tests_list)}")
            hit_budget = sum(1 for t in miss_n_tests_list if t >= 590)
            print(f"    Hit budget limit (>=590): {hit_budget}/{len(miss_n_tests_list)}")

        if miss_displacement_list:
            print(f"\n  Displacement from start at miss positions:")
            print(f"    Mean: {np.mean(miss_displacement_list):.6e}")
            print(f"    Max:  {np.max(miss_displacement_list):.6e}")
            print(f"    Min:  {np.min(miss_displacement_list):.6e}")

        if misses_inside_mesh_bf > 0:
            print(f"\n  *** L2 COVERAGE GAP CONFIRMED ***")
            print(f"  {misses_inside_mesh_bf} positions are inside the mesh (brute-force finds element)")
            print(f"  but L2 search (3x3x3 multi-level, 600 test budget) returns -1.")
            print(f"  This means the mesh-aligned octree does not cover these positions.")
        elif misses_outside_bbox > 0:
            print(f"\n  *** PARTICLES LEAVE MESH DOMAIN ***")
            print(f"  {misses_outside_bbox} mid-step positions are outside the mesh bounding box.")
            print(f"  Velocity is pushing particles outside the domain during RK4 sub-steps.")
        elif misses_outside_mesh_bf > 0:
            print(f"\n  *** PARTICLES IN VOID REGIONS ***")
            print(f"  {misses_outside_mesh_bf} positions are inside bbox but not in any element.")
            print(f"  This suggests mesh gaps (holes) or the positions are in thin boundary layers.")
    else:
        print(f"  No misses during sub-step replay! All {total_substeps} sub-steps found elements.")

    # =========================================================================
    # Section 10: Velocity magnitude analysis for vanishing particles
    # =========================================================================
    print(f"\n[10/10] Velocity magnitude analysis...")
    print("  Checking if 'Displacement' field values are realistic velocities.\n")

    # Sample a few elements and check velocity magnitudes
    sample_elems = found_elems[:min(10, n_found)]
    sample_positions = found_positions[:min(10, n_found)]

    vel_magnitudes = []
    for i in range(len(sample_elems)):
        vel = gpu_interpolate_velocity(
            sample_positions[i], int(sample_elems[i]),
            connectivity_gpu, node_positions_gpu, velocity_field_gpu
        )
        vmag = np.linalg.norm(vel)
        vel_magnitudes.append(vmag)
        if i < 5:
            print(f"    elem={sample_elems[i]:>8d}: vel=({vel[0]:.4e}, {vel[1]:.4e}, {vel[2]:.4e}), |v|={vmag:.4e}")

    if vel_magnitudes:
        vmags = np.array(vel_magnitudes)
        print(f"\n  Velocity magnitudes: mean={vmags.mean():.4e}, max={vmags.max():.4e}, min={vmags.min():.4e}")

        # Estimate displacement per sub-step
        max_disp_half = vmags.max() * DT * 0.5
        max_disp_full = vmags.max() * DT
        print(f"  Max displacement in half-step (k2,k3): {max_disp_half:.6e}")
        print(f"  Max displacement in full-step  (k4):   {max_disp_full:.6e}")
        print(f"  Mesh bbox size: {bbox_max - bbox_min}")

        # Check: does displacement exceed mesh size?
        mesh_size = bbox_max - bbox_min
        if max_disp_full > mesh_size.min() * 0.5:
            print(f"\n  *** WARNING: Max displacement ({max_disp_full:.4e}) exceeds")
            print(f"      half the smallest mesh dimension ({mesh_size.min()*0.5:.4e})!")
            print(f"      This strongly suggests the 'Displacement' field is NOT velocity,")
            print(f"      or DT is too large. Particles can leave the mesh in a single sub-step.")
        elif max_disp_full > mesh_size.min() * 0.01:
            print(f"\n  Note: Max displacement ({max_disp_full:.4e}) is {100*max_disp_full/mesh_size.min():.1f}%")
            print(f"      of the smallest mesh dimension. This is significant.")

    # =========================================================================
    # Section 11: Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Test positions: {N}")
    print(f"  Individual L2 search found: {found_individual}/{N}")
    print(f"  Max vmap batch tested: {max_working_batch}")

    if max_working_batch >= 2:
        n_compare = min(N, max_working_batch)
        vmap_found = int(np.sum(vmapped_elem_ids[:n_compare] >= 0))
        print(f"  Vmapped L2 search found: {vmap_found}/{n_compare}")
        matches = vmapped_elem_ids[:n_compare] == individual_elem_ids[:n_compare]
        print(f"  Exact matches: {int(np.sum(matches))}/{n_compare}")

    print(f"\n  RK4 sub-step replay ({len(replay_indices)} particles):")
    print(f"    Total sub-steps: {total_substeps}")
    print(f"    Misses: {total_misses} ({100*total_misses/max(total_substeps,1):.1f}%)")
    if total_misses > 0:
        print(f"    - Outside bounding box:     {misses_outside_bbox}")
        print(f"    - Inside mesh (BF found):   {misses_inside_mesh_bf}  ← L2 coverage gap")
        print(f"    - Not in any element (BF):  {misses_outside_mesh_bf}  ← void/gap/outside")

    print(f"\n  Conclusions:")
    if misses_outside_bbox > 0 and misses_outside_bbox >= total_misses * 0.5:
        print(f"    PRIMARY: Particles leave mesh domain during RK4 sub-steps")
        print(f"    The velocity field pushes particles outside the bounding box.")
        print(f"    Check: Is 'Displacement' the correct velocity field? Is DT too large?")
    elif misses_inside_mesh_bf > 0:
        print(f"    PRIMARY: L2 search has coverage gaps")
        print(f"    {misses_inside_mesh_bf} positions are inside the mesh but L2 returns -1.")
        print(f"    The 3x3x3 multi-level octree search misses these elements.")
    elif misses_outside_mesh_bf > 0 and misses_outside_bbox == 0:
        print(f"    PRIMARY: Particles move to mesh void regions during sub-steps")
        print(f"    Positions are inside bbox but not inside any element.")
        print(f"    This could be mesh gaps at part boundaries or thin slivers.")
    elif total_misses == 0:
        print(f"    No misses in single-particle replay but vmapped RK4 loses particles.")
        print(f"    This points to a vmap-specific compilation artifact.")
    else:
        print(f"    Mixed causes. Review Section 8 output for details.")

    print("\nDone.")


if __name__ == '__main__':
    main()
