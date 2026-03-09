#!/usr/bin/env python3
"""
GPU vs CPU Mismatch Diagnostic (Single-Particle, No vmap)

The root cause diagnostic showed that CPU replay of vanishing particles
finds elements at EVERY sub-step — no failures. Yet these particles vanish
in the actual GPU vmapped RK4.

This diagnostic replays RK4 sub-steps using GPU single-particle searches
(no vmap — avoids OOM) in float32 (matching GPU precision exactly).

Strategy:
  1. Load particle state from VTU at step N (before vanishing)
  2. For each vanishing particle, replay RK4 sub-steps:
     - Use GPU search_mesh_aligned_octree_multi_local (single particle)
     - Use GPU point_in_tet (via config dispatcher)
     - Use GPU velocity interpolation
     - All in float32, matching production
  3. Compare: does the float32 GPU single-particle replay also lose them?
  4. If yes: the problem is in the search/interpolation, not vmap
     If no: the problem is vmap-specific (batching artifact)

Focus: particle ID 265608 and nearby vanishing particles around step 1685.
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
from collections import defaultdict

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
from jaxtrace.gpu.search.point_in_tet_methods import (
    set_corrected_metadata, set_inverse_matrices_gpu,
    point_in_tet_gpu as point_in_tet_dispatcher,
)
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
import jaxtrace.config as config

from jaxtrace.gpu.search.mesh_aligned_point_location import (
    search_mesh_aligned_octree_multi_local,
)

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError:
    print("ERROR: vtk not available")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

MESH_BASE_PATH          = Path("data/FLA/post/0eule")
MESH_FILE_PATTERN       = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME     = 'Displacement'

EXPORT_DIR = Path(
    "output/benchmark_with_export_L-hits/"
    "Mesh-Aligned_Multi-Cell_+_3×3×3_Local_(Option_A_-_Phase_2)"
)

# Steps to analyse: load these consecutive VTU files
STEP_RANGE = (1683, 1690)

# Specific particle IDs to force-inspect
FOCUS_PIDS = [265608]

# Also inspect up to N randomly-sampled vanishing particles
MAX_RANDOM = 20
RANDOM_SEED = 42

DT = 0.0025
POINT_IN_TET_METHOD = 'inverse'

# RK4 sub-step names for display
STAGE_NAMES = ['k1', 'k2', 'k3', 'k4', 'final']


# =============================================================================
# VTU loading
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
# GPU single-particle helpers (float32, matching production exactly)
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


def gpu_search_l0(pos_f32, cached_elem, connectivity_gpu, node_positions_gpu):
    """GPU L0: check cached element."""
    if cached_elem < 0:
        return -1
    inside = gpu_point_in_tet(pos_f32, cached_elem, connectivity_gpu, node_positions_gpu)
    return cached_elem if inside else -1


def gpu_search_l1(pos_f32, cached_elem, connectivity_gpu, node_positions_gpu, neighbors_gpu):
    """GPU L1: face-neighbor hops (simplified 3-hop, sequential)."""
    if cached_elem < 0:
        return -1
    current = cached_elem
    for hop in range(3):
        nbrs = np.array(neighbors_gpu[current])
        found = -1
        first_valid = -1
        for ni in range(4):
            eid = int(nbrs[ni])
            if eid < 0:
                continue
            if first_valid < 0:
                first_valid = eid
            if found < 0:
                inside = gpu_point_in_tet(pos_f32, eid, connectivity_gpu, node_positions_gpu)
                if inside:
                    found = eid
        if found >= 0:
            return found
        if first_valid >= 0:
            current = first_valid
        else:
            break
    return -1


def gpu_search_l0_l1_l2(pos_f32, cached_elem, connectivity_gpu, node_positions_gpu, neighbors_gpu, octree_gpu):
    """Full L0→L1→L2 hierarchy on GPU (single particle)."""
    l0 = gpu_search_l0(pos_f32, cached_elem, connectivity_gpu, node_positions_gpu)
    if l0 >= 0:
        return l0, 'L0'

    l1 = gpu_search_l1(pos_f32, cached_elem, connectivity_gpu, node_positions_gpu, neighbors_gpu)
    if l1 >= 0:
        return l1, 'L1'

    l2, _ = gpu_search_l2(pos_f32, octree_gpu)
    if l2 >= 0:
        return l2, 'L2'

    return -1, 'MISS'


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
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("GPU vs CPU MISMATCH DIAGNOSTIC (single-particle, no vmap)")
    print("=" * 80)

    # ── 1. Load mesh ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading mesh...")
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
    n_nodes = node_positions.shape[0]
    n_elements = connectivity.shape[0]
    print(f"  {n_nodes:,} nodes, {n_elements:,} elements")

    # ── 2. Build search structures ────────────────────────────────────────────
    print("\n[2/5] Building search structures...")
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
    set_corrected_metadata(aa_metadata_gpu, element_vertices_gpu)
    set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)
    config.POINT_IN_TET_METHOD = POINT_IN_TET_METHOD

    velocity_field_gpu = jax.device_put(jnp.array(velocity_sequence[0], dtype=jnp.float32))

    connectivity_gpu   = mesh_gpu.connectivity
    node_positions_gpu = mesh_gpu.node_positions
    neighbors_gpu      = mesh_gpu.element_neighbors

    connectivity_cpu   = np.array(connectivity_gpu)
    node_positions_cpu = np.array(node_positions_gpu)

    # ── 3. Load VTU snapshots & find vanishing particles ──────────────────────
    print(f"\n[3/5] Loading VTU snapshots for steps {STEP_RANGE}...")

    import re
    step_pat = re.compile(r'particles_step_(\d+)\.vtu$')
    vtu_files = sorted(EXPORT_DIR.glob("particles_step_*.vtu"))

    vtu_with_steps = []
    for f in vtu_files:
        m = step_pat.search(f.name)
        if m:
            s = int(m.group(1))
            if STEP_RANGE[0] <= s <= STEP_RANGE[1]:
                vtu_with_steps.append((f, s))

    snapshots = []
    for f, step_num in vtu_with_steps:
        data = load_vtu(f)
        snap = {
            'step':        step_num,
            'pid_to_pos':  dict(zip(data['particle_ids'].tolist(), data['positions'])),
            'pid_to_eid':  dict(zip(data['particle_ids'].tolist(), data['element_ids'].tolist())),
            'particle_ids': set(data['particle_ids'].tolist()),
        }
        snapshots.append(snap)
        print(f"  Step {step_num}: {len(snap['particle_ids']):,} active")

    # Find vanishing events
    vanishing_events = []
    for i in range(len(snapshots) - 1):
        sb = snapshots[i]
        sa = snapshots[i + 1]
        lost = sb['particle_ids'] - sa['particle_ids']
        for pid in lost:
            vanishing_events.append({
                'step':     sb['step'],
                'pid':      pid,
                'pos':      np.array(sb['pid_to_pos'][pid], dtype=np.float32),
                'elem':     int(sb['pid_to_eid'][pid]),
            })

    print(f"  Total vanishing events: {len(vanishing_events)}")

    # Select events: forced PIDs + random sample
    forced = [ev for ev in vanishing_events if ev['pid'] in FOCUS_PIDS]
    if not forced:
        # PID might not vanish in this range — search all snapshots
        for pid in FOCUS_PIDS:
            for snap in snapshots:
                if pid in snap['pid_to_pos']:
                    print(f"  PID {pid} found at step {snap['step']}: "
                          f"pos={snap['pid_to_pos'][pid]}, elem={snap['pid_to_eid'][pid]}")
            # Add it anyway with its last-known state for RK4 replay
            for snap in reversed(snapshots):
                if pid in snap['pid_to_pos']:
                    forced.append({
                        'step': snap['step'],
                        'pid':  pid,
                        'pos':  np.array(snap['pid_to_pos'][pid], dtype=np.float32),
                        'elem': int(snap['pid_to_eid'][pid]),
                    })
                    print(f"  Added PID {pid} at step {snap['step']} for replay "
                          f"(may not vanish in this range)")
                    break

    remaining = [ev for ev in vanishing_events if ev['pid'] not in FOCUS_PIDS]
    n_pick = min(MAX_RANDOM, len(remaining))
    if n_pick > 0:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(remaining), size=n_pick, replace=False)
        idx.sort()
        sampled = [remaining[i] for i in idx]
    else:
        sampled = []

    events = forced + sampled
    print(f"  Replaying {len(events)} particles ({len(forced)} forced + {len(sampled)} random)")

    # ── 4. GPU float32 RK4 single-particle replay ────────────────────────────
    print(f"\n[4/5] GPU float32 RK4 replay (single-particle, no vmap)...")
    print(f"  This uses the SAME GPU search functions as production,")
    print(f"  but called one particle at a time (no vmap batching).\n")

    replay_results = []

    for ev_idx, ev in enumerate(events):
        pid  = ev['pid']
        pos0 = ev['pos'].copy()  # float32
        elem = ev['elem']
        step = ev['step']

        print(f"  {'='*68}")
        print(f"  Particle {pid} @ step {step}")
        print(f"  pos  = [{pos0[0]:.8e}, {pos0[1]:.8e}, {pos0[2]:.8e}]")
        print(f"  elem = {elem}")

        # Check: is the starting element valid?
        if elem >= 0:
            inside_start = gpu_point_in_tet(pos0, elem, connectivity_gpu, node_positions_gpu)
            print(f"  GPU PIT at start: inside={inside_start}")
        else:
            print(f"  *** elem=-1 at start — already lost! ***")

        # RK4 sub-step replay in float32
        pos = pos0.copy()
        cached = elem
        stages = []
        vel_k1 = vel_k2 = vel_k3 = vel_k4 = np.zeros(3, dtype=np.float32)

        for si, stage_name in enumerate(STAGE_NAMES):
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

            # L0 → L1 → L2 search
            found_elem, level = gpu_search_l0_l1_l2(
                search_pos, cached, connectivity_gpu, node_positions_gpu,
                neighbors_gpu, octree_gpu
            )

            # Also do standalone L2 for comparison
            l2_standalone, l2_tests = gpu_search_l2(search_pos, octree_gpu)

            # Interpolate velocity
            vel = gpu_interpolate_velocity(
                search_pos, found_elem, connectivity_gpu, node_positions_gpu, velocity_field_gpu
            )

            # CPU brute-force check if MISS
            bf_elem = -1
            if found_elem < 0 and l2_standalone < 0:
                # Brute-force: check all elements near this position
                centroids = node_positions_cpu[connectivity_cpu].mean(axis=1)
                dists = np.linalg.norm(centroids - search_pos.astype(np.float64), axis=1)
                candidates = np.where(dists < 0.002)[0]  # 2mm radius
                for ceid in candidates:
                    ok, _ = cpu_point_in_tet(search_pos, int(ceid), connectivity_cpu, node_positions_cpu)
                    if ok:
                        bf_elem = int(ceid)
                        break

            stage_info = {
                'name': stage_name,
                'pos': search_pos.copy(),
                'cached': cached,
                'found': found_elem,
                'level': level,
                'l2_standalone': l2_standalone,
                'l2_tests': l2_tests,
                'vel': vel.copy(),
                'bf_elem': bf_elem,
            }
            stages.append(stage_info)

            miss_flag = " *** MISS ***" if found_elem < 0 else ""
            bf_str = f" BF={bf_elem}" if bf_elem >= 0 else ""
            l2s_str = f"L2s={l2_standalone:>8d}({l2_tests:3d}t)" if l2_standalone != found_elem else ""
            print(f"    {stage_name:>5s}: cached={cached:>8d} → {level:>4s} elem={found_elem:>8d}  "
                  f"|v|={np.linalg.norm(vel):.4e}  {l2s_str}{bf_str}{miss_flag}")

            # Update cached elem and velocities for next stage
            cached = found_elem
            if stage_name == 'k1':
                vel_k1 = vel
            elif stage_name == 'k2':
                vel_k2 = vel
            elif stage_name == 'k3':
                vel_k3 = vel
            elif stage_name == 'k4':
                vel_k4 = vel

        # Check if next step's VTU has this particle
        next_step = step + 1
        next_snap = None
        for snap in snapshots:
            if snap['step'] == next_step:
                next_snap = snap
                break
        survived = next_snap is not None and pid in next_snap['particle_ids']
        final_elem = stages[-1]['found']
        final_pos = stages[-1]['pos']

        status = "SURVIVED" if survived else "VANISHED"
        replay_ok = final_elem >= 0
        replay_str = "replay_OK" if replay_ok else "replay_FAIL"
        match = (survived == replay_ok)
        match_str = "MATCH" if match else "*** MISMATCH ***"

        print(f"    ──── VTU says: {status}  |  GPU replay: {replay_str}  |  {match_str}")

        if not match and survived:
            # VTU says survived but replay fails — very interesting
            print(f"    VTU next pos: {next_snap['pid_to_pos'][pid]}")
            print(f"    replay final pos: {final_pos}")

        if not match and not survived and replay_ok:
            # VTU says vanished but replay finds element — also very interesting
            print(f"    replay final elem: {final_elem}, replay final pos: {final_pos}")

        replay_results.append({
            'pid': pid, 'step': step, 'stages': stages,
            'survived_vtu': survived, 'replay_ok': replay_ok,
        })

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print("=" * 80)

    n_match = sum(1 for r in replay_results if r['survived_vtu'] == r['replay_ok'])
    n_mismatch = len(replay_results) - n_match
    n_replay_fail = sum(1 for r in replay_results if not r['replay_ok'])
    n_replay_ok = sum(1 for r in replay_results if r['replay_ok'])
    n_vtu_vanish = sum(1 for r in replay_results if not r['survived_vtu'])
    n_vtu_survive = sum(1 for r in replay_results if r['survived_vtu'])

    print(f"\n  Particles replayed: {len(replay_results)}")
    print(f"  VTU vanished: {n_vtu_vanish}  |  VTU survived: {n_vtu_survive}")
    print(f"  Replay FAIL: {n_replay_fail}  |  Replay OK: {n_replay_ok}")
    print(f"  Match (VTU agrees with replay): {n_match}")
    print(f"  Mismatch: {n_mismatch}")

    if n_mismatch > 0 and n_replay_ok > 0 and n_vtu_vanish > 0:
        # Replay succeeds but VTU says vanished
        print(f"\n  *** KEY FINDING: GPU single-particle replay SUCCEEDS for particles")
        print(f"      that VANISH in production vmapped RK4! ***")
        print(f"  This proves the issue is vmap-specific:")
        print(f"    - lax.cond under vmap evaluates BOTH branches (SELECT)")
        print(f"    - fori_loop bounds may differ across particles")
        print(f"    - The search function may produce wrong results under vmap")
    elif n_replay_fail > 0:
        print(f"\n  GPU single-particle replay also fails — issue is NOT vmap-specific.")
        print(f"  The search/interpolation itself has a problem.")

    # Classify failure stages
    fail_stages = defaultdict(int)
    for r in replay_results:
        if not r['replay_ok']:
            for s in r['stages']:
                if s['found'] < 0:
                    fail_stages[s['name']] += 1

    if fail_stages:
        print(f"\n  Failure stages (replay):")
        for name, count in sorted(fail_stages.items()):
            print(f"    {name}: {count}")

    print(f"\n{'='*80}")
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
