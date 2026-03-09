#!/usr/bin/env python3
"""
Root Cause Diagnostic for Particle Loss

Investigates 5 suspects:
  1. Refinement face neighboring failure (1:2 or 2:1 element face attachments)
  2. pvtu parts joining affecting neighboring or connectivity
  3. pvtu parts causing velocity jumps
  4. L2 not covering complete 1:2 and 2:1 neighboring
  5. L2 finding wrong elements

Approach:
  For selected vanishing particles, replay the EXACT RK4 sub-steps on CPU
  using the same search hierarchy (L0→L1→L2) and identify WHICH sub-step
  first returns elem_id=-1 and WHY.

  Then checks:
  - Is the failing position inside any element? (brute-force scan)
  - Are there velocity discontinuities at part boundaries?
  - Do face-neighbors at refinement boundaries have gaps?
  - Does L2 search miss elements that brute-force finds?
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
    encode_morton_3d_jax,
    find_cell_by_morton_and_level,
    get_cell_elements,
)
from jaxtrace.gpu.search.aa_detection import precompute_aa_metadata, precompute_element_vertices
from jaxtrace.gpu.search.point_in_tet_methods import set_corrected_metadata, set_inverse_matrices_gpu
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
import jaxtrace.config as config

from jaxtrace.gpu.search.mesh_aligned_point_location import (
    search_mesh_aligned_octree_multi_local,
)

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    HAS_VTK = True
except ImportError:
    HAS_VTK = False
    print("ERROR: vtk not available")
    sys.exit(1)

# =============================================================================
# Configuration
# =============================================================================

MESH_BASE_PATH          = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN       = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME     = 'Displacement'

EXPORT_DIR = Path(
    "output/benchmark_with_export_L-hits/"
    "Mesh-Aligned_Multi-Cell_+_3×3×3_Local_(Option_A_-_Phase_2)"
)

# Which steps to analyze. The diagnostic picks particles that vanish
# between step N and step N+1 and replays the RK4 sub-steps.
STEP_RANGE = (1680, 1690)   # Around the diagonal plane from screenshot

# How many vanishing particles to replay (CPU-intensive!)
MAX_REPLAY = 30

# Sampling: 'sequential' or 'random'
REPLAY_SAMPLING = 'random'
REPLAY_RANDOM_SEED = 42

# RK4 parameters (must match production)
DT = 0.0025
POINT_IN_TET_METHOD = 'inverse'

# Brute-force search limit (elements near the failing position)
BRUTE_FORCE_RADIUS = 0.002  # meters — search all elements with centroid within this radius

# Specific particle IDs to force-inspect (in addition to random selection)
FORCE_INSPECT_PIDS = [265608]  # From user's ParaView selection


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
# CPU search helpers
# =============================================================================

def point_in_tet_cpu(pos, elem_id, connectivity, node_positions, tol=1e-6):
    """CPU point-in-tet. Returns (inside, bary_coords)."""
    nodes_idx = connectivity[elem_id]
    nodes = node_positions[nodes_idx].astype(np.float64)
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


def search_l0_cpu(pos, cached_elem, connectivity, node_positions):
    """L0: Check cached element."""
    if cached_elem < 0:
        return -1
    inside, _ = point_in_tet_cpu(pos, cached_elem, connectivity, node_positions)
    return cached_elem if inside else -1


def search_l1_cpu(pos, cached_elem, connectivity, node_positions, element_neighbors, n_hops=3):
    """L1: Multi-hop neighbor search (simplified — no adaptive hop count)."""
    if cached_elem < 0:
        return -1
    current = cached_elem
    found = False
    for hop in range(n_hops):
        if found:
            break
        neighbors = element_neighbors[current]
        found_this_hop = -1
        first_valid = -1
        for ni in range(4):
            eid = int(neighbors[ni])
            if eid < 0:
                continue
            if first_valid < 0:
                first_valid = eid
            if found_this_hop < 0:
                inside, _ = point_in_tet_cpu(pos, eid, connectivity, node_positions)
                if inside:
                    found_this_hop = eid
        if found_this_hop >= 0:
            return found_this_hop
        if first_valid >= 0:
            current = first_valid
    return -1


def search_3x3x3_cpu(pos, octree_gpu, connectivity, node_positions):
    """CPU 3×3×3 multi-level search (exact replica of GPU kernel)."""
    data    = np.array(octree_gpu.cell_to_elements_data)
    offsets = np.array(octree_gpu.cell_to_elements_offsets)
    morton_offset = int(octree_gpu.morton_offset)
    morton_max    = int(octree_gpu.morton_max_coord)

    for level in range(14, 6, -1):
        cell_size = np.array(octree_gpu.level_cell_sizes[level])
        if np.any(cell_size == 0):
            continue

        i_base = int(np.floor(pos[0] / cell_size[0]))
        j_base = int(np.floor(pos[1] / cell_size[1]))
        k_base = int(np.floor(pos[2] / cell_size[2]))

        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    i_off = int(np.clip(i_base + di + morton_offset, 0, morton_max - 1))
                    j_off = int(np.clip(j_base + dj + morton_offset, 0, morton_max - 1))
                    k_off = int(np.clip(k_base + dk + morton_offset, 0, morton_max - 1))

                    morton = int(encode_morton_3d_jax(
                        jnp.int32(i_off), jnp.int32(j_off), jnp.int32(k_off)
                    ))
                    cell_idx = int(find_cell_by_morton_and_level(
                        jnp.uint64(morton), jnp.uint8(level),
                        octree_gpu.cell_morton_codes, octree_gpu.cell_levels
                    ))
                    if cell_idx < 0:
                        continue

                    start_i = int(offsets[cell_idx])
                    n_e     = int(offsets[cell_idx + 1]) - start_i

                    for off in range(n_e):
                        eid = int(data[start_i + off])
                        inside, _ = point_in_tet_cpu(pos, eid, connectivity, node_positions)
                        if inside:
                            return eid

    return -1


def search_gpu_single(pos_np, octree_gpu):
    """GPU 3×3×3 search for a single position."""
    pos_jax = jnp.array(pos_np, dtype=jnp.float32)
    found, n_tests = search_mesh_aligned_octree_multi_local(
        pos_jax, octree_gpu, max_tests=jnp.int32(600)
    )
    return int(found), int(n_tests)


def brute_force_search(pos, connectivity, node_positions, element_centroids, radius):
    """Brute-force: find ALL elements containing pos within a spatial radius."""
    dists = np.linalg.norm(element_centroids - pos, axis=1)
    candidates = np.where(dists < radius)[0]
    containing = []
    for eid in candidates:
        inside, bary = point_in_tet_cpu(pos, eid, connectivity, node_positions)
        if inside:
            containing.append((int(eid), bary))
    return containing, len(candidates)


def interpolate_velocity_cpu(pos, elem_id, connectivity, node_positions, velocity_field):
    """CPU velocity interpolation (barycentric)."""
    if elem_id < 0:
        return np.zeros(3, dtype=np.float64)
    nodes_idx = connectivity[elem_id]
    nodes = node_positions[nodes_idx].astype(np.float64)
    node_vels = velocity_field[nodes_idx].astype(np.float64)
    p = np.array(pos, dtype=np.float64)

    v0 = nodes[1] - nodes[0]
    v1 = nodes[2] - nodes[0]
    v2 = nodes[3] - nodes[0]
    vp = p - nodes[0]
    M = np.column_stack([v0, v1, v2])
    try:
        bary = np.linalg.solve(M, vp)
    except np.linalg.LinAlgError:
        return np.zeros(3, dtype=np.float64)
    b1, b2, b3 = bary
    b0 = 1.0 - b1 - b2 - b3
    vel = b0 * node_vels[0] + b1 * node_vels[1] + b2 * node_vels[2] + b3 * node_vels[3]
    return vel


# =============================================================================
# Suspect #1: Refinement boundary analysis
# =============================================================================

def analyze_refinement_neighbors(elem_id, connectivity, node_positions, element_neighbors):
    """Check if element is at a refinement boundary (1:2 or 2:1)."""
    nodes_idx = connectivity[elem_id]
    verts = node_positions[nodes_idx]

    # Compute element edge lengths
    edges = []
    for i in range(4):
        for j in range(i+1, 4):
            edges.append(np.linalg.norm(verts[i] - verts[j]))
    edges = np.array(edges)
    min_edge = edges.min()
    max_edge = edges.max()

    neighbors = element_neighbors[elem_id]
    neighbor_info = []
    for fi in range(4):
        nid = int(neighbors[fi])
        if nid < 0:
            neighbor_info.append({'face': fi, 'neighbor': -1, 'ratio': None, 'missing': True})
            continue

        n_nodes = connectivity[nid]
        n_verts = node_positions[n_nodes]
        n_edges = []
        for i in range(4):
            for j in range(i+1, 4):
                n_edges.append(np.linalg.norm(n_verts[i] - n_verts[j]))
        n_edges = np.array(n_edges)
        n_min_edge = n_edges.min()

        ratio = min_edge / n_min_edge if n_min_edge > 0 else float('inf')
        neighbor_info.append({
            'face': fi,
            'neighbor': nid,
            'ratio': ratio,
            'missing': False,
            'is_refinement': abs(ratio - 2.0) < 0.1 or abs(ratio - 0.5) < 0.1,
        })

    return {
        'elem_id': elem_id,
        'min_edge': min_edge,
        'max_edge': max_edge,
        'edge_ratio': max_edge / min_edge,
        'neighbors': neighbor_info,
        'n_missing_faces': sum(1 for n in neighbor_info if n['missing']),
    }


# =============================================================================
# Suspect #2/#3: pvtu part boundary analysis
# =============================================================================

def identify_pvtu_part(elem_id, n_elements, n_parts=64):
    """Estimate which pvtu part an element came from (before dedup)."""
    # Elements are typically loaded in order from pvtu parts
    # This is an approximation — actual part boundaries depend on VTK loading
    elems_per_part = n_elements // n_parts
    return elem_id // elems_per_part


def check_velocity_continuity(pos, elem_id, connectivity, node_positions, velocity_field, element_neighbors):
    """Check velocity continuity between element and its neighbors."""
    if elem_id < 0:
        return None

    vel_center = interpolate_velocity_cpu(pos, elem_id, connectivity, node_positions, velocity_field)
    neighbors = element_neighbors[elem_id]
    results = []

    for fi in range(4):
        nid = int(neighbors[fi])
        if nid < 0:
            results.append({'face': fi, 'neighbor': -1, 'vel_diff': None})
            continue
        vel_neighbor = interpolate_velocity_cpu(pos, nid, connectivity, node_positions, velocity_field)
        diff = np.linalg.norm(vel_center - vel_neighbor)
        results.append({
            'face': fi,
            'neighbor': nid,
            'vel_center': vel_center,
            'vel_neighbor': vel_neighbor,
            'vel_diff': diff,
            'vel_diff_relative': diff / (np.linalg.norm(vel_center) + 1e-12),
        })

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("ROOT CAUSE DIAGNOSTIC: Particle Loss Investigation")
    print("=" * 80)

    # ── 1. Load mesh ──────────────────────────────────────────────────────────
    print("\n[1/8] Loading mesh...")
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=False
    )

    # Keep pre-dedup copies for pvtu part analysis
    n_nodes_raw = node_positions.shape[0]

    node_positions, connectivity, n_dupes, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    n_nodes    = node_positions.shape[0]
    n_elements = connectivity.shape[0]
    print(f"  {n_nodes_raw:,} raw nodes → {n_nodes:,} after dedup ({n_dupes:,} removed)")
    print(f"  {n_elements:,} elements")

    # Precompute centroids for brute-force search
    element_centroids = np.zeros((n_elements, 3), dtype=np.float64)
    for i in range(n_elements):
        element_centroids[i] = node_positions[connectivity[i]].mean(axis=0)

    # ── 2. Build search structures ───────────────────────────────────────────
    print("\n[2/8] Building search structures...")
    aa_metadata      = precompute_aa_metadata(connectivity, node_positions, verbose=False)
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

    connectivity_cpu   = np.array(mesh_gpu.connectivity)
    node_positions_cpu = np.array(mesh_gpu.node_positions)

    # ── 3. Neighbor connectivity health check ────────────────────────────────
    print("\n[3/8] Neighbor connectivity health check...")
    n_missing_faces = 0
    n_elements_with_missing = 0
    missing_face_histogram = defaultdict(int)
    for eid in range(n_elements):
        n_miss = sum(1 for fi in range(4) if element_neighbors[eid, fi] < 0)
        if n_miss > 0:
            n_elements_with_missing += 1
            n_missing_faces += n_miss
            missing_face_histogram[n_miss] += 1

    print(f"  Elements with missing face-neighbors: {n_elements_with_missing:,} / {n_elements:,} "
          f"({100*n_elements_with_missing/n_elements:.1f}%)")
    print(f"  Total missing faces: {n_missing_faces:,}")
    for k in sorted(missing_face_histogram):
        print(f"    {k} missing face(s): {missing_face_histogram[k]:,} elements")

    # ── 4. Load VTU exports and find vanishing particles ─────────────────────
    print(f"\n[4/8] Loading VTU exports for steps {STEP_RANGE}...")
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

    if len(vtu_with_steps) < 2:
        print(f"ERROR: Need at least 2 VTU files in range, found {len(vtu_with_steps)}")
        sys.exit(1)

    snapshots = []
    for f, step_num in vtu_with_steps:
        data = load_vtu(f)
        snap = {
            'step':        step_num,
            'pid_to_pos':  dict(zip(data['particle_ids'], data['positions'])),
            'pid_to_eid':  dict(zip(data['particle_ids'], data['element_ids'])),
            'particle_ids': set(data['particle_ids'].tolist()),
        }
        snapshots.append(snap)
        print(f"  Step {step_num}: {len(snap['particle_ids']):,} active")

    # Find vanishing events
    vanishing_events = []
    for i in range(len(snapshots) - 1):
        snap_before = snapshots[i]
        snap_after  = snapshots[i + 1]
        lost_pids = snap_before['particle_ids'] - snap_after['particle_ids']
        for pid in lost_pids:
            vanishing_events.append({
                'step':        snap_before['step'],
                'particle_id': int(pid),
                'last_pos':    np.array(snap_before['pid_to_pos'][pid], dtype=np.float64),
                'last_elem':   int(snap_before['pid_to_eid'][pid]),
            })

    print(f"  Total vanishing events in range: {len(vanishing_events)}")

    # Check for forced PIDs
    forced_events = [ev for ev in vanishing_events if ev['particle_id'] in FORCE_INSPECT_PIDS]
    if FORCE_INSPECT_PIDS:
        found_pids = {ev['particle_id'] for ev in forced_events}
        for pid in FORCE_INSPECT_PIDS:
            if pid not in found_pids:
                # Try to find in adjacent steps
                print(f"  WARNING: Forced PID {pid} not found vanishing in {STEP_RANGE}. "
                      f"Searching all snapshots...")
                for i in range(len(snapshots)):
                    if pid in snapshots[i]['pid_to_pos']:
                        print(f"    Found at step {snapshots[i]['step']}: "
                              f"pos={snapshots[i]['pid_to_pos'][pid]}, "
                              f"elem={snapshots[i]['pid_to_eid'][pid]}")

    # Select events to replay
    remaining = [ev for ev in vanishing_events if ev['particle_id'] not in FORCE_INSPECT_PIDS]
    n_remaining = min(MAX_REPLAY - len(forced_events), len(remaining))

    if REPLAY_SAMPLING == 'random' and n_remaining > 0:
        rng = np.random.default_rng(REPLAY_RANDOM_SEED)
        indices = rng.choice(len(remaining), size=min(n_remaining, len(remaining)), replace=False)
        indices.sort()
        selected = [remaining[i] for i in indices]
    else:
        selected = remaining[:n_remaining]

    events_to_replay = forced_events + selected
    print(f"  Replaying {len(events_to_replay)} events "
          f"({len(forced_events)} forced + {len(selected)} sampled)")

    # ── 5. RK4 sub-step replay ───────────────────────────────────────────────
    print(f"\n[5/8] RK4 sub-step replay (CPU + GPU verification)...")
    print("  Replaying each vanishing particle's last RK4 step on CPU,")
    print("  checking exactly where the search fails.\n")

    vel_field = velocity_sequence[0]  # Single timestep for now

    replay_results = []

    for ev_idx, ev in enumerate(events_to_replay):
        pid  = ev['particle_id']
        pos  = ev['last_pos'].copy()
        elem = ev['last_elem']
        step = ev['step']

        result = {
            'pid': pid, 'step': step,
            'initial_pos': pos.copy(), 'initial_elem': elem,
            'stages': [],
            'failure_stage': None,
        }

        print(f"\n  {'='*70}")
        print(f"  Particle {pid} @ step {step}")
        print(f"  pos = {pos}")
        print(f"  elem = {elem}")

        # Stage 1: k1 = f(t, y)
        stages = [
            ('k1_search', pos, elem, 'L0(pos, cached)'),
        ]

        # Execute RK4 stages sequentially
        current_pos = pos.copy()
        current_elem = elem

        rk4_positions = [pos.copy()]  # track all positions
        rk4_elems = [elem]

        for stage_name, stage_desc in [
            ('k1', 'Search at pos'),
            ('k2', 'Search at pos + 0.5*dt*vel_k1'),
            ('k3', 'Search at pos + 0.5*dt*vel_k2'),
            ('k4', 'Search at pos + dt*vel_k3'),
            ('final', 'Search at pos + (dt/6)*(k1+2k2+2k3+k4)'),
        ]:
            search_pos = current_pos.copy()

            # CPU L0
            l0_result = search_l0_cpu(search_pos, current_elem, connectivity_cpu, node_positions_cpu)
            # CPU L1
            l1_result = search_l1_cpu(search_pos, current_elem, connectivity_cpu, node_positions_cpu,
                                       element_neighbors) if l0_result < 0 else l0_result
            # CPU L2 (3×3×3)
            l2_result = search_3x3x3_cpu(search_pos, octree_gpu, connectivity_cpu, node_positions_cpu) \
                        if (l0_result < 0 and l1_result < 0) else (l1_result if l1_result >= 0 else l0_result)

            # GPU L2
            gpu_result, gpu_tests = search_gpu_single(search_pos.astype(np.float32), octree_gpu)

            # Brute force (if L2 fails)
            bf_results = []
            bf_n_tested = 0
            if l2_result < 0 and gpu_result < 0:
                bf_results, bf_n_tested = brute_force_search(
                    search_pos, connectivity_cpu, node_positions_cpu,
                    element_centroids, BRUTE_FORCE_RADIUS
                )

            # Determine which level found
            if l0_result >= 0:
                final_elem = l0_result
                level = 'L0'
            elif l1_result >= 0:
                final_elem = l1_result
                level = 'L1'
            elif l2_result >= 0:
                final_elem = l2_result
                level = 'L2_cpu'
            elif gpu_result >= 0:
                final_elem = gpu_result
                level = 'L2_gpu'
            elif bf_results:
                final_elem = bf_results[0][0]
                level = 'BRUTE'
            else:
                final_elem = -1
                level = 'MISS'

            stage_info = {
                'name': stage_name,
                'pos': search_pos.copy(),
                'cached_elem': current_elem,
                'l0': l0_result,
                'l1': l1_result,
                'l2_cpu': l2_result,
                'l2_gpu': gpu_result,
                'gpu_tests': gpu_tests,
                'brute_force': bf_results,
                'bf_n_tested': bf_n_tested,
                'final_elem': final_elem,
                'level': level,
            }
            result['stages'].append(stage_info)

            # Print stage result
            bf_str = f" BF:{[b[0] for b in bf_results]}" if bf_results else ""
            miss_flag = " *** MISS ***" if final_elem < 0 else ""
            print(f"    {stage_name:>5s}: L0={l0_result:>8d}  L1={l1_result:>8d}  "
                  f"L2cpu={l2_result:>8d}  L2gpu={gpu_result:>8d}({gpu_tests:3d}t)  "
                  f"→ {level}{bf_str}{miss_flag}")

            if final_elem < 0 and result['failure_stage'] is None:
                result['failure_stage'] = stage_name

            # Interpolate velocity and advance position
            vel = interpolate_velocity_cpu(
                search_pos, final_elem, connectivity_cpu, node_positions_cpu, vel_field
            )

            # Advance position based on RK4 stage
            if stage_name == 'k1':
                vel_k1 = vel
                current_pos = pos + 0.5 * DT * vel_k1
                current_elem = final_elem
            elif stage_name == 'k2':
                vel_k2 = vel
                current_pos = pos + 0.5 * DT * vel_k2
                current_elem = final_elem
            elif stage_name == 'k3':
                vel_k3 = vel
                current_pos = pos + DT * vel_k3
                current_elem = final_elem
            elif stage_name == 'k4':
                vel_k4 = vel
                current_pos = pos + (DT / 6.0) * (vel_k1 + 2*vel_k2 + 2*vel_k3 + vel_k4)
                current_elem = final_elem
            elif stage_name == 'final':
                pass  # End of RK4

            rk4_positions.append(current_pos.copy())
            rk4_elems.append(final_elem)

            print(f"           vel = [{vel[0]:.6e}, {vel[1]:.6e}, {vel[2]:.6e}]  "
                  f"|v| = {np.linalg.norm(vel):.4e}")

        result['rk4_positions'] = rk4_positions
        result['rk4_elems'] = rk4_elems
        replay_results.append(result)

    # ── 6. SUSPECT 1: Refinement boundary analysis ───────────────────────────
    print(f"\n\n{'='*80}")
    print("SUSPECT 1: Refinement Face Neighboring (1:2 / 2:1)")
    print("=" * 80)

    for res in replay_results:
        if res['failure_stage'] is None:
            continue

        pid = res['pid']
        print(f"\n  Particle {pid} — first failure at stage '{res['failure_stage']}'")

        # Analyze elements at each stage
        for stage in res['stages']:
            elem = stage['cached_elem']
            if elem < 0:
                continue
            ref_info = analyze_refinement_neighbors(elem, connectivity_cpu, node_positions_cpu, element_neighbors)
            n_ref = sum(1 for n in ref_info['neighbors'] if not n['missing'] and n.get('is_refinement', False))
            n_miss = ref_info['n_missing_faces']
            if n_ref > 0 or n_miss > 0:
                print(f"    {stage['name']:>5s}: elem {elem}  edge_ratio={ref_info['edge_ratio']:.2f}  "
                      f"missing_faces={n_miss}  refinement_faces={n_ref}")
                for ni in ref_info['neighbors']:
                    if ni['missing']:
                        print(f"           face {ni['face']}: MISSING NEIGHBOR")
                    elif ni.get('is_refinement', False):
                        print(f"           face {ni['face']}: neighbor {ni['neighbor']}  "
                              f"size_ratio={ni['ratio']:.3f}")

    # ── 7. SUSPECT 2/3: pvtu part boundary & velocity discontinuity ──────────
    print(f"\n\n{'='*80}")
    print("SUSPECT 2/3: pvtu Part Boundary & Velocity Discontinuity")
    print("=" * 80)

    for res in replay_results:
        if res['failure_stage'] is None:
            continue

        pid = res['pid']
        print(f"\n  Particle {pid} — failure at '{res['failure_stage']}'")

        for stage in res['stages']:
            elem = stage['cached_elem']
            if elem < 0:
                continue

            part_id = identify_pvtu_part(elem, n_elements)
            vel_cont = check_velocity_continuity(
                stage['pos'], elem, connectivity_cpu, node_positions_cpu, vel_field, element_neighbors
            )
            if vel_cont is None:
                continue

            max_diff = max((v['vel_diff'] for v in vel_cont if v['vel_diff'] is not None), default=0)
            if max_diff > 1e-4:
                print(f"    {stage['name']:>5s}: elem {elem} (part ~{part_id})  "
                      f"max_vel_diff = {max_diff:.6e}")
                for v in vel_cont:
                    if v['vel_diff'] is not None and v['vel_diff'] > 1e-6:
                        n_part = identify_pvtu_part(v['neighbor'], n_elements)
                        cross = " *** CROSS-PART ***" if n_part != part_id else ""
                        print(f"           face {v['face']}: neighbor {v['neighbor']} (part ~{n_part})  "
                              f"vel_diff={v['vel_diff']:.6e}  "
                              f"rel_diff={v['vel_diff_relative']:.4f}{cross}")

    # ── 8. SUSPECT 4/5: L2 coverage and correctness ─────────────────────────
    print(f"\n\n{'='*80}")
    print("SUSPECT 4/5: L2 Coverage and Correctness")
    print("=" * 80)

    n_gpu_cpu_mismatch = 0
    n_l2_miss_bf_found = 0
    n_total_misses = 0

    for res in replay_results:
        pid = res['pid']
        for stage in res['stages']:
            if stage['l2_gpu'] >= 0 and stage['l2_cpu'] >= 0:
                if stage['l2_gpu'] != stage['l2_cpu']:
                    n_gpu_cpu_mismatch += 1
                    # Verify both are valid
                    inside_gpu, bary_gpu = point_in_tet_cpu(
                        stage['pos'], stage['l2_gpu'], connectivity_cpu, node_positions_cpu
                    )
                    inside_cpu, bary_cpu = point_in_tet_cpu(
                        stage['pos'], stage['l2_cpu'], connectivity_cpu, node_positions_cpu
                    )
                    print(f"  GPU/CPU mismatch: pid={pid} {stage['name']}: "
                          f"gpu={stage['l2_gpu']}(inside={inside_gpu}) "
                          f"cpu={stage['l2_cpu']}(inside={inside_cpu})")

            if stage['final_elem'] < 0:
                n_total_misses += 1

            if stage['brute_force']:
                n_l2_miss_bf_found += 1
                bf_elem = stage['brute_force'][0][0]
                print(f"  L2 MISS but BRUTE found: pid={pid} {stage['name']} "
                      f"pos={stage['pos']}")
                print(f"    Brute-force found elem {bf_elem} "
                      f"(tested {stage['bf_n_tested']} candidates)")
                # Check if this element is in the octree
                octree_data = np.array(octree_gpu.cell_to_elements_data)
                in_octree = bool(np.any(octree_data == bf_elem))
                print(f"    Element {bf_elem} in octree: {in_octree}")
                if in_octree:
                    print(f"    → L2 SEARCH BUG: element is in octree but search missed it!")
                else:
                    print(f"    → OCTREE REGISTRATION BUG: element not registered in octree")

    print(f"\n  Summary:")
    print(f"    Total miss stages: {n_total_misses}")
    print(f"    GPU/CPU L2 mismatches: {n_gpu_cpu_mismatch}")
    print(f"    L2 miss but brute-force found: {n_l2_miss_bf_found}")

    # ── 9. Float32 precision test ────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("SUSPECT 6: Float32 vs Float64 Precision")
    print("=" * 80)
    print("  Testing if float32 rounding causes GPU to compute wrong cell indices...")

    n_precision_issues = 0
    for res in replay_results:
        for stage in res['stages']:
            if stage['final_elem'] < 0 or stage['l2_gpu'] < 0:
                pos64 = stage['pos']
                pos32 = pos64.astype(np.float32)
                pos_roundtrip = pos32.astype(np.float64)
                diff = np.abs(pos64 - pos_roundtrip)
                if np.any(diff > 1e-8):
                    n_precision_issues += 1
                    if n_precision_issues <= 5:
                        print(f"  pid={res['pid']} {stage['name']}: pos64-pos32 diff = {diff}")

                # Check: does CPU search with float32-rounded position also fail?
                cpu_f32 = search_3x3x3_cpu(pos_roundtrip, octree_gpu, connectivity_cpu, node_positions_cpu)
                cpu_f64 = stage['l2_cpu']
                if cpu_f32 != cpu_f64 and (cpu_f32 < 0 or cpu_f64 < 0):
                    print(f"  pid={res['pid']} {stage['name']}: "
                          f"CPU(f32)={cpu_f32} vs CPU(f64)={cpu_f64}  *** PRECISION ISSUE ***")

    print(f"  Total positions with f32 rounding > 1e-8: {n_precision_issues}")

    # ── 10. Overall summary ──────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("OVERALL SUMMARY")
    print("=" * 80)

    failure_stages = defaultdict(int)
    no_failure = 0
    for res in replay_results:
        if res['failure_stage']:
            failure_stages[res['failure_stage']] += 1
        else:
            no_failure += 1

    print(f"\n  Particles replayed: {len(replay_results)}")
    print(f"  No failure (all stages found element): {no_failure}")
    print(f"  Failure by first failing stage:")
    for stage, count in sorted(failure_stages.items()):
        print(f"    {stage}: {count}")

    # Classify failures
    print(f"\n  Failure classification:")
    n_l0_l1_ok_l2_fail = 0
    n_all_fail = 0
    n_velocity_zero = 0
    n_elem_minus1_cascade = 0

    for res in replay_results:
        if res['failure_stage'] is None:
            continue

        # Check if failure is from elem=-1 cascade (velocity becomes zero)
        had_minus1 = False
        for stage in res['stages']:
            if stage['final_elem'] < 0:
                had_minus1 = True
            if had_minus1 and stage['name'] != res['failure_stage']:
                # Check if this stage also fails
                if stage['final_elem'] < 0:
                    n_elem_minus1_cascade += 1
                    break

        # Check velocity at failure
        for stage in res['stages']:
            if stage['name'] == res['failure_stage']:
                vel = interpolate_velocity_cpu(
                    stage['pos'], stage['final_elem'],
                    connectivity_cpu, node_positions_cpu, vel_field
                )
                if np.linalg.norm(vel) < 1e-10:
                    n_velocity_zero += 1
                break

    print(f"    elem=-1 cascading to later stages: {n_elem_minus1_cascade}")
    print(f"    Zero velocity at failure point: {n_velocity_zero}")

    # Check mesh domain coverage
    print(f"\n  Mesh domain (node extents):")
    print(f"    x: [{node_positions_cpu[:, 0].min():.6f}, {node_positions_cpu[:, 0].max():.6f}]")
    print(f"    y: [{node_positions_cpu[:, 1].min():.6f}, {node_positions_cpu[:, 1].max():.6f}]")
    print(f"    z: [{node_positions_cpu[:, 2].min():.6f}, {node_positions_cpu[:, 2].max():.6f}]")

    # Positions of ALL failure stages
    fail_positions = []
    for res in replay_results:
        for stage in res['stages']:
            if stage['final_elem'] < 0:
                fail_positions.append(stage['pos'])

    if fail_positions:
        fail_pos_arr = np.array(fail_positions)
        print(f"\n  Failure position extents ({len(fail_positions)} positions):")
        print(f"    x: [{fail_pos_arr[:, 0].min():.6f}, {fail_pos_arr[:, 0].max():.6f}]")
        print(f"    y: [{fail_pos_arr[:, 1].min():.6f}, {fail_pos_arr[:, 1].max():.6f}]")
        print(f"    z: [{fail_pos_arr[:, 2].min():.6f}, {fail_pos_arr[:, 2].max():.6f}]")

    print(f"\n{'='*80}")
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
