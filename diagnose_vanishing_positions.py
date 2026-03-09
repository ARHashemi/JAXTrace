#!/usr/bin/env python3
"""
Vanishing Position Diagnostic

Strategy:
  1. Load exported VTU files (from benchmark_l2_search_methods_with-export.py).
     Each file has ParticleID + ElementID arrays — active particles only.
     Detect which particle IDs disappear between consecutive steps.
  2. For each vanishing particle:
       - Take its last-known position (from the last step it appeared)
       - Generate probe positions along the velocity direction mimicking
         RK4 sub-steps: pos + frac * dt * vel  for frac in [-0.5 .. 1.5]
  3. Call search_mesh_aligned_octree_multi_local (GPU) on every probe and
     report success/failure.
  4. Inspect octree cell structure at failed probe positions:
       - Which cells exist at each level?
       - How many elements are registered there?
       - Does the cell actually contain the probe (CPU point-in-tet)?
  5. Full CPU 3×3×3 replay at failure positions.
  6. Check whether the last-known element is registered in the octree.
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

# ── Mesh / octree ────────────────────────────────────────────────────────────
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

# ── Search functions ─────────────────────────────────────────────────────────
from jaxtrace.gpu.search.mesh_aligned_point_location import (
    search_mesh_aligned_octree_multi_local,
    search_mesh_aligned_octree_single,
)

# ── VTK reader ───────────────────────────────────────────────────────────────
try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    HAS_VTK = True
except ImportError:
    HAS_VTK = False
    print("WARNING: vtk not available — cannot read VTU files")
    sys.exit(1)

# =============================================================================
# Configuration
# =============================================================================

MESH_BASE_PATH          = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN       = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME     = 'Displacement'

# Exported VTU directory (ParticleID + ElementID arrays required)
EXPORT_DIR = Path(
    "output/benchmark_with_export_fixed/"
    "Mesh-Aligned_Multi-Cell_+_3×3×3_Local_(Option_A_-_Phase_2)"
)

# Step range to scan for vanishing particles.
# Set to None to scan all available VTU files from the beginning.
# Set to (start, end) to scan only VTU files whose embedded step number
# falls in [start, end] inclusive.  Example: STEP_RANGE = (350, 550)
STEP_RANGE = (350, 550)

# ── Inspection selection ──────────────────────────────────────────────────────
# Which vanishing particles to inspect (from the full list of vanishing events
# collected across the STEP_RANGE window, sorted chronologically).
#
# INSPECT_RANGE: skip the first N events and inspect up to MAX_INSPECT from there.
#   None  → start from event 0 (default, same as before)
#   (lo, hi) → inspect only events with list index in [lo, hi) exclusive-right
#              e.g. (200, 400) skips the first 200 events and inspects 200 more
#
# INSPECT_SAMPLING: how to pick events within the selected window.
#   'sequential' → first MAX_INSPECT events in the window (default)
#   'random'     → random sample of MAX_INSPECT events from the window (reproducible)
#
# INSPECT_RANDOM_SEED: seed used when INSPECT_SAMPLING = 'random'
#
INSPECT_RANGE    = None          # e.g. (200, 400) or None
INSPECT_SAMPLING = 'sequential'  # 'sequential' or 'random'
INSPECT_RANDOM_SEED = 42

# Max events to inspect (applies after INSPECT_RANGE + INSPECT_SAMPLING)
MAX_INSPECT = 200

# Probe generation: fracs of (dt * |vel|) displacement along vel direction
# covers the range [pos_in .. pos_final] and slightly beyond
PROBE_DX_FRACTIONS = np.linspace(-0.5, 1.5, 11)

DT = 0.0025
POINT_IN_TET_METHOD = 'inverse'


# =============================================================================
# VTU loading
# =============================================================================

def load_vtu(filepath):
    """
    Load a VTU particle file.  Returns dict with:
      positions:    (N, 3) float32
      particle_ids: (N,)   int32
      element_ids:  (N,)   int32
    """
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
# CPU helpers
# =============================================================================

def point_in_tet_numpy(pos, elem_id, connectivity, node_positions):
    """CPU point-in-tet (inverse / barycentric). Returns (inside, bary_coords)."""
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
    return bool(np.all(coords >= -1e-6)), coords


def octree_cell_info_at_pos(pos, octree_gpu, connectivity_cpu, node_positions_cpu):
    """
    For each octree level (14→7), find the cell that would contain `pos`
    and report its element count + which elements actually contain the point.
    Returns list of dicts (only levels where a cell exists).
    """
    data    = np.array(octree_gpu.cell_to_elements_data)
    offsets = np.array(octree_gpu.cell_to_elements_offsets)
    morton_offset = int(octree_gpu.morton_offset)
    morton_max    = int(octree_gpu.morton_max_coord)
    results = []

    for level in range(14, 6, -1):
        cell_size = np.array(octree_gpu.level_cell_sizes[level])
        if np.any(cell_size == 0):
            continue

        i = int(np.floor(pos[0] / cell_size[0]))
        j = int(np.floor(pos[1] / cell_size[1]))
        k = int(np.floor(pos[2] / cell_size[2]))

        i_off = int(np.clip(i + morton_offset, 0, morton_max - 1))
        j_off = int(np.clip(j + morton_offset, 0, morton_max - 1))
        k_off = int(np.clip(k + morton_offset, 0, morton_max - 1))

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
        n_elems  = int(offsets[cell_idx + 1]) - start_i
        elem_ids = [int(data[start_i + off]) for off in range(min(n_elems, 50))]

        containing = []
        for eid in elem_ids:
            inside, coords = point_in_tet_numpy(pos, eid, connectivity_cpu, node_positions_cpu)
            if inside:
                containing.append((eid, coords))

        results.append({
            'level':      level,
            'cell_size':  cell_size,
            'grid_ijk':   (i, j, k),
            'cell_idx':   cell_idx,
            'n_elements': n_elems,
            'elem_ids':   elem_ids,
            'containing': containing,
        })

    return results


def search_3x3x3_cpu(pos, octree_gpu, connectivity_cpu, node_positions_cpu):
    """
    CPU replica of search_mesh_aligned_octree_multi_local (27-cell 3×3×3).
    Returns (found_elem_id, total_tests, per_level_info).
    """
    data    = np.array(octree_gpu.cell_to_elements_data)
    offsets = np.array(octree_gpu.cell_to_elements_offsets)
    morton_offset = int(octree_gpu.morton_offset)
    morton_max    = int(octree_gpu.morton_max_coord)

    found_elem  = -1
    total_tests = 0
    per_level   = []

    for level in range(14, 6, -1):
        cell_size = np.array(octree_gpu.level_cell_sizes[level])
        if np.any(cell_size == 0):
            continue

        i_base = int(np.floor(pos[0] / cell_size[0]))
        j_base = int(np.floor(pos[1] / cell_size[1]))
        k_base = int(np.floor(pos[2] / cell_size[2]))

        cells_searched = 0
        elems_tested   = 0
        level_found    = -1

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

                    cells_searched += 1
                    start_i = int(offsets[cell_idx])
                    n_e     = int(offsets[cell_idx + 1]) - start_i

                    for off in range(n_e):
                        eid = int(data[start_i + off])
                        inside, _ = point_in_tet_numpy(
                            pos, eid, connectivity_cpu, node_positions_cpu
                        )
                        elems_tested += 1
                        total_tests  += 1
                        if inside and found_elem < 0:
                            found_elem  = eid
                            level_found = eid

        per_level.append({
            'level':         level,
            'cells_searched': cells_searched,
            'elems_tested':  elems_tested,
            'found_elem':    level_found,
        })
        if found_elem >= 0:
            break

    return found_elem, total_tests, per_level


def is_element_in_octree(elem_id, octree_gpu):
    """Check if element appears in any octree cell."""
    data = np.array(octree_gpu.cell_to_elements_data)
    return bool(np.any(data == elem_id))


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("Vanishing Position Diagnostic")
    print("=" * 80)

    # ── 1. Verify VTU directory ───────────────────────────────────────────────
    vtu_files = sorted(EXPORT_DIR.glob("particles_step_*.vtu"))
    if not vtu_files:
        print(f"ERROR: No VTU files found in {EXPORT_DIR}")
        sys.exit(1)
    print(f"\n[1/6] Found {len(vtu_files)} VTU files in:")
    print(f"      {EXPORT_DIR}")

    # ── 2. Load mesh ──────────────────────────────────────────────────────────
    print("\n[2/6] Loading mesh...")
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
    n_nodes    = node_positions.shape[0]
    n_elements = connectivity.shape[0]
    print(f"  {n_nodes:,} nodes, {n_elements:,} elements")

    # ── 3. Precompute metadata & upload ──────────────────────────────────────
    print("\n[3/6] Precomputing metadata and uploading to GPU...")
    aa_metadata      = precompute_aa_metadata(connectivity, node_positions, verbose=False)
    element_vertices = precompute_element_vertices(connectivity, node_positions, verbose=False)
    M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)

    element_neighbors = build_element_neighbors_array(connectivity, method='face', verbose=False)
    mesh_gpu = upload_mesh_to_gpu(connectivity, node_positions, element_neighbors, verbose=False)

    print("  Building mesh-aligned octree (multi-cell vertex registration)...")
    mesh_octree_cells_multi = extract_octree_cells_vertex_multi(
        node_positions, connectivity, tolerance=1e-6, verbose=False
    )
    print(f"    {mesh_octree_cells_multi.n_cells:,} cells, "
          f"{mesh_octree_cells_multi.elements_per_cell_mean:.2f} elem/cell, "
          f"{mesh_octree_cells_multi.cells_per_element_mean:.2f} cells/elem")

    mesh_aligned_octree_multi_gpu = upload_mesh_aligned_octree_to_gpu(
        connectivity, node_positions, mesh_octree_cells_multi, verbose=False
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
    M_inv_gpu            = jax.device_put(M_inv_array)
    p0_gpu               = jax.device_put(p0_array)
    set_corrected_metadata(aa_metadata_gpu, element_vertices_gpu)
    set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)
    config.POINT_IN_TET_METHOD = POINT_IN_TET_METHOD

    # Keep CPU copies for inspection
    connectivity_cpu    = np.array(mesh_gpu.connectivity)
    node_positions_cpu  = np.array(mesh_gpu.node_positions)
    elem_neighbors_cpu  = element_neighbors

    print("  Uploaded to GPU")

    # ── 4. Scan VTU files for vanishing particles ────────────────────────────
    # Parse step number from each filename (e.g. "particles_step_0042.vtu" → 42)
    import re as _re
    _step_pat = _re.compile(r'particles_step_(\d+)\.vtu$')

    def _parse_step(p):
        m = _step_pat.search(p.name)
        return int(m.group(1)) if m else -1

    vtu_with_steps = [(f, _parse_step(f)) for f in vtu_files]
    vtu_with_steps = [(f, s) for f, s in vtu_with_steps if s >= 0]  # skip unparseable

    if STEP_RANGE is not None:
        lo, hi = STEP_RANGE
        vtu_with_steps = [(f, s) for f, s in vtu_with_steps if lo <= s <= hi]
        range_label = f"steps {lo}–{hi}"
    else:
        range_label = f"all {len(vtu_with_steps)} VTU files"

    if not vtu_with_steps:
        print(f"ERROR: No VTU files match STEP_RANGE={STEP_RANGE}")
        sys.exit(1)

    print(f"\n[4/6] Scanning {len(vtu_with_steps)} VTU files ({range_label}) "
          f"for vanishing particles...")

    # Build per-step snapshot: ParticleID → (position, element_id)
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
        print(f"  Loaded step {step_num:5d}: {len(snap['particle_ids']):,} active particles")

    # Detect vanishing events: particle present in step i but absent in step i+1
    vanishing_events = []
    print(f"\n  {'Step':>5s}  {'Active→':>10s}  {'→Active':>10s}  {'Lost':>7s}  {'Cum.Lost':>9s}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*9}")
    cumulative_lost = 0

    for i in range(len(snapshots) - 1):
        snap_before = snapshots[i]
        snap_after  = snapshots[i + 1]

        lost_pids = snap_before['particle_ids'] - snap_after['particle_ids']
        cumulative_lost += len(lost_pids)
        print(f"  {snap_before['step']:5d}  {len(snap_before['particle_ids']):10,d}  "
              f"{len(snap_after['particle_ids']):10,d}  {len(lost_pids):7d}  "
              f"{cumulative_lost:9d}")

        for pid in lost_pids:
            last_pos  = snap_before['pid_to_pos'][pid]
            last_elem = int(snap_before['pid_to_eid'][pid])

            # Get velocity at last-known element for probe direction
            vel_field  = velocity_sequence[snap_before['step'] % len(velocity_sequence)]
            nodes_idx  = connectivity[last_elem]
            node_vels  = vel_field[nodes_idx]
            vel_approx = node_vels.mean(axis=0).astype(np.float32)

            vanishing_events.append({
                'step':       snap_before['step'],
                'particle_id': int(pid),
                'last_pos':    np.array(last_pos, dtype=np.float32),
                'last_elem':   last_elem,
                'vel_approx':  vel_approx,
            })

    total_events = len(vanishing_events)
    print(f"\n  Total vanishing events: {total_events}")

    # --- Apply INSPECT_RANGE to slice the candidate pool ---
    if INSPECT_RANGE is not None:
        lo_idx, hi_idx = INSPECT_RANGE
        candidate_pool = vanishing_events[lo_idx:hi_idx]
        range_desc = f"events [{lo_idx}, {hi_idx})"
    else:
        candidate_pool = vanishing_events
        range_desc = f"all {total_events} events"

    # --- Apply INSPECT_SAMPLING to pick events from the pool ---
    if INSPECT_SAMPLING == 'random':
        rng = np.random.default_rng(INSPECT_RANDOM_SEED)
        n_pick = min(MAX_INSPECT, len(candidate_pool))
        chosen_indices = rng.choice(len(candidate_pool), size=n_pick, replace=False)
        chosen_indices.sort()  # keep chronological order for readability
        events_to_inspect = [candidate_pool[i] for i in chosen_indices]
        sample_desc = f"random sample of {len(events_to_inspect)} from {range_desc}"
    else:
        events_to_inspect = candidate_pool[:MAX_INSPECT]
        sample_desc = f"first {len(events_to_inspect)} from {range_desc}"

    print(f"  Inspecting: {sample_desc}")

    # ── 5. Probe analysis ─────────────────────────────────────────────────────
    print(f"\n[5/6] Probe position analysis...")
    print(f"  {len(PROBE_DX_FRACTIONS)} probes per event × {len(events_to_inspect)} events "
          f"= {len(PROBE_DX_FRACTIONS) * len(events_to_inspect)} total probes")

    probe_results      = []
    search_fail_list   = []
    n_probe_found      = 0
    n_probe_failed     = 0

    for ev_idx, ev in enumerate(events_to_inspect):
        last_pos   = ev['last_pos']
        vel_approx = ev['vel_approx']
        last_elem  = ev['last_elem']

        vel_mag    = float(np.linalg.norm(vel_approx))
        disp_scale = DT * vel_mag if vel_mag > 1e-12 else 1e-6
        vel_dir    = vel_approx / (vel_mag + 1e-12)

        ev_probes = []
        for frac in PROBE_DX_FRACTIONS:
            probe_pos = last_pos + frac * disp_scale * vel_dir

            # GPU 3×3×3 search (single-particle, no vmap — avoids OOM)
            probe_jax = jnp.array(probe_pos, dtype=jnp.float32)
            found_gpu, n_tests = search_mesh_aligned_octree_multi_local(
                probe_jax, mesh_aligned_octree_multi_gpu, max_tests=jnp.int32(600)
            )
            found_gpu = int(found_gpu)
            n_tests   = int(n_tests)

            # CPU: scan last_elem + its face neighbors
            adj_elems = [last_elem] + [int(e) for e in elem_neighbors_cpu[last_elem] if e >= 0]
            cpu_found = -1
            for eid in adj_elems:
                inside, _ = point_in_tet_numpy(probe_pos, eid, connectivity_cpu, node_positions_cpu)
                if inside:
                    cpu_found = eid
                    break

            if found_gpu >= 0:
                n_probe_found += 1
            else:
                n_probe_failed += 1
                search_fail_list.append({
                    'event_idx': ev_idx,
                    'pos':       probe_pos.copy(),
                    'frac':      frac,
                    'last_elem': last_elem,
                    'cpu_found': cpu_found,
                    'step':      ev['step'],
                })

            ev_probes.append({
                'frac':      frac,
                'probe_pos': probe_pos.copy(),
                'found_gpu': found_gpu,
                'n_tests':   n_tests,
                'cpu_found': cpu_found,
            })

        probe_results.append({'event': ev, 'probes': ev_probes})

        if ev_idx % 50 == 0:
            print(f"  ... {ev_idx}/{len(events_to_inspect)} events processed")

    total_probes = len(events_to_inspect) * len(PROBE_DX_FRACTIONS)
    print(f"\n  GPU 3×3×3 found:  {n_probe_found}/{total_probes} "
          f"({100*n_probe_found/total_probes:.1f}%)")
    print(f"  GPU 3×3×3 failed: {n_probe_failed}/{total_probes} "
          f"({100*n_probe_failed/total_probes:.1f}%)")

    # ── 6. Detailed reporting ─────────────────────────────────────────────────
    print("\n[6/6] Detailed reporting")

    # --- Per-event detail (first 20) ---
    print("\n" + "=" * 80)
    print("PER-EVENT DETAILS (first 20 events)")
    print("=" * 80)

    for ev_idx, pr in enumerate(probe_results[:20]):
        ev = pr['event']
        print(f"\n  Event {ev_idx:3d}: step={ev['step']} pid={ev['particle_id']} "
              f"elem={ev['last_elem']}")
        print(f"    last_pos   = {ev['last_pos']}")
        print(f"    vel_approx = {ev['vel_approx']}  |v|={np.linalg.norm(ev['vel_approx']):.4e}")

        print(f"    {'frac':>6s}  {'GPU':>10s}  {'tests':>6s}  {'CPU_adj':>9s}  note")
        print(f"    {'-'*6}  {'-'*10}  {'-'*6}  {'-'*9}  ----")
        any_fail = False
        for p in pr['probes']:
            gpu_str = str(p['found_gpu']) if p['found_gpu'] >= 0 else 'FAIL'
            note    = ''
            if p['found_gpu'] < 0 and p['cpu_found'] >= 0:
                note = '<-- CPU finds but GPU fails (registration gap?)'
            elif p['found_gpu'] < 0 and p['cpu_found'] < 0:
                note = '<-- both fail (position outside covered region)'
            print(f"    {p['frac']:6.2f}  {gpu_str:>10s}  {p['n_tests']:6d}  "
                  f"{p['cpu_found']:9d}  {note}")
            if p['found_gpu'] < 0:
                any_fail = True

        # Full octree inspection at first failure
        if any_fail:
            first_fail = next(p for p in pr['probes'] if p['found_gpu'] < 0)
            fail_pos   = first_fail['probe_pos']
            print(f"\n    OCTREE CELLS at first failure (frac={first_fail['frac']:.2f}):")
            print(f"    pos = {fail_pos}")
            cell_info = octree_cell_info_at_pos(
                fail_pos, mesh_aligned_octree_multi_gpu,
                connectivity_cpu, node_positions_cpu
            )
            if not cell_info:
                print(f"    --> NO CELLS EXIST AT ANY LEVEL")
            else:
                for ci in cell_info:
                    cont_ids = [c[0] for c in ci['containing']]
                    print(f"    lv{ci['level']:2d}: cell_size={ci['cell_size']}  "
                          f"n_elem={ci['n_elements']:4d}  "
                          f"containing={cont_ids}")

            # CPU 3×3×3 replay
            print(f"\n    CPU 3×3×3 replay:")
            cpu_found_3x3, cpu_tests, cpu_levels = search_3x3x3_cpu(
                fail_pos, mesh_aligned_octree_multi_gpu,
                connectivity_cpu, node_positions_cpu
            )
            print(f"    result: found={cpu_found_3x3}, total_tests={cpu_tests}")
            for lv in cpu_levels:
                print(f"      lv{lv['level']:2d}: cells={lv['cells_searched']:3d}  "
                      f"tests={lv['elems_tested']:4d}  found={lv['found_elem']}")

    # --- Failure breakdown by frac ---
    print("\n" + "=" * 80)
    print("FAILURE DISTRIBUTION BY RK4 FRACTION")
    print("=" * 80)
    print("(frac=0.0 is last-known position, frac=0.5 is pos_k1/k2, frac=1.0 is pos_final)")
    print()
    for frac in PROBE_DX_FRACTIONS:
        count_fail = sum(1 for sf in search_fail_list if abs(sf['frac'] - frac) < 0.01)
        count_cpu  = sum(1 for sf in search_fail_list
                         if abs(sf['frac'] - frac) < 0.01 and sf['cpu_found'] >= 0)
        bar = '#' * min(count_fail, 80)
        print(f"  frac={frac:5.2f}  gpu_fail={count_fail:5d}  "
              f"(cpu_finds={count_cpu:5d})  {bar}")

    # --- Failure by last element ---
    print("\n" + "=" * 80)
    print("FAILURE CLUSTERS BY LAST-KNOWN ELEMENT (top 15)")
    print("=" * 80)
    fail_by_elem = defaultdict(list)
    for sf in search_fail_list:
        fail_by_elem[sf['last_elem']].append(sf)

    print(f"\n  {'elem_id':>10s}  {'n_fail':>7s}  {'cpu_finds':>10s}  "
          f"{'in_octree':>10s}  pos_x_range")
    print(f"  {'-'*10}  {'-'*7}  {'-'*10}  {'-'*10}  ----------")

    for elem_id, fails in sorted(fail_by_elem.items(), key=lambda x: -len(x))[:15]:
        n_cpu   = sum(1 for f in fails if f['cpu_found'] >= 0)
        in_oct  = is_element_in_octree(elem_id, mesh_aligned_octree_multi_gpu)
        pos_arr = np.array([f['pos'] for f in fails])
        x_range = f"[{pos_arr[:,0].min():.5f}, {pos_arr[:,0].max():.5f}]"
        print(f"  {elem_id:10d}  {len(fails):7d}  {n_cpu:10d}  "
              f"{'YES' if in_oct else 'NO ':>10s}  {x_range}")

    # --- Deep inspection of top-5 loss elements ---
    print("\n" + "=" * 80)
    print("DEEP OCTREE INSPECTION: TOP 5 LOSS ELEMENTS")
    print("=" * 80)

    top5 = sorted(fail_by_elem.items(), key=lambda x: -len(x))[:5]
    for elem_id, fails in top5:
        in_oct = is_element_in_octree(elem_id, mesh_aligned_octree_multi_gpu)
        print(f"\n  Element {elem_id}  ({len(fails)} failures)  "
              f"in_octree={'YES' if in_oct else 'NO'}")

        nodes_idx = connectivity_cpu[elem_id]
        verts     = node_positions_cpu[nodes_idx]
        centroid  = verts.mean(axis=0)
        print(f"  Centroid: {centroid}")
        for vi, v in enumerate(verts):
            print(f"    vertex {vi}: {v}")

        adj_str = str(elem_neighbors_cpu[elem_id])
        print(f"  Face neighbors: {adj_str}")

        # Octree cells containing centroid
        print(f"  Octree cells at centroid:")
        ci_list = octree_cell_info_at_pos(
            centroid, mesh_aligned_octree_multi_gpu,
            connectivity_cpu, node_positions_cpu
        )
        if not ci_list:
            print(f"    NO CELLS AT ANY LEVEL (registration failure!)")
        else:
            for ci in ci_list:
                reg = 'REG' if elem_id in ci['elem_ids'] else '---'
                cont = [c[0] for c in ci['containing']]
                print(f"    lv{ci['level']:2d}: n_elem={ci['n_elements']:4d}  "
                      f"self={reg}  containing={cont}")

        # CPU 3×3×3 at first failure position
        fail_pos = fails[0]['pos']
        print(f"  CPU 3×3×3 at first failure pos {fail_pos}:")
        cpu_found_3x3, cpu_tests, cpu_levels = search_3x3x3_cpu(
            fail_pos, mesh_aligned_octree_multi_gpu,
            connectivity_cpu, node_positions_cpu
        )
        print(f"    result: found={cpu_found_3x3}, total_tests={cpu_tests}")
        for lv in cpu_levels:
            print(f"      lv{lv['level']:2d}: cells={lv['cells_searched']:3d}  "
                  f"tests={lv['elems_tested']:4d}  found={lv['found_elem']}")

        # CPU point-in-tet directly at failure positions
        n_inside = sum(
            1 for f in fails
            if point_in_tet_numpy(f['pos'], elem_id, connectivity_cpu, node_positions_cpu)[0]
        )
        print(f"  Direct CPU PIT: last_elem contains {n_inside}/{len(fails)} "
              f"failure positions")

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    step_range_str = (f"{snapshots[0]['step']}–{snapshots[-1]['step']}"
                      if snapshots else "none")
    print(f"\n  VTU steps scanned:       {len(snapshots) - 1} "
          f"(steps {step_range_str})")
    print(f"  Total vanishing events:  {total_events}")
    print(f"  Events inspected:        {len(events_to_inspect)}")
    print(f"  Probe positions tested:  {total_probes}")
    print(f"  GPU 3×3×3 found:         {n_probe_found} ({100*n_probe_found/max(total_probes,1):.1f}%)")
    print(f"  GPU 3×3×3 failed:        {n_probe_failed} ({100*n_probe_failed/max(total_probes,1):.1f}%)")

    if search_fail_list:
        cpu_at_gpu_fail = sum(1 for sf in search_fail_list if sf['cpu_found'] >= 0)
        pct = 100 * cpu_at_gpu_fail / len(search_fail_list)
        print(f"\n  Of GPU failures:")
        print(f"    CPU (adjacent scan) finds: {cpu_at_gpu_fail}/{len(search_fail_list)} "
              f"({pct:.1f}%)")
        print(f"    --> If high %: element IS in mesh but NOT in octree → registration gap")
        print(f"    --> If low %:  probe position left covered mesh region")

        no_reg = [e for e, _ in fail_by_elem.items()
                  if not is_element_in_octree(e, mesh_aligned_octree_multi_gpu)]
        print(f"\n  Elements NOT in octree (of {len(fail_by_elem)} loss elements): {len(no_reg)}")
        if no_reg:
            print(f"    {no_reg[:20]}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
