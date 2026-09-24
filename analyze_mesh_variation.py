#!/usr/bin/env python3
"""
Mesh Variation Analyzer for Time-Dependent Mesh Feasibility Study

Loads PVTU mesh files across timesteps and reports:
  1. Per-timestep element and node counts
  2. Connectivity fingerprint (hash) — identifies distinct topologies
  3. Node position differences from first timestep
  4. Bounding box per timestep
  5. Element edge length statistics (min/max/mean)
  6. Refinement level distribution (inferred from edge lengths)
  7. Change localisation — where in the domain topology differs

Outputs:
  - Console summary
  - mesh_variation_report.csv

Usage (on LUMI):
    python analyze_mesh_variation.py \
        --input /path/to/post \
        --pattern "cylA_{timestep}.pvtu" \
        --start 0 --end 2684 --stride 100

    # Quick scan (just 5 samples):
    python analyze_mesh_variation.py \
        --input /path/to/post \
        --pattern "cylA_{timestep}.pvtu" \
        --start 0 --end 2684 --stride 700
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def load_mesh_vtk(mesh_file: Path, load_positions: bool = True):
    """
    Load mesh from a single PVTU file using VTK.

    Returns
    -------
    n_nodes : int
    n_elements : int
    positions : np.ndarray (n_nodes, 3) float64, or None if load_positions=False
    connectivity : np.ndarray (n_elements, 4) int32
    """
    import vtk
    from vtk.util import numpy_support

    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(str(mesh_file))
    reader.Update()
    output = reader.GetOutput()

    if output is None or output.GetPoints() is None:
        raise RuntimeError(f"VTK failed to read {mesh_file}")

    n_nodes = output.GetNumberOfPoints()
    n_elements = output.GetNumberOfCells()

    # Positions
    positions = None
    if load_positions:
        positions = numpy_support.vtk_to_numpy(
            output.GetPoints().GetData()
        ).astype(np.float64).copy()

    # Connectivity (VTK format: [4, n0, n1, n2, n3, 4, n0, ...])
    conn_data = numpy_support.vtk_to_numpy(output.GetCells().GetData())
    connectivity = np.zeros((n_elements, 4), dtype=np.int32)
    for i in range(n_elements):
        connectivity[i] = conn_data[i * 5 + 1: i * 5 + 5]

    return n_nodes, n_elements, positions, connectivity


def connectivity_hash(connectivity: np.ndarray) -> str:
    """Compute a deterministic hash of the connectivity array."""
    return hashlib.sha256(connectivity.tobytes()).hexdigest()[:16]


def compute_edge_lengths(positions: np.ndarray, connectivity: np.ndarray,
                         max_elements: int = 500_000) -> dict:
    """
    Compute element edge length statistics.

    For large meshes, samples up to max_elements elements.
    """
    n_elem = connectivity.shape[0]
    if n_elem > max_elements:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_elem, max_elements, replace=False)
        conn_sample = connectivity[idx]
    else:
        conn_sample = connectivity

    # 6 edges per tetrahedron: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    all_lengths = []
    for a, b in edge_pairs:
        diff = positions[conn_sample[:, a]] - positions[conn_sample[:, b]]
        lengths = np.linalg.norm(diff, axis=1)
        all_lengths.append(lengths)

    all_lengths = np.concatenate(all_lengths)

    return {
        'min': float(np.min(all_lengths)),
        'max': float(np.max(all_lengths)),
        'mean': float(np.mean(all_lengths)),
        'median': float(np.median(all_lengths)),
        'std': float(np.std(all_lengths)),
        'p05': float(np.percentile(all_lengths, 5)),
        'p95': float(np.percentile(all_lengths, 95)),
    }


def compute_refinement_levels(positions: np.ndarray, connectivity: np.ndarray,
                              max_elements: int = 500_000) -> dict:
    """
    Infer refinement levels from element sizes.

    Groups elements by characteristic size into octave bins
    (each level halves edge length).
    """
    n_elem = connectivity.shape[0]
    if n_elem > max_elements:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_elem, max_elements, replace=False)
        conn_sample = connectivity[idx]
    else:
        conn_sample = connectivity

    # Characteristic size = mean edge length per element
    edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    elem_sizes = np.zeros(conn_sample.shape[0])
    for a, b in edge_pairs:
        diff = positions[conn_sample[:, a]] - positions[conn_sample[:, b]]
        elem_sizes += np.linalg.norm(diff, axis=1)
    elem_sizes /= 6.0

    # Assign levels: level 0 = coarsest (largest), based on octave bins
    max_size = np.max(elem_sizes)
    if max_size <= 0:
        return {'levels': {0: len(elem_sizes)}, 'max_size': 0.0}

    # level = floor(log2(max_size / elem_size))
    ratios = max_size / np.maximum(elem_sizes, 1e-15)
    levels = np.floor(np.log2(ratios)).astype(int)
    levels = np.clip(levels, 0, 20)

    unique, counts = np.unique(levels, return_counts=True)
    level_dist = {int(lv): int(ct) for lv, ct in zip(unique, counts)}

    return {
        'levels': level_dist,
        'max_size': float(max_size),
        'n_levels': len(level_dist),
    }


def compare_positions(pos_ref: np.ndarray, pos_cur: np.ndarray) -> dict:
    """Compare node positions between two timesteps.

    When node counts differ (AMR refinement adds nodes), compare on the
    shared index range [0, min(N_ref, N_cur)). For typical AMR this captures
    the displacement of original nodes; new nodes at higher indices are
    excluded from the displacement summary.
    """
    n_ref = pos_ref.shape[0]
    n_cur = pos_cur.shape[0]
    n_shared = min(n_ref, n_cur)

    diff = pos_cur[:n_shared] - pos_ref[:n_shared]
    dist = np.linalg.norm(diff, axis=1)

    return {
        'shape_match': (n_ref == n_cur),
        'n_nodes_ref': int(n_ref),
        'n_nodes_cur': int(n_cur),
        'max_displacement': float(np.max(dist)) if n_shared > 0 else 0.0,
        'mean_displacement': float(np.mean(dist)) if n_shared > 0 else 0.0,
        'median_displacement': float(np.median(dist)) if n_shared > 0 else 0.0,
        'n_moved': int(np.sum(dist > 1e-12)),
        'frac_moved': float(np.mean(dist > 1e-12)) if n_shared > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze mesh topology variation across timesteps"
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Directory containing PVTU files')
    parser.add_argument('--pattern', type=str, default='cylA_{timestep}.pvtu',
                        help='File pattern with {timestep} placeholder')
    parser.add_argument('--start', type=int, default=0,
                        help='First timestep to load')
    parser.add_argument('--end', type=int, default=2684,
                        help='Last timestep to load (inclusive)')
    parser.add_argument('--stride', type=int, default=100,
                        help='Stride between timesteps')
    parser.add_argument('--output', type=str, default='mesh_variation_report.csv',
                        help='Output CSV path')
    parser.add_argument('--edge-stats', action='store_true', default=True,
                        help='Compute edge length statistics (default: on)')
    parser.add_argument('--no-edge-stats', action='store_true',
                        help='Skip edge length statistics (faster)')
    parser.add_argument('--max-sample', type=int, default=500_000,
                        help='Max elements to sample for edge/level stats')

    args = parser.parse_args()
    base_path = Path(args.input)

    if not base_path.exists():
        print(f"ERROR: Input path does not exist: {base_path}")
        sys.exit(1)

    timesteps = list(range(args.start, args.end + 1, args.stride))
    # Always include the last timestep
    if timesteps[-1] != args.end:
        timesteps.append(args.end)

    print(f"Mesh Variation Analysis")
    print(f"=======================")
    print(f"Input:     {base_path}")
    print(f"Pattern:   {args.pattern}")
    print(f"Timesteps: {len(timesteps)} samples from {args.start} to {args.end} (stride={args.stride})")
    print()

    # CSV: open once, write incrementally so partial runs leave usable output.
    import csv
    csv_path = Path(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'timestep', 'status', 'error',
        'n_nodes', 'n_elements', 'conn_hash', 'topology_changed',
        'bbox_min_x', 'bbox_min_y', 'bbox_min_z',
        'bbox_max_x', 'bbox_max_y', 'bbox_max_z',
        'load_time_s',
        'pos_shape_match', 'pos_max_displacement', 'pos_mean_displacement',
        'pos_median_displacement', 'pos_n_moved', 'pos_frac_moved',
        'pos_n_nodes_ref', 'pos_n_nodes_cur',
        'edge_min', 'edge_max', 'edge_mean', 'edge_median', 'edge_std',
        'edge_p05', 'edge_p95',
        'n_refinement_levels', 'refinement_dist',
    ]
    csv_file = open(csv_path, 'w', newline='', buffering=1)
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction='ignore')
    csv_writer.writeheader()
    csv_file.flush()
    print(f"Writing CSV incrementally to: {csv_path}")
    print()

    # Storage for in-memory summary at the end.
    results = []
    ref_positions = None
    ref_conn_hash = None
    distinct_topologies = set()

    for i, ts in enumerate(timesteps):
        fname = args.pattern.replace('{timestep}', str(ts))
        fpath = base_path / fname

        if not fpath.exists():
            print(f"  [{i+1}/{len(timesteps)}] Step {ts:6d}: FILE NOT FOUND ({fname})")
            row = {'timestep': ts, 'status': 'missing'}
            results.append(row)
            csv_writer.writerow(row)
            csv_file.flush()
            continue

        t0 = time.time()
        try:
            n_nodes, n_elements, positions, connectivity = load_mesh_vtk(fpath)
        except Exception as e:
            print(f"  [{i+1}/{len(timesteps)}] Step {ts:6d}: LOAD ERROR: {e}")
            row = {'timestep': ts, 'status': 'error', 'error': str(e)}
            results.append(row)
            csv_writer.writerow(row)
            csv_file.flush()
            continue

        load_time = time.time() - t0

        # Connectivity hash
        c_hash = connectivity_hash(connectivity)
        distinct_topologies.add(c_hash)

        # Bounding box
        bbox_min = positions.min(axis=0)
        bbox_max = positions.max(axis=0)

        row = {
            'timestep': ts,
            'status': 'ok',
            'n_nodes': n_nodes,
            'n_elements': n_elements,
            'conn_hash': c_hash,
            'topology_changed': False,
            'bbox_min_x': bbox_min[0], 'bbox_min_y': bbox_min[1], 'bbox_min_z': bbox_min[2],
            'bbox_max_x': bbox_max[0], 'bbox_max_y': bbox_max[1], 'bbox_max_z': bbox_max[2],
            'load_time_s': load_time,
        }

        # Compare with reference
        if ref_positions is None:
            ref_positions = positions.copy()
            ref_conn_hash = c_hash
            row['topology_changed'] = False
        else:
            row['topology_changed'] = (c_hash != ref_conn_hash)

            # Node position comparison
            pos_cmp = compare_positions(ref_positions, positions)
            row.update({f'pos_{k}': v for k, v in pos_cmp.items()})

        # Edge length statistics
        if args.edge_stats and not args.no_edge_stats:
            edge_stats = compute_edge_lengths(positions, connectivity, args.max_sample)
            row.update({f'edge_{k}': v for k, v in edge_stats.items()})

            # Refinement levels
            ref_levels = compute_refinement_levels(positions, connectivity, args.max_sample)
            row['n_refinement_levels'] = ref_levels['n_levels']
            row['refinement_dist'] = str(ref_levels['levels'])

        results.append(row)
        csv_writer.writerow(row)
        csv_file.flush()

        # Console output
        topo_tag = " TOPOLOGY CHANGED!" if row['topology_changed'] else ""
        pos_info = ""
        if 'pos_max_displacement' in row:
            pos_info = f"  max_disp={row['pos_max_displacement']:.2e}"
            if row.get('pos_frac_moved', 0) > 0:
                pos_info += f"  moved={row['pos_frac_moved']*100:.1f}%"

        edge_info = ""
        if 'edge_min' in row:
            edge_info = f"  edge=[{row['edge_min']:.2e}, {row['edge_max']:.2e}]"

        level_info = ""
        if 'n_refinement_levels' in row:
            level_info = f"  levels={row['n_refinement_levels']}"

        print(
            f"  [{i+1}/{len(timesteps)}] Step {ts:6d}: "
            f"nodes={n_nodes:>10,}  elem={n_elements:>10,}  "
            f"hash={c_hash}  ({load_time:.1f}s)"
            f"{pos_info}{edge_info}{level_info}{topo_tag}"
        )

    # --- Summary ---
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    ok_results = [r for r in results if r.get('status') == 'ok']
    if not ok_results:
        print("No files loaded successfully.")
        sys.exit(1)

    # Topology
    print(f"\nDistinct topologies: {len(distinct_topologies)}")
    if len(distinct_topologies) == 1:
        print("  -> Mesh topology is FIXED across all sampled timesteps.")
        print("  -> No octree rebuild needed. Current fixed-mesh pipeline is sufficient.")
    else:
        print("  -> Mesh topology CHANGES during the simulation!")
        changed_steps = [r['timestep'] for r in ok_results if r.get('topology_changed')]
        print(f"  -> Topology changes detected at steps: {changed_steps}")
        print(f"  -> Will need either pre-built octree set or runtime rebuild.")

    # Element/node counts
    elem_counts = [r['n_elements'] for r in ok_results]
    node_counts = [r['n_nodes'] for r in ok_results]
    print(f"\nElement count: min={min(elem_counts):,}  max={max(elem_counts):,}  "
          f"variation={max(elem_counts)-min(elem_counts):,}")
    print(f"Node count:    min={min(node_counts):,}  max={max(node_counts):,}  "
          f"variation={max(node_counts)-min(node_counts):,}")

    # Position changes
    displacements = [r.get('pos_max_displacement', 0) for r in ok_results
                     if 'pos_max_displacement' in r]
    if displacements:
        print(f"\nNode displacement (from step {ok_results[0]['timestep']}):")
        print(f"  Max displacement: {max(displacements):.6e}")
        print(f"  Mean of max displacements: {np.mean(displacements):.6e}")

        frac_moved = [r.get('pos_frac_moved', 0) for r in ok_results
                      if 'pos_frac_moved' in r]
        if frac_moved:
            print(f"  Fraction of nodes that moved: {max(frac_moved)*100:.2f}% (worst step)")

    # Bounding box variation
    bboxes_min = np.array([[r['bbox_min_x'], r['bbox_min_y'], r['bbox_min_z']]
                           for r in ok_results])
    bboxes_max = np.array([[r['bbox_max_x'], r['bbox_max_y'], r['bbox_max_z']]
                           for r in ok_results])
    print(f"\nBounding box:")
    print(f"  Min corner range: x=[{bboxes_min[:,0].min():.6f}, {bboxes_min[:,0].max():.6f}]  "
          f"y=[{bboxes_min[:,1].min():.6f}, {bboxes_min[:,1].max():.6f}]  "
          f"z=[{bboxes_min[:,2].min():.6f}, {bboxes_min[:,2].max():.6f}]")
    print(f"  Max corner range: x=[{bboxes_max[:,0].min():.6f}, {bboxes_max[:,0].max():.6f}]  "
          f"y=[{bboxes_max[:,1].min():.6f}, {bboxes_max[:,1].max():.6f}]  "
          f"z=[{bboxes_max[:,2].min():.6f}, {bboxes_max[:,2].max():.6f}]")

    bbox_variation = (bboxes_max.max(axis=0) - bboxes_min.min(axis=0)) - \
                     (bboxes_max[0] - bboxes_min[0])
    print(f"  Bbox extent variation: [{bbox_variation[0]:.6e}, {bbox_variation[1]:.6e}, {bbox_variation[2]:.6e}]")

    # Edge length stats
    if ok_results[0].get('edge_min') is not None:
        print(f"\nEdge length statistics (first timestep):")
        r0 = ok_results[0]
        print(f"  Min:    {r0['edge_min']:.6e}")
        print(f"  Max:    {r0['edge_max']:.6e}")
        print(f"  Mean:   {r0['edge_mean']:.6e}")
        print(f"  Ratio:  {r0['edge_max']/r0['edge_min']:.1f}x")

    # Refinement levels
    if ok_results[0].get('n_refinement_levels') is not None:
        print(f"\nRefinement levels (first timestep): {ok_results[0]['n_refinement_levels']}")
        print(f"  Distribution: {ok_results[0]['refinement_dist']}")

    csv_file.close()
    print(f"\nResults written to: {csv_path}")

    # --- Recommendation ---
    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    if len(distinct_topologies) == 1:
        if displacements and max(displacements) > 0:
            print("Topology is fixed but nodes move (ALE/deformation).")
            print("If max displacement is small relative to cell size,")
            print("the current octree remains valid. Otherwise, periodic")
            print("octree rebuild is sufficient.")
        else:
            print("Topology is fixed and nodes are stationary.")
            print("Current fixed-mesh pipeline is fully sufficient.")
            print("No changes needed.")
    elif len(distinct_topologies) <= 10:
        print(f"Found {len(distinct_topologies)} distinct topologies.")
        print("Recommended approach: Pre-built octree set (Phase 1).")
        print(f"GPU memory for {len(distinct_topologies)} octrees: "
              f"~{len(distinct_topologies) * 43:.0f} MB (centroid registration).")
    else:
        print(f"Found {len(distinct_topologies)} distinct topologies.")
        print("Many topology changes detected. Consider:")
        print("  - Full rebuild approach (if changes are infrequent)")
        print("  - GPU re-sort approach (if changes are localised)")
        print("  - Two-level delta buffer (if changes are frequent + small)")


if __name__ == '__main__':
    main()
