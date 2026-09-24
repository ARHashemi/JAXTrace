#!/usr/bin/env python3
"""
Fine-zone envelope scan for reference-mesh design.

Two outputs:
  1. List all PointData and CellData fields available in one PVTU
     (used to identify velocity / temperature / level-set field names
     for the projection preprocessor).
  2. For each sampled cycle index, compute the bounding box of the
     elements at the finest refinement level. Take the union across
     all sampled indices — this is the *swept volume* that the
     reference mesh must resolve at fine resolution.

Usage on LUMI:
    python scan_fine_zone_envelope.py \
        --input /scratch/.../C3.gid/post \
        --pattern "C3_{timestep}.pvtu" \
        --start 0 --end 149 --stride 10 \
        --finest-frac 0.20 \
        --output fine_zone_envelope.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def load_pvtu(mesh_file):
    import vtk
    from vtk.util import numpy_support
    if not mesh_file.exists():
        raise FileNotFoundError(mesh_file)
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(str(mesh_file))
    reader.Update()
    out = reader.GetOutput()
    if out is None or out.GetPoints() is None:
        raise RuntimeError(f"VTK failed to read {mesh_file}")
    positions = numpy_support.vtk_to_numpy(out.GetPoints().GetData()).astype(np.float64).copy()
    n_cells = out.GetNumberOfCells()
    conn_data = numpy_support.vtk_to_numpy(out.GetCells().GetData())
    connectivity = np.zeros((n_cells, 4), dtype=np.int32)
    for i in range(n_cells):
        connectivity[i] = conn_data[i * 5 + 1: i * 5 + 5]
    return positions, connectivity, out


def list_fields(out):
    pd = out.GetPointData()
    cd = out.GetCellData()
    point_fields = []
    for i in range(pd.GetNumberOfArrays()):
        a = pd.GetArray(i)
        point_fields.append({
            'name': pd.GetArrayName(i),
            'components': a.GetNumberOfComponents(),
            'dtype': a.GetDataTypeAsString(),
            'tuples': a.GetNumberOfTuples(),
        })
    cell_fields = []
    for i in range(cd.GetNumberOfArrays()):
        a = cd.GetArray(i)
        cell_fields.append({
            'name': cd.GetArrayName(i),
            'components': a.GetNumberOfComponents(),
            'dtype': a.GetDataTypeAsString(),
            'tuples': a.GetNumberOfTuples(),
        })
    return point_fields, cell_fields


def element_mean_edge(positions, connectivity):
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    s = np.zeros(connectivity.shape[0])
    for a, b in pairs:
        s += np.linalg.norm(positions[connectivity[:, a]]
                            - positions[connectivity[:, b]], axis=1)
    return s / 6.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, required=True)
    p.add_argument('--pattern', type=str, required=True)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=149)
    p.add_argument('--stride', type=int, default=10)
    p.add_argument('--finest-frac', type=float, default=0.20,
                   help='Fraction of smallest-edge elements considered fine (default 0.20)')
    p.add_argument('--output', type=str, default='fine_zone_envelope.json')
    args = p.parse_args()

    base = Path(args.input)
    timesteps = list(range(args.start, args.end + 1, args.stride))
    if timesteps[-1] != args.end:
        timesteps.append(args.end)

    # First sample: report fields.
    f0 = base / args.pattern.replace('{timestep}', str(timesteps[0]))
    print(f"Loading first sample for field listing: {f0}")
    pos0, conn0, out0 = load_pvtu(f0)
    pf, cf = list_fields(out0)
    print()
    print("=== POINT-DATA fields ===")
    for x in pf:
        print(f"  {x['name']:<30}  components={x['components']:<3}  dtype={x['dtype']}  tuples={x['tuples']:,}")
    print("=== CELL-DATA fields ===")
    for x in cf:
        print(f"  {x['name']:<30}  components={x['components']:<3}  dtype={x['dtype']}  tuples={x['tuples']:,}")
    print()

    # Compute global edge stats from this sample (used as fine-edge threshold).
    edges = element_mean_edge(pos0, conn0)
    e_min = float(edges.min())
    e_max = float(edges.max())
    fine_threshold = float(np.quantile(edges, args.finest_frac))
    print(f"Edge stats on first sample: min={e_min:.4e}  max={e_max:.4e}")
    print(f"Fine threshold ({args.finest_frac*100:.0f}-percentile edge): {fine_threshold:.4e}")
    print()

    bbox_full_min = pos0.min(axis=0)
    bbox_full_max = pos0.max(axis=0)
    print(f"Full-mesh bbox: [{bbox_full_min}] -> [{bbox_full_max}]")
    print()

    # Loop and accumulate fine-zone bbox.
    fine_envelope_min = np.full(3, +np.inf)
    fine_envelope_max = np.full(3, -np.inf)
    per_step = []

    print(f"Scanning {len(timesteps)} samples from {args.start} to {args.end} (stride {args.stride})")
    for i, ts in enumerate(timesteps):
        t0 = time.time()
        f = base / args.pattern.replace('{timestep}', str(ts))
        try:
            pos, conn, _ = load_pvtu(f)
        except Exception as e:
            print(f"  [{i+1}/{len(timesteps)}] Step {ts}: load failed ({e})")
            continue
        edges = element_mean_edge(pos, conn)
        fine_mask = edges <= fine_threshold
        n_fine = int(fine_mask.sum())
        if n_fine == 0:
            print(f"  [{i+1}/{len(timesteps)}] Step {ts}: no fine elements (threshold too tight)")
            continue
        # Element-centroid bbox of the fine subset.
        centroids = pos[conn[fine_mask]].mean(axis=1)
        f_min = centroids.min(axis=0)
        f_max = centroids.max(axis=0)
        fine_envelope_min = np.minimum(fine_envelope_min, f_min)
        fine_envelope_max = np.maximum(fine_envelope_max, f_max)
        per_step.append({
            'step': ts,
            'n_elements': int(conn.shape[0]),
            'n_fine': n_fine,
            'frac_fine': float(n_fine / conn.shape[0]),
            'fine_bbox_min': f_min.tolist(),
            'fine_bbox_max': f_max.tolist(),
        })
        print(f"  [{i+1}/{len(timesteps)}] Step {ts:4d}: n_fine={n_fine:>9,} ({100*n_fine/conn.shape[0]:.2f}%)  "
              f"bbox_x=[{f_min[0]:+.4e}, {f_max[0]:+.4e}]  "
              f"({time.time()-t0:.1f}s)")

    print()
    print("=" * 80)
    print("RESULT: union of fine-zone bboxes (the SWEPT VOLUME)")
    print("=" * 80)
    print(f"Fine envelope min: [{fine_envelope_min}]")
    print(f"Fine envelope max: [{fine_envelope_max}]")
    extent = fine_envelope_max - fine_envelope_min
    full_extent = bbox_full_max - bbox_full_min
    print(f"Fine envelope extent:    [{extent}]")
    print(f"Full bbox extent:        [{full_extent}]")
    frac = float(np.prod(extent) / np.prod(full_extent))
    print(f"Volume fraction of fine envelope: {frac:.4f} ({frac*100:.2f}%)")
    print()
    print("Reference mesh design implication:")
    if frac < 0.5:
        print(f"  Use fine resolution (~{e_min:.2e}) inside the swept volume,")
        print(f"  coarse resolution outside. Fine-zone savings: ~{(1-frac)*100:.0f}%")
    else:
        print(f"  Swept volume covers most of the bbox; uniform fine mesh is")
        print(f"  the cleanest choice (saving from refinement masking is small).")

    # JSON output
    result = {
        'pattern': args.pattern,
        'start': args.start, 'end': args.end, 'stride': args.stride,
        'finest_frac': args.finest_frac,
        'fine_threshold': fine_threshold,
        'edge_min_first': e_min,
        'edge_max_first': e_max,
        'bbox_full_min': bbox_full_min.tolist(),
        'bbox_full_max': bbox_full_max.tolist(),
        'fine_envelope_min': fine_envelope_min.tolist(),
        'fine_envelope_max': fine_envelope_max.tolist(),
        'fine_envelope_volume_fraction': frac,
        'point_fields': pf,
        'cell_fields': cf,
        'per_step': per_step,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nReport written to: {out_path}")


if __name__ == '__main__':
    main()
