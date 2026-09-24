#!/usr/bin/env python3
"""
Diagnostic for R2-5 (occupancy 6 vs 24).

Runs `extract_octree_cells_parent_cube` with the SAME arguments as
`bench_malmo_pointloc.py` and `run_tracking.py` (centroid path,
`hybrid_non_kuhn=True`) on a mesh, then reports:

  1. Full per-cell occupancy histogram (how many cells have 1, 2, ..., N
     elements).
  2. Split of registrations by source: Kuhn-centroid (each Kuhn tet in
     exactly 1 cell) vs Non-Kuhn AABB-overlap (each non-Kuhn tet in
     1..8 cells of its borrowed grid).  For each cell, we count how
     many of its entries are Kuhn vs non-Kuhn.
  3. The number of cells that exceed 6 elements, and the mean+max
     non-Kuhn spillover count at those cells.
  4. Whether the compile-time upper bound (Table 4's `24`) is
     actually tight or whether we could safely lower it.

The output is a plain-text summary plus a JSON dump for the checklist.

Usage:
    python3 diagnose_centroid_occupancy.py --vtu <MESH.vtu> [--out result.json]

The script does no GPU work and no queries; it only builds the octree
and inspects the per-cell CSR.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def load_mesh(vtu_path: Path):
    import vtk
    reader = vtk.vtkXMLGenericDataObjectReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    ds = reader.GetOutput()
    if not ds.IsA("vtkUnstructuredGrid"):
        raise ValueError(f"{vtu_path}: not a vtkUnstructuredGrid")
    n_cells = ds.GetNumberOfCells()
    n_tet_native = sum(1 for i in range(n_cells)
                       if ds.GetCell(i).GetCellType() == vtk.VTK_TETRA)
    if n_tet_native != n_cells:
        tri = vtk.vtkDataSetTriangleFilter()
        tri.SetInputData(ds); tri.TetrahedraOnlyOn(); tri.Update()
        ds = tri.GetOutput()
        n_cells = ds.GetNumberOfCells()
    n_pts = ds.GetNumberOfPoints()
    pts = np.empty((n_pts, 3), dtype=np.float64)
    for i in range(n_pts):
        pts[i] = ds.GetPoint(i)
    conn = np.empty((n_cells, 4), dtype=np.int32)
    for cid in range(n_cells):
        c = ds.GetCell(cid)
        ids = c.GetPointIds()
        conn[cid] = [int(ids.GetId(j)) for j in range(4)]
    return pts, conn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtu", required=True, help="Input VTU / PVTU")
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    ap.add_argument("--tolerance", type=float, default=1e-6)
    a = ap.parse_args()

    # Repo root = two levels up from this script (scripts/scone_bench/ -> repo)
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import (
        extract_octree_cells_parent_cube,
    )

    print(f"loading {a.vtu}")
    pts, conn = load_mesh(Path(a.vtu))
    print(f"  {pts.shape[0]:,} vertices, {conn.shape[0]:,} tets")

    print("\nrunning extract_octree_cells_parent_cube(hybrid_non_kuhn=True) ...")
    cells = extract_octree_cells_parent_cube(
        pts, conn, tolerance=a.tolerance, verbose=True,
        orphan_fallback=True, hybrid_non_kuhn=True,
    )
    print(f"\n  n_cells={cells.n_cells:,}  "
          f"max_elements_per_cell={cells.max_elements_per_cell}  "
          f"n_non_kuhn={cells.n_non_kuhn:,}  "
          f"n_non_kuhn_registered={cells.n_non_kuhn_registered:,}")

    # ---- 1. Per-cell occupancy histogram ----
    off = np.asarray(cells.cell_to_elements_offsets, dtype=np.int64)
    per_cell = off[1:] - off[:-1]     # (n_cells,) # of elements in each cell
    hist = Counter(int(x) for x in per_cell)
    hist_sorted = sorted(hist.items())

    print("\n  Per-cell occupancy histogram (occupancy: count of cells)")
    for k, v in hist_sorted:
        print(f"    {k:>3}: {v:>10,}")

    # ---- 2. Kuhn vs non-Kuhn split per cell ----
    # For each cell, count how many of its element IDs were "Kuhn" vs "non-Kuhn".
    # The extractor's stored ordering doesn't distinguish, so we rebuild the
    # non-Kuhn ID set from the OctreeCellData return fields.
    total_elements = conn.shape[0]
    n_non_kuhn = cells.n_non_kuhn
    # The extractor identifies non-Kuhn by find_axis_aligned_edges_single;
    # replicate the diagnostic here to get the ID set:
    from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import (
        find_axis_aligned_edges_single,
    )
    non_kuhn_ids = set()
    for eid in range(total_elements):
        cell_size, _level = find_axis_aligned_edges_single(
            pts[conn[eid]], a.tolerance
        )
        if np.any(cell_size == 0):
            non_kuhn_ids.add(eid)
    print(f"\n  Verified non-Kuhn set size: {len(non_kuhn_ids):,} "
          f"(extractor reported {cells.n_non_kuhn:,}: "
          f"{'OK' if len(non_kuhn_ids) == cells.n_non_kuhn else 'MISMATCH'})")

    csr = np.asarray(cells.cell_to_elements_data, dtype=np.int64)
    kuhn_count = np.zeros(cells.n_cells, dtype=np.int64)
    nonkuhn_count = np.zeros(cells.n_cells, dtype=np.int64)
    for cid in range(cells.n_cells):
        s, e = off[cid], off[cid + 1]
        for eid in csr[s:e]:
            if int(eid) in non_kuhn_ids:
                nonkuhn_count[cid] += 1
            else:
                kuhn_count[cid] += 1

    # ---- 3. Cells exceeding 6 elements ----
    over6 = per_cell > 6
    n_over6 = int(over6.sum())
    print(f"\n  Cells with > 6 elements: {n_over6:,} / {cells.n_cells:,} "
          f"({n_over6/cells.n_cells*100:.4f}%)")
    if n_over6:
        idx = np.where(over6)[0]
        k_at = kuhn_count[idx]
        nk_at = nonkuhn_count[idx]
        print(f"    at over-6 cells:")
        print(f"      Kuhn        mean={k_at.mean():.2f}  max={k_at.max()}")
        print(f"      Non-Kuhn    mean={nk_at.mean():.2f}  max={nk_at.max()}")
        # How many over-6 cells have ANY non-Kuhn contribution?
        n_over6_with_nk = int((nk_at > 0).sum())
        print(f"      with any Non-Kuhn spillover: {n_over6_with_nk:,} "
              f"({n_over6_with_nk/n_over6*100:.2f}%)")
        # And how many are pure Kuhn (>6 from Kuhn centroids alone)?
        n_over6_pure_kuhn = int((nk_at == 0).sum())
        print(f"      pure Kuhn (no Non-Kuhn): {n_over6_pure_kuhn:,} "
              f"({n_over6_pure_kuhn/n_over6*100:.2f}%)")
        if n_over6_pure_kuhn > 0:
            print(f"    *** {n_over6_pure_kuhn:,} cells exceed 6 elements from "
                  f"Kuhn centroids ALONE (not borrowing). ***")
            print(f"    This is the case Reviewer 2 was worried about: not")
            print(f"    every over-6 cell is due to non-Kuhn borrowing.")

    # ---- 4. Max occupancy attribution ----
    max_c = int(per_cell.max())
    argmax_c = int(per_cell.argmax())
    print(f"\n  Max occupancy cell:")
    print(f"    cell #{argmax_c}: {max_c} total = {kuhn_count[argmax_c]} Kuhn "
          f"+ {nonkuhn_count[argmax_c]} Non-Kuhn")

    # ---- Dump ----
    if a.out:
        out = {
            "vtu": str(Path(a.vtu).resolve()),
            "n_vertices": int(pts.shape[0]),
            "n_tets": int(conn.shape[0]),
            "n_cells": int(cells.n_cells),
            "max_elements_per_cell": int(cells.max_elements_per_cell),
            "n_non_kuhn": int(cells.n_non_kuhn),
            "n_non_kuhn_registered": int(cells.n_non_kuhn_registered),
            "occupancy_histogram": {str(k): int(v) for k, v in hist.items()},
            "n_cells_over_6": n_over6,
            "n_over6_with_non_kuhn": int((nonkuhn_count[over6] > 0).sum())
                if n_over6 else 0,
            "n_over6_pure_kuhn": int((nonkuhn_count[over6] == 0).sum())
                if n_over6 else 0,
            "max_occupancy_cell": {
                "index": argmax_c,
                "total": max_c,
                "kuhn": int(kuhn_count[argmax_c]),
                "non_kuhn": int(nonkuhn_count[argmax_c]),
            },
        }
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
