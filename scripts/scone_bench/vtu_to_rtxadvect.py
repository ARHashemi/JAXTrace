#!/usr/bin/env python3
"""
Convert a VTU / PVTU tet mesh to RTXAdvect's plain-text input format
(two files: verts.dat + cells.dat).

RTXAdvect expects (per the paper's supplementary + repo README):
  verts.dat  — one vertex per line, "x y z" (float)
  cells.dat  — one tet per line, "v0 v1 v2 v3" (int, 0-indexed)

Non-tet cells in the input are tetrahedralised via VTK's
vtkDataSetTriangleFilter (matches what bench_malmo_pointloc.py does).

Also emits <basename>.zero_velocity.dat — a zero-velocity field
(one 3-vector per cell) that RTXAdvect can be pointed at so
particles don't move; this reduces the tool to pure point-location
+ host reporting per timestep.

Usage:
    scripts/scone_bench/vtu_to_rtxadvect.py --vtu MESH.vtu --out-prefix OUT
    → OUT.verts.dat  OUT.cells.dat  OUT.zero_velocity.dat
    → also prints the mesh bbox (for the --seeding-box flag)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_tet_mesh(vtu_path: Path):
    import vtk
    reader = vtk.vtkXMLGenericDataObjectReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    ds = reader.GetOutput()
    if not ds.IsA("vtkUnstructuredGrid"):
        raise ValueError(f"{vtu_path}: not a vtkUnstructuredGrid")

    n_cells = ds.GetNumberOfCells()
    n_tet_native = sum(1 for i in range(n_cells) if ds.GetCell(i).GetCellType() == vtk.VTK_TETRA)
    n_other = n_cells - n_tet_native
    if n_other > 0:
        tri = vtk.vtkDataSetTriangleFilter()
        tri.SetInputData(ds)
        tri.TetrahedraOnlyOn()
        tri.Update()
        ds = tri.GetOutput()
        n_cells = ds.GetNumberOfCells()
        print(f"  tetrahedralised {n_other:,} non-tet cells "
              f"(passed through {n_tet_native:,} native tets); "
              f"post-tet: {n_cells:,} tets", flush=True)

    n_pts = ds.GetNumberOfPoints()
    pts = np.empty((n_pts, 3), dtype=np.float64)
    for i in range(n_pts):
        pts[i] = ds.GetPoint(i)

    conn = []
    for cid in range(n_cells):
        c = ds.GetCell(cid)
        if c.GetCellType() == vtk.VTK_TETRA:
            ids = c.GetPointIds()
            conn.append([int(ids.GetId(j)) for j in range(4)])
    conn = np.asarray(conn, dtype=np.int32)
    return pts, conn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtu", required=True, help="Input VTU or PVTU")
    ap.add_argument("--out-prefix", required=True, help="Output prefix (writes .verts.dat, .cells.dat, .zero_velocity.dat)")
    args = ap.parse_args()

    vtu = Path(args.vtu)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"reading {vtu}", flush=True)
    pts, conn = load_tet_mesh(vtu)
    print(f"  {pts.shape[0]:,} vertices, {conn.shape[0]:,} tets", flush=True)

    # Write verts (one vertex per line: x y z)
    verts_path = prefix.with_suffix(prefix.suffix + ".verts.dat") \
                 if prefix.suffix else Path(f"{prefix}.verts.dat")
    cells_path = prefix.with_suffix(prefix.suffix + ".cells.dat") \
                 if prefix.suffix else Path(f"{prefix}.cells.dat")
    zvel_path  = prefix.with_suffix(prefix.suffix + ".zero_velocity.dat") \
                 if prefix.suffix else Path(f"{prefix}.zero_velocity.dat")

    # RTXAdvect's HostTetMesh::readDataSet format (see
    # 3rdParty/RTXAdvect/cuda/HostTetMesh.h:146):
    #   verts.dat:
    #     NumTetVerts = <N>
    #     x y z
    #     <N lines: fx fy fz>
    #   cells.dat:
    #     NumTetCells = <N>
    #     id1 id2 id3 id4
    #     <N lines: i j k l>
    #   solution.dat (vertex-wise, one line per VERTEX):
    #     p u v w
    #     <NumTetVerts lines: 0.0 vx vy vz>
    # RTXAdvect's reader does `vfile >> word >> NumVerts` — so the count
    # must be the SECOND whitespace-separated token, not the third. The
    # docstring in HostTetMesh.h shows `NumTetVerts = 16226` but that's
    # documentation shorthand; the actual reader treats `=` as an integer
    # and silently produces NumVerts=0.  Correct form: no `=` sign.
    with open(verts_path, "w") as f:
        f.write(f"NumTetVerts {pts.shape[0]}\n")
        f.write("x y z\n")
        for x, y, z in pts:
            f.write(f"{x:.15g} {y:.15g} {z:.15g}\n")
    with open(cells_path, "w") as f:
        f.write(f"NumTetCells {conn.shape[0]}\n")
        f.write("id1 id2 id3 id4\n")
        for a, b, c, d in conn:
            f.write(f"{int(a)} {int(b)} {int(c)} {int(d)}\n")
    # Zero-velocity: vertex-wise (NumTetVerts lines), leading `p` column
    with open(zvel_path, "w") as f:
        f.write("p u v w\n")
        for _ in range(pts.shape[0]):
            f.write("0 0 0 0\n")

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    meta = {
        "vtu": str(vtu.resolve()),
        "n_vertices": int(pts.shape[0]),
        "n_tets": int(conn.shape[0]),
        "bbox_min": lo.tolist(),
        "bbox_max": hi.tolist(),
        "verts_dat": str(verts_path.resolve()),
        "cells_dat": str(cells_path.resolve()),
        "zero_velocity_dat": str(zvel_path.resolve()),
        "seeding_box_flag": f"{lo[0]:.6g} {lo[1]:.6g} {lo[2]:.6g} {hi[0]:.6g} {hi[1]:.6g} {hi[2]:.6g}",
    }
    meta_path = Path(f"{prefix}.meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print()
    print(f"wrote {verts_path}")
    print(f"wrote {cells_path}")
    print(f"wrote {zvel_path}")
    print(f"wrote {meta_path}")
    print()
    print(f"seeding-box flag: --seeding-box {meta['seeding_box_flag']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
