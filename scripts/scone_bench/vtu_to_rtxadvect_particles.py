#!/usr/bin/env python3
"""
Generate an in-mesh particle file for RTXAdvect, matching the sampling
scheme used by bench_malmo_pointloc.py's in-mesh mode:

  1. Draw N tet-IDs uniformly with replacement from the mesh
  2. For each tet, draw random barycentric weights (u+v+w+t = 1, uniform
     in the tetrahedron via the standard 3-sort trick)
  3. Write the point + the source tet ID (ground truth) in RTXAdvect's
     particle-file format:
         NumParticles <N>
         x y z tetID
         <x> <y> <z> <tetID>
         ...

Also emits a companion .gt.json that keeps the ground-truth tet IDs
alongside — RTXAdvect doesn't report per-particle host-tet IDs so
we can't directly compute correct_rate_strict, but we can compare
`found_rate` against the mesh interior (should be 100%).

Usage:
    vtu_to_rtxadvect_particles.py --vtu MESH.vtu -n 1000000 \\
                                  --out OUT.particles.dat [--seed 42]
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
    n_tet_native = sum(1 for i in range(n_cells)
                       if ds.GetCell(i).GetCellType() == vtk.VTK_TETRA)
    n_other = n_cells - n_tet_native
    if n_other > 0:
        tri = vtk.vtkDataSetTriangleFilter()
        tri.SetInputData(ds)
        tri.TetrahedraOnlyOn()
        tri.Update()
        ds = tri.GetOutput()
        n_cells = ds.GetNumberOfCells()
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
    return pts, np.asarray(conn, dtype=np.int32)


def uniform_bary_in_tet(rng, n):
    """Uniform samples in the standard 3-simplex → barycentric weights.

    Method: draw s,t in [0,1]; if s+t > 1, replace with 1-s, 1-t; then
    u ∈ [0,1]; the barycentric coords are (1-s-t)*(1-u), s*(1-u), t*(1-u), u.
    Equivalent to the standard 3-sort trick for a 3-simplex.
    """
    s = rng.random(n)
    t = rng.random(n)
    u = rng.random(n)
    swap = (s + t) > 1.0
    s = np.where(swap, 1.0 - s, s)
    t = np.where(swap, 1.0 - t, t)
    w0 = (1.0 - s - t) * (1.0 - u)
    w1 = s * (1.0 - u)
    w2 = t * (1.0 - u)
    w3 = u
    return np.stack([w0, w1, w2, w3], axis=1)  # (n,4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtu", required=True)
    ap.add_argument("-n", "--num-particles", type=int, required=True)
    ap.add_argument("--out", required=True, help="OUT.particles.dat")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    vtu = Path(a.vtu)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)

    print(f"reading {vtu}", flush=True)
    pts, conn = load_tet_mesh(vtu)
    n_tets = conn.shape[0]
    print(f"  {pts.shape[0]:,} vertices, {n_tets:,} tets", flush=True)

    rng = np.random.default_rng(a.seed)
    tet_ids = rng.integers(0, n_tets, size=a.num_particles, dtype=np.int64)
    bary = uniform_bary_in_tet(rng, a.num_particles)

    # Compute Cartesian: (n,4,3) tet_vertices * (n,4,1) weights
    tet_verts = pts[conn[tet_ids]]                # (n,4,3)
    xyz = (tet_verts * bary[:, :, None]).sum(axis=1)  # (n,3)

    print(f"writing {out}", flush=True)
    with open(out, "w") as f:
        f.write(f"NumParticles {a.num_particles}\n")
        f.write("x y z tetID\n")
        for (x, y, z), tid in zip(xyz, tet_ids):
            f.write(f"{x:.15g} {y:.15g} {z:.15g} {int(tid)}\n")

    gt_json = Path(str(out) + ".gt.json")
    gt_json.write_text(json.dumps({
        "vtu": str(vtu.resolve()),
        "n_vertices": int(pts.shape[0]),
        "n_tets": int(n_tets),
        "n_particles": int(a.num_particles),
        "seed": int(a.seed),
        "particles_dat": str(out.resolve()),
    }, indent=2))
    print(f"wrote {gt_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
