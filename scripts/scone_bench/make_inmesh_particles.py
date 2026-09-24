#!/usr/bin/env python3
"""
Write an RTXAdvect particle-seed file with in-mesh barycentric sampling.

RTXAdvect's default seeding (--seeding-box) samples uniformly in the mesh
AABB.  On non-convex geometries (Stanford Bunny, microfluidic channel,
porous media) a large fraction of those samples lie outside the mesh
volume, so the reported found-rate collapses onto the mesh's bbox fill
ratio and is not comparable with MALMO's in-mesh strict-correctness
measurement.

This script produces a particle file using the SAME sampler MALMO uses
(pick a tet with probability proportional to its volume, then a uniform
barycentric point inside it), so both frameworks can be scored on an
identical query distribution.

File format expected by RTXAdvect's cudaInitParticles(..., fileName)
(see 3rdParty/RTXAdvect/cuda/particles.cu):

    NumParticles <N>
    x y z tetID
    <N lines: px py pz tid>

The reader consumes the count as the second whitespace-separated token of
line 1, then four tokens of a comment line, then four tokens per particle.

Usage:
    make_inmesh_particles.py --vtu MESH.vtu --out OUT.particles.dat \
        --n-particles 100000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_tet_mesh(vtu_path: Path):
    """Load a tet mesh, decomposing non-tet cells (matches the MALMO harness)."""
    import vtk

    reader = vtk.vtkXMLGenericDataObjectReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    ds = reader.GetOutput()
    if ds is None or not ds.IsA("vtkUnstructuredGrid"):
        raise ValueError(f"{vtu_path}: expected vtkUnstructuredGrid")

    n_cells = ds.GetNumberOfCells()
    n_tet_native = sum(
        1 for i in range(n_cells)
        if ds.GetCell(i).GetCellType() == vtk.VTK_TETRA
    )
    if n_cells - n_tet_native > 0:
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


def sample_in_mesh(node_positions, connectivity, n, seed):
    """
    Volume-weighted tet choice + uniform barycentric point inside it.

    Mirrors _in_mesh_query_points() in bench_malmo_pointloc.py so the two
    frameworks see the same query distribution for a given seed.
    """
    rng = np.random.default_rng(seed)
    v = node_positions[connectivity]                      # (E,4,3)
    vols = np.abs(np.einsum(
        'ij,ij->i',
        np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]),
        v[:, 3] - v[:, 0],
    )) / 6.0
    probs = vols / vols.sum()

    tet_ids = rng.choice(connectivity.shape[0], size=n, p=probs).astype(np.int32)

    # Uniform on the 3-simplex via sorted uniforms
    r = rng.random((n, 3))
    o = np.sort(r, axis=1)
    a = o[:, 0]
    b = o[:, 1] - o[:, 0]
    c = o[:, 2] - o[:, 1]
    d = 1.0 - o[:, 2]

    verts = node_positions[connectivity[tet_ids]]         # (n,4,3)
    pos = (a[:, None] * verts[:, 0]
           + b[:, None] * verts[:, 1]
           + c[:, None] * verts[:, 2]
           + d[:, None] * verts[:, 3])
    return pos, tet_ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtu", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-particles", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    print(f"reading {a.vtu}", flush=True)
    pts, conn = load_tet_mesh(a.vtu)
    print(f"  {pts.shape[0]:,} vertices, {conn.shape[0]:,} tets", flush=True)

    pos, tet_ids = sample_in_mesh(pts, conn, a.n_particles, a.seed)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        # Header: reader does `vfile >> word >> N`
        f.write(f"NumParticles {a.n_particles}\n")
        # Comment line: reader consumes exactly four tokens
        f.write("x y z tetID\n")
        for (px, py, pz), tid in zip(pos, tet_ids):
            f.write(f"{px:.15g} {py:.15g} {pz:.15g} {int(tid)}\n")

    # Ground-truth sidecar so the parser can score strict correctness
    gt = a.out.with_suffix(a.out.suffix + ".groundtruth.npy")
    np.save(gt, tet_ids)

    meta = {
        "vtu": str(a.vtu.resolve()),
        "particles_dat": str(a.out.resolve()),
        "groundtruth_npy": str(gt.resolve()),
        "n_particles": int(a.n_particles),
        "seed": int(a.seed),
        "sampling": "in_mesh_barycentric_volume_weighted",
        "n_tets": int(conn.shape[0]),
        "n_vertices": int(pts.shape[0]),
    }
    meta_path = a.out.with_suffix(a.out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"wrote {a.out}  ({a.n_particles:,} particles, seed={a.seed})")
    print(f"wrote {gt}")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
