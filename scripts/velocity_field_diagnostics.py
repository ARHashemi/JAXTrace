"""
velocity_field_diagnostics.py -- Eulerian diagnostics for the FOM vs
ROM velocity fields underlying the roadmap §4 investigation.

For a single case, load v_FOM at ts=source-timestep and v_ROM (the
single-frame reconstructed field), then produce:

    1. residual_vtu     v_FOM - v_ROM as a per-node vector field on the
                        FOM mesh (same node coords as v_FOM).  Written
                        as an UnstructuredGrid VTU so it opens in
                        ParaView.

    2. radial_samples   |v| along four radial lines from the origin
                        (0°, 90°, 180°, 270°) for FOM, ROM, and the
                        residual.  Line samples come from an SPR-style
                        nearest-node query (fast, no VTK probing).

    3. ring_samples     |v| along three azimuthal rings at r =
                        r_ring_1, r_ring_2, r_ring_3 for FOM, ROM,
                        residual.  Enables FFT vs θ (azimuthal
                        spectrum).

    4. divergence       Per-element ∇·v for FOM and ROM at every
                        tetrahedron of the mesh, using P1 shape
                        function derivatives.  Reported as a per-
                        element scalar in a companion VTU (attached to
                        the same mesh) plus a summary CSV with
                        statistics binned by cell centroid r.

Output layout (under --out-dir):

    residual.vtu                per-node residual vector field
    div_fom.vtu, div_rom.vtu    per-cell divergence field
    radial_samples.csv          columns: line, s, x, y, |v_fom|,
                                |v_rom|, |v_residual|
    ring_samples.csv            columns: r, theta, x, y, |v_fom|,
                                |v_rom|, |v_residual|
    divergence_summary.csv      per-bin |∇·v| statistics
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import (
    numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy,
)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_pvtu(path: Path):
    """Load a PVTU into a merged UnstructuredGrid, extract points and
    the Displacement (velocity) array."""
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    ug = r.GetOutput()
    pts = vtk_to_numpy(ug.GetPoints().GetData()).astype(np.float64)
    vel = vtk_to_numpy(ug.GetPointData().GetArray("Displacement")).astype(np.float64)
    return ug, pts, vel


def _align_meshes(pts_a: np.ndarray, pts_b: np.ndarray):
    """Return an index array `order_b` such that pts_a == pts_b[order_b].

    Both meshes must have the same node set (as produced by the ROM
    reconstruction — it copies the FOM mesh verbatim).  Uses
    lexicographic sort.
    """
    if pts_a.shape != pts_b.shape:
        raise ValueError(
            f"mesh shapes disagree: {pts_a.shape} vs {pts_b.shape}")
    order_a = np.lexsort((pts_a[:, 2], pts_a[:, 1], pts_a[:, 0]))
    order_b = np.lexsort((pts_b[:, 2], pts_b[:, 1], pts_b[:, 0]))
    # order_a maps sorted -> original_a
    # order_b maps sorted -> original_b
    # We want map: original_a -> original_b
    #    result[i_a] such that pts_a[i_a] == pts_b[result[i_a]]
    # i.e. result = order_b[inv_a] where inv_a = argsort(order_a)
    inv_a = np.argsort(order_a)
    return order_b[inv_a]


def _write_ug_vtu(out_path: Path, ug: vtk.vtkUnstructuredGrid) -> None:
    w = vtk.vtkXMLUnstructuredGridWriter()
    w.SetFileName(str(out_path))
    w.SetInputData(ug)
    w.SetDataModeToBinary()
    w.SetCompressorTypeToZLib()
    w.Write()


def _clone_mesh_only(ug: vtk.vtkUnstructuredGrid) -> vtk.vtkUnstructuredGrid:
    """Return a new UG with the same Points + Cells but no PointData/CellData."""
    out = vtk.vtkUnstructuredGrid()
    out.SetPoints(ug.GetPoints())
    out.SetCells(
        vtk.vtkUnsignedCharArray(),  # placeholder, replaced below
        ug.GetCells())
    # Reuse the input's cell types
    out.ShallowCopy(ug)
    out.GetPointData().Initialize()
    out.GetCellData().Initialize()
    return out


# ---------------------------------------------------------------------------
# Radial + ring line samples via nearest-node queries
# ---------------------------------------------------------------------------

def _probe_ug(ug_source: vtk.vtkUnstructuredGrid,
              queries_xyz: np.ndarray,
              field_name: str = "Displacement") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Probe an UnstructuredGrid at arbitrary XYZ points using proper
    P1 barycentric interpolation via vtkProbeFilter.

    Returns (vel_vec, level, valid_mask):
        vel_vec    (n_queries, 3)   interpolated Displacement
        level      (n_queries,)     interpolated LEVEL (positive
                                    outside pin body, negative inside)
        valid_mask (n_queries,) bool  True if the query lies inside the mesh
    """
    probe_pts = vtk.vtkPoints()
    for x, y, z in queries_xyz:
        probe_pts.InsertNextPoint(float(x), float(y), float(z))
    probe_poly = vtk.vtkPolyData()
    probe_poly.SetPoints(probe_pts)

    probe = vtk.vtkProbeFilter()
    probe.SetInputData(probe_poly)
    probe.SetSourceData(ug_source)
    probe.Update()
    out = probe.GetOutput()

    vel = vtk_to_numpy(out.GetPointData().GetArray(field_name)).astype(np.float64)
    valid = vtk_to_numpy(out.GetPointData().GetArray("vtkValidPointMask")).astype(bool)

    lvl_arr = out.GetPointData().GetArray("LEVEL")
    if lvl_arr is not None:
        level = vtk_to_numpy(lvl_arr).astype(np.float64)
    else:
        level = np.full(queries_xyz.shape[0], np.nan)

    return vel, level, valid


def sample_radial_lines(ug_fom: vtk.vtkUnstructuredGrid,
                        ug_rom: vtk.vtkUnstructuredGrid,
                        r_max: float, n_samples: int, z_target: float,
                        z_tol: float) -> list[dict]:
    """Sample |v| along four radial lines from origin outward at
    θ = 0°, 90°, 180°, 270°.  Uses P1 barycentric interpolation via
    vtkProbeFilter — the sample values are on the same continuous
    field the RK4 tracker would evaluate along the line.  z_tol is
    unused in the probe-based sampler and kept only for CLI back-
    compatibility (the probe uses (x, y, z_target) directly).
    """
    del z_tol  # kept only for signature compatibility with older CLI
    r_grid = np.linspace(r_max / n_samples, r_max, n_samples)
    rows: list[dict] = []
    lines = {
        "east":  np.stack([ r_grid, np.zeros_like(r_grid)], axis=1),
        "north": np.stack([np.zeros_like(r_grid),  r_grid], axis=1),
        "west":  np.stack([-r_grid, np.zeros_like(r_grid)], axis=1),
        "south": np.stack([np.zeros_like(r_grid), -r_grid], axis=1),
    }
    for label, queries in lines.items():
        queries_3d = np.column_stack([queries, np.full(queries.shape[0], z_target)])
        v_fom, level_fom, valid_fom = _probe_ug(ug_fom, queries_3d)
        v_rom, _,        valid_rom = _probe_ug(ug_rom, queries_3d)
        residual = v_fom - v_rom
        for i in range(queries.shape[0]):
            rows.append({
                "line":          label,
                "s":             float(r_grid[i]),
                "x":             float(queries[i, 0]),
                "y":             float(queries[i, 1]),
                "level":         float(level_fom[i]),
                "inside_pin":    bool(level_fom[i] < 0.0)
                                 if np.isfinite(level_fom[i]) else False,
                "valid":         bool(valid_fom[i] and valid_rom[i]),
                "v_fom_mag":     float(np.linalg.norm(v_fom[i])),
                "v_rom_mag":     float(np.linalg.norm(v_rom[i])),
                "residual_mag":  float(np.linalg.norm(residual[i])),
            })
    return rows


def sample_ring_lines(ug_fom: vtk.vtkUnstructuredGrid,
                      ug_rom: vtk.vtkUnstructuredGrid,
                      radii: list[float], n_samples: int,
                      z_target: float, z_tol: float) -> list[dict]:
    """Sample |v| along azimuthal rings at fixed radii, θ ∈ [0, 2π).
    Uses P1 barycentric interpolation via vtkProbeFilter — the ring
    samples are the same continuous field the RK4 tracker would
    evaluate.  Also carries an inside_pin flag derived from the LEVEL
    scalar of the FOM PVTU so downstream figures can filter or
    annotate accordingly.  z_tol is unused (kept for CLI back-compat).
    """
    del z_tol
    rows: list[dict] = []
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    for r in radii:
        queries_xy = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        queries_3d = np.column_stack([queries_xy, np.full(n_samples, z_target)])
        v_fom, level_fom, valid_fom = _probe_ug(ug_fom, queries_3d)
        v_rom, _,        valid_rom = _probe_ug(ug_rom, queries_3d)
        residual = v_fom - v_rom
        for i in range(n_samples):
            rows.append({
                "r":             float(r),
                "theta":         float(theta[i]),
                "x":             float(queries_xy[i, 0]),
                "y":             float(queries_xy[i, 1]),
                "level":         float(level_fom[i]),
                "inside_pin":    bool(level_fom[i] < 0.0)
                                 if np.isfinite(level_fom[i]) else False,
                "valid":         bool(valid_fom[i] and valid_rom[i]),
                "v_fom_mag":     float(np.linalg.norm(v_fom[i])),
                "v_rom_mag":     float(np.linalg.norm(v_rom[i])),
                "residual_mag":  float(np.linalg.norm(residual[i])),
            })
    return rows


# ---------------------------------------------------------------------------
# Divergence — P1 tet, one scalar per element
# ---------------------------------------------------------------------------

def _get_tet_connectivity(ug: vtk.vtkUnstructuredGrid) -> np.ndarray:
    """Return (n_cells, 4) int64 connectivity assuming all cells are
    tetrahedra (VTK_TETRA = 10).  Raises if any cell is not a tet.
    """
    n_cells = ug.GetNumberOfCells()
    conn = np.empty((n_cells, 4), dtype=np.int64)
    cell = vtk.vtkGenericCell()
    for i in range(n_cells):
        ug.GetCell(i, cell)
        if cell.GetCellType() != vtk.VTK_TETRA:
            raise ValueError(
                f"cell {i} is not a tetrahedron (type "
                f"{cell.GetCellType()}); this diagnostic only supports "
                f"pure-tet meshes.")
        ids = cell.GetPointIds()
        for j in range(4):
            conn[i, j] = ids.GetId(j)
    return conn


def compute_p1_divergence(pts: np.ndarray, vel: np.ndarray,
                          conn: np.ndarray) -> np.ndarray:
    """P1 per-element divergence.

    On a tet with vertices p0..p3 and per-vertex velocity v0..v3, the
    linear interpolant has ∇v constant on the element:

        v(x) = sum_i N_i(x) v_i

    where N_i are the barycentric shape functions.  In matrix form
    with M = [p1-p0; p2-p0; p3-p0]ᵀ, the gradient matrix is

        [∇N_1; ∇N_2; ∇N_3] = M^{-1}   (each row is one ∇N_i)
        ∇N_0 = -(∇N_1 + ∇N_2 + ∇N_3)

        ∇v = sum_i v_i ⊗ ∇N_i   (3×3 matrix)
        div v = trace(∇v) = sum_i v_i · ∇N_i

    Vectorised over all cells via batched np.linalg.solve.
    """
    p0 = pts[conn[:, 0]]
    p1 = pts[conn[:, 1]]
    p2 = pts[conn[:, 2]]
    p3 = pts[conn[:, 3]]

    # M is (n_cells, 3, 3), rows = edge vectors from vertex 0
    M = np.stack([p1 - p0, p2 - p0, p3 - p0], axis=1)   # (n, 3, 3)

    v0 = vel[conn[:, 0]]
    v1 = vel[conn[:, 1]]
    v2 = vel[conn[:, 2]]
    v3 = vel[conn[:, 3]]

    # Solve M^T · [∇N_1, ∇N_2, ∇N_3]^T = I  →  ∇N_i are the *rows* of M^{-1}
    # Standard identity:  ∇N_i are the rows of M^{-1} where M has rows
    # (p_i - p_0) for i=1,2,3.  So gN_123 = np.linalg.inv(M) — but we
    # only need div, which is trace of ∇v, so:
    #   div = sum_{i=1..3} v_i · row_i(M^{-1}) + v_0 · (-sum_{i=1..3} row_i(M^{-1}))
    #       = sum_{i=1..3} (v_i - v_0) · row_i(M^{-1})
    dv = np.stack([v1 - v0, v2 - v0, v3 - v0], axis=1)   # (n, 3, 3)

    # We need row_i(M^{-1}) · (v_i - v_0), which is equivalent to
    # solving M^T · x = (v_i - v_0) and taking element i.  Cleaner:
    # solve M · y = dv[:, i] for each i and take y[i]?  Actually:
    #   let G = M^{-1}, so row_i(G) = G[i, :]
    #   div = sum_i G[i, :] @ (v_i - v_0)
    #       = trace( G @ dv )      where dv rows are (v_i - v_0)
    #       = trace( dv @ G_columns via M @ x = e_j etc. )
    # Simplest: solve M @ X = dv → X = M^{-1} @ dv, then div = trace(X)
    # per cell.
    X = np.linalg.solve(M, dv)      # (n, 3, 3) — X = M^{-1} dv
    div = X[:, 0, 0] + X[:, 1, 1] + X[:, 2, 2]
    return div


def cell_centroids(pts: np.ndarray, conn: np.ndarray) -> np.ndarray:
    return 0.25 * (pts[conn[:, 0]] + pts[conn[:, 1]]
                   + pts[conn[:, 2]] + pts[conn[:, 3]])


# ---------------------------------------------------------------------------
# Attach a per-node vector field or per-cell scalar to a mesh clone
# ---------------------------------------------------------------------------

def attach_point_vector(ug: vtk.vtkUnstructuredGrid, arr: np.ndarray,
                        name: str) -> None:
    v = numpy_to_vtk(np.ascontiguousarray(arr.astype(np.float32)), deep=True)
    v.SetName(name)
    v.SetNumberOfComponents(3)
    ug.GetPointData().AddArray(v)


def attach_cell_scalar(ug: vtk.vtkUnstructuredGrid, arr: np.ndarray,
                       name: str) -> None:
    v = numpy_to_vtk(np.ascontiguousarray(arr.astype(np.float32)), deep=True)
    v.SetName(name)
    ug.GetCellData().AddArray(v)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fom-pvtu", type=Path, required=True,
                    help="FOM PVTU at the reference timestep, e.g. "
                         "cylindrical_003.gid/post/cylindrical_119.pvtu")
    ap.add_argument("--rom-pvtu", type=Path, required=True,
                    help="ROM reconstructed PVTU, e.g. "
                         "ROM_recon_centered/cylindrical_003.gid/"
                         "post/cylindrical_0.pvtu")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--case-label", type=str, default=None,
                    help="Label to embed in filenames; defaults to the "
                         "FOM PVTU's grandparent .gid name.")
    ap.add_argument("--radial-samples", type=int, default=200,
                    help="Number of sample points per radial line.")
    ap.add_argument("--radial-r-max", type=float, default=0.020,
                    help="Outer radius of the radial lines (m).")
    ap.add_argument("--ring-samples", type=int, default=360,
                    help="Number of azimuthal samples per ring.")
    ap.add_argument("--ring-radii", type=float, nargs="+",
                    default=[0.006, 0.009, 0.012, 0.015],
                    help="Radii of the azimuthal ring samples (m). "
                         "Default: 6 mm (deep inside pin body), 9 mm "
                         "(near pin surface), 12 mm (just outside pin), "
                         "15 mm (bulk workpiece).  The pin surface "
                         "sits at r ≈ 10 mm on this mesh — see the "
                         "LEVEL scalar per sample for the exact "
                         "inside/outside classification.")
    ap.add_argument("--z-target", type=float, default=0.0,
                    help="Sample plane (z-coord) for line/ring samples.")
    ap.add_argument("--z-tol", type=float, default=0.001,
                    help="Half-thickness of the z-slab used to find "
                         "in-plane nodes for line/ring samples.")
    args = ap.parse_args()

    if args.case_label is None:
        args.case_label = args.fom_pvtu.parent.parent.name.replace(".gid", "")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[diagnostics] case={args.case_label}")
    print(f"[diagnostics] out_dir={args.out_dir}")

    ug_fom, pts_fom, v_fom = _load_pvtu(args.fom_pvtu)
    ug_rom, pts_rom, v_rom = _load_pvtu(args.rom_pvtu)
    print(f"  FOM: {pts_fom.shape[0]:,} nodes, {ug_fom.GetNumberOfCells():,} cells")
    print(f"  ROM: {pts_rom.shape[0]:,} nodes, {ug_rom.GetNumberOfCells():,} cells")

    # Align ROM to FOM node order (should be identical or a permutation).
    order = _align_meshes(pts_fom, pts_rom)
    max_pos_diff = float(np.max(np.abs(pts_fom - pts_rom[order])))
    print(f"  max node-position diff after alignment: {max_pos_diff:.3e}")
    v_rom_aligned = v_rom[order]

    # ------ 1. residual VTU ---------------------------------------------
    residual = v_fom - v_rom_aligned
    residual_mag = np.linalg.norm(residual, axis=1)
    print(f"  residual: mean |v_FOM - v_ROM| = {residual_mag.mean():.4e}, "
          f"max = {residual_mag.max():.4e}")

    # Clone the FOM UG then strip its own arrays; we'll add v_FOM, v_ROM
    # and residual as fresh point-data arrays.
    ug_res = vtk.vtkUnstructuredGrid()
    ug_res.DeepCopy(ug_fom)
    ug_res.GetPointData().Initialize()
    ug_res.GetCellData().Initialize()
    attach_point_vector(ug_res, v_fom,          "v_fom")
    attach_point_vector(ug_res, v_rom_aligned,  "v_rom")
    attach_point_vector(ug_res, residual,       "residual")
    # scalar magnitudes for easy ParaView colouring
    for name, mag in (("v_fom_mag",    np.linalg.norm(v_fom, axis=1)),
                      ("v_rom_mag",    np.linalg.norm(v_rom_aligned, axis=1)),
                      ("residual_mag", residual_mag)):
        va = numpy_to_vtk(mag.astype(np.float32), deep=True)
        va.SetName(name)
        ug_res.GetPointData().AddArray(va)
    _write_ug_vtu(args.out_dir / f"residual_{args.case_label}.vtu", ug_res)
    print(f"  -> residual_{args.case_label}.vtu")

    # ------ 2. radial line samples --------------------------------------
    # Interpolated via vtkProbeFilter — the samples are the same
    # continuous barycentric field the RK4 tracker would evaluate along
    # the line.  This replaces the earlier nearest-node sampler which
    # aliased badly at radii where the mesh coverage was coarse (e.g.
    # r = 15 mm had only 52 unique nodes across 360 azimuthal queries
    # in the earlier version).
    radial_rows = sample_radial_lines(
        ug_fom, ug_rom,
        r_max=args.radial_r_max, n_samples=args.radial_samples,
        z_target=args.z_target, z_tol=args.z_tol,
    )
    _write_csv(args.out_dir / f"radial_samples_{args.case_label}.csv",
               radial_rows,
               ["line", "s", "x", "y", "level", "inside_pin", "valid",
                "v_fom_mag", "v_rom_mag", "residual_mag"])
    print(f"  -> radial_samples_{args.case_label}.csv  "
          f"({len(radial_rows)} rows across 4 lines)")

    # ------ 3. ring samples ---------------------------------------------
    ring_rows = sample_ring_lines(
        ug_fom, ug_rom,
        radii=args.ring_radii, n_samples=args.ring_samples,
        z_target=args.z_target, z_tol=args.z_tol,
    )
    _write_csv(args.out_dir / f"ring_samples_{args.case_label}.csv",
               ring_rows,
               ["r", "theta", "x", "y", "level", "inside_pin", "valid",
                "v_fom_mag", "v_rom_mag", "residual_mag"])
    print(f"  -> ring_samples_{args.case_label}.csv  "
          f"({len(ring_rows)} rows across {len(args.ring_radii)} rings)")

    # ------ 4. divergence -----------------------------------------------
    print("  computing per-tet divergence (P1)...")
    conn = _get_tet_connectivity(ug_fom)
    div_fom = compute_p1_divergence(pts_fom, v_fom,         conn)
    div_rom = compute_p1_divergence(pts_fom, v_rom_aligned, conn)
    centroids = cell_centroids(pts_fom, conn)

    # Attach as cell-data arrays on a mesh clone.
    ug_div = vtk.vtkUnstructuredGrid()
    ug_div.DeepCopy(ug_fom)
    ug_div.GetPointData().Initialize()
    ug_div.GetCellData().Initialize()
    attach_cell_scalar(ug_div, div_fom,           "div_fom")
    attach_cell_scalar(ug_div, div_rom,           "div_rom")
    attach_cell_scalar(ug_div, np.abs(div_fom),   "abs_div_fom")
    attach_cell_scalar(ug_div, np.abs(div_rom),   "abs_div_rom")
    _write_ug_vtu(args.out_dir / f"divergence_{args.case_label}.vtu", ug_div)
    print(f"  -> divergence_{args.case_label}.vtu")

    # Summary CSV binned by cell centroid r.
    r_cell = np.sqrt(centroids[:, 0] ** 2 + centroids[:, 1] ** 2)
    r_bins = [(0.0, 0.010), (0.010, 0.020), (0.020, np.inf)]
    div_rows = []
    for lo, hi in r_bins:
        mask = (r_cell >= lo) & (r_cell < hi)
        n = int(mask.sum())
        for label, field in (("fom", div_fom), ("rom", div_rom)):
            if n:
                d = field[mask]
                abs_d = np.abs(d)
                div_rows.append({
                    "field":  label,
                    "r_lo":   lo,
                    "r_hi":   hi,
                    "n_cells": n,
                    "abs_div_mean":   float(abs_d.mean()),
                    "abs_div_median": float(np.median(abs_d)),
                    "abs_div_rms":    float(np.sqrt((d ** 2).mean())),
                    "abs_div_p95":    float(np.percentile(abs_d, 95)),
                    "abs_div_p99":    float(np.percentile(abs_d, 99)),
                    "abs_div_max":    float(abs_d.max()),
                })
            else:
                div_rows.append({
                    "field": label, "r_lo": lo, "r_hi": hi,
                    "n_cells": 0,
                    "abs_div_mean": float("nan"),
                    "abs_div_median": float("nan"),
                    "abs_div_rms": float("nan"),
                    "abs_div_p95": float("nan"),
                    "abs_div_p99": float("nan"),
                    "abs_div_max": float("nan"),
                })
    _write_csv(args.out_dir / f"divergence_summary_{args.case_label}.csv",
               div_rows,
               ["field", "r_lo", "r_hi", "n_cells",
                "abs_div_mean", "abs_div_median", "abs_div_rms",
                "abs_div_p95", "abs_div_p99", "abs_div_max"])
    print(f"  -> divergence_summary_{args.case_label}.csv  "
          f"({len(div_rows)} rows)")

    print("[diagnostics] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
