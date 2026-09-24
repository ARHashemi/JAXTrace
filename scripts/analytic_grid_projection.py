"""
analytic_grid_projection.py --- mesh -> uniform-grid projection error
for the analytic-velocity validation cases (roadmap §5 preparatory).

For a given case (recirc_2026, rot_cyl_2026, ...):

  1. Load the CASE's field module (defines velocity_fn(pos)) and its
     4-level refined FEM mesh (from generate_meshes.sh output).
  2. Sample the analytic velocity at each mesh node -- this is what the
     tracker actually sees when it runs on that mesh.
  3. Build three TARGET uniform Cartesian grids of increasing complexity:
       * uniform_base           -- one uniform-cell block over the full bbox
       * blockref_2lvl          -- base + one 2x-refined block around the feature
       * blockref_4lvl          -- base + 2x block + 4x block (nested)
  4. For each (target grid) x (recovery method) pair, PROJECT the mesh
     nodal velocity onto every grid cell centre using either:
       * P1 raw barycentric   (via vtkProbeFilter, exact P1 interp)
       * vertex_taylor        (SPR-recovered nodal gradient, per-vertex
                               Taylor expansion blended by P1 shape fns)
  5. Compare each projected v_proj against the analytic ground truth
     v_analytic evaluated at the same grid cell centres.
     Report L2 rms, mean, max of |v_proj - v_analytic| per grid.
  6. Write a per-cell error VTU (unstructured but with hex cells) per
     (grid, method) combination and one summary CSV per case.

Outputs (under --out-dir):
    projection_summary_<case>.csv
    projection_error_<case>_<grid>_<method>.vtu

Runtime: ~2-5 min per case on the workstation-mounted paths.

Design notes
------------
* No new grid-refinement code path in the tracker.  This tool is a
  standalone Eulerian error-estimator; the grid representation lives
  only inside this script.  A future companion tool will consume the
  same 'blocks' definition to build a grid-based velocity provider for
  §5 particle tracking, but that's separate.
* HCT-cubic IS implemented via jaxtrace.gpu.recovery.hct3d.  The
  mesh-side prep uses build_alfeld_geometry + SPR nodal gradients +
  build_hct_bernstein_c1_taylor.  The grid-node query path uses
  vtkCellLocator for the parent tet, sub-tet detection via signed
  volumes at the parent centroid, and bernstein_cubic_evaluate for
  the final velocity.  Slower than P1 (locator + 20-term poly eval
  per query) but this is an offline projection tool, not the tracker.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import (
    numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy,
)

# repo-relative import for the recovery pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jaxtrace.gpu.recovery.gradient_recovery import build_recovery  # noqa: E402
from jaxtrace.gpu.recovery.hct3d import (  # noqa: E402
    AlfeldGeometry, ALFELD_SUBTET_PARENT_VERTS,
    build_alfeld_geometry, build_hct_bernstein_c1_taylor,
    bernstein_cubic_evaluate,
)


# ---------------------------------------------------------------------------
# Load the case's velocity_fn(pos) as a plain callable (numpy interface).
# ---------------------------------------------------------------------------

def load_case_module(field_module_path: Path):
    """Load rot_cyl_2026/rotating_cylinder_field.py (or similar) as a
    Python module.  Returns the module object; caller uses
    ``module.velocity_fn(pos)`` and ``module.DOMAIN_BBOX``."""
    spec = importlib.util.spec_from_file_location("case_field",
                                                   str(field_module_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate_analytic_batched(velocity_fn, points_xyz: np.ndarray) -> np.ndarray:
    """Evaluate the JAX velocity_fn at every point.  The field modules
    accept a single (3,) position, so we vmap-loop.  For a few 10^5
    points this is a few 100 ms; not a bottleneck."""
    import jax
    import jax.numpy as jnp
    vfn_v = jax.vmap(velocity_fn)
    return np.asarray(vfn_v(jnp.asarray(points_xyz)))


# ---------------------------------------------------------------------------
# Target grid construction
# ---------------------------------------------------------------------------

@dataclass
class UniformBlock:
    """One uniform Cartesian brick of cells."""
    x_lo: np.ndarray   # (3,)
    x_hi: np.ndarray   # (3,)
    n_cells: tuple[int, int, int]

    @property
    def dx(self) -> np.ndarray:
        return (self.x_hi - self.x_lo) / np.asarray(self.n_cells,
                                                     dtype=np.float64)

    def cell_centres(self) -> np.ndarray:
        """(N, 3) cell-centre coordinates for the block."""
        nx, ny, nz = self.n_cells
        xs = self.x_lo[0] + (np.arange(nx) + 0.5) * self.dx[0]
        ys = self.x_lo[1] + (np.arange(ny) + 0.5) * self.dx[1]
        zs = self.x_lo[2] + (np.arange(nz) + 0.5) * self.dx[2]
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def cell_corners(self) -> tuple[np.ndarray, np.ndarray]:
        """Corner and connectivity arrays for a VTK hex-cell mesh
        representation of this block.  Returns (points, hex_conn)."""
        nx, ny, nz = self.n_cells
        cxs = self.x_lo[0] + np.arange(nx + 1) * self.dx[0]
        cys = self.x_lo[1] + np.arange(ny + 1) * self.dx[1]
        czs = self.x_lo[2] + np.arange(nz + 1) * self.dx[2]
        X, Y, Z = np.meshgrid(cxs, cys, czs, indexing="ij")
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        idx = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k
        conn = np.empty((nx * ny * nz, 8), dtype=np.int64)
        c = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    conn[c, 0] = idx(i,     j,     k    )
                    conn[c, 1] = idx(i + 1, j,     k    )
                    conn[c, 2] = idx(i + 1, j + 1, k    )
                    conn[c, 3] = idx(i,     j + 1, k    )
                    conn[c, 4] = idx(i,     j,     k + 1)
                    conn[c, 5] = idx(i + 1, j,     k + 1)
                    conn[c, 6] = idx(i + 1, j + 1, k + 1)
                    conn[c, 7] = idx(i,     j + 1, k + 1)
                    c += 1
        return pts, conn


@dataclass
class HierGrid:
    """Hierarchical block-wise refined uniform grid.

    List of blocks in FINEST-FIRST order.  A query point is assigned to
    the FIRST block that contains it; the base block should therefore
    be listed LAST as the fallback.  Blocks may overlap the base but
    not each other at the same refinement level.
    """
    label: str
    blocks: list[UniformBlock]

    def cell_centres(self, mask_by_finer: bool = True) -> np.ndarray:
        """Return the concatenated cell-centre positions across all
        blocks, with cells of a *base* block excluded whenever they lie
        inside any finer block (so we don't double-count the refined
        region in the error stats)."""
        centres_all: list[np.ndarray] = []
        # finer-first
        for i, blk in enumerate(self.blocks):
            c = blk.cell_centres()
            if mask_by_finer and i < len(self.blocks) - 1:
                # this is the finest block: keep all its centres
                centres_all.append(c)
            elif mask_by_finer:
                # base block: drop centres that fall inside any finer block
                keep = np.ones(c.shape[0], dtype=bool)
                for finer in self.blocks[:i]:
                    inside = np.all((c >= finer.x_lo) & (c < finer.x_hi),
                                    axis=1)
                    keep &= ~inside
                centres_all.append(c[keep])
            else:
                centres_all.append(c)
        return np.vstack(centres_all)

    def total_cells(self) -> int:
        return sum(int(np.prod(b.n_cells)) for b in self.blocks)


def build_target_grids(case_name: str, domain_bbox: tuple,
                       feature_centre: tuple[float, float, float],
                       feature_radius_1: float,
                       feature_radius_2: float,
                       base_n: tuple[int, int, int]) -> list[HierGrid]:
    """Return the three target grids for a case:

        uniform_base        -- one block, n_cells = base_n
        blockref_2lvl       -- base + 2x-refined block within feature_radius_1
        blockref_4lvl       -- base + 2x block + 4x block (nested)

    ``feature_centre`` is where the block-refined region is centred (0,0,0
    for both cases here).  ``feature_radius_1`` gives the outer-refined
    block's half-extent (isotropic); ``feature_radius_2`` gives the inner.

    Refinement blocks are square in x-y and full-thin in z (matching the
    2D-in-plane nature of both cases).
    """
    bbox_min = np.array([domain_bbox[0][0], domain_bbox[1][0],
                          domain_bbox[2][0]])
    bbox_max = np.array([domain_bbox[0][1], domain_bbox[1][1],
                          domain_bbox[2][1]])
    fc = np.asarray(feature_centre, dtype=np.float64)

    base = UniformBlock(x_lo=bbox_min, x_hi=bbox_max, n_cells=base_n)

    # 2-lvl refined block (half-extent = radius_1, full z-extent, 2x cells).
    r1 = feature_radius_1
    inner1 = UniformBlock(
        x_lo=np.array([fc[0] - r1, fc[1] - r1, bbox_min[2]]),
        x_hi=np.array([fc[0] + r1, fc[1] + r1, bbox_max[2]]),
        n_cells=(base_n[0] * 2, base_n[1] * 2, base_n[2] * 2),
    )
    # 4-lvl refined block (nested inside 2-lvl, extent radius_2, 4x cells).
    r2 = feature_radius_2
    inner2 = UniformBlock(
        x_lo=np.array([fc[0] - r2, fc[1] - r2, bbox_min[2]]),
        x_hi=np.array([fc[0] + r2, fc[1] + r2, bbox_max[2]]),
        n_cells=(base_n[0] * 4, base_n[1] * 4, base_n[2] * 4),
    )

    return [
        HierGrid("uniform_base",  [base]),
        HierGrid("blockref_2lvl", [inner1, base]),          # finer first
        HierGrid("blockref_4lvl", [inner2, inner1, base]),
    ]


# ---------------------------------------------------------------------------
# Mesh loading + P1 barycentric probe + vertex-Taylor evaluator
# ---------------------------------------------------------------------------

def load_mesh_pvtu(pvtu_path: Path):
    """Load a PVTU (or plain VTU) mesh and return (points, connectivity)
    for the tet cells."""
    if pvtu_path.suffix == ".pvtu":
        r = vtk.vtkXMLPUnstructuredGridReader()
    else:
        r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(pvtu_path))
    r.Update()
    ug = r.GetOutput()
    pts = vtk_to_numpy(ug.GetPoints().GetData()).astype(np.float64)
    n_cells = ug.GetNumberOfCells()
    conn = np.empty((n_cells, 4), dtype=np.int64)
    cell = vtk.vtkGenericCell()
    for i in range(n_cells):
        ug.GetCell(i, cell)
        if cell.GetCellType() != vtk.VTK_TETRA:
            raise ValueError(
                f"cell {i} in {pvtu_path.name} is not a tetrahedron "
                f"(type {cell.GetCellType()}); this tool requires pure-tet meshes.")
        ids = cell.GetPointIds()
        for j in range(4):
            conn[i, j] = ids.GetId(j)
    return ug, pts, conn


def make_mesh_with_field(pts: np.ndarray, conn: np.ndarray,
                         v_nodal: np.ndarray, field_name: str = "v"):
    """Return a vtkUnstructuredGrid built from arrays with a per-node
    vector array attached.  Used as the source for vtkProbeFilter."""
    ug = vtk.vtkUnstructuredGrid()
    vpts = vtk.vtkPoints()
    vpts.SetData(numpy_to_vtk(pts.astype(np.float64), deep=True))
    ug.SetPoints(vpts)
    n_cells = conn.shape[0]
    id_arr = np.empty(5 * n_cells, dtype=np.int64)
    id_arr[0::5] = 4
    id_arr[1::5] = conn[:, 0]
    id_arr[2::5] = conn[:, 1]
    id_arr[3::5] = conn[:, 2]
    id_arr[4::5] = conn[:, 3]
    cell_arr = vtk.vtkCellArray()
    cell_arr.SetCells(n_cells, numpy_to_vtkIdTypeArray(id_arr, deep=True))
    cell_types = np.full(n_cells, vtk.VTK_TETRA, dtype=np.uint8)
    ug.SetCells(numpy_to_vtk(cell_types, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR),
                cell_arr)
    varr = numpy_to_vtk(np.ascontiguousarray(v_nodal.astype(np.float32)),
                        deep=True)
    varr.SetName(field_name)
    varr.SetNumberOfComponents(3)
    ug.GetPointData().AddArray(varr)
    return ug


def probe_field_at_points(source_ug, query_points_xyz: np.ndarray,
                          field_name: str = "v") -> tuple[np.ndarray, np.ndarray]:
    """P1 barycentric interpolation of the source's field at the query
    points, via vtkProbeFilter.  Returns (v_interp, valid_mask)."""
    n = query_points_xyz.shape[0]
    probe_pts = vtk.vtkPoints()
    probe_pts.SetData(numpy_to_vtk(query_points_xyz.astype(np.float64),
                                    deep=True))
    probe_poly = vtk.vtkPolyData()
    probe_poly.SetPoints(probe_pts)

    probe = vtk.vtkProbeFilter()
    probe.SetInputData(probe_poly)
    probe.SetSourceData(source_ug)
    probe.Update()
    out = probe.GetOutput()
    v = vtk_to_numpy(out.GetPointData().GetArray(field_name))
    valid = vtk_to_numpy(out.GetPointData().GetArray("vtkValidPointMask"))
    return v.astype(np.float64), valid.astype(bool)


def evaluate_hct_cubic_at_points(pts_mesh: np.ndarray,
                                 conn: np.ndarray,
                                 hct_coeffs: np.ndarray,
                                 alfeld: AlfeldGeometry,
                                 query_xyz: np.ndarray,
                                 source_ug) -> tuple[np.ndarray, np.ndarray]:
    """HCT-3D cubic Bernstein reconstruction evaluated at grid nodes.

    For each query point:
      1. Find its enclosing parent tet via vtkCellLocator.
      2. Compute the parent-tet barycentric weights.
      3. Detect which of the 4 Alfeld sub-tets the point sits in:
         sub-tet s is opposite parent vertex s, so the point is in
         sub-tet argmin(parent_barycentric).
      4. Compute the sub-tet barycentric weights (using the sub-tet's
         4 vertices: (p_j, p_k, p_l, vc) per ALFELD_SUBTET_PARENT_VERTS).
      5. Evaluate the cubic Bernstein via bernstein_cubic_evaluate.

    Returns (v_out, valid_mask).
    """
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(source_ug)
    locator.BuildLocator()

    n_q = query_xyz.shape[0]
    v_out = np.zeros((n_q, 3), dtype=np.float64)
    valid = np.zeros(n_q, dtype=bool)

    cell = vtk.vtkGenericCell()
    pcoords = [0.0, 0.0, 0.0]
    weights_parent = [0.0, 0.0, 0.0, 0.0]

    for i in range(n_q):
        p = query_xyz[i]
        cid = locator.FindCell(p, 1e-6, cell, pcoords, weights_parent)
        if cid < 0:
            continue
        w_parent = np.asarray(weights_parent[:4], dtype=np.float64)
        # Alfeld sub-tet detection: the point is in sub-tet s where s is
        # the parent vertex that gets replaced by vc.  Standard result:
        # inside sub-tet s the parent-vertex barycentric at s is the
        # smallest (goes to 0 at the sub-tet's vc-facing face).
        s = int(np.argmin(w_parent))
        # Sub-tet s' 4 vertices are (p_j, p_k, p_l, vc) with j,k,l =
        # ALFELD_SUBTET_PARENT_VERTS[s].  Compute sub-tet barycentric.
        jkl = ALFELD_SUBTET_PARENT_VERTS[s]
        tet_nodes = conn[cid]
        v_j = pts_mesh[tet_nodes[jkl[0]]]
        v_k = pts_mesh[tet_nodes[jkl[1]]]
        v_l = pts_mesh[tet_nodes[jkl[2]]]
        v_c = alfeld.centroids[cid]
        # Solve M @ [b_j, b_k, b_l]^T = (p - v_c) where columns of M are
        # (v_j - v_c), (v_k - v_c), (v_l - v_c); then b_vc = 1 - sum.
        M = np.column_stack([v_j - v_c, v_k - v_c, v_l - v_c])
        try:
            bjkl = np.linalg.solve(M, p - v_c)
        except np.linalg.LinAlgError:
            continue
        bvc = 1.0 - bjkl.sum()
        bary = np.array([bjkl[0], bjkl[1], bjkl[2], bvc], dtype=np.float64)
        # Clamp tiny negatives that can come from floating-point roundoff
        # near sub-tet faces (bernstein_cubic_evaluate uses α exponents
        # that don't strictly need non-negative b, but numerically we
        # want to stay inside the sub-tet).
        bary = np.maximum(bary, -1e-6)
        # Evaluate the cubic Bernstein on sub-tet s.
        v_out[i] = bernstein_cubic_evaluate(hct_coeffs[cid, s], bary)
        valid[i] = True
    return v_out, valid


def evaluate_vertex_taylor_at_points(pts_mesh: np.ndarray,
                                     conn: np.ndarray,
                                     v_nodal: np.ndarray,
                                     nodal_gradient: np.ndarray,
                                     query_xyz: np.ndarray,
                                     source_ug) -> tuple[np.ndarray, np.ndarray]:
    """Vertex-Taylor evaluator: for each query point, find its enclosing
    tet (via vtkCellLocator), compute barycentric coords, then

        v(p) = Σₐ Nₐ(p) · [ v_a + G_a · (p − x_a) ]

    where a runs over the 4 vertices of the tet, N_a are the barycentric
    weights, v_a is the nodal velocity, G_a is the SPR-recovered nodal
    gradient tensor.  See gradient_recovery.py docstring for the formula.

    We use vtkCellLocator to find the enclosing cell (same job the
    tracker's L0/L1/L2 does — for this offline error study we don't need
    speed).
    """
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(source_ug)
    locator.BuildLocator()

    n_q = query_xyz.shape[0]
    v_out = np.zeros((n_q, 3), dtype=np.float64)
    valid = np.zeros(n_q, dtype=bool)

    cell = vtk.vtkGenericCell()
    tol2 = vtk.reference(0.0)
    sub_id = vtk.reference(0)
    pcoords = [0.0, 0.0, 0.0]
    weights = [0.0, 0.0, 0.0, 0.0]

    for i in range(n_q):
        p = query_xyz[i]
        cid = locator.FindCell(p, 1e-6, cell, pcoords, weights)
        if cid < 0:
            continue
        # Barycentric weights in `weights` (4 for a tet, matching cell
        # vertex order).  Get the tet's node indices and evaluate.
        # Sanity: cid indexes into the mesh's cell list, which is the
        # same order as our `conn` array.
        tet_nodes = conn[cid]        # (4,) node indices
        x_a = pts_mesh[tet_nodes]    # (4, 3)
        v_a = v_nodal[tet_nodes]     # (4, 3)
        G_a = nodal_gradient[tet_nodes]  # (4, 3, 3)
        w = np.asarray(weights[:4])
        # sum over a: N_a · (v_a + G_a · (p − x_a))
        dx = p - x_a                  # (4, 3)
        Gdx = np.einsum("aij,aj->ai", G_a, dx)   # (4, 3)
        v_out[i] = np.einsum("a,ai->i", w, v_a + Gdx)
        valid[i] = True
    return v_out, valid


# ---------------------------------------------------------------------------
# VTU writer for the per-cell error field
# ---------------------------------------------------------------------------

def write_error_vtu(out_path: Path, grid: HierGrid,
                    err_per_centre: np.ndarray) -> None:
    """Write one VTU per hierarchical grid.  Concatenates the blocks'
    hex cells, attaches |error| per cell."""
    all_pts: list[np.ndarray] = []
    all_conn: list[np.ndarray] = []
    all_err:  list[np.ndarray] = []
    all_mask: list[np.ndarray] = []
    pt_offset = 0
    err_offset = 0

    for bi, blk in enumerate(grid.blocks):
        pts_b, conn_b = blk.cell_corners()
        # mask which centres of THIS block were kept in the error calc.
        # cell_centres() drops centres inside any finer block for base
        # blocks — replicate that here.
        cent = blk.cell_centres()
        kept = np.ones(cent.shape[0], dtype=bool)
        if bi < len(grid.blocks) - 1:
            # finest / mid block: kept everything
            pass
        else:
            for finer in grid.blocks[:bi]:
                inside = np.all((cent >= finer.x_lo) & (cent < finer.x_hi),
                                axis=1)
                kept &= ~inside

        # for cells not kept (double-counted region), fill with NaN so
        # ParaView colouring highlights them
        block_err = np.full(cent.shape[0], np.nan, dtype=np.float64)
        n_kept = int(kept.sum())
        block_err[kept] = err_per_centre[err_offset:err_offset + n_kept]
        err_offset += n_kept

        all_pts.append(pts_b)
        all_conn.append(conn_b + pt_offset)
        all_err.append(block_err)
        pt_offset += pts_b.shape[0]

    pts = np.vstack(all_pts)
    conn = np.vstack(all_conn)
    err = np.concatenate(all_err)

    ug = vtk.vtkUnstructuredGrid()
    vpts = vtk.vtkPoints()
    vpts.SetData(numpy_to_vtk(pts.astype(np.float64), deep=True))
    ug.SetPoints(vpts)
    n_cells = conn.shape[0]
    id_arr = np.empty(9 * n_cells, dtype=np.int64)
    id_arr[0::9] = 8
    for k in range(8):
        id_arr[k + 1::9] = conn[:, k]
    cell_arr = vtk.vtkCellArray()
    cell_arr.SetCells(n_cells, numpy_to_vtkIdTypeArray(id_arr, deep=True))
    cell_types = np.full(n_cells, vtk.VTK_HEXAHEDRON, dtype=np.uint8)
    ug.SetCells(
        numpy_to_vtk(cell_types, deep=True,
                     array_type=vtk.VTK_UNSIGNED_CHAR),
        cell_arr,
    )
    varr = numpy_to_vtk(err.astype(np.float32), deep=True)
    varr.SetName("proj_error")
    ug.GetCellData().AddArray(varr)

    w = vtk.vtkXMLUnstructuredGridWriter()
    w.SetFileName(str(out_path))
    w.SetInputData(ug)
    w.SetDataModeToBinary()
    w.SetCompressorTypeToZLib()
    w.Write()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case-root", type=Path, required=True,
                    help="e.g. /scratch/shared/ROM/FOM_analytic/recirc_2026/")
    ap.add_argument("--field-module", type=Path, required=True,
                    help="e.g. recirculation_field.py")
    ap.add_argument("--source-mesh", type=Path, required=True,
                    help="e.g. mesh_4lvl/mesh_0.pvtu")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--case-name", type=str, required=True,
                    help="Label used in CSV + VTU filenames.")
    ap.add_argument("--base-n", type=int, nargs=3, default=[64, 32, 8],
                    help="Base uniform grid n_cells in x y z.")
    ap.add_argument("--refine-r1", type=float, default=None,
                    help="Half-extent of the 2-lvl refined block (m). "
                         "Default: 1/4 of the bbox x-extent.")
    ap.add_argument("--refine-r2", type=float, default=None,
                    help="Half-extent of the 4-lvl refined block (m). "
                         "Default: 1/8 of the bbox x-extent.")
    ap.add_argument("--methods", type=str, nargs="+",
                    default=["p1_raw", "vertex_taylor", "hct_cubic"],
                    help="Projection methods. Available: p1_raw "
                         "(vtkProbeFilter), vertex_taylor (SPR + per-"
                         "vertex Taylor blend), hct_cubic (Alfeld-split "
                         "Bernstein cubic, Phase 3b Taylor-fit).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the case module and get velocity_fn + DOMAIN_BBOX.
    print(f"[proj] loading case module {args.field_module.name}")
    mod = load_case_module(args.field_module)
    domain_bbox = getattr(mod, "DOMAIN_BBOX", None)
    if domain_bbox is None:
        raise RuntimeError(
            f"{args.field_module}: module has no DOMAIN_BBOX; add one or "
            f"pass explicit bbox via CLI.")
    print(f"[proj] DOMAIN_BBOX = {domain_bbox}")

    # Default feature radii: 1/4 and 1/8 of the bbox x-extent.
    x_extent = domain_bbox[0][1] - domain_bbox[0][0]
    r1 = args.refine_r1 if args.refine_r1 is not None else x_extent / 4.0
    r2 = args.refine_r2 if args.refine_r2 is not None else x_extent / 8.0
    print(f"[proj] refine radii: r1={r1:g}, r2={r2:g}")

    # 2. Load 4-lvl mesh + evaluate analytic velocity at its nodes.
    print(f"[proj] loading source mesh {args.source_mesh}")
    ug_mesh, pts_mesh, conn = load_mesh_pvtu(args.source_mesh)
    print(f"[proj]   {pts_mesh.shape[0]:,} nodes, {conn.shape[0]:,} tets")

    print(f"[proj] evaluating analytic velocity at mesh nodes...")
    t0 = time.time()
    v_nodal = evaluate_analytic_batched(mod.velocity_fn, pts_mesh)
    print(f"[proj]   done ({time.time() - t0:.1f}s)")
    print(f"[proj]   |v_nodal|: min={np.linalg.norm(v_nodal, axis=1).min():.4g}, "
          f"max={np.linalg.norm(v_nodal, axis=1).max():.4g}")

    source_ug = make_mesh_with_field(pts_mesh, conn, v_nodal)

    # 3. Run gradient recovery once if vertex_taylor OR hct_cubic is requested.
    recovery = None
    if any(m in args.methods for m in ("vertex_taylor", "hct_cubic")):
        print(f"[proj] running SPR gradient recovery")
        recovery = build_recovery(pts_mesh, conn, v_nodal,
                                  method="vertex_taylor", verbose=False)
        print(f"[proj]   nodal_gradient shape: {recovery.nodal_gradient.shape}")

    # 3b. Build Alfeld geometry + HCT-3D Bernstein coefficients if needed.
    alfeld = None
    hct_coeffs = None
    if "hct_cubic" in args.methods:
        print(f"[proj] building Alfeld geometry + HCT-3D Bernstein "
              f"coefficients (Phase 3b Taylor-fit)")
        alfeld = build_alfeld_geometry(pts_mesh, conn, verbose=False)
        hct = build_hct_bernstein_c1_taylor(
            pts_mesh, conn, v_nodal,
            recovery.nodal_gradient,
            edge_grads=None,  # ignored by the Taylor-fit variant
            geom=alfeld, verbose=False,
        )
        hct_coeffs = hct.coeffs
        print(f"[proj]   hct_coeffs shape: {hct_coeffs.shape}, "
              f"size {hct_coeffs.nbytes/1e6:.1f} MB")

    # 4. Build target grids.
    grids = build_target_grids(
        args.case_name, domain_bbox,
        feature_centre=(0.0, 0.0, 0.0),
        feature_radius_1=r1, feature_radius_2=r2,
        base_n=tuple(args.base_n),
    )

    rows: list[dict] = []

    # 5. For each grid × method, project + score.
    for grid in grids:
        centres = grid.cell_centres(mask_by_finer=True)
        n_cell = centres.shape[0]
        print(f"\n[proj] === grid={grid.label}  cells={n_cell:,} ===")

        # Ground truth at grid cell centres
        print(f"[proj]   evaluating analytic at grid centres")
        v_true = evaluate_analytic_batched(mod.velocity_fn, centres)

        for method in args.methods:
            print(f"[proj]   method={method}")
            t0 = time.time()
            if method == "p1_raw":
                v_proj, valid = probe_field_at_points(source_ug, centres)
            elif method == "vertex_taylor":
                v_proj, valid = evaluate_vertex_taylor_at_points(
                    pts_mesh, conn, v_nodal, recovery.nodal_gradient,
                    centres, source_ug,
                )
            elif method == "hct_cubic":
                v_proj, valid = evaluate_hct_cubic_at_points(
                    pts_mesh, conn, hct_coeffs, alfeld,
                    centres, source_ug,
                )
            else:
                raise ValueError(f"unknown method: {method}")
            dt = time.time() - t0

            n_valid = int(valid.sum())
            if n_valid == 0:
                print(f"     NO valid queries; skipping")
                continue
            err = np.linalg.norm(v_proj - v_true, axis=1)
            err_valid = err[valid]
            v_true_norm = np.linalg.norm(v_true[valid], axis=1)
            v_scale = float(np.max(v_true_norm))
            rms = float(np.sqrt((err_valid ** 2).mean()))
            mean = float(err_valid.mean())
            mx = float(err_valid.max())
            print(f"     {dt:.1f}s  n_valid={n_valid:,}  "
                  f"rms={rms:.4g}  mean={mean:.4g}  max={mx:.4g}  "
                  f"rel_rms={100*rms/v_scale:.3f}%")
            rows.append({
                "case":       args.case_name,
                "grid":       grid.label,
                "method":     method,
                "n_cells":    n_cell,
                "n_valid":    n_valid,
                "v_scale":    v_scale,
                "err_rms":    rms,
                "err_mean":   mean,
                "err_max":    mx,
                "err_rel_rms_pct": 100.0 * rms / max(v_scale, 1e-30),
                "wall_time_s": dt,
            })
            vtu_path = (args.out_dir /
                        f"projection_error_{args.case_name}_"
                        f"{grid.label}_{method}.vtu")
            write_error_vtu(vtu_path, grid, err)
            print(f"     -> {vtu_path.name}")

    # 6. CSV summary.
    csv_path = args.out_dir / f"projection_summary_{args.case_name}.csv"
    fields = ["case", "grid", "method", "n_cells", "n_valid",
              "v_scale", "err_rms", "err_mean", "err_max",
              "err_rel_rms_pct", "wall_time_s"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[proj] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
