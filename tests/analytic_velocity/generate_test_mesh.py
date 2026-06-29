"""
generate_test_mesh.py
=====================

Build a Kuhn-tetrahedralised structured tetrahedral mesh covering a
user-specified bounding box, evaluate a user-supplied analytic velocity
function at every node, and write the result as a JAXTrace-compatible
PVTU + per-piece VTU.

The resulting mesh-loading path of run_tracking.py reads this PVTU and
sees the same field values that the analytic path computes directly.
Trajectory differences between the two paths are therefore attributable
to **interpolation error** alone (search is exact at nodes, RK4 is
identical, kernel is the same).

Mesh structure
--------------
A regular hex grid of (N_x+1)·(N_y+1)·(N_z+1) nodes covering the bbox.
Each hex cell is decomposed into 6 tetrahedra by the Freudenthal-Kuhn
construction — the same decomposition the paper's MALMO benchmark
assumes, so the spatial search structure matches what cohort runs use.

Node-index convention (local-to-cell, z-major within each cell):
    local 0 = (i,   j,   k  )    pattern 000
    local 1 = (i+1, j,   k  )    pattern 100  (bit 0 = +x)
    local 2 = (i,   j+1, k  )    pattern 010  (bit 1 = +y)
    local 3 = (i+1, j+1, k  )    pattern 110
    local 4 = (i,   j,   k+1)    pattern 001  (bit 2 = +z)
    local 5 = (i+1, j,   k+1)    pattern 101
    local 6 = (i,   j+1, k+1)    pattern 011
    local 7 = (i+1, j+1, k+1)    pattern 111

The 6 tets are the six monotone paths 0 → 7 along axis-aligned edges:
    xyz: 0-1-3-7    xzy: 0-1-5-7
    yxz: 0-2-3-7    yzx: 0-2-6-7
    zxy: 0-4-5-7    zyx: 0-4-6-7

Usage
-----
    python tests/analytic_velocity/generate_test_mesh.py \\
        --velocity-module jaxtrace/analytic_fields/divergence_free_recirculation.py \\
        --bbox -4 4 -2 2 -0.25 0.25 \\
        --n-cells 64 32 8 \\
        --output tests/analytic_velocity/mesh_recirc_n64x32x8/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure jaxtrace package is importable when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

import vtk
from vtk.util import numpy_support


# =============================================================================
# Kuhn tetrahedralisation
# =============================================================================

# The six tets per hex, each as a list of 4 local node indices in the
# 0..7 corner-numbering above. Indices follow the bit-pattern convention.
#
# IMPORTANT: vertex order is chosen so every tet has POSITIVE signed
# volume (right-handed). JAXTrace's point-in-tet test relies on this:
# inverted tets silently fail host-element lookup, leaving every
# particle "lost" with element_id = -1.
#
# Of the six naive monotone paths 0 -> 7, three are right-handed
# (xyz, yzx, zxy) and three are left-handed (xzy, yxz, zyx). The
# inverted ones are fixed by swapping the last two indices (which
# flips the orientation).
KUHN_TETS = (
    (0, 1, 3, 7),   # xyz   (right-handed)
    (0, 1, 7, 5),   # xzy   (fixed: swapped last two)
    (0, 2, 7, 3),   # yxz   (fixed: swapped last two)
    (0, 2, 6, 7),   # yzx   (right-handed)
    (0, 4, 5, 7),   # zxy   (right-handed)
    (0, 4, 7, 6),   # zyx   (fixed: swapped last two)
)


def build_node_grid(bbox_min, bbox_max, n_cells):
    """Build a regular (Nx+1)·(Ny+1)·(Nz+1) node grid covering the bbox.

    Args
    ----
    bbox_min, bbox_max : (3,) array-like of float
        Domain corners in metres.
    n_cells : (3,) array-like of int
        Number of hex cells along each axis.

    Returns
    -------
    nodes : (N, 3) float64
        Node positions, ordered such that node index =
            i + (Nx+1)·j + (Nx+1)·(Ny+1)·k
    shape : (3,) int
        (Nx+1, Ny+1, Nz+1) — the per-axis node count.
    """
    nx, ny, nz = int(n_cells[0]), int(n_cells[1]), int(n_cells[2])
    xs = np.linspace(bbox_min[0], bbox_max[0], nx + 1, dtype=np.float64)
    ys = np.linspace(bbox_min[1], bbox_max[1], ny + 1, dtype=np.float64)
    zs = np.linspace(bbox_min[2], bbox_max[2], nz + 1, dtype=np.float64)
    # 'ij' indexing gives shape (nx+1, ny+1, nz+1, 3).
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    # Flatten in the order matching the local-to-global mapping below:
    # idx = i + (nx+1)·j + (nx+1)·(ny+1)·k
    nodes = np.stack(
        [X.transpose(2, 1, 0).ravel(),
         Y.transpose(2, 1, 0).ravel(),
         Z.transpose(2, 1, 0).ravel()],
        axis=1,
    )
    return nodes, (nx + 1, ny + 1, nz + 1)


def build_kuhn_connectivity(n_cells, n_per_axis):
    """Build the (n_elements, 4) tetrahedral connectivity array.

    Args
    ----
    n_cells : (3,) int
        Number of hex cells along each axis.
    n_per_axis : (3,) int
        (Nx+1, Ny+1, Nz+1) — the per-axis node count.

    Returns
    -------
    connectivity : (n_cells_total · 6, 4) int32
    """
    nx, ny, nz = int(n_cells[0]), int(n_cells[1]), int(n_cells[2])
    Nx, Ny, _Nz = n_per_axis  # node-count per axis

    def node_idx(i, j, k):
        return i + Nx * j + Nx * Ny * k

    # Offsets for the eight hex corners, indexed by bit pattern.
    # local 0..7 corresponds to (i+bit0, j+bit1, k+bit2).
    bit_offsets = [
        (0, 0, 0),  # 0
        (1, 0, 0),  # 1
        (0, 1, 0),  # 2
        (1, 1, 0),  # 3
        (0, 0, 1),  # 4
        (1, 0, 1),  # 5
        (0, 1, 1),  # 6
        (1, 1, 1),  # 7
    ]

    n_hex = nx * ny * nz
    connectivity = np.empty((n_hex * 6, 4), dtype=np.int32)
    t = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # Resolve the 8 corner global node indices for this hex.
                corners = [
                    node_idx(i + di, j + dj, k + dk)
                    for di, dj, dk in bit_offsets
                ]
                for tet in KUHN_TETS:
                    connectivity[t, 0] = corners[tet[0]]
                    connectivity[t, 1] = corners[tet[1]]
                    connectivity[t, 2] = corners[tet[2]]
                    connectivity[t, 3] = corners[tet[3]]
                    t += 1
    return connectivity


# =============================================================================
# Adaptive Kuhn refinement
# =============================================================================
#
# Each hex cell is represented as a 4-tuple (i, j, k, level) where (i, j, k)
# are integer grid indices at that level: the cell occupies
#     [bbox_min + (i, j, k) * cell_size_l,  bbox_min + (i+1, j+1, k+1) * cell_size_l]
# with cell_size_l = base_cell_size / 2**level.
#
# Refinement zone: for each (radius, center) in the user-supplied list,
# we replace every cell whose centroid lies within `radius` of `center`
# by its 8 child cells at level+1. Refinements compose: a level-2 region
# inside a level-1 region needs two passes with successively smaller
# radii.
#
# Result: a flat list of (i, j, k, level) cells AT MULTIPLE LEVELS, plus a
# shared node array deduplicated by exact float position. Nodes are emitted
# per-cell from the 8 corner coordinates; the dedup map collapses
# coincident corners (which is what makes adjacent same-level cells
# share faces; T-junctions between levels do NOT dedup because the
# coarse cell's corner doesn't sit on the fine cell's edge-midpoint).
#
# T-junctions are intentional: JAXTrace's L2 search walks every neighbour
# octree cell across every level, so a particle near a level boundary
# finds candidate tets from both sides. The mesh is "search-conformal"
# without being "topologically conformal".


def _hex_centroid(i, j, k, level, bbox_min, base_cs):
    cs = base_cs / (2 ** level)
    return (
        bbox_min[0] + (i + 0.5) * cs[0],
        bbox_min[1] + (j + 0.5) * cs[1],
        bbox_min[2] + (k + 0.5) * cs[2],
    )


def _subdivide_cell(i, j, k, level):
    """Replace one hex cell with its 8 children at level+1."""
    i2 = 2 * i
    j2 = 2 * j
    k2 = 2 * k
    L2 = level + 1
    return [
        (i2,     j2,     k2,     L2),
        (i2 + 1, j2,     k2,     L2),
        (i2,     j2 + 1, k2,     L2),
        (i2 + 1, j2 + 1, k2,     L2),
        (i2,     j2,     k2 + 1, L2),
        (i2 + 1, j2,     k2 + 1, L2),
        (i2,     j2 + 1, k2 + 1, L2),
        (i2 + 1, j2 + 1, k2 + 1, L2),
    ]


def build_adaptive_kuhn_mesh(bbox_min, bbox_max, base_n_cells, refinements):
    """Build a multi-level Kuhn-tet mesh.

    Args
    ----
    bbox_min, bbox_max : (3,) array of float
    base_n_cells       : (3,) array of int
        Hex grid at level 0 (the coarsest level).
    refinements        : list of (radius, center) tuples
        For each pair, every current cell whose centroid is within
        `radius` of `center` is subdivided into 8 child cells at the
        next level. Pairs are applied in order, so successive entries
        create successively finer nested regions.

    Returns
    -------
    nodes        : (n_nodes, 3) float64
        Deduplicated by exact position.
    connectivity : (n_tets, 4) int32
        Six Kuhn tets per surviving hex cell.
    cell_levels  : (n_cells,) int32
        Level of each hex cell (for diagnostics; len = n_tets // 6).
    """
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_max = np.asarray(bbox_max, dtype=np.float64)
    base_n_cells = np.asarray(base_n_cells, dtype=np.int64)
    base_cs = (bbox_max - bbox_min) / base_n_cells

    # Start with one cell per base grid position at level 0.
    cells = [
        (i, j, k, 0)
        for k in range(int(base_n_cells[2]))
        for j in range(int(base_n_cells[1]))
        for i in range(int(base_n_cells[0]))
    ]

    # Apply refinement zones successively.
    for r_idx, (radius, center) in enumerate(refinements):
        r2 = float(radius) ** 2
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        new_cells = []
        n_refined = 0
        for (i, j, k, level) in cells:
            x, y, z = _hex_centroid(i, j, k, level, bbox_min, base_cs)
            d2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
            if d2 <= r2:
                new_cells.extend(_subdivide_cell(i, j, k, level))
                n_refined += 1
            else:
                new_cells.append((i, j, k, level))
        cells = new_cells
        print(f"  Refinement pass {r_idx + 1}: radius={radius:g}, "
              f"center={tuple(center)}: refined {n_refined:,} cells "
              f"-> {len(cells):,} total")

    # Build per-cell corner coordinates and dedup into a single node array.
    # We key by quantised position to avoid float-fuzz collisions. The
    # quantisation scale is the minimum cell-size component over the
    # finest level present, divided by 1024 — finer than any geometric
    # feature in the mesh.
    if not cells:
        raise RuntimeError("Adaptive mesh ended up empty.")
    max_level = max(level for _, _, _, level in cells)
    finest_cs = base_cs / (2 ** max_level)
    quant_eps = float(np.min(finest_cs)) / 1024.0

    node_map: dict[tuple[int, int, int], int] = {}
    nodes_list: list[tuple[float, float, float]] = []

    def _add_node(x, y, z):
        key = (
            int(round(x / quant_eps)),
            int(round(y / quant_eps)),
            int(round(z / quant_eps)),
        )
        nid = node_map.get(key)
        if nid is None:
            nid = len(nodes_list)
            node_map[key] = nid
            nodes_list.append((x, y, z))
        return nid

    # Bit-pattern → axis offset (matches the uniform mesh convention).
    bit_offsets = [
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
        (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
    ]

    connectivity = np.empty((len(cells) * 6, 4), dtype=np.int32)
    cell_levels_out = np.empty(len(cells), dtype=np.int32)
    t = 0
    for c_idx, (i, j, k, level) in enumerate(cells):
        cell_levels_out[c_idx] = level
        cs = base_cs / (2 ** level)
        x0 = bbox_min[0] + i * cs[0]
        y0 = bbox_min[1] + j * cs[1]
        z0 = bbox_min[2] + k * cs[2]
        corners = []
        for di, dj, dk in bit_offsets:
            nid = _add_node(x0 + di * cs[0], y0 + dj * cs[1], z0 + dk * cs[2])
            corners.append(nid)
        for tet in KUHN_TETS:
            connectivity[t, 0] = corners[tet[0]]
            connectivity[t, 1] = corners[tet[1]]
            connectivity[t, 2] = corners[tet[2]]
            connectivity[t, 3] = corners[tet[3]]
            t += 1

    nodes = np.asarray(nodes_list, dtype=np.float64)
    return nodes, connectivity, cell_levels_out


# =============================================================================
# Evaluate the analytic velocity at every node
# =============================================================================

def evaluate_velocity_at_nodes(nodes, velocity_fn, batch_size=4096):
    """Vmap velocity_fn over the node array.

    Args
    ----
    nodes : (N, 3) float64
        Node positions.
    velocity_fn : Callable
        JAX-pure function with signature velocity_fn(pos) -> (3,).
    batch_size : int
        Particles per batch (for large meshes, controls peak memory).

    Returns
    -------
    velocities : (N, 3) float64
    """
    import jax
    import jax.numpy as jnp

    @jax.jit
    def batch_eval(positions):
        return jax.vmap(velocity_fn)(positions)

    nodes_jax = jnp.asarray(nodes, dtype=jnp.float64)
    n = nodes_jax.shape[0]
    out = np.empty((n, 3), dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        v = batch_eval(nodes_jax[start:end])
        out[start:end] = np.asarray(v)
    return out


# =============================================================================
# VTU / PVTU writing
# =============================================================================

def write_vtu(path, nodes, connectivity, field_name, field_values):
    """Write a single VTU piece.

    Args
    ----
    path : Path
        Output file path (`*.vtu`).
    nodes : (N, 3) float64
    connectivity : (M, 4) int32
        Tetrahedral elements.
    field_name : str
        Name of the per-node vector field.
    field_values : (N, 3) float64
        Vector field at every node.
    """
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(
        nodes.astype(np.float64), deep=True, array_type=vtk.VTK_DOUBLE,
    ))

    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)

    # Cells: each tet is (4, n0, n1, n2, n3).
    n_cells = connectivity.shape[0]
    # vtkCellArray expects [n, i0, i1, ..., n, i0, ...] flat layout.
    cells_flat = np.empty((n_cells, 5), dtype=np.int64)
    cells_flat[:, 0] = 4
    cells_flat[:, 1:] = connectivity
    id_array = numpy_support.numpy_to_vtkIdTypeArray(
        cells_flat.ravel(), deep=True,
    )
    cell_array = vtk.vtkCellArray()
    cell_array.SetCells(n_cells, id_array)
    cell_types = np.full(n_cells, vtk.VTK_TETRA, dtype=np.uint8)
    cell_types_vtk = numpy_support.numpy_to_vtk(
        cell_types, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR,
    )
    grid.SetCells(cell_types_vtk, cell_array)

    # Per-node vector field.
    field_vtk = numpy_support.numpy_to_vtk(
        field_values.astype(np.float64), deep=True, array_type=vtk.VTK_DOUBLE,
    )
    field_vtk.SetName(field_name)
    grid.GetPointData().AddArray(field_vtk)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(grid)
    writer.SetDataModeToBinary()
    writer.Write()


def write_pvtu(pvtu_path, vtu_filename, field_name):
    """Write a 1-piece PVTU header pointing at a single VTU file.

    Args
    ----
    pvtu_path : Path
        Output PVTU file path.
    vtu_filename : str
        Name of the sibling VTU file (relative to the PVTU).
    field_name : str
        Name of the per-node vector field declared in the PVTU.
    """
    body = f"""<?xml version="1.0"?>
<VTKFile type="PUnstructuredGrid" version="1.0" byte_order="LittleEndian">
  <PUnstructuredGrid GhostLevel="0">
    <PPointData>
      <PDataArray type="Float64" Name="{field_name}" NumberOfComponents="3"/>
    </PPointData>
    <PPoints>
      <PDataArray type="Float64" NumberOfComponents="3"/>
    </PPoints>
    <Piece Source="{vtu_filename}"/>
  </PUnstructuredGrid>
</VTKFile>
"""
    Path(pvtu_path).write_text(body)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Generate a Kuhn-tetrahedralised structured mesh with "
                    "an analytic velocity field sampled at every node.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--velocity-module", required=True, type=Path,
        help="Path to a JAXTrace analytic-velocity module exposing "
             "build_provider(domain_bbox, dt, t_start). The provider's "
             "velocity_fn is sampled at every node.",
    )
    ap.add_argument(
        "--bbox", type=float, nargs=6, required=True,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Domain bbox in metres.",
    )
    ap.add_argument(
        "--n-cells", type=int, nargs=3, required=True,
        metavar=("NX", "NY", "NZ"),
        help="Number of hex cells along each axis. Total tets = 6·NX·NY·NZ; "
             "total nodes = (NX+1)·(NY+1)·(NZ+1).",
    )
    ap.add_argument(
        "--output", type=Path, required=True,
        help="Output directory. Two files are written: <stem>.pvtu and "
             "<stem>_0.vtu (one PVTU piece).",
    )
    ap.add_argument(
        "--stem", type=str, default="mesh_0",
        help="Filename stem (the PVTU will be <stem>.pvtu and the VTU "
             "<stem>_0.vtu). The .pvtu file pattern fed to "
             "run_tracking.py should be '<stem>.pvtu' (no {timestep} "
             "placeholder, since the field is steady).",
    )
    ap.add_argument(
        "--field-name", type=str, default="Displacement",
        help="Name of the per-node vector field in the PVTU. Must match "
             "run_tracking.py's --velocity-field.",
    )
    ap.add_argument(
        "--batch-size", type=int, default=4096,
        help="Per-batch node count for the JAX-vmap evaluation.",
    )
    ap.add_argument(
        "--refinement", action="append", default=None, metavar="R,CX,CY,CZ",
        help="Adaptive refinement zone (repeatable). Each cell whose centroid "
             "lies within radius R of (CX, CY, CZ) is subdivided into 8 child "
             "cells AT THIS PASS. Pass --refinement multiple times to nest "
             "successive levels (e.g. once for level-1 region, again with a "
             "smaller R for level-2 inside it). All four values are floats; "
             "delimit with commas. Example: --refinement 1.5,0,0,0",
    )
    args = ap.parse_args()

    # Parse refinement zones.
    refinements = []
    if args.refinement:
        for spec in args.refinement:
            parts = [float(x) for x in spec.split(",")]
            if len(parts) != 4:
                raise SystemExit(
                    f"--refinement '{spec}': expected 4 comma-separated "
                    f"floats (R,CX,CY,CZ), got {len(parts)}"
                )
            refinements.append((parts[0], (parts[1], parts[2], parts[3])))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    bbox_min = np.asarray(args.bbox[::2], dtype=np.float64)
    bbox_max = np.asarray(args.bbox[1::2], dtype=np.float64)
    n_cells = np.asarray(args.n_cells, dtype=np.int64)

    print("=" * 80)
    print("JAXTrace test-mesh generator")
    print("=" * 80)
    print(f"  Velocity module: {args.velocity_module}")
    print(f"  Domain bbox:     {bbox_min} -> {bbox_max}")
    print(f"  N cells:         {tuple(n_cells)} (= {6*int(n_cells[0])*int(n_cells[1])*int(n_cells[2]):,} tets)")
    print(f"  N nodes:         {(int(n_cells[0])+1)*(int(n_cells[1])+1)*(int(n_cells[2])+1):,}")
    print(f"  Output:          {out_dir}")
    print(f"  Stem:            {args.stem}")
    print(f"  Field name:      {args.field_name}")
    print()

    # Load the user analytic-velocity module to get velocity_fn.
    from jaxtrace.gpu.tracking.velocity_provider import load_analytic_provider
    provider = load_analytic_provider(
        module_path=str(args.velocity_module),
        domain_bbox=(
            (bbox_min[0], bbox_max[0]),
            (bbox_min[1], bbox_max[1]),
            (bbox_min[2], bbox_max[2]),
        ),
        dt=0.0,
    )
    if provider.is_time_dependent:
        raise NotImplementedError(
            "Test-mesh generator only supports steady analytic fields. "
            "Time-dependent fields would need a separate PVTU per timestep "
            "and run_tracking.py --vel-range >0 cyclic loading."
        )
    velocity_fn = provider.velocity_fn

    print(f"[1/3] Building node grid + connectivity...")
    if refinements:
        print(f"  Adaptive refinement, {len(refinements)} pass(es):")
        for k, (r, c) in enumerate(refinements):
            print(f"    pass {k + 1}: radius={r:g}, center={c}")
        nodes, connectivity, cell_levels = build_adaptive_kuhn_mesh(
            bbox_min, bbox_max, n_cells, refinements,
        )
        # Level histogram for diagnostics.
        unique_levels, counts = np.unique(cell_levels, return_counts=True)
        print(f"  cell-level histogram:")
        for lvl, cnt in zip(unique_levels.tolist(), counts.tolist()):
            print(f"    level {lvl}: {cnt:,} cells ({6*cnt:,} tets)")
    else:
        nodes, n_per_axis = build_node_grid(bbox_min, bbox_max, n_cells)
        connectivity = build_kuhn_connectivity(n_cells, n_per_axis)
    print(f"  nodes:        {nodes.shape}")
    print(f"  connectivity: {connectivity.shape}")
    print()

    print(f"[2/3] Evaluating analytic velocity at {nodes.shape[0]:,} nodes...")
    velocities = evaluate_velocity_at_nodes(
        nodes, velocity_fn, batch_size=args.batch_size,
    )
    vmag = np.linalg.norm(velocities, axis=1)
    print(f"  velocity magnitude: min={vmag.min():.4g}, "
          f"mean={vmag.mean():.4g}, max={vmag.max():.4g}")
    print()

    print(f"[3/3] Writing VTU + PVTU...")
    vtu_filename = f"{args.stem}_0.vtu"
    write_vtu(
        out_dir / vtu_filename,
        nodes=nodes,
        connectivity=connectivity,
        field_name=args.field_name,
        field_values=velocities,
    )
    pvtu_filename = f"{args.stem}.pvtu"
    write_pvtu(
        out_dir / pvtu_filename,
        vtu_filename=vtu_filename,
        field_name=args.field_name,
    )
    print(f"  Wrote {out_dir / pvtu_filename}")
    print(f"  Wrote {out_dir / vtu_filename}")
    print()

    # Sanity-check by reading the file back through the production loader.
    print("[verify] reading back through load_velocity_sequence_from_pvtu...")
    from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
    pos_back, conn_back, vel_seq = load_velocity_sequence_from_pvtu(
        base_path=out_dir,
        file_pattern=f"{args.stem}.pvtu",
        timestep_range=(0, 0),
        field_name=args.field_name,
        verbose=False,
    )
    print(f"  read nodes:        {pos_back.shape}  dtype={pos_back.dtype}")
    print(f"  read connectivity: {conn_back.shape}  dtype={conn_back.dtype}")
    print(f"  read field:        {vel_seq.shape}  dtype={vel_seq.dtype}")

    # Round-trip parity: the values we wrote should match what we read.
    pos_err = np.abs(pos_back.astype(np.float64) - nodes).max()
    vel_err = np.abs(vel_seq[0].astype(np.float64) - velocities).max()
    print(f"  max node-position round-trip error: {pos_err:.3e}")
    print(f"  max velocity-field round-trip error: {vel_err:.3e}")
    # The loader downcasts the velocity field to float32 by default
    # (matches the cohort runs); so velocity round-trip error of
    # ~1e-6 is float32 epsilon, not a writer bug. Node positions
    # are kept in float64 and should round-trip bit-exactly.
    if pos_err > 1e-10:
        print("  WARNING: node-position round-trip error larger than expected.")
    if vel_err > 1e-5:
        print("  WARNING: velocity-field round-trip error larger than float32 epsilon.")
    if pos_err <= 1e-10 and vel_err <= 1e-5:
        print("  OK — file round-trips at the loader's float32 precision.")

    print()
    print("Done.")
    print(f"\nFeed this to run_tracking.py like:")
    print(f"  python run_tracking.py \\")
    print(f"    --velocity-source mesh \\")
    print(f"    --mesh-dir {out_dir} \\")
    print(f"    --mesh-pattern '{args.stem}.pvtu' \\")
    print(f"    --vel-range 0 0 \\")
    print(f"    --velocity-field {args.field_name} \\")
    print(f"    ...")


if __name__ == "__main__":
    main()
