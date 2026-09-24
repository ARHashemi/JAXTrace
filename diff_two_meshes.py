#!/usr/bin/env python3
"""
Two-PVTU Mesh Diff: spatial localisation of changed elements.

Loads two PVTU files (typically a small step apart in the cyclic mesh
sequence), classifies each element of mesh B as "static" (present in A
with same node tuple) or "dynamic" (new or remeshed), and reports:

  1. Fraction of dynamic elements
  2. Bounding box of dynamic elements vs. full mesh
  3. Per-refinement-level breakdown of dynamic elements
  4. (Optional) Level-set proximity of dynamic elements
  5. Whether dynamic elements occupy the same parent cubes as static ones
     (for the §10.6 "fixed cells, varying registration" idea)

Usage on LUMI:
    python diff_two_meshes.py \
        --input /scratch/.../C3.gid/post \
        --pattern "C3_{timestep}.pvtu" \
        --step-a 0 --step-b 50 \
        --output diff_a0_b50.txt \
        --levelset-field LevelSet
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


def load_mesh(mesh_file: Path, fields: Optional[list] = None):
    """Load PVTU; return (positions, connectivity, point_data_dict)."""
    import vtk
    from vtk.util import numpy_support

    if not mesh_file.exists():
        raise FileNotFoundError(mesh_file)

    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(str(mesh_file))
    reader.Update()
    output = reader.GetOutput()
    if output is None or output.GetPoints() is None:
        raise RuntimeError(f"VTK failed to read {mesh_file}")

    positions = numpy_support.vtk_to_numpy(
        output.GetPoints().GetData()
    ).astype(np.float64).copy()

    n_cells = output.GetNumberOfCells()
    conn_data = numpy_support.vtk_to_numpy(output.GetCells().GetData())
    connectivity = np.zeros((n_cells, 4), dtype=np.int32)
    for i in range(n_cells):
        connectivity[i] = conn_data[i * 5 + 1: i * 5 + 5]

    point_data = {}
    if fields:
        pd = output.GetPointData()
        for fname in fields:
            if pd.HasArray(fname):
                point_data[fname] = numpy_support.vtk_to_numpy(
                    pd.GetArray(fname)
                ).astype(np.float64).copy()
            else:
                available = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
                print(f"  Warning: field '{fname}' not found. Available: {available}")

    return positions, connectivity, point_data


def element_keys(connectivity: np.ndarray) -> np.ndarray:
    """Return a sorted-tuple key per element (order-invariant)."""
    return np.sort(connectivity, axis=1)


def hash_elements(sorted_conn: np.ndarray) -> np.ndarray:
    """Hash each row of sorted connectivity to a single 64-bit integer."""
    # Combine 4 int32 indices into a 128-bit byte string per row, then hash.
    # For exact set membership we use np.unique with axis=0 instead.
    return sorted_conn  # placeholder; we use np.unique in main


def compute_element_centroids(positions: np.ndarray,
                              connectivity: np.ndarray) -> np.ndarray:
    return positions[connectivity].mean(axis=1)


def compute_element_sizes(positions: np.ndarray,
                          connectivity: np.ndarray) -> np.ndarray:
    """Mean edge length per element."""
    edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    n = connectivity.shape[0]
    s = np.zeros(n)
    for a, b in edge_pairs:
        s += np.linalg.norm(positions[connectivity[:, a]]
                            - positions[connectivity[:, b]], axis=1)
    return s / 6.0


def assign_levels(elem_sizes: np.ndarray, max_size: float) -> np.ndarray:
    """Octave-bin levels: 0=coarsest."""
    ratios = max_size / np.maximum(elem_sizes, 1e-15)
    levels = np.floor(np.log2(ratios)).astype(int)
    return np.clip(levels, 0, 20)


def classify_static_dynamic(conn_a: np.ndarray, conn_b: np.ndarray):
    """
    Return dynamic_mask (length len(conn_b), bool):
      True  = element of mesh B has no exact node-tuple match in mesh A
      False = element of B already exists in A (static)

    Uses 64-bit packing of sorted node tuples so set membership is a fast
    numpy in1d/isin call rather than a Python loop.
    """
    keys_a = element_keys(conn_a).astype(np.uint64)
    keys_b = element_keys(conn_b).astype(np.uint64)

    # Pack 4 indices into a single 256-bit-wide structured array.
    # int32 indices easily fit in 32 bits; pack into one uint128 via
    # two uint64 fields. We use a structured dtype so np.isin works directly.
    # Even simpler: pack (i0,i1,i2,i3) into (i0*P0 + i1*P1 + ...) where
    # Pk are large primes. With 32-bit node indices and N_nodes < 2^31,
    # using 32-bit shifts gives a unique 128-bit int; we approximate with
    # XORed pairs into one uint64. To stay correct, use np.void view.

    def to_void(a):
        a = np.ascontiguousarray(a, dtype=np.int64)
        # 4 int64 -> 32-byte void per row
        return a.view(np.dtype((np.void, a.shape[1] * a.dtype.itemsize))).ravel()

    void_a = to_void(keys_a)
    void_b = to_void(keys_b)
    is_static = np.isin(void_b, void_a)
    return ~is_static


def main():
    p = argparse.ArgumentParser(description="Diff two PVTU meshes for spatial localisation")
    p.add_argument('--input', type=str, required=True,
                   help='Directory containing PVTU files')
    p.add_argument('--pattern', type=str, default='C3_{timestep}.pvtu',
                   help='File pattern with {timestep} placeholder')
    p.add_argument('--step-a', type=int, required=True, help='Reference step')
    p.add_argument('--step-b', type=int, required=True, help='Comparison step')
    p.add_argument('--levelset-field', type=str, default=None,
                   help='Name of level-set field in PVTU (optional)')
    p.add_argument('--output', type=str, default='mesh_diff_report.txt',
                   help='Path to write the textual report')
    p.add_argument('--export-changed-vtu', type=str, default=None,
                   help='If set, write a VTU with element flag (1=dynamic,0=static)')
    args = p.parse_args()

    base = Path(args.input)
    fa = base / args.pattern.replace('{timestep}', str(args.step_a))
    fb = base / args.pattern.replace('{timestep}', str(args.step_b))

    fields = [args.levelset_field] if args.levelset_field else None

    print(f"Loading A: {fa}")
    t0 = time.time()
    pos_a, conn_a, _ = load_mesh(fa)
    print(f"  N_a: nodes={pos_a.shape[0]:,}  elements={conn_a.shape[0]:,}  ({time.time()-t0:.1f}s)")

    print(f"Loading B: {fb}")
    t0 = time.time()
    pos_b, conn_b, pd_b = load_mesh(fb, fields=fields)
    print(f"  N_b: nodes={pos_b.shape[0]:,}  elements={conn_b.shape[0]:,}  ({time.time()-t0:.1f}s)")

    print()
    print("Classifying dynamic vs static elements...")
    t0 = time.time()
    dynamic_mask = classify_static_dynamic(conn_a, conn_b)
    n_dyn = int(dynamic_mask.sum())
    n_tot = conn_b.shape[0]
    print(f"  Dynamic elements: {n_dyn:,} / {n_tot:,} ({100*n_dyn/n_tot:.3f}%)  ({time.time()-t0:.1f}s)")

    if n_dyn == 0:
        print("\nNo dynamic elements. Mesh B is a permutation/subset of A.")
        # Still write a report.

    # Centroids and sizes for spatial analysis (mesh B).
    centroids_b = compute_element_centroids(pos_b, conn_b)
    sizes_b = compute_element_sizes(pos_b, conn_b)
    max_size = float(sizes_b.max()) if sizes_b.size else 0.0
    levels_b = assign_levels(sizes_b, max_size) if max_size > 0 else np.zeros(n_tot, dtype=int)

    # Bounding boxes
    bbox_full_min = pos_b.min(axis=0)
    bbox_full_max = pos_b.max(axis=0)
    bbox_full_extent = bbox_full_max - bbox_full_min

    if n_dyn > 0:
        c_dyn = centroids_b[dynamic_mask]
        bbox_dyn_min = c_dyn.min(axis=0)
        bbox_dyn_max = c_dyn.max(axis=0)
        bbox_dyn_extent = bbox_dyn_max - bbox_dyn_min
        frac_volume = float(np.prod(bbox_dyn_extent) / np.prod(bbox_full_extent))
    else:
        bbox_dyn_min = bbox_dyn_max = bbox_dyn_extent = np.zeros(3)
        frac_volume = 0.0

    # Level breakdown
    unique_levels, level_counts_all = np.unique(levels_b, return_counts=True)
    level_counts_dyn = np.zeros_like(level_counts_all)
    for i, lv in enumerate(unique_levels):
        level_counts_dyn[i] = int(np.sum(dynamic_mask & (levels_b == lv)))

    # Level-set proximity (optional)
    levelset_summary = None
    if args.levelset_field and args.levelset_field in pd_b:
        ls = pd_b[args.levelset_field]
        # Map node-based level-set to element by mean.
        ls_elem = ls[conn_b].mean(axis=1)
        if n_dyn > 0:
            ls_dyn = ls_elem[dynamic_mask]
            levelset_summary = {
                'dyn_min': float(np.min(np.abs(ls_dyn))),
                'dyn_max': float(np.max(np.abs(ls_dyn))),
                'dyn_mean_abs': float(np.mean(np.abs(ls_dyn))),
                'dyn_median_abs': float(np.median(np.abs(ls_dyn))),
                'all_max_abs': float(np.max(np.abs(ls_elem))),
                'frac_dyn_within_band': {
                    f'{eps:.1e}': float(np.mean(np.abs(ls_dyn) < eps))
                    for eps in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
                },
            }

    # "Fixed cells, varying registration" check (§10.6):
    # Use a coarse uniform grid sized to the smallest level cell to test whether
    # dynamic centroids fall in cells already occupied by static centroids.
    cell_size = float(np.min(sizes_b))
    # Coarsen to avoid grid blow-up: aim for ~256^3 cells.
    target_cells_per_axis = 256
    cell_size = max(cell_size, max(bbox_full_extent) / target_cells_per_axis)

    def to_cell_id(centroids):
        idx = np.floor((centroids - bbox_full_min) / cell_size).astype(np.int64)
        idx = np.clip(idx, 0, target_cells_per_axis * 4)
        return (idx[:, 0] * (1 << 22) + idx[:, 1] * (1 << 11) + idx[:, 2])

    static_mask_b = ~dynamic_mask
    cell_ids_static = to_cell_id(centroids_b[static_mask_b])
    if n_dyn > 0:
        cell_ids_dyn = to_cell_id(centroids_b[dynamic_mask])
        in_existing = np.isin(cell_ids_dyn, cell_ids_static)
        n_dyn_in_existing_cells = int(in_existing.sum())
        frac_dyn_in_existing = n_dyn_in_existing_cells / n_dyn
    else:
        n_dyn_in_existing_cells = 0
        frac_dyn_in_existing = 0.0

    # ── Report ──
    lines = []
    lines.append("=" * 80)
    lines.append("MESH DIFF REPORT")
    lines.append("=" * 80)
    lines.append(f"A: {fa}")
    lines.append(f"B: {fb}")
    lines.append("")
    lines.append(f"Mesh A: nodes={pos_a.shape[0]:,}  elements={conn_a.shape[0]:,}")
    lines.append(f"Mesh B: nodes={pos_b.shape[0]:,}  elements={conn_b.shape[0]:,}")
    lines.append(f"Element count delta: {conn_b.shape[0] - conn_a.shape[0]:+,}")
    lines.append("")
    lines.append("--- DYNAMIC ELEMENTS ---")
    lines.append(f"Count: {n_dyn:,} / {n_tot:,} ({100*n_dyn/n_tot:.3f}%)")
    lines.append("")
    lines.append("--- SPATIAL LOCALISATION ---")
    lines.append(f"Full mesh bbox extent:    [{bbox_full_extent[0]:.4e}, {bbox_full_extent[1]:.4e}, {bbox_full_extent[2]:.4e}]")
    lines.append(f"Dynamic centroids extent: [{bbox_dyn_extent[0]:.4e}, {bbox_dyn_extent[1]:.4e}, {bbox_dyn_extent[2]:.4e}]")
    lines.append(f"Volume fraction (bbox/bbox): {frac_volume:.4e}")
    lines.append(f"Dynamic bbox: [{bbox_dyn_min[0]:.4e}, {bbox_dyn_min[1]:.4e}, {bbox_dyn_min[2]:.4e}]")
    lines.append(f"           -> [{bbox_dyn_max[0]:.4e}, {bbox_dyn_max[1]:.4e}, {bbox_dyn_max[2]:.4e}]")
    lines.append("")
    lines.append("--- REFINEMENT-LEVEL DISTRIBUTION ---")
    lines.append(f"{'Level':>6}  {'#all':>12}  {'#dynamic':>12}  {'%dynamic':>10}  {'%of_dyn':>10}")
    for i, lv in enumerate(unique_levels):
        all_ct = int(level_counts_all[i])
        dyn_ct = int(level_counts_dyn[i])
        pct_dyn = 100 * dyn_ct / all_ct if all_ct else 0.0
        pct_of_dyn = 100 * dyn_ct / n_dyn if n_dyn else 0.0
        lines.append(f"{lv:>6}  {all_ct:>12,}  {dyn_ct:>12,}  {pct_dyn:>9.3f}%  {pct_of_dyn:>9.3f}%")
    lines.append("")
    if levelset_summary:
        lines.append("--- LEVEL-SET PROXIMITY (dynamic elements) ---")
        lines.append(f"Min |φ|:    {levelset_summary['dyn_min']:.4e}")
        lines.append(f"Max |φ|:    {levelset_summary['dyn_max']:.4e}")
        lines.append(f"Mean |φ|:   {levelset_summary['dyn_mean_abs']:.4e}")
        lines.append(f"Median |φ|: {levelset_summary['dyn_median_abs']:.4e}")
        lines.append(f"Max |φ| over full mesh: {levelset_summary['all_max_abs']:.4e}")
        lines.append("Fraction of dynamic elements with |φ| < band:")
        for eps, frac in levelset_summary['frac_dyn_within_band'].items():
            lines.append(f"  band={eps}: {100*frac:.2f}%")
        lines.append("")
    lines.append("--- FIXED-CELL FEASIBILITY (§10.6) ---")
    lines.append(f"Probe cell size (cube edge): {cell_size:.4e}")
    lines.append(f"Dynamic elements landing in cells already containing")
    lines.append(f"static elements: {n_dyn_in_existing_cells:,} / {n_dyn:,} ({100*frac_dyn_in_existing:.3f}%)")
    if frac_dyn_in_existing > 0.95:
        lines.append("=> Strong evidence that parent cubes can stay fixed;")
        lines.append("   only the cell-to-element registration varies.")
    elif frac_dyn_in_existing > 0.5:
        lines.append("=> Partial overlap. A small number of new cells appear.")
    else:
        lines.append("=> Many dynamic elements occupy new cells; fixed-cell")
        lines.append("   approach would need cell additions per cycle step.")
    lines.append("")
    lines.append("=" * 80)

    report = "\n".join(lines)
    print()
    print(report)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n")
    print(f"\nReport written to: {out_path}")

    # Optional: export VTU with dynamic flag for visualisation.
    if args.export_changed_vtu and n_dyn > 0:
        try:
            import vtk
            from vtk.util import numpy_support
            ug = vtk.vtkUnstructuredGrid()
            pts = vtk.vtkPoints()
            pts.SetData(numpy_support.numpy_to_vtk(pos_b))
            ug.SetPoints(pts)
            ids = vtk.vtkIdTypeArray()
            ids.SetNumberOfComponents(1)
            cells = vtk.vtkCellArray()
            for i in range(n_tot):
                cells.InsertNextCell(4)
                for j in range(4):
                    cells.InsertCellPoint(int(conn_b[i, j]))
            ug.SetCells(vtk.VTK_TETRA, cells)
            arr = numpy_support.numpy_to_vtk(dynamic_mask.astype(np.int32))
            arr.SetName("dynamic")
            ug.GetCellData().AddArray(arr)
            arr2 = numpy_support.numpy_to_vtk(levels_b.astype(np.int32))
            arr2.SetName("level")
            ug.GetCellData().AddArray(arr2)
            w = vtk.vtkXMLUnstructuredGridWriter()
            w.SetFileName(args.export_changed_vtu)
            w.SetInputData(ug)
            w.Write()
            print(f"VTU written to: {args.export_changed_vtu}")
        except Exception as e:
            print(f"VTU export failed: {e}")


if __name__ == '__main__':
    main()
