"""
Spatial residual breakdown for a chosen ROM reconstruction.

Loads FOM velocity at (case, ts), computes the ROM reconstruction with
the requested formula, bins nodes by radial distance from the tool
axis and by height z, and reports the RMS residual per bin.  Also
writes a VTU with the residual as a nodal field so the spatial
pattern can be inspected in ParaView.

Referenced from docs/rom_reconstruction_findings.md.

Usage:
  python3 tests/rom/rom_spatial_residual.py --case 0 --ts 119 --out case00_ts119.vtu

Reads only /scratch/shared/ROM/FOM/.  Writes a single .vtu file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from jaxtrace.rom.velocity_recon import (
    load_basis, load_coefficients, reconstruct,
)
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu


DEFAULT_ROM_ROOT = Path("/scratch/shared/ROM/FOM")


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt((x ** 2).mean()))


def _bin_report(edges: np.ndarray, coord: np.ndarray, coord_name: str,
                resid_mag: np.ndarray, fom_mag: np.ndarray) -> None:
    print(f'  {"bin":>4}  {coord_name + "_lo":>8}  {coord_name + "_hi":>8}  '
          f'{"count":>7}  {"|resid|_rms":>12}  {"|fom|_rms":>12}  {"rel":>7}')
    for i in range(len(edges) - 1):
        m = (coord >= edges[i]) & (coord < edges[i + 1])
        if not m.any():
            continue
        rr = _rms(resid_mag[m])
        ff = _rms(fom_mag[m])
        rel = 100 * rr / ff if ff > 0 else float('nan')
        print(f'  {i:>4}  {edges[i]:>8.3f}  {edges[i+1]:>8.3f}  '
              f'{int(m.sum()):>7}  {rr:>12.4e}  {ff:>12.4e}  {rel:>6.2f}%')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rom-root", type=Path, default=DEFAULT_ROM_ROOT)
    ap.add_argument("--case", type=int, required=True)
    ap.add_argument("--ts", type=int, default=119)
    ap.add_argument("--formula", type=str, default="centered")
    ap.add_argument("--out", type=Path, default=None,
                    help="If set, write a VTU containing the residual field")
    ap.add_argument("--n-bins", type=int, default=10,
                    help="Number of bins in each of the r/z dimensions")
    args = ap.parse_args()

    basis = load_basis(args.rom_root / "cylindrical.som.fswrom.basis",
                       verbose=False)
    coeffs = load_coefficients(
        args.rom_root / "cylindrical.som.fswrom.romdata", verbose=False,
    )

    case_dir = args.rom_root / f"cylindrical_{args.case:03d}.gid" / "post"
    nodes, conn, vs = load_velocity_sequence_from_pvtu(
        base_path=case_dir,
        file_pattern="cylindrical_{timestep}.pvtu",
        timestep_range=(args.ts, args.ts),
        field_name="Displacement",
        verbose=False,
    )
    fom = vs[0].astype(np.float64)

    v_recon = reconstruct(basis, coeffs, args.case, args.formula)
    v_mean = basis.mean
    resid_recon = v_recon - fom
    resid_mean = v_mean - fom

    print(f'case {args.case:02d}, ts={args.ts}, formula={args.formula}: '
          f'nodes={nodes.shape}, elements={conn.shape}')
    fom_rms = _rms(fom)
    print(f'  |FOM|_rms   : {fom_rms:.4e}')
    print(f'  mean_rel    : {100 * _rms(resid_mean) / fom_rms:.2f}%')
    print(f'  {args.formula}_rel: {100 * _rms(resid_recon) / fom_rms:.2f}%')

    r = np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)
    z = nodes[:, 2]
    r_edges = np.linspace(r.min(), r.max(), args.n_bins + 1)
    z_edges = np.linspace(z.min(), z.max(), args.n_bins + 1)
    fom_mag = np.linalg.norm(fom, axis=1)
    resid_mag = np.linalg.norm(resid_recon, axis=1)

    print(f'\n  Node bbox: x={nodes[:,0].min():.3f}..{nodes[:,0].max():.3f}  '
          f'y={nodes[:,1].min():.3f}..{nodes[:,1].max():.3f}  '
          f'z={nodes[:,2].min():.3f}..{nodes[:,2].max():.3f}')
    print(f'  Node r range: [{r.min():.3f}, {r.max():.3f}]')

    print(f'\n  Radial breakdown ({args.formula} residual):')
    _bin_report(r_edges, r, 'r', resid_mag, fom_mag)
    print(f'\n  Vertical breakdown ({args.formula} residual):')
    _bin_report(z_edges, z, 'z', resid_mag, fom_mag)

    if args.out is None:
        return 0

    # VTU export
    try:
        import vtk
        from vtk.util import numpy_support as vns
    except ImportError:
        print(f'  (skip VTU export: vtk import failed)', file=sys.stderr)
        return 0

    ug = vtk.vtkUnstructuredGrid()
    pts = vtk.vtkPoints()
    pts.SetData(vns.numpy_to_vtk(nodes.astype(np.float32), deep=True))
    ug.SetPoints(pts)

    n_elem = conn.shape[0]
    cell_types = vtk.vtkUnsignedCharArray()
    cell_types.SetNumberOfValues(n_elem)
    conn32 = np.hstack(
        [np.full((n_elem, 1), 4, dtype=np.int32), conn.astype(np.int32)]
    ).ravel()
    ca = vtk.vtkCellArray()
    id_arr = vns.numpy_to_vtkIdTypeArray(conn32.astype(np.int64), deep=True)
    ca.SetCells(n_elem, id_arr)
    for i in range(n_elem):
        cell_types.SetValue(i, vtk.VTK_TETRA)
    ug.SetCells(cell_types, ca)

    def _add(name, data):
        arr = vns.numpy_to_vtk(np.ascontiguousarray(data.astype(np.float32)),
                               deep=True)
        arr.SetName(name)
        ug.GetPointData().AddArray(arr)

    _add('fom', fom)
    _add('mean', v_mean)
    _add(args.formula, v_recon)
    _add(f'resid_{args.formula}', resid_recon)
    _add('fom_mag', fom_mag)
    _add(f'resid_{args.formula}_mag', resid_mag)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    w = vtk.vtkXMLUnstructuredGridWriter()
    w.SetFileName(str(args.out))
    w.SetInputData(ug)
    w.SetDataModeToBinary()
    w.Write()
    print(f'\n  Wrote {args.out}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
