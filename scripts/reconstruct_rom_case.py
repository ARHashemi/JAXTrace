"""
Reconstruct the FSW-ROM velocity field on the original FOM mesh for a
single case, and write it out as a PVTU that looks identical (same mesh,
same field name 'Displacement') to a FOM snapshot.  The output PVTU can
be handed straight to run_tracking.py --velocity-source mesh with no
other changes: from JAXTrace's point of view it IS a mesh-loaded field.

Purpose is to answer the question "how well does the ROM-reconstructed
velocity carry particles vs the full-order velocity, holding everything
else fixed?" without any GPU-side plumbing changes.

Layout produced (one per case, one timestep):

    <out-root>/<case>.gid/post/
        cylindrical_0.pvtu
        cylindrical_0_0.vtu

The mesh + connectivity are copied from the case's own FOM PVTU at
--source-timestep (defaults to the last step, 119).  Any other
node/cell data present in the source PVTU (LEVEL, Temperature, etc.)
is passed through unchanged so the tracking pipeline's level-set +
temperature paths still work.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from jaxtrace.rom.velocity_recon import (
    load_basis, load_coefficients, reconstruct,
)


def _read_pvtu(pvtu_path: Path):
    """Return (unstructured_grid, source_paths_relative_to_pvtu)."""
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(str(pvtu_path))
    reader.Update()
    return reader.GetOutput()


def _write_single_piece_pvtu(
    out_pvtu: Path,
    out_vtu: Path,
    ug: vtk.vtkUnstructuredGrid,
) -> None:
    """Write ug as one .vtu piece and emit a minimal .pvtu index pointing
    at it.  Field names/types are copied from ug's point/cell data, so
    the pvtu header is faithful to what a downstream reader will see."""

    # Write the piece itself
    piece_writer = vtk.vtkXMLUnstructuredGridWriter()
    piece_writer.SetFileName(str(out_vtu))
    piece_writer.SetInputData(ug)
    piece_writer.SetDataModeToBinary()
    piece_writer.SetCompressorTypeToZLib()
    piece_writer.Write()

    # Build the .pvtu index by hand.  vtkXMLPUnstructuredGridWriter
    # insists on being MPI-run to produce a valid multi-piece pvtu; a
    # single-piece pvtu is trivially small and lets us stay serial.
    pd = ug.GetPointData()
    cd = ug.GetCellData()

    def _piece_arrays(source_pd) -> str:
        out = []
        for i in range(source_pd.GetNumberOfArrays()):
            arr = source_pd.GetArray(i)
            if arr is None:
                continue
            name = arr.GetName()
            ncomp = arr.GetNumberOfComponents()
            dt = arr.GetDataTypeAsString().capitalize()
            # VTK datatype strings like 'float' or 'double' -> Float32 / Float64
            dtype_map = {
                'Float': 'Float32', 'Double': 'Float64',
                'Int': 'Int32', 'Long': 'Int64',
                'Unsigned char': 'UInt8', 'Unsigned int': 'UInt32',
            }
            dt = dtype_map.get(dt, dt)
            if ncomp > 1:
                out.append(
                    f'      <PDataArray type="{dt}" Name="{name}" '
                    f'NumberOfComponents="{ncomp}"/>'
                )
            else:
                out.append(
                    f'      <PDataArray type="{dt}" Name="{name}"/>'
                )
        return "\n".join(out)

    pt_block = _piece_arrays(pd)
    cd_block = _piece_arrays(cd)

    pvtu_text = f"""<?xml version="1.0"?>
<VTKFile type="PUnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <PUnstructuredGrid GhostLevel="0">
    <PPointData>
{pt_block}
    </PPointData>
    <PCellData>
{cd_block}
    </PCellData>
    <PPoints>
      <PDataArray type="Float32" Name="Points" NumberOfComponents="3"/>
    </PPoints>
    <Piece Source="{out_vtu.name}"/>
  </PUnstructuredGrid>
</VTKFile>
"""
    out_pvtu.write_text(pvtu_text)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--fom-root", type=Path, required=True,
        help="Root that contains the case folders (e.g. "
             "/scratch/shared/ROM/FOM)",
    )
    ap.add_argument(
        "--case", type=int, required=True,
        help="0-based case index",
    )
    ap.add_argument(
        "--case-prefix", type=str, default="cylindrical",
        help="Case-folder prefix.  Folder = <case-prefix>_<case:03d>.gid.",
    )
    ap.add_argument(
        "--source-timestep", type=int, default=119,
        help="Timestep index in the case's FOM PVTU sequence to use as "
             "the template (its mesh, connectivity, and any non-velocity "
             "point/cell data are carried into the ROM PVTU).",
    )
    ap.add_argument(
        "--basis", type=Path, required=True,
        help="Path to <case>.som.fswrom.basis",
    )
    ap.add_argument(
        "--romdata", type=Path, required=True,
        help="Path to <case>.som.fswrom.romdata",
    )
    ap.add_argument(
        "--formula", type=str, default="centered",
        choices=("centered", "sigma_c", "c_over_sig",
                 "no_mean", "no_mean_sig"),
        help="Reconstruction formula.  'centered' matches the FEMUSS "
             "SLEPcExternalFilter + SnapshotsMean convention (colleague "
             "spec).  See jaxtrace/rom/velocity_recon.py.",
    )
    ap.add_argument(
        "--field-group", type=str, default="Displacement",
        help="HDF5 group inside the basis/romdata files (must match the "
             "PVTU field name).",
    )
    ap.add_argument(
        "--out-root", type=Path, required=True,
        help="Where to write the reconstructed case folder.  The full "
             "output path will be <out-root>/<case-prefix>_<idx>.gid/post/"
             "cylindrical_0.pvtu.",
    )
    ap.add_argument(
        "--out-timestep", type=int, default=0,
        help="Timestep index to use in the OUTPUT filename.  Defaults to "
             "0 so run_tracking.py with --vel-range 0 0 picks it up.",
    )
    ap.add_argument(
        "--zero-out-fields", type=str, nargs="*", default=None,
        help="Names of extra PointData / CellData arrays to overwrite "
             "with zeros in the output (useful if you want to strip "
             "e.g. Pressure so it is obvious the ROM is not filling "
             "them).  Default: keep everything unchanged.",
    )
    args = ap.parse_args()

    # ----- Load ROM basis and coefficients -----
    print(f"[reconstruct] loading basis   : {args.basis}")
    basis = load_basis(args.basis, field_group=args.field_group, verbose=False)
    print(f"              n_nodes={basis.n_nodes:,}, n_modes={basis.n_modes}")

    print(f"[reconstruct] loading romdata : {args.romdata}")
    coeffs = load_coefficients(
        args.romdata, field_group=args.field_group, verbose=False,
    )
    print(f"              n_cases={coeffs.n_cases}, n_modes={coeffs.n_modes}")

    if not (0 <= args.case < coeffs.n_cases):
        print(f"[reconstruct] ERROR: --case {args.case} outside [0, "
              f"{coeffs.n_cases})", file=sys.stderr)
        return 3

    c = coeffs.coefficients[:, args.case]
    print(f"[reconstruct] case {args.case:02d} coefficients (top-3): "
          f"{c[:3]}")

    # ----- Reconstruct velocity at every mesh node -----
    v_recon = reconstruct(basis, coeffs, args.case, formula=args.formula)
    print(f"[reconstruct] formula='{args.formula}': "
          f"|v|_max={np.abs(v_recon).max():.4e}, "
          f"|v|_rms={float(np.sqrt((v_recon**2).mean())):.4e}")

    # ----- Load the source PVTU as the template mesh -----
    case_stem = f"{args.case_prefix}_{args.case:03d}"
    source_pvtu = (args.fom_root / f"{case_stem}.gid" / "post"
                   / f"{args.case_prefix}_{args.source_timestep}.pvtu")
    if not source_pvtu.exists():
        print(f"[reconstruct] ERROR: source template PVTU not found: "
              f"{source_pvtu}", file=sys.stderr)
        return 4
    print(f"[reconstruct] loading template: {source_pvtu}")
    ug = _read_pvtu(source_pvtu)
    n_source_nodes = ug.GetNumberOfPoints()
    if n_source_nodes != basis.n_nodes:
        print(f"[reconstruct] ERROR: node count mismatch: template has "
              f"{n_source_nodes:,}, basis has {basis.n_nodes:,}",
              file=sys.stderr)
        return 5

    # ----- Overwrite the Displacement field with the ROM reconstruction -----
    pd = ug.GetPointData()

    def _find_array_index(data_object, name: str) -> int:
        for i in range(data_object.GetNumberOfArrays()):
            if data_object.GetArrayName(i) == name:
                return i
        return -1

    idx = _find_array_index(pd, args.field_group)
    if idx < 0:
        print(f"[reconstruct] ERROR: template PVTU has no PointData array "
              f"named '{args.field_group}'.  Available: "
              f"{[pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]}",
              file=sys.stderr)
        return 6
    old_arr = pd.GetArray(idx)
    old_name = old_arr.GetName()
    old_ncomp = old_arr.GetNumberOfComponents()
    if old_ncomp != 3:
        print(f"[reconstruct] ERROR: template '{args.field_group}' has "
              f"{old_ncomp} components, expected 3", file=sys.stderr)
        return 7
    print(f"[reconstruct] replacing PointData['{args.field_group}'] with "
          f"ROM reconstruction ({v_recon.shape})")
    new_arr = numpy_to_vtk(np.ascontiguousarray(v_recon), deep=True)
    new_arr.SetName(old_name)
    pd.RemoveArray(idx)
    pd.AddArray(new_arr)

    # Optionally zero out other fields the user doesn't want to carry
    if args.zero_out_fields:
        for fname in args.zero_out_fields:
            i = _find_array_index(pd, fname)
            if i < 0:
                i = _find_array_index(ug.GetCellData(), fname)
                if i < 0:
                    print(f"[reconstruct] --zero-out-fields '{fname}': "
                          "not found in template", file=sys.stderr)
                    continue
                arr = ug.GetCellData().GetArray(i)
                np_arr = vtk_to_numpy(arr)
                np_arr[:] = 0.0
                # Force VTK to see the write
                arr.Modified()
            else:
                arr = pd.GetArray(i)
                np_arr = vtk_to_numpy(arr)
                np_arr[:] = 0.0
                arr.Modified()
            print(f"[reconstruct] zeroed field '{fname}'")

    # ----- Write out to <out-root>/<case>.gid/post/ -----
    out_case_dir = args.out_root / f"{case_stem}.gid" / "post"
    out_case_dir.mkdir(parents=True, exist_ok=True)
    out_pvtu = out_case_dir / f"{args.case_prefix}_{args.out_timestep}.pvtu"
    out_vtu = out_case_dir / f"{args.case_prefix}_{args.out_timestep}_0.vtu"
    print(f"[reconstruct] writing         : {out_pvtu}")
    _write_single_piece_pvtu(out_pvtu, out_vtu, ug)

    # Simple manifest so downstream tooling can find provenance
    (out_case_dir / "ROM_MANIFEST.txt").write_text(
        f"case_prefix   : {args.case_prefix}\n"
        f"case_idx      : {args.case}\n"
        f"formula       : {args.formula}\n"
        f"basis         : {args.basis}\n"
        f"romdata       : {args.romdata}\n"
        f"template_pvtu : {source_pvtu}\n"
        f"out_pvtu      : {out_pvtu}\n"
        f"out_timestep  : {args.out_timestep}\n"
        f"n_nodes       : {basis.n_nodes}\n"
        f"n_modes       : {basis.n_modes}\n"
        f"coefficients  : {list(map(float, c[:3]))}\n"
        f"|v|_max       : {float(np.abs(v_recon).max())}\n"
        f"|v|_rms       : {float(np.sqrt((v_recon**2).mean()))}\n"
    )
    print(f"[reconstruct] wrote manifest  : {out_case_dir / 'ROM_MANIFEST.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
