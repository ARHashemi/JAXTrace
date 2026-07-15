"""
Compare the final particle positions from a ROM-tracking run against
the full-order (FOM) tracking run for the same case.

Both runs are expected to have been produced by JAXTrace with identical
seeding (same FEMUSS start), identical N_STEPS, and identical DT.  The
only difference is the velocity field the RK4 integrator saw: FOM vs
ROM-reconstructed.

Reports:
  * Per-particle displacement between the two final positions.
  * Absolute + relative RMS displacement.
  * Fraction of particles whose FOM path escaped (element_id < 0) but
    whose ROM path stayed alive (or vice versa).
  * A per-particle displacement field written back as a VTU so you can
    inspect where in the domain the ROM trajectory diverges most.

Currently assumes VTKHDF export (the default in run_jaxtrace.sh /
run_jaxtrace_rom.sh).  Add a --format vtu switch later if the paper
ends up using the per-step VTU layout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _load_last_positions_vtkhdf(vtkhdf_path: Path):
    """Read the LAST timestep of a JAXTrace VTKHDF particle archive.

    Actual archive layout (inspected on a real particles.vtkhdf):

        /VTKHDF/
            @Type   = 'PolyData'
            @Version
            NumberOfPoints  : (n_steps,) int64  -- point count per step
            Points          : (sum_counts, 3) float32  -- flat concat
            PointData/
                Escaped         : (sum_counts,) uint8
                Group           : (sum_counts,) uint8
                ParticleID      : (sum_counts,) int32
                Temperature     : (sum_counts,) float32
                MaxTemperature  : (sum_counts,) float32
                (ElementID      : optional; only if --export-element-ids was set)
            Steps/
                @NSteps
                PointOffsets            : (n_steps,) int64  -- start index in Points
                PointDataOffsets/<name> : (n_steps,) int64  -- start index for that array
                ...

    Returns
    -------
    positions : (n_particles, 3) float32
    element_ids : (n_particles,) int32   (all -1 if not exported)
    escaped : (n_particles,) uint8       (all 0 if not exported)
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required to read VTKHDF particle archives"
        ) from exc

    with h5py.File(str(vtkhdf_path), "r") as f:
        vtkhdf = f["VTKHDF"]
        n_pts_per_step = vtkhdf["NumberOfPoints"][:]
        pts_offsets = vtkhdf["Steps"]["PointOffsets"][:]
        last_start = int(pts_offsets[-1])
        last_count = int(n_pts_per_step[-1])
        positions = np.asarray(
            vtkhdf["Points"][last_start:last_start + last_count],
            dtype=np.float32,
        )

        eid = np.full(last_count, -1, dtype=np.int32)
        if ("PointData" in vtkhdf and "ElementID" in vtkhdf["PointData"]):
            eid_offsets = vtkhdf["Steps"]["PointDataOffsets"]["ElementID"][:]
            eid_start = int(eid_offsets[-1])
            eid = np.asarray(
                vtkhdf["PointData"]["ElementID"][eid_start:eid_start + last_count],
                dtype=np.int32,
            )

        escaped = np.zeros(last_count, dtype=np.uint8)
        if ("PointData" in vtkhdf and "Escaped" in vtkhdf["PointData"]):
            esc_offsets = vtkhdf["Steps"]["PointDataOffsets"]["Escaped"][:]
            esc_start = int(esc_offsets[-1])
            escaped = np.asarray(
                vtkhdf["PointData"]["Escaped"][esc_start:esc_start + last_count],
                dtype=np.uint8,
            )

    return positions, eid, escaped


def _write_vtu(out_path: Path, positions: np.ndarray, arrays: dict) -> None:
    """Write a Points-only VTU with the given per-particle arrays."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    n = positions.shape[0]
    pd = vtk.vtkPolyData()
    p = vtk.vtkPoints()
    p.SetData(numpy_to_vtk(positions.astype(np.float32), deep=True))
    pd.SetPoints(p)

    # Add vertex cells so ParaView renders them
    vc = vtk.vtkCellArray()
    for i in range(n):
        vc.InsertNextCell(1)
        vc.InsertCellPoint(i)
    pd.SetVerts(vc)

    for name, arr in arrays.items():
        va = numpy_to_vtk(np.ascontiguousarray(arr.astype(np.float32)), deep=True)
        va.SetName(name)
        pd.GetPointData().AddArray(va)

    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(str(out_path))
    w.SetInputData(pd)
    w.SetDataModeToBinary()
    w.Write()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare final particle positions: FOM vs ROM tracking.",
    )
    ap.add_argument("--fom-vtkhdf", type=Path, required=True,
                    help="Path to the FOM tracking's particles.vtkhdf")
    ap.add_argument("--rom-vtkhdf", type=Path, required=True,
                    help="Path to the ROM tracking's particles.vtkhdf")
    ap.add_argument("--out-vtu", type=Path, default=None,
                    help="Optional: write a VTU containing the FOM final "
                         "positions plus per-particle displacement vector "
                         "and magnitude (open in ParaView to see where "
                         "the ROM trajectory diverges most).")
    args = ap.parse_args()

    if not args.fom_vtkhdf.exists():
        print(f"ERROR: FOM archive not found: {args.fom_vtkhdf}", file=sys.stderr)
        return 3
    if not args.rom_vtkhdf.exists():
        print(f"ERROR: ROM archive not found: {args.rom_vtkhdf}", file=sys.stderr)
        return 3

    print(f"[compare] FOM: {args.fom_vtkhdf}")
    fom_pos, fom_eid, fom_escaped = _load_last_positions_vtkhdf(args.fom_vtkhdf)
    print(f"[compare] ROM: {args.rom_vtkhdf}")
    rom_pos, rom_eid, rom_escaped = _load_last_positions_vtkhdf(args.rom_vtkhdf)

    if fom_pos.shape[0] != rom_pos.shape[0]:
        print(f"[compare] WARNING: particle counts differ: "
              f"FOM={fom_pos.shape[0]:,}, ROM={rom_pos.shape[0]:,}. "
              f"Comparing the min({fom_pos.shape[0]}, {rom_pos.shape[0]}) "
              f"leading particles.  This suggests the two runs did not "
              f"seed identically — check the seeding config (SEED_SOURCE, "
              f"SEED_FRACTION, SEED_GRID, FEMUSS_START, inlet cropping).",
              file=sys.stderr)
        n = min(fom_pos.shape[0], rom_pos.shape[0])
        fom_pos = fom_pos[:n]; rom_pos = rom_pos[:n]
        fom_eid = fom_eid[:n]; rom_eid = rom_eid[:n]
        fom_escaped = fom_escaped[:n]; rom_escaped = rom_escaped[:n]

    n_total = fom_pos.shape[0]
    disp = rom_pos - fom_pos
    disp_mag = np.linalg.norm(disp, axis=1)
    fom_span = float(np.linalg.norm(fom_pos.max(axis=0) - fom_pos.min(axis=0)))

    # Escape-flag agreement
    fom_alive = (fom_escaped == 0)
    rom_alive = (rom_escaped == 0)
    both_alive   = fom_alive & rom_alive
    both_escaped = (~fom_alive) & (~rom_alive)
    only_fom_esc = (~fom_alive) & rom_alive
    only_rom_esc = fom_alive & (~rom_alive)

    print()
    print(f"[compare] particles          : {n_total:,}")
    print(f"[compare] FOM bbox           : [{fom_pos.min(0)}] -> [{fom_pos.max(0)}]")
    print(f"[compare] FOM diagonal       : {fom_span:.4e}")
    print()
    print(f"[compare] escape-flag agreement:")
    print(f"          both alive             : {int(both_alive.sum()):>10,}  ({100*both_alive.mean():5.2f}%)")
    print(f"          both escaped           : {int(both_escaped.sum()):>10,}  ({100*both_escaped.mean():5.2f}%)")
    print(f"          only FOM escaped       : {int(only_fom_esc.sum()):>10,}  ({100*only_fom_esc.mean():5.2f}%)")
    print(f"          only ROM escaped       : {int(only_rom_esc.sum()):>10,}  ({100*only_rom_esc.mean():5.2f}%)")

    def _disp_stats(mask, label):
        if not mask.any():
            print(f"[compare] displacement ({label}): no particles in subset")
            return
        dm = disp_mag[mask]
        d  = disp[mask]
        print(f"[compare] displacement ({label}, N={int(mask.sum()):,}):")
        print(f"          mean               : {float(dm.mean()):.4e}")
        print(f"          median             : {float(np.median(dm)):.4e}")
        print(f"          rms                : {float(np.sqrt((dm**2).mean())):.4e}")
        print(f"          p95                : {float(np.percentile(dm, 95)):.4e}")
        print(f"          p99                : {float(np.percentile(dm, 99)):.4e}")
        print(f"          max                : {float(dm.max()):.4e}")
        print(f"          rms / FOM diagonal : {100 * float(np.sqrt((dm**2).mean())) / max(fom_span, 1e-30):.3f}%")
        for j, comp in enumerate("xyz"):
            print(f"          rms_{comp}              : {float(np.sqrt((d[:, j]**2).mean())):.4e}")

    print()
    _disp_stats(np.ones(n_total, dtype=bool), "all particles")
    print()
    _disp_stats(both_alive, "both-alive subset (fair comparison)")

    # Optional VTU dump
    if args.out_vtu:
        arrays = {
            "displacement_vec": disp,
            "displacement_mag": disp_mag,
            "fom_element_id":   fom_eid,
            "rom_element_id":   rom_eid,
            "fom_escaped":      fom_escaped.astype(np.float32),
            "rom_escaped":      rom_escaped.astype(np.float32),
        }
        _write_vtu(args.out_vtu, fom_pos, arrays)
        print(f"\n[compare] wrote {args.out_vtu}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
