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


def _resolve_step(step_arg: str | int, n_steps: int) -> int:
    """Turn a user-facing step spec into a 0-based integer index into the
    NumberOfPoints / PointOffsets arrays.

    Accepted:
        int (positive)         -> that step (clamped into [0, n_steps-1])
        int (negative)         -> from the end: -1 == last, -2 == second-to-last
        'last' | 'end'         -> n_steps - 1
        'first' | 'start' | '0' -> 0

    A JAXTrace VTKHDF archive stores an initial-state row plus one row per
    RK4 step, so n_steps typically equals N_STEPS+1.  Step 0 is the seed
    positions; step -1 is the tracker's final output.
    """
    if isinstance(step_arg, str):
        s = step_arg.strip().lower()
        if s in ("last", "end"):
            return n_steps - 1
        if s in ("first", "start"):
            return 0
        try:
            step_arg = int(s)
        except ValueError as exc:
            raise ValueError(
                f"--step: cannot parse '{step_arg}' (expected int, 'first', "
                f"'last', 'end', or 'start')"
            ) from exc
    idx = int(step_arg)
    if idx < 0:
        idx = n_steps + idx
    if idx < 0 or idx >= n_steps:
        raise IndexError(
            f"--step {step_arg} out of range [0, {n_steps}) (archive has "
            f"{n_steps} snapshots)"
        )
    return idx


def _peek_n_steps(vtkhdf_path: Path) -> int:
    """Return the number of per-step rows in a JAXTrace VTKHDF archive."""
    import h5py
    with h5py.File(str(vtkhdf_path), "r") as f:
        return int(f["VTKHDF"]["NumberOfPoints"].shape[0])


def _load_positions_vtkhdf(vtkhdf_path: Path, step: int):
    """Read one snapshot of a JAXTrace VTKHDF particle archive.

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

    Args
    ----
    vtkhdf_path : Path
    step : int
        0-based step index (already resolved via _resolve_step).

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
        start = int(pts_offsets[step])
        count = int(n_pts_per_step[step])
        positions = np.asarray(
            vtkhdf["Points"][start:start + count],
            dtype=np.float32,
        )

        eid = np.full(count, -1, dtype=np.int32)
        if ("PointData" in vtkhdf and "ElementID" in vtkhdf["PointData"]):
            eid_offsets = vtkhdf["Steps"]["PointDataOffsets"]["ElementID"][:]
            eid_start = int(eid_offsets[step])
            eid = np.asarray(
                vtkhdf["PointData"]["ElementID"][eid_start:eid_start + count],
                dtype=np.int32,
            )

        escaped = np.zeros(count, dtype=np.uint8)
        if ("PointData" in vtkhdf and "Escaped" in vtkhdf["PointData"]):
            esc_offsets = vtkhdf["Steps"]["PointDataOffsets"]["Escaped"][:]
            esc_start = int(esc_offsets[step])
            escaped = np.asarray(
                vtkhdf["PointData"]["Escaped"][esc_start:esc_start + count],
                dtype=np.uint8,
            )

    return positions, eid, escaped


# Retained for backward compatibility with any external caller.
def _load_last_positions_vtkhdf(vtkhdf_path: Path):
    n_steps = _peek_n_steps(vtkhdf_path)
    return _load_positions_vtkhdf(vtkhdf_path, n_steps - 1)


def _write_particles(out_path: Path, positions: np.ndarray,
                     arrays: dict) -> None:
    """Write a Points-only particle dataset with the given per-particle
    arrays.  Format is chosen from the output suffix:

        *.vtp   -> vtkXMLPolyDataWriter    (PolyData root)
        *.vtu   -> vtkXMLUnstructuredGridWriter (UnstructuredGrid root)

    Previously this always used the PolyData writer regardless of
    extension, which produced files with a PolyData root but a .vtu
    name.  ParaView refuses to open .vtu files whose root element is
    not UnstructuredGrid, so those outputs were unreadable (see
    /scratch/shared/ROM/ROM_recon_centered/cylindrical_00{1,4}.gid/
    case00{1,4}_rom_vs_fom.vtu for the historical artefacts).
    """
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    n = positions.shape[0]
    suffix = out_path.suffix.lower()

    # Vertex connectivity: one 1-node cell per particle.  Vectorised
    # via numpy so it stays O(n) rather than 360k Python calls.
    verts = np.empty(2 * n, dtype=np.int64)
    verts[0::2] = 1                                # cell size = 1
    verts[1::2] = np.arange(n, dtype=np.int64)     # the point id
    id_arr = numpy_to_vtkIdTypeArray(verts, deep=True)
    cells = vtk.vtkCellArray()
    cells.SetCells(n, id_arr)

    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(positions.astype(np.float32), deep=True))

    if suffix == ".vtu":
        ug = vtk.vtkUnstructuredGrid()
        ug.SetPoints(pts)
        ug.SetCells(vtk.VTK_VERTEX, cells)
        for name, arr in arrays.items():
            va = numpy_to_vtk(
                np.ascontiguousarray(arr.astype(np.float32)), deep=True,
            )
            va.SetName(name)
            ug.GetPointData().AddArray(va)
        w = vtk.vtkXMLUnstructuredGridWriter()
        w.SetFileName(str(out_path))
        w.SetInputData(ug)
    else:
        # Default to PolyData for any other suffix (including .vtp).
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetVerts(cells)
        for name, arr in arrays.items():
            va = numpy_to_vtk(
                np.ascontiguousarray(arr.astype(np.float32)), deep=True,
            )
            va.SetName(name)
            pd.GetPointData().AddArray(va)
        w = vtk.vtkXMLPolyDataWriter()
        w.SetFileName(str(out_path))
        w.SetInputData(pd)

    w.SetDataModeToBinary()
    w.SetCompressorTypeToZLib()
    w.Write()


# Retained under the old name so callers inside this file keep working.
_write_vtu = _write_particles


def _first_step_with_both_alive(fom_path: Path, rom_path: Path,
                                n_steps: int, coarse: int = 20) -> int | None:
    """Scan the archives on a coarse stride and return the FIRST step
    (from the end walking backwards) at which at least one particle is
    alive in both runs.  Used only for a diagnostic hint when the user
    asks for --step last and gets an empty both-alive subset.

    Returns None if no such step exists.
    """
    if n_steps <= 1:
        return None
    stride = max(1, n_steps // coarse)
    # Walk backwards from the end (skipping the last we already checked).
    # We want the LATEST both-alive step so it's the closest to the
    # user's intended final-state comparison.
    for step in range(n_steps - 1, -1, -stride):
        _, _, fesc = _load_positions_vtkhdf(fom_path, step)
        _, _, resc = _load_positions_vtkhdf(rom_path, step)
        n_common = min(fesc.shape[0], resc.shape[0])
        both_alive = (fesc[:n_common] == 0) & (resc[:n_common] == 0)
        if both_alive.any():
            return step
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare particle positions at a chosen step: "
                    "FOM vs ROM tracking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--fom-vtkhdf", type=Path, required=True,
                    help="Path to the FOM tracking's particles.vtkhdf")
    ap.add_argument("--rom-vtkhdf", type=Path, required=True,
                    help="Path to the ROM tracking's particles.vtkhdf")
    ap.add_argument(
        "--step", type=str, default="last",
        help="Which timestep to compare.  Accepts an integer (0-based "
             "index into the VTKHDF Steps table, where 0 is the seed "
             "positions and -1 is the tracker's final state), or one of "
             "the labels 'first' / 'last' / 'end' / 'start'.  For the "
             "cylindrical ROM cases the archive typically holds "
             "N_STEPS+1 rows.",
    )
    ap.add_argument(
        "--list-steps", action="store_true",
        help="Just print how many steps each archive contains and exit.",
    )
    ap.add_argument(
        "--suggest-alive-step", action="store_true",
        help="If the requested step has NO particles alive in both runs, "
             "scan backwards on a coarse stride to find the latest step "
             "that does, and re-run the comparison there.",
    )
    ap.add_argument("--out-vtu", type=Path, default=None,
                    help="Optional: write a Points-only particle dataset "
                         "containing the FOM final positions plus per-"
                         "particle displacement vector and magnitude "
                         "(open in ParaView to see where the ROM "
                         "trajectory diverges most).  Format follows "
                         "the extension: .vtp -> vtkPolyData (recommended), "
                         ".vtu -> vtkUnstructuredGrid.  Both are ParaView-"
                         "readable; earlier versions of this tool ignored "
                         "the extension and always emitted PolyData, which "
                         "made .vtu-named outputs unreadable.")
    args = ap.parse_args()

    if not args.fom_vtkhdf.exists():
        print(f"ERROR: FOM archive not found: {args.fom_vtkhdf}", file=sys.stderr)
        return 3
    if not args.rom_vtkhdf.exists():
        print(f"ERROR: ROM archive not found: {args.rom_vtkhdf}", file=sys.stderr)
        return 3

    # Peek both archives; each may have a different n_steps if the two
    # runs used different N_STEPS.  We compare per-archive.
    fom_n_steps = _peek_n_steps(args.fom_vtkhdf)
    rom_n_steps = _peek_n_steps(args.rom_vtkhdf)

    if args.list_steps:
        print(f"FOM archive: {fom_n_steps} snapshots  "
              f"(indices 0..{fom_n_steps - 1})")
        print(f"ROM archive: {rom_n_steps} snapshots  "
              f"(indices 0..{rom_n_steps - 1})")
        return 0

    try:
        fom_step = _resolve_step(args.step, fom_n_steps)
        rom_step = _resolve_step(args.step, rom_n_steps)
    except (ValueError, IndexError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4

    if fom_step != rom_step:
        print(f"[compare] NOTE: --step {args.step} resolved to different "
              f"indices per archive (FOM={fom_step}, ROM={rom_step}) — "
              f"the two runs have different snapshot counts "
              f"(FOM={fom_n_steps}, ROM={rom_n_steps}).  Continuing with "
              f"per-archive resolution.",
              file=sys.stderr)

    print(f"[compare] FOM: {args.fom_vtkhdf}  (step {fom_step}/{fom_n_steps - 1})")
    fom_pos, fom_eid, fom_escaped = _load_positions_vtkhdf(
        args.fom_vtkhdf, fom_step,
    )
    print(f"[compare] ROM: {args.rom_vtkhdf}  (step {rom_step}/{rom_n_steps - 1})")
    rom_pos, rom_eid, rom_escaped = _load_positions_vtkhdf(
        args.rom_vtkhdf, rom_step,
    )

    def _report(fom_pos, fom_eid, fom_escaped,
                rom_pos, rom_eid, rom_escaped, out_vtu_path):
        """Print the FOM-vs-ROM stats block and (optionally) write a VTU.
        Returns int(both_alive.sum()) so main() can decide whether to
        retry at an earlier step."""
        if fom_pos.shape[0] != rom_pos.shape[0]:
            print(f"[compare] WARNING: particle counts differ: "
                  f"FOM={fom_pos.shape[0]:,}, ROM={rom_pos.shape[0]:,}. "
                  f"Comparing the min({fom_pos.shape[0]}, "
                  f"{rom_pos.shape[0]}) leading particles.  This suggests "
                  f"the two runs did not seed identically — check the "
                  f"seeding config (SEED_SOURCE, SEED_FRACTION, SEED_GRID, "
                  f"FEMUSS_START, inlet cropping).",
                  file=sys.stderr)
            n = min(fom_pos.shape[0], rom_pos.shape[0])
            fom_pos = fom_pos[:n]; rom_pos = rom_pos[:n]
            fom_eid = fom_eid[:n]; rom_eid = rom_eid[:n]
            fom_escaped = fom_escaped[:n]; rom_escaped = rom_escaped[:n]

        n_total = fom_pos.shape[0]
        disp = rom_pos - fom_pos
        disp_mag = np.linalg.norm(disp, axis=1)
        fom_span = float(
            np.linalg.norm(fom_pos.max(axis=0) - fom_pos.min(axis=0))
        )

        fom_alive = (fom_escaped == 0)
        rom_alive = (rom_escaped == 0)
        both_alive   = fom_alive & rom_alive
        both_escaped = (~fom_alive) & (~rom_alive)
        only_fom_esc = (~fom_alive) & rom_alive
        only_rom_esc = fom_alive & (~rom_alive)

        print()
        print(f"[compare] particles          : {n_total:,}")
        print(f"[compare] FOM bbox           : "
              f"[{fom_pos.min(0)}] -> [{fom_pos.max(0)}]")
        print(f"[compare] FOM diagonal       : {fom_span:.4e}")
        print()
        print(f"[compare] escape-flag agreement:")
        print(f"          both alive             : "
              f"{int(both_alive.sum()):>10,}  ({100*both_alive.mean():5.2f}%)")
        print(f"          both escaped           : "
              f"{int(both_escaped.sum()):>10,}  ({100*both_escaped.mean():5.2f}%)")
        print(f"          only FOM escaped       : "
              f"{int(only_fom_esc.sum()):>10,}  ({100*only_fom_esc.mean():5.2f}%)")
        print(f"          only ROM escaped       : "
              f"{int(only_rom_esc.sum()):>10,}  ({100*only_rom_esc.mean():5.2f}%)")

        def _disp_stats(mask, label):
            if not mask.any():
                print(f"[compare] displacement ({label}): "
                      f"no particles in subset")
                return
            dm = disp_mag[mask]
            d  = disp[mask]
            print(f"[compare] displacement ({label}, N={int(mask.sum()):,}):")
            print(f"          mean               : {float(dm.mean()):.4e}")
            print(f"          median             : "
                  f"{float(np.median(dm)):.4e}")
            print(f"          rms                : "
                  f"{float(np.sqrt((dm**2).mean())):.4e}")
            print(f"          p95                : "
                  f"{float(np.percentile(dm, 95)):.4e}")
            print(f"          p99                : "
                  f"{float(np.percentile(dm, 99)):.4e}")
            print(f"          max                : {float(dm.max()):.4e}")
            print(f"          rms / FOM diagonal : "
                  f"{100 * float(np.sqrt((dm**2).mean())) / max(fom_span, 1e-30):.3f}%")
            for j, comp in enumerate("xyz"):
                print(f"          rms_{comp}              : "
                      f"{float(np.sqrt((d[:, j]**2).mean())):.4e}")

        print()
        _disp_stats(np.ones(n_total, dtype=bool), "all particles")
        print()
        _disp_stats(both_alive, "both-alive subset (fair comparison)")

        if out_vtu_path:
            arrays = {
                "displacement_vec": disp,
                "displacement_mag": disp_mag,
                "fom_element_id":   fom_eid,
                "rom_element_id":   rom_eid,
                "fom_escaped":      fom_escaped.astype(np.float32),
                "rom_escaped":      rom_escaped.astype(np.float32),
            }
            _write_vtu(out_vtu_path, fom_pos, arrays)
            print(f"\n[compare] wrote {out_vtu_path}")

        return int(both_alive.sum())

    n_both_alive = _report(
        fom_pos, fom_eid, fom_escaped,
        rom_pos, rom_eid, rom_escaped,
        args.out_vtu,
    )

    # Optional fallback: if the requested step has an empty both-alive
    # subset, scan backwards for the latest step that has at least one
    # both-alive particle and re-report there.  Only fires when the
    # user asked for it, so the default behaviour is unchanged.
    if n_both_alive == 0 and args.suggest_alive_step:
        # Only scan the FOM archive's index space; the caller can pass
        # different N_STEPS but the two archives should agree on which
        # early snapshots have alive particles.
        print()
        print("[compare] --suggest-alive-step: no both-alive particles at "
              "requested step, scanning backwards for an earlier snapshot "
              "where at least one exists...")
        earlier = _first_step_with_both_alive(
            args.fom_vtkhdf, args.rom_vtkhdf,
            n_steps=min(fom_n_steps, rom_n_steps),
        )
        if earlier is None:
            print("[compare]   no step in either archive has both-alive "
                  "particles — seeding or physics disagree fundamentally.")
        else:
            print(f"[compare]   trying step {earlier} instead")
            fom_pos2, fom_eid2, fom_escaped2 = _load_positions_vtkhdf(
                args.fom_vtkhdf, earlier,
            )
            rom_pos2, rom_eid2, rom_escaped2 = _load_positions_vtkhdf(
                args.rom_vtkhdf, earlier,
            )
            # Only write a distinct VTU if the user provided one — mangle
            # the filename so we don't overwrite the first-step output.
            fallback_vtu = None
            if args.out_vtu:
                fallback_vtu = args.out_vtu.with_name(
                    args.out_vtu.stem + f".step_{earlier}"
                    + args.out_vtu.suffix,
                )
            _report(
                fom_pos2, fom_eid2, fom_escaped2,
                rom_pos2, rom_eid2, rom_escaped2,
                fallback_vtu,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
