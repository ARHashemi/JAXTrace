"""Convert the Phase 3 `damage_models.npz` files to ParaView-readable VTU.

    python3 npz_to_vtu.py                      # all 45 cases
    python3 npz_to_vtu.py --only rom_000 ps_C-ThreadsVariations_C1

⚠️ YOU DO NOT NEED TO RE-RUN ANYTHING for a static view. The 45 completed runs
used `--no-export`, so no per-timestep VTU exists -- but `damage_models.npz`
already holds the FINAL particle positions and their accumulated damage, which is
what you look at anyway when asking "where did damage end up?".

What this gives you, and what it does not:

  ✅ final particle cloud, coloured by ln(Phi/Phi0) or by Cockcroft-Latham C
  ✅ every one of the 45 cases, in seconds, from data already on disk
  ❌ the TRAJECTORIES (no intermediate timesteps were saved)
  ❌ animation over time

To get trajectories you must re-run WITH `--export-format vtu`, which now also
writes `Damage_lnPhi` / `Damage_CL` at every export step. Re-run only the cases you
actually want to animate -- it is ~1 h per PinShapes case.

Writes legacy-free XML VTU via vtk if available, else a hand-written ASCII VTU so
this works even without the vtk module.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np

SRC = Path("/home/arhashemi/lumi/lumi_scratch/hashemia/damage/"
           "phase3_overnight_20260930")


def write_vtu_ascii(path: Path, pos: np.ndarray, arrays: dict) -> None:
    """Minimal ASCII VTU: one vertex cell per particle. No vtk dependency.

    ASCII is ~3x larger than binary but these are 100k points (~8 MB), it is
    human-inspectable, and it removes a dependency from a script whose whole
    point is to be easy to run.
    """
    n = len(pos)
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" '
                'byte_order="LittleEndian">\n  <UnstructuredGrid>\n')
        f.write(f'    <Piece NumberOfPoints="{n}" NumberOfCells="{n}">\n')

        f.write('      <PointData>\n')
        for name, a in arrays.items():
            f.write(f'        <DataArray type="Float32" Name="{name}" '
                    'format="ascii">\n          ')
            f.write(" ".join(f"{v:.6g}" for v in np.asarray(a, dtype=np.float32)))
            f.write("\n        </DataArray>\n")
        f.write('      </PointData>\n')

        f.write('      <Points>\n        <DataArray type="Float32" '
                'NumberOfComponents="3" format="ascii">\n          ')
        f.write(" ".join(f"{v:.6g}" for v in pos.ravel()))
        f.write("\n        </DataArray>\n      </Points>\n")

        f.write('      <Cells>\n')
        f.write('        <DataArray type="Int32" Name="connectivity" '
                'format="ascii">\n          ')
        f.write(" ".join(str(i) for i in range(n)))
        f.write('\n        </DataArray>\n')
        f.write('        <DataArray type="Int32" Name="offsets" '
                'format="ascii">\n          ')
        f.write(" ".join(str(i + 1) for i in range(n)))
        f.write('\n        </DataArray>\n')
        f.write('        <DataArray type="UInt8" Name="types" '
                'format="ascii">\n          ')
        f.write(" ".join("1" for _ in range(n)))        # VTK_VERTEX
        f.write('\n        </DataArray>\n      </Cells>\n')
        f.write('    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=Path("vtu_final"))
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cases = sorted(glob.glob(str(args.src / "*" / "damage_models.npz")))
    if args.only:
        keep = set(args.only)
        cases = [c for c in cases if os.path.basename(os.path.dirname(c)) in keep]
    if not cases:
        print("  no cases found"); return 1

    for c in cases:
        tag = os.path.basename(os.path.dirname(c))
        z = np.load(c, allow_pickle=True)
        dm, pos = z["damage"], z["positions"].astype(np.float64)
        arrays = {}
        if dm.ndim == 2 and dm.shape[1] >= 2:
            ln, C = dm[:, 0], dm[:, 1]
            arrays["Damage_lnPhi"] = ln
            arrays["Damage_CL"] = C
            # ⚠️ Phi itself would be exp(lnPhi), which overflows: lnPhi reaches
            # 1270 on some cases and exp(1270) is ~1e551. Export a CLIPPED
            # saturating indicator instead, so ParaView's colour scale is usable
            # and nobody reads an unbounded number as a porosity (O31).
            ln_fail = float(np.log(1.0 / 1.0e-4))        # = 9.21
            arrays["Damage_fracFailed"] = np.clip(ln / ln_fail, 0.0, 1.0)
        else:
            arrays["Damage_ebar"] = dm.reshape(-1)
        arrays["ElementID"] = z["element_ids"].astype(np.float32)
        # radius is handy for thresholding to the stir zone in ParaView
        arrays["Radius_mm"] = np.hypot(pos[:, 0], pos[:, 1]) * 1e3

        out = args.out / f"{tag}_final.vtu"
        write_vtu_ascii(out, pos, arrays)
        print(f"  {tag:34s} {len(pos):,} particles -> {out.name}")

    print()
    print(f"  wrote {len(cases)} file(s) to {args.out}/")
    print("  ParaView: open, Representation=Points, colour by Damage_lnPhi")
    print("  ⚠️ These are FINAL positions only -- no trajectories. For animation,")
    print("     re-run the case with --export-format vtu (~1 h per PinShapes case).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
