"""
compare_rom_recon.py
====================

Score every ROM reconstruction formula against a FOM snapshot.

Given:
  * the FSW-ROM basis file (*.fswrom.basis)
  * the FSW-ROM romdata file (*.fswrom.romdata)
  * one case's FOM PVTU (the "reference truth" at some timestep)

reconstructs the velocity for that case using each supported formula
in ``jaxtrace.rom.velocity_recon``, then reports per-formula max-abs,
rms, and cosine-similarity vs the FOM snapshot.

The best-scoring formula is what ``run_tracking.py --velocity-source
rom --rom-formula <name>`` should use.

Usage
-----

  python tests/rom/compare_rom_recon.py \\
    --basis   /scratch/shared/ROM/FOM/cylindrical.som.fswrom.basis \\
    --romdata /scratch/shared/ROM/FOM/cylindrical.som.fswrom.romdata \\
    --fom-pvtu-dir /scratch/shared/ROM/FOM/cylindrical_001.gid/post \\
    --fom-pvtu-pattern 'cylindrical_{timestep}.pvtu' \\
    --fom-timestep 119 \\
    --case-idx 1 \\
    --field-name Displacement

Optional ``--output-json <path>`` dumps the ranking to a JSON file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from jaxtrace.rom.velocity_recon import (
    load_basis, load_coefficients,
    reconstruct_all_formulas, score_formulas,
)


def main():
    ap = argparse.ArgumentParser(
        description="Score every ROM reconstruction formula vs a FOM snapshot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--basis", type=Path, required=True,
                    help="Path to *.fswrom.basis")
    ap.add_argument("--romdata", type=Path, required=True,
                    help="Path to *.fswrom.romdata")
    ap.add_argument("--fom-pvtu-dir", type=Path, required=True,
                    help="Directory containing the FOM PVTUs for the case.")
    ap.add_argument("--fom-pvtu-pattern", type=str,
                    default="cylindrical_{timestep}.pvtu",
                    help="PVTU filename pattern. {timestep} is substituted.")
    ap.add_argument("--fom-timestep", type=int, required=True,
                    help="Timestep index whose FOM velocity we compare "
                         "against.")
    ap.add_argument("--case-idx", type=int, required=True,
                    help="0-based ROM case index to reconstruct.")
    ap.add_argument("--field-name", type=str, default="Displacement",
                    help="Name of the velocity field in the FOM PVTU.")
    ap.add_argument("--field-group", type=str, default="Displacement",
                    help="HDF5 group inside the basis/romdata files.")
    ap.add_argument("--output-json", type=Path, default=None,
                    help="Optional output path for the score table.")
    args = ap.parse_args()

    print("=" * 80)
    print("ROM reconstruction vs FOM comparison")
    print("=" * 80)
    print(f"  basis:      {args.basis}")
    print(f"  romdata:    {args.romdata}")
    print(f"  FOM PVTU:   {args.fom_pvtu_dir}/{args.fom_pvtu_pattern}")
    print(f"  timestep:   {args.fom_timestep}")
    print(f"  case_idx:   {args.case_idx}")
    print(f"  field:      {args.field_name}  (basis group: {args.field_group})")
    print()

    # Load basis + coefficients.
    basis = load_basis(args.basis, field_group=args.field_group, verbose=True)
    coeffs = load_coefficients(
        args.romdata, field_group=args.field_group, verbose=True,
    )
    print()

    # Load FOM reference.
    print(f"Loading FOM PVTU reference at timestep {args.fom_timestep}...")
    from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
    nodes, _, vel_seq = load_velocity_sequence_from_pvtu(
        base_path=args.fom_pvtu_dir,
        file_pattern=args.fom_pvtu_pattern,
        timestep_range=(args.fom_timestep, args.fom_timestep),
        field_name=args.field_name,
        verbose=False,
    )
    fom = vel_seq[0].astype(np.float64)
    print(f"  FOM velocity shape: {fom.shape}  |v|_inf = {np.abs(fom).max():.4e}")

    if fom.shape[0] != basis.n_nodes:
        raise SystemExit(
            f"Node count mismatch: FOM has {fom.shape[0]}, basis has "
            f"{basis.n_nodes}. The ROM and the FOM PVTU are on different "
            f"meshes."
        )
    print()

    # Try every formula.
    print("Reconstructing with every formula...")
    reconstructions = reconstruct_all_formulas(basis, coeffs, args.case_idx)
    scored = score_formulas(reconstructions, fom)

    # Report.
    print()
    print(f"{'formula':<15}  {'max_abs':>12}  {'rms':>12}  "
          f"{'cosine':>10}  {'rel_rms':>10}")
    print("-" * 68)
    fom_rms = np.sqrt((fom ** 2).mean())
    for name, mx, rms, cos in scored:
        rel = rms / fom_rms
        print(f"{name:<15}  {mx:>12.4e}  {rms:>12.4e}  {cos:>10.4f}  "
              f"{rel:>10.4f}")

    print()
    best = scored[0]
    print(f"Best formula: {best[0]}  "
          f"(rms={best[2]:.4e}, {100*best[2]/fom_rms:.2f}% of |FOM|_rms)")

    # Also report per-component errors for the best formula, so the
    # user can see whether the mismatch is uniform or concentrated in
    # one velocity component.
    best_recon = reconstructions[best[0]]
    diff = best_recon - fom
    print()
    print(f"Per-component error for best formula '{best[0]}':")
    for j, comp in enumerate("uvw"):
        d = diff[:, j]
        r = fom[:, j]
        print(f"  {comp}: max_abs={np.abs(d).max():.4e}  "
              f"rms={np.sqrt((d*d).mean()):.4e}  "
              f"|ref|_rms={np.sqrt((r*r).mean()):.4e}")

    # Also report the residual against just the SnapshotsMean (formula
    # 'centered' with all c_k = 0). If mean-alone is comparable to the
    # top formula, the basis modes barely help — telling us that the
    # ROM was built on a case set whose modes don't capture much of
    # the FOM at this timestep.
    mean_only_diff = basis.mean - fom
    mean_only_rms = float(np.sqrt((mean_only_diff ** 2).mean()))
    print(f"  (sanity) mean alone: rms={mean_only_rms:.4e}  "
          f"({100*mean_only_rms/fom_rms:.2f}% of |FOM|_rms)")

    # JSON dump.
    if args.output_json:
        payload = {
            "basis": str(args.basis),
            "romdata": str(args.romdata),
            "fom_pvtu": str(args.fom_pvtu_dir / args.fom_pvtu_pattern.format(
                timestep=args.fom_timestep,
            )),
            "case_idx": args.case_idx,
            "n_nodes": int(basis.n_nodes),
            "n_modes_in_basis": int(basis.n_modes),
            "n_modes_in_romdata": int(coeffs.n_modes),
            "n_sigmas": int(coeffs.sigmas.shape[0]),
            "sigmas": coeffs.sigmas.tolist(),
            "fom_rms": float(fom_rms),
            "mean_only_rms": mean_only_rms,
            "ranked": [
                {"formula": name, "max_abs": mx, "rms": rms,
                 "cosine": cos, "rel_rms": rms / fom_rms}
                for (name, mx, rms, cos) in scored
            ],
            "best_formula": best[0],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
