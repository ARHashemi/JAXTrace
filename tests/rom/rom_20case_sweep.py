"""
20-case ROM reconstruction residual sweep at the FOM's final timestep.

For each of the 20 cylindrical cases in /scratch/shared/ROM/FOM, this
script:
  1. loads the shared FSW-ROM basis + coefficients,
  2. reconstructs velocity at the case's mesh nodes for every supported
     formula ('centered', 'sigma_c', 'c_over_sig', ...),
  3. loads the case's own FOM Displacement at --ts (default 119) as
     ground truth,
  4. reports per-case rel_rms against the FOM, plus aggregate mean/std
     across the cohort.

Referenced from docs/rom_reconstruction_findings.md.

Reads only:
  /scratch/shared/ROM/FOM/cylindrical.som.fswrom.basis
  /scratch/shared/ROM/FOM/cylindrical.som.fswrom.romdata
  /scratch/shared/ROM/FOM/cylindrical_<idx>.gid/post/cylindrical_<ts>.pvtu

Usage:
  python3 tests/rom/rom_20case_sweep.py                    # ts=119, all 20 cases
  python3 tests/rom/rom_20case_sweep.py --ts 90 --formula centered

No JAX / no GPU required — pure numpy + h5py + vtk-via-jaxtrace-loader.
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
    load_basis, load_coefficients, reconstruct_all_formulas,
)
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu


DEFAULT_ROM_ROOT = Path("/scratch/shared/ROM/FOM")


def _load_fom(rom_root: Path, case_idx: int, ts: int) -> np.ndarray:
    case_dir = rom_root / f"cylindrical_{case_idx:03d}.gid" / "post"
    _, _, vs = load_velocity_sequence_from_pvtu(
        base_path=case_dir,
        file_pattern="cylindrical_{timestep}.pvtu",
        timestep_range=(ts, ts),
        field_name="Displacement",
        verbose=False,
    )
    return vs[0].astype(np.float64)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt((x ** 2).mean()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rom-root", type=Path, default=DEFAULT_ROM_ROOT)
    ap.add_argument("--ts", type=int, default=119,
                    help="FOM timestep index used as reconstruction target")
    ap.add_argument("--formulas", type=str, nargs="+",
                    default=("centered", "c_over_sig"),
                    help="Which formulas to score for each case")
    args = ap.parse_args()

    basis = load_basis(args.rom_root / "cylindrical.som.fswrom.basis",
                       verbose=False)
    coeffs = load_coefficients(args.rom_root / "cylindrical.som.fswrom.romdata",
                               verbose=False)
    print(f"basis: n_nodes={basis.n_nodes:,}, n_modes={basis.n_modes}")
    print(f"coefs: {coeffs.coefficients.shape}, ts={args.ts}")

    hdr = f'{"case":>4}  {"|FOM|_rms":>10}  {"mean_rel":>9}'
    for f in args.formulas:
        hdr += f'  {f + "_rel":>13}'
    hdr += f'  {"|c1|":>7}  {"|c2|":>7}  {"|c3|":>7}'
    print()
    print(hdr)
    print("-" * len(hdr))

    all_rels = {f: [] for f in args.formulas}
    mean_rels = []
    rows = []
    for case_idx in range(coeffs.n_cases):
        try:
            fom = _load_fom(args.rom_root, case_idx, args.ts)
        except Exception as exc:
            print(f"{case_idx:>4}  SKIP: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue
        recons = reconstruct_all_formulas(basis, coeffs, case_idx)
        fom_rms = _rms(fom)
        mean_rms = _rms(basis.mean - fom)
        mean_rel = mean_rms / fom_rms
        mean_rels.append(mean_rel)
        c = coeffs.coefficients[:, case_idx]
        line = f'{case_idx:>4}  {fom_rms:>10.4e}  {100 * mean_rel:>8.2f}%'
        for f in args.formulas:
            rel = _rms(recons[f] - fom) / fom_rms
            line += f'  {100 * rel:>12.2f}%'
            all_rels[f].append(rel)
        line += f'  {c[0]:>7.2f}  {c[1]:>7.3f}  {c[2]:>7.3f}'
        rows.append(line)
        print(line)

    if not rows:
        print("No cases processed", file=sys.stderr)
        return 3

    print()
    print("Aggregate:")
    print(f"  mean-only : mean={100 * np.mean(mean_rels):6.2f}%  "
          f"sd={100 * np.std(mean_rels):.2f}%")
    for f in args.formulas:
        arr = np.array(all_rels[f])
        print(f"  {f:9s}: mean={100 * arr.mean():6.2f}%  "
              f"sd={100 * arr.std():.2f}%  "
              f"min={100 * arr.min():.2f}%  "
              f"max={100 * arr.max():.2f}%")

    # Best and worst under the primary formula
    primary = args.formulas[0]
    idx_best = int(np.argmin(all_rels[primary]))
    idx_worst = int(np.argmax(all_rels[primary]))
    print()
    print(f'BEST  case: {idx_best:02d}  {primary}_rel = '
          f'{100 * all_rels[primary][idx_best]:.2f}%')
    print(f'WORST case: {idx_worst:02d}  {primary}_rel = '
          f'{100 * all_rels[primary][idx_worst]:.2f}%')
    return 0


if __name__ == "__main__":
    sys.exit(main())
