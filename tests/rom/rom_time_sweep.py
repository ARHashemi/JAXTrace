"""
Per-timestep ROM reconstruction residual for a chosen case.

The stored ROM coefficients are ONE number per case per mode (not a
trajectory), so the reconstruction is a single static field per case.
Comparing that static field against every FOM timestep tells us at
which point of the FOM's transient the ROM prediction matches best,
and how the residual grows if the FOM is still evolving.

Referenced from docs/rom_reconstruction_findings.md.

Usage:
  python3 tests/rom/rom_time_sweep.py --case 0 --stride 5
  python3 tests/rom/rom_time_sweep.py --case 3 --formula centered

Reads only /scratch/shared/ROM/FOM/.  No JAX / no GPU required.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rom-root", type=Path, default=DEFAULT_ROM_ROOT)
    ap.add_argument("--case", type=int, required=True,
                    help="Case index (0..19)")
    ap.add_argument("--formula", type=str, default="centered")
    ap.add_argument("--ts-min", type=int, default=1)
    ap.add_argument("--ts-max", type=int, default=119)
    ap.add_argument("--stride", type=int, default=5,
                    help="Sample every N timesteps")
    args = ap.parse_args()

    basis = load_basis(args.rom_root / "cylindrical.som.fswrom.basis",
                       verbose=False)
    coeffs = load_coefficients(
        args.rom_root / "cylindrical.som.fswrom.romdata", verbose=False,
    )
    v_recon = reconstruct(basis, coeffs, args.case, args.formula)

    print(f"case {args.case:02d}  formula='{args.formula}'  "
          f"stride={args.stride}")
    print(f"{'ts':>4}  {'|FOM|_rms':>10}  {'mean_rel':>9}  {'recon_rel':>9}")
    print("-" * 40)

    best = (None, 1e30)
    worst = (None, -1.0)
    case_dir = args.rom_root / f"cylindrical_{args.case:03d}.gid" / "post"
    for ts in range(args.ts_min, args.ts_max + 1, args.stride):
        try:
            _, _, vs = load_velocity_sequence_from_pvtu(
                base_path=case_dir,
                file_pattern="cylindrical_{timestep}.pvtu",
                timestep_range=(ts, ts),
                field_name="Displacement",
                verbose=False,
            )
        except Exception:
            continue
        fom = vs[0].astype(np.float64)
        if fom.shape[0] != basis.n_nodes:
            continue
        fom_rms = _rms(fom)
        if fom_rms == 0:
            continue
        mean_rel = _rms(basis.mean - fom) / fom_rms
        recon_rel = _rms(v_recon - fom) / fom_rms
        print(f"{ts:>4}  {fom_rms:>10.4e}  {100 * mean_rel:>8.2f}%  "
              f"{100 * recon_rel:>8.2f}%")
        if recon_rel < best[1]:
            best = (ts, recon_rel)
        if recon_rel > worst[1]:
            worst = (ts, recon_rel)

    if best[0] is None:
        return 3
    print()
    print(f"BEST  ts={best[0]}: {args.formula}_rel = {100 * best[1]:.2f}%")
    print(f"WORST ts={worst[0]}: {args.formula}_rel = {100 * worst[1]:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
