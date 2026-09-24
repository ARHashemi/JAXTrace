#!/usr/bin/env python3
"""Build and export our own POD basis for the FSW 20-case cohort.

Replaces the shipped FEMUSS basis for our work. On the ts=119 cohort the
shipped basis reconstructs at ~4 % L2; a POD built here from the same
snapshots reaches ~0.5 % with the same 3 modes.

The cohort is at steady state by the final timestep (case 000 changes by
only 0.003 % L2 between ts=118 and ts=119, and 0.17 % between ts=90 and
119), so a single final-step snapshot per case is the right training set
-- adding snapshots from the last cycle would contribute almost no new
information.

Outputs
-------
  <out>/our_pod_ts<TS>.npz     the basis (mean, modes, coeffs, sigmas)
  <out>/our_pod_ts<TS>.md      a report: spectrum, per-case errors,
                               error vs mode count, and the shipped-basis
                               comparison

Usage
-----
    python3 scripts/build_our_pod_basis.py \
        --rom-root /scratch/shared/ROM/FOM \
        --out      /scratch/shared/ROM/rom_out/our_pod
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jaxtrace.rom.pod_builder import build_pod, save_pod          # noqa: E402
from jaxtrace.gpu.mesh_loader_timedep import (                    # noqa: E402
    load_velocity_sequence_from_pvtu,
)


def load_case(rom_root: Path, case: int, ts: int) -> np.ndarray:
    _, _, vs = load_velocity_sequence_from_pvtu(
        base_path=rom_root / f"cylindrical_{case:03d}.gid" / "post",
        file_pattern="cylindrical_{timestep}.pvtu",
        timestep_range=(ts, ts),
        field_name="Displacement",
        verbose=False,
    )
    return vs[0].astype(np.float64)


def l2_rel(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 error of `a` against reference `b`."""
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / nb * 100.0) if nb > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rom-root", type=Path, default=Path("/scratch/shared/ROM/FOM"))
    ap.add_argument("--out", type=Path,
                    default=Path("/scratch/shared/ROM/rom_out/our_pod"))
    ap.add_argument("--ts", type=int, default=119,
                    help="timestep to build from (cohort is steady here)")
    ap.add_argument("--cases", type=int, nargs="+", default=list(range(20)))
    ap.add_argument("--n-modes", type=int, default=None,
                    help="keep exactly this many modes (default: keep all)")
    ap.add_argument("--energy-target", type=float, default=None,
                    help="keep fewest modes reaching this cumulative sigma^2")
    ap.add_argument("--compare-shipped", action="store_true", default=True,
                    help="also score the shipped FEMUSS basis for reference")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(args.cases)} snapshots at ts={args.ts} ...")
    snaps = np.stack([load_case(args.rom_root, c, args.ts) for c in args.cases])
    print(f"  snapshots: {snaps.shape}")

    pod = build_pod(
        snaps,
        n_modes=args.n_modes,
        energy_target=args.energy_target,
        center=True,
        source=f"cases={args.cases[0]}..{args.cases[-1]} ts={args.ts} "
               f"(centered, unweighted)",
    )
    print(f"  modes kept: {pod.n_modes} / {pod.sigmas.size}")
    print(f"  sigmas[:6]: {np.array2string(pod.sigmas[:6], precision=4)}")

    npz = save_pod(pod, args.out / f"our_pod_ts{args.ts}.npz")
    print(f"  wrote {npz}")

    # --- scoring -------------------------------------------------------
    max_k = pod.n_modes
    ks = [k for k in (1, 2, 3, 4, 5, 6, 8, 10) if k <= max_k]
    if max_k not in ks:
        ks.append(max_k)

    by_k = {}
    for k in ks:
        errs = [l2_rel(pod.reconstruct(i, n_modes=k), snaps[i])
                for i in range(len(args.cases))]
        by_k[k] = np.array(errs)

    per_case = [l2_rel(pod.reconstruct(i), snaps[i])
                for i in range(len(args.cases))]

    shipped = None
    if args.compare_shipped:
        try:
            from jaxtrace.rom.velocity_recon import load_basis, load_coefficients
            b = load_basis(args.rom_root / "cylindrical.som.fswrom.basis")
            c = load_coefficients(args.rom_root / "cylindrical.som.fswrom.romdata")
            shipped = []
            for j, i in enumerate(args.cases):
                rec = b.mean + np.einsum(
                    "k,kij->ij", c.coefficients[:b.n_modes, i], b.modes)
                shipped.append(l2_rel(rec, snaps[j]))
            shipped = np.array(shipped)
        except Exception as exc:                       # pragma: no cover
            print(f"  ! shipped-basis comparison skipped: {exc}")
            shipped = None

    # --- report --------------------------------------------------------
    lines = [
        f"# Our POD basis — cohort ts={args.ts}",
        "",
        f"Built from {len(args.cases)} snapshots, per-node mean subtracted,",
        "one shared basis across the 3 velocity components (FEMUSS convention:",
        "`v = mean + sum_k c_k phi_k`).",
        "",
        f"- modes kept: **{pod.n_modes}** of {pod.sigmas.size}",
        f"- basis file: `{npz.name}`",
        "",
        "## Singular values",
        "",
        "| k | sigma | cum. energy | residual bound |",
        "|---|---|---|---|",
    ]
    for k in range(min(10, pod.sigmas.size)):
        lines.append(f"| {k+1} | {pod.sigmas[k]:.4f} | "
                     f"{pod.energy(k+1)*100:.4f} % | "
                     f"{pod.residual_bound(k+1)*100:.4f} % |")

    lines += [
        "",
        "## Reconstruction error vs mode count",
        "",
        "In-sample L2 error against the training snapshots "
        "(`||rec - v|| / ||v||`), mean over cases.",
        "",
        "| K | mean | median | max |",
        "|---|---|---|---|",
    ]
    for k in ks:
        e = by_k[k]
        lines.append(f"| {k} | **{e.mean():.3f} %** | {np.median(e):.3f} % | "
                     f"{e.max():.3f} % |")

    lines += ["", "## Per-case error (all retained modes)", "",
              "| case | our POD |" + (" shipped FEMUSS |" if shipped is not None else ""),
              "|---|---|" + ("---|" if shipped is not None else "")]
    for j, i in enumerate(args.cases):
        row = f"| {i:03d} | {per_case[j]:.3f} % |"
        if shipped is not None:
            row += f" {shipped[j]:.3f} % |"
        lines.append(row)
    row = f"| **mean** | **{np.mean(per_case):.3f} %** |"
    if shipped is not None:
        row += f" **{shipped.mean():.3f} %** |"
    lines.append(row)

    lines += [
        "",
        "## Caveat",
        "",
        "These are **in-sample** projection errors: each snapshot is",
        "reconstructed by a basis built from a set that includes it. They are",
        "the right comparison for a 'direct reconstruction, no regression'",
        "number, but they are **not** a generalisation estimate. For that, run",
        "leave-one-out (rebuild the basis without the held-out case, then",
        "project it in).",
        "",
    ]
    md = args.out / f"our_pod_ts{args.ts}.md"
    md.write_text("\n".join(lines))
    print(f"  wrote {md}")

    print()
    print(f"  our POD, {pod.n_modes} modes : {np.mean(per_case):.3f} % mean L2")
    for k in ks:
        print(f"    K={k:2d}: {by_k[k].mean():.3f} %")
    if shipped is not None:
        print(f"  shipped FEMUSS basis   : {shipped.mean():.3f} % mean L2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
