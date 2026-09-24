#!/usr/bin/env python3
"""Sweep ROM-vs-FOM particle-tracking error across cases and ROM variants.

`compare_rom_vs_fom_tracking.py` compares ONE pair of vtkhdf files. This
wraps it over a cohort and, optionally, over two ROM variants so the
shipped-FEMUSS and our-POD runs can be put side by side.

Radial binning note
-------------------
Particles are seeded in an annulus at r >= 13.2 mm, so a near-pin bin
defined on the *initial* radius is always empty. This script bins on the
radius at the reporting step instead, which is what "near the pin at the
time we look" actually means for this seeding.

Usage
-----
    # our POD runs alone
    python3 scripts/compare_rom_pt_sweep.py \
        --rom-root /scratch/shared/ROM --variant ourpod3

    # head-to-head against the shipped-basis runs
    python3 scripts/compare_rom_pt_sweep.py \
        --rom-root /scratch/shared/ROM \
        --variant ourpod3 --baseline centered
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

D_PIN_MM = 10.0


def load_step(path: Path, target: float):
    with h5py.File(path, "r") as f:
        off = f["VTKHDF/Steps/PointOffsets"][:]
        val = f["VTKHDF/Steps/Values"][:]
        i = int(np.argmin(np.abs(val - target)))
        n = int(off[1] - off[0]) if len(off) > 1 else f["VTKHDF/Points"].shape[0]
        s = int(off[i])
        return f["VTKHDF/Points"][s:s + n].astype(np.float64), float(val[i])


def find_run(base: Path) -> Path | None:
    for p in sorted(base.glob("run_*/particles.vtkhdf")):
        return p
    return None


def rom_path(rom_root: Path, variant: str, case: int, prefix: str) -> Path | None:
    """ROM PT output lives beside the reconstruction, not in the FOM tree."""
    stem = f"{prefix}_{case:03d}.gid"
    for sub in (f"post_pt/rom_{variant}",
                f"post_pt/rom_{variant}_hct_on",
                f"post_pt_rom_{variant}"):
        p = find_run(rom_root / f"ROM_recon_{variant}" / stem / sub)
        if p is not None:
            return p
    return None


def fom_path(rom_root: Path, case: int, prefix: str) -> Path | None:
    d = rom_root / "FOM" / f"{prefix}_{case:03d}.gid" / "post_pt"
    p = find_run(d / "fom_hct_on")
    if p is not None:
        return p
    p = d / "run_grid-frac_n360000_s2000" / "particles.vtkhdf"
    return p if p.exists() else None


def stats(fom: np.ndarray, rom: np.ndarray) -> dict:
    n = min(len(fom), len(rom))
    a, b = fom[:n], rom[:n]
    e = np.linalg.norm(a - b, axis=1) * 1e3          # mm
    r = np.linalg.norm(a[:, :2], axis=1) * 1e3       # mm, at reporting step
    near = r <= 10.0
    out = ~near
    f = lambda m: float(np.sqrt((e[m] ** 2).mean())) if m.sum() > 10 else float("nan")
    return {
        "n": n,
        "rms": float(np.sqrt((e ** 2).mean())),
        "p95": float(np.percentile(e, 95)),
        "near": f(near),
        "outer": f(out),
        "frac_near": float(near.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rom-root", type=Path, default=Path("/scratch/shared/ROM"))
    ap.add_argument("--variant", default="ourpod3",
                    help="ROM variant label (names the ROM_recon_<v> tree)")
    ap.add_argument("--baseline", default=None,
                    help="optional second variant to compare against, "
                         "e.g. 'centered' for the shipped FEMUSS basis")
    ap.add_argument("--cases", type=int, nargs="+", default=list(range(20)))
    ap.add_argument("--step", type=float, default=950.0)
    ap.add_argument("--case-prefix", default="cylindrical")
    args = ap.parse_args()

    hdr = f"{'case':>4} {'rms mm':>8} {'D_pin':>6} {'near':>7} {'outer':>7} {'p95':>7}"
    if args.baseline:
        hdr += f" | {'base mm':>8} {'ratio':>6}"
    print(f"ROM-vs-FOM tracking error at step {args.step:.0f}  "
          f"(variant '{args.variant}')")
    print(hdr)
    print("-" * len(hdr))

    cur, base = [], []
    for c in args.cases:
        fp = fom_path(args.rom_root, c, args.case_prefix)
        rp = rom_path(args.rom_root, args.variant, c, args.case_prefix)
        if fp is None or rp is None:
            continue
        F, _ = load_step(fp, args.step)
        Rv, _ = load_step(rp, args.step)
        s = stats(F, Rv)
        cur.append(s["rms"])
        line = (f"{c:>4} {s['rms']:8.2f} {s['rms']/D_PIN_MM:6.3f} "
                f"{s['near']:7.2f} {s['outer']:7.2f} {s['p95']:7.2f}")
        if args.baseline:
            bp = rom_path(args.rom_root, args.baseline, c, args.case_prefix)
            if bp is not None:
                B, _ = load_step(bp, args.step)
                sb = stats(F, B)
                base.append(sb["rms"])
                line += f" | {sb['rms']:8.2f} {s['rms']/sb['rms']:6.2f}"
            else:
                line += f" | {'-':>8} {'-':>6}"
        print(line)

    if cur:
        print("-" * len(hdr))
        msg = (f"mean over {len(cur)} cases: {np.mean(cur):.2f} mm "
               f"= {np.mean(cur)/D_PIN_MM:.3f} D_pin")
        if base:
            msg += (f"   |  baseline '{args.baseline}' {np.mean(base):.2f} mm"
                    f"  -> x{np.mean(cur)/np.mean(base):.2f}")
        print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
