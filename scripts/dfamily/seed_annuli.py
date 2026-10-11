#!/usr/bin/env python3
"""Generate a seed file covering ONLY the two radial shells where D2 loses
particles, so a diagnostic run reaches the failure within tens of steps.

Measured from LUMI job 22661472 (patched, full 8000 steps, 288k particles):
the 21,429 INTERIOR losses -- i.e. excluding the 95 that genuinely left through
z_min -- are bimodal in radius, each peak hugging a tool surface from outside:

    r 2.5-3.0 mm : 5,867   pin wall      (tool r_max 2.40-2.68 by depth)
    r 6.5-7.0 mm : 5,382   shoulder edge (6.73 mm)
    => 11,249 of 21,429 (52.5%) inside two 0.5 mm bins

Seeding the full domain wastes ~93% of particles on regions that never fail and
needs ~900 steps before the first loss. Seeding these shells puts every particle
where the defect lives.

Writes a .npy of shape (N, 3) in METRES, which is what
`run_tracking.py --seed-source file --seed-file` expects (it accepts
.npy / .npz / .vtkhdf, not plain text).
"""
from __future__ import annotations
import argparse, numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    # Shells bracket each measured peak with a margin, in mm.
    ap.add_argument("--shell", action="append", default=None, metavar="RLO:RHI",
                    help="radial shell in mm (repeatable). "
                         "Default: 2.2:3.6 and 6.2:7.6")
    ap.add_argument("--z-range", default="-6.0:0.3", metavar="ZLO:ZHI",
                    help="depth range in mm (default -6.0:0.3)")
    ap.add_argument("--n", type=int, default=120_000,
                    help="total particles (default 120,000)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    shells = args.shell or ["2.2:3.6", "6.2:7.6"]
    shells = [tuple(float(v) for v in s.split(":")) for s in shells]
    zlo, zhi = (float(v) for v in args.z_range.split(":"))

    rng = np.random.default_rng(args.seed)
    per = args.n // len(shells)
    pts = []
    for rlo, rhi in shells:
        # area-uniform in the annulus: r = sqrt(U*(rhi^2-rlo^2)+rlo^2)
        u = rng.random(per)
        r = np.sqrt(u * (rhi**2 - rlo**2) + rlo**2)
        th = rng.random(per) * 2.0 * np.pi
        z = rng.uniform(zlo, zhi, per)
        pts.append(np.column_stack([r * np.cos(th), r * np.sin(th), z]))
    P = np.vstack(pts) / 1000.0          # mm -> m

    out = args.out
    if not out.endswith((".npy", ".npz")):
        out = out + ".npy"
        print(f"  (appended .npy: the loader requires .npy/.npz/.vtkhdf)")
    np.save(out, P.astype(np.float64))
    args.out = out
    print(f"wrote {len(P):,} seeds -> {args.out}")
    for rlo, rhi in shells:
        print(f"  shell r {rlo}-{rhi} mm, z {zlo}..{zhi} mm : {per:,} particles")
    print(f"  bounding box (m): x {P[:,0].min():.5f}..{P[:,0].max():.5f}  "
          f"y {P[:,1].min():.5f}..{P[:,1].max():.5f}  "
          f"z {P[:,2].min():.5f}..{P[:,2].max():.5f}")
    print()
    print("NOTE: seeds landing inside the tool (LEVEL<0) will fail initial")
    print("assignment and be excluded; that is expected and harmless -- the")
    print("analysis only counts particles that had a host at step 0.")


if __name__ == "__main__":
    main()
