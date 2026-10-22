"""Side-by-side particle clouds: raw P1 vs divergence-free.

Answers the question numbers cannot: do the STREAMLINES behind the tool look
different? The hypothesis being tested is that making the field divergence-free
gives more uniform streamlines in the wake for uniformly-gridded seeding.

⚠️ The numeric comparison (compare_divfree_trajectories.py) reports the shift
magnitude. This shows the SPATIAL PATTERN, which is what "uniform streamlines"
is a claim about -- a small mean shift could still hide a systematic
rearrangement, and conversely.

Three panels per projection:
  1. baseline cloud
  2. divergence-free cloud
  3. per-particle shift, coloured by magnitude, so any spatial structure in
     the difference is visible (e.g. larger shifts downstream of the tool)

Usage:
    python3 plot_divfree_trajectories.py BASELINE_DIR DIVFREE_DIR OUT.png
            [--plane yz|xy|xz] [--bbox xmin xmax ymin ymax zmin zmax]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

PLANES = {"yz": (1, 2, "y [m]", "z [m]"),
          "xy": (0, 1, "x [m]", "y [m]"),
          "xz": (0, 2, "x [m]", "z [m]")}


def load(d: Path):
    fs = sorted(d.glob("*/final_state.npz"))
    if not fs:
        raise SystemExit(f"no */final_state.npz under {d}")
    z = np.load(fs[0])
    return (np.asarray(z["positions"], dtype=np.float64),
            np.asarray(z["particle_ids"]),
            np.asarray(z["element_ids"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline"); ap.add_argument("divfree"); ap.add_argument("out")
    ap.add_argument("--plane", default="yz", choices=list(PLANES))
    ap.add_argument("--bbox", type=float, nargs=6, default=None,
                    help="xmin xmax ymin ymax zmin zmax clip, in metres")
    ap.add_argument("--point-size", type=float, default=0.4)
    a = ap.parse_args()

    xa, pa, ea = load(Path(a.baseline))
    xb, pb, eb = load(Path(a.divfree))
    # ⚠️ match by particle_id, never by array position
    ia, ib = np.argsort(pa), np.argsort(pb)
    common, ca, cb = np.intersect1d(pa[ia], pb[ib], return_indices=True)
    xa, xb = xa[ia][ca], xb[ib][cb]
    ea, eb = ea[ia][ca], eb[ib][cb]
    keep = (ea >= 0) & (eb >= 0)
    if a.bbox:
        lo = np.array(a.bbox[0::2]); hi = np.array(a.bbox[1::2])
        keep &= np.all((xa >= lo) & (xa <= hi), axis=1)
    xa, xb = xa[keep], xb[keep]
    print(f"plotting {len(xa):,} particles matched by id, alive in both"
          f"{' and inside bbox' if a.bbox else ''}")

    i, j, xl, yl = PLANES[a.plane]
    d = np.linalg.norm(xa - xb, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True)
    for ax, X, title in ((axes[0], xa, "raw P1 (baseline)"),
                         (axes[1], xb, "divergence-free")):
        ax.scatter(X[:, i], X[:, j], s=a.point_size, c="#2b6cb0",
                   linewidths=0, rasterized=True)
        ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect("equal", adjustable="box")

    # ⚠️ Panel 3 is the one that matters: colour by shift so SPATIAL structure
    # in the difference shows up. A uniform wash means the shift is scatter; a
    # gradient or a patch means the correction rearranged something locally.
    sc = axes[2].scatter(xa[:, i], xa[:, j], s=a.point_size, c=d,
                         cmap="viridis", linewidths=0, rasterized=True,
                         vmin=0, vmax=np.percentile(d, 99) if d.size else 1)
    axes[2].set_title(f"|shift| per particle  (p50 {np.median(d):.2e} m, "
                      f"p99 {np.percentile(d, 99):.2e} m)")
    axes[2].set_xlabel(xl); axes[2].set_ylabel(yl)
    axes[2].set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=axes[2], label="|shift| [m]")

    ext = float(np.linalg.norm(xa.max(axis=0) - xa.min(axis=0)))
    fig.suptitle(f"cylA final particle positions, {a.plane} projection — "
                 f"shift p50 {np.median(d)/ext*100:.2e} % of cloud extent",
                 fontsize=11)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
