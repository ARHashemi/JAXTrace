"""Plot raw vs flux-normalised density, and locate where they differ.

⚠️ WHAT THE NUMBERS LEFT OPEN. On cylA the two measures came out nearly
identical (4.56 % vs 4.76 % of voxels below 30 % of median), which admits two
readings: either the deficits are REAL, or both are dominated by
CLOUD-BOUNDARY voxels that are only partly filled and read low whatever we
divide by. These panels separate those, by masking the boundary and by
showing WHERE the two measures disagree.

Panels, on a z-slab through the wake:
  1. raw particle density n
  2. flux-normalised n/|u|          (same colour scale as 1)
  3. speed |u|                      (the thing being divided out)
  4. ratio (n/|u|) / n, normalised  -- i.e. 1/|u| up to a constant: where the
     normalisation ACTS
  5. deficit mask: voxels below the threshold, INTERIOR only
  6. the same with the boundary ring shown, so the reader can see how much of
     the deficit was an edge artefact

Usage:
    python3 plot_flux_normalised_density.py --run DIR --case-post DIR \
        --pattern 'cylA_{timestep}.pvtu' --timestep 159 --out fig.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.colors import LogNorm                            # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.damage.flux_normalised_density import (             # noqa: E402
    load_final, speed_on_grid)


def interior_mask(occ: np.ndarray) -> np.ndarray:
    """Occupied voxels whose 6 face-neighbours are ALL occupied.

    ⚠️ This is the control the raw numbers lacked. A voxel at the edge of the
    particle slab is only partly filled and reads low for a purely geometric
    reason, with nothing to do with the flow. Requiring a full neighbourhood
    removes that class.
    """
    m = occ.copy()
    for ax in (0, 1, 2):
        for sh in (+1, -1):
            m &= np.roll(occ, sh, axis=ax)
    # np.roll wraps; kill the outer shell explicitly
    m[0] = m[-1] = False
    m[:, 0] = m[:, -1] = False
    m[:, :, 0] = m[:, :, -1] = False
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--case-post", required=True, type=Path)
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--timestep", type=int, required=True)
    ap.add_argument("--field", default="Displacement")
    ap.add_argument("--res", type=int, default=40)
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    P = load_final(a.run)
    lo, hi = P.min(axis=0), P.max(axis=0)
    pad = 0.02 * (hi - lo); lo, hi = lo - pad, hi + pad
    ext = hi - lo
    bins = np.maximum(np.round(a.res * ext / ext.max()).astype(int), 4)
    H, edges = np.histogramdd(P, bins=list(bins), range=list(zip(lo, hi)))
    ctr = [0.5 * (e[1:] + e[:-1]) for e in edges]
    centres = np.stack(np.meshgrid(*ctr, indexing="ij"), axis=-1).reshape(-1, 3)
    spd = speed_on_grid(a.case_post, a.pattern, a.timestep, centres,
                        a.field).reshape(H.shape)

    occ = H >= a.min_count
    inter = interior_mask(occ)
    print(f"  {len(P):,} particles, grid {tuple(bins)}")
    print(f"  occupied {int(occ.sum()):,}  interior-only {int(inter.sum()):,} "
          f"({100*inter.sum()/max(occ.sum(),1):.1f} % of occupied)")

    u = np.maximum(spd, 1e-30)
    rho = np.where(occ, H, np.nan)
    rho_hat = np.where(occ, H / u, np.nan)
    rho /= np.nanmedian(rho[inter]) if inter.any() else np.nanmedian(rho)
    rho_hat /= np.nanmedian(rho_hat[inter]) if inter.any() else np.nanmedian(rho_hat)

    # ⚠️ THE NUMBER THAT ANSWERS THE OPEN QUESTION
    for nm, V in (("raw n", rho), ("flux-norm n/|u|", rho_hat)):
        all_def = float(np.nanmean(V[occ] < a.thr))
        int_def = float(np.nanmean(V[inter] < a.thr)) if inter.any() else float("nan")
        print(f"  {nm:16s} below {a.thr:.0%}: all-occupied "
              f"{100*all_def:5.2f} %   INTERIOR ONLY {100*int_def:5.2f} %")

    k = int(np.argmax(occ.sum(axis=(0, 1))))     # fullest z slab
    X, Y = np.meshgrid(ctr[0], ctr[1], indexing="ij")
    sl = lambda A: A[:, :, k]                                     # noqa: E731

    fig, ax = plt.subplots(2, 3, figsize=(16.5, 8.4), constrained_layout=True)
    vmax = float(np.nanpercentile(np.concatenate(
        [rho[occ].ravel(), rho_hat[occ].ravel()]), 99))
    for a_, V, t in ((ax[0, 0], rho, "raw density  n"),
                     (ax[0, 1], rho_hat, r"flux-normalised  $n/|u|$")):
        im = a_.pcolormesh(X, Y, sl(V), cmap="viridis", vmin=0, vmax=vmax,
                           shading="auto")
        a_.set_title(t + "  (/median)", fontsize=10)
        fig.colorbar(im, ax=a_)

    im = ax[0, 2].pcolormesh(X, Y, sl(np.where(occ, u, np.nan)),
                             cmap="magma", shading="auto",
                             norm=LogNorm())
    ax[0, 2].set_title(r"speed $|u|$  (what is divided out)", fontsize=10)
    fig.colorbar(im, ax=ax[0, 2])

    # where the normalisation ACTS = ratio of the two maps
    ratio = np.where(occ, rho_hat / np.maximum(rho, 1e-30), np.nan)
    im = ax[1, 0].pcolormesh(X, Y, sl(ratio), cmap="coolwarm", shading="auto",
                             vmin=0, vmax=2)
    ax[1, 0].set_title(r"$(n/|u|)\,/\,n$ — where normalisation acts",
                       fontsize=10)
    fig.colorbar(im, ax=ax[1, 0])

    # deficits: all occupied vs interior only
    for a_, M, t in (
            (ax[1, 1], occ, f"deficit < {a.thr:.0%}, ALL occupied"),
            (ax[1, 2], inter, f"deficit < {a.thr:.0%}, INTERIOR only")):
        base = np.where(M, 0.3, np.nan)
        d = np.where(M & (rho_hat < a.thr), 1.0, base)
        im = a_.pcolormesh(X, Y, sl(d), cmap="RdYlGn_r", vmin=0, vmax=1,
                           shading="auto")
        n_def = int(np.nansum((M & (rho_hat < a.thr))))
        a_.set_title(f"{t}   ({n_def:,} voxels)", fontsize=10)

    for a_ in ax.ravel():
        a_.set_xlabel("x [m]", fontsize=8); a_.set_ylabel("y [m]", fontsize=8)
        a_.tick_params(labelsize=7); a_.set_aspect("equal", adjustable="box")
    fig.suptitle(f"{a.tag} flux-normalised density, z-slab {k+1}/{bins[2]} "
                 f"— the normalisation changes the map only where |u| varies",
                 fontsize=11)
    fig.savefig(a.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
