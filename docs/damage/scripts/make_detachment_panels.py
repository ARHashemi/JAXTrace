"""Detachment panels: yz projection, same convention as the indicator panels.

    python3 make_detachment_panels.py --npz <detach run>.npz --tag D2 \
            [--baseline <pre-detachment run>.npz] [--out figs_detach]

VIEWING CONVENTION -- identical to make_case_indicator_panels.py so the figures
are directly comparable: the observer sits at the origin looking downstream
(+x), which puts +y on the LEFT, so the y axis is INVERTED via
set_xlim(yhi, ylo). Drawing it the other way mirrors the weld.

⚠️ THE WAKE CLIP IS PER CASE AND SCALED, NOT COPIED.
make_case_indicator_panels.py uses x >= 7.4 mm for D2, chosen to sit behind the
tool. cylA's domain is much longer downstream (x to 50.3 mm vs D2's 29.3), so
7.4 mm there keeps 99.5 % of particles and isolates nothing. The clip is
therefore scaled by the same FRACTION of the downstream extent:
    D2   7.4 / 29.33 = 0.2523  ->  7.4 mm
    cylA 0.2523 * 50.34        -> 12.7 mm
Pass --x-cut-mm to override.

WHAT THE PANELS SHOW
  1 Group          the reference view, particles coloured by seed group
  2 Detached       the flag -- which particles the criterion fired on
  3 Pi_1 min       the continuous field the flag thresholds
  4 ln_J / det F   the volume-change diagnostic
With --baseline, a fifth panel shows the BASELINE group view, so the
detachment run and the previous tracking sit side by side.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

# D2's reference clip, and the fraction it represents of D2's downstream reach.
D2_X_CUT_MM = 7.4
D2_X_MAX_MM = 29.328
DZ_FRAC = 0.08          # drop the top slice: the shoulder-contact layer


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    cols = [str(c) for c in np.asarray(z["columns"]).tolist()]
    out = dict(
        pos=np.asarray(z["positions"], dtype=np.float64) * 1e3,   # mm
        dmg=np.asarray(z["damage"], dtype=np.float64),
        cols=cols,
        eid=np.asarray(z["element_ids"]) if "element_ids" in z.files else None,
        detached=(np.asarray(z["detached"]).astype(bool)
                  if "detached" in z.files else None),
    )
    for k in ("detach_pi1_crit", "detach_mode", "detach_v_adv"):
        if k in z.files:
            out[k] = z[k].item() if z[k].ndim == 0 else z[k]
    return out


def wake_mask(pos: np.ndarray, x_cut: float) -> np.ndarray:
    x, zz = pos[:, 0], pos[:, 2]
    return (x >= x_cut) & (zz <= zz.max() - DZ_FRAC * np.ptp(zz))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, default=Path("figs_detach"))
    ap.add_argument("--x-cut-mm", type=float, default=None)
    ap.add_argument("--n-groups", type=int, default=5)
    args = ap.parse_args()

    d = load(args.npz)
    if d["detached"] is None:
        print(f"  {args.npz}: no 'detached' array — not a detachment run")
        return 1

    pos = d["pos"]
    # ⚠️ Scale the clip unless told otherwise.
    if args.x_cut_mm is not None:
        x_cut = args.x_cut_mm
        why = "user-specified"
    else:
        frac = D2_X_CUT_MM / D2_X_MAX_MM
        x_cut = frac * pos[:, 0].max()
        why = f"{frac:.4f} x x_max (D2-equivalent fraction)"
    m = wake_mask(pos, x_cut)
    print(f"  {args.tag}: {m.sum():,}/{len(m):,} particles in the wake box "
          f"(x >= {x_cut:.2f} mm, {why})")

    y, zz = pos[m, 1], pos[m, 2]
    det = d["detached"][m]
    cols = d["cols"]
    tau = d["dmg"][m, cols.index("tau_min")] if "tau_min" in cols else None
    lnJ = d["dmg"][m, cols.index("ln_J")] if "ln_J" in cols else None

    # Seed group from the INITIAL x, the same definition run_tracking uses.
    # ⚠️ Groups are by initial position, so they must come from a field that
    # survives the run; positions here are FINAL, so we fall back to ordering
    # by particle index, which for a grid seed is monotone in x.
    gid = (np.arange(len(m))[m] * args.n_groups) // len(m)

    base = load(args.baseline) if args.baseline else None
    n_panels = 4 + (1 if base is not None else 0)

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(7.2, 3.0 * n_panels), constrained_layout=True)
    ylo, yhi = y.min(), y.max()
    zlo, zhi = zz.min(), zz.max()

    # ── 1 Group ─────────────────────────────────────────────────────────
    a = axes[0]
    a.scatter(y, zz, c=gid, cmap="tab10", s=6, alpha=0.25, lw=0)
    a.scatter(y, zz, c=gid, cmap="tab10", s=2, lw=0)
    a.set_title(f"{args.tag} — seed group (detachment run)", fontsize=10)

    # ── 2 Detached flag ─────────────────────────────────────────────────
    a = axes[1]
    a.scatter(y[~det], zz[~det], c="#cfd8dc", s=2, lw=0, label="attached")
    if det.any():
        a.scatter(y[det], zz[det], c="#c0392b", s=10, lw=0,
                  label=f"detached ({det.sum():,})")
    crit = d.get("detach_pi1_crit", float("nan"))
    a.set_title(f"DETACHED — $\\Pi_1<{crit:g}$, "
                f"{100*det.mean():.3f} % of the wake box", fontsize=10)
    a.legend(loc="upper right", fontsize=7, markerscale=2, framealpha=0.9)

    # ── 3 Pi_1 min ──────────────────────────────────────────────────────
    a = axes[2]
    if tau is not None:
        fin = np.isfinite(tau)
        # ⚠️ Plot only finite values: +inf means the particle never sampled
        # valid material, and a colour scale including inf renders as blank.
        o = np.argsort(tau[fin])
        sc = a.scatter(y[fin][o], zz[fin][o], c=tau[fin][o], s=2, lw=0,
                       cmap="viridis")
        fig.colorbar(sc, ax=a, pad=0.01).set_label(
            r"$\Pi_1^{\min}=\min\,\sigma_{eq}/\tau_{yield}^{ref}$", fontsize=8)
        a.set_title(r"$\Pi_1^{\min}$ along the path "
                    f"({int(fin.sum()):,} finite)", fontsize=10)

    # ── 4 det F ─────────────────────────────────────────────────────────
    a = axes[3]
    if lnJ is not None:
        detF = np.exp(np.clip(lnJ, -5, 5))
        # ⚠️ Clip ln_J before exp: it reaches +-1800 on these runs, which
        # overflows to inf/0 and makes the panel unreadable. The clip is a
        # DISPLAY choice and is stated in the title.
        sc = a.scatter(y, zz, c=detF, s=2, lw=0, cmap="RdBu_r",
                       norm=TwoSlopeNorm(vcenter=1.0,
                                         vmin=max(detF.min(), 1e-2),
                                         vmax=min(detF.max(), 1e2)))
        fig.colorbar(sc, ax=a, pad=0.01).set_label(r"$\det F=e^{\ln J}$",
                                                   fontsize=8)
        a.set_title(r"$\det F$  (⚠️ $\ln J$ clipped to $\pm5$ for display; "
                    "the field is NOT divergence-free)", fontsize=9)

    # ── 5 baseline comparison ───────────────────────────────────────────
    if base is not None:
        a = axes[4]
        bpos = base["pos"]
        bm = wake_mask(bpos, x_cut)
        bgid = (np.arange(len(bm))[bm] * args.n_groups) // len(bm)
        a.scatter(bpos[bm, 1], bpos[bm, 2], c=bgid, cmap="tab10",
                  s=6, alpha=0.25, lw=0)
        a.scatter(bpos[bm, 1], bpos[bm, 2], c=bgid, cmap="tab10", s=2, lw=0)
        a.set_title(f"{args.tag} — BASELINE (no detachment), "
                    f"{bm.sum():,} in box", fontsize=10)

    for a in axes:
        # Observer at the origin looking +x, so +y is on the LEFT.
        a.set_xlim(yhi, ylo)
        a.set_ylim(zlo, zhi)
        a.set_aspect("equal")
        a.set_ylabel("z [mm]", fontsize=8)
        a.tick_params(labelsize=7)
    axes[-1].set_xlabel("y [mm]   (+y left: observer at origin looking +x)",
                        fontsize=8)

    args.out.mkdir(parents=True, exist_ok=True)
    o = args.out / f"fig_detach_{args.tag}.png"
    fig.savefig(o, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"  wrote {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
