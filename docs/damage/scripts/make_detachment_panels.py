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
# ⚠️ Vertical exaggeration of the yz panels. 1.0 would be true proportion, but
# the wake box is ~4.6x wider than deep and equal aspect wastes most of the
# figure. 2.5 fills the frame while keeping the stir zone recognisable.
ASPECT = 2.5


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


def read_group(run_dir: Path, n_expect: int):
    """Per-particle Group from the run's exported particles, or None.

    ✅ Group is the SEEDING SLAB each particle came from, assigned by
    run_tracking from the initial x. It is exported with the particles, so it
    is read rather than reconstructed.
    """
    import glob
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except Exception:
        return None
    vtus = sorted(glob.glob(str(run_dir / "**" / "particles_step_*.vtu"),
                            recursive=True))
    if not vtus:
        return None
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(vtus[-1])
    r.Update()
    arr = r.GetOutput().GetPointData().GetArray("Group")
    if arr is None:
        return None
    g = vtk_to_numpy(arr)
    # ⚠️ An export written at a different cadence can hold a different count;
    # silently broadcasting a mismatched array would mis-colour every point.
    return g if len(g) == n_expect else None


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

    # ⚠️ Group is assigned from the INITIAL x by run_tracking, and the npz
    # stores only FINAL positions -- so it cannot be reconstructed here. An
    # earlier version faked it by binning the particle INDEX, which is
    # meaningful only for a grid seed and is simply wrong for the femuss seed
    # cylA uses. Read the real array from the run's exported particles, as
    # make_case_indicator_panels.py does, and say so plainly when it is absent.
    gid = read_group(args.npz.parent, len(m))
    gid = gid[m] if gid is not None else None

    base = load(args.baseline) if args.baseline else None
    n_panels = 4 + (1 if base is not None else 0)

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(7.2, 3.0 * n_panels), constrained_layout=True)
    ylo, yhi = y.min(), y.max()
    zlo, zhi = zz.min(), zz.max()

    # ── 1 Group ─────────────────────────────────────────────────────────
    a = axes[0]
    if gid is not None:
        a.scatter(y, zz, c=gid, cmap="tab10", s=6, alpha=0.25, lw=0)
        a.scatter(y, zz, c=gid, cmap="tab10", s=2, lw=0)
        a.set_title(f"{args.tag} — seed group (detachment run)", fontsize=10)
    else:
        a.scatter(y, zz, c="#78909c", s=2, lw=0)
        a.set_title(f"{args.tag} — particles (⚠️ Group not in this run's "
                    f"exports; not reconstructable from the npz)", fontsize=9)

    # ── 2 Detached flag ─────────────────────────────────────────────────
    a = axes[1]
    a.scatter(y[~det], zz[~det], c="#cfd8dc", s=2, lw=0, label="attached")
    if det.any():
        a.scatter(y[det], zz[det], c="#c0392b", s=10, lw=0,
                  label=f"detached ({det.sum():,})")
    crit = d.get("detach_pi1_crit", float("nan"))
    if det.any():
        a.set_title(f"DETACHED — $\\Pi_1<{crit:g}$, "
                    f"{100*det.mean():.3f} % of the wake box", fontsize=10)
        a.legend(loc="upper right", fontsize=7, markerscale=2, framealpha=0.9)
    else:
        # ✅ An empty panel must state WHY, or it reads as a broken figure.
        a.set_title(f"DETACHED — NONE. $\\Pi_1^{{\\min}}$ never drops below "
                    f"{crit:g}", fontsize=10)
        a.text(0.5, 0.5,
               f"no particle detached\n"
               f"min $\\Pi_1$ in box = {np.nanmin(tau[np.isfinite(tau)]):.4f}"
               if tau is not None and np.isfinite(tau).any() else
               "no particle detached",
               transform=a.transAxes, ha="center", va="center",
               fontsize=11, color="#c0392b")

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
        # ⚠️ Scale on PERCENTILES, not min/max. det F has a heavy tail
        # (|det F - 1| reaches 1e2 on a handful of particles), so a min/max
        # range renders ~99 % of the field as flat white and hides the
        # structure the panel exists to show.
        _lo = float(np.percentile(detF, 2))
        _hi = float(np.percentile(detF, 98))
        _half = max(1.0 - _lo, _hi - 1.0, 1e-3)
        sc = a.scatter(y, zz, c=detF, s=2, lw=0, cmap="RdBu_r",
                       norm=TwoSlopeNorm(vcenter=1.0,
                                         vmin=1.0 - _half,
                                         vmax=1.0 + _half))
        fig.colorbar(sc, ax=a, pad=0.01).set_label(r"$\det F=e^{\ln J}$",
                                                   fontsize=8)
        a.set_title(r"$\det F$  (⚠️ $\ln J$ clipped to $\pm5$ for display; "
                    "the field is NOT divergence-free)", fontsize=9)

    # ── 5 baseline comparison ───────────────────────────────────────────
    if base is not None:
        a = axes[4]
        bpos = base["pos"]
        bm = wake_mask(bpos, x_cut)
        bgid = read_group(args.baseline.parent, len(bm))
        if bgid is not None:
            bgid = bgid[bm]
            a.scatter(bpos[bm, 1], bpos[bm, 2], c=bgid, cmap="tab10",
                      s=6, alpha=0.25, lw=0)
            a.scatter(bpos[bm, 1], bpos[bm, 2], c=bgid, cmap="tab10",
                      s=2, lw=0)
        else:
            a.scatter(bpos[bm, 1], bpos[bm, 2], c="#78909c", s=2, lw=0)
        a.set_title(f"{args.tag} — BASELINE (no detachment), "
                    f"{bm.sum():,} in box", fontsize=10)

    for a in axes:
        # Observer at the origin looking +x, so +y is on the LEFT.
        a.set_xlim(yhi, ylo)
        a.set_ylim(zlo, zhi)
        # ⚠️ NOT aspect='equal'. The wake box is ~46 mm wide by ~10 mm deep on
        # cylA, so an equal aspect letterboxes each panel into a thin strip
        # with most of the figure empty. A fixed moderate stretch keeps the
        # cross-section readable; z is still linear and labelled, so no
        # geometry is misrepresented -- only the drawn proportion.
        a.set_aspect(ASPECT)
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
