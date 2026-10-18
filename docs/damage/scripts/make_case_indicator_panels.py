"""One figure per case: Group + every indicator on the same yz cross-section.

    python3 make_case_indicator_panels.py --src <run_root> [--cases D2 B2]
    python3 make_case_indicator_panels.py --src <root> --raw      # extra scatter set

LAYOUT, one figure per case
    panel 0   particles coloured by Group (the seeding slab each came from)
    panel 1+  one smoothed k-NN median field per indicator

⚠️ VIEWING DIRECTION -- the detail that is easiest to get silently wrong.
Every yz figure in this project is drawn as seen by an observer at the ORIGIN
looking DOWNSTREAM, along +x. In a right-handed (x, y, z) frame that puts +y on
the LEFT, so the y axis is INVERTED: xlim runs (yhi, ylo). Drawn the other way
the picture is the view looking back upstream and the advancing and retreating
sides swap sides of the page. make_pt_vs_damage.py carries the same comment for
the same reason; this script reproduces that convention exactly so the panels
can be laid beside the existing report figures without a mental flip.

⚠️ The user's framing was "Group viewed along +x, indicators viewed along -x then
flipped horizontally". Those two operations compose to the SAME on-page
orientation -- viewing from -x mirrors the page, and the horizontal flip mirrors
it back. So every panel here uses one convention, the project's existing one, and
no panel is flipped relative to another. Stating it explicitly because "flip one
of them" is exactly the kind of instruction that produces a left-right error
nobody notices until the advancing side is on the wrong side of a macrograph.

WHERE THE DATA COMES FROM
    Group              the run's own last-step VTU (particles_step_*.vtu), which
                       carries Group when the run was launched with N_GROUPS > 0.
                       ⚠️ Same run as the indicators -- not the older post_pt/
                       tracking that make_pt_vs_damage.py compares against, which
                       existed for only 13 of 20 cases and used a different seed.
    indicators         damage_models.npz, columns named in the archive itself.

⚠️ INDICATOR CHARACTER DIFFERS, so the colour treatment does too; see
make_indicator_figures.py for the measured justification. In brief: lnPhi and C
are log/sequential; t_res is linear with the exact zeros as their own category
(65.6 % of D2 wake particles never enter the shear layer, so a log scale is
impossible); w_bond is diverging about zero with asymmetric limits because it is
27 % negative and strongly right-skewed.

⚠️ t_res ALSO GETS A SECOND, BETTER PANEL. A k-NN MEDIAN of t_res is zero
wherever most neighbours are zero, so that panel reads as coverage rather than
intensity. The entrained FRACTION -- the share of the k neighbours with
t_res > 0 -- is the quantity that panel is really trying to show, and it maps the
stir-zone boundary directly. Both are drawn.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

X_AFTER_TOOL_MM = 7.4
DZ_FRAC = 0.08

SPEC = {
    "ln_phi": dict(label=r"$\ln(\Phi/\Phi_0)$ Rice–Tracey", scale="log",
                   cmap="inferno"),
    "C":      dict(label=r"$C$ Cockcroft–Latham", scale="log", cmap="inferno"),
    "t_res":  dict(label=r"$t_\mathrm{res}$ [s]", scale="linear_zero",
                   cmap="viridis"),
    "w_bond": dict(label=r"$w=\int p/\bar\sigma\,dt$", scale="diverging",
                   cmap="RdBu_r"),
    "A_edot": dict(label=r"$A_{\dot\varepsilon}=\int\dot\varepsilon\,dt$",
                   scale="log", cmap="inferno"),
}
# ⚠️ A_edot is plotted ONLY when the archive actually carries it as its own
# column. Runs before 2026-10-18 derived it in analysis as ln_phi/0.849, which
# is exact only if exp(1.5*eta) == 1 -- measured to vary ~35 % -- so a panel
# from the derived value would be the ln_phi panel rescaled by a constant and
# would show nothing new. Newer runs accumulate it independently.
ORDER = ["ln_phi", "C", "A_edot", "t_res", "w_bond"]


def wake_mask(pos):
    x, z = pos[:, 0] * 1e3, pos[:, 2] * 1e3
    return (x >= X_AFTER_TOOL_MM) & (z <= z.max() - DZ_FRAC * (z.max() - z.min()))


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    d = np.asarray(z["damage"], dtype=np.float64)
    pos = np.asarray(z["positions"], dtype=np.float64)
    cols = ([str(c) for c in np.asarray(z["columns"]).tolist()]
            if "columns" in z.files else ORDER[:d.shape[1]])
    thr = float(z["residence_edot_threshold"]) if \
        "residence_edot_threshold" in z.files else float("nan")
    return d, pos, cols, thr


def load_group(run_dir):
    """Group + positions from the run's LAST exported VTU, or (None, None)."""
    vtus = sorted(glob.glob(os.path.join(run_dir, "particles_step_*.vtu")))
    if not vtus:
        return None, None
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy as v2n
    except Exception:
        return None, None
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(vtus[-1])
    r.Update()
    out = r.GetOutput()
    arr = out.GetPointData().GetArray("Group")
    if arr is None:
        return None, None
    return v2n(out.GetPoints().GetData()).astype(float), v2n(arr).astype(float)


def knn(y, zz, v, gy, gz, k, reduce="median"):
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([y, zz]))
    GY, GZ = np.meshgrid(gy, gz, indexing="ij")
    _, idx = tree.query(np.column_stack([GY.ravel(), GZ.ravel()]),
                        k=min(k, len(y)), workers=-1)
    if idx.ndim == 1:
        idx = idx[:, None]
    nb = v[idx]
    out = (np.median(nb, axis=1) if reduce == "median"
           else (nb > 0).mean(axis=1))          # entrained fraction
    return out.reshape(GY.shape)


def limits(v, scale):
    if scale == "log":
        pos = v[v > 0]
        lo = max(float(np.percentile(pos, 5)), 1e-3) if pos.size else 1e-3
        return lo, float(np.percentile(v, 99.0))
    if scale == "diverging":
        neg, pos = v[v < 0], v[v > 0]
        lo = -float(np.percentile(np.abs(neg), 99.0)) if neg.size else -1e-12
        hi = float(np.percentile(pos, 99.0)) if pos.size else 1e-12
        return lo, hi
    return 0.0, float(np.percentile(v, 99.0))


def build(tag, npz_path, run_dir, outdir, nby, nbz, k, raw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    d, pos, cols, thr = load_npz(npz_path)
    m = wake_mask(pos)
    if m.sum() < 100:
        print(f"  {tag}: only {int(m.sum())} in wake box — skipped")
        return
    y, zz = pos[m, 1] * 1e3, pos[m, 2] * 1e3
    ylo, yhi, zlo, zhi = y.min(), y.max(), zz.min(), zz.max()
    gy = np.linspace(ylo, yhi, nby)
    gz = np.linspace(zlo, zhi, nbz)
    INK2 = "#555555"

    names = [c for c in ORDER if c in cols]
    extra_tres = "t_res" in names          # the entrained-fraction panel
    n_panels = 1 + len(names) + (1 if extra_tres else 0)

    gpos, gval = load_group(run_dir)
    # ⚠️ Height per panel and hspace are tuned together. At 1.85 in with default
    # spacing each axes' title landed on top of the panel above and the x tick
    # labels bled through it. The weld is 30 mm wide and ~5.5 mm deep, so at
    # equal aspect a panel is inherently short and wide and the title needs its
    # own band of whitespace.
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(11.4, 1.95 * n_panels + 0.8))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(hspace=0.12, top=0.93, bottom=0.08)

    # ── panel 0: Group ──────────────────────────────────────────────────
    a = axes[0]
    if gval is not None:
        gm = wake_mask(gpos)
        ng = int(gval[gm].max()) + 1
        # ⚠️ Discrete colormap sized to the groups that ACTUALLY exist. With
        # tab10 at vmin=0/vmax=9 five of the ten colours went to absent groups
        # and the colourbar showed ticks for them; the used colours were also
        # compressed into half the ramp.
        import matplotlib as _mpl
        _gcmap = _mpl.colors.ListedColormap(
            _mpl.colormaps["tab10"].colors[:ng])
        # ⚠️ GAUSSIAN-SPHERE LOOK, as in ParaView. A flat scatter dot reads as a
        # pixel; ParaView's point gaussian renders each particle as a soft
        # sphere. Approximated by drawing a larger, slightly transparent disc
        # under a smaller opaque core -- cheap, and it gives the same impression
        # of volume without a 3-D renderer.
        yv, zv, cv = gpos[gm, 1] * 1e3, gpos[gm, 2] * 1e3, gval[gm]
        # ⚠️ Glyph size is a TRADE-OFF, not "bigger is better". At s=18 the
        # haloes merge into a solid sheet and both the slab structure and the
        # unreached gaps -- the whole point of this panel -- disappear. s=6/2.0
        # keeps the spherical impression while leaving the white voids legible.
        a.scatter(yv, zv, c=cv, s=6.0, cmap=_gcmap, vmin=-0.5, vmax=ng - 0.5,
                  linewidths=0, alpha=0.25, rasterized=True)
        sc_g = a.scatter(yv, zv, c=cv, s=2.0, cmap=_gcmap, vmin=-0.5,
                         vmax=ng - 0.5, linewidths=0, rasterized=True)
        # ⚠️ The Group panel MUST carry a colorbar too. Without one its axes
        # keeps the full figure width while every other panel is narrowed to
        # make room for theirs, so the panels do not line up -- which is exactly
        # the misalignment this comment exists to prevent.
        cb = fig.colorbar(sc_g, ax=a, fraction=0.014, pad=0.010,
                          ticks=list(range(ng)))
        cb.set_label("Group (seeding slab)", fontsize=7.5, color=INK2)
        cb.ax.tick_params(labelsize=6.5, colors=INK2)
    else:
        a.text(0.5, 0.5, "Group not available in this run's VTU exports",
               transform=a.transAxes, ha="center", va="center",
               fontsize=9, color="#a33")
        # Keep the axes the same width as the others even with nothing to show.
        cax = a.inset_axes([1.006, 0.0, 0.014, 1.0])
        cax.axis("off")

    # ── indicator panels ────────────────────────────────────────────────
    pi = 1
    for name in names:
        spec = SPEC[name]
        v = d[m, cols.index(name)]
        lo, hi = limits(v, spec["scale"])
        fld = knn(y, zz, v, gy, gz, k, "median")
        a = axes[pi]
        if spec["scale"] == "log":
            im = a.pcolormesh(gy, gz, np.clip(fld.T, lo, hi), cmap=spec["cmap"],
                              norm=LogNorm(vmin=lo, vmax=hi), shading="auto",
                              rasterized=True)
        elif spec["scale"] == "diverging":
            im = a.pcolormesh(gy, gz, np.clip(fld.T, lo, hi), cmap=spec["cmap"],
                              norm=TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi),
                              shading="auto", rasterized=True)
        else:
            im = a.pcolormesh(gy, gz, np.clip(fld.T, lo, hi), cmap=spec["cmap"],
                              vmin=0.0, vmax=hi, shading="auto", rasterized=True)
        cb = fig.colorbar(im, ax=a, fraction=0.014, pad=0.010)
        _warn = ("\n⚠️ 0 where most neighbours\nare 0 → coverage"
                 if spec["scale"] == "linear_zero" else "")
        cb.set_label(spec["label"] + _warn, fontsize=7.5, color=INK2)
        cb.ax.tick_params(labelsize=6.5, colors=INK2)
        # ⚠️ No per-panel title: the colorbar label already names the
        # indicator, and a title band per panel cost ~0.4 in of vertical space
        # each that the panels themselves could use.
        pi += 1

        # ⚠️ The entrained-FRACTION panel goes DIRECTLY under t_res, not at the
        # bottom of the figure. The median panel is misleading where most
        # neighbours are zero, and the fraction panel alone invites "where is
        # the actual residence time?" -- adjacent, each answers the other's
        # weakness. Appending it last put w_bond between them.
        if name == "t_res" and extra_tres:
            frac = knn(y, zz, v, gy, gz, k, "frac")
            a = axes[pi]
            im = a.pcolormesh(gy, gz, frac.T, cmap="cividis", vmin=0.0,
                              vmax=1.0, shading="auto", rasterized=True)
            cb = fig.colorbar(im, ax=a, fraction=0.014, pad=0.010)
            cb.set_label("entrained fraction\n(stir-zone boundary)",
                         fontsize=7.5, color=INK2)
            cb.ax.tick_params(labelsize=6.5, colors=INK2)
            pi += 1

    for a in axes:
        # ⚠️ Observer at the ORIGIN looking DOWNSTREAM (+x): +y on the LEFT.
        a.set_xlim(yhi, ylo)
        a.set_ylim(zlo, zhi)
        a.set_aspect("equal", adjustable="box")
        a.set_ylabel("z [mm]", fontsize=8, color=INK2)
        a.tick_params(labelsize=7, colors=INK2)
        for sp in a.spines.values():
            sp.set_color("#cccccc")
    # ⚠️ x tick labels ONLY on the bottom panel: every panel shares the same y
    # range, so repeating them six times just adds clutter that the titles then
    # have to compete with.
    for a in axes[:-1]:
        a.tick_params(labelbottom=False)
    axes[-1].set_xlabel("y [mm]   (+y left: viewed from the origin along +x)",
                        fontsize=8.5, color=INK2)
    thr_s = f"   ε̇ threshold {thr:g} 1/s" if np.isfinite(thr) else ""
    fig.suptitle(f"{tag} — wake box x ≥ {X_AFTER_TOOL_MM} mm, "
                 f"{int(m.sum()):,} particles{thr_s}",
                 fontsize=11, color="#1a3a5c", x=0.01, ha="left")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"fig_case_{tag}.png")
    fig.savefig(out, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")

    if raw:
        _raw_set(tag, y, zz, d[m], cols, names, outdir, ylo, yhi, zlo, zhi)


def _raw_set(tag, y, zz, dm, cols, names, outdir, ylo, yhi, zlo, zhi):
    """Backstage scatter panels: same data, no smoothing.

    Kept out of the main figure deliberately -- the smoothed fields are what the
    comparison is for -- but produced so the raw particles are available when
    someone asks whether a feature is an estimator artefact.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm, Normalize
    sub = os.path.join(outdir, "raw")
    os.makedirs(sub, exist_ok=True)
    for name in names:
        spec = SPEC[name]
        v = dm[:, cols.index(name)]
        lo, hi = limits(v, spec["scale"])
        fig, a = plt.subplots(figsize=(10.6, 2.6))
        if spec["scale"] == "log":
            o = np.argsort(v)
            sc = a.scatter(y[o], zz[o], c=np.clip(v[o], lo, hi), s=0.6,
                           cmap=spec["cmap"], norm=LogNorm(vmin=lo, vmax=hi),
                           linewidths=0, rasterized=True)
        elif spec["scale"] == "diverging":
            o = np.argsort(np.abs(v))
            sc = a.scatter(y[o], zz[o], c=np.clip(v[o], lo, hi), s=0.6,
                           cmap=spec["cmap"],
                           norm=TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi),
                           linewidths=0, rasterized=True)
        else:
            z0 = v <= 0
            a.scatter(y[z0], zz[z0], s=0.6, c="#c8c8c8", linewidths=0,
                      rasterized=True,
                      label=f"never entrained ({100*z0.mean():.0f} %)")
            o = np.argsort(v[~z0])
            sc = a.scatter(y[~z0][o], zz[~z0][o], c=np.clip(v[~z0][o], lo, hi),
                           s=0.6, cmap=spec["cmap"],
                           norm=Normalize(vmin=max(lo, 1e-12), vmax=hi),
                           linewidths=0, rasterized=True)
            a.legend(loc="upper right", fontsize=7, markerscale=6)
        fig.colorbar(sc, ax=a, fraction=0.014, pad=0.010).set_label(
            spec["label"], fontsize=7.5)
        a.set_xlim(yhi, ylo)
        a.set_ylim(zlo, zhi)
        a.set_aspect("equal", adjustable="box")
        a.set_title(f"{tag} — {spec['label']} — raw particles",
                    fontsize=9, color="#1a3a5c", loc="left")
        a.set_xlabel("y [mm]", fontsize=8)
        a.set_ylabel("z [mm]", fontsize=8)
        out = os.path.join(sub, f"raw_{tag}_{name}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"    + raw scatter set in {sub}/")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--out", default="figs_cases")
    ap.add_argument("--bins-y", type=int, default=600)
    ap.add_argument("--bins-z", type=int, default=240)
    ap.add_argument("--knn", type=int, default=16)
    ap.add_argument("--raw", action="store_true",
                    help="also write the backstage raw-scatter set")
    args = ap.parse_args()

    hits = (glob.glob(os.path.join(args.src, "*", "damage_*.npz"))
            + glob.glob(os.path.join(args.src, "*", "*", "damage_*.npz")))
    seen: dict = {}
    for p in sorted(hits, key=os.path.getmtime, reverse=True):
        seen.setdefault(os.path.relpath(p, args.src).split(os.sep)[0], p)
    items = sorted(seen.items())
    if args.cases:
        keep = set(args.cases)
        items = [(t, p) for t, p in items if t in keep]
    if not items:
        print("  nothing to plot")
        return 1
    for tag, p in items:
        build(tag, p, os.path.dirname(p), args.out,
              args.bins_y, args.bins_z, args.knn, args.raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
