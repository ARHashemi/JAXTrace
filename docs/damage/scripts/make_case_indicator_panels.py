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
}
ORDER = ["ln_phi", "C", "t_res", "w_bond"]


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
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(10.6, 1.85 * n_panels + 0.9))
    axes = np.atleast_1d(axes)

    # ── panel 0: Group ──────────────────────────────────────────────────
    a = axes[0]
    if gval is not None:
        gm = wake_mask(gpos)
        a.scatter(gpos[gm, 1] * 1e3, gpos[gm, 2] * 1e3, c=gval[gm], s=0.6,
                  cmap="tab10", vmin=0, vmax=9, linewidths=0, rasterized=True)
        a.set_title("Group — seeding slab each particle came from. "
                    "A defect is a region NO slab reaches.",
                    fontsize=9, color="#1a3a5c", loc="left")
    else:
        a.text(0.5, 0.5, "Group not available in this run's VTU exports",
               transform=a.transAxes, ha="center", va="center",
               fontsize=9, color="#a33")
        a.set_title("Group — unavailable", fontsize=9, color="#a33", loc="left")

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
        cb.set_label(spec["label"], fontsize=7.5, color=INK2)
        cb.ax.tick_params(labelsize=6.5, colors=INK2)
        suffix = (" ⚠️ median is 0 where most neighbours are 0 → reads as coverage"
                  if spec["scale"] == "linear_zero" else "")
        a.set_title(f"{spec['label']} — k-NN median{suffix}",
                    fontsize=9, color="#1a3a5c", loc="left")
        pi += 1

    # ── the better t_res panel: entrained fraction ──────────────────────
    if extra_tres:
        v = d[m, cols.index("t_res")]
        frac = knn(y, zz, v, gy, gz, k, "frac")
        a = axes[pi]
        im = a.pcolormesh(gy, gz, frac.T, cmap="cividis", vmin=0.0, vmax=1.0,
                          shading="auto", rasterized=True)
        cb = fig.colorbar(im, ax=a, fraction=0.014, pad=0.010)
        cb.set_label("entrained fraction", fontsize=7.5, color=INK2)
        cb.ax.tick_params(labelsize=6.5, colors=INK2)
        a.set_title("entrained FRACTION — share of the k neighbours with "
                    "$t_\\mathrm{res}>0$; maps the stir-zone boundary directly",
                    fontsize=9, color="#1a3a5c", loc="left")
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
