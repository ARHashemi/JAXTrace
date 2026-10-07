"""Side-by-side: old grid-seeded particle tracking vs new damage indicator (O50).

    python3 make_pt_vs_damage.py                 # every case that has both
    python3 make_pt_vs_damage.py --cases B2 C1

Two independent predictions of the same thing, on the same yz cross-section:

  TOP     old particle-tracking runs in each case's `post_pt/` folder --
          288,000 GRID-seeded particles, coloured by `Group`, the x-slab each
          particle was seeded in. A defect shows as a region no slab reaches.
  BOTTOM  the new pathline-integrated damage, ln(Phi/Phi0).

⚠️ `Group` is a SEEDING LABEL, not a damage value. The five groups are consecutive
x-slabs of material entering the tool (verified: at step 0 they occupy
x = -19.9..-11.1 mm in five bands, full y and z). Colouring by it shows how those
sheets are folded by the tool, and -- the point here -- where none of them arrive.

⚠️ Only 13 of 20 PinShapes cases have a `post_pt` folder (A1, A2old, A3old, A4old,
B1-B4, C1-C4, D3). Cases without one are skipped rather than half-plotted.

⚠️ The two rows do NOT share a particle count or a seeding scheme (288k grid vs
100k random), so compare WHERE features sit, not how dense the clouds look.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

PT_ROOT = "/home/arhashemi/lumi/lumi_scratch/lorenzgl/Cases/PinShapes"
DMG_ROOT = "/home/arhashemi/lumi/lumi_scratch/hashemia/damage/phase4_rk4_30mm"
FAMILIES = {
    "A": "A-FlatsVariations", "B": "B-FluteVariations",
    "C": "C-ThreadsVariations", "D": "D-ConcavityTilt",
}
# ⚠️ TWO different upstream walls, deliberately:
#   X_AFTER_TOOL_MM = 7.4  -- the DAMAGE panel, matching make_wake_metrics.py and
#       make_density_vs_damage.py so every damage figure in the report uses one
#       protocol (O51: 7.0 left a rind of tool-attached particles at 46x the bulk
#       damage; 7.4 clears it for ~0.1 % of the population).
#   --x-min = 11.5         -- the PT panel. The PT run projects EVERY particle beyond
#       the wall onto one plane rather than taking a median, so its wall is set
#       further out to keep the near-tool cap out of the projection entirely.
# They are not interchangeable and neither should be "unified" onto the other.
X_AFTER_TOOL_MM = 7.4
DZ_FRAC = 0.08


def find_pt(case: str):
    fam = FAMILIES.get(case[0])
    if fam is None:
        return None
    runs = sorted(glob.glob(os.path.join(PT_ROOT, fam, f"{case}.gid",
                                         "post_pt", "run_*")))
    if not runs:
        return None
    vtus = glob.glob(os.path.join(runs[-1], "particles_step_*.vtu"))
    if not vtus:
        return None
    # Highest step index = the final state.
    return max(vtus, key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]))


def read_pt(path):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy as v2n
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(path)
    r.Update()
    g = r.GetOutput()
    P = v2n(g.GetPoints().GetData()) * 1e3
    G = v2n(g.GetPointData().GetArray("Group")).astype(float)
    return P, G


def read_damage(case: str):
    fam = FAMILIES.get(case[0])
    p = os.path.join(DMG_ROOT, f"ps_{fam}_{case}", "damage_models.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return z["positions"] * 1e3, z["damage"][:, 0]


def wake(P, v):
    """Everything in the wake box, projected along x. Used for the DAMAGE panel."""
    x, y, zz = P[:, 0], P[:, 1], P[:, 2]
    m = (x >= X_AFTER_TOOL_MM) & (zz <= zz.max() - DZ_FRAC * (zz.max() - zz.min()))
    return y[m], zz[m], v[m]


# ⚠️ PER-CASE PT wall. B2's bundle finished much closer to the tool than every other
# case: its x_max is 24.6 mm against 29.4-29.7 for the rest, and only 33.5 % of its
# particles lie beyond the default 11.5 mm wall (74-76 % for the others). It has
# 52,976 particles in the 6-8 mm band where B1 has 9,377. Using the common wall would
# discard two thirds of its cloud and leave the left of the panel empty.
#
# ⚠️ B2 is the ONLY such case -- checked against all 13. D3 also has a short x_max
# (22.7 mm) but retains 71.7 %, i.e. a normal fraction, so it keeps the default.
PT_WALL_OVERRIDE = {"B2": 6.0}


def wake_pt(P, v, x_min_mm):
    """Everything in the PT box (x >= x_min_mm), projected along x onto the yz plane.

    ⚠️ This projects ~70,000 particles through ~13 mm of x, so a (y,z) cell holds a
    median of 16 particles and up to 82. That overplotting is accepted deliberately:
    the box must contain the same material the damage panel integrates over, and a
    thin slab would show only one station.

    It is made legible two ways rather than by discarding particles:
      * particles are drawn in order of DECREASING x, so the ones nearest the tool
        (which carry the folded structure) land on top rather than being buried;
      * the marker is small enough that the lattice does not merge into a solid
        field.
    """
    x, y, zz = P[:, 0], P[:, 1], P[:, 2]
    zcut = zz.max() - DZ_FRAC * (zz.max() - zz.min())
    m = (x >= x_min_mm) & (zz <= zcut)
    xs, ys, zs, vs = x[m], y[m], zz[m], v[m]
    o = np.argsort(-xs)          # far particles first, near ones drawn last
    return ys[o], zs[o], vs[o], float(x_min_mm)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--out", default="figs_compare")
    # ⚠️ 11.5 mm is 4.5 mm clear of the 7 mm shoulder edge, so the projection
    # contains no particle still attached to the tool.
    ap.add_argument("--x-min", type=float, default=11.5,
                    help="upstream wall [mm] of the PT box; every particle beyond "
                         "it is projected onto yz (default 11.5 = 0.0115 m)")
    # ⚠️ ONE case per figure. Each case needs two stacked panels (PT above, damage
    # below), so even two cases make four and the image shrinks to ~940 px on a
    # slide -- too small to read the folded layer structure that is the point.
    ap.add_argument("--per-fig", type=int, default=1,
                    help="cases per figure; families larger than this are split "
                         "across numbered figures so nothing is crammed")
    args = ap.parse_args()

    wanted = args.cases or ["A1", "A2old", "A3old", "A4old",
                            "B1", "B2", "B3", "B4",
                            "C1", "C2", "C3", "C4", "D3"]
    rows = []
    for c in wanted:
        pt = find_pt(c)
        dm = read_damage(c)
        if pt is None:
            print(f"  SKIP {c}: no post_pt run")
            continue
        if dm is None:
            print(f"  SKIP {c}: no damage result")
            continue
        P, G = read_pt(pt)
        Pd, L = dm
        # PT: a thin cross-section at the box midpoint. Damage: the whole box,
        # unchanged, because its estimator is a median over each cell.
        wall = PT_WALL_OVERRIDE.get(c, args.x_min)
        rows.append((c, wake_pt(P, G, wall), wake(Pd, L)))
        print(f"  {c:7s} pt {P.shape[0]:,} particles   damage {Pd.shape[0]:,}")

    if not rows:
        print("  nothing to plot")
        return 1
    os.makedirs(args.out, exist_ok=True)

    # Split by family, then into chunks so no figure is overcrowded.
    byfam: dict[str, list] = {}
    for r in rows:
        byfam.setdefault(FAMILIES[r[0][0]], []).append(r)
    for fam, items in byfam.items():
        for k in range(0, len(items), args.per_fig):
            chunk = items[k:k + args.per_fig]
            part = f"_{k // args.per_fig + 1}" if len(items) > args.per_fig else ""
            make_fig(fam, chunk, os.path.join(
                args.out, f"figP_{fam}{part}_pt_vs_damage.png"), args.x_min)
    return 0


def make_fig(fam, items, out, x_min):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from scipy.spatial import cKDTree

    n = len(items)
    # Two rows per case: particle tracking on top, damage below.
    # ⚠️ Width scales with the panel count so the ASPECT stays slide-friendly.
    # A fixed width makes a 2-case figure taller than it is wide (aspect 1.04),
    # which then has to be shrunk to ~450 px on a slide — the opposite of the
    # intent. Targeting aspect ~0.5 keeps the image wide and the panels large.
    panel_h = 1.75
    fig_h = panel_h * 2 * n + 0.6
    fig, axes = plt.subplots(2 * n, 1, figsize=(fig_h / 0.5, fig_h), squeeze=False)
    axes = axes.ravel()
    INK2 = "#555555"

    alll = np.concatenate([it[2][2] for it in items])
    lo = max(float(np.percentile(alll[alll > 0], 5)), 1e-3)
    hi = float(np.percentile(alll, 99.0))

    # ⚠️ One shared extent for every panel in the figure, taken from the damage
    # data (the full wake box). A PT slab contains only the particles that reached
    # that station, so letting it set its own limits makes the rows disagree -- the
    # two panels of a pair must be read against each other.
    ylo = min(float(it[2][0].min()) for it in items)
    yhi = max(float(it[2][0].max()) for it in items)
    zlo = min(float(it[2][1].min()) for it in items)
    zhi = max(float(it[2][1].max()) for it in items)

    im_d = None
    for i, (case, (yg, zg, G, xc), (yd, zd, L)) in enumerate(items):
        # --- top: grid-seeded particles coloured by seed slab ---
        a = axes[2 * i]
        # A thin slab holds ~41k particles rather than ~186k, so the markers can be
        # large enough to read the slab colours without overplotting.
        # ⚠️ marker="o" is matplotlib's default, but it is stated explicitly here:
        # at this zoom the grid-seeded lattice can read as square cells, and the
        # explicit round marker plus a small edge makes each particle legible as a
        # discrete sphere rather than part of a mesh.
        a.scatter(yg, zg, c=G, s=4.0, cmap="tab10", vmin=0, vmax=9,
                  marker="o", linewidths=0.0, rasterized=True)
        a.set_facecolor("white")
        a.text(0.006, 0.94,
               f"{case}  ·  particle tracking",
               transform=a.transAxes, fontsize=8.5, va="top", ha="left",
               fontweight="bold", color="#1a1a1a",
               bbox=dict(boxstyle="round,pad=0.22", fc="#ffffffcc", ec="none"))

        # --- bottom: damage, kNN median at display resolution ---
        a = axes[2 * i + 1]
        t = cKDTree(np.column_stack([yd, zd]))
        gy = np.linspace(yd.min(), yd.max(), 500)
        gz = np.linspace(zd.min(), zd.max(), 200)
        GY, GZ = np.meshgrid(gy, gz, indexing="ij")
        d, idx = t.query(np.column_stack([GY.ravel(), GZ.ravel()]), k=16, workers=-1)
        f = np.median(np.log10(np.maximum(L[idx], 1e-4)), axis=1)
        f[d[:, -1] > 0.45] = np.nan
        im_d = a.pcolormesh(gy, gz, np.clip(np.power(10.0, f).reshape(500, 200).T,
                                            lo, hi),
                            cmap="inferno", norm=LogNorm(vmin=lo, vmax=hi),
                            shading="auto", rasterized=True)
        a.set_facecolor("#d9d9d9")
        a.text(0.006, 0.94, f"{case}  ·  damage  ln(Φ/Φ₀)",
               transform=a.transAxes, fontsize=8.5, va="top", ha="left",
               fontweight="bold", color="white",
               bbox=dict(boxstyle="round,pad=0.22", fc="#00000088", ec="none"))

        for a in (axes[2 * i], axes[2 * i + 1]):
            # ⚠️ VIEWING DIRECTION. Both panels are drawn as seen by an observer at
            # the origin looking DOWNSTREAM, along +x. In a right-handed (x, y, z)
            # frame that puts +y on the LEFT, so the y axis is inverted: xlim runs
            # (yhi, ylo) rather than (ylo, yhi). Matplotlib is orthographic, so
            # there is no perspective to switch off.
            #
            # Drawn the other way (+y on the right) the picture would be the view
            # looking back UPSTREAM, and the advancing and retreating sides would
            # swap places on the page -- which is exactly the kind of silent
            # left-right error this comment exists to prevent.
            a.set_xlim(yhi, ylo)
            a.set_ylim(zlo, zhi)
            # 1 mm in y is 1 mm in z: the weld is 30 mm wide and ~5.5 mm deep.
            a.set_aspect("equal", adjustable="box")
            a.set_ylabel("z [mm]", fontsize=8, color=INK2)
            a.tick_params(labelsize=7, colors=INK2)
            for sp in a.spines.values():
                sp.set_color("#cccccc")

    axes[-1].set_xlabel("y [mm]", fontsize=9.5, color=INK2)
    cb = fig.colorbar(im_d, ax=axes.tolist(), fraction=0.012, pad=0.015)
    cb.set_label("ln(Φ/Φ₀)", fontsize=9, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    # ⚠️ No suptitle: the slide heading already names the family, and the figure
    # title only consumes vertical space the panels could use.
    # No in-figure footer: the slide caption and the report both carry the
    # explanation, and repeating it here steals vertical space from the panels.
    fig.savefig(out, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
