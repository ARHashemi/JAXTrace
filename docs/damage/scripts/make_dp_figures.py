"""Figures for the contact-pressure differential B (A.4).

    python3 make_dp_figures.py --src <dp_csv_dir> [--out figs_dp]
    python3 make_dp_figures.py --src <dir> --field-case <case>.gid   # + xy map

⚠️ B IS NOT A FIELD. It reduces ~465,000 annulus points to TWO numbers per
snapshot — the maximum pressure in front of the tool and the minimum behind it
(Shi et al., Eq. 19) — and then to one scalar. So B cannot be plotted the way
ln(Phi), C, t_res and w are: there is no per-particle or per-cell value to map.

What B *is*:
  * a TIME SERIES, one value per snapshot  -> B(t) per case
  * a per-case SCALAR (its median)         -> a ranking bar across the 16 cases

⚠️ AND THE CROSS-SECTION PLANE IS NOT yz. Every other figure in this project
uses a yz cut, because the pathline indicators are compared in the weld
cross-section. B is a FRONT-TO-REAR difference along the travel direction +x,
so a yz plane — which is a plane of constant x — cannot contain both ends of it.
The plane B actually lives in is:

  ✅ xy at the pin mid-depth : contains theta = 0 (front) and theta = 180 (rear)
                               simultaneously. This is the correct map.
  ✅ xz through y = 0        : shows the same front/rear contrast against depth.
  ❌ yz                      : perpendicular to B. Would show the
                               advancing/retreating contrast instead, which is a
                               DIFFERENT quantity (measured -12.47 MPa median on
                               D2 where B is +16.80 — different magnitude AND
                               opposite sign).

So this script draws the pressure FIELD in the xy plane, with the two sampling
sectors marked, so a reader can see where the two numbers come from — and the
B(t) series and ranking bar, which are what B itself supports.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

MPa = 1.0e-6
SHI_THRESHOLD = 15.0        # MPa; Shi et al. Eq. 19, Fig. 21 — voids ABOVE this
FAMILY = {"A": "A-Flats", "B": "B-Flutes", "C": "C-Threads", "D": "D-Concavity"}
FAM_COLOUR = {"A": "#4c78a8", "B": "#f58518", "C": "#54a24b", "D": "#e45756"}


def read_csvs(src: str) -> dict:
    out = {}
    for p in sorted(glob.glob(os.path.join(src, "dp_*.csv"))):
        tag = os.path.basename(p)[3:-4]
        rows = list(csv.DictReader(open(p)))
        if not rows:
            continue
        out[tag] = dict(
            B=np.array([float(r["B_Pa"]) for r in rows]) * MPa,
            p_front=np.array([float(r["p_front_Pa"]) for r in rows]) * MPa,
            p_rear=np.array([float(r["p_rear_Pa"]) for r in rows]) * MPa,
            dp_ar=np.array([float(r.get("dp_adv_ret_Pa", "nan"))
                            for r in rows]) * MPa,
        )
    return out


def fig_series(data: dict, outdir: str) -> None:
    """B(t) for every case, grouped by family."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = sorted({t[0] for t in data})
    fig, axes = plt.subplots(len(fams), 1, figsize=(10.5, 2.3 * len(fams)),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, fams):
        for tag in sorted(t for t in data if t[0] == f):
            B = data[tag]["B"]
            ax.plot(np.arange(len(B)), B, lw=1.0, label=tag, alpha=0.9)
        # ⚠️ The threshold line is drawn, but voids form ABOVE it (Shi Eq. 19) --
        # the opposite of what an earlier draft of our own notes claimed.
        ax.axhline(SHI_THRESHOLD, color="#c0392b", ls="--", lw=1.0)
        ax.text(0.995, SHI_THRESHOLD, " Shi 15 MPa — voids ABOVE", color="#c0392b",
                fontsize=7, va="bottom", ha="right", transform=ax.get_yaxis_transform())
        ax.set_ylabel(f"{FAMILY.get(f, f)}\nB [MPa]", fontsize=8.5, color="#555")
        ax.legend(fontsize=7, ncol=4, loc="lower right", framealpha=0.9)
        ax.tick_params(labelsize=7.5, colors="#555")
        ax.grid(alpha=0.25, lw=0.5)
    axes[-1].set_xlabel("snapshot index (one tool revolution spans several)",
                        fontsize=8.5, color="#555")
    fig.suptitle("Contact-pressure difference  B = P$_{max}$(front) − P$_{min}$(rear)",
                 fontsize=11, color="#1a3a5c", x=0.01, ha="left")
    out = os.path.join(outdir, "fig_dp_series.png")
    fig.savefig(out, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_ranking(data: dict, outdir: str) -> None:
    """Per-case median B with its spread — the quantity that ranks the cases."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = sorted(data)
    med = np.array([np.median(data[t]["B"]) for t in tags])
    p10 = np.array([np.percentile(data[t]["B"], 10) for t in tags])
    p90 = np.array([np.percentile(data[t]["B"], 90) for t in tags])
    order = np.argsort(med)[::-1]

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    for rank, i in enumerate(order):
        t = tags[i]
        ax.barh(rank, med[i], color=FAM_COLOUR.get(t[0], "#888"), height=0.68)
        ax.plot([p10[i], p90[i]], [rank, rank], color="#333", lw=1.4,
                solid_capstyle="butt")
    ax.axvline(SHI_THRESHOLD, color="#c0392b", ls="--", lw=1.2)
    ax.text(SHI_THRESHOLD, -1.0, " Shi: voids ABOVE 15 MPa", color="#c0392b",
            fontsize=8, va="top")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([tags[i] for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("median B [MPa]   (bar = median, line = p10–p90 over snapshots)",
                  fontsize=9, color="#555")
    ax.tick_params(labelsize=7.5, colors="#555")
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAM_COLOUR.values()]
    ax.legend(handles, [FAMILY[k] for k in FAM_COLOUR], fontsize=8, loc="lower right")
    fig.suptitle("Case ranking by contact-pressure difference B",
                 fontsize=11, color="#1a3a5c", x=0.01, ha="left")
    out = os.path.join(outdir, "fig_dp_ranking.png")
    fig.savefig(out, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_xy_map(case: str, outdir: str, field: str, step: int,
               r_pin: float, band: float, sector_deg: float,
               workers: int) -> None:
    """The pressure field in the xy plane at pin mid-depth, sectors marked.

    ⚠️ This is the plane B lives in. A yz cut cannot show a front-to-rear
    difference, because yz is a plane of constant x.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    import re
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from jaxtrace.gpu.mesh_loader_timedep import load_field_sequences_from_pvtu

    post = case if os.path.basename(case) == "post" else os.path.join(case, "post")
    cands = sorted(glob.glob(os.path.join(post, "*.pvtu")))
    stems = {re.sub(r"_\d+\.pvtu$", "", os.path.basename(f)) for f in cands
             if re.search(r"_\d+\.pvtu$", os.path.basename(f))}
    stem = sorted(stems)[0]
    pos, _c, flds = load_field_sequences_from_pvtu(
        base_path=Path(post), file_pattern=stem + "_{timestep}.pvtu",
        timestep_range=(step, step), field_names=[field],
        verbose=False, n_workers=workers)
    p = np.asarray(flds[field]).reshape(-1) * MPa
    pos = np.asarray(pos, dtype=np.float64)

    # Slab at pin mid-depth: thin enough to be a plane, thick enough to be dense.
    zc = 0.5 * (pos[:, 2].min() + pos[:, 2].max())
    half = 0.25e-3
    m = np.abs(pos[:, 2] - zc) <= half
    x, y = pos[m, 0] * 1e3, pos[m, 1] * 1e3
    v = p[m]

    fig, ax = plt.subplots(figsize=(9.0, 7.2))
    lim = float(np.percentile(np.abs(v), 99))
    sc = ax.scatter(x, y, c=np.clip(v, -lim, lim), s=1.2, cmap="RdBu_r",
                    norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                    linewidths=0, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.012)
    cb.set_label(f"{field} [MPa]", fontsize=9)

    # Mark the two sampling sectors so the reader sees where B comes from.
    th = np.linspace(-sector_deg / 2, sector_deg / 2, 40)
    for base, lab, col in ((0.0, "FRONT  θ=0°  → P$_{max}$", "#111"),
                           (180.0, "REAR  θ=180°  → P$_{min}$", "#111")):
        a0 = np.radians(base + th)
        for rr in (r_pin, r_pin * (1 + band)):
            ax.plot(rr * np.cos(a0) * 1e3, rr * np.sin(a0) * 1e3,
                    color=col, lw=1.4)
        for e in (0, -1):
            ax.plot([r_pin * np.cos(a0[e]) * 1e3,
                     r_pin * (1 + band) * np.cos(a0[e]) * 1e3],
                    [r_pin * np.sin(a0[e]) * 1e3,
                     r_pin * (1 + band) * np.sin(a0[e]) * 1e3],
                    color=col, lw=1.4)
        rm = r_pin * (1 + band / 2)
        ax.annotate(lab, (rm * np.cos(np.radians(base)) * 1e3,
                          rm * np.sin(np.radians(base)) * 1e3),
                    fontsize=8, ha="center", va="center", color="white",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#000000aa", ec="none"))

    ax.annotate("", xy=(0.14, 1.02), xytext=(0.02, 1.02),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#1a3a5c", lw=1.6))
    ax.text(0.155, 1.015, "travel  +x", transform=ax.transAxes,
            fontsize=8.5, color="#1a3a5c")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]  (travel direction)", fontsize=9, color="#555")
    ax.set_ylabel("y [mm]", fontsize=9, color="#555")
    ax.tick_params(labelsize=7.5, colors="#555")
    fig.suptitle(f"{os.path.basename(case)} — pressure in the xy plane at pin "
                 f"mid-depth, step {step}\nB is the FRONT−REAR difference, so this "
                 f"is the plane it lives in (a yz cut cannot show it)",
                 fontsize=10.5, color="#1a3a5c", x=0.01, ha="left")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"fig_dp_xy_{os.path.basename(case)}.png")
    fig.savefig(out, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="dir holding dp_<case>.csv")
    ap.add_argument("--out", default="figs_dp")
    ap.add_argument("--field-case", default=None,
                    help="<case>.gid to draw the xy pressure map from")
    ap.add_argument("--field-step", type=int, default=100)
    ap.add_argument("--pressure-field", default="Pressure")
    ap.add_argument("--r-pin", type=float, default=3.0e-3)
    ap.add_argument("--band", type=float, default=0.6)
    ap.add_argument("--sector-deg", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = read_csvs(args.src)
    if data:
        print(f"  {len(data)} cases")
        fig_series(data, args.out)
        fig_ranking(data, args.out)
        print()
        print(f"  {'case':6s} {'median B':>10s} {'p10':>8s} {'p90':>8s}  "
              f"{'% above 15':>10s}")
        for t in sorted(data, key=lambda k: -np.median(data[k]["B"])):
            B = data[t]["B"]
            print(f"  {t:6s} {np.median(B):>10.2f} "
                  f"{np.percentile(B, 10):>8.2f} {np.percentile(B, 90):>8.2f}  "
                  f"{100 * (B > SHI_THRESHOLD).mean():>9.0f} %")
    else:
        print(f"  no dp_*.csv under {args.src}")

    if args.field_case:
        fig_xy_map(args.field_case, args.out, args.pressure_field,
                   args.field_step, args.r_pin, args.band, args.sector_deg,
                   args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
