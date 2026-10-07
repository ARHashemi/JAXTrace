"""Follow individual particles: where and when damage is acquired (O41).

    python3 make_trajectory_figures.py --case <dir> [--out figs_traj]

⚠️ PARTICLE IDENTITY IS VERIFIED, not assumed. `particle_ids = np.arange(n)` is
created ONCE at seeding (run_tracking.py:774) and passed unchanged to every export;
the kernel is a vmap over a fixed array and never reorders. Three independent
checks on ps_A1 agree:

  * ParticleID identical across consecutive files
  * row-wise displacement over 50 steps: median 0.18 mm, vs 12.98 mm for a
    shuffled control (72x separation)
  * lnPhi decreased in 0 of 100,000 rows -- and lnPhi can only accumulate, so any
    row swap would show up as a decrease. This is a physical invariant, not a
    heuristic, and is the strongest of the three.

Grid vs random seeding makes no difference to identity: the ID is a positional
index, not a tracked label.

WHAT IT PRODUCES
----------------
  figT_<case>_3d.png     3D trajectories coloured by instantaneous ln(Phi/Phi0)
  figT_<case>_time.png   lnPhi vs time for the same particles, with tool
                         revolutions marked -- "when", against the 3D "where"

SELECTION (--n-per-stratum each)
--------------------------------
Sampling purely at random would mostly draw from the ~60 % bypass population and
produce flat lines. The strata are chosen so the figure shows the MECHANISM:

  high    top decile of final lnPhi      -- went through the shear layer
  mid     median decile                  -- the typical processed particle
  low     bottom decile (bypass)         -- the control: what "no damage" looks like
  jumpy   largest single-step increment  -- isolates WHERE damage is acquired

⚠️ These are not a random sample and the figure must not be read as a population
statistic. It is a mechanism illustration; the population numbers are in the wake
CSV.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np


def read_series(case_dir: str, stride: int = 1):
    """Positions and damage for every exported step, as (T, N, 3) and (T, N).

    ⚠️ Step 0 carries no damage arrays -- it is the initial state, written before
    any tracking step -- so it is skipped rather than special-cased downstream.
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy as v2n

    files = sorted(glob.glob(os.path.join(case_dir, "particles_step_*.vtu")))
    if not files:
        raise SystemExit(f"  no particles_step_*.vtu in {case_dir}")

    steps, P, L, C = [], [], [], []
    sel = files[::stride]
    n_sel = len(sel)
    for k, f in enumerate(sel):
        # D-Concavity has ~795 exports (O44) and reading them over the LUMI mount
        # takes ~25 min, so report progress rather than appearing hung.
        if n_sel > 200 and k % 100 == 0:
            print(f"    reading {k}/{n_sel} ...", flush=True)
        r = vtk.vtkXMLUnstructuredGridReader()
        r.SetFileName(f)
        r.Update()
        g = r.GetOutput()
        pd = g.GetPointData()
        if pd.GetArray("Damage_lnPhi") is None:
            continue                      # step 0: initial state, no damage yet
        steps.append(int(re.search(r"_(\d+)\.vtu$", f).group(1)))
        P.append(v2n(g.GetPoints().GetData()).astype(np.float32))
        L.append(v2n(pd.GetArray("Damage_lnPhi")).astype(np.float32))
        c = pd.GetArray("Damage_CL")
        C.append(v2n(c).astype(np.float32) if c else np.zeros_like(L[-1]))
    return np.array(steps), np.stack(P), np.stack(L), np.stack(C)


def pick(L: np.ndarray, pos: np.ndarray, k: int, rng) -> dict:
    """Stratified particle indices. See the module docstring for the rationale."""
    final = L[-1]
    order = np.argsort(final)
    n = len(final)
    out = {}

    def take(pool):
        pool = np.asarray(pool)
        return rng.choice(pool, size=min(k, len(pool)), replace=False)

    out["high"] = take(order[int(0.90 * n):])
    out["mid"] = take(order[int(0.45 * n):int(0.55 * n)])
    out["low"] = take(order[:int(0.10 * n)])
    # Largest single-step increment: isolates WHERE damage is picked up, which a
    # final-value ranking cannot show (a high final value could accrue slowly).
    jump = np.diff(L, axis=0).max(axis=0)
    out["jumpy"] = take(np.argsort(jump)[-max(k * 5, 50):])

    # ⚠️ TRAPPED particles: still inside the shoulder radius at the final step.
    # None of the strata above target them -- "high final damage" selects particles
    # that were captured and RELEASED -- yet they are 18.7 % of the D-Concavity run
    # against 2.6-2.8 % elsewhere (O43), and whether they orbit indefinitely is the
    # question that decides if D's void is physical. Empty for a healthy case,
    # which is itself the control.
    r_final = np.hypot(pos[-1, :, 0], pos[-1, :, 1]) * 1e3
    trapped = np.flatnonzero(r_final < 7.0)
    if trapped.size:
        out["trapped"] = take(trapped)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, help="one case directory with VTUs")
    ap.add_argument("--out", default="figs_traj")
    ap.add_argument("--n-per-stratum", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1,
                    help="read every Nth exported file (default all)")
    ap.add_argument("--rpm", type=float, default=None,
                    help="signed RPM, to mark tool revolutions on the time plot")
    ap.add_argument("--dt", type=float, default=None, help="solver dt [s]")
    ap.add_argument("--seed", type=int, default=0)
    # ⚠️ Reading 167 VTU files over the LUMI mount takes ~8 min. The parsed arrays
    # are ~200 MB as float32, so caching them makes re-plotting (different strata,
    # different styling) instant instead of re-reading every time.
    ap.add_argument("--cache", default=None,
                    help="npz to read/write the parsed series "
                         "(default <out>/_cache_<case>.npz; --cache none disables)")
    args = ap.parse_args()

    tag = os.path.basename(args.case.rstrip("/"))
    os.makedirs(args.out, exist_ok=True)
    cache = args.cache
    if cache is None:
        cache = os.path.join(args.out, f"_cache_{tag}.npz")
    if cache != "none" and os.path.exists(cache):
        z = np.load(cache)
        steps, P, L, C = z["steps"], z["P"], z["L"], z["C"]
        print(f"  loaded cache {cache}")
    else:
        steps, P, L, C = read_series(args.case, args.stride)
        if cache != "none":
            np.savez_compressed(cache, steps=steps, P=P, L=L, C=C)
            print(f"  cached -> {cache}")
    print(f"  {tag}: {len(steps)} exported states, {P.shape[1]:,} particles")

    rng = np.random.default_rng(args.seed)
    sel = pick(L, P, args.n_per_stratum, rng)
    for k, v in sel.items():
        print(f"    {k:6s} n={len(v)}  final lnPhi "
              f"{', '.join('%.3g' % L[-1, i] for i in v)}")

    make_3d(tag, steps, P, L, sel, args.out)
    make_time(tag, steps, P, L, sel, args.out, args.rpm, args.dt)
    return 0


COLORS = {"high": "#c0392b", "mid": "#e08214", "low": "#4a6fa5",
          "jumpy": "#1baf7a", "trapped": "#7b4397"}
LABEL = {"high": "top decile (through the shear layer)",
         "mid": "median decile (typical)",
         "low": "bottom decile (bypass — control)",
         "jumpy": "largest single-step jump",
         "trapped": "STILL INSIDE r < 7 mm at the final step"}


def make_3d(tag, steps, P, L, sel, outdir):
    """Two 2D projections, stacked: x-y (top) and x-z (side).

    ⚠️ NOT a 3D axes. matplotlib's 3D tick labels are drawn INSIDE the axes box and
    cannot be given equal data aspect without the z labels colliding with whatever
    sits beside them -- two attempts at a side-by-side 3D + 2D layout both collided.
    The paths are read for WHERE the damage is picked up, and two true-scale
    orthographic views answer that better than one distorted perspective.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.collections import LineCollection

    allsel = np.concatenate(list(sel.values()))
    vmin = max(float(np.percentile(L[-1][allsel], 5)), 1e-3)
    vmax = float(max(L[-1][allsel].max(), vmin * 10))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    allxyz = P[:, allsel, :].reshape(-1, 3) * 1e3

    fig, (axt, axs) = plt.subplots(2, 1, figsize=(12.5, 6.6))
    for grp, idx in sel.items():
        for i in idx:
            xyz = P[:, i, :] * 1e3
            v = np.maximum(L[:, i], vmin)
            for axis, (a, b) in ((axt, (0, 1)), (axs, (0, 2))):
                pt = xyz[:, [a, b]].reshape(-1, 1, 2)
                sg = np.concatenate([pt[:-1], pt[1:]], axis=1)
                lc = LineCollection(sg, cmap="inferno", norm=norm, linewidths=1.5)
                lc.set_array(v[:-1])
                axis.add_collection(lc)
                axis.plot(xyz[0, a], xyz[0, b], "o", color="#2e7d32", ms=4.5)
                axis.plot(xyz[-1, a], xyz[-1, b], "s", color="#1a1a1a", ms=4.5)

    # The tool, so the capture orbits can be read against real geometry.
    th = np.linspace(0, 2 * np.pi, 180)
    for r, c, lbl in ((2.4, "#c0392b", "pin r = 2.4 mm"),
                      (7.0, "#888888", "shoulder r = 7 mm")):
        axt.plot(r * np.cos(th), r * np.sin(th), color=c, lw=1.1, ls="--",
                 zorder=0, label=lbl)
        for sgn in (-1, 1):
            axs.axvline(sgn * r, color=c, lw=1.0, ls="--", alpha=0.7, zorder=0)

    axt.set_xlim(allxyz[:, 0].min(), allxyz[:, 0].max())
    axt.set_ylim(allxyz[:, 1].min(), allxyz[:, 1].max())
    axt.set_ylabel("y [mm]", fontsize=9)
    axt.set_title("top view (x–y) — capture orbits around the tool",
                  fontsize=9.5, color="#1a3a5c", loc="left")
    axt.legend(fontsize=7.5, loc="upper left", frameon=False)

    axs.set_xlim(allxyz[:, 0].min(), allxyz[:, 0].max())
    axs.set_ylim(allxyz[:, 2].min(), allxyz[:, 2].max())
    axs.set_ylabel("z [mm]", fontsize=9)
    axs.set_xlabel("x [mm]      (advance direction →)", fontsize=9)
    axs.set_title("side view (x–z) — depth at which damage is acquired",
                  fontsize=9.5, color="#1a3a5c", loc="left")

    for axis in (axt, axs):
        # 1 mm is 1 mm on both axes: the domain is 40 mm long and 6 mm deep, so
        # letting z stretch would misrepresent where the paths actually run.
        axis.set_aspect("equal", adjustable="box")
        axis.tick_params(labelsize=7.5)
        axis.grid(alpha=0.22, lw=0.5)

    sm = plt.cm.ScalarMappable(cmap="inferno", norm=norm)
    cb = fig.colorbar(sm, ax=(axt, axs), fraction=0.016, pad=0.015)
    cb.set_label("ln(Φ/Φ₀) accumulated at that point on the path", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"Particle trajectories coloured by accumulated damage — {tag}",
                 fontsize=11, color="#1a3a5c", x=0.01, ha="left")
    fig.text(0.01, 0.012,
             "● seed    ■ final position.   Colour is the damage accumulated SO "
             "FAR, so a path brightens exactly where damage is picked up.",
             fontsize=7.5, color="#555555")
    p = os.path.join(outdir, f"figT_{tag}_3d.png")
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {p}")


def make_time(tag, steps, P, L, sel, outdir, rpm, dt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axa, axb) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                                   gridspec_kw=dict(height_ratios=[2, 1]))
    for grp, idx in sel.items():
        for j, i in enumerate(idx):
            axa.plot(steps, np.maximum(L[:, i], 1e-4), color=COLORS[grp],
                     lw=1.4, alpha=0.9,
                     label=LABEL[grp] if j == 0 else None)
            r = np.hypot(P[:, i, 0], P[:, i, 1]) * 1e3
            axb.plot(steps, r, color=COLORS[grp], lw=1.1, alpha=0.8)
    axa.set_yscale("log")
    axa.set_ylabel("ln(Φ/Φ₀)  accumulated", fontsize=9)
    axa.legend(fontsize=7.5, loc="upper left", frameon=False)
    axa.grid(alpha=0.25, lw=0.5)
    axb.axhspan(0, 2.4, color="#c0392b", alpha=0.10)
    axb.axhline(7.0, color="#888888", lw=0.9, ls="--")
    axb.text(steps[-1], 7.2, "shoulder r = 7 mm", fontsize=7,
             color="#555555", ha="right")
    axb.text(steps[-1], 0.6, "pin", fontsize=7, color="#c0392b", ha="right")
    axb.set_ylabel("distance from tool axis [mm]", fontsize=9)
    axb.set_xlabel("tracking step", fontsize=9)
    axb.grid(alpha=0.25, lw=0.5)

    # ⚠️ Revolution marks only when rpm and dt are supplied, and only if they are
    # coarse enough to see: at 166 steps/rev over 8267 steps there are ~50 of
    # them, so they are drawn faintly and only every 5th.
    if rpm and dt:
        spr = (60.0 / abs(rpm)) / dt
        nrev = int(steps[-1] / spr)
        for k in range(0, nrev + 1, max(1, nrev // 10)):
            for a in (axa, axb):
                a.axvline(k * spr, color="#999999", lw=0.5, alpha=0.5, zorder=0)
        axa.set_title(f"{tag}   —   {spr:.0f} steps per tool revolution, "
                      f"{steps[-1] / spr:.0f} revolutions total",
                      fontsize=9.5, color="#1a3a5c", loc="left")
    else:
        axa.set_title(tag, fontsize=9.5, color="#1a3a5c", loc="left")

    fig.suptitle("When is the damage acquired?", fontsize=11,
                 color="#1a3a5c", x=0.01, ha="left")
    fig.text(0.01, 0.005,
             "Upper: accumulated damage. Lower: the same particles' distance from "
             "the tool axis — a step in the upper panel should coincide with a "
             "close approach in the lower one.",
             fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    p = os.path.join(outdir, f"figT_{tag}_time.png")
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {p}")


if __name__ == "__main__":
    raise SystemExit(main())
