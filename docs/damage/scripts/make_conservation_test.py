"""Particle conservation across a control surface around the tool (O46, test 1).

    python3 make_conservation_test.py

THE TEST. In a steady incompressible flow the number of particles inside a fixed
control volume around the tool must reach a steady state -- inflow balances outflow.
The volume fills while the seeded bundle arrives, then DRAINS as material passes
through and emerges downstream. A count that rises and then holds means particles are
entering and not leaving, which is a conservation failure in the TRACKING, not weld
physics.

⚠️ This is the direct test for the void artefact (O46: void area correlates with
trapped fraction at rho = +0.98). It needs NO new simulation -- the exported
trajectory series already contains everything.

Reads the caches written by `make_trajectory_figures.py`, so run that for a case
first. Reference result, ps_A-FlatsVariations_A1 (a healthy case):

    peak 22,914 -> end 2,584   (11 % of peak)
    late-run trend  -88 particles per export   -> DRAINING
    5.8 % of the 44,632 particles that ever entered are still inside

A case that fails this test will show a late-run trend near zero or positive, and a
high "still inside" fraction.
"""
import numpy as np, os, sys, glob

CACHE = "figs_traj"

def load(tag):
    p = os.path.join(CACHE, f"_cache_{tag}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z["steps"], z["P"]

def report(tag, R=7.0):
    d = load(tag)
    if d is None:
        print(f"  {tag}: no cache yet"); return
    steps, P = d
    r = np.hypot(P[:, :, 0], P[:, :, 1]) * 1e3
    inside = (r < R)
    n_in = inside.sum(axis=1)
    # net flux per export interval
    flux = np.diff(n_in.astype(int))
    print(f"\n  === {tag} ===")
    print(f"  control volume r < {R} mm, {len(steps)} exported states")
    print(f"  particles inside: start {n_in[0]}  peak {n_in.max()} "
          f"(at step {steps[n_in.argmax()]})  end {n_in[-1]}")
    print(f"  end / peak = {n_in[-1]/max(n_in.max(),1):.3f}   "
          f"end / start = {n_in[-1]/max(n_in[0],1):.2f}")
    # Has it drained? Compare the last quarter's trend.
    q = len(n_in) // 4
    late = n_in[-q:]
    slope = np.polyfit(np.arange(len(late)), late, 1)[0]
    print(f"  late-run trend: {slope:+.2f} particles per export "
          f"({'DRAINING' if slope < -0.5 else 'STEADY/ACCUMULATING'})")
    # A particle that enters and never leaves: first entry vs last exit
    ever_in = inside.any(axis=0)
    still_in = inside[-1]
    never_left = still_in.sum()
    print(f"  entered the control volume at some point : {ever_in.sum():,}")
    print(f"  still inside at the final step           : {never_left:,} "
          f"({100*never_left/max(ever_in.sum(),1):.1f}% of those that entered)")

def make_fig(tags, out="figs_compare/figC_conservation.png"):
    """Count inside the control volume vs time, one line per case.

    ⚠️ Normalised by each case's PEAK, not by the particle count: the cases differ
    in how much of the bundle ever reaches the tool, and the question here is
    whether what entered gets OUT, not how much entered.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.4))
    for t in tags:
        d = load(t)
        if d is None:
            continue
        steps, P = d
        r = np.hypot(P[:, :, 0], P[:, :, 1]) * 1e3
        n = (r < 7.0).sum(axis=1).astype(float)
        bad = "D-Concavity" in t
        ax.plot(steps / steps[-1], n / max(n.max(), 1),
                lw=2.2 if bad else 1.5,
                color="#c0392b" if bad else "#4a6fa5",
                label=t.replace("ps_", "").replace("_", " "),
                zorder=3 if bad else 2)
    ax.set_xlabel("fraction of the run", fontsize=9)
    ax.set_ylabel("particles inside r < 7 mm\n(÷ that case's peak)", fontsize=9)
    ax.set_title("Does the control volume around the tool DRAIN?",
                 fontsize=10.5, color="#1a3a5c", loc="left")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, frameon=False)
    fig.text(0.01, -0.02,
             "A steady incompressible flow must balance inflow and outflow: the "
             "volume fills as the bundle arrives, then drains. A curve that rises "
             "and HOLDS is a conservation failure in the tracking.",
             fontsize=7.5, color="#555555")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=165, bbox_inches="tight", facecolor="white")
    print(f"\n  wrote {out}")


tags = sorted(os.path.basename(p)[7:-4]
              for p in glob.glob(os.path.join(CACHE, "_cache_*.npz")))
for t in tags:
    report(t)
if len(tags) >= 2:
    make_fig(tags)
else:
    print("\n  (figure needs >= 2 cached cases; run make_trajectory_figures.py "
          "for another case first)")
