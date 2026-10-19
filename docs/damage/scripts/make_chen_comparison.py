"""Chen (2023) mechanism comparison figure."""
import csv, glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
MPa = 1e-6
d = {}
for p in sorted(glob.glob(os.path.join(src, "dp_*.csv"))):
    t = os.path.basename(p)[3:-4]
    r = list(csv.DictReader(open(p)))
    d[t] = dict(B=np.median([float(x["B_Pa"]) for x in r])*MPa,
                f=np.median([float(x["p_front_Pa"]) for x in r])*MPa,
                rr=np.median([float(x["p_rear_Pa"]) for x in r])*MPa)
tags = sorted(d, key=lambda t: -d[t]["B"])
FC = {"A": "#4c78a8", "B": "#f58518", "C": "#54a24b", "D": "#e45756"}

fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.0))

# -- LEFT: Chen's own two cases, his quantity --------------------------------
a = ax[0]
chen = [("sound\n600-20", 30.0, 30.0), ("voided\n600-180\nz=1mm", 50.0, 15.0),
        ("voided\n600-180\nz=2mm", 42.0, 13.0)]
xs = np.arange(len(chen))
a.bar(xs-0.18, [c[1] for c in chen], 0.34, label="front (θ=0°)", color="#1a3a5c")
a.bar(xs+0.18, [c[2] for c in chen], 0.34, label="rear (θ=180°)", color="#c0392b")
for i, c in enumerate(chen):
    a.annotate("", xy=(i+0.18, c[2]), xytext=(i-0.18, c[1]),
               arrowprops=dict(arrowstyle="->", color="#555", lw=1.3))
    a.text(i, max(c[1], c[2])+2.5, f"drop {c[1]-c[2]:.0f}", ha="center",
           fontsize=8.5, color="#555")
a.set_xticks(xs); a.set_xticklabels([c[0] for c in chen], fontsize=8)
a.set_ylabel("frictional shear stress τ [MPa]", fontsize=9)
a.set_title("Chen et al. (2023) — AA6061-T6, 600 rpm\n"
            "their quantity: τ at the tool interface",
            fontsize=9.5, color="#1a3a5c", loc="left")
a.legend(fontsize=8); a.grid(axis="y", alpha=.25, lw=.5)
a.tick_params(labelsize=8)

# -- RIGHT: ours, our quantity ------------------------------------------------
a = ax[1]
y = np.arange(len(tags))
for i, t in enumerate(tags):
    a.plot([d[t]["rr"], d[t]["f"]], [i, i], color="#bbb", lw=1.2, zorder=1)
a.scatter([d[t]["f"] for t in tags], y, s=34, color="#1a3a5c",
          label="front (θ=0°)", zorder=3)
a.scatter([d[t]["rr"] for t in tags], y, s=34, color="#c0392b",
          label="rear (θ=180°)", zorder=3)
a.axvline(0, color="#333", lw=1.0, ls=":")
# ⚠️ Label placed INSIDE the axes at the bottom: at y=-1.25 it sat in the
# title band and collided with the subplot title.
a.text(0, len(tags)-0.3, "rear < 0 = TENSILE", fontsize=7.5, color="#c0392b",
       ha="center", va="top")
# ⚠️ Annotate to the LEFT of the rear marker, not right of the front one:
# right of the front ran D2's label off the panel entirely.
for i, t in enumerate(tags):
    if d[t]["rr"] < 0:
        a.text(d[t]["rr"]-0.35, i, "rear collapse →", fontsize=7.5,
               color="#c0392b", va="center", ha="right")
a.set_xlim(left=min(d[t]["rr"] for t in tags)-3.4)
a.set_yticks(y)
a.set_yticklabels([f"{t}  (B={d[t]['B']:.1f})" for t in tags], fontsize=7.5)
a.invert_yaxis()
a.set_xlabel("pressure in the contact annulus [MPa]", fontsize=9)
a.set_title("Ours — AA6063, 600 rpm, 16 cases\n"
            "our quantity: PRESSURE in the material",
            fontsize=9.5, color="#1a3a5c", loc="left")
a.legend(fontsize=8, loc="lower right"); a.grid(axis="x", alpha=.25, lw=.5)
a.tick_params(labelsize=8)

fig.suptitle("Chen's mechanism tested on our cases: does the REAR collapse where B is high?",
             fontsize=11.5, color="#1a3a5c", x=0.01, ha="left")
fig.text(0.01, -0.035,
         "⚠️ The two panels are NOT on a common axis and the numbers must not be "
         "compared: Chen reports frictional SHEAR STRESS at the tool interface, "
         "ours is a PRESSURE difference in an annulus in the material.\n"
         "Only the PATTERN transfers — a front-high / rear-collapsed signature. "
         "It appears in 4 of our cases (A2, A3, B4, D2); the top four by B are "
         "D2, A2, A3, D3, so the two agree on 14 of 16 cases.",
         fontsize=8, color="#555")
fig.tight_layout()
fig.savefig(out, dpi=165, bbox_inches="tight", facecolor="white")
print(f"  wrote {out}")
