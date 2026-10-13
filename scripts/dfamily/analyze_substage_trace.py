#!/usr/bin/env python3
"""Which RK4 substage loses the particle, and what happens at that position?

Reads the npz written by `run_tracking.py --trace-substages`. Each record is
one particle-step in which at least one of the k1..k4 element searches failed,
with the four substage positions, the four element ids, and the step's final
position and element id.

The questions, in order:

  Q1  Which stage fails, and is it isolated or cascading? If k1 dominates, the
      particle entered the step already outside; if k2..k4 dominate, the step
      walked it out.
  Q2  Does the step RECOVER? A failed substage with a valid elem_final is
      harmless under last_valid_vel. A failed substage with elem_final < 0 is
      a lost particle.
  Q3  WHERE does the failing substage land -- inside the tool (LEVEL<0), above
      the free surface (z>0), outside the bbox, or in apparently valid
      material? Only the last is unexplained.
  Q4  How FAR outside does it go, and how does that compare with the distance
      one substage can travel? A substage that overshoots by much less than
      its own stride is a search problem; one that overshoots by its full
      stride is a physics/timestep problem.
  Q5  Does the loss accumulate over steps, i.e. do the same particles fail
      repeatedly, or is each failure a different particle?

Usage:
  python3 analyze_substage_trace.py <substages.npz> [case.gid]

With the case path it also loads the LEVEL field and the mesh bbox, which Q3
needs. Read-only.
"""
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

NPZ = Path(sys.argv[1])
CASE = Path(sys.argv[2]) if len(sys.argv) > 2 else None

d = np.load(NPZ)
step = d["step"]
particle = d["particle"]
sub_elem = d["sub_elem"]          # (n, 4)
sub_pos = d["sub_pos"]            # (n, 4, 3)
elem_final = d["elem_final"]      # (n,)
pos_final = d["pos_final"]        # (n, 3)
n = len(step)

STAGE = ["k1", "k2", "k3", "k4"]


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


print(f"{NPZ.name}: {n:,} particle-steps with at least one failed substage")
print(f"  steps covered    : {step.min()} .. {step.max()} "
      f"({len(np.unique(step))} logged steps)")
print(f"  distinct particles: {len(np.unique(particle)):,}")

bad = sub_elem < 0
nbad = bad.sum(axis=1)

# ---------------------------------------------------------------------------
banner("Q1  which stage fails, and is it isolated or cascading?")
# ---------------------------------------------------------------------------
print("  failures per stage (a particle-step can appear in several rows):")
for s in range(4):
    c = int(bad[:, s].sum())
    print(f"    {STAGE[s]}: {c:7,}  ({100.0 * c / n:5.1f}%)")

print("\n  how many stages failed together:")
for k in range(1, 5):
    c = int((nbad == k).sum())
    print(f"    exactly {k}: {c:7,}  ({100.0 * c / n:5.1f}%)")

print("\n  when exactly ONE stage failed, which one:")
one = nbad == 1
for s in range(4):
    c = int((one & bad[:, s]).sum())
    print(f"    only {STAGE[s]}: {c:7,}  "
          f"({100.0 * c / max(int(one.sum()), 1):5.1f}% of single failures)")

print("\n  first failing stage (the one that starts the cascade):")
first = np.where(bad.any(axis=1), bad.argmax(axis=1), -1)
cnt = Counter(int(x) for x in first)
for s in range(4):
    c = cnt.get(s, 0)
    print(f"    {STAGE[s]}: {c:7,}  ({100.0 * c / n:5.1f}%)")
print("\n  Reading: k1-dominant means particles ENTER the step already "
      "outside\n  (the previous step left them there). k2..k4-dominant means "
      "the step\n  itself walked them out.")

# ---------------------------------------------------------------------------
banner("Q2  does the step recover?")
# ---------------------------------------------------------------------------
lost = elem_final < 0
print(f"  final search succeeded : {int((~lost).sum()):7,} "
      f"({100.0 * (~lost).mean():5.1f}%)  <- harmless under last_valid_vel")
print(f"  final search FAILED    : {int(lost.sum()):7,} "
      f"({100.0 * lost.mean():5.1f}%)  <- these are lost particles")
print("\n  recovery rate by which stage failed:")
for s in range(4):
    m = bad[:, s]
    if m.sum():
        print(f"    {STAGE[s]} failed -> final ok "
              f"{100.0 * (~lost[m]).mean():5.1f}%  (n={int(m.sum()):,})")
print("\n  recovery rate by number of failed stages:")
for k in range(1, 5):
    m = nbad == k
    if m.sum():
        print(f"    {k} stage(s) failed -> final ok "
              f"{100.0 * (~lost[m]).mean():5.1f}%  (n={int(m.sum()):,})")

# ---------------------------------------------------------------------------
# Geometry, if the case is available
# ---------------------------------------------------------------------------
level_val = ltree = None
bbox_lo = bbox_hi = None
if CASE is not None:
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        from scipy.spatial import cKDTree
        post = CASE / "post"
        txt = (CASE / "run_jaxtrace.sh").read_text(errors="ignore")
        vs = int(re.search(r"^VEL_START=(\d+)", txt, re.M).group(1))
        pv = sorted(post.glob(f"*_{vs}.pvtu"))[0]
        r = vtk.vtkXMLPUnstructuredGridReader()
        r.SetFileName(str(pv))
        r.Update()
        g = r.GetOutput()
        pd = g.GetPointData()
        names = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
        pts = vtk_to_numpy(g.GetPoints().GetData()).astype(np.float64)
        bbox_lo, bbox_hi = pts.min(axis=0), pts.max(axis=0)
        if "LEVEL" in names:
            level_val = vtk_to_numpy(pd.GetArray("LEVEL")).astype(np.float64)
            ltree = cKDTree(pts)
        print(f"\n[geometry] {pv.name}: bbox z "
              f"{bbox_lo[2] * 1000:.3f}..{bbox_hi[2] * 1000:.3f} mm, "
              f"LEVEL {'loaded' if level_val is not None else 'absent'}")
    except Exception as e:                                  # noqa: BLE001
        print(f"\n[geometry] unavailable ({e}); Q3/Q4 reduced")

# ---------------------------------------------------------------------------
banner("Q3  where does the failing substage land?")
# ---------------------------------------------------------------------------
# Take the FIRST failing stage's position for each record.
idx = np.arange(n)
fp = sub_pos[idx, np.maximum(first, 0)]
z = fp[:, 2] * 1000
rad = np.hypot(fp[:, 0], fp[:, 1]) * 1000

print(f"  first-failing-substage position:")
print(f"    radius (mm): min {rad.min():7.3f}  median {np.median(rad):7.3f}  "
      f"max {rad.max():7.3f}")
print(f"    z      (mm): min {z.min():7.3f}  median {np.median(z):7.3f}  "
      f"max {z.max():7.3f}")

cats = Counter()
for i in range(n):
    p = fp[i]
    if bbox_lo is not None and (np.any(p < bbox_lo) or np.any(p > bbox_hi)):
        cats["outside the mesh bbox"] += 1
    elif p[2] > 1e-6:
        cats["above the free surface (z>0)"] += 1
    elif ltree is not None:
        _, nn = ltree.query(p[None, :], k=1)
        cats["inside the tool (LEVEL<0)" if level_val[nn][0] < 0
             else "in material, LEVEL>=0  <- UNEXPLAINED"] += 1
    else:
        cats["unclassified (no LEVEL)"] += 1
print("\n  classification:")
for k, v in cats.most_common():
    print(f"    {k:42}: {v:7,} ({100.0 * v / n:5.1f}%)")

# ---------------------------------------------------------------------------
banner("Q4  how far does it overshoot, vs one substage's stride?")
# ---------------------------------------------------------------------------
# stride = distance from the step's start position to the failing substage
stride = np.linalg.norm(fp - sub_pos[:, 0], axis=1) * 1000
print(f"  distance from step start to the failing substage (mm):")
print(f"    median {np.median(stride):.4f}  p90 "
      f"{np.percentile(stride, 90):.4f}  max {stride.max():.4f}")
if bbox_hi is not None:
    over = np.maximum(0.0, fp[:, 2] - 0.0) * 1000     # above z=0
    m = over > 0
    if m.any():
        print(f"  for those above z=0: overshoot above the surface (mm) "
              f"median {np.median(over[m]):.4f}  max {over[m].max():.4f}")
        print(f"    overshoot / stride: median "
              f"{np.median(over[m] / np.maximum(stride[m], 1e-12)):.3f}")
print("\n  Reading: overshoot << stride means the particle barely left the "
      "domain and\n  a search or boundary treatment should have caught it; "
      "overshoot ~ stride\n  means the step genuinely flew out and the "
      "timestep or velocity is at fault.")

# ---------------------------------------------------------------------------
banner("Q5  do the same particles fail repeatedly?")
# ---------------------------------------------------------------------------
pc = Counter(int(p) for p in particle)
rep = Counter(pc.values())
print("  failures per particle:")
for k in sorted(rep)[:8]:
    print(f"    {k:3} time(s): {rep[k]:7,} particles")
mx = max(pc.values()) if pc else 0
print(f"  most failures for one particle: {mx}")
print(f"  particles failing more than once: "
      f"{sum(1 for v in pc.values() if v > 1):,} of {len(pc):,}")
print("\n  Reading: many repeats mean a particle gets stuck in a bad state "
      "and keeps\n  failing -- consistent with a latched ElementID=-1. Mostly "
      "single failures\n  mean each is a transient excursion.")

banner("summary")
nl = int(lost.sum())
print(f"""Of {n:,} logged particle-steps with a failed substage,
{nl:,} ({100.0 * nl / n:.1f}%) ended with elem_final < 0 and are genuine losses.

Decision guide:
  * k1 dominant + many repeats      -> the lost flag latches; a particle that
                                       fails once keeps failing. Fix the
                                       re-entry path, not the search.
  * k2..k4 dominant + single events -> the step walks the particle out; look
                                       at dt and at the boundary treatment.
  * "in material, LEVEL>=0" large   -> a real search failure in valid mesh,
                                       which is the only case needing a
                                       search/registration fix.
  * overshoot << stride             -> boundary handling, not integration.""")
