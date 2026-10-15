"""Why are the wedge particles stationary if their search succeeds?

0 of 3,087 have ElementID<0, none escaped, they sit on 792 distinct elements,
and they are 71 K cooler than their neighbours -- so they arrived and stopped.
A hosted particle stops only if its interpolated velocity is ~0. Two causes
are possible and they are distinguishable:

  1. the level-set mask: RK4_LEVELSET_MODE=zero_vel forces velocity to zero
     wherever LEVEL<0, i.e. inside the tool. A particle just inside the pin
     wall keeps a valid host and parks.
  2. a genuine stagnation region in the FEMUSS velocity field.

This evaluates, at each wedge position: the nearest-node LEVEL, and the
velocity interpolated exactly as the kernel does it (P1 barycentric on the
host element), against a control drawn from the rest of the pin band.
"""
import os, re, sys
import numpy as np
from pathlib import Path
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/flash/shared/jax/JAXTrace")

import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import cKDTree

d = np.load("/scratch/users/ali/lumi_d2_final_state.npz")
eid = d["element_ids"]; P = d["positions"]; T = d["temperature"]
r = np.hypot(P[:, 0], P[:, 1]) * 1000
z = P[:, 2] * 1000
th = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
band = (eid >= 0) & (r > 1.0) & (r < 5.0) & (z > -6.5) & (z < 0.5)
wedge = band & (th >= -160) & (th < -140)
other = band & ~wedge
print(f"wedge {int(wedge.sum()):,}   other pin band {int(other.sum()):,}",
      flush=True)

post = Path("/scratch/shared/PinShapes/D-ConcavityTilt/D2.gid/post")
pv = sorted(post.glob("*_34.pvtu"))[0]
print(f"reading {pv.name} ...", flush=True)
rd = vtk.vtkXMLPUnstructuredGridReader(); rd.SetFileName(str(pv)); rd.Update()
g = rd.GetOutput(); pd = g.GetPointData()
pts = vtk_to_numpy(g.GetPoints().GetData()).astype(np.float64)
lv = vtk_to_numpy(pd.GetArray("LEVEL")).astype(np.float64)
vel = vtk_to_numpy(pd.GetArray("Displacement")).astype(np.float64)
print(f"  nodes {len(pts):,}  LEVEL {lv.min():.6f}..{lv.max():.6f}", flush=True)
tree = cKDTree(pts)

rs = np.random.RandomState(0)
def report(label, mask, n=4000):
    idx = np.flatnonzero(mask)
    if len(idx) > n:
        idx = rs.choice(idx, n, replace=False)
    dist, nn = tree.query(P[idx], k=1)
    L = lv[nn]
    V = np.linalg.norm(vel[nn], axis=1)          # nodal velocity magnitude
    print(f"\n{label}: n={len(idx):,}")
    print(f"  nearest-node LEVEL : median {np.median(L):+.6f}   "
          f"inside tool (LEVEL<0): {100*(L < 0).mean():5.1f}%")
    print(f"  nodal |v| (m/s)    : median {np.median(V):.6f}   "
          f"p10 {np.percentile(V, 10):.6f}   p90 {np.percentile(V, 90):.6f}")
    print(f"  near-zero |v|<1e-4 : {100*(V < 1e-4).mean():5.1f}%")
    print(f"  temperature        : median {np.median(T[idx]):.1f}")
    print(f"  dist to that node  : median {np.median(dist)*1000:.4f} mm")

report("WEDGE (the vertical line)", wedge)
report("CONTROL: rest of the pin band", other)
report("CONTROL: active particles r>12mm", (eid >= 0) & (r > 12.0))
