#!/usr/bin/env python3
"""Reclassify ElementID<0 against the TRUE MESH bbox, not the particle cloud.

Two reasons the earlier classification was misleading:
  * the Escaped flag is LATCHING and tests ANY ONE axis, so a particle that
    merely touches the z_min floor (z=-6mm) is flagged forever even though it
    never left through an outflow face;
  * the particle cloud's own extent is not the mesh bbox, so "near a face"
    computed from the cloud mislabels interior points.
D2 mesh bbox: x -20..40, y -15..15, z -6..+1.35 mm.
"""
import sys, numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy

f = sys.argv[1]
TOL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05

r = vtk.vtkXMLUnstructuredGridReader(); r.SetFileName(f); r.Update()
g = r.GetOutput()
p = vtk_to_numpy(g.GetPoints().GetData()) * 1000.0
e = vtk_to_numpy(g.GetPointData().GetArray("ElementID")).astype(np.int64)
lost = e < 0

BB = np.array([[-20.0, -15.0, -6.0], [40.0, 15.0, 1.35]])
d = np.minimum(np.abs(p - BB[0]), np.abs(p - BB[1]))
on_any = (d < TOL).any(axis=1)

print("TRUE MESH bbox x -20..40  y -15..15  z -6..1.35 mm   tol %.3g mm" % TOL)
print(f"  lost total           : {int(lost.sum()):,}")
print(f"  lost ON a mesh face  : {int((lost & on_any).sum()):,}  (expected: left the domain)")
print(f"  lost in the INTERIOR : {int((lost & ~on_any).sum()):,}  <-- THE REAL DEFECT")
print()
for ax, nm in enumerate("xyz"):
    for b, lbl in ((BB[0][ax], nm + "_min"), (BB[1][ax], nm + "_max")):
        m = lost & (np.abs(p[:, ax] - b) < TOL)
        if m.sum():
            print(f"  on {lbl:6}: {int(m.sum()):,}")

ii = lost & ~on_any
if ii.sum():
    rr = np.hypot(p[ii, 0], p[ii, 1]); zz = p[ii, 2]
    print(f"\nINTERIOR losses {int(ii.sum()):,}")
    for nm, v in (("x", p[ii, 0]), ("y", p[ii, 1]), ("z", zz), ("r", rr)):
        print(f"  {nm}: min {v.min():7.2f} p5 {np.percentile(v,5):7.2f} "
              f"med {np.median(v):7.2f} p95 {np.percentile(v,95):7.2f} max {v.max():7.2f}")
    print("\n  radial histogram 0.5 mm bins")
    h, ed = np.histogram(rr, bins=np.arange(0, 20.5, 0.5)); mx = max(h.max(), 1)
    for i, c in enumerate(h):
        if c:
            print(f"    r {ed[i]:5.1f}-{ed[i+1]:4.1f} {c:7,d} {'#'*int(42*c/mx)}")
    print("\n  depth histogram 0.5 mm bins")
    h, ed = np.histogram(zz, bins=np.arange(-6.5, 2.0, 0.5)); mx = max(h.max(), 1)
    for i, c in enumerate(h):
        if c:
            print(f"    z {ed[i]:5.1f}-{ed[i+1]:5.1f} {c:7,d} {'#'*int(42*c/mx)}")
    print("\nSEED BOX for the INTERIOR losses only (metres, 1 mm margin):")
    lo = [p[ii, i].min()/1000 - 0.001 for i in range(3)]
    hi = [p[ii, i].max()/1000 + 0.001 for i in range(3)]
    print(f'  SEED_BOX="{lo[0]:.5f} {hi[0]:.5f} {lo[1]:.5f} {hi[1]:.5f} {lo[2]:.5f} {hi[2]:.5f}"')
