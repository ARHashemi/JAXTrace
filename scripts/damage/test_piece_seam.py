"""⚠️ Does a per-piece projection create a SEAM at the merged boundary?

THE CONCERN, stated precisely. If we solve the BDM1 projection independently on
each FEMUSS piece and then merge+deduplicate, a face lying on the interface
between two pieces is solved TWICE -- once per piece -- with different
neighbourhoods. The two answers need not agree, so after merging the normal
component could be DOUBLE VALUED: exactly the H(div) conformity we are paying
for, destroyed at the seam.

⚠️ And the seam is not a small set: 7.8% of points are duplicated across
pieces, and a particle crossing one sees a velocity jump.

This script QUANTIFIES it before any implementation, by checking what the
pieces actually share.
"""
import vtk, numpy as np
from vtk.util.numpy_support import vtk_to_numpy

base = "/flash/users/ali/data/cylA.gid/post/0eule/cylA_159_%d.vtu"

def load(k):
    r = vtk.vtkXMLUnstructuredGridReader(); r.SetFileName(base % k); r.Update()
    d = r.GetOutput()
    pts = vtk_to_numpy(d.GetPoints().GetData()).astype(np.float64)
    cells = vtk_to_numpy(d.GetCells().GetConnectivity()).reshape(-1, 4)
    vel = vtk_to_numpy(d.GetPointData().GetArray("Displacement"))
    return pts, cells, vel

A = load(0); B = load(1)
print(f"piece 0: {len(A[0]):,} pts, {len(A[1]):,} cells")
print(f"piece 1: {len(B[0]):,} pts, {len(B[1]):,} cells")

# shared points, by coordinate
key = lambda P: {tuple(np.round(p, 10)) for p in P}
ka, kb = key(A[0]), key(B[0])
shared = ka & kb
print(f"\nshared POINTS between pieces 0 and 1: {len(shared):,}")

# ⚠️ The real question: do they share CELLS (true ghost overlap) or only
# a surface of points (a clean cut)?
def cellkeys(P, C):
    out = set()
    for t in C:
        out.add(tuple(sorted(tuple(np.round(P[i], 10)) for i in t)))
    return out

ca, cb = cellkeys(*A[:2]), cellkeys(*B[:2])
sc = ca & cb
print(f"shared CELLS: {len(sc):,}")
if sc:
    print("  ✅ TRUE OVERLAP: the same tetrahedra exist in both pieces.")
    print("     => a projection solved on each piece covers the seam from")
    print("        BOTH sides, and the shared cells can be used to detect")
    print("        and reconcile any disagreement.")
else:
    print("  ⚠️ NO shared cells -- the pieces meet at a surface only.")
    print("     => the interface FACES belong to one piece each, so a")
    print("        per-piece solve would pin them inconsistently.")
