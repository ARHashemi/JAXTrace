"""What can BDM1 and SV-P3 represent? The 30-second test RTN0 should have had.

⚠️ RTN0 was built, solved and only THEN found to be rotation-blind. This test
asks the representation question FIRST, with no solver, for both candidates.

A space passes if it reproduces every LINEAR field exactly -- in particular
rotation, which dominates FSW flow near the tool.
"""
import numpy as np
import itertools

OPP = np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]])
p = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
V = np.dot(p[1]-p[0], np.cross(p[2]-p[0], p[3]-p[0]))/6.0

FIELDS = [
    ("constant",        lambda x: np.array([0.3,-0.7,0.2])),
    ("rotation z",      lambda x: np.array([-x[1], x[0], 0.0])),
    ("rotation x",      lambda x: np.array([0.0, -x[2], x[1]])),
    ("shear xz",        lambda x: np.array([x[2], 0.0, 0.0])),
    ("lin div-free",    lambda x: np.array([x[0], -x[1], 0.0])),
    ("lin w/ div",      lambda x: np.array([x[0], x[1], x[2]])),
    ("quadratic",       lambda x: np.array([x[0]*x[1], 0.0, 0.0])),
]

def face_geom():
    area = np.zeros(4); nout = np.zeros((4,3)); ctr = np.zeros((4,3))
    verts = []
    for f in range(4):
        a,b,d = p[OPP[f,0]], p[OPP[f,1]], p[OPP[f,2]]
        nv = np.cross(b-a, d-a); area[f] = 0.5*np.linalg.norm(nv)
        nh = nv/np.linalg.norm(nv)
        nout[f] = nh*np.sign(np.dot(nh, a-p[f]))
        ctr[f] = (a+b+d)/3.0
        verts.append((a,b,d))
    return area, nout, ctr, verts

area, nout, ctr, verts = face_geom()

# ── BDM1: full P1 vector fields with continuous normal component ───────
# dim = 12 on a tet (3 DOF per face: the normal component sampled at the 3
# face vertices, or equivalently P1 normal traces). A P1 VECTOR field has 12
# coefficients too, so BDM1 == {all linear vector fields} on one element.
# ✅ So the representation test is: can a linear field be recovered from its
# P1 normal traces? Build the map and check its rank.
def bdm1_roundtrip(ufun):
    # unknowns: the 12 coefficients of u(x) = A x + b
    # constraints: normal component at the 3 vertices of each face (12 eqs)
    rows, rhs = [], []
    for f in range(4):
        for vtx in verts[f]:
            # n . (A x + b) at this vertex
            r = np.zeros(12)
            for i in range(3):
                r[i*3:(i+1)*3] = nout[f][i] * vtx      # A_i. * x
                r[9+i] = nout[f][i]                     # b_i
            rows.append(r); rhs.append(np.dot(ufun(vtx), nout[f]))
    Mx = np.array(rows); b = np.array(rhs)
    sol, *_ = np.linalg.lstsq(Mx, b, rcond=None)
    A = sol[:9].reshape(3,3); bb = sol[9:]
    xc = p.mean(axis=0)
    return A @ xc + bb, ufun(xc), np.linalg.matrix_rank(Mx)

# ── SV-P3 on the Alfeld split: contains ALL cubics, so certainly all
# linear fields. The question there is not representation but cost.
print("=== BDM1: can it recover a field from its P1 normal traces? ===")
print(f"  {'field':16s} {'rel err':>11s}  rank(12 needed)")
for nm, fn in FIELDS:
    ur, ue, rk = bdm1_roundtrip(fn)
    e = np.linalg.norm(ur-ue)/max(np.linalg.norm(ue), 1e-30)
    flag = "✅" if e < 1e-10 else ("⚠️ " if e < 0.3 else "❌")
    print(f"  {nm:16s} {e:11.3e} {flag}   {rk}")
print()
print("  ✅ BDM1 on a tet has 12 DOF = exactly the 12 coefficients of a P1")
print("     VECTOR field, so it contains ALL linear fields incl. rotation.")
print("  ⚠️ Quadratics are NOT in it -- expected, and not needed: the P1")
print("     source field is itself linear per element.")
print()
print("=== SV-P3 on the Alfeld split ===")
print("  P3 contains all cubics, hence all linear fields trivially.")
print("  ⚠️ The issue is COST, not content:")
for lab, nel, nnod in (("cylA", 3_050_196, 571_533), ("D2", 10_807_677, 1_921_303)):
    # P3 on 4 sub-tets: vertices + 2/edge + 1/face + interior, per sub-tet
    # continuous across the macro-element -> roughly 10 nodes/sub-tet shared
    sv = nel * 4 * 10 * 3 / 4          # crude: 4 subtets, ~10 P3 nodes, 3 comps
    bdm = nel * 2 * 3                   # ~2 faces/elem x 3 DOF
    print(f"   {lab:5s}: BDM1 ~{bdm/1e6:6.1f}M DOF   SV-P3 ~{sv/1e6:6.1f}M DOF"
          f"   ({sv/bdm:.1f}x more)")
