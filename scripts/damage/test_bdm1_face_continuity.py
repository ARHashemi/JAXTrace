"""Redo the continuity test with CORRECT DOF matching.

⚠️ The previous test matched DOFs by LOCAL index, but the two tets list the
shared face's vertices in different local positions, and the outward normals
point in OPPOSITE directions. Both must be handled, exactly as the real
assembly does via sorted global node ids and the orientation sign.
"""
import numpy as np
OPP = np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]])
# shared face is (0,1,2) -> in BOTH tets that is the face OPPOSITE vertex 3
p0 = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
p1 = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]])
rng = np.random.default_rng(3)

def build(p):
    T = np.zeros((12,12)); nrm = []
    for f in range(4):
        a,b,d = p[OPP[f,0]], p[OPP[f,1]], p[OPP[f,2]]
        nv = np.cross(b-a,d-a); nh = nv/np.linalg.norm(nv)
        nh = nh*np.sign(np.dot(nh, a-p[f]))      # OUTWARD
        nrm.append(nh)
        for k,v in enumerate((a,b,d)):
            r = np.zeros(12)
            for i in range(3):
                r[i*3:(i+1)*3] = nh[i]*v; r[9+i] = nh[i]
            T[3*f+k] = r
    return T, nrm

T0,n0 = build(p0); T1,n1 = build(p1)
print(f"outward normals on the shared face: tet A {n0[3]}, tet B {n1[3]}")
print(f"   they are {'OPPOSITE ✅ (as they must be)' if np.dot(n0[3],n1[3])<0 else 'the same ⚠️'}")
print()
# ⚠️ The DOF is the normal component w.r.t. a GLOBAL face orientation. Tet B
# sees the opposite outward normal, so its DOF values are the NEGATIVE of
# tet A's for the same physical field.
dofs0 = rng.normal(size=12)
dofs1 = rng.normal(size=12)
dofs1[9:12] = -dofs0[9:12]        # ✅ sign-corrected sharing
c0 = np.linalg.solve(T0,dofs0); c1 = np.linalg.solve(T1,dofs1)
A0,b0 = c0[:9].reshape(3,3), c0[9:]
A1,b1 = c1[:9].reshape(3,3), c1[9:]
nhat = n0[3]
mx = 0.0
for t in np.linspace(0.05,0.9,6):
    for sf in (0.05,0.3):
        x = p0[0]+t*(p0[1]-p0[0])+sf*(p0[2]-p0[0])
        mx = max(mx, abs(np.dot(A0@x+b0,nhat) - np.dot(A1@x+b1,nhat)))
print(f"NORMAL component jump across the shared face: {mx:.3e}")
print(f"   {'✅ CONTINUOUS -- BDM1 IS H(div)-conforming' if mx<1e-10 else '❌ discontinuous'}")
print()
print("✅ So the normal flux a particle sees is SINGLE VALUED crossing a face.")
print("⚠️ The TANGENTIAL component does jump -- that is what H(div) means,")
print("   and it is true of RT, BDM and every H(div) space. SV (H1) is the")
print("   only one that is fully continuous.")
