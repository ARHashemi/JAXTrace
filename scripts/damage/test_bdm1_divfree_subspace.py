"""⚠️ The property that decides BDM1: can it be made EXACTLY divergence-free?

BDM1 contains all linear fields, so div u is a CONSTANT per element. The
constraint div u = 0 is therefore ONE scalar equation per element -- the same
structure as RTN0, and exactly enforceable by one Lagrange multiplier.

But that leaves a question RTN0 did not have: with div u = 0 imposed, is
there still enough freedom left to match the P1 field?
"""
import numpy as np
OPP = np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]])
p = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
nout = np.zeros((4,3)); verts = []
for f in range(4):
    a,b,d = p[OPP[f,0]], p[OPP[f,1]], p[OPP[f,2]]
    nv = np.cross(b-a,d-a); nh = nv/np.linalg.norm(nv)
    nout[f] = nh*np.sign(np.dot(nh, a-p[f])); verts.append((a,b,d))

# u(x) = A x + b, 12 unknowns.  div u = tr(A).
print("BDM1 on one tet: 12 DOF (= 3 per face x 4 faces)")
print("  div u = tr(A) is a CONSTANT, so div u = 0 is ONE linear equation.")
print("  => 12 DOF - 1 constraint = 11 free parameters.")
print()
print("  ✅ The divergence-free subspace of P1 vector fields has dimension 11,")
print("     and CONTAINS both rotations (tr = 0) and all shears.")
print()
# verify: project a rotation onto the div-free subspace -- must be unchanged
def coeffs(ufun):
    rows, rhs = [], []
    for f in range(4):
        for v in verts[f]:
            r = np.zeros(12)
            for i in range(3):
                r[i*3:(i+1)*3] = nout[f][i]*v; r[9+i] = nout[f][i]
            rows.append(r); rhs.append(np.dot(ufun(v), nout[f]))
    return np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]

for nm, fn in (("rotation z", lambda x: np.array([-x[1],x[0],0.])),
               ("shear xz",   lambda x: np.array([x[2],0.,0.])),
               ("radial (div=3)", lambda x: np.array([x[0],x[1],x[2]]))):
    s = coeffs(fn); A = s[:9].reshape(3,3)
    print(f"  {nm:16s}: tr(A) = {np.trace(A):+.3e}"
          f"   {'✅ already div-free' if abs(np.trace(A))<1e-12 else '⚠️  has divergence'}")
print()
print("⚠️ So imposing div = 0 removes ONLY the radial mode -- precisely the")
print("   spurious part -- and leaves rotation and shear untouched.")
print("   ✅ That is exactly the property the FSW field needs.")
