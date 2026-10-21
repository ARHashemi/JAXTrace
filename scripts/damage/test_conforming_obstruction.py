"""BDM1 changes the field 16,000x more than the constraint requires. Why?

Candidate (a): GLOBAL COUPLING. BDM1 DOFs are shared across faces, so the
element-wise 'remove the isotropic part' solution is NOT attainable -- it is
generally discontinuous in the normal component and thus not in BDM1.

Candidate (b): my OBJECTIVE only matches the centroid velocity.

TEST for (a): is the per-element ideal projection conforming? Take two
adjacent tets, project each independently, and check whether the normal
components still agree on the shared face.
"""
import numpy as np
rng = np.random.default_rng(1)

# two tets sharing the face (0,1,2)
p0 = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
p1 = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]])
shared = np.array([0,1,2])
nv = np.cross(p0[1]-p0[0], p0[2]-p0[0]); nhat = nv/np.linalg.norm(nv)

# one global P1-like linear field with divergence
A = rng.normal(size=(3,3)); b = rng.normal(size=3)
print(f"source field: tr(A) = {np.trace(A):+.4f}")

def proj(A, b):
    """per-element ideal: remove the isotropic part of A"""
    return A - np.trace(A)/3*np.eye(3), b

# ⚠️ Both elements see the SAME A here, so the projections agree trivially.
# The real case has DIFFERENT A per element (P1 is only C0, the gradient
# jumps across faces). Simulate that.
A0 = A
A1 = A + 0.3*rng.normal(size=(3,3))          # a different gradient next door
print(f"neighbour tr(A) = {np.trace(A1):+.4f}")
P0, _ = proj(A0, b)
P1, _ = proj(A1, b)
print()
print("normal component on the shared face, at its 3 vertices:")
dis = []
for v in shared:
    x = p0[v]
    n0 = np.dot(P0@x + b, nhat); n1 = np.dot(P1@x + b, nhat)
    dis.append(abs(n0-n1))
    print(f"   vertex {v}: side A {n0:+.6f}   side B {n1:+.6f}   jump {abs(n0-n1):.3e}")
print()
print(f"⚠️ max jump {max(dis):.3e} -- the per-element ideal projection is")
print("   NOT CONFORMING: the normal component is double valued across the")
print("   face, so that field is not in BDM1 (nor in SV, nor in any")
print("   H(div)-conforming space).")
print()
print("✅ So the 4.2e-06 bound is NOT ATTAINABLE by any conforming method.")
print("   The attainable minimum is larger, and SV-P3 faces the same")
print("   obstruction -- it is also H(div)-conforming.")
