"""Does our corrected field show the SELZER/CORDES failure mode?

Selzer & Cirpka (2020): "the non-conforming property of the velocity field
leads to severe numerical artifacts, possibly including PARTICLE STAGNATION
because the approximated normal velocity component POINTS INTO OPPOSITE
DIRECTIONS at the two sites of an element interface."

That is a sharp, testable claim. For each interior face, compare the normal
velocity seen from each side AT THE SAME POINT (the face centroid):

    u_L . n   and   u_R . n

If they have OPPOSITE SIGNS the face is a sink/source trap: a particle
arriving at it is pushed back from both sides (stagnation) or pulled through
twice (spurious flux).

⚠️ Measure the RAW field too. If raw P1 already has such faces, the
projection is not introducing the defect, only possibly changing it.

MEASURED on cylA ts=159, 101,612 interior faces:
    raw P1          0/101,612 (0.000 %)  jump p99 0.0000  <- EXACTLY continuous
    divergence-free 1,043/101,612 (1.026 %) jump p99 1.0000
So raw P1 is flux-continuous across faces and THE PROJECTION BREAKS THAT.
The 1.03 % is the same order as cylA's ~1 % of large trajectory shifts and
1.84 % of particles ending in a different element.

⚠️ This narrows the method's claim. For ln_J the face jump is harmless --
every RK4 substage lands in SOME element where div u = 0, which is what 4z
measured. For the TRAJECTORY it is not harmless: a particle meeting a
sign-flipped face is pushed back from both sides. det F is sound; pathlines
through those faces are not.
"""
import sys, numpy as np
sys.path.insert(0,'.')
from jaxtrace.damage.divfree_projection import project_element_gradient_divfree

rng = np.random.default_rng(0)

def affine(nodes, conn, vel):
    p = nodes[conn]
    X = np.concatenate([p, np.ones((len(conn),4,1))], axis=2)
    C = np.linalg.solve(X, vel[conn])
    return C[:,:3,:].transpose(0,2,1), C[:,3,:]

def main():
    from jaxtrace.gpu.mesh_loader_timedep import load_field_sequences_from_pvtu
    from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
    from jaxtrace.damage.divfree import build_face_topology, OPP
    from pathlib import Path
    case, pat, ts = sys.argv[1], sys.argv[2], int(sys.argv[3])
    nodes, conn, seq = load_field_sequences_from_pvtu(
        Path(case), pat, (ts,ts), ['Displacement'], verbose=False, n_workers=8)
    o = deduplicate_nodes(nodes, conn, seq['Displacement'], verbose=False)
    nodes, conn, vel = o[0], o[1].astype(np.int64), o[3][0].astype(np.float64)
    print(f"mesh {len(nodes):,} nodes {len(conn):,} el", flush=True)

    # a random connected-ish subset to keep the face map cheap
    sel = rng.choice(len(conn), 400000, replace=False)
    cp = conn[sel]
    uf, fidx, cnt = build_face_topology(cp)
    A, b = affine(nodes, cp, vel)
    Ap = project_element_gradient_divfree(A)
    xc = nodes[cp].mean(axis=1)
    bp = b + np.einsum('mij,mj->mi', A-Ap, xc)

    # interior faces: exactly 2 owners
    interior = np.where(cnt == 2)[0]
    print(f"interior faces in the subset: {len(interior):,}", flush=True)
    # map face -> its two (elem, local) owners
    order = np.argsort(fidx.ravel(), kind='stable')
    fs = fidx.ravel()[order]; ow = order // 4; lo = order % 4
    start = np.searchsorted(fs, interior); end = np.searchsorted(fs, interior, side='right')
    two = (end - start) == 2
    interior, start = interior[two], start[two]
    eL, lL = ow[start], lo[start]
    eR, lR = ow[start+1], lo[start+1]

    # face centroid and normal from the LEFT element's local face
    P = nodes[cp]
    fv = cp[:, OPP.ravel()].reshape(-1,4,3)
    tri = nodes[fv[eL, lL]]                       # (n,3,3)
    fc = tri.mean(axis=1)
    nv = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    n = nv/np.maximum(np.linalg.norm(nv,axis=1),1e-300)[:,None]

    def normal_vel(Am, bm, e, x):
        return np.einsum('mi,mi->m', np.einsum('mij,mj->mi', Am[e], x) + bm[e], n)

    for tag, Am, bm in (("raw P1", A, b), ("divergence-free", Ap, bp)):
        uL = normal_vel(Am, bm, eL, fc)
        uR = normal_vel(Am, bm, eR, fc)
        opp = (uL*uR) < 0
        jump = np.abs(uL-uR)
        scale = np.maximum(np.abs(uL)+np.abs(uR), 1e-300)
        print(f"\n  {tag}:")
        print(f"    faces with OPPOSITE-SIGN normal velocity: "
              f"{opp.sum():,}/{len(opp):,} ({100*opp.mean():.3f} %)")
        print(f"    |u_L.n - u_R.n| / (|u_L.n|+|u_R.n|)  p50 "
              f"{np.median(jump/scale):.4f}  p99 {np.percentile(jump/scale,99):.4f}")

if __name__ == '__main__':
    main()
