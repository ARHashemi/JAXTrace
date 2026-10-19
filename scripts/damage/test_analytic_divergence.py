"""jacfwd divergence must reproduce the analytic P1 divergence exactly.

On a P1 tet the velocity gradient is CONSTANT per element and equals
L = sum_a u_a (x) grad_N_a. So trace(jacfwd(interp)) must equal tr(L) to
float precision, at ANY point inside the element. If it does not, the
divergence_single implementation is wrong.
"""
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

rng = np.random.default_rng(1)
# one reference tet
nodes = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
conn  = np.array([[0,1,2,3]], dtype=np.int32)
u     = rng.normal(size=(4,3))

# analytic P1: grad_N from the inverse Jacobian
e1,e2,e3 = nodes[1]-nodes[0], nodes[2]-nodes[0], nodes[3]-nodes[0]
Jm = np.stack([e1,e2,e3], axis=1)          # columns
Jinv = np.linalg.inv(Jm)
gN = np.zeros((4,3))
gN[1:,:] = Jinv.T @ np.eye(3)              # dN_i/dx for i=1,2,3
gN[1:,:] = Jinv.T
gN[0,:]  = -gN[1:,:].sum(axis=0)
L = np.einsum('ai,aj->ij', u, gN)
div_analytic = np.trace(L)

# the kernel's P1 interpolation, in the same form the kernel uses
def interp(pos):
    v0,v1,v2 = nodes[1]-nodes[0], nodes[2]-nodes[0], nodes[3]-nodes[0]
    M = jnp.stack([v0,v1,v2], axis=1)
    bary = jnp.linalg.inv(M) @ (pos - nodes[0])
    b1,b2,b3 = bary[0],bary[1],bary[2]
    b0 = 1.0-b1-b2-b3
    w = jnp.stack([b0,b1,b2,b3])
    return jnp.einsum('a,ac->c', w, jnp.asarray(u))

def div_jacfwd(pos):
    return jnp.trace(jax.jacfwd(interp)(pos))

print(f"analytic tr(L)      = {div_analytic:+.16e}")
for name, p in (("centroid", nodes.mean(axis=0)),
                ("near vertex", np.array([0.9,0.03,0.03])),
                ("interior", np.array([0.2,0.3,0.1]))):
    d = float(div_jacfwd(jnp.asarray(p)))
    err = abs(d-div_analytic)/max(abs(div_analytic),1e-30)
    print(f"  jacfwd @ {name:12s} = {d:+.16e}   rel err {err:.2e}")
    assert err < 1e-12, f"jacfwd disagrees with analytic P1 at {name}"

# ⚠️ And the key property: a LINEAR velocity field with zero divergence must
# give EXACTLY zero, at any point.
u_df = np.array([[0.,0.,0.],[1.,0.,0.],[0.,-1.,0.],[0.,0.,0.]])  # du/dx - dv/dy
def interp_df(pos):
    v0,v1,v2 = nodes[1]-nodes[0], nodes[2]-nodes[0], nodes[3]-nodes[0]
    M = jnp.stack([v0,v1,v2], axis=1)
    bary = jnp.linalg.inv(M) @ (pos - nodes[0])
    b1,b2,b3 = bary[0],bary[1],bary[2]
    w = jnp.stack([1.0-b1-b2-b3,b1,b2,b3])
    return jnp.einsum('a,ac->c', w, jnp.asarray(u_df))
d0 = float(jnp.trace(jax.jacfwd(interp_df)(jnp.asarray(nodes.mean(axis=0)))))
print(f"\ndiv-free linear field -> {d0:+.3e}  (must be ~0)")
assert abs(d0) < 1e-14

print("\nALL PASS — jacfwd reproduces the analytic P1 divergence.")
