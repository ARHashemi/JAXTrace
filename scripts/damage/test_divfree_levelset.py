"""Does the projection corrupt the LEVEL-SET-MASKED region? (7.7) -- it does not.

The kernel zeroes velocity inside the tool:
    val = where(ls_val >= 0, val, 0.0)
'No material here', so no flow and no damage.

⚠️ THE CONCERN: the wrapper returns  u - (tr(Jac)/3)(pos - xc).
Inside the tool u = 0, but xc is the ELEMENT centroid, and (pos - xc) is NOT
zero. If tr(Jac) is not exactly zero there, the wrapper INJECTS velocity into
a region that is supposed to be stationary -- and worse, the mask is a step
function, so jacfwd through it is zero on each side but the element straddling
the interface mixes masked and unmasked nodes.

Three cases to check:
  (a) fully inside the tool  (all 4 nodes masked)  -> u = 0 everywhere
  (b) fully outside          (no nodes masked)     -> normal behaviour
  (c) STRADDLING the interface                     -> the dangerous one

RESULT: clean in all three. jnp.where has ZERO GRADIENT on the masked side, so
tr(Jac) = 0 exactly where u = 0 and the correction term vanishes with it. The
answer is structural, not luck -- but it was worth measuring rather than
arguing, because "nothing is injected into the tool" is the kind of claim that
is easy to assert and expensive to be wrong about.

Run: python3 scripts/damage/test_divfree_levelset.py
"""
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

rng = np.random.default_rng(5)
p = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
uv = rng.normal(size=(4,3))
M_inv = jnp.asarray(np.linalg.inv(np.stack([p[1]-p[0],p[2]-p[0],p[3]-p[0]],axis=1)))
P=jnp.asarray(p); UV=jnp.asarray(uv); xc=jnp.asarray(p.mean(axis=0))

def make_raw(ls):
    LS=jnp.asarray(ls)
    def raw(pos):
        b=M_inv@(pos-P[0])
        w=jnp.array([1-b[0]-b[1]-b[2],b[0],b[1],b[2]])
        val=w@UV
        ls_val=w@LS
        return jnp.where(ls_val>=0.0, val, jnp.zeros(3))   # the kernel's mask
    return raw

def wrap(raw):
    def f(pos):
        u=raw(pos); J=jax.jacfwd(raw)(pos)
        return u-(jnp.trace(J)/3.0)*(pos-xc)
    return f

cases={
 '(a) all inside tool  ': np.array([-1.,-1.,-1.,-1.]),
 '(b) all outside      ': np.array([ 1., 1., 1., 1.]),
 '(c) STRADDLING       ': np.array([-1.,-1., 1., 1.]),
}
print(f"{'case':22s} {'|u_raw|':>10s} {'|u_wrapped|':>12s} {'injected':>10s}")
for name,ls in cases.items():
    raw=make_raw(ls); g=wrap(raw)
    worst_inj=0.0; detail=[]
    for _ in range(400):
        x=jnp.asarray(p.T@rng.dirichlet(np.ones(4)))
        ur=np.asarray(raw(x)); uw=np.asarray(g(x))
        # "injected" = velocity appearing where the raw field was zero
        if np.linalg.norm(ur)<1e-14:
            worst_inj=max(worst_inj,float(np.linalg.norm(uw)))
        detail.append((np.linalg.norm(ur),np.linalg.norm(uw)))
    d=np.array(detail)
    print(f"{name:22s} {d[:,0].mean():10.3e} {d[:,1].mean():12.3e} "
          f"{worst_inj:10.3e}  {'⚠️ INJECTS' if worst_inj>1e-12 else 'clean'}")

    if worst_inj > 1e-12:
        raise SystemExit("FAIL: velocity injected into the masked region")
print()
print("PASS -- no velocity injected where the mask zeroes the field")
