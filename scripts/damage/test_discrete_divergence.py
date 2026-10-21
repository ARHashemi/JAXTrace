"""Is our FOM velocity DISCRETELY divergence-free against its P1 pressure space?

⚠️ This decides whether Lederer & Schoberl's LOCAL vertex-patch reconstruction
applies. Their operator maps DISCRETELY div-free functions to EXACTLY div-free
ones -- if our field is not discretely div-free, the operator has no premise.

TEST:  b_i = integral over Omega of (div u) * phi_i   for every P1 basis phi_i.
On a tet, div u is CONSTANT, and integral of phi_i over T = V/4. So
    b_i = sum over T containing i of  div(u)_T * V_T / 4
Normalise by the same sum with |div| to get a relative measure.
"""
import numpy as np, sys
sys.path.insert(0, "/flash/shared/jax/JAXTrace")
from pathlib import Path
from jaxtrace.gpu.mesh_loader_timedep import load_field_sequences_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.damage.fields import tet_shape_gradients, velocity_gradient_per_element

nodes, conn, seq = load_field_sequences_from_pvtu(
    Path("/flash/users/ali/data/cylA.gid/post/0eule"), "cylA_{timestep}.pvtu",
    (159, 159), ["Displacement"], verbose=False, n_workers=8)
o = deduplicate_nodes(nodes, conn, seq["Displacement"])
nodes, conn, vel = o[0], o[1].astype(np.int64), o[3][0].astype(np.float64)
c = conn            # WHOLE mesh: the test is global by nature
gN, V = tet_shape_gradients(nodes, c)
L = velocity_gradient_per_element(vel, c, gN)
div = np.trace(L, axis1=1, axis2=2)
Va = np.abs(V)

n = len(nodes)
b = np.zeros(n); babs = np.zeros(n)
for k in range(4):
    np.add.at(b,    c[:, k], div * Va / 4.0)
    np.add.at(babs, c[:, k], np.abs(div) * Va / 4.0)

rel = np.abs(b) / np.maximum(babs, 1e-300)
touched = babs > 0
print(f"  nodes {n:,}, elements {len(c):,}")
print()
print("  DISCRETE divergence  b_i = int (div u) phi_i,  relative to int |div u| phi_i")
for q in (50, 75, 90, 95, 99, 100):
    print(f"     p{q:<4} {np.percentile(rel[touched], q):.4e}")
print(f"     mean {rel[touched].mean():.4e}")
print()
print(f"  POINTWISE divergence for comparison:")
nD = np.sqrt(np.sum((0.5*(L+np.transpose(L,(0,2,1))))**2, axis=(1,2)))
print(f"     median |div u|/|D| = {np.median(np.abs(div)/np.maximum(nD,1e-30)):.4e}")
print()
if np.median(rel[touched]) < 1e-3:
    print("  ✅ DISCRETELY DIVERGENCE-FREE -- Lederer's premise HOLDS,")
    print("     and the local vertex-patch route is available.")
else:
    print("  ❌ NOT discretely divergence-free either. Lederer's operator")
    print("     has no premise, exactly as LN had none.")
