"""At which octree LEVEL is a particle's host actually found?

A two-tier search (fast kernel for the common levels, full kernel for the
rest) only pays off if the overwhelming majority of real particles are found
at one or two levels. Measured on the actual production seed distribution,
not on cell counts -- cells and particles are different populations.
"""
import os, sys
import numpy as np
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "/flash/shared/jax/JAXTrace")
from pathlib import Path
import jax.numpy as jnp
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import extract_octree_cells_parent_cube
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu
import collections

post = Path("/scratch/shared/PinShapes/D-ConcavityTilt/D2.gid/post")
npos, conn, vel = load_velocity_sequence_from_pvtu(
    base_path=post, file_pattern="A2_{timestep}.pvtu", timestep_range=(34, 34),
    field_name="Displacement", verbose=False)
npos, conn, _n, vel = deduplicate_nodes(npos, conn, velocity_sequence=vel, verbose=False)
conn = conn.astype(np.int32)
cells = extract_octree_cells_parent_cube(npos, conn, tolerance=1e-6, verbose=False)
g = upload_mesh_aligned_octree_to_gpu(conn, npos, cells, verbose=False)
off = np.asarray(g.cell_to_elements_offsets).astype(np.int64)
data = np.asarray(g.cell_to_elements_data).astype(np.int64)
lev = np.asarray(cells.cell_levels).astype(np.int64)
gidx = np.asarray(cells.cell_grid_indices).astype(np.int64)
lcs = np.asarray(g.level_cell_sizes).astype(np.float64)

# Reproduce the production seed grid: 60x120x40 over the production SEED_BOX
box = np.array([-0.0199, -0.0111, -0.01499, 0.01499, -0.00599, -0.00019])
nx, ny, nz = 60, 120, 40
xs = np.linspace(box[0], box[1], nx)
ys = np.linspace(box[2], box[3], ny)
zs = np.linspace(box[4], box[5], nz)
P = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
rs = np.random.RandomState(0)
P = P[rs.choice(len(P), 20000, replace=False)]
print(f"probing {len(P):,} seed positions from the production grid")

# Build a (level, i,j,k) -> cell lookup
key = {}
for ci in range(len(lev)):
    key[(int(lev[ci]), int(gidx[ci,0]), int(gidx[ci,1]), int(gidx[ci,2]))] = ci

V = npos[conn]
def bary_in(x, t):
    T = np.column_stack([t[1]-t[0], t[2]-t[0], t[3]-t[0]])
    try:
        b = np.linalg.solve(T, x - t[0])
    except np.linalg.LinAlgError:
        return False
    return (1-b.sum() >= -1e-9) and (b >= -1e-9).all()

levels_walked = list(range(int(g.max_level), int(g.min_level)-1, -1))
print(f"levels walked by the kernel, in order: {levels_walked}")
found_at = []
nf = 0
for x in P:
    hit = None
    for n, L in enumerate(levels_walked):
        p = lcs[L]
        if not np.all(p > 0):
            continue
        b = np.floor(x / p).astype(np.int64)
        done = False
        for di in (-1,0,1):
            for dj in (-1,0,1):
                for dk in (-1,0,1):
                    ci = key.get((L, b[0]+di, b[1]+dj, b[2]+dk))
                    if ci is None:
                        continue
                    for e in data[off[ci]:off[ci+1]]:
                        if bary_in(x, V[e]):
                            hit = (n, L); done = True; break
                    if done: break
                if done: break
            if done: break
        if hit: break
    if hit is None:
        nf += 1
    else:
        found_at.append(hit)

print(f"  not located at any level: {nf:,} ({100*nf/len(P):.1f}%)  "
      f"(seeds outside the mesh)")
c = collections.Counter(f[1] for f in found_at)
it = collections.Counter(f[0] for f in found_at)
tot = len(found_at)
print()
print("  host LEVEL distribution (of located particles):")
for L in sorted(c, reverse=True):
    print(f"    level {L:2d}: {c[L]:7,} ({100*c[L]/tot:6.2f}%)")
print()
print("  ITERATION at which the host was found (0 = first level tried):")
cum = 0
for n in sorted(it):
    cum += it[n]
    print(f"    iteration {n}: {it[n]:7,} ({100*it[n]/tot:6.2f}%)   cumulative {100*cum/tot:6.2f}%")
