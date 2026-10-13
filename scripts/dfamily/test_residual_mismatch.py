"""Does the residual 16.43% lattice mismatch still lose particles?

The Pass-2 fix made non-Kuhn AABB registration use the level's lookup pitch,
and the mismatch fell 18.2% -> 16.43% -- not to zero. The residual comes from
parent_cube.py:195, where Pass 1 records each KUHN element's own cell_size and
cell_metadata is written only for cells that do not already exist. Kuhn
elements at a level also vary in size on an anisotropic mesh, so cells created
in Pass 1 keep a size that may differ from the level median.

This test rebuilds the octree WITH the fix and re-runs the real production
search on the previously-lost particles. Two things matter:

  1. how many of them are now FOUND -- i.e. did the Pass-2 fix work?
  2. for any still lost, is the host on a residually-mismatched cell, and was
     that cell created in Pass 1 (Kuhn) or Pass 2 (non-Kuhn)?

If the still-lost hosts sit on Pass-1 cells, the remedy is to fix the lookup
pitch once, before either pass registers anything, so both passes and the
search share one lattice by construction.
"""
import os, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/flash/shared/jax/JAXTrace")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax, jax.numpy as jnp
from jaxtrace import config
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import (
    extract_octree_cells_parent_cube)
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import (
    upload_mesh_aligned_octree_to_gpu)
from jaxtrace.gpu.search.mesh_aligned_point_location import (
    search_mesh_aligned_octree_multi_local_where)

CASE = Path("/scratch/shared/PinShapes/D-ConcavityTilt/D2.gid")
NPZ = CASE / "post_pt_annuli/run_file_n120000_s600/final_state.npz"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

d = np.load(NPZ)
pos_all = d["positions"].astype(np.float64)
eid_all = d["element_ids"].astype(np.int64)
lost_idx = np.flatnonzero(eid_all < 0)
print(f"{len(pos_all):,} particles, {len(lost_idx):,} were lost by the "
      f"UNFIXED build\n")

post = CASE / "post"
pv = sorted(post.glob("*_34.pvtu"))[0]
pattern = re.sub(r"_(\d+)\.pvtu$", "_{timestep}.pvtu", pv.name)
ts = int(re.search(r"_(\d+)\.pvtu$", pv.name).group(1))
npos, conn, vel = load_velocity_sequence_from_pvtu(
    base_path=post, file_pattern=pattern, timestep_range=(ts, ts),
    field_name="Displacement", verbose=False)
npos, conn, _, _ = deduplicate_nodes(npos, conn, velocity_sequence=vel,
                                     verbose=False)
conn = conn.astype(np.int32)

# WITH the Pass-2 fix in place
cells = extract_octree_cells_parent_cube(npos, conn, tolerance=1e-6,
                                         verbose=False)
oct_gpu = upload_mesh_aligned_octree_to_gpu(conn, npos, cells, verbose=False)
lev = np.asarray(cells.cell_levels)
sz = np.asarray(cells.cell_sizes)
gidx = np.asarray(cells.cell_grid_indices).astype(np.int64)
lcs = np.asarray(oct_gpu.level_cell_sizes)
print(f"octree WITH fix: {cells.n_cells:,} cells, "
      f"{cells.elements_per_cell_mean:.2f} elem/cell "
      f"(max {cells.max_elements_per_cell})\n")

offs = np.asarray(cells.cell_to_elements_offsets)
data = np.asarray(cells.cell_to_elements_data)
cell_of = np.repeat(np.arange(len(offs) - 1), np.diff(offs))
order = np.argsort(data, kind="stable")
es, cs = data[order], cell_of[order]
st = np.searchsorted(es, np.arange(len(conn)), "left")
en = np.searchsorted(es, np.arange(len(conn)), "right")

v = npos[conn]
emin, emax = v.min(axis=1), v.max(axis=1)
TOL = float(config.POINT_IN_TET_TOLERANCE)

def inside(p, idx):
    a = v[idx,1]-v[idx,0]; b = v[idx,2]-v[idx,0]; c = v[idx,3]-v[idx,0]
    r = p - v[idx,0]
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    bad = np.abs(det) < 1e-20
    inv = 1.0/np.where(bad, 1.0, det)
    b1 = np.einsum("ij,ij->i", r, np.cross(b, c))*inv
    b2 = np.einsum("ij,ij->i", a, np.cross(r, c))*inv
    b3 = np.einsum("ij,ij->i", a, np.cross(b, r))*inv
    b0 = 1.0-b1-b2-b3
    t = -TOL
    return (b0>=t)&(b1>=t)&(b2>=t)&(b3>=t)&(~bad)

rng = np.random.default_rng(42)
sel = rng.choice(lost_idx, min(N, len(lost_idx)), replace=False)
qs = jnp.asarray(pos_all[sel], dtype=config.FLOAT_DTYPE_JNP)
srch = jax.jit(jax.vmap(
    lambda q: search_mesh_aligned_octree_multi_local_where(q, oct_gpu)[0]))
got = np.asarray(srch(qs))

# Which elements are Kuhn? Recomputing the classification tells us whether a
# still-failing host's cells came from Pass 1 or Pass 2.
from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import (
    find_axis_aligned_edges_single)

print("=" * 66)
print(f"RESULT: of {len(sel):,} previously-lost particles, the FIXED octree")
print(f"  finds a host for : {int((got >= 0).sum()):,} "
      f"({100*(got >= 0).mean():.1f}%)")
print(f"  still loses      : {int((got < 0).sum()):,} "
      f"({100*(got < 0).mean():.1f}%)")
print("=" * 66)
print()

still = np.flatnonzero(got < 0)
if len(still) == 0:
    print("No residual failures in this sample -- the Pass-2 fix is sufficient.")
    sys.exit(0)

nohost = 0
mism_kuhn = mism_nonkuhn = matched = 0
deltas = []
for n in still:
    pi = sel[n]
    q = pos_all[pi]
    cand = np.flatnonzero((emin[:,0]<=q[0])&(emax[:,0]>=q[0])&
                          (emin[:,1]<=q[1])&(emax[:,1]>=q[1])&
                          (emin[:,2]<=q[2])&(emax[:,2]>=q[2]))
    host = -1
    if len(cand):
        ok = inside(q, cand)
        if ok.any():
            host = int(cand[np.flatnonzero(ok)[0]])
    if host < 0:
        nohost += 1
        continue
    rows = cs[st[host]:en[host]]
    if len(rows) == 0:
        continue
    cell_size_h, _ = find_axis_aligned_edges_single(npos[conn[host]], 1e-6)
    host_is_kuhn = not np.any(cell_size_h == 0)
    mismatched = False; best = None
    for r in rows:
        Lv = int(lev[r]); pitch = lcs[Lv]
        if not np.all(pitch > 0):
            continue
        rel = np.abs(sz[r]-pitch)/np.maximum(np.abs(pitch), 1e-300)
        if (rel > 1e-6).any():
            mismatched = True
        base = np.floor(q/pitch).astype(np.int64)
        dd = int(np.abs(base-gidx[r]).max())
        best = dd if best is None else min(best, dd)
    if mismatched:
        if host_is_kuhn:
            mism_kuhn += 1
        else:
            mism_nonkuhn += 1
    else:
        matched += 1
    if best is not None:
        deltas.append(best)

print(f"of the {len(still):,} still lost:")
print(f"  no containing element            : {nohost:,}")
print(f"  host on MISMATCHED cell, KUHN    : {mism_kuhn:,}   <- Pass 1 origin")
print(f"  host on MISMATCHED cell, non-Kuhn: {mism_nonkuhn:,}   <- Pass 2 origin")
print(f"  host on MATCHING cell            : {matched:,}")
if deltas:
    a = np.array(deltas)
    print(f"\n  index delta: median {np.median(a):.0f}  p90 "
          f"{np.percentile(a,90):.0f}  max {a.max()}")
    for lim in (1, 2, 3):
        print(f"    delta <= {lim}: {100*(a<=lim).mean():5.1f}%")
print()
tot_m = mism_kuhn + mism_nonkuhn
if tot_m and mism_kuhn > mism_nonkuhn:
    print("VERDICT: the residual failures are dominated by PASS-1 (Kuhn) cells.")
    print("  -> fix the lookup pitch ONCE, before either pass registers,")
    print("     so both passes and the search share one lattice.")
elif tot_m:
    print("VERDICT: residual failures are mostly on Pass-2 cells; the Pass-2")
    print("  fix is incomplete in some other way -- investigate before patching.")
else:
    print("VERDICT: residual failures are NOT lattice-related; another cause.")
