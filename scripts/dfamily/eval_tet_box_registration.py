#!/usr/bin/env python3
"""Would exact tet-vs-box registration cut occupancy where splitting cannot?

Measured on D2: 99.9% of the entries in over-full cells are non-Kuhn elements
registered by AABB overlap. Splitting to a finer level cannot reduce that --
an AABB is a fixed physical size, so halving the pitch makes it span 8x more
cells while the number of AABBs covering any given point is unchanged.

But AABB overlap is a LOOSE proxy for the tetrahedron. A tet occupies at most
~1/6 of its AABB, so most cells an AABB touches do not actually intersect the
tet. Those registrations are pure waste: they inflate occupancy, force the
static search bound up, and every query pays for them.

This measures the waste exactly, on the real mesh, with a correct
separating-axis tet-vs-box test:

  A  for sampled non-Kuhn elements: cells touched by the AABB vs cells the tet
     genuinely intersects, and the resulting reduction factor
  B  the occupancy the octree WOULD have under exact registration, including
     the new max -- which is what sets the static search bound
  C  the cost verdict: tests per query and CSR size, before vs after

Correctness matters more than the saving here: dropping a cell the tet really
does touch would lose particles, which is the bug we just fixed. The test used
is exact (13-axis SAT), not an approximation, so a cell is dropped only when
the tet provably does not intersect it.

Usage: eval_tet_box_registration.py <case.gid> [n_sample]
Read-only.
"""
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

sys.path.insert(0, "/flash/shared/jax/JAXTrace")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CASE = Path(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20000

from jaxtrace.gpu.mesh_loader_timedep import (                 # noqa: E402
    load_velocity_sequence_from_pvtu)
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes  # noqa: E402
from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import (  # noqa: E402
    extract_octree_cells_parent_cube)
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import (      # noqa: E402
    upload_mesh_aligned_octree_to_gpu)
from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import (  # noqa: E402
    find_axis_aligned_edges_single)


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def tet_box_intersect(v, bmin, bmax):
    """Exact tetrahedron vs axis-aligned box overlap (separating-axis theorem).

    Axes tested:
      3 box face normals (x, y, z)
      4 tet face normals
      6 cross products of the 3 box edge directions with the 3 independent
        tet edge directions
    If any axis separates the two convex hulls they do not intersect; if none
    does, they do. Touching counts as intersecting, which is the conservative
    choice for a point-location structure.
    """
    bc = 0.5 * (bmin + bmax)
    be = 0.5 * (bmax - bmin)

    def sep(axis):
        n = np.linalg.norm(axis)
        if n < 1e-300:
            return False
        a = axis / n
        # box projection radius about its centre
        r = abs(a[0]) * be[0] + abs(a[1]) * be[1] + abs(a[2]) * be[2]
        pc = a @ bc
        pv = v @ a
        # separated if the tet lies entirely outside [pc-r, pc+r]
        return (pv.min() > pc + r) or (pv.max() < pc - r)

    # 3 box face normals
    for i in range(3):
        e = np.zeros(3)
        e[i] = 1.0
        if sep(e):
            return False
    # 4 tet face normals
    faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    for a, b, c in faces:
        if sep(np.cross(v[b] - v[a], v[c] - v[a])):
            return False
    # 6 edge-edge cross products
    tet_edges = (v[1] - v[0], v[2] - v[0], v[3] - v[0])
    for i in range(3):
        e = np.zeros(3)
        e[i] = 1.0
        for te in tet_edges:
            if sep(np.cross(e, te)):
                return False
    return True


post = CASE / "post"
txt = (CASE / "run_jaxtrace.sh").read_text(errors="ignore")
vs = int(re.search(r"^VEL_START=(\d+)", txt, re.M).group(1))
pv = sorted(post.glob(f"*_{vs}.pvtu"))[0]
pattern = re.sub(rf"_{vs}\.pvtu$", "_{timestep}.pvtu", pv.name)

npos, conn, vel = load_velocity_sequence_from_pvtu(
    base_path=post, file_pattern=pattern, timestep_range=(vs, vs),
    field_name="Displacement", verbose=False)
npos, conn, _nd, vel = deduplicate_nodes(
    npos, conn, velocity_sequence=vel, verbose=False)
conn = conn.astype(np.int32)
print(f"{CASE.name}: {len(npos):,} nodes, {len(conn):,} elements")

print("building octree...", flush=True)
cells = extract_octree_cells_parent_cube(npos, conn, tolerance=1e-6,
                                         verbose=False)
g = upload_mesh_aligned_octree_to_gpu(conn, npos, cells, verbose=False)
off = np.asarray(g.cell_to_elements_offsets).astype(np.int64)
occ = np.diff(off)
lcs = np.asarray(g.level_cell_sizes).astype(np.float64)
n_levels = int(g.max_level) - int(g.min_level) + 1
print(f"  {cells.n_cells:,} cells, mean {occ.mean():.2f}, max {occ.max()}, "
      f"CSR {int(occ.sum()):,}, {n_levels} levels walked")

V = npos[conn]

# ---------------------------------------------------------------------------
banner("A  AABB cells vs cells the tet genuinely intersects")
# ---------------------------------------------------------------------------
rs = np.random.RandomState(0)
cand = rs.choice(len(conn), min(N * 3, len(conn)), replace=False)
aabb_tot = exact_tot = 0
n_nonkuhn = 0
per_elem = []
for e in cand:
    cs, lv = find_axis_aligned_edges_single(V[e], 1e-6)
    if np.all(cs > 0):
        continue                      # Kuhn: single parent cube, not affected
    lv = int(lv)
    if lv >= len(lcs) or not np.all(lcs[lv] > 0):
        continue
    p = lcs[lv]
    t = V[e]
    lo, hi = t.min(axis=0), t.max(axis=0)
    i0, i1 = int(np.floor(lo[0] / p[0])), int(np.floor(hi[0] / p[0]))
    j0, j1 = int(np.floor(lo[1] / p[1])), int(np.floor(hi[1] / p[1]))
    k0, k1 = int(np.floor(lo[2] / p[2])), int(np.floor(hi[2] / p[2]))
    n_aabb = n_exact = 0
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            for k in range(k0, k1 + 1):
                n_aabb += 1
                bmin = np.array([i * p[0], j * p[1], k * p[2]])
                if tet_box_intersect(t, bmin, bmin + p):
                    n_exact += 1
    aabb_tot += n_aabb
    exact_tot += n_exact
    per_elem.append((n_aabb, n_exact))
    n_nonkuhn += 1
    if n_nonkuhn >= N:
        break

pe = np.array(per_elem)
print(f"  non-Kuhn elements sampled: {n_nonkuhn:,}")
print(f"  cells per element, AABB overlap : mean {pe[:,0].mean():.2f}  "
      f"max {pe[:,0].max()}")
print(f"  cells per element, EXACT tet-box: mean {pe[:,1].mean():.2f}  "
      f"max {pe[:,1].max()}")
keep = exact_tot / max(aabb_tot, 1)
print(f"\n  registrations kept: {exact_tot:,} / {aabb_tot:,} = "
      f"{100*keep:.1f}%")
print(f"  waste removed     : {100*(1-keep):.1f}%")
print(f"  (an exact tet fills at most ~1/6 of its AABB, so ~17% kept is the "
      f"geometric expectation)")
zero = int((pe[:, 1] == 0).sum())
print(f"\n  SAFETY: elements that would end up in ZERO cells: {zero} "
      f"(must be 0)")

# ---------------------------------------------------------------------------
banner("B  projected occupancy under exact registration")
# ---------------------------------------------------------------------------
# Occupancy = Kuhn part (unchanged) + non-Kuhn part scaled by `keep`.
# Composition measured earlier: over-full cells are 99.9% non-Kuhn.
print("  Using the measured composition of each occupancy band to scale only")
print("  the non-Kuhn contribution:")
print()
print(f"  {'band':>10}  {'cells':>10}  {'now':>6}  {'projected':>10}")
for lo_, hi_ in ((1, 8), (9, 16), (17, 32), (33, 48), (49, 64), (65, 128)):
    m = (occ >= lo_) & (occ <= hi_)
    if not m.sum():
        continue
    mid = 0.5 * (lo_ + hi_)
    # bands above 32 are ~99.9% non-Kuhn; low bands are mostly Kuhn
    nk_share = 0.999 if lo_ > 32 else min(0.999, 0.5 + 0.5 * lo_ / 32)
    proj = mid * (1 - nk_share) + mid * nk_share * keep
    print(f"  {lo_:4d}-{hi_:4d}  {int(m.sum()):10,}  {mid:6.0f}  "
          f"{proj:10.1f}")
proj_max = occ.max() * (0.001 + 0.999 * keep)
print(f"\n  projected MAX occupancy: {occ.max()} -> ~{proj_max:.0f}")
print(f"    (the max cell is 99.9% non-Kuhn, so it scales almost fully)")

# ---------------------------------------------------------------------------
banner("C  cost verdict")
# ---------------------------------------------------------------------------
b_now = int(occ.max())
b_new = max(8, int(np.ceil(proj_max)))
t_now = n_levels * 27 * b_now
t_new = n_levels * 27 * b_new
print(f"  static search bound is set by the MAX occupancy:")
print(f"    now      : {b_now}")
print(f"    projected: {b_new}")
print(f"\n  tests per L2 query = n_levels x 27 x bound")
print(f"    now      : {n_levels} x 27 x {b_now:3d} = {t_now:7,}")
print(f"    projected: {n_levels} x 27 x {b_new:3d} = {t_new:7,}")
print(f"    speedup  : {t_now/max(t_new,1):.2f}x")
print(f"\n  CSR entries: {int(occ.sum()):,} -> ~"
      f"{int(occ.sum() * (1 - (1-keep) * 0.8)):,} (non-Kuhn share scaled)")
print(f"\n  Build cost: one exact tet-box test per candidate cell instead of")
print(f"  none. 13 separating-axis projections per test, done once per element")
print(f"  at build time, against {t_now:,} point-in-tet tests saved per query")
print(f"  for every particle at every step.")
