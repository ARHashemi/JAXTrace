#!/usr/bin/env python3
"""Would splitting over-full octree cells to a finer level actually help?

The per-cell search cost is  n_levels x 27 x MAX_ELEMS_PER_CELL, and every
level is walked unconditionally (fori_loop(0, n_levels)). So splitting trades
+1 level (+11% at 9 levels) against a lower bound (-59% if 79 -> 32). On paper
that is a 2.2x win -- but only if splitting genuinely caps occupancy.

It may not, and this measures whether it does. Occupancy has two parts:

  Kuhn elements      registered by PARENT CUBE -> 1 cell each. Halving the
                     cell size moves each into one of 8 sub-cells, so their
                     per-cell count drops ~8x.
  non-Kuhn elements  registered by AABB OVERLAP -> every cell the AABB
                     touches. The AABB is a fixed physical size, so halving
                     the cell doubles the span per axis and the element lands
                     in ~8x MORE cells. The number of non-Kuhn elements
                     covering any given point is unchanged.

If the over-full cells are dominated by AABB pileup, splitting cannot fix the
tail and only inflates the CSR.

Reports, for the real octree:
  A  occupancy distribution, and the Kuhn vs non-Kuhn composition of the
     over-full cells specifically
  B  for the worst cells, how many of their elements are there by parent-cube
     vs by AABB overlap
  C  a direct simulation: re-register every element one level finer and
     measure the resulting occupancy distribution, so the predicted bound is
     measured rather than argued
  D  the cost verdict: tests per query before vs after, including the extra
     level, and the CSR size change

Usage: eval_cell_splitting.py <case.gid>
Read-only.
"""
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

sys.path.insert(0, "/flash/shared/jax/JAXTrace")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CASE = Path(sys.argv[1])

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
data = np.asarray(g.cell_to_elements_data).astype(np.int64)
lev = np.asarray(cells.cell_levels).astype(np.int64)
occ = np.diff(off)
n_levels = int(g.max_level) - int(g.min_level) + 1
print(f"  {cells.n_cells:,} cells, mean {occ.mean():.2f}, max {occ.max()}, "
      f"CSR {len(data):,} entries")
print(f"  levels walked: {int(g.min_level)}..{int(g.max_level)} "
      f"({n_levels} levels)")

# Classify every element once.
print("classifying Kuhn / non-Kuhn...", flush=True)
V = npos[conn]
is_kuhn = np.zeros(len(conn), dtype=bool)
elem_size = np.zeros((len(conn), 3))
elem_lvl = np.zeros(len(conn), dtype=np.int64)
for e in range(len(conn)):
    cs, lv = find_axis_aligned_edges_single(V[e], 1e-6)
    is_kuhn[e] = bool(np.all(cs > 0))
    elem_size[e] = cs
    elem_lvl[e] = int(lv)
print(f"  Kuhn {int(is_kuhn.sum()):,} ({100*is_kuhn.mean():.1f}%), "
      f"non-Kuhn {int((~is_kuhn).sum()):,} ({100*(~is_kuhn).mean():.1f}%)")

# ---------------------------------------------------------------------------
banner("A  occupancy distribution, and what fills the over-full cells")
# ---------------------------------------------------------------------------
for lo, hi in ((1, 8), (9, 16), (17, 32), (33, 48), (49, 64), (65, 128)):
    m = (occ >= lo) & (occ <= hi)
    print(f"  occupancy {lo:3d}-{hi:3d}: {int(m.sum()):9,} cells "
          f"({100*m.mean():5.2f}%)")
over = occ > 32
print(f"\n  cells over 32: {int(over.sum()):,} ({100*over.mean():.2f}%)")
print(f"  CSR entries in them: {int(occ[over].sum()):,} "
      f"({100*occ[over].sum()/len(data):.1f}% of all entries)")

# Composition of the worst cells
idx_over = np.flatnonzero(over)
rs = np.random.RandomState(0)
samp = rs.choice(idx_over, min(3000, len(idx_over)), replace=False)
kk = nn = 0
for ci in samp:
    es = data[off[ci]:off[ci + 1]]
    kk += int(is_kuhn[es].sum())
    nn += int((~is_kuhn[es]).sum())
tot = kk + nn
print(f"\n  composition of {len(samp):,} sampled over-full cells:")
print(f"    Kuhn     entries: {kk:8,} ({100*kk/tot:5.1f}%)  <- splitting "
      f"divides these ~8x")
print(f"    non-Kuhn entries: {nn:8,} ({100*nn/tot:5.1f}%)  <- AABB pileup, "
      f"splitting does NOT reduce")

# ---------------------------------------------------------------------------
banner("B  direct simulation: re-register one level finer")
# ---------------------------------------------------------------------------
# Halve the pitch at every level and re-register exactly as the builder does:
# Kuhn by parent cube (1 cell), non-Kuhn by AABB overlap (every cell touched).
lcs = np.asarray(g.level_cell_sizes).astype(np.float64)
pitch = {int(l): lcs[int(l)] for l in np.unique(lev)}
new_counts = Counter()
new_entries = 0
print("  re-registering (this is the expensive part)...", flush=True)
for e in range(len(conn)):
    lv = int(elem_lvl[e])
    p = pitch.get(lv)
    if p is None or not np.all(p > 0):
        continue
    p = p / 2.0                      # one level finer
    t = V[e]
    if is_kuhn[e]:
        c = t.mean(axis=0)
        key = (lv + 1, int(np.floor(c[0] / p[0])),
               int(np.floor(c[1] / p[1])), int(np.floor(c[2] / p[2])))
        new_counts[key] += 1
        new_entries += 1
    else:
        lo_, hi_ = t.min(axis=0), t.max(axis=0)
        i0, i1 = int(np.floor(lo_[0] / p[0])), int(np.floor(hi_[0] / p[0]))
        j0, j1 = int(np.floor(lo_[1] / p[1])), int(np.floor(hi_[1] / p[1]))
        k0, k1 = int(np.floor(lo_[2] / p[2])), int(np.floor(hi_[2] / p[2]))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    new_counts[(lv + 1, i, j, k)] += 1
                    new_entries += 1
    if (e + 1) % 2_000_000 == 0:
        print(f"    {e+1:,}/{len(conn):,}", flush=True)

no = np.array(list(new_counts.values()))
print(f"\n  SPLIT octree (one level finer):")
print(f"    cells       : {len(no):,}  (was {cells.n_cells:,}, "
      f"{len(no)/cells.n_cells:.2f}x)")
print(f"    mean occ    : {no.mean():.2f}  (was {occ.mean():.2f})")
print(f"    MAX occ     : {no.max()}     (was {occ.max()})")
print(f"    CSR entries : {new_entries:,}  (was {len(data):,}, "
      f"{new_entries/len(data):.2f}x)")
for q in (50, 90, 99, 99.9):
    print(f"    p{q:<5} occ  : {np.percentile(no, q):.0f}")

# ---------------------------------------------------------------------------
banner("C  cost verdict")
# ---------------------------------------------------------------------------
b_now, b_split = int(occ.max()), int(no.max())
t_now = n_levels * 27 * b_now
t_split = (n_levels + 1) * 27 * b_split
print(f"  tests per L2 query = n_levels x 27 x bound")
print(f"    now   : {n_levels} x 27 x {b_now:3d} = {t_now:7,}")
print(f"    split : {n_levels+1} x 27 x {b_split:3d} = {t_split:7,}")
print(f"    ratio : {t_split/t_now:.2f}x  "
      f"({'CHEAPER' if t_split < t_now else 'MORE EXPENSIVE'})")
print(f"\n  memory: CSR {len(data):,} -> {new_entries:,} entries "
      f"({new_entries/len(data):.2f}x)")
print(f"  build time: the AABB enumeration loop grows with the cell count")
print(f"\n  NOTE the bound is set by the MAX, not the mean, because the loop")
print(f"  is static. A split that lowers the mean but not the tail buys")
print(f"  nothing for search cost.")
