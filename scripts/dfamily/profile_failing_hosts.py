#!/usr/bin/env python3
"""What are the elements the search fails to find, and why does it miss them?

Input: the freeze positions from a --trace-substages run (death_pos.npy, the
k1 position of each frozen particle's FIRST failure). For each one we find the
true host by brute force and then interrogate it:

  A  Shape: Kuhn or non-Kuhn, volume, edge lengths, aspect ratio, and -- the
     question that matters here -- WHICH AXIS is elongated. The earlier D-family
     bug was a z-pitch disagreement; if these hosts are elongated in x or y
     instead, the same mechanism is biting on an axis the first patch did not
     exercise.

  B  Registration: how many cells hold the element, by which pass
     (1 cell = Pass 1 parent_cube, many = Pass 2 AABB), and at which level.

  C  Reachability -- the decisive test. The search computes
         base = floor(pos / level_cell_sizes[level])
     and scans base + {-1,0,1}^3. For each cell the host is registered in, we
     compute the index delta from that base. delta <= 1 means the search DID
     visit the cell holding the element, so the miss is in the point-in-tet
     test, not the indexing. delta >= 2 means the cell is out of reach and the
     element is invisible however good the predicate is.

  D  If out of reach: by how much, on which axis, and would a 5x5x5
     neighbourhood (delta <= 2) or registering into AABB-neighbour cells fix it?

  E  Per-element pitch mismatch: the element's own cell_size against the level
     median the lattice uses. This is the quantity the first patch equalised
     BETWEEN the two passes; it does not make a single median correct for every
     element, and this measures the residual.

Usage:
  profile_failing_hosts.py <death_pos.npy> <case.gid> [n_sample]

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

POS = Path(sys.argv[1])
CASE = Path(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 400

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


P = np.load(POS).astype(np.float64)
print(f"{POS.name}: {len(P):,} freeze positions")

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
print(f"  mesh: {len(npos):,} nodes, {len(conn):,} elements")

print("  building octree (production path)...", flush=True)
cells = extract_octree_cells_parent_cube(npos, conn, tolerance=1e-6,
                                         verbose=False)
oct_gpu = upload_mesh_aligned_octree_to_gpu(conn, npos, cells, verbose=False)
cell_lev = np.asarray(cells.cell_levels).astype(np.int64)
cell_sz = np.asarray(cells.cell_sizes).astype(np.float64)
cell_gidx = np.asarray(cells.cell_grid_indices).astype(np.int64)
lcs = np.asarray(oct_gpu.level_cell_sizes).astype(np.float64)
print(f"  octree: {cells.n_cells:,} cells, "
      f"{cells.elements_per_cell_mean:.2f} elem/cell")

# CSR inverse: element -> cells, via a sorted (elem, cell) pair array.
_off = np.asarray(oct_gpu.cell_to_elements_offsets).astype(np.int64)
_ee = np.asarray(oct_gpu.cell_to_elements_data).astype(np.int64)
_coe = np.repeat(np.arange(len(_off) - 1, dtype=np.int64), np.diff(_off))
_o = np.argsort(_ee, kind="stable")
_se, _sc = _ee[_o], _coe[_o]


def cells_of(e):
    lo = np.searchsorted(_se, e, side="left")
    hi = np.searchsorted(_se, e, side="right")
    return _sc[lo:hi]


V = npos[conn]
LO, HI = V.min(axis=1), V.max(axis=1)


def bary(x, t):
    T = np.column_stack([t[1] - t[0], t[2] - t[0], t[3] - t[0]])
    try:
        b = np.linalg.solve(T, x - t[0])
    except np.linalg.LinAlgError:
        return None
    return np.array([1.0 - b.sum(), b[0], b[1], b[2]])


def find_host(x):
    for c in np.flatnonzero(np.all((LO <= x) & (HI >= x), axis=1)):
        b = bary(x, V[c])
        if b is not None and b.min() >= -1e-9:
            return int(c)
    return -1


rs = np.random.RandomState(0)
samp = rs.choice(len(P), min(N, len(P)), replace=False)

hosts = []
for i in samp:
    hosts.append(find_host(P[i]))
hosts = np.array(hosts)
ok = hosts >= 0
print(f"\nsampled {len(samp)}: {int(ok.sum())} have a true host "
      f"({100 * ok.mean():.1f}%), {int((~ok).sum())} do not")

work_pos = P[samp][ok]
work_host = hosts[ok]
if len(work_host) == 0:
    print("no hosted freezes in this sample; nothing to profile")
    sys.exit(0)

# ---------------------------------------------------------------------------
banner("A  shape of the elements the search fails to find")
# ---------------------------------------------------------------------------
rows = []
for h in work_host:
    t = V[h]
    cs, lv = find_axis_aligned_edges_single(t, 1e-6)
    ext = t.max(axis=0) - t.min(axis=0)           # AABB extent per axis
    e = np.array([t[j] - t[i] for i in range(4) for j in range(i + 1, 4)])
    L = np.linalg.norm(e, axis=1)
    vol = abs(np.linalg.det(np.column_stack(
        [t[1] - t[0], t[2] - t[0], t[3] - t[0]]))) / 6.0
    rows.append(dict(h=int(h), kuhn=bool(np.all(cs > 0)), cs=cs, lv=int(lv),
                     ext=ext, Lmin=L.min(), Lmax=L.max(),
                     aspect=L.max() / max(L.min(), 1e-30), vol=vol))

nk = sum(1 for r in rows if not r["kuhn"])
print(f"  Kuhn      : {len(rows) - nk:,} / {len(rows):,} "
      f"({100 * (len(rows) - nk) / len(rows):.1f}%)")
print(f"  non-Kuhn  : {nk:,} / {len(rows):,} ({100 * nk / len(rows):.1f}%)")
print(f"  (mesh-wide non-Kuhn on D2 is ~50-59%, so compare against that)")

EXT = np.array([r["ext"] for r in rows])
print(f"\n  AABB extent per axis (mm):")
for a, ax in enumerate("xyz"):
    v = EXT[:, a] * 1000
    print(f"    {ax}: median {np.median(v):.4f}   p90 {np.percentile(v, 90):.4f}"
          f"   max {v.max():.4f}")

print(f"\n  WHICH AXIS IS LONGEST (the question: is this the z bug on another axis?)")
longest = EXT.argmax(axis=1)
for a, ax in enumerate("xyz"):
    c = int((longest == a).sum())
    print(f"    {ax} longest: {c:5,} ({100 * c / len(rows):5.1f}%)")
ratio = EXT.max(axis=1) / np.maximum(EXT.min(axis=1), 1e-30)
print(f"  extent anisotropy max/min: median {np.median(ratio):.3f}  "
      f"p90 {np.percentile(ratio, 90):.3f}  max {ratio.max():.3f}")
print(f"  aspect ratio (edges)     : median "
      f"{np.median([r['aspect'] for r in rows]):.3f}")

# ---------------------------------------------------------------------------
banner("B  how are those elements registered?")
# ---------------------------------------------------------------------------
ncells = np.array([len(cells_of(r["h"])) for r in rows])
print(f"  cells per element: min {ncells.min()}  median "
      f"{int(np.median(ncells))}  p90 {int(np.percentile(ncells, 90))}  "
      f"max {ncells.max()}")
print(f"    exactly 1 cell (Pass 1 parent_cube only): "
      f"{int((ncells == 1).sum()):,} ({100 * (ncells == 1).mean():.1f}%)")
print(f"    >1 cell  (Pass 2 AABB / multi)         : "
      f"{int((ncells > 1).sum()):,} ({100 * (ncells > 1).mean():.1f}%)")
print(f"  levels of the host elements: "
      f"{dict(Counter(r['lv'] for r in rows))}")

# ---------------------------------------------------------------------------
banner("C  DECISIVE: is the host's cell within the 3x3x3 the search scans?")
# ---------------------------------------------------------------------------
best_delta, per_axis = [], []
for x, r in zip(work_pos, rows):
    cl = cells_of(r["h"])
    bd, bax = 99, None
    for ci in cl:
        lv = int(cell_lev[ci])
        base = np.floor(x / lcs[lv]).astype(np.int64)
        dv = np.abs(base - cell_gidx[ci])
        if int(dv.max()) < bd:
            bd, bax = int(dv.max()), dv
    best_delta.append(bd)
    per_axis.append(bax if bax is not None else np.array([99, 99, 99]))
best_delta = np.array(best_delta)
per_axis = np.array(per_axis)

reach3 = best_delta <= 1
reach5 = best_delta <= 2
print(f"  index delta to the NEAREST cell holding the host:")
print(f"    median {int(np.median(best_delta))}  "
      f"p90 {int(np.percentile(best_delta, 90))}  max {int(best_delta.max())}")
print(f"\n    delta <= 1  (3x3x3 DOES visit it) : {int(reach3.sum()):,} "
      f"({100 * reach3.mean():.1f}%)")
print(f"    delta == 2  (5x5x5 would)         : "
      f"{int(((best_delta == 2)).sum()):,} "
      f"({100 * (best_delta == 2).mean():.1f}%)")
print(f"    delta >= 3  (neither)             : "
      f"{int((best_delta >= 3).sum()):,} "
      f"({100 * (best_delta >= 3).mean():.1f}%)")
print(f"\n  INTERPRETATION")
print(f"    delta<=1 dominant -> the cell IS scanned, so the miss is the")
print(f"      point-in-tet test or the per-cell element list, NOT the index.")
print(f"    delta>=2 dominant -> the cell is out of reach: this IS the old")
print(f"      lattice failure recurring, and 5x5x5 or neighbour registration")
print(f"      would fix it.")

oor = best_delta >= 2
if oor.any():
    print(f"\n  for the {int(oor.sum()):,} out-of-reach cases, delta per axis:")
    for a, ax in enumerate("xyz"):
        v = per_axis[oor][:, a]
        print(f"    {ax}: median {int(np.median(v))}  max {int(v.max())}   "
              f"(>=2 in {100 * (v >= 2).mean():.1f}% of them)")
    print(f"  -> the axis with the large delta is the one whose pitch "
          f"disagrees.")

# ---------------------------------------------------------------------------
banner("D  per-element pitch vs the level median the lattice uses")
# ---------------------------------------------------------------------------
print("  The first patch made Pass 1, Pass 2 and the search share ONE pitch")
print("  per level (the median). It did not make that median correct for")
print("  every element. This is the residual:")
mis = []
for r in rows:
    if not r["kuhn"]:
        continue
    lv = r["lv"]
    if lv >= len(lcs):
        continue
    own, med = r["cs"], lcs[lv]
    mis.append(own / np.maximum(med, 1e-30))
if mis:
    M = np.array(mis)
    print(f"\n  own cell_size / level median, per axis "
          f"(Kuhn hosts only, n={len(M):,}):")
    for a, ax in enumerate("xyz"):
        v = M[:, a]
        print(f"    {ax}: median {np.median(v):.4f}  p10 "
              f"{np.percentile(v, 10):.4f}  p90 {np.percentile(v, 90):.4f}")
    off = np.abs(M - 1.0).max(axis=1) > 0.05
    print(f"  hosts whose own size differs from the median by >5% on some "
          f"axis: {int(off.sum()):,} ({100 * off.mean():.1f}%)")
    print(f"  -> a large fraction here means one median per level is too")
    print(f"     coarse for this mesh, and the real fix is a finer level")
    print(f"     split (or per-axis levels), not a wider search.")
else:
    print("  (all hosts non-Kuhn; no own-size to compare)")

banner("summary")
print(f"""hosted freezes profiled : {len(rows):,}
non-Kuhn share          : {100 * nk / len(rows):.1f}%
registered in 1 cell    : {100 * (ncells == 1).mean():.1f}%
longest axis            : x {100 * (longest == 0).mean():.0f}%  """
      f"""y {100 * (longest == 1).mean():.0f}%  z {100 * (longest == 2).mean():.0f}%
within the 3x3x3 scan   : {100 * reach3.mean():.1f}%
would need 5x5x5        : {100 * (best_delta == 2).mean():.1f}%
beyond even 5x5x5       : {100 * (best_delta >= 3).mean():.1f}%""")
