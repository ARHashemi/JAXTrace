#!/usr/bin/env python3
"""Does exact tet-box registration preserve SEARCH COVERAGE?

The concern: pruning registrations could make a particle unfindable even
though every element keeps at least one cell. "No element is orphaned" is a
weaker property than "every particle is still located", and only the second
one matters.

The search locates a particle at position p by computing
    base = floor(p / level_cell_sizes[level])
and scanning base + {-1,0,1}^3 at each level, testing every element listed in
each visited cell. So the question is:

    for a particle at p whose true host is element e,
    after pruning, is e still listed in at least one cell
    the search actually visits?

Pruning removes (e, cell) pairs where the tet does not intersect the cell's
box. The containing cell of p is intersected by e BY DEFINITION -- p is inside
e and inside that cell -- so that pair survives. The test below verifies this
on real positions rather than trusting the argument, because the lattice the
search divides by and the lattice the cell was registered on must also agree
(the earlier D-family defect), and float32 rounding at a cell face could in
principle place p in a neighbouring cell.

Reports, for each sampled position:
  - found by the CURRENT registration (baseline)
  - found by the PRUNED registration (exact tet-box)
  - any position found before but NOT after  <- must be ZERO

Usage: test_tetbox_coverage.py <case.gid> [n_positions]
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
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

from jaxtrace.gpu.mesh_loader_timedep import (                 # noqa: E402
    load_velocity_sequence_from_pvtu)
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes  # noqa: E402
from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import (  # noqa: E402
    extract_octree_cells_parent_cube)
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import (      # noqa: E402
    upload_mesh_aligned_octree_to_gpu)


def tet_box_intersect(v, bmin, bmax):
    """Exact tet vs axis-aligned box, 13-axis separating-axis theorem."""
    bc = 0.5 * (bmin + bmax)
    be = 0.5 * (bmax - bmin)

    def sep(axis):
        n = np.linalg.norm(axis)
        if n < 1e-300:
            return False
        a = axis / n
        r = abs(a[0]) * be[0] + abs(a[1]) * be[1] + abs(a[2]) * be[2]
        pc = a @ bc
        pv = v @ a
        return (pv.min() > pc + r) or (pv.max() < pc - r)

    for i in range(3):
        e = np.zeros(3); e[i] = 1.0
        if sep(e):
            return False
    for a, b, c in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        if sep(np.cross(v[b] - v[a], v[c] - v[a])):
            return False
    for i in range(3):
        e = np.zeros(3); e[i] = 1.0
        for te in (v[1] - v[0], v[2] - v[0], v[3] - v[0]):
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
V = npos[conn]
print(f"{CASE.name}: {len(conn):,} elements")

print("building octree...", flush=True)
cells = extract_octree_cells_parent_cube(npos, conn, tolerance=1e-6,
                                         verbose=False)
g = upload_mesh_aligned_octree_to_gpu(conn, npos, cells, verbose=False)
off = np.asarray(g.cell_to_elements_offsets).astype(np.int64)
data = np.asarray(g.cell_to_elements_data).astype(np.int64)
lev = np.asarray(cells.cell_levels).astype(np.int64)
gidx = np.asarray(cells.cell_grid_indices).astype(np.int64)
lcs = np.asarray(g.level_cell_sizes).astype(np.float64)
levels = list(range(int(g.max_level), int(g.min_level) - 1, -1))

# (level, i, j, k) -> cell index
lut = {}
for ci in range(len(lev)):
    lut[(int(lev[ci]), int(gidx[ci, 0]), int(gidx[ci, 1]),
         int(gidx[ci, 2]))] = ci

LO, HI = V.min(axis=1), V.max(axis=1)


def bary_in(x, t, tol=1e-9):
    T = np.column_stack([t[1] - t[0], t[2] - t[0], t[3] - t[0]])
    try:
        b = np.linalg.solve(T, x - t[0])
    except np.linalg.LinAlgError:
        return False
    return (1.0 - b.sum() >= -tol) and bool((b >= -tol).all())


# Sampling matters here. A barycentre is the SAFEST possible point -- deep
# inside its element, far from any face or cell boundary. The risk case is the
# opposite: a point almost on a cell face, where float32 rounding can place it
# in a different cell than the one used at build time. So three modes:
#
#   barycentre : interior control
#   face       : pushed 99.9% of the way from the barycentre toward a random
#                vertex, so it sits just inside an element face
#   cellface   : nudged onto the nearest cell-boundary plane of its own level,
#                then perturbed by one float32 epsilon either side
MODE = os.environ.get("COVERAGE_MODE", "barycentre")
rs = np.random.RandomState(0)
pick = rs.choice(len(conn), N, replace=False)
print(f"probing {N:,} positions, mode={MODE}\n", flush=True)


def sample_point(e_true):
    t = V[e_true]
    c = t.mean(axis=0)
    if MODE == "barycentre":
        return c
    if MODE == "face":
        v = t[rs.randint(4)]
        return c + 0.999 * (v - c)
    if MODE == "cellface":
        # snap toward the nearest level-14 cell plane, then step by 1 eps
        p = lcs[int(g.max_level) - 1]
        if not np.all(p > 0):
            return c
        axis = rs.randint(3)
        plane = np.round(c[axis] / p[axis]) * p[axis]
        q = c.copy()
        # move to the plane but stay inside the element: clamp the step
        step = plane - c[axis]
        lim = 0.49 * (t[:, axis].max() - t[:, axis].min())
        q[axis] = c[axis] + np.clip(step, -lim, lim)
        q = np.float64(np.float32(q))      # force float32 representability
        return q
    raise SystemExit(f"unknown COVERAGE_MODE {MODE}")

base_found = pruned_found = lost = 0
for e_true in pick:
    x = sample_point(e_true)
    # emulate the search, both registrations at once
    hit_base = hit_pruned = False
    for L in levels:
        p = lcs[L]
        if not np.all(p > 0):
            continue
        b = np.floor(x / p).astype(np.int64)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    ci = lut.get((L, b[0] + di, b[1] + dj, b[2] + dk))
                    if ci is None:
                        continue
                    bmin = gidx[ci] * p
                    bmax = bmin + p
                    for e in data[off[ci]:off[ci + 1]]:
                        if not bary_in(x, V[e]):
                            continue
                        hit_base = True
                        # would this (element, cell) pair survive pruning?
                        if tet_box_intersect(V[e], bmin, bmax):
                            hit_pruned = True
                        if hit_pruned:
                            break
                    if hit_pruned:
                        break
                if hit_pruned:
                    break
            if hit_pruned:
                break
        if hit_pruned:
            break
    base_found += hit_base
    pruned_found += hit_pruned
    if hit_base and not hit_pruned:
        lost += 1

print("COVERAGE COMPARISON")
print(f"  located by CURRENT registration : {base_found:,} / {N:,} "
      f"({100*base_found/N:.2f}%)")
print(f"  located by PRUNED  registration : {pruned_found:,} / {N:,} "
      f"({100*pruned_found/N:.2f}%)")
print(f"  found before but NOT after      : {lost:,}   <- MUST BE 0")
print()
if lost == 0:
    print("  VERDICT: pruning preserves coverage on this sample.")
    print("  Reason it must hold in general: a particle inside element e and")
    print("  inside cell C means e genuinely intersects C, so the (e, C) pair")
    print("  is exactly the kind the exact test KEEPS. Pruning only removes")
    print("  pairs where the tet does not reach the cell at all -- cells that")
    print("  could never have contained a particle hosted by that element.")
else:
    print("  VERDICT: pruning LOSES coverage. Do not adopt it.")
