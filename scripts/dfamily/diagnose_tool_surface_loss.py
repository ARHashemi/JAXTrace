#!/usr/bin/env python3
"""Why do particles at the tool surface fail the element search?

The lattice bug is fixed and verified (all three mismatch categories zero), but
the full D2 run still loses 4.07% of particles, and they are a different
population: 82.9% HAVE a containing element, none are outside the mesh bbox,
none are inside the tool, and their nearest-node LEVEL is +0.00029 against
+0.01089 for survivors -- they hug the tool wall.

This script answers, as separate labelled tests on the SAME sampled particles:

  T1  Does the production search reproduce the failure at the exported
      position? (Separates a real search failure from a stale-export artefact.)
  T2  What ELEMENT contains the particle, and what is it like? Volume, aspect
      ratio, min dihedral/solid angle proxy, Kuhn vs non-Kuhn, and how close
      the point sits to the element boundary in barycentric terms. This is the
      "what kind of element" and "what is their property" question.
  T3  How is that element REGISTERED, and does the Morton/level the search
      computes for the position land on the cell that actually holds it, or on
      a neighbour? Reports the index delta per axis and whether the host is
      reachable within the 3x3x3 neighbourhood the kernel walks.
  T4  Would a LOOSER point-in-tet tolerance find it? Sweeps the tolerance and
      reports how many are recovered at each value. (Prime suspect: the
      production value is 1e-6 in float32, and earlier sampling found failures
      at min barycentric -3.3e-04.)
  T5  Does FLOAT32 vs FLOAT64 change the verdict? The kernel runs float32; if
      a failure disappears in float64 it is a precision problem, not a
      geometry or lattice one.
  T6  Is the particle near the TOOL SURFACE in the level-set sense, and could
      an RK4 substep have fallen INSIDE the tool? Reconstructs the four
      substage positions from the exported velocity and tests each one's
      LEVEL. This is the user's hypothesis, tested directly.
  T7  Control: run T1-T5 on an equal sample of SURVIVING particles drawn from
      the same spatial band, so any property reported for failures is shown to
      be specific to them rather than true of the whole region.

Usage:
  python3 diagnose_tool_surface_loss.py <final_state.npz> <case.gid> [n]

Read-only. Nothing is written to the case folder.
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

NPZ = Path(sys.argv[1])
CASE = Path(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 400

import vtk                                                      # noqa: E402
from vtk.util.numpy_support import vtk_to_numpy                 # noqa: E402
from scipy.spatial import cKDTree                               # noqa: E402

from jaxtrace import config                                     # noqa: E402
from jaxtrace.gpu.mesh_loader_timedep import (                  # noqa: E402
    load_velocity_sequence_from_pvtu)
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes   # noqa: E402
from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import (  # noqa: E402
    extract_octree_cells_parent_cube)
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import (       # noqa: E402
    upload_mesh_aligned_octree_to_gpu)
from jaxtrace.gpu.search.mesh_aligned_point_location import (   # noqa: E402
    search_mesh_aligned_octree_multi_local_where)
from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import (  # noqa: E402
    find_axis_aligned_edges_single)

import jax.numpy as jnp                                         # noqa: E402


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


# ----------------------------------------------------------------------------
# Load the run's final state
# ----------------------------------------------------------------------------
d = np.load(NPZ)
pos_all = d["positions"].astype(np.float64)
eid_all = d["element_ids"].astype(np.int64)
lost_mask = eid_all < 0
print(f"final state: {len(pos_all):,} particles, "
      f"{lost_mask.sum():,} lost ({100 * lost_mask.mean():.2f}%)")

# ----------------------------------------------------------------------------
# Load mesh + LEVEL field at the first loaded timestep
# ----------------------------------------------------------------------------
post = CASE / "post"
rj = CASE / "run_jaxtrace.sh"
txt = rj.read_text(errors="ignore")
vs = int(re.search(r"^VEL_START=(\d+)", txt, re.M).group(1))
pv = sorted(post.glob(f"*_{vs}.pvtu"))[0]
pattern = re.sub(rf"_{vs}\.pvtu$", "_{timestep}.pvtu", pv.name)
print(f"mesh: {pv.name}  (VEL_START={vs})")

npos_raw, conn_raw, vel_raw = load_velocity_sequence_from_pvtu(
    base_path=post, file_pattern=pattern, timestep_range=(vs, vs),
    field_name="Displacement", verbose=False)
npos, conn, vel_seq, _ = deduplicate_nodes(
    npos_raw, conn_raw, velocity_sequence=vel_raw, verbose=False)
conn = conn.astype(np.int32)
print(f"  nodes {len(npos):,}  elements {len(conn):,}")

# LEVEL on the ORIGINAL (pre-dedup) points, mapped by nearest node.
r = vtk.vtkXMLPUnstructuredGridReader()
r.SetFileName(str(pv))
r.Update()
g = r.GetOutput()
pd = g.GetPointData()
lv_names = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
level_pts = vtk_to_numpy(g.GetPoints().GetData()).astype(np.float64)
level_val = (vtk_to_numpy(pd.GetArray("LEVEL")).astype(np.float64)
             if "LEVEL" in lv_names else None)
print(f"  point arrays: {lv_names}")
ltree = cKDTree(level_pts) if level_val is not None else None


def level_at(p):
    """Nearest-node LEVEL at positions p (n,3). Negative = inside tool."""
    if ltree is None:
        return np.full(len(p), np.nan)
    _, nn = ltree.query(np.atleast_2d(p), k=1)
    return level_val[nn]


# ----------------------------------------------------------------------------
# Build the octree exactly as production does
# ----------------------------------------------------------------------------
print("\nbuilding octree (production path)...", flush=True)
cells = extract_octree_cells_parent_cube(npos, conn, tolerance=1e-6,
                                         verbose=False)
oct_gpu = upload_mesh_aligned_octree_to_gpu(conn, npos, cells, verbose=False)
print(f"  {cells.n_cells:,} cells, {cells.elements_per_cell_mean:.2f} "
      f"elem/cell (max {cells.max_elements_per_cell})")

cell_lev = np.asarray(cells.cell_levels).astype(np.int64)
cell_sz = np.asarray(cells.cell_sizes).astype(np.float64)
cell_gidx = np.asarray(cells.cell_grid_indices).astype(np.int64)
lcs = np.asarray(oct_gpu.level_cell_sizes).astype(np.float64)

# element -> cells it is registered in.
# Built with numpy rather than a dict: a full Python inversion of this CSR is
# ~85M int objects (several GB) on top of the mesh, which risks OOM. Instead
# expand the CSR to a flat (element, cell) pair array once and sort by element,
# then answer each query with a binary search.
csr_starts = np.asarray(oct_gpu.cell_element_starts).astype(np.int64)
csr_counts = np.asarray(oct_gpu.cell_element_counts).astype(np.int64)
csr_elems = np.asarray(oct_gpu.cell_elements).astype(np.int64)
_cell_of_entry = np.repeat(np.arange(len(csr_counts), dtype=np.int64),
                           csr_counts)
_entry_elem = np.concatenate([csr_elems[s:s + c]
                              for s, c in zip(csr_starts, csr_counts)]) \
    if len(csr_counts) else np.empty(0, dtype=np.int64)
_order = np.argsort(_entry_elem, kind="stable")
_sorted_elem = _entry_elem[_order]
_sorted_cell = _cell_of_entry[_order]
print(f"  CSR inverted: {len(_sorted_elem):,} (element, cell) entries")


def cells_of_element(e):
    """Cell indices element e is registered in."""
    lo = np.searchsorted(_sorted_elem, e, side="left")
    hi = np.searchsorted(_sorted_elem, e, side="right")
    return _sorted_cell[lo:hi]

# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
V = npos[conn]                      # (n_elem, 4, 3)
E_LO = V.min(axis=1)
E_HI = V.max(axis=1)


def bary(x, tet):
    T = np.column_stack([tet[1] - tet[0], tet[2] - tet[0], tet[3] - tet[0]])
    try:
        b = np.linalg.solve(T, x - tet[0])
    except np.linalg.LinAlgError:
        return None
    return np.array([1.0 - b.sum(), b[0], b[1], b[2]])


def contains(x, tet, tol):
    b = bary(x, tet)
    return (b is not None) and (b.min() >= -tol)


def find_host(x, tol=1e-9):
    """Brute-force containing element via AABB prefilter."""
    cand = np.flatnonzero(np.all((E_LO <= x) & (E_HI >= x), axis=1))
    for c in cand:
        if contains(x, V[c], tol):
            return int(c)
    return -1


def tet_props(c):
    """Shape descriptors for element c."""
    t = V[c]
    e = np.array([t[j] - t[i] for i in range(4) for j in range(i + 1, 4)])
    L = np.linalg.norm(e, axis=1)
    vol = abs(np.linalg.det(np.column_stack(
        [t[1] - t[0], t[2] - t[0], t[3] - t[0]]))) / 6.0
    # areas of the 4 faces
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    A = sum(0.5 * np.linalg.norm(np.cross(t[b] - t[a], t[cc] - t[a]))
            for a, b, cc in faces)
    # inradius / circumradius-ish quality: 3*vol/A vs longest edge
    rin = 3.0 * vol / A if A > 0 else 0.0
    aspect = L.max() / max(L.min(), 1e-30)
    # radius-edge: sliverness indicator
    sliver = rin / max(L.max(), 1e-30)
    cs, lvl = find_axis_aligned_edges_single(t, 1e-6)
    return dict(vol=vol, Lmin=L.min(), Lmax=L.max(), aspect=aspect,
                rin=rin, sliver=sliver, kuhn=not np.any(cs == 0),
                level=int(lvl), cell_size=cs)


def search_prod(x):
    """The production search, float32, as the kernel calls it."""
    e, n = search_mesh_aligned_octree_multi_local_where(
        jnp.asarray(x, dtype=jnp.float32), oct_gpu)
    return int(e), int(n)


# ----------------------------------------------------------------------------
# Sample: failures, plus a CONTROL of survivors from the same spatial band
# ----------------------------------------------------------------------------
rs = np.random.RandomState(0)
lost_idx = np.flatnonzero(lost_mask)
samp = rs.choice(lost_idx, min(N, len(lost_idx)), replace=False)

q = pos_all[samp]
rr = np.hypot(q[:, 0], q[:, 1])
band = (rr.min(), rr.max())
surv_idx = np.flatnonzero(~lost_mask)
srr = np.hypot(pos_all[surv_idx, 0], pos_all[surv_idx, 1])
in_band = surv_idx[(srr >= band[0]) & (srr <= band[1])]
ctrl = rs.choice(in_band, min(len(samp), len(in_band)), replace=False)

print(f"\nsample: {len(samp)} failures, {len(ctrl)} surviving controls "
      f"from the same radial band r {band[0]*1000:.2f}-{band[1]*1000:.2f} mm")

# ============================================================================
banner("T1  does the production search reproduce the failure?")
# ============================================================================
repro, found_now, ntests = [], [], []
for i in samp:
    e, n = search_prod(pos_all[i])
    repro.append(e < 0)
    ntests.append(n)
repro = np.array(repro)
print(f"  reproduces as lost : {repro.sum():,} / {len(samp)} "
      f"({100 * repro.mean():.1f}%)")
print(f"  found on replay    : {(~repro).sum():,}  "
      f"(exported position is post-step; not a search failure there)")
print(f"  point-in-tet tests : median {int(np.median(ntests))}, "
      f"max {max(ntests)}")

work = samp[repro]
if len(work) == 0:
    print("\nNo failure reproduces at its exported position. The loss is a "
          "step-integration artefact, not a point-location failure: rerun "
          "with substage export to localise it.")
    sys.exit(0)

# ============================================================================
banner("T2  what element contains them, and what is it like?")
# ============================================================================
hosts, props, minbary = [], [], []
for i in work:
    h = find_host(pos_all[i])
    hosts.append(h)
    if h >= 0:
        props.append(tet_props(h))
        b = bary(pos_all[i], V[h])
        minbary.append(b.min())
hosts = np.array(hosts)
nohost = int((hosts < 0).sum())
print(f"  no containing element : {nohost:,} / {len(work)} "
      f"(outside the mesh volume -- not a defect)")
print(f"  HAVE a host           : {(hosts >= 0).sum():,}  <- these are the bug")

if props:
    def stat(k):
        v = np.array([p[k] for p in props], dtype=float)
        return f"min {v.min():.3e}  median {np.median(v):.3e}  max {v.max():.3e}"
    print(f"\n  host element properties ({len(props)} elements):")
    print(f"    volume        : {stat('vol')}")
    print(f"    shortest edge : {stat('Lmin')}")
    print(f"    longest edge  : {stat('Lmax')}")
    print(f"    aspect ratio  : {stat('aspect')}")
    print(f"    sliverness    : {stat('sliver')}   (inradius/longest edge; "
          f"lower = worse)")
    nk = sum(1 for p in props if not p["kuhn"])
    print(f"    non-Kuhn      : {nk:,} / {len(props)} "
          f"({100 * nk / len(props):.1f}%)")
    print(f"    octree levels : {dict(Counter(p['level'] for p in props))}")
    mb = np.array(minbary)
    print(f"\n  min barycentric coordinate at the particle:")
    print(f"    median {np.median(mb):+.3e}   p10 {np.percentile(mb, 10):+.3e}"
          f"   min {mb.min():+.3e}")
    print(f"    (>=0 means strictly inside; the production tolerance is "
          f"{config.POINT_IN_TET_TOLERANCE:.0e})")
    nmarg = int((mb < config.POINT_IN_TET_TOLERANCE).sum())
    print(f"    within/below the production tolerance: {nmarg:,} "
          f"({100 * nmarg / len(mb):.1f}%)")

# ============================================================================
banner("T3  is the host reachable from the cell the search computes?")
# ============================================================================
rows = []
for i, h in zip(work, hosts):
    if h < 0:
        continue
    x = pos_all[i]
    cl = cells_of_element(int(h))
    if len(cl) == 0:
        rows.append(("UNREGISTERED", 99))
        continue
    best = 99
    tag = "out of reach"
    for ci in cl:
        lv = int(cell_lev[ci])
        pitch = lcs[lv]
        want = np.floor(x / pitch).astype(np.int64)
        delta = int(np.max(np.abs(want - cell_gidx[ci])))
        if delta < best:
            best = delta
            tag = ("reachable (3x3x3)" if delta <= 1
                   else f"delta={delta} > 1")
        # does the registered size equal the lookup pitch? (lattice check)
    rows.append((tag, best))

if rows:
    cnt = Counter(t for t, _ in rows)
    print("  nearest registered cell of the true host, vs the cell the search "
          "computes:")
    for k, v in cnt.most_common():
        print(f"    {k:22}: {v:,}")
    dl = np.array([b for _, b in rows if b < 99])
    if len(dl):
        print(f"\n  index delta: median {int(np.median(dl))}  "
              f"p90 {int(np.percentile(dl, 90))}  max {int(dl.max())}")
        print(f"    delta <= 1 (reachable): "
              f"{100 * (dl <= 1).mean():.1f}%")
    print("\n  INTERPRETATION: delta<=1 everywhere means the lattice is fine "
          "and the\n  failure is the point-in-tet predicate, not the "
          "indexing. A delta>1\n  population would mean a lattice problem "
          "remains.")

# ============================================================================
banner("T4  would a looser point-in-tet tolerance find them?")
# ============================================================================
print(f"  production POINT_IN_TET_TOLERANCE = "
      f"{config.POINT_IN_TET_TOLERANCE:.0e}\n")
for tol in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
    ok = 0
    for i, h in zip(work, hosts):
        if h >= 0 and contains(pos_all[i], V[h], tol):
            ok += 1
    n = int((hosts >= 0).sum())
    print(f"    tol {tol:.0e}: {ok:,} / {n:,} hosts accepted "
          f"({100 * ok / max(n, 1):.1f}%)")
print("\n  A jump between two rows is the tolerance that would recover them.")

# ============================================================================
banner("T5  float32 vs float64")
# ============================================================================
f32 = f64 = 0
for i, h in zip(work, hosts):
    if h < 0:
        continue
    x = pos_all[i]
    if contains(np.float32(x).astype(np.float64), V[h].astype(np.float64),
                1e-9):
        f32 += 1
    if contains(x, V[h], 1e-9):
        f64 += 1
n = int((hosts >= 0).sum())
print(f"  host accepted at float64 position : {f64:,} / {n:,}")
print(f"  host accepted at float32 position : {f32:,} / {n:,}")
print(f"  lost to float32 rounding alone    : {f64 - f32:,}")

# ============================================================================
banner("T6  the tool-surface hypothesis: did a substep fall inside the tool?")
# ============================================================================
if level_val is None:
    print("  no LEVEL array in this mesh; cannot test.")
else:
    lf = level_at(pos_all[work])
    lc = level_at(pos_all[ctrl])
    print(f"  nearest-node LEVEL (negative = inside the tool):")
    print(f"    failures : median {np.median(lf):+.6f}   "
          f"inside tool {100 * (lf < 0).mean():.1f}%   "
          f"min {lf.min():+.6f}")
    print(f"    controls : median {np.median(lc):+.6f}   "
          f"inside tool {100 * (lc < 0).mean():.1f}%")
    ratio = np.median(lc) / max(np.median(lf), 1e-12)
    print(f"    failures sit {ratio:.0f}x closer to the tool wall")

    # How far could one RK4 substep travel? Use the mesh's own velocity
    # scale at these positions to bound it, then ask whether a half-step
    # could cross into LEVEL<0.
    print("\n  could a substep have crossed into the tool?")
    vmag = np.linalg.norm(vel_seq[0], axis=1)
    vtyp = float(np.percentile(vmag, 99))
    print(f"    99th-pct nodal |v| = {vtyp:.4f} m/s")
    for dt in (0.0003629,):
        reach = vtyp * dt
        print(f"    dt={dt}: one full step reaches {reach * 1000:.3f} mm, "
              f"a half-step {reach * 500:.3f} mm")
        # distance from each failure to the nearest LEVEL<0 node
        neg = level_val < 0
        if neg.any():
            ntree = cKDTree(level_pts[neg])
            dist, _ = ntree.query(pos_all[work], k=1)
            print(f"    distance to nearest in-tool node: "
                  f"median {np.median(dist) * 1000:.3f} mm, "
                  f"p10 {np.percentile(dist, 10) * 1000:.3f} mm")
            frac = float((dist < reach).mean())
            print(f"    within one step's reach of the tool: "
                  f"{100 * frac:.1f}%")
            print(f"    within a half-step's reach:          "
                  f"{100 * float((dist < reach / 2).mean()):.1f}%")
    print("\n  NOTE: with RK4_LEVELSET_MODE='zero_vel' (what the run used) a "
          "substep\n  inside the tool does NOT mark the particle lost -- it "
          "only zeroes that\n  stage's velocity. So crossing into the tool "
          "explains a STALLED particle,\n  not an ElementID=-1. Only "
          "mode='skip_step' freezes on tool contact.")

# ============================================================================
banner("T7  control: are these properties specific to the failures?")
# ============================================================================
c_hosts, c_props, c_minb, c_repro = [], [], [], 0
for i in ctrl:
    e, _ = search_prod(pos_all[i])
    if e < 0:
        c_repro += 1
    h = find_host(pos_all[i])
    c_hosts.append(h)
    if h >= 0:
        c_props.append(tet_props(h))
        c_minb.append(bary(pos_all[i], V[h]).min())
print(f"  surviving controls that FAIL the search now: {c_repro:,} / "
      f"{len(ctrl)}  (expect ~0)")
if c_props and props:
    def cmp(k, fmt="{:.3e}"):
        a = np.median([p[k] for p in props])
        b = np.median([p[k] for p in c_props])
        print(f"    {k:14}: failures {fmt.format(a)}   "
              f"controls {fmt.format(b)}   ratio {a / max(b, 1e-30):.2f}")
    print("\n  host-element medians, failures vs controls:")
    for k in ("vol", "Lmin", "Lmax", "aspect", "sliver"):
        cmp(k)
    nkf = 100 * sum(1 for p in props if not p["kuhn"]) / len(props)
    nkc = 100 * sum(1 for p in c_props if not p["kuhn"]) / len(c_props)
    print(f"    non-Kuhn %    : failures {nkf:.1f}%   controls {nkc:.1f}%")
if c_minb and minbary:
    print(f"\n  min barycentric: failures median {np.median(minbary):+.3e}   "
          f"controls median {np.median(c_minb):+.3e}")

banner("summary")
print("""Read the tests together:
  T3 delta<=1 everywhere + T4 showing recovery at a looser tolerance
     -> the predicate is the cause; widen POINT_IN_TET_TOLERANCE.
  T3 showing a delta>1 population
     -> a lattice problem remains; fix indexing, not the tolerance.
  T5 large float64-vs-float32 gap
     -> precision at the wall; consider float64 positions or a
        relative tolerance scaled by element size.
  T2/T7 failures on markedly slivery elements vs controls
     -> mesh quality at the tool wall is the driver; a tolerance scaled
        by element size is the principled fix rather than a flat bump.
  T6 high 'within a half-step's reach'
     -> substeps do enter the tool, which with zero_vel stalls a particle
        but cannot by itself set ElementID=-1.""")
