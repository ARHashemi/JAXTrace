#!/usr/bin/env python3
"""Critically evaluate pitch-selection strategies for level_cell_sizes.

For each candidate rule (first / mean / median / dominant-mode / min / max),
compute for EVERY cell the index error |floor(c_lo/pitch) - stored_index| that
the search would incur, and report how many cells fall outside the +-1 the
3x3x3 neighbourhood can absorb.

This is the quantity that matters: a cell is findable iff its index error is
<= 1 on every axis.
"""
import os, re, sys
from pathlib import Path
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

case = Path(sys.argv[1])
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import extract_octree_cells_parent_cube

post = case / "post"
pv = sorted(post.glob("*_34.pvtu")) or sorted(post.glob("*.pvtu"))
pattern = re.sub(r"_(\d+)\.pvtu$", "_{timestep}.pvtu", pv[0].name)
ts = int(re.search(r"_(\d+)\.pvtu$", pv[0].name).group(1))
npos, conn, vel = load_velocity_sequence_from_pvtu(
    base_path=post, file_pattern=pattern, timestep_range=(ts, ts),
    field_name="Displacement", verbose=False)
npos, conn, nd, _ = deduplicate_nodes(npos, conn, velocity_sequence=vel, verbose=False)
conn = conn.astype(np.int32)
cells = extract_octree_cells_parent_cube(npos, conn, tolerance=1e-6, verbose=False)

lev = np.asarray(cells.cell_levels)
sz  = np.asarray(cells.cell_sizes)
gidx = np.asarray(cells.cell_grid_indices).astype(np.int64)

print(f"\n{case.name}: {cells.n_cells:,} cells\n")

def mode_size(s, rtol=0.05):
    """Most common size-triple, clustered in RELATIVE terms.

    An absolute quantisation (round(s*1e12)) keeps ~8 significant digits at
    these magnitudes, so a 1% physical jitter from mesh motion splits one real
    cell family into hundreds of distinct keys and the "mode" becomes a
    fragment. Cluster on log2 instead, with a 5% relative tolerance: cells
    whose sizes agree to within rtol on every axis are one family.
    """
    key = np.round(np.log2(np.maximum(s, 1e-300)) / np.log2(1.0 + rtol))
    uniq, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    win = inv == np.argmax(cnt)
    # Representative = median within the winning cluster, so jitter averages out
    return np.median(s[win], axis=0)


def mode_share(s, rtol=0.05):
    """Fraction of cells in the dominant relative-size cluster."""
    key = np.round(np.log2(np.maximum(s, 1e-300)) / np.log2(1.0 + rtol))
    _, cnt = np.unique(key, axis=0, return_counts=True)
    return float(cnt.max()) / len(s)

def _dom_mask(s, rtol=0.05):
    """Boolean mask of the dominant relative-size cluster."""
    key = np.round(np.log2(np.maximum(s, 1e-300)) / np.log2(1.0 + rtol))
    uniq, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return inv == np.argmax(cnt)


RULES = {
    "first (current)": lambda s: s[0],
    "mean":            lambda s: s.mean(axis=0),
    "median":          lambda s: np.median(s, axis=0),
    "dominant mode":   mode_size,
    "cluster median":  lambda s: np.median(s[_dom_mask(s)], axis=0),
    "min":             lambda s: s.min(axis=0),
    "max":             lambda s: s.max(axis=0),
}

print(f"{'rule':18} {'cells |err|>1':>14} {'%':>8}  {'worst |err|':>11}")
print("-" * 58)
results = {}
for name, fn in RULES.items():
    bad = 0; worst = 0
    for Lv in np.unique(lev):
        m = lev == Lv
        s = sz[m]; g = gidx[m]
        pitch = np.asarray(fn(s), dtype=np.float64)
        pitch = np.where(pitch <= 0, 1e-300, pitch)
        # The cell's low corner, reconstructed from its OWN stored index+size.
        # A point inside that cell is what the search must locate.
        lo = g * s
        # Probe the cell CENTRE: the most favourable point for the search.
        probe = lo + 0.5 * s
        k = np.floor(probe / pitch).astype(np.int64)
        err = np.abs(k - g).max(axis=1)
        bad += int((err > 1).sum()); worst = max(worst, int(err.max()))
    results[name] = (bad, worst)
    print(f"{name:18} {bad:>14,} {100*bad/cells.n_cells:>7.2f}%  {worst:>11,}")

print()

# ---- where do the residual (unfindable) cells sit? ----------------------
# This answers "what do we do with the minority that disobeys the dominant
# per-level size?" — first we need to know how many there are and whether
# they are clustered (e.g. at refinement boundaries) or scattered.
print("=" * 70)
print("RESIDUAL ANALYSIS for the dominant-mode rule")
print("=" * 70)
bad_lo = []
for Lv in np.unique(lev):
    m = lev == Lv
    s = sz[m]; g = gidx[m]
    pitch = np.asarray(mode_size(s), dtype=np.float64)
    pitch = np.where(pitch <= 0, 1e-300, pitch)
    lo = g * s
    probe = lo + 0.5 * s
    k = np.floor(probe / pitch).astype(np.int64)
    err = np.abs(k - g).max(axis=1)
    nbad = int((err > 1).sum())
    frac_dom = mode_share(s)
    print(f"  level {int(Lv):2d}: {int(m.sum()):9,} cells, "
          f"dominant share {100*frac_dom:5.1f}%, "
          f"unfindable {nbad:8,} ({100*nbad/max(int(m.sum()),1):5.2f}%)")
    if nbad:
        bad_lo.append(lo[err > 1])

if bad_lo:
    B = np.vstack(bad_lo)
    r = np.hypot(B[:, 0], B[:, 1]) * 1000.0
    z = B[:, 2] * 1000.0
    print()
    print(f"  residual cells: {len(B):,}")
    print(f"    r (mm): min {r.min():.2f} median {np.median(r):.2f} max {r.max():.2f}")
    print(f"    z (mm): min {z.min():.2f} median {np.median(z):.2f} max {z.max():.2f}")
    print(f"    within r<8mm (the tool region): {int((r < 8).sum()):,} "
          f"({100*(r < 8).mean():.1f}%)")
    print()
    print("  If these cluster in the tool region they matter more than their")
    print("  count suggests: that is where the particles actually travel.")
else:
    print()
    print("  No unfindable cells under the dominant-mode rule: a per-axis")
    print("  per-level cuboid pitch indexes EVERY cell correctly.")

print()
best = min(results, key=lambda k: results[k][0])
print(f"Best rule by unfindable-cell count: {best}  "
      f"({results[best][0]:,} cells, {100*results[best][0]/cells.n_cells:.2f}%)")
print()
print("NOTE: this probes each cell's CENTRE, the easiest case. A particle near")
print("a cell face is harder, so these counts are a LOWER bound on the real")
print("failure rate.")
