#!/usr/bin/env python3
"""WHY do D2's level-14 cells span a factor of 2 while A1's do not?

Hypothesis: `level` is derived from the MEAN of an anisotropic cell_size
    level = round(-log2(mean([dx,dy,dz])))
so two cells with the same level can have different per-axis sizes if their
ANISOTROPY differs. A1's cells are near-cubic (dx=dy=dz) so one level == one
size. D2's refined/tilted region has anisotropic cells (e.g. dx = 2*dy), which
land on the same level with genuinely different dx.

This prints, per level, the distinct (dx,dy,dz) triples actually present.
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
lev = np.asarray(cells.cell_levels); sz = np.asarray(cells.cell_sizes)

print(f"\n{case.name}: {cells.n_cells:,} cells\n")
for L in sorted(np.unique(lev)):
    m = lev == L
    s = sz[m]
    # distinct size triples, quantised to kill float noise
    q = np.round(s / s.min() * 1e4).astype(np.int64)
    uniq, counts = np.unique(q, axis=0, return_counts=True)
    order = np.argsort(-counts)
    aniso = (s.max(axis=1) / np.maximum(s.min(axis=1), 1e-300))
    print(f"level {int(L):2d}: {int(m.sum()):9,} cells, "
          f"{len(uniq)} distinct size-triples, "
          f"anisotropy max/min per cell: median {np.median(aniso):.3f} max {aniso.max():.3f}")
    for idx in order[:4]:
        rep = s[np.all(q == uniq[idx], axis=1)][0]
        print(f"      {counts[idx]:9,} cells  "
              f"dx,dy,dz = {rep[0]:.4e} {rep[1]:.4e} {rep[2]:.4e}  "
              f"(dx/dz = {rep[0]/rep[2]:.3f})")
