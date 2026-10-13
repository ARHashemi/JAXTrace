#!/usr/bin/env python3
"""Does every registered cell sit on the lattice the search looks up on?

The search computes a particle's cell index as

    (i, j, k) = floor(pos / level_cell_sizes[level])

and `upload_mesh_aligned_octree_to_gpu` sets `level_cell_sizes[level]` to the
per-axis median of the cell sizes at that level.  An element registered with a
*different* divisor is filed on a different grid than the one it is looked up
on; where the two disagree by more than the +-1 that the 3x3x3 neighbourhood
absorbs, the host cell is never tested and the particle is reported lost.

This test builds two synthetic meshes and checks the invariant

    cell_metadata[(morton, level)].grid_index
        == floor(cell_centre / per_level_median_size)

holds for every cell.  It needs no case data and runs in a second, so it is
the cheap regression guard for the D-family fix.

Usage:  python3 scripts/dfamily/test_lattice_invariant.py
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import (  # noqa: E402
    extract_octree_cells_parent_cube,
)


def kuhn_box(origin, size):
    """The 6 Kuhn tetrahedra tiling one axis-aligned box."""
    ox, oy, oz = origin
    dx, dy, dz = size
    v = np.array(
        [
            [ox, oy, oz], [ox + dx, oy, oz],
            [ox, oy + dy, oz], [ox + dx, oy + dy, oz],
            [ox, oy, oz + dz], [ox + dx, oy, oz + dz],
            [ox, oy + dy, oz + dz], [ox + dx, oy + dy, oz + dz],
        ],
        dtype=np.float64,
    )
    tets = np.array(
        [
            [0, 1, 3, 7], [0, 1, 5, 7], [0, 4, 5, 7],
            [0, 4, 6, 7], [0, 2, 6, 7], [0, 2, 3, 7],
        ],
        dtype=np.int32,
    )
    return v, tets


def build_mesh(sizes, nx=4, ny=4, z0=0.0):
    """Disjoint blocks of boxes, one block per entry in `sizes`.

    The blocks are separated along x so no nodes are shared between them;
    sharing nodes between differently-sized boxes makes the tetrahedra
    degenerate and they classify as non-Kuhn, which is not what we want to
    exercise here.  Each block tiles nx*ny*nz boxes of its own size, so a
    mesh built from several entries carries several *Kuhn* cell sizes that
    collapse onto the same octree level.

    `z0` lifts the whole slab away from the origin.  A pitch disagreement is
    a *relative* error, so its absolute index error grows with distance from
    the origin: at the origin it is 0 and only far out does it exceed the
    +-1 that the 3x3x3 search neighbourhood absorbs.  The real D meshes sit
    at z ~ -6 mm with ~47 um cells, i.e. ~128 cells from the origin.
    """
    nodes, conns = [], []
    x0 = 0.0
    for size in sizes:
        for i in range(nx):
            for j in range(ny):
                for k in range(2):
                    origin = (x0 + i * size[0], j * size[1],
                              z0 + k * size[2])
                    v, t = kuhn_box(origin, size)
                    conns.append(t + 8 * len(nodes))
                    nodes.append(v)
        # Leave a wide gap so the next block shares nothing with this one.
        x0 += (nx + 4) * size[0]
    return np.vstack(nodes), np.vstack(conns).astype(np.int32)


def check(name, node_positions, connectivity):
    cells = extract_octree_cells_parent_cube(
        node_positions, connectivity, verbose=False,
    )

    sizes = np.asarray(cells.cell_sizes)
    levels = np.asarray(cells.cell_levels).astype(int)
    idx = np.asarray(cells.cell_grid_indices)

    # Reproduce the pitch the GPU upload will publish for each level.
    pitch = {
        lv: np.median(sizes[levels == lv], axis=0) for lv in np.unique(levels)
    }

    bad = 0
    worst = 0
    for c in range(len(levels)):
        p = pitch[levels[c]]
        centre = (idx[c] + 0.5) * sizes[c]
        want = np.floor(centre / p).astype(int)
        d = int(np.max(np.abs(want - idx[c])))
        worst = max(worst, d)
        if d > 1:
            bad += 1

    n = len(levels)
    print(f"{name}")
    print(f"  cells                     : {n:,}")
    print(f"  distinct sizes per level  : "
          f"{ {int(lv): len(np.unique(sizes[levels == lv], axis=0)) for lv in np.unique(levels)} }")
    print(f"  max |index error|         : {worst}")
    print(f"  cells unreachable (err>1) : {bad:,} "
          f"({100.0 * bad / max(n, 1):.2f}%)")
    ok = bad == 0
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def main():
    s = 1.0 / 16.0
    results = []

    # Isotropic: one cubic size. The fix must be a no-op here (A/B/C meshes).
    nodes, conn = build_mesh([(s, s, s)] * 3)
    results.append(check("isotropic cubic (A/B/C-like)", nodes, conn))

    # Anisotropic: several sizes collapsing onto one level, the D-family
    # signature. 'level' comes from the MEAN of (dx,dy,dz), so these share it.
    nodes, conn = build_mesh(
        [
            (s, s, s * 1.1972),   # the 92.6%-majority cuboid on D2
            (s, s, s),            # the cubic minority
            (s, s, s * 0.80),
        ],
        z0=128 * s,               # ~128 cells out, as on the real mesh
    )
    results.append(check("anisotropic cuboid (D-family-like)", nodes, conn))

    if all(results):
        print("ALL PASS: every registered cell is reachable from the "
              "lattice the search divides by.")
        return 0
    print("FAILURE: some cells are registered off the lookup lattice; "
          "particles landing in them will be reported lost.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
