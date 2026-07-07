"""
HCT-3D reconstruction — Phase 1 + Phase 2.

Roadmap: docs/hct3d_implementation_plan.md.

This module implements the geometry precompute (Phase 1) and the
per-edge gradient DOF computation (Phase 2) for the Alfeld-split
piecewise-cubic Hermite reconstruction on tetrahedral meshes.

Nothing in this module is wired into run_tracking.py yet. Phase 3
(Bernstein coefficient solve via Worsey-Farin closed-form) and
Phase 4+5 (JAX kernel evaluator + validation gates) will land in
subsequent commits. Until then, ``jaxtrace.gpu.recovery.gradient_recovery``
continues to provide ``centroid_taylor`` and ``vertex_taylor`` as the
only user-selectable reconstruction methods.

What this module gives you today
--------------------------------

    build_alfeld_geometry(node_positions, connectivity)
        Precomputes per-element Alfeld-split geometry: parent centroid,
        sub-tet vertex indices, edge midpoints, face centroids, and
        sub-tet volumes. Returned as an ``AlfeldGeometry`` dataclass.

    edge_midpoint_gradients(node_positions, connectivity, nodal_gradient)
        Given the SPR-recovered nodal gradient tensor from
        gradient_recovery.spr_recover_nodal_gradients(), compute the
        gradient tensor at every parent-tet edge midpoint. Returned as
        a (n_elements, 6, 3, 3) array (6 edges per tet).

Both functions are pure NumPy, run on CPU, and are vectorised over
elements. Wall-clock estimate: ~30 ms per 100k elements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# The 6 edges of a tet, as (local_vertex_a, local_vertex_b) pairs.
# Ordered to match a common convention: three edges meeting at v0
# first, then the three opposite-face edges. This ordering is what
# Phase 3's Bernstein-coefficient formulas will assume, so it's
# fixed here as a module-level constant.
TET_EDGES = np.array([
    [0, 1],   # edge 0: v0-v1
    [0, 2],   # edge 1: v0-v2
    [0, 3],   # edge 2: v0-v3
    [1, 2],   # edge 3: v1-v2
    [1, 3],   # edge 4: v1-v3
    [2, 3],   # edge 5: v2-v3
], dtype=np.int32)

# The 4 faces of a tet, as triples of local vertex indices. Each face
# is opposite the vertex NOT in the triple. Ordering: face i is opposite
# vertex i. Vertices within each face listed in an order that gives a
# consistent outward-facing normal for a right-handed tet
# (positive-volume orientation).
TET_FACES = np.array([
    [1, 2, 3],   # face 0 — opposite v0
    [0, 3, 2],   # face 1 — opposite v1
    [0, 1, 3],   # face 2 — opposite v2
    [0, 2, 1],   # face 3 — opposite v3
], dtype=np.int32)

# Alfeld sub-tet layout. Each parent tet is split at its centroid vc
# into 4 sub-tets. Sub-tet i has vertex vc replacing parent vertex i.
# Sub-tet i's 4 local vertices in the sub-tet coordinate system:
#
#     sub-tet 0:  (v1, v2, v3, vc)      opposite parent face 0
#     sub-tet 1:  (v0, v2, v3, vc)      opposite parent face 1
#     sub-tet 2:  (v0, v1, v3, vc)      opposite parent face 2
#     sub-tet 3:  (v0, v1, v2, vc)      opposite parent face 3
#
# The parent-face triangle sits at the "outer" boundary of each sub-tet;
# vc is the apex. This is the convention Phase 3 assumes when writing
# the C1 continuity constraints at the shared vc-anchored interior
# faces.
#
# ALFELD_SUBTET_PARENT_VERTS[i] gives the three PARENT-vertex indices
# that appear in sub-tet i, in an order that yields POSITIVE volume
# for the sub-tet (vc, p_j, p_k, p_l) when the parent tet is right-
# handed. This is exactly the face-triangle ordering already stored in
# TET_FACES (outward-oriented triangles), because for a right-handed
# parent the outward-oriented face seen from the centroid gives the
# correct sub-tet vertex order. Verified by test_alfeld_subtet_volumes_*
# in tests/recovery/test_hct3d_geometry.py.
ALFELD_SUBTET_PARENT_VERTS = TET_FACES.copy()


@dataclass(frozen=True)
class AlfeldGeometry:
    """Precomputed Alfeld-split geometry for every element in the mesh.

    Attributes
    ----------
    n_elements : int
    centroids : (n_elements, 3) float64
        Parent-tet centroid vc = mean of the 4 vertices.
    edge_midpoints : (n_elements, 6, 3) float64
        Midpoint of each of the 6 parent edges, ordered per TET_EDGES.
    face_centroids : (n_elements, 4, 3) float64
        Centroid of each parent face, ordered per TET_FACES.
    subtet_volumes : (n_elements, 4) float64
        Signed volume of each of the 4 sub-tets. Positive volumes
        indicate right-handed sub-tets; used at build time to warn
        about degenerate parent elements and by Phase 4's sub-tet
        detection kernel.
    parent_volumes : (n_elements,) float64
        Signed volume of the parent tet. Equal to sum of the 4
        sub-tet volumes.

    Notes
    -----
    Sub-tet vertex indices are NOT stored here — they follow directly
    from ``ALFELD_SUBTET_PARENT_VERTS`` plus the parent centroid, so
    Phase 3 and Phase 4 can reconstruct them on the fly. Storing them
    would triple the memory footprint for no benefit.
    """
    n_elements: int
    centroids: np.ndarray
    edge_midpoints: np.ndarray
    face_centroids: np.ndarray
    subtet_volumes: np.ndarray
    parent_volumes: np.ndarray


def _tet_volume(vertices: np.ndarray) -> np.ndarray:
    """Signed volume of a tetrahedron.

    Args
    ----
    vertices : (..., 4, 3) float
        Trailing dims are the 4 vertices and 3 spatial coordinates.
        Any number of leading batch dims.

    Returns
    -------
    (...,) float
        (1/6) * det([v1-v0 | v2-v0 | v3-v0])
    """
    v0 = vertices[..., 0, :]
    e1 = vertices[..., 1, :] - v0
    e2 = vertices[..., 2, :] - v0
    e3 = vertices[..., 3, :] - v0
    # 3x3 determinant via triple product
    return np.einsum(
        "...i,...i->...",
        e1,
        np.cross(e2, e3),
    ) / 6.0


def build_alfeld_geometry(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    verbose: bool = False,
) -> AlfeldGeometry:
    """Precompute per-element Alfeld-split geometry.

    Args
    ----
    node_positions : (n_nodes, 3) float
    connectivity   : (n_elements, 4) int32
        Standard tet connectivity; each row lists the 4 parent-tet
        node indices.

    Returns
    -------
    AlfeldGeometry
    """
    import time
    t0 = time.time()
    node_positions = node_positions.astype(np.float64)
    connectivity = np.asarray(connectivity, dtype=np.int64)
    n_elements = connectivity.shape[0]

    # Gather the 4 vertex positions per element in one shot.
    p = node_positions[connectivity]                    # (n_elements, 4, 3)

    # --- Centroids -----------------------------------------------------
    centroids = p.mean(axis=1)                          # (n_elements, 3)

    # --- Edge midpoints ------------------------------------------------
    # p[:, TET_EDGES[:, 0]] is shape (n_elements, 6, 3), similarly for the
    # second vertex of each edge. Their mean is the midpoint.
    edge_midpoints = 0.5 * (
        p[:, TET_EDGES[:, 0]] + p[:, TET_EDGES[:, 1]]
    )                                                   # (n_elements, 6, 3)

    # --- Face centroids ------------------------------------------------
    face_centroids = p[:, TET_FACES].mean(axis=2)       # (n_elements, 4, 3)

    # --- Sub-tet volumes -----------------------------------------------
    # Build the 4 sub-tet vertex sets for every element in a vectorised
    # way. Sub-tet i's vertices are (vc, p_j, p_k, p_l) for
    # (j, k, l) = ALFELD_SUBTET_PARENT_VERTS[i].
    # Assemble one array of shape (n_elements, 4_subtets, 4_verts, 3).
    subtet_parent_verts = p[:, ALFELD_SUBTET_PARENT_VERTS]
    # shape: (n_elements, 4, 3, 3)  --  batch, sub-tet, three parent
    # vertices per sub-tet, coordinate.
    # Prepend the centroid as the 4th vertex of every sub-tet.
    centroids_broadcast = np.broadcast_to(
        centroids[:, None, None, :], (n_elements, 4, 1, 3),
    )
    subtet_verts = np.concatenate(
        [centroids_broadcast, subtet_parent_verts], axis=2,
    )                                                   # (n_elements, 4, 4, 3)
    subtet_volumes = _tet_volume(subtet_verts)          # (n_elements, 4)

    # --- Parent volumes ------------------------------------------------
    parent_volumes = _tet_volume(p)                     # (n_elements,)

    if verbose:
        # Sanity: sum of sub-tet volumes should match parent volume.
        vol_mismatch = np.abs(subtet_volumes.sum(axis=1) - parent_volumes).max()
        rel_mismatch = vol_mismatch / max(np.abs(parent_volumes).max(), 1e-30)
        print(f"[hct3d/geom] n_elements={n_elements:,}")
        print(f"  parent volume:  min={parent_volumes.min():.3e}, "
              f"max={parent_volumes.max():.3e}, "
              f"median={np.median(parent_volumes):.3e}")
        print(f"  sub-tet volume: min={subtet_volumes.min():.3e}, "
              f"max={subtet_volumes.max():.3e}")
        print(f"  vol sanity: max |sum(sub-vol) - parent-vol| = "
              f"{vol_mismatch:.3e} (relative: {rel_mismatch:.3e})")
        n_neg = (subtet_volumes <= 0).sum()
        if n_neg:
            print(f"  WARNING: {n_neg:,} sub-tet volumes are <= 0 "
                  f"(parent-tet vertex order may be left-handed for "
                  f"{n_neg // 4:,}+ elements)")
        print(f"  wall: {time.time() - t0:.1f}s")

    return AlfeldGeometry(
        n_elements=n_elements,
        centroids=centroids,
        edge_midpoints=edge_midpoints,
        face_centroids=face_centroids,
        subtet_volumes=subtet_volumes,
        parent_volumes=parent_volumes,
    )


def edge_midpoint_gradients(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    nodal_gradient: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """Gradient tensor at every parent-tet edge midpoint.

    For each edge (a, b) of each parent tet, the recovered gradient at
    the edge midpoint is the average of the SPR-recovered gradients at
    the two endpoint nodes.

    Args
    ----
    node_positions : (n_nodes, 3) float
        Kept for signature symmetry with the other precomputes; not
        used by the averaging formula itself.
    connectivity   : (n_elements, 4) int32
    nodal_gradient : (n_nodes, 3, 3) float
        Output of ``gradient_recovery.spr_recover_nodal_gradients``.

    Returns
    -------
    (n_elements, 6, 3, 3) float
        Gradient tensor at each of the 6 edge midpoints of every
        parent tet, ordered per ``TET_EDGES``.

    Notes
    -----
    Averaging the two endpoint gradients is the correct choice IFF the
    recovered nodal gradient field is treated as a piecewise-linear C0
    field on the parent tet (which is what SPR + P1 shape functions
    give us). Under that interpretation the gradient at any point
    inside the tet is the P1 blend of the 4 nodal gradients, and at
    the midpoint of edge (a, b) all other shape functions are 0.
    """
    del node_positions  # not used by the averaging formula
    import time
    t0 = time.time()

    connectivity = np.asarray(connectivity, dtype=np.int64)
    nodal_gradient = np.asarray(nodal_gradient, dtype=np.float64)
    n_elements = connectivity.shape[0]

    # Gather the 4 nodal gradient tensors per element.
    per_elem_grads = nodal_gradient[connectivity]       # (n_elements, 4, 3, 3)

    # For each of the 6 edges, average the two endpoint gradients.
    # TET_EDGES has shape (6, 2), indexing into the per-element axis
    # of size 4. Advanced-index into per_elem_grads:
    grads_a = per_elem_grads[:, TET_EDGES[:, 0]]        # (n_elements, 6, 3, 3)
    grads_b = per_elem_grads[:, TET_EDGES[:, 1]]        # (n_elements, 6, 3, 3)
    edge_grads = 0.5 * (grads_a + grads_b)              # (n_elements, 6, 3, 3)

    if verbose:
        norms = np.linalg.norm(
            edge_grads.reshape(n_elements * 6, 9), axis=1,
        )
        print(f"[hct3d/edge-grad] n_elements={n_elements:,}, "
              f"n_edge_samples={norms.size:,}")
        print(f"  |grad_u*|_F at edges: min={norms.min():.3e}, "
              f"mean={norms.mean():.3e}, max={norms.max():.3e}")
        print(f"  wall: {time.time() - t0:.1f}s")

    return edge_grads.astype(np.float32)
