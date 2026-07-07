"""
HCT-3D reconstruction — Phases 1, 2, 3a, 3a-plus, 3b (Taylor-fit variant).

Roadmap: docs/hct3d_implementation_plan.md.

This module implements:

* Phase 1 — Alfeld-split geometry precompute
* Phase 2 — per-edge gradient DOF computation
* Phase 3a — Bernstein B-coefficient assembly with C⁰ interior
             continuity (Commit 2a of the phased plan)

Nothing in this module is wired into run_tracking.py yet. Phase 3b
(C¹ upgrade via 4×4 macro-element linear solve) and Phase 4+5 (JAX
kernel evaluator + validation gates) will land in subsequent commits.
Until then, ``jaxtrace.gpu.recovery.gradient_recovery`` continues to
provide ``centroid_taylor`` and ``vertex_taylor`` as the only
user-selectable reconstruction methods.

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

    build_hct_bernstein_c0(node_positions, connectivity, node_velocities,
                           nodal_gradient, edge_grads, geom)
        Phase 3a. Assemble the 20 Bernstein B-coefficients per sub-tet
        per component, with C⁰ (not C¹) interior continuity. Exact for
        linear velocity fields.

    build_hct_bernstein_quad_faces(...)
        Phase 3a-plus. Strict refinement of build_hct_bernstein_c0
        that upgrades the face-centroid B-coefficients to
        quadratic-precision via Farin's formula. Adds quadratic
        exactness at parent face centroids while preserving all
        Phase 3a properties.

    build_hct_bernstein_c1_taylor(...)
        Phase 3b. Fits a local degree-2 Taylor expansion around the
        parent centroid vc against the 16 constraints (4 vertex values +
        12 vertex gradient components) via a batched 16x10 least-squares
        solve. Uses the resulting μ = u(vc) and γ = ∇u(vc) in the same
        Bernstein assembly pipeline as Phase 3a-plus. Bit-exact for
        quadratic velocity fields at parent centroid, spoke-edge
        midpoints, parent face centroids, and parent vertices. Does not
        enforce full C¹ continuity across sub-tet interior faces
        (Worsey-Farin), but provides a substantial accuracy improvement
        over Phase 3a-plus with lightweight compute.

Phase 3a/3a-plus/3b coefficient arrays are all returned as
``HCTBernstein`` dataclasses with a (n_elements, 4, 20, 3) float32
coefficient array. The ``continuity`` string field distinguishes them
('c0', 'c0_quad_faces', 'c1_taylor').

Precompute phases 1 and 2 are pure NumPy (already fast — <100 ms on
100k tets). Phase 3 assembly and the Taylor-fit LS are JAX-JIT'd for
5-10x speedup on large meshes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# JAX is imported lazily inside build_hct_bernstein_c0() so that this
# module can still be imported (and its NumPy helpers used) in
# environments where JAX isn't installed. The Phase 1+2 precomputes
# stay pure NumPy — they're already fast and the JAX warmup would
# dominate.
try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:  # pragma: no cover
    _HAS_JAX = False


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


# =============================================================================
# Phase 3a — Bernstein B-coefficient assembly with C⁰ interior continuity
# =============================================================================
#
# Notation
# --------
# For a cubic Bézier on a tet with local vertices (w0, w1, w2, w3), we
# use multi-indices α = (α0, α1, α2, α3) with |α|=3 to enumerate the
# 20 B-coefficients c_α. The physical location of control point c_α is
#     x_α = (α0 w0 + α1 w1 + α2 w2 + α3 w3) / 3.
# In our sub-tets w0 = vc (parent centroid) and (w1, w2, w3) are three
# parent vertices in the order given by ALFELD_SUBTET_PARENT_VERTS.
#
# The 20 multi-indices are enumerated in a canonical order below,
# grouped as:
#   * 4 vertex indices   (positions AT the sub-tet vertices w_i)
#   * 12 edge indices    (2 per edge × 6 edges of the sub-tet)
#   * 4 face indices     (1 per face; barycentric = 1/3 on 3 verts,
#                         0 on the opposite vertex)
#
# The 20-index catalog is stored as ``BERN_INDICES`` (shape (20, 4)).

def _enumerate_bernstein_multi_indices() -> np.ndarray:
    """Return the (20, 4) array of multi-indices α with |α|=3 in the
    canonical order used throughout this module.

    Order: 4 vertex, then 12 edge (2 per edge in TET_EDGES order),
    then 4 face (opposite each vertex in order).
    """
    out = []
    # Vertex indices
    for i in range(4):
        α = [0, 0, 0, 0]
        α[i] = 3
        out.append(α)
    # Edge indices: for each (a, b) in TET_EDGES, add (2 e_a + e_b),
    # then (e_a + 2 e_b).
    for a, b in TET_EDGES:
        α = [0, 0, 0, 0]
        α[a] = 2
        α[b] = 1
        out.append(list(α))
        α = [0, 0, 0, 0]
        α[a] = 1
        α[b] = 2
        out.append(list(α))
    # Face indices: opposite vertex i has three 1s on the other 3 slots.
    for i in range(4):
        α = [1, 1, 1, 1]
        α[i] = 0
        out.append(list(α))
    arr = np.array(out, dtype=np.int32)
    assert arr.shape == (20, 4)
    assert (arr.sum(axis=1) == 3).all(), "multi-indices must sum to 3"
    return arr


# (20, 4) — the canonical multi-index catalog.
BERN_INDICES = _enumerate_bernstein_multi_indices()

# Slice-index groups into BERN_INDICES for lookup convenience.
BERN_VERTEX_SLICE = slice(0, 4)          # (4,) — c_{3 e_i} at w_i
BERN_EDGE_SLICE   = slice(4, 16)         # (12,) — 2 per edge × 6 edges
BERN_FACE_SLICE   = slice(16, 20)        # (4,) — c at face opposite vertex i


# =============================================================================
# Static index tables — computed once at module import, no runtime lookups.
# =============================================================================
#
# Phase 3a's coefficient assembly needs, for each (sub-tet, coefficient-role)
# pair, to know:
#   * which local sub-tet vertex slot(s) the coefficient corresponds to
#   * which slot in BERN_INDICES to write into
# The Python code that USED to do this via np.where inside a hot loop is
# unrolled here into small static tables. That way the runtime function
# is a straight tensor computation with no metadata lookups, which is what
# makes the JAX vmap version fast.

def _bern_index_of(alpha: tuple) -> int:
    """Return the row in BERN_INDICES that equals the given multi-index."""
    alpha = np.asarray(alpha, dtype=np.int32)
    where = np.where((BERN_INDICES == alpha).all(axis=1))[0]
    if where.size != 1:
        raise KeyError(f"multi-index {tuple(alpha)} not found in BERN_INDICES")
    return int(where[0])


# _VERTEX_COEFF_INDEX[i] = row in BERN_INDICES for c_{3 e_i}, i = 0..3.
_VERTEX_COEFF_INDEX = np.array(
    [_bern_index_of(tuple(np.eye(4, dtype=np.int32)[i] * 3)) for i in range(4)],
    dtype=np.int32,
)

# _EDGE_COEFF_INDEX[k, 0] = row for c_{2 e_a + e_b} (1/3 from a to b),
# _EDGE_COEFF_INDEX[k, 1] = row for c_{e_a + 2 e_b} (1/3 from b to a),
# where (a, b) = TET_EDGES[k].
def _edge_coeff_index_table() -> np.ndarray:
    out = np.zeros((6, 2), dtype=np.int32)
    for k, (a, b) in enumerate(TET_EDGES):
        alpha_ab = np.zeros(4, dtype=np.int32); alpha_ab[a] = 2; alpha_ab[b] = 1
        alpha_ba = np.zeros(4, dtype=np.int32); alpha_ba[a] = 1; alpha_ba[b] = 2
        out[k, 0] = _bern_index_of(tuple(alpha_ab))
        out[k, 1] = _bern_index_of(tuple(alpha_ba))
    return out


_EDGE_COEFF_INDEX = _edge_coeff_index_table()  # (6, 2)


# _FACE_COEFF_INDEX[i] = row for c on the face opposite vertex i.
_FACE_COEFF_INDEX = np.zeros(4, dtype=np.int32)
for _i in range(4):
    _alpha = np.array([1, 1, 1, 1], dtype=np.int32); _alpha[_i] = 0
    _FACE_COEFF_INDEX[_i] = _bern_index_of(tuple(_alpha))

# _FACE_VERTEX_INDEX[i] = the 3 sub-tet local vertex slots that lie on
# the face opposite vertex i. Used by the face-coefficient formula.
_FACE_VERTEX_INDEX = np.zeros((4, 3), dtype=np.int32)
for _i in range(4):
    _FACE_VERTEX_INDEX[_i] = np.array(
        [j for j in range(4) if j != _i], dtype=np.int32,
    )

# _FACE_EDGE_COEFF_INDEX[i_opp, k] = the k-th of 6 rows in BERN_INDICES
# corresponding to the 2-per-edge × 3-edges edge B-coefficients on the
# face opposite vertex i_opp. These are the α-tuples with α_{i_opp} = 0
# and non-zero counts summing to 3 across the 3 face vertices.
_FACE_EDGE_COEFF_INDEX = np.zeros((4, 6), dtype=np.int32)
for _i_opp in range(4):
    _face_verts = [j for j in range(4) if j != _i_opp]
    # 3 edges among the face verts, each with 2 B-coeff rows.
    _rows = []
    for _fa, _fb in [(_face_verts[0], _face_verts[1]),
                     (_face_verts[0], _face_verts[2]),
                     (_face_verts[1], _face_verts[2])]:
        _alpha_ab = np.zeros(4, dtype=np.int32)
        _alpha_ab[_fa] = 2; _alpha_ab[_fb] = 1
        _rows.append(_bern_index_of(tuple(_alpha_ab)))
        _alpha_ba = np.zeros(4, dtype=np.int32)
        _alpha_ba[_fa] = 1; _alpha_ba[_fb] = 2
        _rows.append(_bern_index_of(tuple(_alpha_ba)))
    _FACE_EDGE_COEFF_INDEX[_i_opp] = np.array(_rows, dtype=np.int32)


@dataclass(frozen=True)
class HCTBernstein:
    """Per-element Bernstein B-coefficients for the Alfeld-split
    cubic reconstruction.

    Attributes
    ----------
    coeffs : (n_elements, 4, 20, 3) float32
        coeffs[e, s, k, c] = B-coefficient with local index k
        (see ``BERN_INDICES``) of velocity component c on sub-tet s
        of element e.
    continuity : str
        The continuity level enforced across sub-tet interior faces.
        'c0' for the Phase 3a (Commit 2a) implementation;
        'c1' for the Phase 3b upgrade (not yet available).
    """
    coeffs: np.ndarray
    continuity: str


def _bezier_edge_coeff(u_P, grad_u_P, edge_vec):
    """1D cubic Hermite along-edge formula.

    For a Bézier edge from P to Q with tangent (Q - P), the B-coefficient
    at position P + (1/3)(Q - P) is
        c_1/3 = u(P) + (1/3) * grad(u(P)) · (Q - P)
    (Farin 2002, Prop 17.1). Returns the scalar c_1/3 (per component
    handled by the caller via broadcasting).
    """
    return u_P + (1.0 / 3.0) * (grad_u_P @ edge_vec)


def build_hct_bernstein_c0(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    node_velocities: np.ndarray,
    nodal_gradient: np.ndarray,
    edge_grads: np.ndarray,
    geom: "AlfeldGeometry",
    verbose: bool = False,
) -> HCTBernstein:
    """Assemble 20 Bernstein B-coefficients per sub-tet per component,
    with C⁰ interior continuity between sub-tets.

    This is Phase 3a of the HCT-3D construction. It fixes the parent-
    centroid value μ and gradient γ to the P1-average of the 4 parent
    vertex values and gradients respectively, then closes each sub-tet's
    20-coefficient cubic Bézier from those DOFs.

    Compared to the full Worsey-Farin C¹ construction (Phase 3b):

    * Vertex B-coefficients match parent vertex values exactly, and
      c_{3000} at vc is μ (shared across all 4 sub-tets, so C⁰ at vc).
    * Edge B-coefficients on parent edges use the cubic Hermite formula
      with vertex value and vertex gradient. Since parent edges are
      shared by two sub-tets — one where the edge is a "base-triangle"
      edge of the outer face and one where it's not — the two sub-tets
      compute IDENTICAL B-coefficients on the shared edge, giving C⁰
      continuity there.
    * Edge B-coefficients on spoke edges (from vc to a parent vertex)
      use the cubic Hermite formula with (μ, γ) at the vc end and
      (vertex value, vertex gradient) at the parent-vertex end. Two
      sub-tets share each spoke edge as an interior edge, and both
      compute the same (μ, γ) so continuity is preserved.
    * Face B-coefficients c_{111}^face use u(face_centroid) evaluated as
      the P1 blend of the 3 face-vertex values. Exact for linear fields.

    C¹ continuity across sub-tet interior faces is NOT enforced. That
    upgrade lands in Phase 3b (Commit 2b) via a 4×4 linear solve per
    parent tet per component.

    Args
    ----
    node_positions : (n_nodes, 3) float
    connectivity   : (n_elements, 4) int32
    node_velocities: (n_nodes, 3) float
        The nodal velocity field feeding the reconstruction.
    nodal_gradient : (n_nodes, 3, 3) float
        Recovered nodal gradient tensor from Phase 4 of
        gradient_recovery.py.
    edge_grads     : (n_elements, 6, 3, 3) float
        Edge-midpoint gradient tensor from Phase 2
        (edge_midpoint_gradients). Currently unused by the C⁰ variant;
        kept in the signature so Phase 3b can drop in without a
        signature break.
    geom : AlfeldGeometry
        From Phase 1 (build_alfeld_geometry).
    verbose : bool
        Print summary statistics and wall-clock.

    Returns
    -------
    HCTBernstein
    """
    del edge_grads  # unused by the C0 variant; reserved for Phase 3b.
    import time
    t0 = time.time()

    if not _HAS_JAX:
        # Extremely unlikely fallback — JAXTrace depends on JAX everywhere
        # else, but importing this module without JAX shouldn't hard-fail
        # at import time. Users hitting this path can install JAX.
        raise RuntimeError(
            "build_hct_bernstein_c0 requires JAX (jax and jax.numpy). "
            "Install with `pip install jax jaxlib`."
        )

    # Move inputs onto whichever device jax.default_backend() selects.
    # For run_tracking.py this is the GPU; for standalone tests it's the
    # CPU. Either way the same code path runs — jax handles it.
    connectivity_j    = jnp.asarray(np.asarray(connectivity, dtype=np.int32))
    node_velocities_j = jnp.asarray(np.asarray(node_velocities, dtype=np.float32))
    nodal_gradient_j  = jnp.asarray(np.asarray(nodal_gradient, dtype=np.float32))
    centroids_j       = jnp.asarray(np.asarray(geom.centroids, dtype=np.float32))
    node_positions_j  = jnp.asarray(np.asarray(node_positions, dtype=np.float32))

    # Static index tables are baked into the JIT'd body as constants.
    coeffs = _bernstein_c0_jax(
        connectivity_j, node_velocities_j, nodal_gradient_j,
        centroids_j, node_positions_j,
    )
    # Force materialisation so wall-clock and downstream users are honest.
    coeffs.block_until_ready()

    n_elements = int(connectivity.shape[0])

    if verbose:
        elapsed = time.time() - t0
        print(f"[hct3d/bernstein-c0] n_elements={n_elements:,}, "
              f"n_subtets={4 * n_elements:,}")
        print(f"  coeffs shape: {tuple(coeffs.shape)}, dtype float32")
        n_coeff = 4 * 20
        mb = n_coeff * n_elements * 3 * 4 / 1e6
        print(f"  {n_coeff} coefficients per element "
              f"({n_coeff * n_elements:,} total floats × 3 components = "
              f"{mb:.1f} MB float32)")
        print(f"  wall: {elapsed:.1f}s "
              f"(JIT-compile + evaluate; second call would be ~{elapsed / 10:.2f}s)")
        c_min = float(coeffs.min())
        c_max = float(coeffs.max())
        print(f"  coeff range: [{c_min:.3e}, {c_max:.3e}]")

    # Copy back to host to match the existing HCTBernstein contract
    # (NumPy array), so downstream diagnostic code doesn't need to
    # know about JAX arrays. The Phase-4 kernel will re-upload with
    # jax.device_put; that's a no-op on CUDA when the source array
    # already lives on the same device.
    coeffs_np = np.asarray(coeffs)

    return HCTBernstein(
        coeffs=coeffs_np.astype(np.float32),
        continuity="c0",
    )


# =============================================================================
# JAX-vectorised implementation of Phase 3a
# =============================================================================
#
# All the static index tables are captured by closure into the JIT'd
# function body. XLA sees them as constants and folds them into the
# compiled graph, so there are no runtime gathers on 20-element tables.
#
# The n_elements axis is the leading batch dim throughout; there are NO
# Python for-loops over sub-tets, edges, or faces — each of those axes
# is either broadcast or handled by a single tensor op.

def _bernstein_c0_jax_body(
    connectivity: "jnp.ndarray",       # (n_elements, 4) int32
    node_velocities: "jnp.ndarray",    # (n_nodes, 3) float32
    nodal_gradient: "jnp.ndarray",     # (n_nodes, 3, 3) float32
    centroids: "jnp.ndarray",          # (n_elements, 3) float32
    node_positions: "jnp.ndarray",     # (n_nodes, 3) float32
) -> "jnp.ndarray":
    """Return (n_elements, 4_subtets, 20_coeffs, 3_comps) float32."""

    # Static tables as jnp constants — folded into the graph by XLA.
    subtet_pverts     = jnp.asarray(ALFELD_SUBTET_PARENT_VERTS, dtype=jnp.int32)  # (4, 3)
    tet_edges_arr     = jnp.asarray(TET_EDGES, dtype=jnp.int32)                   # (6, 2)
    vertex_coeff_idx  = jnp.asarray(_VERTEX_COEFF_INDEX, dtype=jnp.int32)         # (4,)
    edge_coeff_idx    = jnp.asarray(_EDGE_COEFF_INDEX, dtype=jnp.int32)           # (6, 2)
    face_coeff_idx    = jnp.asarray(_FACE_COEFF_INDEX, dtype=jnp.int32)           # (4,)
    face_vertex_idx   = jnp.asarray(_FACE_VERTEX_INDEX, dtype=jnp.int32)          # (4, 3)

    n_elements = connectivity.shape[0]

    # Gather per-element parent-vertex data.
    p_verts = node_positions[connectivity]        # (n_elements, 4, 3)
    v_verts = node_velocities[connectivity]       # (n_elements, 4, 3)
    G_verts = nodal_gradient[connectivity]        # (n_elements, 4, 3, 3)

    # C0 rule: μ and γ are P1 averages of parent-vertex data.
    mu    = v_verts.mean(axis=1)                  # (n_elements, 3)
    gamma = G_verts.mean(axis=1)                  # (n_elements, 3, 3)

    # -----------------------------------------------------------------
    # Build per-sub-tet local vertex data.
    # Shape target: (n_elements, 4_subtets, 4_local_verts, 3)  for values
    #               (n_elements, 4_subtets, 4_local_verts, 3, 3) for grads
    #               (n_elements, 4_subtets, 4_local_verts, 3) for positions.
    # Local vertex 0 is vc for every sub-tet.
    # Local vertices 1, 2, 3 are the 3 parent vertices given by
    # ALFELD_SUBTET_PARENT_VERTS[s].
    # -----------------------------------------------------------------
    # Gather the 3 parent vertices per sub-tet by taking axis-1 with
    # subtet_pverts. Result: (n_elements, 4_subtets, 3_verts, ...).
    v_parent_by_subtet = v_verts[:, subtet_pverts]        # (n_elements, 4, 3, 3)
    G_parent_by_subtet = G_verts[:, subtet_pverts]        # (n_elements, 4, 3, 3, 3)
    p_parent_by_subtet = p_verts[:, subtet_pverts]        # (n_elements, 4, 3, 3)

    # Broadcast μ/γ/vc to the sub-tet axis to sit at local vertex 0.
    mu_broadcast    = jnp.broadcast_to(
        mu[:, None, None, :], (n_elements, 4, 1, 3),
    )
    gamma_broadcast = jnp.broadcast_to(
        gamma[:, None, None, :, :], (n_elements, 4, 1, 3, 3),
    )
    vc_broadcast    = jnp.broadcast_to(
        centroids[:, None, None, :], (n_elements, 4, 1, 3),
    )

    # Concatenate vc/parent vertices along the local-vertex axis.
    u_local = jnp.concatenate([mu_broadcast, v_parent_by_subtet], axis=2)
    G_local = jnp.concatenate([gamma_broadcast, G_parent_by_subtet], axis=2)
    w_local = jnp.concatenate([vc_broadcast, p_parent_by_subtet], axis=2)
    # Shapes:
    #   u_local: (n_elements, 4_subtets, 4_local_verts, 3_components)
    #   G_local: (n_elements, 4_subtets, 4_local_verts, 3_out, 3_in)
    #   w_local: (n_elements, 4_subtets, 4_local_verts, 3)

    # Output accumulator.
    coeffs = jnp.zeros((n_elements, 4, 20, 3), dtype=jnp.float32)

    # ---- Vertex B-coefficients -----------------------------------------
    # For each local vertex slot i, write u_local[:, :, i, :] into slot
    # vertex_coeff_idx[i]. Use jnp.dynamic_update_slice via .at[].set().
    for i in range(4):
        coeffs = coeffs.at[:, :, int(_VERTEX_COEFF_INDEX[i]), :].set(
            u_local[:, :, i, :]
        )

    # ---- Edge B-coefficients -------------------------------------------
    # For each of the 6 sub-tet edges (a, b), compute the two 1/3-along
    # coefficients using the cubic Hermite formula:
    #   c_{ab -> 1/3} = u(w_a) + (1/3) G(w_a) · (w_b - w_a)
    #   c_{ba -> 1/3} = u(w_b) + (1/3) G(w_b) · (w_a - w_b)
    # Vectorised over (n_elements, 4_subtets) simultaneously.
    for k, (a, b) in enumerate(TET_EDGES):
        u_a = u_local[:, :, int(a), :]     # (n_elements, 4, 3)
        u_b = u_local[:, :, int(b), :]
        G_a = G_local[:, :, int(a), :, :]  # (n_elements, 4, 3, 3)
        G_b = G_local[:, :, int(b), :, :]
        edge_vec = w_local[:, :, int(b), :] - w_local[:, :, int(a), :]
        # jnp.einsum "esij,esj->esi": reduce spatial index j.
        grad_at_a = jnp.einsum("esij,esj->esi", G_a, edge_vec)
        grad_at_b = jnp.einsum("esij,esj->esi", G_b, -edge_vec)
        coeff_ab = u_a + (1.0 / 3.0) * grad_at_a
        coeff_ba = u_b + (1.0 / 3.0) * grad_at_b
        coeffs = coeffs.at[:, :, int(_EDGE_COEFF_INDEX[k, 0]), :].set(coeff_ab)
        coeffs = coeffs.at[:, :, int(_EDGE_COEFF_INDEX[k, 1]), :].set(coeff_ba)

    # ---- Face B-coefficients -------------------------------------------
    # Face opposite local vertex i has coefficient = P1 blend of the 3
    # non-i vertex values. Exact for linear fields; Phase 3b will
    # replace this with a quadratic-precision formula.
    for i_opp in range(4):
        js = _FACE_VERTEX_INDEX[i_opp]     # (3,) local vertex slots on the face
        face_centroid_val = (
            u_local[:, :, int(js[0]), :]
            + u_local[:, :, int(js[1]), :]
            + u_local[:, :, int(js[2]), :]
        ) / 3.0
        coeffs = coeffs.at[:, :, int(_FACE_COEFF_INDEX[i_opp]), :].set(
            face_centroid_val
        )

    return coeffs


# JIT-compiled entrypoint. Cached at module scope so repeated calls with
# the same input dtypes reuse the compilation. First call incurs ~1-3 s
# compile time on GPU; subsequent calls with the same shapes are ~ms.
if _HAS_JAX:
    _bernstein_c0_jax = jax.jit(_bernstein_c0_jax_body)
else:  # pragma: no cover
    _bernstein_c0_jax = None


# =============================================================================
# Phase 3a-plus — quadratic-precision face-centroid coefficients
# =============================================================================
#
# For a cubic Bézier on a triangle, the face-centroid coefficient c_{111}
# that gives QUADRATIC exactness (rather than just linear exactness of
# the "c_111 = P1 average of vertex values" rule used in Phase 3a) is:
#
#     c_{111}^face  =  -1/6 * sum(3 vertex B-coeffs on face)
#                    + 1/4 * sum(6 edge B-coeffs on face)
#
# The formula was verified symbolically in the design work for this
# commit: fit the two coefficients (x_v, x_e) to satisfy c_111 exactness
# for the two independent quadratic monomials u = b1^2 and u = b1*b2 on
# a reference triangle; the unique solution is (-1/6, 1/4). The third
# independent quadratic monomial (b2^2, etc.) satisfies the same
# equation automatically (checked as a consistency test).
#
# This rule is a strict refinement over Phase 3a: Phase 3a's c_111 rule
# is exact for LINEAR fields but NOT for quadratics. This rule is exact
# for BOTH linear and quadratic. The vertex, edge and non-face
# coefficients are unchanged from Phase 3a, so linear exactness is
# preserved everywhere and quadratic exactness is added at the parent
# face centroids.
#
# What this does NOT do: it does not upgrade c_{3000} at vc (= μ, still
# the P1 average) or the spoke-edge coefficients to quadratic exactness.
# Doing that requires solving for μ and γ = ∇u(vc), which is the full
# Phase 3b macro-element C¹ solve. See docs/hct3d_implementation_plan.md
# for the full construction.


def _apply_quadratic_face_upgrade_body(
    coeffs: "jnp.ndarray",   # (n_elements, 4, 20, 3) — from Phase 3a
) -> "jnp.ndarray":
    """Overwrite the 4 face-centroid B-coefficients in each sub-tet
    with the quadratic-precision Farin formula.

    Reads only the vertex and edge B-coefficients on each face (via
    the precomputed static index tables) and writes the 4 face
    coefficients.
    """
    face_coeff_idx    = jnp.asarray(_FACE_COEFF_INDEX, dtype=jnp.int32)      # (4,)
    face_vertex_idx   = jnp.asarray(_FACE_VERTEX_INDEX, dtype=jnp.int32)     # (4, 3)
    face_edge_idx     = jnp.asarray(_FACE_EDGE_COEFF_INDEX, dtype=jnp.int32) # (4, 6)
    vertex_coeff_idx  = jnp.asarray(_VERTEX_COEFF_INDEX, dtype=jnp.int32)    # (4,)

    for i_opp in range(4):
        # 3 vertex coefficients on this face (the 3 non-i_opp vertices).
        # face_vertex_idx[i_opp] gives local vertex slots {0,1,2,3}\{i_opp}.
        # Vertex-coeff row for local vertex j is vertex_coeff_idx[j].
        js = _FACE_VERTEX_INDEX[i_opp]  # length 3
        sum_vertex = (
            coeffs[:, :, int(_VERTEX_COEFF_INDEX[int(js[0])]), :]
            + coeffs[:, :, int(_VERTEX_COEFF_INDEX[int(js[1])]), :]
            + coeffs[:, :, int(_VERTEX_COEFF_INDEX[int(js[2])]), :]
        )
        # 6 edge coefficients on this face.
        edge_rows = _FACE_EDGE_COEFF_INDEX[i_opp]  # length 6
        sum_edge = jnp.zeros_like(sum_vertex)
        for k in range(6):
            sum_edge = sum_edge + coeffs[:, :, int(edge_rows[k]), :]
        # Farin quadratic-precision face-centroid.
        c_face = (-1.0 / 6.0) * sum_vertex + (1.0 / 4.0) * sum_edge
        coeffs = coeffs.at[:, :, int(_FACE_COEFF_INDEX[i_opp]), :].set(c_face)
    return coeffs


if _HAS_JAX:
    _apply_quadratic_face_upgrade = jax.jit(_apply_quadratic_face_upgrade_body)
else:  # pragma: no cover
    _apply_quadratic_face_upgrade = None


def build_hct_bernstein_quad_faces(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    node_velocities: np.ndarray,
    nodal_gradient: np.ndarray,
    edge_grads: np.ndarray,
    geom: "AlfeldGeometry",
    verbose: bool = False,
) -> HCTBernstein:
    """Phase 3a-plus. Same as build_hct_bernstein_c0 but upgrades the
    face-centroid B-coefficients from linear-precision (P1 vertex
    average) to quadratic-precision (Farin's formula involving vertex
    AND edge coefficients on the face).

    All other B-coefficients (vertex, edge, and the P1-averaged μ, γ at
    the parent centroid) are IDENTICAL to Phase 3a. So this function is
    a strict refinement: linear-field exactness is preserved, and
    quadratic-field exactness is added at the parent face centroids.

    Full C¹ continuity across sub-tet interior faces and quadratic/
    cubic exactness at the parent centroid still require the Phase 3b
    macro-element solve (upcoming).

    Args and return type match build_hct_bernstein_c0.
    """
    import time
    t0 = time.time()

    if not _HAS_JAX:
        raise RuntimeError(
            "build_hct_bernstein_quad_faces requires JAX."
        )

    # Compute Phase 3a coefficients first.
    hct_c0 = build_hct_bernstein_c0(
        node_positions, connectivity, node_velocities,
        nodal_gradient, edge_grads, geom, verbose=False,
    )
    # Upgrade face-centroid coefficients in-place on GPU.
    coeffs_upgraded = _apply_quadratic_face_upgrade(
        jnp.asarray(hct_c0.coeffs)
    )
    coeffs_upgraded.block_until_ready()
    coeffs_np = np.asarray(coeffs_upgraded)

    n_elements = int(connectivity.shape[0])
    if verbose:
        elapsed = time.time() - t0
        print(f"[hct3d/bernstein-quad-faces] n_elements={n_elements:,}, "
              f"wall={elapsed:.2f}s (includes Phase 3a build)")

    return HCTBernstein(
        coeffs=coeffs_np.astype(np.float32),
        continuity="c0_quad_faces",
    )


# =============================================================================
# Utility: evaluate a Bernstein cubic at a point (helper for tests/kernel)
# =============================================================================

def bernstein_cubic_evaluate(coeffs_one_subtet: np.ndarray,
                             barycentric: np.ndarray) -> np.ndarray:
    """Evaluate a cubic Bernstein polynomial at a point given the 20
    B-coefficients and 4 barycentric coordinates.

    Args
    ----
    coeffs_one_subtet : (20, 3) float
        B-coefficients for one sub-tet, one row per BERN_INDICES entry,
        one column per velocity component.
    barycentric : (4,) float
        Barycentric coordinates in the sub-tet (must sum to 1).

    Returns
    -------
    (3,) float — the velocity vector evaluated at the point.

    Notes
    -----
    Pure NumPy, no JAX. Used by the unit-test suite and by any Python-
    side diagnostic that wants a reference implementation. The JAX-JIT'd
    kernel (Phase 4) uses a different implementation optimised for
    XLA-HLO compile size and speed.
    """
    from math import factorial
    coeffs_one_subtet = np.asarray(coeffs_one_subtet, dtype=np.float64)
    barycentric = np.asarray(barycentric, dtype=np.float64)
    assert coeffs_one_subtet.shape == (20, 3), coeffs_one_subtet.shape
    assert barycentric.shape == (4,), barycentric.shape
    # Bernstein basis value B^3_α(b) = (3! / α!) * b^α
    fact3 = 6.0
    val = np.zeros(3, dtype=np.float64)
    for k in range(20):
        α = BERN_INDICES[k]
        multi = fact3 / (factorial(α[0]) * factorial(α[1])
                         * factorial(α[2]) * factorial(α[3]))
        b_power = (barycentric[0] ** α[0]
                   * barycentric[1] ** α[1]
                   * barycentric[2] ** α[2]
                   * barycentric[3] ** α[3])
        val += coeffs_one_subtet[k] * (multi * b_power)
    return val


# =============================================================================
# Phase 3b — Taylor-fit (μ, γ) at parent centroid
# =============================================================================
#
# Phase 3a used the P1 average of parent-vertex values for μ = u(vc) and
# the P1 average of parent-vertex gradients for γ = ∇u(vc). Those choices
# preserve linear exactness but NOT quadratic exactness at vc.
#
# Phase 3b upgrades (μ, γ) by fitting a LOCAL DEGREE-2 TAYLOR EXPANSION
# around vc to the 4 parent-vertex values plus the 4 parent-vertex
# gradients:
#
#     u(x) ≈ μ + γ · (x - vc) + ½ (x - vc)^T H (x - vc)
#     ∇u(x) ≈ γ + H (x - vc)
#
# Constraints per component (16 total):
#   * u(p_i) matches the nodal velocity at each of the 4 parent vertices (4)
#   * ∇u(p_i) matches the nodal gradient at each of the 4 parent vertices (12)
#
# Unknowns per component (10 total):
#   * μ (1), γ (3), H_sym = (H_xx, H_yy, H_zz, H_xy, H_xz, H_yz) (6)
#
# The 16×10 system is over-determined; we solve by least-squares.
# Solving three components simultaneously with one design matrix and a
# (16, 3) RHS gives all three (μ, γ, H) triples in one factorization.
#
# We then extract μ and γ per component and pass them into the SAME
# Phase 3a-plus assembly pipeline. The rest of the Bernstein
# coefficients (parent-edge, spoke-edge, face-centroid) inherit the
# improved (μ, γ) automatically — the spoke-edge coefficients become
# quadratic-exact for a quadratic field because γ now equals the
# analytic ∇u at vc (up to the LS residual, which is 0 for polynomials
# of degree ≤ 2).
#
# Testable properties:
#   * Linear exactness (still holds since a linear field has H = 0
#     and the LS solution is exact)
#   * Quadratic exactness at parent VERTICES (still holds — Phase 3a
#     property)
#   * Quadratic exactness at parent CENTROID vc (NEW — μ = u(vc)
#     exactly when u is quadratic)
#   * Quadratic exactness AT PARENT FACE CENTROIDS (still from 3a-plus)
#   * Quadratic exactness on SPOKE EDGES (NEW — γ matches ∇u(vc)
#     exactly, so the cubic Hermite formula reproduces the quadratic
#     value at any point on a spoke edge)
#
# What this does NOT do:
#   * Full C¹ across sub-tet INTERIOR FACES: NOT enforced. The Alfeld-
#     split cubic macro-element needs stronger C¹ face conditions
#     (Worsey-Farin) which this Taylor-fit variant does not solve.
#   * Bit-exact cubic reconstruction: cubic monomials have non-zero
#     3rd derivatives that Taylor-fit can't capture with a degree-2
#     truncation.
#
# For the recirc_2026 case (smooth analytic field with strong
# curvature), the practical accuracy gain of Taylor-fit over Phase
# 3a-plus should be measurable and comes at very low precompute cost
# (one 16×10 LS per element per component, batched via jnp.vmap).


def _build_taylor_lstsq_body(
    connectivity: "jnp.ndarray",       # (n_elements, 4) int32
    node_velocities: "jnp.ndarray",    # (n_nodes, 3) float32
    nodal_gradient: "jnp.ndarray",     # (n_nodes, 3, 3) float32
    centroids: "jnp.ndarray",          # (n_elements, 3) float32
    node_positions: "jnp.ndarray",     # (n_nodes, 3) float32
) -> "tuple[jnp.ndarray, jnp.ndarray]":
    """Return (mu, gamma) per element per component.

    Shapes:
        mu    : (n_elements, 3)          — u(vc) per velocity component
        gamma : (n_elements, 3, 3)       — gamma[e, i, j] = ∂u_i/∂x_j at vc
    """
    n_elements = connectivity.shape[0]

    # Gather parent-vertex data.
    p_verts = node_positions[connectivity]        # (n_elements, 4, 3)
    v_verts = node_velocities[connectivity]       # (n_elements, 4, 3)
    G_verts = nodal_gradient[connectivity]        # (n_elements, 4, 3, 3)
    vc = centroids                                 # (n_elements, 3)

    # Length scale for normalisation: mean sub-tet edge length. Using
    # the average of |p_i - vc| over the 4 parent vertices. This makes
    # the LS design matrix's value rows and gradient rows comparable in
    # magnitude regardless of mesh resolution.
    d_from_vc = p_verts - vc[:, None, :]           # (n_elements, 4, 3)
    d_norms = jnp.linalg.norm(d_from_vc, axis=-1)  # (n_elements, 4)
    L = d_norms.mean(axis=1, keepdims=True)        # (n_elements, 1)
    # Guard against zero (degenerate elements — should not happen but
    # be safe).
    L_safe = jnp.where(L > 1e-30, L, 1.0)          # (n_elements, 1)
    Ls = L_safe[:, 0]                              # (n_elements,)

    # Non-dimensional vertex offsets in the local frame.
    d_hat = d_from_vc / L_safe[:, :, None]         # (n_elements, 4, 3)

    # Build the (16, 10) design matrix per element. The 10 unknowns are
    # ordered as:
    #     0 : μ
    #     1 : γ_x  (times 1/L_scale internally handled)
    #     2 : γ_y
    #     3 : γ_z
    #     4 : H_xx (times 1/(2 L_scale^2) internally)
    #     5 : H_yy
    #     6 : H_zz
    #     7 : H_xy
    #     8 : H_xz
    #     9 : H_yz
    # We keep the physical (μ, γ, H) as the unknowns (not
    # non-dimensional) and instead scale the design-matrix ROWS to
    # equalise value and gradient magnitudes.
    #
    # Value row for vertex i:
    #     u_i = μ + γ · d_i + ½ [H_xx d_ix² + H_yy d_iy² + H_zz d_iz²
    #                          + 2 H_xy d_ix d_iy + 2 H_xz d_ix d_iz
    #                          + 2 H_yz d_iy d_iz]
    # Divided by Ls (a scalar per element) to give unit-magnitude rows.
    #
    # Gradient row for vertex i, component j (i.e., ∂u_i/∂x_j):
    #     grad_ij = γ_j + [H row j] · d_i
    # Divided by 1 (already unit magnitude if velocities are ~1).
    #
    # Actually, we should think about relative scaling. If |u| ~ V and
    # |∇u| ~ V/L, then a "value" row is ~ V and "gradient" row is ~ V/L.
    # Scaling value rows by 1/L makes them V/L, matching gradient rows.

    # Build value rows and gradient rows separately, then stack.
    # Value rows: 4 vertices, 10 columns.
    dx = d_from_vc[..., 0]                          # (n_elements, 4)
    dy = d_from_vc[..., 1]
    dz = d_from_vc[..., 2]
    one = jnp.ones_like(dx)                         # (n_elements, 4)
    zero = jnp.zeros_like(dx)

    # Value row per vertex: [1, dx, dy, dz, ½ dx², ½ dy², ½ dz², dx dy, dx dz, dy dz]
    A_val = jnp.stack([
        one, dx, dy, dz,
        0.5 * dx * dx, 0.5 * dy * dy, 0.5 * dz * dz,
        dx * dy, dx * dz, dy * dz,
    ], axis=-1)                                     # (n_elements, 4, 10)
    # Scale value rows by 1/Ls per element (broadcast over the 4 vertex
    # and 10 column axes).
    A_val = A_val / Ls[:, None, None]

    # Gradient rows: 12 rows per element. For each vertex i and each
    # spatial component j ∈ {x, y, z}, the row expresses ∂u_i/∂x_j.
    # Since μ contributes 0 to any gradient, its column is 0.
    # γ_j contributes 1 to the j-th gradient row.
    # H_kl contributes ∂(½ (x-vc)^T H (x-vc)) / ∂x_j at x_i.
    # For H_kl entry with l=j, ∂/∂x_j = H_kj (with the symmetric-tensor
    # convention 2·H_kj·d for the mixed off-diagonal terms).
    # Explicitly:
    #   ∂u/∂x = γ_x + H_xx dx + H_xy dy + H_xz dz
    #   ∂u/∂y = γ_y + H_xy dx + H_yy dy + H_yz dz
    #   ∂u/∂z = γ_z + H_xz dx + H_yz dy + H_zz dz
    A_grad_x = jnp.stack([zero, one, zero, zero,
                          dx, zero, zero, dy, dz, zero], axis=-1)
    A_grad_y = jnp.stack([zero, zero, one, zero,
                          zero, dy, zero, dx, zero, dz], axis=-1)
    A_grad_z = jnp.stack([zero, zero, zero, one,
                          zero, zero, dz, zero, dx, dy], axis=-1)
    A_grad = jnp.stack([A_grad_x, A_grad_y, A_grad_z], axis=-2)
    # A_grad shape: (n_elements, 4_vertices, 3_grad_components, 10_unknowns)
    A_grad = A_grad.reshape(n_elements, 12, 10)

    # Full design matrix (n_elements, 16, 10). Gradient rows unscaled;
    # value rows already scaled by 1/Ls above.
    A = jnp.concatenate([A_val, A_grad], axis=1)    # (n_elements, 16, 10)

    # Right-hand side: (n_elements, 16, 3_velocity_components).
    # 4 value rows: v_verts, scaled by 1/Ls to match A_val scaling.
    b_val = v_verts / Ls[:, None, None]             # (n_elements, 4, 3)
    # 12 gradient rows: G_verts flattened over (vertex, grad_component).
    # G_verts[e, i, j, k] = (grad u_k) at vertex i, spatial component j.
    # For velocity component k, we want row (vertex i, grad_component j).
    # Reshape to (n_elements, 4, 3, 3) -> (n_elements, 12, 3).
    # Axis order: G_verts is (elem, vertex, out_comp, spatial). We want
    # each of the 12 rows = (vertex, spatial) for each velocity comp.
    # So transpose to (elem, vertex, spatial, out_comp) then reshape.
    b_grad = jnp.transpose(G_verts, (0, 1, 3, 2)).reshape(n_elements, 12, 3)
    b = jnp.concatenate([b_val, b_grad], axis=1)    # (n_elements, 16, 3)

    # Solve the batched least-squares system: for each element,
    # A_e @ X_e = b_e in the LS sense. Since A_e is small (16, 10),
    # use the normal equations: X = (A^T A)^{-1} A^T b.
    # Batched via einsum + jnp.linalg.solve.
    AT_A = jnp.einsum("eij,eik->ejk", A, A)         # (n_elements, 10, 10)
    AT_b = jnp.einsum("eij,eic->ejc", A, b)         # (n_elements, 10, 3)
    X = jnp.linalg.solve(AT_A, AT_b)                # (n_elements, 10, 3)

    # Extract μ, γ, H from the solution.
    mu = X[:, 0, :]                                  # (n_elements, 3)
    gamma_x = X[:, 1, :]                             # (n_elements, 3)
    gamma_y = X[:, 2, :]
    gamma_z = X[:, 3, :]
    # gamma[e, comp, spatial] = ∂u_comp/∂x_spatial at vc.
    gamma = jnp.stack([gamma_x, gamma_y, gamma_z], axis=-1)
    # gamma has shape (n_elements, 3_comp, 3_spatial). But the rest of
    # hct3d.py uses gradient tensor shape (n_elements, 3_out, 3_in) with
    # the convention grad[i, j] = ∂u_i/∂x_j. So (comp, spatial) is
    # already (i, j) — correct.

    return mu, gamma


if _HAS_JAX:
    _build_taylor_lstsq = jax.jit(_build_taylor_lstsq_body)
else:  # pragma: no cover
    _build_taylor_lstsq = None


def _bernstein_c1_jax_body(
    connectivity: "jnp.ndarray",
    node_velocities: "jnp.ndarray",
    nodal_gradient: "jnp.ndarray",
    centroids: "jnp.ndarray",
    node_positions: "jnp.ndarray",
    mu_center: "jnp.ndarray",              # (n_elements, 3)  Taylor-fit μ
    gamma_center: "jnp.ndarray",           # (n_elements, 3, 3) Taylor-fit γ
) -> "jnp.ndarray":
    """Same as _bernstein_c0_jax_body but uses supplied (μ, γ) at vc
    instead of the P1-average rule."""
    subtet_pverts     = jnp.asarray(ALFELD_SUBTET_PARENT_VERTS, dtype=jnp.int32)
    tet_edges_arr     = jnp.asarray(TET_EDGES, dtype=jnp.int32)
    vertex_coeff_idx  = jnp.asarray(_VERTEX_COEFF_INDEX, dtype=jnp.int32)
    edge_coeff_idx    = jnp.asarray(_EDGE_COEFF_INDEX, dtype=jnp.int32)
    face_coeff_idx    = jnp.asarray(_FACE_COEFF_INDEX, dtype=jnp.int32)
    face_vertex_idx   = jnp.asarray(_FACE_VERTEX_INDEX, dtype=jnp.int32)

    n_elements = connectivity.shape[0]

    p_verts = node_positions[connectivity]
    v_verts = node_velocities[connectivity]
    G_verts = nodal_gradient[connectivity]

    # Phase 3b: use supplied μ and γ (from Taylor-fit LS solve).
    mu = mu_center
    gamma = gamma_center

    v_parent_by_subtet = v_verts[:, subtet_pverts]
    G_parent_by_subtet = G_verts[:, subtet_pverts]
    p_parent_by_subtet = p_verts[:, subtet_pverts]

    mu_broadcast    = jnp.broadcast_to(
        mu[:, None, None, :], (n_elements, 4, 1, 3),
    )
    gamma_broadcast = jnp.broadcast_to(
        gamma[:, None, None, :, :], (n_elements, 4, 1, 3, 3),
    )
    vc_broadcast    = jnp.broadcast_to(
        centroids[:, None, None, :], (n_elements, 4, 1, 3),
    )

    u_local = jnp.concatenate([mu_broadcast, v_parent_by_subtet], axis=2)
    G_local = jnp.concatenate([gamma_broadcast, G_parent_by_subtet], axis=2)
    w_local = jnp.concatenate([vc_broadcast, p_parent_by_subtet], axis=2)

    coeffs = jnp.zeros((n_elements, 4, 20, 3), dtype=jnp.float32)

    for i in range(4):
        coeffs = coeffs.at[:, :, int(_VERTEX_COEFF_INDEX[i]), :].set(
            u_local[:, :, i, :]
        )

    for k, (a, b) in enumerate(TET_EDGES):
        u_a = u_local[:, :, int(a), :]
        u_b = u_local[:, :, int(b), :]
        G_a = G_local[:, :, int(a), :, :]
        G_b = G_local[:, :, int(b), :, :]
        edge_vec = w_local[:, :, int(b), :] - w_local[:, :, int(a), :]
        grad_at_a = jnp.einsum("esij,esj->esi", G_a, edge_vec)
        grad_at_b = jnp.einsum("esij,esj->esi", G_b, -edge_vec)
        coeff_ab = u_a + (1.0 / 3.0) * grad_at_a
        coeff_ba = u_b + (1.0 / 3.0) * grad_at_b
        coeffs = coeffs.at[:, :, int(_EDGE_COEFF_INDEX[k, 0]), :].set(coeff_ab)
        coeffs = coeffs.at[:, :, int(_EDGE_COEFF_INDEX[k, 1]), :].set(coeff_ba)

    # Face coefficients: use quadratic-precision formula (Phase 3a-plus).
    for i_opp in range(4):
        js = _FACE_VERTEX_INDEX[i_opp]
        # Vertex coefficients on this face.
        sum_vertex = (
            coeffs[:, :, int(_VERTEX_COEFF_INDEX[int(js[0])]), :]
            + coeffs[:, :, int(_VERTEX_COEFF_INDEX[int(js[1])]), :]
            + coeffs[:, :, int(_VERTEX_COEFF_INDEX[int(js[2])]), :]
        )
        # Edge coefficients on this face.
        edge_rows = _FACE_EDGE_COEFF_INDEX[i_opp]
        sum_edge = jnp.zeros_like(sum_vertex)
        for k in range(6):
            sum_edge = sum_edge + coeffs[:, :, int(edge_rows[k]), :]
        c_face = (-1.0 / 6.0) * sum_vertex + (1.0 / 4.0) * sum_edge
        coeffs = coeffs.at[:, :, int(_FACE_COEFF_INDEX[i_opp]), :].set(c_face)

    return coeffs


if _HAS_JAX:
    _bernstein_c1_jax = jax.jit(_bernstein_c1_jax_body)
else:  # pragma: no cover
    _bernstein_c1_jax = None


def build_hct_bernstein_c1_taylor(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    node_velocities: np.ndarray,
    nodal_gradient: np.ndarray,
    edge_grads: np.ndarray,
    geom: "AlfeldGeometry",
    verbose: bool = False,
) -> HCTBernstein:
    """Phase 3b (Taylor-fit variant). Uses locally-fit Taylor-expansion
    (μ, γ) at the parent centroid vc instead of the P1-average rule.

    All other coefficients follow the Phase 3a-plus recipe (parent-edge
    Hermite, spoke-edge Hermite with the improved γ, quadratic-precision
    face-centroid). The Bernstein cubic is bit-exact for quadratic
    velocity fields at the parent centroid, on spoke edges, at parent
    face centroids, and at all parent vertices.

    Args and return type match build_hct_bernstein_c0. The returned
    HCTBernstein has continuity marker 'c1_taylor'.
    """
    del edge_grads  # not consumed by the Taylor-fit variant either
    import time
    t0 = time.time()

    if not _HAS_JAX:
        raise RuntimeError(
            "build_hct_bernstein_c1_taylor requires JAX."
        )

    connectivity_j    = jnp.asarray(np.asarray(connectivity, dtype=np.int32))
    node_velocities_j = jnp.asarray(np.asarray(node_velocities, dtype=np.float32))
    nodal_gradient_j  = jnp.asarray(np.asarray(nodal_gradient, dtype=np.float32))
    centroids_j       = jnp.asarray(np.asarray(geom.centroids, dtype=np.float32))
    node_positions_j  = jnp.asarray(np.asarray(node_positions, dtype=np.float32))

    # Solve for (μ, γ) via batched Taylor-fit LS.
    mu_j, gamma_j = _build_taylor_lstsq(
        connectivity_j, node_velocities_j, nodal_gradient_j,
        centroids_j, node_positions_j,
    )
    mu_j.block_until_ready()

    # Assemble the Bernstein coefficients using (μ, γ) at vc.
    coeffs = _bernstein_c1_jax(
        connectivity_j, node_velocities_j, nodal_gradient_j,
        centroids_j, node_positions_j,
        mu_j, gamma_j,
    )
    coeffs.block_until_ready()

    n_elements = int(connectivity.shape[0])

    if verbose:
        elapsed = time.time() - t0
        print(f"[hct3d/bernstein-c1-taylor] n_elements={n_elements:,}, "
              f"wall={elapsed:.2f}s (LS + Bernstein assembly, JIT-compiled)")
        c_min = float(coeffs.min())
        c_max = float(coeffs.max())
        print(f"  coeff range: [{c_min:.3e}, {c_max:.3e}]")

    coeffs_np = np.asarray(coeffs)

    return HCTBernstein(
        coeffs=coeffs_np.astype(np.float32),
        continuity="c1_taylor",
    )
