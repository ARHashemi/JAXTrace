"""
Unit tests for HCT-3D Phase 1 (geometry precompute) and Phase 2
(edge-midpoint gradient DOFs).

These test the internal-only ``jaxtrace.gpu.recovery.hct3d`` module.
Nothing here exercises the JAX kernel — that's Phase 4's job.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is on sys.path when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from jaxtrace.gpu.recovery.hct3d import (
    build_alfeld_geometry,
    edge_midpoint_gradients,
    TET_EDGES,
    TET_FACES,
    ALFELD_SUBTET_PARENT_VERTS,
    _tet_volume,
)


# -----------------------------------------------------------------------
# Reference meshes
# -----------------------------------------------------------------------

def make_kuhn_cube_mesh(n_per_axis: int = 4):
    """Regular Kuhn tetrahedralised cube. Every hex sub-cell splits into
    6 right-handed tets. Used because it's the same mesh shape the
    recovery pipeline is tested against elsewhere."""
    n = n_per_axis + 1
    xs = np.linspace(0.0, 1.0, n)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    def nid(i, j, k):
        return i * n * n + j * n + k

    conn = []
    for i in range(n_per_axis):
        for j in range(n_per_axis):
            for k in range(n_per_axis):
                c = [
                    nid(i + di, j + dj, k + dk)
                    for di, dj, dk in [
                        (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                        (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
                    ]
                ]
                for t in [
                    (0, 1, 3, 7), (0, 1, 7, 5), (0, 2, 7, 3),
                    (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 7, 6),
                ]:
                    conn.append([c[t[0]], c[t[1]], c[t[2]], c[t[3]]])

    return nodes, np.array(conn, dtype=np.int32)


# -----------------------------------------------------------------------
# Static-constant sanity tests
# -----------------------------------------------------------------------

def test_tet_edges_are_unique_pairs():
    """The 6 edges must be the 6 distinct pairs of {0,1,2,3}."""
    edges = {tuple(sorted(e)) for e in TET_EDGES}
    assert edges == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    assert TET_EDGES.shape == (6, 2)


def test_tet_faces_are_the_four_triangles():
    """Each parent face is opposite exactly one vertex."""
    assert TET_FACES.shape == (4, 3)
    for i, face in enumerate(TET_FACES):
        opposite_vertex = list({0, 1, 2, 3} - set(face))
        assert len(opposite_vertex) == 1
        assert opposite_vertex[0] == i


def test_alfeld_subtet_parent_verts_covers_each_face():
    """Sub-tet i uses the three parent vertices of face i (as a set)."""
    assert ALFELD_SUBTET_PARENT_VERTS.shape == (4, 3)
    for i in range(4):
        assert set(ALFELD_SUBTET_PARENT_VERTS[i]) == set(TET_FACES[i])


# -----------------------------------------------------------------------
# _tet_volume tests
# -----------------------------------------------------------------------

def test_tet_volume_unit_reference():
    """Reference tet with vertices at (0,0,0), (1,0,0), (0,1,0),
    (0,0,1) has volume 1/6."""
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    v = _tet_volume(verts)
    assert np.isclose(v, 1.0 / 6.0)


def test_tet_volume_batched():
    """Signed volume vectorises correctly over a batch."""
    a = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    b = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
    ])
    batch = np.stack([a, b], axis=0)
    v = _tet_volume(batch)
    assert v.shape == (2,)
    assert np.isclose(v[0], 1.0 / 6.0)
    assert np.isclose(v[1], 8.0 / 6.0)


def test_tet_volume_sign_reversal():
    """Swapping two vertices flips the sign."""
    verts_pos = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    verts_neg = verts_pos.copy()
    verts_neg[[1, 2]] = verts_neg[[2, 1]]
    assert _tet_volume(verts_pos) > 0
    assert _tet_volume(verts_neg) < 0
    assert np.isclose(_tet_volume(verts_pos), -_tet_volume(verts_neg))


# -----------------------------------------------------------------------
# build_alfeld_geometry tests
# -----------------------------------------------------------------------

def test_alfeld_geometry_shapes():
    nodes, conn = make_kuhn_cube_mesh(3)
    geom = build_alfeld_geometry(nodes, conn)
    n_elem = conn.shape[0]
    assert geom.n_elements == n_elem
    assert geom.centroids.shape == (n_elem, 3)
    assert geom.edge_midpoints.shape == (n_elem, 6, 3)
    assert geom.face_centroids.shape == (n_elem, 4, 3)
    assert geom.subtet_volumes.shape == (n_elem, 4)
    assert geom.parent_volumes.shape == (n_elem,)


def test_alfeld_centroid_is_vertex_mean():
    nodes, conn = make_kuhn_cube_mesh(3)
    geom = build_alfeld_geometry(nodes, conn)
    # For every element the centroid must be the mean of its 4 vertices.
    for e in range(conn.shape[0]):
        expected = nodes[conn[e]].mean(axis=0)
        assert np.allclose(geom.centroids[e], expected, atol=1e-12)


def test_alfeld_edge_midpoints_are_pair_means():
    """Every edge midpoint == midpoint of its two endpoint node
    positions."""
    nodes, conn = make_kuhn_cube_mesh(2)
    geom = build_alfeld_geometry(nodes, conn)
    for e in range(conn.shape[0]):
        for k, (a, b) in enumerate(TET_EDGES):
            expected = 0.5 * (nodes[conn[e, a]] + nodes[conn[e, b]])
            assert np.allclose(geom.edge_midpoints[e, k], expected, atol=1e-12)


def test_alfeld_face_centroids_are_face_means():
    nodes, conn = make_kuhn_cube_mesh(2)
    geom = build_alfeld_geometry(nodes, conn)
    for e in range(conn.shape[0]):
        for f in range(4):
            face_nodes = conn[e, TET_FACES[f]]
            expected = nodes[face_nodes].mean(axis=0)
            assert np.allclose(geom.face_centroids[e, f], expected, atol=1e-12)


def test_alfeld_subtet_volumes_sum_to_parent():
    """Signed volumes of the 4 sub-tets must sum to the parent volume
    for every element."""
    nodes, conn = make_kuhn_cube_mesh(4)
    geom = build_alfeld_geometry(nodes, conn)
    sums = geom.subtet_volumes.sum(axis=1)
    # Kuhn tets are all right-handed so parent volumes are positive.
    assert np.all(geom.parent_volumes > 0)
    diff = np.abs(sums - geom.parent_volumes)
    # Machine noise only. Relative to parent volume ~1e-3 (small tets
    # inside a unit cube), 1e-12 absolute means ~1e-9 relative.
    assert diff.max() < 1e-12, f"max diff {diff.max()}"


def test_alfeld_subtet_volumes_all_positive_for_right_handed_parents():
    """When the parent tets are all right-handed (positive volume),
    the 4 Alfeld sub-tets must also all be right-handed."""
    nodes, conn = make_kuhn_cube_mesh(3)
    geom = build_alfeld_geometry(nodes, conn)
    assert np.all(geom.parent_volumes > 0)
    assert np.all(geom.subtet_volumes > 0), (
        f"n_negative sub-tets = {(geom.subtet_volumes <= 0).sum()}"
    )


# -----------------------------------------------------------------------
# edge_midpoint_gradients tests
# -----------------------------------------------------------------------

def test_edge_midpoint_gradient_constant_field():
    """If the recovered nodal gradient is CONSTANT across the mesh (as
    it would be for a linear velocity field), the edge-midpoint gradient
    must equal that constant everywhere."""
    nodes, conn = make_kuhn_cube_mesh(3)
    # Fake nodal gradient: same tensor at every node.
    G = np.array([
        [1.0, 2.0, 3.0],
        [-1.0, 0.5, 0.0],
        [0.5, 1.5, 2.0],
    ])
    nodal_grad = np.broadcast_to(G, (nodes.shape[0], 3, 3)).copy()

    edge_grads = edge_midpoint_gradients(nodes, conn, nodal_grad)
    assert edge_grads.shape == (conn.shape[0], 6, 3, 3)
    # Every edge-midpoint gradient must equal G to float32 precision.
    diff = np.abs(edge_grads - G[None, None]).max()
    assert diff < 1e-6, f"max diff {diff}"


def test_edge_midpoint_gradient_averaging_formula():
    """The edge-midpoint gradient must equal the arithmetic mean of the
    two endpoint nodal gradients."""
    nodes, conn = make_kuhn_cube_mesh(2)
    # Every-node-different nodal gradient: seeded by node index.
    n_nodes = nodes.shape[0]
    rng = np.random.default_rng(42)
    nodal_grad = rng.standard_normal((n_nodes, 3, 3))

    edge_grads = edge_midpoint_gradients(nodes, conn, nodal_grad)
    for e in range(conn.shape[0]):
        for k, (a, b) in enumerate(TET_EDGES):
            ga = nodal_grad[conn[e, a]]
            gb = nodal_grad[conn[e, b]]
            expected = 0.5 * (ga + gb)
            got = edge_grads[e, k]
            diff = np.abs(got - expected.astype(np.float32)).max()
            assert diff < 1e-6, (
                f"element {e} edge {k}: max diff {diff}"
            )


def test_edge_midpoint_gradient_dtype():
    """Output is float32 (matches vertex_taylor's node_gradient upload
    dtype in the kernel)."""
    nodes, conn = make_kuhn_cube_mesh(2)
    nodal_grad = np.ones((nodes.shape[0], 3, 3), dtype=np.float64)
    edge_grads = edge_midpoint_gradients(nodes, conn, nodal_grad)
    assert edge_grads.dtype == np.float32


# -----------------------------------------------------------------------
# Wall-clock sanity check
# -----------------------------------------------------------------------

def test_alfeld_geometry_wall_clock():
    """On a 10x10x10 Kuhn cube (6000 tets) the geometry precompute
    should complete in under 500 ms even in a busy CI."""
    import time
    nodes, conn = make_kuhn_cube_mesh(10)
    t0 = time.time()
    build_alfeld_geometry(nodes, conn)
    elapsed = time.time() - t0
    assert elapsed < 0.5, (
        f"geometry precompute took {elapsed:.2f}s on 6000 tets; "
        f"target < 0.5s"
    )


if __name__ == "__main__":
    # Allow running the file directly with `python test_hct3d_geometry.py`.
    import inspect
    fns = [
        (name, fn) for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    ]
    print(f"Running {len(fns)} tests...\n")
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # pragma: no cover - test infra only
            failed += 1
            print(f"  ERR   {name}: {type(exc).__name__}: {exc}")
    print()
    if failed:
        print(f"{failed} failure(s)")
        sys.exit(1)
    print("all tests passed")
