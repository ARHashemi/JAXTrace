"""
Unit tests for HCT-3D Phase 3a — the C⁰ Bernstein coefficient
assembly in ``jaxtrace.gpu.recovery.hct3d.build_hct_bernstein_c0``.

Phase 3a exactness contract:

* Constant velocity field: all 20 B-coefficients on every sub-tet
  equal the constant, at machine precision.
* Linear velocity field u(x) = A x + b: reconstruction is exact at
  every point in every sub-tet, at machine precision.

Phase 3a does NOT enforce quadratic/cubic exactness — that's the
job of Phase 3b (C¹ upgrade). The tests here don't check those.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from jaxtrace.gpu.recovery.hct3d import (
    build_alfeld_geometry,
    edge_midpoint_gradients,
    build_hct_bernstein_c0,
    bernstein_cubic_evaluate,
    BERN_INDICES,
    BERN_VERTEX_SLICE,
    BERN_EDGE_SLICE,
    BERN_FACE_SLICE,
    ALFELD_SUBTET_PARENT_VERTS,
    TET_EDGES,
)


# -----------------------------------------------------------------------
# Reference mesh — same as in test_hct3d_geometry
# -----------------------------------------------------------------------

def make_kuhn_cube_mesh(n_per_axis: int = 4):
    n = n_per_axis + 1
    xs = np.linspace(0.0, 1.0, n)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    def nid(i, j, k): return i * n * n + j * n + k
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
# BERN_INDICES catalog sanity
# -----------------------------------------------------------------------

def test_bern_indices_shape_and_sum():
    assert BERN_INDICES.shape == (20, 4)
    assert (BERN_INDICES.sum(axis=1) == 3).all()


def test_bern_indices_are_unique():
    seen = {tuple(row) for row in BERN_INDICES}
    assert len(seen) == 20, "multi-indices must all be distinct"


def test_bern_indices_group_sizes():
    """Vertex/edge/face groups should have 4/12/4 entries."""
    assert BERN_VERTEX_SLICE.stop - BERN_VERTEX_SLICE.start == 4
    assert BERN_EDGE_SLICE.stop - BERN_EDGE_SLICE.start == 12
    assert BERN_FACE_SLICE.stop - BERN_FACE_SLICE.start == 4


def test_bern_vertex_indices_have_one_3_and_three_0s():
    for k in range(BERN_VERTEX_SLICE.start, BERN_VERTEX_SLICE.stop):
        α = BERN_INDICES[k]
        assert list(sorted(α)) == [0, 0, 0, 3]


def test_bern_face_indices_have_one_0_and_three_1s():
    for k in range(BERN_FACE_SLICE.start, BERN_FACE_SLICE.stop):
        α = BERN_INDICES[k]
        assert list(sorted(α)) == [0, 1, 1, 1]


def test_bern_edge_indices_have_one_2_one_1_two_0s():
    for k in range(BERN_EDGE_SLICE.start, BERN_EDGE_SLICE.stop):
        α = BERN_INDICES[k]
        assert list(sorted(α)) == [0, 0, 1, 2]


# -----------------------------------------------------------------------
# bernstein_cubic_evaluate reference implementation
# -----------------------------------------------------------------------

def test_bernstein_evaluate_at_vertices_gives_vertex_coeffs():
    """B^3_α(b) evaluated at b = e_i (vertex i) equals 0 unless α = 3 e_i,
    in which case it equals 1. So value = c_{3 e_i}."""
    rng = np.random.default_rng(0)
    coeffs = rng.standard_normal((20, 3))
    for i in range(4):
        bary = np.zeros(4)
        bary[i] = 1.0
        val = bernstein_cubic_evaluate(coeffs, bary)
        # Find the vertex-i coefficient.
        α = np.zeros(4, dtype=np.int32); α[i] = 3
        k = int(np.where((BERN_INDICES == α).all(axis=1))[0][0])
        expected = coeffs[k]
        assert np.allclose(val, expected, atol=1e-12)


def test_bernstein_evaluate_constant_field():
    """If all 20 B-coeffs equal the same constant, evaluation should
    give that constant everywhere (Bernstein basis sums to 1)."""
    const = np.array([2.5, -1.0, 0.75])
    coeffs = np.broadcast_to(const, (20, 3)).copy()
    rng = np.random.default_rng(42)
    for _ in range(10):
        raw = np.abs(rng.standard_normal(4))
        bary = raw / raw.sum()
        val = bernstein_cubic_evaluate(coeffs, bary)
        assert np.allclose(val, const, atol=1e-12)


# -----------------------------------------------------------------------
# build_hct_bernstein_c0 — output shape and continuity marker
# -----------------------------------------------------------------------

def test_build_hct_bernstein_c0_shape():
    nodes, conn = make_kuhn_cube_mesh(2)
    geom = build_alfeld_geometry(nodes, conn)
    vel = np.zeros_like(nodes)
    grad = np.zeros((nodes.shape[0], 3, 3))
    egrads = edge_midpoint_gradients(nodes, conn, grad)
    hct = build_hct_bernstein_c0(nodes, conn, vel, grad, egrads, geom)
    assert hct.coeffs.shape == (conn.shape[0], 4, 20, 3)
    assert hct.coeffs.dtype == np.float32
    assert hct.continuity == "c0"


# -----------------------------------------------------------------------
# Exactness: constant field
# -----------------------------------------------------------------------

def test_build_hct_bernstein_c0_constant_field():
    """For a constant velocity field, every B-coefficient must equal
    the constant on every sub-tet of every element.

    Rationale: with u ≡ c the nodal velocities all equal c, the SPR-
    recovered nodal gradients all equal 0, so:
      - vertex B-coeffs = c
      - edge B-coeffs = c + (1/3) * 0 · edge_vec = c
      - face B-coeffs = (c+c+c)/3 = c
    """
    nodes, conn = make_kuhn_cube_mesh(3)
    geom = build_alfeld_geometry(nodes, conn)
    const_vel = np.array([2.5, -1.0, 0.75])
    vel = np.broadcast_to(const_vel, nodes.shape).copy()
    grad = np.zeros((nodes.shape[0], 3, 3))
    egrads = edge_midpoint_gradients(nodes, conn, grad)
    hct = build_hct_bernstein_c0(nodes, conn, vel, grad, egrads, geom)
    # Every B-coefficient should equal the constant.
    expected = np.broadcast_to(const_vel, hct.coeffs.shape).astype(np.float32)
    diff = np.abs(hct.coeffs - expected).max()
    assert diff < 1e-6, f"max diff = {diff}"


# -----------------------------------------------------------------------
# Exactness: linear field
# -----------------------------------------------------------------------

def test_build_hct_bernstein_c0_linear_field_exact_at_random_points():
    """For a linear velocity field u(x) = A x + b, the Bernstein cubic
    reconstruction (with the exact analytic gradient at every node)
    must recover u(x) exactly at every point in every sub-tet.

    We use the ANALYTIC nodal gradient (== A everywhere) rather than
    SPR-recovered, so this test isolates Phase 3a correctness from
    SPR boundary error."""
    nodes, conn = make_kuhn_cube_mesh(3)
    geom = build_alfeld_geometry(nodes, conn)

    # Random-ish linear field.
    A = np.array([
        [1.0, 2.0, 3.0],
        [-1.0, 0.5, 0.0],
        [0.5, 1.5, 2.0],
    ])
    b = np.array([0.1, -0.2, 0.3])
    vel = (A @ nodes.T).T + b
    # Analytic gradient: A everywhere.
    grad = np.broadcast_to(A, (nodes.shape[0], 3, 3)).copy()
    egrads = edge_midpoint_gradients(nodes, conn, grad)
    hct = build_hct_bernstein_c0(nodes, conn, vel, grad, egrads, geom)

    # Pick 200 random sample points, each inside a random sub-tet of a
    # random element. Verify reconstruction matches u(x) = A x + b.
    rng = np.random.default_rng(0)
    n_test = 200
    max_err = 0.0
    for _ in range(n_test):
        e = rng.integers(0, conn.shape[0])
        s = rng.integers(0, 4)
        # Sample barycentric coords (all > 0.05 to avoid face
        # boundaries where float32 dominates).
        raw = np.abs(rng.standard_normal(4)) + 0.1
        bary = raw / raw.sum()
        # Physical position in this sub-tet.
        vc_e = geom.centroids[e]
        fs = ALFELD_SUBTET_PARENT_VERTS[s]
        w0 = vc_e
        w1 = nodes[conn[e, fs[0]]]
        w2 = nodes[conn[e, fs[1]]]
        w3 = nodes[conn[e, fs[2]]]
        pos = bary[0] * w0 + bary[1] * w1 + bary[2] * w2 + bary[3] * w3
        # Expected velocity.
        v_exact = A @ pos + b
        # Reconstruction.
        v_recon = bernstein_cubic_evaluate(hct.coeffs[e, s], bary)
        err = np.abs(v_recon - v_exact).max()
        max_err = max(max_err, err)

    # Float32 storage → expect ~1e-6 error, allow a comfortable margin.
    assert max_err < 1e-5, f"max err vs linear field = {max_err:.3e}"


def test_build_hct_bernstein_c0_reproduces_vertex_values():
    """The B-coefficient at a sub-tet vertex must equal the vertex
    value exactly (float32 precision). For parent vertices this is the
    nodal velocity; for the vc vertex this is the P1-averaged μ."""
    nodes, conn = make_kuhn_cube_mesh(2)
    geom = build_alfeld_geometry(nodes, conn)
    # Non-trivial per-node velocity so we can distinguish vertex values.
    rng = np.random.default_rng(0)
    vel = rng.standard_normal(nodes.shape)
    grad = rng.standard_normal((nodes.shape[0], 3, 3))
    egrads = edge_midpoint_gradients(nodes, conn, grad)
    hct = build_hct_bernstein_c0(nodes, conn, vel, grad, egrads, geom)

    # For each element and sub-tet, evaluate at the 4 sub-tet vertices
    # (barycentric = e_i) and verify the value matches the vertex data.
    for e in range(min(20, conn.shape[0])):
        vc_e = geom.centroids[e]
        # μ = mean of 4 parent-vertex velocities.
        mu_e = vel[conn[e]].mean(axis=0)
        for s in range(4):
            fs = ALFELD_SUBTET_PARENT_VERTS[s]
            expected_vertex_vals = [
                mu_e,
                vel[conn[e, fs[0]]],
                vel[conn[e, fs[1]]],
                vel[conn[e, fs[2]]],
            ]
            for i in range(4):
                bary = np.zeros(4); bary[i] = 1.0
                val = bernstein_cubic_evaluate(hct.coeffs[e, s], bary)
                err = np.abs(val - expected_vertex_vals[i]).max()
                assert err < 1e-6, (
                    f"element {e} sub-tet {s} vertex {i}: err = {err}"
                )


# -----------------------------------------------------------------------
# C⁰ interior continuity
# -----------------------------------------------------------------------

def test_c0_continuity_at_shared_spoke_edge_midpoint():
    """A spoke edge (from vc to a parent vertex p_j) is shared by three
    sub-tets in the Alfeld split. At the midpoint of the spoke edge,
    all three sub-tets should agree on the reconstructed value (C⁰)."""
    nodes, conn = make_kuhn_cube_mesh(2)
    geom = build_alfeld_geometry(nodes, conn)
    rng = np.random.default_rng(0)
    vel = rng.standard_normal(nodes.shape)
    grad = rng.standard_normal((nodes.shape[0], 3, 3))
    egrads = edge_midpoint_gradients(nodes, conn, grad)
    hct = build_hct_bernstein_c0(nodes, conn, vel, grad, egrads, geom)

    # For element 0 and spoke edge from vc to parent vertex 0:
    # sub-tets 1, 2, 3 all contain this spoke edge (their w0 = vc,
    # and one of their w_i = p_0). Find which local vertex slot is p_0
    # in each of sub-tets 1, 2, 3; then set barycentric = (0.5, 0.5,
    # 0.0, 0.0) with the 0.5 slots on w0 (vc) and the p_0 slot.
    e = 0
    for parent_vertex in range(4):
        # Which sub-tets contain parent_vertex? All sub-tets except
        # ALFELD_SUBTET_PARENT_VERTS[s] not containing parent_vertex
        # (i.e. all sub-tets s where parent_vertex ∈ face triangle s).
        subtets_with_pv = [s for s in range(4)
                           if parent_vertex in ALFELD_SUBTET_PARENT_VERTS[s]]
        # Evaluate at spoke-edge midpoint from each.
        vals = []
        for s in subtets_with_pv:
            fs = ALFELD_SUBTET_PARENT_VERTS[s]
            # Local slot of parent_vertex in sub-tet s.
            local = 1 + int(np.where(fs == parent_vertex)[0][0])
            bary = np.zeros(4)
            bary[0] = 0.5  # vc
            bary[local] = 0.5
            val = bernstein_cubic_evaluate(hct.coeffs[e, s], bary)
            vals.append(val)
        # All should agree.
        for v in vals[1:]:
            err = np.abs(v - vals[0]).max()
            assert err < 1e-6, (
                f"spoke edge to parent_vertex {parent_vertex}: "
                f"disagreement between sub-tets = {err}"
            )


def test_c0_continuity_at_parent_face_edge_midpoint():
    """A parent-face edge (from p_a to p_b) is shared by two sub-tets
    (the two whose base triangle contains this edge). At the midpoint
    of the parent-face edge, the two sub-tets should agree."""
    nodes, conn = make_kuhn_cube_mesh(2)
    geom = build_alfeld_geometry(nodes, conn)
    rng = np.random.default_rng(1)
    vel = rng.standard_normal(nodes.shape)
    grad = rng.standard_normal((nodes.shape[0], 3, 3))
    egrads = edge_midpoint_gradients(nodes, conn, grad)
    hct = build_hct_bernstein_c0(nodes, conn, vel, grad, egrads, geom)

    e = 0
    for a, b in TET_EDGES:
        # Sub-tets whose base triangle contains BOTH a and b.
        subtets_with_edge = [s for s in range(4)
                             if (a in ALFELD_SUBTET_PARENT_VERTS[s] and
                                 b in ALFELD_SUBTET_PARENT_VERTS[s])]
        if len(subtets_with_edge) < 2:
            continue
        vals = []
        for s in subtets_with_edge:
            fs = ALFELD_SUBTET_PARENT_VERTS[s]
            la = 1 + int(np.where(fs == a)[0][0])
            lb = 1 + int(np.where(fs == b)[0][0])
            bary = np.zeros(4)
            bary[la] = 0.5
            bary[lb] = 0.5
            val = bernstein_cubic_evaluate(hct.coeffs[e, s], bary)
            vals.append(val)
        for v in vals[1:]:
            err = np.abs(v - vals[0]).max()
            assert err < 1e-6, (
                f"parent edge ({a}, {b}): disagreement = {err}"
            )


if __name__ == "__main__":
    fns = [(name, fn) for name, fn in globals().items()
           if name.startswith("test_") and callable(fn)]
    print(f"Running {len(fns)} tests...\n")
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"  ERR   {name}: {type(exc).__name__}: {exc}")
    print()
    if failed:
        print(f"{failed} failure(s)")
        sys.exit(1)
    print("all tests passed")
