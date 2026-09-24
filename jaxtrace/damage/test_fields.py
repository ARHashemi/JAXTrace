"""Analytic checks for the damage field builder.

Run: python -m jaxtrace.damage.test_fields

These exist because the shape-gradient construction has a silent failure mode:
transposing the inverse Jacobian still passes on axis-aligned tetrahedra and
only breaks on skewed ones, where it shows up as a violation of
incompressibility rather than as an obvious error.  ``test_skewed_linear_field``
is the regression guard for exactly that.
"""

import numpy as np

from .fields import (
    build_damage_fields,
    tet_shape_gradients,
    velocity_gradient_per_element,
)


def _unit_tet():
    nodes = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return nodes, np.array([[0, 1, 2, 3]])


def test_volume_and_partition_of_unity():
    nodes, conn = _unit_tet()
    grad_N, vol = tet_shape_gradients(nodes, conn)
    assert abs(vol[0] - 1.0 / 6.0) < 1e-12, vol[0]
    # The four shape functions sum to 1 everywhere, so their gradients sum to 0.
    assert np.allclose(grad_N[0].sum(axis=0), 0.0, atol=1e-12)
    print("  ok  volume = 1/6, shape gradients sum to zero")


def test_skewed_linear_field():
    """The regression guard. A linear field u = A x must give back L = A.

    On a SKEWED tet, so that a transposed inverse Jacobian cannot pass.
    """
    rng = np.random.default_rng(1)
    nodes = rng.random((4, 3))
    conn = np.array([[0, 1, 2, 3]])
    grad_N, _ = tet_shape_gradients(nodes, conn)

    A = np.array([[1.0, 2.0, -0.5],
                  [0.3, -0.4, 1.1],
                  [0.7, -0.2, -0.6]])
    assert abs(np.trace(A)) < 1e-14, "test matrix must be traceless"

    u = nodes @ A.T
    L = velocity_gradient_per_element(u, conn, grad_N)[0]

    assert np.allclose(L, A, atol=1e-10), f"\nrecovered\n{L}\nexpected\n{A}"
    assert abs(np.trace(L)) < 1e-10, f"tr(L) = {np.trace(L)}"
    print("  ok  skewed tet recovers L = A exactly, tr(L) = 0")


def test_uniaxial_normalisation():
    """von Mises normalisation: uniaxial at rate e must give edot_eff = e."""
    nodes, conn = _unit_tet()
    grad_N, _ = tet_shape_gradients(nodes, conn)
    # Incompressible uniaxial extension.
    u = np.stack([nodes[:, 0], -nodes[:, 1] / 2, -nodes[:, 2] / 2], axis=1)
    L = velocity_gradient_per_element(u, conn, grad_N)[0]
    D = 0.5 * (L + L.T)
    edot = np.sqrt(2.0 / 3.0 * np.sum(D * D))
    assert abs(edot - 1.0) < 1e-12, edot
    print("  ok  uniaxial edot_eff = 1.0 exactly")


def test_rigid_rotation_is_not_damage():
    """Rigid rotation must give D = 0 — spin does no work, causes no damage.

    This is the formal basis for dismissing vorticity as a void indicator:
    the tool spins everything, but only the symmetric part deforms material.
    """
    nodes, conn = _unit_tet()
    grad_N, _ = tet_shape_gradients(nodes, conn)
    omega = np.array([0.0, 0.0, 2.0])
    u = np.cross(np.broadcast_to(omega, (4, 3)), nodes)
    L = velocity_gradient_per_element(u, conn, grad_N)[0]
    D = 0.5 * (L + L.T)
    W = 0.5 * (L - L.T)
    assert np.abs(D).max() < 1e-12, f"D nonzero: {np.abs(D).max()}"
    assert abs(W[0, 1] + 2.0) < 1e-12, W[0, 1]
    print("  ok  rigid rotation: D = 0, spin W_xy = -2")


def test_pressure_sign_and_triaxiality():
    """sigma_m = -P, and eta has the expected sign."""
    nodes, conn = _unit_tet()
    u = np.stack([nodes[:, 0], -nodes[:, 1] / 2, -nodes[:, 2] / 2], axis=1)
    # Compressive fluid pressure P > 0 -> sigma_m < 0 -> eta < 0.
    P = np.full(4, 5.0e6)
    out = build_damage_fields(nodes, conn, u, P,
                              sigma_flow=np.full(4, 30.0e6), verbose=False)
    assert np.all(out["sigma_m"] < 0), "P>0 must give compressive sigma_m"
    assert np.all(out["eta"] < 0), "compressive sigma_m must give eta<0"
    print("  ok  P>0 -> sigma_m<0 -> eta<0 (voids suppressed under compression)")


def main():
    print("damage/fields.py analytic checks")
    for fn in (
        test_volume_and_partition_of_unity,
        test_skewed_linear_field,
        test_uniaxial_normalisation,
        test_rigid_rotation_is_not_damage,
        test_pressure_sign_and_triaxiality,
    ):
        fn()
    print("all passed")


if __name__ == "__main__":
    main()
