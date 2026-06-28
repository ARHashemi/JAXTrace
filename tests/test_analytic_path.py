"""
Phase 2 unit test: end-to-end RK4 on an analytic velocity field.

Runs the analytic path (provider -> create_rk4_analytic -> jit) against
the uniform field, where the closed-form trajectory is exact, and
checks that after N steps the particle position is x₀ + V_ref·N·dt to
machine precision.

Then runs the divergence-free recirculation field for a few steps and
sanity-checks the result against scipy.integrate.solve_ivp at high
tolerance. Not a parity test (RK4 with finite dt ≠ DOP853 at dt→0) —
just a regression catch.
"""

from __future__ import annotations

import os
import sys

# Path setup.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np


# =============================================================================
# Test 1 — uniform field, closed-form trajectory
# =============================================================================

def test_uniform_closed_form():
    """N steps of size dt on v = (V_ref, 0, 0) starting at x₀ must give
    x₀ + V_ref · N · dt · ê_x to machine precision."""
    import jax
    import jax.numpy as jnp
    from jaxtrace.gpu.tracking.velocity_provider import load_analytic_provider
    from benchmark_femuss_comparison import create_rk4_analytic

    # Build provider from the uniform field module.
    mod_path = os.path.join(_REPO_ROOT, 'jaxtrace', 'analytic_fields', 'uniform.py')
    provider = load_analytic_provider(
        module_path=mod_path,
        domain_bbox=((-1000.0, 1000.0), (-1000.0, 1000.0), (-1000.0, 1000.0)),
        dt=0.0,
    )
    assert provider.is_mesh_based is False
    assert provider.is_time_dependent is False

    # Build the analytic RK4 kernel (no clamp, no boundary projection).
    rk4 = create_rk4_analytic(
        provider,
        use_substep_bbox_clamp=False,
        use_boundary_projection_clamp=False,
    )

    # Run 50 steps of dt = 0.01.
    dt = jnp.float64(0.01)
    n_steps = 50

    # 1024 particles at random positions in the bbox.
    rng = np.random.default_rng(42)
    pos0 = jnp.asarray(
        rng.uniform(-1.0, 1.0, size=(1024, 3)).astype(np.float64)
    )
    elem_ids = jnp.full(pos0.shape[0], -1, dtype=jnp.int32)

    # Dummy velocity_fields_gpu (kernel ignores it on analytic path).
    velocity_fields_gpu = jnp.zeros((1, 1, 3), dtype=jnp.float64)

    pos = pos0
    for step in range(n_steps):
        pos, elem_ids = rk4(pos, elem_ids, dt, velocity_fields_gpu, step)
    pos_final = np.asarray(pos)

    # Closed-form expectation.
    V_ref = 1.0  # uniform.py default
    t_total = float(dt) * n_steps
    expected = np.asarray(pos0) + np.array([V_ref * t_total, 0.0, 0.0])

    err = np.abs(pos_final - expected)
    max_err = err.max()
    print(f"  uniform field, {n_steps} steps × dt={float(dt)}:")
    print(f"    max position error vs closed-form: {max_err:.3e}")
    # RK4 has zero truncation error for constant velocity (linear ODE,
    # exact for any polynomial of degree ≤ 4). So this should be at
    # float64 round-off level.
    assert max_err < 1e-12, f"uniform field RK4 error too large: {max_err:.3e}"
    print("  [OK] uniform field closed-form parity")


# =============================================================================
# Test 2 — divergence-free recirculation, scipy reference
# =============================================================================

def test_divergence_free_recirc_vs_scipy():
    """Run 20 steps of dt=1e-3 on the PDF §A field for one particle far
    upstream (R >> L, so essentially in the uniform-stream regime).
    Compare to a scipy DOP853 reference at rtol=atol=1e-12."""
    import jax
    import jax.numpy as jnp
    from jaxtrace.gpu.tracking.velocity_provider import load_analytic_provider
    from jaxtrace.analytic_fields.divergence_free_recirculation import (
        analytical_trajectory,
    )
    from benchmark_femuss_comparison import create_rk4_analytic

    mod_path = os.path.join(
        _REPO_ROOT, 'jaxtrace', 'analytic_fields',
        'divergence_free_recirculation.py',
    )
    provider = load_analytic_provider(
        module_path=mod_path,
        domain_bbox=((-10.0, 10.0), (-3.0, 3.0), (-0.5, 0.5)),
        dt=0.0,
    )

    rk4 = create_rk4_analytic(
        provider,
        use_substep_bbox_clamp=False,
        use_boundary_projection_clamp=False,
    )

    # One particle far upstream, on y=0.5 (off-axis so we get a nonzero
    # v component from the sinusoid).
    pos0 = jnp.asarray([[-3.0, 0.5, 0.0]], dtype=jnp.float64)
    elem_ids = jnp.full(1, -1, dtype=jnp.int32)
    velocity_fields_gpu = jnp.zeros((1, 1, 3), dtype=jnp.float64)

    dt = jnp.float64(1e-3)
    n_steps = 20
    t_final = float(dt) * n_steps

    pos = pos0
    for step in range(n_steps):
        pos, elem_ids = rk4(pos, elem_ids, dt, velocity_fields_gpu, step)

    # scipy reference.
    ref = analytical_trajectory(np.asarray(pos0[0]), t_final, rtol=1e-12, atol=1e-12)

    err = np.abs(np.asarray(pos[0]) - ref)
    max_err = err.max()
    print(f"  divergence-free recirc, {n_steps} steps × dt={float(dt)}:")
    print(f"    particle:     pos0={pos0[0]}  ->  pos_final={pos[0]}")
    print(f"    scipy ref:    {ref}")
    print(f"    max error:    {max_err:.3e}")
    # RK4 truncation error at dt=1e-3 for this smooth field should be
    # well under 1e-6 for 20 steps. Allow some slack.
    assert max_err < 1e-6, f"divergence-free RK4 vs scipy too large: {max_err:.3e}"
    print("  [OK] divergence-free recirc matches scipy reference")


# =============================================================================
# Test 3 — substep clamp keeps particles in the bbox
# =============================================================================

def test_substep_clamp_keeps_in_bbox():
    """A particle started near the +x edge of a tight bbox, with the
    uniform field pushing it further +x, should be clamped to the bbox
    when use_substep_bbox_clamp=True."""
    import jax.numpy as jnp
    from jaxtrace.gpu.tracking.velocity_provider import load_analytic_provider
    from benchmark_femuss_comparison import create_rk4_analytic

    mod_path = os.path.join(_REPO_ROOT, 'jaxtrace', 'analytic_fields', 'uniform.py')
    provider = load_analytic_provider(
        module_path=mod_path,
        domain_bbox=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        dt=0.0,
    )

    rk4 = create_rk4_analytic(
        provider,
        use_substep_bbox_clamp=True,
        use_boundary_projection_clamp=True,
    )

    pos0 = jnp.asarray([[0.99, 0.0, 0.0]], dtype=jnp.float64)
    elem_ids = jnp.full(1, -1, dtype=jnp.int32)
    velocity_fields_gpu = jnp.zeros((1, 1, 3), dtype=jnp.float64)
    dt = jnp.float64(0.1)

    # 50 steps, each would move the particle +0.1 in x (V_ref=1, dt=0.1).
    # Without clamp, x would reach ~5.99. With clamp, it stays ≤ 1.0.
    pos = pos0
    for step in range(50):
        pos, elem_ids = rk4(pos, elem_ids, dt, velocity_fields_gpu, step)

    x_final = float(pos[0, 0])
    print(f"  substep clamp test: x_final = {x_final:.6f}  (bbox_max = 1.0)")
    assert x_final <= 1.0 + 1e-6, f"clamp failed: x_final={x_final}"
    assert x_final > 0.99, f"clamp pulled too far: x_final={x_final}"
    print("  [OK] substep clamp respects bbox")


# =============================================================================
# Entry point
# =============================================================================

def main():
    print("Phase 2 analytic-path tests")
    print()
    print("Test 1: uniform field, closed-form parity")
    test_uniform_closed_form()
    print()
    print("Test 2: divergence-free recirc vs scipy")
    test_divergence_free_recirc_vs_scipy()
    print()
    print("Test 3: substep clamp keeps particles in bbox")
    test_substep_clamp_keeps_in_bbox()
    print()
    print("All analytic-path tests PASS")


if __name__ == "__main__":
    main()
