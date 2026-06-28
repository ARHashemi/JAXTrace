"""
test_velocity_provider_parity.py
=================================

Phase 1 parity test: confirm the velocity_provider abstraction is a
behavioural no-op for the mesh path.

Two layers of test:

1. Static contract (runs anywhere, no GPU, no mesh files).
   * MeshVelocityProvider.sample() forwards to the provided search +
     interpolate closures with the right argument shapes.
   * AnalyticVelocityProvider.sample() ignores hint_elem and
     velocity_field, dispatches on is_time_dependent.
   * load_analytic_provider() rejects modules without build_provider().

2. Kernel-level parity (run on the workstation).
   Re-run cylindrical_005's tracking with the refactored kernel and
   compare particle.vtkhdf to a saved reference produced by the
   pre-refactor kernel. See the docstring of run_kernel_parity_check()
   below for instructions.
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap

# Ensure the jaxtrace package is importable when this file is run
# directly via `python tests/test_velocity_provider_parity.py` from the
# repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np


# =============================================================================
# Layer 1 — static contract
# =============================================================================

def test_mesh_provider_forwards_to_closures():
    """MeshVelocityProvider.sample() must call the supplied closures
    with the right argument shapes and return their results unchanged."""
    import jax.numpy as jnp
    from jaxtrace.gpu.tracking.velocity_provider import MeshVelocityProvider

    captured = {}

    def fake_search(pos, hint):
        captured['search'] = (pos, hint)
        return jnp.int32(42)

    def fake_interp(pos, elem_id, field):
        captured['interp'] = (pos, elem_id, field)
        return jnp.array([1.0, 2.0, 3.0])

    def fake_tool(pos, elem_id):
        captured['tool'] = (pos, elem_id)
        return jnp.bool_(True)

    p = MeshVelocityProvider(
        search_fn=fake_search,
        interpolate_fn=fake_interp,
        check_inside_tool_fn=fake_tool,
    )

    pos = jnp.array([0.1, 0.2, 0.3])
    hint = jnp.int32(7)
    field = jnp.array([[0.0, 0.0, 0.0]])
    t = jnp.float32(0.0)

    vel, elem = p.sample(pos, hint, field, t)
    assert elem == 42, f"sample() returned elem={elem}, expected 42"
    assert tuple(vel) == (1.0, 2.0, 3.0)
    # search received raw pos + hint
    assert tuple(captured['search'][0]) == (0.1, 0.2, 0.3)
    assert int(captured['search'][1]) == 7
    # interp received the searched elem
    assert int(captured['interp'][1]) == 42

    mask = p.tool_mask(pos, jnp.int32(99), t)
    assert bool(mask) is True
    assert int(captured['tool'][1]) == 99

    print("  [OK] MeshVelocityProvider forwards correctly")


def test_mesh_provider_tool_mask_no_check():
    """When check_inside_tool_fn is None, tool_mask returns False."""
    import jax.numpy as jnp
    from jaxtrace.gpu.tracking.velocity_provider import MeshVelocityProvider

    def trivial_search(p, h): return jnp.int32(0)
    def trivial_interp(p, e, f): return jnp.zeros(3)

    p = MeshVelocityProvider(
        search_fn=trivial_search,
        interpolate_fn=trivial_interp,
        check_inside_tool_fn=None,
    )
    assert bool(p.tool_mask(jnp.zeros(3), jnp.int32(0), jnp.float32(0.0))) is False
    print("  [OK] MeshVelocityProvider tool_mask returns False when no level-set")


def test_analytic_steady():
    """Steady analytic provider must ignore t."""
    import jax.numpy as jnp
    from jaxtrace.gpu.tracking.velocity_provider import AnalyticVelocityProvider

    def vfn(pos):
        return jnp.array([1.0, pos[0], 0.0])

    p = AnalyticVelocityProvider(velocity_fn=vfn, is_time_dependent=False)

    pos = jnp.array([0.5, 0.0, 0.0])
    vel, elem = p.sample(pos, jnp.int32(99), jnp.zeros((1, 3)), jnp.float32(1.0e9))
    assert int(elem) == -1, "analytic elem must be -1"
    assert tuple(vel) == (1.0, 0.5, 0.0)
    print("  [OK] AnalyticVelocityProvider steady mode")


def test_analytic_unsteady():
    """Unsteady analytic provider must pass t."""
    import jax.numpy as jnp
    from jaxtrace.gpu.tracking.velocity_provider import AnalyticVelocityProvider

    def vfn(pos, t):
        return jnp.array([t, 0.0, 0.0])

    p = AnalyticVelocityProvider(velocity_fn=vfn, is_time_dependent=True)
    vel, _ = p.sample(jnp.zeros(3), jnp.int32(0), jnp.zeros((1, 3)), jnp.float32(0.5))
    assert float(vel[0]) == 0.5
    print("  [OK] AnalyticVelocityProvider unsteady mode")


def test_loader_rejects_module_without_build_provider():
    """A module that doesn't expose build_provider() must raise."""
    from jaxtrace.gpu.tracking.velocity_provider import load_analytic_provider

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write("# empty\n")
        path = f.name

    try:
        try:
            load_analytic_provider(path)
        except AttributeError as e:
            assert "build_provider" in str(e)
            print("  [OK] loader rejects module without build_provider()")
            return
        raise AssertionError("loader should have raised AttributeError")
    finally:
        os.unlink(path)


def test_loader_calls_build_provider():
    """Loader must call build_provider() and return its result."""
    from jaxtrace.gpu.tracking.velocity_provider import (
        load_analytic_provider, AnalyticVelocityProvider,
    )

    body = textwrap.dedent("""
        import jax.numpy as jnp
        from jaxtrace.gpu.tracking.velocity_provider import AnalyticVelocityProvider

        def velocity_fn(pos):
            return jnp.array([1.0, 0.0, 0.0])

        def build_provider(domain_bbox, dt, t_start=0.0):
            return AnalyticVelocityProvider(
                velocity_fn=velocity_fn,
                is_time_dependent=False,
                domain_bbox=domain_bbox,
                meta={"name": "test"},
            )
    """).strip()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(body)
        path = f.name

    try:
        bbox = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        p = load_analytic_provider(path, domain_bbox=bbox, dt=0.001)
        assert isinstance(p, AnalyticVelocityProvider)
        assert p.is_time_dependent is False
        assert p.domain_bbox == bbox
        assert p.meta is not None and p.meta.get("name") == "test"
        print("  [OK] loader calls build_provider() correctly")
    finally:
        os.unlink(path)


# =============================================================================
# Layer 2 — kernel-level parity (run on the workstation, not here)
# =============================================================================

def run_kernel_parity_check(reference_particles_vtkhdf: str,
                            cohort_case_dir: str,
                            n_steps: int = 100,
                            tol_position: float = 1e-12):
    """Run cylindrical_005's tracking with the refactored kernel and
    compare positions step-by-step to a saved pre-refactor reference.

    This must be run on the workstation. Steps:

      1. On the PRE-REFACTOR commit (e.g. feature/density HEAD):
           cd /scratch/shared/ROM/FOM/cylindrical_005.gid
           N_STEPS=100 bash run_jaxtrace.sh    # writes particles.vtkhdf
         Copy the result to a stable path (e.g. /tmp/reference.vtkhdf).

      2. Check out feature/analytic-velocity (this branch).

      3. Re-run the same case with the same N_STEPS=100 and the same
         seed. Capture the new particles.vtkhdf.

      4. Call run_kernel_parity_check(reference=<step 1 path>,
                                     cohort_case_dir=<case dir>,
                                     n_steps=100).
         The function loads both files, picks the same step indices, and
         compares particle positions in float64. Differences above
         tol_position (default 1e-12) constitute a regression.

    Implementation note: we deliberately do not auto-run the tracking
    here. The driver is mature, the only thing to verify is that two
    runs with identical inputs produce identical outputs. Doing the
    runs manually keeps the test loop tight and the failure mode
    obvious.
    """
    import h5py

    if not os.path.isfile(reference_particles_vtkhdf):
        raise FileNotFoundError(
            f"reference file missing: {reference_particles_vtkhdf}\n"
            "Run the pre-refactor kernel first (see docstring)."
        )

    # Find the most recent run_*/particles.vtkhdf in the case folder.
    import glob
    candidates = sorted(
        glob.glob(os.path.join(cohort_case_dir, "post_pt", "run_*", "particles.vtkhdf")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no particles.vtkhdf under {cohort_case_dir}/post_pt/run_*\n"
            "Run the refactored kernel first."
        )
    new_particles = candidates[0]

    print(f"reference: {reference_particles_vtkhdf}")
    print(f"new:       {new_particles}")

    def _load_positions(path):
        with h5py.File(path, 'r') as f:
            return f['VTKHDF/Points'][:]

    ref = _load_positions(reference_particles_vtkhdf)
    new = _load_positions(new_particles)

    if ref.shape != new.shape:
        print(f"FAIL: shape mismatch {ref.shape} vs {new.shape}")
        return False

    diff = np.abs(ref.astype(np.float64) - new.astype(np.float64))
    max_diff = float(diff.max())
    print(f"max position diff: {max_diff:.3e}  (tol {tol_position:.3e})")
    if max_diff > tol_position:
        # Top 10 worst points for diagnosis.
        idx = np.argsort(diff.max(axis=1))[-10:]
        print("worst 10:")
        for i in idx:
            print(f"  i={i}  ref={ref[i]}  new={new[i]}  diff={diff[i]}")
        return False

    print("PASS")
    return True


# =============================================================================
# Entry point
# =============================================================================

def main():
    print("Layer 1 — static contract")
    test_mesh_provider_forwards_to_closures()
    test_mesh_provider_tool_mask_no_check()
    test_analytic_steady()
    test_analytic_unsteady()
    test_loader_rejects_module_without_build_provider()
    test_loader_calls_build_provider()
    print()
    print("Layer 1 OK.")
    print()
    print("Layer 2 — kernel parity")
    print("  See run_kernel_parity_check() docstring.")
    print("  Must run on the workstation with cylindrical_005 mounted.")


if __name__ == "__main__":
    main()
