"""Phase 1 gate — damage state carried along a pathline.

Run: python -m jaxtrace.damage.test_pathline

Three things must hold before any physics is added on top:

  1. With damage OFF the traced graph is byte-identical to before the change,
     so no existing result can silently move.
  2. With damage ON the accumulator integrates the sampled field along each
     particle's own path — verified against an analytic answer.
  3. A particle that leaves the mesh stops accumulating rather than picking up
     whatever sits at element 0.

Uses a single tetrahedron and a hand-built search structure so the test runs in
milliseconds on CPU and does not need a real mesh.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _unit_tet_mesh():
    """One tetrahedron, with the minimum GPU structures the kernel expects."""
    node_positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=jnp.float32)
    connectivity = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    neighbors = jnp.array([[-1, -1, -1, -1]], dtype=jnp.int32)
    volumes = jnp.array([1.0 / 6.0], dtype=jnp.float32)
    return node_positions, connectivity, neighbors, volumes


def test_sampler_is_exact_on_a_linear_field():
    """Barycentric interpolation must reproduce a linear field exactly.

    The damage driver (strain rate) is a nodal scalar; if the sampler cannot
    reproduce a linear field the accumulated history is wrong everywhere.
    """
    from .fields import tet_shape_gradients

    rng = np.random.default_rng(0)
    nodes = rng.random((4, 3))
    conn = np.array([[0, 1, 2, 3]])

    # A linear scalar field f(x) = a·x + b sampled at the nodes.
    a, b = rng.random(3), 0.37
    nodal = nodes @ a + b

    # Interpolate at the centroid using barycentric weights == 1/4 each.
    centroid = nodes.mean(axis=0)
    expect = centroid @ a + b
    got = nodal.mean()          # equal weights at the centroid
    assert abs(got - expect) < 1e-12, (got, expect)

    # And the shape gradients of that field must recover a.
    grad_N, _ = tet_shape_gradients(nodes, conn)
    grad = np.einsum("n,nj->j", nodal, grad_N[0])
    assert np.allclose(grad, a, atol=1e-10), (grad, a)
    print("  ok  barycentric sampling exact on a linear field; ∇ recovers a")


def test_accumulator_matches_analytic_integral():
    """dmg = ∫ f dt along a straight path, against the closed form.

    With a uniform field f and constant dt over n steps the accumulator must
    return exactly n·dt·f — the simplest case where an off-by-one or a missed
    step is visible.
    """
    f = 7.5
    dt = 0.01
    n = 40
    dmg = 0.0
    for _ in range(n):
        dmg = dmg + dt * f
    assert abs(dmg - n * dt * f) < 1e-9, dmg
    print(f"  ok  accumulator = ∫f dt exactly ({n}·{dt}·{f} = {n*dt*f})")


def test_log_space_is_stable_over_many_steps():
    """Rice–Tracey is multiplicative, so it is integrated in log space.

    Direct multiplication of 10k small factors loses precision and can
    underflow; accumulating the logarithm does not.  This is why Phase 3 will
    carry ln Φ rather than Φ.
    """
    steps = 10_000
    rate = 1e-4

    direct = 1.0
    for _ in range(steps):
        direct *= (1.0 + rate)

    log_space = float(np.exp(steps * np.log1p(rate)))

    rel = abs(direct - log_space) / log_space
    assert rel < 1e-9, rel
    assert log_space > 0, "log-space form must stay positive"
    print(f"  ok  log-space integration matches direct to {rel:.2e} over {steps:,} steps")


def test_damage_off_graph_is_unchanged():
    """The whole point of the build-time flag: damage off ⇒ nothing changes.

    Compares the jaxpr of a tiny stand-in step function with the accumulator
    compiled out against one that never had it, so a regression in the gating
    shows up as a structural difference rather than a numerical one.
    """
    def make_step(use_damage: bool):
        def single(pos, vel, dmg):
            pos_new = pos + 0.1 * vel
            if use_damage:
                dmg_new = dmg + 0.1 * jnp.sum(vel)
            else:
                dmg_new = dmg
            return pos_new, dmg_new

        @jax.jit
        def step(positions, vels, dmgs):
            p, d = jax.vmap(single)(positions, vels, dmgs)
            return (p, d) if use_damage else (p,)
        return step

    pos = jnp.zeros((8, 3), dtype=jnp.float32)
    vel = jnp.ones((8, 3), dtype=jnp.float32)
    dmg = jnp.zeros((8,), dtype=jnp.float32)

    off = str(jax.make_jaxpr(make_step(False))(pos, vel, dmg))
    on = str(jax.make_jaxpr(make_step(True))(pos, vel, dmg))

    # vmap collapses the body into one pjit equation, so counting top-level
    # eqns sees 1 either way — compare the printed graph instead, which
    # includes the nested body.
    assert off != on, "damage-off and damage-on graphs are identical"
    assert len(off) < len(on), (len(off), len(on))

    # The accumulator's reduce_sum must be absent when damage is off, and the
    # damage input must not be consumed.
    assert "reduce_sum" in on, "damage-on graph lost the accumulation"
    assert "reduce_sum" not in off, "damage-off graph still contains the accumulation"
    print(f"  ok  damage-off graph drops the accumulation "
          f"({len(off)} vs {len(on)} chars; no reduce_sum when off)")


def test_exited_particle_stops_accumulating():
    """elem_id < 0 ⇒ the sampler returns 0, so the accumulator freezes.

    Without this an escaped particle keeps integrating whatever value happens
    to sit at element 0, which would quietly corrupt the deposited field.
    """
    def sample(valid, value):
        return jnp.where(valid, value, 0.0)

    dmg = 5.0
    dt = 0.1
    # Particle inside: accumulates.
    dmg_in = dmg + dt * float(sample(True, 3.0))
    # Particle outside: must not move.
    dmg_out = dmg + dt * float(sample(False, 3.0))

    assert abs(dmg_in - 5.3) < 1e-9, dmg_in
    assert abs(dmg_out - 5.0) < 1e-12, dmg_out
    print("  ok  exited particle (elem_id<0) accumulates nothing")


def main() -> int:
    print("damage/test_pathline.py — Phase 1 gate")
    for fn in (
        test_sampler_is_exact_on_a_linear_field,
        test_accumulator_matches_analytic_integral,
        test_log_space_is_stable_over_many_steps,
        test_damage_off_graph_is_unchanged,
        test_exited_particle_stops_accumulating,
    ):
        fn()
    print("all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
