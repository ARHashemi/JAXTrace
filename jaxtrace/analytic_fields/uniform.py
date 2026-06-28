"""
Uniform velocity field — sanity baseline for the analytic path.

Defines `v(x) = (V_ref, 0, 0)` everywhere. Used as a regression test:
particles seeded at x₀ should arrive at x₀ + V_ref·t·ê_x exactly, to
machine precision, after N RK4 steps of size dt with N·dt = t.

Usage
-----
Pass this file directly to `run_tracking.py --velocity-module`:

    run_tracking.py --velocity-source analytic \\
        --velocity-module jaxtrace/analytic_fields/uniform.py \\
        --domain-bbox "-4 4 -2 2 -0.25 0.25"

Override `V_ref` in a copy of this file if you want a different value.
"""

import jax.numpy as jnp

from jaxtrace.gpu.tracking.velocity_provider import AnalyticVelocityProvider


# Reference parameters. Override by copying this file and editing.
V_REF = 1.0


def velocity_fn(pos):
    """Constant velocity along +x.

    Args
    ----
    pos : jnp.ndarray, shape (3,)
        Position of one particle, in metres.

    Returns
    -------
    jnp.ndarray, shape (3,)
        Velocity (V_ref, 0, 0), in m/s.
    """
    del pos  # uniform field
    return jnp.array([V_REF, 0.0, 0.0])


def analytical_trajectory(pos0, t):
    """Closed-form trajectory: x(t) = x₀ + V_ref · t · ê_x.

    Used by tests/test_analytic_path.py to validate the RK4 step.
    """
    pos0 = jnp.asarray(pos0)
    return pos0 + jnp.array([V_REF * t, 0.0, 0.0])


def build_provider(domain_bbox=None, dt=0.0, t_start=0.0):
    """Construct an AnalyticVelocityProvider for the uniform field.

    Args
    ----
    domain_bbox : ((xmin,xmax),(ymin,ymax),(zmin,zmax)), optional
        Defaults to a generous slab around the origin if not supplied.
        Particles outside this bbox are handled by whatever wall mode
        the driver was configured with.
    dt, t_start : float
        Accepted for interface compatibility; unused here.
    """
    del dt, t_start
    if domain_bbox is None:
        domain_bbox = ((-10.0, 10.0), (-2.0, 2.0), (-0.5, 0.5))
    return AnalyticVelocityProvider(
        velocity_fn=velocity_fn,
        is_time_dependent=False,
        level_set_fn=None,
        domain_bbox=domain_bbox,
        meta={
            "name": "uniform",
            "params": {"V_ref": V_REF},
            "closed_form_trajectory": True,
        },
    )
