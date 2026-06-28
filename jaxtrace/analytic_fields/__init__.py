"""
jaxtrace.analytic_fields
========================

Reference implementations of analytic velocity fields for the
`--velocity-source analytic` path of `run_tracking.py`.

Each submodule exports exactly one symbol — `build_provider(domain_bbox,
dt, t_start=0.0)` — that returns an
`AnalyticVelocityProvider` (see `jaxtrace/gpu/tracking/velocity_provider.py`).

A user can either:
  * point `--velocity-module` at one of these files directly, or
  * copy one of them as a template and edit the `velocity_fn`,
    `domain_bbox`, and `default_params` to define a custom field.

Available fields
----------------
uniform
    Constant `v(x) = (V_ref, 0, 0)`. Sanity baseline; analytic
    trajectories are closed-form (`x(t) = x_0 + V_ref · t`).

divergence_free_recirculation
    From "FSW Internal Summary" appendix §A: a uniform stream with a
    Gaussian-localised, sinusoidal, divergence-free disturbance that
    produces a pair of counter-rotating recirculation cells. Defined
    in the (x, y) plane with w = 0; extruded trivially through z.
"""

__all__ = []  # populated by the submodules themselves if they want
