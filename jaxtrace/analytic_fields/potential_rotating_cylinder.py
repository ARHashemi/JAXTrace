"""
Analytical divergence-free velocity field: potential flow around a
rotating cylinder (with circulation, i.e. the classical Magnus-effect
solution).

Implementation of the field from "FSW Internal Summary" (H. Venghaus,
June 2026), appendix §B.  Constructed from the streamfunction

    ψ(r, θ) = V_ref · (r − a²/r) · sin θ + (Γ / 2π) · ln r,

or equivalently in Cartesian coordinates,

    ψ(x, y) = V_ref · (r − a²/r) · (y / r) + (Γ / 2π) · ln r,     r² = x² + y².

The velocity components are

    u(x, y) = ∂ψ/∂y
            = V_ref · [ 1 − a² (x² − y²) / r⁴ ] − (Γ / 2π) · (y / r²),

    v(x, y) = −∂ψ/∂x
            = − 2 V_ref · a² · x · y / r⁴ + (Γ / 2π) · (x / r²),

    w(x, y, z) = 0.

Properties
----------
* Steady (no time dependency).
* Divergence-free by construction: ∇·u ≡ 0 (derived from a
  streamfunction).
* The cylinder surface r = a is itself a streamline: ψ(r=a, θ) =
  (Γ / 2π) · ln a is constant in θ, so u · n = 0 there.
* Far from the cylinder (r → ∞), u → V_ref, v → 0.
* The circulation Γ introduces asymmetry (Magnus effect) but no wake
  or separation — attached, purely potential flow.
* The velocity is singular AT the origin (Γ/2π · 1/r → ∞), so we do
  not evaluate there.  For particle tracking we recommend using
  boundary handling that stops particles reaching r < a (a solid
  cylinder), or clamps the singularity numerically.

Reference parameters (reproduce PDF Figure 3)
---------------------------------------------
    V_ref = 2.0     free-stream velocity
    a     = 1.0     cylinder radius
    Γ     = 30.0    circulation
    domain: (-16, 16) × (-8, 8) × (-1, 1)      (raw PDF units)

The workstation-mounted rot_cyl_2026 case uses a DIFFERENT parameter
set (V_ref = 5, a = 0.25) because the PDF defaults produced too-slow
particle motion for meaningful mesh comparison — see
`/scratch/shared/ROM/FOM_analytic/rot_cyl_2026/rotating_cylinder_field.py`
for the reasoning.  This module keeps the PDF defaults so it remains
a faithful reference implementation.

Usage
-----
    run_tracking.py --velocity-source analytic \\
        --velocity-module .../potential_rotating_cylinder.py \\
        --domain-bbox "-16 16 -8 8 -1 1"

To override the parameters, copy this file into a case folder and
edit the constants at the top.  See
`/scratch/shared/ROM/FOM_analytic/rot_cyl_2026/rotating_cylinder_field.py`
for the workstation case that scales this field by 1/SCALE to fit
JAXTrace's spatial-search octree levels.
"""

import jax.numpy as jnp

from jaxtrace.gpu.tracking.velocity_provider import AnalyticVelocityProvider


# Reference parameters from the PDF appendix.  Override by copying this
# file and editing.
V_REF = 2.0
A     = 1.0
GAMMA = 30.0

# Small floor to keep 1/r² well-defined near the singularity when a
# particle strays inside the cylinder due to numerical error.  Chosen
# smaller than any reasonable mesh cell.
_R_MIN = 1.0e-6


def velocity_fn(pos):
    """Evaluate the potential-flow-around-cylinder field at one position.

    JAX-pure — safe to inline into the JIT'd RK4 step.

    Args
    ----
    pos : jnp.ndarray, shape (3,)
        Position (x, y, z) in metres.  z is unused (field is 2D
        extruded through z with w = 0).

    Returns
    -------
    jnp.ndarray, shape (3,)
        Velocity (u, v, 0) in m/s.
    """
    x, y = pos[0], pos[1]

    # Cast parameters to the query dtype so vmap doesn't upcast.
    Vref = jnp.float32(V_REF) if pos.dtype == jnp.float32 else jnp.float64(V_REF)
    a    = jnp.float32(A)     if pos.dtype == jnp.float32 else jnp.float64(A)
    Gam  = jnp.float32(GAMMA) if pos.dtype == jnp.float32 else jnp.float64(GAMMA)
    two_pi = jnp.float32(2.0 * jnp.pi) if pos.dtype == jnp.float32 \
             else jnp.float64(2.0 * jnp.pi)

    r2 = x * x + y * y
    # Numerical floor prevents division-by-zero / NaN at the singular
    # origin.  Particles inside the physical cylinder (r < a) should
    # already be trapped by boundary handling; this floor only guards
    # the arithmetic.
    r2_safe = jnp.maximum(r2, jnp.asarray(_R_MIN * _R_MIN, dtype=r2.dtype))
    r4_safe = r2_safe * r2_safe
    a2 = a * a

    # u = V_ref [1 - a²(x²-y²)/r⁴] - (Γ/2π) y / r²
    u = Vref * (1.0 - a2 * (x * x - y * y) / r4_safe) \
        - (Gam / two_pi) * y / r2_safe

    # v = -2 V_ref a² x y / r⁴ + (Γ/2π) x / r²
    v = -2.0 * Vref * a2 * x * y / r4_safe \
        + (Gam / two_pi) * x / r2_safe

    w = jnp.float32(0.0) if pos.dtype == jnp.float32 else jnp.float64(0.0)

    return jnp.stack([u, v, w])


def streamfunction(pos):
    """Evaluate ψ(x, y) at one position — reference for stream-plot tests."""
    x, y = pos[0], pos[1]
    r = jnp.sqrt(jnp.maximum(x * x + y * y, _R_MIN * _R_MIN))
    return V_REF * (r - A * A / r) * (y / r) + (GAMMA / (2.0 * jnp.pi)) * jnp.log(r)


def analytical_trajectory(pos0, t, rtol=1e-12, atol=1e-12, max_step=None):
    """scipy DOP853 reference trajectory — used as the ground truth
    against which mesh-interpolated RK4 trajectories are scored."""
    import numpy as np
    from scipy.integrate import solve_ivp

    pos0 = np.asarray(pos0, dtype=np.float64)
    t_scalar = np.isscalar(t)
    t_eval = None if t_scalar else np.asarray(t, dtype=np.float64)
    t_final = float(t) if t_scalar else float(t_eval[-1])

    def f(_t, p):
        x, y, _ = p
        r2 = x * x + y * y
        if r2 < _R_MIN * _R_MIN:
            r2 = _R_MIN * _R_MIN
        r4 = r2 * r2
        a2 = A * A
        u = V_REF * (1.0 - a2 * (x * x - y * y) / r4) \
            - (GAMMA / (2.0 * np.pi)) * y / r2
        v = -2.0 * V_REF * a2 * x * y / r4 \
            + (GAMMA / (2.0 * np.pi)) * x / r2
        return np.array([u, v, 0.0])

    sol = solve_ivp(
        f, (0.0, t_final), pos0,
        method='DOP853',
        rtol=rtol, atol=atol,
        max_step=max_step if max_step is not None else np.inf,
        t_eval=t_eval if not t_scalar else None,
    )
    if not sol.success:
        raise RuntimeError(f"scipy solve_ivp failed: {sol.message}")
    if t_scalar:
        return sol.y[:, -1].copy()
    return sol.y.T.copy()


DOMAIN_BBOX = (
    (-16.0, 16.0),
    (-8.0, 8.0),
    (-1.0, 1.0),
)


def build_provider(domain_bbox=None, dt=0.0, t_start=0.0):
    """Construct an AnalyticVelocityProvider for the case."""
    del dt, t_start
    if domain_bbox is None:
        domain_bbox = DOMAIN_BBOX
    return AnalyticVelocityProvider(
        velocity_fn=velocity_fn,
        is_time_dependent=False,
        level_set_fn=None,
        domain_bbox=domain_bbox,
        meta={
            "name": "potential_rotating_cylinder",
            "source": "FSW Internal Summary (Venghaus 2026), appendix §B",
            "params": {"V_ref": V_REF, "a": A, "Gamma": GAMMA},
            "closed_form_trajectory": False,
            "divergence_free": True,
            "is_2d_in_plane": "xy",
            "singularity": "1/r² at origin — evaluate only at r > a",
        },
    )
