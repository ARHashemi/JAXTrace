"""
Analytical divergence-free velocity field with localized recirculation.

Implementation of the field from "FSW Internal Summary" (H. Venghaus,
June 2026), appendix §A. Constructed from the streamfunction

    ψ(x, y) = V_ref · y + A · exp(-R²/L²) · sin(2π y / H)

with R² = (x - xc)² + y². The velocity components are

    u(x, y) = ∂ψ/∂y
            = V_ref + A · exp(-R²/L²) · [ (2π/H) cos(2π y / H)
                                          - (2y/L²) sin(2π y / H) ]

    v(x, y) = -∂ψ/∂x
            = (2 A (x - xc) / L²) · exp(-R²/L²) · sin(2π y / H)

    w(x, y, z) = 0

Properties
----------
* Steady (no time dependency).
* Divergence-free by construction: ∇·u = ∂u/∂x + ∂v/∂y ≡ 0.
* Far from the disturbed region (R → ∞), the field asymptotes to a
  uniform stream u → V_ref, v → 0.
* The Gaussian factor localises the disturbance around x = xc; the
  sinusoidal factor in y generates a stack of counter-rotating
  recirculation cells whose vertical wavelength is H.
* Defined in the (x, y) plane; trivially extruded through z with w = 0
  so the JAXTrace 3D RK4 step can use it directly.

Reference parameters (reproduce PDF Figure 2)
---------------------------------------------
V_ref = 5.0     (background flow speed)
A     = 5.0     (disturbance amplitude)
L     = 1.0     (Gaussian half-width)
H     = 1.0     (vertical wavelength of the sinusoid)
xc    = 0.0     (centre of the disturbance)

Usage
-----
    run_tracking.py --velocity-source analytic \\
        --velocity-module jaxtrace/analytic_fields/divergence_free_recirculation.py \\
        --domain-bbox "-4 4 -2 2 -0.25 0.25"

To override the parameters, copy this file and edit the constants at
the top. Or set them via an environment variable parsed in
build_provider (left as a follow-up if needed).
"""

import jax.numpy as jnp

from jaxtrace.gpu.tracking.velocity_provider import AnalyticVelocityProvider


# Reference parameters from the PDF appendix. Override by copying this
# file and editing the constants.
V_REF = 5.0
A     = 5.0
L     = 1.0
H     = 1.0
XC    = 0.0


def velocity_fn(pos):
    """Evaluate the divergence-free recirculation field at one position.

    Implemented entirely in JAX primitives so it can be inlined into
    the JIT'd RK4 step.

    Args
    ----
    pos : jnp.ndarray, shape (3,)
        Position (x, y, z) in metres. z is unused.

    Returns
    -------
    jnp.ndarray, shape (3,)
        Velocity (u, v, 0) in m/s.
    """
    x, y, _ = pos[0], pos[1], pos[2]

    # Local convenience constants. Use literal floats so the JAX trace
    # produces inline constants rather than module-level lookups.
    Vref = jnp.float32(V_REF) if pos.dtype == jnp.float32 else jnp.float64(V_REF)
    Aa   = jnp.float32(A)     if pos.dtype == jnp.float32 else jnp.float64(A)
    Ll   = jnp.float32(L)     if pos.dtype == jnp.float32 else jnp.float64(L)
    Hh   = jnp.float32(H)     if pos.dtype == jnp.float32 else jnp.float64(H)
    Xc   = jnp.float32(XC)    if pos.dtype == jnp.float32 else jnp.float64(XC)

    dx = x - Xc
    R2 = dx * dx + y * y
    L2 = Ll * Ll

    gauss = jnp.exp(-R2 / L2)
    arg = (2.0 * jnp.pi * y) / Hh
    s = jnp.sin(arg)
    c = jnp.cos(arg)

    # u = V_ref + A·exp(-R²/L²) · [ (2π/H) cos(2πy/H) - (2y/L²) sin(2πy/H) ]
    u = Vref + Aa * gauss * ((2.0 * jnp.pi / Hh) * c - (2.0 * y / L2) * s)

    # v = (2 A (x - xc) / L²) · exp(-R²/L²) · sin(2πy/H)
    v = (2.0 * Aa * dx / L2) * gauss * s

    w = jnp.float32(0.0) if pos.dtype == jnp.float32 else jnp.float64(0.0)

    return jnp.stack([u, v, w])


def analytical_trajectory(pos0, t, rtol=1e-12, atol=1e-12, max_step=None):
    """High-accuracy reference trajectory via scipy.integrate.solve_ivp.

    Not a closed-form solution — streamfunction-derived flows do not
    generally admit one — but solving with rtol=atol=1e-12 makes this
    effectively exact compared to RK4 with dt ~ 1e-3.

    Args
    ----
    pos0 : array-like, shape (3,)
        Initial position.
    t : float OR 1-D array
        Final time, or array of times to sample.
    rtol, atol : float
        scipy.integrate.solve_ivp tolerances.
    max_step : float, optional
        Cap on integrator step. None lets scipy decide.

    Returns
    -------
    np.ndarray
        Shape (3,) if t is a scalar, (len(t), 3) if t is an array.
    """
    import numpy as np
    from scipy.integrate import solve_ivp

    pos0 = np.asarray(pos0, dtype=np.float64)
    t_scalar = np.isscalar(t)
    t_eval = None if t_scalar else np.asarray(t, dtype=np.float64)
    t_final = float(t) if t_scalar else float(t_eval[-1])

    # NumPy version of velocity_fn for scipy.
    def f(_t, p):
        x, y, _ = p
        dx = x - XC
        R2 = dx * dx + y * y
        L2 = L * L
        gauss = np.exp(-R2 / L2)
        arg = (2.0 * np.pi * y) / H
        s = np.sin(arg); c = np.cos(arg)
        u = V_REF + A * gauss * ((2.0 * np.pi / H) * c - (2.0 * y / L2) * s)
        v = (2.0 * A * dx / L2) * gauss * s
        return np.array([u, v, 0.0])

    sol = solve_ivp(
        f, (0.0, t_final), pos0,
        method='DOP853',  # 8th-order RK; gold standard for this
        rtol=rtol, atol=atol,
        max_step=max_step if max_step is not None else np.inf,
        t_eval=t_eval if not t_scalar else None,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"scipy solve_ivp failed: {sol.message}")
    if t_scalar:
        return sol.y[:, -1].copy()
    return sol.y.T.copy()  # shape (len(t), 3)


def build_provider(domain_bbox=None, dt=0.0, t_start=0.0):
    """Construct an AnalyticVelocityProvider for the PDF §A field.

    Args
    ----
    domain_bbox : ((xmin,xmax),(ymin,ymax),(zmin,zmax)), optional
        Defaults to the bbox shown in PDF Figure 2 with a thin z slab.
    dt, t_start : float
        Accepted for interface compatibility; unused (steady field).
    """
    del dt, t_start
    if domain_bbox is None:
        # Match PDF Figure 2 in (x, y); thin slab in z.
        domain_bbox = ((-4.0, 4.0), (-2.0, 2.0), (-0.25, 0.25))
    return AnalyticVelocityProvider(
        velocity_fn=velocity_fn,
        is_time_dependent=False,
        level_set_fn=None,
        domain_bbox=domain_bbox,
        meta={
            "name": "divergence_free_recirculation",
            "source": "FSW Internal Summary (Venghaus 2026), appendix §A",
            "params": {
                "V_ref": V_REF, "A": A, "L": L, "H": H, "xc": XC,
            },
            "closed_form_trajectory": False,
            "divergence_free": True,
            "is_2d_in_plane": "xy",  # w = 0
        },
    )
